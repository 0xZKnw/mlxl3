//! Shared-expert rows only: immutable weights, no router/cache/state mutation.
use super::{FlashMoeRow, FloatType, MatrixView, Storage, buffer};
use anyhow::{Context, Result, ensure};
use metal::{Buffer, CommandBufferRef, MTLCommandBufferStatus};

#[cfg(test)]
thread_local! {
    static BATCHED_SHARED_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

impl FlashMoeRow<'_> {
    /// Encode one shared expert for a block. The caller must complete `cmd`
    /// successfully before consuming its outputs or publishing any model state.
    /// Output rows own distinct offset-zero buffers for legacy scalar mix kernels.
    pub(super) fn encode_shared_block(
        &self,
        cmd: &CommandBufferRef,
        rows: &[MatrixView],
        storage: Storage,
    ) -> Result<Vec<Buffer>> {
        let gpu = self.gpu;
        ensure!(
            (1..=4).contains(&rows.len()),
            "shared block must contain 1 to 4 rows"
        );
        ensure!(
            gpu.owns_command_queue(cmd) && cmd.status() == MTLCommandBufferStatus::NotEnqueued,
            "shared block requires an unsubmitted command on its owner queue"
        );
        let dtype = rows[0].dtype;
        let element_bytes = if dtype == FloatType::F16 { 2 } else { 4 };
        let row_bytes = 2560 * element_bytes;
        let mut offsets = Vec::with_capacity(rows.len());
        // Validate every source before recording any command. HC mixed rows
        // are contiguous; never silently reinterpret a strided view as packed.
        for row in rows {
            ensure!(
                row.shape() == [1, 2560]
                    && row.dtype == dtype
                    && row.layout.strides[1] == 1
                    && row.buffer.device().registry_id() == gpu.device.registry_id(),
                "invalid shared block row shape, dtype, stride or device"
            );
            let offset = u64::try_from(row.layout.offset)
                .context("negative shared block offset")?
                .checked_mul(element_bytes)
                .context("shared block byte offset overflow")?;
            ensure!(
                offset
                    .checked_add(row_bytes)
                    .is_some_and(|end| end <= row.buffer.length()),
                "shared block row exceeds its backing"
            );
            offsets.push(offset);
        }
        // Cohort scheduling alone must not opt into numerical batching.
        // Preserve the exact existing scalar invocation unless the independent
        // M5 shared-QMV switch was explicitly enabled during Metal creation.
        if !crate::flash_qmv::shared_verify_enabled(gpu.flash_verify_qmv.is_some())
            || rows.len() == 1
        {
            return rows
                .iter()
                .map(|row| self.shared.encode_view(cmd, row, storage))
                .collect();
        }
        #[cfg(test)]
        crate::flash_qmv::note_shared_path(true);
        let packed = MatrixView::output(
            gpu,
            dtype,
            rows.len() as u32,
            2560,
            Storage::StorageModePrivate,
        )?;
        let blit = cmd.new_blit_command_encoder();
        for (index, (row, offset)) in rows.iter().zip(offsets).enumerate() {
            blit.copy_from_buffer(
                &row.buffer,
                offset,
                &packed.buffer,
                index as u64 * row_bytes,
                row_bytes,
            );
        }
        blit.end_encoding();
        let shared = self
            .shared
            .encode_view(cmd, &packed, Storage::StorageModePrivate)?;
        #[cfg(test)]
        BATCHED_SHARED_CALLS.with(|count| count.set(count.get() + 1));
        ensure!(
            shared.length() == rows.len() as u64 * 5120,
            "invalid shared block output extent"
        );
        let outputs = (0..rows.len())
            .map(|_| buffer(&gpu.device, 5120, storage))
            .collect::<Result<Vec<_>>>()?;
        let blit = cmd.new_blit_command_encoder();
        for (index, output) in outputs.iter().enumerate() {
            blit.copy_from_buffer(&shared, index as u64 * 5120, output, 0, 5120);
        }
        blit.end_encoding();
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{checkpoint, gpu::Metal, tensor_layout::MatrixLayout};

    #[test]
    #[ignore = "physical Flash checkpoint; run shared QMV flag 0 and 1; exact shared-expert oracle"]
    fn shared_expert_block_matches_scalar_with_offset_rows() -> Result<()> {
        let checkpoint =
            checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        let pipelines_loaded = gpu.flash_verify_qmv.is_some();
        let moe = FlashMoeRow::load(&gpu, &checkpoint, "model.language_model.layers.0.mlp", 10)?;
        for fp32 in [false, true] {
            let mut rows = Vec::new();
            for row in 0..4 {
                let values = (0..2567)
                    .map(|i| ((i * 19 + row * 37) % 251) as f32 / 256. - 0.5)
                    .collect::<Vec<_>>();
                let storage = if fp32 {
                    gpu.f32_storage(&values)?
                } else {
                    gpu.f16_storage(
                        &values
                            .iter()
                            .map(|v| half::f16::from_f32(*v).to_bits())
                            .collect::<Vec<_>>(),
                    )?
                };
                rows.push(storage.matrix(MatrixLayout {
                    shape: [1, 2560],
                    strides: [0, 1],
                    offset: 7,
                })?);
            }
            let expected = rows
                .iter()
                .map(|row| moe.shared.submit_fp16_view(row)?.read())
                .collect::<Result<Vec<_>>>()?;
            for enabled in [false, true] {
                let _override = crate::flash_qmv::SharedVerifyOverride::new(enabled);
                for count in 1..=4 {
                    let before = BATCHED_SHARED_CALLS.with(std::cell::Cell::get);
                    let paths_before = crate::flash_qmv::shared_path_counts();
                    let actual = gpu.command(|cmd| {
                        moe.encode_shared_block(cmd, &rows[..count], Storage::StorageModeShared)
                    })?;
                    ensure!(actual.len() == count);
                    let after = BATCHED_SHARED_CALLS.with(std::cell::Cell::get);
                    ensure!(
                        after - before == usize::from(pipelines_loaded && enabled && count > 1)
                    );
                    let paths_after = crate::flash_qmv::shared_path_counts();
                    let expected_packing = pipelines_loaded && enabled && count > 1;
                    ensure!(paths_after.0 - paths_before.0 == u64::from(expected_packing));
                    ensure!((paths_after.1 > paths_before.1) == expected_packing);
                    for (index, output) in actual.iter().enumerate() {
                        ensure!(output.length() == 5120);
                        ensure!(
                            Metal::read(output, 2560) == expected[index],
                            "shared block B{count} row{index} fp32={fp32}"
                        );
                        for other in &actual[..index] {
                            ensure!(!std::ptr::eq::<metal::BufferRef>(
                                output.as_ref(),
                                other.as_ref()
                            ));
                        }
                    }
                }
            }
            let cmd = gpu.queue.new_command_buffer();
            ensure!(
                moe.encode_shared_block(cmd, &[], Storage::StorageModePrivate)
                    .is_err()
            );
            let too_many = vec![rows[0].clone(); 5];
            ensure!(
                moe.encode_shared_block(cmd, &too_many, Storage::StorageModePrivate)
                    .is_err()
            );
            for (offset, stride, width) in [
                (-1, 1, 2560),
                (i64::MAX, 1, 2560),
                (0, 2, 2560),
                (0, 1, 2559),
            ] {
                let mut bad = rows[0].clone();
                bad.layout.offset = offset;
                bad.layout.strides[1] = stride;
                bad.layout.shape[1] = width;
                ensure!(
                    moe.encode_shared_block(
                        cmd,
                        &[rows[0].clone(), bad],
                        Storage::StorageModePrivate
                    )
                    .is_err()
                );
            }
            let mut mixed = rows[0].clone();
            mixed.dtype = if fp32 { FloatType::F16 } else { FloatType::F32 };
            ensure!(
                moe.encode_shared_block(
                    cmd,
                    &[rows[0].clone(), mixed],
                    Storage::StorageModePrivate
                )
                .is_err()
            );
            ensure!(cmd.status() == MTLCommandBufferStatus::NotEnqueued);
        }
        Ok(())
    }
}
