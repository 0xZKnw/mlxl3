use super::*;
use crate::tensor_layout::MatrixLayout;

fn backing_bytes(projection: &FlashProjection<'_>) -> Result<u64> {
    let mut backings = crate::gpu::memory::Backings::default();
    projection.record_memory(&mut backings)?;
    projection.record_memory(&mut backings)?;
    backings.total()
}

fn ids(projection: &FlashProjection<'_>) -> Result<[usize; 2]> {
    let slot = projection.scratch.borrow();
    let pair = slot.as_ref().context("expected resident pair")?;
    Ok([
        pair.prepared.as_ptr() as usize,
        pair.partial.as_ptr() as usize,
    ])
}

#[test]
#[ignore = "physical completed resident scratch vs independent fresh allocations, poison/retention/pending/abandon; exclusive GPU"]
fn physical_ar30_resident_scratch_preserves_retained_outputs_and_pending_owners() -> Result<()> {
    let gpu = Metal::new()?;
    ensure!(
        gpu.resident_scratch.enabled.get(),
        "enable MLXL3_NATIVE_RESIDENT_SCRATCH=1"
    );
    let checkpoint =
        crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let mut projection = FlashProjection::load(
        &gpu,
        &checkpoint,
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
    )?;
    ensure!(projection.dimensions() == [2560, 10240]);
    let x = gpu
        .f32_storage(
            &(0..2560)
                .map(|i| ((i * 17 % 127) as f32 - 63.) / 79.)
                .collect::<Vec<_>>(),
        )?
        .matrix(MatrixLayout {
            shape: [1, 2560],
            strides: [2560, 1],
            offset: 0,
        })?;
    let start = gpu.resident_scratch.reuses.get();
    let base = backing_bytes(&projection)?;
    gpu.resident_scratch.enabled.set(false);
    let expected = gpu
        .command(|cmd| projection.encode_row(cmd, &x, Storage::StorageModePrivate))?
        .export_bytes(&gpu)?;
    ensure!(
        expected.len() == 20480
            && expected
                .as_chunks::<2>()
                .0
                .iter()
                .all(|&w| f16::from_le_bytes(w).is_finite())
            && expected
                .as_chunks::<2>()
                .0
                .iter()
                .any(|&w| u16::from_le_bytes(w) & 0x7fff != 0)
    );
    ensure!(projection.scratch.borrow().is_none() && gpu.resident_scratch.charged() == 0);
    gpu.resident_scratch.enabled.set(true);
    let held = gpu.command(|cmd| projection.encode_row(cmd, &x, Storage::StorageModePrivate))?;
    ensure!(held.export_bytes(&gpu)? == expected && gpu.resident_scratch.reuses.get() == start);
    let pair_ids = ids(&projection)?;
    let charge = gpu.resident_scratch.charged();
    ensure!(charge == 46080 && backing_bytes(&projection)? == base + charge);
    // Deliberately destroy both temporaries: the unchanged prepare/QMV must fully overwrite them.
    gpu.command(|cmd| {
        let slot = projection.scratch.borrow();
        let pair = slot.as_ref().context("expected pair for poison")?;
        let blit = cmd.new_blit_command_encoder();
        for buffer in [&pair.prepared, &pair.partial] {
            blit.fill_buffer(buffer, metal::NSRange::new(0, buffer.length()), 255);
        }
        blit.end_encoding();
        Ok(())
    })?;
    let reused = gpu.command(|cmd| projection.encode_row(cmd, &x, Storage::StorageModePrivate))?;
    ensure!(ids(&projection)? == pair_ids && gpu.resident_scratch.reuses.get() - start == 1);
    ensure!(
        reused.export_bytes(&gpu)? == expected
            && held.export_bytes(&gpu)? == expected
            && reused.buffer.as_ptr() != held.buffer.as_ptr()
    );
    let pending = gpu.queue.new_command_buffer().to_owned();
    let pending_first = projection.encode_row(&pending, &x, Storage::StorageModePrivate)?;
    ensure!(gpu.resident_scratch.reuses.get() - start == 2);
    let pending_second = projection.encode_row(&pending, &x, Storage::StorageModePrivate)?;
    projection.release_idle_scratch()?;
    ensure!(
        ids(&projection)? == pair_ids
            && gpu.resident_scratch.charged() == charge
            && pending.status() == MTLCommandBufferStatus::NotEnqueued
    );
    let while_pending =
        gpu.command(|cmd| projection.encode_row(cmd, &x, Storage::StorageModePrivate))?;
    ensure!(
        while_pending.export_bytes(&gpu)? == expected
            && gpu.resident_scratch.reuses.get() - start == 2
    );
    pending.commit();
    pending.wait_until_completed();
    ensure!(pending.status() == MTLCommandBufferStatus::Completed);
    for value in [&pending_first, &pending_second] {
        ensure!(
            value.export_bytes(&gpu)? == expected && value.buffer.as_ptr() != held.buffer.as_ptr()
        );
    }
    ensure!(pending_first.buffer.as_ptr() != pending_second.buffer.as_ptr());
    let shared = gpu.command(|cmd| projection.encode_row(cmd, &x, Storage::StorageModeShared))?;
    ensure!(
        shared.export_bytes(&gpu)? == expected && gpu.resident_scratch.reuses.get() - start == 2
    );
    let invalid = gpu.f32_storage(&[1.])?.matrix(MatrixLayout {
        shape: [1, 1],
        strides: [1, 1],
        offset: 0,
    })?;
    let rejected = gpu.queue.new_command_buffer().to_owned();
    ensure!(
        projection
            .encode_row(&rejected, &invalid, Storage::StorageModePrivate)
            .is_err()
            && rejected.status() == MTLCommandBufferStatus::NotEnqueued
            && ids(&projection)? == pair_ids
    );
    // OFF retains truthful accounting for already-published storage.
    gpu.resident_scratch.enabled.set(false);
    ensure!(
        gpu.resident_scratch.charged() == charge && backing_bytes(&projection)? == base + charge
    );
    gpu.resident_scratch.enabled.set(true);
    projection.release_idle_scratch()?;
    ensure!(
        projection.scratch.borrow().is_none()
            && gpu.resident_scratch.charged() == 0
            && backing_bytes(&projection)? == base
            && held.export_bytes(&gpu)? == expected
    );
    // A real host failure after prepare encoding must leave the published owner pending.
    let abandoned = gpu.queue.new_command_buffer().to_owned();
    let groups = projection.simdgroups;
    projection.simdgroups = 0;
    let failed = projection.encode_row(&abandoned, &x, Storage::StorageModePrivate);
    projection.simdgroups = groups;
    ensure!(failed.is_err() && abandoned.status() == MTLCommandBufferStatus::NotEnqueued);
    let abandoned_ids = ids(&projection)?;
    projection.release_idle_scratch()?;
    ensure!(ids(&projection)? == abandoned_ids && gpu.resident_scratch.charged() == charge);
    let after_error =
        gpu.command(|cmd| projection.encode_row(cmd, &x, Storage::StorageModePrivate))?;
    ensure!(
        after_error.export_bytes(&gpu)? == expected
            && held.export_bytes(&gpu)? == expected
            && gpu.resident_scratch.reuses.get() - start == 2
            && ids(&projection)? == abandoned_ids
    );
    drop(projection);
    drop(abandoned); // Never submit the partially encoded failed command.
    ensure!(gpu.resident_scratch.valid() && gpu.resident_scratch.charged() == charge);
    eprintln!(
        "AR30_RESIDENT_SCRATCH {{\"parity\":true,\"cases\":8,\"finite_half_words\":81920,\"completed_reuses\":2,\"cached_pair_bytes\":46080,\"retained_outputs\":true,\"poisoned_pair\":true,\"same_command_fallback\":true,\"pending_fallback\":true,\"failed_owner_retained\":true,\"idle_refund\":true,\"abandoned_charge_retained\":true}}"
    );
    Ok(())
}
