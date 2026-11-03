use super::{MatrixView, Metal, buffer, view::FloatType};
use crate::{contracts, tensor_layout::MatrixLayout};
use anyhow::{Context, Result, ensure};
use metal::{
    Buffer, CommandBuffer, MTLCommandBufferStatus, MTLResourceOptions as Storage, MTLSize,
};

/// The command retains its outputs until completion. Scores can then be reused on any queue.
pub struct RouteOutput {
    command: CommandBuffer,
    indices: Buffer,
    scores: MatrixView,
    count: usize,
}

impl RouteOutput {
    /// Publish only after success; a different queue may immediately consume this view.
    pub fn completed_scores(&self) -> Result<&MatrixView> {
        self.command.wait_until_completed();
        ensure!(
            self.command.status() == MTLCommandBufferStatus::Completed,
            "Metal router command failed ({:?})",
            self.command.status()
        );
        Ok(&self.scores)
    }

    /// SSD planning needs the expert IDs on the host while scores remain on the GPU.
    pub fn indices(&self) -> Result<Vec<u32>> {
        self.completed_scores()?;
        Ok(self.words(&self.indices))
    }

    fn words(&self, buf: &Buffer) -> Vec<u32> {
        // Both four-byte outputs are shared, aligned and completely initialized on success.
        unsafe { std::slice::from_raw_parts(buf.contents().cast::<u32>(), self.count).to_vec() }
    }

    /// Return indices and raw FP16/FP32 words (zero-extended for FP16).
    pub fn read(self) -> Result<(Vec<u32>, Vec<u32>)> {
        let indices = self.indices()?;
        let scores = match self.scores.dtype {
            FloatType::F32 => self.words(&self.scores.buffer),
            FloatType::F16 => Metal::read(&self.scores.buffer, self.count)
                .into_iter()
                .map(u32::from)
                .collect(),
        };
        Ok((indices, scores))
    }
}

impl Metal {
    pub fn route(&self, values: &MatrixView, k: u32, normalize: bool) -> Result<RouteOutput> {
        let [rows, experts] = values.layout.shape;
        let count = contracts::router_output_len(rows, experts, k)
            .context("invalid router geometry or index overflow")?;
        ensure!(
            values.buffer.device().registry_id() == self.device.registry_id(),
            "router input belongs to another device"
        );
        let (pipeline, size) = match values.dtype {
            FloatType::F16 => (&self.route_half, 2),
            FloatType::F32 => (&self.route_float, 4),
        };
        ensure!(
            u64::from(experts) <= pipeline.max_total_threads_per_threadgroup(),
            "router exceeds supported threadgroup width"
        );
        let indices = buffer(
            &self.device,
            u64::from(count) * 4,
            Storage::StorageModeShared,
        )?;
        let scores = MatrixView {
            buffer: buffer(
                &self.device,
                u64::from(count) * size,
                Storage::StorageModeShared,
            )?,
            layout: MatrixLayout {
                shape: [rows, k],
                offset: 0,
                strides: [i64::from(k), 1],
            },
            dtype: values.dtype,
        };
        objc::rc::autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            Self::encode(
                cmd,
                pipeline,
                &[&values.buffer, &indices, &scores.buffer],
                [
                    i64::from(experts),
                    i64::from(k),
                    i64::from(normalize),
                    values.layout.offset,
                    values.layout.strides[0],
                    values.layout.strides[1],
                ],
                MTLSize::new(u64::from(rows), 1, 1),
                experts as usize,
            );
            cmd.commit();
            Ok(RouteOutput {
                command: cmd.to_owned(),
                indices,
                scores,
                count: count as usize,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "physical Metal: retain router outputs and consume on a different queue"]
    fn publishes_complete_routes_after_producer_and_input_drop() -> Result<()> {
        let flight = {
            let gpu = Metal::new()?;
            let storage =
                gpu.f32_storage(&[f32::NEG_INFINITY, f32::NAN, -0.0, 0.0, f32::INFINITY])?;
            let view = storage.matrix(MatrixLayout {
                shape: [1, 5],
                strides: [0, 1],
                offset: 0,
            })?;
            gpu.route(&view, 4, false)?
        };
        let another = Metal::new()?;
        ensure!(flight.indices()? == [2, 3, 4, 1]);
        let result = another.route(flight.completed_scores()?, 2, false)?;
        drop(flight);
        let (ids, bits) = result.read()?;
        ensure!(
            ids == [2, 3] && bits[0] == f32::INFINITY.to_bits() && f32::from_bits(bits[1]).is_nan()
        );
        Ok(())
    }
}
