use super::{MatrixView, Metal, buffer, view::FloatType};
use crate::contracts;
use anyhow::{Context, Result, ensure};
use metal::{
    Buffer, CommandBuffer, MTLCommandBufferStatus, MTLResourceOptions as Storage, MTLSize,
};

/// The command retains its outputs until completion. Scores can then be reused on any queue.
pub struct RouteOutput {
    command: CommandBuffer,
    indices: Buffer,
    pub(super) scores: MatrixView,
    count: usize,
    #[cfg(test)]
    waits: std::rc::Rc<std::cell::Cell<usize>>,
}

impl RouteOutput {
    /// Publish only after success; a different queue may immediately consume this view.
    pub fn completed_scores(&self) -> Result<&MatrixView> {
        #[cfg(test)]
        self.waits.set(self.waits.get() + 1);
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
        #[cfg(test)]
        let _copy = crate::flash_profile::span("native.route_readback_copy");
        #[cfg(test)]
        if let Some(span) = &_copy {
            span.data(serde_json::json!({"bytes":self.count*4}));
        }
        // The shared index buffer has exactly count aligned u32s after completion.
        Ok(unsafe {
            std::slice::from_raw_parts(self.indices.contents().cast::<u32>(), self.count).to_vec()
        })
    }

    /// Return indices and raw FP16/FP32 words (zero-extended for FP16).
    pub fn read(self) -> Result<(Vec<u32>, Vec<u32>)> {
        let indices = self.indices()?;
        let scores = self.scores.read_output()?;
        Ok((indices, scores))
    }
}

impl Metal {
    pub fn route(&self, values: &MatrixView, k: u32, normalize: bool) -> Result<RouteOutput> {
        objc::rc::autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            let result = self.encode_route(cmd, values, k, normalize)?;
            cmd.commit();
            Ok(result)
        })
    }
    pub(super) fn encode_route(
        &self,
        cmd: &metal::CommandBufferRef,
        values: &MatrixView,
        k: u32,
        normalize: bool,
    ) -> Result<RouteOutput> {
        let [rows, experts] = values.layout.shape;
        let count = contracts::router_output_len(rows, experts, k)
            .context("invalid router geometry or index overflow")?;
        ensure!(
            values.buffer.device().registry_id() == self.device.registry_id(),
            "router input belongs to another device"
        );
        let pipeline = match values.dtype {
            FloatType::F16 => &self.route_half,
            FloatType::F32 => &self.route_float,
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
        let scores = MatrixView::output(self, values.dtype, rows, k, Storage::StorageModeShared)?;
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
        Ok(RouteOutput {
            command: cmd.to_owned(),
            indices,
            scores,
            count: count as usize,
            #[cfg(test)]
            waits: self.route_waits.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_layout::MatrixLayout;

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
