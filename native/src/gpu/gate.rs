use super::{MatrixView, Metal, RouteOutput, read_weight, view::FloatType};
use crate::checkpoint::Checkpoint;
use anyhow::{Context, Result, ensure};
use metal::{Buffer, MTLResourceOptions as Storage, MTLSize};

pub struct FlashGate<'a> {
    gpu: &'a Metal,
    weight: Buffer,
}

#[derive(serde::Serialize)]
pub struct FlashGateTrace {
    logits: Vec<u32>,
    probabilities: Vec<u32>,
    indices: Vec<u32>,
    scores: Vec<u32>,
}

impl<'a> FlashGate<'a> {
    pub fn load(gpu: &'a Metal, checkpoint: &Checkpoint, prefix: &str) -> Result<Self> {
        ensure!(
            checkpoint.model_type == "qwen4_exp",
            "requires a Flash reference checkpoint"
        );
        let weight = checkpoint
            .tensors
            .get(&format!("{prefix}.weight"))
            .context("missing Flash gate weight")?;
        ensure!(
            weight.dtype == "F16"
                && weight.shape == [512, 2560]
                && !checkpoint.tensors.contains_key(&format!("{prefix}.bias")),
            "Flash gate requires unbiased F16 [512,2560]"
        );
        Ok(Self {
            gpu,
            weight: read_weight(&gpu.device, weight, 512 * 2560 * 2)?,
        })
    }

    pub fn submit(&self, x: &MatrixView) -> Result<RouteOutput> {
        Ok(self.dispatch(x, Storage::StorageModePrivate)?.0)
    }

    pub fn trace(&self, x: &MatrixView) -> Result<FlashGateTrace> {
        let (routes, logits, probabilities) = self.dispatch(x, Storage::StorageModeShared)?;
        let (indices, scores) = routes.read()?;
        Ok(FlashGateTrace {
            indices,
            scores,
            logits: logits.read_output()?,
            probabilities: probabilities.read_output()?,
        })
    }

    fn dispatch(
        &self,
        x: &MatrixView,
        storage: Storage,
    ) -> Result<(RouteOutput, MatrixView, MatrixView)> {
        let gpu = self.gpu;
        let [rows, width] = x.layout.shape;
        ensure!(
            crate::contracts::flash_gate_rows(rows, width),
            "invalid Flash gate input shape or index overflow"
        );
        ensure!(
            x.buffer.device().registry_id() == gpu.device.registry_id(),
            "gate input belongs to another Metal device"
        );
        let precision = usize::from(x.dtype == FloatType::F32);
        let dot = &gpu.gate_dot[precision];
        let softmax = &gpu.gate_softmax[precision];
        ensure!(
            dot.thread_execution_width() == 32
                && dot.max_total_threads_per_threadgroup() >= 32
                && softmax.thread_execution_width() == 32
                && softmax.max_total_threads_per_threadgroup() >= 128,
            "Flash gate requires SIMD32"
        );
        let logits = MatrixView::output(gpu, x.dtype, rows, 512, storage)?;
        let probabilities = MatrixView::output(gpu, x.dtype, rows, 512, storage)?;
        objc::rc::autoreleasepool(|| {
            let cmd = gpu.queue.new_command_buffer();
            Metal::encode(
                cmd,
                dot,
                &[&x.buffer, &self.weight, &logits.buffer],
                [x.layout.offset, x.layout.strides[0], x.layout.strides[1]],
                MTLSize::new(512, u64::from(rows), 1),
                32,
            );
            let groups = MTLSize::new(u64::from(rows), 1, 1);
            Metal::encode(
                cmd,
                softmax,
                &[&logits.buffer, &probabilities.buffer],
                [rows],
                groups,
                128,
            );
            // The reference reduces a [1,10] matrix serially in its own dtype.
            let routes = gpu.encode_route(cmd, &probabilities, 10, true)?;
            cmd.commit();
            Ok((routes, logits, probabilities))
        })
    }
}
