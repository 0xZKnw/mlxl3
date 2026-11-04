use super::{MatrixView, Metal, RouteOutput, read_weight, view::FloatType};
use crate::checkpoint::Checkpoint;
use anyhow::{Context, Result, ensure};
use metal::{Buffer, MTLResourceOptions as Storage, MTLSize};

pub struct FlashGate<'a> {
    gpu: &'a Metal,
    weight: Buffer,
}

// Keep both producer intermediates alive through the enclosing command's wait.
pub(super) struct GateBuffers {
    pub(super) routes: RouteOutput,
    logits: MatrixView,
    probabilities: MatrixView,
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
        Ok(self.dispatch(x, Storage::StorageModePrivate)?.routes)
    }

    pub(super) fn predict(&self, x: &MatrixView, half: bool) -> Result<RouteOutput> {
        ensure!(
            x.buffer.device().registry_id() == self.gpu.device.registry_id(),
            "predictor input belongs to another Metal device"
        );
        let converted;
        let x = if half && x.dtype == FloatType::F32 {
            // The mixed MLP row is a completed contiguous shared output.
            let bits = x
                .read_output()?
                .into_iter()
                .map(|b| half::f16::from_f32(f32::from_bits(b)).to_bits())
                .collect::<Vec<_>>();
            converted = self.gpu.f16_storage(&bits)?.matrix(x.layout)?;
            &converted
        } else {
            x
        };
        let buffers = self.dispatch(x, Storage::StorageModePrivate)?;
        self.gpu.route(&buffers.logits, 16, false)
    }

    pub fn trace(&self, x: &MatrixView) -> Result<FlashGateTrace> {
        let buffers = self.dispatch(x, Storage::StorageModeShared)?;
        let (indices, scores) = buffers.routes.read()?;
        Ok(FlashGateTrace {
            indices,
            scores,
            logits: buffers.logits.read_output()?,
            probabilities: buffers.probabilities.read_output()?,
        })
    }

    fn dispatch(&self, x: &MatrixView, storage: Storage) -> Result<GateBuffers> {
        objc::rc::autoreleasepool(|| {
            let cmd = self.gpu.queue.new_command_buffer();
            let buffers = self.encode(self.gpu, cmd, x, storage)?;
            cmd.commit();
            #[cfg(test)]
            self.gpu.gate_commands.set(self.gpu.gate_commands.get() + 1);
            Ok(buffers)
        })
    }

    pub(super) fn encode(
        &self,
        owner: &Metal,
        cmd: &metal::CommandBufferRef,
        x: &MatrixView,
        storage: Storage,
    ) -> Result<GateBuffers> {
        ensure!(
            std::ptr::eq(self.gpu, owner) && owner.owns_command_queue(cmd),
            "gate chain requires the same Metal instance and queue"
        );
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
        Ok(GateBuffers {
            routes,
            logits,
            probabilities,
        })
    }
}

impl FlashGate<'_> {
    pub(super) fn record_memory(&self, backings: &mut crate::gpu::memory::Backings) -> Result<()> {
        backings.record("fixed_weights", &self.weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_layout::MatrixLayout;

    #[test]
    #[ignore = "physical gate analytic F32/F16 oracle, ties, strided rows and foreign queues"]
    fn encoded_gate_matches_independent_uniform_oracle_and_owner_errors() -> Result<()> {
        let gpu = Metal::new()?;
        let mut weight = vec![0u16; 512 * 2560];
        for row in weight.as_chunks_mut::<2560>().0 {
            row[0] = half::f16::from_f32(0.5).to_bits();
        }
        let gate = FlashGate {
            gpu: &gpu,
            weight: super::super::upload(&gpu.device, &weight)?,
        };
        for half in [false, true] {
            for rows in [1, 3] {
                let mut values = vec![0.0f32; rows * 5121];
                for row in 0..rows {
                    values[row * 5121 + 1] = -2.0;
                }
                let storage = if half {
                    gpu.f16_storage(
                        &values
                            .iter()
                            .map(|&v| half::f16::from_f32(v).to_bits())
                            .collect::<Vec<_>>(),
                    )?
                } else {
                    gpu.f32_storage(&values)?
                };
                let x = storage.matrix(MatrixLayout {
                    shape: [rows as u32, 2560],
                    strides: [5121, 2],
                    offset: 1,
                })?;
                let buffers =
                    gpu.command(|cmd| gate.encode(&gpu, cmd, &x, Storage::StorageModeShared))?;
                let (ids, scores) = buffers.routes.read()?;
                let bits = |v: f32| {
                    if half {
                        u32::from(half::f16::from_f32(v).to_bits())
                    } else {
                        v.to_bits()
                    }
                };
                ensure!(buffers.logits.read_output()? == vec![bits(-1.0); rows * 512]);
                ensure!(
                    buffers.probabilities.read_output()? == vec![bits(1.0 / 512.0); rows * 512]
                );
                ensure!(ids == (0..rows).flat_map(|_| 502..512).collect::<Vec<_>>());
                ensure!(scores == vec![bits(0.1); rows * 10]);
                let raw = gate.trace(&x)?;
                ensure!(raw.indices == ids && raw.scores == scores);
                let (predicted, scores) = gate.predict(&x, false)?.read()?;
                ensure!(predicted == (0..rows).flat_map(|_| 496..512).collect::<Vec<_>>());
                ensure!(scores == vec![bits(-1.0); rows * 16]);
                let contiguous = (0..rows)
                    .flat_map(|_| std::iter::once(-2.0).chain(std::iter::repeat_n(0.0, 2559)))
                    .collect::<Vec<_>>();
                let contiguous = gpu.f32_storage(&contiguous)?.matrix(MatrixLayout {
                    shape: [rows as u32, 2560],
                    strides: [2560, 1],
                    offset: 0,
                })?;
                let (predicted, scores) = gate.predict(&contiguous, true)?.read()?;
                ensure!(predicted == (0..rows).flat_map(|_| 496..512).collect::<Vec<_>>());
                ensure!(scores == vec![u32::from(half::f16::NEG_ONE.to_bits()); rows * 16]);
            }
        }
        let x = gpu.f32_storage(&vec![0.0; 2560])?.matrix(MatrixLayout {
            shape: [1, 2560],
            strides: [2560, 1],
            offset: 0,
        })?;
        let other = Metal::new()?;
        ensure!(
            gpu.command(|cmd| gate.encode(&other, cmd, &x, Storage::StorageModeShared))
                .is_err()
        );
        let other_queue = gpu.device.new_command_queue();
        let cmd = other_queue.new_command_buffer();
        ensure!(
            gate.encode(&gpu, cmd, &x, Storage::StorageModeShared)
                .is_err()
        );
        let bad = gpu.f32_storage(&[0.0])?.matrix(MatrixLayout {
            shape: [1, 1],
            strides: [1, 1],
            offset: 0,
        })?;
        ensure!(
            gpu.command(|cmd| gate.encode(&gpu, cmd, &bad, Storage::StorageModeShared))
                .is_err()
        );
        ensure!(gate.trace(&x)?.indices == (502..512).collect::<Vec<_>>());
        Ok(())
    }
}
