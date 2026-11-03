use super::{MatrixView, Metal, read_weight, upload, view::FloatType};
use crate::checkpoint::{Checkpoint, TensorInfo};
use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use metal::{
    Buffer, CommandBufferRef, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions as Storage, MTLSize,
};

pub(super) struct Pipelines {
    norm: ComputePipelineState,
    narrow: ComputePipelineState,
    wide: ComputePipelineState,
    activate: ComputePipelineState,
    mix: ComputePipelineState,
    write: ComputePipelineState,
}
impl Pipelines {
    pub(super) fn load(
        pipeline: &impl Fn(&str) -> Result<ComputePipelineState>,
        suffix: &str,
    ) -> Result<Self> {
        let stage = |name: &str| pipeline(&format!("hc_{name}_{suffix}"));
        Ok(Self {
            norm: stage("norm")?,
            narrow: stage("narrow")?,
            wide: stage("wide")?,
            activate: stage("activate")?,
            mix: stage("mix")?,
            write: stage("write")?,
        })
    }
}

pub struct FlashHc<'a> {
    gpu: &'a Metal,
    gain: Buffer,
    down: Buffer,
    up: Buffer,
    inject: Option<Buffer>,
    eps: f32,
}

pub struct FlashHcRead {
    pub mixed: MatrixView,
    pub gate: Option<MatrixView>,
    pub trace: Option<FlashHcTrace>,
}

#[derive(serde::Serialize)]
pub struct FlashHcTrace {
    normalized: Vec<u32>,
    down: Vec<u32>,
    active: Vec<u32>,
    up: Vec<u32>,
}

fn float_values(info: &TensorInfo) -> Result<Vec<f32>> {
    let bytes = info.read_bytes()?;
    Ok(match info.dtype.as_str() {
        "F16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| f16::from_le_bytes(v).to_f32())
            .collect(),
        "BF16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| bf16::from_le_bytes(v).to_f32())
            .collect(),
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&v| f32::from_le_bytes(v))
            .collect(),
        _ => bail!("expected floating HC weights"),
    })
}
fn groups(n: u32) -> MTLSize {
    MTLSize::new(u64::from(n), 1, 1)
}

impl<'a> FlashHc<'a> {
    pub fn load(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        eps: f32,
        combine: bool,
    ) -> Result<Self> {
        ensure!(
            c.model_type == "qwen4_exp" && eps.is_finite() && eps > 0.,
            "invalid Flash HC configuration"
        );
        let weight = |part: &str, shape: &[usize]| -> Result<&TensorInfo> {
            let name = format!("{prefix}.{part}");
            let w = c
                .tensors
                .get(&format!("{name}.weight"))
                .with_context(|| format!("missing {name}.weight"))?;
            ensure!(
                w.shape == shape && !c.tensors.contains_key(&format!("{name}.bias")),
                "invalid Flash HC weight {name}"
            );
            Ok(w)
        };
        let dense = |part: &str, rows: usize, cols: usize| -> Result<Buffer> {
            let w = weight(part, &[rows, cols])?;
            if w.dtype == "F16" {
                return read_weight(&gpu.device, w, (rows * cols * 2) as u64);
            }
            let words: Vec<_> = float_values(w)?
                .into_iter()
                .map(|v| f16::from_f32(v).to_bits())
                .collect();
            ensure!(words.len() == rows * cols, "truncated HC matrix");
            upload(&gpu.device, &words)
        };
        let gain = float_values(weight("hc_norm", &[10240])?)?;
        ensure!(gain.len() == 10240, "truncated HC gain");
        Ok(Self {
            gpu,
            eps,
            gain: upload(&gpu.device, &gain)?,
            down: dense("input_mix_weight_down", 320, 10240)?,
            up: dense("input_mix_weight_up", 10240, 320)?,
            inject: combine
                .then(|| dense("block_inject_weight", 4, 10240))
                .transpose()?,
        })
    }

    fn pipeline(&self, x: &MatrixView, width: u32) -> Result<&Pipelines> {
        ensure!(
            x.shape() == [1, width]
                && x.buffer.device().registry_id() == self.gpu.device.registry_id(),
            "invalid Flash HC input shape or device"
        );
        Ok(&self.gpu.hc[usize::from(x.dtype == FloatType::F32)])
    }

    fn command<T>(&self, encode: impl FnOnce(&CommandBufferRef) -> Result<T>) -> Result<T> {
        objc::rc::autoreleasepool(|| {
            let cmd = self.gpu.queue.new_command_buffer();
            let value = encode(cmd)?;
            cmd.commit();
            cmd.wait_until_completed();
            ensure!(
                cmd.status() == MTLCommandBufferStatus::Completed,
                "Flash HC command failed ({:?})",
                cmd.status()
            );
            Ok(value)
        })
    }

    pub fn read(&self, x: &MatrixView, trace: bool) -> Result<FlashHcRead> {
        let p = self.pipeline(x, 10240)?;
        ensure!(
            p.norm.thread_execution_width() == 32
                && p.norm.max_total_threads_per_threadgroup() >= 640,
            "HC needs SIMD32 and 640 threads for normalization"
        );
        let storage = if trace {
            Storage::StorageModeShared
        } else {
            Storage::StorageModePrivate
        };
        let output = |w, mode| MatrixView::output(self.gpu, x.dtype, 1, w, mode);
        let (normalized, down, active, up, mixed, gate) = self.command(|cmd| {
            let normalized = output(10240, storage)?;
            Metal::encode(
                cmd,
                &p.norm,
                &[&x.buffer, &self.gain, &normalized.buffer],
                [
                    x.layout.offset,
                    x.layout.strides[1],
                    i64::from(self.eps.to_bits()),
                ],
                groups(4),
                640,
            );
            let gemv = |v: &MatrixView, weight: &Buffer, n: u32| -> Result<MatrixView> {
                let width = v.shape()[1];
                let (count, threads) = crate::contracts::flash_hc_dispatch(width, n)
                    .context("invalid Flash HC matrix dimensions")?;
                let pipeline = if width == 320 { &p.wide } else { &p.narrow };
                ensure!(
                    pipeline.thread_execution_width() == 32
                        && pipeline.max_total_threads_per_threadgroup() >= u64::from(threads),
                    "unsupported Flash HC GEMV geometry"
                );
                let out = output(n, storage)?;
                Metal::encode(
                    cmd,
                    pipeline,
                    &[&v.buffer, weight, &out.buffer],
                    [n],
                    groups(count),
                    threads as usize,
                );
                Ok(out)
            };
            let activate = |v: &MatrixView, gate: bool| -> Result<MatrixView> {
                let n = v.shape()[1];
                let a = output(
                    n,
                    if gate {
                        Storage::StorageModeShared
                    } else {
                        storage
                    },
                )?;
                Metal::encode(
                    cmd,
                    &p.activate,
                    &[&v.buffer, &a.buffer],
                    [n, u32::from(gate)],
                    groups(n.div_ceil(32)),
                    32,
                );
                Ok(a)
            };
            let down = gemv(&normalized, &self.down, 320)?;
            let active = activate(&down, false)?;
            let up = gemv(&active, &self.up, 10240)?;
            let mixed = output(2560, Storage::StorageModeShared)?;
            Metal::encode(
                cmd,
                &p.mix,
                &[&up.buffer, &normalized.buffer, &mixed.buffer],
                [2560u32],
                groups(80),
                32,
            );
            let gate = self
                .inject
                .as_ref()
                .map(|w| activate(&gemv(&normalized, w, 4)?, true))
                .transpose()?;
            Ok((normalized, down, active, up, mixed, gate))
        })?;
        Ok(FlashHcRead {
            mixed,
            gate,
            trace: trace
                .then(|| -> Result<_> {
                    Ok(FlashHcTrace {
                        normalized: normalized.read_output()?,
                        down: down.read_output()?,
                        active: active.read_output()?,
                        up: up.read_output()?,
                    })
                })
                .transpose()?,
        })
    }

    /// Read the four streams, run the ordinary demand-loaded MLP and inject its residual.
    pub fn forward_moe(
        &self,
        streams: &MatrixView,
        moe: &mut super::FlashMoeRow,
        trace: bool,
    ) -> Result<(MatrixView, super::MoeOutput, FlashHcRead)> {
        ensure!(
            self.inject.is_some(),
            "Flash MLP requires an injection gate"
        );
        let read = self.read(streams, trace)?;
        let experts = moe.submit(&read.mixed)?;
        let written = self.write(
            streams,
            experts.completed()?,
            read.gate.as_ref().context("missing MLP injection")?,
        )?;
        Ok((written, experts, read))
    }
    pub fn write(
        &self,
        streams: &MatrixView,
        branch: &MatrixView,
        gate: &MatrixView,
    ) -> Result<MatrixView> {
        let p = self.pipeline(streams, 10240)?;
        self.pipeline(branch, 2560)?;
        self.pipeline(gate, 4)?;
        ensure!(
            gate.dtype == streams.dtype
                && (streams.dtype == FloatType::F32 || branch.dtype == FloatType::F16),
            "unsupported HC residual dtype promotion"
        );
        self.command(|cmd| {
            let out = MatrixView::output(
                self.gpu,
                streams.dtype,
                1,
                10240,
                Storage::StorageModeShared,
            )?;
            Metal::encode(
                cmd,
                &p.write,
                &[&streams.buffer, &branch.buffer, &gate.buffer, &out.buffer],
                [
                    streams.layout.offset,
                    streams.layout.strides[1],
                    branch.layout.offset,
                    branch.layout.strides[1],
                    gate.layout.offset,
                    gate.layout.strides[1],
                    i64::from(branch.dtype == FloatType::F32),
                ],
                groups(320),
                32,
            );
            Ok(out)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_layout::MatrixLayout;
    use serde_json::Value;

    #[test]
    fn hc_weights_decode_supported_storage_and_reject_integer_weights() -> Result<()> {
        use std::io::Write;
        let values = [-0.0f32, -1.5, 0.125, 32.0];
        for dtype in ["F16", "BF16", "F32", "I32"] {
            let data: Vec<u8> = match dtype {
                "F16" => values
                    .iter()
                    .flat_map(|&x| f16::from_f32(x).to_le_bytes())
                    .collect(),
                "BF16" => values
                    .iter()
                    .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
                    .collect(),
                _ => values.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            };
            let header = serde_json::to_vec(
                &serde_json::json!({"weight":{"dtype":dtype,"shape":[4],"data_offsets":[0,data.len()]}}),
            )?;
            let mut file = tempfile::NamedTempFile::new()?;
            file.write_all(&(header.len() as u64).to_le_bytes())?;
            file.write_all(&header)?;
            file.write_all(&data)?;
            let tensors = crate::checkpoint::read_header(file.path())?;
            let decoded = float_values(&tensors["weight"]);
            if dtype == "I32" {
                assert!(decoded.is_err());
            } else {
                assert_eq!(
                    decoded?.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                    values.map(f32::to_bits)
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "independent Flash HC write capture and physical Apple GPU"]
    fn frozen_residual_writes_preserve_mixed_precision_rounding() -> Result<()> {
        let root = std::path::PathBuf::from(
            std::env::var_os("MLXL3_HC_FROZEN").context("missing frozen HC writes")?,
        );
        let c = crate::checkpoint::inspect(std::path::Path::new(
            &std::env::var_os("MLXL3_HC_MODEL").context("missing model")?,
        ))?;
        let spec: Value = serde_json::from_slice(&std::fs::read(root.join("reference.json"))?)?;
        ensure!(spec["status"] == "complete");
        let cases = spec["hc_write"].as_array().context("missing writes")?;
        ensure!(cases.len() == 6);
        let gpu = Metal::new()?;
        let hc = FlashHc::load(
            &gpu,
            &c,
            "model.language_model.layers.0.mlp_hyper_connection",
            1e-6,
            true,
        )?;
        for case in cases {
            let values = |key: &str, n: u32| -> Result<Vec<f32>> {
                let d = &case[key];
                ensure!(d["count"].as_u64() == Some(u64::from(n)));
                let data = std::fs::read(root.join(d["file"].as_str().context("missing data")?))?;
                ensure!(data.len() == n as usize * 4);
                Ok(data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&v| f32::from_le_bytes(v))
                    .collect())
            };
            let view = |key: &str, n: u32, half: bool| -> Result<MatrixView> {
                let bits = values(key, n)?;
                let storage = if half {
                    gpu.f16_storage(
                        &bits
                            .iter()
                            .map(|&v| f16::from_f32(v).to_bits())
                            .collect::<Vec<_>>(),
                    )?
                } else {
                    gpu.f32_storage(&bits)?
                };
                storage.matrix(MatrixLayout {
                    shape: [1, n],
                    strides: [i64::from(n), 1],
                    offset: 0,
                })
            };
            let half = match case["branch_dtype"].as_str() {
                Some("mlx.core.float16") => true,
                Some("mlx.core.float32") => false,
                _ => bail!("unexpected branch dtype"),
            };
            let streams = view("streams", 10240, false)?;
            let branch = view("branch", 2560, half)?;
            let gate = view("gate", 4, false)?;
            ensure!(hc.write(&streams, &gate, &branch).is_err());
            let output = hc.write(&streams, &branch, &gate)?.read_output()?;
            let expected = values("output", 10240)?;
            ensure!(expected.iter().all(|v| v.is_finite()));
            assert_eq!(
                output,
                expected.into_iter().map(f32::to_bits).collect::<Vec<_>>()
            );
        }
        Ok(())
    }
}
