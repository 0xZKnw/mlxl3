//! Strided operators shared by quantized LFM and standard Qwen blocks; outputs are F16.
use super::{MatrixView, Metal, view::FloatType};
use anyhow::{Result, ensure};
use metal::{CommandBufferRef, ComputePipelineState, MTLResourceOptions as Storage, MTLSize};

pub(super) struct Pipelines {
    norm: [ComputePipelineState; 2],
    dense: [ComputePipelineState; 2],
    gqa: [ComputePipelineState; 3],
    rope: ComputePipelineState,
}

impl Pipelines {
    pub(super) fn load(p: &impl Fn(&str) -> Result<ComputePipelineState>) -> Result<Self> {
        Ok(Self {
            norm: [p("model_rms_half")?, p("model_rms_float")?],
            dense: [p("model_dense_half")?, p("model_dense_float")?],
            gqa: [p("model_gqa_64")?, p("model_gqa_128")?, p("model_gqa_256")?],
            rope: p("model_rope")?,
        })
    }
}

impl Metal {
    pub fn encode_model_rms(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        gain: &MatrixView,
        eps: f32,
        storage: Storage,
    ) -> Result<MatrixView> {
        let [rows, width] = x.shape();
        ensure!(
            rows > 0
                && width > 0
                && x.dtype == FloatType::F16
                && gain.shape() == [1, width]
                && eps.is_finite()
                && eps > 0.,
            "invalid native model RMS input"
        );
        ensure!(
            x.buffer.device().registry_id() == self.device.registry_id()
                && gain.buffer.device().registry_id() == self.device.registry_id(),
            "foreign model RMS storage"
        );
        let threads = u64::from(width.div_ceil(4).div_ceil(32).min(32)) * 32;
        let pipeline = &self.model_ops.norm[usize::from(gain.dtype == FloatType::F32)];
        ensure!(
            pipeline.thread_execution_width() == 32
                && pipeline.max_total_threads_per_threadgroup() >= threads,
            "unsupported model RMS threadgroup"
        );
        let out = MatrixView::output(self, FloatType::F16, rows, width, storage)?;
        Self::encode(
            cmd,
            pipeline,
            &[&x.buffer, &gain.buffer, &out.buffer],
            [
                i64::from(width),
                x.layout.offset,
                x.layout.strides[0],
                x.layout.strides[1],
                gain.layout.offset,
                gain.layout.strides[1],
                i64::from(eps.to_bits()),
            ],
            MTLSize::new(u64::from(rows), 1, 1),
            threads as usize,
        );
        Ok(out)
    }

    pub fn encode_model_dense(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        weight: &MatrixView,
        storage: Storage,
    ) -> Result<MatrixView> {
        let [rows, input] = x.shape();
        let [output, width] = weight.shape();
        // shortcut: F32 weight batches need the separately qualified native GEMM.
        ensure!(
            weight.dtype == FloatType::F16 || rows == 1,
            "FP32-weight GEMM batches are not qualified"
        );
        ensure!(
            rows > 0 && input > 0 && output > 0 && width == input && x.dtype == FloatType::F16,
            "invalid native dense projection"
        );
        ensure!(
            x.buffer.device().registry_id() == self.device.registry_id()
                && weight.buffer.device().registry_id() == self.device.registry_id(),
            "foreign native dense storage"
        );
        let pipeline = &self.model_ops.dense[usize::from(weight.dtype == FloatType::F32)];
        ensure!(
            pipeline.thread_execution_width() == 32
                && pipeline.max_total_threads_per_threadgroup() >= 128,
            "unsupported native dense threadgroup"
        );
        let out = MatrixView::output(self, FloatType::F16, rows, output, storage)?;
        Self::encode(
            cmd,
            pipeline,
            &[&x.buffer, &weight.buffer, &out.buffer],
            [
                i64::from(input),
                i64::from(output),
                x.layout.offset,
                x.layout.strides[0],
                x.layout.strides[1],
                weight.layout.offset,
                weight.layout.strides[0],
                weight.layout.strides[1],
            ],
            MTLSize::new(u64::from(output.div_ceil(16)), u64::from(rows), 1),
            128,
        );
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_model_gqa(
        &self,
        cmd: &CommandBufferRef,
        q: &MatrixView,
        keys: &MatrixView,
        values: &MatrixView,
        q_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        storage: Storage,
    ) -> Result<MatrixView> {
        let [query_width, kv_width] =
            crate::contracts::model_gqa_widths(q_heads, kv_heads, head_dim)
                .ok_or_else(|| anyhow::anyhow!("invalid native GQA geometry"))?;
        ensure!(
            q.shape() == [1, query_width]
                && keys.shape()[0] > 0
                && keys.shape()[1] == kv_width
                && values.shape() == keys.shape(),
            "invalid native GQA shapes"
        );
        ensure!(
            [q, keys, values].iter().all(|v| v.dtype == FloatType::F16
                && v.buffer.device().registry_id() == self.device.registry_id()),
            "invalid native GQA storage"
        );
        let pipeline = &self.model_ops.gqa[match head_dim {
            64 => 0,
            128 => 1,
            _ => 2,
        }];
        ensure!(
            pipeline.thread_execution_width() == 32
                && pipeline.max_total_threads_per_threadgroup() >= 1024,
            "unsupported native GQA threadgroup"
        );
        let out = MatrixView::output(self, FloatType::F16, 1, query_width, storage)?;
        Self::encode(
            cmd,
            pipeline,
            &[&q.buffer, &keys.buffer, &values.buffer, &out.buffer],
            [
                i64::from(keys.shape()[0]),
                i64::from(q_heads),
                i64::from(kv_heads),
                q.layout.offset,
                q.layout.strides[1],
                keys.layout.offset,
                keys.layout.strides[0],
                keys.layout.strides[1],
                values.layout.offset,
                values.layout.strides[0],
                values.layout.strides[1],
                i64::from((head_dim as f32).powf(-0.5).to_bits()),
            ],
            MTLSize::new(u64::from(q_heads), 1, 1),
            1024,
        );
        Ok(out)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_model_rope(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        heads: u32,
        head_dim: u32,
        rotary: u32,
        theta: f32,
        position: u32,
        storage: Storage,
    ) -> Result<MatrixView> {
        let width = heads
            .checked_mul(head_dim)
            .ok_or_else(|| anyhow::anyhow!("native RoPE extent overflow"))?;
        ensure!(
            heads > 0
                && head_dim > 0
                && rotary > 0
                && rotary.is_multiple_of(2)
                && rotary <= head_dim
                && theta.is_finite()
                && theta > 0.,
            "invalid native RoPE geometry"
        );
        ensure!(
            x.dtype == FloatType::F16
                && (x.shape() == [1, width] || x.shape() == [heads, head_dim])
                && x.buffer.device().registry_id() == self.device.registry_id(),
            "invalid native RoPE input"
        );
        let stride = if x.shape()[0] == 1 {
            i64::from(head_dim)
                .checked_mul(x.layout.strides[1])
                .ok_or_else(|| anyhow::anyhow!("native RoPE stride overflow"))?
        } else {
            x.layout.strides[0]
        };
        let out = MatrixView::output(self, FloatType::F16, 1, width, storage)?;
        let pipeline = &self.model_ops.rope;
        ensure!(
            pipeline.thread_execution_width() == 32
                && pipeline.max_total_threads_per_threadgroup() >= 32,
            "unsupported native RoPE threadgroup"
        );
        Self::encode(
            cmd,
            pipeline,
            &[&x.buffer, &out.buffer],
            [
                i64::from(heads),
                i64::from(head_dim),
                i64::from(rotary),
                i64::from(position),
                x.layout.offset,
                stride,
                x.layout.strides[1],
                i64::from(theta.log2().to_bits()),
            ],
            MTLSize::new(
                u64::from((heads * (head_dim - rotary / 2)).div_ceil(32)),
                1,
                1,
            ),
            32,
        );
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_layout::MatrixLayout;
    use anyhow::Context;
    use half::f16;
    use serde::Deserialize;

    #[derive(Deserialize)]
    struct Input {
        values: Vec<f32>,
        layout: MatrixLayout,
        dtype: String,
    }
    impl Input {
        fn view(&self, gpu: &Metal) -> Result<MatrixView> {
            ensure!(
                self.values.iter().all(|v| v.is_finite()),
                "nonfinite model operator fixture"
            );
            let storage = match self.dtype.as_str() {
                "float16" => gpu.f16_storage(
                    &self
                        .values
                        .iter()
                        .map(|&v| f16::from_f32(v).to_bits())
                        .collect::<Vec<_>>(),
                )?,
                "float32" => gpu.f32_storage(&self.values)?,
                _ => anyhow::bail!("unknown fixture dtype"),
            };
            storage.matrix(self.layout)
        }
    }
    #[derive(Deserialize)]
    struct Case {
        name: String,
        op: String,
        x: Input,
        weight: Option<Input>,
        keys: Option<Input>,
        values: Option<Input>,
        heads: u32,
        kv_heads: u32,
        head_dim: u32,
        rotary: u32,
        theta: f32,
        position: u32,
        eps: f32,
        expected: Vec<u16>,
        #[serde(default)]
        expected_error: Option<String>,
    }
    #[test]
    #[ignore = "physical Apple GPU and independent model operator fixtures"]
    fn shared_operators_match_the_independent_mlx_fixtures() -> Result<()> {
        let cases: Vec<Case> = serde_json::from_reader(std::fs::File::open(std::env::var(
            "MLXL3_MODEL_OPS_FIXTURE",
        )?)?)?;
        ensure!(cases.len() >= 20, "incomplete model operator fixture");
        let gpu = Metal::new()?;
        let mut words = 0usize;
        let mut failures = Vec::new();
        for case in cases {
            let x = case.x.view(&gpu)?;
            let weight = case.weight.as_ref().map(|v| v.view(&gpu)).transpose()?;
            let keys = case.keys.as_ref().map(|v| v.view(&gpu)).transpose()?;
            let values = case.values.as_ref().map(|v| v.view(&gpu)).transpose()?;
            let operation = gpu.command(|cmd| match case.op.as_str() {
                "rms" => gpu.encode_model_rms(
                    cmd,
                    &x,
                    weight.as_ref().context("missing RMS gain")?,
                    case.eps,
                    Storage::StorageModeShared,
                ),
                "dense" => gpu.encode_model_dense(
                    cmd,
                    &x,
                    weight.as_ref().context("missing dense weight")?,
                    Storage::StorageModeShared,
                ),
                "gqa" => gpu.encode_model_gqa(
                    cmd,
                    &x,
                    keys.as_ref().context("missing keys")?,
                    values.as_ref().context("missing values")?,
                    case.heads,
                    case.kv_heads,
                    case.head_dim,
                    Storage::StorageModeShared,
                ),
                "rope" => gpu.encode_model_rope(
                    cmd,
                    &x,
                    case.heads,
                    case.head_dim,
                    case.rotary,
                    case.theta,
                    case.position,
                    Storage::StorageModeShared,
                ),
                _ => anyhow::bail!("unknown model fixture operator"),
            });
            if let Some(expected) = case.expected_error {
                let error = operation
                    .err()
                    .context("invalid model operation was accepted")?;
                ensure!(
                    error.to_string().contains(&expected),
                    "wrong operator rejection: {error}"
                );
                eprintln!("model-op {}: rejected as specified", case.name);
                continue;
            }
            let out = operation?;
            let actual = out.read_output()?;
            ensure!(
                !actual.is_empty() && actual.len() == case.expected.len(),
                "{}: wrong result extent",
                case.name
            );
            ensure!(
                actual.iter().all(|&v| f16::from_bits(v as u16).is_finite())
                    && case.expected.iter().all(|&v| f16::from_bits(v).is_finite()),
                "{}: nonfinite result",
                case.name
            );
            let differences = actual
                .iter()
                .zip(&case.expected)
                .filter(|(a, b)| **a != u32::from(**b))
                .count();
            eprintln!(
                "model-op {}: words={}, unequal={differences}",
                case.name,
                actual.len()
            );
            if differences != 0 {
                failures.push((case.name, differences));
            }
            words += actual.len();
        }
        eprintln!("model-ops checked {words} finite words");
        ensure!(
            failures.is_empty(),
            "unequal model operator outputs: {failures:?}"
        );
        Ok(())
    }
}
