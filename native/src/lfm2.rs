//! Batch-one, token-by-token LFM2 dense/MoE inference using MLX arrays.
//! Weights and operation order follow mlx-lm's `lfm2` implementations.
// Architecture adapted from mlx-lm, Copyright © 2025 Apple Inc. (MIT).
use crate::{
    array::{self, Array, Dtype},
    checkpoint::{self, Checkpoint},
    linear::{Exl3Group, Exl3Linear, checkpoint_array},
    moe::Exl3SwitchGlu,
    router,
};
use anyhow::{Context, Result, ensure};
use half::f16;
use serde::Deserialize;
use std::{collections::BTreeSet, fs::File, path::Path};

#[derive(Deserialize)]
pub(crate) struct Config {
    model_type: String,
    vocab_size: i32,
    hidden_size: i32,
    max_position_embeddings: i32,
    num_hidden_layers: usize,
    num_attention_heads: i32,
    num_key_value_heads: Option<i32>,
    norm_eps: f32,
    conv_bias: bool,
    #[serde(rename = "conv_L_cache")]
    conv_length: i32,
    block_dim: Option<i32>,
    block_ff_dim: Option<i32>,
    intermediate_size: Option<i32>,
    block_multiple_of: Option<i32>,
    #[serde(default)]
    block_auto_adjust_ff_dim: bool,
    block_ffn_dim_multiplier: Option<f64>,
    #[serde(default)]
    moe_intermediate_size: i32,
    #[serde(default)]
    num_experts: i32,
    #[serde(default)]
    num_experts_per_tok: usize,
    #[serde(default)]
    num_dense_layers: usize,
    #[serde(default)]
    norm_topk_prob: bool,
    #[serde(default)]
    use_expert_bias: bool,
    #[serde(default = "default_routed_scaling_factor")]
    routed_scaling_factor: f32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    rope_parameters: Option<serde_json::Value>,
    full_attn_idxs: Option<Vec<usize>>,
    layer_types: Option<Vec<String>>,
    head_dim: Option<i32>,
    sliding_window: Option<i32>,
    #[serde(default)]
    use_sandwich_norm: bool,
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn default_routed_scaling_factor() -> f32 {
    1.0
}

fn exact_moe_prefill_config(config: &Config, attention_layers: usize, mapped_k4_mcg: bool) -> bool {
    config.model_type == "lfm2_moe"
        && config.vocab_size == 128000
        && config.num_hidden_layers == 40
        && config.num_dense_layers == 2
        && config.num_attention_heads == 32
        && config.num_key_value_heads == Some(8)
        && config.conv_length == 3
        && attention_layers == 10
        && config.use_expert_bias
        && config.norm_topk_prob
        && config.routed_scaling_factor == 1.0
        && crate::contracts::lfm_moe_exact_prefill_geometry(
            config.hidden_size,
            config.moe_intermediate_size,
            config.num_experts,
            config.num_experts_per_tok,
            mapped_k4_mcg,
        )
}

fn exact_moe_projection_metadata(checkpoint: &Checkpoint) -> bool {
    (2..40).all(|layer| {
        let prefix = format!("model.layers.{layer}.feed_forward");
        let gate = format!("{prefix}.gate.weight");
        !checkpoint
            .modules
            .iter()
            .any(|name| name == &format!("{prefix}.gate"))
            && checkpoint.tensors.get(&gate).is_some_and(|info| {
                info.shape == [64, 2048] && matches!(info.dtype.as_str(), "F16" | "BF16" | "F32")
            })
            && (0..64).all(|expert| {
                [
                    ("w1", [128, 96, 64]),
                    ("w3", [128, 96, 64]),
                    ("w2", [96, 128, 64]),
                ]
                .iter()
                .all(|(projection, expected)| {
                    let base = format!("{prefix}.experts.{expert}.{projection}");
                    checkpoint
                        .tensors
                        .get(&format!("{base}.trellis"))
                        .is_some_and(|info| {
                            info.shape == *expected && matches!(info.dtype.as_str(), "U16" | "I16")
                        })
                        && checkpoint.tensors.contains_key(&format!("{base}.mcg"))
                        && !checkpoint.tensors.contains_key(&format!("{base}.mul1"))
                })
            })
    })
}

pub(crate) fn half_weight(
    checkpoint: &Checkpoint,
    name: &str,
    expected: Option<&[i32]>,
) -> Result<Array> {
    let weight = checkpoint_array(checkpoint, name)?;
    ensure!(
        matches!(
            weight.dtype(),
            Dtype::Float16 | Dtype::Float32 | Dtype::BFloat16
        ),
        "{name}: expected floating-point weights"
    );
    if let Some(shape) = expected {
        ensure!(
            weight.shape() == shape,
            "{name}: shape {:?}, expected {shape:?}",
            weight.shape()
        );
    }
    let weight = weight.astype(Dtype::Float16)?;
    weight.eval()?;
    Ok(weight)
}

/// Resident F16 table (current path) or LFM3-EMBED-LAZY row reads.
fn load_embeddings(
    checkpoint: &Checkpoint,
    name: &str,
    info: &checkpoint::TensorInfo,
    vocab: i32,
    dim: i32,
    lazy: bool,
) -> Result<Embeddings> {
    if !lazy {
        return Ok(Embeddings::Resident(half_weight(
            checkpoint,
            name,
            Some(&[vocab, dim]),
        )?));
    }
    let dtype = crate::linear::checkpoint_dtype(&info.dtype, name)?;
    ensure!(
        matches!(dtype, Dtype::Float16 | Dtype::Float32 | Dtype::BFloat16),
        "{name}: expected floating-point weights"
    );
    ensure!(
        info.shape == [usize::try_from(vocab)?, usize::try_from(dim)?],
        "{name}: shape {:?}, expected [{vocab}, {dim}]",
        info.shape
    );
    let size = if dtype == Dtype::Float32 { 4u64 } else { 2 };
    ensure!(
        info.data_offsets[1].checked_sub(info.data_offsets[0])
            == Some(u64::try_from(vocab)? * u64::try_from(dim)? * size),
        "{name}: byte range does not match its shape"
    );
    info.source()?;
    Ok(Embeddings::Lazy {
        info: Box::new(info.clone()),
        dtype,
        rows: u32::try_from(vocab)?,
        dim,
    })
}

enum ProjectionWeights {
    Exl3(Exl3Linear),
    Affine(crate::affine::AffineLinear),
    Dense(Array),
}

pub(crate) struct Projection {
    weights: ProjectionWeights,
    bias: Option<Array>,
    logical_input: i32,
    logical_output: i32,
}

impl Projection {
    pub(crate) fn collect_weights<'a>(&'a self, out: &mut Vec<&'a Array>) {
        match &self.weights {
            ProjectionWeights::Exl3(w) => out.extend(w.weight_arrays()),
            ProjectionWeights::Affine(w) => out.extend(w.weight_arrays()),
            ProjectionWeights::Dense(w) => out.push(w),
        }
        out.extend(&self.bias);
    }

    pub(crate) fn is_exl3(&self) -> bool {
        matches!(self.weights, ProjectionWeights::Exl3(_))
    }

    /// The F16 weight of a bias-free dense projection (LFM3-CERT-HEAD).
    pub(crate) fn dense_weight(&self) -> Option<&Array> {
        match (&self.weights, &self.bias) {
            (ProjectionWeights::Dense(weight), None) => Some(weight),
            _ => None,
        }
    }

    pub(crate) fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        input: i32,
        output: i32,
        bias: bool,
    ) -> Result<Self> {
        Self::load_logical(checkpoint, prefix, input, output, bias)
    }

    pub(crate) fn load_logical(
        checkpoint: &Checkpoint,
        prefix: &str,
        input: i32,
        output: i32,
        bias: bool,
    ) -> Result<Self> {
        let is_exl3 = checkpoint.modules.iter().any(|name| name == prefix)
            || checkpoint
                .tensors
                .contains_key(&format!("{prefix}.trellis"));
        if bias {
            ensure!(
                checkpoint.tensors.contains_key(&format!("{prefix}.bias")),
                "{prefix}: missing projection bias"
            );
        }
        let weights = if is_exl3 {
            let quantized = Exl3Linear::from_checkpoint(checkpoint, prefix)?;
            ensure!(
                quantized.input_dims() >= input
                    && quantized.input_dims() - input < 128
                    && quantized.output_dims() >= output
                    && quantized.output_dims() - output < 128,
                "{prefix}: invalid EXL3 projection dimensions"
            );
            ProjectionWeights::Exl3(quantized)
        } else if checkpoint.model_type == "qwen3_5_mtp"
            && checkpoint.tensors.contains_key(&format!("{prefix}.scales"))
        {
            ProjectionWeights::Affine(crate::affine::AffineLinear::load(
                checkpoint, prefix, input, output, None,
            )?)
        } else {
            ProjectionWeights::Dense(half_weight(
                checkpoint,
                &format!("{prefix}.weight"),
                Some(&[output, input]),
            )?)
        };
        // Exl3Linear applies its checkpoint bias itself.
        let bias = if bias && !is_exl3 {
            Some(half_weight(
                checkpoint,
                &format!("{prefix}.bias"),
                Some(&[output]),
            )?)
        } else {
            None
        };
        Ok(Self {
            weights,
            bias,
            logical_input: input,
            logical_output: output,
        })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        ensure!(
            x.shape().last() == Some(&self.logical_input),
            "projection expects {} logical inputs",
            self.logical_input
        );
        let value = match &self.weights {
            ProjectionWeights::Affine(linear) => linear.forward(x)?,
            ProjectionWeights::Exl3(linear) => {
                let padded = if linear.input_dims() == self.logical_input {
                    x.try_clone()?
                } else {
                    let mut shape = x.shape().to_vec();
                    *shape
                        .last_mut()
                        .context("projection input must have rank")? =
                        linear.input_dims() - self.logical_input;
                    let zeros = Array::zeros_dtype(&shape, x.dtype())?;
                    Array::concatenate(&[x, &zeros], x.shape().len() as i32 - 1)?
                };
                let value = linear.forward(&padded)?;
                if linear.output_dims() == self.logical_output {
                    value
                } else {
                    value.slice(value.shape().len() as i32 - 1, 0, self.logical_output)?
                }
            }
            ProjectionWeights::Dense(weight) => x.matmul(&weight.transpose(&[1, 0])?)?,
        };
        match &self.bias {
            Some(bias) => value.add(bias),
            None => Ok(value),
        }
    }

    pub(crate) fn forward_flash_hc(&self, x: &Array) -> Result<Array> {
        let ProjectionWeights::Dense(weight) = &self.weights else {
            return self.forward(x);
        };
        let (bm, bn) = match (self.logical_input, self.logical_output) {
            (10240, 320 | 4) => (1, 8),
            (320, 10240) => (8, 1),
            _ => return self.forward(x),
        };
        if self.bias.is_some()
            || x.dtype() != Dtype::Float32
            || !matches!(x.shape(), [1, k] | [1, 1, k] if *k == self.logical_input)
        {
            return self.forward(x);
        }
        // Read half weights directly while retaining MLX 0.32.2's FP32 GEMV order.
        let mut shape = x.shape().to_vec();
        *shape.last_mut().expect("validated HC rank") = self.logical_output;
        Ok(array::metal_kernel(
            &format!(
                "mlxl3_flash_hc_gemv_{}_{}",
                self.logical_input, self.logical_output
            ),
            &["x", "weight"],
            &["out"],
            &format!(
                "#define K {}\n#define N {}\n#define BM {bm}\n#define BN {bn}\n",
                self.logical_input, self.logical_output
            ),
            include_str!("../shaders/flash_hc_gemv.metal"),
            &[x, weight],
            &[shape],
            &[Dtype::Float32],
            [(self.logical_output / (bm * 4)) * (bm * bn * 32), 1, 1],
            [bm * bn * 32, 1, 1],
        )?
        .remove(0))
    }

    pub(crate) fn forward_dense_rows_exact(&self, x: &Array) -> Result<Array> {
        let ProjectionWeights::Dense(weight) = &self.weights else {
            anyhow::bail!("exact dense rows require dense projection weights");
        };
        ensure!(
            self.bias.is_none()
                && x.shape().len() == 2
                && (2..=8).contains(&x.shape()[0])
                && x.shape()[1] == self.logical_input
                && self.logical_input % 128 == 0
                && self.logical_output % 16 == 0,
            "invalid exact dense row input"
        );
        let rows = x.shape()[0];
        let source = format!(
            r#"
                constexpr uint INPUT_DIMS = {input}u;
                constexpr uint OUTPUT_DIMS = {output}u;
                uint output = threadgroup_position_in_grid.x * 16u
                    + simdgroup_index_in_threadgroup * 4u;
                uint row = threadgroup_position_in_grid.y;
                uint lane = thread_index_in_simdgroup;
                float result[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
                uint k = lane * 4u;
                for (uint block = 0u; block < INPUT_DIMS / 128u; ++block) {{
                    float coeff[4];
                    for (uint n = 0u; n < 4u; ++n) {{
                        coeff[n] = float(x[row * INPUT_DIMS + k + n]);
                    }}
                    for (uint m = 0u; m < 4u; ++m) {{
                        for (uint n = 0u; n < 4u; ++n) {{
                            result[m] += float(weight[(output + m) * INPUT_DIMS + k + n])
                                * coeff[n];
                        }}
                    }}
                    k += 128u;
                }}
                for (ushort offset = 16u; offset >= 1u; offset >>= 1u) {{
                    for (uint m = 0u; m < 4u; ++m) {{
                        result[m] += simd_shuffle_down(result[m], offset);
                    }}
                }}
                if (lane == 0u) {{
                    for (uint m = 0u; m < 4u; ++m) {{
                        y[row * OUTPUT_DIMS + output + m] = half(result[m]);
                    }}
                }}
            "#,
            input = self.logical_input,
            output = self.logical_output,
        );
        Ok(array::metal_kernel(
            &format!(
                "mlxl3_rs_dense_rows_exact_{}_{}_v1",
                self.logical_input, self.logical_output
            ),
            &["x", "weight"],
            &["y"],
            "",
            &source,
            &[x, weight],
            &[vec![rows, self.logical_output]],
            &[Dtype::Float16],
            [(self.logical_output / 16) * 128, rows, 1],
            [128, 1, 1],
        )?
        .remove(0))
    }

    /// LFM3-PLD: the `forward_dense_rows_exact` arithmetic for every row, but
    /// each thread loads its weights once and applies them to all rows, so the
    /// head is read from memory once per verification instead of once per row.
    pub(crate) fn forward_dense_rows_shared(&self, x: &Array) -> Result<Array> {
        let ProjectionWeights::Dense(weight) = &self.weights else {
            anyhow::bail!("shared dense rows require dense projection weights");
        };
        ensure!(
            self.bias.is_none()
                && x.shape().len() == 2
                && (1..=LFM3_VERIFY_MAX_ROWS).contains(&x.shape()[0])
                && x.shape()[1] == self.logical_input
                && x.dtype() == Dtype::Float16
                && weight.dtype() == Dtype::Float16
                && self.logical_input % 128 == 0
                && self.logical_output % 16 == 0,
            "invalid shared dense row input"
        );
        let rows = x.shape()[0];
        let source = format!(
            r#"
                constexpr uint INPUT_DIMS = {input}u;
                constexpr uint OUTPUT_DIMS = {output}u;
                constexpr uint ROWS = {rows}u;
                uint output = threadgroup_position_in_grid.x * 16u
                    + simdgroup_index_in_threadgroup * 4u;
                uint lane = thread_index_in_simdgroup;
                float result[ROWS][4];
                for (uint r = 0u; r < ROWS; ++r) {{
                    for (uint m = 0u; m < 4u; ++m) {{
                        result[r][m] = 0.0f;
                    }}
                }}
                uint k = lane * 4u;
                for (uint block = 0u; block < INPUT_DIMS / 128u; ++block) {{
                    float w[4][4];
                    for (uint m = 0u; m < 4u; ++m) {{
                        for (uint n = 0u; n < 4u; ++n) {{
                            w[m][n] = float(weight[(output + m) * INPUT_DIMS + k + n]);
                        }}
                    }}
                    for (uint r = 0u; r < ROWS; ++r) {{
                        float coeff[4];
                        for (uint n = 0u; n < 4u; ++n) {{
                            coeff[n] = float(x[r * INPUT_DIMS + k + n]);
                        }}
                        for (uint m = 0u; m < 4u; ++m) {{
                            for (uint n = 0u; n < 4u; ++n) {{
                                result[r][m] += w[m][n] * coeff[n];
                            }}
                        }}
                    }}
                    k += 128u;
                }}
                for (ushort offset = 16u; offset >= 1u; offset >>= 1u) {{
                    for (uint r = 0u; r < ROWS; ++r) {{
                        for (uint m = 0u; m < 4u; ++m) {{
                            result[r][m] += simd_shuffle_down(result[r][m], offset);
                        }}
                    }}
                }}
                if (lane == 0u) {{
                    for (uint r = 0u; r < ROWS; ++r) {{
                        for (uint m = 0u; m < 4u; ++m) {{
                            y[r * OUTPUT_DIMS + output + m] = half(result[r][m]);
                        }}
                    }}
                }}
            "#,
            input = self.logical_input,
            output = self.logical_output,
        );
        Ok(array::metal_kernel(
            &format!(
                "mlxl3_rs_dense_rows_shared_{}_{}_{rows}_v1",
                self.logical_input, self.logical_output
            ),
            &["x", "weight"],
            &["y"],
            "",
            &source,
            &[x, weight],
            &[vec![rows, self.logical_output]],
            &[Dtype::Float16],
            [(self.logical_output / 16) * 128, 1, 1],
            [128, 1, 1],
        )?
        .remove(0))
    }
}

/// LFM3-PLD: anchor plus at most three prompt-lookup proposals.
pub(crate) const LFM3_VERIFY_MAX_ROWS: i32 = 4;

pub(crate) enum ProjectionBundle<const N: usize> {
    Separate(Box<[Projection; N]>),
    Grouped(Box<Exl3Group>),
}

pub(crate) type Qkv = ProjectionBundle<3>;

impl<const N: usize> ProjectionBundle<N> {
    pub(crate) fn new(projections: [Projection; N]) -> Result<Self> {
        let linears: Vec<_> = projections
            .iter()
            .filter_map(|p| match &p.weights {
                ProjectionWeights::Exl3(l) => Some(l),
                _ => None,
            })
            .collect();
        if linears.len() == N
            && Exl3Group::compatible(&linears)
            && (N != 2
                || projections.iter().zip(&linears).all(|(p, l)| {
                    p.bias.is_none()
                        && l.lfm3_grouped_prefill_enabled()
                        && p.logical_input == l.input_dims()
                        && p.logical_output == l.output_dims()
                }))
        {
            let linears = projections
                .into_iter()
                .map(|p| match p.weights {
                    ProjectionWeights::Exl3(l) => l,
                    _ => unreachable!(),
                })
                .collect();
            Ok(Self::Grouped(Box::new(Exl3Group::new(linears)?)))
        } else {
            Ok(Self::Separate(Box::new(projections)))
        }
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Vec<Array>> {
        match self {
            Self::Grouped(g) => g.forward(x),
            Self::Separate(p) => p.iter().map(|p| p.forward(x)).collect(),
        }
    }
}

struct Attention {
    qkv: Qkv,
    output: Projection,
    q_norm: Array,
    k_norm: Array,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    eps: f32,
    theta: f32,
    keys: Option<Array>,
    values: Option<Array>,
    lfm3_rope: Option<crate::lfm3::Rope>,
    gate: Option<Projection>,
    window: Option<i32>,
}

impl Attention {
    fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        config: &Config,
        theta: f32,
        kind: &str,
    ) -> Result<Self> {
        let dim = config.hidden_size;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads.unwrap_or(heads);
        let lfm3 = config.model_type == "lfm3";
        let head_dim = if lfm3 {
            config.head_dim.unwrap_or(dim / heads)
        } else {
            dim / heads
        };
        Ok(Self {
            qkv: Qkv::new([
                Projection::load(
                    checkpoint,
                    &format!("{prefix}.q_proj"),
                    dim,
                    heads * head_dim,
                    false,
                )?,
                Projection::load(
                    checkpoint,
                    &format!("{prefix}.k_proj"),
                    dim,
                    kv_heads * head_dim,
                    false,
                )?,
                Projection::load(
                    checkpoint,
                    &format!("{prefix}.v_proj"),
                    dim,
                    kv_heads * head_dim,
                    false,
                )?,
            ])?,
            output: Projection::load(
                checkpoint,
                &format!("{prefix}.{}", if lfm3 { "o_proj" } else { "out_proj" }),
                heads * head_dim,
                dim,
                false,
            )?,
            q_norm: half_weight(
                checkpoint,
                &format!(
                    "{prefix}.{}.weight",
                    if lfm3 { "q_norm" } else { "q_layernorm" }
                ),
                Some(&[head_dim]),
            )?,
            k_norm: half_weight(
                checkpoint,
                &format!(
                    "{prefix}.{}.weight",
                    if lfm3 { "k_norm" } else { "k_layernorm" }
                ),
                Some(&[head_dim]),
            )?,
            heads,
            kv_heads,
            head_dim,
            eps: config.norm_eps,
            theta,
            keys: None,
            values: None,
            lfm3_rope: if lfm3 {
                Some(crate::lfm3::Rope::from_config(
                    config
                        .rope_parameters
                        .as_ref()
                        .context("missing LFM3 RoPE parameters")?,
                    kind,
                    head_dim,
                )?)
            } else {
                None
            },
            gate: if lfm3 {
                Some(Projection::load(
                    checkpoint,
                    &format!("{prefix}.gate_proj"),
                    dim,
                    heads,
                    false,
                )?)
            } else {
                None
            },
            window: if kind == "sliding_attention" {
                Some(
                    config
                        .sliding_window
                        .context("missing LFM3 sliding window")?,
                )
            } else {
                None
            },
        })
    }

    fn forward(&mut self, x: &Array, offset: i32) -> Result<Array> {
        let time = *x
            .shape()
            .get(1)
            .context("LFM2 attention input must have time")?;
        ensure!(time > 0, "empty LFM2 attention input");
        let projections = self.qkv.forward(x)?;
        let mut q = projections[0]
            .reshape(&[1, time, self.heads, self.head_dim])?
            .rms_norm(&self.q_norm, self.eps)?
            .transpose(&[0, 2, 1, 3])?;
        let mut k = projections[1]
            .reshape(&[1, time, self.kv_heads, self.head_dim])?
            .rms_norm(&self.k_norm, self.eps)?
            .transpose(&[0, 2, 1, 3])?;
        if let Some(rope) = &self.lfm3_rope {
            q = rope.apply(&q, offset)?;
            k = rope.apply(&k, offset)?;
        } else {
            q = q.rope(self.head_dim, self.theta, offset)?;
            k = k.rope(self.head_dim, self.theta, offset)?;
        }
        let v = projections[2]
            .reshape(&[1, time, self.kv_heads, self.head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let keys = match &self.keys {
            Some(previous) => Array::concatenate(&[previous, &k], 2)?,
            None => k,
        };
        let values = match &self.values {
            Some(previous) => Array::concatenate(&[previous, &v], 2)?,
            None => v,
        };
        let attended = if let Some(window) = self.window {
            let key_count = keys.shape()[2];
            let mask = crate::lfm3::sliding_mask(time, key_count, window)?.astype(q.dtype())?;
            Array::sdpa_mask(&q, &keys, &values, (self.head_dim as f32).powf(-0.5), &mask)?
        } else {
            Array::sdpa(
                &q,
                &keys,
                &values,
                (self.head_dim as f32).powf(-0.5),
                time > 1,
            )?
        };
        let out = if let Some(projection) = &self.gate {
            let gate = projection.forward(x)?;
            if attended.dtype() == Dtype::Float16 && gate.dtype() == Dtype::Float16 {
                crate::lfm3_kernels::attention_gate(&attended, &gate)?
            } else {
                attended
                    .transpose(&[0, 2, 1, 3])?
                    .mul(&gate.sigmoid()?.reshape(&[1, time, self.heads, 1])?)?
                    .reshape(&[1, time, self.heads * self.head_dim])?
            }
        } else {
            attended
                .transpose(&[0, 2, 1, 3])?
                .reshape(&[1, time, self.heads * self.head_dim])?
        };
        // Keep W-1 past keys: the next query itself occupies the last window slot.
        let start = self
            .window
            .map_or(0, |w| (keys.shape()[2] - (w - 1)).max(0));
        let keys = keys.slice(2, start, keys.shape()[2])?;
        let values = values.slice(2, start, values.shape()[2])?;
        self.keys = Some(keys);
        self.values = Some(values);
        self.output.forward(&out)
    }

    /// LFM3-PLD: `rows` T1 steps without mutating the cache. Every query sees
    /// exactly the keys its T1 step would see (same count, same SWA mask).
    fn verify(&self, x: &Array, offset: i32) -> Result<(Array, PendingLayer)> {
        let time = *x
            .shape()
            .get(1)
            .context("LFM3 verification input must have time")?;
        ensure!(time > 0, "empty LFM3 verification input");
        let rope = self
            .lfm3_rope
            .as_ref()
            .context("verification requires LFM3 RoPE")?;
        let projections = self.qkv.forward(x)?;
        let q = rope.apply(
            &projections[0]
                .reshape(&[1, time, self.heads, self.head_dim])?
                .rms_norm(&self.q_norm, self.eps)?
                .transpose(&[0, 2, 1, 3])?,
            offset,
        )?;
        let k = rope.apply(
            &projections[1]
                .reshape(&[1, time, self.kv_heads, self.head_dim])?
                .rms_norm(&self.k_norm, self.eps)?
                .transpose(&[0, 2, 1, 3])?,
            offset,
        )?;
        let v = projections[2]
            .reshape(&[1, time, self.kv_heads, self.head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let previous = self.keys.as_ref().map_or(0, |keys| keys.shape()[2]);
        let scale = (self.head_dim as f32).powf(-0.5);
        let mut attended = Vec::with_capacity(time as usize);
        for row in 0..time {
            let k_row = k.slice(2, row, row + 1)?;
            let v_row = v.slice(2, row, row + 1)?;
            // Keys T1 holds before this row: the trimmed cache plus earlier rows.
            let held = previous + row;
            let start = self.window.map_or(0, |w| (held - (w - 1)).max(0));
            let past = |cache: &Option<Array>, rows: &Array| -> Result<Option<Array>> {
                let mut parts = Vec::new();
                if let Some(cache) = cache
                    && start < previous
                {
                    parts.push(cache.slice(2, start, previous)?);
                }
                let first = (start - previous).max(0);
                if first < row {
                    parts.push(rows.slice(2, first, row)?);
                }
                Ok(match parts.len() {
                    0 => None,
                    1 => parts.pop(),
                    _ => Some(Array::concatenate(&parts.iter().collect::<Vec<_>>(), 2)?),
                })
            };
            let keys = match past(&self.keys, &k)? {
                Some(past) => Array::concatenate(&[&past, &k_row], 2)?,
                None => k_row,
            };
            let values = match past(&self.values, &v)? {
                Some(past) => Array::concatenate(&[&past, &v_row], 2)?,
                None => v_row,
            };
            let q_row = q.slice(2, row, row + 1)?;
            attended.push(if let Some(window) = self.window {
                let mask =
                    crate::lfm3::sliding_mask(1, keys.shape()[2], window)?.astype(q.dtype())?;
                Array::sdpa_mask(&q_row, &keys, &values, scale, &mask)?
            } else {
                Array::sdpa(&q_row, &keys, &values, scale, false)?
            });
        }
        let attended = Array::concatenate(&attended.iter().collect::<Vec<_>>(), 2)?;
        let gate = self
            .gate
            .as_ref()
            .context("verification requires the LFM3 attention gate")?
            .forward(x)?;
        ensure!(
            attended.dtype() == Dtype::Float16 && gate.dtype() == Dtype::Float16,
            "LFM3 verification requires F16 attention"
        );
        let out = crate::lfm3_kernels::attention_gate(&attended, &gate)?;
        let keys = match &self.keys {
            Some(cache) => Array::concatenate(&[cache, &k], 2)?,
            None => k,
        };
        let values = match &self.values {
            Some(cache) => Array::concatenate(&[cache, &v], 2)?,
            None => v,
        };
        Ok((
            self.output.forward(&out)?,
            PendingLayer::Attention {
                keys,
                values,
                previous,
            },
        ))
    }

    fn reset(&mut self) {
        self.keys = None;
        self.values = None;
    }
}

/// LFM3-PLD: cache candidates of one verified layer, committed for a prefix.
pub(crate) enum PendingLayer {
    /// Untrimmed cache plus every verified row.
    Attention {
        keys: Array,
        values: Array,
        previous: i32,
    },
    /// Convolution state after each verified row.
    Conv(Vec<Array>),
}

/// LFM3-PLD: result of `Lfm2::verify_rows`; nothing changes until committed.
pub struct Lfm3Verification {
    rows: i32,
    offset: i32,
    layers: Vec<PendingLayer>,
}

struct ShortConv {
    input: Projection,
    output: Projection,
    weight: Array,
    bias: Option<Array>,
    state: Option<Array>,
    dim: i32,
    length: i32,
    lfm3: bool,
}

impl ShortConv {
    fn load(checkpoint: &Checkpoint, prefix: &str, config: &Config) -> Result<Self> {
        let dim = config.hidden_size;
        let length = config.conv_length;
        let mut weight = half_weight(checkpoint, &format!("{prefix}.conv.weight"), None)?;
        ensure!(
            weight.shape().len() == 3,
            "{prefix}: convolution weight must have rank 3"
        );
        // Hugging Face stores [channels, 1, kernel]; MLX uses [channels, kernel, 1].
        if weight.shape()[2] > weight.shape()[1] {
            weight = weight.transpose(&[0, 2, 1])?;
        }
        ensure!(
            weight.shape() == [dim, length, 1],
            "{prefix}: invalid convolution weight shape"
        );
        Ok(Self {
            input: Projection::load(
                checkpoint,
                &format!("{prefix}.in_proj"),
                dim,
                3 * dim,
                config.conv_bias,
            )?,
            output: Projection::load(
                checkpoint,
                &format!("{prefix}.out_proj"),
                dim,
                dim,
                config.conv_bias,
            )?,
            weight,
            bias: if config.conv_bias {
                Some(half_weight(
                    checkpoint,
                    &format!("{prefix}.conv.bias"),
                    Some(&[dim]),
                )?)
            } else {
                None
            },
            state: None,
            dim,
            length,
            lfm3: config.model_type == "lfm3",
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let time = *x
            .shape()
            .get(1)
            .context("LFM2 convolution input must have time")?;
        ensure!(time > 0, "empty LFM2 convolution input");
        let bcx = self.input.forward(x)?;
        let state = match &self.state {
            Some(state) => state.try_clone()?,
            None => Array::zeros_dtype(&[1, self.length - 1, self.dim], bcx.dtype())?,
        };
        if self.lfm3
            && time == 1
            && self.length == 7
            && self.bias.is_none()
            && bcx.dtype() == Dtype::Float16
        {
            let (mixed, next_state) =
                crate::lfm3_kernels::short_conv_step(&bcx, &state, &self.weight)?;
            self.state = Some(next_state);
            return self.output.forward(&mixed);
        }
        let b = bcx.slice(2, 0, self.dim)?;
        let c = bcx.slice(2, self.dim, 2 * self.dim)?;
        let projected_x = bcx.slice(2, 2 * self.dim, 3 * self.dim)?;
        let bx = b.mul(&projected_x)?;
        let bx = Array::concatenate(&[&state, &bx], 1)?;
        let mut conv = bx.conv1d(&self.weight, self.dim)?;
        if let Some(bias) = &self.bias {
            conv = conv.add(bias)?;
        }
        self.state = Some(
            bx.slice(
                1,
                time,
                time.checked_add(self.length - 1)
                    .context("convolution state offset overflow")?,
            )?,
        );
        self.output.forward(&c.mul(&conv)?)
    }
}

impl ShortConv {
    /// LFM3-PLD: chain the fused T1 convolution step over each row.
    fn verify(&self, x: &Array) -> Result<(Array, PendingLayer)> {
        let time = *x
            .shape()
            .get(1)
            .context("LFM3 convolution input must have time")?;
        ensure!(
            self.lfm3 && self.length == 7 && self.bias.is_none() && time > 0,
            "verification requires the fused LFM3 convolution"
        );
        let bcx = self.input.forward(x)?;
        ensure!(
            bcx.dtype() == Dtype::Float16,
            "LFM3 verification requires F16 convolution"
        );
        let mut state = match &self.state {
            Some(state) => state.try_clone()?,
            None => Array::zeros_dtype(&[1, self.length - 1, self.dim], bcx.dtype())?,
        };
        let mut mixed = Vec::with_capacity(time as usize);
        let mut states = Vec::with_capacity(time as usize);
        for row in 0..time {
            let (value, next) = crate::lfm3_kernels::short_conv_step(
                &bcx.slice(1, row, row + 1)?,
                &state,
                &self.weight,
            )?;
            mixed.push(value);
            states.push(next.try_clone()?);
            state = next;
        }
        let mixed = Array::concatenate(&mixed.iter().collect::<Vec<_>>(), 1)?;
        Ok((self.output.forward(&mixed)?, PendingLayer::Conv(states)))
    }
}

// Stored once per layer; boxing only trades a few KB for extra allocations.
#[allow(clippy::large_enum_variant)]
enum Operator {
    Attention(Attention),
    Conv(ShortConv),
}

struct Layer {
    operator_norm: Array,
    ffn_norm: Array,
    operator: Operator,
    feed_forward: FeedForward,
    eps: f32,
    post_operator_norm: Option<Array>,
    post_ffn_norm: Option<Array>,
}

#[allow(clippy::large_enum_variant)]
enum FeedForward {
    Dense {
        inputs: ProjectionBundle<2>,
        w2: Projection,
    },
    Moe {
        gate: Projection,
        expert_bias: Option<Array>,
        experts: Exl3SwitchGlu,
        top_k: usize,
        normalize: bool,
        epsilon: Array,
        scale: Option<Array>,
        dim: i32,
    },
}

impl FeedForward {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Dense { inputs, w2 } => {
                let projected = inputs.forward(x)?;
                w2.forward(&projected[0].swiglu(&projected[1])?)
            }
            Self::Moe {
                gate,
                expert_bias,
                experts,
                top_k,
                normalize,
                epsilon,
                scale,
                dim,
            } => {
                let time = *x.shape().get(1).context("LFM2 MoE input must have time")?;
                let x = x.reshape(&[time, *dim])?;
                let probabilities = gate.forward(&x)?.sigmoid()?;
                let (selected, mut scores) = match expert_bias {
                    Some(bias) => router::topk_biased(&probabilities, bias, *top_k)?,
                    None => router::topk(&probabilities, *top_k, false)?,
                };
                if *normalize {
                    scores = scores.div(&scores.sum(-1, true)?.add(epsilon)?)?;
                }
                if let Some(scale) = scale {
                    scores = scores.mul(scale)?;
                }
                #[cfg(test)]
                prefill_tests::record_routes(&selected, &scores)?;
                experts
                    .forward(&x, &selected, &scores)?
                    .reshape(&[1, time, *dim])
            }
        }
    }
}

impl FeedForward {
    fn forward_prefill_rows(
        &self,
        values: &[Array],
        stats: &mut PrefillStats,
    ) -> Result<Vec<Array>> {
        let Self::Moe {
            gate,
            expert_bias,
            experts,
            top_k,
            normalize,
            epsilon,
            scale,
            dim,
        } = self
        else {
            return values.iter().map(|value| self.forward(value)).collect();
        };
        stats.router_rows += values.len();
        if values.len() == 1 {
            return values.iter().map(|value| self.forward(value)).collect();
        }
        ensure!(
            (2..=8).contains(&values.len()),
            "LFM2 exact MoE block must have 2..=8 rows"
        );
        let rows = i32::try_from(values.len())?;
        // Preserve the gate's GEMV rounding; only the independent FFN rows are grouped.
        let gates = values
            .iter()
            .map(|value| gate.forward(&value.reshape(&[1, *dim])?))
            .collect::<Result<Vec<_>>>()?;
        let probabilities = Array::concatenate(&gates.iter().collect::<Vec<_>>(), 0)?.sigmoid()?;
        let (selected, mut scores) = match expert_bias {
            Some(bias) => router::topk_biased(&probabilities, bias, *top_k)?,
            None => router::topk(&probabilities, *top_k, false)?,
        };
        if *normalize {
            scores = scores.div(&scores.sum(-1, true)?.add(epsilon)?)?;
        }
        if let Some(scale) = scale {
            scores = scores.mul(scale)?;
        }
        #[cfg(test)]
        prefill_tests::record_routes(&selected, &scores)?;
        let batch =
            Array::concatenate(&values.iter().collect::<Vec<_>>(), 1)?.reshape(&[rows, *dim])?;
        let output = experts.forward(&batch, &selected, &scores)?;
        stats.grouped_moe_calls += 1;
        (0..rows)
            .map(|row| output.slice(0, row, row + 1)?.reshape(&[1, 1, *dim]))
            .collect()
    }
}

impl Layer {
    fn load(
        checkpoint: &Checkpoint,
        index: usize,
        config: &Config,
        attention: bool,
        ff_dim: i32,
        theta: f32,
        root: &str,
    ) -> Result<Self> {
        let prefix = format!("{root}.layers.{index}");
        let dim = config.hidden_size;
        let sandwich = config.model_type == "lfm3" && config.use_sandwich_norm;
        Ok(Self {
            operator_norm: half_weight(
                checkpoint,
                &format!("{prefix}.operator_norm.weight"),
                Some(&[dim]),
            )?,
            ffn_norm: half_weight(
                checkpoint,
                &format!("{prefix}.ffn_norm.weight"),
                Some(&[dim]),
            )?,
            operator: if attention {
                Operator::Attention(Attention::load(
                    checkpoint,
                    &format!("{prefix}.self_attn"),
                    config,
                    theta,
                    config
                        .layer_types
                        .as_ref()
                        .map_or("full_attention", |types| types[index].as_str()),
                )?)
            } else {
                Operator::Conv(ShortConv::load(
                    checkpoint,
                    &format!("{prefix}.conv"),
                    config,
                )?)
            },
            feed_forward: if config.model_type == "lfm2_moe" && index >= config.num_dense_layers {
                let ffn = format!("{prefix}.feed_forward");
                FeedForward::Moe {
                    gate: Projection::load(
                        checkpoint,
                        &format!("{ffn}.gate"),
                        dim,
                        config.num_experts,
                        false,
                    )?,
                    expert_bias: config
                        .use_expert_bias
                        .then(|| {
                            half_weight(
                                checkpoint,
                                &format!("{ffn}.expert_bias"),
                                Some(&[config.num_experts]),
                            )
                        })
                        .transpose()?,
                    experts: Exl3SwitchGlu::from_lfm_checkpoint(
                        checkpoint,
                        &format!("{ffn}.experts"),
                        config.num_experts,
                        config.num_experts_per_tok as i32,
                    )?,
                    top_k: config.num_experts_per_tok,
                    normalize: config.norm_topk_prob,
                    epsilon: Array::from_f16_bits(&[f16::from_f32(1e-6).to_bits()], &[])?,
                    scale: (config.routed_scaling_factor != 1.0)
                        .then(|| {
                            Array::from_f16_bits(
                                &[f16::from_f32(config.routed_scaling_factor).to_bits()],
                                &[],
                            )
                        })
                        .transpose()?,
                    dim,
                }
            } else {
                let inputs = [
                    Projection::load(
                        checkpoint,
                        &format!("{prefix}.feed_forward.w1"),
                        dim,
                        ff_dim,
                        false,
                    )?,
                    Projection::load(
                        checkpoint,
                        &format!("{prefix}.feed_forward.w3"),
                        dim,
                        ff_dim,
                        false,
                    )?,
                ];
                let qualified = crate::lfm3::m5_text8(
                    checkpoint.model_type == "lfm3",
                    checkpoint.bits,
                    array::is_m5_gpu()?,
                );
                let grouped = crate::linear::lfm3_prefill_options(qualified)[1];
                FeedForward::Dense {
                    inputs: if grouped {
                        ProjectionBundle::new(inputs)?
                    } else {
                        ProjectionBundle::Separate(Box::new(inputs))
                    },
                    w2: Projection::load(
                        checkpoint,
                        &format!("{prefix}.feed_forward.w2"),
                        ff_dim,
                        dim,
                        false,
                    )?,
                }
            },
            eps: config.norm_eps,
            post_operator_norm: sandwich
                .then(|| {
                    half_weight(
                        checkpoint,
                        &format!("{prefix}.post_operator_norm.weight"),
                        Some(&[dim]),
                    )
                })
                .transpose()?,
            post_ffn_norm: sandwich
                .then(|| {
                    half_weight(
                        checkpoint,
                        &format!("{prefix}.post_ffn_norm.weight"),
                        Some(&[dim]),
                    )
                })
                .transpose()?,
        })
    }

    fn forward(&mut self, x: &Array, offset: i32) -> Result<Array> {
        let normalized = x.rms_norm(&self.operator_norm, self.eps)?;
        let mut mixed = match &mut self.operator {
            Operator::Attention(attention) => attention.forward(&normalized, offset)?,
            Operator::Conv(conv) => conv.forward(&normalized)?,
        };
        if let Some(norm) = &self.post_operator_norm {
            mixed = mixed.rms_norm(norm, self.eps)?;
        }
        let h = x.add(&mixed)?;
        let normalized = h.rms_norm(&self.ffn_norm, self.eps)?;
        let mut output = self.feed_forward.forward(&normalized)?;
        if let Some(norm) = &self.post_ffn_norm {
            output = output.rms_norm(norm, self.eps)?;
        }
        h.add(&output)
    }

    /// LFM3-PLD: `forward` for several T1 rows without touching the caches.
    fn verify(&self, x: &Array, offset: i32) -> Result<(Array, PendingLayer)> {
        ensure!(
            matches!(self.feed_forward, FeedForward::Dense { .. }),
            "verification requires dense LFM3 feed-forward layers"
        );
        let normalized = x.rms_norm(&self.operator_norm, self.eps)?;
        let (mut mixed, pending) = match &self.operator {
            Operator::Attention(attention) => attention.verify(&normalized, offset)?,
            Operator::Conv(conv) => conv.verify(&normalized)?,
        };
        if let Some(norm) = &self.post_operator_norm {
            mixed = mixed.rms_norm(norm, self.eps)?;
        }
        let h = x.add(&mixed)?;
        let normalized = h.rms_norm(&self.ffn_norm, self.eps)?;
        let mut output = self.feed_forward.forward(&normalized)?;
        if let Some(norm) = &self.post_ffn_norm {
            output = output.rms_norm(norm, self.eps)?;
        }
        Ok((h.add(&output)?, pending))
    }
}

/// Dispatch evidence for the explicitly requested experimental MoE prefill.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PrefillStats {
    pub tokens: usize,
    pub blocks: usize,
    pub grouped_moe_calls: usize,
    pub router_rows: usize,
    pub max_block: usize,
}

/// Largest `top_k` the L1 sampling head certifies.
pub const CERT_SAMPLE_MAX_K: usize = crate::lfm3_cert_head::SAMPLE_MAX_K;

/// Screening copy of the LFM3-CERT-HEAD.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CertScreenKind {
    /// One int8 level per weight, one scale per row (default).
    Int8,
    /// L2: one int4 level per weight, one F16 scale per 64 weights.
    Int4,
}

/// One certified greedy step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CertifiedStep {
    pub id: u32,
    pub certified: bool,
    /// Check oracle only: the screening ID differed from the full head.
    pub mismatch: bool,
}

/// One L1 sampling step.
#[derive(Clone, Debug, PartialEq)]
pub struct SampledStep {
    /// Log probabilities for `Sampler::distribution`.
    pub values: Vec<f32>,
    pub certified: bool,
    /// Check oracle only: the full-head row of a certified step.
    pub reference: Option<Vec<f32>>,
}

/// LFM3-EMBED-LAZY: the input embedding table, resident in F16 or read
/// row by row from the checkpoint (dense separate head only).
enum Embeddings {
    Resident(Array),
    Lazy {
        info: Box<checkpoint::TensorInfo>,
        dtype: Dtype,
        rows: u32,
        dim: i32,
    },
}

/// Absolute file offset of embedding row `id`, or None outside the table.
fn lazy_row_offset(start: u64, row_bytes: u64, id: u32, rows: u32) -> Option<u64> {
    if id >= rows {
        return None;
    }
    u64::from(id).checked_mul(row_bytes)?.checked_add(start)
}

impl Embeddings {
    fn dim(&self) -> i32 {
        match self {
            Self::Resident(table) => table.shape()[1],
            Self::Lazy { dim, .. } => *dim,
        }
    }

    /// The resident table (tied heads, which never use the lazy mode).
    fn table(&self) -> Result<&Array> {
        match self {
            Self::Resident(table) => Ok(table),
            Self::Lazy { .. } => anyhow::bail!("LFM lazy embeddings have no resident table"),
        }
    }

    /// F16 rows `[1, tokens, dim]`; same values as `table.take(ids, 0)`.
    fn take(&self, tokens: &[u32]) -> Result<Array> {
        let time = i32::try_from(tokens.len())?;
        match self {
            Self::Resident(table) => {
                let ids = Array::from_i32(
                    &tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                    &[1, time],
                )?;
                table.take(&ids, 0)
            }
            Self::Lazy {
                info,
                dtype,
                rows,
                dim,
            } => {
                use std::os::unix::fs::FileExt;
                let size = match dtype {
                    Dtype::Float32 => 4usize,
                    _ => 2,
                };
                let row_bytes = usize::try_from(*dim)? * size;
                let (file, start) = info.source()?;
                let mut bytes = vec![0u8; row_bytes * tokens.len()];
                for (&id, row) in tokens.iter().zip(bytes.chunks_exact_mut(row_bytes)) {
                    let offset = lazy_row_offset(start, row_bytes as u64, id, *rows)
                        .context("LFM embedding row outside the table")?;
                    file.read_exact_at(row, offset)?;
                }
                Array::from_bytes(&bytes, &[1, time, *dim], *dtype)?.astype(Dtype::Float16)
            }
        }
    }
}

/// LFM3-EMBED-LAZY gate: `MLXL3_LFM3_EMBED_LAZY=1` with a separate head.
pub(crate) fn lazy_embeddings_enabled(separate_head: bool, flag: Option<&[u8]>) -> bool {
    separate_head && flag == Some(b"1")
}

pub struct Lfm2 {
    embeddings: Embeddings,
    /// Checkpoint record of the embedding table, kept for mode switches.
    /// Name and record, boxed to keep the model enum small.
    embedding_source: Box<(String, checkpoint::TensorInfo)>,
    norm: Array,
    head: Option<Projection>,
    /// LFM3-CERT-HEAD screening copy of `head`, built on request only.
    certified_head: Option<crate::lfm3_cert_head::Screen>,
    layers: Vec<Layer>,
    vocab_size: i32,
    context_limit: i32,
    moe_exact_prefill: bool,
    eps: f32,
    offset: i32,
}

impl Lfm2 {
    pub fn load(path: &Path) -> Result<Self> {
        let checkpoint = checkpoint::inspect(path)?;
        Self::from_checkpoint(&checkpoint)
    }

    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        let config: Config =
            serde_json::from_reader(File::open(checkpoint.path.join("config.json"))?)?;
        Self::from_config(checkpoint, config, "model")
    }

    pub(crate) fn from_config(checkpoint: &Checkpoint, config: Config, root: &str) -> Result<Self> {
        ensure!(
            matches!(config.model_type.as_str(), "lfm2" | "lfm2_moe" | "lfm3"),
            "expected LFM2, got {}",
            config.model_type
        );
        ensure!(
            config.hidden_size > 0
                && config.hidden_size <= i32::MAX / 3
                && config.vocab_size > 0
                && config.max_position_embeddings > 0
                && config.num_hidden_layers > 0
                && config.num_attention_heads > 0,
            "invalid LFM2 dimensions"
        );
        ensure!(
            config.hidden_size % config.num_attention_heads == 0,
            "LFM2 head dimensions do not divide hidden size"
        );
        ensure!(
            (config.hidden_size / config.num_attention_heads) % 2 == 0,
            "LFM2 rotary head dimensions must be even"
        );
        let kv_heads = config
            .num_key_value_heads
            .unwrap_or(config.num_attention_heads);
        ensure!(
            kv_heads > 0 && config.num_attention_heads % kv_heads == 0,
            "invalid LFM2 KV head count"
        );
        ensure!(config.conv_length > 0, "invalid LFM convolution length");
        if config.model_type == "lfm3" {
            let head_dim = config
                .head_dim
                .unwrap_or(config.hidden_size / config.num_attention_heads);
            ensure!(
                head_dim > 0
                    && head_dim % 2 == 0
                    && head_dim.checked_mul(config.num_attention_heads).is_some(),
                "invalid LFM3 head dimension"
            );
            ensure!(
                config.sliding_window.is_some_and(|w| w > 0),
                "invalid LFM3 sliding window"
            );
            let types = config
                .layer_types
                .as_ref()
                .context("missing LFM3 layer types")?;
            ensure!(
                types.len() == config.num_hidden_layers
                    && types.iter().all(|t| matches!(
                        t.as_str(),
                        "conv" | "full_attention" | "sliding_attention"
                    )),
                "invalid LFM3 layer types"
            );
            ensure!(
                config.full_attn_idxs.is_none(),
                "LFM3 uses layer_types, not full_attn_idxs"
            );
            let params = config
                .rope_parameters
                .as_ref()
                .context("missing LFM3 RoPE parameters")?;
            for kind in types.iter().filter(|t| t.as_str() != "conv") {
                crate::lfm3::Rope::from_config(params, kind, head_dim)?;
            }
        }
        if config.model_type == "lfm2" {
            ensure!(
                config.block_dim == Some(config.hidden_size),
                "invalid LFM2 block dimensions"
            );
        } else if config.model_type == "lfm2_moe" {
            ensure!(
                config.moe_intermediate_size > 0
                    && config.num_experts > 0
                    && config.num_experts_per_tok > 0
                    && config.num_experts_per_tok <= config.num_experts as usize
                    && config.num_dense_layers <= config.num_hidden_layers
                    && config.routed_scaling_factor.is_finite()
                    && config.routed_scaling_factor > 0.,
                "invalid LFM2 MoE dimensions"
            );
        }
        ensure!(
            config.norm_eps.is_finite() && config.norm_eps > 0.,
            "invalid LFM2 norm epsilon"
        );
        let theta = config
            .rope_parameters
            .as_ref()
            .and_then(|v| v.get("rope_theta"))
            .and_then(serde_json::Value::as_f64)
            .map(|v| v as f32)
            .unwrap_or(config.rope_theta);
        ensure!(theta.is_finite() && theta > 0., "invalid LFM2 RoPE theta");
        let mut ff_dim = i64::from(
            config
                .block_ff_dim
                .or(config.intermediate_size)
                .context("missing LFM2 feed-forward size")?,
        );
        ensure!(ff_dim > 0, "invalid LFM2 feed-forward size");
        if config.block_auto_adjust_ff_dim {
            ff_dim = 2 * ff_dim / 3;
            if let Some(multiplier) = config.block_ffn_dim_multiplier {
                ensure!(
                    multiplier.is_finite()
                        && multiplier > 0.
                        && multiplier * (ff_dim as f64) <= i32::MAX as f64,
                    "invalid LFM2 feed-forward multiplier"
                );
                ff_dim = (multiplier * ff_dim as f64) as i64;
            }
            ensure!(
                config.block_multiple_of.is_some_and(|value| value > 0),
                "invalid LFM2 feed-forward multiple"
            );
            let multiple = i64::from(config.block_multiple_of.expect("checked above"));
            ff_dim = multiple * ((ff_dim + multiple - 1) / multiple);
        }
        let ff_dim = i32::try_from(ff_dim).context("LFM2 feed-forward size overflow")?;
        ensure!(ff_dim > 0, "invalid adjusted LFM2 feed-forward size");
        let attention: BTreeSet<usize> = if let Some(indices) = &config.full_attn_idxs {
            indices.iter().copied().collect()
        } else {
            let types = config
                .layer_types
                .as_ref()
                .context("missing LFM2 layer types")?;
            ensure!(
                types.len() == config.num_hidden_layers,
                "LFM2 layer types count mismatch"
            );
            ensure!(
                types.iter().all(|kind| kind == "full_attention"
                    || kind == "conv"
                    || kind == "short_conv"
                    || (config.model_type == "lfm3" && kind == "sliding_attention")),
                "unsupported LFM2 layer type"
            );
            types
                .iter()
                .enumerate()
                .filter_map(|(i, kind)| {
                    (kind == "full_attention" || kind == "sliding_attention").then_some(i)
                })
                .collect()
        };
        ensure!(
            attention.iter().all(|&i| i < config.num_hidden_layers),
            "invalid LFM2 attention index"
        );
        let dim = config.hidden_size;
        let embedding_name = format!("{root}.embed_tokens.weight");
        let embedding_info = checkpoint
            .tensors
            .get(&embedding_name)
            .with_context(|| format!("missing tensor {embedding_name}"))?
            .clone();
        let norm = half_weight(
            checkpoint,
            &format!("{root}.embedding_norm.weight"),
            Some(&[dim]),
        )?;
        let head = if checkpoint.modules.iter().any(|name| name == "lm_head")
            || checkpoint.tensors.contains_key("lm_head.weight")
        {
            Some(Projection::load(
                checkpoint,
                "lm_head",
                dim,
                config.vocab_size,
                false,
            )?)
        } else {
            None
        };
        let lazy_flag = std::env::var_os("MLXL3_LFM3_EMBED_LAZY");
        let embeddings = load_embeddings(
            checkpoint,
            &embedding_name,
            &embedding_info,
            config.vocab_size,
            dim,
            lazy_embeddings_enabled(
                head.is_some(),
                lazy_flag.as_deref().map(std::ffi::OsStr::as_encoded_bytes),
            ),
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(Layer::load(
                checkpoint,
                index,
                &config,
                attention.contains(&index),
                ff_dim,
                theta,
                root,
            )?);
        }
        let moe_exact_prefill = exact_moe_prefill_config(&config, attention.len(), true)
            && exact_moe_projection_metadata(checkpoint);
        Ok(Self {
            embeddings,
            embedding_source: Box::new((embedding_name, embedding_info)),
            norm,
            head,
            certified_head: None,
            layers,
            vocab_size: config.vocab_size,
            context_limit: config.max_position_embeddings,
            moe_exact_prefill,
            eps: config.norm_eps,
            offset: 0,
        })
    }

    pub fn supports_batched_prefill(&self) -> bool {
        self.layers
            .iter()
            .all(|layer| matches!(&layer.feed_forward, FeedForward::Dense { .. }))
    }

    /// Eligibility for the registered V2 K4/MCG geometry.
    pub fn supports_moe_exact_prefill(&self) -> bool {
        self.moe_exact_prefill
    }

    /// One selection contract for bridge chunking and the actual dispatch.
    pub fn uses_moe_exact_prefill(&self) -> bool {
        let flag = std::env::var_os("MLXL3_LFM_MOE_PREFILL");
        crate::contracts::lfm_moe_prefill_enabled(
            self.supports_moe_exact_prefill(),
            flag.as_deref().map(std::ffi::OsStr::as_encoded_bytes),
        )
    }

    /// Consumes one token for decode, or at least 24 tokens for dense TensorOps prefill.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Array> {
        self.forward_embedded(tokens, None)
    }

    pub(crate) fn embed_tokens(&self, tokens: &[u32]) -> Result<Array> {
        ensure!(
            !tokens.is_empty() && tokens.iter().all(|&t| t < self.vocab_size as u32),
            "invalid LFM tokens"
        );
        self.embeddings.take(tokens)
    }

    /// LFM3-EMBED-LAZY: switches the embedding table between resident F16
    /// and row reads. Returns false (unchanged) for a tied head.
    pub fn set_lazy_embeddings(&mut self, checkpoint: &Checkpoint, lazy: bool) -> Result<bool> {
        if lazy && self.head.is_none() {
            return Ok(false);
        }
        let dim = self.embeddings.dim();
        self.embeddings = load_embeddings(
            checkpoint,
            &self.embedding_source.0,
            &self.embedding_source.1,
            self.vocab_size,
            dim,
            lazy,
        )?;
        Ok(true)
    }

    pub fn lazy_embeddings(&self) -> bool {
        matches!(self.embeddings, Embeddings::Lazy { .. })
    }

    pub(crate) fn forward_embedded(
        &mut self,
        tokens: &[u32],
        embedded: Option<Array>,
    ) -> Result<Array> {
        self.forward_with(tokens, embedded, |model, hidden, time| {
            match &model.head {
                Some(head) => head.forward(hidden)?,
                None => hidden.matmul(&model.embeddings.table()?.transpose(&[1, 0])?)?,
            }
            .slice(1, time - 1, time)
        })
    }

    /// LFM3-CERT-HEAD: builds the int8 screening copy of a dense F16 head.
    /// Returns false (and keeps the normal head) for unsupported heads.
    pub fn enable_certified_head(&mut self) -> Result<bool> {
        self.enable_certified_head_kind(CertScreenKind::Int8)
    }

    /// LFM3-CERT-HEAD with the chosen screening copy (`Int4`: L2, the greedy
    /// default; `MLXL3_LFM3_CERT_SCREEN=int8` selects `Int8`). A copy of the other
    /// kind is replaced.
    pub fn enable_certified_head_kind(&mut self, kind: CertScreenKind) -> Result<bool> {
        let int4 = kind == CertScreenKind::Int4;
        if self
            .certified_head
            .as_ref()
            .is_some_and(|screen| screen.is_int4() == int4)
        {
            return Ok(true);
        }
        let Some(weight) = self.head.as_ref().and_then(Projection::dense_weight) else {
            return Ok(false);
        };
        if weight.dtype() != Dtype::Float16
            || !crate::lfm3_cert_head::supported(weight.shape()[0], weight.shape()[1])
            || weight.shape()[0] != self.vocab_size
        {
            return Ok(false);
        }
        self.certified_head = None;
        self.certified_head = Some(if int4 {
            crate::lfm3_cert_head::Screen::Int4(crate::lfm3_cert_head::Int4Head::build(weight)?)
        } else {
            crate::lfm3_cert_head::Screen::Int8(crate::lfm3_cert_head::CertifiedHead::build(
                weight,
            )?)
        });
        Ok(true)
    }

    /// LFM3-CERT-HEAD: one T1 step returning the greedy ID of `select_token`
    /// (`chat_greedy_ids` of the step logits) and whether it was certified
    /// without the full head. Uncertified steps run the normal head, so the
    /// caches and the returned ID always match `forward_embedded`. With `check`, certified steps also run
    /// the full head (debug oracle `MLXL3_LFM3_CERT_CHECK=1`), report a
    /// mismatch and return the full-head ID.
    pub(crate) fn forward_certified_checked(
        &mut self,
        tokens: &[u32],
        embedded: Option<Array>,
        check: bool,
    ) -> Result<CertifiedStep> {
        ensure!(
            tokens.len() == 1,
            "certified LFM2 head steps take one token"
        );
        ensure!(
            self.certified_head.is_some(),
            "certified LFM2 head is not enabled"
        );
        self.forward_with(tokens, embedded, |model, hidden, _| {
            let head = model
                .head
                .as_ref()
                .context("certified head requires a head")?;
            let full = || -> Result<u32> {
                head.forward(hidden)?
                    .slice(1, 0, 1)?
                    .chat_greedy_ids()?
                    .into_iter()
                    .next()
                    .context("model returned no logits")
            };
            if let (Some(certified), Some(weight)) = (&model.certified_head, head.dense_weight())
                && hidden.dtype() == Dtype::Float16
            {
                let x = hidden.reshape(&[1, hidden.shape()[hidden.shape().len() - 1]])?;
                let (id, certified) = certified.greedy(weight, &x)?;
                if certified {
                    ensure!(id < model.vocab_size as u32, "certified head ID is invalid");
                    if check {
                        let reference = full()?;
                        return Ok(CertifiedStep {
                            id: reference,
                            certified: true,
                            mismatch: reference != id,
                        });
                    }
                    return Ok(CertifiedStep {
                        id,
                        certified: true,
                        mismatch: false,
                    });
                }
            }
            Ok(CertifiedStep {
                id: full()?,
                certified: false,
                mismatch: false,
            })
        })
    }

    /// L1 (default for sampled steps, `MLXL3_LFM3_CERT_SAMPLE=0` disables it):
    /// one T1 step returning the log probabilities `Sampler::distribution`
    /// receives. Certified steps keep
    /// `-inf` outside the rows the int8 screen recomputed exactly, which
    /// leaves the kept set and its values unchanged (see
    /// `lfm3_cert_head::certified_values`); other steps return the full head
    /// row of `select_token`. With `check`, certified steps also return that
    /// full row.
    pub(crate) fn forward_sampled_embedded(
        &mut self,
        tokens: &[u32],
        embedded: Option<Array>,
        sampler: &crate::sampling::Sampler,
        history: &crate::sampling::History,
        check: bool,
    ) -> Result<SampledStep> {
        ensure!(
            tokens.len() == 1,
            "certified LFM2 head steps take one token"
        );
        ensure!(
            matches!(
                self.certified_head,
                Some(crate::lfm3_cert_head::Screen::Int8(_))
            ),
            "certified LFM2 sampling head is not enabled"
        );
        self.forward_with(tokens, embedded, |model, hidden, _| {
            let head = model
                .head
                .as_ref()
                .context("certified head requires a head")?;
            let full = || -> Result<Vec<f32>> {
                head.forward(hidden)?.slice(1, 0, 1)?.log_probs()?.to_f32()
            };
            let mut values = None;
            if let (Some(crate::lfm3_cert_head::Screen::Int8(certified)), Some(weight), Some(k)) = (
                &model.certified_head,
                head.dense_weight(),
                sampler.certified_k(crate::lfm3_cert_head::SAMPLE_MAX_K),
            ) && hidden.dtype() == Dtype::Float16
            {
                let x = hidden.reshape(&[1, hidden.shape()[hidden.shape().len() - 1]])?;
                let rows = certified.sample(weight, &x, k)?;
                values = crate::lfm3_cert_head::certified_values(
                    &rows,
                    usize::try_from(model.vocab_size)?,
                    sampler,
                    history,
                );
            }
            Ok(match values {
                Some(values) => SampledStep {
                    values,
                    certified: true,
                    reference: if check { Some(full()?) } else { None },
                },
                None => SampledStep {
                    values: full()?,
                    certified: false,
                    reference: None,
                },
            })
        })
    }

    fn forward_with<T>(
        &mut self,
        tokens: &[u32],
        embedded: Option<Array>,
        finish: impl FnOnce(&Self, &Array, i32) -> Result<T>,
    ) -> Result<T> {
        ensure!(
            tokens.len() == 1 || tokens.len() >= 24,
            "LFM2 batches must contain one token or at least 24 tokens"
        );
        ensure!(
            tokens.len() == 1 || self.supports_batched_prefill(),
            "batched LFM2 prefill currently requires dense feed-forward layers"
        );
        ensure!(
            tokens.iter().all(|&token| token < self.vocab_size as u32),
            "LFM2 token is outside vocabulary"
        );
        let time = i32::try_from(tokens.len()).context("LFM2 batch is too large")?;
        ensure!(
            self.offset <= self.context_limit - time,
            "LFM2 context is full ({} tokens); reset the model cache before continuing",
            self.context_limit
        );
        let result = (|| -> Result<T> {
            let next_offset = self
                .offset
                .checked_add(time)
                .context("LFM2 position overflow")?;
            let mut hidden = match embedded {
                Some(hidden) => hidden,
                None => self.embed_tokens(tokens)?,
            };
            for layer in &mut self.layers {
                hidden = layer.forward(&hidden, self.offset)?;
            }
            let hidden = hidden.rms_norm(&self.norm, self.eps)?;
            let value = finish(self, &hidden, time)?;
            self.offset = next_offset;
            Ok(value)
        })();
        match result {
            Ok(value) => Ok(value),
            Err(error) => {
                self.reset();
                Err(error.context("LFM2 forward failed; its cache was reset"))
            }
        }
    }

    /// Experimental MoE prefill: recurrent operators and the final head keep M1
    /// shapes. No caller enables this path before physical parity qualification.
    pub fn prefill_moe_exact(&mut self, tokens: &[u32]) -> Result<(Array, PrefillStats)> {
        ensure!(
            self.supports_moe_exact_prefill(),
            "experimental LFM2 prefill requires the qualified V2 K4/MCG geometry"
        );
        let end = crate::contracts::lfm_prefill_end(tokens.len(), self.offset, self.context_limit)
            .context("empty, overflowing or full LFM2 experimental prefill")?;
        ensure!(
            tokens.iter().all(|&token| token < self.vocab_size as u32),
            "LFM2 token is outside vocabulary"
        );
        let moe_layers = self
            .layers
            .iter()
            .filter(|layer| matches!(layer.feed_forward, FeedForward::Moe { .. }))
            .count();
        ensure!(
            moe_layers > 0,
            "experimental LFM2 prefill requires MoE layers"
        );
        tokens
            .len()
            .checked_mul(moe_layers)
            .context("LFM2 prefill dispatch count overflow")?;
        let result = (|| -> Result<_> {
            let mut stats = PrefillStats {
                tokens: tokens.len(),
                ..Default::default()
            };
            let mut last = None;
            for block in tokens.chunks(8) {
                let mut hidden = block
                    .iter()
                    .map(|&token| self.embeddings.take(&[token]))
                    .collect::<Result<Vec<_>>>()?;
                for layer in &mut self.layers {
                    let mut residuals = Vec::with_capacity(block.len());
                    let mut normalized = Vec::with_capacity(block.len());
                    for (row, value) in hidden.iter().enumerate() {
                        let input = value.rms_norm(&layer.operator_norm, layer.eps)?;
                        let position = self.offset + row as i32;
                        let mixed = match &mut layer.operator {
                            Operator::Attention(attention) => {
                                attention.forward(&input, position)?
                            }
                            Operator::Conv(conv) => conv.forward(&input)?,
                        };
                        let residual = value.add(&mixed)?;
                        normalized.push(residual.rms_norm(&layer.ffn_norm, layer.eps)?);
                        residuals.push(residual);
                    }
                    let outputs = layer
                        .feed_forward
                        .forward_prefill_rows(&normalized, &mut stats)?;
                    hidden = residuals
                        .iter()
                        .zip(outputs)
                        .map(|(residual, output)| residual.add(&output))
                        .collect::<Result<Vec<_>>>()?;
                }
                let value = hidden.pop().context("empty LFM2 prefill block")?;
                // Match bridge cache evaluation without forcing discarded FFN/head outputs.
                for (_, state) in self.state_arrays() {
                    state.eval()?;
                }
                last = Some(value);
                self.offset += block.len() as i32;
                stats.blocks += 1;
                stats.max_block = stats.max_block.max(block.len());
            }
            ensure!(self.offset == end, "LFM2 prefill position mismatch");
            let hidden = last
                .context("empty LFM2 prefill")?
                .rms_norm(&self.norm, self.eps)?;
            let logits = match &self.head {
                Some(head) => head.forward(&hidden)?,
                None => hidden.matmul(&self.embeddings.table()?.transpose(&[1, 0])?)?,
            }
            .slice(1, 0, 1)?;
            Ok((logits, stats))
        })();
        match result {
            Ok(output) => Ok(output),
            Err(error) => {
                self.reset();
                Err(error.context("LFM2 experimental prefill failed; its cache was reset"))
            }
        }
    }

    /// Read-only cache snapshots for numerical parity checks; no array is copied.
    /// LFM3-PLD: logits `[rows, vocab]` that `rows` successive T1 forwards of
    /// `tokens` would return, without changing the model. Commit a prefix with
    /// [`Self::commit_verified`] or drop the result to discard it.
    pub(crate) fn verify_rows(
        &self,
        tokens: &[u32],
        hidden: Array,
    ) -> Result<(Array, Lfm3Verification)> {
        let rows = i32::try_from(tokens.len()).context("too many verification rows")?;
        ensure!(
            (2..=LFM3_VERIFY_MAX_ROWS).contains(&rows),
            "LFM3 verification requires 2 to {LFM3_VERIFY_MAX_ROWS} rows"
        );
        ensure!(
            tokens.iter().all(|&token| token < self.vocab_size as u32),
            "LFM3 token is outside vocabulary"
        );
        ensure!(
            self.offset > 0 && self.offset <= self.context_limit - rows,
            "LFM3 verification needs a prefilled cache with room for {rows} tokens"
        );
        ensure!(
            hidden.shape() == [1, rows, self.embeddings.dim()],
            "invalid LFM3 verification embeddings"
        );
        let head = self
            .head
            .as_ref()
            .context("LFM3 verification requires a dense head")?;
        let mut hidden = hidden;
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            let (next, pending) = layer.verify(&hidden, self.offset)?;
            hidden = next;
            layers.push(pending);
        }
        let hidden = hidden
            .rms_norm(&self.norm, self.eps)?
            .reshape(&[rows, self.embeddings.dim()])?;
        let logits = head.forward_dense_rows_shared(&hidden)?;
        Ok((
            logits,
            Lfm3Verification {
                rows,
                offset: self.offset,
                layers,
            },
        ))
    }

    /// LFM3-PLD: keep the first `accepted` verified rows, exactly as if they
    /// had been forwarded one by one; later rows are discarded.
    pub(crate) fn commit_verified(
        &mut self,
        verification: Lfm3Verification,
        accepted: i32,
    ) -> Result<()> {
        ensure!(
            (1..=verification.rows).contains(&accepted)
                && verification.offset == self.offset
                && verification.layers.len() == self.layers.len(),
            "stale or invalid LFM3 verification commit"
        );
        let index = usize::try_from(accepted - 1)?;
        for (layer, pending) in self.layers.iter_mut().zip(verification.layers) {
            match (&mut layer.operator, pending) {
                (
                    Operator::Attention(attention),
                    PendingLayer::Attention {
                        keys,
                        values,
                        previous,
                    },
                ) => {
                    let end = previous + accepted;
                    let start = attention.window.map_or(0, |w| (end - (w - 1)).max(0));
                    attention.keys = Some(keys.slice(2, start, end)?);
                    attention.values = Some(values.slice(2, start, end)?);
                }
                (Operator::Conv(conv), PendingLayer::Conv(mut states)) => {
                    ensure!(states.len() > index, "missing LFM3 convolution state");
                    conv.state = Some(states.swap_remove(index));
                }
                _ => anyhow::bail!("LFM3 verification layer kinds do not match"),
            }
        }
        self.offset += accepted;
        Ok(())
    }

    pub fn state_arrays(&self) -> Vec<(String, &Array)> {
        let mut states = Vec::new();
        for (index, layer) in self.layers.iter().enumerate() {
            match &layer.operator {
                Operator::Attention(attention) => {
                    if let Some(keys) = &attention.keys {
                        states.push((format!("model.layers.{index}.keys"), keys));
                    }
                    if let Some(values) = &attention.values {
                        states.push((format!("model.layers.{index}.values"), values));
                    }
                }
                Operator::Conv(conv) => {
                    if let Some(state) = &conv.state {
                        states.push((format!("model.layers.{index}.conv_state"), state));
                    }
                }
            }
        }
        states
    }

    pub fn position(&self) -> i32 {
        self.offset
    }

    pub fn context_limit(&self) -> i32 {
        self.context_limit
    }

    /// Logical F16 context states for batch one. Views can retain larger backing
    /// allocations; temporary activations and allocator caches are not included.
    pub fn context_memory_profile(&self) -> Result<serde_json::Value> {
        let mut profile = crate::contracts::LfmContextMemory::default();
        for layer in &self.layers {
            profile = match &layer.operator {
                Operator::Attention(attention) => {
                    profile.attention(attention.kv_heads, attention.head_dim)
                }
                Operator::Conv(conv) => profile.convolution(conv.length, conv.dim),
            }
            .context("invalid or overflowing LFM2 context memory profile")?;
        }
        Ok(profile.into_json())
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        for layer in &mut self.layers {
            match &mut layer.operator {
                Operator::Attention(attention) => attention.reset(),
                Operator::Conv(conv) => conv.state = None,
            }
        }
    }
}

#[cfg(test)]
mod flash_hc_tests {
    use super::*;

    fn projection(weight: Array) -> Projection {
        Projection {
            logical_input: weight.shape()[1],
            logical_output: weight.shape()[0],
            weights: ProjectionWeights::Dense(weight),
            bias: None,
        }
    }

    fn check(p: &Projection, x: &Array) -> Result<Vec<f32>> {
        let expected = p.forward(x)?;
        let actual = p.forward_flash_hc(x)?;
        let values = actual.astype(Dtype::Float32)?.to_f32()?;
        ensure!(
            actual.shape() == expected.shape()
                && actual.dtype() == expected.dtype()
                && !values.is_empty()
                && values.iter().all(|v| v.is_finite()),
            "invalid mixed GEMV output"
        );
        ensure!(
            actual.to_bytes()? == expected.to_bytes()?,
            "mixed GEMV differs: {}→{}, input {:?}/{:?}",
            p.logical_input,
            p.logical_output,
            x.shape(),
            x.dtype()
        );
        Ok(values)
    }

    #[test]
    #[ignore = "physical Apple GPU; exact mixed HC GEMV and fallback contracts"]
    fn mixed_gemv_matches_mlx_cpu_and_fallbacks() -> Result<()> {
        let mut checked = 0;
        for (k, n) in [(10240, 320), (320, 10240), (10240, 4)] {
            let weights: Vec<f32> = (0..k * n)
                .map(|i| ((i * 7 % 15) - 7) as f32 / 16.)
                .collect();
            let p = projection(Array::from_f32(&weights, &[n, k])?.astype(Dtype::Float16)?);
            let values: Vec<f32> = (0..k).map(|i| ((i * 11 % 31) - 15) as f32 / 32.).collect();
            let input = Array::from_f32(&values, &[1, 1, k])?;
            let out = check(&p, &input)?;
            for (row, &actual) in out.iter().enumerate() {
                // These dyadic products and sums are exactly representable in FP32.
                let reference: f64 = weights[row * k as usize..(row + 1) * k as usize]
                    .iter()
                    .zip(&values)
                    .map(|(&w, &x)| f64::from(w) * f64::from(x))
                    .sum();
                ensure!(
                    actual == reference as f32,
                    "CPU dyadic oracle differs at {row}"
                );
            }
            checked += out.len();
            check(&p, &input.astype(Dtype::Float16)?)?;
            check(&p, &Array::zeros(&[1, k])?)?;
            check(&p, &input.reshape(&[1, k])?.broadcast_to(&[2, k])?)?;
            ensure!(
                p.forward_flash_hc(&Array::zeros(&[1, k - 1])?).is_err(),
                "wrong width accepted"
            );

            let arbitrary: Vec<f32> = (0..2 * k)
                .map(|i| {
                    let bits = (i as u32).wrapping_mul(747796405).wrapping_add(2891336453);
                    (bits as i32 as f32 / i32::MAX as f32) * 7.31
                })
                .collect();
            let strided = Array::from_f32(&arbitrary, &[k, 2])?
                .transpose(&[1, 0])?
                .slice(0, 0, 1)?;
            checked += check(&p, &strided)?.len();
            let mut biased = p;
            biased.bias =
                Some(Array::from_f32(&vec![0.25; n as usize], &[n])?.astype(Dtype::Float16)?);
            check(&biased, &input)?;
        }
        let p = projection(Array::from_f32(&vec![0.25; 96 * 8], &[8, 96])?.astype(Dtype::Float16)?);
        check(&p, &Array::from_f32(&vec![0.5; 96], &[1, 96])?)?;
        ensure!(checked == 21128, "coverage changed");
        println!(
            "mixed HC GEMV: {checked} FP32 values plus half/batch/bias/shape fallbacks and CPU oracle"
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU and Flash checkpoint; all real HC weights"]
    fn mixed_gemv_matches_every_flash_hc_weight() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut matrices = 0;
        let mut values = 0;
        for (name, tensor) in &c.tensors {
            if ![
                "input_mix_weight_down.weight",
                "input_mix_weight_up.weight",
                "block_inject_weight.weight",
            ]
            .iter()
            .any(|suffix| name.ends_with(suffix))
            {
                continue;
            }
            let k = tensor.shape[1] as i32;
            let n = tensor.shape[0] as i32;
            let p = Projection::load(
                &c,
                name.strip_suffix(".weight").expect("weight suffix"),
                k,
                n,
                false,
            )?;
            let input: Vec<f32> = (0..k)
                .map(|i| {
                    ((i as u32).wrapping_mul(1664525).wrapping_add(1013904223) as i32) as f32
                        / 1073741824.
                })
                .collect();
            values += check(&p, &Array::from_f32(&input, &[1, 1, k])?)?.len();
            matrices += 1;
        }
        ensure!(
            matrices == 298 && values > 1_000_000,
            "incomplete HC census: {matrices}/{values}"
        );
        println!("mixed HC GEMV: {matrices} real matrices, {values} bit-exact finite values");
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU and Flash checkpoint; microbenchmark, not model throughput"]
    fn mixed_gemv_micro() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        for (name, k, n) in [
            ("input_mix_weight_down", 10240, 320),
            ("input_mix_weight_up", 320, 10240),
            ("block_inject_weight", 10240, 4),
        ] {
            let p = Projection::load(
                &c,
                &format!("model.language_model.layers.0.attn_hyper_connection.{name}"),
                k,
                n,
                false,
            )?;
            let x = Array::from_f32(
                &(0..k).map(|i| (i as f32 * 0.173).sin()).collect::<Vec<_>>(),
                &[1, 1, k],
            )?;
            check(&p, &x)?;
            for _ in 0..8 {
                p.forward(&x)?.eval()?;
                p.forward_flash_hc(&x)?.eval()?;
            }
            for repeat in 0..5 {
                let mut seconds = [0.; 2];
                for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let start = std::time::Instant::now();
                    for _ in 0..32 {
                        if arm == 0 {
                            p.forward(&x)?.eval()?;
                        } else {
                            p.forward_flash_hc(&x)?.eval()?;
                        }
                    }
                    seconds[arm] = start.elapsed().as_secs_f64();
                }
                println!(
                    "HC_GEMV_MICRO {}",
                    serde_json::json!({"shape":[k,n],"repeat":repeat,"iterations":32,"seconds":seconds})
                );
            }
        }
        Ok(())
    }

    #[cfg(kani)]
    #[kani::proof]
    fn actual_mixed_hc_path_matches_mlx() {
        let x = f32::from(kani::any::<i8>());
        let w = f32::from(kani::any::<i8>());
        let weight = Array::from_f32(&[w], &[1, 1])
            .unwrap()
            .astype(Dtype::Float16)
            .unwrap()
            .broadcast_to(&[320, 10240])
            .unwrap();
        let input = Array::from_f32(&[x], &[1, 1])
            .unwrap()
            .broadcast_to(&[1, 10240])
            .unwrap();
        check(&projection(weight), &input).unwrap();
    }
}

#[cfg(test)]
#[path = "lfm2_prefill_tests.rs"]
mod prefill_tests;

#[cfg(test)]
mod lazy_embedding_tests {
    use super::lfm3_verify_tests::{model, state_bits, tokens};
    use super::*;

    #[test]
    fn lazy_row_offsets_cover_the_table_only() {
        assert_eq!(lazy_row_offset(100, 4096, 0, 3), Some(100));
        assert_eq!(lazy_row_offset(100, 4096, 2, 3), Some(100 + 2 * 4096));
        assert_eq!(lazy_row_offset(100, 4096, 3, 3), None);
        assert_eq!(lazy_row_offset(100, 4096, u32::MAX, u32::MAX), None);
        assert_eq!(lazy_row_offset(u64::MAX, 1, 1, 2), None);
        assert_eq!(lazy_row_offset(0, u64::MAX, 2, 3), None);
        // Exhaustive small domain: offsets are strictly increasing and disjoint.
        for rows in 0..9u32 {
            for size in 1..5u64 {
                let offsets: Vec<_> = (0..rows + 2)
                    .map(|id| lazy_row_offset(7, size, id, rows))
                    .collect();
                for (id, offset) in offsets.iter().enumerate() {
                    let expected = (id < rows as usize).then(|| 7 + id as u64 * size);
                    assert_eq!(*offset, expected, "rows={rows} size={size} id={id}");
                }
            }
        }
    }

    #[test]
    fn lazy_embeddings_need_the_flag_and_a_separate_head() {
        assert!(lazy_embeddings_enabled(true, Some(b"1")));
        for flag in [None, Some(&b"0"[..]), Some(b""), Some(b"true"), Some(b"1 ")] {
            assert!(!lazy_embeddings_enabled(true, flag), "{flag:?}");
        }
        assert!(!lazy_embeddings_enabled(false, Some(b"1")));
    }

    fn f16_bits(array: &Array) -> Result<Vec<u16>> {
        array.eval()?;
        ensure!(array.dtype() == Dtype::Float16, "expected F16 rows");
        let bits = array.to_f16_bits()?;
        ensure!(!bits.is_empty(), "empty rows");
        Ok(bits)
    }

    /// Prefill logits, then 16 greedy steps: logits bits and final states.
    fn trace(
        lm: &mut Lfm2,
        prompt: &[u32],
    ) -> Result<(Vec<Vec<u16>>, Vec<super::lfm3_verify_tests::StateBits>)> {
        lm.reset();
        let mut logits = lm.forward(&prompt[..32])?;
        for &token in &prompt[32..] {
            logits = lm.forward(&[token])?;
        }
        let mut out = vec![f16_bits(&logits)?];
        for _ in 0..16 {
            let next = *logits.chat_greedy_ids()?.first().context("no greedy ID")?;
            logits = lm.forward(&[next])?;
            out.push(f16_bits(&logits)?);
        }
        Ok((out, state_bits(lm)?))
    }

    /// LFM3-EMBED-LAZY oracle: row bits, prefill/decode logits and caches
    /// equal the resident table; the shifted-row mutation is detected.
    #[test]
    #[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; LFM3-EMBED-LAZY oracle"]
    fn lazy_embeddings_match_resident() -> Result<()> {
        let path = std::env::var("MLXL3_LFM3_MODEL").context("set MLXL3_LFM3_MODEL")?;
        let checkpoint = checkpoint::inspect(Path::new(&path))?;
        let mut model = model()?;
        let lm = &mut model.language;
        ensure!(
            lm.set_lazy_embeddings(&checkpoint, false)?,
            "resident switch"
        );
        ensure!(!lm.lazy_embeddings(), "resident mode expected");
        let vocab = lm.vocab_size as u32;
        let mut ids = vec![0, 1, vocab - 1, vocab - 1, 0, 2, vocab - 2];
        ids.extend(tokens(256, 91));
        ids.extend((0..64).map(|i| i * 1999 % vocab));
        let resident = f16_bits(&lm.embed_tokens(&ids)?)?;
        let prompt = [&tokens(40, 92)[..], &[0, 1, vocab - 1]].concat();
        let (resident_logits, resident_states) = trace(lm, &prompt)?;

        ensure!(lm.set_lazy_embeddings(&checkpoint, true)?, "lazy switch");
        ensure!(lm.lazy_embeddings(), "lazy mode expected");
        let lazy = lm.embed_tokens(&ids)?;
        ensure!(
            lazy.shape() == [1, ids.len() as i32, lm.embeddings.dim()],
            "lazy shape"
        );
        let lazy = f16_bits(&lazy)?;
        ensure!(lazy.len() == resident.len(), "lazy value count");
        let differing = lazy.iter().zip(&resident).filter(|(a, b)| a != b).count();
        ensure!(differing == 0, "{differing} lazy embedding values differ");
        ensure!(
            lazy.iter().all(|&b| half::f16::from_bits(b).is_finite()),
            "non-finite embedding"
        );
        let shifted: Vec<u32> = ids.iter().map(|&id| (id + 1) % vocab).collect();
        let mutated = f16_bits(&lm.embed_tokens(&shifted)?)?;
        let dim = lm.embeddings.dim() as usize;
        let caught = mutated
            .chunks(dim)
            .zip(resident.chunks(dim))
            .filter(|(a, b)| a != b)
            .count();
        ensure!(
            caught == ids.len(),
            "shifted-row mutation caught {caught}/{}",
            ids.len()
        );
        ensure!(
            lm.embeddings.take(&[vocab]).is_err(),
            "row past the table accepted"
        );
        ensure!(
            lm.embed_tokens(&[vocab]).is_err(),
            "token past the vocabulary accepted"
        );
        let (lazy_logits, lazy_states) = trace(lm, &prompt)?;
        ensure!(lazy_logits == resident_logits, "lazy logits differ");
        ensure!(lazy_states == resident_states, "lazy cache differs");
        ensure!(
            lm.set_lazy_embeddings(&checkpoint, false)?,
            "back to resident"
        );
        let (again, _) = trace(lm, &prompt)?;
        ensure!(again == resident_logits, "resident replay differs");
        eprintln!(
            "EMBED_LAZY ids={} values={} steps={} states={} mutation_caught={caught}",
            ids.len(),
            lazy.len(),
            lazy_logits.len(),
            lazy_states.len()
        );
        Ok(())
    }
}

#[cfg(test)]
#[path = "lfm3_verify_tests.rs"]
mod lfm3_verify_tests;

#[cfg(test)]
#[path = "lfm3_cert_tests.rs"]
mod lfm3_cert_tests;

#[cfg(test)]
mod lfm3_prefill_release_tests {
    use super::*;

    #[test]
    #[ignore = "physical M5 LFM3 MCG8; FFN production grouping and semantic fallbacks"]
    fn grouped_ffn_preserves_bias_padding_and_unqualified_fallbacks() -> Result<()> {
        for case in 0..5 {
            let mut cp = checkpoint::inspect(Path::new(&std::env::var("MLXL3_LFM3_MODEL")?))?;
            if case == 3 {
                cp.model_type = "lfm2".to_owned();
            }
            if case == 4 {
                cp.bits = Some(4.);
            }
            let mut pair = [
                Projection::load(&cp, "model.layers.0.feed_forward.w1", 2048, 7680, false)?,
                Projection::load(&cp, "model.layers.0.feed_forward.w3", 2048, 7680, false)?,
            ];
            if case == 1 {
                pair[0].bias =
                    Some(Array::from_f32(&vec![0.125; 7680], &[7680])?.astype(Dtype::Float16)?);
            }
            let width = if case == 2 {
                for p in &mut pair {
                    p.logical_input = 2047;
                }
                2047
            } else {
                2048
            };
            let values = (0..24 * width)
                .map(|v| ((v * 37 % 997) as f32 / 997.) - 0.5)
                .collect::<Vec<_>>();
            let x = Array::from_f32(&values, &[1, 24, width])?.astype(Dtype::Float16)?;
            let expected = pair
                .iter()
                .map(|p| p.forward(&x))
                .collect::<Result<Vec<_>>>()?;
            let inputs = ProjectionBundle::new(pair)?;
            ensure!(matches!(&inputs, ProjectionBundle::Grouped(_)) == (case == 0));
            let actual = inputs.forward(&x)?;
            ensure!(actual.len() == 2 && expected.len() == 2);
            for (a, e) in actual.iter().zip(&expected) {
                ensure!(a.shape() == [1, 24, 7680] && a.shape() == e.shape());
                let values = a.to_f32()?;
                ensure!(
                    values.len() == 24 * 7680
                        && values.iter().all(|v| v.is_finite())
                        && values.iter().any(|v| v.abs() > 0.001)
                );
                ensure!(
                    a.to_bytes()? == e.to_bytes()?,
                    "FFN fallback changed case{case}"
                );
            }
        }
        Ok(())
    }
}
