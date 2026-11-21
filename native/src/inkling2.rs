//! Native Inkling 2 Mini (`inkling2_loop`) inference: EXL3 routed experts, dense FP16 projections,
//! FP32 residual stream, depth attention and loop messages.
//!
//! Reference: Transformers PR #51050 (`modeling_inkling2_loop.py` at 6173a2f6) and mlx-lm PR #2086.
//! The 24 decoder layers are visited 48 times per token (layers `core_start..core_end` run
//! `num_passes` times with shared weights); every visit owns its KV cache and convolution states.
use crate::{
    array::{Array, Dtype},
    checkpoint::{self, Checkpoint},
    lfm2::{Projection, half_weight},
    linear::checkpoint_array,
    moe::Exl3SwitchGlu,
    router,
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{
    fs::File,
    path::Path,
    time::{Duration, Instant},
};

/// Group size of the MLX affine copies made at load.
pub(crate) const AFFINE_GROUP: i32 = 64;

pub(crate) fn bits_from_env(name: &str, default: i32) -> Result<i32> {
    let bits = std::env::var(name)
        .ok()
        .map(|v| v.parse::<i32>())
        .transpose()?
        .unwrap_or(default);
    ensure!(matches!(bits, 4 | 8 | 16), "{name} must be 4, 8 or 16");
    Ok(bits)
}

/// Subset of the attention projections (`q`, `k`, `v`, `r`, `o`, `gate`, `sink`) as a bit set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DenseSet(u8);

impl DenseSet {
    const NAMES: [&'static str; 7] = ["q", "k", "v", "r", "o", "gate", "sink"];
    pub const ALL: Self = Self(0x7f);

    /// `q,o,gate` or `q+o+gate`.
    pub fn parse(list: &str) -> Result<Self> {
        let mut set = 0u8;
        for name in list.split([',', '+']).filter(|n| !n.is_empty()) {
            let index = Self::NAMES
                .iter()
                .position(|n| *n == name)
                .with_context(|| format!("unknown Inkling projection {name}"))?;
            set |= 1 << index;
        }
        ensure!(set != 0, "empty Inkling projection set");
        Ok(Self(set))
    }

    fn contains(self, name: &str) -> bool {
        Self::NAMES
            .iter()
            .position(|n| *n == name)
            .is_some_and(|i| self.0 & (1 << i) != 0)
    }
}

/// Run-time precision of the dense weights that are not EXL3: 16 keeps the checkpoint FP16 tensors; 8 or 4
/// replaces them by MLX affine copies made at load (lossy). Default: FP16 projections and an 8-bit head
/// (SPEED-2026-11-21-INK2M-P3: KL 1e-5 against the FP16 head, -7% decode time).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Precision {
    /// q, k, v, r, o, gate and sink projections of every attention.
    pub dense_bits: i32,
    /// Output head (unpadded vocabulary rows).
    pub head_bits: i32,
    /// Projections that take `dense_bits`; the others stay FP16.
    pub dense_only: DenseSet,
}

impl Precision {
    /// Every dense weight as stored in the checkpoint (FP16): the reference of the numerical checks.
    pub const FULL: Self = Self {
        dense_bits: 16,
        head_bits: 16,
        dense_only: DenseSet::ALL,
    };

    /// What generation uses unless the environment says otherwise.
    pub const DEFAULT: Self = Self {
        dense_bits: 16,
        head_bits: 8,
        dense_only: DenseSet::ALL,
    };

    /// `MLXL3_INKLING_DENSE_BITS` (default 16) and `MLXL3_INKLING_HEAD_BITS` (default 8), each 4, 8 or 16.
    pub fn from_env() -> Result<Self> {
        Self::from_env_or(Self::DEFAULT)
    }

    /// The environment overrides `default` field by field.
    pub fn from_env_or(default: Self) -> Result<Self> {
        Ok(Self {
            dense_bits: bits_from_env("MLXL3_INKLING_DENSE_BITS", default.dense_bits)?,
            head_bits: bits_from_env("MLXL3_INKLING_HEAD_BITS", default.head_bits)?,
            dense_only: match std::env::var("MLXL3_INKLING_DENSE_ONLY") {
                Ok(list) => DenseSet::parse(&list)?,
                Err(_) => default.dense_only,
            },
        })
    }

    fn validate(self) -> Result<Self> {
        ensure!(
            matches!(self.dense_bits, 4 | 8 | 16) && matches!(self.head_bits, 4 | 8 | 16),
            "Inkling precision bits must be 4, 8 or 16"
        );
        Ok(self)
    }
}

/// A dense projection: the checkpoint FP16 tensor, or an MLX affine copy of it.
pub(crate) enum Linear {
    Dense(Projection),
    Affine {
        w: Array,
        scales: Array,
        biases: Array,
        bits: i32,
    },
}

impl Linear {
    pub(crate) fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        input: i32,
        output: i32,
        bits: i32,
    ) -> Result<Self> {
        let dense = Projection::load(checkpoint, prefix, input, output, false)?;
        if bits >= 16 {
            return Ok(Self::Dense(dense));
        }
        Self::quantized(dense.dense_weight().context("dense weight")?, bits)
    }

    /// `weight`: `[output, input]`, `input` a multiple of the group size.
    pub(crate) fn quantized(weight: &Array, bits: i32) -> Result<Self> {
        let shape = weight.shape();
        ensure!(
            shape.len() == 2 && shape[1] % AFFINE_GROUP == 0 && matches!(bits, 4 | 8),
            "unsupported affine projection {shape:?} at {bits} bits"
        );
        let (w, scales, biases) = weight
            .astype(Dtype::Float16)?
            .quantize_affine(AFFINE_GROUP, bits)?;
        Array::eval_many(&[&w, &scales, &biases])?;
        Ok(Self::Affine {
            w,
            scales,
            biases,
            bits,
        })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        if ablated("dense")
            && let Self::Dense(p) = self
        {
            // same output shape from one row of the weight: no full read of the matrix
            let w = p.dense_weight().context("dense weight")?;
            let rows = x.shape()[0];
            return x
                .slice(1, 0, 1)?
                .astype(Dtype::Float16)?
                .matmul(&w.slice(1, 0, 1)?.reshape(&[1, w.shape()[0]])?)?
                .reshape(&[rows, w.shape()[0]]);
        }
        match self {
            Self::Dense(p) => p.forward(x),
            Self::Affine {
                w,
                scales,
                biases,
                bits,
            } => x
                .astype(Dtype::Float16)?
                .quantized_matmul(w, scales, biases, AFFINE_GROUP, *bits),
        }
    }
}

#[derive(Deserialize)]
struct RootConfig {
    model_type: String,
    text_config: TextConfig,
}

#[derive(Deserialize)]
struct LoopConfig {
    core_start: usize,
    core_end: usize,
    num_passes: usize,
    gate_rank: i32,
    #[serde(default)]
    is_varloop: bool,
}

#[derive(Deserialize)]
struct TextConfig {
    /// Affine bits of the dense attention projections (16 = FP16), set from `Precision` at load.
    #[serde(skip)]
    dense_bits: i32,
    /// Names of the attention projections that take `dense_bits` (the others stay FP16).
    #[serde(skip)]
    dense_only: DenseSet,
    hidden_size: i32,
    num_hidden_layers: usize,
    vocab_size: i32,
    unpadded_vocab_size: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    swa_head_dim: i32,
    swa_num_attention_heads: i32,
    swa_num_key_value_heads: i32,
    d_rel: i32,
    rel_extent: i32,
    log_scaling_n_floor: f32,
    log_scaling_alpha: f32,
    rms_norm_eps: f32,
    qk_l2_clip_threshold: f32,
    rel_l2_clip_threshold: f32,
    #[serde(default)]
    local_layer_ids: Vec<usize>,
    /// Transformers configs: `hybrid_sliding` / `hybrid` per layer; takes precedence over `local_layer_ids`.
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(alias = "conv_kernel_size")]
    sconv_kernel_size: i32,
    final_logit_softcapping: f32,
    sliding_window_size: i32,
    n_layers_per_attn_res_block: usize,
    n_routed_experts: i32,
    num_experts_per_tok: usize,
    n_shared_experts: i32,
    #[serde(alias = "moe_intermediate_size")]
    intermediate_size: i32,
    mlp_gate_softcap: f32,
    mlp_up_softcap: f32,
    route_scale: f32,
    use_gate_bias: bool,
    #[serde(alias = "max_position_embeddings")]
    model_max_length: i32,
    loop_config: LoopConfig,
}

impl TextConfig {
    fn is_sliding(&self, layer: usize) -> Result<bool> {
        match &self.layer_types {
            Some(types) => match types.get(layer).map(String::as_str) {
                Some("hybrid_sliding") => Ok(true),
                Some("hybrid") => Ok(false),
                other => anyhow::bail!("unsupported Inkling layer type {other:?} at layer {layer}"),
            },
            None => Ok(self.local_layer_ids.contains(&layer)),
        }
    }
}

const L2_EPS: f32 = 1e-6;

fn f32_weight(checkpoint: &Checkpoint, name: &str, shape: &[i32]) -> Result<Array> {
    let weight = checkpoint_array(checkpoint, name)?;
    ensure!(
        matches!(
            weight.dtype(),
            Dtype::Float16 | Dtype::Float32 | Dtype::BFloat16
        ),
        "{name}: expected floating-point weights"
    );
    ensure!(
        weight.shape() == shape,
        "{name}: shape {:?}, expected {shape:?}",
        weight.shape()
    );
    let weight = weight.astype(Dtype::Float32)?;
    weight.eval()?;
    Ok(weight)
}

/// Dense FP16 projection stored as `[output, input]` under an arbitrary tensor name.
struct Dense {
    /// `[input, output]` FP16.
    weight_t: Array,
}

impl Dense {
    fn load(checkpoint: &Checkpoint, name: &str, output: i32, input: i32) -> Result<Self> {
        let weight_t = half_weight(checkpoint, name, Some(&[output, input]))?.transpose(&[1, 0])?;
        weight_t.eval()?;
        Ok(Self { weight_t })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        x.astype(Dtype::Float16)?.matmul(&self.weight_t)
    }
}

fn scalar(value: f32) -> Result<Array> {
    Array::from_f32(&[value], &[1])
}

/// Compiled (fused) subgraphs, bit-identical to the unfused op sequences; `MLXL3_INKLING_UNFUSED=1` keeps the
/// unfused path for A/B checks.
fn fused() -> bool {
    static FUSED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FUSED.get_or_init(|| std::env::var_os("MLXL3_INKLING_UNFUSED").is_none())
}

/// Graph simplifications and single-kernel steps of SPEED-2026-11-21-INK2M-P4 (same bits as the plain op
/// sequences): on with the fused path, `MLXL3_INKLING_P4_OFF=1` turns only them off for A/B checks.
fn lean() -> bool {
    static LEAN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LEAN.get_or_init(|| fused() && std::env::var_os("MLXL3_INKLING_P4_OFF").is_none())
}

/// Timing-only ablations for the test harness (`MLXL3_INKLING_ABLATE=experts,head,...`): the named part is
/// replaced by a trivial computation of the same shape so its cost can be read as a time difference. The outputs
/// are meaningless. Compiled out of non-test builds.
#[cfg(test)]
fn ablated(part: &str) -> bool {
    static PARTS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    PARTS
        .get_or_init(|| {
            std::env::var("MLXL3_INKLING_ABLATE")
                .map(|v| v.split(',').map(str::to_string).collect())
                .unwrap_or_default()
        })
        .iter()
        .any(|p| p == part)
}

#[cfg(not(test))]
#[inline(always)]
fn ablated(_part: &str) -> bool {
    false
}

/// `(a + b) * scale`
fn add_scaled(a: &Array, b: &Array, scale: f32) -> Result<Array> {
    if fused() {
        a.add_scaled(b, scale)
    } else {
        a.add(b)?.scalar_mul(scale)
    }
}

/// `x * min(threshold / (|x| + 1e-6), 1)` over the last axis, after an optional per-element scale.
fn l2_clip(x: &Array, scale: Option<&Array>, threshold: f32) -> Result<Array> {
    if fused() {
        return match scale {
            Some(scale) => x.l2_clip_scaled(scale, threshold),
            None => x.l2_clip(threshold),
        };
    }
    let x = x.astype(Dtype::Float32)?;
    let x = match scale {
        Some(scale) => x.mul(scale)?,
        None => x,
    };
    let last = x.shape().len() as i32 - 1;
    let norm = x.mul(&x)?.sum(last, true)?.sqrt()?.scalar_add(L2_EPS)?;
    x.mul(&scalar(threshold)?.div(&norm)?.clamp_max(1.0)?)
}

/// Attention of learned queries over the depth sources of each token.
/// `sources`: N arrays `[T, D]` (FP32, already RMS-normalized, also the values); `queries`: `[Q, D]`.
/// Returns `[T, Q, D]` FP32.
fn depth_attention(sources: &[&Array], queries: &Array) -> Result<Array> {
    depth_attend(&depth_stack(sources)?, queries)
}

/// `[T, N, D]` stack of the depth sources.
fn depth_stack(sources: &[&Array]) -> Result<Array> {
    ensure!(!sources.is_empty(), "depth attention needs a source");
    let [time, dims]: [i32; 2] = sources[0]
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("depth sources must have rank 2"))?;
    let stacked = sources
        .iter()
        .map(|s| s.reshape(&[time, 1, dims]))
        .collect::<Result<Vec<_>>>()?;
    Array::concatenate(&stacked.iter().collect::<Vec<_>>(), 1)
}

fn depth_attend(stacked: &Array, queries: &Array) -> Result<Array> {
    let [time, sources, dims]: [i32; 3] = stacked
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("depth stack must have rank 3"))?;
    let count = queries.shape()[0];
    if ablated("depth") {
        return stacked
            .slice(1, sources - 1, sources)?
            .broadcast_to(&[time, count, dims]);
    }
    // one query: both transposes only move a size-1 axis, so a reshape gives the same array
    let single = count == 1 && lean();
    let queries_t = if single {
        queries.reshape(&[dims, 1])?
    } else {
        queries.transpose(&[1, 0])?
    };
    let scores = stacked
        .matmul(&queries_t)?
        .scalar_mul((dims as f32).powf(-0.5))?;
    let weights = if single {
        scores.reshape(&[time, 1, sources])?
    } else {
        scores.transpose(&[0, 2, 1])?
    }
    .softmax_precise()?;
    weights.matmul(stacked)
}

/// `[T, H, D]` -> `[1, H, T, D]`; with one row the transpose only moves a size-1 axis.
fn heads_first(x: &Array, time: i32, heads: i32, dims: i32) -> Result<Array> {
    if time == 1 && lean() {
        x.reshape(&[1, heads, 1, dims])
    } else {
        x.transpose(&[1, 0, 2])?.reshape(&[1, heads, time, dims])
    }
}

/// Depthwise causal short convolution over a `[T, C]` FP32 sequence with its `K - 1` previous rows.
struct ShortConv {
    /// `[C, K, 1]` FP32 (MLX conv1d layout).
    weight: Array,
    /// `[1]` FP32: `1 / sqrt(K + 1)`.
    scale: Array,
    channels: i32,
    kernel: i32,
}

/// One thread per window element `(channel, row)`: copies the window `[K - 1 + T, C]` (state rows, then the
/// inputs as FP32) and, for the first `T` rows, writes `(conv + x) * scale`. The taps are accumulated exactly as
/// MLX's `depthwise_conv_1d` kernel does (`acc = 0; acc += in * w` in tap order), then the residual add and the
/// scale as the two separate operations they replace, so the result is the same bits as the op sequence.
const SHORT_CONV_SOURCE: &str = r#"
    uint c = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    int K = w_shape[1];
    int K1 = K - 1;
    int T = x_shape[0];
    int C = x_shape[1];
    float v;
    if (int(row) < K1) {
        v = state[row * C + c];
    } else {
        v = static_cast<float>(x[(row - K1) * C + c]);
    }
    window[row * C + c] = v;
    if (int(row) < T) {
        float acc = 0.0;
        for (int i = 0; i < K; ++i) {
            int r = int(row) + i;
            float in_v;
            if (r < K1) {
                in_v = state[r * C + c];
            } else {
                in_v = static_cast<float>(x[(r - K1) * C + c]);
            }
            acc += in_v * w[c * K + i];
        }
        float sum = acc + static_cast<float>(x[row * C + c]);
        out[row * C + c] = sum * scale[0];
    }
"#;

impl ShortConv {
    fn load(checkpoint: &Checkpoint, name: &str, channels: i32, kernel: i32) -> Result<Self> {
        let weight = f32_weight(checkpoint, name, &[channels, 1, kernel])?
            .transpose(&[0, 2, 1])?
            .reshape(&[channels, kernel, 1])?;
        weight.eval()?;
        Ok(Self {
            weight,
            scale: scalar(1.0 / ((kernel + 1) as f32).sqrt())?,
            channels,
            kernel,
        })
    }

    /// Returns `((conv + x) / sqrt(K + 1), window)`; the window `[K - 1 + T, C]` holds the next states.
    fn forward(&self, x: &Array, state: Option<&Array>) -> Result<(Array, Array)> {
        if ablated("conv") {
            let rows = x.shape()[0] + self.kernel - 1;
            return Ok((
                x.astype(Dtype::Float32)?,
                Array::zeros_dtype(&[rows, self.channels], Dtype::Float32)?,
            ));
        }
        if lean() {
            self.forward_kernel(x, state)
        } else {
            self.forward_ops(x, state)
        }
    }

    /// The whole step in one Metal kernel (same bits as `forward_ops`).
    fn forward_kernel(&self, x: &Array, state: Option<&Array>) -> Result<(Array, Array)> {
        let time = x.shape()[0];
        ensure!(
            x.shape() == [time, self.channels]
                && matches!(x.dtype(), Dtype::Float16 | Dtype::Float32),
            "short convolution input"
        );
        let zeros;
        let state = match state {
            Some(state) => state,
            None => {
                zeros = Array::zeros_dtype(&[self.kernel - 1, self.channels], Dtype::Float32)?;
                &zeros
            }
        };
        ensure!(
            state.shape() == [self.kernel - 1, self.channels] && state.dtype() == Dtype::Float32,
            "short convolution state"
        );
        let rows = time + self.kernel - 1;
        let kernel = crate::array::prepared_kernel("inkling_short_conv", [0; 16], || {
            crate::array::PreparedKernel::new(
                "inkling_short_conv",
                &["x", "state", "w", "scale"],
                &["out", "window"],
                "",
                SHORT_CONV_SOURCE,
            )
        })?;
        let mut outputs = kernel.run(
            &[x, state, &self.weight, &self.scale],
            &[&[time, self.channels], &[rows, self.channels]],
            &[Dtype::Float32, Dtype::Float32],
            [self.channels, rows, 1],
            [self.channels.min(256), 1, 1],
        )?;
        let window = outputs.pop().context("short convolution window")?;
        let out = outputs.pop().context("short convolution output")?;
        Ok((out, window))
    }

    fn forward_ops(&self, x: &Array, state: Option<&Array>) -> Result<(Array, Array)> {
        let time = x.shape()[0];
        let x = x.astype(Dtype::Float32)?;
        let window = with_state(&x, state, self.kernel, self.channels)?;
        let conv = window
            .reshape(&[1, time + self.kernel - 1, self.channels])?
            .conv1d(&self.weight, self.channels)?
            .reshape(&[time, self.channels])?;
        let out = add_scaled(&conv, &x, 1.0 / ((self.kernel + 1) as f32).sqrt())?;
        Ok((out, window))
    }
}

fn with_state(x: &Array, state: Option<&Array>, kernel: i32, channels: i32) -> Result<Array> {
    let zeros;
    let state = match state {
        Some(state) => state,
        None => {
            zeros = Array::zeros_dtype(&[kernel - 1, channels], Dtype::Float32)?;
            &zeros
        }
    };
    Array::concatenate(&[state, x], 0)
}

/// Convolution state after the first `rows` inputs of a `[K - 1 + T, C]` window.
fn state_after(window: &Array, kernel: i32, rows: i32) -> Result<Array> {
    window.slice(0, rows, rows + kernel - 1)
}

/// Short convolution whose taps are static per-channel weights plus per-token weights projected
/// from the projection input, shared within each key/value head (full-attention layers).
struct DynamicConv {
    /// `[H, C / H, K]` FP32, tap index = backward distance.
    static_weight: Array,
    weight_proj: Dense,
    heads: i32,
    channels: i32,
    kernel: i32,
}

impl DynamicConv {
    fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        hidden: i32,
        heads: i32,
        channels: i32,
        kernel: i32,
    ) -> Result<Self> {
        let static_weight = f32_weight(
            checkpoint,
            &format!("{prefix}.static_w"),
            &[kernel, channels],
        )?
        .transpose(&[1, 0])?
        .reshape(&[heads, channels / heads, kernel])?;
        static_weight.eval()?;
        Ok(Self {
            static_weight,
            weight_proj: Dense::load(
                checkpoint,
                &format!("{prefix}.weight_proj"),
                heads * kernel,
                hidden,
            )?,
            heads,
            channels,
            kernel,
        })
    }

    fn forward(&self, x: &Array, input: &Array, state: Option<&Array>) -> Result<(Array, Array)> {
        let time = x.shape()[0];
        if ablated("conv") {
            return Ok((
                x.astype(Dtype::Float32)?,
                Array::zeros_dtype(&[time + self.kernel - 1, self.channels], Dtype::Float32)?,
            ));
        }
        let per_head = self.channels / self.heads;
        let weights = self
            .weight_proj
            .forward(input)?
            .astype(Dtype::Float32)?
            .reshape(&[time, self.heads, 1, self.kernel])?
            .add(&self.static_weight)?;
        let x = x.astype(Dtype::Float32)?;
        let window = with_state(&x, state, self.kernel, self.channels)?;
        let mut out: Option<Array> = None;
        for lag in 0..self.kernel {
            let start = self.kernel - 1 - lag;
            let shifted = window
                .slice(0, start, start + time)?
                .reshape(&[time, self.heads, per_head])?;
            let tap = weights
                .slice(3, lag, lag + 1)?
                .reshape(&[time, self.heads, per_head])?;
            let term = shifted.mul(&tap)?;
            out = Some(match out {
                Some(acc) => acc.add(&term)?,
                None => term,
            });
        }
        let out = out
            .context("empty dynamic convolution")?
            .reshape(&[time, self.channels])?;
        let out = add_scaled(&out, &x, 1.0 / ((2 * self.kernel + 1) as f32).sqrt())?;
        Ok((out, window))
    }
}

enum KvConv {
    Static { k: ShortConv, v: ShortConv },
    Dynamic { k: DynamicConv, v: DynamicConv },
}

/// Positions shared by every visit of one attention kind during a forward call.
struct MaskPlan {
    /// `[T * S]` flat indices `row * R + min(distance, R - 1)` into the per-row bias profiles.
    gather: Array,
    /// `[T, S]` 1 where `0 <= distance < R`.
    bias_valid: Array,
    /// `[T, S]` 0 for visible keys, -inf otherwise.
    visible: Array,
    /// Every `bias_valid` entry is 1 / every `visible` entry is 0: the multiply / add are identities.
    all_valid: bool,
    all_visible: bool,
    time: i32,
    keys: i32,
}

impl MaskPlan {
    /// Queries at positions `start + i` (i < time) attend keys at positions `first_key..first_key + keys`.
    fn new(
        start: i32,
        time: i32,
        first_key: i32,
        keys: i32,
        extent: i32,
        window: Option<i32>,
    ) -> Result<Self> {
        let count = (time as usize)
            .checked_mul(keys as usize)
            .context("attention mask overflow")?;
        let (mut gather, mut valid, mut visible) = (
            Vec::with_capacity(count),
            Vec::with_capacity(count),
            Vec::with_capacity(count),
        );
        for row in 0..time {
            for key in 0..keys {
                let distance = start + row - (first_key + key);
                let seen = distance >= 0 && window.is_none_or(|w| distance < w);
                gather.push(row * extent + distance.clamp(0, extent - 1));
                valid.push(if (0..extent).contains(&distance) {
                    1.
                } else {
                    0.
                });
                visible.push(if seen { 0. } else { f32::NEG_INFINITY });
            }
        }
        Ok(Self {
            all_valid: valid.iter().all(|&v| v == 1.),
            all_visible: visible.iter().all(|&v| v == 0.),
            gather: Array::from_i32(&gather, &[time * keys])?,
            bias_valid: Array::from_f32(&valid, &[time, keys])?,
            visible: Array::from_f32(&visible, &[time, keys])?,
            time,
            keys,
        })
    }
}

#[derive(Default)]
struct VisitState {
    /// Cached keys/values FP16. Lean layout (default): `[1, Hkv, S, D]`, the real rows only, kept as a view
    /// of the last attended array so a sliding trim copies nothing. `MLXL3_INKLING_UNFUSED=1`:
    /// `[1, Hkv, 1 + S, D]` with the zero sink row first, trimmed by a copy.
    keys: Option<Array>,
    values: Option<Array>,
    conv: [Option<Array>; 4],
    /// Last call, not committed yet: keys/values with every row of the call (no sliding trim), the row count
    /// before the call, and the four `[K - 1 + T, C]` convolution windows.
    pending_kv: Option<(Array, Array, i32)>,
    pending_conv: [Option<Array>; 4],
}

impl VisitState {
    /// Keep the first `rows` rows of the pending call. Sliding visits then keep the sink row and the last
    /// `window - 1` real keys.
    fn commit(&mut self, rows: i32, calls: i32, window: Option<i32>, kernel: i32) -> Result<()> {
        let (keys, values, before) = self.pending_kv.take().context("no pending Inkling call")?;
        let cut = |a: Array| -> Result<Array> {
            if rows == calls {
                Ok(a)
            } else {
                a.slice(2, 0, before + rows)
            }
        };
        let (keys, values) = (cut(keys)?, cut(values)?);
        let total = keys.shape()[2];
        if lean() {
            // drop the sink row (index 0) and, on sliding visits, everything but the last window - 1 rows:
            // a view of the attended array, same rows in the same order as the copying trim below
            let from = match window {
                Some(window) if total - 1 > window - 1 => total - (window - 1),
                _ => 1,
            };
            ensure!(from < total, "Inkling cache would be empty");
            self.keys = Some(keys.slice(2, from, total)?);
            self.values = Some(values.slice(2, from, total)?);
            for i in 0..4 {
                let window = self.pending_conv[i]
                    .take()
                    .context("missing pending convolution")?;
                self.conv[i] = Some(state_after(&window, kernel, rows)?);
            }
            return Ok(());
        }
        let keep = match window {
            Some(window) if total - 1 > window - 1 => Some(total - (window - 1)),
            _ => None,
        };
        let trim = |a: &Array| -> Result<Array> {
            match keep {
                Some(from) => {
                    Array::concatenate(&[&a.slice(2, 0, 1)?, &a.slice(2, from, total)?], 2)
                }
                None => a.try_clone(),
            }
        };
        self.keys = Some(trim(&keys)?);
        self.values = Some(trim(&values)?);
        for i in 0..4 {
            let window = self.pending_conv[i]
                .take()
                .context("missing pending convolution")?;
            self.conv[i] = Some(state_after(&window, kernel, rows)?);
        }
        Ok(())
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    r: Linear,
    o: Linear,
    gate: Linear,
    sink: Linear,
    rel_proj: Array,
    q_scale: Array,
    rel_scale: Array,
    conv: KvConv,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    d_rel: i32,
    extent: i32,
    window: Option<i32>,
    log_scaling: Option<(f32, f32)>,
    qk_clip: f32,
    rel_clip: f32,
    eps: f32,
}

impl Attention {
    fn load(checkpoint: &Checkpoint, layer: usize, sliding: bool, c: &TextConfig) -> Result<Self> {
        let prefix = format!("model.llm.layers.{layer}.attn");
        let (heads, kv_heads, head_dim) = if sliding {
            (
                c.swa_num_attention_heads,
                c.swa_num_key_value_heads,
                c.swa_head_dim,
            )
        } else {
            (c.num_attention_heads, c.num_key_value_heads, c.head_dim)
        };
        ensure!(
            heads > 0 && kv_heads > 0 && heads % kv_heads == 0 && head_dim > 0,
            "invalid Inkling attention heads"
        );
        let extent = if sliding {
            c.sliding_window_size
        } else {
            c.rel_extent
        };
        let hidden = c.hidden_size;
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let proj = |role: &str, name: &str, input: i32, output: i32| {
            Linear::load(
                checkpoint,
                &format!("{prefix}.{name}"),
                input,
                output,
                if c.dense_only.contains(role) {
                    c.dense_bits
                } else {
                    16
                },
            )
        };
        let conv = if sliding {
            KvConv::Static {
                k: ShortConv::load(
                    checkpoint,
                    &format!("{prefix}.k_sconv.weight"),
                    kv_dim,
                    c.sconv_kernel_size,
                )?,
                v: ShortConv::load(
                    checkpoint,
                    &format!("{prefix}.v_sconv.weight"),
                    kv_dim,
                    c.sconv_kernel_size,
                )?,
            }
        } else {
            KvConv::Dynamic {
                k: DynamicConv::load(
                    checkpoint,
                    &format!("{prefix}.k_sconv"),
                    hidden,
                    kv_heads,
                    kv_dim,
                    c.sconv_kernel_size,
                )?,
                v: DynamicConv::load(
                    checkpoint,
                    &format!("{prefix}.v_sconv"),
                    hidden,
                    kv_heads,
                    kv_dim,
                    c.sconv_kernel_size,
                )?,
            }
        };
        Ok(Self {
            q: proj("q", "wq_du", hidden, q_dim)?,
            k: proj("k", "wk_dv", hidden, kv_dim)?,
            v: proj("v", "wv_dv", hidden, kv_dim)?,
            r: proj("r", "wr_du", hidden, heads * c.d_rel)?,
            o: proj("o", "wo_ud", q_dim, hidden)?,
            gate: proj("gate", "gate_attn", hidden, q_dim)?,
            sink: proj("sink", "attn_sink_proj", hidden, heads)?,
            rel_proj: f32_weight(
                checkpoint,
                &format!("{prefix}.rel_logits_proj.proj"),
                &[c.d_rel, extent],
            )?,
            q_scale: f32_weight(
                checkpoint,
                &format!("{prefix}.q_l2_clip.scale"),
                &[heads, head_dim],
            )?,
            rel_scale: f32_weight(
                checkpoint,
                &format!("{prefix}.rel_l2_clip.scale"),
                &[heads, c.d_rel],
            )?,
            conv,
            heads,
            kv_heads,
            head_dim,
            d_rel: c.d_rel,
            extent,
            window: sliding.then_some(c.sliding_window_size),
            log_scaling: (!sliding).then_some((c.log_scaling_alpha, c.log_scaling_n_floor)),
            qk_clip: c.qk_l2_clip_threshold,
            rel_clip: c.rel_l2_clip_threshold,
            eps: c.rms_norm_eps,
        })
    }

    /// `tau = 1 + alpha * log(max((position + 1) / n_floor, 1))` per query row, `[T, 1, 1]`.
    fn tau(&self, start: i32, time: i32) -> Result<Option<Array>> {
        let Some((alpha, floor)) = self.log_scaling else {
            return Ok(None);
        };
        let values = (0..time)
            .map(|i| 1.0 + alpha * (((start + i + 1) as f32 / floor).max(1.0)).ln())
            .collect::<Vec<_>>();
        Ok(Some(Array::from_f32(&values, &[time, 1, 1])?))
    }

    /// `hq`, `hk`, `hv`: normalized FP16 inputs `[T, hidden]`. Returns the attention output `[T, hidden]` FP16.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hq: &Array,
        hk: &Array,
        hv: &Array,
        start: i32,
        plan: &MaskPlan,
        state: &mut VisitState,
    ) -> Result<Array> {
        let time = hq.shape()[0];
        let (k, v, k_state, v_state) = match &self.conv {
            KvConv::Static { k, v } => {
                let (ko, ks) = k.forward(&self.k.forward(hk)?, state.conv[0].as_ref())?;
                let (vo, vs) = v.forward(&self.v.forward(hv)?, state.conv[1].as_ref())?;
                (ko, vo, ks, vs)
            }
            KvConv::Dynamic { k, v } => {
                let (ko, ks) = k.forward(&self.k.forward(hk)?, hk, state.conv[0].as_ref())?;
                let (vo, vs) = v.forward(&self.v.forward(hv)?, hv, state.conv[1].as_ref())?;
                (ko, vo, ks, vs)
            }
        };
        let tau = self.tau(start, time)?;
        let mut q = l2_clip(
            &self
                .q
                .forward(hq)?
                .reshape(&[time, self.heads, self.head_dim])?,
            Some(&self.q_scale),
            self.qk_clip,
        )?;
        let mut rel = l2_clip(
            &self
                .r
                .forward(hq)?
                .reshape(&[time, self.heads, self.d_rel])?,
            Some(&self.rel_scale),
            self.rel_clip,
        )?;
        if let Some(tau) = &tau {
            q = q.mul(tau)?;
            rel = rel.mul(tau)?;
        }
        let k = l2_clip(
            &k.reshape(&[time, self.kv_heads, self.head_dim])?,
            None,
            self.qk_clip,
        )?;
        let v = v
            .reshape(&[time, self.kv_heads, self.head_dim])?
            .rms_norm_without_weight(self.eps)?;
        let q = heads_first(&q.astype(Dtype::Float16)?, time, self.heads, self.head_dim)?;
        let k = heads_first(
            &k.astype(Dtype::Float16)?,
            time,
            self.kv_heads,
            self.head_dim,
        )?;
        let v = heads_first(
            &v.astype(Dtype::Float16)?,
            time,
            self.kv_heads,
            self.head_dim,
        )?;
        let (keys, values) = self.append(state, &k, &v, start)?;
        ensure!(
            keys.shape()[2] == plan.keys + 1 && plan.time == time,
            "Inkling attention cache and mask disagree"
        );
        // Bias profiles [T, H, R] -> per (head, query, key) through the distance gather
        let lean = lean();
        let one = time == 1 && lean;
        if ablated("attn") {
            // keep the projections and the cache append; drop bias, mask and attention
            state.pending_conv[0] = Some(k_state);
            state.pending_conv[1] = Some(v_state);
            let gate = self.gate.forward(hq)?;
            let _ = (&rel, &keys, &values);
            return self.o.forward(
                &q.reshape(&[time, self.heads * self.head_dim])?
                    .add(&gate.scalar_mul(0.0)?)?,
            );
        }
        let profiles = rel.matmul(&self.rel_proj)?;
        let profiles = if one {
            profiles.reshape(&[self.heads, self.extent])?
        } else {
            profiles
                .transpose(&[1, 0, 2])?
                .reshape(&[self.heads, time * self.extent])?
        };
        let mut bias = if ablated("bias") {
            Array::zeros_dtype(&[self.heads, time, plan.keys], Dtype::Float32)?
        } else {
            profiles
                .take(&plan.gather, 1)?
                .reshape(&[self.heads, time, plan.keys])?
        };
        // x * 1 and x + 0 are exact identities (a -0 bias would become +0, which the attention scores
        // absorb identically); skip them when the whole mask says so
        if !(lean && plan.all_valid) {
            bias = bias.mul(&plan.bias_valid)?;
        }
        if !(lean && plan.all_visible) {
            bias = bias.add(&plan.visible)?;
        }
        let sinks = self.sink.forward(hq)?.astype(Dtype::Float32)?;
        let sinks = if one {
            sinks.reshape(&[self.heads, 1, 1])?
        } else {
            sinks.transpose(&[1, 0])?.reshape(&[self.heads, time, 1])?
        };
        let mask = Array::concatenate(&[&sinks, &bias], 2)?
            .reshape(&[1, self.heads, time, plan.keys + 1])?
            .astype(Dtype::Float16)?;
        let attended =
            Array::sdpa_mask(&q, &keys, &values, (self.head_dim as f32).powf(-0.5), &mask)?;
        let attended = if one {
            attended.reshape(&[1, self.heads * self.head_dim])?
        } else {
            attended
                .reshape(&[self.heads, time, self.head_dim])?
                .transpose(&[1, 0, 2])?
                .reshape(&[time, self.heads * self.head_dim])?
        };
        let gate = self.gate.forward(hq)?.astype(Dtype::Float32)?.sigmoid()?;
        let gated = attended
            .astype(Dtype::Float32)?
            .mul(&gate)?
            .astype(Dtype::Float16)?;
        state.pending_conv[0] = Some(k_state);
        state.pending_conv[1] = Some(v_state);
        self.o.forward(&gated)
    }

    /// Appends the chunk and returns the keys/values it attends (sink row, past, chunk). Sliding
    /// layers then keep only the sink row and the last `window - 1` real keys for the next call.
    fn append(
        &self,
        state: &mut VisitState,
        k: &Array,
        v: &Array,
        start: i32,
    ) -> Result<(Array, Array)> {
        if lean() {
            // lean layout: the cache holds the real rows only; one concatenation builds [sink, past, chunk]
            let zero = Array::zeros_dtype(&[1, self.kv_heads, 1, self.head_dim], Dtype::Float16)?;
            let (keys, values, before) = match (&state.keys, &state.values) {
                (Some(keys), Some(values)) => (
                    Array::concatenate(&[&zero, keys, k], 2)?,
                    Array::concatenate(&[&zero, values, v], 2)?,
                    keys.shape()[2] + 1,
                ),
                _ => {
                    ensure!(start == 0, "Inkling cache must start at position zero");
                    (
                        Array::concatenate(&[&zero, k], 2)?,
                        Array::concatenate(&[&zero, v], 2)?,
                        1,
                    )
                }
            };
            state.pending_kv = Some((keys.try_clone()?, values.try_clone()?, before));
            return Ok((keys, values));
        }
        let (keys, values) = match (&state.keys, &state.values) {
            (Some(keys), Some(values)) => (
                Array::concatenate(&[keys, k], 2)?,
                Array::concatenate(&[values, v], 2)?,
            ),
            _ => {
                ensure!(start == 0, "Inkling cache must start at position zero");
                let zero =
                    Array::zeros_dtype(&[1, self.kv_heads, 1, self.head_dim], Dtype::Float16)?;
                (
                    Array::concatenate(&[&zero, k], 2)?,
                    Array::concatenate(&[&zero, v], 2)?,
                )
            }
        };
        let before = state.keys.as_ref().map_or(1, |k| k.shape()[2]);
        state.pending_kv = Some((keys.try_clone()?, values.try_clone()?, before));
        Ok((keys, values))
    }
}

/// `(1 - tanh(forget(s))) * h + tanh(gain(s)) * m`, `s = down([rms(h), rms(m)])`.
struct MessageGate {
    down: Dense,
    /// `[rank, hidden]` FP32 (transposed for matmul).
    forget_t: Array,
    gain_t: Array,
    eps: f32,
}

impl MessageGate {
    fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        hidden: i32,
        rank: i32,
        eps: f32,
    ) -> Result<Self> {
        let t = |name: &str| -> Result<Array> {
            let w = f32_weight(checkpoint, &format!("{prefix}.{name}"), &[hidden, rank])?
                .transpose(&[1, 0])?;
            w.eval()?;
            Ok(w)
        };
        Ok(Self {
            down: Dense::load(
                checkpoint,
                &format!("{prefix}.down_weight"),
                rank,
                2 * hidden,
            )?,
            forget_t: t("forget_weight")?,
            gain_t: t("gain_weight")?,
            eps,
        })
    }

    fn forward(&self, hidden: &Array, message: &Array) -> Result<Array> {
        self.forward_normed(hidden, &hidden.rms_norm_without_weight(self.eps)?, message)
    }

    /// `hidden_norm`: `rms(hidden)`, shared by the gates that read the same hidden state.
    fn forward_normed(
        &self,
        hidden: &Array,
        hidden_norm: &Array,
        message: &Array,
    ) -> Result<Array> {
        if ablated("gates") {
            return hidden.try_clone();
        }
        let states = Array::concatenate(
            &[hidden_norm, &message.rms_norm_without_weight(self.eps)?],
            1,
        )?;
        let states = self.down.forward(&states)?.astype(Dtype::Float32)?;
        if fused() {
            return Array::message_gate(&states, &self.forget_t, &self.gain_t, hidden, message);
        }
        let forget = states.matmul(&self.forget_t)?.tanh()?;
        let gain = states.matmul(&self.gain_t)?.tanh()?;
        hidden.sub(&forget.mul(hidden)?)?.add(&gain.mul(message)?)
    }
}

struct Gates {
    q: MessageGate,
    k: MessageGate,
    v: MessageGate,
    mlp: MessageGate,
}

enum InputNorms {
    Shared(Array),
    Split { q: Array, k: Array, v: Array },
}

struct Moe {
    /// `[hidden, experts]` FP32.
    router_t: Array,
    bias: Array,
    experts: Exl3SwitchGlu,
    top_k: usize,
    scale: f32,
}

impl Moe {
    fn forward(&self, x: &Array) -> Result<Array> {
        if ablated("moe") {
            return x.scalar_mul(0.0);
        }
        let logits = x.astype(Dtype::Float32)?.matmul(&self.router_t)?;
        let (selected, scores) =
            router::grouped_topk_biased(&logits, &self.bias, self.top_k, 1, 1, self.scale)?;
        if ablated("experts") {
            // keep the router in the graph, drop the expert kernels
            return x
                .scalar_mul(0.0)?
                .add(
                    &scores
                        .sum(1, true)?
                        .astype(Dtype::Float16)?
                        .scalar_mul(0.0)?,
                )?
                .add(
                    &selected
                        .astype(Dtype::Float16)?
                        .sum(1, true)?
                        .scalar_mul(0.0)?,
                );
        }
        self.experts
            .forward(&x.astype(Dtype::Float16)?, &selected, &scores)
    }
}

struct Layer {
    attention_query: Option<Array>,
    mlp_query: Array,
    norms: InputNorms,
    post_attention_norm: Array,
    attention: Attention,
    post_attention_conv: ShortConv,
    post_mlp_conv: ShortConv,
    moe: Moe,
    gates: Option<Gates>,
    block_boundary: bool,
    sliding: bool,
    eps: f32,
}

impl Layer {
    fn load(checkpoint: &Checkpoint, index: usize, c: &TextConfig) -> Result<Self> {
        let prefix = format!("model.llm.layers.{index}");
        let hidden = c.hidden_size;
        let sliding = c.is_sliding(index)?;
        let attention_query = if index == 0 {
            None
        } else {
            let name = format!("model.llm.attn_res.attn_queries.{}", index - 1);
            let shape = if sliding {
                vec![1, hidden]
            } else {
                vec![3, hidden]
            };
            let raw = checkpoint_array(checkpoint, &name)?;
            ensure!(
                raw.shape().iter().product::<i32>() == shape.iter().product::<i32>(),
                "{name}: unexpected depth query shape {:?}",
                raw.shape()
            );
            let q = raw.astype(Dtype::Float32)?.reshape(&shape)?;
            q.eval()?;
            Some(q)
        };
        ensure!(
            sliding || index > 0,
            "full attention at layer 0 is unsupported"
        );
        let mlp_query = f32_weight(
            checkpoint,
            &format!("model.llm.attn_res.mlp_queries.{index}"),
            &[hidden],
        )?
        .reshape(&[1, hidden])?;
        let norm =
            |name: &str| f32_weight(checkpoint, &format!("{prefix}.{name}.weight"), &[hidden]);
        let norms = if sliding {
            InputNorms::Shared(norm("attn_norm")?)
        } else {
            InputNorms::Split {
                q: norm("q_input_norm")?,
                k: norm("k_input_norm")?,
                v: norm("v_input_norm")?,
            }
        };
        let looped = (c.loop_config.core_start..c.loop_config.core_end).contains(&index);
        let gates = if looped {
            let g = |name: &str| {
                MessageGate::load(
                    checkpoint,
                    &format!("{prefix}.controllers.{name}"),
                    hidden,
                    c.loop_config.gate_rank,
                    c.rms_norm_eps,
                )
            };
            Some(Gates {
                q: g("attn")?,
                k: g("k")?,
                v: g("v")?,
                mlp: g("mlp")?,
            })
        } else {
            None
        };
        let router = f32_weight(
            checkpoint,
            &format!("{prefix}.mlp.gate.weight"),
            &[c.n_routed_experts, hidden],
        )?
        .transpose(&[1, 0])?;
        router.eval()?;
        let bias = f32_weight(
            checkpoint,
            &format!("{prefix}.mlp.gate.bias"),
            &[c.n_routed_experts],
        )?;
        Ok(Self {
            attention_query,
            mlp_query,
            norms,
            post_attention_norm: norm("mlp_norm")?,
            attention: Attention::load(checkpoint, index, sliding, c)?,
            post_attention_conv: ShortConv::load(
                checkpoint,
                &format!("{prefix}.attn_sconv.weight"),
                hidden,
                c.sconv_kernel_size,
            )?,
            post_mlp_conv: ShortConv::load(
                checkpoint,
                &format!("{prefix}.mlp_sconv.weight"),
                hidden,
                c.sconv_kernel_size,
            )?,
            moe: Moe {
                router_t: router,
                bias,
                experts: Exl3SwitchGlu::from_inkling_checkpoint(
                    checkpoint,
                    &format!("{prefix}.mlp.experts"),
                    c.n_routed_experts,
                    c.num_experts_per_tok as i32,
                    c.mlp_gate_softcap,
                    c.mlp_up_softcap,
                )?,
                top_k: c.num_experts_per_tok,
                scale: c.route_scale,
            },
            gates,
            block_boundary: index.is_multiple_of(c.n_layers_per_attn_res_block),
            sliding,
            eps: c.rms_norm_eps,
        })
    }

    /// One visit. `hidden`: layer input `[T, D]` FP32; `blocks`: normalized block states.
    /// `messages`: `[T, 4, D]` FP32 in the loop passes after the first.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &Array,
        blocks: &mut Vec<Array>,
        messages: Option<&Array>,
        start: i32,
        plan: &MaskPlan,
        state: &mut VisitState,
    ) -> Result<Array> {
        let time = hidden.shape()[0];
        let dims = hidden.shape()[1];
        let normed_input = hidden.rms_norm_without_weight(self.eps)?;
        let inputs: Vec<Array> = match &self.attention_query {
            Some(queries) => {
                let mut sources: Vec<&Array> = blocks.iter().collect();
                sources.push(&normed_input);
                let out = depth_attention(&sources, queries)?;
                (0..queries.shape()[0])
                    .map(|i| out.slice(1, i, i + 1)?.reshape(&[time, dims]))
                    .collect::<Result<_>>()?
            }
            None => vec![hidden.try_clone()?],
        };
        let message = |i: i32| -> Result<Option<Array>> {
            messages
                .map(|m| m.slice(1, i, i + 1)?.reshape(&[time, dims]))
                .transpose()
        };
        let (hq, hk, hv) = match &self.norms {
            InputNorms::Shared(weight) => {
                let base = &inputs[0];
                match (&self.gates, messages) {
                    (Some(g), Some(_)) => {
                        // the three gates read the same hidden state: normalize it once
                        let base_norm = base.rms_norm_without_weight(self.eps)?;
                        let gate = |g: &MessageGate, i: i32| -> Result<Array> {
                            let message = message(i)?.context("message")?;
                            if lean() {
                                g.forward_normed(base, &base_norm, &message)
                            } else {
                                g.forward(base, &message)
                            }?
                            .rms_norm(weight, self.eps)?
                            .astype(Dtype::Float16)
                        };
                        (gate(&g.q, 0)?, gate(&g.k, 1)?, gate(&g.v, 2)?)
                    }
                    _ => {
                        let n = base.rms_norm(weight, self.eps)?;
                        if lean() {
                            // one cast for the three identical inputs
                            let n = n.astype(Dtype::Float16)?;
                            (n.try_clone()?, n.try_clone()?, n)
                        } else {
                            (
                                n.astype(Dtype::Float16)?,
                                n.astype(Dtype::Float16)?,
                                n.astype(Dtype::Float16)?,
                            )
                        }
                    }
                }
            }
            InputNorms::Split { q, k, v } => {
                ensure!(inputs.len() == 3, "full attention needs three depth inputs");
                let (iq, ik, iv) = match (&self.gates, messages) {
                    (Some(g), Some(_)) => (
                        g.q.forward(&inputs[0], &message(0)?.context("message")?)?,
                        g.k.forward(&inputs[1], &message(1)?.context("message")?)?,
                        g.v.forward(&inputs[2], &message(2)?.context("message")?)?,
                    ),
                    _ => (
                        inputs[0].try_clone()?,
                        inputs[1].try_clone()?,
                        inputs[2].try_clone()?,
                    ),
                };
                (
                    iq.rms_norm(q, self.eps)?.astype(Dtype::Float16)?,
                    ik.rms_norm(k, self.eps)?.astype(Dtype::Float16)?,
                    iv.rms_norm(v, self.eps)?.astype(Dtype::Float16)?,
                )
            }
        };
        let attended = self.attention.forward(&hq, &hk, &hv, start, plan, state)?;
        let (attended, conv_state) = self
            .post_attention_conv
            .forward(&attended, state.conv[2].as_ref())?;
        state.pending_conv[2] = Some(conv_state);
        let residual = if self.block_boundary {
            blocks.push(normed_input);
            attended
        } else {
            hidden.add(&attended)?
        };
        let mut sources: Vec<&Array> = blocks.iter().collect();
        let normed = residual.rms_norm_without_weight(self.eps)?;
        sources.push(&normed);
        let mut mlp_input = depth_attention(&sources, &self.mlp_query)?.reshape(&[time, dims])?;
        if let (Some(g), Some(_)) = (&self.gates, messages) {
            mlp_input = g
                .mlp
                .forward(&mlp_input, &message(3)?.context("message")?)?;
        }
        let mlp = self.moe.forward(
            &mlp_input
                .rms_norm(&self.post_attention_norm, self.eps)?
                .astype(Dtype::Float16)?,
        )?;
        let (mlp, conv_state) = self.post_mlp_conv.forward(&mlp, state.conv[3].as_ref())?;
        state.pending_conv[3] = Some(conv_state);
        residual.add(&mlp)
    }
}

/// Time spent in `run` since the last `take_profile`: graph construction (CPU) and evaluation.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunProfile {
    pub calls: u64,
    pub build: Duration,
    pub eval: Duration,
}

pub struct Inkling2 {
    profile: RunProfile,
    /// Rows of an uncommitted `verify` call.
    pending: Option<i32>,
    kernel: i32,
    embeddings: Array,
    embed_norm: Array,
    norm: Array,
    final_query: Array,
    message_queries: Array,
    head: Projection,
    /// Affine copy of the unpadded head rows when `precision.head_bits < 16`.
    head_q: Option<Linear>,
    precision: Precision,
    layers: Vec<Layer>,
    visits: Vec<usize>,
    states: Vec<VisitState>,
    core: (usize, usize),
    passes: usize,
    vocab: i32,
    unpadded_vocab: i32,
    hidden: i32,
    eps: f32,
    softcap: f32,
    window: i32,
    extent: i32,
    context_limit: i32,
    offset: i32,
}

impl Inkling2 {
    pub fn load(path: &Path) -> Result<Self> {
        let checkpoint = checkpoint::inspect(path)?;
        Self::from_checkpoint(&checkpoint)
    }

    pub fn load_with(path: &Path, precision: Precision) -> Result<Self> {
        Self::from_checkpoint_with(&checkpoint::inspect(path)?, precision)
    }

    /// Precision from the environment (`Precision::from_env`): FP16 projections and an 8-bit head by default.
    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        Self::from_checkpoint_with(checkpoint, Precision::from_env()?)
    }

    pub fn from_checkpoint_with(checkpoint: &Checkpoint, precision: Precision) -> Result<Self> {
        let precision = precision.validate()?;
        let root: RootConfig =
            serde_json::from_reader(File::open(checkpoint.path.join("config.json"))?)?;
        ensure!(
            root.model_type == "inkling2_loop",
            "expected Inkling 2 (inkling2_loop)"
        );
        let mut c = root.text_config;
        c.dense_bits = precision.dense_bits;
        c.dense_only = precision.dense_only;
        let lc = &c.loop_config;
        ensure!(
            c.hidden_size > 0
                && c.num_hidden_layers > 0
                && lc.core_start < lc.core_end
                && lc.core_end <= c.num_hidden_layers
                && lc.num_passes >= 1
                && !lc.is_varloop
                && c.n_shared_experts == 0
                && c.use_gate_bias
                && c.sconv_kernel_size > 1
                && c.unpadded_vocab_size > 0
                && c.unpadded_vocab_size <= c.vocab_size
                && c.n_layers_per_attn_res_block > 0
                && c.final_logit_softcapping > 0.0
                && c.rel_extent > 0
                && c.sliding_window_size > 0
                && c.intermediate_size > 0
                && c.intermediate_size % 128 == 0,
            "unsupported Inkling 2 configuration"
        );
        let hidden = c.hidden_size;
        let embeddings = half_weight(
            checkpoint,
            "model.llm.embed.weight",
            Some(&[c.vocab_size, hidden]),
        )?;
        let embed_norm = f32_weight(checkpoint, "model.llm.embed_norm.weight", &[hidden])?;
        let norm = f32_weight(checkpoint, "model.llm.norm.weight", &[hidden])?;
        let final_query = f32_weight(checkpoint, "model.llm.attn_res.output_query", &[hidden])?
            .reshape(&[1, hidden])?;
        // Checkpoint: [query, mlp] rows and [key, value] rows per loop layer -> [loop, 4 (q, k, v, mlp), D]
        let loop_layers = (lc.core_end - lc.core_start) as i32;
        let qm = f32_weight(
            checkpoint,
            "model.llm.middle_cycle_cross_pass_queries",
            &[2 * loop_layers, hidden],
        )?
        .reshape(&[loop_layers, 2, hidden])?;
        let kv = f32_weight(
            checkpoint,
            "model.llm.middle_cycle_cross_pass_kv_queries",
            &[2 * loop_layers, hidden],
        )?
        .reshape(&[loop_layers, 2, hidden])?;
        let message_queries =
            Array::concatenate(&[&qm.slice(1, 0, 1)?, &kv, &qm.slice(1, 1, 2)?], 1)?;
        message_queries.eval()?;
        let head = Projection::load(checkpoint, "model.llm.unembed", hidden, c.vocab_size, false)?;
        let head_q = if precision.head_bits < 16 {
            Some(Linear::quantized(
                &head.dense_weight().context("dense Inkling head")?.slice(
                    0,
                    0,
                    c.unpadded_vocab_size,
                )?,
                precision.head_bits,
            )?)
        } else {
            None
        };
        let layers = (0..c.num_hidden_layers)
            .map(|i| Layer::load(checkpoint, i, &c).with_context(|| format!("Inkling layer {i}")))
            .collect::<Result<Vec<_>>>()?;
        let mut visits: Vec<usize> = (0..lc.core_start).collect();
        for _ in 0..lc.num_passes {
            visits.extend(lc.core_start..lc.core_end);
        }
        visits.extend(lc.core_end..c.num_hidden_layers);
        let states = visits.iter().map(|_| VisitState::default()).collect();
        Ok(Self {
            profile: RunProfile::default(),
            pending: None,
            kernel: c.sconv_kernel_size,
            embeddings,
            embed_norm,
            norm,
            final_query,
            message_queries,
            head,
            head_q,
            precision,
            layers,
            visits,
            states,
            core: (lc.core_start, lc.core_end),
            passes: lc.num_passes,
            vocab: c.vocab_size,
            unpadded_vocab: c.unpadded_vocab_size,
            hidden,
            eps: c.rms_norm_eps,
            softcap: c.final_logit_softcapping,
            window: c.sliding_window_size,
            extent: c.rel_extent,
            context_limit: c.model_max_length,
            offset: 0,
        })
    }

    pub fn context_limit(&self) -> i32 {
        self.context_limit
    }

    pub fn take_profile(&mut self) -> RunProfile {
        std::mem::take(&mut self.profile)
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        self.pending = None;
        for state in &mut self.states {
            *state = VisitState::default();
        }
    }

    pub fn forward(&mut self, token: u32) -> Result<Array> {
        self.forward_tokens(&[token])
    }

    /// Logits `[1, unpadded_vocab]` FP32 of the last token.
    pub fn forward_tokens(&mut self, tokens: &[u32]) -> Result<Array> {
        match self.run(tokens, false, None, true) {
            Ok((logits, _)) => Ok(logits),
            Err(error) => {
                self.reset();
                Err(error.context("Inkling forward failed; its cache was reset"))
            }
        }
    }

    /// Logits `[T, unpadded_vocab]` FP32 of every token (parity checks).
    pub fn forward_all(&mut self, tokens: &[u32]) -> Result<Array> {
        match self.run(tokens, true, None, true) {
            Ok((logits, _)) => Ok(logits),
            Err(error) => {
                self.reset();
                Err(error.context("Inkling forward failed; its cache was reset"))
            }
        }
    }

    /// Normalized token embeddings `[n, D]` FP32 (`embed_norm(embed(ids))`).
    pub fn embed_tokens(&self, ids: &Array) -> Result<Array> {
        self.embeddings
            .take(ids, 0)?
            .astype(Dtype::Float32)?
            .rms_norm(&self.embed_norm, self.eps)
    }

    /// Unsoftcapped vocabulary scores `[n, unpadded_vocab]` of FP16 hidden rows.
    pub fn head_scores(&self, hidden: &Array) -> Result<Array> {
        let hidden = hidden.astype(Dtype::Float16)?;
        if ablated("head") {
            let rows = hidden.shape()[0];
            return hidden
                .slice(1, 0, 1)?
                .broadcast_to(&[rows, self.unpadded_vocab]);
        }
        match &self.head_q {
            Some(head) => head.forward(&hidden),
            None => self.head.forward(&hidden)?.slice(1, 0, self.unpadded_vocab),
        }
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// `[vocab, D]` FP16 output head (padded rows included).
    pub fn head_weight(&self) -> Option<&Array> {
        self.head.dense_weight()
    }

    pub fn logit_softcap(&self) -> f32 {
        self.softcap
    }

    pub fn unpadded_vocab(&self) -> i32 {
        self.unpadded_vocab
    }

    pub fn hidden_size(&self) -> i32 {
        self.hidden
    }

    /// Number of decoder-layer visits per token (the draft feature indices refer to these).
    pub fn visit_count(&self) -> usize {
        self.visits.len()
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Speculative verification: logits `[1, T, vocab]` of every row and the draft features `[T, taps * D]`
    /// (FP32 residual stream before each tapped visit, then the final normalized hidden for the index equal to
    /// `visit_count`). Nothing is committed until `commit`.
    pub fn verify(&mut self, tokens: &[u32], taps: &[usize]) -> Result<(Array, Array)> {
        ensure!(
            !taps.is_empty()
                && taps.windows(2).all(|w| w[0] < w[1])
                && *taps.last().unwrap_or(&0) == self.visits.len(),
            "Inkling draft taps must be increasing and end at the visit count"
        );
        match self.run(tokens, true, Some(taps), false) {
            Ok((logits, features)) => Ok((logits, features.context("features")?)),
            Err(error) => {
                self.reset();
                Err(error.context("Inkling verify failed; its cache was reset"))
            }
        }
    }

    /// Committed forward of `tokens` that also captures the drafter features of every row: the logits of
    /// the last row only (`[1, 1, vocab]`) and the features `[T, taps * D]`. Prefill of a DSpark session.
    pub fn forward_with_features(
        &mut self,
        tokens: &[u32],
        taps: &[usize],
    ) -> Result<(Array, Array)> {
        ensure!(
            !taps.is_empty()
                && taps.windows(2).all(|w| w[0] < w[1])
                && *taps.last().unwrap_or(&0) == self.visits.len(),
            "Inkling draft taps must be increasing and end at the visit count"
        );
        match self.run(tokens, false, Some(taps), true) {
            Ok((logits, features)) => Ok((logits, features.context("features")?)),
            Err(error) => {
                self.reset();
                Err(error.context("Inkling forward failed; its cache was reset"))
            }
        }
    }

    /// Keep the first `rows` rows of the last `verify` call (1 <= rows <= its length).
    pub fn commit(&mut self, rows: usize) -> Result<()> {
        let calls = self.pending.context("no pending Inkling verify")?;
        let rows = i32::try_from(rows).context("commit rows")?;
        ensure!(
            rows >= 1 && rows <= calls,
            "commit rows outside the verified call"
        );
        let result = self.commit_rows(rows, calls).and_then(|()| {
            let mut pending = Vec::new();
            for state in &self.states {
                pending.extend([&state.keys, &state.values].into_iter().flatten());
                pending.extend(state.conv.iter().flatten());
            }
            Array::eval_many(&pending)
        });
        if let Err(error) = result {
            self.reset();
            return Err(error.context("Inkling commit failed; its cache was reset"));
        }
        Ok(())
    }

    fn commit_rows(&mut self, rows: i32, calls: i32) -> Result<()> {
        for (visit, state) in self.states.iter_mut().enumerate() {
            let window = self.layers[self.visits[visit]].attention.window;
            state.commit(rows, calls, window, self.kernel)?;
        }
        self.offset += rows;
        self.pending = None;
        Ok(())
    }

    pub fn eval_state(&self) -> Result<()> {
        for state in &self.states {
            for a in [&state.keys, &state.values].into_iter().flatten() {
                a.eval()?;
            }
            for a in state.conv.iter().flatten() {
                a.eval()?;
            }
        }
        Ok(())
    }

    fn run(
        &mut self,
        tokens: &[u32],
        all: bool,
        taps: Option<&[usize]>,
        commit: bool,
    ) -> Result<(Array, Option<Array>)> {
        let started = Instant::now();
        ensure!(
            self.pending.is_none(),
            "Inkling has an uncommitted verify call"
        );
        ensure!(!tokens.is_empty(), "Inkling needs at least one token");
        ensure!(
            tokens.iter().all(|&t| t < self.vocab as u32),
            "Inkling token is outside vocabulary"
        );
        let time = i32::try_from(tokens.len()).context("Inkling batch is too large")?;
        let start = self.offset;
        let end = start
            .checked_add(time)
            .context("Inkling position overflow")?;
        ensure!(end <= self.context_limit, "Inkling context is full");
        let ids = Array::from_i32(
            &tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
            &[time],
        )?;
        let hidden = self
            .embeddings
            .take(&ids, 0)?
            .astype(Dtype::Float32)?
            .rms_norm(&self.embed_norm, self.eps)?;
        // Sliding visits keep at most window - 1 past keys; full visits keep everything
        let sliding_first = (start - (self.window - 1)).max(0);
        let sliding = MaskPlan::new(
            start,
            time,
            sliding_first,
            end - sliding_first,
            self.window,
            Some(self.window),
        )?;
        let full = MaskPlan::new(start, time, 0, end, self.extent, None)?;
        let (core_start, core_end) = self.core;
        let mut blocks: Vec<Array> = Vec::new();
        let mut hidden = hidden;
        let mut visit = 0usize;
        let mut features: Vec<Array> = Vec::new();
        let tapped = |visit: usize| taps.is_some_and(|t| t.contains(&visit));
        let layers = &self.layers;
        let states = &mut self.states;
        let mut run_layer = |layer: usize,
                             hidden: &Array,
                             blocks: &mut Vec<Array>,
                             messages: Option<&Array>,
                             visit: usize|
         -> Result<Array> {
            let l = &layers[layer];
            let plan = if l.sliding { &sliding } else { &full };
            l.forward(hidden, blocks, messages, start, plan, &mut states[visit])
        };
        for layer in 0..core_start {
            if tapped(visit) {
                features.push(hidden.try_clone()?);
            }
            hidden = run_layer(layer, &hidden, &mut blocks, None, visit)?;
            visit += 1;
        }
        let pre_hidden = hidden.try_clone()?;
        let pre_blocks = blocks
            .iter()
            .map(Array::try_clone)
            .collect::<Result<Vec<_>>>()?;
        for pass in 0..self.passes {
            let bank = if pass == 0 {
                None
            } else {
                let previous: Vec<Array> = blocks[pre_blocks.len() + 1..]
                    .iter()
                    .map(Array::try_clone)
                    .collect::<Result<_>>()?;
                let partial = hidden.rms_norm_without_weight(self.eps)?;
                let stacked = if lean() {
                    let mut sources: Vec<&Array> = previous.iter().collect();
                    sources.push(&partial);
                    Some(depth_stack(&sources)?)
                } else {
                    None
                };
                hidden = pre_hidden.try_clone()?;
                blocks = pre_blocks
                    .iter()
                    .map(Array::try_clone)
                    .collect::<Result<_>>()?;
                Some((previous, partial, stacked))
            };
            for (i, layer) in (core_start..core_end).enumerate() {
                let messages = match &bank {
                    Some((previous, partial, stacked)) => {
                        let queries = self
                            .message_queries
                            .slice(0, i as i32, i as i32 + 1)?
                            .reshape(&[4, self.hidden])?;
                        Some(match stacked {
                            Some(stacked) => depth_attend(stacked, &queries)?,
                            None => {
                                let mut sources: Vec<&Array> = previous.iter().collect();
                                sources.push(partial);
                                depth_attention(&sources, &queries)?
                            }
                        })
                    }
                    None => None,
                };
                if tapped(visit) {
                    features.push(hidden.try_clone()?);
                }
                hidden = run_layer(layer, &hidden, &mut blocks, messages.as_ref(), visit)?;
                visit += 1;
            }
        }
        for layer in core_end..layers.len() {
            if tapped(visit) {
                features.push(hidden.try_clone()?);
            }
            hidden = run_layer(layer, &hidden, &mut blocks, None, visit)?;
            visit += 1;
        }
        ensure!(visit == self.visits.len(), "Inkling visit count mismatch");
        let partial = hidden.rms_norm_without_weight(self.eps)?;
        let mut sources: Vec<&Array> = blocks.iter().collect();
        sources.push(&partial);
        let hidden = depth_attention(&sources, &self.final_query)?
            .reshape(&[time, self.hidden])?
            .rms_norm(&self.norm, self.eps)?;
        let features = match taps {
            Some(taps) => {
                features.push(hidden.try_clone()?);
                ensure!(
                    features.len() == taps.len(),
                    "Inkling draft taps not all reached"
                );
                Some(Array::concatenate(&features.iter().collect::<Vec<_>>(), 1)?)
            }
            None => None,
        };
        let hidden = if all {
            hidden
        } else {
            hidden.slice(0, time - 1, time)?
        };
        let rows = hidden.shape()[0];
        let logits = self
            .head_scores(&hidden)?
            .astype(Dtype::Float32)?
            .softcap(self.softcap)?;
        // Same layout as the other chat models: [1, rows, vocab]
        let logits = logits.reshape(&[1, rows, self.unpadded_vocab])?;
        if commit {
            self.pending = Some(time);
            self.commit_rows(time, time)?;
        } else {
            self.pending = Some(time);
        }
        let mut pending = vec![&logits];
        pending.extend(features.iter());
        for state in &self.states {
            if commit {
                pending.extend([&state.keys, &state.values].into_iter().flatten());
                pending.extend(state.conv.iter().flatten());
            } else {
                if let Some((k, v, _)) = &state.pending_kv {
                    pending.extend([k, v]);
                }
                pending.extend(state.pending_conv.iter().flatten());
            }
        }
        static GRAPH_STATS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *GRAPH_STATS.get_or_init(|| std::env::var_os("MLXL3_INKLING_GRAPH_STATS").is_some()) {
            eprintln!(
                "inkling graph rows={time}\n{}",
                Array::graph_stats(&pending)?
            );
        }
        let built = Instant::now();
        Array::eval_many(&pending)?;
        let done = Instant::now();
        self.profile.calls += 1;
        self.profile.build += built - started;
        self.profile.eval += done - built;
        debug_assert!(commit || self.offset == start);
        let _ = end;
        Ok((logits, features))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn precision_defaults_and_projection_sets() -> Result<()> {
        // generation default: exact projections, 8-bit head; numerical reference: everything exact
        assert_eq!(
            (Precision::DEFAULT.dense_bits, Precision::DEFAULT.head_bits),
            (16, 8)
        );
        assert_eq!(
            (Precision::FULL.dense_bits, Precision::FULL.head_bits),
            (16, 16)
        );
        assert_eq!(Precision::DEFAULT.dense_only, DenseSet::ALL);
        Precision::DEFAULT.validate()?;
        for bad in [0, 2, 3, 5, 12, 32] {
            let precision = Precision {
                head_bits: bad,
                ..Precision::FULL
            };
            assert!(precision.validate().is_err(), "head bits {bad} accepted");
            let precision = Precision {
                dense_bits: bad,
                ..Precision::FULL
            };
            assert!(precision.validate().is_err(), "dense bits {bad} accepted");
        }
        let set = DenseSet::parse("q+o,gate")?;
        for name in ["q", "o", "gate"] {
            assert!(set.contains(name));
        }
        for name in ["k", "v", "r", "sink", "head", ""] {
            assert!(!set.contains(name));
        }
        assert_eq!(DenseSet::parse("q,k,v,r,o,gate,sink")?, DenseSet::ALL);
        assert!(DenseSet::parse("").is_err());
        assert!(DenseSet::parse("q,unknown").is_err());
        Ok(())
    }

    /// Deterministic values in roughly [-scale, scale] with a wide spread of magnitudes.
    fn pseudo_random(count: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
                let magnitude = 10f32.powi(((state >> 8) % 5) as i32 - 3);
                unit * magnitude * scale
            })
            .collect()
    }

    /// The fused short-convolution kernel against the op sequence it replaces: same bits for the output and
    /// for the window (next state), for 1 to 300 rows, FP16 and FP32 inputs, with and without a state.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn short_conv_kernel_matches_ops() -> Result<()> {
        let kernel = 4;
        let mut checked = 0usize;
        for channels in [256, 1024, 2304] {
            let weight = Array::from_f32(
                &pseudo_random((channels * kernel) as usize, 7 + channels as u64, 0.7),
                &[channels, kernel, 1],
            )?;
            let conv = ShortConv {
                weight,
                scale: scalar(1.0 / ((kernel + 1) as f32).sqrt())?,
                channels,
                kernel,
            };
            for time in [1, 2, 3, 7, 26, 128, 300] {
                for dtype in [Dtype::Float16, Dtype::Float32] {
                    for with_state in [false, true] {
                        let seed = (channels * 1000 + time * 10 + i32::from(with_state)) as u64;
                        let x = Array::from_f32(
                            &pseudo_random((time * channels) as usize, seed, 50.0),
                            &[time, channels],
                        )?
                        .astype(dtype)?;
                        let state = Array::from_f32(
                            &pseudo_random(((kernel - 1) * channels) as usize, seed + 1, 3.0),
                            &[kernel - 1, channels],
                        )?;
                        let state = with_state.then_some(&state);
                        let (out, window) = conv.forward_kernel(&x, state)?;
                        let (expected, expected_window) = conv.forward_ops(&x, state)?;
                        ensure!(
                            out.shape() == [time, channels]
                                && window.shape() == [time + kernel - 1, channels]
                                && out.dtype() == Dtype::Float32
                                && window.dtype() == Dtype::Float32,
                            "kernel output layout"
                        );
                        let bits = |a: &Array| -> Result<Vec<u32>> {
                            Ok(a.to_f32()?.iter().map(|x| x.to_bits()).collect())
                        };
                        let (out, expected) = (bits(&out)?, bits(&expected)?);
                        ensure!(
                            out.len() == (time * channels) as usize && out == expected,
                            "output bits differ: channels {channels} rows {time} {dtype:?} state {with_state}"
                        );
                        ensure!(
                            bits(&window)? == bits(&expected_window)?,
                            "window bits differ: channels {channels} rows {time} {dtype:?} state {with_state}"
                        );
                        ensure!(
                            out.iter().any(|&b| b != 0)
                                && out.iter().all(|&b| f32::from_bits(b).is_finite()),
                            "degenerate comparison"
                        );
                        checked += 1;
                    }
                }
            }
        }
        ensure!(checked == 84, "expected 84 cases, ran {checked}");
        Ok(())
    }

    #[test]
    fn mask_plan_distances_window_and_extent() -> Result<()> {
        // two queries at positions 5 and 6 over keys 3..7, window 3, extent 2
        let plan = MaskPlan::new(5, 2, 3, 4, 2, Some(3))?;
        assert_eq!(
            plan.gather.astype(Dtype::Float32)?.to_f32()?,
            vec![1., 1., 0., 0., 3., 3., 3., 2.]
        );
        assert_eq!(
            plan.bias_valid.to_f32()?,
            vec![0., 1., 1., 0., 0., 0., 1., 1.]
        );
        let seen: Vec<bool> = plan
            .visible
            .to_f32()?
            .iter()
            .map(|x| x.is_finite())
            .collect();
        assert_eq!(seen, vec![true, true, true, false, false, true, true, true]);
        assert!(MaskPlan::new(0, 1, 0, 1, 4, None).is_ok());
        Ok(())
    }

    fn read_f32(path: &Path) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        let (words, rest) = bytes.as_chunks::<4>();
        ensure!(rest.is_empty(), "truncated oracle file");
        Ok(words.iter().map(|w| f32::from_le_bytes(*w)).collect())
    }

    /// Relative RMSE of the vocabulary-centered logits, per position.
    fn centered_rel_rmse(actual: &[f32], expected: &[f32]) -> f64 {
        let mean = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
        let (ma, me) = (mean(actual), mean(expected));
        let (mut num, mut den) = (0f64, 0f64);
        for (&a, &e) in actual.iter().zip(expected) {
            let (a, e) = (a as f64 - ma, e as f64 - me);
            num += (a - e) * (a - e);
            den += e * e;
        }
        (num / den.max(1e-30)).sqrt()
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    }

    /// Engine logits against a Transformers oracle written by the conversion driver
    /// (`oracle_to_bin.py`): one pass, chunked prefill and token-by-token decode.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ORACLE"]
    fn matches_oracle_logits() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let oracle = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_ORACLE")?);
        let limit: f64 = std::env::var("MLXL3_INKLING_MAX_RELRMSE")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(0.05);
        let chunk: usize = std::env::var("MLXL3_INKLING_CHUNK")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(7);
        let manifest: serde_json::Value =
            serde_json::from_reader(File::open(oracle.join("manifest.json"))?)?;
        let mut model = Inkling2::load_with(
            Path::new(&model_dir),
            Precision::from_env_or(Precision::FULL)?,
        )?;
        let mut worst = 0f64;
        let mut report = Vec::new();
        for (index, row) in manifest["rows"]
            .as_array()
            .context("rows")?
            .iter()
            .enumerate()
        {
            let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone())?;
            let tail = row["tail"].as_u64().context("tail")? as usize;
            let vocab = row["vocab"].as_u64().context("vocab")? as usize;
            let expected = read_f32(&oracle.join(row["file"].as_str().context("file")?))?;
            ensure!(
                expected.len() == tail * vocab && tail <= tokens.len(),
                "row {index}: bad oracle"
            );
            let start = tokens.len() - tail;
            let mut modes: Vec<(&str, Vec<Vec<f32>>)> = Vec::new();
            // one pass
            model.reset();
            let all = model.forward_all(&tokens)?;
            ensure!(
                all.shape() == [1, tokens.len() as i32, vocab as i32],
                "row {index}: logits shape"
            );
            let values = all.to_f32()?;
            modes.push((
                "full",
                (start..tokens.len())
                    .map(|t| values[t * vocab..(t + 1) * vocab].to_vec())
                    .collect(),
            ));
            // chunked prefill
            model.reset();
            let mut rows_out = Vec::new();
            for piece in tokens.chunks(chunk) {
                let out = model.forward_all(piece)?.to_f32()?;
                ensure!(
                    out.len() == piece.len() * vocab,
                    "row {index}: chunk logits"
                );
                rows_out.extend(out.chunks_exact(vocab).map(<[f32]>::to_vec));
            }
            modes.push(("chunked", rows_out[start..].to_vec()));
            // prefill then decode the tail token by token
            model.reset();
            let mut decoded = Vec::new();
            if start > 0 {
                let out = model.forward_all(&tokens[..start])?.to_f32()?;
                ensure!(out.len() == start * vocab, "row {index}: prefill logits");
            }
            for &token in &tokens[start..] {
                let out = model.forward(token)?;
                ensure!(
                    out.shape() == [1, 1, vocab as i32],
                    "row {index}: decode shape"
                );
                decoded.push(out.to_f32()?);
            }
            modes.push(("decode", decoded));
            if let Ok(dump) = std::env::var("MLXL3_INKLING_DUMP") {
                let dir = std::path::PathBuf::from(dump);
                std::fs::create_dir_all(&dir)?;
                for (mode, actual) in &modes {
                    let bytes: Vec<u8> = actual
                        .iter()
                        .flatten()
                        .flat_map(|x| x.to_le_bytes())
                        .collect();
                    std::fs::write(dir.join(format!("row{index}.{mode}.f32")), bytes)?;
                }
            }
            for (mode, actual) in &modes {
                let mut max_rel = 0f64;
                let mut agree = 0usize;
                for (t, logits) in actual.iter().enumerate() {
                    let want = &expected[t * vocab..(t + 1) * vocab];
                    ensure!(
                        logits.iter().all(|x| x.is_finite()),
                        "row {index} {mode}: non-finite logits"
                    );
                    max_rel = max_rel.max(centered_rel_rmse(logits, want));
                    agree += usize::from(argmax(logits) == argmax(want));
                }
                worst = worst.max(max_rel);
                report.push(
                    serde_json::json!({"row": index, "mode": mode, "positions": actual.len(),
                    "max_rel_rmse": max_rel, "argmax_agree": agree as f64 / actual.len() as f64}),
                );
            }
        }
        println!("{}", serde_json::to_string_pretty(&report)?);
        ensure!(
            worst <= limit,
            "max centered relative RMSE {worst} > {limit}"
        );
        Ok(())
    }

    fn fnv(hash: &mut u64, bytes: &[u8]) {
        for &b in bytes {
            *hash ^= u64::from(b);
            *hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }

    fn quantiles(mut v: Vec<f64>) -> (f64, f64, f64) {
        v.sort_by(f64::total_cmp);
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        (q(0.25), q(0.5), q(0.75))
    }

    /// Decode timing harness: fixed prompt, greedy decode, per-step construction/evaluation time and a
    /// FNV-1a fingerprint of every decoded FP32 logit and token (lossless-change gate).
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL"]
    fn decode_profile() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let var = |name: &str, default: usize| -> Result<usize> {
            Ok(std::env::var(name)
                .ok()
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(default))
        };
        let steps = var("MLXL3_INKLING_STEPS", 128)?;
        let reps = var("MLXL3_INKLING_REPS", 3)?;
        let prompt: Vec<u32> = match std::env::var("MLXL3_INKLING_ORACLE") {
            Ok(dir) => {
                let manifest: serde_json::Value =
                    serde_json::from_reader(File::open(Path::new(&dir).join("manifest.json"))?)?;
                serde_json::from_value(manifest["rows"][0]["tokens"].clone())?
            }
            Err(_) => vec![200002, 200004, 34182, 279],
        };
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let argmax_of = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i as u32)
        };
        let once = |model: &mut Inkling2, steps: usize| -> Result<serde_json::Value> {
            model.reset();
            model.take_profile();
            let t = Instant::now();
            let logits = model.forward_tokens(&prompt)?;
            let prefill = t.elapsed().as_secs_f64() * 1e3;
            let mut next = argmax_of(&logits.to_f32()?);
            let (mut total, mut build, mut eval) = (Vec::new(), Vec::new(), Vec::new());
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            let mut tokens = Vec::new();
            model.take_profile();
            for _ in 0..steps {
                let t = Instant::now();
                let logits = model.forward(next)?;
                total.push(t.elapsed().as_secs_f64() * 1e3);
                let p = model.take_profile();
                build.push(p.build.as_secs_f64() * 1e3);
                eval.push(p.eval.as_secs_f64() * 1e3);
                let values = logits.to_f32()?;
                for x in &values {
                    fnv(&mut hash, &x.to_le_bytes());
                }
                next = argmax_of(&values);
                fnv(&mut hash, &next.to_le_bytes());
                tokens.push(next);
            }
            let (t25, t50, t75) = quantiles(total);
            let (_, b50, _) = quantiles(build);
            let (_, e50, _) = quantiles(eval);
            Ok(
                serde_json::json!({"prompt_tokens": prompt.len(), "prefill_ms": prefill, "steps": steps,
                "decode_ms_p25": t25, "decode_ms_median": t50, "decode_ms_p75": t75,
                "build_ms_median": b50, "eval_ms_median": e50, "decode_tok_s": 1000.0 / t50,
                "fingerprint": format!("{hash:016x}"), "tokens_head": &tokens[..tokens.len().min(16)]}),
            )
        };
        once(&mut model, 16)?;
        let runs = (0..reps)
            .map(|_| once(&mut model, steps))
            .collect::<Result<Vec<_>>>()?;
        println!("{}", serde_json::to_string_pretty(&runs)?);
        ensure!(
            runs.iter()
                .all(|r| r["fingerprint"] == runs[0]["fingerprint"]),
            "decode is not deterministic across repetitions"
        );
        Ok(())
    }

    /// Speculative commit mechanics on a model where batched and single-row numerics agree (tiny export with
    /// decisive routing): verify blocks with arbitrary proposals, commit partial prefixes, cross the sliding
    /// window, then compare the next-token logits with a teacher-forced AR pass over the committed tokens.
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL (tiny stable export)"]
    fn commit_rollback_matches_teacher_forced_ar() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let mut model = Inkling2::load_with(
            Path::new(&model_dir),
            Precision::from_env_or(Precision::FULL)?,
        )?;
        let taps = vec![1, 5, model.visit_count()];
        let prompt: Vec<u32> = (0..9).map(|i| 1000 + 37 * i).collect();
        let mut committed = prompt.clone();
        model.reset();
        let (_, features) = model.verify(&prompt, &taps)?;
        assert_eq!(features.shape(), [9, 3 * model.hidden_size()]);
        model.commit(prompt.len())?;
        let mut next = 5000u32;
        let mut step = 0u32;
        while committed.len() < 150 {
            let width = 1 + (step % 7) as usize;
            let mut block = vec![next];
            block.extend((0..width).map(|i| 7000 + 13 * step + i as u32));
            let keep = 1 + (step as usize * 5) % (width + 1); // 1..=width+1 rows
            let (logits, features) = model.verify(&block, &taps)?;
            assert_eq!(
                logits.shape(),
                [1, block.len() as i32, model.unpadded_vocab()]
            );
            assert_eq!(
                features.shape(),
                [block.len() as i32, 3 * model.hidden_size()]
            );
            model.commit(keep)?;
            committed.extend(&block[..keep]);
            assert_eq!(model.offset() as usize, committed.len());
            next = 3000 + step * 11;
            step += 1;
        }
        // an uncommitted verify must be refused before the next call
        model.verify(&[next], &taps)?;
        assert!(model.forward(next).is_err());
        model.reset();
        model.forward_tokens(&committed[..prompt.len()])?;
        for &t in &committed[prompt.len()..] {
            model.forward(t)?;
        }
        let ar = model.forward(next)?.to_f32()?;
        // replay the speculative schedule (same rows committed, same order)
        model.reset();
        let (_, _) = model.verify(&prompt, &taps)?;
        model.commit(prompt.len())?;
        let mut cursor = prompt.len();
        let mut next2 = 5000u32;
        let mut step = 0u32;
        while cursor < committed.len() {
            let width = 1 + (step % 7) as usize;
            let mut block = vec![next2];
            block.extend((0..width).map(|i| 7000 + 13 * step + i as u32));
            let keep = 1 + (step as usize * 5) % (width + 1);
            model.verify(&block, &taps)?;
            model.commit(keep)?;
            cursor += keep;
            next2 = 3000 + step * 11;
            step += 1;
        }
        let spec = model.forward(next)?.to_f32()?;
        let rel = centered_rel_rmse(&spec, &ar);
        println!(
            "committed {} tokens over {step} blocks, rel {rel}",
            committed.len()
        );
        ensure!(
            rel < 0.05,
            "speculative state differs from teacher-forced AR: {rel}"
        );
        Ok(())
    }

    /// Verification cost by row count (diagnostic): median wall time of `verify` for T = 1..8 rows.
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL"]
    fn verify_cost_by_rows() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let taps: Vec<usize> = vec![5, 23, 41, model.visit_count()];
        let prompt: Vec<u32> = (0..26).map(|i| 1000 + 37 * i).collect();
        let mut rows = Vec::new();
        for t in 1..=8usize {
            let mut times = Vec::new();
            for rep in 0..6 {
                model.reset();
                model.forward_tokens(&prompt)?;
                let block: Vec<u32> = (0..t as u32).map(|i| 2000 + 17 * i + rep).collect();
                let start = Instant::now();
                model.verify(&block, &taps)?;
                times.push(start.elapsed().as_secs_f64() * 1e3);
                model.commit(t)?;
            }
            let (_, median, _) = quantiles(times[1..].to_vec());
            rows.push(serde_json::json!({"rows": t, "verify_ms_median": median}));
        }
        println!("{}", serde_json::to_string(&rows)?);
        Ok(())
    }

    /// Quality of the affine dense/head variants against the FP16 engine itself: KL(FP16 || variant) and top-1
    /// agreement on the tail positions of fixed token rows. One model alive at a time.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn dense_bits_kl() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let variants: Vec<Precision> = std::env::var("MLXL3_INKLING_VARIANTS")
            .unwrap_or_else(|_| "8-8,4-8,8-4,4-4".into())
            .split(',')
            .map(|v| {
                let (bits, only) = match v.split_once('@') {
                    Some((bits, only)) => (bits, DenseSet::parse(only)?),
                    None => (v, DenseSet::ALL),
                };
                let (d, h) = bits
                    .split_once('-')
                    .context("variant must be DENSE-HEAD[@proj+proj]")?;
                Ok(Precision {
                    dense_bits: d.parse()?,
                    head_bits: h.parse()?,
                    dense_only: only,
                })
            })
            .collect::<Result<_>>()?;
        let rows: Vec<(String, Vec<u32>, usize)> = rows["rows"]
            .as_array()
            .context("rows")?
            .iter()
            .map(|r| {
                Ok((
                    r["name"].as_str().context("name")?.to_string(),
                    serde_json::from_value(r["tokens"].clone())?,
                    r["tail"].as_u64().context("tail")? as usize,
                ))
            })
            .collect::<Result<_>>()?;
        // log-probabilities [tail, vocab] FP32 of every row
        let run = |precision: Precision| -> Result<Vec<Vec<f32>>> {
            let mut model = Inkling2::load_with(&dir, precision)?;
            ensure!(model.precision() == precision, "precision not applied");
            let vocab = model.unpadded_vocab() as usize;
            rows.iter()
                .map(|(name, tokens, tail)| {
                    ensure!(*tail >= 1 && *tail <= tokens.len(), "{name}: bad tail");
                    model.reset();
                    let rows = tokens.len() as i32;
                    let values = model
                        .forward_all(tokens)?
                        .reshape(&[rows, vocab as i32])?
                        .slice(0, rows - *tail as i32, rows)?
                        .log_probs()?
                        .to_f32()?;
                    ensure!(
                        values.len() == tail * vocab && values.iter().all(|x| x.is_finite()),
                        "{name}: empty or non-finite log-probabilities"
                    );
                    Ok(values)
                })
                .collect()
        };
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &x)| {
                    if x > best.1 { (i, x) } else { best }
                })
                .0
        };
        let reference = run(Precision::FULL)?;
        let mut report = Vec::new();
        for precision in variants {
            let actual = run(precision)?;
            let (mut total_kl, mut total_same, mut total_positions) = (0f64, 0usize, 0usize);
            let mut per_row = Vec::new();
            for (((name, _, tail), reference), actual) in rows.iter().zip(&reference).zip(&actual) {
                let vocab = reference.len() / tail;
                let (mut sum, mut worst, mut same) = (0f64, 0f64, 0usize);
                for (r, a) in reference
                    .chunks_exact(vocab)
                    .zip(actual.chunks_exact(vocab))
                {
                    let kl: f64 = r
                        .iter()
                        .zip(a)
                        .map(|(&r, &a)| (r as f64).exp() * (r as f64 - a as f64))
                        .sum();
                    sum += kl;
                    worst = worst.max(kl);
                    same += usize::from(argmax(r) == argmax(a));
                }
                per_row.push(serde_json::json!({"row": name, "positions": tail, "kl_mean": sum / *tail as f64,
                    "kl_max": worst, "top1": same as f64 / *tail as f64}));
                total_kl += sum;
                total_same += same;
                total_positions += tail;
            }
            report.push(serde_json::json!({
                "dense_bits": precision.dense_bits, "head_bits": precision.head_bits,
                "dense_only": format!("{:?}", precision.dense_only),
                "positions": total_positions, "kl_mean": total_kl / total_positions as f64,
                "top1": total_same as f64 / total_positions as f64, "rows": per_row,
            }));
        }
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }

    /// DIAG-2026-11-21-INK2M-LOCALREF. With `MLXL3_INKLING_EVAL_WRITE=1`: writes the logits of every
    /// `stride`-th position of each row (`MLXL3_INKLING_ROWS`) as FP32 under `MLXL3_INKLING_REF_DUMP`.
    /// Otherwise: KL(reference || model) and top-1 agreement of the loaded model against those files.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_INKLING_ROWS and MLXL3_INKLING_REF_DUMP"]
    fn eval_against_reference() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let dump = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_REF_DUMP")?);
        let write = std::env::var_os("MLXL3_INKLING_EVAL_WRITE").is_some();
        let stride: i32 = std::env::var("MLXL3_INKLING_STRIDE")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(4);
        let only: Option<String> = std::env::var("MLXL3_INKLING_EVAL_ROWS").ok();
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let mut model = Inkling2::load_with(&dir, Precision::from_env_or(Precision::FULL)?)?;
        let vocab = model.unpadded_vocab();
        if write {
            std::fs::create_dir_all(&dump)?;
        }
        let mut report = Vec::new();
        for row in rows["rows"].as_array().context("rows")? {
            let name = row["name"].as_str().context("name")?;
            if only
                .as_ref()
                .is_some_and(|list| !list.split(',').any(|n| n == name))
            {
                continue;
            }
            let domain = row["domain"].as_str().unwrap_or(name);
            let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone())?;
            let time = tokens.len() as i32;
            ensure!(
                stride >= 1 && time >= stride,
                "{name}: row shorter than the stride"
            );
            let positions: Vec<i32> = (stride - 1..time).step_by(stride as usize).collect();
            let count = positions.len() as i32;
            model.reset();
            let logits = model
                .forward_all(&tokens)?
                .reshape(&[time, vocab])?
                .take(&Array::from_i32(&positions, &[count])?, 0)?;
            let file = dump.join(format!("{name}.f32"));
            if write {
                let values = logits.to_f32()?;
                ensure!(
                    values.len() == (count * vocab) as usize
                        && values.iter().all(|x| x.is_finite()),
                    "{name}: empty or non-finite reference logits"
                );
                std::fs::write(
                    &file,
                    values
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )?;
                report.push(serde_json::json!({"row": name, "positions": count, "written": file}));
                continue;
            }
            let bytes =
                std::fs::read(&file).with_context(|| format!("reference {}", file.display()))?;
            ensure!(
                bytes.len() == (count * vocab) as usize * 4,
                "{name}: reference file does not match the row"
            );
            let reference = Array::from_bytes(&bytes, &[count, vocab], Dtype::Float32)?;
            drop(bytes);
            let lr = reference.log_probs()?;
            let lm = logits.log_probs()?;
            let kl = lr.exp()?.mul(&lr.sub(&lm)?)?.sum(1, false)?.to_f32()?;
            let same: Vec<u32> = reference.argmax()?.reshape(&[-1])?.to_u32()?;
            let ours: Vec<u32> = logits.argmax()?.reshape(&[-1])?.to_u32()?;
            ensure!(
                kl.len() == count as usize && kl.iter().all(|x| x.is_finite()),
                "{name}: invalid KL"
            );
            let agree = same.iter().zip(&ours).filter(|(a, b)| a == b).count();
            report.push(serde_json::json!({
                "row": name, "domain": domain, "positions": count,
                "kl_mean": kl.iter().map(|&x| x as f64).sum::<f64>() / count as f64,
                "kl_max": kl.iter().cloned().fold(0f32, f32::max),
                "top1": agree as f64 / count as f64, "kl": kl,
            }));
        }
        ensure!(!report.is_empty(), "no row evaluated");
        println!("{}", serde_json::to_string(&report)?);
        Ok(())
    }
}
