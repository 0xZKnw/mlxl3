//! DFLASH2-2026-11-24-INK2M-P0: the DFlash2 drafter of Inkling 2 Mini (`incoai/Inkling-2-Mini-DFlash2`, a
//! block-diffusion drafter with a candidate path selector), in MLX: 8-bit affine projections by default, FP16
//! with `MLXL3_INKLING_DFLASH2_BITS=16`.
//!
//! Reference (read, not executed): SGLang `f4284f29ba56` `python/sglang/srt/models/dflash.py`
//! (`DFlashGroupedConv`, `DFlashDecoderLayer`, `DFlashDraftModel`, `CandidateSelector`, `DFlash2DraftModel`) and
//! `speculative/dflash_worker_v2.py`, with the Inkling patch shipped in the model repository
//! (`sglang_inkling_dflash2.py`: DSpark's features, the real vocabulary for the candidates, the selector softcap
//! after the edge correction).
//!
//! Context: the target features of every committed position (`Inkling2::forward_with_features` with the drafter's
//! taps, `[T, taps * D]`) are projected by `fc` and `hidden_norm`, then each layer keeps their keys (`k_proj`,
//! per-head `k_norm`, RoPE at the position) and values (`v_proj`) over a sliding window. Block: the target
//! embeddings (`embed_norm(embed)`) of the anchor and of `block - 1` mask tokens at positions `P..P + block`, the
//! anchor at the position the target has not run yet; non-causal attention over the context window and the block.
//! Proposals: rows `1..block` through the target head, top-k candidates, the selector lattice and its greedy walk.
//! Only the proposals depend on this module; the target verifies them with its exact rules.

use crate::{
    array::{Array, Dtype},
    checkpoint::{self, Checkpoint},
    inkling2::{Inkling2, Linear},
    linear::checkpoint_array,
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{fs::File, path::Path};

#[derive(Clone, Debug, Deserialize)]
pub struct DflashConfig {
    pub block_size: usize,
    pub conv_group_size: i32,
    pub conv_kernel_size: i32,
    pub final_logit_softcapping: Option<f32>,
    pub mask_token_id: u32,
    pub selector_rank: i32,
    pub selector_softcap_mode: Option<String>,
    pub selector_top_k: i32,
    pub target_layer_ids: Vec<usize>,
    #[serde(default)]
    pub output_multiplier: Option<f32>,
}

#[derive(Clone, Debug, Deserialize)]
struct RopeParameters {
    rope_theta: f32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Dflash2Config {
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f32,
    pub sliding_window: i32,
    pub vocab_size: i32,
    rope_parameters: RopeParameters,
    pub dflash_config: DflashConfig,
}

impl Dflash2Config {
    pub fn rope_theta(&self) -> f32 {
        self.rope_parameters.rope_theta
    }
}

/// Grouped dynamic K-tap convolution across a block (`DFlashGroupedConv`): `base [2, taps, H]` FP32 (input and
/// output sides) plus a per-row kernel offset projected from the sublayer's normalized input.
struct GroupedConv {
    base: Array,
    projection: Linear,
    taps: i32,
    groups: i32,
    group: i32,
}

impl GroupedConv {
    fn load(ckpt: &Checkpoint, prefix: &str, c: &Dflash2Config, bits: i32) -> Result<Self> {
        let h = c.hidden_size;
        let taps = c.dflash_config.conv_kernel_size;
        let group = c.dflash_config.conv_group_size;
        ensure!(
            taps >= 1 && group > 0 && h % group == 0,
            "invalid DFlash2 convolution geometry"
        );
        let groups = h / group;
        let base = checkpoint_array(ckpt, &format!("{prefix}.base_kernel"))?;
        ensure!(
            base.shape() == [2, taps, h],
            "{prefix}.base_kernel: shape {:?}",
            base.shape()
        );
        let base = base.astype(Dtype::Float32)?;
        base.eval()?;
        Ok(Self {
            base,
            projection: Linear::load(
                ckpt,
                &format!("{prefix}.kernel_projection"),
                h,
                2 * taps * groups,
                bits,
            )?,
            taps,
            groups,
            group,
        })
    }

    /// The per-row offsets of both sides `[rows, 2, taps, groups]` FP32 from the sublayer input `x` (`[rows, H]`).
    fn coefficients(&self, x: &Array) -> Result<Array> {
        let rows = x.shape()[0];
        self.projection
            .forward(&x.astype(Dtype::Float16)?)?
            .astype(Dtype::Float32)?
            .reshape(&[rows, 2, self.taps, self.groups])
    }

    /// P3 (2): `convolve` of `side`, added to `residual` when given (the output side), in one Metal kernel; the
    /// MLX composition under `MLXL3_INKLING_DFLASH2_CONV=0`. `[rows, H]` FP32.
    fn apply(
        &self,
        x: &Array,
        coefficients: &Array,
        side: i32,
        residual: Option<&Array>,
    ) -> Result<Array> {
        if !conv_kernel_enabled() {
            let out = self.convolve(x, coefficients, side)?;
            return match residual {
                Some(r) => r.add(&out),
                None => Ok(out),
            };
        }
        let rows = x.shape()[0];
        let h = self.groups * self.group;
        ensure!(
            x.shape() == [rows, h]
                && coefficients.shape() == [rows, 2, self.taps, self.groups]
                && (0..2).contains(&side)
                && residual.is_none_or(|r| r.shape() == [rows, h]),
            "invalid DFlash2 convolution operands"
        );
        #[cfg(test)]
        let mutated = CONV_MUTATION.with(std::cell::Cell::get);
        #[cfg(not(test))]
        let mutated = false;
        let finish = u8::from(residual.is_some());
        let header = format!(
            "#define ROWS {rows}\n#define HIDDEN {h}\n#define GROUP {}\n#define GROUPS {}\n#define TAPS {}\n#define SIDE {side}\n#define FINISH {finish}\n#define MUTATED {}\n",
            self.group,
            self.groups,
            self.taps,
            u8::from(mutated)
        );
        let name = format!(
            "mlxl3_inkling_dflash2_conv_r{rows}_h{h}_g{}_t{}_s{side}_f{finish}_m{}_v1",
            self.group,
            self.taps,
            u8::from(mutated)
        );
        let x = x.astype(Dtype::Float32)?;
        let coef = coefficients.astype(Dtype::Float32)?;
        let residual = residual.map(|r| r.astype(Dtype::Float32)).transpose()?;
        let mut names = vec!["x", "coef", "base"];
        let mut inputs = vec![&x, &coef, &self.base];
        if let Some(r) = &residual {
            names.push("residual");
            inputs.push(r);
        }
        crate::array::metal_kernel(
            &name,
            &names,
            &["out"],
            &header,
            GROUPED_CONV_SOURCE,
            &inputs,
            &[vec![rows, h]],
            &[Dtype::Float32],
            [rows * h, 1, 1],
            [256, 1, 1],
        )?
        .pop()
        .context("grouped convolution output")
    }

    /// `_grouped_conv`: out[i] = sum_t (base[side, t] + delta[i, t]) * x[i - t], rows before the block's first
    /// taken as zero (the block starts at row 0). `x` `[rows, H]`, `coefficients` from `coefficients`.
    fn convolve(&self, x: &Array, coefficients: &Array, side: i32) -> Result<Array> {
        let rows = x.shape()[0];
        let blocks = x
            .astype(Dtype::Float32)?
            .reshape(&[rows, self.groups, self.group])?;
        let delta =
            coefficients
                .slice(1, side, side + 1)?
                .reshape(&[rows, self.taps, self.groups, 1])?;
        let base = self.base.slice(0, side, side + 1)?.reshape(&[
            1,
            self.taps,
            self.groups,
            self.group,
        ])?;
        let coef = base.add(&delta)?;
        let mut out = coef
            .slice(1, 0, 1)?
            .reshape(&[rows, self.groups, self.group])?
            .mul(&blocks)?;
        for tap in 1..self.taps {
            if tap >= rows {
                break;
            }
            #[cfg(test)]
            if CONV_MUTATION.with(std::cell::Cell::get) {
                // mutation: the tap reads the same row instead of the one `tap` rows before
                out = out.add(
                    &coef
                        .slice(1, tap, tap + 1)?
                        .reshape(&[rows, self.groups, self.group])?
                        .mul(&blocks)?,
                )?;
                continue;
            }
            let zeros = Array::zeros_dtype(&[tap, self.groups, self.group], Dtype::Float32)?;
            let shifted = Array::concatenate(&[&zeros, &blocks.slice(0, 0, rows - tap)?], 0)?;
            out = out.add(
                &coef
                    .slice(1, tap, tap + 1)?
                    .reshape(&[rows, self.groups, self.group])?
                    .mul(&shifted)?,
            )?;
        }
        out.reshape(&[rows, self.groups * self.group])
    }
}

#[cfg(test)]
thread_local! {
    /// Oracle mutation: the convolution taps read the current row (both paths).
    static CONV_MUTATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// DFLASH2-2026-11-24-INK2M-P3 (2): one thread per output value of `_grouped_conv`, optionally added to the
/// residual (`FINISH`): out = [residual +] sum_t (base[side, t, c] + delta[row, side, t, g(c)]) * x[row - t, c],
/// rows before the block's first taken as zero. `MUTATED` (oracle mutation only): every tap reads the row itself.
const GROUPED_CONV_SOURCE: &str = r#"
    const uint i = thread_position_in_grid.x;
    if (i >= uint(ROWS) * uint(HIDDEN)) {
        return;
    }
    const uint row = i / uint(HIDDEN);
    const uint c = i % uint(HIDDEN);
    const uint g = c / uint(GROUP);
    float acc = 0.0f;
    for (uint t = 0; t < uint(TAPS); ++t) {
        if (row >= t) {
            const float k = base[(uint(SIDE) * uint(TAPS) + t) * uint(HIDDEN) + c]
                + coef[((row * 2u + uint(SIDE)) * uint(TAPS) + t) * uint(GROUPS) + g];
#if MUTATED
            acc += k * x[row * uint(HIDDEN) + c];
#else
            acc += k * x[(row - t) * uint(HIDDEN) + c];
#endif
        }
    }
#if FINISH
    out[i] = residual[i] + acc;
#else
    out[i] = acc;
#endif
"#;

/// P3 (2): the convolution kernel unless `MLXL3_INKLING_DFLASH2_CONV=0` (MLX composition).
fn conv_kernel_enabled() -> bool {
    std::env::var("MLXL3_INKLING_DFLASH2_CONV").as_deref() != Ok("0")
}

struct Layer {
    input_norm: Array,
    post_norm: Array,
    attention_conv: GroupedConv,
    mlp_conv: GroupedConv,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    q_norm: Array,
    k_norm: Array,
    gate: Linear,
    up: Linear,
    down: Linear,
}

/// Context keys (normalized, rotated) and values of one layer `[1, KV, S, D]` FP16 over the window, and the
/// absolute position of the next context row (the DSpark cache type, so D1 snapshots both drafters alike).
type Cache = crate::inkling2_draft::DraftCache;

pub struct Inkling2Dflash2 {
    config: Dflash2Config,
    fc: Linear,
    hidden_norm: Array,
    layers: Vec<Layer>,
    norm: Array,
    selector_projection: Linear,
    /// `[vocab, rank]` BF16 codebooks of the selector (as stored; the GPU selector reads them so).
    predecessor: Array,
    successor: Array,
    /// P2: the bit width of the projections (8 = MLX affine, the default; 16 = FP16; `MLXL3_INKLING_DFLASH2_BITS`).
    bits: i32,
    caches: std::cell::RefCell<Vec<Cache>>,
    /// DFLASH2-2026-11-24-INK2M-S4: committed features `[n, taps * D]` not added to `caches` yet; they are added
    /// in one call (one projection over their rows, one concatenation per cache) before the caches are read.
    pending: std::cell::RefCell<Vec<Array>>,
    /// S4: whether `append_context` defers its features (`MLXL3_INKLING_DFLASH2_BATCH_CONTEXT=0`: adds them at once).
    batch_context: bool,
    /// S4b test variant: deferred features submitted for evaluation as they are added (`async_eval`).
    eager_features: bool,
    /// DFLASH2-2026-11-25-INK2M-F1: every request runs the trained block of `block_size` rows and reads its first
    /// `count` masked rows (`MLXL3_INKLING_DFLASH2_FULL_BLOCK=0`: a block of `count + 1` rows, measured 8 to 14
    /// points of acceptance lower on the first proposal for 3 and 1 proposals).
    full_block: bool,
    /// The target's real vocabulary (candidates are taken below it).
    valid_vocab: i32,
}

/// F1 setting of `MLXL3_INKLING_DFLASH2_FULL_BLOCK`: the whole trained block unless "0".
pub(crate) fn full_block_setting(value: Option<&str>) -> bool {
    value != Some("0")
}

/// S4 setting of `MLXL3_INKLING_DFLASH2_BATCH_CONTEXT`: batched unless "0".
pub(crate) fn batch_context_setting(value: Option<&str>) -> bool {
    value != Some("0")
}

fn f32_vector(ckpt: &Checkpoint, name: &str, len: i32) -> Result<Array> {
    let w = checkpoint_array(ckpt, name)?;
    ensure!(w.shape() == [len], "{name}: shape {:?}", w.shape());
    let w = w.astype(Dtype::Float32)?;
    w.eval()?;
    Ok(w)
}

/// The block outputs of `Inkling2Dflash2::propose_debug`: final hidden rows `[block, H]` FP32, candidate ids
/// `[n, k]`, unary values `[n, k]`, lattice scores `[n, k, k]` and the proposals.
pub struct Dflash2Block {
    pub hidden: Array,
    pub head_logits: Array,
    pub candidates: Vec<u32>,
    pub scores: Vec<f32>,
    pub proposals: Vec<u32>,
}

impl Inkling2Dflash2 {
    pub fn load(dir: &Path, target: &Inkling2) -> Result<Self> {
        let config: Dflash2Config = serde_json::from_reader(File::open(dir.join("config.json"))?)?;
        let d = &config.dflash_config;
        let h = config.hidden_size;
        ensure!(
            h == target.hidden_size()
                && d.target_layer_ids.last() == Some(&target.visit_count())
                && d.target_layer_ids.windows(2).all(|w| w[0] < w[1])
                && d.block_size >= 2
                && config.head_dim * config.num_attention_heads > 0
                && config.num_attention_heads % config.num_key_value_heads == 0
                && d.mask_token_id < target.unpadded_vocab() as u32
                && d.selector_top_k >= 1
                && matches!(
                    d.selector_softcap_mode.as_deref(),
                    None | Some("before_correction") | Some("after_correction")
                ),
            "unsupported Inkling 2 DFlash2 configuration"
        );
        let path = dir.join("model.safetensors");
        let ckpt = Checkpoint {
            path: dir.to_path_buf(),
            model_type: "inkling2_dflash2".into(),
            bits: None,
            size_bytes: std::fs::metadata(&path)?.len(),
            modules: Vec::new(),
            tensors: checkpoint::read_header(&path)?,
        };
        let (q_dim, kv_dim) = (
            config.num_attention_heads * config.head_dim,
            config.num_key_value_heads * config.head_dim,
        );
        let taps = d.target_layer_ids.len() as i32;
        // DFLASH2-2026-11-24-INK2M-P2 (b): 8-bit affine projections by default (AL 0.9986 x FP16;
        // `MLXL3_INKLING_DFLASH2_BITS=16`: FP16)
        let bits = crate::inkling2::bits_from_env("MLXL3_INKLING_DFLASH2_BITS", 8)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("layers.{i}");
            let lin = |name: &str, input: i32, output: i32| {
                Linear::load(&ckpt, &format!("{p}.{name}"), input, output, bits)
            };
            layers.push(Layer {
                input_norm: f32_vector(&ckpt, &format!("{p}.input_layernorm.weight"), h)?,
                post_norm: f32_vector(&ckpt, &format!("{p}.post_attention_layernorm.weight"), h)?,
                attention_conv: GroupedConv::load(
                    &ckpt,
                    &format!("{p}.attention_conv"),
                    &config,
                    bits,
                )?,
                mlp_conv: GroupedConv::load(&ckpt, &format!("{p}.mlp_conv"), &config, bits)?,
                q: lin("self_attn.q_proj", h, q_dim)?,
                k: lin("self_attn.k_proj", h, kv_dim)?,
                v: lin("self_attn.v_proj", h, kv_dim)?,
                o: lin("self_attn.o_proj", q_dim, h)?,
                q_norm: f32_vector(
                    &ckpt,
                    &format!("{p}.self_attn.q_norm.weight"),
                    config.head_dim,
                )?,
                k_norm: f32_vector(
                    &ckpt,
                    &format!("{p}.self_attn.k_norm.weight"),
                    config.head_dim,
                )?,
                gate: lin("mlp.gate_proj", h, config.intermediate_size)?,
                up: lin("mlp.up_proj", h, config.intermediate_size)?,
                down: lin("mlp.down_proj", config.intermediate_size, h)?,
            });
        }
        let codebook = |name: &str| -> Result<Array> {
            let w = checkpoint_array(&ckpt, name)?;
            ensure!(
                w.shape() == [config.vocab_size, d.selector_rank],
                "{name}: shape {:?}",
                w.shape()
            );
            let w = w.astype(Dtype::BFloat16)?;
            w.eval()?;
            Ok(w)
        };
        let draft = Self {
            fc: Linear::load(&ckpt, "fc", taps * h, h, bits)?,
            hidden_norm: f32_vector(&ckpt, "hidden_norm.weight", h)?,
            norm: f32_vector(&ckpt, "norm.weight", h)?,
            selector_projection: Linear::load(
                &ckpt,
                "candidate_selector.hidden_projection",
                h,
                d.selector_rank,
                bits,
            )?,
            predecessor: codebook("candidate_selector.predecessor_codebook")?,
            successor: codebook("candidate_selector.successor_codebook")?,
            caches: std::cell::RefCell::new(vec![Cache::default(); config.num_hidden_layers]),
            pending: std::cell::RefCell::new(Vec::new()),
            batch_context: batch_context_setting(
                std::env::var("MLXL3_INKLING_DFLASH2_BATCH_CONTEXT")
                    .ok()
                    .as_deref(),
            ),
            eager_features: false,
            full_block: full_block_setting(
                std::env::var("MLXL3_INKLING_DFLASH2_FULL_BLOCK")
                    .ok()
                    .as_deref(),
            ),
            valid_vocab: target.unpadded_vocab(),
            bits,
            layers,
            config,
        };
        crate::array::clear_cache()?;
        Ok(draft)
    }

    pub fn config(&self) -> &Dflash2Config {
        &self.config
    }

    pub fn taps(&self) -> &[usize] {
        &self.config.dflash_config.target_layer_ids
    }

    pub fn max_proposals(&self) -> usize {
        self.config.dflash_config.block_size - 1
    }

    pub fn reset(&mut self) {
        for cache in self.caches.get_mut() {
            *cache = Cache::default();
        }
        self.pending.get_mut().clear();
    }

    /// Absolute position of the next context row (deferred rows included).
    pub fn offset(&self) -> i32 {
        let pending: i32 = self.pending.borrow().iter().map(|f| f.shape()[0]).sum();
        self.caches.borrow().first().map_or(0, |c| c.offset) + pending
    }

    /// S4: whether context features are deferred until the caches are read.
    pub fn batches_context(&self) -> bool {
        self.batch_context
    }

    /// S4 test setting: defer the context features (`true`) or add them at once.
    #[cfg(test)]
    pub(crate) fn set_batch_context(&mut self, enabled: bool) -> Result<()> {
        self.flush_context()?;
        self.batch_context = enabled;
        Ok(())
    }

    /// F1 test setting: the whole trained block (`true`) or a block of `count + 1` rows.
    #[cfg(test)]
    pub(crate) fn set_full_block(&mut self, enabled: bool) {
        self.full_block = enabled;
    }

    /// F1: the rows of the block a request of `count` proposals runs.
    fn block_rows(&self, count: usize) -> usize {
        if self.full_block {
            self.config.dflash_config.block_size
        } else {
            count + 1
        }
    }

    /// S4b test setting: submit deferred features for evaluation as they are added.
    #[cfg(test)]
    pub(crate) fn set_eager_features(&mut self, enabled: bool) {
        self.eager_features = enabled;
    }

    /// D1: the context caches now (appends concatenate into new arrays; views copied to their own rows).
    pub fn context_snapshot(&self) -> Result<crate::inkling2_draft::DraftContext> {
        self.flush_context()?;
        Ok(crate::inkling2_draft::DraftContext {
            caches: self
                .caches
                .borrow()
                .iter()
                .map(Cache::compacted)
                .collect::<Result<_>>()?,
        })
    }

    /// D1: back to the context caches of `saved` (from this drafter).
    pub fn restore_context(&mut self, saved: &crate::inkling2_draft::DraftContext) -> Result<()> {
        ensure!(
            saved.caches.len() == self.caches.get_mut().len(),
            "DFlash2 context of another drafter"
        );
        *self.caches.get_mut() = saved.caches.clone();
        self.pending.get_mut().clear();
        Ok(())
    }

    fn heads(&self, x: &Array, heads: i32, norm: Option<&Array>) -> Result<Array> {
        let rows = x.shape()[0];
        let d = self.config.head_dim;
        let x = x.reshape(&[rows, heads, d])?;
        let x = match norm {
            Some(w) => x
                .astype(Dtype::Float32)?
                .rms_norm(w, self.config.rms_norm_eps)?,
            None => x.astype(Dtype::Float32)?,
        };
        x.astype(Dtype::Float16)?
            .transpose(&[1, 0, 2])?
            .reshape(&[1, heads, rows, d])
    }

    /// Feeds committed target features `[T, taps * D]` (token order, positions `offset..offset + T`). S4: deferred
    /// (added with the next ones in one call before the caches are read) unless batching is off.
    pub fn append_context(&mut self, features: &Array) -> Result<()> {
        let rows = features.shape()[0];
        let h = self.config.hidden_size;
        ensure!(
            features.shape() == [rows, self.taps().len() as i32 * h],
            "DFlash2 context features shape {:?}",
            features.shape()
        );
        if self.batch_context {
            if self.eager_features {
                features.async_eval()?;
            }
            self.pending.get_mut().push(features.clone());
            return Ok(());
        }
        self.add_context(&mut self.caches.borrow_mut(), features)
    }

    /// S4: adds the deferred features to the caches in one call (no-op without any).
    pub fn flush_context(&self) -> Result<()> {
        let pending = std::mem::take(&mut *self.pending.borrow_mut());
        let features = match pending.len() {
            0 => return Ok(()),
            1 => pending.into_iter().next().context("deferred features")?,
            _ => Array::concatenate(&pending.iter().collect::<Vec<_>>(), 0)?,
        };
        self.add_context(&mut self.caches.borrow_mut(), &features)
    }

    /// The context computation of `features` `[T, taps * D]` added to `caches`.
    fn add_context(&self, caches: &mut [Cache], features: &Array) -> Result<()> {
        let eps = self.config.rms_norm_eps;
        let context = self
            .fc
            .forward(&features.astype(Dtype::Float16)?)?
            .astype(Dtype::Float32)?
            .rms_norm(&self.hidden_norm, eps)?
            .astype(Dtype::Float16)?;
        let window = self.config.sliding_window;
        let theta = self.config.rope_theta();
        let (kv, d) = (self.config.num_key_value_heads, self.config.head_dim);
        let rows = features.shape()[0];
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            let k = {
                let rows = context.shape()[0];
                layer
                    .k
                    .forward(&context)?
                    .reshape(&[rows, kv, d])?
                    .astype(Dtype::Float32)?
                    .rms_norm(&layer.k_norm, eps)?
                    .astype(Dtype::Float16)?
                    .transpose(&[1, 0, 2])?
                    .reshape(&[1, kv, rows, d])?
                    .rope(d, theta, cache.offset)?
            };
            let v = layer
                .v
                .forward(&context)?
                .reshape(&[rows, kv, d])?
                .transpose(&[1, 0, 2])?
                .reshape(&[1, kv, rows, d])?;
            let (mut keys, mut values) = match (&cache.keys, &cache.values) {
                (Some(keys), Some(values)) => (
                    Array::concatenate(&[keys, &k], 2)?,
                    Array::concatenate(&[values, &v], 2)?,
                ),
                _ => (k, v),
            };
            let total = keys.shape()[2];
            if total > window {
                keys = keys.slice(2, total - window, total)?;
                values = values.slice(2, total - window, total)?;
            }
            cache.keys = Some(keys);
            cache.values = Some(values);
            cache.offset += rows;
        }
        Ok(())
    }

    /// Final hidden rows `[rows, H]` FP32 of the block `[anchor, mask x (rows - 1)]` whose embeddings are `embeds`
    /// (`[rows, H]` FP32), at positions `offset..offset + rows` (`DFlashDraftModel.forward`).
    pub fn block_hidden(&self, embeds: &Array) -> Result<Array> {
        let rows = embeds.shape()[0];
        let c = &self.config;
        ensure!(
            rows >= 2
                && rows as usize <= c.dflash_config.block_size
                && embeds.shape()[1] == c.hidden_size,
            "invalid DFlash2 block"
        );
        let eps = c.rms_norm_eps;
        self.flush_context()?;
        let caches = self.caches.borrow();
        let position = caches.first().map_or(0, |c| c.offset);
        let theta = c.rope_theta();
        let (heads, kv, d) = (c.num_attention_heads, c.num_key_value_heads, c.head_dim);
        let mut residual = embeds.astype(Dtype::Float32)?;
        let mut hidden = residual.rms_norm(&self.layers[0].input_norm, eps)?;
        let mut pending: Option<(usize, Array, Array)> = None; // (layer, mlp output, its coefficients)
        // P3 (1): every layer holds the same context rows (`append_context` feeds them all), so the sliding-window
        // mask is built once per block. The query at position + j sees keys at positions > position + j - window;
        // context index i holds position (position - context + i), excluded when i <= j + context - window.
        let context = caches[0].keys.as_ref().map_or(0, |k| k.shape()[2]);
        let window_mask = if context + rows > c.sliding_window {
            let mut mask = vec![0f32; (rows * (context + rows)) as usize];
            for j in 0..rows {
                let end = (j + context - c.sliding_window + 1).clamp(0, context);
                for i in 0..end {
                    mask[(j * (context + rows) + i) as usize] = f32::NEG_INFINITY;
                }
            }
            Some(Array::from_f32(&mask, &[1, 1, rows, context + rows])?.astype(Dtype::Float16)?)
        } else {
            None
        };
        for (index, (layer, cache)) in self.layers.iter().zip(caches.iter()).enumerate() {
            if let Some((prev, out, coefficients)) = pending.take() {
                residual =
                    self.layers[prev]
                        .mlp_conv
                        .apply(&out, &coefficients, 1, Some(&residual))?;
                hidden = residual.rms_norm(&layer.input_norm, eps)?;
            }
            // attention
            let coefficients = layer.attention_conv.coefficients(&hidden)?;
            let x = layer
                .attention_conv
                .apply(&hidden, &coefficients, 0, None)?
                .astype(Dtype::Float16)?;
            let q = self
                .heads(&layer.q.forward(&x)?, heads, Some(&layer.q_norm))?
                .rope(d, theta, position)?;
            let k = self
                .heads(&layer.k.forward(&x)?, kv, Some(&layer.k_norm))?
                .rope(d, theta, position)?;
            let v = self.heads(&layer.v.forward(&x)?, kv, None)?;
            let (keys, values, layer_context) = match (&cache.keys, &cache.values) {
                (Some(keys), Some(values)) => (
                    Array::concatenate(&[keys, &k], 2)?,
                    Array::concatenate(&[values, &v], 2)?,
                    keys.shape()[2],
                ),
                _ => (k, v, 0),
            };
            ensure!(
                layer_context == context,
                "DFlash2 layer {index}: {layer_context} context rows, layer 0 has {context}"
            );
            let scale = (d as f32).powf(-0.5);
            let attended = match &window_mask {
                Some(mask) => Array::sdpa_mask(&q, &keys, &values, scale, mask)?,
                None => Array::sdpa(&q, &keys, &values, scale, false)?,
            };
            let attended = attended
                .reshape(&[heads, rows, d])?
                .transpose(&[1, 0, 2])?
                .reshape(&[rows, heads * d])?;
            let out = layer.o.forward(&attended)?.astype(Dtype::Float32)?;
            residual = layer
                .attention_conv
                .apply(&out, &coefficients, 1, Some(&residual))?;
            hidden = residual.rms_norm(&layer.post_norm, eps)?;
            // MLP
            let coefficients = layer.mlp_conv.coefficients(&hidden)?;
            let x = layer
                .mlp_conv
                .apply(&hidden, &coefficients, 0, None)?
                .astype(Dtype::Float16)?;
            let out = layer
                .down
                .forward(&layer.gate.forward(&x)?.swiglu(&layer.up.forward(&x)?)?)?
                .astype(Dtype::Float32)?;
            pending = Some((index, out, coefficients));
        }
        let (prev, out, coefficients) = pending.context("DFlash2 without layers")?;
        residual = self.layers[prev]
            .mlp_conv
            .apply(&out, &coefficients, 1, Some(&residual))?;
        residual.rms_norm(&self.norm, eps)
    }

    /// Proposals after `anchor` (the token the target has not run yet, at position `offset()`): the block of
    /// `count + 1` rows, rows `1..=count` through the target head, top-k candidates, the selector lattice and its
    /// greedy walk. Also returns the intermediate values (oracle checks).
    pub fn propose_debug(
        &self,
        target: &Inkling2,
        anchor: u32,
        count: usize,
    ) -> Result<Dflash2Block> {
        let d = &self.config.dflash_config;
        ensure!(
            count >= 1 && count < d.block_size,
            "DFlash2 proposal count outside the trained block"
        );
        let rows = count as i32 + 1;
        // F1: the trained block; its first `count` masked rows are read
        let block_rows = self.block_rows(count);
        let mut ids = vec![d.mask_token_id; block_rows];
        ids[0] = anchor;
        let embeds = target.embed_tokens(&Array::from_u32(&ids, &[block_rows as i32])?)?;
        let hidden = self.block_hidden(&embeds)?;
        let predicted = hidden.slice(0, 1, rows)?;
        let head_logits = target
            .head_scores(&predicted.astype(Dtype::Float16)?)?
            .astype(Dtype::Float32)?
            .slice(1, 0, self.valid_vocab)?;
        let k = d.selector_top_k;
        let n = count as i32;
        // top-k: the k largest per row (ties: the order of argsort)
        let order = head_logits
            .negative()?
            .argsort()?
            .slice(1, 0, k)?
            .to_u32()?;
        let flat: Vec<u32> = order
            .iter()
            .enumerate()
            .map(|(i, &id)| id + (i / k as usize) as u32 * self.valid_vocab as u32)
            .collect();
        let mut unary = head_logits
            .reshape(&[n * self.valid_vocab])?
            .take(&Array::from_u32(&flat, &[n * k])?, 0)?
            .reshape(&[n, k])?;
        let multiplier = d.output_multiplier.unwrap_or(1.0);
        if multiplier != 1.0 {
            unary = unary.scalar_mul(multiplier)?;
        }
        let after = d.selector_softcap_mode.as_deref() == Some("after_correction");
        if let (Some(cap), false) = (d.final_logit_softcapping, after) {
            unary = unary.softcap(cap)?;
        }
        let candidates = Array::from_u32(&order, &[n, k])?;
        // lattice: score[e, p, c] = unary[e, c] + <A[pred[e, p]] * proj(h[e]), B[cand[e, c]]>
        let rank = d.selector_rank;
        let projected = self
            .selector_projection
            .forward(&predicted.astype(Dtype::Float16)?)?
            .astype(Dtype::Float32)?; // [n, rank]
        let keys = self
            .successor
            .take(&candidates.reshape(&[n * k])?, 0)?
            .astype(Dtype::Float32)?
            .reshape(&[n, k, rank])?;
        let anchor_row = Array::from_u32(&vec![anchor; k as usize], &[1, k])?;
        let predecessor_ids = if n > 1 {
            Array::concatenate(&[&anchor_row, &candidates.slice(0, 0, n - 1)?], 0)?
        } else {
            anchor_row
        };
        let predecessors = self
            .predecessor
            .take(&predecessor_ids.reshape(&[n * k])?, 0)?
            .astype(Dtype::Float32)?
            .reshape(&[n, k, rank])?
            .mul(&projected.reshape(&[n, 1, rank])?)?;
        let mut scores = predecessors
            .matmul(&keys.transpose(&[0, 2, 1])?)?
            .add(&unary.reshape(&[n, 1, k])?)?;
        if let (Some(cap), true) = (d.final_logit_softcapping, after) {
            scores = scores.softcap(cap)?;
        }
        let candidate_ids = candidates.to_u32()?;
        let score_values = scores.to_f32()?;
        let k = k as usize;
        // greedy walk: the best candidate of slot 0 (its predecessor is the anchor), then the best transition
        let mut index = (0..k)
            .max_by(|&a, &b| score_values[a].total_cmp(&score_values[b]).then(b.cmp(&a)))
            .context("empty selector")?;
        let mut proposals = vec![candidate_ids[index]];
        for e in 1..count {
            let row = &score_values[(e * k + index) * k..(e * k + index + 1) * k];
            index = (0..k)
                .max_by(|&a, &b| row[a].total_cmp(&row[b]).then(b.cmp(&a)))
                .context("empty selector row")?;
            proposals.push(candidate_ids[e * k + index]);
        }
        Ok(Dflash2Block {
            hidden,
            head_logits,
            candidates: candidate_ids,
            scores: score_values,
            proposals,
        })
    }

    /// The `count` proposals after `anchor`.
    pub fn propose(&self, target: &Inkling2, anchor: u32, count: usize) -> Result<Vec<u32>> {
        Ok(self.propose_debug(target, anchor, count)?.proposals)
    }

    pub fn bits(&self) -> i32 {
        self.bits
    }

    /// DFLASH2-2026-11-24-INK2M-P2 (a): the same proposals computed on the GPU and left there (`[count]` u32,
    /// lazy): the block embeddings from the anchor `[1]` u32 without reading it, then the selector kernels of
    /// `shaders/dflash_select.metal` (top-16 by 8 shards and merge, edges, greedy walk) with Inkling's real
    /// vocabulary and edge scale 1. Logits and unary values in BF16 as in those kernels (and in SGLang); the
    /// softcap after the correction is increasing, so the greedy walk ignores it.
    pub fn propose_device(&self, target: &Inkling2, anchor: &Array, count: usize) -> Result<Array> {
        let (candidates, unary, edges) = self.selector_lattice(target, anchor, count)?;
        let n = count as i32;
        let v = self.valid_vocab;
        Ok(crate::array::metal_kernel(
            &format!("mlxl3_inkling_dflash2_select_greedy_p{n}_v{v}_v1"),
            &["candidates", "unary", "edges"],
            &["tokens"],
            &self.selector_stage(n, 3),
            include_str!("../shaders/dflash_select.metal"),
            &[&candidates, &unary, &edges],
            &[vec![n]],
            &[Dtype::UInt32],
            [1, 1, 1],
            [1, 1, 1],
        )?
        .remove(0))
    }

    /// The header of stage `stage` of `shaders/dflash_select.metal` for `n` positions.
    fn selector_stage(&self, n: i32, stage: i32) -> String {
        format!(
            "#define DFLASH_STAGE {stage}\n#define DFLASH_VOCABULARY {}u\n#define DFLASH_POSITIONS {n}u\n#define DFLASH_EDGE_SCALE 1.0f\n{}",
            self.valid_vocab,
            include_str!("../shaders/dflash_select.h")
        )
    }

    /// DFLASH2-2026-11-24-INK2M-S3: `count` proposals after `anchor` drawn from the selector's law at
    /// `temperature` (`ratio_walk`, drafter noise from `seed`) with that law, on the GPU.
    pub fn propose_ratio_device(
        &self,
        target: &Inkling2,
        anchor: &Array,
        count: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<crate::inkling2_draft::RatioProposals> {
        let d = &self.config.dflash_config;
        let softcap = match (
            d.final_logit_softcapping,
            d.selector_softcap_mode.as_deref(),
        ) {
            (None, _) => 0.0,
            (Some(cap), Some("after_correction")) => cap.max(0.0),
            _ => anyhow::bail!("the sampled DFlash2 walk needs the softcap after the correction"),
        };
        let (candidates, unary, edges) = self.selector_lattice(target, anchor, count)?;
        let (tokens, q, chosen) =
            ratio_walk(&candidates, &unary, &edges, temperature, softcap, seed)?;
        Ok(crate::inkling2_draft::RatioProposals {
            tokens,
            candidates,
            q,
            chosen,
        })
    }

    /// The selector lattice of the block after `anchor` (`[1]` u32) on the GPU, stages 1-2 of
    /// `shaders/dflash_select.metal`: candidates `[n, 16]` u32, unary values `[n, 16]` BF16 and edges `[n, 16, 16]`
    /// FP32 (edge [e, p, c] from candidate p of position e - 1, the anchor at e = 0, to candidate c).
    fn selector_lattice(
        &self,
        target: &Inkling2,
        anchor: &Array,
        count: usize,
    ) -> Result<(Array, Array, Array)> {
        use crate::array::metal_kernel;
        let d = &self.config.dflash_config;
        ensure!(
            count >= 1 && count < d.block_size && anchor.shape() == [1],
            "invalid DFlash2 device proposal request"
        );
        ensure!(
            d.selector_top_k == 16 && d.selector_rank == 256,
            "the GPU DFlash2 selector needs top-16 and rank 256"
        );
        let n = count as i32;
        // F1: the trained block; its first `count` masked rows are read
        let masked = self.block_rows(count) - 1;
        let masks = Array::from_u32(&vec![d.mask_token_id; masked], &[masked as i32])?;
        let ids = Array::concatenate(&[&anchor.astype(Dtype::UInt32)?, &masks], 0)?;
        let hidden = self.block_hidden(&target.embed_tokens(&ids)?)?;
        let predicted = hidden.slice(0, 1, n + 1)?.astype(Dtype::Float16)?;
        let logits = target
            .head_scores(&predicted)?
            .slice(1, 0, self.valid_vocab)?
            .astype(Dtype::BFloat16)?;
        let selector = self
            .selector_projection
            .forward(&predicted)?
            .astype(Dtype::BFloat16)?;
        let stage = |value: i32| self.selector_stage(n, value);
        let source = include_str!("../shaders/dflash_select.metal");
        let v = self.valid_vocab;
        let mut partial = metal_kernel(
            &format!("mlxl3_inkling_dflash2_select_top16_p{n}_v{v}_v1"),
            &["logits"],
            &["partial_ids", "partial_values"],
            &stage(1),
            source,
            &[&logits],
            &[vec![n * 8, 16], vec![n * 8, 16]],
            &[Dtype::UInt32, Dtype::Float32],
            [n * 8 * 256, 1, 1],
            [256, 1, 1],
        )?;
        let partial_values = partial.pop().context("selector partial values")?;
        let partial_ids = partial.pop().context("selector partial ids")?;
        let mut scored = metal_kernel(
            &format!("mlxl3_inkling_dflash2_select_edges_p{n}_v{v}_v1"),
            &[
                "partial_ids",
                "partial_values",
                "selector",
                "predecessor",
                "successor",
                "anchor",
            ],
            &["candidates", "unary", "edges"],
            &stage(2),
            source,
            &[
                &partial_ids,
                &partial_values,
                &selector,
                &self.predecessor,
                &self.successor,
                &anchor.astype(Dtype::UInt32)?,
            ],
            &[vec![n, 16], vec![n, 16], vec![n, 16, 16]],
            &[Dtype::UInt32, Dtype::BFloat16, Dtype::Float32],
            [n * 256, 1, 1],
            [256, 1, 1],
        )?;
        let edges = scored.pop().context("selector edges")?;
        let unary = scored.pop().context("selector unary")?;
        let candidates = scored.pop().context("selector candidates")?;
        Ok((candidates, unary, edges))
    }
}

/// DFLASH2-2026-11-24-INK2M-S3: the sampled walk of the selector in one thread. At each position the 16 scores
/// unary + edge from the previous proposal (the anchor at 0), softcapped after the correction, give
/// q = softmax(score / T); the proposal is the Gumbel-max draw argmax(score / T + noise) (a draw from q). Writes
/// the proposals, q `[n, 16]` and q of each proposal.
const RATIO_WALK_SOURCE: &str = r#"
    uint predecessor = 0u;
    const float it = inv_t[0];
    for (uint position = 0u; position < uint(POSITIONS); ++position) {
        float s[16];
        float m = -INFINITY;
        for (uint c = 0u; c < 16u; ++c) {
            float score = float(unary[position * 16u + c]) +
                          edges[(position * 16u + predecessor) * 16u + c];
            if (SOFTCAP > 0.0f) {
                score = SOFTCAP * metal::precise::tanh(score / SOFTCAP);
            }
            s[c] = score * it;
            m = metal::max(m, s[c]);
        }
        float z = 0.0f;
        for (uint c = 0u; c < 16u; ++c) {
            z += metal::exp(s[c] - m);
        }
        uint selected = 0u;
        float best = -INFINITY;
        for (uint c = 0u; c < 16u; ++c) {
            const float key = s[c] + noise[position * 16u + c];
            if (key > best) {
                best = key;
                selected = c;
            }
        }
        for (uint c = 0u; c < 16u; ++c) {
            q[position * 16u + c] = metal::exp(s[c] - m) / z;
        }
        chosen[position] = metal::exp(s[selected] - m) / z;
        tokens[position] = candidates[position * 16u + selected];
        predecessor = selected;
    }
"#;

/// DFLASH2-2026-11-24-INK2M-S3: the sampled selector walk over a lattice (`candidates` `[n, 16]` u32, `unary`
/// `[n, 16]` BF16, `edges` `[n, 16, 16]` FP32) at `temperature`, softcap `softcap` after the correction (0: none),
/// drafter noise from `seed`: (proposals `[n]` u32, q `[n, 16]` FP32, q of each proposal `[n]` FP32), lazy.
pub(crate) fn ratio_walk(
    candidates: &Array,
    unary: &Array,
    edges: &Array,
    temperature: f32,
    softcap: f32,
    seed: u64,
) -> Result<(Array, Array, Array)> {
    let n = candidates.shape().first().copied().unwrap_or(0);
    ensure!(
        n >= 1
            && candidates.shape() == [n, 16]
            && unary.shape() == [n, 16]
            && edges.shape() == [n, 16, 16],
        "ratio_walk expects a [n, 16] lattice"
    );
    ensure!(
        temperature.is_finite() && temperature > 0.0 && softcap.is_finite() && softcap >= 0.0,
        "ratio_walk needs a positive temperature and a finite softcap"
    );
    let noise = Array::gumbel_noise(&[n, 16], seed)?;
    let inv_t = Array::from_f32(&[1.0 / temperature], &[1])?;
    let mut out = crate::array::metal_kernel(
        &format!(
            "mlxl3_inkling_dflash2_ratio_walk_v1_p{n}_c{:08x}",
            softcap.to_bits()
        ),
        &["candidates", "unary", "edges", "noise", "inv_t"],
        &["tokens", "q", "chosen"],
        &format!("#define POSITIONS {n}\n#define SOFTCAP {softcap:?}f\n"),
        RATIO_WALK_SOURCE,
        &[candidates, unary, edges, &noise, &inv_t],
        &[vec![n], vec![n, 16], vec![n]],
        &[Dtype::UInt32, Dtype::Float32, Dtype::Float32],
        [1, 1, 1],
        [1, 1, 1],
    )?;
    ensure!(out.len() == 3, "missing ratio walk outputs");
    let chosen = out.pop().context("chosen")?;
    let q = out.pop().context("q")?;
    let tokens = out.pop().context("tokens")?;
    Ok((tokens, q, chosen))
}

impl crate::inkling2_draft::BlockDrafter for Inkling2Dflash2 {
    fn name(&self) -> &'static str {
        "dflash2"
    }
    fn taps(&self) -> &[usize] {
        Inkling2Dflash2::taps(self)
    }
    fn max_proposals(&self) -> usize {
        Inkling2Dflash2::max_proposals(self)
    }
    fn reset(&mut self) {
        Inkling2Dflash2::reset(self)
    }
    fn append_context(&mut self, features: &Array) -> Result<()> {
        Inkling2Dflash2::append_context(self, features)
    }
    fn context_snapshot(&self) -> Result<crate::inkling2_draft::DraftContext> {
        Inkling2Dflash2::context_snapshot(self)
    }
    fn restore_context(&mut self, saved: &crate::inkling2_draft::DraftContext) -> Result<()> {
        Inkling2Dflash2::restore_context(self, saved)
    }
    /// The selector's greedy path on the GPU (`propose_device`; `MLXL3_INKLING_DFLASH2_HOST=1`: the host path of
    /// P0); the drafter keeps its own context, so `last_feature` is not read.
    fn propose(
        &self,
        target: &Inkling2,
        anchor: &Array,
        _last_feature: &Array,
        count: usize,
    ) -> Result<Array> {
        if std::env::var("MLXL3_INKLING_DFLASH2_HOST").as_deref() != Ok("1") {
            return self.propose_device(target, anchor, count);
        }
        let anchor = crate::inkling2::single_token(anchor)?;
        let proposals = Inkling2Dflash2::propose(self, target, anchor, count)?;
        Array::from_u32(&proposals, &[proposals.len() as i32])
    }
    /// S3: the selector's sampled walk with its law (`propose_ratio_device`).
    fn propose_ratio(
        &self,
        target: &Inkling2,
        anchor: &Array,
        _last_feature: &Array,
        count: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<crate::inkling2_draft::RatioProposals> {
        self.propose_ratio_device(target, anchor, count, temperature, seed)
    }
    fn supports_ratio(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::inkling2_draft::Inkling2Draft;

    /// DFLASH2-2026-11-24-INK2M-P3 (2): the convolution kernel against the MLX composition on deterministic
    /// pseudo-random operands, every row count of a block (2..=8), both sides, with and without the residual: same
    /// shapes, finite, within 1e-5 relative (only the rounding of the FP32 sums may differ); the mutated kernel
    /// (taps read the row itself) differs from the reference.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn grouped_conv_kernel_matches_mlx() -> Result<()> {
        let (taps, groups, group) = (2, 144, 16); // the checkpoint's geometry (H = 2304)
        let h = groups * group;
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut values = |n: i32, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| {
                    seed ^= seed << 13;
                    seed ^= seed >> 7;
                    seed ^= seed << 17;
                    ((seed >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * scale
                })
                .collect()
        };
        let conv = GroupedConv {
            base: Array::from_f32(&values(2 * taps * h, 1.0), &[2, taps, h])?,
            projection: Linear::Affine {
                w: Array::zeros_dtype(&[1], Dtype::UInt32)?,
                scales: Array::zeros_dtype(&[1], Dtype::Float16)?,
                biases: Array::zeros_dtype(&[1], Dtype::Float16)?,
                bits: 8,
                group: 64,
            },
            taps,
            groups,
            group,
        };
        let relative = |a: &[f32], b: &[f32]| -> f32 {
            let num: f32 = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).powi(2))
                .sum::<f32>()
                .sqrt();
            let den: f32 = b.iter().map(|y| y * y).sum::<f32>().sqrt();
            num / den.max(f32::MIN_POSITIVE)
        };
        for rows in 2..=8 {
            let x = Array::from_f32(&values(rows * h, 4.0), &[rows, h])?;
            let coef = Array::from_f32(
                &values(rows * 2 * taps * groups, 0.5),
                &[rows, 2, taps, groups],
            )?;
            let residual = Array::from_f32(&values(rows * h, 8.0), &[rows, h])?;
            for side in 0..2 {
                for with_residual in [false, true] {
                    let r = with_residual.then_some(&residual);
                    let reference = {
                        let out = conv.convolve(&x, &coef, side)?;
                        match r {
                            Some(r) => r.add(&out)?,
                            None => out,
                        }
                    };
                    let kernel = conv.apply(&x, &coef, side, r)?;
                    assert_eq!(kernel.shape(), [rows, h]);
                    assert_eq!(kernel.dtype(), Dtype::Float32);
                    let (k, e) = (kernel.to_f32()?, reference.to_f32()?);
                    assert_eq!(k.len(), (rows * h) as usize);
                    assert!(k.iter().all(|v| v.is_finite()), "rows {rows} side {side}");
                    let rel = relative(&k, &e);
                    assert!(
                        rel < 1e-5,
                        "rows {rows} side {side} residual {with_residual}: rel {rel}"
                    );
                    CONV_MUTATION.with(|m| m.set(true));
                    let mutated = conv.apply(&x, &coef, side, r)?.to_f32();
                    CONV_MUTATION.with(|m| m.set(false));
                    let rel = relative(&mutated?, &e);
                    assert!(
                        rel > 1e-2,
                        "mutation not detected: rows {rows} side {side} rel {rel}"
                    );
                }
            }
        }
        Ok(())
    }

    /// DFLASH2-2026-11-24-INK2M-P3c: `warm_up_drafter` over the 7 widths leaves no state: the prefill logits, the
    /// device proposals of every width and the target logits of a verified block (all bits), and a greedy GPU
    /// session of 48 tokens, are identical before the warm-up, after it and after a second one; prints the duration
    /// of both warm-ups (the first one compiles).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL and MLXL3_INKLING_ROWS"]
    fn warm_up_drafter_leaves_no_state() -> Result<()> {
        use crate::inkling2::{DrawRule, DrawSeeds};
        use crate::inkling2_draft::{DsparkSession, warm_up_drafter};
        let mut target = Inkling2::load(Path::new(&std::env::var("MLXL3_INKLING_MODEL")?))?;
        let mut dflash =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        let rows = rows_file()?;
        let (_, tokens) = row_tokens(&rows, 2, target.unpadded_vocab() as u32)?;
        let prompt = tokens[..160].to_vec();
        let limit = usize::try_from(target.context_limit())?;
        let run = |target: &mut Inkling2,
                   dflash: &mut Inkling2Dflash2|
         -> Result<(Vec<u32>, Vec<u32>)> {
            let taps = dflash.taps().to_vec();
            let mut bits = Vec::new();
            let (features, logits) = prefill(target, &prompt, &taps)?;
            bits.extend(logits.to_f32()?.iter().map(|v| v.to_bits()));
            dflash.reset();
            dflash.append_context(&features)?;
            let anchor = logits.chat_greedy_ids()?[0];
            let device_anchor = Array::from_u32(&[anchor], &[1])?;
            for count in 1..=dflash.max_proposals() {
                bits.extend(
                    dflash
                        .propose_device(target, &device_anchor, count)?
                        .to_u32()?,
                );
            }
            let proposals = dflash.propose_device(target, &device_anchor, 3)?;
            let block = Array::concatenate(&[&device_anchor, &proposals], 0)?;
            let (verified, _) = target.verify_device(&block, &taps)?;
            bits.extend(verified.to_f32()?.iter().map(|v| v.to_bits()));
            target.commit(1)?;
            // a greedy session with GPU-decided blocks, one step ahead
            let mut session = DsparkSession::new(dflash, dflash.max_proposals())?
                .with_device_draw(DrawRule::Greedy, DrawSeeds::new(5), true)
                .with_device_rounds(true);
            let logits = session.prefill(target, dflash, &prompt, &|| false)?;
            let mut next = logits.chat_greedy_ids()?[0];
            let mut out = vec![next];
            while out.len() < 48 {
                let mut choose = |proposals: &[u32],
                                  _: Option<&[Vec<f32>]>,
                                  verified: &Array|
                 -> Result<(usize, u32)> {
                    let a =
                        crate::speculative::greedy_accept(proposals, &verified.chat_greedy_ids()?)
                            .context("acceptance")?;
                    Ok((a.accepted_draft_tokens, a.target_token))
                };
                next = session.advance(target, dflash, next, limit, 48 - out.len(), &mut choose)?;
                out.push(next);
            }
            ensure!(session.stats.blocks > 0, "the session ran no block");
            Ok((bits, out))
        };
        let before = run(&mut target, &mut dflash)?;
        let widths = dflash.max_proposals();
        let t = std::time::Instant::now();
        warm_up_drafter(&mut target, &mut dflash, widths)?;
        let first = t.elapsed().as_secs_f64();
        let middle = run(&mut target, &mut dflash)?;
        let t = std::time::Instant::now();
        warm_up_drafter(&mut target, &mut dflash, widths)?;
        let second = t.elapsed().as_secs_f64();
        let after = run(&mut target, &mut dflash)?;
        println!(
            "DFLASH2WARM widths {widths}: first warm-up {first:.2} s, second {second:.2} s; {} bits, tokens {:?}",
            before.0.len(),
            &before.1[..8]
        );
        ensure!(
            before.0.len() > 200_000 && before.1.len() == 48,
            "the compared values are missing"
        );
        ensure!(middle == before, "the first warm-up changed the outputs");
        ensure!(after == before, "the second warm-up changed the outputs");
        Ok(())
    }

    /// DFLASH2-2026-11-24-INK2M-P4-0 (diagnostic): how the routes of real DFlash2 verification blocks share experts.
    /// Rows en0, fr0, code0, chat0, fr1, code1: 64 real tokens of prefill, then greedy steps; every 8 positions the
    /// anchor and its 7 DFlash2 proposals are verified with the routes captured (`[8, 8]` per MoE call) and the
    /// anchor committed. For T = 1..=8 (the first T rows; a row's routing depends only on the rows before it):
    /// routes, distinct experts, duplicated share, members per expert, and the route-equivalents of a hybrid kernel
    /// (one member: the current GEMV, cost 1; two or more: one TensorOps tile of cost c).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL and MLXL3_INKLING_ROWS"]
    fn verify_route_sharing() -> Result<()> {
        use crate::inkling2::MOE_ROUTES;
        const PREFIX: usize = 64;
        const STEPS: usize = 128;
        const EVERY: usize = 8;
        const ROWS: usize = 8;
        let model = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let mut target = Inkling2::load(&model)?;
        let mut dflash =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        let taps = dflash.taps().to_vec();
        ensure!(
            dflash.max_proposals() + 1 == ROWS,
            "DFlash2 block is not 8 rows"
        );
        let rows = rows_file()?;
        let vocab = target.unpadded_vocab() as u32;
        // per T: routes, distinct experts, members histogram (index m = experts with m members)
        let mut routes_t = [0usize; ROWS + 1];
        let mut distinct_t = [0usize; ROWS + 1];
        let mut members_t = vec![vec![0usize; ROWS + 1]; ROWS + 1];
        let (mut blocks, mut calls_per_block) = (0usize, None::<usize>);
        let mut per_row = Vec::new();
        for index in [0usize, 3, 6, 9, 4, 7] {
            let (name, tokens) = row_tokens(&rows, index, vocab)?;
            let (features, logits) = prefill(&mut target, &tokens[..PREFIX], &taps)?;
            dflash.reset();
            dflash.append_context(&features)?;
            let n = features.shape()[0];
            let mut last = features.slice(0, n - 1, n)?;
            let mut next = logits.chat_greedy_ids()?[0];
            let (mut row_routes, mut row_distinct) = (0usize, 0usize);
            for t in 0..STEPS {
                if t % EVERY == 0 {
                    let anchor = Array::from_u32(&[next], &[1])?;
                    let proposals = crate::inkling2_draft::BlockDrafter::propose(
                        &dflash,
                        &target,
                        &anchor,
                        &last,
                        ROWS - 1,
                    )?
                    .to_u32()?;
                    let mut block = vec![next];
                    block.extend(&proposals);
                    MOE_ROUTES.with(|c| *c.borrow_mut() = Some(Vec::new()));
                    let verified = target.verify(&block, &taps);
                    let captured = MOE_ROUTES
                        .with(|c| c.borrow_mut().take())
                        .context("routes")?;
                    let (logits, features) = verified?;
                    ensure!(!captured.is_empty(), "no MoE call captured");
                    match calls_per_block {
                        None => calls_per_block = Some(captured.len()),
                        Some(c) => ensure!(c == captured.len(), "MoE calls differ between blocks"),
                    }
                    for (call_rows, ids) in &captured {
                        ensure!(
                            *call_rows == ROWS && ids.len() == ROWS * 8,
                            "MoE call of {call_rows} rows / {} ids",
                            ids.len()
                        );
                        ensure!(ids.iter().all(|&e| e < 128), "expert id out of range");
                        for t_rows in 1..=ROWS {
                            let mut counts = std::collections::BTreeMap::<u32, usize>::new();
                            for &e in &ids[..t_rows * 8] {
                                *counts.entry(e).or_default() += 1;
                            }
                            routes_t[t_rows] += t_rows * 8;
                            distinct_t[t_rows] += counts.len();
                            for &m in counts.values() {
                                members_t[t_rows][m] += 1;
                            }
                            if t_rows == ROWS {
                                row_routes += t_rows * 8;
                                row_distinct += counts.len();
                            }
                        }
                    }
                    blocks += 1;
                    target.commit(1)?;
                    let first = features.slice(0, 0, 1)?;
                    dflash.append_context(&first)?;
                    last = first;
                    next = logits.chat_greedy_ids()?[0];
                } else {
                    let (logits, features) = target.forward_with_features(&[next], &taps)?;
                    dflash.append_context(&features)?;
                    last = features;
                    next = logits.chat_greedy_ids()?[0];
                }
            }
            per_row.push(serde_json::json!({
                "row": name, "duplicated_share_t8": 1.0 - row_distinct as f64 / row_routes as f64,
            }));
        }
        ensure!(
            blocks == 6 * STEPS / EVERY,
            "expected 96 blocks, ran {blocks}"
        );
        let calls = calls_per_block.context("calls")? * blocks;
        let report: Vec<serde_json::Value> = (1..=ROWS)
            .map(|t| {
                let equivalents = |c: f64| -> f64 {
                    members_t[t]
                        .iter()
                        .enumerate()
                        .map(|(m, &n)| n as f64 * if m <= 1 { m as f64 } else { c })
                        .sum::<f64>()
                        / calls as f64
                };
                serde_json::json!({
                    "rows": t,
                    "routes_per_call": routes_t[t] as f64 / calls as f64,
                    "distinct_per_call": distinct_t[t] as f64 / calls as f64,
                    "duplicated_share": 1.0 - distinct_t[t] as f64 / routes_t[t] as f64,
                    "members": members_t[t][1..=t].to_vec(),
                    "equivalents_c1_3": equivalents(1.3),
                    "equivalents_c1_5": equivalents(1.5),
                })
            })
            .collect();
        println!(
            "P4ROUTES {}",
            serde_json::to_string(&serde_json::json!({
                "blocks": blocks, "moe_calls_per_block": calls / blocks, "by_rows": report, "by_row": per_row,
            }))?
        );
        Ok(())
    }

    fn rows_file() -> Result<serde_json::Value> {
        Ok(serde_json::from_reader(File::open(std::env::var(
            "MLXL3_INKLING_ROWS",
        )?)?)?)
    }

    fn row_tokens(
        rows: &serde_json::Value,
        index: usize,
        vocab: u32,
    ) -> Result<(String, Vec<u32>)> {
        let row = &rows["rows"][index];
        let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone())?;
        Ok((
            row["name"].as_str().unwrap_or("?").to_owned(),
            tokens.into_iter().map(|t| t % vocab).collect(),
        ))
    }

    fn write_f32(path: &Path, a: &Array) -> Result<()> {
        let values = a.astype(Dtype::Float32)?.to_f32()?;
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        Ok(())
    }

    /// Prefills `tokens` by chunks of 128 with the drafter's taps; returns the features `[n, taps * D]` and the
    /// last row's logits.
    fn prefill(target: &mut Inkling2, tokens: &[u32], taps: &[usize]) -> Result<(Array, Array)> {
        target.reset();
        let mut features = Vec::new();
        let mut logits = None;
        for chunk in tokens.chunks(128) {
            let (l, f) = target.forward_with_features(chunk, taps)?;
            features.push(f);
            logits = Some(l);
        }
        Ok((
            Array::concatenate(&features.iter().collect::<Vec<_>>(), 0)?,
            logits.context("empty prefill")?,
        ))
    }

    /// DFLASH2-2026-11-24-INK2M-P0 oracle export: for rows en0, fr0 and code0 of `MLXL3_INKLING_ROWS`, the first 200
    /// tokens as context and the next one as anchor: the context features, the block embeddings, the engine's final
    /// hidden rows (also with the convolution mutation), the target head logits of rows 1..7, the top-16
    /// candidates, the lattice scores and the 7 proposals, under `MLXL3_DFLASH2_ORACLE_OUT/<row>/` for
    /// `benchmarks/inkling2_dflash2_oracle.py`.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL, MLXL3_INKLING_ROWS, MLXL3_DFLASH2_ORACLE_OUT"]
    fn export_oracle_cases() -> Result<()> {
        let mut target = Inkling2::load(Path::new(&std::env::var("MLXL3_INKLING_MODEL")?))?;
        let mut draft =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        let out = std::path::PathBuf::from(std::env::var("MLXL3_DFLASH2_ORACLE_OUT")?);
        let rows = rows_file()?;
        let vocab = target.unpadded_vocab() as u32;
        let taps = draft.taps().to_vec();
        for index in [0usize, 3, 6] {
            let (name, tokens) = row_tokens(&rows, index, vocab)?;
            const CONTEXT: usize = 200;
            let (features, _) = prefill(&mut target, &tokens[..CONTEXT], &taps)?;
            draft.reset();
            draft.append_context(&features)?;
            let anchor = tokens[CONTEXT];
            let block = draft.propose_debug(&target, anchor, draft.max_proposals())?;
            // P2 (a): the GPU selector against the host path the oracle checks
            let device = draft
                .propose_device(
                    &target,
                    &Array::from_u32(&[anchor], &[1])?,
                    draft.max_proposals(),
                )?
                .to_u32()?;
            println!(
                "DFLASH2 device {name}: {device:?} (host {:?}, same {})",
                block.proposals,
                device == block.proposals
            );
            let mask = draft.config().dflash_config.mask_token_id;
            let mut ids = vec![mask; draft.max_proposals() + 1];
            ids[0] = anchor;
            let embeds = target.embed_tokens(&Array::from_u32(&ids, &[ids.len() as i32])?)?;
            CONV_MUTATION.with(|m| m.set(true));
            let mutated = draft.block_hidden(&embeds);
            CONV_MUTATION.with(|m| m.set(false));
            let mutated = mutated?;
            let dir = out.join(&name);
            std::fs::create_dir_all(&dir)?;
            write_f32(&dir.join("features.f32"), &features)?;
            write_f32(&dir.join("embeds.f32"), &embeds)?;
            write_f32(&dir.join("hidden.f32"), &block.hidden)?;
            write_f32(&dir.join("hidden_mutated.f32"), &mutated)?;
            write_f32(&dir.join("head_logits.f32"), &block.head_logits)?;
            let meta = serde_json::json!({
                "row": name, "context": CONTEXT, "anchor": anchor, "mask": mask,
                "valid_vocab": draft.valid_vocab, "taps": taps, "features_shape": features.shape(),
                "candidates": block.candidates, "scores": block.scores, "proposals": block.proposals,
                "proposals_device": device, "bits": draft.bits(),
                "next_tokens": &tokens[CONTEXT + 1..CONTEXT + 8],
            });
            std::fs::write(dir.join("meta.json"), serde_json::to_string(&meta)?)?;
            println!("DFLASH2 export {name}: proposals {:?}", block.proposals);
        }
        Ok(())
    }

    /// DFLASH2-2026-11-24-INK2M-P0 acceptance: rows en0, fr0, code0, chat0, fr1, code1, 64 real tokens of prefill
    /// then 136 greedy AR steps whose features feed both drafters; at each of the first 128 anchors, the 7
    /// proposals of DFlash2 and of DSpark against the greedy continuation (leading proposals equal to the next
    /// greedy tokens). Prints the accepted counts, AL = accepted + 1 and the acceptance by position.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL and MLXL3_INKLING_ROWS"]
    fn acceptance_against_dspark() -> Result<()> {
        const PREFIX: usize = 64;
        const ANCHORS: usize = 128;
        let model = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let mut target = Inkling2::load(&model)?;
        let mut dflash =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        let mut dspark = Inkling2Draft::load(&model, &target)?;
        ensure!(
            dflash.taps() == dspark.taps(),
            "the drafters read different taps"
        );
        let taps = dflash.taps().to_vec();
        let width = dflash.max_proposals();
        ensure!(
            dspark.max_proposals() >= width,
            "DSpark block narrower than DFlash2's"
        );
        let rows = rows_file()?;
        let vocab = target.unpadded_vocab() as u32;
        let (mut all_f, mut all_s, mut all_n) =
            (vec![0usize; width + 1], vec![0usize; width + 1], 0usize);
        let mut report = Vec::new();
        for index in [0usize, 3, 6, 9, 4, 7] {
            let (name, tokens) = row_tokens(&rows, index, vocab)?;
            let (features, logits) = prefill(&mut target, &tokens[..PREFIX], &taps)?;
            dflash.reset();
            dspark.reset();
            dflash.append_context(&features)?;
            dspark.append_context(&features)?;
            let rows_n = features.shape()[0];
            let mut last = features.slice(0, rows_n - 1, rows_n)?;
            let mut greedy = vec![logits.chat_greedy_ids()?[0]];
            let (mut proposals_f, mut proposals_s) = (Vec::new(), Vec::new());
            let started = std::time::Instant::now();
            let mut dflash_ms = 0f64;
            for t in 0..ANCHORS + width + 1 {
                let anchor = greedy[t];
                if t < ANCHORS {
                    let t0 = std::time::Instant::now();
                    // the bridge's path (GPU selector unless MLXL3_INKLING_DFLASH2_HOST=1)
                    proposals_f.push(
                        crate::inkling2_draft::BlockDrafter::propose(
                            &dflash,
                            &target,
                            &Array::from_u32(&[anchor], &[1])?,
                            &last,
                            width,
                        )?
                        .to_u32()?,
                    );
                    dflash_ms += t0.elapsed().as_secs_f64() * 1e3;
                    proposals_s.push(
                        dspark
                            .propose(&target, &Array::from_u32(&[anchor], &[1])?, &last, width)?
                            .to_u32()?,
                    );
                }
                let (logits, features) = target.forward_with_features(&[anchor], &taps)?;
                dflash.append_context(&features)?;
                dspark.append_context(&features)?;
                last = features;
                greedy.push(logits.chat_greedy_ids()?[0]);
            }
            let accepted = |proposals: &[u32], t: usize| {
                proposals
                    .iter()
                    .zip(&greedy[t + 1..])
                    .take_while(|(p, g)| p == g)
                    .count()
            };
            let (mut hist_f, mut hist_s) = (vec![0usize; width + 1], vec![0usize; width + 1]);
            for t in 0..ANCHORS {
                hist_f[accepted(&proposals_f[t], t)] += 1;
                hist_s[accepted(&proposals_s[t], t)] += 1;
            }
            let mean = |h: &[usize]| {
                h.iter()
                    .enumerate()
                    .map(|(a, &n)| (a * n) as f64)
                    .sum::<f64>()
                    / ANCHORS as f64
            };
            let by_position = |h: &[usize]| -> Vec<f64> {
                (1..=width)
                    .map(|p| {
                        let reached: usize = h[p - 1..].iter().sum();
                        let kept: usize = h[p..].iter().sum();
                        if reached == 0 {
                            0.0
                        } else {
                            kept as f64 / reached as f64
                        }
                    })
                    .collect()
            };
            let (af, as_) = (mean(&hist_f), mean(&hist_s));
            println!(
                "DFLASH2ACC {name}: DFlash2 accepted {af:.3} (AL {:.3}) by position {:.3?}; DSpark accepted {as_:.3} (AL {:.3}) by position {:.3?}; DFlash2 propose {:.1} ms/block; row {:.1} s",
                af + 1.0,
                by_position(&hist_f),
                as_ + 1.0,
                by_position(&hist_s),
                dflash_ms / ANCHORS as f64,
                started.elapsed().as_secs_f64()
            );
            report.push(serde_json::json!({"row": name, "dflash2": hist_f, "dspark": hist_s}));
            for a in 0..=width {
                all_f[a] += hist_f[a];
                all_s[a] += hist_s[a];
            }
            all_n += ANCHORS;
        }
        let total = |h: &[usize]| {
            h.iter()
                .enumerate()
                .map(|(a, &n)| (a * n) as f64)
                .sum::<f64>()
                / all_n as f64
        };
        println!(
            "DFLASH2ACC total ({all_n} anchors): DFlash2 accepted {:.3} (AL {:.3}), DSpark accepted {:.3} (AL {:.3}); histograms {all_f:?} / {all_s:?}",
            total(&all_f),
            total(&all_f) + 1.0,
            total(&all_s),
            total(&all_s) + 1.0
        );
        println!("DFLASH2ACC json {}", serde_json::Value::Array(report));
        Ok(())
    }

    /// (blocks, accepted proposals, delivered tokens, AR replay differences (position, gap, rank)).
    type SessionRun = (usize, usize, Vec<u32>, Vec<(usize, f32, usize)>, usize);

    /// DFLASH2-2026-11-24-INK2M-P0 session check: greedy DFlash2 blocks of 7 proposals verified by the target on
    /// the prompt of row en0 (200 tokens), 160 tokens; a forced AR replay of the delivered tokens must find only
    /// rank-1 quasi-ties (gap <= 0.25, the bound of the D5 reference).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL and MLXL3_INKLING_ROWS"]
    fn greedy_session_matches_ar() -> Result<()> {
        const TOKENS: usize = 160;
        let mut target = Inkling2::load(Path::new(&std::env::var("MLXL3_INKLING_MODEL")?))?;
        let mut dflash =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        let taps = dflash.taps().to_vec();
        let rows = rows_file()?;
        let (_, tokens) = row_tokens(&rows, 0, target.unpadded_vocab() as u32)?;
        let prompt = &tokens[..200];
        let (features, logits) = prefill(&mut target, prompt, &taps)?;
        dflash.reset();
        dflash.append_context(&features)?;
        let first = logits.chat_greedy_ids()?[0];
        let model = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let mut dspark = Inkling2Draft::load(&model, &target)?;
        // the same greedy loop with either drafter's 7 proposals, then a forced AR replay of what it delivered
        let run = |target: &mut Inkling2,
                   dflash: &mut Inkling2Dflash2,
                   dspark: &mut Inkling2Draft,
                   use_dflash: bool|
         -> Result<SessionRun> {
            let (features, _) = prefill(target, prompt, &taps)?;
            dflash.reset();
            dspark.reset();
            dflash.append_context(&features)?;
            dspark.append_context(&features)?;
            let n = features.shape()[0];
            let mut last = features.slice(0, n - 1, n)?;
            let mut out = vec![first];
            let (mut blocks, mut accepted_total) = (0usize, 0usize);
            // INVARIANT-2026-11-24-INK2M-L1L4: the bits of every verified row the session used, by the position it
            // predicts
            let mut used_rows: Vec<Option<Vec<u32>>> = vec![None; TOKENS + 8];
            while out.len() < TOKENS {
                let anchor = *out.last().context("anchor")?;
                let count = dflash.max_proposals().min(TOKENS - out.len());
                let proposals = if count == 0 {
                    Vec::new()
                } else if use_dflash {
                    dflash.propose(target, anchor, count)?
                } else {
                    dspark
                        .propose(target, &Array::from_u32(&[anchor], &[1])?, &last, count)?
                        .to_u32()?
                };
                let mut block = vec![anchor];
                block.extend(&proposals);
                let (verified, features) = target.verify(&block, &taps)?;
                let rows_argmax = verified.chat_greedy_ids()?;
                let base = out.len();
                let accepted = proposals
                    .iter()
                    .zip(&rows_argmax)
                    .take_while(|(p, r)| p == r)
                    .count();
                target.commit(accepted + 1)?;
                let kept = features.slice(0, 0, accepted as i32 + 1)?;
                dflash.append_context(&kept)?;
                dspark.append_context(&kept)?;
                last = features.slice(0, accepted as i32, accepted as i32 + 1)?;
                let flat = verified.reshape(&[block.len() as i32, -1])?;
                for j in 0..=accepted {
                    if base + j < used_rows.len() {
                        used_rows[base + j] = Some(
                            flat.slice(0, j as i32, j as i32 + 1)?
                                .astype(Dtype::Float32)?
                                .to_f32()?
                                .iter()
                                .map(|v| v.to_bits())
                                .collect(),
                        );
                    }
                }
                out.extend(&proposals[..accepted]);
                out.push(rows_argmax[accepted]);
                blocks += 1;
                accepted_total += accepted;
            }
            out.truncate(TOKENS);
            // the AR replay from the session's own prefill (same chunks of 128, same states), then one step per token
            let (_, mut row) = prefill(target, prompt, &taps)?;
            let mut differences = Vec::new();
            let mut rows_differ = 0usize;
            for (i, &token) in out.iter().enumerate() {
                let best = row.chat_greedy_ids()?[0];
                if best != token {
                    let values = row.reshape(&[-1])?.to_f32()?;
                    let got = values[token as usize];
                    let rank = values.iter().filter(|&&v| v > got).count();
                    differences.push((i, values[best as usize] - got, rank));
                }
                if i + 1 < out.len() {
                    row = target.forward(token)?;
                    if let Some(Some(used)) = used_rows.get(i + 1) {
                        let replay: Vec<u32> = row
                            .reshape(&[1, -1])?
                            .astype(Dtype::Float32)?
                            .to_f32()?
                            .iter()
                            .map(|v| v.to_bits())
                            .collect();
                        rows_differ += usize::from(&replay != used);
                    }
                }
            }
            Ok((blocks, accepted_total, out, differences, rows_differ))
        };
        let (blocks, accepted_total, out, differences, rows_differ) =
            run(&mut target, &mut dflash, &mut dspark, true)?;
        let (r_blocks, r_accepted, _, reference, r_rows_differ) =
            run(&mut target, &mut dflash, &mut dspark, false)?;
        println!(
            "DFLASH2SESSION verified logit rows differing from the AR replay: DFlash2 {rows_differ}, DSpark {r_rows_differ}"
        );
        println!(
            "DFLASH2SESSION: {blocks} blocks, {accepted_total} accepted, {:.2} tokens per block; replay differences {differences:?}",
            out.len() as f64 / blocks as f64
        );
        println!(
            "DFLASH2SESSION reference (DSpark, 7 proposals): {r_blocks} blocks, {r_accepted} accepted; replay differences {reference:?}"
        );
        // INVARIANT-2026-11-24-INK2M-L1L4: with the invariant verification, both sessions are the AR replay exactly
        if crate::inkling2::invariant_rows() {
            ensure!(
                differences.is_empty()
                    && reference.is_empty()
                    && rows_differ == 0
                    && r_rows_differ == 0,
                "invariant verification: the sessions differ from the AR replay: {differences:?} / {reference:?}, logit rows {rows_differ} / {r_rows_differ}"
            );
            println!(
                "DFLASH2SESSION invariant: both sessions identical to the AR replay ({} tokens)",
                out.len()
            );
        }
        let bound = reference.iter().map(|d| d.1).fold(0.25f32, f32::max);
        ensure!(
            differences
                .iter()
                .all(|&(_, gap, rank)| rank == 1 && gap <= bound)
                && reference.iter().all(|&(_, _, rank)| rank == 1),
            "DFlash2 session differs from the AR replay beyond the 8-row reference (bound {bound}): {differences:?}"
        );
        Ok(())
    }
    /// DFLASH2-P2-DIAG: medians of 9 (after 2 warm-ups) of the DFlash2 proposals (7, GPU path, waited), verify calls
    /// of 2 / 4 / 8 rows (waited, then `commit(1)`) and a plain step, at 200 then 2 600 tokens of real context.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL and MLXL3_INKLING_ROWS"]
    fn block_cost_by_context() -> Result<()> {
        let mut target = Inkling2::load(Path::new(&std::env::var("MLXL3_INKLING_MODEL")?))?;
        let mut dflash =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        let taps = dflash.taps().to_vec();
        let rows = rows_file()?;
        let vocab = target.unpadded_vocab() as u32;
        let mut text = Vec::new();
        for index in 0..rows["rows"].as_array().context("rows")?.len() {
            text.extend(row_tokens(&rows, index, vocab)?.1);
        }
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        for context in [200usize, 2600] {
            ensure!(text.len() > context + 64, "not enough text");
            let (features, _) = prefill(&mut target, &text[..context], &taps)?;
            dflash.reset();
            dflash.append_context(&features)?;
            let mut position = context;
            let (mut draft, mut plain) = (Vec::new(), Vec::new());
            let mut verify = [Vec::new(), Vec::new(), Vec::new()];
            for round in 0..11 {
                let anchor = text[position];
                let t = std::time::Instant::now();
                let proposals = dflash
                    .propose_device(&target, &Array::from_u32(&[anchor], &[1])?, 7)?
                    .to_u32()?;
                let draft_ms = t.elapsed().as_secs_f64() * 1e3;
                let mut verify_ms = [0f64; 3];
                for (slot, rows) in [2usize, 4, 8].into_iter().enumerate() {
                    // each verify starts at the current anchor, keeps it (commit 1) and feeds the drafter, so the
                    // next one starts one position later (the proposals only set the row count)
                    let mut block = vec![text[position]];
                    block.extend(&proposals[..rows - 1]);
                    let t = std::time::Instant::now();
                    let (logits, features) = target.verify(&block, &taps)?;
                    logits.eval()?;
                    features.eval()?;
                    verify_ms[slot] = t.elapsed().as_secs_f64() * 1e3;
                    target.commit(1)?;
                    dflash.append_context(&features.slice(0, 0, 1)?)?;
                    position += 1;
                }
                let t = std::time::Instant::now();
                let (logits, features) = target.forward_with_features(&[text[position]], &taps)?;
                logits.eval()?;
                plain.push(t.elapsed().as_secs_f64() * 1e3);
                dflash.append_context(&features)?;
                position += 1;
                if round >= 2 {
                    draft.push(draft_ms);
                    for (slot, ms) in verify_ms.iter().enumerate() {
                        verify[slot].push(*ms);
                    }
                }
            }
            println!(
                "DFLASH2DIAG context {context}: draft {:.1} ms, verify 2/4/8 rows {:.1} / {:.1} / {:.1} ms, plain {:.1} ms",
                median(draft),
                median(verify[0].clone()),
                median(verify[1].clone()),
                median(verify[2].clone()),
                median(plain)
            );
        }
        Ok(())
    }

    /// Acceptance of one DFlash2 proposal slot under the three rules at temperature `t`: target logits
    /// `logits` (softcapped, the whole vocabulary), the slot's candidate ids and their selector scores after the
    /// predecessor's edge (softcapped as the selector does). Returns (point, coupled, ratio):
    /// point = p(argmax score), the current rule (exact); ratio = sum_c min(p_c, q_c), the rule of a proposal drawn
    /// from q = softmax(score / t) kept with min(1, p / q) (exact); coupled = P(argmax_c(log q_c + G_c) =
    /// argmax_v(log p_v + G_v)) estimated over `draws` Gumbel draws, the vocabulary outside the candidates drawn as
    /// one Gumbel of location log(1 - sum_c p_c) (the law of its maximum).
    /// A target row (softcapped logits) and its `log_norm` at each temperature of the S1 diagnostic.
    type TargetRow = (Vec<f32>, Vec<(f64, f64)>);

    /// R0 trace: one block's candidates, lattice scores and first proposal.
    type TracedBlock = (Vec<u32>, Vec<f32>, u32);

    /// A cache's offset, key shape, keys and values (host copies).
    type CacheValues = (i32, Vec<i32>, Vec<f32>, Vec<f32>);

    /// (max, log sum_v exp((l_v - max) / t)) of a target row at temperature `t`.
    fn log_norm(logits: &[f32], t: f32) -> (f64, f64) {
        let t = f64::from(t);
        let max = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let log_z = logits
            .iter()
            .map(|&l| ((f64::from(l) - max) / t).exp())
            .sum::<f64>()
            .ln();
        (max, log_z)
    }

    fn slot_acceptance(
        logits: &[f32],
        candidates: &[u32],
        scores: &[f32],
        t: f32,
        draws: usize,
        rng: &mut u64,
    ) -> (f64, f64, f64) {
        slot_acceptance_normed(
            logits,
            log_norm(logits, t),
            candidates,
            scores,
            t,
            draws,
            rng,
        )
    }

    /// `slot_acceptance` with the row's `log_norm` at `t` computed once.
    fn slot_acceptance_normed(
        logits: &[f32],
        (max, log_z): (f64, f64),
        candidates: &[u32],
        scores: &[f32],
        t: f32,
        draws: usize,
        rng: &mut u64,
    ) -> (f64, f64, f64) {
        let t = f64::from(t);
        // log p at temperature t: (l - max) / t - log sum_v exp((l_v - max) / t)
        let log_p: Vec<f64> = candidates
            .iter()
            .map(|&c| (f64::from(logits[c as usize]) - max) / t - log_z)
            .collect();
        let p: Vec<f64> = log_p.iter().map(|v| v.exp()).collect();
        let s_max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let w: Vec<f64> = scores
            .iter()
            .map(|&s| ((f64::from(s) - s_max) / t).exp())
            .collect();
        let w_sum: f64 = w.iter().sum();
        let q: Vec<f64> = w.iter().map(|v| v / w_sum).collect();
        let log_q: Vec<f64> = q.iter().map(|v| v.ln()).collect();
        // the selector's greedy pick (ties: the lower index, as the walk)
        let best = (0..scores.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a)))
            .unwrap_or(0);
        let point = p[best];
        let ratio: f64 = p.iter().zip(&q).map(|(a, b)| a.min(*b)).sum();
        let rest = (1.0 - p.iter().sum::<f64>()).max(0.0);
        let mut gumbel = || {
            *rng ^= *rng << 13;
            *rng ^= *rng >> 7;
            *rng ^= *rng << 17;
            let u = ((*rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
            -(-u.ln()).ln()
        };
        let mut same = 0usize;
        for _ in 0..draws {
            let (mut target_best, mut target_value) = (usize::MAX, f64::NEG_INFINITY);
            let (mut draft_best, mut draft_value) = (usize::MAX, f64::NEG_INFINITY);
            for c in 0..candidates.len() {
                let g = gumbel();
                if log_p[c] + g > target_value {
                    target_value = log_p[c] + g;
                    target_best = c;
                }
                if log_q[c] + g > draft_value {
                    draft_value = log_q[c] + g;
                    draft_best = c;
                }
            }
            let other = if rest > 0.0 {
                rest.ln() + gumbel()
            } else {
                f64::NEG_INFINITY
            };
            if target_value > other && target_best == draft_best {
                same += 1;
            }
        }
        (point, same as f64 / draws as f64, ratio)
    }

    /// `slot_acceptance` against exact values on a small synthetic case: p = q gives ratio 1 and coupled 1;
    /// the point rule is p of the best score; a target off the candidates gives 0 for every rule.
    #[test]
    fn slot_acceptance_rules_on_synthetic_laws() {
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        // target over 4 tokens: logits ln(0.4, 0.3, 0.2, 0.1); candidates = all 4 with the same scores
        let logits: Vec<f32> = [0.4f32, 0.3, 0.2, 0.1].iter().map(|v| v.ln()).collect();
        let candidates = [0u32, 1, 2, 3];
        let (point, coupled, ratio) =
            slot_acceptance(&logits, &candidates, &logits, 1.0, 4000, &mut rng);
        assert!((point - 0.4).abs() < 1e-6, "point {point}");
        assert!((ratio - 1.0).abs() < 1e-6, "ratio {ratio}");
        assert!((coupled - 1.0).abs() < 1e-9, "coupled {coupled}");
        // uniform draft over the same candidates: ratio = sum min(p, 0.25) = 0.25 + 0.25 + 0.2 + 0.1
        let flat = [0f32; 4];
        let (point, coupled, ratio) =
            slot_acceptance(&logits, &candidates, &flat, 1.0, 20000, &mut rng);
        assert!((point - 0.4).abs() < 1e-6, "flat point {point}");
        assert!((ratio - 0.8).abs() < 1e-6, "flat ratio {ratio}");
        assert!(coupled > 0.3 && coupled < 0.8, "flat coupled {coupled}");
        // candidates the target never draws: every rule 0
        let logits = [0f32, -1e4, -1e4];
        let (point, coupled, ratio) =
            slot_acceptance(&logits, &[1, 2], &[1.0, 0.0], 1.0, 2000, &mut rng);
        assert!(point < 1e-12 && ratio < 1e-12 && coupled == 0.0);
        // temperature: p at t = 0.5 is the square law renormalized
        let logits: Vec<f32> = [0.6f32, 0.4].iter().map(|v| v.ln()).collect();
        let (point, _, _) = slot_acceptance(&logits, &[0, 1], &[1.0, 0.0], 0.5, 1, &mut rng);
        assert!((point - 0.36 / 0.52).abs() < 1e-6, "t 0.5 point {point}");
    }

    /// DFLASH2-2026-11-24-INK2M-S1 (diagnostic): acceptance of DFlash2's proposals at T = 1 (and 0.7) under the
    /// point rule (current), the coupled rule (D2) and the ratio rule (Splash), slot by slot on the same real text:
    /// rows en0, fr0, code0, chat0, fr1, code1 of `MLXL3_INKLING_ROWS`, 64 prefix tokens then 192 anchors; slot e's
    /// predecessor is the real token before it, and the slot counts only when the real tokens before it are among
    /// their slots' candidates. Prints, per row and in total, the slots reached, the mean acceptance by slot and
    /// rule, and the expected accepted tokens of a 7-proposal block (sum of the products of the slot acceptances
    /// along the real text).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL and MLXL3_INKLING_ROWS"]
    fn acceptance_by_rule_t1() -> Result<()> {
        const PREFIX: usize = 64;
        const ANCHORS: usize = 192;
        const DRAWS: usize = 2048;
        let temperatures = [1.0f32, 0.7];
        let mut target = Inkling2::load(Path::new(&std::env::var("MLXL3_INKLING_MODEL")?))?;
        let mut dflash =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        let taps = dflash.taps().to_vec();
        let width = dflash.max_proposals();
        let k = dflash.config().dflash_config.selector_top_k as usize;
        let rows = rows_file()?;
        let vocab = target.unpadded_vocab() as u32;
        let mut rng = 0x51ec_7104_u64;
        // [temperature][rule][slot] sums, reached per slot, expected tokens per rule
        let rules = ["point", "coupled", "ratio"];
        let mut total = vec![vec![vec![0f64; width]; 3]; temperatures.len()];
        let mut total_expected = vec![vec![0f64; 3]; temperatures.len()];
        let (mut total_reached, mut total_anchors) = (vec![0usize; width], 0usize);
        println!("S1 drafter bits {}", dflash.bits());
        for index in [0usize, 3, 6, 9, 4, 7] {
            let (name, tokens) = row_tokens(&rows, index, vocab)?;
            ensure!(
                tokens.len() > PREFIX + ANCHORS + width + 1,
                "row {name} too short"
            );
            let (features, _) = prefill(&mut target, &tokens[..PREFIX], &taps)?;
            dflash.reset();
            dflash.append_context(&features)?;
            let mut blocks = Vec::with_capacity(ANCHORS);
            let mut target_rows: Vec<TargetRow> = Vec::with_capacity(ANCHORS + width);
            for i in PREFIX..PREFIX + ANCHORS + width {
                if i < PREFIX + ANCHORS {
                    let block = dflash.propose_debug(&target, tokens[i], width)?;
                    ensure!(
                        block.candidates.len() == width * k && block.scores.len() == width * k * k,
                        "block shapes"
                    );
                    blocks.push((block.candidates, block.scores));
                }
                let (logits, features) = target.forward_with_features(&tokens[i..i + 1], &taps)?;
                let row = logits.reshape(&[vocab as i32])?.to_f32()?;
                ensure!(
                    row.len() == vocab as usize && row.iter().all(|v| v.is_finite()),
                    "target row"
                );
                let norms = temperatures.iter().map(|&t| log_norm(&row, t)).collect();
                target_rows.push((row, norms));
                dflash.append_context(&features)?;
            }
            let mut sums = vec![vec![vec![0f64; width]; 3]; temperatures.len()];
            let mut expected = vec![vec![0f64; 3]; temperatures.len()];
            let mut reached = vec![0usize; width];
            for (a, (candidates, scores)) in blocks.iter().enumerate() {
                let j = PREFIX + a; // the anchor's index in `tokens`
                let mut chain = vec![vec![1f64; 3]; temperatures.len()];
                for e in 0..width {
                    // slot e predicts tokens[j + 1 + e]; its predecessor is tokens[j + e]
                    let p = if e == 0 {
                        0
                    } else {
                        match candidates[(e - 1) * k..e * k]
                            .iter()
                            .position(|&c| c == tokens[j + e])
                        {
                            Some(p) => p,
                            None => break,
                        }
                    };
                    reached[e] += 1;
                    let slot_candidates = &candidates[e * k..(e + 1) * k];
                    let slot_scores = &scores[(e * k + p) * k..(e * k + p + 1) * k];
                    let (logits, norms) = &target_rows[a + e];
                    for (ti, &t) in temperatures.iter().enumerate() {
                        let (point, coupled, ratio) = slot_acceptance_normed(
                            logits,
                            norms[ti],
                            slot_candidates,
                            slot_scores,
                            t,
                            DRAWS,
                            &mut rng,
                        );
                        for (r, value) in [point, coupled, ratio].into_iter().enumerate() {
                            ensure!(
                                value.is_finite() && (0.0..=1.0 + 1e-9).contains(&value),
                                "rule value"
                            );
                            sums[ti][r][e] += value;
                            chain[ti][r] *= value;
                            expected[ti][r] += chain[ti][r];
                        }
                    }
                }
            }
            for (ti, &t) in temperatures.iter().enumerate() {
                for (r, rule) in rules.iter().enumerate() {
                    let by_slot: Vec<String> = (0..width)
                        .map(|e| {
                            if reached[e] == 0 {
                                "-".into()
                            } else {
                                format!("{:.3}", sums[ti][r][e] / reached[e] as f64)
                            }
                        })
                        .collect();
                    println!(
                        "S1 {name} T {t} {rule}: by slot [{}], expected accepted {:.3} per block",
                        by_slot.join(", "),
                        expected[ti][r] / ANCHORS as f64
                    );
                    for e in 0..width {
                        total[ti][r][e] += sums[ti][r][e];
                    }
                    total_expected[ti][r] += expected[ti][r];
                }
            }
            println!("S1 {name} reached by slot {reached:?} of {ANCHORS} anchors");
            for e in 0..width {
                total_reached[e] += reached[e];
            }
            total_anchors += ANCHORS;
        }
        for (ti, &t) in temperatures.iter().enumerate() {
            for (r, rule) in rules.iter().enumerate() {
                let by_slot: Vec<String> = (0..width)
                    .map(|e| format!("{:.3}", total[ti][r][e] / total_reached[e].max(1) as f64))
                    .collect();
                println!(
                    "S1 total T {t} {rule}: by slot [{}], expected accepted {:.3} per block",
                    by_slot.join(", "),
                    total_expected[ti][r] / total_anchors as f64
                );
            }
        }
        println!("S1 total reached by slot {total_reached:?} of {total_anchors} anchors");
        Ok(())
    }

    /// DFLASH2-2026-11-24-INK2M-S3: the sampled selector walk (`ratio_walk`) draws from the law it returns. A
    /// synthetic lattice of 2 positions (16 candidates, BF16 unary values, edges), T = 0.8, softcap 30, 20 000
    /// draws: q of each position equals the host's softmax(softcap(unary + edge from the drawn predecessor) / T)
    /// within 1e-5, `chosen` = q of the proposal, and the frequencies of the first proposal and of the second
    /// after the most frequent first one match their q (chi-square below the 0.999 quantile, 15 degrees of
    /// freedom); a draw with another noise than the walk's own (mutation) is detected.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn ratio_walk_follows_its_law() -> Result<()> {
        const DRAWS: usize = 20_000;
        const BATCH: usize = 1_000;
        let (t, cap) = (0.8f32, 30.0f32);
        let mut rng = 0x1234_5678_9abc_def1u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        let candidates: Vec<u32> = (0..32u32).map(|i| 1000 + 7 * i).collect();
        let unary: Vec<f32> = (0..32).map(|_| (next() * 6.0 - 3.0) as f32).collect();
        let edges: Vec<f32> = (0..512).map(|_| (next() * 4.0 - 2.0) as f32).collect();
        let unary_bf16 = Array::from_f32(&unary, &[2, 16])?.astype(Dtype::BFloat16)?;
        let unary_host = unary_bf16.astype(Dtype::Float32)?.to_f32()?;
        let cand = Array::from_u32(&candidates, &[2, 16])?;
        let edge = Array::from_f32(&edges, &[2, 16, 16])?;
        let law = |position: usize, predecessor: usize| -> Vec<f64> {
            let s: Vec<f64> = (0..16)
                .map(|c| {
                    let score = unary_host[position * 16 + c]
                        + edges[(position * 16 + predecessor) * 16 + c];
                    f64::from(cap * (score / cap).tanh()) / f64::from(t)
                })
                .collect();
            let top = s.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
            let w: Vec<f64> = s.iter().map(|&v| (v - top).exp()).collect();
            let z: f64 = w.iter().sum();
            w.iter().map(|v| v / z).collect()
        };
        let chi2_bound = |df: f64| {
            let z = 3.090_232_306;
            df * (1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt()).powi(3)
        };
        let chi2 = |counts: &[f64], law: &[f64]| -> f64 {
            let n: f64 = counts.iter().sum();
            counts
                .iter()
                .zip(law)
                .map(|(o, p)| (o - p * n).powi(2) / (p * n))
                .sum()
        };
        let run = |mutated: bool| -> Result<(f64, f64, f64)> {
            let mut first = vec![0f64; 16];
            let mut second = vec![vec![0f64; 16]; 16];
            let mut worst = 0f64;
            for batch in 0..DRAWS / BATCH {
                let draws = (0..BATCH)
                    .map(|i| {
                        let seed = (batch * BATCH + i) as u64 * 3 + 17;
                        ratio_walk(&cand, &unary_bf16, &edge, t, cap, seed)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let all: Vec<&Array> = draws.iter().flat_map(|(a, b, c)| [a, b, c]).collect();
                Array::eval_many(&all)?;
                for (i, (tokens, q, chosen)) in draws.iter().enumerate() {
                    let tokens = tokens.to_u32()?;
                    let q = q.to_f32()?;
                    let chosen = chosen.to_f32()?;
                    ensure!(
                        tokens.len() == 2 && q.len() == 32 && chosen.len() == 2,
                        "shapes"
                    );
                    let mut index = [0usize; 2];
                    for e in 0..2 {
                        index[e] = candidates[e * 16..(e + 1) * 16]
                            .iter()
                            .position(|&c| c == tokens[e])
                            .context("proposal outside its candidates")?;
                    }
                    if mutated {
                        // mutation: a draw from the host with another noise in place of the walk's own
                        let u = next_shared((batch * BATCH + i) as u64);
                        let l = law(0, 0);
                        let mut acc = 0.0;
                        index[0] = 15;
                        for (c, w) in l.iter().enumerate() {
                            acc += w;
                            if u < acc.powf(0.7) {
                                index[0] = c;
                                break;
                            }
                        }
                    }
                    for e in 0..2 {
                        let host = law(e, if e == 0 { 0 } else { index[0] });
                        if !mutated {
                            for c in 0..16 {
                                worst = worst.max((f64::from(q[e * 16 + c]) - host[c]).abs());
                            }
                            ensure!(
                                chosen[e] == q[e * 16 + index[e]],
                                "chosen is not q of the proposal"
                            );
                        }
                    }
                    first[index[0]] += 1.0;
                    second[index[0]][index[1]] += 1.0;
                }
            }
            let c0 = chi2(&first, &law(0, 0));
            let top = (0..16)
                .max_by(|&a, &b| first[a].total_cmp(&first[b]))
                .unwrap_or(0);
            let c1 = chi2(&second[top], &law(1, top));
            Ok((worst, c0, c1))
        };
        let (worst, c0, c1) = run(false)?;
        let bound = chi2_bound(15.0);
        println!(
            "S3 walk: q max error {worst:.2e}, chi2 first {c0:.2} second {c1:.2} (bound {bound:.2})"
        );
        ensure!(
            worst < 1e-5,
            "q of the walk differs from the host law by {worst}"
        );
        ensure!(
            c0 < bound && c1 < bound,
            "the walk does not draw from its law"
        );
        let (_, m0, _) = run(true)?;
        println!("S3 walk mutation: chi2 first {m0:.2}");
        ensure!(m0 > bound, "the mutated draw was not detected");
        Ok(())
    }

    /// A uniform in (0, 1) from `i` (splitmix64), for the walk test's mutation.
    fn next_shared(i: u64) -> f64 {
        let mut z = i.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    /// DFLASH2-2026-11-24-INK2M-S3: a DFlash2 session at T = 1 with the ratio rule, fixed width 7 (every block has
    /// proposals, all decided on the GPU), 96 tokens after the 200-token prompt of row en0: every block is a ratio
    /// block, tokens inside the vocabulary; prints the acceptance by position next to the point rule's (same
    /// session without the ratio rule).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL and MLXL3_INKLING_ROWS"]
    fn ratio_session_at_t1() -> Result<()> {
        use crate::inkling2::{DrawRule, DrawSeeds};
        use crate::inkling2_draft::DsparkSession;
        const TOKENS: usize = 96;
        let mut target = Inkling2::load(Path::new(&std::env::var("MLXL3_INKLING_MODEL")?))?;
        let mut dflash =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        let rows = rows_file()?;
        let vocab = target.unpadded_vocab() as u32;
        let (_, tokens) = row_tokens(&rows, 0, vocab)?;
        let prompt = tokens[..200].to_vec();
        let limit = usize::try_from(target.context_limit())?;
        for ratio in [false, true] {
            let width = dflash.max_proposals();
            let mut session = DsparkSession::new(&dflash, width)?
                .with_device_draw(DrawRule::Softmax(1.0), DrawSeeds::new(41), false)
                .with_device_rounds(true)
                .with_coupling(false)
                .with_ratio(ratio);
            ensure!(session.ratio_rule() == ratio, "ratio rule not applied");
            let logits = session.prefill(&mut target, &mut dflash, &prompt, &|| false)?;
            let mut next = logits.chat_greedy_ids()?[0];
            let mut out = vec![next];
            while out.len() < TOKENS {
                let mut choose =
                    |_: &[u32], _: Option<&[Vec<f32>]>, _: &Array| -> Result<(usize, u32)> {
                        anyhow::bail!("a block was not decided on the GPU")
                    };
                next = session.advance(
                    &mut target,
                    &mut dflash,
                    next,
                    limit,
                    TOKENS - out.len(),
                    &mut choose,
                )?;
                out.push(next);
            }
            let s = &session.stats;
            println!(
                "S3 session ratio {ratio}: {} blocks ({} ratio), {} accepted of {} proposed, by position {:?}",
                s.blocks, s.ratio_blocks, s.accepted, s.proposed, s.accepted_at
            );
            ensure!(
                out.iter().all(|&t| t < vocab),
                "token outside the vocabulary"
            );
            ensure!(s.blocks > 0 && s.proposed > 0, "no block with proposals");
            // every block with proposals is a ratio block; only the last one, left with one token of budget, has none
            ensure!(
                if ratio {
                    s.ratio_blocks + 1 >= s.blocks && s.ratio_blocks > 0
                } else {
                    s.ratio_blocks == 0
                },
                "ratio blocks {} of {}",
                s.ratio_blocks,
                s.blocks
            );
        }
        Ok(())
    }

    /// DFLASH2-2026-11-24-INK2M-S4: context batching is on unless the setting is "0".
    #[test]
    fn full_block_is_on_unless_zero() {
        assert!(full_block_setting(None));
        assert!(full_block_setting(Some("1")));
        assert!(full_block_setting(Some("")));
        assert!(!full_block_setting(Some("0")));
    }

    /// DFLASH2-2026-11-25-INK2M-F1: a request of 1 or 3 proposals reads the first rows of the trained 8-row block:
    /// on 64 real anchors (rows en0 and code0, 64 prefix tokens, teacher forcing) its candidates and proposals
    /// are those of the first slots of the 7-proposal request (the head on 1 or 3 rows may round differently from
    /// the head on 7: at least 98 % of the cases, scores within 1e-2 relative), on the host path and on the GPU
    /// path. Mutation: with blocks of `count + 1` rows the first proposal of a 1-proposal request matches in
    /// fewer than 90 % of the cases.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL and MLXL3_INKLING_ROWS"]
    fn narrow_requests_use_the_full_block() -> Result<()> {
        const PREFIX: usize = 64;
        const ANCHORS: usize = 32;
        let mut target = Inkling2::load(Path::new(&std::env::var("MLXL3_INKLING_MODEL")?))?;
        let mut dflash =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        dflash.set_full_block(true);
        let taps = dflash.taps().to_vec();
        let k = dflash.config().dflash_config.selector_top_k as usize;
        let rows = rows_file()?;
        let vocab = target.unpadded_vocab() as u32;
        // [count 1, count 3]: host candidates, host proposals, GPU proposals equal to the 7-proposal request's
        let mut equal = [[0usize; 3]; 2];
        let (mut narrow_first, mut cases, mut worst) = (0usize, 0usize, 0f64);
        for index in [0usize, 6] {
            let (name, tokens) = row_tokens(&rows, index, vocab)?;
            ensure!(tokens.len() > PREFIX + ANCHORS + 1, "row {name} too short");
            let (features, _) = prefill(&mut target, &tokens[..PREFIX], &taps)?;
            dflash.reset();
            dflash.append_context(&features)?;
            for &anchor in &tokens[PREFIX..PREFIX + ANCHORS] {
                let whole = dflash.propose_debug(&target, anchor, 7)?;
                let anchor_ids = Array::from_u32(&[anchor], &[1])?;
                let whole_device = dflash.propose_device(&target, &anchor_ids, 7)?.to_u32()?;
                ensure!(
                    whole.proposals.len() == 7 && whole_device.len() == 7,
                    "7 proposals"
                );
                for (slot, count) in [1usize, 3].into_iter().enumerate() {
                    let part = dflash.propose_debug(&target, anchor, count)?;
                    ensure!(
                        part.candidates.len() == count * k && part.scores.len() == count * k * k,
                        "narrow shapes"
                    );
                    if part.candidates == whole.candidates[..count * k] {
                        equal[slot][0] += 1;
                        let rel = part
                            .scores
                            .iter()
                            .zip(&whole.scores[..count * k * k])
                            .map(|(a, b)| f64::from((a - b).abs()) / f64::from(b.abs().max(1.0)))
                            .fold(0.0, f64::max);
                        worst = worst.max(rel);
                    }
                    if part.proposals == whole.proposals[..count] {
                        equal[slot][1] += 1;
                    }
                    let device = dflash
                        .propose_device(&target, &anchor_ids, count)?
                        .to_u32()?;
                    ensure!(device.len() == count, "GPU proposals");
                    if device == whole_device[..count] {
                        equal[slot][2] += 1;
                    }
                }
                // mutation: the block of 2 rows
                dflash.set_full_block(false);
                if dflash.propose_debug(&target, anchor, 1)?.proposals[0] == whole.proposals[0] {
                    narrow_first += 1;
                }
                dflash.set_full_block(true);
                cases += 1;
                let (_, features) = target.forward_with_features(&[anchor], &taps)?;
                dflash.append_context(&features)?;
            }
        }
        println!(
            "F1 {cases} anchors: equal to the 7-proposal request [candidates, proposals, GPU proposals] count 1 {:?}, count 3 {:?}; scores rel max {worst:.2e}; blocks of 2 rows: first proposal equal {narrow_first}",
            equal[0], equal[1]
        );
        ensure!(cases == 2 * ANCHORS, "anchors");
        for row in &equal {
            for &n in row {
                ensure!(
                    n * 100 >= cases * 98,
                    "narrow request differs from the whole block: {equal:?}"
                );
            }
        }
        ensure!(worst < 1e-2, "scores differ by {worst}");
        ensure!(
            narrow_first * 100 < cases * 90,
            "the mutation (blocks of count + 1 rows) is not detected: {narrow_first} / {cases}"
        );
        Ok(())
    }

    #[test]
    fn batch_context_is_on_unless_zero() {
        assert!(batch_context_setting(None));
        assert!(batch_context_setting(Some("1")));
        assert!(batch_context_setting(Some("")));
        assert!(!batch_context_setting(Some("0")));
    }

    /// The caches' (offset, keys, values) as host values (`[1, KV, S, D]` flattened).
    fn cache_values(draft: &Inkling2Dflash2) -> Result<Vec<CacheValues>> {
        draft.flush_context()?;
        draft
            .caches
            .borrow()
            .iter()
            .map(|c| {
                let keys = c.keys.as_ref().context("no keys")?;
                let values = c.values.as_ref().context("no values")?;
                Ok((
                    c.offset,
                    keys.shape().to_vec(),
                    keys.astype(Dtype::Float32)?.to_f32()?,
                    values.astype(Dtype::Float32)?.to_f32()?,
                ))
            })
            .collect()
    }

    /// DFLASH2-2026-11-24-INK2M-S4: context features added by batches (deferred, then one call) give the caches
    /// of single appends: same offsets and shapes, K/V within 1e-3 relative, finite, across the sliding window
    /// (2 100 rows of context, then 40 single appends and a D1 snapshot after deferred features); the block
    /// hidden rows built on them agree within 1e-2 relative; `offset` counts the deferred rows. A mutation (one
    /// deferred row dropped) is detected by the offsets.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL and MLXL3_INKLING_ROWS"]
    fn batched_context_matches_single_appends() -> Result<()> {
        const BASE: usize = 2100;
        const SINGLE: usize = 40;
        let mut target = Inkling2::load(Path::new(&std::env::var("MLXL3_INKLING_MODEL")?))?;
        let mut dflash =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        let taps = dflash.taps().to_vec();
        let rows = rows_file()?;
        let vocab = target.unpadded_vocab() as u32;
        let mut text = Vec::new();
        for index in 0..rows["rows"].as_array().context("rows")?.len() {
            text.extend(row_tokens(&rows, index, vocab)?.1);
        }
        ensure!(text.len() > BASE + SINGLE + 1, "not enough text");
        let (base, _) = prefill(&mut target, &text[..BASE], &taps)?;
        let mut singles = Vec::new();
        for &token in &text[BASE..BASE + SINGLE] {
            let (_, f) = target.forward_with_features(&[token], &taps)?;
            f.eval()?;
            singles.push(f);
        }
        let relative = |a: &[f32], b: &[f32]| -> f64 {
            let num: f64 = a
                .iter()
                .zip(b)
                .map(|(x, y)| f64::from(x - y).powi(2))
                .sum::<f64>()
                .sqrt();
            let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum::<f64>().sqrt();
            num / den.max(f64::MIN_POSITIVE)
        };
        let run = |dflash: &mut Inkling2Dflash2, batch: bool, drop_one: bool| -> Result<_> {
            dflash.reset();
            dflash.set_batch_context(batch)?;
            dflash.append_context(&base)?;
            for (i, f) in singles.iter().enumerate() {
                if drop_one && i == SINGLE / 2 {
                    continue;
                }
                dflash.append_context(f)?;
            }
            let offset = dflash.offset();
            let snapshot = dflash.context_snapshot()?;
            let caches = cache_values(dflash)?;
            let mask = dflash.config().dflash_config.mask_token_id;
            let ids = [
                text[BASE + SINGLE],
                mask,
                mask,
                mask,
                mask,
                mask,
                mask,
                mask,
            ];
            let embeds = target.embed_tokens(&Array::from_u32(&ids, &[8])?)?;
            let hidden = dflash.block_hidden(&embeds)?.to_f32()?;
            Ok((offset, snapshot.offset(), caches, hidden))
        };
        let (offset_a, snap_a, caches_a, hidden_a) = run(&mut dflash, false, false)?;
        let (offset_b, snap_b, caches_b, hidden_b) = run(&mut dflash, true, false)?;
        assert_eq!(offset_a, (BASE + SINGLE) as i32);
        assert_eq!(offset_b, offset_a, "offset with deferred rows");
        assert_eq!(snap_a, Some(offset_a));
        assert_eq!(snap_b, snap_a, "D1 snapshot after deferred features");
        assert_eq!(caches_a.len(), caches_b.len());
        let mut worst = 0f64;
        for ((oa, sa, ka, va), (ob, sb, kb, vb)) in caches_a.iter().zip(&caches_b) {
            assert_eq!(oa, ob, "cache offsets");
            assert_eq!(sa, sb, "cache shapes");
            assert_eq!(sa[2], dflash.config().sliding_window, "window not reached");
            assert!(kb.iter().chain(vb).all(|v| v.is_finite()), "finite caches");
            worst = worst.max(relative(kb, ka)).max(relative(vb, va));
        }
        let hidden_rel = relative(&hidden_b, &hidden_a);
        println!("S4 batched context: caches rel {worst:.2e}, block hidden rel {hidden_rel:.2e}");
        assert!(worst < 1e-3, "caches rel {worst}");
        assert!(hidden_rel < 1e-2, "block hidden rel {hidden_rel}");
        let (offset_m, _, _, _) = run(&mut dflash, true, true)?;
        assert_ne!(offset_m, offset_a, "the dropped row was not detected");
        dflash.set_batch_context(true)?;
        Ok(())
    }

    /// DFLASH2-2026-11-24-INK2M-S4 (diagnostic): drafter time of a 7-proposal block (GPU path, waited) after N plain
    /// steps whose features fed the context, at 2 600 tokens of real context, N = 0 / 8 / 32 / 64, context
    /// batching off / on / on / off, median of 5 per (mode, N); prints also the plain-step medians.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL and MLXL3_INKLING_ROWS"]
    fn block_cost_after_plain_steps() -> Result<()> {
        const CONTEXT: usize = 2600;
        let mut target = Inkling2::load(Path::new(&std::env::var("MLXL3_INKLING_MODEL")?))?;
        let mut dflash =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        let taps = dflash.taps().to_vec();
        let rows = rows_file()?;
        let vocab = target.unpadded_vocab() as u32;
        let mut text = Vec::new();
        for index in 0..rows["rows"].as_array().context("rows")?.len() {
            text.extend(row_tokens(&rows, index, vocab)?.1);
        }
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let (features, _) = prefill(&mut target, &text[..CONTEXT], &taps)?;
        let mut position = CONTEXT;
        let arms = [
            (false, false),
            (true, false),
            (true, true),
            (true, true),
            (true, false),
            (false, false),
        ];
        for (arm, (batch, eager)) in arms.into_iter().enumerate() {
            dflash.reset();
            dflash.set_batch_context(batch)?;
            dflash.set_eager_features(eager);
            // the same 2 600 rows of context in every arm (the target itself keeps going along the text)
            dflash.append_context(&features)?;
            dflash.flush_context()?;
            for n in [0usize, 8, 32, 64] {
                let (mut draft, mut plain, mut flush) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..6 {
                    for _ in 0..n {
                        let t = std::time::Instant::now();
                        let (logits, f) = target.forward_with_features(&[text[position]], &taps)?;
                        logits.eval()?;
                        plain.push(t.elapsed().as_secs_f64() * 1e3);
                        dflash.append_context(&f)?;
                        position += 1;
                    }
                    // S4b: the deferred (or lazy) context first, then the proposals alone
                    let t = std::time::Instant::now();
                    dflash.flush_context()?;
                    {
                        let caches = dflash.caches.borrow();
                        let all: Vec<&Array> = caches
                            .iter()
                            .flat_map(|c| [c.keys.as_ref(), c.values.as_ref()])
                            .flatten()
                            .collect();
                        Array::eval_many(&all)?;
                    }
                    flush.push(t.elapsed().as_secs_f64() * 1e3);
                    let t = std::time::Instant::now();
                    let proposals = dflash
                        .propose_device(&target, &Array::from_u32(&[text[position]], &[1])?, 7)?
                        .to_u32()?;
                    draft.push(t.elapsed().as_secs_f64() * 1e3);
                    ensure!(proposals.len() == 7, "proposals");
                    // the anchor's plain step (the block's verification is not timed here)
                    let (_, f) = target.forward_with_features(&[text[position]], &taps)?;
                    dflash.append_context(&f)?;
                    position += 1;
                }
                draft.remove(0);
                flush.remove(0);
                println!(
                    "S4DIAG arm {arm} batch {batch} eager {eager} N {n}: context {:.2} ms, then proposals {:.2} ms (medians of 5), plain {:.2} ms",
                    median(flush),
                    median(draft),
                    if plain.is_empty() { 0.0 } else { median(plain) }
                );
            }
        }
        Ok(())
    }
    /// (ratio, point, max of q, entropy of q) of one slot at temperature `t`: `slot_acceptance_normed` without its
    /// coupled estimate, plus the drafter's own confidence.
    fn slot_trace(
        logits: &[f32],
        (max, log_z): (f64, f64),
        candidates: &[u32],
        scores: &[f32],
        t: f32,
    ) -> (f64, f64, f64, f64) {
        let t = f64::from(t);
        let p: Vec<f64> = candidates
            .iter()
            .map(|&c| ((f64::from(logits[c as usize]) - max) / t - log_z).exp())
            .collect();
        let s_max = f64::from(scores.iter().copied().fold(f32::NEG_INFINITY, f32::max));
        let w: Vec<f64> = scores
            .iter()
            .map(|&s| ((f64::from(s) - s_max) / t).exp())
            .collect();
        let w_sum: f64 = w.iter().sum();
        let q: Vec<f64> = w.iter().map(|v| v / w_sum).collect();
        let best = (0..scores.len())
            .max_by(|&a, &b| scores[a].total_cmp(&scores[b]).then(b.cmp(&a)))
            .unwrap_or(0);
        let ratio: f64 = p.iter().zip(&q).map(|(a, b)| a.min(*b)).sum();
        let q_max = q.iter().copied().fold(0.0, f64::max);
        let q_entropy: f64 = -q
            .iter()
            .filter(|v| **v > 0.0)
            .map(|v| v * v.ln())
            .sum::<f64>();
        (ratio, p[best], q_max, q_entropy)
    }

    /// `slot_trace` on the synthetic laws of `slot_acceptance_rules_on_synthetic_laws`: same ratio and point
    /// values, q's maximum and entropy exact.
    #[test]
    fn slot_trace_on_synthetic_laws() {
        let logits: Vec<f32> = [0.4f32, 0.3, 0.2, 0.1].iter().map(|v| v.ln()).collect();
        let norm = log_norm(&logits, 1.0);
        let (ratio, point, q_max, q_entropy) =
            slot_trace(&logits, norm, &[0, 1, 2, 3], &logits, 1.0);
        assert!((ratio - 1.0).abs() < 1e-6 && (point - 0.4).abs() < 1e-6);
        assert!((q_max - 0.4).abs() < 1e-6);
        let h: f64 = -[0.4f64, 0.3, 0.2, 0.1]
            .iter()
            .map(|v| v * v.ln())
            .sum::<f64>();
        assert!((q_entropy - h).abs() < 1e-6, "entropy {q_entropy} {h}");
        let (ratio, point, q_max, q_entropy) =
            slot_trace(&logits, norm, &[0, 1, 2, 3], &[0f32; 4], 1.0);
        assert!((ratio - 0.8).abs() < 1e-6 && (point - 0.4).abs() < 1e-6);
        assert!((q_max - 0.25).abs() < 1e-9 && (q_entropy - 4f64.ln()).abs() < 1e-9);
    }

    /// DFLASH2-2026-11-25-INK2M-R0 (a, b; diagnostic): the trace the policy simulator
    /// (`benchmarks/inkling2_dflash2_policy_sim.py`) replays. Every row of `MLXL3_INKLING_ROWS`, 64 prefix tokens
    /// then 192 anchors in teacher forcing on the real text; at each anchor the blocks of 1, 3 and 7 proposals
    /// and, for every slot whose real predecessors are among their slots' candidates, the ratio and point
    /// acceptances at T = 1 and the maximum and entropy of q; per position the target's log p of the real next
    /// token, its entropy and its maximum. Writes `policy-trace.json` under `MLXL3_DFLASH2_ORACLE_OUT` and prints
    /// the per-count slot means (R0-a: narrow blocks against the trained 8-row block).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL, MLXL3_INKLING_ROWS, MLXL3_DFLASH2_ORACLE_OUT"]
    fn policy_trace_t1() -> Result<()> {
        const PREFIX: usize = 64;
        const ANCHORS: usize = 192;
        const T: f32 = 1.0;
        let counts = [1usize, 3, 7];
        let mut target = Inkling2::load(Path::new(&std::env::var("MLXL3_INKLING_MODEL")?))?;
        let mut dflash =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        let out = std::path::PathBuf::from(std::env::var("MLXL3_DFLASH2_ORACLE_OUT")?);
        std::fs::create_dir_all(&out)?;
        let taps = dflash.taps().to_vec();
        let width = dflash.max_proposals();
        ensure!(width == 7, "the trace expects 7 proposals");
        let k = dflash.config().dflash_config.selector_top_k as usize;
        let rows = rows_file()?;
        let vocab = target.unpadded_vocab() as u32;
        let row_count = rows["rows"].as_array().context("rows")?.len();
        let mut traced = Vec::new();
        // [count][slot]: ratio sum, point sum, reached
        let mut total = vec![vec![(0f64, 0f64, 0usize); width]; counts.len()];
        let (mut same, mut anchors) = (vec![0usize; counts.len()], 0usize);
        let sampled = std::env::var("MLXL3_DFLASH2_TRACE_SAMPLED").as_deref() == Ok("1");
        println!("R0 drafter bits {}, sampled text {sampled}", dflash.bits());
        for index in 0..row_count {
            let (name, mut tokens) = row_tokens(&rows, index, vocab)?;
            if tokens.len() <= PREFIX + ANCHORS + width + 1 {
                println!("R0 {name}: skipped ({} tokens)", tokens.len());
                continue;
            }
            if sampled {
                // R0-d: the text after the real prefix is drawn from the target at T = 1 (whole vocabulary,
                // inversion on the host, a fixed seed per row)
                let (_, mut logits) = prefill(&mut target, &tokens[..PREFIX], &taps)?;
                tokens.truncate(PREFIX);
                let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ ((index as u64 + 1) << 32);
                for _ in 0..ANCHORS + width + 2 {
                    let row = logits.reshape(&[vocab as i32])?.to_f32()?;
                    ensure!(row.len() == vocab as usize, "sampled row");
                    let (max, log_z) = log_norm(&row, T);
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let u = ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
                    let mut cumulative = 0f64;
                    let mut pick = vocab - 1;
                    for (id, &l) in row.iter().enumerate() {
                        cumulative += ((f64::from(l) - max) / f64::from(T) - log_z).exp();
                        if cumulative >= u {
                            pick = id as u32;
                            break;
                        }
                    }
                    tokens.push(pick);
                    logits = target.forward_with_features(&[pick], &taps)?.0;
                }
            }
            let (features, _) = prefill(&mut target, &tokens[..PREFIX], &taps)?;
            dflash.reset();
            dflash.append_context(&features)?;
            // blocks[a][ci] = (candidates, scores, first proposal)
            let mut blocks: Vec<Vec<TracedBlock>> = Vec::with_capacity(ANCHORS);
            let mut target_rows: Vec<(Vec<f32>, (f64, f64))> = Vec::with_capacity(ANCHORS + width);
            let (mut lp_real, mut entropy, mut p_max) = (Vec::new(), Vec::new(), Vec::new());
            for i in PREFIX..PREFIX + ANCHORS + width {
                if i < PREFIX + ANCHORS {
                    let mut per_count = Vec::with_capacity(counts.len());
                    for &count in &counts {
                        let block = dflash.propose_debug(&target, tokens[i], count)?;
                        ensure!(
                            block.candidates.len() == count * k
                                && block.scores.len() == count * k * k
                                && block.proposals.len() == count
                                && block.scores.iter().all(|v| v.is_finite()),
                            "block shapes"
                        );
                        per_count.push((block.candidates, block.scores, block.proposals[0]));
                    }
                    blocks.push(per_count);
                }
                let (logits, features) = target.forward_with_features(&tokens[i..i + 1], &taps)?;
                let row = logits.reshape(&[vocab as i32])?.to_f32()?;
                ensure!(
                    row.len() == vocab as usize && row.iter().all(|v| v.is_finite()),
                    "target row"
                );
                let (max, log_z) = log_norm(&row, T);
                // the row after tokens[i] predicts tokens[i + 1]
                let mut h = 0f64;
                let mut top = 0f64;
                for &l in &row {
                    let lp = (f64::from(l) - max) / f64::from(T) - log_z;
                    let p = lp.exp();
                    if p > 0.0 {
                        h -= p * lp;
                    }
                    top = top.max(p);
                }
                lp_real.push((f64::from(row[tokens[i + 1] as usize]) - max) / f64::from(T) - log_z);
                entropy.push(h);
                p_max.push(top);
                target_rows.push((row, (max, log_z)));
                dflash.append_context(&features)?;
            }
            let mut by_count = serde_json::Map::new();
            for (ci, &count) in counts.iter().enumerate() {
                let (mut ratio_rows, mut point_rows, mut qmax_rows, mut qent_rows) =
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new());
                for (a, per_count) in blocks.iter().enumerate() {
                    let j = PREFIX + a;
                    let (candidates, scores, first) = &per_count[ci];
                    if *first == per_count[counts.len() - 1].2 {
                        same[ci] += 1;
                    }
                    // -1: slot not reached (a real predecessor is not among its slot's candidates)
                    let (mut ratios, mut points, mut qmaxs, mut qents) = (
                        vec![-1f64; count],
                        vec![-1f64; count],
                        vec![-1f64; count],
                        vec![-1f64; count],
                    );
                    for e in 0..count {
                        let p = if e == 0 {
                            0
                        } else {
                            match candidates[(e - 1) * k..e * k]
                                .iter()
                                .position(|&c| c == tokens[j + e])
                            {
                                Some(p) => p,
                                None => break,
                            }
                        };
                        let (logits, norm) = &target_rows[a + e];
                        let (ratio, point, q_max, q_entropy) = slot_trace(
                            logits,
                            *norm,
                            &candidates[e * k..(e + 1) * k],
                            &scores[(e * k + p) * k..(e * k + p + 1) * k],
                            T,
                        );
                        ensure!(
                            [ratio, point, q_max, q_entropy]
                                .iter()
                                .all(|v| v.is_finite())
                                && (0.0..=1.0 + 1e-9).contains(&ratio),
                            "slot values"
                        );
                        ratios[e] = ratio;
                        points[e] = point;
                        qmaxs[e] = q_max;
                        qents[e] = q_entropy;
                        let slot = &mut total[ci][e];
                        slot.0 += ratio;
                        slot.1 += point;
                        slot.2 += 1;
                    }
                    ratio_rows.push(ratios);
                    point_rows.push(points);
                    qmax_rows.push(qmaxs);
                    qent_rows.push(qents);
                }
                by_count.insert(
                    count.to_string(),
                    serde_json::json!({
                        "ratio": ratio_rows, "point": point_rows, "q_max": qmax_rows, "q_entropy": qent_rows,
                    }),
                );
            }
            anchors += ANCHORS;
            traced.push(serde_json::json!({
                "name": name, "lp_real": lp_real, "entropy": entropy, "p_max": p_max, "blocks": by_count,
            }));
            println!("R0 {name}: traced");
        }
        ensure!(anchors >= 6 * ANCHORS, "only {anchors} anchors traced");
        for (ci, &count) in counts.iter().enumerate() {
            let mean = |pick: fn(&(f64, f64, usize)) -> f64| -> Vec<String> {
                total[ci]
                    .iter()
                    .take(count)
                    .map(|slot| format!("{:.3}", pick(slot) / slot.2.max(1) as f64))
                    .collect()
            };
            let reached: Vec<usize> = total[ci].iter().take(count).map(|slot| slot.2).collect();
            println!(
                "R0 total count {count}: ratio by slot [{}], point by slot [{}], reached {reached:?} of {anchors}, first proposal equal to the 7-block's {} / {anchors}",
                mean(|s| s.0).join(", "),
                mean(|s| s.1).join(", "),
                same[ci]
            );
        }
        let path = out.join(if sampled {
            "policy-trace-sampled.json"
        } else {
            "policy-trace.json"
        });
        std::fs::write(
            &path,
            serde_json::to_vec(&serde_json::json!({
                "temperature": T, "prefix": PREFIX, "anchors": ANCHORS, "counts": counts,
                "bits": dflash.bits(), "sampled": sampled, "rows": traced,
            }))?,
        )?;
        println!("R0 trace written to {}", path.display());
        Ok(())
    }

    /// DFLASH2-2026-11-25-INK2M-R0 (c; diagnostic): where a block of 7 proposals spends its time at T = 1, each
    /// stage evaluated alone after a real context of 200 then 2 600 tokens: the block's hidden rows, the candidate
    /// head on 7 rows, the whole drafter (`propose_ratio_device`), the verification of 8 rows, the ratio decision
    /// on evaluated logits, the point decision for comparison, and the commit. Medians of 9 after 2 warm-ups.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_DFLASH2_MODEL and MLXL3_INKLING_ROWS"]
    fn block_stage_cost() -> Result<()> {
        let mut target = Inkling2::load(Path::new(&std::env::var("MLXL3_INKLING_MODEL")?))?;
        let mut dflash =
            Inkling2Dflash2::load(Path::new(&std::env::var("MLXL3_DFLASH2_MODEL")?), &target)?;
        let taps = dflash.taps().to_vec();
        let rows = rows_file()?;
        let vocab = target.unpadded_vocab() as u32;
        let mut text = Vec::new();
        for index in 0..rows["rows"].as_array().context("rows")?.len() {
            text.extend(row_tokens(&rows, index, vocab)?.1);
        }
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        let ms = |t: std::time::Instant| t.elapsed().as_secs_f64() * 1e3;
        let mask = dflash.config().dflash_config.mask_token_id;
        const STAGES: [&str; 8] = [
            "hidden", "head7", "drafter", "verify8", "ratio", "point", "commit", "plain",
        ];
        for context in [200usize, 2600] {
            ensure!(text.len() > context + 64, "not enough text");
            let (features, _) = prefill(&mut target, &text[..context], &taps)?;
            dflash.reset();
            dflash.append_context(&features)?;
            dflash.flush_context()?;
            let mut position = context;
            let mut times = vec![Vec::new(); STAGES.len()];
            for round in 0..11u64 {
                let anchor = text[position];
                let mut ids = vec![mask; 8];
                ids[0] = anchor;
                let embeds = target.embed_tokens(&Array::from_u32(&ids, &[8])?)?;
                embeds.eval()?;
                let t = std::time::Instant::now();
                let hidden = dflash.block_hidden(&embeds)?;
                hidden.eval()?;
                let hidden_ms = ms(t);
                let predicted = hidden.slice(0, 1, 8)?.astype(Dtype::Float16)?;
                predicted.eval()?;
                let t = std::time::Instant::now();
                let head = target
                    .head_scores(&predicted)?
                    .slice(1, 0, vocab as i32)?
                    .astype(Dtype::BFloat16)?;
                head.eval()?;
                let head_ms = ms(t);
                let anchor_ids = Array::from_u32(&[anchor], &[1])?;
                let t = std::time::Instant::now();
                let drawn =
                    dflash.propose_ratio_device(&target, &anchor_ids, 7, 1.0, 77 + round)?;
                Array::eval_many(&[&drawn.tokens, &drawn.candidates, &drawn.q, &drawn.chosen])?;
                let drafter_ms = ms(t);
                let block = Array::concatenate(&[&anchor_ids, &drawn.tokens], 0)?;
                block.eval()?;
                let t = std::time::Instant::now();
                let (logits, features) = target.verify_device(&block, &taps)?;
                Array::eval_many(&[&logits, &features])?;
                let verify_ms = ms(t);
                ensure!(
                    logits.shape() == [1, 8, vocab as i32],
                    "verified logits shape"
                );
                let t = std::time::Instant::now();
                let packet = crate::inkling2::decide_ratio(
                    &logits,
                    &drawn.tokens,
                    &drawn.candidates,
                    &drawn.q,
                    &drawn.chosen,
                    1.0,
                    1000 + round,
                    2000 + round,
                )?
                .to_u32()?;
                let ratio_ms = ms(t);
                ensure!(
                    packet.len() == 9 && packet[0] <= 7 && packet[1] < vocab,
                    "ratio packet"
                );
                let t = std::time::Instant::now();
                let point = crate::inkling2::decide_block(
                    &logits,
                    &drawn.tokens,
                    crate::inkling2::DrawRule::Softmax(1.0),
                    3000 + round,
                    4000 + round,
                )?
                .to_u32()?;
                let point_ms = ms(t);
                ensure!(
                    point.len() == 9 && point[0] <= 7 && point[1] < vocab,
                    "point packet"
                );
                // the real text goes on: only the anchor is kept
                let t = std::time::Instant::now();
                target.commit(1)?;
                let commit_ms = ms(t);
                dflash.append_context(&features.slice(0, 0, 1)?)?;
                dflash.flush_context()?;
                position += 1;
                let t = std::time::Instant::now();
                let (logits, features) = target.forward_with_features(&[text[position]], &taps)?;
                logits.eval()?;
                let plain_ms = ms(t);
                dflash.append_context(&features)?;
                dflash.flush_context()?;
                position += 1;
                if round >= 2 {
                    for (slot, value) in [
                        hidden_ms, head_ms, drafter_ms, verify_ms, ratio_ms, point_ms, commit_ms,
                        plain_ms,
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        times[slot].push(value);
                    }
                }
            }
            let line: Vec<String> = STAGES
                .iter()
                .zip(times)
                .map(|(name, values)| format!("{name} {:.2}", median(values)))
                .collect();
            println!("R0STAGE context {context}: {} ms", line.join(", "));
        }
        Ok(())
    }
}
