//! Inkling 2 Mini bundled DSpark drafter (`mtp.safetensors`): DFlare feature fusion, three dense draft layers
//! with separate context/block key-value projections, and an RNN Markov head over the target vocabulary.
//!
//! References (read, not executed): mlx-vlm Neywa-Labs@111ac668 `speculative/drafters/inkling2_dspark/dspark.py`
//! (adapted from vLLM), SGLang@6515ac29 `inkling2_mtp.py`, Ollama@cecac8bf `mlxrunner/model/inkling2/draft.go`.
//! Features: the target residual stream before visits `target_layer_ids[..-1]` and the final normalized hidden
//! (last index = visit count). Only target context enters the persistent draft KV; proposals never do.
use crate::{
    array::{Array, Dtype},
    checkpoint::{self, Checkpoint},
    inkling2::{Inkling2, Linear as DraftLinear, bits_from_env},
    lfm2::half_weight,
    linear::checkpoint_array,
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{fs::File, path::Path, time::Instant};

#[derive(Deserialize)]
struct Root {
    mtp_config: DraftConfig,
}

#[derive(Clone, Deserialize)]
pub struct DraftConfig {
    pub hidden_size: i32,
    pub num_hidden_layers: usize,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub intermediate_size: i32,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub local_layer_ids: Vec<usize>,
    pub sliding_window_size: i32,
    pub block_size: usize,
    pub markov_rank: i32,
    pub markov_head_type: String,
    pub target_layer_ids: Vec<usize>,
    /// Affine bits of the draft projections (16 = FP16 dense), from `MLXL3_INKLING_DSPARK_BITS`.
    #[serde(skip)]
    pub layer_bits: i32,
    /// Affine bits of the draft-only head copy and Markov fold, from `MLXL3_INKLING_DSPARK_HEAD_BITS`.
    #[serde(skip)]
    pub head_bits: i32,
    #[serde(default)]
    pub q_bias: bool,
    #[serde(default)]
    pub o_bias: bool,
}

/// `(prefix, input, output)` of every dense draft projection, in load order (the nine `DraftLinear` of a layer);
/// used by the slim exporter.
#[cfg(test)]
pub(crate) fn draft_projections(c: &DraftConfig) -> Vec<(String, i32, i32)> {
    let (h, q_dim, kv_dim, inter) = (
        c.hidden_size,
        c.num_attention_heads * c.head_dim,
        c.num_key_value_heads * c.head_dim,
        c.intermediate_size,
    );
    let mut list = Vec::new();
    for layer in 0..c.num_hidden_layers {
        let a = format!("layers.{layer}.self_attn");
        let m = format!("layers.{layer}.mlp");
        list.extend([
            (format!("{a}.q_proj"), h, q_dim),
            (format!("{a}.k_proj"), h, kv_dim),
            (format!("{a}.v_proj"), h, kv_dim),
            (format!("{a}.k_proj_ctx"), h, kv_dim),
            (format!("{a}.v_proj_ctx"), h, kv_dim),
            (format!("{a}.o_proj"), q_dim, h),
            (format!("{m}.gate_proj"), h, inter),
            (format!("{m}.up_proj"), h, inter),
            (format!("{m}.down_proj"), inter, h),
        ]);
    }
    list
}

fn f32_weight(checkpoint: &Checkpoint, name: &str, shape: &[i32]) -> Result<Array> {
    let weight = checkpoint_array(checkpoint, name)?;
    ensure!(
        weight.shape() == shape,
        "{name}: shape {:?}, expected {shape:?}",
        weight.shape()
    );
    let weight = weight.astype(Dtype::Float32)?;
    weight.eval()?;
    Ok(weight)
}

struct DraftAttention {
    q: DraftLinear,
    k: DraftLinear,
    v: DraftLinear,
    k_ctx: DraftLinear,
    v_ctx: DraftLinear,
    o: DraftLinear,
    q_norm: Array,
    k_norm: Array,
    k_norm_ctx: Array,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    window: Option<i32>,
    theta: f32,
    eps: f32,
}

#[derive(Default, Clone)]
pub(crate) struct DraftCache {
    /// `[1, Hkv, S, D]` FP16 context keys (RoPE applied) and values.
    pub(crate) keys: Option<Array>,
    pub(crate) values: Option<Array>,
    /// Absolute position of the next context token.
    pub(crate) offset: i32,
}

impl DraftCache {
    /// D1: the cache with its views copied to their own rows (same values).
    pub(crate) fn compacted(&self) -> Result<Self> {
        let compact = |a: &Option<Array>| a.as_ref().map(Array::compact_for_cache).transpose();
        Ok(Self {
            keys: compact(&self.keys)?,
            values: compact(&self.values)?,
            offset: self.offset,
        })
    }
}

/// DSPARK-2026-11-24-INK2M-MAYA2-D1: the drafter's context caches at a prefill chunk boundary
/// (`Inkling2Draft::context_snapshot`).
pub struct DraftContext {
    pub(crate) caches: Vec<DraftCache>,
}

impl DraftContext {
    /// Absolute position of the next context token (the same in every layer).
    pub fn offset(&self) -> Option<i32> {
        self.caches.first().map(|c| c.offset)
    }

    /// Bytes these caches keep alive.
    pub fn retained_bytes(&self) -> Result<usize> {
        let mut bytes = 0usize;
        for a in self
            .caches
            .iter()
            .flat_map(|c| [&c.keys, &c.values])
            .flatten()
        {
            bytes = bytes
                .checked_add(a.retained_bytes()?)
                .context("DSpark context size overflow")?;
        }
        Ok(bytes)
    }
}

impl DraftAttention {
    fn load(ckpt: &Checkpoint, layer: usize, c: &DraftConfig) -> Result<Self> {
        let p = format!("layers.{layer}.self_attn");
        let (h, q_dim, kv_dim) = (
            c.hidden_size,
            c.num_attention_heads * c.head_dim,
            c.num_key_value_heads * c.head_dim,
        );
        let proj = |name: &str, i: i32, o: i32| {
            DraftLinear::load(ckpt, &format!("{p}.{name}"), i, o, c.layer_bits)
        };
        Ok(Self {
            q: proj("q_proj", h, q_dim)?,
            k: proj("k_proj", h, kv_dim)?,
            v: proj("v_proj", h, kv_dim)?,
            k_ctx: proj("k_proj_ctx", h, kv_dim)?,
            v_ctx: proj("v_proj_ctx", h, kv_dim)?,
            o: proj("o_proj", q_dim, h)?,
            q_norm: f32_weight(ckpt, &format!("{p}.q_norm.weight"), &[c.head_dim])?,
            k_norm: f32_weight(ckpt, &format!("{p}.k_norm.weight"), &[c.head_dim])?,
            k_norm_ctx: f32_weight(ckpt, &format!("{p}.k_norm_ctx.weight"), &[c.head_dim])?,
            heads: c.num_attention_heads,
            kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim,
            window: c
                .local_layer_ids
                .contains(&layer)
                .then_some(c.sliding_window_size),
            theta: c.rope_theta,
            eps: c.rms_norm_eps,
        })
    }

    fn heads(&self, x: &Array, heads: i32, norm: Option<&Array>) -> Result<Array> {
        let rows = x.shape()[0];
        let x = x.reshape(&[rows, heads, self.head_dim])?;
        let x = match norm {
            Some(w) => x
                .astype(Dtype::Float32)?
                .rms_norm(w, self.eps)?
                .astype(Dtype::Float16)?,
            None => x,
        };
        x.transpose(&[1, 0, 2])?
            .reshape(&[1, heads, rows, self.head_dim])
    }

    /// Append context rows `[Tc, D]` (FP16, mixed features) at the cache offset.
    fn append_context(&self, context: &Array, cache: &mut DraftCache) -> Result<()> {
        let mut context = context.try_clone()?;
        let mut rows = context.shape()[0];
        if let Some(window) = self.window
            && rows > window
        {
            let skip = rows - window;
            context = context.slice(0, skip, rows)?;
            cache.offset += skip;
            rows = window;
        }
        let k = self
            .heads(
                &self.k_ctx.forward(&context)?,
                self.kv_heads,
                Some(&self.k_norm_ctx),
            )?
            .rope(self.head_dim, self.theta, cache.offset)?;
        let v = self.heads(&self.v_ctx.forward(&context)?, self.kv_heads, None)?;
        let (mut keys, mut values) = match (&cache.keys, &cache.values) {
            (Some(keys), Some(values)) => (
                Array::concatenate(&[keys, &k], 2)?,
                Array::concatenate(&[values, &v], 2)?,
            ),
            _ => (k, v),
        };
        if let Some(window) = self.window {
            let total = keys.shape()[2];
            if total > window {
                keys = keys.slice(2, total - window, total)?;
                values = values.slice(2, total - window, total)?;
            }
        }
        cache.keys = Some(keys);
        cache.values = Some(values);
        cache.offset += rows;
        Ok(())
    }

    /// Bidirectional block attention over the context cache and the block itself (no mask).
    fn forward_block(&self, x: &Array, cache: &DraftCache) -> Result<Array> {
        let rows = x.shape()[0];
        let offset = cache.offset;
        let q = self
            .heads(&self.q.forward(x)?, self.heads, Some(&self.q_norm))?
            .rope(self.head_dim, self.theta, offset)?;
        let bk = self
            .heads(&self.k.forward(x)?, self.kv_heads, Some(&self.k_norm))?
            .rope(self.head_dim, self.theta, offset)?;
        let bv = self.heads(&self.v.forward(x)?, self.kv_heads, None)?;
        let (keys, values) = match (&cache.keys, &cache.values) {
            (Some(keys), Some(values)) => (
                Array::concatenate(&[keys, &bk], 2)?,
                Array::concatenate(&[values, &bv], 2)?,
            ),
            _ => (bk, bv),
        };
        let out = Array::sdpa(&q, &keys, &values, (self.head_dim as f32).powf(-0.5), false)?
            .reshape(&[self.heads, rows, self.head_dim])?
            .transpose(&[1, 0, 2])?
            .reshape(&[rows, self.heads * self.head_dim])?;
        self.o.forward(&out)
    }
}

struct DraftLayer {
    input_norm: Array,
    post_norm: Array,
    dflare_norm: Array,
    attention: DraftAttention,
    gate: DraftLinear,
    up: DraftLinear,
    down: DraftLinear,
}

impl DraftLayer {
    fn load(ckpt: &Checkpoint, layer: usize, c: &DraftConfig) -> Result<Self> {
        let p = format!("layers.{layer}");
        let h = c.hidden_size;
        let norm = |name: &str| f32_weight(ckpt, &format!("{p}.{name}.weight"), &[h]);
        Ok(Self {
            input_norm: norm("input_layernorm")?,
            post_norm: norm("post_attention_layernorm")?,
            dflare_norm: norm("dflare_norm")?,
            attention: DraftAttention::load(ckpt, layer, c)?,
            gate: DraftLinear::load(
                ckpt,
                &format!("{p}.mlp.gate_proj"),
                h,
                c.intermediate_size,
                c.layer_bits,
            )?,
            up: DraftLinear::load(
                ckpt,
                &format!("{p}.mlp.up_proj"),
                h,
                c.intermediate_size,
                c.layer_bits,
            )?,
            down: DraftLinear::load(
                ckpt,
                &format!("{p}.mlp.down_proj"),
                c.intermediate_size,
                h,
                c.layer_bits,
            )?,
        })
    }

    fn mlp(&self, x: &Array) -> Result<Array> {
        let gate = self.gate.forward(x)?.astype(Dtype::Float32)?;
        let up = self.up.forward(x)?.astype(Dtype::Float32)?;
        self.down
            .forward(&gate.silu()?.mul(&up)?.astype(Dtype::Float16)?)
    }
}

pub struct Inkling2Draft {
    config: DraftConfig,
    layers: Vec<DraftLayer>,
    norm: Array,
    markov_norm: Array,
    /// `softmax(fuse_weight)` rows `[layers + 1, 1, taps]` FP32.
    alpha: Vec<Array>,
    mask_embed: Array,
    markov_embed: Array,
    /// `rnn_proj` split: recurrent `[2 * rank, 3 * rank]` and context `[2 * D, 3 * rank]` (transposed), bias.
    rnn_recurrent_t: Array,
    rnn_context_t: Array,
    rnn_bias: Array,
    /// `lm_head @ hidden_out`: `[rank, vocab]` FP16 (transposed) and `[vocab]` FP16 bias.
    fold_t: Array,
    fold_bias: Array,
    /// Affine copies of the unpadded head `[vocab, D]` and of the fold `[vocab, rank]` when `head_bits < 16`.
    head_q: Option<DraftLinear>,
    fold_q: Option<DraftLinear>,
    caches: Vec<DraftCache>,
    softcap: f32,
    vocab: i32,
}

impl Inkling2Draft {
    pub fn load(model_dir: &Path, target: &Inkling2) -> Result<Self> {
        let root: Root = serde_json::from_reader(File::open(model_dir.join("config.json"))?)?;
        let mut c = root.mtp_config;
        c.layer_bits = bits_from_env("MLXL3_INKLING_DSPARK_BITS", 8)?;
        c.head_bits = bits_from_env("MLXL3_INKLING_DSPARK_HEAD_BITS", 16)?;
        let taps = c.target_layer_ids.len() as i32;
        ensure!(
            c.markov_head_type == "rnn_hidden"
                && !c.q_bias
                && !c.o_bias
                && c.hidden_size == target.hidden_size()
                && c.block_size >= 2
                && c.num_attention_heads % c.num_key_value_heads == 0
                && c.target_layer_ids.last() == Some(&target.visit_count())
                && c.target_layer_ids.windows(2).all(|w| w[0] < w[1]),
            "unsupported Inkling 2 DSpark configuration"
        );
        let path = model_dir.join("mtp.safetensors");
        let ckpt = Checkpoint {
            path: model_dir.to_path_buf(),
            model_type: "inkling2_dspark".into(),
            bits: None,
            size_bytes: std::fs::metadata(&path)?.len(),
            modules: Vec::new(),
            tensors: checkpoint::read_header(&path)?,
        };
        let h = c.hidden_size;
        let rank = c.markov_rank;
        let layers = (0..c.num_hidden_layers)
            .map(|i| DraftLayer::load(&ckpt, i, &c))
            .collect::<Result<Vec<_>>>()?;
        // softmax(fuse_weight) per row, computed once on the host in f64
        let fuse = f32_weight(
            &ckpt,
            "fuse_weight",
            &[c.num_hidden_layers as i32 + 1, taps],
        )?
        .to_f32()?;
        let alpha = fuse
            .chunks(taps as usize)
            .map(|row| {
                let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
                let e: Vec<f64> = row.iter().map(|&x| (x as f64 - m).exp()).collect();
                let z: f64 = e.iter().sum();
                let a: Vec<f32> = e.iter().map(|x| (x / z) as f32).collect();
                Array::from_f32(&a, &[1, taps])
            })
            .collect::<Result<Vec<_>>>()?;
        let rnn = f32_weight(
            &ckpt,
            "markov_head.rnn_proj.weight",
            &[3 * rank, 2 * rank + 2 * h],
        )?;
        let rnn_recurrent_t = rnn.slice(1, 0, 2 * rank)?.transpose(&[1, 0])?;
        let rnn_context_t = rnn
            .slice(1, 2 * rank, 2 * rank + 2 * h)?
            .transpose(&[1, 0])?;
        let vocab = target.unpadded_vocab();
        // SIZE-2026-11-21-INK2M-SLIM: a checkpoint whose head is stored in 8 bits ships the fold computed from
        // the FP16 head (`fold_from_head`), since it cannot be recomputed from the affine copy.
        let (fold, fold_bias) = if ckpt.tensors.contains_key(FOLD_WEIGHT) {
            let fold = checkpoint_array(&ckpt, FOLD_WEIGHT)?;
            let fold_bias = checkpoint_array(&ckpt, FOLD_BIAS)?;
            ensure!(
                fold.dtype() == Dtype::Float16
                    && fold.shape() == [vocab, rank]
                    && fold_bias.dtype() == Dtype::Float16
                    && fold_bias.shape() == [vocab],
                "invalid stored DSpark fold"
            );
            Array::eval_many(&[&fold, &fold_bias])?;
            (fold, fold_bias)
        } else {
            let head = target
                .head_weight()
                .context("the DSpark fold needs a dense target head")?;
            let hidden_out = f32_weight(&ckpt, "markov_head.hidden_out.weight", &[h, rank])?;
            let hidden_bias = f32_weight(&ckpt, "markov_head.hidden_out.bias", &[h])?;
            fold_from_head(head, &hidden_out, &hidden_bias, vocab)?
        };
        let fold_t = fold.transpose(&[1, 0])?;
        // draft-only affine copies; the target head stays the EXL3/FP16 one used for verification
        let (head_q, fold_q) = if c.head_bits < 16 {
            let head = target.head_weight().context(
                "MLXL3_INKLING_DSPARK_HEAD_BITS below 16 needs the dense target head (this checkpoint stores an affine head)",
            )?;
            (
                Some(DraftLinear::quantized(
                    &head.slice(0, 0, vocab)?,
                    c.head_bits,
                )?),
                Some(DraftLinear::quantized(&fold, c.head_bits)?),
            )
        } else {
            (None, None)
        };
        let draft = Self {
            layers,
            norm: f32_weight(&ckpt, "norm.weight", &[h])?,
            markov_norm: f32_weight(&ckpt, "markov_head.dflare_norm.weight", &[h])?,
            alpha,
            mask_embed: f32_weight(&ckpt, "mask_embed", &[h])?.reshape(&[1, h])?,
            markov_embed: half_weight(
                &ckpt,
                "markov_head.markov_w1.weight",
                Some(&[target.padded_vocab(), rank]),
            )?,
            rnn_recurrent_t,
            rnn_context_t,
            rnn_bias: f32_weight(&ckpt, "markov_head.rnn_proj.bias", &[3 * rank])?,
            fold_t,
            fold_bias,
            head_q,
            fold_q,
            caches: (0..c.num_hidden_layers)
                .map(|_| DraftCache::default())
                .collect(),
            softcap: target.logit_softcap(),
            vocab,
            config: c,
        };
        for a in [
            &draft.rnn_recurrent_t,
            &draft.rnn_context_t,
            &draft.fold_t,
            &draft.fold_bias,
        ] {
            a.eval()?;
        }
        // The fold went through a 1.8 GB FP32 copy of the head: give that memory back now, or MLX keeps it
        // cached and pushes the target's mapped weights out of RAM before its first forward.
        crate::array::clear_cache()?;
        Ok(draft)
    }

    pub fn config(&self) -> &DraftConfig {
        &self.config
    }

    pub fn taps(&self) -> &[usize] {
        &self.config.target_layer_ids
    }

    pub fn max_proposals(&self) -> usize {
        self.config.block_size - 1
    }

    pub fn reset(&mut self) {
        for c in &mut self.caches {
            *c = DraftCache::default();
        }
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D1: the context caches now. `append_context` concatenates into new
    /// arrays and never writes a saved one, so these references keep the values of the capture point; views
    /// that retain a larger buffer (the windowed layers' slices) are copied to their own rows (same values).
    pub fn context_snapshot(&self) -> Result<DraftContext> {
        let compact = |a: &Option<Array>| a.as_ref().map(Array::compact_for_cache).transpose();
        Ok(DraftContext {
            caches: self
                .caches
                .iter()
                .map(|c| {
                    Ok(DraftCache {
                        keys: compact(&c.keys)?,
                        values: compact(&c.values)?,
                        offset: c.offset,
                    })
                })
                .collect::<Result<_>>()?,
        })
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D1: back to the context caches of `saved` (from this drafter).
    pub fn restore_context(&mut self, saved: &DraftContext) -> Result<()> {
        ensure!(
            saved.caches.len() == self.caches.len(),
            "DSpark context of another drafter"
        );
        for (live, saved) in self.caches.iter_mut().zip(&saved.caches) {
            *live = DraftCache {
                keys: saved.keys.as_ref().map(Array::try_clone).transpose()?,
                values: saved.values.as_ref().map(Array::try_clone).transpose()?,
                offset: saved.offset,
            };
        }
        Ok(())
    }

    /// `rms_norm(sum_j alpha[row, j] * feature_j)`, features `[T, taps * D]` FP32 -> `[T, D]`.
    fn mix(&self, features: &Array, row: usize, norm: &Array) -> Result<Array> {
        let rows = features.shape()[0];
        let h = self.config.hidden_size;
        let taps = self.config.target_layer_ids.len() as i32;
        features
            .astype(Dtype::Float32)?
            .reshape(&[rows, taps, h])?
            .transpose(&[0, 2, 1])?
            .matmul(&self.alpha[row].transpose(&[1, 0])?)?
            .reshape(&[rows, h])?
            .rms_norm(norm, self.config.rms_norm_eps)
    }

    /// Feed committed target features `[Tc, taps * D]` (in token order) to every draft layer's context cache.
    pub fn append_context(&mut self, features: &Array) -> Result<()> {
        for (i, layer) in self.layers.iter().enumerate() {
            let context = self
                .mix(features, i, &layer.dflare_norm)?
                .astype(Dtype::Float16)?;
            layer
                .attention
                .append_context(&context, &mut self.caches[i])?;
        }
        Ok(())
    }

    /// `propose` plus the normalized draft hidden rows `[count, D]` (oracle checks).
    pub fn propose_debug(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
    ) -> Result<(Array, Array, Array)> {
        self.propose_inner(target, anchor, last_feature, count)
    }

    /// Greedy proposals for the tokens after `anchor`: `[count]` u32, on the GPU until read.
    /// `last_feature`: `[1, taps * D]` features of the last committed token.
    pub fn propose(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
    ) -> Result<Array> {
        Ok(self
            .propose_with(target, anchor, last_feature, count, DraftDraw::Best)?
            .0)
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D2: proposals coupled to the target's draws: proposal i is
    /// `inkling2::coupled_pick` of its logits with row i of the block's Gumbel `noise` `[1 + count, V_target]`
    /// (the noise `inkling2::decide_coupled` draws row i of the verification with). `[count]` u32, lazy.
    pub fn propose_coupled(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
        temperature: f32,
        noise: &Array,
    ) -> Result<Array> {
        ensure!(
            noise.shape().len() == 2 && noise.shape()[0] == count as i32 + 1,
            "coupled DSpark proposals need one noise row per verified row"
        );
        Ok(self
            .propose_with(
                target,
                anchor,
                last_feature,
                count,
                DraftDraw::Coupled(temperature, noise),
            )?
            .0)
    }

    /// DSPARK-INK2M-P3: proposals drawn from the draft law softmax(logits / temperature) by Gumbel-max on
    /// the GPU (noise from `seed`, `seed + 1`, ... per position). Returns the `[count]` tokens and the
    /// `[count, vocab]` draft logits they were drawn from (before the temperature).
    pub fn propose_sampled(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<(Array, Array)> {
        ensure!(
            temperature.is_finite() && temperature > 0.,
            "sampled DSpark proposals need a positive temperature"
        );
        let (tokens, _, _, rows) = self.propose_with(
            target,
            anchor,
            last_feature,
            count,
            DraftDraw::Seeded(temperature, seed),
        )?;
        Ok((
            tokens,
            rows.context("sampled proposals without draft rows")?,
        ))
    }

    fn propose_inner(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
    ) -> Result<(Array, Array, Array)> {
        let (tokens, x, base, _) =
            self.propose_with(target, anchor, last_feature, count, DraftDraw::Best)?;
        Ok((tokens, x, base))
    }

    /// Proposals under `draw`: the draft's best tokens, Gumbel-max draws from softmax(logits / T) with noise
    /// seeded by `seed + position` (then also the stacked draft logits), or coupled draws.
    fn propose_with(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
        draw: DraftDraw<'_>,
    ) -> Result<(Array, Array, Array, Option<Array>)> {
        ensure!(
            count >= 1 && count < self.config.block_size,
            "DSpark proposal count outside the trained block"
        );
        let block = self.config.block_size as i32;
        let h = self.config.hidden_size;
        let embed = target.embed_tokens(anchor)?;
        let masks = self.mask_embed.broadcast_to(&[block - 1, h])?;
        let mut x = Array::concatenate(&[&embed, &masks], 0)?;
        let mut residual: Option<Array> = None;
        let eps = self.config.rms_norm_eps;
        for (i, layer) in self.layers.iter().enumerate() {
            let summed = match &residual {
                Some(r) => x.astype(Dtype::Float32)?.add(r)?,
                None => x.astype(Dtype::Float32)?,
            };
            let attn = layer.attention.forward_block(
                &summed
                    .rms_norm(&layer.input_norm, eps)?
                    .astype(Dtype::Float16)?,
                &self.caches[i],
            )?;
            let summed2 = attn.astype(Dtype::Float32)?.add(&summed)?;
            x = layer.mlp(
                &summed2
                    .rms_norm(&layer.post_norm, eps)?
                    .astype(Dtype::Float16)?,
            )?;
            residual = Some(summed2);
        }
        let x = x
            .astype(Dtype::Float32)?
            .add(residual.as_ref().context("draft layers")?)?
            .rms_norm(&self.norm, eps)?
            .slice(0, 1, 1 + count as i32)?;
        let base = match &self.head_q {
            Some(head) => head.forward(&x)?,
            None => target.head_scores(&x)?,
        }
        .astype(Dtype::Float32)?;
        let f_rnn = self.mix(
            last_feature,
            self.config.num_hidden_layers,
            &self.markov_norm,
        )?;
        let context = Array::concatenate(&[&x, &f_rnn.broadcast_to(&[count as i32, h])?], 1)?
            .matmul(&self.rnn_context_t)?
            .add(&self.rnn_bias)?;
        let rank = self.config.markov_rank;
        let mut state = Array::zeros_dtype(&[1, rank], Dtype::Float32)?;
        let mut previous = anchor.try_clone()?;
        let mut tokens = Vec::with_capacity(count);
        let mut rows = Vec::with_capacity(if matches!(draw, DraftDraw::Seeded(..)) {
            count
        } else {
            0
        });
        for i in 0..count as i32 {
            let inputs = Array::concatenate(
                &[
                    &state,
                    &self
                        .markov_embed
                        .take(&previous, 0)?
                        .astype(Dtype::Float32)?,
                ],
                1,
            )?;
            let raw = inputs
                .matmul(&self.rnn_recurrent_t)?
                .add(&context.slice(0, i, i + 1)?)?;
            let gate = raw.slice(1, 0, rank)?.sigmoid()?;
            let candidate = raw.slice(1, rank, 2 * rank)?.tanh()?;
            let latent = raw.slice(1, 2 * rank, 3 * rank)?.tanh()?;
            state = gate
                .mul(&state)?
                .add(&gate.scalar_mul(-1.0)?.scalar_add(1.0)?.mul(&candidate)?)?;
            let latent = latent.astype(Dtype::Float16)?;
            let bias = match &self.fold_q {
                Some(fold) => fold.forward(&latent)?,
                None => latent.matmul(&self.fold_t)?,
            }
            .add(&self.fold_bias)?
            .astype(Dtype::Float32)?;
            let logits = base.slice(0, i, i + 1)?.add(&bias)?.softcap(self.softcap)?;
            previous = match draw {
                DraftDraw::Best => logits.argmax()?,
                DraftDraw::Seeded(temperature, seed) => {
                    let previous = gumbel_draw(&logits, temperature, seed.wrapping_add(i as u64))?;
                    rows.push(logits);
                    previous
                }
                DraftDraw::Coupled(temperature, noise) => {
                    crate::inkling2::coupled_pick(&logits, temperature, &noise.slice(0, i, i + 1)?)?
                }
            };
            tokens.push(previous.try_clone()?);
        }
        let _ = self.vocab;
        let tokens = Array::concatenate(&tokens.iter().collect::<Vec<_>>(), 0)?;
        let rows = if rows.is_empty() {
            None
        } else {
            Some(Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0)?)
        };
        Ok((tokens, x, base, rows))
    }
}

/// How `propose_with` turns each position's draft logits into a proposal.
#[derive(Clone, Copy)]
enum DraftDraw<'a> {
    /// the draft's argmax
    Best,
    /// DSPARK-INK2M-P3: `(temperature, seed)`, a draw from the draft law with its own noise
    Seeded(f32, u64),
    /// DSPARK-2026-11-24-INK2M-MAYA2-D2: `(temperature, noise [1 + count, V_target])`, the target's noise
    Coupled(f32, &'a Array),
}

/// SPEED-INK2M-P6a: runs every kernel shape a generation uses once (long prefill on the TensorOps
/// expert path, short prefill, plain decode step, DSpark verifications of 2 to 4 rows, the committed
/// width-0 step with features, the drafter's context and proposals), then resets both models and
/// releases the buffer cache. The first request after loading otherwise paid the Metal kernel
/// compilation (about 8.5 s measured on Inkling 2 Mini). No state survives: logits after it are
/// identical to logits of a model that never ran it.
pub fn warm_up(target: &mut Inkling2, draft: Option<&mut Inkling2Draft>) -> Result<()> {
    // ordinary text token ids, inside every vocabulary of the family
    let tokens: Vec<u32> = (0..72u32).map(|i| 1000 + 37 * i).collect();
    target.reset();
    target.forward_tokens(&tokens)?.eval()?;
    target.reset();
    target.forward_tokens(&tokens[..16])?.eval()?;
    target.forward(tokens[16])?.eval()?;
    if let Some(draft) = draft {
        let taps = draft.taps().to_vec();
        target.reset();
        draft.reset();
        let (_, features) = target.forward_with_features(&tokens, &taps)?;
        draft.append_context(&features)?;
        let rows = features.shape()[0];
        let mut last = features.slice(0, rows - 1, rows)?;
        let counts = draft.max_proposals().min(3);
        ensure!(tokens.len() > counts, "warm-up prompt too short");
        for (count, &token) in tokens.iter().enumerate().skip(1).take(counts) {
            let anchor = Array::from_u32(&[token], &[1])?;
            let proposals = draft.propose(target, &anchor, &last, count)?.to_u32()?;
            let mut block = vec![token];
            block.extend(&proposals);
            let (logits, features) = target.verify(&block, &taps)?;
            logits.eval()?;
            target.commit(1)?;
            draft.append_context(&features.slice(0, 0, 1)?)?;
            last = features.slice(0, 0, 1)?;
        }
        let (logits, features) = target.forward_with_features(&tokens[..1], &taps)?;
        logits.eval()?;
        draft.append_context(&features)?;
        draft.reset();
    }
    target.reset();
    crate::array::synchronize()?;
    crate::array::clear_cache()
}

/// DFLASH2-2026-11-24-INK2M-P3c: compiles the GPU path of the blocks with proposals of `draft` for every width
/// `1..=widths` (drafter kernels, the target's verification of `1 + w` rows, the greedy and T = 1 decisions)
/// while loading, so no Metal kernel is compiled on the first block of a width during decoding. Leaves the target
/// and the drafter reset.
pub fn warm_up_drafter<D: BlockDrafter + ?Sized>(
    target: &mut Inkling2,
    draft: &mut D,
    widths: usize,
) -> Result<()> {
    // ordinary text token ids, inside every vocabulary of the family (as `warm_up`)
    let tokens: Vec<u32> = (0..72u32).map(|i| 1000 + 37 * i).collect();
    let widths = widths.min(draft.max_proposals());
    ensure!(tokens.len() > widths + 1, "warm-up prompt too short");
    let taps = draft.taps().to_vec();
    target.reset();
    draft.reset();
    let (_, features) = target.forward_with_features(&tokens, &taps)?;
    draft.append_context(&features)?;
    let rows = features.shape()[0];
    let mut last = features.slice(0, rows - 1, rows)?;
    for (width, &token) in tokens.iter().enumerate().skip(1).take(widths) {
        let anchor = Array::from_u32(&[token], &[1])?;
        let proposals = draft.propose(target, &anchor, &last, width)?;
        let block = Array::concatenate(&[&anchor, &proposals], 0)?;
        let (logits, features) = target.verify_device(&block, &taps)?;
        for rule in [
            crate::inkling2::DrawRule::Greedy,
            crate::inkling2::DrawRule::Softmax(1.0),
        ] {
            crate::inkling2::decide_block(&logits, &proposals, rule, 1, 2)?.to_u32()?;
        }
        target.commit(1)?;
        let first = features.slice(0, 0, 1)?;
        draft.append_context(&first)?;
        last = first;
    }
    target.reset();
    draft.reset();
    crate::array::synchronize()?;
    crate::array::clear_cache()
}

/// One draw per row of `logits` (`[rows, vocab]`) from softmax(logits / temperature):
/// argmax(logits / T + G) with G = -log(-log U), U ~ uniform[0, 1) seeded by `seed` (U = 0 gives -inf
/// and is never chosen). Returns `[rows]` token ids on the GPU.
pub fn gumbel_draw(logits: &Array, temperature: f32, seed: u64) -> Result<Array> {
    let shape = logits.shape();
    ensure!(shape.len() == 2, "Gumbel draw expects [rows, vocab] logits");
    let gumbel = Array::random_uniform(shape, seed)?
        .log()?
        .scalar_mul(-1.0)?
        .log()?
        .scalar_mul(-1.0)?;
    logits
        .astype(Dtype::Float32)?
        .scalar_mul(1.0 / temperature)?
        .add(&gumbel)?
        .argmax()
}

/// Names of the precomputed fold in `mtp.safetensors` (SIZE-2026-11-21-INK2M-SLIM).
pub(crate) const FOLD_WEIGHT: &str = "markov_head.fold.weight";
pub(crate) const FOLD_BIAS: &str = "markov_head.fold.bias";

/// `lm_head @ hidden_out` (`[vocab, rank]`) and `lm_head @ hidden_bias` (`[vocab]`) over the unpadded rows of the
/// FP16 head, in FP32 by blocks of rows, rounded to FP16: a whole FP32 copy of the head would take 1.8 GB and push
/// the target's mapped weights out of RAM before its first forward.
pub(crate) fn fold_from_head(
    head: &Array,
    hidden_out: &Array,
    hidden_bias: &Array,
    vocab: i32,
) -> Result<(Array, Array)> {
    const FOLD_ROWS: i32 = 16384;
    let h = hidden_bias.shape()[0];
    let (mut fold_parts, mut bias_parts) = (Vec::new(), Vec::new());
    let mut start = 0;
    while start < vocab {
        let end = (start + FOLD_ROWS).min(vocab);
        let rows = head.slice(0, start, end)?.astype(Dtype::Float32)?;
        let part = rows.matmul(hidden_out)?.astype(Dtype::Float16)?;
        let bias = rows
            .matmul(&hidden_bias.reshape(&[h, 1])?)?
            .reshape(&[end - start])?
            .astype(Dtype::Float16)?;
        Array::eval_many(&[&part, &bias])?;
        fold_parts.push(part);
        bias_parts.push(bias);
        start = end;
    }
    let fold = Array::concatenate(&fold_parts.iter().collect::<Vec<_>>(), 0)?;
    let fold_bias = Array::concatenate(&bias_parts.iter().collect::<Vec<_>>(), 0)?;
    Ok((fold, fold_bias))
}

/// DFLASH2-2026-11-24-INK2M-P0: what a block session needs from its drafter (DSpark or DFlash2): its taps, its
/// context of committed target features (and its D1 snapshot), and the proposals after an anchor.
pub trait BlockDrafter {
    fn name(&self) -> &'static str;
    fn taps(&self) -> &[usize];
    fn max_proposals(&self) -> usize;
    fn reset(&mut self);
    /// Feeds committed target features `[Tc, taps * D]` (token order).
    fn append_context(&mut self, features: &Array) -> Result<()>;
    fn context_snapshot(&self) -> Result<DraftContext>;
    fn restore_context(&mut self, saved: &DraftContext) -> Result<()>;
    /// Deterministic proposals after `anchor` (`[1]` u32): `[count]` u32. `last_feature` `[1, taps * D]`: the
    /// features of the last committed token (drafters that keep their own context may ignore it).
    fn propose(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
    ) -> Result<Array>;
    /// DSPARK-INK2M-P3 proposals drawn from the draft law (with their draft logits), when supported.
    fn propose_sampled(
        &self,
        _target: &Inkling2,
        _anchor: &Array,
        _last_feature: &Array,
        _count: usize,
        _temperature: f32,
        _seed: u64,
    ) -> Result<(Array, Array)> {
        anyhow::bail!("{} drafter has no sampled proposals", self.name())
    }
    /// D2 proposals coupled to the target's noise, when supported (`supports_coupling`).
    fn propose_coupled(
        &self,
        _target: &Inkling2,
        _anchor: &Array,
        _last_feature: &Array,
        _count: usize,
        _temperature: f32,
        _noise: &Array,
    ) -> Result<Array> {
        anyhow::bail!("{} drafter has no coupled proposals", self.name())
    }
    fn supports_coupling(&self) -> bool {
        false
    }
    /// DFLASH2-2026-11-24-INK2M-S3: proposals drawn from the drafter's own law at `temperature`, with that law,
    /// for the ratio rule (`inkling2::decide_ratio`), when supported (`supports_ratio`).
    fn propose_ratio(
        &self,
        _target: &Inkling2,
        _anchor: &Array,
        _last_feature: &Array,
        _count: usize,
        _temperature: f32,
        _seed: u64,
    ) -> Result<RatioProposals> {
        anyhow::bail!("{} drafter has no ratio-rule proposals", self.name())
    }
    fn supports_ratio(&self) -> bool {
        false
    }
}

/// DFLASH2-2026-11-24-INK2M-S3: `count` proposals drawn from the drafter's law and that law, on the GPU:
/// `tokens` `[k]` u32, each row's candidates `[k, C]` u32 and their probabilities `q` `[k, C]` FP32 (row i
/// conditioned on the proposals before it), and `chosen` `[k]` FP32 = q of each proposal.
pub struct RatioProposals {
    pub tokens: Array,
    pub candidates: Array,
    pub q: Array,
    pub chosen: Array,
}

impl BlockDrafter for Inkling2Draft {
    fn name(&self) -> &'static str {
        "dspark"
    }
    fn taps(&self) -> &[usize] {
        Inkling2Draft::taps(self)
    }
    fn max_proposals(&self) -> usize {
        Inkling2Draft::max_proposals(self)
    }
    fn reset(&mut self) {
        Inkling2Draft::reset(self)
    }
    fn append_context(&mut self, features: &Array) -> Result<()> {
        Inkling2Draft::append_context(self, features)
    }
    fn context_snapshot(&self) -> Result<DraftContext> {
        Inkling2Draft::context_snapshot(self)
    }
    fn restore_context(&mut self, saved: &DraftContext) -> Result<()> {
        Inkling2Draft::restore_context(self, saved)
    }
    fn propose(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
    ) -> Result<Array> {
        Inkling2Draft::propose(self, target, anchor, last_feature, count)
    }
    fn propose_sampled(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<(Array, Array)> {
        Inkling2Draft::propose_sampled(self, target, anchor, last_feature, count, temperature, seed)
    }
    fn propose_coupled(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
        temperature: f32,
        noise: &Array,
    ) -> Result<Array> {
        Inkling2Draft::propose_coupled(
            self,
            target,
            anchor,
            last_feature,
            count,
            temperature,
            noise,
        )
    }
    fn supports_coupling(&self) -> bool {
        true
    }
}

/// DFLASH2-2026-11-24-INK2M-P0: the drafter the bridge loaded for Inkling 2 Mini (kept in a `Box` by the bridge;
/// both variants are about a kilobyte of handles).
#[allow(clippy::large_enum_variant)]
pub enum InklingDrafter {
    Dspark(Inkling2Draft),
    Dflash2(crate::inkling2_dflash2::Inkling2Dflash2),
}

macro_rules! drafter_dispatch {
    ($self:ident, $d:ident => $e:expr) => {
        match $self {
            InklingDrafter::Dspark($d) => $e,
            InklingDrafter::Dflash2($d) => $e,
        }
    };
}

impl BlockDrafter for InklingDrafter {
    fn name(&self) -> &'static str {
        drafter_dispatch!(self, d => BlockDrafter::name(d))
    }
    fn taps(&self) -> &[usize] {
        drafter_dispatch!(self, d => BlockDrafter::taps(d))
    }
    fn max_proposals(&self) -> usize {
        drafter_dispatch!(self, d => BlockDrafter::max_proposals(d))
    }
    fn reset(&mut self) {
        drafter_dispatch!(self, d => BlockDrafter::reset(d))
    }
    fn append_context(&mut self, features: &Array) -> Result<()> {
        drafter_dispatch!(self, d => BlockDrafter::append_context(d, features))
    }
    fn context_snapshot(&self) -> Result<DraftContext> {
        drafter_dispatch!(self, d => BlockDrafter::context_snapshot(d))
    }
    fn restore_context(&mut self, saved: &DraftContext) -> Result<()> {
        drafter_dispatch!(self, d => BlockDrafter::restore_context(d, saved))
    }
    fn propose(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
    ) -> Result<Array> {
        drafter_dispatch!(self, d => BlockDrafter::propose(d, target, anchor, last_feature, count))
    }
    fn propose_sampled(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<(Array, Array)> {
        drafter_dispatch!(self, d => BlockDrafter::propose_sampled(
            d, target, anchor, last_feature, count, temperature, seed
        ))
    }
    fn propose_coupled(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
        temperature: f32,
        noise: &Array,
    ) -> Result<Array> {
        drafter_dispatch!(self, d => BlockDrafter::propose_coupled(
            d, target, anchor, last_feature, count, temperature, noise
        ))
    }
    fn supports_coupling(&self) -> bool {
        drafter_dispatch!(self, d => BlockDrafter::supports_coupling(d))
    }
    fn propose_ratio(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
        temperature: f32,
        seed: u64,
    ) -> Result<RatioProposals> {
        drafter_dispatch!(self, d => BlockDrafter::propose_ratio(
            d, target, anchor, last_feature, count, temperature, seed
        ))
    }
    fn supports_ratio(&self) -> bool {
        drafter_dispatch!(self, d => BlockDrafter::supports_ratio(d))
    }
}

/// DSpark in the Desktop bridge: one draft block per anchor, the verified tokens delivered one per call
/// (the bridge emits each before asking for the next). Acceptance is exact: greedy rows use the bridge's
/// own argmax rule, sampled rows the point-mass rejection rule of `sampling` (each delivered token has the
/// law of an ordinary draw from the same target logits).
pub struct DsparkSession {
    width: usize,
    taps: Vec<usize>,
    /// Verified tokens not delivered yet; the last one is the next anchor.
    pending: std::collections::VecDeque<u32>,
    /// The token `advance` returned last (the caller delivers it, then passes it back as `next`).
    expected: Option<u32>,
    last_feature: Option<Array>,
    /// DSPARK-INK2M-P3: `(temperature, xorshift state)` when proposals are drawn from the draft law.
    sampled: Option<(f32, u64)>,
    /// DSPARK-INK2M-P4: adaptive width (`None` = always `width`).
    policy: Option<WidthPolicy>,
    /// SPEED-INK2M-P14: GPU token choice of width-0 blocks (`None` = the caller's `choose`).
    device: Option<DeviceDraw>,
    /// SPEED-INK2M-P14: the width-0 block anchored on the token `advance` returned last, already submitted:
    /// the token after that anchor, on the GPU.
    ahead: Option<Array>,
    /// SPEED-INK2M-P14: the width drawn from the policy for the next block before its anchor was read.
    decided: Option<usize>,
    /// DSPARK-2026-11-23-INK2M-MTPLX-A: blocks with proposals decided on the GPU (with `device`, deterministic
    /// proposals only); `MLXL3_INKLING_DSPARK_DEVICE=0` keeps the caller's `choose`.
    device_rounds: bool,
    /// DSPARK-2026-11-23-INK2M-MTPLX-A3: a GPU-decided block submits the drafter's proposals before building
    /// the verification (`MLXL3_INKLING_DSPARK_OVERLAP=0` submits both together).
    overlap: bool,
    /// DSPARK-2026-11-24-INK2M-MAYA2-D2: sampled blocks with proposals draw the drafter's proposals with the
    /// noise of the target's own draws at their positions (`MLXL3_INKLING_DSPARK_COUPLE=1`; otherwise the
    /// point rule).
    couple: bool,
    /// DFLASH2-2026-11-24-INK2M-S3: sampled blocks with proposals use the ratio rule of Splash (proposals drawn
    /// from the drafter's law, kept with min(1, p / q), residual draw) when the drafter supports it
    /// (on by default since the user's decision of 2026-11-25; `MLXL3_INKLING_DRAFT_RATIO=0` keeps the point rule
    /// or the coupling); it takes precedence over the coupling.
    ratio: bool,
    /// DSPARK-2026-11-24-INK2M-MAYA2-D5: prompt lookup in the same controller (`with_lookup`).
    lookup: Option<LookupSource>,
    pub stats: SpecStats,
}

/// DSPARK-2026-11-24-INK2M-MAYA2-D5: the prompt lookup of a DSpark session: the text so far (prompt and
/// delivered tokens) and the lookup's own width policy (acceptance, probes, block costs of LK).
struct LookupSource {
    index: crate::speculative::NgramIndex,
    policy: crate::inkling2_lookup::LookupPolicy,
    /// Shadow observation: a proposal the lookup had where no lookup block ran, and how many of the tokens
    /// delivered since matched it.
    shadow: Option<(Vec<u32>, usize)>,
}

impl LookupSource {
    /// Reads the delivered `token`: the text grows, and a pending shadow proposal is compared with it; at its
    /// first mismatch, or once matched to its end, the lookup's acceptance observes it as a block of its length.
    fn read(&mut self, token: u32) {
        self.index.push(token);
        if let Some((proposal, matched)) = self.shadow.as_mut() {
            if proposal.get(*matched) == Some(&token) {
                *matched += 1;
                if *matched == proposal.len() {
                    self.policy.observe(proposal.len(), proposal.len());
                    self.shadow = None;
                }
            } else {
                self.policy.observe(proposal.len(), *matched);
                self.shadow = None;
            }
        }
    }

    /// Starts a shadow observation of `proposal` unless one is pending.
    fn shadow(&mut self, proposal: &[u32]) {
        if self.shadow.is_none() && !proposal.is_empty() {
            self.shadow = Some((proposal.to_vec(), 0));
        }
    }
}

const LOOKUP_MAX_WIDTH: usize = crate::inkling2_lookup::MAX_WIDTH;
const LOOKUP_MIN_MATCH: usize = crate::inkling2_lookup::MIN_MATCH;

#[cfg(test)]
thread_local! {
    /// D5 oracle mutation: lookup proposals delivered without the target's decision.
    static D5_MUTATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// D5 oracle: every lookup proposal runs as a block of its whole length (policies bypassed).
    static D5_FORCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// D5 oracle record of each GPU-decided lookup block, filled while set.
    static D5_RECORD: std::cell::RefCell<Option<Vec<LookupRecord>>> =
        const { std::cell::RefCell::new(None) };
}

/// D5 oracle: (proposals, argmax of every verified row, accepted, token) of one lookup block.
#[cfg(test)]
type LookupRecord = (Vec<u32>, Vec<u32>, usize, u32);

/// DSPARK-2026-11-24-INK2M-MAYA2-D1: what a DSpark prefill holds at a prompt chunk boundary, for the next
/// turn of the same conversation: the target's committed prefix, the drafter's context and the tap
/// features of the boundary's last row (the next block's `last_feature` when nothing is left to prefill).
pub struct DsparkPrefix {
    target: crate::inkling2::InklingPrefix,
    context: DraftContext,
    last_feature: Array,
}

impl DsparkPrefix {
    pub fn offset(&self) -> i32 {
        self.target.offset()
    }

    /// Bytes this prefix keeps alive beyond the live state.
    pub fn retained_bytes(&self) -> Result<usize> {
        self.target
            .retained_bytes()?
            .checked_add(self.context.retained_bytes()?)
            .and_then(|b| b.checked_add(self.last_feature.retained_bytes().ok()?))
            .context("DSpark prefix size overflow")
    }
}

#[cfg(test)]
thread_local! {
    /// D1 oracle mutations: 1 = the drafter keeps its live context on resume, 2 = the resumed session keeps
    /// its own last feature.
    static D1_MUTATION: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// DSPARK-2026-11-24-INK2M-MAYA2-D2: the coupled rule only when `value` is "1" (an option: its preregistered
/// criterion was not met on code; first-proposal acceptance +6 points in D2-DIAG).
pub fn coupling_setting(value: Option<&str>) -> bool {
    value == Some("1")
}

/// DFLASH2-2026-11-24-INK2M-S3: the ratio rule of `MLXL3_INKLING_DRAFT_RATIO`, on unless "0" (default since the
/// user's decision of 2026-11-25).
pub fn ratio_setting(value: Option<&str>) -> bool {
    value != Some("0")
}

/// DSPARK-2026-11-23-INK2M-MTPLX-A: GPU decision of blocks with proposals unless `value` is "0".
pub fn device_rounds_setting(value: Option<&str>) -> bool {
    value != Some("0")
}

/// SPEED-INK2M-P14: how width-0 blocks choose their token on the GPU.
struct DeviceDraw {
    rule: crate::inkling2::DrawRule,
    seeds: crate::inkling2::DrawSeeds,
    pipelined: bool,
}

/// Target-side choice of one block: `(proposals, draft log-probabilities, verified logits [1, 1 + n, vocab])`
/// -> accepted count and the target token after them. The draft rows (`[n]` rows of `vocab` log
/// probabilities) are given only when the proposals were drawn from the draft law (DSPARK-INK2M-P3);
/// otherwise the proposals are deterministic.
pub type BlockChoice<'a> =
    dyn FnMut(&[u32], Option<&[Vec<f32>]>, &Array) -> Result<(usize, u32)> + 'a;

impl DsparkSession {
    pub const DEFAULT_WIDTH: usize = 3;
    pub const PREFILL_CHUNK: usize = 128;

    pub fn new<D: BlockDrafter + ?Sized>(draft: &D, width: usize) -> Result<Self> {
        ensure!(
            width >= 1 && width <= draft.max_proposals(),
            "DSpark width outside the trained block"
        );
        Ok(Self {
            width,
            taps: draft.taps().to_vec(),
            pending: Default::default(),
            expected: None,
            last_feature: None,
            sampled: None,
            policy: None,
            device: None,
            ahead: None,
            decided: None,
            device_rounds: false,
            overlap: false,
            couple: false,
            ratio: false,
            lookup: None,
            stats: SpecStats {
                accepted_at: vec![0; width],
                drafter: draft.name(),
                ..SpecStats::default()
            },
        })
    }

    /// Draw proposals from softmax(draft logits / `temperature`) instead of taking the draft's argmax. Only
    /// for plain softmax sampling (no top-k, top-p or penalties); `seed` starts the noise sequence.
    pub fn with_sampled_drafts(mut self, temperature: f32, seed: u64) -> Result<Self> {
        ensure!(
            temperature.is_finite() && temperature > 0.,
            "sampled DSpark proposals need a positive temperature"
        );
        // xorshift needs a non-zero state
        self.sampled = Some((temperature, seed | 1));
        Ok(self)
    }

    pub fn samples_drafts(&self) -> bool {
        self.sampled.is_some()
    }

    /// SPEED-INK2M-P14: width-0 blocks choose their token on the GPU under `rule` (the same law as the
    /// caller's `choose` for that sampler); with `pipelined`, a width-0 block whose anchor is such a token
    /// is submitted before the anchor is read.
    pub fn with_device_draw(
        mut self,
        rule: crate::inkling2::DrawRule,
        seeds: crate::inkling2::DrawSeeds,
        pipelined: bool,
    ) -> Self {
        self.device = Some(DeviceDraw {
            rule,
            seeds,
            pipelined,
        });
        self.device_rounds =
            device_rounds_setting(std::env::var("MLXL3_INKLING_DSPARK_DEVICE").ok().as_deref());
        self.overlap = device_rounds_setting(
            std::env::var("MLXL3_INKLING_DSPARK_OVERLAP")
                .ok()
                .as_deref(),
        );
        self.couple =
            coupling_setting(std::env::var("MLXL3_INKLING_DSPARK_COUPLE").ok().as_deref());
        self.ratio = ratio_setting(std::env::var("MLXL3_INKLING_DRAFT_RATIO").ok().as_deref());
        self
    }

    /// DSPARK-2026-11-23-INK2M-MTPLX-A3: whether a GPU-decided block submits its proposals before building
    /// the verification.
    pub fn with_overlap(mut self, enabled: bool) -> Self {
        self.overlap = enabled;
        self
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D5: proposals copied from the text (`prompt`, then every delivered token)
    /// compete with the drafter's at each block (Project Maya's `SpecCtl::pick`): the lookup's block runs when
    /// its expected tokens per cost beat the drafter policy's choice. Without it the session is unchanged.
    pub fn with_lookup(mut self, prompt: &[u32]) -> Self {
        self.lookup = Some(LookupSource {
            index: crate::speculative::NgramIndex::new(prompt),
            policy: crate::inkling2_lookup::LookupPolicy::new(),
            shadow: None,
        });
        self
    }

    /// Whether the prompt lookup takes part in this session.
    pub fn looks_up(&self) -> bool {
        self.lookup.is_some()
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D2: whether sampled GPU-decided blocks couple the drafter's proposals
    /// to the target's draws (shared Gumbel noise) instead of the point rule.
    pub fn with_coupling(mut self, enabled: bool) -> Self {
        self.couple = enabled;
        self
    }

    /// DFLASH2-2026-11-24-INK2M-S3: whether sampled GPU-decided blocks use the ratio rule (drafters that support
    /// it).
    pub fn with_ratio(mut self, enabled: bool) -> Self {
        self.ratio = enabled;
        self
    }

    /// Whether the next sampled block with proposals uses the ratio rule (when its drafter supports it).
    pub fn ratio_rule(&self) -> bool {
        self.device_rounds()
            && self.ratio
            && matches!(
                self.device.as_ref().map(|d| d.rule),
                Some(crate::inkling2::DrawRule::Softmax(_))
            )
    }

    /// Whether the next sampled block with proposals uses the coupled rule.
    pub fn couples(&self) -> bool {
        self.device_rounds()
            && self.couple
            && matches!(
                self.device.as_ref().map(|d| d.rule),
                Some(crate::inkling2::DrawRule::Softmax(_))
            )
    }

    /// DSPARK-2026-11-23-INK2M-MTPLX-A: whether blocks with proposals are decided on the GPU (only with a GPU
    /// draw rule and deterministic proposals; otherwise the caller's `choose` decides).
    pub fn with_device_rounds(mut self, enabled: bool) -> Self {
        self.device_rounds = enabled;
        self
    }

    pub fn device_rounds(&self) -> bool {
        self.device_rounds && self.device.is_some() && self.sampled.is_none()
    }

    /// Choose each block's width (0 to the session width) from the acceptance observed so far.
    pub fn with_adaptive_width(mut self, policy: WidthPolicy) -> Result<Self> {
        ensure!(
            policy.max_width() == self.width,
            "DSpark width policy does not match the session width"
        );
        self.policy = Some(policy);
        self.stats.width_blocks = vec![0; self.width + 1];
        self.stats.width_ms = vec![0.0; self.width + 1];
        Ok(self)
    }

    /// Resets both models and prefills `tokens` by chunks, giving the drafter its context. Returns the
    /// logits of the last prompt row `[1, 1, vocab]`.
    pub fn prefill<D: BlockDrafter + ?Sized>(
        &mut self,
        target: &mut Inkling2,
        draft: &mut D,
        tokens: &[u32],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Array> {
        self.prefill_with_images(target, draft, tokens, &[], cancelled)
    }

    /// `prefill` of a prompt whose image placeholders read the features of `images`
    /// (`Inkling2::prepare_images`).
    pub fn prefill_with_images<D: BlockDrafter + ?Sized>(
        &mut self,
        target: &mut Inkling2,
        draft: &mut D,
        tokens: &[u32],
        images: &[crate::lfm3::PreparedImage],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Array> {
        Ok(self
            .prefill_resumable(
                target,
                draft,
                tokens,
                images,
                None,
                None,
                &mut |_, _| Ok(()),
                cancelled,
                &mut |_| Ok(()),
            )?
            .0)
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D1: `prefill_with_images` that first goes back to `resume` (a prefix of
    /// `tokens` at a chunk boundary, taken by this session type from this same state lineage, with the logits
    /// of its last row), then prefills the rest by the same chunks a cold prefill uses, so every value is the
    /// cold prefill's. When the target refuses the prefix, both models are reset and the whole prompt is
    /// prefilled as before. `capture` receives the state and logits at `checkpoint` (a chunk boundary within
    /// `tokens`), including when the resumed prefix already ends there. Returns the last row's logits and the
    /// number of resumed tokens.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_resumable<D: BlockDrafter + ?Sized>(
        &mut self,
        target: &mut Inkling2,
        draft: &mut D,
        tokens: &[u32],
        images: &[crate::lfm3::PreparedImage],
        resume: Option<(DsparkPrefix, Array)>,
        checkpoint: Option<usize>,
        capture: &mut dyn FnMut(DsparkPrefix, &Array) -> Result<()>,
        cancelled: &dyn Fn() -> bool,
        progress: &mut dyn FnMut(usize) -> Result<()>,
    ) -> Result<(Array, usize)> {
        ensure!(!tokens.is_empty(), "DSpark prefill needs tokens");
        ensure!(
            checkpoint.is_none_or(|c| c > 0 && c <= tokens.len() && c % Self::PREFILL_CHUNK == 0),
            "DSpark checkpoint must be a chunk boundary of the prompt"
        );
        let t = Instant::now();
        self.pending.clear();
        self.expected = None;
        self.ahead = None;
        self.decided = None;
        let mut start = 0;
        let mut logits = None;
        if let Some((prefix, saved_logits)) = resume {
            let offset = usize::try_from(prefix.target.offset())?;
            ensure!(
                images.is_empty()
                    && offset > 0
                    && offset <= tokens.len()
                    && offset % Self::PREFILL_CHUNK == 0
                    && prefix.context.offset() == Some(prefix.target.offset()),
                "invalid DSpark prefix"
            );
            let DsparkPrefix {
                target: saved,
                context,
                last_feature,
            } = prefix;
            if target.resume_prefix(saved)? {
                #[cfg(test)]
                let restore = !matches!(D1_MUTATION.with(std::cell::Cell::get), 1);
                #[cfg(not(test))]
                let restore = true;
                if restore {
                    draft.restore_context(&context)?;
                }
                #[cfg(test)]
                let last_feature = if D1_MUTATION.with(std::cell::Cell::get) == 2 {
                    self.last_feature.take().unwrap_or(last_feature)
                } else {
                    last_feature
                };
                self.last_feature = Some(last_feature);
                start = offset;
                logits = Some(saved_logits);
            }
        }
        if start == 0 {
            target.reset();
            draft.reset();
        }
        target.prepare_images(tokens, images)?;
        if start > 0 && checkpoint == Some(start) {
            capture(
                self.prefix(target, draft)?,
                logits.as_ref().context("resumed logits")?,
            )?;
        }
        // SPEED-2026-11-25-INK2M-PREFILL-G: a span is 1 to `group` whole chunks (their experts computed
        // together) or the final partial chunk alone; group 1 is the chunk-by-chunk prefill
        let group = crate::inkling2::moe_group()?;
        let mut position = start;
        while let Some(end) =
            crate::prefill_span_end(position, tokens.len(), Self::PREFILL_CHUNK, group)
        {
            ensure!(!cancelled(), "generation cancelled");
            // tokens already read, before each span: the bridge shows them while a long context is read
            progress(position)?;
            let chunk = &tokens[position..end];
            let started = Instant::now();
            let (chunk_logits, features) = if chunk.len() > Self::PREFILL_CHUNK {
                target.forward_with_features_grouped(chunk, &self.taps)?
            } else {
                target.forward_with_features(chunk, &self.taps)?
            };
            let target_done = started.elapsed();
            let rows = chunk.len() as i32;
            if chunk.len() > Self::PREFILL_CHUNK {
                // the drafter receives the features chunk by chunk, each as an array of its own, exactly
                // what the chunk-by-chunk prefill hands it
                let step = Self::PREFILL_CHUNK as i32;
                let mut row = 0;
                while row < rows {
                    let next = (row + step).min(rows);
                    draft.append_context(&features.slice(0, row, next)?.compact_for_cache()?)?;
                    row = next;
                }
            } else {
                draft.append_context(&features)?;
            }
            if std::env::var_os("MLXL3_INKLING_DSPARK_TRACE").is_some() {
                eprintln!(
                    "dspark prefill chunk {} rows: target {:.1} ms, draft context {:.1} ms",
                    chunk.len(),
                    target_done.as_secs_f64() * 1e3,
                    (started.elapsed() - target_done).as_secs_f64() * 1e3
                );
            }
            let last_feature = features.slice(0, rows - 1, rows)?;
            self.last_feature = Some(if chunk.len() > Self::PREFILL_CHUNK {
                // do not keep the features of the whole span alive for one row
                last_feature.compact_for_cache()?
            } else {
                last_feature
            });
            logits = Some(chunk_logits);
            position = end;
            if checkpoint == Some(position) {
                capture(
                    self.prefix(target, draft)?,
                    logits.as_ref().context("chunk logits")?,
                )?;
            }
        }
        self.stats.prefill_ms += t.elapsed().as_secs_f64() * 1e3;
        Ok((logits.context("empty DSpark prefill")?, start))
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D1: the committed state of a prefill at a chunk boundary.
    fn prefix<D: BlockDrafter + ?Sized>(
        &self,
        target: &Inkling2,
        draft: &D,
    ) -> Result<DsparkPrefix> {
        Ok(DsparkPrefix {
            target: target.prefix_snapshot()?,
            context: draft.context_snapshot()?,
            last_feature: self
                .last_feature
                .as_ref()
                .context("DSpark prefix without context")?
                .compact_for_cache()?,
        })
    }

    /// `next`: the token the caller just delivered. Returns the token after it, running a draft block
    /// when nothing verified is left. `remaining` >= 1 tokens may still be delivered; `context_limit`
    /// bounds the target positions.
    pub fn advance<D: BlockDrafter + ?Sized>(
        &mut self,
        target: &mut Inkling2,
        draft: &mut D,
        next: u32,
        context_limit: usize,
        remaining: usize,
        choose: &mut BlockChoice<'_>,
    ) -> Result<u32> {
        ensure!(remaining >= 1, "DSpark advance without budget");
        if let Some(expected) = self.expected.take() {
            ensure!(
                expected == next,
                "DSpark session out of step (delivered {next}, expected {expected})"
            );
        }
        // D5: the lookup reads every delivered token (and its pending shadow proposal is checked against it)
        if let Some(lookup) = self.lookup.as_mut() {
            lookup.read(next);
        }
        if let Some(token) = self.pending.pop_front() {
            self.expected = Some(token);
            return Ok(token);
        }
        // SPEED-INK2M-P14: the block anchored on `next` already ran (submitted before `next` was read)
        if let Some(after) = self.ahead.take() {
            // D5: a lookup proposal at `next` can only be counted as missed
            let likely = match self.lookup.as_mut() {
                Some(lookup) => {
                    let found = lookup
                        .index
                        .lookup_min(LOOKUP_MAX_WIDTH, LOOKUP_MIN_MATCH)
                        .map(<[u32]>::to_vec);
                    if let Some(found) = &found {
                        lookup.policy.missed();
                        lookup.shadow(found);
                    }
                    found.is_some()
                }
                None => false,
            };
            return self.deliver(target, draft, after, context_limit, remaining, likely);
        }
        // `next` is the anchor: the target has not seen it yet
        let offset = usize::try_from(target.offset())?;
        ensure!(offset < context_limit, "Inkling context is full");
        let width = match self.decided.take() {
            Some(width) => width,
            None => self.next_width(),
        };
        let room = (remaining - 1).min(context_limit - offset - 1);
        let count = width.min(room);
        // DSPARK-2026-11-24-INK2M-MAYA2-D5: the lookup's block when it is worth more than the drafter's choice
        let found: Vec<u32> = match self.lookup.as_mut() {
            Some(lookup) if room > 0 => lookup
                .index
                .lookup_min(LOOKUP_MAX_WIDTH.min(room), LOOKUP_MIN_MATCH)
                .map(<[u32]>::to_vec)
                .unwrap_or_default(),
            _ => Vec::new(),
        };
        #[cfg(test)]
        if D5_FORCE.with(std::cell::Cell::get) && !found.is_empty() {
            return self.lookup_block(target, draft, next, &found, choose);
        }
        if !found.is_empty() {
            let drafter_value = match &self.policy {
                Some(policy) if count > 0 => policy.expected_tokens(count) / policy.cost(count),
                _ => 1.0,
            };
            let lookup = self.lookup.as_mut().context("lookup")?;
            let lookup_width = lookup.policy.choose(found.len()).min(found.len());
            if lookup_width > 0 {
                let value = lookup.policy.expected_tokens(lookup_width)
                    / crate::inkling2_lookup::BLOCK_COSTS[lookup_width];
                // a probe of the lookup (value <= 1) runs only where the drafter would take a plain step
                if value > drafter_value || (value <= 1.0 && count == 0) {
                    return self.lookup_block(target, draft, next, &found[..lookup_width], choose);
                }
            }
            // not run: its proposal is checked against what the session delivers instead
            lookup.shadow(&found);
        }
        let likely = !found.is_empty();
        if count == 0 && self.device.is_some() {
            let anchor = Array::from_u32(&[next], &[1])?;
            let after = self.device_block(target, draft, &anchor)?;
            return self.deliver(target, draft, after, context_limit, remaining, likely);
        }
        let last_feature = self
            .last_feature
            .take()
            .context("DSpark session without context")?;
        let block_started = Instant::now();
        // DSPARK-2026-11-23-INK2M-MTPLX-A: proposals left on the GPU, verified and decided there, one read
        if count > 0 && self.sampled.is_none() && self.device_rounds && self.device.is_some() {
            let (proposals, accepted, token, features) =
                self.device_round(target, draft, next, &last_feature, count)?;
            let finish = Instant::now();
            let first = self.finish_block(
                target,
                draft,
                block_started,
                proposals,
                accepted,
                token,
                features,
                false,
                false,
            )?;
            self.stats.device.finish_ms += finish.elapsed().as_secs_f64() * 1e3;
            return Ok(first);
        }
        let t = Instant::now();
        let mut draft_rows: Option<Vec<Vec<f32>>> = None;
        let proposals = if count == 0 {
            Vec::new()
        } else {
            let anchor = Array::from_u32(&[next], &[1])?;
            match self.sampled.as_mut() {
                None => draft
                    .propose(target, &anchor, &last_feature, count)?
                    .to_u32()?,
                Some((temperature, state)) => {
                    *state ^= *state << 13;
                    *state ^= *state >> 7;
                    *state ^= *state << 17;
                    let (tokens, logits) = draft.propose_sampled(
                        target,
                        &anchor,
                        &last_feature,
                        count,
                        *temperature,
                        *state,
                    )?;
                    let vocab = usize::try_from(logits.shape()[1])?;
                    let values = logits.log_probs()?.to_f32()?;
                    ensure!(
                        values.len() == count * vocab,
                        "draft rows do not match the proposals"
                    );
                    draft_rows = Some(values.chunks_exact(vocab).map(<[f32]>::to_vec).collect());
                    tokens.to_u32()?
                }
            }
        };
        self.stats.draft_ms += t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mut block = Vec::with_capacity(1 + proposals.len());
        block.push(next);
        block.extend(&proposals);
        // DSPARK-INK2M-P4b: without proposals the anchor is always kept, so the committed decode
        // forward (features captured) replaces the rollback-ready verification and its commit
        let committed = proposals.is_empty();
        let (logits, features) = if committed {
            target.forward_with_features(&block, &self.taps)?
        } else {
            target.verify(&block, &self.taps)?
        };
        let (accepted, token) = choose(&proposals, draft_rows.as_deref(), &logits)?;
        ensure!(
            accepted <= proposals.len(),
            "DSpark accepted more than proposed"
        );
        self.stats.verify_ms += t.elapsed().as_secs_f64() * 1e3;
        self.finish_block(
            target,
            draft,
            block_started,
            proposals,
            accepted,
            token,
            features,
            committed,
            false,
        )
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D5: a block of the lookup's `proposals` after the anchor `next`: verified with
    /// the drafter's taps (its context follows the kept rows), decided on the GPU when the session draws there
    /// (`inkling2::decide_block`: the greedy equality or the sampled point rule, both exact for deterministic
    /// proposals), else by the caller's `choose`. Counted in the lookup's statistics and acceptance only.
    fn lookup_block<D: BlockDrafter + ?Sized>(
        &mut self,
        target: &mut Inkling2,
        draft: &mut D,
        next: u32,
        proposals: &[u32],
        choose: &mut BlockChoice<'_>,
    ) -> Result<u32> {
        ensure!(!proposals.is_empty(), "empty lookup block");
        if let Some(lookup) = self.lookup.as_mut() {
            // this block observes the lookup itself: no shadow over the same tokens
            lookup.shadow = None;
        }
        let started = Instant::now();
        let mut block = Vec::with_capacity(1 + proposals.len());
        block.push(next);
        block.extend(proposals);
        let (accepted, token, features) = match self.device.as_mut() {
            Some(device) if self.device_rounds => {
                let rule = device.rule;
                let noise_seed = device.seeds.next_seed();
                let uniform_seed = device.seeds.next_seed();
                let ids = Array::from_u32(&block, &[block.len() as i32])?;
                let given = Array::from_u32(proposals, &[proposals.len() as i32])?;
                let (logits, features) = target.verify_device(&ids, &self.taps)?;
                let packet =
                    crate::inkling2::decide_block(&logits, &given, rule, noise_seed, uniform_seed)?
                        .to_u32()?;
                ensure!(
                    packet.len() == 2 + proposals.len() && packet[2..] == *proposals,
                    "invalid lookup decision packet {packet:?}"
                );
                #[cfg(test)]
                D5_RECORD.with(|record| -> Result<()> {
                    if let Some(record) = record.borrow_mut().as_mut() {
                        let shape = logits.shape();
                        let rows = logits
                            .reshape(&[shape[1], shape[2]])?
                            .log_probs()?
                            .argmax()?
                            .to_u32()?;
                        record.push((proposals.to_vec(), rows, packet[0] as usize, packet[1]));
                    }
                    Ok(())
                })?;
                (usize::try_from(packet[0])?, packet[1], features)
            }
            _ => {
                let (logits, features) = target.verify(&block, &self.taps)?;
                let (accepted, token) = choose(proposals, None, &logits)?;
                (accepted, token, features)
            }
        };
        #[cfg(test)]
        let (accepted, token) = if D5_MUTATION.with(std::cell::Cell::get) {
            // mutation: every lookup proposal delivered, whatever the target chose
            (proposals.len(), token)
        } else {
            (accepted, token)
        };
        let vocab = u32::try_from(target.unpadded_vocab())?;
        ensure!(
            accepted <= proposals.len() && token < vocab,
            "invalid lookup decision ({accepted} of {}, token {token})",
            proposals.len()
        );
        let first = self.finish_block(
            target,
            draft,
            started,
            proposals.to_vec(),
            accepted,
            token,
            features,
            false,
            true,
        )?;
        self.stats.lookup_ms += started.elapsed().as_secs_f64() * 1e3;
        Ok(first)
    }

    /// DSPARK-2026-11-23-INK2M-MTPLX-A: one block of `count` >= 1 proposals with nothing read on the way:
    /// the drafter's proposals stay on the GPU, the verification of the anchor and the proposals is built
    /// from them, the decision (`inkling2::decide_block`, the session's GPU rule) runs after it, and the
    /// packet is the only read. Returns the proposals, the accepted count, the token after them and the
    /// verified features; nothing is committed yet.
    fn device_round<D: BlockDrafter + ?Sized>(
        &mut self,
        target: &mut Inkling2,
        draft: &D,
        next: u32,
        last_feature: &Array,
        count: usize,
    ) -> Result<(Vec<u32>, usize, u32, Array)> {
        let ratio = self.ratio_rule() && draft.supports_ratio();
        let couple = !ratio && self.couples() && draft.supports_coupling();
        let device = self
            .device
            .as_mut()
            .context("DSpark device round without GPU draw")?;
        let rule = device.rule;
        let noise_seed = device.seeds.next_seed();
        let uniform_seed = device.seeds.next_seed();
        // S3: the drafter's own noise, a third independent seed (only for ratio blocks: the other rules keep their
        // seed sequence)
        let draft_seed = if ratio { device.seeds.next_seed() } else { 0 };
        let t0 = Instant::now();
        let anchor = Array::from_u32(&[next], &[1])?;
        // D2: one standard Gumbel row per verified row, shared by the drafter's proposals and the target's
        // draws; a new seed every block
        let coupled = match rule {
            crate::inkling2::DrawRule::Softmax(temperature) if couple => Some((
                temperature,
                Array::gumbel_noise(&[count as i32 + 1, target.unpadded_vocab()], noise_seed)?,
            )),
            _ => None,
        };
        let ratio_draw = match rule {
            crate::inkling2::DrawRule::Softmax(temperature) if ratio => Some((
                temperature,
                draft.propose_ratio(
                    target,
                    &anchor,
                    last_feature,
                    count,
                    temperature,
                    draft_seed,
                )?,
            )),
            _ => None,
        };
        let proposals = match (&ratio_draw, &coupled) {
            (Some((_, drawn)), _) => drawn.tokens.clone(),
            (None, Some((temperature, noise))) => {
                draft.propose_coupled(target, &anchor, last_feature, count, *temperature, noise)?
            }
            (None, None) => draft.propose(target, &anchor, last_feature, count)?,
        };
        if self.overlap {
            // A3: the drafter runs on the GPU while the verification graph is built
            proposals.async_eval()?;
        }
        let block = Array::concatenate(&[&anchor, &proposals], 0)?;
        let t1 = Instant::now();
        let (logits, features) = target.verify_device(&block, &self.taps)?;
        let t2 = Instant::now();
        let packet = match (&ratio_draw, &coupled) {
            (Some((temperature, drawn)), _) => {
                self.stats.ratio_blocks += 1;
                crate::inkling2::decide_ratio(
                    &logits,
                    &proposals,
                    &drawn.candidates,
                    &drawn.q,
                    &drawn.chosen,
                    *temperature,
                    noise_seed,
                    uniform_seed,
                )?
            }
            (None, Some((temperature, noise))) => {
                self.stats.coupled_blocks += 1;
                crate::inkling2::decide_coupled(&logits, &proposals, *temperature, noise)?
            }
            (None, None) => {
                crate::inkling2::decide_block(&logits, &proposals, rule, noise_seed, uniform_seed)?
            }
        };
        let t3 = Instant::now();
        let packet = packet.to_u32()?;
        let t4 = Instant::now();
        let ms = |a: Instant, b: Instant| (b - a).as_secs_f64() * 1e3;
        self.stats.draft_ms += ms(t0, t1);
        self.stats.verify_ms += ms(t1, t4);
        let split = &mut self.stats.device;
        split.blocks += 1;
        split.draft_build_ms += ms(t0, t1);
        split.verify_build_ms += ms(t1, t2);
        split.decide_build_ms += ms(t2, t3);
        split.wait_ms += ms(t3, t4);
        ensure!(
            packet.len() == 2 + count,
            "DSpark decision packet has {} values, expected {}",
            packet.len(),
            2 + count
        );
        let accepted = usize::try_from(packet[0])?;
        let token = packet[1];
        let vocab = u32::try_from(target.unpadded_vocab())?;
        ensure!(
            accepted <= count && token < vocab && packet[2..].iter().all(|&p| p < vocab),
            "invalid DSpark decision packet {packet:?}"
        );
        Ok((packet[2..].to_vec(), accepted, token, features))
    }

    /// The end of a block: commit of the kept rows (unless the forward already committed them), the drafter's
    /// context, statistics and the delivery queue. Returns the first token to deliver.
    #[allow(clippy::too_many_arguments)]
    fn finish_block<D: BlockDrafter + ?Sized>(
        &mut self,
        target: &mut Inkling2,
        draft: &mut D,
        block_started: Instant,
        proposals: Vec<u32>,
        accepted: usize,
        token: u32,
        features: Array,
        committed: bool,
        from_lookup: bool,
    ) -> Result<u32> {
        let t = Instant::now();
        if !committed {
            target.commit(accepted + 1)?;
        }
        draft.append_context(&features.slice(0, 0, accepted as i32 + 1)?)?;
        self.last_feature = Some(features.slice(0, accepted as i32, accepted as i32 + 1)?);
        self.stats.commit_ms += t.elapsed().as_secs_f64() * 1e3;
        if from_lookup {
            // D5: the lookup's own acceptance and counters; the drafter's estimates do not move
            if let Some(lookup) = self.lookup.as_mut() {
                lookup.policy.observe(proposals.len(), accepted);
            }
            self.stats.lookup_blocks += 1;
            self.stats.lookup_proposed += proposals.len();
            self.stats.lookup_accepted += accepted;
            if self.stats.lookup_width_blocks.len() <= proposals.len() {
                self.stats
                    .lookup_width_blocks
                    .resize(proposals.len() + 1, 0);
            }
            self.stats.lookup_width_blocks[proposals.len()] += 1;
            self.pending.extend(&proposals[..accepted]);
            self.pending.push_back(token);
            let first = self.pending.pop_front().context("empty lookup block")?;
            self.expected = Some(first);
            return Ok(first);
        }
        self.stats.blocks += 1;
        self.stats.proposed += proposals.len();
        self.stats.accepted += accepted;
        if let Some(policy) = self.policy.as_mut() {
            policy.observe(proposals.len(), accepted);
            if let Some(slot) = self.stats.width_blocks.get_mut(proposals.len()) {
                *slot += 1;
            }
            if let Some(slot) = self.stats.width_ms.get_mut(proposals.len()) {
                *slot += block_started.elapsed().as_secs_f64() * 1e3;
            }
        }
        for slot in self.stats.accepted_at.iter_mut().take(accepted) {
            *slot += 1;
        }
        self.pending.extend(&proposals[..accepted]);
        self.pending.push_back(token);
        let first = self.pending.pop_front().context("empty DSpark block")?;
        self.expected = Some(first);
        Ok(first)
    }

    fn next_width(&mut self) -> usize {
        match self.policy.as_mut() {
            Some(policy) => policy.next_width(),
            None => self.width,
        }
    }

    /// SPEED-INK2M-P14: a width-0 block anchored on `anchor` (`[1]` on the GPU), committed, its token drawn
    /// on the GPU (lazy; submitted without waiting when pipelined).
    fn device_block<D: BlockDrafter + ?Sized>(
        &mut self,
        target: &mut Inkling2,
        draft: &mut D,
        anchor: &Array,
    ) -> Result<Array> {
        let device = self
            .device
            .as_mut()
            .context("DSpark session without GPU draw")?;
        let started = Instant::now();
        let (logits, features) =
            target.forward_device(anchor, Some(&self.taps), device.pipelined)?;
        let features = features.context("features")?;
        let token = crate::inkling2::draw_token(&logits, device.rule, device.seeds.next_seed())?;
        if device.pipelined {
            token.async_eval()?;
        }
        self.stats.verify_ms += started.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        draft.append_context(&features)?;
        self.last_feature = Some(features);
        self.stats.commit_ms += t.elapsed().as_secs_f64() * 1e3;
        self.stats.blocks += 1;
        if let Some(policy) = self.policy.as_mut() {
            policy.observe(0, 0);
            if let Some(slot) = self.stats.width_blocks.get_mut(0) {
                *slot += 1;
            }
            if let Some(slot) = self.stats.width_ms.get_mut(0) {
                *slot += started.elapsed().as_secs_f64() * 1e3;
            }
        }
        Ok(token)
    }

    /// SPEED-INK2M-P14: delivers `after`, the token after the anchor of the block that just ran (`remaining`
    /// counts it). When pipelined and a token will follow it, the next block's width is drawn first: a
    /// width-0 block is submitted before `after` is read, otherwise the width is kept for `advance`.
    fn deliver<D: BlockDrafter + ?Sized>(
        &mut self,
        target: &mut Inkling2,
        draft: &mut D,
        after: Array,
        context_limit: usize,
        remaining: usize,
        likely: bool,
    ) -> Result<u32> {
        let pipelined = self.device.as_ref().is_some_and(|d| d.pipelined);
        let offset = usize::try_from(target.offset())?;
        let hold = self
            .lookup
            .as_ref()
            .is_some_and(|lookup| lookup.policy.wants_block(likely));
        if hold && pipelined && remaining >= 2 && offset < context_limit {
            // D5: the token before `after` had a lookup proposal and a lookup block would pay: wait for
            // `after` so the lookup can read it (the next `advance` decides)
            self.stats.lookup_held += 1;
        } else if pipelined && remaining >= 2 && offset < context_limit {
            let width = self.next_width();
            if width.min(remaining - 2).min(context_limit - offset - 1) == 0 {
                self.ahead = Some(self.device_block(target, draft, &after)?);
            } else {
                self.decided = Some(width);
            }
        }
        let token = crate::inkling2::single_token(&after)?;
        self.expected = Some(token);
        Ok(token)
    }
}

/// DSPARK-INK2M-P4: the DSpark width of each block, chosen from the acceptance observed so far.
///
/// `acceptance[i]` estimates the probability that proposal `i` is accepted given that proposals `0..i`
/// were (exponential average over the blocks that reached it). A block of width `w` delivers
/// E(w) = 1 + sum over i < w of the product of acceptance[0..=i] tokens and costs C(w) plain decoding
/// steps. The width maximizing E(w) / C(w) is used; width 0 is a plain step. Every `PROBE_EVERY` blocks a
/// wider block is tried so the estimates stay current. The choice depends only on the accepted counts and
/// constant costs, never on timings, so a generation stays reproducible; the acceptance rule, and so the
/// law of every token, does not depend on the width.
///
/// Two settings: P4c (`new`), C(w) = 1 + (VERIFY_COST + DRAFT_COST) * w calibrated on the FP16 dense target
/// and probes at the full width every `PROBE_EVERY` blocks; P10c (`measured`), C(w) from the block times
/// measured on the 8-bit dense target and probes one position past the best width, spaced out (8, 16, 32,
/// up to `PROBE_MAX` blocks) while the best width stays 0 after them, so plain steps pay little for them.
#[derive(Clone, Debug)]
pub struct WidthPolicy {
    acceptance: Vec<f64>,
    /// costs[w]: cost of a block of width w, in plain decoding steps (costs[0] = 1)
    costs: Vec<f64>,
    /// probe at best + 1, spaced out while DSpark does not pay (P10c), instead of the full width every
    /// `PROBE_EVERY` blocks (P4)
    narrow_probes: bool,
    blocks: usize,
    /// DFLASH2-2026-11-25-INK2M-F3: a probe whose proposals were all kept is followed by another probe (gap 1)
    /// instead of waiting 8 to 64 blocks; a probe with a rejection keeps the P10c rule (gap at least `PROBE_EVERY`)
    streak: bool,
    /// P10c: blocks between probes, blocks since the last probe, and whether the last block was a probe
    probe_gap: usize,
    since_probe: usize,
    probing: bool,
}

impl WidthPolicy {
    pub const ALPHA: f64 = 0.15;
    pub const PRIOR: f64 = 0.75;
    pub const PROBE_EVERY: usize = 8;
    /// P10c: largest gap between probes while the best width stays 0.
    pub const PROBE_MAX: usize = 64;
    pub const VERIFY_COST: f64 = 0.20;
    pub const DRAFT_COST: f64 = 0.07;
    /// DSPARK-INK2M-P10a: block wall time of widths 0..=3 over the width-0 block, Inkling 2 Mini R3 with
    /// 8-bit dense projections in the bridge, sampled rows (T = 1: every verified row is drawn on the CPU).
    pub const SAMPLED_8BIT_COSTS: [f64; 4] = [1.0, 1.56, 1.83, 2.17];
    /// Same, greedy rows (argmax of every verified row; width-0 cost = the bridge greedy AR step). Measured
    /// only: greedy sessions keep the P4c costs, this table lost 2.2 % on short greedy answers (P10b).
    pub const GREEDY_8BIT_COSTS: [f64; 4] = [1.0, 1.62, 1.95, 2.22];
    /// DFLASH2-2026-11-24-INK2M-P2 (c): block of `w` DFlash2 proposals over a plain step, measured at the bridge
    /// (R3-vision, gencode T = 1, 8-bit drafter with the GPU selector: 36.2 ms plain; 97.7 / 95.4 / 101.3 / 122.4 /
    /// 150.4 / 134.7 ms at widths 1 / 3 / 4 / 5 / 6 / 7 over 45 / 28 / 87 / 16 / 13 / 8 blocks), fitted as
    /// 82.5 + 6.7 w ms (least squares weighted by blocks) since the raw means do not grow with the width.
    pub const DFLASH2_COSTS_P2: [f64; 8] = [1.0, 2.46, 2.65, 2.83, 3.02, 3.2, 3.39, 3.57];
    /// DFLASH2-2026-11-25-INK2M-F2: the same blocks after S4 (the context backlog no longer rides on every block):
    /// 37.2 ms plain and 66.5 / 118.1 / 128.4 / 149.6 ms at widths 1 / 5 / 6 / 7 over 6 / 9 / 19 / 111 blocks
    /// (bridge, R3-vision, gencode T = 1, ratio rule), fitted as 52 + 13.9 w ms = 1.40 + 0.375 w plain steps.
    pub const DFLASH2_COSTS_F2: [f64; 8] = [1.0, 1.775, 2.15, 2.525, 2.9, 3.275, 3.65, 4.025];
    /// DFLASH2-2026-11-25-INK2M-F3: the block means of 12 bridge passes with the whole block (F1), at least 139
    /// blocks a width: 35.5 ms plain and 61.1 / 72.8 / 83.7 / 96.5 / 108.8 / 123.6 / 140.7 ms at widths 1..=7.
    pub const DFLASH2_COSTS_F3: [f64; 8] = [1.0, 1.72, 2.05, 2.36, 2.72, 3.06, 3.48, 3.96];

    /// INVARIANT-2026-11-25-INK2M-L3-3: the F3 means less the harness gain of the expert panel on a verification
    /// of w + 1 rows (1.4 / 3.3 / 4.2 / 7.2 / 11.2 / 16.1 / 16.5 ms). Bridge blocks with the panel then measured
    /// 1.60-1.64 / 1.92 / 2.16 / 2.46 / 2.70-2.74 / 3.03-3.09 / 3.44-3.48 steps; this table against the F3 one with
    /// the panel: 37.6 against 34.9 tok/s (gencode T = 1, 4 passes each). The default.
    pub const DFLASH2_COSTS: [f64; 8] = [1.0, 1.68, 1.96, 2.24, 2.52, 2.75, 3.03, 3.5];

    /// Setting of `MLXL3_INKLING_DFLASH2_COSTS`: "p2" the table measured before S4, "f2" the F2 fit, "f3" the
    /// means measured without the expert panel, else the current one (with the panel; "l3" names it too).
    pub fn dflash2_costs(value: Option<&str>) -> &'static [f64; 8] {
        match value {
            Some("p2") => &Self::DFLASH2_COSTS_P2,
            Some("f2") => &Self::DFLASH2_COSTS_F2,
            Some("f3") => &Self::DFLASH2_COSTS_F3,
            _ => &Self::DFLASH2_COSTS,
        }
    }

    /// F3 setting of `MLXL3_INKLING_DFLASH2_STREAK`: probes in a row after a kept probe unless "0".
    pub fn probe_streak_setting(value: Option<&str>) -> bool {
        value != Some("0")
    }

    /// P4c setting: linear FP16 costs, full-width probes.
    pub fn new(max_width: usize) -> Self {
        Self {
            acceptance: vec![Self::PRIOR; max_width],
            costs: (0..=max_width).map(Self::linear_cost).collect(),
            narrow_probes: false,
            streak: false,
            blocks: 0,
            probe_gap: Self::PROBE_EVERY,
            since_probe: 0,
            probing: false,
        }
    }

    /// P10c setting: measured costs of widths `0..=max_width` (the first `max_width + 1` of `costs`),
    /// probes one position past the best width, spaced out while the best width stays 0.
    pub fn measured(costs: &[f64], max_width: usize) -> Result<Self> {
        ensure!(max_width >= 1, "adaptive DSpark width needs a proposal");
        let costs = costs
            .get(..=max_width)
            .context("no measured cost for the DSpark width")?;
        ensure!(costs[0] == 1.0, "a width-0 block is one plain step");
        ensure!(
            costs.windows(2).all(|w| w[1].is_finite() && w[1] > w[0]),
            "DSpark block costs must grow with the width"
        );
        Ok(Self {
            acceptance: vec![Self::PRIOR; max_width],
            costs: costs.to_vec(),
            narrow_probes: true,
            streak: false,
            blocks: 0,
            probe_gap: Self::PROBE_EVERY,
            since_probe: 0,
            probing: false,
        })
    }

    /// DFLASH2-2026-11-24-INK2M-P3 (3): the full width every `PROBE_EVERY` blocks (P4 probes) with these costs, so
    /// the far positions keep being observed when the best width is narrow.
    pub fn with_full_probes(mut self) -> Self {
        self.narrow_probes = false;
        self
    }

    /// DFLASH2-2026-11-25-INK2M-F3: probes in a row while their proposals are all kept.
    pub fn with_probe_streak(mut self) -> Self {
        self.streak = true;
        self
    }

    pub fn max_width(&self) -> usize {
        self.acceptance.len()
    }

    /// Tokens a block of `width` delivers on average under the current estimates.
    pub fn expected_tokens(&self, width: usize) -> f64 {
        let mut reach = 1.0;
        let mut total = 1.0;
        for &a in self.acceptance.iter().take(width) {
            reach *= a;
            total += reach;
        }
        total
    }

    /// P4 cost model of a block of `width`, in plain decoding steps.
    pub fn linear_cost(width: usize) -> f64 {
        1.0 + (Self::VERIFY_COST + Self::DRAFT_COST) * width as f64
    }

    /// Cost of a block of `width` (at most `max_width`) under this policy, in plain decoding steps.
    pub fn cost(&self, width: usize) -> f64 {
        self.costs[width]
    }

    /// The width with the most tokens per unit of cost (the smaller one on a tie).
    pub fn best(&self) -> usize {
        (0..=self.max_width())
            .map(|w| (w, self.expected_tokens(w) / self.cost(w)))
            .fold((0, f64::MIN), |best, (w, rate)| {
                if rate > best.1 + 1e-12 {
                    (w, rate)
                } else {
                    best
                }
            })
            .0
    }

    /// The width of the next block: the best one, or a wider one on a probe block.
    pub fn next_width(&mut self) -> usize {
        self.blocks += 1;
        let best = self.best();
        if !self.narrow_probes {
            return if self.blocks.is_multiple_of(Self::PROBE_EVERY) {
                self.max_width()
            } else {
                best
            };
        }
        self.since_probe += 1;
        if self.since_probe < self.probe_gap {
            return best;
        }
        self.since_probe = 0;
        self.probing = true;
        (best + 1).min(self.max_width())
    }

    /// P10c: blocks until the next probe may run (the current gap).
    pub fn probe_gap(&self) -> usize {
        self.probe_gap
    }

    /// Records a block of `width` proposals of which the first `accepted` were accepted.
    pub fn observe(&mut self, width: usize, accepted: usize) {
        for (i, a) in self.acceptance.iter_mut().enumerate().take(width) {
            // proposal i is reached only when all the earlier ones were accepted
            if i > accepted {
                break;
            }
            let hit = if accepted > i { 1.0 } else { 0.0 };
            *a = (1.0 - Self::ALPHA) * *a + Self::ALPHA * hit;
        }
        // P10c: a probe after which plain steps stay best doubles the gap; one that makes DSpark pay
        // again resets it
        if std::mem::take(&mut self.probing) {
            self.probe_gap = if self.streak && width > 0 && accepted >= width {
                // F3: a kept probe pays for itself; the next block probes again
                1
            } else if self.best() == 0 {
                // at least PROBE_EVERY again after a streak (the gap was 1)
                (self.probe_gap * 2).clamp(Self::PROBE_EVERY, Self::PROBE_MAX)
            } else {
                Self::PROBE_EVERY
            };
        }
    }

    pub fn acceptance(&self) -> &[f64] {
        &self.acceptance
    }

    #[cfg(test)]
    fn with_acceptance(acceptance: Vec<f64>) -> Self {
        let mut p = Self::new(acceptance.len());
        p.acceptance = acceptance;
        p
    }
}

/// Counters of one speculative generation.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct SpecStats {
    pub blocks: usize,
    pub proposed: usize,
    pub accepted: usize,
    /// accepted[i]: blocks whose proposal i was accepted
    pub accepted_at: Vec<usize>,
    pub draft_ms: f64,
    pub verify_ms: f64,
    pub commit_ms: f64,
    pub prefill_ms: f64,
    /// DSPARK-INK2M-P4: blocks verified at each width (index = width), filled when the width adapts.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub width_blocks: Vec<usize>,
    /// DSPARK-INK2M-P10a: wall time (ms) of the blocks of each width (draft + verification + commit).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub width_ms: Vec<f64>,
    /// DSPARK-2026-11-23-INK2M-MTPLX-A3: host-side split of the blocks decided on the GPU.
    #[serde(skip_serializing_if = "DeviceRoundTimes::is_empty")]
    pub device: DeviceRoundTimes,
    /// DSPARK-2026-11-24-INK2M-MAYA2-D2: blocks with proposals decided by the coupled rule.
    #[serde(skip_serializing_if = "is_zero")]
    pub coupled_blocks: usize,
    /// DFLASH2-2026-11-24-INK2M-S3: blocks with proposals decided by the ratio rule.
    #[serde(skip_serializing_if = "is_zero")]
    pub ratio_blocks: usize,
    /// DFLASH2-2026-11-24-INK2M-P0: the drafter of the session (`dspark` or `dflash2`).
    #[serde(skip_serializing_if = "str::is_empty")]
    pub drafter: &'static str,
    /// DSPARK-2026-11-24-INK2M-MAYA2-D5: lookup blocks (not counted in `blocks`), their proposals and
    /// accepted proposals, blocks by width, steps not run ahead so a lookup could follow, and their wall time.
    #[serde(skip_serializing_if = "is_zero")]
    pub lookup_blocks: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub lookup_proposed: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub lookup_accepted: usize,
    #[serde(skip_serializing_if = "is_zero")]
    pub lookup_held: usize,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub lookup_width_blocks: Vec<usize>,
    #[serde(skip_serializing_if = "is_zero_ms")]
    pub lookup_ms: f64,
}

fn is_zero_ms(value: &f64) -> bool {
    *value == 0.0
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

/// DSPARK-2026-11-23-INK2M-MTPLX-A3: where the host spends a GPU-decided block (sums, ms): building the
/// drafter's graph (and submitting it when overlapped), building and submitting the verification, building
/// the decision, waiting for the packet, then the commit and the drafter's context.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct DeviceRoundTimes {
    pub blocks: usize,
    pub draft_build_ms: f64,
    pub verify_build_ms: f64,
    pub decide_build_ms: f64,
    pub wait_ms: f64,
    pub finish_ms: f64,
}

impl DeviceRoundTimes {
    pub fn is_empty(&self) -> bool {
        self.blocks == 0
    }
}

fn argmax_rows(logits: &Array) -> Result<Vec<u32>> {
    logits.argmax()?.reshape(&[-1])?.to_u32()
}

/// Greedy speculative decoding: every emitted token is the target's argmax at its verified row.
pub fn generate_greedy(
    target: &mut Inkling2,
    draft: &mut Inkling2Draft,
    prompt: &[u32],
    max_new: usize,
    width: usize,
    stop: &[u32],
) -> Result<(Vec<u32>, SpecStats)> {
    ensure!(
        width >= 1 && width <= draft.max_proposals(),
        "DSpark width outside the trained block"
    );
    target.reset();
    draft.reset();
    let taps = draft.taps().to_vec();
    let mut stats = SpecStats {
        accepted_at: vec![0; width],
        ..SpecStats::default()
    };
    let t = Instant::now();
    let (logits, features) = target.verify(prompt, &taps)?;
    target.commit(prompt.len())?;
    draft.append_context(&features)?;
    let rows = prompt.len() as i32;
    let mut last_feature = features.slice(0, rows - 1, rows)?;
    let mut anchor = *argmax_rows(&logits.reshape(&[rows, -1])?.slice(0, rows - 1, rows)?)?
        .first()
        .context("prefill logits")?;
    stats.prefill_ms = t.elapsed().as_secs_f64() * 1e3;
    let mut out = vec![anchor];
    while out.len() < max_new && !stop.contains(&anchor) {
        let count = width.min(max_new - out.len());
        let t = Instant::now();
        let anchor_ids = Array::from_u32(&[anchor], &[1])?;
        let proposals = draft
            .propose(target, &anchor_ids, &last_feature, count)?
            .to_u32()?;
        stats.draft_ms += t.elapsed().as_secs_f64() * 1e3;
        let t = Instant::now();
        let mut block = vec![anchor];
        block.extend(&proposals);
        let (logits, features) = target.verify(&block, &taps)?;
        let ids = argmax_rows(&logits.reshape(&[block.len() as i32, -1])?)?;
        stats.verify_ms += t.elapsed().as_secs_f64() * 1e3;
        let accepted = proposals
            .iter()
            .zip(&ids)
            .take_while(|(p, t)| p == t)
            .count();
        let t = Instant::now();
        target.commit(accepted + 1)?;
        let kept = features.slice(0, 0, accepted as i32 + 1)?;
        draft.append_context(&kept)?;
        last_feature = features.slice(0, accepted as i32, accepted as i32 + 1)?;
        stats.commit_ms += t.elapsed().as_secs_f64() * 1e3;
        stats.blocks += 1;
        stats.proposed += count;
        stats.accepted += accepted;
        for slot in stats.accepted_at.iter_mut().take(accepted) {
            *slot += 1;
        }
        for &token in &proposals[..accepted] {
            out.push(token);
            if stop.contains(&token) || out.len() >= max_new {
                return Ok((out, stats));
            }
        }
        anchor = ids[accepted];
        out.push(anchor);
    }
    out.truncate(max_new);
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn centered_rel(a: &[f32], b: &[f32]) -> f64 {
        let mean = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
        let (ma, mb) = (mean(a), mean(b));
        let (mut n, mut d) = (0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            let (x, y) = (x as f64 - ma, y as f64 - mb);
            n += (x - y) * (x - y);
            d += y * y;
        }
        (n / d.max(1e-30)).sqrt()
    }

    fn prompt() -> Result<Vec<u32>> {
        let manifest: serde_json::Value = serde_json::from_reader(File::open(
            std::path::Path::new(&std::env::var("MLXL3_INKLING_ORACLE")?).join("manifest.json"),
        )?)?;
        let index: usize = std::env::var("MLXL3_INKLING_PROMPT_ROW")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(0);
        Ok(serde_json::from_value(
            manifest["rows"][index]["tokens"].clone(),
        )?)
    }

    /// AR greedy against DSpark greedy (widths 1..7): tokens, time, acceptance, and the target state after
    /// rejections checked against a teacher-forced AR recomputation of the same tokens.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ORACLE"]
    fn speculative_profile() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let steps: usize = std::env::var("MLXL3_INKLING_STEPS")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(64);
        let widths: Vec<usize> = std::env::var("MLXL3_INKLING_WIDTHS")
            .unwrap_or_else(|_| "1,3,5,7".into())
            .split(',')
            .map(|v| v.parse())
            .collect::<std::result::Result<_, _>>()?;
        let prompt = prompt()?;
        let mut target = Inkling2::load(&dir)?;
        let mut draft = Inkling2Draft::load(&dir, &target)?;
        let ar = |target: &mut Inkling2| -> Result<(Vec<u32>, f64)> {
            target.reset();
            let t = Instant::now();
            let mut next = target
                .forward_tokens(&prompt)?
                .argmax()?
                .reshape(&[-1])?
                .to_u32()?[0];
            let mut out = vec![next];
            while out.len() < steps {
                next = target.forward(next)?.argmax()?.reshape(&[-1])?.to_u32()?[0];
                out.push(next);
            }
            Ok((out, t.elapsed().as_secs_f64() * 1e3))
        };
        ar(&mut target)?; // warm-up
        let (ar_tokens, ar_ms) = ar(&mut target)?;
        let mut report = vec![
            serde_json::json!({"mode": "ar", "ms": ar_ms, "tok_s": steps as f64 / ar_ms * 1e3}),
        ];
        for &width in &widths {
            generate_greedy(&mut target, &mut draft, &prompt, 16, width, &[])?; // warm-up
            let t = Instant::now();
            let (tokens, stats) =
                generate_greedy(&mut target, &mut draft, &prompt, steps, width, &[])?;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            ensure!(tokens.len() == steps, "speculative output length");
            let same_prefix = tokens
                .iter()
                .zip(&ar_tokens)
                .take_while(|(a, b)| a == b)
                .count();
            let same = tokens
                .iter()
                .zip(&ar_tokens)
                .filter(|(a, b)| a == b)
                .count();
            // state check: next-token logits after the speculative run vs a teacher-forced AR pass
            let spec_next = target.forward(tokens[steps - 1])?.to_f32()?;
            target.reset();
            let mut seq = prompt.clone();
            seq.extend(&tokens[..steps - 1]);
            target.forward_tokens(&seq[..prompt.len()])?;
            for &token in &seq[prompt.len()..] {
                target.forward(token)?;
            }
            let ar_next = target.forward(tokens[steps - 1])?.to_f32()?;
            let state_rel = centered_rel(&spec_next, &ar_next);
            report.push(serde_json::json!({
                "mode": format!("dspark_w{width}"), "ms": ms, "tok_s": steps as f64 / ms * 1e3,
                "speedup": ar_ms / ms, "tokens_per_block": (stats.accepted + stats.blocks) as f64 / stats.blocks as f64,
                "acceptance": stats.accepted as f64 / stats.proposed.max(1) as f64, "stats": stats,
                "ar_same_prefix": same_prefix, "ar_same_tokens": same, "state_rel_vs_teacher_forced": state_rel,
            }));
            // Rollback mechanics are checked on the tiny export (inkling2::tests::commit_rollback_*); on the
            // real model batched and single-row numerics already differ, so only gross errors fail here.
            ensure!(
                state_rel < 1.0,
                "target state after speculative commits diverges grossly ({state_rel})"
            );
        }
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }

    fn write_f32(path: &std::path::Path, a: &Array) -> Result<()> {
        let v = a.astype(Dtype::Float32)?.to_f32()?;
        std::fs::write(
            path,
            v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>(),
        )?;
        Ok(())
    }

    /// SIZE-2026-11-21-INK2M-SLIM exporter (not a check): writes into the new directory `MLXL3_INKLING_SLIM_OUT`
    /// the tensors a slim checkpoint stores in place of dense ones, each computed by the load path it replaces:
    /// `head.safetensors` = the 8-bit head copy of `Precision::DEFAULT` (`Linear::quantized` of the unpadded FP16
    /// rows), `mtp-replacements.safetensors` = the draft projections at `MLXL3_INKLING_DSPARK_BITS=8`
    /// (`DraftLinear::load`) and the fold of `fold_from_head`, plus `export.json`.
    #[test]
    #[ignore = "writes files; requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_SLIM_OUT"]
    fn export_slim_tensors() -> Result<()> {
        use crate::inkling2::{Precision, write_safetensors};
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let out = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_SLIM_OUT")?);
        std::fs::create_dir(&out).context("MLXL3_INKLING_SLIM_OUT must not exist yet")?;
        let target = Inkling2::load_with(&dir, Precision::FULL)?;
        let vocab = target.unpadded_vocab();
        let head = target
            .head_weight()
            .context("the source checkpoint must have a dense head")?;
        let affine = |linear: DraftLinear, name: &str| -> Result<[Array; 3]> {
            match linear {
                DraftLinear::Affine {
                    w,
                    scales,
                    biases,
                    bits: 8,
                    group: crate::inkling2::AFFINE_GROUP,
                } => Ok([w, scales, biases]),
                _ => anyhow::bail!("{name}: expected an 8-bit affine copy"),
            }
        };
        let head_q = affine(
            DraftLinear::quantized(&head.slice(0, 0, vocab)?, 8)?,
            "head",
        )?;
        let names = ["weight", "scales", "biases"];
        let named = |prefix: &str, arrays: [Array; 3]| -> Vec<(String, Array)> {
            names
                .iter()
                .zip(arrays)
                .map(|(n, a)| (format!("{prefix}.{n}"), a))
                .collect()
        };
        let head_tensors = named("model.llm.unembed", head_q);
        write_safetensors(
            &out.join("head.safetensors"),
            &head_tensors
                .iter()
                .map(|(n, a)| (n.clone(), a))
                .collect::<Vec<_>>(),
        )?;
        let root: Root = serde_json::from_reader(File::open(dir.join("config.json"))?)?;
        let c = root.mtp_config;
        let path = dir.join("mtp.safetensors");
        let ckpt = Checkpoint {
            path: dir.clone(),
            model_type: "inkling2_dspark".into(),
            bits: None,
            size_bytes: std::fs::metadata(&path)?.len(),
            modules: Vec::new(),
            tensors: checkpoint::read_header(&path)?,
        };
        let mut tensors = Vec::new();
        let projections = draft_projections(&c);
        for (prefix, input, output) in &projections {
            let arrays = affine(
                DraftLinear::load(&ckpt, prefix, *input, *output, 8)?,
                prefix,
            )?;
            tensors.extend(named(prefix, arrays));
        }
        let (h, rank) = (c.hidden_size, c.markov_rank);
        let hidden_out = f32_weight(&ckpt, "markov_head.hidden_out.weight", &[h, rank])?;
        let hidden_bias = f32_weight(&ckpt, "markov_head.hidden_out.bias", &[h])?;
        let (fold, fold_bias) = fold_from_head(head, &hidden_out, &hidden_bias, vocab)?;
        Array::eval_many(&[&fold, &fold_bias])?;
        ensure!(
            fold.shape() == [vocab, rank] && fold_bias.shape() == [vocab],
            "fold shapes"
        );
        tensors.push((FOLD_WEIGHT.to_string(), fold));
        tensors.push((FOLD_BIAS.to_string(), fold_bias));
        write_safetensors(
            &out.join("mtp-replacements.safetensors"),
            &tensors
                .iter()
                .map(|(n, a)| (n.clone(), a))
                .collect::<Vec<_>>(),
        )?;
        let report = serde_json::json!({
            "trial": "SIZE-2026-11-21-INK2M-SLIM",
            "source": dir,
            "head": {"prefix": "model.llm.unembed", "bits": 8, "group": crate::inkling2::AFFINE_GROUP, "rows": vocab},
            "draft_projections": projections.iter().map(|(p, i, o)| serde_json::json!([p, i, o])).collect::<Vec<_>>(),
            "draft_bits": 8,
            "fold": [FOLD_WEIGHT, FOLD_BIAS],
        });
        std::fs::write(
            out.join("export.json"),
            serde_json::to_string_pretty(&report)?,
        )?;
        println!("{report}");
        Ok(())
    }

    /// SPEED-INK2M-P6a: `warm_up` leaves no state: the logits of a prompt and three decode steps, and
    /// a DSpark session's first proposals, are bit-identical before and after it; also reports how long
    /// the first and the second warm-up take.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ORACLE"]
    fn warm_up_leaves_no_state() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let prompt = prompt()?;
        let mut target = Inkling2::load(&dir)?;
        let mut draft = Inkling2Draft::load(&dir, &target)?;
        let run = |target: &mut Inkling2, draft: &mut Inkling2Draft| -> Result<Vec<u32>> {
            target.reset();
            draft.reset();
            let mut bits = Vec::new();
            let logits = target.forward_tokens(&prompt)?;
            bits.extend(logits.to_f32()?.iter().map(|v| v.to_bits()));
            for t in [1234u32, 278, 13] {
                bits.extend(target.forward(t)?.to_f32()?.iter().map(|v| v.to_bits()));
            }
            let mut session = DsparkSession::new(draft, DsparkSession::DEFAULT_WIDTH)?;
            let logits = session.prefill(target, draft, &prompt, &|| false)?;
            let mut next = *logits.chat_greedy_ids()?.first().context("prefill")?;
            for _ in 0..6 {
                let mut choose = |proposals: &[u32],
                                  _drafts: Option<&[Vec<f32>]>,
                                  verified: &Array|
                 -> Result<(usize, u32)> {
                    let a =
                        crate::speculative::greedy_accept(proposals, &verified.chat_greedy_ids()?)
                            .context("acceptance")?;
                    Ok((a.accepted_draft_tokens, a.target_token))
                };
                next = session.advance(target, draft, next, 4096, 64, &mut choose)?;
                bits.push(next);
            }
            Ok(bits)
        };
        let before = run(&mut target, &mut draft)?;
        let t = Instant::now();
        warm_up(&mut target, Some(&mut draft))?;
        let first = t.elapsed().as_secs_f64();
        let t = Instant::now();
        warm_up(&mut target, Some(&mut draft))?;
        let second = t.elapsed().as_secs_f64();
        let after = run(&mut target, &mut draft)?;
        ensure!(
            !before.is_empty() && before.len() == after.len(),
            "runs differ in length"
        );
        ensure!(
            before == after,
            "warm_up changed the logits or the DSpark tokens"
        );
        println!(
            "warm_up leaves no state ({} values compared); warm-up {first:.2} s, again {second:.2} s",
            before.len()
        );
        Ok(())
    }

    /// DSPARK-INK2M-P4b: a width-0 DSpark step (committed forward with features) gives the logits of the
    /// plain decode step bit for bit, over several consecutive tokens, and features close to those of the
    /// verification path on the same token.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ORACLE"]
    fn width_zero_step_is_the_plain_decode_step() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let prompt = prompt()?;
        let mut target = Inkling2::load(&dir)?;
        let draft = Inkling2Draft::load(&dir, &target)?;
        let taps = draft.taps().to_vec();
        let steps = [1234u32, 278, 13, 9001];
        // plain decode
        target.reset();
        target.forward_tokens(&prompt)?.eval()?;
        let mut plain = Vec::new();
        for &t in &steps {
            plain.push(target.forward(t)?.to_f32()?);
        }
        // width-0 DSpark steps
        target.reset();
        target.forward_tokens(&prompt)?.eval()?;
        let mut committed = Vec::new();
        let mut committed_features = Vec::new();
        for &t in &steps {
            let (logits, features) = target.forward_with_features(&[t], &taps)?;
            committed.push(logits.to_f32()?);
            committed_features.push(features.to_f32()?);
        }
        for (i, (a, b)) in plain.iter().zip(&committed).enumerate() {
            ensure!(
                !a.is_empty() && a.len() == b.len() && a.iter().all(|v| v.is_finite()),
                "step {i}: logits shape or finiteness"
            );
            ensure!(
                a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()),
                "step {i}: width-0 logits differ from the plain decode step"
            );
        }
        // the verification path on the same tokens: features at the path's tolerance
        target.reset();
        target.forward_tokens(&prompt)?.eval()?;
        let mut worst = 0f64;
        for (i, &t) in steps.iter().enumerate() {
            let (_, features) = target.verify(&[t], &taps)?;
            target.commit(1)?;
            let f = features.to_f32()?;
            let g = &committed_features[i];
            ensure!(
                f.len() == g.len() && !f.is_empty(),
                "step {i}: feature shape"
            );
            let (mut num, mut den) = (0f64, 0f64);
            for (x, y) in f.iter().zip(g) {
                num += f64::from(x - y).powi(2);
                den += f64::from(*x).powi(2);
            }
            worst = worst.max((num / den.max(1e-30)).sqrt());
        }
        println!(
            "width-0 logits identical to plain decode on {} steps; feature rel RMSE vs verify {worst:.2e}",
            steps.len()
        );
        ensure!(
            worst < 0.05,
            "features differ from the verification path: {worst}"
        );
        Ok(())
    }

    /// Wall time of each step of a cold, then a warm DSpark bridge session (prefill, then `advance` calls),
    /// with an evaluation after each step so the time lands where it is spent.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ORACLE"]
    fn dspark_session_cold_start() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let prompt = prompt()?;
        let t = Instant::now();
        let mut target = Inkling2::load(&dir)?;
        let load_target = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let mut draft = Inkling2Draft::load(&dir, &target)?;
        let load_draft = t.elapsed().as_secs_f64();
        let mut report =
            vec![serde_json::json!({"load_target_s": load_target, "load_draft_s": load_draft})];
        if std::env::var_os("MLXL3_INKLING_COLD_AR").is_some() {
            // the ordinary path first: is the cold cost the target's or the taps'?
            for round in 0..2 {
                target.reset();
                let t = Instant::now();
                target.forward_tokens(&prompt)?.eval()?;
                report.push(serde_json::json!({"ar_prefill_round": round, "seconds": t.elapsed().as_secs_f64()}));
            }
            target.reset();
        }
        for round in 0..2 {
            let mut session = DsparkSession::new(&draft, DsparkSession::DEFAULT_WIDTH)?;
            let t = Instant::now();
            let logits = session.prefill(&mut target, &mut draft, &prompt, &|| false)?;
            let mut next = *logits.chat_greedy_ids()?.first().context("prefill")?;
            let prefill = t.elapsed().as_secs_f64();
            let mut steps = Vec::new();
            for _ in 0..24 {
                let t = Instant::now();
                let mut choose = |proposals: &[u32],
                                  _drafts: Option<&[Vec<f32>]>,
                                  verified: &Array|
                 -> Result<(usize, u32)> {
                    let a =
                        crate::speculative::greedy_accept(proposals, &verified.chat_greedy_ids()?)
                            .context("acceptance")?;
                    Ok((a.accepted_draft_tokens, a.target_token))
                };
                next = session.advance(&mut target, &mut draft, next, 4096, 64, &mut choose)?;
                steps.push((t.elapsed().as_secs_f64() * 1e3 * 10.).round() / 10.);
            }
            report.push(
                serde_json::json!({"round": round, "prefill_s": prefill, "steps_ms": steps,
                "stats": session.stats}),
            );
        }
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D2: the coupled rule is on only for exactly "1".
    #[test]
    fn ratio_setting_is_on_unless_zero() {
        assert!(!ratio_setting(Some("0")));
        for value in [None, Some("1"), Some(""), Some("true")] {
            assert!(ratio_setting(value), "{value:?}");
        }
    }

    /// DFLASH2-2026-11-24-INK2M-S3: `inkling2::decide_ratio` keeps the target's law whatever the drafter proposes.
    /// Fixed target rows p_i over 24 tokens, drafter rows q_i over 16 of them (drawn per row, some tokens of p
    /// outside every candidate set), proposals drawn from q_i on the host, 40 000 blocks per case: for each position
    /// i, the token delivered at i among the blocks that reach it against p_i = softmax(row i / T) (chi-square
    /// below the 0.999 quantile, 23 degrees of freedom); the share of first proposals kept is sum min(p_0, q_0)
    /// within 0.01; the mutation that keeps every proposal is detected.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn decide_ratio_follows_the_target_law() -> Result<()> {
        use crate::inkling2::{RATIO_MUTATION, decide_ratio};
        const V: usize = 24;
        const C: usize = 16;
        const BLOCKS: usize = 40_000;
        const BATCH: usize = 1_000;
        let mut rng = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            ((rng >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        };
        // target rows in [-2, 2]; drafter rows: 16 distinct candidates, scores in [-2, 2]
        let p_rows: Vec<Vec<f32>> = (0..4)
            .map(|_| (0..V).map(|_| (next() * 4.0 - 2.0) as f32).collect())
            .collect();
        let mut q_sets = Vec::new();
        for _ in 0..3 {
            let mut ids: Vec<u32> = (0..V as u32).collect();
            for i in (1..V).rev() {
                let j = (next() * (i + 1) as f64) as usize % (i + 1);
                ids.swap(i, j);
            }
            ids.truncate(C);
            let scores: Vec<f64> = (0..C).map(|_| next() * 4.0 - 2.0).collect();
            q_sets.push((ids, scores));
        }
        let softmax = |row: &[f64], t: f64| -> Vec<f64> {
            let top = row.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v / t));
            let w: Vec<f64> = row.iter().map(|&v| (v / t - top).exp()).collect();
            let total: f64 = w.iter().sum();
            w.iter().map(|x| x / total).collect()
        };
        let p_law = |i: usize, t: f64| {
            softmax(
                &p_rows[i].iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                t,
            )
        };
        let chi2_bound = |df: f64| {
            let z = 3.090_232_306;
            df * (1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt()).powi(3)
        };
        let mut law_check = |k: usize, t: f32, seed: u64| -> Result<(bool, f64, f64)> {
            let q_laws: Vec<Vec<f64>> = q_sets[..k]
                .iter()
                .map(|(_, s)| softmax(s, f64::from(t)))
                .collect();
            let mut cand = Vec::new();
            let mut q = Vec::new();
            for (i, (ids, _)) in q_sets[..k].iter().enumerate() {
                cand.extend(ids);
                q.extend(q_laws[i].iter().map(|&v| v as f32));
            }
            let cand = Array::from_u32(&cand, &[k as i32, C as i32])?;
            let q_arr = Array::from_f32(&q, &[k as i32, C as i32])?;
            let flat: Vec<f32> = p_rows[..=k].iter().flatten().copied().collect();
            let logits = Array::from_f32(&flat, &[1, k as i32 + 1, V as i32])?;
            let mut counts = vec![vec![0f64; V]; k + 1];
            let (mut first_kept, mut kept) = (0usize, 0usize);
            for batch in 0..BLOCKS / BATCH {
                let mut blocks = Vec::with_capacity(BATCH);
                for b in 0..BATCH {
                    let mut x = Vec::with_capacity(k);
                    let mut chosen = Vec::with_capacity(k);
                    for (i, law) in q_laws.iter().enumerate() {
                        let u = next();
                        let mut acc = 0.0;
                        let mut c = C - 1;
                        for (j, &w) in law.iter().enumerate() {
                            acc += w;
                            if u < acc {
                                c = j;
                                break;
                            }
                        }
                        x.push(q_sets[i].0[c]);
                        chosen.push(q[i * C + c]);
                    }
                    let proposals = Array::from_u32(&x, &[k as i32])?;
                    let chosen = Array::from_f32(&chosen, &[k as i32])?;
                    let block_seed = seed + 2 * (batch * BATCH + b) as u64;
                    let packet = decide_ratio(
                        &logits,
                        &proposals,
                        &cand,
                        &q_arr,
                        &chosen,
                        t,
                        block_seed,
                        block_seed + 1,
                    )?;
                    blocks.push((packet, x));
                }
                Array::eval_many(&blocks.iter().map(|(p, _)| p).collect::<Vec<_>>())?;
                for (packet, x) in &blocks {
                    let p = packet.to_u32()?;
                    ensure!(
                        p.len() == 2 + k
                            && p[2..] == x[..]
                            && (p[0] as usize) <= k
                            && (p[1] as usize) < V,
                        "invalid packet {p:?}"
                    );
                    let a = p[0] as usize;
                    kept += a;
                    first_kept += usize::from(a > 0);
                    for (i, row) in counts.iter_mut().enumerate().take(a + 1) {
                        let token = if i < a { x[i] } else { p[1] };
                        row[token as usize] += 1.0;
                    }
                }
            }
            let mut ok = true;
            for (i, row) in counts.iter().enumerate() {
                let law = p_law(i, f64::from(t));
                let reached: f64 = row.iter().sum();
                if reached < 2_000.0 {
                    continue;
                }
                let chi2: f64 = (0..V)
                    .map(|j| (row[j] - law[j] * reached).powi(2) / (law[j] * reached))
                    .sum();
                let bound = chi2_bound((V - 1) as f64);
                println!(
                    "S3 k={k} T={t} position {i}: reached {reached}, chi2 {chi2:.2} bound {bound:.2}"
                );
                ok &= chi2 < bound;
            }
            // expected share of first proposals kept: sum over the candidates of min(p_0, q_0)
            let p0 = p_law(0, f64::from(t));
            let expected: f64 = q_sets[0]
                .0
                .iter()
                .zip(&q_laws[0])
                .map(|(&id, &w)| p0[id as usize].min(w))
                .sum();
            let rate = first_kept as f64 / BLOCKS as f64;
            println!(
                "S3 k={k} T={t}: first proposal kept {rate:.4} (sum min(p, q) {expected:.4}), {:.3} of the proposals",
                kept as f64 / (BLOCKS * k) as f64
            );
            Ok((ok, rate, expected))
        };
        for (k, t, seed) in [
            (2usize, 1.0f32, 11u64),
            (3, 0.7, 2_000_011),
            (3, 1.3, 4_000_011),
        ] {
            let (ok, rate, expected) = law_check(k, t, seed)?;
            ensure!(
                ok,
                "ratio decisions do not follow the target law (k={k}, T={t})"
            );
            ensure!(
                (rate - expected).abs() < 0.01,
                "first proposal kept {rate} instead of {expected} (k={k}, T={t})"
            );
        }
        RATIO_MUTATION.with(|m| m.set(true));
        let mutated = law_check(3, 1.0, 9_000_011);
        RATIO_MUTATION.with(|m| m.set(false));
        let (ok, _, _) = mutated?;
        ensure!(
            !ok,
            "the mutation that keeps every proposal was not detected"
        );
        Ok(())
    }

    #[test]
    fn coupling_setting_is_off_unless_one() {
        assert!(coupling_setting(Some("1")));
        for value in [
            None,
            Some("0"),
            Some(""),
            Some("true"),
            Some("01"),
            Some(" 1"),
        ] {
            assert!(!coupling_setting(value), "{value:?}");
        }
    }

    /// DSPARK-2026-11-23-INK2M-MTPLX-A: GPU decisions are on unless the setting is exactly "0".
    #[test]
    fn device_rounds_setting_is_on_unless_zero() {
        assert!(device_rounds_setting(None));
        assert!(!device_rounds_setting(Some("0")));
        for value in ["1", "", "true", "false", "00", " 0"] {
            assert!(device_rounds_setting(Some(value)), "{value:?}");
        }
    }

    /// DSPARK-INK2M-P4: the width follows the acceptance (high acceptance: wide blocks; the measured
    /// sampled free-text acceptance 0.37 / 0.40 / 0.45: width 1; rejections: plain steps), probes at
    /// full width every `PROBE_EVERY` blocks, observes only the positions a block reached, and is a pure
    /// function of the accepted counts.
    #[test]
    fn width_policy_follows_acceptance_and_probes() {
        let prior = WidthPolicy::new(3);
        assert_eq!(prior.max_width(), 3);
        assert!((prior.expected_tokens(0) - 1.0).abs() < 1e-12);
        assert!((prior.expected_tokens(2) - 2.3125).abs() < 1e-12);
        assert!((prior.cost(3) - 1.81).abs() < 1e-12);
        assert_eq!(
            prior.best(),
            3,
            "prior 0.75: full width (1.51 tokens per step)"
        );
        assert_eq!(
            WidthPolicy::with_acceptance(vec![0.37, 0.40, 0.45]).best(),
            1
        );
        assert_eq!(WidthPolicy::with_acceptance(vec![0.75, 0.8, 0.8]).best(), 3);
        assert_eq!(WidthPolicy::with_acceptance(vec![0.1, 0.9, 0.9]).best(), 0);
        // every rejection at the first position drives the policy to plain steps
        let mut policy = WidthPolicy::new(3);
        for _ in 0..200 {
            let width = policy.next_width();
            policy.observe(width, 0);
        }
        // width 1 pays off only above 1 + a0 = cost(1) = 1.27; once below, only the probes update a0
        assert!(policy.acceptance()[0] < 0.15, "{:?}", policy.acceptance());
        assert_eq!(policy.best(), 0);
        // probes: the full width every PROBE_EVERY blocks, the best width otherwise
        let widths: Vec<usize> = (0..16).map(|_| policy.next_width()).collect();
        for (i, &w) in widths.iter().enumerate() {
            let probe = (policy.blocks - 16 + i + 1).is_multiple_of(WidthPolicy::PROBE_EVERY);
            assert_eq!(w, if probe { 3 } else { 0 }, "block {i}");
        }
        // only reached positions are updated
        let mut p = WidthPolicy::with_acceptance(vec![0.5, 0.5, 0.5]);
        p.observe(3, 1);
        assert!((p.acceptance()[0] - 0.575).abs() < 1e-12);
        assert!((p.acceptance()[1] - 0.425).abs() < 1e-12);
        assert!((p.acceptance()[2] - 0.5).abs() < 1e-12);
        p.observe(0, 0);
        p.observe(1, 1);
        assert!(
            (p.acceptance()[1] - 0.425).abs() < 1e-12,
            "width 1 never reaches position 1"
        );
        // same accepted counts, same widths
        let run = |seed: u64| {
            let mut p = WidthPolicy::new(3);
            let mut state = seed;
            (0..300)
                .map(|_| {
                    let w = p.next_width();
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let accepted = ((state >> 33) as usize % (w + 1)).min(w);
                    p.observe(w, accepted);
                    w
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(run(7), run(7));
    }

    /// DSPARK-INK2M-P10b/P10c: measured 8-bit costs and narrow, spaced probes. At the sampled free-text
    /// acceptance measured in P10a (0.45 / 0.4 / 0.35) the policy stays on plain steps and probes at width
    /// 1 after 8, 16, 32 then every 64 blocks; at the greedy code acceptance of P4b (0.84 / 0.82 / 0.85) it
    /// uses the full width; acceptance back after rejections resets the gap; costs are validated.
    /// DFLASH2-2026-11-25-INK2M-F2/F3 and INVARIANT-2026-11-25-INK2M-L3-3: the DFlash2 tables are valid measured
    /// tables; the default is the one with the expert panel, cheaper than the F3 means at every width; with the
    /// prior a block of at least 3 proposals is the best start; the earlier tables stay selectable.
    #[test]
    fn dflash2_costs_follow_the_bridge_means() {
        let costs = WidthPolicy::dflash2_costs(None);
        assert_eq!(costs, &WidthPolicy::DFLASH2_COSTS);
        assert_eq!(
            WidthPolicy::dflash2_costs(Some("")),
            &WidthPolicy::DFLASH2_COSTS
        );
        assert_eq!(
            WidthPolicy::dflash2_costs(Some("l3")),
            &WidthPolicy::DFLASH2_COSTS
        );
        assert_eq!(
            WidthPolicy::dflash2_costs(Some("p2")),
            &WidthPolicy::DFLASH2_COSTS_P2
        );
        assert_eq!(
            WidthPolicy::dflash2_costs(Some("f2")),
            &WidthPolicy::DFLASH2_COSTS_F2
        );
        assert_eq!(
            WidthPolicy::dflash2_costs(Some("f3")),
            &WidthPolicy::DFLASH2_COSTS_F3
        );
        for (w, &cost) in WidthPolicy::DFLASH2_COSTS_F2.iter().enumerate().skip(1) {
            assert!((cost - (1.40 + 0.375 * w as f64)).abs() < 1e-9, "width {w}");
        }
        for (w, &cost) in WidthPolicy::DFLASH2_COSTS_F3.iter().enumerate().skip(1) {
            // the means measured without the panel stay within 0.1 step of their fit 1.28 + 0.373 w
            assert!((cost - (1.28 + 0.373 * w as f64)).abs() < 0.1, "width {w}");
        }
        for (with, without) in costs.iter().zip(&WidthPolicy::DFLASH2_COSTS_F3).skip(1) {
            assert!(with < without, "the panel makes every block cheaper");
        }
        for table in [
            &WidthPolicy::DFLASH2_COSTS_P2,
            &WidthPolicy::DFLASH2_COSTS_F2,
            &WidthPolicy::DFLASH2_COSTS_F3,
        ] {
            assert!(WidthPolicy::measured(table, 7).is_ok());
        }
        let policy = WidthPolicy::measured(costs, 7).unwrap();
        assert!(policy.best() >= 3, "best start {}", policy.best());
        assert_eq!(
            WidthPolicy::measured(&WidthPolicy::DFLASH2_COSTS_F3, 7)
                .unwrap()
                .best(),
            3
        );
        let mut one = WidthPolicy::measured(costs, 1).unwrap();
        one.acceptance = vec![0.69];
        assert_eq!(one.best(), 1);
        one.acceptance = vec![0.67];
        assert_eq!(one.best(), 0);
    }

    /// DFLASH2-2026-11-25-INK2M-F3: from width 0, a probe whose proposal is kept is followed by another probe at
    /// the next block; a probe with a rejection doubles the gap as before; without the option a kept probe
    /// still doubles the gap while the best width stays 0 (P10c unchanged).
    #[test]
    fn probe_streak_follows_kept_probes() {
        assert!(WidthPolicy::probe_streak_setting(None));
        assert!(WidthPolicy::probe_streak_setting(Some("1")));
        assert!(!WidthPolicy::probe_streak_setting(Some("0")));
        let costs = WidthPolicy::dflash2_costs(None);
        let low = || {
            let mut p = WidthPolicy::measured(costs, 7).unwrap();
            p.acceptance = vec![0.1; 7];
            p
        };
        // the first probe comes after PROBE_EVERY blocks of the best width (0)
        let to_probe = |p: &mut WidthPolicy| {
            let mut plain = 0;
            loop {
                let w = p.next_width();
                if w > 0 {
                    return (plain, w);
                }
                p.observe(0, 0);
                plain += 1;
                assert!(plain < 200, "no probe");
            }
        };
        let mut streak = low().with_probe_streak();
        assert_eq!(to_probe(&mut streak), (WidthPolicy::PROBE_EVERY - 1, 1));
        streak.observe(1, 1);
        assert_eq!(streak.best(), 0);
        assert_eq!(streak.probe_gap(), 1);
        assert_eq!(
            streak.next_width(),
            1,
            "a kept probe is followed by a probe"
        );
        streak.observe(1, 0);
        // a rejection after the streak: back to the first gap, then doubled by the next rejected probe
        assert_eq!(streak.probe_gap(), WidthPolicy::PROBE_EVERY);
        assert_eq!(to_probe(&mut streak).0, WidthPolicy::PROBE_EVERY - 1);
        streak.observe(1, 0);
        assert_eq!(streak.probe_gap(), 2 * WidthPolicy::PROBE_EVERY);
        // kept probes in a row bring the estimate back: the best width leaves 0 and the gap returns to 8
        let mut climb = low().with_probe_streak();
        to_probe(&mut climb);
        let mut probes = 0;
        loop {
            climb.observe(1, 1);
            probes += 1;
            if climb.best() > 0 {
                break;
            }
            assert_eq!(climb.next_width(), 1);
            assert!(probes < 40, "the estimate does not recover");
        }
        assert!(probes <= 12, "{probes} kept probes to recover");
        // P10c without the option: the gap doubles after a kept probe while the best width stays 0
        let mut plain = low();
        assert_eq!(to_probe(&mut plain), (WidthPolicy::PROBE_EVERY - 1, 1));
        plain.observe(1, 1);
        assert_eq!(plain.probe_gap(), 2 * WidthPolicy::PROBE_EVERY);
        assert_eq!(plain.next_width(), 0);
    }

    #[test]
    fn measured_width_policy_probes_one_past_the_best_width() {
        let costs = WidthPolicy::SAMPLED_8BIT_COSTS;
        let mut p = WidthPolicy::measured(&costs, 3).unwrap();
        assert!((p.cost(1) - 1.56).abs() < 1e-12 && (p.cost(3) - 2.17).abs() < 1e-12);
        assert!(p.best() >= 2, "prior 0.75 still starts wide");
        p.acceptance = vec![0.45, 0.4, 0.35];
        assert_eq!(p.best(), 0, "1.45 tokens for 1.56 steps does not pay");
        let old = WidthPolicy::with_acceptance(vec![0.45, 0.4, 0.35]);
        assert_eq!(old.best(), 1, "the P4c linear costs pick width 1 here");
        p.acceptance = vec![0.84, 0.82, 0.85];
        assert_eq!(p.best(), 3);
        p.acceptance = vec![0.6, 0.3, 0.3];
        assert_eq!(
            p.best(),
            1,
            "1.6 tokens for 1.56 steps pays, the second proposal does not"
        );
        // probes: best + 1 (width 1 while plain steps are best), after 8, 16, 32 then every 64 blocks
        // while they are rejected
        let mut p = WidthPolicy::measured(&costs, 3).unwrap();
        p.acceptance = vec![0.1, 0.1, 0.1];
        let mut probes = Vec::new();
        for block in 1..=400 {
            let w = p.next_width();
            assert!(w <= 1, "block {block}: {w}");
            if w == 1 {
                probes.push(block);
            }
            p.observe(w, 0);
        }
        assert_eq!(&probes[..6], &[8, 24, 56, 120, 184, 248], "{probes:?}");
        assert_eq!(p.probe_gap(), WidthPolicy::PROBE_MAX);
        // an accepted probe that makes width 1 pay resets the gap
        p.acceptance = vec![0.55, 0.3, 0.3];
        while p.next_width() == 0 {}
        p.observe(1, 1);
        assert!(p.best() >= 1, "{:?}", p.acceptance());
        assert_eq!(p.probe_gap(), WidthPolicy::PROBE_EVERY);
        // the P4c setting keeps its fixed full-width probes
        let mut old = WidthPolicy::with_acceptance(vec![0.1, 0.1, 0.1]);
        let widths: Vec<usize> = (1..=32)
            .map(|_| {
                let w = old.next_width();
                old.observe(w, 0);
                w
            })
            .collect();
        for (i, &w) in widths.iter().enumerate() {
            let probe = (i + 1).is_multiple_of(WidthPolicy::PROBE_EVERY);
            assert_eq!(w, if probe { 3 } else { 0 }, "P4c block {i}");
        }
        // a probe at the full width stays at the full width
        let mut p = WidthPolicy::measured(&costs, 3).unwrap();
        p.acceptance = vec![0.9, 0.9, 0.9];
        assert!((0..16).all(|_| p.next_width() == 3));
        // every rejection: plain steps; repeated acceptance: back to the full width
        let mut p = WidthPolicy::measured(&costs, 3).unwrap();
        for _ in 0..200 {
            let w = p.next_width();
            p.observe(w, 0);
        }
        assert_eq!(p.best(), 0);
        let mut widths = Vec::new();
        for _ in 0..400 {
            let w = p.next_width();
            widths.push(w);
            p.observe(w, w);
        }
        assert_eq!(*widths.last().unwrap(), 3, "{:?}", p.acceptance());
        // invalid costs
        assert!(WidthPolicy::measured(&costs, 0).is_err());
        assert!(WidthPolicy::measured(&costs, 4).is_err());
        assert!(WidthPolicy::measured(&[1.1, 1.5], 1).is_err());
        assert!(WidthPolicy::measured(&[1.0, 0.9], 1).is_err());
        assert!(WidthPolicy::measured(&[1.0, f64::NAN], 1).is_err());
        assert_eq!(WidthPolicy::measured(&costs, 2).unwrap().max_width(), 2);
    }

    /// SPEED-INK2M-P14: a session whose width-0 blocks draw on the GPU (synchronous or one step ahead)
    /// delivers the host session's greedy tokens, blocks and widths; at T = 1 the pipelined and synchronous
    /// device sessions agree for one seed stream and one acceptance state, and (where width 0 dominates)
    /// another seed changes the tokens. Two policies: adaptive from the prior, and costs that keep width 0
    /// between probes.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ORACLE"]
    fn device_draw_session_matches_host_session() -> Result<()> {
        use crate::inkling2::{DrawRule, DrawSeeds};
        use crate::sampling::{Draw, History, Sampler};
        const TOKENS: usize = 128;
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let prompt = prompt()?;
        let mut target = Inkling2::load(&dir)?;
        let mut draft = Inkling2Draft::load(&dir, &target)?;
        let limit = usize::try_from(target.context_limit())?;
        type Run = (Vec<u32>, usize, usize, Vec<usize>);
        // DSPARK-2026-11-23-INK2M-MTPLX-A: `device` = (pipelined, seed, blocks with proposals decided on the GPU,
        // DSPARK-2026-11-24-INK2M-MAYA2-D2 coupled sampled blocks)
        let mut run = |sampler: Sampler,
                       adaptive: bool,
                       device: Option<(bool, u64, bool, bool)>|
         -> Result<Run> {
            let policy = if adaptive {
                WidthPolicy::new(DsparkSession::DEFAULT_WIDTH)
            } else {
                WidthPolicy::measured(&[1.0, 5.0, 9.0, 13.0], DsparkSession::DEFAULT_WIDTH)?
            };
            let mut session = DsparkSession::new(&draft, DsparkSession::DEFAULT_WIDTH)?
                .with_adaptive_width(policy)?;
            if let Some((pipelined, seed, rounds, couple)) = device {
                let rule = DrawRule::for_sampler(&sampler).context("no GPU rule")?;
                session = session
                    .with_device_draw(rule, DrawSeeds::new(seed), pipelined)
                    .with_device_rounds(rounds)
                    .with_coupling(couple);
                ensure!(
                    session.device_rounds() == rounds,
                    "device rounds not applied"
                );
                ensure!(
                    session.couples() == (rounds && couple && !sampler.is_greedy()),
                    "coupling not applied"
                );
            }
            let logits = session.prefill(&mut target, &mut draft, &prompt, &|| false)?;
            let history = History::default();
            let mut random = 0x1234_5678_9abc_def1u64;
            let mut next = if sampler.is_greedy() {
                logits.chat_greedy_ids()?[0]
            } else {
                Draw {
                    sampler,
                    history: &history,
                    random: &mut random,
                }
                .sample_last_row(&logits)?
            };
            let mut out = vec![next];
            while out.len() < TOKENS {
                let mut choose = |proposals: &[u32],
                                  _: Option<&[Vec<f32>]>,
                                  verified: &Array|
                 -> Result<(usize, u32)> {
                    let acceptance = if sampler.is_greedy() {
                        crate::speculative::greedy_accept(proposals, &verified.chat_greedy_ids()?)
                            .context("invalid greedy acceptance")?
                    } else {
                        Draw {
                            sampler,
                            history: &history,
                            random: &mut random,
                        }
                        .speculative_accept_logits(proposals, verified)?
                    };
                    Ok((acceptance.accepted_draft_tokens, acceptance.target_token))
                };
                next = session.advance(
                    &mut target,
                    &mut draft,
                    next,
                    limit,
                    TOKENS - out.len(),
                    &mut choose,
                )?;
                out.push(next);
            }
            let s = &session.stats;
            ensure!(
                (s.coupled_blocks > 0) == session.couples(),
                "coupled blocks counted without the coupled rule, or the reverse"
            );
            Ok((out, s.blocks, s.accepted, s.width_blocks.clone()))
        };
        let greedy = Sampler::greedy();
        let sampled = Sampler::new(1.0, 0, 1.0, 1.0, 0.0)?;
        for adaptive in [true, false] {
            let host = run(greedy, adaptive, None)?;
            println!(
                "greedy adaptive={adaptive}: blocks {} accepted {} widths {:?}",
                host.1, host.2, host.3
            );
            ensure!(
                host.3[0] > 0 && host.3[1..].iter().sum::<usize>() > 0,
                "both block kinds must run"
            );
            for rounds in [false, true] {
                // greedy: coupling never applies
                let sync = run(greedy, adaptive, Some((false, 9, rounds, true)))?;
                let ahead = run(greedy, adaptive, Some((true, 9, rounds, true)))?;
                ensure!(
                    sync == host,
                    "greedy device session (synchronous, GPU rounds {rounds}) differs from the host session"
                );
                ensure!(
                    ahead == host,
                    "greedy device session (one step ahead, GPU rounds {rounds}) differs from the host session"
                );
                // D2: the point rule (coupling off) keeps its synchronous = one step ahead equality
                let point_sync = run(sampled, adaptive, Some((false, 9, rounds, false)))?;
                let point_ahead = run(sampled, adaptive, Some((true, 9, rounds, false)))?;
                ensure!(
                    point_ahead == point_sync,
                    "T = 1 point-rule sessions (GPU rounds {rounds}) differ between synchronous and one step ahead"
                );
                let sync = run(sampled, adaptive, Some((false, 9, rounds, true)))?;
                let ahead = run(sampled, adaptive, Some((true, 9, rounds, true)))?;
                println!(
                    "T=1 adaptive={adaptive} rounds={rounds}: blocks {} accepted {} widths {:?} (point rule: blocks {} accepted {} widths {:?})",
                    sync.1, sync.2, sync.3, point_sync.1, point_sync.2, point_sync.3
                );
                ensure!(
                    ahead == sync,
                    "T = 1 device sessions (GPU rounds {rounds}) differ between synchronous and one step ahead"
                );
                // both block kinds compared: with the coupled rule the adaptive policy may keep every block wide
                // (acceptance is higher), so width 0 is required of the point rule there and of every run where
                // the measured costs keep width 0 between probes
                ensure!(
                    point_sync.3[0] > 0,
                    "no width-0 block at T = 1 (point rule)"
                );
                ensure!(adaptive || sync.3[0] > 0, "no width-0 block at T = 1");
                ensure!(
                    sync.3[1..].iter().sum::<usize>() > 0
                        && point_sync.3[1..].iter().sum::<usize>() > 0,
                    "no block with proposals ran at T = 1"
                );
                // where width 0 dominates (between probes), the GPU draw picks most tokens: another seed must
                // change them (from the prior, a few width-0 blocks may all draw the same confident token)
                if !adaptive {
                    ensure!(sync.3[0] * 2 > sync.1, "width 0 does not dominate");
                    ensure!(
                        run(sampled, adaptive, Some((true, 10, rounds, true)))?.0 != sync.0,
                        "the seed does not change the tokens"
                    );
                }
            }
        }
        Ok(())
    }

    /// DSPARK-2026-11-23-INK2M-MTPLX-C0 (diagnostic): ms per token of plain GPU-drawn decoding (`DeviceDecode`,
    /// one step ahead) against the width-0 blocks of a DSpark session whose costs keep it at width 0 between
    /// probes (the drafter's context is still fed every step), T = 1, 160 tokens after the prompt, alternated
    /// A B B A over 3 rounds. Prints the medians.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ORACLE"]
    fn width_zero_tax_profile() -> Result<()> {
        use crate::inkling2::{DeviceDecode, DrawRule, DrawSeeds};
        use crate::sampling::{Draw, History, Sampler};
        const TOKENS: usize = 160;
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let prompt = prompt()?;
        let mut target = Inkling2::load(&dir)?;
        let mut draft = Inkling2Draft::load(&dir, &target)?;
        let limit = usize::try_from(target.context_limit())?;
        let sampler = Sampler::new(1.0, 0, 1.0, 1.0, 0.0)?;
        let rule = DrawRule::for_sampler(&sampler).context("no GPU rule")?;
        let mut plain = Vec::new();
        let mut zero = Vec::new();
        for round in 0..3u64 {
            for dspark in [false, true, true, false] {
                let seed = 100 + round;
                if dspark {
                    let mut session = DsparkSession::new(&draft, DsparkSession::DEFAULT_WIDTH)?
                        .with_adaptive_width(WidthPolicy::measured(
                            &[1.0, 1000.0, 1001.0, 1002.0],
                            DsparkSession::DEFAULT_WIDTH,
                        )?)?
                        .with_device_draw(rule, DrawSeeds::new(seed), true);
                    let logits = session.prefill(&mut target, &mut draft, &prompt, &|| false)?;
                    let history = History::default();
                    let mut random = seed;
                    let mut next = Draw {
                        sampler,
                        history: &history,
                        random: &mut random,
                    }
                    .sample_last_row(&logits)?;
                    for _ in 0..TOKENS {
                        let mut choose = |proposals: &[u32],
                                          _: Option<&[Vec<f32>]>,
                                          verified: &Array|
                         -> Result<(usize, u32)> {
                            let a = Draw {
                                sampler,
                                history: &history,
                                random: &mut random,
                            }
                            .speculative_accept_logits(proposals, verified)?;
                            Ok((a.accepted_draft_tokens, a.target_token))
                        };
                        next = session.advance(
                            &mut target,
                            &mut draft,
                            next,
                            limit,
                            1 << 20,
                            &mut choose,
                        )?;
                    }
                    let s = &session.stats;
                    // skip the first 8 width-0 blocks (warm-up): use the mean over the remaining ones
                    ensure!(s.width_blocks[0] > 16, "too few width-0 blocks");
                    zero.push(s.width_ms[0] / s.width_blocks[0] as f64);
                } else {
                    target.reset();
                    let logits = target.forward_tokens(&prompt)?;
                    let mut decode =
                        DeviceDecode::start(&logits, rule, DrawSeeds::new(seed), true)?;
                    for _ in 0..8 {
                        decode.next_token(&mut target, true)?;
                    }
                    let start = Instant::now();
                    for _ in 0..TOKENS {
                        decode.next_token(&mut target, true)?;
                    }
                    plain.push(start.elapsed().as_secs_f64() * 1e3 / TOKENS as f64);
                }
            }
        }
        let median = |mut v: Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v[v.len() / 2]
        };
        println!(
            "WIDTH_ZERO_TAX {}",
            serde_json::json!({"plain_ms_per_token": plain.clone(), "width0_ms_per_block": zero.clone(),
                "plain_median": median(plain), "width0_median": median(zero)})
        );
        Ok(())
    }

    /// DSPARK-2026-11-23-INK2M-MTPLX-A3: wall time of a DSpark block at a fixed width (1, then 3), T = 1,
    /// with the CPU decision, the GPU decision (A) and the GPU decision with the drafter overlapped (A3),
    /// alternated in one process (order A B C C B A per round, `MLXL3_INKLING_PROFILE_ROUNDS` rounds, 3 by
    /// default). Each measurement prefills the prompt again, runs 8 warm-up blocks, then times 64 blocks.
    /// Prints one JSON line: median ms per block per variant and width, and the host split of the GPU
    /// blocks.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ORACLE"]
    fn device_round_profile() -> Result<()> {
        use crate::inkling2::{DrawRule, DrawSeeds};
        use crate::sampling::{Draw, History, Sampler};
        const WARM: usize = 8;
        const BLOCKS: usize = 64;
        let rounds: usize = std::env::var("MLXL3_INKLING_PROFILE_ROUNDS")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(3);
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let prompt = prompt()?;
        let mut target = Inkling2::load(&dir)?;
        let mut draft = Inkling2Draft::load(&dir, &target)?;
        let limit = usize::try_from(target.context_limit())?;
        let sampler = Sampler::new(1.0, 0, 1.0, 1.0, 0.0)?;
        let rule = DrawRule::for_sampler(&sampler).context("no GPU rule")?;
        // (ms per block, tokens per block, GPU split per block)
        type Sample = (f64, f64, [f64; 5]);
        let mut measure = |width: usize,
                           variant: &str,
                           seed: u64|
         -> Result<(f64, f64, [f64; 5])> {
            let mut session = DsparkSession::new(&draft, width)?
                .with_device_draw(rule, DrawSeeds::new(seed), true)
                .with_device_rounds(variant != "cpu")
                .with_overlap(variant == "overlap");
            let logits = session.prefill(&mut target, &mut draft, &prompt, &|| false)?;
            let history = History::default();
            let mut random = seed ^ 0x9e37_79b9_7f4a_7c15;
            let mut next = Draw {
                sampler,
                history: &history,
                random: &mut random,
            }
            .sample_last_row(&logits)?;
            let mut started = None;
            let (mut blocks0, mut tokens, mut split0) = (0, 0usize, DeviceRoundTimes::default());
            loop {
                let blocks = session.stats.blocks;
                if blocks == WARM && started.is_none() {
                    started = Some(Instant::now());
                    blocks0 = blocks;
                    split0 = session.stats.device.clone();
                    tokens = 0;
                }
                if blocks >= WARM + BLOCKS && session.pending.is_empty() {
                    break;
                }
                let mut choose = |proposals: &[u32],
                                  _: Option<&[Vec<f32>]>,
                                  verified: &Array|
                 -> Result<(usize, u32)> {
                    let acceptance = Draw {
                        sampler,
                        history: &history,
                        random: &mut random,
                    }
                    .speculative_accept_logits(proposals, verified)?;
                    Ok((acceptance.accepted_draft_tokens, acceptance.target_token))
                };
                next =
                    session.advance(&mut target, &mut draft, next, limit, 1 << 20, &mut choose)?;
                tokens += 1;
            }
            let elapsed = started.context("no block ran")?.elapsed().as_secs_f64() * 1e3;
            let n = (session.stats.blocks - blocks0) as f64;
            let d = &session.stats.device;
            let per = |a: f64, b: f64| if variant == "cpu" { 0.0 } else { (a - b) / n };
            Ok((
                elapsed / n,
                tokens as f64 / n,
                [
                    per(d.draft_build_ms, split0.draft_build_ms),
                    per(d.verify_build_ms, split0.verify_build_ms),
                    per(d.decide_build_ms, split0.decide_build_ms),
                    per(d.wait_ms, split0.wait_ms),
                    per(d.finish_ms, split0.finish_ms),
                ],
            ))
        };
        let variants = ["cpu", "gpu", "overlap"];
        let mut report = serde_json::Map::new();
        for width in [1usize, 3] {
            let mut samples: std::collections::BTreeMap<&str, Vec<Sample>> = Default::default();
            for round in 0..rounds {
                for (i, variant) in variants.iter().chain(variants.iter().rev()).enumerate() {
                    let seed = 1_000 * round as u64 + i as u64 + 17;
                    let result = measure(width, variant, seed)?;
                    println!(
                        "width {width} round {round} {variant}: {:.2} ms/block, {:.2} tokens/block, split {:?}",
                        result.0, result.1, result.2
                    );
                    samples.entry(variant).or_default().push(result);
                }
            }
            let median = |mut v: Vec<f64>| {
                v.sort_by(f64::total_cmp);
                v[v.len() / 2]
            };
            let mut entry = serde_json::Map::new();
            for (variant, values) in samples {
                let ms: Vec<f64> = values.iter().map(|v| v.0).collect();
                entry.insert(
                    variant.to_owned(),
                    serde_json::json!({
                        "ms_per_block_median": median(ms.clone()),
                        "ms_per_block": ms,
                        "tokens_per_block_median": median(values.iter().map(|v| v.1).collect()),
                        "split_ms_median": (0..5).map(|k| median(values.iter().map(|v| v.2[k]).collect())).collect::<Vec<_>>(),
                    }),
                );
            }
            report.insert(format!("width_{width}"), serde_json::Value::Object(entry));
        }
        println!("DEVICE_ROUND_PROFILE {}", serde_json::Value::Object(report));
        Ok(())
    }

    /// DSPARK-INK2M-P3: the GPU Gumbel draw follows softmax(logits / T): 40 000 independent rows per
    /// case against the exact probabilities (Pearson chi-square, 5 degrees of freedom, p < 1e-4 bound
    /// 25.7), and a fixed seed gives a fixed draw.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn gumbel_draw_follows_the_tempered_softmax() -> Result<()> {
        const ROWS: i32 = 40_000;
        let logits = [2.0f32, 1.0, 0.5, 0.0, -1.0, -3.0];
        let vocab = logits.len() as i32;
        for (temperature, seed) in [(1.0f32, 11u64), (0.7, 12), (1.6, 13)] {
            let row: Vec<f32> = (0..ROWS).flat_map(|_| logits).collect();
            let batch = Array::from_f32(&row, &[ROWS, vocab])?;
            let draws = gumbel_draw(&batch, temperature, seed)?
                .reshape(&[-1])?
                .to_u32()?;
            ensure!(draws.len() == ROWS as usize, "one draw per row");
            let mut counts = vec![0f64; logits.len()];
            for &token in &draws {
                ensure!(
                    (token as usize) < logits.len(),
                    "draw outside the vocabulary"
                );
                counts[token as usize] += 1.0;
            }
            let weights: Vec<f64> = logits
                .iter()
                .map(|&v| (f64::from(v) / f64::from(temperature)).exp())
                .collect();
            let total: f64 = weights.iter().sum();
            let chi2: f64 = weights
                .iter()
                .zip(&counts)
                .map(|(w, c)| {
                    let expected = w / total * f64::from(ROWS);
                    (c - expected).powi(2) / expected
                })
                .sum();
            println!("T={temperature} counts={counts:?} chi2={chi2:.2}");
            ensure!(
                chi2 < 25.7,
                "Gumbel draws do not follow softmax(logits / T): chi2 {chi2}"
            );
        }
        let single = Array::from_f32(&logits, &[1, vocab])?;
        let first = gumbel_draw(&single, 1.0, 99)?.to_u32()?;
        ensure!(
            first == gumbel_draw(&single, 1.0, 99)?.to_u32()?,
            "same seed, different draw"
        );
        let distinct: std::collections::HashSet<u32> = (0..64)
            .map(|s| Ok(gumbel_draw(&single, 1.0, s)?.to_u32()?[0]))
            .collect::<Result<_>>()?;
        ensure!(distinct.len() > 1, "the seed does not change the draw");
        Ok(())
    }

    /// DSPARK-2026-11-23-INK2M-MTPLX-A: the GPU decision of a block follows the exact speculative law of the
    /// point-mass rule. Two proposals over a 6-token vocabulary, 40 000 decisions per case (one seed pair
    /// each) against the joint law of (accepted, token) computed in f64: P(0, t) = p0(t) for t != x0,
    /// P(1, t) = p0(x0) p1(t) for t != x1, P(2, t) = p0(x0) p1(x1) p2(t), with p_i = softmax(log_softmax(row
    /// i) / T). Pearson chi-square below the 0.999 quantile (Wilson-Hilferty), no draw in a cell of zero
    /// mass; a proposal of zero mass is never accepted.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn decide_block_follows_the_speculative_law() -> Result<()> {
        use crate::inkling2::{DrawRule, decide_block};
        const DRAWS: usize = 40_000;
        const BATCH: usize = 1_000;
        let rows: [[f32; 6]; 3] = [
            [2.0, 1.0, 0.5, 0.0, -1.0, -3.0],
            [-0.5, 1.5, 0.2, 1.0, -2.0, 0.0],
            [0.3, -1.0, 2.5, 0.0, 1.0, -0.5],
        ];
        let cases: [(f32, [u32; 2], Option<f32>, u64); 4] = [
            (1.0, [1, 3], None, 1_000),
            (0.7, [0, 1], None, 2_000_000),
            (1.3, [2, 5], None, 3_000_000),
            // proposal 0 of zero mass: always rejected, the token follows p0 without it
            (1.0, [4, 1], Some(-1.0e4), 4_000_000),
        ];
        for (temperature, proposals, zero, seed) in cases {
            let mut table = rows;
            if let Some(value) = zero {
                table[0][proposals[0] as usize] = value;
            }
            let flat: Vec<f32> = table.iter().flatten().copied().collect();
            let logits = Array::from_f32(&flat, &[1, 3, 6])?;
            let ids = Array::from_u32(&proposals, &[2])?;
            let law: Vec<Vec<f64>> = table
                .iter()
                .map(|row| {
                    let max = row
                        .iter()
                        .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
                    let lse = max
                        + row
                            .iter()
                            .map(|&v| (f64::from(v) - max).exp())
                            .sum::<f64>()
                            .ln();
                    let scaled: Vec<f64> = row
                        .iter()
                        .map(|&v| (f64::from(v) - lse) / f64::from(temperature))
                        .collect();
                    let top = scaled.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
                    let weights: Vec<f64> = scaled.iter().map(|&v| (v - top).exp()).collect();
                    let total: f64 = weights.iter().sum();
                    weights.iter().map(|w| w / total).collect()
                })
                .collect();
            let (x0, x1) = (proposals[0] as usize, proposals[1] as usize);
            let mut expected = [[0f64; 6]; 3];
            for t in 0..6 {
                if t != x0 {
                    expected[0][t] = law[0][t];
                }
                if t != x1 {
                    expected[1][t] = law[0][x0] * law[1][t];
                }
                expected[2][t] = law[0][x0] * law[1][x1] * law[2][t];
            }
            let mut counts = [[0f64; 6]; 3];
            for batch in 0..DRAWS / BATCH {
                let packets = (0..BATCH)
                    .map(|i| {
                        let s = seed + 2 * (batch * BATCH + i) as u64;
                        decide_block(&logits, &ids, DrawRule::Softmax(temperature), s, s + 1)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Array::eval_many(&packets.iter().collect::<Vec<_>>())?;
                for packet in &packets {
                    let p = packet.to_u32()?;
                    ensure!(
                        p.len() == 4 && p[2..] == proposals && p[0] <= 2 && p[1] < 6,
                        "invalid packet {p:?}"
                    );
                    ensure!(
                        p[0] == 2 || p[1] != proposals[p[0] as usize],
                        "a rejected proposal was emitted: {p:?}"
                    );
                    counts[p[0] as usize][p[1] as usize] += 1.0;
                }
            }
            let mut chi2 = 0.0;
            let mut cells = 0usize;
            for (row, (e, c)) in expected.iter().zip(&counts).enumerate() {
                for t in 0..6 {
                    let mean = e[t] * DRAWS as f64;
                    if mean < 1e-9 {
                        ensure!(
                            c[t] == 0.0,
                            "draw in a zero-mass cell ({row}, {t}): {}",
                            c[t]
                        );
                        continue;
                    }
                    cells += 1;
                    chi2 += (c[t] - mean).powi(2) / mean;
                }
            }
            let df = (cells - 1) as f64;
            let z = 3.090_232_306;
            let bound = df * (1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt()).powi(3);
            println!(
                "T={temperature} proposals={proposals:?} zero={zero:?}: accepted {:?}, chi2 {chi2:.2} df {df} bound {bound:.2}",
                counts
                    .iter()
                    .map(|r| r.iter().sum::<f64>())
                    .collect::<Vec<_>>()
            );
            ensure!(
                chi2 < bound,
                "GPU decisions do not follow the speculative law: chi2 {chi2} >= {bound}"
            );
        }
        Ok(())
    }

    /// DSPARK-2026-11-23-INK2M-MTPLX-A: the greedy GPU decision is the bridge's rule (`greedy_accept` over
    /// `chat_greedy_ids` of the same rows) on random rows of the real vocabulary width, every acceptance
    /// count included, and the softmax decision rejects a proposal of zero mass and keeps one of full mass.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn decide_block_greedy_is_the_bridge_rule() -> Result<()> {
        use crate::inkling2::{DrawRule, decide_block};
        const VOCAB: i32 = 200_058;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32
        };
        let mut seen = [0usize; 4];
        for case in 0..64u64 {
            let rows = 4;
            let values: Vec<f32> = (0..rows * VOCAB).map(|_| 30.0 * uniform() - 15.0).collect();
            let logits = Array::from_f32(&values, &[1, rows, VOCAB])?;
            let argmax = logits.chat_greedy_ids()?;
            // proposals that match the argmax for a prefix of `case % 4` rows, then differ
            let keep = (case % 4) as usize;
            let proposals: Vec<u32> = (0..3)
                .map(|i| {
                    if i < keep {
                        argmax[i]
                    } else {
                        (argmax[i] + 1 + case as u32) % VOCAB as u32
                    }
                })
                .collect();
            let expected = crate::speculative::greedy_accept(&proposals, &argmax)
                .context("greedy acceptance")?;
            let packet = decide_block(
                &logits,
                &Array::from_u32(&proposals, &[3])?,
                DrawRule::Greedy,
                case,
                case + 1,
            )?
            .to_u32()?;
            ensure!(
                packet[0] as usize == expected.accepted_draft_tokens
                    && packet[1] == expected.target_token
                    && packet[2..] == proposals[..],
                "case {case}: GPU {packet:?}, bridge {expected:?}"
            );
            seen[expected.accepted_draft_tokens] += 1;
        }
        ensure!(seen.iter().all(|&n| n > 0), "acceptance counts {seen:?}");
        // full-vocabulary softmax: a proposal holding all the mass is kept, one with none is rejected
        let mut values = vec![-30.0f32; 3 * VOCAB as usize];
        values[7] = 30.0;
        values[VOCAB as usize + 9] = 30.0;
        values[2 * VOCAB as usize + 11] = 30.0;
        let logits = Array::from_f32(&values, &[1, 3, VOCAB])?;
        for seed in 0..32u64 {
            let full = decide_block(
                &logits,
                &Array::from_u32(&[7, 9], &[2])?,
                DrawRule::Softmax(1.0),
                seed,
                seed + 100,
            )?
            .to_u32()?;
            ensure!(full[..2] == [2, 11], "full-mass proposals: {full:?}");
            let none = decide_block(
                &logits,
                &Array::from_u32(&[8, 9], &[2])?,
                DrawRule::Softmax(1.0),
                seed,
                seed + 100,
            )?
            .to_u32()?;
            ensure!(none[0] == 0 && none[1] == 7, "zero-mass proposal: {none:?}");
        }
        Ok(())
    }

    /// Dump draft inputs/outputs for the independent MLX Python oracle (`dspark_oracle.py`).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_INKLING_ORACLE and MLXL3_INKLING_DUMP"]
    fn dump_draft_case() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let out = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_DUMP")?);
        std::fs::create_dir_all(&out)?;
        let prompt = prompt()?;
        let mut target = Inkling2::load_with(
            &dir,
            crate::inkling2::Precision::from_env_or(crate::inkling2::Precision::FULL)?,
        )?;
        let mut draft = Inkling2Draft::load(&dir, &target)?;
        let taps = draft.taps().to_vec();
        target.reset();
        draft.reset();
        let (logits, features) = target.verify(&prompt, &taps)?;
        target.commit(prompt.len())?;
        draft.append_context(&features)?;
        let rows = prompt.len() as i32;
        let anchor = logits
            .reshape(&[rows, -1])?
            .slice(0, rows - 1, rows)?
            .argmax()?
            .reshape(&[-1])?;
        let last = features.slice(0, rows - 1, rows)?;
        let count = draft.max_proposals();
        let (tokens, x, base) = draft.propose_debug(&target, &anchor, &last, count)?;
        write_f32(&out.join("features.f32"), &features)?;
        write_f32(&out.join("x.f32"), &x)?;
        write_f32(&out.join("base.f32"), &base)?;
        let meta = serde_json::json!({"prompt": prompt, "anchor": anchor.to_u32()?[0], "proposals": tokens.to_u32()?,
            "rows": rows, "count": count, "hidden": target.hidden_size(), "taps": taps,
            "vocab": target.unpadded_vocab()});
        std::fs::write(out.join("meta.json"), serde_json::to_string_pretty(&meta)?)?;
        println!("{meta}");
        Ok(())
    }

    fn fnv_f32(hash: &mut u64, values: &[f32]) {
        for v in values {
            for b in v.to_bits().to_le_bytes() {
                *hash ^= u64::from(b);
                *hash = hash.wrapping_mul(0x100_0000_01b3);
            }
        }
    }

    /// Real token rows of `MLXL3_INKLING_ROWS` (`rows.json` of the local reference), concatenated.
    fn text_rows() -> Result<Vec<u32>> {
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let mut tokens = Vec::new();
        for row in rows["rows"].as_array().context("rows")? {
            let row: Vec<u32> = serde_json::from_value(row["tokens"].clone())?;
            tokens.extend(row);
        }
        ensure!(!tokens.is_empty(), "no tokens in MLXL3_INKLING_ROWS");
        Ok(tokens)
    }

    /// Fingerprint of the drafter's context caches (every key and value, and the offsets).
    fn context_hash(draft: &Inkling2Draft) -> Result<u64> {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for cache in &draft.caches {
            for a in [&cache.keys, &cache.values].into_iter().flatten() {
                fnv_f32(&mut hash, &a.astype(Dtype::Float32)?.to_f32()?);
            }
            fnv_f32(&mut hash, &[cache.offset as f32]);
        }
        Ok(hash)
    }

    /// One prefill and its greedy DSpark decode at the fixed width 3, CPU choice: everything that depends
    /// on the state (last logits, drafter context, last features, then every verified row and token).
    #[derive(Debug, PartialEq)]
    struct D1Run {
        cached: usize,
        logits: u64,
        context: u64,
        last_feature: u64,
        decode: u64,
        tokens: Vec<u32>,
    }

    #[allow(clippy::too_many_arguments)]
    fn d1_run(
        session: &mut DsparkSession,
        target: &mut Inkling2,
        draft: &mut Inkling2Draft,
        prompt: &[u32],
        resume: Option<(DsparkPrefix, Array)>,
        checkpoint: Option<usize>,
        saved: &mut Option<(DsparkPrefix, Array)>,
        steps: usize,
    ) -> Result<D1Run> {
        let limit = usize::try_from(target.context_limit())?;
        let (logits, cached) = session.prefill_resumable(
            target,
            draft,
            prompt,
            &[],
            resume,
            checkpoint,
            &mut |prefix, logits| {
                *saved = Some((prefix, logits.try_clone()?));
                Ok(())
            },
            &|| false,
            &mut |_| Ok(()),
        )?;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        fnv_f32(&mut h, &logits.to_f32()?);
        let logits_hash = h;
        let context = context_hash(draft)?;
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        fnv_f32(
            &mut h,
            &session
                .last_feature
                .as_ref()
                .context("no last feature")?
                .to_f32()?,
        );
        let last_feature = h;
        let mut decode = 0xcbf2_9ce4_8422_2325u64;
        let mut next = logits.chat_greedy_ids()?[0];
        let mut tokens = vec![next];
        while tokens.len() < steps {
            let mut choose = |proposals: &[u32],
                              _: Option<&[Vec<f32>]>,
                              verified: &Array|
             -> Result<(usize, u32)> {
                fnv_f32(&mut decode, &verified.to_f32()?);
                let a = crate::speculative::greedy_accept(proposals, &verified.chat_greedy_ids()?)
                    .context("invalid greedy acceptance")?;
                Ok((a.accepted_draft_tokens, a.target_token))
            };
            next = session.advance(
                target,
                draft,
                next,
                limit,
                steps - tokens.len(),
                &mut choose,
            )?;
            tokens.push(next);
        }
        Ok(D1Run {
            cached,
            logits: logits_hash,
            context,
            last_feature,
            decode,
            tokens,
        })
    }

    /// SPEED-2026-11-25-INK2M-PREFILL-G: a DSpark session whose prefill computes the experts of 2, 4 or 8
    /// chunks together gives the chunk-by-chunk session's last logits, drafter context and last features to the
    /// bit, then the same verified rows and tokens over a greedy DSpark decode (prompts without and with a
    /// tail, before and past the 1 024-row windows).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn dspark_grouped_prefill_matches_chunks() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let mut target = Inkling2::load(&dir)?;
        let mut draft = Inkling2Draft::load(&dir, &target)?;
        let vocab = target.unpadded_vocab() as u32;
        let text: Vec<u32> = text_rows()?.into_iter().map(|t| t % vocab).collect();
        const STEPS: usize = 48;
        let mut compared = 0usize;
        for len in [300usize, 1024, 1100, 1300] {
            ensure!(text.len() >= len, "not enough tokens");
            let mut run = |group: usize| -> Result<D1Run> {
                crate::inkling2::with_moe_group(group, || {
                    let mut session = DsparkSession::new(&draft, DsparkSession::DEFAULT_WIDTH)?;
                    let mut saved = None;
                    d1_run(
                        &mut session,
                        &mut target,
                        &mut draft,
                        &text[..len],
                        None,
                        None,
                        &mut saved,
                        STEPS,
                    )
                })
            };
            let reference = run(1)?;
            ensure!(
                reference.cached == 0 && reference.tokens.len() == STEPS,
                "invalid reference at {len} tokens"
            );
            for group in [2usize, 4, 8] {
                let grouped = run(group)?;
                println!(
                    "len {len} group {group}: identical {}",
                    grouped == reference
                );
                ensure!(
                    grouped == reference,
                    "group {group} at {len} tokens: {grouped:?} differs from {reference:?}"
                );
                compared += 1;
            }
        }
        ensure!(compared == 12, "layouts compared");
        Ok(())
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D1: a DSpark prompt resumed from the previous turn's prefix (after that
    /// turn decoded with DSpark) gives a cold DSpark prefill's last logits, drafter context and last features to
    /// the bit, and then the same verified rows and tokens over a greedy decode, before and past the 1 024-row
    /// windows; a prompt equal to the saved prefix resumes everything (saved logits and last features). The two
    /// mutations (live drafter context, the session's own last features) must differ, a prefix of another lineage
    /// must be refused (cold prefill), and the prefix must fit the default prompt cache budget.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn dspark_prompt_cache_resume_matches_cold_prefill() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let mut target = Inkling2::load(&dir)?;
        let mut draft = Inkling2Draft::load(&dir, &target)?;
        let vocab = target.unpadded_vocab() as u32;
        let text: Vec<u32> = text_rows()?.into_iter().map(|t| t % vocab).collect();
        const ANSWER: usize = 40;
        const STEPS: usize = 48;
        // (first prompt length, extra tokens after the answer; 0 = the next prompt is the saved prefix itself)
        for (first, extra) in [(300usize, 199usize), (1300, 199), (256, 0)] {
            ensure!(text.len() >= first + extra + 2, "not enough tokens");
            let boundary = first / 128 * 128;
            // turn 1: prefill with the capture, then a DSpark decode; returns the prefix and the next prompt
            let turn1 = |target: &mut Inkling2,
                         draft: &mut Inkling2Draft,
                         session: &mut DsparkSession|
             -> Result<((DsparkPrefix, Array), Vec<u32>)> {
                let mut saved = None;
                let run = d1_run(
                    session,
                    target,
                    draft,
                    &text[..first],
                    None,
                    Some(boundary),
                    &mut saved,
                    ANSWER,
                )?;
                ensure!(run.cached == 0, "turn 1 resumed");
                let saved = saved.context("no DSpark prefix captured")?;
                ensure!(saved.0.offset() as usize == boundary, "prefix offset");
                let prompt = if extra == 0 {
                    text[..first].to_vec()
                } else {
                    let mut p = text[..first].to_vec();
                    p.extend(&run.tokens);
                    p.extend(&text[first + 1..first + 1 + extra]);
                    p
                };
                Ok((saved, prompt))
            };
            let mut session = DsparkSession::new(&draft, DsparkSession::DEFAULT_WIDTH)?;
            let (prefix, prompt) = turn1(&mut target, &mut draft, &mut session)?;
            let retained = prefix.0.retained_bytes()?;
            println!(
                "first {first}: prefix at {boundary} retains {retained} bytes (drafter context {})",
                prefix.0.context.retained_bytes()?
            );
            ensure!(
                retained <= crate::memory_policy::DEFAULT_PROMPT_CACHE_BYTES,
                "the DSpark prefix exceeds the default prompt cache budget"
            );
            let checkpoint = prompt.len() / 128 * 128;
            // turn 2 resumed in a new session (as the bridge does), with the capture of the next boundary
            let mut next_prefix = None;
            let mut resumed_session = DsparkSession::new(&draft, DsparkSession::DEFAULT_WIDTH)?;
            let resumed = d1_run(
                &mut resumed_session,
                &mut target,
                &mut draft,
                &prompt,
                Some(prefix),
                Some(checkpoint),
                &mut next_prefix,
                STEPS,
            )?;
            ensure!(resumed.cached == boundary, "turn 2 did not resume");
            let next_prefix = next_prefix.context("no prefix captured after the resume")?;
            ensure!(
                next_prefix.0.offset() as usize == checkpoint,
                "recaptured offset"
            );
            // the same prompt from scratch
            let mut unused = None;
            let mut cold_session = DsparkSession::new(&draft, DsparkSession::DEFAULT_WIDTH)?;
            let cold = d1_run(
                &mut cold_session,
                &mut target,
                &mut draft,
                &prompt,
                None,
                None,
                &mut unused,
                STEPS,
            )?;
            println!(
                "first {first}: prompt {} tokens, resumed {}, cold {:?}",
                prompt.len(),
                resumed.cached,
                (cold.logits, cold.context, cold.last_feature, cold.decode)
            );
            ensure!(
                D1Run {
                    cached: 0,
                    ..resumed
                } == cold,
                "resumed DSpark prefill differs from the cold prefill"
            );
            // a prefix of another lineage (the cold prefill reset the target since): refused, cold prefill
            let mut unused = None;
            let mut refused_session = DsparkSession::new(&draft, DsparkSession::DEFAULT_WIDTH)?;
            let refused = d1_run(
                &mut refused_session,
                &mut target,
                &mut draft,
                &prompt,
                Some(next_prefix),
                None,
                &mut unused,
                STEPS,
            )?;
            ensure!(
                refused == cold,
                "a refused prefix did not prefill from scratch"
            );
            // mutations, each from a fresh turn 1 whose session is reused for turn 2
            for mutation in [1u8, 2] {
                if mutation == 2 && extra != 0 {
                    // the next chunk replaces the last features: only a fully resumed prompt reads them
                    continue;
                }
                let mut session = DsparkSession::new(&draft, DsparkSession::DEFAULT_WIDTH)?;
                let (prefix, prompt) = turn1(&mut target, &mut draft, &mut session)?;
                D1_MUTATION.with(|m| m.set(mutation));
                let mut unused = None;
                let mutated = d1_run(
                    &mut session,
                    &mut target,
                    &mut draft,
                    &prompt,
                    Some(prefix),
                    None,
                    &mut unused,
                    STEPS,
                );
                D1_MUTATION.with(|m| m.set(0));
                let mutated = mutated?;
                println!(
                    "first {first}: mutation {mutation}: context {} last feature {} decode {}",
                    mutated.context != cold.context,
                    mutated.last_feature != cold.last_feature,
                    mutated.decode != cold.decode
                );
                ensure!(
                    D1Run {
                        cached: 0,
                        ..mutated
                    } != cold,
                    "mutation {mutation} not detected"
                );
            }
        }
        Ok(())
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D2: the coupled decision delivers draws of the target law at every position,
    /// whatever the drafter proposes. Fixed target rows p_i and drafter rows q_i over 6 tokens, proposals drawn by
    /// `coupled_pick` with the block's noise, 40 000 blocks per case: for each position i, the token delivered at
    /// i among the blocks that reach it against p_i = softmax(log_softmax(row i) / T) (Pearson chi-square below
    /// the 0.999 quantile per position); with q = p every proposal is kept (the drafter and the target read the
    /// same noise row); the mutation that keeps every proposal is detected.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn decide_coupled_follows_the_target_law() -> Result<()> {
        use crate::inkling2::{COUPLE_MUTATION, coupled_pick, decide_coupled};
        const BLOCKS: usize = 40_000;
        const BATCH: usize = 1_000;
        let p_rows: [[f32; 6]; 4] = [
            [2.0, 1.0, 0.5, 0.0, -1.0, -3.0],
            [-0.5, 1.5, 0.2, 1.0, -2.0, 0.0],
            [0.3, -1.0, 2.5, 0.0, 1.0, -0.5],
            [1.0, 0.0, -0.5, 2.0, 0.5, -1.0],
        ];
        let q_rows: [[f32; 6]; 3] = [
            [1.5, 1.6, 0.0, 0.2, -0.5, -2.0],
            [0.0, 1.0, 0.8, 1.2, -1.0, 0.3],
            [0.8, -0.2, 1.8, 0.4, 0.6, 0.0],
        ];
        let softmax = |row: &[f32], t: f32| -> Vec<f64> {
            let max = row
                .iter()
                .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
            let lse = max
                + row
                    .iter()
                    .map(|&v| (f64::from(v) - max).exp())
                    .sum::<f64>()
                    .ln();
            let scaled: Vec<f64> = row
                .iter()
                .map(|&v| (f64::from(v) - lse) / f64::from(t))
                .collect();
            let top = scaled.iter().fold(f64::NEG_INFINITY, |m, &v| m.max(v));
            let w: Vec<f64> = scaled.iter().map(|&v| (v - top).exp()).collect();
            let total: f64 = w.iter().sum();
            w.iter().map(|x| x / total).collect()
        };
        // one block: (packet, proposals) for k proposals from the q rows (or the p rows when `same`)
        let block = |k: usize, t: f32, same: bool, seed: u64| -> Result<(Array, Array)> {
            let noise = Array::gumbel_noise(&[k as i32 + 1, 6], seed)?;
            let mut picks = Vec::with_capacity(k);
            for i in 0..k {
                let row: &[f32] = if same { &p_rows[i] } else { &q_rows[i] };
                picks.push(coupled_pick(
                    &Array::from_f32(row, &[1, 6])?,
                    t,
                    &noise.slice(0, i as i32, i as i32 + 1)?,
                )?);
            }
            let proposals = Array::concatenate(&picks.iter().collect::<Vec<_>>(), 0)?;
            let flat: Vec<f32> = p_rows[..=k].iter().flatten().copied().collect();
            let logits = Array::from_f32(&flat, &[1, k as i32 + 1, 6])?;
            Ok((decide_coupled(&logits, &proposals, t, &noise)?, proposals))
        };
        let chi2_bound = |df: f64| {
            let z = 3.090_232_306;
            df * (1.0 - 2.0 / (9.0 * df) + z * (2.0 / (9.0 * df)).sqrt()).powi(3)
        };
        // per position i: counts of the delivered token among the blocks reaching i
        let law_check = |k: usize, t: f32, seed: u64| -> Result<(bool, f64)> {
            let mut counts = vec![[0f64; 6]; k + 1];
            let mut accepted = 0usize;
            for batch in 0..BLOCKS / BATCH {
                let blocks = (0..BATCH)
                    .map(|i| block(k, t, false, seed + (batch * BATCH + i) as u64))
                    .collect::<Result<Vec<_>>>()?;
                Array::eval_many(&blocks.iter().map(|(p, _)| p).collect::<Vec<_>>())?;
                for (packet, proposals) in &blocks {
                    let p = packet.to_u32()?;
                    let x = proposals.to_u32()?;
                    ensure!(
                        p.len() == 2 + k && p[2..] == x[..] && (p[0] as usize) <= k && p[1] < 6,
                        "invalid packet {p:?}"
                    );
                    let a = p[0] as usize;
                    accepted += a;
                    for (i, row) in counts.iter_mut().enumerate().take(a + 1) {
                        let token = if i < a { x[i] } else { p[1] };
                        row[token as usize] += 1.0;
                    }
                }
            }
            let mut ok = true;
            for (i, row) in counts.iter().enumerate() {
                let law = softmax(&p_rows[i], t);
                let reached: f64 = row.iter().sum();
                if reached < 200.0 {
                    continue;
                }
                let chi2: f64 = (0..6)
                    .map(|j| (row[j] - law[j] * reached).powi(2) / (law[j] * reached))
                    .sum();
                let bound = chi2_bound(5.0);
                println!(
                    "k={k} T={t} position {i}: reached {reached}, chi2 {chi2:.2} bound {bound:.2}"
                );
                ok &= chi2 < bound;
            }
            Ok((ok, accepted as f64 / (BLOCKS * k) as f64))
        };
        for (k, t, seed) in [
            (2usize, 1.0f32, 11u64),
            (3, 0.7, 2_000_011),
            (3, 1.3, 4_000_011),
        ] {
            let (ok, rate) = law_check(k, t, seed)?;
            println!("k={k} T={t}: kept {rate:.3} of the proposals");
            ensure!(
                ok,
                "coupled decisions do not follow the target law (k={k}, T={t})"
            );
            // q = p: the drafter's coupled picks are the target's draws, every proposal is kept
            for i in 0..200u64 {
                let (packet, _) = block(k, t, true, seed + 7_000_000 + i)?;
                let p = packet.to_u32()?;
                ensure!(p[0] as usize == k, "q = p, a proposal was rejected: {p:?}");
            }
        }
        COUPLE_MUTATION.with(|m| m.set(true));
        let mutated = law_check(3, 1.0, 9_000_011);
        COUPLE_MUTATION.with(|m| m.set(false));
        let (ok, rate) = mutated?;
        println!("mutation (every proposal kept): law ok {ok}, kept {rate:.3}");
        ensure!(
            !ok,
            "the mutation that keeps every proposal was not detected"
        );
        Ok(())
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D2-DIAG: acceptance of the first proposal under each rule at T = 1, on the same
    /// real text (teacher forcing, no sampling variance between rules): rows en0, fr0, code0, chat0, fr1, code1 of
    /// `MLXL3_INKLING_ROWS`, 64 prefix tokens then 192 positions each; at every position the drafter's logits q of
    /// its first proposal (true prefix) and the target's row p of the next token. Point rule: p(argmax q), exact.
    /// Coupled with drafter temperature share tau: the share of 256 Gumbel draws G where argmax(log p + G) =
    /// argmax(q / tau + G). Prints the means per row and over all positions.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn coupling_acceptance_by_rule() -> Result<()> {
        const PREFIX: usize = 64;
        const POSITIONS: usize = 192;
        const DRAWS: i32 = 256;
        let taus = [1.0f32, 0.7, 0.5, 0.3];
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let mut target = Inkling2::load(&dir)?;
        let mut draft = Inkling2Draft::load(&dir, &target)?;
        let vocab = target.unpadded_vocab();
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let rows = rows["rows"].as_array().context("rows")?;
        let (mut all_point, mut all_coupled, mut all_n) = (0f64, vec![0f64; taus.len()], 0usize);
        let mut seed = 0x5eed_u64;
        for index in [0usize, 3, 6, 9, 4, 7] {
            let name = rows[index]["name"].as_str().unwrap_or("?").to_owned();
            let tokens: Vec<u32> =
                serde_json::from_value::<Vec<u32>>(rows[index]["tokens"].clone())?
                    .into_iter()
                    .map(|t| t % vocab as u32)
                    .collect();
            ensure!(
                tokens.len() > PREFIX + POSITIONS + 1,
                "row {name} too short"
            );
            let mut session = DsparkSession::new(&draft, DsparkSession::DEFAULT_WIDTH)?;
            session.prefill(&mut target, &mut draft, &tokens[..PREFIX], &|| false)?;
            let mut last = session.last_feature.take().context("no last feature")?;
            let (mut point, mut coupled) = (0f64, vec![0f64; taus.len()]);
            for j in PREFIX..PREFIX + POSITIONS {
                let anchor = Array::from_u32(&[tokens[j]], &[1])?;
                let (_, q) = draft.propose_sampled(&target, &anchor, &last, 1, 1.0, seed)?;
                let (logits, features) =
                    target.forward_with_features(&tokens[j..j + 1], &self_taps(&draft))?;
                let log_p = logits.reshape(&[1, vocab])?.log_probs()?;
                let q = q.reshape(&[1, vocab])?.astype(Dtype::Float32)?;
                // point rule: p of the drafter's best token
                let best = q.argmax()?.to_u32()?[0] as usize;
                let row = log_p.to_f32()?;
                point += f64::from(row[best]).exp();
                // coupled: one noise for every tau
                let noise = Array::gumbel_noise(&[DRAWS, vocab], seed ^ 0xc0ff_ee00)?;
                let target_draws = log_p.add(&noise)?.argmax()?.to_u32()?;
                for (slot, &tau) in coupled.iter_mut().zip(&taus) {
                    let draft_draws = q.scalar_mul(1.0 / tau)?.add(&noise)?.argmax()?.to_u32()?;
                    let same = target_draws
                        .iter()
                        .zip(&draft_draws)
                        .filter(|(a, b)| a == b)
                        .count();
                    *slot += same as f64 / f64::from(DRAWS);
                }
                draft.append_context(&features)?;
                last = features;
                seed += 1;
            }
            let n = POSITIONS as f64;
            println!(
                "D2DIAG {name}: point {:.4} coupled {}",
                point / n,
                taus.iter()
                    .zip(&coupled)
                    .map(|(t, c)| format!("tau {t}: {:.4}", c / n))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            all_point += point;
            for (a, c) in all_coupled.iter_mut().zip(&coupled) {
                *a += c;
            }
            all_n += POSITIONS;
        }
        let n = all_n as f64;
        println!(
            "D2DIAG total ({all_n} positions): point {:.4} coupled {}",
            all_point / n,
            taus.iter()
                .zip(&all_coupled)
                .map(|(t, c)| format!("tau {t}: {:.4}", c / n))
                .collect::<Vec<_>>()
                .join(", ")
        );
        Ok(())
    }

    fn self_taps(draft: &Inkling2Draft) -> Vec<usize> {
        draft.taps().to_vec()
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D5: a DSpark session with the prompt lookup, on a prompt whose end copies its
    /// start (300 real tokens, then their first 40 again). Greedy, decided on the GPU, one step ahead: lookup
    /// blocks run and a forced AR replay of the delivered tokens finds every token the argmax of its AR row; the
    /// mutation that delivers lookup proposals without the target's decision is found by the same replay. T = 1:
    /// synchronous = one step ahead (same seed, same tokens), lookup blocks included.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn dspark_lookup_session() -> Result<()> {
        use crate::inkling2::{DrawRule, DrawSeeds};
        use crate::sampling::{Draw, History, Sampler};
        const TOKENS: usize = 160;
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let mut target = Inkling2::load(&dir)?;
        let mut draft = Inkling2Draft::load(&dir, &target)?;
        let vocab = target.unpadded_vocab() as u32;
        let text: Vec<u32> = text_rows()?.into_iter().map(|t| t % vocab).collect();
        let mut prompt = text[..300].to_vec();
        prompt.extend(&text[..40]);
        // the forced lookup runs on the same start without the copy: its 3-token matches are often wrong
        let plain = text[..300].to_vec();
        let limit = usize::try_from(target.context_limit())?;
        let run = |target: &mut Inkling2,
                   draft: &mut Inkling2Draft,
                   prompt: &[u32],
                   sampler: Sampler,
                   lookup: bool,
                   pipelined: bool,
                   seed: u64|
         -> Result<(Vec<u32>, SpecStats)> {
            let rule = DrawRule::for_sampler(&sampler).context("no GPU rule")?;
            let mut session = DsparkSession::new(draft, DsparkSession::DEFAULT_WIDTH)?
                .with_adaptive_width(WidthPolicy::measured(
                    &WidthPolicy::SAMPLED_8BIT_COSTS,
                    DsparkSession::DEFAULT_WIDTH,
                )?)?
                .with_device_draw(rule, DrawSeeds::new(seed), pipelined)
                .with_device_rounds(true)
                .with_coupling(false);
            if lookup {
                session = session.with_lookup(prompt);
            }
            ensure!(session.looks_up() == lookup, "lookup not applied");
            let logits = session.prefill(target, draft, prompt, &|| false)?;
            let history = History::default();
            let mut random = seed ^ 0x9e37_79b9;
            let mut next = if sampler.is_greedy() {
                logits.chat_greedy_ids()?[0]
            } else {
                Draw {
                    sampler,
                    history: &history,
                    random: &mut random,
                }
                .sample_last_row(&logits)?
            };
            let mut out = vec![next];
            while out.len() < TOKENS {
                let mut choose = |proposals: &[u32],
                                  _: Option<&[Vec<f32>]>,
                                  verified: &Array|
                 -> Result<(usize, u32)> {
                    let acceptance = if sampler.is_greedy() {
                        crate::speculative::greedy_accept(proposals, &verified.chat_greedy_ids()?)
                            .context("invalid greedy acceptance")?
                    } else {
                        Draw {
                            sampler,
                            history: &history,
                            random: &mut random,
                        }
                        .speculative_accept_logits(proposals, verified)?
                    };
                    Ok((acceptance.accepted_draft_tokens, acceptance.target_token))
                };
                next =
                    session.advance(target, draft, next, limit, TOKENS - out.len(), &mut choose)?;
                out.push(next);
            }
            Ok((out, session.stats.clone()))
        };
        // the differences of a forced AR replay of `tokens`: (position, AR max - delivered logit, rank)
        let replay = |target: &mut Inkling2,
                      prompt: &[u32],
                      tokens: &[u32]|
         -> Result<Vec<(usize, f32, usize)>> {
            target.reset();
            let mut row = target.forward_tokens(prompt)?;
            let mut differences = Vec::new();
            for (i, &token) in tokens.iter().enumerate() {
                let best = row.chat_greedy_ids()?[0];
                if best != token {
                    let values = row.reshape(&[-1])?.to_f32()?;
                    let got = values[token as usize];
                    let rank = values.iter().filter(|&&v| v > got).count();
                    differences.push((i, values[best as usize] - got, rank));
                }
                if i + 1 < tokens.len() {
                    row = target.forward(token)?;
                }
            }
            Ok(differences)
        };
        let greedy = Sampler::greedy();
        let sampled = Sampler::new(1.0, 0, 1.0, 1.0, 0.0)?;
        let (tokens, stats) = run(&mut target, &mut draft, &prompt, greedy, true, true, 5)?;
        let drafter_wide: usize = stats.width_blocks.iter().skip(1).sum();
        println!(
            "D5 greedy: lookup blocks {} accepted {}/{} widths {:?} held {}; drafter blocks {} wide {} accepted {}/{}",
            stats.lookup_blocks,
            stats.lookup_accepted,
            stats.lookup_proposed,
            stats.lookup_width_blocks,
            stats.lookup_held,
            stats.blocks,
            drafter_wide,
            stats.accepted,
            stats.proposed
        );
        ensure!(
            stats.lookup_blocks > 0 && stats.lookup_accepted > 0,
            "no lookup block was kept"
        );
        let copy_differences = replay(&mut target, &prompt, &tokens)?;
        println!("D5 greedy replay (copy): differences {copy_differences:?}");
        // every proposal run as a full lookup block on the text without the copy (rejections), recorded
        D5_FORCE.with(|m| m.set(true));
        D5_RECORD.with(|r| *r.borrow_mut() = Some(Vec::new()));
        let forced = run(&mut target, &mut draft, &plain, greedy, true, true, 5);
        let record = D5_RECORD
            .with(|r| r.borrow_mut().take())
            .unwrap_or_default();
        D5_MUTATION.with(|m| m.set(true));
        let mutated = run(&mut target, &mut draft, &plain, greedy, true, true, 5);
        D5_MUTATION.with(|m| m.set(false));
        D5_FORCE.with(|m| m.set(false));
        let (forced, forced_stats) = forced?;
        let (mutated, _) = mutated?;
        println!(
            "D5 forced: lookup blocks {} accepted {}/{} widths {:?}",
            forced_stats.lookup_blocks,
            forced_stats.lookup_accepted,
            forced_stats.lookup_proposed,
            forced_stats.lookup_width_blocks
        );
        ensure!(
            forced_stats.lookup_blocks > 0
                && forced_stats.lookup_accepted < forced_stats.lookup_proposed,
            "the forced lookup ran no block with a rejection"
        );
        // (a) the rule itself, without tolerance: kept proposals and the token are the verified rows' argmax
        ensure!(
            record.len() == forced_stats.lookup_blocks,
            "lookup blocks not all recorded"
        );
        for (proposals, rows, accepted, token) in &record {
            let keep = proposals
                .iter()
                .zip(rows)
                .take_while(|(p, r)| p == r)
                .count();
            ensure!(
                *accepted == keep && *token == rows[keep],
                "lookup block {proposals:?} rows {rows:?}: accepted {accepted}, token {token}"
            );
        }
        println!(
            "D5 forced rule: {} lookup blocks follow their verified rows",
            record.len()
        );
        // (b) the reference: DSpark without the lookup at the fixed width 3 on the same text
        let reference = {
            let rule = DrawRule::Greedy;
            let mut session = DsparkSession::new(&draft, DsparkSession::DEFAULT_WIDTH)?
                .with_device_draw(rule, DrawSeeds::new(5), true)
                .with_device_rounds(true)
                .with_coupling(false);
            let logits = session.prefill(&mut target, &mut draft, &plain, &|| false)?;
            let mut next = logits.chat_greedy_ids()?[0];
            let mut out = vec![next];
            while out.len() < TOKENS {
                let mut choose = |proposals: &[u32],
                                  _: Option<&[Vec<f32>]>,
                                  verified: &Array|
                 -> Result<(usize, u32)> {
                    let a =
                        crate::speculative::greedy_accept(proposals, &verified.chat_greedy_ids()?)
                            .context("invalid greedy acceptance")?;
                    Ok((a.accepted_draft_tokens, a.target_token))
                };
                next = session.advance(
                    &mut target,
                    &mut draft,
                    next,
                    limit,
                    TOKENS - out.len(),
                    &mut choose,
                )?;
                out.push(next);
            }
            out
        };
        let reference_differences = replay(&mut target, &plain, &reference)?;
        let forced_differences = replay(&mut target, &plain, &forced)?;
        let mutated_differences = replay(&mut target, &plain, &mutated)?;
        println!(
            "D5 reference (drafter only, width 3) replay: differences {reference_differences:?}"
        );
        println!("D5 forced replay: differences {forced_differences:?}");
        println!(
            "D5 mutation replay: {} differences",
            mutated_differences.len()
        );
        let reference_max = reference_differences
            .iter()
            .map(|d| d.1)
            .fold(0f32, f32::max);
        let bound = if reference_differences.is_empty() {
            0.25
        } else {
            reference_max
        };
        ensure!(
            forced_differences
                .iter()
                .chain(&copy_differences)
                .all(|&(_, gap, rank)| rank == 1 && gap <= bound),
            "lookup differences beyond the reference (bound {bound}): {forced_differences:?} {copy_differences:?}"
        );
        if reference_differences.is_empty() {
            ensure!(
                forced_differences.len() + copy_differences.len() <= 2,
                "more lookup differences than the fallback bound"
            );
            println!("D5: reference without differences, fallback bound used (non conclusive)");
        }
        // (c) the mutation
        ensure!(
            mutated_differences.len() >= 10
                && mutated_differences.len() >= 3 * forced_differences.len(),
            "the unverified lookup mutation was not detected"
        );
        // T = 1: synchronous = one step ahead, lookup included
        let (sync, sync_stats) = run(&mut target, &mut draft, &prompt, sampled, true, false, 11)?;
        let (ahead, ahead_stats) = run(&mut target, &mut draft, &prompt, sampled, true, true, 11)?;
        println!(
            "D5 T=1: lookup blocks {} accepted {}/{} (ahead: {} {}/{}, held {})",
            sync_stats.lookup_blocks,
            sync_stats.lookup_accepted,
            sync_stats.lookup_proposed,
            ahead_stats.lookup_blocks,
            ahead_stats.lookup_accepted,
            ahead_stats.lookup_proposed,
            ahead_stats.lookup_held
        );
        ensure!(sync_stats.lookup_blocks > 0, "no lookup block at T = 1");
        ensure!(
            sync == ahead,
            "T = 1 lookup sessions differ between synchronous and one step ahead"
        );
        Ok(())
    }
}
