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

#[derive(Default)]
struct DraftCache {
    /// `[1, Hkv, S, D]` FP16 context keys (RoPE applied) and values.
    keys: Option<Array>,
    values: Option<Array>,
    /// Absolute position of the next context token.
    offset: i32,
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
        let head = target
            .head_weight()
            .context("the DSpark fold needs a dense target head")?;
        let hidden_out = f32_weight(&ckpt, "markov_head.hidden_out.weight", &[h, rank])?;
        let hidden_bias = f32_weight(&ckpt, "markov_head.hidden_out.bias", &[h])?;
        let vocab = target.unpadded_vocab();
        // lm_head @ hidden_out and lm_head @ hidden_bias in FP32 by blocks of rows: a whole FP32 copy of the
        // head would take 1.8 GB and push the target's mapped weights out of RAM before its first forward.
        const FOLD_ROWS: i32 = 16384;
        let (mut fold_parts, mut bias_parts) = (Vec::new(), Vec::new());
        let mut start = 0;
        while start < vocab {
            let end = (start + FOLD_ROWS).min(vocab);
            let rows = head.slice(0, start, end)?.astype(Dtype::Float32)?;
            let part = rows.matmul(&hidden_out)?.astype(Dtype::Float16)?;
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
        drop((fold_parts, bias_parts));
        let fold_t = fold.transpose(&[1, 0])?;
        // draft-only affine copies; the target head stays the EXL3/FP16 one used for verification
        let (head_q, fold_q) = if c.head_bits < 16 {
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
                Some(&[target_vocab(target)?, rank]),
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
            .propose_with(target, anchor, last_feature, count, None)?
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
            Some((temperature, seed)),
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
        let (tokens, x, base, _) = self.propose_with(target, anchor, last_feature, count, None)?;
        Ok((tokens, x, base))
    }

    /// Greedy proposals when `sampled` is `None`, Gumbel-max draws from softmax(logits / T) with noise
    /// seeded by `seed + position` otherwise (then also the stacked draft logits).
    fn propose_with(
        &self,
        target: &Inkling2,
        anchor: &Array,
        last_feature: &Array,
        count: usize,
        sampled: Option<(f32, u64)>,
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
        let mut rows = Vec::with_capacity(if sampled.is_some() { count } else { 0 });
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
            previous = match sampled {
                None => logits.argmax()?,
                Some((temperature, seed)) => {
                    let previous = gumbel_draw(&logits, temperature, seed.wrapping_add(i as u64))?;
                    rows.push(logits);
                    previous
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

fn target_vocab(target: &Inkling2) -> Result<i32> {
    Ok(target.head_weight().context("dense target head")?.shape()[0])
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
    pub stats: SpecStats,
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

    pub fn new(draft: &Inkling2Draft, width: usize) -> Result<Self> {
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
            stats: SpecStats {
                accepted_at: vec![0; width],
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

    /// Choose each block's width (0 to the session width) from the acceptance observed so far.
    pub fn with_adaptive_width(mut self) -> Self {
        self.policy = Some(WidthPolicy::new(self.width));
        self.stats.width_blocks = vec![0; self.width + 1];
        self
    }

    /// Resets both models and prefills `tokens` by chunks, giving the drafter its context. Returns the
    /// logits of the last prompt row `[1, 1, vocab]`.
    pub fn prefill(
        &mut self,
        target: &mut Inkling2,
        draft: &mut Inkling2Draft,
        tokens: &[u32],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Array> {
        ensure!(!tokens.is_empty(), "DSpark prefill needs tokens");
        let t = Instant::now();
        target.reset();
        draft.reset();
        self.pending.clear();
        self.expected = None;
        let mut logits = None;
        for chunk in tokens.chunks(Self::PREFILL_CHUNK) {
            ensure!(!cancelled(), "generation cancelled");
            let started = Instant::now();
            let (chunk_logits, features) = target.forward_with_features(chunk, &self.taps)?;
            let target_done = started.elapsed();
            draft.append_context(&features)?;
            if std::env::var_os("MLXL3_INKLING_DSPARK_TRACE").is_some() {
                eprintln!(
                    "dspark prefill chunk {} rows: target {:.1} ms, draft context {:.1} ms",
                    chunk.len(),
                    target_done.as_secs_f64() * 1e3,
                    (started.elapsed() - target_done).as_secs_f64() * 1e3
                );
            }
            let rows = chunk.len() as i32;
            self.last_feature = Some(features.slice(0, rows - 1, rows)?);
            logits = Some(chunk_logits);
        }
        self.stats.prefill_ms += t.elapsed().as_secs_f64() * 1e3;
        logits.context("empty DSpark prefill")
    }

    /// `next`: the token the caller just delivered. Returns the token after it, running a draft block
    /// when nothing verified is left. `remaining` >= 1 tokens may still be delivered; `context_limit`
    /// bounds the target positions.
    pub fn advance(
        &mut self,
        target: &mut Inkling2,
        draft: &mut Inkling2Draft,
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
        if let Some(token) = self.pending.pop_front() {
            self.expected = Some(token);
            return Ok(token);
        }
        // `next` is the anchor: the target has not seen it yet
        let offset = usize::try_from(target.offset())?;
        ensure!(offset < context_limit, "Inkling context is full");
        let width = match self.policy.as_mut() {
            Some(policy) => policy.next_width(),
            None => self.width,
        };
        let count = width.min(remaining - 1).min(context_limit - offset - 1);
        let last_feature = self
            .last_feature
            .take()
            .context("DSpark session without context")?;
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
        let t = Instant::now();
        if !committed {
            target.commit(accepted + 1)?;
        }
        draft.append_context(&features.slice(0, 0, accepted as i32 + 1)?)?;
        self.last_feature = Some(features.slice(0, accepted as i32, accepted as i32 + 1)?);
        self.stats.commit_ms += t.elapsed().as_secs_f64() * 1e3;
        self.stats.blocks += 1;
        self.stats.proposed += proposals.len();
        self.stats.accepted += accepted;
        if let Some(policy) = self.policy.as_mut() {
            policy.observe(proposals.len(), accepted);
            if let Some(slot) = self.stats.width_blocks.get_mut(proposals.len()) {
                *slot += 1;
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
}

/// DSPARK-INK2M-P4: the DSpark width of each block, chosen from the acceptance observed so far.
///
/// `acceptance[i]` estimates the probability that proposal `i` is accepted given that proposals `0..i`
/// were (exponential average over the blocks that reached it). A block of width `w` delivers
/// E(w) = 1 + sum over i < w of the product of acceptance[0..=i] tokens and costs C(w) = 1 + (VERIFY_COST +
/// DRAFT_COST) * w plain decoding steps (one extra verified row, about +0.2 of a step on Inkling 2 Mini
/// because every row reads its own experts, and one draft position, about +0.07). The width maximizing
/// E(w) / C(w) is used; width 0 is a plain step. Every `PROBE_EVERY` blocks the full width is tried so the
/// estimates of the positions not used meanwhile stay current. The choice depends only on the accepted
/// counts, never on timings, so a generation stays reproducible; the acceptance rule, and so the law of
/// every token, does not depend on the width.
#[derive(Clone, Debug)]
pub struct WidthPolicy {
    acceptance: Vec<f64>,
    blocks: usize,
}

impl WidthPolicy {
    pub const ALPHA: f64 = 0.15;
    pub const PRIOR: f64 = 0.75;
    pub const PROBE_EVERY: usize = 8;
    pub const VERIFY_COST: f64 = 0.20;
    pub const DRAFT_COST: f64 = 0.07;

    pub fn new(max_width: usize) -> Self {
        Self {
            acceptance: vec![Self::PRIOR; max_width],
            blocks: 0,
        }
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

    /// Cost of a block of `width`, in plain decoding steps.
    pub fn cost(width: usize) -> f64 {
        1.0 + (Self::VERIFY_COST + Self::DRAFT_COST) * width as f64
    }

    /// The width with the most tokens per unit of cost (the smaller one on a tie).
    pub fn best(&self) -> usize {
        (0..=self.max_width())
            .map(|w| (w, self.expected_tokens(w) / Self::cost(w)))
            .fold((0, f64::MIN), |best, (w, rate)| {
                if rate > best.1 + 1e-12 {
                    (w, rate)
                } else {
                    best
                }
            })
            .0
    }

    /// The width of the next block: the best one, or the full width on a probe block.
    pub fn next_width(&mut self) -> usize {
        self.blocks += 1;
        if self.blocks.is_multiple_of(Self::PROBE_EVERY) {
            self.max_width()
        } else {
            self.best()
        }
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
    }

    pub fn acceptance(&self) -> &[f64] {
        &self.acceptance
    }

    #[cfg(test)]
    fn with_acceptance(acceptance: Vec<f64>) -> Self {
        Self {
            acceptance,
            blocks: 0,
        }
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
        assert!((WidthPolicy::cost(3) - 1.81).abs() < 1e-12);
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
}
