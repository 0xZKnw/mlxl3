//! Native Ling 3 / Bailing MoE V3 one-token inference using EXL3 projections.
use crate::{
    array::{Array, Dtype},
    checkpoint::{self, Checkpoint},
    gated_delta,
    lfm2::{Projection, half_weight},
    moe::Exl3SwitchGlu,
    router,
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{fs::File, path::Path};

#[derive(Deserialize)]
struct Config {
    model_type: String,
    vocab_size: i32,
    hidden_size: i32,
    intermediate_size: i32,
    moe_intermediate_size: i32,
    moe_shared_expert_intermediate_size: i32,
    num_hidden_layers: usize,
    num_attention_heads: i32,
    num_experts: i32,
    num_experts_per_tok: usize,
    num_shared_experts: i32,
    first_k_dense_replace: usize,
    n_group: usize,
    topk_group: usize,
    routed_scaling_factor: f32,
    layer_group_size: usize,
    head_dim: i32,
    q_lora_rank: i32,
    kv_lora_rank: i32,
    qk_nope_head_dim: i32,
    qk_rope_head_dim: i32,
    v_head_dim: i32,
    short_conv_kernel_size: i32,
    kda_lower_bound: f32,
    kda_safe_gate: bool,
    no_kda_lora: bool,
    rms_norm_eps: f32,
    rope_theta: f32,
    max_position_embeddings: i32,
    rope_interleave: bool,
    tie_word_embeddings: bool,
}

fn norm(checkpoint: &Checkpoint, name: &str, width: i32) -> Result<Array> {
    half_weight(checkpoint, name, Some(&[width]))
}

fn is_kda_layer(index: usize, group: usize) -> bool {
    !(index + 1).is_multiple_of(group)
}

struct Mlp {
    gate: Projection,
    up: Projection,
    down: Projection,
}

impl Mlp {
    fn load(checkpoint: &Checkpoint, prefix: &str, hidden: i32, intermediate: i32) -> Result<Self> {
        Ok(Self {
            gate: Projection::load(
                checkpoint,
                &format!("{prefix}.gate_proj"),
                hidden,
                intermediate,
                false,
            )?,
            up: Projection::load(
                checkpoint,
                &format!("{prefix}.up_proj"),
                hidden,
                intermediate,
                false,
            )?,
            down: Projection::load(
                checkpoint,
                &format!("{prefix}.down_proj"),
                intermediate,
                hidden,
                false,
            )?,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        self.down
            .forward(&self.gate.forward(x)?.swiglu(&self.up.forward(x)?)?)
    }
}

struct SparseMoe {
    gate_weight: Array,
    expert_bias: Array,
    experts: Exl3SwitchGlu,
    shared: Mlp,
    hidden: i32,
    top_k: usize,
    groups: usize,
    top_groups: usize,
    scale: f32,
}

impl SparseMoe {
    fn load(checkpoint: &Checkpoint, prefix: &str, config: &Config) -> Result<Self> {
        let gate_weight = half_weight(
            checkpoint,
            &format!("{prefix}.gate.weight"),
            Some(&[config.num_experts, config.hidden_size]),
        )?
        .astype(Dtype::Float32)?
        .transpose(&[1, 0])?;
        let bias_name = format!("{prefix}.gate.expert_bias");
        let bias_name = if checkpoint.tensors.contains_key(&bias_name) {
            bias_name
        } else {
            format!("{prefix}.expert_bias")
        };
        let expert_bias = half_weight(checkpoint, &bias_name, Some(&[config.num_experts]))?
            .astype(Dtype::Float32)?;
        gate_weight.eval()?;
        expert_bias.eval()?;
        Ok(Self {
            gate_weight,
            expert_bias,
            experts: Exl3SwitchGlu::from_checkpoint(
                checkpoint,
                &format!("{prefix}.experts"),
                config.num_experts,
                config.num_experts_per_tok as i32,
            )?,
            shared: Mlp::load(
                checkpoint,
                &format!("{prefix}.shared_experts"),
                config.hidden_size,
                config.moe_shared_expert_intermediate_size * config.num_shared_experts,
            )?,
            hidden: config.hidden_size,
            top_k: config.num_experts_per_tok,
            groups: config.n_group,
            top_groups: config.topk_group,
            scale: config.routed_scaling_factor,
        })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        ensure!(x.shape() == [1, self.hidden], "Ling MoE expects one token");
        let logits = x.astype(Dtype::Float32)?.matmul(&self.gate_weight)?;
        let (selected, scores) = router::grouped_topk_biased(
            &logits,
            &self.expert_bias,
            self.top_k,
            self.groups,
            self.top_groups,
            self.scale,
        )?;
        self.experts
            .forward(x, &selected, &scores)?
            .astype(Dtype::Float16)?
            .add(&self.shared.forward(x)?)
    }
}

enum FeedForward {
    Dense(Box<Mlp>),
    Sparse(Box<SparseMoe>),
}

impl FeedForward {
    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Dense(mlp) => mlp.forward(x),
            Self::Sparse(moe) => moe.forward(x),
        }
    }
}

struct Kda {
    q: Projection,
    k: Projection,
    v: Projection,
    f: Projection,
    b_weight: Array,
    output_gate: Projection,
    output_norm: Array,
    output: Projection,
    q_conv: Array,
    k_conv: Array,
    v_conv: Array,
    a_log: Array,
    dt_bias: Array,
    conv_q: Option<Array>,
    conv_k: Option<Array>,
    conv_v: Option<Array>,
    recurrent: Option<Array>,
    heads: i32,
    dim: i32,
    conv_length: i32,
    eps: f32,
    lower_bound: f32,
}

impl Kda {
    fn load(checkpoint: &Checkpoint, prefix: &str, config: &Config) -> Result<Self> {
        let projection = config.num_attention_heads * config.head_dim;
        let conv = |name: &str| {
            half_weight(
                checkpoint,
                &format!("{prefix}.{name}_conv1d.weight"),
                Some(&[projection, 1, config.short_conv_kernel_size]),
            )?
            .transpose(&[0, 2, 1])
        };
        let b_weight = half_weight(
            checkpoint,
            &format!("{prefix}.b_proj.weight"),
            Some(&[config.num_attention_heads, config.hidden_size]),
        )?
        .astype(Dtype::Float32)?
        .transpose(&[1, 0])?;
        let a_log = crate::linear::checkpoint_array(checkpoint, &format!("{prefix}.A_log"))?
            .astype(Dtype::Float32)?;
        let dt_bias = crate::linear::checkpoint_array(checkpoint, &format!("{prefix}.dt_bias"))?
            .astype(Dtype::Float32)?
            .reshape(&[config.num_attention_heads, config.head_dim])?;
        for value in [&b_weight, &a_log, &dt_bias] {
            value.eval()?;
        }
        Ok(Self {
            q: Projection::load(
                checkpoint,
                &format!("{prefix}.q_proj"),
                config.hidden_size,
                projection,
                false,
            )?,
            k: Projection::load(
                checkpoint,
                &format!("{prefix}.k_proj"),
                config.hidden_size,
                projection,
                false,
            )?,
            v: Projection::load(
                checkpoint,
                &format!("{prefix}.v_proj"),
                config.hidden_size,
                projection,
                false,
            )?,
            f: Projection::load(
                checkpoint,
                &format!("{prefix}.f_proj"),
                config.hidden_size,
                projection,
                false,
            )?,
            b_weight,
            output_gate: Projection::load(
                checkpoint,
                &format!("{prefix}.g_proj"),
                config.hidden_size,
                projection,
                false,
            )?,
            output_norm: norm(
                checkpoint,
                &format!("{prefix}.o_norm.weight"),
                config.head_dim,
            )?,
            output: Projection::load(
                checkpoint,
                &format!("{prefix}.o_proj"),
                projection,
                config.hidden_size,
                false,
            )?,
            q_conv: conv("q")?,
            k_conv: conv("k")?,
            v_conv: conv("v")?,
            a_log,
            dt_bias,
            conv_q: None,
            conv_k: None,
            conv_v: None,
            recurrent: None,
            heads: config.num_attention_heads,
            dim: config.head_dim,
            conv_length: config.short_conv_kernel_size,
            eps: config.rms_norm_eps,
            lower_bound: config.kda_lower_bound,
        })
    }

    fn causal_conv(
        value: Array,
        weight: &Array,
        state: &mut Option<Array>,
        length: i32,
    ) -> Result<Array> {
        let width = value.shape()[2];
        let previous = match state {
            Some(previous) => previous.try_clone()?,
            None => Array::zeros_dtype(&[1, length - 1, width], Dtype::Float16)?,
        };
        let input = Array::concatenate(&[&previous, &value], 1)?;
        *state = Some(input.slice(1, 1, length)?);
        input.conv1d(weight, width)?.silu()
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let projection = self.heads * self.dim;
        let q = Self::causal_conv(
            self.q.forward(x)?,
            &self.q_conv,
            &mut self.conv_q,
            self.conv_length,
        )?
        .reshape(&[1, 1, self.heads, self.dim])?
        .rms_norm_without_weight(self.eps / self.dim as f32)?
        .scalar_mul(1.0 / self.dim as f32)?;
        let k = Self::causal_conv(
            self.k.forward(x)?,
            &self.k_conv,
            &mut self.conv_k,
            self.conv_length,
        )?
        .reshape(&[1, 1, self.heads, self.dim])?
        .rms_norm_without_weight(self.eps / self.dim as f32)?
        .scalar_mul(1.0 / (self.dim as f32).sqrt())?;
        let v = Self::causal_conv(
            self.v.forward(x)?,
            &self.v_conv,
            &mut self.conv_v,
            self.conv_length,
        )?
        .reshape(&[1, 1, self.heads, self.dim])?;
        let raw_gate = self
            .f
            .forward(x)?
            .reshape(&[1, 1, self.heads, self.dim])?
            .astype(Dtype::Float32)?;
        let decay = raw_gate
            .add(&self.dt_bias)?
            .mul(&self.a_log.exp()?.reshape(&[self.heads, 1])?)?
            .sigmoid()?
            .scalar_mul(self.lower_bound)?
            .exp()?;
        let beta = x
            .astype(Dtype::Float32)?
            .matmul(&self.b_weight)?
            .sigmoid()?
            .astype(Dtype::Float16)?;
        let state = match &self.recurrent {
            Some(state) => state.try_clone()?,
            None => Array::zeros_dtype(&[1, self.heads, self.dim, self.dim], Dtype::Float32)?,
        };
        let (output, state) = gated_delta::step_vector(&q, &k, &v, &decay, &beta, &state)?;
        self.recurrent = Some(state);
        let output = output.rms_norm(&self.output_norm, self.eps)?;
        let gate = self
            .output_gate
            .forward(x)?
            .reshape(&[1, 1, self.heads, self.dim])?
            .astype(Dtype::Float32)?
            .sigmoid()?
            .astype(Dtype::Float16)?;
        self.output
            .forward(&output.mul(&gate)?.reshape(&[1, 1, projection])?)
    }

    fn eval_state(&self) -> Result<()> {
        for value in [&self.conv_q, &self.conv_k, &self.conv_v, &self.recurrent] {
            value.as_ref().context("missing Ling KDA state")?.eval()?;
        }
        Ok(())
    }

    fn reset(&mut self) {
        self.conv_q = None;
        self.conv_k = None;
        self.conv_v = None;
        self.recurrent = None;
    }
}

struct Mla {
    q_a: Projection,
    q_a_norm: Array,
    q_b: Projection,
    kv_a: Projection,
    kv_a_norm: Array,
    embed_q: Array,
    unembed_out: Array,
    gate: Projection,
    output: Projection,
    kv_cache: Option<Array>,
    rope_cache: Option<Array>,
    heads: i32,
    kv_rank: i32,
    nope_dim: i32,
    rope_dim: i32,
    value_dim: i32,
    theta: f32,
    eps: f32,
}

impl Mla {
    fn load(checkpoint: &Checkpoint, prefix: &str, config: &Config) -> Result<Self> {
        let qk_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
        let kv_b = half_weight(
            checkpoint,
            &format!("{prefix}.kv_b_proj.weight"),
            Some(&[
                config.num_attention_heads * (config.qk_nope_head_dim + config.v_head_dim),
                config.kv_lora_rank,
            ]),
        )?
        .reshape(&[
            config.num_attention_heads,
            config.qk_nope_head_dim + config.v_head_dim,
            config.kv_lora_rank,
        ])?;
        let embed_q = kv_b.slice(1, 0, config.qk_nope_head_dim)?;
        let unembed_out = kv_b
            .slice(
                1,
                config.qk_nope_head_dim,
                config.qk_nope_head_dim + config.v_head_dim,
            )?
            .transpose(&[0, 2, 1])?;
        embed_q.eval()?;
        unembed_out.eval()?;
        Ok(Self {
            q_a: Projection::load(
                checkpoint,
                &format!("{prefix}.q_a_proj"),
                config.hidden_size,
                config.q_lora_rank,
                false,
            )?,
            q_a_norm: norm(
                checkpoint,
                &format!("{prefix}.q_a_layernorm.weight"),
                config.q_lora_rank,
            )?,
            q_b: Projection::load(
                checkpoint,
                &format!("{prefix}.q_b_proj"),
                config.q_lora_rank,
                config.num_attention_heads * qk_dim,
                false,
            )?,
            kv_a: Projection::load(
                checkpoint,
                &format!("{prefix}.kv_a_proj_with_mqa"),
                config.hidden_size,
                config.kv_lora_rank + config.qk_rope_head_dim,
                false,
            )?,
            kv_a_norm: norm(
                checkpoint,
                &format!("{prefix}.kv_a_layernorm.weight"),
                config.kv_lora_rank,
            )?,
            embed_q,
            unembed_out,
            gate: Projection::load(
                checkpoint,
                &format!("{prefix}.g_proj"),
                config.hidden_size,
                config.num_attention_heads,
                false,
            )?,
            output: Projection::load(
                checkpoint,
                &format!("{prefix}.dense"),
                config.num_attention_heads * config.v_head_dim,
                config.hidden_size,
                false,
            )?,
            kv_cache: None,
            rope_cache: None,
            heads: config.num_attention_heads,
            kv_rank: config.kv_lora_rank,
            nope_dim: config.qk_nope_head_dim,
            rope_dim: config.qk_rope_head_dim,
            value_dim: config.v_head_dim,
            theta: config.rope_theta,
            eps: config.rms_norm_eps,
        })
    }

    fn forward(&mut self, x: &Array, offset: i32) -> Result<Array> {
        let qk_dim = self.nope_dim + self.rope_dim;
        let q = self
            .q_b
            .forward(&self.q_a.forward(x)?.rms_norm(&self.q_a_norm, self.eps)?)?
            .reshape(&[1, 1, self.heads, qk_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let q_nope = q.slice(3, 0, self.nope_dim)?;
        let q_rope = q
            .slice(3, self.nope_dim, qk_dim)?
            .rope(self.rope_dim, self.theta, offset)?;
        let compressed = self.kv_a.forward(x)?;
        let kv = compressed
            .slice(2, 0, self.kv_rank)?
            .rms_norm(&self.kv_a_norm, self.eps)?
            .reshape(&[1, 1, 1, self.kv_rank])?;
        let k_rope = compressed
            .slice(2, self.kv_rank, self.kv_rank + self.rope_dim)?
            .reshape(&[1, 1, 1, self.rope_dim])?
            .rope(self.rope_dim, self.theta, offset)?;
        let keys = match &self.kv_cache {
            Some(previous) => Array::concatenate(&[previous, &kv], 2)?,
            None => kv,
        };
        let rope_keys = match &self.rope_cache {
            Some(previous) => Array::concatenate(&[previous, &k_rope], 2)?,
            None => k_rope,
        };
        let scale = (qk_dim as f32).powf(-0.5);
        let pe_scores = q_rope
            .scalar_mul(scale)?
            .matmul(&rope_keys.transpose(&[0, 1, 3, 2])?)?;
        let embedded_q = q_nope.matmul(&self.embed_q)?;
        let attended = Array::sdpa_mask(&embedded_q, &keys, &keys, scale, &pe_scores)?
            .matmul(&self.unembed_out)?
            .transpose(&[0, 2, 1, 3])?;
        let gate = self
            .gate
            .forward(x)?
            .astype(Dtype::Float32)?
            .sigmoid()?
            .astype(Dtype::Float16)?
            .reshape(&[1, 1, self.heads, 1])?;
        self.kv_cache = Some(keys);
        self.rope_cache = Some(rope_keys);
        self.output.forward(
            &attended
                .mul(&gate)?
                .reshape(&[1, 1, self.heads * self.value_dim])?,
        )
    }

    fn eval_state(&self) -> Result<()> {
        self.kv_cache
            .as_ref()
            .context("missing Ling MLA cache")?
            .eval()?;
        self.rope_cache
            .as_ref()
            .context("missing Ling RoPE cache")?
            .eval()
    }

    fn reset(&mut self) {
        self.kv_cache = None;
        self.rope_cache = None;
    }
}

enum Attention {
    Kda(Box<Kda>),
    Mla(Box<Mla>),
}

impl Attention {
    fn forward(&mut self, x: &Array, offset: i32) -> Result<Array> {
        match self {
            Self::Kda(attention) => attention.forward(x),
            Self::Mla(attention) => attention.forward(x, offset),
        }
    }

    fn eval_state(&self) -> Result<()> {
        match self {
            Self::Kda(attention) => attention.eval_state(),
            Self::Mla(attention) => attention.eval_state(),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Kda(attention) => attention.reset(),
            Self::Mla(attention) => attention.reset(),
        }
    }
}

struct Layer {
    input_norm: Array,
    post_attention_norm: Array,
    attention: Attention,
    feed_forward: FeedForward,
    hidden: i32,
    eps: f32,
}

impl Layer {
    fn load(checkpoint: &Checkpoint, index: usize, config: &Config) -> Result<Self> {
        let prefix = format!("model.layers.{index}");
        let linear = is_kda_layer(index, config.layer_group_size);
        Ok(Self {
            input_norm: norm(
                checkpoint,
                &format!("{prefix}.input_layernorm.weight"),
                config.hidden_size,
            )?,
            post_attention_norm: norm(
                checkpoint,
                &format!("{prefix}.post_attention_layernorm.weight"),
                config.hidden_size,
            )?,
            attention: if linear {
                Attention::Kda(Box::new(Kda::load(
                    checkpoint,
                    &format!("{prefix}.attention"),
                    config,
                )?))
            } else {
                Attention::Mla(Box::new(Mla::load(
                    checkpoint,
                    &format!("{prefix}.attention"),
                    config,
                )?))
            },
            feed_forward: if index >= config.first_k_dense_replace {
                FeedForward::Sparse(Box::new(SparseMoe::load(
                    checkpoint,
                    &format!("{prefix}.mlp"),
                    config,
                )?))
            } else {
                FeedForward::Dense(Box::new(Mlp::load(
                    checkpoint,
                    &format!("{prefix}.mlp"),
                    config.hidden_size,
                    config.intermediate_size,
                )?))
            },
            hidden: config.hidden_size,
            eps: config.rms_norm_eps,
        })
    }

    fn forward(&mut self, x: &Array, offset: i32) -> Result<Array> {
        let hidden = x.add(
            &self
                .attention
                .forward(&x.rms_norm(&self.input_norm, self.eps)?, offset)?,
        )?;
        let feed_forward = self.feed_forward.forward(
            &hidden
                .rms_norm(&self.post_attention_norm, self.eps)?
                .reshape(&[1, self.hidden])?,
        )?;
        hidden.add(&feed_forward.reshape(&[1, 1, self.hidden])?)
    }
}

pub struct Ling {
    embeddings: Array,
    norm: Array,
    head: Projection,
    layers: Vec<Layer>,
    vocab: i32,
    context_limit: i32,
    eps: f32,
    offset: i32,
}

impl Ling {
    pub fn load(path: &Path) -> Result<Self> {
        let config: Config = serde_json::from_reader(File::open(path.join("config.json"))?)?;
        ensure!(
            config.model_type == "bailing_hybrid"
                && config.hidden_size > 0
                && config.vocab_size > 0
                && config.num_hidden_layers > 0
                && config.layer_group_size > 0
                && config.num_attention_heads > 0
                && config.head_dim == 128
                && config.q_lora_rank > 0
                && config.kv_lora_rank > 0
                && config.qk_nope_head_dim > 0
                && config.qk_rope_head_dim > 0
                && config.v_head_dim == 128
                && config.short_conv_kernel_size > 1
                && config.kda_safe_gate
                && config.no_kda_lora
                && config.rope_interleave
                && !config.tie_word_embeddings
                && config.num_experts > 0
                && config.moe_intermediate_size > 0
                && config.num_experts_per_tok > 0
                && config.num_experts_per_tok <= config.num_experts as usize,
            "unsupported Ling/Bailing V3 configuration"
        );
        let checkpoint = checkpoint::inspect(path)?;
        let embeddings = half_weight(
            &checkpoint,
            "model.word_embeddings.weight",
            Some(&[config.vocab_size, config.hidden_size]),
        )?;
        let norm = norm(&checkpoint, "model.norm.weight", config.hidden_size)?;
        let head = Projection::load(
            &checkpoint,
            "lm_head",
            config.hidden_size,
            config.vocab_size,
            false,
        )?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(Layer::load(&checkpoint, index, &config)?);
        }
        Ok(Self {
            embeddings,
            norm,
            head,
            layers,
            vocab: config.vocab_size,
            context_limit: config.max_position_embeddings,
            eps: config.rms_norm_eps,
            offset: 0,
        })
    }

    pub fn forward(&mut self, token: u32) -> Result<Array> {
        match self.run(token) {
            Ok(logits) => Ok(logits),
            Err(error) => {
                self.reset();
                Err(error.context("Ling forward failed; its cache was reset"))
            }
        }
    }

    fn run(&mut self, token: u32) -> Result<Array> {
        ensure!(
            token < self.vocab as u32,
            "Ling token is outside vocabulary"
        );
        ensure!(self.offset < self.context_limit, "Ling context is full");
        let id = Array::from_i32(&[token as i32], &[1, 1])?;
        let mut hidden = self.embeddings.take(&id, 0)?;
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden, self.offset)?;
            hidden.eval()?;
            layer.attention.eval_state()?;
        }
        let logits = self.head.forward(&hidden.rms_norm(&self.norm, self.eps)?)?;
        self.offset += 1;
        Ok(logits)
    }

    pub fn context_limit(&self) -> i32 {
        self.context_limit
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        for layer in &mut self.layers {
            layer.attention.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ling_tiny_config() {
        assert!(is_kda_layer(0, 4));
        assert!(!is_kda_layer(3, 4));
        assert!(!is_kda_layer(23, 4));
    }
}
