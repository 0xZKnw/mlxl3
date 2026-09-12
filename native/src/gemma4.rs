//! Native Gemma 4 26B-A4B decode inference using serialized EXL3 weights.
use crate::{
    array::Array,
    checkpoint::{self, Checkpoint},
    lfm2::{Projection, half_weight},
    moe::Exl3SwitchGlu,
    router,
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{fs::File, path::Path};

#[derive(Deserialize)]
struct RootConfig {
    model_type: String,
    text_config: Config,
}

#[derive(Deserialize)]
struct Config {
    hidden_size: i32,
    num_hidden_layers: usize,
    intermediate_size: i32,
    num_attention_heads: i32,
    head_dim: i32,
    global_head_dim: i32,
    num_key_value_heads: i32,
    num_global_key_value_heads: i32,
    num_kv_shared_layers: usize,
    hidden_size_per_layer_input: i32,
    vocab_size: i32,
    max_position_embeddings: i32,
    rms_norm_eps: f32,
    sliding_window: i32,
    attention_k_eq_v: bool,
    final_logit_softcapping: f32,
    enable_moe_block: bool,
    num_experts: i32,
    top_k_experts: usize,
    moe_intermediate_size: i32,
    layer_types: Vec<String>,
    rope_parameters: RopeParameters,
    #[serde(default)]
    rope_traditional: bool,
}

#[derive(Deserialize)]
struct RopeParameters {
    full_attention: Rope,
    sliding_attention: Rope,
}

#[derive(Deserialize)]
struct Rope {
    rope_theta: f32,
    #[serde(default = "full_rotation")]
    partial_rotary_factor: f32,
    #[serde(default = "full_rotation")]
    factor: f32,
}

fn full_rotation() -> f32 {
    1.0
}

fn norm(checkpoint: &Checkpoint, name: &str, width: i32) -> Result<Array> {
    half_weight(checkpoint, name, Some(&[width]))
}

struct DenseMlp {
    gate: Projection,
    up: Projection,
    down: Projection,
}

impl DenseMlp {
    fn load(checkpoint: &Checkpoint, prefix: &str, hidden: i32, intermediate: i32) -> Result<Self> {
        Ok(Self {
            gate: Projection::load_logical(
                checkpoint,
                &format!("{prefix}.gate_proj"),
                hidden,
                intermediate,
                false,
            )?,
            up: Projection::load_logical(
                checkpoint,
                &format!("{prefix}.up_proj"),
                hidden,
                intermediate,
                false,
            )?,
            down: Projection::load_logical(
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
            .forward(&self.gate.forward(x)?.geglu(&self.up.forward(x)?)?)
    }

    fn forward_trace(&self, x: &Array, trace: &mut Vec<(String, Array)>) -> Result<Array> {
        let gate = self.gate.forward(x)?;
        let up = self.up.forward(x)?;
        let hidden = gate.geglu(&up)?;
        trace.push(("layer0.dense_gate".into(), gate));
        trace.push(("layer0.dense_up".into(), up));
        trace.push(("layer0.dense_hidden".into(), hidden.try_clone()?));
        self.down.forward(&hidden)
    }
}

struct Moe {
    router_norm: Array,
    router: Projection,
    expert_scale: Option<Array>,
    experts: Exl3SwitchGlu,
    hidden: i32,
    top_k: usize,
    eps: f32,
}

impl Moe {
    fn load(checkpoint: &Checkpoint, prefix: &str, config: &Config) -> Result<Self> {
        let router_prefix = format!("{prefix}.router");
        let router_norm = norm(
            checkpoint,
            &format!("{router_prefix}.scale"),
            config.hidden_size,
        )?
        .scalar_mul((config.hidden_size as f32).powf(-0.5))?;
        router_norm.eval()?;
        let scale_name = format!("{router_prefix}.per_expert_scale");
        let expert_scale = if checkpoint.tensors.contains_key(&scale_name) {
            Some(norm(checkpoint, &scale_name, config.num_experts)?)
        } else {
            None
        };
        Ok(Self {
            router_norm,
            router: Projection::load(
                checkpoint,
                &format!("{router_prefix}.proj"),
                config.hidden_size,
                config.num_experts,
                false,
            )?,
            expert_scale,
            experts: Exl3SwitchGlu::from_gemma_checkpoint(
                checkpoint,
                &format!("{prefix}.experts"),
                config.num_experts,
                config.top_k_experts as i32,
                config.moe_intermediate_size,
            )?,
            hidden: config.hidden_size,
            top_k: config.top_k_experts,
            eps: config.rms_norm_eps,
        })
    }

    fn forward(&self, router_x: &Array, expert_x: &Array) -> Result<Array> {
        self.forward_inner(router_x, expert_x, None)
    }

    fn forward_trace(
        &self,
        router_x: &Array,
        expert_x: &Array,
        trace: &mut Vec<(String, Array)>,
    ) -> Result<Array> {
        self.forward_inner(router_x, expert_x, Some(trace))
    }

    fn forward_inner(
        &self,
        router_x: &Array,
        expert_x: &Array,
        trace: Option<&mut Vec<(String, Array)>>,
    ) -> Result<Array> {
        ensure!(
            router_x.shape() == [1, self.hidden] && expert_x.shape() == [1, self.hidden],
            "Gemma MoE expects one token"
        );
        let router_input = router_x.rms_norm(&self.router_norm, self.eps)?;
        let logits = self.router.forward(&router_input)?;
        let (selected, raw_scores) = router::topk(&logits, self.top_k, false)?;
        let mut scores = raw_scores.softmax()?;
        if let Some(scale) = &self.expert_scale {
            scores = scores.mul(&scale.take(&selected, 0)?)?;
        }
        if let Some(trace) = trace {
            trace.push(("layer0.router_input".into(), router_input.try_clone()?));
            trace.push(("layer0.router_logits".into(), logits.try_clone()?));
            trace.push(("layer0.router_indices".into(), selected.try_clone()?));
            trace.push(("layer0.router_scores".into(), scores.try_clone()?));
        }
        self.experts.forward(expert_x, &selected, &scores)
    }
}

struct Attention {
    q: Projection,
    k: Projection,
    v: Option<Projection>,
    output: Projection,
    q_norm: Array,
    k_norm: Array,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    rope_freqs: Option<Array>,
    rope_base: f32,
    eps: f32,
    window: Option<i32>,
    keys: Option<Array>,
    values: Option<Array>,
}

impl Attention {
    fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        config: &Config,
        layer_type: &str,
    ) -> Result<Self> {
        let sliding = layer_type == "sliding_attention";
        ensure!(
            sliding || layer_type == "full_attention",
            "unsupported Gemma attention type"
        );
        let head_dim = if sliding {
            config.head_dim
        } else {
            config.global_head_dim
        };
        let kv_heads = if sliding {
            config.num_key_value_heads
        } else {
            config.num_global_key_value_heads
        };
        let rope = if sliding {
            &config.rope_parameters.sliding_attention
        } else {
            &config.rope_parameters.full_attention
        };
        let rope_dims = (head_dim as f32 * rope.partial_rotary_factor) as i32;
        let use_k_eq_v = config.attention_k_eq_v && !sliding;
        ensure!(
            config.num_attention_heads > 0
                && kv_heads > 0
                && config.num_attention_heads % kv_heads == 0
                && head_dim > 0
                && rope_dims > 0
                && rope_dims <= head_dim
                && rope_dims % 2 == 0,
            "invalid Gemma attention dimensions"
        );
        let rope_freqs = if sliding {
            None
        } else {
            let frequencies: Vec<_> = (0..head_dim)
                .step_by(2)
                .map(|index| {
                    if index < rope_dims {
                        rope.factor * rope.rope_theta.powf(index as f32 / head_dim as f32)
                    } else {
                        f32::INFINITY
                    }
                })
                .collect();
            let frequencies = Array::from_f32(&frequencies, &[head_dim / 2])?;
            frequencies.eval()?;
            Some(frequencies)
        };
        Ok(Self {
            q: Projection::load_logical(
                checkpoint,
                &format!("{prefix}.q_proj"),
                config.hidden_size,
                config.num_attention_heads * head_dim,
                false,
            )?,
            k: Projection::load_logical(
                checkpoint,
                &format!("{prefix}.k_proj"),
                config.hidden_size,
                kv_heads * head_dim,
                false,
            )?,
            v: if use_k_eq_v {
                None
            } else {
                Some(Projection::load_logical(
                    checkpoint,
                    &format!("{prefix}.v_proj"),
                    config.hidden_size,
                    kv_heads * head_dim,
                    false,
                )?)
            },
            output: Projection::load_logical(
                checkpoint,
                &format!("{prefix}.o_proj"),
                config.num_attention_heads * head_dim,
                config.hidden_size,
                false,
            )?,
            q_norm: norm(checkpoint, &format!("{prefix}.q_norm.weight"), head_dim)?,
            k_norm: norm(checkpoint, &format!("{prefix}.k_norm.weight"), head_dim)?,
            heads: config.num_attention_heads,
            kv_heads,
            head_dim,
            rope_freqs,
            rope_base: rope.rope_theta,
            eps: config.rms_norm_eps,
            window: sliding.then_some(config.sliding_window),
            keys: None,
            values: None,
        })
    }

    fn forward(&mut self, x: &Array, offset: i32) -> Result<Array> {
        let q = self
            .q
            .forward(x)?
            .reshape(&[1, 1, self.heads, self.head_dim])?
            .rms_norm(&self.q_norm, self.eps)?
            .transpose(&[0, 2, 1, 3])?;
        let q = self.apply_rope(&q, offset)?;
        let raw_k = self
            .k
            .forward(x)?
            .reshape(&[1, 1, self.kv_heads, self.head_dim])?;
        let k = raw_k
            .rms_norm(&self.k_norm, self.eps)?
            .transpose(&[0, 2, 1, 3])?;
        let k = self.apply_rope(&k, offset)?;
        let v = match &self.v {
            Some(value) => value
                .forward(x)?
                .reshape(&[1, 1, self.kv_heads, self.head_dim])?,
            None => raw_k,
        }
        .rms_norm_without_weight(self.eps)?
        .transpose(&[0, 2, 1, 3])?;
        let mut keys = match &self.keys {
            Some(previous) => Array::concatenate(&[previous, &k], 2)?,
            None => k,
        };
        let mut values = match &self.values {
            Some(previous) => Array::concatenate(&[previous, &v], 2)?,
            None => v,
        };
        if let Some(window) = self.window
            && keys.shape()[2] > window
        {
            let start = keys.shape()[2] - window;
            keys = keys.slice(2, start, keys.shape()[2])?;
            values = values.slice(2, start, values.shape()[2])?;
        }
        let attended = Array::sdpa(&q, &keys, &values, 1.0, false)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, 1, self.heads * self.head_dim])?;
        self.keys = Some(keys);
        self.values = Some(values);
        self.output.forward(&attended)
    }

    fn apply_rope(&self, value: &Array, offset: i32) -> Result<Array> {
        match &self.rope_freqs {
            Some(frequencies) => value.rope_with_freqs(self.head_dim, offset, frequencies),
            None => value.rope(self.head_dim, self.rope_base, offset),
        }
    }

    fn eval_state(&self) -> Result<()> {
        self.keys.as_ref().context("missing Gemma keys")?.eval()?;
        self.values.as_ref().context("missing Gemma values")?.eval()
    }

    fn reset(&mut self) {
        self.keys = None;
        self.values = None;
    }
}

struct Layer {
    input_norm: Array,
    post_attention_norm: Array,
    pre_ff_norm: Array,
    pre_ff_norm_2: Array,
    post_ff_norm: Array,
    post_ff_norm_1: Array,
    post_ff_norm_2: Array,
    attention: Attention,
    dense: DenseMlp,
    moe: Moe,
    layer_scalar: Option<Array>,
    hidden: i32,
    eps: f32,
}

impl Layer {
    fn load(checkpoint: &Checkpoint, index: usize, config: &Config) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{index}");
        let layer_scalar_name = format!("{prefix}.layer_scalar");
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
            pre_ff_norm: norm(
                checkpoint,
                &format!("{prefix}.pre_feedforward_layernorm.weight"),
                config.hidden_size,
            )?,
            pre_ff_norm_2: norm(
                checkpoint,
                &format!("{prefix}.pre_feedforward_layernorm_2.weight"),
                config.hidden_size,
            )?,
            post_ff_norm: norm(
                checkpoint,
                &format!("{prefix}.post_feedforward_layernorm.weight"),
                config.hidden_size,
            )?,
            post_ff_norm_1: norm(
                checkpoint,
                &format!("{prefix}.post_feedforward_layernorm_1.weight"),
                config.hidden_size,
            )?,
            post_ff_norm_2: norm(
                checkpoint,
                &format!("{prefix}.post_feedforward_layernorm_2.weight"),
                config.hidden_size,
            )?,
            attention: Attention::load(
                checkpoint,
                &format!("{prefix}.self_attn"),
                config,
                &config.layer_types[index],
            )?,
            dense: DenseMlp::load(
                checkpoint,
                &format!("{prefix}.mlp"),
                config.hidden_size,
                config.intermediate_size,
            )?,
            moe: Moe::load(checkpoint, &prefix, config)?,
            layer_scalar: checkpoint
                .tensors
                .contains_key(&layer_scalar_name)
                .then(|| half_weight(checkpoint, &layer_scalar_name, Some(&[1])))
                .transpose()?,
            hidden: config.hidden_size,
            eps: config.rms_norm_eps,
        })
    }

    fn forward(&mut self, x: &Array, offset: i32) -> Result<Array> {
        self.forward_inner(x, offset, None)
    }

    fn forward_trace(
        &mut self,
        x: &Array,
        offset: i32,
        trace: &mut Vec<(String, Array)>,
    ) -> Result<Array> {
        self.forward_inner(x, offset, Some(trace))
    }

    fn forward_inner(
        &mut self,
        x: &Array,
        offset: i32,
        mut trace: Option<&mut Vec<(String, Array)>>,
    ) -> Result<Array> {
        fn capture(
            trace: &mut Option<&mut Vec<(String, Array)>>,
            name: &str,
            value: &Array,
        ) -> Result<()> {
            if let Some(trace) = trace.as_deref_mut() {
                trace.push((name.into(), value.try_clone()?));
            }
            Ok(())
        }

        ensure!(
            x.shape() == [1, 1, self.hidden],
            "Gemma layer expects one token"
        );
        let input_norm = x.rms_norm(&self.input_norm, self.eps)?;
        capture(&mut trace, "layer0.input_norm", &input_norm)?;
        let attention = self.attention.forward(&input_norm, offset)?;
        capture(&mut trace, "layer0.attention", &attention)?;
        let attention_norm = attention.rms_norm(&self.post_attention_norm, self.eps)?;
        capture(&mut trace, "layer0.attention_norm", &attention_norm)?;
        let hidden = x.add(&attention_norm)?;
        capture(&mut trace, "layer0.after_attention", &hidden)?;
        let dense_input = hidden.rms_norm(&self.pre_ff_norm, self.eps)?;
        capture(&mut trace, "layer0.dense_input", &dense_input)?;
        let dense_raw = match trace.as_deref_mut() {
            Some(trace) => self.dense.forward_trace(&dense_input, trace)?,
            None => self.dense.forward(&dense_input)?,
        };
        capture(&mut trace, "layer0.dense_raw", &dense_raw)?;
        let dense = dense_raw.rms_norm(&self.post_ff_norm_1, self.eps)?;
        capture(&mut trace, "layer0.dense_norm", &dense)?;
        let routed_input = hidden
            .rms_norm(&self.pre_ff_norm_2, self.eps)?
            .reshape(&[1, self.hidden])?;
        capture(&mut trace, "layer0.routed_input", &routed_input)?;
        let router_x = hidden.reshape(&[1, self.hidden])?;
        let routed = match trace.as_deref_mut() {
            Some(trace) => self.moe.forward_trace(&router_x, &routed_input, trace)?,
            None => self.moe.forward(&router_x, &routed_input)?,
        }
        .reshape(&[1, 1, self.hidden])?;
        capture(&mut trace, "layer0.routed_raw", &routed)?;
        let routed = routed.rms_norm(&self.post_ff_norm_2, self.eps)?;
        capture(&mut trace, "layer0.routed_norm", &routed)?;
        let merged = dense.add(&routed)?;
        capture(&mut trace, "layer0.merged", &merged)?;
        let feedforward = merged.rms_norm(&self.post_ff_norm, self.eps)?;
        capture(&mut trace, "layer0.feedforward_norm", &feedforward)?;
        let output = hidden.add(&feedforward)?;
        capture(&mut trace, "layer0.output", &output)?;
        match &self.layer_scalar {
            Some(scale) => output.mul(scale),
            None => Ok(output),
        }
    }
}

pub struct Gemma4 {
    embeddings: Array,
    norm: Array,
    head: Projection,
    layers: Vec<Layer>,
    hidden: i32,
    vocab: i32,
    context_limit: i32,
    eps: f32,
    softcap: f32,
    offset: i32,
}

impl Gemma4 {
    pub fn load(path: &Path) -> Result<Self> {
        let root: RootConfig = serde_json::from_reader(File::open(path.join("config.json"))?)?;
        let config = root.text_config;
        ensure!(root.model_type == "gemma4", "expected Gemma 4");
        ensure!(
            config.hidden_size > 0
                && config.vocab_size > 0
                && config.num_hidden_layers > 0
                && config.layer_types.len() == config.num_hidden_layers
                && config.max_position_embeddings > 0
                && config.rms_norm_eps > 0.0
                && config.sliding_window > 0
                && config.num_kv_shared_layers == 0
                && config.hidden_size_per_layer_input == 0
                && config.enable_moe_block
                && config.num_experts > 0
                && config.top_k_experts > 0
                && config.top_k_experts <= config.num_experts as usize
                && config.moe_intermediate_size > 0
                && config.final_logit_softcapping > 0.0
                && !config.rope_traditional,
            "unsupported Gemma 4 configuration"
        );
        let checkpoint = checkpoint::inspect(path)?;
        let embeddings = half_weight(
            &checkpoint,
            "model.language_model.embed_tokens.weight",
            Some(&[config.vocab_size, config.hidden_size]),
        )?;
        let norm = norm(
            &checkpoint,
            "model.language_model.norm.weight",
            config.hidden_size,
        )?;
        let head = Projection::load_logical(
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
            hidden: config.hidden_size,
            vocab: config.vocab_size,
            context_limit: config.max_position_embeddings,
            eps: config.rms_norm_eps,
            softcap: config.final_logit_softcapping,
            offset: 0,
        })
    }

    pub fn forward(&mut self, token: u32) -> Result<Array> {
        match self.run(token, None) {
            Ok(logits) => Ok(logits),
            Err(error) => {
                self.reset();
                Err(error.context("Gemma forward failed; its cache was reset"))
            }
        }
    }

    pub fn forward_trace(&mut self, token: u32) -> Result<(Array, Vec<(String, Array)>)> {
        let mut trace = Vec::with_capacity(self.layers.len() + 1);
        match self.run(token, Some(&mut trace)) {
            Ok(logits) => Ok((logits, trace)),
            Err(error) => {
                self.reset();
                Err(error.context("Gemma trace failed; its cache was reset"))
            }
        }
    }

    fn run(&mut self, token: u32, mut trace: Option<&mut Vec<(String, Array)>>) -> Result<Array> {
        ensure!(
            token < self.vocab as u32,
            "Gemma token is outside vocabulary"
        );
        ensure!(self.offset < self.context_limit, "Gemma context is full");
        let id = Array::from_i32(&[token as i32], &[1, 1])?;
        let mut hidden = self
            .embeddings
            .take(&id, 0)?
            .scalar_mul((self.hidden as f32).sqrt())?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.push(("model.embedding".into(), hidden.try_clone()?));
        }
        for (index, layer) in self.layers.iter_mut().enumerate() {
            hidden = if index == 0 {
                match trace.as_deref_mut() {
                    Some(trace) => layer.forward_trace(&hidden, self.offset, trace)?,
                    None => layer.forward(&hidden, self.offset)?,
                }
            } else {
                layer.forward(&hidden, self.offset)?
            };
            hidden.eval()?;
            layer.attention.eval_state()?;
            if let Some(trace) = trace.as_deref_mut() {
                trace.push((format!("model.layers.{index}.hidden"), hidden.try_clone()?));
            }
        }
        let normalized = hidden.rms_norm(&self.norm, self.eps)?;
        if let Some(trace) = trace.as_deref_mut() {
            trace.push(("model.norm".into(), normalized.try_clone()?));
        }
        let raw_logits = self.head.forward(&normalized)?;
        if let Some(trace) = trace {
            trace.push(("model.raw_logits".into(), raw_logits.try_clone()?));
        }
        let logits = raw_logits.softcap(self.softcap)?;
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
    fn parses_gemma_rope_defaults() {
        let rope: Rope = serde_json::from_str(r#"{"rope_theta":10000}"#).unwrap();
        assert_eq!(rope.partial_rotary_factor, 1.0);
    }
}
