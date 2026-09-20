//! Native Qwen3.5 dense/MoE inference, validated against MLXL3 layerwise.
use crate::{
    array::{Array, Dtype},
    checkpoint::Checkpoint,
    gated_delta,
    lfm2::{Projection, half_weight},
    linear::{Exl3Group, Exl3Linear, checkpoint_array},
    moe::Exl3SwitchGlu,
    router,
};
use anyhow::{Context, Result, ensure};
use half::f16;
use serde::Deserialize;
use std::{fs::File, path::Path};

pub(crate) enum ProjectionBundle {
    Grouped(Box<Exl3Group>),
    Separate(Vec<Projection>),
}

impl ProjectionBundle {
    pub(crate) fn load(
        checkpoint: &Checkpoint,
        input: i32,
        outputs: Vec<(String, i32)>,
    ) -> Result<Self> {
        if outputs
            .iter()
            .all(|(prefix, _)| checkpoint.modules.contains(prefix))
        {
            let linears = outputs
                .iter()
                .map(|(prefix, _)| Exl3Linear::from_checkpoint(checkpoint, prefix))
                .collect::<Result<Vec<_>>>()?;
            if Exl3Group::compatible(&linears.iter().collect::<Vec<_>>()) {
                return Ok(Self::Grouped(Box::new(Exl3Group::new(linears)?)));
            }
        }
        Ok(Self::Separate(
            outputs
                .into_iter()
                .map(|(prefix, output)| Projection::load(checkpoint, &prefix, input, output, false))
                .collect::<Result<Vec<_>>>()?,
        ))
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Vec<Array>> {
        match self {
            Self::Grouped(group) => group.forward(x),
            Self::Separate(projections) => projections.iter().map(|p| p.forward(x)).collect(),
        }
    }
}

fn sanitized_norm(checkpoint: &Checkpoint, name: &str, expected: &[i32]) -> Result<Array> {
    let weight = checkpoint_array(checkpoint, name)?;
    ensure!(
        weight.shape() == expected
            && matches!(
                weight.dtype(),
                Dtype::Float16 | Dtype::BFloat16 | Dtype::Float32
            ),
        "{name}: invalid Qwen norm weight"
    );
    let one = Array::from_f16_bits(&[f16::ONE.to_bits()], &[])?.astype(weight.dtype())?;
    let weight = weight.add(&one)?.astype(Dtype::Float16)?;
    weight.eval()?;
    Ok(weight)
}

pub struct Moe {
    gate: Projection,
    experts: Exl3SwitchGlu,
    shared_inputs: ProjectionBundle,
    shared_down: Projection,
    shared_multiplier: Projection,
    hidden: i32,
    top_k: usize,
}

impl Moe {
    pub fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        hidden: i32,
        expert_hidden: i32,
        num_experts: i32,
        top_k: usize,
    ) -> Result<Self> {
        ensure!(
            top_k > 0 && top_k <= num_experts as usize,
            "invalid MoE top-k"
        );
        Ok(Self {
            gate: Projection::load(
                checkpoint,
                &format!("{prefix}.gate"),
                hidden,
                num_experts,
                false,
            )?,
            experts: Exl3SwitchGlu::from_checkpoint(
                checkpoint,
                &format!("{prefix}.experts"),
                num_experts,
                top_k as i32,
            )?,
            shared_inputs: ProjectionBundle::load(
                checkpoint,
                hidden,
                vec![
                    (format!("{prefix}.shared_expert.gate_proj"), expert_hidden),
                    (format!("{prefix}.shared_expert.up_proj"), expert_hidden),
                ],
            )?,
            shared_down: Projection::load(
                checkpoint,
                &format!("{prefix}.shared_expert.down_proj"),
                expert_hidden,
                hidden,
                false,
            )?,
            shared_multiplier: Projection::load(
                checkpoint,
                &format!("{prefix}.shared_expert_gate"),
                hidden,
                1,
                false,
            )?,
            hidden,
            top_k,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        ensure!(
            x.shape().len() == 2 && x.shape()[0] > 0 && x.shape()[1] == self.hidden,
            "invalid Qwen MoE input"
        );
        let probabilities = self.gate.forward(x)?.softmax_precise()?;
        let (selected, scores) = router::topk(&probabilities, self.top_k, true)?;
        let routed = self.experts.forward(x, &selected, &scores)?;
        let shared_inputs = self.shared_inputs.forward(x)?;
        let shared = self
            .shared_down
            .forward(&shared_inputs[0].swiglu(&shared_inputs[1])?)?;
        routed.add(&shared.mul(&self.shared_multiplier.forward(x)?.sigmoid()?)?)
    }
}

struct DenseMlp {
    inputs: ProjectionBundle,
    down: Projection,
}

enum Mlp {
    Dense(Box<DenseMlp>),
    Moe(Box<Moe>),
}

impl Mlp {
    fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        hidden: i32,
        intermediate: i32,
        experts: i32,
        top_k: usize,
    ) -> Result<Self> {
        if experts > 0 {
            return Ok(Self::Moe(Box::new(Moe::load(
                checkpoint,
                prefix,
                hidden,
                intermediate,
                experts,
                top_k,
            )?)));
        }
        ensure!(top_k == 0, "dense Qwen MLP cannot route experts");
        Ok(Self::Dense(Box::new(DenseMlp {
            inputs: ProjectionBundle::load(
                checkpoint,
                hidden,
                vec![
                    (format!("{prefix}.gate_proj"), intermediate),
                    (format!("{prefix}.up_proj"), intermediate),
                ],
            )?,
            down: Projection::load(
                checkpoint,
                &format!("{prefix}.down_proj"),
                intermediate,
                hidden,
                false,
            )?,
        })))
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Dense(mlp) => {
                let inputs = mlp.inputs.forward(x)?;
                mlp.down.forward(&inputs[0].swiglu(&inputs[1])?)
            }
            Self::Moe(mlp) => mlp.forward(x),
        }
    }
}

pub struct GatedDelta {
    inputs: ProjectionBundle,
    b: Projection,
    a: Projection,
    output: Projection,
    conv_weight: Array,
    norm: Array,
    a_log: Array,
    dt_bias: Array,
    conv_state: Option<Array>,
    recurrent_state: Option<Array>,
    hidden: i32,
    key_heads: i32,
    value_heads: i32,
    key_dim: i32,
    value_dim: i32,
    conv_length: i32,
    eps: f32,
}

pub struct LinearLayer {
    input_norm: Array,
    post_norm: Array,
    attention: GatedDelta,
    mlp: Mlp,
    hidden: i32,
    eps: f32,
}

pub struct Attention {
    qkv: ProjectionBundle,
    output: Projection,
    q_norm: Array,
    k_norm: Array,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    rope_dims: i32,
    theta: f32,
    eps: f32,
    keys: Option<Array>,
    values: Option<Array>,
}

impl Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        hidden: i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        rope_dims: i32,
        theta: f32,
        eps: f32,
    ) -> Result<Self> {
        ensure!(
            hidden > 0
                && heads > 0
                && kv_heads > 0
                && heads % kv_heads == 0
                && head_dim > 0
                && rope_dims > 0
                && rope_dims <= head_dim
                && rope_dims % 2 == 0
                && theta.is_finite()
                && theta > 0.,
            "invalid Qwen attention dimensions"
        );
        Ok(Self {
            qkv: ProjectionBundle::load(
                checkpoint,
                hidden,
                vec![
                    (format!("{prefix}.q_proj"), heads * head_dim * 2),
                    (format!("{prefix}.k_proj"), kv_heads * head_dim),
                    (format!("{prefix}.v_proj"), kv_heads * head_dim),
                ],
            )?,
            output: Projection::load(
                checkpoint,
                &format!("{prefix}.o_proj"),
                heads * head_dim,
                hidden,
                false,
            )?,
            q_norm: sanitized_norm(checkpoint, &format!("{prefix}.q_norm.weight"), &[head_dim])?,
            k_norm: sanitized_norm(checkpoint, &format!("{prefix}.k_norm.weight"), &[head_dim])?,
            heads,
            kv_heads,
            head_dim,
            rope_dims,
            theta,
            eps,
            keys: None,
            values: None,
        })
    }

    pub fn set_state(&mut self, keys: Option<Array>, values: Option<Array>) -> Result<()> {
        ensure!(
            keys.as_ref().is_none_or(|x| {
                x.shape().len() == 4
                    && x.shape()[0] == 1
                    && x.shape()[1] == self.kv_heads
                    && x.shape()[3] == self.head_dim
                    && x.dtype() == Dtype::Float16
            }) && values.as_ref().is_none_or(|x| {
                x.shape().len() == 4
                    && x.shape()[0] == 1
                    && x.shape()[1] == self.kv_heads
                    && x.shape()[3] == self.head_dim
                    && x.dtype() == Dtype::Float16
            }) && keys.as_ref().map(Array::shape) == values.as_ref().map(Array::shape),
            "invalid Qwen attention cache"
        );
        self.keys = keys;
        self.values = values;
        Ok(())
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let time = *x
            .shape()
            .get(1)
            .context("Qwen attention input must have time")?;
        ensure!(
            x.shape().len() == 3 && x.shape()[0] == 1 && time > 0,
            "invalid Qwen attention input"
        );
        let qkv = self.qkv.forward(x)?;
        let q_gate = qkv[0].reshape(&[1, time, self.heads, self.head_dim * 2])?;
        let q = q_gate
            .slice(3, 0, self.head_dim)?
            .rms_norm(&self.q_norm, self.eps)?
            .transpose(&[0, 2, 1, 3])?;
        let gate = q_gate
            .slice(3, self.head_dim, self.head_dim * 2)?
            .reshape(&[1, time, self.heads * self.head_dim])?;
        let k = qkv[1]
            .reshape(&[1, time, self.kv_heads, self.head_dim])?
            .rms_norm(&self.k_norm, self.eps)?
            .transpose(&[0, 2, 1, 3])?;
        let v = qkv[2]
            .reshape(&[1, time, self.kv_heads, self.head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let offset = self.keys.as_ref().map_or(0, |keys| keys.shape()[2]);
        let q = q.rope(self.rope_dims, self.theta, offset)?;
        let k = k.rope(self.rope_dims, self.theta, offset)?;
        let keys = match &self.keys {
            Some(previous) => Array::concatenate(&[previous, &k], 2)?,
            None => k,
        };
        let values = match &self.values {
            Some(previous) => Array::concatenate(&[previous, &v], 2)?,
            None => v,
        };
        let attended = Array::sdpa(
            &q,
            &keys,
            &values,
            (self.head_dim as f32).powf(-0.5),
            time > 1,
        )?
        .transpose(&[0, 2, 1, 3])?
        .reshape(&[1, time, self.heads * self.head_dim])?
        .mul(&gate.sigmoid()?)?;
        self.keys = Some(keys);
        self.values = Some(values);
        self.output.forward(&attended)
    }

    fn forward_verification(&mut self, x: &Array) -> Result<Array> {
        let time = *x
            .shape()
            .get(1)
            .context("Qwen attention input must have time")?;
        ensure!(
            x.shape().len() == 3 && x.shape()[0] == 1 && (1..=8).contains(&time),
            "invalid Qwen verification attention input"
        );
        let qkv = self.qkv.forward(x)?;
        let mut attended = Vec::with_capacity(time as usize);
        for row in 0..time {
            let q_gate =
                qkv[0]
                    .slice(1, row, row + 1)?
                    .reshape(&[1, 1, self.heads, self.head_dim * 2])?;
            let q = q_gate
                .slice(3, 0, self.head_dim)?
                .rms_norm(&self.q_norm, self.eps)?
                .transpose(&[0, 2, 1, 3])?;
            let gate = q_gate
                .slice(3, self.head_dim, self.head_dim * 2)?
                .reshape(&[1, 1, self.heads * self.head_dim])?;
            let k = qkv[1]
                .slice(1, row, row + 1)?
                .reshape(&[1, 1, self.kv_heads, self.head_dim])?
                .rms_norm(&self.k_norm, self.eps)?
                .transpose(&[0, 2, 1, 3])?;
            let v = qkv[2]
                .slice(1, row, row + 1)?
                .reshape(&[1, 1, self.kv_heads, self.head_dim])?
                .transpose(&[0, 2, 1, 3])?;
            let offset = self.keys.as_ref().map_or(0, |keys| keys.shape()[2]);
            let q = q.rope(self.rope_dims, self.theta, offset)?;
            let k = k.rope(self.rope_dims, self.theta, offset)?;
            let keys = match &self.keys {
                Some(previous) => Array::concatenate(&[previous, &k], 2)?,
                None => k,
            };
            let values = match &self.values {
                Some(previous) => Array::concatenate(&[previous, &v], 2)?,
                None => v,
            };
            attended.push(
                Array::sdpa(&q, &keys, &values, (self.head_dim as f32).powf(-0.5), false)?
                    .transpose(&[0, 2, 1, 3])?
                    .reshape(&[1, 1, self.heads * self.head_dim])?
                    .mul(&gate.sigmoid()?)?,
            );
            self.keys = Some(keys);
            self.values = Some(values);
        }
        self.output.forward(&Array::concatenate(
            &attended.iter().collect::<Vec<_>>(),
            1,
        )?)
    }

    pub fn states(&self) -> Result<(&Array, &Array)> {
        Ok((
            self.keys.as_ref().context("missing attention keys")?,
            self.values.as_ref().context("missing attention values")?,
        ))
    }

    fn reset(&mut self) {
        self.keys = None;
        self.values = None;
    }
}

pub struct AttentionLayer {
    input_norm: Array,
    post_norm: Array,
    attention: Attention,
    mlp: Mlp,
    hidden: i32,
    eps: f32,
}

impl AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        checkpoint: &Checkpoint,
        index: usize,
        hidden: i32,
        heads: i32,
        kv_heads: i32,
        head_dim: i32,
        rope_dims: i32,
        theta: f32,
        expert_hidden: i32,
        num_experts: i32,
        top_k: usize,
        eps: f32,
    ) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{index}");
        Ok(Self {
            input_norm: sanitized_norm(
                checkpoint,
                &format!("{prefix}.input_layernorm.weight"),
                &[hidden],
            )?,
            post_norm: sanitized_norm(
                checkpoint,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[hidden],
            )?,
            attention: Attention::load(
                checkpoint,
                &format!("{prefix}.self_attn"),
                hidden,
                heads,
                kv_heads,
                head_dim,
                rope_dims,
                theta,
                eps,
            )?,
            mlp: Mlp::load(
                checkpoint,
                &format!("{prefix}.mlp"),
                hidden,
                expert_hidden,
                num_experts,
                top_k,
            )?,
            hidden,
            eps,
        })
    }

    pub fn set_state(&mut self, keys: Option<Array>, values: Option<Array>) -> Result<()> {
        self.attention.set_state(keys, values)
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let time = *x
            .shape()
            .get(1)
            .context("Qwen layer input must have time")?;
        ensure!(
            x.shape() == [1, time, self.hidden] && time > 0,
            "invalid Qwen layer input"
        );
        let attention = self
            .attention
            .forward(&x.rms_norm(&self.input_norm, self.eps)?)?;
        let hidden = x.add(&attention)?;
        let mlp = self.mlp.forward(
            &hidden
                .rms_norm(&self.post_norm, self.eps)?
                .reshape(&[time, self.hidden])?,
        )?;
        hidden.add(&mlp.reshape(&[1, time, self.hidden])?)
    }

    fn forward_verification(&mut self, values: &mut [Array]) -> Result<()> {
        let normalized = values
            .iter()
            .map(|value| value.rms_norm(&self.input_norm, self.eps))
            .collect::<Result<Vec<_>>>()?;
        let normalized = Array::concatenate(&normalized.iter().collect::<Vec<_>>(), 1)?;
        let attention = self.attention.forward_verification(&normalized)?;
        for (time, value) in values.iter_mut().enumerate() {
            let hidden = value.add(&attention.slice(1, time as i32, time as i32 + 1)?)?;
            let post = hidden.rms_norm(&self.post_norm, self.eps)?;
            let mlp = self.mlp.forward(&post.reshape(&[1, self.hidden])?)?;
            *value = hidden.add(&mlp.reshape(&[1, 1, self.hidden])?)?;
        }
        Ok(())
    }

    pub fn states(&self) -> Result<(&Array, &Array)> {
        self.attention.states()
    }

    fn reset(&mut self) {
        self.attention.reset();
    }
}

#[derive(Deserialize)]
struct RootConfig {
    model_type: String,
    text_config: Config,
}

#[derive(Deserialize)]
struct Config {
    hidden_size: i32,
    num_hidden_layers: usize,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    linear_num_value_heads: i32,
    linear_num_key_heads: i32,
    linear_key_head_dim: i32,
    linear_value_head_dim: i32,
    linear_conv_kernel_dim: i32,
    intermediate_size: Option<i32>,
    #[serde(default)]
    num_experts: i32,
    #[serde(default)]
    num_experts_per_tok: usize,
    #[serde(default)]
    shared_expert_intermediate_size: i32,
    #[serde(default)]
    moe_intermediate_size: i32,
    vocab_size: i32,
    max_position_embeddings: i32,
    rms_norm_eps: f32,
    partial_rotary_factor: f32,
    rope_parameters: Rope,
    layer_types: Vec<String>,
    tie_word_embeddings: bool,
}

#[derive(Deserialize)]
struct Rope {
    rope_theta: f32,
}

enum Layer {
    Linear(Box<LinearLayer>),
    Attention(Box<AttentionLayer>),
}

enum LayerSnapshot {
    Linear {
        conv: Option<Array>,
        recurrent: Option<Array>,
    },
    Attention {
        keys: Option<Array>,
        values: Option<Array>,
    },
}

/// Opaque, cheap checkpoint of the mutable Qwen decode state.
///
/// Arrays are immutable MLX graph handles, so taking a snapshot retains the
/// current state without copying its GPU storage.
pub struct QwenSnapshot {
    offset: i32,
    layers: Vec<LayerSnapshot>,
}

impl Layer {
    fn forward(&mut self, x: &Array) -> Result<Array> {
        match self {
            Self::Linear(layer) => layer.forward(x),
            Self::Attention(layer) => layer.forward(x),
        }
    }

    fn forward_verification(&mut self, values: &mut [Array]) -> Result<()> {
        match self {
            Self::Linear(layer) => layer.forward_verification(values),
            Self::Attention(layer) => layer.forward_verification(values),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Linear(layer) => layer.reset(),
            Self::Attention(layer) => layer.reset(),
        }
    }

    fn snapshot(&self) -> Result<LayerSnapshot> {
        Ok(match self {
            Self::Linear(layer) => LayerSnapshot::Linear {
                conv: layer
                    .attention
                    .conv_state
                    .as_ref()
                    .map(Array::try_clone)
                    .transpose()?,
                recurrent: layer
                    .attention
                    .recurrent_state
                    .as_ref()
                    .map(Array::try_clone)
                    .transpose()?,
            },
            Self::Attention(layer) => LayerSnapshot::Attention {
                keys: layer
                    .attention
                    .keys
                    .as_ref()
                    .map(Array::try_clone)
                    .transpose()?,
                values: layer
                    .attention
                    .values
                    .as_ref()
                    .map(Array::try_clone)
                    .transpose()?,
            },
        })
    }

    fn restore(&mut self, snapshot: LayerSnapshot) -> Result<()> {
        match (self, snapshot) {
            (Self::Linear(layer), LayerSnapshot::Linear { conv, recurrent }) => {
                layer.set_state(conv, recurrent)
            }
            (Self::Attention(layer), LayerSnapshot::Attention { keys, values }) => {
                layer.set_state(keys, values)
            }
            _ => anyhow::bail!("Qwen snapshot layer types do not match the model"),
        }
    }
}

pub struct Qwen35Moe {
    embeddings: Array,
    norm: Array,
    head: Projection,
    layers: Vec<Layer>,
    vocab: i32,
    context_limit: i32,
    eps: f32,
    offset: i32,
}

impl Qwen35Moe {
    pub fn load(path: &Path) -> Result<Self> {
        let checkpoint = crate::checkpoint::inspect(path)?;
        Self::from_checkpoint(&checkpoint)
    }

    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        let path = &checkpoint.path;
        let root: RootConfig = serde_json::from_reader(File::open(path.join("config.json"))?)?;
        let config = root.text_config;
        ensure!(
            root.model_type == "qwen3_5" || root.model_type == "qwen3_5_moe",
            "expected qwen3_5 or qwen3_5_moe"
        );
        ensure!(
            config.hidden_size > 0
                && config.vocab_size > 0
                && config.max_position_embeddings > 0
                && config.num_hidden_layers > 0
                && config.layer_types.len() == config.num_hidden_layers
                && config.head_dim > 0
                && config.partial_rotary_factor > 0.
                && config.partial_rotary_factor <= 1.
                && !config.tie_word_embeddings,
            "unsupported Qwen3.5 configuration"
        );
        let mlp_hidden = if config.num_experts > 0 {
            ensure!(
                config.num_experts_per_tok > 0
                    && config.num_experts_per_tok <= config.num_experts as usize
                    && config.moe_intermediate_size > 0
                    && config.moe_intermediate_size == config.shared_expert_intermediate_size,
                "unsupported Qwen3.5 MoE configuration"
            );
            config.moe_intermediate_size
        } else {
            ensure!(
                config.num_experts_per_tok == 0,
                "invalid dense Qwen routing"
            );
            config
                .intermediate_size
                .filter(|&size| size > 0)
                .context("missing dense Qwen intermediate size")?
        };
        let hidden = config.hidden_size;
        let embeddings = half_weight(
            checkpoint,
            "model.language_model.embed_tokens.weight",
            Some(&[config.vocab_size, hidden]),
        )?;
        let norm = sanitized_norm(checkpoint, "model.language_model.norm.weight", &[hidden])?;
        let head = Projection::load(checkpoint, "lm_head", hidden, config.vocab_size, false)?;
        let rope_dims = (config.head_dim as f32 * config.partial_rotary_factor) as i32;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for (index, kind) in config.layer_types.iter().enumerate() {
            layers.push(match kind.as_str() {
                "linear_attention" => Layer::Linear(Box::new(LinearLayer::load(
                    checkpoint,
                    index,
                    hidden,
                    config.linear_num_key_heads,
                    config.linear_num_value_heads,
                    config.linear_key_head_dim,
                    config.linear_value_head_dim,
                    config.linear_conv_kernel_dim,
                    mlp_hidden,
                    config.num_experts,
                    config.num_experts_per_tok,
                    config.rms_norm_eps,
                )?)),
                "full_attention" => Layer::Attention(Box::new(AttentionLayer::load(
                    checkpoint,
                    index,
                    hidden,
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.head_dim,
                    rope_dims,
                    config.rope_parameters.rope_theta,
                    mlp_hidden,
                    config.num_experts,
                    config.num_experts_per_tok,
                    config.rms_norm_eps,
                )?)),
                other => anyhow::bail!("unsupported Qwen layer type {other}"),
            });
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
        match self.run(token, false) {
            Ok((logits, _)) => Ok(logits),
            Err(error) => {
                self.reset();
                Err(error.context("Qwen forward failed; its cache was reset"))
            }
        }
    }

    pub fn trace(&mut self, token: u32) -> Result<(Array, Vec<Vec<u16>>)> {
        self.run(token, true)
    }

    pub fn forward_tokens(&mut self, tokens: &[u32]) -> Result<Array> {
        match self.run_tokens(tokens, false, true) {
            Ok((logits, _)) => Ok(logits),
            Err(error) => {
                self.reset();
                Err(error.context("Qwen batched forward failed; its cache was reset"))
            }
        }
    }

    /// Runs target verification in the exact autoregressive order, then
    /// synchronizes the resulting Metal graph once.
    pub fn verify_tokens_exact(&mut self, tokens: &[u32]) -> Result<Array> {
        let result = (|| {
            ensure!(
                !tokens.is_empty() && tokens.len() <= 8,
                "target verification requires 1 to 8 tokens"
            );
            let hidden = self.run_hidden_tokens_layer_major(tokens)?;
            let output = self.head.forward(&hidden)?;
            output.eval()?;
            self.offset += i32::try_from(tokens.len())?;
            Ok(output)
        })();
        if result.is_err() {
            self.reset();
        }
        result.context("Qwen target verification failed; its cache was reset")
    }

    fn run(&mut self, token: u32, trace: bool) -> Result<(Array, Vec<Vec<u16>>)> {
        self.run_tokens(&[token], trace, true)
    }

    fn run_tokens(
        &mut self,
        tokens: &[u32],
        trace: bool,
        evaluate: bool,
    ) -> Result<(Array, Vec<Vec<u16>>)> {
        let (normalized, layers) = self.run_hidden_tokens(tokens, trace)?;
        let logits = self.head.forward(&normalized)?;
        if evaluate {
            logits.eval()?;
        }
        self.offset += i32::try_from(tokens.len())?;
        Ok((logits, layers))
    }

    fn run_hidden_tokens(&mut self, tokens: &[u32], trace: bool) -> Result<(Array, Vec<Vec<u16>>)> {
        ensure!(
            !tokens.is_empty() && tokens.iter().all(|&token| token < self.vocab as u32),
            "Qwen token batch is empty or outside vocabulary"
        );
        let time = i32::try_from(tokens.len())?;
        ensure!(
            self.offset + time <= self.context_limit,
            "Qwen context is full"
        );
        let ids = tokens.iter().map(|&token| token as i32).collect::<Vec<_>>();
        let id = Array::from_i32(&ids, &[1, time])?;
        let mut hidden = self.embeddings.take(&id, 0)?;
        let mut layers = Vec::with_capacity(if trace { self.layers.len() + 2 } else { 0 });
        if trace {
            layers.push(hidden.to_f16_bits()?);
        }
        for layer in &mut self.layers {
            hidden = layer.forward(&hidden)?;
            if trace {
                layers.push(hidden.to_f16_bits()?);
            }
        }
        let normalized = hidden
            .slice(1, time - 1, time)?
            .rms_norm(&self.norm, self.eps)?;
        if trace {
            layers.push(normalized.to_f16_bits()?);
        }
        Ok((normalized, layers))
    }

    fn run_hidden_tokens_layer_major(&mut self, tokens: &[u32]) -> Result<Array> {
        ensure!(
            !tokens.is_empty() && tokens.iter().all(|&token| token < self.vocab as u32),
            "Qwen token batch is empty or outside vocabulary"
        );
        let time = i32::try_from(tokens.len())?;
        ensure!(
            self.offset + time <= self.context_limit,
            "Qwen context is full"
        );
        let mut hidden = tokens
            .iter()
            .map(|&token| {
                let id = Array::from_i32(&[token as i32], &[1, 1])?;
                self.embeddings.take(&id, 0)
            })
            .collect::<Result<Vec<_>>>()?;
        for layer in &mut self.layers {
            layer.forward_verification(&mut hidden)?;
        }
        let normalized = hidden
            .into_iter()
            .map(|value| value.rms_norm(&self.norm, self.eps))
            .collect::<Result<Vec<_>>>()?;
        Array::concatenate(&normalized.iter().collect::<Vec<_>>(), 1)
    }

    pub fn context_limit(&self) -> i32 {
        self.context_limit
    }

    pub fn snapshot(&self) -> Result<QwenSnapshot> {
        Ok(QwenSnapshot {
            offset: self.offset,
            layers: self
                .layers
                .iter()
                .map(Layer::snapshot)
                .collect::<Result<_>>()?,
        })
    }

    pub fn restore(&mut self, snapshot: QwenSnapshot) -> Result<()> {
        ensure!(
            snapshot.offset >= 0
                && snapshot.offset <= self.context_limit
                && snapshot.layers.len() == self.layers.len(),
            "Qwen snapshot does not match the model"
        );
        ensure!(
            self.layers
                .iter()
                .zip(&snapshot.layers)
                .all(|(layer, state)| matches!(
                    (layer, state),
                    (Layer::Linear(_), LayerSnapshot::Linear { .. })
                        | (Layer::Attention(_), LayerSnapshot::Attention { .. })
                )),
            "Qwen snapshot layer types do not match the model"
        );
        if let Err(error) = self
            .layers
            .iter_mut()
            .zip(snapshot.layers)
            .try_for_each(|(layer, state)| layer.restore(state))
        {
            self.reset();
            return Err(error.context("invalid Qwen snapshot; model state was reset"));
        }
        self.offset = snapshot.offset;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}

impl LinearLayer {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        checkpoint: &Checkpoint,
        index: usize,
        hidden: i32,
        key_heads: i32,
        value_heads: i32,
        key_dim: i32,
        value_dim: i32,
        conv_length: i32,
        expert_hidden: i32,
        num_experts: i32,
        top_k: usize,
        eps: f32,
    ) -> Result<Self> {
        let prefix = format!("model.language_model.layers.{index}");
        Ok(Self {
            input_norm: sanitized_norm(
                checkpoint,
                &format!("{prefix}.input_layernorm.weight"),
                &[hidden],
            )?,
            post_norm: sanitized_norm(
                checkpoint,
                &format!("{prefix}.post_attention_layernorm.weight"),
                &[hidden],
            )?,
            attention: GatedDelta::load(
                checkpoint,
                &format!("{prefix}.linear_attn"),
                hidden,
                key_heads,
                value_heads,
                key_dim,
                value_dim,
                conv_length,
                eps,
            )?,
            mlp: Mlp::load(
                checkpoint,
                &format!("{prefix}.mlp"),
                hidden,
                expert_hidden,
                num_experts,
                top_k,
            )?,
            hidden,
            eps,
        })
    }

    pub fn set_state(&mut self, conv: Option<Array>, recurrent: Option<Array>) -> Result<()> {
        self.attention.set_state(conv, recurrent)
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        Ok(self.run(x, false)?.0)
    }

    fn forward_verification(&mut self, values: &mut [Array]) -> Result<()> {
        let normalized = values
            .iter()
            .map(|value| value.rms_norm(&self.input_norm, self.eps))
            .collect::<Result<Vec<_>>>()?;
        let normalized = Array::concatenate(&normalized.iter().collect::<Vec<_>>(), 1)?;
        let attention = self.attention.forward_verification(&normalized)?;
        for (time, value) in values.iter_mut().enumerate() {
            let hidden = value.add(&attention.slice(1, time as i32, time as i32 + 1)?)?;
            let post = hidden.rms_norm(&self.post_norm, self.eps)?;
            let mlp = self.mlp.forward(&post.reshape(&[1, self.hidden])?)?;
            *value = hidden.add(&mlp.reshape(&[1, 1, self.hidden])?)?;
        }
        Ok(())
    }

    pub fn trace(&mut self, x: &Array) -> Result<(Array, Vec<Array>)> {
        self.run(x, true)
    }

    fn run(&mut self, x: &Array, trace: bool) -> Result<(Array, Vec<Array>)> {
        let time = *x
            .shape()
            .get(1)
            .context("Qwen layer input must have time")?;
        ensure!(
            x.shape() == [1, time, self.hidden] && time > 0,
            "invalid Qwen layer input"
        );
        let normalized = x.rms_norm(&self.input_norm, self.eps)?;
        let (attention, inner) = if trace {
            self.attention.trace(&normalized)?
        } else {
            (self.attention.forward(&normalized)?, Vec::new())
        };
        let hidden = x.add(&attention)?;
        let post = hidden.rms_norm(&self.post_norm, self.eps)?;
        let mlp = self.mlp.forward(&post.reshape(&[time, self.hidden])?)?;
        let mlp = mlp.reshape(&[1, time, self.hidden])?;
        let output = hidden.add(&mlp)?;
        let values = if trace {
            let mut values = vec![normalized];
            values.extend(inner);
            values.extend([hidden, post, mlp, output.try_clone()?]);
            values
        } else {
            Vec::new()
        };
        Ok((output, values))
    }

    pub fn states(&self) -> Result<(&Array, &Array)> {
        self.attention.states()
    }

    fn reset(&mut self) {
        self.attention.reset();
    }
}

impl GatedDelta {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        hidden: i32,
        key_heads: i32,
        value_heads: i32,
        key_dim: i32,
        value_dim: i32,
        conv_length: i32,
        eps: f32,
    ) -> Result<Self> {
        ensure!(
            hidden > 0
                && key_heads > 0
                && value_heads > 0
                && value_heads % key_heads == 0
                && key_dim == 128
                && value_dim > 0
                && value_dim % 8 == 0
                && conv_length > 0
                && eps.is_finite()
                && eps > 0.,
            "invalid Qwen Gated DeltaNet dimensions"
        );
        let keys = key_heads * key_dim;
        let values = value_heads * value_dim;
        let conv_dims = 2 * keys + values;
        let weight = half_weight(
            checkpoint,
            &format!("{prefix}.conv1d.weight"),
            Some(&[conv_dims, 1, conv_length]),
        )?
        .transpose(&[0, 2, 1])?;
        weight.eval()?;
        let a_log =
            checkpoint_array(checkpoint, &format!("{prefix}.A_log"))?.astype(Dtype::Float32)?;
        let dt_bias = half_weight(
            checkpoint,
            &format!("{prefix}.dt_bias"),
            Some(&[value_heads]),
        )?;
        a_log.eval()?;
        Ok(Self {
            inputs: ProjectionBundle::load(
                checkpoint,
                hidden,
                vec![
                    (format!("{prefix}.in_proj_qkv"), conv_dims),
                    (format!("{prefix}.in_proj_z"), values),
                ],
            )?,
            b: Projection::load(
                checkpoint,
                &format!("{prefix}.in_proj_b"),
                hidden,
                value_heads,
                false,
            )?,
            a: Projection::load(
                checkpoint,
                &format!("{prefix}.in_proj_a"),
                hidden,
                value_heads,
                false,
            )?,
            output: Projection::load(
                checkpoint,
                &format!("{prefix}.out_proj"),
                values,
                hidden,
                false,
            )?,
            conv_weight: weight,
            norm: half_weight(
                checkpoint,
                &format!("{prefix}.norm.weight"),
                Some(&[value_dim]),
            )?,
            a_log,
            dt_bias,
            conv_state: None,
            recurrent_state: None,
            hidden,
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            conv_length,
            eps,
        })
    }

    pub fn set_state(&mut self, conv: Option<Array>, recurrent: Option<Array>) -> Result<()> {
        let conv_dims = 2 * self.key_heads * self.key_dim + self.value_heads * self.value_dim;
        ensure!(
            conv.as_ref().is_none_or(|x| {
                x.shape() == [1, self.conv_length - 1, conv_dims] && x.dtype() == Dtype::Float16
            }),
            "invalid convolution state"
        );
        ensure!(
            recurrent.as_ref().is_none_or(|x| {
                x.shape() == [1, self.value_heads, self.value_dim, self.key_dim]
                    && x.dtype() == Dtype::Float32
            }),
            "invalid recurrent state"
        );
        self.conv_state = conv;
        self.recurrent_state = recurrent;
        Ok(())
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        Ok(self.run(x, false)?.0)
    }

    fn forward_verification(&mut self, x: &Array) -> Result<Array> {
        let time = *x.shape().get(1).context("Qwen GDN input must have time")?;
        ensure!(
            x.shape() == [1, time, self.hidden] && (1..=8).contains(&time),
            "invalid Qwen verification GDN input"
        );
        let keys = self.key_heads * self.key_dim;
        let values = self.value_heads * self.value_dim;
        let conv_dims = 2 * keys + values;
        let inputs = self.inputs.forward(x)?;
        let qkv = inputs[0].try_clone()?;
        let z = inputs[1].reshape(&[1, time, self.value_heads, self.value_dim])?;
        let b = self.b.forward(x)?;
        let a = self.a.forward(x)?;
        let q_scale =
            Array::from_f16_bits(&[f16::from_f32(1.0 / self.key_dim as f32).to_bits()], &[])?;
        let k_scale = Array::from_f16_bits(
            &[f16::from_f32(1.0 / (self.key_dim as f32).sqrt()).to_bits()],
            &[],
        )?;
        let mut conv_state = match &self.conv_state {
            Some(state) => state.try_clone()?,
            None => Array::zeros_dtype(&[1, self.conv_length - 1, conv_dims], Dtype::Float16)?,
        };
        let mut recurrent_state = match &self.recurrent_state {
            Some(state) => state.try_clone()?,
            None => Array::zeros_dtype(
                &[1, self.value_heads, self.value_dim, self.key_dim],
                Dtype::Float32,
            )?,
        };
        let mut gated = Vec::with_capacity(time as usize);
        for row in 0..time {
            let conv_input = Array::concatenate(&[&conv_state, &qkv.slice(1, row, row + 1)?], 1)?;
            conv_state = conv_input.slice(1, 1, self.conv_length)?;
            let conv = conv_input.conv1d(&self.conv_weight, conv_dims)?.silu()?;
            let q = conv
                .slice(2, 0, keys)?
                .reshape(&[1, 1, self.key_heads, self.key_dim])?
                .rms_norm_without_weight(1e-6)?
                .mul(&q_scale)?;
            let k = conv
                .slice(2, keys, 2 * keys)?
                .reshape(&[1, 1, self.key_heads, self.key_dim])?
                .rms_norm_without_weight(1e-6)?
                .mul(&k_scale)?;
            let v = conv.slice(2, 2 * keys, conv_dims)?.reshape(&[
                1,
                1,
                self.value_heads,
                self.value_dim,
            ])?;
            let (out, state) = gated_delta::step_with_gates(
                &q,
                &k,
                &v,
                &a.slice(1, row, row + 1)?,
                &b.slice(1, row, row + 1)?,
                &self.a_log,
                &self.dt_bias,
                &recurrent_state,
            )?;
            recurrent_state = state;
            let normalized = out.rms_norm(&self.norm, self.eps)?;
            gated.push(z.slice(1, row, row + 1)?.precise_swiglu(&normalized)?);
        }
        self.conv_state = Some(conv_state);
        self.recurrent_state = Some(recurrent_state);
        self.output.forward(
            &Array::concatenate(&gated.iter().collect::<Vec<_>>(), 1)?
                .reshape(&[1, time, values])?,
        )
    }

    pub fn trace(&mut self, x: &Array) -> Result<(Array, Vec<Array>)> {
        self.run(x, true)
    }

    fn run(&mut self, x: &Array, trace: bool) -> Result<(Array, Vec<Array>)> {
        let time = *x.shape().get(1).context("Qwen GDN input must have time")?;
        ensure!(
            x.shape() == [1, time, self.hidden] && time > 0,
            "invalid Qwen GDN input"
        );
        let keys = self.key_heads * self.key_dim;
        let values = self.value_heads * self.value_dim;
        let conv_dims = 2 * keys + values;
        let inputs = self.inputs.forward(x)?;
        let qkv = inputs[0].try_clone()?;
        let z = inputs[1].reshape(&[1, time, self.value_heads, self.value_dim])?;
        let b = self.b.forward(x)?;
        let a = self.a.forward(x)?;
        let conv_state = match &self.conv_state {
            Some(state) => state.try_clone()?,
            None => Array::zeros_dtype(&[1, self.conv_length - 1, conv_dims], Dtype::Float16)?,
        };
        let conv_input = Array::concatenate(&[&conv_state, &qkv], 1)?;
        self.conv_state = Some(conv_input.slice(1, time, time + self.conv_length - 1)?);
        let conv = conv_input.conv1d(&self.conv_weight, conv_dims)?.silu()?;
        let q = conv
            .slice(2, 0, keys)?
            .reshape(&[1, time, self.key_heads, self.key_dim])?;
        let k = conv
            .slice(2, keys, 2 * keys)?
            .reshape(&[1, time, self.key_heads, self.key_dim])?;
        let v = conv.slice(2, 2 * keys, 2 * keys + values)?.reshape(&[
            1,
            time,
            self.value_heads,
            self.value_dim,
        ])?;
        let q_scale =
            Array::from_f16_bits(&[f16::from_f32(1.0 / (self.key_dim as f32)).to_bits()], &[])?;
        let k_scale = Array::from_f16_bits(
            &[f16::from_f32(1.0 / (self.key_dim as f32).sqrt()).to_bits()],
            &[],
        )?;
        let q = q.rms_norm_without_weight(1e-6)?.mul(&q_scale)?;
        let k = k.rms_norm_without_weight(1e-6)?.mul(&k_scale)?;
        let state = match &self.recurrent_state {
            Some(state) => state.try_clone()?,
            None => Array::zeros_dtype(
                &[1, self.value_heads, self.value_dim, self.key_dim],
                Dtype::Float32,
            )?,
        };
        let (out, state, beta, softplus, g) = if time == 1 && !trace {
            let (out, state) = gated_delta::step_with_gates(
                &q,
                &k,
                &v,
                &a,
                &b,
                &self.a_log,
                &self.dt_bias,
                &state,
            )?;
            (out, state, None, None, None)
        } else {
            let beta = b.sigmoid()?;
            let zero = Array::from_f16_bits(&[0], &[])?;
            let softplus = a.add(&self.dt_bias)?.logaddexp(&zero)?;
            let g = self.a_log.exp()?.mul(&softplus)?.negative()?.exp()?;
            let (out, state) = gated_delta::step(&q, &k, &v, &g, &beta, &state)?;
            (out, state, Some(beta), Some(softplus), Some(g))
        };
        self.recurrent_state = Some(state);
        let normalized = out.rms_norm(&self.norm, self.eps)?;
        let gated = z.precise_swiglu(&normalized)?;
        let output = self.output.forward(&gated.reshape(&[1, time, values])?)?;
        let arrays = if trace {
            vec![
                qkv,
                z,
                a,
                b,
                conv,
                q,
                k,
                v,
                beta.context("missing traced Gated DeltaNet beta")?,
                softplus.context("missing traced Gated DeltaNet softplus")?,
                g.context("missing traced Gated DeltaNet decay")?,
                out,
                normalized,
                gated,
                output.try_clone()?,
            ]
        } else {
            Vec::new()
        };
        Ok((output, arrays))
    }

    pub fn states(&self) -> Result<(&Array, &Array)> {
        Ok((
            self.conv_state
                .as_ref()
                .context("missing convolution state")?,
            self.recurrent_state
                .as_ref()
                .context("missing recurrent state")?,
        ))
    }

    fn reset(&mut self) {
        self.conv_state = None;
        self.recurrent_state = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn state_bytes(model: &Qwen35Moe) -> Result<Vec<Vec<u8>>> {
        model
            .layers
            .iter()
            .flat_map(|layer| match layer {
                Layer::Linear(layer) => [
                    layer.attention.conv_state.as_ref(),
                    layer.attention.recurrent_state.as_ref(),
                ],
                Layer::Attention(layer) => [
                    layer.attention.keys.as_ref(),
                    layer.attention.values.as_ref(),
                ],
            })
            .map(|state| state.context("missing Qwen test state")?.to_bytes())
            .collect()
    }

    fn batched_head_token_major(model: &mut Qwen35Moe, tokens: &[u32]) -> Result<Array> {
        let mut hidden = Vec::with_capacity(tokens.len());
        for &token in tokens {
            hidden.push(model.run_hidden_tokens(&[token], false)?.0);
            model.offset += 1;
        }
        let hidden = Array::concatenate(&hidden.iter().collect::<Vec<_>>(), 1)?;
        let output = model.head.forward(&hidden)?;
        output.eval()?;
        Ok(output)
    }

    #[test]
    #[ignore = "requires local Qwen checkpoint and Apple GPU"]
    fn snapshot_restore_replays_logits_and_state_exactly() -> Result<()> {
        let mut model = Qwen35Moe::load(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        for token in [1, 2, 3] {
            model.forward(token)?;
        }
        let snapshot = model.snapshot()?;
        let expected_logits = [4, 5]
            .into_iter()
            .map(|token| model.forward(token)?.to_f16_bits())
            .collect::<Result<Vec<_>>>()?;
        let expected_state = state_bytes(&model)?;

        model.restore(snapshot)?;
        for (token, expected) in [4, 5].into_iter().zip(expected_logits) {
            assert_eq!(model.forward(token)?.to_f16_bits()?, expected);
        }
        assert_eq!(state_bytes(&model)?, expected_state);
        assert_eq!(model.offset, 5);
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen checkpoint and Apple GPU"]
    fn verification_widths_match_token_major() -> Result<()> {
        let mut model = Qwen35Moe::load(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        for width in [1, 2, 4, 8] {
            model.reset();
            for token in [1, 2, 3] {
                model.forward(token)?;
            }
            let tokens = (4..4 + width).collect::<Vec<_>>();
            let expected = batched_head_token_major(&mut model, &tokens)?.to_f16_bits()?;
            let expected_state = state_bytes(&model)?;

            model.reset();
            for token in [1, 2, 3] {
                model.forward(token)?;
            }
            assert_eq!(
                model.verify_tokens_exact(&tokens)?.to_f16_bits()?,
                expected,
                "verification logits differ at M={width}"
            );
            assert_eq!(
                state_bytes(&model)?,
                expected_state,
                "verification state differs at M={width}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen checkpoint and Apple GPU"]
    fn deferred_target_verification_matches_sequential() -> Result<()> {
        let mut model = Qwen35Moe::load(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        for token in [1, 2, 3] {
            model.forward(token)?;
        }
        let snapshot = model.snapshot()?;
        let tokens = [4, 5, 6, 7, 8, 9, 10, 11];

        model.verify_tokens_exact(&tokens)?;
        model.restore(snapshot)?;
        let mut token_major_samples = Vec::new();
        let mut layer_major_samples = Vec::new();
        let mut expected = None;
        for candidate_first in [false, true, true, false, false, true] {
            for candidate in [candidate_first, !candidate_first] {
                let snapshot = model.snapshot()?;
                let start = Instant::now();
                let output = if candidate {
                    model.verify_tokens_exact(&tokens)?
                } else {
                    batched_head_token_major(&mut model, &tokens)?
                };
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                let bits = output.to_f16_bits()?;
                let state = state_bytes(&model)?;
                if let Some((expected_bits, expected_state)) = &expected {
                    assert_eq!(&bits, expected_bits, "target logits differ");
                    assert_eq!(&state, expected_state, "target state differs");
                } else {
                    expected = Some((bits, state));
                }
                if candidate {
                    layer_major_samples.push(elapsed);
                } else {
                    token_major_samples.push(elapsed);
                }
                model.restore(snapshot)?;
            }
        }
        token_major_samples.sort_by(f64::total_cmp);
        layer_major_samples.sort_by(f64::total_cmp);
        let token_major_ms = token_major_samples[token_major_samples.len() / 2];
        let layer_major_ms = layer_major_samples[layer_major_samples.len() / 2];
        eprintln!(
            "Qwen target M=8: token_major={token_major_ms:.3}ms layer_major={layer_major_ms:.3}ms speedup={:.2}x samples={token_major_samples:?}/{layer_major_samples:?}",
            token_major_ms / layer_major_ms
        );
        Ok(())
    }
}
