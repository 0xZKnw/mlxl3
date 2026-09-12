//! Batch-one, token-by-token LFM2 dense inference using MLX arrays.
//! Weights and operation order follow mlx-lm's `models/lfm2.py`.
// Architecture adapted from mlx-lm, Copyright © 2025 Apple Inc. (MIT).
use crate::{
    array::{Array, Dtype},
    checkpoint::{self, Checkpoint},
    linear::{Exl3Group, Exl3Linear, checkpoint_array},
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{collections::BTreeSet, fs::File, path::Path};

#[derive(Deserialize)]
struct Config {
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
    block_dim: i32,
    block_ff_dim: Option<i32>,
    intermediate_size: Option<i32>,
    block_multiple_of: i32,
    block_auto_adjust_ff_dim: bool,
    block_ffn_dim_multiplier: Option<f64>,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    rope_parameters: Option<serde_json::Value>,
    full_attn_idxs: Option<Vec<usize>>,
    layer_types: Option<Vec<String>>,
}

fn default_rope_theta() -> f32 {
    1_000_000.0
}

fn half_weight(checkpoint: &Checkpoint, name: &str, expected: Option<&[i32]>) -> Result<Array> {
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

enum ProjectionWeights {
    Exl3(Exl3Linear),
    Dense(Array),
}

struct Projection {
    weights: ProjectionWeights,
    bias: Option<Array>,
}

impl Projection {
    fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        input: i32,
        output: i32,
        bias: bool,
    ) -> Result<Self> {
        let is_exl3 = checkpoint.modules.iter().any(|name| name == prefix);
        if bias {
            ensure!(
                checkpoint.tensors.contains_key(&format!("{prefix}.bias")),
                "{prefix}: missing projection bias"
            );
        }
        let weights = if is_exl3 {
            let quantized = Exl3Linear::from_checkpoint(checkpoint, prefix)?;
            ensure!(
                quantized.input_dims() == input && quantized.output_dims() == output,
                "{prefix}: invalid EXL3 projection dimensions"
            );
            ProjectionWeights::Exl3(quantized)
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
        Ok(Self { weights, bias })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        let value = match &self.weights {
            ProjectionWeights::Exl3(linear) => linear.forward(x)?,
            ProjectionWeights::Dense(weight) => x.matmul(&weight.transpose(&[1, 0])?)?,
        };
        match &self.bias {
            Some(bias) => value.add(bias),
            None => Ok(value),
        }
    }
}

enum Qkv {
    Separate(Box<[Projection; 3]>),
    Grouped(Box<Exl3Group>),
}

impl Qkv {
    fn new(projections: [Projection; 3]) -> Result<Self> {
        let linears: Vec<_> = projections
            .iter()
            .filter_map(|p| match &p.weights {
                ProjectionWeights::Exl3(l) => Some(l),
                _ => None,
            })
            .collect();
        if linears.len() == 3 && Exl3Group::compatible(&linears) {
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

    fn forward(&self, x: &Array) -> Result<Vec<Array>> {
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
}

impl Attention {
    fn load(checkpoint: &Checkpoint, prefix: &str, config: &Config, theta: f32) -> Result<Self> {
        let dim = config.hidden_size;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads.unwrap_or(heads);
        let head_dim = dim / heads;
        Ok(Self {
            qkv: Qkv::new([
                Projection::load(checkpoint, &format!("{prefix}.q_proj"), dim, dim, false)?,
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
            output: Projection::load(checkpoint, &format!("{prefix}.out_proj"), dim, dim, false)?,
            q_norm: half_weight(
                checkpoint,
                &format!("{prefix}.q_layernorm.weight"),
                Some(&[head_dim]),
            )?,
            k_norm: half_weight(
                checkpoint,
                &format!("{prefix}.k_layernorm.weight"),
                Some(&[head_dim]),
            )?,
            heads,
            kv_heads,
            head_dim,
            eps: config.norm_eps,
            theta,
            keys: None,
            values: None,
        })
    }

    fn forward(&mut self, x: &Array, offset: i32) -> Result<Array> {
        let projections = self.qkv.forward(x)?;
        let q = projections[0]
            .reshape(&[1, 1, self.heads, self.head_dim])?
            .rms_norm(&self.q_norm, self.eps)?
            .transpose(&[0, 2, 1, 3])?
            .rope(self.head_dim, self.theta, offset)?;
        let k = projections[1]
            .reshape(&[1, 1, self.kv_heads, self.head_dim])?
            .rms_norm(&self.k_norm, self.eps)?
            .transpose(&[0, 2, 1, 3])?
            .rope(self.head_dim, self.theta, offset)?;
        let v = projections[2]
            .reshape(&[1, 1, self.kv_heads, self.head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let keys = match &self.keys {
            Some(previous) => Array::concatenate(&[previous, &k], 2)?,
            None => k,
        };
        let values = match &self.values {
            Some(previous) => Array::concatenate(&[previous, &v], 2)?,
            None => v,
        };
        // One query token can attend every cached position, including itself.
        let out = Array::sdpa(&q, &keys, &values, (self.head_dim as f32).powf(-0.5), false)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, 1, self.heads * self.head_dim])?;
        self.keys = Some(keys);
        self.values = Some(values);
        self.output.forward(&out)
    }

    fn reset(&mut self) {
        self.keys = None;
        self.values = None;
    }
}

struct ShortConv {
    input: Projection,
    output: Projection,
    weight: Array,
    bias: Option<Array>,
    state: Option<Array>,
    dim: i32,
    length: i32,
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
        })
    }

    fn forward(&mut self, x: &Array) -> Result<Array> {
        let bcx = self.input.forward(x)?;
        let b = bcx.slice(2, 0, self.dim)?;
        let c = bcx.slice(2, self.dim, 2 * self.dim)?;
        let projected_x = bcx.slice(2, 2 * self.dim, 3 * self.dim)?;
        let bx = b.mul(&projected_x)?;
        let state = match &self.state {
            Some(state) => state.try_clone()?,
            None => Array::zeros_dtype(&[1, self.length - 1, self.dim], bx.dtype())?,
        };
        let bx = Array::concatenate(&[&state, &bx], 1)?;
        let mut conv = bx.conv1d(&self.weight, self.dim)?;
        if let Some(bias) = &self.bias {
            conv = conv.add(bias)?;
        }
        self.state = Some(bx.slice(1, 1, self.length)?);
        self.output.forward(&c.mul(&conv)?)
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
    w1: Projection,
    w3: Projection,
    w2: Projection,
    eps: f32,
}

impl Layer {
    fn load(
        checkpoint: &Checkpoint,
        index: usize,
        config: &Config,
        attention: bool,
        ff_dim: i32,
        theta: f32,
    ) -> Result<Self> {
        let prefix = format!("model.layers.{index}");
        let dim = config.hidden_size;
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
                )?)
            } else {
                Operator::Conv(ShortConv::load(
                    checkpoint,
                    &format!("{prefix}.conv"),
                    config,
                )?)
            },
            w1: Projection::load(
                checkpoint,
                &format!("{prefix}.feed_forward.w1"),
                dim,
                ff_dim,
                false,
            )?,
            w3: Projection::load(
                checkpoint,
                &format!("{prefix}.feed_forward.w3"),
                dim,
                ff_dim,
                false,
            )?,
            w2: Projection::load(
                checkpoint,
                &format!("{prefix}.feed_forward.w2"),
                ff_dim,
                dim,
                false,
            )?,
            eps: config.norm_eps,
        })
    }

    fn forward(&mut self, x: &Array, offset: i32) -> Result<Array> {
        let normalized = x.rms_norm(&self.operator_norm, self.eps)?;
        let mixed = match &mut self.operator {
            Operator::Attention(attention) => attention.forward(&normalized, offset)?,
            Operator::Conv(conv) => conv.forward(&normalized)?,
        };
        let h = x.add(&mixed)?;
        let normalized = h.rms_norm(&self.ffn_norm, self.eps)?;
        let gate = self.w1.forward(&normalized)?;
        let up = self.w3.forward(&normalized)?;
        let activated = gate.swiglu(&up)?;
        h.add(&self.w2.forward(&activated)?)
    }
}

pub struct Lfm2 {
    embeddings: Array,
    norm: Array,
    head: Option<Projection>,
    layers: Vec<Layer>,
    vocab_size: i32,
    context_limit: i32,
    eps: f32,
    offset: i32,
}

impl Lfm2 {
    pub fn load(path: &Path) -> Result<Self> {
        let config: Config = serde_json::from_reader(File::open(path.join("config.json"))?)?;
        ensure!(
            config.model_type == "lfm2",
            "expected dense LFM2, got {}",
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
        ensure!(
            config.conv_length > 0 && config.block_dim == config.hidden_size,
            "invalid LFM2 convolution/block dimensions"
        );
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
                config.block_multiple_of > 0,
                "invalid LFM2 feed-forward multiple"
            );
            let multiple = i64::from(config.block_multiple_of);
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
                types
                    .iter()
                    .all(|kind| kind == "full_attention" || kind == "conv" || kind == "short_conv"),
                "unsupported LFM2 layer type"
            );
            types
                .iter()
                .enumerate()
                .filter_map(|(i, kind)| (kind == "full_attention").then_some(i))
                .collect()
        };
        ensure!(
            attention.iter().all(|&i| i < config.num_hidden_layers),
            "invalid LFM2 attention index"
        );
        let checkpoint = checkpoint::inspect(path)?;
        let dim = config.hidden_size;
        let embeddings = half_weight(
            &checkpoint,
            "model.embed_tokens.weight",
            Some(&[config.vocab_size, dim]),
        )?;
        let norm = half_weight(&checkpoint, "model.embedding_norm.weight", Some(&[dim]))?;
        let head = if checkpoint.modules.iter().any(|name| name == "lm_head")
            || checkpoint.tensors.contains_key("lm_head.weight")
        {
            Some(Projection::load(
                &checkpoint,
                "lm_head",
                dim,
                config.vocab_size,
                false,
            )?)
        } else {
            None
        };
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for index in 0..config.num_hidden_layers {
            layers.push(Layer::load(
                &checkpoint,
                index,
                &config,
                attention.contains(&index),
                ff_dim,
                theta,
            )?);
        }
        Ok(Self {
            embeddings,
            norm,
            head,
            layers,
            vocab_size: config.vocab_size,
            context_limit: config.max_position_embeddings,
            eps: config.norm_eps,
            offset: 0,
        })
    }

    /// Consumes one token and returns logits shaped [1, 1, vocabulary].
    /// Call repeatedly for prompt ingestion; batched prefill is not implemented.
    pub fn forward(&mut self, tokens: &[u32]) -> Result<Array> {
        ensure!(
            tokens.len() == 1,
            "native LFM2 currently accepts exactly one token per forward"
        );
        ensure!(
            tokens[0] < self.vocab_size as u32,
            "LFM2 token is outside vocabulary"
        );
        ensure!(
            self.offset < self.context_limit,
            "LFM2 context is full ({} tokens); reset the model cache before continuing",
            self.context_limit
        );
        let result = (|| -> Result<Array> {
            let next_offset = self
                .offset
                .checked_add(1)
                .context("LFM2 position overflow")?;
            let ids = Array::from_i32(&[tokens[0] as i32], &[1, 1])?;
            let mut hidden = self.embeddings.take(&ids, 0)?;
            for layer in &mut self.layers {
                hidden = layer.forward(&hidden, self.offset)?;
            }
            let hidden = hidden.rms_norm(&self.norm, self.eps)?;
            let logits = match &self.head {
                Some(head) => head.forward(&hidden)?,
                None => hidden.matmul(&self.embeddings.transpose(&[1, 0])?)?,
            };
            self.offset = next_offset;
            Ok(logits)
        })();
        match result {
            Ok(logits) => Ok(logits),
            Err(error) => {
                self.reset();
                Err(error.context("LFM2 forward failed; its cache was reset"))
            }
        }
    }

    /// Read-only cache snapshots for numerical parity checks; no array is copied.
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
