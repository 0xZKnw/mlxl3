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

const DFLASH_CAPTURE_LAYERS: [usize; 8] = [1, 6, 11, 16, 22, 27, 32, 37];

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
    let weight = if checkpoint.model_type == "qwen3_5_mtp" {
        // The supported MLX sidecar already contains absolute-convention gains.
        weight.astype(Dtype::Float16)?
    } else {
        let one = Array::from_f16_bits(&[f16::ONE.to_bits()], &[])?.astype(weight.dtype())?;
        weight.add(&one)?.astype(Dtype::Float16)?
    };
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
        self.routed(x)?.add(&self.shared(x)?)
    }

    fn routed(&self, x: &Array) -> Result<Array> {
        let (selected, scores) = self.routes(x)?;
        self.experts.forward(x, &selected, &scores)
    }

    fn routes(&self, x: &Array) -> Result<(Array, Array)> {
        let probabilities = self.gate.forward(x)?.softmax_precise()?;
        router::topk(&probabilities, self.top_k, true)
    }

    fn shared(&self, x: &Array) -> Result<Array> {
        let shared_inputs = self.shared_inputs.forward(x)?;
        let shared = self
            .shared_down
            .forward(&shared_inputs[0].swiglu(&shared_inputs[1])?)?;
        shared.mul(&self.shared_multiplier.forward(x)?.sigmoid()?)
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

    fn forward_verification(&self, values: &[Array], hidden: i32) -> Result<Vec<Array>> {
        ensure!(
            !values.is_empty()
                && values.len() <= 8
                && values.iter().all(|value| value.shape() == [1, 1, hidden]),
            "invalid Qwen verification MLP input"
        );
        let Self::Moe(mlp) = self else {
            return values
                .iter()
                .map(|value| {
                    self.forward(&value.reshape(&[1, hidden])?)?
                        .reshape(&[1, 1, hidden])
                })
                .collect();
        };
        let rows = i32::try_from(values.len())?;
        let batch =
            Array::concatenate(&values.iter().collect::<Vec<_>>(), 1)?.reshape(&[rows, hidden])?;
        let gates = if rows == 1 {
            mlp.gate.forward(&batch)?
        } else {
            mlp.gate.forward_dense_rows_exact(&batch)?
        };
        let probabilities = gates.softmax_precise()?;
        let (selected, scores) = router::topk(&probabilities, mlp.top_k, true)?;
        let routed = mlp.experts.forward(&batch, &selected, &scores)?;
        let shared = mlp.shared(&batch)?;
        let output = routed.add(&shared)?;
        (0..rows)
            .map(|row| output.slice(0, row, row + 1)?.reshape(&[1, 1, hidden]))
            .collect()
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
    verification_conv_input: Option<Array>,
    verification_recurrent_tape: Option<gated_delta::RecurrentTape>,
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
    verification_base: Option<i32>,
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
            verification_base: None,
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
        self.verification_base = None;
        Ok(())
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        self.verification_base = None;
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

    /// Populate the same K/V cache when the attention output is unused.
    pub(crate) fn append_kv(&mut self, x: &Array) -> Result<()> {
        let time = *x
            .shape()
            .get(1)
            .context("Qwen cache input must have time")?;
        ensure!(
            x.shape().len() == 3 && x.shape()[0] == 1 && time > 0,
            "invalid Qwen cache input"
        );
        let offset = self.keys.as_ref().map_or(0, |keys| keys.shape()[2]);
        crate::contracts::mtp_cache_end(offset, time).context("MTP cache position overflow")?;
        let (k, v) = match &self.qkv {
            ProjectionBundle::Separate(projections) => {
                (projections[1].forward(x)?, projections[2].forward(x)?)
            }
            ProjectionBundle::Grouped(group) => {
                let qkv = group.forward(x)?;
                (qkv[1].try_clone()?, qkv[2].try_clone()?)
            }
        };
        let k = k
            .reshape(&[1, time, self.kv_heads, self.head_dim])?
            .rms_norm(&self.k_norm, self.eps)?
            .transpose(&[0, 2, 1, 3])?
            .rope(self.rope_dims, self.theta, offset)?;
        let v = v
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
        self.keys = Some(keys);
        self.values = Some(values);
        self.verification_base = None;
        Ok(())
    }

    fn forward_verification(&mut self, x: &Array) -> Result<Array> {
        self.forward_verification_impl(x, false)
    }

    fn forward_verification_dflash(&mut self, x: &Array) -> Result<Array> {
        self.forward_verification_impl(x, true)
    }

    fn forward_verification_impl(&mut self, x: &Array, retain_history: bool) -> Result<Array> {
        let time = *x
            .shape()
            .get(1)
            .context("Qwen attention input must have time")?;
        ensure!(
            x.shape().len() == 3 && x.shape()[0] == 1 && (1..=8).contains(&time),
            "invalid Qwen verification attention input"
        );
        self.verification_base =
            retain_history.then(|| self.keys.as_ref().map_or(0, |keys| keys.shape()[2]));
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

    fn commit_verification_prefix(&mut self, retained: i32, total: i32) -> Result<()> {
        let base = self
            .verification_base
            .take()
            .context("attention has no pending DFlash verification")?;
        ensure!(
            retained > 0 && retained <= total && total <= 8,
            "invalid attention verification commit"
        );
        if retained < total {
            let end = base + retained;
            self.keys = Some(
                self.keys
                    .as_ref()
                    .context("missing verified attention keys")?
                    .slice(2, 0, end)?,
            );
            self.values = Some(
                self.values
                    .as_ref()
                    .context("missing verified attention values")?
                    .slice(2, 0, end)?,
            );
        }
        Ok(())
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
        self.verification_base = None;
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
        self.forward_verification_impl(values, false)
    }

    fn forward_verification_dflash(&mut self, values: &mut [Array]) -> Result<()> {
        self.forward_verification_impl(values, true)
    }

    fn forward_verification_impl(
        &mut self,
        values: &mut [Array],
        retain_history: bool,
    ) -> Result<()> {
        let normalized = values
            .iter()
            .map(|value| value.rms_norm(&self.input_norm, self.eps))
            .collect::<Result<Vec<_>>>()?;
        let normalized = Array::concatenate(&normalized.iter().collect::<Vec<_>>(), 1)?;
        let attention = if retain_history {
            self.attention.forward_verification_dflash(&normalized)?
        } else {
            self.attention.forward_verification(&normalized)?
        };
        let mut residuals = Vec::with_capacity(values.len());
        let mut posts = Vec::with_capacity(values.len());
        for (time, value) in values.iter_mut().enumerate() {
            let hidden = value.add(&attention.slice(1, time as i32, time as i32 + 1)?)?;
            posts.push(hidden.rms_norm(&self.post_norm, self.eps)?);
            residuals.push(hidden);
        }
        let mlp = self.mlp.forward_verification(&posts, self.hidden)?;
        for ((value, hidden), output) in values.iter_mut().zip(residuals).zip(mlp) {
            *value = hidden.add(&output)?;
        }
        Ok(())
    }

    fn commit_verification_prefix(&mut self, retained: i32, total: i32) -> Result<()> {
        self.attention.commit_verification_prefix(retained, total)
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

#[derive(Clone)]
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
#[derive(Clone)]
pub struct QwenSnapshot {
    offset: i32,
    layers: Vec<LayerSnapshot>,
    last_hidden: Option<Array>,
}

impl QwenSnapshot {
    /// Logical payload, excluding shared weights and MLX allocator capacity.
    pub fn byte_len(&self) -> Result<usize> {
        self.layers
            .iter()
            .flat_map(|layer| match layer {
                LayerSnapshot::Linear { conv, recurrent } => [conv.as_ref(), recurrent.as_ref()],
                LayerSnapshot::Attention { keys, values } => [keys.as_ref(), values.as_ref()],
            })
            .flatten()
            .try_fold(
                self.last_hidden
                    .as_ref()
                    .map(Array::byte_len)
                    .transpose()?
                    .unwrap_or(0),
                |bytes, array| {
                    bytes
                        .checked_add(array.byte_len()?)
                        .context("snapshot size overflow")
                },
            )
    }
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

    fn forward_verification_dflash(&mut self, values: &mut [Array]) -> Result<()> {
        match self {
            Self::Linear(layer) => layer.forward_verification_dflash(values),
            Self::Attention(layer) => layer.forward_verification_dflash(values),
        }
    }

    fn commit_verification_prefix(&mut self, retained: i32, total: i32) -> Result<()> {
        match self {
            Self::Linear(layer) => layer.commit_verification_prefix(retained, total),
            Self::Attention(layer) => layer.commit_verification_prefix(retained, total),
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
    last_hidden: Option<Array>,
    mtp_layout: crate::mtp::Layout,
}

impl Qwen35Moe {
    pub fn load(path: &Path) -> Result<Self> {
        let checkpoint = crate::checkpoint::inspect(path)?;
        Self::from_checkpoint(&checkpoint)
    }

    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        let path = &checkpoint.path;
        let root: RootConfig = serde_json::from_reader(File::open(path.join("config.json"))?)?;
        let json: serde_json::Value =
            serde_json::from_reader(File::open(path.join("config.json"))?)?;
        let mtp_layout = serde_json::from_value(json["text_config"].clone())?;
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
            last_hidden: None,
            mtp_layout,
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

    pub fn prefill_serial(&mut self, tokens: &[u32]) -> Result<Array> {
        let result = (|| {
            let (&last, prefix) = tokens.split_last().context("empty Qwen prefill")?;
            for &token in prefix {
                let (hidden, _) = self.run_hidden_tokens(&[token], false)?;
                hidden.eval()?;
                self.offset += 1;
            }
            self.forward(last)
        })();
        if result.is_err() {
            self.reset();
        }
        result
    }

    pub fn prefill_serial_with_dflash_capture(&mut self, tokens: &[u32]) -> Result<(Array, Array)> {
        let result = (|| {
            let (&last, prefix) = tokens.split_last().context("empty Qwen prefill")?;
            let mut captured = Vec::with_capacity(tokens.len());
            for &token in prefix {
                let (hidden, capture) = self.run_hidden_tokens_with_dflash_capture(&[token])?;
                hidden.eval()?;
                capture.eval()?;
                self.offset += 1;
                captured.push(capture);
            }
            let (logits, capture) = self.forward_tokens_with_dflash_capture(&[last])?;
            captured.push(capture);
            let capture = Array::concatenate(&captured.iter().collect::<Vec<_>>(), 0)?;
            Ok((logits, capture))
        })();
        if result.is_err() {
            self.reset();
        }
        result
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

    pub fn forward_tokens_with_dflash_capture(&mut self, tokens: &[u32]) -> Result<(Array, Array)> {
        let result: Result<(Array, Array)> = (|| {
            let (hidden, captured) = self.run_hidden_tokens_with_dflash_capture(tokens)?;
            let logits = self.head.forward(&hidden)?;
            logits.eval()?;
            captured.eval()?;
            self.offset += i32::try_from(tokens.len())?;
            Ok((logits, captured))
        })();
        if result.is_err() {
            self.reset();
        }
        result.context("Qwen captured prefill failed; its cache was reset")
    }

    /// Runs target verification in the exact autoregressive order, then
    /// synchronizes the resulting Metal graph once.
    pub fn verify_tokens_exact(&mut self, tokens: &[u32]) -> Result<Array> {
        let result: Result<Array> = (|| {
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

    pub fn verify_tokens_exact_with_dflash_capture(
        &mut self,
        tokens: &[u32],
    ) -> Result<(Array, Array)> {
        let result: Result<(Array, Array)> = (|| {
            ensure!(
                !tokens.is_empty() && tokens.len() <= 8,
                "target verification requires 1 to 8 tokens"
            );
            let (hidden, captured) = self.run_hidden_tokens_layer_major_captured(tokens)?;
            let output = self.head.forward(&hidden)?;
            output.eval()?;
            captured.eval()?;
            self.offset += i32::try_from(tokens.len())?;
            Ok((output, captured))
        })();
        if result.is_err() {
            self.reset();
        }
        result.context("Qwen captured verification failed; its cache was reset")
    }

    pub fn dflash_input(&self, anchor: u32, mask: u32) -> Result<Array> {
        ensure!(
            anchor < self.vocab as u32 && mask < self.vocab as u32,
            "DFlash input token is outside vocabulary"
        );
        let mut tokens = [mask as i32; 8];
        tokens[0] = anchor as i32;
        self.embeddings
            .take(&Array::from_i32(&tokens, &[1, 8])?, 0)?
            .reshape(&[8, 2048])?
            .astype(Dtype::BFloat16)
    }

    pub fn dflash_logits(&self, hidden: &Array, positions: usize) -> Result<Array> {
        ensure!(
            hidden.shape() == [8, 2048] && (1..=7).contains(&positions),
            "DFlash head expects eight hidden rows"
        );
        let end = i32::try_from(positions)? + 1;
        self.head
            .forward(&hidden.slice(0, 1, end)?.astype(Dtype::Float16)?)
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
        self.last_hidden = Some(hidden.slice(1, time - 1, time)?);
        let normalized = hidden
            .slice(1, time - 1, time)?
            .rms_norm(&self.norm, self.eps)?;
        if trace {
            layers.push(normalized.to_f16_bits()?);
        }
        Ok((normalized, layers))
    }

    fn run_hidden_tokens_with_dflash_capture(&mut self, tokens: &[u32]) -> Result<(Array, Array)> {
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
        let mut captured = Vec::with_capacity(DFLASH_CAPTURE_LAYERS.len());
        for (index, layer) in self.layers.iter_mut().enumerate() {
            hidden = layer.forward(&hidden)?;
            if DFLASH_CAPTURE_LAYERS.contains(&index) {
                captured.push(hidden.try_clone()?);
            }
        }
        ensure!(
            captured.len() == DFLASH_CAPTURE_LAYERS.len(),
            "Qwen model is missing DFlash capture layers"
        );
        let capture = Array::concatenate(&captured.iter().collect::<Vec<_>>(), 2)?
            .reshape(&[time, 8 * 2048])?
            .astype(Dtype::BFloat16)?;
        let normalized = hidden
            .slice(1, time - 1, time)?
            .rms_norm(&self.norm, self.eps)?;
        Ok((normalized, capture))
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

    fn run_hidden_tokens_layer_major_captured(&mut self, tokens: &[u32]) -> Result<(Array, Array)> {
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
        let mut captured = Vec::with_capacity(DFLASH_CAPTURE_LAYERS.len());
        for (index, layer) in self.layers.iter_mut().enumerate() {
            layer.forward_verification_dflash(&mut hidden)?;
            if DFLASH_CAPTURE_LAYERS.contains(&index) {
                captured.push(Array::concatenate(&hidden.iter().collect::<Vec<_>>(), 1)?);
            }
        }
        ensure!(
            captured.len() == DFLASH_CAPTURE_LAYERS.len(),
            "Qwen model is missing DFlash capture layers"
        );
        let capture = Array::concatenate(&captured.iter().collect::<Vec<_>>(), 2)?
            .reshape(&[time, 8 * 2048])?
            .astype(Dtype::BFloat16)?;
        let normalized = hidden
            .into_iter()
            .map(|value| value.rms_norm(&self.norm, self.eps))
            .collect::<Result<Vec<_>>>()?;
        Ok((
            Array::concatenate(&normalized.iter().collect::<Vec<_>>(), 1)?,
            capture,
        ))
    }

    pub fn context_limit(&self) -> i32 {
        self.context_limit
    }

    pub fn mtp_layout(&self) -> &crate::mtp::Layout {
        &self.mtp_layout
    }
    pub fn mtp_hidden(&self) -> Result<&Array> {
        self.last_hidden
            .as_ref()
            .context("MTP target hidden is unavailable")
    }
    pub fn set_mtp_hidden(&mut self, hidden: Array) -> Result<()> {
        ensure!(
            hidden.shape() == [1, 1, self.mtp_layout.hidden_size],
            "invalid MTP target hidden"
        );
        self.last_hidden = Some(hidden);
        Ok(())
    }
    pub fn mtp_embeddings(&self, tokens: &[u32]) -> Result<Array> {
        ensure!(
            !tokens.is_empty() && tokens.iter().all(|&id| id < self.vocab as u32),
            "invalid MTP token IDs"
        );
        let ids = tokens.iter().map(|&id| id as i32).collect::<Vec<_>>();
        self.embeddings
            .take(&Array::from_i32(&ids, &[1, i32::try_from(ids.len())?])?, 0)
    }
    pub fn mtp_logits(&self, normalized: &Array) -> Result<Array> {
        self.head.forward(normalized)
    }

    /// Internal draft IDs come from an argmax over this target's vocabulary.
    /// Keeping them on device avoids one CPU synchronization per draft depth.
    pub(crate) fn mtp_embedding_ids(&self, ids: &Array) -> Result<Array> {
        ensure!(ids.shape() == [1, 1], "invalid MTP device token shape");
        self.embeddings.take(ids, 0)
    }

    /// Return pre-final-norm residuals without re-running the trunk. Used to
    /// prefill the MTP cache with real target positions, including long prompts.
    pub fn forward_mtp(&mut self, tokens: &[u32]) -> Result<(Array, Array)> {
        let result = (|| {
            ensure!(
                !tokens.is_empty() && tokens.iter().all(|&t| t < self.vocab as u32),
                "invalid MTP prefill tokens"
            );
            let time = i32::try_from(tokens.len())?;
            ensure!(
                time <= self.context_limit - self.offset,
                "MTP target context is full"
            );
            // Match the ordinary bridge's short-prefill execution shapes.
            // A different batch shape can otherwise change floating rounding.
            if time < 24 {
                let (&last, prefix) = tokens.split_last().unwrap();
                let mut raw = Vec::with_capacity(tokens.len());
                for &token in prefix {
                    self.run_hidden_tokens(&[token], false)?.0.eval()?;
                    self.offset += 1;
                    raw.push(self.mtp_hidden()?.try_clone()?);
                }
                let output = self.forward(last)?;
                raw.push(self.mtp_hidden()?.try_clone()?);
                let raw = Array::concatenate(&raw.iter().collect::<Vec<_>>(), 1)?;
                return Ok((output, raw));
            }
            let mut raw = self.mtp_embeddings(tokens)?;
            for layer in &mut self.layers {
                raw = layer.forward(&raw)?;
            }
            let last = raw.slice(1, time - 1, time)?;
            let logits = self.head.forward(&last.rms_norm(&self.norm, self.eps)?)?;
            logits.eval()?;
            self.last_hidden = Some(last);
            self.offset += time;
            Ok((logits, raw))
        })();
        if result.is_err() {
            self.reset();
        }
        result
    }

    pub fn verify_mtp(&mut self, tokens: &[u32]) -> Result<(Array, Array)> {
        let result = (|| {
            ensure!(
                (1..=8).contains(&tokens.len()) && tokens.iter().all(|&t| t < self.vocab as u32),
                "invalid MTP verification tokens"
            );
            let time = i32::try_from(tokens.len())?;
            ensure!(
                time <= self.context_limit - self.offset,
                "MTP verification exceeds context"
            );
            let mut values = tokens
                .iter()
                .map(|&t| self.mtp_embeddings(&[t]))
                .collect::<Result<Vec<_>>>()?;
            for layer in &mut self.layers {
                layer.forward_verification_dflash(&mut values)?;
            }
            let raw = Array::concatenate(&values.iter().collect::<Vec<_>>(), 1)?;
            let normalized = values
                .iter()
                .map(|value| value.rms_norm(&self.norm, self.eps))
                .collect::<Result<Vec<_>>>()?;
            let logits = self.head.forward(&Array::concatenate(
                &normalized.iter().collect::<Vec<_>>(),
                1,
            )?)?;
            logits.eval()?;
            self.offset += time;
            Ok((logits, raw))
        })();
        if result.is_err() {
            self.reset();
        }
        result
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// The shipped DFlash2 draft expects this hidden/capture layout, not a
    /// particular directory name. Target verification still decides every token.
    pub fn supports_dflash(&self) -> bool {
        self.embeddings.shape() == [248_320, 2048]
            && self.layers.len() == 40
            && self
                .layers
                .iter()
                .enumerate()
                .all(|(index, layer)| matches!(layer, Layer::Attention(_)) == (index % 4 == 3))
    }

    pub fn commit_dflash_verification(&mut self, retained: usize, total: usize) -> Result<()> {
        ensure!(
            retained > 0 && retained <= total && total <= 8,
            "invalid DFlash verification commit"
        );
        let retained = i32::try_from(retained)?;
        let total = i32::try_from(total)?;
        if let Err(error) = self
            .layers
            .iter_mut()
            .try_for_each(|layer| layer.commit_verification_prefix(retained, total))
        {
            self.reset();
            return Err(error.context("DFlash selective commit failed; model state was reset"));
        }
        self.offset -= total - retained;
        Ok(())
    }

    pub fn snapshot(&self) -> Result<QwenSnapshot> {
        Ok(QwenSnapshot {
            offset: self.offset,
            last_hidden: self.last_hidden.clone(),
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
        self.last_hidden = snapshot.last_hidden;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        self.last_hidden = None;
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
        self.forward_verification_impl(values, false)
    }

    fn forward_verification_dflash(&mut self, values: &mut [Array]) -> Result<()> {
        self.forward_verification_impl(values, true)
    }

    fn forward_verification_impl(
        &mut self,
        values: &mut [Array],
        retain_history: bool,
    ) -> Result<()> {
        let normalized = values
            .iter()
            .map(|value| value.rms_norm(&self.input_norm, self.eps))
            .collect::<Result<Vec<_>>>()?;
        let normalized = Array::concatenate(&normalized.iter().collect::<Vec<_>>(), 1)?;
        let attention = if retain_history {
            self.attention.forward_verification_dflash(&normalized)?
        } else {
            self.attention.forward_verification(&normalized)?
        };
        let mut residuals = Vec::with_capacity(values.len());
        let mut posts = Vec::with_capacity(values.len());
        for (time, value) in values.iter_mut().enumerate() {
            let hidden = value.add(&attention.slice(1, time as i32, time as i32 + 1)?)?;
            posts.push(hidden.rms_norm(&self.post_norm, self.eps)?);
            residuals.push(hidden);
        }
        let mlp = self.mlp.forward_verification(&posts, self.hidden)?;
        for ((value, hidden), output) in values.iter_mut().zip(residuals).zip(mlp) {
            *value = hidden.add(&output)?;
        }
        Ok(())
    }

    fn commit_verification_prefix(&mut self, retained: i32, total: i32) -> Result<()> {
        self.attention.commit_verification_prefix(retained, total)
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
            verification_conv_input: None,
            verification_recurrent_tape: None,
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
        self.verification_conv_input = None;
        self.verification_recurrent_tape = None;
        Ok(())
    }

    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        self.verification_conv_input = None;
        self.verification_recurrent_tape = None;
        Ok(self.run(x, false)?.0)
    }

    fn forward_verification(&mut self, x: &Array) -> Result<Array> {
        self.forward_verification_impl(x, false)
    }

    fn forward_verification_dflash(&mut self, x: &Array) -> Result<Array> {
        self.forward_verification_impl(x, true)
    }

    fn prepare_recurrence_inputs(
        &mut self,
        qkv: &Array,
        time: i32,
        retain_history: bool,
        fused: bool,
    ) -> Result<(Array, Array, Array, Option<Array>)> {
        let keys = self.key_heads * self.key_dim;
        let values = self.value_heads * self.value_dim;
        let conv_dims = 2 * keys + values;
        let q_scale =
            Array::from_f16_bits(&[f16::from_f32(1.0 / self.key_dim as f32).to_bits()], &[])?;
        let k_scale = Array::from_f16_bits(
            &[f16::from_f32(1.0 / (self.key_dim as f32).sqrt()).to_bits()],
            &[],
        )?;
        let conv_state = match &self.conv_state {
            Some(state) => state.try_clone()?,
            None => Array::zeros_dtype(&[1, self.conv_length - 1, conv_dims], Dtype::Float16)?,
        };
        if fused {
            self.verification_conv_input = if retain_history {
                Some(Array::concatenate(&[&conv_state, qkv], 1)?)
            } else {
                None
            };
            let (q, k, v, state) = gated_delta::prepare_qkv(
                qkv,
                &conv_state,
                &self.conv_weight,
                &q_scale,
                &k_scale,
                self.key_heads,
                self.value_heads,
                self.key_dim,
                self.value_dim,
                self.conv_length,
            )?;
            self.conv_state = Some(state);
            return Ok((q, k, v, None));
        }
        let conv_input = Array::concatenate(&[&conv_state, qkv], 1)?;
        self.conv_state = Some(conv_input.slice(1, time, time + self.conv_length - 1)?);
        self.verification_conv_input = retain_history.then(|| conv_input.clone());
        let conv = conv_input.conv1d(&self.conv_weight, conv_dims)?.silu()?;
        let q = conv
            .slice(2, 0, keys)?
            .reshape(&[1, time, self.key_heads, self.key_dim])?
            .rms_norm_without_weight(1e-6)?
            .mul(&q_scale)?;
        let k = conv
            .slice(2, keys, 2 * keys)?
            .reshape(&[1, time, self.key_heads, self.key_dim])?
            .rms_norm_without_weight(1e-6)?
            .mul(&k_scale)?;
        let v = conv.slice(2, 2 * keys, conv_dims)?.reshape(&[
            1,
            time,
            self.value_heads,
            self.value_dim,
        ])?;
        Ok((q, k, v, Some(conv)))
    }

    fn forward_verification_impl(&mut self, x: &Array, retain_history: bool) -> Result<Array> {
        let time = *x.shape().get(1).context("Qwen GDN input must have time")?;
        ensure!(
            x.shape() == [1, time, self.hidden] && (1..=8).contains(&time),
            "invalid Qwen verification GDN input"
        );
        let values = self.value_heads * self.value_dim;
        let inputs = self.inputs.forward(x)?;
        let qkv = inputs[0].try_clone()?;
        let z = inputs[1].reshape(&[1, time, self.value_heads, self.value_dim])?;
        let (b, a) = if retain_history && time > 1 {
            let project = |projection: &Projection| -> Result<Array> {
                let rows = (0..time)
                    .map(|row| projection.forward(&x.slice(1, row, row + 1)?))
                    .collect::<Result<Vec<_>>>()?;
                Array::concatenate(&rows.iter().collect::<Vec<_>>(), 1)
            };
            (project(&self.b)?, project(&self.a)?)
        } else {
            (self.b.forward(x)?, self.a.forward(x)?)
        };
        let (q, k, v, _) = self.prepare_recurrence_inputs(&qkv, time, retain_history, true)?;
        let recurrent_state = match &self.recurrent_state {
            Some(state) => state.try_clone()?,
            None => Array::zeros_dtype(
                &[1, self.value_heads, self.value_dim, self.key_dim],
                Dtype::Float32,
            )?,
        };
        let (out, recurrent_state) = if retain_history {
            let (out, state, tape) = gated_delta::step_with_gates_tape(
                &q,
                &k,
                &v,
                &a,
                &b,
                &self.a_log,
                &self.dt_bias,
                &recurrent_state,
            )?;
            self.verification_recurrent_tape = Some(tape);
            (out, state)
        } else {
            self.verification_recurrent_tape = None;
            gated_delta::step_with_gates(
                &q,
                &k,
                &v,
                &a,
                &b,
                &self.a_log,
                &self.dt_bias,
                &recurrent_state,
            )?
        };
        self.recurrent_state = Some(recurrent_state);
        let normalized = out.rms_norm(&self.norm, self.eps)?;
        self.output
            .forward(&z.precise_swiglu(&normalized)?.reshape(&[1, time, values])?)
    }

    fn commit_verification_prefix(&mut self, retained: i32, total: i32) -> Result<()> {
        ensure!(
            retained > 0 && retained <= total && total <= 8,
            "invalid GDN verification commit"
        );
        let conv_input = self
            .verification_conv_input
            .take()
            .context("GDN has no pending DFlash convolution history")?;
        let recurrent_tape = self
            .verification_recurrent_tape
            .take()
            .context("GDN has no pending DFlash recurrent history")?;
        if retained < total {
            self.conv_state =
                Some(conv_input.slice(1, retained, retained + self.conv_length - 1)?);
            self.recurrent_state = Some(recurrent_tape.replay(retained)?);
        }
        Ok(())
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
        let values = self.value_heads * self.value_dim;
        let inputs = self.inputs.forward(x)?;
        let qkv = inputs[0].try_clone()?;
        let z = inputs[1].reshape(&[1, time, self.value_heads, self.value_dim])?;
        let b = self.b.forward(x)?;
        let a = self.a.forward(x)?;
        let use_fused = !trace && time <= 8;
        let (q, k, v, conv) = self.prepare_recurrence_inputs(&qkv, time, false, use_fused)?;
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
                conv.context("missing traced Gated DeltaNet convolution")?,
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
        self.verification_conv_input = None;
        self.verification_recurrent_tape = None;
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

    #[test]
    #[ignore = "requires local Qwen checkpoint and physical Apple GPU"]
    fn mtp_prefill_verification_and_rollback_match_target() -> Result<()> {
        let mut model = Qwen35Moe::load(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        for time in [1, 23, 24, 129, 256] {
            let prefix = (0..time).map(|i| 1 + i % 31).collect::<Vec<_>>();
            model.reset();
            let expected = if time < 24 {
                model.prefill_serial(&prefix)?
            } else {
                model.forward_tokens(&prefix)?
            }
            .to_f16_bits()?;
            let state = state_bytes(&model)?;
            model.reset();
            let (actual, raw) = model.forward_mtp(&prefix)?;
            assert_eq!(actual.to_f16_bits()?, expected, "MTP prefill {time}");
            assert_eq!(state_bytes(&model)?, state, "MTP prefill states {time}");
            assert_eq!(raw.shape(), [1, time as i32, model.mtp_layout.hidden_size]);
            let saved = model.snapshot()?;
            for total in 2..=4 {
                let verification = (1..=total as u32).collect::<Vec<_>>();
                for retained in 1..=total {
                    model.restore(saved.clone())?;
                    let mut expected_logits = Vec::new();
                    for &token in &verification {
                        expected_logits.extend(model.forward(token)?.to_f16_bits()?);
                    }
                    model.restore(saved.clone())?;
                    for token in &verification[..retained] {
                        model.forward(*token)?.eval()?;
                    }
                    let states = state_bytes(&model)?;
                    let hidden = model.mtp_hidden()?.to_f16_bits()?;
                    model.restore(saved.clone())?;
                    let (actual, raw) = model.verify_mtp(&verification)?;
                    assert_eq!(actual.to_f16_bits()?, expected_logits, "MTP verify {time}");
                    model.commit_dflash_verification(retained, total)?;
                    model.set_mtp_hidden(raw.slice(1, retained as i32 - 1, retained as i32)?)?;
                    assert_eq!(
                        state_bytes(&model)?,
                        states,
                        "MTP rollback {time}/{retained}"
                    );
                    assert_eq!(model.mtp_hidden()?.to_f16_bits()?, hidden);
                }
            }
        }
        for invalid in [vec![], vec![model.vocab as u32], vec![1; 9]] {
            assert!(model.verify_mtp(&invalid).is_err());
            assert_eq!(
                model.offset(),
                0,
                "failed verification must discard partial state"
            );
        }
        assert!(model.forward_mtp(&[]).is_err());
        assert!(model.forward_mtp(&[model.vocab as u32]).is_err());
        model.context_limit = 1;
        model.forward_mtp(&[1])?;
        assert!(model.verify_mtp(&[1]).is_err());
        assert_eq!(model.offset(), 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen/MTP checkpoints and physical Apple GPU"]
    fn mtp_recursive_sessions_match_greedy_and_exact_caches() -> Result<()> {
        use crate::mtp::{Head, Session};
        let mut model = Qwen35Moe::load(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        let mut head = Head::load(Path::new("models/Qwen3.6-35B-A3B-MTP-4bit"), &model)?;
        for depth in 1..=3 {
            model.reset();
            head.reset()?;
            let prefix = vec![1; 24];
            let (logits, raw) = model.forward_mtp(&prefix)?;
            head.extend_cache(&model, &raw.slice(1, 0, 23)?, &prefix[1..])?;
            let mut anchor = logits.chat_greedy_ids()?[0];
            let mut session = Session::new(depth)?;
            for block in 0..8 {
                let saved = model.snapshot()?;
                let head_saved = head.snapshot()?;
                let start = model.offset();
                let first = session.advance(&mut model, &mut head, anchor, 512, 64)?;
                let retained = (model.offset() - start) as usize;
                assert!((1..=depth + 1).contains(&retained));
                let mut actual = vec![first];
                assert!(
                    session
                        .advance(&mut model, &mut head, first, 512, 0)
                        .is_err()
                );
                for _ in 1..retained {
                    let next =
                        session.advance(&mut model, &mut head, *actual.last().unwrap(), 512, 64)?;
                    actual.push(next);
                }
                let actual_state = state_bytes(&model)?;
                let actual_hidden = model.mtp_hidden()?.to_bytes()?;
                let actual_head = head.cache_bytes()?;
                model.restore(saved)?;
                head.restore(head_saved)?;
                let mut expected = Vec::new();
                for _ in 0..retained {
                    let previous_hidden = model.mtp_hidden()?.try_clone()?;
                    head.extend_cache(&model, &previous_hidden, &[anchor])?;
                    anchor = model.forward(anchor)?.chat_greedy_ids()?[0];
                    expected.push(anchor);
                }
                assert_eq!(
                    actual, expected,
                    "target tokens depth={depth} block={block}"
                );
                assert_eq!(
                    actual_state,
                    state_bytes(&model)?,
                    "target states depth={depth} block={block}"
                );
                assert_eq!(actual_hidden, model.mtp_hidden()?.to_bytes()?);
                let expected_head = head.cache_bytes()?;
                assert!(
                    actual_head == expected_head,
                    "head KV depth={depth} block={block}, first key/value byte differences={:?}/{:?}",
                    actual_head
                        .0
                        .iter()
                        .zip(&expected_head.0)
                        .position(|(a, b)| a != b),
                    actual_head
                        .1
                        .iter()
                        .zip(&expected_head.1)
                        .position(|(a, b)| a != b)
                );
            }
            eprintln!(
                "MTP{depth}: {}/{} accepted in {} exact blocks",
                session.accepted, session.proposed, session.blocks
            );
        }
        assert!(Session::new(0).is_err());
        assert!(Session::new(4).is_err());
        Ok(())
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
    fn dense_gate_rows_match_serial() -> Result<()> {
        let checkpoint =
            crate::checkpoint::inspect(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        let gate = Projection::load(
            &checkpoint,
            "model.language_model.layers.0.mlp.gate",
            2048,
            256,
            false,
        )?;
        for rows in [2, 4, 8] {
            let values = (0..rows * 2048)
                .map(|index| f16::from_f32(((index % 251) as f32 - 125.0) / 128.0).to_bits())
                .collect::<Vec<_>>();
            let input = Array::from_f16_bits(&values, &[rows, 2048])?;
            let serial = (0..rows)
                .map(|row| gate.forward(&input.slice(0, row, row + 1)?))
                .collect::<Result<Vec<_>>>()?;
            let serial = Array::concatenate(&serial.iter().collect::<Vec<_>>(), 0)?;
            let actual = gate.forward_dense_rows_exact(&input)?;
            assert_eq!(
                actual.to_f16_bits()?,
                serial.to_f16_bits()?,
                "dense gate differs at M={rows}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen checkpoint and Apple GPU"]
    fn fused_gdn_preparation_matches_mlx() -> Result<()> {
        let mut model = Qwen35Moe::load(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        let Layer::Linear(layer) = &mut model.layers[0] else {
            anyhow::bail!("Qwen layer 0 is not Gated DeltaNet");
        };
        let gdn = &layer.attention;
        let keys = gdn.key_heads * gdn.key_dim;
        let values = gdn.value_heads * gdn.value_dim;
        let conv_dims = 2 * keys + values;
        let q_scale =
            Array::from_f16_bits(&[f16::from_f32(1.0 / gdn.key_dim as f32).to_bits()], &[])?;
        let k_scale = Array::from_f16_bits(
            &[f16::from_f32(1.0 / (gdn.key_dim as f32).sqrt()).to_bits()],
            &[],
        )?;
        let state_values = (0..(gdn.conv_length - 1) * conv_dims)
            .map(|index| f16::from_f32(((index % 127) as f32 - 63.0) / 128.0).to_bits())
            .collect::<Vec<_>>();
        let state = Array::from_f16_bits(&state_values, &[1, gdn.conv_length - 1, conv_dims])?;
        for time in [1, 6, 8] {
            let qkv_values = (0..time * conv_dims)
                .map(|index| f16::from_f32(((index % 251) as f32 - 125.0) / 128.0).to_bits())
                .collect::<Vec<_>>();
            let qkv = Array::from_f16_bits(&qkv_values, &[1, time, conv_dims])?;
            let conv_input = Array::concatenate(&[&state, &qkv], 1)?;
            let expected_state = conv_input.slice(1, time, time + gdn.conv_length - 1)?;
            let conv = conv_input.conv1d(&gdn.conv_weight, conv_dims)?.silu()?;
            let expected_q = conv
                .slice(2, 0, keys)?
                .reshape(&[1, time, gdn.key_heads, gdn.key_dim])?
                .rms_norm_without_weight(1e-6)?
                .mul(&q_scale)?;
            let expected_k = conv
                .slice(2, keys, 2 * keys)?
                .reshape(&[1, time, gdn.key_heads, gdn.key_dim])?
                .rms_norm_without_weight(1e-6)?
                .mul(&k_scale)?;
            let expected_v = conv.slice(2, 2 * keys, conv_dims)?.reshape(&[
                1,
                time,
                gdn.value_heads,
                gdn.value_dim,
            ])?;
            let (q, k, v, next_state) = gated_delta::prepare_qkv(
                &qkv,
                &state,
                &gdn.conv_weight,
                &q_scale,
                &k_scale,
                gdn.key_heads,
                gdn.value_heads,
                gdn.key_dim,
                gdn.value_dim,
                gdn.conv_length,
            )?;
            for (name, actual, expected) in [
                ("q", q, expected_q),
                ("k", k, expected_k),
                ("v", v, expected_v),
                ("state", next_state, expected_state),
            ] {
                assert_eq!(
                    actual.to_f16_bits()?,
                    expected.to_f16_bits()?,
                    "fused GDN {name} differs at T={time}"
                );
            }
            if std::env::var_os("MLXL3_GDN_PREP_BENCH").is_some() {
                let reference = || -> Result<(Array, Array, Array, Array)> {
                    let conv_input = Array::concatenate(&[&state, &qkv], 1)?;
                    let next_state = conv_input.slice(1, time, time + gdn.conv_length - 1)?;
                    let conv = conv_input.conv1d(&gdn.conv_weight, conv_dims)?.silu()?;
                    let q = conv
                        .slice(2, 0, keys)?
                        .reshape(&[1, time, gdn.key_heads, gdn.key_dim])?
                        .rms_norm_without_weight(1e-6)?
                        .mul(&q_scale)?;
                    let k = conv
                        .slice(2, keys, 2 * keys)?
                        .reshape(&[1, time, gdn.key_heads, gdn.key_dim])?
                        .rms_norm_without_weight(1e-6)?
                        .mul(&k_scale)?;
                    let v = conv.slice(2, 2 * keys, conv_dims)?.reshape(&[
                        1,
                        time,
                        gdn.value_heads,
                        gdn.value_dim,
                    ])?;
                    Ok((q, k, v, next_state))
                };
                let fused = || {
                    gated_delta::prepare_qkv(
                        &qkv,
                        &state,
                        &gdn.conv_weight,
                        &q_scale,
                        &k_scale,
                        gdn.key_heads,
                        gdn.value_heads,
                        gdn.key_dim,
                        gdn.value_dim,
                        gdn.conv_length,
                    )
                };
                let evaluate = |output: (Array, Array, Array, Array)| -> Result<()> {
                    output.0.eval()?;
                    output.1.eval()?;
                    output.2.eval()?;
                    output.3.eval()
                };
                for _ in 0..8 {
                    evaluate(reference()?)?;
                    evaluate(fused()?)?;
                }
                let mut reference_ms = Vec::with_capacity(40);
                let mut fused_ms = Vec::with_capacity(40);
                for index in 0..40 {
                    for use_fused in [index % 2 == 0, index % 2 != 0] {
                        let started = Instant::now();
                        evaluate(if use_fused { fused()? } else { reference()? })?;
                        let elapsed = started.elapsed().as_secs_f64() * 1000.0;
                        if use_fused {
                            fused_ms.push(elapsed);
                        } else {
                            reference_ms.push(elapsed);
                        }
                    }
                }
                reference_ms.sort_by(f64::total_cmp);
                fused_ms.sort_by(f64::total_cmp);
                eprintln!(
                    "GDN prepare T={time}: fused={:.3} ms reference={:.3} ms delta={:+.1}%",
                    fused_ms[fused_ms.len() / 2],
                    reference_ms[reference_ms.len() / 2],
                    (fused_ms[fused_ms.len() / 2] / reference_ms[reference_ms.len() / 2] - 1.0)
                        * 100.0
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen checkpoint and Apple GPU"]
    fn qwen_serial_prefill_skips_only_unused_heads() -> Result<()> {
        let mut model = Qwen35Moe::load(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        for length in [1, 2, 23, 24, 25] {
            let tokens: Vec<_> = (1..=length).collect();
            model.reset();
            let mut expected = None;
            for &token in &tokens {
                expected = Some(model.forward(token)?);
            }
            let expected = expected.unwrap().to_f16_bits()?;
            let expected_state = state_bytes(&model)?;
            model.reset();
            assert_eq!(model.prefill_serial(&tokens)?.to_f16_bits()?, expected);
            assert_eq!(state_bytes(&model)?, expected_state);
            assert_eq!(model.offset, length as i32);
            let mut times = [Vec::new(), Vec::new()];
            for fast in [false, true, true, false, false, true, true, false] {
                model.reset();
                let started = std::time::Instant::now();
                if fast {
                    model.prefill_serial(&tokens)?.eval()?;
                } else {
                    for &token in &tokens {
                        model.forward(token)?.eval()?;
                    }
                }
                times[usize::from(fast)].push(started.elapsed().as_secs_f64() * 1000.);
            }
            for sample in &mut times {
                sample.sort_by(f64::total_cmp);
            }
            eprintln!(
                "Qwen serial prefill {length}: full head {:.3} ms, final head {:.3} ms; logits + 80 states exact",
                (times[0][1] + times[0][2]) / 2.,
                (times[1][1] + times[1][2]) / 2.
            );
        }
        assert!(model.prefill_serial(&[u32::MAX]).is_err());
        assert_eq!(model.offset, 0);
        Ok(())
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
        // Bridge cache checkpoints sit on full prefill chunks, not only M=1.
        for chunk in [128, 256] {
            model.reset();
            let prefix: Vec<_> = (0..chunk).map(|index| (index % 97 + 1) as u32).collect();
            model.forward_tokens(&prefix)?.eval()?;
            let snapshot = model.snapshot()?;
            let expected_logits = model.forward_tokens(&[42; 24])?.to_f16_bits()?;
            let expected_state = state_bytes(&model)?;
            model.restore(snapshot)?;
            assert_eq!(
                model.forward_tokens(&[42; 24])?.to_f16_bits()?,
                expected_logits
            );
            assert_eq!(
                state_bytes(&model)?,
                expected_state,
                "state after chunk {chunk}"
            );
            assert_eq!(model.offset, chunk + 24);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen checkpoint and Apple GPU"]
    fn long_prompt_dflash_capture_keeps_target_exact() -> Result<()> {
        use crate::tokenizer::ChatTokenizer;
        let path = Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw");
        let tokenizer = ChatTokenizer::load(path)?;
        let prompt = std::fs::read_to_string("docs/audit-runtime-2026-09-04.md")?;
        let rendered = tokenizer.render_values(
            &serde_json::json!([{"role":"user", "content":prompt}]),
            None,
        )?;
        let tokens = tokenizer.encode(&rendered)?;
        assert_eq!(tokens.len(), 2825);
        let mut model = Qwen35Moe::load(path)?;
        let mut run = |capture: bool| -> Result<_> {
            model.reset();
            let mut logits = None;
            for part in tokens.chunks(crate::dflash::PREFILL_CHUNK) {
                logits = Some(if capture {
                    model.forward_tokens_with_dflash_capture(part)?.0
                } else {
                    model.forward_tokens(part)?
                });
            }
            Ok((
                logits.context("empty prompt")?.to_f16_bits()?,
                state_bytes(&model)?,
            ))
        };
        let expected = run(false)?;
        let actual = run(true)?;
        ensure!(actual.0 == expected.0, "capture changed final logits");
        ensure!(actual.1 == expected.1, "capture changed final states");
        model.reset();
        let (prefix, tail) = tokens
            .split_at(tokens.len() / crate::dflash::PREFILL_CHUNK * crate::dflash::PREFILL_CHUNK);
        for part in prefix.chunks(crate::dflash::PREFILL_CHUNK) {
            model.forward_tokens(part)?;
        }
        let expected_logits = model.prefill_serial(tail)?.to_f16_bits()?;
        let expected_state = state_bytes(&model)?;
        model.reset();
        for part in prefix.chunks(crate::dflash::PREFILL_CHUNK) {
            model.forward_tokens_with_dflash_capture(part)?;
        }
        let (actual_logits, captured) = model.prefill_serial_with_dflash_capture(tail)?;
        ensure!(
            actual_logits.to_f16_bits()? == expected_logits,
            "serial tail logits"
        );
        ensure!(state_bytes(&model)? == expected_state, "serial tail states");
        assert_eq!(captured.shape(), [tail.len() as i32, 8 * 2048]);
        for length in [1, 23] {
            model.reset();
            model.forward_tokens(&tokens[..crate::dflash::PREFILL_CHUNK])?;
            let expected_logits = model.prefill_serial(&tokens[..length])?.to_f16_bits()?;
            let expected_state = state_bytes(&model)?;
            model.reset();
            model.forward_tokens_with_dflash_capture(&tokens[..crate::dflash::PREFILL_CHUNK])?;
            let (actual_logits, captured) =
                model.prefill_serial_with_dflash_capture(&tokens[..length])?;
            ensure!(
                actual_logits.to_f16_bits()? == expected_logits,
                "serial tail logits at {length}"
            );
            ensure!(
                state_bytes(&model)? == expected_state,
                "serial tail states at {length}"
            );
            assert_eq!(captured.shape(), [length as i32, 8 * 2048]);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen checkpoint and Apple GPU"]
    fn long_prefix_verification_keeps_exact_state() -> Result<()> {
        use crate::tokenizer::ChatTokenizer;
        let path = Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw");
        let tokenizer = ChatTokenizer::load(path)?;
        let prompt = std::fs::read_to_string("docs/audit-runtime-2026-09-04.md")?;
        let tokens = tokenizer.encode(&tokenizer.render_values(
            &serde_json::json!([{"role":"user", "content":prompt}]),
            None,
        )?)?;
        let mut model = Qwen35Moe::load(path)?;
        let mut logits = None;
        for part in tokens.chunks(crate::dflash::PREFILL_CHUNK) {
            logits = Some(model.forward_tokens(part)?);
        }
        let mut logits = logits.context("empty prompt")?;
        for generated in [0, 48] {
            let saved = model.snapshot()?;
            let verify = [1, 2, 3, 4, 5, 6];
            for retained in 1..=verify.len() {
                model.restore(saved.clone())?;
                let expected = batched_head_token_major(&mut model, &verify[..retained])?;
                let expected_logits = (retained == verify.len())
                    .then(|| expected.to_f16_bits())
                    .transpose()?;
                let expected_state = state_bytes(&model)?;
                model.restore(saved.clone())?;
                let (actual, _) = model.verify_tokens_exact_with_dflash_capture(&verify)?;
                let mismatch = if let Some(expected_logits) = expected_logits {
                    actual
                        .to_f16_bits()?
                        .iter()
                        .zip(&expected_logits)
                        .position(|(left, right)| left != right)
                } else {
                    None
                };
                model.commit_dflash_verification(retained, verify.len())?;
                let actual_state = state_bytes(&model)?;
                let state_mismatch = actual_state
                    .iter()
                    .zip(&expected_state)
                    .position(|(left, right)| left != right);
                eprintln!(
                    "verify after {generated}, retained={retained}: logit row {:?}, state index {state_mismatch:?}",
                    mismatch.map(|index| index / model.vocab as usize)
                );
                ensure!(
                    state_mismatch.is_none(),
                    "verification state differs after {generated} generated tokens, retained={retained}, first state={}",
                    state_mismatch.unwrap()
                );
                ensure!(
                    mismatch.is_none(),
                    "verification logits differ after {generated} generated tokens, retained={retained}, first row={}",
                    mismatch.unwrap() / model.vocab as usize
                );
            }
            model.restore(saved)?;
            if generated == 0 {
                for _ in 0..48 {
                    let next = logits.chat_greedy_ids()?[0];
                    logits = model.forward(next)?;
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen checkpoint and Apple GPU"]
    fn captured_prefill_keeps_uncaptured_target_state() -> Result<()> {
        let mut model = Qwen35Moe::load(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        for length in [23, 24, 127, 128, 129] {
            model.reset();
            model.forward_tokens(&[37; 128])?.eval()?;
            let snapshot = model.snapshot()?;
            let tokens: Vec<_> = (0..length).map(|i| (i % 97 + 1) as u32).collect();
            let expected = model.forward_tokens(&tokens)?.to_f16_bits()?;
            let states = state_bytes(&model)?;
            model.restore(snapshot)?;
            let (actual, captured) = model.forward_tokens_with_dflash_capture(&tokens)?;
            assert_eq!(
                actual.to_f16_bits()?,
                expected,
                "capture changed logits at {length}"
            );
            assert_eq!(
                state_bytes(&model)?,
                states,
                "capture changed states at {length}"
            );
            assert_eq!(captured.shape(), [length, 8 * 2048]);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen checkpoint and Apple GPU"]
    fn verification_widths_match_token_major() -> Result<()> {
        let mut model = Qwen35Moe::load(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        for width in 1..=8 {
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
    fn dflash_limited_head_matches_full_rows() -> Result<()> {
        let model = Qwen35Moe::load(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        let hidden = model.dflash_input(1, 2)?;
        let limited = model.dflash_logits(&hidden, 5)?.to_f16_bits()?;
        let full = model
            .head
            .forward(&hidden.astype(Dtype::Float16)?)?
            .slice(0, 1, 6)?
            .to_f16_bits()?;
        assert_eq!(limited, full);
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen checkpoint and Apple GPU"]
    fn selective_dflash_commit_matches_exact_prefix() -> Result<()> {
        let mut model = Qwen35Moe::load(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        let tokens = [4, 5, 6, 7, 8, 9, 10, 11];
        for retained in 1..=8 {
            model.reset();
            model.forward_tokens(&[1, 2, 3])?;
            model.verify_tokens_exact(&tokens[..retained])?;
            let expected = state_bytes(&model)?;

            model.reset();
            model.forward_tokens(&[1, 2, 3])?;
            model.verify_tokens_exact_with_dflash_capture(&tokens)?;
            model.commit_dflash_verification(retained, tokens.len())?;
            assert_eq!(model.offset(), 3 + retained as i32);
            assert_eq!(
                state_bytes(&model)?,
                expected,
                "selective DFlash state differs at M={retained}"
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
