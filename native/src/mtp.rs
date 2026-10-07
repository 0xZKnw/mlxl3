//! Native Qwen3.5/3.6/3.8 NextN predictor. Architecture follows Qwen and the
//! MLX/MTPLX reference: pre-norm trunk hidden, [embedding, hidden] concat,
//! one full-attention block, absolute norm gains, shared target lm_head.
use crate::checkpoint::{Checkpoint, read_header};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{fs::File, path::Path};

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Layout {
    pub hidden_size: i32,
    pub num_attention_heads: i32,
    pub num_key_value_heads: i32,
    pub head_dim: i32,
    pub vocab_size: i32,
    #[serde(default)]
    pub num_experts: i32,
    #[serde(default)]
    pub num_experts_per_tok: usize,
    #[serde(default)]
    pub moe_intermediate_size: i32,
    #[serde(default)]
    pub shared_expert_intermediate_size: i32,
    pub intermediate_size: Option<i32>,
    pub rms_norm_eps: f32,
    pub partial_rotary_factor: f32,
    pub rope_parameters: Rope,
}
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Rope {
    pub rope_theta: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ManagedHead {
    Qwen38Dense,
    Qwen36Moe,
}

impl Layout {
    /// Choose by the actual architecture, including vocabulary and RoPE, so
    /// renamed EXL3 folders work and unrelated Qwen sizes never download a head.
    pub fn managed_head(&self, layers: usize) -> Option<ManagedHead> {
        if self.vocab_size != 248_320
            || self.head_dim != 256
            || self.rms_norm_eps != 1e-6
            || self.partial_rotary_factor != 0.25
            || self.rope_parameters.rope_theta != 10_000_000.
        {
            return None;
        }
        match (
            self.hidden_size,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.num_experts,
            layers,
        ) {
            (5120, 24, 4, 0, 64)
                if self.intermediate_size == Some(17_408)
                    && self.num_experts_per_tok == 0
                    && self.moe_intermediate_size == 0
                    && self.shared_expert_intermediate_size == 0 =>
            {
                Some(ManagedHead::Qwen38Dense)
            }
            (2048, 16, 2, 256, 40)
                if self.num_experts_per_tok == 8
                    && self.intermediate_size.is_none()
                    && self.moe_intermediate_size == 512
                    && self.shared_expert_intermediate_size == 512 =>
            {
                Some(ManagedHead::Qwen36Moe)
            }
            _ => None,
        }
    }
}

pub fn target_layout(path: &Path) -> Result<(Layout, usize)> {
    let config: serde_json::Value = serde_json::from_reader(File::open(path.join("config.json"))?)?;
    ensure!(
        matches!(
            config["model_type"].as_str(),
            Some("qwen3_5" | "qwen3_5_moe")
        ),
        "MTP requires a Qwen3.5/3.6/3.8 target"
    );
    let text = &config["text_config"];
    ensure!(
        text["mtp_use_dedicated_embeddings"].as_bool() != Some(true),
        "MTP requires shared target embeddings"
    );
    let layers = usize::try_from(
        text["num_hidden_layers"]
            .as_u64()
            .context("missing target layer count")?,
    )?;
    Ok((serde_json::from_value(text.clone())?, layers))
}

#[cfg(kani)]
#[kani::proof]
fn managed_mtp_never_crosses_architectures() {
    let layout = Layout {
        hidden_size: kani::any(),
        num_attention_heads: kani::any(),
        num_key_value_heads: kani::any(),
        head_dim: kani::any(),
        vocab_size: kani::any(),
        num_experts: kani::any(),
        num_experts_per_tok: kani::any(),
        moe_intermediate_size: kani::any(),
        shared_expert_intermediate_size: kani::any(),
        intermediate_size: kani::any(),
        rms_norm_eps: kani::any(),
        partial_rotary_factor: kani::any(),
        rope_parameters: Rope {
            rope_theta: kani::any(),
        },
    };
    let layers: usize = kani::any();
    let result = layout.managed_head(layers);
    if let Some(kind) = result {
        assert!(layout.vocab_size == 248_320 && layout.head_dim == 256);
        assert!(layout.rms_norm_eps == 1e-6 && layout.partial_rotary_factor == 0.25);
        assert!(layout.rope_parameters.rope_theta == 10_000_000.);
        match kind {
            ManagedHead::Qwen38Dense => {
                assert!(layout.hidden_size == 5120 && layers == 64);
                assert!(layout.num_attention_heads == 24 && layout.num_key_value_heads == 4);
                assert!(layout.intermediate_size == Some(17_408) && layout.num_experts == 0);
                assert!(
                    layout.num_experts_per_tok == 0
                        && layout.moe_intermediate_size == 0
                        && layout.shared_expert_intermediate_size == 0
                );
            }
            ManagedHead::Qwen36Moe => {
                assert!(layout.hidden_size == 2048 && layers == 40);
                assert!(layout.num_attention_heads == 16 && layout.num_key_value_heads == 2);
                assert!(layout.intermediate_size.is_none() && layout.num_experts == 256);
                assert!(
                    layout.num_experts_per_tok == 8
                        && layout.moe_intermediate_size == 512
                        && layout.shared_expert_intermediate_size == 512
                );
            }
        }
    }
    kani::cover!(result == Some(ManagedHead::Qwen38Dense));
    kani::cover!(result == Some(ManagedHead::Qwen36Moe));
    kani::cover!(result.is_none());
}

#[derive(Deserialize)]
struct Config {
    model_type: String,
    quantization: Quantization,
    text_config: Layout,
}
#[derive(Deserialize)]
struct Quantization {
    bits: i32,
    group_size: i32,
    mode: String,
}

#[cfg(any(feature = "mlx", test, kani))]
fn valid_draft_projection(
    input: i32,
    output: i32,
    hidden: i32,
    vocab: i32,
    bits: i32,
    group: i32,
    mode: &str,
) -> bool {
    input > 0
        && input % 64 == 0
        && output > 0
        && input == hidden
        && output <= vocab
        && bits == 4
        && group == 64
        && mode == "affine"
}

#[cfg(any(feature = "mlx", test, kani))]
fn valid_draft_rows(rows: i32, vocab: i32, ids: Option<&[u32]>) -> bool {
    if rows <= 0 || vocab <= 0 {
        return false;
    }
    match ids {
        None => rows == vocab,
        Some(ids) => {
            rows < vocab
                && ids.len() == rows as usize
                && ids.iter().all(|&id| id < vocab as u32)
                && ids.windows(2).all(|pair| pair[0] < pair[1])
        }
    }
}

pub fn inspect(path: &Path) -> Result<(Checkpoint, Layout)> {
    let path = path.canonicalize().context("opening MTP head folder")?;
    let config: Config = serde_json::from_reader(File::open(path.join("config.json"))?)?;
    ensure!(
        config.model_type == "qwen3_5_mtp"
            && config.quantization.bits == 4
            && config.quantization.group_size == 64
            && config.quantization.mode == "affine",
        "MTP requires an MLX affine 4-bit/group64 Qwen head"
    );
    let h = &config.text_config;
    ensure!(
        h.hidden_size > 0
            && h.hidden_size % 64 == 0
            && h.hidden_size <= i32::MAX / 2
            && h.num_attention_heads > 0
            && h.num_key_value_heads > 0
            && h.num_attention_heads % h.num_key_value_heads == 0
            && h.head_dim > 0
            && h.num_attention_heads
                .checked_mul(h.head_dim)
                .and_then(|x| x.checked_mul(2))
                .is_some()
            && h.num_attention_heads * h.head_dim % 64 == 0
            && (h.head_dim as f32 * h.partial_rotary_factor) as i32 > 0
            && (h.head_dim as f32 * h.partial_rotary_factor) as i32 % 2 == 0
            && h.vocab_size > 0
            && h.rms_norm_eps.is_finite()
            && h.rms_norm_eps > 0.
            && h.partial_rotary_factor.is_finite()
            && h.partial_rotary_factor > 0.
            && h.partial_rotary_factor <= 1.
            && h.rope_parameters.rope_theta.is_finite()
            && h.rope_parameters.rope_theta > 0.,
        "invalid MTP layout"
    );
    ensure!(
        if h.num_experts > 0 {
            h.num_experts_per_tok > 0
                && h.num_experts_per_tok <= h.num_experts as usize
                && h.moe_intermediate_size > 0
                && h.moe_intermediate_size % 64 == 0
                && h.shared_expert_intermediate_size == h.moe_intermediate_size
        } else {
            h.num_experts == 0
                && h.num_experts_per_tok == 0
                && h.intermediate_size.is_some_and(|n| n > 0 && n % 64 == 0)
        },
        "invalid MTP MLP layout"
    );
    let file = path.join("model.safetensors");
    let tensors = read_header(&file)?;
    ensure!(
        !tensors.is_empty() && tensors.keys().all(|name| !name.starts_with("mtp.")),
        "expected sanitized MLX MTP tensor names"
    );
    let mut expected = std::collections::BTreeMap::new();
    for (name, size) in [
        ("pre_fc_norm_embedding", h.hidden_size),
        ("pre_fc_norm_hidden", h.hidden_size),
        ("norm", h.hidden_size),
        ("layers.0.input_layernorm", h.hidden_size),
        ("layers.0.post_attention_layernorm", h.hidden_size),
        ("layers.0.self_attn.q_norm", h.head_dim),
        ("layers.0.self_attn.k_norm", h.head_dim),
    ] {
        expected.insert(format!("{name}.weight"), (false, vec![size as usize]));
    }
    let mut linear = |name: &str, input: i32, output: i32, experts: Option<i32>| {
        let mut dimensions = experts.into_iter().map(|n| n as usize).collect::<Vec<_>>();
        dimensions.push(output as usize);
        let mut packed = dimensions.clone();
        packed.push(input as usize / 8);
        dimensions.push(input as usize / 64);
        expected.insert(format!("{name}.weight"), (true, packed));
        expected.insert(format!("{name}.scales"), (false, dimensions.clone()));
        expected.insert(format!("{name}.biases"), (false, dimensions));
    };
    linear("fc", h.hidden_size * 2, h.hidden_size, None);
    linear(
        "layers.0.self_attn.q_proj",
        h.hidden_size,
        h.num_attention_heads * h.head_dim * 2,
        None,
    );
    for name in ["k_proj", "v_proj"] {
        linear(
            &format!("layers.0.self_attn.{name}"),
            h.hidden_size,
            h.num_key_value_heads * h.head_dim,
            None,
        );
    }
    linear(
        "layers.0.self_attn.o_proj",
        h.num_attention_heads * h.head_dim,
        h.hidden_size,
        None,
    );
    if h.num_experts == 0 {
        let intermediate = h
            .intermediate_size
            .context("missing MTP intermediate size")?;
        for name in ["gate_proj", "up_proj"] {
            linear(
                &format!("layers.0.mlp.{name}"),
                h.hidden_size,
                intermediate,
                None,
            );
        }
        linear("layers.0.mlp.down_proj", intermediate, h.hidden_size, None);
    } else {
        linear("layers.0.mlp.gate", h.hidden_size, h.num_experts, None);
        linear("layers.0.mlp.shared_expert_gate", h.hidden_size, 1, None);
        for (prefix, experts) in [("shared_expert", None), ("switch_mlp", Some(h.num_experts))] {
            for name in ["gate_proj", "up_proj"] {
                linear(
                    &format!("layers.0.mlp.{prefix}.{name}"),
                    h.hidden_size,
                    h.moe_intermediate_size,
                    experts,
                );
            }
            linear(
                &format!("layers.0.mlp.{prefix}.down_proj"),
                h.moe_intermediate_size,
                h.hidden_size,
                experts,
            );
        }
    }
    ensure!(
        tensors.len() == expected.len()
            && expected.iter().all(|(name, (packed, shape))| {
                tensors.get(name).is_some_and(|t| {
                    t.shape == *shape
                        && if *packed {
                            t.dtype == "U32"
                        } else {
                            matches!(t.dtype.as_str(), "F16" | "BF16" | "F32")
                        }
                })
            }),
        "MTP tensor coverage, shapes or dtypes do not match its configuration"
    );
    let checkpoint = Checkpoint {
        path,
        model_type: config.model_type,
        bits: Some(4.),
        size_bytes: file.metadata()?.len(),
        modules: Vec::new(),
        tensors,
    };
    Ok((checkpoint, config.text_config))
}

#[cfg(feature = "mlx")]
mod native {
    use super::*;
    use crate::{
        affine::AffineLinear,
        array::Array,
        lfm2::half_weight,
        qwen35::{Attention, Qwen35Moe},
        router,
    };
    use std::collections::VecDeque;

    pub struct Head {
        layout: Layout,
        embedding_norm: Array,
        hidden_norm: Array,
        fc: AffineLinear,
        norm: Array,
        input_norm: Array,
        post_norm: Array,
        attention: Attention,
        mlp: Mlp,
        draft_head: Option<AffineLinear>,
        draft_ids: Option<Array>,
    }

    pub struct Cache {
        keys: Array,
        values: Array,
    }

    enum Mlp {
        Dense(Box<Dense>),
        Moe(Box<Moe>),
    }
    struct Dense {
        gate: AffineLinear,
        up: AffineLinear,
        down: AffineLinear,
    }
    struct Moe {
        router: AffineLinear,
        gate: AffineLinear,
        up: AffineLinear,
        down: AffineLinear,
        shared_gate: AffineLinear,
        shared_up: AffineLinear,
        shared_down: AffineLinear,
        multiplier: AffineLinear,
    }

    impl Head {
        pub fn load(path: &Path, target: &Qwen35Moe) -> Result<Self> {
            let (checkpoint, layout) = inspect(path)?;
            ensure!(
                target.mtp_layout() == &layout,
                "MTP head does not match the target model dimensions"
            );
            let h = layout.hidden_size;
            // Optional, explicitly generated experiment. The target projection
            // remains untouched and validates every proposed token.
            let draft_metadata = path.join("draft_head_q4.json");
            let mut draft_ids = None;
            let draft_head = if draft_metadata.exists() {
                #[derive(Deserialize)]
                #[serde(deny_unknown_fields)]
                struct DraftMetadata {
                    input: i32,
                    output: i32,
                    bits: i32,
                    group_size: i32,
                    mode: String,
                    token_ids: Option<Vec<u32>>,
                }
                let metadata: DraftMetadata = serde_json::from_reader(File::open(draft_metadata)?)?;
                ensure!(
                    valid_draft_projection(
                        metadata.input,
                        metadata.output,
                        h,
                        layout.vocab_size,
                        metadata.bits,
                        metadata.group_size,
                        &metadata.mode
                    ) && valid_draft_rows(
                        metadata.output,
                        layout.vocab_size,
                        metadata.token_ids.as_deref()
                    ),
                    "draft Q4 projection does not match MTP target"
                );
                let file = path.join("draft_head_q4.safetensors");
                let tensors = read_header(&file)?;
                ensure!(
                    tensors.len() == 3
                        && [
                            ("lm_head.weight", "U32", vec![metadata.output as usize, h as usize / 8]),
                            ("lm_head.scales", "F16", vec![metadata.output as usize, h as usize / 64]),
                            ("lm_head.biases", "F16", vec![metadata.output as usize, h as usize / 64]),
                        ]
                            .iter()
                            .all(|(name, dtype, shape)| tensors.get(*name)
                                .is_some_and(|tensor| tensor.dtype == *dtype && tensor.shape == *shape)),
                    "invalid draft Q4 tensor coverage, shapes or dtypes"
                );
                if let Some(ids) = &metadata.token_ids {
                    draft_ids = Some(Array::from_u32(ids, &[metadata.output])?);
                }
                let checkpoint = Checkpoint {
                    path: path.to_owned(),
                    model_type: "draft_head_q4".into(),
                    bits: Some(4.),
                    size_bytes: file.metadata()?.len(),
                    modules: Vec::new(),
                    tensors,
                };
                Some(AffineLinear::load(
                    &checkpoint,
                    "lm_head",
                    h,
                    metadata.output,
                    None,
                )?)
            } else {
                None
            };
            let norm = |name: &str, n| half_weight(&checkpoint, name, Some(&[n]));
            let linear = |name: &str, input: i32, output: i32, experts: Option<i32>| {
                AffineLinear::load(&checkpoint, name, input, output, experts)
            };
            let prefix = "layers.0.mlp";
            let mlp = if layout.num_experts == 0 {
                let intermediate = layout
                    .intermediate_size
                    .context("missing dense MTP intermediate size")?;
                Mlp::Dense(Box::new(Dense {
                    gate: linear(&format!("{prefix}.gate_proj"), h, intermediate, None)?,
                    up: linear(&format!("{prefix}.up_proj"), h, intermediate, None)?,
                    down: linear(&format!("{prefix}.down_proj"), intermediate, h, None)?,
                }))
            } else {
                let intermediate = layout.moe_intermediate_size;
                let experts = Some(layout.num_experts);
                Mlp::Moe(Box::new(Moe {
                    router: linear(&format!("{prefix}.gate"), h, layout.num_experts, None)?,
                    gate: linear(
                        &format!("{prefix}.switch_mlp.gate_proj"),
                        h,
                        intermediate,
                        experts,
                    )?,
                    up: linear(
                        &format!("{prefix}.switch_mlp.up_proj"),
                        h,
                        intermediate,
                        experts,
                    )?,
                    down: linear(
                        &format!("{prefix}.switch_mlp.down_proj"),
                        intermediate,
                        h,
                        experts,
                    )?,
                    shared_gate: linear(
                        &format!("{prefix}.shared_expert.gate_proj"),
                        h,
                        intermediate,
                        None,
                    )?,
                    shared_up: linear(
                        &format!("{prefix}.shared_expert.up_proj"),
                        h,
                        intermediate,
                        None,
                    )?,
                    shared_down: linear(
                        &format!("{prefix}.shared_expert.down_proj"),
                        intermediate,
                        h,
                        None,
                    )?,
                    multiplier: linear(&format!("{prefix}.shared_expert_gate"), h, 1, None)?,
                }))
            };
            Ok(Self {
                embedding_norm: norm("pre_fc_norm_embedding.weight", h)?,
                hidden_norm: norm("pre_fc_norm_hidden.weight", h)?,
                fc: linear("fc", h * 2, h, None)?,
                norm: norm("norm.weight", h)?,
                input_norm: norm("layers.0.input_layernorm.weight", h)?,
                post_norm: norm("layers.0.post_attention_layernorm.weight", h)?,
                attention: Attention::load(
                    &checkpoint,
                    "layers.0.self_attn",
                    h,
                    layout.num_attention_heads,
                    layout.num_key_value_heads,
                    layout.head_dim,
                    (layout.head_dim as f32 * layout.partial_rotary_factor) as i32,
                    layout.rope_parameters.rope_theta,
                    layout.rms_norm_eps,
                )?,
                layout,
                mlp,
                draft_head,
                draft_ids,
            })
        }

        pub fn reset(&mut self) -> Result<()> {
            self.attention.set_state(None, None)
        }

        pub fn snapshot(&self) -> Result<Cache> {
            let (keys, values) = self.attention.states()?;
            Ok(Cache {
                keys: keys.try_clone()?,
                values: values.try_clone()?,
            })
        }

        pub fn restore(&mut self, cache: Cache) -> Result<()> {
            self.attention
                .set_state(Some(cache.keys), Some(cache.values))
        }

        pub fn cache_bytes(&self) -> Result<(Vec<u8>, Vec<u8>)> {
            let (k, v) = self.attention.states()?;
            Ok((k.to_bytes()?, v.to_bytes()?))
        }

        fn mixed(&self, target: &Qwen35Moe, hidden: &Array, next: &[u32]) -> Result<Array> {
            let time = i32::try_from(next.len())?;
            let h = self.layout.hidden_size;
            ensure!(
                time > 0 && hidden.shape() == [1, time, h],
                "invalid MTP hidden input"
            );
            self.mixed_embedding(hidden, &target.mtp_embeddings(next)?)
        }

        fn mixed_embedding(&self, hidden: &Array, embedding: &Array) -> Result<Array> {
            let e = embedding.rms_norm(&self.embedding_norm, self.layout.rms_norm_eps)?;
            let hidden = hidden.rms_norm(&self.hidden_norm, self.layout.rms_norm_eps)?;
            self.fc.forward(&Array::concatenate(&[&e, &hidden], 2)?)
        }

        /// Complete known target pairs without computing an unused prediction.
        pub fn extend_cache(
            &mut self,
            target: &Qwen35Moe,
            hidden: &Array,
            next: &[u32],
        ) -> Result<()> {
            self.append_cache(target, hidden, next)?;
            self.eval_cache()
        }

        fn append_cache(&mut self, target: &Qwen35Moe, hidden: &Array, next: &[u32]) -> Result<()> {
            let mixed = self.mixed(target, hidden, next)?;
            self.attention
                .append_kv(&mixed.rms_norm(&self.input_norm, self.layout.rms_norm_eps)?)
        }

        fn eval_cache(&self) -> Result<()> {
            let (keys, values) = self.attention.states()?;
            keys.eval()?;
            values.eval()
        }

        pub fn hidden(
            &mut self,
            target: &Qwen35Moe,
            hidden: &Array,
            next: &[u32],
        ) -> Result<Array> {
            self.draft_residual(target, hidden, next)?
                .rms_norm(&self.norm, self.layout.rms_norm_eps)
        }

        pub fn draft_residual(
            &mut self,
            target: &Qwen35Moe,
            hidden: &Array,
            next: &[u32],
        ) -> Result<Array> {
            let mixed = self.mixed(target, hidden, next)?;
            self.residual(&mixed)
        }

        /// Recurrent drafting feeds the pre-output-norm residual back, as in
        /// MTPLX's Qwen MTP chain. The output norm belongs only before lm_head.
        fn residual(&mut self, mixed: &Array) -> Result<Array> {
            let time = mixed.shape()[1];
            let h = self.layout.hidden_size;
            let attention = self
                .attention
                .forward(&mixed.rms_norm(&self.input_norm, self.layout.rms_norm_eps)?)?;
            static FUSED_NORM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            let fused = *FUSED_NORM
                .get_or_init(|| std::env::var("MLXL3_MTP_FUSED_NORM").as_deref() == Ok("1"));
            let (residual, normalized) = if fused {
                mixed.add_rms_norm(&attention, &self.post_norm, self.layout.rms_norm_eps)?
            } else {
                let residual = mixed.add(&attention)?;
                let normalized = residual.rms_norm(&self.post_norm, self.layout.rms_norm_eps)?;
                (residual, normalized)
            };
            let input = normalized.reshape(&[time, h])?;
            let mlp = match &self.mlp {
                Mlp::Dense(layer) => {
                    let Dense { gate, up, down } = &**layer;
                    down.forward(&gate.forward(&input)?.swiglu(&up.forward(&input)?)?)?
                }
                Mlp::Moe(layer) => {
                    let Moe {
                        router: gate_router,
                        gate,
                        up,
                        down,
                        shared_gate,
                        shared_up,
                        shared_down,
                        multiplier,
                    } = &**layer;
                    let probabilities = gate_router.forward(&input)?.softmax_precise()?;
                    let routes =
                        router::routes(&probabilities, self.layout.num_experts_per_tok, true)?;
                    let broadcast = input.reshape(&[time, 1, 1, h])?;
                    let gate = gate.gather_routed(&broadcast, &routes)?;
                    let up = up.gather_routed(&broadcast, &routes)?;
                    let routed = down
                        .gather_routed(&gate.swiglu(&up)?, &routes)?
                        .reshape(&[time, self.layout.num_experts_per_tok as i32, h])?
                        .mul(&routes.scores().reshape(&[
                            time,
                            self.layout.num_experts_per_tok as i32,
                            1,
                        ])?)?
                        .sum(1, false)?;
                    let shared = shared_down
                        .forward(
                            &shared_gate
                                .forward(&input)?
                                .swiglu(&shared_up.forward(&input)?)?,
                        )?
                        .mul(&multiplier.forward(&input)?.sigmoid()?)?;
                    routed.add(&shared)?
                }
            };
            residual.add(&mlp.reshape(&[1, time, h])?)
        }

        fn draft_chain(
            &mut self,
            target: &Qwen35Moe,
            hidden: &Array,
            anchor: u32,
            width: usize,
        ) -> Result<(Vec<u32>, Cache)> {
            ensure!((1..=3).contains(&width), "invalid MTP chain width");
            let mut token = Array::from_u32(&[anchor], &[1, 1])?;
            let mut hidden = hidden.try_clone()?;
            let mut proposed = Vec::with_capacity(width);
            let mut exact = None;
            for index in 0..width {
                let mixed = self.mixed_embedding(&hidden, &target.mtp_embedding_ids(&token)?)?;
                hidden = self.residual(&mixed)?;
                if index == 0 {
                    exact = Some(self.snapshot()?);
                }
                token = self
                    .draft_logits(
                        target,
                        &hidden.rms_norm(&self.norm, self.layout.rms_norm_eps)?,
                    )?
                    .log_probs()?
                    .argmax()?
                    .reshape(&[1, 1])?;
                if let Some(ids) = &self.draft_ids {
                    token = ids.take(&token, 0)?;
                }
                proposed.push(token.try_clone()?);
            }
            // One lazy device chain / one transfer, rather than D host argmaxes.
            let proposed = Array::concatenate(&proposed.iter().collect::<Vec<_>>(), 1)?.to_u32()?;
            Ok((proposed, exact.context("missing MTP first cache")?))
        }

        pub fn forward(
            &mut self,
            target: &Qwen35Moe,
            hidden: &Array,
            next: &[u32],
        ) -> Result<Array> {
            let hidden = self.hidden(target, hidden, next)?;
            // This standalone API retains full-vocabulary logits. Production
            // recursive drafts use compact scores and map their selected IDs.
            if self.draft_ids.is_some() {
                return target.mtp_logits(&hidden);
            }
            self.draft_logits(target, &hidden)
        }

        fn draft_logits(&self, target: &Qwen35Moe, hidden: &Array) -> Result<Array> {
            if let Some(head) = &self.draft_head {
                head.forward(hidden)
            } else {
                target.mtp_logits(hidden)
            }
        }
    }

    pub struct Session {
        depth: usize,
        pending: VecDeque<u32>,
        history: Vec<u32>,
        pub proposed: usize,
        pub accepted: usize,
        pub blocks: usize,
        pub lookup_blocks: usize,
    }

    impl Default for Session {
        fn default() -> Self {
            Self {
                depth: 1,
                pending: VecDeque::new(),
                history: Vec::new(),
                proposed: 0,
                accepted: 0,
                blocks: 0,
                lookup_blocks: 0,
            }
        }
    }

    impl Session {
        pub fn new(depth: usize) -> Result<Self> {
            ensure!((1..=3).contains(&depth), "MTP depth must be 1, 2 or 3");
            Ok(Self {
                depth,
                ..Default::default()
            })
        }

        /// Seed once after prefill. Only committed target inputs are remembered.
        pub fn enable_prompt_lookup(&mut self, prompt: &[u32]) {
            self.history = prompt[prompt
                .len()
                .saturating_sub(crate::speculative::LOOKUP_WINDOW)..]
                .to_vec();
        }

        fn remember(&mut self, tokens: &[u32]) {
            if !self.history.is_empty() {
                self.history.extend_from_slice(tokens);
                let excess = self
                    .history
                    .len()
                    .saturating_sub(crate::speculative::LOOKUP_WINDOW);
                self.history.drain(..excess);
            }
        }

        /// Every delivered token is selected by the target. Approximate draft
        /// KV is discarded; repair uses only accepted real trunk residuals.
        pub fn advance(
            &mut self,
            target: &mut Qwen35Moe,
            head: &mut Head,
            anchor: u32,
            context_limit: usize,
            output_remaining: usize,
            allow_lookup: impl Fn(&[u32]) -> bool,
        ) -> Result<u32> {
            ensure!(output_remaining > 0, "MTP output budget exhausted");
            if let Some(next) = self.pending.pop_front() {
                return Ok(next);
            }
            let remaining = context_limit
                .checked_sub(usize::try_from(target.offset())?)
                .context("MTP context exhausted")?;
            let width = crate::speculative::mtp_width(self.depth, remaining, output_remaining)
                .context("MTP depth or budget exhausted")?;
            let hidden = target.mtp_hidden()?.try_clone()?;
            if width == 0 {
                let logits = target.forward(anchor)?;
                self.remember(&[anchor]);
                return logits
                    .chat_greedy_ids()?
                    .last()
                    .copied()
                    .context("missing MTP fallback token");
            }
            let lookup =
                crate::speculative::prompt_lookup(&self.history, anchor, width, allow_lookup);
            let is_lookup = lookup.is_some();
            let (proposals, exact_first_cache) = if let Some(proposals) = lookup {
                let proposals = proposals.to_vec();
                head.append_cache(target, &hidden, &[anchor])?;
                (proposals, head.snapshot()?)
            } else {
                head.draft_chain(target, &hidden, anchor, width)?
            };
            let width = proposals.len();
            let verification = std::iter::once(anchor)
                .chain(proposals.iter().copied())
                .collect::<Vec<_>>();
            let (logits, raw) = target.verify_mtp(&verification)?;
            let tokens = logits.chat_greedy_ids()?;
            let accepted = crate::speculative::greedy_accept(&proposals, &tokens)
                .context("invalid MTP verification output")?;
            let retained = 1 + accepted.accepted_draft_tokens;
            target.commit_dflash_verification(retained, width + 1)?;
            target.set_mtp_hidden(raw.slice(1, retained as i32 - 1, retained as i32)?)?;
            head.restore(exact_first_cache)?;
            if accepted.accepted_draft_tokens > 0 {
                // Keep single-row arithmetic (batched RoPE can round differently)
                // but evaluate only the final K/V graph. No output attention/MoE
                // or vocabulary projection is needed for known history.
                for (index, &token) in proposals[..accepted.accepted_draft_tokens]
                    .iter()
                    .enumerate()
                {
                    head.append_cache(
                        target,
                        &raw.slice(1, index as i32, index as i32 + 1)?,
                        &[token],
                    )?;
                }
            }
            if accepted.accepted_draft_tokens > 0 || is_lookup {
                head.eval_cache()?;
            }
            if is_lookup && accepted.accepted_draft_tokens == 0 {
                self.history.clear();
            } else {
                self.remember(&verification[..retained]);
            }
            self.pending
                .extend(proposals[..accepted.accepted_draft_tokens].iter().copied());
            self.pending.push_back(accepted.target_token);
            self.proposed += width;
            self.accepted += accepted.accepted_draft_tokens;
            self.blocks += 1;
            self.lookup_blocks += usize::from(is_lookup);
            self.pending.pop_front().context("missing MTP target token")
        }
    }

    #[cfg(test)]
    mod cache_tests {
        use super::*;

        #[test]
        fn lookup_history_is_bounded_and_disabled_until_seeded() -> Result<()> {
            let mut session = Session::new(3)?;
            session.remember(&[1, 2, 3]);
            assert!(session.history.is_empty());
            let prompt = (0..2048).collect::<Vec<_>>();
            session.enable_prompt_lookup(&prompt);
            assert_eq!(session.history, prompt[1024..]);
            session.remember(&[2048, 2049, 2050]);
            assert_eq!(session.history.len(), 1024);
            assert_eq!(session.history[0], 1027);
            assert_eq!(&session.history[1021..], &[2048, 2049, 2050]);
            session.history.clear();
            session.remember(&[77]);
            assert!(session.history.is_empty());
            Ok(())
        }

        fn bytes(cache: &Cache) -> Result<(Vec<u8>, Vec<u8>)> {
            Ok((cache.keys.to_bytes()?, cache.values.to_bytes()?))
        }

        #[test]
        #[ignore = "requires local Qwen/MTP checkpoints and physical Apple GPU"]
        fn cache_only_matches_full_head_and_next_prediction() -> Result<()> {
            let target = Qwen35Moe::load(Path::new(
                &std::env::var("MLXL3_MTP_TEST_MODEL")
                    .unwrap_or_else(|_| "models/Qwen3.6-35B-A3B-EXL3-2.49bpw".into()),
            ))?;
            let mut head = Head::load(
                Path::new(
                    &std::env::var("MLXL3_MTP_TEST_HEAD")
                        .unwrap_or_else(|_| "models/Qwen3.6-35B-A3B-MTP-4bit".into()),
                ),
                &target,
            )?;
            let h = target.mtp_layout().hidden_size;
            let fixture = |time, seed| {
                let bits = (0..time * h)
                    .map(|i| half::f16::from_f32(((i + seed) % 251 - 125) as f32 / 128.).to_bits())
                    .collect::<Vec<_>>();
                Array::from_f16_bits(&bits, &[1, time, h])
            };
            let probe = fixture(1, 13)?;
            for (index, time) in [1, 2, 3, 17, 23, 24, 255].into_iter().enumerate() {
                let initial = if index == 0 {
                    None
                } else {
                    Some(head.snapshot()?)
                };
                let hidden = fixture(time, time)?;
                let tokens = (0..time).map(|i| 1 + i as u32 % 31).collect::<Vec<_>>();
                // Independent expected cache comes from the unchanged full
                // attention/MoE head; it also predicts the following position.
                head.hidden(&target, &hidden, &tokens)?.eval()?;
                let complete = head.snapshot()?;
                let expected = bytes(&complete)?;
                let expected_prediction = head.hidden(&target, &probe, &[37])?.to_bytes()?;
                match initial {
                    Some(cache) => head.restore(cache)?,
                    None => head.reset()?,
                }
                head.extend_cache(&target, &hidden, &tokens)?;
                assert_eq!(bytes(&head.snapshot()?)?, expected, "K/V rows={time}");
                assert!(head.extend_cache(&target, &probe, &[]).is_err());
                assert!(
                    head.extend_cache(&target, &probe, &[target.mtp_layout().vocab_size as u32])
                        .is_err()
                );
                assert!(head.extend_cache(&target, &probe, &[1, 2]).is_err());
                let bad_width = Array::from_f16_bits(&[0; 64], &[1, 1, 64])?;
                assert!(head.extend_cache(&target, &bad_width, &[1]).is_err());
                assert_eq!(
                    bytes(&head.snapshot()?)?,
                    expected,
                    "error rollback rows={time}"
                );
                assert_eq!(
                    head.hidden(&target, &probe, &[37])?.to_bytes()?,
                    expected_prediction,
                    "next prediction rows={time}"
                );
                head.restore(complete)?;
            }
            head.reset()?;
            assert!(head.extend_cache(&target, &probe, &[]).is_err());
            assert!(head.snapshot().is_err());
            Ok(())
        }
    }
}

#[cfg(feature = "mlx")]
pub use native::{Cache, Head, Session};

#[cfg(test)]
mod draft_projection_tests {
    use super::*;

    #[test]
    fn optional_projection_must_match_target_and_affine_contract() {
        assert!(valid_draft_projection(
            2048, 248320, 2048, 248320, 4, 64, "affine"
        ));
        for (input, output, hidden, vocab, bits, group, mode) in [
            (0, 1, 0, 1, 4, 64, "affine"),
            (63, 1, 63, 1, 4, 64, "affine"),
            (64, 0, 64, 0, 4, 64, "affine"),
            (64, 1, 128, 1, 4, 64, "affine"),
            (64, 1, 64, 2, 4, 64, "affine"),
            (64, 1, 64, 1, 8, 64, "affine"),
            (64, 1, 64, 1, 4, 32, "affine"),
            (64, 1, 64, 1, 4, 64, "symmetric"),
        ] {
            assert!(
                !(valid_draft_projection(input, output, hidden, vocab, bits, group, mode)
                    && valid_draft_rows(output, vocab, None))
            );
        }
    }

    #[test]
    fn compact_rows_are_nonempty_ordered_unique_and_in_target_vocab() {
        assert!(valid_draft_rows(128, 128, None));
        assert!(valid_draft_rows(3, 128, Some(&[0, 63, 127])));
        for (rows, vocab, ids) in [
            (0, 128, Some(&[][..])),
            (1, 128, None),
            (2, 128, Some(&[0][..])),
            (2, 128, Some(&[1, 1][..])),
            (2, 128, Some(&[2, 1][..])),
            (2, 128, Some(&[0, 128][..])),
            (1, 1, Some(&[0][..])),
            (1, -1, Some(&[0][..])),
        ] {
            assert!(!valid_draft_rows(rows, vocab, ids));
        }
    }
}

#[cfg(kani)]
mod draft_projection_verification {
    use super::*;

    #[kani::proof]
    // The two mode representatives contain at most nine bytes. Keep the
    // library string-comparison unwinding assertion enabled at that bound.
    #[kani::unwind(10)]
    fn optional_projection_guards_dimensions_and_quantization() {
        let (input, output, hidden, vocab, bits, group): (i32, i32, i32, i32, i32, i32) =
            kani::any();
        let mode = if kani::any() { "affine" } else { "symmetric" };
        let valid = valid_draft_projection(input, output, hidden, vocab, bits, group, mode);
        assert!(
            valid
                == (input > 0
                    && input % 64 == 0
                    && output > 0
                    && input == hidden
                    && output <= vocab
                    && bits == 4
                    && group == 64
                    && mode == "affine")
        );
        if valid {
            assert!(input / 64 > 0 && input / 8 > 0);
            assert!(input / 8 == (input / 64) * 8);
            assert!(output <= vocab);
        }
        kani::cover!(valid);
        kani::cover!(!valid && input > 0 && hidden == input && vocab != output);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn compact_draft_ids_stay_sorted_and_inside_target_vocabulary() {
        let ids: [u32; 8] = kani::any();
        let len: usize = kani::any();
        // Bounded map-check domain: 0..8 entries, full scalar/ID domains.
        if len > ids.len() {
            return;
        }
        let rows: i32 = kani::any();
        let vocab: i32 = kani::any();
        let map = if kani::any() { Some(&ids[..len]) } else { None };
        let valid = valid_draft_rows(rows, vocab, map);
        if valid {
            assert!(rows > 0 && rows <= vocab);
            if map.is_some() {
                assert!(rows < vocab && rows as usize == len);
                for index in 0..len {
                    assert!(ids[index] < vocab as u32);
                    if index > 0 {
                        assert!(ids[index - 1] < ids[index]);
                    }
                }
            } else {
                assert_eq!(rows, vocab);
            }
        }
        kani::cover!(valid && map.is_some() && len == 8);
        kani::cover!(valid && map.is_none());
        kani::cover!(!valid && map.is_some() && len == 2 && ids[0] == ids[1]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn managed_head_uses_target_geometry_and_rejects_unknown_variants() -> Result<()> {
        let dense = serde_json::json!({"hidden_size":5120,"num_attention_heads":24,
            "num_key_value_heads":4,"head_dim":256,"vocab_size":248320,"intermediate_size":17408,
            "rms_norm_eps":1e-6,"partial_rotary_factor":0.25,"rope_parameters":{"rope_theta":10000000}});
        let moe = serde_json::json!({"hidden_size":2048,"num_attention_heads":16,
            "num_key_value_heads":2,"head_dim":256,"vocab_size":248320,"num_experts":256,
            "num_experts_per_tok":8,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,
            "rms_norm_eps":1e-6,"partial_rotary_factor":0.25,"rope_parameters":{"rope_theta":10000000}});
        for (text, layers, kind) in [
            (dense, 64, ManagedHead::Qwen38Dense),
            (moe, 40, ManagedHead::Qwen36Moe),
        ] {
            let layout: Layout = serde_json::from_value(text.clone())?;
            assert_eq!(layout.managed_head(layers), Some(kind));
            for count in [0, 1, layers - 1, layers + 1, usize::MAX] {
                assert_eq!(layout.managed_head(count), None);
            }
            for field in [
                "hidden_size",
                "num_attention_heads",
                "num_key_value_heads",
                "head_dim",
                "vocab_size",
            ] {
                for value in [i32::MIN, -1, 0, 1, i32::MAX] {
                    let mut changed = text.clone();
                    changed[field] = serde_json::json!(value);
                    assert_eq!(
                        serde_json::from_value::<Layout>(changed)?.managed_head(layers),
                        None
                    );
                }
            }
            let directory = tempfile::tempdir()?;
            let mut config = serde_json::json!({"model_type":if layers == 64 {"qwen3_5"} else {"qwen3_5_moe"},"text_config":text});
            config["text_config"]["num_hidden_layers"] = serde_json::json!(layers);
            std::fs::write(
                directory.path().join("config.json"),
                serde_json::to_vec(&config)?,
            )?;
            assert_eq!(target_layout(directory.path())?, (layout, layers));
            for (pointer, value) in [
                ("/model_type", serde_json::json!("qwen3")),
                ("/text_config/num_hidden_layers", serde_json::json!(-1)),
                (
                    "/text_config/mtp_use_dedicated_embeddings",
                    serde_json::json!(true),
                ),
            ] {
                let mut changed = config.clone();
                if pointer.ends_with("mtp_use_dedicated_embeddings") {
                    changed["text_config"]["mtp_use_dedicated_embeddings"] = value;
                } else {
                    *changed.pointer_mut(pointer).unwrap() = value;
                }
                std::fs::write(
                    directory.path().join("config.json"),
                    serde_json::to_vec(&changed)?,
                )?;
                assert!(target_layout(directory.path()).is_err());
            }
        }
        Ok(())
    }
    use serde_json::{Value, json};
    use std::{collections::BTreeMap, io::Write};

    fn fixture(path: &Path, moe: bool) -> Result<Value> {
        let config = json!({"model_type":"qwen3_5_mtp", "quantization":{"bits":4,"group_size":64,"mode":"affine"},
            "text_config":{"hidden_size":64,"num_attention_heads":1,"num_key_value_heads":1,"head_dim":64,
                "vocab_size":128,"num_experts":if moe {8} else {0},"num_experts_per_tok":if moe {2} else {0},
                "moe_intermediate_size":64,"shared_expert_intermediate_size":64,"intermediate_size":64,
                "rms_norm_eps":0.000001,"partial_rotary_factor":0.5,"rope_parameters":{"rope_theta":10000000.0}}});
        std::fs::write(path.join("config.json"), serde_json::to_vec(&config)?)?;
        let mut entries = BTreeMap::new();
        for name in [
            "pre_fc_norm_embedding",
            "pre_fc_norm_hidden",
            "norm",
            "layers.0.input_layernorm",
            "layers.0.post_attention_layernorm",
            "layers.0.self_attn.q_norm",
            "layers.0.self_attn.k_norm",
        ] {
            entries.insert(format!("{name}.weight"), ("F16", vec![64]));
        }
        let mut add = |name: &str, input: usize, output: usize, experts: bool| {
            let mut shape = if experts { vec![8] } else { vec![] };
            shape.push(output);
            let mut packed = shape.clone();
            packed.push(input / 8);
            shape.push(input / 64);
            entries.insert(format!("{name}.weight"), ("U32", packed));
            entries.insert(format!("{name}.scales"), ("F16", shape.clone()));
            entries.insert(format!("{name}.biases"), ("F16", shape));
        };
        add("fc", 128, 64, false);
        add("layers.0.self_attn.q_proj", 64, 128, false);
        add("layers.0.self_attn.k_proj", 64, 64, false);
        add("layers.0.self_attn.v_proj", 64, 64, false);
        add("layers.0.self_attn.o_proj", 64, 64, false);
        if moe {
            add("layers.0.mlp.gate", 64, 8, false);
            add("layers.0.mlp.shared_expert_gate", 64, 1, false);
            for kind in ["shared_expert", "switch_mlp"] {
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    add(
                        &format!("layers.0.mlp.{kind}.{projection}"),
                        64,
                        64,
                        kind == "switch_mlp",
                    );
                }
            }
        } else {
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                add(&format!("layers.0.mlp.{projection}"), 64, 64, false);
            }
        }
        assert_eq!(entries.len(), if moe { 46 } else { 31 });
        let mut offset = 0;
        let entries = entries
            .into_iter()
            .map(|(name, (dtype, shape))| {
                let next =
                    offset + shape.iter().product::<usize>() * if dtype == "U32" { 4 } else { 2 };
                let value = json!({"dtype":dtype,"shape":shape,"data_offsets":[offset,next]});
                offset = next;
                (name, value)
            })
            .collect::<BTreeMap<_, _>>();
        let header = serde_json::to_vec(&entries)?;
        let mut file = File::create(path.join("model.safetensors"))?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        file.write_all(&vec![0; offset])?;
        Ok(config)
    }

    #[test]
    fn canonical_dense_and_moe_checkpoints_are_inspected_without_gpu() -> Result<()> {
        for moe in [false, true] {
            let directory = tempfile::tempdir()?;
            fixture(directory.path(), moe)?;
            let (checkpoint, layout) = inspect(directory.path())?;
            assert_eq!(checkpoint.tensors.len(), if moe { 46 } else { 31 });
            assert_eq!(layout.hidden_size, 64);
        }
        Ok(())
    }

    #[test]
    fn incompatible_or_incomplete_heads_fail_before_gpu_loading() -> Result<()> {
        let directory = tempfile::tempdir()?;
        for (location, value) in [
            ("/model_type", json!("qwen3_5_moe")),
            ("/quantization/bits", json!(8)),
            ("/quantization/group_size", json!(32)),
            ("/quantization/mode", json!("mxfp4")),
            ("/text_config/hidden_size", json!(0)),
            ("/text_config/hidden_size", json!(2147483647)),
            ("/text_config/num_attention_heads", json!(0)),
            ("/text_config/head_dim", json!(2147483647)),
            ("/text_config/head_dim", json!(63)),
            ("/text_config/partial_rotary_factor", json!(0.001)),
            ("/text_config/rms_norm_eps", json!(-1)),
            ("/text_config/rope_parameters/rope_theta", json!(0)),
            ("/text_config/num_experts_per_tok", json!(9)),
            ("/text_config/shared_expert_intermediate_size", json!(128)),
        ] {
            let mut config = fixture(directory.path(), true)?;
            *config.pointer_mut(location).unwrap() = value;
            std::fs::write(
                directory.path().join("config.json"),
                serde_json::to_vec(&config)?,
            )?;
            assert!(
                inspect(directory.path()).is_err(),
                "bad head accepted at {location}"
            );
        }
        fixture(directory.path(), false)?;
        let file = directory.path().join("model.safetensors");
        std::fs::OpenOptions::new()
            .write(true)
            .open(&file)?
            .set_len(9)?;
        assert!(
            inspect(directory.path()).is_err(),
            "truncated payload accepted"
        );
        std::fs::remove_file(file)?;
        assert!(inspect(directory.path()).is_err(), "missing head accepted");
        Ok(())
    }
}
