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
        array::{Array, Dtype},
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
        wiring: Wiring,
        /// Q36-MTP-EVICT-T3B: truncate the verify width by draft confidence.
        evict: bool,
    }

    /// Which MTP inputs are normalized (Q36-MTP-POSTNORM-T1/T1B). Production
    /// selects `post` (both true, the default) or `pre` (both false, `MLXL3_MTP_POST_NORM=0`);
    /// the single-sided variants exist only for the T1 diagnostic.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub(crate) struct Wiring {
        /// Apply the trunk's final norm to the target hidden before `pre_fc_norm_hidden`.
        seed: bool,
        /// Feed the head's own `rms_norm(residual, norm)` back to the next level.
        chain: bool,
    }

    impl Wiring {
        const PRE: Self = Self {
            seed: false,
            chain: false,
        };
        const POST: Self = Self {
            seed: true,
            chain: true,
        };

        fn from_env() -> Result<Self> {
            const MESSAGE: &str = "MLXL3_MTP_POST_NORM must be 0 or 1";
            let value = match std::env::var("MLXL3_MTP_POST_NORM") {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(error) => return Err(error).context(MESSAGE),
            };
            let post =
                crate::contracts::mtp_post_norm_setting(value.as_deref()).context(MESSAGE)?;
            Ok(if post { Self::POST } else { Self::PRE })
        }
    }

    /// Strict `MLXL3_MTP_EVICT` (Q36-MTP-EVICT-T3B), read once with the head.
    fn evict_from_env() -> Result<bool> {
        const MESSAGE: &str = "MLXL3_MTP_EVICT must be 0 or 1";
        let value = match std::env::var("MLXL3_MTP_EVICT") {
            Ok(value) => Some(value),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(error).context(MESSAGE),
        };
        crate::contracts::mtp_evict_setting(value.as_deref()).context(MESSAGE)
    }

    pub struct Cache {
        keys: Array,
        values: Array,
    }

    impl Cache {
        pub fn compact_for_cache(mut self) -> Result<Self> {
            self.keys = self.keys.compact_for_cache()?;
            self.values = self.values.compact_for_cache()?;
            Ok(self)
        }
        pub fn retained_bytes(&self) -> Result<usize> {
            self.keys
                .retained_bytes()?
                .checked_add(self.values.retained_bytes()?)
                .context("MTP cache size overflow")
        }
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
            Self::load_cancellable(path, target, &std::sync::atomic::AtomicBool::new(false))
        }

        pub fn load_cancellable(
            path: &Path,
            target: &Qwen35Moe,
            cancelled: &std::sync::atomic::AtomicBool,
        ) -> Result<Self> {
            ensure!(
                !cancelled.load(std::sync::atomic::Ordering::Relaxed),
                "generation cancelled"
            );
            let wiring = Wiring::from_env()?;
            let evict = evict_from_env()?;
            let (checkpoint, layout) = inspect(path)?;
            ensure!(
                target.mtp_layout() == &layout,
                "MTP head does not match the target model dimensions"
            );
            let h = layout.hidden_size;
            let projection_path = if path.join("draft_head_q4.json").exists() {
                path.to_owned()
            } else {
                match target.mtp_projection_path(cancelled) {
                    Ok(Some(cache)) => cache,
                    Ok(None) => path.to_owned(),
                    Err(error) => {
                        if cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                            return Err(error);
                        }
                        eprintln!(
                            "Compact MTP projection unavailable; using target head: {error:#}"
                        );
                        path.to_owned()
                    }
                }
            };
            let draft_metadata = projection_path.join("draft_head_q4.json");
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
                let file = projection_path.join("draft_head_q4.safetensors");
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
                    path: projection_path,
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
            let loaded = Self {
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
                wiring,
                evict,
            };
            ensure!(
                !cancelled.load(std::sync::atomic::Ordering::Relaxed),
                "generation cancelled"
            );
            Ok(loaded)
        }

        pub fn reset(&mut self) -> Result<()> {
            self.attention.set_state(None, None)
        }

        #[cfg(test)]
        pub(crate) fn evict(&self) -> bool {
            self.evict
        }

        #[cfg(test)]
        pub(crate) fn compact_vocabulary(&self) -> bool {
            self.draft_ids.is_some()
        }

        /// `pre`, `post`, or a T1 test-only single-sided wiring.
        #[cfg(test)]
        pub(crate) fn wiring_name(&self) -> &'static str {
            match (self.wiring.seed, self.wiring.chain) {
                (false, false) => "pre",
                (true, false) => "seed",
                (false, true) => "chain",
                (true, true) => "post",
            }
        }

        #[cfg(test)]
        pub(crate) fn set_test_wiring(&mut self, seed: bool, chain: bool) {
            self.wiring = Wiring { seed, chain };
        }

        /// Drop the compact projection so drafts use the target's full lm_head.
        #[cfg(test)]
        pub(crate) fn use_target_draft_head(&mut self) {
            self.draft_head = None;
            self.draft_ids = None;
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
            let embedding = target.mtp_embeddings(next)?;
            if self.wiring.seed {
                return self.mixed_embedding(&target.mtp_final_norm(hidden)?, &embedding);
            }
            self.mixed_embedding(hidden, &embedding)
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

        /// The attention/MLP residual. `draft_chain` feeds it back as is (pre-norm,
        /// as in MTPLX's Qwen MTP chain) or after the output norm with `post` wiring.
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
        ) -> Result<(Vec<u32>, Cache, Option<Vec<f32>>)> {
            ensure!((1..=3).contains(&width), "invalid MTP chain width");
            let mut token = Array::from_u32(&[anchor], &[1, 1])?;
            let mut hidden = hidden.try_clone()?;
            let mut proposed = Vec::with_capacity(width);
            // EVICT: each level's top-1 log-probability, as f32 bits riding the ID transfer.
            let mut tops = Vec::new();
            let mut exact = None;
            #[cfg(test)]
            let mut captured = Vec::new();
            if self.wiring.seed {
                hidden = target.mtp_final_norm(&hidden)?;
            }
            for index in 0..width {
                let mixed = self.mixed_embedding(&hidden, &target.mtp_embedding_ids(&token)?)?;
                let residual = self.residual(&mixed)?;
                if index == 0 {
                    exact = Some(self.snapshot()?);
                }
                let normalized = residual.rms_norm(&self.norm, self.layout.rms_norm_eps)?;
                let log_probs = self.draft_logits(target, &normalized)?.log_probs()?;
                // Post-norm recursion reuses the lm_head input; pre-norm keeps the residual.
                hidden = if self.wiring.chain {
                    normalized
                } else {
                    residual
                };
                #[cfg(test)]
                if super::cycle_trace::confidence() {
                    captured.push(log_probs.try_clone()?);
                }
                token = log_probs.argmax()?.reshape(&[1, 1])?;
                if self.evict {
                    let vocab = *log_probs.shape().last().context("empty MTP log-probs")?;
                    tops.push(
                        log_probs
                            .reshape(&[vocab])?
                            .take(&token.reshape(&[1])?, 0)?
                            .astype(Dtype::Float32)?
                            .view(Dtype::UInt32)?
                            .reshape(&[1, 1])?,
                    );
                }
                if let Some(ids) = &self.draft_ids {
                    token = ids.take(&token, 0)?;
                }
                proposed.push(token.try_clone()?);
            }
            #[cfg(test)]
            super::cycle_trace::mark("draft_graph");
            // One lazy device chain / one transfer, rather than D host argmaxes.
            let mut proposed =
                Array::concatenate(&proposed.iter().chain(&tops).collect::<Vec<_>>(), 1)?
                    .to_u32()?;
            let tops = if self.evict {
                ensure!(proposed.len() == 2 * width, "invalid MTP EVICT transfer");
                Some(
                    proposed
                        .split_off(width)
                        .into_iter()
                        .map(f32::from_bits)
                        .collect(),
                )
            } else {
                None
            };
            #[cfg(test)]
            super::cycle_trace::mark("draft_ready");
            #[cfg(test)]
            if !captured.is_empty() {
                // Diagnostic readback after the production transfer; never timed.
                let ids = self.draft_ids.as_ref().map(Array::to_u32).transpose()?;
                for (&proposal, log_probs) in proposed.iter().zip(&captured) {
                    super::cycle_trace::level(proposal, &log_probs.to_f32()?, ids.as_deref());
                }
                super::cycle_trace::mark("confidence_ready");
            }
            Ok((proposed, exact.context("missing MTP first cache")?, tops))
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
        adaptive: Option<crate::memory_policy::AdaptiveMtp>,
        pending: VecDeque<u32>,
        history: Vec<u32>,
        pub proposed: usize,
        pub accepted: usize,
        pub blocks: usize,
        pub lookup_blocks: usize,
        pub depth_blocks: [usize; 4],
        /// Head-drafted cycles and width-0 target steps, by width and accepted prefix.
        pub accept_histogram: crate::speculative::AcceptHistogram,
        /// Prompt-lookup cycles, by width and accepted prefix.
        pub lookup_accept_histogram: crate::speculative::AcceptHistogram,
    }

    impl Default for Session {
        fn default() -> Self {
            Self {
                depth: 1,
                adaptive: None,
                pending: VecDeque::new(),
                history: Vec::new(),
                proposed: 0,
                accepted: 0,
                blocks: 0,
                lookup_blocks: 0,
                depth_blocks: [0; 4],
                accept_histogram: Default::default(),
                lookup_accept_histogram: Default::default(),
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

        pub fn enable_adaptive(&mut self) {
            self.adaptive = Some(crate::memory_policy::AdaptiveMtp::default());
        }

        #[cfg(test)]
        pub(crate) fn force_depth_for_check(&mut self, depth: usize) {
            assert!(depth <= 3);
            self.depth = depth;
            self.enable_adaptive();
        }

        fn observe_block(&mut self, started: std::time::Instant, committed: usize) {
            self.depth_blocks[self.depth] = self.depth_blocks[self.depth].saturating_add(1);
            if let Some(policy) = &mut self.adaptive {
                policy.observe(
                    self.depth,
                    started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                    committed,
                );
            }
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
            self.advance_sampled(
                target,
                head,
                anchor,
                context_limit,
                output_remaining,
                allow_lookup,
                None,
            )
        }

        /// [`Self::advance`] with an optional non-greedy target policy. The
        /// head still drafts deterministically; verification then uses exact
        /// speculative rejection sampling (`crate::sampling`). A greedy
        /// policy keeps the greedy verification unchanged.
        #[allow(clippy::too_many_arguments)]
        pub fn advance_sampled(
            &mut self,
            target: &mut Qwen35Moe,
            head: &mut Head,
            anchor: u32,
            context_limit: usize,
            output_remaining: usize,
            allow_lookup: impl Fn(&[u32]) -> bool,
            draw: Option<crate::sampling::Draw<'_>>,
        ) -> Result<u32> {
            let mut draw = draw.filter(|draw| !draw.sampler.is_greedy());
            ensure!(output_remaining > 0, "MTP output budget exhausted");
            if let Some(next) = self.pending.pop_front() {
                return Ok(next);
            }
            if let Some(policy) = &mut self.adaptive {
                self.depth = policy.next(self.depth);
            }
            let started = std::time::Instant::now();
            #[cfg(test)]
            super::cycle_trace::mark("cycle_start");
            let remaining = context_limit
                .checked_sub(usize::try_from(target.offset())?)
                .context("MTP context exhausted")?;
            let width = if self.depth == 0 && self.adaptive.is_some() {
                crate::speculative::bounded_proposals(remaining, output_remaining).map(|_| 0)
            } else {
                crate::speculative::mtp_width(self.depth, remaining, output_remaining)
            }
            .context("MTP depth or budget exhausted")?;
            let hidden = target.mtp_hidden()?.try_clone()?;
            if width == 0 {
                // Pair the previous trunk hidden with this committed input.
                // Required before resuming speculative decoding after D0.
                if self.adaptive.is_some() {
                    head.append_cache(target, &hidden, &[anchor])?;
                }
                let logits = target.forward(anchor)?;
                self.remember(&[anchor]);
                let next = if let Some(draw) = draw.as_mut() {
                    let rows = crate::sampling::log_prob_rows(&logits)?;
                    draw.sample(rows.last().context("missing MTP fallback row")?.to_vec())?
                } else {
                    logits
                        .chat_greedy_ids()?
                        .last()
                        .copied()
                        .context("missing MTP fallback token")?
                };
                if self.adaptive.is_some() {
                    head.eval_cache()?;
                }
                ensure!(
                    self.accept_histogram.record(0, 0),
                    "invalid MTP target step"
                );
                self.observe_block(started, 1);
                #[cfg(test)]
                super::cycle_trace::mark("target_step_end");
                return Ok(next);
            }
            let lookup =
                crate::speculative::prompt_lookup(&self.history, anchor, width, allow_lookup);
            let is_lookup = lookup.is_some();
            let (mut proposals, exact_first_cache, tops) = if let Some(proposals) = lookup {
                let proposals = proposals.to_vec();
                head.append_cache(target, &hidden, &[anchor])?;
                (proposals, head.snapshot()?, None)
            } else {
                head.draft_chain(target, &hidden, anchor, width)?
            };
            if let Some(tops) = tops {
                // Exact greedy verify accepts any prefix; EVICT only chooses its length.
                let bands = tops
                    .iter()
                    .map(|&top| crate::contracts::evict_band(top))
                    .collect::<Option<Vec<_>>>()
                    .context("non-finite MTP draft confidence")?;
                let kept = crate::contracts::evict_width(&bands, proposals.len())
                    .context("invalid MTP EVICT width")?;
                #[cfg(test)]
                super::cycle_trace::evict(proposals.len(), &tops, kept);
                proposals.truncate(kept);
            }
            let width = proposals.len();
            let verification = std::iter::once(anchor)
                .chain(proposals.iter().copied())
                .collect::<Vec<_>>();
            let (logits, raw) = target.verify_mtp(&verification)?;
            #[cfg(test)]
            super::cycle_trace::mark("verify_ready");
            let accepted = if let Some(draw) = draw.as_mut() {
                let rows = crate::sampling::log_prob_rows(&logits)?;
                ensure!(rows.len() == width + 1, "invalid MTP verification output");
                draw.speculative_accept(&proposals, |index| Ok(rows[index].to_vec()))?
            } else {
                let tokens = logits.chat_greedy_ids()?;
                #[cfg(test)]
                super::cycle_trace::mark("greedy_ready");
                crate::speculative::greedy_accept(&proposals, &tokens)
                    .context("invalid MTP verification output")?
            };
            // Validate before any commit (an impossible cycle fails closed);
            // publish with the other counters once the cycle is complete.
            let mut histogram = if is_lookup {
                self.lookup_accept_histogram
            } else {
                self.accept_histogram
            };
            ensure!(
                histogram.record(width, accepted.accepted_draft_tokens),
                "invalid MTP cycle"
            );
            let retained = 1 + accepted.accepted_draft_tokens;
            target.commit_dflash_verification(retained, width + 1)?;
            target.set_mtp_hidden(raw.slice(1, retained as i32 - 1, retained as i32)?)?;
            head.restore(exact_first_cache)?;
            #[cfg(test)]
            super::cycle_trace::mark("commit_graph");
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
            #[cfg(test)]
            super::cycle_trace::mark("repair_graph");
            if accepted.accepted_draft_tokens > 0 || is_lookup {
                head.eval_cache()?;
            }
            #[cfg(test)]
            super::cycle_trace::mark("repair_ready");
            if is_lookup && accepted.accepted_draft_tokens == 0 {
                self.history.clear();
            } else {
                self.remember(&verification[..retained]);
            }
            self.pending
                .extend(proposals[..accepted.accepted_draft_tokens].iter().copied());
            self.pending.push_back(accepted.target_token);
            if is_lookup {
                self.lookup_accept_histogram = histogram;
            } else {
                self.accept_histogram = histogram;
            }
            self.proposed += width;
            self.accepted += accepted.accepted_draft_tokens;
            self.blocks += 1;
            self.lookup_blocks += usize::from(is_lookup);
            self.observe_block(started, retained);
            #[cfg(test)]
            super::cycle_trace::mark("cycle_end");
            self.pending.pop_front().context("missing MTP target token")
        }
    }

    #[cfg(test)]
    mod cache_tests {
        use super::*;

        #[test]
        #[ignore = "physical Apple GPU, local stock/compact Qwen MTP; run alone"]
        fn default_projection_loads_reuses_cache_and_respects_explicit_head() -> Result<()> {
            use std::{sync::atomic::AtomicBool, time::Instant};
            let target = Qwen35Moe::load(Path::new(&std::env::var("MLXL3_MTP_TEST_MODEL")?))?;
            let stock = std::env::var("MLXL3_MTP_TEST_HEAD")?;
            let explicit = std::env::var("MLXL3_MTP_TEST_COMPACT_HEAD")?;
            ensure!(
                !Path::new(&stock).join("draft_head_q4.json").exists(),
                "requires stock fixture"
            );
            assert!(
                Head::load_cancellable(Path::new(&stock), &target, &AtomicBool::new(true)).is_err()
            );
            let started = Instant::now();
            let automatic = Head::load(Path::new(&stock), &target)?;
            eprintln!(
                "Default head first load: {:.3}s",
                started.elapsed().as_secs_f64()
            );
            assert_eq!(automatic.draft_ids.as_ref().unwrap().shape(), &[79_616]);
            let path = target
                .mtp_projection_path(&AtomicBool::new(false))?
                .unwrap();
            let stamp = std::fs::metadata(path.join("draft_head_q4.safetensors"))?.modified()?;
            let started = Instant::now();
            let warm = Head::load(Path::new(&stock), &target)?;
            eprintln!(
                "Default head cached load: {:.3}s",
                started.elapsed().as_secs_f64()
            );
            assert_eq!(
                std::fs::metadata(path.join("draft_head_q4.safetensors"))?.modified()?,
                stamp
            );
            let explicit = Head::load(Path::new(&explicit), &target)?;
            let expected = explicit.draft_head.as_ref().unwrap().weight_arrays();
            for head in [&automatic, &warm] {
                assert_eq!(
                    head.draft_ids.as_ref().unwrap().to_u32()?,
                    explicit.draft_ids.as_ref().unwrap().to_u32()?
                );
                for (actual, expected) in head
                    .draft_head
                    .as_ref()
                    .unwrap()
                    .weight_arrays()
                    .iter()
                    .zip(&expected)
                {
                    assert_eq!(actual.to_bytes()?, expected.to_bytes()?);
                }
            }
            Ok(())
        }

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

/// Test-only MTP cycle instrumentation: draft confidence per level (T0a) and
/// host phase marks on the `CLOCK_UPTIME_RAW` base used by the P4 command
/// buffer timing dylib (T0b). Nothing is recorded unless a test installs a
/// recorder on its own thread, and production builds contain none of it.
#[cfg(all(test, feature = "mlx"))]
pub(crate) mod cycle_trace {
    use std::cell::RefCell;

    #[derive(Clone, Debug, serde::Serialize)]
    pub(crate) struct Mark {
        pub name: &'static str,
        pub uptime_ns: u64,
    }

    /// One drafted level: the device proposal, and the host's reading of the
    /// same compact-vocabulary log-probabilities.
    #[derive(Clone, Debug, serde::Serialize)]
    pub(crate) struct Level {
        pub proposal: u32,
        pub host_proposal: u32,
        pub top1_log_prob: f32,
        pub top2_log_prob: f32,
        pub top1_ties: usize,
        pub finite: bool,
    }

    /// One EVICT decision: drafted levels, the device top-1 log-probability bits
    /// that rode the ID transfer, and the verified prefix length.
    #[derive(Clone, Debug, serde::Serialize)]
    pub(crate) struct Eviction {
        pub drafted: usize,
        pub top_bits: Vec<u32>,
        pub kept: usize,
    }

    #[derive(Debug, Default, serde::Serialize)]
    pub(crate) struct Recording {
        pub confidence: bool,
        pub capacity: usize,
        pub overflowed: bool,
        pub marks: Vec<Mark>,
        pub levels: Vec<Level>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        pub evictions: Vec<Eviction>,
    }

    thread_local! {
        static RECORDER: RefCell<Option<Recording>> = const { RefCell::new(None) };
    }

    pub(crate) fn uptime_ns() -> u64 {
        unsafe extern "C" {
            fn clock_gettime_nsec_np(clock: u32) -> u64;
        }
        const CLOCK_UPTIME_RAW: u32 = 8;
        unsafe { clock_gettime_nsec_np(CLOCK_UPTIME_RAW) }
    }

    /// Install a bounded recorder; at most `capacity` marks and levels each.
    pub(crate) fn start(confidence: bool, capacity: usize) {
        RECORDER.set(Some(Recording {
            confidence,
            capacity,
            ..Default::default()
        }));
    }

    pub(crate) fn finish() -> Option<Recording> {
        RECORDER.take()
    }

    pub(crate) fn mark(name: &'static str) {
        RECORDER.with_borrow_mut(|recording| {
            if let Some(r) = recording {
                if r.marks.len() < r.capacity {
                    r.marks.push(Mark {
                        name,
                        uptime_ns: uptime_ns(),
                    });
                } else {
                    r.overflowed = true;
                }
            }
        });
    }

    pub(crate) fn evict(drafted: usize, tops: &[f32], kept: usize) {
        RECORDER.with_borrow_mut(|recording| {
            if let Some(r) = recording {
                if r.evictions.len() < r.capacity {
                    r.evictions.push(Eviction {
                        drafted,
                        top_bits: tops.iter().map(|top| top.to_bits()).collect(),
                        kept,
                    });
                } else {
                    r.overflowed = true;
                }
            }
        });
    }

    pub(crate) fn confidence() -> bool {
        RECORDER.with_borrow(|r| r.as_ref().is_some_and(|r| r.confidence))
    }

    /// Host top-1/top-2 of one level's log-probabilities. The host proposal is
    /// the first maximum, mapped through the compact vocabulary when present.
    pub(crate) fn level(proposal: u32, log_probs: &[f32], ids: Option<&[u32]>) {
        let mut best = 0usize;
        let mut top1 = f32::NEG_INFINITY;
        let mut top2 = f32::NEG_INFINITY;
        let mut ties = 0usize;
        let mut finite = !log_probs.is_empty();
        for (index, &value) in log_probs.iter().enumerate() {
            finite &= value.is_finite();
            if value > top1 {
                top2 = top1;
                top1 = value;
                best = index;
                ties = 1;
            } else {
                if value == top1 {
                    ties += 1;
                }
                if value > top2 {
                    top2 = value;
                }
            }
        }
        let host_proposal = match ids {
            Some(ids) => ids.get(best).copied().unwrap_or(u32::MAX),
            None => u32::try_from(best).unwrap_or(u32::MAX),
        };
        RECORDER.with_borrow_mut(|recording| {
            if let Some(r) = recording {
                if r.levels.len() < r.capacity {
                    r.levels.push(Level {
                        proposal,
                        host_proposal,
                        top1_log_prob: top1,
                        top2_log_prob: top2,
                        top1_ties: ties,
                        finite,
                    });
                } else {
                    r.overflowed = true;
                }
            }
        });
    }

    #[test]
    fn host_level_reading_keeps_first_maximum_ties_and_bounds() {
        start(true, 2);
        level(7, &[-3.0, -0.5, -1.0, -0.5], None);
        level(9, &[-2.0, -0.25, f32::NAN], Some(&[40, 41, 42]));
        level(1, &[0.0], None);
        let r = finish().unwrap();
        assert!(r.overflowed && r.levels.len() == 2 && finish().is_none());
        let (a, b) = (&r.levels[0], &r.levels[1]);
        assert_eq!((a.host_proposal, a.top1_ties, a.finite), (1, 2, true));
        assert_eq!((a.top1_log_prob, a.top2_log_prob), (-0.5, -0.5));
        assert_eq!((b.host_proposal, b.top1_ties, b.finite), (41, 1, false));
        assert_eq!((b.top1_log_prob, b.top2_log_prob), (-0.25, -2.0));
        mark("ignored without recorder");
        evict(3, &[-0.5, -1.0, -2.0], 1);
        assert!(!confidence() && finish().is_none());
        start(false, 1);
        evict(2, &[-0.25, f32::NEG_INFINITY], 1);
        evict(1, &[0.0], 1);
        let r = finish().unwrap();
        assert!(r.overflowed && r.evictions.len() == 1 && r.levels.is_empty());
        let e = &r.evictions[0];
        assert_eq!((e.drafted, e.kept), (2, 1));
        assert_eq!(
            e.top_bits,
            [(-0.25f32).to_bits(), f32::NEG_INFINITY.to_bits()]
        );
    }
}

/// Q36-MTP-ACCEPT-POS-T0A / Q36-MTP-CYCLE-TIMELINE-T0B: real `Head`/`Session`
/// cycles on the 35B, with per-cycle width/acceptance, host phase marks and
/// (optionally) draft confidence. Every MTP sample must equal the AR reference
/// token for token, and the production histogram must match the counters.
#[cfg(all(test, feature = "mlx", feature = "chat"))]
mod cycle_diagnostics {
    use super::{Head, Session, cycle_trace};
    use crate::{qwen35::Qwen35Moe, tokenizer::ChatTokenizer};
    use anyhow::{Context, Result, ensure};
    use std::path::Path;

    fn env_or(name: &str, default: &str) -> String {
        std::env::var(name).unwrap_or_else(|_| default.into())
    }

    fn prefill(model: &mut Qwen35Moe, head: &mut Head, input: &[u32]) -> Result<u32> {
        model.reset();
        head.reset()?;
        let (logits, raw) = model.forward_mtp(input)?;
        head.extend_cache(
            model,
            &raw.slice(1, 0, input.len() as i32 - 1)?,
            &input[1..],
        )?;
        logits
            .chat_greedy_ids()?
            .last()
            .copied()
            .context("missing first token")
    }

    #[test]
    #[ignore = "physical Apple GPU, MLXL3_MTP_TEST_MODEL/HEAD, MLXL3_MTP_DIAG_PROMPTS and MLXL3_MTP_DIAG_OUTPUT; diagnostic, run alone"]
    fn mtp_cycle_diagnostics() -> Result<()> {
        let model_path = std::env::var("MLXL3_MTP_TEST_MODEL")?;
        let prompts_dir = std::env::var("MLXL3_MTP_DIAG_PROMPTS")?;
        let output = std::env::var("MLXL3_MTP_DIAG_OUTPUT")?;
        let tokens: usize = env_or("MLXL3_MTP_DIAG_TOKENS", "128").parse()?;
        let warmup: usize = env_or("MLXL3_MTP_DIAG_WARMUP", "128").parse()?;
        let confidence = match env_or("MLXL3_MTP_DIAG_CONFIDENCE", "0").as_str() {
            "0" => false,
            "1" => true,
            other => anyhow::bail!("MLXL3_MTP_DIAG_CONFIDENCE must be 0 or 1, not {other:?}"),
        };
        // T1 test-only wirings; the production `post` comes from MLXL3_MTP_POST_NORM.
        let forced = match std::env::var("MLXL3_MTP_DIAG_HIDDEN") {
            Err(std::env::VarError::NotPresent) => None,
            Ok(value) => Some(match value.as_str() {
                "pre" => (false, false),
                "seed" => (true, false),
                "chain" => (false, true),
                "post" => (true, true),
                other => anyhow::bail!(
                    "MLXL3_MTP_DIAG_HIDDEN must be pre, seed, chain or post, not {other:?}"
                ),
            }),
            Err(error) => return Err(error.into()),
        };
        let env_post_norm = std::env::var("MLXL3_MTP_POST_NORM").ok();
        ensure!(
            forced.is_none() || env_post_norm.is_none(),
            "MLXL3_MTP_DIAG_HIDDEN and MLXL3_MTP_POST_NORM are exclusive"
        );
        let compact = match env_or("MLXL3_MTP_DIAG_COMPACT", "1").as_str() {
            "1" => true,
            "0" => false,
            other => anyhow::bail!("MLXL3_MTP_DIAG_COMPACT must be 0 or 1, not {other:?}"),
        };
        let depths = env_or("MLXL3_MTP_DIAG_DEPTHS", "1,2,3")
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<usize>, _>>()?;
        ensure!(
            (2..=1024).contains(&tokens) && warmup <= 1024,
            "unsupported token budget"
        );
        ensure!(
            !depths.is_empty() && depths.iter().all(|d| (1..=3).contains(d)),
            "depths must be 1..=3"
        );
        let mut prompt_files = std::fs::read_dir(&prompts_dir)?
            .map(|entry| Ok(entry?.path()))
            .collect::<Result<Vec<_>>>()?;
        prompt_files.retain(|path| path.extension().is_some_and(|e| e == "txt"));
        prompt_files.sort();
        ensure!(!prompt_files.is_empty(), "no prompt files in {prompts_dir}");

        let mut model = Qwen35Moe::load(Path::new(&model_path))?;
        let mut head = Head::load(Path::new(&std::env::var("MLXL3_MTP_TEST_HEAD")?), &model)?;
        if let Some((seed, chain)) = forced {
            head.set_test_wiring(seed, chain);
        }
        if !compact {
            // Requalification: draft with the target's full lm_head (248 320 IDs).
            head.use_target_draft_head();
        }
        let hidden_variant = head.wiring_name();
        let evict = head.evict();
        let tokenizer = ChatTokenizer::load(Path::new(&model_path))?;
        let mut report = serde_json::json!({
            "tokens": tokens, "warmup": warmup, "confidence": confidence,
            "hidden_variant": hidden_variant,
            "env_post_norm": env_post_norm, "evict": evict,
            "env_evict": std::env::var("MLXL3_MTP_EVICT").ok(),
            "env_pipeline": std::env::var("MLXL3_QWEN_PIPELINE").ok(),
            "env_head_lane": std::env::var("MLXL3_Q36_HEAD_LANE").ok(),
            "compact_head": head.compact_vocabulary(), "cases": [],
        });
        for prompt_file in &prompt_files {
            let name = prompt_file
                .file_stem()
                .and_then(|s| s.to_str())
                .context("prompt name")?
                .to_string();
            let prompt = std::fs::read_to_string(prompt_file)?;
            let rendered = tokenizer.render_values(
                &serde_json::json!([{"role":"user","content":prompt}]),
                Some(&[]),
            )?;
            let input = tokenizer.encode(&rendered)?;
            ensure!(
                input.len() > 1 && input.len() + tokens.max(warmup) + 8 <= 4096,
                "unsupported prompt {name}"
            );
            // Independent AR reference: one target step per token.
            let mut reference = Vec::with_capacity(tokens);
            let mut next = prefill(&mut model, &mut head, &input)?;
            while reference.len() < tokens {
                reference.push(next);
                if reference.len() < tokens {
                    next = model
                        .forward(next)?
                        .chat_greedy_ids()?
                        .last()
                        .copied()
                        .context("missing AR token")?;
                }
            }
            for &depth in &depths {
                let mut run = |budget: usize, record: bool| -> Result<serde_json::Value> {
                    let mut next = prefill(&mut model, &mut head, &input)?;
                    let mut session = Session::new(depth)?;
                    let mut generated = Vec::with_capacity(budget);
                    let mut cycles = Vec::new();
                    if record {
                        cycle_trace::start(confidence, 64 * budget + 64);
                    }
                    let started = std::time::Instant::now();
                    while generated.len() < budget {
                        generated.push(next);
                        if generated.len() == budget {
                            break;
                        }
                        let before = (session.proposed, session.accepted, session.depth_blocks);
                        next = session.advance(
                            &mut model,
                            &mut head,
                            next,
                            4096,
                            budget - generated.len(),
                            |_| false,
                        )?;
                        if session.depth_blocks != before.2 {
                            cycles.push([session.proposed - before.0, session.accepted - before.1]);
                        }
                    }
                    crate::array::synchronize()?;
                    let seconds = started.elapsed().as_secs_f64();
                    let recording = if record { cycle_trace::finish() } else { None };
                    let histogram = session.accept_histogram;
                    ensure!(
                        generated[..] == reference[..budget.min(tokens)] || budget > tokens,
                        "{name} depth {depth}: MTP tokens differ from AR"
                    );
                    ensure!(session.lookup_blocks == 0, "lookup must stay off");
                    ensure!(
                        session.lookup_accept_histogram == Default::default(),
                        "lookup histogram must stay empty"
                    );
                    ensure!(
                        histogram.proposed() == session.proposed
                            && histogram.accepted() == session.accepted
                            && histogram.speculative_cycles() == session.blocks
                            && histogram.cycles() == session.depth_blocks.iter().sum::<usize>()
                            && histogram.cycles() == cycles.len()
                            && histogram.cycles() + histogram.accepted() == budget - 1,
                        "{name} depth {depth}: histogram disagrees with counters"
                    );
                    for &[width, accepted] in &cycles {
                        ensure!(accepted <= width && width <= depth, "invalid cycle");
                    }
                    let mut case = serde_json::json!({
                        "prompt": name, "input_tokens": input.len(), "depth": depth,
                        "budget": budget, "seconds_not_a_benchmark": seconds,
                        "tokens": generated, "cycles": cycles,
                        "histogram": histogram, "proposed": session.proposed,
                        "accepted": session.accepted, "blocks": session.blocks,
                        "depth_blocks": session.depth_blocks,
                    });
                    if let Some(recording) = recording {
                        ensure!(!recording.overflowed, "trace overflowed");
                        let starts = recording
                            .marks
                            .iter()
                            .filter(|m| m.name == "cycle_start")
                            .count();
                        ensure!(starts == cycles.len(), "one cycle_start per cycle");
                        ensure!(
                            recording
                                .marks
                                .windows(2)
                                .all(|w| w[0].uptime_ns <= w[1].uptime_ns),
                            "marks are not monotonic"
                        );
                        // EVICT drafts `drafted` levels and verifies the first `kept`.
                        let drafted = if evict {
                            let e = &recording.evictions;
                            ensure!(
                                e.len() == session.blocks
                                    && e.iter().map(|e| e.kept).sum::<usize>() == session.proposed
                                    && e.iter().zip(cycles.iter().filter(|c| c[0] > 0)).all(
                                        |(e, c)| e.kept == c[0]
                                            && e.kept <= e.drafted
                                            && e.drafted <= depth
                                            && e.top_bits.len() == e.drafted
                                    ),
                                "{name} depth {depth}: EVICT records disagree with cycles"
                            );
                            for e in e {
                                let bands = e
                                    .top_bits
                                    .iter()
                                    .map(|&bits| crate::contracts::evict_band(f32::from_bits(bits)))
                                    .collect::<Option<Vec<_>>>()
                                    .context("non-finite EVICT top")?;
                                ensure!(
                                    crate::contracts::evict_width(&bands, e.drafted)
                                        == Some(e.kept),
                                    "{name} depth {depth}: recorded EVICT width is not the policy's"
                                );
                            }
                            e.iter().map(|e| e.drafted).sum()
                        } else {
                            ensure!(recording.evictions.is_empty(), "EVICT recorded while off");
                            session.proposed
                        };
                        if confidence {
                            ensure!(
                                recording.levels.len() == drafted,
                                "one confidence level per drafted level"
                            );
                            if evict {
                                let device = recording.evictions.iter().flat_map(|e| &e.top_bits);
                                let differing = recording
                                    .levels
                                    .iter()
                                    .zip(device)
                                    .filter(|(l, bits)| l.top1_log_prob.to_bits() != **bits)
                                    .count();
                                case["evict_top_bit_mismatches"] = differing.into();
                                ensure!(
                                    differing == 0,
                                    "{name} depth {depth}: {differing} device/host top-1 bit mismatches"
                                );
                            }
                            ensure!(
                                recording.levels.iter().all(|l| l.finite
                                    && l.top1_log_prob.is_finite()
                                    && l.top1_log_prob <= 0.0
                                    && l.top2_log_prob <= l.top1_log_prob),
                                "invalid confidence values"
                            );
                            let mismatched = recording
                                .levels
                                .iter()
                                .filter(|l| l.proposal != l.host_proposal)
                                .count();
                            case["host_argmax_mismatches"] = mismatched.into();
                            ensure!(
                                mismatched == 0,
                                "{name} depth {depth}: {mismatched} host argmax mismatches"
                            );
                        }
                        case["trace"] = serde_json::to_value(&recording)?;
                    }
                    model.reset();
                    head.reset()?;
                    crate::array::release_idle_cache()?;
                    Ok(case)
                };
                if warmup > 0 {
                    run(warmup, false)?;
                }
                let case = run(tokens, true)?;
                eprintln!(
                    "{name} depth {depth}: {} cycles, histogram {}",
                    case["cycles"].as_array().map_or(0, Vec::len),
                    case["histogram"]
                );
                report["cases"].as_array_mut().unwrap().push(case);
            }
        }
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
        Ok(())
    }
}

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
