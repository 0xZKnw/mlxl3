//! Native Qwen3.5/3.6 NextN predictor. Architecture follows Qwen and the
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

        fn mixed(&self, target: &Qwen35Moe, hidden: &Array, next: &[u32]) -> Result<Array> {
            let time = i32::try_from(next.len())?;
            let h = self.layout.hidden_size;
            ensure!(
                time > 0 && hidden.shape() == [1, time, h],
                "invalid MTP hidden input"
            );
            let e = target
                .mtp_embeddings(next)?
                .rms_norm(&self.embedding_norm, self.layout.rms_norm_eps)?;
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
            let mixed = self.mixed(target, hidden, next)?;
            self.attention
                .append_kv(&mixed.rms_norm(&self.input_norm, self.layout.rms_norm_eps)?)?;
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
            let time = i32::try_from(next.len())?;
            let h = self.layout.hidden_size;
            let mixed = self.mixed(target, hidden, next)?;
            let attention = self
                .attention
                .forward(&mixed.rms_norm(&self.input_norm, self.layout.rms_norm_eps)?)?;
            let residual = mixed.add(&attention)?;
            let input = residual
                .rms_norm(&self.post_norm, self.layout.rms_norm_eps)?
                .reshape(&[time, h])?;
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
                    let (indices, scores) =
                        router::topk(&probabilities, self.layout.num_experts_per_tok, true)?;
                    let broadcast = input.reshape(&[time, 1, 1, h])?;
                    let gate = gate.gather(&broadcast, &indices)?;
                    let up = up.gather(&broadcast, &indices)?;
                    let routed = down
                        .gather(&gate.swiglu(&up)?, &indices)?
                        .reshape(&[time, self.layout.num_experts_per_tok as i32, h])?
                        .mul(&scores.reshape(&[
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
            let output = residual
                .add(&mlp.reshape(&[1, time, h])?)?
                .rms_norm(&self.norm, self.layout.rms_norm_eps)?;
            Ok(output)
        }

        pub fn forward(
            &mut self,
            target: &Qwen35Moe,
            hidden: &Array,
            next: &[u32],
        ) -> Result<Array> {
            let hidden = self.hidden(target, hidden, next)?;
            target.mtp_logits(&hidden)
        }
    }

    #[derive(Default)]
    pub struct Session {
        pending: VecDeque<u32>,
        pub proposed: usize,
        pub accepted: usize,
        pub blocks: usize,
    }

    impl Session {
        /// Depth one keeps the draft KV exact: the predictor always consumes a
        /// real target residual. Every delivered token is chosen by the target.
        pub fn advance(
            &mut self,
            target: &mut Qwen35Moe,
            head: &mut Head,
            anchor: u32,
            context_limit: usize,
            output_remaining: usize,
        ) -> Result<u32> {
            if let Some(next) = self.pending.pop_front() {
                return Ok(next);
            }
            let remaining = context_limit
                .checked_sub(usize::try_from(target.offset())?)
                .context("MTP context exhausted")?;
            let width = crate::speculative::bounded_proposals(remaining, output_remaining)
                .context("MTP budget exhausted")?
                .min(1);
            let hidden = target.mtp_hidden()?.try_clone()?;
            if width == 0 {
                let logits = target.forward(anchor)?;
                return logits
                    .chat_greedy_ids()?
                    .last()
                    .copied()
                    .context("missing MTP fallback token");
            }
            let proposal = head
                .forward(target, &hidden, &[anchor])?
                .chat_greedy_ids()?[0];
            let (logits, raw) = target.verify_mtp(&[anchor, proposal])?;
            let tokens = logits.chat_greedy_ids()?;
            let accepted = crate::speculative::greedy_accept(&[proposal], &tokens)
                .context("invalid MTP verification output")?;
            let retained = 1 + accepted.accepted_draft_tokens;
            target.commit_dflash_verification(retained, 2)?;
            target.set_mtp_hidden(raw.slice(1, retained as i32 - 1, retained as i32)?)?;
            if accepted.accepted_draft_tokens == 1 {
                // Replace approximate drafting history with the real trunk
                // residual before predicting another position.
                head.extend_cache(target, &raw.slice(1, 0, 1)?, &[proposal])?;
                self.pending.push_back(accepted.target_token);
            }
            self.proposed += 1;
            self.accepted += accepted.accepted_draft_tokens;
            self.blocks += 1;
            Ok(if accepted.accepted_draft_tokens == 1 {
                proposal
            } else {
                accepted.target_token
            })
        }
    }

    #[cfg(test)]
    mod cache_tests {
        use super::*;

        fn bytes(cache: &Cache) -> Result<(Vec<u8>, Vec<u8>)> {
            Ok((cache.keys.to_bytes()?, cache.values.to_bytes()?))
        }

        #[test]
        #[ignore = "requires local Qwen/MTP checkpoints and physical Apple GPU"]
        fn cache_only_matches_full_head_and_next_prediction() -> Result<()> {
            let target = Qwen35Moe::load(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
            let mut head = Head::load(Path::new("models/Qwen3.6-35B-A3B-MTP-4bit"), &target)?;
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
mod tests {
    use super::*;
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
