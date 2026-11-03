use crate::{checkpoint::Checkpoint, flash_support};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::fs::File;

pub(crate) struct FlashConfig {
    pub eps: f32,
    pub rope: f32,
    pub context: i32,
}
impl FlashConfig {
    pub fn load(c: &Checkpoint) -> Result<Self> {
        ensure!(
            c.model_type == "qwen4_exp",
            "expected Flash Next (qwen4_exp)"
        );
        let root: Value = serde_json::from_reader(File::open(c.path.join("config.json"))?)?;
        let t = &root["text_config"];
        for (key, value) in [
            ("hidden_size", 2560),
            ("hc_count", 4),
            ("hc_lowrank", 320),
            ("num_hidden_layers", 48),
            ("num_experts", 512),
            ("num_experts_per_tok", 10),
            ("moe_intermediate_size", 640),
            ("head_dim", 256),
            ("num_attention_heads", 24),
            ("num_key_value_heads", 2),
            ("linear_key_head_dim", 128),
            ("linear_value_head_dim", 128),
            ("linear_num_key_heads", 16),
            ("linear_num_value_heads", 48),
            ("linear_conv_kernel_dim", 4),
            ("indexer_n_heads", 4),
            ("indexer_kv_heads", 1),
            ("indexer_head_dim", 128),
            ("indexer_compress_ratio", 4),
            ("indexer_budget", 2048),
            ("eos_token_id", 248044),
            ("vocab_size", 248320),
            ("ple_embed_dim", 2560),
            ("ngram_size", 3),
            ("heads_per_ngram", 8),
            ("ple_conv_kernel_size", 4),
        ] {
            ensure!(
                t[key].as_i64() == Some(value),
                "unsupported Flash Next {key}"
            );
        }
        ensure!(
            t["output_gate_type"] == "sigmoid" && t["ple_layer_ids"] == serde_json::json!([2]),
            "unsupported Flash PLE/gate configuration"
        );
        let eps = t["rms_norm_eps"]
            .as_f64()
            .context("missing Flash epsilon")? as f32;
        let rope = t["rope_parameters"]["rope_theta"]
            .as_f64()
            .context("missing Flash RoPE")? as f32;
        let context = i32::try_from(
            t["max_position_embeddings"]
                .as_u64()
                .context("missing context limit")?,
        )?;
        ensure!(
            eps.is_finite()
                && eps > 0.
                && rope.is_finite()
                && rope > 0.
                && context > 0
                && t["rope_parameters"]["partial_rotary_factor"] == 0.25,
            "invalid Flash numerical configuration"
        );
        let types = t["layer_types"].as_array().context("missing layer types")?;
        ensure!(types.len() == 48, "invalid layer count");
        for (i, kind) in types.iter().enumerate() {
            let linear = i % 4 != 3;
            ensure!(
                *kind
                    == if linear {
                        "linear_attention"
                    } else {
                        "full_attention"
                    },
                "invalid Flash layer type"
            );
        }
        Ok(Self { eps, rope, context })
    }
}
pub(crate) fn layer_slots() -> Result<[usize; 48]> {
    let budget = match std::env::var("MLXL3_SSD_CACHE_GIB") {
        Ok(value) => Some(
            value
                .parse::<f64>()
                .context("invalid MLXL3_SSD_CACHE_GIB")?,
        ),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    let slots = if let Some(budget) = budget {
        ensure!(
            budget.is_finite() && (0.6..=12.).contains(&budget),
            "SSD expert cache budget must be 0.6..12 GiB"
        );
        ((budget * (1u64 << 30) as f64 / (48. * flash_support::EXPERT_BYTES as f64)).floor()
            as usize)
            .clamp(10, 512)
    } else {
        #[cfg(target_os = "macos")]
        let available = flash_support::available_memory_bytes()?;
        #[cfg(not(target_os = "macos"))]
        let available = 14 << 30;
        let slots = flash_support::automatic_cache_slots(available)
            .context("not enough system memory for Flash's minimum SSD cache and memory reserve")?;
        eprintln!(
            "Flash Next: auto cache {:.2} GiB ({slots} slots/layer), estimated available {:.2} GiB",
            slots as f64 * 48. * flash_support::EXPERT_BYTES as f64 / (1u64 << 30) as f64,
            available as f64 / (1u64 << 30) as f64
        );
        slots
    };
    let layer_slots = match std::env::var("MLXL3_SSD_LAYER_SLOTS") {
            Ok(value) => flash_support::parse_layer_slots(&value, slots).context(
                "MLXL3_SSD_LAYER_SLOTS requires 48 capacities in 20..512 with exactly the current total cache budget",
            )?,
            Err(std::env::VarError::NotPresent) => [slots; 48],
            Err(error) => return Err(error.into()),
        };
    Ok(layer_slots)
}
