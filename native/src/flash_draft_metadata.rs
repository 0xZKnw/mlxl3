//! Shared CPU-only validation for the raw Flash drafts; no weight payloads are read.
use crate::checkpoint::TensorInfo;
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{collections::BTreeMap, fs::File, io::Read, path::Path};

const H: i32 = 2560;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Dflash2,
    #[cfg(test)]
    Dspark,
}

impl Kind {
    #[cfg(test)]
    pub(crate) fn parse(value: &str) -> Result<Self> {
        match value {
            "dflash2" => Ok(Self::Dflash2),
            "dspark" => Ok(Self::Dspark),
            _ => anyhow::bail!("unknown Flash draft kind"),
        }
    }
    pub(crate) fn captures(self) -> [usize; 5] {
        match self {
            Self::Dflash2 => [5, 15, 25, 35, 45],
            #[cfg(test)]
            Self::Dspark => [1, 12, 23, 34, 45],
        }
    }
    pub(crate) fn mask(self) -> u32 {
        if self == Self::Dflash2 {
            248077
        } else {
            248070
        }
    }
    #[cfg(any(all(feature = "mlx", feature = "chat"), test))]
    pub(crate) fn rope(self) -> f32 {
        if self == Self::Dflash2 {
            10_000.0
        } else {
            10_000_000.0
        }
    }
    pub(crate) fn intermediate(self) -> i32 {
        if self == Self::Dflash2 { 7680 } else { 9728 }
    }
    #[cfg(any(all(feature = "mlx", feature = "chat"), test, kani))]
    pub(crate) fn rows(self, proposals: usize) -> Result<i32> {
        ensure!((1..=7).contains(&proposals), "invalid draft proposal count");
        Ok(i32::try_from(proposals)? + i32::from(self == Self::Dflash2))
    }
}

pub(crate) struct RawDraft {
    pub(crate) tensors: BTreeMap<String, TensorInfo>,
    pub(crate) shapes: BTreeMap<String, Vec<i32>>,
}

pub(crate) fn inspect(path: &Path, kind: Kind) -> Result<RawDraft> {
    let tensors = crate::checkpoint::read_header(&path.join("model.safetensors"))
        .context("reading raw BF16 Flash draft header")?;
    let mut data = Vec::new();
    File::open(path.join("config.json"))?
        .take(65_537)
        .read_to_end(&mut data)?;
    ensure!(data.len() <= 65_536, "oversized Flash draft config");
    let config: Value = serde_json::from_slice(&data)?;
    let architecture = if kind == Kind::Dflash2 {
        "DFlash2DraftModel"
    } else {
        "DSparkDraftModel"
    };
    ensure!(
        config["architectures"] == serde_json::json!([architecture])
            && config["hidden_size"] == H
            && config["num_hidden_layers"] == 5
            && config["intermediate_size"] == kind.intermediate()
            && config["num_attention_heads"] == 32
            && config["num_key_value_heads"] == 8
            && config["head_dim"] == 128
            && config["vocab_size"] == 248320
            && config["rms_norm_eps"] == 1e-6
            && config["attention_bias"] == false
            && config["dflash_config"]["target_layer_ids"] == serde_json::json!(kind.captures())
            && config["dflash_config"]["mask_token_id"] == kind.mask(),
        "unsupported Flash draft configuration"
    );
    if kind == Kind::Dflash2 {
        ensure!(
            config["rope_parameters"]["rope_theta"] == 10000
                && config["sliding_window"] == 2048
                && config["use_sliding_window"] == true
                && config["dflash_config"]["conv_group_size"] == 16
                && config["dflash_config"]["conv_kernel_size"] == 2
                && config["dflash_config"]["selector_rank"] == 256
                && config["dflash_config"]["selector_top_k"] == 16,
            "unsupported DFlash2 geometry"
        );
    } else {
        ensure!(
            config["rope_theta"] == 10_000_000
                && config["sliding_window"].is_null()
                && config["use_sliding_window"] == false
                && config["dflash_config"]["markov_head_type"] == "vanilla"
                && config["dflash_config"]["markov_rank"] == 256
                && config["dflash_config"]["confidence_head_with_markov"] == true
                && config["sample_from_anchor"].as_bool().unwrap_or(true),
            "unsupported DSpark geometry"
        );
    }
    let mut shapes = BTreeMap::new();
    shapes.insert("fc.weight".to_owned(), vec![H, 5 * H]);
    for name in ["hidden_norm.weight", "norm.weight"] {
        shapes.insert(name.to_owned(), vec![H]);
    }
    for layer in 0..5 {
        let p = format!("layers.{layer}");
        for name in ["input_layernorm", "post_attention_layernorm"] {
            shapes.insert(format!("{p}.{name}.weight"), vec![H]);
        }
        for name in ["q_norm", "k_norm"] {
            shapes.insert(format!("{p}.self_attn.{name}.weight"), vec![128]);
        }
        for (name, output, input) in [
            ("q_proj", 4096, H),
            ("k_proj", 1024, H),
            ("v_proj", 1024, H),
            ("o_proj", H, 4096),
        ] {
            shapes.insert(format!("{p}.self_attn.{name}.weight"), vec![output, input]);
        }
        for name in ["gate_proj", "up_proj"] {
            shapes.insert(
                format!("{p}.mlp.{name}.weight"),
                vec![kind.intermediate(), H],
            );
        }
        shapes.insert(
            format!("{p}.mlp.down_proj.weight"),
            vec![H, kind.intermediate()],
        );
        if kind == Kind::Dflash2 {
            for name in ["attention_conv", "mlp_conv"] {
                shapes.insert(format!("{p}.{name}.base_kernel"), vec![2, 2, H]);
                shapes.insert(format!("{p}.{name}.kernel_projection.weight"), vec![640, H]);
            }
        }
    }
    if kind == Kind::Dflash2 {
        shapes.insert(
            "candidate_selector.hidden_projection.weight".to_owned(),
            vec![256, H],
        );
        for name in ["predecessor_codebook", "successor_codebook"] {
            shapes.insert(format!("candidate_selector.{name}"), vec![248320, 256]);
        }
    } else {
        for name in ["markov_w1", "markov_w2"] {
            shapes.insert(format!("markov_head.{name}.weight"), vec![248320, 256]);
        }
        shapes.insert("confidence_head.proj.weight".to_owned(), vec![1, 2816]);
        shapes.insert("confidence_head.proj.bias".to_owned(), vec![1]);
    }
    ensure!(
        tensors.len() == shapes.len(),
        "unexpected or missing draft tensors"
    );
    let raw = RawDraft { tensors, shapes };
    for (name, shape) in &raw.shapes {
        let info = raw
            .tensors
            .get(name)
            .with_context(|| format!("missing {name}"))?;
        ensure!(
            info.dtype == "BF16"
                && info.shape == shape.iter().map(|&n| n as usize).collect::<Vec<_>>(),
            "invalid draft tensor {name}"
        );
    }
    Ok(raw)
}

#[cfg(kani)]
#[kani::proof]
fn raw_flash_draft_rows_reject_outside_one_to_seven() {
    let proposals: usize = kani::any();
    let result = Kind::Dflash2.rows(proposals);
    if proposals > 0 && proposals < 8 {
        let rows = result.as_ref().expect("valid proposal count");
        assert_eq!(usize::try_from(*rows).unwrap(), proposals + 1);
    } else {
        assert!(result.is_err());
    }
    kani::cover!(proposals == 1 && result.is_ok());
    kani::cover!(proposals == 7 && result.is_ok());
    kani::cover!(proposals == 0 && result.is_err());
    kani::cover!(proposals == usize::MAX && result.is_err());
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, io::Write};
    use tempfile::TempDir;

    // Header-only oracle captured from incoai/Qwen3.8-Flash-Next-DFlash2 @ a22a1744.
    const FLASH_SHAPES: &[(&str, &[usize])] = &[
        ("candidate_selector.hidden_projection.weight", &[256, 2560]),
        ("candidate_selector.predecessor_codebook", &[248320, 256]),
        ("candidate_selector.successor_codebook", &[248320, 256]),
        ("fc.weight", &[2560, 12800]),
        ("hidden_norm.weight", &[2560]),
        ("layers.0.attention_conv.base_kernel", &[2, 2, 2560]),
        (
            "layers.0.attention_conv.kernel_projection.weight",
            &[640, 2560],
        ),
        ("layers.0.input_layernorm.weight", &[2560]),
        ("layers.0.mlp.down_proj.weight", &[2560, 7680]),
        ("layers.0.mlp.gate_proj.weight", &[7680, 2560]),
        ("layers.0.mlp.up_proj.weight", &[7680, 2560]),
        ("layers.0.mlp_conv.base_kernel", &[2, 2, 2560]),
        ("layers.0.mlp_conv.kernel_projection.weight", &[640, 2560]),
        ("layers.0.post_attention_layernorm.weight", &[2560]),
        ("layers.0.self_attn.k_norm.weight", &[128]),
        ("layers.0.self_attn.k_proj.weight", &[1024, 2560]),
        ("layers.0.self_attn.o_proj.weight", &[2560, 4096]),
        ("layers.0.self_attn.q_norm.weight", &[128]),
        ("layers.0.self_attn.q_proj.weight", &[4096, 2560]),
        ("layers.0.self_attn.v_proj.weight", &[1024, 2560]),
        ("layers.1.attention_conv.base_kernel", &[2, 2, 2560]),
        (
            "layers.1.attention_conv.kernel_projection.weight",
            &[640, 2560],
        ),
        ("layers.1.input_layernorm.weight", &[2560]),
        ("layers.1.mlp.down_proj.weight", &[2560, 7680]),
        ("layers.1.mlp.gate_proj.weight", &[7680, 2560]),
        ("layers.1.mlp.up_proj.weight", &[7680, 2560]),
        ("layers.1.mlp_conv.base_kernel", &[2, 2, 2560]),
        ("layers.1.mlp_conv.kernel_projection.weight", &[640, 2560]),
        ("layers.1.post_attention_layernorm.weight", &[2560]),
        ("layers.1.self_attn.k_norm.weight", &[128]),
        ("layers.1.self_attn.k_proj.weight", &[1024, 2560]),
        ("layers.1.self_attn.o_proj.weight", &[2560, 4096]),
        ("layers.1.self_attn.q_norm.weight", &[128]),
        ("layers.1.self_attn.q_proj.weight", &[4096, 2560]),
        ("layers.1.self_attn.v_proj.weight", &[1024, 2560]),
        ("layers.2.attention_conv.base_kernel", &[2, 2, 2560]),
        (
            "layers.2.attention_conv.kernel_projection.weight",
            &[640, 2560],
        ),
        ("layers.2.input_layernorm.weight", &[2560]),
        ("layers.2.mlp.down_proj.weight", &[2560, 7680]),
        ("layers.2.mlp.gate_proj.weight", &[7680, 2560]),
        ("layers.2.mlp.up_proj.weight", &[7680, 2560]),
        ("layers.2.mlp_conv.base_kernel", &[2, 2, 2560]),
        ("layers.2.mlp_conv.kernel_projection.weight", &[640, 2560]),
        ("layers.2.post_attention_layernorm.weight", &[2560]),
        ("layers.2.self_attn.k_norm.weight", &[128]),
        ("layers.2.self_attn.k_proj.weight", &[1024, 2560]),
        ("layers.2.self_attn.o_proj.weight", &[2560, 4096]),
        ("layers.2.self_attn.q_norm.weight", &[128]),
        ("layers.2.self_attn.q_proj.weight", &[4096, 2560]),
        ("layers.2.self_attn.v_proj.weight", &[1024, 2560]),
        ("layers.3.attention_conv.base_kernel", &[2, 2, 2560]),
        (
            "layers.3.attention_conv.kernel_projection.weight",
            &[640, 2560],
        ),
        ("layers.3.input_layernorm.weight", &[2560]),
        ("layers.3.mlp.down_proj.weight", &[2560, 7680]),
        ("layers.3.mlp.gate_proj.weight", &[7680, 2560]),
        ("layers.3.mlp.up_proj.weight", &[7680, 2560]),
        ("layers.3.mlp_conv.base_kernel", &[2, 2, 2560]),
        ("layers.3.mlp_conv.kernel_projection.weight", &[640, 2560]),
        ("layers.3.post_attention_layernorm.weight", &[2560]),
        ("layers.3.self_attn.k_norm.weight", &[128]),
        ("layers.3.self_attn.k_proj.weight", &[1024, 2560]),
        ("layers.3.self_attn.o_proj.weight", &[2560, 4096]),
        ("layers.3.self_attn.q_norm.weight", &[128]),
        ("layers.3.self_attn.q_proj.weight", &[4096, 2560]),
        ("layers.3.self_attn.v_proj.weight", &[1024, 2560]),
        ("layers.4.attention_conv.base_kernel", &[2, 2, 2560]),
        (
            "layers.4.attention_conv.kernel_projection.weight",
            &[640, 2560],
        ),
        ("layers.4.input_layernorm.weight", &[2560]),
        ("layers.4.mlp.down_proj.weight", &[2560, 7680]),
        ("layers.4.mlp.gate_proj.weight", &[7680, 2560]),
        ("layers.4.mlp.up_proj.weight", &[7680, 2560]),
        ("layers.4.mlp_conv.base_kernel", &[2, 2, 2560]),
        ("layers.4.mlp_conv.kernel_projection.weight", &[640, 2560]),
        ("layers.4.post_attention_layernorm.weight", &[2560]),
        ("layers.4.self_attn.k_norm.weight", &[128]),
        ("layers.4.self_attn.k_proj.weight", &[1024, 2560]),
        ("layers.4.self_attn.o_proj.weight", &[2560, 4096]),
        ("layers.4.self_attn.q_norm.weight", &[128]),
        ("layers.4.self_attn.q_proj.weight", &[4096, 2560]),
        ("layers.4.self_attn.v_proj.weight", &[1024, 2560]),
        ("norm.weight", &[2560]),
    ];
    const FLASH_CONFIG: &str = r#"{
  "architectures": [
    "DFlash2DraftModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 248044,
  "dflash_config": {
    "block_size": 8,
    "conv_group_size": 16,
    "conv_kernel_size": 2,
    "mask_token_id": 248077,
    "selector_rank": 256,
    "selector_top_k": 16,
    "target_layer_ids": [
      5,
      15,
      25,
      35,
      45
    ]
  },
  "dtype": "bfloat16",
  "eos_token_id": 248044,
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 2560,
  "initializer_range": 0.02,
  "intermediate_size": 7680,
  "is_causal": false,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention"
  ],
  "max_position_embeddings": 262144,
  "max_window_layers": 5,
  "model_type": "qwen3",
  "num_attention_heads": 32,
  "num_hidden_layers": 5,
  "num_key_value_heads": 8,
  "num_target_layers": 48,
  "pad_token_id": null,
  "rms_norm_eps": 1e-06,
  "rope_parameters": {
    "rope_theta": 10000,
    "rope_type": "default"
  },
  "sliding_window": 2048,
  "tie_word_embeddings": false,
  "transformers_version": "5.12.1",
  "use_cache": false,
  "use_sliding_window": true,
  "vocab_size": 248320
}"#;

    struct Fixture {
        dir: TempDir,
        config: Value,
        entries: serde_json::Map<String, Value>,
        payload_bytes: u64,
    }

    impl Fixture {
        fn new() -> Self {
            let mut fixture = Self {
                dir: tempfile::tempdir().unwrap(),
                config: serde_json::from_str(FLASH_CONFIG).unwrap(),
                entries: FLASH_SHAPES
                    .iter()
                    .map(|&(name, shape)| {
                        (
                            name.to_owned(),
                            serde_json::json!({"dtype": "BF16", "shape": shape}),
                        )
                    })
                    .collect(),
                payload_bytes: 0,
            };
            fixture.repack();
            fixture.write();
            fixture
        }

        fn repack(&mut self) {
            let mut offset = 0u64;
            for entry in self.entries.values_mut() {
                let bytes = entry["shape"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|n| n.as_u64().unwrap())
                    .product::<u64>()
                    * 2;
                entry["data_offsets"] = serde_json::json!([offset, offset + bytes]);
                offset += bytes;
            }
            self.payload_bytes = offset;
        }

        fn write(&self) {
            fs::write(
                self.dir.path().join("config.json"),
                serde_json::to_vec(&self.config).unwrap(),
            )
            .unwrap();
            let header = serde_json::to_vec(&self.entries).unwrap();
            let mut file = File::create(self.dir.path().join("model.safetensors")).unwrap();
            file.write_all(&(header.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(&header).unwrap();
            // Sparse length: real geometry without allocating or reading a 1.2GB body.
            file.set_len(8 + header.len() as u64 + self.payload_bytes)
                .unwrap();
        }

        fn rejects(&self, expected: &str) {
            let error = inspect(self.dir.path(), Kind::Dflash2)
                .err()
                .expect("invalid metadata accepted");
            assert!(format!("{error:#}").contains(expected), "{error:#}");
        }
    }

    #[test]
    fn pinned_sparse_header_and_qwen3_config_are_accepted() {
        let fixture = Fixture::new();
        assert_eq!(fixture.config["model_type"], "qwen3");
        let raw = inspect(fixture.dir.path(), Kind::Dflash2).unwrap();
        assert_eq!(raw.tensors.len(), 81);
        assert_eq!(raw.shapes.len(), 81);
        assert_eq!(fixture.payload_bytes, 1_206_131_200);
        let parameters: usize = raw
            .tensors
            .values()
            .map(|t| t.shape.iter().product::<usize>())
            .sum();
        assert_eq!(parameters, 603_065_600);
        for &(name, shape) in FLASH_SHAPES {
            assert_eq!(raw.tensors[name].dtype, "BF16");
            assert_eq!(raw.tensors[name].shape, shape);
            assert_eq!(
                raw.shapes[name],
                shape
                    .iter()
                    .map(|&n| i32::try_from(n).unwrap())
                    .collect::<Vec<_>>()
            );
            raw.tensors[name].source().unwrap();
        }
    }

    #[test]
    fn geometry_fields_are_checked_by_shared_inspector() {
        let mutations = [
            ("/architectures", serde_json::json!(["DSparkDraftModel"])),
            ("/hidden_size", serde_json::json!(2559)),
            ("/num_hidden_layers", serde_json::json!(4)),
            ("/intermediate_size", serde_json::json!(9728)),
            ("/num_attention_heads", serde_json::json!(31)),
            ("/num_key_value_heads", serde_json::json!(7)),
            ("/head_dim", serde_json::json!(64)),
            ("/vocab_size", serde_json::json!(248319)),
            ("/rms_norm_eps", serde_json::json!(0.00001)),
            ("/attention_bias", serde_json::json!(true)),
            (
                "/dflash_config/target_layer_ids",
                serde_json::json!([1, 12, 23, 34, 45]),
            ),
            ("/dflash_config/mask_token_id", serde_json::json!(248070)),
            ("/rope_parameters/rope_theta", serde_json::json!(10000000)),
            ("/sliding_window", serde_json::json!(null)),
            ("/use_sliding_window", serde_json::json!(false)),
            ("/dflash_config/conv_group_size", serde_json::json!(8)),
            ("/dflash_config/conv_kernel_size", serde_json::json!(3)),
            ("/dflash_config/selector_rank", serde_json::json!(128)),
            ("/dflash_config/selector_top_k", serde_json::json!(8)),
        ];
        let mut fixture = Fixture::new();
        let original = fixture.config.clone();
        for (path, invalid) in mutations {
            fixture.config = original.clone();
            *fixture.config.pointer_mut(path).unwrap() = invalid;
            fixture.write();
            fixture.rejects("unsupported");
        }
        fixture.config = original;
        fixture
            .config
            .as_object_mut()
            .unwrap()
            .remove("dflash_config");
        fixture.write();
        fixture.rejects("unsupported Flash draft configuration");
    }

    #[test]
    fn config_size_limit_accepts_exact_boundary_and_rejects_one_more_byte() {
        let fixture = Fixture::new();
        let path = fixture.dir.path().join("config.json");
        let mut bytes = fs::read(&path).unwrap();
        bytes.resize(65_536, b' ');
        fs::write(&path, &bytes).unwrap();
        assert!(inspect(fixture.dir.path(), Kind::Dflash2).is_ok());
        bytes.push(b' ');
        fs::write(&path, &bytes).unwrap();
        fixture.rejects("oversized Flash draft config");
    }

    #[test]
    fn tensor_dtype_shape_name_missing_and_extra_are_rejected() {
        let mut fixture = Fixture::new();
        let original = fixture.entries.clone();
        fixture.entries.get_mut("fc.weight").unwrap()["dtype"] = serde_json::json!("F16");
        fixture.write();
        fixture.rejects("invalid draft tensor fc.weight");
        fixture.entries = original.clone();
        fixture.entries.get_mut("fc.weight").unwrap()["shape"] = serde_json::json!([5120, 6400]);
        fixture.write();
        fixture.rejects("invalid draft tensor fc.weight");
        fixture.entries = original.clone();
        let tensor = fixture.entries.remove("fc.weight").unwrap();
        fixture.entries.insert("foreign.weight".to_owned(), tensor);
        fixture.write();
        fixture.rejects("missing fc.weight");
        fixture.entries = original.clone();
        fixture.entries.remove("norm.weight");
        fixture.repack();
        fixture.write();
        fixture.rejects("unexpected or missing draft tensors");
        fixture.entries = original;
        fixture.entries.insert(
            "extra.weight".to_owned(),
            serde_json::json!({"dtype":"BF16","shape":[1]}),
        );
        fixture.repack();
        fixture.write();
        fixture.rejects("unexpected or missing draft tensors");
    }

    #[test]
    fn invalid_offsets_truncated_files_and_invalid_config_are_rejected() {
        let mut fixture = Fixture::new();
        fixture.entries.get_mut("fc.weight").unwrap()["data_offsets"] =
            serde_json::json!([u64::MAX - 1, u64::MAX]);
        fixture.write();
        fixture.rejects("invalid data range");
        let fixture = Fixture::new();
        let path = fixture.dir.path().join("model.safetensors");
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(4)
            .unwrap();
        fixture.rejects("missing safetensors header length");
        fixture.write();
        let len = path.metadata().unwrap().len();
        File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(len - 1)
            .unwrap();
        fixture.rejects("invalid data range");
        fixture.write();
        fs::write(fixture.dir.path().join("config.json"), b"{").unwrap();
        assert!(inspect(fixture.dir.path(), Kind::Dflash2).is_err());
        fs::remove_file(fixture.dir.path().join("config.json")).unwrap();
        assert!(inspect(fixture.dir.path(), Kind::Dflash2).is_err());
        fs::remove_file(path).unwrap();
        fixture.rejects("reading raw BF16 Flash draft header");
    }

    #[test]
    fn inspected_tensor_handle_preserves_truncation_guard() {
        let fixture = Fixture::new();
        let raw = inspect(fixture.dir.path(), Kind::Dflash2).unwrap();
        File::options()
            .write(true)
            .open(fixture.dir.path().join("model.safetensors"))
            .unwrap()
            .set_len(8)
            .unwrap();
        assert!(raw.tensors["fc.weight"].source().is_err());
    }

    #[test]
    fn kinds_preserve_capture_mask_rope_width_and_dspark_is_separate() {
        assert_eq!(Kind::parse("dflash2").unwrap(), Kind::Dflash2);
        assert_eq!(Kind::parse("dspark").unwrap(), Kind::Dspark);
        assert!(Kind::parse("unknown").is_err());
        assert_eq!(Kind::Dflash2.captures(), [5, 15, 25, 35, 45]);
        assert_eq!(Kind::Dspark.captures(), [1, 12, 23, 34, 45]);
        assert_eq!(Kind::Dflash2.mask(), 248077);
        assert_eq!(Kind::Dspark.mask(), 248070);
        assert_eq!(Kind::Dflash2.rope(), 10_000.0);
        assert_eq!(Kind::Dspark.rope(), 10_000_000.0);
        assert_eq!(Kind::Dflash2.intermediate(), 7680);
        assert_eq!(Kind::Dspark.intermediate(), 9728);
        for proposals in 1..=7 {
            assert_eq!(Kind::Dflash2.rows(proposals).unwrap(), proposals as i32 + 1);
            assert_eq!(Kind::Dspark.rows(proposals).unwrap(), proposals as i32);
        }
        for invalid in [0, 8, usize::MAX] {
            assert!(Kind::Dflash2.rows(invalid).is_err());
            assert!(Kind::Dspark.rows(invalid).is_err());
        }
        let fixture = Fixture::new();
        assert!(inspect(fixture.dir.path(), Kind::Dspark).is_err());
    }

    #[test]
    #[ignore = "requires MLXL3_FLASH_DRAFT_PATH; CPU header-only inspection, no payload reads"]
    fn qualified_local_raw_header_matches_fixture() {
        let path =
            std::env::var_os("MLXL3_FLASH_DRAFT_PATH").expect("MLXL3_FLASH_DRAFT_PATH required");
        let raw = inspect(Path::new(&path), Kind::Dflash2).unwrap();
        assert_eq!(raw.tensors.len(), 81);
        assert_eq!(raw.shapes.len(), 81);
        for &(name, shape) in FLASH_SHAPES {
            assert_eq!(raw.tensors[name].dtype, "BF16");
            assert_eq!(raw.tensors[name].shape, shape);
        }
    }
}
