use super::*;
use serde_json::{Value, json};
use std::{fs, os::unix::fs::symlink};

fn target(config: &Value) -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    fs::write(
        directory.path().join("config.json"),
        serde_json::to_vec(config).unwrap(),
    )
    .unwrap();
    directory
}

fn flash() -> Value {
    json!({"model_type":"qwen4_exp","text_config":{"hidden_size":2560,"num_hidden_layers":48,"vocab_size":248320}})
}

#[test]
fn flash_target_selection_uses_geometry_and_pins_raw_abi() {
    let directory = target(&flash());
    let kind = ManagedDraft::from_target(directory.path()).unwrap();
    assert_eq!(kind, ManagedDraft::FlashNext);
    assert_eq!(kind.name(), "Qwen3.8-Flash-Next-DFlash2-a22a1744");
    assert_eq!(kind.repository(), "incoai/Qwen3.8-Flash-Next-DFlash2");
    assert_eq!(kind.revision(), "a22a1744e85c17364dcc54a6a420af7135d4c7f6");
    for name in ["model.safetensors", "model.bin", "quantization_config.json"] {
        assert!(
            !directory.path().join(name).exists(),
            "target selection must not need weights"
        );
    }
    for field in ["hidden_size", "num_hidden_layers", "vocab_size"] {
        for invalid in [
            json!(null),
            json!(false),
            json!(0),
            json!(-1),
            json!("2560"),
            json!(u64::MAX),
            json!(2560.5),
        ] {
            let mut config = flash();
            config["text_config"][field] = invalid;
            let directory = target(&config);
            assert!(
                ManagedDraft::from_target(directory.path()).is_err(),
                "bad field {field}"
            );
        }
    }
    for config in [
        json!({}),
        json!([]),
        json!({"model_type":"qwen4_exp","hidden_size":2560,"num_hidden_layers":48,"vocab_size":248320}),
        json!({"model_type":"qwen3","text_config":flash()["text_config"]}),
    ] {
        assert!(ManagedDraft::from_target(target(&config).path()).is_err());
    }
}

#[test]
fn packed_target_selection_preserves_both_legacy_families() {
    let dense = json!({"model_type":"qwen3_5","text_config":{"hidden_size":5120,"num_hidden_layers":64,"num_attention_heads":24,"num_key_value_heads":4,"head_dim":256,"vocab_size":248320,"intermediate_size":17408,"rms_norm_eps":1e-6,"partial_rotary_factor":0.25,"rope_parameters":{"rope_theta":10000000}}});
    let moe = json!({"model_type":"qwen3_5_moe","text_config":{"hidden_size":2048,"num_hidden_layers":40,"num_attention_heads":16,"num_key_value_heads":2,"head_dim":256,"vocab_size":248320,"num_experts":256,"num_experts_per_tok":8,"moe_intermediate_size":512,"shared_expert_intermediate_size":512,"rms_norm_eps":1e-6,"partial_rotary_factor":0.25,"rope_parameters":{"rope_theta":10000000}}});
    for (config, family) in [(dense, Family::Qwen38Dense), (moe, Family::Qwen36Moe)] {
        let directory = target(&config);
        let selected = ManagedDraft::from_target(directory.path()).unwrap();
        assert_eq!(selected, ManagedDraft::Packed(family));
        assert_eq!(selected.name(), family.name());
        assert_eq!(selected.repository(), family.repository());
        assert_eq!(selected.revision(), family.revision());
        let mut invalid = config;
        invalid["text_config"]["mtp_use_dedicated_embeddings"] = json!(true);
        assert!(ManagedDraft::from_target(target(&invalid).path()).is_err());
    }
}

#[test]
fn target_config_limit_is_exact_and_missing_or_malformed_config_fails() {
    let directory = target(&flash());
    let path = directory.path().join("config.json");
    let mut data = fs::read(&path).unwrap();
    data.resize(65536, b' ');
    fs::write(&path, &data).unwrap();
    assert_eq!(
        ManagedDraft::from_target(directory.path()).unwrap(),
        ManagedDraft::FlashNext
    );
    data.push(b' ');
    fs::write(&path, &data).unwrap();
    assert!(
        ManagedDraft::from_target(directory.path())
            .unwrap_err()
            .to_string()
            .contains("oversized")
    );
    fs::write(&path, b"{").unwrap();
    assert!(ManagedDraft::from_target(directory.path()).is_err());
    fs::remove_file(path).unwrap();
    assert!(ManagedDraft::from_target(directory.path()).is_err());
}

#[test]
fn flash_inspection_rejects_symlink_files_and_never_accepts_packed_abi() {
    for name in ["config.json", "model.safetensors"] {
        let directory = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("sentinel"), b"preserve").unwrap();
        fs::write(directory.path().join("config.json"), b"{}").unwrap();
        fs::write(directory.path().join("model.safetensors"), b"not weights").unwrap();
        fs::remove_file(directory.path().join(name)).unwrap();
        symlink(outside.path().join("sentinel"), directory.path().join(name)).unwrap();
        let error = ManagedDraft::FlashNext
            .inspect(directory.path())
            .unwrap_err();
        assert!(error.to_string().contains("regular file"));
        assert_eq!(
            fs::read(outside.path().join("sentinel")).unwrap(),
            b"preserve"
        );
    }
    let packed = tempfile::tempdir().unwrap();
    fs::write(packed.path().join("model.bin"), b"MDFD0004").unwrap();
    assert!(ManagedDraft::FlashNext.inspect(packed.path()).is_err());
    let raw = target(&flash());
    fs::write(raw.path().join("model.safetensors"), b"not weights").unwrap();
    assert!(
        ManagedDraft::Packed(Family::Qwen36Moe)
            .inspect(raw.path())
            .is_err()
    );
}
