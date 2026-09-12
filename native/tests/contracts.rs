use mlxl3_native::{checkpoint, registry};
use serde_json::{Value, json};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output, Stdio},
    sync::{Arc, Barrier},
    thread,
};

const MODULE: &str = "model.layers.0.self_attn.q_proj";

fn write_json(path: &Path, value: &Value) {
    fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
}

fn write_safetensors(path: &Path, header: &Value, payload: &[u8]) {
    let mut header = serde_json::to_vec(header).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut file = fs::File::create(path).unwrap();
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    file.write_all(payload).unwrap();
}

fn tensors() -> Value {
    json!({
        "__metadata__": {"format": "pt"},
        format!("{MODULE}.trellis"): {"dtype": "U16", "shape": [1, 1, 64], "data_offsets": [0, 128]},
        format!("{MODULE}.suh"): {"dtype": "F16", "shape": [16], "data_offsets": [128, 160]},
        format!("{MODULE}.svh"): {"dtype": "F16", "shape": [16], "data_offsets": [160, 192]}
    })
}

fn make_checkpoint(root: &Path, name: &str) -> PathBuf {
    let model = root.join(name);
    fs::create_dir_all(&model).unwrap();
    write_json(&model.join("config.json"), &json!({"model_type": "qwen3"}));
    write_json(
        &model.join("quantization_config.json"),
        &json!({
            "quant_method": "exl3", "bits": 4.0,
            "tensor_storage": {
                MODULE: {"quant_format": "exl3", "stored_tensors": {
                    format!("{MODULE}.trellis"): {"shape": [1, 1, 64]},
                    format!("{MODULE}.suh"): {"shape": [16]},
                    format!("{MODULE}.svh"): {"shape": [16]}
                }}
            }
        }),
    );
    write_json(&model.join("tokenizer.json"), &json!({"version": "1.0"}));
    let payload: Vec<u8> = (0..192).map(|n| n as u8).collect();
    write_safetensors(&model.join("model.safetensors"), &tensors(), &payload);
    model
}

fn cli(registry: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_mlxl3-rs"))
        .arg("--registry")
        .arg(registry)
        .args(args)
        .output()
        .unwrap()
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn checkpoint_round_trip_and_lazy_payload_read() {
    let root = tempfile::tempdir().unwrap();
    let model = make_checkpoint(root.path(), "tiny");
    let checkpoint = checkpoint::inspect(&model).unwrap();
    assert_eq!(checkpoint.path, model.canonicalize().unwrap());
    assert_eq!(checkpoint.model_type, "qwen3");
    assert_eq!(checkpoint.bits, Some(4.0));
    assert_eq!(checkpoint.modules, [MODULE]);
    assert_eq!(checkpoint.tensors.len(), 3);
    let expected_size: u64 = fs::read_dir(&model)
        .unwrap()
        .map(|entry| entry.unwrap().metadata().unwrap().len())
        .sum();
    assert_eq!(checkpoint.size_bytes, expected_size);
    let trellis = &checkpoint.tensors[&format!("{MODULE}.trellis")];
    assert_eq!(trellis.read_u16().unwrap()[..3], [256, 770, 1284]);
    assert!(
        checkpoint.tensors[&format!("{MODULE}.suh")]
            .read_u16()
            .is_err()
    );
    fs::OpenOptions::new()
        .write(true)
        .open(&trellis.file)
        .unwrap()
        .set_len(trellis.payload_offset + 1)
        .unwrap();
    assert!(
        trellis
            .read_bytes()
            .unwrap_err()
            .to_string()
            .contains("truncated")
    );
}

#[test]
fn safetensors_accepts_scalars_and_zero_sized_tensors() {
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("scalar.safetensors");
    write_safetensors(
        &file,
        &json!({
            "empty": {"dtype": "U8", "shape": [0, 2], "data_offsets": [0, 0]},
            "scalar": {"dtype": "U16", "shape": [], "data_offsets": [0, 2]}
        }),
        &[42, 0],
    );
    let header = checkpoint::read_header(&file).unwrap();
    assert!(header["empty"].read_bytes().unwrap().is_empty());
    assert_eq!(header["scalar"].read_u16().unwrap(), [42]);
}

#[test]
fn safetensors_rejects_invalid_headers_and_ranges() {
    let root = tempfile::tempdir().unwrap();
    let file = root.path().join("invalid.safetensors");
    for raw in [
        vec![],
        vec![0; 7],
        100_000_001u64.to_le_bytes().to_vec(),
        vec![255; 16],
    ] {
        fs::write(&file, raw).unwrap();
        assert!(checkpoint::read_header(&file).is_err());
    }
    let cases = [
        (
            json!({"t": {"dtype": "F128", "shape": [1], "data_offsets": [0, 1]}}),
            1,
        ),
        (
            json!({"t": {"dtype": "U16", "shape": [2], "data_offsets": [0, 2]}}),
            2,
        ),
        (
            json!({"t": {"dtype": "U8", "shape": [1], "data_offsets": [1, 0]}}),
            1,
        ),
        (
            json!({"t": {"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}}),
            1,
        ),
        (
            json!({"t": {"dtype": "U8", "shape": [1], "data_offsets": [1, 2]}}),
            2,
        ),
        (
            json!({"t": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}),
            2,
        ),
        (
            json!({"a": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}, "b": {"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}}),
            1,
        ),
        (
            json!({"t": {"dtype": "U64", "shape": [u64::MAX, 2], "data_offsets": [0, 0]}}),
            0,
        ),
        (json!({"__metadata__": {"format": 3}}), 0),
    ];
    for (header, size) in cases {
        write_safetensors(&file, &header, &vec![0; size]);
        assert!(
            checkpoint::read_header(&file).is_err(),
            "accepted invalid header: {header}"
        );
    }
}

#[test]
fn checkpoint_rejects_missing_or_inconsistent_files() {
    let root = tempfile::tempdir().unwrap();
    for missing in [
        "config.json",
        "quantization_config.json",
        "tokenizer.json",
        "model.safetensors",
    ] {
        let model = make_checkpoint(root.path(), missing);
        fs::remove_file(model.join(missing)).unwrap();
        assert!(
            checkpoint::inspect(&model).is_err(),
            "accepted missing {missing}"
        );
    }
    let model = make_checkpoint(root.path(), "duplicate-shard");
    fs::copy(
        model.join("model.safetensors"),
        model.join("model-00002.safetensors"),
    )
    .unwrap();
    assert!(
        checkpoint::inspect(&model)
            .unwrap_err()
            .to_string()
            .contains("duplicate tensor")
    );
    let model = make_checkpoint(root.path(), "shape");
    let mut quant: Value =
        serde_json::from_slice(&fs::read(model.join("quantization_config.json")).unwrap()).unwrap();
    quant["tensor_storage"][MODULE]["stored_tensors"][format!("{MODULE}.trellis")]["shape"] =
        json!([1, 2, 64]);
    write_json(&model.join("quantization_config.json"), &quant);
    assert!(
        checkpoint::inspect(&model)
            .unwrap_err()
            .to_string()
            .contains("shape")
    );
}

#[test]
fn checkpoint_supports_legacy_scales_and_validates_shard_map() {
    let root = tempfile::tempdir().unwrap();
    let model = make_checkpoint(root.path(), "legacy");
    let mut header = tensors();
    for (modern, legacy) in [("suh", "su"), ("svh", "sv")] {
        let scale = header
            .as_object_mut()
            .unwrap()
            .remove(&format!("{MODULE}.{modern}"))
            .unwrap();
        header[format!("{MODULE}.{legacy}")] = scale;
    }
    write_safetensors(&model.join("model.safetensors"), &header, &[0; 192]);
    write_json(
        &model.join("quantization_config.json"),
        &json!({
            "quant_method": "exl3", "tensor_storage": {MODULE: {"quant_format": "exl3"}}
        }),
    );
    assert!(checkpoint::inspect(&model).is_ok());
    let index = model.join("model.safetensors.index.json");
    write_json(
        &index,
        &json!({"weight_map": {format!("{MODULE}.trellis"): "model.safetensors"}}),
    );
    assert!(checkpoint::inspect(&model).is_ok());
    for shard in ["../model.safetensors", "model-missing.safetensors"] {
        write_json(
            &index,
            &json!({"weight_map": {format!("{MODULE}.trellis"): shard}}),
        );
        assert!(checkpoint::inspect(&model).is_err());
    }
    write_json(
        &index,
        &json!({"weight_map": {"absent": "model.safetensors"}}),
    );
    assert!(checkpoint::inspect(&model).is_err());
}

#[test]
fn checkpoint_validates_trellis_and_scale_layouts() {
    let root = tempfile::tempdir().unwrap();
    let model = make_checkpoint(root.path(), "layouts");
    write_json(
        &model.join("quantization_config.json"),
        &json!({
            "quant_method": "exl3", "tensor_storage": {MODULE: {"quant_format": "exl3"}}
        }),
    );
    let mut wrong_dtype = tensors();
    let mut signed = tensors();
    signed[format!("{MODULE}.trellis")]["dtype"] = json!("I16");
    write_safetensors(&model.join("model.safetensors"), &signed, &[255; 192]);
    assert!(checkpoint::inspect(&model).is_ok());
    wrong_dtype[format!("{MODULE}.trellis")]["dtype"] = json!("F16");
    let mut wrong_rank = tensors();
    wrong_rank[format!("{MODULE}.trellis")]["shape"] = json!([1, 64]);
    let mut wrong_width = tensors();
    wrong_width[format!("{MODULE}.trellis")]["shape"] = json!([1, 16, 4]);
    let mut wrong_scale = tensors();
    wrong_scale[format!("{MODULE}.suh")]["shape"] = json!([8, 2]);
    for header in [wrong_dtype, wrong_rank, wrong_width, wrong_scale] {
        write_safetensors(&model.join("model.safetensors"), &header, &[0; 192]);
        assert!(checkpoint::read_header(&model.join("model.safetensors")).is_ok());
        assert!(
            checkpoint::inspect(&model).is_err(),
            "accepted invalid EXL3 layout: {header}"
        );
    }
    let mut missing_scale = tensors();
    missing_scale
        .as_object_mut()
        .unwrap()
        .remove(&format!("{MODULE}.svh"));
    write_safetensors(&model.join("model.safetensors"), &missing_scale, &[0; 160]);
    assert!(
        checkpoint::inspect(&model)
            .unwrap_err()
            .to_string()
            .contains("missing scale")
    );
}

#[test]
fn registry_replacement_and_remove_protect_model_files() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("registry/models.json");
    let first = make_checkpoint(root.path(), "first");
    let second = make_checkpoint(root.path(), "second");
    assert!(registry::load(&path).unwrap().is_empty());
    registry::register(&path, "owner/model:4bpw", &first, false).unwrap();
    registry::register(&path, "owner/model:4bpw", &first, false).unwrap();
    let before = fs::read(&path).unwrap();
    assert!(registry::register(&path, "owner/model:4bpw", &second, false).is_err());
    assert_eq!(fs::read(&path).unwrap(), before);
    registry::register(&path, "owner/model:4bpw", &second, true).unwrap();
    assert!(registry::remove(&path, "owner/model:4bpw", Some(&first)).is_err());
    assert_eq!(registry::load(&path).unwrap().len(), 1);
    registry::remove(
        &path,
        "owner/model:4bpw",
        Some(&second.canonicalize().unwrap()),
    )
    .unwrap();
    assert!(registry::load(&path).unwrap().is_empty());
    assert!(first.join("model.safetensors").exists());
    assert!(second.join("model.safetensors").exists());
}

#[test]
fn registry_rejects_invalid_names_and_preserves_corrupt_registry() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("models.json");
    let model = make_checkpoint(root.path(), "tiny");
    for name in [
        "",
        "../escape",
        "/absolute",
        "has space",
        "a\nb",
        "a\\b",
        "modèle",
    ] {
        assert!(registry::register(&path, name, &model, false).is_err());
        assert!(!path.exists());
    }
    for bytes in [
        b"{broken".as_slice(),
        br#"{"version":2,"models":{}}"#.as_slice(),
    ] {
        fs::write(&path, bytes).unwrap();
        assert!(registry::load(&path).is_err());
        assert!(registry::register(&path, "valid", &model, false).is_err());
        assert_eq!(fs::read(&path).unwrap(), bytes);
    }
}

#[test]
fn concurrent_registry_writers_do_not_lose_entries() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("models.json");
    let model = make_checkpoint(root.path(), "tiny");
    let writers = 16;
    let barrier = Arc::new(Barrier::new(writers));
    let jobs: Vec<_> = (0..writers)
        .map(|i| {
            let (path, model, barrier) = (path.clone(), model.clone(), barrier.clone());
            thread::spawn(move || {
                barrier.wait();
                registry::register(&path, &format!("model-{i}"), &model, false).unwrap();
                // A reader racing other writers must always observe complete JSON.
                assert!(!registry::load(&path).unwrap().is_empty());
            })
        })
        .collect();
    for job in jobs {
        job.join().unwrap();
    }
    let entries = registry::load(&path).unwrap();
    assert_eq!(entries.len(), writers);
    for i in 0..writers {
        assert!(entries.contains_key(&format!("model-{i}")));
    }
}

#[test]
fn cli_register_list_inspect_remove_uses_isolated_registry() {
    let root = tempfile::tempdir().unwrap();
    let path = root.path().join("state/models.json");
    let model = make_checkpoint(root.path(), "model with spaces");
    let model_path = model.to_str().unwrap();
    assert_success(&cli(&path, &["add", "tiny", model_path]));
    let output = cli(&path, &["ls", "--json"]);
    assert_success(&output);
    let entries: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(entries.as_array().unwrap().len(), 1);
    assert_eq!(entries[0]["name"], "tiny");
    assert_eq!(entries[0]["format"], "EXL3");
    assert_eq!(entries[0]["modules"], 1);
    let inspect = cli(&path, &["inspect", model_path]);
    assert_success(&inspect);
    let info: Value = serde_json::from_slice(&inspect.stdout).unwrap();
    assert_eq!(info["modules"], json!([MODULE]));
    assert_success(&cli(&path, &["rm", "tiny", "--expected-path", model_path]));
    assert!(registry::load(&path).unwrap().is_empty());
    assert!(model.join("model.safetensors").exists());
    let missing = cli(&path, &["remove", "missing"]);
    assert_eq!(missing.status.code(), Some(2));
    assert!(missing.stdout.is_empty());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("unknown model"));
}

#[test]
fn cli_default_registry_respects_explicit_app_home() {
    let root = tempfile::tempdir().unwrap();
    let app_home = root.path().join("isolated app home");
    let model = make_checkpoint(root.path(), "tiny");
    let output = Command::new(env!("CARGO_BIN_EXE_mlxl3-rs"))
        .env("MLXL3_HOME", &app_home)
        .args(["register", "tiny"])
        .arg(&model)
        .output()
        .unwrap();
    assert_success(&output);
    assert_eq!(
        registry::load(&app_home.join("models.json")).unwrap().len(),
        1
    );
}

#[test]
fn cli_relative_registry_mutations_succeed() {
    let root = tempfile::tempdir().unwrap();
    let model = make_checkpoint(root.path(), "tiny");
    let register = Command::new(env!("CARGO_BIN_EXE_mlxl3-rs"))
        .current_dir(root.path())
        .args(["--registry", "models.json", "register", "tiny"])
        .arg(&model)
        .output()
        .unwrap();
    assert_success(&register);
    let path = root.path().join("models.json");
    assert_eq!(registry::load(&path).unwrap().len(), 1);
    let remove = Command::new(env!("CARGO_BIN_EXE_mlxl3-rs"))
        .current_dir(root.path())
        .args(["--registry", "models.json", "remove", "tiny"])
        .output()
        .unwrap();
    assert_success(&remove);
    assert!(registry::load(&path).unwrap().is_empty());
    assert!(model.join("model.safetensors").exists());
}

#[test]
fn cli_empty_app_home_uses_user_config_directory() {
    let root = tempfile::tempdir().unwrap();
    let model = make_checkpoint(root.path(), "tiny");
    let output = Command::new(env!("CARGO_BIN_EXE_mlxl3-rs"))
        .current_dir(root.path())
        .env("HOME", root.path())
        .env("MLXL3_HOME", "")
        .args(["register", "tiny"])
        .arg(&model)
        .output()
        .unwrap();
    assert_success(&output);
    assert_eq!(
        registry::load(&root.path().join(".config/mlxl3/models.json"))
            .unwrap()
            .len(),
        1
    );
    assert!(!root.path().join("models.json").exists());
}

#[test]
fn cli_codec_recovers_after_invalid_request() {
    let root = tempfile::tempdir().unwrap();
    let mut child = Command::new(env!("CARGO_BIN_EXE_mlxl3-rs"))
        .arg("--registry")
        .arg(root.path().join("models.json"))
        .arg("codec")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"not json\n{\"op\":\"unknown\"}\n{\"op\":\"permutation\"}\n")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert_success(&output);
    let responses: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(responses.len(), 3);
    assert!(responses[0]["error"].is_string());
    assert!(responses[1]["error"].is_string());
    assert!(responses[2]["data"].is_array());
    assert!(!root.path().join("models.json").exists());
}
