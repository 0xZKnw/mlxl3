use super::*;

fn tempdir() -> tempfile::TempDir {
    // Match production's canonical storage root, including macOS /var symlinks.
    tempfile::tempdir_in(std::env::temp_dir().canonicalize().unwrap()).unwrap()
}

fn fixture(root: &Path) -> (PathBuf, PathBuf, PendingDownload, PathBuf) {
    let stage = root.join(".downloads/0123456789ab");
    let assembled = stage.join("model");
    fs::create_dir_all(&assembled).unwrap();
    let module = "model.layers.0.self_attn.q_proj.weight";
    write_json(
        &assembled.join("config.json"),
        &serde_json::json!({"model_type":"qwen3"}),
    )
    .unwrap();
    write_json(
        &assembled.join("tokenizer.json"),
        &serde_json::json!({"version":"1.0"}),
    )
    .unwrap();
    write_json(
        &assembled.join("quantization_config.json"),
        &serde_json::json!({
            "quant_method":"exl3", "bits":4.0, "tensor_storage": {
                module: {"quant_format":"exl3", "stored_tensors": {
                    format!("{module}.trellis"): {"shape":[1,1,64]},
                    format!("{module}.suh"): {"shape":[16]},
                    format!("{module}.svh"): {"shape":[16]}
                }}
            }
        }),
    )
    .unwrap();
    let header = serde_json::to_vec(&serde_json::json!({
        "__metadata__":{"format":"pt"},
        format!("{module}.trellis"): {"dtype":"U16","shape":[1,1,64],"data_offsets":[0,128]},
        format!("{module}.suh"): {"dtype":"F16","shape":[16],"data_offsets":[128,160]},
        format!("{module}.svh"): {"dtype":"F16","shape":[16],"data_offsets":[160,192]}
    }))
    .unwrap();
    let mut weights = File::create(assembled.join("model.safetensors")).unwrap();
    weights
        .write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    weights.write_all(&header).unwrap();
    weights.write_all(&[0; 192]).unwrap();
    let request = PendingDownload {
        id: "0123456789ab".into(),
        repo: "test/model".into(),
        commit: "a".repeat(40),
        folder: String::new(),
        name: "model".into(),
        destination: root.join("installed").display().to_string(),
        size_bytes: 192,
        retained_bytes: 0,
    };
    write_json(&stage.join("request.json"), &request).unwrap();
    let registry = root.join("models.json");
    (stage, assembled, request, registry)
}

fn publish(assembled: &Path, request: &PendingDownload) {
    publish_checkpoint(
        assembled,
        Path::new(&request.destination),
        &AtomicBool::new(false),
    )
    .unwrap();
}

#[test]
fn crash_boundary_child() {
    let Ok(root) = std::env::var("MLXL3_TRANSACTION_TEST_ROOT") else {
        return;
    };
    let boundary = std::env::var("MLXL3_TRANSACTION_TEST_BOUNDARY").unwrap();
    if boundary == "resume" {
        let mut phases = Vec::new();
        let (name, entry) = resume_with_phase(
            "0123456789ab",
            &Path::new(&root).join("models.json"),
            |_, _, phase| phases.push(phase.to_owned()),
        )
        .unwrap();
        assert_eq!(name, "model");
        assert!(entry.installation_id.is_some());
        assert_eq!(phases, ["install"]);
        return;
    }
    let (stage, assembled, request, registry) = fixture(Path::new(&root));
    let transaction = InstallTransaction::prepare(&stage, &assembled, &request, &registry).unwrap();
    if boundary == "intent" {
        std::process::exit(86);
    }
    publish(&assembled, &request);
    if boundary == "publication" {
        std::process::exit(86);
    }
    transaction.register(&registry, &request).unwrap();
    // Exit without destructors, completion marker, or staging cleanup.
    std::process::exit(86);
}

#[test]
fn restart_reconciles_every_commit_boundary_without_network() {
    for boundary in ["intent", "publication", "registry"] {
        let root = tempdir();
        let status = wait_child(
            std::process::Command::new(std::env::current_exe().unwrap())
                .arg("--exact")
                .arg("hub::install_transaction::tests::crash_boundary_child")
                .env("MLXL3_TRANSACTION_TEST_ROOT", root.path())
                .env("MLXL3_TRANSACTION_TEST_BOUNDARY", boundary)
                .spawn()
                .unwrap(),
        );
        assert_eq!(status.code(), Some(86));
        let stage = root.path().join(".downloads/0123456789ab");
        let request: PendingDownload =
            serde_json::from_reader(File::open(stage.join("request.json")).unwrap()).unwrap();
        let registry = root.path().join("models.json");
        assert_eq!(completed_download(&stage, &request), boundary == "registry");
        if boundary == "intent" {
            assert!(
                recover_publication(&stage, &request, &registry)
                    .unwrap()
                    .is_none()
            );
            publish(&stage.join("model"), &request);
        }
        let first = recover_publication(&stage, &request, &registry)
            .unwrap()
            .unwrap();
        let retry = recover_publication(&stage, &request, &registry)
            .unwrap()
            .unwrap();
        assert_eq!(first.added_at, retry.added_at);
        assert_eq!(first.installation_id, retry.installation_id);
        assert!(completed_download(&stage, &request));
        assert!(!stage.join("completed.json").exists());
        assert!(
            pending_downloads_in(&root.path().join(".downloads"))
                .unwrap()
                .is_empty()
        );
        // A marker cannot be written here; stage retention must still not offer resume.
        fs::create_dir(stage.join("completed.json")).unwrap();
        assert!(write_json(&stage.join("completed.json"), &request).is_err());
        cleanup_download_stage(&stage, |_| Err(std::io::ErrorKind::PermissionDenied.into()));
        assert!(stage.is_dir());
        assert!(completed_download(&stage, &request));
        assert!(
            pending_downloads_in(&root.path().join(".downloads"))
                .unwrap()
                .is_empty()
        );
    }
}

#[test]
fn failed_registration_retains_publication_and_retry_finishes() {
    let root = tempdir();
    let (stage, assembled, request, registry) = fixture(root.path());
    let transaction = InstallTransaction::prepare(&stage, &assembled, &request, &registry).unwrap();
    publish(&assembled, &request);
    fs::write(&registry, b"broken registry").unwrap();
    assert!(transaction.register(&registry, &request).is_err());
    assert!(Path::new(&request.destination).is_dir());
    assert!(!completed_download(&stage, &request));
    fs::remove_file(&registry).unwrap();
    assert!(
        recover_publication(&stage, &request, &registry)
            .unwrap()
            .is_some()
    );
    assert!(completed_download(&stage, &request));
}

#[test]
fn reinstall_requires_new_registry_identity_and_rejects_foreign_receipt() {
    let root = tempdir();
    let (stage, assembled, request, registry) = fixture(root.path());
    let first = InstallTransaction::prepare(&stage, &assembled, &request, &registry).unwrap();
    publish(&assembled, &request);
    first.register(&registry, &request).unwrap();
    fs::remove_dir_all(&request.destination).unwrap();
    let (_, assembled, _, _) = fixture(root.path());
    let second = InstallTransaction::prepare(&stage, &assembled, &request, &registry).unwrap();
    assert_ne!(first.attempt, second.attempt);
    publish(&assembled, &request);
    assert!(!completed_download(&stage, &request)); // stale same-path registry is insufficient
    assert!(
        recover_publication(&stage, &request, &registry)
            .unwrap()
            .is_some()
    );
    assert!(completed_download(&stage, &request));
    write_json(&Path::new(&request.destination).join(RECEIPT), &first).unwrap();
    assert!(!completed_download(&stage, &request));
    assert!(recover_publication(&stage, &request, &registry).is_err());
    assert!(Path::new(&request.destination).is_dir());
}

#[test]
fn registry_destination_and_request_mismatch_never_hide_or_delete_files() {
    let root = tempdir();
    let (stage, assembled, request, registry) = fixture(root.path());
    InstallTransaction::prepare(&stage, &assembled, &request, &registry).unwrap();
    publish(&assembled, &request);
    assert!(recover_publication(&stage, &request, &root.path().join("other.json")).is_err());
    let mut foreign = request.clone();
    foreign.commit = "b".repeat(40);
    assert!(!completed_download(&stage, &foreign));
    assert!(recover_publication(&stage, &foreign, &registry).is_err());
    assert!(
        Path::new(&request.destination)
            .join("model.safetensors")
            .is_file()
    );
}

#[test]
fn directory_fsync_failures_after_rename_are_reconciled_without_rollback() {
    for registry_failure in [false, true] {
        let root = tempdir();
        let (stage, assembled, request, registry) = fixture(root.path());
        let transaction =
            InstallTransaction::prepare(&stage, &assembled, &request, &registry).unwrap();
        if registry_failure {
            publish(&assembled, &request);
            crate::registry::FAIL_REGISTRY_SYNC.set(true);
            let result = transaction.register(&registry, &request);
            crate::registry::FAIL_REGISTRY_SYNC.set(false);
            assert!(result.unwrap_err().to_string().contains("fsync failure"));
            assert!(completed_download(&stage, &request));
        } else {
            let result = publish_checkpoint_with_sync(
                &assembled,
                Path::new(&request.destination),
                &AtomicBool::new(false),
                |_| anyhow::bail!("injected publication directory fsync failure"),
            );
            assert!(result.is_err());
            assert!(!completed_download(&stage, &request));
        }
        assert!(Path::new(&request.destination).is_dir());
        assert!(!assembled.exists());
        assert!(
            recover_publication(&stage, &request, &registry)
                .unwrap()
                .is_some()
        );
        assert!(completed_download(&stage, &request));
    }
}

#[test]
fn production_resume_completes_published_and_committed_crashes() {
    for boundary in ["publication", "registry"] {
        let root = tempdir();
        let run = |boundary: &str| {
            wait_child(
                std::process::Command::new(std::env::current_exe().unwrap())
                    .arg("--exact")
                    .arg("hub::install_transaction::tests::crash_boundary_child")
                    .env("MLXL3_TRANSACTION_TEST_ROOT", root.path())
                    .env("MLXL3_MODELS_DIR", root.path())
                    .env("MLXL3_TRANSACTION_TEST_BOUNDARY", boundary)
                    .spawn()
                    .unwrap(),
            )
        };
        assert_eq!(run(boundary).code(), Some(86));
        assert!(run("resume").success());
        assert!(!root.path().join(".downloads/0123456789ab").exists());
        assert!(root.path().join("installed/model.safetensors").is_file());
        assert_eq!(
            crate::registry::load(&root.path().join("models.json"))
                .unwrap()
                .len(),
            1
        );
    }
}

fn wait_child(mut child: std::process::Child) -> std::process::ExitStatus {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status,
            Ok(None) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            other => {
                let _ = child.kill();
                let _ = child.wait();
                panic!("crash/restart child did not finish: {other:?}");
            }
        }
    }
}
