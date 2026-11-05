//! Cross-process oracle: immutable compiler and IO switches never share the control process.
use super::{
    ar20_kernel_tests::compiled_bits,
    ar30_kernel_tests::compiled_cooperative,
    ssd_direct_tests::{check_initial, check_tensors, digest},
    *,
};
use half::f16;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs, io::Write, path::PathBuf};

#[derive(Serialize, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
struct TensorHash {
    name: String,
    shape: Vec<u32>,
    bytes: usize,
    sha256: String,
}
#[derive(Serialize, Deserialize, PartialEq, Debug)]
#[serde(deny_unknown_fields)]
struct RowHash {
    token: u32,
    offset: u32,
    logits_words: usize,
    logits: String,
    hidden_bytes: usize,
    hidden: String,
    next_logits_words: usize,
    next_logits: String,
    states: Vec<TensorHash>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Oracle {
    schema: u32,
    status: String,
    parity: bool,
    reference_kernel_bits: u32,
    model_path: PathBuf,
    state_words: usize,
    logit_words: usize,
    hidden_bytes: usize,
    cases: Vec<RowHash>,
}

fn valid_sha(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit())
}
fn validate(oracle: &Oracle) -> Result<()> {
    ensure!(
        oracle.schema == 1
            && oracle.status == "reference_captured"
            && !oracle.parity
            && oracle.reference_kernel_bits == 0
    );
    ensure!(
        oracle.cases.len() == 2
            && oracle.state_words == 57_606_144
            && oracle.logit_words == 993_280
            && oracle.hidden_bytes == 81920
    );
    for (i, row) in oracle.cases.iter().enumerate() {
        ensure!(
            row.token == [248044, 1][i]
                && row.offset == i as u32 + 1
                && row.logits_words == 248320
                && row.next_logits_words == 248320
                && row.hidden_bytes == 40960
        );
        ensure!(
            valid_sha(&row.logits)
                && valid_sha(&row.next_logits)
                && valid_sha(&row.hidden)
                && row.states.len() == 84
        );
        let mut names = BTreeSet::new();
        let mut words = 0usize;
        for t in &row.states {
            let count = t
                .shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
                .context("state shape overflow")?;
            ensure!(
                !t.shape.is_empty()
                    && count > 0
                    && count.checked_mul(4) == Some(t.bytes)
                    && valid_sha(&t.sha256)
                    && names.insert(t.name.as_str())
                    && !t.name.is_empty()
            );
            words = words.checked_add(count).context("state count overflow")?;
        }
        ensure!(words == 28_803_072);
    }
    Ok(())
}
fn logits_hash(words: &[u16]) -> Result<String> {
    ensure!(words.len() == 248320 && words.iter().all(|&v| f16::from_bits(v).is_finite()));
    digest(
        &words
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}
fn actual_paths(gpu: &Metal, model: &FlashNext) -> Result<serde_json::Value> {
    let readers = gpu.readers()?;
    let slru = model.slru_counts()?;
    let readahead = readers.readahead_counts();
    Ok(
        serde_json::json!({"kernel_bits":gpu.ar20_kernel_bits,"ar30_cooperative":gpu.ar30_cooperative,"ar30_cooperative_dispatches":gpu.ar30_cooperative_uses.each_ref().map(std::cell::Cell::get),"head_gates":gpu.gdn_head_gate_uses.get(),"qsa_chains":gpu.qsa_chain_uses.get(),"qsa_prefetch":gpu.qsa_prefetch_uses.get(),"resident_bundle":gpu.gdn_resident_bundle_uses.get(),"finish_mix":gpu.moe_finish_mix_uses.get(),"spin_wait":gpu.spin_wait_stats.each_ref().map(std::cell::Cell::get),"readahead_fds":readahead[0],"readahead_records":readahead[1],"slru_layers":slru.0,"slru_reservations":slru.1,"cross_layer_triples":gpu.cross_layer_triples.get(),"moe_deferred":gpu.moe_deferred_uses.get(),"expert_axis_qmv_dispatches":gpu.expert_axis_dispatches.get(),"reused_destinations":readers.reused_destinations.get(),"split_records":readers.split_records.get(),"split_pieces":readers.split_pieces.get(),"device_greedy":model.device_greedy_counts(),"router_pair_uses":gpu.router_pair.uses.get(),"readers_in_flight":readers.in_flight()}),
    )
}

fn run_capture() -> Result<()> {
    let capture = match std::env::var("MLXL3_AR20_CAPTURE")?.as_str() {
        "1" => true,
        "0" => false,
        _ => anyhow::bail!("CAPTURE must be exactly0or1"),
    };
    let path =
        PathBuf::from(std::env::var_os("MLXL3_AR20_ORACLE").context("oracle path required")?);
    let model_path =
        PathBuf::from(std::env::var_os("MLXL3_FLASH_MODEL").context("model path required")?);
    let reference = if capture {
        ensure!(!path.exists(), "refusing to overwrite an oracle");
        None
    } else {
        ensure!(
            fs::metadata(&path)?.len() <= 1_048_576,
            "oracle JSON exceeds its bound"
        );
        let bytes = fs::read(&path)?;
        let oracle: Oracle = serde_json::from_slice(&bytes)?;
        validate(&oracle)?;
        ensure!(oracle.model_path == model_path);
        Some((oracle, digest(&bytes)?))
    };
    let checkpoint = crate::checkpoint::inspect(&model_path)?;
    let gpu = Metal::new()?;
    ensure!(
        std::env::var("MLXL3_NATIVE_DEVICE_GREEDY").as_deref() != Ok("1")
            || gpu.device_greedy_available(),
        "requested C03 path is unavailable"
    );
    let bits = compiled_bits(&gpu)?;
    let cooperative = compiled_cooperative(&gpu)?;
    if capture {
        ensure!(
            bits == 0 && !cooperative && !gpu.device_greedy_available() && !gpu.router_pair.enabled.get(),
            "reference compiler options must be OFF"
        );
        ensure!(
            !gpu.gdn_head_gates.get()
                && !gpu.qsa_chains.get()
                && !gpu.qsa_prefetch.get()
                && !gpu.gdn_resident_bundle.get()
                && !gpu.moe_finish_mix.get()
                && !gpu.spin_wait_enabled,
            "reference runtime options must be OFF"
        );
    }
    ensure!(
        gpu.cross_layer.get()
            && gpu.moe_deferred.get()
            && gpu.hc_write_norm.get()
            && gpu.routed_nt2_gu.get()
            && gpu.routed_nt2_down.get()
            && gpu.expert_ar_enabled.iter().all(|v| v.get())
            && gpu.gdn_whole.get()
            && gpu.gdn_joint_ab.get()
            && gpu.qsa_prepare.get()
            && gpu.qsa_gate_fused.get(),
        "accepted ALL12 baseline required"
    );
    let mut model = FlashNext::load(&gpu, &checkpoint, 4096)?;
    ensure!(model.prefetcher.is_none());
    if capture {
        ensure!(
            model.slru_counts()? == (0, 0) && gpu.readers()?.readahead_counts() == [0; 2],
            "reference IO/cache options must be OFF"
        );
    }
    let empty = model.snapshot();
    let (initial, states) = model.forward_trace(248044)?;
    check_initial(&initial, &states)?;
    check_tensors(&states)?;
    drop(states);
    model.restore(&empty)?;
    let mut cases = Vec::new();
    let mut state_words = 0;
    let mut paths = Vec::new();
    for token in [248044, 1] {
        let before = actual_paths(&gpu, &model)?;
        let router_before = gpu.router_pair.uses.get();
        let cooperative_before = gpu
            .ar30_cooperative_uses
            .each_ref()
            .map(std::cell::Cell::get);
        let logits = if gpu.device_greedy_available() {
            let counts = model.device_greedy_counts();
            let (selected, logits) = model.forward_greedy_captured(token)?;
            let canonical = crate::speculative::greedy_log_prob_id(&model.log_probs(&logits)?)
                .context("reference normalized distribution is invalid")?;
            ensure!(
                selected == canonical
                    && model.device_greedy_counts() == [counts[0] + 1, counts[1] + 1]
            );
            logits
        } else {
            model.forward(token)?
        };
        let logits = logits_hash(&logits)?;
        let hidden = model
            .last_hidden()
            .context("missing hidden")?
            .export_bytes(&gpu)?;
        ensure!(
            hidden.len() == 40960
                && hidden
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|v| f32::from_le_bytes(*v).is_finite())
        );
        let control = model.snapshot();
        let offset = model.offset();
        let after = actual_paths(&gpu, &model)?;
        ensure!(!gpu.router_pair.enabled.get() || gpu.router_pair.uses.get() > router_before, "paired router was not encoded on the real model path");
        if cooperative {
            ensure!(
                gpu.ar30_cooperative_uses
                    .each_ref()
                    .map(std::cell::Cell::get)
                    .into_iter()
                    .zip(cooperative_before)
                    .all(|(after, before)| after > before),
                "cooperative resident/routed paths were not both executed"
            );
        }
        let (next, states) = model.forward_trace(11)?;
        let next_logits = logits_hash(&next)?;
        state_words += check_tensors(&states)?;
        let states = states
            .into_iter()
            .map(|t| {
                Ok(TensorHash {
                    name: t.name,
                    shape: t.shape,
                    bytes: t.data.len(),
                    sha256: digest(&t.data)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        cases.push(RowHash {
            token,
            offset,
            logits_words: 248320,
            logits,
            hidden_bytes: 40960,
            hidden: digest(&hidden)?,
            next_logits_words: 248320,
            next_logits,
            states,
        });
        paths.push(serde_json::json!({"token":token,"before":before,"after":after}));
        model.restore(&control)?;
    }
    ensure!(state_words == 57_606_144 && model.offset() == 2 && gpu.readers()?.in_flight() == 0);
    if gpu.device_greedy_available() {
        let retained = model.snapshot();
        gpu.device_greedy_reject_result.set(true);
        let rejected = model.forward_greedy(11);
        gpu.device_greedy_reject_result.set(false);
        ensure!(
            rejected.is_err() && model.offset() == 2,
            "failed scalar selection advanced state"
        );
        let (next, states) = model.forward_trace(11)?;
        ensure!(logits_hash(&next)? == cases[1].next_logits);
        check_tensors(&states)?;
        let actual = states
            .iter()
            .map(|state| {
                Ok(TensorHash {
                    name: state.name.clone(),
                    shape: state.shape.clone(),
                    bytes: state.data.len(),
                    sha256: digest(&state.data)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            actual == cases[1].states,
            "scalar failure rollback changed recurrent state"
        );
        model.restore(&retained)?;
        ensure!(model.offset() == 2 && model.device_greedy_counts() == [2, 2]);
    }
    if let Some((reference, oracle_sha)) = reference {
        ensure!(
            cases == reference.cases,
            "cross-process full-model fingerprint differs"
        );
        ensure!(
            digest(&fs::read(&path)?)? == oracle_sha,
            "oracle changed during comparison"
        );
        eprintln!(
            "AR20_MODEL_COMPARE {}",
            serde_json::json!({"parity":true,"status":"validated","case_count":2,"compared_state_words":state_words,"compared_logits":993280,"compared_hidden_bytes":81920,"initial_archive_states":84,"kernel_bits":bits,"compiled_cooperative":cooperative,"router_pair_uses":gpu.router_pair.uses.get(),"device_greedy_counts":model.device_greedy_counts(),"device_greedy_rollback":gpu.device_greedy_available(),"actual_paths":paths,"oracle_sha256":oracle_sha,"all_readers_joined":true})
        );
    } else {
        let oracle = Oracle {
            schema: 1,
            status: "reference_captured".into(),
            parity: false,
            reference_kernel_bits: bits,
            model_path,
            state_words,
            logit_words: 993280,
            hidden_bytes: 81920,
            cases,
        };
        validate(&oracle)?;
        let parent = path.parent().context("oracle parent required")?;
        fs::create_dir_all(parent)?;
        let mut file = tempfile::NamedTempFile::new_in(parent)?;
        serde_json::to_writer(file.as_file_mut(), &oracle)?;
        file.as_file_mut().write_all(b"\n")?;
        file.as_file().sync_all()?;
        file.persist_noclobber(&path).map_err(|e| e.error)?;
        eprintln!(
            "AR20_MODEL_CAPTURE {}",
            serde_json::json!({"parity":false,"status":"reference_captured","case_count":2,"compared_state_words":state_words,"compared_logits":993280,"compared_hidden_bytes":81920,"initial_archive_states":84,"kernel_bits":bits,"actual_paths":paths,"oracle_sha256":digest(&fs::read(&path)?)?,"all_readers_joined":true})
        );
    }
    Ok(())
}
#[test]
#[ignore = "physical AR20 independent-process full-model oracle; exclusive GPU, root controlled"]
fn physical_ar20_model_matches_independent_process_oracle() -> Result<()> {
    let result = run_capture();
    if let Err(error) = &result {
        eprintln!(
            "AR20_MODEL_FAILURE {}",
            serde_json::json!({"parity":false,"status":"failed","error":format!("{error:#}")})
        );
    }
    result
}

#[test]
fn cross_process_oracle_refuses_incomplete_or_ambiguous_metadata() -> Result<()> {
    let tensors = || {
        (0..84)
            .map(|i| TensorHash {
                name: format!("state{i}"),
                shape: vec![if i == 83 { 28_803_072 - 83 } else { 1 }],
                bytes: if i == 83 { (28_803_072 - 83) * 4 } else { 4 },
                sha256: "0".repeat(64),
            })
            .collect()
    };
    let rows = [248044, 1]
        .into_iter()
        .enumerate()
        .map(|(i, token)| RowHash {
            token,
            offset: i as u32 + 1,
            logits_words: 248320,
            logits: "0".repeat(64),
            hidden_bytes: 40960,
            hidden: "0".repeat(64),
            next_logits_words: 248320,
            next_logits: "0".repeat(64),
            states: tensors(),
        })
        .collect();
    let mut oracle = Oracle {
        schema: 1,
        status: "reference_captured".into(),
        parity: false,
        reference_kernel_bits: 0,
        model_path: PathBuf::from("fixture-only"),
        state_words: 57_606_144,
        logit_words: 993280,
        hidden_bytes: 81920,
        cases: rows,
    };
    validate(&oracle)?;
    oracle.parity = true;
    assert!(validate(&oracle).is_err());
    oracle.parity = false;
    oracle.cases[0].states[1].name = "state0".into();
    assert!(validate(&oracle).is_err());
    oracle.cases[0].states[1].name = "state1".into();
    oracle.cases[0].hidden.clear();
    assert!(validate(&oracle).is_err());
    oracle.cases[0].hidden = "0".repeat(64);
    oracle.cases.pop();
    assert!(validate(&oracle).is_err());
    assert!(serde_json::from_slice::<Oracle>(b"{\"schema\":1,\"schema\":1}").is_err());
    Ok(())
}
