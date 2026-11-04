use super::super::{
    FlashNext, MatrixView, Metal,
    ssd_direct_tests::{check_initial, check_tensors},
};
use super::{DeferredHcWrite, FlashGdnLayer};
use crate::{checkpoint, tensor_layout::MatrixLayout};
use anyhow::{Context, Result, ensure};
use half::f16;

fn finite_logits(logits: &[u16]) -> bool {
    logits.len() == 248320 && logits.iter().all(|&v| f16::from_bits(v).is_finite())
}
fn view(gpu: &Metal, values: &[f32]) -> Result<MatrixView> {
    gpu.f32_storage(values)?.matrix(MatrixLayout {
        shape: [1, values.len().try_into()?],
        strides: [0, 1],
        offset: 0,
    })
}

#[test]
#[ignore = "physical deferred write errors and full-state cross-layer parity; run alone"]
fn physical_cross_layer_preserves_states_rollbacks_and_actual_boundaries() -> Result<()> {
    let c = checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    ensure!(
        gpu.cross_layer.get() && gpu.hc_gdn_chain.get(),
        "qualified M5 and CROSS_LAYER=1 required"
    );
    {
        let mut layer =
            FlashGdnLayer::load_body(&gpu, &c, "model.language_model.layers.0", 1e-6, 10)?;
        let input = view(&gpu, &vec![0.25; 10240])?;
        let branch = view(&gpu, &vec![0.125; 2560])?;
        let gate = view(&gpu, &[0.5; 4])?;
        let written = gpu.command(|cmd| gpu.encode_hc_write(cmd, &input, &branch, &gate))?;
        let before = layer.snapshot();
        let expected = layer.forward(&written)?.read_output()?;
        layer.restore(&before)?;
        let commits = gpu.helper_commits.get();
        let bad = DeferredHcWrite {
            streams: input.clone(),
            branch: branch.clone(),
            gate: view(&gpu, &[0.5; 3])?,
        };
        ensure!(layer.forward_pending(bad, false).is_err());
        ensure!(gpu.helper_commits.get() == commits && layer.recurrent().is_none());
        let good = DeferredHcWrite {
            streams: input,
            branch,
            gate,
        };
        let actual = layer
            .forward_pending(good, false)?
            .complete(&gpu)?
            .read_output()?;
        ensure!(
            actual == expected
                && actual.len() == 10240
                && actual.iter().all(|&v| f32::from_bits(v).is_finite())
        );
        ensure!(gpu.readers()?.in_flight() == 0);
    }
    let mut model = FlashNext::load(&gpu, &c, 4096)?;
    ensure!(model.prefetcher.is_none());
    gpu.cross_layer.set(false);
    let empty = model.snapshot();
    let (initial, states) = model.forward_trace(248044)?;
    check_initial(&initial, &states)?;
    check_tensors(&states)?;
    drop(states);
    model.restore(&empty)?;
    let mut cases = Vec::new();
    let mut compared = 0;
    for token in [248044, 1, 42, 7] {
        let saved = model.snapshot();
        gpu.cross_layer.set(false);
        ensure!(finite_logits(&model.forward(19)?));
        model.restore(&saved)?;
        let control_commits = gpu.helper_commits.get();
        let expected = model.forward(token)?;
        let control_commits = gpu.helper_commits.get() - control_commits;
        ensure!(finite_logits(&expected));
        let hidden = model
            .last_hidden()
            .context("missing reference hidden")?
            .export_bytes(&gpu)?;
        ensure!(
            hidden.len() == 40960
                && hidden
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|v| f32::from_le_bytes(*v).is_finite())
        );
        let (expected_next, expected_states) = model.forward_trace(11)?;
        check_tensors(&expected_states)?;
        model.restore(&saved)?;
        gpu.cross_layer.set(true);
        let (triples, doubles, writes, commits, waits) = (
            gpu.cross_layer_triples.get(),
            gpu.hc_gdn_pairs.get(),
            gpu.hc_final_writes.get(),
            gpu.helper_commits.get(),
            gpu.helper_waits.get(),
        );
        let actual = model.forward(token)?;
        let used_commits = gpu.helper_commits.get() - commits;
        ensure!(
            gpu.cross_layer_triples.get() - triples == 34
                && gpu.hc_gdn_pairs.get() - doubles == 2
                && gpu.hc_final_writes.get() - writes == 14
        );
        ensure!(
            used_commits + 34 == control_commits && gpu.helper_waits.get() - waits == used_commits
        );
        ensure!(
            actual == expected && finite_logits(&actual) && model.offset() == saved.position() + 1
        );
        ensure!(
            model
                .last_hidden()
                .context("missing candidate hidden")?
                .export_bytes(&gpu)?
                == hidden
        );
        let candidate = model.snapshot();
        let triples = gpu.cross_layer_triples.get();
        // Top-level exports must force completion even though the switch stays ON.
        let (next, states) = model.forward_trace(11)?;
        ensure!(
            gpu.cross_layer_triples.get() == triples
                && next == expected_next
                && finite_logits(&next)
        );
        compared += check_tensors(&states)?;
        ensure!(states.len() == expected_states.len());
        for (left, right) in states.iter().zip(&expected_states) {
            ensure!(
                left.name == right.name && left.shape == right.shape && left.data == right.data,
                "state changed {}",
                left.name
            );
        }
        model.restore(&candidate)?;
        cases.push(serde_json::json!({"token":token,"triples":34,"doubles":2,"final_writes":14,"control_commands":control_commits,"candidate_commands":used_commits}));
    }
    ensure!(compared == 115_212_288 && model.offset() == 4);
    let prefix = model.snapshot();
    let held_hidden = model
        .last_hidden()
        .context("missing prefix hidden")?
        .clone();
    let held_bytes = held_hidden.export_bytes(&gpu)?;
    gpu.cross_layer.set(false);
    let recovery = model.forward(19)?;
    let (recovery_next, recovery_states) = model.forward_trace(11)?;
    model.restore(&prefix)?;
    gpu.cross_layer.set(true);
    gpu.cross_layer_fail_after.set(Some(20));
    let triples = gpu.cross_layer_triples.get();
    ensure!(model.forward(42).is_err());
    gpu.cross_layer_fail_after.set(None);
    ensure!(gpu.cross_layer_triples.get() > triples && model.offset() == prefix.position());
    ensure!(
        held_hidden.export_bytes(&gpu)? == held_bytes
            && model
                .last_hidden()
                .context("rollback lost hidden")?
                .export_bytes(&gpu)?
                == held_bytes
    );
    ensure!(model.forward(19)? == recovery);
    let (next, states) = model.forward_trace(11)?;
    ensure!(next == recovery_next && states.len() == recovery_states.len());
    check_tensors(&states)?;
    for (a, b) in states.iter().zip(&recovery_states) {
        ensure!(
            a.name == b.name && a.shape == b.shape && a.data == b.data,
            "rollback changed {}",
            a.name
        );
    }
    model.restore(&prefix)?;
    // Captures use the synchronous production branch too, without running a draft model.
    #[cfg(feature = "chat")]
    {
        let triples = gpu.cross_layer_triples.get();
        let (logits, captures) = model.forward_captured(19, &[0, 4, 8, 12, 16])?;
        ensure!(
            logits == recovery
                && captures.rows == 1
                && captures.cols == 12800
                && gpu.cross_layer_triples.get() == triples
        );
        ensure!(captures.values.len() == 12800 && captures.values.iter().all(|v| v.is_finite()));
        model.restore(&prefix)?;
    }
    ensure!(gpu.readers()?.in_flight() == 0);
    eprintln!(
        "NATIVE_CROSS_LAYER_MODEL {}",
        serde_json::json!({"steps":cases,"compared_state_words":compared,"compared_logits":1986560,"initial_archive_states":84,"encode_failure_no_commit":true,"late_failure_layer":20,"retained_prefix":true,"recovery":true,"export_fallback":true,"parity":true})
    );
    Ok(())
}
