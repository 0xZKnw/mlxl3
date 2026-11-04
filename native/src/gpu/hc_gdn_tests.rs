use super::super::{
    FlashHc, FlashHcRead, FlashNext, MatrixView, Metal,
    ssd_direct_tests::{check_initial, check_tensors, digest},
    view::FloatType,
};
use super::{FlashGdn, FlashGdnLayer};
use crate::{checkpoint, tensor_layout::MatrixLayout};
use anyhow::{Context, Result, ensure};
use half::f16;
use metal::MTLCommandBufferStatus;

fn finite_logits(logits: &[u16]) -> bool {
    logits.len() == 248320 && logits.iter().all(|&v| f16::from_bits(v).is_finite())
}
fn finite_values(values: &[u32]) -> bool {
    values.len() == 2560
        && values.iter().all(|&v| f32::from_bits(v).is_finite())
        && values.iter().any(|&v| v & 0x7fffffff != 0)
}
fn check_read(read: &FlashHcRead) -> Result<(Vec<u32>, Vec<u32>)> {
    let gate = read.gate.as_ref().context("missing component HC gate")?;
    let mixed = read.mixed.read_output()?;
    let gate_words = gate.read_output()?;
    for (view, words, count) in [(&read.mixed, &mixed, 2560), (gate, &gate_words, 4)] {
        ensure!(
            words.len() == count
                && words.iter().all(|&v| match view.dtype {
                    FloatType::F16 => u16::try_from(v).is_ok_and(|v| f16::from_bits(v).is_finite()),
                    FloatType::F32 => f32::from_bits(v).is_finite(),
                })
        );
    }
    Ok((mixed, gate_words))
}
fn check_trace(trace: &std::collections::BTreeMap<&str, Vec<u32>>) -> Result<()> {
    ensure!(trace.len() == 13);
    for (name, count) in [
        ("qkv", 10240),
        ("z", 6144),
        ("a", 48),
        ("b", 48),
        ("conv", 10240),
        ("q", 2048),
        ("k", 2048),
        ("v", 6144),
        ("delta", 6144),
        ("normed", 6144),
        ("gated", 6144),
        ("history", 30720),
        ("state", 786432),
    ] {
        let words = trace.get(name).context("missing GDN trace tensor")?;
        ensure!(
            words.len() == count
                && words
                    .iter()
                    .all(|&v| if matches!(name, "normed" | "state") {
                        f32::from_bits(v).is_finite()
                    } else {
                        u16::try_from(v).is_ok_and(|v| f16::from_bits(v).is_finite())
                    }),
            "invalid GDN trace {name}"
        );
    }
    Ok(())
}
fn streams(gpu: &Metal, half: bool) -> Result<MatrixView> {
    let row = (0..10240)
        .map(|i| ((i % 113) as f32 - 56.) / 79.)
        .collect::<Vec<_>>();
    let storage = if half {
        gpu.f16_storage(
            &row.iter()
                .map(|&v| f16::from_f32(v).to_bits())
                .collect::<Vec<_>>(),
        )?
    } else {
        gpu.f32_storage(&row)?
    };
    storage.matrix(MatrixLayout {
        shape: [1, 10240],
        strides: [10240, 1],
        offset: 0,
    })
}

#[test]
#[ignore = "physical HC/GDN untraced join, trace fallback, pending ownership and late layer rollback; run alone"]
fn physical_hc_gdn_chain_preserves_component_outputs_and_failure_ownership() -> Result<()> {
    let c = checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    ensure!(
        gpu.hc_gdn_chain.get(),
        "qualified M5 and HC_GDN_CHAIN=1 required"
    );
    let hc = FlashHc::load(
        &gpu,
        &c,
        "model.language_model.layers.0.attn_hyper_connection",
        1e-6,
        true,
    )?;
    let mut gdn = FlashGdn::load(&gpu, &c, "model.language_model.layers.0.linear_attn", 1e-6)?;
    let mut cases = Vec::new();
    for half in [false, true] {
        let input = streams(&gpu, half)?;
        gpu.hc_gdn_chain.set(false);
        gdn.forward_after_hc(&hc, &input, false)?;
        let saved = gdn.snapshot();
        let commands = gpu.helper_commits.get();
        let waits = gpu.helper_waits.get();
        let pairs = gpu.hc_gdn_pairs.get();
        let (control_read, control) = gdn.forward_after_hc(&hc, &input, false)?;
        let control_values = control.values.read_output()?;
        let read_values = check_read(&control_read)?;
        let control_commands = gpu.helper_commits.get() - commands;
        let control_waits = gpu.helper_waits.get() - waits;
        ensure!(
            control_commands == 2
                && control_waits == 2
                && gpu.hc_gdn_pairs.get() == pairs
                && finite_values(&control_values)
        );
        let (_, continuation) = gdn.forward_after_hc(&hc, &input, true)?;
        let continuation_values = continuation.values.read_output()?;
        let continuation_trace = continuation.trace.context("missing legacy GDN trace")?;
        check_trace(&continuation_trace)?;
        ensure!(finite_values(&continuation_values));
        gdn.restore(&saved)?;
        gpu.hc_gdn_chain.set(true);
        let commands = gpu.helper_commits.get();
        let waits = gpu.helper_waits.get();
        let pairs = gpu.hc_gdn_pairs.get();
        let (actual_read, actual) = gdn.forward_after_hc(&hc, &input, false)?;
        ensure!(
            gpu.hc_gdn_pairs.get() - pairs == 1
                && gpu.helper_commits.get() - commands == 1
                && gpu.helper_waits.get() - waits == 1
        );
        ensure!(
            actual.values.read_output()? == control_values
                && check_read(&actual_read)? == read_values
        );
        let trace_pairs = gpu.hc_gdn_pairs.get();
        let (_, actual_next) = gdn.forward_after_hc(&hc, &input, true)?;
        ensure!(
            gpu.hc_gdn_pairs.get() == trace_pairs
                && actual_next.values.read_output()? == continuation_values
                && actual_next.trace.as_ref() == Some(&continuation_trace)
        );
        gdn.restore(&saved)?;
        let pairs = gpu.hc_gdn_pairs.get();
        let commands = gpu.helper_commits.get();
        let (traced_read, traced) = gdn.forward_after_hc(&hc, &input, true)?;
        ensure!(
            traced_read.trace.is_some()
                && traced.trace.is_some()
                && traced.values.read_output()? == control_values
                && gpu.hc_gdn_pairs.get() == pairs
                && gpu.helper_commits.get() - commands == 2
        );
        ensure!(check_read(&traced_read)? == read_values);
        check_trace(traced.trace.as_ref().unwrap())?;
        gdn.restore(&saved)?;
        let command = gpu.queue.new_command_buffer().to_owned();
        let read = hc.encode_read(&command, &input, false)?;
        let bad = gpu.f32_storage(&vec![0.; 1280])?.matrix(MatrixLayout {
            shape: [1, 1280],
            strides: [1280, 1],
            offset: 0,
        })?;
        let commands = gpu.helper_commits.get();
        ensure!(gdn.encode(&command, &bad, false).is_err());
        let pending = gdn.encode(&command, read.mixed(), false)?;
        ensure!(
            gdn.finish(pending).is_err()
                && command.status() == MTLCommandBufferStatus::NotEnqueued
                && gpu.helper_commits.get() == commands
        );
        drop((read, command));
        let (_, recovered) = gdn.forward_after_hc(&hc, &input, false)?;
        ensure!(
            recovered.values.read_output()? == control_values,
            "uncommitted pending state was published"
        );
        let (_, after_recovery) = gdn.forward_after_hc(&hc, &input, true)?;
        ensure!(
            after_recovery.values.read_output()? == continuation_values
                && after_recovery.trace.as_ref() == Some(&continuation_trace)
        );
        cases.push(serde_json::json!({"half":half,"control_commands":2,"joined_commands":1,"control_waits":2,"joined_waits":1,"actual_pairs":1,"trace_fallback":true,"uncommitted_rejected":true,"recovery":true}));
    }
    gpu.hc_gdn_chain.set(true);
    let input = streams(&gpu, false)?;
    let foreign = Metal::new()?;
    let foreign_hc = FlashHc::load(
        &foreign,
        &c,
        "model.language_model.layers.0.attn_hyper_connection",
        1e-6,
        true,
    )?;
    let owner_saved = gdn.snapshot();
    gpu.hc_gdn_chain.set(false);
    let (owner_read, owner_control) = gdn.forward_after_hc(&hc, &input, false)?;
    let owner_values = owner_control.values.read_output()?;
    let owner_read_values = check_read(&owner_read)?;
    gdn.restore(&owner_saved)?;
    gpu.hc_gdn_chain.set(true);
    let commands = gpu.helper_commits.get();
    let foreign_commands = foreign.helper_commits.get();
    let pairs = gpu.hc_gdn_pairs.get();
    let (foreign_read, foreign_output) = gdn.forward_after_hc(&foreign_hc, &input, false)?;
    ensure!(
        gpu.hc_gdn_pairs.get() == pairs
            && gpu.helper_commits.get() - commands == 1
            && foreign.helper_commits.get() - foreign_commands == 1
            && foreign_output.values.read_output()? == owner_values
            && check_read(&foreign_read)? == owner_read_values
    );
    gdn.restore(&owner_saved)?;
    let mut layer = FlashGdnLayer::load_body(&gpu, &c, "model.language_model.layers.0", 1e-6, 10)?;
    layer.forward(&input)?;
    let saved = layer.snapshot();
    let recurrent = layer
        .recurrent()
        .context("missing warm recurrent state")?
        .clone();
    let before = recurrent.export_bytes(&gpu)?;
    let expected = layer.forward(&input)?.read_output()?;
    layer.restore(&saved)?;
    let pairs = gpu.hc_gdn_pairs.get();
    ensure!(
        layer.forward_lookahead(&input, None, 11).is_err() && gpu.hc_gdn_pairs.get() - pairs == 1,
        "late failure did not execute joined attention"
    );
    let after = layer.recurrent().context("rollback lost recurrent state")?;
    ensure!(
        std::ptr::eq(after.buffer.as_ref(), recurrent.buffer.as_ref())
            && after.export_bytes(&gpu)? == before
    );
    ensure!(
        layer.forward(&input)?.read_output()? == expected,
        "late layer failure changed recovery"
    );
    ensure!(gpu.readers()?.in_flight() == 0);
    eprintln!(
        "NATIVE_HC_GDN_COMPONENT {}",
        serde_json::json!({"cases":cases,"finite_words_per_case":2560,"foreign_owner_fallback":true,"late_failure_actual_pairs":1,"old_frame_retained":true,"layer_recovery":true,"parity":true})
    );
    Ok(())
}

#[test]
#[ignore = "physical untraced HC/GDN model join, context4096, full continuation states and independent archives; run alone"]
fn physical_hc_gdn_chain_preserves_untraced_model_states_and_actual_joins() -> Result<()> {
    let c = checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    ensure!(
        gpu.hc_gdn_chain.get(),
        "qualified M5 and HC_GDN_CHAIN=1 required"
    );
    let mut model = FlashNext::load(&gpu, &c, 4096)?;
    ensure!(model.prefetcher.is_none());
    gpu.hc_gdn_chain.set(false);
    let empty = model.snapshot();
    let (logits, states) = model.forward_trace(248044)?;
    check_initial(&logits, &states)?;
    check_tensors(&states)?;
    drop(states);
    model.restore(&empty)?;
    let mut cases = Vec::new();
    let mut compared = 0usize;
    for token in [248044, 1, 42, 7] {
        let saved = model.snapshot();
        gpu.hc_gdn_chain.set(false);
        ensure!(finite_logits(&model.forward(19)?));
        model.restore(&saved)?;
        let commands = gpu.helper_commits.get();
        let waits = gpu.helper_waits.get();
        let pairs = gpu.hc_gdn_pairs.get();
        let expected = model.forward(token)?;
        let control_commands = gpu.helper_commits.get() - commands;
        let control_waits = gpu.helper_waits.get() - waits;
        ensure!(finite_logits(&expected) && gpu.hc_gdn_pairs.get() == pairs);
        if cases.is_empty() {
            ensure!(
                digest(
                    &expected
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>()
                )? == "2f78a4d78cb113666fa3eda97cf115abe293bd428e593bc54400744eae422445"
            );
        }
        let hidden = model
            .last_hidden()
            .context("missing control hidden")?
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
        gpu.hc_gdn_chain.set(true);
        let commands = gpu.helper_commits.get();
        let waits = gpu.helper_waits.get();
        let pairs = gpu.hc_gdn_pairs.get();
        // This is the actual optimized path; exporting a trace alone is not its activation oracle.
        let actual = model.forward(token)?;
        let candidate_commands = gpu.helper_commits.get() - commands;
        let candidate_waits = gpu.helper_waits.get() - waits;
        ensure!(
            gpu.hc_gdn_pairs.get() - pairs == 36
                && candidate_commands + 36 == control_commands
                && candidate_waits + 36 == control_waits
        );
        ensure!(
            actual == expected
                && finite_logits(&actual)
                && model.offset() == saved.position() + 1
                && model
                    .last_hidden()
                    .context("missing joined hidden")?
                    .export_bytes(&gpu)?
                    == hidden
        );
        let candidate = model.snapshot();
        // forward_trace exports states but does not request component trace=true. Force legacy
        // continuation explicitly, starting from the state produced by the joined untraced row.
        gpu.hc_gdn_chain.set(false);
        let pairs = gpu.hc_gdn_pairs.get();
        let (next, states) = model.forward_trace(11)?;
        ensure!(gpu.hc_gdn_pairs.get() == pairs && next == expected_next && finite_logits(&next));
        compared += check_tensors(&states)?;
        ensure!(states.len() == expected_states.len());
        for (left, right) in states.iter().zip(&expected_states) {
            ensure!(
                left.name == right.name && left.shape == right.shape && left.data == right.data,
                "joined state changed {}",
                left.name
            );
        }
        model.restore(&candidate)?;
        cases.push(serde_json::json!({"token":token,"actual_pairs":36,"control_commands":control_commands,"joined_commands":candidate_commands,"control_waits":control_waits,"joined_waits":candidate_waits,"legacy_continuation_pairs":0,"snapshot_position":candidate.position()}));
    }
    ensure!(compared == 115_212_288 && model.offset() == 4 && gpu.readers()?.in_flight() == 0);
    eprintln!(
        "NATIVE_HC_GDN_MODEL {}",
        serde_json::json!({"steps":cases,"untraced_joined_rows":4,"actual_pairs":144,"compared_state_words":compared,"compared_logits":1986560,"initial_archive_states":84,"context":4096,"explicit_legacy_continuation":true,"parity":true})
    );
    Ok(())
}
