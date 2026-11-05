use super::super::{
    moe::EXTRA_WAVE_SUBMISSIONS,
    ssd_direct_tests::{check_initial, check_tensors, digest},
};
use super::*;

fn select(model: &mut FlashNext<'_>, enabled: bool, cold: bool) -> Result<()> {
    for layer in &mut model.layers {
        let moe = layer.moe();
        moe.extra_wave_enabled = enabled;
        if cold {
            moe.clear_test_cache()?;
        }
    }
    Ok(())
}
fn finite_logits(values: &[u16]) -> bool {
    values.len() == 248320 && values.iter().all(|&v| f16::from_bits(v).is_finite())
}

#[test]
#[ignore = "physical L2 OFF/ON ctx4096 actual extra waves, independent initial archive and full continuation states; run alone"]
fn bounded_extra_wave_preserves_model_states_logits_hidden_and_continuation() -> Result<()> {
    let checkpoint =
        crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    let initial_cross_layer_enabled = gpu.cross_layer.get();
    ensure!(
        initial_cross_layer_enabled,
        "requires accepted L1 on qualified M5"
    );
    let mut model = FlashNext::load(&gpu, &checkpoint, 4096)?;
    ensure!(model.prefetcher.is_none() && model.lookahead_depth == 0);
    ensure!(
        model
            .layers
            .iter_mut()
            .all(|layer| layer.moe().extra_wave_enabled),
        "requires loaded flag1 on exact Apple M5"
    );
    select(&mut model, false, true)?;
    let empty = model.snapshot();
    let (initial, states) = model.forward_trace(248044)?;
    check_initial(&initial, &states)?;
    check_tensors(&states)?;
    drop(states);
    model.restore(&empty)?;
    let mut cases = Vec::new();
    let mut compared = 0usize;
    let mut actual_extra_submissions = 0;
    for token in [248044, 1, 42, 7] {
        let saved = model.snapshot();
        select(&mut model, false, false)?;
        ensure!(finite_logits(&model.forward(19)?));
        model.restore(&saved)?;
        select(&mut model, false, true)?;
        let count = EXTRA_WAVE_SUBMISSIONS.with(std::cell::Cell::get);
        let triples = gpu.cross_layer_triples.get();
        let expected = model.forward(token)?;
        let control_cross_layer_triples = gpu.cross_layer_triples.get() - triples;
        ensure!(control_cross_layer_triples == 34);
        ensure!(
            EXTRA_WAVE_SUBMISSIONS.with(std::cell::Cell::get) == count && finite_logits(&expected)
        );
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
        let triples = gpu.cross_layer_triples.get();
        let (expected_next, expected_states) = model.forward_trace(11)?;
        ensure!(gpu.cross_layer_triples.get() == triples);
        check_tensors(&expected_states)?;
        model.restore(&saved)?;
        select(&mut model, true, true)?;
        let count = EXTRA_WAVE_SUBMISSIONS.with(std::cell::Cell::get);
        let triples = gpu.cross_layer_triples.get();
        let actual = model.forward(token)?;
        let cross_layer_triples = gpu.cross_layer_triples.get() - triples;
        ensure!(cross_layer_triples == 34);
        let extra = EXTRA_WAVE_SUBMISSIONS.with(std::cell::Cell::get) - count;
        actual_extra_submissions += extra;
        ensure!(extra <= 48 && actual == expected && finite_logits(&actual));
        ensure!(
            model.offset() == saved.position() + 1
                && model
                    .last_hidden()
                    .context("missing candidate hidden")?
                    .export_bytes(&gpu)?
                    == hidden
        );
        let candidate = model.snapshot();
        select(&mut model, false, false)?;
        let count = EXTRA_WAVE_SUBMISSIONS.with(std::cell::Cell::get);
        let triples = gpu.cross_layer_triples.get();
        let (next, states) = model.forward_trace(11)?;
        let continuation_cross_layer_triples = gpu.cross_layer_triples.get() - triples;
        ensure!(continuation_cross_layer_triples == 0);
        ensure!(
            EXTRA_WAVE_SUBMISSIONS.with(std::cell::Cell::get) == count
                && next == expected_next
                && finite_logits(&next)
        );
        compared += check_tensors(&states)?;
        ensure!(states.len() == expected_states.len());
        for (actual, expected) in states.iter().zip(&expected_states) {
            ensure!(
                actual.name == expected.name
                    && actual.shape == expected.shape
                    && actual.data == expected.data,
                "L2 changed {}",
                actual.name
            );
        }
        model.restore(&candidate)?;
        cases.push(serde_json::json!({"token":token,"control_cross_layer_triples":control_cross_layer_triples,"cross_layer_triples":cross_layer_triples,"continuation_cross_layer_triples":continuation_cross_layer_triples,"actual_extra_submissions":extra,"control_extra_submissions":0,"snapshot_position":candidate.position()}));
    }
    ensure!(
        actual_extra_submissions > 0
            && cases.len() == 4
            && compared == 115_212_288
            && model.offset() == 4
            && gpu.readers()?.in_flight() == 0
    );
    // Exercise the accepted deferred boundary after real extra SSD work, then replay the
    // retained prefix. A merely enabled flag with no extra submission is not a valid oracle.
    let prefix = model.snapshot();
    let held_hidden = model
        .last_hidden()
        .context("missing retained hidden")?
        .clone();
    let held_bytes = held_hidden.export_bytes(&gpu)?;
    select(&mut model, false, true)?;
    let recovery = model.forward(19)?;
    let (recovery_next, recovery_states) = model.forward_trace(11)?;
    ensure!(finite_logits(&recovery) && finite_logits(&recovery_next));
    check_tensors(&recovery_states)?;
    model.restore(&prefix)?;
    select(&mut model, true, true)?;
    gpu.cross_layer_fail_after.set(Some(20));
    let extras = EXTRA_WAVE_SUBMISSIONS.with(std::cell::Cell::get);
    let triples = gpu.cross_layer_triples.get();
    let failed = model.forward(42);
    gpu.cross_layer_fail_after.set(None);
    let failure_extra_submissions = EXTRA_WAVE_SUBMISSIONS.with(std::cell::Cell::get) - extras;
    let failure_cross_layer_triples = gpu.cross_layer_triples.get() - triples;
    let error = failed.err().context("expected the injected late failure")?;
    ensure!(format!("{error:#}").contains("injected late layer failure 20"));
    ensure!(
        failure_extra_submissions > 0
            && failure_cross_layer_triples > 0
            && model.offset() == prefix.position()
    );
    ensure!(
        held_hidden.export_bytes(&gpu)? == held_bytes
            && model
                .last_hidden()
                .context("rollback lost hidden")?
                .export_bytes(&gpu)?
                == held_bytes
    );
    select(&mut model, true, true)?;
    ensure!(model.forward(19)? == recovery);
    select(&mut model, false, false)?;
    let (next, states) = model.forward_trace(11)?;
    ensure!(next == recovery_next && finite_logits(&next) && states.len() == recovery_states.len());
    let rollback_state_words = check_tensors(&states)?;
    for (actual, expected) in states.iter().zip(&recovery_states) {
        ensure!(
            actual.name == expected.name
                && actual.shape == expected.shape
                && actual.data == expected.data,
            "L1+L2 rollback changed {}",
            actual.name
        );
    }
    model.restore(&prefix)?;
    ensure!(model.offset() == 4 && gpu.readers()?.in_flight() == 0);
    eprintln!(
        "NATIVE_SSD_EXTRA_WAVE_MODEL {}",
        serde_json::json!({"steps":cases,"actual_extra_submissions":actual_extra_submissions,"compared_state_words":compared,"compared_logits":1986560,"initial_cross_layer_enabled":initial_cross_layer_enabled,"initial_archive_states":84,"context":4096,"cold_control_and_candidate":true,"late_failure_layer":20,"failure_extra_submissions":failure_extra_submissions,"failure_cross_layer_triples":failure_cross_layer_triples,"rollback_state_words":rollback_state_words,"rollback_logits":496640,"retained_prefix":true,"late_failure_recovery":true,"explicit_legacy_continuation":true,"parity":true})
    );
    Ok(())
}
