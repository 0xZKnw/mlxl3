//! One shared model gate for the composed A01–A10 runtime.
use super::{
    ssd_direct_tests::{check_initial, check_tensors},
    *,
};
use half::f16;

fn set_new(gpu: &Metal, on: bool) {
    gpu.hc_write_norm.set(on);
    gpu.routed_nt2_gu.set(on);
    gpu.routed_nt2_down.set(on);
    for cell in &gpu.expert_ar_enabled {
        cell.set(on);
    }
    gpu.gdn_whole.set(on);
    gpu.gdn_joint_ab.set(on);
    gpu.qsa_prepare.set(on);
    gpu.qsa_gate_fused.set(on);
}

fn flags(gpu: &Metal) -> [bool; 11] {
    [
        gpu.hc_write_norm.get(),
        gpu.routed_nt2_gu.get(),
        gpu.routed_nt2_down.get(),
        gpu.expert_ar_enabled[0].get(),
        gpu.expert_ar_enabled[1].get(),
        gpu.expert_ar_enabled[2].get(),
        gpu.expert_ar_enabled[3].get(),
        gpu.gdn_whole.get(),
        gpu.qsa_prepare.get(),
        gpu.qsa_gate_fused.get(),
        gpu.gdn_joint_ab.get(),
    ]
}

fn counts(gpu: &Metal) -> [u64; 11] {
    [
        gpu.hc_write_norm_uses.get() as u64,
        gpu.routed_nt2_uses[0].get(),
        gpu.routed_nt2_uses[1].get(),
        gpu.expert_ar_uses[0].get(),
        gpu.expert_ar_uses[1].get(),
        gpu.expert_ar_uses[2].get(),
        gpu.expert_ar_uses[3].get(),
        gpu.gdn_whole_uses.get() as u64,
        gpu.qsa_prepare_uses.get() as u64,
        gpu.qsa_gate_fused_uses.get() as u64,
        gpu.gdn_joint_ab_uses.get() as u64,
    ]
}
fn delta(gpu: &Metal, before: [u64; 11]) -> [u64; 11] {
    let after = counts(gpu);
    std::array::from_fn(|i| after[i] - before[i])
}
fn finite_logits(words: &[u16]) -> bool {
    words.len() == 248320 && words.iter().all(|&v| f16::from_bits(v).is_finite())
}
fn finite_hidden(bytes: &[u8]) -> bool {
    bytes.len() == 40960
        && bytes
            .as_chunks::<4>()
            .0
            .iter()
            .all(|v| f32::from_le_bytes(*v).is_finite())
}

#[test]
#[ignore = "physical Flash Next model, composed A01–A10 flags explicitly1 before Metal construction; exclusive GPU"]
fn physical_ar10_composition_preserves_full_state_and_continuation() -> Result<()> {
    let c = crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    let initial = flags(&gpu);
    ensure!(
        initial.iter().all(|&on| on) && gpu.cross_layer.get() && gpu.moe_deferred.get(),
        "all PSOs and accepted defaults must be initially enabled"
    );
    set_new(&gpu, false);
    let mut model = FlashNext::load(&gpu, &c, 4096)?;
    ensure!(model.prefetcher.is_none());
    let empty = model.snapshot();
    let (first, states) = model.forward_trace(248044)?;
    check_initial(&first, &states)?;
    check_tensors(&states)?;
    drop(states);
    model.restore(&empty)?;
    let mut cases = Vec::new();
    let zero_uses = [0u64; 11];
    let mut state_words = 0usize;
    for token in [248044, 1] {
        let saved = model.snapshot();
        set_new(&gpu, false);
        let count = counts(&gpu);
        let expected = model.forward(token)?;
        ensure!(finite_logits(&expected) && delta(&gpu, count) == [0; 11]);
        let hidden = model
            .last_hidden()
            .context("missing canonical hidden")?
            .export_bytes(&gpu)?;
        ensure!(finite_hidden(&hidden));
        let control = model.snapshot();
        let (expected_next, expected_states) = model.forward_trace(11)?;
        ensure!(finite_logits(&expected_next));
        check_tensors(&expected_states)?;
        model.restore(&saved)?;
        set_new(&gpu, true);
        let count = counts(&gpu);
        let triples = gpu.cross_layer_triples.get();
        let deferred = gpu.moe_deferred_uses.get();
        let dispatch = gpu.expert_axis_dispatches.get();
        let actual = model.forward(token)?;
        let used = delta(&gpu, count);
        let axis_dispatches = gpu.expert_axis_dispatches.get() - dispatch;
        ensure!(
            used[0] > 0 && used[1..] == [960, 480, 480, 480, 48, 480, 36, 12, 12, 36],
            "missing actual composed path: {used:?}"
        );
        ensure!(
            (144..1440).contains(&axis_dispatches) && axis_dispatches.is_multiple_of(3),
            "no actual expert grouping"
        );
        ensure!(
            gpu.cross_layer_triples.get() - triples == 34
                && gpu.moe_deferred_uses.get() - deferred == 34
        );
        ensure!(actual == expected && finite_logits(&actual));
        let actual_hidden = model
            .last_hidden()
            .context("missing candidate hidden")?
            .export_bytes(&gpu)?;
        ensure!(actual_hidden == hidden && finite_hidden(&actual_hidden));
        set_new(&gpu, false);
        let count = counts(&gpu);
        let triples = gpu.cross_layer_triples.get();
        let (next, states) = model.forward_trace(11)?;
        ensure!(
            next == expected_next
                && finite_logits(&next)
                && delta(&gpu, count) == [0; 11]
                && gpu.cross_layer_triples.get() == triples
        );
        state_words += check_tensors(&states)?;
        for (actual, expected) in states.iter().zip(&expected_states) {
            ensure!(
                actual.name == expected.name
                    && actual.shape == expected.shape
                    && actual.data == expected.data,
                "composed state {}",
                actual.name
            );
        }
        cases.push(serde_json::json!({"token":token,"parity":true,"actual_uses":used,"axis_qmv_dispatches":axis_dispatches,"cross_layer_triples":34,"moe_deferred":34,"legacy_continuation":true,"continuation_new_uses":zero_uses,"states":84}));
        model.restore(&control)?;
    }
    ensure!(
        cases.len() == 2
            && state_words == 57606144
            && model.offset() == 2
            && gpu.readers()?.in_flight() == 0
    );
    eprintln!(
        "NATIVE_AR10_COMPOSITION_MODEL {}",
        serde_json::json!({"parity":true,"initial_flags":initial,"context":4096,"cases":cases,"case_count":2,"compared_state_words":state_words,"compared_logits":993280,"compared_hidden_bytes":81920,"initial_archive_states":84,"legacy_continuation":true,"all_readers_joined":true})
    );
    Ok(())
}

#[test]
#[ignore = "physical Metal initialization, all AR10 flags explicitly OFF; exclusive GPU"]
fn physical_ar10_idle_does_not_create_expert_pipelines() -> Result<()> {
    let gpu = Metal::new()?;
    ensure!(gpu.expert_ar_enabled.iter().all(|v| !v.get()));
    ensure!(
        gpu.expert_up_finish.is_none()
            && gpu.expert_gu_prepare.is_none()
            && gpu.expert_axis_qmv.is_none()
    );
    eprintln!("AR10_IDLE_PIPELINES {{\"parity\":true,\"absent_expert_psos\":6}}");
    Ok(())
}
