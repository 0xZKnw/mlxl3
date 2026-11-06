use super::super::{
    FlashNext, Metal,
    ssd_direct_tests::{check_initial, check_tensors},
};
use anyhow::{Context, Result, ensure};
use half::f16;

fn finite(logits: &[u16]) -> bool {
    logits.len() == 248320 && logits.iter().all(|&v| f16::from_bits(v).is_finite())
}

#[test]
#[ignore = "exclusive physical GDN/P chain, exact model states and producer failure recovery"]
fn physical_gdn_route_preserves_states_and_removes_36_commands() -> Result<()> {
    check_route_composition(false)
}

#[test]
#[ignore = "physical P02 + five-buffer Hot workspace + compact index composition; exclusive GPU"]
fn physical_ar42_composition_keeps_full_states_and_error_recovery() -> Result<()> {
    check_route_composition(true)
}

fn check_route_composition(compose: bool) -> Result<()> {
    let c = crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    ensure!(gpu.cross_layer.get() && gpu.hc_gdn_chain.get());
    let mut model = FlashNext::load(&gpu, &c, 4096)?;
    let initial = model.snapshot();
    gpu.gdn_route_chain.set(compose);
    gpu.hot_workspace.set(false);
    gpu.qsa_index_key_only.set(false);
    let (logits, states) = model.forward_trace(248044)?;
    check_initial(&logits, &states)?;
    check_tensors(&states)?;
    model.restore(&initial)?;
    let mut cases = Vec::new();
    let mut words = 0;
    for token in [248044, 42] {
        let prefix = model.snapshot();
        gpu.gdn_route_chain.set(compose);
        gpu.hot_workspace.set(false);
        gpu.qsa_index_key_only.set(false);
        let before = gpu.helper_commits.get();
        let expected = model.forward(token)?;
        let baseline_commands = gpu.helper_commits.get() - before;
        let hidden = model
            .last_hidden()
            .context("reference hidden")?
            .export_bytes(&gpu)?;
        ensure!(finite(&expected) && hidden.len() == 40960);
        let (next, expected_states) = model.forward_trace(11)?;
        check_tensors(&expected_states)?;
        model.restore(&prefix)?;
        gpu.gdn_route_chain.set(true);
        let (before, uses) = (gpu.helper_commits.get(), gpu.gdn_route_uses.get());
        gpu.hot_workspace.set(compose);
        gpu.qsa_index_key_only.set(compose);
        let (workspace_uses, index_uses) =
            (gpu.hot_workspace_reuses.get(), gpu.index_key_uses.get());
        if compose {
            crate::flash_profile::start(false)?;
        }
        let row = compose.then(|| crate::flash_profile::span("native.row"));
        let actual = model.forward(token);
        drop(row);
        if compose {
            let trace = crate::flash_profile::finish()?;
            ensure!(trace.len() > 1000);
            crate::flash_profile::validate_cross_layer_ancestry(&trace)?;
        }
        ensure!(actual? == expected);
        if compose {
            ensure!(gpu.hot_workspace_reuses.get() - workspace_uses >= 47);
            ensure!(gpu.index_key_uses.get() - index_uses == 12);
        }
        let candidate_commands = gpu.helper_commits.get() - before;
        ensure!(gpu.gdn_route_uses.get() - uses == 36);
        ensure!(candidate_commands + if compose { 0 } else { 36 } == baseline_commands);
        ensure!(
            model
                .last_hidden()
                .context("candidate hidden")?
                .export_bytes(&gpu)?
                == hidden
        );
        let candidate = model.snapshot();
        let uses = gpu.gdn_route_uses.get();
        let (actual_next, actual_states) = model.forward_trace(11)?;
        ensure!(
            gpu.gdn_route_uses.get() == uses,
            "export took the asynchronous route"
        );
        ensure!(
            gpu.gdn_route_chain.get(),
            "export did not restore the route setting"
        );
        ensure!(
            next == actual_next && finite(&actual_next),
            "continuation logits changed"
        );
        words += check_tensors(&actual_states)?;
        ensure!(actual_states.len() == 84 && expected_states.len() == 84);
        for (a, b) in actual_states.iter().zip(&expected_states) {
            ensure!(
                a.name == b.name && a.shape == b.shape && a.data == b.data,
                "state changed {}",
                a.name
            );
        }
        model.restore(&candidate)?;
        cases.push(serde_json::json!({"token":token,"uses":36,"baseline_commands":baseline_commands,"candidate_commands":candidate_commands}));
    }
    ensure!(words == 57_606_144 && model.offset() == 2);
    if compose {
        ensure!(finite(&model.forward(31)?));
    }
    let prefix = model.snapshot();
    let held = model.last_hidden().context("held hidden")?.clone();
    let held_bytes = held.export_bytes(&gpu)?;
    gpu.gdn_route_chain.set(compose);
    gpu.hot_workspace.set(false);
    gpu.qsa_index_key_only.set(false);
    let expected = model.forward(19)?;
    model.restore(&prefix)?;
    gpu.gdn_route_chain.set(true);
    gpu.hot_workspace.set(compose);
    gpu.qsa_index_key_only.set(compose);
    super::super::hc::FAIL_NEXT_PENDING_HC_ROUTE.with(|v| v.set(true));
    let error = model
        .forward(19)
        .err()
        .context("pending route failure was lost")?;
    ensure!(format!("{error:#}").contains("injected pending HC route failure"));
    ensure!(model.offset() == prefix.position() && held.export_bytes(&gpu)? == held_bytes);
    ensure!(model.forward(19)? == expected);
    model.restore(&prefix)?;
    gpu.moe_deferred_fail.set(true);
    let error = model
        .forward(19)
        .err()
        .context("previous MoE failure was lost")?;
    ensure!(format!("{error:#}").contains("injected deferred MoE producer failure"));
    ensure!(model.offset() == prefix.position() && held.export_bytes(&gpu)? == held_bytes);
    ensure!(model.forward(19)? == expected && gpu.readers()?.in_flight() == 0);
    if compose {
        let (reference_next, reference_states) = model.forward_trace(11)?;
        model.restore(&prefix)?;
        gpu.cross_layer_fail_after.set(Some(20));
        let failed = model.forward(19);
        gpu.cross_layer_fail_after.set(None);
        ensure!(failed.is_err() && model.offset() == prefix.position());
        ensure!(held.export_bytes(&gpu)? == held_bytes);
        ensure!(model.forward(19)? == expected);
        let (next, states) = model.forward_trace(11)?;
        ensure!(next == reference_next && states.len() == 84 && reference_states.len() == 84);
        check_tensors(&states)?;
        for (a, b) in states.iter().zip(&reference_states) {
            ensure!(
                a.name == b.name && a.shape == b.shape && a.data == b.data,
                "late rollback changed {}",
                a.name
            );
        }
    }

    eprintln!(
        "{} {}",
        if compose {
            "AR42_COMPOSITION"
        } else {
            "AR41_GDN_ROUTE"
        },
        serde_json::json!({"parity":true,"cases":cases,"state_words":words,"independent_initial":true,"pending_failure_recovery":true,"previous_moe_failure_recovery":true,"trace_fallback":true})
    );
    Ok(())
}
