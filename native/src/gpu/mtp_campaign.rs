//! Opt-in matched-allocation diagnostic. Never part of release code.
use super::*;
use crate::tokenizer::ChatTokenizer;
use serde_json::{Value, json};
use std::{path::Path, sync::atomic::AtomicBool, time::Instant};

const ARMS: [&str; 5] = ["ar", "d1", "b2", "b2_ssd", "b2_ssd_shared"];
const PROMPTS: [(&str, &str); 2] = [
    (
        "MAIN",
        "Explique en français, avec un exemple concret, comment un cache LFU évite les lectures SSD inutiles pendant l’inférence d’un modèle MoE.",
    ),
    (
        "HOLD",
        "Écris une fonction Python merge_intervals qui fusionne les intervalles qui se chevauchent, puis explique sa complexité et donne un exemple.",
    ),
];
fn slots(target: &mut FlashNext<'_>) -> Vec<usize> {
    target
        .layers
        .iter_mut()
        .map(|l| l.moe().campaign_slots())
        .collect()
}
fn clear(target: &mut FlashNext<'_>, head: &mut FlashMtpHead<'_>) -> Result<()> {
    target.reset();
    head.reset();
    for layer in &mut target.layers {
        layer.moe().clear_test_cache()?;
    }
    head.layer.residual.moe.clear_test_cache()?;
    Ok(())
}
fn run<'a>(
    target: &mut FlashNext<'a>,
    head: &mut FlashMtpHead<'a>,
    prompt: &[u32],
    arm: usize,
    budget: usize,
    eos: &[u32],
) -> Result<Value> {
    ensure!(arm < ARMS.len() && budget >= 2 && prompt.len() + budget < 4096);
    head.block_verify = arm >= 2;
    target.verify_ssd_cohorts = arm >= 3;
    let _shared_override = crate::flash_qmv::SharedVerifyOverride::new(arm == 4);
    let cancel = AtomicBool::new(false);
    let mut session = FlashMtpSession::default();
    let begin = Instant::now();
    let logits = if arm == 0 {
        target.reset();
        target.prefill(prompt)?
    } else {
        session.prefill(target, head, prompt, &cancel)?
    };
    let mut anchor = target.greedy_id(&logits)?;
    let prefill = begin.elapsed().as_secs_f64();
    let mut ids = vec![anchor];
    ensure!(!eos.contains(&anchor), "first token EOS");
    let initial_memory = target.memory_stats()?;
    let mut max_footprint = initial_memory.process_footprint_bytes;
    let mut max_metal = initial_memory.native_metal_allocated_bytes;
    let mut calls = Vec::new();
    crate::ssd_reads::campaign_reads(true);
    let paths_before = crate::flash_qmv::shared_path_counts();
    let decode = Instant::now();
    while ids.len() < budget {
        let proposed = session.proposed;
        let accepted = session.accepted;
        let pending = session.pending.is_some();
        let step = Instant::now();
        anchor = if arm == 0 {
            let logits = target.forward(anchor)?;
            target.greedy_id(&logits)?
        } else {
            session.advance_with_eos(target, head, anchor, 4096, budget - ids.len(), eos)?
        };
        let elapsed = step.elapsed().as_secs_f64();
        ensure!(
            !eos.contains(&anchor),
            "early EOS invalidates fixed-token comparison"
        );
        ids.push(anchor);
        calls.push(json!({"seconds":elapsed,"pending_before":pending,"proposed":session.proposed-proposed,"accepted":session.accepted-accepted,"useful_tokens":1}));
        // Boundary samples are explicitly not an instantaneous peak measurement.
        let memory = target.memory_stats()?;
        max_footprint = max_footprint.max(memory.process_footprint_bytes);
        max_metal = max_metal.max(memory.native_metal_allocated_bytes);
    }
    let seconds = decode.elapsed().as_secs_f64();
    let (bytes, jobs) = crate::ssd_reads::campaign_reads(false);
    ensure!(seconds.is_finite() && seconds > 0. && ids.len() == budget);
    let memory = target.memory_stats()?;
    let paths_after = crate::flash_qmv::shared_path_counts();
    let packed = paths_after.0 - paths_before.0;
    let dispatched = paths_after.1 - paths_before.1;
    ensure!(
        arm == 4 || (packed == 0 && dispatched == 0),
        "QMV OFF arm executed the shared packed path"
    );
    ensure!(
        arm != 4 || session.proposed == 0 || (packed > 0 && dispatched > 0),
        "QMV ON arm did not exercise packing and specialized dispatch"
    );
    Ok(
        json!({"arm":ARMS[arm],"tokens":ids,"generated_tokens":budget,"decode_tokens":budget-1,"decode_seconds":seconds,"decode_tps":(budget-1) as f64/seconds,"prefill_seconds":prefill,"proposed":session.proposed,"accepted":session.accepted,"submitted_expert_read_bytes":bytes,"submitted_read_jobs":jobs,"actual_disk_bytes":null,"shared_packed_blocks":packed,"shared_qmv_dispatches":dispatched,"cycle_calls":calls,"sampled_process_footprint_max":max_footprint,"sampled_metal_allocated_max":max_metal,"process_lifetime_peak_not_arm_peak":memory.process_lifetime_peak_bytes,"memory_start":initial_memory,"memory_end":memory}),
    )
}
#[test]
fn rotations_cover_every_arm_position() {
    for position in 0..5 {
        let mut seen = [false; 5];
        for rotation in 0..5 {
            seen[(rotation + position) % 5] = true;
        }
        assert!(seen.into_iter().all(|v| v));
    }
}
#[test]
#[ignore = "explicit matched-allocation MTP campaign; local output path, exclusive GPU, no cache override"]
fn matched_mtp_campaign() -> Result<()> {
    ensure!(
        std::env::var_os("MLXL3_SSD_CACHE_GIB").is_none()
            && std::env::var_os("MLXL3_SSD_LAYER_SLOTS").is_none(),
        "campaign forbids cache overrides"
    );
    let path = std::env::var("MLXL3_FLASH_MODEL")?;
    let output = std::env::var("MLXL3_MTP_CAMPAIGN_OUTPUT")?;
    let smoke = std::env::var("MLXL3_MTP_CAMPAIGN_SMOKE").as_deref() == Ok("1");
    let budget = if smoke { 8 } else { 128 };
    let rounds = if smoke { 1 } else { 5 };
    let c = crate::checkpoint::inspect(Path::new(&path))?;
    let tokenizer = ChatTokenizer::load(Path::new(&path))?;
    let gpu = Metal::new()?;
    ensure!(
        gpu.flash_verify_qmv.is_some(),
        "load shared projection pipelines explicitly"
    );
    let mut target = FlashNext::load(&gpu, &c, 4096)?;
    ensure!(
        target.lookahead_depth == 0 && target.layers.iter_mut().all(|l| l.moe().block_eligible()),
        "controlled cohort-compatible baseline required"
    );
    let capacities = slots(&mut target);
    let mut head = FlashMtpHead::load(&c, &target)?;
    let mut report = json!({"status":"incomplete","runtime_base":"see frozen binary/source manifest","controlled_baseline_not_shipped_best":true,"smoke":smoke,"context":4096,"budget":budget,"slots":capacities,"first_requests":[],"warmups":[],"runs":[],"actual_os_disk_traffic_measured":false,"all_hit": "not qualified by streaming campaign","draft_fixed_tensors_resident":true,"draft_experts_conditioned_for_all_arms":true,"mtp_device_greedy":head.device_greedy});
    let save = |report: &Value| -> Result<()> {
        std::fs::write(&output, serde_json::to_vec_pretty(report)?)?;
        Ok(())
    };
    save(&report)?;
    let result = (|| -> Result<()> {
        for (label, text) in PROMPTS {
            let rendered =
                tokenizer.render_values(&json!([{"role":"user","content":text}]), Some(&[]))?;
            let prompt = tokenizer.encode(&rendered)?;
            clear(&mut target, &mut head)?;
            let golden = run(
                &mut target,
                &mut head,
                &prompt,
                0,
                budget,
                tokenizer.eos_ids(),
            )?;
            let expected = golden["tokens"].clone();
            report["first_requests"].as_array_mut().unwrap().push(json!({"prompt":label,"prompt_ids":prompt,"sample":golden,"cache_state":"empty application expert cache; OS state uncontrolled"}));
            save(&report)?;
            for round in 0..=rounds {
                for position in 0..5 {
                    let arm = if round == 0 {
                        position
                    } else {
                        (round - 1 + position) % 5
                    };
                    ensure!(slots(&mut target) == capacities, "cache capacity changed");
                    // Same common conditioning, then keep expert banks and reset only sequence states.
                    clear(&mut target, &mut head)?;
                    let condition = run(
                        &mut target,
                        &mut head,
                        &prompt,
                        1,
                        budget,
                        tokenizer.eos_ids(),
                    )?;
                    ensure!(
                        condition["tokens"] == expected,
                        "conditioning differs from golden"
                    );
                    let draft_ready = head.layer.residual.moe.campaign_ready_experts();
                    ensure!(
                        draft_ready > 0,
                        "common conditioning did not warm draft experts"
                    );
                    let sample = run(
                        &mut target,
                        &mut head,
                        &prompt,
                        arm,
                        budget,
                        tokenizer.eos_ids(),
                    )?;
                    ensure!(
                        sample["tokens"] == expected,
                        "arm token sequence differs from AR golden"
                    );
                    ensure!(
                        slots(&mut target) == capacities,
                        "cache capacity changed after run"
                    );
                    let key = if round == 0 { "warmups" } else { "runs" };
                    report[key].as_array_mut().unwrap().push(
                        json!({"prompt":label,"round":round,"position":position,"draft_ready_experts_after_conditioning":draft_ready,"sample":sample}),
                    );
                    save(&report)?;
                    eprintln!("campaign {label} round{round} arm{} complete", ARMS[arm]);
                }
            }
        }
        Ok(())
    })();
    crate::ssd_reads::campaign_reads(false);
    report["status"] = if result.is_ok() {
        json!("complete")
    } else {
        json!("failed")
    };
    if let Err(error) = &result {
        report["error"] = json!(format!("{error:#}"));
    }
    save(&report)?;
    result
}
