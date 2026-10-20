//! Physical diagnostic runner, separate from release throughput.
use super::*;
use crate::flash_profile as profile;
use std::path::Path;
use std::{
    io::BufWriter,
    process::{Command, Stdio},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
fn epoch() -> Result<f64> {
    Ok(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64())
}
fn snapshot() -> Result<Value> {
    let mut child = Command::new("/usr/bin/vm_stat")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    let deadline = Instant::now() + Duration::from_secs(2);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {}
            Err(error) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error.into());
            }
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("vm_stat deadline exceeded");
        }
        std::thread::sleep(Duration::from_millis(2));
    }
    let output = child.wait_with_output()?;
    ensure!(output.status.success(), "vm_stat failed");
    let memory = crate::array::memory_stats(false)?;
    ensure!(
        memory
            .process_lifetime_peak_bytes
            .is_some_and(|v| v <= 12_000_000_000),
        "physical footprint limit exceeded"
    );
    Ok(
        serde_json::json!({"epoch":epoch()?,"memory":memory,"vm_stat":String::from_utf8(output.stdout)?}),
    )
}
fn counts(t: &FlashNext) -> Vec<[u64; 3]> {
    t.layers
        .iter()
        .map(|l| [l.moe.hits, l.moe.misses, l.moe.read_bytes])
        .collect()
}
fn save(p: &Path, v: &impl serde::Serialize) -> Result<()> {
    let file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(p)?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, v)?;
    std::io::Write::flush(&mut writer)?;
    Ok(())
}
#[test]
#[ignore = "physical Apple GPU, full Flash checkpoint; isolated diagnostic, not release throughput"]
fn flash_detailed_profile_preserves_full_outputs_and_states() -> Result<()> {
    let directory = PathBuf::from(std::env::var("MLXL3_PROFILE_OUTPUT")?);
    let kind = std::env::var("MLXL3_PROFILE_KIND")?;
    let mtp = match std::env::var("MLXL3_FLASH_PROFILE_MTP")?.as_str() {
        "0" => false,
        "1" => true,
        _ => anyhow::bail!("invalid mode"),
    };
    let label = std::env::var("MLXL3_FLASH_PROFILE_PROMPT")?;
    let prompt = match label.as_str() {
        "MAIN" => {
            "Explique en français, avec un exemple concret, comment un cache LFU évite les lectures SSD inutiles pendant l’inférence d’un modèle MoE."
        }
        "HOLD" => {
            "Écris une fonction Python merge_intervals qui fusionne les intervalles qui se chevauchent, puis explique sa complexité et donne un exemple."
        }
        _ => anyhow::bail!("invalid prompt"),
    };
    let (budget, phases): (usize, &[(&str, bool, bool)]) = match kind.as_str() {
        "natural" => (
            128,
            &[
                ("warmup", false, false),
                ("control_before", false, false),
                ("trace_1", true, false),
                ("trace_2", true, false),
                ("trace_3", true, false),
                ("control_after", false, false),
            ],
        ),
        "fenced" => (
            33,
            &[
                ("warmup", false, false),
                ("oracle", false, false),
                ("fenced_1", true, true),
                ("fenced_2", true, true),
            ],
        ),
        "quality" => (
            17,
            &[
                ("warmup", false, false),
                ("oracle", false, false),
                ("trace", true, false),
                ("fenced", true, true),
            ],
        ),
        _ => anyhow::bail!("invalid profile kind"),
    };
    ensure!(directory.is_dir(), "output directory required");
    save(
        &directory.join("started.json"),
        &serde_json::json!({"status":"incomplete","kind":kind,"mtp":mtp,"prompt":label,"budget":budget,"epoch":epoch()?}),
    )?;
    let path = PathBuf::from(std::env::var("MLXL3_FLASH_MODEL")?);
    let load_start = Instant::now();
    let checkpoint = crate::checkpoint::inspect(&path)?;
    let tokenizer = crate::tokenizer::ChatTokenizer::load(&path)?;
    let prompt_ids = tokenizer.encode(&tokenizer.render(&[crate::tokenizer::Message {
        role: "user".into(),
        content: prompt.into(),
    }])?)?;
    let mut target = FlashNext::from_checkpoint(&checkpoint)?;
    let mut head = if mtp {
        Some(FlashMtpHead::load(&checkpoint, &target)?)
    } else {
        None
    };
    let load_seconds = load_start.elapsed().as_secs_f64();
    ensure!(!prompt_ids.is_empty() && prompt_ids.len() + budget <= target.context_limit() as usize);
    let mut expected = None;
    let mut summaries = Vec::new();
    for &(phase, traced, fenced) in phases {
        target.reset();
        if let Some(h) = &mut head {
            h.reset();
        }
        let before = snapshot()?;
        let before_counts = counts(&target);
        crate::array::disk_profile_enable(traced);
        if traced {
            profile::start(fenced)?;
        }
        let prefill_epoch = epoch()?;
        let start = Instant::now();
        let mut logits = None;
        {
            let _span = profile::span("prefill");
            for (step, &token) in prompt_ids.iter().enumerate() {
                let step_span = profile::span("prefill_step");
                if let Some(s) = &step_span {
                    s.data(serde_json::json!({"step":step,"token":token}));
                }
                if let (Some(h), Some(hidden)) = (&mut head, &target.last_hidden) {
                    let _h = profile::span("prefill_head_extend");
                    h.extend_cache(&target, hidden, token)?;
                }
                logits = Some(target.forward(token)?);
            }
        }
        let mut anchor = logits.context("empty prefill")?.chat_greedy_ids()?[0];
        let prefill_seconds = start.elapsed().as_secs_f64();
        let prefill_events = if traced {
            profile::finish()?
        } else {
            Vec::new()
        };
        let prefill_counts = counts(&target);
        let prefill_snapshot = snapshot()?;
        let mut output = vec![anchor];
        let mut session = FlashMtpSession::default();
        let mut steps = Vec::with_capacity(budget - 1);
        if traced {
            profile::start(fenced)?;
        }
        let decode_epoch = epoch()?;
        let start = Instant::now();
        {
            let _decode = profile::span("decode");
            for step in 1..budget {
                let mark = Instant::now();
                let step_span = profile::span("output_step");
                let old = (session.proposed, session.accepted, session.blocks);
                anchor = if let Some(h) = &mut head {
                    session.advance(
                        &mut target,
                        h,
                        anchor,
                        prompt_ids.len() + budget - 1,
                        budget - step,
                    )?
                } else {
                    target.forward(anchor)?.chat_greedy_ids()?[0]
                };
                let seconds = mark.elapsed().as_secs_f64();
                let item = serde_json::json!({"step":step,"token":anchor,"offset":target.offset(),"seconds":seconds,"elapsed_seconds":start.elapsed().as_secs_f64(),"mtp_proposed":session.proposed-old.0,"mtp_accepted":session.accepted-old.1,"mtp_blocks":session.blocks-old.2});
                if let Some(s) = &step_span {
                    s.data(item.clone());
                }
                steps.push(item);
                output.push(anchor);
            }
            target.drain()?;
        }
        let decode_seconds = start.elapsed().as_secs_f64();
        let decode_events = if traced {
            profile::finish()?
        } else {
            Vec::new()
        };
        crate::array::disk_profile_enable(false);
        let after = snapshot()?;
        let final_counts = counts(&target);
        ensure!(
            output.len() == budget
                && session.pending.is_empty()
                && target.offset() as usize == prompt_ids.len() + budget - 1
        );
        if let Some(h) = &head {
            ensure!(h.offset() == target.offset() - 1);
        }
        let logits = target.logits(
            target
                .last_hidden
                .as_ref()
                .context("missing final hidden")?,
        )?;
        let host = logits.to_f32()?;
        ensure!(
            logits.shape() == [1, 248320]
                && host.len() == 248320
                && host.iter().all(|v| v.is_finite())
        );
        let state = super::tests::state_bytes(&target)?;
        ensure!(state.len() > 100 && state.iter().all(|s| !s.is_empty()));
        let head_state = if let Some(h) = &head {
            let Attention::Qsa(q) = &h.layer.attention else {
                anyhow::bail!("invalid head")
            };
            super::tests::qsa_bytes(q)?
        } else {
            Vec::new()
        };
        let parity = (output.clone(), logits.to_bytes()?, state, head_state);
        if let Some(e) = &expected {
            ensure!(&parity == e, "full profile parity failed: {phase}");
        } else {
            expected = Some(parity);
        }
        ensure!(!traced || (!prefill_events.is_empty() && !decode_events.is_empty()));
        let hash = output
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .fold(0xcbf29ce484222325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
            });
        let summary = serde_json::json!({"phase":phase,"traced":traced,"fenced":fenced,"mtp":mtp,"prompt":label,"prompt_ids":prompt_ids,"tokens":output,"token_hash":format!("{hash:016x}"),"text":tokenizer.decode(&output)?,"prefill_epoch":prefill_epoch,"decode_epoch":decode_epoch,"prefill_seconds":prefill_seconds,"decode_seconds":decode_seconds,"diagnostic_tps":(budget-1)as f64/decode_seconds,"before_counts":before_counts,"prefill_counts":prefill_counts,"after_counts":final_counts,"before":before,"prefill_boundary":prefill_snapshot,"after":after,"steps":steps,"mtp_proposed":session.proposed,"mtp_accepted":session.accepted,"mtp_blocks":session.blocks,"mtp_head_verify_repair_total_seconds":session.profile_seconds,"logit_count":host.len(),"full_logits_target_head_states_byte_exact":true,"prefill_event_count":prefill_events.len(),"decode_event_count":decode_events.len()});
        save(&directory.join(format!("{phase}-summary.json")), &summary)?;
        if traced {
            save(
                &directory.join(format!("{phase}-prefill.json")),
                &prefill_events,
            )?;
            save(
                &directory.join(format!("{phase}-decode.json")),
                &decode_events,
            )?;
        }
        println!(
            "FLASH_DETAILED_PHASE {}",
            serde_json::json!({"phase":phase,"diagnostic_tps":summary["diagnostic_tps"],"decode_seconds":decode_seconds,"events":prefill_events.len()+decode_events.len(),"epoch":epoch()?})
        );
        summaries.push(summary);
    }
    let final_snapshot = snapshot()?;
    save(&directory.join("final-snapshot.json"), &final_snapshot)?;
    save(
        &directory.join("complete.json"),
        &serde_json::json!({"status":"complete_diagnostic_internal_full_parity_external_golden_pending","load_seconds":load_seconds,"kind":kind,"mtp":mtp,"prompt":label,"summaries":summaries,"limits":"nested wall intervals overlap; natural route readback includes upstream work; fenced times include dispatch and waits, not exclusive GPU; VM global; serialization and snapshots outside phase clocks"}),
    )?;
    Ok(())
}
#[cfg(kani)]
#[kani::proof]
fn actual_detailed_profile_path() {
    flash_detailed_profile_preserves_full_outputs_and_states().unwrap();
}
