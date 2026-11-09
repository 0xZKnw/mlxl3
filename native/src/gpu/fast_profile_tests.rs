//! Diagnostic host spans around the production AR/MTP entry points.
use super::*;
use anyhow::{Context, Result, ensure};
use std::sync::atomic::AtomicBool;

#[derive(serde::Serialize)]
struct DistributionMetrics {
    reference_nll: f64,
    candidate_nll: f64,
    kl: f64,
    top1_equal: bool,
    max_logit_error: f64,
}

fn distribution_metrics(
    reference: &[u16],
    candidate: &[u16],
    token: usize,
) -> Result<DistributionMetrics> {
    ensure!(!reference.is_empty() && reference.len() == candidate.len() && token < reference.len());
    let floats = |words: &[u16]| -> Result<Vec<f64>> {
        let values: Vec<_> = words
            .iter()
            .map(|&v| f64::from(half::f16::from_bits(v)))
            .collect();
        ensure!(values.iter().all(|v| v.is_finite()));
        Ok(values)
    };
    let reference = floats(reference)?;
    let candidate = floats(candidate)?;
    let log_partition = |values: &[f64]| {
        let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        max + values.iter().map(|v| (v - max).exp()).sum::<f64>().ln()
    };
    let zr = log_partition(&reference);
    let zc = log_partition(&candidate);
    let kl = reference
        .iter()
        .zip(&candidate)
        .map(|(&r, &c)| (r - zr).exp() * ((r - zr) - (c - zc)))
        .sum::<f64>();
    ensure!(zr.is_finite() && zc.is_finite() && kl.is_finite() && kl >= -1e-9);
    // Ascending scan deliberately keeps the smallest ID on ties.
    let top1 = |values: &[f64]| {
        values
            .iter()
            .enumerate()
            .fold(0, |best, (i, v)| if *v > values[best] { i } else { best })
    };
    Ok(DistributionMetrics {
        reference_nll: zr - reference[token],
        candidate_nll: zc - candidate[token],
        kl,
        top1_equal: top1(&reference) == top1(&candidate),
        max_logit_error: reference
            .iter()
            .zip(&candidate)
            .map(|(r, c)| (r - c).abs())
            .fold(0., f64::max),
    })
}

#[test]
fn full_vocabulary_metrics_detect_missing_values_divergence_and_nonfinite_logits() -> Result<()> {
    let bits = |values: &[f32]| {
        values
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect::<Vec<_>>()
    };
    let equal = distribution_metrics(&bits(&[0., 0.]), &bits(&[0., 0.]), 1)?;
    ensure!((equal.reference_nll - 2f64.ln()).abs() < 1e-12 && equal.kl == 0. && equal.top1_equal);
    let changed = distribution_metrics(&bits(&[0., 0.]), &bits(&[0., 2.]), 1)?;
    ensure!((changed.kl - 1f64.cosh().ln()).abs() < 1e-12 && !changed.top1_equal);
    ensure!((changed.candidate_nll - (1. + (-2f64).exp()).ln()).abs() < 1e-12);
    for (r, c, t) in [
        (vec![], vec![], 0),
        (bits(&[0.]), bits(&[0., 0.]), 0),
        (bits(&[0.]), bits(&[0.]), 1),
        (bits(&[f32::NAN]), bits(&[0.]), 0),
        (bits(&[0.]), bits(&[f32::INFINITY]), 0),
    ] {
        ensure!(distribution_metrics(&r, &c, t).is_err());
    }
    Ok(())
}

/// Teacher forcing keeps entire trajectories; only the model is reset between documents.
#[test]
#[ignore = "physical M5, corpus and reference directory; run serially"]
fn native_numerical_profile_teacher_forced_quality() -> Result<()> {
    use std::io::{Read, Write};
    let plan: serde_json::Value = serde_json::from_reader(std::fs::File::open(std::env::var(
        "MLXL3_NATIVE_QUALITY_CORPUS",
    )?)?)?;
    let documents = plan.as_array().context("quality corpus array")?;
    ensure!(!documents.is_empty() && documents.len() <= 32);
    let tau = std::env::var("MLXL3_NATIVE_Q04_QUALITY_TAU")
        .ok()
        .map(|v| v.parse::<f32>())
        .transpose()?;
    let limit = std::env::var("MLXL3_NATIVE_Q04_QUALITY_LIMIT")
        .unwrap_or_else(|_| "1".into())
        .parse()?;
    let quality = super::route_swap_trace::Quality::start(tau, limit)?;
    let output = std::path::PathBuf::from(std::env::var("MLXL3_NATIVE_QUALITY_OUTPUT")?);
    let reference =
        std::env::var_os("MLXL3_NATIVE_QUALITY_REFERENCE").map(std::path::PathBuf::from);
    let path = std::path::PathBuf::from(std::env::var("MLXL3_FLASH_MODEL")?);
    let checkpoint = crate::checkpoint::inspect(&path)?;
    let checkpoint_key = crate::checkpoint::artifact_key(&path)?;
    let tokenizer = crate::tokenizer::ChatTokenizer::load(&path)?;
    let gpu = Metal::new()?;
    let mut model = FlashNext::load(&gpu, &checkpoint, 4096)?;
    if reference.is_none() {
        ensure!(tau.is_none(), "Q04 cannot overwrite the exact reference");
        ensure!(model.numerical_profile() == "exact");
    }
    std::fs::create_dir_all(&output)?;
    let prior = reference
        .as_ref()
        .map(|p| -> Result<serde_json::Value> {
            let value: serde_json::Value =
                serde_json::from_reader(std::fs::File::open(p.join("report.json"))?)?;
            ensure!(
                value["status"] == "complete"
                    && value["profile"] == "exact"
                    && value["corpus"] == plan
                    && value["checkpoint"] == checkpoint_key
            );
            Ok(value)
        })
        .transpose()?;
    let mut route_capture = super::route_swap_trace::Capture::start(
        std::env::var_os("MLXL3_NATIVE_ROUTE_TRACE").map(std::path::PathBuf::from),
    )?;
    let mut results = Vec::new();
    for (index, document) in documents.iter().enumerate() {
        let text = document["text"].as_str().context("document text")?;
        let tokens = tokenizer.encode(text)?;
        let prefix = usize::try_from(
            document["prefix_tokens"]
                .as_u64()
                .context("prefix tokens")?,
        )?;
        let count = usize::try_from(document["score_tokens"].as_u64().context("score tokens")?)?;
        let end = prefix
            .checked_add(count)
            .context("document token overflow")?;
        ensure!(prefix > 0 && count > 0 && count <= 512 && end <= tokens.len() && end <= 4096);
        let tokens = &tokens[..end];
        ensure!(tokens.iter().all(|&v| v < 248320));
        if let Some(prior) = &prior {
            ensure!(prior["documents"][index]["tokens"] == serde_json::to_value(tokens)?);
        }
        let mut reader = reference
            .as_ref()
            .map(|p| std::fs::File::open(p.join(format!("{index}.f16"))))
            .transpose()?;
        if let Some(reader) = &reader {
            ensure!(reader.metadata()?.len() == (count as u64) * 248320 * 2);
        }
        let mut writer = if reference.is_none() {
            Some(std::fs::File::create(output.join(format!("{index}.f16")))?)
        } else {
            None
        };
        route_capture.document(
            index,
            document["domain"].as_str().context("domain string")?,
            end - 1,
        )?;
        quality.document(prefix, end - 1)?;
        model.reset();
        let mut logits = model.prefill(&tokens[..prefix])?;
        let mut scores = Vec::with_capacity(count);
        for (position, &token) in tokens[prefix..].iter().enumerate() {
            ensure!(
                logits.len() == 248320
                    && logits.iter().all(|&v| half::f16::from_bits(v).is_finite())
            );
            let old = if let Some(reader) = &mut reader {
                let mut bytes = vec![0; 248320 * 2];
                reader.read_exact(&mut bytes)?;
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|v| u16::from_le_bytes([v[0], v[1]]))
                    .collect::<Vec<_>>()
            } else {
                logits.clone()
            };
            if let Some(writer) = &mut writer {
                let bytes: Vec<_> = logits.iter().flat_map(|v| v.to_le_bytes()).collect();
                writer.write_all(&bytes)?;
            }
            if route_capture.active() && reference.is_some() {
                ensure!(
                    old == logits,
                    "route instrumentation changed exact target logits"
                );
            }
            scores.push(distribution_metrics(&old, &logits, token as usize)?);
            if position + 1 < count {
                logits = model.forward(token)?;
            }
        }
        if let Some(reader) = &mut reader {
            let mut byte = [0];
            ensure!(reader.read(&mut byte)? == 0);
        }
        if let Some(writer) = &mut writer {
            writer.flush()?;
        }
        ensure!(scores.len() == count);
        results
            .push(serde_json::json!({"domain":document["domain"],"tokens":tokens,"scores":scores}));
    }
    ensure!(results.len() == documents.len() && gpu.readers()?.in_flight() == 0);
    route_capture.finish()?;
    let policy = quality.finish()?;
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"status":"complete","profile":if tau.is_some() {"q04-ar-diagnostic"} else {model.numerical_profile()},"policy":policy,"checkpoint":checkpoint_key,"corpus":plan,"documents":results}),
        )?,
    )?;
    Ok(())
}

#[test]
#[ignore = "physical Flash checkpoint, serial Q04 AR prototype timing"]
fn native_q04_ar_generation_pilot() -> Result<()> {
    let mtp = match std::env::var("MLXL3_NATIVE_Q04_SPEED_MTP").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(_) => false,
        _ => anyhow::bail!("invalid MTP pilot mode"),
    };
    let path = std::path::PathBuf::from(std::env::var("MLXL3_FLASH_MODEL")?);
    let checkpoint = crate::checkpoint::inspect(&path)?;
    let tokenizer = crate::tokenizer::ChatTokenizer::load(&path)?;
    let prompt = "Explique en français, avec un exemple concret, comment un cache LFU évite les lectures SSD inutiles pendant l’inférence d’un modèle MoE.";
    let rendered = tokenizer.render_values_with_thinking(
        &serde_json::json!([{"role":"user","content":prompt}]),
        None,
        false,
    )?;
    let tokens = tokenizer.encode(&rendered)?;
    ensure!(!tokens.is_empty() && tokens.len() + 63 <= 4096);
    let tau = std::env::var("MLXL3_NATIVE_Q04_QUALITY_TAU")
        .ok()
        .map(|v| v.parse::<f32>())
        .transpose()?;
    let gpu = Metal::new()?;
    let mut model = FlashNext::load(&gpu, &checkpoint, 4096)?;
    let mut head = mtp
        .then(|| FlashMtpHead::load(&checkpoint, &model))
        .transpose()?;
    let mut session = FlashMtpSession::default();
    let depth = std::env::var("MLXL3_NATIVE_Q04_MTP_DEPTH")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    let block = std::env::var("MLXL3_NATIVE_Q04_MTP_BLOCK")
        .ok()
        .map(|v| v.parse::<usize>())
        .transpose()?
        .unwrap_or(1);
    ensure!((1..=4).contains(&depth) && (1..=3).contains(&block));
    let experimental = mtp && (depth > 1 || block > 1);
    let q04_oracle = match std::env::var("MLXL3_NATIVE_Q04_ORACLE") {
        Ok(value) => crate::flash_support::hc_chain_flag(Some(&value), false)
            .context("MLXL3_NATIVE_Q04_ORACLE must be 0 or 1")?,
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => return Err(error.into()),
    };
    let cancelled = AtomicBool::new(false);
    ensure!(model.device_greedy_eligible(0.0, 1, 1.0, false));
    let mut runs = Vec::new();
    let mut retained_cache = None;
    for warmup in [true, false] {
        let mut pilot = experimental
            .then(|| super::flash::ExperimentalMtp::new(depth, block, &gpu))
            .transpose()?;
        let quality = super::route_swap_trace::Quality::start_with_oracle(tau, 3, q04_oracle)?;
        quality.prepare(&gpu)?;
        super::route_swap_trace::restore_cache_only(retained_cache.as_ref());
        quality.document(tokens.len(), tokens.len() + 63)?;
        model.reset();
        let hot_before = gpu.device_hot_submissions();
        let start = std::time::Instant::now();
        let logits = if let Some(head) = &mut head {
            session.prefill(&mut model, head, &tokens, &cancelled)?
        } else {
            model.prefill(&tokens)?
        };
        let mut generated = vec![model.greedy_from_logits(&logits)?];
        let prefill_seconds = start.elapsed().as_secs_f64();
        let read_before = gpu.readers()?.requested_counts();
        let decode = std::time::Instant::now();
        while generated.len() < 64 {
            let last = *generated.last().context("missing generated token")?;
            ensure!(
                !tokenizer.eos_ids().contains(&last),
                "early EOS makes this pilot incomparable"
            );
            let next = if let Some(head) = &mut head {
                if let Some(pilot) = &mut pilot {
                    pilot.advance(
                        &mut model,
                        head,
                        last,
                        64 - generated.len(),
                        tokenizer.eos_ids(),
                        &cancelled,
                    )?
                } else {
                    session.advance_with_eos_cancellable(
                        &mut model,
                        head,
                        last,
                        4096,
                        64 - generated.len(),
                        tokenizer.eos_ids(),
                        &cancelled,
                    )?
                }
            } else {
                model.forward_greedy(last)?
            };
            generated.push(next);
        }
        let decode_seconds = decode.elapsed().as_secs_f64();
        let total_seconds = start.elapsed().as_secs_f64();
        let read_after = gpu.readers()?.requested_counts();
        let hot_submissions = gpu.device_hot_submissions() - hot_before;
        ensure!(generated.len() == 64 && generated.iter().all(|&t| t < 248320));
        ensure!(model.offset() as usize == tokens.len() + 63 && gpu.readers()?.in_flight() == 0);
        ensure!(decode_seconds.is_finite() && decode_seconds > 0.0 && prefill_seconds.is_finite());
        let text = tokenizer.decode(&generated)?;
        ensure!(!text.trim().is_empty());
        retained_cache = super::route_swap_trace::decision_snapshot();
        let policy = quality.finish()?;
        if let Some(policy) = &policy {
            ensure!(
                policy["retained_layer_rows"] == 3024
                    && policy["rows"].as_u64().is_some_and(|n| n >= 3024)
                    && policy["swaps"].as_u64().is_some_and(|n| n > 0)
            );
        }
        runs.push(
            serde_json::json!({"warmup":warmup,"tokens":generated,"text":text,
            "prefill_tokens":tokens.len(),"prefill_seconds":prefill_seconds,
            "ttft_seconds":prefill_seconds,"decode_tokens":63,"decode_seconds":decode_seconds,
            "decode_tokens_per_second":63.0/decode_seconds,"total_seconds":total_seconds,
            "policy":policy,"device_hot_submissions":hot_submissions,"memory":model.memory_stats()?,
            "mtp_proposed":pilot.as_ref().map_or(session.proposed,|p|p.proposed),
            "mtp_accepted":pilot.as_ref().map_or(session.accepted,|p|p.accepted),
            "ssd_records_requested":read_after[0]-read_before[0],"ssd_bytes_requested":read_after[1]-read_before[1],
            "active_context_final":model.offset(),"context_capacity":4096,
            "mtp_work":pilot.as_ref().map(|p|&p.stats),
            "verify_widths":pilot.as_ref().map(|p|p.widths),"draft_lengths":pilot.as_ref().map(|p|p.lengths)}),
        );
    }
    std::fs::write(
        std::env::var("MLXL3_NATIVE_Q04_SPEED_OUTPUT")?,
        serde_json::to_vec_pretty(
            &serde_json::json!({"status":if experimental {"complete_native_mtp_depth_prototype_pilot"} else if mtp {"complete_native_mtp_d1_prototype_pilot"} else {"complete_native_ar_prototype_pilot"},"tau":tau,
            "checkpoint":crate::checkpoint::artifact_key(&path)?,"prompt":rendered,"prompt_tokens":tokens,
            "qmv_shapes":&*gpu.qmv_shapes.borrow(),"mb3":gpu.flash_verify_mb3.as_ref().map(|p|p.resources(&gpu.device)),"qmv_observation":gpu.observe_qmv,
            "q04_oracle":q04_oracle,"recursive_cpu":std::env::var("MLXL3_NATIVE_MTP_RECURSIVE_CPU").ok(),
            "max_swaps":3,"mtp":mtp,"draft_depth":if mtp {depth} else {0},"verify_block_max_rows":block,"experimental_mtp":experimental,"production_cli_benchmark":false,"runs":runs}),
        )?,
    )?;
    Ok(())
}

#[test]
#[ignore = "physical M5, local Flash checkpoint; run serially"]
fn current_native_ar_and_mtp_round_profile() -> Result<()> {
    let out = std::path::PathBuf::from(std::env::var("MLXL3_NATIVE_PROFILE_OUTPUT")?);
    std::fs::create_dir_all(&out)?;
    let path = std::path::PathBuf::from(std::env::var("MLXL3_FLASH_MODEL")?);
    let checkpoint = crate::checkpoint::inspect(&path)?;
    let tokenizer = crate::tokenizer::ChatTokenizer::load(&path)?;
    let prompt = tokenizer.encode(&tokenizer.render_values(&serde_json::json!([
        {"role":"user", "content":"Explique en français, avec un exemple concret, comment un cache LFU évite les lectures SSD inutiles pendant l’inférence d’un modèle MoE."}
    ]), None)?)?;
    ensure!(prompt.len() == 87);
    let cancelled = AtomicBool::new(false);
    let mut modes = Vec::new();
    for mtp in [false, true] {
        let gpu = Metal::new()?;
        ensure!(gpu.device_name() == "Apple M5");
        let mut target = FlashNext::load(&gpu, &checkpoint, 4096)?;
        let mut head = mtp
            .then(|| FlashMtpHead::load(&checkpoint, &target))
            .transpose()?;
        let mut expected = Vec::new();
        for observed in [false, true] {
            target.reset();
            let mut session = FlashMtpSession::default();
            let logits = if let Some(head) = &mut head {
                session.prefill(&mut target, head, &prompt, &cancelled)?
            } else {
                target.prefill(&prompt)?
            };
            ensure!(
                logits.len() == 248320
                    && logits.iter().all(|&v| half::f16::from_bits(v).is_finite())
            );
            let mut token = target.greedy_from_logits(&logits)?;
            let mut tokens = vec![token];
            if observed {
                crate::flash_profile::start(false)?;
            }
            for index in 1..8 {
                let _round = crate::flash_profile::span("native.production_output_step");
                token = if let Some(head) = &mut head {
                    session.advance_with_eos_cancellable(
                        &mut target,
                        head,
                        token,
                        4096,
                        8 - index,
                        &[],
                        &cancelled,
                    )?
                } else {
                    target.forward_greedy(token)?
                };
                ensure!(token < 248320);
                tokens.push(token);
            }
            if observed {
                let events = crate::flash_profile::finish()?;
                ensure!(tokens == expected && tokens.len() == 8 && !events.is_empty());
                let events = serde_json::to_value(events)?;
                ensure!(
                    events
                        .as_array()
                        .context("profile events")?
                        .iter()
                        .filter(|v| v["name"] == "native.production_output_step")
                        .count()
                        == 7
                );
                std::fs::write(
                    out.join(if mtp {
                        "mtp-events.json"
                    } else {
                        "ar-events.json"
                    }),
                    serde_json::to_vec(&events)?,
                )?;
                modes.push(serde_json::json!({"mtp":mtp,"tokens":tokens,"proposed":session.proposed,"accepted":session.accepted,"memory":target.memory_stats()?,"event_count":events.as_array().unwrap().len()}));
            } else {
                expected = tokens;
            }
        }
        ensure!(gpu.readers()?.in_flight() == 0);
    }
    ensure!(modes.len() == 2 && modes[0]["tokens"] == modes[1]["tokens"]);
    std::fs::write(
        out.join("report.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"status":"validated_diagnostic_only","modes":modes,"exclusive_gpu_time":false}),
        )?,
    )?;
    Ok(())
}
