//! Resident-model tuner. No chat history, tools or reusable prompt cache.
use super::*;
use mlxl3_native::{
    mtp::{Head, Session},
    tokenizer::ChatTokenizer,
};

pub(super) fn artifact_key(path: &std::path::Path) -> Result<String> {
    use std::hash::{Hash, Hasher};
    let mut fingerprint = std::collections::hash_map::DefaultHasher::new();
    path.canonicalize()?.hash(&mut fingerprint);
    let mut files = std::fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;
    files.sort_by_key(|f| f.file_name());
    for file in files {
        if file
            .path()
            .extension()
            .is_some_and(|e| e == "json" || e == "safetensors" || e == "bin")
        {
            let meta = file.metadata()?;
            file.file_name().hash(&mut fingerprint);
            meta.len().hash(&mut fingerprint);
            meta.modified()?.hash(&mut fingerprint);
        }
    }
    Ok(format!("{:016x}", fingerprint.finish()))
}

pub(super) fn runtime_key(path: &std::path::Path, context: usize) -> Result<String> {
    let hardware = std::process::Command::new("/usr/sbin/sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_else(|| std::env::consts::ARCH.to_owned());
    let pipeline = match std::env::var("MLXL3_QWEN_PIPELINE").as_deref() {
        Ok("0") => "off",
        Ok("1") => "all",
        _ => "mtp-m5-v1",
    };
    Ok(format!(
        "{}:{}:{}:{}:{}:{}:{context}:pipeline={pipeline}:lookup={}",
        artifact_key(path)?,
        env!("CARGO_PKG_VERSION"),
        env!("MLXL3_BUILD_REVISION"),
        env!("MLXL3_BUILD_PROFILE"),
        env!("MLXL3_MLX_VERSION"),
        hardware,
        u8::from(std::env::var("MLXL3_MTP_LOOKUP").as_deref() == Ok("1"))
    ))
}

enum Draft<'a> {
    Mtp(&'a mut Head),
    DFlash(&'a mlxl3_native::dflash::DFlashWeights),
}
impl Draft<'_> {
    fn event_prefix(&self) -> &'static str {
        match self {
            Self::Mtp(_) => "mtp",
            Self::DFlash(_) => "dflash",
        }
    }
}

fn tuning_prompts(dflash: bool) -> [&'static str; 2] {
    [
        "Write a Python function for multiplying two matrices, with detailed comments and a complete worked example. Continue until the example is complete.",
        if dflash {
            "Explique en français, en trois points concis, pourquoi le ciel est bleu."
        } else {
            "Explain how to implement a least recently used cache. Give complete Python code with detailed comments and examples, then discuss its complexity."
        },
    ]
}

struct Sample {
    tokens: Vec<u32>,
    seconds: f64,
    accepted: usize,
    proposed: usize,
    lookup_blocks: usize,
}

#[allow(clippy::too_many_arguments)]
fn sample(
    model: &mut NativeChatModel,
    draft: &mut Draft<'_>,
    tokenizer: &ChatTokenizer,
    prompt: &str,
    depth: usize,
    budget: usize,
    context: usize,
    cancelled: &AtomicBool,
) -> Result<Sample> {
    let rendered =
        tokenizer.render_values(&json!([{"role":"user","content":prompt}]), Some(&[]))?;
    let input = tokenizer.encode(&rendered)?;
    anyhow::ensure!(
        input.len() + budget <= context,
        "Tune MTP needs at least {} context tokens",
        input.len() + budget
    );
    let mut cache = None;
    let (logits, mut dflash, _) = match draft {
        Draft::Mtp(head) if depth > 0 => prefill_mtp(
            model, head, &input, &mut cache, "mtp-tune", false, cancelled,
        )?,
        Draft::DFlash(weights) if depth > 0 => prefill_round(
            model,
            &input,
            Some(weights),
            &mut cache,
            "dflash-tune",
            false,
            cancelled,
        )?,
        _ => prefill_round(model, &input, None, &mut cache, "tune", false, cancelled)?,
    };
    let mut next = logits.chat_greedy_ids()?[0];
    if let (Some(session), Draft::DFlash(weights)) = (&mut dflash, &*draft) {
        session.configure(depth, weights.family, &input);
    }
    let mut session = (depth > 0 && matches!(draft, Draft::Mtp(_)))
        .then(|| Session::new(depth))
        .transpose()?;
    if std::env::var("MLXL3_MTP_LOOKUP").as_deref() == Ok("1")
        && let Some(session) = &mut session
    {
        session.enable_prompt_lookup(&input);
    }
    let mut tokens = Vec::with_capacity(budget);
    // Same denominator as chat: the first token was computed by prefill.
    let started = Instant::now();
    while tokens.len() < budget {
        anyhow::ensure!(!cancelled.load(Ordering::Relaxed), "MTP tuning cancelled");
        if tokenizer.eos_ids().contains(&next) {
            break;
        }
        tokens.push(next);
        if tokens.len() == budget {
            break;
        }
        next = if let Some(session) = &mut session {
            let NativeChatModel::Qwen(target) = model else {
                bail!("MTP requires Qwen3.5/3.6");
            };
            let Draft::Mtp(head) = draft else {
                unreachable!()
            };
            session.advance(target, head, next, context, budget - tokens.len(), |ids| {
                tokenizer
                    .tokenizer()
                    .decode(ids, false)
                    .is_ok_and(|s| s.contains('\n'))
            })?
        } else if let (Some(session), Draft::DFlash(weights)) = (&mut dflash, &mut *draft) {
            let NativeChatModel::Qwen(target) = model else {
                bail!("DFlash requires Qwen");
            };
            session.advance(target, weights, next, context, budget - tokens.len())?
        } else {
            model.forward(next)?.chat_greedy_ids()?[0]
        };
    }
    Ok(Sample {
        tokens,
        seconds: started.elapsed().as_secs_f64(),
        accepted: session
            .as_ref()
            .map_or_else(|| dflash.as_ref().map_or(0, |s| s.accepted), |s| s.accepted),
        proposed: session
            .as_ref()
            .map_or_else(|| dflash.as_ref().map_or(0, |s| s.proposed), |s| s.proposed),
        lookup_blocks: session.as_ref().map_or_else(
            || dflash.as_ref().map_or(0, |s| s.lookup_blocks),
            |s| s.lookup_blocks,
        ),
    })
}

pub(super) fn run(
    model: &mut NativeChatModel,
    head: &mut Head,
    tokenizer: &ChatTokenizer,
    request: &str,
    key: &str,
    context: usize,
    cancelled: &AtomicBool,
) -> Result<()> {
    run_modes(
        model,
        &mut Draft::Mtp(head),
        tokenizer,
        request,
        key,
        context,
        cancelled,
    )
}

pub(super) fn run_dflash(
    model: &mut NativeChatModel,
    weights: &mlxl3_native::dflash::DFlashWeights,
    tokenizer: &ChatTokenizer,
    request: &str,
    key: &str,
    context: usize,
    cancelled: &AtomicBool,
) -> Result<()> {
    run_modes(
        model,
        &mut Draft::DFlash(weights),
        tokenizer,
        request,
        key,
        context,
        cancelled,
    )
}

fn run_modes(
    model: &mut NativeChatModel,
    draft: &mut Draft<'_>,
    tokenizer: &ChatTokenizer,
    request: &str,
    key: &str,
    context: usize,
    cancelled: &AtomicBool,
) -> Result<()> {
    let prompts = tuning_prompts(matches!(draft, Draft::DFlash(_)));
    let prefix = draft.event_prefix();
    let mut completed = 0;
    let mut warmup_tokens = None;
    let progress = |depth, phase, completed| {
        emit_event(
            json!({"type":format!("{prefix}_tune_progress"), "request_id":request,
        "depth":depth, "phase":phase, "completed":completed, "total":12}),
        )
    };
    for depth in 0..=3 {
        progress(depth, "warmup", completed)?;
        let warm = sample(
            model, draft, tokenizer, prompts[0], depth, 32, context, cancelled,
        )?;
        anyhow::ensure!(!warm.tokens.is_empty(), "Tune produced an empty warmup");
        if let Some(baseline) = &warmup_tokens {
            anyhow::ensure!(
                &warm.tokens == baseline,
                "Tune warmup differs from the target"
            );
        } else {
            warmup_tokens = Some(warm.tokens);
        }
        completed += 1;
    }
    let mut samples: [Vec<Sample>; 4] = std::array::from_fn(|_| Vec::new());
    // Reverse the second pass: each candidate has the same average position.
    for (pass, prompt) in prompts.into_iter().enumerate() {
        let order = if pass == 0 {
            [0, 1, 2, 3]
        } else {
            [3, 2, 1, 0]
        };
        for depth in order {
            progress(depth, "measure", completed)?;
            samples[depth].push(sample(
                model, draft, tokenizer, prompt, depth, 96, context, cancelled,
            )?);
            completed += 1;
            progress(depth, "measured", completed)?;
        }
    }
    let mut scores = [None; 4];
    let mut rows = Vec::new();
    for depth in 0..=3 {
        let runs = &samples[depth];
        let parity = runs
            .iter()
            .zip(&samples[0])
            .all(|(a, b)| a.tokens == b.tokens);
        let enough = runs.iter().all(|s| s.tokens.len() >= 32);
        let tokens = runs
            .iter()
            .map(|s| s.tokens.len().saturating_sub(1))
            .sum::<usize>();
        let seconds = runs.iter().map(|s| s.seconds).sum::<f64>();
        let accepted = runs.iter().map(|s| s.accepted).sum::<usize>();
        let proposed = runs.iter().map(|s| s.proposed).sum::<usize>();
        let lookup_blocks = runs.iter().map(|s| s.lookup_blocks).sum::<usize>();
        let tps = tokens as f64 / seconds;
        scores[depth] = mlxl3_native::speculative::mtp_tuning_score(
            depth,
            tps,
            parity && enough,
            accepted,
            proposed,
        );
        rows.push(json!({"depth":depth,"decode_tps":tps,"decode_tokens":tokens,"decode_seconds":seconds,
            "accepted_tokens":accepted,"proposed_tokens":proposed,"lookup_blocks":lookup_blocks,"eligible":scores[depth].is_some(),
            "reason":if !parity { Some("target_mismatch") } else if !enough { Some("too_few_tokens") }
                else if depth > 0 && accepted == 0 { Some("zero_acceptance") } else { None },
            "token_hashes":runs.iter().map(|s| token_hash(&s.tokens)).collect::<Vec<_>>() }));
    }
    let best = mlxl3_native::speculative::best_mtp_depth(scores)
        .context("Tune MTP did not produce a valid baseline; previous setting preserved")?;
    anyhow::ensure!(!cancelled.load(Ordering::Relaxed), "MTP tuning cancelled");
    emit_event(
        json!({"type":format!("{prefix}_tune_complete"),"request_id":request,"tuning_key":key,
        "best_depth":best,"rows":rows,"noise_margin_percent":3,"sample_tokens":96,"samples":2}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Qwen/DFlash checkpoints, MLXL3_DFLASH_CONTEXT_COPY=1 and Apple GPU"]
    fn dflash_copy_tune_uses_the_chat_policy() -> Result<()> {
        anyhow::ensure!(
            dflash_context_copy_enabled(Some(mlxl3_native::dflash::Family::Qwen36Moe)),
            "run with MLXL3_DFLASH_CONTEXT_COPY=1"
        );
        let path = std::path::Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw");
        let mut model = NativeChatModel::load(path)?;
        let tokenizer = ChatTokenizer::load(path)?;
        let weights = mlxl3_native::dflash::DFlashWeights::load(&mlxl3_native::dflash::inspect(
            "models/Qwen3.6-35B-A3B-DFlash2",
        )?)?;
        let code = "def normalize_name(value):\n    return value.strip().casefold()\n\ndef average(values):\n    return sum(values) / len(values) if values else 0\n\n";
        let prompt = format!(
            "Recopie exactement le code suivant, sans commentaire ni changement :\n```python\n{}```",
            code.repeat(8)
        );
        let cancelled = AtomicBool::new(false);
        let mut draft = Draft::DFlash(&weights);
        let reference = sample(
            &mut model, &mut draft, &tokenizer, &prompt, 0, 256, 4096, &cancelled,
        )?;
        assert!(!reference.tokens.is_empty());
        for mode in 1..=3 {
            let actual = sample(
                &mut model, &mut draft, &tokenizer, &prompt, mode, 256, 4096, &cancelled,
            )?;
            assert_eq!(actual.tokens, reference.tokens);
            if mode == 1 {
                assert!(actual.lookup_blocks > 0);
            } else {
                assert_eq!(actual.lookup_blocks, 0);
            }
        }
        eprintln!(
            "4 Tune samples: exact nonempty tokens, copy exercised in Auto and absent from fixed modes"
        );
        Ok(())
    }

    #[test]
    fn dflash_tune_measures_code_and_french_while_mtp_keeps_its_prompts() {
        let mtp = tuning_prompts(false);
        let dflash = tuning_prompts(true);
        assert_eq!(dflash[0], mtp[0]);
        assert!(dflash[0].contains("multiplying two matrices"));
        assert!(mtp[1].contains("least recently used cache"));
        assert_eq!(
            dflash[1],
            "Explique en français, en trois points concis, pourquoi le ciel est bleu."
        );
    }
}
