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
            .is_some_and(|e| e == "json" || e == "safetensors")
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
    runtime_key_with_options(
        path,
        context,
        std::env::var("MLXL3_EXPERIMENTAL_DENSE_MLP_BATCH").as_deref() == Ok("1"),
        std::env::var("MLXL3_EXPERIMENTAL_GROUPED_MB3").as_deref() == Ok("1"),
        allocator_cache_mib()?,
    )
}

fn allocator_cache_mib() -> Result<Option<usize>> {
    match std::env::var("MLXL3_ALLOCATOR_CACHE_MIB") {
        Ok(value) => Ok(Some(
            value.parse().context("invalid allocator cache limit")?,
        )),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn runtime_key_with_options(
    path: &std::path::Path,
    context: usize,
    dense_batch: bool,
    grouped_mb3: bool,
    allocator_mib: Option<usize>,
) -> Result<String> {
    let allocator_key = mlxl3_native::memory_policy::allocator_cache_key(allocator_mib)
        .context("MLXL3_ALLOCATOR_CACHE_MIB must be an integer in 0..4096")?;
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
        "{}:{}:{}:{}:{}:{}:{context}:pipeline={pipeline}:adaptive={}:embedding-pack={}:allocator-cache-v1={allocator_key}:lookup={}:smallm-v1={}",
        artifact_key(path)?,
        env!("CARGO_PKG_VERSION"),
        env!("MLXL3_BUILD_REVISION"),
        env!("MLXL3_BUILD_PROFILE"),
        env!("MLXL3_MLX_VERSION"),
        hardware,
        u8::from(std::env::var("MLXL3_MTP_ADAPTIVE").as_deref() == Ok("1")),
        u8::from(std::env::var("MLXL3_EMBEDDINGS_PACKED").as_deref() == Ok("1")),
        u8::from(std::env::var("MLXL3_MTP_LOOKUP").as_deref() == Ok("1")),
        mlxl3_native::smallm_kernel_key(dense_batch, grouped_mb3)
    ))
}

#[test]
fn runtime_key_distinguishes_smallm_options() -> Result<()> {
    let directory = tempfile::tempdir()?;
    std::fs::write(directory.path().join("config.json"), "{}")?;
    let mut keys = std::collections::HashSet::new();
    for (dense, mb3, suffix) in [
        (false, false, "0"),
        (true, false, "1"),
        (false, true, "2"),
        (true, true, "3"),
    ] {
        let key =
            runtime_key_with_options(directory.path(), 4096, dense, mb3, allocator_cache_mib()?)?;
        anyhow::ensure!(key.ends_with(&format!(":smallm-v1={suffix}")));
        anyhow::ensure!(keys.insert(key), "kernel calibration key collision");
    }
    let production = runtime_key(directory.path(), 4096)?;
    anyhow::ensure!(keys.contains(&production), "production key not recognized");
    Ok(())
}

#[test]
fn runtime_key_separates_allocator_policies_and_rejects_invalid_limits() -> Result<()> {
    let directory = tempfile::tempdir()?;
    std::fs::write(directory.path().join("config.json"), "{}")?;
    let mut keys = std::collections::HashSet::new();
    for limit in [None, Some(0), Some(256), Some(4096)] {
        let key = runtime_key_with_options(directory.path(), 4096, false, false, limit)?;
        anyhow::ensure!(keys.insert(key), "allocator calibration key collision");
    }
    for invalid in [4097, usize::MAX] {
        anyhow::ensure!(
            runtime_key_with_options(directory.path(), 4096, false, false, Some(invalid)).is_err()
        );
    }
    Ok(())
}

struct Sample {
    tokens: Vec<u32>,
    seconds: f64,
    accepted: usize,
    proposed: usize,
}

#[allow(clippy::too_many_arguments)]
fn sample(
    model: &mut NativeChatModel,
    head: &mut Head,
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
    let (logits, _, _) = if depth > 0 {
        prefill_mtp(
            model, head, &input, &mut cache, "mtp-tune", false, cancelled,
        )?
    } else {
        prefill_round(
            model, &input, None, &mut cache, "mtp-tune", false, cancelled,
        )?
    };
    let mut next = logits.chat_greedy_ids()?[0];
    let mut session = (depth > 0).then(|| Session::new(depth)).transpose()?;
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
            session.advance(target, head, next, context, budget - tokens.len(), |ids| {
                tokenizer
                    .tokenizer()
                    .decode(ids, false)
                    .is_ok_and(|s| s.contains('\n'))
            })?
        } else {
            model.forward(next)?.chat_greedy_ids()?[0]
        };
    }
    Ok(Sample {
        tokens,
        seconds: started.elapsed().as_secs_f64(),
        accepted: session.as_ref().map_or(0, |s| s.accepted),
        proposed: session.as_ref().map_or(0, |s| s.proposed),
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
    const PROMPTS: [&str; 2] = [
        "Write a Python function for multiplying two matrices, with detailed comments and a complete worked example. Continue until the example is complete.",
        "Explain how to implement a least recently used cache. Give complete Python code with detailed comments and examples, then discuss its complexity.",
    ];
    let mut completed = 0;
    let progress = |depth, phase, completed| {
        emit_event(json!({"type":"mtp_tune_progress", "request_id":request,
        "depth":depth, "phase":phase, "completed":completed, "total":12}))
    };
    for depth in 0..=3 {
        progress(depth, "warmup", completed)?;
        sample(
            model, head, tokenizer, PROMPTS[0], depth, 32, context, cancelled,
        )?;
        completed += 1;
    }
    let mut samples: [Vec<Sample>; 4] = std::array::from_fn(|_| Vec::new());
    // Reverse the second pass: each candidate has the same average position.
    for (pass, prompt) in PROMPTS.into_iter().enumerate() {
        let order = if pass == 0 {
            [0, 1, 2, 3]
        } else {
            [3, 2, 1, 0]
        };
        for depth in order {
            progress(depth, "measure", completed)?;
            samples[depth].push(sample(
                model, head, tokenizer, prompt, depth, 96, context, cancelled,
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
        let tps = tokens as f64 / seconds;
        scores[depth] = mlxl3_native::speculative::mtp_tuning_score(
            depth,
            tps,
            parity && enough,
            accepted,
            proposed,
        );
        rows.push(json!({"depth":depth,"decode_tps":tps,"decode_tokens":tokens,"decode_seconds":seconds,
            "accepted_tokens":accepted,"proposed_tokens":proposed,"eligible":scores[depth].is_some(),
            "reason":if !parity { Some("target_mismatch") } else if !enough { Some("too_few_tokens") }
                else if depth > 0 && accepted == 0 { Some("zero_acceptance") } else { None },
            "token_hashes":runs.iter().map(|s| token_hash(&s.tokens)).collect::<Vec<_>>() }));
    }
    let best = mlxl3_native::speculative::best_mtp_depth(scores)
        .context("Tune MTP did not produce a valid baseline; previous setting preserved")?;
    anyhow::ensure!(!cancelled.load(Ordering::Relaxed), "MTP tuning cancelled");
    emit_event(
        json!({"type":"mtp_tune_complete","request_id":request,"tuning_key":key,
        "best_depth":best,"rows":rows,"noise_margin_percent":3,"sample_tokens":96,"samples":2}),
    )
}
