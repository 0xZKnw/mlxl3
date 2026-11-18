//! Q27D-2026-11-17-02: decode-step anatomy of the dense Qwen27B. Diagnostic
//! only: nothing here runs in production and no production path is changed.
use super::*;
use std::time::Instant;

type Chain<'a> = Box<dyn Fn(&Array) -> Result<Array> + 'a>;
type Trace = Vec<(Vec<u16>, Vec<Vec<u8>>)>;

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

/// Deterministic finite F16 row in [-1, 1].
fn fixed(width: i32, seed: u32) -> Result<Array> {
    let bits = (0..width as u32)
        .map(|i| {
            let mixed = i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 8;
            f16::from_f32((mixed % 2001) as f32 / 1000.0 - 1.0).to_bits()
        })
        .collect::<Vec<_>>();
    let value = Array::from_f16_bits(&bits, &[1, width])?;
    value.eval()?;
    Ok(value)
}

fn evaluate(outputs: &[Array]) -> Result<()> {
    Array::async_eval_all(&outputs.iter().collect::<Vec<_>>())?;
    crate::array::synchronize()
}

/// Median wall time of `build` evaluated in one GPU submission, after two warm runs.
fn timed(mut build: impl FnMut() -> Result<Vec<Array>>) -> Result<(f64, usize)> {
    let mut samples = Vec::new();
    let mut count = 0;
    for round in 0..7 {
        crate::array::synchronize()?;
        let start = Instant::now();
        let outputs = build()?;
        evaluate(&outputs)?;
        let elapsed = ms(start);
        count = outputs.len();
        ensure!(count > 0, "empty anatomy class");
        if round >= 2 {
            samples.push(elapsed);
        }
    }
    Ok((median(samples), count))
}

fn class_of(name: &str) -> Option<&'static str> {
    let stem = name.strip_suffix(".trellis")?;
    Some(if stem.ends_with("lm_head") {
        "head"
    } else if stem.ends_with("linear_attn.in_proj_qkv") || stem.ends_with("linear_attn.in_proj_z") {
        "gdn_inputs"
    } else if stem.ends_with("in_proj_b") || stem.ends_with("in_proj_a") {
        "gdn_ab"
    } else if stem.ends_with("linear_attn.out_proj") {
        "gdn_out"
    } else if stem.ends_with("q_proj") || stem.ends_with("k_proj") || stem.ends_with("v_proj") {
        "attn_qkv"
    } else if stem.ends_with("o_proj") {
        "attn_out"
    } else if stem.ends_with("gate_proj") || stem.ends_with("up_proj") {
        "mlp_inputs"
    } else if stem.ends_with("down_proj") {
        "mlp_down"
    } else {
        return None;
    })
}

#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_ANATOMY; run alone"]
fn q27_decode_step_anatomy() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_ANATOMY")?);
    ensure!(!output.exists(), "preserve previous anatomy reports");
    let path = std::path::PathBuf::from(std::env::var("MLXL3_QWEN_TEST_MODEL")?);
    let checkpoint = crate::checkpoint::inspect(&path)?;
    let mut bytes = std::collections::BTreeMap::<&str, (u64, usize, [usize; 9])>::new();
    for (name, info) in &checkpoint.tensors {
        if let Some(class) = class_of(name) {
            let entry = bytes.entry(class).or_default();
            entry.0 += info.data_offsets[1] - info.data_offsets[0];
            entry.1 += 1;
            let k = info.shape.last().copied().unwrap_or(0) / 16;
            ensure!((1..=8).contains(&k), "{name}: unexpected trellis width");
            entry.2[k] += 1;
        }
    }
    let mut model = Qwen35Moe::from_checkpoint(&checkpoint)?;
    ensure!(
        model.layers.len() == 64 && model.vocab == 248320 && model.mtp_layout.hidden_size == 5120,
        "requires dense Qwen27B fixture"
    );
    let mut report = serde_json::json!({
        "status": "running",
        "pipeline": model.pipeline, "pipeline_stride": model.pipeline_stride,
        "trellis_bytes": bytes.iter().map(|(class, (size, count, k))| {
            (class.to_string(), serde_json::json!({"bytes": size, "tensors": count, "k_histogram": k}))
        }).collect::<serde_json::Map<_, _>>(),
    });
    let save = |report: &serde_json::Value| -> Result<()> {
        std::fs::write(&output, serde_json::to_vec_pretty(report)?)?;
        Ok(())
    };
    save(&report)?;

    // (a) Shipped autoregressive step after a 69-token prefix.
    let prefix = (0..69).map(|i| (i * 37 + 1) as u32).collect::<Vec<_>>();
    model.forward_mtp(&prefix)?;
    crate::array::synchronize()?;
    let snapshot = model.snapshot()?;
    let step_tokens = (0..40).map(|i| (i * 53 + 11) as u32).collect::<Vec<_>>();
    let mut ar = Vec::new();
    for (index, &token) in step_tokens.iter().enumerate() {
        let start = Instant::now();
        model.forward(token)?;
        crate::array::synchronize()?;
        if index >= 8 {
            ar.push(ms(start));
        }
    }
    let ar_ms = median(ar.clone());
    report["ar_step_ms"] = serde_json::json!({"median": ar_ms, "samples": ar});
    save(&report)?;

    // (b) Pipeline off: lazy graph construction, then evaluation.
    model.restore(snapshot.clone())?;
    let shipped = (model.pipeline, model.mtp_pipeline);
    model.pipeline = false;
    model.mtp_pipeline = false;
    let (mut build, mut eval) = (Vec::new(), Vec::new());
    for (index, &token) in step_tokens.iter().enumerate() {
        crate::array::synchronize()?;
        let start = Instant::now();
        let (logits, _) = model.run_tokens(&[token], false, false)?;
        let built = ms(start);
        logits.eval()?;
        crate::array::synchronize()?;
        if index >= 8 {
            build.push(built);
            eval.push(ms(start) - built);
        }
    }
    report["pipeline_off"] = serde_json::json!({
        "graph_build_ms": median(build), "evaluate_ms": median(eval)});
    (model.pipeline, model.mtp_pipeline) = shipped;
    save(&report)?;

    // (c) Every projection of one decode step on fixed inputs, one submission per class.
    let x5120 = fixed(5120, 1)?;
    let x6144 = fixed(6144, 2)?;
    let x17408 = fixed(17408, 3)?;
    let mut classes = serde_json::Map::new();
    type Build<'a> = Box<dyn Fn(&Qwen35Moe) -> Result<Vec<Array>> + 'a>;
    fn dense(layer: &Layer) -> Result<&DenseMlp> {
        match layer.mlp() {
            Mlp::Dense(mlp) => Ok(mlp),
            Mlp::Moe(_) => anyhow::bail!("dense MLP expected"),
        }
    }
    let builders: Vec<(&str, Build)> = vec![
        (
            "gdn_inputs",
            Box::new(|model| {
                let mut out = Vec::new();
                for layer in &model.layers {
                    if let Layer::Linear(layer) = layer {
                        out.extend(layer.attention.inputs.forward(&x5120)?);
                    }
                }
                Ok(out)
            }),
        ),
        (
            "gdn_ab",
            Box::new(|model| {
                let mut out = Vec::new();
                for layer in &model.layers {
                    if let Layer::Linear(layer) = layer {
                        out.push(layer.attention.b.forward(&x5120)?);
                        out.push(layer.attention.a.forward(&x5120)?);
                    }
                }
                Ok(out)
            }),
        ),
        (
            "gdn_out",
            Box::new(|model| {
                let mut out = Vec::new();
                for layer in &model.layers {
                    if let Layer::Linear(layer) = layer {
                        out.push(layer.attention.output.forward(&x6144)?);
                    }
                }
                Ok(out)
            }),
        ),
        (
            "attn_qkv",
            Box::new(|model| {
                let mut out = Vec::new();
                for layer in &model.layers {
                    if let Layer::Attention(layer) = layer {
                        out.extend(layer.attention.qkv.forward(&x5120)?);
                    }
                }
                Ok(out)
            }),
        ),
        (
            "attn_out",
            Box::new(|model| {
                let mut out = Vec::new();
                for layer in &model.layers {
                    if let Layer::Attention(layer) = layer {
                        out.push(layer.attention.output.forward(&x6144)?);
                    }
                }
                Ok(out)
            }),
        ),
        (
            "mlp_inputs",
            Box::new(|model| {
                let mut out = Vec::new();
                for layer in &model.layers {
                    out.extend(dense(layer)?.inputs.forward(&x5120)?);
                }
                Ok(out)
            }),
        ),
        (
            "mlp_down",
            Box::new(|model| {
                let mut out = Vec::new();
                for layer in &model.layers {
                    out.push(dense(layer)?.down.forward(&x17408)?);
                }
                Ok(out)
            }),
        ),
        (
            "head",
            Box::new(|model| Ok(vec![model.head.forward(&x5120)?])),
        ),
    ];
    let mut projections_ms = 0.0;
    for (class, build) in &builders {
        let (elapsed, outputs) = timed(|| build(&model))?;
        let size = bytes.get(class).map_or(0, |entry| entry.0);
        projections_ms += elapsed;
        classes.insert(
            class.to_string(),
            serde_json::json!({
            "ms": elapsed, "outputs": outputs, "trellis_bytes": size,
            "gib_per_second": size as f64 / (1u64 << 30) as f64 / (elapsed / 1e3)}),
        );
    }
    let (all_ms, _) = timed(|| {
        let mut out = Vec::new();
        for (_, build) in &builders {
            out.extend(build(&model)?);
        }
        Ok(out)
    })?;
    report["projections"] = serde_json::json!({
        "classes": classes, "sum_of_classes_ms": projections_ms, "all_in_one_submission_ms": all_ms});
    save(&report)?;

    // (c') The MLX prologue/epilogue of the same projections without their QMV kernel.
    let scale5120 = fixed(5120, 4)?.reshape(&[5120])?;
    let lane = Array::from_i32(
        &(0..5120)
            .map(|i| crate::contracts::exl3_lane_source_index(i as u32) as i32)
            .collect::<Vec<_>>(),
        &[5120],
    )?;
    let prologue = |x: &Array, scale: &Array, width: i32| -> Result<Array> {
        x.astype(Dtype::Float16)?
            .reshape(&[1, width])?
            .mul(scale)?
            .reshape(&[1, width / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[width])
    };
    let epilogue = |partials: &Array, scale: &Array, width: i32| -> Result<Array> {
        partials
            .sum(0, false)?
            .astype(Dtype::Float16)?
            .reshape(&[1, width / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[width])?
            .mul(scale)
    };
    let partial = |splits: i32, width: i32, seed: u32| -> Result<Array> {
        let value = Array::concatenate(
            &(0..splits)
                .map(|s| fixed(width, seed + s as u32))
                .collect::<Result<Vec<_>>>()?
                .iter()
                .collect::<Vec<_>>(),
            0,
        )?
        .astype(Dtype::Float32)?;
        value.eval()?;
        Ok(value)
    };
    let p17408 = partial(2, 17408 * 2, 10)?;
    let s34816 = fixed(34816, 5)?.reshape(&[34816])?;
    let p5120 = partial(2, 5120, 20)?;
    let s17408 = fixed(17408, 6)?.reshape(&[17408])?;
    let (mlp_glue_ms, _) = timed(|| {
        let mut out = Vec::new();
        for _ in 0..64 {
            out.push(prologue(&x5120, &scale5120, 5120)?.take(&lane, 0)?);
            out.push(epilogue(&p17408, &s34816, 34816)?);
            out.push(prologue(&x17408, &s17408, 17408)?);
            out.push(epilogue(&p5120, &scale5120, 5120)?);
        }
        Ok(out)
    })?;
    report["mlp_glue_only_64_layers_ms"] = mlp_glue_ms.into();
    save(&report)?;

    // (d) Cost of one small MLX primitive inside a long dependent chain.
    let mut micro = serde_json::Map::new();
    let one = Array::from_f16_bits(&[f16::ONE.to_bits()], &[])?;
    let chains: Vec<(&str, Chain)> = vec![
        ("mul_f16_5120", Box::new(|x| x.mul(&one))),
        (
            "hadamard_f16_5120",
            Box::new(|x| {
                x.reshape(&[1, 40, 128])?
                    .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
                    .reshape(&[1, 5120])
            }),
        ),
        (
            "astype_f32_f16_5120",
            Box::new(|x| x.astype(Dtype::Float32)?.astype(Dtype::Float16)),
        ),
        (
            "take_lane_5120",
            Box::new(|x| x.reshape(&[5120])?.take(&lane, 0)?.reshape(&[1, 5120])),
        ),
    ];
    for (name, step) in &chains {
        let (elapsed, _) = timed(|| {
            let mut value = x5120.try_clone()?;
            for _ in 0..1024 {
                value = step(&value)?;
            }
            Ok(vec![value])
        })?;
        micro.insert(
            name.to_string(),
            serde_json::json!({"chain_1024_ms": elapsed, "microseconds_each": elapsed}),
        );
    }
    report["micro_chain"] = micro.into();
    save(&report)?;

    // (e) Verification pass of M rows (commit included) relative to the AR step.
    let mut passes = serde_json::Map::new();
    for rows in 1..=8usize {
        let mut samples = Vec::new();
        for round in 0..6u32 {
            model.restore(snapshot.clone())?;
            crate::array::synchronize()?;
            let tokens = (0..rows)
                .map(|i| round * 53 + i as u32 + 1)
                .collect::<Vec<_>>();
            let start = Instant::now();
            let logits = model.verify_tokens_exact_with_dflash_capture(&tokens)?.0;
            model.commit_dflash_verification(rows, rows)?;
            crate::array::synchronize()?;
            let elapsed = ms(start);
            ensure!(
                logits.shape().iter().product::<i32>() == rows as i32 * 248320,
                "short logits"
            );
            if round >= 2 {
                samples.push(elapsed);
            }
        }
        let value = median(samples.clone());
        passes.insert(
            rows.to_string(),
            serde_json::json!({
            "ms": value, "relative_to_ar_step": value / ar_ms, "samples": samples}),
        );
    }
    report["verify_pass"] = passes.into();
    report["status"] = "complete".into();
    save(&report)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// Q27D-2026-11-17-04: shared-decode verification against the stock layouts and the
/// token-by-token autoregressive replay, then the cost of one pass in both modes.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_SHARED_REPORT; run alone"]
fn q27_shared_verify_parity_and_cost() -> Result<()> {
    use super::tests::state_bytes;
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_SHARED_REPORT")?);
    ensure!(!output.exists(), "preserve previous shared-verify reports");
    let mut model = Qwen35Moe::load(Path::new(&std::env::var("MLXL3_QWEN_TEST_MODEL")?))?;
    ensure!(
        model.layers.len() == 64 && model.vocab == 248320 && model.mtp_layout.hidden_size == 5120,
        "requires dense Qwen27B fixture"
    );
    let mut report =
        serde_json::json!({"status": "running", "parity": false, "widths": [], "cost": {}});
    let save = |report: &serde_json::Value| -> Result<()> {
        std::fs::write(&output, serde_json::to_vec_pretty(report)?)?;
        Ok(())
    };
    save(&report)?;
    let result: Result<()> = (|| {
        let prefix = (0..69).map(|i| (i * 37 + 1) as u32).collect::<Vec<_>>();
        model.forward_mtp(&prefix)?;
        crate::array::synchronize()?;
        let snapshot = model.snapshot()?;
        let verify = |model: &mut Qwen35Moe, shared: bool, tokens: &[u32], retained: usize| {
            crate::linear::with_dense_shared_test(shared, || -> Result<_> {
                model.restore(snapshot.clone())?;
                let (logits, capture) = model.verify_tokens_exact_with_dflash_capture(tokens)?;
                let logits = logits.to_f16_bits()?;
                let capture = capture.astype(Dtype::Float32)?.to_f32()?;
                model.commit_dflash_verification(retained, tokens.len())?;
                Ok((logits, capture, state_bytes(model)?))
            })
        };
        let mut compared = 0usize;
        for rows in 1..=8usize {
            let tokens = (0..rows)
                .map(|i| (rows * 131 + i * 53 + 7) as u32)
                .collect::<Vec<_>>();
            // Token-by-token autoregressive oracle with the state after every token.
            model.restore(snapshot.clone())?;
            let mut ar_logits = Vec::new();
            let mut ar_states = Vec::new();
            for &token in &tokens {
                ar_logits.extend(model.forward(token)?.to_f16_bits()?);
                ar_states.push(state_bytes(&model)?);
            }
            ensure!(
                ar_logits.len() == rows * 248320
                    && ar_logits.iter().all(|&v| f16::from_bits(v).is_finite())
                    && ar_states.iter().all(|states| states.len() == 128),
                "incomplete autoregressive oracle"
            );
            let stock = verify(&mut model, false, &tokens, rows)?;
            let shared = verify(&mut model, true, &tokens, rows)?;
            ensure!(
                shared.0.len() == rows * 248320
                    && !shared.1.is_empty()
                    && shared.2.len() == 128
                    && shared.1.iter().all(|v| v.is_finite()),
                "incomplete shared verification M={rows}"
            );
            ensure!(
                shared.0 == ar_logits,
                "shared logits differ from autoregressive replay M={rows}"
            );
            ensure!(
                stock.0 == ar_logits,
                "stock logits differ from autoregressive replay M={rows}"
            );
            ensure!(
                shared.1 == stock.1,
                "shared DFlash capture differs from stock M={rows}"
            );
            ensure!(
                shared.2 == ar_states[rows - 1],
                "shared states differ from autoregressive replay M={rows}"
            );
            ensure!(
                stock.2 == ar_states[rows - 1],
                "stock states differ from autoregressive replay M={rows}"
            );
            let mut partial = 0usize;
            for retained in 1..rows {
                let kept = verify(&mut model, true, &tokens, retained)?;
                ensure!(
                    kept.0 == ar_logits,
                    "shared logits changed with the commit M={rows}"
                );
                ensure!(
                    kept.2 == ar_states[retained - 1],
                    "shared partial commit {retained}/{rows} differs from autoregressive replay"
                );
                partial += 1;
            }
            compared += 1;
            report["widths"]
                .as_array_mut()
                .context("widths")?
                .push(serde_json::json!({
                "rows": rows, "logits": shared.0.len(), "capture_values": shared.1.len(),
                "state_arrays": shared.2.len(), "partial_commits": partial, "bit_exact": true}));
            save(&report)?;
            println!(
                "M={rows} exact ({} logits, {} capture values, 128 states, {partial} partial commits)",
                shared.0.len(),
                shared.1.len()
            );
        }
        ensure!(compared == 8, "incomplete width matrix");
        report["parity"] = true.into();
        save(&report)?;

        // Cost of one pass, commit included, modes interleaved (stock, shared, shared, stock) x2.
        let mut cost = serde_json::Map::new();
        let mut ar = Vec::new();
        for rows in 1..=8usize {
            let mut samples = [Vec::new(), Vec::new()];
            for (round, shared) in [
                false, true, true, false, false, true, true, false, false, true,
            ]
            .into_iter()
            .enumerate()
            {
                let tokens = (0..rows)
                    .map(|i| (round * 53 + i + 1) as u32)
                    .collect::<Vec<_>>();
                let elapsed = crate::linear::with_dense_shared_test(shared, || -> Result<f64> {
                    model.restore(snapshot.clone())?;
                    crate::array::synchronize()?;
                    let start = Instant::now();
                    model.verify_tokens_exact_with_dflash_capture(&tokens)?;
                    model.commit_dflash_verification(rows, rows)?;
                    crate::array::synchronize()?;
                    Ok(ms(start))
                })?;
                if round >= 2 {
                    samples[usize::from(shared)].push(elapsed);
                }
            }
            let (stock, shared) = (median(samples[0].clone()), median(samples[1].clone()));
            if rows == 1 {
                ar.push(stock);
            }
            cost.insert(
                rows.to_string(),
                serde_json::json!({
                "stock_ms": stock, "shared_ms": shared,
                "stock_steps": stock / ar[0], "shared_steps": shared / ar[0],
                "samples": {"stock": samples[0], "shared": samples[1]}}),
            );
            println!(
                "M={rows} stock {stock:.1} ms ({:.2} steps) shared {shared:.1} ms ({:.2} steps)",
                stock / ar[0],
                shared / ar[0]
            );
        }
        report["cost"] = cost.into();
        Ok(())
    })();
    match &result {
        Ok(()) => report["status"] = "complete".into(),
        Err(error) => {
            report["status"] = "failed".into();
            report["parity"] = false.into();
            report["error"] = format!("{error:#}").into();
        }
    }
    save(&report)?;
    result
}

/// Q27D-2026-11-18-07: per-token cost against context length, and the K/V concat share.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_LONG_REPORT; run alone"]
fn q27_long_context_decode() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_LONG_REPORT")?);
    ensure!(!output.exists(), "preserve previous long-context reports");
    let mut model = Qwen35Moe::load(Path::new(&std::env::var("MLXL3_QWEN_TEST_MODEL")?))?;
    ensure!(
        model.layers.len() == 64 && model.vocab == 248320 && model.mtp_layout.hidden_size == 5120,
        "requires dense Qwen27B fixture"
    );
    let contexts = [256usize, 4096, 8192, 16384];
    ensure!(
        model.context_limit() as usize > contexts[3] + 64,
        "context limit too small"
    );
    let mut report = serde_json::json!({"status": "running", "contexts": []});
    let save = |report: &serde_json::Value| -> Result<()> {
        std::fs::write(&output, serde_json::to_vec_pretty(report)?)?;
        Ok(())
    };
    save(&report)?;
    let inplace = std::env::var("MLXL3_Q27_LONG_INPLACE").map_or(true, |value| value == "1");
    report["kv_inplace"] = inplace.into();
    let result: Result<()> = super::with_kv_inplace(inplace, || -> Result<()> {
        let tokens = (0..contexts[3])
            .map(|i| ((i * 7919 + 13) % 200_000 + 1000) as u32)
            .collect::<Vec<_>>();
        let mut filled = 0usize;
        for &context in &contexts {
            let started = Instant::now();
            for chunk in tokens[filled..context].chunks(256) {
                model.forward_tokens(chunk)?;
            }
            crate::array::synchronize()?;
            let prefill_s = started.elapsed().as_secs_f64();
            filled = context;
            ensure!(model.offset() as usize == context, "unexpected offset");
            let snapshot = model.snapshot()?;
            let mut ar = Vec::new();
            for index in 0..20u32 {
                let start = Instant::now();
                model.forward(index * 41 + 5)?;
                crate::array::synchronize()?;
                if index >= 4 {
                    ar.push(ms(start));
                }
            }
            model.restore(snapshot.clone())?;
            let mut verify = serde_json::Map::new();
            for rows in [2usize, 3] {
                let mut samples = Vec::new();
                for round in 0..6u32 {
                    model.restore(snapshot.clone())?;
                    crate::array::synchronize()?;
                    let probe = (0..rows)
                        .map(|i| round * 53 + i as u32 + 1)
                        .collect::<Vec<_>>();
                    let start = Instant::now();
                    model.verify_tokens_exact_with_dflash_capture(&probe)?;
                    model.commit_dflash_verification(rows, rows)?;
                    crate::array::synchronize()?;
                    if round >= 2 {
                        samples.push(ms(start));
                    }
                }
                verify.insert(format!("m{rows}_ms"), median(samples).into());
            }
            model.restore(snapshot.clone())?;
            // Isolated K/V growth copy at this context (16 layers, keys and values).
            let (kv_ms, _) = {
                let previous = Array::zeros_dtype(&[1, 4, context as i32, 256], Dtype::Float16)?;
                let step = Array::zeros_dtype(&[1, 4, 1, 256], Dtype::Float16)?;
                previous.eval()?;
                step.eval()?;
                timed(|| {
                    (0..32)
                        .map(|_| Array::concatenate(&[&previous, &step], 2))
                        .collect()
                })?
            };
            let memory = crate::array::memory_stats(false)?;
            let ar_ms = median(ar.clone());
            let row = serde_json::json!({
                "context": context, "prefill_seconds": prefill_s, "ar_step_ms": ar_ms, "ar_samples": ar,
                "verify": verify, "kv_concat_32_ms": kv_ms, "memory": memory});
            println!("{row}");
            report["contexts"]
                .as_array_mut()
                .context("contexts")?
                .push(row);
            save(&report)?;
        }
        Ok(())
    });
    match &result {
        Ok(()) => report["status"] = "complete".into(),
        Err(error) => {
            report["status"] = "failed".into();
            report["error"] = format!("{error:#}").into();
        }
    }
    save(&report)?;
    result
}

/// Q27D-11: in-place K/V against concatenation on the same model, bit for bit.
#[test]
#[ignore = "physical Qwen27 GPU; run alone"]
fn q27_kv_inplace_matches_concatenation() -> Result<()> {
    use super::tests::state_bytes;
    let mut model = Qwen35Moe::load(Path::new(&std::env::var("MLXL3_QWEN_TEST_MODEL")?))?;
    ensure!(
        model.layers.len() == 64 && model.vocab == 248320 && model.mtp_layout.hidden_size == 5120,
        "requires dense Qwen27B fixture"
    );
    ensure!(
        model
            .layers
            .iter()
            .any(|layer| matches!(layer, Layer::Attention(a) if a.attention.kv_inplace)),
        "dense attention layers must be eligible"
    );
    let mut compared = 0usize;
    for context in [256usize, 300, 4096] {
        let tokens = (0..context)
            .map(|i| ((i * 7919 + 13) % 200_000 + 1000) as u32)
            .collect::<Vec<_>>();
        // Prefill under both stores, chunks of 256 as in production.
        let mut prefill = |inplace: bool| -> Result<(Vec<u16>, Vec<Vec<u8>>, QwenSnapshot)> {
            super::with_kv_inplace(inplace, || {
                model.reset();
                let mut last = None;
                for chunk in tokens.chunks(256) {
                    last = Some(model.forward_tokens(chunk)?);
                }
                crate::array::synchronize()?;
                Ok((
                    last.context("no prefill")?.to_f16_bits()?,
                    state_bytes(&model)?,
                    model.snapshot()?,
                ))
            })
        };
        let (stock_logits, stock_states, snapshot) = prefill(false)?;
        let (inplace_logits, inplace_states, _) = prefill(true)?;
        ensure!(
            stock_logits.len() == 248320 && stock_states.len() == 128,
            "incomplete prefill"
        );
        ensure!(
            stock_logits == inplace_logits,
            "prefill logits differ at {context}"
        );
        ensure!(
            stock_states == inplace_states,
            "prefill states differ at {context}"
        );
        compared += 1;
        // Decode script: AR steps, then verification blocks with partial commits.
        let script = |model: &mut Qwen35Moe, inplace: bool| -> Result<Trace> {
            super::with_kv_inplace(inplace, || {
                model.restore(snapshot.clone())?;
                let mut out = Vec::new();
                for step in 0..6u32 {
                    let logits = model.forward(step * 31 + 7)?.to_f16_bits()?;
                    out.push((logits, state_bytes(model)?));
                }
                for (rows, retained) in [(2usize, 1usize), (3, 3), (4, 2), (4, 4), (3, 1)] {
                    let probe = (0..rows)
                        .map(|i| (rows * 97 + i * 13 + 5) as u32)
                        .collect::<Vec<_>>();
                    let logits = model
                        .verify_tokens_exact_with_dflash_capture(&probe)?
                        .0
                        .to_f16_bits()?;
                    model.commit_dflash_verification(retained, rows)?;
                    out.push((logits, state_bytes(model)?));
                }
                for step in 0..3u32 {
                    let logits = model.forward(step * 17 + 3)?.to_f16_bits()?;
                    out.push((logits, state_bytes(model)?));
                }
                crate::array::synchronize()?;
                Ok(out)
            })
        };
        let stock = script(&mut model, false)?;
        let inplace = script(&mut model, true)?;
        ensure!(
            stock.len() == 14 && stock.len() == inplace.len(),
            "incomplete decode script"
        );
        for (index, (a, b)) in stock.iter().zip(&inplace).enumerate() {
            ensure!(!a.0.is_empty() && a.1.len() == 128, "empty step {index}");
            ensure!(
                a.0 == b.0,
                "logits differ at context {context}, step {index}"
            );
            ensure!(
                a.1 == b.1,
                "states differ at context {context}, step {index}"
            );
            compared += 1;
        }
        println!(
            "context {context}: prefill + 14 decode/verify steps bit-exact (logits and 128 states)"
        );
    }
    ensure!(compared == 45, "incomplete comparison");
    Ok(())
}

/// Q27D-16 (not bit-exact): teacher-forced quality of TensorOps verification against the
/// exact QMV path on real text, eight rows per verification pass.
#[cfg(feature = "chat")]
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_TENSOR_QUALITY; run alone"]
fn q27_tensor_verify_quality() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_TENSOR_QUALITY")?);
    ensure!(!output.exists(), "preserve previous quality reports");
    let path = std::path::PathBuf::from(std::env::var("MLXL3_QWEN_TEST_MODEL")?);
    let tokenizer = crate::tokenizer::ChatTokenizer::load(&path)?;
    let mut model = Qwen35Moe::load(&path)?;
    let threshold: i32 =
        std::env::var("MLXL3_Q27_TENSOR_THRESHOLD").map_or(Ok(5), |v| v.parse())?;
    let texts = [
        (
            "french",
            std::fs::read_to_string("docs/measurements/lfm3-pld-20261116/long.txt")?,
        ),
        (
            "code",
            std::fs::read_to_string("native/src/speculative.rs")?
                .chars()
                .take(8000)
                .collect(),
        ),
    ];
    let mut report = serde_json::json!({"status": "running", "threshold": threshold, "texts": []});
    let mut totals = (0usize, 0usize, 0f64, 0f64, 0f64, 0f64);
    for (label, text) in &texts {
        let ids = tokenizer.encode(text)?;
        ensure!(ids.len() > 600, "text too short");
        let ids = &ids[..ids.len().min(1600)];
        let prefix = 64usize;
        let run = |model: &mut Qwen35Moe, tensor: Option<i32>| -> Result<Vec<Vec<f32>>> {
            crate::linear::with_dense_tensor_rows(tensor, || {
                model.reset();
                let mut rows = Vec::new();
                model.forward_tokens(&ids[..prefix])?;
                let mut position = prefix;
                while position + 8 <= ids.len() {
                    let logits = model.verify_tokens_exact(&ids[position..position + 8])?;
                    let values = logits.log_probs()?.astype(Dtype::Float32)?.to_f32()?;
                    ensure!(values.len() == 8 * 248320, "short verification rows");
                    rows.extend(values.chunks(248320).map(<[f32]>::to_vec));
                    position += 8;
                }
                Ok(rows)
            })
        };
        let exact = run(&mut model, None)?;
        let tensor = run(&mut model, Some(threshold))?;
        ensure!(
            exact.len() == tensor.len() && !exact.is_empty(),
            "row count mismatch"
        );
        let (mut kl_sum, mut kl_max, mut disagree, mut nll_exact, mut nll_tensor) =
            (0f64, 0f64, 0usize, 0f64, 0f64);
        let mut scored = 0usize;
        for (index, (p, q)) in exact.iter().zip(&tensor).enumerate() {
            let argmax = |row: &[f32]| {
                row.iter()
                    .enumerate()
                    .fold(
                        (0, f32::NEG_INFINITY),
                        |a, (i, &v)| if v > a.1 { (i, v) } else { a },
                    )
                    .0
            };
            if argmax(p) != argmax(q) {
                disagree += 1;
            }
            let kl: f64 = p
                .iter()
                .zip(q)
                .map(|(&lp, &lq)| {
                    let pp = (lp as f64).exp();
                    if pp > 0.0 {
                        pp * (lp as f64 - lq as f64)
                    } else {
                        0.0
                    }
                })
                .sum();
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            let next = prefix + index + 1;
            if next < ids.len() {
                nll_exact -= p[ids[next] as usize] as f64;
                nll_tensor -= q[ids[next] as usize] as f64;
                scored += 1;
            }
        }
        let positions = exact.len();
        let row = serde_json::json!({
            "text": label, "positions": positions, "scored": scored,
            "mean_kl": kl_sum / positions as f64, "max_kl": kl_max,
            "argmax_disagreements": disagree,
            "perplexity_exact": (nll_exact / scored as f64).exp(),
            "perplexity_tensor": (nll_tensor / scored as f64).exp(),
        });
        println!("{row}");
        totals = (
            totals.0 + positions,
            totals.1 + disagree,
            totals.2 + kl_sum,
            totals.3.max(kl_max),
            totals.4 + nll_exact,
            totals.5 + nll_tensor,
        );
        report["texts"].as_array_mut().context("texts")?.push(row);
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["status"] = "complete".into();
    report["all"] = serde_json::json!({"positions": totals.0, "argmax_disagreements": totals.1,
        "mean_kl": totals.2 / totals.0 as f64, "max_kl": totals.3});
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

/// Q27P-03: quality of a prefill variant. Prefill 1 024 tokens in 256-token chunks with the
/// shared prefill kernel at depth 16 (bit-exact control) or 32 (candidate), then score the next
/// tokens with exact 8-row verification: any difference comes from the prefilled states.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_PREFILL_QUALITY; run alone"]
fn q27_prefill_depth_quality() -> Result<()> {
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_PREFILL_QUALITY")?);
    ensure!(!output.exists(), "preserve previous quality reports");
    let path = std::path::PathBuf::from(std::env::var("MLXL3_QWEN_TEST_MODEL")?);
    let tokenizer = crate::tokenizer::ChatTokenizer::load(&path)?;
    let mut model = Qwen35Moe::load(&path)?;
    let texts = [
        (
            "french",
            std::fs::read_to_string("docs/measurements/lfm3-pld-20261116/long.txt")?,
        ),
        (
            "code",
            std::fs::read_to_string("native/src/speculative.rs")?,
        ),
    ];
    let mut report = serde_json::json!({"status": "running", "texts": []});
    for (label, text) in &texts {
        let ids = tokenizer.encode(text)?;
        let prefix = 1024usize;
        ensure!(ids.len() > prefix + 512, "text too short: {}", ids.len());
        let ids = &ids[..prefix + 512 + 1];
        let run = |model: &mut Qwen35Moe, shared: i32, depth: i32| -> Result<Vec<Vec<f32>>> {
            crate::linear::with_dense_tensor_rows(None, || {
                crate::linear::with_dense_prefill_shared(shared, || {
                    crate::linear::with_dense_prefill_depth(depth, || -> Result<()> {
                        model.reset();
                        for chunk in ids[..prefix].chunks(256) {
                            model.forward_tokens(chunk)?;
                        }
                        Ok(())
                    })?;
                    Ok::<(), anyhow::Error>(())
                })?;
                let mut rows = Vec::new();
                let mut position = prefix;
                while position + 8 <= prefix + 512 {
                    let logits = model.verify_tokens_exact(&ids[position..position + 8])?;
                    let values = logits.log_probs()?.astype(Dtype::Float32)?.to_f32()?;
                    ensure!(
                        values.len() == 8 * 248320
                            && values
                                .iter()
                                .all(|v| v.is_finite() || *v == f32::NEG_INFINITY),
                        "bad verification rows"
                    );
                    rows.extend(values.chunks(248320).map(<[f32]>::to_vec));
                    position += 8;
                }
                Ok(rows)
            })
        };
        let stock = run(&mut model, 0, 16)?;
        let control = run(&mut model, 2, 16)?;
        let candidate = run(&mut model, 2, 32)?;
        let identical_control = stock
            .iter()
            .zip(&control)
            .all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()));
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .fold(
                    (0, f32::NEG_INFINITY),
                    |a, (i, &v)| if v > a.1 { (i, v) } else { a },
                )
                .0
        };
        let (mut kl_sum, mut kl_max, mut disagree, mut nll_stock, mut nll_candidate) =
            (0f64, 0f64, 0usize, 0f64, 0f64);
        for (index, (p, q)) in stock.iter().zip(&candidate).enumerate() {
            if argmax(p) != argmax(q) {
                disagree += 1;
            }
            let kl: f64 = p
                .iter()
                .zip(q)
                .map(|(&lp, &lq)| {
                    let pp = (lp as f64).exp();
                    if pp > 0.0 {
                        pp * (lp as f64 - lq as f64)
                    } else {
                        0.0
                    }
                })
                .sum();
            kl_sum += kl;
            kl_max = kl_max.max(kl);
            let next = ids[prefix + index + 1] as usize;
            nll_stock -= p[next] as f64;
            nll_candidate -= q[next] as f64;
        }
        let n = stock.len() as f64;
        let row = serde_json::json!({"text": label, "rows": stock.len(), "control_bit_identical": identical_control,
            "kl_mean": kl_sum / n, "kl_max": kl_max, "argmax_disagreements": disagree,
            "perplexity_stock": (nll_stock / n).exp(), "perplexity_depth32": (nll_candidate / n).exp()});
        println!("{row}");
        report["texts"].as_array_mut().context("texts")?.push(row);
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

/// Q27D-37 oracle: tree verification against path-only verification and the production chain,
/// then tree commits (prefix and non-prefix paths) against chain commits, one AR step after.
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_TREE_ORACLE; run alone"]
fn q27_tree_verify_matches_paths() -> Result<()> {
    use crate::speculative::DraftTree;
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_TREE_ORACLE")?);
    ensure!(!output.exists(), "preserve previous oracle reports");
    let path = std::path::PathBuf::from(std::env::var("MLXL3_QWEN_TEST_MODEL")?);
    let tokenizer = crate::tokenizer::ChatTokenizer::load(&path)?;
    let mut model = Qwen35Moe::load(&path)?;
    let text = std::fs::read_to_string("docs/measurements/lfm3-pld-20261116/long.txt")?;
    let ids = tokenizer.encode(&text)?;
    ensure!(ids.len() > 700, "text too short");
    let vocab = 248_320usize;
    let argmax = |row: &[f32]| {
        row.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |a, (i, &v)| {
                if v > a.1 { (i, v) } else { a }
            })
            .0
    };
    let rows_of = |logits: &Array| -> Result<Vec<Vec<f32>>> {
        let values = logits.log_probs()?.astype(Dtype::Float32)?.to_f32()?;
        ensure!(
            values.len() % vocab == 0
                && values
                    .iter()
                    .all(|v| v.is_finite() || *v == f32::NEG_INFINITY),
            "bad logits"
        );
        Ok(values.chunks(vocab).map(<[f32]>::to_vec).collect())
    };
    // (differing words, max |Δ log p|, argmax changes, mean KL(reference ‖ candidate)).
    let compare = |a: &[Vec<f32>], b: &[Vec<f32>]| -> (usize, f32, usize, f64) {
        let mut differing = 0usize;
        let mut max = 0f32;
        let mut argmax_changes = 0usize;
        let mut kl = 0f64;
        for (x, y) in a.iter().zip(b) {
            for (&p, &q) in x.iter().zip(y) {
                differing += usize::from(p.to_bits() != q.to_bits());
                if p.is_finite() && q.is_finite() {
                    max = max.max((p - q).abs());
                }
                let reference = f64::from(q).exp();
                if reference > 0.0 {
                    kl += reference * (f64::from(q) - f64::from(p));
                }
            }
            argmax_changes += usize::from(argmax(x) != argmax(y));
        }
        (differing, max, argmax_changes, kl / a.len().max(1) as f64)
    };
    let mut report = serde_json::json!({"status": "running", "cases": []});
    let mut worst = (0f32, 0usize);
    crate::linear::with_dense_tensor_rows(Some(3), || -> Result<()> {
        for prefix in [299usize, 640] {
            model.reset();
            for chunk in ids[..prefix].chunks(256) {
                model.forward_tokens(chunk)?;
            }
            let snapshot = model.snapshot()?;
            let t = |offset: usize| ids[prefix + offset];
            // Alternatives distinct from the main tokens (siblings must differ).
            let alt = |offset: usize| {
                let token = ids[prefix + 200 + offset];
                if (0..8).any(|o| ids[prefix + o] == token) {
                    (token + 7 + offset as u32) % 200_000 + 1000
                } else {
                    token
                }
            };
            // 12 rows: main chain 0-1-3-6-8-9-10-11, branch 0-2-4-7, side 1-5.
            let tree = DraftTree {
                tokens: vec![
                    ids[prefix - 1],
                    t(0),
                    alt(0),
                    t(1),
                    alt(1),
                    alt(2),
                    t(2),
                    alt(3),
                    t(3),
                    t(4),
                    t(5),
                    t(6),
                ],
                parents: vec![0, 0, 0, 1, 2, 1, 3, 4, 6, 8, 9, 10],
                depths: vec![0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7],
            };
            ensure!(tree.is_valid(), "oracle tree is invalid");
            let (tree_logits, tree_capture) = model.verify_tree_with_dflash_capture(&tree)?;
            let tree_rows = rows_of(&tree_logits)?;
            let tree_capture = tree_capture.astype(Dtype::Float32)?.to_f32()?;
            ensure!(tree_rows.len() == tree.rows(), "short tree logits");
            model.restore(snapshot.clone())?;
            for path in tree.paths() {
                let tokens = path.iter().map(|&row| tree.tokens[row]).collect::<Vec<_>>();
                let chain = DraftTree {
                    tokens: tokens.clone(),
                    parents: (0..tokens.len()).map(|i| i.saturating_sub(1)).collect(),
                    depths: (0..tokens.len()).collect(),
                };
                let (chain_logits, chain_capture) =
                    model.verify_tree_with_dflash_capture(&chain)?;
                let chain_rows = rows_of(&chain_logits)?;
                model.restore(snapshot.clone())?;
                let (block_logits, _) = model.verify_tokens_exact_with_dflash_capture(&tokens)?;
                let block_rows = rows_of(&block_logits)?;
                model.restore(snapshot.clone())?;
                let in_tree = path
                    .iter()
                    .map(|&row| tree_rows[row].clone())
                    .collect::<Vec<_>>();
                let (d1, m1, a1, k1) = compare(&in_tree, &chain_rows);
                let (d2, m2, a2, k2) = compare(&chain_rows, &block_rows);
                let chain_capture = chain_capture.astype(Dtype::Float32)?.to_f32()?;
                let width = chain_capture.len() / tokens.len();
                let capture_differing = path
                    .iter()
                    .enumerate()
                    .map(|(i, &row)| {
                        tree_capture[row * width..(row + 1) * width]
                            .iter()
                            .zip(&chain_capture[i * width..(i + 1) * width])
                            .filter(|(p, q)| p.to_bits() != q.to_bits())
                            .count()
                    })
                    .sum::<usize>();
                worst = (worst.0.max(m1).max(m2), worst.1 + a1 + a2);
                let case = serde_json::json!({"prefix": prefix, "path": path, "kind": "logits",
                    "tree_vs_path_differing": d1, "tree_vs_path_max": m1, "tree_vs_path_argmax_changes": a1, "tree_vs_path_kl": k1, "path_vs_production_chain_kl": k2,
                    "path_vs_production_chain_differing": d2, "path_vs_production_chain_max": m2,
                    "path_vs_production_chain_argmax_changes": a2, "capture_tree_vs_path_differing": capture_differing});
                println!("{case}");
                report["cases"].as_array_mut().context("cases")?.push(case);
            }
            // Commit oracle: accept a prefix of a path, then one AR step, against the chain.
            for accepted in [
                vec![0usize, 1, 3],
                vec![0, 2, 4],
                vec![0, 1, 5],
                vec![0usize, 1, 3, 6, 8, 9, 10, 11],
                vec![0],
            ] {
                model.restore(snapshot.clone())?;
                model.verify_tree_with_dflash_capture(&tree)?;
                model.commit_dflash_tree(&tree, &accepted)?;
                let next = ids[prefix + 400];
                let after_tree = rows_of(&model.forward_tokens(&[next])?)?;
                let offset_tree = model.offset;
                model.restore(snapshot.clone())?;
                let full = tree
                    .paths()
                    .into_iter()
                    .find(|p| p.starts_with(&accepted))
                    .context("path")?;
                let tokens = full.iter().map(|&row| tree.tokens[row]).collect::<Vec<_>>();
                model.verify_tokens_exact_with_dflash_capture(&tokens)?;
                model.commit_dflash_verification(accepted.len(), tokens.len())?;
                let after_chain = rows_of(&model.forward_tokens(&[next])?)?;
                ensure!(model.offset == offset_tree, "offsets differ after commit");
                let (d, m, a, k) = compare(&after_tree, &after_chain);
                worst = (worst.0.max(m), worst.1 + a);
                let case = serde_json::json!({"prefix": prefix, "accepted": accepted, "kind": "commit",
                    "next_step_differing": d, "next_step_max": m, "next_step_argmax_changes": a, "next_step_kl": k, "offset": offset_tree});
                println!("{case}");
                report["cases"].as_array_mut().context("cases")?.push(case);
            }
        }
        Ok(())
    })?;
    report["max_abs_log_prob_difference"] = worst.0.into();
    report["argmax_changes"] = worst.1.into();
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

/// Q27D-39: KL and perplexity of tree verification against the exact path, with the production
/// TensorOps chain as the known reference (Q27D-16 protocol: same texts, 8-row passes).
#[cfg(feature = "chat")]
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_TREE_QUALITY; run alone"]
fn q27_tree_verify_quality() -> Result<()> {
    use crate::speculative::DraftTree;
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_TREE_QUALITY")?);
    ensure!(!output.exists(), "preserve previous quality reports");
    let path = std::path::PathBuf::from(std::env::var("MLXL3_QWEN_TEST_MODEL")?);
    let tokenizer = crate::tokenizer::ChatTokenizer::load(&path)?;
    let mut model = Qwen35Moe::load(&path)?;
    let vocab = 248_320usize;
    let texts = [
        (
            "french",
            std::fs::read_to_string("docs/measurements/lfm3-pld-20261116/long.txt")?,
        ),
        (
            "code",
            std::fs::read_to_string("native/src/speculative.rs")?
                .chars()
                .take(8000)
                .collect(),
        ),
    ];
    // F32 log-softmax of each row, finite rows only.
    let rows_of = |logits: &Array, expected: usize| -> Result<Vec<Vec<f32>>> {
        let values = logits.astype(Dtype::Float32)?.log_probs()?.to_f32()?;
        ensure!(
            values.len() == expected * vocab
                && values
                    .iter()
                    .all(|v| v.is_finite() || *v == f32::NEG_INFINITY),
            "bad verification rows"
        );
        Ok(values.chunks(vocab).map(<[f32]>::to_vec).collect())
    };
    let ranked = |row: &[f32]| -> [usize; 3] {
        let mut best = [(f32::NEG_INFINITY, 0usize); 3];
        for (i, &v) in row.iter().enumerate() {
            if v > best[2].0 {
                best[2] = (v, i);
                best.sort_by(|a, b| b.0.total_cmp(&a.0));
            }
        }
        [best[0].1, best[1].1, best[2].1]
    };
    #[derive(Default)]
    struct Score {
        kl: f64,
        kl_max: f64,
        disagree: usize,
        nll: f64,
        rows: usize,
    }
    let score = |score: &mut Score, exact: &[f32], candidate: &[f32], next: Option<u32>| {
        let kl: f64 = exact
            .iter()
            .zip(candidate)
            .map(|(&lp, &lq)| {
                let pp = f64::from(lp).exp();
                if pp > 0.0 {
                    pp * (f64::from(lp) - f64::from(lq))
                } else {
                    0.0
                }
            })
            .sum();
        score.kl += kl;
        score.kl_max = score.kl_max.max(kl);
        score.disagree += usize::from(ranked(exact)[0] != ranked(candidate)[0]);
        if let Some(next) = next {
            score.nll -= f64::from(candidate[next as usize]);
        }
        score.rows += 1;
    };
    let mut report = serde_json::json!({"status": "running", "texts": [],
        "tree": "12 rows: main 0-1-3-6-8-9-10-11 (real tokens), branch 0-2-4-7 and side 1-5 (2nd/3rd exact choices)"});
    for (label, text) in &texts {
        let ids = tokenizer.encode(text)?;
        ensure!(ids.len() > 600, "text too short");
        let ids = &ids[..ids.len().min(1600)];
        let prefix = 64usize;
        let starts = (prefix..)
            .step_by(8)
            .take_while(|p| p + 8 <= ids.len())
            .collect::<Vec<_>>();
        // (A) exact reference rows.
        let mut exact = Vec::new();
        crate::linear::with_dense_tensor_rows(None, || -> Result<()> {
            model.reset();
            model.forward_tokens(&ids[..prefix])?;
            for &p in &starts {
                exact.extend(rows_of(&model.verify_tokens_exact(&ids[p..p + 8])?, 8)?);
            }
            Ok(())
        })?;
        ensure!(exact.len() == starts.len() * 8, "exact row count");
        let next_of = |index: usize| {
            let next = prefix + index + 1;
            (next < ids.len()).then(|| ids[next])
        };
        let mut nll_exact = 0f64;
        let mut scored = 0usize;
        for (index, row) in exact.iter().enumerate() {
            if let Some(next) = next_of(index) {
                nll_exact -= f64::from(row[next as usize]);
                scored += 1;
            }
        }
        // (B) production TensorOps chain, (C) tree; both scored against (A) row by row.
        let (mut chain, mut tree, mut tree_vs_chain) = (Score::default(), Score::default(), 0usize);
        crate::linear::with_dense_tensor_rows(Some(3), || -> Result<()> {
            model.reset();
            model.forward_tokens(&ids[..prefix])?;
            let mut chain_rows = Vec::new();
            for &p in &starts {
                chain_rows.extend(rows_of(&model.verify_tokens_exact(&ids[p..p + 8])?, 8)?);
            }
            for (index, (a, b)) in exact.iter().zip(&chain_rows).enumerate() {
                score(&mut chain, a, b, next_of(index));
            }
            model.reset();
            model.forward_tokens(&ids[..prefix])?;
            for (block, &p) in starts.iter().enumerate() {
                let rows = &exact[block * 8..block * 8 + 8];
                let t = |offset: usize| ids[p + offset];
                // Plausible alternatives: the exact model's next choices, distinct from siblings.
                let other = |row: usize, avoid: u32, rank: usize| -> u32 {
                    let choices = ranked(&rows[row]);
                    let picks = choices
                        .iter()
                        .map(|&c| c as u32)
                        .filter(|&c| c != avoid)
                        .collect::<Vec<_>>();
                    picks[rank.min(picks.len() - 1)]
                };
                let tokens = vec![
                    t(0),
                    t(1),
                    other(0, t(1), 0),
                    t(2),
                    other(1, t(2), 1),
                    other(1, t(2), 0),
                    t(3),
                    other(2, t(3), 0),
                    t(4),
                    t(5),
                    t(6),
                    t(7),
                ];
                let draft = DraftTree {
                    tokens,
                    parents: vec![0, 0, 0, 1, 2, 1, 3, 4, 6, 8, 9, 10],
                    depths: vec![0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7],
                };
                ensure!(draft.is_valid(), "quality tree is invalid at {p}");
                let (logits, _) = model.verify_tree_with_dflash_capture(&draft)?;
                let tree_rows = rows_of(&logits, draft.rows())?;
                let main = [0usize, 1, 3, 6, 8, 9, 10, 11];
                for (offset, &row) in main.iter().enumerate() {
                    let index = block * 8 + offset;
                    score(&mut tree, &exact[index], &tree_rows[row], next_of(index));
                    tree_vs_chain +=
                        usize::from(ranked(&tree_rows[row])[0] != ranked(&chain_rows[index])[0]);
                }
                model.commit_dflash_tree(&draft, &main)?;
            }
            Ok(())
        })?;
        ensure!(
            chain.rows == exact.len() && tree.rows == exact.len(),
            "scored row counts differ"
        );
        let summary = |s: &Score| {
            serde_json::json!({"mean_kl": s.kl / s.rows as f64, "max_kl": s.kl_max,
                "argmax_disagreements": s.disagree, "perplexity": (s.nll / scored as f64).exp()})
        };
        let row = serde_json::json!({
            "text": label, "positions": exact.len(), "scored": scored,
            "perplexity_exact": (nll_exact / scored as f64).exp(),
            "chain_tensor_ops": summary(&chain), "tree": summary(&tree),
            "tree_vs_chain_argmax_disagreements": tree_vs_chain,
        });
        println!("{row}");
        report["texts"].as_array_mut().context("texts")?.push(row);
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    }
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

/// Q27D-41: verification cost of trees with a fixed number of rows and a varying number of
/// root-to-leaf paths (one GDN convolution and recurrence pass per path).
#[cfg(feature = "chat")]
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_TREE_PATH_COST; run alone"]
fn q27_tree_path_cost() -> Result<()> {
    use crate::speculative::DraftTree;
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_TREE_PATH_COST")?);
    ensure!(!output.exists(), "preserve previous reports");
    let path = std::path::PathBuf::from(std::env::var("MLXL3_QWEN_TEST_MODEL")?);
    let tokenizer = crate::tokenizer::ChatTokenizer::load(&path)?;
    let mut model = Qwen35Moe::load(&path)?;
    let text = std::fs::read_to_string("docs/measurements/lfm3-pld-20261116/long.txt")?;
    let ids = tokenizer.encode(&text)?;
    ensure!(ids.len() > 700, "text too short");
    // Depth-ordered tree from (parent, token) pairs listed depth by depth.
    let tree_of = |nodes: &[(usize, u32)]| -> Result<DraftTree> {
        let mut tree = DraftTree {
            tokens: vec![ids[299]],
            parents: vec![0],
            depths: vec![0],
        };
        for &(parent, token) in nodes {
            let depth = tree.depths[parent] + 1;
            tree.tokens.push(token);
            tree.parents.push(parent);
            tree.depths.push(depth);
        }
        ensure!(tree.is_valid(), "synthetic tree is invalid");
        Ok(tree)
    };
    let t = |offset: usize| ids[300 + offset];
    let alt = |offset: usize| ids[500 + offset];
    // Main chain = rows of depth 1..7; side leaves hang off main nodes (one path each).
    let chain = |extra_leaves: &[(usize, usize)], branch: usize| -> Result<DraftTree> {
        // Build depth by depth: main node of depth d, then leaves of parents at depth d - 1.
        let mut nodes: Vec<(usize, u32)> = Vec::new();
        let mut main = vec![0usize];
        let mut branch_nodes = vec![0usize];
        for depth in 1..=7 {
            nodes.push((main[depth - 1], t(depth - 1)));
            let row = nodes.len();
            main.push(row);
            for &(leaf_depth, index) in extra_leaves {
                if leaf_depth == depth {
                    nodes.push((main[depth - 1], alt(index)));
                }
            }
            if depth <= branch {
                nodes.push((branch_nodes[depth - 1], alt(40 + depth)));
                branch_nodes.push(nodes.len());
            }
        }
        tree_of(&nodes)
    };
    let shapes: Vec<(&str, DraftTree)> = vec![
        ("8 rows / 1 path", chain(&[], 0)?),
        ("12 rows / 2 paths", chain(&[], 4)?),
        (
            "12 rows / 5 paths",
            chain(&[(1, 0), (2, 1), (3, 2), (4, 3)], 0)?,
        ),
        ("15 rows / 2 paths", chain(&[], 7)?),
        (
            "16 rows / 9 paths",
            chain(
                &[
                    (1, 0),
                    (2, 1),
                    (3, 2),
                    (4, 3),
                    (5, 4),
                    (6, 5),
                    (7, 6),
                    (2, 7),
                ],
                0,
            )?,
        ),
    ];
    let mut report = serde_json::json!({"status": "running", "context": 300, "cases": []});
    crate::linear::with_dense_tensor_rows(Some(3), || -> Result<()> {
        model.reset();
        for chunk in ids[..299].chunks(256) {
            model.forward_tokens(chunk)?;
        }
        crate::array::synchronize()?;
        let snapshot = model.snapshot()?;
        for (label, tree) in &shapes {
            // Q27D-46: process CPU time beside wall time.
            let cpu_ms = || {
                let mut time = libc::timespec {
                    tv_sec: 0,
                    tv_nsec: 0,
                };
                // SAFETY: valid pointer to a timespec owned by this frame.
                unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut time) };
                time.tv_sec as f64 * 1e3 + time.tv_nsec as f64 / 1e6
            };
            let mut samples = Vec::new();
            let mut cpu = Vec::new();
            let mut build = Vec::new();
            for repeat in 0..12 {
                model.restore(snapshot.clone())?;
                crate::array::synchronize()?;
                let start = Instant::now();
                let cpu_start = cpu_ms();
                let (logits, capture) = model.verify_tree_with_dflash_capture(tree)?;
                logits.eval()?;
                capture.eval()?;
                crate::array::synchronize()?;
                if repeat >= 2 {
                    samples.push(ms(start));
                    cpu.push(cpu_ms() - cpu_start);
                }
            }
            // Graph construction alone: every layer built lazily, nothing evaluated.
            let pipeline = model.pipeline;
            model.pipeline = false;
            for repeat in 0..6 {
                model.restore(snapshot.clone())?;
                crate::array::synchronize()?;
                let start = Instant::now();
                let built = model.build_tree_graph_for_test(tree)?;
                if repeat >= 1 {
                    build.push(ms(start));
                }
                drop(built);
            }
            model.pipeline = pipeline;
            let case = serde_json::json!({"shape": label, "rows": tree.rows(), "paths": tree.paths().len(),
                "median_ms": median(samples.clone()), "samples_ms": samples,
                "cpu_median_ms": median(cpu.clone()), "build_only_median_ms": median(build.clone())});
            println!("{case}");
            report["cases"].as_array_mut().context("cases")?.push(case);
            std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
        }
        Ok(())
    })?;
    report["status"] = "complete".into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    Ok(())
}

/// Q27D-44 oracle: the batched GDN tree verification is bit-identical to the path-by-path one,
/// for logits, DFlash captures and the state after committing accepted paths.
#[cfg(feature = "chat")]
#[test]
#[ignore = "physical Qwen27 GPU; writes MLXL3_Q27_TREE_BATCHED_ORACLE; run alone"]
fn q27_tree_batched_gdn_matches_paths() -> Result<()> {
    use crate::speculative::DraftTree;
    let output = std::path::PathBuf::from(std::env::var("MLXL3_Q27_TREE_BATCHED_ORACLE")?);
    ensure!(!output.exists(), "preserve previous oracle reports");
    let path = std::path::PathBuf::from(std::env::var("MLXL3_QWEN_TEST_MODEL")?);
    let tokenizer = crate::tokenizer::ChatTokenizer::load(&path)?;
    let mut model = Qwen35Moe::load(&path)?;
    let text = std::fs::read_to_string("docs/measurements/lfm3-pld-20261116/long.txt")?;
    let ids = tokenizer.encode(&text)?;
    ensure!(ids.len() > 1200, "text too short");
    let bits = |array: &Array| -> Result<Vec<u32>> {
        let values = array.astype(Dtype::Float32)?.to_f32()?;
        ensure!(!values.is_empty(), "empty comparison array");
        Ok(values.iter().map(|v| v.to_bits()).collect())
    };
    let mut report = serde_json::json!({"status": "running", "cases": []});
    let mut differing_total = 0usize;
    let mut compared_total = 0usize;
    crate::linear::with_dense_tensor_rows(Some(3), || -> Result<()> {
        for prefix in [299usize, 640] {
            model.reset();
            for chunk in ids[..prefix].chunks(256) {
                model.forward_tokens(chunk)?;
            }
            let snapshot = model.snapshot()?;
            let t = |offset: usize| ids[prefix + offset];
            let alt = |offset: usize| {
                let token = ids[prefix + 300 + offset];
                if (0..8).any(|o| ids[prefix + o] == token) {
                    (token + 11 + offset as u32) % 200_000 + 1000
                } else {
                    token
                }
            };
            // Q27D-37 oracle shape, and a comb of 15 rows / 8 paths (a side leaf at every depth).
            let mut trees = vec![DraftTree {
                tokens: vec![
                    ids[prefix - 1],
                    t(0),
                    alt(0),
                    t(1),
                    alt(1),
                    alt(2),
                    t(2),
                    alt(3),
                    t(3),
                    t(4),
                    t(5),
                    t(6),
                ],
                parents: vec![0, 0, 0, 1, 2, 1, 3, 4, 6, 8, 9, 10],
                depths: vec![0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 7],
            }];
            let mut comb = DraftTree {
                tokens: vec![ids[prefix - 1]],
                parents: vec![0],
                depths: vec![0],
            };
            let mut main = 0usize;
            for depth in 1..=7usize {
                comb.tokens.push(t(depth - 1));
                comb.parents.push(main);
                comb.depths.push(depth);
                let row = comb.tokens.len() - 1;
                comb.tokens.push(alt(10 + depth));
                comb.parents.push(main);
                comb.depths.push(depth);
                main = row;
            }
            if comb.is_valid() {
                trees.push(comb);
            }
            for (shape, tree) in trees.iter().enumerate() {
                ensure!(tree.is_valid(), "oracle tree {shape} is invalid");
                model.restore(snapshot.clone())?;
                let (reference_logits, reference_capture) =
                    crate::qwen35::with_tree_batched_gdn(false, || {
                        model.verify_tree_with_dflash_capture(tree)
                    })?;
                let (reference_logits, reference_capture) =
                    (bits(&reference_logits)?, bits(&reference_capture)?);
                model.restore(snapshot.clone())?;
                let (batched_logits, batched_capture) =
                    crate::qwen35::with_tree_batched_gdn(true, || {
                        model.verify_tree_with_dflash_capture(tree)
                    })?;
                let (batched_logits, batched_capture) =
                    (bits(&batched_logits)?, bits(&batched_capture)?);
                ensure!(
                    reference_logits.len() == batched_logits.len()
                        && reference_capture.len() == batched_capture.len(),
                    "batched verification shapes differ"
                );
                let differing = reference_logits
                    .iter()
                    .zip(&batched_logits)
                    .filter(|(a, b)| a != b)
                    .count()
                    + reference_capture
                        .iter()
                        .zip(&batched_capture)
                        .filter(|(a, b)| a != b)
                        .count();
                differing_total += differing;
                compared_total += reference_logits.len() + reference_capture.len();
                let case = serde_json::json!({"prefix": prefix, "shape": shape, "rows": tree.rows(), "paths": tree.paths().len(),
                    "kind": "verify", "values": reference_logits.len() + reference_capture.len(), "differing": differing});
                println!("{case}");
                report["cases"].as_array_mut().context("cases")?.push(case);
                // Commit every path's full length and a short prefix, then one AR step.
                let mut accepted_sets: Vec<Vec<usize>> = tree.paths();
                accepted_sets.extend(tree.paths().iter().map(|p| p[..p.len().min(2)].to_vec()));
                accepted_sets.push(vec![0]);
                for accepted in accepted_sets {
                    let mut after = Vec::new();
                    for batched in [false, true] {
                        model.restore(snapshot.clone())?;
                        crate::qwen35::with_tree_batched_gdn(batched, || -> Result<()> {
                            model.verify_tree_with_dflash_capture(tree)?;
                            model.commit_dflash_tree(tree, &accepted)
                        })?;
                        let next = model.forward_tokens(&[ids[prefix + 500]])?;
                        after.push((bits(&next)?, model.offset));
                    }
                    let differing = after[0]
                        .0
                        .iter()
                        .zip(&after[1].0)
                        .filter(|(a, b)| a != b)
                        .count();
                    ensure!(after[0].1 == after[1].1, "offsets differ after commit");
                    differing_total += differing;
                    compared_total += after[0].0.len();
                    let case = serde_json::json!({"prefix": prefix, "shape": shape, "kind": "commit",
                        "accepted": accepted, "values": after[0].0.len(), "differing": differing});
                    println!("{case}");
                    report["cases"].as_array_mut().context("cases")?.push(case);
                }
                std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
            }
        }
        Ok(())
    })?;
    report["compared_values"] = compared_total.into();
    report["differing_values"] = differing_total.into();
    report["status"] = if differing_total == 0 {
        "complete"
    } else {
        "differs"
    }
    .into();
    std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
    ensure!(
        differing_total == 0,
        "batched GDN differs from the path-by-path verification"
    );
    Ok(())
}
