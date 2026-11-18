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
