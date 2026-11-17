//! LFM3-CERT-HEAD-20261116: physical oracles for the certified greedy head.
//!
//! The references are the production head (`Projection::forward`, MLX GEMV)
//! and `chat_greedy_ids`, the greedy rule of `select_token`; nothing here
//! reuses the screening arithmetic as its own oracle.
use super::Lfm2;
use super::lfm3_verify_tests::{input, model, state_bits, tokens};
use crate::array::{Array, Dtype};
use crate::lfm3_cert_head::{
    CertifiedHead, Int4Head, Int4Prepared, Prepared, SAMPLE_MAX_K, certified_values,
};
use crate::sampling::{History, Sampler};
use anyhow::{Context, Result, ensure};

fn f16_bits(array: &Array) -> Result<Vec<u16>> {
    array.eval()?;
    ensure!(array.dtype() == Dtype::Float16, "expected F16 values");
    let bits = array.to_f16_bits()?;
    ensure!(!bits.is_empty(), "empty oracle output");
    Ok(bits)
}

fn greedy(logits: &Array) -> Result<u32> {
    logits
        .chat_greedy_ids()?
        .first()
        .copied()
        .context("no greedy ID")
}

/// One decode step through the production layers, keeping the normed hidden
/// state and the production logits of that step.
fn step(lm: &mut Lfm2, token: u32) -> Result<(Array, Array)> {
    lm.forward_with(&[token], None, |model, hidden, time| {
        let head = model.head.as_ref().context("dense head")?;
        let logits = head.forward(hidden)?.slice(1, time - 1, time)?;
        hidden.eval()?;
        logits.eval()?;
        Ok((hidden.try_clone()?, logits))
    })
}

/// Hidden states and logits of real greedy decode steps after `prompt`.
fn collect(lm: &mut Lfm2, prompt: &[u32], steps: usize) -> Result<Vec<(Array, Array)>> {
    lm.reset();
    let mut last = None;
    for chunk in prompt.chunks(256) {
        if chunk.len() == 1 || chunk.len() >= 24 {
            last = Some(lm.forward(chunk)?);
        } else {
            for &token in chunk {
                last = Some(lm.forward(&[token])?);
            }
        }
    }
    let mut next = greedy(&last.context("empty prompt")?)?;
    let mut out = Vec::with_capacity(steps);
    for _ in 0..steps {
        let (hidden, logits) = step(lm, next)?;
        next = greedy(&logits)?;
        out.push((hidden, logits));
    }
    Ok(out)
}

fn fixtures() -> Vec<(&'static str, Vec<u32>)> {
    let span = tokens(48, 77);
    vec![
        (
            "induction",
            [&tokens(30, 78)[..], &span, &tokens(5, 79), &span[..20]].concat(),
        ),
        ("random", tokens(64, 4242)),
        ("short", tokens(3, 9)),
    ]
}

struct Checked {
    steps: usize,
    certified: usize,
    candidates: Vec<usize>,
    bound_violations: usize,
    max_ratio: f64,
    wrong_ids: usize,
}

/// Oracles (1)-(3) for one head copy and one hidden state.
fn check(
    certified: &CertifiedHead,
    prepared: &Prepared,
    weight: &Array,
    hidden: &Array,
    logits: &Array,
    stats: &mut Checked,
) -> Result<()> {
    let dims = hidden.shape()[hidden.shape().len() - 1];
    let x = hidden.reshape(&[1, dims])?;
    let reference = f16_bits(logits)?;
    let rows = reference.len();
    ensure!(
        rows == prepared.scale.len(),
        "logit count differs from head rows"
    );
    // (1) every row recomputed: bit-identical to the production head.
    let mut screened = certified.screen(&x)?;
    let norm = screened.pop().context("norm")?;
    let partial = screened.pop().context("partial")?;
    let upper = screened.pop().context("upper")?;
    let everything = Array::from_f32(&[f32::NEG_INFINITY], &[1])?;
    let all_rows = f16_bits(&certified.exact(weight, &x, &upper, &everything)?)?;
    ensure!(all_rows.len() == rows, "exact kernel value count");
    let mismatches = all_rows
        .iter()
        .zip(&reference)
        .filter(|(a, b)| a != b)
        .count();
    ensure!(
        mismatches == 0,
        "exact kernel differs from the head on {mismatches}/{rows} logits"
    );
    // (2) |ref - A| <= |x| E (+ half an F16 ulp of the stored reference).
    let upper = upper.to_f32()?;
    let norm = f64::from(norm.to_f32()?[0]);
    let x_norm = x.astype(Dtype::Float32)?.to_f32()?;
    let exact_norm = x_norm
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt();
    ensure!(
        norm.is_finite() && norm >= exact_norm,
        "GPU |x| {norm} below its f64 value {exact_norm}"
    );
    for (row, (&bits, &upper)) in reference.iter().zip(&upper).enumerate() {
        let value = half::f16::from_bits(bits).to_f64();
        ensure!(
            value.is_finite() && upper.is_finite(),
            "row {row} not finite"
        );
        let error = norm * f64::from(prepared.bound[row]);
        let approx = f64::from(upper) - error;
        let half_ulp =
            f64::from(half::f16::from_f64(value.abs()).to_f32()).max(6.1e-5) * f64::powi(2.0, -11);
        let distance = (value - approx).abs();
        if distance > error + half_ulp + 1e-6 * f64::from(upper).abs() {
            stats.bound_violations += 1;
        }
        if error > 0.0 {
            stats.max_ratio = stats.max_ratio.max(distance / error);
        }
    }
    // (3) certified IDs equal the greedy rule; candidates are exact.
    let threshold = certified.threshold(&partial)?;
    let upper_gpu = Array::from_f32(&upper, &[rows as i32])?;
    let candidates = f16_bits(&certified.exact(weight, &x, &upper_gpu, &threshold)?)?;
    let negative = half::f16::NEG_INFINITY.to_bits();
    let mut count = 0;
    for (row, (&value, &expected)) in candidates.iter().zip(&reference).enumerate() {
        if value != negative {
            ensure!(value == expected, "candidate {row} is not exact");
            count += 1;
        }
    }
    let expected = greedy(logits)?;
    ensure!(
        candidates[expected as usize] == reference[expected as usize],
        "the greedy row {expected} was screened out"
    );
    let (id, ok) = certified.greedy(weight, &x)?;
    stats.steps += 1;
    stats.candidates.push(count);
    if ok {
        stats.certified += 1;
        if id != expected {
            stats.wrong_ids += 1;
        }
    }
    Ok(())
}

fn report(label: &str, stats: &Checked) {
    let mut sorted = stats.candidates.clone();
    sorted.sort_unstable();
    let pick = |q: f64| sorted[((sorted.len() - 1) as f64 * q) as usize];
    eprintln!(
        "CERT_ORACLE {label} steps={} certified={} wrong_ids={} bound_violations={} \
         max_ratio={:.4} candidates_median={} p90={} max={}",
        stats.steps,
        stats.certified,
        stats.wrong_ids,
        stats.bound_violations,
        stats.max_ratio,
        pick(0.5),
        pick(0.9),
        sorted[sorted.len() - 1]
    );
}

fn checked() -> Checked {
    Checked {
        steps: 0,
        certified: 0,
        candidates: Vec::new(),
        bound_violations: 0,
        max_ratio: 0.0,
        wrong_ids: 0,
    }
}

/// Oracles (1)-(4) on real decode states, synthetic inputs and the mutation.
#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; LFM3-CERT-HEAD oracle"]
fn certified_head_matches_head_bits_bounds_and_greedy_ids() -> Result<()> {
    let mut model = model()?;
    let lm = &mut model.language;
    let weight = lm
        .head
        .as_ref()
        .and_then(super::Projection::dense_weight)
        .context("dense head")?
        .try_clone()?;
    let started = std::time::Instant::now();
    let prepared = CertifiedHead::prepare(&weight)?;
    let prepare_seconds = started.elapsed().as_secs_f64();
    let (rows, dims) = (weight.shape()[0], weight.shape()[1]);
    let certified = CertifiedHead::from_prepared(&prepared, rows, dims)?;
    eprintln!(
        "CERT_ORACLE prepare_seconds={prepare_seconds:.2} retained_bytes={}",
        certified.retained_bytes()?
    );
    let mut stats = checked();
    let mut states = Vec::new();
    for (name, prompt) in fixtures() {
        let steps = collect(lm, &prompt, 40)?;
        let mut local = checked();
        for (hidden, logits) in &steps {
            check(&certified, &prepared, &weight, hidden, logits, &mut local)?;
        }
        report(name, &local);
        stats.steps += local.steps;
        stats.certified += local.certified;
        stats.wrong_ids += local.wrong_ids;
        stats.bound_violations += local.bound_violations;
        stats.max_ratio = stats.max_ratio.max(local.max_ratio);
        stats.candidates.extend(local.candidates);
        states.extend(steps);
    }
    // Synthetic activations at several scales (not real decode states).
    for seed in 0..6u64 {
        let x = input(1, dims, 31 + seed)?.astype(Dtype::Float32)?;
        let x = x
            .scalar_mul(0.25 * (seed + 1) as f32)?
            .astype(Dtype::Float16)?;
        let logits = lm.head.as_ref().context("head")?.forward(&x)?;
        check(&certified, &prepared, &weight, &x, &logits, &mut stats)?;
    }
    report("total", &stats);
    ensure!(stats.bound_violations == 0, "bound violated");
    ensure!(stats.wrong_ids == 0, "certified a wrong ID");
    ensure!(stats.certified > 0, "nothing was certified");
    // (3b) exact zeros: every logit ties at zero, so the step must fall back.
    let zero = Array::zeros_dtype(&[1, dims], Dtype::Float16)?;
    let (_, ok) = certified.greedy(&weight, &zero)?;
    ensure!(!ok, "an all-tie step was certified");
    // (3c) a NaN activation must fall back.
    let mut values = input(1, dims, 5)?.astype(Dtype::Float32)?.to_f32()?;
    values[17] = f32::NAN;
    let nan = Array::from_f32(&values, &[1, dims])?.astype(Dtype::Float16)?;
    let (_, ok) = certified.greedy(&weight, &nan)?;
    ensure!(!ok, "a NaN step was certified");
    // (4) mutation: E x 0.01 must be caught by the bound oracle.
    let mutated = Prepared {
        q: prepared.q.clone(),
        scale: prepared.scale.clone(),
        bound: prepared.bound.iter().map(|b| b * 0.01).collect(),
    };
    let weak = CertifiedHead::from_prepared(&mutated, rows, dims)?;
    let mut caught = checked();
    for (hidden, logits) in &states {
        check(&weak, &mutated, &weight, hidden, logits, &mut caught)?;
    }
    report("mutation_e_x0.01", &caught);
    ensure!(
        caught.bound_violations > 0,
        "the E x 0.01 mutation was not detected"
    );
    Ok(())
}

/// Near ties on a small synthetic head: duplicate and one-ulp rows fall
/// back, a clear winner is certified with the greedy ID of the exact head.
#[test]
#[ignore = "physical Apple GPU; LFM3-CERT-HEAD near-tie oracle"]
fn certified_head_falls_back_on_near_ties() -> Result<()> {
    let (rows, dims) = (128i32, 512i32);
    let x = input(1, dims, 11)?.reshape(&[1, dims])?;
    let xs = x.astype(Dtype::Float32)?.to_f32()?;
    let base = input(rows, dims, 12)?.astype(Dtype::Float32)?.to_f32()?;
    let mut cases = 0;
    for (label, winner, rival, expect_ok) in [
        ("duplicate", 0.08f32, Some(0.0f32), false),
        ("one_ulp", 0.08, Some(1.0), false),
        ("clear", 0.08, None, true),
    ] {
        let mut values: Vec<f32> = base.iter().map(|v| v * 0.01).collect();
        for k in 0..dims as usize {
            values[5 * dims as usize + k] = xs[k] * winner;
        }
        if let Some(nudge) = rival {
            for k in 0..dims as usize {
                values[93 * dims as usize + k] = values[5 * dims as usize + k];
            }
            if nudge > 0.0 {
                // Raise one weight of row 93 by one F16 ulp.
                let k = (0..dims as usize)
                    .max_by(|&a, &b| xs[a].total_cmp(&xs[b]))
                    .unwrap();
                let w = half::f16::from_f32(values[93 * dims as usize + k]);
                values[93 * dims as usize + k] = half::f16::from_bits(w.to_bits() + 1).to_f32();
            }
        }
        let weight = Array::from_f32(&values, &[rows, dims])?.astype(Dtype::Float16)?;
        weight.eval()?;
        let prepared = CertifiedHead::prepare(&weight)?;
        let certified = CertifiedHead::from_prepared(&prepared, rows, dims)?;
        let reference = x
            .reshape(&[1, 1, dims])?
            .matmul(&weight.transpose(&[1, 0])?)?;
        let expected = greedy(&reference)?;
        let (id, ok) = certified.greedy(&weight, &x)?;
        let mut stats = checked();
        check(&certified, &prepared, &weight, &x, &reference, &mut stats)?;
        eprintln!("CERT_TIE {label} expected={expected} id={id} certified={ok}");
        ensure!(ok == expect_ok, "{label}: certified={ok}");
        ensure!(!ok || id == expected, "{label}: wrong certified ID");
        ensure!(stats.bound_violations == 0, "{label}: bound violated");
        cases += 1;
    }
    ensure!(cases == 3, "near-tie cases skipped");
    Ok(())
}

/// Oracle (5): decoding with the certified step emits the T1 greedy IDs and
/// leaves bit-identical caches, for every budget of the fixture.
#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; LFM3-CERT-HEAD session oracle"]
fn certified_decode_matches_t1_greedy_ids_and_cache() -> Result<()> {
    let mut model = model()?;
    ensure!(model.enable_certified_head()?, "LFM3 head not supported");
    let mut total = [0usize; 2];
    for (name, prompt) in fixtures() {
        for budget in [1usize, 2, 7, 48] {
            let mut ids = [Vec::new(), Vec::new()];
            let mut states = [Vec::new(), Vec::new()];
            for (mode, (ids, states)) in ids.iter_mut().zip(&mut states).enumerate() {
                model.reset();
                let mut logits = None;
                for chunk in prompt.chunks(256) {
                    logits = Some(if chunk.len() == 1 || chunk.len() >= 24 {
                        model.forward(chunk)?
                    } else {
                        let mut last = None;
                        for &token in chunk {
                            last = Some(model.forward(&[token])?);
                        }
                        last.context("chunk")?
                    });
                }
                let mut next = greedy(&logits.context("prompt")?)?;
                loop {
                    ids.push(next);
                    if ids.len() == budget {
                        break;
                    }
                    next = if mode == 0 {
                        greedy(&model.forward(&[next])?)?
                    } else {
                        let (id, ok) = model.forward_greedy_certified(next)?;
                        total[usize::from(!ok)] += 1;
                        id
                    };
                }
                *states = state_bits(&model.language)?;
            }
            ensure!(ids[0] == ids[1], "{name} budget {budget}: IDs differ");
            ensure!(
                states[0] == states[1] && !states[0].is_empty(),
                "{name} budget {budget}: caches differ"
            );
        }
        eprintln!("CERT_SESSION {name} ok");
    }
    eprintln!("CERT_SESSION certified={} fallbacks={}", total[0], total[1]);
    ensure!(total[0] > 0, "no certified step");
    Ok(())
}

/// Indicative GPU cost (not a generation benchmark): production head plus
/// `chat_greedy_ids` against the certified pipeline, on fixed real states.
#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; LFM3-CERT-HEAD microbenchmark"]
fn certified_head_microbenchmark() -> Result<()> {
    let mut model = model()?;
    let lm = &mut model.language;
    let weight = lm
        .head
        .as_ref()
        .and_then(super::Projection::dense_weight)
        .context("dense head")?
        .try_clone()?;
    let certified = CertifiedHead::build(&weight)?;
    let states = collect(lm, &fixtures()[0].1, 16)?;
    let head = lm.head.as_ref().context("head")?;
    let inputs = states
        .iter()
        .map(|(hidden, _)| hidden.reshape(&[1, weight.shape()[1]]))
        .collect::<Result<Vec<_>>>()?;
    let mut times = [Vec::new(), Vec::new()];
    for round in 0..12 {
        for order in 0..2 {
            let mode = (round + order) % 2;
            let started = std::time::Instant::now();
            for (x, (hidden, _)) in inputs.iter().zip(&states) {
                if mode == 0 {
                    greedy(&head.forward(hidden)?)?;
                } else {
                    certified.greedy(&weight, x)?;
                }
            }
            if round >= 2 {
                times[mode].push(started.elapsed().as_secs_f64() * 1e3 / inputs.len() as f64);
            }
        }
    }
    for t in &mut times {
        t.sort_by(f64::total_cmp);
    }
    eprintln!(
        "CERT_BENCH head_ms_median={:.3} certified_ms_median={:.3} (n={})",
        times[0][times[0].len() / 2],
        times[1][times[1].len() / 2],
        times[0].len()
    );
    Ok(())
}

/// Samplers of the L1 oracles: Desktop default and LFM3 presets, a nucleus
/// with presence penalty, and the largest certified top-k.
fn samplers() -> Vec<Sampler> {
    [
        (0.2, 80, 1., 1.05, 0.),
        (0.1, 50, 1., 1.1, 0.),
        (0.7, 20, 0.9, 1., 0.3),
        (0.8, 256, 1., 1., 0.),
    ]
    .into_iter()
    .map(|(t, k, p, r, q)| Sampler::new(t, k, p, r, q).unwrap())
    .collect()
}

/// L1 oracle on real decode states: every candidate logit is bit-identical
/// to the production head, the unpenalized top-k are candidates, and an
/// accepted row gives the canonical distribution of the production
/// `log_probs` row. Mutation: dropping the best candidate is detected.
#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; L1 sampled certified head oracle"]
fn sampled_head_matches_full_distribution() -> Result<()> {
    let mut model = model()?;
    let lm = &mut model.language;
    let weight = lm
        .head
        .as_ref()
        .and_then(super::Projection::dense_weight)
        .context("dense head")?
        .try_clone()?;
    let certified = CertifiedHead::build(&weight)?;
    let samplers = samplers();
    let vocab = usize::try_from(weight.shape()[0])?;
    let (mut accepted, mut rejected, mut steps) = (vec![0usize; 4], vec![0usize; 4], 0);
    let mut overflow = vec![0usize; 4];
    let (mut candidates, mut mutation_caught, mut mutation_accepted) = (Vec::new(), 0, 0);
    for (name, prompt) in fixtures() {
        let states = collect(lm, &prompt, 40)?;
        let history = History::from_tokens(&prompt[prompt.len().saturating_sub(64)..]);
        for (hidden, logits) in &states {
            let x = hidden.reshape(&[1, weight.shape()[1]])?;
            let reference = f16_bits(logits)?;
            let full = logits.log_probs()?.to_f32()?;
            ensure!(full.len() == vocab && reference.len() == vocab, "row size");
            steps += 1;
            for (index, sampler) in samplers.iter().enumerate() {
                let k = sampler.certified_k(SAMPLE_MAX_K).context("certified k")?;
                let rows = certified.sample(&weight, &x, k)?;
                ensure!(rows.lse.is_finite(), "{name}: lse not finite");
                for (&id, &bits) in rows.ids.iter().zip(&rows.logits) {
                    ensure!(
                        bits == reference[id as usize],
                        "{name}: candidate {id} not exact"
                    );
                }
                if rows.overflow {
                    // More than SAMPLE_CAPACITY candidates: the step must fall back.
                    ensure!(
                        certified_values(&rows, vocab, sampler, &history).is_none(),
                        "{name}: an overflowing row was certified"
                    );
                    overflow[index] += 1;
                    rejected[index] += 1;
                    continue;
                }
                let mut order: Vec<usize> = (0..vocab).collect();
                order.sort_by(|&a, &b| full[b].total_cmp(&full[a]));
                let kth = full[order[k - 1]];
                for &row in order.iter().take_while(|&&row| full[row] > kth) {
                    ensure!(
                        rows.ids.contains(&(row as u32)),
                        "{name}: top row {row} screened out"
                    );
                }
                candidates.push(rows.ids.len());
                let expected = sampler
                    .distribution(full.clone(), &history, &[])?
                    .canonical();
                match certified_values(&rows, vocab, sampler, &history) {
                    Some(values) => {
                        accepted[index] += 1;
                        for &id in &rows.ids {
                            ensure!(
                                values[id as usize].to_bits() == full[id as usize].to_bits(),
                                "{name}: log probability of {id} differs"
                            );
                        }
                        let got = sampler
                            .distribution(values.clone(), &history, &[])?
                            .canonical();
                        ensure!(got == expected, "{name}: certified distribution differs");
                        // Mutation: drop the best candidate.
                        let best = (0..rows.ids.len())
                            .max_by(|&a, &b| {
                                half::f16::from_bits(rows.logits[a])
                                    .to_f32()
                                    .total_cmp(&half::f16::from_bits(rows.logits[b]).to_f32())
                            })
                            .context("best candidate")?;
                        let mut weak = rows.clone();
                        weak.ids.remove(best);
                        weak.logits.remove(best);
                        if let Some(values) = certified_values(&weak, vocab, sampler, &history) {
                            mutation_accepted += 1;
                            let got = sampler.distribution(values, &history, &[])?.canonical();
                            mutation_caught += usize::from(got != expected);
                        }
                    }
                    None => rejected[index] += 1,
                }
            }
        }
    }
    candidates.sort_unstable();
    eprintln!(
        "L1_ORACLE steps={steps} accepted={accepted:?} rejected={rejected:?} overflow={overflow:?} \
         candidates_median={} p90={} max={} mutation_accepted={mutation_accepted} \
         mutation_caught={mutation_caught}",
        candidates[candidates.len() / 2],
        candidates[candidates.len() * 9 / 10],
        candidates[candidates.len() - 1]
    );
    ensure!(
        accepted.iter().all(|&n| n > 0),
        "a sampler was never certified"
    );
    ensure!(
        mutation_accepted > 0 && mutation_caught == mutation_accepted,
        "the dropped-candidate mutation was not detected"
    );
    Ok(())
}

/// Prompt prefill shared by the session oracles; returns the last logits.
fn prefill(model: &mut crate::lfm3::Lfm3, prompt: &[u32]) -> Result<Array> {
    model.reset();
    let mut logits = None;
    for chunk in prompt.chunks(256) {
        logits = Some(if chunk.len() == 1 || chunk.len() >= 24 {
            model.forward(chunk)?
        } else {
            let mut last = None;
            for &token in chunk {
                last = Some(model.forward(&[token])?);
            }
            last.context("chunk")?
        });
    }
    logits.context("prompt")
}

/// L1 session oracle: with the same seed, sampled decoding through
/// `forward_sampled_certified` draws the IDs of the full head (both sides in
/// canonical order) and leaves bit-identical caches; the check oracle of the
/// bridge reports no mismatch.
#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; L1 sampled session oracle"]
fn sampled_decode_matches_full_head_and_cache() -> Result<()> {
    let mut model = model()?;
    ensure!(model.enable_certified_head()?, "LFM3 head not supported");
    let mut total = [0usize; 2];
    for sampler in &samplers()[..2] {
        for (name, prompt) in fixtures() {
            for budget in [1usize, 7, 48] {
                let mut ids = [Vec::new(), Vec::new()];
                let mut states = [Vec::new(), Vec::new()];
                for mode in 0..2 {
                    let logits = prefill(&mut model, &prompt)?;
                    let mut random = 0x5eed_u64 + budget as u64;
                    let mut history = History::default();
                    let mut next = sampler
                        .distribution(logits.log_probs()?.to_f32()?, &history, &[])?
                        .canonical()
                        .sample(&mut random);
                    loop {
                        ids[mode].push(next);
                        history.push(next);
                        if ids[mode].len() == budget {
                            break;
                        }
                        let values = if mode == 0 {
                            model.forward(&[next])?.log_probs()?.to_f32()?
                        } else {
                            let step =
                                model.forward_sampled_certified(next, sampler, &history, true)?;
                            total[usize::from(!step.certified)] += 1;
                            if let Some(reference) = step.reference {
                                ensure!(
                                    sampler.distribution(reference, &history, &[])?.canonical()
                                        == sampler
                                            .distribution(step.values.clone(), &history, &[])?
                                            .canonical(),
                                    "{name}: check oracle mismatch"
                                );
                            }
                            step.values
                        };
                        next = sampler
                            .distribution(values, &history, &[])?
                            .canonical()
                            .sample(&mut random);
                    }
                    states[mode] = state_bits(&model.language)?;
                }
                ensure!(ids[0] == ids[1], "{name} budget {budget}: IDs differ");
                ensure!(
                    states[0] == states[1] && !states[0].is_empty(),
                    "{name} budget {budget}: caches differ"
                );
            }
        }
    }
    eprintln!("L1_SESSION certified={} fallbacks={}", total[0], total[1]);
    ensure!(total[0] > 0, "no certified sampled step");
    Ok(())
}

/// L2 oracle on real decode states and synthetic inputs: the int4 screen
/// bounds the production logits, certified IDs are the greedy IDs, zeros and
/// NaN fall back. Mutation: E x 0.01 violates the bound.
#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; L2 int4 screen oracle"]
fn int4_head_matches_bounds_and_greedy_ids() -> Result<()> {
    let mut model = model()?;
    let lm = &mut model.language;
    let weight = lm
        .head
        .as_ref()
        .and_then(super::Projection::dense_weight)
        .context("dense head")?
        .try_clone()?;
    let (rows, dims) = (weight.shape()[0], weight.shape()[1]);
    let bits = weight.to_f16_bits()?;
    let started = std::time::Instant::now();
    let prepared = crate::lfm3_cert_head::prepare_int4_rows(&bits, usize::try_from(dims)?)?;
    let prepare_seconds = started.elapsed().as_secs_f64();
    let head = Int4Head::from_prepared(&prepared, rows, dims)?;
    eprintln!(
        "L2_ORACLE prepare_seconds={prepare_seconds:.2} retained_bytes={}",
        head.retained_bytes()?
    );
    let mut states = Vec::new();
    for (_, prompt) in fixtures() {
        states.extend(collect(lm, &prompt, 40)?);
    }
    for seed in 0..6u64 {
        let x = input(1, dims, 31 + seed)?.astype(Dtype::Float32)?;
        let x = x
            .scalar_mul(0.25 * (seed + 1) as f32)?
            .astype(Dtype::Float16)?;
        let logits = lm.head.as_ref().context("head")?.forward(&x)?;
        states.push((x, logits));
    }
    let run = |head: &Int4Head, prepared: &Int4Prepared| -> Result<[usize; 4]> {
        // [steps, certified, wrong IDs, bound violations]
        let mut stats = [0usize; 4];
        for (hidden, logits) in &states {
            let x = hidden.reshape(&[1, dims])?;
            let reference = f16_bits(logits)?;
            let mut screened = head.screen(&x)?;
            let norm = f64::from(screened.pop().context("norm")?.to_f32()?[0]);
            let upper = screened.remove(0).to_f32()?;
            for (row, (&bits, &upper)) in reference.iter().zip(&upper).enumerate() {
                let value = half::f16::from_bits(bits).to_f64();
                ensure!(
                    value.is_finite() && upper.is_finite(),
                    "row {row} not finite"
                );
                let error = norm * f64::from(prepared.bound[row]);
                let approx = f64::from(upper) - error;
                let half_ulp = f64::from(half::f16::from_f64(value.abs()).to_f32()).max(6.1e-5)
                    * f64::powi(2.0, -11);
                if (value - approx).abs() > error + half_ulp + 1e-6 * f64::from(upper).abs() {
                    stats[3] += 1;
                }
            }
            let (id, ok) = head.greedy(&weight, &x)?;
            stats[0] += 1;
            if ok {
                stats[1] += 1;
                stats[2] += usize::from(id != greedy(logits)?);
            }
        }
        Ok(stats)
    };
    let stats = run(&head, &prepared)?;
    eprintln!(
        "L2_ORACLE steps={} certified={} wrong_ids={} bound_violations={}",
        stats[0], stats[1], stats[2], stats[3]
    );
    ensure!(stats[3] == 0, "int4 bound violated");
    ensure!(stats[2] == 0, "int4 head certified a wrong ID");
    ensure!(stats[1] > 0, "nothing was certified");
    let zero = Array::zeros_dtype(&[1, dims], Dtype::Float16)?;
    ensure!(
        !head.greedy(&weight, &zero)?.1,
        "an all-tie step was certified"
    );
    let mut values = input(1, dims, 5)?.astype(Dtype::Float32)?.to_f32()?;
    values[17] = f32::NAN;
    let nan = Array::from_f32(&values, &[1, dims])?.astype(Dtype::Float16)?;
    ensure!(!head.greedy(&weight, &nan)?.1, "a NaN step was certified");
    let mutated = Int4Prepared {
        q: prepared.q.clone(),
        scale: prepared.scale.clone(),
        bound: prepared.bound.iter().map(|b| b * 0.01).collect(),
    };
    let weak = Int4Head::from_prepared(&mutated, rows, dims)?;
    let caught = run(&weak, &mutated)?;
    eprintln!("L2_ORACLE mutation_e_x0.01 bound_violations={}", caught[3]);
    ensure!(caught[3] > 0, "the E x 0.01 mutation was not detected");
    Ok(())
}

/// L2 session oracle: greedy decoding with the int4 screen emits the T1
/// greedy IDs and leaves bit-identical caches; the check oracle agrees.
#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; L2 int4 session oracle"]
fn int4_decode_matches_t1_greedy_ids_and_cache() -> Result<()> {
    let mut model = model()?;
    ensure!(
        model.enable_certified_head_kind(super::CertScreenKind::Int4)?,
        "LFM3 head not supported"
    );
    let mut total = [0usize; 3];
    for (name, prompt) in fixtures() {
        for budget in [1usize, 2, 7, 48] {
            let mut ids = [Vec::new(), Vec::new()];
            let mut states = [Vec::new(), Vec::new()];
            for mode in 0..2 {
                let logits = prefill(&mut model, &prompt)?;
                let mut next = greedy(&logits)?;
                loop {
                    ids[mode].push(next);
                    if ids[mode].len() == budget {
                        break;
                    }
                    next = if mode == 0 {
                        greedy(&model.forward(&[next])?)?
                    } else {
                        let step = model.forward_greedy_checked(next, true)?;
                        total[usize::from(!step.certified)] += 1;
                        total[2] += usize::from(step.mismatch);
                        step.id
                    };
                }
                states[mode] = state_bits(&model.language)?;
            }
            ensure!(ids[0] == ids[1], "{name} budget {budget}: IDs differ");
            ensure!(
                states[0] == states[1] && !states[0].is_empty(),
                "{name} budget {budget}: caches differ"
            );
        }
    }
    eprintln!(
        "L2_SESSION certified={} fallbacks={} check_mismatches={}",
        total[0], total[1], total[2]
    );
    ensure!(total[0] > 0 && total[2] == 0, "int4 session oracle failed");
    Ok(())
}

/// Indicative GPU cost of the L1 and L2 heads against the production head
/// (not a generation benchmark).
#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; L1/L2 microbenchmark"]
fn boost_heads_microbenchmark() -> Result<()> {
    let mut model = model()?;
    let lm = &mut model.language;
    let weight = lm
        .head
        .as_ref()
        .and_then(super::Projection::dense_weight)
        .context("dense head")?
        .try_clone()?;
    let int8 = CertifiedHead::build(&weight)?;
    let int4 = Int4Head::build(&weight)?;
    let states = collect(lm, &fixtures()[0].1, 16)?;
    let head = lm.head.as_ref().context("head")?;
    let inputs = states
        .iter()
        .map(|(hidden, _)| hidden.reshape(&[1, weight.shape()[1]]))
        .collect::<Result<Vec<_>>>()?;
    let labels = [
        "head_logprobs",
        "l1_sample_k80",
        "int8_greedy",
        "l2_int4_greedy",
    ];
    let mut times = vec![Vec::new(); labels.len()];
    for round in 0..12 {
        for order in 0..labels.len() {
            let mode = (round + order) % labels.len();
            let started = std::time::Instant::now();
            for (x, (hidden, _)) in inputs.iter().zip(&states) {
                match mode {
                    0 => {
                        head.forward(hidden)?.log_probs()?.to_f32()?;
                    }
                    1 => {
                        int8.sample(&weight, x, 80)?;
                    }
                    2 => {
                        int8.greedy(&weight, x)?;
                    }
                    _ => {
                        int4.greedy(&weight, x)?;
                    }
                }
            }
            if round >= 2 {
                times[mode].push(started.elapsed().as_secs_f64() * 1e3 / inputs.len() as f64);
            }
        }
    }
    for (label, t) in labels.iter().zip(&mut times) {
        t.sort_by(f64::total_cmp);
        eprintln!(
            "BOOST_BENCH {label}_ms_median={:.3} (n={})",
            t[t.len() / 2],
            t.len()
        );
    }
    Ok(())
}
