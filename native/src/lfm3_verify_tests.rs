//! LFM3-PLD-20261116: physical oracles for bit-exact multi-row verification.
use super::{FeedForward, Lfm2, Operator, Projection};
use crate::array::{Array, Dtype};
use anyhow::{Context, Result, ensure};

pub(super) fn model() -> Result<crate::lfm3::Lfm3> {
    let path = std::env::var("MLXL3_LFM3_MODEL").context("set MLXL3_LFM3_MODEL")?;
    let cp = crate::checkpoint::inspect(std::path::Path::new(&path))?;
    crate::lfm3::Lfm3::from_checkpoint(&cp)
}

/// Deterministic pseudo-normal F16 input, independent of the code under test.
pub(super) fn input(rows: i32, dim: i32, seed: u64) -> Result<Array> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let values = (0..rows * dim)
        .map(|_| {
            let mut sum = 0.0f32;
            for _ in 0..4 {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                sum += (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
            }
            sum * 1.7
        })
        .collect::<Vec<_>>();
    Array::from_f32(&values, &[1, rows, dim])?.astype(Dtype::Float16)
}

pub(super) fn bits(array: &Array) -> Result<Vec<u32>> {
    array.eval()?;
    let values = array.to_f32()?;
    ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "empty or non-finite oracle output"
    );
    Ok(values.into_iter().map(f32::to_bits).collect())
}

/// Returns (mismatching values, total values, max abs difference).
fn compare(
    label: &str,
    rows: i32,
    dim: i32,
    forward: impl Fn(&Array) -> Result<Array>,
) -> Result<usize> {
    let x = input(rows, dim, u64::from(rows as u32) * 7919 + dim as u64)?;
    let batch = forward(&x)?;
    let per_row = (0..rows)
        .map(|row| forward(&x.slice(1, row, row + 1)?))
        .collect::<Result<Vec<_>>>()?;
    let per_row = Array::concatenate(&per_row.iter().collect::<Vec<_>>(), 1)?;
    ensure!(batch.shape() == per_row.shape(), "{label}: shape mismatch");
    let (a, b) = (bits(&batch)?, bits(&per_row)?);
    ensure!(a.len() == b.len(), "{label}: value count mismatch");
    let mismatches = a.iter().zip(&b).filter(|(x, y)| x != y).count();
    let max = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (f32::from_bits(*x) - f32::from_bits(*y)).abs())
        .fold(0.0f32, f32::max);
    eprintln!(
        "PLD_ORACLE {label} rows={rows} values={} mismatches={mismatches} max_abs={max}",
        a.len()
    );
    Ok(mismatches)
}

fn projection(label: &str, p: &Projection, rows: i32, total: &mut usize) -> Result<()> {
    *total += compare(label, rows, p.logical_input, |x| p.forward(x))?;
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; LFM3-PLD component oracle"]
fn batch_projections_match_single_row_bits() -> Result<()> {
    let model = model()?;
    let lm: &Lfm2 = &model.language;
    let mut total = 0;
    for rows in 2..=4 {
        // Two layers of each kind are enough to classify every projection shape.
        let mut seen = [0usize; 2];
        for (index, layer) in lm.layers.iter().enumerate() {
            let kind = usize::from(matches!(layer.operator, Operator::Attention(_)));
            if seen[kind] >= 2 {
                continue;
            }
            seen[kind] += 1;
            match &layer.operator {
                Operator::Conv(conv) => {
                    projection(&format!("L{index}.conv.in"), &conv.input, rows, &mut total)?;
                    projection(
                        &format!("L{index}.conv.out"),
                        &conv.output,
                        rows,
                        &mut total,
                    )?;
                }
                Operator::Attention(attention) => {
                    for part in 0..3 {
                        total += compare(&format!("L{index}.qkv{part}"), rows, 2048, |x| {
                            Ok(attention.qkv.forward(x)?.remove(part))
                        })?;
                    }
                    projection(
                        &format!("L{index}.attn.out"),
                        &attention.output,
                        rows,
                        &mut total,
                    )?;
                    if let Some(gate) = &attention.gate {
                        projection(&format!("L{index}.attn.gate"), gate, rows, &mut total)?;
                    }
                }
            }
            if let FeedForward::Dense { inputs, w2 } = &layer.feed_forward {
                for part in 0..2 {
                    total += compare(&format!("L{index}.ffn.in{part}"), rows, 2048, |x| {
                        Ok(inputs.forward(x)?.remove(part))
                    })?;
                }
                projection(&format!("L{index}.ffn.w2"), w2, rows, &mut total)?;
            }
        }
        if let Some(head) = &lm.head {
            projection("head.matmul", head, rows, &mut total)?;
        }
    }
    eprintln!("PLD_ORACLE total_mismatches={total}");
    Ok(())
}

/// Candidate multi-row head kernels against the production T1 head (`matmul`).
#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; LFM3-PLD head oracle"]
fn shared_head_rows_match_single_row_matmul_bits() -> Result<()> {
    let model = model()?;
    let head = model.language.head.as_ref().context("LFM3 dense head")?;
    let mut total = 0;
    for rows in 1..=super::LFM3_VERIFY_MAX_ROWS {
        let x = input(rows, head.logical_input, 104_729 + rows as u64)?;
        let single = (0..rows)
            .map(|row| head.forward(&x.slice(1, row, row + 1)?))
            .collect::<Result<Vec<_>>>()?;
        let single = bits(&Array::concatenate(&single.iter().collect::<Vec<_>>(), 1)?)?;
        let matrix = x.reshape(&[rows, head.logical_input])?;
        let mut candidates = vec![("shared", head.forward_dense_rows_shared(&matrix)?)];
        if rows >= 2 {
            candidates.push(("exact_rows", head.forward_dense_rows_exact(&matrix)?));
        }
        for (label, candidate) in candidates {
            let candidate = bits(&candidate)?;
            ensure!(candidate.len() == single.len(), "{label}: value count");
            let mismatches = candidate
                .iter()
                .zip(&single)
                .filter(|(a, b)| a != b)
                .count();
            eprintln!(
                "PLD_ORACLE head.{label} rows={rows} values={} mismatches={mismatches}",
                single.len()
            );
            total += mismatches;
        }
    }
    eprintln!("PLD_ORACLE total_mismatches={total}");
    Ok(())
}

type Snapshot = (i32, Vec<(Option<Array>, Option<Array>)>);

fn snapshot(lm: &Lfm2) -> Result<Snapshot> {
    let layers = lm
        .layers
        .iter()
        .map(|layer| {
            Ok(match &layer.operator {
                Operator::Attention(a) => (
                    a.keys.as_ref().map(Array::try_clone).transpose()?,
                    a.values.as_ref().map(Array::try_clone).transpose()?,
                ),
                Operator::Conv(c) => (c.state.as_ref().map(Array::try_clone).transpose()?, None),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((lm.offset, layers))
}

fn restore(lm: &mut Lfm2, snapshot: &Snapshot) -> Result<()> {
    lm.offset = snapshot.0;
    for (layer, (first, second)) in lm.layers.iter_mut().zip(&snapshot.1) {
        let copy = |a: &Option<Array>| a.as_ref().map(Array::try_clone).transpose();
        match &mut layer.operator {
            Operator::Attention(a) => {
                a.keys = copy(first)?;
                a.values = copy(second)?;
            }
            Operator::Conv(c) => c.state = copy(first)?,
        }
    }
    Ok(())
}

/// Name, shape and raw bits of one state array.
pub(super) type StateBits = (String, Vec<i32>, Vec<u32>);

pub(super) fn state_bits(lm: &Lfm2) -> Result<Vec<StateBits>> {
    lm.state_arrays()
        .into_iter()
        .map(|(name, array)| Ok((name, array.shape().to_vec(), bits(array)?)))
        .collect()
}

pub(super) fn tokens(count: usize, seed: u32) -> Vec<u32> {
    let mut state = seed | 1;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            // Common text IDs keep activations in their usual range.
            1000 + state % 60_000
        })
        .collect()
}

fn prefill(model: &mut crate::lfm3::Lfm3, prompt: &[u32]) -> Result<()> {
    model.reset();
    for chunk in prompt.chunks(256) {
        if chunk.len() == 1 || chunk.len() >= 24 {
            model.forward(chunk)?;
        } else {
            for token in chunk {
                model.forward(&[*token])?;
            }
        }
    }
    for (_, state) in model.state_arrays() {
        state.eval()?;
    }
    Ok(())
}

/// Logits rows, verification commits, states and the following T1 step all
/// match sequential T1 bit for bit, including across the SWA window edge.
#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; LFM3-PLD end-to-end oracle"]
fn verify_rows_and_commits_match_sequential_t1_bits() -> Result<()> {
    let mut model = model()?;
    let mut cases = 0;
    for (prefix, rows) in [
        (24usize, 2usize),
        (24, 4),
        (253, 4),
        (254, 3),
        (300, 4),
        (300, 2),
    ] {
        let prompt = tokens(prefix, 17 + prefix as u32);
        let run = tokens(rows + 1, 911 + prefix as u32 + rows as u32);
        prefill(&mut model, &prompt)?;
        let base = snapshot(&model.language)?;
        // Sequential reference: logits and states after each of rows + 1 steps.
        let mut reference = Vec::new();
        for &token in &run {
            let logits = bits(&model.forward(&[token])?)?;
            ensure!(logits.len() == 128_000, "unexpected T1 logit count");
            reference.push((logits, state_bits(&model.language)?));
        }
        restore(&mut model.language, &base)?;
        let (logits, _) = model.verify_rows(&run[..rows])?;
        ensure!(
            logits.shape() == [rows as i32, 128_000],
            "verification shape"
        );
        let logits = bits(&logits)?;
        for row in 0..rows {
            let actual = &logits[row * 128_000..(row + 1) * 128_000];
            let mismatches = actual
                .iter()
                .zip(&reference[row].0)
                .filter(|(a, b)| a != b)
                .count();
            eprintln!(
                "PLD_E2E prefix={prefix} rows={rows} row={row} logit_mismatches={mismatches}"
            );
            ensure!(mismatches == 0, "verification logits differ from T1");
        }
        ensure!(
            state_bits(&model.language)? == state_bits_from(&base, &mut model)?,
            "verification mutated the cache"
        );
        for accepted in 1..=rows {
            restore(&mut model.language, &base)?;
            let (_, verification) = model.verify_rows(&run[..rows])?;
            model.commit_verified(verification, accepted as i32)?;
            ensure!(model.language.position() as usize == prefix + accepted);
            let states = state_bits(&model.language)?;
            let expected = &reference[accepted - 1].1;
            ensure!(states.len() == expected.len(), "state count differs");
            for (actual, wanted) in states.iter().zip(expected) {
                ensure!(
                    actual.0 == wanted.0 && actual.1 == wanted.1 && actual.2 == wanted.2,
                    "state {} differs after committing {accepted}/{rows}",
                    actual.0
                );
            }
            let next = bits(&model.forward(&[run[accepted]])?)?;
            ensure!(
                next == reference[accepted].0,
                "T1 step after committing {accepted}/{rows} differs"
            );
            eprintln!(
                "PLD_E2E prefix={prefix} rows={rows} commit={accepted} states={} next_step=exact",
                states.len()
            );
            cases += 1;
        }
    }
    // Invalid commits are refused and reset the model.
    let prompt = tokens(40, 5);
    prefill(&mut model, &prompt)?;
    let (_, verification) = model.verify_rows(&tokens(3, 6))?;
    ensure!(model.commit_verified(verification, 4).is_err());
    ensure!(model.language.position() == 0 && model.state_arrays().is_empty());
    prefill(&mut model, &prompt)?;
    let (_, stale) = model.verify_rows(&tokens(2, 7))?;
    model.forward(&[1234])?;
    ensure!(model.commit_verified(stale, 1).is_err());
    ensure!(model.verify_rows(&tokens(1, 8)).is_err());
    ensure!(model.verify_rows(&tokens(5, 9)).is_err());
    eprintln!("PLD_E2E commits={cases} invalid=refused");
    ensure!(cases == 19, "unexpected case count {cases}");
    Ok(())
}

fn state_bits_from(snapshot: &Snapshot, model: &mut crate::lfm3::Lfm3) -> Result<Vec<StateBits>> {
    let current = snapshot_states(model)?;
    restore(&mut model.language, snapshot)?;
    let base = state_bits(&model.language)?;
    restore(&mut model.language, &current)?;
    Ok(base)
}

fn snapshot_states(model: &crate::lfm3::Lfm3) -> Result<Snapshot> {
    snapshot(&model.language)
}

/// Indicative GPU cost of one verification against one T1 step (not a
/// generation benchmark): same cache, alternating order, greedy IDs read back.
#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; LFM3-PLD microbenchmark"]
fn verify_rows_cost_microbenchmark() -> Result<()> {
    let mut model = model()?;
    let prefix = std::env::var("MLXL3_PLD_PREFIX")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(300usize);
    prefill(&mut model, &tokens(prefix, 3))?;
    let base = snapshot(&model.language)?;
    let run = tokens(4, 4);
    let mut samples: Vec<Vec<f64>> = vec![Vec::new(); 4];
    for iteration in 0..50 {
        for slot in 0..4 {
            let rows = (slot + iteration) % 4 + 1;
            restore(&mut model.language, &base)?;
            let started = std::time::Instant::now();
            let ids = if rows == 1 {
                model.forward(&run[..1])?.chat_greedy_ids()?
            } else {
                let (logits, verification) = model.verify_rows(&run[..rows])?;
                let ids = (0..rows as i32)
                    .map(|row| {
                        logits
                            .slice(0, row, row + 1)?
                            .reshape(&[1, 1, 128_000])?
                            .log_probs()?
                            .argmax()?
                            .reshape(&[1])
                    })
                    .collect::<Result<Vec<_>>>()?;
                let ids = Array::concatenate(&ids.iter().collect::<Vec<_>>(), 0)?.to_u32()?;
                model.commit_verified(verification, rows as i32)?;
                ids
            };
            for (_, state) in model.state_arrays() {
                state.eval()?;
            }
            let elapsed = started.elapsed().as_secs_f64() * 1e3;
            ensure!(ids.len() == rows, "missing greedy IDs");
            if iteration >= 10 {
                samples[rows - 1].push(elapsed);
            }
        }
    }
    for (index, values) in samples.iter_mut().enumerate() {
        values.sort_by(f64::total_cmp);
        let median = values[values.len() / 2];
        eprintln!(
            "PLD_BENCH prefix={prefix} rows={} n={} median_ms={median:.3} p10={:.3} p90={:.3}",
            index + 1,
            values.len(),
            values[values.len() / 10],
            values[values.len() * 9 / 10]
        );
    }
    Ok(())
}

fn greedy_id(logits: &Array) -> Result<u32> {
    logits
        .chat_greedy_ids()?
        .first()
        .copied()
        .context("no greedy ID")
}

/// The bridge loop: stop before an end ID, never forward the last budget ID.
fn decode(
    model: &mut crate::lfm3::Lfm3,
    prompt: &[u32],
    budget: usize,
    eos: &[u32],
    lookup: bool,
) -> Result<(Vec<u32>, usize)> {
    model.reset();
    let mut logits = None;
    for chunk in prompt.chunks(256) {
        logits = Some(model.forward(chunk)?);
    }
    let mut next = greedy_id(&logits.context("empty prompt")?)?;
    let mut session = crate::lfm3::PromptLookup::new(prompt);
    let mut generated = Vec::new();
    loop {
        if eos.contains(&next) {
            break;
        }
        generated.push(next);
        if generated.len() == budget {
            break;
        }
        next = if lookup {
            session.advance(model, next, budget - generated.len(), eos, |_| true)?
        } else {
            greedy_id(&model.forward(&[next])?)?
        };
    }
    Ok((generated, session.blocks))
}

/// Emitted IDs and the final cache equal plain T1 greedy decoding for every
/// budget (blocks clipped by the budget) and for an end ID inside a block.
#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_MODEL; LFM3-PLD session oracle"]
fn prompt_lookup_session_matches_t1_greedy_ids_and_cache() -> Result<()> {
    let mut model = model()?;
    // A repeated random span: induction makes the target copy it.
    let span = tokens(48, 77);
    let prompt = [&tokens(30, 78)[..], &span, &tokens(5, 79), &span[..20]].concat();
    let (reference, _) = decode(&mut model, &prompt, 48, &[], false)?;
    let reference_states = state_bits(&model.language)?;
    let mut blocks = 0;
    let mut checked = 0;
    for budget in (1..=24).chain([31, 32, 33, 34, 47, 48]) {
        let (plain, _) = decode(&mut model, &prompt, budget, &[], false)?;
        let plain_states = state_bits(&model.language)?;
        let plain_position = model.language.position();
        let (lookup, used) = decode(&mut model, &prompt, budget, &[], true)?;
        ensure!(
            plain == reference[..budget],
            "T1 replay is not deterministic"
        );
        ensure!(lookup == plain, "budget {budget}: lookup IDs differ");
        ensure!(
            model.language.position() == plain_position
                && state_bits(&model.language)? == plain_states,
            "budget {budget}: lookup cache differs"
        );
        if budget == 48 {
            ensure!(
                plain_states == reference_states,
                "final reference cache differs"
            );
        }
        blocks += used;
        checked += 1;
    }
    ensure!(blocks > 0, "the fixture never produced a lookup block");
    // End IDs taken from inside the copied region.
    let mut stops = 0;
    for index in [6usize, 9, 13, 20, 27] {
        let eos = [reference[index]];
        let (plain, _) = decode(&mut model, &prompt, 48, &eos, false)?;
        let plain_states = state_bits(&model.language)?;
        let (lookup, used) = decode(&mut model, &prompt, 48, &eos, true)?;
        ensure!(lookup == plain, "end ID {index}: lookup IDs differ");
        ensure!(
            state_bits(&model.language)? == plain_states,
            "end ID {index}: lookup cache differs"
        );
        eprintln!(
            "PLD_SESSION eos_index={index} emitted={} blocks={used}",
            plain.len()
        );
        stops += 1;
    }
    eprintln!(
        "PLD_SESSION budgets={checked} blocks={blocks} eos_cases={stops} reference={reference:?}"
    );
    Ok(())
}
