use super::*;
use half::bf16;

fn finite_values(a: &Array) -> Result<Vec<f32>> {
    let values = a.astype(Dtype::Float32)?.to_f32()?;
    ensure!(
        !values.is_empty() && values.iter().all(|v| v.is_finite()),
        "empty/non-finite tensor"
    );
    Ok(values)
}
fn relative_error(actual: &[f32], expected: &[f32]) -> Result<f64> {
    ensure!(actual.len() == expected.len() && !actual.is_empty());
    ensure!(actual.iter().chain(expected).all(|v| v.is_finite()));
    let square = actual
        .iter()
        .zip(expected)
        .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
        .sum::<f64>();
    let norm = expected.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
    Ok((square / norm.max(1e-30)).sqrt())
}

#[test]
#[ignore = "physical Apple GPU; Q4 oracle independent of MLX quantization"]
fn splash_q4_matches_scalar_quantization_and_independent_dot() -> Result<()> {
    let weights: Vec<f32> = (0..16 * 256)
        .map(|i| {
            let group = i / 64;
            bf16::from_f32(match group % 5 {
                0 => 0.,
                1 => 0.125,
                2 => -0.25,
                _ => ((i * 73 % 257) as f32 - 128.) / 256.,
            })
            .to_f32()
        })
        .collect();
    let matrix = Array::from_f32(&weights, &[16, 256])?.astype(Dtype::BFloat16)?;
    let (w, s, b) = matrix.quantize4()?;
    ensure!(
        w.shape() == [16, 32]
            && s.shape() == [16, 4]
            && b.shape() == [16, 4]
            && w.dtype() == Dtype::UInt32
            && s.dtype() == Dtype::BFloat16
            && b.dtype() == Dtype::BFloat16
    );
    let packed = w.to_u32()?;
    let scales = finite_values(&s)?;
    let biases = finite_values(&b)?;
    for (g, group) in weights.as_chunks::<64>().0.iter().enumerate() {
        let min = group.iter().copied().fold(f32::INFINITY, f32::min);
        let max = group.iter().copied().fold(0., f32::max);
        let negative = min.abs() > max.abs();
        let edge = if negative { min } else { max };
        let mut step = ((max - min) / 15.).max(1e-7) * if negative { 1. } else { -1. };
        let q0 = (edge / step).round();
        let bias = if q0 == 0. {
            0.
        } else {
            step = edge / q0;
            edge
        };
        ensure!(
            scales[g].to_bits() == bf16::from_f32(step).to_f32().to_bits()
                && biases[g].to_bits() == bf16::from_f32(bias).to_f32().to_bits(),
            "Q4 affine parameters differ at {g}"
        );
        for (j, value) in group.iter().enumerate() {
            let n = g * 64 + j;
            let actual = (packed[n / 8] >> ((n % 8) * 4)) & 15;
            let expected = ((value - bias) / step).round().clamp(0., 15.) as u32;
            ensure!(actual == expected, "Q4 code differs at {n}");
        }
    }
    let mut errors = Vec::new();
    for rows in [1, 4, 8, 17, 256] {
        let input: Vec<f32> = (0..rows * 256)
            .map(|i| bf16::from_f32(((i * 17 % 101) as f32 - 50.) / 64.).to_f32())
            .collect();
        let x = Array::from_f32(&input, &[rows as i32, 256])?.astype(Dtype::BFloat16)?;
        let actual = finite_values(&x.affine4(&w, &s, &b, None)?)?;
        let mut expected = vec![0.; rows * 16];
        let mut wrong = expected.clone();
        for row in 0..rows {
            for out in 0..16 {
                let mut sum = 0f64;
                let mut mutated = 0f64;
                for col in 0..256 {
                    let n = out * 256 + col;
                    let group = n / 64;
                    let code = (packed[n / 8] >> ((n % 8) * 4)) & 15;
                    let bad = (packed[n / 8] >> (((n % 8 + 1) % 8) * 4)) & 15;
                    sum += f64::from(input[row * 256 + col])
                        * (f64::from(code) * f64::from(scales[group]) + f64::from(biases[group]));
                    mutated += f64::from(input[row * 256 + col])
                        * (f64::from(bad) * f64::from(scales[group]) + f64::from(biases[group]));
                }
                expected[row * 16 + out] = bf16::from_f64(sum).to_f32();
                wrong[row * 16 + out] = bf16::from_f64(mutated).to_f32();
            }
        }
        let error = relative_error(&actual, &expected)?;
        ensure!(
            error <= 0.004 && relative_error(&actual, &wrong)? > 0.01,
            "Q4 dot oracle/mutation {rows}: {error}"
        );
        errors.push((rows, error));
    }
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = vec![0.; 64];
        values[17] = value;
        ensure!(
            Array::from_f32(&values, &[1, 64])?
                .astype(Dtype::BFloat16)?
                .quantize4()
                .is_err()
        );
    }
    let extreme = (0..64)
        .map(|i| {
            if i % 2 == 0 {
                bf16::MAX.to_f32()
            } else {
                -bf16::MAX.to_f32()
            }
        })
        .collect::<Vec<_>>();
    ensure!(
        Array::from_f32(&extreme, &[1, 64])?
            .astype(Dtype::BFloat16)?
            .quantize4()
            .is_err(),
        "overflowing finite Q4 range accepted"
    );
    for shape in [vec![64], vec![0, 64], vec![1, 65]] {
        ensure!(
            Array::zeros_dtype(&shape, Dtype::BFloat16)?
                .quantize4()
                .is_err()
        );
    }
    println!(
        "FLASH_SPLASH_Q4 {}",
        serde_json::json!({"status":"passed","groups":64,"codes":4096,"dot_errors":errors})
    );
    Ok(())
}

fn kv(rows: i32, start: i32, salt: i32) -> Result<Array> {
    let data: Vec<_> = (0..8 * rows * 128)
        .map(|i| {
            let h = i / (rows * 128);
            let row = i / 128 % rows;
            let d = i % 128;
            (((h * 13 + (row + start) * 7 + d * 3 + salt) % 127) as f32 - 63.) / 64.
        })
        .collect();
    Array::from_f32(&data, &[1, 8, rows, 128])?.astype(Dtype::BFloat16)
}

#[test]
#[ignore = "physical Apple GPU and pinned DFlash2 weights; scalar Markov/confidence oracle"]
fn splash_confidence_matches_independent_markov_scores() -> Result<()> {
    let path = PathBuf::from(std::env::var("MLXL3_FLASH_DRAFT")?);
    let hidden_values = (0..7 * H)
        .map(|i| ((i * 17 % 127) as f32 - 63.) / 64.)
        .collect::<Vec<_>>();
    let hidden = Array::from_f32(&hidden_values, &[7, H])?.astype(Dtype::BFloat16)?;
    let book_ids: Vec<u32> = std::iter::once(7).chain(101..117).collect();
    let ids = Array::from_u32(&book_ids, &[17])?;
    let mut checks = Vec::new();
    for q4 in [false, true] {
        let draft = Draft::load_options(&path, Kind::Dflash2, q4, false)?;
        let projected =
            finite_values(&draft.linear(&hidden, "candidate_selector.hidden_projection")?)?;
        let pred = finite_values(
            &draft
                .weight("candidate_selector.predecessor_codebook")
                .take(&ids, 0)?,
        )?;
        let keys = finite_values(
            &draft
                .weight("candidate_selector.successor_codebook")
                .take(&ids, 0)?,
        )?;
        for rows in [1, 2, 3, 7] {
            let mut logits = vec![-10000.; rows * 248320];
            for row in 0..rows {
                for j in 0..16 {
                    logits[row * 248320 + 101 + j] =
                        bf16::from_f32(((j * 5 + row * 3) % 11) as f32 / 16.).to_f32();
                }
            }
            let array =
                Array::from_f32(&logits, &[rows as i32, 248320])?.astype(Dtype::BFloat16)?;
            let (tokens, confidence) =
                draft.select_logits(&array, &hidden.slice(0, 0, rows as i32)?, 7)?;
            let mut previous = 0;
            let mut errors = Vec::new();
            for row in 0..rows {
                let latent: Vec<_> = (0..256)
                    .map(|d| {
                        bf16::from_f32(pred[previous * 256 + d] * projected[row * 256 + d]).to_f32()
                    })
                    .collect();
                let mut scores = [0f32; 16];
                let mut order: Vec<_> = (0..16).collect();
                order.sort_by(|&a, &b| {
                    logits[row * 248320 + 101 + b]
                        .total_cmp(&logits[row * 248320 + 101 + a])
                        .then(a.cmp(&b))
                });
                for j in 0..16 {
                    let dot: f64 = latent
                        .iter()
                        .zip(&keys[(j + 1) * 256..(j + 2) * 256])
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    scores[j] = logits[row * 248320 + 101 + j] + bf16::from_f64(dot).to_f32();
                }
                let mut best = order[0];
                for j in order {
                    if scores[j] > scores[best] {
                        best = j;
                    }
                }
                let probability = 1.
                    / scores
                        .iter()
                        .map(|&v| f64::from(v - scores[best]).exp())
                        .sum::<f64>();
                ensure!(
                    tokens[row] == 101 + best as u32,
                    "Markov token differs q4={q4}, row={row}"
                );
                let error = (f64::from(confidence[row]) - probability).abs();
                ensure!(error <= 1e-4, "DFlash confidence q4={q4}/{row}: {error}");
                errors.push(error);
                previous = best + 1;
            }
            checks.push(serde_json::json!({"q4":q4,"rows":rows,"confidence_error":errors}));
        }
        drop(draft);
        array::synchronize()?;
        array::clear_cache()?;
    }
    println!(
        "FLASH_SPLASH_CONFIDENCE {}",
        serde_json::json!({"status":"passed","checks":checks})
    );
    Ok(())
}
#[test]
#[ignore = "physical Apple GPU; wraparound and independent SDPA attention oracle"]
fn splash_ring_matches_chronological_cache_after_multiple_wraps() -> Result<()> {
    let mut cache = None;
    let mut position = 0;
    for rows in [1, 7, 2039, 1, 1, 4099] {
        cache = Some(append_ring(
            cache,
            &kv(rows, position, 0)?,
            &kv(rows, position, 19)?,
            position,
        )?);
        position += rows;
        let (keys, values) = cache.as_ref().unwrap();
        let live = position.min(2048);
        let start = position - live;
        let ids = Array::from_u32(
            &(start..position)
                .map(|p| (p % 2048) as u32)
                .collect::<Vec<_>>(),
            &[live],
        )?;
        ensure!(
            keys.take(&ids, 2)?.to_bytes()? == kv(live, start, 0)?.to_bytes()?,
            "ring keys differ at {position}"
        );
        ensure!(
            values.transpose(&[0, 1, 3, 2])?.take(&ids, 2)?.to_bytes()?
                == kv(live, start, 19)?.to_bytes()?,
            "ring values differ at {position}"
        );
        ensure!(keys.shape() == [1, 8, 2048, 128] && values.shape() == [1, 8, 128, 2048]);
    }
    let mut errors = Vec::new();
    for position in [0, 1, 2047, 2048, 2049, 4095, 8193] {
        let past = position.min(2048);
        let start = position - past;
        let (keys, values) = if past == 0 {
            (
                Array::zeros_dtype(&[1, 8, 2048, 128], Dtype::BFloat16)?,
                Array::zeros_dtype(&[1, 8, 128, 2048], Dtype::BFloat16)?,
            )
        } else {
            append_ring(None, &kv(past, start, 0)?, &kv(past, start, 19)?, start)?
        };
        for rows in [1, 2, 4, 8] {
            let q = kv(rows, position, 5)?.reshape(&[1, 8, rows, 128])?;
            let q = Array::concatenate(&[&q, &q, &q, &q], 1)?;
            let k = kv(rows, position, 2)?;
            let v = kv(rows, position, 23)?;
            let actual = ring_attention(&q, &k, &v, &keys, &values, position)?;
            let all_k = Array::concatenate(&[&kv(past, start, 0)?, &k], 2)?;
            let all_v = Array::concatenate(&[&kv(past, start, 19)?, &v], 2)?;
            let mut mask = vec![0f32; (rows * (past + rows)) as usize];
            for row in 0..rows {
                for col in 0..past {
                    if start + col < (position + row + 1 - 2048).max(0) {
                        mask[(row * (past + rows) + col) as usize] = f32::NEG_INFINITY;
                    }
                }
            }
            let expected = Array::sdpa_mask(
                &q,
                &all_k,
                &all_v,
                128f32.sqrt().recip(),
                &Array::from_f32(&mask, &[1, 1, rows, past + rows])?.astype(Dtype::BFloat16)?,
            )?;
            let error = relative_error(&finite_values(&actual)?, &finite_values(&expected)?)?;
            ensure!(
                actual.shape() == [1, 32, rows, 128] && error <= 0.004,
                "ring attention {position}/{rows}: {error}"
            );
            errors.push((position, rows, error));
        }
    }
    println!(
        "FLASH_SPLASH_RING {}",
        serde_json::json!({"status":"passed","append_cases":6,"attention_errors":errors})
    );
    Ok(())
}
