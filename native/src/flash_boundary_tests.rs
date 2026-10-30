//! Physical oracles for issue 59. These tests do not time or promote a candidate.
use super::*;
use std::path::Path;

fn exact_finite(actual: &Array, expected: &Array) -> Result<()> {
    ensure!(actual.shape() == expected.shape() && actual.dtype() == expected.dtype());
    let values = actual.to_f32()?;
    ensure!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    let reference = expected.to_f32()?;
    ensure!(reference.len() == values.len() && reference.iter().all(|v| v.is_finite()));
    ensure!(
        actual.to_bytes()? == expected.to_bytes()?,
        "candidate differs in bits"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; T17 exact input to out, no checkpoint needed"]
fn t17_epilogue_matches_uncompiled_fp32_sigmoid_product() -> Result<()> {
    for rows in [1, 2, 4, 8] {
        let n = rows as usize * 6144;
        for strided in [false, true] {
            // Include every finite half gate across successive batches, including ±0,
            // subnormals, saturation and values at half rounding boundaries.
            for batch in 0..65536usize.div_ceil(n) {
                let gates = (0..n)
                    .map(|i| {
                        let bits = ((i + batch * n) % 65536) as u16;
                        if f16::from_bits(bits).is_finite() {
                            bits
                        } else {
                            0
                        }
                    })
                    .collect::<Vec<_>>();
                let values = (0..n)
                    .map(|i| f16::from_f32(((i * 37 % 8191) as f32 - 4095.) / 512.).to_bits())
                    .collect::<Vec<_>>();
                let shape = if strided {
                    [1, rows, 128, 48]
                } else {
                    [1, rows, 48, 128]
                };
                let mut normed = Array::from_f16_bits(&values, &shape)?;
                let mut z = Array::from_f16_bits(&gates, &shape)?;
                if strided {
                    normed = normed.transpose(&[0, 1, 3, 2])?;
                    z = z.transpose(&[0, 1, 3, 2])?;
                }
                let expected = normed
                    .astype(Dtype::Float32)?
                    .mul(&z.astype(Dtype::Float32)?.sigmoid()?)?
                    .astype(Dtype::Float16)?;
                exact_finite(&normed.qwen4_gdn_epilogue(&z)?, &expected)?;
                ensure!(
                    normed
                        .qwen4_gdn_epilogue(&z.astype(Dtype::Float32)?)
                        .is_err()
                );
                ensure!(normed.qwen4_gdn_epilogue(&z.slice(3, 0, 1)?).is_err());
            }
        }
    }
    // A SwiGLU substitution must be rejected by this oracle (gate zero, norm one).
    let normed = Array::from_f32(&vec![1.; 6144], &[1, 1, 48, 128])?.astype(Dtype::Float16)?;
    let z = Array::zeros_dtype(&[1, 1, 48, 128], Dtype::Float16)?;
    let expected = normed
        .astype(Dtype::Float32)?
        .mul(&z.astype(Dtype::Float32)?.sigmoid()?)?
        .astype(Dtype::Float16)?;
    let mutant = z.precise_swiglu(&normed)?;
    ensure!(
        exact_finite(&mutant, &expected).is_err(),
        "oracle accepted extra z factor"
    );
    exact_finite(&normed.qwen4_gdn_epilogue(&z)?, &expected)
}

#[test]
#[ignore = "physical Apple GPU and Flash checkpoint; all GDN layers, B1/B2/B4/B8 tapes"]
fn t17_real_gdn_outputs_states_and_every_tape_prefix() -> Result<()> {
    let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let mut layers = 0;
    for layer in 0..48 {
        let p = format!("model.language_model.layers.{layer}.linear_attn");
        if !c.tensors.contains_key(&format!("{p}.A_log")) {
            continue;
        }
        layers += 1;
        let mut gdn = Gdn::load(&c, &p, 1e-6)?;
        for rows in [1, 2, 4, 8] {
            let inputs = (0..rows)
                .map(|row| {
                    let x = (0..H)
                        .map(|i| ((i * 17 + row * 101) % 2039) as f32 / 2039. - 0.5)
                        .collect::<Vec<_>>();
                    Array::from_f32(&x, &[1, H])
                })
                .collect::<Result<Vec<_>>>()?;
            gdn.reset();
            gdn.fused_epilogue = false;
            let (reference, reference_tape) = if rows == 1 {
                (vec![gdn.forward(&inputs[0])?], Vec::new())
            } else {
                gdn.forward_rows(&inputs)?
            };
            let expected_gate = gdn
                .stages
                .iter()
                .find(|(name, _)| name == "gated")
                .context("missing reference input to out")?
                .1
                .try_clone()?;
            let history = gdn
                .history
                .as_ref()
                .context("missing history")?
                .to_bytes()?;
            let state = gdn.state.as_ref().context("missing state")?.to_bytes()?;
            gdn.reset();
            gdn.fused_epilogue = true;
            gdn.fused_epilogue_calls.set(0);
            let (candidate, candidate_tape) = if rows == 1 {
                (vec![gdn.forward(&inputs[0])?], Vec::new())
            } else {
                gdn.forward_rows(&inputs)?
            };
            let actual_gate = &gdn
                .stages
                .iter()
                .find(|(name, _)| name == "gated")
                .context("missing candidate input to out")?
                .1;
            exact_finite(actual_gate, &expected_gate)?;
            ensure!(
                gdn.fused_epilogue_calls.get() == 1,
                "T17 did not execute once per batch"
            );
            ensure!(candidate.len() == rows as usize && candidate_tape.len() == rows as usize - 1);
            for (actual, expected) in candidate.iter().zip(&reference) {
                exact_finite(actual, expected)?;
            }
            ensure!(
                gdn.history
                    .as_ref()
                    .context("missing history")?
                    .to_bytes()?
                    == history
            );
            ensure!(gdn.state.as_ref().context("missing state")?.to_bytes()? == state);
            for (actual, expected) in candidate_tape.into_iter().zip(reference_tape) {
                let (
                    AttentionState::GdnTape {
                        history: ah,
                        tape: at,
                        retained: ar,
                    },
                    AttentionState::GdnTape {
                        history: eh,
                        tape: et,
                        retained: er,
                    },
                ) = (actual, expected)
                else {
                    anyhow::bail!("missing GDN prefix tape")
                };
                ensure!(ar == er && ar > 0 && ar < rows);
                ensure!(
                    ah.context("missing prefix history")?.to_bytes()?
                        == eh.context("missing reference history")?.to_bytes()?
                );
                exact_finite(&at.replay(ar)?, &et.replay(er)?)?;
            }
        }
    }
    ensure!(layers > 0, "fixture selected no GDN layer");
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; T04 two F16 rounding boundaries and native H128"]
fn t04_swiglu_down_preserves_both_half_roundings_and_h128() -> Result<()> {
    let mut mutation_detected = false;
    for rows in [1, 2, 10, 20, 80] {
        for strided in [false, true] {
            let n = rows as usize * 640;
            let values = |seed: usize, scale: f32| {
                (0..n)
                    .map(|i| f16::from_f32(((i * seed % 8191) as f32 - 4095.) / scale).to_bits())
                    .collect::<Vec<_>>()
            };
            let shape = if strided { [640, rows] } else { [rows, 640] };
            let view = |bits: &[u16]| -> Result<Array> {
                let x = Array::from_f16_bits(bits, &shape)?;
                if strided { x.transpose(&[1, 0]) } else { Ok(x) }
            };
            let gate = view(&values(37, 512.))?;
            let up = view(&values(101, 1024.))?;
            let scale = view(&values(73, 2048.))?;
            let expected = gate.swiglu(&up)?.mul(&scale)?;
            let actual = gate.swiglu_scaled(&up, &scale)?;
            exact_finite(&actual, &expected)?;
            exact_finite(
                &actual
                    .reshape(&[rows, 5, 128])?
                    .hadamard_transform(Some(1. / 128f32.sqrt()))?,
                &expected
                    .reshape(&[rows, 5, 128])?
                    .hadamard_transform(Some(1. / 128f32.sqrt()))?,
            )?;
            let mutant = gate
                .astype(Dtype::Float32)?
                .swiglu(&up.astype(Dtype::Float32)?)?
                .mul(&scale.astype(Dtype::Float32)?)?
                .astype(Dtype::Float16)?;
            mutation_detected |= exact_finite(&mutant, &expected).is_err();
            ensure!(
                gate.swiglu_scaled(&up, &scale.astype(Dtype::Float32)?)
                    .is_err()
            );
            ensure!(gate.swiglu_scaled(&up.slice(1, 0, 1)?, &scale).is_err());
        }
    }
    ensure!(
        mutation_detected,
        "oracle did not detect deferred half rounding"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and full Flash checkpoint; candidate vs reference logits/states/rollback"]
fn cloud59_candidates_match_reference_logits_states_and_pair_commits() -> Result<()> {
    let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let mut model = FlashNext::from_checkpoint(&c)?;
    model.multi_token = true;
    let configure = |model: &mut FlashNext, gdn: bool, down: bool| -> Result<()> {
        model.reset();
        for layer in &mut model.layers {
            layer.moe.drain()?;
            layer.moe.cache_for_test_invalidate();
            layer.moe.swiglu_down_boundary = down;
            layer.moe.swiglu_down_calls.set(0);
            if let Attention::Gdn(g) = &mut layer.attention {
                g.fused_epilogue = gdn;
                g.fused_epilogue_calls.set(0);
            }
        }
        Ok(())
    };
    let tokens = [EOS, 78768, 11, 7, 5, 19];
    configure(&mut model, false, false)?;
    let mut logits = Vec::new();
    let mut states = Vec::new();
    for token in tokens {
        let value = model.forward(token)?;
        ensure!(value.shape() == [1, 248320]);
        let host = value.to_f32()?;
        ensure!(host.len() == 248320 && host.iter().all(|v| v.is_finite()));
        logits.push(value.to_bytes()?);
        states.push(super::tests::state_bytes(&model)?);
    }
    // Each optimization is checked on its own, then their combination.
    for (gdn, down) in [(true, false), (false, true), (true, true)] {
        for prefix in [0, 3] {
            for retained in [1, 2] {
                configure(&mut model, gdn, down)?;
                for step in 0..prefix {
                    ensure!(model.forward(tokens[step])?.to_bytes()? == logits[step]);
                    ensure!(super::tests::state_bytes(&model)? == states[step]);
                }
                let before = super::tests::state_bytes(&model)?;
                ensure!(model.verify_pair([248320, 7]).is_err());
                ensure!(super::tests::state_bytes(&model)? == before);
                let (output, _) = model.verify_pair([tokens[prefix], tokens[prefix + 1]])?;
                ensure!(output.shape() == [2, 248320]);
                let values = output.to_f32()?;
                ensure!(values.len() == 2 * 248320 && values.iter().all(|v| v.is_finite()));
                let mut expected = logits[prefix].clone();
                expected.extend(&logits[prefix + 1]);
                ensure!(
                    output.to_bytes()? == expected,
                    "candidate paired logits differ"
                );
                model.commit_pair(retained)?;
                ensure!(
                    super::tests::state_bytes(&model)? == states[prefix + retained as usize - 1]
                );
                let next = prefix + retained as usize;
                ensure!(model.forward(tokens[next])?.to_bytes()? == logits[next]);
                ensure!(super::tests::state_bytes(&model)? == states[next]);
                if gdn {
                    ensure!(
                        model.layers.iter().any(|layer| matches!(
                            &layer.attention, Attention::Gdn(g) if g.fused_epilogue_calls.get() > 0
                        )),
                        "T17 did not run"
                    );
                }
                if down {
                    ensure!(
                        model
                            .layers
                            .iter()
                            .any(|l| l.moe.swiglu_down_calls.get() > 0),
                        "T04 did not run"
                    );
                }
            }
        }
    }
    Ok(())
}
