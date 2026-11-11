//! Prefill component prototypes. The production dispatch is unchanged.
use super::*;
use crate::checkpoint;
use std::{path::Path, time::Instant};

use super::lfm3_prefill::{grouped_qmm, reconstruct, reconstruct_mm};

fn input(rows: i32, width: i32, strided: bool) -> Result<Array> {
    let f = |r: i32, c: i32| ((r * 37 + c * 13) % 997) as f32 / 997. - 0.5;
    let x = if strided {
        let values = (0..width)
            .flat_map(|c| (0..rows).map(move |r| f(r, c)))
            .collect::<Vec<_>>();
        Array::from_f32(&values, &[width, rows])?
            .astype(Dtype::Float16)?
            .transpose(&[1, 0])?
    } else {
        let values = (0..rows)
            .flat_map(|r| (0..width).map(move |c| f(r, c)))
            .collect::<Vec<_>>();
        Array::from_f32(&values, &[rows, width])?.astype(Dtype::Float16)?
    };
    let x = x.reshape(&[1, rows, width])?;
    x.eval()?;
    Ok(x)
}

fn compare(name: &str, exact: bool, actual: &[Array], reference: &[Array]) -> Result<f32> {
    ensure!(!actual.is_empty() && actual.len() == reference.len());
    let mut max_abs = 0f32;
    for (a, e) in actual.iter().zip(reference) {
        ensure!(a.shape() == e.shape() && a.dtype() == e.dtype());
        let a_values = a.to_f32()?;
        let e_values = e.to_f32()?;
        ensure!(
            !a_values.is_empty()
                && a_values.len() == e_values.len()
                && e_values.iter().any(|v| v.abs() > 0.0001)
        );
        for (&a, &e) in a_values.iter().zip(&e_values) {
            ensure!(
                a.is_finite() && e.is_finite() && (a - e).abs() <= 0.002 + 0.002 * e.abs(),
                "prefill mismatch {name}: {a} != {e}"
            );
            max_abs = max_abs.max((a - e).abs());
        }
        if exact {
            ensure!(
                a.to_bytes()? == e.to_bytes()?,
                "grouped QMM is not bit exact: {name}"
            );
        }
    }
    Ok(max_abs)
}

fn screen(
    name: &str,
    exact: bool,
    rows: i32,
    mut run: impl FnMut(bool) -> Result<Vec<Array>>,
) -> Result<()> {
    let baseline = run(false)?;
    let candidate = run(true)?;
    let max_abs = compare(name, exact, &candidate, &baseline)?;
    drop(baseline);
    drop(candidate);
    for _ in 0..10 {
        for enabled in [false, true] {
            for x in run(enabled)? {
                x.eval()?;
            }
        }
    }
    let mut samples = [Vec::new(), Vec::new()];
    for repeat in 0..20 {
        for arm in if repeat % 2 == 0 { [0, 1] } else { [1, 0] } {
            let start = Instant::now();
            for _ in 0..10 {
                for x in run(arm == 1)? {
                    x.eval()?;
                }
            }
            samples[arm].push(start.elapsed().as_secs_f64() * 100.);
        }
    }
    println!(
        "LFM3_PREFILL_SCREEN {}",
        serde_json::json!({"name":name,"rows":rows,"max_abs":max_abs,"bit_exact_required":exact,"samples_ms":samples,"scope":"projection only, reconstruction included in P1"})
    );
    Ok(())
}

#[test]
#[ignore = "physical M5 LFM3 MCG8; reconstructed weights must match independent CPU codec"]
fn reconstructed_weights_match_independent_cpu_codec() -> Result<()> {
    let cp = checkpoint::inspect(Path::new(&std::env::var("MLXL3_LFM3_MODEL")?))?;
    for prefix in [
        "model.layers.0.feed_forward.w1",
        "model.layers.0.feed_forward.w2",
        "model.layers.0.conv.in_proj",
        "model.layers.2.self_attn.o_proj",
    ] {
        let mut linear = Exl3Linear::from_checkpoint(&cp, prefix)?;
        let gpu = reconstruct(&linear)?;
        gpu.eval()?;
        let packed = linear
            .trellis
            .to_bytes()?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_ne_bytes(*b))
            .collect::<Vec<_>>();
        let cpu = codec::reconstruct_inner(
            &packed,
            linear.k,
            linear.cb,
            linear.rows as usize,
            linear.cols as usize,
        )?;
        ensure!(
            cpu.len() == (linear.rows * linear.cols) as usize && cpu.iter().all(|v| v.is_finite())
        );
        let expected = cpu
            .iter()
            .flat_map(|v| v.to_bits().to_ne_bytes())
            .collect::<Vec<_>>();
        ensure!(
            gpu.to_bytes()? == expected,
            "reconstruction differs from CPU: {prefix}"
        );
        linear.k = 7;
        ensure!(reconstruct(&linear).is_err());
        println!(
            "LFM3_RECON_QUALITY {}",
            serde_json::json!({"name":prefix,"elements":cpu.len(),"bytes":expected.len(),"bit_exact":true})
        );
    }
    Ok(())
}

#[test]
#[ignore = "physical M5 LFM3 MCG8; P1 parity before inclusive reconstruction/GEMM micro"]
fn reconstruct_then_gemm_prefill_component_screen() -> Result<()> {
    let cp = checkpoint::inspect(Path::new(&std::env::var("MLXL3_LFM3_MODEL")?))?;
    for prefix in [
        "model.layers.0.feed_forward.w1",
        "model.layers.0.feed_forward.w2",
        "model.layers.0.conv.in_proj",
        "model.layers.2.self_attn.o_proj",
    ] {
        let linear = Exl3Linear::from_checkpoint(&cp, prefix)?;
        for rows in [24, 64, 66, 78, 128, 256] {
            let x = input(rows, linear.rows, false)?;
            screen(&format!("P1:{prefix}"), false, rows, |new| {
                Ok(vec![if new {
                    reconstruct_mm(&linear, &x)?
                } else {
                    linear.forward(&x)?
                }])
            })?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "physical M5 LFM3 MCG8; P2 exact ragged grouped QMM then component micro"]
fn grouped_tensor_qmm_prefill_component_screen() -> Result<()> {
    let cp = checkpoint::inspect(Path::new(&std::env::var("MLXL3_LFM3_MODEL")?))?;
    for (name, prefixes) in [
        (
            "ffn",
            vec![
                "model.layers.0.feed_forward.w1",
                "model.layers.0.feed_forward.w3",
            ],
        ),
        (
            "qkv",
            vec![
                "model.layers.2.self_attn.q_proj",
                "model.layers.2.self_attn.k_proj",
                "model.layers.2.self_attn.v_proj",
            ],
        ),
    ] {
        let mut group = Exl3Group::new(
            prefixes
                .iter()
                .map(|p| Exl3Linear::from_checkpoint(&cp, p))
                .collect::<Result<Vec<_>>>()?,
        )?;
        ensure!(grouped_qmm(&group, &input(23, group.rows, false)?).is_err());
        let original = group.widths[0];
        group.widths[0] = 128;
        ensure!(grouped_qmm(&group, &input(66, group.rows, false)?).is_err());
        group.widths[0] = original;
        let x = input(66, group.rows, true)?;
        compare(name, true, &grouped_qmm(&group, &x)?, &group.forward(&x)?)?;
        for rows in [24, 34, 64, 66, 78, 128, 256] {
            let x = input(rows, group.rows, false)?;
            screen(&format!("P2:{name}"), true, rows, |new| {
                if new {
                    grouped_qmm(&group, &x)
                } else {
                    group.forward(&x)
                }
            })?;
        }
    }
    Ok(())
}

#[test]
#[ignore = "physical M5 LFM3 MCG8; production prefill paths, all states and continuation logits"]
fn prefill_production_composition_preserves_model_states_and_logits() -> Result<()> {
    struct Restore(Option<[bool; 2]>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PREFILL_OVERRIDE.set(self.0);
        }
    }
    let _restore = Restore(PREFILL_OVERRIDE.replace(Some([false; 2])));
    let cp = checkpoint::inspect(Path::new(&std::env::var("MLXL3_LFM3_MODEL")?))?;
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(std::env::var("MLXL3_LFM3_REFERENCE")?)?)?;
    let source: Vec<u32> = serde_json::from_value(fixture["cases"][0]["tokens"].clone())?;
    ensure!(source.len() >= 256);
    let tokens: Vec<u32> = source.into_iter().cycle().take(1098).collect();
    type State = (String, Vec<i32>, Dtype, Vec<f32>, Vec<u8>);
    type Capture = (Vec<f32>, Vec<u8>, Vec<State>);
    let capture = |model: &crate::lfm3::Lfm3, logits: &Array| -> Result<Capture> {
        ensure!(logits.shape() == [1, 1, 128000]);
        let values = logits.to_f32()?;
        ensure!(
            values.len() == 128000
                && values.iter().all(|v| v.is_finite())
                && values.iter().any(|v| v.abs() > 0.01)
        );
        let states = model
            .state_arrays()
            .into_iter()
            .map(|(name, x)| {
                let values = x.to_f32()?;
                ensure!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
                Ok((name, x.shape().to_vec(), x.dtype(), values, x.to_bytes()?))
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(states.len() == 35);
        Ok((values, logits.to_bytes()?, states))
    };
    for length in [24, 66, 78, 128, 256, 512, 1090] {
        let mut reference: Option<Vec<Capture>> = None;
        for override_flags in [
            Some([false, false]),
            Some([false, true]),
            Some([true, false]),
            Some([true, true]),
            None,
        ] {
            let flags = override_flags.unwrap_or([true, true]);
            PREFILL_OVERRIDE.set(override_flags);
            PREFILL_DISPATCHES.set([0; 2]);
            let mut model = crate::lfm3::Lfm3::from_checkpoint(&cp)?;
            let mut logits = None;
            for chunk in tokens[..length].chunks(256) {
                logits = Some(model.forward(chunk)?);
                logits.as_ref().unwrap().eval()?;
            }
            let mut actual = vec![capture(&model, &logits.context("no prefill")?)?];
            for &token in &tokens[length..length + 8] {
                let logits = model.forward(&[token])?;
                actual.push(capture(&model, &logits)?);
            }
            let expected_counts = tokens[..length]
                .chunks(256)
                .fold([0; 2], |mut counts, chunk| {
                    if flags[0] {
                        counts[0] += if chunk.len() == 256 {
                            if flags[1] { 61 } else { 109 }
                        } else if chunk.len() >= 128 {
                            48
                        } else {
                            0
                        };
                    }
                    if flags[1] && chunk.len() >= 24 {
                        counts[1] += 35;
                    }
                    counts
                });
            ensure!(
                PREFILL_DISPATCHES.get() == expected_counts,
                "dispatches {:?} != {:?} at {length}/{flags:?}",
                PREFILL_DISPATCHES.get(),
                expected_counts
            );
            let (mut max_kl, mut max_state_l2) = (0f64, 0f64);
            if let Some(expected) = &reference {
                ensure!(actual.len() == 9 && expected.len() == 9);
                for (snapshot_index, (a, e)) in actual.iter().zip(expected).enumerate() {
                    if !flags[0] && snapshot_index == 0 {
                        ensure!(a.1 == e.1, "P2 logits differ at {length}");
                    }
                    let lse = |v: &[f32]| {
                        let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                        max + v
                            .iter()
                            .map(|x| (f64::from(*x) - max).exp())
                            .sum::<f64>()
                            .ln()
                    };
                    let (za, ze) = (lse(&a.0), lse(&e.0));
                    let kl =
                        a.0.iter()
                            .zip(&e.0)
                            .map(|(&a, &e)| {
                                (f64::from(e) - ze).exp() * (f64::from(e) - ze - f64::from(a) + za)
                            })
                            .sum::<f64>();
                    ensure!(
                        kl.is_finite() && kl <= 0.001,
                        "KL {kl} at {length}/{flags:?}"
                    );
                    max_kl = max_kl.max(kl);
                    for (a, e) in a.2.iter().zip(&e.2) {
                        ensure!(a.0 == e.0 && a.1 == e.1 && a.2 == e.2 && a.3.len() == e.3.len());
                        if !flags[0] && snapshot_index == 0 {
                            ensure!(a.4 == e.4, "P2 state {} differs at {length}", a.0);
                        }
                        let norm = e.3.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
                        let diff =
                            a.3.iter()
                                .zip(&e.3)
                                .map(|(a, e)| (f64::from(*a) - f64::from(*e)).powi(2))
                                .sum::<f64>();
                        let relative = (diff / norm.max(1e-20)).sqrt();
                        ensure!(
                            relative.is_finite() && relative <= 0.01,
                            "state {} L2 {relative} at {length}/{flags:?}",
                            a.0
                        );
                        max_state_l2 = max_state_l2.max(relative);
                    }
                }
            } else {
                reference = Some(actual);
            }
            println!(
                "LFM3_PREFILL_MODEL {}",
                serde_json::json!({"context":length,"flags":flags,"dispatches":expected_counts,"snapshots":9,"states_per_snapshot":35,"logits_per_snapshot":128000,"max_kl":max_kl,"max_state_relative_l2":max_state_l2,"prefill_bit_exact_required":!flags[0],"default_selection":override_flags.is_none()})
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "physical M5 LFM3 MCG8; production cache-key reuse, ragged shapes and fallbacks"]
fn grouped_prefill_cached_factory_preserves_production_shapes_and_fallbacks() -> Result<()> {
    let cp = checkpoint::inspect(Path::new(&std::env::var("MLXL3_LFM3_MODEL")?))?;
    for prefixes in [
        vec![
            "model.layers.0.feed_forward.w1",
            "model.layers.0.feed_forward.w3",
        ],
        vec![
            "model.layers.2.self_attn.q_proj",
            "model.layers.2.self_attn.k_proj",
            "model.layers.2.self_attn.v_proj",
        ],
    ] {
        let mut group = Exl3Group::new(
            prefixes
                .iter()
                .map(|p| Exl3Linear::from_checkpoint(&cp, p))
                .collect::<Result<Vec<_>>>()?,
        )?;
        for rows in [1, 23, 24, 34, 64, 66, 78, 96, 128, 256, 257] {
            let x = input(rows, group.rows, rows == 66)?;
            group.lfm3_prefill = false;
            let expected = group.forward(&x)?;
            PREFILL_DISPATCHES.set([0; 2]);
            group.lfm3_prefill = true;
            compare(
                "production factory reuse",
                true,
                &group.forward(&x)?,
                &expected,
            )?;
            ensure!(PREFILL_DISPATCHES.get() == [0, usize::from((24..=256).contains(&rows))]);
        }
        let x = input(66, group.rows, false)?.astype(Dtype::Float32)?;
        group.lfm3_prefill = false;
        let expected = group.forward(&x)?;
        PREFILL_DISPATCHES.set([0; 2]);
        group.lfm3_prefill = true;
        compare("F32 fallback", true, &group.forward(&x)?, &expected)?;
        ensure!(PREFILL_DISPATCHES.get() == [0; 2]);
    }
    Ok(())
}
