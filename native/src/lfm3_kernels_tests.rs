use super::*;
use half::f16;
use std::sync::atomic::Ordering;

fn half(value: f32) -> f32 {
    f16::from_f32(value).to_f32()
}
fn bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|x| f16::from_f32(*x).to_bits()).collect()
}
fn close(actual: &[f32], expected: &[f32]) {
    assert!(!actual.is_empty());
    assert_eq!(actual.len(), expected.len());
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && e.is_finite() && (a - e).abs() <= 0.002 + 0.002 * e.abs(),
            "value {i}: {a} != {e}"
        );
    }
}

#[test]
#[ignore = "physical Apple GPU; LFM3 fused convolution scalar/state oracle"]
fn short_conv_matches_scalar_and_mlx_across_eight_steps() -> Result<()> {
    for dim in [1, 3, 31, 32, 33, 2048] {
        let weights: Vec<f32> = (0..dim * 7)
            .map(|i| half((i % 7 - 3) as f32 / 8.))
            .collect();
        let weight = Array::from_f16_bits(&bits(&weights), &[dim, 7, 1])?;
        let mut host_state = vec![0.; dim as usize * 6];
        let mut state = Array::zeros_dtype(&[1, 6, dim], Dtype::Float16)?;
        for step in 0..8 {
            let data: Vec<f32> = (0..dim * 3)
                .map(|i| {
                    half(if i % 11 == 0 {
                        0.001
                    } else {
                        (((i * 17 + step * 13) % 41) - 20) as f32
                    })
                })
                .collect();
            let data: Vec<f32> = data.into_iter().map(|x| half(x / 16.)).collect();
            let bcx = Array::from_f16_bits(&bits(&data), &[1, 1, 3 * dim])?;
            let mut expected = Vec::new();
            let mut next = vec![0.; dim as usize * 6];
            for c in 0..dim as usize {
                let bx = half(data[c] * data[2 * dim as usize + c]);
                let history: f32 = (0..6)
                    .map(|tap| host_state[tap * dim as usize + c] * weights[7 * c + tap])
                    .sum();
                let conv = half(history + bx * weights[7 * c + 6]);
                expected.push(half(data[dim as usize + c] * conv));
                for tap in 0..5 {
                    next[tap * dim as usize + c] = host_state[(tap + 1) * dim as usize + c];
                }
                next[5 * dim as usize + c] = bx;
            }
            let bx = bcx
                .slice(2, 0, dim)?
                .mul(&bcx.slice(2, 2 * dim, 3 * dim)?)?;
            let joined = Array::concatenate(&[&state, &bx], 1)?;
            let reference = bcx
                .slice(2, dim, 2 * dim)?
                .mul(&joined.conv1d(&weight, dim)?)?;
            let (actual, next_state) = short_conv_step(&bcx, &state, &weight)?;
            assert_eq!(actual.shape(), [1, 1, dim]);
            close(&actual.to_f32()?, &expected);
            close(&actual.to_f32()?, &reference.to_f32()?);
            assert_eq!(
                next_state.to_f16_bits()?,
                bits(&next),
                "state dim={dim}, step={step}"
            );
            assert_eq!(
                next_state.to_f16_bits()?,
                joined.slice(1, 1, 7)?.to_f16_bits()?
            );
            state = next_state;
            host_state = next;
        }
    }
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; LFM3 sigmoid/layout scalar oracle"]
fn attention_gate_matches_scalar_and_noncontiguous_mlx() -> Result<()> {
    for time in [1, 2, 7, 257] {
        for heads in [1, 3, 16] {
            for dim in [1, 7, 128] {
                let count = (time * heads * dim) as usize;
                let data: Vec<f32> = (0..count)
                    .map(|i| half(((i * 17 % 127) as f32 - 63.) / 32.))
                    .collect();
                let gate_values: Vec<f32> = (0..time * heads)
                    .map(|i| [-80., -15., -1., 0., 1., 15., 80.][i as usize % 7])
                    .collect();
                let gate = Array::from_f16_bits(&bits(&gate_values), &[1, time, heads])?;
                for noncontiguous in [false, true] {
                    let input = if noncontiguous {
                        Array::from_f16_bits(&bits(&data), &[1, time, heads, dim])?
                            .transpose(&[0, 2, 1, 3])?
                    } else {
                        Array::from_f16_bits(&bits(&data), &[1, heads, time, dim])?
                    };
                    let logical = input.to_f32()?;
                    let mut expected = Vec::with_capacity(count);
                    for t in 0..time {
                        for h in 0..heads {
                            for d in 0..dim {
                                let g = gate_values[(t * heads + h) as usize];
                                let scale = half(1. / (1. + (-g).exp()));
                                expected.push(half(
                                    logical[((h * time + t) * dim + d) as usize] * scale,
                                ));
                            }
                        }
                    }
                    let actual = attention_gate(&input, &gate)?;
                    assert_eq!(actual.shape(), [1, time, heads * dim]);
                    close(&actual.to_f32()?, &expected);
                    let reference = input
                        .transpose(&[0, 2, 1, 3])?
                        .mul(&gate.sigmoid()?.reshape(&[1, time, heads, 1])?)?;
                    close(&actual.to_f32()?, &reference.to_f32()?);
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; malformed LFM3 fused-kernel metadata"]
fn malformed_layouts_and_dtypes_do_not_dispatch() -> Result<()> {
    let half_zero = |shape: &[i32]| Array::zeros_dtype(shape, Dtype::Float16);
    let input = half_zero(&[1, 1, 9])?;
    let state = half_zero(&[1, 6, 3])?;
    let weight = half_zero(&[3, 7, 1])?;
    let before = CONV_DISPATCHES.load(Ordering::Relaxed);
    for shape in [&[2, 1, 9][..], &[1, 2, 9][..], &[1, 1, 10][..], &[9][..]] {
        assert!(short_conv_step(&half_zero(shape)?, &state, &weight).is_err());
    }
    assert!(short_conv_step(&input, &half_zero(&[1, 1, 3])?, &weight).is_err());
    assert!(short_conv_step(&input, &state, &half_zero(&[3, 1, 7])?).is_err());
    assert!(short_conv_step(&input.astype(Dtype::Float32)?, &state, &weight).is_err());
    assert_eq!(before, CONV_DISPATCHES.load(Ordering::Relaxed));
    let attention = half_zero(&[1, 3, 7, 4])?;
    let gate = half_zero(&[1, 7, 3])?;
    let before = GATE_DISPATCHES.load(Ordering::Relaxed);
    assert!(attention_gate(&attention, &half_zero(&[1, 3, 7])?).is_err());
    assert!(attention_gate(&half_zero(&[3, 7, 4])?, &gate).is_err());
    assert!(attention_gate(&attention, &gate.astype(Dtype::Float32)?).is_err());
    assert_eq!(before, GATE_DISPATCHES.load(Ordering::Relaxed));
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and MLXL3_LFM3_REFERENCE; real decode dispatch/reset"]
fn real_text_and_vl_decode_use_both_kernels_and_reset_exactly() -> Result<()> {
    use crate::{checkpoint, lfm3::Lfm3, vision_input};
    let reference = std::env::var("MLXL3_LFM3_REFERENCE")?;
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(reference)?)?;
    let cases = data["cases"].as_array().context("missing model cases")?;
    assert!(cases.len() >= 4);
    for index in [0, 3] {
        let case = &cases[index];
        let cp = checkpoint::inspect(std::path::Path::new(
            case["model"].as_str().context("model path")?,
        ))?;
        let mut model = Lfm3::from_checkpoint(&cp)?;
        let tokens: Vec<u32> = serde_json::from_value(case["tokens"].clone())?;
        assert!(!tokens.is_empty());
        let images = case["image"]
            .as_str()
            .map(|p| {
                vision_input::decode_image(&vision_input::file_data_url(std::path::Path::new(p))?)
            })
            .transpose()?
            .into_iter()
            .collect::<Vec<_>>();
        assert_eq!(model.supports_images(), index == 3);
        let mut replay = None;
        for _ in 0..2 {
            model.reset();
            assert!(model.state_arrays().is_empty());
            if !images.is_empty() {
                model.prepare_images(&tokens, &images)?;
            }
            let mut logits = None;
            for chunk in tokens.chunks(256) {
                logits = Some(model.forward(chunk)?);
            }
            let token = logits.context("no prefill")?.chat_greedy_ids()?[0];
            let c = CONV_DISPATCHES.load(Ordering::Relaxed);
            let g = GATE_DISPATCHES.load(Ordering::Relaxed);
            let output = model.forward(&[token])?;
            assert!(
                CONV_DISPATCHES.load(Ordering::Relaxed) > c
                    && GATE_DISPATCHES.load(Ordering::Relaxed) > g
            );
            let values = output.to_f32()?;
            assert_eq!(values.len(), 128000);
            assert!(values.iter().all(|x| x.is_finite()));
            let states = model
                .state_arrays()
                .into_iter()
                .map(|(name, x)| Ok((name, x.to_f16_bits()?)))
                .collect::<Result<Vec<_>>>()?;
            let result = (output.to_f16_bits()?, states);
            if let Some(previous) = &replay {
                assert_eq!(previous, &result, "decode/reset differs");
            } else {
                replay = Some(result);
            }
        }
    }
    Ok(())
}
