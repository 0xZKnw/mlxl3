#![cfg(feature = "mlx")]
use anyhow::{Context, Result, ensure};
use half::f16;
use mlxl3_native::{
    array::Array,
    codec::Codebook,
    linear::{Exl3Group, Exl3Linear},
};

#[test]
#[ignore = "physical GPU and fixture from scripts/check-mtp-reference.py"]
fn native_mtp_head_matches_independent_mlx_reference() -> Result<()> {
    use mlxl3_native::{mtp::Head, qwen35::Qwen35Moe};
    let fixture = std::env::var("MLXL3_MTP_REFERENCE")?;
    let data: serde_json::Value = serde_json::from_slice(&std::fs::read(fixture)?)?;
    let target = std::env::var("MLXL3_MTP_TEST_MODEL")
        .unwrap_or_else(|_| "models/Qwen3.6-35B-A3B-EXL3-2.49bpw".into());
    let head_path = std::env::var("MLXL3_MTP_TEST_HEAD")
        .unwrap_or_else(|_| "models/Qwen3.6-35B-A3B-MTP-4bit".into());
    let model = Qwen35Moe::load(std::path::Path::new(&target))?;
    let mut head = Head::load(std::path::Path::new(&head_path), &model)?;
    let steps = data["steps"]
        .as_array()
        .context("missing MTP reference steps")?;
    ensure!(!steps.is_empty(), "empty MTP reference");
    for step in steps {
        let tokens: Vec<u32> = serde_json::from_value(step["tokens"].clone())?;
        let hidden: Vec<u16> = serde_json::from_value(step["hidden"].clone())?;
        let recursive = step.get("residual").is_some();
        let expected: Vec<u16> = serde_json::from_value(
            step[if recursive { "residual" } else { "normalized" }].clone(),
        )?;
        ensure!(
            !tokens.is_empty()
                && expected.len() == tokens.len() * model.mtp_layout().hidden_size as usize,
            "invalid MTP reference dimensions"
        );
        ensure!(
            expected
                .iter()
                .all(|&bits| f16::from_bits(bits).is_finite()),
            "nonfinite MTP reference"
        );
        let hidden = Array::from_f16_bits(
            &hidden,
            &[1, tokens.len() as i32, model.mtp_layout().hidden_size],
        )?;
        let actual = if recursive {
            head.draft_residual(&model, &hidden, &tokens)?
        } else {
            head.hidden(&model, &hidden, &tokens)?
        }
        .to_f16_bits()?;
        ensure!(
            actual.len() == expected.len()
                && actual.iter().all(|&bits| f16::from_bits(bits).is_finite()),
            "invalid native MTP output"
        );
        let mismatches = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
        let maximum_error = actual
            .iter()
            .zip(&expected)
            .map(|(a, b)| (f16::from_bits(*a).to_f32() - f16::from_bits(*b).to_f32()).abs())
            .fold(0.0f32, f32::max);
        eprintln!(
            "MTP reference time={} mismatches={} max_abs={}",
            tokens.len(),
            mismatches,
            maximum_error
        );
        assert_eq!(
            actual,
            expected,
            "MTP head differs from independent MLX at time={}",
            tokens.len()
        );
        let (keys, values) = head.cache_bytes()?;
        for (name, actual) in [("keys", keys), ("values", values)] {
            let expected: Vec<u16> = serde_json::from_value(step[name].clone())?;
            ensure!(
                !expected.is_empty() && expected.iter().all(|&b| f16::from_bits(b).is_finite()),
                "invalid independent MTP cache"
            );
            let expected: Vec<u8> = expected.into_iter().flat_map(u16::to_le_bytes).collect();
            assert_eq!(actual, expected, "independent MTP {name} differs");
        }
    }
    Ok(())
}

fn linear(k: usize, width: i32) -> Result<Exl3Linear> {
    let data = (0..8 * (width / 16) * 16 * k as i32)
        .map(|i| (i as u16).wrapping_mul(1777))
        .collect::<Vec<_>>();
    Exl3Linear::new(
        Array::from_u16(&data, &[8, width / 16, 16 * k as i32])?,
        Array::from_f16_bits(&vec![f16::ONE.to_bits(); 128], &[128])?,
        Array::from_f16_bits(&vec![f16::ONE.to_bits(); width as usize], &[width])?,
        None,
        k,
        Codebook::Default,
    )
}

#[test]
#[ignore = "physical Apple GPU; run with MLXL3_DISABLE_TENSOR_OPS=1"]
fn portable_exl3_prefill_matches_independent_single_rows() -> Result<()> {
    ensure!(
        std::env::var_os("MLXL3_DISABLE_TENSOR_OPS").is_some(),
        "this check must exercise the non-TensorOps path"
    );
    for k in 1..=8 {
        let layer = linear(k, 128)?;
        for rows in [1, 2, 3, 4, 5, 6, 7, 8, 23, 24, 25, 46, 47, 256, 513] {
            let input = (0..rows * 128)
                .map(|i| f16::from_f32(((i % 251) as f32 - 125.) / 128.).to_bits())
                .collect::<Vec<_>>();
            let x = Array::from_f16_bits(&input, &[1, rows, 128])?;
            let expected = (0..rows)
                .map(|row| layer.forward(&x.slice(1, row, row + 1)?))
                .collect::<Result<Vec<_>>>()?;
            let expected =
                Array::concatenate(&expected.iter().collect::<Vec<_>>(), 1)?.to_f16_bits()?;
            assert_eq!(
                layer.forward(&x)?.to_f16_bits()?,
                expected,
                "portable K={k}, rows={rows}"
            );
        }
        if k == 7 {
            continue;
        }
        let group = Exl3Group::new(vec![linear(k, 128)?, linear(k, 256)?])?;
        for rows in [2, 3, 4, 5, 6, 7, 8, 23, 24, 25, 46, 47, 256] {
            let x = Array::from_f16_bits(
                &vec![f16::from_f32(0.125).to_bits(); rows as usize * 128],
                &[rows, 128],
            )?;
            let actual = group.forward(&x)?;
            for (index, width) in [128, 256].into_iter().enumerate() {
                let separate = linear(k, width)?;
                assert_eq!(
                    actual[index].to_f16_bits()?,
                    separate.forward(&x)?.to_f16_bits()?,
                    "portable grouped K={k}, rows={rows}, width={width}"
                );
            }
        }
    }
    Ok(())
}
