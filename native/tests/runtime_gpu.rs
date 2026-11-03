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

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
#[test]
#[ignore = "physical GPU; launched by tests/physical_flash_projection.py"]
fn flash_projection_stock_oracle() -> Result<()> {
    use mlxl3_native::{
        checkpoint,
        gpu::{FlashExpert, FlashProjection, Metal},
    };
    use serde::Deserialize;
    use std::{fs, path::PathBuf};
    #[derive(Deserialize)]
    struct Case {
        op: String,
        path: PathBuf,
        prefix: String,
        x: Vec<u16>,
    }
    let dir = PathBuf::from(
        std::env::var_os("MLXL3_PROJECTION_FIXTURES").context("fixture directory required")?,
    );
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(dir.join("cases.json"))?)?;
    ensure!(!cases.is_empty());
    let mut expected = Vec::new();
    let mut last = None;
    let mut expert_lifetime_checked = false;
    let mut generic_lifetime_checked = false;
    for case in cases {
        if last.as_ref().is_none_or(|(path, _)| path != &case.path) {
            last = Some((case.path.clone(), checkpoint::inspect(&case.path)?));
        }
        let checkpoint = &last.as_ref().unwrap().1;
        let expert = case.op == "metal-checkpoint-expert";
        let load = |name: &str| Exl3Linear::from_checkpoint(checkpoint, name);
        let (stock, tail) = if expert {
            (
                load(&format!("{}.gate_proj", case.prefix))?,
                Some((
                    load(&format!("{}.up_proj", case.prefix))?,
                    load(&format!("{}.down_proj", case.prefix))?,
                )),
            )
        } else {
            (load(&case.prefix)?, None)
        };
        let cols = tail
            .as_ref()
            .map_or(stock.output_dims(), |(_, down)| down.output_dims());
        let width = stock.input_dims() as usize;
        ensure!(!case.x.is_empty() && case.x.len().is_multiple_of(width));
        let mut bits = Vec::new();
        for row in case.x.chunks_exact(width) {
            let x = Array::from_f16_bits(row, &[1, width as i32])?;
            let mut output = stock.forward(&x)?;
            if let Some((up, down)) = &tail {
                output = down.forward(&output.swiglu(&up.forward(&x)?)?)?;
            }
            ensure!(output.shape() == [1, cols]);
            bits.extend(output.to_f16_bits()?);
        }
        ensure!(
            bits.len() == case.x.len() / width * cols as usize
                && bits.iter().all(|&v| f16::from_bits(v).is_finite())
        );
        if expert && !expert_lifetime_checked {
            let (first, second) = {
                let gpu = Metal::new()?;
                let layer = FlashExpert::load(&gpu, checkpoint, &case.prefix)?;
                let first = layer.submit(&case.x)?;
                ensure!(layer.submit(&[]).is_err());
                let second = layer.submit(&case.x)?;
                drop(layer.submit(&case.x)?);
                (first, second)
            };
            ensure!(
                second.read()? == bits && first.read()? == bits,
                "early expert drop or reverse read changed output"
            );
            expert_lifetime_checked = true;
        } else if (expected.is_empty()
            || case.op == "metal-exl3-linear" && !generic_lifetime_checked)
            && !expert
        {
            let (first, second) = {
                let device = Metal::new()?;
                let load = if case.op == "metal-exl3-linear" {
                    FlashProjection::load_exl3
                } else {
                    FlashProjection::load
                };
                let layer = load(&device, checkpoint, &case.prefix)?;
                let first = layer.submit(&case.x)?;
                ensure!(layer.submit(&[]).is_err() && layer.submit(&case.x[1..]).is_err());
                let second = layer.submit(&case.x)?;
                drop(layer.submit(&case.x)?);
                (first, second)
            };
            ensure!(
                second.read()? == bits && first.read()? == bits,
                "early drop or reverse read changed output"
            );
            generic_lifetime_checked |= case.op == "metal-exl3-linear";
        }
        expected.push(bits);
    }
    serde_json::to_writer(fs::File::create_new(dir.join("stock.json"))?, &expected)?;
    Ok(())
}
