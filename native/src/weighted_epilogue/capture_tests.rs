use super::{apply, kernel_requests_for_test, reset_kernel_requests_for_test};
use crate::array::{Array, Dtype, MetalCapture, synchronize};
use anyhow::{Context, Result, ensure};
use std::path::PathBuf;

#[test]
#[ignore = "physical Apple GPU; MTL_CAPTURE_ENABLED=1 and fresh MLXL3_WEIGHTED_CAPTURE_OUTPUT; run alone"]
fn captures_real_legacy_and_fused_epilogue_commands() -> Result<()> {
    ensure!(
        std::env::var("MTL_CAPTURE_ENABLED").is_ok_and(|value| value == "1"),
        "capture requires MTL_CAPTURE_ENABLED=1 before process startup"
    );
    let directory = PathBuf::from(
        std::env::var_os("MLXL3_WEIGHTED_CAPTURE_OUTPUT")
            .context("MLXL3_WEIGHTED_CAPTURE_OUTPUT is required")?,
    );
    ensure!(directory.is_dir(), "capture output directory must exist");
    let legacy_path = directory.join("legacy.gputrace");
    let fused_path = directory.join("fused.gputrace");
    ensure!(
        !legacy_path.exists() && !fused_path.exists(),
        "refusing to overwrite either capture"
    );

    const WIDTH: usize = 2560;
    const ROUTES: usize = 10;
    let hot_bits = (0..10 * WIDTH)
        .map(|i| 0x2000 + (i % 0x1000) as u16)
        .collect::<Vec<_>>();
    let cold_bits = (0..5 * WIDTH)
        .map(|i| 0x3400 + (i % 0x0800) as u16)
        .collect::<Vec<_>>();
    let hot = Array::from_f16_bits(&hot_bits, &[10, WIDTH as i32])?;
    let cold = Array::from_f16_bits(&cold_bits, &[5, WIDTH as i32])?;
    let order = [0, 10, 2, 11, 4, 12, 6, 13, 8, 14];
    let weights = Array::from_f32(
        &[
            0.11, -0.23, 0.37, -0.41, 0.53, -0.67, 0.71, -0.83, 0.89, -0.97,
        ],
        &[ROUTES as i32, 1],
    )?;
    for input in [&hot, &cold, &weights] {
        input.eval()?;
    }
    let legacy = || -> Result<Array> {
        let order_array = Array::from_u32(&order, &[ROUTES as i32])?;
        Array::concatenate(&[&hot, &cold], 0)?
            .take(&order_array, 0)?
            .astype(Dtype::Float32)?
            .mul(&weights)
    };
    legacy()?.eval()?;
    apply(&hot, &cold, &order, &weights, Dtype::Float32)?.eval()?;
    synchronize()?;

    let capture = MetalCapture::start(&legacy_path)?;
    // Metal captures only new command buffers; rotate MLX's preexisting buffer first.
    synchronize()?;
    let reference = legacy()?;
    reference.eval()?;
    capture.finish()?;

    reset_kernel_requests_for_test();
    let capture = MetalCapture::start(&fused_path)?;
    synchronize()?;
    let fused = apply(&hot, &cold, &order, &weights, Dtype::Float32)?;
    fused.eval()?;
    capture.finish()?;
    ensure!(
        kernel_requests_for_test() == 1,
        "capture did not exercise the fused kernel request path"
    );

    ensure!(
        reference.shape() == [ROUTES as i32, WIDTH as i32]
            && fused.shape() == reference.shape()
            && reference.dtype() == Dtype::Float32
            && fused.dtype() == Dtype::Float32,
        "capture changed output metadata"
    );
    let reference_bytes = reference.to_bytes()?;
    let fused_bytes = fused.to_bytes()?;
    ensure!(
        reference_bytes.len() == ROUTES * WIDTH * 4 && fused_bytes == reference_bytes,
        "captured output differs from the complete legacy oracle"
    );
    for output in [&reference, &fused] {
        let values = output.to_f32()?;
        ensure!(
            values.len() == ROUTES * WIDTH && values.iter().all(|v| v.is_finite()),
            "empty or nonfinite captured output"
        );
    }
    for path in [&legacy_path, &fused_path] {
        ensure!(
            path.is_dir() && std::fs::read_dir(path)?.next().is_some(),
            "missing or empty Metal capture"
        );
    }
    println!(
        "WEIGHTED_CAPTURE captures=2 values={} bytes={} status=captured_exact_outputs_dispatch_inspection_pending",
        ROUTES * WIDTH,
        reference_bytes.len()
    );
    Ok(())
}
