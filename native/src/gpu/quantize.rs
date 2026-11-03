//! Direct Metal execution of the conversion trellis, without model weights.
use super::{Metal, buffer, upload};
use crate::{codec::Codebook, quantize::tile_search_plan};
use anyhow::{Context, Result, ensure};
use metal::{
    CommandQueue, CompileOptions, Device, MTLCommandBufferStatus, MTLResourceOptions as Storage,
    MTLSize,
};
use std::sync::atomic::{AtomicBool, Ordering};

pub fn quantize_tiles(
    values: &[f32],
    k: u32,
    codebook: Codebook,
    scratch_bytes: u64,
    cancelled: &AtomicBool,
) -> Result<(Vec<f32>, Vec<u16>)> {
    let (launch_tiles, scratch_per_tile) = tile_search_plan(values.len(), k, scratch_bytes)
        .context("invalid tile count, K or scratch budget")?;
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "tile values must be finite"
    );
    ensure!(!cancelled.load(Ordering::Relaxed), "conversion cancelled");
    let device = Device::system_default().context("no Metal device")?;
    let queue: CommandQueue = device.new_command_queue();
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let source = format!(
        "#define MLXL3_QUANT_K {k}\n#define MLXL3_QUANT_CB {}\n{}",
        codebook as u32,
        include_str!("../../shaders/quantize_tiles.metal")
    );
    let library = device
        .new_library_with_source(&source, &options)
        .map_err(anyhow::Error::msg)?;
    let function = library
        .get_function("mlxl3_quantize_tiles", None)
        .map_err(anyhow::Error::msg)?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        pipeline.max_total_threads_per_threadgroup() >= 256,
        "unsupported quantizer threadgroup"
    );
    let mut decoded = Vec::new();
    let mut states = Vec::new();
    decoded.try_reserve_exact(values.len())?;
    states.try_reserve_exact(values.len())?;
    for chunk in values.chunks(launch_tiles * 256) {
        ensure!(!cancelled.load(Ordering::Relaxed), "conversion cancelled");
        objc::rc::autoreleasepool(|| -> Result<()> {
            let n = chunk.len() as u64;
            let input = upload(&device, chunk)?;
            let output = buffer(&device, n * 4, Storage::StorageModeShared)?;
            let indices = buffer(&device, n * 2, Storage::StorageModeShared)?;
            let edges = (1u64 << 16) >> k;
            let back = buffer(&device, n * edges * 2, Storage::StorageModePrivate)?;
            let costs = buffer(
                &device,
                n / 256 * 2 * edges * 2,
                Storage::StorageModePrivate,
            )?;
            ensure!(back.length() + costs.length() == n / 256 * scratch_per_tile);
            let cmd = queue.new_command_buffer();
            Metal::encode(
                cmd,
                &pipeline,
                &[&input, &output, &indices, &back, &costs],
                [] as [u32; 0],
                MTLSize::new(n / 256, 1, 1),
                256,
            );
            cmd.commit();
            cmd.wait_until_completed();
            ensure!(
                cmd.status() == MTLCommandBufferStatus::Completed,
                "Metal trellis failed ({:?})",
                cmd.status()
            );
            ensure!(!cancelled.load(Ordering::Relaxed), "conversion cancelled");
            // The completed buffer is shared, aligned and allocated for exactly this FP32 row count.
            decoded.extend_from_slice(unsafe {
                std::slice::from_raw_parts(output.contents().cast::<f32>(), chunk.len())
            });
            states.extend(Metal::read(&indices, chunk.len()));
            Ok(())
        })?;
    }
    ensure!(
        decoded.len() == values.len()
            && states.len() == values.len()
            && decoded.iter().all(|v| v.is_finite())
    );
    Ok((decoded, states))
}

#[cfg(test)]
#[test]
fn conversion_validation_and_cancellation_reject_before_device_allocation() {
    let cancelled = AtomicBool::new(true);
    assert!(
        quantize_tiles(&[0.; 256], 4, Codebook::Default, 1 << 28, &cancelled)
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );
    let active = AtomicBool::new(false);
    for values in [&[][..], &[0.; 255], &[f32::NAN; 256], &[f32::INFINITY; 256]] {
        assert!(quantize_tiles(values, 4, Codebook::Default, 1 << 28, &active).is_err());
    }
}
