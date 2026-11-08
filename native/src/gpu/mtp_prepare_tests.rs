use super::*;

#[test]
#[ignore = "physical M5 with local Flash checkpoint; run serially"]
fn fused_mtp_norm_preparation_matches_independent_two_kernel_composition() -> Result<()> {
    let path = std::path::PathBuf::from(std::env::var("MLXL3_FLASH_MODEL")?);
    let checkpoint = crate::checkpoint::inspect(&path)?;
    let gpu = Metal::new()?;
    ensure!(gpu.device_name() == "Apple M5");
    let options = metal::CompileOptions::new();
    options.set_fast_math_enabled(false);
    let reference_library = gpu
        .device
        .new_library_with_source(
            include_str!(
                "../../../docs/measurements/issue165-q02-20261108/mtp_hidden_norm_reference.metal"
            ),
            &options,
        )
        .map_err(|e| anyhow::anyhow!(e))?;
    let reference_function = reference_library
        .get_function("mtp_hidden_norm", None)
        .map_err(|e| anyhow::anyhow!(e))?;
    let reference_norm = gpu
        .device
        .new_compute_pipeline_state_with_function(&reference_function)
        .map_err(|e| anyhow::anyhow!(e))?;
    let projection = FlashProjection::load(&gpu, &checkpoint, "mtp.fc_hidden")?;
    let gain = crate::gpu::checkpoint_float_weight(
        &gpu,
        &checkpoint,
        "mtp.pre_fc_norm_hidden.weight",
        &[10240],
        super::super::view::FloatType::F32,
    )?;
    let undersized = MatrixView {
        buffer: buffer(&gpu.device, 4, Storage::StorageModeShared)?,
        layout: crate::tensor_layout::MatrixLayout {
            shape: [1, 10240],
            strides: [10240, 1],
            offset: 0,
        },
        dtype: super::super::view::FloatType::F32,
    };
    let unsubmitted = gpu.queue.new_command_buffer();
    ensure!(
        projection
            .prepare_mtp_hidden(unsubmitted, &undersized, &gain, 1e-6)
            .is_err()
    );
    let mut words = 0;
    for pattern in 0..4 {
        let values: Vec<f32> = (0..10240u32)
            .map(|i| {
                let signed = ((i.wrapping_mul(1664525).wrapping_add(1013904223) >> 8) as f32
                    / (1u32 << 24) as f32)
                    * 2.
                    - 1.;
                match pattern {
                    0 => {
                        if i % 2 == 0 {
                            0.
                        } else {
                            -0.
                        }
                    }
                    1 => signed,
                    2 => signed * 2f32.powi((i % 33) as i32 - 16),
                    _ => signed * 1e-12,
                }
            })
            .collect();
        let input = MatrixView {
            buffer: crate::gpu::upload(&gpu.device, &values)?,
            layout: crate::tensor_layout::MatrixLayout {
                shape: [1, 10240],
                strides: [10240, 1],
                offset: 0,
            },
            dtype: super::super::view::FloatType::F32,
        };
        for eps in [1e-6f32, 1e-5] {
            let (reference, prepared, old_output, new_output) = gpu.command(|cmd| {
                let mut normalized = MatrixView::output(
                    &gpu,
                    super::super::view::FloatType::F32,
                    1,
                    10240,
                    Storage::StorageModePrivate,
                )?;
                Metal::encode(
                    cmd,
                    &reference_norm,
                    &[&input.buffer, &gain, &normalized.buffer],
                    [eps.to_bits()],
                    MTLSize::new(1, 1, 1),
                    1024,
                );
                normalized.layout = crate::tensor_layout::MatrixLayout {
                    shape: [4, 2560],
                    strides: [2560, 1],
                    offset: 0,
                };
                let scales =
                    MatrixView::column_scales(&projection.suh, projection.offsets.suh, 4, 2560)?;
                let reference = buffer(&gpu.device, 10240 * 2, Storage::StorageModePrivate)?;
                gpu.encode_prepare_view(cmd, &normalized, &scales, &reference)?;
                let prepared = projection.prepare_mtp_hidden(cmd, &input, &gain, eps)?;
                let old_output =
                    projection.encode_rows(cmd, &normalized, Storage::StorageModePrivate)?;
                let new_output = projection.encode_mtp_hidden_normalized(
                    cmd,
                    &input,
                    &gain,
                    eps,
                    Storage::StorageModePrivate,
                )?;
                Ok((reference, prepared, old_output, new_output))
            })?;
            let export = |buffer| {
                MatrixView {
                    buffer,
                    layout: crate::tensor_layout::MatrixLayout {
                        shape: [4, 2560],
                        strides: [2560, 1],
                        offset: 0,
                    },
                    dtype: super::super::view::FloatType::F16,
                }
                .export_bytes(&gpu)
            };
            let reference = export(reference)?;
            let prepared = export(prepared)?;
            ensure!(
                reference.len() == 20480 && prepared.len() == 20480 && reference == prepared,
                "prepared words differ: pattern{pattern}, eps{eps}"
            );
            ensure!(
                prepared
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|v| half::f16::from_bits(u16::from_le_bytes(*v)).is_finite())
            );
            if pattern != 0 {
                ensure!(prepared.iter().any(|&v| v != 0));
            }
            let expected = old_output.export_bytes(&gpu)?;
            let actual = new_output.export_bytes(&gpu)?;
            ensure!(
                expected.len() == 20480 && actual == expected,
                "projected rows differ"
            );
            words += 10240;
        }
    }
    ensure!(words == 81920 && gpu.mtp_norm_prepare_uses.get() == 16);
    println!(
        "MTP norm/prepare: {words} exact prepared words plus81920 projected words,16actual fused dispatches"
    );
    Ok(())
}
