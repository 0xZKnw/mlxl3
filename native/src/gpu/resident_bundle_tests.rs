use super::*;
use crate::tensor_layout::MatrixLayout;

#[test]
#[ignore = "physical fixed resident GDN qkv/z bundle vs separate canonical projections; exclusive GPU"]
fn physical_ar20_resident_bundle_preserves_distinct_transforms_and_outputs() -> Result<()> {
    let gpu = Metal::new()?;
    ensure!(
        gpu.gdn_resident_bundle.get() && gpu.gdn_resident_qmv.is_some(),
        "enable GDN_RESIDENT_BUNDLE=1"
    );
    let checkpoint =
        crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let mut qkv = FlashProjection::load(
        &gpu,
        &checkpoint,
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
    )?;
    let mut z = FlashProjection::load(
        &gpu,
        &checkpoint,
        "model.language_model.layers.0.linear_attn.in_proj_z",
    )?;
    ensure!(qkv.nt == 2 && z.nt == 2 && qkv.k == 4 && z.k == 4);
    let before = gpu.gdn_resident_bundle_uses.get();
    let mut words = 0usize;
    for phase in 0..4 {
        if phase == 2 {
            qkv.suh = upload(
                &gpu.device,
                &(0..2560)
                    .map(|i| f16::from_f32([1., 0.5, -1., -0.25][i % 4]).to_bits())
                    .collect::<Vec<_>>(),
            )?;
            z.suh = upload(
                &gpu.device,
                &(0..2560)
                    .map(|i| f16::from_f32([-0.5, -1., 0.25, 1.][i % 4]).to_bits())
                    .collect::<Vec<_>>(),
            )?;
        }
        let mut input = vec![f16::from_f32(7.).to_bits(); 2564];
        for i in 0..2560 {
            input[2561 - i] =
                f16::from_f32((((i + phase * 13) % 113) as f32 - 56.) / 79.).to_bits();
        }
        let storage = if phase % 2 == 0 {
            gpu.f16_storage(&input)?
        } else {
            gpu.f32_storage(
                &input
                    .iter()
                    .map(|&v| f16::from_bits(v).to_f32())
                    .collect::<Vec<_>>(),
            )?
        };
        let x = storage.matrix(MatrixLayout {
            shape: [1, 2560],
            strides: [2564, -1],
            offset: 2561,
        })?;
        let expected = gpu.command(|cmd| {
            Ok([
                qkv.encode_row(cmd, &x, Storage::StorageModeShared)?,
                z.encode_row(cmd, &x, Storage::StorageModeShared)?,
            ])
        })?;
        let expected = [expected[0].read_output()?, expected[1].read_output()?];
        ensure!(expected[0].len() == 10240 && expected[1].len() == 6144);
        ensure!(
            expected
                .iter()
                .flatten()
                .all(|&v| u16::try_from(v).is_ok_and(|v| f16::from_bits(v).is_finite()))
                && expected.iter().flatten().any(|&v| v & 0x7fff != 0)
        );
        let actual = gpu
            .command(|cmd| qkv.encode_resident_pair(&z, cmd, &x, Storage::StorageModeShared))?
            .context("resident bundle cannot qualify on fallback")?;
        for i in 0..2 {
            ensure!(actual[i].layout.offset == 0 && actual[i].read_output()? == expected[i]);
            words += expected[i].len();
        }
    }
    ensure!(gpu.gdn_resident_bundle_uses.get() - before == 4 && words == 65536);
    let bits = vec![0u16; 2560];
    let x = gpu.f16_storage(&bits)?.matrix(MatrixLayout {
        shape: [1, 2560],
        strides: [2560, 1],
        offset: 0,
    })?;
    let cmd = gpu.queue.new_command_buffer().to_owned();
    ensure!(
        qkv.encode_resident_pair(&qkv, &cmd, &x, Storage::StorageModeShared)?
            .is_none()
    );
    qkv.nt = 1;
    ensure!(
        qkv.encode_resident_pair(&z, &cmd, &x, Storage::StorageModeShared)?
            .is_none()
    );
    qkv.nt = 2;
    z.bias = Some(upload(&gpu.device, &vec![0u16; 6144])?);
    ensure!(
        qkv.encode_resident_pair(&z, &cmd, &x, Storage::StorageModeShared)?
            .is_none()
    );
    ensure!(
        cmd.status() == MTLCommandBufferStatus::NotEnqueued
            && gpu.gdn_resident_bundle_uses.get() - before == 4
    );
    eprintln!(
        "AR20_GDN_RESIDENT_BUNDLE {{\"parity\":true,\"cases\":4,\"finite_half_words\":65536,\"actual_bundle_uses\":4,\"distinct_synthetic_transforms\":true,\"negative_strides\":true,\"fallback_cases\":3}}"
    );
    Ok(())
}
