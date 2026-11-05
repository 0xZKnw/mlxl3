use super::*;
use crate::tensor_layout::MatrixLayout;
use half::f16;

#[test]
#[ignore = "physical AR10 joint a/b exact actual kernels and real checkpoint; run alone"]
fn physical_ar10_joint_ab_matches_canonical_dense() -> Result<()> {
    let gpu = Metal::new()?;
    let c = crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gdn = FlashGdn::load(&gpu, &c, "model.language_model.layers.0.linear_attn", 1e-6)?;
    let joint = gpu
        .gdn
        .joint_ab
        .as_ref()
        .context("initialize GDN_JOINT_AB=1")?;
    let mut distinct = false;
    for phase in 0..2 {
        let bits = (0..2560)
            .map(|i| {
                f16::from_f32(if phase == 0 {
                    0.0
                } else {
                    ((i % 113) as f32 - 56.) / 79.
                })
                .to_bits()
            })
            .collect::<Vec<_>>();
        let x = gpu.f16_storage(&bits)?.matrix(MatrixLayout {
            shape: [1, 2560],
            strides: [2560, 1],
            offset: 0,
        })?;
        let output = || MatrixView::output(&gpu, FloatType::F16, 1, 48, Storage::StorageModeShared);
        let a = output()?;
        let b = output()?;
        let ja = output()?;
        let jb = output()?;
        gpu.command(|cmd| {
            for (weight, out) in [(&gdn.a, &a), (&gdn.b, &b)] {
                Metal::encode(
                    cmd,
                    &gpu.gdn.dense,
                    &[&x.buffer, weight, &out.buffer],
                    [48u32],
                    MTLSize::new(12, 1, 1),
                    256,
                );
            }
            Metal::encode(
                cmd,
                joint,
                &[&x.buffer, &gdn.a, &gdn.b, &ja.buffer, &jb.buffer],
                [48u32],
                MTLSize::new(24, 1, 1),
                256,
            );
            Ok(())
        })?;
        let a = a.read_output()?;
        let b = b.read_output()?;
        ensure!(
            a.len() == 48
                && b.len() == 48
                && a.iter()
                    .chain(&b)
                    .all(|&x| u16::try_from(x).is_ok_and(|v| f16::from_bits(v).is_finite()))
        );
        distinct |= a != b;
        ensure!(ja.read_output()? == a && jb.read_output()? == b);
    }
    ensure!(distinct, "fixture must detect swapped a/b");
    eprintln!("AR10_JOINT_AB {{\"parity\":true,\"cases\":2,\"finite_half_words\":192}}");
    Ok(())
}

#[test]
#[ignore = "physical AR10 whole GDN vs split canonical, every history/state/gated/output word; run alone"]
fn physical_ar10_whole_gdn_matches_canonical_frames() -> Result<()> {
    let gpu = Metal::new()?;
    ensure!(gpu.gdn_whole.get());
    gpu.gdn_joint_ab.set(false);
    gpu.gdn_half_elision.set(false);
    gpu.gdn_norm_gate.set(false);
    let c = crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let mut gdn = FlashGdn::load(&gpu, &c, "model.language_model.layers.0.linear_attn", 1e-6)?;
    let read = |p: &PendingGdn| -> Result<[Vec<u8>; 4]> {
        Ok([
            p.next.history.export_bytes(&gpu)?,
            p.next.recurrent.export_bytes(&gpu)?,
            p.test_gated.export_bytes(&gpu)?,
            p.values.export_bytes(&gpu)?,
        ])
    };
    for phase in 0..2 {
        let row = (0..2560)
            .map(|i| (((i + phase * 13) % 113) as f32 - 56.) / 79.)
            .collect::<Vec<_>>();
        let x = gpu.f32_storage(&row)?.matrix(MatrixLayout {
            shape: [1, 2560],
            strides: [2560, 1],
            offset: 0,
        })?;
        let before = gdn.snapshot();
        gpu.gdn_whole.set(false);
        let expected = gpu.command(|cmd| gdn.encode_for_consumer(cmd, &x, false, true))?;
        let bytes = read(&expected)?;
        ensure!(
            bytes.iter().map(Vec::len).collect::<Vec<_>>()
                == [30720 * 2, 786432 * 4, 6144 * 2, 2560 * 4]
        );
        for (i, raw) in bytes.iter().enumerate() {
            ensure!(if i == 0 || i == 2 {
                raw.as_chunks::<2>()
                    .0
                    .iter()
                    .all(|b| f16::from_bits(u16::from_le_bytes(*b)).is_finite())
            } else {
                raw.as_chunks::<4>()
                    .0
                    .iter()
                    .all(|b| f32::from_le_bytes(*b).is_finite())
            });
        }
        gdn.finish(expected)?;
        let held = gdn.snapshot();
        gdn.restore(&before)?;
        gpu.gdn_whole.set(true);
        let uses = gpu.gdn_whole_uses.get();
        let actual = gpu.command(|cmd| gdn.encode_for_consumer(cmd, &x, false, true))?;
        ensure!(
            gpu.gdn_whole_uses.get() - uses == 1,
            "whole GDN fallback cannot qualify"
        );
        ensure!(read(&actual)? == bytes, "whole GDN phase {phase}");
        gdn.finish(actual)?;
        let frame = held.frame.as_ref().context("missing retained frame")?;
        ensure!(
            frame.history.export_bytes(&gpu)? == bytes[0]
                && frame.recurrent.export_bytes(&gpu)? == bytes[1]
        );
    }
    eprintln!(
        "AR10_GDN_WHOLE {{\"parity\":true,\"cases\":2,\"state_words\":1572864,\"history_half_words\":61440,\"gated_half_words\":12288,\"output_float_words\":5120,\"actual_whole_uses\":2,\"retained_frames\":true}}"
    );
    Ok(())
}
