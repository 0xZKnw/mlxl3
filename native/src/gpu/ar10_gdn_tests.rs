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
