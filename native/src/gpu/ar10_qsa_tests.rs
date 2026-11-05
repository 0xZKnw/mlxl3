use super::*;
use crate::tensor_layout::MatrixLayout;
use half::f16;
#[test]
#[ignore = "physical AR10 QSA canonical norm/RoPE and fused prepare; run alone"]
fn physical_ar10_qsa_prepare_matches_canonical() -> Result<()> {
    let gpu = Metal::new()?;
    let fused = gpu
        .qsa
        .prepare
        .as_ref()
        .context("initialize QSA_PREPARE=1")?;
    let half = |n: u32| MatrixView::output(&gpu, FloatType::F16, 1, n, Storage::StorageModeShared);
    let input = |n: usize| -> Result<MatrixView> {
        let bits = (0..n)
            .map(|i| f16::from_f32(((i % 131) as f32 - 65.) / 97.).to_bits())
            .collect::<Vec<_>>();
        gpu.f16_storage(&bits)?.matrix(MatrixLayout {
            shape: [1, n as u32],
            strides: [n as i64, 1],
            offset: 0,
        })
    };
    let q = input(12288)?;
    let k = input(512)?;
    let gain = gpu
        .f32_storage(
            &(0..256)
                .map(|i| ((i % 17) as f32 - 8.) / 101.)
                .collect::<Vec<_>>(),
        )?
        .matrix(MatrixLayout {
            shape: [1, 256],
            strides: [256, 1],
            offset: 0,
        })?;
    let freq = gpu
        .f32_storage(
            &(0..32)
                .map(|i| 1.0 / 10000f32.powf(i as f32 / 32.))
                .collect::<Vec<_>>(),
        )?
        .matrix(MatrixLayout {
            shape: [1, 32],
            strides: [32, 1],
            offset: 0,
        })?;
    for position in [0u32, 1, 87, 1023] {
        let qn = half(6144)?;
        let kn = half(512)?;
        let qr = half(6144)?;
        let kr = half(512)?;
        let fq = half(6144)?;
        let fk = half(512)?;
        gpu.command(|cmd| {
            for (x, n, r, heads, stride) in [(&q, &qn, &qr, 24u32, 512u32), (&k, &kn, &kr, 2, 256)]
            {
                Metal::encode(
                    cmd,
                    &gpu.qsa.norm,
                    &[&x.buffer, &gain.buffer, &n.buffer],
                    [stride, 1e-6f32.to_bits(), 256],
                    MTLSize::new(heads as u64, 1, 1),
                    32,
                );
                Metal::encode(
                    cmd,
                    &gpu.qsa.rope,
                    &[&n.buffer, &freq.buffer, &r.buffer],
                    [position, heads],
                    MTLSize::new(heads as u64 * 7, 1, 1),
                    32,
                );
            }
            Metal::encode(
                cmd,
                fused,
                &[
                    &q.buffer,
                    &k.buffer,
                    &gain.buffer,
                    &gain.buffer,
                    &freq.buffer,
                    &fq.buffer,
                    &fk.buffer,
                ],
                [position, 1e-6f32.to_bits()],
                MTLSize::new(26, 1, 1),
                32,
            );
            Ok(())
        })?;
        for (a, b, n) in [(&qr, &fq, 6144), (&kr, &fk, 512)] {
            let a = a.read_output()?;
            ensure!(
                a.len() == n
                    && a.iter()
                        .all(|&x| u16::try_from(x).is_ok_and(|x| f16::from_bits(x).is_finite()))
            );
            ensure!(b.read_output()? == a, "QSA prepare at position {position}");
        }
    }
    eprintln!("AR10_QSA_PREPARE {{\"parity\":true,\"cases\":4,\"finite_half_words\":26624}}");
    Ok(())
}

#[test]
#[ignore = "physical AR10 one-pass attention half store then gate vs canonical two kernels; run alone"]
fn physical_ar10_qsa_gated_store_matches_canonical() -> Result<()> {
    let gpu = Metal::new()?;
    let fused = gpu
        .qsa
        .gated_attention
        .as_ref()
        .context("initialize QSA_GATE=1")?;
    let input = |n: usize, seed: usize| -> Result<MatrixView> {
        let bits = (0..n)
            .map(|i| f16::from_f32((((i * seed) % 131) as f32 - 65.) / 97.).to_bits())
            .collect::<Vec<_>>();
        gpu.f16_storage(&bits)?.matrix(MatrixLayout {
            shape: [1, n as u32],
            strides: [n as i64, 1],
            offset: 0,
        })
    };
    let q = input(6144, 7)?;
    let qg = input(12288, 11)?;
    for keys in [1u32, 87, 119] {
        let k = input(keys as usize * 512, 13)?;
        let v = input(keys as usize * 512, 17)?;
        let output =
            || MatrixView::output(&gpu, FloatType::F16, 1, 6144, Storage::StorageModeShared);
        let attention = output()?;
        let expected = output()?;
        let actual = output()?;
        gpu.command(|cmd| {
            Metal::encode(
                cmd,
                &gpu.qsa.attention,
                &[
                    &q.buffer,
                    &k.buffer,
                    &v.buffer,
                    &attention.buffer,
                    &q.buffer,
                ],
                [keys, 0],
                MTLSize::new(24, 1, 1),
                1024,
            );
            Metal::encode(
                cmd,
                &gpu.qsa.gate,
                &[&attention.buffer, &qg.buffer, &expected.buffer],
                [256u32],
                MTLSize::new(192, 1, 1),
                32,
            );
            Metal::encode(
                cmd,
                fused,
                &[
                    &q.buffer,
                    &k.buffer,
                    &v.buffer,
                    &actual.buffer,
                    &q.buffer,
                    &qg.buffer,
                ],
                [keys, 0],
                MTLSize::new(24, 1, 1),
                1024,
            );
            Ok(())
        })?;
        let words = expected.read_output()?;
        ensure!(
            words.len() == 6144
                && words.iter().any(|&x| x & 0x7fff != 0)
                && words
                    .iter()
                    .all(|&x| u16::try_from(x).is_ok_and(|x| f16::from_bits(x).is_finite()))
        );
        ensure!(actual.read_output()? == words, "QSA gate keys {keys}");
    }
    eprintln!("AR10_QSA_GATE {{\"parity\":true,\"cases\":3,\"finite_half_words\":18432}}");
    Ok(())
}
