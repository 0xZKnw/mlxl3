use super::*;
use crate::tensor_layout::MatrixLayout;
use half::f16;

#[test]
#[ignore = "physical canonical QSA vs chain and load-pipeline variants with all tails; exclusive GPU"]
fn physical_ar20_qsa_chains_and_prefetch_preserve_canonical_words() -> Result<()> {
    let gpu = Metal::new()?;
    ensure!(
        gpu.qsa_chains.get() && gpu.qsa_prefetch.get(),
        "enable QSA_CHAINS/PREFETCH=1 to construct every PSO"
    );
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
    let output = || MatrixView::output(&gpu, FloatType::F16, 1, 6144, Storage::StorageModeShared);
    let mut words = 0usize;
    let mut cases = 0usize;
    for keys in [1u32, 31, 32, 33, 87, 119, 127, 128, 130, 1023] {
        let k = input(keys as usize * 512, 13)?;
        let v = input(keys as usize * 512, 17)?;
        let attention = output()?;
        let gated = output()?;
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
                &[&attention.buffer, &qg.buffer, &gated.buffer],
                [256u32],
                MTLSize::new(192, 1, 1),
                32,
            );
            Ok(())
        })?;
        for fuse_gate in [false, true] {
            let expected = if fuse_gate {
                gated.read_output()?
            } else {
                attention.read_output()?
            };
            ensure!(
                expected.len() == 6144
                    && expected.iter().any(|&v| v & 0x7fff != 0)
                    && expected
                        .iter()
                        .all(|&v| u16::try_from(v).is_ok_and(|v| f16::from_bits(v).is_finite()))
            );
            for (chains, prefetch) in [(true, false), (false, true), (true, true)] {
                gpu.qsa_chains.set(chains);
                gpu.qsa_prefetch.set(prefetch);
                let actual = gpu.f16_storage(&vec![0x7e00; 6144])?.matrix(MatrixLayout {
                    shape: [1, 6144],
                    strides: [6144, 1],
                    offset: 0,
                })?;
                let before = [gpu.qsa_chain_uses.get(), gpu.qsa_prefetch_uses.get()];
                gpu.command(|cmd| {
                    ensure!(
                        gpu.encode_ar20_qsa(
                            cmd,
                            [&q.buffer, &k.buffer, &v.buffer, &actual.buffer, &qg.buffer],
                            keys,
                            [false, true, false],
                            fuse_gate
                        )?,
                        "candidate cannot qualify on fallback"
                    );
                    Ok(())
                })?;
                ensure!(
                    [
                        gpu.qsa_chain_uses.get() - before[0],
                        gpu.qsa_prefetch_uses.get() - before[1]
                    ] == [usize::from(chains), usize::from(prefetch)]
                );
                ensure!(
                    actual.read_output()? == expected,
                    "QSA keys{keys}/chains{chains}/prefetch{prefetch}/gate{fuse_gate}"
                );
                words += expected.len();
                cases += 1;
            }
        }
    }
    let dummy = output()?;
    for (keys, domain) in [
        (0, [false, true, false]),
        (1024, [false, true, false]),
        (87, [true, true, false]),
        (87, [false, false, false]),
        (87, [false, true, true]),
    ] {
        let before = [gpu.qsa_chain_uses.get(), gpu.qsa_prefetch_uses.get()];
        let cmd = gpu.queue.new_command_buffer().to_owned();
        ensure!(!gpu.encode_ar20_qsa(
            &cmd,
            [
                &q.buffer,
                &dummy.buffer,
                &dummy.buffer,
                &dummy.buffer,
                &qg.buffer
            ],
            keys,
            domain,
            true
        )?);
        ensure!([gpu.qsa_chain_uses.get(), gpu.qsa_prefetch_uses.get()] == before);
        ensure!(cmd.status() == metal::MTLCommandBufferStatus::NotEnqueued);
    }
    ensure!(cases == 60 && words == 368640);
    eprintln!(
        "AR20_QSA_CHAINS_PREFETCH {{\"parity\":true,\"cases\":60,\"finite_half_words\":368640,\"contexts\":10,\"independent_canonical_psos\":true,\"guard_fallback_cases\":5,\"poisoned_destinations\":true}}"
    );
    Ok(())
}
