use super::*;
use crate::tensor_layout::MatrixLayout;
use half::f16;

fn words(buffers: &GateBuffers) -> Result<[Vec<u32>; 4]> {
    Ok([
        buffers.logits.read_output()?,
        buffers.probabilities.read_output()?,
        buffers.routes.indices()?,
        buffers.routes.completed_scores()?.read_output()?,
    ])
}

fn finite(values: &[u32], half: bool) -> bool {
    values.iter().all(|&v| {
        if half {
            u16::try_from(v).is_ok_and(|v| f16::from_bits(v).is_finite())
        } else {
            f32::from_bits(v).is_finite()
        }
    })
}

#[test]
#[ignore = "physical paired router vs independent original PSOs, all logits/probabilities/routes; exclusive GPU"]
fn physical_ar30_paired_router_preserves_reference_words_strides_and_routes() -> Result<()> {
    let gpu = Metal::new()?;
    ensure!(
        gpu.router_pair.enabled.get() && gpu.router_pair.pipelines.is_some(),
        "enable MLXL3_NATIVE_ROUTER_PAIR=1 on the qualified GPU"
    );
    let weight = (0..512 * 2560)
        .map(|i| {
            f16::from_f32((((i / 2560) * 13 + (i % 2560) * 7) % 127) as f32 / 1024. - 63. / 1024.)
                .to_bits()
        })
        .collect::<Vec<_>>();
    let gate = FlashGate {
        gpu: &gpu,
        weight: super::super::upload(&gpu.device, &weight)?,
    };
    let start = gpu.router_pair.uses.get();
    let mut cases = 0usize;
    let mut logits_words = 0usize;
    let mut probability_words = 0usize;
    let mut route_words = 0usize;
    for half in [false, true] {
        for (rows, stride, offset) in [
            (1, [2560, 1], 0),
            (3, [5201, 2], 1),
            (3, [5201, -2], 5119),
            (3, [5201, 0], 1),
        ] {
            let values = (0..rows as usize * 5201)
                .map(|i| ((i * 17 + 7) % 127) as f32 / 64. - 63. / 64.)
                .collect::<Vec<_>>();
            let storage = if half {
                gpu.f16_storage(
                    &values
                        .iter()
                        .map(|&v| f16::from_f32(v).to_bits())
                        .collect::<Vec<_>>(),
                )?
            } else {
                gpu.f32_storage(&values)?
            };
            let x = storage.matrix(MatrixLayout {
                shape: [rows, 2560],
                strides: stride,
                offset,
            })?;
            gpu.router_pair.enabled.set(false);
            let before = gpu.router_pair.uses.get();
            let expected =
                gpu.command(|cmd| gate.encode(&gpu, cmd, &x, Storage::StorageModeShared))?;
            ensure!(gpu.router_pair.uses.get() == before);
            let expected = words(&expected)?;
            ensure!(
                expected[0].len() == rows as usize * 512
                    && expected[1].len() == rows as usize * 512
                    && expected[2].len() == rows as usize * 10
                    && expected[3].len() == rows as usize * 10
                    && expected[0]
                        .iter()
                        .any(|&v| v & if half { 0x7fff } else { 0x7fffffff } != 0)
                    && expected[2].iter().all(|&v| v < 512)
                    && finite(&expected[0], half)
                    && finite(&expected[1], half)
                    && finite(&expected[3], half)
            );
            gpu.router_pair.enabled.set(true);
            let actual =
                gpu.command(|cmd| gate.encode(&gpu, cmd, &x, Storage::StorageModeShared))?;
            ensure!(gpu.router_pair.uses.get() - before == 1);
            ensure!(
                words(&actual)? == expected,
                "paired router dtype half={half}, rows{rows}, strides{stride:?}"
            );
            logits_words += expected[0].len();
            probability_words += expected[1].len();
            route_words += expected[2].len();
            cases += 1;
        }
    }
    // A wrong second weight row must be visible to the complete original oracle.
    let mut wrong_weight = weight.clone();
    for row in (1..512).step_by(2) {
        wrong_weight.copy_within((row - 1) * 2560..row * 2560, row * 2560);
    }
    let wrong = FlashGate {
        gpu: &gpu,
        weight: super::super::upload(&gpu.device, &wrong_weight)?,
    };
    let mut tie_weight = vec![0u16; 512 * 2560];
    for row in tie_weight.as_chunks_mut::<2560>().0 {
        row[0] = f16::from_f32(0.5).to_bits();
    }
    let tie = FlashGate {
        gpu: &gpu,
        weight: super::super::upload(&gpu.device, &tie_weight)?,
    };
    let mut x = vec![0.0; 2560];
    x[0] = -2.;
    let mutation_input = gpu.f32_storage(&x)?.matrix(MatrixLayout {
        shape: [1, 2560],
        strides: [2560, 1],
        offset: 0,
    })?;
    gpu.router_pair.enabled.set(false);
    let correct =
        gpu.command(|cmd| gate.encode(&gpu, cmd, &mutation_input, Storage::StorageModeShared))?;
    let mutant =
        gpu.command(|cmd| wrong.encode(&gpu, cmd, &mutation_input, Storage::StorageModeShared))?;
    ensure!(
        words(&correct)?[0] != words(&mutant)?[0],
        "oracle missed wrong second projection row"
    );
    for half in [false, true] {
        let storage = if half {
            gpu.f16_storage(
                &x.iter()
                    .map(|&v| f16::from_f32(v).to_bits())
                    .collect::<Vec<_>>(),
            )?
        } else {
            gpu.f32_storage(&x)?
        };
        let x = storage.matrix(MatrixLayout {
            shape: [1, 2560],
            strides: [2560, 1],
            offset: 0,
        })?;
        gpu.router_pair.enabled.set(false);
        let expected = gpu.command(|cmd| tie.encode(&gpu, cmd, &x, Storage::StorageModeShared))?;
        let expected = words(&expected)?;
        let bits = |value: f32| {
            if half {
                u32::from(f16::from_f32(value).to_bits())
            } else {
                value.to_bits()
            }
        };
        ensure!(
            expected[0] == vec![bits(-1.); 512]
                && expected[1] == vec![bits(1. / 512.); 512]
                && expected[2] == (502..512).collect::<Vec<_>>()
                && expected[3] == vec![bits(0.1); 10]
        );
        let before = gpu.router_pair.uses.get();
        gpu.router_pair.enabled.set(true);
        let actual = gpu.command(|cmd| tie.encode(&gpu, cmd, &x, Storage::StorageModeShared))?;
        ensure!(words(&actual)? == expected && gpu.router_pair.uses.get() - before == 1);
        logits_words += 512;
        probability_words += 512;
        route_words += 10;
        cases += 1;
    }
    let bad = gpu.f32_storage(&[1.])?.matrix(MatrixLayout {
        shape: [1, 1],
        strides: [1, 1],
        offset: 0,
    })?;
    let cmd = gpu.queue.new_command_buffer();
    ensure!(
        gate.encode(&gpu, cmd, &bad, Storage::StorageModeShared)
            .is_err()
    );
    ensure!(cmd.status() == metal::MTLCommandBufferStatus::NotEnqueued);
    ensure!(
        cases == 10 && logits_words == 11264 && probability_words == 11264 && route_words == 220
    );
    ensure!(gpu.router_pair.uses.get() - start == 10);
    eprintln!(
        "AR30_ROUTER_PAIR {}",
        serde_json::json!({"parity":true,"cases":cases,"finite_logit_words":logits_words,"finite_probability_words":probability_words,"route_ids":route_words,"finite_route_score_words":route_words,"actual_pair_uses":gpu.router_pair.uses.get()-start,"independent_canonical_psos":true,"wrong_second_row_mutant_detected":true,"analytic_ties":true,"negative_and_broadcast_strides":true,"rejected_before_encoding":true})
    );
    Ok(())
}
