use super::routed_scratch_tests::{Case, check_words, input, record};
use super::*;
use crate::tensor_layout::MatrixLayout;
use std::{fs, path::PathBuf};

fn row(gpu: &Metal, values: &[f32]) -> Result<MatrixView> {
    let width = u32::try_from(values.len())?;
    gpu.f32_storage(values)?.matrix(MatrixLayout {
        shape: [1, width],
        strides: [i64::from(width), 1],
        offset: 0,
    })
}
fn finite(words: &[u32]) -> bool {
    words.len() == 2560
        && words.iter().all(|&w| f32::from_bits(w).is_finite())
        && words.iter().any(|&w| f32::from_bits(w) != 0.)
}

#[test]
#[ignore = "physical real EXL3 pending finish/mix, frozen independent stock + canonical mix, run alone"]
fn physical_ar20_pending_finish_mix_preserves_canonical_routes_and_owners() -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("MLXL3_ROUTED_SCRATCH_FIXTURES").context("fixtures required")?,
    );
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(root.join("cases.json"))?)?;
    let stock: Vec<Vec<u16>> = serde_json::from_slice(&fs::read(root.join("stock.json"))?)?;
    ensure!(cases.len() >= 5 && cases.len() == stock.len());
    let gpu = Metal::new()?;
    let pipeline = gpu
        .moe_finish_mix_pipeline
        .as_ref()
        .context("initialize MOE_FINISH_MIX=1")?;
    ensure!(
        gpu.moe_finish_mix.get()
            && gpu.expert_ar_enabled[3].get()
            && pipeline.thread_execution_width() == 32
            && pipeline.max_total_threads_per_threadgroup() >= 640
    );
    let initial = gpu.moe_finish_mix_uses.get();
    let axis = gpu.expert_ar_uses[3].get();
    let order = [9usize, 1, 5, 0, 8, 2, 6, 4, 3, 7];
    let mixed = [1usize, 0, 0, 1, 0, 1, 1, 1, 0, 1];
    let mut checked = 0usize;
    for phase in 0..4 {
        let backings = cases[..4]
            .iter()
            .map(|c| record(&gpu, c))
            .collect::<Result<Vec<_>>>()?;
        let experts = backings
            .iter()
            .map(|b| FlashExpert::from_backing(&gpu, b))
            .collect::<Result<Vec<_>>>()?;
        let inputs = cases[..4]
            .iter()
            .map(|c| input(&gpu, c, true))
            .collect::<Result<Vec<_>>>()?;
        let shared_bits = (0..2560)
            .map(|i| f16::from_f32((i as f32 % 113. - 56.) / 97.).to_bits())
            .collect::<Vec<_>>();
        let shared = upload(&gpu.device, &shared_bits)?;
        let scores = row(
            &gpu,
            &[
                0.125, -0.25, 0.5, 0.75, -0.125, 0.0625, 0.875, -0.5, 0.25, 0.375,
            ],
        )?;
        let multiplier = row(&gpu, &[-0.75])?;
        let mut route_cases = [0usize; 10];
        let groups: Vec<Vec<usize>> = match phase {
            0 => order.iter().map(|&r| vec![r]).collect(),
            1 => order.chunks_exact(2).map(|g| g.to_vec()).collect(),
            _ => vec![order.to_vec()],
        };
        for (g, routes) in groups.iter().enumerate() {
            for &r in routes {
                route_cases[r] = match phase {
                    0 => r % 4,
                    1 => g % 4,
                    2 => 0,
                    _ => mixed[r],
                };
            }
        }
        let reference = if phase < 3 {
            route_cases
                .iter()
                .map(|&i| stock[i].clone())
                .collect::<Vec<_>>()
        } else {
            let canonical = experts[..2]
                .iter()
                .map(|e| e.submit_fp16_view(&inputs[0])?.read())
                .collect::<Result<Vec<_>>>()?;
            check_words(&canonical[0], &stock[0])?;
            ensure!(
                canonical[0] != canonical[1]
                    && canonical.iter().all(
                        |r| r.len() == 2560 && r.iter().all(|&v| f16::from_bits(v).is_finite())
                    )
            );
            route_cases.iter().map(|&i| canonical[i].clone()).collect()
        };
        for words in &reference {
            ensure!(words.len() == 2560 && words.iter().all(|&v| f16::from_bits(v).is_finite()));
        }
        let finished = reference
            .iter()
            .map(|w| upload(&gpu.device, w))
            .collect::<Result<Vec<_>>>()?;
        let expected = MatrixView::output(
            &gpu,
            super::super::view::FloatType::F32,
            1,
            2560,
            Storage::StorageModeShared,
        )?;
        gpu.command(|cmd| {
            let mut bindings = finished.iter().map(|b| (b, 0)).collect::<Vec<_>>();
            bindings.extend([
                (&shared, 0),
                (&scores.buffer, 0),
                (&multiplier.buffer, 0),
                (&expected.buffer, 0),
            ]);
            Metal::encode_bound(
                cmd,
                &gpu.moe_mix[1],
                bindings.into_iter(),
                [2560u32],
                MTLSize::new(80, 1, 1),
                32,
            );
            Ok(())
        })?;
        let expected = expected.read_output()?;
        ensure!(finite(&expected));
        drop(finished);
        let command = gpu.queue.new_command_buffer().to_owned();
        let mut pending: Vec<Option<PendingExpertFinish>> = (0..10).map(|_| None).collect();
        for routes in &groups {
            let idx = route_cases[routes[0]];
            let refs = routes
                .iter()
                .map(|&r| &experts[route_cases[r]])
                .collect::<Vec<_>>();
            let x = if phase == 3 { &inputs[0] } else { &inputs[idx] };
            let values = FlashExpert::encode_ready_cohort_pending(
                &command,
                x,
                &refs,
                &vec![None; routes.len()],
            )?;
            ensure!(values.len() == routes.len());
            for (&r, term) in routes.iter().zip(values) {
                ensure!(pending[r].replace(term).is_none());
            }
        }
        drop(experts);
        drop(backings);
        drop(inputs);
        let terms = pending
            .iter()
            .map(|p| p.as_ref().context("missing route"))
            .collect::<Result<Vec<_>>>()?;
        let actual = MatrixView::output(
            &gpu,
            super::super::view::FloatType::F32,
            1,
            2560,
            Storage::StorageModeShared,
        )?;
        if phase == 0 {
            let bad = MatrixView::output(
                &gpu,
                super::super::view::FloatType::F16,
                1,
                2560,
                Storage::StorageModeShared,
            )?;
            let before = gpu.moe_finish_mix_uses.get();
            ensure!(
                PendingExpertFinish::encode_mix(
                    &gpu,
                    &command,
                    &terms,
                    &shared,
                    [&scores, &multiplier, &bad]
                )
                .is_err()
            );
            ensure!(
                gpu.moe_finish_mix_uses.get() == before
                    && command.status() == MTLCommandBufferStatus::NotEnqueued
            );
            let corrupt = PendingExpertFinish {
                partial: terms[0].partial.clone(),
                partial_offset: terms[0].partial.length(),
                svh: terms[0].svh.clone(),
                svh_offset: terms[0].svh_offset,
            };
            let mut wrong = terms.clone();
            wrong[0] = &corrupt;
            ensure!(
                PendingExpertFinish::encode_mix(
                    &gpu,
                    &command,
                    &wrong,
                    &shared,
                    [&scores, &multiplier, &actual]
                )
                .is_err()
            );
            ensure!(
                gpu.moe_finish_mix_uses.get() == before
                    && command.status() == MTLCommandBufferStatus::NotEnqueued
            );
        }
        PendingExpertFinish::encode_mix(
            &gpu,
            &command,
            &terms,
            &shared,
            [&scores, &multiplier, &actual],
        )?;
        drop(terms);
        drop(pending);
        drop(scores);
        drop(multiplier);
        drop(shared);
        command.commit();
        command.wait_until_completed();
        ensure!(command.status() == MTLCommandBufferStatus::Completed);
        let words = actual.read_output()?;
        ensure!(
            finite(&words) && words == expected,
            "pending finish/mix differs phase{phase}"
        );
        checked += words.len();
    }
    ensure!(
        gpu.moe_finish_mix_uses.get() - initial == 4 && gpu.expert_ar_uses[3].get() - axis == 40
    );
    eprintln!(
        "AR20_MOE_FINISH_MIX {}",
        serde_json::json!({"parity":true,"cases":4,"finite_float_words":checked,"actual_mix_uses":4,"actual_axis_experts":40,"frozen_stock_route_words":76800,"distinct_records":2,"reordered_duplicate_routes":true,"retained_owners_dropped_before_commit":true,"invalid_views_rejected":true})
    );
    Ok(())
}
