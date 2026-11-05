//! Physical integration gate: actual routed records, independent stock words,
//! actual allocations/bindings, owners dropped before command submission.
use super::*;
use crate::tensor_layout::MatrixLayout;
use serde::Deserialize;
use std::{collections::BTreeSet, fs, os::unix::fs::FileExt, path::PathBuf};

#[derive(Deserialize)]
pub(super) struct Case {
    path: PathBuf,
    prefix: String,
    x: Vec<u16>,
}

pub(super) fn record(gpu: &Metal, case: &Case) -> Result<Buffer> {
    let c = crate::checkpoint::inspect(&case.path)?;
    let (prefix, index) = case
        .prefix
        .rsplit_once(".experts.")
        .context("routed prefix required")?;
    let records = crate::checkpoint::flash_expert_records(&c, prefix)?;
    let info = records
        .get(index.parse::<usize>()?)
        .context("missing record")?;
    let (file, offset) = info.source()?;
    let mut bytes = vec![0; flash_support::EXPERT_BYTES as usize];
    file.read_exact_at(&mut bytes, offset)?;
    upload(&gpu.device, &bytes)
}

pub(super) fn input(gpu: &Metal, case: &Case, fp32: bool) -> Result<MatrixView> {
    ensure!(case.x.len() == 2560);
    let mut values = vec![f16::from_f32(7.).to_bits(); 2564];
    for (index, &value) in case.x.iter().enumerate() {
        values[2561 - index] = value;
    }
    let storage = if fp32 {
        gpu.f32_storage(
            &values
                .iter()
                .map(|&v| f16::from_bits(v).to_f32())
                .collect::<Vec<_>>(),
        )?
    } else {
        gpu.f16_storage(&values)?
    };
    storage.matrix(MatrixLayout {
        shape: [1, 2560],
        strides: [2564, -1],
        offset: 2561,
    })
}

fn requests() -> usize {
    super::super::BUFFER_REQUESTS.with(std::cell::Cell::get)
}

fn check_bindings(gpu: &Metal, before: usize, reuse: bool) -> Result<[usize; 2]> {
    ensure!(
        gpu.flash_scratch_uses.get() - before == if reuse { 3 } else { 0 },
        "wrong scratch use count"
    );
    let ids = gpu.flash_expert_outputs.get();
    ensure!(
        ids.iter().all(|&id| id != 0) && ids.into_iter().collect::<BTreeSet<_>>().len() == 4,
        "expert destinations alias"
    );
    if !reuse {
        return Ok([0; 2]);
    }
    let pairs = gpu.flash_scratch_bindings.get();
    ensure!(
        pairs.iter().all(|&pair| pair == pairs[0])
            && pairs[0][0] != 0
            && pairs[0][1] != 0
            && pairs[0][0] != pairs[0][1]
            && ids.iter().all(|id| !pairs[0].contains(id)),
        "scratch buffers were not reused independently of destinations"
    );
    Ok(pairs[0])
}

pub(super) fn check_words(actual: &[u16], expected: &[u16]) -> Result<()> {
    ensure!(
        actual.len() == 2560
            && actual == expected
            && expected.iter().any(|&v| f16::from_bits(v).to_f32() != 0.)
            && actual.iter().all(|&v| f16::from_bits(v).is_finite()),
        "routed expert stock divergence"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; independent real expert fixtures required"]
fn physical_routed_expert_scratch() -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("MLXL3_ROUTED_SCRATCH_FIXTURES").context("fixtures required")?,
    );
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(root.join("cases.json"))?)?;
    let refs: Vec<Vec<u16>> = serde_json::from_slice(&fs::read(root.join("stock.json"))?)?;
    ensure!(
        cases.len() >= 5
            && cases.len() == refs.len()
            && cases[0].prefix != cases[1].prefix
            && cases[..cases.len() - 1]
                .iter()
                .map(|c| &c.prefix)
                .collect::<BTreeSet<_>>()
                .len()
                >= 2,
        "at least two experts and varied inputs required"
    );
    ensure!(
        cases[0].x != cases[2].x,
        "replay needs different activation values"
    );
    ensure!(
        cases
            .last()
            .context("missing shared case")?
            .prefix
            .ends_with(".shared_expert"),
        "last case must be the shared expert"
    );
    let mut gpu = Metal::new()?;
    let snapshot = gpu.flash_expert_scratch_setting;
    ensure!(
        snapshot
            == crate::flash_support::hc_chain_flag(
                std::env::var("MLXL3_NATIVE_ROUTED_SCRATCH").ok().as_deref(),
                gpu.device_name() == "Apple M5"
            )
    );
    let mut checks = Vec::new();
    for reuse in [false, true] {
        gpu.flash_expert_scratch_setting = Some(reuse);
        for (index, (case, expected)) in cases[..cases.len() - 1].iter().zip(&refs).enumerate() {
            for fp32 in [false, true] {
                let backing = record(&gpu, case)?;
                let expert = FlashExpert::from_backing(&gpu, &backing)?;
                let x = input(&gpu, case, fp32)?;
                let before = requests();
                let uses = gpu.flash_scratch_uses.get();
                let output = gpu.command(|cmd| {
                    let output = expert.encode_view(cmd, &x, Storage::StorageModeShared)?;
                    drop(x);
                    drop(expert);
                    drop(backing);
                    Ok(output)
                })?;
                let count = requests() - before;
                ensure!(
                    count == if reuse { 6 } else { 10 },
                    "actual newBuffer count: {count}"
                );
                let pair = check_bindings(&gpu, uses, reuse)?;
                check_words(&Metal::read(&output, 2560), expected)?;
                checks.push(serde_json::json!({"case":index,"fp32":fp32,"reuse":reuse,"actual_new_buffer":count,"pair":pair}));
            }
        }
    }
    gpu.flash_expert_scratch_setting = Some(true);
    // Three encodes in a single command: two different experts and one duplicate.
    let mut pairs = BTreeSet::new();
    let outputs = gpu.command(|cmd| {
        [0, 1, 0]
            .into_iter()
            .map(|index| {
                let case = &cases[index];
                let backing = record(&gpu, case)?;
                let expert = FlashExpert::from_backing(&gpu, &backing)?;
                let x = input(&gpu, case, true)?;
                let before = requests();
                let uses = gpu.flash_scratch_uses.get();
                let output = expert.encode_view(cmd, &x, Storage::StorageModeShared)?;
                ensure!(requests() - before == 6);
                let pair = check_bindings(&gpu, uses, true)?;
                ensure!(
                    pairs.insert(pair[0]) && pairs.insert(pair[1]),
                    "scratch shared across experts"
                );
                Ok(output)
            })
            .collect::<Result<Vec<_>>>()
    })?;
    for (output, index) in outputs.iter().zip([0, 1, 0]) {
        check_words(&Metal::read(output, 2560), &refs[index])?;
    }
    // Poison both oversized temporary backings, then execute all three real
    // projections. The down stage must overwrite and consume only its prefixes.
    let backing = record(&gpu, &cases[2])?;
    let expert = FlashExpert::from_backing(&gpu, &backing)?;
    let x = upload(&gpu.device, &cases[2].x)?;
    let prepared = upload(&gpu.device, &vec![0xffu8; 5120])?;
    let partial = upload(&gpu.device, &vec![0xffu8; 10240])?;
    let before_down = buffer(&gpu.device, 5120, Storage::StorageModeShared)?;
    let output = gpu.command(|cmd| {
        let scratch = Some((&prepared, &partial));
        let gate = expert.gate.encode_with_scratch(
            cmd,
            Activation::Contiguous(&x),
            crate::flash_expert_scratch::EXTENTS[0],
            Storage::StorageModePrivate,
            scratch,
        )?;
        let up = expert.up.encode_with_scratch(
            cmd,
            Activation::Contiguous(&x),
            crate::flash_expert_scratch::EXTENTS[1],
            Storage::StorageModePrivate,
            scratch,
        )?;
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(&prepared, 0, &before_down, 0, 5120);
        blit.end_encoding();
        let activated = gpu.encode_swiglu(cmd, &gate, &up, Storage::StorageModePrivate)?;
        expert.down.encode_with_scratch(
            cmd,
            Activation::Contiguous(&activated),
            crate::flash_expert_scratch::EXTENTS[2],
            Storage::StorageModeShared,
            scratch,
        )
    })?;
    check_words(&Metal::read(&output, 2560), &refs[2])?;
    ensure!(
        Metal::read(&prepared, 2560)[640..] == Metal::read(&before_down, 2560)[640..],
        "down overwrote the prepared tail beyond its640values"
    );
    let cmd = gpu.queue.new_command_buffer();
    ensure!(
        expert
            .gate
            .encode_with_scratch(
                cmd,
                Activation::Contiguous(&x),
                crate::flash_expert_scratch::EXTENTS[0],
                Storage::StorageModeShared,
                Some((&partial, &partial))
            )
            .is_err()
    );
    ensure!(cmd.status() == MTLCommandBufferStatus::NotEnqueued);
    // Capacity rejections happen before any encoder; a subsequent valid expert
    // must still work. A complete expert encoded before an injected error is
    // never submitted.
    for capacities in [[5119, 10240], [5120, 2559], [0, 10240]] {
        let short = if capacities[0] == 0 {
            prepared.clone()
        } else {
            buffer(&gpu.device, capacities[0], Storage::StorageModePrivate)?
        };
        let second = buffer(&gpu.device, capacities[1], Storage::StorageModePrivate)?;
        let cmd = gpu.queue.new_command_buffer();
        let extent = if capacities[0] == 0 {
            [0, 640, 640]
        } else {
            crate::flash_expert_scratch::EXTENTS[0]
        };
        ensure!(
            expert
                .gate
                .encode_with_scratch(
                    cmd,
                    Activation::Contiguous(&x),
                    extent,
                    Storage::StorageModeShared,
                    Some((&short, &second))
                )
                .is_err()
        );
        ensure!(cmd.status() == MTLCommandBufferStatus::NotEnqueued);
    }
    let mut observed = None;
    let rejected = submit_encoded(&gpu, |cmd| {
        let _ = expert.encode(
            cmd,
            Activation::Contiguous(&x),
            crate::flash_expert_scratch::EXTENTS,
            Storage::StorageModeShared,
        )?;
        observed = Some(cmd.to_owned());
        bail!("injected expert before commit")
    });
    ensure!(
        rejected.is_err()
            && observed.context("missing failure command")?.status()
                == MTLCommandBufferStatus::NotEnqueued
    );
    ensure!(expert.submit(&[]).is_err());
    check_words(&expert.submit(&cases[2].x)?.read()?, &refs[2])?;
    // A malformed flag rejects only the eligible record path. Shared and batch
    // inputs continue through the original path, with no scratch uses.
    drop(expert);
    gpu.flash_expert_scratch_setting = None;
    let expert = FlashExpert::from_backing(&gpu, &backing)?;
    ensure!(expert.submit(&cases[2].x).is_err());
    let repeated = [cases[2].x.as_slice(), cases[2].x.as_slice()].concat();
    let uses = gpu.flash_scratch_uses.get();
    let batch = expert.submit(&repeated)?.read()?;
    ensure!(
        gpu.flash_scratch_uses.get() == uses
            && batch == [refs[2].as_slice(), refs[2].as_slice()].concat()
    );
    let c = crate::checkpoint::inspect(&cases[0].path)?;
    let shared = cases.last().context("missing shared case")?;
    let ordinary = FlashExpert::load(&gpu, &c, &shared.prefix)?;
    check_words(
        &ordinary.submit(&shared.x)?.read()?,
        refs.last().context("missing shared reference")?,
    )?;
    ensure!(gpu.flash_scratch_uses.get() == uses);
    fs::write(
        root.join("native-scratch.json"),
        serde_json::to_vec_pretty(
            &serde_json::json!({"parity":true,"cases":checks,"duplicate_encodes":3,"poison":true,"rejections_recovered":true,"fallbacks":true}),
        )?,
    )?;
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; independent real routed record fixtures required"]
fn physical_expert_ar_epilogues_match_independent_stock() -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("MLXL3_ROUTED_SCRATCH_FIXTURES").context("fixtures required")?,
    );
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(root.join("cases.json"))?)?;
    let refs: Vec<Vec<u16>> = serde_json::from_slice(&fs::read(root.join("stock.json"))?)?;
    ensure!(cases.len() >= 5 && cases.len() == refs.len() && cases[0].prefix != cases[1].prefix);
    let gpu = Metal::new()?;
    let initial = gpu.expert_ar_enabled.each_ref().map(std::cell::Cell::get);
    let mut compared = 0usize;
    for cell in &gpu.expert_ar_enabled {
        cell.set(false);
    }
    for (enabled, dual) in [(false, false), (true, false), (false, true), (true, true)] {
        gpu.expert_ar_enabled[0].set(enabled);
        gpu.expert_ar_enabled[1].set(dual);
        for (case, expected) in cases[..4].iter().zip(&refs) {
            for fp32 in [false, true] {
                let backing = record(&gpu, case)?;
                let expert = FlashExpert::from_backing(&gpu, &backing)?;
                let x = input(&gpu, case, fp32)?;
                let uses = gpu.expert_ar_uses[0].get();
                let dual_uses = gpu.expert_ar_uses[1].get();
                if dual {
                    let expected_gate = gpu.prepare_view(
                        &x,
                        &MatrixView::column_scales(
                            &expert.gate.suh,
                            expert.gate.offsets.suh,
                            1,
                            2560,
                        )?,
                    )?;
                    let expected_up = gpu.prepare_view(
                        &x,
                        &MatrixView::column_scales(&expert.up.suh, expert.up.offsets.suh, 1, 2560)?,
                    )?;
                    let prefixes = gpu.command(|cmd| {
                        let output = upload(&gpu.device, &vec![0x7e00u16; 5120])?;
                        Metal::encode_bound(
                            cmd,
                            &gpu.expert_gu_prepare,
                            [
                                (&x.buffer, 0),
                                (&expert.gate.suh, expert.gate.offsets.suh),
                                (&expert.up.suh, expert.up.offsets.suh),
                                (&output, 0),
                            ]
                            .into_iter(),
                            [x.layout.offset, x.layout.strides[1], i64::from(fp32), 2560],
                            MTLSize::new(20, 1, 1),
                            64,
                        );
                        Ok(output)
                    })?;
                    let words = Metal::read(&prefixes, 5120);
                    ensure!(words[..2560] == expected_gate && words[2560..] == expected_up);
                }
                let allocations = requests();
                let output = gpu.command(|cmd| {
                    let out = expert.encode_view(cmd, &x, Storage::StorageModeShared)?;
                    drop(x);
                    drop(expert);
                    drop(backing);
                    Ok(out)
                })?;
                ensure!(gpu.expert_ar_uses[0].get() - uses == u64::from(enabled));
                ensure!(gpu.expert_ar_uses[1].get() - dual_uses == u64::from(dual));
                ensure!(requests() - allocations == if enabled { 5 } else { 6 });
                check_words(&Metal::read(&output, 2560), expected)?;
                compared += 2560;
            }
        }
    }
    ensure!(compared == 81920);
    let synthetic_x = gpu.f16_storage(&vec![0x3c00; 2560])?.matrix(MatrixLayout {
        shape: [1, 2560],
        strides: [2560, 1],
        offset: 0,
    })?;
    let gate_scale = gpu.f16_storage(&vec![0x3c00; 2560])?.matrix(MatrixLayout {
        shape: [1, 2560],
        strides: [2560, 1],
        offset: 0,
    })?;
    let up_scale = gpu.f16_storage(&vec![0x4000; 2560])?.matrix(MatrixLayout {
        shape: [1, 2560],
        strides: [2560, 1],
        offset: 0,
    })?;
    let expected_gate = gpu.prepare_view(&synthetic_x, &gate_scale)?;
    let expected_up = gpu.prepare_view(&synthetic_x, &up_scale)?;
    ensure!(expected_gate != expected_up);
    for swapped in [false, true, false] {
        let (gate, up) = if swapped {
            (&up_scale, &gate_scale)
        } else {
            (&gate_scale, &up_scale)
        };
        let output = gpu.command(|cmd| {
            let out = upload(&gpu.device, &vec![0x7e00u16; 5120])?;
            Metal::encode_bound(
                cmd,
                &gpu.expert_gu_prepare,
                [
                    (&synthetic_x.buffer, 0),
                    (&gate.buffer, 0),
                    (&up.buffer, 0),
                    (&out, 0),
                ]
                .into_iter(),
                [0i64, 1, 0, 2560],
                MTLSize::new(20, 1, 1),
                64,
            );
            Ok(out)
        })?;
        let words = Metal::read(&output, 5120);
        ensure!(
            (words[..2560] == expected_gate && words[2560..] == expected_up) == !swapped,
            "Suh swap mutation was not detected/restored"
        );
    }
    for cell in &gpu.expert_ar_enabled {
        cell.set(false);
    }
    let slab = upload(&gpu.device, &vec![0x7e00u16; 25600])?;
    let before = requests();
    let command = gpu.queue.new_command_buffer().to_owned();
    for route in [9, 1, 5, 0, 8, 2, 6, 4, 3, 7] {
        let case = &cases[route % 4];
        let backing = record(&gpu, case)?;
        let expert = FlashExpert::from_backing(&gpu, &backing)?;
        let x = input(&gpu, case, route % 2 != 0)?;
        let prefix = crate::expert_ar_policy::prefix(route, 10, 2560, slab.length())
            .context("slab prefix")?;
        let output = expert.encode_view_into(
            &command,
            &x,
            Storage::StorageModeShared,
            Some((&slab, prefix)),
        )?;
        ensure!(output.as_ptr() == slab.as_ptr());
    }
    command.commit();
    command.wait_until_completed();
    ensure!(command.status() == MTLCommandBufferStatus::Completed);
    // Each record upload and input upload adds two requests outside expert encoding.
    ensure!(requests() - before == 70);
    let words = Metal::read(&slab, 25600);
    for route in 0..10 {
        check_words(&words[route * 2560..(route + 1) * 2560], &refs[route % 4])?;
    }
    eprintln!(
        "NATIVE_EXPERT_AR_COMPONENT {}",
        serde_json::json!({"parity":true,"words":compared,"slab_words":25600,"slab_reordered_routes":10,"suh_swap_detected_and_restored":true,"initial":initial,"up_finish_uses":gpu.expert_ar_uses[0].get()})
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; independent real routed record fixtures required"]
fn physical_expert_axis_matches_independent_stock_and_ready_cohorts() -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("MLXL3_ROUTED_SCRATCH_FIXTURES").context("fixtures required")?,
    );
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(root.join("cases.json"))?)?;
    let refs: Vec<Vec<u16>> = serde_json::from_slice(&fs::read(root.join("stock.json"))?)?;
    ensure!(cases.len() >= 5 && cases.len() == refs.len() && cases[0].prefix != cases[1].prefix);
    let gpu = Metal::new()?;
    for cell in &gpu.expert_ar_enabled {
        cell.set(false);
    }
    let mut words = 0usize;
    let mut cohorts = 0usize;
    for composed in [false, true] {
        gpu.expert_ar_enabled[0].set(composed);
        gpu.expert_ar_enabled[1].set(composed);
        gpu.expert_ar_enabled[3].set(true);
        for (case, expected) in cases[..4].iter().zip(&refs) {
            for fp32 in [false, true] {
                for n in [1, 2, 10] {
                    let backing = record(&gpu, case)?;
                    let experts = (0..n)
                        .map(|_| FlashExpert::from_backing(&gpu, &backing))
                        .collect::<Result<Vec<_>>>()?;
                    let x = input(&gpu, case, fp32)?;
                    let before = gpu.expert_axis_dispatches.get();
                    let uses = gpu.expert_ar_uses[3].get();
                    let finish_uses = gpu.expert_ar_uses[0].get();
                    let prepare_uses = gpu.expert_ar_uses[1].get();
                    let outputs = gpu.command(|cmd| {
                        let expert_refs = experts.iter().collect::<Vec<_>>();
                        let out = FlashExpert::encode_ready_cohort(
                            cmd,
                            &x,
                            &expert_refs,
                            &vec![None; n],
                            Storage::StorageModeShared,
                        )?;
                        drop(expert_refs);
                        drop(experts);
                        drop(backing);
                        drop(x);
                        Ok(out)
                    })?;
                    ensure!(
                        gpu.expert_axis_dispatches.get() - before == 3
                            && gpu.expert_ar_uses[3].get() - uses == n as u64
                    );
                    ensure!(
                        gpu.expert_ar_uses[0].get() - finish_uses
                            == if composed { n as u64 } else { 0 }
                    );
                    ensure!(
                        gpu.expert_ar_uses[1].get() - prepare_uses
                            == if composed { n as u64 } else { 0 }
                    );
                    ensure!(
                        outputs
                            .iter()
                            .map(|b| b.as_ptr() as usize)
                            .collect::<BTreeSet<_>>()
                            .len()
                            == n
                    );
                    for output in outputs {
                        check_words(&Metal::read(&output, 2560), expected)?;
                        words += 2560;
                    }
                    cohorts += 1;
                }
            }
        }
    }
    // Distinct record identities share one input; capture the canonical branch
    // before running the candidate, then compare each permuted slab prefix.
    for cell in &gpu.expert_ar_enabled {
        cell.set(false);
    }
    let backings = cases[..2]
        .iter()
        .map(|case| record(&gpu, case))
        .collect::<Result<Vec<_>>>()?;
    let experts = backings
        .iter()
        .map(|backing| FlashExpert::from_backing(&gpu, backing))
        .collect::<Result<Vec<_>>>()?;
    let x = input(&gpu, &cases[0], true)?;
    let expected = experts
        .iter()
        .map(|expert| expert.submit_fp16_view(&x)?.read())
        .collect::<Result<Vec<_>>>()?;
    ensure!(expected[0] == refs[0] && expected[0] != expected[1]);
    for cell in &gpu.expert_ar_enabled {
        cell.set(true);
    }
    let slab = upload(&gpu.device, &vec![0x7e00u16; 25600])?;
    let order = [1, 0, 0, 1, 0, 1, 1, 1, 0, 1];
    let expert_refs = order.iter().map(|&i| &experts[i]).collect::<Vec<_>>();
    let destinations = (0..10).map(|i| Some((&slab, i * 5120))).collect::<Vec<_>>();
    let before = gpu.expert_axis_dispatches.get();
    gpu.command(|cmd| {
        FlashExpert::encode_ready_cohort(
            cmd,
            &x,
            &expert_refs,
            &destinations,
            Storage::StorageModeShared,
        )?;
        Ok(())
    })?;
    ensure!(gpu.expert_axis_dispatches.get() - before == 3);
    let actual = Metal::read(&slab, 25600);
    for (i, &record) in order.iter().enumerate() {
        check_words(&actual[i * 2560..(i + 1) * 2560], &expected[record])?;
    }
    let before = gpu.expert_axis_dispatches.get();
    let command = gpu.queue.new_command_buffer().to_owned();
    ensure!(
        FlashExpert::encode_ready_cohort(&command, &x, &[], &[], Storage::StorageModeShared)
            .is_err()
    );
    ensure!(
        FlashExpert::encode_ready_cohort(
            &command,
            &x,
            &expert_refs,
            &[],
            Storage::StorageModeShared
        )
        .is_err()
    );
    ensure!(
        gpu.expert_axis_dispatches.get() == before
            && command.status() == MTLCommandBufferStatus::NotEnqueued
    );
    // An undersized final destination errors after real QMV encodes, without
    // committing the command. A fresh command on the same owners must recover.
    let bad = [Some((&slab, slab.length() - 2))];
    ensure!(
        FlashExpert::encode_ready_cohort(
            &command,
            &x,
            &expert_refs[..1],
            &bad,
            Storage::StorageModeShared
        )
        .is_err()
    );
    ensure!(command.status() == MTLCommandBufferStatus::NotEnqueued);
    drop(command);
    let output = gpu.command(|cmd| {
        FlashExpert::encode_ready_cohort(
            cmd,
            &x,
            &expert_refs[..1],
            &[None],
            Storage::StorageModeShared,
        )
    })?;
    check_words(&Metal::read(&output[0], 2560), &expected[1])?;
    ensure!(cohorts == 48 && words == 532480);
    eprintln!(
        "NATIVE_EXPERT_AXIS_COMPONENT {}",
        serde_json::json!({"parity":true,"cohorts":cohorts,"stock_words":words,"mixed_slab_words":25600,"dispatches_per_cohort":3,"empty_rejected":true,"uncommitted_error_recovered":true})
    );
    Ok(())
}
