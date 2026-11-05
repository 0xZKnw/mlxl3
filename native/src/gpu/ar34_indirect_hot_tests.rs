use super::*;
use crate::{checkpoint, flash_support, tensor_layout::MatrixLayout};
use half::f16;
use std::{fs, os::unix::fs::FileExt, path::PathBuf};

fn fixture(gpu: &Metal) -> Result<(Buffer, MatrixView, Vec<u16>)> {
    let root = PathBuf::from(
        std::env::var_os("MLXL3_ROUTED_SCRATCH_FIXTURES").context("routed fixtures required")?,
    );
    let cases: Vec<serde_json::Value> =
        serde_json::from_slice(&fs::read(root.join("cases.json"))?)?;
    let mut refs: Vec<Vec<u16>> = serde_json::from_slice(&fs::read(root.join("stock.json"))?)?;
    ensure!(cases.len() >= 5 && refs.len() == cases.len() && refs[0].len() == 2560);
    let first = &cases[0];
    let c = checkpoint::inspect(&PathBuf::from(
        first["path"].as_str().context("fixture path")?,
    ))?;
    let (prefix, index) = first["prefix"]
        .as_str()
        .context("fixture prefix")?
        .rsplit_once(".experts.")
        .context("routed expert required")?;
    let records = checkpoint::flash_expert_records(&c, prefix)?;
    let info = records
        .get(index.parse::<usize>()?)
        .context("fixture record")?;
    let (file, offset) = info.source()?;
    let mut bytes = vec![0; flash_support::EXPERT_BYTES as usize];
    file.read_exact_at(&mut bytes, offset)?;
    let record = upload(&gpu.device, &bytes)?;
    let x: Vec<u16> = serde_json::from_value(first["x"].clone())?;
    ensure!(x.len() == 2560);
    let x = gpu
        .f32_storage(
            &x.into_iter()
                .map(|v| f16::from_bits(v).to_f32())
                .collect::<Vec<_>>(),
        )?
        .matrix(MatrixLayout {
            shape: [1, 2560],
            strides: [2560, 1],
            offset: 0,
        })?;
    Ok((record, x, refs.remove(0)))
}

fn words(hot: &HotOutput) -> Result<Vec<u16>> {
    hot.completed()?;
    let value = Metal::read(&hot.slab, 25600);
    ensure!(value.len() == 25600 && value.iter().all(|&v| f16::from_bits(v).is_finite()));
    Ok(value)
}

#[test]
#[ignore = "physical AR34 Tier2 argument table53, independent ten-binding projections, missing/invalid/stale/retained/mutants; exclusive GPU"]
fn physical_ar34_indirect_hot_preserves_all_slots_and_canonical_contributions() -> Result<()> {
    let gpu = Metal::new()?;
    ensure!(
        gpu.device_resident_route_available(),
        "enable DEVICE_RESIDENT_ROUTE on exact M5/Tier2"
    );
    let function = gpu
        .ar20_library
        .get_function("device_hot_compiled", None)
        .map_err(anyhow::Error::msg)?;
    let compiled = gpu
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(anyhow::Error::msg)?;
    let flag = upload(&gpu.device, &[0u32])?;
    gpu.command(|cmd| {
        Metal::encode(cmd, &compiled, &[&flag], [0u32], MTLSize::new(1, 1, 1), 1);
        Ok(())
    })?;
    ensure!(Metal::read(&flag, 2) == [1, 0]);
    let (base, x, frozen) = fixture(&gpu)?;
    let original = Metal::read(&base, flash_support::EXPERT_BYTES as usize / 2);
    let offset = flash_support::record_offsets(0).unwrap().svh as usize / 2;
    let mut records = Vec::new();
    for slot in 0..53 {
        let mut data = original.clone();
        for word in &mut data[offset..offset + 2560] {
            let value = f16::from_f32(f16::from_bits(*word).to_f32() * (1. + slot as f32 / 64.));
            ensure!(value.is_finite());
            *word = value.to_bits();
        }
        records.push(upload(&gpu.device, &data)?);
    }
    let experts = records
        .iter()
        .map(|r| FlashExpert::from_backing(&gpu, r))
        .collect::<Result<Vec<_>>>()?;
    let reference = gpu.command(|cmd| {
        let mut outputs = Vec::new();
        for chunk in experts.chunks(10) {
            outputs.extend(FlashExpert::encode_ready_cohort(
                cmd,
                &x,
                &chunk.iter().collect::<Vec<_>>(),
                &vec![None; chunk.len()],
                Storage::StorageModeShared,
            )?);
        }
        Ok(outputs)
    })?;
    let reference = reference
        .iter()
        .map(|r| Metal::read(r, 2560))
        .collect::<Vec<_>>();
    ensure!(
        reference.len() == 53
            && reference[0] == frozen
            && reference
                .iter()
                .all(|r| r.len() == 2560 && r.iter().all(|&v| f16::from_bits(v).is_finite()))
            && reference[0] != reference[1]
    );
    let expert_id = |slot: usize| (slot as u32 * 7 + 13) % 512;
    let map_of = |slots: &[usize]| {
        let mut map = [u32::MAX; 512];
        for &slot in slots {
            map[expert_id(slot) as usize] = slot as u32;
        }
        map
    };
    let all = (0..53).collect::<Vec<_>>();
    let make = |map| ResidentSnapshot::for_test(map, records.iter().cloned().map(Some).collect());
    let mut checked = 0;
    let mut cases = 0;
    for (start, hits) in
        (0..53)
            .step_by(10)
            .map(|s| (s, 10))
            .chain([(0, 0), (0, 1), (0, 9), (0, 10)])
    {
        let slots = (0..10).map(|r| (start + r) % 53).collect::<Vec<_>>();
        let map = if hits == 10 {
            map_of(&all)
        } else {
            map_of(&slots[..hits])
        };
        let ids = upload(
            &gpu.device,
            &slots.iter().map(|&s| expert_id(s)).collect::<Vec<_>>(),
        )?;
        let snapshot = make(map)?;
        let hot = gpu.submit_resident_hot_inner(&x, &ids, snapshot, Storage::StorageModeShared)?;
        ensure!(
            !gpu.device_resident_route_available(),
            "active snapshot admitted a second transaction"
        );
        let actual = words(&hot)?;
        for (route, &slot) in slots.iter().enumerate() {
            let row = &actual[route * 2560..(route + 1) * 2560];
            if route < hits {
                ensure!(row == reference[slot]);
            } else {
                ensure!(row.iter().all(|&w| w == 0));
            }
            ensure!(hot.contribution(route)?.1 == route as u64 * 5120);
        }
        ensure!(hot.contribution(10).is_err() && hot.contribution(usize::MAX).is_err());
        checked += actual.len();
        cases += 1;
        drop(hot);
        ensure!(gpu.device_resident_route_available());
    }
    let full_map = map_of(&all);
    let mut current = vec![None; 53];
    for &slot in &all {
        current[slot] = Some((expert_id(slot), 1));
    }
    let valid_ids = (0..10).map(expert_id).collect::<Vec<_>>();
    let mut invalid_ids = valid_ids.clone();
    invalid_ids[0] = 512;
    invalid_ids[1] = u32::MAX;
    let ids = upload(&gpu.device, &invalid_ids)?;
    let hot =
        gpu.submit_resident_hot_inner(&x, &ids, make(full_map)?, Storage::StorageModeShared)?;
    let actual = words(&hot)?;
    ensure!(actual[..5120].iter().all(|&w| w == 0));
    ensure!(
        hot.snapshot()
            .validate_routes(&current, &invalid_ids)
            .is_err()
    );
    drop(hot);
    let ids = upload(&gpu.device, &valid_ids)?;
    let hot =
        gpu.submit_resident_hot_inner(&x, &ids, make(full_map)?, Storage::StorageModeShared)?;
    let mut stale = current.clone();
    stale[0] = Some((expert_id(0), 2));
    ensure!(hot.snapshot().validate_routes(&stale, &valid_ids).is_err());
    ensure!(
        hot.snapshot()
            .validate_routes(&current, &valid_ids)?
            .iter()
            .all(Option::is_some)
    );
    let actual = words(&hot)?;
    ensure!(actual[..2560] == reference[0]);
    drop(hot);
    let p = gpu.indirect_hot.as_ref().unwrap();
    p.wrong_binding.set(true);
    let mutant =
        gpu.submit_resident_hot_inner(&x, &ids, make(full_map)?, Storage::StorageModeShared)?;
    p.wrong_binding.set(false);
    let wrong = words(&mutant)?;
    ensure!(
        wrong[..2560] != reference[0],
        "shifted pointer table escaped the independent oracle"
    );
    drop(mutant);
    let mut duplicate_ids = valid_ids.clone();
    duplicate_ids[1] = duplicate_ids[0];
    let duplicates = upload(&gpu.device, &duplicate_ids)?;
    p.reject_result.set(true);
    let rejected = gpu.submit_resident_hot_inner(
        &x,
        &duplicates,
        make(full_map)?,
        Storage::StorageModeShared,
    )?;
    p.reject_result.set(false);
    ensure!(rejected.completed().is_err());
    ensure!(rejected.command.status() == MTLCommandBufferStatus::Completed);
    let duplicate_rows = Metal::read(&rejected.slab, 25600);
    ensure!(
        duplicate_rows.len() == 25600
            && duplicate_rows
                .iter()
                .all(|&v| f16::from_bits(v).is_finite())
            && duplicate_rows[..2560] == reference[0]
            && duplicate_rows[2560..5120] == reference[0]
    );
    ensure!(
        rejected
            .snapshot()
            .validate_routes(&current, &duplicate_ids)
            .is_err()
    );
    drop(rejected);
    let snapshot = make(full_map)?;
    drop(experts);
    drop(records);
    drop(base);
    let mut retained =
        gpu.submit_resident_hot_inner(&x, &ids, snapshot, Storage::StorageModeShared)?;
    ensure!(words(&retained)?[..2560] == reference[0]);
    let unsubmitted = gpu.queue.new_command_buffer().to_owned();
    retained.retain_consumer(&unsubmitted)?;
    ensure!(retained.retain_consumer(&unsubmitted).is_err());
    drop(retained);
    ensure!(unsubmitted.status() == MTLCommandBufferStatus::NotEnqueued);
    ensure!(gpu.device_resident_route_available());
    let mut invalid_map = full_map;
    invalid_map[expert_id(0) as usize] = 53;
    ensure!(ResidentSnapshot::for_test(invalid_map, vec![None; 53]).is_err());
    ensure!(checked == 256000 && cases == 10 && gpu.device_hot_submissions() == 15);
    eprintln!(
        "AR34_INDIRECT_HOT {}",
        serde_json::json!({
            "parity":true,"compiled_device_lookup":true,"slots":53,"canonical_half_words":135680,
            "compared_half_words":checked,"cases":cases,"hit_counts":[0,1,9,10],
            "invalid_ids_zeroed_then_rejected":true,"duplicate_ids_rejected":true,"stale_generation_rejected":true,
            "retained_snapshot":true,"wrong_pointer_mutant_detected":true,
            "unsubmitted_consumer_abandon":true,
            "publication_failure_rejected":true,"active_released":true,"actual_hot_submissions":15,
            "limits":"Synthetic publication rejection is not a real Metal command error; full-model/CLI gates remain separate"
        })
    );
    Ok(())
}
