use super::*;

type Fixture = (Checkpoint, Vec<Array>, Vec<Vec<u8>>);

#[cfg(kani)]
#[kani::proof]
fn actual_cold_ready_refinement_entry() {
    let failure_paths: bool = kani::any();
    let result = if failure_paths {
        ready_cold_preparation_preserves_completed_fenced_and_failure_paths()
    } else {
        ready_cold_waves_preserve_external_rows_duplicates_and_fallbacks()
    };
    assert!(result.is_ok());
    kani::cover!(failure_paths);
    kani::cover!(!failure_paths);
}

fn fixture() -> Result<Fixture> {
    use std::{fs::File, path::Path};
    let root = std::env::var("MLXL3_FLASH_REFERENCE")?;
    let root = Path::new(&root);
    let reference: serde_json::Value =
        serde_json::from_reader(File::open(root.join("reference.json"))?)?;
    ensure!(reference["status"] == "complete" && reference["mode"] == "eval");
    let checkpoint = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for step in [0, 1, 4] {
        let entry = &reference["steps"][step]["components"][0]["mlp"];
        let read = |kind: &str| -> Result<Vec<u8>> {
            ensure!(entry[kind]["count"] == H, "invalid external fixture count");
            let bytes =
                std::fs::read(root.join(entry[kind]["file"].as_str().context("missing fixture")?))?;
            ensure!(
                bytes.len() == H as usize * 4
                    && bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|&v| f32::from_le_bytes(v).is_finite()),
                "invalid external fixture values"
            );
            Ok(bytes)
        };
        inputs.push(Array::from_bytes(&read("input")?, &[1, H], Dtype::Float32)?);
        outputs.push(read("output")?);
    }
    Ok((checkpoint, inputs, outputs))
}

fn load(checkpoint: &Checkpoint, capacity: usize) -> Result<SsdMoe> {
    let mut moe = SsdMoe::load(
        checkpoint,
        "model.language_model.layers.0.mlp",
        capacity,
        true,
    )?;
    ensure!(
        moe.cold_ready && !moe.shared_exl3 && !moe.weighted_epilogue,
        "test requires COLD_READY=1, SHARED_EXL3=0, WEIGHTED_EPILOGUE=0"
    );
    moe.readers = 4;
    moe.gpu_hot = true;
    moe.gpu_hot_wide = true;
    moe.cold_ready_verify_overlap = true;
    Ok(moe)
}

fn reset(moe: &mut SsdMoe) -> Result<()> {
    moe.drain()?;
    moe.cache.borrow_mut().invalidate();
    Ok(())
}

fn prime(moe: &mut SsdMoe, ids: &[u32]) -> Result<()> {
    let routes = moe.cache.borrow_mut().reserve(ids)?;
    ensure!(routes.iter().all(|&(_, hit)| !hit));
    let jobs = ids
        .iter()
        .zip(routes)
        .map(|(&id, (slot, _))| (id, slot))
        .collect::<Vec<_>>();
    moe.fill_many(&jobs)
}

fn check(actual: &Array, expected: &[u8], rows: usize) -> Result<()> {
    ensure!(actual.shape() == [rows as i32, H]);
    let values = actual.to_f32()?;
    ensure!(
        values.len() == rows * H as usize && values.iter().all(|v| v.is_finite()),
        "invalid Cold output"
    );
    ensure!(
        expected.len() == rows * H as usize * 4 && actual.to_bytes()? == expected,
        "Cold output differs from canonical oracle"
    );
    Ok(())
}

#[test]
#[ignore = "select alone with COLD_READY=invalid; exercises loader without a model or GPU"]
fn ready_cold_invalid_flag_fails_before_loading_weights() -> Result<()> {
    ensure!(std::env::var("MLXL3_SSD_COLD_READY").as_deref() == Ok("invalid"));
    let checkpoint = Checkpoint {
        path: Default::default(),
        model_type: String::new(),
        bits: None,
        size_bytes: 0,
        modules: Vec::new(),
        tensors: Default::default(),
    };
    let error = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 53, true)
        .err()
        .context("invalid Cold flag accepted")?;
    ensure!(error.to_string() == "MLXL3_SSD_COLD_READY must be 0 or 1");
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and external Flash fixtures; COLD_READY=1; select alone"]
fn ready_cold_waves_preserve_external_rows_duplicates_and_fallbacks() -> Result<()> {
    crate::array::set_default_cache_limit(512 << 20)?;
    let (checkpoint, inputs, expected) = fixture()?;
    let mut mixed = 0;
    let mut hot_controls = 0;
    for capacity in [20, 53] {
        let mut moe = load(&checkpoint, capacity)?;
        for gpu_hot in [false, true] {
            moe.gpu_hot = gpu_hot;
            for pair in [&[0][..], &[0, 1], &[1, 0], &[0, 0], &[1, 2]] {
                reset(&mut moe)?;
                let x =
                    Array::concatenate(&pair.iter().map(|&i| &inputs[i]).collect::<Vec<_>>(), 0)?;
                let ids = (0..pair.len())
                    .map(|row| {
                        moe.route(&x.slice(0, row as i32, row as i32 + 1)?)?
                            .0
                            .to_u32()
                    })
                    .collect::<Result<Vec<_>>>()?
                    .concat();
                let residents = ids[..10].iter().step_by(2).copied().collect::<Vec<_>>();
                prime(&mut moe, &residents)?;
                let missing = ids
                    .iter()
                    .filter(|id| !residents.contains(id))
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .len();
                ensure!(missing >= 3);
                let oracle = pair
                    .iter()
                    .flat_map(|&i| expected[i].iter().copied())
                    .collect::<Vec<_>>();
                let before = (moe.cold_ready_waves.len(), moe.read_bytes);
                let overlaps = moe.cold_ready_verified_overlap;
                let prepared = moe.cold_ready_prepared_rest;
                // Holding job zero forces a non-prefix first wave without timing assumptions.
                crate::array::hold_read_batch_job_for_test(Some(0));
                let output = moe.forward_many(&x)?;
                check(&output, &oracle, pair.len())?;
                let dynamic = std::env::var("MLXL3_SSD_DYNAMIC_READERS").as_deref() != Ok("0")
                    && missing > moe.readers;
                ensure!(
                    crate::array::disk_profile_schedule()
                        == if dynamic { "dynamic" } else { "static" },
                    "Cold batch did not use the requested effective schedule"
                );
                ensure!(
                    moe.cold_ready_waves.len() == before.0 + 1
                        && moe.cold_ready_verified_overlap == overlaps + 1,
                    "Cold split or compute-before-tail was not exercised"
                );
                ensure!(
                    moe.cold_ready_prepared_rest == prepared + 1,
                    "W2 was not prepared before join"
                );
                ensure!(moe.read_bytes - before.1 == missing as u64 * EXPERT_BYTES);
                mixed += 1;
                let before = (moe.cold_ready_waves.len(), moe.read_bytes);
                check(&moe.forward_many(&x)?, &oracle, pair.len())?;
                ensure!(
                    before == (moe.cold_ready_waves.len(), moe.read_bytes),
                    "all-Hot control submitted another Cold wave or read"
                );
                hot_controls += 1;
            }
        }
        reset(&mut moe)?;
        moe.gpu_hot = true;
        let waves = moe.cold_ready_waves.len();
        check(&moe.forward(&inputs[0])?, &expected[0], 1)?;
        ensure!(
            moe.cold_ready_waves.len() == waves,
            "empty cache must use stock"
        );
        for missing in [1, 2] {
            reset(&mut moe)?;
            let ids = moe.route(&inputs[0])?.0.to_u32()?;
            prime(&mut moe, &ids[..10 - missing])?;
            check(&moe.forward(&inputs[0])?, &expected[0], 1)?;
            ensure!(
                moe.cold_ready_waves.len() == waves,
                "small lot must use stock"
            );
        }
        moe.drain()?;
    }
    ensure!(mixed == 20 && hot_controls == 20);

    let mut moe = load(&checkpoint, 53)?;
    let ids = moe.route(&inputs[0])?.0.to_u32()?;
    let scaled = inputs[0].scalar_mul(0.5)?;
    ensure!(
        scaled.to_bytes()? != inputs[0].to_bytes()? && moe.route(&scaled)?.0.to_u32()? == ids,
        "distinct-activation duplicate fixture is not valid"
    );
    moe.cold_ready = false;
    let second = moe.forward(&scaled)?.to_bytes()?;
    ensure!(second != expected[0]);
    reset(&mut moe)?;
    moe.cold_ready = true;
    prime(&mut moe, &ids[..5])?;
    let x = Array::concatenate(&[&inputs[0], &scaled], 0)?;
    let mut oracle = expected[0].clone();
    oracle.extend(&second);
    crate::array::hold_read_batch_job_for_test(Some(0));
    check(&moe.forward_many(&x)?, &oracle, 2)?;
    ensure!(
        moe.cold_ready_waves.len() == 1 && moe.cold_ready_verified_overlap == 1,
        "duplicate slots did not enter a two-row Cold wave"
    );
    ensure!(
        oracle != expected[0].repeat(2),
        "duplicate-output mutation escaped oracle"
    );

    reset(&mut moe)?;
    prime(&mut moe, &ids[..5])?;
    let x = Array::concatenate(&[&inputs[0], &inputs[1], &inputs[2]], 0)?;
    let oracle = expected.iter().flatten().copied().collect::<Vec<_>>();
    let waves = moe.cold_ready_waves.len();
    check(&moe.forward_draft_rows(&x)?, &oracle, 3)?;
    ensure!(
        moe.cold_ready_waves.len() == waves,
        "B3 must retain stock dispatch"
    );
    moe.drain()?;
    println!(
        "COLD_READY_QUALITY mixed=20 all_hot=20 empty_cache=2 small_lots=4 distinct_activation_duplicate=1 wide_fallback=1 parity=byte_exact overlap=verified_before_held_tail"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and external Flash fixtures; COLD_READY=1; select alone"]
fn ready_cold_preparation_preserves_completed_fenced_and_failure_paths() -> Result<()> {
    crate::array::set_default_cache_limit(512 << 20)?;
    let (checkpoint, inputs, expected) = fixture()?;
    let mut moe = load(&checkpoint, 53)?;
    let ids = moe.route(&inputs[0])?.0.to_u32()?;
    let mut cases = 0;
    for rows in [1, 2] {
        let x = if rows == 1 {
            inputs[0].try_clone()?
        } else {
            Array::concatenate(&[&inputs[0], &inputs[0]], 0)?
        };
        for fenced in [false, true] {
            reset(&mut moe)?;
            prime(&mut moe, &ids[..5])?;
            moe.cold_ready_complete_before_submit = !fenced;
            moe.cold_ready_verify_overlap = fenced;
            let before = moe.cold_ready_prepared_rest;
            crate::array::hold_read_batch_job_for_test(Some(0));
            crate::flash_profile::start(fenced)?;
            let output = moe.forward_many(&x);
            let events = serde_json::to_value(crate::flash_profile::finish()?)?;
            let events = events.as_array().context("missing events")?;
            check(&output?, &expected[0].repeat(rows), rows)?;
            if fenced {
                ensure!(
                    moe.cold_ready_prepared_rest == before,
                    "fenced W2 prepared before join"
                );
            } else {
                ensure!(moe.cold_ready_prepared_rest == before + 1);
                let mut batches = 0;
                let mut cold_submits = 0;
                for (i, event) in events.iter().enumerate() {
                    if event["name"] == "cold_ready_batch" {
                        batches += 1;
                    }
                    if event["name"] != "host_async_submit" {
                        continue;
                    }
                    let mut parent = events[i]["parent"].as_u64();
                    while let Some(index) = parent {
                        if events[index as usize]["name"] == "cold_ready_batch" {
                            cold_submits += 1;
                            break;
                        }
                        parent = events[index as usize]["parent"].as_u64();
                    }
                }
                ensure!(batches == 1);
                ensure!(cold_submits == 1, "unexpected Cold submissions");
            }
            cases += 1;
        }
    }
    {
        reset(&mut moe)?;
        prime(&mut moe, &ids[..5])?;
        moe.cold_ready_complete_before_submit = false;
        moe.cold_ready_verify_overlap = true;
        moe.cold_ready_fail_after_prepare = true;
        let prepared = moe.cold_ready_prepared_rest;
        crate::array::hold_read_batch_job_for_test(Some(0));
        let error = moe
            .forward(&inputs[0])
            .err()
            .context("preparation failure accepted")?;
        ensure!(
            error
                .chain()
                .any(|e| e.to_string().contains("injected Cold preparation failure"))
        );
        ensure!(moe.cold_ready_prepared_rest == prepared + 1);
        ensure!(ids.iter().all(|&id| !moe.cache.borrow().contains(id)));
        ensure!(
            moe.last
                .as_ref()
                .context("missing Hot lease")?
                .is_available_for_test()?
        );
        moe.cold_ready_fail_after_prepare = false;
        prime(&mut moe, &ids[..5])?;
        crate::array::hold_read_batch_job_for_test(Some(0));
        check(&moe.forward(&inputs[0])?, &expected[0], 1)?;
    }
    moe.drain()?;
    ensure!(cases == 4);
    println!(
        "COLD_READY_PREPARATION completed=2 fenced=2 prepared_failure=1 recovery=1 parity=byte_exact"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and external Flash fixtures; COLD_READY=1; select alone"]
fn ready_cold_failure_drains_consumers_invalidates_reservations_and_recovers() -> Result<()> {
    crate::array::set_default_cache_limit(512 << 20)?;
    let (checkpoint, inputs, expected) = fixture()?;
    let mut moe = load(&checkpoint, 53)?;
    let ids = moe.route(&inputs[0])?.0.to_u32()?;
    prime(&mut moe, &ids[..5])?;
    moe.cold_ready_verify_overlap = false;
    moe.cold_ready_fail_after_wave = true;
    moe.slow_hot_for_test = true;
    crate::array::hold_read_batch_job_for_test(Some(0));
    let error = moe
        .forward(&inputs[0])
        .err()
        .context("injected failure succeeded")?;
    ensure!(
        error
            .chain()
            .any(|error| error.to_string().contains("injected Cold wave failure"))
    );
    ensure!(
        moe.cold_ready_waves.len() == 1,
        "failure preceded W1 submission"
    );
    ensure!(ids.iter().all(|&id| !moe.cache.borrow().contains(id)));
    ensure!(
        moe.last
            .as_ref()
            .context("missing Hot lease")?
            .is_available_for_test()?,
        "error returned with Hot consumer still pending"
    );
    moe.cold_ready_fail_after_wave = false;
    moe.slow_hot_for_test = false;
    moe.cold_ready_verify_overlap = true;
    prime(&mut moe, &ids[..5])?;
    crate::array::hold_read_batch_job_for_test(Some(0));
    check(&moe.forward(&inputs[0])?, &expected[0], 1)?;
    ensure!(moe.cold_ready_waves.len() == 2 && moe.cold_ready_verified_overlap == 1);
    reset(&mut moe)?;
    check(&moe.forward(&inputs[1])?, &expected[1], 1)?;
    moe.drain()?;
    println!(
        "COLD_READY_FAILURE submitted_wave=1 reservations_invalidated=true recovery_and_reset=byte_exact"
    );
    Ok(())
}
