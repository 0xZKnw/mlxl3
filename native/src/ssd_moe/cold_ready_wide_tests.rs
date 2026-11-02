use super::*;

#[cfg(kani)]
#[kani::proof]
fn actual_cold_ready_wide_refinement_entry() {
    let failures: bool = kani::any();
    assert!(
        if failures {
            wide_cold_failures_invalidate_and_recover()
        } else {
            wide_cold_matches_external_rows_and_fallbacks()
        }
        .is_ok()
    );
    kani::cover!(failures);
    kani::cover!(!failures);
}

fn rows(inputs: &[Array], pattern: &[usize]) -> Result<Array> {
    Array::concatenate(&pattern.iter().map(|&i| &inputs[i]).collect::<Vec<_>>(), 0)
}

fn oracle(expected: &[Vec<u8>], pattern: &[usize]) -> Vec<u8> {
    pattern
        .iter()
        .flat_map(|&i| expected[i].iter().copied())
        .collect()
}

fn selected(moe: &SsdMoe, x: &Array) -> Result<Vec<u32>> {
    (0..x.shape()[0])
        .map(|row| moe.route(&x.slice(0, row, row + 1)?)?.0.to_u32())
        .collect::<Result<Vec<_>>>()
        .map(|ids| ids.concat())
}

#[test]
#[ignore = "physical Apple GPU and external Flash fixtures; select alone"]
fn wide_cold_matches_external_rows_and_fallbacks() -> Result<()> {
    crate::array::set_default_cache_limit(64 << 20)?;
    let (checkpoint, inputs, expected) = fixture()?;
    let patterns: &[&[usize]] = &[
        &[0, 1, 2],
        &[2, 0, 1],
        &[0, 0, 0],
        &[0, 1, 0],
        &[0, 1, 2, 0],
        &[2, 0, 1, 2],
        &[0, 0, 0, 0],
        &[0, 0, 1, 1],
    ];
    let mut checked = 0;
    for maximum in [2, 4] {
        for priority in [false, true] {
            let mut moe = load(&checkpoint, 53)?;
            moe.cold_ready_rows = maximum;
            moe.cold_priority = priority;
            for &pattern in patterns {
                reset(&mut moe)?;
                let x = rows(&inputs, pattern)?;
                let ids = selected(&moe, &x)?;
                let residents = ids[..10].iter().step_by(2).copied().collect::<Vec<_>>();
                prime(&mut moe, &residents)?;
                let missing = ids
                    .iter()
                    .filter(|id| !residents.contains(id))
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .len();
                ensure!(missing >= 3);
                let before = (
                    moe.cold_ready_waves.len(),
                    moe.read_bytes,
                    moe.cold_priority_batches,
                );
                if maximum == 4 {
                    crate::array::hold_read_batch_job_for_test(Some(0));
                }
                check(
                    &moe.forward_draft_rows(&x)?,
                    &oracle(&expected, pattern),
                    pattern.len(),
                )?;
                ensure!(moe.read_bytes - before.1 == missing as u64 * EXPERT_BYTES);
                ensure!(moe.cold_ready_waves.len() == before.0 + usize::from(maximum == 4));
                ensure!(
                    moe.cold_priority_batches == before.2,
                    "B3/B4 activated B2 priority"
                );
                let hot_before = (moe.cold_ready_waves.len(), moe.read_bytes);
                check(
                    &moe.forward_draft_rows(&x)?,
                    &oracle(&expected, pattern),
                    pattern.len(),
                )?;
                ensure!(hot_before == (moe.cold_ready_waves.len(), moe.read_bytes));
                checked += 1;
            }
            moe.drain()?;
        }
    }
    // An empty resident map uses the stock all-Cold path even with the wide option.
    let mut moe = load(&checkpoint, 53)?;
    moe.cold_ready_rows = 4;
    for pattern in [&[0, 1, 2][..], &[2, 1, 0, 2]] {
        reset(&mut moe)?;
        let before = moe.cold_ready_waves.len();
        check(
            &moe.forward_draft_rows(&rows(&inputs, pattern)?)?,
            &oracle(&expected, pattern),
            pattern.len(),
        )?;
        ensure!(moe.cold_ready_waves.len() == before);
    }
    moe.drain()?;
    // A small cache must preserve every row through cohort splitting and slot reuse.
    let mut small = load(&checkpoint, 10)?;
    small.cold_ready_rows = 4;
    small.unique_cohorts = true;
    small.cold_ready_verify_overlap = false;
    for pattern in [&[0, 1, 2][..], &[2, 1, 0, 2]] {
        let x = rows(&inputs, pattern)?;
        ensure!(
            selected(&small, &x)?
                .into_iter()
                .collect::<BTreeSet<_>>()
                .len()
                > 10
        );
        let before = small.cohort_sizes.len();
        check(
            &small.forward_draft_rows(&x)?,
            &oracle(&expected, pattern),
            pattern.len(),
        )?;
        let cohorts = &small.cohort_sizes[before..];
        ensure!(cohorts.len() > 1 && cohorts.iter().sum::<usize>() == pattern.len());
    }
    small.drain()?;
    ensure!(checked == 32);
    println!("COLD_READY_WIDE mixed=32 hot=32 empty=2 small_cohorts=2 parity=byte_exact");
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and external Flash fixtures; select alone"]
fn wide_cold_failures_invalidate_and_recover() -> Result<()> {
    crate::array::set_default_cache_limit(64 << 20)?;
    let (checkpoint, inputs, expected) = fixture()?;
    let mut moe = load(&checkpoint, 53)?;
    moe.cold_ready_rows = 4;
    let ids = moe.route(&inputs[0])?.0.to_u32()?;
    for count in [3, 4] {
        let pattern = vec![0; count];
        let x = rows(&inputs, &pattern)?;
        for after_prepare in [false, true] {
            reset(&mut moe)?;
            prime(&mut moe, &ids[..5])?;
            moe.cold_ready_fail_after_wave = !after_prepare;
            moe.cold_ready_fail_after_prepare = after_prepare;
            crate::array::hold_read_batch_job_for_test(Some(0));
            let error = moe
                .forward_draft_rows(&x)
                .err()
                .context("wide injected failure accepted")?;
            let message = if after_prepare {
                "injected Cold preparation failure"
            } else {
                "injected Cold wave failure"
            };
            ensure!(error.chain().any(|e| e.to_string().contains(message)));
            ensure!(ids.iter().all(|&id| !moe.cache.borrow().contains(id)));
            ensure!(
                moe.last
                    .as_ref()
                    .context("missing Hot lease")?
                    .is_available_for_test()?
            );
            moe.cold_ready_fail_after_wave = false;
            moe.cold_ready_fail_after_prepare = false;
            prime(&mut moe, &ids[..5])?;
            crate::array::hold_read_batch_job_for_test(Some(0));
            check(
                &moe.forward_draft_rows(&x)?,
                &oracle(&expected, &pattern),
                count,
            )?;
        }
    }
    reset(&mut moe)?;
    check(&moe.forward(&inputs[1])?, &expected[1], 1)?;
    moe.drain()?;
    println!("COLD_READY_WIDE_FAILURE failures=4 recoveries=4 reset_b1=1 parity=byte_exact");
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and external Flash fixtures; select alone"]
fn cold_priority_batches_preserve_outputs_and_report_launch_order() -> Result<()> {
    let (checkpoint, inputs, expected) = fixture()?;
    let mut moe = load(&checkpoint, 53)?;
    moe.cold_priority = true;
    let pattern = [0, 1];
    let x = rows(&inputs, &pattern)?;
    let ids = selected(&moe, &x)?;
    let residents = ids[..10].iter().step_by(2).copied().collect::<Vec<_>>();
    prime(&mut moe, &residents)?;
    let before = moe.cold_priority_batches;
    let _retention = crate::flash_profile::StageRetention::new(false);
    crate::array::hold_read_batch_job_for_test(Some(0));
    crate::flash_profile::start(false)?;
    let result = moe.forward_draft_rows(&x);
    let events = serde_json::to_value(crate::flash_profile::finish()?)?;
    check(&result?, &oracle(&expected, &pattern), 2)?;
    ensure!(moe.cold_priority_batches == before + 1);
    let batches = events
        .as_array()
        .context("missing priority events")?
        .iter()
        .filter(|event| event["name"] == "cold_ready_batch")
        .collect::<Vec<_>>();
    ensure!(batches.len() == 1);
    let data = &batches[0]["data"];
    ensure!(data["status"] == "finished" && data["route_rows"] == 2 && data["priority"] == true);
    let multiplicities = data["multiplicities"]
        .as_array()
        .context("missing multiplicities")?;
    let jobs = data["jobs_mapping"].as_array().context("missing jobs")?;
    ensure!(jobs.len() >= 3 && jobs.len() == multiplicities.len());
    let demanded = ids
        .iter()
        .filter(|id| !residents.contains(id))
        .copied()
        .collect::<BTreeSet<_>>();
    let actual_ids = jobs
        .iter()
        .map(|job| {
            job[0]
                .as_u64()
                .and_then(|id| u32::try_from(id).ok())
                .context("invalid job id")
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        actual_ids.len() == demanded.len()
            && actual_ids.iter().copied().collect::<BTreeSet<_>>() == demanded
    );
    let occurrences = |id: u32| ids.iter().filter(|&&used| used == id).count();
    for (id, count) in actual_ids.iter().zip(multiplicities) {
        ensure!(count.as_u64() == Some(occurrences(*id) as u64));
    }
    for pair in actual_ids.windows(2) {
        let left = occurrences(pair[0]);
        let right = occurrences(pair[1]);
        ensure!(
            left > right
                || (left == right
                    && ids.iter().position(|&id| id == pair[0])
                        < ids.iter().position(|&id| id == pair[1]))
        );
    }
    ensure!(
        data["cold_occurrences"]
            .as_array()
            .context("missing occurrences")?
            .len()
            == ids.iter().filter(|id| !residents.contains(id)).count()
    );
    moe.drain()?;
    Ok(())
}

#[test]
#[ignore = "set COLD_READY_ROWS=3 and select alone; no checkpoint or GPU"]
fn invalid_wide_cold_option_fails_before_weights() -> Result<()> {
    ensure!(std::env::var("MLXL3_SSD_COLD_READY_ROWS").as_deref() == Ok("3"));
    invalid_option("MLXL3_SSD_COLD_READY_ROWS must be 2 or 4")
}

#[test]
#[ignore = "set COLD_PRIORITY=2 and select alone; no checkpoint or GPU"]
fn invalid_cold_priority_option_fails_before_weights() -> Result<()> {
    ensure!(std::env::var("MLXL3_SSD_COLD_PRIORITY").as_deref() == Ok("2"));
    invalid_option("MLXL3_SSD_COLD_PRIORITY must be 0 or 1")
}

fn invalid_option(message: &str) -> Result<()> {
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
        .context("invalid Cold option accepted")?;
    ensure!(error.to_string() == message);
    Ok(())
}
