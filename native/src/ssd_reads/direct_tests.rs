use super::*;

#[test]
fn direct_destination_and_policy_reject_every_unqualified_boundary() {
    for bytes in [
        0,
        1,
        17,
        EXPERT_BYTES as usize,
        EXPERT_BYTES as usize + 1,
        usize::MAX,
    ] {
        for length in [0, 1, 17, EXPERT_BYTES, EXPERT_BYTES + 1, u64::MAX] {
            for shared in [false, true] {
                assert_eq!(
                    direct_destination_valid(bytes, length, shared),
                    bytes > 0
                        && (bytes as u128) <= u128::from(EXPERT_BYTES)
                        && bytes as u128 == u128::from(length)
                        && shared
                );
            }
        }
    }
    for depth in [0, 1, 10, usize::MAX] {
        for enabled in [false, true] {
            for prefetcher in [false, true] {
                assert_eq!(
                    direct_read_eligible(enabled, depth, prefetcher),
                    enabled && depth == 0 && !prefetcher
                );
            }
        }
    }
    for qualified in [false, true] {
        assert_eq!(direct_flag(None, qualified), Some(false));
        assert_eq!(direct_flag(Some("0"), qualified), Some(false));
        assert_eq!(direct_flag(Some("1"), qualified), Some(qualified));
        for value in ["", "01", "true", " 1", "1\n", "é", "\0"] {
            assert_eq!(direct_flag(Some(value), qualified), None);
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn native_direct_destination_requires_exact_shared_extent() {
    let bytes: usize = kani::any();
    let length: u64 = kani::any();
    let shared: bool = kani::any();
    let valid = direct_destination_valid(bytes, length, shared);
    assert_eq!(
        valid,
        bytes > 0
            && (bytes as u128) <= u128::from(EXPERT_BYTES)
            && bytes as u128 == u128::from(length)
            && shared
    );
    kani::cover!(valid && bytes == 1);
    kani::cover!(valid && bytes as u64 == EXPERT_BYTES);
    kani::cover!(!valid && bytes == 0);
    kani::cover!(!valid && bytes as u128 > u128::from(EXPERT_BYTES));
    kani::cover!(!valid && bytes == 1 && length == 2 && shared);
    kani::cover!(!valid && bytes == 1 && length == 1 && !shared);
}

#[cfg(kani)]
#[kani::proof]
fn native_direct_reads_require_depth_zero_without_prefetch() {
    let enabled = kani::any();
    let depth: usize = kani::any();
    let prefetcher = kani::any();
    let actual = direct_read_eligible(enabled, depth, prefetcher);
    assert_eq!(
        actual,
        if enabled && !prefetcher {
            depth == 0
        } else {
            false
        }
    );
    kani::cover!(actual);
    kani::cover!(!actual && !enabled && depth == 0 && !prefetcher);
    kani::cover!(!actual && enabled && depth > 0 && !prefetcher);
    kani::cover!(!actual && enabled && depth == 0 && prefetcher);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn native_direct_flag_preserves_off_and_qualified_exact_override() {
    let raw: [u8; 4] = kani::any();
    let len = usize::from(kani::any::<u8>() % 5);
    let qualified: bool = kani::any();
    assert_eq!(direct_flag(None, qualified), Some(false));
    if let Ok(value) = std::str::from_utf8(&raw[..len]) {
        let actual = direct_flag(Some(value), qualified);
        assert_eq!(
            actual,
            match value {
                "0" => Some(false),
                "1" => Some(qualified),
                _ => None,
            }
        );
        kani::cover!(actual == Some(true));
        kani::cover!(value == "1" && !qualified && actual == Some(false));
        kani::cover!(value == "0" && qualified && actual == Some(false));
        kani::cover!(value.is_empty() && actual.is_none());
        kani::cover!(value == "01" && actual.is_none());
    }
}

#[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
mod physical {
    use super::*;
    use crate::flash_support::{DemandPlan, ExpertCache};
    use std::{cell::RefCell, collections::HashSet, io::Write, rc::Rc, sync::Barrier};

    fn cache() -> Result<Rc<RefCell<ExpertCache>>> {
        Ok(Rc::new(RefCell::new(ExpertCache::new(10)?)))
    }
    fn shared(payload: ReadPayload, expected: &[u8]) -> Result<metal::Buffer> {
        let ReadPayload::Shared(buffer) = payload else {
            anyhow::bail!("direct read returned copied bytes")
        };
        ensure!(
            buffer.storage_mode() == metal::MTLStorageMode::Shared
                && buffer.length() == expected.len() as u64
                && !buffer.contents().is_null()
        );
        // A Shared payload is observable only after the real worker completed read_exact_at.
        let actual =
            unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>(), expected.len()) };
        ensure!(
            actual == expected,
            "owned destination bytes differ from independent fixture"
        );
        Ok(buffer)
    }
    fn drain(active: &AtomicUsize) -> Result<()> {
        let until = Instant::now() + Duration::from_secs(2);
        while active.load(Ordering::Acquire) != 0 && Instant::now() < until {
            std::thread::yield_now();
        }
        ensure!(
            active.load(Ordering::Acquire) == 0,
            "physical disk worker failed to drain"
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple Shared buffers and owned SSD workers; run alone"]
    fn shared_destinations_preserve_bytes_failures_generations_and_worker_lifetimes() -> Result<()>
    {
        let device = metal::Device::system_default().context("Apple Metal device required")?;
        let pool = ReadService::new(4)?;
        let mut file = tempfile::tempfile()?;
        let source = (0..8192)
            .map(|i| ((i * 73 + i / 11) % 251) as u8)
            .collect::<Vec<_>>();
        file.write_all(&source)?;
        let cache = cache()?;
        let plan = DemandPlan::reserve(&cache, &[17, 18, 19, 20])?;
        let ranges = [(0, 1), (3, 17), (4093, 67), (1024, 4096)];
        let reads = plan
            .reads
            .iter()
            .zip(ranges)
            .map(|(&lease, (at, n))| Ok((lease, file.try_clone()?, at, n)))
            .collect::<Result<Vec<_>>>()?;
        let mut batch = pool.start_shared(reads, Duration::from_secs(2), &device)?;
        let completed = batch.ready(true)?;
        ensure!(completed.len() == ranges.len() && batch.remaining() == 0);
        let mut backings = Vec::new();
        for (lease, payload) in completed {
            let which = plan
                .reads
                .iter()
                .position(|&l| l == lease)
                .context("unknown completion lease")?;
            let (at, n) = ranges[which];
            backings.push(shared(payload, &source[at as usize..at as usize + n])?);
            ensure!(plan.publish(lease));
        }
        ensure!(
            backings
                .iter()
                .map(|b| b.contents() as usize)
                .collect::<HashSet<_>>()
                .len()
                == ranges.len(),
            "simultaneously owned destinations alias"
        );
        for (i, left) in backings.iter().enumerate() {
            for right in &backings[i + 1..] {
                let a = left.contents() as usize;
                let b = right.contents() as usize;
                let ae = a
                    .checked_add(left.length() as usize)
                    .context("left range overflow")?;
                let be = b
                    .checked_add(right.length() as usize)
                    .context("right range overflow")?;
                ensure!(ae <= b || be <= a, "owned read extents overlap");
            }
        }
        ensure!(pool.in_flight() == 0);
        drop(backings);
        drop(batch);
        drop(plan);

        let token = DemandPlan::reserve(&cache, &[21])?.reads[0];
        for bytes in [0, EXPERT_BYTES as usize + 1, usize::MAX] {
            ensure!(
                pool.start_shared(
                    vec![(token, file.try_clone()?, 0, bytes)],
                    Duration::ZERO,
                    &device
                )
                .is_err()
            );
            ensure!(pool.in_flight() == 0);
        }
        let mut eof = pool.start_shared(
            vec![(token, file.try_clone()?, 8190, 8)],
            Duration::from_secs(2),
            &device,
        )?;
        let error = eof
            .ready(true)
            .err()
            .context("partial EOF returned a visible destination")?;
        ensure!(
            error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::UnexpectedEof)
        );
        drop(eof);
        ensure!(pool.in_flight() == 0);

        let mut truncated = tempfile::tempfile()?;
        truncated.write_all(&source[..32])?;
        let gate = Arc::new(Barrier::new(2));
        pool.set_test_gate(Some(gate));
        let mut pending = pool.start_shared(
            vec![(token, truncated.try_clone()?, 4, 16)],
            Duration::from_secs(2),
            &device,
        )?;
        let truncation = truncated.set_len(11);
        pool.release_test_gate();
        truncation?;
        ensure!(pending.ready(true).is_err(), "truncated record published");
        drop(pending);
        ensure!(pool.in_flight() == 0);

        let gate = Arc::new(Barrier::new(2));
        pool.set_test_gate(Some(gate));
        let mut pending = pool.start_shared(
            vec![(token, file.try_clone()?, 3, 17)],
            Duration::ZERO,
            &device,
        )?;
        let timed_out = pending.ready(true).is_err();
        drop(pending);
        let retained = pool.in_flight() == 1;
        let reuse_refused = pool
            .start_shared(Vec::new(), Duration::ZERO, &device)
            .is_err();
        pool.release_test_gate();
        drain(&pool.active)?;
        ensure!(timed_out && retained && reuse_refused);

        let gate = Arc::new(Barrier::new(2));
        pool.set_test_gate(Some(gate));
        let old = DemandPlan::reserve(&cache, &[22])?;
        let old_lease = old.reads[0];
        let mut pending = pool.start_shared(
            vec![(old_lease, file.try_clone()?, 0, 17)],
            Duration::from_secs(2),
            &device,
        )?;
        drop(old);
        let current = DemandPlan::reserve(&cache, &[22])?;
        let current_lease = current.reads[0];
        let newer = old_lease != current_lease && old_lease.slot == current_lease.slot;
        pool.release_test_gate();
        let completed = pending.ready(true)?;
        ensure!(completed.len() == 1 && completed[0].0 == old_lease && newer);
        let (_, payload) = completed.into_iter().next().unwrap();
        let stale = shared(payload, &source[..17])?;
        ensure!(!current.publish(old_lease) && !cache.borrow().contains(22));
        let mut fresh = pool.start_shared(
            vec![(current_lease, file.try_clone()?, 32, 17)],
            Duration::from_secs(2),
            &device,
        )?;
        let completed = fresh.ready(true)?;
        ensure!(completed.len() == 1 && completed[0].0 == current_lease);
        let (_, payload) = completed.into_iter().next().unwrap();
        let replacement = shared(payload, &source[32..49])?;
        ensure!(stale.contents() != replacement.contents() && current.publish(current_lease));
        ensure!(!current.publish(old_lease) && cache.borrow().contains(22));
        drop((pending, fresh));

        let retiring = ReadService::new(1)?;
        let gate = Arc::new(Barrier::new(2));
        retiring.set_test_gate(Some(gate.clone()));
        let pending = retiring.start_shared(
            vec![(token, file.try_clone()?, 0, 17)],
            Duration::ZERO,
            &device,
        )?;
        let active = retiring.active.clone();
        drop(pending);
        let retained = retiring.in_flight() == 1;
        drop(retiring);
        gate.wait();
        drain(&active)?;
        ensure!(retained);

        let directory = tempfile::tempdir()?;
        let path = directory.path().join("owned-record.safetensors");
        let bytes = EXPERT_BYTES as usize;
        let header = serde_json::to_vec(
            &serde_json::json!({"first":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"rest":{"dtype":"U8","shape":[bytes-4],"data_offsets":[4,bytes]}}),
        )?;
        let expected = (0..bytes)
            .map(|i| ((i * 37 + i / 13) % 251) as u8)
            .collect::<Vec<_>>();
        let mut record_file = File::create(&path)?;
        record_file.write_all(&(header.len() as u64).to_le_bytes())?;
        record_file.write_all(&header)?;
        record_file.write_all(&expected)?;
        let record = crate::checkpoint::read_header(&path)?["first"].clone();
        let mut batch = pool.submit_shared([(token, record.clone())], &device)?;
        let completed = batch.ready(true)?;
        ensure!(completed.len() == 1);
        shared(completed.into_iter().next().unwrap().1, &expected)?;
        record_file.set_len((8 + header.len() + bytes - 1) as u64)?;
        ensure!(pool.submit_shared([(token, record)], &device).is_err() && pool.in_flight() == 0);
        eprintln!(
            "NATIVE_SSD_DIRECT_IO {}",
            serde_json::json!({"range_cases":4,"distinct_live_backings":4,"exact_record_bytes":EXPERT_BYTES,"partial_eof":true,"post_enqueue_truncation":true,"timeout_retained":true,"stale_generation_rejected":true,"batch_and_service_drop_drained":true,"parity":true})
        );
        Ok(())
    }
}
