use super::*;

#[test]
fn split_record_ranges_are_disjoint_complete_and_bounded() {
    for enabled in [false, true] {
        for records in [0, 1, 2, 10] {
            for workers in [1, 2, 4, 5] {
                for bytes in [0, 8191, 8192, 8193, EXPERT_BYTES as usize, usize::MAX] {
                    let ranges = isolated_record_ranges(enabled, records, workers, bytes, 0);
                    let valid = enabled
                        && records == 1
                        && (2..=4).contains(&workers)
                        && (8192..=EXPERT_BYTES as usize).contains(&bytes);
                    assert_eq!(ranges.is_some(), valid);
                    if let Some([[a, n], [b, m]]) = ranges {
                        assert_eq!([a, b], [0, n]);
                        assert!(n >= 4096 && m >= 4096 && n % 4096 == 0);
                        assert_eq!(n + m, bytes);
                    }
                }
            }
        }
    }
    assert!(isolated_record_ranges(true, 1, 4, 8192, u64::MAX).is_none());
    assert_eq!(claim_record_part(0, 0), Some(1));
    assert_eq!(claim_record_part(1, 1), Some(3));
    assert_eq!(claim_record_part(3, 1), None);
    assert_eq!(claim_record_part(0, usize::MAX), None);
}

#[cfg(kani)]
#[kani::proof]
fn native_isolated_record_ranges_partition_exactly_one_valid_destination() {
    let enabled: bool = kani::any();
    let records: usize = kani::any();
    let workers: usize = kani::any();
    let bytes: usize = kani::any();
    let offset: u64 = kani::any();
    let ranges = isolated_record_ranges(enabled, records, workers, bytes, offset);
    let valid = enabled
        && records == 1
        && workers >= 2
        && workers <= 4
        && bytes >= 8192
        && (bytes as u128) <= u128::from(EXPERT_BYTES)
        && u128::from(offset) + bytes as u128 <= u128::from(u64::MAX);
    assert_eq!(ranges.is_some(), valid);
    if let Some([[a, n], [b, m]]) = ranges {
        assert_eq!(a, 0);
        assert_eq!(n, b);
        assert!(n >= 4096 && m >= 4096 && n % 4096 == 0);
        assert_eq!(n as u128 + m as u128, bytes as u128);
        assert!(u128::from(offset) + b as u128 + m as u128 <= u128::from(u64::MAX));
        kani::cover!(bytes == 8192);
        kani::cover!(bytes as u64 == EXPERT_BYTES);
    }
    kani::cover!(!valid && enabled && records == 2);
    kani::cover!(!valid && enabled && workers == 1);
    kani::cover!(!valid && enabled && offset == u64::MAX);
}

#[cfg(kani)]
#[kani::proof]
fn native_split_record_claims_never_overlap_or_repeat() {
    let claimed: u8 = kani::any();
    let index: usize = kani::any();
    let next = claim_record_part(claimed, index);
    let valid = claimed <= 3
        && index < 2
        && (if index < 2 {
            claimed & (1 << index) == 0
        } else {
            false
        });
    assert_eq!(next.is_some(), valid);
    if let Some(next) = next {
        assert!(next <= 3 && next & claimed == claimed);
        assert!(claim_record_part(next, index).is_none());
    }
    kani::cover!(next == Some(1));
    kani::cover!(next == Some(2));
    kani::cover!(next == Some(3));
    kani::cover!(next.is_none() && index >= 2);
    kani::cover!(next.is_none() && index == 0 && claimed == 1);
}

#[cfg(test)]
fn record(pool: &ReadService, file: &File, bytes: usize, limit: Duration) -> Result<ReadBatch> {
    use crate::flash_support::{DemandPlan, ExpertCache};
    use std::{cell::RefCell, rc::Rc};
    let cache = Rc::new(RefCell::new(ExpertCache::new(10)?));
    let lease = DemandPlan::reserve(&cache, &[17])?.reads[0];
    pool.start_payloads(
        vec![(
            lease,
            file.try_clone()?,
            0,
            bytes,
            ReadPayload::Bytes(Vec::new()),
        )]
        .into_iter(),
        limit,
        false,
        true,
    )
}

#[test]
fn split_record_whole_bytes_eof_and_retry_use_one_logical_completion() -> Result<()> {
    use std::io::Write;
    let pool = ReadService::new_with_options(4, false, true)?;
    let mut file = tempfile::tempfile()?;
    let expected = (0..8193).map(|i| (i % 251) as u8).collect::<Vec<_>>();
    file.write_all(&expected)?;
    let mut batch = record(&pool, &file, expected.len(), Duration::from_secs(2))?;
    assert_eq!(batch.remaining(), 1);
    let result = batch.ready(true)?;
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1.legacy_bytes(), expected);
    for length in [4096, 0] {
        file.set_len(length)?;
        let mut broken = record(&pool, &file, 8193, Duration::from_secs(2))?;
        assert!(broken.ready(true).is_err());
        drop(broken);
        assert_eq!(pool.in_flight(), 0);
    }
    file.set_len(0)?;
    // set_len does not rewind the file cursor; positional writes restore offset0.
    file.write_all_at(&expected, 0)?;
    let mut retry = record(&pool, &file, expected.len(), Duration::from_secs(2))?;
    assert_eq!(retry.ready(true)?[0].1.legacy_bytes(), expected);
    assert_eq!(pool.split_records.get(), 4);
    assert_eq!(pool.split_pieces.get(), 8);
    Ok(())
}

#[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
#[test]
#[ignore = "physical Metal Shared record: A12 two ranges and one publication"]
fn physical_isolated_shared_record_uses_two_pieces_and_preserves_every_byte() -> Result<()> {
    use crate::flash_support::{DemandPlan, ExpertCache};
    use std::{cell::RefCell, io::Write, rc::Rc};
    let device = metal::Device::system_default().context("Metal device required")?;
    ensure!(device.name() == "Apple M5", "qualified M5 required");
    let pool = ReadService::new_with_options(4, false, true)?;
    let cache = Rc::new(RefCell::new(ExpertCache::new(10)?));
    let lease = DemandPlan::reserve(&cache, &[17])?.reads[0];
    let expected = (0..EXPERT_BYTES as usize)
        .map(|i| (i % 251) as u8)
        .collect::<Vec<_>>();
    let mut file = tempfile::tempfile()?;
    file.write_all(&expected)?;
    let mut batch = pool.start_shared(
        vec![(lease, file, 0, expected.len())],
        Duration::from_secs(2),
        &device,
    )?;
    let mut complete = batch.ready(true)?;
    ensure!(complete.len() == 1 && batch.remaining() == 0);
    let ReadPayload::Shared(buffer) = complete.pop().unwrap().1 else {
        anyhow::bail!("Shared expected")
    };
    let actual =
        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>(), expected.len()) };
    ensure!(actual == expected && pool.in_flight() == 0);
    ensure!(pool.split_records.get() == 1 && pool.split_pieces.get() == 2);
    eprintln!(
        "AR10_READ_SPLIT {{\"logical_records\":1,\"pieces\":2,\"bytes\":{},\"parity\":true}}",
        expected.len()
    );
    Ok(())
}

#[test]
fn split_record_timeout_and_partial_enqueue_retain_owner_until_both_jobs_end() -> Result<()> {
    use std::io::Write;
    for partial_enqueue in [false, true] {
        let mut pool = ReadService::new_with_options(4, false, true)?;
        let gates = [
            Arc::new(TestReadGate::default()),
            Arc::new(TestReadGate::default()),
        ];
        *pool.split_gates.borrow_mut() = Some(gates.clone());
        let mut file = tempfile::tempfile()?;
        file.write_all(&vec![43; 8192])?;
        if partial_enqueue {
            let (dead_sender, receiver) = mpsc::sync_channel(1);
            drop(receiver);
            let old = std::mem::replace(&mut pool.senders[1], dead_sender);
            drop(old);
        }
        let attempt = record(&pool, &file, 8192, Duration::ZERO);
        let rejected = match attempt {
            Ok(mut pending) => {
                let failed = pending.ready(true).is_err();
                drop(pending);
                failed
            }
            Err(_) => partial_enqueue,
        };
        let retained = pool.in_flight() == 1;
        let refused = pool.start(Vec::new(), Duration::ZERO).is_err();
        for gate in gates {
            gate.release();
        }
        let deadline = Instant::now() + Duration::from_secs(2);
        while pool.in_flight() != 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        let drained = pool.in_flight() == 0;
        drop(pool);
        assert!(rejected && retained && refused && drained);
    }
    Ok(())
}

#[test]
fn split_record_overlaps_both_jobs_and_never_publishes_after_one_piece() -> Result<()> {
    use std::io::Write;
    let pool = ReadService::new_with_options(4, false, true)?;
    let gates = [
        Arc::new(TestReadGate::default()),
        Arc::new(TestReadGate::default()),
    ];
    gates[1].release();
    *pool.split_gates.borrow_mut() = Some(gates.clone());
    let mut file = tempfile::tempfile()?;
    let expected = vec![53; 8193];
    file.write_all(&expected)?;
    let mut pending = record(&pool, &file, expected.len(), Duration::from_secs(2))?;
    let deadline = Instant::now() + Duration::from_secs(2);
    while (pool.split_started.load(Ordering::Acquire) != 2
        || pool.split_finished.load(Ordering::Acquire) == 0)
        && Instant::now() < deadline
    {
        std::thread::yield_now();
    }
    let both_started = pool.split_started.load(Ordering::Acquire) == 2;
    let first_finished = pool.split_finished.load(Ordering::Acquire) == 1;
    // Retain any wrongly early payload until ALL jobs joined, so an early-
    // publication mutant is rejected without freeing a still-written buffer.
    let observed = pending.poll();
    let unpublished = matches!(&observed, Ok(None));
    gates[0].release();
    let result = pending.ready(true);
    drop(pending);
    let deadline = Instant::now() + Duration::from_secs(2);
    while pool.in_flight() != 0 && Instant::now() < deadline {
        std::thread::yield_now();
    }
    let drained = pool.in_flight() == 0;
    drop(pool);
    drop(observed);
    assert!(both_started && first_finished && unpublished && drained);
    let result = result?;
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].1.legacy_bytes(), expected);
    Ok(())
}
