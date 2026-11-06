use super::*;
#[cfg(all(target_os = "macos", feature = "direct-metal"))]
use metal::foreign_types::ForeignType;

#[test]
fn record_reuse_requires_quiescence_and_foreground_ownership() {
    for enabled in [false, true] {
        for depth in [0, 1, 10, usize::MAX] {
            for prefetcher in [false, true] {
                for completed in [false, true] {
                    assert_eq!(
                        record_reuse_eligible(enabled, depth, prefetcher, completed),
                        enabled && depth == 0 && !prefetcher && completed
                    );
                }
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn native_record_reuse_requires_completed_producers_and_private_foreground_slots() {
    let enabled: bool = kani::any();
    let depth: usize = kani::any();
    let prefetcher: bool = kani::any();
    let completed: bool = kani::any();
    let actual = record_reuse_eligible(enabled, depth, prefetcher, completed);
    assert_eq!(actual, enabled && depth == 0 && !prefetcher && completed);
    kani::cover!(actual);
    kani::cover!(!actual && !enabled && depth == 0 && !prefetcher && completed);
    kani::cover!(!actual && enabled && depth > 0 && !prefetcher && completed);
    kani::cover!(!actual && enabled && depth == 0 && prefetcher && completed);
    kani::cover!(!actual && enabled && depth == 0 && !prefetcher && !completed);
}

#[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
#[test]
#[ignore = "physical Metal Shared destinations: A11 reuse/EOF/fallback/recovery"]
fn physical_record_reuse_preserves_bytes_and_rejects_invalid_owners() -> Result<()> {
    check_record_reuse(false)
}
#[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
#[test]
#[ignore = "physical early Shared read path, reuse/EOF/invalid owners"]
fn physical_ar43_early_reads_preserve_bytes_and_ownership() -> Result<()> {
    check_record_reuse(true)
}
#[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
fn check_record_reuse(early: bool) -> Result<()> {
    use crate::flash_support::{DemandPlan, ExpertCache};
    use std::{cell::RefCell, io::Write, rc::Rc};
    let device = metal::Device::system_default().context("Metal device required")?;
    let mut pool = ReadService::new_with_schedule(2, false)?;
    pool.early_reads = early;
    let cache = Rc::new(RefCell::new(ExpertCache::new(10)?));
    let lease = DemandPlan::reserve(&cache, &[17])?.reads[0];
    let mut file = tempfile::tempfile()?;
    file.write_all(b"first record dataSECOND RECORD TXT")?;
    let take = |mut batch: ReadBatch| -> Result<metal::Buffer> {
        let mut complete = batch.ready(true)?;
        ensure!(complete.len() == 1);
        match complete.pop().unwrap().1 {
            ReadPayload::Shared(buffer) => Ok(buffer),
            ReadPayload::Bytes(_) => anyhow::bail!("Shared record required"),
        }
    };
    let read = |offset| -> Result<Vec<(SlotLease, File, u64, usize)>> {
        Ok(vec![(lease, file.try_clone()?, offset, 17)])
    };
    ensure!(
        pool.start_shared_reusing(read(0)?, vec![], Duration::from_secs(2), &device)
            .is_err()
    );
    ensure!(
        pool.start_shared_reusing(
            vec![(lease, file.try_clone()?, 0, 0)],
            vec![None],
            Duration::from_secs(2),
            &device
        )
        .is_err()
    );
    ensure!(pool.in_flight() == 0 && pool.fresh_destinations.get() == 0);
    let first = take(pool.start_shared(read(0)?, Duration::from_secs(2), &device)?)?;
    let original = first.as_ptr() as usize;
    let reused = take(pool.start_shared_reusing(
        read(17)?,
        vec![Some(first)],
        Duration::from_secs(2),
        &device,
    )?)?;
    ensure!(reused.as_ptr() as usize == original);
    let bytes = unsafe { std::slice::from_raw_parts(reused.contents().cast::<u8>(), 17) };
    ensure!(bytes == b"SECOND RECORD TXT");
    let mut short = pool.start_shared_reusing(
        read(100)?,
        vec![Some(reused)],
        Duration::from_secs(2),
        &device,
    )?;
    ensure!(short.ready(true).is_err());
    drop(short);
    ensure!(pool.in_flight() == 0);
    let invalid = crate::gpu::buffer(&device, 1, metal::MTLResourceOptions::StorageModeShared)?;
    let recovered = take(pool.start_shared_reusing(
        read(0)?,
        vec![Some(invalid)],
        Duration::from_secs(2),
        &device,
    )?)?;
    let bytes = unsafe { std::slice::from_raw_parts(recovered.contents().cast::<u8>(), 17) };
    ensure!(bytes == b"first record data");
    ensure!(pool.reused_destinations.get() == 2 && pool.fresh_destinations.get() == 2);
    ensure!(pool.in_flight() == 0);
    eprintln!(
        "AR10_RECORD_REUSE {{\"reused\":2,\"fresh\":2,\"bytes\":17,\"eof_recovery\":true,\"parity\":true}}"
    );
    Ok(())
}

#[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
#[test]
#[ignore = "physical real pread/allocation ordering; disposable10-record Shared microcheck"]
fn physical_ar43_first_pread_timing() -> Result<()> {
    use crate::flash_support::{DemandPlan, ExpertCache};
    use std::{cell::RefCell, io::Write, rc::Rc};
    let device = metal::Device::system_default().context("Metal device required")?;
    let bytes = EXPERT_BYTES as usize;
    let mut file = tempfile::tempfile()?;
    for i in 0..10 {
        file.write_all(&vec![i as u8 + 1; bytes])?;
    }
    let mut cases = Vec::new();
    for early in [false, true] {
        let mut pool = ReadService::new_with_schedule(4, false)?;
        pool.early_reads = early;
        let trace = Arc::new(Mutex::new(Vec::new()));
        *pool.shared_trace.borrow_mut() = Some(trace.clone());
        let cache = Rc::new(RefCell::new(ExpertCache::new(10)?));
        let plan = DemandPlan::reserve(&cache, &(0..10).collect::<Vec<_>>())?;
        let records = plan
            .reads
            .iter()
            .enumerate()
            .map(|(i, &lease)| Ok((lease, file.try_clone()?, (i * bytes) as u64, bytes)))
            .collect::<Result<Vec<_>>>()?;
        let start = Instant::now();
        let mut batch = pool.start_shared(records, Duration::from_secs(5), &device)?;
        let rows = batch.ready(true)?;
        ensure!(rows.len() == 10 && pool.in_flight() == 0);
        for (lease, payload) in rows {
            let ReadPayload::Shared(buffer) = payload else {
                anyhow::bail!("Shared destination required")
            };
            let actual =
                unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u8>(), bytes) };
            ensure!(actual.iter().all(|&v| v == lease.expert as u8 + 1));
        }
        let trace = trace.lock().unwrap();
        let allocations = trace
            .iter()
            .filter(|(k, _)| *k == "allocation")
            .map(|(_, t)| *t)
            .collect::<Vec<_>>();
        let reads = trace
            .iter()
            .filter(|(k, _)| *k == "pread")
            .map(|(_, t)| *t)
            .collect::<Vec<_>>();
        ensure!(allocations.len() == 10 && reads.len() == 10);
        let first_read = *reads.iter().min().unwrap();
        let last_allocation = *allocations.iter().max().unwrap();
        if !early {
            ensure!(first_read >= last_allocation);
        }
        cases.push(serde_json::json!({"early":early,"allocations":10,"preads":10,"first_pread_ns":first_read.duration_since(start).as_nanos(),"last_allocation_ns":last_allocation.duration_since(start).as_nanos(),"observed_overlap":first_read<last_allocation}));
    }
    eprintln!(
        "AR43_PREAD_TIMING {}",
        serde_json::json!({"parity":true,"cases":cases,"limits":"host timestamps immediately before read_exact_at, instrumented temporary-file microcheck; not physical NAND or speed proof"})
    );
    Ok(())
}
