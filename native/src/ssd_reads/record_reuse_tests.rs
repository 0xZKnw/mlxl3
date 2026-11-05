use super::*;

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
    use crate::flash_support::{DemandPlan, ExpertCache};
    use std::{cell::RefCell, io::Write, rc::Rc};
    let device = metal::Device::system_default().context("Metal device required")?;
    let pool = ReadService::new_with_schedule(2, false)?;
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
