use super::*;

#[test]
fn common_queue_policy_defaults_off_and_rejects_non_literals() {
    assert_eq!(common_queue_flag(None), Some(false));
    assert_eq!(common_queue_flag(Some("0")), Some(false));
    assert_eq!(common_queue_flag(Some("1")), Some(true));
    for value in ["", "01", "true", " 1", "1\n", "é"] {
        assert_eq!(common_queue_flag(Some(value)), None);
    }
}

#[test]
fn common_queue_free_worker_finishes_later_records_while_first_is_held() -> Result<()> {
    use crate::flash_support::{DemandPlan, ExpertCache};
    use std::{cell::RefCell, io::Write, rc::Rc};
    let pool = ReadService::new_with_schedule(2, true)?;
    let held = Arc::new(TestReadGate::default());
    *pool.held_first.borrow_mut() = Some(held.clone());
    let cache = Rc::new(RefCell::new(ExpertCache::new(10)?));
    let plan = DemandPlan::reserve(&cache, &[17, 18, 19])?;
    let mut file = tempfile::tempfile()?;
    file.write_all(&[10, 20, 30])?;
    let records = plan
        .reads
        .iter()
        .enumerate()
        .map(|(i, &lease)| Ok((lease, file.try_clone()?, i as u64, 1)))
        .collect::<Result<Vec<_>>>()?;
    let mut batch = match pool.start(records, Duration::from_secs(2)) {
        Ok(batch) => batch,
        Err(error) => {
            held.release();
            return Err(error);
        }
    };
    let middle = batch.ready_min(2);
    let first_still_owned = pool.in_flight() == 1;
    // Release and drain before assertions or propagating any fixture failure.
    held.release();
    let tail = batch.ready(true);
    drop(batch);
    let cleanup_deadline = Instant::now() + Duration::from_secs(2);
    while pool.in_flight() != 0 && Instant::now() < cleanup_deadline {
        std::thread::yield_now();
    }
    let drained = pool.in_flight() == 0;
    let count = pool.dynamic_submissions.get();
    drop(pool);
    let mut middle = middle?
        .into_iter()
        .map(|(_, bytes)| bytes.legacy_bytes().to_vec())
        .collect::<Vec<_>>();
    middle.sort();
    assert_eq!(middle, vec![vec![20], vec![30]]);
    assert!(first_still_owned);
    assert!(drained);
    let tail = tail?;
    assert_eq!(tail.len(), 1);
    assert_eq!(tail[0].1.legacy_bytes(), &[10]);
    assert_eq!(count, 3);
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn native_common_queue_flag_is_strict_and_off_without_override() {
    let bytes: [u8; 4] = kani::any();
    let length = usize::from(kani::any::<u8>() % 5);
    let present: bool = kani::any();
    if !present {
        assert_eq!(common_queue_flag(None), Some(false));
        kani::cover!(common_queue_flag(None) == Some(false));
    } else if let Ok(value) = std::str::from_utf8(&bytes[..length]) {
        let actual = common_queue_flag(Some(value));
        assert_eq!(
            actual,
            match value {
                "0" => Some(false),
                "1" => Some(true),
                _ => None,
            }
        );
        kani::cover!(value == "0" && actual == Some(false));
        kani::cover!(value == "1" && actual == Some(true));
        kani::cover!(value == "é" && actual.is_none());
    }
}
