use super::*;

#[test]
fn extra_wave_policy_keeps_the_threshold_cap_and_strict_hardware_flag() {
    for remaining in [0, 1, 2, 3, 4, 10, usize::MAX] {
        for enabled in [false, true] {
            for used in [false, true] {
                assert_eq!(
                    extra_wave(enabled, remaining, used),
                    enabled && !used && remaining > 2
                );
            }
        }
    }
    for qualified in [false, true] {
        assert_eq!(extra_wave_flag(None, qualified), Some(false));
        assert_eq!(extra_wave_flag(Some("0"), qualified), Some(false));
        assert_eq!(extra_wave_flag(Some("1"), qualified), Some(qualified));
        for invalid in ["", "01", "true", " 1", "1\n", "é", "\0"] {
            assert_eq!(extra_wave_flag(Some(invalid), qualified), None);
        }
    }
}

#[test]
fn ready_min_waits_for_two_then_returns_before_the_held_tail() -> Result<()> {
    let cache = std::rc::Rc::new(std::cell::RefCell::new(
        crate::flash_support::ExpertCache::new(10)?,
    ));
    let leases = crate::flash_support::DemandPlan::reserve(&cache, &[1, 2, 3])?
        .reads
        .clone();
    let (sender, receiver) = mpsc::channel();
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut batch = ReadBatch {
        io_cancelled: Arc::new(AtomicBool::new(false)),
        receiver,
        cancelled: Arc::new(AtomicBool::new(false)),
        draining: Arc::new(AtomicBool::new(false)),
        abandoned: false,
        remaining: 3,
        deadline,
        trace: Vec::new(),
        #[cfg(test)]
        wait_started: None,
    };
    sender.send(Completed::new(leases[0], Ok(vec![17])))?;
    let (started, entered) = mpsc::channel();
    let (result, finished) = mpsc::channel();
    let thread = std::thread::spawn(move || {
        let _ = started.send(());
        let values = batch.ready_min(2);
        let _ = result.send((values, batch));
    });
    let outcome = (|| -> Result<_> {
        entered.recv_timeout(Duration::from_secs(1))?;
        let premature = finished.recv_timeout(Duration::from_millis(100));
        // The second completion is explicitly withheld until the first minimum wait.
        sender.send(Completed::new(leases[1], Ok(vec![23])))?;
        let (values, mut batch) = match premature {
            Ok(value) => value,
            Err(mpsc::RecvTimeoutError::Timeout) => {
                finished.recv_timeout(Duration::from_secs(1))?
            }
            Err(error) => return Err(error.into()),
        };
        let was_premature = batch.remaining != 1;
        sender.send(Completed::new(leases[2], Ok(vec![31])))?;
        let values = values?;
        let tail = batch.ready(true)?;
        Ok((was_premature, values, tail, batch.deadline))
    })();
    // Disconnect even on an early error, then join before inspecting the oracle.
    drop(sender);
    thread
        .join()
        .map_err(|_| anyhow::anyhow!("minimum worker panicked"))?;
    let (was_premature, values, tail, actual_deadline) = outcome?;
    assert!(!was_premature, "minimum returned before two completions");
    assert_eq!(values.len(), 2);
    assert_eq!(
        values
            .iter()
            .map(|(_, v)| v.legacy_bytes()[0])
            .collect::<Vec<_>>(),
        [17, 23]
    );
    assert_eq!(tail.len(), 1);
    assert_eq!(tail[0].1.legacy_bytes(), [31]);
    assert_eq!(actual_deadline, deadline);
    Ok(())
}

#[test]
fn ready_min_preserves_absolute_deadline_and_rejects_invalid_or_failed_completions() -> Result<()> {
    let cache = std::rc::Rc::new(std::cell::RefCell::new(
        crate::flash_support::ExpertCache::new(10)?,
    ));
    let lease = crate::flash_support::DemandPlan::reserve(&cache, &[1])?.reads[0];
    let (sender, receiver) = mpsc::channel();
    let deadline = Instant::now();
    let mut batch = ReadBatch {
        io_cancelled: Arc::new(AtomicBool::new(false)),
        receiver,
        cancelled: Arc::new(AtomicBool::new(false)),
        draining: Arc::new(AtomicBool::new(false)),
        abandoned: false,
        remaining: 2,
        deadline,
        trace: Vec::new(),
        #[cfg(test)]
        wait_started: None,
    };
    assert!(batch.ready_min(0).is_err());
    assert!(batch.ready_min(3).is_err());
    assert_eq!(batch.remaining(), 2);
    sender.send(Completed::new(lease, Ok(vec![7])))?;
    assert!(batch.ready_min(2).is_err());
    assert_eq!(batch.remaining(), 1);
    assert_eq!(batch.deadline, deadline);
    sender.send(Completed::new(lease, Err(anyhow::anyhow!("injected EOF"))))?;
    assert!(
        format!(
            "{:#}",
            batch
                .ready_min(1)
                .err()
                .context("failed completion accepted")?
        )
        .contains("injected EOF")
    );
    assert_eq!(batch.remaining(), 0);
    assert!(batch.ready_min(0)?.is_empty());
    assert!(batch.ready_min(1).is_err());
    drop(sender);
    let (sender, receiver) = mpsc::channel();
    drop(sender);
    let mut disconnected = ReadBatch {
        io_cancelled: Arc::new(AtomicBool::new(false)),
        receiver,
        cancelled: Arc::new(AtomicBool::new(false)),
        draining: Arc::new(AtomicBool::new(false)),
        abandoned: false,
        remaining: 1,
        deadline,
        trace: Vec::new(),
        #[cfg(test)]
        wait_started: None,
    };
    assert!(disconnected.ready_min(1).is_err());
    Ok(())
}

#[test]
fn real_workers_release_middle_before_tail_without_new_reads_or_workers() -> Result<()> {
    use std::io::Write;
    let pool = ReadService::new(4)?;
    let mut file = tempfile::tempfile()?;
    file.write_all(b"0123456789")?;
    let cache = std::rc::Rc::new(std::cell::RefCell::new(
        crate::flash_support::ExpertCache::new(10)?,
    ));
    let leases = crate::flash_support::DemandPlan::reserve(&cache, &(0..10).collect::<Vec<_>>())?
        .reads
        .clone();
    let gates = pool.set_wave_test_gates();
    let outcome = (|| -> Result<()> {
        let reads = leases
            .iter()
            .enumerate()
            .map(|(i, &lease)| Ok((lease, file.try_clone()?, i as u64, 1)))
            .collect::<Result<Vec<_>>>()?;
        let mut batch = pool.start(reads, Duration::from_secs(2))?;
        let first = batch.ready(false)?;
        ensure!(first.len() == 1 && batch.remaining() == 9);
        pool.release_wave_test_gate(false);
        let middle = batch.ready_min(2)?;
        ensure!(middle.len() == 2 && batch.remaining() == 7);
        ensure!(pool.in_flight() == 7);
        pool.release_wave_test_gate(true);
        let tail = batch.ready(true)?;
        let mut values = first
            .into_iter()
            .chain(middle)
            .chain(tail)
            .map(|(_, v)| v.legacy_bytes()[0])
            .collect::<Vec<_>>();
        values.sort();
        ensure!(values == b"0123456789" && pool.in_flight() == 0 && pool.workers.len() == 4);
        Ok(())
    })();
    for gate in gates {
        gate.release();
    }
    pool.release_wave_test_gate(true);
    let deadline = Instant::now() + Duration::from_secs(2);
    while pool.in_flight() != 0 && Instant::now() < deadline {
        std::thread::yield_now();
    }
    ensure!(pool.in_flight() == 0, "test workers did not drain");
    outcome
}

#[cfg(kani)]
#[kani::proof]
fn native_extra_wave_is_bounded_and_never_splits_two_or_fewer_remaining_reads() {
    let enabled: bool = kani::any();
    let remaining: usize = kani::any();
    let used: bool = kani::any();
    let actual = extra_wave(enabled, remaining, used);
    assert_eq!(actual, enabled && !used && remaining > 2);
    if actual {
        assert!(!extra_wave(enabled, remaining, true));
    }
    kani::cover!(actual && remaining == 3);
    kani::cover!(actual && remaining == 10);
    kani::cover!(!actual && enabled && !used && remaining == 2);
    kani::cover!(!actual && enabled && used && remaining >= 3);
    kani::cover!(!actual && !enabled && !used && remaining >= 3);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn native_extra_wave_flag_is_strict_default_off_and_qualified_only() {
    let raw: [u8; 4] = kani::any();
    let length = usize::from(kani::any::<u8>() % 5);
    let qualified: bool = kani::any();
    assert_eq!(extra_wave_flag(None, qualified), Some(false));
    if let Ok(value) = std::str::from_utf8(&raw[..length]) {
        let actual = extra_wave_flag(Some(value), qualified);
        assert_eq!(
            actual,
            match value {
                "0" => Some(false),
                "1" => Some(qualified),
                _ => None,
            }
        );
        kani::cover!(value == "1" && qualified && actual == Some(true));
        kani::cover!(value == "1" && !qualified && actual == Some(false));
        kani::cover!(value == "0" && actual == Some(false));
        kani::cover!(value != "0" && value != "1" && actual.is_none());
    }
}

#[cfg(kani)]
#[kani::proof]
fn native_ready_minimum_rejects_impossible_thresholds_before_waiting() {
    let minimum: usize = kani::any();
    let remaining: usize = kani::any();
    let valid = ready_minimum_valid(minimum, remaining);
    assert_eq!(
        valid,
        if remaining == 0 {
            minimum == 0
        } else {
            minimum >= 1 && minimum <= remaining
        }
    );
    kani::cover!(valid && minimum == 2 && remaining == 3);
    kani::cover!(valid && minimum == 0 && remaining == 0);
    kani::cover!(!valid && minimum == 0 && remaining > 0);
    kani::cover!(!valid && minimum > remaining);
}
