//! Terminal read failures must not turn cleanup into another blocking read.
use super::*;
use std::io::Write;

pub(super) fn drained(pool: &ReadService) -> bool {
    let until = Instant::now() + Duration::from_secs(5);
    while pool.in_flight() != 0 && Instant::now() < until {
        std::thread::yield_now();
    }
    pool.in_flight() == 0
}

#[test]
fn terminal_read_failure_detaches_before_held_owner_and_allows_retry_after_drain() -> Result<()> {
    for dynamic in [false, true] {
        let pool = ReadService::new_with_schedule(2, dynamic)?;
        let held = Arc::new(TestReadGate::default());
        *pool.held_first.borrow_mut() = Some(held.clone());
        let mut file = tempfile::tempfile()?;
        file.write_all(b"retained")?;
        let mut batch = pool.start(
            vec![
                (tests::lease()?, file.try_clone()?, 0, 8),
                (tests::lease()?, file.try_clone()?, 99, 8),
            ],
            Duration::from_secs(2),
        )?;
        let stopped = batch.io_cancelled.clone();
        let (done, ack) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let result = batch.ready(true);
            let original_eof = result.as_ref().err().is_some_and(|error| {
                error
                    .downcast_ref::<std::io::Error>()
                    .is_some_and(|e| e.kind() == std::io::ErrorKind::UnexpectedEof)
            });
            drop(batch);
            let _ = done.send(original_eof);
        });
        let acknowledged = ack.recv_timeout(Duration::from_secs(1));
        let retained = pool.in_flight();
        let stopped_before_release = stopped.load(Ordering::Acquire);
        let refused_demand = pool.start(Vec::new(), Duration::ZERO).is_err();
        let refused_background = pool
            .start_mode(
                vec![(tests::lease()?, file.try_clone()?, 0, 8)],
                Duration::ZERO,
                true,
            )
            .is_err();
        held.release();
        *pool.held_first.borrow_mut() = None;
        waiter
            .join()
            .map_err(|_| anyhow::anyhow!("failure waiter panicked"))?;
        let idle = drained(&pool);
        assert_eq!(
            acknowledged.ok(),
            Some(true),
            "EOF cleanup waited for held I/O"
        );
        assert_eq!(retained, 1, "the pending worker lost its ownership ticket");
        assert!(stopped_before_release && refused_demand && refused_background && idle);
        let mut retry = pool.start(vec![(tests::lease()?, file, 0, 8)], Duration::from_secs(2))?;
        assert_eq!(retry.ready(true)?[0].1.legacy_bytes(), b"retained");
    }
    Ok(())
}

#[test]
fn terminal_timeout_cannot_publish_late_completion_even_while_batch_is_retained() -> Result<()> {
    for polling in [false, true] {
        let pool = ReadService::new(1)?;
        let held = Arc::new(TestReadGate::default());
        *pool.held_first.borrow_mut() = Some(held.clone());
        let mut file = tempfile::tempfile()?;
        file.write_all(b"late")?;
        let mut batch = pool.start(vec![(tests::lease()?, file, 0, 4)], Duration::ZERO)?;
        let failed = if polling {
            batch.poll().is_err()
        } else {
            batch.ready(true).is_err()
        };
        let remaining = batch.remaining();
        let stopped = batch.io_cancelled.load(Ordering::Acquire);
        held.release();
        let idle = drained(&pool);
        let revived = batch.ready(true).is_ok() || batch.poll().is_ok();
        drop(batch);
        assert!(
            failed && idle && !revived,
            "failed batch was revived by a late result"
        );
        assert_eq!(remaining, 0);
        assert!(stopped);
    }
    Ok(())
}

#[test]
fn terminal_partial_enqueue_failure_stops_owned_work_without_waiting() -> Result<()> {
    let held = Arc::new(TestReadGate::default());
    let worker_gate = held.clone();
    let (done, ack) = mpsc::channel();
    let waiter = std::thread::spawn(move || -> Result<()> {
        let pool = ReadService::new_with_schedule(2, false)?;
        *pool.held_first.borrow_mut() = Some(worker_gate);
        let mut file = tempfile::tempfile()?;
        file.write_all(b"retry")?;
        let reads = (0..2).map(|i| {
            ensure!(i == 0, "injected destination failure");
            Ok((
                tests::lease()?,
                file.try_clone()?,
                0,
                5,
                ReadPayload::Bytes(Vec::new()),
            ))
        });
        let result = pool.start_fallible_payloads(reads, Duration::from_secs(2), false, false);
        let _ = done.send((result.err().map(|e| e.to_string()), pool.in_flight()));
        *pool.held_first.borrow_mut() = None;
        ensure!(drained(&pool), "failed-enqueue worker did not drain");
        let mut retry = pool.start(vec![(tests::lease()?, file, 0, 5)], Duration::from_secs(2))?;
        ensure!(retry.ready(true)?[0].1.legacy_bytes() == b"retry");
        Ok(())
    });
    let acknowledged = ack.recv_timeout(Duration::from_secs(1));
    held.release();
    waiter
        .join()
        .map_err(|_| anyhow::anyhow!("enqueue waiter panicked"))??;
    assert_eq!(
        acknowledged.ok(),
        Some((Some("injected destination failure".into()), 1))
    );
    Ok(())
}

#[test]
fn terminal_error_discards_queued_success_and_disconnect_is_sticky() -> Result<()> {
    for disconnected in [false, true] {
        let (sender, receiver) = mpsc::channel();
        let mut batch = ReadBatch {
            io_cancelled: Arc::new(AtomicBool::new(false)),
            receiver,
            cancelled: Arc::new(AtomicBool::new(false)),
            draining: Arc::new(AtomicBool::new(false)),
            abandoned: false,
            remaining: 2,
            deadline: Instant::now() + Duration::from_secs(2),
            trace: Vec::new(),
            wait_started: None,
        };
        if !disconnected {
            sender.send(Completed::new(
                tests::lease()?,
                Err(anyhow::anyhow!("injected read failure")),
            ))?;
            sender.send(Completed::new(tests::lease()?, Ok(vec![7])))?;
        }
        drop(sender);
        let failure = batch
            .ready(true)
            .err()
            .context("expected terminal read failure")?;
        if !disconnected {
            assert_eq!(failure.to_string(), "injected read failure");
        }
        let revived = batch.ready(true).is_ok();
        let remaining = batch.remaining();
        drop(batch);
        assert!(!revived && remaining == 0);
    }
    Ok(())
}

#[test]
fn successful_empty_poll_and_invalid_minimum_do_not_abandon_live_work() -> Result<()> {
    let pool = ReadService::new(1)?;
    let held = Arc::new(TestReadGate::default());
    *pool.held_first.borrow_mut() = Some(held.clone());
    let mut file = tempfile::tempfile()?;
    file.write_all(b"live")?;
    let mut batch = pool.start(vec![(tests::lease()?, file, 0, 4)], Duration::from_secs(2))?;
    let empty = matches!(batch.poll(), Ok(None));
    let invalid = batch.ready_min(2).is_err();
    let stopped = batch.io_cancelled.load(Ordering::Acquire);
    held.release();
    let result = batch.ready(true);
    assert!(empty && invalid && !stopped);
    assert_eq!(result?[0].1.legacy_bytes(), b"live");
    Ok(())
}
