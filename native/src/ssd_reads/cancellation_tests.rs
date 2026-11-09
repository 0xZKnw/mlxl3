use super::*;
use cancellable_io::{READ_CHUNK_BYTES, ReadProbe};

fn wait_until(mut condition: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(2);
    while !condition() {
        if Instant::now() >= deadline {
            return false;
        }
        std::thread::yield_now();
    }
    true
}

#[test]
fn production_workers_cancel_queued_and_partial_reads_without_reviving_after_reset() -> Result<()> {
    for (dynamic, split, background) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (false, false, true),
    ] {
        for partial in [false, true] {
            let mut pool = ReadService::new_with_options(2, dynamic, split)?;
            let requested = Arc::new(AtomicBool::new(false));
            pool.set_cancellation(requested.clone());
            let after_first = Arc::new(TestReadGate::default());
            let probe = Arc::new(ReadProbe {
                calls: AtomicUsize::new(0),
                after_first: partial.then(|| after_first.clone()),
            });
            *pool.io_probe.borrow_mut() = Some(probe.clone());
            let gates = [
                Arc::new(TestReadGate::default()),
                Arc::new(TestReadGate::default()),
            ];
            if split {
                *pool.split_gates.borrow_mut() = Some(gates.clone());
            } else if !partial {
                *pool.held_first.borrow_mut() = Some(gates[0].clone());
            }
            let file = tempfile::tempfile()?;
            let expected = (0..(4 * READ_CHUNK_BYTES + 17))
                .map(|i| (i % 251) as u8)
                .collect::<Vec<_>>();
            file.write_all_at(&expected, 0)?;
            let mut batch = pool.start_payloads(
                vec![(
                    tests::lease()?,
                    file.try_clone()?,
                    0,
                    expected.len(),
                    ReadPayload::Bytes(Vec::new()),
                )]
                .into_iter(),
                Duration::from_secs(5),
                background,
                split,
            )?;
            if partial && split {
                gates[0].release();
            }
            let entered = !partial || wait_until(|| probe.calls.load(Ordering::Acquire) == 1);
            requested.store(true, Ordering::Release);
            let rejected = if background {
                batch.poll().is_err()
            } else {
                batch.ready(true).is_err()
            };
            drop(batch);
            requested.store(false, Ordering::Release);
            // Clear test hooks for the retry. Old jobs retain their own hooks.
            *pool.held_first.borrow_mut() = None;
            *pool.split_gates.borrow_mut() = None;
            *pool.io_probe.borrow_mut() = None;
            let retained = pool.in_flight();
            let refused = pool.start(Vec::new(), Duration::ZERO).is_err();
            after_first.release();
            for gate in gates {
                gate.release();
            }
            let drained = wait_until(|| pool.in_flight() == 0);
            assert!(entered && rejected && refused && drained);
            assert_eq!(retained, 1);
            // Unlike logical detachment, no remaining positional read is issued.
            assert_eq!(probe.calls.load(Ordering::Acquire), usize::from(partial));
            let mut retry = pool.start_payloads(
                vec![(
                    tests::lease()?,
                    file,
                    0,
                    expected.len(),
                    ReadPayload::Bytes(Vec::new()),
                )]
                .into_iter(),
                Duration::from_secs(5),
                background,
                split,
            )?;
            let result = retry.ready(true)?;
            assert_eq!(result.len(), 1);
            assert_eq!(result[0].1.legacy_bytes(), expected);
        }
    }
    Ok(())
}
