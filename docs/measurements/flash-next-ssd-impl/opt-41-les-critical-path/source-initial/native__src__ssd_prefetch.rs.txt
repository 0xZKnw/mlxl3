//! Bounded reads into persistent GPU slots; only the inference thread owns MLX.
pub(crate) const PREDICTION_RANKS: usize = 16;

/// Strongest rank first, alternating target rows before using the next rank.
pub(crate) fn order_predictions(ids: &mut [u32], rows: usize) -> bool {
    if !(1..=2).contains(&rows)
        || ids.len() != rows * PREDICTION_RANKS
        || ids.iter().any(|&id| id >= 512)
    {
        return false;
    }
    let mut source = [0u32; 2 * PREDICTION_RANKS];
    source[..ids.len()].copy_from_slice(ids);
    let mut position = 0;
    for rank in (0..PREDICTION_RANKS).rev() {
        for row in 0..rows {
            ids[position] = source[row * PREDICTION_RANKS + rank];
            position += 1;
        }
    }
    true
}

#[cfg(feature = "mlx")]
mod gpu {
    use anyhow::Result;
    use std::{
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
            mpsc::{self, Receiver, SyncSender, TrySendError},
        },
        thread::{self, JoinHandle},
    };

    #[derive(Default)]
    pub(crate) struct Counters {
        pub submitted: AtomicU64,
        pub completed: AtomicU64,
        pub used: AtomicU64,
        pub discarded: AtomicU64,
        pub failed: AtomicU64,
        pub late_needed: AtomicU64,
        pub bytes: AtomicU64,
    }

    struct Pending {
        layer: usize,
        expert: u32,
        slot: usize,
        cache: std::rc::Rc<std::cell::RefCell<crate::flash_support::ExpertCache>>,
        _read: crate::array::PreparedRead,
    }
    struct Completed {
        valid: bool,
    }
    struct Worker {
        tx: Option<SyncSender<usize>>,
        rx: Receiver<Completed>,
        thread: Option<JoinHandle<()>>,
        pending: Option<Pending>,
    }
    pub(crate) struct ReaderPool {
        workers: Vec<Worker>,
        pub counters: Arc<Counters>,
    }

    impl ReaderPool {
        pub fn new() -> Result<Self> {
            let mut pool = Self {
                workers: Vec::new(),
                counters: Arc::default(),
            };
            for index in 0..4 {
                let (tx, jobs) = mpsc::sync_channel::<usize>(1);
                let (results, rx) = mpsc::sync_channel(1);
                let counters = Arc::clone(&pool.counters);
                let thread = thread::Builder::new()
                    .name(format!("ssd-lookahead-{index}"))
                    .spawn(move || {
                        while let Ok(token) = jobs.recv() {
                            // The main-thread Pending owns the lease through completion or join.
                            let (valid, bytes) = unsafe { crate::array::PreparedRead::run(token) };
                            counters.bytes.fetch_add(bytes, Ordering::Relaxed);
                            counters.completed.fetch_add(1, Ordering::Relaxed);
                            if !valid {
                                counters.failed.fetch_add(1, Ordering::Relaxed);
                            }
                            if results.send(Completed { valid }).is_err() {
                                break;
                            }
                        }
                    })?;
                pool.workers.push(Worker {
                    tx: Some(tx),
                    rx,
                    thread: Some(thread),
                    pending: None,
                });
            }
            Ok(pool)
        }

        pub fn has_room(&self) -> bool {
            self.workers
                .iter()
                .any(|w| w.pending.is_none() && w.tx.is_some())
        }

        pub fn submit(
            &mut self,
            layer: usize,
            expert: u32,
            slot: usize,
            cache: std::rc::Rc<std::cell::RefCell<crate::flash_support::ExpertCache>>,
            read: crate::array::PreparedRead,
        ) -> bool {
            let token = read.token();
            let pending = Pending {
                layer,
                expert,
                slot,
                cache,
                _read: read,
            };
            let worker = self
                .workers
                .iter_mut()
                .find(|w| w.pending.is_none() && w.tx.is_some());
            if let Some(worker) = worker {
                match worker
                    .tx
                    .as_ref()
                    .expect("available reader")
                    .try_send(token)
                {
                    Ok(()) => {
                        worker.pending = Some(pending);
                        self.counters.submitted.fetch_add(1, Ordering::Relaxed);
                        return true;
                    }
                    Err(TrySendError::Disconnected(_)) => {
                        worker.tx = None;
                    }
                    Err(TrySendError::Full(_)) => {}
                }
            }
            pending
                .cache
                .borrow_mut()
                .publish_prefetch(slot, expert, false);
            false
        }

        fn finish(worker: &mut Worker, valid: bool, counters: &Counters) {
            if let Some(pending) = worker.pending.take()
                && !pending
                    .cache
                    .borrow_mut()
                    .publish_prefetch(pending.slot, pending.expert, valid)
            {
                counters.discarded.fetch_add(1, Ordering::Relaxed);
            }
            // Drops the MLX owners on the inference thread, after the worker is done.
        }

        /// Publish completions from every layer, without waiting for unfinished reads.
        pub fn poll(&mut self) {
            for worker in &mut self.workers {
                match worker.rx.try_recv() {
                    Ok(completed) => Self::finish(worker, completed.valid, &self.counters),
                    Err(mpsc::TryRecvError::Disconnected) => {
                        worker.tx = None;
                        Self::finish(worker, false, &self.counters);
                    }
                    Err(mpsc::TryRecvError::Empty) => {}
                }
            }
        }

        pub fn note_late(&self, layer: usize, ids: &[u32]) {
            for worker in &self.workers {
                if worker
                    .pending
                    .as_ref()
                    .is_some_and(|p| p.layer == layer && ids.contains(&p.expert))
                {
                    self.counters.late_needed.fetch_add(1, Ordering::Relaxed);
                }
            }
        }

        #[cfg(test)]
        pub fn wait_for_test(&mut self) -> Result<()> {
            for worker in &mut self.workers {
                if worker.pending.is_some() {
                    let completed = worker.rx.recv_timeout(std::time::Duration::from_secs(5))?;
                    Self::finish(worker, completed.valid, &self.counters);
                }
            }
            Ok(())
        }
    }

    impl Drop for ReaderPool {
        fn drop(&mut self) {
            for worker in &mut self.workers {
                worker.tx.take();
            }
            for worker in &mut self.workers {
                if let Some(thread) = worker.thread.take() {
                    let _ = thread.join();
                }
                let valid = worker.rx.try_recv().is_ok_and(|c| c.valid);
                Self::finish(worker, valid, &self.counters);
            }
        }
    }
}
#[cfg(feature = "mlx")]
pub(crate) use gpu::ReaderPool;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_interleave_strongest_ranks_and_reject_invalid_shapes() {
        let original = (0..32).collect::<Vec<_>>();
        let mut ids = original.clone();
        assert!(order_predictions(&mut ids, 2));
        assert_eq!(
            ids,
            (0..16).rev().flat_map(|i| [i, i + 16]).collect::<Vec<_>>()
        );
        let mut ids = original[..16].to_vec();
        assert!(order_predictions(&mut ids, 1));
        assert_eq!(ids, (0..16).rev().collect::<Vec<_>>());
        for rows in [0, 3, usize::MAX] {
            let mut ids = original.clone();
            assert!(!order_predictions(&mut ids, rows));
            assert_eq!(ids, original);
        }
        for len in [0, 1, 15, 17, 31] {
            let mut ids = original[..len].to_vec();
            let before = ids.clone();
            assert!(!order_predictions(&mut ids, 2));
            assert_eq!(ids, before);
        }
        let mut invalid = original;
        invalid[31] = 512;
        let before = invalid.clone();
        assert!(!order_predictions(&mut invalid, 2));
        assert_eq!(invalid, before);
        let mut duplicates = [511; 16];
        assert!(order_predictions(&mut duplicates, 1));
        assert_eq!(duplicates, [511; 16]);
    }

    #[cfg(feature = "mlx")]
    #[test]
    #[ignore = "physical Apple GPU; owned direct reads, partial I/O and destruction"]
    fn persistent_reads_keep_owners_publish_once_and_join_before_drop() -> anyhow::Result<()> {
        use crate::{
            array::{Array, Dtype},
            flash_support::ExpertCache,
        };
        use anyhow::ensure;
        use std::{
            cell::RefCell,
            io::Write,
            rc::Rc,
            sync::{Arc, atomic::Ordering},
        };
        let mut file = tempfile::tempfile()?;
        let bytes = (0..4096).map(|i| (i % 251) as u8).collect::<Vec<_>>();
        file.write_all(&bytes)?;
        let cache = Rc::new(RefCell::new(ExpertCache::new(24)?));
        let mut banks = vec![Array::zeros_dtype(&[24, 4096], Dtype::UInt8)?];
        let observer = banks[0].try_clone()?;
        let mut pool = ReaderPool::new()?;
        for id in 0..4 {
            let slot = cache.borrow_mut().claim_prefetch(id)?.unwrap();
            let read = Array::prepare_read(&file, 0, &mut banks, &[(0, slot * 4096, 4096)])?;
            ensure!(pool.submit(7, id, slot, Rc::clone(&cache), read));
        }
        ensure!(!pool.has_room());
        // Even if OS reads already finished, slots stay pending until the owner polls.
        ensure!((0..4).all(|id| !cache.borrow().contains(id)));
        let demand = cache.borrow_mut().reserve(&[0])?[0];
        ensure!(!demand.1 && demand.0 >= 4);
        Array::read_record(&file, 0, &mut banks, &[(0, demand.0 * 4096, 4096)])?;
        drop(banks);
        pool.wait_for_test()?;
        ensure!((0..4).all(|id| cache.borrow().contains(id)));
        ensure!(pool.counters.discarded.load(Ordering::Relaxed) == 1);
        let result = observer.to_bytes()?;
        ensure!(result.len() == 24 * 4096);
        for slot in 0..4 {
            ensure!(result[slot * 4096..(slot + 1) * 4096] == bytes);
        }
        pool.poll();
        ensure!(pool.counters.completed.load(Ordering::Relaxed) == 4);
        ensure!(pool.counters.bytes.load(Ordering::Relaxed) == 4 * 4096);
        let mut banks = vec![Array::zeros_dtype(&[4096], Dtype::UInt8)?];
        let slot = cache.borrow_mut().claim_prefetch(511)?.unwrap();
        let read = Array::prepare_read(&file, 0, &mut banks, &[(0, 0, 4096)])?;
        file.set_len(7)?;
        ensure!(pool.submit(7, 511, slot, Rc::clone(&cache), read));
        pool.wait_for_test()?;
        ensure!(!cache.borrow().contains(511));
        ensure!(pool.counters.failed.load(Ordering::Relaxed) == 1);
        ensure!(pool.counters.bytes.load(Ordering::Relaxed) == 4 * 4096 + 7);
        ensure!(Array::prepare_read(&file, u64::MAX, &mut banks, &[(0, 0, 1)]).is_err());
        ensure!(Array::prepare_read(&file, 0, &mut banks, &[(0, 4096, 1)]).is_err());
        ensure!(Array::prepare_read(&file, 0, &mut banks, &[(0, 0, 4), (0, 2, 2)]).is_err());
        file.set_len(4096)?;
        for id in 4..8 {
            let slot = cache.borrow_mut().claim_prefetch(id)?.unwrap();
            let mut bank = vec![Array::zeros_dtype(&[4096], Dtype::UInt8)?];
            let read = Array::prepare_read(&file, 0, &mut bank, &[(0, 0, 4096)])?;
            ensure!(pool.submit(7, id, slot, Rc::clone(&cache), read));
        }
        let counters = Arc::clone(&pool.counters);
        drop(file);
        drop(pool);
        ensure!(
            counters.completed.load(Ordering::Relaxed)
                == counters.submitted.load(Ordering::Relaxed)
        );
        ensure!((4..8).all(|id| cache.borrow().contains(id)));
        Ok(())
    }

    #[cfg(all(kani, feature = "mlx"))]
    #[kani::proof]
    fn actual_owned_read_pool_and_drop_path() {
        persistent_reads_keep_owners_publish_once_and_join_before_drop().unwrap();
    }
}

#[cfg(kani)]
mod verification {
    use super::*;
    #[kani::proof]
    #[kani::unwind(33)]
    fn prediction_order_preserves_all_values_within_two_rows() {
        let ids: [u32; 32] = kani::any();
        let length = usize::from(kani::any::<u8>()).min(32);
        let rows: usize = kani::any();
        let mut output = ids;
        let valid = (1..=2).contains(&rows)
            && length == rows * 16
            && ids[..length].iter().all(|&id| id < 512);
        let result = order_predictions(&mut output[..length], rows);
        assert_eq!(result, valid);
        if result {
            for position in 0..length {
                let expected = if rows == 1 {
                    ids[15 - position]
                } else {
                    ids[(position % 2) * 16 + 15 - position / 2]
                };
                assert_eq!(output[position], expected);
                assert!(output[position] < 512);
            }
            kani::cover!(rows == 1);
            kani::cover!(rows == 2);
        } else {
            for position in 0..32 {
                assert_eq!(output[position], ids[position]);
            }
            kani::cover!(rows > 2);
            kani::cover!(rows == 0);
            kani::cover!(length == 0);
            kani::cover!(length == 32 && rows == 2 && ids[31] >= 512);
        }
    }
}
