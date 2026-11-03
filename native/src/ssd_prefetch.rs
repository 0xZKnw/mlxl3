//! Bounded reads into persistent GPU slots; only the inference thread owns MLX.
pub(crate) const PREDICTION_RANKS: usize = 16;

pub(crate) fn prediction_rows(rows: usize) -> Option<usize> {
    (1..=8).contains(&rows).then_some(rows.min(2))
}

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

#[cfg(kani)]
#[kani::proof]
fn draft_prediction_geometry_is_bounded_for_all_usize() {
    let rows: usize = kani::any();
    let selected = prediction_rows(rows);
    assert_eq!(selected.is_some(), rows > 0 && rows <= 8);
    if let Some(n) = selected {
        assert_eq!(n, if rows == 1 { 1 } else { 2 });
        assert!(n <= rows && n * PREDICTION_RANKS <= 32);
    }
    kani::cover!(rows == 0 && selected.is_none());
    kani::cover!(rows == 1 && selected == Some(1));
    kani::cover!(rows == 2 && selected == Some(2));
    kani::cover!(rows == 8 && selected == Some(2));
    kani::cover!(rows == usize::MAX && selected.is_none());
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
        pub demand_reused: AtomicU64,
        pub demand_timeouts: AtomicU64,
        pub demand_wait_ns: AtomicU64,
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

        fn finish(worker: &mut Worker, valid: bool, counters: &Counters) -> bool {
            let Some(pending) = worker.pending.take() else {
                return false;
            };
            let published =
                pending
                    .cache
                    .borrow_mut()
                    .publish_prefetch(pending.slot, pending.expert, valid);
            if !published {
                counters.discarded.fetch_add(1, Ordering::Relaxed);
            }
            // Drops the MLX owners on the inference thread, after the worker is done.
            published
        }

        /// Publish completions from every layer, without waiting for unfinished reads.
        pub fn poll(&mut self) {
            for worker in &mut self.workers {
                match worker.rx.try_recv() {
                    Ok(completed) => {
                        Self::finish(worker, completed.valid, &self.counters);
                    }
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

        /// A timeout retains the write lease; urgent reads may use another slot.
        pub fn wait_demand(
            &mut self,
            layer: usize,
            cache: &std::rc::Rc<std::cell::RefCell<crate::flash_support::ExpertCache>>,
            ids: &[u32],
        ) {
            let started = std::time::Instant::now();
            let deadline = started + std::time::Duration::from_millis(1);
            let mut needed = false;
            for worker in &mut self.workers {
                if !worker.pending.as_ref().is_some_and(|p| {
                    p.layer == layer
                        && std::rc::Rc::ptr_eq(&p.cache, cache)
                        && ids.contains(&p.expert)
                }) {
                    continue;
                }
                needed = true;
                match worker
                    .rx
                    .recv_timeout(deadline.saturating_duration_since(std::time::Instant::now()))
                {
                    Ok(completed) => {
                        if Self::finish(worker, completed.valid, &self.counters) {
                            self.counters.demand_reused.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        self.counters
                            .demand_timeouts
                            .fetch_add(1, Ordering::Relaxed);
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => {
                        worker.tx = None;
                        Self::finish(worker, false, &self.counters);
                    }
                }
            }
            if needed {
                self.counters.demand_wait_ns.fetch_add(
                    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    Ordering::Relaxed,
                );
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

    #[cfg(test)]
    mod demand_tests {
        use super::*;
        use crate::{
            array::{Array, Dtype, PreparedRead},
            flash_support::ExpertCache,
        };
        use anyhow::ensure;
        use std::{cell::RefCell, io::Write, rc::Rc};

        #[test]
        #[ignore = "physical Apple GPU; bounded persistent owners over 256 real reads"]
        fn pending_leases_release_every_batch_without_retaining_allocations() -> Result<()> {
            let mut file = tempfile::tempfile()?;
            let bytes = vec![37u8; 4096];
            file.write_all(&bytes)?;
            let cache = Rc::new(RefCell::new(ExpertCache::new(24)?));
            let weak = Rc::downgrade(&cache);
            let mut banks = vec![Array::zeros_dtype(&[24, 4096], Dtype::UInt8)?];
            banks[0].eval()?;
            let active = crate::array::memory_stats(false)?.mlx_active_bytes;
            let mut pool = ReaderPool::new()?;
            let (mut minimum, mut maximum) = (active, 0);
            for _ in 0..64 {
                cache.borrow_mut().invalidate();
                ensure!(Rc::strong_count(&cache) == 1);
                for id in 0..4 {
                    let slot = cache.borrow_mut().claim_prefetch(id)?.unwrap();
                    let read =
                        Array::prepare_read(&file, 0, &mut banks, &[(0, slot * 4096, 4096)])?;
                    ensure!(pool.submit(0, id, slot, Rc::clone(&cache), read));
                }
                ensure!(Rc::strong_count(&cache) == 5 && !pool.has_room());
                pool.wait_for_test()?;
                ensure!(pool.workers.iter().all(|w| w.pending.is_none()));
                ensure!(Rc::strong_count(&cache) == 1);
                let now = crate::array::memory_stats(false)?.mlx_active_bytes;
                minimum = minimum.min(now);
                maximum = maximum.max(now);
                ensure!(now <= active, "persistent read allocations grew");
                ensure!(banks[0].to_bytes()?[..4 * 4096] == bytes.repeat(4));
            }
            ensure!(pool.counters.completed.load(Ordering::Relaxed) == 256);
            drop(pool);
            drop(cache);
            ensure!(weak.upgrade().is_none());
            println!(
                "SSD_PREFETCH_LEASES batches=64 reads=256 peak_pending=4 cache_owners=1_to_5_to_1 initial_active={active} min_active={minimum} max_active={maximum} retained_after_drop=false"
            );
            Ok(())
        }

        #[test]
        #[ignore = "physical Apple GPU; real records with deterministic completion/error channels"]
        fn demand_reuses_only_finished_matching_records_and_preserves_timeout_leases() -> Result<()>
        {
            let data = (0..64).map(|i| (i * 3 + 1) as u8).collect::<Vec<_>>();
            for outcome in 0..4 {
                let mut file = tempfile::tempfile()?;
                file.write_all(&data)?;
                let cache = Rc::new(RefCell::new(ExpertCache::new(24)?));
                let other = Rc::new(RefCell::new(ExpertCache::new(24)?));
                let mut banks = vec![Array::zeros_dtype(&[24, 64], Dtype::UInt8)?];
                let slot = cache.borrow_mut().claim_prefetch(7)?.unwrap();
                let read = Array::prepare_read(&file, 0, &mut banks, &[(0, slot * 64, 64)])?;
                let token = read.token();
                let (sender, receiver) = mpsc::sync_channel(1);
                let mut pool = ReaderPool {
                    workers: vec![Worker {
                        tx: None,
                        rx: receiver,
                        thread: None,
                        pending: Some(Pending {
                            layer: 3,
                            expert: 7,
                            slot,
                            cache: Rc::clone(&cache),
                            _read: read,
                        }),
                    }],
                    counters: Arc::default(),
                };
                pool.wait_demand(4, &cache, &[7]);
                pool.wait_demand(3, &other, &[7]);
                pool.wait_demand(3, &cache, &[8]);
                ensure!(pool.workers[0].pending.is_some() && !cache.borrow().contains(7));
                ensure!(pool.counters.demand_timeouts.load(Ordering::Relaxed) == 0);
                if outcome <= 1 {
                    if outcome == 1 {
                        file.set_len(0)?;
                    }
                    // The Pending retains the real read lease until this synchronous read finishes.
                    let (valid, bytes) = unsafe { PreparedRead::run(token) };
                    ensure!(valid == (outcome == 0) && bytes == if valid { 64 } else { 0 });
                    sender.send(Completed { valid })?;
                    pool.wait_demand(3, &cache, &[7, 7]);
                    ensure!(pool.workers[0].pending.is_none());
                    ensure!(cache.borrow().contains(7) == valid);
                    ensure!(
                        pool.counters.demand_reused.load(Ordering::Relaxed) == u64::from(valid)
                    );
                    if valid {
                        ensure!(cache.borrow_mut().reserve(&[7])? == [(slot, true)]);
                        ensure!(&banks[0].to_bytes()?[slot * 64..(slot + 1) * 64] == data);
                    } else {
                        ensure!(!cache.borrow_mut().reserve(&[7])?[0].1);
                    }
                } else if outcome == 2 {
                    pool.wait_demand(3, &cache, &[7]);
                    ensure!(pool.counters.demand_timeouts.load(Ordering::Relaxed) == 1);
                    ensure!(pool.workers[0].pending.is_some() && !cache.borrow().contains(7));
                    let urgent = cache.borrow_mut().reserve(&[7])?[0];
                    ensure!(!urgent.1 && urgent.0 != slot);
                    Array::read_record(&file, 0, &mut banks, &[(0, urgent.0 * 64, 64)])?;
                    let (valid, bytes) = unsafe { PreparedRead::run(token) };
                    ensure!(valid && bytes == 64);
                    sender.send(Completed { valid })?;
                    pool.wait_demand(3, &cache, &[7]);
                    ensure!(pool.workers[0].pending.is_none());
                    ensure!(cache.borrow_mut().reserve(&[7])? == [(urgent.0, true)]);
                    ensure!(pool.counters.demand_reused.load(Ordering::Relaxed) == 0);
                    ensure!(pool.counters.discarded.load(Ordering::Relaxed) == 1);
                    let actual = banks[0].to_bytes()?;
                    ensure!(&actual[slot * 64..(slot + 1) * 64] == data);
                    ensure!(&actual[urgent.0 * 64..(urgent.0 + 1) * 64] == data);
                } else {
                    drop(sender);
                    pool.wait_demand(3, &cache, &[7]);
                    ensure!(pool.workers[0].pending.is_none() && !cache.borrow().contains(7));
                    ensure!(pool.counters.discarded.load(Ordering::Relaxed) == 1);
                }
            }
            println!(
                "SSD_DEMAND_REUSE cases=4 identity_checks=12 records=real timeout_pending=true duplicate_rejected=true"
            );
            Ok(())
        }
    }
}
#[cfg(feature = "mlx")]
pub(crate) use gpu::ReaderPool;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draft_predictions_only_cover_valid_prefix_rows() {
        for rows in 0..=10 {
            let expected = match rows {
                1 => Some(1),
                2..=8 => Some(2),
                _ => None,
            };
            assert_eq!(prediction_rows(rows), expected);
        }
        assert_eq!(prediction_rows(usize::MAX), None);
    }

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
    fn prediction_case<const ROWS: usize, const LEN: usize>() {
        let ids: [u32; 32] = kani::any();
        let mut output = ids;
        let valid = ids[..LEN].iter().all(|&id| id < 512);
        let result = order_predictions(&mut output[..LEN], ROWS);
        assert_eq!(result, valid);
        if result {
            for position in 0..LEN {
                let expected = if ROWS == 1 {
                    ids[15 - position]
                } else {
                    ids[(position % 2) * 16 + 15 - position / 2]
                };
                assert_eq!(output[position], expected);
                assert!(output[position] < 512);
            }
        } else {
            for position in 0..32 {
                assert_eq!(output[position], ids[position]);
            }
        }
        kani::cover!(result);
        kani::cover!(!result && ids[LEN - 1] >= 512);
    }
    #[kani::proof]
    #[kani::unwind(33)]
    fn prediction_order_one_row() {
        prediction_case::<1, 16>();
    }
    #[kani::proof]
    #[kani::unwind(33)]
    fn prediction_order_two_rows() {
        prediction_case::<2, 32>();
    }
    #[kani::proof]
    #[kani::unwind(33)]
    fn prediction_order_rejects_every_other_geometry() {
        let ids: [u32; 32] = kani::any();
        let length = usize::from(kani::any::<u8>()).min(32);
        let rows: usize = kani::any();
        kani::assume(!((rows == 1 && length == 16) || (rows == 2 && length == 32)));
        let mut output = ids;
        assert!(!order_predictions(&mut output[..length], rows));
        for position in 0..32 {
            assert_eq!(output[position], ids[position]);
        }
        kani::cover!(rows > 2);
        kani::cover!(rows == 0);
        kani::cover!(length == 0);
    }
}
