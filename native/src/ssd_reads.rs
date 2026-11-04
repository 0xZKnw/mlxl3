//! Bounded owned disk reads. Workers own their entire unpublished destination.
use crate::{
    checkpoint::TensorInfo,
    flash_support::{EXPERT_BYTES, SlotLease},
};
use anyhow::{Context, Result, ensure};
use std::{
    cell::Cell,
    fs::File,
    os::unix::fs::FileExt,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender, SyncSender},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

pub(crate) enum ReadPayload {
    Bytes(Vec<u8>),
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    Shared(metal::Buffer),
}

impl ReadPayload {
    #[cfg(test)]
    fn legacy_bytes(&self) -> &[u8] {
        match self {
            Self::Bytes(bytes) => bytes,
            #[cfg(all(target_os = "macos", feature = "direct-metal"))]
            Self::Shared(_) => panic!("legacy bytes expected"),
        }
    }
}

#[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
pub(crate) fn direct_flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    crate::flash_support::hc_chain_flag(value, qualified_m5).map(|enabled| enabled && qualified_m5)
}

#[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
pub(crate) fn direct_read_eligible(enabled: bool, depth: usize, prefetcher: bool) -> bool {
    enabled && depth == 0 && !prefetcher
}

#[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
pub(crate) fn direct_destination_valid(bytes: usize, length: u64, shared: bool) -> bool {
    bytes > 0 && bytes <= EXPERT_BYTES as usize && length == bytes as u64 && shared
}

pub(crate) struct Completed {
    lease: SlotLease,
    result: Result<ReadPayload>,
    #[cfg(test)]
    timing: (Instant, Instant),
}
#[cfg(test)]
impl Completed {
    fn new(lease: SlotLease, result: Result<Vec<u8>>) -> Self {
        Self {
            lease,
            result: result.map(ReadPayload::Bytes),
            #[cfg(test)]
            timing: (Instant::now(), Instant::now()),
        }
    }
}

struct Ticket(Arc<AtomicUsize>);
impl Drop for Ticket {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}
struct Read {
    lease: SlotLease,
    file: File,
    offset: u64,
    bytes: usize,
    payload: ReadPayload,
    result: Sender<Completed>,
    ticket: Ticket,
    #[cfg(test)]
    gate: Option<Arc<std::sync::Barrier>>,
}
impl Read {
    fn run(self) {
        let Read {
            lease,
            file,
            offset,
            bytes,
            mut payload,
            result: sender,
            ticket,
            #[cfg(test)]
            gate,
        } = self;
        #[cfg(test)]
        let started = Instant::now();
        #[cfg(test)]
        if let Some(gate) = gate {
            gate.wait();
        }
        let result = (|| -> Result<ReadPayload> {
            match &mut payload {
                ReadPayload::Bytes(destination) => {
                    destination.try_reserve_exact(bytes)?;
                    destination.resize(bytes, 0);
                    file.read_exact_at(destination, offset)?;
                }
                #[cfg(all(target_os = "macos", feature = "direct-metal"))]
                ReadPayload::Shared(destination) => {
                    ensure!(
                        direct_destination_valid(
                            bytes,
                            destination.length(),
                            destination.storage_mode() == metal::MTLStorageMode::Shared
                        ),
                        "invalid owned SSD destination"
                    );
                    ensure!(!destination.contents().is_null(), "null SSD destination");
                    // Only this job owns this fresh allocation. No views or buffer
                    // clones exist until the completed payload is received.
                    let target = unsafe {
                        std::slice::from_raw_parts_mut(destination.contents().cast::<u8>(), bytes)
                    };
                    file.read_exact_at(target, offset)?;
                }
            }
            Ok(payload)
        })();
        drop(ticket);
        let result = Completed {
            lease,
            result,
            #[cfg(test)]
            timing: (started, Instant::now()),
        };
        let _ = sender.send(result);
    }
}

pub(crate) struct ReadService {
    next: Cell<usize>,
    workers: Vec<(SyncSender<Read>, JoinHandle<()>)>,
    active: Arc<AtomicUsize>,
    #[cfg(test)]
    gate: std::cell::RefCell<Option<Arc<std::sync::Barrier>>>,
}
impl ReadService {
    pub(crate) fn new(readers: usize) -> Result<Self> {
        ensure!((1..=4).contains(&readers), "SSD readers must be in 1..=4");
        let mut service = Self {
            next: Cell::new(0),
            workers: Vec::new(),
            active: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            gate: std::cell::RefCell::new(None),
        };
        for index in 0..readers {
            let (sender, receiver) = mpsc::sync_channel::<Read>(10);
            let thread = std::thread::Builder::new()
                .name(format!("native-ssd-{index}"))
                .spawn(move || {
                    while let Ok(read) = receiver.recv() {
                        read.run();
                    }
                })?;
            service.workers.push((sender, thread));
        }
        Ok(service)
    }
    #[cfg(test)]
    pub(crate) fn set_test_gate(&self, gate: Option<Arc<std::sync::Barrier>>) {
        *self.gate.borrow_mut() = gate;
    }
    #[cfg(test)]
    pub(crate) fn release_test_gate(&self) {
        if let Some(gate) = self.gate.borrow_mut().take() {
            gate.wait();
        }
    }
    pub(crate) fn in_flight(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }
    pub(crate) fn submit(
        &self,
        records: impl IntoIterator<Item = (SlotLease, TensorInfo)>,
    ) -> Result<ReadBatch> {
        self.start(Self::sources(records)?, Duration::from_secs(30))
    }
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    pub(crate) fn submit_shared(
        &self,
        records: impl IntoIterator<Item = (SlotLease, TensorInfo)>,
        device: &metal::DeviceRef,
    ) -> Result<ReadBatch> {
        self.start_shared(Self::sources(records)?, Duration::from_secs(30), device)
    }
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    fn start_shared(
        &self,
        reads: Vec<(SlotLease, File, u64, usize)>,
        limit: Duration,
        device: &metal::DeviceRef,
    ) -> Result<ReadBatch> {
        self.validate_start(reads.len(), false)?;
        let reads = reads
            .into_iter()
            .map(|(lease, file, offset, bytes)| {
                ensure!(
                    direct_destination_valid(bytes, bytes as u64, true),
                    "invalid owned SSD destination extent"
                );
                let destination = crate::gpu::buffer(
                    device,
                    bytes as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                )?;
                Ok((lease, file, offset, bytes, ReadPayload::Shared(destination)))
            })
            .collect::<Result<Vec<_>>>()?;
        self.start_payloads(reads.into_iter(), limit, false)
    }
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    pub(crate) fn submit_background(
        &self,
        lease: SlotLease,
        record: TensorInfo,
    ) -> Result<ReadBatch> {
        self.start_mode(
            Self::sources([(lease, record)])?,
            Duration::from_secs(30),
            true,
        )
    }
    fn sources(
        records: impl IntoIterator<Item = (SlotLease, TensorInfo)>,
    ) -> Result<Vec<(SlotLease, File, u64, usize)>> {
        let mut reads = Vec::new();
        for (lease, record) in records {
            // The inspected record descriptor is its first constituent tensor.
            let (file, offset) = record.source()?;
            let end = offset
                .checked_add(EXPERT_BYTES)
                .context("Flash record offset overflow")?;
            ensure!(
                end <= file.metadata()?.len(),
                "checkpoint record was truncated after inspection"
            );
            reads.push((lease, file.try_clone()?, offset, EXPERT_BYTES as usize));
        }
        Ok(reads)
    }
    fn start(
        &self,
        reads: Vec<(SlotLease, File, u64, usize)>,
        limit: Duration,
    ) -> Result<ReadBatch> {
        self.start_mode(reads, limit, false)
    }
    fn start_mode(
        &self,
        reads: Vec<(SlotLease, File, u64, usize)>,
        limit: Duration,
        background: bool,
    ) -> Result<ReadBatch> {
        let reads = reads.into_iter().map(|(lease, file, offset, bytes)| {
            (lease, file, offset, bytes, ReadPayload::Bytes(Vec::new()))
        });
        self.start_payloads(reads, limit, background)
    }
    fn validate_start(&self, count: usize, background: bool) -> Result<()> {
        ensure!(count <= 10, "native SSD demand exceeds one AR cohort");
        ensure!(
            !background || (count == 1 && self.in_flight() < 4),
            "native prefetch exceeds four readers"
        );
        ensure!(
            background || self.active.load(Ordering::Acquire) == 0,
            "previous native SSD reads are still draining"
        );
        Ok(())
    }
    fn start_payloads(
        &self,
        reads: impl ExactSizeIterator<Item = (SlotLease, File, u64, usize, ReadPayload)>,
        limit: Duration,
        background: bool,
    ) -> Result<ReadBatch> {
        self.validate_start(reads.len(), background)?;
        let (result, receiver) = mpsc::channel();
        let mut batch = ReadBatch {
            receiver,
            remaining: 0,
            deadline: Instant::now() + limit,
            #[cfg(test)]
            trace: Vec::new(),
        };
        #[cfg(test)]
        let count = reads.len();
        for (lease, file, offset, bytes, payload) in reads {
            self.active.fetch_add(1, Ordering::Relaxed);
            let read = Read {
                lease,
                file,
                offset,
                bytes,
                payload,
                result: result.clone(),
                ticket: Ticket(self.active.clone()),
                #[cfg(test)]
                gate: (batch.remaining + 1 == count)
                    .then(|| self.gate.borrow().clone())
                    .flatten(),
            };
            let worker = self.next.get();
            self.next.set((worker + 1) % self.workers.len());
            self.workers[worker]
                .0
                .try_send(read)
                .map_err(|_| anyhow::anyhow!("native SSD reader unavailable"))?;
            batch.remaining += 1;
        }
        Ok(batch)
    }
}
impl Drop for ReadService {
    fn drop(&mut self) {
        let workers = std::mem::take(&mut self.workers);
        for (sender, thread) in workers {
            drop(sender);
            if self.active.load(Ordering::Acquire) == 0 {
                let _ = thread.join();
            }
            // A timed-out read owns its file and destination through physical completion.
        }
    }
}

pub(crate) struct ReadBatch {
    receiver: Receiver<Completed>,
    remaining: usize,
    deadline: Instant,
    #[cfg(test)]
    pub(crate) trace: Vec<(SlotLease, Instant, Instant)>,
}
impl ReadBatch {
    pub(crate) fn remaining(&self) -> usize {
        self.remaining
    }
    pub(crate) fn poll(&mut self) -> Result<Option<(SlotLease, ReadPayload)>> {
        ensure!(self.remaining == 1, "prefetch must own one pending read");
        match self.receiver.try_recv() {
            Ok(next) => {
                self.remaining = 0;
                Ok(Some((next.lease, next.result?)))
            }
            Err(mpsc::TryRecvError::Empty) => {
                ensure!(
                    Instant::now() < self.deadline,
                    "native SSD prefetch deadline"
                );
                Ok(None)
            }
            Err(error) => Err(error.into()),
        }
    }
    pub(crate) fn ready(&mut self, all: bool) -> Result<Vec<(SlotLease, ReadPayload)>> {
        let mut completed = Vec::new();
        while self.remaining > 0 {
            let next = if completed.is_empty() || all {
                self.receiver
                    .recv_timeout(self.deadline.saturating_duration_since(Instant::now()))
                    .context("native SSD demand deadline or reader failure")?
            } else {
                match self.receiver.try_recv() {
                    Ok(next) => next,
                    Err(mpsc::TryRecvError::Empty) => break,
                    Err(error) => return Err(error.into()),
                }
            };
            self.remaining -= 1;
            #[cfg(test)]
            self.trace.push((next.lease, next.timing.0, next.timing.1));
            completed.push((next.lease, next.result?));
        }
        Ok(completed)
    }
}
impl Drop for ReadBatch {
    fn drop(&mut self) {
        while self.remaining > 0 {
            if self
                .receiver
                .recv_timeout(self.deadline.saturating_duration_since(Instant::now()))
                .is_err()
            {
                break;
            }
            self.remaining -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_support::{DemandPlan, ExpertCache};
    use std::{cell::RefCell, io::Write, rc::Rc};
    fn lease() -> Result<SlotLease> {
        Ok(DemandPlan::reserve(&Rc::new(RefCell::new(ExpertCache::new(10)?)), &[17])?.reads[0])
    }
    #[test]
    fn reads_distinct_owned_ranges_and_drains_eof_before_reuse() -> Result<()> {
        let pool = ReadService::new(4)?;
        let mut file = tempfile::tempfile()?;
        file.write_all(b"0123456789abcdef")?;
        let token = lease()?;
        let mut batch = pool.start(
            vec![
                (token, file.try_clone()?, 2, 5),
                (token, file.try_clone()?, 9, 7),
            ],
            Duration::from_secs(2),
        )?;
        let mut words = batch
            .ready(true)?
            .into_iter()
            .map(|(_, v)| v.legacy_bytes().to_vec())
            .collect::<Vec<_>>();
        words.sort();
        assert_eq!(words, vec![b"23456".to_vec(), b"9abcdef".to_vec()]);
        assert_eq!(batch.remaining(), 0);
        let mut broken = pool.start(
            vec![
                (token, file.try_clone()?, 20, 1),
                (token, file.try_clone()?, 0, 4),
            ],
            Duration::from_secs(2),
        )?;
        assert!(broken.ready(true).is_err());
        drop(broken);
        assert_eq!(pool.in_flight(), 0);
        let mut retry = pool.start(vec![(token, file, 0, 16)], Duration::from_secs(2))?;
        assert_eq!(retry.ready(true)?[0].1.legacy_bytes(), b"0123456789abcdef");
        Ok(())
    }
    #[test]
    fn a_record_uses_the_first_tensor_offset_and_checks_the_full_pinned_extent() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("records.safetensors");
        let bytes = usize::try_from(EXPERT_BYTES)?;
        let header = serde_json::to_vec(&serde_json::json!({
            "first":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},
            "rest":{"dtype":"U8","shape":[bytes-4],"data_offsets":[4,bytes]}
        }))?;
        let mut file = File::create(&path)?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        file.write_all(&vec![71; bytes])?;
        let record = crate::checkpoint::read_header(&path)?["first"].clone();
        assert_eq!(record.data_offsets, [0, 4]);
        let pool = ReadService::new(1)?;
        let mut batch = pool.submit([(lease()?, record.clone())])?;
        let result = batch.ready(true)?;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1.legacy_bytes().len(), bytes);
        assert!(result[0].1.legacy_bytes().iter().all(|&v| v == 71));
        file.set_len((8 + header.len() + bytes - 1) as u64)?;
        assert!(pool.submit([(lease()?, record)]).is_err());
        Ok(())
    }
    #[test]
    fn timeout_preserves_the_real_worker_and_rejects_reuse_until_drain() -> Result<()> {
        let pool = ReadService::new(1)?;
        let mut file = tempfile::tempfile()?;
        file.write_all(b"owned across cancellation")?;
        let gate = Arc::new(std::sync::Barrier::new(2));
        pool.set_test_gate(Some(gate.clone()));
        let mut pending = pool.start(vec![(lease()?, file.try_clone()?, 0, 5)], Duration::ZERO)?;
        let timed_out = pending.ready(true).is_err();
        drop(pending);
        let still_active = pool.in_flight() == 1;
        let refused_reuse = pool.submit(std::iter::empty()).is_err();
        pool.release_test_gate();
        let deadline = Instant::now() + Duration::from_secs(2);
        while pool.in_flight() != 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(timed_out && still_active && refused_reuse);
        assert_eq!(pool.in_flight(), 0);
        let mut retry = pool.start(vec![(lease()?, file, 0, 5)], Duration::from_secs(2))?;
        assert_eq!(retry.ready(true)?[0].1.legacy_bytes(), b"owned");
        Ok(())
    }
    #[test]
    fn refused_allocation_returns_a_failure_and_releases_the_reader() -> Result<()> {
        let pool = ReadService::new(1)?;
        let file = tempfile::tempfile()?;
        let mut batch = pool.start(
            vec![(lease()?, file, 0, usize::MAX)],
            Duration::from_secs(2),
        )?;
        assert!(batch.ready(true).is_err());
        drop(batch);
        assert_eq!(pool.in_flight(), 0);
        Ok(())
    }
    #[test]
    fn four_background_workers_poll_without_blocking_and_recover_after_timeout() -> Result<()> {
        let pool = ReadService::new(4)?;
        let mut file = tempfile::tempfile()?;
        file.write_all(b"owned background bytes")?;
        let gate = Arc::new(std::sync::Barrier::new(5));
        pool.set_test_gate(Some(gate.clone()));
        let token = lease()?;
        let mut pending = (0..4)
            .map(|_| {
                pool.start_mode(
                    vec![(token, file.try_clone()?, 0, 5)],
                    Duration::from_secs(2),
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let active = pool.in_flight();
        let refused = pool
            .start_mode(vec![(token, file.try_clone()?, 0, 5)], Duration::ZERO, true)
            .is_err();
        let polls_empty = pending.iter_mut().all(|r| matches!(r.poll(), Ok(None)));
        pool.release_test_gate();
        assert_eq!(active, 4);
        assert!(refused && polls_empty);
        for read in &mut pending {
            assert_eq!(read.ready(true)?[0].1.legacy_bytes(), b"owned");
        }
        assert_eq!(pool.in_flight(), 0);
        assert!(pool.start_mode(Vec::new(), Duration::ZERO, true).is_err());
        assert!(
            pool.start_mode(
                vec![
                    (token, file.try_clone()?, 0, 5),
                    (token, file.try_clone()?, 0, 5)
                ],
                Duration::ZERO,
                true
            )
            .is_err()
        );
        let gate = Arc::new(std::sync::Barrier::new(2));
        pool.set_test_gate(Some(gate));
        let mut timed = pool.start_mode(vec![(token, file, 0, 5)], Duration::ZERO, true)?;
        let failed = timed.poll().is_err();
        drop(timed);
        let retained = pool.in_flight();
        pool.release_test_gate();
        let deadline = Instant::now() + Duration::from_secs(2);
        while pool.in_flight() != 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(failed && retained == 1 && pool.in_flight() == 0);
        Ok(())
    }
    #[test]
    fn unavailable_cohorts_and_reader_counts_fail_without_io() -> Result<()> {
        for count in [0, 5, usize::MAX] {
            assert!(ReadService::new(count).is_err());
        }
        let pool = ReadService::new(1)?;
        pool.active.store(1, Ordering::Release);
        assert!(pool.start(Vec::new(), Duration::from_millis(1)).is_err());
        pool.active.store(0, Ordering::Release);
        let mut empty = pool.submit(std::iter::empty())?;
        assert!(empty.ready(true)?.is_empty());
        Ok(())
    }
    #[test]
    fn reversed_completions_and_deadlines_keep_ownership() -> Result<()> {
        let token = lease()?;
        let (sender, receiver) = mpsc::channel();
        let mut batch = ReadBatch {
            receiver,
            remaining: 2,
            deadline: Instant::now() + Duration::from_secs(1),
            trace: Vec::new(),
        };
        sender.send(Completed::new(token, Ok(vec![2])))?;
        sender.send(Completed::new(token, Ok(vec![1])))?;
        assert_eq!(
            batch
                .ready(false)?
                .into_iter()
                .map(|(_, v)| v.legacy_bytes().to_vec())
                .collect::<Vec<_>>(),
            vec![vec![2], vec![1]]
        );
        let (sender, receiver) = mpsc::channel();
        let mut timed = ReadBatch {
            receiver,
            remaining: 1,
            deadline: Instant::now(),
            trace: Vec::new(),
        };
        assert!(timed.ready(true).is_err());
        assert_eq!(timed.remaining(), 1);
        sender.send(Completed::new(token, Ok(vec![9])))?;
        drop(timed);
        Ok(())
    }
}

#[cfg(any(test, kani))]
#[path = "ssd_reads/direct_tests.rs"]
mod direct_tests;
