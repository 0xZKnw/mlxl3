//! Bounded owned disk reads. No GPU pointer crosses the worker boundary.
use crate::{
    checkpoint::TensorInfo,
    flash_support::{EXPERT_BYTES, SlotLease},
};
use anyhow::{Context, Result, ensure};
use std::{
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

pub(crate) struct Completed {
    lease: SlotLease,
    result: Result<Vec<u8>>,
    #[cfg(test)]
    timing: (Instant, Instant),
}
#[cfg(test)]
impl Completed {
    fn new(lease: SlotLease, result: Result<Vec<u8>>) -> Self {
        Self {
            lease,
            result,
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
    result: Sender<Completed>,
    ticket: Ticket,
    #[cfg(test)]
    gate: Option<Arc<std::sync::Barrier>>,
}
impl Read {
    fn run(self) {
        #[cfg(test)]
        let started = Instant::now();
        #[cfg(test)]
        if let Some(gate) = self.gate {
            gate.wait();
        }
        let result = (|| -> Result<Vec<u8>> {
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(self.bytes)?;
            bytes.resize(self.bytes, 0);
            self.file.read_exact_at(&mut bytes, self.offset)?;
            Ok(bytes)
        })();
        drop(self.ticket);
        let result = Completed {
            lease: self.lease,
            result,
            #[cfg(test)]
            timing: (started, Instant::now()),
        };
        let _ = self.result.send(result);
    }
}

pub(crate) struct ReadService {
    workers: Vec<(SyncSender<Read>, JoinHandle<()>)>,
    active: Arc<AtomicUsize>,
    #[cfg(test)]
    gate: std::cell::RefCell<Option<Arc<std::sync::Barrier>>>,
}
impl ReadService {
    pub(crate) fn new(readers: usize) -> Result<Self> {
        ensure!((1..=4).contains(&readers), "SSD readers must be in 1..=4");
        let mut service = Self {
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
    pub(crate) fn submit(
        &self,
        records: impl IntoIterator<Item = (SlotLease, TensorInfo)>,
    ) -> Result<ReadBatch> {
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
        self.start(reads, Duration::from_secs(30))
    }
    fn start(
        &self,
        reads: Vec<(SlotLease, File, u64, usize)>,
        limit: Duration,
    ) -> Result<ReadBatch> {
        ensure!(reads.len() <= 10, "native SSD demand exceeds one AR cohort");
        ensure!(
            self.active.load(Ordering::Acquire) == 0,
            "previous native SSD reads are still draining"
        );
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
        for (index, (lease, file, offset, bytes)) in reads.into_iter().enumerate() {
            self.active.fetch_add(1, Ordering::Relaxed);
            let read = Read {
                lease,
                file,
                offset,
                bytes,
                result: result.clone(),
                ticket: Ticket(self.active.clone()),
                #[cfg(test)]
                gate: (index + 1 == count)
                    .then(|| self.gate.borrow().clone())
                    .flatten(),
            };
            self.workers[index % self.workers.len()]
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
    pub(crate) fn ready(&mut self, all: bool) -> Result<Vec<(SlotLease, Vec<u8>)>> {
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
            .map(|(_, v)| v)
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
        assert_eq!(pool.active.load(Ordering::Acquire), 0);
        let mut retry = pool.start(vec![(token, file, 0, 16)], Duration::from_secs(2))?;
        assert_eq!(retry.ready(true)?[0].1, b"0123456789abcdef");
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
        assert_eq!(result[0].1.len(), bytes);
        assert!(result[0].1.iter().all(|&v| v == 71));
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
        let still_active = pool.active.load(Ordering::Acquire) == 1;
        let refused_reuse = pool.submit(std::iter::empty()).is_err();
        pool.release_test_gate();
        let deadline = Instant::now() + Duration::from_secs(2);
        while pool.active.load(Ordering::Acquire) != 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(timed_out && still_active && refused_reuse);
        assert_eq!(pool.active.load(Ordering::Acquire), 0);
        let mut retry = pool.start(vec![(lease()?, file, 0, 5)], Duration::from_secs(2))?;
        assert_eq!(retry.ready(true)?[0].1, b"owned");
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
        assert_eq!(pool.active.load(Ordering::Acquire), 0);
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
                .map(|(_, v)| v)
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
