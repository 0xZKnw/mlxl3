//! A single result queue with bounded admission; unstarted records own no new payload.
use super::*;
use std::collections::VecDeque;

fn credits(pending: usize, outstanding: usize) -> Option<usize> {
    if pending > 40 || outstanding > 10 {
        return None;
    }
    Some(pending.min(10 - outstanding))
}

struct Source {
    lease: SlotLease,
    file: File,
    offset: u64,
    bytes: usize,
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    owner: Option<metal::Buffer>,
}

enum Destination {
    Bytes,
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    Shared(metal::Device),
}

pub(crate) struct ReadWindow<'a> {
    pool: &'a ReadService,
    pending: VecDeque<Source>,
    batch: Option<ReadBatch>,
    result: Sender<Completed>,
    destination: Destination,
    failed: bool,
    #[cfg(test)]
    admitted: usize,
}

impl<'a> ReadWindow<'a> {
    fn new(
        pool: &'a ReadService,
        pending: VecDeque<Source>,
        destination: Destination,
        limit: Duration,
    ) -> Result<Self> {
        ensure!(
            pending.len() <= 40
                && pending.iter().all(|source| source.bytes > 0
                    && source.bytes <= EXPERT_BYTES as usize
                    && source.offset.checked_add(source.bytes as u64).is_some()),
            "invalid SSD window sources"
        );
        pool.validate_start(0, false)?;
        let (result, receiver) = mpsc::channel();
        let mut window = Self {
            pool,
            pending,
            result,
            destination,
            failed: false,
            batch: Some(ReadBatch {
                io_cancelled: Arc::new(AtomicBool::new(false)),
                receiver,
                cancelled: Arc::clone(&pool.cancelled),
                draining: Arc::clone(&pool.draining),
                abandoned: false,
                remaining: 0,
                deadline: Instant::now() + limit,
                #[cfg(test)]
                trace: Vec::new(),
                #[cfg(test)]
                wait_started: None,
            }),
            #[cfg(test)]
            admitted: 0,
        };
        pool.window_active.set(true);
        window.refill()?;
        Ok(window)
    }

    fn refill(&mut self) -> Result<()> {
        ensure!(
            !self.failed && self.pool.window_active.get(),
            "failed or unowned SSD window"
        );
        let batch = self.batch.as_mut().context("retired SSD window")?;
        let count =
            credits(self.pending.len(), batch.remaining).context("SSD window credit overflow")?;
        ensure!(
            self.pool.in_flight() <= batch.remaining,
            "foreign SSD work during window"
        );
        // No allocations for queued descriptors. Admission is bounded by complete records,
        // and each window job is whole: split parts never return an early credit.
        let sources = self.pending.drain(..count).collect::<Vec<_>>();
        let reads = sources.into_iter().map(|source| {
            let payload = match &self.destination {
                Destination::Bytes => ReadPayload::Bytes(Vec::new()),
                #[cfg(all(target_os = "macos", feature = "direct-metal"))]
                Destination::Shared(device) => ReadPayload::Shared(self.pool.owned_destination(
                    device,
                    source.owner,
                    source.bytes,
                )?),
            };
            Ok((
                source.lease,
                source.file,
                source.offset,
                source.bytes,
                payload,
            ))
        });
        self.pool
            .enqueue_payloads(batch, &self.result, reads, false, false)?;
        #[cfg(test)]
        {
            self.admitted += count;
        }
        Ok(())
    }

    pub(crate) fn remaining(&self) -> usize {
        self.pending.len() + self.batch.as_ref().map_or(0, |batch| batch.remaining)
    }

    pub(crate) fn ready(&mut self) -> Result<Vec<(SlotLease, ReadPayload)>> {
        ensure!(!self.failed, "SSD window previously failed");
        let result = (|| {
            let completed = self
                .batch
                .as_mut()
                .context("retired SSD window")?
                .ready(false)?;
            self.refill()?;
            Ok(completed)
        })();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}

impl Drop for ReadWindow<'_> {
    fn drop(&mut self) {
        self.pending.clear();
        drop(self.batch.take());
        // Logical ownership ends here. Physical owners still block admission,
        // but must not permanently poison window_active after they drain.
        if self.pool.in_flight() != 0 {
            self.pool.draining.store(true, Ordering::Release);
        }
        self.pool.window_active.set(false);
    }
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
pub(crate) enum BlockReads<'a> {
    Batch(ReadBatch),
    Window(ReadWindow<'a>),
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
impl BlockReads<'_> {
    pub(crate) fn remaining(&self) -> usize {
        match self {
            Self::Batch(batch) => batch.remaining(),
            Self::Window(window) => window.remaining(),
        }
    }
    pub(crate) fn ready(&mut self) -> Result<Vec<(SlotLease, ReadPayload)>> {
        match self {
            Self::Batch(batch) => batch.ready(false),
            Self::Window(window) => window.ready(),
        }
    }
    #[cfg(test)]
    pub(crate) fn trace(&self) -> &[(SlotLease, Instant, Instant)] {
        match self {
            Self::Batch(batch) => &batch.trace,
            Self::Window(window) => &window.batch.as_ref().expect("live window").trace,
        }
    }
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
impl ReadService {
    pub(crate) fn submit_window(
        &self,
        records: impl IntoIterator<Item = (SlotLease, TensorInfo, Option<metal::Buffer>)>,
        device: Option<&metal::DeviceRef>,
    ) -> Result<BlockReads<'_>> {
        let (records, owners): (Vec<_>, Vec<_>) = records
            .into_iter()
            .map(|(lease, record, owner)| ((lease, record), owner))
            .unzip();
        ensure!(records.len() <= 40, "native SSD window exceeds B4 union");
        let pending = self
            .sources(records, device.is_some_and(|d| d.name() == "Apple M5"))?
            .into_iter()
            .zip(owners)
            .map(|((lease, file, offset, bytes), owner)| Source {
                lease,
                file,
                offset,
                bytes,
                owner,
            })
            .collect();
        ReadWindow::new(
            self,
            pending,
            device.map_or(Destination::Bytes, |d| Destination::Shared(d.to_owned())),
            Duration::from_secs(30),
        )
        .map(BlockReads::Window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_support::{DemandPlan, ExpertCache};
    use std::{cell::RefCell, io::Write, rc::Rc};

    fn sources(file: &File, count: usize, bad: Option<usize>) -> Result<VecDeque<Source>> {
        // Each logical test record has an independent real lease.
        (0..count)
            .map(|i| {
                let cache = Rc::new(RefCell::new(ExpertCache::new(10)?));
                let lease = DemandPlan::reserve(&cache, &[i as u32])?.reads[0];
                Ok(Source {
                    lease,
                    file: file.try_clone()?,
                    offset: if bad == Some(i) { 100 } else { i as u64 },
                    bytes: 1,
                    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
                    owner: None,
                })
            })
            .collect()
    }

    #[test]
    fn cancelled_window_can_retry_only_after_physical_drain() -> Result<()> {
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut pool = ReadService::new(1)?;
        pool.set_cancellation(cancelled.clone());
        let held = Arc::new(TestReadGate::default());
        *pool.held_first.borrow_mut() = Some(held.clone());
        let mut file = tempfile::tempfile()?;
        file.write_all(&[42])?;
        let mut window = ReadWindow::new(
            &pool,
            sources(&file, 1, None)?,
            Destination::Bytes,
            Duration::from_secs(2),
        )?;
        cancelled.store(true, Ordering::Release);
        let failed = window.ready().is_err();
        drop(window);
        cancelled.store(false, Ordering::Release);
        let released_owner = !pool.window_active.get();
        let retained = pool.in_flight();
        let refused = pool.start(Vec::new(), Duration::ZERO).is_err();
        held.release();
        *pool.held_first.borrow_mut() = None;
        let until = Instant::now() + Duration::from_secs(2);
        while pool.in_flight() != 0 && Instant::now() < until {
            std::thread::yield_now();
        }
        assert!(failed && released_owner && refused);
        assert_eq!(retained, 1);
        assert_eq!(pool.in_flight(), 0);
        let mut retry = ReadWindow::new(
            &pool,
            sources(&file, 1, None)?,
            Destination::Bytes,
            Duration::from_secs(2),
        )?;
        assert_eq!(retry.ready()?[0].1.legacy_bytes(), &[42]);
        Ok(())
    }

    #[test]
    fn sliding_admission_passes_the_first_tail_and_drains_before_reuse() -> Result<()> {
        for count in [10, 11, 19, 20, 40] {
            let pool = ReadService::new_with_options(4, false, false)?;
            let held = Arc::new(TestReadGate::default());
            *pool.held_first.borrow_mut() = Some(held.clone());
            let mut file = tempfile::tempfile()?;
            file.write_all(&(0..40u8).collect::<Vec<_>>())?;
            let mut window = ReadWindow::new(
                &pool,
                sources(&file, count, None)?,
                Destination::Bytes,
                Duration::from_secs(5),
            )?;
            let outcome = (|| -> Result<()> {
                ensure!(pool.start(Vec::new(), Duration::from_secs(1)).is_err());
                let first = window.ready()?;
                ensure!(!first.is_empty() && window.batch.as_ref().unwrap().remaining <= 10);
                if count > 10 {
                    ensure!(
                        window.admitted > 10,
                        "admission waited for held first record"
                    );
                }
                held.release();
                let mut values = first
                    .into_iter()
                    .map(|(_, value)| value.legacy_bytes()[0])
                    .collect::<Vec<_>>();
                while window.remaining() > 0 {
                    values.extend(
                        window
                            .ready()?
                            .into_iter()
                            .map(|(_, value)| value.legacy_bytes()[0]),
                    );
                    ensure!(pool.in_flight() <= 10);
                }
                values.sort();
                ensure!(values == (0..count as u8).collect::<Vec<_>>());
                Ok(())
            })();
            held.release();
            drop(window);
            outcome?;
            ensure!(pool.in_flight() == 0 && !pool.window_active.get());
            *pool.held_first.borrow_mut() = None;
            let mut broken = ReadWindow::new(
                &pool,
                sources(&file, count, Some(1))?,
                Destination::Bytes,
                Duration::from_secs(5),
            )?;
            let mut failed = false;
            while broken.remaining() > 0 {
                if broken.ready().is_err() {
                    failed = true;
                    break;
                }
            }
            ensure!(failed);
            drop(broken);
            ensure!(pool.in_flight() == 0 && !pool.window_active.get());
            drop(ReadWindow::new(
                &pool,
                sources(&file, count, None)?,
                Destination::Bytes,
                Duration::from_secs(5),
            )?);
            ensure!(pool.in_flight() == 0);
        }
        Ok(())
    }
}

#[cfg(kani)]
#[kani::proof]
fn sliding_credits_bound_whole_jobs_and_pending_descriptors() {
    let pending: usize = kani::any();
    let outstanding: usize = kani::any();
    let admitted = credits(pending, outstanding);
    assert_eq!(admitted.is_some(), pending <= 40 && outstanding <= 10);
    if let Some(n) = admitted {
        assert!(n <= pending && n <= 10 && n + outstanding <= 10);
        assert!((n as u64) * EXPERT_BYTES <= 10 * EXPERT_BYTES);
        if pending > 0 && outstanding < 10 {
            assert!(n > 0);
        }
    }
    kani::cover!(pending == 30 && outstanding == 9 && admitted == Some(1));
    kani::cover!(pending == 40 && outstanding == 0 && admitted == Some(10));
}
