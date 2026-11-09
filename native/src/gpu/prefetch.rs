//! The inference thread owns leases, Metal slots and publication; workers own bytes only.
use super::{FlashExpert, Metal};
use crate::{
    checkpoint::TensorInfo,
    flash_support::{ExpertCache, SlotLease},
    ssd_reads::{ReadBatch, ReadService},
};
use anyhow::{Result, ensure};
use std::{cell::RefCell, rc::Rc};

pub(super) type Slots<'a> = Rc<RefCell<Vec<Option<FlashExpert<'a>>>>>;
#[derive(Clone, Default, serde::Serialize)]
pub(crate) struct PrefetchStats {
    pub submitted: u64,
    pub published: u64,
    pub discarded: u64,
    pub failed: u64,
    pub demand_reused: u64,
}
struct Pending<'a> {
    lease: SlotLease,
    cache: Rc<RefCell<ExpertCache>>,
    slots: Slots<'a>,
    read: ReadBatch,
}
pub(in crate::gpu) struct PrefetchPool<'a> {
    gpu: &'a Metal,
    readers: ReadService,
    pending: Vec<Pending<'a>>,
    pub stats: PrefetchStats,
}
impl<'a> PrefetchPool<'a> {
    pub(in crate::gpu) fn abandon(&mut self) {
        for mut job in self.pending.drain(..) {
            job.cache.borrow_mut().finish_lease(job.lease, false);
            job.read.abandon();
            self.stats.discarded += 1;
        }
    }
    pub fn new(gpu: &'a Metal) -> Result<Self> {
        let mut readers = ReadService::new(4)?;
        readers.set_cancellation(gpu.cancellation_flag());
        Ok(Self {
            gpu,
            readers,
            pending: Vec::new(),
            stats: PrefetchStats::default(),
        })
    }
    #[cfg(test)]
    pub(super) fn set_test_gate(&self, gate: std::sync::Arc<std::sync::Barrier>) {
        self.readers.set_test_gate(Some(gate));
    }
    #[cfg(test)]
    pub(super) fn release_test_gate(&self) {
        self.readers.release_test_gate();
    }
    #[cfg(test)]
    pub(super) fn lease_for(
        &self,
        cache: &Rc<RefCell<ExpertCache>>,
        expert: u32,
    ) -> Option<SlotLease> {
        self.pending
            .iter()
            .find(|job| Rc::ptr_eq(cache, &job.cache) && job.lease.expert == expert)
            .map(|job| job.lease)
    }
    #[cfg(test)]
    pub(super) fn pending_count(&self) -> usize {
        self.pending.len()
    }
    pub fn in_flight(&self) -> usize {
        self.readers.in_flight()
    }
    pub fn room(&self) -> usize {
        crate::flash_support::native_prefetch_room(self.pending.len(), self.in_flight())
            .expect("native prefetch ownership is bounded")
    }
    pub fn submit(
        &mut self,
        lease: SlotLease,
        record: TensorInfo,
        cache: &Rc<RefCell<ExpertCache>>,
        slots: &Slots<'a>,
    ) -> Result<()> {
        ensure!(self.room() > 0, "native prefetch pool is full");
        let read = match self.readers.submit_background(lease, record) {
            Ok(read) => read,
            Err(error) => {
                cache.borrow_mut().finish_lease(lease, false);
                self.stats.failed += 1;
                return Err(error);
            }
        };
        self.pending.push(Pending {
            lease,
            cache: cache.clone(),
            slots: slots.clone(),
            read,
        });
        self.stats.submitted += 1;
        Ok(())
    }
    /// Poll every layer, waiting only for a demanded lease or an explicit drain.
    pub fn poll(
        &mut self,
        demand: Option<&Rc<RefCell<ExpertCache>>>,
        ids: &[u32],
        all: bool,
    ) -> Result<()> {
        if let Err(error) = self.gpu.check_cancelled() {
            self.abandon();
            return Err(error);
        }
        let mut index = 0;
        let mut first_error = None;
        while index < self.pending.len() {
            let job = &mut self.pending[index];
            let needed = demand.is_some_and(|cache| Rc::ptr_eq(cache, &job.cache))
                && ids.contains(&job.lease.expert);
            let result = if all || needed {
                job.read.ready(true).and_then(|mut ready| {
                    ensure!(ready.len() == 1, "invalid native prefetch completion count");
                    Ok(ready.pop())
                })
            } else {
                job.read.poll()
            };
            if matches!(result, Ok(None)) {
                index += 1;
                continue;
            }
            let job = self.pending.remove(index);
            let result = result.and_then(|completed| {
                self.gpu.check_cancelled()?;
                let (lease, payload) = completed.expect("nonempty completion checked above");
                ensure!(lease == job.lease, "native prefetch returned another lease");
                if job.cache.borrow().pending_lease(lease.slot, lease.expert) != Some(lease) {
                    self.stats.discarded += 1;
                    return Ok(());
                }
                let backing = self.gpu.completed_record(payload)?;
                let expert = FlashExpert::from_backing(self.gpu, &backing)?;
                if job.cache.borrow_mut().finish_lease(lease, true) {
                    job.slots.borrow_mut()[lease.slot] = Some(expert);
                    self.stats.published += 1;
                    self.stats.demand_reused += u64::from(needed);
                } else {
                    self.stats.discarded += 1;
                }
                Ok(())
            });
            if let Err(error) = result {
                job.cache.borrow_mut().finish_lease(job.lease, false);
                self.stats.failed += 1;
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
        if let Some(error) = first_error {
            return Err(error);
        }
        Ok(())
    }
    pub fn drain(&mut self) -> Result<()> {
        self.poll(None, &[], true)?;
        ensure!(
            self.in_flight() == 0,
            "native prefetch I/O is still draining"
        );
        Ok(())
    }
}
impl Drop for PrefetchPool<'_> {
    fn drop(&mut self) {
        self.abandon();
        // Cancelled batches detach immediately; workers retain files and bytes until completion.
    }
}
