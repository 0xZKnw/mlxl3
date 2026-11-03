use super::{FlashMtpHead, FlashNext, Metal};
use crate::{
    flash_memory::{ResourcePlan, fits, shrink_slots},
    runtime_stats::MemoryStats,
};
use anyhow::{Context, Result, ensure};
use metal::{Buffer, foreign_types::ForeignType};
use std::collections::{BTreeMap, HashSet};

#[derive(Default)]
pub(super) struct Backings {
    seen: HashSet<usize>,
    bytes: BTreeMap<&'static str, u64>,
    pub slots: Vec<usize>,
    pub resident_experts: usize,
    pub ple_rows: usize,
}
impl Backings {
    pub(super) fn record(&mut self, category: &'static str, buffer: &Buffer) -> Result<()> {
        if self.seen.insert(buffer.as_ptr() as usize) {
            let total = self.bytes.entry(category).or_default();
            *total = total
                .checked_add(buffer.length())
                .context("native backing accounting overflow")?;
        }
        Ok(())
    }
}

#[derive(serde::Serialize)]
pub struct MemoryLedger {
    request_limit_bytes: Option<u64>,
    program_cache: serde_json::Value,
    expert_slots_by_layer: Vec<usize>,
    resident_experts: usize,
    draft_expert_slots: Vec<usize>,
    expert_bank_capacity_bytes: u64,
    expert_bank_payload_bytes: u64,
    expert_bank_backing_bytes: u64,
    fixed_weight_backing_bytes: u64,
    fixed_weight_arrays_not_ready: usize,
    state_backing_bytes: BTreeMap<&'static str, u64>,
    ple_host_row_payload_bytes: usize,
    native_allocated_other_bytes: Option<u64>,
    reads_in_flight: usize,
    prefetch: Option<crate::gpu::moe::PrefetchStats>,
    memory: MemoryStats,
}

impl Metal {
    pub(super) fn ensure_reads_idle(&self) -> Result<()> {
        ensure!(
            self.readers.get().is_none_or(|pool| pool.in_flight() == 0),
            "native SSD reads are still draining"
        );
        Ok(())
    }
    pub(super) fn memory_stats(&self) -> Result<MemoryStats> {
        unsafe extern "C" {
            fn mlxl3_read_process_stats(values: *mut u64, count: usize) -> i32;
        }
        let mut host = [0; 9];
        ensure!(
            unsafe { mlxl3_read_process_stats(host.as_mut_ptr(), host.len()) } == 0,
            "invalid native process statistics extent"
        );
        Ok(MemoryStats::native(
            host,
            self.device.current_allocated_size(),
            std::process::id(),
        ))
    }
    fn observed_memory(&self) -> Result<u64> {
        let stats = self.memory_stats()?;
        Ok(stats
            .process_footprint_bytes
            .context("Flash admission requires process footprint")?
            .max(
                stats
                    .native_metal_allocated_bytes
                    .context("missing native allocations")?,
            ))
    }
    pub(super) fn admit_load(&self, limit: Option<u64>, slots: usize) -> Result<()> {
        if let Some(limit) = limit {
            let reserved = u64::try_from(slots)?
                .checked_mul(crate::flash_support::EXPERT_BYTES)
                .context("Flash expert capacity overflow")?;
            ensure!(
                fits(self.observed_memory()?, reserved, limit),
                "Flash memory budget too small for expert banks"
            );
        }
        Ok(())
    }
}

impl FlashNext<'_> {
    pub fn request_memory_limit(&self) -> Option<u64> {
        self.request_limit
    }
    pub fn memory_ledger(&self, mtp: Option<&FlashMtpHead>) -> Result<MemoryLedger> {
        let mut backings = Backings::default();
        self.head.record_memory(&mut backings)?;
        self.mixer.record_memory(&mut backings)?;
        for layer in &self.layers {
            layer.record_memory(&mut backings)?;
        }
        if let Some(row) = &self.hidden {
            backings.record("last_hidden", &row.buffer)?;
        }
        if let Some(head) = mtp {
            head.record_memory(self, &mut backings)?;
        }
        let owned = backings
            .bytes
            .values()
            .try_fold(0u64, |sum, n| sum.checked_add(*n))
            .context("native live bytes overflow")?;
        let memory = self.memory_stats()?;
        let bank_bytes = backings.bytes.remove("expert_banks").unwrap_or(0);
        let fixed = backings.bytes.remove("fixed_weights").unwrap_or(0);
        let capacity = backings
            .slots
            .iter()
            .try_fold(0u64, |sum, &n| sum.checked_add(n as u64))
            .and_then(|n| n.checked_mul(crate::flash_support::EXPERT_BYTES))
            .context("native expert capacity overflow")?;
        let draft_slots = backings.slots.split_off(self.layers.len());
        Ok(MemoryLedger {
            request_limit_bytes: self.request_limit,
            program_cache: serde_json::json!({"kind":"static_metal_pipelines","hits":null,"creations":null,"evictions":null,"entries":null,"retained_host_object_bytes":null,"cached_tensor_bytes":0}),
            expert_slots_by_layer: backings.slots,
            resident_experts: backings.resident_experts,
            draft_expert_slots: draft_slots,
            expert_bank_capacity_bytes: capacity,
            expert_bank_payload_bytes: backings.resident_experts as u64
                * crate::flash_support::EXPERT_BYTES,
            expert_bank_backing_bytes: bank_bytes,
            fixed_weight_backing_bytes: fixed,
            fixed_weight_arrays_not_ready: 0,
            state_backing_bytes: backings.bytes,
            ple_host_row_payload_bytes: backings.ple_rows,
            native_allocated_other_bytes: memory
                .native_metal_allocated_bytes
                .and_then(|n| n.checked_sub(owned)),
            prefetch: self.prefetcher.as_ref().map(|p| p.borrow().stats.clone()),
            reads_in_flight: self
                .prefetcher
                .as_ref()
                .map_or(0, |p| p.borrow().in_flight())
                + self
                    .gpu
                    .readers
                    .get()
                    .map_or(0, crate::ssd_reads::ReadService::in_flight),
            memory,
        })
    }
    pub fn release_idle_state(&mut self) -> Result<()> {
        self.gpu.ensure_reads_idle()?;
        self.gpu.command(|_| Ok(()))?;
        self.reset();
        for layer in &mut self.layers {
            layer.release_idle_state()?;
        }
        Ok(())
    }
    pub fn admit_request(&mut self, positions: usize) -> Result<Option<ResourcePlan>> {
        let Some(limit) = self.request_limit else {
            return Ok(None);
        };
        let plan = ResourcePlan::native_ar(u32::try_from(positions)?, self.context)
            .context("invalid Flash request length")?;
        self.release_idle_state()?;
        let mut ledger = self.memory_ledger(None)?;
        let minimum_cache = self.layers.len() as u64 * 10 * crate::flash_support::EXPERT_BYTES;
        let mut base = self.gpu.observed_memory()?;
        ensure!(
            fits(
                base.saturating_sub(ledger.expert_bank_backing_bytes),
                plan.extra_bytes()
                    .checked_add(minimum_cache)
                    .context("Flash admission overflow")?,
                limit
            ),
            "Flash request exceeds memory budget at minimum cache"
        );
        let mut extra = plan
            .extra_bytes()
            .checked_add(
                ledger
                    .expert_bank_capacity_bytes
                    .saturating_sub(ledger.expert_bank_backing_bytes),
            )
            .context("Flash admission overflow")?;
        for index in 0..self.layers.len() {
            if fits(base, extra, limit) {
                break;
            }
            let capacity = ledger.expert_slots_by_layer[index];
            let slots = shrink_slots(capacity, base.saturating_add(extra).saturating_sub(limit));
            if slots < capacity {
                self.layers[index].shrink(slots)?;
                ledger = self.memory_ledger(None)?;
                base = self.gpu.observed_memory()?;
                extra = plan
                    .extra_bytes()
                    .checked_add(
                        ledger
                            .expert_bank_capacity_bytes
                            .saturating_sub(ledger.expert_bank_backing_bytes),
                    )
                    .context("Flash admission overflow")?;
            }
        }
        ensure!(
            fits(base, extra, limit),
            "Flash request exceeds measured memory budget"
        );
        Ok(Some(plan))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn same_states(
        expected: &[crate::gpu::FlashTraceTensor],
        actual: &[crate::gpu::FlashTraceTensor],
    ) -> Result<()> {
        ensure!(expected.len() == 84 && actual.len() == 84);
        let mut words = 0usize;
        for (before, after) in expected.iter().zip(actual) {
            let count = after
                .shape
                .iter()
                .try_fold(1usize, |count, &n| count.checked_mul(n as usize))
                .context("trace shape overflow")?;
            ensure!(count > 0 && after.data.len() == count * 4);
            ensure!(
                after
                    .data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|&bytes| f32::from_le_bytes(bytes).is_finite())
            );
            ensure!(
                before.name == after.name
                    && before.shape == after.shape
                    && before.data == after.data,
                "state replay differs: {}",
                after.name
            );
            words += count;
        }
        ensure!(words == 28_803_072);
        Ok(())
    }

    #[test]
    #[ignore = "physical Flash model, unique backings, snapshot aliases, budget shrink and recovery"]
    fn native_ledger_tracks_aliases_and_request_shrink_preserves_replay() -> Result<()> {
        let path = std::env::var("MLXL3_FLASH_MODEL")?;
        let c = crate::checkpoint::inspect(Path::new(&path))?;
        let gpu = Metal::new()?;
        let mut target = FlashNext::load(&gpu, &c, 4096)?;
        let initial = target.memory_ledger(None)?;
        ensure!(
            initial.expert_slots_by_layer.len() == 48
                && initial.expert_slots_by_layer.iter().any(|&n| n > 10)
        );
        ensure!(
            initial.resident_experts == 0
                && initial.expert_bank_backing_bytes == 0
                && initial.fixed_weight_backing_bytes > 0
        );
        let once =
            gpu.f32_storage(&[1., 2., 3., 4.])?
                .matrix(crate::tensor_layout::MatrixLayout {
                    shape: [1, 4],
                    strides: [4, 1],
                    offset: 0,
                })?;
        let alias = once.clone();
        let mut counted = Backings::default();
        counted.record("first", &once.buffer)?;
        counted.record("alias", &alias.buffer)?;
        ensure!(counted.bytes.len() == 1 && counted.bytes["first"] == once.buffer.length());
        let logits = target.forward(248044)?;
        ensure!(logits.len() == 248320);
        let snapshot = target.snapshot();
        let (next, before_states) = target.forward_trace(1)?;
        ensure!(
            logits
                .iter()
                .chain(&next)
                .all(|&bits| half::f16::from_bits(bits).is_finite())
        );
        let active = target.memory_ledger(None)?;
        ensure!(
            active.resident_experts > 0
                && active.expert_bank_payload_bytes == active.expert_bank_backing_bytes
        );
        ensure!(
            active.state_backing_bytes["gdn"] > 0 && active.state_backing_bytes["last_hidden"] > 0
        );
        target.release_idle_state()?;
        let idle = target.memory_ledger(None)?;
        ensure!(
            !idle.state_backing_bytes.contains_key("gdn")
                && !idle.state_backing_bytes.contains_key("last_hidden")
        );
        target.restore(&snapshot)?;
        let (replayed, replayed_states) = target.forward_trace(1)?;
        ensure!(replayed == next, "snapshot was changed by idle release");
        same_states(&before_states, &replayed_states)?;
        drop(replayed_states);
        drop(snapshot);
        target.release_idle_state()?;
        let before = target.memory_ledger(None)?;
        let plan = ResourcePlan::native_ar(64, target.context).unwrap();
        let minimum_cache = 48 * 10 * crate::flash_support::EXPERT_BYTES;
        let limit = gpu
            .observed_memory()?
            .checked_sub(before.expert_bank_backing_bytes)
            .and_then(|n| n.checked_add(plan.extra_bytes()))
            .and_then(|n| n.checked_add(minimum_cache))
            .and_then(|n| n.checked_add(64 << 20))
            .context("invalid budget fixture")?;
        target.request_limit = Some(limit);
        ensure!(target.admit_request(64)?.is_some());
        let shrunk = target.memory_ledger(None)?;
        ensure!(
            shrunk
                .expert_slots_by_layer
                .iter()
                .zip(&before.expert_slots_by_layer)
                .all(|(&a, &b)| a >= 10 && a <= b)
        );
        ensure!(
            shrunk.expert_slots_by_layer != before.expert_slots_by_layer,
            "budget never exercised shrink"
        );
        let capacities = shrunk.expert_slots_by_layer.clone();
        target.request_limit = Some(1);
        ensure!(target.admit_request(64).is_err());
        ensure!(
            target.memory_ledger(None)?.expert_slots_by_layer == capacities,
            "impossible request evicted extra cache"
        );
        ensure!(target.admit_request(0).is_err() && target.admit_request(4097).is_err());
        ensure!(
            FlashMtpHead::load(&c, &target).is_err(),
            "budget admitted speculative head"
        );
        target.request_limit = None;
        target.reset();
        ensure!(
            target.forward(248044)? == logits,
            "admission changed first token replay"
        );
        let (after, after_states) = target.forward_trace(1)?;
        ensure!(after == next, "admission changed next token replay");
        same_states(&before_states, &after_states)?;
        eprintln!(
            "native memory ledger {}",
            serde_json::json!({"initial":initial,"active":active,"idle":idle,"before_shrink":before,"after_shrink":shrunk})
        );
        Ok(())
    }
}
