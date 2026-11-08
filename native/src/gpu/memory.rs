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
    fixed: Option<Vec<Buffer>>,
}
impl Backings {
    pub(super) fn collect_fixed() -> Self {
        Self {
            fixed: Some(Vec::new()),
            ..Self::default()
        }
    }
    pub(super) fn take_fixed(self) -> Result<(Vec<Buffer>, u64)> {
        Ok((
            self.fixed
                .context("fixed-weight collection was not enabled")?,
            self.bytes.get("fixed_weights").copied().unwrap_or(0),
        ))
    }
    pub(super) fn total(&self) -> Result<u64> {
        self.bytes
            .values()
            .try_fold(0u64, |sum, &bytes| sum.checked_add(bytes))
            .context("native backing total overflow")
    }
    pub(super) fn record(&mut self, category: &'static str, buffer: &Buffer) -> Result<()> {
        if self.seen.insert(buffer.as_ptr() as usize) {
            if category == "fixed_weights"
                && let Some(buffers) = &mut self.fixed
            {
                buffers.push(buffer.clone());
            }
            let total = self.bytes.entry(category).or_default();
            *total = total
                .checked_add(buffer.length())
                .context("native backing accounting overflow")?;
        }
        Ok(())
    }
}

pub(super) struct DraftBudget {
    total: std::rc::Rc<std::cell::Cell<u64>>,
    charged: u64,
}
impl DraftBudget {
    pub(super) fn new(total: &std::rc::Rc<std::cell::Cell<u64>>, context: u32) -> Result<Self> {
        let charged = crate::native_executor::draft_capacity_bytes(context)
            .context("invalid draft capacity forecast")?;
        let next = crate::native_executor::draft_reserve_after(total.get(), charged, true)
            .context("draft bank reserve overflow")?;
        total.set(next);
        Ok(Self {
            total: total.clone(),
            charged,
        })
    }
}
impl Drop for DraftBudget {
    fn drop(&mut self) {
        self.total.set(
            crate::native_executor::draft_reserve_after(self.total.get(), self.charged, false)
                .expect("balanced draft bank reserve"),
        );
    }
}

#[derive(serde::Serialize)]
pub struct MemoryLedger {
    request_limit_bytes: Option<u64>,
    prospective_draft_capacity_reserve_bytes: u64,
    program_cache: serde_json::Value,
    expert_slots_by_layer: Vec<usize>,
    resident_experts: usize,
    draft_expert_slots: Vec<usize>,
    expert_bank_capacity_bytes: u64,
    expert_bank_payload_bytes: u64,
    expert_bank_backing_bytes: u64,
    fixed_weight_backing_bytes: u64,
    fixed_weight_arrays_not_ready: usize,
    resident_scratch_charge_bytes: u64,
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
            let mut reserved = u64::try_from(slots)?
                .checked_mul(crate::flash_support::EXPERT_BYTES)
                .context("Flash expert capacity overflow")?;
            if self.resident_scratch.enabled.get() || self.resident_scratch.charged() > 0 {
                reserved = reserved
                    .checked_add(crate::resident_scratch_policy::CAP)
                    .context("resident scratch admission overflow")?;
            }
            if self.device_resident_route_enabled() {
                reserved = reserved
                    .checked_add(crate::device_hot_policy::RESERVE_BYTES)
                    .context("resident route admission overflow")?;
            }
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
    /// Legacy paths remain AR-only under a budget; common transactions admit before allocation.
    pub fn speculation_permitted(&self) -> bool {
        self.request_limit.is_none() || self.gpu.common_executor
    }
    pub(super) fn admit_draft_load(
        &self,
        checkpoint: &crate::checkpoint::Checkpoint,
    ) -> Result<()> {
        let Some(limit) = self.request_limit else {
            return Ok(());
        };
        ensure!(
            self.gpu.common_executor,
            "budgeted draft requires the common executor"
        );
        let weights = checkpoint
            .tensors
            .iter()
            .filter(|(name, _)| name.starts_with("mtp."))
            .try_fold(0u64, |sum, (_, tensor)| {
                sum.checked_add(tensor.data_offsets[1].checked_sub(tensor.data_offsets[0])?)
            })
            .context("draft weight forecast overflow")?;
        let ledger = self.memory_ledger(None)?;
        let reserve = weights
            .checked_add(
                crate::native_executor::draft_capacity_bytes(self.context)
                    .context("invalid draft capacity forecast")?,
            )
            .and_then(|n| n.checked_add(self.draft_capacity_reserve.get()))
            .and_then(|n| {
                n.checked_add(
                    ledger
                        .expert_bank_capacity_bytes
                        .saturating_sub(ledger.expert_bank_backing_bytes),
                )
            })
            .context("draft load reserve overflow")?;
        ensure!(
            fits(self.gpu.observed_memory()?, reserve, limit),
            "Flash memory budget cannot admit draft weights and expert banks"
        );
        Ok(())
    }
    pub(super) fn native_speculation_plan(&self) -> Result<crate::speculation_plan::MemoryPlan> {
        let ledger = self.memory_ledger(None)?;
        let prefixes = self
            .layer_plans
            .iter()
            .try_fold(0u64, |sum, plan| sum.checked_add(plan.prefix_bytes()))
            .context("native prefix reserve overflow")?;
        // Conservative scratch forecast, not an allocation or an observed peak.
        let projection = (2560u64 * 4 + 640 * 4 + 2560 * 4 + 2560 * 2) * 10;
        let attention = u64::from(self.context)
            .checked_mul(24 * 256 * 2)
            .context("attention reserve overflow")?;
        let scratch = attention
            .checked_add(projection)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(248320 * 8))
            .context("verification scratch reserve overflow")?;
        Ok(crate::speculation_plan::MemoryPlan {
            base_bytes: self.gpu.observed_memory()?,
            reserve_bytes: ledger
                .expert_bank_capacity_bytes
                .saturating_sub(ledger.expert_bank_backing_bytes)
                .checked_add(self.draft_capacity_reserve.get())
                .context("draft bank forecast overflow")?,
            verify_scratch_bytes: scratch,
            prefix_state_bytes_per_row: prefixes,
            retained_expert_generations_bytes: 40 * crate::flash_support::EXPERT_BYTES
                + if self.gpu.device_resident_route_enabled() {
                    4 * crate::device_hot_policy::RESERVE_BYTES
                } else {
                    0
                },
            ..Default::default()
        })
    }
    pub fn admitted_mtp_depth(
        &self,
        requested: usize,
        context_remaining: usize,
        output_remaining: usize,
    ) -> Result<usize> {
        if !self.speculation_permitted() {
            return Ok(0);
        }
        let Some(limit) = self.request_limit else {
            return Ok(requested
                .min(3)
                .min(context_remaining.saturating_sub(1))
                .min(output_remaining.saturating_sub(1)));
        };
        Ok(self
            .native_speculation_plan()?
            .admitted_depth(requested, context_remaining, output_remaining, limit)
            .unwrap_or(0))
    }
    pub(super) fn admit_native_block(&self, rows: usize) -> Result<()> {
        ensure!((1..=4).contains(&rows), "invalid native block row count");
        if let Some(limit) = self.request_limit {
            ensure!(
                self.gpu.common_executor,
                "Flash request memory budget supports AR only"
            );
            let plan = self.native_speculation_plan()?;
            ensure!(
                plan.peak_bytes(rows.saturating_sub(1).max(1))
                    .is_some_and(|peak| peak <= limit),
                "Flash request memory budget cannot retain the requested block"
            );
        }
        Ok(())
    }
    pub fn memory_ledger(&self, mtp: Option<&FlashMtpHead>) -> Result<MemoryLedger> {
        self.memory_ledger_with_prefix(mtp, None, None)
    }
    pub fn memory_ledger_with_prefix(
        &self,
        mtp: Option<&FlashMtpHead>,
        prefix: Option<&crate::gpu::FlashSnapshot>,
        mtp_prefix: Option<&crate::gpu::FlashQsaState>,
    ) -> Result<MemoryLedger> {
        let mut backings = Backings::default();
        ensure!(
            self.gpu.resident_scratch.valid(),
            "resident scratch charge underflow"
        );
        self.head.record_memory(&mut backings)?;
        self.mixer.record_memory(&mut backings)?;
        self.record_embedding_memory(&mut backings)?;
        for layer in &self.layers {
            layer.record_memory(&mut backings)?;
        }
        if let Some(row) = &self.hidden {
            backings.record("last_hidden", &row.buffer)?;
        }
        if let Some(head) = mtp {
            head.record_memory(self, &mut backings)?;
        }
        if let Some(prefix) = prefix {
            ensure!(
                self.owns_snapshot(prefix),
                "prompt cache belongs to a different Flash session"
            );
            prefix.record_memory(&mut backings)?;
        }
        if let Some(prefix) = mtp_prefix {
            ensure!(
                mtp.is_some_and(|head| head.owns_prefix(prefix)),
                "cached MTP state belongs to another head"
            );
            prefix.record_memory(&mut backings, "prompt_cache")?;
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
            prospective_draft_capacity_reserve_bytes: self.draft_capacity_reserve.get(),
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
            resident_scratch_charge_bytes: self.gpu.resident_scratch.charged(),
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
        if self.fixed_residency.is_some() {
            self.gpu.command(|_| Ok(()))?;
            self.fixed_residency = None;
        }
        self.fixed_residency_checked = false;
        self.reset();
        self.head.release_idle_scratch()?;
        self.mixer.release_workspace()?;
        for layer in &mut self.layers {
            layer.release_idle_state()?;
        }
        self.gpu.clear_hot_workspace()?;
        Ok(())
    }
    pub fn admit_request(&mut self, positions: usize) -> Result<Option<ResourcePlan>> {
        let Some(limit) = self.request_limit else {
            return Ok(None);
        };
        let mut plan = ResourcePlan::native_ar(u32::try_from(positions)?, self.context)
            .context("invalid Flash request length")?;
        if self.gpu.resident_scratch.enabled.get() || self.gpu.resident_scratch.charged() > 0 {
            plan = plan
                .with_resident_scratch()
                .context("resident scratch reserve overflow")?;
        }
        if self.gpu.device_resident_route_enabled() {
            plan = plan
                .with_device_hot()
                .context("resident route reserve overflow")?;
        }
        if self.gpu.hc_workspace_enabled.get() {
            plan = plan
                .with_hc_workspace(self.layers.len() as u64 * 2 + 1, !self.gpu.hc_up_mix.get())
                .context("HC workspace admission overflow")?;
        }
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
        if gpu.hc_workspace_enabled.get() {
            let expected = if gpu.hc_up_mix.get() {
                4_221_440
            } else {
                8_194_560
            };
            ensure!(active.state_backing_bytes.get("hc_workspace") == Some(&expected));
        }
        target.release_idle_state()?;
        let idle = target.memory_ledger(None)?;
        ensure!(
            !idle.state_backing_bytes.contains_key("gdn")
                && !idle.state_backing_bytes.contains_key("last_hidden")
                && !idle.state_backing_bytes.contains_key("hc_workspace")
        );
        target.restore(&snapshot)?;
        let (replayed, replayed_states) = target.forward_trace(1)?;
        ensure!(replayed == next, "snapshot was changed by idle release");
        same_states(&before_states, &replayed_states)?;
        drop(replayed_states);
        drop(snapshot);
        target.release_idle_state()?;
        let before = target.memory_ledger(None)?;
        let mut plan = ResourcePlan::native_ar(64, target.context).unwrap();
        if gpu.device_resident_route_enabled() {
            plan = plan.with_device_hot().unwrap();
            ensure!(
                serde_json::to_value(plan)?["device_hot_reserve_bytes"]
                    == crate::device_hot_policy::RESERVE_BYTES
            );
        }
        if gpu.hc_workspace_enabled.get() {
            let with_up = !gpu.hc_up_mix.get();
            plan = plan.with_hc_workspace(97, with_up).unwrap();
            let expected = if with_up { 8_194_560 } else { 4_221_440 };
            ensure!(serde_json::to_value(plan)?["hc_workspace_reserve_bytes"] == expected);
        }
        let minimum_cache = 48 * 10 * crate::flash_support::EXPERT_BYTES;
        let limit = gpu
            .observed_memory()?
            .checked_sub(before.expert_bank_backing_bytes)
            .and_then(|n| n.checked_add(plan.extra_bytes()))
            .and_then(|n| n.checked_add(minimum_cache))
            .and_then(|n| n.checked_add(64 << 20))
            .context("invalid budget fixture")?;
        target.request_limit = Some(limit);
        let warm_admission = target.admit_request(64);
        let warm_shrunk = target.memory_ledger(None)?;
        eprintln!(
            "native memory warm admission {}",
            serde_json::json!({"limit":limit,"plan_extra":plan.extra_bytes(),"before":before,"after":warm_shrunk,"measured_after":gpu.observed_memory()?,"error":warm_admission.as_ref().err().map(|e|format!("{e:#}"))})
        );
        ensure!(warm_shrunk.expert_bank_backing_bytes < before.expert_bank_backing_bytes);
        if let Err(error) = warm_admission {
            ensure!(
                error
                    .to_string()
                    .contains("Flash request exceeds measured memory budget"),
                "unexpected warm-bank refusal: {error:#}"
            );
        }
        // A logical bank byte need not reduce process footprint by one byte. Use
        // unpopulated reservations for the guaranteed successful admission case.
        drop(target);
        let mut target = FlashNext::load(&gpu, &c, 4096)?;
        let before_fresh = target.memory_ledger(None)?;
        ensure!(
            before_fresh.expert_bank_backing_bytes == 0
                && before_fresh.expert_bank_capacity_bytes > minimum_cache
        );
        let limit_fresh = gpu
            .observed_memory()?
            .checked_add(plan.extra_bytes())
            .and_then(|n| n.checked_add(minimum_cache))
            .and_then(|n| {
                n.checked_add((before_fresh.expert_bank_capacity_bytes - minimum_cache) / 2)
            })
            .context("fresh reservation budget overflow")?;
        target.request_limit = Some(limit_fresh);
        let admitted = target.admit_request(64)?;
        ensure!(admitted.is_some());
        eprintln!(
            "native memory fresh admission {}",
            serde_json::json!({"limit":limit_fresh,"plan_extra":plan.extra_bytes(),"before":before_fresh,"after":target.memory_ledger(None)?})
        );
        let shrunk = target.memory_ledger(None)?;
        ensure!(
            shrunk
                .expert_slots_by_layer
                .iter()
                .zip(&before_fresh.expert_slots_by_layer)
                .all(|(&a, &b)| a >= 10 && a <= b)
        );
        ensure!(
            shrunk.expert_slots_by_layer != before_fresh.expert_slots_by_layer,
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
        #[cfg(feature = "chat")]
        ensure!(
            crate::native_draft::Session::load(
                std::path::Path::new("/a/nonexistent/draft/that/must/not/be/opened"),
                1,
                &target
            )
            .err()
            .is_some_and(|error| error.to_string().contains("AR only")),
            "budget reached native draft materialization"
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
            serde_json::json!({"initial":initial,"active":active,"idle":idle,"warm_before":before,"warm_after":warm_shrunk,"before_shrink":before_fresh,"after_shrink":shrunk,"success_uses_fresh_unpopulated_reservations":true})
        );
        Ok(())
    }
}

#[cfg(test)]
#[test]
#[ignore = "physical Flash checkpoint; common executor request and draft admission"]
fn physical_common_budget_admits_before_mutation_and_releases_draft_capacity() -> Result<()> {
    let checkpoint =
        crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    ensure!(gpu.common_executor, "common executor must be enabled");
    let mut target = FlashNext::load(&gpu, &checkpoint, 64)?;
    target.prefill(&[248044, 78768, 11])?;
    let before = target.snapshot();
    let hidden = target
        .last_hidden()
        .context("missing target hidden")?
        .export_bytes(&gpu)?;
    ensure!(!hidden.is_empty() && target.draft_capacity_reserve.get() == 0);
    target.request_limit = Some(1);
    ensure!(target.admitted_mtp_depth(3, 61, 16)? == 0);
    ensure!(target.stage_block(&[1, 2, 3, 4]).is_err());
    ensure!(FlashMtpHead::load(&checkpoint, &target).is_err());
    ensure!(
        target.offset() == before.position()
            && target.draft_capacity_reserve.get() == 0
            && target.last_hidden().unwrap().export_bytes(&gpu)? == hidden
    );
    target.request_limit = Some(24 * 1024u64.pow(3));
    ensure!(target.admitted_mtp_depth(3, 61, 16)? == 3);
    drop(target.stage_block(&[1, 2, 3, 4])?);
    ensure!(
        target.offset() == before.position()
            && target.last_hidden().unwrap().export_bytes(&gpu)? == hidden
    );
    let head = FlashMtpHead::load(&checkpoint, &target)?;
    ensure!(
        target.draft_capacity_reserve.get()
            == crate::native_executor::draft_capacity_bytes(64).unwrap()
    );
    let ledger = target.memory_ledger(Some(&head))?;
    ensure!(ledger.prospective_draft_capacity_reserve_bytes == target.draft_capacity_reserve.get());
    drop(head);
    ensure!(target.draft_capacity_reserve.get() == 0);
    Ok(())
}
