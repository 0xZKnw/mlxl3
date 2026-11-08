//! Opt-in block SSD reuse. Every wave is complete before cache slots are reused.
//! Expert arithmetic and each row's rank-ordered mix remain scalar/canonical.
use super::*;
use crate::speculation_plan::{MAX_ROWS, TOP_K, resident_expert_cohorts};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
thread_local! {
    // One-shot deterministic cancellation after a genuinely submitted/waited wave.
    static CANCEL_AFTER_WAVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static CANCEL_AFTER_READ_BATCH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl FlashMoeRow<'_> {
    /// Caller checks this before preparing any stateful row. Rechecked below.
    pub(in crate::gpu) fn block_eligible(&self) -> bool {
        self.prefetcher.is_none()
            && !self.gpu.device_resident_route_enabled()
            && !self.gpu.hc_workspace_enabled.get()
            && !self.gpu.moe_deferred.get()
            && !self.gpu.moe_finish_mix.get()
            && self.gpu.moe_switch.iter().all(|switch| {
                switch.thread_execution_width() == 32
                    && switch.max_total_threads_per_threadgroup() >= 512
            })
    }

    /// Read each missing unique expert once, using it for every row consumer.
    /// Outputs are already GPU-complete; no record generations escape this call.
    /// `read_bytes` is charged to an expert's first consumer only (sum over rows).
    #[cfg(test)]
    pub(in crate::gpu) fn forward_block(&mut self, rows: &[MatrixView]) -> Result<Vec<MoeOutput>> {
        self.forward_block_cancellable(rows, &AtomicBool::new(false))
    }

    pub(in crate::gpu) fn forward_block_cancellable(
        &mut self,
        rows: &[MatrixView],
        cancelled: &AtomicBool,
    ) -> Result<Vec<MoeOutput>> {
        let check_cancelled = || -> Result<()> {
            ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
            Ok(())
        };
        check_cancelled()?;
        ensure!(self.block_eligible(), "unsupported Flash block SSD mode");
        ensure!(
            (1..=MAX_ROWS).contains(&rows.len()),
            "Flash SSD block expects 1..4 rows"
        );
        for x in rows {
            ensure!(
                x.shape() == [1, 2560]
                    && x.dtype == rows[0].dtype
                    && x.layout.strides[1] == 1
                    && x.layout.offset >= 0
                    && x.buffer.device().registry_id() == self.gpu.device.registry_id(),
                "invalid Flash SSD block row"
            );
            let switch = &self.gpu.moe_switch[usize::from(x.dtype == FloatType::F32)];
            ensure!(
                switch.thread_execution_width() == 32
                    && switch.max_total_threads_per_threadgroup() >= 512,
                "unsupported Flash shared expert switch threadgroup"
            );
        }
        self.drain_commands()?;
        let mut routed = Vec::with_capacity(rows.len());
        let mut ids = Vec::with_capacity(rows.len());
        // RouteOutput has no draining Drop. Finish each submitted route before
        // another fallible submission, including error/cancellation paths.
        for x in rows {
            check_cancelled()?;
            let route = self.gate.submit(x)?;
            ids.push(route.indices()?);
            routed.push(route);
        }
        let routes = ids
            .iter()
            .map(|ids| -> Result<[u16; TOP_K]> {
                ensure!(
                    ids.len() == TOP_K && ids.iter().all(|&id| id < 512),
                    "invalid block routes"
                );
                Ok(std::array::from_fn(|rank| ids[rank] as u16))
            })
            .collect::<Result<Vec<_>>>()?;
        let (width, resident) = {
            let cache = self.cache.borrow();
            (cache.available_slots(), cache.ready_slot_snapshot())
        };
        let resident = resident
            .iter()
            .flatten()
            .map(|&(id, _)| id as u16)
            .collect::<Vec<_>>();
        let initially_hot = |expert: u16| resident.contains(&expert);
        let cohorts = resident_expert_cohorts(&routes, width, &resident)
            .context("invalid Flash block expert cohort")?;
        // Consume all initially resident experts in the first wave before any
        // later reservation can evict them. Rank ordering is independent of this.
        let hits = routes
            .iter()
            .map(|row| row.iter().filter(|&&id| initially_hot(id)).count())
            .collect::<Vec<_>>();
        let mut read_bytes = vec![0u64; rows.len()];
        let mut contributions: Vec<[Option<Buffer>; TOP_K]> = (0..rows.len())
            .map(|_| std::array::from_fn(|_| None))
            .collect();
        #[cfg(test)]
        let mut timelines = vec![Vec::new(); rows.len()];
        let guard = ExpertFill::new(&self.cache);
        for wave in &cohorts.waves {
            check_cancelled()?;
            let cohort = &cohorts.cohorts[wave.clone()];
            let requested = cohort
                .iter()
                .map(|entry| u32::from(entry.expert))
                .collect::<Vec<_>>();
            let plan = DemandPlan::reserve(&self.cache, &requested)?;
            // No submitted command still refers to these slot generations.
            for lease in &plan.reads {
                self.slots.borrow_mut()[lease.slot] = None;
                self.record_backings[lease.slot] = None;
                let entry = cohort
                    .iter()
                    .find(|entry| u32::from(entry.expert) == lease.expert)
                    .context("missing block demand consumer")?;
                read_bytes[entry.consumers[0].row] += EXPERT_BYTES;
            }
            // Cache reservation width is independent of reader queue width.
            // Keep the full plan alive (including every initially hot expert),
            // but never submit more than one AR cohort to the reader pool.
            for batch in plan.reads.chunks(TOP_K) {
                check_cancelled()?;
                let records = batch.iter().map(|&lease| (lease, self.records[lease.expert as usize].clone()));
                let mut reads = if self.gpu.ssd_direct_enabled {
                    self.gpu.readers()?.submit_shared_fresh(records, &self.gpu.device)?
                } else {
                    self.gpu.readers()?.submit(records)?
                };
                for (lease, payload) in reads.ready(true)? {
                    let backing = self.gpu.completed_record(payload)?;
                    let expert = FlashExpert::from_backing(self.gpu, &backing)?;
                    ensure!(plan.publish(lease), "stale Flash block SSD completion");
                    self.slots.borrow_mut()[lease.slot] = Some(expert);
                    // Fresh immutable records; no live GPU generation is reused.
                }
                ensure!(reads.remaining() == 0, "incomplete Flash block SSD batch");
                #[cfg(test)]
                if plan.reads.len() > TOP_K && CANCEL_AFTER_READ_BATCH.with(|inject| inject.replace(false)) {
                    cancelled.store(true, Ordering::Relaxed);
                }
                check_cancelled()?;
                #[cfg(test)]
                for (lease, _, _) in &reads.trace {
                    let entry = cohort.iter().find(|entry| u32::from(entry.expert) == lease.expert)
                        .context("missing completed block read consumer")?;
                    timelines[entry.consumers[0].row].push(serde_json::json!({
                        "event": "block_read", "expert": lease.expert, "slot": lease.slot,
                        "batch_size": batch.len()
                    }));
                }
            }
            objc::rc::autoreleasepool(|| -> Result<()> {
                let command = self.gpu.queue.new_command_buffer().to_owned();
                {
                    let slots = self.slots.borrow();
                    for (entry, &(slot, _)) in cohort.iter().zip(&plan.routes) {
                        let expert = slots[slot].as_ref().context("missing Flash block expert")?;
                        for consumer in &entry.consumers {
                            check_cancelled()?;
                            ensure!(
                                contributions[consumer.row][consumer.rank].is_none(),
                                "duplicate block contribution"
                            );
                            contributions[consumer.row][consumer.rank] =
                                Some(expert.encode_view_into(
                                    &command,
                                    &rows[consumer.row],
                                    Storage::StorageModePrivate,
                                    None,
                                )?);
                        }
                    }
                }
                self.last.push(command.clone());
                command.commit();
                self.drain_commands()
            })?;
            #[cfg(test)]
            if CANCEL_AFTER_WAVE.with(|inject| inject.replace(false)) {
                cancelled.store(true, Ordering::Relaxed);
            }
        }
        check_cancelled()?;
        let shared = objc::rc::autoreleasepool(|| -> Result<Vec<Buffer>> {
            let command = self.gpu.queue.new_command_buffer().to_owned();
            let shared = self.encode_shared_block(&command, rows, Storage::StorageModePrivate)?;
            self.last.push(command.clone());
            command.commit();
            self.drain_commands()?;
            Ok(shared)
        })?;
        let mut outputs = Vec::with_capacity(rows.len());
        for (row, x) in rows.iter().enumerate() {
            check_cancelled()?;
            let output = objc::rc::autoreleasepool(|| -> Result<MoeOutput> {
                let dtype = usize::from(x.dtype == FloatType::F32);
                let weights = routed[row].completed_scores()?;
                let command = self.gpu.queue.new_command_buffer().to_owned();
                let shared = &shared[row];
                let multiplier =
                    MatrixView::output(self.gpu, x.dtype, 1, 1, Storage::StorageModePrivate)?;
                let values =
                    MatrixView::output(self.gpu, x.dtype, 1, 2560, Storage::StorageModeShared)?;
                Metal::encode(
                    &command,
                    &self.gpu.moe_switch[dtype],
                    &[&x.buffer, &self.switch, &multiplier.buffer],
                    [x.layout.offset, x.layout.strides[1], 0],
                    MTLSize::new(1, 1, 1),
                    512,
                );
                let mut bindings = contributions[row]
                    .iter()
                    .map(|term| {
                        term.as_ref()
                            .map(|buffer| (buffer, 0u64))
                            .context("missing canonical block contribution")
                    })
                    .collect::<Result<Vec<_>>>()?;
                bindings.extend([
                    (shared, 0),
                    (&weights.buffer, 0),
                    (&multiplier.buffer, 0),
                    (&values.buffer, 0),
                ]);
                Metal::encode_bound(
                    &command,
                    &self.gpu.moe_mix[dtype],
                    bindings.into_iter(),
                    [2560u32],
                    MTLSize::new(80, 1, 1),
                    32,
                );
                self.last.push(command.clone());
                command.commit();
                self.drain_commands()?;
                Ok(MoeOutput {
                    spin_wait_enabled: self.gpu.spin_wait_enabled,
                    #[cfg(test)]
                    spin_wait_stats: self.gpu.spin_wait_stats.clone(),
                    commands: vec![command],
                    hot: RefCell::new(None),
                    hot_error: RefCell::new(None),
                    values,
                    indices: ids[row].clone(),
                    hits: hits[row],
                    read_bytes: read_bytes[row],
                    multiplier_trace: None,
                    #[cfg(test)]
                    timeline: timelines[row].clone(),
                    #[cfg(test)]
                    extra_wave_submissions: 0,
                    #[cfg(test)]
                    injected_failure: false,
                })
            })?;
            outputs.push(output);
        }
        check_cancelled()?;
        guard.commit();
        Ok(outputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_layout::MatrixLayout;
    use std::collections::BTreeSet;

    #[test]
    #[ignore = "physical Flash checkpoint; block SSD reads, canonical outputs and failed-fill recovery"]
    fn block_experts_read_once_and_recover_after_failed_fill() -> Result<()> {
        let path = std::env::var("MLXL3_MOE_CHECKPOINT")?;
        let checkpoint = checkpoint::inspect(std::path::Path::new(&path))?;
        let gpu = Metal::new()?;
        let prefix = "model.language_model.layers.0.mlp";
        let mut wide_cold_case = false;
        for capacity in [10, 20, 40] {
            let mut block = FlashMoeRow::load(&gpu, &checkpoint, prefix, capacity)?;
            ensure!(
                block.block_eligible(),
                "disable Hot, HC workspace, deferred and finish-mix for block test"
            );
            for fp32 in [false, true] {
                for repeated in [false, true] {
                    let rows = (0..MAX_ROWS)
                        .map(|row| -> Result<MatrixView> {
                            let phase = if repeated { 0 } else { row * 19 };
                            let data = (0..2560)
                                .map(|i| (((i + phase) % 73) as f32 - 36.) / 53.)
                                .collect::<Vec<_>>();
                            let storage = if fp32 {
                                gpu.f32_storage(&data)?
                            } else {
                                gpu.f16_storage(
                                    &data
                                        .iter()
                                        .map(|&v| half::f16::from_f32(v).to_bits())
                                        .collect::<Vec<_>>(),
                                )?
                            };
                            storage.matrix(MatrixLayout {
                                shape: [1, 2560],
                                strides: [2560, 1],
                                offset: 0,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let mut scalar = FlashMoeRow::load(&gpu, &checkpoint, prefix, capacity)?;
                    let expected = rows
                        .iter()
                        .map(|x| -> Result<_> {
                            let out = scalar.submit(x)?;
                            Ok((out.indices.clone(), out.read()?))
                        })
                        .collect::<Result<Vec<_>>>()?;
                    for width in 1..=MAX_ROWS {
                        block.clear_test_cache()?;
                        let actual = block.forward_block(&rows[..width])?;
                        let union = expected[..width]
                            .iter()
                            .flat_map(|(ids, _)| ids.iter().copied())
                            .collect::<BTreeSet<_>>();
                        let mut reads = Vec::new();
                        for (row, out) in actual.iter().enumerate() {
                            ensure!(
                                out.indices == expected[row].0 && out.read()? == expected[row].1
                            );
                            ensure!(out.hits == 0);
                            for event in &out.timeline {
                                if event["event"] == "block_read" {
                                    let batch_size = event["batch_size"].as_u64().context("missing read batch size")?;
                                    ensure!((1..=TOP_K as u64).contains(&batch_size));
                                    reads.push(event["expert"].as_u64().context("missing read ID")? as u32);
                                }
                            }
                        }
                        wide_cold_case |= capacity > TOP_K && union.len() > TOP_K;
                        ensure!(reads.len() == union.len() && reads.iter().copied().collect::<BTreeSet<_>>() == union);
                        ensure!(actual.iter().map(|out| out.read_bytes).sum::<u64>() == union.len() as u64 * EXPERT_BYTES);
                        if repeated {
                            ensure!(union.len() == TOP_K);
                            let hot = block.forward_block(&rows[..width])?;
                            ensure!(
                                hot.iter()
                                    .all(|out| out.read_bytes == 0 && out.hits == TOP_K)
                            );
                            for (row, out) in hot.iter().enumerate() {
                                ensure!(out.read()? == expected[row].1);
                            }
                        }
                    }
                    if capacity > TOP_K && expected.iter().flat_map(|(ids, _)| ids.iter().copied()).collect::<BTreeSet<_>>().len() > TOP_K {
                        block.clear_test_cache()?;
                        let cancelled = AtomicBool::new(false);
                        CANCEL_AFTER_READ_BATCH.with(|inject| inject.set(true));
                        let stopped = block.forward_block_cancellable(&rows, &cancelled);
                        CANCEL_AFTER_READ_BATCH.with(|inject| inject.set(false));
                        ensure!(stopped.is_err() && cancelled.load(Ordering::Relaxed));
                        ensure!(block.last.is_empty());
                        gpu.ensure_reads_idle()?;
                        let cache = block.cache.borrow();
                        ensure!(cache.available_slots() == capacity);
                        ensure!(cache.ready_slot_snapshot().iter().all(Option::is_none));
                        drop(cache);
                        let recovered = block.forward_block(&rows)?;
                        for (row, out) in recovered.iter().enumerate() { ensure!(out.read()? == expected[row].1); }
                    }
                    block.clear_test_cache()?;
                    let before = block.cache.borrow().ready_slot_snapshot();
                    let cancelled = AtomicBool::new(true);
                    ensure!(block.forward_block_cancellable(&rows, &cancelled).is_err());
                    ensure!(block.cache.borrow().ready_slot_snapshot() == before);
                    cancelled.store(false, Ordering::Relaxed);
                    CANCEL_AFTER_WAVE.with(|inject| inject.set(true));
                    let stopped = block.forward_block_cancellable(&rows, &cancelled);
                    CANCEL_AFTER_WAVE.with(|inject| inject.set(false));
                    ensure!(stopped.is_err() && cancelled.load(Ordering::Relaxed));
                    ensure!(
                        block.last.is_empty(),
                        "cancellation left unretired GPU work"
                    );
                    gpu.ensure_reads_idle()?;
                    ensure!(
                        block
                            .cache
                            .borrow()
                            .ready_slot_snapshot()
                            .iter()
                            .all(Option::is_none)
                    );
                    let recovered = block.forward_block(&rows)?;
                    for (row, out) in recovered.iter().enumerate() {
                        ensure!(out.read()? == expected[row].1);
                    }
                    block.clear_test_cache()?;
                    let expert = expected[0].0[0] as usize;
                    let old = block.records[expert].data_offsets;
                    block.records[expert].data_offsets = [u64::MAX - EXPERT_BYTES, u64::MAX];
                    ensure!(
                        block.forward_block(&rows).is_err(),
                        "invalid SSD offset unexpectedly succeeded"
                    );
                    block.records[expert].data_offsets = old;
                    gpu.ensure_reads_idle()?;
                    ensure!(
                        block
                            .cache
                            .borrow()
                            .ready_slot_snapshot()
                            .iter()
                            .all(Option::is_none)
                    );
                    let recovered = block.forward_block(&rows)?;
                    for (row, out) in recovered.iter().enumerate() {
                        ensure!(out.read()? == expected[row].1);
                    }
                }
            }
        }
        ensure!(wide_cold_case, "fixture must exercise more than ten cold experts in a wide cache");
        Ok(())
    }
}
