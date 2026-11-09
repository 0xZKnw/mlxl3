//! Opt-in block SSD reuse. Every wave is complete before cache slots are reused.
//! Ready experts reuse the qualified AR axis; each row keeps canonical rank mixing.
use super::*;
use crate::gpu::RouteOutput;
use crate::speculation_plan::{ExpertCohort, MAX_ROWS, TOP_K, resident_expert_cohorts};
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(test)]
thread_local! {
    static ROUTED_CONSUMER_BATCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    // One-shot deterministic cancellation after a genuinely submitted/waited wave.
    static CANCEL_AFTER_INITIAL_SUBMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static CANCEL_AFTER_WAVE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static CANCEL_AFTER_READ_BATCH: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl FlashMoeRow<'_> {
    /// One-shot session rollback probe, also resettable when preflight exits early.
    #[cfg(test)]
    pub(in crate::gpu) fn set_cancel_after_initial_submit_for_test(enabled: bool) {
        CANCEL_AFTER_INITIAL_SUBMIT.with(|inject| inject.set(enabled));
    }

    /// Caller checks this before preparing any stateful row. Rechecked below.
    pub(in crate::gpu) fn block_eligible(&self) -> bool {
        self.prefetcher.is_none()
            && (!self.gpu.device_resident_route_enabled()
                || self.gpu.common_executor && self.device_hot_gate_eligible_for_encoding(0))
            && (self.gpu.common_executor
                || (!self.gpu.hc_workspace_enabled.get()
                    && !self.gpu.moe_deferred.get()
                    && !self.gpu.moe_finish_mix.get()))
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

    #[cfg(test)]
    pub(in crate::gpu) fn forward_block_cancellable(
        &mut self,
        rows: &[MatrixView],
        cancelled: &AtomicBool,
    ) -> Result<Vec<MoeOutput>> {
        self.forward_block_routed_cancellable(
            rows,
            (0..rows.len()).map(|_| None).collect(),
            cancelled,
        )
    }

    pub(in crate::gpu) fn forward_block_routed_cancellable(
        &mut self,
        rows: &[MatrixView],
        routing: Vec<Option<RouteOutput>>,
        cancelled: &AtomicBool,
    ) -> Result<Vec<MoeOutput>> {
        let check_cancelled = || -> Result<()> {
            ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
            Ok(())
        };
        check_cancelled()?;
        ensure!(self.block_eligible(), "unsupported Flash block SSD mode");
        ensure!(
            (1..=MAX_ROWS).contains(&rows.len()) && routing.len() == rows.len(),
            "Flash SSD block expects 1..4 rows with matching routes"
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
        let device_hot = self.gpu.common_executor && self.gpu.device_resident_route_enabled();
        ensure!(
            !device_hot || rows.iter().all(|row| row.dtype == FloatType::F32),
            "private Hot rows require F32 inputs"
        );
        let mut hot_outputs = Vec::with_capacity(rows.len());
        let mut routed = Vec::with_capacity(rows.len());
        let mut ids = Vec::with_capacity(rows.len());
        // RouteOutput has no draining Drop. Finish each submitted route before
        // another fallible submission, including error/cancellation paths.
        for (x, route) in rows.iter().zip(routing) {
            check_cancelled()?;
            let route = match route {
                Some(route) => route,
                None => self.gate.submit(x)?,
            };
            #[cfg(test)]
            let route = {
                let mut route = route;
                crate::gpu::route_swap_trace::block_pick(self.gpu, &mut route)?;
                route
            };
            let hot = if device_hot {
                let snapshot = ResidentSnapshot::from_cache(
                    self.cache.borrow().ready_slot_snapshot(),
                    &self.record_backings,
                )?;
                Some(
                    self.gpu
                        .submit_resident_hot_block(x, route.device_indices(), snapshot)?,
                )
            } else {
                None
            };
            let indices = route.indices()?;
            if let Some(hot) = &hot {
                hot.snapshot()
                    .validate_routes(&self.cache.borrow().ready_slot_snapshot(), &indices)?;
            }
            hot_outputs.push(hot);
            ids.push(indices);
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
        let access = self.cache.borrow_mut().block_access(&routes)?;
        // Consume all initially resident experts in the first wave before any
        // later reservation can evict them. Rank ordering is independent of this.
        let hits = routes
            .iter()
            .map(|row| row.iter().filter(|&&id| initially_hot(id)).count())
            .collect::<Vec<_>>();
        let mut read_bytes = vec![0u64; rows.len()];
        let finish_mix = self.gpu.common_executor
            && self.gpu.moe_finish_mix.get()
            && rows[0].dtype == FloatType::F32;
        let mut pending_finishes: Vec<[Option<PendingExpertFinish>; TOP_K]> = (0..rows.len())
            .map(|_| std::array::from_fn(|_| None))
            .collect();
        let mut contributions: Vec<[Option<Buffer>; TOP_K]> = (0..rows.len())
            .map(|_| std::array::from_fn(|_| None))
            .collect();
        // Each rank owns a fixed output slice, independent of completion order.
        // Reuse the same qualified AR policy/offset contract, rather than placing
        // outputs in SSD completion order and accidentally changing reduction.
        let mut slabs = rows
            .iter()
            .map(|_| {
                (self.gpu.expert_ar_enabled[2].get() && !finish_mix && !device_hot)
                    .then(|| {
                        let backing = buffer(&self.gpu.device, 51200, Storage::StorageModePrivate)?;
                        #[cfg(test)]
                        self.gpu.expert_ar_uses[2].set(self.gpu.expert_ar_uses[2].get() + 1);
                        Ok::<_, anyhow::Error>(backing)
                    })
                    .transpose()
            })
            .collect::<Result<Vec<_>>>()?;
        for (row, hot) in hot_outputs.iter().enumerate() {
            if let Some(hot) = hot {
                let slab = hot.contribution(0)?.0.clone();
                for (rank, &expert) in routes[row].iter().enumerate() {
                    if initially_hot(expert) {
                        contributions[row][rank] = Some(slab.clone());
                    }
                }
                slabs[row] = Some(slab);
            }
        }
        #[cfg(test)]
        let mut timelines = vec![Vec::new(); rows.len()];
        let guard = ExpertFill::new(&self.cache);
        let reuse = crate::ssd_reads::record_reuse_eligible(
            self.record_reuse_enabled && self.gpu.ssd_direct_enabled,
            0,
            self.prefetcher.is_some(),
            true, // Entry drain and each wave drain precede owner transfer.
        );
        let mut shared = None;
        for wave in &cohorts.waves {
            check_cancelled()?;
            let cohort = &cohorts.cohorts[wave.clone()];
            let requested = cohort
                .iter()
                .map(|entry| u32::from(entry.expert))
                .collect::<Vec<_>>();
            let plan = DemandPlan::reserve_block(&self.cache, &requested, &access)?;
            // The previous wave was drained, including on failure. Drop all old
            // projection views before transferring backing ownership to readers.
            let mut owners = Vec::with_capacity(plan.reads.len());
            for lease in &plan.reads {
                let old_views = self.slots.borrow_mut()[lease.slot].take();
                let owner = self.record_backings[lease.slot].take();
                drop(old_views);
                owners.push(if reuse {
                    owner.and_then(RecordBacking::into_reusable)
                } else {
                    None
                });
                let entry = cohort
                    .iter()
                    .find(|entry| u32::from(entry.expert) == lease.expert)
                    .context("missing block demand consumer")?;
                read_bytes[entry.consumers[0].row] += EXPERT_BYTES;
            }
            // Keep the entire cache reservation pinned, but bound each reader
            // submission to TOP_K. Submit I/O before shared/resident GPU work.
            // Even an all-hot wave runs the initial command exactly once.
            let wave_result = objc::rc::autoreleasepool(|| -> Result<()> {
                let submitted = (|| -> Result<()> {
                    let sliding = plan.reads.len() > TOP_K && self.gpu.readers()?.window_enabled;
                    let batch_count = if sliding {
                        1
                    } else {
                        plan.reads.len().div_ceil(TOP_K).max(1)
                    };
                    for batch_index in 0..batch_count {
                        check_cancelled()?;
                        let start = (batch_index * TOP_K).min(plan.reads.len());
                        let end = if sliding {
                            plan.reads.len()
                        } else {
                            (start + TOP_K).min(plan.reads.len())
                        };
                        let batch = &plan.reads[start..end];
                        let mut reads = if batch.is_empty() {
                            None
                        } else {
                            let records = batch.iter().enumerate().map(|(index, &lease)| {
                                (lease, self.records[lease.expert as usize].clone(), index)
                            });
                            Some(if sliding {
                                self.gpu.readers()?.submit_window(
                                    records.map(|(lease, record, index)| {
                                        (lease, record, owners[start + index].take())
                                    }),
                                    self.gpu.ssd_direct_enabled.then_some(&self.gpu.device),
                                )?
                            } else {
                                crate::ssd_reads::BlockReads::Batch(
                                    if self.gpu.ssd_direct_enabled {
                                        if reuse {
                                            self.gpu.readers()?.submit_shared_reusing(
                                                records.map(|(lease, record, index)| {
                                                    (lease, record, owners[start + index].take())
                                                }),
                                                &self.gpu.device,
                                            )?
                                        } else {
                                            self.gpu.readers()?.submit_shared(
                                                records.map(|(lease, record, _)| (lease, record)),
                                                &self.gpu.device,
                                            )?
                                        }
                                    } else {
                                        self.gpu.readers()?.submit(
                                            records.map(|(lease, record, _)| (lease, record)),
                                        )?
                                    },
                                )
                            })
                        };
                        let hot = if batch_index == 0 && !device_hot {
                            plan.routes
                                .iter()
                                .filter_map(|&(slot, hit)| hit.then_some(slot))
                                .collect::<Vec<_>>()
                        } else {
                            Vec::new()
                        };
                        if shared.is_none() || !hot.is_empty() {
                            let command = self.gpu.queue.new_command_buffer().to_owned();
                            if shared.is_none() {
                                shared = Some(self.encode_shared_block(
                                    &command,
                                    rows,
                                    Storage::StorageModePrivate,
                                )?);
                            }
                            self.encode_block_ready(
                                &command,
                                rows,
                                cohort,
                                &plan.routes,
                                &hot,
                                &slabs,
                                &mut contributions,
                                &mut pending_finishes,
                            )?;
                            self.last.push(command.clone());
                            command.commit();
                            #[cfg(test)]
                            timelines[0].push(serde_json::json!({
                                "event": "block_initial_submit", "cold_reads": batch.len(),
                                "resident_experts": hot.len()
                            }));
                            #[cfg(test)]
                            if CANCEL_AFTER_INITIAL_SUBMIT.with(|inject| inject.replace(false)) {
                                cancelled.store(true, Ordering::Relaxed);
                            }
                        }
                        if let Some(reads) = &mut reads {
                            while reads.remaining() > 0 {
                                check_cancelled()?;
                                // Consume ready completions without waiting for the
                                // whole batch. Their compute overlaps the tail I/O.
                                let completed = reads.ready()?;
                                check_cancelled()?;
                                let mut ready = Vec::with_capacity(completed.len());
                                for (lease, payload) in completed {
                                    let direct = matches!(
                                        &payload,
                                        crate::ssd_reads::ReadPayload::Shared(_)
                                    );
                                    let backing = self.gpu.completed_record(payload)?;
                                    let expert = FlashExpert::from_backing(self.gpu, &backing)?;
                                    ensure!(
                                        plan.publish(lease),
                                        "stale Flash block SSD completion"
                                    );
                                    self.slots.borrow_mut()[lease.slot] = Some(expert);
                                    if reuse && direct {
                                        self.record_backings[lease.slot] =
                                            Some(RecordBacking::new(backing, device_hot));
                                    }
                                    ready.push(lease.slot);
                                }
                                let command = self.gpu.queue.new_command_buffer().to_owned();
                                self.encode_block_ready(
                                    &command,
                                    rows,
                                    cohort,
                                    &plan.routes,
                                    &ready,
                                    &slabs,
                                    &mut contributions,
                                    &mut pending_finishes,
                                )?;
                                self.last.push(command.clone());
                                command.commit();
                            }
                            #[cfg(test)]
                            for (lease, _, _) in reads.trace() {
                                let entry = cohort
                                    .iter()
                                    .find(|entry| u32::from(entry.expert) == lease.expert)
                                    .context("missing completed block read consumer")?;
                                timelines[entry.consumers[0].row].push(serde_json::json!({
                                    "event": "block_read", "expert": lease.expert,
                                    "slot": lease.slot, "batch_size": batch.len().min(TOP_K), "sliding_window": sliding
                                }));
                            }
                        }
                        #[cfg(test)]
                        if plan.reads.len() > TOP_K
                            && CANCEL_AFTER_READ_BATCH.with(|inject| inject.replace(false))
                        {
                            cancelled.store(true, Ordering::Relaxed);
                        }
                        check_cancelled()?;
                    }
                    Ok(())
                })();
                // Retire all submitted work while this autorelease pool is
                // still alive, before error propagation releases cache leases.
                let drained = self.drain_commands();
                submitted?;
                drained
            });
            wave_result?;
            #[cfg(test)]
            if CANCEL_AFTER_WAVE.with(|inject| inject.replace(false)) {
                cancelled.store(true, Ordering::Relaxed);
            }
        }
        check_cancelled()?;
        let shared = shared.context("missing block shared expert outputs")?;
        let mut outputs = Vec::with_capacity(rows.len());
        let command = self.gpu.queue.new_command_buffer().to_owned();
        for (row, x) in rows.iter().enumerate() {
            check_cancelled()?;
            let output = objc::rc::autoreleasepool(|| -> Result<MoeOutput> {
                let dtype = usize::from(x.dtype == FloatType::F32);
                let weights = routed[row].completed_scores()?;
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
                if finish_mix {
                    let terms = pending_finishes[row]
                        .iter()
                        .map(|term| {
                            term.as_ref()
                                .context("missing canonical block pending finish")
                        })
                        .collect::<Result<Vec<_>>>()?;
                    PendingExpertFinish::encode_mix(
                        self.gpu,
                        &command,
                        &terms,
                        shared,
                        [weights, &multiplier, &values],
                    )?;
                } else {
                    let mut bindings = contributions[row]
                        .iter()
                        .enumerate()
                        .map(|(rank, term)| {
                            let buffer = term
                                .as_ref()
                                .context("missing canonical block contribution")?;
                            let offset = slabs[row]
                                .as_ref()
                                .map(|slab| {
                                    crate::expert_ar_policy::prefix(
                                        rank,
                                        TOP_K,
                                        2560,
                                        slab.length(),
                                    )
                                    .context("invalid canonical block contribution prefix")
                                })
                                .transpose()?
                                .unwrap_or(0);
                            Ok((buffer, offset))
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
                }
                Ok(MoeOutput {
                    spin_wait_enabled: self.gpu.spin_wait_enabled,
                    #[cfg(test)]
                    spin_wait_stats: self.gpu.spin_wait_stats.clone(),
                    commands: vec![command.clone()],
                    hot: RefCell::new(hot_outputs[row].take()),
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
        for output in &outputs {
            if let Some(hot) = output.hot.borrow_mut().as_mut() {
                hot.retain_consumer(&command)?;
            }
        }
        self.last.push(command.clone());
        command.commit();
        if !self.gpu.common_executor
            || !self.gpu.moe_deferred.get()
            || cancelled.load(Ordering::Relaxed)
        {
            self.drain_commands()?;
        }
        check_cancelled()?;
        guard.commit();
        Ok(outputs)
    }

    /// Batch only already-ready experts for each row. The existing expert-axis
    /// encoder retains its scalar fallback and qualified precision boundaries.
    #[allow(clippy::too_many_arguments)]
    fn encode_block_ready(
        &self,
        command: &metal::CommandBufferRef,
        rows: &[MatrixView],
        cohort: &[ExpertCohort],
        routes: &[(usize, bool)],
        ready: &[usize],
        slabs: &[Option<Buffer>],
        contributions: &mut [[Option<Buffer>; TOP_K]],
        pending_finishes: &mut [[Option<PendingExpertFinish>; TOP_K]],
    ) -> Result<()> {
        let slots = self.slots.borrow();
        let finish_mix = self.gpu.common_executor
            && self.gpu.moe_finish_mix.get()
            && rows[0].dtype == FloatType::F32;
        let batch_consumers =
            crate::flash_qmv::shared_verify_enabled(self.gpu.flash_verify_qmv.is_some())
                && (!finish_mix || self.gpu.expert_consumer_qmv.is_some());
        let mut packed_rows: [Option<MatrixView>; 16] = std::array::from_fn(|_| None);
        let consumer_grid = batch_consumers && self.gpu.expert_consumer_qmv.is_some();
        if consumer_grid {
            for (mask, packed) in packed_rows.iter_mut().enumerate().skip(1) {
                if mask.count_ones() < 2 {
                    continue;
                }
                let selected = cohort
                    .iter()
                    .zip(routes)
                    .filter(|(entry, (slot, _))| {
                        ready.contains(slot)
                            && entry.consumers.iter().fold(0usize, |m, c| m | (1 << c.row)) == mask
                    })
                    .collect::<Vec<_>>();
                for group in selected.chunks(TOP_K) {
                    if group.is_empty() {
                        continue;
                    }
                    let experts = group
                        .iter()
                        .map(|(_, (slot, _))| {
                            slots[*slot]
                                .as_ref()
                                .context("missing expert/consumer record")
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let inputs = group[0]
                        .0
                        .consumers
                        .iter()
                        .map(|c| &rows[c.row])
                        .collect::<Vec<_>>();
                    if packed.is_none() {
                        *packed = Some(experts[0].pack_consumer_rows(command, &inputs)?);
                    }
                    let targets = if finish_mix {
                        Vec::new()
                    } else {
                        group
                            .iter()
                            .map(|(entry, _)| {
                                entry
                                    .consumers
                                    .iter()
                                    .map(|c| -> Result<_> {
                                        ensure!(
                                            contributions[c.row][c.rank].is_none(),
                                            "duplicate consumer contribution"
                                        );
                                        if let Some(slab) = &slabs[c.row] {
                                            Ok((
                                                slab.clone(),
                                                crate::expert_ar_policy::prefix(
                                                    c.rank,
                                                    TOP_K,
                                                    2560,
                                                    slab.length(),
                                                )
                                                .context("invalid consumer destination")?,
                                            ))
                                        } else {
                                            Ok((
                                                buffer(
                                                    &self.gpu.device,
                                                    5120,
                                                    Storage::StorageModePrivate,
                                                )?,
                                                0,
                                            ))
                                        }
                                    })
                                    .collect::<Result<Vec<_>>>()
                            })
                            .collect::<Result<Vec<_>>>()?
                    };
                    let bindings = targets
                        .iter()
                        .map(|destinations| {
                            destinations
                                .iter()
                                .map(|(buffer, offset)| (buffer, *offset))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    let pending = FlashExpert::encode_consumer_cohort(
                        command,
                        packed.as_ref().expect("packed consumers"),
                        &experts,
                        &bindings,
                        finish_mix,
                    )?;
                    for ((entry, _), terms) in group.iter().zip(pending) {
                        for (consumer, term) in entry.consumers.iter().zip(terms) {
                            ensure!(
                                pending_finishes[consumer.row][consumer.rank].is_none(),
                                "duplicate pending finish"
                            );
                            pending_finishes[consumer.row][consumer.rank] = Some(term);
                        }
                    }
                    for ((entry, _), destinations) in group.iter().zip(targets) {
                        for (c, (buffer, _)) in entry.consumers.iter().zip(destinations) {
                            contributions[c.row][c.rank] = Some(buffer);
                        }
                    }
                    #[cfg(test)]
                    ROUTED_CONSUMER_BATCHES.with(|count| count.set(count.get() + group.len()));
                }
            }
        }
        if batch_consumers && !consumer_grid {
            for (entry, &(slot, _)) in cohort.iter().zip(routes) {
                if entry.consumers.len() < 2 || !ready.contains(&slot) {
                    continue;
                }
                let expert = slots[slot]
                    .as_ref()
                    .context("missing ready consumer expert")?;
                let inputs = entry
                    .consumers
                    .iter()
                    .map(|c| &rows[c.row])
                    .collect::<Vec<_>>();
                let mask = entry
                    .consumers
                    .iter()
                    .fold(0usize, |mask, c| mask | (1 << c.row));
                if packed_rows[mask].is_none() {
                    packed_rows[mask] = Some(expert.pack_consumer_rows(command, &inputs)?);
                }
                let values = expert.encode_view(
                    command,
                    packed_rows[mask].as_ref().expect("packed consumers"),
                    Storage::StorageModePrivate,
                )?;
                ensure!(
                    values.length() == entry.consumers.len() as u64 * 5120,
                    "invalid consumer expert output"
                );
                for (index, consumer) in entry.consumers.iter().enumerate() {
                    ensure!(
                        contributions[consumer.row][consumer.rank].is_none(),
                        "duplicate batched expert contribution"
                    );
                    let (output, offset) = if let Some(slab) = &slabs[consumer.row] {
                        (
                            slab.clone(),
                            crate::expert_ar_policy::prefix(
                                consumer.rank,
                                TOP_K,
                                2560,
                                slab.length(),
                            )
                            .context("invalid batched contribution prefix")?,
                        )
                    } else {
                        (
                            buffer(&self.gpu.device, 5120, Storage::StorageModePrivate)?,
                            0,
                        )
                    };
                    let blit = command.new_blit_command_encoder();
                    blit.copy_from_buffer(&values, index as u64 * 5120, &output, offset, 5120);
                    blit.end_encoding();
                    contributions[consumer.row][consumer.rank] = Some(output);
                }
                #[cfg(test)]
                ROUTED_CONSUMER_BATCHES.with(|count| count.set(count.get() + 1));
            }
        }
        for (row, x) in rows.iter().enumerate() {
            let selected = cohort
                .iter()
                .zip(routes)
                .flat_map(|(entry, &(slot, _))| {
                    entry.consumers.iter().filter_map(move |consumer| {
                        (consumer.row == row
                            && ready.contains(&slot)
                            && !(batch_consumers && entry.consumers.len() > 1))
                            .then_some((slot, consumer.rank))
                    })
                })
                .collect::<Vec<_>>();
            if selected.is_empty() {
                continue;
            }
            let experts = selected
                .iter()
                .map(|&(slot, rank)| {
                    ensure!(
                        contributions[row][rank].is_none(),
                        "duplicate block contribution"
                    );
                    slots[slot]
                        .as_ref()
                        .context("missing ready Flash block expert")
                })
                .collect::<Result<Vec<_>>>()?;
            let targets = selected
                .iter()
                .map(|&(_, rank)| {
                    slabs[row]
                        .as_ref()
                        .map(|slab| {
                            let offset =
                                crate::expert_ar_policy::prefix(rank, TOP_K, 2560, slab.length())
                                    .context("invalid canonical block contribution prefix")?;
                            Ok((slab, offset))
                        })
                        .transpose()
                })
                .collect::<Result<Vec<_>>>()?;
            if finish_mix {
                let values = FlashExpert::encode_ready_cohort_pending(
                    command,
                    x,
                    &experts,
                    &(0..experts.len()).map(|_| None).collect::<Vec<_>>(),
                )?;
                ensure!(
                    values.len() == selected.len(),
                    "incomplete pending finish cohort"
                );
                for ((_, rank), term) in selected.into_iter().zip(values) {
                    ensure!(
                        pending_finishes[row][rank].is_none(),
                        "duplicate pending finish"
                    );
                    pending_finishes[row][rank] = Some(term);
                }
                continue;
            }
            let values = FlashExpert::encode_ready_cohort(
                command,
                x,
                &experts,
                &targets,
                Storage::StorageModePrivate,
            )?;
            ensure!(
                values.len() == selected.len(),
                "incomplete block ready cohort"
            );
            for ((_, rank), value) in selected.into_iter().zip(values) {
                contributions[row][rank] = Some(value);
            }
        }
        Ok(())
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
                    if capacity == TOP_K && !fp32 && !repeated {
                        let before = block.cache.borrow().ready_slot_snapshot();
                        let gates = gpu.gate_commands.get();
                        ensure!(
                            block
                                .forward_block_routed_cancellable(
                                    &rows,
                                    (1..rows.len()).map(|_| None).collect(),
                                    &AtomicBool::new(false),
                                )
                                .is_err()
                        );
                        ensure!(block.cache.borrow().ready_slot_snapshot() == before);
                        ensure!(gpu.gate_commands.get() == gates);
                        gpu.ensure_reads_idle()?;
                    }
                    for width in 1..=MAX_ROWS {
                        block.clear_test_cache()?;
                        let axis_before = gpu.expert_ar_uses[3].get();
                        let fused_before = gpu.expert_ar_uses[0].get();
                        let dual_before = gpu.expert_ar_uses[1].get();
                        let batches_before = ROUTED_CONSUMER_BATCHES.with(std::cell::Cell::get);
                        let specialized_before = gpu.flash_verify_specialized_uses.get();
                        let consumer_before = gpu.expert_consumer_dispatches.get();
                        let down_prepare_before = gpu.expert_down_prepare_uses.get();
                        let actual = block.forward_block(&rows[..width])?;
                        if repeated && width > 1 && gpu.expert_consumer_qmv.is_some() {
                            ensure!(
                                gpu.expert_consumer_dispatches.get() > consumer_before,
                                "expert/consumer QMV grid was not reached"
                            );
                        }
                        if gpu.expert_down_prepare.is_some() {
                            ensure!(
                                gpu.expert_down_prepare_uses.get() > down_prepare_before,
                                "fused down preparation was not reached"
                            );
                        }
                        if repeated
                            && width > 1
                            && gpu.expert_consumer_qmv.is_none()
                            && gpu.flash_verify_specialized.is_some()
                            && crate::flash_qmv::shared_verify_enabled(
                                gpu.flash_verify_qmv.is_some(),
                            )
                            && gpu.flash_fixed_setting != Some(false)
                        {
                            let counts = gpu.flash_verify_specialized_uses.get();
                            ensure!(
                                counts[5] > specialized_before[5]
                                    && counts[6] > specialized_before[6],
                                "routed consumer batches did not execute both fixed MB2 projections"
                            );
                        }
                        if repeated
                            && width > 1
                            && crate::flash_qmv::shared_verify_enabled(
                                gpu.flash_verify_qmv.is_some(),
                            )
                        {
                            ensure!(
                                ROUTED_CONSUMER_BATCHES.with(std::cell::Cell::get) > batches_before,
                                "common routed experts did not execute multirow QMV"
                            );
                        }
                        if gpu.expert_ar_enabled[3].get() {
                            if repeated
                                && width > 1
                                && gpu.expert_ar_enabled[1].get()
                                && crate::flash_qmv::shared_verify_enabled(
                                    gpu.flash_verify_qmv.is_some(),
                                )
                            {
                                ensure!(
                                    gpu.expert_ar_uses[1].get() > dual_before,
                                    "batched routed experts lost the qualified dual input preparation"
                                );
                            }
                            if repeated
                                && width > 1
                                && gpu.expert_ar_enabled[0].get()
                                && crate::flash_qmv::shared_verify_enabled(
                                    gpu.flash_verify_qmv.is_some(),
                                )
                            {
                                ensure!(
                                    gpu.expert_ar_uses[0].get() > fused_before,
                                    "batched routed experts lost the qualified up/SwiGLU finish"
                                );
                            }
                            ensure!(
                                gpu.expert_ar_uses[3].get() > axis_before
                                    || ROUTED_CONSUMER_BATCHES.with(std::cell::Cell::get)
                                        > batches_before,
                                "block path bypassed both expert-axis and consumer batching"
                            );
                        }
                        ensure!(
                            actual[0].timeline.first().is_some_and(|event| {
                                event["event"] == "block_initial_submit"
                                    && event["cold_reads"].as_u64().is_some_and(|n| n > 0)
                            }),
                            "cold block must submit initial GPU work before collecting reads"
                        );
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
                                    let batch_size = event["batch_size"]
                                        .as_u64()
                                        .context("missing read batch size")?;
                                    ensure!((1..=TOP_K as u64).contains(&batch_size));
                                    reads
                                        .push(event["expert"].as_u64().context("missing read ID")?
                                            as u32);
                                }
                            }
                        }
                        wide_cold_case |= capacity > TOP_K && union.len() > TOP_K;
                        ensure!(
                            reads.len() == union.len()
                                && reads.iter().copied().collect::<BTreeSet<_>>() == union
                        );
                        ensure!(
                            actual.iter().map(|out| out.read_bytes).sum::<u64>()
                                == union.len() as u64 * EXPERT_BYTES
                        );
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
                    if capacity > TOP_K
                        && expected
                            .iter()
                            .flat_map(|(ids, _)| ids.iter().copied())
                            .collect::<BTreeSet<_>>()
                            .len()
                            > TOP_K
                    {
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
                        for (row, out) in recovered.iter().enumerate() {
                            ensure!(out.read()? == expected[row].1);
                        }
                    }
                    // Force misses while preserving retired record owners. Unlike
                    // clear_test_cache, invalidation intentionally retains the
                    // recyclable backing allocations for the next reservation.
                    if block.record_reuse_enabled && gpu.ssd_direct_enabled {
                        // Invalidation preserves ages, so fill even unused slots first.
                        for _ in 0..capacity.div_ceil(TOP_K) {
                            block.cache.borrow_mut().invalidate();
                            block.forward_block(&rows[..1])?;
                        }
                        ensure!(block.record_backings.iter().all(Option::is_some));
                        block.cache.borrow_mut().invalidate();
                        let reused_before = gpu.readers()?.reused_destinations.get();
                        let recovered = block.forward_block(&rows[..1])?;
                        ensure!(
                            gpu.readers()?.reused_destinations.get() > reused_before,
                            "block misses did not reuse retired record owners"
                        );
                        ensure!(recovered[0].read()? == expected[0].1);
                    }
                    block.clear_test_cache()?;
                    let cancelled = AtomicBool::new(false);
                    CANCEL_AFTER_INITIAL_SUBMIT.with(|inject| inject.set(true));
                    let stopped = block.forward_block_cancellable(&rows, &cancelled);
                    CANCEL_AFTER_INITIAL_SUBMIT.with(|inject| inject.set(false));
                    ensure!(stopped.is_err() && cancelled.load(Ordering::Relaxed));
                    ensure!(
                        block.last.is_empty(),
                        "early cancellation left submitted GPU work"
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
        ensure!(
            wide_cold_case,
            "fixture must exercise more than ten cold experts in a wide cache"
        );
        Ok(())
    }
}
