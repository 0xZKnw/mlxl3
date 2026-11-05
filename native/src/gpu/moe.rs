use super::{FlashExpert, FlashGate, MatrixView, Metal, buffer, read_weight, view::FloatType};
use crate::{
    checkpoint::{self, Checkpoint, TensorInfo},
    flash_support::{DemandPlan, EXPERT_BYTES, ExpertCache, ExpertFill},
};
use anyhow::{Context, Result, ensure};
use metal::{
    Buffer, CommandBuffer, MTLCommandBufferStatus, MTLResourceOptions as Storage, MTLSize,
};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

#[path = "prefetch.rs"]
mod prefetch;
pub(super) use prefetch::PrefetchPool;
pub(crate) use prefetch::PrefetchStats;

#[cfg(test)]
thread_local! {
    pub(super) static EXTRA_WAVE_SUBMISSIONS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// One AR row, with immutable per-slot compressed records.
pub struct FlashMoeRow<'a> {
    gpu: &'a Metal,
    gate: FlashGate<'a>,
    shared: FlashExpert<'a>,
    switch: Buffer,
    records: Vec<TensorInfo>,
    cache: Rc<RefCell<ExpertCache>>,
    slots: prefetch::Slots<'a>,
    // Only direct records received by this instance have recyclable provenance.
    record_backings: Vec<Option<Buffer>>,
    pub(super) prefetcher: Option<Rc<RefCell<PrefetchPool<'a>>>>,
    predictor_half: bool,
    pub(super) extra_wave_enabled: bool,
    pub(super) record_reuse_enabled: bool,
    last: Vec<CommandBuffer>,
}

pub struct MoeOutput {
    commands: Vec<CommandBuffer>,
    values: MatrixView,
    pub indices: Vec<u32>,
    pub hits: usize,
    pub read_bytes: u64,
    multiplier_trace: Option<MatrixView>,
    #[cfg(test)]
    timeline: Vec<serde_json::Value>,
    #[cfg(test)]
    pub(super) extra_wave_submissions: usize,
    #[cfg(test)]
    pub(super) injected_failure: bool,
}

impl MoeOutput {
    pub fn completed(&self) -> Result<&MatrixView> {
        #[cfg(test)]
        let _wait = crate::flash_profile::span("native.moe_wait");
        let mut failure = None;
        for command in &self.commands {
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed && failure.is_none() {
                failure = Some(anyhow::anyhow!(
                    "Flash MoE command failed ({:?})",
                    command.status()
                ));
            }
            #[cfg(test)]
            crate::flash_profile::native_gpu_interval(command, "moe");
        }
        if let Some(error) = failure {
            return Err(error);
        }
        #[cfg(test)]
        ensure!(
            !self.injected_failure,
            "injected deferred MoE producer failure"
        );
        Ok(&self.values)
    }
    pub(super) fn values_for_gpu(&self) -> &MatrixView {
        &self.values
    }
    pub fn multiplier_trace(&self) -> Result<Option<Vec<u32>>> {
        self.multiplier_trace
            .as_ref()
            .map(|v| {
                self.completed()?;
                v.read_output()
            })
            .transpose()
    }
    pub fn read(&self) -> Result<Vec<u32>> {
        self.completed()?.read_output()
    }
}

impl<'a> FlashMoeRow<'a> {
    pub fn load(
        gpu: &'a Metal,
        checkpoint: &Checkpoint,
        prefix: &str,
        capacity: usize,
    ) -> Result<Self> {
        let extra_wave_enabled = match std::env::var("MLXL3_NATIVE_SSD_EXTRA_WAVE") {
            Ok(value) => {
                crate::ssd_reads::extra_wave_flag(Some(&value), gpu.device.name() == "Apple M5")
            }
            Err(std::env::VarError::NotPresent) => {
                crate::ssd_reads::extra_wave_flag(None, gpu.device.name() == "Apple M5")
            }
            Err(error) => return Err(error.into()),
        }
        .context("MLXL3_NATIVE_SSD_EXTRA_WAVE must be 0 or 1")?;
        let record_reuse_enabled = match std::env::var("MLXL3_NATIVE_RECORD_REUSE") {
            Ok(value) => {
                crate::ssd_reads::extra_wave_flag(Some(&value), gpu.device.name() == "Apple M5")
            }
            Err(std::env::VarError::NotPresent) => {
                crate::ssd_reads::extra_wave_flag(None, gpu.device.name() == "Apple M5")
            }
            Err(error) => return Err(error.into()),
        }
        .context("MLXL3_NATIVE_RECORD_REUSE must be 0 or 1")?;
        let mut cache = ExpertCache::new(capacity)?;
        let control = |name, default: &str| -> Result<String> {
            match std::env::var(name) {
                Ok(value) => Ok(value),
                Err(std::env::VarError::NotPresent) => Ok(default.into()),
                Err(error) => Err(error.into()),
            }
        };
        let force_lru =
            crate::flash_support::parse_lru_override(&control("MLXL3_SSD_CACHE_POLICY", "auto")?)
                .context("MLXL3_SSD_CACHE_POLICY must be auto or lru")?;
        let predictor_half = crate::flash_support::parse_predictor_half(&control(
            "MLXL3_SSD_PREDICTOR_DTYPE",
            "f32",
        )?)
        .context("MLXL3_SSD_PREDICTOR_DTYPE must be f32 or f16")?;
        if force_lru {
            cache.enable_predictive_lru();
        }
        let cache = Rc::new(RefCell::new(cache));
        let records = checkpoint::flash_expert_records(checkpoint, prefix)?;
        let gate = FlashGate::load(gpu, checkpoint, &format!("{prefix}.gate"))?;
        let shared = FlashExpert::load(gpu, checkpoint, &format!("{prefix}.shared_expert"))?;
        ensure!(
            shared.input_hidden() == [2560, 640],
            "invalid shared Flash expert widths"
        );
        let switch = checkpoint
            .tensors
            .get(&format!("{prefix}.shared_expert_gate.weight"))
            .context("missing Flash shared expert switch")?;
        ensure!(
            switch.shape == [1, 2560]
                && switch.dtype == "F16"
                && !checkpoint
                    .tensors
                    .contains_key(&format!("{prefix}.shared_expert_gate.bias")),
            "Flash shared expert switch must be unbiased F16 [1,2560]"
        );
        Ok(Self {
            gpu,
            gate,
            shared,
            records,
            cache,
            slots: Rc::new(RefCell::new((0..capacity).map(|_| None).collect())),
            record_backings: (0..capacity).map(|_| None).collect(),
            prefetcher: None,
            predictor_half,
            extra_wave_enabled,
            record_reuse_enabled,
            switch: read_weight(&gpu.device, switch, 5120)?,
            last: Vec::new(),
        })
    }

    pub(super) fn enable_prefetch(&mut self, pool: Rc<RefCell<PrefetchPool<'a>>>) {
        self.record_backings
            .iter_mut()
            .for_each(|owner| *owner = None);
        self.cache.borrow_mut().enable_predictive_lru();
        self.prefetcher = Some(pool);
    }
    pub fn submit(&mut self, x: &MatrixView) -> Result<MoeOutput> {
        self.forward(x, false, None, 0)
    }
    pub fn trace(&mut self, x: &MatrixView) -> Result<MoeOutput> {
        self.forward(x, true, None, 0)
    }
    pub(super) fn lookahead(
        &mut self,
        x: &MatrixView,
        next: Option<&mut FlashMoeRow<'a>>,
        depth: usize,
    ) -> Result<MoeOutput> {
        self.forward(x, false, next, depth)
    }
    pub(super) fn joint_gate_eligible(&self, depth: usize) -> bool {
        crate::flash_support::hc_gate_chain_eligible(
            self.gpu.hc_gate_chain.get(),
            self.gpu.hc_chain.get(),
            depth,
            self.prefetcher.is_some(),
        )
    }
    pub(super) fn encode_gate(
        &self,
        owner: &Metal,
        cmd: &metal::CommandBufferRef,
        x: &MatrixView,
    ) -> Result<super::gate::GateBuffers> {
        ensure!(
            std::ptr::eq(self.gpu, owner),
            "MoE gate chain requires the same Metal instance"
        );
        self.gate.encode(owner, cmd, x, Storage::StorageModePrivate)
    }
    pub(super) fn with_completed_gate(
        &mut self,
        x: &MatrixView,
        routing: super::RouteOutput,
    ) -> Result<MoeOutput> {
        ensure!(self.joint_gate_eligible(0), "ineligible joint HC gate");
        self.forward_routed(x, false, None, 0, Some(routing))
    }
    fn forward(
        &mut self,
        x: &MatrixView,
        trace: bool,
        next: Option<&mut FlashMoeRow<'a>>,
        depth: usize,
    ) -> Result<MoeOutput> {
        self.forward_routed(x, trace, next, depth, None)
    }
    fn forward_routed(
        &mut self,
        x: &MatrixView,
        trace: bool,
        mut next: Option<&mut FlashMoeRow<'a>>,
        depth: usize,
        routing: Option<super::RouteOutput>,
    ) -> Result<MoeOutput> {
        ensure!(depth <= 10, "lookahead depth must be in 0..=10");
        if let Some(pool) = &self.prefetcher {
            pool.borrow_mut().poll(None, &[], false)?;
        }
        ensure!(
            x.shape() == [1, 2560],
            "Flash MoE expects one 2560-wide AR row"
        );
        #[cfg(test)]
        let epoch = std::time::Instant::now();
        let gpu = self.gpu;
        let dtype = usize::from(x.dtype == FloatType::F32);
        let switch = &gpu.moe_switch[dtype];
        ensure!(
            switch.thread_execution_width() == 32
                && switch.max_total_threads_per_threadgroup() >= 512,
            "unsupported Flash shared expert switch threadgroup"
        );
        let routing = match routing {
            Some(routing) => routing,
            None => self.gate.submit(x)?,
        };
        // SSD demand starts as soon as the route is known. This queue has also passed its previous row.
        let ids = routing.indices()?;
        let weights = routing.completed_scores()?;
        if let Some(pool) = &self.prefetcher {
            pool.borrow_mut().poll(Some(&self.cache), &ids, false)?;
        }
        let reuse_requested = crate::ssd_reads::direct_read_eligible(
            self.record_reuse_enabled && gpu.ssd_direct_enabled,
            depth,
            self.prefetcher.is_some(),
        );
        let quiescent = if reuse_requested {
            self.drain_commands()?;
            true
        } else {
            for last in self.last.drain(..) {
                ensure!(
                    last.status() == MTLCommandBufferStatus::Completed,
                    "previous Flash MoE command failed ({:?})",
                    last.status()
                );
            }
            false
        };
        let reuse = crate::ssd_reads::record_reuse_eligible(
            reuse_requested,
            depth,
            self.prefetcher.is_some(),
            quiescent,
        );
        let guard = ExpertFill::new(&self.cache);
        let plan = DemandPlan::reserve(&self.cache, &ids)?;
        let hits = plan.routes.iter().filter(|(_, hit)| *hit).count();
        let mut owners = Vec::with_capacity(plan.reads.len());
        for lease in &plan.reads {
            let old_views = self.slots.borrow_mut()[lease.slot].take();
            let owner = self.record_backings[lease.slot].take();
            // Snapshots contain recurrent state, never expert weights. With no
            // prefetch, these private slots were the last future-access views.
            drop(old_views);
            owners.push(if reuse { owner } else { None });
        }
        let records = || {
            plan.reads
                .iter()
                .map(|&lease| (lease, self.records[lease.expert as usize].clone()))
        };
        let mut reads = if crate::ssd_reads::direct_read_eligible(
            gpu.ssd_direct_enabled,
            depth,
            self.prefetcher.is_some(),
        ) {
            gpu.readers()?.submit_shared_reusing(
                records()
                    .zip(owners)
                    .map(|((lease, record), owner)| (lease, record, owner)),
                &gpu.device,
            )?
        } else {
            gpu.readers()?.submit(records())?
        };
        // Demanded I/O has already started before the optional prediction is submitted.
        let prediction = if depth > 0 {
            next.as_ref()
                .map(|moe| moe.gate.predict(x, moe.predictor_half))
                .transpose()?
        } else {
            None
        };
        let output = objc::rc::autoreleasepool(|| {
            #[cfg(test)]
            let mut timeline = Vec::new();
            #[cfg(test)]
            let mut extra_wave_submissions = 0;
            let mut outputs = BTreeMap::new();
            let multiplier = MatrixView::output(
                gpu,
                x.dtype,
                1,
                if trace { 2 } else { 1 },
                if trace {
                    Storage::StorageModeShared
                } else {
                    Storage::StorageModePrivate
                },
            )?;
            let values = MatrixView::output(gpu, x.dtype, 1, 2560, Storage::StorageModeShared)?;
            let slab = if gpu.expert_ar_enabled[2].get() && !trace && x.layout.shape == [1, 2560] {
                let backing = buffer(&gpu.device, 51200, Storage::StorageModePrivate)?;
                #[cfg(test)]
                gpu.expert_ar_uses[2].set(gpu.expert_ar_uses[2].get() + 1);
                Some(backing)
            } else {
                None
            };
            ensure!(plan.routes.len() == 10, "invalid contribution route count");
            let canonical_slots: [usize; 10] = std::array::from_fn(|index| plan.routes[index].0);
            let destination = |slot| -> Result<Option<(&Buffer, u64)>> {
                slab.as_ref()
                    .map(|backing| {
                        let route = canonical_slots
                            .iter()
                            .position(|&candidate| candidate == slot)
                            .context("missing canonical route")?;
                        let offset = crate::expert_ar_policy::prefix(
                            route,
                            canonical_slots.len(),
                            2560,
                            backing.length(),
                        )
                        .context("invalid canonical contribution prefix")?;
                        Ok((backing, offset))
                    })
                    .transpose()
            };
            let mut command = gpu.queue.new_command_buffer().to_owned();
            let shared = self
                .shared
                .encode_view(&command, x, Storage::StorageModePrivate)?;
            Metal::encode(
                &command,
                switch,
                &[&x.buffer, &self.switch, &multiplier.buffer],
                [x.layout.offset, x.layout.strides[1], i64::from(trace)],
                MTLSize::new(1, 1, 1),
                512,
            );
            let axis = gpu.expert_ar_enabled[3].get() && !trace && x.layout.shape == [1, 2560];
            if axis {
                let mut hot = Vec::new();
                for &(slot, hit) in &plan.routes {
                    if hit && !hot.contains(&slot) {
                        hot.push(slot);
                    }
                }
                if !hot.is_empty() {
                    let slots = self.slots.borrow();
                    let experts = hot
                        .iter()
                        .map(|&slot| slots[slot].as_ref().context("missing resident expert"))
                        .collect::<Result<Vec<_>>>()?;
                    let targets = hot
                        .iter()
                        .map(|&slot| destination(slot))
                        .collect::<Result<Vec<_>>>()?;
                    let contributions = FlashExpert::encode_ready_cohort(
                        &command,
                        x,
                        &experts,
                        &targets,
                        Storage::StorageModePrivate,
                    )?;
                    outputs.extend(hot.into_iter().zip(contributions));
                }
            } else {
                for &(slot, hit) in &plan.routes {
                    if hit && !outputs.contains_key(&slot) {
                        outputs.insert(
                            slot,
                            self.slots.borrow()[slot]
                                .as_ref()
                                .context("missing resident expert")?
                                .encode_view_into(
                                    &command,
                                    x,
                                    Storage::StorageModePrivate,
                                    destination(slot)?,
                                )?,
                        );
                    }
                }
            }
            // Every cold wave binds only CPU-complete, generation-checked immutable buffers.
            if reads.remaining() > 0 {
                #[cfg(test)] timeline.push(serde_json::json!({"event":"hot_submit","time_ns":epoch.elapsed().as_nanos(),"remaining_reads":reads.remaining()}));
                command.commit();
                self.last.push(command);
                command = gpu.queue.new_command_buffer().to_owned();
                let mut wave = reads.ready(false)?;
                let mut extra_used = false;
                loop {
                    let mut ready = Vec::new();
                    for (lease, payload) in wave {
                        let direct = matches!(&payload, crate::ssd_reads::ReadPayload::Shared(_));
                        let backing = gpu.completed_record(payload)?;
                        let expert = FlashExpert::from_backing(gpu, &backing)?;
                        ensure!(plan.publish(lease), "stale native SSD demand completion");
                        self.slots.borrow_mut()[lease.slot] = Some(expert);
                        if reuse && direct {
                            self.record_backings[lease.slot] = Some(backing);
                        }
                        if axis {
                            ready.push(lease.slot);
                        } else {
                            outputs.insert(
                                lease.slot,
                                self.slots.borrow()[lease.slot]
                                    .as_ref()
                                    .context("missing cold expert")?
                                    .encode_view_into(
                                        &command,
                                        x,
                                        Storage::StorageModePrivate,
                                        destination(lease.slot)?,
                                    )?,
                            );
                        }
                    }
                    if axis && !ready.is_empty() {
                        let slots = self.slots.borrow();
                        let experts = ready
                            .iter()
                            .map(|&slot| slots[slot].as_ref().context("missing cold expert"))
                            .collect::<Result<Vec<_>>>()?;
                        let targets = ready
                            .iter()
                            .map(|&slot| destination(slot))
                            .collect::<Result<Vec<_>>>()?;
                        let contributions = FlashExpert::encode_ready_cohort(
                            &command,
                            x,
                            &experts,
                            &targets,
                            Storage::StorageModePrivate,
                        )?;
                        outputs.extend(ready.into_iter().zip(contributions));
                    }
                    if reads.remaining() == 0 {
                        break;
                    }
                    #[cfg(test)] timeline.push(serde_json::json!({"event":if extra_used { "extra_submit" } else { "w1_submit" },"time_ns":epoch.elapsed().as_nanos(),"remaining_reads":reads.remaining()}));
                    command.commit();
                    self.last.push(command);
                    #[cfg(test)]
                    if extra_used {
                        extra_wave_submissions += 1;
                        EXTRA_WAVE_SUBMISSIONS.with(|count| count.set(count.get() + 1));
                    }
                    let extra = crate::ssd_reads::extra_wave(
                        crate::ssd_reads::direct_read_eligible(
                            self.extra_wave_enabled,
                            depth,
                            self.prefetcher.is_some(),
                        ),
                        reads.remaining(),
                        extra_used,
                    );
                    #[cfg(test)]
                    {
                        gpu.readers()?.release_test_gate();
                        gpu.readers()?.release_wave_test_gate(!extra);
                    }
                    command = gpu.queue.new_command_buffer().to_owned();
                    wave = if extra {
                        extra_used = true;
                        reads.ready_min(2)?
                    } else {
                        reads.ready(true)?
                    };
                }
            }
            let mut bindings = plan
                .routes
                .iter()
                .map(|(slot, _)| {
                    let output = outputs
                        .get(slot)
                        .context("missing ordered expert contribution")?;
                    Ok((output, destination(*slot)?.map_or(0, |(_, offset)| offset)))
                })
                .collect::<Result<Vec<_>>>()?;
            bindings.extend([
                (&shared, 0),
                (&weights.buffer, 0),
                (&multiplier.buffer, 0),
                (&values.buffer, 0),
            ]);
            Metal::encode_bound(
                &command,
                &gpu.moe_mix[dtype],
                bindings.into_iter(),
                [2560u32],
                MTLSize::new(80, 1, 1),
                32,
            );
            #[cfg(test)]
            {
                timeline.push(serde_json::json!({"event":"final_submit","time_ns":epoch.elapsed().as_nanos(),"remaining_reads":reads.remaining()}));
                for (lease, started, ended) in &reads.trace {
                    timeline.push(serde_json::json!({"event":"read","expert":lease.expert,"slot":lease.slot,"start_ns":started.duration_since(epoch).as_nanos(),"end_ns":ended.duration_since(epoch).as_nanos()}));
                }
            }
            command.commit();
            self.last.push(command);
            guard.commit();
            Ok(MoeOutput {
                commands: self.last.clone(),
                #[cfg(test)]
                timeline,
                #[cfg(test)]
                extra_wave_submissions,
                #[cfg(test)]
                injected_failure: false,
                values,
                read_bytes: plan.reads.len() as u64 * EXPERT_BYTES,
                multiplier_trace: trace.then_some(multiplier),
                indices: ids,
                hits,
            })
        })?;
        #[cfg(test)]
        if let Some(span) = crate::flash_profile::span("native.ssd_timeline") {
            span.data(serde_json::json!({"events":output.timeline,"read_bytes":output.read_bytes,"indices":output.indices,"hits":output.hits,"clock":"relative to MoE CPU epoch; do not align with GPU clock"}));
        }
        if let (Some(next), Some(prediction)) = (next.as_mut(), prediction) {
            let mut ids = prediction.indices()?;
            ensure!(
                crate::ssd_prefetch::order_predictions(&mut ids, 1),
                "invalid native lookahead prediction"
            );
            next.prefetch(&ids, depth)?;
        }
        Ok(output)
    }
    fn prefetch(&mut self, ids: &[u32], budget: usize) -> Result<()> {
        ensure!(
            !ids.is_empty() && ids.len() <= 32 && ids.iter().all(|&id| id < 512),
            "invalid predicted experts"
        );
        let Some(pool) = self.prefetcher.clone() else {
            return Ok(());
        };
        let mut pool = pool.borrow_mut();
        pool.poll(None, &[], false)?;
        self.cache.borrow_mut().protect_predictions(ids)?;
        let budget = budget.min(4).min(pool.room());
        if budget == 0 {
            return Ok(());
        }
        self.drain_commands()?;
        let mut submitted = 0;
        for &id in ids {
            if submitted == budget {
                break;
            }
            let Some(lease) = self.cache.borrow_mut().claim_lease(id)? else {
                continue;
            };
            self.slots.borrow_mut()[lease.slot] = None;
            pool.submit(
                lease,
                self.records[id as usize].clone(),
                &self.cache,
                &self.slots,
            )?;
            submitted += 1;
        }
        Ok(())
    }
}

impl FlashMoeRow<'_> {
    pub(super) fn record_memory(&self, backings: &mut crate::gpu::memory::Backings) -> Result<()> {
        backings.slots.push(self.slots.borrow().len());
        self.gate.record_memory(backings)?;
        self.shared.record_memory(backings, "fixed_weights")?;
        backings.record("fixed_weights", &self.switch)?;
        for expert in self.slots.borrow().iter().flatten() {
            backings.resident_experts += 1;
            expert.record_memory(backings, "expert_banks")?;
        }
        Ok(())
    }
    pub(super) fn drain(&mut self) -> Result<()> {
        if let Some(pool) = &self.prefetcher {
            pool.borrow_mut().drain()?;
        }
        self.gpu.ensure_reads_idle()?;
        self.drain_commands()
    }
    fn drain_commands(&mut self) -> Result<()> {
        let mut failure = None;
        for command in &self.last {
            command.wait_until_completed();
            if command.status() != MTLCommandBufferStatus::Completed && failure.is_none() {
                failure = Some(anyhow::anyhow!("Flash MoE command failed during drain"));
            }
        }
        if let Some(error) = failure {
            return Err(error);
        }
        self.last.clear();
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn clear_test_cache(&mut self) -> Result<()> {
        self.drain()?;
        self.cache.borrow_mut().invalidate();
        self.slots
            .borrow_mut()
            .iter_mut()
            .for_each(|slot| *slot = None);
        self.record_backings
            .iter_mut()
            .for_each(|owner| *owner = None);
        Ok(())
    }
    pub(super) fn shrink(&mut self, slots: usize) -> Result<()> {
        ensure!(
            (10..=self.slots.borrow().len()).contains(&slots),
            "invalid Flash cache shrink"
        );
        if slots == self.slots.borrow().len() {
            return Ok(());
        }
        self.drain()?;
        let mut cache = ExpertCache::new(slots)?;
        if self.prefetcher.is_some() || self.cache.borrow().predictive_lru_enabled() {
            cache.enable_predictive_lru();
        }
        self.slots = Rc::new(RefCell::new((0..slots).map(|_| None).collect()));
        self.record_backings = (0..slots).map(|_| None).collect();
        self.cache = Rc::new(RefCell::new(cache));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_layout::MatrixLayout;

    #[test]
    #[ignore = "physical Flash checkpoint: bounded extra cold wave versus old schedule"]
    fn bounded_extra_wave_submits_two_records_before_the_held_tail() -> Result<()> {
        let path = std::env::var_os("MLXL3_MOE_CHECKPOINT").context("Flash checkpoint required")?;
        let checkpoint = checkpoint::inspect(std::path::Path::new(&path))?;
        let gpu = Metal::new()?;
        let initial = crate::ssd_reads::extra_wave_flag(
            std::env::var("MLXL3_NATIVE_SSD_EXTRA_WAVE").ok().as_deref(),
            gpu.device.name() == "Apple M5",
        );
        ensure!(
            initial == Some(true),
            "requires explicit extra-wave flag1 on qualified M5"
        );
        let input = (0..2560)
            .map(|i| ((i % 73) as f32 - 36.) / 53.)
            .collect::<Vec<_>>();
        let initial_submissions = EXTRA_WAVE_SUBMISSIONS.with(std::cell::Cell::get);
        let mut cases = 0;
        for fp32 in [false, true] {
            let layout = MatrixLayout {
                shape: [1, 2560],
                strides: [2560, 1],
                offset: 0,
            };
            let x = if fp32 {
                gpu.f32_storage(&input)?
            } else {
                gpu.f16_storage(
                    &input
                        .iter()
                        .map(|&v| half::f16::from_f32(v).to_bits())
                        .collect::<Vec<_>>(),
                )?
            }
            .matrix(layout)?;
            let mut expected = None;
            for enabled in [false, true] {
                let mut layer =
                    FlashMoeRow::load(&gpu, &checkpoint, "model.language_model.layers.0.mlp", 10)?;
                ensure!(layer.extra_wave_enabled, "load did not snapshot flag1");
                layer.extra_wave_enabled = enabled;
                let gates = gpu.readers()?.set_wave_test_gates();
                let outcome = (|| -> Result<Vec<u32>> {
                    let output = layer.submit(&x)?;
                    ensure!(output.hits == 0 && output.read_bytes == 10 * EXPERT_BYTES);
                    ensure!(output.commands.len() == if enabled { 4 } else { 3 });
                    ensure!(output.extra_wave_submissions == usize::from(enabled));
                    let extras = output
                        .timeline
                        .iter()
                        .filter(|v| v["event"] == "extra_submit")
                        .collect::<Vec<_>>();
                    ensure!(extras.len() == usize::from(enabled));
                    if enabled {
                        ensure!(extras[0]["remaining_reads"] == 7);
                    }
                    let values = output.read()?;
                    ensure!(values.len() == 2560 && values.iter().any(|&v| v != 0));
                    ensure!(values.iter().all(|&v| if fp32 {
                        f32::from_bits(v).is_finite()
                    } else {
                        half::f16::from_bits(v as u16).is_finite()
                    }));
                    let hot = layer.submit(&x)?;
                    ensure!(
                        hot.hits == 10
                            && hot.read_bytes == 0
                            && hot.commands.len() == 1
                            && hot.extra_wave_submissions == 0
                    );
                    ensure!(hot.read()? == values);
                    Ok(values)
                })();
                for gate in gates {
                    gate.release();
                }
                gpu.readers()?.release_wave_test_gate(true);
                let values = outcome?;
                gpu.ensure_reads_idle()?;
                if let Some(expected) = &expected {
                    ensure!(&values == expected, "OFF/ON cold wave mismatch");
                } else {
                    expected = Some(values);
                }
                cases += 1;
            }
        }
        let actual_submissions =
            EXTRA_WAVE_SUBMISSIONS.with(std::cell::Cell::get) - initial_submissions;
        ensure!(cases == 4 && actual_submissions == 2);
        eprintln!(
            "NATIVE_SSD_EXTRA_WAVE {}",
            serde_json::json!({"cases":cases,"finite_words":cases*2560,"actual_extra_submissions":actual_submissions,"max_commands":4,"oracle":"unchanged OFF schedule; independent full-model oracle separate"})
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical Flash checkpoint: next-gate predictions, bounded prefetch, faults and shrink"]
    fn native_lookahead_preserves_routes_and_drains_generation_owned_reads() -> Result<()> {
        let path = std::env::var_os("MLXL3_MOE_CHECKPOINT").context("Flash checkpoint required")?;
        let checkpoint = checkpoint::inspect(std::path::Path::new(&path))?;
        let gpu = Metal::new()?;
        let prefix = "model.language_model.layers.0.mlp";
        let input = (0..2560)
            .map(|i| ((i % 73) as f32 - 36.) / 53.)
            .collect::<Vec<_>>();
        let mut cases = 0;
        let mut reused_cases = 0;
        for capacity in [10, 19, 20, 24] {
            for fp32 in [false, true] {
                let layout = MatrixLayout {
                    shape: [1, 2560],
                    strides: [2560, 1],
                    offset: 0,
                };
                let x = if fp32 {
                    gpu.f32_storage(&input)?
                } else {
                    gpu.f16_storage(
                        &input
                            .iter()
                            .map(|&v| half::f16::from_f32(v).to_bits())
                            .collect::<Vec<_>>(),
                    )?
                }
                .matrix(layout)?;
                let mut current = FlashMoeRow::load(&gpu, &checkpoint, prefix, capacity)?;
                let mut next = FlashMoeRow::load(&gpu, &checkpoint, prefix, capacity)?;
                let pool = Rc::new(RefCell::new(PrefetchPool::new(&gpu)?));
                current.enable_prefetch(pool.clone());
                next.enable_prefetch(pool.clone());
                let expected = current.submit(&x)?.read()?;
                ensure!(expected.len() == 2560 && expected.iter().any(|&v| v != 0));
                ensure!(expected.iter().all(|&v| if fp32 {
                    f32::from_bits(v).is_finite()
                } else {
                    half::f16::from_bits(v as u16).is_finite()
                }));
                for predictor_half in [false, true] {
                    next.predictor_half = predictor_half;
                    let oracle_x;
                    let oracle_x = if predictor_half && fp32 {
                        oracle_x = gpu
                            .f16_storage(
                                &input
                                    .iter()
                                    .map(|&v| half::f16::from_f32(v).to_bits())
                                    .collect::<Vec<_>>(),
                            )?
                            .matrix(layout)?;
                        &oracle_x
                    } else {
                        &x
                    };
                    let trace = next.gate.trace(oracle_x)?;
                    let logits = serde_json::to_value(trace)?["logits"]
                        .as_array()
                        .context("missing predictor logits")?
                        .iter()
                        .map(|v| {
                            let bits = v.as_u64().unwrap() as u32;
                            if oracle_x.dtype == FloatType::F32 {
                                f32::from_bits(bits)
                            } else {
                                half::f16::from_bits(bits as u16).to_f32()
                            }
                        })
                        .collect::<Vec<_>>();
                    ensure!(logits.len() == 512 && logits.iter().all(|v| v.is_finite()));
                    let mut sorted = (0..512u32).collect::<Vec<_>>();
                    sorted.sort_by(|&a, &b| {
                        logits[a as usize]
                            .partial_cmp(&logits[b as usize])
                            .unwrap()
                            .then(a.cmp(&b))
                    });
                    ensure!(next.gate.predict(&x, predictor_half)?.indices()? == sorted[496..]);
                    ensure!(current.lookahead(&x, Some(&mut next), 4)?.read()? == expected);
                    pool.borrow_mut().drain()?;
                    ensure!(next.submit(&x)?.read()? == expected);
                    ensure!(pool.borrow().pending_count() == 0 && pool.borrow().in_flight() == 0);
                    cases += 1;
                }
                if capacity == 10 || capacity == 20 {
                    ensure!(pool.borrow().stats.submitted == 0);
                } else {
                    ensure!(pool.borrow().stats.submitted > 0 && pool.borrow().stats.published > 0);
                }
                if capacity == 24 {
                    let actual = next.gate.submit(&x)?.indices()?;
                    next.cache.borrow_mut().invalidate();
                    next.prefetch(&actual[..1], 1)?;
                    ensure!(pool.borrow().pending_count() == 1);
                    let reused = pool.borrow().stats.demand_reused;
                    pool.borrow_mut().poll(Some(&next.cache), &actual, false)?;
                    ensure!(pool.borrow().stats.demand_reused == reused + 1);
                    let adopted = next.submit(&x)?;
                    ensure!(adopted.hits > 0 && adopted.read()? == expected);
                    reused_cases += 1;
                    let cold = (0..512)
                        .filter(|id| !next.cache.borrow().contains(*id) && !actual.contains(id))
                        .take(4)
                        .collect::<Vec<_>>();
                    ensure!(cold.len() == 4);
                    next.prefetch(&[cold[0], cold[0], cold[1]], 4)?;
                    ensure!(pool.borrow().pending_count() == 2);
                    pool.borrow_mut().drain()?;
                    ensure!(next.submit(&x)?.read()? == expected);
                    let fault = cold[2] as usize;
                    let old = next.records[fault].data_offsets;
                    next.records[fault].data_offsets = [u64::MAX - EXPERT_BYTES, u64::MAX];
                    ensure!(next.prefetch(&[fault as u32], 1).is_err());
                    next.records[fault].data_offsets = old;
                    ensure!(!next.cache.borrow().contains(fault as u32));
                    next.prefetch(&[fault as u32], 1)?;
                    // Cancel a lease before its physical completion; its bytes must be discarded.
                    let lease = pool
                        .borrow()
                        .lease_for(&next.cache, fault as u32)
                        .context("missing test prefetch lease")?;
                    ensure!(!next.cache.borrow_mut().finish_lease(lease, false));
                    let discarded = pool.borrow().stats.discarded;
                    pool.borrow_mut().drain()?;
                    ensure!(
                        pool.borrow().stats.discarded == discarded + 1
                            && !next.cache.borrow().contains(fault as u32)
                    );
                    // The file is valid at submission and truncated while its real worker is gated.
                    let dir = tempfile::tempdir()?;
                    let path = dir.path().join("late-truncated.safetensors");
                    let header = serde_json::to_vec(
                        &serde_json::json!({"first":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},"rest":{"dtype":"U8","shape":[EXPERT_BYTES-4],"data_offsets":[4,EXPERT_BYTES]}}),
                    )?;
                    let mut file = std::fs::File::create(&path)?;
                    use std::io::Write;
                    file.write_all(&(header.len() as u64).to_le_bytes())?;
                    file.write_all(&header)?;
                    file.write_all(&vec![0; EXPERT_BYTES as usize])?;
                    let record = crate::checkpoint::read_header(&path)?["first"].clone();
                    let original = std::mem::replace(&mut next.records[fault], record);
                    pool.borrow()
                        .set_test_gate(std::sync::Arc::new(std::sync::Barrier::new(2)));
                    next.prefetch(&[fault as u32], 1)?;
                    file.set_len(0)?;
                    pool.borrow().release_test_gate();
                    let failed = pool.borrow().stats.failed;
                    ensure!(pool.borrow_mut().drain().is_err());
                    next.records[fault] = original;
                    ensure!(
                        pool.borrow().stats.failed == failed + 1
                            && !next.cache.borrow().contains(fault as u32)
                    );
                    ensure!(next.submit(&x)?.read()? == expected);
                    next.prefetch(&[cold[3]], 1)?;
                    next.shrink(10)?;
                    ensure!(next.slots.borrow().len() == 10 && pool.borrow().pending_count() == 0);
                    ensure!(next.submit(&x)?.read()? == expected);
                    ensure!(next.prefetch(&[], 4).is_err() && next.prefetch(&[512], 4).is_err());
                }
            }
        }
        ensure!(cases == 16 && reused_cases == 2);
        eprintln!(
            "native prefetch: {}",
            serde_json::json!({"cases":cases,"explicit_pending_demand_reuse_cases":reused_cases,"oracle":"CPU stable ascending gate-logit sort, then last sixteen","outputs_compared":cases*2560,"capacity_boundaries":[10,19,20,24]})
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical Flash checkpoint: partial fill failure, recovery, reuse and early release"]
    #[allow(
        unexpected_cfgs,
        reason = "objc 0.2 selector macro tests the legacy cargo-clippy cfg"
    )]
    fn partial_record_read_recovers_before_a_replacement_is_visible() -> Result<()> {
        let path = std::env::var_os("MLXL3_MOE_CHECKPOINT").context("Flash checkpoint required")?;
        let checkpoint = checkpoint::inspect(std::path::Path::new(&path))?;
        let gpu = Metal::new()?;
        let mut short = tempfile::tempfile()?;
        std::io::Write::write_all(&mut short, &[7; 32])?;
        let eof = super::super::read_buffer(&gpu.device, &short, 16, 32)
            .err()
            .context("short record published")?;
        ensure!(
            eof.downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::UnexpectedEof)
        );
        let mut layer =
            FlashMoeRow::load(&gpu, &checkpoint, "model.language_model.layers.0.mlp", 10)?;
        let x = gpu
            .f32_storage(
                &(0..2560)
                    .map(|i| ((i % 73) as f32 - 36.) / 53.)
                    .collect::<Vec<_>>(),
            )?
            .matrix(MatrixLayout {
                shape: [1, 2560],
                strides: [0, 1],
                offset: 0,
            })?;
        let ids = layer.gate.submit(&x)?.indices()?;
        let id = ids[5] as usize;
        let old = layer.records[id].data_offsets;
        layer.records[id].data_offsets = [u64::MAX - EXPERT_BYTES, u64::MAX];
        let failure = layer
            .submit(&x)
            .err()
            .context("partially failed read accepted")?;
        assert!(format!("{failure:#}").contains("offset"), "{failure:#}");
        layer.records[id].data_offsets = old;
        gpu.readers()?
            .set_test_gate(Some(std::sync::Arc::new(std::sync::Barrier::new(2))));
        let first = layer.submit(&x)?;
        let second = layer.submit(&x)?;
        ensure!(
            first.extra_wave_submissions <= 1
                && first.commands.len() == 3 + first.extra_wave_submissions
                && second.commands.len() == 1
                && first.hits == 0
                && first.read_bytes == 10 * EXPERT_BYTES
                && second.hits == 10
                && second.read_bytes == 0
        );
        drop(layer);
        drop(gpu);
        drop(x);
        drop(checkpoint);
        for (label, output) in [("cold", &first), ("hot", &second)] {
            output.completed()?;
            let gpu_times = output
                .commands
                .iter()
                .map(|c| {
                    use objc::{msg_send, sel, sel_impl};
                    let start: f64 = unsafe { msg_send![c.as_ref(), GPUStartTime] };
                    let end: f64 = unsafe { msg_send![c.as_ref(), GPUEndTime] };
                    serde_json::json!({"start":start,"end":end})
                })
                .collect::<Vec<_>>();
            ensure!(gpu_times.iter().all(|v| v["start"].as_f64().is_some_and(
                |start| start > 0. && v["end"].as_f64().is_some_and(|end| end >= start)
            )));
            eprintln!(
                "native SSD {label}: {}",
                serde_json::json!({"cpu":output.timeline,"gpu":gpu_times})
            );
        }
        let expected = second.read()?;
        ensure!(
            expected.len() == 2560 && expected.iter().any(|&x| x != 0) && first.read()? == expected
        );
        Ok(())
    }
}
