use super::{FlashExpert, FlashGate, MatrixView, Metal, read_weight, view::FloatType};
use crate::{
    checkpoint::{self, Checkpoint, TensorInfo},
    flash_support::{DemandPlan, EXPERT_BYTES, ExpertCache, ExpertFill},
};
use anyhow::{Context, Result, ensure};
use metal::{
    Buffer, CommandBuffer, MTLCommandBufferStatus, MTLResourceOptions as Storage, MTLSize,
};
use std::{cell::RefCell, collections::BTreeMap, rc::Rc};

/// One AR row, with immutable per-slot compressed records and demand-only I/O.
pub struct FlashMoeRow<'a> {
    gpu: &'a Metal,
    gate: FlashGate<'a>,
    shared: FlashExpert<'a>,
    switch: Buffer,
    records: Vec<TensorInfo>,
    cache: Rc<RefCell<ExpertCache>>,
    slots: Vec<Option<FlashExpert<'a>>>,
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
}

impl MoeOutput {
    pub fn completed(&self) -> Result<&MatrixView> {
        for command in &self.commands {
            command.wait_until_completed();
            ensure!(
                command.status() == MTLCommandBufferStatus::Completed,
                "Flash MoE command failed ({:?})",
                command.status()
            );
        }
        Ok(&self.values)
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
        let cache = Rc::new(RefCell::new(ExpertCache::new(capacity)?));
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
            slots: (0..capacity).map(|_| None).collect(),
            switch: read_weight(&gpu.device, switch, 5120)?,
            last: Vec::new(),
        })
    }

    pub fn submit(&mut self, x: &MatrixView) -> Result<MoeOutput> {
        self.forward(x, false)
    }
    pub fn trace(&mut self, x: &MatrixView) -> Result<MoeOutput> {
        self.forward(x, true)
    }
    fn forward(&mut self, x: &MatrixView, trace: bool) -> Result<MoeOutput> {
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
        let routing = self.gate.submit(x)?;
        // SSD demand starts as soon as the route is known. This queue has also passed its previous row.
        let ids = routing.indices()?;
        let weights = routing.completed_scores()?;
        for last in self.last.drain(..) {
            ensure!(
                last.status() == MTLCommandBufferStatus::Completed,
                "previous Flash MoE command failed ({:?})",
                last.status()
            );
        }
        let guard = ExpertFill::new(&self.cache);
        let plan = DemandPlan::reserve(&self.cache, &ids)?;
        let hits = plan.routes.iter().filter(|(_, hit)| *hit).count();
        for lease in &plan.reads {
            self.slots[lease.slot] = None;
        }
        let mut reads = gpu.readers()?.submit(
            plan.reads
                .iter()
                .map(|&lease| (lease, self.records[lease.expert as usize].clone())),
        )?;
        objc::rc::autoreleasepool(|| {
            #[cfg(test)]
            let mut timeline = Vec::new();
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
            for &(slot, hit) in &plan.routes {
                if hit && !outputs.contains_key(&slot) {
                    outputs.insert(
                        slot,
                        self.slots[slot]
                            .as_ref()
                            .context("missing resident expert")?
                            .encode_view(&command, x, Storage::StorageModePrivate)?,
                    );
                }
            }
            // Hot work is submitted before waiting for any cold record. The next two
            // waves bind only CPU-complete, generation-checked immutable buffers.
            if reads.remaining() > 0 {
                #[cfg(test)] timeline.push(serde_json::json!({"event":"hot_submit","time_ns":epoch.elapsed().as_nanos(),"remaining_reads":reads.remaining()}));
                command.commit();
                self.last.push(command);
                command = gpu.queue.new_command_buffer().to_owned();
                let mut wave = reads.ready(false)?;
                loop {
                    for (lease, bytes) in wave {
                        let backing = super::upload(&gpu.device, &bytes)?;
                        let expert = FlashExpert::from_backing(gpu, &backing)?;
                        ensure!(plan.publish(lease), "stale native SSD demand completion");
                        self.slots[lease.slot] = Some(expert);
                        outputs.insert(
                            lease.slot,
                            self.slots[lease.slot]
                                .as_ref()
                                .context("missing cold expert")?
                                .encode_view(&command, x, Storage::StorageModePrivate)?,
                        );
                    }
                    if reads.remaining() == 0 {
                        break;
                    }
                    #[cfg(test)] timeline.push(serde_json::json!({"event":"w1_submit","time_ns":epoch.elapsed().as_nanos(),"remaining_reads":reads.remaining()}));
                    command.commit();
                    self.last.push(command);
                    #[cfg(test)]
                    gpu.readers()?.release_test_gate();
                    command = gpu.queue.new_command_buffer().to_owned();
                    wave = reads.ready(true)?;
                }
            }
            let mut bindings = plan
                .routes
                .iter()
                .map(|(slot, _)| {
                    outputs
                        .get(slot)
                        .context("missing ordered expert contribution")
                })
                .collect::<Result<Vec<_>>>()?;
            bindings.extend([&shared, &weights.buffer, &multiplier.buffer, &values.buffer]);
            Metal::encode_bound(
                &command,
                &gpu.moe_mix[dtype],
                bindings.into_iter().map(|b| (b, 0)),
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
                values,
                read_bytes: plan.reads.len() as u64 * EXPERT_BYTES,
                multiplier_trace: trace.then_some(multiplier),
                indices: ids,
                hits,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_layout::MatrixLayout;

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
            first.commands.len() == 3
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

impl FlashMoeRow<'_> {
    pub(super) fn record_memory(&self, backings: &mut crate::gpu::memory::Backings) -> Result<()> {
        backings.slots.push(self.slots.len());
        self.gate.record_memory(backings)?;
        self.shared.record_memory(backings, "fixed_weights")?;
        backings.record("fixed_weights", &self.switch)?;
        for expert in self.slots.iter().flatten() {
            backings.resident_experts += 1;
            expert.record_memory(backings, "expert_banks")?;
        }
        Ok(())
    }
    pub(super) fn drain(&mut self) -> Result<()> {
        self.gpu.ensure_reads_idle()?;
        for command in &self.last {
            command.wait_until_completed();
            ensure!(
                command.status() == MTLCommandBufferStatus::Completed,
                "Flash MoE command failed during drain"
            );
        }
        self.last.clear();
        Ok(())
    }
    pub(super) fn shrink(&mut self, slots: usize) -> Result<()> {
        ensure!(
            (10..=self.slots.len()).contains(&slots),
            "invalid Flash cache shrink"
        );
        if slots == self.slots.len() {
            return Ok(());
        }
        self.drain()?;
        let cache = ExpertCache::new(slots)?;
        self.slots = (0..slots).map(|_| None).collect();
        self.cache = Rc::new(RefCell::new(cache));
        Ok(())
    }
}
