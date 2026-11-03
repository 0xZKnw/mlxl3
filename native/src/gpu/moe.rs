use super::{FlashExpert, FlashGate, MatrixView, Metal, read_weight, view::FloatType};
use crate::{
    checkpoint::{self, Checkpoint, TensorInfo},
    flash_support::{EXPERT_BYTES, ExpertCache, ExpertFill},
};
use anyhow::{Context, Result, ensure};
use metal::{
    Buffer, CommandBuffer, MTLCommandBufferStatus, MTLResourceOptions as Storage, MTLSize,
};
use std::{cell::RefCell, rc::Rc};

/// One AR row, with immutable per-slot compressed records and demand-only I/O.
pub struct FlashMoeRow<'a> {
    gpu: &'a Metal,
    gate: FlashGate<'a>,
    shared: FlashExpert<'a>,
    switch: Buffer,
    records: Vec<TensorInfo>,
    cache: Rc<RefCell<ExpertCache>>,
    slots: Vec<Option<FlashExpert<'a>>>,
    last: Option<CommandBuffer>,
}

pub struct MoeOutput {
    command: CommandBuffer,
    values: MatrixView,
    pub indices: Vec<u32>,
    pub hits: usize,
    pub read_bytes: u64,
    multiplier_trace: Option<MatrixView>,
}

impl MoeOutput {
    pub fn completed(&self) -> Result<&MatrixView> {
        self.command.wait_until_completed();
        ensure!(
            self.command.status() == MTLCommandBufferStatus::Completed,
            "Flash MoE command failed ({:?})",
            self.command.status()
        );
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
            last: None,
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
        if let Some(last) = self.last.take() {
            ensure!(
                last.status() == MTLCommandBufferStatus::Completed,
                "previous Flash MoE command failed ({:?})",
                last.status()
            );
        }
        let guard = ExpertFill::new(&self.cache);
        let reserved = self.cache.borrow_mut().reserve(&ids)?;
        let mut hits = 0;
        for (&id, &(slot, hit)) in ids.iter().zip(&reserved) {
            if hit {
                hits += 1;
            } else {
                // Drop this completed generation before allocating its replacement.
                self.slots[slot] = None;
                self.slots[slot] = Some(FlashExpert::from_record(gpu, &self.records[id as usize])?);
            }
        }
        guard.commit();
        objc::rc::autoreleasepool(|| {
            let cmd = gpu.queue.new_command_buffer();
            let mut outputs = Vec::with_capacity(14);
            for (slot, _) in reserved {
                outputs.push(
                    self.slots[slot]
                        .as_ref()
                        .context("missing published expert")?
                        .encode_view(cmd, x, Storage::StorageModePrivate)?,
                );
            }
            outputs.push(
                self.shared
                    .encode_view(cmd, x, Storage::StorageModePrivate)?,
            );
            outputs.push(weights.buffer.clone());
            let multiplier = if trace {
                MatrixView::output(gpu, x.dtype, 1, 2, Storage::StorageModeShared)?
            } else {
                MatrixView::output(gpu, x.dtype, 1, 1, Storage::StorageModePrivate)?
            };
            Metal::encode(
                cmd,
                switch,
                &[&x.buffer, &self.switch, &multiplier.buffer],
                [x.layout.offset, x.layout.strides[1], i64::from(trace)],
                MTLSize::new(1, 1, 1),
                512,
            );
            outputs.push(multiplier.buffer.clone());
            let values = MatrixView::output(gpu, x.dtype, 1, 2560, Storage::StorageModeShared)?;
            outputs.push(values.buffer.clone());
            Metal::encode_bound(
                cmd,
                &gpu.moe_mix[dtype],
                outputs.iter().map(|b| (b, 0)),
                [2560u32],
                MTLSize::new(80, 1, 1),
                32,
            );
            cmd.commit();
            self.last = Some(cmd.to_owned());
            Ok(MoeOutput {
                command: cmd.to_owned(),
                values,
                read_bytes: (ids.len() - hits) as u64 * EXPERT_BYTES,
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
        layer.records[id].data_offsets = [u64::MAX; 2];
        let failure = layer
            .submit(&x)
            .err()
            .context("partially failed read accepted")?;
        assert!(format!("{failure:#}").contains("offset"), "{failure:#}");
        layer.records[id].data_offsets = old;
        let first = layer.submit(&x)?;
        let second = layer.submit(&x)?;
        ensure!(
            first.hits == 0
                && first.read_bytes == 10 * EXPERT_BYTES
                && second.hits == 10
                && second.read_bytes == 0
        );
        drop(layer);
        drop(gpu);
        drop(x);
        drop(checkpoint);
        let expected = second.read()?;
        ensure!(
            expected.len() == 2560 && expected.iter().any(|&x| x != 0) && first.read()? == expected
        );
        Ok(())
    }
}
