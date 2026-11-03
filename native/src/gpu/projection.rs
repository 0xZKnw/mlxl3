//! Resident Flash EXL3 weights and a single retained Metal submission.
use super::{MatrixView, Metal, buffer, upload};
use crate::{
    checkpoint::{Checkpoint, TensorInfo},
    codec::Codebook,
    contracts, flash_support,
};
use anyhow::{Context, Result, bail, ensure};
use half::{bf16, f16};
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, DeviceRef, MTLCommandBufferStatus,
    MTLResourceOptions as Storage, MTLSize,
};

pub struct FlashProjection<'a> {
    metal: &'a Metal,
    packed: Buffer,
    suh: Buffer,
    svh: Buffer,
    bias: Option<Buffer>,
    input: u32,
    output: u32,
    k: u32,
    cb: Codebook,
    splits: u32,
}

#[derive(Clone, Copy)]
enum Activation<'a> {
    Contiguous(&'a Buffer),
    View(&'a MatrixView),
}

pub struct ProjectionOutput {
    command: CommandBuffer,
    buffer: Buffer,
    count: usize,
}

impl ProjectionOutput {
    pub fn read(self) -> Result<Vec<u16>> {
        self.command.wait_until_completed();
        ensure!(
            self.command.status() == MTLCommandBufferStatus::Completed,
            "Flash projection command failed ({:?})",
            self.command.status()
        );
        Ok(Metal::read(&self.buffer, self.count))
    }
}

fn scale(device: &DeviceRef, tensor: &TensorInfo, width: u32) -> Result<Buffer> {
    ensure!(
        tensor.shape == [width as usize],
        "invalid projection scale/bias shape"
    );
    let bytes = tensor.read_bytes()?;
    let words: Vec<u16> = match tensor.dtype.as_str() {
        "F16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| u16::from_le_bytes(v))
            .collect(),
        "BF16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| f16::from_f32(bf16::from_le_bytes(v).to_f32()).to_bits())
            .collect(),
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&v| f16::from_f32(f32::from_le_bytes(v)).to_bits())
            .collect(),
        dtype => bail!("unsupported projection scale/bias dtype {dtype}"),
    };
    ensure!(words.len() == width as usize, "truncated scale/bias");
    upload(device, &words)
}

impl<'a> FlashProjection<'a> {
    pub fn load(metal: &'a Metal, checkpoint: &Checkpoint, name: &str) -> Result<Self> {
        ensure!(
            checkpoint.model_type == "qwen4_exp",
            "requires a Flash reference checkpoint"
        );
        let tensors = &checkpoint.tensors;
        let t = tensors
            .get(&format!("{name}.trellis"))
            .context("missing projection trellis")?;
        let &[rt, ct, words] = t.shape.as_slice() else {
            bail!("invalid trellis rank")
        };
        ensure!(
            matches!(t.dtype.as_str(), "I16" | "U16") && words.is_multiple_of(16),
            "invalid packed trellis"
        );
        let input = u32::try_from(rt.checked_mul(16).context("input width overflow")?)?;
        let output = u32::try_from(ct.checked_mul(16).context("output width overflow")?)?;
        let k = u32::try_from(words / 16)?;
        let count = contracts::qmv_words(input as usize, output as usize, k as usize)
            .and_then(|v| u32::try_from(v).ok())
            .context("invalid packed geometry or index overflow")?;
        let splits = if k == 7 {
            1
        } else {
            flash_support::reference_splits(input / 16, output / 16)
        };
        contracts::projection_extents(1, input, output, splits)
            .context("unsupported projection geometry/split count")?;
        ensure!(
            metal.flash_qmv_pipeline.thread_execution_width() == 32
                && metal.flash_qmv_pipeline.max_total_threads_per_threadgroup() >= 128
                && metal.k7_pipeline.thread_execution_width() == 32
                && metal.k7_pipeline.max_total_threads_per_threadgroup() >= 128
                && metal
                    .prepare128_pipeline
                    .max_total_threads_per_threadgroup()
                    >= 64
                && metal.finish128_pipeline.max_total_threads_per_threadgroup() >= 64,
            "unsupported Flash projection threadgroup"
        );
        let bytes = u64::from(count) * 2;
        let packed =
            super::read_weight(&metal.device, t, bytes).context("reading projection trellis")?;
        let get_scale = |primary, legacy, width| {
            let t = tensors
                .get(&format!("{name}.{primary}"))
                .or_else(|| tensors.get(&format!("{name}.{legacy}")))
                .context("missing projection scale")?;
            scale(&metal.device, t, width)
        };
        let cb = if tensors.contains_key(&format!("{name}.mul1")) {
            Codebook::Mul1
        } else if tensors.contains_key(&format!("{name}.mcg")) {
            Codebook::Mcg
        } else {
            Codebook::Default
        };
        Ok(Self {
            metal,
            packed,
            suh: get_scale("suh", "su", input)?,
            svh: get_scale("svh", "sv", output)?,
            bias: tensors
                .get(&format!("{name}.bias"))
                .map(|t| scale(&metal.device, t, output))
                .transpose()?,
            input,
            output,
            k,
            cb,
            splits,
        })
    }

    /// Each encoder uses tracked resources. The standard command buffer retains them
    /// (including both scratch buffers) through completion even if this layer is dropped.
    pub fn submit(&self, x: &[u16]) -> Result<ProjectionOutput> {
        let extents = self.extents(x.len())?;
        submit(self.metal, x, |cmd, x| {
            self.encode(
                cmd,
                Activation::Contiguous(x),
                extents,
                Storage::StorageModeShared,
            )
        })
    }

    fn extents(&self, elements: usize) -> Result<[u32; 3]> {
        ensure!(
            elements.is_multiple_of(self.input as usize),
            "input is not a complete row"
        );
        let batch = u32::try_from(elements / self.input as usize)?;
        contracts::projection_extents(batch, self.input, self.output, self.splits)
            .context("invalid Flash projection batch or index overflow")
    }

    fn view_extents(&self, x: &MatrixView) -> Result<[u32; 3]> {
        let [rows, cols] = x.layout.shape;
        ensure!(cols == self.input, "invalid projection view width");
        contracts::projection_extents(rows, self.input, self.output, self.splits)
            .context("invalid Flash projection view batch or index overflow")
    }

    /// Cast the source to FP16 in the preparation kernel.
    pub fn submit_fp16_view(&self, x: &MatrixView) -> Result<ProjectionOutput> {
        let extents = self.view_extents(x)?;
        submit_encoded(self.metal, |cmd| {
            self.encode(
                cmd,
                Activation::View(x),
                extents,
                Storage::StorageModeShared,
            )
        })
    }

    fn encode(
        &self,
        cmd: &CommandBufferRef,
        x: Activation<'_>,
        [inputs, partials, outputs]: [u32; 3],
        storage: Storage,
    ) -> Result<Buffer> {
        let gpu = self.metal;
        let batch = inputs / self.input;
        let prepared = buffer(
            &gpu.device,
            u64::from(inputs) * 2,
            Storage::StorageModePrivate,
        )?;
        match x {
            Activation::Contiguous(x) => Metal::encode(
                cmd,
                &gpu.prepare128_pipeline,
                &[x, &self.suh, &prepared],
                [self.input, 0, 0, 0, 0],
                MTLSize::new(u64::from(inputs / 128), 1, 1),
                64,
            ),
            Activation::View(x) => {
                let scales = MatrixView::column_scales(&self.suh, batch, self.input)?;
                gpu.encode_prepare_view(cmd, x, &scales, &prepared)?;
            }
        }
        let partial = buffer(
            &gpu.device,
            u64::from(partials) * 4,
            Storage::StorageModePrivate,
        )?;
        let dest = buffer(&gpu.device, u64::from(outputs) * 2, storage)?;
        let (pipeline, third, groups) = if self.k == 7 {
            (&gpu.k7_pipeline, &gpu.inverse, self.output)
        } else {
            (&gpu.flash_qmv_pipeline, &self.svh, self.output / 16)
        };
        Metal::encode(
            cmd,
            pipeline,
            &[&prepared, &self.packed, third, &partial],
            [batch, self.k, self.cb as u32, self.input, self.output],
            MTLSize::new(u64::from(groups), u64::from(batch), u64::from(self.splits)),
            128,
        );
        Metal::encode(
            cmd,
            &gpu.finish128_pipeline,
            &[
                &partial,
                &self.svh,
                self.bias.as_ref().unwrap_or(&self.svh),
                &dest,
            ],
            [
                self.splits,
                0,
                u32::from(self.bias.is_some()),
                0,
                self.output,
            ],
            MTLSize::new(u64::from(outputs / 128), 1, 1),
            64,
        );
        Ok(dest)
    }
}

fn submit(
    gpu: &Metal,
    x: &[u16],
    encode: impl FnOnce(&CommandBufferRef, &Buffer) -> Result<Buffer>,
) -> Result<ProjectionOutput> {
    submit_encoded(gpu, |cmd| {
        let input = upload(&gpu.device, x)?;
        encode(cmd, &input)
    })
}

pub(super) fn submit_encoded(
    gpu: &Metal,
    encode: impl FnOnce(&CommandBufferRef) -> Result<Buffer>,
) -> Result<ProjectionOutput> {
    objc::rc::autoreleasepool(|| {
        let cmd = gpu.queue.new_command_buffer();
        let dest = encode(cmd)?;
        cmd.commit();
        Ok(ProjectionOutput {
            command: cmd.to_owned(),
            count: dest.length() as usize / 2,
            buffer: dest,
        })
    })
}

impl Metal {
    fn encode_swiglu(
        &self,
        cmd: &CommandBufferRef,
        gate: &Buffer,
        up: &Buffer,
        storage: Storage,
    ) -> Result<Buffer> {
        ensure!(
            gate.length() == up.length() && gate.length() > 0 && gate.length().is_multiple_of(2),
            "invalid SwiGLU input extents"
        );
        let count = u32::try_from(gate.length() / 2)?;
        let output = buffer(&self.device, gate.length(), storage)?;
        let threads = self.swiglu_pipeline.thread_execution_width();
        Self::encode(
            cmd,
            &self.swiglu_pipeline,
            &[gate, up, &output],
            [count, 0, 0, 0, 0],
            MTLSize::new(u64::from(count).div_ceil(threads), 1, 1),
            threads as usize,
        );
        Ok(output)
    }

    pub fn swiglu(&self, gate: &[u16], up: &[u16]) -> Result<ProjectionOutput> {
        ensure!(
            !gate.is_empty() && gate.len() == up.len(),
            "invalid SwiGLU input extents"
        );
        u32::try_from(gate.len()).context("SwiGLU exceeds Metal index range")?;
        submit(self, gate, |cmd, gate| {
            let up = upload(&self.device, up)?;
            self.encode_swiglu(cmd, gate, &up, Storage::StorageModeShared)
        })
    }
}

pub struct FlashExpert<'a> {
    gate: FlashProjection<'a>,
    up: FlashProjection<'a>,
    down: FlashProjection<'a>,
}

impl<'a> FlashExpert<'a> {
    pub fn load(gpu: &'a Metal, checkpoint: &Checkpoint, prefix: &str) -> Result<Self> {
        let layer = |name| FlashProjection::load(gpu, checkpoint, &format!("{prefix}.{name}_proj"));
        let gate = layer("gate")?;
        let up = layer("up")?;
        let down = layer("down")?;
        ensure!(
            contracts::compatible_flash_expert(
                [gate.input, gate.output],
                [up.input, up.output],
                [down.input, down.output]
            ),
            "incompatible Flash expert projection geometry"
        );
        Ok(Self { gate, up, down })
    }

    pub fn submit(&self, x: &[u16]) -> Result<ProjectionOutput> {
        let extents = self.expert_extents(self.gate.extents(x.len())?)?;
        submit(self.gate.metal, x, |cmd, x| {
            self.encode(cmd, Activation::Contiguous(x), extents)
        })
    }

    pub fn submit_fp16_view(&self, x: &MatrixView) -> Result<ProjectionOutput> {
        let extents = self.expert_extents(self.gate.view_extents(x)?)?;
        submit_encoded(self.gate.metal, |cmd| {
            self.encode(cmd, Activation::View(x), extents)
        })
    }

    fn expert_extents(&self, gate: [u32; 3]) -> Result<[[u32; 3]; 3]> {
        Ok([
            gate,
            self.up.extents(gate[0] as usize)?,
            self.down.extents(gate[2] as usize)?,
        ])
    }

    fn encode(
        &self,
        cmd: &CommandBufferRef,
        x: Activation<'_>,
        [gate_extents, up_extents, down_extents]: [[u32; 3]; 3],
    ) -> Result<Buffer> {
        let gate = self
            .gate
            .encode(cmd, x, gate_extents, Storage::StorageModePrivate)?;
        let up = self
            .up
            .encode(cmd, x, up_extents, Storage::StorageModePrivate)?;
        let activated =
            self.gate
                .metal
                .encode_swiglu(cmd, &gate, &up, Storage::StorageModePrivate)?;
        self.down.encode(
            cmd,
            Activation::Contiguous(&activated),
            down_extents,
            Storage::StorageModeShared,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "physical Apple GPU; failure after encoding and before commit"]
    fn refused_encoding_never_publishes_partial_work() -> Result<()> {
        let metal = Metal::new()?;
        let one = f16::ONE.to_bits();
        let twenty = f16::from_f32(20.0).to_bits();
        let first = metal.swiglu(&[twenty], &[one])?;
        let mut observed = None;
        let failure = submit(&metal, &[one], |cmd, x| {
            let _output = metal.encode_swiglu(cmd, x, x, Storage::StorageModePrivate)?;
            observed = Some(cmd.to_owned());
            bail!("injected after a valid encoder")
        });
        assert!(
            failure
                .err()
                .is_some_and(|e| e.to_string().contains("injected"))
        );
        assert_eq!(
            observed.unwrap().status(),
            MTLCommandBufferStatus::NotEnqueued
        );
        ensure!(metal.swiglu(&[0], &[one])?.read()? == [0] && first.read()? == [twenty]);
        Ok(())
    }
}
