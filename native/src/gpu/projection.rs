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
    offsets: flash_support::RecordOffsets,
    input: u32,
    output: u32,
    k: u32,
    cb: Codebook,
    splits: u32,
    mtp_hidden: bool,
    generic: bool,
    simdgroups: u32,
    nt: u32,
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
        Self::load_impl(metal, checkpoint, name, false)
    }

    pub fn load_exl3(metal: &'a Metal, checkpoint: &Checkpoint, name: &str) -> Result<Self> {
        Self::load_impl(
            metal,
            checkpoint,
            name,
            checkpoint.model_type != "qwen4_exp",
        )
    }

    fn load_impl(
        metal: &'a Metal,
        checkpoint: &Checkpoint,
        name: &str,
        generic: bool,
    ) -> Result<Self> {
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
        let mtp_hidden = !generic && name == "mtp.fc_hidden";
        if mtp_hidden {
            ensure!(
                input == 2560 && output == 2560 && k != 7,
                "unsupported MTP hidden projection"
            );
        }
        let splits = if mtp_hidden {
            2
        } else if k == 7 {
            1
        } else if generic {
            contracts::exl3_projection_splits(input, output).context("invalid EXL3 dimensions")?
        } else {
            flash_support::reference_splits(input / 16, output / 16)
        };
        (if generic {
            contracts::exl3_projection_extents
        } else {
            contracts::projection_extents
        })(1, input, output, splits)
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
        let simdgroups = if generic
            && (metal.device_name() == "Apple M5" || metal.device_name().starts_with("Apple M5 "))
            && std::env::var_os("MLXL3_DISABLE_TENSOR_OPS").is_none()
        {
            8
        } else {
            4
        };
        let nt: u32 = if generic {
            if output / 16 >= 1024 { 2 } else { 4 }
        } else {
            1
        };
        if generic {
            let pipeline =
                &metal.dense_qmv[usize::from(simdgroups == 8)][nt.trailing_zeros() as usize];
            ensure!(
                pipeline.thread_execution_width() == 32
                    && pipeline.max_total_threads_per_threadgroup() >= u64::from(simdgroups * 32)
                    && metal.dense_finish.max_total_threads_per_threadgroup() >= 64,
                "unsupported EXL3 projection threadgroup"
            );
        }
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
            offsets: flash_support::RecordOffsets::default(),
            bias: tensors
                .get(&format!("{name}.bias"))
                .map(|t| scale(&metal.device, t, output))
                .transpose()?,
            input,
            output,
            k,
            cb,
            splits,
            mtp_hidden,
            generic,
            simdgroups,
            nt,
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
        (if self.generic {
            contracts::exl3_projection_extents
        } else {
            contracts::projection_extents
        })(batch, self.input, self.output, self.splits)
        .context("invalid Flash projection batch or index overflow")
    }

    fn view_extents(&self, x: &MatrixView) -> Result<[u32; 3]> {
        let [rows, cols] = x.layout.shape;
        ensure!(cols == self.input, "invalid projection view width");
        (if self.generic {
            contracts::exl3_projection_extents
        } else {
            contracts::projection_extents
        })(rows, self.input, self.output, self.splits)
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

    pub(super) fn dimensions(&self) -> [u32; 2] {
        [self.input, self.output]
    }
    pub(super) fn encode_row(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        storage: Storage,
    ) -> Result<MatrixView> {
        ensure!(x.shape()[0] == 1, "expected one projection row");
        self.encode_rows(cmd, x, storage)
    }

    pub(super) fn encode_rows(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        storage: Storage,
    ) -> Result<MatrixView> {
        let b = self.encode(cmd, Activation::View(x), self.view_extents(x)?, storage)?;
        Ok(MatrixView {
            buffer: b,
            layout: crate::tensor_layout::MatrixLayout {
                shape: [x.shape()[0], self.output],
                strides: [i64::from(self.output), 1],
                offset: 0,
            },
            dtype: super::view::FloatType::F16,
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
        ensure!(
            !self.mtp_hidden || batch == 4,
            "MTP hidden projection requires four streams"
        );
        let prepared = buffer(
            &gpu.device,
            u64::from(inputs) * 2,
            Storage::StorageModePrivate,
        )?;
        match x {
            Activation::Contiguous(x) => Metal::encode_bound(
                cmd,
                &gpu.prepare128_pipeline,
                [(x, 0), (&self.suh, self.offsets.suh), (&prepared, 0)].into_iter(),
                [self.input, 0, 0, 0, 0],
                MTLSize::new(u64::from(inputs / 128), 1, 1),
                64,
            ),
            Activation::View(x) => {
                let scales =
                    MatrixView::column_scales(&self.suh, self.offsets.suh, batch, self.input)?;
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
            (&gpu.k7_pipeline, (&gpu.inverse, 0), self.output)
        } else {
            (
                if self.mtp_hidden {
                    &gpu.mtp_hidden_qmv
                } else if self.generic {
                    &gpu.dense_qmv[usize::from(self.simdgroups == 8)]
                        [self.nt.trailing_zeros() as usize]
                } else {
                    &gpu.flash_qmv_pipeline
                },
                (&self.svh, self.offsets.svh),
                self.output / 16 / self.nt,
            )
        };
        Metal::encode_bound(
            cmd,
            pipeline,
            [
                (&prepared, 0),
                (&self.packed, self.offsets.trellis),
                third,
                (&partial, 0),
            ]
            .into_iter(),
            [batch, self.k, self.cb as u32, self.input, self.output],
            MTLSize::new(
                u64::from(groups),
                if self.mtp_hidden { 1 } else { u64::from(batch) },
                u64::from(self.splits),
            ),
            contracts::projection_threads(self.k, self.mtp_hidden, self.simdgroups)
                .context("invalid projection kernel thread count")? as usize,
        );
        Metal::encode_bound(
            cmd,
            if self.generic {
                &gpu.dense_finish
            } else {
                &gpu.finish128_pipeline
            },
            [
                (&partial, 0),
                (&self.svh, self.offsets.svh),
                self.bias
                    .as_ref()
                    .map_or((&self.svh, self.offsets.svh), |b| (b, 0)),
                (&dest, 0),
            ]
            .into_iter(),
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
        Self::from_projections(gate, up, down)
    }

    pub fn from_projections(
        gate: FlashProjection<'a>,
        up: FlashProjection<'a>,
        down: FlashProjection<'a>,
    ) -> Result<Self> {
        ensure!(
            std::ptr::eq(gate.metal, up.metal) && std::ptr::eq(gate.metal, down.metal),
            "expert projections require one Metal context"
        );
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

    pub(super) fn input_hidden(&self) -> [u32; 2] {
        [self.gate.input, self.gate.output]
    }

    pub(super) fn from_record(gpu: &'a Metal, record: &TensorInfo) -> Result<Self> {
        let (file, offset) = record.source()?;
        let backing = super::read_buffer(&gpu.device, file, offset, flash_support::EXPERT_BYTES)?;
        let projection = |index, input, output| -> Result<_> {
            let offsets =
                flash_support::record_offsets(index).context("invalid Flash record projection")?;
            let splits = flash_support::reference_splits(input / 16, output / 16);
            contracts::projection_extents(1, input, output, splits)
                .context("invalid Flash record shape")?;
            Ok(FlashProjection {
                metal: gpu,
                packed: backing.clone(),
                suh: backing.clone(),
                svh: backing.clone(),
                bias: None,
                offsets,
                input,
                output,
                k: 2,
                cb: Codebook::Mul1,
                splits,
                mtp_hidden: false,
                generic: false,
                simdgroups: 4,
                nt: 1,
            })
        };
        Ok(Self {
            down: projection(0, 640, 2560)?,
            gate: projection(1, 2560, 640)?,
            up: projection(2, 2560, 640)?,
        })
    }
    pub(super) fn encode_view(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        storage: Storage,
    ) -> Result<Buffer> {
        let extents = self.expert_extents(self.gate.view_extents(x)?)?;
        self.encode(cmd, Activation::View(x), extents, storage)
    }

    pub fn submit(&self, x: &[u16]) -> Result<ProjectionOutput> {
        let extents = self.expert_extents(self.gate.extents(x.len())?)?;
        submit(self.gate.metal, x, |cmd, x| {
            self.encode(
                cmd,
                Activation::Contiguous(x),
                extents,
                Storage::StorageModeShared,
            )
        })
    }

    pub fn submit_fp16_view(&self, x: &MatrixView) -> Result<ProjectionOutput> {
        let extents = self.expert_extents(self.gate.view_extents(x)?)?;
        submit_encoded(self.gate.metal, |cmd| {
            self.encode(
                cmd,
                Activation::View(x),
                extents,
                Storage::StorageModeShared,
            )
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
        storage: Storage,
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
            storage,
        )
    }
}

/// Routed EXL3 experts retain their distinct fused float SwiGLU contract.
/// Ordinary dense/shared MLPs use `FlashExpert` instead.
pub struct MappedSwitchGlu<'a> {
    gpu: &'a Metal,
    experts: Vec<FlashExpert<'a>>,
    top_k: u32,
    hidden: u32,
    intermediate: u32,
    gu_plan: [u32; 3],
    down_plan: [u32; 3],
}

impl<'a> MappedSwitchGlu<'a> {
    pub fn load(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        num_experts: u32,
        top_k: u32,
        names: [&str; 3],
    ) -> Result<Self> {
        ensure!(
            num_experts > 0 && num_experts <= 1024 && top_k > 0 && top_k <= num_experts,
            "invalid mapped expert bank geometry"
        );
        let mut experts: Vec<FlashExpert<'a>> = Vec::with_capacity(num_experts as usize);
        for id in 0..num_experts {
            let projection = |name| -> Result<_> {
                let name = format!("{prefix}.{id}.{name}");
                for (primary, legacy) in [("suh", "su"), ("svh", "sv")] {
                    let tensor = c
                        .tensors
                        .get(&format!("{name}.{primary}"))
                        .or_else(|| c.tensors.get(&format!("{name}.{legacy}")))
                        .context("missing mapped expert scale")?;
                    ensure!(tensor.dtype == "F16", "mapped expert scales require F16");
                }
                ensure!(
                    !c.tensors.contains_key(&format!("{name}.bias")),
                    "mapped expert must be unbiased"
                );
                FlashProjection::load_exl3(gpu, c, &name)
            };
            let expert = FlashExpert::from_projections(
                projection(names[0])?,
                projection(names[1])?,
                projection(names[2])?,
            )?;
            ensure!(
                expert.gate.k != 7
                    && expert.gate.k == expert.up.k
                    && expert.gate.k == expert.down.k
                    && expert.gate.cb == expert.up.cb
                    && expert.gate.cb == expert.down.cb,
                "mapped expert requires one supported bit width and codebook"
            );
            if let Some(first) = experts.first() {
                ensure!(
                    expert.input_hidden() == first.input_hidden()
                        && expert.gate.k == first.gate.k
                        && expert.gate.cb == first.gate.cb,
                    "incompatible mapped expert bank"
                );
            }
            experts.push(expert);
        }
        let first = &experts[0];
        let [hidden, intermediate] = first.input_hidden();
        hidden
            .checked_mul(top_k)
            .context("mapped expert output bank overflow")?;
        let m5 = (gpu.device_name() == "Apple M5" || gpu.device_name().starts_with("Apple M5 "))
            && std::env::var_os("MLXL3_DISABLE_TENSOR_OPS").is_none();
        let gu_plan =
            contracts::mapped_expert_plan(hidden, intermediate, top_k, 2, first.gate.k, m5)
                .context("invalid mapped gate/up plan")?;
        let down_plan =
            contracts::mapped_expert_plan(intermediate, hidden, top_k, 1, first.gate.k, m5)
                .context("invalid mapped down plan")?;
        Ok(Self {
            gpu,
            experts,
            top_k,
            hidden,
            intermediate,
            gu_plan,
            down_plan,
        })
    }

    pub fn hidden_size(&self) -> u32 {
        self.hidden
    }

    pub fn forward(&self, x: &MatrixView, ids: &[u32], scores: &MatrixView) -> Result<MatrixView> {
        use super::view::FloatType;
        ensure!(
            x.shape() == [1, self.hidden]
                && x.dtype == FloatType::F16
                && ids.len() == self.top_k as usize
                && ids.iter().all(|&i| (i as usize) < self.experts.len())
                && scores.shape() == [1, self.top_k]
                && scores.dtype == FloatType::F16
                && scores.layout.offset == 0
                && scores.layout.strides[1] == 1,
            "invalid mapped expert input, routes or scores"
        );
        ensure!(
            x.buffer.device().registry_id() == self.gpu.device.registry_id()
                && scores.buffer.device().registry_id() == self.gpu.device.registry_id(),
            "foreign mapped expert storage"
        );
        let gpu = self.gpu;
        gpu.command(|cmd| {
            let stack = MatrixView::output(
                gpu,
                FloatType::F16,
                self.top_k,
                self.hidden,
                Storage::StorageModePrivate,
            )?;
            for (route, &id) in ids.iter().enumerate() {
                let expert = &self.experts[id as usize];
                let prepare = |p: &FlashProjection<'_>| -> Result<Buffer> {
                    let prepared = buffer(
                        &gpu.device,
                        u64::from(p.input) * 2,
                        Storage::StorageModePrivate,
                    )?;
                    let scale = MatrixView::column_scales(&p.suh, p.offsets.suh, 1, p.input)?;
                    gpu.encode_prepare_view(cmd, x, &scale, &prepared)?;
                    Ok(prepared)
                };
                let gate =
                    expert
                        .gate
                        .encode_mapped_raw(cmd, &prepare(&expert.gate)?, self.gu_plan)?;
                let up = expert
                    .up
                    .encode_mapped_raw(cmd, &prepare(&expert.up)?, self.gu_plan)?;
                let activated = buffer(
                    &gpu.device,
                    u64::from(self.intermediate) * 2,
                    Storage::StorageModePrivate,
                )?;
                ensure!(
                    gpu.mapped_glu.thread_execution_width() == 32
                        && gpu.mapped_glu.max_total_threads_per_threadgroup() >= 128,
                    "unsupported mapped GLU threadgroup"
                );
                Metal::encode_bound(
                    cmd,
                    &gpu.mapped_glu,
                    [
                        (&gate, 0),
                        (&up, 0),
                        (&expert.gate.svh, expert.gate.offsets.svh),
                        (&expert.up.svh, expert.up.offsets.svh),
                        (&expert.down.suh, expert.down.offsets.suh),
                        (&activated, 0),
                    ]
                    .into_iter(),
                    [self.intermediate, self.gu_plan[0]],
                    MTLSize::new(u64::from(self.intermediate / 128), 1, 1),
                    128,
                );
                let raw = expert
                    .down
                    .encode_mapped_raw(cmd, &activated, self.down_plan)?;
                let offset = u64::from(
                    contracts::mapped_expert_row_offset(
                        u32::try_from(route)?,
                        self.hidden,
                        u32::try_from(stack.buffer.length() / 2)?,
                    )
                    .context("invalid mapped output row")?,
                ) * 2;
                Metal::encode_bound(
                    cmd,
                    &gpu.dense_finish,
                    [
                        (&raw, 0),
                        (&expert.down.svh, expert.down.offsets.svh),
                        (&expert.down.svh, expert.down.offsets.svh),
                        (&stack.buffer, offset),
                    ]
                    .into_iter(),
                    [self.down_plan[0], 0, 0, 0, self.hidden],
                    MTLSize::new(u64::from(self.hidden / 128), 1, 1),
                    64,
                );
            }
            let output = MatrixView::output(
                gpu,
                FloatType::F16,
                1,
                self.hidden,
                Storage::StorageModeShared,
            )?;
            ensure!(
                gpu.mapped_mix.thread_execution_width() == 32
                    && gpu.mapped_mix.max_total_threads_per_threadgroup() >= 128,
                "unsupported mapped expert reduction threadgroup"
            );
            Metal::encode(
                cmd,
                &gpu.mapped_mix,
                &[&stack.buffer, &scores.buffer, &output.buffer],
                [self.hidden, self.top_k],
                MTLSize::new(u64::from(self.hidden.div_ceil(128)), 1, 1),
                128,
            );
            Ok(output)
        })
    }
}

impl FlashProjection<'_> {
    fn encode_mapped_raw(
        &self,
        cmd: &CommandBufferRef,
        prepared: &Buffer,
        [splits, nt, groups]: [u32; 3],
    ) -> Result<Buffer> {
        let gpu = self.metal;
        let partials = self
            .output
            .checked_mul(splits)
            .context("mapped raw output overflow")?;
        let raw = buffer(
            &gpu.device,
            u64::from(partials) * 4,
            Storage::StorageModePrivate,
        )?;
        let p = &gpu.dense_qmv[usize::from(groups == 8)][nt.trailing_zeros() as usize];
        ensure!(
            p.thread_execution_width() == 32
                && p.max_total_threads_per_threadgroup() >= u64::from(groups * 32),
            "unsupported mapped expert QMV threadgroup"
        );
        Metal::encode_bound(
            cmd,
            p,
            [
                (prepared, 0),
                (&self.packed, self.offsets.trellis),
                (&self.svh, self.offsets.svh),
                (&raw, 0),
            ]
            .into_iter(),
            [1, self.k, self.cb as u32, self.input, self.output],
            MTLSize::new(u64::from(self.output / 16 / nt), 1, u64::from(splits)),
            (groups * 32) as usize,
        );
        Ok(raw)
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

#[cfg(test)]
#[test]
#[ignore = "physical Qwen mapped experts: rejected routes and retained output after weight release"]
fn mapped_expert_rejections_recover_and_outputs_outlive_the_bank() -> Result<()> {
    use super::view::FloatType;
    use crate::tensor_layout::MatrixLayout;
    let mut c = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_NATIVE_MODEL",
    )?))?;
    let gpu = Metal::new()?;
    let prefix = "model.language_model.layers.0.mlp.experts";
    let names = ["gate_proj", "up_proj", "down_proj"];
    let name = format!("{prefix}.0.gate_proj.svh");
    let old = c
        .tensors
        .get_mut(&name)
        .context("missing mapped scale fixture")?
        .dtype
        .clone();
    c.tensors.get_mut(&name).unwrap().dtype = "F32".into();
    ensure!(
        MappedSwitchGlu::load(&gpu, &c, prefix, 256, 8, names)
            .err()
            .is_some_and(|e| e.to_string().contains("F16"))
    );
    c.tensors.get_mut(&name).unwrap().dtype = old;
    let bank = MappedSwitchGlu::load(&gpu, &c, prefix, 256, 8, names)?;
    let width = bank.hidden_size();
    let data = (0..width)
        .map(|i| f16::from_f32(((i * 29) % 193) as f32 / 779. - 96. / 779.).to_bits())
        .collect::<Vec<_>>();
    let x = gpu.f16_storage(&data)?.matrix(MatrixLayout {
        shape: [1, width],
        strides: [i64::from(width), 1],
        offset: 0,
    })?;
    let scores = gpu
        .f16_storage(&[f16::from_f32(0.125).to_bits(); 8])?
        .matrix(MatrixLayout {
            shape: [1, 8],
            strides: [8, 1],
            offset: 0,
        })?;
    let ids = [0, 17, 31, 64, 127, 191, 254, 255];
    let expected = bank.forward(&x, &ids, &scores)?.read_output()?;
    ensure!(
        expected.len() == width as usize
            && expected.iter().any(|&v| v != 0)
            && expected
                .iter()
                .all(|&v| f16::from_bits(v as u16).is_finite())
    );
    ensure!(bank.forward(&x, &[256; 8], &scores).is_err());
    ensure!(bank.forward(&x, &ids[..7], &scores).is_err());
    let short = gpu.f16_storage(&[0; 7])?.matrix(MatrixLayout {
        shape: [1, 7],
        strides: [7, 1],
        offset: 0,
    })?;
    ensure!(bank.forward(&x, &ids, &short).is_err());
    let wrong = gpu
        .f32_storage(&vec![0.; width as usize])?
        .matrix(MatrixLayout {
            shape: [1, width],
            strides: [i64::from(width), 1],
            offset: 0,
        })?;
    ensure!(wrong.dtype == FloatType::F32 && bank.forward(&wrong, &ids, &scores).is_err());
    let result = bank.forward(&x, &ids, &scores)?;
    drop(bank);
    drop(c);
    drop(scores);
    drop(x);
    drop(gpu);
    ensure!(result.read_output()? == expected);
    Ok(())
}
