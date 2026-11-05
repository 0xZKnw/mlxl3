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
    foreign_types::{ForeignType, ForeignTypeRef},
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
    routed_record: bool,
    simdgroups: u32,
    nt: u32,
    static_key: Option<usize>,
    fixed_key: Option<u8>,
    scratch: std::cell::RefCell<Option<ResidentScratch<'a>>>,
    #[cfg(test)]
    nt2_submissions: std::cell::Cell<u64>,
}

struct ResidentScratch<'a> {
    prepared: Buffer,
    partial: Buffer,
    owner: CommandBuffer,
    bytes: u64,
    budget: &'a crate::resident_scratch_policy::Budget,
}
impl Drop for ResidentScratch<'_> {
    fn drop(&mut self) {
        // Abandoned/error commands retain their conservative charge until Metal teardown.
        if self.owner.status() == MTLCommandBufferStatus::Completed {
            self.budget.refund(self.bytes);
        }
    }
}

#[derive(Clone, Copy)]
enum Activation<'a> {
    Contiguous(&'a Buffer),
    View(&'a MatrixView),
}

#[derive(Default)]
struct ExpertStage<'a> {
    prepared: Option<(&'a Buffer, u64)>,
    gate: Option<&'a Buffer>,
    destination: Option<(&'a Buffer, u64)>,
}

pub struct ProjectionOutput {
    command: CommandBuffer,
    buffer: Buffer,
    count: usize,
}

impl ProjectionOutput {
    pub fn read(self) -> Result<Vec<u16>> {
        #[cfg(test)]
        let wait = crate::flash_profile::span("native.projection_wait");
        self.command.wait_until_completed();
        ensure!(
            self.command.status() == MTLCommandBufferStatus::Completed,
            "Flash projection command failed ({:?})",
            self.command.status()
        );
        #[cfg(test)]
        {
            drop(wait);
            crate::flash_profile::native_gpu_interval(&self.command, "projection");
        }
        #[cfg(test)]
        let _copy = crate::flash_profile::span("native.projection_readback_copy");
        #[cfg(test)]
        if let Some(span) = &_copy {
            span.data(serde_json::json!({"bytes":self.count*2}));
        }
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
        let cb = if tensors.contains_key(&format!("{name}.mul1")) {
            Codebook::Mul1
        } else if tensors.contains_key(&format!("{name}.mcg")) {
            Codebook::Mcg
        } else {
            Codebook::Default
        };
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
            let eligible = !mtp_hidden
                && (metal.device_name() == "Apple M5"
                    || metal.device_name().starts_with("Apple M5 "));
            let candidate = contracts::flash_resident_nt(output, k, cb as u32, eligible, true);
            let enabled = if candidate == 2 {
                let flag = std::env::var("MLXL3_NATIVE_FLASH_NT2");
                let value = match &flag {
                    Ok(v) => Some(v.as_str()),
                    Err(std::env::VarError::NotPresent) => None,
                    Err(e) => bail!("MLXL3_NATIVE_FLASH_NT2 must be 0 or 1: {e}"),
                };
                contracts::flash_nt2_flag(value).context("MLXL3_NATIVE_FLASH_NT2 must be 0 or 1")?
            } else {
                false
            };
            contracts::flash_resident_nt(output, k, cb as u32, eligible, enabled)
        };
        if nt == 2 && !generic {
            ensure!(
                metal.flash_qmv_nt2.thread_execution_width() == 32
                    && metal.flash_qmv_nt2.max_total_threads_per_threadgroup() >= 128,
                "unsupported Flash NT2 projection threadgroup"
            );
        }
        let static_key = static_key(metal, k, cb, nt, !generic && !mtp_hidden)?;
        let fixed_key = fixed_key(
            metal,
            [input, output],
            splits,
            [k, cb as u32, nt],
            !generic && !mtp_hidden,
        )?;
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
            routed_record: false,
            simdgroups,
            nt,
            static_key,
            fixed_key,
            scratch: std::cell::RefCell::new(None),
            #[cfg(test)]
            nt2_submissions: std::cell::Cell::new(0),
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

    pub(super) fn encode_resident_pair(
        &self,
        other: &Self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        storage: Storage,
    ) -> Result<Option<[MatrixView; 2]>> {
        let gpu = self.metal;
        ensure!(
            std::ptr::eq(gpu, other.metal),
            "resident pair requires the same Metal instance"
        );
        if !gpu.gdn_resident_bundle.get() {
            return Ok(None);
        }
        let Some(pipeline) = gpu.gdn_resident_qmv.as_ref() else {
            return Ok(None);
        };
        let resident = |p: &Self| {
            !p.generic
                && !p.mtp_hidden
                && !p.routed_record
                && [p.offsets.trellis, p.offsets.suh, p.offsets.svh] == [0; 3]
        };
        if !crate::flash_support::ar20_resident_pair_eligible(
            true,
            [[self.input, self.output], [other.input, other.output]],
            [
                [self.k, self.cb as u32, self.nt],
                [other.k, other.cb as u32, other.nt],
            ],
            [self.splits, other.splits],
            [resident(self), resident(other)],
            [self.bias.is_some(), other.bias.is_some()],
            [
                pipeline.thread_execution_width() as u32,
                pipeline.max_total_threads_per_threadgroup() as u32,
            ],
        ) {
            return Ok(None);
        }
        ensure!(
            x.shape() == [1, 2560] && x.buffer.device().registry_id() == gpu.device.registry_id(),
            "invalid resident pair input"
        );
        let prepared = [
            buffer(&gpu.device, 5120, Storage::StorageModePrivate)?,
            buffer(&gpu.device, 5120, Storage::StorageModePrivate)?,
        ];
        let partial = [
            buffer(&gpu.device, 40960, Storage::StorageModePrivate)?,
            buffer(&gpu.device, 24576, Storage::StorageModePrivate)?,
        ];
        let output = [
            MatrixView::output(gpu, super::view::FloatType::F16, 1, 10240, storage)?,
            MatrixView::output(gpu, super::view::FloatType::F16, 1, 6144, storage)?,
        ];
        for (projection, prepared) in [(self, &prepared[0]), (other, &prepared[1])] {
            projection.encode_input_into(cmd, Activation::View(x), prepared, 0)?;
        }
        Metal::encode(
            cmd,
            pipeline,
            &[
                &prepared[0],
                &self.packed,
                &prepared[1],
                &other.packed,
                &partial[0],
                &partial[1],
            ],
            [0u32],
            MTLSize::new(512, 1, 1),
            128,
        );
        self.encode_finish(cmd, &partial[0], 0, &output[0].buffer, 0, 10240, None)?;
        other.encode_finish(cmd, &partial[1], 0, &output[1].buffer, 0, 6144, None)?;
        #[cfg(test)]
        gpu.gdn_resident_bundle_uses
            .set(gpu.gdn_resident_bundle_uses.get() + 1);
        Ok(Some(output))
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
        let extents = [inputs, partials, outputs];
        let budget = &self.metal.resident_scratch;
        if budget.enabled.get()
            && budget.valid()
            && crate::resident_scratch_policy::eligible(
                true,
                self.generic,
                self.mtp_hidden,
                self.routed_record,
                inputs == self.input,
                storage == Storage::StorageModePrivate,
                cmd.status() == MTLCommandBufferStatus::NotEnqueued,
            )
            && let Ok(mut slot) = self.scratch.try_borrow_mut()
        {
            let sizes = [u64::from(inputs) * 2, u64::from(partials) * 4];
            if slot.is_none() && budget.claim(sizes[0] + sizes[1]) {
                let pair = (|| -> Result<_> {
                    Ok(ResidentScratch {
                        prepared: buffer(
                            &self.metal.device,
                            sizes[0],
                            Storage::StorageModePrivate,
                        )?,
                        partial: buffer(&self.metal.device, sizes[1], Storage::StorageModePrivate)?,
                        owner: cmd.to_owned(),
                        bytes: sizes[0] + sizes[1],
                        budget,
                    })
                })();
                match pair {
                    Ok(pair) => {
                        let pair = slot.insert(pair);
                        return self.encode_with_scratch(
                            cmd,
                            x,
                            extents,
                            storage,
                            Some((&pair.prepared, &pair.partial)),
                        );
                    }
                    Err(error) => {
                        budget.refund(sizes[0] + sizes[1]);
                        return Err(error);
                    }
                }
            }
            if let Some(pair) = slot.as_mut()
                && crate::resident_scratch_policy::reusable(
                    pair.owner.status() == MTLCommandBufferStatus::Completed,
                    pair.owner.as_ptr() == cmd.as_ptr(),
                    pair.prepared.length() == sizes[0]
                        && pair.partial.length() == sizes[1]
                        && pair.prepared.device().registry_id() == self.metal.device.registry_id()
                        && pair.partial.device().registry_id() == self.metal.device.registry_id()
                        && pair.prepared.as_ptr() != pair.partial.as_ptr(),
                )
            {
                pair.owner = cmd.to_owned();
                #[cfg(test)]
                budget.reuses.set(budget.reuses.get() + 1);
                return self.encode_with_scratch(
                    cmd,
                    x,
                    extents,
                    storage,
                    Some((&pair.prepared, &pair.partial)),
                );
            }
        }
        self.encode_with_scratch(cmd, x, extents, storage, None)
    }

    fn encode_with_scratch(
        &self,
        cmd: &CommandBufferRef,
        x: Activation<'_>,
        [inputs, partials, outputs]: [u32; 3],
        storage: Storage,
        scratch: Option<(&Buffer, &Buffer)>,
    ) -> Result<Buffer> {
        self.encode_expert_stage(
            cmd,
            x,
            [inputs, partials, outputs],
            storage,
            scratch,
            ExpertStage::default(),
        )
    }

    fn encode_expert_stage(
        &self,
        cmd: &CommandBufferRef,
        x: Activation<'_>,
        [inputs, partials, outputs]: [u32; 3],
        storage: Storage,
        scratch: Option<(&Buffer, &Buffer)>,
        stage: ExpertStage<'_>,
    ) -> Result<Buffer> {
        let gpu = self.metal;
        if let Some((prepared, partial)) = scratch {
            ensure!(
                crate::flash_expert_scratch::fits(
                    [inputs, partials, outputs],
                    [prepared.length(), partial.length()]
                ) && prepared.device().registry_id() == gpu.device.registry_id()
                    && partial.device().registry_id() == gpu.device.registry_id()
                    && prepared.as_ptr() != partial.as_ptr(),
                "invalid expert scratch prefixes"
            );
        }
        let batch = inputs / self.input;
        ensure!(
            !self.mtp_hidden || batch == 4,
            "MTP hidden projection requires four streams"
        );
        let owned_prepared;
        let prepared = if let Some((prepared, _)) = stage.prepared {
            prepared
        } else if let Some((prepared, _)) = scratch {
            prepared
        } else {
            owned_prepared = buffer(
                &gpu.device,
                u64::from(inputs) * 2,
                Storage::StorageModePrivate,
            )?;
            &owned_prepared
        };
        let prepared_offset = stage.prepared.map_or(0, |(_, offset)| offset);
        ensure!(
            prepared_offset.is_multiple_of(2)
                && prepared_offset
                    .checked_add(u64::from(inputs) * 2)
                    .is_some_and(|end| end <= prepared.length()),
            "invalid prepared prefix"
        );
        if stage.prepared.is_none() {
            match x {
                Activation::Contiguous(x) => Metal::encode_bound(
                    cmd,
                    &gpu.prepare128_pipeline,
                    [(x, 0), (&self.suh, self.offsets.suh), (prepared, 0)].into_iter(),
                    [self.input, 0, 0, 0, 0],
                    MTLSize::new(u64::from(inputs / 128), 1, 1),
                    64,
                ),
                Activation::View(x) => {
                    let scales =
                        MatrixView::column_scales(&self.suh, self.offsets.suh, batch, self.input)?;
                    gpu.encode_prepare_view(cmd, x, &scales, prepared)?;
                }
            }
        }
        let owned_partial;
        let partial = if let Some((_, partial)) = scratch {
            partial
        } else {
            owned_partial = buffer(
                &gpu.device,
                u64::from(partials) * 4,
                Storage::StorageModePrivate,
            )?;
            &owned_partial
        };
        #[cfg(test)]
        if scratch.is_some() {
            let count = gpu.flash_scratch_uses.get();
            let mut bindings = gpu.flash_scratch_bindings.get();
            bindings[count % 3] = [prepared.as_ptr() as usize, partial.as_ptr() as usize];
            gpu.flash_scratch_bindings.set(bindings);
            gpu.flash_scratch_uses.set(count + 1);
        }
        let dest = if let Some((dest, offset)) = stage.destination {
            ensure!(
                offset.is_multiple_of(2)
                    && offset
                        .checked_add(u64::from(outputs) * 2)
                        .is_some_and(|end| end <= dest.length())
                    && dest.device().registry_id() == gpu.device.registry_id(),
                "invalid expert destination prefix"
            );
            dest.clone()
        } else {
            buffer(&gpu.device, u64::from(outputs) * 2, storage)?
        };
        let dest_offset = stage.destination.map_or(0, |(_, offset)| offset);
        let routed_nt2 = crate::flash_qmv::routed_nt2_key(
            [self.input, self.output],
            self.splits,
            [self.k, self.cb as u32, self.nt],
            self.routed_record && batch == 1,
            gpu.routed_nt2_gu.get(),
            gpu.routed_nt2_down.get(),
        );
        let (pipeline, third, groups) = if let Some(key) = routed_nt2 {
            (
                &gpu.routed_nt2_pipelines
                    .as_ref()
                    .context("routed NT2 pipelines were not loaded")?[key],
                (&self.svh, self.offsets.svh),
                contracts::flash_reference_groups(self.output, 2)
                    .context("invalid routed NT2 grid")?,
            )
        } else if self.k == 7 {
            (&gpu.k7_pipeline, (&gpu.inverse, 0), self.output)
        } else {
            (
                if self.mtp_hidden {
                    &gpu.mtp_hidden_qmv
                } else if self.generic {
                    &gpu.dense_qmv[usize::from(self.simdgroups == 8)]
                        [self.nt.trailing_zeros() as usize]
                } else if let Some(key) = self.fixed_key {
                    &gpu.flash_fixed_qmv
                        .as_ref()
                        .context("missing Flash fixed dimensions PSO")?[usize::from(key)]
                } else if let Some(key) = self.static_key {
                    &gpu.flash_static_qmv[key]
                } else if self.nt == 2 {
                    &gpu.flash_qmv_nt2
                } else {
                    &gpu.flash_qmv_pipeline
                },
                (&self.svh, self.offsets.svh),
                if self.generic {
                    self.output / 16 / self.nt
                } else {
                    contracts::flash_reference_groups(self.output, self.nt)
                        .context("invalid Flash reference tile grid")?
                },
            )
        };
        Metal::encode_bound(
            cmd,
            pipeline,
            [
                (prepared, prepared_offset),
                (&self.packed, self.offsets.trellis),
                third,
                (partial, 0),
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
        self.encode_finish(cmd, partial, 0, &dest, dest_offset, outputs, stage.gate)?;
        #[cfg(test)]
        if gpu.ar30_cooperative && (2..=4).contains(&self.k) && !self.mtp_hidden {
            let count = &gpu.ar30_cooperative_uses[usize::from(self.routed_record)];
            count.set(count.get() + 1);
        }
        #[cfg(test)]
        if let Some(key) = routed_nt2 {
            gpu.routed_nt2_uses[key].set(gpu.routed_nt2_uses[key].get() + 1);
        } else if let Some(key) = self.fixed_key {
            let count = &gpu.flash_fixed_submissions[usize::from(key)];
            count.set(count.get() + 1);
        } else if let Some(key) = self.static_key {
            let count = &gpu.flash_static_submissions[key];
            count.set(count.get() + 1);
        }
        #[cfg(test)]
        if !self.generic && self.nt == 2 {
            self.nt2_submissions.set(self.nt2_submissions.get() + 1);
        }
        Ok(dest)
    }
    #[allow(clippy::too_many_arguments)] // Explicit buffer offsets mirror the Metal bindings.
    fn encode_finish(
        &self,
        cmd: &CommandBufferRef,
        partial: &Buffer,
        partial_offset: u64,
        dest: &Buffer,
        dest_offset: u64,
        outputs: u32,
        gate: Option<&Buffer>,
    ) -> Result<()> {
        let gpu = self.metal;
        ensure!(
            partial_offset.is_multiple_of(4)
                && partial_offset
                    .checked_add(u64::from(outputs) * u64::from(self.splits) * 4)
                    .is_some_and(|end| end <= partial.length())
                && dest_offset.is_multiple_of(2)
                && dest_offset
                    .checked_add(u64::from(outputs) * 2)
                    .is_some_and(|end| end <= dest.length()),
            "invalid finish prefixes"
        );
        let finish_bindings = [
            (partial, partial_offset),
            (&self.svh, self.offsets.svh),
            self.bias
                .as_ref()
                .map_or((&self.svh, self.offsets.svh), |b| (b, 0)),
            (dest, dest_offset),
            gate.map_or((&self.svh, self.offsets.svh), |gate| (gate, 0)),
        ];
        if let Some(gate) = gate {
            ensure!(
                !self.generic && gate.length() == u64::from(outputs) * 2,
                "invalid up finish gate"
            );
        }
        Metal::encode_bound(
            cmd,
            if gate.is_some() {
                gpu.expert_up_finish
                    .as_ref()
                    .context("expert up finish pipeline was not loaded")?
            } else if self.generic {
                &gpu.dense_finish
            } else {
                &gpu.finish128_pipeline
            },
            finish_bindings[..if gate.is_some() { 5 } else { 4 }]
                .iter()
                .copied(),
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
        Ok(())
    }

    fn encode_input_into(
        &self,
        cmd: &CommandBufferRef,
        x: Activation<'_>,
        output: &Buffer,
        offset: u64,
    ) -> Result<()> {
        ensure!(
            offset
                .checked_add(u64::from(self.input) * 2)
                .is_some_and(|end| end <= output.length()),
            "invalid expert prepared prefix"
        );
        match x {
            Activation::Contiguous(x) => Metal::encode_bound(
                cmd,
                &self.metal.prepare128_pipeline,
                [(x, 0), (&self.suh, self.offsets.suh), (output, offset)].into_iter(),
                [self.input, 0, 0, 0, 0],
                MTLSize::new(u64::from(self.input / 128), 1, 1),
                64,
            ),
            Activation::View(x) => self.metal.encode_prepare_view_into(
                cmd,
                x,
                &MatrixView::column_scales(&self.suh, self.offsets.suh, 1, self.input)?,
                output,
                offset,
            )?,
        }
        Ok(())
    }
}

fn fixed_key(
    metal: &Metal,
    shape: [u32; 2],
    splits: u32,
    format: [u32; 3],
    eligible: bool,
) -> Result<Option<u8>> {
    crate::flash_qmv::fixed_select(shape, splits, format, eligible, metal.flash_fixed_setting)
        .map(|key| key.map(|key| key as u8))
        .map_err(|()| anyhow::anyhow!("MLXL3_NATIVE_FLASH_FIXED_DIMS must be 0 or 1"))
}

fn static_key(
    metal: &Metal,
    k: u32,
    cb: Codebook,
    nt: u32,
    eligible: bool,
) -> Result<Option<usize>> {
    let key = crate::flash_qmv::select(k, cb as u32, nt, eligible, metal.flash_static_setting)
        .map_err(|()| anyhow::anyhow!("MLXL3_NATIVE_FLASH_STATIC_K must be 0 or 1"))?;
    if let Some(key) = key {
        ensure!(
            metal.flash_static_supported[key],
            "unsupported Flash static K projection threadgroup"
        );
    }
    Ok(key)
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

#[test]
#[ignore = "physical Apple M5; launched by tests/physical_flash_nt2.py"]
fn physical_flash_resident_nt2() -> Result<()> {
    use crate::tensor_layout::MatrixLayout;
    use serde::Deserialize;
    use std::{fs, path::PathBuf};
    #[derive(Deserialize)]
    struct Case {
        path: PathBuf,
        prefix: String,
        x: Vec<u16>,
        fp32: bool,
        generic: bool,
        candidate_nt: u32,
        control_nt: u32,
    }
    let root = PathBuf::from(std::env::var_os("MLXL3_NT2_FIXTURES").context("fixtures required")?);
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(root.join("native-cases.json"))?)?;
    let refs: Vec<Vec<u16>> = serde_json::from_slice(&fs::read(root.join("stock.json"))?)?;
    ensure!(
        !cases.is_empty() && cases.len() == refs.len(),
        "incomplete fixture matrix"
    );
    let literal = std::env::var("MLXL3_NATIVE_FLASH_NT2").ok();
    let gpu = Metal::new()?;
    ensure!(
        gpu.device_name() == "Apple M5" || gpu.device_name().starts_with("Apple M5 "),
        "unqualified hardware"
    );
    let real = crate::checkpoint::inspect(&cases.last().context("missing real fixture")?.path)?;
    let hidden = FlashProjection::load(&gpu, &real, "mtp.fc_hidden")?;
    ensure!(
        hidden.nt == 1
            && hidden.mtp_hidden
            && hidden.splits == 2
            && hidden.static_key.is_none()
            && hidden.fixed_key.is_none()
    );
    ensure!(hidden.submit(&vec![0; 2560]).is_err() && hidden.nt2_submissions.get() == 0);
    let selected_case = cases
        .iter()
        .find(|c| !c.generic && c.candidate_nt == 2)
        .context("missing NT2 case")?;
    let selected_checkpoint = crate::checkpoint::inspect(&selected_case.path)?;
    let Some(enabled) = contracts::flash_nt2_flag(literal.as_deref()) else {
        let error = FlashProjection::load(&gpu, &selected_checkpoint, &selected_case.prefix)
            .err()
            .context("invalid flag accepted")?;
        ensure!(
            error
                .to_string()
                .contains("MLXL3_NATIVE_FLASH_NT2 must be 0 or 1")
        );
        let generic = cases
            .iter()
            .find(|c| c.generic)
            .context("missing generic control")?;
        let c = crate::checkpoint::inspect(&generic.path)?;
        ensure!(FlashProjection::load_exl3(&gpu, &c, &generic.prefix)?.nt == generic.control_nt);
        fs::write(
            root.join("invalid-flag.json"),
            b"{\"rejected\":true,\"fc_hidden_nt\":1,\"generic_unchanged\":true}",
        )?;
        return Ok(());
    };
    if gpu.flash_static_setting.is_none() {
        let mut fallbacks = [0; 2];
        let error = FlashProjection::load(&gpu, &selected_checkpoint, &selected_case.prefix)
            .err()
            .context("invalid static K flag accepted")?;
        ensure!(
            error
                .to_string()
                .contains("MLXL3_NATIVE_FLASH_STATIC_K must be 0 or 1")
        );
        for (i, case) in cases
            .iter()
            .enumerate()
            .filter(|(_, c)| c.generic || c.prefix == "p")
        {
            let checkpoint = crate::checkpoint::inspect(&case.path)?;
            let k7 = checkpoint.tensors[&format!("{}.trellis", case.prefix)].shape[2] == 112;
            if !case.generic && !k7 {
                continue;
            }
            let layer = if case.generic {
                FlashProjection::load_exl3(&gpu, &checkpoint, &case.prefix)?
            } else {
                FlashProjection::load(&gpu, &checkpoint, &case.prefix)?
            };
            ensure!(layer.static_key.is_none());
            ensure!(
                layer.submit(&case.x)?.read()? == refs[i],
                "static invalid flag broke fallback"
            );
            fallbacks[usize::from(case.generic)] += 1;
        }
        ensure!(
            fallbacks.iter().all(|&count| count > 0),
            "missing fallback control"
        );
        fs::write(
            root.join("static-invalid-flag.json"),
            b"{\"rejected\":true,\"fc_hidden\":true,\"fallbacks\":true}",
        )?;
        return Ok(());
    }
    if gpu.flash_fixed_setting.is_none() {
        let case = cases
            .iter()
            .find(|c| c.prefix.contains("index_qk_proj"))
            .context("real indexer fixture required")?;
        let checkpoint = crate::checkpoint::inspect(&case.path)?;
        let error = FlashProjection::load(&gpu, &checkpoint, &case.prefix)
            .err()
            .context("invalid fixed flag accepted")?;
        ensure!(
            error
                .to_string()
                .contains("MLXL3_NATIVE_FLASH_FIXED_DIMS must be 0 or 1")
        );
        let fallback = FlashProjection::load(&gpu, &selected_checkpoint, &selected_case.prefix)?;
        ensure!(fallback.fixed_key.is_none() && hidden.fixed_key.is_none());
        fs::write(
            root.join("fixed-invalid-flag.json"),
            b"{\"rejected\":true,\"fc_hidden\":true,\"fallback\":true}",
        )?;
        return Ok(());
    }
    for invalid in 0..5 {
        let mut c = crate::checkpoint::inspect(&selected_case.path)?;
        let trellis = c
            .tensors
            .get_mut(&format!("{}.trellis", selected_case.prefix))
            .context("missing test trellis")?;
        match invalid {
            0 => trellis.shape[0] += 1,
            1 => trellis.shape[1] += 1,
            2 => trellis.shape[2] += 1,
            3 => trellis.dtype = "U8".into(),
            _ => {
                c.tensors
                    .get_mut(&format!("{}.bias", selected_case.prefix))
                    .context("missing test bias")?
                    .shape[0] -= 1
            }
        }
        ensure!(
            FlashProjection::load(&gpu, &c, &selected_case.prefix).is_err(),
            "malformed case {invalid} accepted"
        );
    }
    let mut result = Vec::new();
    let mut unequal = 0;
    let mut selected = 0;
    for (i, (case, expected)) in cases.iter().zip(&refs).enumerate() {
        let checkpoint = crate::checkpoint::inspect(&case.path)?;
        let load = if case.generic {
            FlashProjection::load_exl3
        } else {
            FlashProjection::load
        };
        let layer = load(&gpu, &checkpoint, &case.prefix)?;
        let before_fixed = gpu
            .flash_fixed_submissions
            .each_ref()
            .map(std::cell::Cell::get);
        let before_static = gpu
            .flash_static_submissions
            .each_ref()
            .map(std::cell::Cell::get);
        let nt = if enabled {
            case.candidate_nt
        } else {
            case.control_nt
        };
        ensure!(
            layer.nt == nt,
            "case {i}: expected NT{nt}, selected NT{}",
            layer.nt
        );
        ensure!(layer.submit(&[]).is_err() && layer.submit(&case.x[1..]).is_err());
        ensure!(
            layer.nt2_submissions.get() == 0,
            "invalid input encoded a command"
        );
        ensure!(
            gpu.flash_static_submissions
                .each_ref()
                .map(std::cell::Cell::get)
                == before_static
        );
        ensure!(
            gpu.flash_fixed_submissions
                .each_ref()
                .map(std::cell::Cell::get)
                == before_fixed,
            "invalid input encoded fixed QMV"
        );
        let actual = if case.fp32 {
            let values: Vec<f32> = case
                .x
                .iter()
                .map(|&x| f32::from_bits(f16::from_bits(x).to_f32().to_bits() ^ 1))
                .collect();
            let input = gpu.f32_storage(&values)?.matrix(MatrixLayout {
                shape: [
                    u32::try_from(case.x.len() / layer.input as usize)?,
                    layer.input,
                ],
                strides: [i64::from(layer.input), 1],
                offset: 0,
            })?;
            layer.submit_fp16_view(&input)?.read()?
        } else {
            layer.submit(&case.x)?.read()?
        };
        let submissions = u64::from(enabled && !case.generic && nt == 2);
        ensure!(
            layer.nt2_submissions.get() == submissions,
            "case {i}: wrong encoded path count"
        );
        selected += submissions;
        let expected_fixed = if gpu.flash_fixed_setting == Some(true)
            && !case.generic
            && layer.k == 2
            && layer.cb == Codebook::Mul1
            && nt == 1
            && layer.splits == 1
        {
            match [layer.input, layer.output] {
                [2560, 640] => Some(0),
                [640, 2560] => Some(1),
                _ => None,
            }
        } else {
            None
        };
        ensure!(
            layer.fixed_key == expected_fixed,
            "case {i}: wrong fixed PSO identity"
        );
        let fixed_submissions = gpu
            .flash_fixed_submissions
            .each_ref()
            .map(std::cell::Cell::get)
            .iter()
            .zip(before_fixed)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        ensure!(
            fixed_submissions
                .iter()
                .enumerate()
                .all(|(key, &n)| n == u64::from(expected_fixed == Some(key as u8)))
        );
        let expected_static = if gpu.flash_static_setting == Some(true)
            && !case.generic
            && layer.cb == Codebook::Mul1
        {
            match (layer.k, nt) {
                (2, 1) => Some(0),
                (3, 1) => Some(1),
                (3, 2) => Some(2),
                (4, 1) => Some(3),
                (4, 2) => Some(4),
                _ => None,
            }
        } else {
            None
        };
        ensure!(
            layer.static_key == expected_static,
            "case {i}: wrong static PSO identity"
        );
        let static_submissions = gpu
            .flash_static_submissions
            .each_ref()
            .map(std::cell::Cell::get)
            .iter()
            .zip(before_static)
            .map(|(after, before)| after - before)
            .collect::<Vec<_>>();
        ensure!(
            static_submissions
                .iter()
                .enumerate()
                .all(|(key, &count)| count
                    == u64::from(expected_fixed.is_none() && expected_static == Some(key)))
        );
        ensure!(
            actual.len() == expected.len() && !actual.is_empty(),
            "case {i}: truncated output"
        );
        ensure!(
            actual
                .iter()
                .chain(expected)
                .all(|&w| f16::from_bits(w).is_finite()),
            "case {i}: nonfinite output"
        );
        ensure!(
            expected.iter().any(|&w| f16::from_bits(w) != f16::ZERO),
            "case {i}: trivial oracle"
        );
        let differs = actual.iter().zip(expected).filter(|(a, b)| a != b).count();
        unequal += differs;
        fs::write(
            root.join(format!("native-{}-{i}.bin", u8::from(enabled))),
            actual
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        result.push(serde_json::json!({"case":i,"prefix":case.prefix,"fp32":case.fp32,"nt":nt,"nt2_submissions":submissions,"words":actual.len(),"unequal":differs,"static_key":if expected_fixed.is_some() {None} else {expected_static},"static_submissions":static_submissions,"fixed_key":expected_fixed,"fixed_submissions":fixed_submissions}));
    }
    fs::write(
        root.join(format!("native-{}.json", u8::from(enabled))),
        serde_json::to_vec_pretty(
            &serde_json::json!({"parity":unequal==0,"cases":result,"nt2_submissions":selected,"unequal":unequal,"static_k":gpu.flash_static_setting == Some(true),"static_submissions":gpu.flash_static_submissions.each_ref().map(std::cell::Cell::get),"fixed_dims":gpu.flash_fixed_setting == Some(true),"fixed_submissions":gpu.flash_fixed_submissions.each_ref().map(std::cell::Cell::get)}),
        )?,
    )?;
    ensure!(!enabled || selected > 0, "candidate never encoded NT2");
    ensure!(unequal == 0, "resident NT2 diverged in {unequal} words");
    if gpu.ar30_cooperative {
        ensure!(super::ar30_kernel_tests::compiled_cooperative(&gpu)?);
        let dispatches = gpu.ar30_cooperative_uses[0].get();
        ensure!(dispatches > 0, "cooperative resident path was not executed");
        eprintln!(
            "AR30_COOPERATIVE_RESIDENT {}",
            serde_json::json!({
                "parity":true,"cases":cases.len(),"compiled_cooperative":true,
                "resident_dispatches":dispatches,"unequal":unequal
            })
        );
    }
    Ok(())
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
    routed: bool,
}

pub(super) struct PendingExpertFinish {
    pub(super) partial: Buffer,
    pub(super) partial_offset: u64,
    pub(super) svh: Buffer,
    pub(super) svh_offset: u64,
}
enum ReadyCohort {
    Finished(Vec<Buffer>),
    Pending(Vec<PendingExpertFinish>),
}

impl PendingExpertFinish {
    pub(super) fn encode_mix(
        gpu: &Metal,
        cmd: &CommandBufferRef,
        terms: &[&Self],
        shared: &Buffer,
        views: [&MatrixView; 3],
    ) -> Result<()> {
        let [scores, multiplier, out] = views;
        ensure!(
            terms.len() == 10 && gpu.owns_command_queue(cmd),
            "invalid finish/mix route count or command owner"
        );
        let pipeline = gpu
            .moe_finish_mix_pipeline
            .as_ref()
            .context("finish/mix pipeline was not loaded")?;
        ensure!(
            crate::expert_ar_policy::finish_mix_eligible(
                gpu.moe_finish_mix.get(),
                out.dtype == super::view::FloatType::F32,
                false,
                gpu.expert_ar_enabled[3].get(),
                pipeline.thread_execution_width() as usize,
                pipeline.max_total_threads_per_threadgroup() as usize
            ),
            "ineligible finish/mix pipeline"
        );
        for (view, width) in views.into_iter().zip([10, 1, 2560]) {
            ensure!(
                view.dtype == super::view::FloatType::F32
                    && view.shape() == [1, width]
                    && view.layout.offset == 0
                    && view.layout.strides[1] == 1
                    && view.buffer.device().registry_id() == gpu.device.registry_id(),
                "invalid finish/mix scalar/output view"
            );
        }
        ensure!(
            shared.length() >= 5120 && shared.device().registry_id() == gpu.device.registry_id(),
            "invalid shared contribution"
        );
        for term in terms {
            ensure!(
                crate::expert_ar_policy::finish_view_valid(
                    term.partial_offset,
                    term.partial.length(),
                    term.svh_offset,
                    term.svh.length()
                ) && term.partial.device().registry_id() == gpu.device.registry_id()
                    && term.svh.device().registry_id() == gpu.device.registry_id(),
                "invalid pending expert finish view"
            );
        }
        let mut bindings = terms
            .iter()
            .map(|term| (&term.partial, term.partial_offset))
            .collect::<Vec<_>>();
        bindings.extend(terms.iter().map(|term| (&term.svh, term.svh_offset)));
        bindings.extend([
            (shared, 0),
            (&scores.buffer, 0),
            (&multiplier.buffer, 0),
            (&out.buffer, 0),
        ]);
        Metal::encode_bound(
            cmd,
            pipeline,
            bindings.into_iter(),
            [2560u32],
            MTLSize::new(20, 1, 1),
            640,
        );
        #[cfg(test)]
        gpu.moe_finish_mix_uses
            .set(gpu.moe_finish_mix_uses.get() + 1);
        Ok(())
    }
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
        Ok(Self {
            gate,
            up,
            down,
            routed: false,
        })
    }

    pub(super) fn input_hidden(&self) -> [u32; 2] {
        [self.gate.input, self.gate.output]
    }

    pub(super) fn from_backing(gpu: &'a Metal, backing: &Buffer) -> Result<Self> {
        ensure!(
            backing.length() == flash_support::EXPERT_BYTES,
            "invalid Flash expert record length"
        );
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
                routed_record: true,
                simdgroups: 4,
                nt: 1,
                static_key: static_key(gpu, 2, Codebook::Mul1, 1, true)?,
                fixed_key: fixed_key(gpu, [input, output], splits, [2, 2, 1], true)?,
                scratch: std::cell::RefCell::new(None),
                #[cfg(test)]
                nt2_submissions: std::cell::Cell::new(0),
            })
        };
        Ok(Self {
            down: projection(0, 640, 2560)?,
            gate: projection(1, 2560, 640)?,
            up: projection(2, 2560, 640)?,
            routed: true,
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

    pub(super) fn encode_view_into(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        storage: Storage,
        destination: Option<(&Buffer, u64)>,
    ) -> Result<Buffer> {
        let extents = self.expert_extents(self.gate.view_extents(x)?)?;
        ensure!(
            destination.is_none() || crate::expert_ar_policy::eligible(self.routed, extents),
            "ineligible expert destination"
        );
        self.encode_into(cmd, Activation::View(x), extents, storage, destination)
    }

    /// Only experts already returned by the current ready cohort enter this grid.
    pub(super) fn encode_ready_cohort(
        cmd: &CommandBufferRef,
        x: &MatrixView,
        experts: &[&Self],
        destinations: &[Option<(&Buffer, u64)>],
        storage: Storage,
    ) -> Result<Vec<Buffer>> {
        match Self::encode_ready_cohort_impl(cmd, x, experts, destinations, storage, false)? {
            ReadyCohort::Finished(values) => Ok(values),
            ReadyCohort::Pending(_) => anyhow::bail!("unexpected pending expert finish"),
        }
    }
    pub(super) fn encode_ready_cohort_pending(
        cmd: &CommandBufferRef,
        x: &MatrixView,
        experts: &[&Self],
        destinations: &[Option<(&Buffer, u64)>],
    ) -> Result<Vec<PendingExpertFinish>> {
        match Self::encode_ready_cohort_impl(
            cmd,
            x,
            experts,
            destinations,
            Storage::StorageModePrivate,
            true,
        )? {
            ReadyCohort::Pending(values) => Ok(values),
            ReadyCohort::Finished(_) => anyhow::bail!("missing pending expert finish"),
        }
    }
    fn encode_ready_cohort_impl(
        cmd: &CommandBufferRef,
        x: &MatrixView,
        experts: &[&Self],
        destinations: &[Option<(&Buffer, u64)>],
        storage: Storage,
        pending_finish: bool,
    ) -> Result<ReadyCohort> {
        ensure!(
            crate::expert_ar_policy::cohort(experts.len(), destinations.len()),
            "invalid ready expert cohort"
        );
        let gpu = experts[0].gate.metal;
        let eligible = experts.iter().all(|expert| {
            std::ptr::eq(gpu, expert.gate.metal) && expert.routed && x.layout.shape == [1, 2560]
        });
        if !gpu.expert_ar_enabled[3].get() || !eligible {
            ensure!(
                !pending_finish,
                "pending finishes require the actual ready expert axis"
            );
            return experts
                .iter()
                .zip(destinations)
                .map(|(expert, &destination)| expert.encode_view_into(cmd, x, storage, destination))
                .collect::<Result<Vec<_>>>()
                .map(ReadyCohort::Finished);
        }
        for expert in experts {
            ensure!(
                crate::expert_ar_policy::eligible(
                    expert.routed,
                    expert.expert_extents(expert.gate.view_extents(x)?)?
                ),
                "invalid expert-axis extents"
            );
        }
        let n = experts.len();
        let dual = gpu.expert_ar_enabled[1].get();
        let fused = gpu.expert_ar_enabled[0].get();
        let stride = if dual { 5120u32 } else { 2560u32 };
        let prepared = buffer(
            &gpu.device,
            u64::from(stride) * 2 * n as u64,
            Storage::StorageModePrivate,
        )?;
        let partial = buffer(&gpu.device, 10240 * n as u64, Storage::StorageModePrivate)?;
        for (index, expert) in experts.iter().enumerate() {
            let offset = u64::from(stride) * 2 * index as u64;
            if dual {
                Metal::encode_bound(
                    cmd,
                    gpu.expert_gu_prepare
                        .as_ref()
                        .context("expert dual preparation pipeline was not loaded")?,
                    [
                        (&x.buffer, 0),
                        (&expert.gate.suh, expert.gate.offsets.suh),
                        (&expert.up.suh, expert.up.offsets.suh),
                        (&prepared, offset),
                    ]
                    .into_iter(),
                    [
                        x.layout.offset,
                        x.layout.strides[1],
                        i64::from(x.dtype == super::view::FloatType::F32),
                        2560,
                    ],
                    MTLSize::new(20, 1, 1),
                    64,
                );
                #[cfg(test)]
                gpu.expert_ar_uses[1].set(gpu.expert_ar_uses[1].get() + 1);
            } else {
                expert
                    .gate
                    .encode_input_into(cmd, Activation::View(x), &prepared, offset)?;
            }
        }
        Self::encode_axis_qmv(cmd, experts, 0, &prepared, 0, stride, &partial)?;
        let mut gates = Vec::with_capacity(n);
        for (index, expert) in experts.iter().enumerate() {
            let gate = buffer(&gpu.device, 1280, Storage::StorageModePrivate)?;
            expert
                .gate
                .encode_finish(cmd, &partial, index as u64 * 2560, &gate, 0, 640, None)?;
            gates.push(gate);
        }
        if !dual {
            for (index, expert) in experts.iter().enumerate() {
                expert.up.encode_input_into(
                    cmd,
                    Activation::View(x),
                    &prepared,
                    index as u64 * 5120,
                )?;
            }
        }
        Self::encode_axis_qmv(
            cmd,
            experts,
            1,
            &prepared,
            if dual { 5120 } else { 0 },
            stride,
            &partial,
        )?;
        let mut activations = Vec::with_capacity(n);
        for (index, expert) in experts.iter().enumerate() {
            let up = buffer(&gpu.device, 1280, Storage::StorageModePrivate)?;
            expert.up.encode_finish(
                cmd,
                &partial,
                index as u64 * 2560,
                &up,
                0,
                640,
                fused.then_some(&gates[index]),
            )?;
            let activated = if fused {
                #[cfg(test)]
                gpu.expert_ar_uses[0].set(gpu.expert_ar_uses[0].get() + 1);
                up
            } else {
                gpu.encode_swiglu(cmd, &gates[index], &up, Storage::StorageModePrivate)?
            };
            activations.push(activated);
        }
        for (index, (expert, activation)) in experts.iter().zip(&activations).enumerate() {
            expert.down.encode_input_into(
                cmd,
                Activation::Contiguous(activation),
                &prepared,
                index as u64 * 1280,
            )?;
        }
        Self::encode_axis_qmv(cmd, experts, 2, &prepared, 0, 640, &partial)?;
        #[cfg(test)]
        gpu.expert_ar_uses[3].set(gpu.expert_ar_uses[3].get() + n as u64);
        if pending_finish {
            ensure!(
                destinations.iter().all(Option::is_none),
                "pending finishes cannot materialize destinations"
            );
            let values = experts
                .iter()
                .enumerate()
                .map(|(index, expert)| PendingExpertFinish {
                    partial: partial.clone(),
                    partial_offset: index as u64 * 10240,
                    svh: expert.down.svh.clone(),
                    svh_offset: expert.down.offsets.svh,
                })
                .collect();
            return Ok(ReadyCohort::Pending(values));
        }
        let mut outputs = Vec::with_capacity(n);
        for (index, (expert, &destination)) in experts.iter().zip(destinations).enumerate() {
            let (output, offset) = if let Some((output, offset)) = destination {
                ensure!(
                    output.device().registry_id() == gpu.device.registry_id(),
                    "foreign expert-axis destination"
                );
                (output.clone(), offset)
            } else {
                (buffer(&gpu.device, 5120, storage)?, 0)
            };
            expert.down.encode_finish(
                cmd,
                &partial,
                index as u64 * 10240,
                &output,
                offset,
                2560,
                None,
            )?;
            outputs.push(output);
        }
        Ok(ReadyCohort::Finished(outputs))
    }

    fn axis_projection(&self, stage: usize) -> &FlashProjection<'a> {
        match stage {
            0 => &self.gate,
            1 => &self.up,
            _ => &self.down,
        }
    }

    fn axis_nt(projection: &FlashProjection<'_>) -> u32 {
        let gpu = projection.metal;
        if crate::flash_qmv::routed_nt2_key(
            [projection.input, projection.output],
            projection.splits,
            [projection.k, projection.cb as u32, projection.nt],
            projection.routed_record,
            gpu.routed_nt2_gu.get(),
            gpu.routed_nt2_down.get(),
        )
        .is_some()
        {
            2
        } else {
            projection.nt
        }
    }

    fn encode_axis_qmv(
        cmd: &CommandBufferRef,
        experts: &[&Self],
        stage: usize,
        prepared: &Buffer,
        offset: u64,
        stride: u32,
        partial: &Buffer,
    ) -> Result<()> {
        let first = experts[0].axis_projection(stage);
        let gpu = first.metal;
        let nt = Self::axis_nt(first);
        ensure!(matches!(nt, 1 | 2), "invalid expert-axis NT");
        ensure!(
            experts.iter().all(|expert| {
                let p = expert.axis_projection(stage);
                p.k == 2
                    && p.cb == Codebook::Mul1
                    && p.splits == 1
                    && p.input == first.input
                    && p.output == first.output
                    && (p.offsets.trellis, p.offsets.suh, p.offsets.svh)
                        == (first.offsets.trellis, first.offsets.suh, first.offsets.svh)
                    && Self::axis_nt(p) == nt
            }),
            "incompatible expert-axis record geometry"
        );
        let bindings: [(&Buffer, u64); 12] = std::array::from_fn(|index| match index {
            0 => (prepared, offset),
            11 => (partial, 0),
            _ => (
                &experts[(index - 1).min(experts.len() - 1)]
                    .axis_projection(stage)
                    .packed,
                0,
            ),
        });
        Metal::encode_bound(
            cmd,
            &gpu.expert_axis_qmv
                .as_ref()
                .context("expert-axis pipelines were not loaded")?[usize::from(stage == 2)]
                [usize::from(nt == 2)],
            bindings.into_iter(),
            [
                1,
                2,
                2,
                first.input,
                first.output,
                u32::try_from(first.offsets.trellis)?,
                u32::try_from(first.offsets.svh)?,
                stride,
            ],
            MTLSize::new(
                u64::from(
                    contracts::flash_reference_groups(first.output, nt)
                        .context("invalid axis grid")?,
                ),
                1,
                experts.len() as u64,
            ),
            128,
        );
        #[cfg(test)]
        if nt == 2 {
            let key = usize::from(stage == 2);
            gpu.routed_nt2_uses[key].set(gpu.routed_nt2_uses[key].get() + experts.len() as u64);
        }
        #[cfg(test)]
        gpu.expert_axis_dispatches
            .set(gpu.expert_axis_dispatches.get() + 1);
        #[cfg(test)]
        if gpu.ar30_cooperative {
            let count = &gpu.ar30_cooperative_uses[1];
            count.set(count.get() + 1);
        }
        Ok(())
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
        self.encode_into(
            cmd,
            x,
            [gate_extents, up_extents, down_extents],
            storage,
            None,
        )
    }

    fn encode_into(
        &self,
        cmd: &CommandBufferRef,
        x: Activation<'_>,
        [gate_extents, up_extents, down_extents]: [[u32; 3]; 3],
        storage: Storage,
        destination: Option<(&Buffer, u64)>,
    ) -> Result<Buffer> {
        let gpu = self.gate.metal;
        let reuse = crate::flash_expert_scratch::select(
            self.routed,
            [gate_extents, up_extents, down_extents],
            gpu.flash_expert_scratch_setting,
        )
        .map_err(|()| anyhow::anyhow!("MLXL3_NATIVE_ROUTED_SCRATCH must be 0 or 1"))?;
        let eligible = crate::expert_ar_policy::eligible(
            self.routed,
            [gate_extents, up_extents, down_extents],
        );
        let dual = eligible && gpu.expert_ar_enabled[1].get();
        let pair = if reuse {
            let [prepared, partial] = crate::flash_expert_scratch::CAPACITIES;
            Some((
                buffer(
                    &gpu.device,
                    if dual { prepared * 2 } else { prepared },
                    Storage::StorageModePrivate,
                )?,
                buffer(&gpu.device, partial, Storage::StorageModePrivate)?,
            ))
        } else {
            None
        };
        let scratch = pair.as_ref().map(|(prepared, partial)| (prepared, partial));
        let owned_dual = if dual && pair.is_none() {
            Some(buffer(&gpu.device, 10240, Storage::StorageModePrivate)?)
        } else {
            None
        };
        let prepared = if dual {
            Some(pair.as_ref().map_or_else(
                || owned_dual.as_ref().expect("owned dual preparation"),
                |pair| &pair.0,
            ))
        } else {
            None
        };
        if let Some(prepared) = prepared {
            let (input, offset, stride, fp32) = match x {
                Activation::Contiguous(input) => (input, 0, 1, false),
                Activation::View(x) => (
                    &x.buffer,
                    x.layout.offset,
                    x.layout.strides[1],
                    x.dtype == super::view::FloatType::F32,
                ),
            };
            ensure!(prepared.length() >= 10240, "short dual preparation");
            Metal::encode_bound(
                cmd,
                gpu.expert_gu_prepare
                    .as_ref()
                    .context("expert dual preparation pipeline was not loaded")?,
                [
                    (input, 0),
                    (&self.gate.suh, self.gate.offsets.suh),
                    (&self.up.suh, self.up.offsets.suh),
                    (prepared, 0),
                ]
                .into_iter(),
                [offset, stride, i64::from(fp32), 2560],
                MTLSize::new(20, 1, 1),
                64,
            );
            #[cfg(test)]
            gpu.expert_ar_uses[1].set(gpu.expert_ar_uses[1].get() + 1);
        }
        let gate = self.gate.encode_expert_stage(
            cmd,
            x,
            gate_extents,
            Storage::StorageModePrivate,
            scratch,
            ExpertStage {
                prepared: prepared.map(|b| (b, 0)),
                ..Default::default()
            },
        )?;
        let fused_up = crate::expert_ar_policy::eligible(
            self.routed,
            [gate_extents, up_extents, down_extents],
        ) && gpu.expert_ar_enabled[0].get();
        #[cfg(test)]
        let mut up_id = 0;
        let activated = if fused_up {
            let activated = self.up.encode_expert_stage(
                cmd,
                x,
                up_extents,
                Storage::StorageModePrivate,
                scratch,
                ExpertStage {
                    gate: Some(&gate),
                    prepared: prepared.map(|b| (b, 5120)),
                    ..Default::default()
                },
            )?;
            #[cfg(test)]
            gpu.expert_ar_uses[0].set(gpu.expert_ar_uses[0].get() + 1);
            activated
        } else {
            let up = self.up.encode_expert_stage(
                cmd,
                x,
                up_extents,
                Storage::StorageModePrivate,
                scratch,
                ExpertStage {
                    prepared: prepared.map(|b| (b, 5120)),
                    ..Default::default()
                },
            )?;
            #[cfg(test)]
            {
                up_id = up.as_ptr() as usize;
            }
            self.gate
                .metal
                .encode_swiglu(cmd, &gate, &up, Storage::StorageModePrivate)?
        };
        let output = self.down.encode_expert_stage(
            cmd,
            Activation::Contiguous(&activated),
            down_extents,
            storage,
            scratch,
            ExpertStage {
                destination,
                ..Default::default()
            },
        )?;
        #[cfg(test)]
        gpu.flash_expert_outputs.set([
            gate.as_ptr() as usize,
            up_id,
            activated.as_ptr() as usize,
            output.as_ptr() as usize,
        ]);
        Ok(output)
    }
}

impl FlashProjection<'_> {
    pub(super) fn release_idle_scratch(&self) -> Result<()> {
        let mut slot = self
            .scratch
            .try_borrow_mut()
            .map_err(|_| anyhow::anyhow!("resident scratch is borrowed"))?;
        if slot
            .as_ref()
            .is_some_and(|pair| pair.owner.status() == MTLCommandBufferStatus::Completed)
        {
            *slot = None;
        }
        ensure!(
            self.metal.resident_scratch.valid(),
            "resident scratch charge underflow"
        );
        Ok(())
    }
    pub(super) fn record_memory(&self, backings: &mut crate::gpu::memory::Backings) -> Result<()> {
        let scratch = self
            .scratch
            .try_borrow()
            .map_err(|_| anyhow::anyhow!("resident scratch is borrowed"))?;
        if let Some(pair) = scratch.as_ref() {
            backings.record("resident_projection_scratch", &pair.prepared)?;
            backings.record("resident_projection_scratch", &pair.partial)?;
        }
        for buffer in [&self.packed, &self.suh, &self.svh]
            .into_iter()
            .chain(&self.bias)
        {
            backings.record("fixed_weights", buffer)?;
        }
        Ok(())
    }
}
impl FlashExpert<'_> {
    pub(super) fn record_memory(
        &self,
        backings: &mut crate::gpu::memory::Backings,
        category: &'static str,
    ) -> Result<()> {
        for projection in [&self.gate, &self.up, &self.down] {
            for buffer in [&projection.packed, &projection.suh, &projection.svh]
                .into_iter()
                .chain(&projection.bias)
            {
                backings.record(category, buffer)?;
            }
        }
        Ok(())
    }
}

/// Ordinary projection bundles preserve the combined output-width plan.
pub struct ProjectionGroup<'a> {
    gpu: &'a Metal,
    layers: Vec<FlashProjection<'a>>,
    input: u32,
    plan: Option<[u32; 3]>,
}
impl<'a> ProjectionGroup<'a> {
    pub fn load<S: AsRef<str>>(gpu: &'a Metal, c: &Checkpoint, prefixes: &[S]) -> Result<Self> {
        Self::from_projections(
            prefixes
                .iter()
                .map(|p| FlashProjection::load_exl3(gpu, c, p.as_ref()))
                .collect::<Result<Vec<_>>>()?,
        )
    }
    pub fn from_projections(layers: Vec<FlashProjection<'a>>) -> Result<Self> {
        let first = layers.first().context("empty projection bundle")?;
        let gpu = first.metal;
        let input = first.input;
        ensure!(
            layers
                .iter()
                .all(|p| std::ptr::eq(p.metal, gpu) && p.input == input),
            "projection bundle requires one input width and Metal context"
        );
        let compatible = layers.len() >= 2
            && first.k != 7
            && layers
                .iter()
                .all(|p| p.generic && p.k == first.k && p.cb == first.cb);
        let plan = if compatible {
            let widths = layers.iter().map(|p| p.output).collect::<Vec<_>>();
            let total = contracts::projection_bundle_width(&widths)
                .context("projection bundle width overflow")?;
            let m5 = first.simdgroups == 8;
            let mut plan = contracts::mapped_expert_plan(input, total, 1, 1, first.k, m5)
                .context("invalid grouped projection plan")?;
            if std::env::var("MLXL3_DENSE_DECODE_NT4").as_deref() != Ok("0")
                && i32::try_from(input)
                    .ok()
                    .zip(i32::try_from(total).ok())
                    .is_some_and(|(i, o)| {
                        contracts::dense_decode_nt4(
                            i,
                            o,
                            first.k as usize,
                            first.cb == Codebook::Mul1,
                            m5,
                        )
                    })
            {
                plan[1] = 4;
            }
            Some(plan)
        } else {
            None
        };
        Ok(Self {
            gpu,
            layers,
            input,
            plan,
        })
    }
    pub fn input_width(&self) -> u32 {
        self.input
    }
    pub fn output_widths(&self) -> Vec<u32> {
        self.layers.iter().map(|p| p.output).collect()
    }
    pub fn is_grouped(&self) -> bool {
        self.plan.is_some()
    }
    pub fn forward(&self, x: &MatrixView) -> Result<Vec<MatrixView>> {
        self.gpu
            .command(|cmd| self.encode_row(cmd, x, Storage::StorageModeShared))
    }
    pub(super) fn encode_row(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        storage: Storage,
    ) -> Result<Vec<MatrixView>> {
        use super::view::FloatType;
        ensure!(
            x.shape() == [1, self.input]
                && x.dtype == FloatType::F16
                && x.buffer.device().registry_id() == self.gpu.device.registry_id(),
            "invalid ordinary projection bundle input"
        );
        self.layers
            .iter()
            .map(|p| {
                let Some(plan) = self.plan else {
                    return p.encode_row(cmd, x, storage);
                };
                let prepared = buffer(
                    &self.gpu.device,
                    u64::from(self.input) * 2,
                    Storage::StorageModePrivate,
                )?;
                let scales = MatrixView::column_scales(&p.suh, p.offsets.suh, 1, self.input)?;
                self.gpu.encode_prepare_view(cmd, x, &scales, &prepared)?;
                let raw = p.encode_mapped_raw(cmd, &prepared, plan)?;
                let output = MatrixView::output(self.gpu, FloatType::F16, 1, p.output, storage)?;
                Metal::encode_bound(
                    cmd,
                    &self.gpu.dense_finish,
                    [
                        (&raw, 0),
                        (&p.svh, p.offsets.svh),
                        p.bias.as_ref().map_or((&p.svh, p.offsets.svh), |b| (b, 0)),
                        (&output.buffer, 0),
                    ]
                    .into_iter(),
                    [plan[0], 0, u32::from(p.bias.is_some()), 0, p.output],
                    MTLSize::new(u64::from(p.output / 128), 1, 1),
                    64,
                );
                Ok(output)
            })
            .collect()
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

#[cfg(test)]
#[test]
#[ignore = "physical Qwen ordinary bundle: rejected inputs and outputs retained after owner release"]
fn ordinary_projection_group_preserves_owners_and_rejects_wrong_inputs() -> Result<()> {
    use crate::tensor_layout::MatrixLayout;
    let c = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
        "MLXL3_QWEN_NATIVE_MODEL",
    )?))?;
    let gpu = Metal::new()?;
    let prefixes = ["q_proj", "k_proj", "v_proj"]
        .map(|p| format!("model.language_model.layers.3.self_attn.{p}"));
    let layer = ProjectionGroup::load(&gpu, &c, &prefixes)?;
    ensure!(layer.is_grouped() && layer.output_widths() == [8192, 512, 512]);
    ensure!(ProjectionGroup::load::<&str>(&gpu, &c, &[]).is_err());
    let mixed = [
        prefixes[0].as_str(),
        "model.language_model.layers.0.mlp.shared_expert.down_proj",
    ];
    ensure!(ProjectionGroup::load(&gpu, &c, &mixed).is_err());
    let width = layer.input_width();
    let x = gpu
        .f16_storage(
            &(0..width)
                .map(|i| f16::from_f32(((i * 37) % 199) as f32 / 233. - 99. / 233.).to_bits())
                .collect::<Vec<_>>(),
        )?
        .matrix(MatrixLayout {
            shape: [1, width],
            strides: [i64::from(width), 1],
            offset: 0,
        })?;
    let bad = gpu.f16_storage(&[0])?.matrix(MatrixLayout {
        shape: [1, 1],
        strides: [1, 1],
        offset: 0,
    })?;
    ensure!(layer.forward(&bad).is_err());
    let wrong = gpu
        .f32_storage(&vec![0.; width as usize])?
        .matrix(MatrixLayout {
            shape: [1, width],
            strides: [i64::from(width), 1],
            offset: 0,
        })?;
    ensure!(layer.forward(&wrong).is_err());
    let expected = layer
        .forward(&x)?
        .iter()
        .map(|v| v.read_output())
        .collect::<Result<Vec<_>>>()?;
    ensure!(expected.iter().all(|a| !a.is_empty()
        && a.iter().any(|&v| v != 0)
        && a.iter().all(|&v| f16::from_bits(v as u16).is_finite())));
    let outputs = layer.forward(&x)?;
    drop(layer);
    drop(c);
    drop(x);
    drop(gpu);
    ensure!(
        outputs
            .iter()
            .map(|v| v.read_output())
            .collect::<Result<Vec<_>>>()?
            == expected
    );
    Ok(())
}

#[cfg(test)]
#[path = "routed_scratch_tests.rs"]
mod routed_scratch_tests;

#[cfg(test)]
#[path = "routed_nt2_tests.rs"]
mod routed_nt2_tests;

#[cfg(test)]
#[path = "ar20_projection_tests.rs"]
mod ar20_projection_tests;

#[cfg(test)]
#[path = "resident_bundle_tests.rs"]
mod resident_bundle_tests;

#[cfg(test)]
#[path = "ar20_finish_mix_tests.rs"]
mod ar20_finish_mix_tests;

#[cfg(test)]
#[path = "resident_scratch_tests.rs"]
mod resident_scratch_tests;
