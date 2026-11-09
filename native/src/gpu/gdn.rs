use super::CachedPipeline as ComputePipelineState;
use super::{FlashProjection, MatrixView, Metal, view::FloatType};
use crate::checkpoint::Checkpoint;
use anyhow::{Context, Result, ensure};
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, MTLCommandBufferStatus, MTLResourceOptions as Storage,
    MTLSize, NSRange,
};
use std::{collections::BTreeMap, rc::Rc};

pub(super) struct Pipelines {
    from: [ComputePipelineState; 2],
    dense: ComputePipelineState,
    joint_ab: Option<ComputePipelineState>,
    whole: Option<ComputePipelineState>,
    pub(super) temporal: Option<[ComputePipelineState; 2]>,
    whole_head_gates: Option<ComputePipelineState>,
    prepare: ComputePipelineState,
    step: ComputePipelineState,
    norm: ComputePipelineState,
    norm_gate: Option<ComputePipelineState>,
    gate: ComputePipelineState,
    pub(super) to_float: ComputePipelineState,
}
impl Pipelines {
    pub(super) fn load(
        p: &impl Fn(&str) -> Result<ComputePipelineState>,
        norm_gate: bool,
        joint_ab: bool,
        whole: bool,
        head_gates: bool,
        temporal: bool,
    ) -> Result<Self> {
        Ok(Self {
            from: [p("gdn_from_half")?, p("gdn_from_float")?],
            dense: p("gdn_dense_gate")?,
            joint_ab: joint_ab.then(|| p("gdn_joint_ab")).transpose()?,
            whole: whole.then(|| p("ar10_gdn_whole")).transpose()?,
            temporal: (whole && temporal)
                .then(|| -> Result<_> {
                    Ok([p("flash_gdn_temporal_b2")?, p("flash_gdn_temporal_b4")?])
                })
                .transpose()?,
            whole_head_gates: (whole && head_gates)
                .then(|| p("ar20_gdn_whole_head_gates"))
                .transpose()?,
            prepare: p("gdn_prepare_row")?,
            step: p("gdn_step_row")?,
            norm: p("gdn_norm_row")?,
            norm_gate: norm_gate.then(|| p("gdn_norm_gate_row")).transpose()?,
            gate: p("gdn_gate_row")?,
            to_float: p("gdn_to_float")?,
        })
    }
}

#[derive(Clone)]
struct Frame {
    generation: u64,
    history: MatrixView,
    recurrent: MatrixView,
}
/// The retained state belongs to one checkpoint layer and may be restored after a divergent branch.
#[derive(Clone)]
pub struct FlashGdnState {
    owner: Rc<()>,
    frame: Option<Rc<Frame>>,
}

// The standard command buffer retains encoded resources through completion.
// Pending keeps the state/output owners and the publication-status guard.
struct PendingGdn {
    #[cfg(test)]
    test_gated: MatrixView,
    command: CommandBuffer,
    next: Rc<Frame>,
    _previous: Rc<Frame>,
    values: MatrixView,
    steps: Option<[(&'static str, MatrixView); 11]>,
}

struct GdnProjections {
    input: MatrixView,
    qkv: MatrixView,
    z: MatrixView,
}

pub(super) struct PreparedGdnRow {
    read: super::FlashHcRead,
    pending: PendingGdn,
    post: Option<EncodedMoeRow>,
}
struct EncodedMoeRow {
    streams: MatrixView,
    read: super::hc::ReadBuffers,
    routing: Option<super::gate::GateBuffers>,
}
impl EncodedMoeRow {
    fn finish(self) -> Result<PreparedMoeRow> {
        let read = self.read.finish(false)?;
        Ok(PreparedMoeRow {
            mixed: read.mixed,
            gate: read.gate.context("missing prepared MLP gate")?,
            streams: self.streams,
            routing: self.routing.map(|gate| gate.routes),
        })
    }
}

#[cfg(test)]
thread_local! {
    static BLOCK_PROJECTION_BATCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static TEMPORAL_BATCHES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub type FlashGdnOutput = super::FlashAttentionOutput;

pub(super) struct DeferredHcWrite {
    streams: MatrixView,
    branch: MatrixView,
    gate: MatrixView,
    producer: Option<super::MoeOutput>,
}
#[allow(
    clippy::large_enum_variant,
    reason = "one pending row owns its GPU producers on the stack, avoiding a per-boundary allocation"
)]
pub(super) enum LayerOutput {
    Ready(MatrixView),
    Pending(DeferredHcWrite),
}
impl LayerOutput {
    pub(super) fn complete(self, gpu: &Metal) -> Result<MatrixView> {
        match self {
            Self::Ready(streams) => Ok(streams),
            Self::Pending(mut write) => {
                let encoded = gpu.command(|cmd| write.encode(gpu, cmd));
                let producer = write.finish_producer();
                let streams = encoded?;
                producer?;
                Ok(streams)
            }
        }
    }
    pub(super) fn ready(&self) -> Result<&MatrixView> {
        match self {
            Self::Ready(streams) => Ok(streams),
            Self::Pending(_) => anyhow::bail!("deferred HC output cannot escape the layer loop"),
        }
    }
}
impl DeferredHcWrite {
    fn encode(&self, gpu: &Metal, cmd: &CommandBufferRef) -> Result<MatrixView> {
        gpu.encode_hc_write(cmd, &self.streams, &self.branch, &self.gate)
    }
    fn finish_producer(&mut self) -> Result<()> {
        if let Some(producer) = self.producer.take() {
            producer.completed()?;
        }
        Ok(())
    }
}
impl Drop for DeferredHcWrite {
    fn drop(&mut self) {
        // An abandoned/failed consumer must still join all submitted producers.
        let _ = self.finish_producer();
    }
}

pub struct FlashGdn<'a> {
    gpu: &'a Metal,
    qkv: FlashProjection<'a>,
    z: FlashProjection<'a>,
    out: FlashProjection<'a>,
    a: Buffer,
    b: Buffer,
    conv: Buffer,
    a_log: Buffer,
    dt_bias: Buffer,
    gain: Buffer,
    eps: f32,
    state: FlashGdnState,
    frame_pool: Option<std::cell::RefCell<Vec<Rc<Frame>>>>,
}
impl<'a> FlashGdn<'a> {
    pub fn load(gpu: &'a Metal, c: &Checkpoint, prefix: &str, eps: f32) -> Result<Self> {
        ensure!(eps.is_finite() && eps > 0., "invalid Flash GDN epsilon");
        let proj = |name: &str, dims| -> Result<_> {
            let p = FlashProjection::load(gpu, c, &format!("{prefix}.{name}"))?;
            ensure!(
                p.dimensions() == dims,
                "invalid Flash GDN projection {name}"
            );
            Ok(p)
        };
        let weight = |name: &str, shape: &[usize], half: bool| {
            super::checkpoint_float_weight(
                gpu,
                c,
                &format!("{prefix}.{name}"),
                shape,
                if half { FloatType::F16 } else { FloatType::F32 },
            )
        };
        let arena_flag = std::env::var("MLXL3_NATIVE_FLASH_FRAME_ARENA");
        let arena = match &arena_flag {
            Ok(value) => Some(value.as_str()),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(anyhow::anyhow!("invalid frame arena setting: {error}")),
        };
        let arena = crate::native_executor::default_flag(
            arena,
            gpu.device_name() == "Apple M5",
            gpu.flash_defaults,
        )
        .context("MLXL3_NATIVE_FLASH_FRAME_ARENA must be 0 or 1")?;
        Ok(Self {
            gpu,
            frame_pool: arena.then(|| std::cell::RefCell::new(Vec::new())),
            qkv: proj("in_proj_qkv", [2560, 10240])?,
            z: proj("in_proj_z", [2560, 6144])?,
            out: proj("out_proj", [6144, 2560])?,
            a: weight("in_proj_a.weight", &[48, 2560], true)?,
            b: weight("in_proj_b.weight", &[48, 2560], true)?,
            conv: weight("conv1d.weight", &[10240, 1, 4], true)?,
            a_log: weight("A_log", &[48], false)?,
            dt_bias: weight("dt_bias", &[48], true)?,
            gain: weight("norm.weight", &[128], false)?,
            eps,
            state: FlashGdnState {
                owner: Rc::new(()),
                frame: None,
            },
        })
    }
    fn lease_frame(&self, storage: Storage) -> Result<Rc<Frame>> {
        let make = |generation| -> Result<Rc<Frame>> {
            #[cfg(test)]
            let _scope = crate::flash_profile::span("native.gdn.frame");
            let frame = Rc::new(Frame {
                generation,
                history: MatrixView::output(self.gpu, FloatType::F16, 1, 30720, storage)?,
                recurrent: MatrixView::output(
                    self.gpu,
                    FloatType::F32,
                    1,
                    48 * 128 * 128,
                    storage,
                )?,
            });
            ensure!(
                frame.history.buffer.length() + frame.recurrent.buffer.length()
                    == crate::frame_arena::FRAME_BYTES,
                "unexpected GDN frame allocation"
            );
            Ok(frame)
        };
        let Some(pool) = &self.frame_pool else {
            return make(0);
        };
        let mut pool = pool.borrow_mut();
        for slot in pool.iter_mut() {
            if let Some(epoch) =
                crate::frame_arena::reusable_epoch(Rc::strong_count(slot), slot.generation)
            {
                let shared = storage == Storage::StorageModeShared;
                if (slot.history.buffer.storage_mode() == metal::MTLStorageMode::Shared) == shared {
                    Rc::get_mut(slot)
                        .context("frame arena has a weak owner")?
                        .generation = epoch;
                } else {
                    *slot = make(epoch)?;
                }
                return Ok(slot.clone());
            }
        }
        ensure!(
            pool.len() < crate::frame_arena::CAPACITY,
            "GDN frame arena exhausted by retained snapshots or GPU consumers"
        );
        let frame = make(0)?;
        pool.push(frame.clone());
        Ok(frame)
    }

    pub fn snapshot(&self) -> FlashGdnState {
        self.state.clone()
    }
    pub fn restore(&mut self, snapshot: &FlashGdnState) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.state.owner, &snapshot.owner),
            "GDN state belongs to a different layer"
        );
        self.state.frame = snapshot.frame.clone();
        Ok(())
    }
    pub fn reset(&mut self) {
        self.state.frame = None;
    }

    pub fn forward(&mut self, x: &MatrixView, trace: bool) -> Result<FlashGdnOutput> {
        #[cfg(test)]
        let _scope = crate::flash_profile::span("native.gdn.body");
        let pending = self.gpu.command(|cmd| self.encode(cmd, x, trace))?;
        self.finish(pending)
    }

    fn encode(&self, cmd: &CommandBufferRef, x: &MatrixView, trace: bool) -> Result<PendingGdn> {
        self.encode_for_consumer(cmd, x, trace, false)
    }

    fn encode_norm_gate(
        &self,
        cmd: &CommandBufferRef,
        delta: &MatrixView,
        z: &MatrixView,
        trace: bool,
        private_f32_consumer: bool,
    ) -> Result<(Option<MatrixView>, MatrixView)> {
        let gpu = self.gpu;
        let p = &gpu.gdn;
        let storage = if trace {
            Storage::StorageModeShared
        } else {
            Storage::StorageModePrivate
        };
        if crate::flash_support::gdn_private_epilogue_eligible(
            gpu.gdn_norm_gate.get(),
            trace,
            private_f32_consumer,
        ) {
            let pipeline = p
                .norm_gate
                .as_ref()
                .context("GDN norm/gate pipeline was not selected at initialization")?;
            ensure!(pipeline.thread_execution_width() == 32);
            let gated = MatrixView::output(gpu, FloatType::F16, 1, 6144, storage)?;
            Metal::encode(
                cmd,
                pipeline,
                &[&delta.buffer, &self.gain, &z.buffer, &gated.buffer],
                [self.eps.to_bits()],
                MTLSize::new(48, 1, 1),
                32,
            );
            #[cfg(test)]
            gpu.gdn_norm_gate_uses.set(gpu.gdn_norm_gate_uses.get() + 1);
            return Ok((None, gated));
        }
        let normed = MatrixView::output(gpu, FloatType::F32, 1, 6144, storage)?;
        Metal::encode(
            cmd,
            &p.norm,
            &[&delta.buffer, &self.gain, &normed.buffer],
            [self.eps.to_bits()],
            MTLSize::new(48, 1, 1),
            32,
        );
        let gated = MatrixView::output(gpu, FloatType::F16, 1, 6144, storage)?;
        Metal::encode(
            cmd,
            &p.gate,
            &[&normed.buffer, &z.buffer, &gated.buffer],
            [6144u32],
            MTLSize::new(192, 1, 1),
            32,
        );
        Ok((Some(normed), gated))
    }

    fn encode_for_consumer(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        trace: bool,
        private_f32_consumer: bool,
    ) -> Result<PendingGdn> {
        self.encode_for_consumer_prepared(cmd, x, trace, private_f32_consumer, None, None)
    }

    fn encode_for_consumer_prepared(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        trace: bool,
        private_f32_consumer: bool,
        projected: Option<&GdnProjections>,
        previous: Option<&Rc<Frame>>,
    ) -> Result<PendingGdn> {
        let gpu = self.gpu;
        ensure!(
            x.shape() == [1, 2560] && x.buffer.device().registry_id() == gpu.device.registry_id(),
            "invalid Flash GDN input shape or device"
        );
        let p = &gpu.gdn;
        ensure!(
            p.dense.thread_execution_width() == 32
                && p.dense.max_total_threads_per_threadgroup() >= 256
                && p.step.thread_execution_width() == 32
                && p.step.max_total_threads_per_threadgroup() >= 64
                && p.prepare.thread_execution_width() == 32
                && p.norm.thread_execution_width() == 32,
            "unsupported Flash GDN Metal SIMD width"
        );
        let storage = if trace {
            Storage::StorageModeShared
        } else {
            Storage::StorageModePrivate
        };
        let matrix = |dtype, cols| MatrixView::output(gpu, dtype, 1, cols, storage);
        let half = |cols| matrix(FloatType::F16, cols);
        let (value_groups, _) =
            crate::contracts::packed_gdn_grid(48, 128).context("invalid Flash GDN state extent")?;
        let g = |x, y, z| MTLSize::new(x, y, z);
        let prev = match previous.or(self.state.frame.as_ref()) {
            Some(previous) => previous.clone(),
            None => {
                let zero = self.lease_frame(storage)?;
                let blit = cmd.new_blit_command_encoder();
                for v in [&zero.history, &zero.recurrent] {
                    blit.fill_buffer(&v.buffer, NSRange::new(0, v.buffer.length()), 0);
                }
                blit.end_encoding();
                zero
            }
        };
        let next = self.lease_frame(storage)?;
        let input = if let Some(projected) = projected {
            projected.input.clone()
        } else {
            let input = half(2560)?;
            Metal::encode(
                cmd,
                &p.from[usize::from(x.dtype == FloatType::F32)],
                &[&x.buffer, &input.buffer],
                [x.layout.offset, x.layout.strides[1]],
                g(80, 1, 1),
                32,
            );
            input
        };
        let pair = if projected.is_none()
            && crate::flash_support::gdn_private_epilogue_eligible(
                gpu.gdn_resident_bundle.get(),
                trace,
                private_f32_consumer,
            ) {
            self.qkv
                .encode_resident_pair(&self.z, cmd, &input, storage)?
        } else {
            None
        };
        let [qkv, z] = match (projected, pair) {
            (Some(projected), _) => [projected.qkv.clone(), projected.z.clone()],
            (_, Some(pair)) => pair,
            _ => [
                self.qkv.encode_row(cmd, &input, storage)?,
                self.z.encode_row(cmd, &input, storage)?,
            ],
        };
        let dense = |w: &Buffer| -> Result<MatrixView> {
            let gate = half(48)?;
            Metal::encode(
                cmd,
                &p.dense,
                &[&input.buffer, w, &gate.buffer],
                [48u32],
                g(12, 1, 1),
                256,
            );
            Ok(gate)
        };
        let (a, b) = if crate::flash_support::gdn_private_epilogue_eligible(
            gpu.gdn_joint_ab.get(),
            trace,
            private_f32_consumer,
        ) {
            let a = half(48)?;
            let b = half(48)?;
            Metal::encode(
                cmd,
                p.joint_ab
                    .as_ref()
                    .context("joint a/b pipeline not initialized")?,
                &[&input.buffer, &self.a, &self.b, &a.buffer, &b.buffer],
                [48u32],
                g(24, 1, 1),
                256,
            );
            #[cfg(test)]
            gpu.gdn_joint_ab_uses.set(gpu.gdn_joint_ab_uses.get() + 1);
            (a, b)
        } else {
            (dense(&self.a)?, dense(&self.b)?)
        };
        let head_gate_pipeline = p.whole_head_gates.as_ref().filter(|p| {
            crate::flash_support::ar20_head_gates_eligible(
                gpu.gdn_head_gates.get(),
                gpu.gdn_whole.get(),
                trace,
                private_f32_consumer,
                p.thread_execution_width() as u32,
                p.max_total_threads_per_threadgroup() as u32,
            )
        });
        let whole_pipeline = head_gate_pipeline.or_else(|| {
            p.whole.as_ref().filter(|p| {
                crate::flash_support::ar10_whole_gdn_eligible(
                    gpu.gdn_whole.get(),
                    trace,
                    private_f32_consumer,
                    p.thread_execution_width() as u32,
                    p.max_total_threads_per_threadgroup() as u32,
                )
            })
        });
        let (conv, q, k, v, delta, normed, gated) = if let Some(pipeline) = whole_pipeline {
            let gated = half(6144)?;
            Metal::encode(
                cmd,
                pipeline,
                &[
                    &qkv.buffer,
                    &z.buffer,
                    &a.buffer,
                    &b.buffer,
                    &prev.history.buffer,
                    &self.conv,
                    &self.a_log,
                    &self.dt_bias,
                    &prev.recurrent.buffer,
                    &next.history.buffer,
                    &next.recurrent.buffer,
                    &self.gain,
                    &gated.buffer,
                ],
                [self.eps.to_bits()],
                g(48, 1, 1),
                512,
            );
            #[cfg(test)]
            {
                gpu.gdn_whole_uses.set(gpu.gdn_whole_uses.get() + 1);
                if head_gate_pipeline.is_some() {
                    gpu.gdn_head_gate_uses.set(gpu.gdn_head_gate_uses.get() + 1);
                }
            }
            (None, None, None, None, None, None, gated)
        } else {
            let conv = half(10240)?;
            let q = half(2048)?;
            let k = half(2048)?;
            let v = half(6144)?;
            Metal::encode(
                cmd,
                &p.prepare,
                &[
                    &qkv.buffer,
                    &prev.history.buffer,
                    &self.conv,
                    &conv.buffer,
                    &q.buffer,
                    &k.buffer,
                    &v.buffer,
                    &next.history.buffer,
                ],
                [10240u32],
                g(1, 80, 1),
                32,
            );
            let delta = half(6144)?;
            Metal::encode(
                cmd,
                &p.step,
                &[
                    &q.buffer,
                    &k.buffer,
                    &v.buffer,
                    &a.buffer,
                    &b.buffer,
                    &self.a_log,
                    &self.dt_bias,
                    &prev.recurrent.buffer,
                    &delta.buffer,
                    &next.recurrent.buffer,
                ],
                [128u32],
                g(1, u64::from(value_groups), 48),
                64,
            );
            let (normed, gated) =
                self.encode_norm_gate(cmd, &delta, &z, trace, private_f32_consumer)?;
            (
                Some(conv),
                Some(q),
                Some(k),
                Some(v),
                Some(delta),
                normed,
                gated,
            )
        };
        let out_half = self.out.encode_row(cmd, &gated, storage)?;
        let values = if crate::flash_support::gdn_private_epilogue_eligible(
            gpu.gdn_half_elision.get(),
            trace,
            private_f32_consumer,
        ) {
            #[cfg(test)]
            gpu.gdn_half_elisions.set(gpu.gdn_half_elisions.get() + 1);
            out_half
        } else {
            let values =
                MatrixView::output(gpu, FloatType::F32, 1, 2560, Storage::StorageModeShared)?;
            Metal::encode(
                cmd,
                &p.to_float,
                &[&out_half.buffer, &values.buffer],
                [2560u32],
                g(80, 1, 1),
                32,
            );
            values
        };
        #[cfg(test)]
        let test_gated = gated.clone();
        let steps = if trace {
            Some([
                ("qkv", qkv),
                ("z", z),
                ("a", a),
                ("b", b),
                ("conv", conv.context("trace conv missing")?),
                ("q", q.context("trace q missing")?),
                ("k", k.context("trace k missing")?),
                ("v", v.context("trace v missing")?),
                ("delta", delta.context("trace delta missing")?),
                (
                    "normed",
                    normed.context("trace requires materialized norm")?,
                ),
                ("gated", gated),
            ])
        } else {
            None
        };
        Ok(PendingGdn {
            #[cfg(test)]
            test_gated,
            command: cmd.to_owned(),
            next,
            _previous: prev,
            values,
            steps,
        })
    }

    fn finish(&mut self, pending: PendingGdn) -> Result<FlashGdnOutput> {
        ensure!(
            pending.command.status() == MTLCommandBufferStatus::Completed,
            "GDN state cannot publish before command completion"
        );
        let PendingGdn {
            next,
            values,
            steps,
            ..
        } = pending;
        let trace = steps
            .map(|steps| -> Result<BTreeMap<_, _>> {
                let mut result = steps
                    .into_iter()
                    .map(|(k, v)| Ok((k, v.read_output()?)))
                    .collect::<Result<BTreeMap<_, _>>>()?;
                result.insert("history", next.history.read_output()?);
                result.insert("state", next.recurrent.read_output()?);
                Ok(result)
            })
            .transpose()?;
        self.state.frame = Some(next);
        Ok(FlashGdnOutput { values, trace })
    }

    pub(super) fn forward_after_hc(
        &mut self,
        hc: &super::FlashHc,
        streams: &MatrixView,
        trace: bool,
    ) -> Result<(super::FlashHcRead, FlashGdnOutput)> {
        let gpu = self.gpu;
        if !crate::flash_support::hc_gdn_chain_eligible(
            gpu.hc_gdn_chain.get(),
            trace,
            std::ptr::eq(gpu, hc.gpu),
            streams.shape(),
        ) {
            let read = hc.read(streams, trace)?;
            let attention = self.forward(&read.mixed, trace)?;
            return Ok((read, attention));
        }
        #[cfg(test)]
        let _scope = crate::flash_profile::span("native.hc_gdn.joined");
        let (read, pending) = gpu.command(|cmd| {
            let read = hc.encode_read(cmd, streams, false)?;
            let pending = self.encode_for_consumer(
                cmd,
                read.mixed(),
                false,
                streams.dtype == FloatType::F32,
            )?;
            Ok((read, pending))
        })?;
        let read = read.finish(false)?;
        let attention = self.finish(pending)?;
        #[cfg(test)]
        gpu.hc_gdn_pairs.set(gpu.hc_gdn_pairs.get() + 1);
        Ok((read, attention))
    }
}

/// Owned rows crossing the temporal attention / stateless MoE boundary.
/// HC workspace must be disabled while these rows are retained.
pub(super) struct PreparedMoeRow {
    pub(super) mixed: MatrixView,
    pub(super) routing: Option<super::RouteOutput>,
    streams: MatrixView,
    gate: MatrixView,
}

pub(super) struct LayerResidual<'a> {
    attn_hc: super::FlashHc<'a>,
    mlp_hc: super::FlashHc<'a>,
    pub(super) moe: super::FlashMoeRow<'a>,
}
impl<'a> LayerResidual<'a> {
    pub(super) fn load(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        eps: f32,
        slots: usize,
    ) -> Result<Self> {
        Ok(Self {
            attn_hc: super::FlashHc::load(
                gpu,
                c,
                &format!("{prefix}.attn_hyper_connection"),
                eps,
                true,
            )?,
            mlp_hc: super::FlashHc::load(
                gpu,
                c,
                &format!("{prefix}.mlp_hyper_connection"),
                eps,
                true,
            )?,
            moe: super::FlashMoeRow::load(gpu, c, &format!("{prefix}.mlp"), slots)?,
        })
    }
    pub(super) fn attention_input(&self, streams: &MatrixView) -> Result<MatrixView> {
        Ok(self.attn_hc.read(streams, false)?.mixed)
    }

    pub(super) fn encode_attention_input(
        &self,
        cmd: &metal::CommandBufferRef,
        streams: &MatrixView,
    ) -> Result<super::hc::ReadBuffers> {
        self.attn_hc.encode_read(cmd, streams, false)
    }
    pub(super) fn prepare_moe(
        &self,
        streams: &MatrixView,
        attend: impl FnOnce(&MatrixView) -> Result<super::FlashAttentionOutput>,
    ) -> Result<PreparedMoeRow> {
        ensure!(
            (!self.mlp_hc.gpu.hc_workspace_enabled.get() || self.mlp_hc.gpu.common_executor),
            "cohort rows require owned HC activations"
        );
        let read = self.attn_hc.read(streams, false)?;
        let attention = attend(&read.mixed)?;
        self.prepare_after_attention(streams, read, attention)
    }

    fn prepare_after_attention(
        &self,
        streams: &MatrixView,
        read: super::FlashHcRead,
        attention: super::FlashAttentionOutput,
    ) -> Result<PreparedMoeRow> {
        ensure!(
            (!self.mlp_hc.gpu.hc_workspace_enabled.get() || self.mlp_hc.gpu.common_executor),
            "cohort rows require owned HC activations"
        );
        let gate = read
            .gate
            .as_ref()
            .context("missing attention injection gate")?;
        let (streams, mlp_read, routing) = if self.moe.joint_gate_eligible(0) {
            let (streams, read, routing) = self.mlp_hc.read_after_write_with_gate(
                &self.attn_hc,
                streams,
                &attention.values,
                gate,
                &self.moe,
            )?;
            (streams, read, Some(routing))
        } else {
            let (streams, read) =
                self.mlp_hc
                    .read_after_write(&self.attn_hc, streams, &attention.values, gate)?;
            (streams, read, None)
        };
        Ok(PreparedMoeRow {
            mixed: mlp_read.mixed,
            routing,
            streams,
            gate: mlp_read.gate.context("missing MLP injection gate")?,
        })
    }

    pub(super) fn finish_prepared(
        &self,
        cmd: &metal::CommandBufferRef,
        row: &PreparedMoeRow,
        experts: &super::MoeOutput,
    ) -> Result<MatrixView> {
        let branch = if self.mlp_hc.gpu.common_executor {
            experts.values_for_gpu()
        } else {
            experts.completed()?
        };
        let output = self
            .mlp_hc
            .encode_write(cmd, &row.streams, branch, &row.gate)?;
        #[cfg(test)]
        self.mlp_hc
            .gpu
            .hc_final_writes
            .set(self.mlp_hc.gpu.hc_final_writes.get() + 1);
        Ok(output)
    }

    pub(super) fn forward(
        &mut self,
        streams: &MatrixView,
        attend: impl FnOnce(&MatrixView) -> Result<super::FlashAttentionOutput>,
        next: Option<&mut super::FlashMoeRow<'a>>,
        depth: usize,
        defer_write: bool,
    ) -> Result<LayerOutput> {
        let read = self.attn_hc.read(streams, false)?;
        ensure!(read.gate.is_some(), "missing attention injection gate");
        let attention = attend(&read.mixed)?;
        self.forward_after_attention_read(streams, read, attention, next, depth, defer_write)
    }
    fn forward_after_attention_read(
        &mut self,
        streams: &MatrixView,
        read: super::FlashHcRead,
        attention: super::FlashAttentionOutput,
        next: Option<&mut super::FlashMoeRow<'a>>,
        depth: usize,
        defer_write: bool,
    ) -> Result<LayerOutput> {
        let gate = read
            .gate
            .as_ref()
            .context("missing attention injection gate")?;
        if self.moe.device_hot_gate_eligible(depth)
            && !self.mlp_hc.gpu.hc_workspace_enabled.get()
            && streams.dtype == FloatType::F32
        {
            let pending = self.mlp_hc.submit_after_write_with_gate(
                &self.attn_hc,
                streams,
                &attention.values,
                gate,
                &self.moe,
            )?;
            let (written, mlp_read, experts) = self.moe.with_pending_gate(pending)?;
            let gate = mlp_read.gate.as_ref().context("missing MLP injection")?;
            return self.final_write(&written, experts, gate, defer_write);
        }
        if self.moe.joint_gate_eligible(depth) {
            let (written, mlp_read, routing) = self.mlp_hc.read_after_write_with_gate(
                &self.attn_hc,
                streams,
                &attention.values,
                gate,
                &self.moe,
            )?;
            let gate = mlp_read.gate.as_ref().context("missing MLP injection")?;
            let experts = self.moe.with_completed_gate(&mlp_read.mixed, routing)?;
            return self.final_write(&written, experts, gate, defer_write);
        }
        let (written, mlp_read) =
            self.mlp_hc
                .read_after_write(&self.attn_hc, streams, &attention.values, gate)?;
        let gate = mlp_read.gate.as_ref().context("missing MLP injection")?;
        let experts = self.moe.lookahead(&mlp_read.mixed, next, depth)?;
        self.final_write(&written, experts, gate, defer_write)
    }
    fn final_write(
        &self,
        streams: &MatrixView,
        #[allow(unused_mut)] mut experts: super::MoeOutput,
        gate: &MatrixView,
        defer_write: bool,
    ) -> Result<LayerOutput> {
        let deferred = crate::flash_support::defer_moe_completion(
            self.mlp_hc.gpu.moe_deferred.get(),
            defer_write,
        );
        let branch = if deferred {
            experts.values_for_gpu()
        } else {
            experts.completed()?
        }
        .clone();
        #[cfg(test)]
        if deferred {
            let gpu = self.mlp_hc.gpu;
            gpu.moe_deferred_uses.set(gpu.moe_deferred_uses.get() + 1);
            experts.injected_failure = gpu.moe_deferred_fail.replace(false);
        }
        if defer_write {
            Ok(LayerOutput::Pending(DeferredHcWrite {
                streams: streams.clone(),
                branch,
                gate: gate.clone(),
                producer: deferred.then_some(experts),
            }))
        } else {
            let written = self.mlp_hc.write(streams, &branch, gate)?;
            #[cfg(test)]
            self.mlp_hc
                .gpu
                .hc_final_writes
                .set(self.mlp_hc.gpu.hc_final_writes.get() + 1);
            Ok(LayerOutput::Ready(written))
        }
    }
}

/// A complete non-PLE Flash layer with recurrent attention.
pub struct FlashGdnLayer<'a> {
    pub(super) residual: LayerResidual<'a>,
    attention: FlashGdn<'a>,
}
impl<'a> FlashGdnLayer<'a> {
    #[cfg(test)]
    pub(super) fn block_projection_count() -> usize {
        BLOCK_PROJECTION_BATCHES.with(std::cell::Cell::get)
    }
    #[cfg(test)]
    pub(super) fn temporal_batch_count() -> usize {
        TEMPORAL_BATCHES.with(std::cell::Cell::get)
    }

    pub(super) fn prepare_block_inputs(
        &self,
        rows: &[MatrixView],
    ) -> Result<Option<Vec<PreparedGdnRow>>> {
        if !self.attention.qkv.supports_verify_rows(rows.len())
            || !self.attention.z.supports_verify_rows(rows.len())
        {
            return Ok(None);
        }
        self.prepare_block_after(rows.len(), |_| Ok(rows))
            .map(|(_, prepared)| prepared)
    }

    pub(super) fn prepare_block_after<R: AsRef<[MatrixView]>>(
        &self,
        count: usize,
        before: impl FnOnce(&CommandBufferRef) -> Result<R>,
    ) -> Result<(R, Option<Vec<PreparedGdnRow>>)> {
        let gpu = self.attention.gpu;
        if !self.attention.qkv.supports_verify_rows(count)
            || !self.attention.z.supports_verify_rows(count)
        {
            let rows = self.attention.gpu.command(before)?;
            ensure!(rows.as_ref().len() == count, "incomplete preceding HC rows");
            return Ok((rows, None));
        }
        ensure!(
            (!gpu.hc_workspace_enabled.get() || gpu.common_executor),
            "GDN block inputs require owned HC rows"
        );
        #[cfg(test)]
        let _scope = crate::flash_profile::span("native.gdn.block_projections");
        let encode = |cmd: &CommandBufferRef| {
            let owner = before(cmd)?;
            let rows = owner.as_ref();
            ensure!(
                rows.len() == count && (1..=4).contains(&count),
                "invalid chained HC batch"
            );
            let (reads, projections, pending) = (|| -> Result<_> {
                let reads = rows
                    .iter()
                    .map(|row| self.residual.encode_attention_input(cmd, row))
                    .collect::<Result<Vec<_>>>()?;
                let packed = MatrixView::output(
                    gpu,
                    FloatType::F16,
                    rows.len() as u32,
                    2560,
                    Storage::StorageModePrivate,
                )?;
                let mut inputs = Vec::with_capacity(rows.len());
                for (i, read) in reads.iter().enumerate() {
                    let x = read.mixed();
                    let input = MatrixView::output(
                        gpu,
                        FloatType::F16,
                        1,
                        2560,
                        Storage::StorageModePrivate,
                    )?;
                    Metal::encode(
                        cmd,
                        &gpu.gdn.from[usize::from(x.dtype == FloatType::F32)],
                        &[&x.buffer, &input.buffer],
                        [x.layout.offset, x.layout.strides[1]],
                        MTLSize::new(80, 1, 1),
                        32,
                    );
                    let blit = cmd.new_blit_command_encoder();
                    blit.copy_from_buffer(&input.buffer, 0, &packed.buffer, i as u64 * 5120, 5120);
                    blit.end_encoding();
                    inputs.push(input);
                }
                let qkv =
                    self.attention
                        .qkv
                        .encode_rows(cmd, &packed, Storage::StorageModePrivate)?;
                let z = self
                    .attention
                    .z
                    .encode_rows(cmd, &packed, Storage::StorageModePrivate)?;
                if gpu.gdn.temporal.is_some()
                    && gpu.gdn_whole.get()
                    && rows.iter().all(|row| row.dtype == FloatType::F32)
                {
                    let pending = self.attention.encode_temporal(cmd, &inputs, &qkv, &z)?;
                    return Ok((reads, Vec::new(), pending));
                }
                let mut projections = Vec::with_capacity(rows.len());
                for (i, input) in inputs.into_iter().enumerate() {
                    let q = MatrixView::output(
                        gpu,
                        FloatType::F16,
                        1,
                        10240,
                        Storage::StorageModePrivate,
                    )?;
                    let gate = MatrixView::output(
                        gpu,
                        FloatType::F16,
                        1,
                        6144,
                        Storage::StorageModePrivate,
                    )?;
                    let blit = cmd.new_blit_command_encoder();
                    blit.copy_from_buffer(&qkv.buffer, i as u64 * 20480, &q.buffer, 0, 20480);
                    blit.copy_from_buffer(&z.buffer, i as u64 * 12288, &gate.buffer, 0, 12288);
                    blit.end_encoding();
                    projections.push(GdnProjections {
                        input,
                        qkv: q,
                        z: gate,
                    });
                }
                let mut pending: Vec<PendingGdn> = Vec::with_capacity(rows.len());
                for (i, (read, projected)) in reads.iter().zip(&projections).enumerate() {
                    let step = self.attention.encode_for_consumer_prepared(
                        cmd,
                        read.mixed(),
                        false,
                        rows[i].dtype == FloatType::F32,
                        Some(projected),
                        pending.last().map(|p| &p.next),
                    )?;
                    pending.push(step);
                }
                Ok((reads, projections, pending))
            })()?;
            let posts = if gpu.segment_enabled {
                reads
                    .iter()
                    .zip(&pending)
                    .zip(rows)
                    .map(|((read, pending), streams)| {
                        let (written, mlp) = self.residual.mlp_hc.encode_after_write(
                            cmd,
                            streams,
                            &pending.values,
                            read.injection_gate()?,
                        )?;
                        let routing = if self.residual.moe.joint_gate_eligible(0) {
                            let gate = self.residual.moe.encode_gate(gpu, cmd, mlp.mixed())?;
                            #[cfg(test)]
                            gpu.hc_gate_chain_pairs
                                .set(gpu.hc_gate_chain_pairs.get() + 1);
                            Some(gate)
                        } else {
                            None
                        };
                        #[cfg(test)]
                        gpu.hc_chain_pairs.set(gpu.hc_chain_pairs.get() + 1);
                        Ok(Some(EncodedMoeRow {
                            streams: written,
                            read: mlp,
                            routing,
                        }))
                    })
                    .collect::<Result<Vec<_>>>()?
            } else {
                (0..rows.len()).map(|_| None).collect()
            };
            Ok((owner, reads, projections, pending, posts))
        };
        let (rows, reads, _projections, pending, posts) = if gpu.segment_enabled {
            gpu.segment(encode)?.submit()?.retire()?
        } else {
            gpu.command(encode)?
        };
        #[cfg(test)]
        {
            BLOCK_PROJECTION_BATCHES.with(|count| count.set(count.get() + 1));
            gpu.hc_gdn_pairs.set(gpu.hc_gdn_pairs.get() + count);
        }
        let prepared = reads
            .into_iter()
            .zip(pending)
            .zip(posts)
            .map(|((read, pending), post)| {
                Ok(PreparedGdnRow {
                    read: read.finish(false)?,
                    pending,
                    post,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((rows, Some(prepared)))
    }

    pub(super) fn prepare_moe_projected(
        &mut self,
        streams: &MatrixView,
        row: PreparedGdnRow,
    ) -> Result<PreparedMoeRow> {
        let before = self.snapshot();
        let result = (|| {
            let attention = self.attention.finish(row.pending)?;
            if let Some(post) = row.post {
                post.finish()
            } else {
                self.residual
                    .prepare_after_attention(streams, row.read, attention)
            }
        })();
        if result.is_err() {
            self.restore(&before)?;
        }
        result
    }

    pub fn load(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        eps: f32,
        slots: usize,
    ) -> Result<Self> {
        let ple = format!("{prefix}.ple.");
        ensure!(
            !c.tensors
                .range(ple.clone()..)
                .next()
                .is_some_and(|(k, _)| k.starts_with(&ple)),
            "Flash layer requires PLE before attention"
        );
        Self::load_body(gpu, c, prefix, eps, slots)
    }
    pub(super) fn load_body(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        eps: f32,
        slots: usize,
    ) -> Result<Self> {
        Ok(Self {
            attention: FlashGdn::load(gpu, c, &format!("{prefix}.linear_attn"), eps)?,
            residual: LayerResidual::load(gpu, c, prefix, eps, slots)?,
        })
    }
    pub(super) fn recurrent(&self) -> Option<&MatrixView> {
        self.attention.state.frame.as_ref().map(|f| &f.recurrent)
    }
    pub fn snapshot(&self) -> FlashGdnState {
        self.attention.snapshot()
    }
    pub fn restore(&mut self, state: &FlashGdnState) -> Result<()> {
        self.attention.restore(state)
    }
    pub fn reset(&mut self) {
        self.attention.reset()
    }
    pub fn forward(&mut self, streams: &MatrixView) -> Result<MatrixView> {
        self.forward_lookahead(streams, None, 0)
    }
    pub(super) fn forward_lookahead(
        &mut self,
        streams: &MatrixView,
        next: Option<&mut super::FlashMoeRow<'a>>,
        depth: usize,
    ) -> Result<MatrixView> {
        self.forward_deferred(streams, next, depth, false)?
            .complete(self.attention.gpu)
    }
    fn route_chain_eligible(&self, streams: &MatrixView, depth: usize) -> bool {
        let gpu = self.attention.gpu;
        crate::flash_support::gdn_route_eligible(
            gpu.gdn_route_chain.get(),
            gpu.hc_gdn_chain.get(),
            self.residual
                .moe
                .device_hot_gate_eligible_for_encoding(depth),
            streams.dtype == FloatType::F32,
            gpu.hc_workspace_enabled.get(),
        )
    }

    fn forward_route_chain(
        &mut self,
        streams: &MatrixView,
        mut previous: Option<&mut DeferredHcWrite>,
        defer_write: bool,
    ) -> Result<LayerOutput> {
        let gpu = self.attention.gpu;
        ensure!(
            std::ptr::eq(gpu, self.residual.mlp_hc.gpu)
                && std::ptr::eq(gpu, self.residual.attn_hc.gpu),
            "GDN route requires one Metal owner"
        );
        #[cfg(test)]
        let scope = crate::flash_profile::span(if previous.is_some() {
            "native.hc_gdn.cross_layer"
        } else {
            "native.hc_gdn.joined"
        });
        let command = gpu.queue.new_command_buffer().to_owned();
        let (streams, read) = if let Some(write) = previous.as_deref() {
            self.residual.attn_hc.encode_after_write(
                &command,
                &write.streams,
                &write.branch,
                &write.gate,
            )?
        } else {
            (
                streams.clone(),
                self.residual
                    .attn_hc
                    .encode_read(&command, streams, false)?,
            )
        };
        let gdn = self
            .attention
            .encode_for_consumer(&command, read.mixed(), false, true)?;
        // These views stay GPU-only until the common GDN/P command completes.
        let read = read.finish(false)?;
        #[cfg(test)]
        drop(scope);
        let pending = self.residual.mlp_hc.submit_after_write_with_gate_on(
            command,
            &self.residual.attn_hc,
            &streams,
            &gdn.values,
            read.gate
                .as_ref()
                .context("missing attention injection gate")?,
            &self.residual.moe,
        )?;
        // Release the previous H/F owner before acquiring the next exclusive Hot lease.
        if let Some(write) = previous.as_mut() {
            write.finish_producer()?;
        }
        let (written, mlp_read, experts) = self.residual.moe.with_pending_gate(pending)?;
        self.attention.finish(gdn)?;
        #[cfg(test)]
        {
            gpu.gdn_route_uses.set(gpu.gdn_route_uses.get() + 1);
            if previous.is_some() {
                gpu.cross_layer_triples
                    .set(gpu.cross_layer_triples.get() + 1);
            } else {
                gpu.hc_gdn_pairs.set(gpu.hc_gdn_pairs.get() + 1);
            }
        }
        self.residual.final_write(
            &written,
            experts,
            mlp_read.gate.as_ref().context("missing MLP injection")?,
            defer_write,
        )
    }

    pub(super) fn prepare_moe(&mut self, streams: &MatrixView) -> Result<PreparedMoeRow> {
        ensure!(
            (!self.attention.gpu.hc_workspace_enabled.get() || self.attention.gpu.common_executor),
            "cohort rows require owned HC activations"
        );
        let before = self.snapshot();
        let result = (|| {
            let (read, attention) =
                self.attention
                    .forward_after_hc(&self.residual.attn_hc, streams, false)?;
            self.residual
                .prepare_after_attention(streams, read, attention)
        })();
        if result.is_err() {
            self.restore(&before)?;
        }
        result
    }

    pub(super) fn forward_deferred(
        &mut self,
        streams: &MatrixView,
        next: Option<&mut super::FlashMoeRow<'a>>,
        depth: usize,
        defer_write: bool,
    ) -> Result<LayerOutput> {
        let before = self.snapshot();
        let residual = (|| {
            if self.route_chain_eligible(streams, depth) {
                return self.forward_route_chain(streams, None, defer_write);
            }
            let (read, attention) =
                self.attention
                    .forward_after_hc(&self.residual.attn_hc, streams, false)?;
            self.residual.forward_after_attention_read(
                streams,
                read,
                attention,
                next,
                depth,
                defer_write,
            )
        })();
        if residual.is_err() {
            self.restore(&before)?;
        }
        residual
    }

    pub(super) fn forward_pending(
        &mut self,
        mut write: DeferredHcWrite,
        defer_write: bool,
    ) -> Result<LayerOutput> {
        let before = self.snapshot();
        let result = (|| {
            if self.route_chain_eligible(&write.streams, 0) {
                let streams = write.streams.clone();
                return self.forward_route_chain(&streams, Some(&mut write), defer_write);
            }
            let gpu = self.attention.gpu;
            #[cfg(test)]
            let _scope = crate::flash_profile::span("native.hc_gdn.cross_layer");
            let encoded = gpu.command(|cmd| {
                let (streams, read) = self.residual.attn_hc.encode_after_write(
                    cmd,
                    &write.streams,
                    &write.branch,
                    &write.gate,
                )?;
                let pending = self.attention.encode_for_consumer(
                    cmd,
                    read.mixed(),
                    false,
                    streams.dtype == FloatType::F32,
                )?;
                Ok((streams, read, pending))
            });
            #[cfg(test)]
            drop(_scope);
            let producer = write.finish_producer();
            let (streams, read, pending) = encoded?;
            producer?;
            let read = read.finish(false)?;
            let attention = self.attention.finish(pending)?;
            #[cfg(test)]
            gpu.cross_layer_triples
                .set(gpu.cross_layer_triples.get() + 1);
            self.residual.forward_after_attention_read(
                &streams,
                read,
                attention,
                None,
                0,
                defer_write,
            )
        })();
        if result.is_err() {
            self.restore(&before)?;
        }
        result
    }
}

impl FlashGdnState {
    pub(super) fn record_memory(
        &self,
        backings: &mut crate::gpu::memory::Backings,
        category: &'static str,
    ) -> Result<()> {
        if let Some(frame) = &self.frame {
            backings.record(category, &frame.history.buffer)?;
            backings.record(category, &frame.recurrent.buffer)?;
        }
        Ok(())
    }
}
impl FlashGdn<'_> {
    fn record_memory(&self, backings: &mut crate::gpu::memory::Backings) -> Result<()> {
        for projection in [&self.qkv, &self.z, &self.out] {
            projection.record_memory(backings)?;
        }
        for buffer in [
            &self.a,
            &self.b,
            &self.conv,
            &self.a_log,
            &self.dt_bias,
            &self.gain,
        ] {
            backings.record("fixed_weights", buffer)?;
        }
        if let Some(frame) = &self.state.frame {
            backings.record("gdn", &frame.history.buffer)?;
            backings.record("gdn", &frame.recurrent.buffer)?;
        }
        if let Some(pool) = &self.frame_pool {
            for frame in pool.borrow().iter() {
                backings.record("gdn_frame_arena", &frame.history.buffer)?;
                backings.record("gdn_frame_arena", &frame.recurrent.buffer)?;
            }
        }
        Ok(())
    }
}
impl LayerResidual<'_> {
    pub(super) fn record_memory(&self, backings: &mut crate::gpu::memory::Backings) -> Result<()> {
        self.attn_hc.record_memory(backings)?;
        self.mlp_hc.record_memory(backings)?;
        self.moe.record_memory(backings)
    }
    pub(super) fn release_idle_state(&mut self) -> Result<()> {
        self.moe.drain()?;
        self.attn_hc.release_workspace()?;
        self.mlp_hc.release_workspace()
    }
    pub(super) fn shrink(&mut self, slots: usize) -> Result<()> {
        self.moe.shrink(slots)
    }
}
impl FlashGdnLayer<'_> {
    pub(super) fn record_memory(&self, backings: &mut crate::gpu::memory::Backings) -> Result<()> {
        self.attention.record_memory(backings)?;
        self.residual.record_memory(backings)
    }
    pub(super) fn release_idle_state(&mut self) -> Result<()> {
        self.reset();
        for projection in [&self.attention.qkv, &self.attention.z, &self.attention.out] {
            projection.release_idle_scratch()?;
        }
        self.residual.release_idle_state()
    }
    pub(super) fn shrink(&mut self, slots: usize) -> Result<()> {
        self.residual.shrink(slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_layout::MatrixLayout;
    #[test]
    #[ignore = "physical Flash GDN layer rollback, reset and retained outputs"]
    fn downstream_failure_does_not_advance_recurrent_state() -> Result<()> {
        let path = std::env::var_os("MLXL3_GDN_MODEL").context("Flash model required")?;
        let c = crate::checkpoint::inspect(std::path::Path::new(&path))?;
        let gpu = Metal::new()?;
        let mut layer = FlashGdnLayer::load(&gpu, &c, "model.language_model.layers.0", 1e-6, 10)?;
        let other = FlashGdn::load(&gpu, &c, "model.language_model.layers.2.linear_attn", 1e-6)?;
        ensure!(layer.restore(&other.snapshot()).is_err());
        let row = |step: i32| -> Result<MatrixView> {
            let values = (0..10240)
                .map(|i| (((i * 17 + step * 83) % 257) - 128) as f32 / 237.)
                .collect::<Vec<_>>();
            gpu.f32_storage(&values)?.matrix(MatrixLayout {
                shape: [1, 10240],
                strides: [10240, 1],
                offset: 0,
            })
        };
        let x = row(0)?;
        let y = row(1)?;
        let initial = layer.snapshot();
        let first = layer.forward(&x)?;
        let prefix = layer.snapshot();
        let expected = layer.forward(&y)?;
        layer.restore(&prefix)?;
        let final_mixer = super::super::FlashHc::load(
            &gpu,
            &c,
            "model.language_model.hyper_connection_mixer",
            1e-6,
            false,
        )?;
        let good = std::mem::replace(&mut layer.residual.mlp_hc, final_mixer);
        let error = layer
            .forward(&y)
            .err()
            .context("missing injection was used")?;
        ensure!(error.to_string().contains("injection gate"));
        layer.residual.mlp_hc = good;
        let resumed = layer.forward(&y)?;
        layer.reset();
        let reset = layer.forward(&x)?;
        layer.restore(&initial)?;
        let fresh_y = layer.forward(&y)?;
        drop(layer);
        let expected = expected.read_output()?;
        ensure!(expected.len() == 10240 && expected != fresh_y.read_output()?);
        ensure!(resumed.read_output()? == expected && reset.read_output()? == first.read_output()?);
        Ok(())
    }
}

#[cfg(test)]
#[path = "hc_gdn_tests.rs"]
mod hc_gdn_tests;

#[cfg(test)]
#[path = "cross_layer_tests.rs"]
mod cross_layer_tests;

#[cfg(test)]
#[path = "gdn_uar_tests.rs"]
mod gdn_uar_tests;

#[cfg(test)]
#[path = "ar10_gdn_tests.rs"]
mod ar10_gdn_tests;

#[cfg(test)]
#[path = "ar20_gdn_tests.rs"]
mod ar20_gdn_tests;

#[cfg(test)]
#[path = "gdn_route_tests.rs"]
mod gdn_route_tests;

#[path = "gdn_temporal.rs"]
mod temporal;

#[cfg(test)]
#[test]
#[ignore = "physical Flash checkpoint; bounded arena with live external snapshots"]
fn physical_frame_arena_preserves_external_snapshots_and_fails_before_overwrite() -> Result<()> {
    let checkpoint =
        crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    let mut attention = FlashGdn::load(
        &gpu,
        &checkpoint,
        "model.language_model.layers.0.linear_attn",
        1e-6,
    )?;
    ensure!(
        attention.frame_pool.is_some(),
        "frame arena must be enabled"
    );
    let storage = gpu.f32_storage(
        &(0..2560)
            .map(|i| (i % 97) as f32 / 127. - 0.3)
            .collect::<Vec<_>>(),
    )?;
    let input = storage.matrix(crate::tensor_layout::MatrixLayout {
        shape: [1, 2560],
        strides: [2560, 1],
        offset: 0,
    })?;
    let mut kept = Vec::new();
    for _ in 0..crate::frame_arena::CAPACITY {
        attention.forward(&input, false)?;
        kept.push(attention.snapshot());
    }
    let first = kept[0].clone();
    let state = |s: &FlashGdnState| -> Result<_> {
        let frame = s.frame.as_ref().context("missing retained frame")?;
        Ok((
            frame.history.export_bytes(&gpu)?,
            frame.recurrent.export_bytes(&gpu)?,
        ))
    };
    let expected = state(&first)?;
    let last = state(&attention.snapshot())?;
    ensure!(attention.forward(&input, false).is_err());
    ensure!(state(&first)? == expected && state(&attention.snapshot())? == last);
    kept.clear();
    attention.restore(&first)?;
    attention.forward(&input, false)?;
    ensure!(state(&first)? == expected);
    let pool = attention.frame_pool.as_ref().unwrap().borrow();
    ensure!(
        pool.len() == crate::frame_arena::CAPACITY && pool.iter().any(|frame| frame.generation > 0)
    );
    Ok(())
}
