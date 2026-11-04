use super::{FlashProjection, MatrixView, Metal, view::FloatType};
use crate::checkpoint::Checkpoint;
use anyhow::{Context, Result, ensure};
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, ComputePipelineState, MTLCommandBufferStatus,
    MTLResourceOptions as Storage, MTLSize, NSRange,
};
use std::{collections::BTreeMap, rc::Rc};

pub(super) struct Pipelines {
    from: [ComputePipelineState; 2],
    dense: ComputePipelineState,
    prepare: ComputePipelineState,
    step: ComputePipelineState,
    norm: ComputePipelineState,
    gate: ComputePipelineState,
    pub(super) to_float: ComputePipelineState,
}
impl Pipelines {
    pub(super) fn load(p: &impl Fn(&str) -> Result<ComputePipelineState>) -> Result<Self> {
        Ok(Self {
            from: [p("gdn_from_half")?, p("gdn_from_float")?],
            dense: p("gdn_dense_gate")?,
            prepare: p("gdn_prepare_row")?,
            step: p("gdn_step_row")?,
            norm: p("gdn_norm_row")?,
            gate: p("gdn_gate_row")?,
            to_float: p("gdn_to_float")?,
        })
    }
}

#[derive(Clone)]
struct Frame {
    history: MatrixView,
    recurrent: MatrixView,
}
/// The retained state belongs to one checkpoint layer and may be restored after a divergent branch.
#[derive(Clone)]
pub struct FlashGdnState {
    owner: Rc<()>,
    frame: Option<Frame>,
}

// The standard command buffer retains encoded resources through completion.
// Pending keeps the state/output owners and the publication-status guard.
struct PendingGdn {
    command: CommandBuffer,
    next: Frame,
    values: MatrixView,
    steps: Option<[(&'static str, MatrixView); 11]>,
}

pub type FlashGdnOutput = super::FlashAttentionOutput;
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
        Ok(Self {
            gpu,
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
        let (value_groups, state_words) =
            crate::contracts::packed_gdn_grid(48, 128).context("invalid Flash GDN state extent")?;
        let frame = || -> Result<Frame> {
            #[cfg(test)]
            let _scope = crate::flash_profile::span("native.gdn.frame");
            Ok(Frame {
                history: half(30720)?,
                recurrent: matrix(FloatType::F32, state_words)?,
            })
        };
        let g = |x, y, z| MTLSize::new(x, y, z);
        let prev = match &self.state.frame {
            Some(previous) => previous.clone(),
            None => {
                let zero = frame()?;
                let blit = cmd.new_blit_command_encoder();
                for v in [&zero.history, &zero.recurrent] {
                    blit.fill_buffer(&v.buffer, NSRange::new(0, v.buffer.length()), 0);
                }
                blit.end_encoding();
                zero
            }
        };
        let next = frame()?;
        let input = half(2560)?;
        Metal::encode(
            cmd,
            &p.from[usize::from(x.dtype == FloatType::F32)],
            &[&x.buffer, &input.buffer],
            [x.layout.offset, x.layout.strides[1]],
            g(80, 1, 1),
            32,
        );
        let qkv = self.qkv.encode_row(cmd, &input, storage)?;
        let z = self.z.encode_row(cmd, &input, storage)?;
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
        let a = dense(&self.a)?;
        let b = dense(&self.b)?;
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
        let normed = matrix(FloatType::F32, 6144)?;
        Metal::encode(
            cmd,
            &p.norm,
            &[&delta.buffer, &self.gain, &normed.buffer],
            [self.eps.to_bits()],
            g(48, 1, 1),
            32,
        );
        let gated = half(6144)?;
        Metal::encode(
            cmd,
            &p.gate,
            &[&normed.buffer, &z.buffer, &gated.buffer],
            [6144u32],
            g(192, 1, 1),
            32,
        );
        let out_half = self.out.encode_row(cmd, &gated, storage)?;
        let values = MatrixView::output(gpu, FloatType::F32, 1, 2560, Storage::StorageModeShared)?;
        Metal::encode(
            cmd,
            &p.to_float,
            &[&out_half.buffer, &values.buffer],
            [2560u32],
            g(80, 1, 1),
            32,
        );
        let steps = trace.then_some([
            ("qkv", qkv),
            ("z", z),
            ("a", a),
            ("b", b),
            ("conv", conv),
            ("q", q),
            ("k", k),
            ("v", v),
            ("delta", delta),
            ("normed", normed),
            ("gated", gated),
        ]);
        Ok(PendingGdn {
            command: cmd.to_owned(),
            next,
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
            let pending = self.encode(cmd, read.mixed(), false)?;
            Ok((read, pending))
        })?;
        let read = read.finish(false)?;
        let attention = self.finish(pending)?;
        #[cfg(test)]
        gpu.hc_gdn_pairs.set(gpu.hc_gdn_pairs.get() + 1);
        Ok((read, attention))
    }
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
    pub(super) fn forward(
        &mut self,
        streams: &MatrixView,
        attend: impl FnOnce(&MatrixView) -> Result<super::FlashAttentionOutput>,
        next: Option<&mut super::FlashMoeRow<'a>>,
        depth: usize,
    ) -> Result<MatrixView> {
        let read = self.attn_hc.read(streams, false)?;
        ensure!(read.gate.is_some(), "missing attention injection gate");
        let attention = attend(&read.mixed)?;
        self.forward_after_attention_read(streams, read, attention, next, depth)
    }
    fn forward_after_attention_read(
        &mut self,
        streams: &MatrixView,
        read: super::FlashHcRead,
        attention: super::FlashAttentionOutput,
        next: Option<&mut super::FlashMoeRow<'a>>,
        depth: usize,
    ) -> Result<MatrixView> {
        let gate = read
            .gate
            .as_ref()
            .context("missing attention injection gate")?;
        if self.moe.joint_gate_eligible(depth) {
            let (written, mlp_read, routing) = self.mlp_hc.read_after_write_with_gate(
                &self.attn_hc,
                streams,
                &attention.values,
                gate,
                &self.moe,
            )?;
            let experts = self.moe.with_completed_gate(&mlp_read.mixed, routing)?;
            return self.mlp_hc.write(
                &written,
                experts.completed()?,
                mlp_read.gate.as_ref().context("missing MLP injection")?,
            );
        }
        let (written, mlp_read) =
            self.mlp_hc
                .read_after_write(&self.attn_hc, streams, &attention.values, gate)?;
        Ok(self
            .mlp_hc
            .forward_moe_read(&written, &mut self.moe, mlp_read, next, depth)?
            .0)
    }
}

/// A complete non-PLE Flash layer with recurrent attention.
pub struct FlashGdnLayer<'a> {
    pub(super) residual: LayerResidual<'a>,
    attention: FlashGdn<'a>,
}
impl<'a> FlashGdnLayer<'a> {
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
        let before = self.snapshot();
        let residual = (|| {
            let (read, attention) =
                self.attention
                    .forward_after_hc(&self.residual.attn_hc, streams, false)?;
            self.residual
                .forward_after_attention_read(streams, read, attention, next, depth)
        })();
        if residual.is_err() {
            self.restore(&before)?;
        }
        residual
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
