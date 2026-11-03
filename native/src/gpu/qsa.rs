use super::{FlashAttentionOutput, FlashProjection, MatrixView, Metal, view::FloatType};
use crate::{checkpoint::Checkpoint, flash_support::qsa_append};
use anyhow::{Context, Result, ensure};
use metal::{Buffer, ComputePipelineState, MTLResourceOptions as Storage, MTLSize};
use std::{cell::Cell, collections::BTreeMap, rc::Rc};

pub(super) struct Pipelines {
    norm: ComputePipelineState,
    freq: ComputePipelineState,
    rope: ComputePipelineState,
    attention: ComputePipelineState,
    gate: ComputePipelineState,
}
impl Pipelines {
    pub(super) fn load(p: &impl Fn(&str) -> Result<ComputePipelineState>) -> Result<Self> {
        Ok(Self {
            norm: p("qsa_norm")?,
            freq: p("qsa_freq")?,
            rope: p("qsa_rope")?,
            attention: p("qsa_attention")?,
            gate: p("qsa_gate")?,
        })
    }
}
struct Slab {
    keys: Buffer,
    values: Buffer,
    published: Cell<u32>,
}
impl Slab {
    fn new(gpu: &Metal, capacity: u32) -> Result<Rc<Self>> {
        let alloc = || {
            super::buffer(
                &gpu.device,
                u64::from(capacity) * 1024,
                Storage::StorageModePrivate,
            )
        };
        Ok(Rc::new(Self {
            keys: alloc()?,
            values: alloc()?,
            published: Cell::new(0),
        }))
    }
}

#[derive(Clone)]
pub struct FlashQsaState {
    owner: Rc<()>,
    slab: Rc<Slab>,
    offset: u32,
}
pub struct FlashQsa<'a> {
    gpu: &'a Metal,
    q: FlashProjection<'a>,
    k: FlashProjection<'a>,
    v: FlashProjection<'a>,
    out: FlashProjection<'a>,
    q_norm: Buffer,
    k_norm: Buffer,
    freq: Buffer,
    eps: f32,
    capacity: u32,
    state: FlashQsaState,
}
impl<'a> FlashQsa<'a> {
    /// Below the sparse selection threshold every past position contributes, including the current token.
    pub fn load(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        eps: f32,
        theta: f32,
        capacity: u32,
    ) -> Result<Self> {
        ensure!(
            eps.is_finite() && eps > 0. && theta.is_finite() && theta > 0.,
            "invalid Flash QSA epsilon or theta"
        );
        ensure!(
            (1..=512).contains(&capacity),
            "dense Flash QSA context must be within 1..=512"
        );
        ensure!(
            gpu.qsa.attention.thread_execution_width() == 32
                && gpu.qsa.attention.max_total_threads_per_threadgroup() >= 1024
                && gpu.qsa.norm.thread_execution_width() == 32,
            "unsupported Flash QSA threadgroup"
        );
        let proj = |name: &str, dims| -> Result<_> {
            let p = FlashProjection::load(gpu, c, &format!("{prefix}.{name}"))?;
            ensure!(
                p.dimensions() == dims,
                "invalid Flash QSA projection {name}"
            );
            Ok(p)
        };
        let weight = |name| {
            super::checkpoint_float_weight(
                gpu,
                c,
                &format!("{prefix}.{name}.weight"),
                &[256],
                FloatType::F32,
            )
        };
        let freq = super::buffer(&gpu.device, 128, Storage::StorageModePrivate)?;
        gpu.command(|cmd| {
            Metal::encode(
                cmd,
                &gpu.qsa.freq,
                &[&freq],
                [theta],
                MTLSize::new(1, 1, 1),
                32,
            );
            Ok(())
        })?;
        Ok(Self {
            gpu,
            q: proj("q_proj", [2560, 12288])?,
            k: proj("k_proj", [2560, 512])?,
            v: proj("v_proj", [2560, 512])?,
            out: proj("o_proj", [6144, 2560])?,
            q_norm: weight("q_norm")?,
            k_norm: weight("k_norm")?,
            freq,
            eps,
            capacity,
            state: FlashQsaState {
                owner: Rc::new(()),
                slab: Slab::new(gpu, capacity)?,
                offset: 0,
            },
        })
    }
    pub fn snapshot(&self) -> FlashQsaState {
        self.state.clone()
    }
    pub fn restore(&mut self, s: &FlashQsaState) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&s.owner, &self.state.owner),
            "QSA state belongs to a different layer"
        );
        self.state = s.clone();
        Ok(())
    }
    pub fn reset(&mut self) {
        self.state.offset = 0;
    }

    pub fn forward(&mut self, x: &MatrixView, trace: bool) -> Result<FlashAttentionOutput> {
        let gpu = self.gpu;
        ensure!(
            x.shape() == [1, 2560] && x.buffer.device().registry_id() == gpu.device.registry_id(),
            "invalid Flash QSA input shape or device"
        );
        let offset = self.state.offset;
        let prior = &self.state.slab;
        let (at, fork) = qsa_append(offset, prior.published.get(), self.capacity)
            .context("Flash QSA dense context exhausted")?;
        let slab = if fork {
            Slab::new(gpu, self.capacity)?
        } else {
            prior.clone()
        };
        let storage = if trace {
            Storage::StorageModeShared
        } else {
            Storage::StorageModePrivate
        };
        let half = |n| MatrixView::output(gpu, FloatType::F16, 1, n, storage);
        let groups = |n| MTLSize::new(n, 1, 1);
        let (values, steps) = gpu.command(|cmd| {
            let qg = self.q.encode_row(cmd, x, storage)?;
            let k = self.k.encode_row(cmd, x, storage)?;
            let v = self.v.encode_row(cmd, x, storage)?;
            let prepare =
                |input: &MatrixView, gain: &Buffer, heads: u32, stride: u32| -> Result<_> {
                    let norm = half(heads * 256)?;
                    Metal::encode(
                        cmd,
                        &gpu.qsa.norm,
                        &[&input.buffer, gain, &norm.buffer],
                        [stride, self.eps.to_bits()],
                        groups(u64::from(heads)),
                        32,
                    );
                    let rope = half(heads * 256)?;
                    Metal::encode(
                        cmd,
                        &gpu.qsa.rope,
                        &[&norm.buffer, &self.freq, &rope.buffer],
                        [offset, heads],
                        groups(u64::from(heads) * 7),
                        32,
                    );
                    Ok((norm, rope))
                };
            let (qn, qr) = prepare(&qg, &self.q_norm, 24, 512)?;
            let (kn, kr) = prepare(&k, &self.k_norm, 2, 256)?;
            let blit = cmd.new_blit_command_encoder();
            if fork && at > 0 {
                blit.copy_from_buffer(&prior.keys, 0, &slab.keys, 0, at);
                blit.copy_from_buffer(&prior.values, 0, &slab.values, 0, at);
            }
            blit.copy_from_buffer(&kr.buffer, 0, &slab.keys, at, 1024);
            blit.copy_from_buffer(&v.buffer, 0, &slab.values, at, 1024);
            blit.end_encoding();
            let attention = half(6144)?;
            Metal::encode(
                cmd,
                &gpu.qsa.attention,
                &[&qr.buffer, &slab.keys, &slab.values, &attention.buffer],
                [offset + 1],
                groups(24),
                1024,
            );
            let gated = half(6144)?;
            Metal::encode(
                cmd,
                &gpu.qsa.gate,
                &[&attention.buffer, &qg.buffer, &gated.buffer],
                [256u32],
                groups(192),
                32,
            );
            let out = self
                .out
                .encode_row(cmd, &gated, Storage::StorageModeShared)?;
            let values = if x.dtype == FloatType::F32 {
                let f =
                    MatrixView::output(gpu, FloatType::F32, 1, 2560, Storage::StorageModeShared)?;
                Metal::encode(
                    cmd,
                    &gpu.gdn.to_float,
                    &[&out.buffer, &f.buffer],
                    [2560u32],
                    groups(80),
                    32,
                );
                f
            } else {
                out
            };
            Ok((
                values,
                trace.then_some([
                    ("q_proj", qg),
                    ("q_norm", qn),
                    ("q_rope", qr),
                    ("k_proj", k),
                    ("k_norm", kn),
                    ("k_rope", kr),
                    ("v_proj", v),
                    ("gated", gated),
                ]),
            ))
        })?;
        let trace = steps
            .map(|items| {
                items
                    .into_iter()
                    .map(|(name, v)| Ok((name, v.read_output()?)))
                    .collect::<Result<BTreeMap<_, _>>>()
            })
            .transpose()?;
        slab.published.set(offset + 1);
        self.state.slab = slab;
        self.state.offset = offset + 1;
        Ok(FlashAttentionOutput { values, trace })
    }
}

pub struct FlashQsaLayer<'a> {
    attention: FlashQsa<'a>,
    residual: super::gdn::LayerResidual<'a>,
}
impl<'a> FlashQsaLayer<'a> {
    pub fn load(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        eps: f32,
        theta: f32,
        capacity: u32,
        slots: usize,
    ) -> Result<Self> {
        Ok(Self {
            attention: FlashQsa::load(
                gpu,
                c,
                &format!("{prefix}.self_attn"),
                eps,
                theta,
                capacity,
            )?,
            residual: super::gdn::LayerResidual::load(gpu, c, prefix, eps, slots)?,
        })
    }
    pub fn snapshot(&self) -> FlashQsaState {
        self.attention.snapshot()
    }
    pub fn restore(&mut self, s: &FlashQsaState) -> Result<()> {
        self.attention.restore(s)
    }
    pub fn reset(&mut self) {
        self.attention.reset();
    }
    pub fn forward(&mut self, streams: &MatrixView) -> Result<MatrixView> {
        let before = self.snapshot();
        let out = self
            .residual
            .forward(streams, |x| self.attention.forward(x, false));
        if out.is_err() {
            self.restore(&before)?;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_layout::MatrixLayout;

    #[test]
    #[ignore = "physical Flash QSA retained branches and full context"]
    fn divergent_suffix_preserves_both_snapshots_and_rejects_full_context() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_QSA_MODEL")?))?;
        let gpu = Metal::new()?;
        let layer = |prefix| FlashQsa::load(&gpu, &c, prefix, 1e-6, 1e7, 3);
        let mut a = layer("model.language_model.layers.15.self_attn")?;
        let other = layer("model.language_model.layers.35.self_attn")?;
        let row = |t| {
            gpu.f32_storage(
                &(0..2560)
                    .map(|i| ((i * 17 + t * 83) % 257 - 128) as f32 / 237.)
                    .collect::<Vec<_>>(),
            )?
            .matrix(MatrixLayout {
                shape: [1, 2560],
                strides: [2560, 1],
                offset: 0,
            })
        };
        let (x, y, z) = (row(0)?, row(1)?, row(2)?);
        let first = a.forward(&x, false)?.values;
        let prefix = a.snapshot();
        a.forward(&y, false)?;
        let xy = a.snapshot();
        let xyz = a.forward(&z, false)?.values;
        ensure!(
            a.forward(&x, false)
                .err()
                .context("accepted full context")?
                .to_string()
                .contains("exhausted")
        );
        ensure!(a.restore(&other.snapshot()).is_err());
        a.restore(&prefix)?;
        a.forward(&z, false)?;
        let xzy = a.forward(&y, false)?.values;
        a.restore(&xy)?;
        ensure!(a.forward(&z, false)?.values.read_output()? == xyz.read_output()?);
        a.reset();
        ensure!(a.forward(&x, false)?.values.read_output()? == first.read_output()?);
        a.forward(&z, false)?;
        ensure!(
            a.forward(&y, false)?.values.read_output()? == xzy.read_output()?
                && xzy.read_output()? != xyz.read_output()?
        );
        Ok(())
    }
}
