use super::CachedPipeline as ComputePipelineState;
use super::{MatrixView, Metal, checkpoint_float_weight, view::FloatType};
use crate::{
    checkpoint::Checkpoint,
    flash_ngrams::{EOS, Ngrams},
    tensor_layout::MatrixLayout,
};
use anyhow::{Result, ensure};
use metal::{Buffer, MTLResourceOptions as Storage, MTLSize, NSRange};
use std::{collections::BTreeMap, rc::Rc};

pub(super) struct Pipelines {
    key: ComputePipelineState,
    value: ComputePipelineState,
    route: [ComputePipelineState; 2],
    gated: ComputePipelineState,
    residual: [ComputePipelineState; 2],
}
impl Pipelines {
    pub(super) fn load(p: &impl Fn(&str) -> Result<ComputePipelineState>) -> Result<Self> {
        Ok(Self {
            key: p("ple_key")?,
            value: p("ple_value")?,
            route: [p("ple_route_half")?, p("ple_route_float")?],
            gated: p("ple_gated")?,
            residual: [p("ple_residual_half")?, p("ple_residual_float")?],
        })
    }
}

#[derive(Clone)]
pub struct FlashPleState {
    owner: Rc<()>,
    previous: [u32; 2],
    history: Option<MatrixView>,
}

pub struct FlashPle<'a> {
    gpu: &'a Metal,
    eps: f32,
    rows: Ngrams,
    key: Buffer,
    value: Buffer,
    nk: Buffer,
    nq: Buffer,
    nc: Buffer,
    conv: Buffer,
    history: Option<MatrixView>,
    owner: Rc<()>,
}

pub struct FlashPleOutput {
    pub values: MatrixView,
    pub read_bytes: u64,
    pub trace: Option<BTreeMap<&'static str, Vec<u32>>>,
}

impl<'a> FlashPle<'a> {
    pub fn load(gpu: &'a Metal, c: &Checkpoint, prefix: &str, eps: f32) -> Result<Self> {
        ensure!(
            c.model_type == "qwen4_exp" && eps.is_finite() && eps > 0.,
            "invalid Flash PLE configuration"
        );
        let weight = |name: &str, shape: &[usize], dtype| {
            checkpoint_float_weight(gpu, c, &format!("{prefix}.{name}.weight"), shape, dtype)
        };
        let rows = Ngrams::load(c, &format!("{prefix}.ple_embedding.ngram_embedding"))?;
        Ok(Self {
            gpu,
            rows,
            eps,
            key: weight("key_proj", &[10240, 2560], FloatType::F16)?,
            value: weight("value_proj", &[2560, 2560], FloatType::F16)?,
            nk: weight("norm_key", &[10240], FloatType::F32)?,
            nq: weight("norm_query", &[10240], FloatType::F32)?,
            nc: weight("norm_conv", &[10240], FloatType::F32)?,
            conv: weight("conv1d", &[10240, 1, 4], FloatType::F16)?,
            history: None,
            owner: Rc::new(()),
        })
    }
    pub fn snapshot(&self) -> FlashPleState {
        FlashPleState {
            owner: Rc::clone(&self.owner),
            previous: self.rows.previous,
            history: self.history.clone(),
        }
    }
    pub fn restore(&mut self, s: &FlashPleState) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.owner, &s.owner),
            "PLE state belongs to a different layer"
        );
        self.rows.previous = s.previous;
        self.history = s.history.clone();
        Ok(())
    }
    pub fn reset(&mut self) {
        self.history = None;
        self.rows.previous = [EOS; 2];
    }
    pub fn forward(
        &mut self,
        streams: &MatrixView,
        token: u32,
        trace: bool,
    ) -> Result<FlashPleOutput> {
        ensure!(
            streams.shape() == [1, 10240]
                && streams.buffer.device().registry_id() == self.gpu.device.registry_id(),
            "invalid Flash PLE input shape or device"
        );
        let previous = self.rows.previous;
        let before = self.rows.read_bytes;
        let words = self.rows.read(token)?;
        let result = self.compute(streams, &words, trace);
        if result.is_err() {
            self.rows.previous = previous;
        }
        let (next, mut output) = result?;
        self.history = Some(next);
        output.read_bytes = self.rows.read_bytes - before;
        Ok(output)
    }

    fn compute(
        &self,
        x: &MatrixView,
        words: &[u16],
        trace: bool,
    ) -> Result<(MatrixView, FlashPleOutput)> {
        let gpu = self.gpu;
        let p = &gpu.ple;
        let ix = usize::from(x.dtype == FloatType::F32);
        ensure!(
            p.route[ix].thread_execution_width() == 32
                && p.route[ix].max_total_threads_per_threadgroup() >= 640
                && p.key.thread_execution_width() == 32
                && p.key.max_total_threads_per_threadgroup() >= 256
                && p.value.thread_execution_width() == 32
                && p.value.max_total_threads_per_threadgroup() >= 128,
            "unsupported Flash PLE threadgroup"
        );
        let storage = if trace {
            Storage::StorageModeShared
        } else {
            Storage::StorageModePrivate
        };
        let matrix = |dtype, width| MatrixView::output(gpu, dtype, 1, width, storage);
        let groups = |n| MTLSize::new(n, 1, 1);
        let embedding = gpu.f16_storage(words)?.matrix(MatrixLayout {
            shape: [1, 2560],
            strides: [2560, 1],
            offset: 0,
        })?;
        let (next, out, stages) = gpu.command(|cmd| {
            let dense =
                |w: &Buffer, pipe: &ComputePipelineState, n: u32, bm: u32| -> Result<MatrixView> {
                    let out = matrix(FloatType::F16, n)?;
                    Metal::encode(
                        cmd,
                        pipe,
                        &[&embedding.buffer, w, &out.buffer],
                        [n],
                        groups(u64::from(n / (bm * 4))),
                        bm as usize * 32,
                    );
                    Ok(out)
                };
            let key_raw = dense(&self.key, &p.key, 10240, 8)?;
            let value = dense(&self.value, &p.value, 2560, 4)?;
            let key = gpu.encode_stream_norm(cmd, &key_raw, &self.nk, self.eps, storage)?;
            let query = gpu.encode_stream_norm(cmd, x, &self.nq, self.eps, storage)?;
            let raw = matrix(FloatType::F32, 4)?;
            let gate = matrix(FloatType::F32, 4)?;
            Metal::encode(
                cmd,
                &p.route[ix],
                &[&key.buffer, &query.buffer, &raw.buffer, &gate.buffer],
                [2560.0f32.sqrt()],
                groups(4),
                640,
            );
            let gated = matrix(FloatType::F32, 10240)?;
            Metal::encode(
                cmd,
                &p.gated,
                &[&value.buffer, &gate.buffer, &gated.buffer],
                [10240u32],
                groups(320),
                32,
            );
            let normed = gpu.encode_stream_norm(cmd, &gated, &self.nc, self.eps, storage)?;
            let previous = if let Some(h) = &self.history {
                h.clone()
            } else {
                let h = matrix(FloatType::F32, 9 * 10240)?;
                let blit = cmd.new_blit_command_encoder();
                blit.fill_buffer(&h.buffer, NSRange::new(0, h.buffer.length()), 0);
                blit.end_encoding();
                h
            };
            let next = matrix(FloatType::F32, 9 * 10240)?;
            let raw_conv = matrix(FloatType::F32, 10240)?;
            let convolution = matrix(FloatType::F32, 10240)?;
            let out =
                MatrixView::output(gpu, FloatType::F32, 1, 10240, Storage::StorageModeShared)?;
            Metal::encode(
                cmd,
                &p.residual[ix],
                &[
                    &x.buffer,
                    &gated.buffer,
                    &normed.buffer,
                    &previous.buffer,
                    &self.conv,
                    &next.buffer,
                    &raw_conv.buffer,
                    &convolution.buffer,
                    &out.buffer,
                ],
                [x.layout.offset, x.layout.strides[1]],
                groups(320),
                32,
            );
            let stages = trace.then_some([
                ("embedding", embedding),
                ("key", key),
                ("query", query),
                ("raw", raw),
                ("gate", gate),
                ("gated", gated),
                ("normed", normed),
                ("conv_raw", raw_conv),
                ("convolution", convolution),
            ]);
            Ok((next, out, stages))
        })?;
        let stages = stages
            .map(|list| -> Result<BTreeMap<_, _>> {
                let mut stages = list
                    .into_iter()
                    .map(|(k, v)| Ok((k, v.read_output()?)))
                    .collect::<Result<BTreeMap<_, _>>>()?;
                stages.insert("history", next.read_output()?);
                Ok(stages)
            })
            .transpose()?;
        Ok((
            next,
            FlashPleOutput {
                values: out,
                read_bytes: 0,
                trace: stages,
            },
        ))
    }
}

pub struct FlashPleLayer<'a> {
    ple: FlashPle<'a>,
    pub(super) body: super::FlashGdnLayer<'a>,
}
#[derive(Clone)]
pub struct FlashPleLayerState(FlashPleState, super::FlashGdnState);
impl<'a> FlashPleLayer<'a> {
    pub fn load(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        eps: f32,
        slots: usize,
    ) -> Result<Self> {
        Ok(Self {
            ple: FlashPle::load(gpu, c, &format!("{prefix}.ple"), eps)?,
            body: super::FlashGdnLayer::load_body(gpu, c, prefix, eps, slots)?,
        })
    }
    pub(super) fn recurrent(&self) -> Option<&MatrixView> {
        self.body.recurrent()
    }
    pub fn snapshot(&self) -> FlashPleLayerState {
        FlashPleLayerState(self.ple.snapshot(), self.body.snapshot())
    }
    pub fn restore(&mut self, state: &FlashPleLayerState) -> Result<()> {
        self.ple.restore(&state.0)?;
        self.body.restore(&state.1)
    }
    pub fn reset(&mut self) {
        self.ple.reset();
        self.body.reset();
    }
    pub fn forward(&mut self, x: &MatrixView, token: u32) -> Result<MatrixView> {
        self.forward_lookahead(x, token, None, 0)
    }
    pub(super) fn forward_lookahead(
        &mut self,
        x: &MatrixView,
        token: u32,
        next: Option<&mut super::FlashMoeRow<'a>>,
        depth: usize,
    ) -> Result<MatrixView> {
        self.forward_deferred(x, token, next, depth, false)?
            .complete(self.ple.gpu)
    }
    pub(super) fn forward_deferred(
        &mut self,
        x: &MatrixView,
        token: u32,
        next: Option<&mut super::FlashMoeRow<'a>>,
        depth: usize,
        defer_write: bool,
    ) -> Result<super::gdn::LayerOutput> {
        let before = self.ple.snapshot();
        let streams = self.ple.forward(x, token, false)?;
        let result = self
            .body
            .forward_deferred(&streams.values, next, depth, defer_write);
        if result.is_err() {
            self.ple.restore(&before)?;
        }
        result
    }
}

impl FlashPleLayerState {
    pub(super) fn record_memory(
        &self,
        backings: &mut crate::gpu::memory::Backings,
        category: &'static str,
    ) -> Result<()> {
        if let Some(history) = &self.0.history {
            backings.record(category, &history.buffer)?;
        }
        self.1.record_memory(backings, category)
    }
}
impl FlashPleLayer<'_> {
    pub(super) fn record_memory(&self, backings: &mut crate::gpu::memory::Backings) -> Result<()> {
        for buffer in [
            &self.ple.key,
            &self.ple.value,
            &self.ple.nk,
            &self.ple.nq,
            &self.ple.nc,
            &self.ple.conv,
        ] {
            backings.record("fixed_weights", buffer)?;
        }
        if let Some(history) = &self.ple.history {
            backings.record("ple", &history.buffer)?;
        }
        backings.ple_rows += self.ple.rows.cache_bytes();
        self.body.record_memory(backings)
    }
    pub(super) fn release_idle_state(&mut self) -> Result<()> {
        self.reset();
        self.body.release_idle_state()
    }
    pub(super) fn shrink(&mut self, slots: usize) -> Result<()> {
        self.body.shrink(slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    #[test]
    #[ignore = "physical PLE/GDN integration, downstream SSD failure, reset and replay"]
    fn expert_io_failure_rolls_back_both_ple_histories() -> Result<()> {
        let path = std::env::var_os("MLXL3_PLE_MODEL").context("missing Flash checkpoint")?;
        let mut c = crate::checkpoint::inspect(std::path::Path::new(&path))?;
        let gpu = Metal::new()?;
        let prefix = "model.language_model.layers.1";
        let mut layer = FlashPleLayer::load(&gpu, &c, prefix, 1e-6, 10)?;
        let row = |phase: i32| -> Result<_> {
            let data = (0..10240)
                .map(|i| ((((i * 17 + phase * 11) % 257) - 128) as f32) / 87.)
                .collect::<Vec<_>>();
            gpu.f32_storage(&data)?.matrix(MatrixLayout {
                shape: [1, 10240],
                strides: [0, 1],
                offset: 0,
            })
        };
        let x = row(0)?;
        let y = row(1)?;
        let first = layer.forward(&x, EOS)?;
        let saved = layer.snapshot();
        let expected = layer.forward(&y, 42)?;
        layer.restore(&saved)?;
        let records = format!("{prefix}.mlp.experts.");
        let mut fields = 0;
        for (_, tensor) in c
            .tensors
            .range_mut(records.clone()..)
            .take_while(|(k, _)| k.starts_with(&records))
        {
            for offset in &mut tensor.data_offsets {
                *offset += 1 << 44;
            }
            fields += 1;
        }
        ensure!(fields == 512 * 12, "expected all routed expert fields");
        let broken = super::super::FlashGdnLayer::load_body(&gpu, &c, prefix, 1e-6, 10)?;
        let good = std::mem::replace(&mut layer.body, broken);
        let error = layer
            .forward(&y, 42)
            .err()
            .context("unreadable expert was used")?;
        ensure!(
            error
                .to_string()
                .contains("checkpoint was truncated after inspection"),
            "unexpected error: {error:#}"
        );
        layer.body = good;
        ensure!(layer.ple.rows.previous == saved.0.previous);
        let resumed = layer.forward(&y, 42)?;
        layer.reset();
        let reset = layer.forward(&x, EOS)?;
        drop(layer);
        ensure!(
            resumed.read_output()? == expected.read_output()?
                && reset.read_output()? == first.read_output()?
        );
        Ok(())
    }
}
