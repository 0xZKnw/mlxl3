//! Flash Next autoregressive session with checkpoint-backed expert and embedding reads.
use super::{
    FlashGdnLayer, FlashGdnState, FlashHc, FlashPleLayer, FlashPleLayerState, FlashProjection,
    FlashQsaLayer, FlashQsaState, MatrixView, Metal,
};
use crate::{
    checkpoint::{Checkpoint, TensorInfo},
    flash_config::{FlashConfig, layer_slots},
    flash_support::row_offset,
    tensor_layout::MatrixLayout,
};
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use std::{cell::RefCell, os::unix::fs::FileExt, rc::Rc};

#[path = "mtp.rs"]
mod mtp;
pub use mtp::{FlashMtpHead, FlashMtpSession};

#[path = "flash_verify.rs"]
mod verify;
pub use verify::FlashVerification;

pub(super) enum Layer<'a> {
    Gdn(Box<FlashGdnLayer<'a>>),
    Ple(Box<FlashPleLayer<'a>>),
    Qsa(Box<FlashQsaLayer<'a>>),
}
#[derive(Clone)]
enum State {
    Gdn(FlashGdnState),
    Ple(FlashPleLayerState),
    Qsa(FlashQsaState),
}
impl<'a> Layer<'a> {
    fn forward(
        &mut self,
        x: &MatrixView,
        token: u32,
        next: Option<&mut super::FlashMoeRow<'a>>,
        depth: usize,
        defer_write: bool,
    ) -> Result<super::gdn::LayerOutput> {
        match self {
            Self::Gdn(a) => a.forward_deferred(x, next, depth, defer_write),
            Self::Ple(a) => a.forward_deferred(x, token, next, depth, defer_write),
            Self::Qsa(a) => a.forward_deferred(x, next, depth, defer_write),
        }
    }
    fn moe(&mut self) -> &mut super::FlashMoeRow<'a> {
        match self {
            Self::Gdn(a) => &mut a.residual.moe,
            Self::Ple(a) => &mut a.body.residual.moe,
            Self::Qsa(a) => &mut a.residual.moe,
        }
    }
    fn prepare_moe(&mut self, row: &MatrixView, token: u32) -> Result<super::gdn::PreparedMoeRow> {
        match self {
            Self::Gdn(layer) => layer.prepare_moe(row),
            Self::Ple(layer) => layer.prepare_moe(row, token),
            Self::Qsa(layer) => layer.prepare_moe(row),
        }
    }
    fn finish_prepared(
        &self,
        row: super::gdn::PreparedMoeRow,
        experts: super::MoeOutput,
    ) -> Result<MatrixView> {
        match self {
            Self::Gdn(layer) => layer.residual.finish_prepared(row, experts),
            Self::Ple(layer) => layer.body.residual.finish_prepared(row, experts),
            Self::Qsa(layer) => layer.residual.finish_prepared(row, experts),
        }
    }
    fn snapshot(&self) -> State {
        match self {
            Self::Gdn(a) => State::Gdn(a.snapshot()),
            Self::Ple(a) => State::Ple(a.snapshot()),
            Self::Qsa(a) => State::Qsa(a.snapshot()),
        }
    }
    fn restore(&mut self, s: &State) -> Result<()> {
        match (self, s) {
            (Self::Gdn(a), State::Gdn(s)) => a.restore(s),
            (Self::Ple(a), State::Ple(s)) => a.restore(s),
            (Self::Qsa(a), State::Qsa(s)) => a.restore(s),
            _ => anyhow::bail!("invalid Flash layer snapshot"),
        }
    }
    fn recurrent(&self) -> Option<&MatrixView> {
        match self {
            Self::Gdn(l) => l.recurrent(),
            Self::Ple(l) => l.recurrent(),
            Self::Qsa(_) => None,
        }
    }
    pub(super) fn record_memory(&self, backings: &mut crate::gpu::memory::Backings) -> Result<()> {
        match self {
            Self::Gdn(layer) => layer.record_memory(backings),
            Self::Ple(layer) => layer.record_memory(backings),
            Self::Qsa(layer) => layer.record_memory(backings),
        }
    }
    pub(super) fn release_idle_state(&mut self) -> Result<()> {
        match self {
            Self::Gdn(layer) => layer.release_idle_state(),
            Self::Ple(layer) => layer.release_idle_state(),
            Self::Qsa(layer) => layer.release_idle_state(),
        }
    }
    pub(super) fn shrink(&mut self, slots: usize) -> Result<()> {
        match self {
            Self::Gdn(layer) => layer.shrink(slots),
            Self::Ple(layer) => layer.shrink(slots),
            Self::Qsa(layer) => layer.shrink(slots),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Gdn(a) => a.reset(),
            Self::Ple(a) => a.reset(),
            Self::Qsa(a) => a.reset(),
        }
    }
}

#[derive(Clone)]
pub struct FlashSnapshot {
    owner: Rc<()>,
    offset: u32,
    layers: Vec<State>,
    hidden: Option<MatrixView>,
}

impl FlashSnapshot {
    pub(super) fn record_memory(&self, backings: &mut crate::gpu::memory::Backings) -> Result<()> {
        if let Some(hidden) = &self.hidden {
            backings.record("prompt_cache", &hidden.buffer)?;
        }
        for layer in &self.layers {
            match layer {
                State::Gdn(state) => state.record_memory(backings, "prompt_cache")?,
                State::Ple(state) => state.record_memory(backings, "prompt_cache")?,
                State::Qsa(state) => state.record_memory(backings, "prompt_cache")?,
            }
        }
        Ok(())
    }
    pub fn retained_bytes(&self) -> Result<u64> {
        let mut backings = crate::gpu::memory::Backings::default();
        self.record_memory(&mut backings)?;
        backings.total()
    }
    pub fn position(&self) -> u32 {
        self.offset
    }
}

#[derive(serde::Serialize)]
pub struct FlashTraceTensor {
    pub name: String,
    pub shape: Vec<u32>,
    pub data: Vec<u8>,
}
pub struct FlashNext<'a> {
    pub(super) fixed_residency: Option<super::fixed_residency::FixedResidency>,
    pub(super) fixed_residency_checked: bool,
    checkpoint_path: std::path::PathBuf,
    lookahead_depth: usize,
    verify_ssd_cohorts: bool,
    #[cfg(feature = "chat")]
    pub(crate) prefill_heads: bool,
    #[cfg(test)]
    pub(crate) target_head_calls: usize,
    pub(super) prefetcher: Option<Rc<RefCell<super::moe::PrefetchPool<'a>>>>,
    owner: Rc<()>,
    pub(super) hidden: Option<MatrixView>,
    pub(super) gpu: &'a Metal,
    embed: TensorInfo,
    pub(super) layers: Vec<Layer<'a>>,
    pub(super) mixer: FlashHc<'a>,
    pub(super) head: FlashProjection<'a>,
    pub(super) context: u32,
    offset: u32,
    pub(super) request_limit: Option<u64>,
}
impl<'a> FlashNext<'a> {
    #[cfg(test)]
    pub(super) fn slru_counts(&self) -> Result<(usize, usize)> {
        self.layers
            .iter()
            .try_fold((0usize, 0usize), |(enabled, reservations), layer| {
                let (on, count) = match layer {
                    Layer::Gdn(a) => a.residual.moe.slru_counts(),
                    Layer::Ple(a) => a.body.residual.moe.slru_counts(),
                    Layer::Qsa(a) => a.residual.moe.slru_counts(),
                };
                Ok((
                    enabled
                        .checked_add(usize::from(on))
                        .context("SLRU layer count overflow")?,
                    reservations
                        .checked_add(count)
                        .context("SLRU reservation count overflow")?,
                ))
            })
    }

    #[cfg(test)]
    pub(super) fn ar10_record_reuse_layers(&mut self) -> usize {
        self.layers
            .iter_mut()
            .map(|layer| usize::from(layer.moe().record_reuse_enabled))
            .sum()
    }

    pub fn load(gpu: &'a Metal, c: &Checkpoint, context: u32) -> Result<Self> {
        let config = FlashConfig::load(c)?;
        #[cfg(feature = "chat")]
        let prefill_heads = match std::env::var("MLXL3_FLASH_PREFILL_HEADS") {
            Ok(v) if v == "0" => false,
            Ok(v) if v == "1" => true,
            Err(std::env::VarError::NotPresent) => true,
            _ => anyhow::bail!("MLXL3_FLASH_PREFILL_HEADS must be 0 or 1"),
        };
        let context = if context == 0 {
            u32::try_from(config.context)?
        } else {
            context
        };
        ensure!(
            (1..=crate::flash_support::FLASH_MAX_CONTEXT).contains(&context)
                && context <= config.context as u32,
            "native Flash context exceeds the supported checkpoint limit"
        );
        let lookahead_depth = match std::env::var("MLXL3_SSD_LOOKAHEAD_DEPTH") {
            Ok(value) => value
                .parse::<usize>()
                .context("invalid SSD lookahead depth")?,
            Err(std::env::VarError::NotPresent) => 0,
            Err(error) => return Err(error.into()),
        };
        ensure!(
            lookahead_depth <= 10,
            "SSD lookahead depth must be in 0..=10"
        );
        let request_limit = crate::flash_memory::request_limit()?;
        let slots = layer_slots()?;
        let total_slots = slots.iter().sum();
        gpu.admit_load(request_limit, total_slots)?;
        let embed = c
            .tensors
            .get("model.language_model.embed_tokens.weight")
            .context("missing token embedding")?
            .clone();
        ensure!(
            embed.dtype == "BF16" && embed.shape == [248320, 2560],
            "unsupported token embedding"
        );
        let mut layers: Vec<Layer> = slots
            .into_iter()
            .enumerate()
            .map(|(i, slots)| -> Result<_> {
                gpu.admit_load(request_limit, total_slots)?;
                let prefix = format!("model.language_model.layers.{i}");
                Ok(if i == 1 {
                    Layer::Ple(Box::new(FlashPleLayer::load(
                        gpu, c, &prefix, config.eps, slots,
                    )?))
                } else if i % 4 == 3 {
                    Layer::Qsa(Box::new(FlashQsaLayer::load(
                        gpu,
                        c,
                        &prefix,
                        config.eps,
                        config.rope,
                        context,
                        slots,
                    )?))
                } else {
                    Layer::Gdn(Box::new(FlashGdnLayer::load(
                        gpu, c, &prefix, config.eps, slots,
                    )?))
                })
            })
            .collect::<Result<_>>()?;
        let prefetcher = (lookahead_depth > 0)
            .then(|| super::moe::PrefetchPool::new(gpu).map(|pool| Rc::new(RefCell::new(pool))))
            .transpose()?;
        if let Some(pool) = &prefetcher {
            for layer in &mut layers {
                let moe = layer.moe();
                moe.enable_prefetch(pool.clone());
            }
        }
        let mixer = FlashHc::load(
            gpu,
            c,
            "model.language_model.hyper_connection_mixer",
            config.eps,
            false,
        )?;
        let head = FlashProjection::load(gpu, c, "lm_head")?;
        ensure!(
            head.dimensions() == [2560, 248320],
            "invalid Flash language model head"
        );
        gpu.admit_load(request_limit, total_slots)?;
        let mut model = Self {
            fixed_residency: None,
            fixed_residency_checked: false,
            checkpoint_path: c.path.clone(),
            lookahead_depth,
            verify_ssd_cohorts: {
                let value = match std::env::var("MLXL3_NATIVE_MTP_SSD_COHORTS") {
                    Ok(value) => Some(value),
                    Err(std::env::VarError::NotPresent) => None,
                    Err(error) => return Err(error.into()),
                };
                crate::flash_support::hc_kernel_flag(value.as_deref(), true)
                    .context("MLXL3_NATIVE_MTP_SSD_COHORTS must be 0 or 1")?
            },
            #[cfg(feature = "chat")]
            prefill_heads,
            #[cfg(test)]
            target_head_calls: 0,
            prefetcher,
            owner: Rc::new(()),
            hidden: None,
            gpu,
            embed,
            layers,
            mixer,
            head,
            context,
            offset: 0,
            request_limit,
        };
        model.ensure_fixed_residency()?;
        Ok(model)
    }
    pub fn snapshot(&self) -> FlashSnapshot {
        FlashSnapshot {
            owner: self.owner.clone(),
            offset: self.offset,
            hidden: self.hidden.clone(),
            layers: self.layers.iter().map(Layer::snapshot).collect(),
        }
    }
    pub(super) fn owns_snapshot(&self, snapshot: &FlashSnapshot) -> bool {
        Rc::ptr_eq(&self.owner, &snapshot.owner)
    }
    pub fn restore(&mut self, s: &FlashSnapshot) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.owner, &s.owner)
                && s.layers.len() == self.layers.len()
                && s.offset <= self.context,
            "snapshot belongs to a different Flash session"
        );
        for (l, state) in self.layers.iter_mut().zip(&s.layers) {
            l.restore(state)?;
        }
        self.offset = s.offset;
        self.hidden = s.hidden.clone();
        Ok(())
    }
    pub fn last_hidden(&self) -> Option<&MatrixView> {
        self.hidden.as_ref()
    }
    pub fn reset(&mut self) {
        for l in &mut self.layers {
            l.reset();
        }
        self.offset = 0;
        self.hidden = None;
    }
    pub fn log_probs(&self, logits: &[u16]) -> Result<Vec<f32>> {
        self.gpu.flash_log_probs(logits)
    }
    pub fn memory_stats(&self) -> Result<crate::runtime_stats::MemoryStats> {
        self.gpu.memory_stats()
    }
    pub fn offset(&self) -> u32 {
        self.offset
    }

    #[cfg(feature = "chat")]
    pub(crate) fn checkpoint_path(&self) -> &std::path::Path {
        &self.checkpoint_path
    }
    pub fn context_limit(&self) -> u32 {
        self.context
    }

    fn embedding_bits(&self, token: u32) -> Result<Vec<u16>> {
        let (file, base) = self.embed.source()?;
        let at = row_offset(base, u64::from(token), 248320, 5120)
            .context("token outside Flash vocabulary or embedding offset overflow")?;
        let mut bytes = [0u8; 5120];
        file.read_exact_at(&mut bytes, at)?;
        Ok(bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16::from_f32(bf16::from_le_bytes(*b).to_f32()).to_bits())
            .collect())
    }
    fn streams(&self, token: u32) -> Result<MatrixView> {
        let values: Vec<_> = self
            .embedding_bits(token)?
            .into_iter()
            .map(|b| f16::from_bits(b).to_f32())
            .collect();
        self.gpu
            .f32_storage(&values.repeat(4))?
            .matrix(MatrixLayout {
                shape: [1, 10240],
                strides: [10240, 1],
                offset: 0,
            })
    }

    pub fn forward(&mut self, token: u32) -> Result<Vec<u16>> {
        self.prefill(&[token])
    }

    /// Request admission is evaluated before any prefix or decode state advances.
    pub fn device_greedy_eligible(
        &self,
        temperature: f32,
        top_k: usize,
        penalty: f32,
        speculative: bool,
    ) -> bool {
        crate::gpu_greedy_policy::eligible(
            self.gpu.device_greedy_available(),
            temperature,
            top_k,
            penalty,
            speculative,
        )
    }
    pub fn device_greedy_counts(&self) -> [u64; 2] {
        [
            self.gpu.device_greedy_calls.get(),
            self.gpu.device_greedy_heads.get(),
        ]
    }
    pub fn greedy_from_logits(&self, logits: &[u16]) -> Result<u32> {
        self.gpu.device_greedy_from_logits(logits)
    }
    /// A scalar head result belongs to the same rollback boundary as recurrent state.
    pub fn forward_greedy(&mut self, token: u32) -> Result<u32> {
        self.forward_greedy_inner(token, false)
            .map(|(index, _)| index)
    }
    #[cfg(test)]
    pub(super) fn forward_greedy_captured(&mut self, token: u32) -> Result<(u32, Vec<u16>)> {
        let (index, logits) = self.forward_greedy_inner(token, true)?;
        Ok((
            index,
            logits.context("missing test-only device head capture")?,
        ))
    }
    fn forward_greedy_inner(
        &mut self,
        token: u32,
        capture: bool,
    ) -> Result<(u32, Option<Vec<u16>>)> {
        ensure!(
            self.gpu.device_greedy_available(),
            "device greedy is disabled or unsupported"
        );
        let end = crate::flash_support::flash_prompt_end(self.offset, 1, self.context)
            .context("Flash context exhausted")?;
        ensure!(token < 248_320, "token outside Flash vocabulary");
        self.ensure_fixed_residency()?;
        let before = self.snapshot();
        let result = (|| {
            let streams = self.hidden(token, None, None)?;
            let mixed = self.mixer.read(&streams, false)?.mixed;
            #[cfg(test)]
            {
                self.target_head_calls += 1;
            }
            let selected = self.gpu.device_greedy_head(&self.head, &mixed, capture)?;
            self.hidden = Some(streams);
            Ok(selected)
        })();
        if result.is_err() {
            self.restore(&before)?;
        } else {
            self.offset = end;
        }
        result
    }

    fn hidden(
        &mut self,
        token: u32,
        mut trace: Option<&mut Vec<FlashTraceTensor>>,
        mut captures: Option<(&[usize], &mut Vec<MatrixView>)>,
    ) -> Result<MatrixView> {
        use super::gdn::LayerOutput;
        struct RestoreRoute<'a>(&'a std::cell::Cell<bool>, bool);
        impl Drop for RestoreRoute<'_> {
            fn drop(&mut self) {
                self.0.set(self.1);
            }
        }
        // Exports retain the synchronous path, including on early errors.
        let gpu = self.gpu;
        let _route = (trace.is_some() || captures.is_some())
            .then(|| RestoreRoute(&gpu.gdn_route_chain, gpu.gdn_route_chain.replace(false)));
        let mut streams = LayerOutput::Ready(self.streams(token)?);
        for i in 0..self.layers.len() {
            #[cfg(test)]
            let _layer = crate::flash_profile::span("native.layer");
            #[cfg(test)]
            if let Some(span) = &_layer {
                span.data(serde_json::json!({"layer":i,"input_token":token}));
            }
            let (current, remaining) = self.layers.split_at_mut(i + 1);
            let layer = &mut current[i];
            let defer_write = crate::flash_support::cross_layer_eligible(
                self.gpu.cross_layer.get(),
                self.gpu.hc_gdn_chain.get(),
                self.lookahead_depth,
                trace.is_some() || captures.is_some(),
                matches!(remaining.first(), Some(Layer::Gdn(_))),
            );
            let next = remaining.first_mut().map(Layer::moe);
            streams = match streams {
                LayerOutput::Ready(streams) => {
                    layer.forward(&streams, token, next, self.lookahead_depth, defer_write)
                }
                LayerOutput::Pending(write) => match layer {
                    Layer::Gdn(layer) => {
                        layer.forward_pending(write, defer_write).with_context(|| {
                            format!("consuming deferred HC output of Flash layer {}", i - 1)
                        })
                    }
                    _ => anyhow::bail!("deferred HC output requires a regular GDN consumer"),
                },
            }
            .with_context(|| format!("running Flash layer {i}"))?;
            #[cfg(test)]
            ensure!(
                self.gpu.cross_layer_fail_after.get() != Some(i),
                "injected late layer failure {i}"
            );
            if let Some((ids, rows)) = &mut captures
                && ids.contains(&i)
            {
                rows.push(streams.ready()?.clone());
            }
            if let Some(t) = &mut trace {
                t.push(FlashTraceTensor {
                    name: format!("layer_{i}"),
                    shape: vec![1, 10240],
                    data: streams.ready()?.export_bytes(self.gpu)?,
                });
                if let Some(r) = layer.recurrent() {
                    t.push(FlashTraceTensor {
                        name: format!("gdn_state_{i}"),
                        shape: vec![1, 48, 128, 128],
                        data: r.export_bytes(self.gpu)?,
                    });
                }
            }
        }
        streams.complete(self.gpu)
    }

    #[cfg(feature = "chat")]
    pub(crate) fn forward_captured(
        &mut self,
        token: u32,
        ids: &[usize; 5],
    ) -> Result<(Vec<u16>, crate::native_bf16::Matrix)> {
        let (logits, captured) = self.forward_captured_inner(token, ids, true)?;
        Ok((logits.context("missing target head")?, captured))
    }

    #[cfg(feature = "chat")]
    pub(crate) fn prefill_captured(
        &mut self,
        token: u32,
        ids: &[usize; 5],
        row: usize,
        rows: usize,
    ) -> Result<(Option<Vec<u16>>, crate::native_bf16::Matrix)> {
        let head = crate::flash_support::prefill_head_required(row, rows, self.prefill_heads)
            .context("invalid native draft prefill row")?;
        self.forward_captured_inner(token, ids, head)
    }

    #[cfg(feature = "chat")]
    fn forward_captured_inner(
        &mut self,
        token: u32,
        ids: &[usize; 5],
        head: bool,
    ) -> Result<(Option<Vec<u16>>, crate::native_bf16::Matrix)> {
        let before = self.snapshot();
        let operation = (|| {
            let mut rows = Vec::with_capacity(5);
            let logits = self.prefill_inner(&[token], None, Some((ids, &mut rows)), head)?;
            ensure!(rows.len() == 5, "incomplete native draft captures");
            let mut captured = Vec::with_capacity(12800);
            for row in rows {
                ensure!(row.shape() == [1, 10240] && row.dtype == crate::gpu::view::FloatType::F32);
                let bytes = row.export_bytes(self.gpu)?;
                ensure!(bytes.len() == 40960);
                let values: Vec<_> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&b| f32::from_le_bytes(b))
                    .collect();
                for col in 0..2560 {
                    captured.push(crate::native_bf16::round(
                        ((values[col] + values[2560 + col])
                            + values[5120 + col]
                            + values[7680 + col])
                            * 0.25,
                    ));
                }
            }
            Ok((logits, crate::native_bf16::Matrix::new(1, 12800, captured)?))
        })();
        if operation.is_err() {
            self.restore(&before)?;
        }
        operation
    }

    #[cfg(feature = "chat")]
    pub(crate) fn draft_embedding(&self, token: u32) -> Result<Vec<f32>> {
        Ok(self
            .embedding_bits(token)?
            .into_iter()
            .map(|bits| crate::native_bf16::round(f16::from_bits(bits).to_f32()))
            .collect())
    }

    #[cfg(feature = "chat")]
    pub(crate) fn draft_logits(&self, rows: &crate::native_bf16::Matrix) -> Result<Vec<Vec<f32>>> {
        ensure!(
            rows.cols == 2560 && (1..=7).contains(&rows.rows),
            "invalid native draft head rows"
        );
        rows.values
            .as_chunks::<2560>()
            .0
            .iter()
            .map(|row| {
                let half: Vec<_> = row.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
                Ok(self
                    .head
                    .submit(&half)?
                    .read()?
                    .into_iter()
                    .map(|v| crate::native_bf16::round(f16::from_bits(v).to_f32()))
                    .collect())
            })
            .collect()
    }

    pub fn forward_trace(&mut self, token: u32) -> Result<(Vec<u16>, Vec<FlashTraceTensor>)> {
        let mut trace = Vec::with_capacity(84);
        let logits = self
            .prefill_inner(&[token], Some(&mut trace), None, true)?
            .context("missing target trace head")?;
        Ok((logits, trace))
    }
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>> {
        self.prefill_inner(tokens, None, None, true)?
            .context("missing target prefill head")
    }
    fn prefill_inner(
        &mut self,
        tokens: &[u32],
        trace: Option<&mut Vec<FlashTraceTensor>>,
        captures: Option<(&[usize], &mut Vec<MatrixView>)>,
        head: bool,
    ) -> Result<Option<Vec<u16>>> {
        let end = crate::flash_support::flash_prompt_end(self.offset, tokens.len(), self.context)
            .context("empty prompt or Flash context exhausted")?;
        ensure!(
            tokens.iter().all(|&t| t < 248320),
            "token outside Flash vocabulary"
        );
        self.ensure_fixed_residency()?;
        let (&last, prefix) = tokens.split_last().context("empty Flash prompt")?;
        let before = self.snapshot();
        let result = (|| {
            for &token in prefix {
                self.hidden(token, None, None)?;
            }
            let streams = self.hidden(last, trace, captures)?;
            let logits = if head {
                #[cfg(test)]
                let _head = crate::flash_profile::span("native.head");
                let mixed = self.mixer.read(&streams, false)?.mixed;
                #[cfg(test)]
                {
                    self.target_head_calls += 1;
                }
                Some(self.head.submit_fp16_view(&mixed)?.read()?)
            } else {
                None
            };
            self.hidden = Some(streams);
            Ok(logits)
        })();
        if result.is_err() {
            self.restore(&before)?;
        } else {
            self.offset = end;
        }
        result
    }
}

impl Metal {
    /// FP16 logsumexp and rounded subtraction preserve stock greedy ties.
    pub fn flash_log_probs(&self, logits: &[u16]) -> Result<Vec<f32>> {
        #[cfg(test)]
        let _logprobs = crate::flash_profile::span("native.logprobs");
        #[cfg(test)]
        if let Some(span) = &_logprobs {
            span.data(serde_json::json!({"input_bytes":logits.len()*2,"normalizer_readback_bytes":2,"output_F32_words":logits.len()}));
        }
        ensure!(logits.len() == 248320, "invalid Flash vocabulary width");
        ensure!(
            self.logsumexp.thread_execution_width() == 32
                && self.logsumexp.max_total_threads_per_threadgroup() >= 1024,
            "unsupported Flash logsumexp threadgroup"
        );
        let x = super::upload(&self.device, logits)?;
        let lse = super::buffer(
            &self.device,
            2,
            metal::MTLResourceOptions::StorageModeShared,
        )?;
        self.command(|cmd| {
            Metal::encode(
                cmd,
                &self.logsumexp,
                &[&x, &lse],
                [248320i32],
                metal::MTLSize::new(1, 1, 1),
                1024,
            );
            Ok(())
        })?;
        let lse = f16::from_bits(Metal::read(&lse, 1)[0]).to_f32();
        ensure!(lse.is_finite(), "non-finite Flash normalizer");
        Ok(logits
            .iter()
            .map(|&v| f16::from_f32(f16::from_bits(v).to_f32() - lse).to_f32())
            .collect())
    }
}

#[cfg(test)]
fn first_embedding(embed: &TensorInfo) -> Result<TensorInfo> {
    let temp = tempfile::tempdir()?;
    let path = temp.path().join("single.safetensors");
    let header = br#"{"embed":{"dtype":"BF16","shape":[1,2560],"data_offsets":[0,5120]}}"#;
    let (file, start) = embed.source()?;
    let mut row = [0u8; 5120];
    file.read_exact_at(&mut row, start)?;
    use std::io::Write;
    let mut f = std::fs::File::create(&path)?;
    f.write_all(&(header.len() as u64).to_le_bytes())?;
    f.write_all(header)?;
    f.write_all(&row)?;
    drop(f);
    crate::checkpoint::read_header(&path)?
        .remove("embed")
        .context("missing test row")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "physical current baseline native8 sampling-first diagnostic; exclusive GPU"]
    fn native8_sampling_profile_preserves_outputs() -> Result<()> {
        use crate::flash_profile as profile;
        use std::io::Write;
        fn digest(bytes: &[u8]) -> Result<String> {
            let mut file = tempfile::NamedTempFile::new()?;
            file.write_all(bytes)?;
            let out = std::process::Command::new("/usr/bin/shasum")
                .args(["-a", "256"])
                .arg(file.path())
                .output()?;
            ensure!(out.status.success());
            Ok(String::from_utf8(out.stdout)?
                .split_whitespace()
                .next()
                .context("missing digest")?
                .to_owned())
        }
        fn finite(x: &[u16]) -> bool {
            x.len() == 248320 && x.iter().all(|&v| f16::from_bits(v).is_finite())
        }
        fn select(model: &FlashNext, logits: &[u16]) -> Result<u32> {
            ensure!(finite(logits));
            let values = model.log_probs(logits)?;
            let _selection = profile::span("native.cpu_select");
            crate::speculative::greedy_log_prob_id(&values).context("invalid plain AR greedy")
        }
        fn run(
            model: &mut FlashNext,
            prompt: &[u32],
            observed: bool,
        ) -> Result<(Vec<u32>, Vec<serde_json::Value>)> {
            model.reset();
            let logits = model.prefill(prompt)?;
            let mut next = select(model, &logits)?;
            let mut ids = vec![next];
            if observed {
                profile::start(false)?;
            }
            for row in 1..8 {
                let request_before = super::super::BUFFER_REQUESTS.with(std::cell::Cell::get);
                let bytes_before = super::super::BUFFER_REQUESTED_BYTES.with(std::cell::Cell::get);
                let alloc_before = super::super::BUFFER_ALLOCATION_NS.with(std::cell::Cell::get);
                let row_span = profile::span("native.row");
                let input = next;
                let logits = model.forward(input)?;
                next = select(model, &logits)?;
                if let Some(span) = &row_span {
                    span.data(serde_json::json!({"row":row,"input_token":input,"selected_token":next,"vocabulary_words":logits.len(),"metal_requests":super::super::BUFFER_REQUESTS.with(std::cell::Cell::get)-request_before,"metal_requested_bytes":super::super::BUFFER_REQUESTED_BYTES.with(std::cell::Cell::get)-bytes_before,"metal_allocation_host_ns":super::super::BUFFER_ALLOCATION_NS.with(std::cell::Cell::get)-alloc_before}));
                }
                ids.push(next);
            }
            let events = if observed {
                let trace = profile::finish()?;
                profile::validate_cross_layer_ancestry(&trace)?;
                serde_json::to_value(trace)?
                    .as_array()
                    .context("events")?
                    .clone()
            } else {
                Vec::new()
            };
            ensure!(ids.len() == 8 && model.offset() as usize == prompt.len() + 7);
            Ok((ids, events))
        }
        fn terminal(
            model: &mut FlashNext,
            gpu: &Metal,
        ) -> Result<(Vec<u8>, Vec<u16>, Vec<FlashTraceTensor>)> {
            let hidden = model
                .last_hidden()
                .context("missing hidden")?
                .export_bytes(gpu)?;
            ensure!(
                hidden.len() == 40960
                    && hidden
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|v| f32::from_le_bytes(*v).is_finite())
            );
            let (logits, states) = model.forward_trace(7)?;
            ensure!(finite(&logits) && states.len() == 84);
            Ok((hidden, logits, states))
        }
        let path = std::path::PathBuf::from(std::env::var("MLXL3_FLASH_MODEL")?);
        let tokenizer = crate::tokenizer::ChatTokenizer::load(&path)?;
        let prompt=tokenizer.encode(&tokenizer.render_values(&serde_json::json!([{"role":"user","content":"Explique en français, avec un exemple concret, comment un cache LFU évite les lectures SSD inutiles pendant l’inférence d’un modèle MoE."}]),None)?)?;
        ensure!(prompt.len() == 87);
        let c = crate::checkpoint::inspect(&path)?;
        let gpu = Metal::new()?;
        ensure!(
            [
                "MLXL3_NATIVE_SSD_DIRECT",
                "MLXL3_NATIVE_AR_HC_CHAIN",
                "MLXL3_NATIVE_AR_HC_GATE_CHAIN",
                "MLXL3_NATIVE_FLASH_STATIC_K",
                "MLXL3_NATIVE_FLASH_NT2",
                "MLXL3_NATIVE_FLASH_FIXED_DIMS",
                "MLXL3_NATIVE_ROUTED_SCRATCH",
                "MLXL3_NATIVE_HC_GDN_CHAIN",
                "MLXL3_NATIVE_CROSS_LAYER",
                "MLXL3_NATIVE_HC_WORKSPACE",
                "MLXL3_NATIVE_SSD_EXTRA_WAVE",
                "MLXL3_NATIVE_HC_UP_MIX",
                "MLXL3_NATIVE_HC_DOWN_INJECT",
            ]
            .iter()
            .all(|name| std::env::var_os(name).is_none()),
            "native8 requires actual default policies, no explicit optimization flags"
        );
        ensure!(
            gpu.device.name() == "Apple M5"
                && gpu.ssd_direct_enabled
                && gpu.hc_chain.get()
                && gpu.hc_gate_chain.get()
                && gpu.cross_layer.get()
                && gpu.hc_gdn_chain.get(),
            "native8 current baseline requires qualified M5 and accepted defaults"
        );
        let mut model = FlashNext::load(&gpu, &c, 4096)?;
        ensure!(model.lookahead_depth == 0 && model.prefetcher.is_none());
        let (first, first_states) = model.forward_trace(248044)?;
        ensure!(finite(&first) && first_states.len() == 84);
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/native-coverage/flash-state/initial.json"
        ))?;
        let state = &first_states[0];
        let reference = &oracle["step_tensors"][0];
        ensure!(
            state.name == reference["name"]
                && serde_json::to_value(&state.shape)? == reference["shape"]
                && digest(&state.data)? == reference["sha256"]
        );
        ensure!(
            digest(
                &first
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            )? == "2f78a4d78cb113666fa3eda97cf115abe293bd428e593bc54400744eae422445"
        );
        drop(first_states);
        let (warm, _) = run(&mut model, &prompt, false)?;
        let (control, _) = run(&mut model, &prompt, false)?;
        let expected = terminal(&mut model, &gpu)?;
        let (observed, events) = run(&mut model, &prompt, true)?;
        let actual = terminal(&mut model, &gpu)?;
        ensure!(
            warm == control
                && observed == control
                && expected.0 == actual.0
                && expected.1 == actual.1
        );
        let mut state_words = 0usize;
        for (a, b) in expected.2.iter().zip(&actual.2) {
            let n = a
                .shape
                .iter()
                .try_fold(1usize, |n, &v| n.checked_mul(v as usize))
                .context("state overflow")?;
            ensure!(
                n > 0
                    && a.name == b.name
                    && a.shape == b.shape
                    && a.data == b.data
                    && a.data.len() == n * 4
                    && a.data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|v| f32::from_le_bytes(*v).is_finite())
            );
            state_words += n;
        }
        ensure!(state_words == 28803072 && gpu.readers()?.in_flight() == 0);
        for (name, count) in [
            ("native.row", 7),
            ("native.layer", 336),
            ("native.head", 7),
            ("native.logprobs", 7),
            ("native.cpu_select", 7),
            ("native.projection_readback_copy", 7),
            ("native.hc.encode_read", 7 * 97),
            ("native.hc.chain_write_read_gate", 7 * 48),
            ("native.gdn.frame", 7 * 36),
        ] {
            ensure!(
                events.iter().filter(|e| e["name"] == name).count() == count,
                "wrong event coverage{name}"
            );
        }
        let separate_gdn = events
            .iter()
            .filter(|e| e["name"] == "native.gdn.body")
            .count();
        let joined_gdn = events
            .iter()
            .filter(|e| e["name"] == "native.hc_gdn.joined")
            .count();
        let cross_layer_gdn = events
            .iter()
            .filter(|e| e["name"] == "native.hc_gdn.cross_layer")
            .count();
        ensure!(cross_layer_gdn == 7 * 34 && joined_gdn == 7 * 2 && separate_gdn == 0);
        ensure!(separate_gdn + joined_gdn + cross_layer_gdn == 7 * 36);
        let moe_wait_events = events
            .iter()
            .filter(|e| e["name"] == "native.moe_wait")
            .count();
        let moe_gpu_intervals = events
            .iter()
            .filter(|e| e["name"] == "native.gpu_interval" && e["data"]["owner"] == "moe")
            .count();
        ensure!(moe_wait_events > 0 && moe_gpu_intervals > 0);
        for (exclusive, row_total) in [
            ("allocation_requests", "metal_requests"),
            ("allocation_requested_bytes", "metal_requested_bytes"),
            ("allocation_api_ns", "metal_allocation_host_ns"),
        ] {
            let attributed = events
                .iter()
                .map(|e| e[exclusive].as_u64().unwrap())
                .sum::<u64>();
            let rows = events
                .iter()
                .filter(|e| e["name"] == "native.row")
                .map(|e| e["data"][row_total].as_u64().unwrap())
                .sum::<u64>();
            ensure!(
                attributed > 0 && attributed == rows,
                "allocation attribution lost or double-counted {exclusive}"
            );
        }
        let workspace_requests = events
            .iter()
            .filter(|e| {
                e["name"]
                    .as_str()
                    .is_some_and(|name| name.starts_with("native.hc.workspace."))
            })
            .map(|e| e["allocation_requests"].as_u64().unwrap())
            .sum::<u64>();
        ensure!(workspace_requests == 7 * 97 * 4);
        ensure!(
            events
                .iter()
                .all(|e| e["end_ns"].as_u64() >= e["start_ns"].as_u64())
        );
        let head_gpu = events
            .iter()
            .filter(|e| {
                e["name"] == "native.gpu_interval"
                    && e["data"]["owner"] == "projection"
                    && e["data"]["valid"] == true
            })
            .count();
        ensure!(head_gpu == 7);
        eprintln!(
            "I120_NATIVE8_PROFILE {}",
            serde_json::json!({"status":"diagnostic_complete","parity":true,"tokens":observed,"prompt_tokens":87,"context":4096,"decode_rows":7,"cross_layer_triples":cross_layer_gdn,"hc_gdn_doubles":joined_gdn,"gdn_separate":separate_gdn,"profile_scope_ancestry_checked":true,"moe_wait_events":moe_wait_events,"moe_gpu_intervals":moe_gpu_intervals,"independent_first_state":true,"independent_first_logits":true,"continuation_state_words":state_words,"events":events,"scope":"test-only intrusive sampling-first accepted L1 baseline on exact M5/defaults; no release speed","limits":["CPU/GPU clocks not aligned; intervals overlap and are not additive","GPUcoverage central/projection/MoE completed CB only, no devicewide utilization or perkernel split","SSDtimeline is existing worker bytes/CPU offsets, not NAND traffic","CPUencode coverage central helper only; MoE encode CPU residual and allocator-internal work not isolated","allocation counters are native Metal requests, not process RAM; counters saturate only at u64MAX"]})
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical Flash AR HC chain counts, full target states and late IO rollback"]
    fn hc_chain_keeps_ar_logits_states_and_actual_command_savings() -> Result<()> {
        check_hc_ar_chain(false)
    }

    #[test]
    #[ignore = "physical HC gate joint path, 48 real CB savings and complete restored AR states"]
    fn hc_gate_chain_keeps_ar_logits_states_and_actual_command_savings() -> Result<()> {
        check_hc_ar_chain(true)
    }

    fn check_hc_ar_chain(gate_chain: bool) -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        eprintln!("HC chain initially enabled: {}", gpu.hc_chain.get());
        let mut model = FlashNext::load(&gpu, &c, 64)?;
        let prompt = [248044, 78768, 11, 7, 19, 31, 13, 29, 17];
        let finite =
            |x: &[u16]| x.len() == 248320 && x.iter().all(|&v| f16::from_bits(v).is_finite());
        let mut compared = 0;
        for prefix in [0, 2, 9] {
            model.reset();
            gpu.hc_chain.set(gate_chain);
            gpu.hc_gate_chain.set(false);
            if prefix > 0 {
                ensure!(finite(&model.prefill(&prompt[..prefix])?));
            }
            let saved = model.snapshot();
            // Give both restored branches the same QSA copy-on-write cost.
            ensure!(finite(&model.forward(11)?));
            model.restore(&saved)?;
            let commands = gpu.helper_commits.get();
            let pairs = gpu.hc_chain_pairs.get();
            let gate_commands = gpu.gate_commands.get();
            let waits = gpu.helper_waits.get() + gpu.route_waits.get();
            let expected = model.forward(11)?;
            let control_gate_commands = gpu.gate_commands.get() - gate_commands;
            let control_waits = gpu.helper_waits.get() + gpu.route_waits.get() - waits;
            let control_commands = gpu.helper_commits.get() - commands;
            ensure!(
                finite(&expected)
                    && gpu.hc_chain_pairs.get() - pairs == if gate_chain { 48 } else { 0 }
            );
            let hidden = model
                .last_hidden()
                .context("missing control row")?
                .export_bytes(&gpu)?;
            ensure!(
                hidden.len() == 40960
                    && hidden
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|v| f32::from_le_bytes(*v).is_finite())
            );
            let (next, states) = model.forward_trace(7)?;
            ensure!(finite(&next) && states.len() == 84);
            model.restore(&saved)?;
            gpu.hc_chain.set(true);
            gpu.hc_gate_chain.set(gate_chain);
            let gate_pairs = gpu.hc_gate_chain_pairs.get();
            let gate_commands = gpu.gate_commands.get();
            let waits = gpu.helper_waits.get() + gpu.route_waits.get();
            let commands = gpu.helper_commits.get();
            let pairs = gpu.hc_chain_pairs.get();
            let actual = model.forward(11)?;
            let candidate_commands = gpu.helper_commits.get() - commands;
            let candidate_gate_commands = gpu.gate_commands.get() - gate_commands;
            let candidate_waits = gpu.helper_waits.get() + gpu.route_waits.get() - waits;
            ensure!(gpu.hc_chain_pairs.get() - pairs == 48);
            if gate_chain {
                ensure!(model.lookahead_depth == 0 && model.prefetcher.is_none());
                ensure!(gpu.hc_gate_chain_pairs.get() - gate_pairs == 48);
                ensure!(control_gate_commands == 48 && candidate_gate_commands == 0);
                ensure!(candidate_commands == control_commands && candidate_waits == control_waits);
            } else {
                ensure!(candidate_commands + 48 == control_commands);
            }
            eprintln!(
                "NATIVE_FLASH_CHAIN_COUNTS {}",
                serde_json::json!({"gate_chain":gate_chain,"prefix":prefix,"control_helper_cb":control_commands,"candidate_helper_cb":candidate_commands,"control_gate_cb":control_gate_commands,"candidate_gate_cb":candidate_gate_commands,"control_wait_calls":control_waits,"candidate_wait_calls":candidate_waits,"joint_gate_pairs":gpu.hc_gate_chain_pairs.get()-gate_pairs})
            );
            ensure!(actual == expected && finite(&actual) && model.offset() as usize == prefix + 1);
            ensure!(
                model
                    .last_hidden()
                    .context("missing candidate row")?
                    .export_bytes(&gpu)?
                    == hidden
            );
            let (actual_next, actual_states) = model.forward_trace(7)?;
            ensure!(actual_next == next && actual_states.len() == 84);
            let mut words = 0;
            for (a, b) in actual_states.iter().zip(&states) {
                let count = a
                    .shape
                    .iter()
                    .try_fold(1usize, |n, &v| n.checked_mul(v as usize))
                    .context("HC trace overflow")?;
                ensure!(
                    count > 0
                        && a.name == b.name
                        && a.shape == b.shape
                        && a.data == b.data
                        && a.data.len() == count * 4
                        && a.data
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .all(|v| f32::from_le_bytes(*v).is_finite()),
                    "HC chain changed state {} at prefix={prefix}",
                    a.name
                );
                words += count;
            }
            ensure!(words == 28_803_072);
            compared += words;
            eprintln!(
                "HC chain AR prefix={prefix}:48joint pairs, helper commands={control_commands}->{candidate_commands}, logits=248320, finite state_words={words}"
            );
        }
        ensure!(compared == 86_409_216);
        model.reset();
        gpu.hc_chain.set(true);
        model.prefill(&prompt[..2])?;
        let before = model.snapshot();
        let expected = model.forward(7)?;
        model.restore(&before)?;
        let short = super::first_embedding(&model.embed)?;
        let embed = std::mem::replace(&mut model.embed, short);
        ensure!(model.prefill(&[0, 1]).is_err() && model.offset() == before.offset);
        model.embed = embed;
        let recovered = model.forward(7)?;
        ensure!(
            finite(&recovered) && recovered == expected,
            "late failed row changed AR recovery"
        );
        let before = model.snapshot();
        ensure!(model.forward(248320).is_err() && model.offset() == before.offset);
        ensure!(model.prefill(&[]).is_err() && model.offset() == before.offset);
        let static_submissions = gpu
            .flash_static_submissions
            .each_ref()
            .map(std::cell::Cell::get);
        if gpu.flash_static_setting == Some(true) {
            ensure!(
                (static_submissions[0] > 0
                    || gpu.flash_fixed_submissions.iter().all(|c| c.get() > 0))
                    && static_submissions[3] + static_submissions[4] > 0
            );
        } else {
            ensure!(static_submissions == [0; 5]);
        }
        eprintln!("HC chain static QMV submissions: {static_submissions:?}");
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple Metal half-vocabulary logsumexp"]
    fn normalized_greedy_preserves_half_ties_and_rejects_invalid_distribution() -> Result<()> {
        let gpu = Metal::new()?;
        let mut logits = vec![0u16; 248320];
        logits[1] = f16::from_f32(0.001).to_bits();
        let probabilities = gpu.flash_log_probs(&logits)?;
        let lse =
            f16::from_f64((248319.0 + f64::from(f16::from_bits(logits[1]).to_f32()).exp()).ln())
                .to_f32();
        ensure!(
            probabilities.len() == logits.len()
                && probabilities
                    .iter()
                    .zip(&logits)
                    .all(|(&p, &x)| p == f16::from_f32(f16::from_bits(x).to_f32() - lse).to_f32())
        );
        ensure!(logits[0] != logits[1] && probabilities[0] == probabilities[1]);
        ensure!(gpu.flash_log_probs(&logits[..100]).is_err());
        logits.fill(f16::NEG_INFINITY.to_bits());
        ensure!(gpu.flash_log_probs(&logits).is_err());
        logits.fill(0);
        logits[2] = f16::NAN.to_bits();
        ensure!(gpu.flash_log_probs(&logits).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "physical Flash lookahead, archived independent full states/logits and continuation"]
    fn predictive_reads_match_archived_full_states_and_imposed_continuation() -> Result<()> {
        check_archived_full_states(true)
    }

    #[test]
    #[ignore = "physical depth0 joint HC gate against independent archived full states/logits"]
    fn hc_gate_depth_zero_matches_archived_full_states_and_continuation() -> Result<()> {
        check_archived_full_states(false)
    }

    fn check_archived_full_states(predictive: bool) -> Result<()> {
        fn digest(bytes: &[u8]) -> Result<String> {
            use std::io::Write;
            let mut file = tempfile::NamedTempFile::new()?;
            file.write_all(bytes)?;
            let output = std::process::Command::new("/usr/bin/shasum")
                .args(["-a", "256"])
                .arg(file.path())
                .output()?;
            ensure!(output.status.success(), "cannot hash native state");
            let output = String::from_utf8(output.stdout)?;
            let hash = output
                .split_whitespace()
                .next()
                .context("missing state digest")?;
            ensure!(hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()));
            Ok(hash.to_owned())
        }
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        let hc_initially_enabled = gpu.hc_chain.get();
        let hc_gate_initially_enabled = gpu.hc_gate_chain.get();
        let fixed_initially_enabled = gpu.flash_fixed_setting == Some(true);
        let scratch_initially_enabled = gpu.flash_expert_scratch_setting == Some(true);
        let mut model = FlashNext::load(&gpu, &c, 64)?;
        if predictive {
            ensure!(model.lookahead_depth > 0 && model.prefetcher.is_some());
        } else {
            ensure!(
                model.lookahead_depth == 0
                    && model.prefetcher.is_none()
                    && gpu.hc_gate_chain.get()
                    && gpu.hc_chain.get()
            );
        }
        let oracle: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/native-coverage/flash-state/initial.json"
        ))?;
        let (first, states) = model.forward_trace(248044)?;
        let words = first
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>();
        ensure!(first.len() == 248320 && first.iter().all(|&v| f16::from_bits(v).is_finite()));
        ensure!(
            digest(&words)? == "2f78a4d78cb113666fa3eda97cf115abe293bd428e593bc54400744eae422445"
        );
        let expected = oracle["step_tensors"]
            .as_array()
            .context("missing archived states")?;
        ensure!(states.len() == 84 && expected.len() == 84);
        let mut count = 0;
        for (state, reference) in states.iter().zip(expected) {
            ensure!(
                state.name == reference["name"]
                    && serde_json::to_value(&state.shape)? == reference["shape"]
            );
            let elements = state
                .shape
                .iter()
                .try_fold(1usize, |n, &v| n.checked_mul(v as usize))
                .context("state shape overflow")?;
            ensure!(
                elements > 0
                    && state.data.len() == elements * 4
                    && state
                        .data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|b| f32::from_le_bytes(*b).is_finite())
            );
            ensure!(
                digest(&state.data)? == reference["sha256"],
                "lookahead changed {}",
                state.name
            );
            count += elements;
        }
        ensure!(count == 28_803_072);
        drop(states);
        let second = model.forward(1)?;
        ensure!(second.len() == 248320 && second.iter().all(|&v| f16::from_bits(v).is_finite()));
        ensure!(
            digest(
                &second
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>()
            )? == "833ef923345928bdfb2ea39847a722c568cb709feaa225e4476ba5f6daac3815"
        );
        model.reset();
        ensure!(model.prefill(&[248044, 1])? == second);
        if predictive {
            let pool = model.prefetcher.as_ref().unwrap();
            pool.borrow_mut().drain()?;
            ensure!(
                pool.borrow().stats.submitted > 0
                    && pool.borrow().stats.published > 0
                    && pool.borrow().in_flight() == 0
            );
            ensure!(gpu.hc_gate_chain_pairs.get() == 0 && gpu.gate_commands.get() > 0);
            eprintln!(
                "native prefetch model: {}",
                serde_json::json!({"state_words":count,"logits":first.len()+second.len(),"prefetch":pool.borrow().stats})
            );
        } else {
            ensure!(gpu.hc_gate_chain_pairs.get() == 192 && gpu.gate_commands.get() == 0);
        }
        let static_submissions = gpu
            .flash_static_submissions
            .each_ref()
            .map(std::cell::Cell::get);
        if gpu.flash_static_setting == Some(true) {
            ensure!(
                (static_submissions[0] > 0
                    || gpu.flash_fixed_submissions.iter().all(|c| c.get() > 0))
                    && static_submissions[3] + static_submissions[4] > 0,
                "static K target did not execute both cold K2 and resident K4"
            );
        } else {
            ensure!(static_submissions == [0; 5]);
        }
        let fixed_submissions = gpu
            .flash_fixed_submissions
            .each_ref()
            .map(std::cell::Cell::get);
        ensure!(
            fixed_submissions
                == if gpu.flash_fixed_setting == Some(true) {
                    [3840, 1920]
                } else {
                    [0, 0]
                }
        );
        ensure!(
            gpu.flash_scratch_uses.get()
                == if gpu.flash_expert_scratch_setting == Some(true) {
                    5760
                } else {
                    0
                }
        );
        eprintln!(
            "NATIVE_FLASH_STATIC_SUBMISSIONS {}",
            serde_json::json!(static_submissions)
        );
        eprintln!(
            "NATIVE_FLASH_COMPOSED_FEATURES {}",
            serde_json::json!({"hc_initially_enabled":hc_initially_enabled,"hc_gate_initially_enabled":hc_gate_initially_enabled,"fixed_initially_enabled":fixed_initially_enabled,"scratch_initially_enabled":scratch_initially_enabled,"hc_joint_pairs":gpu.hc_chain_pairs.get(),"hc_gate_pairs":gpu.hc_gate_chain_pairs.get(),"gate_commands":gpu.gate_commands.get(),"static_submissions":static_submissions,"fixed_dims":gpu.flash_fixed_setting == Some(true),"fixed_submissions":fixed_submissions,"scratch_enabled":gpu.flash_expert_scratch_setting == Some(true),"scratch_uses":gpu.flash_scratch_uses.get()})
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical Flash fixed K2 dimensions context4096 full states, indexer and continuation"]
    fn fixed_dimensions_keep_context4096_states_logits_and_continuation() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut gpu = Metal::new()?;
        ensure!(
            gpu.flash_fixed_setting == Some(true) && gpu.flash_fixed_qmv.is_some(),
            "requires fixed dims1 on Apple M5"
        );
        let finite =
            |x: &[u16]| x.len() == 248320 && x.iter().all(|&v| f16::from_bits(v).is_finite());
        let initially_enabled = [
            gpu.hc_gate_chain.get(),
            gpu.flash_fixed_setting == Some(true),
            gpu.flash_expert_scratch_setting == Some(true),
        ];
        eprintln!(
            "NATIVE_FLASH_INITIAL_DEFAULTS {}",
            serde_json::json!({"hc_gate":initially_enabled[0],"fixed_dims":initially_enabled[1],"scratch":initially_enabled[2]})
        );
        let mut expected = Vec::new();
        let mut expected_continuation = None;
        for enabled in [false, true] {
            gpu.flash_fixed_setting = Some(enabled);
            let before_scratch = gpu.flash_scratch_uses.get();
            let before_gate = gpu.hc_gate_chain_pairs.get();
            let before = gpu
                .flash_fixed_submissions
                .each_ref()
                .map(std::cell::Cell::get);
            let mut model = FlashNext::load(&gpu, &c, 4096)?;
            for (step, token) in [248044, 1, 7].into_iter().enumerate() {
                let (logits, states) = model.forward_trace(token)?;
                ensure!(finite(&logits) && states.len() == 84);
                let mut words = 0;
                for state in &states {
                    let count = state
                        .shape
                        .iter()
                        .try_fold(1usize, |n, &v| n.checked_mul(v as usize))
                        .context("fixed trace overflow")?;
                    ensure!(
                        count > 0
                            && state.data.len() == count * 4
                            && state
                                .data
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .all(|v| f32::from_le_bytes(*v).is_finite())
                    );
                    words += count;
                }
                ensure!(words == 28_803_072);
                if enabled {
                    let (ref_logits, ref_states): &(Vec<u16>, Vec<FlashTraceTensor>) =
                        &expected[step];
                    ensure!(logits == *ref_logits);
                    ensure!(
                        states.iter().zip(ref_states).all(|(a, b)| a.name == b.name
                            && a.shape == b.shape
                            && a.data == b.data)
                    );
                } else {
                    expected.push((logits, states));
                }
            }
            let saved = model.snapshot();
            let next = model.forward(11)?;
            ensure!(finite(&next));
            if enabled {
                ensure!(expected_continuation.as_ref() == Some(&next));
            } else {
                expected_continuation = Some(next.clone());
            }
            model.restore(&saved)?;
            ensure!(model.forward(11)? == next);
            if let Some(pool) = &model.prefetcher {
                pool.borrow_mut().drain()?;
                ensure!(pool.borrow().in_flight() == 0);
            }
            let gate_eligible = model.lookahead_depth == 0 && model.prefetcher.is_none();
            drop(model);
            let after = gpu
                .flash_fixed_submissions
                .each_ref()
                .map(std::cell::Cell::get);
            let delta: Vec<_> = after.iter().zip(before).map(|(a, b)| a - b).collect();
            ensure!(
                if enabled {
                    delta == [4860, 2400]
                } else {
                    after == before
                },
                "fixed4096 did not execute all routed and12 resident indexer projections per row: {delta:?}"
            );
            let scratch_uses = gpu.flash_scratch_uses.get() - before_scratch;
            let gate_pairs = gpu.hc_gate_chain_pairs.get() - before_gate;
            ensure!(
                scratch_uses
                    == if gpu.flash_expert_scratch_setting == Some(true) {
                        7200
                    } else {
                        0
                    }
            );
            ensure!(
                gate_pairs
                    == if gpu.hc_gate_chain.get() && gpu.hc_chain.get() && gate_eligible {
                        240
                    } else {
                        0
                    }
            );
            eprintln!(
                "NATIVE_FLASH_FIXED_4096 {}",
                serde_json::json!({"enabled":enabled,"steps":3,"states":252,"state_words":86_409_216,"logits":744960,"fixed_submissions":delta,"scratch_uses":scratch_uses,"hc_gate_pairs":gate_pairs,"continuation":true})
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "physical whole-model Flash snapshot ownership, fork and replay"]
    fn retained_snapshots_restore_hidden_logits_and_empty_session() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut model = FlashNext::load(&gpu, &c, 4)?;
        let empty = model.snapshot();
        ensure!(model.last_hidden().is_none());
        let first = model.forward(248044)?;
        let prefix = model.snapshot();
        let hidden = model
            .last_hidden()
            .context("missing hidden row")?
            .export_bytes(&gpu)?;
        let left = model.forward(42)?;
        let left_state = model.snapshot();
        let left_end = model.forward(7)?;
        model.restore(&prefix)?;
        ensure!(model.offset == 1 && model.last_hidden().unwrap().export_bytes(&gpu)? == hidden);
        let right = model.forward(7)?;
        let right_state = model.snapshot();
        let right_end = model.forward(42)?;
        ensure!(left != right && left_end != right_end);
        let mut foreign = prefix.clone();
        foreign.owner = Rc::new(());
        ensure!(model.restore(&foreign).is_err() && model.offset == 3);
        model.restore(&left_state)?;
        ensure!(model.forward(7)? == left_end);
        model.reset();
        model.restore(&right_state)?;
        ensure!(model.forward(42)? == right_end);
        model.restore(&empty)?;
        ensure!(
            model.last_hidden().is_none() && model.offset == 0 && model.forward(248044)? == first
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical Flash whole-model prefill transaction and late read failure"]
    fn partial_prompt_io_failure_rolls_back_every_layer() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut model = FlashNext::load(&gpu, &c, 3)?;
        let expected_first = model.forward(42)?;
        let expected = model.prefill(&[0, 1])?;
        ensure!(model.forward(0).is_err() && model.offset == 3);
        model.reset();
        ensure!(
            model.prefill(&[]).is_err()
                && model.prefill(&[0, 248320]).is_err()
                && model.offset == 0
        );
        ensure!(model.forward(42)? == expected_first);
        // A real pinned one-row shard lets token 0 finish, then forces EOF when reading token 1.
        let short = first_embedding(&model.embed)?;
        let original = std::mem::replace(&mut model.embed, short);
        let error = model
            .prefill(&[0, 1])
            .err()
            .context("short embedding accepted")?;
        ensure!(
            error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|e| e.kind() == std::io::ErrorKind::UnexpectedEof),
            "unexpected error: {error:#}"
        );
        ensure!(model.offset == 1);
        model.embed = original;
        ensure!(model.prefill(&[0, 1])? == expected && model.offset == 3);
        Ok(())
    }
}

#[cfg(test)]
#[path = "ssd_wave_model_tests.rs"]
mod ssd_wave_tests;
