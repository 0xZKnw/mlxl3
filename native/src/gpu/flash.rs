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

pub(super) enum Layer<'a> {
    Gdn(FlashGdnLayer<'a>),
    Ple(Box<FlashPleLayer<'a>>),
    Qsa(FlashQsaLayer<'a>),
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
    ) -> Result<MatrixView> {
        match self {
            Self::Gdn(a) => a.forward_lookahead(x, next, depth),
            Self::Ple(a) => a.forward_lookahead(x, token, next, depth),
            Self::Qsa(a) => a.forward_lookahead(x, next, depth),
        }
    }
    fn moe(&mut self) -> &mut super::FlashMoeRow<'a> {
        match self {
            Self::Gdn(a) => &mut a.residual.moe,
            Self::Ple(a) => &mut a.body.residual.moe,
            Self::Qsa(a) => &mut a.residual.moe,
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
    checkpoint_path: std::path::PathBuf,
    lookahead_depth: usize,
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
                    Layer::Qsa(FlashQsaLayer::load(
                        gpu,
                        c,
                        &prefix,
                        config.eps,
                        config.rope,
                        context,
                        slots,
                    )?)
                } else {
                    Layer::Gdn(FlashGdnLayer::load(gpu, c, &prefix, config.eps, slots)?)
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
        Ok(Self {
            checkpoint_path: c.path.clone(),
            lookahead_depth,
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
        })
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

    fn hidden(
        &mut self,
        token: u32,
        mut trace: Option<&mut Vec<FlashTraceTensor>>,
        mut captures: Option<(&[usize], &mut Vec<MatrixView>)>,
    ) -> Result<MatrixView> {
        let mut streams = self.streams(token)?;
        for i in 0..self.layers.len() {
            let (current, remaining) = self.layers.split_at_mut(i + 1);
            let layer = &mut current[i];
            let next = remaining.first_mut().map(Layer::moe);
            streams = layer
                .forward(&streams, token, next, self.lookahead_depth)
                .with_context(|| format!("running Flash layer {i}"))?;
            if let Some((ids, rows)) = &mut captures
                && ids.contains(&i)
            {
                rows.push(streams.clone());
            }
            if let Some(t) = &mut trace {
                t.push(FlashTraceTensor {
                    name: format!("layer_{i}"),
                    shape: vec![1, 10240],
                    data: streams.export_bytes(self.gpu)?,
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
        Ok(streams)
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
        let (&last, prefix) = tokens.split_last().context("empty Flash prompt")?;
        let before = self.snapshot();
        let result = (|| {
            for &token in prefix {
                self.hidden(token, None, None)?;
            }
            let streams = self.hidden(last, trace, captures)?;
            let logits = if head {
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
    #[ignore = "physical Flash AR HC chain counts, full target states and late IO rollback"]
    fn hc_chain_keeps_ar_logits_states_and_actual_command_savings() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut model = FlashNext::load(&gpu, &c, 64)?;
        let prompt = [248044, 78768, 11, 7, 19, 31, 13, 29, 17];
        let finite =
            |x: &[u16]| x.len() == 248320 && x.iter().all(|&v| f16::from_bits(v).is_finite());
        let mut compared = 0;
        for prefix in [0, 2, 9] {
            model.reset();
            gpu.hc_chain.set(false);
            if prefix > 0 {
                ensure!(finite(&model.prefill(&prompt[..prefix])?));
            }
            let saved = model.snapshot();
            let commands = gpu.helper_commands.get();
            let pairs = gpu.hc_chain_pairs.get();
            let expected = model.forward(11)?;
            let control_commands = gpu.helper_commands.get() - commands;
            ensure!(finite(&expected) && gpu.hc_chain_pairs.get() == pairs);
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
            let commands = gpu.helper_commands.get();
            let pairs = gpu.hc_chain_pairs.get();
            let actual = model.forward(11)?;
            let candidate_commands = gpu.helper_commands.get() - commands;
            ensure!(
                gpu.hc_chain_pairs.get() - pairs == 48
                    && candidate_commands + 48 == control_commands,
                "joint encoding did not remove exactly48 helper commands at prefix={prefix}: {control_commands}->{candidate_commands}"
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
        let mut model = FlashNext::load(&gpu, &c, 64)?;
        ensure!(model.lookahead_depth > 0 && model.prefetcher.is_some());
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
        let pool = model.prefetcher.as_ref().unwrap();
        pool.borrow_mut().drain()?;
        ensure!(
            pool.borrow().stats.submitted > 0
                && pool.borrow().stats.published > 0
                && pool.borrow().in_flight() == 0
        );
        eprintln!(
            "native prefetch model: {}",
            serde_json::json!({"state_words":count,"logits":first.len()+second.len(),"prefetch":pool.borrow().stats})
        );
        let static_submissions = gpu
            .flash_static_submissions
            .each_ref()
            .map(std::cell::Cell::get);
        if gpu.flash_static_setting == Some(true) {
            ensure!(
                static_submissions[0] > 0 && static_submissions[3] + static_submissions[4] > 0,
                "static K target did not execute both cold K2 and resident K4"
            );
        } else {
            ensure!(static_submissions == [0; 5]);
        }
        eprintln!(
            "NATIVE_FLASH_STATIC_SUBMISSIONS {}",
            serde_json::json!(static_submissions)
        );
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
