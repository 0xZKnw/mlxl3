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
use std::{os::unix::fs::FileExt, rc::Rc};

#[path = "mtp.rs"]
mod mtp;
pub use mtp::{FlashMtpHead, FlashMtpSession};

enum Layer<'a> {
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
impl Layer<'_> {
    fn forward(&mut self, x: &MatrixView, token: u32) -> Result<MatrixView> {
        match self {
            Self::Gdn(a) => a.forward(x),
            Self::Ple(a) => a.forward(x, token),
            Self::Qsa(a) => a.forward(x),
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

#[derive(serde::Serialize)]
pub struct FlashTraceTensor {
    pub name: String,
    pub shape: Vec<u32>,
    pub data: Vec<u8>,
}
pub struct FlashNext<'a> {
    checkpoint_path: std::path::PathBuf,
    owner: Rc<()>,
    hidden: Option<MatrixView>,
    gpu: &'a Metal,
    embed: TensorInfo,
    layers: Vec<Layer<'a>>,
    mixer: FlashHc<'a>,
    head: FlashProjection<'a>,
    context: u32,
    offset: u32,
}
impl<'a> FlashNext<'a> {
    pub fn load(gpu: &'a Metal, c: &Checkpoint, context: u32) -> Result<Self> {
        let config = FlashConfig::load(c)?;
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
        ensure!(
            std::env::var_os("MLXL3_FLASH_REQUEST_MIB").is_none(),
            "native Flash request memory budgets are not available for this build"
        );
        let slots = layer_slots()?;
        let embed = c
            .tensors
            .get("model.language_model.embed_tokens.weight")
            .context("missing token embedding")?
            .clone();
        ensure!(
            embed.dtype == "BF16" && embed.shape == [248320, 2560],
            "unsupported token embedding"
        );
        let layers = slots
            .into_iter()
            .enumerate()
            .map(|(i, slots)| -> Result<_> {
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
        Ok(Self {
            checkpoint_path: c.path.clone(),
            owner: Rc::new(()),
            hidden: None,
            gpu,
            embed,
            layers,
            mixer,
            head,
            context,
            offset: 0,
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
        unsafe extern "C" {
            fn mlxl3_read_process_stats(values: *mut u64, count: usize) -> i32;
        }
        let mut host = [0u64; 9];
        ensure!(
            unsafe { mlxl3_read_process_stats(host.as_mut_ptr(), host.len()) } == 0,
            "invalid native process statistics extent"
        );
        Ok(crate::runtime_stats::MemoryStats::native(
            host,
            self.gpu.device.current_allocated_size(),
            std::process::id(),
        ))
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
    ) -> Result<MatrixView> {
        let mut streams = self.streams(token)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            streams = layer
                .forward(&streams, token)
                .with_context(|| format!("running Flash layer {i}"))?;
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

    pub fn forward_trace(&mut self, token: u32) -> Result<(Vec<u16>, Vec<FlashTraceTensor>)> {
        let mut trace = Vec::with_capacity(84);
        let logits = self.prefill_inner(&[token], Some(&mut trace))?;
        Ok((logits, trace))
    }
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>> {
        self.prefill_inner(tokens, None)
    }
    fn prefill_inner(
        &mut self,
        tokens: &[u32],
        trace: Option<&mut Vec<FlashTraceTensor>>,
    ) -> Result<Vec<u16>> {
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
                self.hidden(token, None)?;
            }
            let streams = self.hidden(last, trace)?;
            let mixed = self.mixer.read(&streams, false)?.mixed;
            let logits = self.head.submit_fp16_view(&mixed)?.read()?;
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
