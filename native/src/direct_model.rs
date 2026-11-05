//! Family dispatch for the common production bridge. Flash-only features stay explicit.
use super::{ChatMtpHead, RoundDraft, prefix, prompt_cache_budget};
use anyhow::{Context, Result, bail, ensure};
use mlxl3_native::{
    checkpoint::Checkpoint,
    gpu::{FlashMtpSession, FlashNext, FlashTraceTensor, Metal, Qwen35, QwenSnapshot},
};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, Ordering};

pub(crate) enum NativeChatModel<'a> {
    Flash(Box<FlashNext<'a>>),
    Qwen(Box<Qwen35<'a>>),
}
impl<'a> NativeChatModel<'a> {
    pub(crate) fn load(gpu: &'a Metal, c: &Checkpoint, context: u32) -> Result<Self> {
        match c.model_type.as_str() {
            "qwen4_exp" => Ok(Self::Flash(Box::new(FlashNext::load(gpu, c, context)?))),
            "qwen3_5_moe" => Ok(Self::Qwen(Box::new(Qwen35::load(gpu, c, context)?))),
            other => bail!("native Metal inference does not support {other:?}"),
        }
    }
    pub(crate) fn is_flash(&self) -> bool {
        matches!(self, Self::Flash(_))
    }
    pub(crate) fn name(&self) -> &'static str {
        match self {
            Self::Flash(_) => "Flash Next / direct Metal",
            Self::Qwen(_) => "Qwen3.6-35B-A3B / direct Metal",
        }
    }
    pub(crate) fn flash(&self) -> Result<&FlashNext<'a>> {
        match self {
            Self::Flash(f) => Ok(f),
            _ => bail!("native Qwen MTP/DFlash is not supported yet"),
        }
    }
    pub(crate) fn flash_mut(&mut self) -> Result<&mut FlashNext<'a>> {
        match self {
            Self::Flash(f) => Ok(f),
            _ => bail!("native Qwen MTP/DFlash is not supported yet"),
        }
    }
    pub(crate) fn context_limit(&self) -> u32 {
        match self {
            Self::Flash(f) => f.context_limit(),
            Self::Qwen(q) => q.context_limit(),
        }
    }
    pub(crate) fn reset(&mut self) {
        match self {
            Self::Flash(f) => f.reset(),
            Self::Qwen(q) => q.reset(),
        }
    }
    pub(crate) fn forward(&mut self, t: u32) -> Result<Vec<u16>> {
        match self {
            Self::Flash(f) => f.forward(t),
            Self::Qwen(q) => q.forward(t),
        }
    }
    pub(crate) fn device_greedy_eligible(
        &self,
        temperature: f32,
        top_k: usize,
        penalty: f32,
        speculative: bool,
    ) -> bool {
        self.flash()
            .is_ok_and(|f| f.device_greedy_eligible(temperature, top_k, penalty, speculative))
    }
    pub(crate) fn device_greedy_counts(&self) -> [u64; 2] {
        self.flash().map_or([0; 2], FlashNext::device_greedy_counts)
    }
    pub(crate) fn greedy_from_logits(&self, logits: &[u16]) -> Result<u32> {
        self.flash()?.greedy_from_logits(logits)
    }
    pub(crate) fn forward_greedy(&mut self, token: u32) -> Result<u32> {
        self.flash_mut()?.forward_greedy(token)
    }
    pub(crate) fn forward_trace(&mut self, t: u32) -> Result<(Vec<u16>, Vec<FlashTraceTensor>)> {
        match self {
            Self::Flash(f) => f.forward_trace(t),
            Self::Qwen(q) => q.forward_trace(t),
        }
    }
    pub(crate) fn prefill(&mut self, t: &[u32]) -> Result<Vec<u16>> {
        match self {
            Self::Flash(f) => f.prefill(t),
            Self::Qwen(q) => q.prefill(t),
        }
    }
    pub(crate) fn log_probs(&self, l: &[u16]) -> Result<Vec<f32>> {
        match self {
            Self::Flash(f) => f.log_probs(l),
            Self::Qwen(q) => q.log_probs(l),
        }
    }
    pub(crate) fn memory_stats(&self) -> Result<mlxl3_native::runtime_stats::MemoryStats> {
        match self {
            Self::Flash(f) => f.memory_stats(),
            Self::Qwen(q) => q.memory_stats(),
        }
    }
    pub(crate) fn request_memory_limit(&self) -> Option<u64> {
        match self {
            Self::Flash(f) => f.request_memory_limit(),
            Self::Qwen(_) => None,
        }
    }
    pub(crate) fn admit_request(
        &mut self,
        positions: usize,
    ) -> Result<Option<mlxl3_native::flash_memory::ResourcePlan>> {
        match self {
            Self::Flash(f) => f.admit_request(positions),
            Self::Qwen(q) => {
                ensure!(
                    positions <= q.context_limit() as usize,
                    "Qwen request exceeds context"
                );
                Ok(None)
            }
        }
    }
    pub(crate) fn memory_ledger(&self, head: Option<&ChatMtpHead>) -> Result<Value> {
        match self {
            Self::Flash(f) => Ok(serde_json::to_value(f.memory_ledger(head)?)?),
            Self::Qwen(q) => {
                ensure!(head.is_none(), "unexpected native Qwen speculative head");
                Ok(
                    json!({"backend":"direct-metal","model_family":"qwen3_5_moe","memory":q.memory_stats()?,"active_state_payload_bytes":q.snapshot().retained_bytes()}),
                )
            }
        }
    }
    pub(crate) fn release_idle_state(&mut self) -> Result<()> {
        match self {
            Self::Flash(f) => f.release_idle_state(),
            Self::Qwen(q) => {
                q.reset();
                Ok(())
            }
        }
    }
}

pub(crate) enum PromptCache {
    Flash(prefix::PromptCache),
    Qwen(QwenCache),
}
pub(crate) struct QwenCache {
    conversation: String,
    tokens: Vec<u32>,
    logits: Vec<u16>,
    target: QwenSnapshot,
}
impl QwenCache {
    fn bytes(&self) -> Option<usize> {
        self.target
            .retained_bytes()
            .checked_add(self.conversation.capacity())?
            .checked_add(self.tokens.capacity().checked_mul(4)?)?
            .checked_add(self.logits.capacity().checked_mul(2)?)
    }
}
pub(crate) fn ledger(
    model: &NativeChatModel,
    head: Option<&ChatMtpHead>,
    draft: Option<&RoundDraft>,
    cache: Option<&PromptCache>,
) -> Result<Value> {
    match model {
        NativeChatModel::Flash(f) => prefix::ledger(
            f,
            head,
            draft,
            cache.and_then(|c| match c {
                PromptCache::Flash(c) => Some(c),
                _ => None,
            }),
        ),
        NativeChatModel::Qwen(_) => {
            let mut ledger = model.memory_ledger(head)?;
            ledger["prompt_cache"] = match cache {
                Some(PromptCache::Qwen(c)) => {
                    json!({"checkpoint_tokens":c.tokens.len(),"retained_bytes_conservative":c.bytes().context("Qwen cache extent overflow")?})
                }
                _ => Value::Null,
            };
            Ok(ledger)
        }
    }
}
pub(crate) fn admit_request(
    model: &mut NativeChatModel,
    positions: usize,
    cache: &mut Option<PromptCache>,
    reuse: bool,
) -> Result<Option<mlxl3_native::flash_memory::ResourcePlan>> {
    if !reuse {
        *cache = None;
    }
    if let NativeChatModel::Flash(f) = model {
        let mut saved = match cache.take() {
            Some(PromptCache::Flash(c)) => Some(c),
            _ => None,
        };
        let result = prefix::admit_request(f, positions, &mut saved, reuse);
        *cache = saved.map(PromptCache::Flash);
        result
    } else {
        model.admit_request(positions)
    }
}
#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill(
    model: &mut NativeChatModel,
    head: Option<&mut ChatMtpHead>,
    draft: Option<&RoundDraft>,
    tokens: &[u32],
    depth: usize,
    cache: &mut Option<PromptCache>,
    conversation: &str,
    reuse: bool,
    cancelled: &AtomicBool,
) -> Result<(Vec<u16>, usize, Option<FlashMtpSession>)> {
    if let NativeChatModel::Flash(f) = model {
        let mut saved = match cache.take() {
            Some(PromptCache::Flash(c)) => Some(c),
            _ => None,
        };
        let result = prefix::prefill(
            f,
            head,
            draft,
            tokens,
            depth,
            &mut saved,
            conversation,
            reuse,
            cancelled,
        );
        *cache = saved.map(PromptCache::Flash);
        return result;
    }
    let NativeChatModel::Qwen(q) = model else {
        unreachable!()
    };
    let previous = cache
        .take()
        .and_then(|c| match c {
            PromptCache::Qwen(c) => Some(c),
            _ => None,
        })
        .filter(|c| {
            mlxl3_native::native_prompt_reuse(
                &c.tokens,
                tokens,
                c.conversation == conversation,
                head.is_none() && draft.is_none(),
                reuse,
            ) > 0
        });
    let operation = (|| {
        ensure!(
            head.is_none() && draft.is_none(),
            "native Qwen MTP/DFlash is not supported yet"
        );
        ensure!(!tokens.is_empty(), "empty native Qwen prompt");
        ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
        let (mut logits, cached) = if let Some(saved) = previous {
            q.restore(&saved.target)?;
            (Some(saved.logits), saved.tokens.len())
        } else {
            q.reset();
            (None, 0)
        };
        for &token in &tokens[cached..] {
            ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
            logits = Some(q.forward(token)?);
        }
        ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
        let logits = logits.context("empty native Qwen suffix")?;
        ensure!(
            logits.len() == 248320 && logits.iter().all(|&b| half::f16::from_bits(b).is_finite()),
            "invalid Qwen cached logits"
        );
        if reuse {
            let budget = prompt_cache_budget()?;
            let saved = QwenCache {
                conversation: conversation.to_owned(),
                tokens: tokens.to_vec(),
                logits: logits.clone(),
                target: q.snapshot(),
            };
            if saved.bytes().is_some_and(|n| n <= budget) {
                *cache = Some(PromptCache::Qwen(saved));
            }
        }
        Ok((logits, cached, None))
    })();
    if operation.is_err() {
        *cache = None;
        q.reset();
    }
    operation
}
