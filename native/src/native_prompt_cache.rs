//! One owned prompt checkpoint; speculative decode state is never cached.
use super::{RoundDraft, prompt_cache_budget};
use anyhow::{Context, Result, bail, ensure};
use mlxl3_native::gpu::FlashNext as NativeChatModel;
use mlxl3_native::gpu::{FlashMtpHead, FlashMtpSession, FlashQsaState, FlashSnapshot};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, Ordering};

enum AuxiliaryPrefix {
    Ar,
    Mtp(FlashQsaState),
    Draft(mlxl3_native::native_draft::Prefix),
}
pub(crate) struct PromptCache {
    conversation: String,
    tokens: Vec<u32>,
    target: FlashSnapshot,
    logits: Vec<u16>,
    auxiliary: AuxiliaryPrefix,
}
impl PromptCache {
    fn components(&self) -> Result<[usize; 6]> {
        let (draft, mtp) = match &self.auxiliary {
            AuxiliaryPrefix::Ar => (0, 0),
            AuxiliaryPrefix::Mtp(prefix) => (0, usize::try_from(prefix.retained_bytes()?)?),
            AuxiliaryPrefix::Draft(prefix) => (prefix.retained_bytes()?, 0),
        };
        Ok([
            usize::try_from(self.target.retained_bytes()?)?,
            self.logits
                .capacity()
                .checked_mul(2)
                .context("cached logits overflow")?,
            draft,
            mtp,
            self.tokens
                .capacity()
                .checked_mul(4)
                .context("cached token overflow")?,
            self.conversation.capacity(),
        ])
    }
    fn summary(&self) -> Result<Value> {
        let parts = self.components()?;
        Ok(
            json!({"checkpoint_tokens":self.tokens.len(),"target_metal_backing_bytes":parts[0],"logits_host_payload_bytes":parts[1],"draft_host_state_bytes":parts[2],"mtp_metal_backing_bytes":parts[3],"tokens_host_payload_bytes":parts[4],"conversation_host_bytes":parts[5],"retained_bytes_conservative":parts.into_iter().try_fold(0usize,usize::checked_add).context("prefix size overflow")?,"budget_bytes":prompt_cache_budget()?}),
        )
    }
    fn capture(
        model: &NativeChatModel,
        head: Option<&FlashMtpHead>,
        draft: Option<&RoundDraft>,
        conversation: &str,
        tokens: &[u32],
        logits: &[u16],
    ) -> Result<Option<Self>> {
        let budget = prompt_cache_budget()?;
        if budget == 0 {
            return Ok(None);
        }
        ensure!(
            model.offset() as usize == tokens.len()
                && logits.len() == 248320
                && logits.iter().all(|&b| half::f16::from_bits(b).is_finite()),
            "invalid native prompt cache checkpoint"
        );
        let target = model.snapshot();
        if usize::try_from(target.retained_bytes()?)? > budget {
            return Ok(None);
        }
        // Target/MTP snapshots own handles, not deep copies. Admit the complete
        // prospective payload before cloning the potentially 80 MiB draft ring.
        let mtp = head.map(FlashMtpHead::snapshot_prefix);
        ensure!(
            !(head.is_some() && draft.is_some()),
            "cannot cache two speculative modes"
        );
        let parts = [
            usize::try_from(target.retained_bytes()?)?,
            logits
                .len()
                .checked_mul(2)
                .context("cached logits overflow")?,
            draft
                .map(RoundDraft::prefix_clone_bytes)
                .transpose()?
                .unwrap_or(0),
            mtp.as_ref()
                .map(|prefix| prefix.retained_bytes())
                .transpose()?
                .map(usize::try_from)
                .transpose()?
                .unwrap_or(0),
            tokens
                .len()
                .checked_mul(4)
                .context("cached token overflow")?,
            conversation.len(),
        ];
        mlxl3_native::memory_policy::capture_if_fits(parts, budget, || -> Result<Option<Self>> {
            let auxiliary = match (mtp, draft) {
                (None, None) => AuxiliaryPrefix::Ar,
                (Some(prefix), None) => AuxiliaryPrefix::Mtp(prefix),
                (None, Some(draft)) => AuxiliaryPrefix::Draft(draft.snapshot_prefix(model)?),
                _ => bail!("cannot cache two speculative modes"),
            };
            let cache = Self {
                conversation: conversation.to_owned(),
                tokens: tokens.to_vec(),
                target,
                logits: logits.to_vec(),
                auxiliary,
            };
            Ok(
                mlxl3_native::memory_policy::cache_fits(cache.components()?, budget)
                    .then_some(cache),
            )
        })
        .map(Option::flatten)
    }

    fn matches_mode(&self, head: bool, draft: bool) -> bool {
        matches!(
            (&self.auxiliary, head, draft),
            (AuxiliaryPrefix::Ar, false, false)
                | (AuxiliaryPrefix::Mtp(_), true, false)
                | (AuxiliaryPrefix::Draft(_), false, true)
        )
    }
}
pub(super) fn ledger(
    model: &NativeChatModel,
    head: Option<&FlashMtpHead>,
    draft: Option<&RoundDraft>,
    cache: Option<&PromptCache>,
) -> Result<Value> {
    let target = cache.map(|c| &c.target);
    let mtp = cache.and_then(|c| match &c.auxiliary {
        AuxiliaryPrefix::Mtp(prefix) => Some(prefix),
        _ => None,
    });
    let mut ledger = serde_json::to_value(model.memory_ledger_with_prefix(head, target, mtp)?)?;
    ledger["prompt_cache"] = cache
        .map(PromptCache::summary)
        .transpose()?
        .unwrap_or(Value::Null);
    ledger["draft_host_payload"] = if let Some(draft) = draft {
        serde_json::to_value(draft.memory_payload()?)?
    } else {
        Value::Null
    };
    Ok(ledger)
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
    let admitted = model.admit_request(positions);
    if admitted.is_err() && cache.is_some() {
        // An optional saved prefix must not be the reason an otherwise admissible request fails.
        *cache = None;
        model.admit_request(positions)
    } else {
        admitted
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prefill(
    model: &mut NativeChatModel,
    mut head: Option<&mut FlashMtpHead>,
    draft: Option<&RoundDraft>,
    tokens: &[u32],
    depth: usize,
    cache: &mut Option<PromptCache>,
    conversation: &str,
    reuse: bool,
    cancelled: &AtomicBool,
) -> Result<(Vec<u16>, usize, Option<FlashMtpSession>)> {
    if cancelled.load(Ordering::Relaxed) || (head.is_some() && depth != 1) {
        *cache = None;
        model.reset();
        if let Some(head) = head.as_deref_mut() {
            head.reset();
        }
        if let Some(draft) = draft {
            draft.reset();
        }
        ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
        bail!("Flash MTP supports depth 1");
    }
    let previous = cache.take().filter(|saved| {
        mlxl3_native::native_prompt_reuse(
            &saved.tokens,
            tokens,
            saved.conversation == conversation,
            saved.matches_mode(head.is_some(), draft.is_some()),
            reuse,
        ) > 0
    });
    let mut session = head.as_ref().map(|_| FlashMtpSession::default());
    let operation = (|| {
        let (mut logits, cached) = if let Some(saved) = previous {
            model.restore(&saved.target)?;
            match saved.auxiliary {
                AuxiliaryPrefix::Ar => {}
                AuxiliaryPrefix::Mtp(prefix) => head
                    .as_deref_mut()
                    .context("missing cached MTP head")?
                    .restore_prefix(&prefix)?,
                AuxiliaryPrefix::Draft(prefix) => draft
                    .context("missing cached draft")?
                    .restore_prefix(model, &prefix)?,
            }
            (Some(saved.logits), saved.tokens.len())
        } else {
            model.reset();
            if let Some(head) = head.as_deref_mut() {
                head.reset();
            }
            if let Some(draft) = draft {
                draft.reset();
            }
            (None, 0)
        };
        let suffix = &tokens[cached..];
        if !suffix.is_empty() {
            logits = Some(if let Some(draft) = draft {
                if cached == 0 {
                    draft.prefill(model, suffix, cancelled)?
                } else {
                    draft.extend_prefill(model, suffix, cancelled)?
                }
            } else if let Some((head, session)) = head.as_deref_mut().zip(session.as_mut()) {
                if cached == 0 {
                    session.prefill(model, head, suffix, cancelled)?
                } else {
                    session.extend_prefill(model, head, suffix, cancelled)?
                }
            } else {
                let mut last = None;
                for chunk in suffix.chunks(mlxl3_native::flash_support::PREFILL_CHUNK) {
                    ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
                    last = Some(model.prefill(chunk)?);
                }
                last.context("empty native prompt suffix")?
            });
        }
        let logits = logits.context("empty native prompt")?;
        ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
        if reuse {
            *cache =
                PromptCache::capture(model, head.as_deref(), draft, conversation, tokens, &logits)?;
        }
        Ok((logits, cached, session))
    })();
    if operation.is_err() {
        *cache = None;
        model.reset();
        if let Some(head) = head {
            head.reset();
        }
        if let Some(draft) = draft {
            draft.reset();
        }
    }
    operation
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlxl3_native::{
        checkpoint,
        gpu::{FlashMtpHead, FlashNext, FlashTraceTensor, Metal},
    };
    fn same_trace(expected: &[FlashTraceTensor], actual: &[FlashTraceTensor]) -> Result<usize> {
        ensure!(expected.len() == 84 && actual.len() == 84);
        let mut words = 0;
        for (a, b) in expected.iter().zip(actual) {
            ensure!(
                a.name == b.name && a.shape == b.shape && a.data == b.data,
                "cached continuation state differs: {}",
                a.name
            );
            let count = a
                .shape
                .iter()
                .try_fold(1usize, |n, &v| n.checked_mul(v as usize))
                .context("state shape overflow")?;
            ensure!(
                count > 0
                    && a.data.len() == count * 4
                    && a.data
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .all(|v| f32::from_le_bytes(*v).is_finite())
            );
            words += count;
        }
        ensure!(words == 28_803_072);
        Ok(words)
    }
    #[test]
    #[ignore = "physical Flash target: owned native AR/MTP/DFlash prefix reuse, continuation and cleanup"]
    fn native_prefix_reuse_restores_states_and_releases_owned_checkpoints() -> Result<()> {
        let checkpoint =
            checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut model = FlashNext::load(&gpu, &checkpoint, 128)?;
        let cancelled = AtomicBool::new(false);
        let inputs = [vec![248044, 1, 42], vec![248044, 1, 42, 7, 9]];
        let mut controls = Vec::new();
        for tokens in &inputs {
            model.reset();
            let logits = model.prefill(tokens)?;
            ensure!(
                logits.len() == 248320
                    && logits.iter().all(|&v| half::f16::from_bits(v).is_finite())
            );
            let (continuation, states) = model.forward_trace(11)?;
            controls.push((logits, continuation, states));
        }
        let permitted = match std::env::var("MLXL3_EXPECT_PREFIX_CACHE")?.as_str() {
            "0" => false,
            "1" => true,
            _ => bail!("MLXL3_EXPECT_PREFIX_CACHE must be 0 or 1"),
        };
        let mut cache = None;
        let mut comparisons = 0;
        let mut reused = 0;
        for mode in 0..if permitted { 3 } else { 1 } {
            let mut head = if mode == 1 {
                Some(FlashMtpHead::load(&checkpoint, &model)?)
            } else {
                None
            };
            let draft = if mode == 2 {
                Some(RoundDraft::load(
                    std::path::Path::new(&std::env::var("MLXL3_FLASH_DRAFT_MODEL")?),
                    1,
                    &model,
                )?)
            } else {
                None
            };
            for (index, tokens) in inputs.iter().enumerate() {
                let (logits, cached, _) = prefill(
                    &mut model,
                    head.as_mut(),
                    draft.as_ref(),
                    tokens,
                    1,
                    &mut cache,
                    "conversation",
                    true,
                    &cancelled,
                )?;
                ensure!(logits == controls[index].0);
                ensure!(
                    cached
                        == if permitted && index > 0 {
                            inputs[index - 1].len()
                        } else {
                            0
                        }
                );
                reused += usize::from(cached > 0);
                ensure!(cache.is_some() == permitted);
                if let Some(cache) = &cache {
                    ensure!(
                        cache.target.position() as usize == tokens.len()
                            && cache.components()?[0] > 0
                    );
                    let active = serde_json::to_value(model.memory_ledger(head.as_ref())?)?;
                    let owned = ledger(&model, head.as_ref(), draft.as_ref(), Some(cache))?;
                    ensure!(
                        active["state_backing_bytes"] == owned["state_backing_bytes"],
                        "an alias was counted twice"
                    );
                    ensure!(owned["prompt_cache"]["checkpoint_tokens"] == tokens.len());
                    if draft.is_some() {
                        ensure!(
                            owned["draft_host_payload"]["weights_bytes"]
                                .as_u64()
                                .is_some_and(|n| n > 0)
                                && owned["prompt_cache"]["draft_host_state_bytes"]
                                    .as_u64()
                                    .is_some_and(|n| n > 0)
                        );
                    }
                }
                let (again, cached, _) = prefill(
                    &mut model,
                    head.as_mut(),
                    draft.as_ref(),
                    tokens,
                    1,
                    &mut cache,
                    "conversation",
                    true,
                    &cancelled,
                )?;
                ensure!(
                    again == controls[index].0
                        && cached == if permitted { tokens.len() } else { 0 }
                );
                reused += usize::from(cached > 0);
                let (continuation, states) = model.forward_trace(11)?;
                ensure!(continuation == controls[index].1);
                comparisons += same_trace(&controls[index].2, &states)?;
            }
            let (logits, cached, _) = prefill(
                &mut model,
                head.as_mut(),
                draft.as_ref(),
                &inputs[1],
                1,
                &mut cache,
                "another conversation",
                true,
                &cancelled,
            )?;
            ensure!(logits == controls[1].0 && cached == 0);
            cancelled.store(true, Ordering::Relaxed);
            ensure!(
                prefill(
                    &mut model,
                    head.as_mut(),
                    draft.as_ref(),
                    &inputs[1],
                    1,
                    &mut cache,
                    "another conversation",
                    true,
                    &cancelled
                )
                .is_err()
            );
            ensure!(cache.is_none() && model.offset() == 0);
            cancelled.store(false, Ordering::Relaxed);
            ensure!(
                prefill(
                    &mut model,
                    head.as_mut(),
                    draft.as_ref(),
                    &[248320],
                    1,
                    &mut cache,
                    "conversation",
                    true,
                    &cancelled
                )
                .is_err()
            );
            ensure!(cache.is_none() && model.offset() == 0);
            let (logits, cached, _) = prefill(
                &mut model,
                head.as_mut(),
                draft.as_ref(),
                &inputs[0],
                1,
                &mut cache,
                "conversation",
                false,
                &cancelled,
            )?;
            ensure!(logits == controls[0].0 && cached == 0 && cache.is_none());
        }
        if permitted {
            prefill(
                &mut model,
                None,
                None,
                &inputs[0],
                1,
                &mut cache,
                "conversation",
                true,
                &cancelled,
            )?;
            model.release_idle_state()?;
            let held = ledger(&model, None, None, cache.as_ref())?;
            ensure!(
                held["state_backing_bytes"]["prompt_cache"]
                    .as_u64()
                    .is_some_and(|n| n > 0)
            );
            let (logits, cached, _) = prefill(
                &mut model,
                None,
                None,
                &inputs[0],
                1,
                &mut cache,
                "conversation",
                true,
                &cancelled,
            )?;
            ensure!(cached == inputs[0].len() && logits == controls[0].0);
            drop(model);
            let mut foreign = FlashNext::load(&gpu, &checkpoint, 128)?;
            ensure!(
                prefill(
                    &mut foreign,
                    None,
                    None,
                    &inputs[0],
                    1,
                    &mut cache,
                    "conversation",
                    true,
                    &cancelled
                )
                .is_err()
            );
            ensure!(cache.is_none() && foreign.offset() == 0);
        }
        ensure!(reused == if permitted { 9 } else { 0 });
        eprintln!(
            "native prefix {}",
            json!({"cache_permitted":permitted,"reused_checkpoints":reused,"finite_state_word_comparisons":comparisons,"modes":if permitted {3} else {1}})
        );
        Ok(())
    }
}
