use super::{FlashHc, FlashNext, FlashProjection, FlashQsaLayer, MatrixLayout, MatrixView, Metal};
use crate::{
    checkpoint::Checkpoint,
    flash_config::FlashConfig,
    gpu::{checkpoint_float_weight, view::FloatType},
};
use anyhow::{Context, Result, ensure};
use metal::{Buffer, MTLResourceOptions as Storage, MTLSize};
use std::rc::Rc;

pub struct FlashMtpHead<'a> {
    owner: Rc<()>,
    embedding_gain: Buffer,
    hidden_gain: Buffer,
    embedding_fc: FlashProjection<'a>,
    hidden_fc: FlashProjection<'a>,
    layer: FlashQsaLayer<'a>,
    mixer: FlashHc<'a>,
    eps: f32,
    device_greedy: bool,
    block_verify: bool,
}

impl<'a> FlashMtpHead<'a> {
    pub fn load(c: &Checkpoint, target: &FlashNext<'a>) -> Result<Self> {
        ensure!(
            crate::flash_memory::native_speculation_permitted(target.request_memory_limit()),
            "Flash request memory budget supports AR only"
        );
        ensure!(
            target.checkpoint_path == c.path,
            "MTP requires the target checkpoint"
        );
        let config = FlashConfig::load(c)?;
        let metadata: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(c.path.join("config.json"))?)?;
        ensure!(
            metadata["text_config"]["mtp_num_hidden_layers"] == 1,
            "Flash MTP requires one draft layer"
        );
        let gpu = target.gpu;
        let experimental = |name: &str| -> Result<bool> {
            let value = match std::env::var(name) {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(error) => return Err(error.into()),
            };
            crate::flash_support::hc_kernel_flag(value.as_deref(), true)
                .with_context(|| format!("{name} must be 0 or 1"))
        };
        let gain = |name, width| checkpoint_float_weight(gpu, c, name, &[width], FloatType::F32);
        let embedding_fc = FlashProjection::load(gpu, c, "mtp.fc_embedding")?;
        let hidden_fc = FlashProjection::load(gpu, c, "mtp.fc_hidden")?;
        ensure!(
            embedding_fc.dimensions() == [2560, 2560] && hidden_fc.dimensions() == [2560, 2560],
            "invalid Flash MTP FC geometry"
        );
        Ok(Self {
            owner: target.owner.clone(),
            embedding_gain: gain("mtp.pre_fc_norm_embedding.weight", 2560)?,
            hidden_gain: gain("mtp.pre_fc_norm_hidden.weight", 10240)?,
            embedding_fc,
            hidden_fc,
            layer: FlashQsaLayer::load(
                gpu,
                c,
                "mtp.layers.0",
                config.eps,
                config.rope,
                target.context,
                512,
            )?,
            mixer: FlashHc::load(gpu, c, "mtp.hyper_connection_mixer", config.eps, false)?,
            eps: config.eps,
            device_greedy: experimental("MLXL3_NATIVE_MTP_DEVICE_GREEDY")?
                && gpu.device_greedy_available(),
            block_verify: experimental("MLXL3_NATIVE_MTP_BLOCK_VERIFY")?,
        })
    }

    fn mixed(&self, target: &FlashNext, hidden: &MatrixView, token: u32) -> Result<MatrixView> {
        let gpu = target.gpu;
        ensure!(
            Rc::ptr_eq(&self.owner, &target.owner),
            "MTP belongs to a different target"
        );
        ensure!(
            hidden.shape() == [1, 10240]
                && hidden.dtype == FloatType::F32
                && hidden.layout.offset == 0
                && hidden.layout.strides[1] == 1
                && hidden.buffer.device().registry_id() == gpu.device.registry_id(),
            "invalid Flash MTP hidden row"
        );
        let embedding = gpu
            .f16_storage(&target.embedding_bits(token)?)?
            .matrix(MatrixLayout {
                shape: [1, 2560],
                strides: [2560, 1],
                offset: 0,
            })?;
        let private = Storage::StorageModePrivate;
        #[cfg(test)]
        let capture =
            std::env::var_os("MLXL3_FLASH_MTP_DIRECT_PARTS").map(std::path::PathBuf::from);
        #[cfg(test)]
        let mut parts = Vec::new();
        let result = gpu.command(|cmd| {
            let norm =
                gpu.encode_stream_norm(cmd, &embedding, &self.embedding_gain, self.eps, private)?;
            let projected_embedding = self.embedding_fc.encode_row(cmd, &norm, private)?;
            let mut normalized = MatrixView::output(gpu, FloatType::F32, 1, 10240, private)?;
            ensure!(
                gpu.mtp_norm.thread_execution_width() == 32
                    && gpu.mtp_norm.max_total_threads_per_threadgroup() >= 1024,
                "unsupported Flash MTP reduction"
            );
            Metal::encode(
                cmd,
                &gpu.mtp_norm,
                &[&hidden.buffer, &self.hidden_gain, &normalized.buffer],
                [self.eps.to_bits()],
                MTLSize::new(1, 1, 1),
                1024,
            );
            normalized.layout = MatrixLayout {
                shape: [4, 2560],
                strides: [2560, 1],
                offset: 0,
            };
            let projected_hidden = self.hidden_fc.encode_rows(cmd, &normalized, private)?;
            let mixed = MatrixView::output(gpu, FloatType::F16, 1, 10240, private)?;
            Metal::encode(
                cmd,
                &gpu.mtp_combine,
                &[
                    &projected_embedding.buffer,
                    &projected_hidden.buffer,
                    &mixed.buffer,
                ],
                [2560u32],
                MTLSize::new(320, 1, 1),
                32,
            );
            #[cfg(test)]
            if capture.is_some() {
                parts.extend([
                    ("embedding", embedding),
                    ("embedding_norm", norm),
                    ("embedding_fc", projected_embedding),
                    ("hidden_norm", normalized),
                    ("hidden_fc", projected_hidden),
                    ("mixed", mixed.clone()),
                ]);
            }
            Ok(mixed)
        })?;
        #[cfg(test)]
        if let Some(dir) = capture {
            for (name, row) in parts {
                std::fs::write(
                    dir.join(format!("{token}-{name}.bin")),
                    row.export_bytes(gpu)?,
                )?;
            }
        }
        Ok(result)
    }
    pub fn forward(
        &mut self,
        target: &FlashNext,
        hidden: &MatrixView,
        token: u32,
    ) -> Result<Vec<u16>> {
        let streams = self.mixed(target, hidden, token)?;
        let before = self.layer.snapshot();
        let result = (|| {
            let output = self.layer.forward(&streams)?;
            let mixed = self.mixer.read(&output, false)?.mixed;
            target.head.submit_fp16_view(&mixed)?.read()
        })();
        if result.is_err() {
            self.layer.restore(&before)?;
        }
        result
    }

    /// Normalized Half selection is identical to the canonical host selector.
    /// This remains separately opt-in from ordinary AR device greedy admission.
    fn propose(&mut self, target: &FlashNext, hidden: &MatrixView, token: u32) -> Result<u32> {
        if !self.device_greedy {
            return target.greedy_id(&self.forward(target, hidden, token)?);
        }
        let streams = self.mixed(target, hidden, token)?;
        let before = self.layer.snapshot();
        let result = (|| {
            let output = self.layer.forward(&streams)?;
            let mixed = self.mixer.read(&output, false)?.mixed;
            target
                .gpu
                .device_greedy_head(&target.head, &mixed, false)
                .map(|v| v.0)
        })();
        if result.is_err() {
            self.layer.restore(&before)?;
        }
        result
    }

    pub fn reset(&mut self) {
        self.layer.reset();
    }
}

impl FlashNext<'_> {
    fn greedy_id(&self, logits: &[u16]) -> Result<u32> {
        let normalized = self.gpu.flash_log_probs(logits)?;
        crate::speculative::greedy_log_prob_id(&normalized)
            .context("invalid Flash MTP probabilities")
    }
}

/// Greedy verification retains at most one already-verified target token.
#[derive(Default)]
pub struct FlashMtpSession {
    pending: Option<(u32, u32, MatrixView)>,
    pub proposed: usize,
    pub accepted: usize,
}

impl FlashMtpHead<'_> {
    pub fn snapshot_prefix(&self) -> super::FlashQsaState {
        self.layer.snapshot()
    }
    pub(in crate::gpu) fn owns_prefix(&self, state: &super::FlashQsaState) -> bool {
        self.layer.owns_prefix(state)
    }
    pub fn restore_prefix(&mut self, state: &super::FlashQsaState) -> Result<()> {
        self.layer.restore(state)
    }
    fn extend(&mut self, target: &FlashNext, hidden: &MatrixView, token: u32) -> Result<()> {
        self.layer.extend(&self.mixed(target, hidden, token)?)
    }
}

impl FlashMtpSession {
    pub fn prefill(
        &mut self,
        target: &mut FlashNext,
        head: &mut FlashMtpHead,
        tokens: &[u32],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<Vec<u16>> {
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= target.context as usize
                && tokens.iter().all(|&t| t < 248320),
            "invalid Flash MTP prefill"
        );
        ensure!(
            Rc::ptr_eq(&target.owner, &head.owner),
            "MTP belongs to a different target"
        );
        *self = Self::default();
        target.reset();
        head.reset();
        self.extend_prefill(target, head, tokens, cancelled)
    }
    /// Append only a real prompt suffix after restoring the paired target/head prefix.
    pub fn extend_prefill(
        &mut self,
        target: &mut FlashNext,
        head: &mut FlashMtpHead,
        tokens: &[u32],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<Vec<u16>> {
        ensure!(
            !tokens.is_empty()
                && tokens.iter().all(|&t| t < 248320)
                && usize::try_from(target.offset())
                    .ok()
                    .and_then(|n| n.checked_add(tokens.len()))
                    .is_some_and(|end| end <= target.context as usize),
            "invalid Flash MTP suffix"
        );
        ensure!(
            Rc::ptr_eq(&target.owner, &head.owner)
                && head.layer.prefix_position() == target.offset().saturating_sub(1),
            "MTP prefix differs from the target"
        );
        *self = Self::default();
        let result = (|| {
            let mut logits = None;
            for (index, &token) in tokens.iter().enumerate() {
                ensure!(
                    !cancelled.load(std::sync::atomic::Ordering::Relaxed),
                    "generation cancelled"
                );
                if let Some(hidden) = target.last_hidden() {
                    head.extend(target, hidden, token)?;
                }
                // All verified hidden rows still feed the draft catch-up. Only
                // the final prompt row needs the expensive vocabulary head.
                logits = target.prefill_inner(&[token], None, None, index + 1 == tokens.len())?;
            }
            logits.context("empty Flash MTP prefill")
        })();
        if result.is_err() {
            target.reset();
            head.reset();
        }
        result
    }

    pub fn advance(
        &mut self,
        target: &mut FlashNext,
        head: &mut FlashMtpHead,
        anchor: u32,
        context_limit: u32,
        output_remaining: usize,
    ) -> Result<u32> {
        self.advance_with_eos(target, head, anchor, context_limit, output_remaining, &[])
    }

    #[allow(clippy::too_many_arguments)]
    pub fn advance_with_eos(
        &mut self,
        target: &mut FlashNext,
        head: &mut FlashMtpHead,
        anchor: u32,
        context_limit: u32,
        output_remaining: usize,
        eos: &[u32],
    ) -> Result<u32> {
        self.advance_with_eos_cancellable(
            target,
            head,
            anchor,
            context_limit,
            output_remaining,
            eos,
            &std::sync::atomic::AtomicBool::new(false),
        )
    }

    /// Cancellation rolls back semantic target/draft/session state. Physical
    /// reads and submitted GPU work still retire before their owners are reused.
    #[allow(clippy::too_many_arguments)]
    pub fn advance_with_eos_cancellable(
        &mut self,
        target: &mut FlashNext,
        head: &mut FlashMtpHead,
        anchor: u32,
        context_limit: u32,
        output_remaining: usize,
        eos: &[u32],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<u32> {
        let check_cancelled = || -> Result<()> {
            ensure!(
                !cancelled.load(std::sync::atomic::Ordering::Relaxed),
                "generation cancelled"
            );
            Ok(())
        };
        check_cancelled()?;
        ensure!(
            output_remaining > 0
                && anchor < 248320
                && context_limit <= target.context
                && Rc::ptr_eq(&target.owner, &head.owner),
            "invalid Flash MTP budget, token or session"
        );
        if let Some((expected_anchor, token, row)) = &self.pending {
            ensure!(
                anchor == *expected_anchor
                    && target.offset <= context_limit
                    && target
                        .last_hidden()
                        .is_some_and(|h| std::ptr::eq::<metal::BufferRef>(
                            h.buffer.as_ref(),
                            row.buffer.as_ref()
                        )),
                "pending Flash MTP token belongs to a different prefix"
            );
            let verified = *token;
            check_cancelled()?;
            self.pending = None;
            return Ok(verified);
        }
        let remaining = context_limit
            .checked_sub(target.offset)
            .context("Flash MTP context exhausted")?;
        let width = crate::speculative::mtp_width(1, remaining as usize, output_remaining)
            .context("Flash MTP context exhausted")?;
        ensure!(
            target.offset.checked_sub(1) == Some(head.layer.offset()),
            "Flash MTP caches are not aligned"
        );
        let hidden = target
            .last_hidden()
            .context("uninitialized Flash MTP target")?
            .clone();
        let target_before = target.snapshot();
        let head_before = head.layer.snapshot();
        let session_before = (self.pending.clone(), self.proposed, self.accepted);
        let device_greedy = head.device_greedy;
        let select = |target: &mut FlashNext, token| -> Result<u32> {
            check_cancelled()?;
            let token = if device_greedy {
                target.forward_greedy(token)?
            } else {
                let logits = target.forward(token)?;
                target.greedy_id(&logits)?
            };
            check_cancelled()?;
            Ok(token)
        };
        let result = (|| {
            if width == 0 {
                head.extend(target, &hidden, anchor)?;
                return select(target, anchor);
            }
            let proposal = head.propose(target, &hidden, anchor)?;
            check_cancelled()?;
            if head.block_verify {
                // Explicit research path: canonical scalar kernels scheduled
                // layer-major. It pays for both rows even on rejection; no
                // shared-weight or throughput claim is implied by this flag.
                let gpu = target.gpu;
                let staged = target.stage_block_cancellable(&[anchor, proposal], cancelled)?;
                let first = crate::speculative::greedy_log_prob_id(
                    &gpu.flash_log_probs(&staged.logits()[0])?,
                )
                .context("invalid Flash MTP probabilities")?;
                let accepted = first == proposal;
                if !accepted || eos.contains(&first) {
                    staged.commit(1)?;
                    self.proposed += 1;
                    self.accepted += usize::from(accepted);
                    return Ok(first);
                }
                let bonus = crate::speculative::greedy_log_prob_id(
                    &gpu.flash_log_probs(&staged.logits()[1])?,
                )
                .context("invalid Flash MTP probabilities")?;
                let retained_hidden = staged
                    .prefix_hidden(1)
                    .context("missing verified anchor hidden row")?
                    .clone();
                staged.commit(2)?;
                head.extend(target, &retained_hidden, proposal)?;
                self.pending = Some((
                    proposal,
                    bonus,
                    target
                        .last_hidden()
                        .context("missing verified bonus row")?
                        .clone(),
                ));
                self.proposed += 1;
                self.accepted += 1;
                return Ok(proposal);
            }
            let first = select(target, anchor)?;
            // EOS is emitted without advancing its discarded successor. The
            // retained target/head prefix is the ordinary prefix before EOS.
            if eos.contains(&first) {
                self.proposed += 1;
                self.accepted += usize::from(first == proposal);
                return Ok(first);
            }
            let retained_hidden = (first == proposal)
                .then(|| target.last_hidden().cloned())
                .flatten();
            // A rejected proposal must not trigger another target pass (and SSD reads).
            // The first pass already leaves the exact retained anchor prefix in place.
            let choice = crate::speculative::greedy_accept_one(proposal, first, || {
                select(target, proposal)
            })?;
            if choice.accepted_draft_tokens != 0 {
                head.extend(
                    target,
                    retained_hidden
                        .as_ref()
                        .context("missing retained hidden row")?,
                    proposal,
                )?;
                self.pending = Some((
                    proposal,
                    choice.target_token,
                    target
                        .last_hidden()
                        .context("missing MTP verified row")?
                        .clone(),
                ));
            }
            self.proposed += 1;
            self.accepted += choice.accepted_draft_tokens;
            Ok(if choice.accepted_draft_tokens == 0 {
                choice.target_token
            } else {
                proposal
            })
        })();
        let result = result.and_then(|token| {
            check_cancelled()?;
            Ok(token)
        });
        if result.is_err() {
            self.pending = session_before.0;
            self.proposed = session_before.1;
            self.accepted = session_before.2;
            // Attempt both restores even if one fails; never leave the draft
            // advanced merely because the target restore reported an error.
            let target_restore = target.restore(&target_before);
            let head_restore = head.layer.restore(&head_before);
            target_restore?;
            head_restore?;
        }
        result
    }
}

#[cfg(test)]
#[test]
#[ignore = "physical Flash MTP head and the held-out independent checkpoint oracle"]
fn draft_head_replays_retained_steps_and_vocabulary() -> Result<()> {
    use half::f16;
    use std::path::{Path, PathBuf};
    let model = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let dir = PathBuf::from(std::env::var("MLXL3_FLASH_MTP_REFERENCE")?);
    let reference: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(dir.join("reference.json"))?)?;
    ensure!(reference["status"] == "complete" && reference["mode"] == "eval");
    let steps = reference["mtp_steps"]
        .as_array()
        .context("missing MTP steps")?;
    ensure!((4..=31).contains(&steps.len()));
    let gpu = Metal::new()?;
    let target = FlashNext::load(&gpu, &model, steps.len() as u32 + 1)?;
    let mut head = FlashMtpHead::load(&model, &target)?;
    let words = |entry: &serde_json::Value, count: usize| -> Result<Vec<f32>> {
        ensure!(entry["count"].as_u64() == Some(count as u64));
        let bytes =
            std::fs::read(dir.join(entry["file"].as_str().context("missing oracle file")?))?;
        ensure!(bytes.len() == count * 4 && count > 0);
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect::<Vec<_>>();
        ensure!(values.iter().all(|x| x.is_finite()));
        Ok(values)
    };
    for (i, case) in steps.iter().enumerate() {
        ensure!(case["position"] == i);
        for (key, width) in [
            ("input", 10240),
            ("mixed", 10240),
            ("layer", 10240),
            ("output", 2560),
            ("logits", 248320),
        ] {
            ensure!(
                case[key]["shape"] == serde_json::json!([1, 1, width]),
                "invalid {key} shape at {i}"
            );
        }
        for key in ["keys", "values"] {
            ensure!(
                case[key]["shape"] == serde_json::json!([1, 2, i + 1, 256]),
                "invalid {key} shape at {i}"
            );
        }
        if (i + 1).is_multiple_of(4) {
            ensure!(case["index_tail"].is_null());
        } else {
            ensure!(case["index_tail"]["shape"] == serde_json::json!([1, (i + 1) % 4, 128]));
        }
        let token = u32::try_from(case["token"].as_u64().context("missing token")?)?;
        let hidden = gpu
            .f32_storage(&words(&case["input"], 10240)?)?
            .matrix(MatrixLayout {
                shape: [1, 10240],
                strides: [10240, 1],
                offset: 0,
            })?;
        let mixed = head.mixed(&target, &hidden, token)?;
        let read = |x: &MatrixView| -> Result<Vec<f32>> {
            let bytes = x.export_bytes(&gpu)?;
            Ok(if x.dtype == FloatType::F16 {
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .copied()
                    .map(|b| f16::from_le_bytes(b).to_f32())
                    .collect()
            } else {
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .copied()
                    .map(f32::from_le_bytes)
                    .collect()
            })
        };
        let check = |key: &str, actual: Vec<f32>, n| -> Result<()> {
            let expected = words(&case[key], n)?;
            let differences = actual
                .iter()
                .zip(&expected)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            ensure!(
                actual.len() == n && differences == 0,
                "MTP step {i} {key}: {differences} unequal words out of {n}"
            );
            Ok(())
        };
        check("mixed", read(&mixed)?, 10240)?;
        let before = head.layer.snapshot();
        let layer = head.layer.forward(&mixed)?;
        check("layer", read(&layer)?, 10240)?;
        check(
            "output",
            read(&head.mixer.read(&layer, false)?.mixed)?,
            2560,
        )?;
        head.layer.restore(&before)?;
        let logits = head.forward(&target, &hidden, token)?;
        check(
            "logits",
            logits
                .into_iter()
                .map(|v| f16::from_bits(v).to_f32())
                .collect(),
            248320,
        )?;
        let full = head.layer.retained_half_words()?;
        for key in ["keys", "values", "index_tail"] {
            if let Some(bits) = full.get(key) {
                check(
                    key,
                    bits.iter().map(|&v| f16::from_bits(v).to_f32()).collect(),
                    bits.len(),
                )?;
            } else {
                ensure!(case[key].is_null(), "missing {key} at step {i}");
            }
        }
        head.layer.restore(&before)?;
        head.extend(&target, &hidden, token)?;
        ensure!(
            head.layer.retained_half_words()? == full,
            "cache-only MTP extension differs at step {i}"
        );
    }
    Ok(())
}

#[cfg(test)]
#[test]
#[ignore = "physical Flash MTP greedy acceptance, cache commits, cancellation and output budget"]
fn speculative_commits_match_stock_sequence_and_ordinary_state() -> Result<()> {
    use std::{path::Path, sync::atomic::AtomicBool};
    let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let stock: serde_json::Value = serde_json::from_reader(std::fs::File::open(std::env::var(
        "MLXL3_FLASH_MTP_SESSION_ORACLE",
    )?)?)?;
    let gpu = Metal::new()?;
    let mut target = FlashNext::load(&gpu, &c, 32)?;
    let mut head = FlashMtpHead::load(&c, &target)?;
    let mut session = FlashMtpSession::default();
    let prompt = [248044, 78768, 11];
    let cancelled = AtomicBool::new(true);
    ensure!(
        session
            .prefill(&mut target, &mut head, &prompt, &cancelled)
            .is_err()
    );
    ensure!(target.offset == 0 && head.layer.offset() == 0 && target.last_hidden().is_none());
    cancelled.store(false, std::sync::atomic::Ordering::Relaxed);
    let greedy = |logits: Vec<u16>| -> Result<u32> {
        crate::speculative::greedy_log_prob_id(&gpu.flash_log_probs(&logits)?)
            .context("invalid reference probabilities")
    };
    let mut counts = (0, 0);
    for budget in [1usize, 2, 3, 17] {
        let mut anchor = greedy(session.prefill(&mut target, &mut head, &prompt, &cancelled)?)?;
        let mut expected = Vec::new();
        for _ in 0..budget {
            head.extend(&target, target.last_hidden().unwrap(), anchor)?;
            anchor = greedy(target.forward(anchor)?)?;
            expected.push(anchor);
        }
        let expected_head = head.layer.retained_half_words()?;
        let expected_hidden = target.last_hidden().unwrap().export_bytes(&gpu)?;
        let expected_next = target.forward(7)?;
        let mut anchor = greedy(session.prefill(&mut target, &mut head, &prompt, &cancelled)?)?;
        let limit = (prompt.len() + budget) as u32;
        ensure!(
            session
                .advance(&mut target, &mut head, anchor, limit, 0)
                .is_err()
        );
        ensure!(
            session
                .advance(&mut target, &mut head, 248320, limit, budget)
                .is_err()
        );
        for (i, &wanted) in expected.iter().enumerate() {
            if session.pending.is_some() {
                ensure!(
                    session
                        .advance(
                            &mut target,
                            &mut head,
                            (anchor + 1) % 248320,
                            limit,
                            budget - i
                        )
                        .is_err()
                );
            }
            anchor = session.advance(&mut target, &mut head, anchor, limit, budget - i)?;
            ensure!(anchor == wanted, "wrong draft token at {i} of {budget}");
        }
        let row = &stock[budget.to_string()];
        ensure!(
            row["tokens"] == serde_json::json!(expected)
                && row["proposed"] == session.proposed
                && row["accepted"] == session.accepted,
            "stock MTP transcript differs for {budget}"
        );
        ensure!(
            session.pending.is_none() && target.offset == limit && head.layer.offset() + 1 == limit
        );
        ensure!(
            head.layer.retained_half_words()? == expected_head
                && target.last_hidden().unwrap().export_bytes(&gpu)? == expected_hidden
                && target.forward(7)? == expected_next
        );
        counts.0 += session.proposed;
        counts.1 += session.accepted;
    }
    ensure!(counts.1 > 0 && counts.0 > counts.1);
    session.prefill(&mut target, &mut head, &prompt, &cancelled)?;
    let before = target.snapshot();
    let head_before = head.layer.retained_half_words()?;
    let hidden_before = target.last_hidden().unwrap().export_bytes(&gpu)?;
    let short = super::first_embedding(&target.embed)?;
    let original = std::mem::replace(&mut target.embed, short);
    // Fail on the anchor embedding itself. A missing proposal embedding is no
    // longer a deterministic error: D1 rejection must skip that read entirely.
    // The CPU lazy-successor test separately covers accepted-successor errors.
    let error = session
        .advance(&mut target, &mut head, 1, 32, 3)
        .err()
        .context("truncated anchor embedding succeeded")?;
    ensure!(
        error
            .downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == std::io::ErrorKind::UnexpectedEof),
        "unexpected verification failure: {error:#}"
    );
    target.embed = original;
    ensure!(
        target.offset == before.offset
            && head.layer.retained_half_words()? == head_before
            && target.last_hidden().unwrap().export_bytes(&gpu)? == hidden_before
            && session.pending.is_none()
    );
    let expected = greedy(target.forward(0)?)?;
    target.restore(&before)?;
    ensure!(session.advance(&mut target, &mut head, 0, 32, 3)? == expected);
    Ok(())
}

impl FlashMtpHead<'_> {
    pub(in crate::gpu) fn record_memory(
        &self,
        target: &FlashNext,
        backings: &mut crate::gpu::memory::Backings,
    ) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.owner, &target.owner),
            "MTP belongs to a different target"
        );
        backings.record("fixed_weights", &self.embedding_gain)?;
        backings.record("fixed_weights", &self.hidden_gain)?;
        self.embedding_fc.record_memory(backings)?;
        self.hidden_fc.record_memory(backings)?;
        self.layer.record_memory(backings)?;
        self.mixer.record_memory(backings)
    }
}

#[cfg(test)]
#[test]
#[ignore = "physical Flash checkpoint; production scalar/B2 session and EOS rollback parity"]
fn block_verification_matches_scalar_mtp_and_eos_prefixes() -> Result<()> {
    use std::{path::Path, sync::atomic::AtomicBool};
    fn copy_session(session: &FlashMtpSession) -> FlashMtpSession {
        FlashMtpSession {
            pending: session.pending.clone(),
            proposed: session.proposed,
            accepted: session.accepted,
        }
    }
    fn same_session(a: &FlashMtpSession, b: &FlashMtpSession) -> Result<()> {
        ensure!(a.proposed == b.proposed && a.accepted == b.accepted);
        ensure!(
            a.pending
                .as_ref()
                .map(|(anchor, token, _)| (*anchor, *token))
                == b.pending
                    .as_ref()
                    .map(|(anchor, token, _)| (*anchor, *token))
        );
        Ok(())
    }
    let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    let mut target = FlashNext::load(&gpu, &c, 32)?;
    ensure!(target.lookahead_depth == 0 && target.prefetcher.is_none());
    let mut head = FlashMtpHead::load(&c, &target)?;
    head.device_greedy = false;
    let mut session = FlashMtpSession::default();
    let cancelled = AtomicBool::new(false);
    let mut seen_accept = false;
    let mut seen_reject = false;
    let mut seen_pending = false;
    for budget in [1usize, 2, 3, 17] {
        let logits = session.prefill(&mut target, &mut head, &[248044, 78768, 11], &cancelled)?;
        let mut anchor = target.greedy_id(&logits)?;
        let limit = 3 + budget as u32;
        for index in 0..budget {
            let before = target.snapshot();
            let draft_before = head.layer.snapshot();
            let session_before = copy_session(&session);
            // Cancellation must not consume an already-verified pending token
            // or change counters/target/draft state.
            let hidden_before_cancel = target.last_hidden().unwrap().export_bytes(&gpu)?;
            let draft_before_cancel = head.layer.retained_half_words()?;
            cancelled.store(true, std::sync::atomic::Ordering::Relaxed);
            ensure!(
                session
                    .advance_with_eos_cancellable(
                        &mut target,
                        &mut head,
                        anchor,
                        limit,
                        budget - index,
                        &[],
                        &cancelled
                    )
                    .is_err()
            );
            same_session(&session, &session_before)?;
            ensure!(target.offset == before.offset);
            ensure!(target.last_hidden().unwrap().export_bytes(&gpu)? == hidden_before_cancel);
            ensure!(head.layer.retained_half_words()? == draft_before_cancel);
            cancelled.store(false, std::sync::atomic::Ordering::Relaxed);
            seen_pending |= session.pending.is_some();
            head.block_verify = false;
            let wanted = session.advance_with_eos(
                &mut target,
                &mut head,
                anchor,
                limit,
                budget - index,
                &[],
            )?;
            let after = target.snapshot();
            let draft_after = head.layer.snapshot();
            let wanted_session = copy_session(&session);
            let wanted_draft = head.layer.retained_half_words()?;
            let wanted_hidden = target
                .last_hidden()
                .context("missing scalar MTP hidden")?
                .export_bytes(&gpu)?;
            let wanted_continuation = target.forward_trace(7)?;
            seen_accept |= session.accepted > session_before.accepted;
            seen_reject |= session.proposed > session_before.proposed
                && session.accepted == session_before.accepted;
            target.restore(&before)?;
            head.layer.restore(&draft_before)?;
            session = copy_session(&session_before);
            head.block_verify = true;
            let actual = session.advance_with_eos(
                &mut target,
                &mut head,
                anchor,
                limit,
                budget - index,
                &[],
            )?;
            ensure!(actual == wanted && target.offset == after.offset);
            same_session(&session, &wanted_session)?;
            ensure!(head.layer.retained_half_words()? == wanted_draft);
            ensure!(
                target
                    .last_hidden()
                    .context("missing B2 hidden")?
                    .export_bytes(&gpu)?
                    == wanted_hidden
            );
            if let Some((_, _, row)) = &session.pending {
                ensure!(target.last_hidden().is_some_and(
                    |hidden| std::ptr::eq::<metal::BufferRef>(
                        hidden.buffer.as_ref(),
                        row.buffer.as_ref()
                    )
                ));
            }
            let actual_continuation = target.forward_trace(7)?;
            ensure!(actual_continuation.0 == wanted_continuation.0);
            ensure!(
                !actual_continuation.1.is_empty()
                    && actual_continuation.1.len() == wanted_continuation.1.len()
            );
            for (actual, wanted) in actual_continuation.1.iter().zip(&wanted_continuation.1) {
                ensure!(
                    actual.name == wanted.name
                        && actual.shape == wanted.shape
                        && actual.data == wanted.data
                );
            }
            // Force the independently selected first result to be EOS. This
            // covers accepted and rejected proposals without relying on a
            // model naturally emitting EOS within a short test transcript.
            if session_before.pending.is_none() {
                target.restore(&before)?;
                head.layer.restore(&draft_before)?;
                session = copy_session(&session_before);
                head.block_verify = false;
                ensure!(
                    session.advance_with_eos(
                        &mut target,
                        &mut head,
                        anchor,
                        limit,
                        budget - index,
                        &[wanted]
                    )? == wanted
                );
                let eos_state = target.snapshot();
                let eos_session = copy_session(&session);
                let eos_draft = head.layer.retained_half_words()?;
                let eos_hidden = target
                    .last_hidden()
                    .context("missing scalar EOS hidden")?
                    .export_bytes(&gpu)?;
                let eos_continuation = target.forward_trace(7)?;
                target.restore(&before)?;
                head.layer.restore(&draft_before)?;
                session = copy_session(&session_before);
                head.block_verify = true;
                ensure!(
                    session.advance_with_eos(
                        &mut target,
                        &mut head,
                        anchor,
                        limit,
                        budget - index,
                        &[wanted]
                    )? == wanted
                );
                ensure!(
                    session.pending.is_none()
                        && target.offset == before.offset + 1
                        && target.offset == eos_state.offset
                );
                same_session(&session, &eos_session)?;
                ensure!(head.layer.retained_half_words()? == eos_draft);
                ensure!(
                    target
                        .last_hidden()
                        .context("missing B2 EOS hidden")?
                        .export_bytes(&gpu)?
                        == eos_hidden
                );
                let actual_continuation = target.forward_trace(7)?;
                ensure!(actual_continuation.0 == eos_continuation.0);
                ensure!(
                    !actual_continuation.1.is_empty()
                        && actual_continuation.1.len() == eos_continuation.1.len()
                );
                for (actual, wanted) in actual_continuation.1.iter().zip(&eos_continuation.1) {
                    ensure!(
                        actual.name == wanted.name
                            && actual.shape == wanted.shape
                            && actual.data == wanted.data
                    );
                }
            }
            // Restore the ordinary accepted/rejected result, including the
            // pending handle's exact target backing, before the next call.
            target.restore(&after)?;
            head.layer.restore(&draft_after)?;
            session = wanted_session;
            anchor = wanted;
        }
        ensure!(session.pending.is_none() && target.offset == limit);
    }
    ensure!(
        seen_accept && seen_reject && seen_pending,
        "checkpoint fixture must exercise all MTP decisions"
    );
    Ok(())
}

#[cfg(test)]
mod optimization_regression_tests {
    use super::*;
    use std::{collections::BTreeMap, path::Path, sync::atomic::AtomicBool};

    fn target_state(target: &FlashNext<'_>) -> Result<BTreeMap<String, Vec<u8>>> {
        let mut state = BTreeMap::new();
        state.insert("offset".into(), target.offset.to_le_bytes().to_vec());
        state.insert(
            "hidden".into(),
            target
                .last_hidden()
                .context("missing target hidden")?
                .export_bytes(target.gpu)?,
        );
        for (index, layer) in target.layers.iter().enumerate() {
            if let Some(row) = layer.recurrent() {
                state.insert(format!("recurrent_{index}"), row.export_bytes(target.gpu)?);
            }
            if let super::super::Layer::Qsa(layer) = layer {
                state.insert(
                    format!("qsa_{index}_offset"),
                    layer.offset().to_le_bytes().to_vec(),
                );
                for (name, words) in layer.retained_half_words()? {
                    state.insert(
                        format!("qsa_{index}_{name}"),
                        words.into_iter().flat_map(u16::to_le_bytes).collect(),
                    );
                }
            }
        }
        ensure!(state.len() > 2 && state.values().all(|bytes| !bytes.is_empty()));
        Ok(state)
    }

    #[test]
    #[ignore = "physical Flash checkpoint; exact old/new prefill states and real vocabulary-head counts; exclusive GPU"]
    fn final_head_only_prefill_matches_per_token_heads() -> Result<()> {
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut target = FlashNext::load(&gpu, &c, 32)?;
        let mut head = FlashMtpHead::load(&c, &target)?;
        let cancelled = AtomicBool::new(false);
        for prefix in [&[][..], &[248044, 78768, 11][..]] {
            for suffix in [&[7][..], &[7, 19, 31, 13, 29][..]] {
                target.reset();
                head.reset();
                let mut session = FlashMtpSession::default();
                if !prefix.is_empty() {
                    session.prefill(&mut target, &mut head, prefix, &cancelled)?;
                }
                let before = target.snapshot();
                let draft_before = head.layer.snapshot();
                let calls = target.target_head_calls;
                let mut expected_logits = None;
                for &token in suffix {
                    if let Some(hidden) = target.last_hidden() {
                        head.extend(&target, hidden, token)?;
                    }
                    expected_logits = Some(target.forward(token)?);
                }
                ensure!(target.target_head_calls - calls == suffix.len());
                let expected_logits = expected_logits.context("empty control suffix")?;
                ensure!(
                    expected_logits.len() == 248320
                        && expected_logits
                            .iter()
                            .all(|&v| half::f16::from_bits(v).is_finite())
                );
                let expected_state = target_state(&target)?;
                let expected_draft = head.layer.retained_half_words()?;
                let expected_position = head.layer.offset();
                target.restore(&before)?;
                head.layer.restore(&draft_before)?;
                let calls = target.target_head_calls;
                let actual = session.extend_prefill(&mut target, &mut head, suffix, &cancelled)?;
                ensure!(
                    target.target_head_calls - calls == 1,
                    "prefill must evaluate exactly one vocabulary head"
                );
                ensure!(actual == expected_logits);
                ensure!(target_state(&target)? == expected_state);
                ensure!(
                    head.layer.offset() == expected_position
                        && head.layer.retained_half_words()? == expected_draft
                );
                ensure!(
                    session.pending.is_none() && session.proposed == 0 && session.accepted == 0
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "physical Flash checkpoint and held-out MTP oracle; explicitly enable MTP device greedy and its pipelines; exclusive GPU"]
    fn device_proposal_matches_normalized_host_on_oracle_rows() -> Result<()> {
        ensure!(
            std::env::var("MLXL3_NATIVE_MTP_DEVICE_GREEDY").as_deref() == Ok("1"),
            "explicitly enable MLXL3_NATIVE_MTP_DEVICE_GREEDY=1"
        );
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_FLASH_MTP_REFERENCE")?);
        let reference: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(dir.join("reference.json"))?)?;
        ensure!(reference["status"] == "complete" && reference["mode"] == "eval");
        let steps = reference["mtp_steps"]
            .as_array()
            .context("missing oracle steps")?;
        ensure!((4..=31).contains(&steps.len()));
        let gpu = Metal::new()?;
        let target = FlashNext::load(&gpu, &c, steps.len() as u32 + 1)?;
        let mut head = FlashMtpHead::load(&c, &target)?;
        ensure!(
            gpu.device_greedy_available() && head.device_greedy,
            "device greedy pipeline unavailable or disabled; test not qualified"
        );
        let read = |entry: &serde_json::Value, count: usize| -> Result<Vec<f32>> {
            ensure!(entry["count"].as_u64() == Some(count as u64));
            let name = entry["file"].as_str().context("missing oracle file")?;
            ensure!(Path::new(name).components().count() == 1);
            let bytes = std::fs::read(dir.join(name))?;
            ensure!(bytes.len() == count * 4 && count > 0);
            let values = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .copied()
                .map(f32::from_le_bytes)
                .collect::<Vec<_>>();
            ensure!(values.iter().all(|v| v.is_finite()));
            Ok(values)
        };
        let mut checked = 0;
        for (position, case) in steps.iter().enumerate() {
            ensure!(case["position"] == position);
            ensure!(case["input"]["shape"] == serde_json::json!([1, 1, 10240]));
            let input = read(&case["input"], 10240)?;
            let hidden = gpu.f32_storage(&input)?.matrix(MatrixLayout {
                shape: [1, 10240],
                strides: [10240, 1],
                offset: 0,
            })?;
            let token = u32::try_from(case["token"].as_u64().context("missing oracle token")?)?;
            let before = head.layer.snapshot();
            let logits = head.forward(&target, &hidden, token)?;
            let oracle = read(&case["logits"], 248320)?;
            ensure!(
                logits.len() == oracle.len()
                    && logits
                        .iter()
                        .zip(&oracle)
                        .all(|(&a, &b)| half::f16::from_bits(a).to_f32().to_bits() == b.to_bits()),
                "canonical logits differ from independent oracle at {position}"
            );
            let expected = target.greedy_id(&logits)?;
            let expected_state = head.layer.retained_half_words()?;
            let expected_position = head.layer.offset();
            head.layer.restore(&before)?;
            let counts = target.device_greedy_counts();
            let actual = head.propose(&target, &hidden, token)?;
            let after = target.device_greedy_counts();
            ensure!(
                after[0] == counts[0] + 1 && after[1] == counts[1] + 1,
                "proposal did not execute the GPU selector/head"
            );
            ensure!(
                actual == expected,
                "GPU proposal changed normalized greedy ID at {position}"
            );
            ensure!(
                head.layer.offset() == expected_position
                    && head.layer.retained_half_words()? == expected_state
            );
            checked += 1;
        }
        ensure!(checked == steps.len() && checked >= 4);
        Ok(())
    }
}

#[cfg(all(test, feature = "chat"))]
#[path = "mtp_campaign.rs"]
mod campaign;

#[cfg(test)]
#[test]
#[ignore = "physical Flash checkpoint; production MTP cancellation after GPU submission"]
fn mtp_request_cancel_rolls_back_after_ssd_submission() -> Result<()> {
    use std::{
        path::Path,
        sync::atomic::{AtomicBool, Ordering},
    };
    let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    let mut target = FlashNext::load(&gpu, &c, 32)?;
    ensure!(target.lookahead_depth == 0 && target.prefetcher.is_none());
    target.verify_ssd_cohorts = true;
    let mut head = FlashMtpHead::load(&c, &target)?;
    head.block_verify = true;
    let mut session = FlashMtpSession::default();
    let cancelled = AtomicBool::new(false);
    let logits = session.prefill(&mut target, &mut head, &[248044, 78768, 11], &cancelled)?;
    let anchor = target.greedy_id(&logits)?;
    let before = target.snapshot();
    let draft_before = head.layer.retained_half_words()?;
    let hidden_before = target.last_hidden().unwrap().export_bytes(&gpu)?;
    let expected_next = target.forward_trace(anchor)?;
    target.restore(&before)?;
    super::FlashMoeRow::set_cancel_after_initial_submit_for_test(true);
    let result = session.advance_with_eos_cancellable(
        &mut target,
        &mut head,
        anchor,
        32,
        4,
        &[],
        &cancelled,
    );
    super::FlashMoeRow::set_cancel_after_initial_submit_for_test(false);
    ensure!(
        result.is_err() && cancelled.load(Ordering::Relaxed),
        "cancellation injection not exercised"
    );
    ensure!(
        target.offset == before.offset
            && session.proposed == 0
            && session.accepted == 0
            && session.pending.is_none()
    );
    ensure!(head.layer.retained_half_words()? == draft_before);
    ensure!(target.last_hidden().unwrap().export_bytes(&gpu)? == hidden_before);
    cancelled.store(false, Ordering::Relaxed);
    let actual_next = target.forward_trace(anchor)?;
    ensure!(
        actual_next.0 == expected_next.0
            && !actual_next.1.is_empty()
            && actual_next.1.len() == expected_next.1.len()
    );
    for (actual, expected) in actual_next.1.iter().zip(expected_next.1) {
        ensure!(
            actual.name == expected.name
                && actual.shape == expected.shape
                && actual.data == expected.data
        );
    }
    Ok(())
}
