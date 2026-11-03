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

    pub fn reset(&mut self) {
        self.layer.reset();
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
        let result = (|| {
            let mut logits = None;
            for &token in tokens {
                ensure!(
                    !cancelled.load(std::sync::atomic::Ordering::Relaxed),
                    "generation cancelled"
                );
                if let Some(hidden) = target.last_hidden() {
                    head.extend(target, hidden, token)?;
                }
                logits = Some(target.forward(token)?);
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
        let gpu = target.gpu;
        let greedy = |logits: Vec<u16>| -> Result<u32> {
            let log_probs = gpu.flash_log_probs(&logits)?;
            crate::speculative::greedy_log_prob_id(&log_probs)
                .context("invalid Flash MTP probabilities")
        };
        let result = (|| {
            if width == 0 {
                head.extend(target, &hidden, anchor)?;
                return greedy(target.forward(anchor)?);
            }
            let proposal = greedy(head.forward(target, &hidden, anchor)?)?;
            let first = greedy(target.forward(anchor)?)?;
            let cutoff = target.snapshot();
            let second = greedy(target.forward(proposal)?)?;
            let choice = crate::speculative::greedy_accept(&[proposal], &[first, second])
                .context("invalid Flash MTP verification")?;
            if choice.accepted_draft_tokens == 0 {
                target.restore(&cutoff)?;
            } else {
                head.extend(
                    target,
                    cutoff
                        .hidden
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
        if result.is_err() {
            target.restore(&target_before)?;
            head.layer.restore(&head_before)?;
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
    let error = session
        .advance(&mut target, &mut head, 0, 32, 3)
        .err()
        .context("truncated verification succeeded")?;
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
