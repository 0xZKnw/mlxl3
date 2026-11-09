//! Test-only linear depth sweep. Verification stays within existing 1..3-row blocks.
use super::*;
use std::collections::VecDeque;

pub(in crate::gpu) struct Pilot {
    depth: usize,
    block: usize,
    pending: VecDeque<u32>,
    expected_anchor: u32,
    pending_prefix: Option<(u32, MatrixView)>,
    pub(in crate::gpu) proposed: usize,
    pub(in crate::gpu) accepted: usize,
    pub(in crate::gpu) widths: [usize; 4],
    pub(in crate::gpu) lengths: [usize; 4],
}
impl Pilot {
    pub(in crate::gpu) fn new(depth: usize, block: usize) -> Result<Self> {
        ensure!((1..=4).contains(&depth) && (1..=3).contains(&block));
        Ok(Self {
            depth,
            block,
            pending: VecDeque::new(),
            expected_anchor: 0,
            pending_prefix: None,
            proposed: 0,
            accepted: 0,
            widths: [0; 4],
            lengths: [0; 4],
        })
    }
    pub(in crate::gpu) fn advance(
        &mut self,
        target: &mut FlashNext,
        head: &mut FlashMtpHead,
        anchor: u32,
        remaining: usize,
        eos: &[u32],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<u32> {
        ensure!(!cancelled.load(std::sync::atomic::Ordering::Relaxed) && remaining > 0);
        if let Some(token) = self.pending.front().copied() {
            let (offset, row) = self
                .pending_prefix
                .as_ref()
                .context("missing pending MTP prefix")?;
            ensure!(
                anchor == self.expected_anchor
                    && target.offset() == *offset
                    && target
                        .last_hidden()
                        .is_some_and(|h| std::ptr::eq::<metal::BufferRef>(
                            h.buffer.as_ref(),
                            row.buffer.as_ref()
                        ))
            );
            self.pending.pop_front();
            self.expected_anchor = token;
            if self.pending.is_empty() {
                self.pending_prefix = None;
            }
            return Ok(token);
        }
        ensure!(head.layer.offset() + 1 == target.offset());
        let available = (target.context - target.offset()) as usize;
        let depth = self.depth.min(remaining.min(available).saturating_sub(1));
        ensure!(available > 0);
        let root = target.snapshot();
        let draft_root = head.layer.snapshot();
        let before_hidden = target
            .last_hidden()
            .context("missing MTP target hidden")?
            .clone();
        let operation = (|| -> Result<(Vec<u32>, usize, usize)> {
            if depth == 0 {
                head.extend(target, &before_hidden, anchor)?;
                return Ok((vec![target.forward_greedy(anchor)?], 0, 0));
            }
            let mut proposals = Vec::with_capacity(depth);
            let mut draft_hidden = before_hidden.clone();
            let mut token = anchor;
            let mut first_draft = None;
            for step in 0..depth {
                ensure!(!cancelled.load(std::sync::atomic::Ordering::Relaxed));
                let (next, hidden) =
                    propose_hidden(head, target, &draft_hidden, token, step + 1 < depth)?;
                if step == 0 {
                    first_draft = Some(head.layer.snapshot());
                }
                proposals.push(next);
                draft_hidden = hidden;
                token = next;
                if eos.contains(&next) {
                    break;
                }
            }
            let n = proposals.len();
            let mut inputs = vec![anchor];
            inputs.extend_from_slice(&proposals);
            let mut outputs = Vec::with_capacity(n + 1);
            let mut hidden = Vec::with_capacity(n + 1);
            let mut accepted = 0;
            let mut at = 0;
            let mut stopped = false;
            while at < inputs.len() && !stopped {
                let end = (at + self.block).min(inputs.len());
                self.widths[end - at - 1] += 1;
                let mut staged = target.stage_block_headless(&inputs[at..end], cancelled)?;
                let mut keep = 0;
                for row in 0..end - at {
                    ensure!(!cancelled.load(std::sync::atomic::Ordering::Relaxed));
                    let next = staged.greedy(row + 1, true)?;
                    hidden.push(
                        staged
                            .prefix_hidden(row + 1)
                            .context("missing verified draft repair row")?
                            .clone(),
                    );
                    outputs.push(next);
                    keep += 1;
                    let index = at + row;
                    if index < n {
                        if next == proposals[index] {
                            accepted += 1;
                        } else {
                            stopped = true;
                        }
                    }
                    if eos.contains(&next) {
                        stopped = true;
                    }
                    if stopped {
                        break;
                    }
                }
                staged.commit(keep)?;
                at += keep;
            }
            head.layer
                .restore(&first_draft.context("empty draft chain")?)?;
            for index in 1..outputs.len() {
                head.extend(target, &hidden[index - 1], inputs[index])?;
            }
            ensure!(head.layer.offset() + 1 == target.offset() && outputs.len() <= remaining);
            Ok((outputs, n, accepted))
        })();
        match operation {
            Ok((outputs, proposed, accepted)) => {
                ensure!(!outputs.is_empty());
                self.proposed += proposed;
                self.accepted += accepted;
                if proposed > 0 {
                    self.lengths[proposed - 1] += 1;
                }
                let first = outputs[0];
                self.pending = outputs[1..].iter().copied().collect();
                self.expected_anchor = first;
                if !self.pending.is_empty() {
                    self.pending_prefix = Some((
                        target.offset(),
                        target
                            .last_hidden()
                            .context("missing pending prefix hidden")?
                            .clone(),
                    ));
                }
                Ok(first)
            }
            Err(error) => {
                let a = target.restore(&root);
                let b = head.layer.restore(&draft_root);
                a?;
                b?;
                Err(error)
            }
        }
    }
}

fn propose_hidden(
    head: &mut FlashMtpHead,
    target: &FlashNext,
    hidden: &MatrixView,
    token: u32,
    need_hidden: bool,
) -> Result<(u32, MatrixView)> {
    let _draft = super::super::super::route_swap_trace::SuspendDraft::new();
    let streams = head.mixed(target, hidden, token)?;
    let output = head.layer.forward(&streams)?;
    let mixed = head.mixer.read(&output, false)?.mixed;
    let token = target
        .gpu
        .device_greedy_head(&target.head, &mixed, false)?
        .0;
    if !need_hidden {
        return Ok((token, hidden.clone()));
    }
    if output.dtype == FloatType::F32 {
        return Ok((token, output));
    }
    let bytes = output.export_bytes(target.gpu)?;
    ensure!(bytes.len() == 10240 * 2);
    let values = bytes
        .chunks_exact(2)
        .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite recursive draft hidden"
    );
    let output = target.gpu.f32_storage(&values)?.matrix(MatrixLayout {
        shape: [1, 10240],
        strides: [10240, 1],
        offset: 0,
    })?;
    Ok((token, output))
}
