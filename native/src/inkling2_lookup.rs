//! SPEED-2026-11-23-INK2M-MAYA LK: prompt lookup for Inkling 2 Mini, without a drafter.
//!
//! Proposals continue the latest earlier occurrence of the longest suffix of the text (prompt and
//! delivered tokens, `speculative::NgramIndex`), as Project Maya's `suffix_drafter`. The target checks
//! `[anchor, proposals]` in one verification block and the caller's acceptance rule keeps the law of every
//! token (greedy: argmax of each verified row; sampled: point-mass speculative acceptance). Steps without a
//! block are plain decode steps whose token is drawn on the GPU (SPEED-INK2M-P14), one step ahead unless a
//! block may follow. A controller (`LookupPolicy`) picks the width from the recent acceptance and constant
//! block costs, so a generation does not depend on timings.

use crate::array::Array;
use crate::inkling2::{DrawRule, DrawSeeds, Inkling2, draw_token, single_token};
use crate::speculative::NgramIndex;
use anyhow::{Context, Result, ensure};
use std::collections::VecDeque;
use std::time::Instant;

/// Widest block (proposals after the anchor).
pub const MAX_WIDTH: usize = 7;
/// Shortest matched suffix (anchor included) that may propose.
pub const MIN_MATCH: usize = 3;

/// Block of `w` proposals: wall time of a `1 + w`-row verification over a plain pipelined step
/// (SPEED-INK2M-P11 D1 verify times 37.6 / 51.2 / 62.0 / 70.4 / 80.5 / 92.6 / 105.2 / 115.0 ms for 1..8
/// rows on R3 against a ≈ 34 ms pipelined plain step, plus ≈ 0.1 for the step that is not run ahead).
pub const BLOCK_COSTS: [f64; MAX_WIDTH + 1] = [1.0, 1.6, 1.9, 2.15, 2.45, 2.8, 3.15, 3.45];

/// Width choice of lookup blocks, after Maya's `SpecCtl`: a block of `w` proposals is expected to deliver
/// E(w) = (1 - a^(w+1)) / (1 - a) tokens, `a` the recent acceptance of one proposal given the previous one
/// (decayed counts of accepted proposals `s` and of blocks that stopped on a rejection `f`). The width with
/// the most tokens per plain step, E(w) / C(w), is used when it beats a plain step; after `PROBE_AFTER`
/// refused or missed proposals (a step already run ahead) a block of up to `PROBE_WIDTH` runs anyway, so
/// the estimate keeps up with the text. The prior (a = 0.625) lets a copy start at once; a few rejections
/// bring a plain text back to plain steps.
#[derive(Clone, Debug)]
pub struct LookupPolicy {
    s: f64,
    f: f64,
    refused: usize,
}

impl Default for LookupPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl LookupPolicy {
    pub const DECAY: f64 = 0.85;
    pub const PROBE_AFTER: usize = 16;
    pub const PROBE_WIDTH: usize = 4;

    pub fn new() -> Self {
        Self {
            s: 2.0,
            f: 1.0,
            refused: 0,
        }
    }

    pub fn acceptance(&self) -> f64 {
        ((self.s + 0.5) / (self.s + self.f + 1.0)).min(0.98)
    }

    pub fn expected_tokens(&self, width: usize) -> f64 {
        let a = self.acceptance();
        (1.0 - a.powi(width as i32 + 1)) / (1.0 - a)
    }

    fn best(&self, available: usize) -> usize {
        let mut best = 0;
        let mut value = 1.0;
        for (w, cost) in BLOCK_COSTS
            .iter()
            .enumerate()
            .take(available.min(MAX_WIDTH) + 1)
            .skip(1)
        {
            let v = self.expected_tokens(w) / cost;
            if v > value {
                value = v;
                best = w;
            }
        }
        best
    }

    /// Width of the block for `available` proposals (0: a plain step).
    pub fn choose(&mut self, available: usize) -> usize {
        if available == 0 {
            return 0;
        }
        let best = self.best(available);
        if best > 0 {
            return best;
        }
        self.refused += 1;
        if self.refused >= Self::PROBE_AFTER {
            self.refused = 0;
            return available.min(Self::PROBE_WIDTH);
        }
        0
    }

    /// A proposal existed but its anchor was already consumed by a step run ahead.
    pub fn missed(&mut self) {
        self.refused += 1;
    }

    /// Whether the step consuming the next token should wait for it (not run ahead), so a lookup can see
    /// it: only when the token before it had a proposal (`likely`) and a block would run (or a probe is due).
    pub fn wants_block(&self, likely: bool) -> bool {
        likely && (self.best(MAX_WIDTH) > 0 || self.refused + 1 >= Self::PROBE_AFTER)
    }

    pub fn observe(&mut self, width: usize, accepted: usize) {
        if width == 0 {
            return;
        }
        self.s = Self::DECAY * self.s + accepted as f64;
        self.f = Self::DECAY * self.f + if accepted < width { 1.0 } else { 0.0 };
        self.refused = 0;
    }
}

/// Counters of one session.
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct LookupStats {
    pub blocks: usize,
    pub proposed: usize,
    pub accepted: usize,
    pub plain_steps: usize,
    /// plain steps not run ahead because a block might follow
    pub held_steps: usize,
    pub block_ms: f64,
    pub plain_ms: f64,
    /// blocks run at each width (index = width)
    pub width_blocks: Vec<usize>,
}

/// Target-side choice of one block: `(proposals, verified logits [1, 1 + n, vocab])` -> accepted count
/// and the target token after them.
pub type LookupChoice<'a> = dyn FnMut(&[u32], &Array) -> Result<(usize, u32)> + 'a;

/// Prompt lookup session of one generation round.
pub struct LookupSession {
    index: NgramIndex,
    policy: LookupPolicy,
    rule: DrawRule,
    seeds: DrawSeeds,
    pipelined: bool,
    /// Verified tokens not delivered yet.
    pending: VecDeque<u32>,
    /// The token `advance` returned last (passed back as `next`).
    expected: Option<u32>,
    /// The plain step anchored on the token `advance` returned last, already submitted: the token after it.
    ahead: Option<Array>,
    taps: Vec<usize>,
    pub stats: LookupStats,
}

impl LookupSession {
    /// `prompt`: the prefilled tokens. The first generated token is chosen by the caller.
    pub fn new(
        target: &Inkling2,
        prompt: &[u32],
        rule: DrawRule,
        seeds: DrawSeeds,
        pipelined: bool,
    ) -> Self {
        Self {
            index: NgramIndex::new(prompt),
            policy: LookupPolicy::new(),
            rule,
            seeds,
            pipelined,
            pending: VecDeque::new(),
            expected: None,
            ahead: None,
            taps: vec![target.visit_count()],
            stats: LookupStats {
                width_blocks: vec![0; MAX_WIDTH + 1],
                ..LookupStats::default()
            },
        }
    }

    /// `next`: the token the caller just delivered (the target has not consumed it yet unless a step was
    /// run ahead). Returns the token after it. `remaining` >= 1 tokens may still be delivered.
    pub fn advance(
        &mut self,
        target: &mut Inkling2,
        next: u32,
        context_limit: usize,
        remaining: usize,
        choose: &mut LookupChoice<'_>,
    ) -> Result<u32> {
        ensure!(remaining >= 1, "lookup advance without budget");
        if let Some(expected) = self.expected.take() {
            ensure!(
                expected == next,
                "lookup session out of step (delivered {next}, expected {expected})"
            );
        }
        self.index.push(next);
        if let Some(token) = self.pending.pop_front() {
            self.expected = Some(token);
            return Ok(token);
        }
        if let Some(after) = self.ahead.take() {
            // the step consuming `next` already runs: a proposal at `next` can only be counted
            let likely = self.index.lookup_min(1, MIN_MATCH).is_some();
            if likely {
                self.policy.missed();
            }
            return self.deliver(target, after, context_limit, remaining, likely);
        }
        let offset = usize::try_from(target.offset())?;
        ensure!(offset < context_limit, "Inkling context is full");
        let room = (remaining - 1).min(context_limit - offset - 1);
        let proposals: Vec<u32> = match self.index.lookup_min(MAX_WIDTH.min(room), MIN_MATCH) {
            Some(found) => found.to_vec(),
            None => Vec::new(),
        };
        let width = self.policy.choose(proposals.len()).min(proposals.len());
        if width == 0 {
            let started = Instant::now();
            let after = self.plain_step(target, &Array::from_u32(&[next], &[1])?)?;
            self.stats.plain_ms += started.elapsed().as_secs_f64() * 1e3;
            let likely = !proposals.is_empty();
            return self.deliver(target, after, context_limit, remaining, likely);
        }
        let started = Instant::now();
        let proposals = &proposals[..width];
        let mut block = Vec::with_capacity(1 + width);
        block.push(next);
        block.extend(proposals);
        let (logits, _) = target.verify(&block, &self.taps)?;
        let (accepted, token) = choose(proposals, &logits)?;
        ensure!(accepted <= width, "lookup accepted more than proposed");
        target.commit(accepted + 1)?;
        self.policy.observe(width, accepted);
        self.stats.blocks += 1;
        self.stats.proposed += width;
        self.stats.accepted += accepted;
        self.stats.width_blocks[width] += 1;
        self.stats.block_ms += started.elapsed().as_secs_f64() * 1e3;
        self.pending.extend(&proposals[..accepted]);
        self.pending.push_back(token);
        let first = self.pending.pop_front().context("empty lookup block")?;
        self.expected = Some(first);
        Ok(first)
    }

    /// The committed decode step of `anchor` (`[1]`), its token drawn on the GPU.
    fn plain_step(&mut self, target: &mut Inkling2, anchor: &Array) -> Result<Array> {
        let (logits, _) = target.forward_device(anchor, None, self.pipelined)?;
        let token = draw_token(&logits, self.rule, self.seeds.next_seed())?;
        if self.pipelined {
            token.async_eval()?;
        }
        self.stats.plain_steps += 1;
        Ok(token)
    }

    /// Delivers `after` (on the GPU). When pipelined, a token will follow it and no block is likely (`likely`:
    /// the token before `after` had a proposal), the step consuming `after` is submitted before it is read.
    fn deliver(
        &mut self,
        target: &mut Inkling2,
        after: Array,
        context_limit: usize,
        remaining: usize,
        likely: bool,
    ) -> Result<u32> {
        let offset = usize::try_from(target.offset())?;
        if self.pipelined && remaining >= 2 && offset < context_limit {
            if self.policy.wants_block(likely) {
                self.stats.held_steps += 1;
            } else {
                let started = Instant::now();
                self.ahead = Some(self.plain_step(target, &after)?);
                self.stats.plain_ms += started.elapsed().as_secs_f64() * 1e3;
            }
        }
        let token = single_token(&after)?;
        self.expected = Some(token);
        Ok(token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_tries_a_copy_then_probes_a_plain_text() {
        let mut policy = LookupPolicy::new();
        // prior a = 0.625: a proposal runs at once
        assert!(policy.choose(6) > 0);
        assert!(policy.wants_block(true));
        assert!(!policy.wants_block(false));
        // rejections: back to plain steps, then a probe after PROBE_AFTER refused or missed proposals
        for _ in 0..4 {
            policy.observe(3, 0);
        }
        assert_eq!(policy.best(MAX_WIDTH), 0);
        let mut widths = Vec::new();
        for i in 0..LookupPolicy::PROBE_AFTER {
            if i % 2 == 0 {
                policy.missed();
                widths.push(0);
            } else {
                widths.push(policy.choose(6));
            }
            // a probe is due once one more refusal would reach PROBE_AFTER; the probe resets the count
            assert_eq!(
                policy.wants_block(true),
                i + 2 == LookupPolicy::PROBE_AFTER,
                "step {i}"
            );
        }
        assert!(
            widths[..LookupPolicy::PROBE_AFTER - 1]
                .iter()
                .all(|&w| w == 0)
        );
        assert_eq!(
            widths[LookupPolicy::PROBE_AFTER - 1],
            LookupPolicy::PROBE_WIDTH
        );
        assert_eq!(policy.choose(0), 0);
    }

    #[test]
    fn policy_follows_copies_and_backs_off() {
        let mut policy = LookupPolicy::new();
        for _ in 0..6 {
            policy.observe(4, 4);
        }
        assert!(policy.acceptance() > 0.85);
        assert!(policy.wants_block(true));
        let wide = policy.choose(MAX_WIDTH);
        assert!(wide >= 5, "copies should run wide blocks, got {wide}");
        assert_eq!(policy.choose(2), 2);
        for _ in 0..12 {
            policy.observe(3, 0);
        }
        assert!(policy.acceptance() < 0.35);
        assert_eq!(policy.best(MAX_WIDTH), 0);
        assert!(!policy.wants_block(true));
    }

    #[test]
    fn expected_tokens_is_the_geometric_sum() {
        let policy = LookupPolicy {
            s: 3.5,
            f: 0.5,
            refused: 0,
        };
        let a = policy.acceptance();
        for w in 0..=MAX_WIDTH {
            let direct: f64 = (0..=w).map(|i| a.powi(i as i32)).sum();
            assert!((policy.expected_tokens(w) - direct).abs() < 1e-12);
        }
        assert!((policy.expected_tokens(0) - 1.0).abs() < 1e-12);
    }

    /// GPU oracle: a greedy lookup session on a prompt that repeats a passage (the continuation copies it)
    /// runs blocks and accepts proposals; every delivered token is the argmax of the single-row (AR) logits
    /// of a teacher-forced replay, except near-ties of the batched verification (top-2 gap reported); the
    /// committed state matches the delivered tokens.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn greedy_lookup_session_delivers_the_target_choices() -> anyhow::Result<()> {
        use std::path::Path;
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let rows: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let mut text = Vec::new();
        for row in rows["rows"].as_array().context("rows")? {
            text.extend(serde_json::from_value::<Vec<u32>>(row["tokens"].clone())?);
        }
        let vocab = model.unpadded_vocab() as u32;
        let passage: Vec<u32> = text[..300].iter().map(|t| t % vocab).collect();
        let mut prompt = passage.clone();
        prompt.extend(&passage[..40]);
        let steps = 160usize;
        let context = model.context_limit() as usize;
        let prefill = |model: &mut Inkling2| -> anyhow::Result<Array> {
            model.reset();
            let mut logits = None;
            for chunk in prompt.chunks(128) {
                logits = Some(model.forward_tokens(chunk)?);
            }
            logits.context("no prompt")
        };
        let logits = prefill(&mut model)?;
        let first = logits.chat_greedy_ids()?[0];
        let mut session =
            LookupSession::new(&model, &prompt, DrawRule::Greedy, DrawSeeds::new(3), true);
        let mut choose = |proposals: &[u32], verified: &Array| -> anyhow::Result<(usize, u32)> {
            let a = crate::speculative::greedy_accept(proposals, &verified.chat_greedy_ids()?)
                .context("greedy acceptance")?;
            Ok((a.accepted_draft_tokens, a.target_token))
        };
        let mut delivered = vec![first];
        for i in 0..steps {
            let next = *delivered.last().unwrap();
            let remaining = steps - i;
            delivered.push(session.advance(&mut model, next, context, remaining, &mut choose)?);
        }
        let stats = session.stats.clone();
        // committed: the prompt and every delivered token but the last, plus verified tokens not delivered
        // yet and the step run ahead
        let committed = model.offset() as usize;
        let low = prompt.len() + delivered.len() - 1;
        ensure!(
            committed >= low && committed <= low + MAX_WIDTH + 1,
            "committed {committed} for {low} delivered positions"
        );
        drop(session);
        // teacher-forced AR replay of the delivered tokens
        let mut logits = prefill(&mut model)?;
        let (mut mismatches, mut worst_gap) = (0usize, 0f32);
        for pair in delivered.windows(2) {
            let rows = logits.log_probs()?.to_f32()?;
            let best = logits.chat_greedy_ids()?[0];
            if pair[0] != best {
                mismatches += 1;
                worst_gap = worst_gap.max(rows[best as usize] - rows[pair[0] as usize]);
            }
            logits = model.forward(pair[0])?;
        }
        println!(
            "lookup greedy: delivered {} blocks {} proposed {} accepted {} plain {} held {} \
             block_ms {:.1} plain_ms {:.1} widths {:?}; AR replay mismatches {mismatches} worst gap {worst_gap:.5}",
            delivered.len(),
            stats.blocks,
            stats.proposed,
            stats.accepted,
            stats.plain_steps,
            stats.held_steps,
            stats.block_ms,
            stats.plain_ms,
            stats.width_blocks
        );
        ensure!(
            stats.blocks > 0 && stats.accepted > 0,
            "no lookup block ran"
        );
        ensure!(
            mismatches <= 2 && worst_gap < 0.05,
            "delivered tokens are not the target choices"
        );
        Ok(())
    }

    #[test]
    fn costs_increase_with_width() {
        assert_eq!(BLOCK_COSTS[0], 1.0);
        assert!(BLOCK_COSTS.windows(2).all(|w| w[0] < w[1]));
    }
}
