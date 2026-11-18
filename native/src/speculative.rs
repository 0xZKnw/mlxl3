//! Exact target-side acceptance for speculative decoding.

/// Serial target verification. Each call consumes one input; its output is
/// emitted but is not consumed after EOS or rejection. The caller bounds the
/// proposal slice to remaining output/context capacity minus one.
pub fn verify_serial<E>(
    anchor: u32,
    proposals: &[u32],
    eos_ids: &[u32],
    mut forward: impl FnMut(u32) -> Result<u32, E>,
) -> Result<(Vec<u32>, usize), E> {
    let mut emitted = Vec::new();
    let mut accepted = 0;
    let mut current = anchor;
    for proposed in proposals
        .iter()
        .copied()
        .map(Some)
        .chain(std::iter::once(None))
    {
        let next = forward(current)?;
        emitted.push(next);
        if proposed == Some(next) {
            accepted += 1;
        }
        if eos_ids.contains(&next) || proposed != Some(next) {
            break;
        }
        current = next;
    }
    Ok((emitted, accepted))
}

#[test]
fn serial_verification_stops_at_terminal_and_matches_ar_oracle() {
    // Independent AR sequence is 1,2,3,4; no forward of terminal is permitted.
    for budget in 1..=4 {
        for terminal in 1..=4 {
            for rejected in 0..=budget {
                let proposals: Vec<_> = (1..budget)
                    .map(|i| if i == rejected { 99 } else { i })
                    .collect();
                let mut inputs = Vec::new();
                let (tokens, accepted) = verify_serial(0, &proposals, &[terminal], |input| {
                    assert_ne!(input, terminal, "forward after terminal");
                    inputs.push(input);
                    Ok::<_, ()>(input + 1)
                })
                .unwrap();
                let stop = if rejected > 0 && rejected < budget {
                    rejected
                } else {
                    budget
                };
                let count = stop.min(terminal);
                assert_eq!(tokens, (1..=count).collect::<Vec<_>>());
                assert_eq!(inputs, (0..count).collect::<Vec<_>>());
                let matching = proposals
                    .iter()
                    .zip(&tokens)
                    .take_while(|(a, b)| a == b)
                    .count();
                assert_eq!(accepted, matching);
                assert!(tokens.len() <= budget as usize);
            }
        }
    }
    let mut calls = 0;
    assert!(
        verify_serial(0, &[1, 2], &[], |_| {
            calls += 1;
            Err::<u32, _>("fault")
        })
        .is_err()
    );
    assert_eq!(calls, 1);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(6)]
fn serial_verification_never_consumes_eos() {
    let proposals: [u32; 3] = kani::any();
    let outputs: [u32; 4] = kani::any();
    let terminal: u32 = kani::any();
    let n: usize = kani::any();
    kani::assume(n <= 3);
    let mut calls = 0;
    let (tokens, accepted) = verify_serial(0, &proposals[..n], &[terminal], |_| {
        if calls > 0 {
            assert_ne!(outputs[calls - 1], terminal);
        }
        let token = outputs[calls];
        calls += 1;
        Ok::<_, ()>(token)
    })
    .unwrap();
    assert!(calls > 0 && calls <= n + 1 && accepted <= n);
    assert_eq!(tokens.len(), calls);
    kani::cover!(accepted == 1 && tokens.last() == Some(&terminal));
    kani::cover!(calls == 4);
}

/// Select the first maximum after model-dtype normalization. Negative infinity masks a token.
pub fn greedy_log_prob_id(probabilities: &[f32]) -> Option<u32> {
    let mut best = None;
    for (index, &value) in probabilities.iter().enumerate() {
        if value.is_nan() || value == f32::INFINITY {
            return None;
        }
        if value.is_finite() && best.is_none_or(|(_, prior)| value > prior) {
            best = Some((u32::try_from(index).ok()?, value));
        }
    }
    best.map(|(index, _)| index)
}

#[test]
fn greedy_probabilities_require_a_finite_winner_and_keep_first_tie() {
    assert_eq!(
        greedy_log_prob_id(&[f32::NEG_INFINITY, -0.0, 0.0, -1.0]),
        Some(1)
    );
    for row in [
        &[][..],
        &[f32::NEG_INFINITY],
        &[0., f32::NAN],
        &[f32::INFINITY, 0.],
    ] {
        assert_eq!(greedy_log_prob_id(row), None);
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(6)]
fn normalized_greedy_retains_first_finite_maximum() {
    let p: [f32; 3] = kani::any();
    let n = usize::from(kani::any::<u8>() % 4);
    let row = &p[..n];
    let result = greedy_log_prob_id(row);
    assert_eq!(
        result.is_some(),
        row.iter().all(|v| !v.is_nan() && *v != f32::INFINITY) && row.iter().any(|v| v.is_finite())
    );
    if let Some(index) = result {
        let i = index as usize;
        assert!(i < n && row[i].is_finite());
        for (j, &x) in row.iter().enumerate() {
            assert!(if j < i { x < row[i] } else { x <= row[i] });
        }
        kani::cover!(n == 3 && i == 1 && row[1] == row[2]);
        kani::cover!(n == 3 && i == 2 && row[0] == f32::NEG_INFINITY);
    }
}

/// The accepted draft prefix plus the target token that ends the cycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GreedyAcceptance {
    pub accepted_draft_tokens: usize,
    pub target_token: u32,
}

/// Verify one greedy proposal without evaluating its successor on rejection.
/// The callback must advance the target exactly once and return its greedy ID.
/// Errors are propagated so the caller can restore its target and draft snapshots.
pub fn greedy_accept_one<E>(
    proposal: u32,
    first: u32,
    successor: impl FnOnce() -> Result<u32, E>,
) -> Result<GreedyAcceptance, E> {
    if first != proposal {
        return Ok(GreedyAcceptance {
            accepted_draft_tokens: 0,
            target_token: first,
        });
    }
    Ok(GreedyAcceptance {
        accepted_draft_tokens: 1,
        target_token: successor()?,
    })
}

#[test]
fn greedy_one_rejects_without_successor_and_preserves_bonus() {
    for proposal in 0..8 {
        for first in 0..8 {
            let mut calls = 0;
            let actual = greedy_accept_one(proposal, first, || {
                calls += 1;
                Ok::<_, ()>(7)
            })
            .unwrap();
            assert_eq!(actual, greedy_accept(&[proposal], &[first, 7]).unwrap());
            assert_eq!(calls, usize::from(first == proposal));
        }
    }
    assert_eq!(greedy_accept_one(3, 3, || Err("failed")), Err("failed"));
    assert_eq!(
        greedy_accept_one(3, 2, || Err("must not run")),
        Ok(GreedyAcceptance {
            accepted_draft_tokens: 0,
            target_token: 2,
        })
    );
}

#[cfg(kani)]
#[kani::proof]
fn greedy_one_successor_is_evaluated_iff_accepted() {
    let proposal: u32 = kani::any();
    let first: u32 = kani::any();
    let second: u32 = kani::any();
    let fail: bool = kani::any();
    let mut calls = 0;
    let result = greedy_accept_one(proposal, first, || {
        calls += 1;
        if fail { Err(()) } else { Ok(second) }
    });
    assert_eq!(calls, usize::from(proposal == first));
    if first != proposal {
        assert_eq!(
            result,
            Ok(GreedyAcceptance {
                accepted_draft_tokens: 0,
                target_token: first,
            })
        );
    } else if fail {
        assert!(result.is_err());
    } else {
        assert_eq!(
            result,
            Ok(GreedyAcceptance {
                accepted_draft_tokens: 1,
                target_token: second,
            })
        );
    }
    kani::cover!(first != proposal && fail);
    kani::cover!(first == proposal && fail);
    kani::cover!(first == proposal && !fail);
}

/// Keep room for the anchor/target token in both the context and output budget.
/// Zero proposals means one ordinary target step, not a truncated draft network.
pub fn bounded_proposals(context_remaining: usize, output_remaining: usize) -> Option<usize> {
    context_remaining
        .min(output_remaining)
        .checked_sub(1)
        .map(|n| n.min(5))
}

/// Public MTP depths are 1..=3. Reserve the target token even at the last step.
pub fn mtp_width(depth: usize, context: usize, output: usize) -> Option<usize> {
    if !(1..=3).contains(&depth) {
        return None;
    }
    bounded_proposals(context, output).map(|n| n.min(depth))
}

pub const LOOKUP_WINDOW: usize = 1024;

/// Continue the latest earlier occurrence of seven committed IDs plus the anchor.
/// Returned IDs are only proposals; the target must verify their entire prefix.
pub fn prompt_lookup(
    history: &[u32],
    anchor: u32,
    width: usize,
    allow_context: impl Fn(&[u32]) -> bool,
) -> Option<&[u32]> {
    prompt_lookup_bounded(history, anchor, width, 3, allow_context)
}

/// `prompt_lookup` with widths up to `limit` (Q27D-19: DFlash2 copies of up to seven IDs).
pub fn prompt_lookup_bounded(
    history: &[u32],
    anchor: u32,
    width: usize,
    limit: usize,
    allow_context: impl Fn(&[u32]) -> bool,
) -> Option<&[u32]> {
    if history.len() < 9 || width == 0 || width > limit.min(7) {
        return None;
    }
    let history = &history[history.len().saturating_sub(LOOKUP_WINDOW)..];
    let tail = &history[history.len() - 7..];
    if !allow_context(tail) {
        return None;
    }
    // ponytail: scan at most 1024 IDs; index only if this becomes measurable.
    let start = history[..history.len() - 1]
        .windows(8)
        .rposition(|site| site[..7] == *tail && site[7] == anchor)?;
    let end = start + 8;
    Some(&history[end..end + width.min(history.len() - end)])
}

/// Longest n-gram of `NgramIndex` (the anchor included).
pub const NGRAM_MAX: usize = 8;
/// Shortest n-gram of `NgramIndex` (L3: two IDs).
pub const NGRAM_MIN: usize = 2;

/// L4+L3 (`MLXL3_LFM3_LOOKUP=v2`): prompt lookup over the whole context.
/// The proposal continues the latest earlier occurrence of the longest
/// suffix of `NGRAM_MAX` down to `NGRAM_MIN` IDs (the anchor included), as
/// the R2 simulation. One hash map per length stores the latest end of each
/// n-gram; a hit is compared with the IDs, and a hash collision falls back to
/// a scan, so the result is the one of `ngram_lookup_scan`.
pub struct NgramIndex {
    sequence: Vec<u32>,
    /// N-grams ending at most at this length are indexed.
    indexed: usize,
    ends: [std::collections::HashMap<u64, usize>; NGRAM_MAX - NGRAM_MIN + 1],
}

fn ngram_hash(ids: &[u32]) -> u64 {
    let mut hash = 0x9e37_79b9_7f4a_7c15u64 ^ ids.len() as u64;
    for &id in ids {
        hash = (hash.rotate_left(5) ^ u64::from(id)).wrapping_mul(0x5170_9b72_2ac1_29b3);
    }
    hash ^ (hash >> 29)
}

/// Reference of `NgramIndex::lookup`: a backward scan per length.
pub fn ngram_lookup_scan(sequence: &[u32], width: usize) -> Option<&[u32]> {
    let length = sequence.len();
    if width == 0 {
        return None;
    }
    for n in (NGRAM_MIN..=NGRAM_MAX).rev() {
        if n >= length {
            continue;
        }
        let gram = &sequence[length - n..];
        if let Some(end) = (n..length)
            .rev()
            .find(|&end| &sequence[end - n..end] == gram)
        {
            return Some(&sequence[end..(end + width).min(length)]);
        }
    }
    None
}

impl NgramIndex {
    pub fn new(prompt: &[u32]) -> Self {
        Self {
            sequence: prompt.to_vec(),
            indexed: 0,
            ends: Default::default(),
        }
    }

    pub fn push(&mut self, id: u32) {
        self.sequence.push(id);
    }

    pub fn sequence(&self) -> &[u32] {
        &self.sequence
    }

    fn index_until(&mut self, limit: usize) {
        while self.indexed < limit {
            self.indexed += 1;
            let end = self.indexed;
            for n in NGRAM_MIN..=NGRAM_MAX.min(end) {
                let hash = ngram_hash(&self.sequence[end - n..end]);
                self.ends[n - NGRAM_MIN].insert(hash, end);
            }
        }
    }

    /// Up to `width` proposed IDs after the current sequence (which ends with
    /// the anchor).
    pub fn lookup(&mut self, width: usize) -> Option<&[u32]> {
        let length = self.sequence.len();
        if width == 0 || length < NGRAM_MIN + 1 {
            return None;
        }
        self.index_until(length - 1);
        let mut found = None;
        for n in (NGRAM_MIN..=NGRAM_MAX).rev() {
            if n >= length {
                continue;
            }
            let gram = &self.sequence[length - n..];
            match self.ends[n - NGRAM_MIN].get(&ngram_hash(gram)) {
                Some(&end) if &self.sequence[end - n..end] == gram => {
                    found = Some(end);
                    break;
                }
                Some(_) => {
                    // Collision: the map may hide an older occurrence.
                    if let Some(end) = (n..length)
                        .rev()
                        .find(|&end| &self.sequence[end - n..end] == gram)
                    {
                        found = Some(end);
                        break;
                    }
                }
                None => {}
            }
        }
        let end = found?;
        Some(&self.sequence[end..(end + width).min(length)])
    }
}

/// Request-local copy proposals; target verification remains authoritative.
pub struct DFlashCopy {
    history: [u32; LOOKUP_WINDOW],
    len: usize,
    enabled: bool,
}

impl DFlashCopy {
    pub fn new(prompt: &[u32]) -> Self {
        let prompt = &prompt[prompt.len().saturating_sub(LOOKUP_WINDOW)..];
        let mut history = [0; LOOKUP_WINDOW];
        history[..prompt.len()].copy_from_slice(prompt);
        Self {
            history,
            len: prompt.len(),
            enabled: true,
        }
    }

    pub fn propose(&self, anchor: u32, width: usize) -> Option<&[u32]> {
        self.propose_up_to(anchor, width, 3)
    }

    /// Q27D-19: wider copies when verification of long blocks is cheap (TensorOps mode).
    pub fn propose_up_to(&self, anchor: u32, width: usize, limit: usize) -> Option<&[u32]> {
        self.enabled
            .then(|| {
                prompt_lookup_bounded(
                    &self.history[..self.len],
                    anchor,
                    width.min(limit),
                    limit,
                    |_| true,
                )
            })
            .flatten()
    }

    /// Call only after target commit; exclude the pending corrective token.
    pub fn record(&mut self, anchor: u32, accepted: &[u32], copied: bool) {
        self.enabled &= !copied || !accepted.is_empty();
        if accepted.len() >= LOOKUP_WINDOW {
            self.history
                .copy_from_slice(&accepted[accepted.len() - LOOKUP_WINDOW..]);
            self.len = LOOKUP_WINDOW;
        } else {
            let retained = self.len.min(LOOKUP_WINDOW - accepted.len() - 1);
            let excess = self.len - retained;
            if excess > 0 {
                self.history.copy_within(excess..self.len, 0);
            }
            self.history[retained] = anchor;
            let end = retained + 1 + accepted.len();
            self.history[retained + 1..end].copy_from_slice(accepted);
            self.len = end;
        }
    }
}

/// Scores are positive milli-tokens/second, after quality/acceptance validation.
/// Prefer the shallower mode on ties and baseline inside the 3% noise margin.
pub fn best_mtp_depth(scores: [Option<u64>; 4]) -> Option<usize> {
    let baseline = scores[0].filter(|&score| score > 0)?;
    let mut best = 0;
    let mut fastest = baseline;
    for (depth, score) in scores.into_iter().enumerate().skip(1) {
        if let Some(score) = score
            && score > fastest
            && u128::from(score) * 100 > u128::from(baseline) * 103
        {
            best = depth;
            fastest = score;
        }
    }
    Some(best)
}

pub fn mtp_tuning_score(
    depth: usize,
    tps: f64,
    parity: bool,
    accepted: usize,
    proposed: usize,
) -> Option<u64> {
    if depth > 3
        || !parity
        || !tps.is_finite()
        || tps <= 0.
        || tps >= u64::MAX as f64 / 1000.
        || (depth > 0 && (accepted == 0 || proposed == 0 || accepted > proposed))
    {
        return None;
    }
    Some((tps * 1000.) as u64).filter(|&score| score > 0)
}

/// Shorter verification blocks after repeated low draft acceptance.
pub fn adaptive_proposals(
    context_remaining: usize,
    output_remaining: usize,
    blocks: usize,
    accepted: usize,
    proposed: usize,
) -> Option<usize> {
    let width = if blocks >= 8 && accepted.saturating_mul(5) < proposed.saturating_mul(3) {
        2
    } else {
        5
    };
    bounded_proposals(context_remaining, output_remaining).map(|n| n.min(width))
}

/// 1 = current adaptive policy, 2 = two proposals, 3 = seven proposals.
/// Every proposal remains verified by the target; reserve the corrective token.
pub fn dflash_proposals(
    mode: usize,
    context: usize,
    output: usize,
    blocks: usize,
    accepted: usize,
    proposed: usize,
) -> Option<usize> {
    match mode {
        1 => adaptive_proposals(context, output, blocks, accepted, proposed),
        2 | 3 => context
            .min(output)
            .checked_sub(1)
            .map(|n| n.min(if mode == 2 { 2 } else { 7 })),
        _ => None,
    }
}

/// Q27D-08 (D1): widest adaptive DFlash2 block for the dense 27B once its verification
/// shares the decode of up to four rows (`MLXL3_DENSE_SHARED_VERIFY`): three proposals keep
/// the block at four verified rows, below the five-row cost step. `None` keeps the policy.
pub fn dflash_width_cap(dense: bool, shared_verify: bool, mode: usize) -> Option<usize> {
    (dense && shared_verify && mode == 1).then_some(3)
}

#[test]
fn dflash_width_cap_only_limits_dense_adaptive_blocks() {
    for mode in 0..=4 {
        for dense in [false, true] {
            for shared in [false, true] {
                let cap = dflash_width_cap(dense, shared, mode);
                assert_eq!(cap.is_some(), dense && shared && mode == 1);
                if let Some(cap) = cap {
                    assert_eq!(cap, 3);
                    // The capped block still reserves the corrective token.
                    for output in 0..10 {
                        let width = dflash_proposals(1, 4096, output, 8, 8, 8).map(|n| n.min(cap));
                        assert!(width.is_none_or(|n| n <= 3 && n < output.max(1)));
                    }
                }
            }
        }
    }
}

/// Accepts the longest proposal prefix selected by the target.
///
/// `target_tokens[i]` is the target's token after the first `i` proposals.
/// One additional target token is therefore required even when every proposal
/// is accepted. Returning it makes the target, rather than the drafter, the
/// source of every committed token.
pub fn greedy_accept(proposals: &[u32], target_tokens: &[u32]) -> Option<GreedyAcceptance> {
    if target_tokens.len() != proposals.len().checked_add(1)? {
        return None;
    }
    let accepted = proposals
        .iter()
        .zip(target_tokens)
        .take_while(|(draft, target)| draft == target)
        .count();
    Some(GreedyAcceptance {
        accepted_draft_tokens: accepted,
        target_token: target_tokens[accepted],
    })
}

/// Widest MTP proposal chain a cycle can verify.
pub const MAX_MTP_WIDTH: usize = 3;

/// Q27D-37: a DFlash2 verification tree. Row 0 is the anchor; rows are in depth order and
/// every row's parent precedes it, so a row's ancestors are a prefix-closed set.
/// Q27D-47: (accepted rows from the root, accepted draft tokens, token drawn after them,
/// summed (KL(p̂ ‖ p), −Σ p̂ log p, −Σ p log p) of the emitted tokens' laws).
pub type RelaxedAcceptance = (Vec<usize>, usize, u32, (f64, f64, f64));

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftTree {
    pub tokens: Vec<u32>,
    pub parents: Vec<usize>,
    pub depths: Vec<usize>,
}

/// Lattice width of the DFlash2 selector (top-16 candidates per position).
pub const SELECTOR_CANDIDATES: usize = 16;

impl DraftTree {
    pub fn rows(&self) -> usize {
        self.tokens.len()
    }

    /// Root-to-leaf paths as row lists, in row order of their leaves.
    pub fn paths(&self) -> Vec<Vec<usize>> {
        let rows = self.rows();
        let mut has_child = vec![false; rows];
        for row in 1..rows {
            has_child[self.parents[row]] = true;
        }
        (0..rows)
            .filter(|&row| !has_child[row])
            .map(|leaf| {
                let mut path = vec![leaf];
                let mut row = leaf;
                while row != 0 {
                    row = self.parents[row];
                    path.push(row);
                }
                path.reverse();
                path
            })
            .collect()
    }

    /// Whether `ancestor` is `row` or one of its ancestors.
    pub fn is_ancestor(&self, ancestor: usize, row: usize) -> bool {
        let mut row = row;
        loop {
            if row == ancestor {
                return true;
            }
            if row == 0 {
                return false;
            }
            row = self.parents[row];
        }
    }

    /// Greedy descent: from the root, follow the child whose token is the target's greedy
    /// token at the current row. Returns the accepted rows (root first), the number of
    /// accepted draft tokens and the target's token after the last accepted row.
    pub fn accept_greedy(&self, target_tokens: &[u32]) -> Option<(Vec<usize>, usize, u32)> {
        if target_tokens.len() != self.rows() || self.rows() == 0 {
            return None;
        }
        let mut path = vec![0usize];
        let mut row = 0usize;
        loop {
            let wanted = target_tokens[row];
            let next = (row + 1..self.rows())
                .find(|&child| self.parents[child] == row && self.tokens[child] == wanted);
            match next {
                Some(child) => {
                    path.push(child);
                    row = child;
                }
                None => return Some((path.clone(), path.len() - 1, wanted)),
            }
        }
    }

    /// Q27D-40: exact speculative sampling down the tree. `target(row, extra)` is the target
    /// law of `row`, `extra` being the tokens of its path below the root (penalty history).
    /// At each node the children are verified in order by `Distribution::verify_among`;
    /// returns (accepted rows from the root, accepted draft tokens, token drawn after them).
    pub fn accept_sampled(
        &self,
        target: impl FnMut(usize, &[u32]) -> anyhow::Result<crate::sampling::Distribution>,
        random: &mut u64,
    ) -> anyhow::Result<(Vec<usize>, usize, u32)> {
        let (path, accepted, token, _) = self.accept_sampled_relaxed(target, random, 0.)?;
        Ok((path, accepted, token))
    }

    /// Q27D-47: `accept_sampled` where each node's law is first inclined toward its children
    /// under the divergence budget `relax` (nats per emitted token, 0 = exact). Also returns
    /// the summed (KL(p̂ ‖ p), −Σ p̂ log p, −Σ p log p) of the emitted tokens' laws.
    pub fn accept_sampled_relaxed(
        &self,
        mut target: impl FnMut(usize, &[u32]) -> anyhow::Result<crate::sampling::Distribution>,
        random: &mut u64,
        relax: f64,
    ) -> anyhow::Result<RelaxedAcceptance> {
        anyhow::ensure!(self.is_valid(), "invalid verification tree");
        let mut path = vec![0usize];
        let mut extra = Vec::new();
        let mut row = 0usize;
        let mut terms = (0., 0., 0.);
        loop {
            let law = target(row, &extra)?;
            let children = (row + 1..self.rows())
                .filter(|&child| self.parents[child] == row)
                .collect::<Vec<_>>();
            let candidates = children
                .iter()
                .map(|&child| self.tokens[child])
                .collect::<Vec<_>>();
            let law = if relax > 0. && !candidates.is_empty() {
                let (tilted, _) = law.tilted_toward(&candidates, relax)?;
                let (kl, cross, entropy) = tilted.divergence_terms(&law);
                terms = (terms.0 + kl, terms.1 + cross, terms.2 + entropy);
                tilted
            } else {
                if relax > 0. {
                    let (_, cross, entropy) = law.divergence_terms(&law);
                    terms = (terms.0, terms.1 + cross, terms.2 + entropy);
                }
                law
            };
            match law.verify_among(&candidates, random)? {
                crate::sampling::Choice::Candidate(index) => {
                    row = children[index];
                    path.push(row);
                    extra.push(self.tokens[row]);
                }
                crate::sampling::Choice::Token(token) => {
                    let accepted = path.len() - 1;
                    return Ok((path, accepted, token, terms));
                }
            }
        }
    }

    /// Q27D-48: expected accepted draft tokens of a sampled tree verification and expected
    /// summed KL of the emitted tokens' laws, when every node's law (`laws[row]`, built with
    /// its path history) is inclined toward its children with budget `relax` (0 = exact).
    pub fn expected_acceptance(
        &self,
        laws: &[crate::sampling::Distribution],
        relax: f64,
    ) -> anyhow::Result<(f64, f64)> {
        anyhow::ensure!(
            self.is_valid() && laws.len() == self.rows(),
            "invalid tree or laws"
        );
        let mut reach = vec![0f64; self.rows()];
        reach[0] = 1.;
        let (mut accepted, mut divergence) = (0f64, 0f64);
        for row in 0..self.rows() {
            if reach[row] == 0. {
                continue;
            }
            let children = (row + 1..self.rows())
                .filter(|&child| self.parents[child] == row)
                .collect::<Vec<_>>();
            let candidates = children
                .iter()
                .map(|&child| self.tokens[child])
                .collect::<Vec<_>>();
            let (law, kl) = if relax > 0. && !candidates.is_empty() {
                laws[row].tilted_toward(&candidates, relax)?
            } else {
                (laws[row].clone(), 0.)
            };
            divergence += reach[row] * kl;
            for (&child, &token) in children.iter().zip(&candidates) {
                reach[child] = reach[row] * law.probability(token);
                accepted += reach[child];
            }
        }
        Ok((accepted, divergence))
    }

    /// Structural invariants: root row, parents before children, depth = parent depth + 1,
    /// and distinct sibling tokens.
    pub fn is_valid(&self) -> bool {
        let rows = self.rows();
        if rows == 0
            || self.parents.len() != rows
            || self.depths.len() != rows
            || self.depths[0] != 0
        {
            return false;
        }
        (1..rows).all(|row| {
            let parent = self.parents[row];
            parent < row
                && self.depths[row] == self.depths[parent] + 1
                && self.depths[row] >= self.depths[row - 1]
                && !(1..row).any(|other| {
                    self.parents[other] == parent && self.tokens[other] == self.tokens[row]
                })
        })
    }
}

/// Q27D-36/37: best-first tree over the DFlash2 selector lattice (TensorFold's scheme). A node
/// at depth `d` (1-based) holds candidate `c` of position `d - 1`; its children are the 16
/// candidates of the next position scored by `log_softmax((unary + scale * edge[c]) / tau)`, and
/// nodes are taken in order of cumulative score until `nodes` draft nodes are chosen.
/// `edges[(p * 16 + parent) * 16 + child]`; position 0 only has parent index 0 (the anchor).
#[allow(clippy::too_many_arguments)]
pub fn build_selector_tree(
    anchor: u32,
    candidates: &[u32],
    unary: &[f32],
    edges: &[f32],
    positions: usize,
    nodes: usize,
    scale: f32,
    tau: f32,
) -> Option<DraftTree> {
    const C: usize = SELECTOR_CANDIDATES;
    if positions == 0
        || nodes == 0
        || candidates.len() != positions * C
        || unary.len() != positions * C
        || edges.len() != positions * C * C
        || !(tau.is_finite() && tau > 0.0 && scale.is_finite())
        || !unary.iter().all(|v| v.is_finite())
    {
        return None;
    }
    let log_softmax = |position: usize, parent: usize| -> Option<[f32; C]> {
        let mut scores = [0f32; C];
        for (child, score) in scores.iter_mut().enumerate() {
            let edge = edges[(position * C + parent) * C + child];
            if !edge.is_finite() {
                return None;
            }
            *score = (unary[position * C + child] + scale * edge) / tau;
        }
        let top = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let total = scores.iter().map(|s| (s - top).exp()).sum::<f32>().ln() + top;
        for score in &mut scores {
            *score -= total;
        }
        Some(scores)
    };
    // Frontier entries: (cumulative score, sequence number, parent row, position, candidate).
    let mut frontier: Vec<(f32, usize, usize, usize, usize)> = Vec::new();
    let mut sequence = 0usize;
    for (child, score) in log_softmax(0, 0)?.into_iter().enumerate() {
        frontier.push((score, sequence, 0, 0, child));
        sequence += 1;
    }
    let mut tokens = vec![anchor];
    let mut parents = vec![0usize];
    let mut depths = vec![0usize];
    let mut cumulative = vec![0f32];
    let mut candidate_of = vec![usize::MAX];
    while tokens.len() <= nodes && !frontier.is_empty() {
        // Highest score first; earlier insertion breaks ties.
        let best = (0..frontier.len()).max_by(|&a, &b| {
            frontier[a]
                .0
                .total_cmp(&frontier[b].0)
                .then(frontier[b].1.cmp(&frontier[a].1))
        })?;
        let (score, _, parent, position, child) = frontier.swap_remove(best);
        let row = tokens.len();
        tokens.push(candidates[position * C + child]);
        parents.push(parent);
        depths.push(position + 1);
        cumulative.push(score);
        candidate_of.push(child);
        if position + 1 < positions {
            for (next, step) in log_softmax(position + 1, child)?.into_iter().enumerate() {
                frontier.push((score + step, sequence, row, position + 1, next));
                sequence += 1;
            }
        }
    }
    // Reorder rows by depth (stable), keeping parents before children.
    let mut order: Vec<usize> = (0..tokens.len()).collect();
    order.sort_by_key(|&row| depths[row]);
    let mut position_of = vec![0usize; tokens.len()];
    for (new, &old) in order.iter().enumerate() {
        position_of[old] = new;
    }
    let tree = DraftTree {
        tokens: order.iter().map(|&old| tokens[old]).collect(),
        parents: order
            .iter()
            .map(|&old| {
                if old == 0 {
                    0
                } else {
                    position_of[parents[old]]
                }
            })
            .collect(),
        depths: order.iter().map(|&old| depths[old]).collect(),
    };
    let _ = (cumulative, candidate_of);
    tree.is_valid().then_some(tree)
}

/// Completed MTP cycles, indexed `[proposal width][accepted prefix]`.
///
/// Width 0 is an ordinary target step: the last token of a budget, or an
/// adaptive D0 step. This is host bookkeeping only. It never reaches the model,
/// so recording a cycle cannot change a token.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
#[serde(transparent)]
pub struct AcceptHistogram {
    counts: [[usize; MAX_MTP_WIDTH + 1]; MAX_MTP_WIDTH + 1],
}

impl AcceptHistogram {
    /// Counts one cycle. An impossible cycle (`accepted > width` or
    /// `width > MAX_MTP_WIDTH`) returns false and leaves every count unchanged.
    pub fn record(&mut self, width: usize, accepted: usize) -> bool {
        if width > MAX_MTP_WIDTH || accepted > width {
            return false;
        }
        let cell = &mut self.counts[width][accepted];
        *cell = cell.saturating_add(1);
        true
    }

    pub fn merge(&mut self, other: &Self) {
        for (row, other) in self.counts.iter_mut().zip(&other.counts) {
            for (cell, &value) in row.iter_mut().zip(other) {
                *cell = cell.saturating_add(value);
            }
        }
    }

    pub fn counts(&self) -> &[[usize; MAX_MTP_WIDTH + 1]; MAX_MTP_WIDTH + 1] {
        &self.counts
    }

    fn weighted(&self, weight: impl Fn(usize, usize) -> usize) -> usize {
        let mut total = 0usize;
        for (width, row) in self.counts.iter().enumerate() {
            for (accepted, &count) in row.iter().enumerate() {
                total = total.saturating_add(count.saturating_mul(weight(width, accepted)));
            }
        }
        total
    }

    /// Every recorded cycle, width-0 target steps included.
    pub fn cycles(&self) -> usize {
        self.weighted(|_, _| 1)
    }

    /// Cycles that verified at least one proposal.
    pub fn speculative_cycles(&self) -> usize {
        self.weighted(|width, _| usize::from(width > 0))
    }

    pub fn proposed(&self) -> usize {
        self.weighted(|width, _| width)
    }

    pub fn accepted(&self) -> usize {
        self.weighted(|_, accepted| accepted)
    }

    /// `(accepted, tested)` at 0-based draft `position`: cycles that proposed
    /// it after every earlier proposal was accepted, and those that accepted it.
    /// Greedy acceptance keeps a prefix, so the ratio is the conditional rate.
    pub fn position(&self, position: usize) -> (usize, usize) {
        (
            self.weighted(|width, accepted| usize::from(width > position && accepted > position)),
            self.weighted(|width, accepted| usize::from(width > position && accepted >= position)),
        )
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn accept_histogram_record_is_exact_and_rejects_impossible_cycles() {
    // `record` is the only writer, so a reachable histogram has no cell with
    // more accepted proposals than its width. Counts are bounded so that the
    // exact totals below cannot saturate; saturation has its own harness.
    let mut counts = [[0usize; MAX_MTP_WIDTH + 1]; MAX_MTP_WIDTH + 1];
    for (width, row) in counts.iter_mut().enumerate() {
        for (accepted, cell) in row.iter_mut().enumerate() {
            if accepted <= width {
                *cell = usize::from(kani::any::<u16>());
            }
        }
    }
    let mut histogram = AcceptHistogram { counts };
    let before = histogram;
    let width: usize = kani::any();
    let accepted: usize = kani::any();
    let position: usize = kani::any();
    kani::assume(position <= MAX_MTP_WIDTH + 1);
    let recorded = histogram.record(width, accepted);
    assert_eq!(recorded, width <= MAX_MTP_WIDTH && accepted <= width);
    // Cell by cell: array equality lowers to a 128-byte memcmp loop.
    for row in 0..=MAX_MTP_WIDTH {
        for column in 0..=MAX_MTP_WIDTH {
            let target = recorded && row == width && column == accepted;
            assert_eq!(
                histogram.counts[row][column],
                before.counts[row][column] + usize::from(target)
            );
        }
    }
    if recorded {
        assert_eq!(histogram.cycles(), before.cycles() + 1);
        assert_eq!(
            histogram.speculative_cycles(),
            before.speculative_cycles() + usize::from(width > 0)
        );
        assert_eq!(histogram.proposed(), before.proposed() + width);
        assert_eq!(histogram.accepted(), before.accepted() + accepted);
        let (hit, tested) = histogram.position(position);
        let (old_hit, old_tested) = before.position(position);
        assert_eq!(
            tested,
            old_tested + usize::from(width > position && accepted >= position)
        );
        assert_eq!(
            hit,
            old_hit + usize::from(width > position && accepted > position)
        );
    }
    for position in 0..=MAX_MTP_WIDTH {
        let (hit, tested) = histogram.position(position);
        assert!(hit <= tested);
        if position < MAX_MTP_WIDTH {
            // Accepting position p is the only way to test position p + 1.
            assert!(histogram.position(position + 1).1 <= hit);
        }
    }
    assert_eq!(histogram.position(0).1, histogram.speculative_cycles());
    assert_eq!(histogram.position(MAX_MTP_WIDTH), (0, 0));
    kani::cover!(recorded && width == 0);
    kani::cover!(recorded && width == MAX_MTP_WIDTH && accepted == MAX_MTP_WIDTH);
    kani::cover!(recorded && width == 2 && accepted == 1 && position == 1);
    kani::cover!(!recorded && accepted > width && width <= MAX_MTP_WIDTH);
    kani::cover!(!recorded && width > MAX_MTP_WIDTH);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn accept_histogram_counts_saturate_without_wrapping() {
    let width: usize = kani::any();
    let accepted: usize = kani::any();
    kani::assume(width <= MAX_MTP_WIDTH && accepted <= width);
    let mut histogram = AcceptHistogram::default();
    histogram.counts[width][accepted] = usize::MAX;
    assert!(histogram.record(width, accepted));
    assert_eq!(histogram.counts[width][accepted], usize::MAX);
    assert_eq!(histogram.cycles(), usize::MAX);
    let mut merged = histogram;
    merged.merge(&histogram);
    assert_eq!(merged.counts[width][accepted], usize::MAX);
    assert_eq!(merged.cycles(), usize::MAX);
    assert!(merged.proposed() == 0 || merged.proposed() == usize::MAX);
    kani::cover!(width == MAX_MTP_WIDTH && accepted == 0);
    kani::cover!(width == 0);
}

#[cfg(kani)]
#[kani::proof]
fn dflash_modes_preserve_context_and_output_budgets() {
    let mode: usize = kani::any();
    let context: usize = kani::any();
    let output: usize = kani::any();
    let width = dflash_proposals(mode, context, output, kani::any(), kani::any(), kani::any());
    assert_eq!(
        width.is_some(),
        (1..=3).contains(&mode) && context > 0 && output > 0
    );
    if let Some(width) = width {
        assert!(width <= 7 && width < context && width < output);
        if mode == 2 {
            assert!(width <= 2);
        }
    }
    kani::cover!(mode == 3 && width == Some(7));
    kani::cover!(mode == 1 && width == Some(2));
    kani::cover!(width == Some(0));
    kani::cover!(width.is_none());
}

#[cfg(test)]
mod tests {
    use super::{DraftTree, build_selector_tree};

    fn lattice(positions: usize, seed: u32) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (state >> 8) as f32 / (1u32 << 24) as f32
        };
        let candidates = (0..positions * 16)
            .map(|i| 1000 + (i % 16) as u32 * 7 + (i / 16) as u32 * 131)
            .collect();
        let unary = (0..positions * 16)
            .map(|i| 10.0 - (i % 16) as f32 * 0.7 + next())
            .collect();
        let edges = (0..positions * 256).map(|_| next() * 4.0 - 2.0).collect();
        (candidates, unary, edges)
    }

    #[test]
    fn selector_tree_has_the_requested_nodes_in_depth_order() {
        for seed in 0..50u32 {
            for nodes in [1usize, 3, 7, 11, 15] {
                let (candidates, unary, edges) = lattice(7, seed);
                let tree = build_selector_tree(7, &candidates, &unary, &edges, 7, nodes, 0.6, 1.0)
                    .expect("tree");
                assert!(tree.is_valid());
                assert_eq!(tree.rows(), nodes + 1);
                assert_eq!(tree.tokens[0], 7);
                assert!(tree.depths.iter().all(|&d| d <= 7));
                let paths = tree.paths();
                let mut covered = vec![false; tree.rows()];
                for path in &paths {
                    assert_eq!(path[0], 0);
                    for pair in path.windows(2) {
                        assert_eq!(tree.parents[pair[1]], pair[0]);
                    }
                    for &row in path {
                        covered[row] = true;
                    }
                }
                assert!(covered.iter().all(|&c| c));
            }
        }
    }

    #[test]
    fn selector_tree_with_one_dominant_candidate_is_a_chain() {
        let positions = 7;
        let candidates: Vec<u32> = (0..positions * 16).map(|i| i as u32 + 1).collect();
        let unary: Vec<f32> = (0..positions * 16)
            .map(|i| if i % 16 == 3 { 50.0 } else { 0.0 })
            .collect();
        let edges = vec![0.0; positions * 256];
        let tree = build_selector_tree(0, &candidates, &unary, &edges, positions, 7, 0.6, 1.0)
            .expect("tree");
        assert_eq!(tree.depths, (0..=7).collect::<Vec<_>>());
        assert_eq!(tree.parents, vec![0, 0, 1, 2, 3, 4, 5, 6]);
        let chain: Vec<u32> = (0..positions).map(|p| candidates[p * 16 + 3]).collect();
        assert_eq!(&tree.tokens[1..], &chain[..]);
    }

    #[test]
    fn selector_tree_rejects_invalid_lattices() {
        let (candidates, unary, edges) = lattice(7, 1);
        assert!(build_selector_tree(0, &candidates, &unary, &edges, 7, 0, 0.6, 1.0).is_none());
        assert!(build_selector_tree(0, &candidates[1..], &unary, &edges, 7, 5, 0.6, 1.0).is_none());
        assert!(build_selector_tree(0, &candidates, &unary, &edges, 7, 5, 0.6, 0.0).is_none());
        let mut bad = unary.clone();
        bad[5] = f32::NAN;
        assert!(build_selector_tree(0, &candidates, &bad, &edges, 7, 5, 0.6, 1.0).is_none());
    }

    fn law(pairs: &[(u32, f64)]) -> crate::sampling::Distribution {
        crate::sampling::Distribution::Weighted {
            tokens: pairs.iter().map(|&(t, _)| t).collect(),
            weights: pairs.iter().map(|&(_, w)| w).collect(),
            total: pairs.iter().map(|&(_, w)| w).sum(),
        }
    }

    /// Q27D-40: point laws give the greedy path; a chain gives the chain verification.
    #[test]
    fn tree_sampled_acceptance_matches_greedy_and_chain() -> anyhow::Result<()> {
        let tree = DraftTree {
            tokens: vec![100, 10, 20, 30, 40, 50],
            parents: vec![0, 0, 0, 1, 2, 3],
            depths: vec![0, 1, 1, 2, 2, 3],
        };
        for targets in [
            [20u32, 30, 40, 99, 77, 88],
            [10, 30, 0, 50, 0, 5],
            [11, 0, 0, 0, 0, 0],
        ] {
            let mut random = 7;
            let sampled = tree.accept_sampled(
                |row, _| Ok(crate::sampling::Distribution::Point(targets[row])),
                &mut random,
            )?;
            assert_eq!(Some(sampled), tree.accept_greedy(&targets));
        }
        // A chain tree consumes the same draws as `Sampler::accept_with`.
        let chain = DraftTree {
            tokens: vec![100, 3, 7, 1],
            parents: vec![0, 0, 1, 2],
            depths: vec![0, 1, 2, 3],
        };
        let laws = [
            law(&[(3, 0.6), (7, 0.3), (9, 0.1)]),
            law(&[(7, 0.5), (2, 0.5)]),
            law(&[(1, 0.2), (4, 0.8)]),
            law(&[(5, 0.7), (6, 0.3)]),
        ];
        for seed in 1..2000u64 {
            let (mut a, mut b) = (seed * 0x9E37_79B9, seed * 0x9E37_79B9);
            let reference = crate::sampling::Sampler::accept_with(
                &[3, 7, 1],
                |index, _| Ok(laws[index].clone()),
                &mut a,
            )?;
            let (path, accepted, token) =
                chain.accept_sampled(|row, _| Ok(laws[row].clone()), &mut b)?;
            assert_eq!(
                (accepted, token, path.len(), a),
                (
                    reference.accepted_draft_tokens,
                    reference.target_token,
                    accepted + 1,
                    b
                )
            );
        }
        Ok(())
    }

    /// Q27D-48: the expected acceptance equals the greedy path for point laws and the mean
    /// of `accept_sampled_relaxed` for weighted laws, exact or inclined.
    #[test]
    fn tree_expected_acceptance_matches_sampling() -> anyhow::Result<()> {
        let tree = DraftTree {
            tokens: vec![100, 10, 20, 40, 60],
            parents: vec![0, 0, 0, 1, 2],
            depths: vec![0, 1, 1, 2, 2],
        };
        let greedy_targets = [10u32, 40, 0, 0, 0];
        let points = greedy_targets.map(crate::sampling::Distribution::Point);
        let (expected, kl) = tree.expected_acceptance(&points, 0.01)?;
        assert_eq!(
            (expected, kl),
            (tree.accept_greedy(&greedy_targets).unwrap().1 as f64, 0.)
        );
        let laws = [
            law(&[(10, 0.3), (20, 0.25), (30, 0.45)]),
            law(&[(40, 0.35), (50, 0.65)]),
            law(&[(60, 0.8), (70, 0.2)]),
            law(&[(80, 1.0)]),
            law(&[(90, 1.0)]),
        ];
        for relax in [0., 0.01, 0.05] {
            let (expected, expected_kl) = tree.expected_acceptance(&laws, relax)?;
            let blocks = 300_000usize;
            let mut random = 0xABCD_EF01_2345_6789;
            let (mut total, mut total_kl) = (0f64, 0f64);
            let mut squares = 0f64;
            for _ in 0..blocks {
                let (_, accepted, _, terms) = tree.accept_sampled_relaxed(
                    |row, _| Ok(laws[row].clone()),
                    &mut random,
                    relax,
                )?;
                total += accepted as f64;
                squares += (accepted * accepted) as f64;
                total_kl += terms.0;
            }
            let mean = total / blocks as f64;
            let sigma = ((squares / blocks as f64 - mean * mean) / blocks as f64).sqrt();
            assert!(
                (mean - expected).abs() <= 4. * sigma,
                "relax {relax}: {mean} vs {expected}"
            );
            assert!(
                (total_kl / blocks as f64 - expected_kl).abs() <= 1e-3,
                "relax {relax} KL"
            );
            if relax > 0. {
                assert!(
                    expected > tree.expected_acceptance(&laws, 0.)?.0,
                    "inclination did not help"
                );
            }
        }
        Ok(())
    }

    /// Q27D-40: the produced sequence follows the target laws: first token ~ root law,
    /// second token ~ the law of the row reached (fixed seed, 300 000 blocks, 4 sigma).
    #[test]
    fn tree_sampled_acceptance_reproduces_the_joint_law() -> anyhow::Result<()> {
        // root -> {1: 10, 2: 20}; 1 -> {3: 40}; 2 -> {}.
        let tree = DraftTree {
            tokens: vec![100, 10, 20, 40],
            parents: vec![0, 0, 0, 1],
            depths: vec![0, 1, 1, 2],
        };
        let laws = [
            law(&[(10, 0.3), (20, 0.25), (30, 0.45)]),
            law(&[(40, 0.35), (50, 0.65)]),
            law(&[(60, 0.8), (70, 0.2)]),
            law(&[(80, 1.0)]),
        ];
        let blocks = 300_000usize;
        let mut random = 0xC0FF_EE12_3456_789B;
        let mut first = std::collections::HashMap::<u32, usize>::new();
        let mut second = std::collections::HashMap::<(u32, u32), usize>::new();
        for _ in 0..blocks {
            let (path, accepted, token) = tree.accept_sampled(
                |row, extra| {
                    anyhow::ensure!(extra.len() == tree.depths[row], "path history length");
                    Ok(laws[row].clone())
                },
                &mut random,
            )?;
            let sequence = path[1..]
                .iter()
                .map(|&row| tree.tokens[row])
                .chain(std::iter::once(token))
                .collect::<Vec<_>>();
            assert_eq!(sequence.len(), accepted + 1);
            *first.entry(sequence[0]).or_default() += 1;
            if sequence.len() > 1 {
                *second.entry((sequence[0], sequence[1])).or_default() += 1;
            }
        }
        let near = |count: usize, total: usize, p: f64| {
            let observed = count as f64 / total as f64;
            (observed - p).abs() <= 4. * (p * (1. - p) / total as f64).sqrt().max(1e-9)
        };
        for (token, p) in [(10, 0.3), (20, 0.25), (30, 0.45)] {
            assert!(
                near(first.get(&token).copied().unwrap_or(0), blocks, p),
                "first {token}"
            );
        }
        for (parent, child, p) in [(10, 40, 0.35), (10, 50, 0.65), (20, 60, 0.8), (20, 70, 0.2)] {
            let given = first[&parent];
            assert!(
                near(second.get(&(parent, child)).copied().unwrap_or(0), given, p),
                "{parent} -> {child}"
            );
        }
        Ok(())
    }

    #[test]
    fn tree_greedy_acceptance_follows_the_target_path() {
        // root 0 -> {1: a, 2: b}; 1 -> {3: c}; 2 -> {4: d}; 3 -> {5: e}
        let tree = DraftTree {
            tokens: vec![100, 10, 20, 30, 40, 50],
            parents: vec![0, 0, 0, 1, 2, 3],
            depths: vec![0, 1, 1, 2, 2, 3],
        };
        assert!(tree.is_valid());
        // Target wants b after the root, then d, then 77.
        let targets = [20, 30, 40, 99, 77, 88];
        assert_eq!(tree.accept_greedy(&targets), Some((vec![0, 2, 4], 2, 77)));
        // Target wants a, c, e, then 5.
        let targets = [10, 30, 0, 50, 0, 5];
        assert_eq!(tree.accept_greedy(&targets), Some((vec![0, 1, 3, 5], 3, 5)));
        // Root rejected.
        let targets = [11, 0, 0, 0, 0, 0];
        assert_eq!(tree.accept_greedy(&targets), Some((vec![0], 0, 11)));
        assert_eq!(tree.accept_greedy(&targets[..5]), None);
        assert_eq!(tree.paths(), vec![vec![0, 2, 4], vec![0, 1, 3, 5]]);
        assert!(tree.is_ancestor(1, 5) && !tree.is_ancestor(2, 5) && tree.is_ancestor(0, 4));
    }

    #[test]
    fn tree_validity_rejects_bad_structures() {
        let ok = DraftTree {
            tokens: vec![1, 2, 3],
            parents: vec![0, 0, 1],
            depths: vec![0, 1, 2],
        };
        assert!(ok.is_valid());
        let sibling_dup = DraftTree {
            tokens: vec![1, 2, 2],
            parents: vec![0, 0, 0],
            depths: vec![0, 1, 1],
        };
        assert!(!sibling_dup.is_valid());
        let bad_depth = DraftTree {
            tokens: vec![1, 2, 3],
            parents: vec![0, 0, 1],
            depths: vec![0, 1, 1],
        };
        assert!(!bad_depth.is_valid());
        let late_parent = DraftTree {
            tokens: vec![1, 2, 3],
            parents: vec![0, 2, 0],
            depths: vec![0, 2, 1],
        };
        assert!(!late_parent.is_valid());
    }

    fn sequence(seed: u64, length: usize, alphabet: u64) -> Vec<u32> {
        let mut state = seed | 1;
        (0..length)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Repeated spans make long matches common.
                if i % 50 < 20 {
                    (i % 13) as u32
                } else {
                    (state % alphabet) as u32
                }
            })
            .collect()
    }

    /// L4+L3: the incremental index equals the scan reference at every
    /// position and width, on random and repetitive sequences.
    #[test]
    fn ngram_index_matches_the_scan_reference() {
        let mut hits = 0;
        let mut checked = 0;
        for (seed, alphabet) in [(1u64, 3u64), (2, 7), (3, 50), (4, 1000)] {
            let ids = sequence(seed, 700, alphabet);
            let mut index = NgramIndex::new(&ids[..5]);
            for &id in &ids[5..] {
                index.push(id);
                for width in 0..=3 {
                    let expected = ngram_lookup_scan(index.sequence(), width).map(<[u32]>::to_vec);
                    let got = index.lookup(width).map(<[u32]>::to_vec);
                    assert_eq!(
                        got,
                        expected,
                        "seed {seed} length {}",
                        index.sequence().len()
                    );
                    hits += usize::from(got.is_some());
                    checked += 1;
                }
            }
        }
        assert!(hits > checked / 3 && hits < checked, "{hits}/{checked}");
    }

    #[test]
    fn ngram_scan_policy_boundaries() {
        // The longest suffix wins over a more recent shorter one.
        let ids = [1, 2, 3, 9, 7, 3, 8, 1, 2, 3];
        assert_eq!(ngram_lookup_scan(&ids, 3), Some(&[9, 7, 3][..]));
        // The proposal stops at the end of the sequence.
        assert_eq!(ngram_lookup_scan(&[5, 6, 5, 6], 3), Some(&[5, 6][..]));
        // A single repeated ID is not a two-ID match.
        assert_eq!(ngram_lookup_scan(&[4, 1, 2, 4], 3), None);
        assert_eq!(ngram_lookup_scan(&[4, 4], 3), None);
        assert_eq!(ngram_lookup_scan(&[4, 4, 4], 3), Some(&[4][..]));
        assert_eq!(ngram_lookup_scan(&[1, 2, 1, 2], 0), None);
        let mut index = NgramIndex::new(&[]);
        assert_eq!(index.lookup(3), None);
        for id in [4, 4, 4] {
            index.push(id);
        }
        assert_eq!(index.lookup(3), Some(&[4][..]));
        // The whole context is searched, past the old 1024-ID window.
        let mut far = vec![10, 11, 12];
        far.extend((0..3000).map(|i| 100 + i));
        far.extend([10, 11]);
        assert_eq!(ngram_lookup_scan(&far, 2), Some(&[12, 100][..]));
        let mut index = NgramIndex::new(&far);
        assert_eq!(index.lookup(2), Some(&[12, 100][..]));
    }

    use super::*;

    #[test]
    fn accept_histogram_matches_cycle_list_oracle() {
        // Independent oracle: count straight from the cycle list, never from cells.
        let mut state = 0x9e37_79b9_u32;
        for length in [0usize, 1, 2, 7, 64, 257] {
            let mut histogram = AcceptHistogram::default();
            let mut cycles = Vec::new();
            for _ in 0..length {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                let width = (state >> 8) as usize % (MAX_MTP_WIDTH + 1);
                let accepted = (state >> 16) as usize % (width + 1);
                assert!(histogram.record(width, accepted));
                cycles.push((width, accepted));
            }
            assert_eq!(histogram.cycles(), cycles.len());
            assert_eq!(
                histogram.speculative_cycles(),
                cycles.iter().filter(|c| c.0 > 0).count()
            );
            assert_eq!(
                histogram.proposed(),
                cycles.iter().map(|c| c.0).sum::<usize>()
            );
            assert_eq!(
                histogram.accepted(),
                cycles.iter().map(|c| c.1).sum::<usize>()
            );
            for position in 0..=MAX_MTP_WIDTH + 1 {
                let tested = cycles
                    .iter()
                    .filter(|&&(w, a)| w > position && (0..position).all(|p| p < a))
                    .count();
                let hit = cycles
                    .iter()
                    .filter(|&&(w, a)| w > position && (0..=position).all(|p| p < a))
                    .count();
                assert_eq!(histogram.position(position), (hit, tested));
            }
            let mut halves = AcceptHistogram::default();
            let mut second = AcceptHistogram::default();
            for (index, &(width, accepted)) in cycles.iter().enumerate() {
                let half = if index % 2 == 0 {
                    &mut halves
                } else {
                    &mut second
                };
                assert!(half.record(width, accepted));
            }
            halves.merge(&second);
            assert_eq!(halves, histogram);
        }
    }

    #[test]
    fn accept_histogram_rejects_impossible_cycles_unchanged() {
        let mut histogram = AcceptHistogram::default();
        assert!(histogram.record(2, 1));
        let before = histogram;
        for (width, accepted) in [(0, 1), (1, 2), (3, 4), (4, 0), (4, 4), (usize::MAX, 0)] {
            assert!(!histogram.record(width, accepted), "{width}/{accepted}");
            assert_eq!(histogram, before);
        }
        // REL MTP2 short: 54 width-2 cycles with 72 accepted plus one final
        // width-0 target step give 54 + 72 + 1 = 127 decode tokens.
        let mut rel = AcceptHistogram::default();
        for accepted in std::iter::repeat_n(2, 18).chain(std::iter::repeat_n(1, 36)) {
            assert!(rel.record(2, accepted));
        }
        assert!(rel.record(0, 0));
        assert_eq!(
            (rel.speculative_cycles(), rel.proposed(), rel.accepted()),
            (54, 108, 72)
        );
        assert_eq!(rel.cycles() + rel.accepted(), 127);
        assert_eq!(rel.position(0), (54, 54));
        assert_eq!(rel.position(1), (18, 54));
        assert_eq!(
            serde_json::to_value(rel).unwrap(),
            serde_json::json!([[1, 0, 0, 0], [0, 0, 0, 0], [0, 36, 18, 0], [0, 0, 0, 0]])
        );
    }

    #[test]
    fn dflash_copy_remembers_only_commits_and_stops_after_total_refusal() {
        let prompt = [1, 2, 3, 4, 5, 6, 7, 8, 91, 92, 93, 1, 2, 3, 4, 5, 6, 7];
        for width in [0, 1, 2, 3, 5, 7, usize::MAX] {
            let copy = DFlashCopy::new(&prompt);
            let expected = if width == 0 {
                None
            } else {
                Some(&[91, 92, 93][..width.min(3)])
            };
            assert_eq!(copy.propose(8, width), expected);
        }
        let mut copy = DFlashCopy::new(&prompt);
        copy.record(8, &[91, 92], true);
        assert_eq!(&copy.history[prompt.len()..copy.len], &[8, 91, 92]);
        assert!(copy.enabled);
        copy.record(u32::MAX, &[], false);
        assert!(copy.enabled, "neural refusal must not disable copies");
        copy.record(0, &[], true);
        assert!(!copy.enabled);
        copy.record(8, &[91, 92, 93], true);
        assert!(!copy.enabled, "a request never re-enables failed copying");
        copy.history[..prompt.len()].copy_from_slice(&prompt);
        copy.len = prompt.len();
        assert_eq!(copy.propose(8, 3), None);
    }

    #[test]
    fn dflash_copy_bounds_prompt_and_committed_history() {
        for length in [0usize, 1, 8, 9, 1023, 1024, 1025, 2048] {
            let prompt = (0..length as u32).collect::<Vec<_>>();
            for accepted in [0, 1, 3, 7, 1023, 1024, 1025] {
                let prefix = vec![u32::MAX; accepted];
                let mut copy = DFlashCopy::new(&prompt);
                assert_eq!(copy.history.len(), LOOKUP_WINDOW);
                assert_eq!(
                    copy.history[..copy.len],
                    prompt[length.saturating_sub(1024)..]
                );
                let expected = prompt
                    .iter()
                    .copied()
                    .chain([248_319])
                    .chain(prefix.iter().copied())
                    .collect::<Vec<_>>();
                copy.record(248_319, &prefix, true);
                assert_eq!(
                    copy.history[..copy.len],
                    expected[expected.len().saturating_sub(1024)..]
                );
                assert!(copy.len <= 1024);
                assert_eq!(copy.enabled, accepted > 0);
            }
        }
    }

    #[test]
    fn dflash_copy_transitions_match_bounded_stream_over_long_sequences() {
        use std::collections::VecDeque;
        for mut seed in [0u64, 1, 7, 8, 31, u32::MAX as u64, u64::MAX, 0xc0ffee] {
            let prompt = (0..1020 + seed as usize % 9)
                .map(|id| id as u32)
                .collect::<Vec<_>>();
            let mut copy = DFlashCopy::new(&prompt);
            let mut stream: VecDeque<_> = prompt.into_iter().collect();
            while stream.len() > LOOKUP_WINDOW {
                stream.pop_front();
            }
            let mut stopped = false;
            for step in 0..512 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let anchor = (seed >> 32) as u32;
                let count = if step % 64 == 0 {
                    1023 + (step / 64) % 3
                } else {
                    (seed as usize >> 1) % 8
                };
                let copied = seed & 1 != 0;
                let accepted = (0..count)
                    .map(|_| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                        (seed >> 32) as u32
                    })
                    .collect::<Vec<_>>();
                copy.record(anchor, &accepted, copied);
                stopped |= copied && accepted.is_empty();
                for token in std::iter::once(anchor).chain(accepted) {
                    stream.push_back(token);
                    if stream.len() > LOOKUP_WINDOW {
                        stream.pop_front();
                    }
                }
                assert_eq!(copy.len, stream.len());
                assert!(copy.history[..copy.len].iter().eq(stream.iter()));
                assert_eq!(copy.enabled, !stopped);
            }
        }
    }

    #[test]
    fn dflash_modes_bound_every_small_budget_and_reject_unknown_modes() {
        for mode in 0..=4 {
            for context in 0..=17 {
                for output in 0..=17 {
                    let expected = if (1..=3).contains(&mode) && context > 0 && output > 0 {
                        Some((context.min(output) - 1).min([0, 5, 2, 7, 0][mode]))
                    } else {
                        None
                    };
                    assert_eq!(dflash_proposals(mode, context, output, 0, 0, 0), expected);
                }
            }
        }
        assert_eq!(dflash_proposals(1, 20, 20, 8, 23, 40), Some(2));
        assert_eq!(
            dflash_proposals(3, usize::MAX, usize::MAX, 0, 0, 0),
            Some(7)
        );
    }

    #[test]
    fn lookup_uses_latest_known_continuation_and_respects_limits() {
        let history = [
            1, 2, 3, 4, 5, 6, 7, 8, 91, 92, 93, 1, 2, 3, 4, 5, 6, 7, 8, 81, 82, 83, 1, 2, 3, 4, 5,
            6, 7,
        ];
        for width in 1..=3 {
            assert_eq!(
                prompt_lookup(&history, 8, width, |tail| {
                    assert_eq!(tail, &[1, 2, 3, 4, 5, 6, 7]);
                    true
                }),
                Some(&[81, 82, 83][..width])
            );
        }
        assert_eq!(prompt_lookup(&history, 8, 3, |_| false), None);
        assert_eq!(prompt_lookup(&history, 9, 3, |_| true), None);
        for width in [0, 4, usize::MAX] {
            assert_eq!(
                prompt_lookup(&history, 8, width, |_| panic!("invalid width")),
                None
            );
        }
        for len in 0..=8 {
            assert_eq!(
                prompt_lookup(&[8; 8][..len], 8, 3, |_| panic!("short history")),
                None
            );
        }
        assert_eq!(prompt_lookup(&[8; 9], 8, 3, |_| true), Some(&[8][..]));
        // Q27D-19: the bounded form serves widths up to seven and keeps the shared limit.
        let long = [
            1, 2, 3, 4, 5, 6, 7, 8, 71, 72, 73, 74, 75, 76, 77, 1, 2, 3, 4, 5, 6, 7,
        ];
        for width in 1..=7 {
            assert_eq!(
                prompt_lookup_bounded(&long, 8, width, 7, |_| true),
                Some(&[71, 72, 73, 74, 75, 76, 77][..width])
            );
        }
        assert_eq!(prompt_lookup_bounded(&long, 8, 8, 7, |_| true), None);
        assert_eq!(prompt_lookup_bounded(&long, 8, 4, 3, |_| true), None);
        assert_eq!(
            prompt_lookup_bounded(&long, 8, 4, usize::MAX, |_| true),
            Some(&[71, 72, 73, 74][..])
        );
        // A matching anchor at the end has no known continuation.
        assert_eq!(
            prompt_lookup(&[1, 2, 3, 4, 5, 6, 7, 8], 8, 1, |_| true),
            None
        );
        let mut border = history[..11].to_vec();
        border.extend(vec![0; LOOKUP_WINDOW - 18]);
        border.extend(&history[22..]);
        assert_eq!(border.len(), LOOKUP_WINDOW);
        assert_eq!(
            prompt_lookup(&border, 8, 3, |_| true),
            Some(&[91, 92, 93][..])
        );
        border.insert(11, 0);
        assert_eq!(prompt_lookup(&border, 8, 3, |_| true), None);
        let mut far = history[..11].to_vec();
        far.extend([0; LOOKUP_WINDOW]);
        far.extend(&history[22..]);
        assert_eq!(prompt_lookup(&far, 8, 3, |_| true), None);
        far.splice(11..11 + LOOKUP_WINDOW, []);
        assert_eq!(prompt_lookup(&far, 8, 3, |_| true), Some(&[91, 92, 93][..]));
    }

    #[test]
    fn mtp_depths_budgets_and_invalid_depths() {
        for depth in 0..=4 {
            for context in [0, 1, 2, 3, 4, 17, usize::MAX] {
                for output in 0..=17 {
                    let expected = (1..=3)
                        .contains(&depth)
                        .then(|| (0..=depth).rev().find(|&n| n < context && n < output))
                        .flatten();
                    assert_eq!(mtp_width(depth, context, output), expected);
                }
            }
        }
    }

    #[test]
    fn tuning_requires_valid_baseline_and_selects_eligible_gain() {
        assert_eq!(best_mtp_depth([None, Some(100), Some(200), None]), None);
        assert_eq!(best_mtp_depth([Some(0), Some(1), None, None]), None);
        assert_eq!(best_mtp_depth([Some(100), Some(103), None, None]), Some(0));
        assert_eq!(
            best_mtp_depth([Some(100), Some(104), Some(104), None]),
            Some(1)
        );
        assert_eq!(
            best_mtp_depth([Some(100), Some(120), Some(140), Some(150)]),
            Some(3)
        );
        assert_eq!(
            best_mtp_depth([Some(u64::MAX), Some(u64::MAX), None, None]),
            Some(0)
        );
    }

    #[test]
    fn tuning_excludes_collapsed_acceptance_divergence_and_invalid_rates() {
        assert_eq!(mtp_tuning_score(0, 42., true, 0, 0), Some(42000));
        assert_eq!(mtp_tuning_score(3, 62., true, 50, 80), Some(62000));
        for rate in [f64::NAN, f64::INFINITY, -1., 0., f64::MAX] {
            assert_eq!(mtp_tuning_score(1, rate, true, 1, 2), None);
        }
        for (depth, parity, accepted, proposed) in [
            (4, true, 1, 2),
            (1, false, 1, 2),
            (2, true, 0, 2),
            (3, true, 2, 1),
            (1, true, 0, 0),
        ] {
            assert_eq!(
                mtp_tuning_score(depth, 100., parity, accepted, proposed),
                None
            );
        }
    }

    #[test]
    fn speculative_work_stays_inside_both_budgets() {
        for context in [0, 1, 2, 5, 6, 16, usize::MAX] {
            for output in 0..=16 {
                let expected = (0..=5).rev().find(|&n| n < context && n < output);
                assert_eq!(bounded_proposals(context, output), expected);
            }
        }
    }

    #[test]
    fn greedy_accepts_prefix_and_finishes_with_target() {
        assert_eq!(
            greedy_accept(&[10, 11, 12], &[10, 11, 99, 13]),
            Some(GreedyAcceptance {
                accepted_draft_tokens: 2,
                target_token: 99,
            })
        );
        assert_eq!(
            greedy_accept(&[10, 11, 12], &[10, 11, 12, 13]),
            Some(GreedyAcceptance {
                accepted_draft_tokens: 3,
                target_token: 13,
            })
        );
        assert_eq!(greedy_accept(&[], &[7]).unwrap().target_token, 7);
        assert_eq!(greedy_accept(&[1], &[1]), None);
    }

    #[test]
    fn adapts_only_after_sustained_low_acceptance() {
        assert_eq!(adaptive_proposals(16, 16, 7, 0, 35), Some(5));
        assert_eq!(adaptive_proposals(16, 16, 8, 23, 40), Some(2));
        assert_eq!(adaptive_proposals(16, 16, 8, 24, 40), Some(5));
        assert_eq!(adaptive_proposals(16, 2, 8, 23, 40), Some(1));
        assert_eq!(adaptive_proposals(0, 16, 8, 23, 40), None);
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    fn dflash_copy_transition<const LEN: usize, const ACCEPTED: usize>()
    -> (bool, bool, bool, usize, usize) {
        let prompt: [u32; 20] = kani::any();
        let prefix: [u32; 7] = kani::any();
        let len = LEN;
        let accepted = ACCEPTED;
        let anchor: u32 = kani::any();
        let copied: bool = kani::any();
        let enabled: bool = kani::any();
        // Every valid index fits in 0..28; include invalid indices through 31.
        let index = usize::from(kani::any::<u8>() & 31);
        let mut copy = DFlashCopy::new(&prompt[..len]);
        copy.enabled = enabled;
        copy.record(anchor, &prefix[..accepted], copied);
        assert_eq!(copy.len, len + 1 + accepted);
        for i in 0..LEN {
            assert_eq!(copy.history[i], prompt[i]);
        }
        assert_eq!(copy.history[LEN], anchor);
        for i in 0..7 {
            if i < accepted {
                assert_eq!(copy.history[LEN + 1 + i], prefix[i]);
            }
        }
        assert_eq!(copy.enabled, enabled && (!copied || accepted > 0));
        (enabled, copied, copy.enabled, index, copy.len)
    }

    // Emit only the goals that apply to each constant case. A runtime if on a
    // const generic leaves an unreachable cover in Kani's reported inventory.
    macro_rules! copy_prompt_goal {
        (0, $index:expr, 7, $total:expr) => {
            kani::cover!($total == 8);
        };
        (0, $index:expr, $accepted:tt, $total:expr) => {};
        ($len:tt, $index:expr, $accepted:tt, $total:expr) => {
            kani::cover!($index < $len);
        };
    }
    macro_rules! copy_accepted_goals {
        (0, $enabled:expr, $copied:expr, $live:expr, $index:expr, $len:tt, $total:expr) => {
            kani::cover!($enabled && $copied && !$live);
            kani::cover!(!$copied && $live);
        };
        (3, $enabled:expr, $copied:expr, $live:expr, $index:expr, $len:tt, $total:expr) => {
            kani::cover!($copied && $live);
            kani::cover!($index > $len && $index < $total);
        };
        (7, $enabled:expr, $copied:expr, $live:expr, $index:expr, $len:tt, $total:expr) => {
            kani::cover!(!$enabled && !$live);
            kani::cover!($index > $len && $index < $total);
        };
        ($accepted:tt, $enabled:expr, $copied:expr, $live:expr, $index:expr, $len:tt, $total:expr) => {
            kani::cover!($index > $len && $index < $total);
        };
    }
    macro_rules! copy_case_proof {
        ($name:ident, $len:tt, $accepted:tt) => {
            #[kani::proof]
            #[kani::unwind(33)]
            #[kani::solver(kissat)]
            fn $name() {
                let (enabled, copied, live, index, total) =
                    dflash_copy_transition::<$len, $accepted>();
                let _ = (enabled, copied, live);
                copy_prompt_goal!($len, index, $accepted, total);
                copy_accepted_goals!($accepted, enabled, copied, live, index, $len, total);
                kani::cover!(index == $len);
                kani::cover!(index == 31 && index >= total);
            }
        };
    }
    // The complete 21 x 8 domain is split into small proofs, within the same CI
    // shard. All u32 data and flags stay symbolic; the real 1024-entry buffer is
    // retained. No symbolic accepted count is dropped by this partition.
    macro_rules! copy_history_proofs {
        ($($name:ident: $len:tt),* $(,)?) => {$ (
            mod $name {
                use super::*;
                copy_case_proof!(accepted_0, $len, 0);
                copy_case_proof!(accepted_1, $len, 1);
                copy_case_proof!(accepted_2, $len, 2);
                copy_case_proof!(accepted_3, $len, 3);
                copy_case_proof!(accepted_4, $len, 4);
                copy_case_proof!(accepted_5, $len, 5);
                copy_case_proof!(accepted_6, $len, 6);
                copy_case_proof!(accepted_7, $len, 7);
            }
        )*};
    }

    copy_history_proofs! {
        dflash_copy_len_0: 0, dflash_copy_len_1: 1, dflash_copy_len_2: 2,
        dflash_copy_len_3: 3, dflash_copy_len_4: 4, dflash_copy_len_5: 5,
        dflash_copy_len_6: 6, dflash_copy_len_7: 7, dflash_copy_len_8: 8,
        dflash_copy_len_9: 9, dflash_copy_len_10: 10, dflash_copy_len_11: 11,
        dflash_copy_len_12: 12, dflash_copy_len_13: 13, dflash_copy_len_14: 14,
        dflash_copy_len_15: 15, dflash_copy_len_16: 16, dflash_copy_len_17: 17,
        dflash_copy_len_18: 18, dflash_copy_len_19: 19, dflash_copy_len_20: 20,
    }

    #[kani::proof]
    #[kani::unwind(33)]
    #[kani::solver(kissat)]
    fn dflash_disabled_copy_never_proposes() {
        let prompt: [u32; 20] = kani::any();
        let len = usize::from(kani::any::<u8>()) % 21;
        let anchor: u32 = kani::any();
        let width: usize = kani::any();
        let mut copy = DFlashCopy::new(&prompt[..len]);
        // The transition harness establishes this state after a total refusal.
        copy.enabled = false;
        assert!(copy.propose(anchor, width).is_none());
        kani::cover!(len == 0 && width == 0);
        kani::cover!(len == 20 && width == usize::MAX);
    }

    fn lookup_transition<const LEN: usize>() -> (usize, bool, Option<usize>) {
        let data: [u32; 20] = kani::any();
        let len = LEN;
        let width: usize = kani::any();
        let anchor: u32 = kani::any();
        let allowed: bool = kani::any();
        let history = &data[..len];
        let actual = prompt_lookup(history, anchor, width, |tail| {
            assert_eq!(tail.len(), 7);
            assert_eq!(tail, &history[len - 7..]);
            allowed
        });
        let mut expected = None;
        // Independent forward scan, with explicit scalar comparisons.
        if allowed && len >= 9 && (1..=3).contains(&width) {
            for start in 0..len - 8 {
                let mut matched = history[start + 7] == anchor;
                for j in 0..7 {
                    matched &= history[start + j] == history[len - 7 + j];
                }
                if matched {
                    let end = start + 8;
                    expected = Some(&history[end..end + width.min(len - end)]);
                }
            }
        }
        assert_eq!(actual, expected);
        if let Some(proposed) = actual {
            assert!(!proposed.is_empty() && proposed.len() <= width && proposed.len() <= 3);
        }
        (width, allowed, actual.map(<[u32]>::len))
    }

    macro_rules! lookup_goals {
        (short, $width:expr, $allowed:expr, $size:expr) => {
            kani::cover!($size.is_none());
        };
        (long, $width:expr, $allowed:expr, $size:expr) => {
            kani::cover!($width == 3 && $size == Some(1));
            kani::cover!($width == 3 && $size.is_none());
            kani::cover!(!$allowed && $width == 3 && $size.is_none());
        };
        (full, $width:expr, $allowed:expr, $size:expr) => {
            lookup_goals!(long, $width, $allowed, $size);
            kani::cover!($size == Some(3));
        };
    }
    // Complete partition of the former symbolic length modulo 21. Width,
    // history values, anchor and closure decision retain their full domains.
    macro_rules! lookup_proofs {
        ($($name:ident: $len:literal, $kind:ident);* $(;)?) => {$(
            #[kani::proof]
            #[kani::unwind(33)]
            fn $name() {
                let (width, allowed, size) = lookup_transition::<$len>();
                let _ = (width, allowed);
                lookup_goals!($kind, width, allowed, size);
            }
        )*};
    }
    lookup_proofs! {
        lookup_len_0: 0, short;
        lookup_len_1: 1, short;
        lookup_len_2: 2, short;
        lookup_len_3: 3, short;
        lookup_len_4: 4, short;
        lookup_len_5: 5, short;
        lookup_len_6: 6, short;
        lookup_len_7: 7, short;
        lookup_len_8: 8, short;
        lookup_len_9: 9, long;
        lookup_len_10: 10, long;
        lookup_len_11: 11, long;
        lookup_len_12: 12, long;
        lookup_len_13: 13, long;
        lookup_len_14: 14, long;
        lookup_len_15: 15, long;
        lookup_len_16: 16, long;
        lookup_len_17: 17, long;
        lookup_len_18: 18, long;
        lookup_len_19: 19, long;
        lookup_len_20: 20, full;
    }

    #[kani::proof]
    fn tuning_requires_parity_and_nonzero_acceptance() {
        let depth: usize = kani::any();
        let parity: bool = kani::any();
        let accepted: usize = kani::any();
        let proposed: usize = kani::any();
        let score = mtp_tuning_score(depth, 64., parity, accepted, proposed);
        assert_eq!(
            score.is_some(),
            depth <= 3 && parity && (depth == 0 || (accepted > 0 && proposed >= accepted))
        );
        if let Some(score) = score {
            assert_eq!(score, 64000);
        }
        kani::cover!(depth == 3 && accepted == 0 && score.is_none());
        kani::cover!(depth == 3 && score.is_some());
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn mtp_all_depths_commit_target_prefix_within_budgets() {
        let depth: usize = kani::any();
        let context: usize = kani::any();
        let output: usize = kani::any();
        let draft: [u32; 3] = kani::any();
        let target: [u32; 4] = kani::any();
        let width = mtp_width(depth, context, output);
        assert_eq!(
            width.is_some(),
            (1..=3).contains(&depth) && context > 0 && output > 0
        );
        if let Some(width) = width {
            assert!(width <= depth && width <= 3);
            assert!(width < context && width < output);
            let accepted = greedy_accept(&draft[..width], &target[..=width]).unwrap();
            let retained = 1 + accepted.accepted_draft_tokens;
            assert!(retained <= context && retained <= output && retained <= width + 1);
            for i in 0..accepted.accepted_draft_tokens {
                assert_eq!(draft[i], target[i]);
            }
            assert_eq!(
                accepted.target_token,
                target[accepted.accepted_draft_tokens]
            );
            kani::cover!(depth == 3 && accepted.accepted_draft_tokens == 3);
            kani::cover!(depth == 2 && accepted.accepted_draft_tokens == 1);
            kani::cover!(depth == 3 && accepted.accepted_draft_tokens == 0);
            kani::cover!(width == 0);
        }
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn tuning_never_selects_missing_or_slower_candidate() {
        let scores: [Option<u64>; 4] = kani::any();
        let result = best_mtp_depth(scores);
        assert_eq!(result.is_some(), scores[0].is_some_and(|s| s > 0));
        if let Some(depth) = result {
            assert!(depth <= 3);
            let baseline = scores[0].unwrap();
            let winner = scores[depth].unwrap();
            assert!(winner >= baseline);
            if depth > 0 {
                assert!(u128::from(winner) * 100 > u128::from(baseline) * 103);
            }
            for (i, score) in scores.into_iter().enumerate() {
                if let Some(score) = score
                    && u128::from(score) * 100 > u128::from(baseline) * 103
                {
                    assert!(winner >= score);
                    if depth > 0 && score == winner {
                        assert!(depth <= i);
                    }
                }
            }
            kani::cover!(depth == 0);
            kani::cover!(depth == 3);
        }
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn mtp_depth_one_commits_only_target_tokens_inside_budgets() {
        let proposal: u32 = kani::any();
        let target: [u32; 2] = kani::any();
        let context: usize = kani::any();
        let output: usize = kani::any();
        let width = bounded_proposals(context, output).map(|n| n.min(1));
        if width == Some(1) {
            let accepted = greedy_accept(&[proposal], &target).unwrap();
            let retained = 1 + accepted.accepted_draft_tokens;
            assert!(retained <= context && retained <= output);
            let first = if accepted.accepted_draft_tokens == 1 {
                proposal
            } else {
                accepted.target_token
            };
            assert_eq!(first, target[0]);
            if accepted.accepted_draft_tokens == 1 {
                assert_eq!(accepted.target_token, target[1]);
            }
            kani::cover!(accepted.accepted_draft_tokens == 0);
            kani::cover!(accepted.accepted_draft_tokens == 1);
        }
        kani::cover!(width == Some(0) && output == 1);
        kani::cover!(width.is_none() && context == 0);
    }

    #[kani::proof]
    fn speculative_work_reserves_target_within_both_budgets() {
        let context: usize = kani::any();
        let output: usize = kani::any();
        let result = bounded_proposals(context, output);
        assert_eq!(result.is_none(), context == 0 || output == 0);
        if let Some(proposals) = result {
            assert!(proposals <= 5);
            assert!(proposals < context && proposals < output);
            assert!(proposals == 5 || proposals + 1 == context || proposals + 1 == output);
            kani::cover!(proposals == 0 && output == 1);
            kani::cover!(proposals == 3 && context > output);
            kani::cover!(proposals == 5);
        }
    }

    #[kani::proof]
    fn adaptive_work_never_exceeds_the_existing_budget() {
        let context: usize = kani::any();
        let output: usize = kani::any();
        let blocks: usize = kani::any();
        let accepted: usize = kani::any();
        let proposed: usize = kani::any();
        let baseline = bounded_proposals(context, output);
        let actual = adaptive_proposals(context, output, blocks, accepted, proposed);
        assert_eq!(actual.is_some(), baseline.is_some());
        if let (Some(limit), Some(actual)) = (baseline, actual) {
            assert!(actual <= limit && actual < context && actual < output);
            if blocks >= 8 && accepted.saturating_mul(5) < proposed.saturating_mul(3) {
                assert!(actual <= 2);
                kani::cover!(actual == 2);
            } else {
                assert_eq!(actual, limit);
                kani::cover!(actual == 5);
            }
        }
    }

    #[kani::proof]
    #[kani::unwind(9)]
    fn greedy_acceptance_is_the_maximal_matching_prefix() {
        let proposals: [u32; 7] = kani::any();
        let target: [u32; 8] = kani::any();
        let length: u8 = kani::any();
        kani::assume(length <= 7);
        let length = usize::from(length);
        let result = greedy_accept(&proposals[..length], &target[..=length]).unwrap();
        assert!(result.accepted_draft_tokens <= length);
        for index in 0..result.accepted_draft_tokens {
            assert_eq!(proposals[index], target[index]);
        }
        if result.accepted_draft_tokens < length {
            assert_ne!(
                proposals[result.accepted_draft_tokens],
                target[result.accepted_draft_tokens]
            );
        }
        assert_eq!(result.target_token, target[result.accepted_draft_tokens]);
        kani::cover!(result.accepted_draft_tokens == 0);
        kani::cover!(result.accepted_draft_tokens == length);
    }
}
