//! Exact target-side acceptance for speculative decoding.

/// The accepted draft prefix plus the target token that ends the cycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GreedyAcceptance {
    pub accepted_draft_tokens: usize,
    pub target_token: u32,
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
    if history.len() < 9 || !(1..=3).contains(&width) {
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
        self.enabled
            .then(|| prompt_lookup(&self.history[..self.len], anchor, width.min(3), |_| true))
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
    use super::*;

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

    fn dflash_copy_transition<const LEN: usize>() {
        let prompt: [u32; 20] = kani::any();
        let prefix: [u32; 7] = kani::any();
        let len = LEN;
        let accepted = usize::from(kani::any::<u8>()) % 8;
        let anchor: u32 = kani::any();
        let copied: bool = kani::any();
        let enabled: bool = kani::any();
        // Every valid index fits in 0..28; include invalid indices through 31.
        let index = usize::from(kani::any::<u8>() & 31);
        let mut copy = DFlashCopy::new(&prompt[..len]);
        copy.enabled = enabled;
        copy.record(anchor, &prefix[..accepted], copied);
        assert_eq!(copy.len, len + 1 + accepted);
        if index < len {
            assert_eq!(copy.history[index], prompt[index]);
        } else if index == len {
            assert_eq!(copy.history[index], anchor);
        } else if index < copy.len {
            assert_eq!(copy.history[index], prefix[index - len - 1]);
        }
        assert_eq!(copy.enabled, enabled && (!copied || accepted > 0));
        kani::cover!(enabled && copied && accepted == 0 && !copy.enabled);
        kani::cover!(copied && accepted == 3 && copy.enabled);
        kani::cover!(!copied && accepted == 0 && copy.enabled);
        if LEN == 0 {
            kani::cover!(accepted == 7);
        }
        kani::cover!(!enabled && accepted == 7 && !copy.enabled);
        if LEN > 0 {
            kani::cover!(index < len);
        }
        kani::cover!(index == len);
        kani::cover!(index > len && index < copy.len);
        kani::cover!(index == 31 && index >= copy.len);
    }

    macro_rules! copy_history_proofs {
        ($($name:ident: $len:expr),* $(,)?) => {$ (
            #[kani::proof]
            #[kani::unwind(33)]
            #[kani::solver(kissat)]
            fn $name() { dflash_copy_transition::<$len>(); }
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

    #[kani::proof]
    #[kani::unwind(33)]
    fn lookup_proposes_only_latest_in_bounds_history() {
        let data: [u32; 20] = kani::any();
        let len = kani::any::<usize>() % 21;
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
            kani::cover!(proposed.len() == 3);
            kani::cover!(width == 3 && proposed.len() == 1);
        }
        kani::cover!(len >= 9 && width == 3 && actual.is_none());
        kani::cover!(len < 9 && actual.is_none());
        kani::cover!(!allowed && len >= 9 && width == 3 && actual.is_none());
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
