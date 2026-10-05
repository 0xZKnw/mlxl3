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

#[cfg(test)]
mod tests {
    use super::*;

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
