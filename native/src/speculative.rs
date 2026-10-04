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
