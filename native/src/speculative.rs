//! Exact target-side acceptance for speculative decoding.

/// The accepted draft prefix plus the target token that ends the cycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GreedyAcceptance {
    pub accepted_draft_tokens: usize,
    pub target_token: u32,
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
}

#[cfg(kani)]
mod verification {
    use super::*;

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
