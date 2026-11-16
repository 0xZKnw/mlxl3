//! CPU token sampling shared by ordinary decoding and speculative verification.
//!
//! Ordinary decoding draws from [`Distribution::sample`]. Speculative decoding
//! with a deterministic draft (MTP argmax, prompt lookup) verifies each
//! proposal `x` against the target distribution `p` of its row: accept with
//! probability `p(x)`, otherwise draw from `p` without `x`, renormalized. Every
//! delivered token then has exactly the law of an ordinary draw from the same
//! target logits (speculative rejection sampling with a point-mass draft).

use crate::speculative::GreedyAcceptance;
use anyhow::{Context, Result, ensure};
use std::collections::{HashSet, VecDeque};

/// Most recent generated tokens penalized by `repetition_penalty`.
pub const REPETITION_WINDOW: usize = 20;

/// Sampling policy of one generation request.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampler {
    pub temperature: f32,
    /// 0 keeps the whole vocabulary.
    pub top_k: usize,
    /// Nucleus mass in (0, 1]; 1 disables it.
    pub top_p: f32,
    pub repetition_penalty: f32,
    /// Subtracted once from every token already generated; 0 disables it.
    pub presence_penalty: f32,
}

impl Sampler {
    pub fn new(
        temperature: f32,
        top_k: usize,
        top_p: f32,
        repetition_penalty: f32,
        presence_penalty: f32,
    ) -> Result<Self> {
        ensure!(
            temperature.is_finite() && temperature >= 0.,
            "temperature must be non-negative"
        );
        ensure!(
            top_p.is_finite() && top_p > 0. && top_p <= 1.,
            "top_p must be greater than 0 and at most 1"
        );
        ensure!(
            repetition_penalty.is_finite() && repetition_penalty > 0.,
            "repetition_penalty must be positive"
        );
        ensure!(
            presence_penalty.is_finite() && (-2.0..=2.0).contains(&presence_penalty),
            "presence_penalty must be between -2 and 2"
        );
        Ok(Self {
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            presence_penalty,
        })
    }

    pub const fn greedy() -> Self {
        Self {
            temperature: 0.,
            top_k: 1,
            top_p: 1.,
            repetition_penalty: 1.,
            presence_penalty: 0.,
        }
    }

    /// The choice is the argmax of the raw target logits: no penalty applies
    /// and no randomness is consumed. Speculative paths keep their exact
    /// greedy verification in this case.
    pub fn is_greedy(&self) -> bool {
        self.deterministic() && self.repetition_penalty == 1. && self.presence_penalty == 0.
    }

    fn deterministic(&self) -> bool {
        self.temperature == 0. || self.top_k == 1
    }

    /// Target distribution of one row. `values` are the row's log
    /// probabilities; the penalty history is `history` followed by `extra`
    /// (speculative tokens accepted earlier in the same verification).
    pub fn distribution(
        &self,
        mut values: Vec<f32>,
        history: &History,
        extra: &[u32],
    ) -> Result<Distribution> {
        if self.repetition_penalty != 1. {
            for &token in history
                .recent
                .iter()
                .chain(extra)
                .rev()
                .take(REPETITION_WINDOW)
            {
                if let Some(value) = values.get_mut(token as usize) {
                    *value = if *value < 0. {
                        *value * self.repetition_penalty
                    } else {
                        *value / self.repetition_penalty
                    };
                }
            }
        }
        if self.presence_penalty != 0. {
            let mut penalize = |token: u32| {
                if let Some(value) = values.get_mut(token as usize) {
                    *value -= self.presence_penalty;
                }
            };
            history.seen.iter().copied().for_each(&mut penalize);
            for (index, &token) in extra.iter().enumerate() {
                if !history.seen.contains(&token) && !extra[..index].contains(&token) {
                    penalize(token);
                }
            }
        }
        if self.deterministic() {
            return values
                .iter()
                .enumerate()
                .filter(|(_, value)| value.is_finite())
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(index, _)| Distribution::Point(index as u32))
                .context("model returned no finite logits");
        }
        let keep = if self.top_k == 0 {
            values.len()
        } else {
            self.top_k.min(values.len())
        };
        let mut candidates: Vec<_> = values.into_iter().enumerate().collect();
        if keep < candidates.len() {
            candidates.select_nth_unstable_by(keep, |left, right| right.1.total_cmp(&left.1));
            candidates.truncate(keep);
        }
        candidates.retain(|(_, value)| value.is_finite());
        let maximum = candidates
            .iter()
            .map(|(_, value)| *value)
            .max_by(f32::total_cmp)
            .context("model returned no finite logits")?;
        let mut total = 0f64;
        let mut weights: Vec<_> = candidates
            .iter()
            .map(|(_, value)| {
                let weight = (((*value - maximum) / self.temperature) as f64).exp();
                total += weight;
                weight
            })
            .collect();
        ensure!(
            total.is_finite() && total > 0.,
            "invalid sampling distribution"
        );
        let mut tokens: Vec<_> = candidates.iter().map(|(index, _)| *index as u32).collect();
        if self.top_p < 1. {
            let mut order: Vec<_> = (0..tokens.len()).collect();
            order.sort_by(|&left, &right| {
                weights[right]
                    .total_cmp(&weights[left])
                    .then(tokens[left].cmp(&tokens[right]))
            });
            let threshold = f64::from(self.top_p) * total;
            let mut mass = 0.;
            let mut kept = 0;
            for &index in &order {
                mass += weights[index];
                kept += 1;
                if mass >= threshold {
                    break;
                }
            }
            order.truncate(kept);
            tokens = order.iter().map(|&index| tokens[index]).collect();
            weights = order.iter().map(|&index| weights[index]).collect();
            total = weights.iter().sum();
            ensure!(
                total.is_finite() && total > 0.,
                "invalid sampling distribution"
            );
        }
        Ok(Distribution::Weighted {
            tokens,
            weights,
            total,
        })
    }

    /// Exact speculative verification of deterministic `proposals`.
    /// `row(i)` returns the log probabilities predicting the token after
    /// `proposals[..i]`; rows past the first rejection are never requested.
    pub fn speculative_accept(
        &self,
        proposals: &[u32],
        history: &History,
        mut row: impl FnMut(usize) -> Result<Vec<f32>>,
        random: &mut u64,
    ) -> Result<GreedyAcceptance> {
        for (index, &proposal) in proposals.iter().enumerate() {
            let distribution = self.distribution(row(index)?, history, &proposals[..index])?;
            if let Verdict::Reject(target_token) = distribution.verify(proposal, random)? {
                return Ok(GreedyAcceptance {
                    accepted_draft_tokens: index,
                    target_token,
                });
            }
        }
        let distribution = self.distribution(row(proposals.len())?, history, proposals)?;
        Ok(GreedyAcceptance {
            accepted_draft_tokens: proposals.len(),
            target_token: distribution.sample(random),
        })
    }
}

/// A non-greedy policy with the state a speculative cycle needs.
pub struct Draw<'a> {
    pub sampler: Sampler,
    /// Generated tokens up to and including the cycle anchor.
    pub history: &'a History,
    pub random: &'a mut u64,
}

impl Draw<'_> {
    pub fn sample(&mut self, values: Vec<f32>) -> Result<u32> {
        Ok(self
            .sampler
            .distribution(values, self.history, &[])?
            .sample(self.random))
    }

    pub fn speculative_accept(
        &mut self,
        proposals: &[u32],
        row: impl FnMut(usize) -> Result<Vec<f32>>,
    ) -> Result<GreedyAcceptance> {
        self.sampler
            .speculative_accept(proposals, self.history, row, self.random)
    }
}

/// Host log probabilities of every row of `logits` (last axis = vocabulary).
#[cfg(feature = "mlx")]
pub fn log_prob_rows(logits: &crate::array::Array) -> Result<Vec<Vec<f32>>> {
    let vocab = usize::try_from(*logits.shape().last().context("logits without shape")?)?;
    let values = logits.log_probs()?.to_f32()?;
    ensure!(
        vocab > 0 && !values.is_empty() && values.len() % vocab == 0,
        "invalid logits shape"
    );
    Ok(values.chunks(vocab).map(<[f32]>::to_vec).collect())
}

/// Generated tokens visible to the penalties.
#[derive(Clone, Debug, Default)]
pub struct History {
    recent: VecDeque<u32>,
    seen: HashSet<u32>,
}

impl History {
    pub fn from_tokens(tokens: &[u32]) -> Self {
        let mut history = Self::default();
        tokens.iter().for_each(|&token| history.push(token));
        history
    }

    pub fn push(&mut self, token: u32) {
        if self.recent.len() == REPETITION_WINDOW {
            self.recent.pop_front();
        }
        self.recent.push_back(token);
        self.seen.insert(token);
    }
}

/// Speculative verdict for one proposal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Accept,
    /// Rejected; the target token replaces the proposal.
    Reject(u32),
}

#[derive(Clone, Debug, PartialEq)]
pub enum Distribution {
    /// Deterministic choice (temperature 0 or top-k 1).
    Point(u32),
    /// Unnormalized weights; `total` is their sum.
    Weighted {
        tokens: Vec<u32>,
        weights: Vec<f64>,
        total: f64,
    },
}

fn xorshift(random: &mut u64) -> u64 {
    *random ^= *random << 13;
    *random ^= *random >> 7;
    *random ^= *random << 17;
    *random
}

/// Uniform in [0, 1) from the top 53 bits.
fn unit(raw: u64) -> f64 {
    (raw >> 11) as f64 / (1u64 << 53) as f64
}

impl Distribution {
    /// Ordinary draw. Points consume no randomness.
    pub fn sample(&self, random: &mut u64) -> u32 {
        match self {
            Self::Point(token) => *token,
            Self::Weighted {
                tokens,
                weights,
                total,
            } => {
                let mut target = (xorshift(random) as f64 / u64::MAX as f64) * total;
                for (&token, &weight) in tokens.iter().zip(weights) {
                    if target <= weight {
                        return token;
                    }
                    target -= weight;
                }
                *tokens.last().expect("nonempty candidates")
            }
        }
    }

    /// Normalized probability of `token`.
    pub fn probability(&self, token: u32) -> f64 {
        match self {
            Self::Point(choice) => f64::from(u8::from(*choice == token)),
            Self::Weighted {
                tokens,
                weights,
                total,
            } => tokens
                .iter()
                .zip(weights)
                .filter(|(candidate, _)| **candidate == token)
                .map(|(_, weight)| weight / total)
                .sum(),
        }
    }

    /// Accept `proposal` with probability p(proposal); otherwise draw from
    /// p without the proposal. Points accept only their own token and consume
    /// no randomness, matching greedy verification.
    pub fn verify(&self, proposal: u32, random: &mut u64) -> Result<Verdict> {
        self.verify_with(proposal, || xorshift(random))
    }

    /// `next` yields raw 64-bit draws: the first decides acceptance (top 53
    /// bits as a uniform in [0, 1)), the next one picks the residual token.
    fn verify_with(&self, proposal: u32, mut next: impl FnMut() -> u64) -> Result<Verdict> {
        let Self::Weighted {
            tokens, weights, ..
        } = self
        else {
            let Self::Point(token) = self else {
                unreachable!()
            };
            return Ok(if *token == proposal {
                Verdict::Accept
            } else {
                Verdict::Reject(*token)
            });
        };
        let (mut accepted, mut residual) = (0f64, 0f64);
        for (&token, &weight) in tokens.iter().zip(weights) {
            if token == proposal {
                accepted += weight;
            } else {
                residual += weight;
            }
        }
        if residual == 0. {
            return Ok(Verdict::Accept);
        }
        if accepted > 0. && unit(next()) * (accepted + residual) < accepted {
            return Ok(Verdict::Accept);
        }
        ensure!(residual.is_finite(), "invalid sampling distribution");
        let mut target = (next() as f64 / u64::MAX as f64) * residual;
        let mut last = None;
        for (&token, &weight) in tokens.iter().zip(weights) {
            if token == proposal {
                continue;
            }
            if target <= weight {
                return Ok(Verdict::Reject(token));
            }
            target -= weight;
            last = Some(token);
        }
        Ok(Verdict::Reject(
            last.context("empty residual distribution")?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Same synthetic logits as the golden capture of the previous bridge
    /// sampler (`9eabe8ed` main.rs `sample_values`).
    fn logits(seed: u64, vocab: usize) -> Vec<f32> {
        let mut s = seed | 1;
        (0..vocab)
            .map(|i| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                if i % 37 == 5 {
                    f32::NEG_INFINITY
                } else {
                    -((s >> 40) as f32 / 4194304.0) - 0.01
                }
            })
            .collect()
    }

    #[test]
    fn ordinary_sampling_matches_previous_bridge_golden() -> Result<()> {
        // Tokens and final random state produced by the previous
        // implementation; top-p 1 and presence 0 must leave it unchanged.
        let golden: [(f32, usize, f32, [u32; 24], u64); 7] = [
            (
                0.7,
                0,
                1.0,
                [
                    89, 48, 48, 14, 9, 47, 53, 77, 87, 57, 0, 23, 34, 1, 69, 66, 19, 57, 93, 60,
                    28, 65, 1, 48,
                ],
                9272385106312786398,
            ),
            (
                1.0,
                20,
                1.0,
                [
                    63, 82, 63, 39, 47, 0, 53, 90, 78, 72, 48, 53, 7, 27, 89, 7, 89, 53, 21, 87,
                    92, 65, 31, 28,
                ],
                15968529369295114709,
            ),
            (
                0.6,
                50,
                1.1,
                [
                    77, 37, 6, 1, 26, 30, 17, 74, 1, 46, 86, 70, 11, 76, 53, 62, 66, 58, 21, 0, 63,
                    14, 72, 37,
                ],
                5284440346753171387,
            ),
            (
                0.1,
                50,
                1.05,
                [
                    30, 87, 0, 27, 0, 63, 11, 61, 0, 0, 78, 6, 55, 84, 0, 0, 81, 23, 0, 66, 0, 28,
                    55, 92,
                ],
                1469202091123269552,
            ),
            (
                0.0,
                0,
                1.1,
                [
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 0, 0, 0, 16, 27, 0, 93, 0, 0, 0, 0, 92, 59, 55,
                ],
                11400714819322674197,
            ),
            (
                1.3,
                3,
                1.0,
                [
                    20, 46, 38, 93, 0, 53, 0, 20, 0, 21, 0, 33, 0, 16, 20, 71, 35, 0, 65, 26, 8, 0,
                    22, 0,
                ],
                4701805532569348504,
            ),
            (
                0.9,
                1,
                1.2,
                [
                    0, 0, 0, 0, 75, 0, 0, 0, 0, 0, 89, 0, 24, 9, 20, 0, 33, 7, 65, 0, 59, 54, 0, 14,
                ],
                11400714819322936341,
            ),
        ];
        for (case, (temperature, top_k, penalty, expected, state)) in golden.into_iter().enumerate()
        {
            let sampler = Sampler::new(temperature, top_k, 1., penalty, 0.)?;
            let mut random = (0x9e37_79b9_7f4a_7c15u64 ^ ((case as u64) << 17)) | 1;
            let mut history = History::from_tokens(&[3, 9, 3, 41]);
            let mut actual = Vec::new();
            for step in 0..24u64 {
                let token = sampler
                    .distribution(logits(step * 31 + case as u64 + 7, 97), &history, &[])?
                    .sample(&mut random);
                history.push(token);
                actual.push(token);
            }
            assert_eq!(actual, expected, "case {case}");
            assert_eq!(random, state, "random state, case {case}");
        }
        Ok(())
    }

    /// Independent oracle: softmax in f64 over the penalized log probs, then
    /// top-k by value and top-p by sorted mass.
    fn oracle(values: &[f32], sampler: &Sampler, history: &[u32]) -> Vec<f64> {
        let mut adjusted: Vec<f64> = values.iter().map(|&v| f64::from(v)).collect();
        for &token in history.iter().rev().take(REPETITION_WINDOW) {
            let value = &mut adjusted[token as usize];
            let penalty = f64::from(sampler.repetition_penalty);
            *value = if *value < 0. {
                *value * penalty
            } else {
                *value / penalty
            };
        }
        let unique: HashSet<_> = history.iter().copied().collect();
        for token in unique {
            adjusted[token as usize] -= f64::from(sampler.presence_penalty);
        }
        let mut order: Vec<_> = (0..adjusted.len())
            .filter(|&i| adjusted[i].is_finite())
            .collect();
        order.sort_by(|&a, &b| adjusted[b].total_cmp(&adjusted[a]).then(a.cmp(&b)));
        let mut probabilities = vec![0.; values.len()];
        if sampler.temperature == 0. || sampler.top_k == 1 {
            probabilities[order[0]] = 1.;
            return probabilities;
        }
        if sampler.top_k > 0 {
            order.truncate(sampler.top_k);
        }
        let best = adjusted[order[0]];
        let weights: Vec<f64> = order
            .iter()
            .map(|&i| ((adjusted[i] - best) / f64::from(sampler.temperature)).exp())
            .collect();
        let total: f64 = weights.iter().sum();
        let mut mass = 0.;
        let mut kept = 0.;
        for (&index, &weight) in order.iter().zip(&weights) {
            probabilities[index] = weight;
            kept += weight;
            mass += weight / total;
            if mass >= f64::from(sampler.top_p) - 1e-12 {
                break;
            }
        }
        probabilities.iter_mut().for_each(|p| *p /= kept);
        probabilities
    }

    fn row(vocab: usize, seed: u64) -> Vec<f32> {
        // Distinct finite log-probability-like values (no ties), one -inf.
        let mut values: Vec<f32> = (0..vocab)
            .map(|i| -0.15 - ((i as u64 * 7 + seed * 3) % vocab as u64) as f32 * 0.31)
            .collect();
        values[vocab - 1] = f32::NEG_INFINITY;
        values
    }

    const GRID: usize = 400;

    /// Exhaustive grid over (u1, u2): the law of the delivered token must be
    /// the oracle distribution for every proposal, within the grid step.
    #[test]
    fn verify_delivers_exact_target_law_on_a_uniform_grid() -> Result<()> {
        let vocab = 9;
        let samplers = [
            Sampler::new(0.8, 0, 1., 1., 0.)?,
            Sampler::new(1.0, 5, 1., 1.1, 0.)?,
            Sampler::new(0.6, 0, 0.7, 1., 0.)?,
            Sampler::new(1.3, 4, 0.9, 1.05, 1.5)?,
            Sampler::new(0.7, 0, 0.8, 1., -0.4)?,
        ];
        let history = [2u32, 5, 2];
        let mut covered_accept_and_reject = (false, false);
        for (case, sampler) in samplers.iter().enumerate() {
            let values = row(vocab, case as u64);
            let expected = oracle(&values, sampler, &history);
            let distribution =
                sampler.distribution(values, &History::from_tokens(&history), &[])?;
            for token in 0..vocab as u32 {
                assert!(
                    (distribution.probability(token) - expected[token as usize]).abs() < 1e-6,
                    "case {case} token {token}"
                );
            }
            for proposal in 0..vocab as u32 {
                let mut law = vec![0.; vocab];
                for first in 0..GRID {
                    for second in 0..GRID {
                        let u1 = ((((first as f64 + 0.5) / GRID as f64) * (1u64 << 53) as f64)
                            as u64)
                            << 11;
                        let u2 = (((second as f64 + 0.5) / GRID as f64) * u64::MAX as f64) as u64;
                        let mut calls = 0;
                        let verdict = distribution.verify_with(proposal, || {
                            calls += 1;
                            if calls == 1 { u1 } else { u2 }
                        })?;
                        let token = match verdict {
                            Verdict::Accept => {
                                covered_accept_and_reject.0 = true;
                                proposal
                            }
                            Verdict::Reject(token) => {
                                covered_accept_and_reject.1 = true;
                                assert_ne!(token, proposal);
                                token
                            }
                        };
                        law[token as usize] += 1. / (GRID * GRID) as f64;
                    }
                }
                for token in 0..vocab {
                    assert!(
                        (law[token] - expected[token]).abs() <= 2.5 / GRID as f64,
                        "case {case} proposal {proposal} token {token}: {} vs {}",
                        law[token],
                        expected[token]
                    );
                }
            }
        }
        assert_eq!(covered_accept_and_reject, (true, true));
        Ok(())
    }

    /// Chains of depth 1-3: the delivered prefix (accepted proposals plus the
    /// target token) must follow the product of the per-row target laws,
    /// with each row penalized by the tokens before it.
    #[test]
    fn speculative_chain_matches_sequential_target_sampling() -> Result<()> {
        let vocab = 6;
        let sampler = Sampler::new(0.9, 0, 0.95, 1.2, 0.5)?;
        let base = [1u32, 4];
        let draws = 60_000;
        for depth in 1..=3usize {
            let proposals: Vec<u32> = [3u32, 0, 3][..depth].to_vec();
            // Row i depends on the delivered prefix only through penalties.
            let rows: Vec<Vec<f32>> = (0..=depth).map(|i| row(vocab, 11 + i as u64)).collect();
            let mut counts = std::collections::HashMap::<Vec<u32>, usize>::new();
            let mut random = 0x2545_f491_4f6c_dd1du64;
            for _ in 0..draws {
                let accepted = sampler.speculative_accept(
                    &proposals,
                    &History::from_tokens(&base),
                    |i| Ok(rows[i].clone()),
                    &mut random,
                )?;
                let mut delivered = proposals[..accepted.accepted_draft_tokens].to_vec();
                delivered.push(accepted.target_token);
                *counts.entry(delivered).or_default() += 1;
            }
            // Exact law of the first divergence from the proposals.
            let mut total = 0.;
            let mut prefix_probability = 1.;
            for i in 0..=depth {
                let mut history = base.to_vec();
                history.extend_from_slice(&proposals[..i]);
                let law = oracle(&rows[i], &sampler, &history);
                for token in 0..vocab as u32 {
                    if i < depth && token == proposals[i] {
                        continue;
                    }
                    let mut delivered = proposals[..i].to_vec();
                    delivered.push(token);
                    let expected = prefix_probability * law[token as usize];
                    let observed = *counts.get(&delivered).unwrap_or(&0) as f64 / draws as f64;
                    total += expected;
                    let tolerance = 4. * (expected * (1. - expected) / draws as f64).sqrt() + 1e-4;
                    assert!(
                        (observed - expected).abs() <= tolerance,
                        "depth {depth} {delivered:?}: {observed} vs {expected}"
                    );
                }
                if i < depth {
                    prefix_probability *= law[proposals[i] as usize];
                }
            }
            assert!((total - 1.).abs() < 1e-9);
            assert!(
                counts.keys().any(|k| k.len() == depth + 1),
                "full acceptance covered"
            );
            assert!(
                counts.keys().any(|k| k.len() == 1),
                "first rejection covered"
            );
        }
        Ok(())
    }

    #[test]
    fn greedy_policies_keep_exact_greedy_verification() -> Result<()> {
        let values = row(8, 3);
        let argmax = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        for sampler in [Sampler::greedy(), Sampler::new(0., 50, 0.9, 1., 0.)?] {
            assert!(sampler.is_greedy());
            let distribution = sampler.distribution(values.clone(), &History::default(), &[])?;
            assert_eq!(distribution, Distribution::Point(argmax));
            let mut random = 7;
            assert_eq!(distribution.verify(argmax, &mut random)?, Verdict::Accept);
            assert_eq!(
                distribution.verify(argmax ^ 1, &mut random)?,
                Verdict::Reject(argmax)
            );
            assert_eq!(random, 7, "greedy verification consumes no randomness");
            // Same rule as speculative::greedy_accept.
            let accepted = sampler.speculative_accept(
                &[argmax, argmax ^ 1],
                &History::default(),
                |_| Ok(values.clone()),
                &mut random,
            )?;
            assert_eq!(
                accepted,
                crate::speculative::greedy_accept(&[argmax, argmax ^ 1], &[argmax; 3]).unwrap()
            );
        }
        assert!(!Sampler::new(0., 1, 1., 1.1, 0.)?.is_greedy());
        assert!(!Sampler::new(0., 1, 1., 1., 0.5)?.is_greedy());
        assert!(!Sampler::new(0.7, 20, 1., 1., 0.)?.is_greedy());
        Ok(())
    }

    #[test]
    fn boundaries_and_invalid_parameters() -> Result<()> {
        for invalid in [
            Sampler::new(-0.1, 0, 1., 1., 0.),
            Sampler::new(f32::NAN, 0, 1., 1., 0.),
            Sampler::new(1., 0, 0., 1., 0.),
            Sampler::new(1., 0, 1.01, 1., 0.),
            Sampler::new(1., 0, 1., 0., 0.),
            Sampler::new(1., 0, 1., 1., 2.5),
            Sampler::new(1., 0, 1., 1., f32::INFINITY),
        ] {
            assert!(invalid.is_err());
        }
        let sampler = Sampler::new(1., 0, 1., 1., 0.)?;
        let mut random = 99;
        // A proposal with zero mass is always rejected; a sole candidate is
        // always accepted.
        let values = vec![-0.1, f32::NEG_INFINITY, -2.0];
        let distribution = sampler.distribution(values, &History::default(), &[])?;
        for _ in 0..200 {
            assert_ne!(distribution.verify(1, &mut random)?, Verdict::Accept);
        }
        let tiny = Sampler::new(1., 0, 1e-6, 1., 0.)?;
        let distribution = tiny.distribution(row(7, 2), &History::default(), &[])?;
        let Distribution::Weighted { tokens, .. } = &distribution else {
            panic!("weighted expected")
        };
        assert_eq!(tokens.len(), 1, "minimal nucleus keeps one token");
        for _ in 0..200 {
            assert_eq!(
                distribution.verify(tokens[0], &mut random)?,
                Verdict::Accept
            );
        }
        assert!(
            sampler
                .distribution(vec![f32::NEG_INFINITY; 4], &History::default(), &[])
                .is_err()
        );
        // Presence applies once per distinct token across history and prefix;
        // the repetition window covers the last 20 of both.
        let presence = Sampler::new(1., 0, 1., 1., 1.)?;
        let history = History::from_tokens(&[0, 0, 1]);
        let shifted = presence.distribution(vec![-1.; 4], &history, &[1, 2, 2])?;
        let ratio = shifted.probability(3) / shifted.probability(0);
        assert!((ratio - std::f64::consts::E).abs() < 1e-9);
        assert!((shifted.probability(2) - shifted.probability(0)).abs() < 1e-12);
        let mut long = History::default();
        (0..30).for_each(|t| long.push(t % 3 + 10));
        assert_eq!(long.recent.len(), REPETITION_WINDOW);
        Ok(())
    }
}
