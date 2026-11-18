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

/// Strict Q3 gate: exactly `1` draws MTP drafts from the head's law.
pub fn sampled_draft_enabled_by(flag: Option<&[u8]>) -> bool {
    flag == Some(b"1")
}

/// `MLXL3_MTP_SAMPLED_DRAFT`, read once.
pub fn sampled_draft_enabled() -> bool {
    static FLAG: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FLAG.get_or_init(|| {
        let flag = std::env::var_os("MLXL3_MTP_SAMPLED_DRAFT");
        sampled_draft_enabled_by(flag.as_deref().map(std::ffi::OsStr::as_encoded_bytes))
    })
}

static SAMPLED_DRAFT_LEVELS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Q3: MTP draft levels drawn from the head's law since process start.
pub fn sampled_draft_levels() -> usize {
    SAMPLED_DRAFT_LEVELS.load(std::sync::atomic::Ordering::Relaxed)
}

/// Q3b: a compact draft row (`values[c]` belongs to vocabulary ID `ids[c]`)
/// written into a full-vocabulary row, `-inf` outside the compact set, so the
/// draft law and its penalties use the real IDs. `ids` must be in range and
/// strictly increasing (the `valid_draft_rows` contract of the MTP loader).
pub fn expand_compact_row(values: &[f32], ids: &[u32], vocab: usize) -> Result<Vec<f32>> {
    ensure!(
        values.len() == ids.len() && ids.len() <= vocab,
        "compact MTP row does not match its token IDs"
    );
    ensure!(
        ids.iter().all(|&id| (id as usize) < vocab) && ids.windows(2).all(|w| w[0] < w[1]),
        "compact MTP token IDs out of range or unordered"
    );
    let mut full = vec![f32::NEG_INFINITY; vocab];
    for (&id, &value) in ids.iter().zip(values) {
        full[id as usize] = value;
    }
    Ok(full)
}

/// Largest top-k served by `Sampler::sparse_distribution`.
pub const SPARSE_MAX_K: usize = 256;

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

    pub(crate) fn deterministic(&self) -> bool {
        self.temperature == 0. || self.top_k == 1
    }

    /// How many of the best penalized values `distribution` keeps, when
    /// penalties can only lower values (repetition penalty at least 1,
    /// presence penalty at least 0) and at most `max` are kept: 1 for a
    /// deterministic choice, `top_k` otherwise. Used by certified heads that
    /// compute only the best rows exactly.
    pub fn certified_k(&self, max: usize) -> Option<usize> {
        if !(self.repetition_penalty >= 1. && self.presence_penalty >= 0.) {
            return None;
        }
        if self.deterministic() {
            return Some(1);
        }
        (1..=max).contains(&self.top_k).then_some(self.top_k)
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
        self.apply_penalties(&mut values, history, extra);
        if self.deterministic() {
            return Self::point(values.into_iter().enumerate());
        }
        self.select(values.into_iter().enumerate().collect())
    }

    /// The deterministic choice of `distribution`: the last best finite
    /// value in iteration order.
    fn point(candidates: impl Iterator<Item = (usize, f32)>) -> Result<Distribution> {
        candidates
            .filter(|(_, value)| value.is_finite())
            .max_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(index, _)| Distribution::Point(index as u32))
            .context("model returned no finite logits")
    }

    /// The selection of `distribution` over penalized `(token, value)`
    /// candidates, in the order given.
    fn select(&self, candidates: Vec<(usize, f32)>) -> Result<Distribution> {
        if self.deterministic() {
            return Self::point(candidates.into_iter());
        }
        let mut candidates = candidates;
        let keep = if self.top_k == 0 {
            candidates.len()
        } else {
            self.top_k.min(candidates.len())
        };
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

    /// The repetition and presence penalties of `distribution`, in place.
    pub fn apply_penalties(&self, values: &mut [f32], history: &History, extra: &[u32]) {
        self.apply_penalties_by(history, extra, |token, update| {
            if let Some(value) = values.get_mut(token as usize) {
                *value = update(*value);
            }
        });
    }

    /// The penalty sequence of `apply_penalties`: `apply(token, update)` is
    /// called once per penalty, in order, and replaces the token's value `v`
    /// with `update(v)`.
    fn apply_penalties_by(
        &self,
        history: &History,
        extra: &[u32],
        mut apply: impl FnMut(u32, &dyn Fn(f32) -> f32),
    ) {
        if self.repetition_penalty != 1. {
            let penalty = self.repetition_penalty;
            let repeat = move |value: f32| {
                if value < 0. {
                    value * penalty
                } else {
                    value / penalty
                }
            };
            for &token in history
                .recent
                .iter()
                .chain(extra)
                .rev()
                .take(REPETITION_WINDOW)
            {
                apply(token, &repeat);
            }
        }
        if self.presence_penalty != 0. {
            let penalty = self.presence_penalty;
            let present = move |value: f32| value - penalty;
            for &token in &history.seen {
                apply(token, &present);
            }
            for (index, &token) in extra.iter().enumerate() {
                if !history.seen.contains(&token) && !extra[..index].contains(&token) {
                    apply(token, &present);
                }
            }
        }
    }

    /// Distinct tokens `apply_penalties` may change for this history.
    pub fn penalty_span(&self, history: &History, extra: &[u32]) -> usize {
        let mut tokens = HashSet::new();
        self.apply_penalties_by(history, extra, |token, _| {
            tokens.insert(token);
        });
        tokens.len()
    }

    /// Q1-TOPK-READBACK: `distribution(full, history, extra).canonical()`
    /// from the candidates of a row only. `candidates` must hold every
    /// `(token, value)` of the full row whose value is at least
    /// `threshold`, all other values being below it. Returns None (use the
    /// full row) unless penalties only lower values, top-k is bounded, no
    /// candidate is NaN, the k-th penalized candidate is at least
    /// `threshold` and no candidate ties it.
    pub fn sparse_distribution(
        &self,
        candidates: &[(u32, f32)],
        threshold: f32,
        history: &History,
        extra: &[u32],
    ) -> Result<Option<Distribution>> {
        let Some(k) = self.certified_k(SPARSE_MAX_K) else {
            return Ok(None);
        };
        if !threshold.is_finite() || candidates.len() < k || candidates.iter().any(|c| c.1.is_nan())
        {
            return Ok(None);
        }
        let mut sorted = candidates.to_vec();
        sorted.sort_unstable_by_key(|c| c.0);
        if sorted.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Ok(None);
        }
        let mut values: Vec<f32> = sorted.iter().map(|c| c.1).collect();
        self.apply_penalties_by(history, extra, |token, update| {
            if let Ok(index) = sorted.binary_search_by_key(&token, |c| c.0) {
                values[index] = update(values[index]);
            }
        });
        let mut order = values.clone();
        order.sort_unstable_by(|a, b| b.total_cmp(a));
        let boundary = order[k - 1];
        if boundary < threshold || order.get(k) == Some(&boundary) {
            return Ok(None);
        }
        let selected = sorted
            .iter()
            .zip(values)
            .map(|(c, value)| (c.0 as usize, value))
            .collect();
        Ok(Some(self.select(selected)?.canonical()))
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
        Self::accept_with(
            proposals,
            |index, extra| self.distribution(row(index)?, history, extra),
            random,
        )
    }

    /// Q3: speculative sampling of `proposals` drawn from `drafts`
    /// (`proposals[i]` ~ `drafts[i]`); `target(i, extra)` as in
    /// `accept_with`.
    pub fn accept_drafts_with(
        proposals: &[u32],
        drafts: &[Distribution],
        mut target: impl FnMut(usize, &[u32]) -> Result<Distribution>,
        random: &mut u64,
    ) -> Result<GreedyAcceptance> {
        ensure!(
            drafts.len() == proposals.len(),
            "one draft distribution per proposal"
        );
        for (index, (&proposal, draft)) in proposals.iter().zip(drafts).enumerate() {
            let distribution = target(index, &proposals[..index])?;
            if let Verdict::Reject(target_token) =
                distribution.verify_draft(draft, proposal, random)?
            {
                return Ok(GreedyAcceptance {
                    accepted_draft_tokens: index,
                    target_token,
                });
            }
        }
        let distribution = target(proposals.len(), proposals)?;
        Ok(GreedyAcceptance {
            accepted_draft_tokens: proposals.len(),
            target_token: distribution.sample(random),
        })
    }

    /// `speculative_accept` over target distributions: `target(i, extra)`
    /// is the distribution of row `i` with penalty history `extra`.
    pub fn accept_with(
        proposals: &[u32],
        mut target: impl FnMut(usize, &[u32]) -> Result<Distribution>,
        random: &mut u64,
    ) -> Result<GreedyAcceptance> {
        for (index, &proposal) in proposals.iter().enumerate() {
            let distribution = target(index, &proposals[..index])?;
            if let Verdict::Reject(target_token) = distribution.verify(proposal, random)? {
                return Ok(GreedyAcceptance {
                    accepted_draft_tokens: index,
                    target_token,
                });
            }
        }
        let distribution = target(proposals.len(), proposals)?;
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

#[cfg(feature = "mlx")]
impl Draw<'_> {
    /// Draws from the last row of `logits` (Q1-TOPK-READBACK when enabled).
    pub fn sample_last_row(&mut self, logits: &crate::array::Array) -> Result<u32> {
        if crate::topk_readback::mtp_enabled() {
            let mut rows =
                crate::topk_readback::ReadbackRows::new(logits, &self.sampler, self.history, &[])?;
            let last = rows.len().checked_sub(1).context("missing sampled row")?;
            return Ok(rows
                .distribution(last, &self.sampler, self.history, &[])?
                .sample(self.random));
        }
        let rows = log_prob_rows(logits)?;
        self.sample(rows.last().context("missing sampled row")?.to_vec())
    }

    /// Q3: `speculative_accept_logits` for proposals drawn from `drafts`.
    pub fn accept_drafts_logits(
        &mut self,
        proposals: &[u32],
        drafts: &[Distribution],
        logits: &crate::array::Array,
    ) -> Result<GreedyAcceptance> {
        let sampler = self.sampler;
        let history = self.history;
        if crate::topk_readback::mtp_enabled() {
            let mut rows =
                crate::topk_readback::ReadbackRows::new(logits, &sampler, history, proposals)?;
            ensure!(
                rows.len() == proposals.len() + 1,
                "invalid speculative verification output"
            );
            return Sampler::accept_drafts_with(
                proposals,
                drafts,
                |index, extra| rows.distribution(index, &sampler, history, extra),
                self.random,
            );
        }
        let rows = log_prob_rows(logits)?;
        ensure!(
            rows.len() == proposals.len() + 1,
            "invalid speculative verification output"
        );
        Sampler::accept_drafts_with(
            proposals,
            drafts,
            |index, extra| sampler.distribution(rows[index].clone(), history, extra),
            self.random,
        )
    }

    /// Q3: draws a draft token from the draft log probabilities `values`
    /// (draft law = the target policy with `extra` penalties).
    pub fn draw_draft(&mut self, values: Vec<f32>, extra: &[u32]) -> Result<(u32, Distribution)> {
        let draft = self.sampler.distribution(values, self.history, extra)?;
        SAMPLED_DRAFT_LEVELS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok((draft.sample(self.random), draft))
    }

    /// `speculative_accept` over the `proposals.len() + 1` rows of `logits`
    /// (Q1-TOPK-READBACK when enabled).
    pub fn speculative_accept_logits(
        &mut self,
        proposals: &[u32],
        logits: &crate::array::Array,
    ) -> Result<GreedyAcceptance> {
        if crate::topk_readback::mtp_enabled() {
            let sampler = self.sampler;
            let history = self.history;
            let mut rows =
                crate::topk_readback::ReadbackRows::new(logits, &sampler, history, proposals)?;
            ensure!(
                rows.len() == proposals.len() + 1,
                "invalid speculative verification output"
            );
            return Sampler::accept_with(
                proposals,
                |index, extra| rows.distribution(index, &sampler, history, extra),
                self.random,
            );
        }
        let rows = log_prob_rows(logits)?;
        ensure!(
            rows.len() == proposals.len() + 1,
            "invalid speculative verification output"
        );
        self.speculative_accept(proposals, |index| Ok(rows[index].to_vec()))
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

/// Q27D-40: outcome of `Distribution::verify_among`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Choice {
    /// The candidate at this index is accepted.
    Candidate(usize),
    /// Every candidate was rejected; this token was drawn from the rest.
    Token(u32),
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
    /// The same law with tokens in increasing order and `total` summed in that
    /// order. `distribution` lists the kept tokens in the order of an
    /// unstable selection over the whole row; a certified head that only
    /// knows the best rows reaches the same set and weights in another order,
    /// so both sides are compared (and drawn) in this order (L1).
    pub fn canonical(self) -> Self {
        match self {
            Self::Point(_) => self,
            Self::Weighted {
                tokens, weights, ..
            } => {
                let mut pairs: Vec<_> = tokens.into_iter().zip(weights).collect();
                pairs.sort_unstable_by_key(|&(token, _)| token);
                let total = pairs.iter().map(|&(_, weight)| weight).sum();
                let (tokens, weights) = pairs.into_iter().unzip();
                Self::Weighted {
                    tokens,
                    weights,
                    total,
                }
            }
        }
    }

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

    /// `(token, probability)` pairs of the support, in stored order.
    fn support(&self) -> Vec<(u32, f64)> {
        match self {
            Self::Point(token) => vec![(*token, 1.)],
            Self::Weighted {
                tokens,
                weights,
                total,
            } => tokens
                .iter()
                .zip(weights)
                .map(|(&token, &weight)| (token, weight / total))
                .collect(),
        }
    }

    /// Q3-MTP-SAMPLED-DRAFT: standard speculative sampling of a `proposal`
    /// drawn from `draft` (q): accept with probability min(1, p(x)/q(x)),
    /// otherwise draw from max(0, p - q) normalized (from p if that
    /// residual is numerically empty). The delivered token has law p.
    pub fn verify_draft(&self, draft: &Self, proposal: u32, random: &mut u64) -> Result<Verdict> {
        self.verify_draft_with(draft, proposal, || xorshift(random))
    }

    fn verify_draft_with(
        &self,
        draft: &Self,
        proposal: u32,
        mut next: impl FnMut() -> u64,
    ) -> Result<Verdict> {
        let q = draft.probability(proposal);
        ensure!(q > 0., "draft proposal outside the draft distribution");
        let p = self.probability(proposal);
        if p >= q || unit(next()) * q < p {
            return Ok(Verdict::Accept);
        }
        let residual: Vec<(u32, f64)> = self
            .support()
            .into_iter()
            .map(|(token, mass)| (token, (mass - draft.probability(token)).max(0.)))
            .filter(|&(_, mass)| mass > 0.)
            .collect();
        let residual = if residual.is_empty() {
            self.support()
        } else {
            residual
        };
        let total: f64 = residual.iter().map(|&(_, mass)| mass).sum();
        ensure!(
            total.is_finite() && total > 0.,
            "invalid residual distribution"
        );
        let mut target = (next() as f64 / u64::MAX as f64) * total;
        for &(token, mass) in &residual {
            if target <= mass {
                return Ok(Verdict::Reject(token));
            }
            target -= mass;
        }
        Ok(Verdict::Reject(
            residual.last().context("empty residual distribution")?.0,
        ))
    }

    /// Accept `proposal` with probability p(proposal); otherwise draw from
    /// p without the proposal. Points accept only their own token and consume
    /// no randomness, matching greedy verification.
    pub fn verify(&self, proposal: u32, random: &mut u64) -> Result<Verdict> {
        self.verify_with(proposal, || xorshift(random))
    }

    /// Q27D-47: this law inclined toward `candidates` under a divergence budget:
    /// p̂ ∝ p · e^{β·1[t ∈ C]} with β ≥ 0 such that KL(p̂ ‖ p) = `budget` (nats), ratios kept
    /// inside and outside C. Returns the law and its exact divergence; a point, a set of mass
    /// 0 or 1, or a zero budget is returned unchanged.
    pub fn tilted_toward(&self, candidates: &[u32], budget: f64) -> Result<(Self, f64)> {
        ensure!(
            budget.is_finite() && budget >= 0.,
            "invalid divergence budget"
        );
        let Self::Weighted {
            tokens,
            weights,
            total,
        } = self
        else {
            return Ok((self.clone(), 0.));
        };
        ensure!(
            total.is_finite() && *total > 0.,
            "invalid sampling distribution"
        );
        let inside: f64 = tokens
            .iter()
            .zip(weights)
            .filter(|(token, _)| candidates.contains(token))
            .map(|(_, &weight)| weight)
            .sum();
        let m = inside / total;
        if budget == 0. || m <= 0. || m >= 1. {
            return Ok((self.clone(), 0.));
        }
        let divergence = |x: f64| x * (x / m).ln() + (1. - x) * ((1. - x) / (1. - m)).ln();
        // KL grows on [m, 1) toward ln(1/m); keep some mass outside C.
        let (mut low, mut high) = (m, 1. - 1e-9);
        let x = if divergence(high) <= budget {
            high
        } else {
            for _ in 0..100 {
                let middle = 0.5 * (low + high);
                if divergence(middle) < budget {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            low
        };
        let (scale_in, scale_out) = (x / m, (1. - x) / (1. - m));
        let weights: Vec<f64> = tokens
            .iter()
            .zip(weights)
            .map(|(token, &weight)| {
                weight
                    * if candidates.contains(token) {
                        scale_in
                    } else {
                        scale_out
                    }
            })
            .collect();
        let total = weights.iter().sum();
        Ok((
            Self::Weighted {
                tokens: tokens.clone(),
                weights,
                total,
            },
            divergence(x),
        ))
    }

    /// Q27D-47: (KL(self ‖ reference), −Σ self · log reference, −Σ reference · log reference)
    /// in nats, for a law obtained from `reference` by reweighting the same support.
    pub fn divergence_terms(&self, reference: &Self) -> (f64, f64, f64) {
        let support = reference.support();
        let mut terms = (0., 0., 0.);
        for (token, p) in support {
            if p <= 0. {
                continue;
            }
            let q = self.probability(token);
            if q > 0. {
                terms.0 += q * (q / p).ln();
            }
            terms.1 -= q * p.ln();
            terms.2 -= p * p.ln();
        }
        terms
    }

    /// Q27D-40: exact speculative verification of several deterministic, distinct
    /// `candidates` (the children of a tree node), tried in order. Candidate `c` is
    /// accepted with probability mass(c) / remaining mass; a rejected candidate leaves
    /// the law, which is renormalized; if all are rejected, the token is drawn from the
    /// rest. By induction `P(c_i) = p(c_i)` and every other token keeps `p(t)`, so the
    /// produced token follows this distribution exactly. One candidate consumes the
    /// same draws as `verify`.
    pub fn verify_among(&self, candidates: &[u32], random: &mut u64) -> Result<Choice> {
        self.verify_among_with(candidates, || xorshift(random))
    }

    fn verify_among_with(
        &self,
        candidates: &[u32],
        mut next: impl FnMut() -> u64,
    ) -> Result<Choice> {
        ensure!(
            candidates
                .iter()
                .enumerate()
                .all(|(index, candidate)| !candidates[..index].contains(candidate)),
            "speculative candidates must be distinct"
        );
        let Self::Weighted {
            tokens, weights, ..
        } = self
        else {
            let Self::Point(token) = self else {
                unreachable!()
            };
            return Ok(candidates
                .iter()
                .position(|candidate| candidate == token)
                .map_or(Choice::Token(*token), Choice::Candidate));
        };
        let mut rejected: Vec<u32> = Vec::with_capacity(candidates.len());
        let mut residual = 0f64;
        for (index, &candidate) in candidates.iter().enumerate() {
            let (mut accepted, mut rest) = (0f64, 0f64);
            for (&token, &weight) in tokens.iter().zip(weights) {
                if rejected.contains(&token) {
                    continue;
                }
                if token == candidate {
                    accepted += weight;
                } else {
                    rest += weight;
                }
            }
            if rest == 0. {
                return Ok(Choice::Candidate(index));
            }
            if accepted > 0. && unit(next()) * (accepted + rest) < accepted {
                return Ok(Choice::Candidate(index));
            }
            rejected.push(candidate);
            residual = rest;
        }
        if candidates.is_empty() {
            // A leaf draws like `sample`, from the stored total.
            let Self::Weighted { total, .. } = self else {
                unreachable!()
            };
            residual = *total;
        }
        ensure!(
            residual.is_finite() && residual > 0.,
            "invalid sampling distribution"
        );
        let mut target = (next() as f64 / u64::MAX as f64) * residual;
        let mut last = None;
        for (&token, &weight) in tokens.iter().zip(weights) {
            if rejected.contains(&token) {
                continue;
            }
            if target <= weight {
                return Ok(Choice::Token(token));
            }
            target -= weight;
            last = Some(token);
        }
        Ok(Choice::Token(last.context("empty residual distribution")?))
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

    fn weighted(pairs: &[(u32, f64)]) -> Distribution {
        Distribution::Weighted {
            tokens: pairs.iter().map(|&(token, _)| token).collect(),
            weights: pairs.iter().map(|&(_, weight)| weight).collect(),
            total: pairs.iter().map(|&(_, weight)| weight).sum(),
        }
    }

    /// Q27D-40: the token produced by `verify_among` follows the target law, for several
    /// laws, candidate sets and orders (fixed seeds, 400 000 draws, 4-sigma bounds).
    #[test]
    fn verify_among_reproduces_the_target_law() -> Result<()> {
        let laws = [
            vec![(3, 0.4), (7, 0.25), (1, 0.2), (9, 0.1), (4, 0.05)],
            vec![(10, 3.0), (11, 1.0), (12, 0.0), (13, 6.0)],
            vec![(5, 1.0), (6, 1.0)],
        ];
        let candidate_sets: [&[u32]; 7] = [
            &[],
            &[7],
            &[9, 3],
            &[1, 3, 4],
            &[12, 11],
            &[13, 10, 11],
            &[6, 5],
        ];
        let draws = 400_000usize;
        let mut checked = 0;
        for (law_index, law) in laws.iter().enumerate() {
            let distribution = weighted(law);
            let total: f64 = law.iter().map(|&(_, w)| w).sum();
            for candidates in candidate_sets {
                if !candidates
                    .iter()
                    .all(|c| law.iter().any(|&(token, _)| token == *c))
                {
                    continue;
                }
                let mut random = 0x5DEE_CE66_D1CE_4E5B
                    ^ (law_index as u64 + 1)
                    ^ ((candidates.len() as u64) << 40);
                let mut counts = std::collections::HashMap::<u32, usize>::new();
                for _ in 0..draws {
                    let token = match distribution.verify_among(candidates, &mut random)? {
                        Choice::Candidate(index) => candidates[index],
                        Choice::Token(token) => {
                            ensure!(
                                !candidates.contains(&token),
                                "a rejected candidate was drawn"
                            );
                            token
                        }
                    };
                    *counts.entry(token).or_default() += 1;
                }
                for &(token, weight) in law {
                    let p = weight / total;
                    let observed = counts.get(&token).copied().unwrap_or(0) as f64 / draws as f64;
                    let sigma = (p * (1. - p) / draws as f64).sqrt().max(1e-9);
                    ensure!(
                        (observed - p).abs() <= 4. * sigma,
                        "law {law_index} candidates {candidates:?} token {token}: {observed} vs {p}"
                    );
                }
                checked += 1;
            }
        }
        ensure!(
            checked == 9,
            "unexpected number of checked cases: {checked}"
        );
        Ok(())
    }

    /// Q27D-47: the tilt reaches its budget, keeps ratios inside and outside the set, and
    /// leaves points, empty or full sets and zero budgets unchanged.
    #[test]
    fn tilt_meets_the_divergence_budget_and_keeps_ratios() -> Result<()> {
        let law = weighted(&[(3, 0.4), (7, 0.25), (1, 0.2), (9, 0.1), (4, 0.05)]);
        for (candidates, budget) in [
            (&[7u32, 1][..], 0.01),
            (&[3][..], 0.005),
            (&[9, 4][..], 0.02),
        ] {
            let (tilted, kl) = law.tilted_toward(candidates, budget)?;
            let (measured, _, _) = tilted.divergence_terms(&law);
            ensure!(
                (kl - budget).abs() < 1e-9 && (measured - budget).abs() < 1e-9,
                "budget {budget}: {kl} {measured}"
            );
            let mass = |d: &Distribution| candidates.iter().map(|&c| d.probability(c)).sum::<f64>();
            ensure!(mass(&tilted) > mass(&law), "the set did not gain mass");
            let ratio = |d: &Distribution, a, b| d.probability(a) / d.probability(b);
            if candidates.len() == 2 {
                ensure!(
                    (ratio(&tilted, candidates[0], candidates[1])
                        - ratio(&law, candidates[0], candidates[1]))
                    .abs()
                        < 1e-12
                );
            }
            let outside: Vec<u32> = [3u32, 7, 1, 9, 4]
                .into_iter()
                .filter(|t| !candidates.contains(t))
                .collect();
            ensure!(
                (ratio(&tilted, outside[0], outside[1]) - ratio(&law, outside[0], outside[1]))
                    .abs()
                    < 1e-12
            );
        }
        // A budget above ln(1/m) is capped with mass kept outside the set.
        let (capped, kl) = law.tilted_toward(&[4], 10.0)?;
        ensure!(
            kl <= (1.0f64 / 0.05).ln() + 1e-6 && capped.probability(3) > 0.,
            "uncapped tilt"
        );
        for (candidates, budget) in [
            (&[3u32, 7, 1, 9, 4][..], 0.01),
            (&[42][..], 0.01),
            (&[7][..], 0.0),
        ] {
            let (same, kl) = law.tilted_toward(candidates, budget)?;
            ensure!(same == law && kl == 0., "law changed for {candidates:?}");
        }
        let point = Distribution::Point(5);
        ensure!(point.tilted_toward(&[5, 6], 0.1)? == (Distribution::Point(5), 0.));
        ensure!(
            law.tilted_toward(&[7], -1.0).is_err() && law.tilted_toward(&[7], f64::NAN).is_err()
        );
        // The token produced by exact verification of the tilted law follows the tilted law.
        let (tilted, _) = law.tilted_toward(&[7, 1], 0.02)?;
        let mut random = 0xFEED_F00D_1234_5677u64;
        let draws = 400_000usize;
        let mut counts = std::collections::HashMap::<u32, usize>::new();
        for _ in 0..draws {
            let token = match tilted.verify_among(&[7, 1], &mut random)? {
                Choice::Candidate(index) => [7, 1][index],
                Choice::Token(token) => token,
            };
            *counts.entry(token).or_default() += 1;
        }
        for token in [3u32, 7, 1, 9, 4] {
            let p = tilted.probability(token);
            let observed = counts.get(&token).copied().unwrap_or(0) as f64 / draws as f64;
            ensure!(
                (observed - p).abs() <= 4. * (p * (1. - p) / draws as f64).sqrt(),
                "token {token}"
            );
        }
        Ok(())
    }

    /// One candidate is exactly the chain verification (same draws, same verdicts).
    #[test]
    fn verify_among_with_one_candidate_matches_verify() -> Result<()> {
        let distribution = weighted(&[(3, 0.4), (7, 0.25), (1, 0.2), (9, 0.1), (4, 0.05)]);
        for proposal in [3, 7, 4, 42] {
            let (mut a, mut b) = (0x1234_5678_9ABC_DEF1u64, 0x1234_5678_9ABC_DEF1u64);
            for _ in 0..10_000 {
                let chain = distribution.verify(proposal, &mut a)?;
                let tree = distribution.verify_among(&[proposal], &mut b)?;
                let same = match (chain, tree) {
                    (Verdict::Accept, Choice::Candidate(0)) => true,
                    (Verdict::Reject(x), Choice::Token(y)) => x == y,
                    _ => false,
                };
                ensure!(same && a == b, "proposal {proposal}: {chain:?} vs {tree:?}");
            }
        }
        // Points accept the matching candidate, otherwise return their token.
        let point = Distribution::Point(8);
        let mut random = 1;
        assert_eq!(
            point.verify_among(&[2, 8], &mut random)?,
            Choice::Candidate(1)
        );
        assert_eq!(point.verify_among(&[2, 5], &mut random)?, Choice::Token(8));
        assert_eq!(point.verify_among(&[], &mut random)?, Choice::Token(8));
        assert!(distribution.verify_among(&[3, 3], &mut random).is_err());
        Ok(())
    }

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

    /// L1 contract: the canonical order only depends on the kept set and its
    /// weights; the unstable selection order of `distribution` does not.
    fn fuzz_row(seed: u64, vocab: usize) -> Vec<f32> {
        let mut state = seed | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        (0..vocab)
            .map(|_| {
                let r = next();
                match r % 23 {
                    // Coarse values force ties, including at the boundary.
                    0..=5 => -1. - ((r >> 40) % 12) as f32 * 0.5,
                    6 => f32::NEG_INFINITY,
                    _ => -((r >> 40) as f32 / 1048576.0) - 0.001,
                }
            })
            .collect()
    }

    /// Q1 contract: whenever the candidates (all values at least the
    /// threshold) certify a row, the distribution is the canonical
    /// full-row distribution.
    #[test]
    fn sparse_distribution_matches_the_full_row() -> Result<()> {
        let samplers = [
            Sampler::new(0.2, 80, 1., 1.05, 0.)?,
            Sampler::new(0.1, 50, 1., 1.1, 0.)?,
            Sampler::new(0.7, 20, 0.9, 1., 0.3)?,
            Sampler::new(1.0, 5, 0.95, 1.3, 1.5)?,
            Sampler::new(0., 1, 1., 1.2, 0.)?,
            Sampler::new(0.6, 1, 1., 1., 0.4)?,
            Sampler::new(0.8, 256, 1., 1., 0.)?,
        ];
        let mut accepted = [0usize; 7];
        let mut cases = 0;
        for case in 0..700u64 {
            let vocab = 300 + (case as usize * 37) % 1700;
            let values = fuzz_row(case * 7919 + 3, vocab);
            let history: Vec<u32> = fuzz_row(case + 11, 40)
                .iter()
                .enumerate()
                .map(|(i, v)| ((v.to_bits() as usize + i * 97) % vocab) as u32)
                .collect();
            let history = History::from_tokens(&history);
            let extra: Vec<u32> = (0..(case % 4) as u32)
                .map(|i| (i * 131 + case as u32) % vocab as u32)
                .collect();
            let mut finite: Vec<f32> = values.iter().copied().filter(|v| v.is_finite()).collect();
            finite.sort_unstable_by(|a, b| b.total_cmp(a));
            for (index, sampler) in samplers.iter().enumerate() {
                let rank = 1 + (case as usize * 13 + index * 7) % 400;
                let Some(&threshold) = finite.get(rank.min(finite.len() - 1)) else {
                    continue;
                };
                let candidates: Vec<(u32, f32)> = values
                    .iter()
                    .enumerate()
                    .filter(|(_, v)| **v >= threshold)
                    .map(|(i, v)| (i as u32, *v))
                    .rev()
                    .collect();
                cases += 1;
                let expected = sampler
                    .distribution(values.clone(), &history, &extra)?
                    .canonical();
                if let Some(got) =
                    sampler.sparse_distribution(&candidates, threshold, &history, &extra)?
                {
                    accepted[index] += 1;
                    ensure!(
                        got == expected,
                        "case {case} sampler {index}: distributions differ"
                    );
                }
                ensure!(
                    sampler
                        .sparse_distribution(&candidates[1..], threshold, &history, &extra)
                        .is_ok(),
                    "case {case}: sparse distribution errored"
                );
            }
        }
        eprintln!("Q1_FUZZ accepted={accepted:?} cases={cases}");
        ensure!(
            accepted.iter().all(|&n| n > 20),
            "too few certified rows: {accepted:?} of {cases}"
        );
        Ok(())
    }

    #[test]
    fn sparse_distribution_rejects_unsafe_rows() -> Result<()> {
        let sampler = Sampler::new(0.2, 2, 1., 1.5, 0.)?;
        let none = History::default();
        let row = [(0, -1.), (1, -2.), (2, -3.)];
        ensure!(
            sampler
                .sparse_distribution(&row, -3., &none, &[])?
                .is_some(),
            "safe row"
        );
        // Penalty pushes the 2nd candidate under the threshold.
        let penalized = History::from_tokens(&[1]);
        ensure!(
            sampler
                .sparse_distribution(&[(0, -1.), (1, -2.)], -2., &penalized, &[])?
                .is_none(),
            "under threshold"
        );
        // Boundary tie among candidates.
        ensure!(
            sampler
                .sparse_distribution(&[(0, -1.), (1, -2.), (2, -2.)], -2., &none, &[])?
                .is_none(),
            "tie"
        );
        // NaN, duplicates, too few, non-finite threshold.
        ensure!(
            sampler
                .sparse_distribution(&[(0, f32::NAN), (1, -2.)], -2., &none, &[])?
                .is_none(),
            "NaN"
        );
        ensure!(
            sampler
                .sparse_distribution(&[(0, -1.), (0, -1.5)], -2., &none, &[])?
                .is_none(),
            "duplicate"
        );
        ensure!(
            sampler
                .sparse_distribution(&[(0, -1.)], -2., &none, &[])?
                .is_none(),
            "too few"
        );
        ensure!(
            sampler
                .sparse_distribution(&row, f32::NEG_INFINITY, &none, &[])?
                .is_none(),
            "threshold"
        );
        // Penalties that can raise values, or no top-k bound.
        for raising in [
            Sampler::new(0.2, 2, 1., 0.9, 0.)?,
            Sampler::new(0.2, 2, 1., 1., -0.5)?,
            Sampler::new(0.2, 0, 1., 1., 0.)?,
        ] {
            ensure!(
                raising
                    .sparse_distribution(&row, -3., &none, &[])?
                    .is_none(),
                "{raising:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn penalty_span_covers_every_changed_value() -> Result<()> {
        for (case, sampler) in [
            Sampler::new(0.2, 80, 1., 1.05, 0.)?,
            Sampler::new(0.7, 20, 1., 1., 0.5)?,
            Sampler::new(0.7, 20, 1., 1.3, 1.5)?,
            Sampler::new(0.7, 20, 1., 1., 0.)?,
        ]
        .into_iter()
        .enumerate()
        {
            let tokens: Vec<u32> = (0..60).map(|i| (i * 7 + case as u32) % 45).collect();
            let history = History::from_tokens(&tokens);
            let extra = [3, 50, 50, 51];
            let mut values = vec![-1f32; 64];
            sampler.apply_penalties(&mut values, &history, &extra);
            let changed = values.iter().filter(|&&v| v != -1.).count();
            let span = sampler.penalty_span(&history, &extra);
            ensure!(
                changed == span,
                "case {case}: changed {changed}, span {span}"
            );
        }
        Ok(())
    }

    fn grid_draws(first: usize, second: usize) -> (u64, u64) {
        let u1 = ((((first as f64 + 0.5) / GRID as f64) * (1u64 << 53) as f64) as u64) << 11;
        let u2 = (((second as f64 + 0.5) / GRID as f64) * u64::MAX as f64) as u64;
        (u1, u2)
    }

    /// Q3: Σ_x q(x)·law(output | x drawn from q) equals p on a uniform
    /// grid of the two raw draws, for overlapping, disjoint, equal and
    /// point distributions.
    #[test]
    fn verify_draft_delivers_the_target_law_on_a_uniform_grid() -> Result<()> {
        let vocab = 9;
        let history = History::from_tokens(&[2, 5, 2]);
        let dist = |sampler: Sampler, seed: u64| -> Result<Distribution> {
            sampler.distribution(row(vocab, seed), &history, &[])
        };
        let wide = Sampler::new(0.8, 0, 1., 1., 0.)?;
        let pairs = [
            (dist(wide, 1)?, dist(wide, 2)?),
            (
                dist(Sampler::new(1.0, 5, 1., 1.1, 0.)?, 3)?,
                dist(Sampler::new(1.3, 4, 0.9, 1.05, 1.5)?, 4)?,
            ),
            (
                dist(Sampler::new(0.6, 2, 1., 1., 0.)?, 5)?,
                dist(Sampler::new(0.6, 2, 1., 1., 0.)?, 6)?,
            ),
            (dist(wide, 7)?, dist(wide, 7)?),
            (Distribution::Point(3), dist(wide, 8)?),
            (dist(wide, 9)?, Distribution::Point(4)),
            (Distribution::Point(1), Distribution::Point(1)),
            (Distribution::Point(1), Distribution::Point(2)),
        ];
        let mut covered = (false, false);
        for (case, (target, draft)) in pairs.iter().enumerate() {
            let mut law = vec![0f64; vocab];
            for proposal in 0..vocab as u32 {
                let q = draft.probability(proposal);
                if q == 0. {
                    ensure!(
                        target.verify_draft(draft, proposal, &mut 1).is_err(),
                        "case {case}: proposal outside q"
                    );
                    continue;
                }
                for first in 0..GRID {
                    for second in 0..GRID {
                        let (u1, u2) = grid_draws(first, second);
                        let mut calls = 0;
                        let verdict = target.verify_draft_with(draft, proposal, || {
                            calls += 1;
                            if calls == 1 { u1 } else { u2 }
                        })?;
                        let token = match verdict {
                            Verdict::Accept => {
                                covered.0 = true;
                                proposal
                            }
                            Verdict::Reject(token) => {
                                covered.1 = true;
                                token
                            }
                        };
                        law[token as usize] += q / (GRID * GRID) as f64;
                    }
                }
            }
            for token in 0..vocab as u32 {
                let expected = target.probability(token);
                ensure!(
                    (law[token as usize] - expected).abs() <= 3. / GRID as f64,
                    "case {case} token {token}: {} vs {expected}",
                    law[token as usize]
                );
            }
        }
        ensure!(covered == (true, true), "accept and reject covered");
        Ok(())
    }

    /// Q3 chain: drafts drawn from per-level head laws, verified by
    /// `accept_drafts_with`, then completed by ordinary target draws, give
    /// the sequential target law over every full sequence.
    #[test]
    fn sampled_draft_chain_matches_sequential_target_sampling() -> Result<()> {
        // Head laws differ from the target rows (row() shifts by 3·seed mod vocab).
        let heads: Vec<Vec<f32>> = (0..2).map(|i| row(5, 42 + 2 * i as u64)).collect();
        check_sampled_chain(&heads, None)
    }

    /// Q3b: compact head rows over IDs {0, 2, 4}, expanded to the full
    /// vocabulary: drafts stay in the compact set, the delivered law is still
    /// the sequential target law (target mass on 1 and 3 comes from residuals).
    #[test]
    fn compact_sampled_draft_chain_matches_sequential_target_sampling() -> Result<()> {
        let ids = [0u32, 2, 4];
        let heads = (0..2)
            .map(|i| {
                let compact = row(3, 42 + 2 * i as u64);
                expand_compact_row(&compact, &ids, 5)
            })
            .collect::<Result<Vec<_>>>()?;
        check_sampled_chain(&heads, Some(&ids))
    }

    fn check_sampled_chain(heads: &[Vec<f32>], support: Option<&[u32]>) -> Result<()> {
        let vocab = 5;
        let sampler = Sampler::new(0.9, 0, 0.95, 1.2, 0.5)?;
        let base = [1u32, 4];
        let draws = 120_000;
        let depth = heads.len();
        let rows: Vec<Vec<f32>> = (0..=depth).map(|i| row(vocab, 21 + i as u64)).collect();
        let history = History::from_tokens(&base);
        let mut counts = std::collections::HashMap::<Vec<u32>, usize>::new();
        let mut random = 0x2545_f491_4f6c_dd1du64;
        let (mut accepted_total, mut rejected) = (0usize, 0usize);
        for _ in 0..draws {
            let mut proposals = Vec::new();
            let mut drafts = Vec::new();
            for head in heads {
                let draft = sampler.distribution(head.clone(), &history, &proposals)?;
                let choice = draft.sample(&mut random);
                ensure!(
                    support.is_none_or(|ids| ids.contains(&choice)),
                    "draft {choice} outside the compact set"
                );
                proposals.push(choice);
                drafts.push(draft);
            }
            let accepted = Sampler::accept_drafts_with(
                &proposals,
                &drafts,
                |i, extra| sampler.distribution(rows[i].clone(), &history, extra),
                &mut random,
            )?;
            accepted_total += accepted.accepted_draft_tokens;
            rejected += usize::from(accepted.accepted_draft_tokens < depth);
            let mut delivered = proposals[..accepted.accepted_draft_tokens].to_vec();
            delivered.push(accepted.target_token);
            while delivered.len() <= depth {
                let i = delivered.len();
                let next = sampler
                    .distribution(rows[i].clone(), &history, &delivered)?
                    .sample(&mut random);
                delivered.push(next);
            }
            *counts.entry(delivered).or_default() += 1;
        }
        let mut total = 0.;
        let sequences = (0..vocab.pow(depth as u32 + 1)).map(|n| {
            (0..=depth)
                .map(|i| (n / vocab.pow(i as u32) % vocab) as u32)
                .collect::<Vec<_>>()
        });
        for sequence in sequences {
            let mut expected = 1.;
            for i in 0..=depth {
                let mut prefix = base.to_vec();
                prefix.extend_from_slice(&sequence[..i]);
                expected *= oracle(&rows[i], &sampler, &prefix)[sequence[i] as usize];
            }
            total += expected;
            let observed = *counts.get(&sequence).unwrap_or(&0) as f64 / draws as f64;
            let tolerance = 4.5 * (expected * (1. - expected) / draws as f64).sqrt() + 2e-4;
            ensure!(
                (observed - expected).abs() <= tolerance,
                "{sequence:?}: {observed} vs {expected}"
            );
        }
        ensure!((total - 1.).abs() < 1e-9, "oracle mass {total}");
        ensure!(
            accepted_total > 0 && rejected > 0,
            "accept and reject covered: accepted {accepted_total} rejected {rejected}"
        );
        ensure!(
            Sampler::accept_drafts_with(&[1], &[], |_, _| Ok(Distribution::Point(0)), &mut random)
                .is_err(),
            "draft count"
        );
        Ok(())
    }

    #[test]
    fn compact_row_expands_to_full_vocabulary() -> Result<()> {
        // Exhaustive finite domain: every vocabulary of 0..=6 IDs and every
        // increasing ID subset of it; independent oracle by membership search.
        for vocab in 0..=6usize {
            for mask in 0u32..(1 << vocab) {
                let ids: Vec<u32> = (0..vocab as u32).filter(|i| mask >> i & 1 == 1).collect();
                let values: Vec<f32> = ids.iter().map(|&id| -0.5 - id as f32).collect();
                let full = expand_compact_row(&values, &ids, vocab)?;
                ensure!(full.len() == vocab, "length {vocab} {mask}");
                for (j, &value) in full.iter().enumerate() {
                    let expected = match ids.iter().position(|&id| id as usize == j) {
                        Some(c) => values[c],
                        None => f32::NEG_INFINITY,
                    };
                    ensure!(value.to_bits() == expected.to_bits(), "{vocab} {mask} {j}");
                }
            }
        }
        for (values, ids, vocab) in [
            (&[0.0f32][..], &[][..], 4usize),
            (&[0.0, 1.0][..], &[0u32, 4][..], 4),
            (&[0.0, 1.0][..], &[2, 2][..], 4),
            (&[0.0, 1.0][..], &[3, 1][..], 4),
            (&[0.0, 1.0, 2.0][..], &[0, 1, 2][..], 2),
        ] {
            ensure!(
                expand_compact_row(values, ids, vocab).is_err(),
                "{values:?} {ids:?} {vocab}"
            );
        }
        Ok(())
    }

    #[test]
    fn sampled_draft_gate_is_strict() {
        assert!(sampled_draft_enabled_by(Some(b"1")));
        for flag in [None, Some(&b"0"[..]), Some(b""), Some(b"yes"), Some(b"1\n")] {
            assert!(!sampled_draft_enabled_by(flag), "{flag:?}");
        }
    }

    #[test]
    fn canonical_distribution_ignores_selection_order() -> Result<()> {
        let sampler = Sampler::new(0.7, 5, 1., 1.05, 0.)?;
        let history = History::from_tokens(&[3, 9]);
        let full = logits(11, 300);
        // Same kept set and values, other entries pushed far below them.
        let mut masked = full.clone();
        let mut order: Vec<_> = (0..full.len()).collect();
        order.sort_by(|&a, &b| full[b].total_cmp(&full[a]));
        for &index in &order[40..] {
            masked[index] = f32::NEG_INFINITY;
        }
        let a = sampler.distribution(full, &history, &[])?.canonical();
        let b = sampler.distribution(masked, &history, &[])?.canonical();
        assert_eq!(a, b);
        let Distribution::Weighted {
            tokens,
            weights,
            total,
        } = &a
        else {
            panic!("weighted expected")
        };
        assert_eq!(tokens.len(), 5);
        assert!(tokens.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(*total, weights.iter().sum::<f64>());
        // Reversed input order gives the same canonical law and draws.
        let reversed = Distribution::Weighted {
            tokens: tokens.iter().rev().copied().collect(),
            weights: weights.iter().rev().copied().collect(),
            total: 0.0,
        }
        .canonical();
        assert_eq!(reversed, a);
        let (mut left, mut right) = (5u64, 5u64);
        for _ in 0..100 {
            assert_eq!(a.sample(&mut left), reversed.sample(&mut right));
        }
        assert_eq!(Distribution::Point(4).canonical(), Distribution::Point(4));
        Ok(())
    }

    #[test]
    fn certified_k_requires_lowering_penalties_and_a_bounded_top_k() -> Result<()> {
        assert_eq!(
            Sampler::new(0.2, 80, 1., 1.05, 0.)?.certified_k(256),
            Some(80)
        );
        assert_eq!(Sampler::new(0.2, 80, 1., 1.05, 0.)?.certified_k(79), None);
        assert_eq!(Sampler::new(0.2, 0, 1., 1., 0.)?.certified_k(256), None);
        assert_eq!(Sampler::new(0.2, 4, 1., 1., 0.5)?.certified_k(256), Some(4));
        assert_eq!(Sampler::new(0.2, 4, 1., 1., -0.5)?.certified_k(256), None);
        assert_eq!(Sampler::new(0.2, 4, 1., 0.9, 0.)?.certified_k(256), None);
        assert_eq!(Sampler::new(0., 4, 1., 1.2, 0.)?.certified_k(256), Some(1));
        assert_eq!(Sampler::new(0.9, 1, 1., 1.2, 0.)?.certified_k(256), Some(1));
        Ok(())
    }

    /// `apply_penalties` is the penalty stage of `distribution`.
    #[test]
    fn penalties_only_lower_log_probabilities() -> Result<()> {
        let sampler = Sampler::new(0.7, 0, 1., 1.3, 0.4)?;
        let history = History::from_tokens(&[1, 2, 2, 7]);
        let values = logits(5, 50);
        let mut penalized = values.clone();
        sampler.apply_penalties(&mut penalized, &history, &[9]);
        for (index, (&before, &after)) in values.iter().zip(&penalized).enumerate() {
            if [1, 2, 7, 9].contains(&index) && before.is_finite() {
                assert!(after < before, "{index}");
            } else {
                assert_eq!(after.to_bits(), before.to_bits(), "{index}");
            }
        }
        Ok(())
    }
}
