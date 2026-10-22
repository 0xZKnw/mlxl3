//! Single-lane adaptation of Splash VerifyBudget (Apache-2.0), commit1482f6bb.
//! Wall time prices a whole cycle; only the verified prefix supplies labels.
use std::collections::VecDeque;

const PRIOR: [f64; 8] = [0.7, 0., -0.35, -0.8, -0.75, -0.75, -0.85, -0.45];

/// Per-request B2/B3/B4 costs, measured before choosing the next draft shape.
#[derive(Default, Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreDraftBudget {
    totals: [[u64; 2]; 3], // Emitted tokens and whole-cycle nanoseconds.
    cycles: u64,
}
impl PreDraftBudget {
    pub(crate) fn choose(&self, maximum: usize) -> usize {
        let limit = maximum.min(3);
        if limit == 0 {
            return 0;
        }
        if let Some(index) = (0..limit).find(|&i| self.totals[i][1] == 0) {
            return index + 1;
        }
        if self.cycles.is_multiple_of(16) {
            return (self.cycles / 16 % limit as u64) as usize + 1;
        }
        let mut best = 0;
        for i in 1..limit {
            if u128::from(self.totals[i][0]) * u128::from(self.totals[best][1])
                > u128::from(self.totals[best][0]) * u128::from(self.totals[i][1])
            {
                best = i;
            }
        }
        best + 1
    }

    pub(crate) fn observe(&mut self, width: usize, emitted: usize, nanos: u64) -> Option<()> {
        if !(1..=3).contains(&width) || emitted == 0 || emitted > width + 1 || nanos == 0 {
            return None;
        }
        let row = self.totals[width - 1];
        let updated = [
            row[0].checked_add(emitted as u64)?,
            row[1].checked_add(nanos)?,
        ];
        let cycles = self.cycles.checked_add(1)?;
        self.totals[width - 1] = updated;
        self.cycles = cycles;
        Some(())
    }
}
#[derive(Clone, Copy, Debug)]
struct Label {
    position: usize,
    feature: f64,
    accepted: bool,
}

pub(crate) struct VerifyBudget {
    theta: [f64; 8],
    labels: VecDeque<Label>,
    sums: [f64; 5],
    cycles: u64,
}
impl Default for VerifyBudget {
    fn default() -> Self {
        Self {
            theta: PRIOR,
            labels: VecDeque::new(),
            sums: [0.; 5],
            cycles: 0,
        }
    }
}
fn logit(p: f64) -> f64 {
    let p = p.clamp(1e-6, 1. - 1e-6);
    (p / (1. - p)).ln()
}
fn sigmoid(z: f64) -> f64 {
    let t = (-z.abs()).exp();
    if z >= 0. { 1. / (1. + t) } else { t / (1. + t) }
}
fn seed(rows: f64) -> f64 {
    let (lo, hi, a, b) = if rows <= 2. {
        (1., 2., 0.60, 0.69)
    } else if rows <= 4. {
        (2., 4., 0.69, 0.85)
    } else {
        (4., 8., 0.85, 1.)
    };
    a + (rows - lo) / (hi - lo) * (b - a)
}

/// The anchor is always verified. Invalid values never reach a GPU dispatch.
pub(crate) fn best_prefix(probabilities: &[f64], costs: &[f64]) -> Option<usize> {
    if probabilities.len() > 7
        || costs.len() != probabilities.len() + 1
        || probabilities
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || costs.iter().any(|c| !c.is_finite() || *c <= 0.)
    {
        return None;
    }
    let (mut best, mut score, mut survival, mut expected) = (0, 1. / costs[0], 1., 1.);
    for (i, p) in probabilities.iter().enumerate() {
        survival *= p;
        expected += survival;
        let candidate = expected / costs[i + 1];
        if candidate > score {
            best = i + 1;
            score = candidate;
        }
    }
    Some(best)
}

/// At most two non-overlapping physical writes, retaining only the final window.
pub(crate) fn ring_writes(start: i32, rows: i32) -> Option<[(i32, i32, i32); 2]> {
    if start < 0 || rows <= 0 {
        return None;
    }
    start.checked_add(rows)?;
    let source = (rows - 2048).max(0);
    let kept = rows - source;
    let slot = (start + source) % 2048;
    let first = kept.min(2048 - slot);
    Some([(source, slot, first), (source + first, 0, kept - first)])
}

impl VerifyBudget {
    fn line(&self) -> Option<(f64, f64)> {
        let [w, r, rr, t, rt] = self.sums;
        if w <= 0. {
            return None;
        }
        let mean_r = r / w;
        let mean_t = t / w;
        let spread = (rr / w - mean_r * mean_r).max(0.);
        let covariance = rt / w - mean_r * mean_t;
        let prior_slope = (1. - 0.60) / 7. * mean_t / seed(mean_r.round());
        let slope = ((w * covariance + 5. * prior_slope) / (w * spread + 5.)).max(0.);
        Some((mean_t - slope * mean_r, slope))
    }
    fn costs(&self) -> [f64; 8] {
        std::array::from_fn(|i| match self.line() {
            None => seed((i + 1) as f64),
            Some((a, b)) => (a + b * (i + 1) as f64).max(0.05 * self.sums[3] / self.sums[0]),
        })
    }
    pub(crate) fn choose(&self, confidence: &[f32]) -> Option<usize> {
        if confidence.len() > 7
            || confidence
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            return None;
        }
        if self.cycles % 32 == 31 {
            return Some(confidence.len());
        }
        let p = confidence
            .iter()
            .enumerate()
            .map(|(i, &p)| sigmoid(self.theta[0] * logit(f64::from(p)) + self.theta[i + 1]))
            .collect::<Vec<_>>();
        best_prefix(&p, &self.costs()[..=p.len()])
    }
    pub(crate) fn observe(
        &mut self,
        confidence: &[f32],
        accepted: usize,
        seconds: f64,
    ) -> Option<()> {
        if confidence.len() > 7
            || accepted > confidence.len()
            || !seconds.is_finite()
            || !(1e-9..=360.).contains(&seconds)
            || confidence
                .iter()
                .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        {
            return None;
        }
        let rows = (confidence.len() + 1) as f64;
        let time = match self.line() {
            Some((a, b)) => {
                let predicted = (a + b * rows).max(1e-3);
                seconds.clamp(0.5 * predicted, 2. * predicted)
            }
            None => seconds,
        };
        for (sum, value) in self
            .sums
            .iter_mut()
            .zip([1., rows, rows * rows, time, rows * time])
        {
            *sum = 0.97 * *sum + value;
        }
        for (position, &p) in confidence
            .iter()
            .take(accepted.saturating_add(1))
            .enumerate()
        {
            if self.labels.len() == 4096 {
                self.labels.pop_front();
            }
            self.labels.push_back(Label {
                position,
                feature: logit(f64::from(p)),
                accepted: position < accepted,
            });
        }
        self.cycles = self.cycles.wrapping_add(1);
        if self.cycles.is_multiple_of(64) && !self.labels.is_empty() {
            self.fit();
        }
        Some(())
    }
    fn objective(&self, theta: &[f64; 8]) -> f64 {
        let data: f64 = self
            .labels
            .iter()
            .map(|l| {
                let z = theta[0] * l.feature + theta[l.position + 1];
                let margin = if l.accepted { z } else { -z };
                (-margin).max(0.) + (-margin.abs()).exp().ln_1p()
            })
            .sum();
        data + theta
            .iter()
            .zip(PRIOR)
            .enumerate()
            .map(|(i, (a, b))| 0.5 * if i == 0 { 50. } else { 4. } * (a - b).powi(2))
            .sum::<f64>()
    }
    fn fit(&mut self) {
        let mut theta = self.theta;
        for _ in 0..16 {
            let mut gradient = [0.; 8];
            let mut h = [[0.; 8]; 8];
            for l in &self.labels {
                let b = l.position + 1;
                let p = sigmoid(theta[0] * l.feature + theta[b]);
                let residual = p - f64::from(l.accepted);
                let weight = p * (1. - p);
                gradient[0] += residual * l.feature;
                gradient[b] += residual;
                h[0][0] += weight * l.feature * l.feature;
                h[0][b] += weight * l.feature;
                h[b][0] += weight * l.feature;
                h[b][b] += weight;
            }
            for i in 0..8 {
                let precision = if i == 0 { 50. } else { 4. };
                gradient[i] += precision * (theta[i] - PRIOR[i]);
                h[i][i] += precision;
            }
            for col in 0..8 {
                let pivot = (col..8)
                    .max_by(|&a, &b| h[a][col].abs().total_cmp(&h[b][col].abs()))
                    .unwrap();
                h.swap(col, pivot);
                gradient.swap(col, pivot);
                for row in col + 1..8 {
                    let factor = h[row][col] / h[col][col];
                    let pivot_row = h[col];
                    for (value, pivot) in h[row].iter_mut().zip(pivot_row).skip(col) {
                        *value -= factor * pivot;
                    }
                    gradient[row] -= factor * gradient[col];
                }
            }
            let mut step = [0.; 8];
            for row in (0..8).rev() {
                step[row] = (gradient[row]
                    - (row + 1..8).map(|k| h[row][k] * step[k]).sum::<f64>())
                    / h[row][row];
            }
            let current = self.objective(&theta);
            let mut scale = 1.;
            let mut moved = false;
            for _ in 0..30 {
                let next = std::array::from_fn(|i| theta[i] - scale * step[i]);
                if next.iter().all(|v| v.is_finite()) && self.objective(&next) <= current {
                    theta = next;
                    moved = true;
                    break;
                }
                scale *= 0.5;
            }
            if !moved || step.iter().map(|v| (scale * v).abs()).fold(0., f64::max) < 1e-9 {
                break;
            }
        }
        self.theta = theta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ring_writes_match_independent_unbounded_history() {
        for start in [0, 1, 2047, 2048, 2049, 4095, 100_000, i32::MAX - 8192] {
            for rows in [1, 2, 8, 2047, 2048, 2049, 8192] {
                let mut actual = vec![None; 2048];
                for (src, slot, n) in ring_writes(start, rows).unwrap() {
                    for i in 0..n {
                        actual[(slot + i) as usize] = Some(start + src + i);
                    }
                }
                let mut expected = vec![None; 2048];
                for position in start..start + rows {
                    expected[position as usize % 2048] = Some(position);
                }
                assert_eq!(actual, expected);
            }
        }
        for (start, rows) in [(-1, 1), (0, 0), (0, -1), (i32::MAX, 1)] {
            assert!(ring_writes(start, rows).is_none());
        }
    }
    #[test]
    fn prefix_optimum_and_ties_match_bernoulli_enumeration() {
        for bits in 0..(1usize << 14) {
            let p: Vec<_> = (0..7)
                .map(|i| f64::from(((bits >> (2 * i)) & 3) as u8) / 3.)
                .collect();
            let costs = std::array::from_fn::<_, 8, _>(|i| (1 + i) as f64);
            let mut rewards = [0.; 8];
            for outcomes in 0..128 {
                let mass = (0..7)
                    .map(|i| {
                        if outcomes & (1 << i) != 0 {
                            p[i]
                        } else {
                            1. - p[i]
                        }
                    })
                    .product::<f64>();
                for (n, reward) in rewards.iter_mut().enumerate() {
                    let accepted = (0..n).take_while(|&i| outcomes & (1 << i) != 0).count();
                    *reward += mass * (1 + accepted) as f64;
                }
            }
            let actual = best_prefix(&p, &costs).unwrap();
            assert!(
                rewards
                    .iter()
                    .zip(costs)
                    .all(|(reward, cost)| rewards[actual] / costs[actual] + 1e-12 >= reward / cost)
            );
        }
        assert_eq!(best_prefix(&[1., 1.], &[1., 2., 3.]), Some(0));
        assert!(best_prefix(&[f64::NAN], &[1., 2.]).is_none());
        assert!(best_prefix(&[0.5], &[0., 2.]).is_none());
        assert!(best_prefix(&[0.5; 8], &[1.; 9]).is_none());
    }
    #[test]
    fn newton_fit_matches_independent_one_dimensional_posterior() {
        for accepted in [0, 7] {
            let mut budget = VerifyBudget::default();
            for _ in 0..128 {
                budget.observe(&[0.5; 7], accepted, 0.1).unwrap();
            }
            assert!((budget.theta[0] - PRIOR[0]).abs() < 1e-12);
            for (position, &prior) in PRIOR[1..].iter().enumerate() {
                if accepted == 0 && position > 0 {
                    assert_eq!(budget.theta[position + 1], prior);
                    continue;
                }
                let (mut lo, mut hi) = (-50., 50.);
                for _ in 0..100 {
                    let mid = (lo + hi) / 2.;
                    let derivative = 4. * (mid - prior)
                        + 128. * (sigmoid(mid) - if accepted == 7 { 1. } else { 0. });
                    if derivative > 0. {
                        hi = mid;
                    } else {
                        lo = mid;
                    }
                }
                assert!((budget.theta[position + 1] - (lo + hi) / 2.).abs() < 1e-7);
            }
        }
    }
    #[test]
    fn learning_is_bounded_and_never_labels_a_tail_after_rejection() {
        let mut budget = VerifyBudget::default();
        for cycle in 0..1024 {
            let accepted = cycle % 8;
            let before = budget.labels.len();
            budget
                .observe(&[0.8; 7], accepted, 0.1 + accepted as f64 * 0.01)
                .unwrap();
            let added = (accepted + 1).min(7);
            assert_eq!(budget.labels.len(), (before + added).min(4096));
            for l in budget.labels.iter().rev().take(added) {
                assert!(l.position <= accepted);
                assert_eq!(l.accepted, l.position < accepted);
            }
            let costs = budget.costs();
            assert!(costs.iter().all(|c| c.is_finite() && *c > 0.));
            assert!(costs.windows(2).all(|p| p[0] <= p[1]));
            assert!(budget.choose(&[0.8; 7]).unwrap() <= 7);
        }
        assert_ne!(budget.theta, PRIOR);
        let before = (budget.cycles, budget.labels.len(), budget.sums);
        assert!(budget.observe(&[0.5], 2, 1.).is_none());
        assert!(budget.observe(&[f32::NAN], 0, 1.).is_none());
        assert!(budget.observe(&[], 0, 0.).is_none());
        assert!(budget.observe(&[0.5], 0, f64::MAX).is_none());
        assert_eq!((budget.cycles, budget.labels.len(), budget.sums), before);
        assert!(budget.choose(&[1.1]).is_none());
    }
}

#[cfg(kani)]
mod verification {
    use super::*;
    #[kani::proof]
    #[kani::unwind(3)]
    fn circular_writes_are_bounded_disjoint_and_cannot_wrap() {
        let start: i32 = kani::any();
        let rows: i32 = kani::any();
        if let Some(parts) = ring_writes(start, rows) {
            let [(source, slot, first), (second, zero, last)] = parts;
            assert!(start >= 0 && rows > 0 && start.checked_add(rows).is_some());
            assert!(source >= 0 && first > 0 && first + last == rows.min(2048));
            assert!(slot >= 0 && slot + first <= 2048 && zero == 0 && last <= slot);
            assert!(second == source + first && second + last == rows);
            kani::cover!(last > 0);
            kani::cover!(rows > 2048);
        }
    }
    #[kani::proof]
    #[kani::unwind(9)]
    fn adaptive_prefix_is_bounded_and_rejects_invalid_probabilities() {
        let probabilities: [f64; 7] = kani::any();
        let costs: [f64; 8] = kani::any();
        if let Some(n) = best_prefix(&probabilities, &costs) {
            assert!(n <= 7);
            for p in probabilities {
                assert!(p.is_finite() && (0.0..=1.0).contains(&p));
            }
            for c in costs {
                assert!(c.is_finite() && c > 0.);
            }
            kani::cover!(n == 0);
            kani::cover!(n == 7);
        }
    }
}

#[cfg(test)]
mod pre_draft_tests {
    use super::PreDraftBudget;

    #[test]
    fn pre_draft_warms_each_legal_shape_then_prices_emitted_tokens() {
        let mut b = PreDraftBudget::default();
        assert_eq!(b.choose(0), 0);
        assert_eq!(b.choose(usize::MAX), 1);
        b.observe(1, 2, 200).unwrap();
        assert_eq!(b.choose(1), 1);
        assert_eq!(b.choose(3), 2);
        b.observe(2, 3, 1000).unwrap();
        assert_eq!(b.choose(2), 1);
        assert_eq!(b.choose(8), 3);
        b.observe(3, 4, 1000).unwrap();
        assert_eq!(b.choose(3), 1);
        b.observe(2, 3, 10).unwrap();
        assert_eq!(b.choose(3), 1); // A fast sample does not erase previous work.
        b.observe(3, 4, 10).unwrap();
        assert_eq!(b.choose(3), 1);
    }

    #[test]
    fn pre_draft_explores_all_three_shapes_and_clips_to_remaining_budget() {
        let mut b = PreDraftBudget {
            totals: [[2, 100], [3, 100], [4, 100]],
            cycles: 3,
        };
        assert_eq!(b.choose(3), 3);
        assert_eq!(b.choose(2), 2);
        assert_eq!(b.choose(1), 1);
        for (cycle, width) in [(16, 2), (32, 3), (48, 1)] {
            b.cycles = cycle;
            assert_eq!(b.choose(3), width);
            assert!((1..=2).contains(&b.choose(2)));
        }
    }

    #[test]
    fn pre_draft_rejects_invalid_or_exhausted_counters_without_mutation() {
        for (width, emitted, nanos) in [
            (0, 1, 1),
            (4, 1, 1),
            (usize::MAX, 1, 1),
            (1, 0, 1),
            (1, 3, 1),
            (3, usize::MAX, 1),
            (1, 1, 0),
        ] {
            let mut b = PreDraftBudget::default();
            let before = b.clone();
            assert_eq!(b.observe(width, emitted, nanos), None);
            assert_eq!(b, before);
        }
        for (tokens, nanos, cycles) in [(u64::MAX, 1, 1), (1, u64::MAX, 1), (1, 1, u64::MAX)] {
            let mut b = PreDraftBudget {
                totals: [[tokens, nanos]; 3],
                cycles,
            };
            let before = b.clone();
            assert_eq!(b.observe(1, 1, 1), None);
            assert_eq!(b, before);
        }
    }

    #[test]
    fn pre_draft_integer_prices_and_long_trace_stay_bounded() {
        let mut b = PreDraftBudget {
            totals: [[u64::MAX - 1, u64::MAX]; 3],
            cycles: 3,
        };
        b.totals[2] = [u64::MAX, u64::MAX - 1];
        assert_eq!(b.choose(usize::MAX), 3);
        let mut b = PreDraftBudget::default();
        let mut widths = [0; 3];
        for _ in 0..192 {
            let width = b.choose(7);
            assert!((1..=3).contains(&width));
            widths[width - 1] += 1;
            b.observe(width, width + 1, [100, 300, 500][width - 1])
                .unwrap();
        }
        assert!(widths[0] > 170 && widths[1] > 1 && widths[2] > 1);
        assert_eq!(b.cycles, 192);
        assert_eq!(PreDraftBudget::default().choose(3), 1);
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn pre_draft_choice_is_bounded_for_all_states() {
    let b = PreDraftBudget {
        totals: kani::any(),
        cycles: kani::any(),
    };
    let maximum: usize = kani::any();
    let choice = b.choose(maximum);
    assert!(choice <= maximum.min(3));
    assert_eq!(choice == 0, maximum == 0);
    kani::cover!(choice == 0);
    kani::cover!(choice == 1);
    kani::cover!(choice == 2);
    kani::cover!(choice == 3);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn pre_draft_observation_is_atomic_and_checked() {
    let mut b = PreDraftBudget {
        totals: kani::any(),
        cycles: kani::any(),
    };
    let before = b.clone();
    let width: usize = kani::any();
    let emitted: usize = kani::any();
    let nanos: u64 = kani::any();
    let result = b.observe(width, emitted, nanos);
    if result.is_some() {
        assert!((1..=3).contains(&width) && (1..=width + 1).contains(&emitted) && nanos > 0);
        assert_eq!(u128::from(b.cycles), u128::from(before.cycles) + 1);
        for i in 0..3 {
            if i == width - 1 {
                assert_eq!(
                    u128::from(b.totals[i][0]),
                    u128::from(before.totals[i][0]) + emitted as u128
                );
                assert_eq!(
                    u128::from(b.totals[i][1]),
                    u128::from(before.totals[i][1]) + u128::from(nanos)
                );
            } else {
                assert_eq!(b.totals[i][0], before.totals[i][0]);
                assert_eq!(b.totals[i][1], before.totals[i][1]);
            }
        }
    } else {
        assert_eq!(b.cycles, before.cycles);
        for i in 0..3 {
            assert_eq!(b.totals[i][0], before.totals[i][0]);
            assert_eq!(b.totals[i][1], before.totals[i][1]);
        }
    }
    kani::cover!(result.is_none());
    kani::cover!(result.is_some() && width == 1);
    kani::cover!(result.is_some() && width == 2);
    kani::cover!(result.is_some() && width == 3);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn pre_draft_prices_choose_the_fastest_observed_shape() {
    let b = PreDraftBudget {
        totals: kani::any(),
        cycles: kani::any(),
    };
    let maximum: usize = kani::any();
    let limit = maximum.min(3);
    if limit > 0 && !b.cycles.is_multiple_of(16) && (0..limit).all(|i| b.totals[i][1] > 0) {
        let choice = b.choose(maximum) - 1;
        for i in 0..limit {
            assert!(
                u128::from(b.totals[choice][0]) * u128::from(b.totals[i][1])
                    >= u128::from(b.totals[i][0]) * u128::from(b.totals[choice][1])
            );
        }
        kani::cover!(choice == 0);
        kani::cover!(choice == 1);
        kani::cover!(choice == 2);
    }
}
