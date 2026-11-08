//! Bounded CPU planning for native speculative execution.
//!
//! These plans do not allocate memory, perform I/O, or authorize a GPU path.
//! The caller must provide disjoint allocation upper bounds and keep the existing
//! runtime admission guard until those bounds cover the actual execution path.

/// Number of proposals, excluding the anchor. B4 verification has three proposals.
pub const MAX_DEPTH: usize = 3;
pub const MAX_ROWS: usize = MAX_DEPTH + 1;
pub const TOP_K: usize = 10;
pub const EXPERTS: u16 = 512;

/// Peak live allocations, not RSS plus a second count of the same GPU buffers.
/// `base_bytes` includes the AR target, its expert cache and its ordinary scratch.
/// All other fields are additional, disjoint upper bounds for speculation.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MemoryPlan {
    pub base_bytes: u64,
    pub reserve_bytes: u64,
    pub draft_weights_bytes: u64,
    pub draft_kv_bytes: u64,
    pub draft_scratch_bytes: u64,
    pub verify_scratch_bytes: u64,
    pub prefix_state_bytes_per_row: u64,
    pub retained_expert_generations_bytes: u64,
}

impl MemoryPlan {
    /// Return None on unsupported depth or overflow, never wrap an admission sum.
    /// Depth zero is AR and allocates no draft/verifier buffers from this plan.
    pub fn peak_bytes(self, depth: usize) -> Option<u64> {
        if depth > MAX_DEPTH {
            return None;
        }
        let base = self.base_bytes.checked_add(self.reserve_bytes)?;
        if depth == 0 {
            return Some(base);
        }
        let prefixes = self
            .prefix_state_bytes_per_row
            .checked_mul((depth + 1) as u64)?;
        base.checked_add(self.draft_weights_bytes)?
            .checked_add(self.draft_kv_bytes)?
            .checked_add(self.draft_scratch_bytes)?
            .checked_add(self.verify_scratch_bytes)?
            .checked_add(prefixes)?
            .checked_add(self.retained_expert_generations_bytes)
    }

    /// Choose before allocating/drafting. None means even the AR step cannot fit
    /// or there is no output/context budget. This does not bypass runtime guards.
    pub fn admitted_depth(
        self,
        requested: usize,
        context_remaining: usize,
        output_remaining: usize,
        limit_bytes: u64,
    ) -> Option<usize> {
        let depth = context_remaining
            .min(output_remaining)
            .checked_sub(1)?
            .min(requested)
            .min(MAX_DEPTH);
        (0..=depth)
            .rev()
            .find(|&d| self.peak_bytes(d).is_some_and(|peak| peak <= limit_bytes))
    }
}

/// Actual complete wall-clock cycles for one fixed depth/cache configuration.
/// Count correction/bonus tokens when committed, including a pending bonus, once
/// in the original cycle. Do not add overlapping GPU and SSD timing intervals.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DepthStats {
    depth: Option<usize>,
    cycles: u64,
    wall_ns: u64,
    emitted: u64,
}

impl DepthStats {
    /// Invalid/overflowing samples leave the old accumulator unchanged.
    pub fn record(&mut self, depth: usize, wall_ns: u64, emitted: usize) -> bool {
        if depth > MAX_DEPTH
            || self.depth.is_some_and(|old| old != depth)
            || wall_ns == 0
            || emitted == 0
            || emitted > depth + 1
        {
            return false;
        }
        let next = (|| {
            Some(Self {
                depth: Some(depth),
                cycles: self.cycles.checked_add(1)?,
                wall_ns: self.wall_ns.checked_add(wall_ns)?,
                emitted: self.emitted.checked_add(emitted as u64)?,
            })
        })();
        match next {
            Some(next) => {
                *self = next;
                true
            }
            None => false,
        }
    }

    pub fn totals(self) -> (u64, u64, u64) {
        (self.cycles, self.wall_ns, self.emitted)
    }

    fn faster_than(self, other: Self) -> bool {
        u128::from(self.emitted) * u128::from(other.wall_ns)
            > u128::from(other.emitted) * u128::from(self.wall_ns)
    }
}

/// Select measured useful throughput, not draft confidence, before the next
/// greedy draft. Unknown candidates are not guessed; callers can gather fixed-
/// depth calibration separately. Reset samples when model/cache/sampling changes.
/// Ties choose shallower depth. No valid AR reference conservatively selects AR.
pub fn measured_depth(
    samples: &[DepthStats; MAX_ROWS],
    maximum: usize,
    minimum_cycles: u64,
) -> usize {
    let required = minimum_cycles.max(1);
    if samples[0].depth != Some(0) || samples[0].cycles < required || samples[0].wall_ns == 0 {
        return 0;
    }
    let mut selected = 0;
    for depth in 1..=maximum.min(MAX_DEPTH) {
        if samples[depth].depth == Some(depth)
            && samples[depth].cycles >= required
            && samples[depth].wall_ns != 0
            && samples[depth].faster_than(samples[selected])
        {
            selected = depth;
        }
    }
    selected
}

/// A contribution's canonical destination; reduce by rank within each row even
/// when experts are read/computed in another order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Consumer {
    pub row: usize,
    pub rank: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExpertCohort {
    pub expert: u16,
    pub consumers: Vec<Consumer>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CohortPlan {
    pub cohorts: Vec<ExpertCohort>,
    /// Sequential half-open ranges into cohorts. Acquire/pin only this wave,
    /// then release after its GPU consumers finish, not all cohorts at once.
    pub waves: Vec<std::ops::Range<usize>>,
}

/// Plan one layer's *actual* routes, in stable first-use order (anchor first).
/// Caller still owns cache generations, resident checks, read errors and leases.
/// No routing approximation, sorting by cache residency or whole-union pinning.
pub fn expert_cohorts(routes: &[[u16; TOP_K]], max_inflight: usize) -> Option<CohortPlan> {
    if routes.is_empty() || routes.len() > MAX_ROWS || max_inflight == 0 {
        return None;
    }
    let mut cohorts: Vec<ExpertCohort> = Vec::with_capacity(routes.len() * TOP_K);
    for (row, experts) in routes.iter().enumerate() {
        for (rank, &expert) in experts.iter().enumerate() {
            if expert >= EXPERTS || experts[..rank].contains(&expert) {
                return None;
            }
            let consumer = Consumer { row, rank };
            if let Some(cohort) = cohorts.iter_mut().find(|group| group.expert == expert) {
                cohort.consumers.push(consumer);
            } else {
                cohorts.push(ExpertCohort {
                    expert,
                    consumers: vec![consumer],
                });
            }
        }
    }
    let width = max_inflight.min(cohorts.len());
    let waves = (0..cohorts.len())
        .step_by(width)
        .map(|start| start..(start + width).min(cohorts.len()))
        .collect();
    Some(CohortPlan { cohorts, waves })
}

pub const MAX_SUFFIX_LOOKBACK: usize = 4096;
pub const MAX_SUFFIX_ORDER: usize = 32;

/// Request-local lookup only: history must be the verified token IDs for the
/// current model/tokenizer/template. No cache survives a request or model change.
/// Proposals still require target verification. This is a greedy proposer, not a
/// sampled distribution q. Stop at and include the first EOS when encountered.
pub fn suffix_proposals(
    history: &[u32],
    lookback: usize,
    max_order: usize,
    max_proposals: usize,
    eos: &[u32],
) -> Vec<u32> {
    let count = max_proposals.min(MAX_DEPTH);
    let lookback = lookback.min(MAX_SUFFIX_LOOKBACK).min(history.len());
    if count == 0 || lookback < 2 || max_order == 0 {
        return Vec::new();
    }
    let window = &history[history.len() - lookback..];
    // A historical match and at least one continuation must precede the suffix.
    let largest = max_order.min(MAX_SUFFIX_ORDER).min((window.len() - 1) / 2);
    for order in (1..=largest).rev() {
        let suffix_start = window.len() - order;
        for start in (0..suffix_start - order).rev() {
            if window[start..start + order] != window[suffix_start..] {
                continue;
            }
            let continuation = &window[start + order..suffix_start];
            let mut result = Vec::with_capacity(count);
            for &token in continuation.iter().take(count) {
                result.push(token);
                if eos.contains(&token) {
                    break;
                }
            }
            return result;
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_admission_counts_overlap_and_never_wraps() {
        let plan = MemoryPlan {
            base_bytes: 100,
            reserve_bytes: 10,
            draft_weights_bytes: 20,
            draft_kv_bytes: 30,
            draft_scratch_bytes: 40,
            verify_scratch_bytes: 50,
            prefix_state_bytes_per_row: 60,
            retained_expert_generations_bytes: 70,
        };
        assert_eq!(plan.peak_bytes(0), Some(110));
        assert_eq!(plan.peak_bytes(1), Some(440));
        assert_eq!(plan.peak_bytes(3), Some(560));
        for limit in 0..=600 {
            let expected = [110, 440, 500, 560].iter().rposition(|&n| n <= limit);
            assert_eq!(plan.admitted_depth(usize::MAX, 4, 4, limit), expected);
        }
        assert_eq!(plan.admitted_depth(3, 1, 4, 600), Some(0));
        assert_eq!(plan.admitted_depth(3, 4, 1, 600), Some(0));
        assert_eq!(plan.admitted_depth(3, 0, 4, 600), None);
        assert_eq!(plan.admitted_depth(3, 4, 0, 600), None);
        assert_eq!(plan.peak_bytes(MAX_DEPTH + 1), None);
        let overflow = MemoryPlan { base_bytes: u64::MAX, reserve_bytes: 1, ..plan };
        assert_eq!(overflow.admitted_depth(3, 4, 4, u64::MAX), None);
        let prefix_overflow = MemoryPlan { prefix_state_bytes_per_row: u64::MAX, ..plan };
        assert_eq!(prefix_overflow.admitted_depth(3, 4, 4, u64::MAX), Some(0));
    }

    #[test]
    fn measured_depth_includes_overhead_and_can_return_to_ar() {
        let mut stats = [DepthStats::default(); MAX_ROWS];
        assert_eq!(measured_depth(&stats, 3, 1), 0);
        assert!(stats[0].record(0, 100, 1));
        assert!(stats[1].record(1, 150, 2));
        assert!(stats[2].record(2, 250, 3));
        assert!(stats[3].record(3, 500, 4));
        assert_eq!(measured_depth(&stats, 3, 1), 1);
        assert_eq!(measured_depth(&stats, 0, 1), 0);
        assert_eq!(measured_depth(&stats, 3, 2), 0);
        assert!(stats[1].record(1, 1000, 1));
        assert_eq!(measured_depth(&stats, 3, 1), 2);
        assert!(stats[2].record(2, 1000, 1));
        assert_eq!(measured_depth(&stats, 3, 1), 0);
        let mut tie = [DepthStats::default(); MAX_ROWS];
        assert!(tie[0].record(0, 100, 1));
        assert!(tie[1].record(1, 200, 2));
        assert_eq!(measured_depth(&tie, 3, 1), 0);
        let before = stats[0];
        assert!(!stats[0].record(0, 0, 1));
        assert!(!stats[0].record(0, 100, 2));
        assert!(!stats[0].record(4, 100, 1));
        assert!(!stats[0].record(1, 100, 1));
        assert_eq!(stats[0], before);
        let mut overflow = DepthStats { depth: Some(1), cycles: u64::MAX, wall_ns: 1, emitted: 1 };
        let before = overflow;
        assert!(!overflow.record(1, 2, 2));
        assert_eq!(overflow, before);
    }

    #[test]
    fn cohorts_share_reads_but_preserve_every_row_rank() {
        let routes = [
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
            [9, 29, 30, 31, 32, 33, 34, 35, 36, 37],
        ];
        for width in 1..=41 {
            let plan = expert_cohorts(&routes, width).unwrap();
            assert_eq!(plan.cohorts.len(), 28);
            let mut reconstructed = [[u16::MAX; TOP_K]; MAX_ROWS];
            let mut previous_end = 0;
            for wave in &plan.waves {
                assert_eq!(wave.start, previous_end);
                assert!(!wave.is_empty() && wave.len() <= width);
                for group in &plan.cohorts[wave.clone()] {
                    for consumer in &group.consumers {
                        assert_eq!(reconstructed[consumer.row][consumer.rank], u16::MAX);
                        reconstructed[consumer.row][consumer.rank] = group.expert;
                    }
                }
                previous_end = wave.end;
            }
            assert_eq!(previous_end, plan.cohorts.len());
            assert_eq!(reconstructed, routes);
        }
        assert!(expert_cohorts(&routes, 0).is_none());
        assert!(expert_cohorts(&[], 1).is_none());
        assert!(expert_cohorts(&[[0; TOP_K]], 1).is_none());
        assert!(expert_cohorts(&[[512, 1, 2, 3, 4, 5, 6, 7, 8, 9]], 1).is_none());
        assert!(expert_cohorts(&[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]; 5], 1).is_none());
    }

    #[test]
    fn suffix_is_request_local_bounded_and_stops_at_eos() {
        let h = [1, 2, 8, 9, 10, 1, 2];
        assert_eq!(suffix_proposals(&h, 100, 4, 3, &[]), vec![8, 9, 10]);
        assert_eq!(suffix_proposals(&h, 100, 4, 3, &[9]), vec![8, 9]);
        assert_eq!(suffix_proposals(&h, 100, 4, 1, &[]), vec![8]);
        assert!(suffix_proposals(&h, 4, 4, 3, &[]).is_empty());
        assert!(suffix_proposals(&h, 100, 0, 3, &[]).is_empty());
        assert!(suffix_proposals(&h, 100, 4, 0, &[]).is_empty());
        assert_eq!(suffix_proposals(&[1, 7, 1], 100, 32, 3, &[]), vec![7]);
        assert!(suffix_proposals(&[1, 1], 100, 32, 3, &[]).is_empty());
        assert!(suffix_proposals(&[], usize::MAX, usize::MAX, usize::MAX, &[]).is_empty());
        // At equal suffix length prefer the most recent completed occurrence.
        assert_eq!(suffix_proposals(&[1, 2, 8, 1, 2, 9, 1, 2], 100, 2, 1, &[]), vec![9]);
    }

    #[test]
    fn suffix_matches_independent_completed_occurrence_search() {
        // Exhaustive binary histories up to ten tokens, not a second copy of
        // the descending search: enumerate all completed substrings and select
        // the maximum (length, latest start) satisfying the public contract.
        for len in 0usize..=10 {
            for bits in 0..1usize << len {
                let history: Vec<u32> = (0..len).map(|i| ((bits >> i) & 1) as u32).collect();
                for lookback in [0, 3, 6, 10] {
                    let tail = &history[len.saturating_sub(lookback)..];
                    let mut candidates = Vec::new();
                    for start in 0..tail.len() {
                        for end in start + 1..tail.len() {
                            let order = end - start;
                            if order <= 3
                                && end < tail.len() - order
                                && tail.ends_with(&tail[start..end])
                            {
                                candidates.push((order, start, end));
                            }
                        }
                    }
                    let expected = candidates.into_iter().max().map_or_else(Vec::new, |(order, _, end)| {
                        tail[end..tail.len() - order].iter().take(3).copied().collect()
                    });
                    assert_eq!(suffix_proposals(&history, lookback, 3, 3, &[]), expected);
                }
            }
        }
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(6)]
    fn memory_admission_never_exceeds_limit_or_budget() {
        let plan = MemoryPlan {
            base_bytes: kani::any(),
            reserve_bytes: kani::any(),
            draft_weights_bytes: kani::any(),
            draft_kv_bytes: kani::any(),
            draft_scratch_bytes: kani::any(),
            verify_scratch_bytes: kani::any(),
            prefix_state_bytes_per_row: kani::any(),
            retained_expert_generations_bytes: kani::any(),
        };
        let requested: usize = kani::any();
        let context: usize = kani::any();
        let output: usize = kani::any();
        let limit: u64 = kani::any();
        let result = plan.admitted_depth(requested, context, output, limit);
        if let Some(depth) = result {
            assert!(depth <= MAX_DEPTH && depth <= requested);
            assert!(depth < context && depth < output);
            assert!(plan.peak_bytes(depth).unwrap() <= limit);
        }
        kani::cover!(result == Some(0) && requested == 3);
        kani::cover!(result == Some(3));
        kani::cover!(result.is_none() && context > 0 && output > 0);
    }

    #[kani::proof]
    fn invalid_sample_preserves_statistics() {
        let mut stats = DepthStats { depth: kani::any(), cycles: kani::any(), wall_ns: kani::any(), emitted: kani::any() };
        let before = stats;
        let depth: usize = kani::any();
        let ns: u64 = kani::any();
        let emitted: usize = kani::any();
        let recorded = stats.record(depth, ns, emitted);
        if recorded {
            assert!(depth <= MAX_DEPTH && emitted > 0 && emitted <= depth + 1 && ns > 0);
            assert_eq!(stats.cycles, before.cycles + 1);
            assert_eq!(stats.wall_ns, before.wall_ns + ns);
            assert_eq!(stats.emitted, before.emitted + emitted as u64);
        } else {
            assert_eq!(stats, before);
        }
        kani::cover!(recorded);
        kani::cover!(!recorded && before.cycles == u64::MAX);
    }
}
