//! CPU policies; no claim about Metal numerical correctness or physical RAM.

pub const DEFAULT_PROMPT_CACHE_BYTES: usize = 256 * 1024 * 1024;

pub fn compact_wasteful_view(logical: usize, retained: usize) -> bool {
    logical > 0
        && retained > logical.saturating_mul(2)
        && retained.saturating_sub(logical) >= 65_536
}

pub fn capture_matches(target: &[u8], request: &[u8], already_captured: bool) -> bool {
    !already_captured && !target.is_empty() && target == request
}

pub fn mib_budget(mib: usize) -> Option<usize> {
    (mib <= 4096).then(|| mib.checked_mul(1024)?.checked_mul(1024))?
}

/// Canonical tuning tag: default is distinct from every explicit cache limit.
pub fn allocator_cache_key(limit_mib: Option<usize>) -> Option<usize> {
    match limit_mib {
        None => Some(0),
        Some(mib) => mib_budget(mib).map(|_| mib + 1),
    }
}

/// Count every retained component conservatively, including duplicate aliases.
pub fn cache_fits(components: [usize; 6], budget: usize) -> bool {
    components
        .into_iter()
        .try_fold(0usize, usize::checked_add)
        .is_some_and(|bytes| bytes <= budget)
}

/// Eight blocks between changes, 5% hysteresis, occasional eight-block probes.
/// Costs are nanoseconds per committed target input, never acceptance alone.
pub fn adaptive_depth(current: usize, costs: [u64; 4], dwell: usize, probe: bool) -> Option<usize> {
    if current > 3 {
        return None;
    }
    if dwell < 8 {
        return Some(current);
    }
    if let Some(missing) = costs.iter().position(|&cost| cost == 0) {
        return Some(missing);
    }
    if probe {
        return Some((current + 1) % 4);
    }
    let mut best = current;
    for (depth, &cost) in costs.iter().enumerate() {
        if u128::from(cost) * 100 < u128::from(costs[current]) * 95 && cost < costs[best] {
            best = depth;
        }
    }
    Some(best)
}

#[derive(Default)]
pub struct AdaptiveMtp {
    costs: [u64; 4],
    samples: [usize; 4],
    dwell: usize,
    since_probe: usize,
}

impl AdaptiveMtp {
    pub fn observe(&mut self, depth: usize, nanos: u64, committed: usize) {
        if depth > 3 || nanos == 0 || committed == 0 || committed > 4 {
            return;
        }
        let warm = self.samples[depth] >= 2;
        self.samples[depth] = self.samples[depth].saturating_add(1);
        self.dwell = self.dwell.saturating_add(1);
        self.since_probe = self.since_probe.saturating_add(1);
        if !warm {
            return;
        }
        let cost = (nanos / committed as u64).max(1);
        let previous = self.costs[depth];
        self.costs[depth] = if previous == 0 {
            cost
        } else {
            previous - previous / 4 + cost / 4
        };
    }

    pub fn next(&mut self, current: usize) -> usize {
        let next =
            adaptive_depth(current, self.costs, self.dwell, self.since_probe >= 32).unwrap_or(1);
        if next != current {
            self.dwell = 0;
            self.since_probe = 0;
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn allocator_tuning_keys_distinguish_default_and_all_valid_limits() {
        let mut keys = std::collections::HashSet::new();
        assert!(keys.insert(allocator_cache_key(None).unwrap()));
        for mib in 0..=4096 {
            assert!(keys.insert(allocator_cache_key(Some(mib)).unwrap()));
        }
        assert_eq!(keys.len(), 4098);
        for mib in [4097, usize::MAX] {
            assert_eq!(allocator_cache_key(Some(mib)), None);
        }
    }
    #[test]
    fn compact_view_requires_material_waste_without_overflow() {
        assert!(compact_wasteful_view(2, 65_538));
        assert!(!compact_wasteful_view(2, 65_537));
        assert!(!compact_wasteful_view(65_536, 131_072));
        assert!(compact_wasteful_view(65_536, 131_073));
        assert!(!compact_wasteful_view(0, usize::MAX));
        assert!(!compact_wasteful_view(usize::MAX, usize::MAX));
    }
    #[test]
    fn capture_selects_only_the_named_request_once() {
        assert!(capture_matches(b"profile", b"profile", false));
        assert!(!capture_matches(b"profile", b"warmup", false));
        assert!(!capture_matches(b"", b"", false));
        assert!(!capture_matches(b"profile", b"profile", true));
    }
    #[test]
    fn mib_limits_reject_invalid_and_overflowing_values() {
        for mib in [0, 1, 256, 512, 4096] {
            assert_eq!(mib_budget(mib), mib.checked_mul(1_048_576));
        }
        for mib in [4097, usize::MAX] {
            assert_eq!(mib_budget(mib), None);
        }
    }
    #[test]
    fn budget_includes_draft_logits_history_and_backing_without_overflow() {
        assert!(cache_fits([100, 20, 30, 40, 8, 2], 200));
        assert!(!cache_fits([100, 20, 30, 40, 8, 2], 199));
        assert!(cache_fits([0; 6], 0));
        assert!(!cache_fits([1, 0, 0, 0, 0, 0], 0));
        assert!(!cache_fits([usize::MAX, 1, 0, 0, 0, 0], usize::MAX));
    }
    #[test]
    fn adaptive_excludes_two_cold_samples_and_rejects_invalid_samples() {
        let mut policy = AdaptiveMtp::default();
        policy.observe(2, u64::MAX, 1);
        policy.observe(2, u64::MAX, 1);
        assert_eq!(policy.costs, [0; 4]);
        for (depth, nanos, committed) in [(4, 80, 1), (2, 0, 1), (2, 80, 0), (2, 80, 5)] {
            policy.observe(depth, nanos, committed);
        }
        assert_eq!(policy.samples, [0, 0, 2, 0]);
        assert_eq!(policy.dwell, 2);
        policy.observe(2, 80, 4);
        assert_eq!(policy.costs, [0, 0, 20, 0]);
    }
    #[test]
    fn adaptive_uses_costs_hysteresis_and_reprobes() {
        assert_eq!(adaptive_depth(2, [10, 12, 20, 30], 7, false), Some(2));
        assert_eq!(adaptive_depth(2, [10, 12, 20, 30], 8, false), Some(0));
        assert_eq!(adaptive_depth(2, [96, 98, 100, 99], 8, false), Some(2));
        assert_eq!(adaptive_depth(2, [0, 12, 20, 30], 8, false), Some(0));
        assert_eq!(adaptive_depth(3, [10, 12, 20, 30], 8, true), Some(0));
        assert_eq!(adaptive_depth(4, [1; 4], 8, false), None);
        let mut policy = AdaptiveMtp::default();
        for _ in 0..8 {
            policy.observe(2, 80, 4);
        }
        assert_eq!(policy.next(2), 0);
        for (n, count) in [(0, 1), (10, 0), (10, 5)] {
            policy.observe(0, n, count);
        }
        assert_eq!(policy.next(0), 0);
        for _ in 0..8 {
            policy.observe(0, u64::MAX, 1);
        }
        assert_eq!(policy.next(0), 1);
    }
}

#[cfg(kani)]
mod verification {
    use super::*;
    #[kani::proof]
    fn allocator_tuning_key_is_bounded_and_injective() {
        let a: Option<usize> = kani::any();
        let b: Option<usize> = kani::any();
        let ka = allocator_cache_key(a);
        let kb = allocator_cache_key(b);
        if let Some(key) = ka {
            assert!(key <= 4097);
            assert_eq!(key == 0, a.is_none());
            if ka == kb {
                assert_eq!(a, b);
            }
        } else {
            assert!(a.is_some_and(|mib| mib > 4096));
        }
        kani::cover!(a.is_none() && ka == Some(0));
        kani::cover!(a == Some(0) && ka == Some(1));
        kani::cover!(a == Some(4096) && ka == Some(4097));
        kani::cover!(ka.is_none());
    }
    #[kani::proof]
    fn compaction_requires_large_real_waste_without_overflow() {
        let logical: usize = kani::any();
        let retained: usize = kani::any();
        let compact = compact_wasteful_view(logical, retained);
        if compact {
            assert!(logical > 0);
            assert!(retained as u128 > logical as u128 * 2);
            assert!(retained - logical >= 65_536);
        }
        kani::cover!(compact && logical == 2);
        kani::cover!(!compact && logical == usize::MAX);
    }
    #[kani::proof]
    #[kani::unwind(10)]
    fn capture_selection_matches_bounded_symbolic_ids() {
        let a: [u8; 8] = kani::any();
        let b: [u8; 8] = kani::any();
        let n: usize = kani::any();
        let m: usize = kani::any();
        let already: bool = kani::any();
        if n > 8 || m > 8 {
            return;
        }
        let selected = capture_matches(&a[..n], &b[..m], already);
        if selected {
            assert!(!already && n > 0 && n == m);
            for i in 0..n {
                assert_eq!(a[i], b[i]);
            }
        }
        kani::cover!(selected && n == 8);
        kani::cover!(!selected && n == 0);
    }
    #[kani::proof]
    fn cache_mib_budget_is_bounded() {
        let mib: usize = kani::any();
        if let Some(bytes) = mib_budget(mib) {
            assert!(mib <= 4096);
            assert_eq!(bytes as u128, mib as u128 * 1_048_576);
            kani::cover!(mib == 0);
            kani::cover!(mib == 512);
        }
    }
    #[kani::proof]
    // Array equality reaches memcmp over 32 bytes on the 64-bit target.
    #[kani::unwind(34)]
    fn adaptive_observation_is_bounded_and_ignores_invalid_samples() {
        let costs: [u64; 4] = kani::any();
        let samples: [usize; 4] = kani::any();
        let dwell: usize = kani::any();
        let since_probe: usize = kani::any();
        let depth: usize = kani::any();
        let nanos: u64 = kani::any();
        let committed: usize = kani::any();
        let mut policy = AdaptiveMtp {
            costs,
            samples,
            dwell,
            since_probe,
        };
        policy.observe(depth, nanos, committed);
        if depth > 3 || nanos == 0 || committed == 0 || committed > 4 {
            assert_eq!(policy.costs, costs);
            assert_eq!(policy.samples, samples);
            assert_eq!(policy.dwell, dwell);
            assert_eq!(policy.since_probe, since_probe);
        } else {
            if samples[depth] >= 2 {
                assert!(policy.costs[depth] > 0);
            } else {
                assert_eq!(policy.costs[depth], costs[depth]);
            }
            assert!(policy.samples[depth] >= samples[depth]);
            for i in 0..4 {
                if i != depth {
                    assert_eq!(policy.costs[i], costs[i]);
                    assert_eq!(policy.samples[i], samples[i]);
                }
            }
            assert!(policy.dwell >= dwell && policy.since_probe >= since_probe);
            kani::cover!(costs[depth] == u64::MAX && nanos == u64::MAX);
        }
    }
    #[kani::proof]
    #[kani::unwind(34)]
    fn adaptive_next_is_bounded_and_resets_only_on_switch() {
        let costs: [u64; 4] = kani::any();
        let samples: [usize; 4] = kani::any();
        let dwell: usize = kani::any();
        let since_probe: usize = kani::any();
        let current: usize = kani::any();
        let mut policy = AdaptiveMtp {
            costs,
            samples,
            dwell,
            since_probe,
        };
        let next = policy.next(current);
        assert!(next <= 3);
        assert_eq!(policy.costs, costs);
        assert_eq!(policy.samples, samples);
        if next != current {
            assert_eq!(policy.dwell, 0);
            assert_eq!(policy.since_probe, 0);
        } else {
            assert_eq!(policy.dwell, dwell);
            assert_eq!(policy.since_probe, since_probe);
        }
        kani::cover!(current == 0 && next == 3);
        kani::cover!(current == 3 && next == 0);
    }
    #[kani::proof]
    #[kani::unwind(8)]
    fn cache_budget_never_accepts_an_overflow_or_excess() {
        let components: [usize; 6] = kani::any();
        let budget: usize = kani::any();
        let fits = cache_fits(components, budget);
        if fits {
            let total: u128 = components.into_iter().map(|x| x as u128).sum();
            assert!(total <= budget as u128);
        }
        kani::cover!(fits && budget > 0 && components[3] > 0);
        kani::cover!(!fits && components[0] == usize::MAX && components[1] > 0);
    }
    #[kani::proof]
    #[kani::unwind(6)]
    fn adaptive_choice_is_bounded_and_preserves_hysteresis() {
        let current: usize = kani::any();
        let costs: [u64; 4] = kani::any();
        let dwell: usize = kani::any();
        let probe: bool = kani::any();
        match adaptive_depth(current, costs, dwell, probe) {
            None => assert!(current > 3),
            Some(next) => {
                assert!(next <= 3);
                if dwell < 8 {
                    assert_eq!(next, current);
                }
                if dwell >= 8 && !probe && costs.iter().all(|&x| x > 0) && next != current {
                    assert!(u128::from(costs[next]) * 100 < u128::from(costs[current]) * 95);
                }
                kani::cover!(current == 0 && next == 3);
                kani::cover!(current == 3 && next == 0);
            }
        }
    }
}
