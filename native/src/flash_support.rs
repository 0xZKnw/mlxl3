//! CPU contracts for Flash Next's disk rows and exact expert cache.
use anyhow::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheError {
    Capacity,
    InvalidIds,
    Clock,
    NoSlot,
}
impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Capacity => "expert cohort exceeds cache capacity (10..512 slots required)",
            Self::InvalidIds => "invalid expert cohort",
            Self::Clock => "expert cache clock exhausted",
            Self::NoSlot => "no unprotected expert slot",
        })
    }
}
impl std::error::Error for CacheError {}

pub const EXPERT_BYTES: u64 = 1_248_012;
pub const NGRAM_DIM: usize = 160;
pub const NGRAM_WORDS: usize = 41;

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn slots_fit(ids: &[u32], capacity: i32) -> bool {
    !ids.is_empty()
        && ids.len() <= i32::MAX as usize
        && capacity > 0
        && ids.iter().all(|&id| id < capacity as u32)
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn parse_lru_override(value: &str) -> Option<bool> {
    match value {
        "auto" => Some(false),
        "lru" => Some(true),
        _ => None,
    }
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn parse_predictor_half(value: &str) -> Option<bool> {
    match value {
        "f32" => Some(false),
        "f16" => Some(true),
        _ => None,
    }
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn disk_destination_fits(start: usize, length: usize, capacity: usize) -> bool {
    length > 0 && start.checked_add(length).is_some_and(|end| end <= capacity)
}

/// Reserve resident weights, MTP, scratch buffers and room for other applications.
pub fn automatic_cache_slots(available_bytes: u64) -> Option<usize> {
    let budget = available_bytes.checked_sub(8 << 30)?.min(6 << 30);
    let slots = budget / (48 * EXPERT_BYTES);
    (slots >= 10).then_some(slots as usize)
}

#[cfg(target_os = "macos")]
fn available_memory_response_complete(count: libc::mach_msg_type_number_t) -> bool {
    // Older kernels return shorter revisions; only require fields we read.
    let required = (std::mem::offset_of!(libc::vm_statistics64, external_page_count)
        + std::mem::size_of::<libc::natural_t>())
        / std::mem::size_of::<libc::integer_t>();
    (required..=libc::HOST_VM_INFO64_COUNT as usize).contains(&(count as usize))
}

#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc already provides the Mach ABI; no extra dependency.
pub fn available_memory_bytes() -> Result<u64> {
    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    let mut info: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let host = unsafe { libc::mach_host_self() };
    let status = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&mut info as *mut libc::vm_statistics64).cast(),
            &mut count,
        )
    };
    let released = unsafe { mach_port_deallocate(libc::mach_task_self(), host) };
    anyhow::ensure!(
        status == 0 && available_memory_response_complete(count) && released == 0,
        "cannot query available system memory ({status}/{released}, {count} words)"
    );
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    anyhow::ensure!(page_size > 0, "invalid system page size");
    // Speculative file pages are included in both free_count and external_count.
    let pages = u64::from(info.free_count.saturating_sub(info.speculative_count))
        + u64::from(info.external_page_count);
    pages
        .checked_mul(page_size as u64)
        .ok_or_else(|| anyhow::anyhow!("system memory size overflow"))
}

/// Match PonyExl3's single-row accumulation order, including small experts.
pub fn reference_splits(input_tiles: u32, output_tiles: u32) -> u32 {
    let mut splits = 1u32;
    while u64::from(output_tiles) * u64::from(splits) < 8192
        && u64::from(input_tiles) / (u64::from(splits) * 2) >= 128
    {
        splits *= 2;
    }
    splits
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn pair_commit_offset(base: i32, current: i32, retained: i32) -> Option<i32> {
    if base < 0 || !(1..=2).contains(&retained) || base.checked_add(2)? != current {
        return None;
    }
    base.checked_add(retained)
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn expert_cohort_rows(capacity: usize) -> Option<usize> {
    (10..=512)
        .contains(&capacity)
        .then_some((capacity / 10).min(8))
}

/// Greedy-only scheduler: prefer the smallest prefix on equal expected throughput.
#[cfg(any(test, kani))]
pub(crate) fn draft_prefix(confidence: &[f64], costs: &[f64]) -> Option<usize> {
    if confidence.len() > 3
        || costs.len() != confidence.len() + 1
        || confidence
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || costs
            .iter()
            .any(|c| !c.is_finite() || !(1e-9..=360.0).contains(c))
    {
        return None;
    }
    let mut survival = 1.;
    let mut expected = 1.;
    let mut best_expected = 1.;
    let mut best = 0;
    for (position, &probability) in confidence.iter().enumerate() {
        survival *= probability;
        expected += survival;
        if expected * costs[best] > best_expected * costs[position + 1] {
            best = position + 1;
            best_expected = expected;
        }
    }
    Some(best)
}

#[cfg(test)]
mod draft_contract_tests {
    use super::*;

    #[test]
    fn draft_prefix_matches_exhaustive_quarter_probability_outcomes() {
        let cost_tables = [
            [1., 2., 3., 4.],
            [1., 1., 1., 1.],
            [1., 1.5, 2., 4.],
            [1., 3., 1.5, 8.],
        ];
        for p0 in 0..=4 {
            for p1 in 0..=4 {
                for p2 in 0..=4 {
                    let probabilities = [p0 as f64 / 4., p1 as f64 / 4., p2 as f64 / 4.];
                    for length in 0..=3 {
                        let mut emitted = [0u32; 4];
                        for draw0 in 0..4 {
                            for draw1 in 0..4 {
                                for draw2 in 0..4 {
                                    let draws = [draw0, draw1, draw2];
                                    for (prefix, total) in
                                        emitted.iter_mut().enumerate().take(length + 1)
                                    {
                                        let accepted = draws[..prefix]
                                            .iter()
                                            .zip([p0, p1, p2])
                                            .take_while(|(draw, p)| **draw < *p)
                                            .count();
                                        *total += 1 + accepted as u32;
                                    }
                                }
                            }
                        }
                        for costs in cost_tables {
                            let mut expected = 0;
                            for j in 1..=length {
                                if emitted[j] as f64 * costs[expected]
                                    > emitted[expected] as f64 * costs[j]
                                {
                                    expected = j;
                                }
                            }
                            assert_eq!(
                                draft_prefix(&probabilities[..length], &costs[..=length]),
                                Some(expected)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn draft_prefix_rejects_invalid_profiles_and_prefers_shorter_ties() {
        assert_eq!(draft_prefix(&[], &[0.1]), Some(0));
        assert_eq!(draft_prefix(&[0., 0., 0.], &[1.; 4]), Some(0));
        assert_eq!(draft_prefix(&[1., 1., 1.], &[1.; 4]), Some(3));
        assert_eq!(draft_prefix(&[1.; 4], &[1.; 5]), None);
        assert_eq!(draft_prefix(&[1.], &[1.]), None);
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1., 1.1] {
            assert_eq!(draft_prefix(&[invalid], &[1.; 2]), None);
        }
        for invalid in [f64::NAN, f64::INFINITY, 0., -1., 361., 1e-12] {
            assert_eq!(draft_prefix(&[0.5], &[1., invalid]), None);
        }
    }

    #[test]
    fn expert_cohorts_fit_even_the_smallest_streaming_cache() {
        for capacity in 0..=1024 {
            match expert_cohort_rows(capacity) {
                Some(rows) => {
                    assert!((10..=512).contains(&capacity));
                    assert!((1..=8).contains(&rows));
                    assert!(rows * 10 <= capacity);
                    assert!(rows == 8 || (rows + 1) * 10 > capacity);
                }
                None => assert!(!(10..=512).contains(&capacity)),
            }
        }
        assert_eq!(expert_cohort_rows(usize::MAX), None);
    }

    #[test]
    fn draft_commits_match_independent_wide_arithmetic_and_reject_stale_tapes() {
        for base in [-1, 0, 1, 4096, i32::MAX - 8, i32::MAX - 1, i32::MAX] {
            for total in (0..=10).chain([usize::MAX]) {
                for retained in (0..=10).chain([usize::MAX]) {
                    for current in [
                        i32::MIN,
                        0,
                        base.saturating_add(total.min(9) as i32),
                        i32::MAX,
                    ] {
                        let valid = base >= 0
                            && (1..=8).contains(&total)
                            && (1..=total).contains(&retained)
                            && i64::from(base) + total as i64 == i64::from(current);
                        let expected = valid.then(|| (i64::from(base) + retained as i64) as i32);
                        assert_eq!(
                            draft_commit_offset(base, current, total, retained),
                            expected
                        );
                    }
                }
            }
        }
    }
}

#[cfg(kani)]
mod draft_contract_proofs {
    use super::*;

    #[kani::proof]
    #[kani::unwind(5)]
    fn draft_scheduler_cannot_exceed_available_prefix_or_accept_invalid_costs() {
        let confidence = kani::any::<[f64; 3]>();
        let costs = kani::any::<[f64; 4]>();
        let length = kani::any::<usize>();
        kani::assume(length <= 3);
        if let Some(prefix) = draft_prefix(&confidence[..length], &costs[..=length]) {
            assert!(prefix <= length);
            assert!(
                costs[..=length]
                    .iter()
                    .all(|c| c.is_finite() && (1e-9..=360.).contains(c))
            );
            assert!(
                confidence[..length]
                    .iter()
                    .all(|p| p.is_finite() && (0.0..=1.).contains(p))
            );
            kani::cover!(length == 3 && prefix == 0);
            kani::cover!(length == 3 && prefix == 3);
        }
    }

    #[kani::proof]
    fn expert_cohorts_never_exceed_live_slot_capacity() {
        let capacity: usize = kani::any();
        if let Some(rows) = expert_cohort_rows(capacity) {
            assert!((10..=512).contains(&capacity));
            assert!((1..=8).contains(&rows));
            assert!(rows * 10 <= capacity);
            assert!(rows == 8 || (rows + 1) * 10 > capacity);
            kani::cover!(rows == 1);
            kani::cover!(rows == 7);
            kani::cover!(rows == 8);
        } else {
            assert!(!(10..=512).contains(&capacity));
        }
    }

    #[kani::proof]
    fn wide_draft_commit_accepts_only_exact_finite_prefixes() {
        let base: i32 = kani::any();
        let current: i32 = kani::any();
        let total: usize = kani::any();
        let retained: usize = kani::any();
        let result = draft_commit_offset(base, current, total, retained);
        let valid = base >= 0
            && (1..=8).contains(&total)
            && (1..=total).contains(&retained)
            && i64::from(base) + total as i64 == i64::from(current);
        assert_eq!(result.is_some(), valid);
        if let Some(end) = result {
            assert_eq!(i64::from(end), i64::from(base) + retained as i64);
            assert!(end <= current);
            kani::cover!(total == 8 && retained == 1);
            kani::cover!(total == 8 && retained == 8);
            kani::cover!(total == 4 && retained == 3);
        }
    }
}

#[cfg(any(test, kani))]
pub(crate) fn draft_commit_offset(
    base: i32,
    current: i32,
    total: usize,
    retained: usize,
) -> Option<i32> {
    if base < 0
        || !(1..=8).contains(&total)
        || !(1..=total).contains(&retained)
        || base.checked_add(total as i32)? != current
    {
        return None;
    }
    base.checked_add(retained as i32)
}

pub fn row_offset(base: u64, row: u64, rows: u64, bytes: u64) -> Option<u64> {
    if row >= rows || bytes == 0 {
        return None;
    }
    let start = base.checked_add(row.checked_mul(bytes)?)?;
    let end = start.checked_add(bytes)?;
    (end <= i64::MAX as u64).then_some(start)
}

/// The reference multiplies signed int64 tokens with wraparound, then uses
/// Euclidean remainder. EOS padding prevents n-grams crossing a segment.
pub fn ngram_row(
    tokens: [u32; 3],
    multipliers: [u64; 3],
    n: usize,
    offset: u64,
    size: u64,
) -> Option<u64> {
    if !(2..=3).contains(&n) || size == 0 || size > i64::MAX as u64 {
        return None;
    }
    let mut bits = 0u64;
    for i in 0..n {
        bits ^= u64::from(tokens[i]).wrapping_mul(multipliers[i]);
    }
    offset.checked_add((bits as i64).rem_euclid(size as i64) as u64)
}

pub fn segment_tokens(current: u32, previous: [u32; 2], eos: u32) -> [u32; 3] {
    [
        current,
        previous[0],
        if previous[0] == eos { eos } else { previous[1] },
    ]
}

/// Reservation is atomic: every routed expert gets a distinct protected slot.
/// A failed read must invalidate its reservation before it can become a hit.
pub struct ExpertCache {
    keys: Vec<Option<u32>>,
    pending: Vec<Option<u32>>,
    prefetched: Vec<bool>,
    predicted: [bool; 512],
    predictive_lru: bool,
    counts: [u32; 512],
    ages: Vec<u64>,
    clock: u64,
    pub prefetch_hits: u64,
}

impl ExpertCache {
    pub fn new(slots: usize) -> Result<Self, CacheError> {
        if !(10..=512).contains(&slots) {
            return Err(CacheError::Capacity);
        }
        Ok(Self {
            keys: vec![None; slots],
            pending: vec![None; slots],
            prefetched: vec![false; slots],
            predicted: [false; 512],
            predictive_lru: false,
            counts: [0; 512],
            ages: vec![0; slots],
            clock: 0,
            prefetch_hits: 0,
        })
    }

    pub fn reserve(&mut self, ids: &[u32]) -> Result<Vec<(usize, bool)>, CacheError> {
        if ids.is_empty() || ids.iter().any(|&id| id >= 512) {
            return Err(CacheError::InvalidIds);
        }
        let mut unique = Vec::with_capacity(ids.len().min(512));
        for &id in ids {
            let position = unique
                .iter()
                .position(|&key| key >= id)
                .unwrap_or(unique.len());
            if unique.get(position) != Some(&id) {
                unique.insert(position, id);
            }
        }
        if unique.len() > self.pending.iter().filter(|p| p.is_none()).count() {
            return Err(CacheError::Capacity);
        }
        self.clock = self.clock.checked_add(1).ok_or(CacheError::Clock)?;
        if self.clock.is_multiple_of(16) {
            for count in &mut self.counts {
                *count /= 2;
            }
        }
        let mut reserved = Vec::with_capacity(unique.len());
        for &id in &unique {
            let present = self.keys.iter().position(|&key| key == Some(id));
            let slot = match present {
                Some(slot) => slot,
                None => self.victim(&unique).ok_or(CacheError::NoSlot)?,
            };
            if present.is_some() && self.prefetched[slot] {
                self.prefetch_hits = self.prefetch_hits.saturating_add(1);
            }
            self.prefetched[slot] = false;
            self.keys[slot] = Some(id);
            self.counts[id as usize] = self.counts[id as usize].saturating_add(1);
            self.ages[slot] = self.clock;
            reserved.push((id, slot, present.is_some()));
        }
        Ok(ids
            .iter()
            .map(|id| {
                let &(_, slot, hit) = reserved
                    .iter()
                    .find(|&&(key, _, _)| key == *id)
                    .expect("reserved expert");
                (slot, hit)
            })
            .collect())
    }

    pub fn contains(&self, id: u32) -> bool {
        self.keys.contains(&Some(id))
    }

    fn victim(&self, routed: &[u32]) -> Option<usize> {
        (0..self.keys.len())
            .filter(|&i| {
                self.pending[i].is_none() && self.keys[i].is_none_or(|key| !routed.contains(&key))
            })
            .min_by_key(|&i| {
                (
                    u8::from(self.keys[i].is_some()),
                    u8::from(self.keys[i].is_some_and(|key| self.predicted[key as usize])),
                    if self.predictive_lru {
                        0
                    } else {
                        self.keys[i].map_or(0, |key| self.counts[key as usize])
                    },
                    self.ages[i],
                    i,
                )
            })
    }

    pub(crate) fn enable_predictive_lru(&mut self) {
        // Unused prefetches need a lifetime without inventing demand frequency.
        self.predictive_lru = true;
    }

    pub fn protect_predictions(&mut self, ids: &[u32]) -> Result<(), CacheError> {
        if ids.len() > 32 || ids.iter().any(|&id| id >= 512) {
            return Err(CacheError::InvalidIds);
        }
        self.predicted.fill(false);
        self.enable_predictive_lru();
        for &id in ids {
            self.predicted[id as usize] = true;
        }
        Ok(())
    }

    /// Leave enough non-pending slots for the next actual one/two-row cohort.
    pub fn claim_prefetch(&mut self, id: u32) -> Result<Option<usize>, CacheError> {
        if id >= 512 {
            return Err(CacheError::InvalidIds);
        }
        let pending = self.pending.iter().filter(|p| p.is_some()).count();
        let cohort = if self.capacity() >= 20 { 20 } else { 10 };
        if self.contains(id)
            || self.pending.contains(&Some(id))
            || self.capacity() - pending <= cohort
        {
            return Ok(None);
        }
        let slot = self.victim(&[]).ok_or(CacheError::NoSlot)?;
        self.keys[slot] = None;
        self.prefetched[slot] = false;
        self.pending[slot] = Some(id);
        Ok(Some(slot))
    }

    /// Only the inference thread publishes a completed write. Demand may have won.
    pub fn publish_prefetch(&mut self, slot: usize, id: u32, valid: bool) -> bool {
        if self.pending.get(slot) != Some(&Some(id)) {
            return false;
        }
        self.pending[slot] = None;
        if !valid || self.contains(id) {
            return false;
        }
        let Some(clock) = self.clock.checked_add(1) else {
            return false;
        };
        self.clock = clock;
        self.keys[slot] = Some(id);
        self.prefetched[slot] = true;
        self.ages[slot] = self.clock;
        true
    }

    pub fn invalidate(&mut self) {
        self.keys.fill(None);
        self.prefetched.fill(false);
    }
    pub fn capacity(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn slot_validation_and_les_controls_reject_invalid_inputs() {
        for capacity in [-1, 0, 1, 3, 10, 20, 89, 512, i32::MAX] {
            assert!(!slots_fit(&[], capacity));
            for ids in [vec![0], vec![1, 1, 0], vec![88, 89], vec![u32::MAX]] {
                let expected =
                    capacity > 0 && ids.iter().all(|&id| i64::from(id) < i64::from(capacity));
                assert_eq!(slots_fit(&ids, capacity), expected);
            }
        }
        assert_eq!(parse_lru_override("auto"), Some(false));
        assert_eq!(parse_lru_override("lru"), Some(true));
        assert_eq!(parse_predictor_half("f32"), Some(false));
        assert_eq!(parse_predictor_half("f16"), Some(true));
        for invalid in ["", "0", "1", "LFU", "LRU", " lru", "f32\n", "f64", "é"] {
            assert_eq!(parse_lru_override(invalid), None);
            assert_eq!(parse_predictor_half(invalid), None);
        }
    }

    #[test]
    fn disk_ranges_reject_empty_overflow_and_out_of_bounds() {
        assert!(disk_destination_fits(0, 1, 1));
        assert!(disk_destination_fits(7, 9, 16));
        assert!(disk_destination_fits(usize::MAX - 1, 1, usize::MAX));
        assert!(!disk_destination_fits(0, 0, 16));
        assert!(!disk_destination_fits(7, 10, 16));
        assert!(!disk_destination_fits(usize::MAX, 1, usize::MAX));
    }
    #[test]
    fn automatic_cache_reserves_headroom_and_stays_bounded() {
        let minimum = (8 << 30) + 10 * 48 * EXPERT_BYTES;
        for bytes in [0, 1, (8 << 30) - 1, 8 << 30, minimum - 1] {
            assert_eq!(automatic_cache_slots(bytes), None);
        }
        assert_eq!(automatic_cache_slots(minimum), Some(10));
        assert_eq!(automatic_cache_slots(12 << 30), Some(71));
        assert_eq!(automatic_cache_slots(14 << 30), Some(107));
        assert_eq!(automatic_cache_slots(u64::MAX), Some(107));
        for slots in 10..=107 {
            let threshold = (8 << 30) + slots * 48 * EXPERT_BYTES;
            assert_eq!(automatic_cache_slots(threshold), Some(slots as usize));
            assert_eq!(automatic_cache_slots(threshold + 1), Some(slots as usize));
            assert_eq!(
                automatic_cache_slots(threshold - 1),
                if slots == 10 {
                    None
                } else {
                    Some(slots as usize - 1)
                }
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn available_memory_accepts_older_complete_revisions() {
        // Apple's HOST_VM_INFO64_REV1_COUNT includes all fields we read.
        let rev1 = std::mem::offset_of!(libc::vm_statistics64, swapped_count)
            / std::mem::size_of::<libc::integer_t>();
        assert!(rev1 < libc::HOST_VM_INFO64_COUNT as usize);
        assert!(available_memory_response_complete(rev1 as u32));
        assert!(available_memory_response_complete(
            libc::HOST_VM_INFO64_COUNT
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn available_memory_rejects_truncated_and_oversized_snapshots() {
        let mut info: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        let base = (&mut info as *mut libc::vm_statistics64) as usize;
        let field_end = (&info.external_page_count as *const libc::natural_t) as usize
            + std::mem::size_of_val(&info.external_page_count);
        let minimum_words = (field_end - base) / std::mem::size_of::<libc::integer_t>();
        assert_eq!(
            (field_end - base) % std::mem::size_of::<libc::integer_t>(),
            0
        );
        for count in 0..=libc::HOST_VM_INFO64_COUNT + 1 {
            assert_eq!(
                available_memory_response_complete(count),
                (minimum_words..=libc::HOST_VM_INFO64_COUNT as usize).contains(&(count as usize)),
                "count={count}"
            );
        }
        assert!(!available_memory_response_complete(u32::MAX));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn available_memory_query_returns_a_positive_snapshot() {
        assert!(available_memory_bytes().unwrap() > 0);
    }
    #[test]
    fn pair_commits_reject_stale_invalid_and_overflowing_positions() {
        for base in [
            i32::MIN,
            -1,
            0,
            1,
            2047,
            i32::MAX - 2,
            i32::MAX - 1,
            i32::MAX,
        ] {
            for current in [i32::MIN, 0, 2, 3, 2049, i32::MAX] {
                for retained in [-1, 0, 1, 2, 3, i32::MAX] {
                    let valid = base >= 0
                        && (1..=2).contains(&retained)
                        && i64::from(base) + 2 == i64::from(current);
                    let expected = valid.then(|| (i64::from(base) + i64::from(retained)) as i32);
                    assert_eq!(pair_commit_offset(base, current, retained), expected);
                }
            }
        }
    }
    #[test]
    fn reference_partition_covers_real_shapes_and_extremes() {
        for (input, output, expected) in [
            (160, 40, 1),
            (40, 160, 1),
            (384, 160, 2),
            (768, 4, 4),
            (160, 15520, 1),
            (0, 0, 1),
            (u32::MAX, u32::MAX, 1),
            (u32::MAX, 0, 1 << 24),
        ] {
            assert_eq!(reference_splits(input, output), expected);
        }
        for input in 0..4096 {
            for output in [0, 1, 4, 40, 160, 15520, u32::MAX] {
                let splits = reference_splits(input, output);
                assert!(splits.is_power_of_two() && splits <= (input / 128).max(1));
                assert!(
                    u64::from(output) * u64::from(splits) >= 8192
                        || u64::from(input) / (2 * u64::from(splits)) < 128
                );
            }
        }
    }
    #[test]
    fn disk_offsets_reject_boundaries_and_overflow() {
        assert_eq!(row_offset(8, 2, 3, 82), Some(172));
        assert_eq!(row_offset(8, 3, 3, 82), None);
        assert_eq!(row_offset(0, 0, 1, 0), None);
        assert_eq!(row_offset(i64::MAX as u64, 0, 1, 1), None);
        assert_eq!(row_offset(0, u64::MAX - 1, u64::MAX, 82), None);
    }
    #[test]
    fn hashes_use_signed_remainder_and_eos_segments() {
        assert_eq!(ngram_row([1, 2, 3], [u64::MAX, 2, 3], 2, 100, 7), Some(102));
        assert_eq!(segment_tokens(4, [9, 5], 9), [4, 9, 9]);
        assert_eq!(segment_tokens(9, [4, 5], 9), [9, 4, 5]);
        assert_eq!(ngram_row([0; 3], [0; 3], 1, 0, 7), None);
        assert_eq!(ngram_row([0; 3], [0; 3], 3, 0, 0), None);
    }
    #[test]
    fn complete_cohorts_protect_hits_and_have_unique_slots() -> Result<()> {
        let mut cache = ExpertCache::new(10)?;
        let ids = (0..10).collect::<Vec<_>>();
        let first = cache.reserve(&ids)?;
        assert!(first.iter().all(|&(_, hit)| !hit));
        assert_eq!(
            cache.reserve(&ids)?,
            first
                .iter()
                .map(|&(slot, _)| (slot, true))
                .collect::<Vec<_>>()
        );
        let saved = cache.keys.clone();
        assert!(cache.reserve(&(0..11).collect::<Vec<_>>()).is_err());
        assert_eq!(cache.keys, saved);
        let next = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 511];
        let selected = cache.reserve(&next)?;
        for (i, &id) in next.iter().enumerate() {
            assert_eq!(cache.keys[selected[i].0], Some(id));
            for (j, &other) in next.iter().enumerate() {
                assert_eq!(id == other, selected[i].0 == selected[j].0);
            }
        }
        cache.invalidate();
        assert!(cache.reserve(&ids)?.iter().all(|&(_, hit)| !hit));
        assert!(cache.reserve(&[512]).is_err());
        Ok(())
    }

    #[test]
    fn pending_records_never_hit_alias_or_publish_over_a_demand() -> Result<()> {
        let mut cache = ExpertCache::new(24)?;
        let slot = cache.claim_prefetch(511)?.unwrap();
        assert!(!cache.contains(511));
        assert_eq!(cache.claim_prefetch(511)?, None);
        cache.invalidate();
        let routes = cache.reserve(&[511, 511, 1])?;
        assert!(routes.iter().all(|&(s, hit)| s != slot && !hit));
        assert!(!cache.publish_prefetch(slot, 510, true));
        assert_eq!(cache.pending[slot], Some(511));
        assert!(!cache.publish_prefetch(slot, 511, true));
        assert_eq!(cache.keys.iter().filter(|&&k| k == Some(511)).count(), 1);
        let other = cache.claim_prefetch(510)?.unwrap();
        assert!(!cache.publish_prefetch(other, 510, false));
        assert!(!cache.contains(510));
        let other = cache.claim_prefetch(510)?.unwrap();
        assert!(cache.publish_prefetch(other, 510, true));
        assert!(cache.contains(510));
        assert_eq!(cache.reserve(&[510, 510])?, [(other, true); 2]);
        assert_eq!(cache.prefetch_hits, 1);
        assert!(cache.reserve(&[510])?[0].1);
        assert_eq!(cache.prefetch_hits, 1);
        assert!(!cache.publish_prefetch(usize::MAX, 0, true));
        assert!(cache.claim_prefetch(512).is_err());
        Ok(())
    }

    #[test]
    fn predictions_protect_victims_without_inventing_frequency_or_blocking_demand() -> Result<()> {
        let mut cache = ExpertCache::new(10)?;
        cache.reserve(&(0..10).collect::<Vec<_>>())?;
        cache.protect_predictions(&[0])?;
        let counts = cache.counts;
        let clock = cache.clock;
        assert!(cache.protect_predictions(&[512]).is_err());
        assert!(cache.protect_predictions(&[0; 33]).is_err());
        assert_eq!(cache.counts, counts);
        assert_eq!(cache.clock, clock);
        assert_eq!(cache.reserve(&[511])?, [(1, false)]);
        assert!(cache.contains(0));
        cache.protect_predictions(&(0..10).chain([511]).collect::<Vec<_>>())?;
        assert!(cache.reserve(&[510])?.iter().all(|&(_, hit)| !hit));
        // All slots protected still permits a real route through the bounded fallback.
        assert!(cache.contains(510));
        Ok(())
    }

    #[test]
    fn predictive_victims_replay_symbolic_case_and_all_protection_masks() -> Result<()> {
        let mut cache = ExpertCache::new(10)?;
        let values = [
            (7, 3, true, true),
            (7, 3, true, false),
            (7, 2, false, true),
            (7, 2, true, true),
            (7, 2, true, true),
            (7, 3, false, true),
            (2, 3, false, true),
            (7, 1, true, true),
            (2, 2, false, true),
            (2, 3, false, false),
        ];
        for (i, &(count, age, predicted, pending)) in values.iter().enumerate() {
            cache.keys[i] = Some(i as u32);
            cache.counts[i] = count;
            cache.ages[i] = age;
            cache.predicted[i] = predicted;
            cache.pending[i] = pending.then_some(511);
        }
        assert_eq!(cache.victim(&[5, 4]), Some(9));
        for mask in 0..1024_u16 {
            for i in 0..10 {
                cache.predicted[i] = mask & (1 << i) != 0;
                cache.pending[i] = None;
                cache.counts[i] = if i % 2 == 0 { u32::MAX } else { 0 };
                cache.ages[i] = if i % 3 == 0 { u64::MAX } else { 0 };
            }
            let slot = cache.victim(&[0, 9]).unwrap();
            assert!((1..9).contains(&slot));
            assert_eq!(cache.predicted[slot], (1..9).all(|i| cache.predicted[i]));
        }
        Ok(())
    }

    #[test]
    fn les_admission_retains_recent_unused_records_without_inventing_frequency() -> Result<()> {
        for capacity in [24, 89] {
            for demanded in [false, true] {
                let mut cache = ExpertCache::new(capacity)?;
                for first in (0..capacity).step_by(10) {
                    let ids = (0..10)
                        .map(|i| ((first + i) % capacity) as u32)
                        .collect::<Vec<_>>();
                    cache.reserve(&ids)?;
                }
                assert!(!cache.predictive_lru);
                assert!(cache.protect_predictions(&[512]).is_err());
                assert!(!cache.predictive_lru);
                cache.protect_predictions(&[511])?;
                let first = cache.claim_prefetch(511)?.unwrap();
                let clock = cache.clock;
                assert!(cache.publish_prefetch(first, 511, true));
                assert_eq!(cache.counts[511], 0);
                assert_eq!(cache.ages[first], clock + 1);
                if demanded {
                    assert!(cache.reserve(&[511])?[0].1);
                }
                cache.protect_predictions(&[510])?;
                let second = cache.claim_prefetch(510)?.unwrap();
                assert_ne!(first, second, "new prefetch evicted before older records");
                assert!(cache.publish_prefetch(second, 510, true));
                assert!(cache.contains(511));
                assert_eq!(cache.counts[511], u32::from(demanded));
            }
        }
        let mut cache = ExpertCache::new(10)?;
        cache.reserve(&(0..10).collect::<Vec<_>>())?;
        for _ in 0..3 {
            cache.reserve(&[0])?;
        }
        cache.reserve(&(1..10).collect::<Vec<_>>())?;
        assert_eq!(cache.victim(&[]), Some(1), "ordinary cache must retain LFU");
        cache.protect_predictions(&[])?;
        assert_eq!(cache.victim(&[]), Some(0), "LES window must use recency");
        Ok(())
    }

    #[test]
    fn publication_clock_exhaustion_frees_pending_without_a_false_hit() -> Result<()> {
        let mut cache = ExpertCache::new(24)?;
        cache.protect_predictions(&[511])?;
        let slot = cache.claim_prefetch(511)?.unwrap();
        cache.clock = u64::MAX;
        assert!(!cache.publish_prefetch(slot, 511, true));
        assert!(!cache.contains(511));
        assert_eq!(cache.pending[slot], None);
        assert_eq!(cache.clock, u64::MAX);
        assert_eq!(cache.reserve(&[511]), Err(CacheError::Clock));
        Ok(())
    }

    #[test]
    fn free_slots_precede_evictions_through_failed_and_duplicate_publications() -> Result<()> {
        #[derive(Clone, Copy, PartialEq)]
        enum Slot {
            Empty,
            Ready(u32),
            Pending(u32),
        }
        for capacity in [24, 89] {
            let mut cache = ExpertCache::new(capacity)?;
            let mut slots = vec![Slot::Empty; capacity];
            let mut seed = 39_u64;
            let mut random = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            for step in 0..10_000 {
                match random() % 5 {
                    0 | 1 => {
                        let ids = (0..10).map(|_| (random() % 128) as u32).collect::<Vec<_>>();
                        let routes = cache.reserve(&ids)?;
                        for (&id, &(slot, hit)) in ids.iter().zip(&routes) {
                            if hit {
                                assert!(slots[slot] == Slot::Ready(id));
                            }
                            assert!(!matches!(slots[slot], Slot::Pending(_)));
                            slots[slot] = Slot::Ready(id);
                        }
                    }
                    2 => {
                        let ids = (0..16).map(|_| (random() % 128) as u32).collect::<Vec<_>>();
                        cache.protect_predictions(&ids)?;
                    }
                    3 if slots
                        .iter()
                        .filter(|s| matches!(s, Slot::Pending(_)))
                        .count()
                        < 4 =>
                    {
                        let id = (random() % 128) as u32;
                        let free = slots.contains(&Slot::Empty);
                        if let Some(slot) = cache.claim_prefetch(id)? {
                            assert!(
                                !(free && matches!(slots[slot], Slot::Ready(_))),
                                "free slot ignored: capacity={capacity}, step={step}"
                            );
                            assert!(!matches!(slots[slot], Slot::Pending(_)));
                            slots[slot] = Slot::Pending(id);
                        }
                    }
                    4 => {
                        let choices = slots
                            .iter()
                            .enumerate()
                            .filter_map(|(i, s)| matches!(s, Slot::Pending(_)).then_some(i))
                            .collect::<Vec<_>>();
                        if !choices.is_empty() {
                            let slot = choices[random() as usize % choices.len()];
                            let Slot::Pending(id) = slots[slot] else {
                                unreachable!()
                            };
                            let valid = random() % 4 != 0;
                            let duplicate = slots.contains(&Slot::Ready(id));
                            let published = cache.publish_prefetch(slot, id, valid);
                            assert_eq!(published, valid && !duplicate);
                            slots[slot] = if published {
                                Slot::Ready(id)
                            } else {
                                Slot::Empty
                            };
                        }
                    }
                    _ => {}
                }
                for slot in &slots {
                    if let Slot::Ready(id) = slot {
                        assert!(cache.contains(*id));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn speculative_admission_preserves_actual_cohort_space_and_failed_reservations() -> Result<()> {
        for capacity in [10, 11, 19, 20, 21, 24, 89, 512] {
            let mut cache = ExpertCache::new(capacity)?;
            let cohort = if capacity >= 20 { 20 } else { 10 };
            for id in 0..512 {
                cache.claim_prefetch(id)?;
            }
            let pending = cache.pending.clone();
            assert_eq!(
                pending.iter().filter(|p| p.is_some()).count(),
                capacity - cohort
            );
            let ids = (512 - cohort as u32..512).collect::<Vec<_>>();
            let routes = cache.reserve(&ids)?;
            assert!(
                routes
                    .iter()
                    .all(|&(slot, hit)| pending[slot].is_none() && !hit)
            );
            let old = (cache.keys.clone(), cache.clock, cache.counts);
            assert!(
                cache
                    .reserve(&(0..cohort as u32 + 1).collect::<Vec<_>>())
                    .is_err()
            );
            assert_eq!((cache.keys.clone(), cache.clock, cache.counts), old);
            cache.invalidate();
            assert_eq!(cache.pending, pending);
            for (slot, &id) in pending.iter().enumerate() {
                if let Some(id) = id {
                    cache.publish_prefetch(slot, id, true);
                }
            }
            assert!(cache.pending.iter().all(Option::is_none));
            cache.clock = u64::MAX;
            let keys = cache.keys.clone();
            assert_eq!(cache.reserve(&[511]), Err(CacheError::Clock));
            assert_eq!(cache.keys, keys);
        }
        Ok(())
    }

    #[test]
    fn persistent_prefetch_sequences_preserve_cache_invariants() -> Result<()> {
        for capacity in [10, 20, 24, 89] {
            let mut cache = ExpertCache::new(capacity)?;
            let mut rng = 0x83a2_5417_u64;
            for step in 0..256 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let ids = (0..20)
                    .map(|i| ((rng + i * 13) % 512) as u32)
                    .collect::<Vec<_>>();
                cache.protect_predictions(&ids)?;
                for &id in &ids[..4] {
                    cache.claim_prefetch(id)?;
                }
                if step % 7 == 0 {
                    cache.invalidate();
                }
                let pending = cache.pending.clone();
                let routes = cache.reserve(&ids[..if capacity >= 20 { 20 } else { 10 }])?;
                for (i, &(slot, _)) in routes.iter().enumerate() {
                    assert!(pending[slot].is_none());
                    assert_eq!(cache.keys[slot], Some(ids[i]));
                }
                for (slot, id) in pending.into_iter().enumerate() {
                    if let Some(id) = id {
                        cache.publish_prefetch(slot, id, step % 3 != 0);
                    }
                }
                for (slot, &key) in cache.keys.iter().enumerate() {
                    assert!(key.is_none() || cache.pending[slot].is_none());
                    if let Some(key) = key {
                        assert!(key < 512);
                        assert!(!cache.keys[slot + 1..].contains(&Some(key)));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn seeded_cache_cohorts_survive_evictions_aging_and_read_failures() -> Result<()> {
        for slots in [10, 20, 64, 512] {
            let mut cache = ExpertCache::new(slots)?;
            let mut rng = 0xdace_3487u64;
            for step in 0..500 {
                let ids = (0..10)
                    .map(|_| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        (rng % 512) as u32
                    })
                    .collect::<Vec<_>>();
                let old = cache.keys.clone();
                let reservation = cache.reserve(&ids)?;
                for (i, &(slot, hit)) in reservation.iter().enumerate() {
                    assert!(slot < slots);
                    assert_eq!(cache.keys[slot], Some(ids[i]));
                    assert_eq!(hit, old.contains(&Some(ids[i])));
                    for (j, &other) in ids.iter().enumerate() {
                        assert_eq!(ids[i] == other, slot == reservation[j].0);
                    }
                }
                let keys = cache.keys.clone();
                let clock = cache.clock;
                assert!(cache.reserve(&[]).is_err());
                assert!(cache.reserve(&[512, 1]).is_err());
                assert_eq!(cache.keys, keys);
                assert_eq!(cache.clock, clock);
                if step % 37 == 0 {
                    cache.invalidate();
                }
            }
        }
        Ok(())
    }
}

#[cfg(kani)]
mod verification {
    use super::*;
    #[kani::proof]
    #[kani::unwind(22)]
    fn cpu_slot_validation_preserves_bounds_for_real_route_widths() {
        let ids: [u32; 20] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= 20);
        let capacity: i32 = kani::any();
        let valid = slots_fit(&ids[..len], capacity);
        let mut expected = len != 0 && capacity > 0;
        for &id in &ids[..len] {
            expected &= i64::from(id) < i64::from(capacity);
        }
        assert_eq!(valid, expected);
        kani::cover!(valid && len == 20);
        kani::cover!(valid && len == 10);
        kani::cover!(!valid && len == 0);
        kani::cover!(!valid && capacity <= 0 && len > 0);
        kani::cover!(!valid && ids[0] == u32::MAX && len > 0);
    }

    #[kani::proof]
    #[kani::unwind(7)]
    fn les_control_parsers_accept_only_the_documented_literals() {
        let bytes: [u8; 5] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= 5);
        if let Ok(value) = std::str::from_utf8(&bytes[..len]) {
            let lru = parse_lru_override(value);
            let half = parse_predictor_half(value);
            assert_eq!(lru.is_some(), value == "auto" || value == "lru");
            assert_eq!(half.is_some(), value == "f32" || value == "f16");
            assert_eq!(lru == Some(true), value == "lru");
            assert_eq!(half == Some(true), value == "f16");
            kani::cover!(lru == Some(false));
            kani::cover!(lru == Some(true));
            kani::cover!(half == Some(false));
            kani::cover!(half == Some(true));
            kani::cover!(lru.is_none() && half.is_none());
        }
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn predictive_victims_exclude_pending_and_fall_back_only_when_needed() {
        let mut cache = ExpertCache::new(10).unwrap();
        cache.predictive_lru = kani::any();
        for i in 0..10 {
            cache.keys[i] = Some(i as u32);
            cache.counts[i] = kani::any();
            cache.ages[i] = kani::any();
            cache.predicted[i] = kani::any();
            cache.pending[i] = if kani::any() { Some(511) } else { None };
        }
        let routed: [u32; 2] = kani::any();
        let eligible = |i: usize| cache.pending[i].is_none() && !routed.contains(&(i as u32));
        let available = (0..10).any(eligible);
        let unpredicted = (0..10).any(|i| eligible(i) && !cache.predicted[i]);
        let victim = cache.victim(&routed);
        assert_eq!(victim.is_some(), available);
        if let Some(slot) = victim {
            assert!(slot < 10 && eligible(slot));
            assert_eq!(cache.predicted[slot], !unpredicted);
            kani::cover!(unpredicted);
            kani::cover!(!unpredicted);
        }
        kani::cover!(!available);
        kani::cover!(cache.predictive_lru && available);
        kani::cover!(!cache.predictive_lru && available);
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn free_slots_precede_all_ready_victims_in_both_policies() {
        let mut cache = ExpertCache::new(10).unwrap();
        cache.predictive_lru = kani::any();
        for i in 0..10 {
            cache.keys[i] = if kani::any() { Some(i as u32) } else { None };
            cache.pending[i] = if cache.keys[i].is_none() && kani::any() {
                Some(511)
            } else {
                None
            };
            cache.counts[i] = kani::any();
            cache.ages[i] = kani::any();
            cache.predicted[i] = kani::any();
        }
        let routed: [u32; 2] = kani::any();
        let free = (0..10).any(|i| cache.keys[i].is_none() && cache.pending[i].is_none());
        let victim = cache.victim(&routed);
        if free {
            let slot = victim.unwrap();
            assert!(slot < 10 && cache.keys[slot].is_none() && cache.pending[slot].is_none());
            kani::cover!(cache.predictive_lru);
            kani::cover!(!cache.predictive_lru);
        }
        if let Some(slot) = victim {
            assert!(cache.pending[slot].is_none());
            kani::cover!(!free && cache.keys[slot].is_some());
        }
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn les_lru_uses_age_without_penalizing_zero_frequency_prefetches() {
        let mut cache = ExpertCache::new(10).unwrap();
        cache.predictive_lru = true;
        for i in 0..10 {
            cache.keys[i] = Some(i as u32);
            cache.counts[i] = kani::any();
            cache.ages[i] = kani::any();
            cache.predicted[i] = kani::any();
        }
        let routed: [u32; 2] = kani::any();
        let slot = cache.victim(&routed).unwrap();
        assert!(slot < 10 && !routed.contains(&(slot as u32)));
        for i in 0..10 {
            if !routed.contains(&(i as u32)) {
                assert!(!cache.predicted[slot] || cache.predicted[i]);
                if cache.predicted[slot] == cache.predicted[i] {
                    assert!(cache.ages[slot] <= cache.ages[i]);
                    if cache.ages[slot] == cache.ages[i] {
                        assert!(slot <= i);
                    }
                }
            }
        }
        kani::cover!(cache.predicted[slot]);
        kani::cover!(!cache.predicted[slot]);
        kani::cover!(cache.counts[slot] == 0 && cache.ages[slot] == 0);
        kani::cover!(cache.counts[slot] == u32::MAX && cache.ages[slot] == 0);
    }

    #[kani::proof]
    #[kani::unwind(25)]
    fn pending_publish_rejects_stale_failed_and_duplicate_records() {
        let mut cache = ExpertCache::new(24).unwrap();
        let slot: usize = kani::any();
        let id: u32 = kani::any();
        kani::assume(slot < 24 && id < 512);
        cache.pending[slot] = Some(id);
        let duplicate: bool = kani::any();
        if duplicate {
            cache.keys[(slot + 1) % 24] = Some(id);
        }
        let received: u32 = kani::any();
        let valid: bool = kani::any();
        let clock: u64 = kani::any();
        cache.clock = clock;
        let published = cache.publish_prefetch(slot, received, valid);
        assert_eq!(
            published,
            received == id && valid && !duplicate && clock < u64::MAX
        );
        assert_eq!(cache.pending[slot].is_none(), received == id);
        assert_eq!(cache.keys[slot] == Some(id), published);
        assert_eq!(cache.prefetched[slot], published);
        if published {
            assert_eq!(cache.clock, clock + 1);
            assert_eq!(cache.ages[slot], cache.clock);
        } else {
            assert_eq!(cache.clock, clock);
        }
        kani::cover!(published);
        kani::cover!(received == id && !valid);
        kani::cover!(received == id && valid && duplicate);
        kani::cover!(received != id);
        kani::cover!(received == id && valid && !duplicate && clock == u64::MAX);
    }

    #[kani::proof]
    #[kani::unwind(25)]
    fn speculative_claim_excludes_pending_and_preserves_pair_space() {
        let mut cache = ExpertCache::new(24).unwrap();
        cache.predictive_lru = kani::any();
        let id: u32 = kani::any();
        let pending: u8 = kani::any();
        kani::assume(pending <= 4);
        for i in 0..usize::from(pending) {
            cache.pending[i] = Some(i as u32);
        }
        let claimed = cache.claim_prefetch(id);
        if let Ok(Some(slot)) = claimed {
            assert!(id < 512 && pending < 4 && id >= u32::from(pending));
            assert!(slot >= usize::from(pending) && slot < 24);
            assert_eq!(cache.pending[slot], Some(id));
            assert_eq!(cache.keys[slot], None);
            kani::cover!(pending == 3);
        }
        if pending == 4 && id < 512 {
            assert_eq!(claimed, Ok(None));
        }
        kani::cover!(id >= 512 && claimed == Err(CacheError::InvalidIds));
    }
    #[kani::proof]
    fn disk_destinations_cannot_wrap_or_exceed_owned_storage() {
        let start: usize = kani::any();
        let length: usize = kani::any();
        let capacity: usize = kani::any();
        let fits = disk_destination_fits(start, length, capacity);
        assert_eq!(
            fits,
            length > 0 && start as u128 + length as u128 <= capacity as u128
        );
        kani::cover!(fits && start + length == capacity);
        kani::cover!(!fits && length == 0);
        kani::cover!(!fits && start as u128 + length as u128 > usize::MAX as u128);
    }
    #[kani::proof]
    fn automatic_cache_cannot_spend_reserved_memory() {
        let available: u64 = kani::any();
        let result = automatic_cache_slots(available);
        let minimum = (8u128 << 30) + 10 * 48 * u128::from(EXPERT_BYTES);
        assert_eq!(result.is_some(), u128::from(available) >= minimum);
        if let Some(slots) = result {
            assert!((10..=107).contains(&slots));
            let allocated = slots as u128 * 48 * u128::from(EXPERT_BYTES);
            assert!(allocated <= 6u128 << 30);
            assert!(allocated + (8u128 << 30) <= u128::from(available));
            kani::cover!(slots == 10);
            kani::cover!(slots == 71);
            kani::cover!(slots == 107);
        }
        kani::cover!(available == u64::MAX);
        kani::cover!(result.is_none());
    }

    #[kani::proof]
    fn automatic_cache_is_monotone() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();
        if a <= b {
            assert!(automatic_cache_slots(a) <= automatic_cache_slots(b));
        }
        kani::cover!(
            a < b && automatic_cache_slots(a) == Some(10) && automatic_cache_slots(b) == Some(107)
        );
    }

    #[cfg(target_os = "macos")]
    #[kani::proof]
    fn available_memory_count_covers_fields_within_buffer() {
        let count: u32 = kani::any();
        let bytes = u64::from(count) * std::mem::size_of::<libc::integer_t>() as u64;
        let external_end = std::mem::offset_of!(libc::vm_statistics64, external_page_count)
            + std::mem::size_of::<libc::natural_t>();
        let valid = available_memory_response_complete(count);
        assert_eq!(
            valid,
            bytes >= external_end as u64
                && bytes <= std::mem::size_of::<libc::vm_statistics64>() as u64
        );
        if valid {
            assert!(bytes >= std::mem::offset_of!(libc::vm_statistics64, free_count) as u64 + 4);
            assert!(
                bytes >= std::mem::offset_of!(libc::vm_statistics64, speculative_count) as u64 + 4
            );
        }
        kani::cover!(valid && count < libc::HOST_VM_INFO64_COUNT);
        kani::cover!(valid && count == libc::HOST_VM_INFO64_COUNT);
        kani::cover!(!valid && bytes < external_end as u64);
        kani::cover!(!valid && count > libc::HOST_VM_INFO64_COUNT);
    }

    #[cfg(target_os = "macos")]
    #[kani::proof]
    fn available_memory_real_os_path() {
        assert!(available_memory_bytes().unwrap() > 0);
    }
    #[kani::proof]
    fn pair_commit_preserves_exact_valid_prefix_without_overflow() {
        let base: i32 = kani::any();
        let current: i32 = kani::any();
        let retained: i32 = kani::any();
        let result = pair_commit_offset(base, current, retained);
        let valid =
            base >= 0 && (1..=2).contains(&retained) && i64::from(base) + 2 == i64::from(current);
        assert_eq!(result.is_some(), valid);
        if let Some(end) = result {
            assert!(end > base && end <= current);
            assert_eq!(i64::from(end), i64::from(base) + i64::from(retained));
            kani::cover!(retained == 1);
            kani::cover!(retained == 2);
        }
        kani::cover!(base == i32::MAX && result.is_none());
        kani::cover!(base == 0 && current == 3 && result.is_none());
    }
    #[kani::proof]
    #[kani::unwind(26)]
    fn reference_partition_is_bounded_and_matches_stopping_rule() {
        let input: u32 = kani::any();
        let output: u32 = kani::any();
        let splits = reference_splits(input, output);
        assert!(splits.is_power_of_two());
        assert!(splits <= (input / 128).max(1));
        assert!(
            u64::from(output) * u64::from(splits) >= 8192
                || u64::from(input) / (2 * u64::from(splits)) < 128
        );
        if splits > 1 {
            assert!(u64::from(output) * u64::from(splits / 2) < 8192);
            assert!(input / splits >= 128);
        }
        kani::cover!(splits == 1 && input == 160 && output == 40);
        kani::cover!(splits > 1);
        kani::cover!(output == 0);
    }
    #[kani::proof]
    fn disk_row_offsets_are_bounded_and_do_not_wrap() {
        let base: u64 = kani::any();
        let row: u64 = kani::any();
        let rows: u64 = kani::any();
        let bytes: u64 = kani::any();
        if let Some(start) = row_offset(base, row, rows, bytes) {
            assert!(row < rows && bytes > 0 && start >= base);
            assert_eq!(start - base, row * bytes);
            assert!(start + bytes <= i64::MAX as u64);
            kani::cover!(row == 0);
            kani::cover!(row + 1 == rows);
        }
    }
    #[kani::proof]
    #[kani::unwind(4)]
    fn ngram_hash_stays_in_its_head_range() {
        let tokens: [u32; 3] = kani::any();
        let multipliers: [u64; 3] = kani::any();
        let n: usize = kani::any();
        let offset: u64 = kani::any();
        let size: u64 = kani::any();
        if let Some(row) = ngram_row(tokens, multipliers, n, offset, size) {
            assert!((2..=3).contains(&n) && size > 0);
            assert!(row >= offset && row - offset < size);
            kani::cover!(n == 2);
            kani::cover!(n == 3);
        }
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn two_routes_preserve_hits_and_never_alias_distinct_experts() {
        let mut cache = ExpertCache::new(10).unwrap();
        for i in 0..10 {
            cache.keys[i] = Some(i as u32);
            cache.ages[i] = 1;
            cache.counts[i] = 1;
        }
        cache.clock = 1;
        let ids: [u32; 2] = kani::any();
        if let Ok(routes) = cache.reserve(&ids) {
            assert!(ids[0] < 512 && ids[1] < 512);
            assert_eq!(routes.len(), 2);
            for i in 0..2 {
                assert!(routes[i].0 < 10);
                assert_eq!(cache.keys[routes[i].0], Some(ids[i]));
                assert_eq!(routes[i].1, ids[i] < 10);
            }
            assert_eq!(routes[0].0 == routes[1].0, ids[0] == ids[1]);
            kani::cover!(ids[0] < 10 && ids[1] >= 10);
            kani::cover!(ids[0] == ids[1]);
        } else {
            assert!(ids[0] >= 512 || ids[1] >= 512);
            for i in 0..10 {
                assert_eq!(cache.keys[i], Some(i as u32));
            }
        }
    }
}
