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

/// Reserve resident weights, MTP, scratch buffers and room for other applications.
pub fn automatic_cache_slots(available_bytes: u64) -> Option<usize> {
    let budget = available_bytes.checked_sub(8 << 30)?.min(6 << 30);
    let slots = budget / (48 * EXPERT_BYTES);
    (slots >= 10).then_some(slots as usize)
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
        status == 0 && count == libc::HOST_VM_INFO64_COUNT && released == 0,
        "cannot query available system memory ({status}/{released})"
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
    counts: [u32; 512],
    ages: Vec<u64>,
    clock: u64,
}

impl ExpertCache {
    pub fn new(slots: usize) -> Result<Self, CacheError> {
        if !(10..=512).contains(&slots) {
            return Err(CacheError::Capacity);
        }
        Ok(Self {
            keys: vec![None; slots],
            counts: [0; 512],
            ages: vec![0; slots],
            clock: 0,
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
        if unique.len() > self.keys.len() {
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
                None => (0..self.keys.len())
                    .filter(|&i| self.keys[i].is_none_or(|key| !unique.contains(&key)))
                    .min_by_key(|&i| {
                        (
                            self.keys[i].map_or(0, |key| self.counts[key as usize]),
                            self.ages[i],
                            i,
                        )
                    })
                    .ok_or(CacheError::NoSlot)?,
            };
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

    pub fn invalidate(&mut self) {
        self.keys.fill(None);
    }
    pub fn capacity(&self) -> usize {
        self.keys.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
