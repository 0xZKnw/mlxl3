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
