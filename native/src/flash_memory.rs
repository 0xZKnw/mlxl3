//! Prospective AR allocation at the request boundary, in bytes. The base already
//! includes resident weights and banks. Never sum RSS, footprint and MLX active.
use serde::Serialize;

#[derive(Clone, Copy, Debug, Serialize)]
pub struct ResourcePlan {
    kv_bytes: u64,
    index_bytes: u64,
    gdn_and_history_bytes: u64,
    ple_and_hidden_bytes: u64,
    append_overlap_bytes: u64,
    scratch_reserve_bytes: u64,
}

impl ResourcePlan {
    pub fn ar(positions: u32) -> Option<Self> {
        if positions == 0 || positions > i32::MAX as u32 {
            return None;
        }
        let n = u64::from(positions);
        let kv_bytes = n * 12 * 2 * 2 * 256 * 2;
        // Reserve the maximum three-token tail even at an exact block boundary.
        let index_bytes = (n / 4 + 3) * 12 * 128 * 2;
        Some(Self {
            kv_bytes,
            index_bytes,
            // The three-token history is a view of a four-token backing.
            gdn_and_history_bytes: 36 * (48 * 128 * 128 * 4 + 4 * 10240 * 2),
            ple_and_hidden_bytes: (10 + 1) * 10240 * 4,
            // Concatenating old K/V and block arrays may retain both generations.
            append_overlap_bytes: kv_bytes + index_bytes,
            // Policy headroom for transient graphs, logits, staging and programs.
            // This is an admission estimate, independent of MLX's allocator limit.
            scratch_reserve_bytes: 512 << 20,
        })
    }

    pub fn extra_bytes(self) -> u64 {
        self.kv_bytes
            + self.index_bytes
            + self.gdn_and_history_bytes
            + self.ple_and_hidden_bytes
            + self.append_overlap_bytes
            + self.scratch_reserve_bytes
    }
}

/// The default cache may be non-uniform. Only take slots above the floor,
/// and compare the *observed* process memory again after actually releasing banks.
pub fn shrink_slots(capacity: usize, excess: u64) -> usize {
    let remove = excess.div_ceil(crate::flash_support::EXPERT_BYTES).min(512) as usize;
    capacity.saturating_sub(remove).max(10).min(capacity)
}

pub fn fits(base: u64, extra: u64, limit: u64) -> bool {
    base <= limit && extra <= limit - base
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn admission_includes_overlap_and_preserves_minimum_slots() {
        assert!(ResourcePlan::ar(0).is_none());
        assert!(ResourcePlan::ar(u32::MAX).is_none());
        for n in [1, 3, 4, 2051, 2052, 32768, i32::MAX as u32] {
            let p = ResourcePlan::ar(n).unwrap();
            assert_eq!(p.kv_bytes, u64::from(n) * 24576);
            assert_eq!(p.append_overlap_bytes, p.kv_bytes + p.index_bytes);
            assert!(fits(42, p.extra_bytes(), 42 + p.extra_bytes()));
            assert!(!fits(42, p.extra_bytes(), 41 + p.extra_bytes()));
        }
        assert!(!fits(u64::MAX, 1, u64::MAX));
        assert_eq!(shrink_slots(10, u64::MAX), 10);
        assert_eq!(shrink_slots(53, 0), 53);
        assert_eq!(shrink_slots(53, crate::flash_support::EXPERT_BYTES + 1), 51);
    }
}

#[cfg(kani)]
#[kani::proof]
fn admission_and_shrink_respect_memory_bounds() {
    let (base, limit, extra): (u64, u64, u64) = kani::any();
    assert_eq!(
        fits(base, extra, limit),
        u128::from(base) + u128::from(extra) <= u128::from(limit)
    );
    let slots: usize = kani::any();
    kani::assume((10..=512).contains(&slots));
    let chosen = shrink_slots(slots, extra);
    assert!((10..=slots).contains(&chosen));
    kani::cover!(chosen < slots && chosen > 10);
    let positions: u32 = kani::any();
    if let Some(plan) = ResourcePlan::ar(positions) {
        assert!(positions > 0 && positions <= i32::MAX as u32);
        assert!(plan.extra_bytes() >= plan.kv_bytes * 2 + plan.scratch_reserve_bytes);
        kani::cover!(positions == 2052);
    }
}
