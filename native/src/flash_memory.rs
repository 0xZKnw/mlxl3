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
    hc_workspace_reserve_bytes: u64,
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
            hc_workspace_reserve_bytes: 0,
        })
    }

    pub fn native_ar(positions: u32, context: u32) -> Option<Self> {
        if positions == 0
            || positions > context
            || context > crate::flash_support::FLASH_MAX_CONTEXT
        {
            return None;
        }
        let capacity = positions.checked_next_power_of_two()?.max(32).min(context);
        let mut plan = Self::ar(positions)?;
        plan.kv_bytes = u64::from(capacity) * 24576;
        plan.index_bytes = (u64::from(capacity) / 4 + 3) * 12 * 128 * 2;
        // A restored prefix may fork a complete slab of this capacity.
        plan.append_overlap_bytes = plan.kv_bytes + plan.index_bytes;
        Some(plan)
    }

    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
    pub(crate) fn with_hc_workspace(mut self, count: u64, up: bool) -> Option<Self> {
        let bytes = crate::hc_workspace_policy::Key { fp32: true, up }.reserve(count)?;
        self.hc_workspace_reserve_bytes = 0;
        self.extra_bytes().checked_add(bytes)?;
        self.hc_workspace_reserve_bytes = bytes;
        Some(self)
    }

    pub fn extra_bytes(self) -> u64 {
        self.kv_bytes
            + self.index_bytes
            + self.gdn_and_history_bytes
            + self.ple_and_hidden_bytes
            + self.append_overlap_bytes
            + self.scratch_reserve_bytes
            + self.hc_workspace_reserve_bytes
    }
}

/// Budget mode preserves the reference policy: reject speculation before loading its weights.
pub fn native_speculation_permitted(limit: Option<u64>) -> bool {
    limit.is_none()
}

#[cfg(kani)]
#[kani::proof]
fn native_request_budget_keeps_every_configured_limit_ar_only() {
    let configured: bool = kani::any();
    let bytes: u64 = kani::any();
    let limit = configured.then_some(bytes);
    assert_eq!(native_speculation_permitted(limit), !configured);
    kani::cover!(!configured && native_speculation_permitted(limit));
    kani::cover!(configured && bytes == 0 && !native_speculation_permitted(limit));
    kani::cover!(configured && bytes == u64::MAX && !native_speculation_permitted(limit));
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

#[cfg(any(feature = "mlx", all(target_os = "macos", feature = "direct-metal")))]
pub(crate) fn request_limit() -> anyhow::Result<Option<u64>> {
    use anyhow::Context;
    match std::env::var("MLXL3_FLASH_REQUEST_MIB") {
        Ok(text) => Ok(Some(
            text.parse::<u64>()
                .ok()
                .filter(|n| *n > 0)
                .and_then(|n| n.checked_mul(1 << 20))
                .context("MLXL3_FLASH_REQUEST_MIB requires a positive integer MiB budget")?,
        )),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
#[test]
fn native_plan_reserves_geometric_slabs_and_their_forks() {
    for (positions, context, capacity) in [
        (1, 1, 1),
        (1, 512, 32),
        (33, 50, 50),
        (2053, 4096, 4096),
        (131073, 262144, 262144),
    ] {
        let plan = ResourcePlan::native_ar(positions, context).unwrap();
        assert_eq!(plan.kv_bytes, capacity * 24576);
        assert_eq!(plan.append_overlap_bytes, plan.kv_bytes + plan.index_bytes);
    }
    for (n, c) in [(0, 512), (2, 1), (1, 0), (1, u32::MAX)] {
        assert!(ResourcePlan::native_ar(n, c).is_none());
    }
}

#[cfg(kani)]
#[kani::proof]
fn native_admission_covers_the_complete_geometric_slab() {
    let (positions, context): (u32, u32) = kani::any();
    let plan = ResourcePlan::native_ar(positions, context);
    assert_eq!(
        plan.is_some(),
        positions > 0 && positions <= context && context <= crate::flash_support::FLASH_MAX_CONTEXT
    );
    if let Some(plan) = plan {
        let capacity = plan.kv_bytes / 24576;
        assert!(capacity >= u64::from(positions) && capacity <= u64::from(context));
        assert!(capacity >= u64::from(context.min(32)));
        assert!(capacity == u64::from(context) || capacity.is_power_of_two());
        let candidate: u32 = kani::any();
        if candidate >= 32
            && candidate.is_power_of_two()
            && candidate >= positions
            && candidate < context
        {
            assert!(capacity <= u64::from(candidate));
        }
        assert_eq!(plan.append_overlap_bytes, plan.kv_bytes + plan.index_bytes);
        assert!(plan.extra_bytes() >= 2 * plan.kv_bytes + (512 << 20));
        kani::cover!(positions == 2053 && context == 4096 && capacity == 4096);
        kani::cover!(positions == 33 && context == 50 && capacity == 50);
        kani::cover!(positions == 1 && context == 262144 && capacity == 32);
    }
}
