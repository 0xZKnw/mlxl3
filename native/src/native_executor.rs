//! Immutable shape and prospective resource contracts shared by native B1 and Bn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Role {
    Gdn,
    Qsa,
    Ple,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerPlan {
    role: Role,
    context: u32,
}
impl LayerPlan {
    pub(crate) fn new(role: Role, context: u32) -> Option<Self> {
        (context > 0 && context <= crate::flash_support::FLASH_MAX_CONTEXT)
            .then_some(Self { role, context })
    }
    pub(crate) fn admits(self, rows: usize) -> bool {
        (1..=4).contains(&rows) && rows <= self.context as usize
    }
    /// Upper bounds for new prefixes, separate from already-live, deduplicated backings.
    pub(crate) fn prefix_bytes(self) -> u64 {
        match self.role {
            Role::Gdn => crate::frame_arena::FRAME_BYTES,
            Role::Ple => crate::frame_arena::FRAME_BYTES + 9 * 10240 * 4,
            Role::Qsa => {
                u64::from(self.context) * 2048 + u64::from(self.context.div_ceil(4)) * 256 + 3 * 256
            }
        }
    }
}

pub(crate) fn hot_owners_after(active: usize, owned: bool) -> Option<usize> {
    if active >= if owned { 4 } else { 1 } {
        return None;
    }
    Some(active + 1)
}

#[test]
fn plans_reject_invalid_contexts_and_bound_every_role() {
    for role in [Role::Gdn, Role::Qsa, Role::Ple] {
        assert!(LayerPlan::new(role, 0).is_none());
        assert!(LayerPlan::new(role, u32::MAX).is_none());
        let plan = LayerPlan::new(role, 4096).unwrap();
        for rows in [0, 1, 2, 3, 4, 5, usize::MAX] {
            assert_eq!(plan.admits(rows), (1..=4).contains(&rows));
        }
        assert!(plan.prefix_bytes() > 0);
    }
    for active in [0, 1, 3, 4, usize::MAX] {
        for owned in [false, true] {
            assert_eq!(
                hot_owners_after(active, owned).is_some(),
                active < if owned { 4 } else { 1 }
            );
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn native_plan_admission_and_resource_bounds_cover_all_roles() {
    let kind: u8 = kani::any();
    let context: u32 = kani::any();
    let rows: usize = kani::any();
    let role = match kind % 3 {
        0 => Role::Gdn,
        1 => Role::Qsa,
        _ => Role::Ple,
    };
    let plan = LayerPlan::new(role, context);
    assert_eq!(
        plan.is_some(),
        context >= 1 && context <= crate::flash_support::FLASH_MAX_CONTEXT
    );
    if let Some(plan) = plan {
        assert_eq!(
            plan.admits(rows),
            rows >= 1 && rows <= 4 && rows <= context as usize
        );
        assert!(plan.prefix_bytes() > 0 && plan.prefix_bytes() < 1_000_000_000);
        let draft = draft_capacity_bytes(context).unwrap();
        assert!(draft >= 512 * crate::flash_support::EXPERT_BYTES);
        assert!(draft < 2_000_000_000);
        kani::cover!(role == Role::Qsa && context == 4096 && plan.admits(4));
        kani::cover!(role == Role::Ple && plan.admits(1));
    }
}

#[cfg(kani)]
#[kani::proof]
fn private_hot_owners_preserve_bounded_scalar_and_block_leases() {
    let active: usize = kani::any();
    let owned: bool = kani::any();
    let next = hot_owners_after(active, owned);
    assert_eq!(next.is_some(), active < if owned { 4 } else { 1 });
    if let Some(next) = next {
        assert!(next <= 4 && next == active + 1);
    }
    kani::cover!(owned && active == 3 && next == Some(4));
    kani::cover!(!owned && active == 1 && next.is_none());
}

pub(crate) fn draft_reserve_after(current: u64, bytes: u64, acquire: bool) -> Option<u64> {
    if acquire {
        current.checked_add(bytes)
    } else {
        current.checked_sub(bytes)
    }
}

#[cfg(kani)]
#[kani::proof]
fn draft_reserves_never_wrap_or_refund_an_unowned_bank() {
    let current: u64 = kani::any();
    let bytes: u64 = kani::any();
    let acquire: bool = kani::any();
    let next = draft_reserve_after(current, bytes, acquire);
    if acquire {
        assert_eq!(
            next.is_some(),
            u128::from(current) + u128::from(bytes) <= u128::from(u64::MAX)
        );
        if let Some(next) = next {
            assert_eq!(draft_reserve_after(next, bytes, false), Some(current));
        }
    } else {
        assert_eq!(next.is_some(), current >= bytes);
        if let Some(next) = next {
            assert_eq!(next + bytes, current);
        }
    }
    kani::cover!(acquire && next.is_none());
    kani::cover!(!acquire && bytes > current && next.is_none());
    kani::cover!(acquire && bytes > 0 && next.is_some());
}

pub(crate) fn draft_capacity_bytes(context: u32) -> Option<u64> {
    LayerPlan::new(Role::Qsa, context)?
        .prefix_bytes()
        .checked_add(512 * crate::flash_support::EXPERT_BYTES)
}
