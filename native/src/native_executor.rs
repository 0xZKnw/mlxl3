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
    assert_eq!(draft_capacity_bytes(0), None);
    assert_eq!(draft_capacity_bytes(u32::MAX), None);
    assert_eq!(
        draft_capacity_bytes(4096),
        Some(512 * crate::flash_support::EXPERT_BYTES + 8_651_520)
    );
    assert_eq!(draft_reserve_after(100, 200, true), Some(300));
    assert_eq!(draft_reserve_after(300, 200, false), Some(100));
    assert_eq!(draft_reserve_after(u64::MAX, 1, true), None);
    assert_eq!(draft_reserve_after(100, 200, false), None);
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
#[kani::unwind(13)]
fn native_plan_admission_and_resource_bounds_cover_all_roles() {
    let m5: bool = kani::any();
    let budget: bool = kani::any();
    let lookahead: bool = kani::any();
    let disabled: bool = kani::any();
    let profile = default_profile(m5, budget, lookahead, disabled);
    assert_eq!(profile, m5 && !budget && !lookahead && !disabled);
    kani::cover!(profile);
    kani::cover!(m5 && budget && !profile);
    kani::cover!(m5 && lookahead && !profile);
    kani::cover!(m5 && disabled && !profile);
    let bytes: [u8; 2] = kani::any();
    let length = usize::from(kani::any::<u8>() % 3);
    assert_eq!(default_flag(None, m5, profile), Some(profile));
    if let Ok(value) = std::str::from_utf8(&bytes[..length]) {
        assert_eq!(
            default_flag(Some(value), m5, profile),
            match value {
                "0" => Some(false),
                "1" => Some(m5),
                _ => None,
            }
        );
        kani::cover!(value == "1" && m5);
        kani::cover!(value == "0" && profile);
        kani::cover!(value == "é");
    }
    let bytes: [u8; 12] = kani::any();
    let length = usize::from(kani::any::<u8>() % 13);
    let value = &bytes[..length];
    let parsed = NumericalProfile::parse(Some(value), m5);
    assert_eq!(
        parsed,
        match value {
            b"exact" => Some(NumericalProfile::Exact),
            b"parallel-v1" if m5 => Some(NumericalProfile::ParallelV1),
            _ => None,
        }
    );
    if let Some(profile) = parsed {
        assert_eq!(
            NumericalProfile::parse(Some(profile.name().as_bytes()), m5),
            parsed
        );
        assert_eq!(
            profile.reference_accumulation() == 1,
            profile == NumericalProfile::Exact
        );
    }
    kani::cover!(value == b"parallel-v1" && parsed.is_some());
    kani::cover!(value == b"parallel-v1" && !m5 && parsed.is_none());
    kani::cover!(value == b"exact" && parsed.is_some());
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

/// Keep unqualified budget/prefetch combinations on their existing defaults.
pub(crate) fn default_profile(m5: bool, budgeted: bool, lookahead: bool, disabled: bool) -> bool {
    m5 && !budgeted && !lookahead && !disabled
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
pub(crate) fn defaults(m5: bool) -> bool {
    default_profile(
        m5,
        std::env::var_os("MLXL3_FLASH_REQUEST_MIB").is_some(),
        std::env::var_os("MLXL3_SSD_LOOKAHEAD_DEPTH").is_some_and(|value| value != "0"),
        std::env::var_os("MLXL3_NATIVE_FLASH_COMMON_EXECUTOR").is_some_and(|value| value == "0"),
    )
}

pub(crate) fn default_flag(value: Option<&str>, m5: bool, default: bool) -> Option<bool> {
    crate::flash_support::hc_chain_flag(value, default).map(|on| on && m5)
}

#[test]
fn measured_defaults_preserve_overrides_and_compatibility() {
    for bits in 0..16 {
        let [m5, budget, lookahead, disabled] = [0, 1, 2, 3].map(|bit| bits & (1 << bit) != 0);
        assert_eq!(default_profile(m5, budget, lookahead, disabled), bits == 1);
        assert_eq!(default_flag(None, m5, true), Some(m5));
        assert_eq!(default_flag(None, m5, false), Some(false));
        assert_eq!(default_flag(Some("0"), m5, true), Some(false));
        assert_eq!(default_flag(Some("1"), m5, false), Some(m5));
        for bad in ["", "01", "true", " 1", "1\n", "é"] {
            assert_eq!(default_flag(Some(bad), m5, true), None);
        }
    }
}

/// Immutable numerical target; prefixes belong to the model owner that loaded it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum NumericalProfile {
    Exact,
    ParallelV1,
}
impl NumericalProfile {
    pub(crate) fn parse(value: Option<&[u8]>, m5: bool) -> Option<Self> {
        match value {
            None | Some(b"exact") => Some(Self::Exact),
            Some(b"parallel-v1") if m5 => Some(Self::ParallelV1),
            _ => None,
        }
    }
    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::ParallelV1 => "parallel-v1",
        }
    }
    pub(crate) fn reference_accumulation(self) -> u8 {
        u8::from(self == Self::Exact)
    }
}

#[test]
fn numerical_profile_is_explicit_versioned_and_hardware_qualified() {
    for m5 in [false, true] {
        assert_eq!(
            NumericalProfile::parse(None, m5),
            Some(NumericalProfile::Exact)
        );
        assert_eq!(
            NumericalProfile::parse(Some(b"exact"), m5),
            Some(NumericalProfile::Exact)
        );
        assert_eq!(
            NumericalProfile::parse(Some(b"parallel-v1"), m5),
            m5.then_some(NumericalProfile::ParallelV1)
        );
        for bad in [
            "",
            "fast",
            "parallel",
            "parallel-v2",
            " parallel-v1",
            "exact\n",
            "é",
        ] {
            assert_eq!(NumericalProfile::parse(Some(bad.as_bytes()), m5), None);
        }
    }
    for profile in [NumericalProfile::Exact, NumericalProfile::ParallelV1] {
        assert_eq!(
            NumericalProfile::parse(Some(profile.name().as_bytes()), true),
            Some(profile)
        );
        assert_eq!(
            profile.reference_accumulation(),
            u8::from(profile == NumericalProfile::Exact)
        );
    }
}
