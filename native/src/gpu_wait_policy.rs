//! The production poll loop uses both a time and an iteration bound.
pub(crate) fn may_spin(enabled: bool, elapsed_ns: u128, polls: u32) -> bool {
    enabled && elapsed_ns < 50_000 && polls < 4096
}

#[test]
fn polling_stops_at_either_bound_and_when_disabled() {
    assert!(may_spin(true, 0, 0));
    assert!(may_spin(true, 49_999, 4095));
    for (time, polls) in [(50_000, 0), (0, 4096), (u128::MAX, u32::MAX)] {
        assert!(!may_spin(true, time, polls));
    }
    assert!(!may_spin(false, 0, 0));
}

#[cfg(kani)]
#[kani::proof]
fn gpu_spin_budget_never_exceeds_either_production_limit() {
    let enabled: bool = kani::any();
    let elapsed: u128 = kani::any();
    let polls: u32 = kani::any();
    let actual = may_spin(enabled, elapsed, polls);
    assert_eq!(actual, enabled && elapsed <= 49_999 && polls <= 4095);
    kani::cover!(actual && elapsed == 49_999 && polls == 4095);
    kani::cover!(!actual && enabled && elapsed == 50_000 && polls == 0);
    kani::cover!(!actual && enabled && elapsed == 0 && polls == 4096);
    kani::cover!(!actual && !enabled && elapsed == 0 && polls == 0);
}
