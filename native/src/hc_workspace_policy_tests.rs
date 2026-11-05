use super::*;

#[test]
fn workspace_claim_excludes_busy_and_foreign_dtype_and_releases_after_error() {
    let key = Key {
        fp32: true,
        up: true,
    };
    let mut state = State::default();
    assert!(state.acquire(key));
    assert!(!state.acquire(key));
    assert!(!state.clear());
    state.release(true);
    assert_eq!(state.key(), Some(key));
    assert!(!state.acquire(Key { fp32: false, ..key }));
    assert!(state.acquire(key));
    state.release(false);
    assert_eq!(state.key(), None);
    assert!(state.acquire(Key {
        fp32: false,
        up: false
    }));
    state.release(true);
    assert!(state.clear());
    assert_eq!(state.key(), None);
}

#[test]
fn workspace_three_buffer_layout_has_no_unused_up_or_hidden_budget() {
    for fp32 in [false, true] {
        let four = Key { fp32, up: true };
        let three = Key { fp32, up: false };
        let item = if fp32 { 4 } else { 2 };
        assert_eq!(four.bytes(), 21120 * item);
        assert_eq!(three.bytes(), 10880 * item);
        assert_eq!(four.reserve(97), Some(2048640 * item));
        assert_eq!(three.reserve(97), Some(1055360 * item));
        assert_eq!(four.reserve(u64::MAX), None);
        let mut state = State::default();
        assert!(state.acquire(four));
        state.release(true);
        assert!(state.acquire(three));
        assert_eq!(state.key(), Some(three));
    }
}

#[test]
fn workspace_flag_is_strict_default_off_and_m5_only() {
    for qualified in [false, true] {
        assert_eq!(flag(None, qualified), Some(false));
        assert_eq!(flag(Some("0"), qualified), Some(false));
        assert_eq!(flag(Some("1"), qualified), Some(qualified));
        for invalid in ["", "01", "10", " 1", "1 ", "1\n", "true", "2", "é", "1\0"] {
            assert_eq!(flag(Some(invalid), qualified), None);
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(6)]
fn hc_workspace_exclusivity_and_error_clear_follow_actual_state_transitions() {
    let first = Key {
        fp32: kani::any(),
        up: kani::any(),
    };
    let second = Key {
        fp32: kani::any(),
        up: kani::any(),
    };
    let completed: bool = kani::any();
    let mut state = State::default();
    assert!(state.acquire(first));
    assert!(!state.acquire(second));
    assert!(!state.clear());
    state.release(completed);
    assert_eq!(state.key(), completed.then_some(first));
    let allowed = !completed || first.fp32 == second.fp32;
    let actual = state.acquire(second);
    assert_eq!(actual, allowed);
    kani::cover!(completed && !actual);
    kani::cover!(completed && actual && first.up != second.up);
    kani::cover!(!completed && actual && first.fp32 != second.fp32);
    if actual {
        state.release(true);
    }
    assert!(state.clear());
    assert_eq!(state.key(), None);
}

#[cfg(kani)]
#[kani::proof]
fn hc_workspace_reserve_is_exact_or_rejects_overflow() {
    let key = Key {
        fp32: kani::any(),
        up: kani::any(),
    };
    let count: u64 = kani::any();
    let expected = (10880u128 + if key.up { 10240 } else { 0 })
        * if key.fp32 { 4 } else { 2 }
        * u128::from(count);
    assert_eq!(key.reserve(count), u64::try_from(expected).ok());
    kani::cover!(key.fp32 && key.up && count == 97 && key.reserve(count) == Some(8194560));
    kani::cover!(key.fp32 && !key.up && count == 97 && key.reserve(count) == Some(4221440));
    kani::cover!(key.reserve(count).is_none());
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn hc_workspace_flag_preserves_off_default_and_bounded_literals() {
    let qualified: bool = kani::any();
    let present: bool = kani::any();
    let raw: [u8; 4] = kani::any();
    let length = usize::from(kani::any::<u8>() % 5);
    if !present {
        assert_eq!(flag(None, qualified), Some(false));
    } else if let Ok(value) = std::str::from_utf8(&raw[..length]) {
        let actual = flag(Some(value), qualified);
        assert_eq!(
            actual,
            match value {
                "0" => Some(false),
                "1" => Some(qualified),
                _ => None,
            }
        );
        kani::cover!(qualified && actual == Some(true));
        kani::cover!(!qualified && value == "1" && actual == Some(false));
        kani::cover!(value.is_empty() && actual.is_none());
    }
}

#[test]
fn workspace_reserve_reaches_actual_admission_plan_and_rejects_overflow() {
    let base = crate::flash_memory::ResourcePlan::native_ar(64, 4096).unwrap();
    for (up, bytes) in [(false, 4221440), (true, 8194560)] {
        let plan = base.with_hc_workspace(97, up).unwrap();
        assert_eq!(plan.extra_bytes() - base.extra_bytes(), bytes);
        assert_eq!(
            serde_json::to_value(plan).unwrap()["hc_workspace_reserve_bytes"],
            bytes
        );
    }
    assert!(base.with_hc_workspace(u64::MAX, true).is_none());
}

#[cfg(kani)]
#[kani::proof]
fn hc_workspace_actual_admission_sum_is_checked_and_includes_every_buffer() {
    let count: u64 = kani::any();
    let up: bool = kani::any();
    let base = crate::flash_memory::ResourcePlan::native_ar(64, 4096).unwrap();
    let expected =
        u128::from(base.extra_bytes()) + u128::from(count) * if up { 84480 } else { 43520 };
    let actual = base.with_hc_workspace(count, up);
    assert_eq!(actual.is_some(), expected <= u128::from(u64::MAX));
    if let Some(plan) = actual {
        assert_eq!(u128::from(plan.extra_bytes()), expected);
    }
    kani::cover!(count == 97 && up && actual.is_some());
    kani::cover!(count == 97 && !up && actual.is_some());
    kani::cover!(actual.is_none());
}
