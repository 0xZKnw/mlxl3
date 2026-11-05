//! Admission and readback guards for the optional normalized device selector.
pub(crate) fn eligible(
    enabled: bool,
    temperature: f32,
    top_k: usize,
    penalty: f32,
    speculative: bool,
) -> bool {
    enabled
        && !speculative
        && temperature.is_finite()
        && temperature >= 0.
        && (temperature == 0. || top_k == 1)
        && penalty == 1.
}

pub(crate) fn valid_result(index: u32, words: usize) -> bool {
    words == 248_320 && index < 248_320
}

#[test]
fn device_selection_requires_plain_greedy_and_exact_nonempty_readback() {
    assert!(eligible(true, 0., 0, 1., false));
    assert!(eligible(true, 0.7, 1, 1., false));
    for (enabled, temperature, k, penalty, speculative) in [
        (false, 0., 1, 1., false),
        (true, 0., 1, 1., true),
        (true, 0.7, 2, 1., false),
        (true, -1., 1, 1., false),
        (true, f32::NAN, 1, 1., false),
        (true, f32::INFINITY, 1, 1., false),
        (true, 0., 1, 0., false),
        (true, 0., 1, 1.1, false),
    ] {
        assert!(!eligible(enabled, temperature, k, penalty, speculative));
    }
    assert!(valid_result(0, 248_320));
    assert!(valid_result(248_319, 248_320));
    for (index, words) in [
        (248_320, 248_320),
        (u32::MAX, 248_320),
        (0, 0),
        (0, 248_319),
        (0, usize::MAX),
    ] {
        assert!(!valid_result(index, words));
    }
}

#[cfg(kani)]
#[kani::proof]
fn device_greedy_admission_and_readback_preserve_actual_bounds() {
    let (e, s): (bool, bool) = kani::any();
    let t: f32 = kani::any();
    let p: f32 = kani::any();
    let k: usize = kani::any();
    let admitted = eligible(e, t, k, p, s);
    assert!(!admitted || (e && !s && t.is_finite() && t >= 0. && p == 1. && (t == 0. || k == 1)));
    let i: u32 = kani::any();
    let n: usize = kani::any();
    assert_eq!(valid_result(i, n), n == 248_320 && i < 248_320);
    kani::cover!(admitted && t == 0. && k != 1);
    kani::cover!(admitted && t > 0. && k == 1);
    kani::cover!(!admitted && s);
    kani::cover!(valid_result(i, n) && i == 248_319);
    kani::cover!(!valid_result(i, n) && i == u32::MAX && n == 248_320);
}
