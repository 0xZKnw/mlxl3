//! Guards shared by the four qualified, fixed-record AR expert paths.

pub(crate) fn flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    crate::ar10_defaults::flag(value, qualified_m5)
}

pub(crate) fn eligible(routed: bool, extents: [[u32; 3]; 3]) -> bool {
    routed && extents == crate::flash_expert_scratch::EXTENTS
}

pub(crate) fn cohort(count: usize, destinations: usize) -> bool {
    count > 0 && count <= 10 && count == destinations
}

pub(crate) fn prefix(index: usize, count: usize, width: u32, length: u64) -> Option<u64> {
    if count == 0 || count > 10 || index >= count || width == 0 {
        return None;
    }
    let bytes = u64::from(width) * 2;
    let offset = (index as u64).checked_mul(bytes)?;
    (offset.checked_add(bytes)? <= length).then_some(offset)
}

#[test]
fn expert_ar_flags_and_destination_prefixes_are_strict() {
    for qualified in [false, true] {
        assert_eq!(flag(None, qualified), Some(qualified));
        assert_eq!(flag(Some("0"), qualified), Some(false));
        assert_eq!(flag(Some("1"), qualified), Some(qualified));
        for value in ["", "01", "true", " 1", "1\n", "é"] {
            assert_eq!(flag(Some(value), qualified), None);
        }
    }
    for count in 1..=10 {
        for index in 0..count {
            assert_eq!(
                prefix(index, count, 2560, count as u64 * 5120),
                Some(index as u64 * 5120)
            );
        }
        assert_eq!(prefix(count, count, 2560, u64::MAX), None);
    }
    assert_eq!(prefix(0, 0, 2560, u64::MAX), None);
    assert_eq!(prefix(0, 11, 2560, u64::MAX), None);
    assert_eq!(prefix(0, 1, 0, u64::MAX), None);
    assert_eq!(prefix(9, 10, 2560, 51199), None);
    for count in [0, 1, 2, 10, 11, usize::MAX] {
        for destinations in [0, 1, 2, 10, 11, usize::MAX] {
            assert_eq!(
                cohort(count, destinations),
                (1..=10).contains(&count) && count == destinations
            );
        }
    }
    assert!(eligible(true, crate::flash_expert_scratch::EXTENTS));
    assert!(!eligible(false, crate::flash_expert_scratch::EXTENTS));
    assert!(!eligible(true, [[u32::MAX; 3]; 3]));
}

#[cfg(kani)]
#[kani::proof]
fn expert_ar_prefixes_are_bounded_disjoint_and_never_overflow() {
    let index: usize = kani::any();
    let count: usize = kani::any();
    let width: u32 = kani::any();
    let length: u64 = kani::any();
    let actual = prefix(index, count, width, length);
    let valid = count > 0
        && count <= 10
        && index < count
        && width > 0
        && (index as u128 + 1) * u128::from(width) * 2 <= u128::from(length);
    assert_eq!(actual.is_some(), valid);
    if let Some(offset) = actual {
        assert_eq!(u128::from(offset), index as u128 * u128::from(width) * 2);
        assert!(offset + u64::from(width) * 2 <= length);
    }
    kani::cover!(actual.is_some() && index == 9 && count == 10);
    kani::cover!(actual.is_none() && length == 51199 && index == 9 && width == 2560);
    kani::cover!(actual.is_none() && count == usize::MAX);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(40)]
fn expert_ar_selection_requires_exact_actual_record_extents() {
    let routed: bool = kani::any();
    let extents: [[u32; 3]; 3] = kani::any();
    let selected = eligible(routed, extents);
    assert_eq!(
        selected,
        routed && extents == [[2560, 640, 640], [2560, 640, 640], [640, 2560, 2560]]
    );
    kani::cover!(selected);
    kani::cover!(!selected && routed);
    kani::cover!(!selected && !routed);
}

#[cfg(kani)]
#[kani::proof]
fn expert_axis_admits_only_nonempty_bounded_ready_tables() {
    let count: usize = kani::any();
    let destinations: usize = kani::any();
    let actual = cohort(count, destinations);
    assert_eq!(actual, count != 0 && count < 11 && destinations == count);
    if actual {
        assert!(count - 1 < 10);
    }
    kani::cover!(actual && count == 1);
    kani::cover!(actual && count == 10);
    kani::cover!(!actual && count == 0);
    kani::cover!(!actual && count == usize::MAX);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn expert_ar_flags_preserve_qualified_defaults_and_overrides() {
    let bytes: [u8; 4] = kani::any();
    let length = usize::from(kani::any::<u8>() % 5);
    let qualified: bool = kani::any();
    let present: bool = kani::any();
    if !present {
        assert_eq!(flag(None, qualified), Some(qualified));
    } else if let Ok(value) = std::str::from_utf8(&bytes[..length]) {
        let expected = if length == 1 && bytes[0] == b'0' {
            Some(false)
        } else if length == 1 && bytes[0] == b'1' {
            Some(qualified)
        } else {
            None
        };
        assert_eq!(flag(Some(value), qualified), expected);
        kani::cover!(value == "1" && qualified);
        kani::cover!(value == "1" && !qualified);
        kani::cover!(value == "0");
        kani::cover!(value == "é");
    }
    kani::cover!(!present);
}

pub(crate) fn finish_mix_eligible(
    enabled: bool,
    fp32: bool,
    trace: bool,
    axis: bool,
    simd: usize,
    threads: usize,
) -> bool {
    enabled && fp32 && !trace && axis && simd == 32 && threads >= 640
}
pub(crate) fn finish_view_valid(
    partial_offset: u64,
    partial_length: u64,
    svh_offset: u64,
    svh_length: u64,
) -> bool {
    partial_offset.is_multiple_of(4)
        && svh_offset.is_multiple_of(2)
        && partial_offset
            .checked_add(10240)
            .is_some_and(|end| end <= partial_length)
        && svh_offset
            .checked_add(5120)
            .is_some_and(|end| end <= svh_length)
}
#[test]
fn finish_mix_requires_private_fp32_axis_and_complete_partial_scale_views() {
    assert!(finish_mix_eligible(true, true, false, true, 32, 640));
    for (e, f, t, a, s, n) in [
        (false, true, false, true, 32, 640),
        (true, false, false, true, 32, 640),
        (true, true, true, true, 32, 640),
        (true, true, false, false, 32, 640),
        (true, true, false, true, 16, 640),
        (true, true, false, true, 32, 639),
    ] {
        assert!(!finish_mix_eligible(e, f, t, a, s, n));
    }
    assert!(finish_view_valid(10240, 20480, 2, 5122));
    for (p, n, s, m) in [
        (1, 10241, 0, 5120),
        (0, 10239, 0, 5120),
        (0, 10240, 1, 5121),
        (0, 10240, 0, 5119),
        (u64::MAX - 3, u64::MAX, 0, 5120),
    ] {
        assert!(!finish_view_valid(p, n, s, m));
    }
}
#[cfg(kani)]
#[kani::proof]
fn finish_mix_preserves_actual_capability_and_record_view_bounds() {
    let e: bool = kani::any();
    let f: bool = kani::any();
    let t: bool = kani::any();
    let a: bool = kani::any();
    let s: usize = kani::any();
    let n: usize = kani::any();
    assert_eq!(
        finish_mix_eligible(e, f, t, a, s, n),
        e && f && !t && a && s == 32 && n >= 640
    );
    let p: u64 = kani::any();
    let l: u64 = kani::any();
    let v: u64 = kani::any();
    let m: u64 = kani::any();
    assert_eq!(
        finish_view_valid(p, l, v, m),
        p % 4 == 0
            && v % 2 == 0
            && u128::from(p) + 10240 <= u128::from(l)
            && u128::from(v) + 5120 <= u128::from(m)
    );
    kani::cover!(finish_mix_eligible(e, f, t, a, s, n) && n == 640);
    kani::cover!(!finish_mix_eligible(e, f, t, a, s, n) && n == 639);
    kani::cover!(finish_view_valid(p, l, v, m) && p > 0 && v > 0);
    kani::cover!(!finish_view_valid(p, l, v, m) && p == u64::MAX - 3);
}
