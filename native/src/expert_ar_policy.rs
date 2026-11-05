//! Guards shared by the four opt-in, fixed-record AR expert paths.

pub(crate) fn flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    crate::flash_support::hc_chain_flag(value, false).map(|on| on && qualified_m5)
}

pub(crate) fn eligible(routed: bool, extents: [[u32; 3]; 3]) -> bool {
    routed && extents == crate::flash_expert_scratch::EXTENTS
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
        assert_eq!(flag(None, qualified), Some(false));
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
