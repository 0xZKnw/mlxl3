//! Optional residency never raises the already admitted fixed-weight payload limit.
pub(crate) fn eligible(enabled: bool, api: bool, bytes: u64, limit: u64) -> bool {
    enabled && api && bytes > 0 && bytes <= limit
}

#[cfg(test)]
#[test]
fn fixed_residency_requires_capability_nonempty_payload_and_existing_limit() {
    assert!(eligible(true, true, 17, 17));
    for (enabled, api, bytes, limit) in [
        (false, true, 17, 17),
        (true, false, 17, 17),
        (true, true, 0, 17),
        (true, true, 18, 17),
        (true, true, u64::MAX, u64::MAX - 1),
    ] {
        assert!(!eligible(enabled, api, bytes, limit));
    }
    assert!(eligible(true, true, u64::MAX, u64::MAX));
    for qualified in [false, true] {
        assert_eq!(
            crate::flash_support::hc_kernel_flag(None, qualified),
            Some(false)
        );
        assert_eq!(
            crate::flash_support::hc_kernel_flag(Some("0"), qualified),
            Some(false)
        );
        assert_eq!(
            crate::flash_support::hc_kernel_flag(Some("1"), qualified),
            Some(qualified)
        );
        assert_eq!(
            crate::flash_support::hc_kernel_flag(Some("01"), qualified),
            None
        );
    }
}

#[cfg(kani)]
#[kani::proof]
fn fixed_residency_enrollment_requires_capability_and_admitted_nonempty_bytes() {
    let enabled: bool = kani::any();
    let api: bool = kani::any();
    let bytes: u64 = kani::any();
    let limit: u64 = kani::any();
    let actual = eligible(enabled, api, bytes, limit);
    assert_eq!(actual, enabled && api && bytes != 0 && bytes <= limit);
    if actual {
        assert!(bytes <= limit);
        kani::cover!(bytes == limit);
        kani::cover!(bytes < limit);
    }
    kani::cover!(!actual && !enabled && api && bytes > 0 && bytes <= limit);
    kani::cover!(!actual && enabled && !api);
    kani::cover!(!actual && enabled && api && bytes == 0);
    kani::cover!(!actual && enabled && api && bytes > limit);
}
