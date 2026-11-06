//! CPU admission/identity/ownership guards used by the native device-Hot transaction.

pub(crate) const SLOTS: usize = 53;
#[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
pub(crate) fn capacity(value: Option<&str>) -> Option<usize> {
    match value {
        None | Some("53") => Some(SLOTS),
        Some("64") => Some(64),
        _ => None,
    }
}
pub(crate) fn supported_slots(slots: usize) -> bool {
    matches!(slots, SLOTS | 64)
}
pub(crate) fn retirement_bit(slot: usize, slots: usize) -> Option<u64> {
    (supported_slots(slots) && slot < slots).then(|| 1u64 << slot)
}
#[test]
fn capacities_and_retirement_bits_are_strict_and_bounded() {
    assert_eq!(capacity(None), Some(53));
    for (text, slots) in [("53", 53), ("64", 64)] {
        assert_eq!(capacity(Some(text)), Some(slots));
        assert_eq!(retirement_bit(slots - 1, slots), Some(1u64 << (slots - 1)));
        assert_eq!(retirement_bit(slots, slots), None);
    }
    for text in ["", "063", "65", "80", " 64", "64\n"] {
        assert_eq!(capacity(Some(text)), None);
    }
    assert_eq!(retirement_bit(64, 64), None);
    assert_eq!(retirement_bit(0, usize::MAX), None);
}

pub(crate) const EXPERTS: usize = 512;
pub(crate) const ROUTES: usize = 10;
pub(crate) const RECORD_BYTES: u64 = 1_248_012;
pub(crate) const CONTRIBUTION_BYTES: u64 = 2 * 2560;
pub(crate) const AUXILIARY_LIMIT: u64 = 2 * 1024 * 1024;
pub(crate) const RESERVE_BYTES: u64 = ROUTES as u64 * RECORD_BYTES + AUXILIARY_LIMIT;

pub(crate) fn workspace_valid(lengths: [u64; 5], private: [bool; 5], private_slab: bool) -> bool {
    lengths == [102400, 102400, 12800, 12800, 51200]
        && private == [true, true, true, true, private_slab]
}

pub(crate) fn workspace_recyclable(
    producer_terminal: bool,
    consumer_terminal: Option<bool>,
) -> bool {
    producer_terminal && consumer_terminal.unwrap_or(true)
}

#[cfg(kani)]
#[kani::proof]
fn hot_workspace_never_recycles_a_live_or_unsubmitted_consumer() {
    let producer: bool = kani::any();
    let consumer: Option<bool> = kani::any();
    let reusable = workspace_recyclable(producer, consumer);
    assert_eq!(reusable, producer && consumer != Some(false));
    kani::cover!(reusable && consumer.is_none());
    kani::cover!(reusable && consumer == Some(true));
    kani::cover!(!reusable && producer && consumer == Some(false));
}

#[test]
fn workspace_rejects_wrong_extents_and_storage() {
    for (producer, consumer, expected) in [
        (false, None, false),
        (false, Some(false), false),
        (false, Some(true), false),
        (true, None, true),
        (true, Some(false), false),
        (true, Some(true), true),
    ] {
        assert_eq!(workspace_recyclable(producer, consumer), expected);
    }

    let sizes = [102400, 102400, 12800, 12800, 51200];
    for slab in [false, true] {
        let modes = [true, true, true, true, slab];
        assert!(workspace_valid(sizes, modes, slab));
        for i in 0..5 {
            let mut bad = sizes;
            bad[i] += 1;
            assert!(!workspace_valid(bad, modes, slab));
            let mut bad = modes;
            bad[i] = !bad[i];
            assert!(!workspace_valid(sizes, bad, slab));
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn hot_workspace_requires_exact_extents_and_storage() {
    let sizes: [u64; 5] = kani::any();
    let modes: [bool; 5] = kani::any();
    let slab: bool = kani::any();
    let ok = workspace_valid(sizes, modes, slab);
    assert_eq!(
        ok,
        sizes == [102400, 102400, 12800, 12800, 51200] && modes == [true, true, true, true, slab]
    );
    kani::cover!(ok && slab);
    kani::cover!(ok && !slab);
}

/// Qualified default and strict overrides; unsupported hardware always falls back.
pub(crate) fn flag(value: Option<&str>, qualified_m5: bool, tier2: bool) -> Option<bool> {
    crate::ar10_defaults::flag(value, qualified_m5).map(|enabled| enabled && tier2)
}

pub(crate) fn admitted(
    gpu_available: bool,
    slots: usize,
    direct: bool,
    depth: usize,
    prefetch: bool,
    record_reuse: bool,
) -> bool {
    gpu_available && supported_slots(slots) && direct && depth == 0 && !prefetch && record_reuse
}

pub(crate) fn identity_matches(expert: u32, generation: u64, current: Option<(u32, u64)>) -> bool {
    expert < EXPERTS as u32 && current == Some((expert, generation))
}

pub(crate) fn contribution_offset(route: usize) -> Option<u64> {
    (route < ROUTES).then(|| route as u64 * CONTRIBUTION_BYTES)
}

pub(crate) fn retirement_bytes(records: usize) -> Option<u64> {
    (records <= ROUTES).then(|| records as u64 * RECORD_BYTES)
}

pub(crate) fn checked_peak(retired: usize, auxiliary: u64) -> Option<u64> {
    let records = retirement_bytes(retired)?;
    (auxiliary <= AUXILIARY_LIMIT).then(|| records + auxiliary)
}

pub(crate) fn readers_after(readers: usize, acquire: bool) -> Option<usize> {
    if acquire {
        readers.checked_add(1)
    } else {
        readers.checked_sub(1)
    }
}

pub(crate) fn hot_publishable(producers_complete: bool, hot_error_absent: bool) -> bool {
    producers_complete && hot_error_absent
}

pub(crate) fn pending_route_needs_wait(submitted: bool, terminal: bool) -> bool {
    submitted && !terminal
}

pub(crate) fn pending_route_publishable(submitted: bool, completed: bool) -> bool {
    submitted && completed
}

#[test]
fn device_hot_defaults_require_both_hardware_guards_and_strict_overrides() {
    for m5 in [false, true] {
        for tier2 in [false, true] {
            assert_eq!(flag(None, m5, tier2), Some(m5 && tier2));
            assert_eq!(flag(Some("0"), m5, tier2), Some(false));
            assert_eq!(flag(Some("1"), m5, tier2), Some(m5 && tier2));
            for invalid in ["", "01", "true", " 1", "1\n", "é"] {
                assert_eq!(flag(Some(invalid), m5, tier2), None);
            }
        }
    }
    assert!(!crate::ar10_defaults::FLAGS.contains(&"MLXL3_NATIVE_DEVICE_RESIDENT_ROUTE"));
}

#[test]
fn device_hot_guard_rejects_every_scope_escape_and_bad_identity() {
    assert!(admitted(true, 53, true, 0, false, true));
    assert!(admitted(true, 64, true, 0, false, true));
    for slots in [0, 10, 52, 54, usize::MAX] {
        assert!(!admitted(true, slots, true, 0, false, true));
    }
    assert!(!admitted(false, 53, true, 0, false, true));
    assert!(!admitted(true, 53, false, 0, false, true));
    assert!(!admitted(true, 53, true, 1, false, true));
    assert!(!admitted(true, 53, true, 0, true, true));
    assert!(!admitted(true, 53, true, 0, false, false));
    assert!(identity_matches(511, u64::MAX, Some((511, u64::MAX))));
    assert!(!identity_matches(512, 1, Some((512, 1))));
    assert!(!identity_matches(2, 1, Some((2, 2))));
    assert!(!identity_matches(2, 1, Some((3, 1))));
    assert!(!identity_matches(2, 1, None));
}

#[test]
fn device_hot_bounds_ownership_and_latched_errors_are_explicit() {
    assert_eq!(contribution_offset(9), Some(46_080));
    assert_eq!(contribution_offset(10), None);
    assert_eq!(contribution_offset(usize::MAX), None);
    assert_eq!(checked_peak(10, AUXILIARY_LIMIT), Some(14_577_272));
    assert_eq!(checked_peak(11, 0), None);
    assert_eq!(checked_peak(0, u64::MAX), None);
    assert_eq!(readers_after(0, false), None);
    assert_eq!(readers_after(usize::MAX, true), None);
    assert_eq!(readers_after(1, false), Some(0));
    assert_eq!(readers_after(0, true), Some(1));
    assert!(!hot_publishable(true, false));
    assert!(!hot_publishable(false, true));
    assert!(hot_publishable(true, true));
    assert!(!pending_route_needs_wait(false, false));
    assert!(pending_route_needs_wait(true, false));
    assert!(!pending_route_needs_wait(true, true));
    assert!(!pending_route_publishable(false, true));
    assert!(pending_route_publishable(true, true));
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn device_hot_defaults_preserve_strict_literals_and_both_hardware_guards() {
    let m5: bool = kani::any();
    let tier2: bool = kani::any();
    let present: bool = kani::any();
    let bytes: [u8; 4] = kani::any();
    let length = usize::from(kani::any::<u8>() % 5);
    if !present {
        let result = flag(None, m5, tier2);
        assert_eq!(result, Some(m5 && tier2));
        kani::cover!(m5 && tier2 && result == Some(true));
        kani::cover!(!m5 && tier2 && result == Some(false));
        kani::cover!(m5 && !tier2 && result == Some(false));
    } else if let Ok(value) = std::str::from_utf8(&bytes[..length]) {
        let result = flag(Some(value), m5, tier2);
        assert_eq!(
            result,
            match value {
                "0" => Some(false),
                "1" => Some(m5 && tier2),
                _ => None,
            }
        );
        kani::cover!(value == "1" && m5 && tier2 && result == Some(true));
        kani::cover!(value == "0" && m5 && tier2 && result == Some(false));
        kani::cover!(value == "1" && !m5 && tier2 && result == Some(false));
        kani::cover!(value == "1" && m5 && !tier2 && result == Some(false));
        kani::cover!(value.is_empty() && result.is_none());
        kani::cover!(value == "é" && result.is_none());
    }
}

#[cfg(kani)]
#[kani::proof]
fn device_hot_actual_admission_offsets_and_peak_are_bounded() {
    let gpu: bool = kani::any();
    let slots: usize = kani::any();
    let direct: bool = kani::any();
    let depth: usize = kani::any();
    let prefetch: bool = kani::any();
    let reuse: bool = kani::any();
    let accepted = admitted(gpu, slots, direct, depth, prefetch, reuse);
    assert!(
        accepted == (gpu && matches!(slots, 53 | 64) && direct && depth == 0 && !prefetch && reuse)
    );
    let route: usize = kani::any();
    let offset = contribution_offset(route);
    assert!(offset.is_some() == (route < 10));
    if let Some(offset) = offset {
        assert!(offset + CONTRIBUTION_BYTES <= 51_200);
    }
    let retired: usize = kani::any();
    let auxiliary: u64 = kani::any();
    let peak = checked_peak(retired, auxiliary);
    assert!(peak.is_some() == (retired <= 10 && auxiliary <= AUXILIARY_LIMIT));
    if let Some(peak) = peak {
        assert!(peak <= RESERVE_BYTES);
    }
    let bit = retirement_bit(route, slots);
    assert_eq!(bit.is_some(), matches!(slots, 53 | 64) && route < slots);
    if let Some(bit) = bit {
        assert_eq!(bit.count_ones(), 1);
    }
    kani::cover!(accepted && slots == 53);
    kani::cover!(accepted && slots == 64);
    kani::cover!(slots == 64 && route == 63 && bit == Some(1u64 << 63));
    kani::cover!(!accepted && depth > 0);
    kani::cover!(route == 9 && offset == Some(46_080));
    kani::cover!(retired == 10 && auxiliary == AUXILIARY_LIMIT);
}

#[cfg(kani)]
#[kani::proof]
fn device_hot_actual_identity_readers_and_pending_lifecycle_are_fail_closed() {
    let expert: u32 = kani::any();
    let generation: u64 = kani::any();
    let current: Option<(u32, u64)> = kani::any();
    let matches = identity_matches(expert, generation, current);
    assert!(matches == (expert < 512 && current == Some((expert, generation))));
    let readers: usize = kani::any();
    let acquire: bool = kani::any();
    let next = readers_after(readers, acquire);
    assert!(
        next.is_some()
            == if acquire {
                readers < usize::MAX
            } else {
                readers > 0
            }
    );
    let submitted: bool = kani::any();
    let terminal: bool = kani::any();
    let completed: bool = kani::any();
    let error_absent: bool = kani::any();
    assert!(pending_route_needs_wait(submitted, terminal) == (submitted && !terminal));
    assert!(pending_route_publishable(submitted, completed) == (submitted && completed));
    assert!(!hot_publishable(completed, error_absent) || (completed && error_absent));
    kani::cover!(matches && expert == 511 && generation == u64::MAX);
    kani::cover!(current.is_none() && !matches);
    kani::cover!(submitted && !terminal && !completed);
    kani::cover!(!submitted && completed && !pending_route_publishable(submitted, completed));
}
