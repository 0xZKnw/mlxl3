//! Checked identities for optional native Flash K/codebook specializations.

pub(crate) struct Spec {
    pub(crate) k: u32,
    pub(crate) nt: u32,
}

pub(crate) const SPECS: [Spec; 5] = [
    Spec { k: 2, nt: 1 },
    Spec { k: 3, nt: 1 },
    Spec { k: 3, nt: 2 },
    Spec { k: 4, nt: 1 },
    Spec { k: 4, nt: 2 },
];

pub(crate) fn routed_nt2_key(
    shape: [u32; 2],
    splits: u32,
    format: [u32; 3],
    routed: bool,
    gu: bool,
    down: bool,
) -> Option<usize> {
    if !routed || splits != 1 || format != [2, 2, 1] {
        return None;
    }
    match shape {
        [2560, 640] if gu => Some(0),
        [640, 2560] if down => Some(1),
        _ => None,
    }
}

#[test]
fn routed_nt2_preserves_records_shapes_and_independent_stage_controls() {
    for routed in [false, true] {
        for gu in [false, true] {
            for down in [false, true] {
                for (shape, key, selected) in [([2560, 640], 0, gu), ([640, 2560], 1, down)] {
                    assert_eq!(
                        routed_nt2_key(shape, 1, [2, 2, 1], routed, gu, down),
                        (routed && selected).then_some(key)
                    );
                    assert!(routed_nt2_key(shape, 2, [2, 2, 1], routed, gu, down).is_none());
                    assert!(routed_nt2_key(shape, 1, [3, 2, 1], routed, gu, down).is_none());
                }
                assert!(routed_nt2_key([2560, 641], 1, [2, 2, 1], routed, gu, down).is_none());
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn routed_nt2_requires_exact_records_and_covers_the_complete_output() {
    let shape: [u32; 2] = kani::any();
    let splits: u32 = kani::any();
    let format: [u32; 3] = kani::any();
    let routed: bool = kani::any();
    let gu: bool = kani::any();
    let down: bool = kani::any();
    let key = routed_nt2_key(shape, splits, format, routed, gu, down);
    assert_eq!(
        key,
        if routed && splits == 1 && format == [2, 2, 1] {
            if shape == [2560, 640] && gu {
                Some(0)
            } else if shape == [640, 2560] && down {
                Some(1)
            } else {
                None
            }
        } else {
            None
        }
    );
    if key.is_some() {
        let groups = crate::contracts::flash_reference_groups(shape[1], 2).unwrap();
        assert_eq!(groups * 32, shape[1]);
        assert!(groups > 0);
    }
    kani::cover!(key == Some(0));
    kani::cover!(key == Some(1));
    kani::cover!(!routed && key.is_none());
}

pub(crate) fn flag(value: Option<&str>) -> Option<bool> {
    match value {
        None | Some("1") => Some(true),
        Some("0") => Some(false),
        Some(_) => None,
    }
}

/// Called at projection load, using the immutable context setting. An invalid
/// setting only rejects shapes/families for which a specialization exists.
pub(crate) fn select(
    k: u32,
    cb: u32,
    nt: u32,
    eligible: bool,
    setting: Option<bool>,
) -> Result<Option<usize>, ()> {
    if !eligible || cb != 2 {
        return Ok(None);
    }
    let key = match (k, nt) {
        (2, 1) => 0,
        (3, 1) => 1,
        (3, 2) => 2,
        (4, 1) => 3,
        (4, 2) => 4,
        _ => return Ok(None),
    };
    match setting {
        Some(true) => Ok(Some(key)),
        Some(false) => Ok(None),
        None => Err(()),
    }
}

#[test]
fn static_qmv_default_flags_and_each_pso_identity() {
    for (value, expected) in [
        (None, Some(true)),
        (Some("0"), Some(false)),
        (Some("1"), Some(true)),
        (Some(""), None),
        (Some("true"), None),
        (Some(" 1"), None),
        (Some("01"), None),
        (Some("1\n"), None),
        (Some("é"), None),
    ] {
        assert_eq!(flag(value), expected);
    }
    for (key, spec) in SPECS.iter().enumerate() {
        assert_eq!(select(spec.k, 2, spec.nt, true, Some(true)), Ok(Some(key)));
        assert_eq!(select(spec.k, 2, spec.nt, true, Some(false)), Ok(None));
        assert_eq!(select(spec.k, 2, spec.nt, true, None), Err(()));
        assert_eq!(select(spec.k, 2, spec.nt, false, None), Ok(None));
    }
    for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, u32::MAX] {
        for nt in [0, 1, 2, 3, 4, u32::MAX] {
            for cb in [0, 1, 2, 3, u32::MAX] {
                let expected = cb == 2 && matches!((k, nt), (2, 1) | (3 | 4, 1 | 2));
                let actual = select(k, cb, nt, true, Some(true)).unwrap();
                assert_eq!(actual.is_some(), expected);
                assert_eq!(select(k, cb, nt, false, None), Ok(None));
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn native_static_qmv_key_matches_the_actual_pso_table_and_every_fallback() {
    let k: u32 = kani::any();
    let cb: u32 = kani::any();
    let nt: u32 = kani::any();
    let eligible: bool = kani::any();
    let setting = match kani::any::<u8>() % 3 {
        0 => None,
        1 => Some(false),
        _ => Some(true),
    };
    let actual = select(k, cb, nt, eligible, setting);
    let supported = eligible && cb == 2 && matches!((k, nt), (2, 1) | (3 | 4, 1 | 2));
    assert_eq!(actual.is_err(), supported && setting.is_none());
    if let Ok(Some(key)) = actual {
        assert!(key < SPECS.len());
        assert_eq!(SPECS[key].k, k);
        assert_eq!(SPECS[key].nt, nt);
        assert_eq!(cb, 2);
        assert!(eligible && setting == Some(true));
        kani::cover!(key == 0);
        kani::cover!(key == 1);
        kani::cover!(key == 2);
        kani::cover!(key == 3);
        kani::cover!(key == 4);
    } else {
        assert!(!supported || setting != Some(true));
        kani::cover!(!eligible && setting.is_none());
        kani::cover!(k == 7 && cb == 2 && setting == Some(true));
        kani::cover!(supported && setting.is_none());
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn native_static_qmv_flag_preserves_default_and_bounded_utf8_literals() {
    let bytes: [u8; 4] = kani::any();
    let length = usize::from(kani::any::<u8>() % 5);
    let present: bool = kani::any();
    if !present {
        assert_eq!(flag(None), Some(true));
        kani::cover!(!present);
    } else if let Ok(value) = std::str::from_utf8(&bytes[..length]) {
        let expected = match (length, bytes[0]) {
            (1, 0x30) => Some(false),
            (1, 0x31) => Some(true),
            _ => None,
        };
        assert_eq!(flag(Some(value)), expected);
        kani::cover!(value == "0");
        kani::cover!(value == "1");
        kani::cover!(value.is_empty());
        kani::cover!(value == "é");
    }
}

pub(crate) const FIXED_SHAPES: [[u32; 2]; 2] = [[2560, 640], [640, 2560]];

/// The complete key is checked once for resident and cold-backed projections.
pub(crate) fn fixed_select(
    shape: [u32; 2],
    splits: u32,
    format: [u32; 3],
    eligible: bool,
    setting: Option<bool>,
) -> Result<Option<usize>, ()> {
    if !eligible || splits != 1 || format != [2, 2, 1] {
        return Ok(None);
    }
    let key = match shape {
        [2560, 640] => 0,
        [640, 2560] => 1,
        _ => return Ok(None),
    };
    match setting {
        Some(true) => Ok(Some(key)),
        Some(false) => Ok(None),
        None => Err(()),
    }
}

#[test]
fn fixed_qmv_shapes_splits_and_fallbacks_are_exact() {
    for (key, shape) in FIXED_SHAPES.into_iter().enumerate() {
        assert_eq!(
            fixed_select(shape, 1, [2, 2, 1], true, Some(true)),
            Ok(Some(key))
        );
        assert_eq!(
            fixed_select(shape, 1, [2, 2, 1], true, Some(false)),
            Ok(None)
        );
        assert_eq!(fixed_select(shape, 1, [2, 2, 1], true, None), Err(()));
        assert_eq!(fixed_select(shape, 1, [2, 2, 1], false, None), Ok(None));
        for axis in 0..2 {
            for value in [0, 1, shape[axis] - 1, shape[axis] + 1, u32::MAX] {
                let mut other = shape;
                other[axis] = value;
                assert_eq!(fixed_select(other, 1, [2, 2, 1], true, None), Ok(None));
            }
        }
        for splits in [0, 2, 3, u32::MAX] {
            assert_eq!(fixed_select(shape, splits, [2, 2, 1], true, None), Ok(None));
        }
        for axis in 0..3 {
            for value in [0, 1, 3, 4, 7, u32::MAX] {
                let mut format = [2, 2, 1];
                if format[axis] == value {
                    continue;
                }
                format[axis] = value;
                assert_eq!(fixed_select(shape, 1, format, true, None), Ok(None));
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn native_fixed_qmv_key_checks_complete_shape_split_and_format() {
    let shape: [u32; 2] = kani::any();
    let splits: u32 = kani::any();
    let format: [u32; 3] = kani::any();
    let eligible: bool = kani::any();
    let setting = match kani::any::<u8>() % 3 {
        0 => None,
        1 => Some(false),
        _ => Some(true),
    };
    let actual = fixed_select(shape, splits, format, eligible, setting);
    let supported = eligible
        && splits == 1
        && format == [2, 2, 1]
        && matches!(shape, [2560, 640] | [640, 2560]);
    assert_eq!(actual.is_err(), supported && setting.is_none());
    if let Ok(Some(key)) = actual {
        assert!(key < FIXED_SHAPES.len());
        assert_eq!(FIXED_SHAPES[key], shape);
        assert!(splits == 1 && format == [2, 2, 1] && eligible && setting == Some(true));
        kani::cover!(key == 0);
        kani::cover!(key == 1);
    } else {
        assert!(!supported || setting != Some(true));
        kani::cover!(supported && setting.is_none());
        kani::cover!(supported && setting == Some(false));
        kani::cover!(!eligible && setting.is_none());
        kani::cover!(format[0] == 7 && setting == Some(true));
        kani::cover!(splits == 2 && setting == Some(true));
    }
}

/// Optional canonical MB2 target path. B3/B4 use two groups with row-tail guards.
/// The caller excludes generic, routed, and special MTP-hidden projections.
pub(crate) fn verify_rows_key(rows: u32, nt: u32, k: u32, eligible: bool) -> Option<usize> {
    if eligible && (2..=4).contains(&rows) && (2..=6).contains(&k) {
        match nt {
            1 => Some(0),
            2 => Some(1),
            _ => None,
        }
    } else {
        None
    }
}

pub(crate) fn verify_row_count(rows: usize, nt: u32, k: u32, eligible: bool) -> bool {
    (2..=4).contains(&rows) && verify_rows_key(rows as u32, nt, k, eligible).is_some()
}

/// MB2 reuses the qualified B1 constants; fixed dimensions take priority.
pub(crate) fn verify_specialized_key(
    rows: u32,
    static_key: Option<usize>,
    fixed_key: Option<u8>,
    enabled: bool,
) -> Option<usize> {
    if !enabled || !(2..=4).contains(&rows) {
        return None;
    }
    match fixed_key {
        Some(key) if key < 2 => Some(5 + usize::from(key)),
        Some(_) => None,
        None => static_key.filter(|&key| key < 5),
    }
}

#[test]
fn verify_specializations_preserve_keys_priority_and_all_fallbacks() {
    for rows in 0..=5 {
        for enabled in [false, true] {
            for key in 0..7 {
                assert_eq!(
                    verify_specialized_key(rows, Some(key), None, enabled),
                    (enabled && (2..=4).contains(&rows) && key < 5).then_some(key)
                );
            }
            for key in 0..=2 {
                assert_eq!(
                    verify_specialized_key(rows, Some(0), Some(key), enabled),
                    (enabled && (2..=4).contains(&rows) && key < 2).then_some(5 + usize::from(key))
                );
            }
        }
    }
    assert_eq!(
        verify_specialized_key(2, Some(usize::MAX), None, true),
        None
    );
    assert_eq!(verify_specialized_key(2, None, None, true), None);
}

#[cfg(kani)]
#[kani::proof]
fn verify_specialization_keys_preserve_admission_and_table_bounds() {
    let rows: u32 = kani::any();
    let plain: Option<usize> = kani::any();
    let fixed: Option<u8> = kani::any();
    let enabled: bool = kani::any();
    let selected = verify_specialized_key(rows, plain, fixed, enabled);
    let expected = if enabled && rows >= 2 && rows <= 4 {
        match fixed {
            Some(0) => Some(5),
            Some(1) => Some(6),
            Some(_) => None,
            None => match plain {
                Some(0..=4) => plain,
                _ => None,
            },
        }
    } else {
        None
    };
    assert_eq!(selected, expected);
    if let Some(key) = selected {
        assert!(key < 7);
    }
    kani::cover!(selected == Some(5));
    kani::cover!(selected == Some(6) && rows == 3);
    kani::cover!(selected == Some(4) && rows == 4);
    kani::cover!(selected.is_none() && !enabled && fixed == Some(0));
    kani::cover!(selected.is_none() && rows == 1);
}

#[test]
fn verify_row_count_rejects_wide_host_counts_before_narrowing() {
    for rows in [0, 1, 5, usize::MAX, (u32::MAX as usize).saturating_add(3)] {
        assert!(!verify_row_count(rows, 1, 2, true));
    }
    for rows in 2..=4 {
        assert!(verify_row_count(rows, 1, 2, true));
        assert!(!verify_row_count(rows, 1, 2, false));
        assert!(!verify_row_count(rows, 4, 2, true));
    }
}

#[cfg(kani)]
#[kani::proof]
fn verify_host_row_count_preserves_bounded_projection_dispatch() {
    let rows: usize = kani::any();
    let nt: u32 = kani::any();
    let k: u32 = kani::any();
    let enabled: bool = kani::any();
    let actual = verify_row_count(rows, nt, k, enabled);
    assert_eq!(
        actual,
        rows >= 2 && rows <= 4 && enabled && k >= 2 && k <= 6 && (nt == 1 || nt == 2)
    );
    kani::cover!(actual && rows == 3 && nt == 2);
    kani::cover!(!actual && rows > u32::MAX as usize);
}

#[test]
fn verify_rows_dispatch_preserves_scalar_and_unsupported_paths() {
    for eligible in [false, true] {
        for rows in [0, 1, 2, 3, 4, 5, u32::MAX] {
            for nt in [0, 1, 2, 3, u32::MAX] {
                for k in [0, 1, 2, 3, 4, 5, 6, 7, u32::MAX] {
                    let actual = verify_rows_key(rows, nt, k, eligible);
                    let supported = eligible
                        && (2..=4).contains(&rows)
                        && (2..=6).contains(&k)
                        && (nt == 1 || nt == 2);
                    assert_eq!(actual.is_some(), supported);
                    if let Some(key) = actual {
                        assert_eq!(key, (nt - 1) as usize);
                        assert!(rows.div_ceil(2) * 2 >= rows);
                        assert!(rows.div_ceil(2) <= 2);
                    }
                }
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn verify_rows_dispatch_is_bounded_and_opt_in() {
    let rows: u32 = kani::any();
    let nt: u32 = kani::any();
    let k: u32 = kani::any();
    let eligible: bool = kani::any();
    let actual = verify_rows_key(rows, nt, k, eligible);
    let supported = eligible && rows >= 2 && rows <= 4 && k >= 2 && k <= 6 && (nt == 1 || nt == 2);
    assert_eq!(actual.is_some(), supported);
    if let Some(key) = actual {
        assert!(key < 2);
        assert_eq!(key, (nt - 1) as usize);
        assert!(rows.div_ceil(2) * 2 >= rows);
        assert!(rows.div_ceil(2) <= 2);
        kani::cover!(rows == 3 && nt == 2);
        kani::cover!(rows == 4 && nt == 1);
    }
}

// A campaign may disable the optional path even when its pipelines are loaded.
// Packing and dispatch must use this same decision, not pipeline presence alone.
pub(crate) fn shared_verify_enabled(pipelines_loaded: bool) -> bool {
    #[cfg(test)]
    let allowed = SHARED_VERIFY_OVERRIDE.get();
    #[cfg(not(test))]
    let allowed = true;
    pipelines_loaded && allowed
}

#[cfg(test)]
thread_local! {
    static SHARED_VERIFY_OVERRIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

#[cfg(test)]
pub(crate) struct SharedVerifyOverride {
    previous: bool,
    // The override belongs to its creating thread, including during unwinding.
    _thread: std::marker::PhantomData<std::rc::Rc<()>>,
}

#[cfg(test)]
impl SharedVerifyOverride {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            previous: SHARED_VERIFY_OVERRIDE.replace(enabled),
            _thread: std::marker::PhantomData,
        }
    }
}

#[cfg(test)]
impl Drop for SharedVerifyOverride {
    fn drop(&mut self) {
        SHARED_VERIFY_OVERRIDE.set(self.previous);
    }
}

#[test]
fn shared_verify_override_restores_nested_and_error_scopes() {
    assert!(shared_verify_enabled(true));
    {
        let _disabled = SharedVerifyOverride::new(false);
        assert!(!shared_verify_enabled(true));
        let fail = || -> Result<(), ()> {
            let _enabled = SharedVerifyOverride::new(true);
            assert!(shared_verify_enabled(true));
            assert!(!shared_verify_enabled(false));
            Err(())
        };
        assert!(fail().is_err());
        assert!(!shared_verify_enabled(true));
    }
    assert!(shared_verify_enabled(true));
    assert!(!shared_verify_enabled(false));
    let _ = std::panic::catch_unwind(|| {
        let _disabled = SharedVerifyOverride::new(false);
        panic!("exercise unwinding restoration");
    });
    assert!(shared_verify_enabled(true));
}

#[cfg(kani)]
#[kani::proof]
fn shared_verify_requires_loaded_pipelines() {
    let loaded: bool = kani::any();
    assert_eq!(shared_verify_enabled(loaded), loaded);
    kani::cover!(loaded);
    kani::cover!(!loaded);
}

#[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
thread_local! {
    static SHARED_PATH_COUNTS: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

#[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
pub(crate) fn note_shared_path(packing: bool) {
    let (packed, dispatched) = shared_path_counts();
    SHARED_PATH_COUNTS.set(if packing {
        (packed + 1, dispatched)
    } else {
        (packed, dispatched + 1)
    });
}

#[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
pub(crate) fn shared_path_counts() -> (u64, u64) {
    SHARED_PATH_COUNTS.get()
}

/// Test candidate only: preserve MB2 everywhere except an actual three-row block.
#[cfg(any(test, kani))]
pub(crate) fn verify_geometry(rows: u32, mb3: bool) -> (u32, u32) {
    let mb = if mb3 && rows == 3 { 3 } else { 2 };
    (mb, rows.div_ceil(mb))
}

#[test]
fn third_row_geometry_never_pads_a_fourth_row_or_changes_other_widths() {
    for rows in 0..=4 {
        assert_eq!(verify_geometry(rows, false), (2, rows.div_ceil(2)));
        assert_eq!(
            verify_geometry(rows, true),
            if rows == 3 {
                (3, 1)
            } else {
                (2, rows.div_ceil(2))
            }
        );
    }
}

#[cfg(kani)]
#[kani::proof]
fn third_row_geometry_covers_exactly_the_input_without_overflow() {
    let rows: u32 = kani::any();
    let enabled: bool = kani::any();
    let (mb, groups) = verify_geometry(rows, enabled);
    assert!(mb == 2 || (mb == 3 && rows == 3 && enabled));
    assert!(u64::from(groups) * u64::from(mb) >= u64::from(rows));
    if groups > 0 {
        assert!(u64::from(groups - 1) * u64::from(mb) < u64::from(rows));
    }
    if rows == 3 && enabled {
        assert_eq!((mb, groups), (3, 1));
    }
    kani::cover!(rows == 3 && enabled && groups == 1);
    kani::cover!(rows == u32::MAX && !enabled);
}
