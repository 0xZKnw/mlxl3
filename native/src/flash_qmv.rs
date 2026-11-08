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

#[test]
fn verify_rows_dispatch_preserves_scalar_and_unsupported_paths() {
    for eligible in [false, true] {
        for rows in [0, 1, 2, 3, 4, 5, u32::MAX] {
            for nt in [0, 1, 2, 3, u32::MAX] {
                for k in [0, 1, 2, 3, 4, 5, 6, 7, u32::MAX] {
                    let actual = verify_rows_key(rows, nt, k, eligible);
                    let supported = eligible && rows >= 2 && rows <= 4
                        && k >= 2 && k <= 6 && (nt == 1 || nt == 2);
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
    let supported = eligible && rows >= 2 && rows <= 4
        && k >= 2 && k <= 6 && (nt == 1 || nt == 2);
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
