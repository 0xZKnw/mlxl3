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

pub(crate) fn flag(value: Option<&str>) -> Option<bool> {
    match value {
        None | Some("0") => Some(false),
        Some("1") => Some(true),
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
        (None, Some(false)),
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
        assert_eq!(flag(None), Some(false));
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
