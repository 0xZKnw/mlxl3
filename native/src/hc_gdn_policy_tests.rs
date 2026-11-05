use crate::flash_support::{hc_gdn_chain_eligible, hc_gdn_chain_flag};

#[test]
fn hc_gdn_flag_is_strict_and_defaults_only_to_qualified_hardware() {
    for qualified in [false, true] {
        assert_eq!(hc_gdn_chain_flag(None, qualified), Some(qualified));
        assert_eq!(hc_gdn_chain_flag(Some("0"), qualified), Some(false));
        assert_eq!(hc_gdn_chain_flag(Some("1"), qualified), Some(qualified));
        for value in [
            "", "01", "10", " 1", "1 ", "1\n", "true", "false", "2", "é", "1\0",
        ] {
            assert_eq!(hc_gdn_chain_flag(Some(value), qualified), None);
        }
    }
}

#[test]
fn hc_gdn_eligibility_keeps_every_production_guard_and_shape_boundary() {
    for rows in [0, 1, 2, u32::MAX] {
        for columns in [0, 1, 2560, 10239, 10240, 10241, u32::MAX] {
            for enabled in [false, true] {
                for trace in [false, true] {
                    for same_owner in [false, true] {
                        let expected = (enabled, trace, same_owner, rows, columns)
                            == (true, false, true, 1, 10240);
                        assert_eq!(
                            hc_gdn_chain_eligible(enabled, trace, same_owner, [rows, columns]),
                            expected,
                            "enabled={enabled}, trace={trace}, same_owner={same_owner}, shape={rows}x{columns}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn hc_gdn_trace_or_foreign_owner_never_enters_the_joint_path() {
    assert!(!hc_gdn_chain_eligible(true, true, true, [1, 10240]));
    assert!(!hc_gdn_chain_eligible(true, false, false, [1, 10240]));
    assert!(hc_gdn_chain_eligible(true, false, true, [1, 10240]));
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn hc_gdn_joint_selection_requires_every_guard_for_symbolic_shapes() {
    let enabled: bool = kani::any();
    let trace: bool = kani::any();
    let same_owner: bool = kani::any();
    let shape: [u32; 2] = kani::any();
    let actual = hc_gdn_chain_eligible(enabled, trace, same_owner, shape);
    assert_eq!(
        actual,
        (enabled, trace, same_owner, shape[0], shape[1]) == (true, false, true, 1, 10240)
    );
    kani::cover!(actual);
    kani::cover!(!actual && !enabled && !trace && same_owner && shape == [1, 10240]);
    kani::cover!(!actual && enabled && trace && same_owner && shape == [1, 10240]);
    kani::cover!(!actual && enabled && !trace && !same_owner && shape == [1, 10240]);
    kani::cover!(!actual && enabled && !trace && same_owner && shape == [0, 10240]);
    kani::cover!(!actual && enabled && !trace && same_owner && shape == [2, 10240]);
    kani::cover!(!actual && enabled && !trace && same_owner && shape == [1, 10239]);
    kani::cover!(!actual && enabled && !trace && same_owner && shape == [1, 10241]);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn hc_gdn_flag_preserves_qualified_defaults_and_exact_literals() {
    let qualified: bool = kani::any();
    let present: bool = kani::any();
    let raw: [u8; 4] = kani::any();
    let length = usize::from(kani::any::<u8>() % 5);
    if !present {
        let actual = hc_gdn_chain_flag(None, qualified);
        assert_eq!(actual, Some(qualified));
        kani::cover!(qualified && actual == Some(true));
        kani::cover!(!qualified && actual == Some(false));
    } else if let Ok(value) = std::str::from_utf8(&raw[..length]) {
        let actual = hc_gdn_chain_flag(Some(value), qualified);
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
        kani::cover!(qualified && value == "0" && actual == Some(false));
        kani::cover!(value.is_empty() && actual.is_none());
        kani::cover!(value == "01" && actual.is_none());
    }
}
