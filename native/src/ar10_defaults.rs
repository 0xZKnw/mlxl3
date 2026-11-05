//! Defaults for the qualified PR132 composition; other experiments stay independent.
use anyhow::{Context, Result, ensure};

pub(crate) const FLAGS: [&str; 13] = [
    "MLXL3_NATIVE_HC_WRITE_NORM",
    "MLXL3_NATIVE_ROUTED_NT2_GU",
    "MLXL3_NATIVE_ROUTED_NT2_DOWN",
    "MLXL3_NATIVE_EXPERT_UP_FINISH",
    "MLXL3_NATIVE_EXPERT_GU_PREPARE",
    "MLXL3_NATIVE_EXPERT_OUTPUT_SLAB",
    "MLXL3_NATIVE_EXPERT_AXIS_QMV",
    "MLXL3_NATIVE_GDN_WHOLE",
    "MLXL3_NATIVE_GDN_JOINT_AB",
    "MLXL3_NATIVE_QSA_PREPARE",
    "MLXL3_NATIVE_QSA_GATE",
    "MLXL3_NATIVE_RECORD_REUSE",
    "MLXL3_NATIVE_READ_SPLIT",
];

pub(crate) fn flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    crate::flash_support::hc_chain_flag(value, qualified_m5).map(|on| on && qualified_m5)
}

// The request bit preserves the legacy CPU parser; actual Shared dispatch still checks M5.
pub(crate) fn split_flag(value: Option<&str>, qualified_default: bool) -> Option<bool> {
    crate::flash_support::hc_chain_flag(value, qualified_default)
}

fn setting_value(value: std::result::Result<String, std::env::VarError>, m5: bool) -> Result<bool> {
    let value = match value {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    flag(value.as_deref(), m5).context("must be 0 or 1")
}

pub(crate) fn setting(name: &str, qualified_m5: bool) -> Result<bool> {
    ensure!(
        FLAGS.contains(&name),
        "not a qualified AR10 setting: {name}"
    );
    setting_value(std::env::var(name), qualified_m5)
        .with_context(|| format!("{name} must be 0 or 1"))
}

#[test]
fn defaults_match_the_qualified_composition_and_preserve_every_override() {
    let plan: serde_json::Value = serde_json::from_str(include_str!(
        "../../docs/measurements/flash-ar10-20261105/plan-final.json"
    ))
    .unwrap();
    let expected = plan["arms"]
        .as_array()
        .unwrap()
        .iter()
        .find(|arm| arm["name"] == "all")
        .unwrap()["flags"]
        .as_array()
        .unwrap()
        .iter()
        .map(|name| name.as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(FLAGS.as_slice(), expected);
    for m5 in [false, true] {
        assert_eq!(flag(None, m5), Some(m5));
        assert_eq!(
            setting_value(Err(std::env::VarError::NotPresent), m5).unwrap(),
            m5
        );
        assert_eq!(split_flag(None, m5), Some(m5));
        assert_eq!(flag(Some("0"), m5), Some(false));
        assert!(!setting_value(Ok("0".into()), m5).unwrap());
        assert_eq!(setting_value(Ok("1".into()), m5).unwrap(), m5);
        assert_eq!(split_flag(Some("0"), m5), Some(false));
        assert_eq!(flag(Some("1"), m5), Some(m5));
        assert_eq!(split_flag(Some("1"), m5), Some(true));
        for bad in ["", "01", "true", " 1", "1\n", "é"] {
            assert_eq!(flag(Some(bad), m5), None);
            assert_eq!(split_flag(Some(bad), m5), None);
            assert!(setting_value(Ok(bad.into()), m5).is_err());
        }
        assert!(setting_value(Err(std::env::VarError::NotUnicode("invalid".into())), m5).is_err());
    }
    for other in [
        "MLXL3_NATIVE_HC_UP_MIX",
        "MLXL3_NATIVE_GDN_HALF_ELISION",
        "MLXL3_NATIVE_SSD_EXTRA_WAVE",
    ] {
        assert!(!FLAGS.contains(&other));
        assert!(setting(other, true).is_err());
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn qualified_defaults_preserve_strict_overrides_and_legacy_io() {
    let bytes: [u8; 4] = kani::any();
    let length = usize::from(kani::any::<u8>() % 5);
    let m5: bool = kani::any();
    let present: bool = kani::any();
    if !present {
        assert_eq!(flag(None, m5), Some(m5));
        assert_eq!(split_flag(None, m5), Some(m5));
        kani::cover!(m5);
        kani::cover!(!m5);
    } else if let Ok(value) = std::str::from_utf8(&bytes[..length]) {
        assert_eq!(
            flag(Some(value), m5),
            match value {
                "0" => Some(false),
                "1" => Some(m5),
                _ => None,
            }
        );
        assert_eq!(
            split_flag(Some(value), m5),
            match value {
                "0" => Some(false),
                "1" => Some(true),
                _ => None,
            }
        );
        kani::cover!(value == "1" && m5);
        kani::cover!(value == "1" && !m5);
        kani::cover!(value == "0");
        kani::cover!(value == "é");
    }
}
