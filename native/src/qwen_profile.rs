//! One explicit profile shared by CLI and bridge model construction.
use anyhow::{Context, Result, bail};

pub const FLAG_NAMES: [&str; 8] = [
    "MLXL3_NATIVE_QWEN_MOE_JOIN",
    "MLXL3_NATIVE_QWEN_LAYER_JOIN",
    "MLXL3_NATIVE_QWEN_LAYER_CHAIN",
    "MLXL3_NATIVE_QWEN_DEVICE_KV",
    "MLXL3_NATIVE_QWEN_PREFILL_LAST_HEAD",
    "MLXL3_NATIVE_QWEN_GROUP_EXPERT_QMV",
    "MLXL3_NATIVE_QWEN_GROUP_EXPERT_STAGES",
    "MLXL3_NATIVE_QWEN_DEVICE_GREEDY",
];

#[derive(Clone, Debug, serde::Serialize)]
pub struct QwenProfile {
    pub name: &'static str,
    pub flags: [bool; 8],
}

impl QwenProfile {
    pub fn resolve(profile: Option<&str>, overrides: [Option<&str>; 8]) -> Result<Self> {
        let (name, optimized) = match profile {
            None | Some("reference") => ("reference", false),
            Some("optimized") => ("optimized", true),
            Some(_) => bail!("MLXL3_NATIVE_QWEN_PROFILE must be reference or optimized"),
        };
        let mut flags = [false; 8];
        for i in 0..8 {
            // DEVICE_GREEDY is not part of the historically qualified composition.
            flags[i] = match overrides[i] {
                None => optimized && i != 7,
                Some("0") => false,
                Some("1") => true,
                Some(_) => bail!("{} must be 0 or 1", FLAG_NAMES[i]),
            };
        }
        Ok(Self { name, flags })
    }

    pub fn from_environment() -> Result<Self> {
        let read = |key| match std::env::var(key) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(error) => Err(error).with_context(|| format!("invalid {key}")),
        };
        let profile = read("MLXL3_NATIVE_QWEN_PROFILE")?;
        let mut overrides: [Option<String>; 8] = Default::default();
        for (i, name) in FLAG_NAMES.iter().enumerate() {
            overrides[i] = read(name)?;
        }
        Self::resolve(
            profile.as_deref(),
            std::array::from_fn(|i| overrides[i].as_deref()),
        )
    }

    /// Dependencies are resolved after strict parsing, including disabled branches.
    pub fn effective(&self, device_greedy: bool, device_kv: bool) -> [bool; 8] {
        let mut flags = self.flags;
        for i in 2..8 {
            flags[i] &= flags[1];
        }
        flags[3] &= device_kv;
        flags[6] &= flags[5];
        flags[7] &= device_greedy;
        flags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_defaults_overrides_and_invalid_disabled_flags() {
        assert_eq!(
            QwenProfile::resolve(None, [None; 8]).unwrap().flags,
            [false; 8]
        );
        let optimized = QwenProfile::resolve(Some("optimized"), [None; 8]).unwrap();
        assert_eq!(
            optimized.flags,
            [true, true, true, true, true, true, true, false]
        );
        for i in 0..8 {
            let mut overrides = [None; 8];
            overrides[i] = Some("0");
            assert!(
                !QwenProfile::resolve(Some("optimized"), overrides)
                    .unwrap()
                    .flags[i]
            );
            overrides[i] = Some("1");
            assert!(QwenProfile::resolve(None, overrides).unwrap().flags[i]);
            for invalid in ["", "true", " 1", "01", "2"] {
                overrides[i] = Some(invalid);
                assert!(QwenProfile::resolve(None, overrides).is_err());
            }
        }
        assert!(QwenProfile::resolve(Some("auto"), [None; 8]).is_err());
    }

    #[test]
    fn profile_dependencies_exhaust_every_flag_combination() {
        for mask in 0..256 {
            let p = QwenProfile {
                name: "reference",
                flags: std::array::from_fn(|i| mask & (1 << i) != 0),
            };
            for greedy in [false, true] {
                for kv in [false, true] {
                    let f = p.effective(greedy, kv);
                    assert_eq!(f[0], p.flags[0]);
                    assert_eq!(f[1], p.flags[1]);
                    for i in [2, 4, 5] {
                        assert_eq!(f[i], p.flags[i] && p.flags[1]);
                    }
                    assert_eq!(f[3], p.flags[3] && p.flags[1] && kv);
                    assert_eq!(f[6], p.flags[6] && p.flags[5] && p.flags[1]);
                    assert_eq!(f[7], p.flags[7] && p.flags[1] && greedy);
                }
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(9)]
fn qwen_profile_preserves_overrides_and_dependencies() {
    let p = QwenProfile {
        name: "reference",
        flags: kani::any(),
    };
    let (greedy, kv) = kani::any();
    let f = p.effective(greedy, kv);
    assert_eq!(f[0], p.flags[0]);
    assert_eq!(f[1], p.flags[1]);
    assert!(!f[3] || (kv && f[1]));
    assert!(!f[6] || (f[5] && f[1]));
    assert!(!f[7] || (greedy && f[1]));
    for i in 0..8 {
        assert!(!f[i] || p.flags[i]);
    }
    kani::cover!(f[3] && f[6] && f[7]);
    kani::cover!(!f[1] && p.flags[6] && !f[6]);
}
