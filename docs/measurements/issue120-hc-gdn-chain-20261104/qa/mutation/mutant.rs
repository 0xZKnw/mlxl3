mod flash_support {
pub(crate) fn hc_chain_flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    match value {
        None => Some(qualified_m5),
        Some("0") => Some(false),
        Some("1") => Some(true),
        _ => None,
    }
}
pub(crate) fn hc_gdn_chain_flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    hc_chain_flag(value, false).map(|enabled| enabled && qualified_m5)
}
pub(crate) fn hc_gdn_chain_eligible(
    enabled: bool,
    trace: bool,
    same_owner: bool,
    shape: [u32; 2],
) -> bool {
    enabled  && same_owner && shape == [1, 10240]
}
}

#[path = "/Users/justin/.codex/worktrees/issue120-hc-gdn-chain/mix-stq1_0/native/src/hc_gdn_policy_tests.rs"]
mod policy_tests;
