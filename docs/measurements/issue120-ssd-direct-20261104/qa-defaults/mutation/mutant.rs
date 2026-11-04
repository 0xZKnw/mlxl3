mod flash_support {
pub(crate) fn hc_chain_flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    match value {
        None => Some(qualified_m5),
        Some("0") => Some(false),
        Some("1") => Some(true),
        _ => None,
    }
}
}
pub(crate) fn direct_flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    crate::flash_support::hc_chain_flag(value, false).map(|enabled| enabled && qualified_m5)
}
#[test]
fn actual_qualified_default_contract() { for qualified in [false,true] { assert_eq!(direct_flag(None, qualified), Some(qualified)); } }
