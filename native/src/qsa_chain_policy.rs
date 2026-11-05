//! AR36 command-boundary guards; numerical QSA kernels remain unchanged.

pub(crate) fn setting(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    match value {
        None | Some("0") => Some(false),
        Some("1") => Some(qualified_m5),
        _ => None,
    }
}

pub(crate) fn append_setting(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    setting(value.or(Some("1")), qualified_m5)
}

pub(crate) fn inline_eligible(indexed: bool, trace: bool, offset: u32) -> bool {
    !trace
        && offset
            .checked_add(1)
            .is_some_and(|length| !indexed || length <= 2051)
}

pub(crate) fn append_chain_eligible(
    enabled: bool,
    indexed: bool,
    trace: bool,
    offset: u32,
) -> bool {
    enabled && indexed && inline_eligible(indexed, trace, offset)
}

pub(crate) fn publication_matches(
    current: u32,
    staged: u32,
    same_state: bool,
    completed: bool,
) -> bool {
    same_state && completed && current == staged && staged < crate::flash_support::FLASH_MAX_CONTEXT
}

pub(crate) fn submitted_needs_wait(committed: bool, terminal: bool) -> bool {
    committed && !terminal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qualified_append_default_keeps_explicit_disable_and_strict_literals() {
        for qualified in [false, true] {
            assert_eq!(append_setting(None, qualified), Some(qualified));
            assert_eq!(append_setting(Some("1"), qualified), Some(qualified));
            assert_eq!(append_setting(Some("0"), qualified), Some(false));
            for invalid in ["", "true", " 1", "2", "01", "é"] {
                assert_eq!(append_setting(Some(invalid), qualified), None);
            }
            assert_eq!(setting(None, qualified), Some(false));
        }
    }

    #[test]
    fn qsa_chain_boundaries_keep_sparse_trace_and_unindexed_paths_explicit() {
        assert_eq!(setting(None, true), Some(false));
        assert_eq!(setting(Some("1"), true), Some(true));
        assert_eq!(setting(Some("1"), false), Some(false));
        for invalid in ["", "true", " 1", "2", "01"] {
            assert_eq!(setting(Some(invalid), true), None);
        }
        for offset in [0, 2, 3, 1022, 1023, 2046, 2047, 2048, 2049, 2050] {
            assert!(append_chain_eligible(true, true, false, offset));
        }
        for offset in [2051, 2052, u32::MAX] {
            assert!(!append_chain_eligible(true, true, false, offset));
        }
        assert!(inline_eligible(false, false, 2051));
        assert!(!append_chain_eligible(true, false, false, 0));
        assert!(!append_chain_eligible(true, true, true, 0));
        assert!(!append_chain_eligible(false, true, false, 0));
    }

    #[test]
    fn qsa_publication_and_abandon_require_the_actual_terminal_owner() {
        assert!(publication_matches(2050, 2050, true, true));
        assert!(!publication_matches(2050, 2051, true, true));
        assert!(!publication_matches(0, 0, false, true));
        assert!(!publication_matches(0, 0, true, false));
        assert!(!publication_matches(u32::MAX, u32::MAX, true, true));
        for committed in [false, true] {
            for terminal in [false, true] {
                assert_eq!(
                    submitted_needs_wait(committed, terminal),
                    committed && !terminal
                );
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn qsa_append_default_actual_parser_is_strict_and_hardware_qualified() {
    let qualified: bool = kani::any();
    let present: bool = kani::any();
    let bytes: [u8; 4] = kani::any();
    let length = usize::from(kani::any::<u8>() % 5);
    if !present {
        let result = append_setting(None, qualified);
        assert_eq!(result, Some(qualified));
        assert_eq!(setting(None, qualified), Some(false));
        kani::cover!(qualified && result == Some(true));
        kani::cover!(!qualified && result == Some(false));
    } else if let Ok(value) = std::str::from_utf8(&bytes[..length]) {
        let result = append_setting(Some(value), qualified);
        assert_eq!(
            result,
            match value {
                "0" => Some(false),
                "1" => Some(qualified),
                _ => None,
            }
        );
        kani::cover!(value == "1" && qualified && result == Some(true));
        kani::cover!(value == "0" && qualified && result == Some(false));
        kani::cover!(value == "1" && !qualified && result == Some(false));
        kani::cover!(value.is_empty() && result.is_none());
        kani::cover!(value == "é" && result.is_none());
    }
}

#[cfg(kani)]
#[kani::proof]
fn qsa_append_chain_actual_guard_preserves_dense_cpu_free_boundary() {
    let (enabled, indexed, trace, offset) = (kani::any(), kani::any(), kani::any(), kani::any());
    let chain = append_chain_eligible(enabled, indexed, trace, offset);
    assert_eq!(chain, enabled && indexed && !trace && offset < 2051);
    if chain {
        assert!((offset + 1) / 4 <= 512);
        assert!(inline_eligible(indexed, trace, offset));
    }
    kani::cover!(chain && offset == 3);
    kani::cover!(chain && offset == 2050);
    kani::cover!(!chain && enabled && indexed && !trace && offset == 2051);
    kani::cover!(inline_eligible(false, false, 2051));
}

#[cfg(kani)]
#[kani::proof]
fn qsa_actual_publication_and_abandon_are_fail_closed() {
    let (current, staged, same, completed) = (kani::any(), kani::any(), kani::any(), kani::any());
    let publish = publication_matches(current, staged, same, completed);
    assert_eq!(
        publish,
        same && completed && current == staged && staged < crate::flash_support::FLASH_MAX_CONTEXT
    );
    let (committed, terminal) = (kani::any(), kani::any());
    assert_eq!(
        submitted_needs_wait(committed, terminal),
        committed && !terminal
    );
    kani::cover!(publish && staged == 2050);
    kani::cover!(!publish && same && current == staged && !completed);
    kani::cover!(submitted_needs_wait(committed, terminal));
    kani::cover!(!submitted_needs_wait(committed, terminal) && !committed);
}
