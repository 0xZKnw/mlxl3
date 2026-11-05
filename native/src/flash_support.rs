//! CPU contracts for Flash Next's disk rows and exact expert cache.
use anyhow::Result;

pub const PREFILL_CHUNK: usize = 8;

#[cfg(any(
    all(target_os = "macos", feature = "direct-metal", feature = "chat"),
    test,
    kani
))]
pub(crate) fn prefill_head_required(row: usize, rows: usize, headonly: bool) -> Option<bool> {
    (row < rows).then(|| !headonly || row == rows - 1)
}

#[test]
fn prefill_heads_keep_the_final_row_of_every_call() {
    for rows in [0, 1, 2, 3, 9, usize::MAX] {
        for row in [0, 1, 2, 8, usize::MAX - 1, usize::MAX] {
            for headonly in [false, true] {
                let expected = if row >= rows {
                    None
                } else {
                    Some(!headonly || row as u128 + 1 == rows as u128)
                };
                assert_eq!(prefill_head_required(row, rows, headonly), expected);
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn prefill_heads_never_skip_a_valid_final_row_or_accept_an_invalid_row() {
    let (row, rows, headonly): (usize, usize, bool) = kani::any();
    let result = prefill_head_required(row, rows, headonly);
    if row >= rows {
        assert_eq!(result, None);
    } else {
        let final_row = row as u128 + 1 == rows as u128;
        assert_eq!(result, Some(!headonly || final_row));
        kani::cover!(rows == 1 && headonly && result == Some(true));
        kani::cover!(rows > 1 && headonly && result == Some(false));
        kani::cover!(row > 0 && !headonly && !final_row && result == Some(true));
        kani::cover!(rows == usize::MAX && final_row);
    }
    kani::cover!(rows == 0 && result.is_none());
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn hc_chain_flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    match value {
        None => Some(qualified_m5),
        Some("0") => Some(false),
        Some("1") => Some(true),
        _ => None,
    }
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn hc_gdn_chain_flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    hc_chain_flag(value, qualified_m5).map(|enabled| enabled && qualified_m5)
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn hc_gdn_chain_eligible(
    enabled: bool,
    trace: bool,
    same_owner: bool,
    shape: [u32; 2],
) -> bool {
    enabled && !trace && same_owner && shape == [1, 10240]
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn cross_layer_eligible(
    enabled: bool,
    hc_gdn: bool,
    depth: usize,
    exporting: bool,
    next_gdn: bool,
) -> bool {
    enabled && hc_gdn && depth == 0 && !exporting && next_gdn
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn defer_moe_completion(enabled: bool, deferred_write: bool) -> bool {
    enabled && deferred_write
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn hc_write_norm_eligible(enabled: bool, fp32: bool, workspace: bool) -> bool {
    enabled && fp32 && !workspace
}

#[test]
fn hc_write_norm_keeps_half_and_workspace_fallbacks() {
    for enabled in [false, true] {
        for fp32 in [false, true] {
            for workspace in [false, true] {
                assert_eq!(
                    hc_write_norm_eligible(enabled, fp32, workspace),
                    (enabled, fp32, workspace) == (true, true, false)
                );
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn hc_write_norm_preserves_the_actual_dtype_and_workspace_guards() {
    let enabled: bool = kani::any();
    let fp32: bool = kani::any();
    let workspace: bool = kani::any();
    let actual = hc_write_norm_eligible(enabled, fp32, workspace);
    assert_eq!(actual, enabled && fp32 && !workspace);
    kani::cover!(actual);
    kani::cover!(enabled && !fp32 && !actual);
    kani::cover!(enabled && fp32 && workspace && !actual);
}

#[test]
fn moe_completion_requires_a_bounded_deferred_consumer() {
    for enabled in [false, true] {
        for deferred_write in [false, true] {
            assert_eq!(
                defer_moe_completion(enabled, deferred_write),
                (enabled, deferred_write) == (true, true)
            );
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn deferred_moe_cannot_escape_the_existing_cross_layer_guard() {
    let enabled: bool = kani::any();
    let deferred_write: bool = kani::any();
    let result = defer_moe_completion(enabled, deferred_write);
    assert_eq!(result, enabled && deferred_write);
    kani::cover!(result);
    kani::cover!(enabled && !deferred_write && !result);
}

#[test]
fn cross_layer_keeps_every_export_depth_and_consumer_fallback() {
    for enabled in [false, true] {
        for hc_gdn in [false, true] {
            for depth in [0, 1, 11, usize::MAX] {
                for exporting in [false, true] {
                    for next_gdn in [false, true] {
                        assert_eq!(
                            cross_layer_eligible(enabled, hc_gdn, depth, exporting, next_gdn),
                            (enabled, hc_gdn, depth, exporting, next_gdn)
                                == (true, true, 0, false, true)
                        );
                    }
                }
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn cross_layer_selection_requires_all_transaction_guards() {
    let (enabled, hc_gdn, depth, exporting, next_gdn): (bool, bool, usize, bool, bool) =
        kani::any();
    let result = cross_layer_eligible(enabled, hc_gdn, depth, exporting, next_gdn);
    assert_eq!(
        result,
        (enabled, hc_gdn, depth, exporting, next_gdn) == (true, true, 0, false, true)
    );
    kani::cover!(result);
    kani::cover!(!enabled && hc_gdn && depth == 0 && !exporting && next_gdn);
    kani::cover!(enabled && !hc_gdn && depth == 0 && !exporting && next_gdn);
    kani::cover!(enabled && hc_gdn && depth == 1 && !exporting && next_gdn);
    kani::cover!(enabled && hc_gdn && depth == 0 && exporting && next_gdn);
    kani::cover!(enabled && hc_gdn && depth == 0 && !exporting && !next_gdn);
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn hc_gate_chain_eligible(
    enabled: bool,
    hc_enabled: bool,
    depth: usize,
    has_prefetcher: bool,
) -> bool {
    enabled && hc_enabled && depth == 0 && !has_prefetcher
}

#[test]
fn hc_gate_chain_keeps_every_fallback_and_default_off() {
    assert_eq!(hc_chain_flag(None, false), Some(false));
    for enabled in [false, true] {
        for hc_enabled in [false, true] {
            for depth in [0, 1, 10, usize::MAX] {
                for has_prefetcher in [false, true] {
                    assert_eq!(
                        hc_gate_chain_eligible(enabled, hc_enabled, depth, has_prefetcher),
                        (enabled, hc_enabled, depth, has_prefetcher) == (true, true, 0, false)
                    );
                }
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn hc_gate_chain_selection_requires_every_production_guard() {
    let enabled: bool = kani::any();
    let hc_enabled: bool = kani::any();
    let depth: usize = kani::any();
    let has_prefetcher: bool = kani::any();
    let selected = hc_gate_chain_eligible(enabled, hc_enabled, depth, has_prefetcher);
    assert_eq!(
        selected,
        (enabled, hc_enabled, depth, has_prefetcher) == (true, true, 0, false)
    );
    assert_eq!(hc_chain_flag(None, false), Some(false));
    kani::cover!(selected);
    kani::cover!(!enabled && hc_enabled && depth == 0 && !has_prefetcher);
    kani::cover!(enabled && !hc_enabled && depth == 0 && !has_prefetcher);
    kani::cover!(enabled && hc_enabled && depth == 1 && !has_prefetcher);
    kani::cover!(enabled && hc_enabled && depth == 0 && has_prefetcher);
}

#[test]
fn hc_chain_flag_preserves_hardware_default_and_strict_overrides() {
    for qualified_m5 in [false, true] {
        assert_eq!(hc_chain_flag(None, qualified_m5), Some(qualified_m5));
        assert_eq!(hc_chain_flag(Some("0"), qualified_m5), Some(false));
        assert_eq!(hc_chain_flag(Some("1"), qualified_m5), Some(true));
        for invalid in ["", "01", " 1", "1 ", "true", "false", "2", "é", "1\0"] {
            assert_eq!(hc_chain_flag(Some(invalid), qualified_m5), None);
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn hc_chain_flag_uses_qualified_hardware_default_and_exact_literals() {
    let qualified_m5: bool = kani::any();
    let present: bool = kani::any();
    let bytes: [u8; 4] = kani::any();
    let size = usize::from(kani::any::<u8>() % 5);
    if !present {
        assert_eq!(hc_chain_flag(None, qualified_m5), Some(qualified_m5));
        kani::cover!(qualified_m5);
        kani::cover!(!qualified_m5);
    } else if let Ok(value) = std::str::from_utf8(&bytes[..size]) {
        let result = hc_chain_flag(Some(value), qualified_m5);
        assert_eq!(
            result,
            match value.as_bytes() {
                [b'0'] => Some(false),
                [b'1'] => Some(true),
                _ => None,
            }
        );
        kani::cover!(result == Some(true));
        kani::cover!(result == Some(false));
        kani::cover!(size == 0 && result.is_none());
        kani::cover!(size == 4 && result.is_none());
        kani::cover!(value == "é" && result.is_none());
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(14)]
fn slot_reads_initialize_exact_disjoint_bank_regions() {
    let slot: usize = kani::any();
    kani::assume(slot < 512); // Both callers obtain slots from ExpertCache, capacity <=512.
    let writes = expert_slot_destinations(slot);
    let widths = [819200, 409600, 10240, 2560, 1280, 5120, 12];
    let mut total = 0;
    for &(bank, _, count) in &writes {
        assert!(bank < 7 && count > 0);
        total += count;
    }
    assert_eq!(total as u64, EXPERT_BYTES);
    for (bank, width) in widths.into_iter().enumerate() {
        let mut end = slot * width;
        for &(destination, start, count) in &writes {
            if destination == bank {
                assert_eq!(start, end);
                end += count;
            }
        }
        assert_eq!(end, (slot + 1) * width);
        assert!(end <= 512 * width);
    }
    kani::cover!(slot == 0);
    kani::cover!(slot == 511);
}

#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal"),
    test,
    kani
))]
pub(crate) fn expert_slot_destinations(slot: usize) -> [(usize, usize, usize); 12] {
    assert!(slot < 512, "expert slot exceeds bank limit");
    [
        (4, slot * 640 * 2, 640 * 2),
        (5, slot * 2560 * 2, 2560 * 2),
        (6, slot * 12, 4),
        (1, slot * 409600, 409600),
        (2, slot * 2 * 2560 * 2, 2560 * 2),
        (3, slot * 2 * 640 * 2, 640 * 2),
        (6, slot * 12 + 4, 4),
        (0, slot * 2 * 409600, 409600),
        (2, (slot * 2 + 1) * 2560 * 2, 2560 * 2),
        (3, (slot * 2 + 1) * 640 * 2, 640 * 2),
        (6, slot * 12 + 8, 4),
        (0, (slot * 2 + 1) * 409600, 409600),
    ]
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn draft_demand_bytes(total: u64, before: u64, after: u64) -> Option<u64> {
    total.checked_add(after.checked_sub(before)?)
}

#[test]
fn draft_demand_counts_deltas_and_rejects_reversal_or_exhaustion() {
    assert_eq!(draft_demand_bytes(0, 10, 17), Some(7));
    assert_eq!(draft_demand_bytes(5, 10, 17), Some(12));
    assert_eq!(draft_demand_bytes(u64::MAX, 10, 10), Some(u64::MAX));
    assert_eq!(draft_demand_bytes(0, u64::MAX, 0), None);
    assert_eq!(draft_demand_bytes(u64::MAX, 0, 1), None);
    assert_eq!(draft_demand_bytes(0, 0, u64::MAX), Some(u64::MAX));
}

#[cfg(kani)]
#[kani::proof]
fn draft_demand_counters_are_exact_or_reject_without_wrapping() {
    let (total, before, after): (u64, u64, u64) = kani::any();
    let result = draft_demand_bytes(total, before, after);
    if after < before {
        assert_eq!(result, None);
        kani::cover!(after == 0 && before == u64::MAX);
    } else {
        let exact = u128::from(total) + u128::from(after) - u128::from(before);
        if exact <= u128::from(u64::MAX) {
            assert_eq!(result.map(u128::from), Some(exact));
            kani::cover!(after > before && total > 0);
            kani::cover!(after == before && total == u64::MAX);
        } else {
            assert_eq!(result, None);
            kani::cover!(total == u64::MAX && after > before);
        }
    }
}

#[cfg(feature = "mlx")]
pub(crate) fn multi_token_verifier() -> Result<bool> {
    match std::env::var("MLXL3_FLASH_MULTI_TOKEN") {
        Ok(value) => parse_gpu_hot_wide(&value)
            .ok_or_else(|| anyhow::anyhow!("MLXL3_FLASH_MULTI_TOKEN must be 0 or 1")),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(error) => Err(error.into()),
    }
}
#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal"),
    test,
    kani
))]
use std::{cell::RefCell, rc::Rc};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheError {
    Capacity,
    InvalidIds,
    Clock,
    NoSlot,
}
impl std::fmt::Display for CacheError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Capacity => "expert cohort exceeds cache capacity (10..512 slots required)",
            Self::InvalidIds => "invalid expert cohort",
            Self::Clock => "expert cache clock exhausted",
            Self::NoSlot => "no unprotected expert slot",
        })
    }
}
impl std::error::Error for CacheError {}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn native_prepare_shape(rows: i32, width: i32) -> bool {
    (1..=20).contains(&rows) && matches!(width, 640 | 2560)
}

#[test]
fn native_prepare_only_accepts_qualified_slot_layouts() {
    for rows in [-1, 0, 1, 2, 19, 20, 21] {
        for width in [128, 639, 640, 641, 2559, 2560, 2561] {
            assert_eq!(
                native_prepare_shape(rows, width),
                rows > 0 && rows <= 20 && [640, 2560].contains(&width)
            );
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn native_prepare_grid_contains_complete_hadamard_groups() {
    let rows: i32 = kani::any();
    let width: i32 = kani::any();
    if native_prepare_shape(rows, width) {
        let elements = rows * width;
        assert!(elements > 0 && elements <= 51200 && elements % 128 == 0);
        kani::cover!(rows == 20 && width == 2560);
        kani::cover!(rows == 1 && width == 640);
    } else {
        kani::cover!(rows == 21);
    }
}

pub const FLASH_MAX_CONTEXT: u32 = 262144;
pub const EXPERT_BYTES: u64 = 1_248_012;
pub const NGRAM_DIM: usize = 160;
pub const NGRAM_WORDS: usize = 41;

#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal"),
    test,
    kani
))]
pub(crate) fn valid_layer_slots(slots: &[usize; 48], uniform: usize) -> bool {
    if !(10..=512).contains(&uniform) {
        return false;
    }
    let mut total = 0usize;
    for &capacity in slots {
        if !(20..=512).contains(&capacity) {
            return false;
        }
        total += capacity;
    }
    total == uniform * 48
}

#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal"),
    test,
    kani
))]
pub(crate) fn parse_layer_slots(value: &str, uniform: usize) -> Option<[usize; 48]> {
    let bytes = value.as_bytes();
    let mut start = 0;
    let mut slots = [0; 48];
    for (index, slot) in slots.iter_mut().enumerate() {
        let mut end = start;
        while end < bytes.len() && bytes[end] != b',' {
            end += 1;
        }
        *slot = parse_slot_capacity(&bytes[start..end])?;
        if index < 47 {
            if end == bytes.len() {
                return None;
            }
            start = end + 1;
        } else if end != bytes.len() {
            return None;
        }
    }
    valid_layer_slots(&slots, uniform).then_some(slots)
}

#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal"),
    test,
    kani
))]
fn parse_slot_capacity(bytes: &[u8]) -> Option<usize> {
    std::str::from_utf8(bytes).ok()?.parse().ok()
}

#[cfg(any(test, kani))]
fn layer_slot_proof_text(suffix: [u8; 2]) -> [u8; 143] {
    let mut text = [0; 143];
    for row in 0..47 {
        text[row * 3..row * 3 + 3].copy_from_slice(b"53,");
    }
    text[141..].copy_from_slice(&suffix);
    text
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn slots_fit(ids: &[u32], capacity: i32) -> bool {
    !ids.is_empty()
        && ids.len() <= i32::MAX as usize
        && capacity > 0
        && ids.iter().all(|&id| id < capacity as u32)
}

/// Original route indices, grouped in chunks of up to four equal expert slots.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn expert_route_groups(ids: &[u32], capacity: i32) -> Option<[Vec<u32>; 4]> {
    if ids.len() > 80 || capacity > 512 || !slots_fit(ids, capacity) {
        return None;
    }
    let mut groups: [Vec<u32>; 4] = std::array::from_fn(|_| Vec::new());
    for (first, &slot) in ids.iter().enumerate() {
        if ids[..first].contains(&slot) {
            continue;
        }
        let mut chunk = [0; 4];
        let mut count = 0;
        for (row, &other) in ids.iter().enumerate() {
            if other == slot {
                chunk[count] = row as u32;
                count += 1;
                if count == 4 {
                    groups[3].extend_from_slice(&chunk);
                    count = 0;
                }
            }
        }
        if count > 0 {
            groups[count - 1].extend_from_slice(&chunk[..count]);
        }
    }
    Some(groups)
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn expert_group_chunks(ids: &[u32]) -> Option<usize> {
    if ids.is_empty() || ids.len() > 80 {
        return None;
    }
    let mut chunks = 0;
    for (row, slot) in ids.iter().enumerate() {
        let prior = ids[..row].iter().filter(|other| *other == slot).count();
        if prior % 4 == 0 {
            chunks += 1;
        }
    }
    Some(chunks)
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn expert_grouping_worthwhile(rows: usize, chunks: usize) -> bool {
    (16..=80).contains(&rows)
        && chunks >= rows.div_ceil(4)
        && chunks <= rows
        && (rows - chunks) * 5 >= rows
}

#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal"),
    test,
    kani
))]
pub(crate) fn parse_lru_override(value: &str) -> Option<bool> {
    match value {
        "auto" => Some(false),
        "lru" => Some(true),
        _ => None,
    }
}

#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal"),
    test,
    kani
))]
pub(crate) fn parse_predictor_half(value: &str) -> Option<bool> {
    match value {
        "f32" => Some(false),
        "f16" => Some(true),
        _ => None,
    }
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn disk_destination_fits(start: usize, length: usize, capacity: usize) -> bool {
    length > 0 && start.checked_add(length).is_some_and(|end| end <= capacity)
}

/// Reserve resident weights, MTP, scratch buffers and room for other applications.
pub fn automatic_cache_slots(available_bytes: u64) -> Option<usize> {
    let budget = available_bytes.checked_sub(8 << 30)?.min(6 << 30);
    let slots = budget / (48 * EXPERT_BYTES);
    (slots >= 10).then_some(slots as usize)
}

#[cfg(target_os = "macos")]
fn available_memory_response_complete(count: libc::mach_msg_type_number_t) -> bool {
    // Older kernels return shorter revisions; only require fields we read.
    let required = (std::mem::offset_of!(libc::vm_statistics64, external_page_count)
        + std::mem::size_of::<libc::natural_t>())
        / std::mem::size_of::<libc::integer_t>();
    (required..=libc::HOST_VM_INFO64_COUNT as usize).contains(&(count as usize))
}

#[cfg(target_os = "macos")]
#[allow(deprecated)] // libc already provides the Mach ABI; no extra dependency.
pub fn available_memory_bytes() -> Result<u64> {
    unsafe extern "C" {
        fn mach_port_deallocate(
            task: libc::mach_port_t,
            name: libc::mach_port_t,
        ) -> libc::kern_return_t;
    }
    let mut info: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
    let mut count = libc::HOST_VM_INFO64_COUNT;
    let host = unsafe { libc::mach_host_self() };
    let status = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            (&mut info as *mut libc::vm_statistics64).cast(),
            &mut count,
        )
    };
    let released = unsafe { mach_port_deallocate(libc::mach_task_self(), host) };
    anyhow::ensure!(
        status == 0 && available_memory_response_complete(count) && released == 0,
        "cannot query available system memory ({status}/{released}, {count} words)"
    );
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    anyhow::ensure!(page_size > 0, "invalid system page size");
    // Speculative file pages are included in both free_count and external_count.
    let pages = u64::from(info.free_count.saturating_sub(info.speculative_count))
        + u64::from(info.external_page_count);
    pages
        .checked_mul(page_size as u64)
        .ok_or_else(|| anyhow::anyhow!("system memory size overflow"))
}

/// Match PonyExl3's single-row accumulation order, including small experts.
pub fn reference_splits(input_tiles: u32, output_tiles: u32) -> u32 {
    let mut splits = 1u32;
    while u64::from(output_tiles) * u64::from(splits) < 8192
        && u64::from(input_tiles) / (u64::from(splits) * 2) >= 128
    {
        splits *= 2;
    }
    splits
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn pair_commit_offset(base: i32, current: i32, retained: i32) -> Option<i32> {
    if base < 0 || !(1..=2).contains(&retained) || base.checked_add(2)? != current {
        return None;
    }
    base.checked_add(retained)
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn expert_cohort_rows(capacity: usize) -> Option<usize> {
    (10..=512)
        .contains(&capacity)
        .then_some((capacity / 10).min(8))
}

/// Largest complete prefix whose distinct experts fit the non-pending slots.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn largest_expert_prefix(ids: &[u32], available: usize) -> Option<usize> {
    if ids.is_empty()
        || ids.len() > 80
        || !ids.len().is_multiple_of(10)
        || available > 512
        || ids.iter().any(|&id| id >= 512)
    {
        return None;
    }
    let mut seen = [0u64; 8];
    let mut unique = 0;
    let mut rows = 0;
    for row in ids.as_chunks::<10>().0 {
        for &id in row {
            let word = &mut seen[(id / 64) as usize];
            let bit = 1u64 << (id % 64);
            if *word & bit == 0 {
                *word |= bit;
                unique += 1;
            }
        }
        if unique > available {
            break;
        }
        rows += 1;
    }
    Some(rows)
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn parse_gpu_hot_wide(value: &str) -> Option<bool> {
    match value {
        "0" => Some(false),
        "1" => Some(true),
        _ => None,
    }
}

#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal", feature = "chat"),
    test,
    kani
))]
pub(crate) fn fixed_draft_width(value: &str, adaptive: bool) -> Option<usize> {
    match (value.as_bytes(), adaptive) {
        ([digit @ b'1'..=b'7'], false) => Some(usize::from(*digit - b'0')),
        _ => None,
    }
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn expert_output_stride(routes: u32, projections: u32, width: u32) -> Option<u32> {
    if routes == 0 || projections == 0 || width == 0 {
        return None;
    }
    routes
        .checked_mul(projections)?
        .checked_mul(width)
        .filter(|&n| n <= i32::MAX as u32)
}

/// Greedy-only scheduler: prefer the smallest prefix on equal expected throughput.
#[cfg(any(test, kani))]
pub(crate) fn draft_prefix(confidence: &[f64], costs: &[f64]) -> Option<usize> {
    if confidence.len() > 3
        || costs.len() != confidence.len() + 1
        || confidence
            .iter()
            .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
        || costs
            .iter()
            .any(|c| !c.is_finite() || !(1e-9..=360.0).contains(c))
    {
        return None;
    }
    let mut survival = 1.;
    let mut expected = 1.;
    let mut best_expected = 1.;
    let mut best = 0;
    for (position, &probability) in confidence.iter().enumerate() {
        survival *= probability;
        expected += survival;
        if expected * costs[best] > best_expected * costs[position + 1] {
            best = position + 1;
            best_expected = expected;
        }
    }
    Some(best)
}

#[cfg(test)]
mod draft_contract_tests {
    use super::*;

    #[test]
    fn expert_output_strides_match_wide_arithmetic_and_reject_empty_or_overflow() {
        for routes in [0, 1, 2, 7, 8, 10, 20, 53, 80, u32::MAX] {
            for projections in [0, 1, 2, u32::MAX] {
                for width in [0, 1, 640, 2560, i32::MAX as u32, u32::MAX] {
                    let wide = u128::from(routes) * u128::from(projections) * u128::from(width);
                    let expected =
                        (routes > 0 && projections > 0 && width > 0 && wide <= i32::MAX as u128)
                            .then_some(wide as u32);
                    assert_eq!(expert_output_stride(routes, projections, width), expected);
                }
            }
        }
    }

    #[test]
    fn draft_prefix_matches_exhaustive_quarter_probability_outcomes() {
        let cost_tables = [
            [1., 2., 3., 4.],
            [1., 1., 1., 1.],
            [1., 1.5, 2., 4.],
            [1., 3., 1.5, 8.],
        ];
        for p0 in 0..=4 {
            for p1 in 0..=4 {
                for p2 in 0..=4 {
                    let probabilities = [p0 as f64 / 4., p1 as f64 / 4., p2 as f64 / 4.];
                    for length in 0..=3 {
                        let mut emitted = [0u32; 4];
                        for draw0 in 0..4 {
                            for draw1 in 0..4 {
                                for draw2 in 0..4 {
                                    let draws = [draw0, draw1, draw2];
                                    for (prefix, total) in
                                        emitted.iter_mut().enumerate().take(length + 1)
                                    {
                                        let accepted = draws[..prefix]
                                            .iter()
                                            .zip([p0, p1, p2])
                                            .take_while(|(draw, p)| **draw < *p)
                                            .count();
                                        *total += 1 + accepted as u32;
                                    }
                                }
                            }
                        }
                        for costs in cost_tables {
                            let mut expected = 0;
                            for j in 1..=length {
                                if emitted[j] as f64 * costs[expected]
                                    > emitted[expected] as f64 * costs[j]
                                {
                                    expected = j;
                                }
                            }
                            assert_eq!(
                                draft_prefix(&probabilities[..length], &costs[..=length]),
                                Some(expected)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn draft_prefix_rejects_invalid_profiles_and_prefers_shorter_ties() {
        assert_eq!(draft_prefix(&[], &[0.1]), Some(0));
        assert_eq!(draft_prefix(&[0., 0., 0.], &[1.; 4]), Some(0));
        assert_eq!(draft_prefix(&[1., 1., 1.], &[1.; 4]), Some(3));
        assert_eq!(draft_prefix(&[1.; 4], &[1.; 5]), None);
        assert_eq!(draft_prefix(&[1.], &[1.]), None);
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1., 1.1] {
            assert_eq!(draft_prefix(&[invalid], &[1.; 2]), None);
        }
        for invalid in [f64::NAN, f64::INFINITY, 0., -1., 361., 1e-12] {
            assert_eq!(draft_prefix(&[0.5], &[1., invalid]), None);
        }
    }

    #[test]
    fn expert_cohorts_fit_even_the_smallest_streaming_cache() {
        for capacity in 0..=1024 {
            match expert_cohort_rows(capacity) {
                Some(rows) => {
                    assert!((10..=512).contains(&capacity));
                    assert!((1..=8).contains(&rows));
                    assert!(rows * 10 <= capacity);
                    assert!(rows == 8 || (rows + 1) * 10 > capacity);
                }
                None => assert!(!(10..=512).contains(&capacity)),
            }
        }
        assert_eq!(expert_cohort_rows(usize::MAX), None);
    }

    #[test]
    fn unique_prefix_matches_set_oracle_and_respects_pending_ownership() -> Result<(), CacheError> {
        use std::collections::BTreeSet;
        let mut seed = 0x638053u64;
        for case in 0..4096 {
            let rows = case % 8 + 1;
            let ids = (0..rows * 10)
                .map(|_| {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((seed >> 32) % (case % 81 + 1) as u64) as u32
                })
                .collect::<Vec<_>>();
            let available = case % 81;
            let expected = (1..=rows)
                .take_while(|&n| ids[..n * 10].iter().collect::<BTreeSet<_>>().len() <= available)
                .count();
            assert_eq!(largest_expert_prefix(&ids, available), Some(expected));
        }
        for length in [0, 1, 9, 11, 79, 81, 90] {
            assert_eq!(largest_expert_prefix(&vec![0; length], 53), None);
        }
        for bad in [512, u32::MAX] {
            let mut ids = [0; 80];
            ids[79] = bad;
            assert_eq!(largest_expert_prefix(&ids, 0), None);
        }
        assert_eq!(largest_expert_prefix(&[511; 80], 1), Some(8));
        assert_eq!(largest_expert_prefix(&[511; 80], 0), Some(0));
        assert_eq!(largest_expert_prefix(&[0; 10], 513), None);
        assert_eq!(largest_expert_prefix(&[0; 10], usize::MAX), None);
        let ids = (0..8)
            .flat_map(|row| {
                let group = if row == 5 {
                    0
                } else if row > 5 {
                    row - 1
                } else {
                    row
                };
                group * 10..group * 10 + 10
            })
            .collect::<Vec<u32>>();
        for pending in [0, 4] {
            let mut cache = ExpertCache::new(53)?;
            let claims = (500..500 + pending)
                .map(|id| Ok((cache.claim_prefetch(id)?.unwrap(), id)))
                .collect::<Result<Vec<_>, CacheError>>()?;
            let mut begin = 0;
            let mut sizes = Vec::new();
            while begin < ids.len() {
                let n = largest_expert_prefix(&ids[begin..], cache.available_slots()).unwrap();
                assert!(n > 0);
                let routes = cache.reserve(&ids[begin..begin + n * 10])?;
                assert!(
                    routes
                        .iter()
                        .all(|&(slot, _)| claims.iter().all(|&(busy, _)| busy != slot))
                );
                assert!(
                    claims
                        .iter()
                        .all(|&(slot, id)| cache.pending[slot] == Some(id))
                );
                begin += n * 10;
                sizes.push(n);
            }
            assert_eq!(sizes, if pending == 0 { vec![6, 2] } else { vec![4, 4] });
        }
        Ok(())
    }

    #[test]
    fn gpu_hot_wide_accepts_only_explicit_boolean_literals() {
        for (value, expected) in [("0", Some(false)), ("1", Some(true))] {
            assert_eq!(parse_gpu_hot_wide(value), expected);
        }
        for value in ["", "true", "false", "2", "01", " 1", "1\n", "é", "１"] {
            assert_eq!(parse_gpu_hot_wide(value), None);
        }
    }

    #[test]
    fn fixed_draft_width_rejects_nonliteral_and_adaptive_controls() {
        for a in 0..=255u8 {
            for bytes in [vec![a], vec![a, b'1']] {
                if let Ok(value) = std::str::from_utf8(&bytes) {
                    let expected = (bytes.len() == 1 && (b'1'..=b'7').contains(&a))
                        .then(|| usize::from(a - b'0'));
                    assert_eq!(fixed_draft_width(value, false), expected);
                    assert_eq!(fixed_draft_width(value, true), None);
                }
            }
        }
        for value in [
            "",
            "0",
            "8",
            "01",
            "+1",
            " 1",
            "1\n",
            "é",
            "１",
            &"1".repeat(4096),
        ] {
            assert_eq!(fixed_draft_width(value, false), None);
        }
        for width in 1..=7 {
            assert_eq!(fixed_draft_width(&width.to_string(), false), Some(width));
        }
    }

    #[test]
    fn draft_commits_match_independent_wide_arithmetic_and_reject_stale_tapes() {
        for base in [-1, 0, 1, 4096, i32::MAX - 8, i32::MAX - 1, i32::MAX] {
            for total in (0..=10).chain([usize::MAX]) {
                for retained in (0..=10).chain([usize::MAX]) {
                    for current in [
                        i32::MIN,
                        0,
                        base.saturating_add(total.min(9) as i32),
                        i32::MAX,
                    ] {
                        let valid = base >= 0
                            && (1..=8).contains(&total)
                            && (1..=total).contains(&retained)
                            && i64::from(base) + total as i64 == i64::from(current);
                        let expected = valid.then(|| (i64::from(base) + retained as i64) as i32);
                        assert_eq!(
                            draft_commit_offset(base, current, total, retained),
                            expected
                        );
                    }
                }
            }
        }
    }
}

#[cfg(kani)]
mod draft_contract_proofs {
    use super::*;

    #[kani::proof]
    fn expert_output_stride_cannot_wrap_and_matches_wide_arithmetic() {
        let routes: u32 = kani::any();
        let projections: u32 = kani::any();
        let width: u32 = kani::any();
        let actual = expert_output_stride(routes, projections, width);
        let wide = u128::from(routes) * u128::from(projections) * u128::from(width);
        let valid = routes > 0 && projections > 0 && width > 0 && wide <= i32::MAX as u128;
        assert_eq!(actual.is_some(), valid);
        if let Some(stride) = actual {
            assert_eq!(u128::from(stride), wide);
        }
        kani::cover!(routes == 80 && projections == 2 && width == 640 && actual == Some(102400));
        kani::cover!(routes == 7 && projections == 1 && width == 2560 && actual == Some(17920));
        kani::cover!(routes == 0 && actual.is_none());
        kani::cover!(wide > u128::from(u32::MAX) && actual.is_none());
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn fixed_draft_width_is_exact_and_rejects_adaptive_or_invalid_literals() {
        let bytes: [u8; 3] = kani::any();
        let length = usize::from(kani::any::<u8>() % 4);
        let adaptive: bool = kani::any();
        if let Ok(value) = std::str::from_utf8(&bytes[..length]) {
            let actual = fixed_draft_width(value, adaptive);
            let expected = if !adaptive && length == 1 && (b'1'..=b'7').contains(&bytes[0]) {
                Some(usize::from(bytes[0]) - usize::from(b'0'))
            } else {
                None
            };
            assert_eq!(actual, expected);
            kani::cover!(actual == Some(1));
            kani::cover!(actual == Some(7));
            kani::cover!(adaptive && actual.is_none());
            kani::cover!(!adaptive && length == 3 && actual.is_none());
        }
    }

    #[kani::proof]
    #[kani::unwind(5)]
    fn draft_scheduler_cannot_exceed_available_prefix_or_accept_invalid_costs() {
        let confidence = kani::any::<[f64; 3]>();
        let costs = kani::any::<[f64; 4]>();
        let length = kani::any::<usize>();
        kani::assume(length <= 3);
        if let Some(prefix) = draft_prefix(&confidence[..length], &costs[..=length]) {
            assert!(prefix <= length);
            assert!(
                costs[..=length]
                    .iter()
                    .all(|c| c.is_finite() && (1e-9..=360.).contains(c))
            );
            assert!(
                confidence[..length]
                    .iter()
                    .all(|p| p.is_finite() && (0.0..=1.).contains(p))
            );
            kani::cover!(length == 3 && prefix == 0);
            kani::cover!(length == 3 && prefix == 3);
        }
    }

    #[kani::proof]
    fn expert_cohorts_never_exceed_live_slot_capacity() {
        let capacity: usize = kani::any();
        if let Some(rows) = expert_cohort_rows(capacity) {
            assert!((10..=512).contains(&capacity));
            assert!((1..=8).contains(&rows));
            assert!(rows * 10 <= capacity);
            assert!(rows == 8 || (rows + 1) * 10 > capacity);
            kani::cover!(rows == 1);
            kani::cover!(rows == 7);
            kani::cover!(rows == 8);
        } else {
            assert!(!(10..=512).contains(&capacity));
        }
    }

    fn unique_expert_prefix_case<const ROWS: usize>() -> Option<usize> {
        let ids: [u32; 80] = kani::any();
        let rows = ROWS;
        let available: usize = kani::any();
        if let Some(prefix) = largest_expert_prefix(&ids[..rows * 10], available) {
            assert!(prefix <= rows && available <= 512);
            assert!(ids[..rows * 10].iter().all(|&id| id < 512));
            if available >= rows * 10 {
                assert_eq!(prefix, rows);
            }
            if available == 0 {
                assert_eq!(prefix, 0);
            }
            return Some(prefix);
        }
        None
    }
    macro_rules! prefix_goal {
        (8, $prefix:expr) => {
            kani::cover!($prefix == Some(8));
        };
        ($rows:literal, $prefix:expr) => {};
    }
    // Exhaustive partition of the original rows in 1..=8, with every u32 ID
    // and every usize capacity still symbolic and the real bitset unchanged.
    macro_rules! prefix_proofs {
        ($($name:ident: $rows:tt),* $(,)?) => {$(
            #[kani::proof]
            #[kani::unwind(81)]
            fn $name() {
                let prefix = unique_expert_prefix_case::<$rows>();
                kani::cover!(prefix == Some(0));
                prefix_goal!($rows, prefix);
            }
        )*};
    }
    prefix_proofs! {
        unique_expert_prefix_rows_1: 1,
        unique_expert_prefix_rows_2: 2,
        unique_expert_prefix_rows_3: 3,
        unique_expert_prefix_rows_4: 4,
        unique_expert_prefix_rows_5: 5,
        unique_expert_prefix_rows_6: 6,
        unique_expert_prefix_rows_7: 7,
        unique_expert_prefix_rows_8: 8,
    }

    #[kani::proof]
    #[kani::unwind(21)]
    fn unique_expert_prefix_matches_independent_twenty_route_oracle() {
        let ids: [u32; 20] = kani::any();
        let available: usize = kani::any();
        kani::assume(available <= 512);
        for &id in &ids {
            kani::assume(id < 512);
        }
        let mut expected = 0;
        for rows in 1..=2 {
            let prefix = &ids[..rows * 10];
            let unique = prefix
                .iter()
                .enumerate()
                .filter(|&(i, id)| !prefix[..i].contains(id))
                .count();
            if unique <= available {
                expected = rows;
            }
        }
        let actual = largest_expert_prefix(&ids, available).unwrap();
        assert_eq!(actual, expected);
        kani::cover!(actual == 0);
        kani::cover!(actual == 1);
        kani::cover!(actual == 2);
    }

    #[kani::proof]
    fn wide_draft_commit_accepts_only_exact_finite_prefixes() {
        let base: i32 = kani::any();
        let current: i32 = kani::any();
        let total: usize = kani::any();
        let retained: usize = kani::any();
        let result = draft_commit_offset(base, current, total, retained);
        let valid = base >= 0
            && (1..=8).contains(&total)
            && (1..=total).contains(&retained)
            && i64::from(base) + total as i64 == i64::from(current);
        assert_eq!(result.is_some(), valid);
        if let Some(end) = result {
            assert_eq!(i64::from(end), i64::from(base) + retained as i64);
            assert!(end <= current);
            kani::cover!(total == 8 && retained == 1);
            kani::cover!(total == 8 && retained == 8);
            kani::cover!(total == 4 && retained == 3);
        }
    }
}

#[cfg(any(all(feature = "mlx", feature = "chat"), test, kani))]
pub(crate) fn draft_commit_offset(
    base: i32,
    current: i32,
    total: usize,
    retained: usize,
) -> Option<i32> {
    if base < 0
        || !(1..=8).contains(&total)
        || !(1..=total).contains(&retained)
        || base.checked_add(total as i32)? != current
    {
        return None;
    }
    base.checked_add(retained as i32)
}

pub fn row_offset(base: u64, row: u64, rows: u64, bytes: u64) -> Option<u64> {
    if row >= rows || bytes == 0 {
        return None;
    }
    let start = base.checked_add(row.checked_mul(bytes)?)?;
    let end = start.checked_add(bytes)?;
    (end <= i64::MAX as u64).then_some(start)
}

/// The reference multiplies signed int64 tokens with wraparound, then uses
/// Euclidean remainder. EOS padding prevents n-grams crossing a segment.
pub fn ngram_row(
    tokens: [u32; 3],
    multipliers: [u64; 3],
    n: usize,
    offset: u64,
    size: u64,
) -> Option<u64> {
    if !(2..=3).contains(&n) || size == 0 || size > i64::MAX as u64 {
        return None;
    }
    let mut bits = 0u64;
    for i in 0..n {
        bits ^= u64::from(tokens[i]).wrapping_mul(multipliers[i]);
    }
    offset.checked_add((bits as i64).rem_euclid(size as i64) as u64)
}

pub fn segment_tokens(current: u32, previous: [u32; 2], eos: u32) -> [u32; 3] {
    [
        current,
        previous[0],
        if previous[0] == eos { eos } else { previous[1] },
    ]
}

/// Four successive packed symbols, newest first. The first word is the row scale.
pub fn ple_codeword(words: &[u16; 41], position: usize) -> Option<u16> {
    if position >= 160 {
        return None;
    }
    let mut state = 0u16;
    for j in 0..4 {
        let symbol = (position + 160 - j) % 160;
        state |= ((words[1 + symbol / 4] >> (4 * (symbol % 4))) & 15) << (j * 4);
    }
    Some(state)
}
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn ple_codewords_select_the_wrapped_nibbles() {
    let words: [u16; 41] = kani::any();
    let i: usize = kani::any();
    let j: u8 = kani::any();
    let codeword = ple_codeword(&words, i);
    assert_eq!(codeword.is_some(), i < 160);
    if let Some(value) = codeword
        && j < 4
    {
        let tap = usize::from(j);
        let from = if tap > i { 160 - (tap - i) } else { i - tap };
        let expected = words[1 + from / 4] >> (4 * (from % 4)) & 15;
        assert_eq!((value >> (4 * tap)) & 15, expected);
        kani::cover!(i == 0 && j == 3);
        kani::cover!(i == 159 && j == 0);
    }
}
#[test]
fn ple_codeword_reads_four_symbols_across_column_zero() {
    let mut row = [0u16; 41];
    row[0] = 0xffff;
    row[1] = 0x4321;
    row[40] = 0xabcd;
    assert_eq!(ple_codeword(&row, 0), Some(0xcba1));
    assert_eq!(ple_codeword(&row, 3), Some(0x1234));
    assert_eq!(ple_codeword(&row, 159), Some(0xdcba));
    assert!(ple_codeword(&row, usize::MAX).is_none());
}

/// Identifies one outstanding write, even when the same slot and expert recur.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SlotLease {
    pub(crate) slot: usize,
    pub(crate) expert: u32,
    generation: u64,
}

#[path = "flash_support/slru.rs"]
mod slru;

/// Reservation is atomic: every routed expert gets a distinct protected slot.
/// A failed read must invalidate its reservation before it can become a hit.
pub struct ExpertCache {
    keys: Vec<Option<u32>>,
    pending: Vec<Option<u32>>,
    generations: Vec<u64>,
    prefetched: Vec<bool>,
    predicted: [bool; 512],
    predictive_lru: bool,
    slru: Option<slru::Slru>,
    counts: [u32; 512],
    ages: Vec<u64>,
    clock: u64,
    pub prefetch_hits: u64,
}

/// Unfilled demand reservations must never survive an error or unwind.
#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal"),
    test,
    kani
))]
pub(crate) struct ExpertFill {
    cache: Rc<RefCell<ExpertCache>>,
    complete: bool,
}

#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal"),
    test,
    kani
))]
impl ExpertFill {
    pub(crate) fn new(cache: &Rc<RefCell<ExpertCache>>) -> Self {
        Self {
            cache: Rc::clone(cache),
            complete: false,
        }
    }
    pub(crate) fn commit(mut self) {
        self.complete = true;
    }
}

#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal"),
    test,
    kani
))]
impl Drop for ExpertFill {
    fn drop(&mut self) {
        if !self.complete {
            self.cache.borrow_mut().invalidate();
        }
    }
}

impl ExpertCache {
    pub fn new(slots: usize) -> Result<Self, CacheError> {
        if !(10..=512).contains(&slots) {
            return Err(CacheError::Capacity);
        }
        Ok(Self {
            keys: vec![None; slots],
            pending: vec![None; slots],
            generations: vec![0; slots],
            prefetched: vec![false; slots],
            predicted: [false; 512],
            predictive_lru: false,
            slru: None,
            counts: [0; 512],
            ages: vec![0; slots],
            clock: 0,
            prefetch_hits: 0,
        })
    }

    pub fn reserve(&mut self, ids: &[u32]) -> Result<Vec<(usize, bool)>, CacheError> {
        if ids.is_empty() || ids.iter().any(|&id| id >= 512) {
            return Err(CacheError::InvalidIds);
        }
        let mut unique = Vec::with_capacity(ids.len().min(512));
        for &id in ids {
            let position = unique
                .iter()
                .position(|&key| key >= id)
                .unwrap_or(unique.len());
            if unique.get(position) != Some(&id) {
                unique.insert(position, id);
            }
        }
        if unique.len() > self.available_slots() {
            return Err(CacheError::Capacity);
        }
        self.clock = self.clock.checked_add(1).ok_or(CacheError::Clock)?;
        if self.clock.is_multiple_of(16) {
            for count in &mut self.counts {
                *count /= 2;
            }
        }
        if let Some(slru) = &mut self.slru {
            slru.record_reservation();
            let mut seen = [0u64; 8];
            for &id in ids {
                let word = id as usize / 64;
                let bit = 1u64 << (id % 64);
                if seen[word] & bit == 0 {
                    seen[word] |= bit;
                    if let Some(slot) = self.keys.iter().position(|&key| key == Some(id)) {
                        assert!(slru.touch(slot, true));
                    }
                }
            }
        }
        let mut reserved = Vec::with_capacity(unique.len());
        for &id in &unique {
            let present = self.keys.iter().position(|&key| key == Some(id));
            let slot = match present {
                Some(slot) => slot,
                None => self.victim(&unique).ok_or(CacheError::NoSlot)?,
            };
            if present.is_some() && self.prefetched[slot] {
                self.prefetch_hits = self.prefetch_hits.saturating_add(1);
            }
            self.prefetched[slot] = false;
            self.keys[slot] = Some(id);
            if present.is_none()
                && let Some(slru) = &mut self.slru
            {
                assert!(slru.touch(slot, false));
            }
            self.counts[id as usize] = self.counts[id as usize].saturating_add(1);
            self.ages[slot] = self.clock;
            reserved.push((id, slot, present.is_some()));
        }
        Ok(ids
            .iter()
            .map(|id| {
                let &(_, slot, hit) = reserved
                    .iter()
                    .find(|&&(key, _, _)| key == *id)
                    .expect("reserved expert");
                (slot, hit)
            })
            .collect())
    }

    pub fn contains(&self, id: u32) -> bool {
        self.keys.contains(&Some(id))
    }

    /// Snapshot after completed demand fills; pending writes never become GPU hits.
    #[cfg(any(feature = "mlx", test, kani))]
    pub(crate) fn resident_slot_map(&self) -> Result<[u32; 512], CacheError> {
        let mut map = [u32::MAX; 512];
        for (slot, (key, pending)) in self.keys.iter().zip(&self.pending).enumerate() {
            if pending.is_none()
                && let Some(id) = key
            {
                let entry = map.get_mut(*id as usize).ok_or(CacheError::InvalidIds)?;
                if *entry == u32::MAX {
                    *entry = slot as u32;
                }
            }
        }
        Ok(map)
    }

    fn victim(&self, routed: &[u32]) -> Option<usize> {
        if let Some(slru) = &self.slru {
            let eligible = |i: usize| {
                self.pending[i].is_none() && self.keys[i].is_none_or(|key| !routed.contains(&key))
            };
            return slru
                .ordered()
                .find(|&i| eligible(i) && self.keys[i].is_none())
                .or_else(|| slru.ordered().find(|&i| eligible(i)));
        }
        (0..self.keys.len())
            .filter(|&i| {
                self.pending[i].is_none() && self.keys[i].is_none_or(|key| !routed.contains(&key))
            })
            .min_by_key(|&i| {
                (
                    u8::from(self.keys[i].is_some()),
                    u8::from(self.keys[i].is_some_and(|key| self.predicted[key as usize])),
                    if self.predictive_lru {
                        0
                    } else {
                        self.keys[i].map_or(0, |key| self.counts[key as usize])
                    },
                    self.ages[i],
                    i,
                )
            })
    }

    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    pub(crate) fn predictive_lru_enabled(&self) -> bool {
        self.predictive_lru
    }

    #[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
    pub(crate) fn enable_slru(&mut self) {
        if !self.predictive_lru {
            self.slru = Some(slru::Slru::new(self.capacity()));
        }
    }
    #[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
    pub(crate) fn slru_enabled(&self) -> bool {
        self.slru.is_some()
    }
    #[cfg(test)]
    pub(crate) fn slru_reservations(&self) -> usize {
        self.slru.as_ref().map_or(0, |s| s.reservations())
    }

    pub(crate) fn enable_predictive_lru(&mut self) {
        self.slru = None;
        // Unused prefetches need a lifetime without inventing demand frequency.
        self.predictive_lru = true;
    }

    pub fn protect_predictions(&mut self, ids: &[u32]) -> Result<(), CacheError> {
        if ids.len() > 32 || ids.iter().any(|&id| id >= 512) {
            return Err(CacheError::InvalidIds);
        }
        self.predicted.fill(false);
        self.enable_predictive_lru();
        for &id in ids {
            self.predicted[id as usize] = true;
        }
        Ok(())
    }

    /// Leave enough non-pending slots for the next actual one/two-row cohort.
    pub fn claim_prefetch(&mut self, id: u32) -> Result<Option<usize>, CacheError> {
        if id >= 512 {
            return Err(CacheError::InvalidIds);
        }
        let cohort = if self.capacity() >= 20 { 20 } else { 10 };
        if self.contains(id) || self.pending.contains(&Some(id)) || self.available_slots() <= cohort
        {
            return Ok(None);
        }
        let slot = self.victim(&[]).ok_or(CacheError::NoSlot)?;
        self.generations[slot] = self.generations[slot]
            .checked_add(1)
            .ok_or(CacheError::Clock)?;
        self.keys[slot] = None;
        self.prefetched[slot] = false;
        self.pending[slot] = Some(id);
        if let Some(slru) = &mut self.slru {
            assert!(slru.touch(slot, false));
        }
        Ok(Some(slot))
    }

    #[cfg(any(feature = "mlx", all(target_os = "macos", feature = "direct-metal")))]
    pub(crate) fn claim_lease(&mut self, expert: u32) -> Result<Option<SlotLease>, CacheError> {
        Ok(self
            .claim_prefetch(expert)?
            .and_then(|slot| self.pending_lease(slot, expert)))
    }

    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
    fn begin_demand(&mut self, slot: usize, expert: u32) -> Result<SlotLease, CacheError> {
        if self.keys.get(slot) != Some(&Some(expert)) || self.pending[slot].is_some() {
            return Err(CacheError::NoSlot);
        }
        let generation = self.generations[slot]
            .checked_add(1)
            .ok_or(CacheError::Clock)?;
        self.generations[slot] = generation;
        self.keys[slot] = None;
        self.pending[slot] = Some(expert);
        Ok(SlotLease {
            slot,
            expert,
            generation,
        })
    }

    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
    fn finish_demand(&mut self, lease: SlotLease, valid: bool) -> bool {
        if self.pending_lease(lease.slot, lease.expert) != Some(lease) {
            return false;
        }
        self.pending[lease.slot] = None;
        // Reserve already accounted this demand. Arrival order must not change replacement ages.
        self.keys[lease.slot] = valid.then_some(lease.expert);
        valid
    }

    pub(crate) fn pending_lease(&self, slot: usize, expert: u32) -> Option<SlotLease> {
        if self.pending.get(slot) != Some(&Some(expert)) {
            return None;
        }
        Some(SlotLease {
            slot,
            expert,
            generation: self.generations[slot],
        })
    }

    pub(crate) fn finish_lease(&mut self, lease: SlotLease, valid: bool) -> bool {
        if self.pending_lease(lease.slot, lease.expert) != Some(lease) {
            return false;
        }
        self.publish_current(lease.slot, lease.expert, valid)
    }

    /// Synchronous compatibility entry point. An asynchronous reader must keep
    /// its original lease across the I/O and finish that generation instead.
    pub fn publish_prefetch(&mut self, slot: usize, id: u32, valid: bool) -> bool {
        let Some(lease) = self.pending_lease(slot, id) else {
            return false;
        };
        self.finish_lease(lease, valid)
    }

    fn publish_current(&mut self, slot: usize, id: u32, valid: bool) -> bool {
        self.pending[slot] = None;
        if !valid || self.contains(id) {
            return false;
        }
        let Some(clock) = self.clock.checked_add(1) else {
            return false;
        };
        self.clock = clock;
        self.keys[slot] = Some(id);
        self.prefetched[slot] = true;
        self.ages[slot] = self.clock;
        true
    }

    pub fn invalidate(&mut self) {
        if let Some(slru) = &mut self.slru {
            slru.clear();
        }
        self.keys.fill(None);
        self.prefetched.fill(false);
    }
    pub fn capacity(&self) -> usize {
        self.keys.len()
    }

    pub(crate) fn available_slots(&self) -> usize {
        self.pending.iter().filter(|p| p.is_none()).count()
    }
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) struct DemandPlan {
    cache: Rc<RefCell<ExpertCache>>,
    pub(crate) routes: Vec<(usize, bool)>,
    pub(crate) reads: Vec<SlotLease>,
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
impl DemandPlan {
    pub(crate) fn reserve(
        cache: &Rc<RefCell<ExpertCache>>,
        ids: &[u32],
    ) -> Result<Self, CacheError> {
        let guard = ExpertFill::new(cache);
        let routes = cache.borrow_mut().reserve(ids)?;
        let mut plan = Self {
            cache: cache.clone(),
            routes,
            reads: Vec::new(),
        };
        for (&expert, &(slot, hit)) in ids.iter().zip(&plan.routes) {
            if !hit && !plan.reads.iter().any(|lease| lease.slot == slot) {
                let result = { cache.borrow_mut().begin_demand(slot, expert) };
                plan.reads.push(result?);
            }
        }
        guard.commit();
        Ok(plan)
    }

    pub(crate) fn publish(&self, lease: SlotLease) -> bool {
        self.reads.contains(&lease) && self.cache.borrow_mut().finish_demand(lease, true)
    }
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
impl Drop for DemandPlan {
    fn drop(&mut self) {
        let mut cache = self.cache.borrow_mut();
        for &lease in &self.reads {
            cache.finish_demand(lease, false);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn delayed_completion_cannot_publish_a_recycled_expert_slot() -> Result<()> {
        let mut cache = ExpertCache::new(11)?;
        let slot = cache.claim_prefetch(17)?.unwrap();
        let old = cache.pending_lease(slot, 17).unwrap();
        assert!(!cache.finish_lease(old, false));
        assert_eq!(cache.claim_prefetch(17)?, Some(slot));
        let current = cache.pending_lease(slot, 17).unwrap();
        assert_ne!(old, current);
        assert!(!cache.finish_lease(old, true));
        assert!(!cache.contains(17));
        assert!(cache.finish_lease(current, true));
        assert!(cache.contains(17));
        assert!(!cache.finish_lease(current, true));
        Ok(())
    }

    #[test]
    fn layer_slot_proof_fixture_matches_original_for_every_suffix() {
        let prefix = "53,".repeat(47).into_bytes();
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                let suffix = [a, b];
                let actual = layer_slot_proof_text(suffix);
                let mut expected = prefix.clone();
                expected.extend_from_slice(&suffix);
                assert_eq!(actual.as_slice(), expected.as_slice());
                assert_eq!(
                    std::str::from_utf8(&actual).is_ok(),
                    std::str::from_utf8(&suffix).is_ok()
                );
            }
        }
    }

    #[test]
    fn layer_slot_parser_preserves_all_two_byte_suffixes() {
        fn previous(value: &str, uniform: usize) -> Option<[usize; 48]> {
            let mut parts = value.split(',');
            let mut slots = [0; 48];
            for slot in &mut slots {
                *slot = parts.next()?.parse().ok()?;
            }
            (parts.next().is_none() && valid_layer_slots(&slots, uniform)).then_some(slots)
        }
        let prefix = "53,".repeat(47);
        let mut examined = 0;
        for a in 0..=255u8 {
            for b in 0..=255u8 {
                let bytes = [a, b];
                if let Ok(last) = std::str::from_utf8(&bytes) {
                    let text = prefix.clone() + last;
                    assert_eq!(parse_layer_slots(&text, 53), previous(&text, 53));
                    assert_eq!(parse_layer_slots(&text, 53).is_some(), last == "53");
                    examined += 1;
                }
            }
        }
        assert!(examined > 0);
    }

    #[test]
    fn per_layer_slots_preserve_budget_and_reject_malformed_profiles() {
        let encode = |slots: &[usize; 48]| {
            slots
                .iter()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(",")
        };
        for uniform in [20, 53, 512] {
            assert_eq!(
                parse_layer_slots(&encode(&[uniform; 48]), uniform),
                Some([uniform; 48])
            );
        }
        let mut seed = 47u32;
        for _ in 0..1000 {
            let mut slots = [53; 48];
            for pair in slots.as_chunks_mut::<2>().0 {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                let delta = seed as usize % 34;
                pair[0] -= delta;
                pair[1] += delta;
            }
            let expected = slots.iter().all(|&n| (20..=512).contains(&n))
                && slots.iter().map(|&n| n as u128).sum::<u128>() == 48 * 53;
            assert!(expected);
            assert!(valid_layer_slots(&slots, 53));
            assert_eq!(parse_layer_slots(&encode(&slots), 53), Some(slots));
            slots[0] += 1;
            assert!(!valid_layer_slots(&slots, 53));
        }
        for uniform in [0, 9, 513, usize::MAX] {
            assert!(!valid_layer_slots(&[53; 48], uniform));
        }
        for invalid in [0, 19, 513, usize::MAX] {
            let mut slots = [53; 48];
            slots[0] = invalid;
            assert!(!valid_layer_slots(&slots, 53));
        }
        let prefix = "53,".repeat(47);
        for invalid in [
            "",
            "-1",
            "1.0",
            "NaN",
            "é",
            "５３",
            "53 ",
            " 53",
            "184467440737095516160",
            "53,53",
        ] {
            assert!(parse_layer_slots(&(prefix.clone() + invalid), 53).is_none());
        }
        assert!(parse_layer_slots(&"53,".repeat(46), 53).is_none());
        assert!(parse_layer_slots(&encode(&[54; 48]), 53).is_none());
        assert!(parse_layer_slots(&encode(&[20; 48]), 10).is_none());
    }
    #[test]
    fn fill_reservation_preserves_ready_weights_only_on_commit_and_keeps_pending() {
        for complete in [false, true] {
            let cache = Rc::new(RefCell::new(ExpertCache::new(24).unwrap()));
            let pending = cache.borrow_mut().claim_prefetch(511).unwrap().unwrap();
            cache.borrow_mut().reserve(&[7, 8]).unwrap();
            {
                let fill = ExpertFill::new(&cache);
                cache.borrow_mut().reserve(&[9, 10]).unwrap();
                if complete {
                    fill.commit();
                }
            }
            assert!(
                [7, 8, 9, 10]
                    .iter()
                    .all(|&id| cache.borrow().contains(id) == complete)
            );
            assert_eq!(cache.borrow().pending[pending], Some(511));
            assert!(cache.borrow_mut().publish_prefetch(pending, 511, true));
        }
        let cache = Rc::new(RefCell::new(ExpertCache::new(10).unwrap()));
        cache.borrow_mut().reserve(&[7]).unwrap();
        let result = (|| -> Result<(), CacheError> {
            let _fill = ExpertFill::new(&cache);
            cache.borrow_mut().reserve(&[512])?;
            Ok(())
        })();
        assert!(result.is_err() && !cache.borrow().contains(7));
        cache.borrow_mut().reserve(&[7]).unwrap();
        let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _fill = ExpertFill::new(&cache);
            cache.borrow_mut().reserve(&[8]).unwrap();
            panic!("injected reservation unwind");
        }));
        assert!(unwind.is_err() && !cache.borrow().contains(7) && !cache.borrow().contains(8));
    }
    #[test]
    fn slot_validation_and_les_controls_reject_invalid_inputs() {
        for capacity in [-1, 0, 1, 3, 10, 20, 89, 512, i32::MAX] {
            assert!(!slots_fit(&[], capacity));
            for ids in [vec![0], vec![1, 1, 0], vec![88, 89], vec![u32::MAX]] {
                let expected =
                    capacity > 0 && ids.iter().all(|&id| i64::from(id) < i64::from(capacity));
                assert_eq!(slots_fit(&ids, capacity), expected);
            }
        }
        assert_eq!(parse_lru_override("auto"), Some(false));
        assert_eq!(parse_lru_override("lru"), Some(true));
        assert_eq!(parse_predictor_half("f32"), Some(false));
        assert_eq!(parse_predictor_half("f16"), Some(true));
        for invalid in ["", "0", "1", "LFU", "LRU", " lru", "f32\n", "f64", "é"] {
            assert_eq!(parse_lru_override(invalid), None);
            assert_eq!(parse_predictor_half(invalid), None);
        }
    }

    #[test]
    fn resident_slot_snapshot_is_read_only_and_excludes_pending_writes() {
        for capacity in [10, 20, 53, 512] {
            let mut cache = ExpertCache::new(capacity).unwrap();
            let mut control = ExpertCache::new(capacity).unwrap();
            assert!(
                cache
                    .resident_slot_map()
                    .unwrap()
                    .iter()
                    .all(|&slot| slot == u32::MAX)
            );
            for round in 0..80u32 {
                let ids = (0..10)
                    .map(|i| (round * 37 + i * 13) % 512)
                    .collect::<Vec<_>>();
                assert_eq!(cache.reserve(&ids), control.reserve(&ids));
                let map = cache.resident_slot_map().unwrap();
                for id in 0..512u32 {
                    let expected = cache
                        .keys
                        .iter()
                        .position(|&key| key == Some(id))
                        .map_or(u32::MAX, |slot| slot as u32);
                    assert_eq!(map[id as usize], expected);
                }
                assert_eq!((cache.clock, cache.counts), (control.clock, control.counts));
                assert_eq!(cache.ages, control.ages);
            }
        }
        let mut cache = ExpertCache::new(24).unwrap();
        cache.reserve(&[511]).unwrap();
        assert_ne!(cache.resident_slot_map().unwrap()[511], u32::MAX);
        let slot = cache.claim_prefetch(510).unwrap().unwrap();
        assert_eq!(cache.resident_slot_map().unwrap()[510], u32::MAX);
        cache.keys[slot] = Some(510);
        assert_eq!(cache.resident_slot_map().unwrap()[510], u32::MAX);
        cache.keys[slot] = None;
        assert!(!cache.publish_prefetch(slot, 510, false));
        assert_eq!(cache.resident_slot_map().unwrap()[510], u32::MAX);
        let slot = cache.claim_prefetch(510).unwrap().unwrap();
        assert!(cache.publish_prefetch(slot, 510, true));
        assert_eq!(cache.resident_slot_map().unwrap()[510], slot as u32);
        cache.invalidate();
        assert!(
            cache
                .resident_slot_map()
                .unwrap()
                .iter()
                .all(|&slot| slot == u32::MAX)
        );
        cache.keys[0] = Some(512);
        assert_eq!(cache.resident_slot_map(), Err(CacheError::InvalidIds));
    }

    #[test]
    fn disk_ranges_reject_empty_overflow_and_out_of_bounds() {
        assert!(disk_destination_fits(0, 1, 1));
        assert!(disk_destination_fits(7, 9, 16));
        assert!(disk_destination_fits(usize::MAX - 1, 1, usize::MAX));
        assert!(!disk_destination_fits(0, 0, 16));
        assert!(!disk_destination_fits(7, 10, 16));
        assert!(!disk_destination_fits(usize::MAX, 1, usize::MAX));
    }
    #[test]
    fn automatic_cache_reserves_headroom_and_stays_bounded() {
        let minimum = (8 << 30) + 10 * 48 * EXPERT_BYTES;
        for bytes in [0, 1, (8 << 30) - 1, 8 << 30, minimum - 1] {
            assert_eq!(automatic_cache_slots(bytes), None);
        }
        assert_eq!(automatic_cache_slots(minimum), Some(10));
        assert_eq!(automatic_cache_slots(12 << 30), Some(71));
        assert_eq!(automatic_cache_slots(14 << 30), Some(107));
        assert_eq!(automatic_cache_slots(u64::MAX), Some(107));
        for slots in 10..=107 {
            let threshold = (8 << 30) + slots * 48 * EXPERT_BYTES;
            assert_eq!(automatic_cache_slots(threshold), Some(slots as usize));
            assert_eq!(automatic_cache_slots(threshold + 1), Some(slots as usize));
            assert_eq!(
                automatic_cache_slots(threshold - 1),
                if slots == 10 {
                    None
                } else {
                    Some(slots as usize - 1)
                }
            );
        }
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn available_memory_accepts_older_complete_revisions() {
        // Apple's HOST_VM_INFO64_REV1_COUNT includes all fields we read.
        let rev1 = std::mem::offset_of!(libc::vm_statistics64, swapped_count)
            / std::mem::size_of::<libc::integer_t>();
        assert!(rev1 < libc::HOST_VM_INFO64_COUNT as usize);
        assert!(available_memory_response_complete(rev1 as u32));
        assert!(available_memory_response_complete(
            libc::HOST_VM_INFO64_COUNT
        ));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn available_memory_rejects_truncated_and_oversized_snapshots() {
        let mut info: libc::vm_statistics64 = unsafe { std::mem::zeroed() };
        let base = (&mut info as *mut libc::vm_statistics64) as usize;
        let field_end = (&info.external_page_count as *const libc::natural_t) as usize
            + std::mem::size_of_val(&info.external_page_count);
        let minimum_words = (field_end - base) / std::mem::size_of::<libc::integer_t>();
        assert_eq!(
            (field_end - base) % std::mem::size_of::<libc::integer_t>(),
            0
        );
        for count in 0..=libc::HOST_VM_INFO64_COUNT + 1 {
            assert_eq!(
                available_memory_response_complete(count),
                (minimum_words..=libc::HOST_VM_INFO64_COUNT as usize).contains(&(count as usize)),
                "count={count}"
            );
        }
        assert!(!available_memory_response_complete(u32::MAX));
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn available_memory_query_returns_a_positive_snapshot() {
        assert!(available_memory_bytes().unwrap() > 0);
    }
    #[test]
    fn pair_commits_reject_stale_invalid_and_overflowing_positions() {
        for base in [
            i32::MIN,
            -1,
            0,
            1,
            2047,
            i32::MAX - 2,
            i32::MAX - 1,
            i32::MAX,
        ] {
            for current in [i32::MIN, 0, 2, 3, 2049, i32::MAX] {
                for retained in [-1, 0, 1, 2, 3, i32::MAX] {
                    let valid = base >= 0
                        && (1..=2).contains(&retained)
                        && i64::from(base) + 2 == i64::from(current);
                    let expected = valid.then(|| (i64::from(base) + i64::from(retained)) as i32);
                    assert_eq!(pair_commit_offset(base, current, retained), expected);
                }
            }
        }
    }
    #[test]
    fn reference_partition_covers_real_shapes_and_extremes() {
        for (input, output, expected) in [
            (160, 40, 1),
            (40, 160, 1),
            (384, 160, 2),
            (768, 4, 4),
            (160, 15520, 1),
            (0, 0, 1),
            (u32::MAX, u32::MAX, 1),
            (u32::MAX, 0, 1 << 24),
        ] {
            assert_eq!(reference_splits(input, output), expected);
        }
        for input in 0..4096 {
            for output in [0, 1, 4, 40, 160, 15520, u32::MAX] {
                let splits = reference_splits(input, output);
                assert!(splits.is_power_of_two() && splits <= (input / 128).max(1));
                assert!(
                    u64::from(output) * u64::from(splits) >= 8192
                        || u64::from(input) / (2 * u64::from(splits)) < 128
                );
            }
        }
    }
    #[test]
    fn disk_offsets_reject_boundaries_and_overflow() {
        assert_eq!(row_offset(8, 2, 3, 82), Some(172));
        assert_eq!(row_offset(8, 3, 3, 82), None);
        assert_eq!(row_offset(0, 0, 1, 0), None);
        assert_eq!(row_offset(i64::MAX as u64, 0, 1, 1), None);
        assert_eq!(row_offset(0, u64::MAX - 1, u64::MAX, 82), None);
    }
    #[test]
    fn hashes_use_signed_remainder_and_eos_segments() {
        assert_eq!(ngram_row([1, 2, 3], [u64::MAX, 2, 3], 2, 100, 7), Some(102));
        assert_eq!(segment_tokens(4, [9, 5], 9), [4, 9, 9]);
        assert_eq!(segment_tokens(9, [4, 5], 9), [9, 4, 5]);
        assert_eq!(ngram_row([0; 3], [0; 3], 1, 0, 7), None);
        assert_eq!(ngram_row([0; 3], [0; 3], 3, 0, 0), None);
    }
    #[test]
    fn complete_cohorts_protect_hits_and_have_unique_slots() -> Result<()> {
        let mut cache = ExpertCache::new(10)?;
        let ids = (0..10).collect::<Vec<_>>();
        let first = cache.reserve(&ids)?;
        assert!(first.iter().all(|&(_, hit)| !hit));
        assert_eq!(
            cache.reserve(&ids)?,
            first
                .iter()
                .map(|&(slot, _)| (slot, true))
                .collect::<Vec<_>>()
        );
        let saved = cache.keys.clone();
        assert!(cache.reserve(&(0..11).collect::<Vec<_>>()).is_err());
        assert_eq!(cache.keys, saved);
        let next = [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 511];
        let selected = cache.reserve(&next)?;
        for (i, &id) in next.iter().enumerate() {
            assert_eq!(cache.keys[selected[i].0], Some(id));
            for (j, &other) in next.iter().enumerate() {
                assert_eq!(id == other, selected[i].0 == selected[j].0);
            }
        }
        cache.invalidate();
        assert!(cache.reserve(&ids)?.iter().all(|&(_, hit)| !hit));
        assert!(cache.reserve(&[512]).is_err());
        Ok(())
    }

    #[test]
    fn pending_records_never_hit_alias_or_publish_over_a_demand() -> Result<()> {
        let mut cache = ExpertCache::new(24)?;
        let slot = cache.claim_prefetch(511)?.unwrap();
        assert!(!cache.contains(511));
        assert_eq!(cache.claim_prefetch(511)?, None);
        cache.invalidate();
        let routes = cache.reserve(&[511, 511, 1])?;
        assert!(routes.iter().all(|&(s, hit)| s != slot && !hit));
        assert!(!cache.publish_prefetch(slot, 510, true));
        assert_eq!(cache.pending[slot], Some(511));
        assert!(!cache.publish_prefetch(slot, 511, true));
        assert_eq!(cache.keys.iter().filter(|&&k| k == Some(511)).count(), 1);
        let other = cache.claim_prefetch(510)?.unwrap();
        assert!(!cache.publish_prefetch(other, 510, false));
        assert!(!cache.contains(510));
        let other = cache.claim_prefetch(510)?.unwrap();
        assert!(cache.publish_prefetch(other, 510, true));
        assert!(cache.contains(510));
        assert_eq!(cache.reserve(&[510, 510])?, [(other, true); 2]);
        assert_eq!(cache.prefetch_hits, 1);
        assert!(cache.reserve(&[510])?[0].1);
        assert_eq!(cache.prefetch_hits, 1);
        assert!(!cache.publish_prefetch(usize::MAX, 0, true));
        assert!(cache.claim_prefetch(512).is_err());
        Ok(())
    }

    #[test]
    fn predictions_protect_victims_without_inventing_frequency_or_blocking_demand() -> Result<()> {
        let mut cache = ExpertCache::new(10)?;
        cache.reserve(&(0..10).collect::<Vec<_>>())?;
        cache.protect_predictions(&[0])?;
        let counts = cache.counts;
        let clock = cache.clock;
        assert!(cache.protect_predictions(&[512]).is_err());
        assert!(cache.protect_predictions(&[0; 33]).is_err());
        assert_eq!(cache.counts, counts);
        assert_eq!(cache.clock, clock);
        assert_eq!(cache.reserve(&[511])?, [(1, false)]);
        assert!(cache.contains(0));
        cache.protect_predictions(&(0..10).chain([511]).collect::<Vec<_>>())?;
        assert!(cache.reserve(&[510])?.iter().all(|&(_, hit)| !hit));
        // All slots protected still permits a real route through the bounded fallback.
        assert!(cache.contains(510));
        Ok(())
    }

    #[test]
    fn predictive_victims_replay_symbolic_case_and_all_protection_masks() -> Result<()> {
        let mut cache = ExpertCache::new(10)?;
        let values = [
            (7, 3, true, true),
            (7, 3, true, false),
            (7, 2, false, true),
            (7, 2, true, true),
            (7, 2, true, true),
            (7, 3, false, true),
            (2, 3, false, true),
            (7, 1, true, true),
            (2, 2, false, true),
            (2, 3, false, false),
        ];
        for (i, &(count, age, predicted, pending)) in values.iter().enumerate() {
            cache.keys[i] = Some(i as u32);
            cache.counts[i] = count;
            cache.ages[i] = age;
            cache.predicted[i] = predicted;
            cache.pending[i] = pending.then_some(511);
        }
        assert_eq!(cache.victim(&[5, 4]), Some(9));
        for mask in 0..1024_u16 {
            for i in 0..10 {
                cache.predicted[i] = mask & (1 << i) != 0;
                cache.pending[i] = None;
                cache.counts[i] = if i % 2 == 0 { u32::MAX } else { 0 };
                cache.ages[i] = if i % 3 == 0 { u64::MAX } else { 0 };
            }
            let slot = cache.victim(&[0, 9]).unwrap();
            assert!((1..9).contains(&slot));
            assert_eq!(cache.predicted[slot], (1..9).all(|i| cache.predicted[i]));
        }
        Ok(())
    }

    #[test]
    fn les_admission_retains_recent_unused_records_without_inventing_frequency() -> Result<()> {
        for capacity in [24, 89] {
            for demanded in [false, true] {
                let mut cache = ExpertCache::new(capacity)?;
                for first in (0..capacity).step_by(10) {
                    let ids = (0..10)
                        .map(|i| ((first + i) % capacity) as u32)
                        .collect::<Vec<_>>();
                    cache.reserve(&ids)?;
                }
                assert!(!cache.predictive_lru);
                assert!(cache.protect_predictions(&[512]).is_err());
                assert!(!cache.predictive_lru);
                cache.protect_predictions(&[511])?;
                let first = cache.claim_prefetch(511)?.unwrap();
                let clock = cache.clock;
                assert!(cache.publish_prefetch(first, 511, true));
                assert_eq!(cache.counts[511], 0);
                assert_eq!(cache.ages[first], clock + 1);
                if demanded {
                    assert!(cache.reserve(&[511])?[0].1);
                }
                cache.protect_predictions(&[510])?;
                let second = cache.claim_prefetch(510)?.unwrap();
                assert_ne!(first, second, "new prefetch evicted before older records");
                assert!(cache.publish_prefetch(second, 510, true));
                assert!(cache.contains(511));
                assert_eq!(cache.counts[511], u32::from(demanded));
            }
        }
        let mut cache = ExpertCache::new(10)?;
        cache.reserve(&(0..10).collect::<Vec<_>>())?;
        for _ in 0..3 {
            cache.reserve(&[0])?;
        }
        cache.reserve(&(1..10).collect::<Vec<_>>())?;
        assert_eq!(cache.victim(&[]), Some(1), "ordinary cache must retain LFU");
        cache.protect_predictions(&[])?;
        assert_eq!(cache.victim(&[]), Some(0), "LES window must use recency");
        Ok(())
    }

    #[test]
    fn publication_clock_exhaustion_frees_pending_without_a_false_hit() -> Result<()> {
        let mut cache = ExpertCache::new(24)?;
        cache.protect_predictions(&[511])?;
        let slot = cache.claim_prefetch(511)?.unwrap();
        cache.clock = u64::MAX;
        assert!(!cache.publish_prefetch(slot, 511, true));
        assert!(!cache.contains(511));
        assert_eq!(cache.pending[slot], None);
        assert_eq!(cache.clock, u64::MAX);
        assert_eq!(cache.reserve(&[511]), Err(CacheError::Clock));
        Ok(())
    }

    #[test]
    fn free_slots_precede_evictions_through_failed_and_duplicate_publications() -> Result<()> {
        #[derive(Clone, Copy, PartialEq)]
        enum Slot {
            Empty,
            Ready(u32),
            Pending(u32),
        }
        for capacity in [24, 89] {
            let mut cache = ExpertCache::new(capacity)?;
            let mut slots = vec![Slot::Empty; capacity];
            let mut seed = 39_u64;
            let mut random = || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            for step in 0..10_000 {
                match random() % 5 {
                    0 | 1 => {
                        let ids = (0..10).map(|_| (random() % 128) as u32).collect::<Vec<_>>();
                        let routes = cache.reserve(&ids)?;
                        for (&id, &(slot, hit)) in ids.iter().zip(&routes) {
                            if hit {
                                assert!(slots[slot] == Slot::Ready(id));
                            }
                            assert!(!matches!(slots[slot], Slot::Pending(_)));
                            slots[slot] = Slot::Ready(id);
                        }
                    }
                    2 => {
                        let ids = (0..16).map(|_| (random() % 128) as u32).collect::<Vec<_>>();
                        cache.protect_predictions(&ids)?;
                    }
                    3 if slots
                        .iter()
                        .filter(|s| matches!(s, Slot::Pending(_)))
                        .count()
                        < 4 =>
                    {
                        let id = (random() % 128) as u32;
                        let free = slots.contains(&Slot::Empty);
                        if let Some(slot) = cache.claim_prefetch(id)? {
                            assert!(
                                !(free && matches!(slots[slot], Slot::Ready(_))),
                                "free slot ignored: capacity={capacity}, step={step}"
                            );
                            assert!(!matches!(slots[slot], Slot::Pending(_)));
                            slots[slot] = Slot::Pending(id);
                        }
                    }
                    4 => {
                        let choices = slots
                            .iter()
                            .enumerate()
                            .filter_map(|(i, s)| matches!(s, Slot::Pending(_)).then_some(i))
                            .collect::<Vec<_>>();
                        if !choices.is_empty() {
                            let slot = choices[random() as usize % choices.len()];
                            let Slot::Pending(id) = slots[slot] else {
                                unreachable!()
                            };
                            let valid = random() % 4 != 0;
                            let duplicate = slots.contains(&Slot::Ready(id));
                            let published = cache.publish_prefetch(slot, id, valid);
                            assert_eq!(published, valid && !duplicate);
                            slots[slot] = if published {
                                Slot::Ready(id)
                            } else {
                                Slot::Empty
                            };
                        }
                    }
                    _ => {}
                }
                for slot in &slots {
                    if let Slot::Ready(id) = slot {
                        assert!(cache.contains(*id));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn speculative_admission_preserves_actual_cohort_space_and_failed_reservations() -> Result<()> {
        for capacity in [10, 11, 19, 20, 21, 24, 89, 512] {
            let mut cache = ExpertCache::new(capacity)?;
            let cohort = if capacity >= 20 { 20 } else { 10 };
            for id in 0..512 {
                cache.claim_prefetch(id)?;
            }
            let pending = cache.pending.clone();
            assert_eq!(
                pending.iter().filter(|p| p.is_some()).count(),
                capacity - cohort
            );
            let ids = (512 - cohort as u32..512).collect::<Vec<_>>();
            let routes = cache.reserve(&ids)?;
            assert!(
                routes
                    .iter()
                    .all(|&(slot, hit)| pending[slot].is_none() && !hit)
            );
            let old = (cache.keys.clone(), cache.clock, cache.counts);
            assert!(
                cache
                    .reserve(&(0..cohort as u32 + 1).collect::<Vec<_>>())
                    .is_err()
            );
            assert_eq!((cache.keys.clone(), cache.clock, cache.counts), old);
            cache.invalidate();
            assert_eq!(cache.pending, pending);
            for (slot, &id) in pending.iter().enumerate() {
                if let Some(id) = id {
                    cache.publish_prefetch(slot, id, true);
                }
            }
            assert!(cache.pending.iter().all(Option::is_none));
            cache.clock = u64::MAX;
            let keys = cache.keys.clone();
            assert_eq!(cache.reserve(&[511]), Err(CacheError::Clock));
            assert_eq!(cache.keys, keys);
        }
        Ok(())
    }

    #[test]
    fn wide_cohorts_exclude_pending_slots_before_reserving() -> Result<()> {
        for capacity in [10, 19, 20, 24, 53, 80, 512] {
            let mut cache = ExpertCache::new(capacity)?;
            let mut pending = Vec::new();
            for id in 0..4 {
                if let Some(slot) = cache.claim_prefetch(id)? {
                    pending.push((slot, id));
                }
            }
            let available = capacity - pending.len();
            assert_eq!(cache.available_slots(), available);
            let cohort = expert_cohort_rows(cache.available_slots()).unwrap();
            let ids = (100..100 + cohort as u32 * 10).collect::<Vec<_>>();
            let routes = cache.reserve(&ids)?;
            assert!(
                routes
                    .iter()
                    .all(|&(slot, hit)| { !hit && pending.iter().all(|&(busy, _)| busy != slot) })
            );
            assert_eq!(cache.available_slots(), available);
            if capacity == 53 || capacity == 80 {
                assert_eq!(cohort, capacity / 10 - 1);
                assert!(
                    cache
                        .reserve(&(100..100 + (capacity / 10 * 10) as u32).collect::<Vec<_>>())
                        .is_err()
                );
            }
            for (slot, id) in pending {
                cache.publish_prefetch(slot, id, false);
            }
            assert_eq!(cache.available_slots(), capacity);
        }
        Ok(())
    }

    #[test]
    fn persistent_prefetch_sequences_preserve_cache_invariants() -> Result<()> {
        for capacity in [10, 20, 24, 89] {
            let mut cache = ExpertCache::new(capacity)?;
            let mut rng = 0x83a2_5417_u64;
            for step in 0..256 {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let ids = (0..20)
                    .map(|i| ((rng + i * 13) % 512) as u32)
                    .collect::<Vec<_>>();
                cache.protect_predictions(&ids)?;
                for &id in &ids[..4] {
                    cache.claim_prefetch(id)?;
                }
                if step % 7 == 0 {
                    cache.invalidate();
                }
                let pending = cache.pending.clone();
                let routes = cache.reserve(&ids[..if capacity >= 20 { 20 } else { 10 }])?;
                for (i, &(slot, _)) in routes.iter().enumerate() {
                    assert!(pending[slot].is_none());
                    assert_eq!(cache.keys[slot], Some(ids[i]));
                }
                for (slot, id) in pending.into_iter().enumerate() {
                    if let Some(id) = id {
                        cache.publish_prefetch(slot, id, step % 3 != 0);
                    }
                }
                for (slot, &key) in cache.keys.iter().enumerate() {
                    assert!(key.is_none() || cache.pending[slot].is_none());
                    if let Some(key) = key {
                        assert!(key < 512);
                        assert!(!cache.keys[slot + 1..].contains(&Some(key)));
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn seeded_cache_cohorts_survive_evictions_aging_and_read_failures() -> Result<()> {
        for slots in [10, 20, 64, 512] {
            let mut cache = ExpertCache::new(slots)?;
            let mut rng = 0xdace_3487u64;
            for step in 0..500 {
                let ids = (0..10)
                    .map(|_| {
                        rng ^= rng << 13;
                        rng ^= rng >> 7;
                        rng ^= rng << 17;
                        (rng % 512) as u32
                    })
                    .collect::<Vec<_>>();
                let old = cache.keys.clone();
                let reservation = cache.reserve(&ids)?;
                for (i, &(slot, hit)) in reservation.iter().enumerate() {
                    assert!(slot < slots);
                    assert_eq!(cache.keys[slot], Some(ids[i]));
                    assert_eq!(hit, old.contains(&Some(ids[i])));
                    for (j, &other) in ids.iter().enumerate() {
                        assert_eq!(ids[i] == other, slot == reservation[j].0);
                    }
                }
                let keys = cache.keys.clone();
                let clock = cache.clock;
                assert!(cache.reserve(&[]).is_err());
                assert!(cache.reserve(&[512, 1]).is_err());
                assert_eq!(cache.keys, keys);
                assert_eq!(cache.clock, clock);
                if step % 37 == 0 {
                    cache.invalidate();
                }
            }
        }
        Ok(())
    }
}

#[cfg(kani)]
mod verification {
    use super::*;
    #[kani::proof]
    #[kani::unwind(81)]
    fn wide_cohorts_fit_every_pending_pattern_in_eighty_slots() {
        let mut cache = ExpertCache::new(80).unwrap();
        let mut occupied = 0usize;
        for slot in 0..80 {
            if kani::any() {
                cache.pending[slot] = Some(511);
                occupied += 1;
            }
        }
        let available = cache.available_slots();
        assert_eq!(available + occupied, 80);
        if let Some(rows) = expert_cohort_rows(available) {
            assert!(rows * 10 + occupied <= cache.capacity());
            kani::cover!(rows == 7 && occupied == 4);
            kani::cover!(rows == 8);
        } else {
            assert!(occupied > 70);
            kani::cover!(occupied == 80);
        }
    }

    #[kani::proof]
    #[kani::solver(kissat)]
    #[kani::unwind(513)]
    fn resident_slot_snapshot_excludes_pending_and_preserves_lfu() {
        let mut keys = [None; 20];
        let mut pending = [None; 20];
        for slot in 0..20 {
            let id: u32 = kani::any();
            kani::assume(id < 512);
            keys[slot] = if kani::any() { Some(id) } else { None };
            pending[slot] = if kani::any() { Some(id) } else { None };
        }
        let mut cache = ExpertCache::new(20).unwrap();
        cache.keys.copy_from_slice(&keys);
        cache.pending.copy_from_slice(&pending);
        let query: u32 = kani::any();
        kani::assume(query < 512);
        let expected = cache
            .keys
            .iter()
            .zip(&cache.pending)
            .position(|(&key, pending)| key == Some(query) && pending.is_none())
            .map_or(u32::MAX, |slot| slot as u32);
        let map = cache.resident_slot_map().unwrap();
        assert_eq!(map[query as usize], expected);
        assert_eq!(cache.clock, 0);
        // query is universally chosen over 0..512 and is not an input to the
        // snapshot. This checks every LFU entry without a second 512-step loop.
        assert_eq!(cache.counts[query as usize], 0);
        kani::cover!(query == 511 && expected == 19);
        kani::cover!(expected == u32::MAX && cache.pending[0].is_some());
        kani::cover!(expected == 0);
    }

    #[kani::proof]
    #[kani::unwind(49)]
    fn layer_slot_budget_is_exact_and_cannot_overflow() {
        let slots: [usize; 48] = kani::any();
        let uniform: usize = kani::any();
        let valid = valid_layer_slots(&slots, uniform);
        if valid {
            assert!((20..=512).contains(&uniform));
            let mut total = 0u128;
            for &capacity in &slots {
                assert!((20..=512).contains(&capacity));
                total += capacity as u128;
            }
            assert_eq!(total, uniform as u128 * 48);
            assert!(total <= 48 * 512);
        }
        kani::cover!(valid);
        kani::cover!(valid && slots[0] != slots[47]);
        kani::cover!(!valid && uniform == usize::MAX);
        kani::cover!(!valid && slots[0] == usize::MAX);
    }

    fn proof_two_byte_capacity(bytes: &[u8]) -> Option<usize> {
        assert!(bytes.len() <= 2);
        match bytes {
            [a] if a.is_ascii_digit() => Some(usize::from(a - b'0')),
            [b'+', b] if b.is_ascii_digit() => Some(usize::from(b - b'0')),
            [a, b] if a.is_ascii_digit() && b.is_ascii_digit() => {
                Some(usize::from(a - b'0') * 10 + usize::from(b - b'0'))
            }
            _ => None,
        }
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn slot_capacity_empty_matches_std() {
        assert!(parse_slot_capacity(&[]) == proof_two_byte_capacity(&[]));
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn slot_capacity_one_byte_matches_std() {
        let bytes: [u8; 1] = kani::any();
        let actual = parse_slot_capacity(&bytes);
        assert!(actual == proof_two_byte_capacity(&bytes));
        kani::cover!(actual == Some(9));
        kani::cover!(actual.is_none());
    }

    #[kani::proof]
    #[kani::unwind(4)]
    fn slot_capacity_two_bytes_matches_std() {
        let bytes: [u8; 2] = kani::any();
        let actual = parse_slot_capacity(&bytes);
        assert!(actual == proof_two_byte_capacity(&bytes));
        kani::cover!(actual == Some(53));
        kani::cover!(bytes == *b"+5" && actual == Some(5));
        kani::cover!(bytes == [255, 255] && actual.is_none());
    }

    #[kani::proof]
    #[kani::stub(super::parse_slot_capacity, proof_two_byte_capacity)]
    #[kani::unwind(144)]
    fn layer_slot_parser_checks_last_two_utf8_bytes_and_full_shape() {
        let bytes: [u8; 2] = kani::any();
        #[repr(align(8))]
        struct AlignedProfile([u8; 143]);
        let text = AlignedProfile(layer_slot_proof_text(bytes));
        assert_eq!(text.0.as_ptr().align_offset(8), 0);
        if let Ok(value) = std::str::from_utf8(&text.0) {
            let parsed = parse_layer_slots(value, 53);
            assert_eq!(parsed.is_some(), bytes == *b"53");
            if let Some(slots) = parsed {
                for slot in slots {
                    assert_eq!(slot, 53);
                }
            }
            kani::cover!(bytes == *b"53");
            kani::cover!(bytes != *b"53" && parsed.is_none());
        }
    }
    #[kani::proof]
    #[kani::unwind(27)]
    fn failed_fill_invalidates_ready_keys_without_releasing_pending_slots() {
        let keys: [Option<u32>; 24] = kani::any();
        let pending: [Option<u32>; 24] = kani::any();
        let prefetched: [bool; 24] = kani::any();
        let complete: bool = kani::any();
        let mut state = ExpertCache::new(24).unwrap();
        state.keys.copy_from_slice(&keys);
        state.pending.copy_from_slice(&pending);
        state.prefetched.copy_from_slice(&prefetched);
        let cache = Rc::new(RefCell::new(state));
        {
            let fill = ExpertFill::new(&cache);
            if complete {
                fill.commit();
            }
        }
        let state = cache.borrow();
        for i in 0..24 {
            assert_eq!(state.keys[i], if complete { keys[i] } else { None });
            assert_eq!(state.prefetched[i], complete && prefetched[i]);
            assert_eq!(state.pending[i], pending[i]);
        }
        kani::cover!(complete);
        kani::cover!(!complete);
    }
    #[kani::proof]
    #[kani::unwind(22)]
    fn cpu_slot_validation_preserves_bounds_for_real_route_widths() {
        let ids: [u32; 20] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= 20);
        let capacity: i32 = kani::any();
        let valid = slots_fit(&ids[..len], capacity);
        let mut expected = len != 0 && capacity > 0;
        for &id in &ids[..len] {
            expected &= i64::from(id) < i64::from(capacity);
        }
        assert_eq!(valid, expected);
        kani::cover!(valid && len == 20);
        kani::cover!(valid && len == 10);
        kani::cover!(!valid && len == 0);
        kani::cover!(!valid && capacity <= 0 && len > 0);
        kani::cover!(!valid && ids[0] == u32::MAX && len > 0);
    }

    #[kani::proof]
    #[kani::unwind(7)]
    fn les_control_parsers_accept_only_the_documented_literals() {
        let bytes: [u8; 5] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= 5);
        if let Ok(value) = std::str::from_utf8(&bytes[..len]) {
            let lru = parse_lru_override(value);
            let half = parse_predictor_half(value);
            let wide = parse_gpu_hot_wide(value);
            assert_eq!(lru.is_some(), value == "auto" || value == "lru");
            assert_eq!(half.is_some(), value == "f32" || value == "f16");
            assert_eq!(lru == Some(true), value == "lru");
            assert_eq!(half == Some(true), value == "f16");
            assert_eq!(wide.is_some(), value == "0" || value == "1");
            assert_eq!(wide == Some(true), value == "1");
            kani::cover!(wide == Some(false));
            kani::cover!(wide == Some(true));
            kani::cover!(lru == Some(false));
            kani::cover!(lru == Some(true));
            kani::cover!(half == Some(false));
            kani::cover!(half == Some(true));
            kani::cover!(lru.is_none() && half.is_none());
        }
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn predictive_victims_exclude_pending_and_fall_back_only_when_needed() {
        let mut cache = ExpertCache::new(10).unwrap();
        cache.predictive_lru = kani::any();
        for i in 0..10 {
            cache.keys[i] = Some(i as u32);
            cache.counts[i] = kani::any();
            cache.ages[i] = kani::any();
            cache.predicted[i] = kani::any();
            cache.pending[i] = if kani::any() { Some(511) } else { None };
        }
        let routed: [u32; 2] = kani::any();
        let eligible = |i: usize| cache.pending[i].is_none() && !routed.contains(&(i as u32));
        let available = (0..10).any(eligible);
        let unpredicted = (0..10).any(|i| eligible(i) && !cache.predicted[i]);
        let victim = cache.victim(&routed);
        assert_eq!(victim.is_some(), available);
        if let Some(slot) = victim {
            assert!(slot < 10 && eligible(slot));
            assert_eq!(cache.predicted[slot], !unpredicted);
            kani::cover!(unpredicted);
            kani::cover!(!unpredicted);
        }
        kani::cover!(!available);
        kani::cover!(cache.predictive_lru && available);
        kani::cover!(!cache.predictive_lru && available);
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn free_slots_precede_all_ready_victims_in_both_policies() {
        let mut cache = ExpertCache::new(10).unwrap();
        cache.predictive_lru = kani::any();
        for i in 0..10 {
            cache.keys[i] = if kani::any() { Some(i as u32) } else { None };
            cache.pending[i] = if cache.keys[i].is_none() && kani::any() {
                Some(511)
            } else {
                None
            };
            cache.counts[i] = kani::any();
            cache.ages[i] = kani::any();
            cache.predicted[i] = kani::any();
        }
        let routed: [u32; 2] = kani::any();
        let free = (0..10).any(|i| cache.keys[i].is_none() && cache.pending[i].is_none());
        let victim = cache.victim(&routed);
        if free {
            let slot = victim.unwrap();
            assert!(slot < 10 && cache.keys[slot].is_none() && cache.pending[slot].is_none());
            kani::cover!(cache.predictive_lru);
            kani::cover!(!cache.predictive_lru);
        }
        if let Some(slot) = victim {
            assert!(cache.pending[slot].is_none());
            kani::cover!(!free && cache.keys[slot].is_some());
        }
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn les_lru_uses_age_without_penalizing_zero_frequency_prefetches() {
        let mut cache = ExpertCache::new(10).unwrap();
        cache.predictive_lru = true;
        for i in 0..10 {
            cache.keys[i] = Some(i as u32);
            cache.counts[i] = kani::any();
            cache.ages[i] = kani::any();
            cache.predicted[i] = kani::any();
        }
        let routed: [u32; 2] = kani::any();
        let slot = cache.victim(&routed).unwrap();
        assert!(slot < 10 && !routed.contains(&(slot as u32)));
        for i in 0..10 {
            if !routed.contains(&(i as u32)) {
                assert!(!cache.predicted[slot] || cache.predicted[i]);
                if cache.predicted[slot] == cache.predicted[i] {
                    assert!(cache.ages[slot] <= cache.ages[i]);
                    if cache.ages[slot] == cache.ages[i] {
                        assert!(slot <= i);
                    }
                }
            }
        }
        kani::cover!(cache.predicted[slot]);
        kani::cover!(!cache.predicted[slot]);
        kani::cover!(cache.counts[slot] == 0 && cache.ages[slot] == 0);
        kani::cover!(cache.counts[slot] == u32::MAX && cache.ages[slot] == 0);
    }

    #[kani::proof]
    #[kani::unwind(25)]
    fn pending_publish_rejects_stale_failed_and_duplicate_records() {
        let mut cache = ExpertCache::new(24).unwrap();
        let slot: usize = kani::any();
        let id: u32 = kani::any();
        kani::assume(slot < 24 && id < 512);
        cache.pending[slot] = Some(id);
        let duplicate: bool = kani::any();
        if duplicate {
            cache.keys[(slot + 1) % 24] = Some(id);
        }
        let received: u32 = kani::any();
        let valid: bool = kani::any();
        let clock: u64 = kani::any();
        cache.clock = clock;
        let published = cache.publish_prefetch(slot, received, valid);
        assert_eq!(
            published,
            received == id && valid && !duplicate && clock < u64::MAX
        );
        assert_eq!(cache.pending[slot].is_none(), received == id);
        assert_eq!(cache.keys[slot] == Some(id), published);
        assert_eq!(cache.prefetched[slot], published);
        if published {
            assert_eq!(cache.clock, clock + 1);
            assert_eq!(cache.ages[slot], cache.clock);
        } else {
            assert_eq!(cache.clock, clock);
        }
        kani::cover!(published);
        kani::cover!(received == id && !valid);
        kani::cover!(received == id && valid && duplicate);
        kani::cover!(received != id);
        kani::cover!(received == id && valid && !duplicate && clock == u64::MAX);
    }

    #[kani::proof]
    #[kani::unwind(25)]
    fn speculative_claim_excludes_pending_and_preserves_pair_space() {
        let mut cache = ExpertCache::new(24).unwrap();
        cache.predictive_lru = kani::any();
        let id: u32 = kani::any();
        let pending: u8 = kani::any();
        kani::assume(pending <= 4);
        for i in 0..usize::from(pending) {
            cache.pending[i] = Some(i as u32);
        }
        let claimed = cache.claim_prefetch(id);
        if let Ok(Some(slot)) = claimed {
            assert!(id < 512 && pending < 4 && id >= u32::from(pending));
            assert!(slot >= usize::from(pending) && slot < 24);
            assert_eq!(cache.pending[slot], Some(id));
            assert_eq!(cache.keys[slot], None);
            kani::cover!(pending == 3);
        }
        if pending == 4 && id < 512 {
            assert_eq!(claimed, Ok(None));
        }
        kani::cover!(id >= 512 && claimed == Err(CacheError::InvalidIds));
    }
    #[kani::proof]
    fn disk_destinations_cannot_wrap_or_exceed_owned_storage() {
        let start: usize = kani::any();
        let length: usize = kani::any();
        let capacity: usize = kani::any();
        let fits = disk_destination_fits(start, length, capacity);
        assert_eq!(
            fits,
            length > 0 && start as u128 + length as u128 <= capacity as u128
        );
        kani::cover!(fits && start + length == capacity);
        kani::cover!(!fits && length == 0);
        kani::cover!(!fits && start as u128 + length as u128 > usize::MAX as u128);
    }
    #[kani::proof]
    fn automatic_cache_cannot_spend_reserved_memory() {
        let available: u64 = kani::any();
        let result = automatic_cache_slots(available);
        let minimum = (8u128 << 30) + 10 * 48 * u128::from(EXPERT_BYTES);
        assert_eq!(result.is_some(), u128::from(available) >= minimum);
        if let Some(slots) = result {
            assert!((10..=107).contains(&slots));
            let allocated = slots as u128 * 48 * u128::from(EXPERT_BYTES);
            assert!(allocated <= 6u128 << 30);
            assert!(allocated + (8u128 << 30) <= u128::from(available));
            kani::cover!(slots == 10);
            kani::cover!(slots == 71);
            kani::cover!(slots == 107);
        }
        kani::cover!(available == u64::MAX);
        kani::cover!(result.is_none());
    }

    #[kani::proof]
    fn automatic_cache_is_monotone() {
        let a: u64 = kani::any();
        let b: u64 = kani::any();
        if a <= b {
            assert!(automatic_cache_slots(a) <= automatic_cache_slots(b));
        }
        kani::cover!(
            a < b && automatic_cache_slots(a) == Some(10) && automatic_cache_slots(b) == Some(107)
        );
    }

    #[cfg(target_os = "macos")]
    #[kani::proof]
    fn available_memory_count_covers_fields_within_buffer() {
        let count: u32 = kani::any();
        let bytes = u64::from(count) * std::mem::size_of::<libc::integer_t>() as u64;
        let external_end = std::mem::offset_of!(libc::vm_statistics64, external_page_count)
            + std::mem::size_of::<libc::natural_t>();
        let valid = available_memory_response_complete(count);
        assert_eq!(
            valid,
            bytes >= external_end as u64
                && bytes <= std::mem::size_of::<libc::vm_statistics64>() as u64
        );
        if valid {
            assert!(bytes >= std::mem::offset_of!(libc::vm_statistics64, free_count) as u64 + 4);
            assert!(
                bytes >= std::mem::offset_of!(libc::vm_statistics64, speculative_count) as u64 + 4
            );
        }
        kani::cover!(valid && count < libc::HOST_VM_INFO64_COUNT);
        kani::cover!(valid && count == libc::HOST_VM_INFO64_COUNT);
        kani::cover!(!valid && bytes < external_end as u64);
        kani::cover!(!valid && count > libc::HOST_VM_INFO64_COUNT);
    }

    #[cfg(target_os = "macos")]
    #[kani::proof]
    fn available_memory_real_os_path() {
        assert!(available_memory_bytes().unwrap() > 0);
    }
    #[kani::proof]
    fn pair_commit_preserves_exact_valid_prefix_without_overflow() {
        let base: i32 = kani::any();
        let current: i32 = kani::any();
        let retained: i32 = kani::any();
        let result = pair_commit_offset(base, current, retained);
        let valid =
            base >= 0 && (1..=2).contains(&retained) && i64::from(base) + 2 == i64::from(current);
        assert_eq!(result.is_some(), valid);
        if let Some(end) = result {
            assert!(end > base && end <= current);
            assert_eq!(i64::from(end), i64::from(base) + i64::from(retained));
            kani::cover!(retained == 1);
            kani::cover!(retained == 2);
        }
        kani::cover!(base == i32::MAX && result.is_none());
        kani::cover!(base == 0 && current == 3 && result.is_none());
    }
    #[kani::proof]
    #[kani::unwind(26)]
    fn reference_partition_is_bounded_and_matches_stopping_rule() {
        let input: u32 = kani::any();
        let output: u32 = kani::any();
        let splits = reference_splits(input, output);
        assert!(splits.is_power_of_two());
        assert!(splits <= (input / 128).max(1));
        assert!(
            u64::from(output) * u64::from(splits) >= 8192
                || u64::from(input) / (2 * u64::from(splits)) < 128
        );
        if splits > 1 {
            assert!(u64::from(output) * u64::from(splits / 2) < 8192);
            assert!(input / splits >= 128);
        }
        kani::cover!(splits == 1 && input == 160 && output == 40);
        kani::cover!(splits > 1);
        kani::cover!(output == 0);
    }
    #[kani::proof]
    fn disk_row_offsets_are_bounded_and_do_not_wrap() {
        let base: u64 = kani::any();
        let row: u64 = kani::any();
        let rows: u64 = kani::any();
        let bytes: u64 = kani::any();
        if let Some(start) = row_offset(base, row, rows, bytes) {
            assert!(row < rows && bytes > 0 && start >= base);
            assert_eq!(start - base, row * bytes);
            assert!(start + bytes <= i64::MAX as u64);
            kani::cover!(row == 0);
            kani::cover!(row + 1 == rows);
        }
    }
    #[kani::proof]
    #[kani::unwind(4)]
    fn ngram_hash_stays_in_its_head_range() {
        let tokens: [u32; 3] = kani::any();
        let multipliers: [u64; 3] = kani::any();
        let n: usize = kani::any();
        let offset: u64 = kani::any();
        let size: u64 = kani::any();
        if let Some(row) = ngram_row(tokens, multipliers, n, offset, size) {
            assert!((2..=3).contains(&n) && size > 0);
            assert!(row >= offset && row - offset < size);
            kani::cover!(n == 2);
            kani::cover!(n == 3);
        }
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn two_routes_preserve_hits_and_never_alias_distinct_experts() {
        let mut cache = ExpertCache::new(10).unwrap();
        for i in 0..10 {
            cache.keys[i] = Some(i as u32);
            cache.ages[i] = 1;
            cache.counts[i] = 1;
        }
        cache.clock = 1;
        let ids: [u32; 2] = kani::any();
        if let Ok(routes) = cache.reserve(&ids) {
            assert!(ids[0] < 512 && ids[1] < 512);
            assert_eq!(routes.len(), 2);
            for i in 0..2 {
                assert!(routes[i].0 < 10);
                assert_eq!(cache.keys[routes[i].0], Some(ids[i]));
                assert_eq!(routes[i].1, ids[i] < 10);
            }
            assert_eq!(routes[0].0 == routes[1].0, ids[0] == ids[1]);
            kani::cover!(ids[0] < 10 && ids[1] >= 10);
            kani::cover!(ids[0] == ids[1]);
        } else {
            assert!(ids[0] >= 512 || ids[1] >= 512);
            for i in 0..10 {
                assert_eq!(cache.keys[i], Some(i as u32));
            }
        }
    }
}

#[cfg(test)]
mod grouped_route_tests {
    use super::{expert_group_chunks, expert_grouping_worthwhile, expert_route_groups};
    use std::collections::BTreeMap;

    fn check(ids: &[u32], capacity: i32) {
        let mut counts = BTreeMap::<u32, usize>::new();
        for &id in ids {
            *counts.entry(id).or_default() += 1;
        }
        let expected_chunks = (!ids.is_empty() && ids.len() <= 80)
            .then(|| counts.values().map(|n| n.div_ceil(4)).sum::<usize>());
        assert_eq!(expert_group_chunks(ids), expected_chunks);
        let actual = expert_route_groups(ids, capacity);
        let valid = !ids.is_empty()
            && ids.len() <= 80
            && (1..=512).contains(&capacity)
            && ids
                .iter()
                .all(|&slot| i64::from(slot) < i64::from(capacity));
        if !valid {
            assert!(actual.is_none());
            return;
        }
        let mut by_slot: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for (row, &slot) in ids.iter().enumerate() {
            by_slot.entry(slot).or_default().push(row as u32);
        }
        let mut lists = by_slot.into_values().collect::<Vec<_>>();
        lists.sort_by_key(|rows| rows[0]);
        let mut expected: [Vec<u32>; 4] = std::array::from_fn(|_| Vec::new());
        for rows in lists {
            for chunk in rows.chunks(4) {
                expected[chunk.len() - 1].extend_from_slice(chunk);
            }
        }
        let actual = actual.unwrap();
        assert_eq!(actual, expected);
        let mut coverage = actual.into_iter().flatten().collect::<Vec<_>>();
        coverage.sort_unstable();
        assert_eq!(coverage, (0..ids.len() as u32).collect::<Vec<_>>());
    }

    #[test]
    fn grouped_routes_preserve_every_original_position_and_reject_invalid_slots() {
        for capacity in [-1, 0, 1, 3, 512, 513] {
            for n in 0..=7 {
                for bits in 0..3usize.pow(n) {
                    let mut value = bits;
                    let ids = (0..n)
                        .map(|_| {
                            let slot = value % 3;
                            value /= 3;
                            slot as u32
                        })
                        .collect::<Vec<_>>();
                    check(&ids, capacity);
                }
            }
            for n in [1, 2, 3, 4, 5, 8, 79, 80, 81] {
                check(&vec![0; n], capacity);
                check(&(0..n as u32).collect::<Vec<_>>(), capacity);
                check(&vec![u32::MAX; n], capacity);
                check(&vec![511; n], capacity);
            }
        }
        let mut seed = 17u32;
        for n in (1..=80).cycle().take(1024) {
            let ids = (0..n)
                .map(|_| {
                    seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                    seed % 53
                })
                .collect::<Vec<_>>();
            check(&ids, 53);
        }
        assert_eq!(
            expert_route_groups(&[2, 1, 2, 2, 2, 2, 1, 0], 3).unwrap(),
            [vec![5, 7], vec![1, 6], vec![], vec![0, 2, 3, 4]]
        );
    }

    #[test]
    fn grouped_dispatch_threshold_matches_wide_arithmetic_at_all_boundaries() {
        for rows in (0..=100).chain([usize::MAX]) {
            for chunks in (0..=101).chain([usize::MAX]) {
                let expected = (16..=80).contains(&rows)
                    && (chunks as u128) * 4 >= rows as u128
                    && chunks <= rows
                    && ((rows as u128) - chunks as u128) * 5 >= rows as u128;
                assert_eq!(expert_grouping_worthwhile(rows, chunks), expected);
            }
        }
        assert!(expert_grouping_worthwhile(20, 16));
        assert!(!expert_grouping_worthwhile(20, 17));
        assert!(!expert_grouping_worthwhile(15, 4));
        assert_eq!(expert_group_chunks(&[u32::MAX; 80]), Some(20));
    }
}

#[cfg(kani)]
mod grouped_route_verification {
    use super::{expert_group_chunks, expert_grouping_worthwhile, expert_route_groups};

    #[kani::proof]
    fn grouped_dispatch_gate_matches_wide_arithmetic_for_all_usize_inputs() {
        let rows: usize = kani::any();
        let chunks: usize = kani::any();
        let expected = (16..=80).contains(&rows)
            && (chunks as u128) * 4 >= rows as u128
            && chunks <= rows
            && ((rows as u128) - chunks as u128) * 5 >= rows as u128;
        assert_eq!(expert_grouping_worthwhile(rows, chunks), expected);
        kani::cover!(rows == 20 && chunks == 16 && expected);
        kani::cover!(rows == 20 && chunks == 17 && !expected);
        kani::cover!(rows == usize::MAX && !expected);
    }

    #[kani::proof]
    #[kani::unwind(9)]
    fn grouped_chunk_counter_matches_eight_symbolic_occurrence_counts() {
        let storage: [u32; 8] = kani::any();
        let length = usize::from(kani::any::<u8>() % 9);
        let ids = &storage[..length];
        let mut expected = 0;
        for row in 0..length {
            if (0..row).all(|prior| ids[prior] != ids[row]) {
                let total = (0..length).filter(|&other| ids[other] == ids[row]).count();
                expected += total.div_ceil(4);
            }
        }
        let actual = expert_group_chunks(ids);
        assert_eq!(actual, (length > 0).then_some(expected));
        if let Some(chunks) = actual {
            assert!((length.div_ceil(4)..=length).contains(&chunks));
            kani::cover!(length == 8 && chunks == 8);
            kani::cover!(length == 8 && chunks == 2);
        }
        kani::cover!(actual.is_none());
    }

    #[kani::proof]
    #[kani::unwind(9)]
    fn grouped_expert_routes_cover_eight_symbolic_origins_exactly() {
        let ids: [u32; 8] = kani::any();
        let count = usize::from(kani::any::<u8>() % 9);
        let capacity = i32::from(kani::any::<u16>() % 514);
        let ids = &ids[..count];
        let valid = count > 0
            && (1..=512).contains(&capacity)
            && ids.iter().all(|&slot| slot < capacity as u32);
        let groups = expert_route_groups(ids, capacity);
        assert_eq!(groups.is_some(), valid);
        kani::cover!(!valid);
        if let Some(groups) = groups {
            let mut seen = [0u8; 8];
            for (i, rows) in groups.iter().enumerate() {
                assert_eq!(rows.len() % (i + 1), 0);
                for chunk in rows.chunks(i + 1) {
                    for &row in chunk {
                        assert!((row as usize) < count);
                        assert_eq!(ids[row as usize], ids[chunk[0] as usize]);
                        seen[row as usize] += 1;
                    }
                }
            }
            for (row, &uses) in seen.iter().enumerate() {
                assert_eq!(uses, u8::from(row < count));
            }
            kani::cover!(groups[0].len() == 8);
            kani::cover!(groups[3].len() == 8);
        }
    }

    #[kani::proof]
    #[kani::unwind(81)]
    fn grouped_expert_routes_eighty_symbolic_inputs_are_bounded() {
        let ids: [u32; 80] = kani::any();
        let capacity = i32::from(kani::any::<u16>() % 514);
        if let Some(groups) = expert_route_groups(&ids, capacity) {
            assert_eq!(groups.iter().map(Vec::len).sum::<usize>(), 80);
            for rows in &groups {
                for &row in rows {
                    assert!(row < 80);
                }
            }
        }
    }
}

#[test]
fn slot_read_segments_initialize_every_byte_before_publication() {
    for capacity in [10, 24, 53, 512] {
        let widths = [819200, 409600, 10240, 2560, 1280, 5120, 12];
        for slot in 0..capacity {
            let ranges = expert_slot_destinations(slot);
            assert_eq!(ranges.iter().map(|r| r.2 as u64).sum::<u64>(), EXPERT_BYTES);
            for (bank, width) in widths.into_iter().enumerate() {
                let mut writes = ranges.iter().filter(|r| r.0 == bank).collect::<Vec<_>>();
                writes.sort_by_key(|r| r.1);
                let mut end = slot * width;
                for &&(_, start, count) in &writes {
                    assert_eq!(start, end, "hole/overlap in bank{bank} slot{slot}");
                    assert!(count > 0);
                    end += count;
                }
                assert_eq!(end, (slot + 1) * width);
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(14)]
fn stale_prefetch_completion_preserves_new_reservation() {
    let mut cache = ExpertCache::new(11).unwrap();
    let expert = u32::from(kani::any::<u8>()) * 2 + u32::from(kani::any::<bool>());
    let slot = cache.claim_prefetch(expert).unwrap().unwrap();
    let old = cache.pending_lease(slot, expert).unwrap();
    assert!(!cache.finish_lease(old, false));
    assert_eq!(cache.claim_prefetch(expert).unwrap(), Some(slot));
    let current = cache.pending_lease(slot, expert).unwrap();
    let valid: bool = kani::any();
    assert!(!cache.finish_lease(old, valid));
    assert_eq!(cache.pending_lease(slot, expert), Some(current));
    kani::cover!(valid && expert == 511);
    kani::cover!(!valid && expert == 0);
}

/// Byte offsets inside an inspected K2/MUL1 record, ordered down, gate, up.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
#[derive(Clone, Copy, Default)]
pub(crate) struct RecordOffsets {
    pub suh: u64,
    pub svh: u64,
    pub trellis: u64,
}
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn record_offsets(index: usize) -> Option<RecordOffsets> {
    let records = expert_slot_destinations(0);
    let first = index.checked_mul(4)?;
    let fields = records.get(first..first.checked_add(4)?)?;
    let suh = records[..first].iter().map(|r| r.2 as u64).sum::<u64>();
    let svh = suh + fields[0].2 as u64;
    Some(RecordOffsets {
        suh,
        svh,
        trellis: svh + fields[1].2 as u64 + fields[2].2 as u64,
    })
}
#[test]
fn immutable_flash_records_match_all_three_packed_intervals() {
    for (index, input) in [640, 2560, 2560].into_iter().enumerate() {
        let r = record_offsets(index).unwrap();
        assert_eq!(
            (r.suh, r.svh, r.trellis),
            (
                index as u64 * 416004,
                index as u64 * 416004 + input * 2,
                index as u64 * 416004 + 6404
            )
        );
    }
    assert!(record_offsets(3).is_none() && record_offsets(usize::MAX).is_none());
}
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(14)]
fn flash_record_projection_bindings_fit_the_pinned_read() {
    let index: usize = kani::any();
    let r = record_offsets(index);
    assert_eq!(r.is_some(), index < 3);
    if let Some(r) = r {
        assert!(r.suh < r.svh && r.svh < r.trellis && r.trellis + 409600 <= EXPERT_BYTES);
        assert!(r.suh % 4 == 0 && r.svh % 4 == 0 && r.trellis % 4 == 0);
        kani::cover!(index == 0);
        kani::cover!(r.trellis + 409600 == EXPERT_BYTES);
    }
}

/// Byte offset for a dense QSA KV append, and whether live suffixes require a new slab.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn qsa_append(offset: u32, published: u32, capacity: u32) -> Option<(u64, bool)> {
    (capacity <= FLASH_MAX_CONTEXT
        && offset < capacity
        && offset <= published
        && published <= capacity)
        .then(|| (u64::from(offset) * 1024, offset < published))
}

#[cfg(kani)]
#[kani::proof]
fn qsa_append_preserves_all_published_rows() {
    let (offset, published, capacity) = (kani::any(), kani::any(), kani::any());
    let plan = qsa_append(offset, published, capacity);
    let valid = offset <= published
        && published <= capacity
        && capacity <= FLASH_MAX_CONTEXT
        && offset < capacity;
    assert_eq!(plan.is_some(), valid);
    if let Some((bytes, copy)) = plan {
        assert_eq!(bytes / 1024, u64::from(offset));
        assert!(bytes + 1024 <= u64::from(capacity) * 1024);
        assert!(copy || offset == published);
        assert!(!copy || offset < published);
        kani::cover!(copy && offset > 0);
        kani::cover!(!copy && offset + 1 == capacity);
    }
}

#[cfg(test)]
#[test]
fn qsa_append_handles_empty_full_and_forked_prefixes() {
    assert_eq!(qsa_append(0, 0, 1), Some((0, false)));
    assert_eq!(qsa_append(510, 512, 512), Some((522240, true)));
    for (offset, end, cap) in [(1, 1, 1), (1, 0, 1), (0, 2, 1), (0, 0, 0), (0, 0, 262145)] {
        assert_eq!(qsa_append(offset, end, cap), None);
    }
}

/// Publish the new cursor only after a whole Flash prefill has succeeded.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn flash_prompt_end(offset: u32, tokens: usize, context: u32) -> Option<u32> {
    let n = u32::try_from(tokens).ok()?;
    let end = offset.checked_add(n)?;
    (n > 0 && context <= FLASH_MAX_CONTEXT && end <= context).then_some(end)
}
#[cfg(kani)]
#[kani::proof]
fn dense_flash_prompt_cursor_never_overflows_or_partially_accepts() {
    let (offset, tokens, context) = (kani::any(), kani::any(), kani::any());
    let end = flash_prompt_end(offset, tokens, context);
    let wide = u128::from(offset) + tokens as u128;
    assert_eq!(
        end.is_some(),
        tokens > 0 && context <= FLASH_MAX_CONTEXT && wide <= u128::from(context)
    );
    if let Some(end) = end {
        assert_eq!(u128::from(end), wide);
        kani::cover!(end == context && offset > 0);
        kani::cover!(offset == 0 && tokens > 1);
    }
}

#[cfg(test)]
#[test]
fn dense_prompt_rejects_partial_empty_and_overflowing_inputs() {
    assert_eq!(flash_prompt_end(0, 512, 512), Some(512));
    assert_eq!(flash_prompt_end(1, 262143, 262144), Some(262144));
    assert_eq!(flash_prompt_end(1, 512, 512), None);
    assert_eq!(flash_prompt_end(512, 0, 512), None);
    assert_eq!(flash_prompt_end(u32::MAX, usize::MAX, 512), None);
    assert_eq!(flash_prompt_end(0, 1, FLASH_MAX_CONTEXT + 1), None);
}

/// A fork reuses the current extent; a full slab grows up to the model context.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn qsa_slab_capacity(offset: u32, allocated: u32, limit: u32) -> Option<u32> {
    if allocated == 0
        || allocated > limit
        || limit > FLASH_MAX_CONTEXT
        || offset > allocated
        || offset >= limit
    {
        return None;
    }
    Some(if offset == allocated {
        (allocated * 2).min(limit)
    } else {
        allocated
    })
}

#[test]
fn qsa_slab_forks_preserve_capacity_and_growth_is_capped() {
    for (offset, allocated, limit, expected) in [
        (0, 32, 33, Some(32)),
        (31, 32, 33, Some(32)),
        (32, 32, 33, Some(33)),
        (32, 32, 512, Some(64)),
        (0, 0, 1, None),
        (33, 32, 64, None),
        (32, 32, 32, None),
        (0, 33, 32, None),
        (0, 1, FLASH_MAX_CONTEXT + 1, None),
    ] {
        assert_eq!(qsa_slab_capacity(offset, allocated, limit), expected);
    }
}

#[cfg(kani)]
#[kani::proof]
fn qsa_growth_fits_all_copied_and_appended_rows() {
    let (offset, allocated, limit) = (kani::any(), kani::any(), kani::any());
    let next = qsa_slab_capacity(offset, allocated, limit);
    assert_eq!(
        next.is_some(),
        allocated > 0
            && allocated <= limit
            && limit <= FLASH_MAX_CONTEXT
            && offset <= allocated
            && offset < limit
    );
    if let Some(next) = next {
        assert!(allocated <= next && next <= limit && offset < next);
        assert!((u64::from(offset) + 1) * 1024 <= u64::from(next) * 1024);
        assert!((offset + 1) / 4 <= next.div_ceil(4));
        assert_eq!(next > allocated, offset == allocated);
        kani::cover!(next == limit && next > allocated && limit == FLASH_MAX_CONTEXT);
        kani::cover!(next == allocated && offset % 4 == 3);
    }
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
fn rank_qsa_blocks(scores: &[f32], ids: &mut [u32], keep: usize) -> Option<usize> {
    if scores.len() != ids.len()
        || keep == 0
        || keep > ids.len()
        || scores.iter().any(|s| !s.is_finite())
    {
        return None;
    }
    for (i, id) in ids.iter_mut().enumerate() {
        *id = u32::try_from(i).ok()?;
    }
    let split = ids.len() - keep;
    ids.select_nth_unstable_by(split, |&a, &b| {
        scores[a as usize]
            .partial_cmp(&scores[b as usize])
            .unwrap()
            .then_with(|| a.cmp(&b))
    });
    ids[split..].sort_unstable();
    Some(split)
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn qsa_stable_ranking_preserves_best_blocks_and_time_order() {
    let values: [i8; 3] = kani::any();
    let scores = values.map(f32::from);
    let mut ids = [0; 3];
    let split = rank_qsa_blocks(&scores, &mut ids, 2).unwrap();
    assert_eq!(split, 1);
    let [discarded, first, second] = ids.map(|id| id as usize);
    assert!(first < second && second < 3 && discarded < 3);
    assert!(discarded != first && discarded != second);
    for kept in [first, second] {
        assert!((values[discarded], discarded) < (values[kept], kept));
    }
    kani::cover!(values[0] == values[1] && values[1] == values[2] && discarded == 0);
    kani::cover!(discarded == 2);
}

/// Stable ascending argsort followed by the last 512 blocks; ties at the cutoff keep later blocks.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn sparse_qsa_positions(scores: &[f32], length: u32) -> Option<Vec<u32>> {
    let blocks = length / 4;
    if length > FLASH_MAX_CONTEXT || blocks <= 512 || scores.len() != blocks as usize {
        return None;
    }
    let mut ids = vec![0; blocks as usize];
    let split = rank_qsa_blocks(scores, &mut ids, 512)?;
    let keep = &ids[split..];
    let mut tokens = Vec::with_capacity(2048 + (length % 4) as usize);
    for &block in keep.iter() {
        tokens.extend(block * 4..block * 4 + 4);
    }
    tokens.extend(blocks * 4..length);
    Some(tokens)
}

#[cfg(test)]
#[test]
fn sparse_qsa_positions_preserve_ties_and_current_tail() {
    for n in [2052, 2053, 2054, 2055, 4095, 4096] {
        let mut scores = vec![0.; (n / 4) as usize];
        let last = scores.len() - 1;
        scores[0] = 2.;
        scores[last] = -1.;
        let ids = sparse_qsa_positions(&scores, n).unwrap();
        let mut expected: Vec<u32> = (0..n / 4).collect();
        expected.sort_by(|&a, &b| scores[a as usize].partial_cmp(&scores[b as usize]).unwrap());
        let mut keep = expected[expected.len() - 512..].to_vec();
        keep.sort_unstable();
        let reference = keep
            .into_iter()
            .flat_map(|b| b * 4..b * 4 + 4)
            .chain(n / 4 * 4..n)
            .collect::<Vec<_>>();
        assert_eq!(ids, reference);
        assert_eq!(ids.len(), 2048 + (n % 4) as usize);
        assert!(ids.windows(2).all(|p| p[0] < p[1]));
    }
    for n in [2048, 2051, 262145] {
        assert!(sparse_qsa_positions(&[], n).is_none());
    }
    assert!(sparse_qsa_positions(&[f32::NAN; 513], 2052).is_none());
    assert!(sparse_qsa_positions(&[0.; 512], 2052).is_none());
}

#[cfg(all(
    test,
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
pub(crate) fn qsa_probe_row(t: u32) -> Vec<f32> {
    (0..2560)
        .map(|i| (((i * 17 + t * 83) % 257) as i32 - 128) as f32 / 237.)
        .collect()
}
#[cfg(all(
    test,
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
pub(crate) fn qsa_probe_checkpoint(n: u32) -> bool {
    matches!(
        n,
        1 | 4 | 32 | 33 | 512 | 513 | 1023 | 1024 | 2047..=2055 | 4095 | 4096
    )
}

/// Flash GQA has twelve query heads per KV head; sparse attention has at most 2051 positions.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn qsa_attention_blocks(arch: u8, count: u32) -> Option<u32> {
    if count < 1024 {
        return None;
    }
    match (arch, count) {
        (b's', 1024) => Some(64),
        (b's' | b'd', _) => Some(128),
        _ => None,
    }
}
#[cfg(test)]
#[test]
fn qsa_attention_two_pass_boundary() {
    for (a, n, b) in [
        (b's', 1023, None),
        (b's', 1024, Some(64)),
        (b's', 1025, Some(128)),
        (b'd', 1024, Some(128)),
        (b'g', 2051, None),
    ] {
        assert_eq!(qsa_attention_blocks(a, n), b);
    }
}

#[cfg(test)]
#[test]
fn native_demand_deduplicates_reads_but_preserves_routes_and_fails_closed() -> Result<()> {
    let cache = Rc::new(RefCell::new(ExpertCache::new(10)?));
    let plan = DemandPlan::reserve(&cache, &[9, 7, 9, 8, 7])?;
    assert_eq!(plan.reads.len(), 3);
    assert_eq!(plan.routes[0], plan.routes[2]);
    assert_eq!(plan.routes[1], plan.routes[4]);
    assert_eq!(cache.borrow().available_slots(), 7);
    assert!(!cache.borrow().contains(9));
    for &lease in plan.reads.iter().rev() {
        assert!(plan.publish(lease));
    }
    assert!(!plan.publish(plan.reads[0]));
    drop(plan);
    let hot = DemandPlan::reserve(&cache, &[7, 8, 9, 7])?;
    assert!(hot.reads.is_empty() && hot.routes.iter().all(|(_, hit)| *hit));
    drop(hot);
    let failed = DemandPlan::reserve(&cache, &[100, 101])?;
    let old = failed.reads[0];
    drop(failed);
    assert_eq!(cache.borrow().available_slots(), 10);
    assert!(!cache.borrow().contains(100));
    let retried = DemandPlan::reserve(&cache, &[100, 101])?;
    assert_ne!(old, retried.reads[0]);
    assert!(!retried.publish(old));
    assert!(retried.publish(retried.reads[0]));
    assert!(DemandPlan::reserve(&cache, &[]).is_err());
    assert!(DemandPlan::reserve(&cache, &[512]).is_err());
    assert!(DemandPlan::reserve(&cache, &(0..11).collect::<Vec<_>>()).is_err());
    drop(retried);
    cache.borrow_mut().generations.fill(u64::MAX);
    assert!(DemandPlan::reserve(&cache, &[211]).is_err());
    assert_eq!(cache.borrow().available_slots(), 10);
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(12)]
fn native_demand_generation_rejects_stale_and_duplicate_publication() {
    let mut cache = ExpertCache::new(10).unwrap();
    let expert: u32 = kani::any();
    kani::assume(expert < 512);
    let generation: u64 = kani::any();
    let valid: bool = kani::any();
    cache.keys[0] = Some(expert);
    cache.generations[0] = generation;
    cache.ages[0] = 79;
    let result = cache.begin_demand(0, expert);
    assert_eq!(result.is_ok(), generation < u64::MAX);
    if let Ok(lease) = result {
        assert_eq!(lease.generation, generation + 1);
        assert!(!cache.contains(expert));
        let stale = SlotLease {
            generation,
            ..lease
        };
        assert!(!cache.finish_demand(stale, true));
        assert_eq!(cache.pending_lease(0, expert), Some(lease));
        assert_eq!(cache.finish_demand(lease, valid), valid);
        assert_eq!(cache.contains(expert), valid);
        assert_eq!(cache.ages[0], 79);
        assert!(!cache.finish_demand(lease, true));
        kani::cover!(valid);
        kani::cover!(!valid);
    } else {
        assert_eq!(cache.keys[0], Some(expert));
        assert!(cache.pending[0].is_none());
        kani::cover!(generation == u64::MAX);
    }
}

/// At most four owned prefetch records, including completed results awaiting publication.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn native_prefetch_room(pending: usize, active: usize) -> Option<usize> {
    if pending > 4 || active > 4 {
        return None;
    }
    Some(4 - pending.max(active))
}
#[cfg(kani)]
#[kani::proof]
fn native_prefetch_room_bounds_owned_and_running_records() {
    let pending: usize = kani::any();
    let active: usize = kani::any();
    let room = native_prefetch_room(pending, active);
    assert_eq!(room.is_some(), pending <= 4 && active <= 4);
    if let Some(room) = room {
        assert!(room <= 4 && pending + room <= 4 && active + room <= 4);
        assert!(pending + room == 4 || active + room == 4);
        kani::cover!(room == 4);
        kani::cover!(room == 0 && pending == 4 && active == 0);
        kani::cover!(room == 0 && pending == 0 && active == 4);
    }
    kani::cover!(room.is_none());
}
#[cfg(test)]
#[test]
fn native_prefetch_room_rejects_invalid_ownership_counts() {
    for pending in 0..=5 {
        for active in 0..=5 {
            let room = native_prefetch_room(pending, active);
            assert_eq!(room.is_some(), pending <= 4 && active <= 4);
            if let Some(room) = room {
                assert_eq!(room + pending.max(active), 4);
            }
        }
    }
    assert_eq!(native_prefetch_room(usize::MAX, 0), None);
    assert_eq!(native_prefetch_room(0, usize::MAX), None);
}

/// Experimental HC kernels remain off unless explicitly selected on qualified M5.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn hc_kernel_flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    hc_chain_flag(value, false).map(|enabled| enabled && qualified_m5)
}

#[test]
fn hc_kernels_default_off_and_reject_non_literals() {
    for m5 in [false, true] {
        assert_eq!(hc_kernel_flag(None, m5), Some(false));
        assert_eq!(hc_kernel_flag(Some("0"), m5), Some(false));
        assert_eq!(hc_kernel_flag(Some("1"), m5), Some(m5));
        for value in ["", "01", "true", " 1", "1\n", "é"] {
            assert_eq!(hc_kernel_flag(Some(value), m5), None);
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn hc_kernel_flags_are_off_by_default_and_require_exact_m5_literals() {
    let m5: bool = kani::any();
    let bytes: [u8; 4] = kani::any();
    let length = usize::from(kani::any::<u8>() % 5);
    let present: bool = kani::any();
    if !present {
        assert_eq!(hc_kernel_flag(None, m5), Some(false));
        kani::cover!(m5);
        kani::cover!(!m5);
    } else if let Ok(value) = std::str::from_utf8(&bytes[..length]) {
        let result = hc_kernel_flag(Some(value), m5);
        assert_eq!(
            result,
            match value {
                "0" => Some(false),
                "1" => Some(m5),
                _ => None,
            }
        );
        kani::cover!(value == "1" && m5 && result == Some(true));
        kani::cover!(value == "1" && !m5 && result == Some(false));
        kani::cover!(value == "0");
        kani::cover!(value == "é" && result.is_none());
    }
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn gdn_private_epilogue_eligible(
    enabled: bool,
    trace: bool,
    private_f32_consumer: bool,
) -> bool {
    enabled && !trace && private_f32_consumer
}

#[test]
fn gdn_private_epilogue_keeps_public_trace_and_half_stream_fallbacks() {
    for enabled in [false, true] {
        for trace in [false, true] {
            for private_f32_consumer in [false, true] {
                let actual = gdn_private_epilogue_eligible(enabled, trace, private_f32_consumer);
                assert_eq!(
                    actual,
                    (enabled, trace, private_f32_consumer) == (true, false, true)
                );
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn gdn_private_epilogue_requires_every_actual_consumer_guard() {
    let enabled: bool = kani::any();
    let trace: bool = kani::any();
    let private_f32_consumer: bool = kani::any();
    let actual = gdn_private_epilogue_eligible(enabled, trace, private_f32_consumer);
    if actual {
        assert!(enabled && !trace && private_f32_consumer);
    }
    if enabled && !trace && private_f32_consumer {
        assert!(actual);
    }
    kani::cover!(actual);
    kani::cover!(!enabled && !actual);
    kani::cover!(trace && !actual);
    kani::cover!(!private_f32_consumer && !actual);
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn ar10_qsa_gate_eligible(
    enabled: bool,
    trace: bool,
    fp32: bool,
    keys: u32,
    sparse: bool,
) -> bool {
    enabled && !trace && fp32 && !sparse && (1..1024).contains(&keys)
}
#[test]
fn ar10_qsa_gate_keeps_short_dense_private_domain() {
    for keys in [0, 1, 87, 1023, 1024, u32::MAX] {
        assert_eq!(
            ar10_qsa_gate_eligible(true, false, true, keys, false),
            (1..1024).contains(&keys)
        );
        for (enabled, trace, fp32, sparse) in [
            (false, false, true, false),
            (true, true, true, false),
            (true, false, false, false),
            (true, false, true, true),
        ] {
            assert!(!ar10_qsa_gate_eligible(enabled, trace, fp32, keys, sparse));
        }
    }
}
#[cfg(kani)]
#[kani::proof]
fn ar10_qsa_gate_requires_every_actual_short_dense_guard() {
    let (enabled, trace, fp32, sparse): (bool, bool, bool, bool) = kani::any();
    let keys: u32 = kani::any();
    let actual = ar10_qsa_gate_eligible(enabled, trace, fp32, keys, sparse);
    if actual {
        assert!(enabled && !trace && fp32 && !sparse && keys > 0 && keys < 1024);
    }
    if enabled && !trace && fp32 && !sparse && keys > 0 && keys < 1024 {
        assert!(actual);
    }
    kani::cover!(actual && keys == 1);
    kani::cover!(actual && keys == 1023);
    kani::cover!(!actual && keys == 0);
    kani::cover!(!actual && sparse);
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn ar10_whole_gdn_eligible(
    enabled: bool,
    trace: bool,
    private_fp32: bool,
    simd: u32,
    threads: u32,
) -> bool {
    gdn_private_epilogue_eligible(enabled, trace, private_fp32) && simd == 32 && threads >= 512
}
#[test]
fn ar10_whole_gdn_preserves_capability_and_private_guards() {
    assert!(ar10_whole_gdn_eligible(true, false, true, 32, 512));
    for (e, t, f, s, n) in [
        (false, false, true, 32, 512),
        (true, true, true, 32, 512),
        (true, false, false, 32, 512),
        (true, false, true, 16, 512),
        (true, false, true, 32, 511),
    ] {
        assert!(!ar10_whole_gdn_eligible(e, t, f, s, n));
    }
}
#[cfg(kani)]
#[kani::proof]
fn ar10_whole_gdn_requires_every_real_private_capability_guard() {
    let (e, t, f): (bool, bool, bool) = kani::any();
    let s: u32 = kani::any();
    let n: u32 = kani::any();
    let actual = ar10_whole_gdn_eligible(e, t, f, s, n);
    if actual {
        assert!(e && !t && f && s == 32 && n >= 512);
    }
    if e && !t && f && s == 32 && n >= 512 {
        assert!(actual);
    }
    kani::cover!(actual && n == 512);
    kani::cover!(!actual && n == 511);
    kani::cover!(!actual && t);
    kani::cover!(!actual && s != 32);
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn ar20_head_gates_eligible(
    enabled: bool,
    whole: bool,
    trace: bool,
    private_f32: bool,
    simd: u32,
    threads: u32,
) -> bool {
    enabled && whole && ar10_whole_gdn_eligible(true, trace, private_f32, simd, threads)
}
#[test]
fn ar20_head_gates_require_the_actual_whole_private_capability() {
    assert!(ar20_head_gates_eligible(true, true, false, true, 32, 512));
    for values in [
        (false, true, false, true, 32, 512),
        (true, false, false, true, 32, 512),
        (true, true, true, true, 32, 512),
        (true, true, false, false, 32, 512),
        (true, true, false, true, 64, 512),
        (true, true, false, true, 32, 511),
    ] {
        assert!(!ar20_head_gates_eligible(
            values.0, values.1, values.2, values.3, values.4, values.5
        ));
    }
}
#[cfg(kani)]
#[kani::proof]
fn ar20_head_gates_require_all_actual_runtime_and_capability_guards() {
    let e: bool = kani::any();
    let w: bool = kani::any();
    let t: bool = kani::any();
    let f: bool = kani::any();
    let s: u32 = kani::any();
    let n: u32 = kani::any();
    let actual = ar20_head_gates_eligible(e, w, t, f, s, n);
    assert_eq!(actual, e && w && !t && f && s == 32 && n >= 512);
    kani::cover!(actual);
    kani::cover!(!actual && e && !w);
    kani::cover!(!actual && e && w && t);
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn ar20_qsa_chains_eligible(
    enabled: bool,
    trace: bool,
    fp32: bool,
    keys: u32,
    sparse: bool,
    capability: [u32; 4],
) -> bool {
    ar10_qsa_gate_eligible(enabled, trace, fp32, keys, sparse)
        && capability[0] == 32
        && capability[1] >= 128
        && capability[2] == 32
        && capability[3] >= 1024
}
#[test]
fn ar20_qsa_chains_guard_keeps_short_dense_private_capable_domain() {
    assert!(ar20_qsa_chains_eligible(
        true,
        false,
        true,
        1,
        false,
        [32, 128, 32, 1024]
    ));
    for keys in [0, 1024, u32::MAX] {
        assert!(!ar20_qsa_chains_eligible(
            true,
            false,
            true,
            keys,
            false,
            [32, 128, 32, 1024]
        ));
    }
    for caps in [
        [64, 128, 32, 1024],
        [32, 127, 32, 1024],
        [32, 128, 64, 1024],
        [32, 128, 32, 1023],
    ] {
        assert!(!ar20_qsa_chains_eligible(
            true, false, true, 87, false, caps
        ));
    }
}
#[cfg(kani)]
#[kani::proof]
fn ar20_qsa_chain_selection_preserves_dense_private_and_both_pipeline_capabilities() {
    let (e, t, f, s): (bool, bool, bool, bool) = kani::any();
    let k: u32 = kani::any();
    let c: [u32; 4] = kani::any();
    let a = ar20_qsa_chains_eligible(e, t, f, k, s, c);
    assert_eq!(
        a,
        e && !t
            && f
            && !s
            && k > 0
            && k < 1024
            && c[0] == 32
            && c[1] >= 128
            && c[2] == 32
            && c[3] >= 1024
    );
    kani::cover!(a && k == 1);
    kani::cover!(a && k == 1023);
    kani::cover!(!a && s);
    kani::cover!(!a && c[3] < 1024);
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn ar20_resident_pair_eligible(
    enabled: bool,
    shape: [[u32; 2]; 2],
    format: [[u32; 3]; 2],
    splits: [u32; 2],
    resident: [bool; 2],
    bias: [bool; 2],
    capability: [u32; 2],
) -> bool {
    enabled
        && shape == [[2560, 10240], [2560, 6144]]
        && format == [[4, 2, 2]; 2]
        && splits == [1; 2]
        && resident == [true; 2]
        && bias == [false; 2]
        && capability[0] == 32
        && capability[1] >= 128
}
#[test]
fn ar20_resident_bundle_requires_exact_live_canonical_pair() {
    assert!(ar20_resident_pair_eligible(
        true,
        [[2560, 10240], [2560, 6144]],
        [[4, 2, 2]; 2],
        [1; 2],
        [true; 2],
        [false; 2],
        [32, 128]
    ));
    assert!(!ar20_resident_pair_eligible(
        true,
        [[2560, 10240], [2560, 6144]],
        [[4, 2, 1]; 2],
        [1; 2],
        [true; 2],
        [false; 2],
        [32, 128]
    ));
    assert!(!ar20_resident_pair_eligible(
        true,
        [[2560, 10240], [2560, 6144]],
        [[4, 2, 2]; 2],
        [1; 2],
        [true; 2],
        [false, true],
        [32, 128]
    ));
}
#[cfg(kani)]
#[kani::proof]
#[kani::unwind(40)]
fn ar20_resident_bundle_preserves_exact_formats_offsets_and_complete_output_grids() {
    let e: bool = kani::any();
    let shape: [[u32; 2]; 2] = kani::any();
    let f: [[u32; 3]; 2] = kani::any();
    let splits: [u32; 2] = kani::any();
    let resident: [bool; 2] = kani::any();
    let bias: [bool; 2] = kani::any();
    let c: [u32; 2] = kani::any();
    let a = ar20_resident_pair_eligible(e, shape, f, splits, resident, bias, c);
    assert_eq!(
        a,
        e && shape == [[2560, 10240], [2560, 6144]]
            && f == [[4, 2, 2]; 2]
            && splits == [1; 2]
            && resident == [true; 2]
            && bias == [false; 2]
            && c[0] == 32
            && c[1] >= 128
    );
    if a {
        assert_eq!(320u32 * 32, shape[0][1]);
        assert_eq!(192u32 * 32, shape[1][1]);
        assert_eq!(320 + 192, 512);
    }
    kani::cover!(a);
    kani::cover!(!a && f[0][2] == 1);
    kani::cover!(!a && bias[1]);
}
