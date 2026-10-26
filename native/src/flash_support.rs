//! CPU contracts for Flash Next's disk rows and exact expert cache.
use anyhow::Result;

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

#[cfg(any(feature = "mlx", test, kani))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ExpertRead {
    All,
    GateUp,
    Down,
}

#[cfg(any(feature = "mlx", test, kani))]
impl ExpertRead {
    pub(crate) fn layout(self) -> (std::ops::Range<usize>, u64, u64) {
        match self {
            Self::All => (0..12, 0, EXPERT_BYTES),
            Self::GateUp => (4..12, EXPERT_BYTES / 3, EXPERT_BYTES / 3 * 2),
            Self::Down => (0..4, 0, EXPERT_BYTES / 3),
        }
    }
}

#[test]
fn staged_expert_reads_partition_the_original_records() {
    for slot in 0..512 {
        let destinations = expert_slot_destinations(slot);
        let mut total = 0;
        for stage in [ExpertRead::Down, ExpertRead::GateUp] {
            let (range, offset, bytes) = stage.layout();
            assert_eq!(offset, total);
            assert_eq!(
                destinations[range].iter().map(|r| r.2 as u64).sum::<u64>(),
                bytes
            );
            total += bytes;
        }
        assert_eq!(total, EXPERT_BYTES);
        for &(bank, start, len) in &destinations[..4] {
            for &(other_bank, other_start, other_len) in &destinations[4..] {
                assert!(
                    bank != other_bank
                        || start + len <= other_start
                        || other_start + other_len <= start
                );
            }
        }
    }
    assert_eq!(ExpertRead::All.layout(), (0..12, 0, EXPERT_BYTES));
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(16)]
fn actual_staged_record_partition() {
    let slot: usize = kani::any();
    kani::assume(slot < 512);
    let destinations = expert_slot_destinations(slot);
    let gate_up: bool = kani::any();
    let stage = if gate_up {
        ExpertRead::GateUp
    } else {
        ExpertRead::Down
    };
    let (range, offset, bytes) = stage.layout();
    assert!(range.start < range.end && range.end <= destinations.len());
    assert_eq!(
        destinations[range].iter().map(|r| r.2 as u64).sum::<u64>(),
        bytes
    );
    assert!(offset + bytes <= EXPERT_BYTES);
    for &(bank, start, len) in &destinations[..4] {
        for &(other_bank, other_start, other_len) in &destinations[4..] {
            assert!(
                bank != other_bank
                    || start + len <= other_start
                    || other_start + other_len <= start
            );
        }
    }
    kani::cover!(gate_up && slot == 511);
    kani::cover!(!gate_up && slot == 0);
}

#[cfg(any(feature = "mlx", test, kani))]
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
#[cfg(any(feature = "mlx", test, kani))]
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

pub const EXPERT_BYTES: u64 = 1_248_012;
pub const NGRAM_DIM: usize = 160;
pub const NGRAM_WORDS: usize = 41;

#[cfg(any(feature = "mlx", test, kani))]
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

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn parse_layer_slots(value: &str, uniform: usize) -> Option<[usize; 48]> {
    let mut parts = value.split(',');
    let mut slots = [0; 48];
    for slot in &mut slots {
        *slot = parts.next()?.parse().ok()?;
    }
    (parts.next().is_none() && valid_layer_slots(&slots, uniform)).then_some(slots)
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

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn parse_lru_override(value: &str) -> Option<bool> {
    match value {
        "auto" => Some(false),
        "lru" => Some(true),
        _ => None,
    }
}

#[cfg(any(feature = "mlx", test, kani))]
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

#[cfg(any(feature = "mlx", test, kani))]
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

    #[kani::proof]
    #[kani::unwind(81)]
    fn unique_expert_prefix_is_bounded_for_eighty_routes() {
        let ids: [u32; 80] = kani::any();
        let rows: usize = kani::any();
        kani::assume((1..=8).contains(&rows));
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
            kani::cover!(prefix == 0);
            kani::cover!(prefix == 8);
        }
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

/// Reservation is atomic: every routed expert gets a distinct protected slot.
/// A failed read must invalidate its reservation before it can become a hit.
pub struct ExpertCache {
    keys: Vec<Option<u32>>,
    pending: Vec<Option<u32>>,
    prefetched: Vec<bool>,
    predicted: [bool; 512],
    predictive_lru: bool,
    counts: [u32; 512],
    ages: Vec<u64>,
    clock: u64,
    pub prefetch_hits: u64,
}

/// Unfilled demand reservations must never survive an error or unwind.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) struct ExpertFill {
    cache: Rc<RefCell<ExpertCache>>,
    complete: bool,
}

#[cfg(any(feature = "mlx", test, kani))]
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

#[cfg(any(feature = "mlx", test, kani))]
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
            prefetched: vec![false; slots],
            predicted: [false; 512],
            predictive_lru: false,
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

    pub(crate) fn enable_predictive_lru(&mut self) {
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
        self.keys[slot] = None;
        self.prefetched[slot] = false;
        self.pending[slot] = Some(id);
        Ok(Some(slot))
    }

    /// Only the inference thread publishes a completed write. Demand may have won.
    pub fn publish_prefetch(&mut self, slot: usize, id: u32, valid: bool) -> bool {
        if self.pending.get(slot) != Some(&Some(id)) {
            return false;
        }
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

#[cfg(test)]
mod tests {
    use super::*;
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
    #[kani::unwind(513)]
    fn resident_slot_snapshot_excludes_pending_and_preserves_lfu() {
        let mut cache = ExpertCache::new(20).unwrap();
        for slot in 0..20 {
            let id: u32 = kani::any();
            kani::assume(id < 512);
            cache.keys[slot] = if kani::any() { Some(id) } else { None };
            cache.pending[slot] = if kani::any() { Some(id) } else { None };
        }
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
        assert!(cache.counts.iter().all(|&count| count == 0));
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

    #[kani::proof]
    #[kani::unwind(192)]
    fn layer_slot_parser_checks_last_two_utf8_bytes_and_full_shape() {
        let bytes: [u8; 2] = kani::any();
        if let Ok(last) = std::str::from_utf8(&bytes) {
            let text = "53,".repeat(47) + last;
            let parsed = parse_layer_slots(&text, 53);
            assert_eq!(parsed.is_some(), last == "53");
            if let Some(slots) = parsed {
                assert_eq!(slots, [53; 48]);
            }
            kani::cover!(last == "53");
            kani::cover!(last != "53" && parsed.is_none());
        }
    }
    #[kani::proof]
    #[kani::unwind(27)]
    fn failed_fill_invalidates_ready_keys_without_releasing_pending_slots() {
        let keys: [Option<u32>; 24] = kani::any();
        let pending: [Option<u32>; 24] = kani::any();
        let prefetched: [bool; 24] = kani::any();
        let complete: bool = kani::any();
        let cache = Rc::new(RefCell::new(ExpertCache::new(24).unwrap()));
        {
            let mut state = cache.borrow_mut();
            state.keys.copy_from_slice(&keys);
            state.pending.copy_from_slice(&pending);
            state.prefetched.copy_from_slice(&prefetched);
        }
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
