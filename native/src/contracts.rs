//! Checked arithmetic shared by checkpoint and GPU-facing production paths.

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn use_tensor_ops(rows: i32, capable: bool) -> bool {
    capable && rows >= 24
}

/// Exact QMV batches reuse decoded weights. Odd pairs reserve one padded row;
/// kernels must zero its loads and never write it. Preserve existing large-M
/// choices while enabling the small speculative verify shapes.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn qmv_batch_layout(rows: i32, grouped: bool, wide: bool) -> Option<(i32, i32)> {
    if !(2..24).contains(&rows) {
        return None;
    }
    let paired = matches!(rows, 2 | 4)
        || if grouped {
            matches!(rows, 6 | 8)
        } else {
            (5..=8).contains(&rows)
        };
    let mb = if rows == 3 && !grouped && wide {
        3
    } else if paired {
        2
    } else {
        1
    };
    Some((mb, (rows + mb - 1) / mb))
}

#[test]
fn qmv_pair_layout_covers_rows_and_bounds_padding() {
    for grouped in [false, true] {
        for wide in [false, true] {
            for rows in [i32::MIN, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 23, 24, i32::MAX] {
                match qmv_batch_layout(rows, grouped, wide) {
                    Some((mb, groups)) => {
                        assert!((2..24).contains(&rows));
                        assert!((groups - 1) * mb < rows && groups * mb >= rows);
                        assert!(groups * mb - rows < mb);
                        for row in 0..rows {
                            assert!(row / mb < groups);
                        }
                        if matches!(rows, 2 | 4) {
                            assert_eq!(mb, 2);
                        }
                        if rows == 3 {
                            assert_eq!(
                                (mb, groups),
                                if !grouped && wide { (3, 1) } else { (1, 3) }
                            );
                        }
                    }
                    None => assert!(!(2..24).contains(&rows)),
                }
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn qmv_pairs_cover_every_valid_row_with_at_most_one_padding_row() {
    let rows: i32 = kani::any();
    let grouped: bool = kani::any();
    let wide: bool = kani::any();
    match qmv_batch_layout(rows, grouped, wide) {
        Some((mb, groups)) => {
            assert!((2..24).contains(&rows));
            assert!((1..=3).contains(&mb));
            assert!((groups - 1) * mb < rows && groups * mb >= rows);
            assert!(groups * mb - rows < mb);
            assert!(groups * mb - rows <= 1);
            assert_eq!(mb == 3, rows == 3 && !grouped && wide);
            if mb == 3 {
                assert!(rows == 3 && !grouped && wide && groups == 1);
            }
            let row: i32 = kani::any();
            kani::assume(0 <= row && row < rows);
            assert!(row / mb < groups);
            kani::cover!(rows == 3 && grouped && row == 2);
            kani::cover!(mb == 3 && row == 2);
            kani::cover!(rows == 4 && !grouped && row == 3);
        }
        None => assert!(!(2..24).contains(&rows)),
    }
}

#[test]
fn older_gpus_never_select_tensor_ops() {
    for rows in [i32::MIN, 0, 1, 23, 24, 25, 256, i32::MAX] {
        assert!(!use_tensor_ops(rows, false));
        assert_eq!(use_tensor_ops(rows, true), rows >= 24);
    }
}

/// Only partial commits need history; keep the existing nonempty T=1 output.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn gdn_history_rows(time: i32) -> Option<i32> {
    match time {
        1 => Some(1),
        2..=8 => Some(time - 1),
        _ => None,
    }
}

#[test]
fn gdn_history_capacity_covers_partial_commits() {
    for time in [i32::MIN, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, i32::MAX] {
        match gdn_history_rows(time) {
            Some(rows) => {
                assert!((1..=8).contains(&time));
                assert_eq!(rows, if time == 1 { 1 } else { time - 1 });
                for retained in 1..time {
                    assert!((0..rows).contains(&(retained - 1)));
                }
            }
            None => assert!(!(1..=8).contains(&time)),
        }
    }
}

/// Reuse only a whole saved prefill boundary, never a partial recurrent state.
/// The caller separately checks resident model, draft and conversation identity.
pub fn reusable_prefix_len(saved: &[u32], next: &[u32], chunk_size: usize) -> usize {
    if chunk_size != 0
        && !saved.is_empty()
        && saved.len().is_multiple_of(chunk_size)
        && next.starts_with(saved)
    {
        saved.len()
    } else {
        0
    }
}

pub(crate) fn tensor_bytes(shape: &[usize], item_size: u64) -> Option<u64> {
    shape
        .iter()
        .try_fold(item_size, |bytes, &dim| bytes.checked_mul(dim as u64))
}

#[cfg(any(feature = "mlx", kani))]
pub(crate) fn array_bytes(shape: &[i32], item_size: usize) -> Option<usize> {
    shape.iter().try_fold(item_size, |bytes, &dim| {
        bytes.checked_mul(usize::try_from(dim).ok()?)
    })
}

pub(crate) fn valid_data_range(start: u64, end: u64, payload: u64, bytes: u64) -> bool {
    end >= start && end <= payload && end - start == bytes
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), kani))]
pub(crate) fn packed_words(states: usize, k: usize) -> Option<usize> {
    ((1..=8).contains(&k) && states.is_multiple_of(256))
        .then(|| states.checked_div(256)?.checked_mul(16)?.checked_mul(k))?
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), kani))]
pub(crate) fn decoded_states(words: usize, k: usize) -> Option<usize> {
    ((1..=8).contains(&k) && words.is_multiple_of(16 * k))
        .then(|| words.checked_div(16 * k)?.checked_mul(256))?
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), kani))]
pub(crate) fn qmv_words(rows: usize, cols: usize, k: usize) -> Option<usize> {
    (rows > 0
        && cols > 0
        && rows.is_multiple_of(16)
        && cols.is_multiple_of(16)
        && (1..=8).contains(&k))
    .then(|| {
        rows.checked_div(16)?
            .checked_mul(cols.checked_div(16)?)?
            .checked_mul(16)?
            .checked_mul(k)
    })?
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn tensor_dispatch_requires_hardware_and_valid_batch() {
        let rows: i32 = kani::any();
        let capable: bool = kani::any();
        let selected = use_tensor_ops(rows, capable);
        assert!(!selected || (capable && rows >= 24));
        assert_eq!(selected, capable && rows >= 24);
        kani::cover!(rows == 24 && selected);
        kani::cover!(rows > 24 && !capable && !selected);
    }

    #[kani::proof]
    fn gdn_tape_prefix_covers_all_valid_commits() {
        let time: i32 = kani::any();
        let retained: i32 = kani::any();
        let result = gdn_tape_prefix(time, retained);
        assert_eq!(
            result.is_some(),
            time >= 1 && time <= 8 && retained >= 1 && retained <= time
        );
        if let Some(count) = result {
            assert_eq!(count, retained);
            assert!(count <= 8);
            kani::cover!(time == 8 && count == 7);
            kani::cover!(time == 8 && count == 8);
            kani::cover!(time == 1 && count == 1);
        } else {
            kani::cover!(time == i32::MAX);
            kani::cover!(retained == i32::MIN);
        }
    }

    #[kani::proof]
    fn mtp_cache_append_cannot_wrap_or_accept_empty_rows() {
        let offset: i32 = kani::any();
        let rows: i32 = kani::any();
        let end = mtp_cache_end(offset, rows);
        let wide = i64::from(offset) + i64::from(rows);
        assert_eq!(
            end.is_some(),
            offset >= 0 && rows > 0 && wide <= i64::from(i32::MAX)
        );
        if let Some(end) = end {
            assert_eq!(i64::from(end), wide);
            assert!(end > offset);
            kani::cover!(offset == 0 && rows == 1);
            kani::cover!(end == i32::MAX);
        } else {
            kani::cover!(rows == 0);
            kani::cover!(offset == i32::MAX && rows == 1);
        }
    }

    #[kani::proof]
    fn gdn_history_capacity_is_exact_and_covers_partial_commits() {
        let time: i32 = kani::any();
        let retained: i32 = kani::any();
        match gdn_history_rows(time) {
            Some(rows) => {
                assert!(time >= 1 && time <= 8);
                assert!(rows >= 1 && rows <= 7);
                assert_eq!(rows, if time == 1 { 1 } else { time - 1 });
                if retained > 0 && retained < time {
                    assert!(retained - 1 >= 0 && retained - 1 < rows);
                    kani::cover!(retained == 1);
                    kani::cover!(retained == time - 1 && time == 8);
                }
                kani::cover!(time == 1);
            }
            None => {
                assert!(time < 1 || time > 8);
                kani::cover!(time == i32::MIN);
            }
        }
    }

    #[kani::proof]
    #[kani::unwind(34)] // starts_with uses memcmp over up to 8 * 4 bytes.
    fn cached_prefix_requires_exact_ids_and_chunk_boundary() {
        let saved: [u32; 8] = kani::any();
        let next: [u32; 8] = kani::any();
        let saved_len: u8 = kani::any();
        let next_len: u8 = kani::any();
        let chunk: u8 = kani::any();
        kani::assume(saved_len <= 8 && next_len <= 8);
        let a = &saved[..usize::from(saved_len)];
        let b = &next[..usize::from(next_len)];
        let actual = reusable_prefix_len(a, b, usize::from(chunk));
        let mut matches = saved_len > 0 && saved_len <= next_len && chunk > 0;
        if chunk > 0 {
            matches &= saved_len % chunk == 0;
        }
        for index in 0..usize::from(saved_len.min(next_len)) {
            matches &= saved[index] == next[index];
        }
        assert_eq!(actual, if matches { usize::from(saved_len) } else { 0 });
        assert!(actual <= b.len());
        kani::cover!(actual > 0 && actual < b.len());
        kani::cover!(actual > 0 && actual == b.len());
        kani::cover!(actual == 0 && chunk == 0);
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn tensor_bytes_matches_wide_arithmetic_for_four_dimensions() {
        let dims: [u8; 4] = kani::any();
        let selector: u8 = kani::any();
        kani::assume(selector < 4);
        let item_size = [1u64, 2, 4, 8][selector as usize];
        let shape = dims.map(usize::from);
        let wide = shape
            .iter()
            .fold(item_size as u128, |n, &dim| n * dim as u128);
        let actual = tensor_bytes(&shape, item_size);
        assert_eq!(actual.map(u128::from), Some(wide));
        assert!(tensor_bytes(&[usize::MAX, 2], 1).is_none());
        kani::cover!(actual.is_some());
    }

    #[kani::proof]
    #[kani::unwind(6)]
    fn array_bytes_rejects_negative_and_overflowing_dimensions() {
        let dims: [i8; 4] = kani::any();
        let selector: u8 = kani::any();
        kani::assume(selector < 4);
        let item_size = [1usize, 2, 4, 8][selector as usize];
        let shape = dims.map(i32::from);
        let actual = array_bytes(&shape, item_size);
        let nonnegative = dims.iter().all(|&dim| dim >= 0);
        let wide = dims.iter().fold(item_size as u128, |n, &dim| {
            n * u128::from(dim.max(0) as u8)
        });
        assert_eq!(actual.is_some(), nonnegative);
        if let Some(actual) = actual {
            assert_eq!(actual as u128, wide);
        }
        kani::cover!(actual.is_none());
    }

    #[kani::proof]
    fn accepted_data_ranges_are_exact_and_in_bounds() {
        let start: u64 = kani::any();
        let end: u64 = kani::any();
        let payload: u64 = kani::any();
        let bytes: u64 = kani::any();
        if valid_data_range(start, end, payload, bytes) {
            assert!(start <= end);
            assert!(end <= payload);
            assert_eq!(u128::from(end) - u128::from(start), u128::from(bytes));
        }
        kani::cover!(valid_data_range(0, 0, payload, 0));
        kani::cover!(!valid_data_range(start, end, payload, bytes));
    }

    #[kani::proof]
    fn exl3_pack_and_decode_shapes_are_inverse_for_u32_sized_tiles() {
        let tiles: u16 = kani::any();
        let k: u8 = kani::any();
        kani::assume((1..=8).contains(&k));
        let states = usize::from(tiles) * 256;
        let words = packed_words(states, usize::from(k)).unwrap();
        assert_eq!(words, usize::from(tiles) * 16 * usize::from(k));
        assert_eq!(decoded_states(words, usize::from(k)), Some(states));
        kani::cover!(k == 1);
        kani::cover!(k == 8);
    }

    #[kani::proof]
    fn qmv_shape_matches_wide_arithmetic() {
        let row_tiles: u8 = kani::any();
        let col_tiles: u8 = kani::any();
        let k: u8 = kani::any();
        kani::assume(row_tiles > 0 && col_tiles > 0 && (1..=8).contains(&k));
        let rows = usize::from(row_tiles) * 16;
        let cols = usize::from(col_tiles) * 16;
        let expected = u128::from(row_tiles) * u128::from(col_tiles) * 16 * u128::from(k);
        let actual = qmv_words(rows, cols, usize::from(k));
        assert_eq!(actual.map(|n| n as u128), Some(expected));
        kani::cover!(k == 1);
        kani::cover!(k == 8);
    }
}
/// Valid accepted prefix of a bounded GDN verification transaction.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn gdn_tape_prefix(time: i32, retained: i32) -> Option<i32> {
    ((1..=8).contains(&time) && retained > 0 && retained <= time).then_some(retained)
}

#[test]
fn gdn_tape_requires_a_valid_commit() {
    for time in 1..=8 {
        for retained in 1..=time {
            assert_eq!(gdn_tape_prefix(time, retained), Some(retained));
        }
    }
    for (time, retained) in [(0, 0), (9, 1), (2, 0), (3, 4), (8, -1), (i32::MAX, 1)] {
        assert_eq!(gdn_tape_prefix(time, retained), None);
    }
}

/// A positive append must fit the signed MLX/RoPE cache position.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn mtp_cache_end(offset: i32, rows: i32) -> Option<i32> {
    if offset < 0 || rows <= 0 {
        return None;
    }
    offset.checked_add(rows)
}

#[test]
fn mtp_cache_append_checks_empty_negative_and_overflowing_positions() {
    for (offset, rows, expected) in [
        (0, 1, Some(1)),
        (255, 1, Some(256)),
        (256, 255, Some(511)),
        (i32::MAX - 1, 1, Some(i32::MAX)),
        (i32::MAX, 1, None),
        (0, 0, None),
        (0, -1, None),
        (-1, 1, None),
        (i32::MIN, i32::MAX, None),
    ] {
        assert_eq!(mtp_cache_end(offset, rows), expected);
    }
}
