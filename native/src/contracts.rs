//! Checked arithmetic shared by checkpoint and GPU-facing production paths.

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

pub(crate) fn packed_words(states: usize, k: usize) -> Option<usize> {
    ((1..=8).contains(&k) && states.is_multiple_of(256))
        .then(|| states.checked_div(256)?.checked_mul(16)?.checked_mul(k))?
}

pub(crate) fn decoded_states(words: usize, k: usize) -> Option<usize> {
    ((1..=8).contains(&k) && words.is_multiple_of(16 * k))
        .then(|| words.checked_div(16 * k)?.checked_mul(256))?
}

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
