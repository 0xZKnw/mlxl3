//! CPU guards shared by Qwen loaders; rejected configurations allocate no weights.

/// Q/gate, K/V and output projection widths, in the backend's signed shape domain.
pub(crate) fn attention_widths(heads: i32, kv_heads: i32, head_dim: i32) -> Option<[i32; 3]> {
    if heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0 || head_dim <= 0 {
        return None;
    }
    let output = heads.checked_mul(head_dim)?;
    Some([
        output.checked_mul(2)?,
        kv_heads.checked_mul(head_dim)?,
        output,
    ])
}

/// Key, value and concatenated convolution widths, preserving the supported GDN geometry.
pub(crate) fn gated_delta_widths(
    key_heads: i32,
    value_heads: i32,
    key_dim: i32,
    value_dim: i32,
) -> Option<[i32; 3]> {
    if key_heads <= 0
        || value_heads <= 0
        || value_heads % key_heads != 0
        || key_dim != 128
        || value_dim <= 0
        || value_dim % 8 != 0
    {
        return None;
    }
    let keys = key_heads.checked_mul(key_dim)?;
    let values = value_heads.checked_mul(value_dim)?;
    Some([keys, values, keys.checked_mul(2)?.checked_add(values)?])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attention_geometry_rejects_invalid_or_overflowing_projection_widths() {
        assert_eq!(attention_widths(24, 4, 256), Some([12288, 1024, 6144]));
        assert_eq!(attention_widths(16, 2, 256), Some([8192, 512, 4096]));
        for (heads, kv_heads, head_dim) in [
            (0, 1, 128),
            (1, 0, 128),
            (3, 2, 128),
            (-1, 1, 128),
            (1, -1, 128),
            (1, 1, 0),
            (1, 1, -128),
            (i32::MIN, -1, 128),
            (i32::MAX, 1, 2),
            (1, 1, i32::MAX),
            (1 << 22, 1, 256),
        ] {
            assert_eq!(attention_widths(heads, kv_heads, head_dim), None);
        }
        let largest = i32::MAX / 2;
        assert_eq!(
            attention_widths(1, 1, largest),
            Some([largest * 2, largest, largest])
        );
        assert_eq!(attention_widths(1, 1, largest + 1), None);
    }

    #[test]
    fn gated_delta_geometry_rejects_overflow_before_convolution_width_arithmetic() {
        assert_eq!(
            gated_delta_widths(16, 48, 128, 128),
            Some([2048, 6144, 10240])
        );
        assert_eq!(
            gated_delta_widths(16, 32, 128, 128),
            Some([2048, 4096, 8192])
        );
        for (key_heads, value_heads, key_dim, value_dim) in [
            (0, 32, 128, 128),
            (16, 0, 128, 128),
            (3, 4, 128, 128),
            (16, 32, 64, 128),
            (16, 32, 128, 7),
            (16, 32, 128, 0),
            (-1, -1, 128, 128),
            (i32::MAX, i32::MAX, 128, 128),
            (1, 1, 128, i32::MAX - 7),
            (1 << 23, 1 << 23, 128, 8),
        ] {
            assert_eq!(
                gated_delta_widths(key_heads, value_heads, key_dim, value_dim),
                None
            );
        }
        let largest_value = i32::MAX - 263;
        assert_eq!(
            gated_delta_widths(1, 1, 128, largest_value),
            Some([128, largest_value, i32::MAX - 7])
        );
        assert_eq!(gated_delta_widths(1, 1, 128, largest_value + 8), None);
    }

    #[test]
    fn projection_widths_match_independent_wide_integer_oracles() {
        let dimensions = [
            i32::MIN,
            -1,
            0,
            1,
            2,
            3,
            4,
            8,
            16,
            24,
            48,
            128,
            256,
            1 << 23,
            i32::MAX / 2,
            i32::MAX,
        ];
        for heads in dimensions {
            for other_heads in dimensions {
                for dim in dimensions {
                    let output = i64::from(heads) * i64::from(dim);
                    let valid = heads > 0
                        && other_heads > 0
                        && heads % other_heads == 0
                        && dim > 0
                        && output * 2 <= i64::from(i32::MAX);
                    let actual = attention_widths(heads, other_heads, dim);
                    assert_eq!(actual.is_some(), valid);
                    if let Some(widths) = actual {
                        assert_eq!(
                            widths.map(i64::from),
                            [output * 2, i64::from(other_heads) * i64::from(dim), output],
                        );
                    }
                    let conv = 2 * i64::from(heads) * 128 + i64::from(other_heads) * i64::from(dim);
                    let valid = heads > 0
                        && other_heads > 0
                        && other_heads % heads == 0
                        && dim > 0
                        && dim % 8 == 0
                        && conv <= i64::from(i32::MAX);
                    let actual = gated_delta_widths(heads, other_heads, 128, dim);
                    assert_eq!(actual.is_some(), valid);
                    if let Some(widths) = actual {
                        assert_eq!(
                            widths.map(i64::from),
                            [
                                i64::from(heads) * 128,
                                i64::from(other_heads) * i64::from(dim),
                                conv
                            ],
                        );
                    }
                }
            }
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn qwen_attention_widths_match_wide_arithmetic_or_reject() {
    let (heads, kv_heads, dim): (i32, i32, i32) = kani::any();
    let result = attention_widths(heads, kv_heads, dim);
    let shape_valid = heads > 0 && kv_heads > 0 && heads % kv_heads == 0 && dim > 0;
    let output = i64::from(heads) * i64::from(dim);
    let kv = i64::from(kv_heads) * i64::from(dim);
    assert_eq!(
        result.is_some(),
        shape_valid && output * 2 <= i64::from(i32::MAX)
    );
    if let Some([query, key_value, projected]) = result {
        assert_eq!(i64::from(query), output * 2);
        assert_eq!(i64::from(key_value), kv);
        assert_eq!(i64::from(projected), output);
    }
    kani::cover!(result.is_some() && heads == 24 && kv_heads == 4 && dim == 256);
    kani::cover!(shape_valid && result.is_none());
    kani::cover!(!shape_valid && result.is_none());
}

#[cfg(kani)]
#[kani::proof]
fn qwen_gated_delta_widths_match_wide_arithmetic_or_reject() {
    let (key_heads, value_heads, key_dim, value_dim): (i32, i32, i32, i32) = kani::any();
    let result = gated_delta_widths(key_heads, value_heads, key_dim, value_dim);
    let shape_valid = key_heads > 0
        && value_heads > 0
        && value_heads % key_heads == 0
        && key_dim == 128
        && value_dim > 0
        && value_dim % 8 == 0;
    let keys = i64::from(key_heads) * i64::from(key_dim);
    let values = i64::from(value_heads) * i64::from(value_dim);
    // Only the valid key_dim=128 domain is relevant to the sum oracle.
    if shape_valid {
        let conv = 2 * keys + values;
        assert_eq!(result.is_some(), conv <= i64::from(i32::MAX));
        if let Some([actual_keys, actual_values, actual_conv]) = result {
            assert_eq!(i64::from(actual_keys), keys);
            assert_eq!(i64::from(actual_values), values);
            assert_eq!(i64::from(actual_conv), conv);
        }
    } else {
        assert_eq!(result, None);
    }
    kani::cover!(result.is_some() && key_heads == 16 && value_heads == 48);
    kani::cover!(shape_valid && result.is_none());
    kani::cover!(!shape_valid && result.is_none());
}
