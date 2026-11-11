//! CPU guards shared by Qwen loaders; rejected configurations allocate no weights.

/// Validate before a token can change any layer state. No wrapping cursor arithmetic.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn native_step_end(token: u32, vocab: u32, offset: u32, limit: u32) -> Option<u32> {
    if token >= vocab || offset >= limit {
        return None;
    }
    offset.checked_add(1)
}

#[cfg(test)]
#[test]
fn native_cursor_checks_token_and_context_before_advancing() {
    assert_eq!(native_step_end(248319, 248320, 4095, 4096), Some(4096));
    assert_eq!(native_step_end(248320, 248320, 0, 4096), None);
    assert_eq!(native_step_end(0, 0, 0, 1), None);
    assert_eq!(native_step_end(0, 1, 4096, 4096), None);
    assert_eq!(
        native_step_end(0, 1, u32::MAX - 1, u32::MAX),
        Some(u32::MAX)
    );
    assert_eq!(native_step_end(0, 1, u32::MAX, u32::MAX), None);
}

#[cfg(kani)]
#[kani::proof]
fn native_qwen_step_validates_before_state_publication() {
    let (token, vocab, offset, limit): (u32, u32, u32, u32) = kani::any();
    let actual = native_step_end(token, vocab, offset, limit);
    assert_eq!(actual.is_some(), token < vocab && offset < limit);
    if let Some(end) = actual {
        assert_eq!(u64::from(end), u64::from(offset) + 1);
        assert!(end <= limit);
    }
    kani::cover!(actual.is_some() && offset + 1 == limit);
    kani::cover!(token >= vocab && actual.is_none());
    kani::cover!(offset >= limit && actual.is_none());
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) const NATIVE_BLOCK_MAX: usize = 4;

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn native_block_end(tokens: &[u32], vocab: u32, offset: u32, limit: u32) -> Option<u32> {
    if !(1..=NATIVE_BLOCK_MAX).contains(&tokens.len()) || tokens.iter().any(|&t| t >= vocab) {
        return None;
    }
    let end = offset.checked_add(tokens.len() as u32)?;
    (end <= limit).then_some(end)
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn native_retained_prefix(retained: usize, verified: usize) -> Option<Option<usize>> {
    if !(1..=NATIVE_BLOCK_MAX).contains(&verified) || retained > verified {
        return None;
    }
    Some(retained.checked_sub(1))
}

#[test]
fn native_block_guards_validate_boundaries_before_publication() {
    assert_eq!(
        native_block_end(&[0; 4], 1, u32::MAX - 4, u32::MAX),
        Some(u32::MAX)
    );
    assert_eq!(native_block_end(&[0; 4], 1, u32::MAX - 3, u32::MAX), None);
    assert_eq!(native_block_end(&[0; 5], 1, 0, 8), None);
    assert_eq!(native_block_end(&[], 1, 0, 8), None);
    assert_eq!(native_block_end(&[0, 1, 0], 1, 0, 8), None);
    assert_eq!(native_retained_prefix(0, 4), Some(None));
    assert_eq!(native_retained_prefix(4, 4), Some(Some(3)));
    assert_eq!(native_retained_prefix(5, 4), None);
    assert_eq!(native_retained_prefix(0, 0), None);
    assert_eq!(native_retained_prefix(1, 5), None);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(7)]
fn native_qwen_block_preflights_every_row() {
    let tokens: [u32; 5] = kani::any();
    let n = usize::from(kani::any::<u8>() % 6);
    let (vocab, offset, limit): (u32, u32, u32) = kani::any();
    let end = u64::from(offset) + n as u64;
    let valid = n > 0
        && n <= 4
        && end <= u64::from(limit)
        && (n < 1 || tokens[0] < vocab)
        && (n < 2 || tokens[1] < vocab)
        && (n < 3 || tokens[2] < vocab)
        && (n < 4 || tokens[3] < vocab);
    let actual = native_block_end(&tokens[..n], vocab, offset, limit);
    assert_eq!(actual.map(u64::from), valid.then_some(end));
    kani::cover!(actual == Some(u32::MAX) && n == 4);
    kani::cover!(n == 4 && tokens[3] >= vocab && actual.is_none());
    kani::cover!(n == 5 && actual.is_none());
}

#[cfg(kani)]
#[kani::proof]
fn native_qwen_verification_retains_only_a_verified_prefix() {
    let (retained, verified): (usize, usize) = kani::any();
    let actual = native_retained_prefix(retained, verified);
    assert_eq!(
        actual.is_some(),
        verified > 0 && verified <= 4 && retained <= verified
    );
    if let Some(index) = actual {
        if let Some(index) = index {
            assert!(index < verified);
            assert_eq!(index + 1, retained);
        } else {
            assert_eq!(retained, 0);
        }
    }
    kani::cover!(actual == Some(None));
    kani::cover!(actual == Some(Some(3)));
    kani::cover!(verified == 4 && retained == 5 && actual.is_none());
}

/// Bind one packed affine4 expert. The shader indexes a single slice in u32.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn native_affine4_offsets(
    input: u32,
    output: u32,
    expert: u32,
    experts: u32,
) -> Option<[u64; 2]> {
    if input == 0 || !input.is_multiple_of(64) || output == 0 || expert >= experts {
        return None;
    }
    let packed_words = (input / 8).checked_mul(output)?;
    let expert = u64::from(expert);
    let words = u64::from(packed_words);
    Some([
        words.checked_mul(expert)?.checked_mul(4)?,
        (words / 8).checked_mul(expert)?.checked_mul(2)?,
    ])
}

#[test]
fn native_affine4_validates_the_selected_expert_extent() {
    assert_eq!(native_affine4_offsets(128, 5, 2, 3), Some([640, 40]));
    assert_eq!(native_affine4_offsets(128, 5, 3, 3), None);
    assert_eq!(native_affine4_offsets(127, 5, 0, 1), None);
    assert_eq!(native_affine4_offsets(64, u32::MAX, 0, 1), None);
    assert_eq!(
        native_affine4_offsets(64, u32::MAX / 8, u32::MAX - 1, u32::MAX),
        None
    );
}

#[cfg(kani)]
#[kani::proof]
fn native_mtp_affine_binds_a_complete_expert_slice() {
    let (input, output, experts): (u32, u32, u32) = kani::any();
    let expert: u32 = kani::any();
    // Sizes of the supported head's individual projection layouts.
    kani::assume(matches!(
        (input, output, experts),
        (4096, 2048, 1) | (2048, 8192 | 512 | 256 | 1, 1) | (2048, 512, 256) | (512, 2048, 1 | 256)
    ));
    let actual = native_affine4_offsets(input, output, expert, experts);
    assert_eq!(actual.is_some(), expert < experts);
    if let Some([packed, params]) = actual {
        let elements = u64::from(input) * u64::from(output);
        assert_eq!(packed, u64::from(expert) * (elements / 2));
        assert_eq!(params, u64::from(expert) * (elements / 32));
        assert!(packed + elements / 2 <= u64::from(experts) * (elements / 2));
        assert!(params + elements / 32 <= u64::from(experts) * (elements / 32));
    }
    kani::cover!(expert == 255 && actual.is_some());
    kani::cover!(expert == 256 && actual.is_none());
}

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn native_chain_next(layer: usize, count: usize, enabled: bool) -> Option<usize> {
    if !enabled {
        return None;
    }
    let next = layer.checked_add(1)?;
    (next < count).then_some(next)
}

#[test]
fn native_layer_chain_never_crosses_the_last_layer_or_trace_boundary() {
    assert_eq!(native_chain_next(0, 40, true), Some(1));
    assert_eq!(native_chain_next(38, 40, true), Some(39));
    assert_eq!(native_chain_next(39, 40, true), None);
    assert_eq!(native_chain_next(usize::MAX, 40, true), None);
    assert_eq!(native_chain_next(0, 40, false), None);
}

#[cfg(kani)]
#[kani::proof]
fn native_qwen_layer_chain_only_prepares_the_adjacent_layer() {
    let (layer, count, enabled): (usize, usize, bool) = kani::any();
    let actual = native_chain_next(layer, count, enabled);
    assert_eq!(actual.is_some(), enabled && layer < count.saturating_sub(1));
    if let Some(next) = actual {
        assert_eq!(next - 1, layer);
        assert!(next < count);
    }
    kani::cover!(actual == Some(39) && count == 40);
    kani::cover!(actual.is_none() && layer == 39 && count == 40 && enabled);
    kani::cover!(actual.is_none() && layer == usize::MAX);
}

/// Plan a Qwen append against the backing's reserved extent, including abandoned writes.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn native_qwen_kv_plan(
    offset: u32,
    reserved: u32,
    capacity: u32,
    limit: u32,
) -> Option<(u32, u64, bool)> {
    use crate::flash_support::{qsa_append, qsa_slab_capacity};
    if offset > reserved || reserved > capacity {
        return None;
    }
    let allocated = if capacity == 0 {
        limit.min(64)
    } else {
        capacity
    };
    let size = qsa_slab_capacity(offset, allocated, limit)?;
    let (at, fork) = qsa_append(offset, reserved, size)?;
    Some((size, at, fork || capacity != 0 && size != capacity))
}

#[test]
fn native_qwen_kv_plan_covers_initial_extent_grow_and_restore() {
    assert_eq!(native_qwen_kv_plan(0, 0, 0, 137), Some((64, 0, false)));
    assert_eq!(
        native_qwen_kv_plan(63, 63, 64, 137),
        Some((64, 64512, false))
    );
    assert_eq!(
        native_qwen_kv_plan(64, 64, 64, 137),
        Some((128, 65536, true))
    );
    assert_eq!(
        native_qwen_kv_plan(128, 128, 128, 137),
        Some((137, 131072, true))
    );
    assert_eq!(native_qwen_kv_plan(1, 3, 64, 137), Some((64, 1024, true)));
    assert_eq!(native_qwen_kv_plan(0, 1, 0, 137), None);
    assert_eq!(native_qwen_kv_plan(137, 137, 137, 137), None);
}

#[cfg(kani)]
#[kani::proof]
fn native_qwen_kv_plan_never_overwrites_a_reserved_prefix() {
    let (offset, reserved, capacity, limit): (u32, u32, u32, u32) = kani::any();
    let result = native_qwen_kv_plan(offset, reserved, capacity, limit);
    let valid = offset <= reserved
        && reserved <= capacity
        && capacity <= limit
        && offset < limit
        && limit <= crate::flash_support::FLASH_MAX_CONTEXT;
    assert_eq!(result.is_some(), valid);
    if let Some((size, at, fork)) = result {
        assert!(size <= limit && size >= capacity && offset < size);
        assert_eq!(at, u64::from(offset) * 1024);
        assert!(at + 1024 <= u64::from(size) * 1024);
        assert_eq!(
            fork,
            offset < reserved || (capacity > 0 && offset == capacity)
        );
        kani::cover!(capacity == 0 && size == 64);
        kani::cover!(fork && offset < reserved && offset != 0);
        kani::cover!(fork && offset == capacity && size == limit);
        kani::cover!(!fork && offset + 1 == limit);
    }
    kani::cover!(result.is_none() && reserved > capacity);
}

/// Address of a typed buffer range bound through a Metal argument buffer.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn native_qwen_qmv_binding(
    address: u64,
    offset: u64,
    bytes: u64,
    length: u64,
    alignment: u64,
) -> Option<u64> {
    if address == 0
        || bytes == 0
        || !alignment.is_power_of_two()
        || offset.checked_add(bytes)? > length
    {
        return None;
    }
    let bound = address.checked_add(offset)?;
    bound.checked_add(bytes)?;
    bound.is_multiple_of(alignment).then_some(bound)
}

#[test]
fn native_qwen_qmv_binding_rejects_wrapped_misaligned_and_short_ranges() {
    assert_eq!(native_qwen_qmv_binding(0x1000, 4, 12, 16, 4), Some(0x1004));
    assert_eq!(native_qwen_qmv_binding(0x1000, 2, 12, 16, 4), None);
    assert_eq!(native_qwen_qmv_binding(0x1000, 4, 13, 16, 4), None);
    assert_eq!(native_qwen_qmv_binding(u64::MAX - 3, 0, 4, 4, 4), None);
    assert_eq!(native_qwen_qmv_binding(0, 4, 12, 16, 4), None);
}

#[cfg(kani)]
#[kani::proof]
fn native_qwen_qmv_binding_stays_in_the_allocated_range() {
    let (address, offset, bytes, length): (u64, u64, u64, u64) = kani::any();
    let alignment = if kani::any() { 2 } else { 4 };
    let actual = native_qwen_qmv_binding(address, offset, bytes, length, alignment);
    let bound = u128::from(address) + u128::from(offset);
    let end = bound + u128::from(bytes);
    let valid = address != 0
        && bytes != 0
        && u128::from(offset) + u128::from(bytes) <= u128::from(length)
        && end <= u128::from(u64::MAX)
        && bound % u128::from(alignment) == 0;
    assert_eq!(actual.map(u128::from), valid.then_some(bound));
    kani::cover!(actual.is_some() && offset != 0 && offset + bytes == length);
    kani::cover!(actual.is_none() && address != 0 && end > u128::from(u64::MAX));
    kani::cover!(actual.is_none() && address != 0 && bound % u128::from(alignment) != 0);
}

/// [jobs, input width, output width] for the single-split Qwen MoE stages.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) const QWEN_STAGE_SHAPES: [[u32; 3]; 2] = [[16, 2048, 512], [8, 512, 2048]];

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn native_qwen_stage_offsets(job: u32, down: bool) -> Option<[u64; 2]> {
    let [jobs, input, output] = QWEN_STAGE_SHAPES[usize::from(down)];
    (job < jobs).then(|| {
        [
            u64::from(job) * u64::from(input) * 2,
            u64::from(job) * u64::from(output) * 4,
        ]
    })
}

#[test]
fn native_qwen_stage_offsets_reject_jobs_past_the_batch() {
    assert_eq!(native_qwen_stage_offsets(15, false), Some([61_440, 30_720]));
    assert_eq!(native_qwen_stage_offsets(7, true), Some([7_168, 57_344]));
    assert_eq!(native_qwen_stage_offsets(16, false), None);
    assert_eq!(native_qwen_stage_offsets(8, true), None);
    assert_eq!(native_qwen_stage_offsets(u32::MAX, true), None);
}

#[cfg(kani)]
#[kani::proof]
fn native_qwen_stage_jobs_have_disjoint_in_bounds_slices() {
    let (job, other, down): (u32, u32, bool) = kani::any();
    let (count, input, output): (u32, u64, u64) = if down {
        (8, 1024, 8192)
    } else {
        (16, 4096, 2048)
    };
    let a = native_qwen_stage_offsets(job, down);
    let b = native_qwen_stage_offsets(other, down);
    assert_eq!(a.is_some(), job < count);
    if let Some([a_in, a_out]) = a {
        let row = u64::from(job);
        assert_eq!(a_in, row * input);
        assert_eq!(a_out, row * output);
        assert!(a_in + input <= u64::from(count) * input);
        assert!(a_out + output <= u64::from(count) * output);
        if let Some([b_in, b_out]) = b
            && job < other
        {
            assert!(a_in + input <= b_in);
            assert!(a_out + output <= b_out);
        }
        kani::cover!(down && job == 7);
        kani::cover!(!down && job == 15);
    }
    kani::cover!(a.is_none() && b.is_some());
}

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
