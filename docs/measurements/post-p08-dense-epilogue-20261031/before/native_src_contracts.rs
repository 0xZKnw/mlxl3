//! Checked arithmetic shared by checkpoint and GPU-facing production paths.

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn biased_router_domain(rows: i32, experts: i32, top_k: usize) -> bool {
    rows > 0
        && (1..=64).contains(&experts)
        && top_k > 0
        && top_k <= experts as usize
        && rows.checked_mul(experts).is_some()
}

#[test]
fn biased_router_domain_covers_both_simd_halves() {
    for rows in [
        i32::MIN,
        -1,
        0,
        1,
        2,
        3,
        7,
        8,
        23,
        24,
        25,
        63,
        64,
        65,
        128,
        266,
        i32::MAX,
    ] {
        for experts in -1..=65 {
            for top_k in 0..=65 {
                assert_eq!(
                    biased_router_domain(rows, experts, top_k),
                    rows > 0
                        && experts > 0
                        && experts <= 64
                        && top_k > 0
                        && top_k <= experts as usize
                        && i64::from(rows) * i64::from(experts) <= i64::from(i32::MAX)
                );
            }
        }
    }
    assert!(!biased_router_domain(1, i32::MAX, usize::MAX));
    assert!(!biased_router_domain(i32::MIN, i32::MIN, 1));
    assert!(biased_router_domain(i32::MAX, 1, 1));
    assert!(biased_router_domain(i32::MAX / 64, 64, 64));
    assert!(!biased_router_domain(i32::MAX / 64 + 1, 64, 64));
}

#[cfg(kani)]
#[kani::proof]
fn biased_router_domain_bounds_dispatch() {
    let rows: i32 = kani::any();
    let experts: i32 = kani::any();
    let top_k: usize = kani::any();
    let valid = biased_router_domain(rows, experts, top_k);
    assert_eq!(
        valid,
        rows > 0
            && experts >= 1
            && experts <= 64
            && top_k >= 1
            && top_k <= experts as usize
            && i64::from(rows) * i64::from(experts) <= i64::from(i32::MAX)
    );
    if valid {
        assert!((1..=64).contains(&(top_k as i32)));
        assert!(rows.checked_mul(top_k as i32).is_some());
    }
    kani::cover!(valid && rows == 266 && experts == 64 && top_k == 4);
    kani::cover!(valid && rows == 2 && experts == 33 && top_k == 33);
    kani::cover!(valid && rows == i32::MAX && experts == 1 && top_k == 1);
    kani::cover!(!valid && rows == 0);
    kani::cover!(!valid && rows == i32::MAX && experts == 64);
    kani::cover!(!valid && top_k == usize::MAX);
}

#[test]
fn biased_router_offsets_cover_registered_rows_and_all_expert_widths() {
    for experts in 1..=64 {
        for rows in [
            1,
            2,
            3,
            7,
            8,
            23,
            24,
            25,
            63,
            64,
            65,
            128,
            266,
            i32::MAX / experts,
        ] {
            for top_k in [1, 4.min(experts as usize), experts as usize] {
                assert!(biased_router_domain(rows, experts, top_k));
                for row in [0, rows / 2, rows - 1] {
                    let input = row.checked_mul(experts).unwrap();
                    let output = row.checked_mul(top_k as i32).unwrap();
                    assert!(
                        i64::from(input) + i64::from(experts) - 1
                            < i64::from(rows) * i64::from(experts)
                    );
                    assert!(i64::from(output) + top_k as i64 - 1 < i64::from(rows) * top_k as i64);
                    assert!(input.checked_add(experts - 1).is_some());
                    assert!(output.checked_add(top_k as i32 - 1).is_some());
                }
            }
        }
    }
}

/// Only an explicit opt-in selects the numerically qualified microblocks.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn lfm_moe_prefill_enabled(qualified: bool, flag: Option<&[u8]>) -> bool {
    qualified && matches!(flag, Some([b'1']))
}

#[test]
fn lfm_moe_prefill_selection_preserves_default_and_explicit_fallback() {
    for qualified in [false, true] {
        assert!(!lfm_moe_prefill_enabled(qualified, None));
        for byte in 0..=u8::MAX {
            assert_eq!(
                lfm_moe_prefill_enabled(qualified, Some(&[byte])),
                qualified && byte == b'1'
            );
        }
        for invalid in [vec![], vec![b'1'; 2], vec![b'1'; 1024], vec![0xff; 1024]] {
            assert!(!lfm_moe_prefill_enabled(qualified, Some(&invalid)));
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(6)]
fn lfm_moe_prefill_selection_respects_qualification_and_override() {
    let qualified: bool = kani::any();
    let present: bool = kani::any();
    let bytes: [u8; 4] = kani::any();
    // Finite string domain 0..4, constructed without a caller precondition.
    let length = usize::from(kani::any::<u8>()) % 5;
    let flag = present.then_some(&bytes[..length]);
    let enabled = lfm_moe_prefill_enabled(qualified, flag);
    assert_eq!(
        enabled,
        qualified && present && length == 1 && bytes[0] == b'1'
    );
    kani::cover!(qualified && !present && !enabled);
    kani::cover!(qualified && present && length == 1 && bytes[0] == b'1' && enabled);
    kani::cover!(qualified && present && length == 1 && bytes[0] == b'0' && !enabled);
    kani::cover!(!qualified && !present && !enabled);
    kani::cover!(qualified && present && length == 2 && !enabled);
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn lfm_moe_exact_prefill_geometry(
    hidden: i32,
    expert_hidden: i32,
    experts: i32,
    top_k: usize,
    mapped_k4_mcg: bool,
) -> bool {
    hidden == 2048 && expert_hidden == 1536 && experts == 64 && top_k == 4 && mapped_k4_mcg
}

#[test]
fn lfm_moe_exact_prefill_geometry_requires_the_qualified_projection_layout() {
    assert!(lfm_moe_exact_prefill_geometry(2048, 1536, 64, 4, true));
    for invalid in [i32::MIN, -1, 0, 1, 2047, 2049, i32::MAX] {
        assert!(!lfm_moe_exact_prefill_geometry(invalid, 1536, 64, 4, true));
    }
    for invalid in [i32::MIN, -1, 0, 1535, 1537, i32::MAX] {
        assert!(!lfm_moe_exact_prefill_geometry(2048, invalid, 64, 4, true));
    }
    for invalid in [i32::MIN, -1, 0, 32, 63, 65, i32::MAX] {
        assert!(!lfm_moe_exact_prefill_geometry(
            2048, 1536, invalid, 4, true
        ));
    }
    for invalid in [0, 1, 3, 5, 64, usize::MAX] {
        assert!(!lfm_moe_exact_prefill_geometry(
            2048, 1536, 64, invalid, true
        ));
    }
    assert!(!lfm_moe_exact_prefill_geometry(2048, 1536, 64, 4, false));
}

#[cfg(kani)]
#[kani::proof]
fn lfm_moe_exact_prefill_accepts_only_the_qualified_geometry() {
    let hidden: i32 = kani::any();
    let expert_hidden: i32 = kani::any();
    let experts: i32 = kani::any();
    let top_k: usize = kani::any();
    let mapped: bool = kani::any();
    let eligible = lfm_moe_exact_prefill_geometry(hidden, expert_hidden, experts, top_k, mapped);
    if eligible {
        assert_eq!(hidden, 2048);
        assert_eq!(expert_hidden, 1536);
        assert_eq!(experts, 64);
        assert_eq!(top_k, 4);
        assert!(mapped);
    }
    if hidden == 2048 && expert_hidden == 1536 && experts == 64 && top_k == 4 && mapped {
        assert!(eligible);
    }
    kani::cover!(eligible);
    kani::cover!(!eligible && !mapped);
    kani::cover!(!eligible && top_k == usize::MAX);
}

/// Validate a complete experimental prefill before mutating any recurrent state.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn lfm_prefill_end(tokens: usize, offset: i32, limit: i32) -> Option<i32> {
    if tokens == 0 || offset < 0 || limit <= 0 {
        return None;
    }
    let end = offset.checked_add(i32::try_from(tokens).ok()?)?;
    (end <= limit).then_some(end)
}

#[test]
fn lfm_prefill_end_rejects_empty_invalid_and_overflowing_ranges() {
    for (tokens, offset, limit, expected) in [
        (0, 0, 4096, None),
        (1, -1, 4096, None),
        (1, 0, 0, None),
        (1, 0, 1, Some(1)),
        (8, 4090, 4098, Some(4098)),
        (8, 4090, 4097, None),
        (1, i32::MAX - 1, i32::MAX, Some(i32::MAX)),
        (1, i32::MAX, i32::MAX, None),
        (usize::MAX, 0, i32::MAX, None),
    ] {
        assert_eq!(lfm_prefill_end(tokens, offset, limit), expected);
    }
}

#[cfg(kani)]
#[kani::proof]
fn lfm_prefill_end_preserves_context_and_all_block_positions() {
    let tokens: usize = kani::any();
    let offset: i32 = kani::any();
    let limit: i32 = kani::any();
    let expected = if tokens > 0
        && offset >= 0
        && limit > 0
        && tokens as u128 + offset as u128 <= limit as u128
    {
        Some((tokens as u128 + offset as u128) as i32)
    } else {
        None
    };
    let end = lfm_prefill_end(tokens, offset, limit);
    assert_eq!(end, expected);
    if let Some(end) = end {
        let row: usize = kani::any();
        kani::assume(row < tokens); // A row produced by iteration over these tokens.
        assert!(offset.checked_add(row as i32).is_some());
        assert!(offset + (row as i32) < end);
        kani::cover!(tokens == 266 && offset == 17 && row == 265);
        kani::cover!(end == i32::MAX);
    }
    kani::cover!(tokens == 0 && end.is_none());
    kani::cover!(tokens == usize::MAX && end.is_none());
}

/// Distinguishes every opt-in kernel combination in persisted MTP calibrations.
pub fn smallm_kernel_key(dense_batch: bool, grouped_mb3: bool) -> u8 {
    u8::from(dense_batch) | (u8::from(grouped_mb3) << 1)
}

#[test]
fn smallm_kernel_key_distinguishes_all_options() {
    for (dense, mb3, expected) in [
        (false, false, 0),
        (true, false, 1),
        (false, true, 2),
        (true, true, 3),
    ] {
        assert_eq!(smallm_kernel_key(dense, mb3), expected);
    }
}

#[cfg(kani)]
#[kani::proof]
fn smallm_kernel_key_is_injective() {
    let dense: bool = kani::any();
    let mb3: bool = kani::any();
    let other_dense: bool = kani::any();
    let other_mb3: bool = kani::any();
    let key = smallm_kernel_key(dense, mb3);
    assert!(key <= 3);
    assert_eq!(
        key == smallm_kernel_key(other_dense, other_mb3),
        dense == other_dense && mb3 == other_mb3
    );
    kani::cover!(key == 0);
    kani::cover!(key == 1);
    kani::cover!(key == 2);
    kani::cover!(key == 3);
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn dense_mlp_batch(
    rows: usize,
    hidden: i32,
    grouped_exl3: bool,
    down_exl3: bool,
    m5: bool,
    enabled: bool,
) -> bool {
    // M3 batching regressed alone; only M2/M4 passed the physical cost screen.
    enabled && m5 && hidden == 5120 && matches!(rows, 2 | 4) && grouped_exl3 && down_exl3
}

#[test]
fn dense_mlp_batch_requires_supported_weights_and_rows() {
    for rows in 0..=9 {
        assert_eq!(
            dense_mlp_batch(rows, 5120, true, true, true, true),
            matches!(rows, 2 | 4)
        );
    }
    for (hidden, grouped, down, m5, enabled) in [
        (2048, true, true, true, true),
        (5120, false, true, true, true),
        (5120, true, false, true, true),
        (5120, true, true, false, true),
        (5120, true, true, true, false),
    ] {
        for rows in [2, 4] {
            assert!(!dense_mlp_batch(rows, hidden, grouped, down, m5, enabled));
        }
    }
    assert!(!dense_mlp_batch(
        usize::MAX,
        i32::MAX,
        true,
        true,
        true,
        true
    ));
}

#[cfg(kani)]
#[kani::proof]
fn dense_mlp_batch_stays_within_supported_domain() {
    let rows: usize = kani::any();
    let hidden: i32 = kani::any();
    let grouped: bool = kani::any();
    let down: bool = kani::any();
    let m5: bool = kani::any();
    let enabled: bool = kani::any();
    let selected = dense_mlp_batch(rows, hidden, grouped, down, m5, enabled);
    assert_eq!(
        selected,
        enabled && m5 && hidden == 5120 && (rows == 2 || rows == 4) && grouped && down
    );
    if selected {
        assert!(rows <= i32::MAX as usize);
        assert!(rows * hidden as usize <= i32::MAX as usize);
        let row: usize = kani::any();
        if row < rows {
            assert!((row as i32) + 1 <= rows as i32);
        }
    }
    kani::cover!(selected && rows == 2);
    kani::cover!(!selected && rows == 3 && enabled && m5 && grouped && down && hidden == 5120);
    kani::cover!(selected && rows == 4);
    kani::cover!(!selected && !grouped);
    kani::cover!(!selected && rows == 8);
    kani::cover!(!selected && rows == 2 && !enabled && m5 && grouped && down && hidden == 5120);
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn mtp_pipeline_default(
    hidden: i32,
    layers: usize,
    experts: i32,
    top_k: usize,
    m5: bool,
) -> bool {
    m5 && hidden == 2048 && layers == 40 && experts == 256 && top_k == 8
}

#[test]
fn mtp_pipeline_default_requires_measured_geometry() {
    assert!(mtp_pipeline_default(2048, 40, 256, 8, true));
    for (hidden, layers, experts, top_k, m5) in [
        (2048, 40, 256, 8, false),
        (5120, 64, 0, 0, true),
        (2048, 39, 256, 8, true),
        (2048, 40, 0, 0, true),
        (2048, 40, 128, 8, true),
        (2048, 40, 256, 4, true),
        (i32::MAX, usize::MAX, i32::MAX, usize::MAX, true),
    ] {
        assert!(!mtp_pipeline_default(hidden, layers, experts, top_k, m5));
    }
}

#[cfg(kani)]
#[kani::proof]
fn mtp_pipeline_default_is_limited_to_measured_target() {
    let hidden: i32 = kani::any();
    let layers: usize = kani::any();
    let experts: i32 = kani::any();
    let top_k: usize = kani::any();
    let m5: bool = kani::any();
    let enabled = mtp_pipeline_default(hidden, layers, experts, top_k, m5);
    assert_eq!(
        enabled,
        m5 && hidden == 2048 && layers == 40 && experts == 256 && top_k == 8
    );
    kani::cover!(enabled);
    kani::cover!(!enabled && !m5);
    kani::cover!(!enabled && experts == 0);
    kani::cover!(!enabled && layers == 0);
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn pipeline_layer(index: usize, layers: usize, enabled: bool) -> bool {
    enabled && index < layers.saturating_sub(1) && (index == 0 || index % 4 == 3)
}

#[test]
fn pipeline_layer_stays_inside_model() {
    for layers in 0..=12 {
        let expected = [0, 3, 7, 11]
            .into_iter()
            .filter(|&index| index + 1 < layers)
            .collect::<Vec<_>>();
        let actual = (0..=13)
            .filter(|&index| pipeline_layer(index, layers, true))
            .collect::<Vec<_>>();
        assert_eq!(actual, expected);
        assert!((0..=13).all(|index| !pipeline_layer(index, layers, false)));
    }
    assert!(!pipeline_layer(usize::MAX, usize::MAX, true));
    assert!(!pipeline_layer(usize::MAX - 1, usize::MAX, true));
    assert!(pipeline_layer(usize::MAX - 4, usize::MAX, true));
}

#[cfg(kani)]
#[kani::proof]
fn pipeline_layer_has_no_final_or_outside_submission() {
    let index: usize = kani::any();
    let layers: usize = kani::any();
    let enabled: bool = kani::any();
    let submit = pipeline_layer(index, layers, enabled);
    if submit {
        assert!(enabled && layers > 1 && index < layers - 1);
        assert!(index == 0 || (index + 1) % 4 == 0);
    }
    assert_eq!(
        submit,
        enabled && layers > 1 && index < layers - 1 && (index == 0 || index % 4 == 3)
    );
    kani::cover!(submit && index == 0);
    kani::cover!(submit && index > 0);
    kani::cover!(!submit && layers > 0 && index == layers - 1);
    kani::cover!(!submit && !enabled);
}

#[cfg(any(feature = "mlx", kani))]
pub(crate) fn mtp_add_norm_launch(rows: i32, width: i32) -> Option<(i32, i32)> {
    if !(1..=4).contains(&rows) || !matches!(width, 2048 | 5120) {
        return None;
    }
    let threads = (width / 4).min(1024);
    Some((threads, rows * threads))
}

#[cfg(kani)]
#[kani::proof]
fn mtp_add_norm_launch_bounds() {
    let rows: i32 = kani::any();
    let width: i32 = kani::any();
    let launch = mtp_add_norm_launch(rows, width);
    if let Some((threads, grid)) = launch {
        assert!((1..=4).contains(&rows) && matches!(width, 2048 | 5120));
        assert!(threads > 0 && threads <= 1024 && threads % 32 == 0);
        assert!(grid == rows * threads && grid <= 4096);
        assert!(rows * width <= 20_480);
    }
    kani::cover!(rows == 4 && width == 5120 && launch == Some((1024, 4096)));
    kani::cover!(rows == 1 && width == 2048 && launch == Some((512, 512)));
    kani::cover!(launch.is_none());
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn use_tensor_ops(rows: i32, capable: bool) -> bool {
    capable && rows >= 24
}

/// Shapes measured on the M5 dense checkpoint, with unchanged per-output sums.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn dense_decode_nt4(input: i32, output: i32, k: usize, mul1: bool, m5: bool) -> bool {
    m5 && mul1
        && input == 5120
        && ((k == 2 && matches!(output, 16384 | 34816)) || (k == 3 && output == 248320))
}

#[cfg(kani)]
#[kani::proof]
fn dense_decode_tile_never_crosses_unmeasured_shapes_or_output_boundaries() {
    let input: i32 = kani::any();
    let output: i32 = kani::any();
    let k: usize = kani::any();
    let mul1: bool = kani::any();
    let m5: bool = kani::any();
    let enabled = dense_decode_nt4(input, output, k, mul1, m5);
    if enabled {
        assert!(input == 5120 && m5 && mul1);
        assert!(output > 0 && output % 128 == 0);
        assert!((k == 2 && (output == 16384 || output == 34816)) || (k == 3 && output == 248320));
    }
    kani::cover!(enabled && k == 2);
    kani::cover!(enabled && k == 3);
    kani::cover!(!enabled && input == 5120 && k == 2);
}

#[test]
fn dense_decode_nt4_keeps_other_shapes_on_existing_tiles() {
    for (input, output, k, mul1, m5, expected) in [
        (5120, 16384, 2, true, true, true),
        (5120, 34816, 2, true, true, true),
        (5120, 248320, 3, true, true, true),
        (5120, 248320, 2, true, true, false),
        (2048, 16384, 2, true, true, false),
        (5120, 34816, 4, true, true, false),
        (5120, 34816, 2, false, true, false),
        (5120, 34816, 2, true, false, false),
        (5120, 34815, 2, true, true, false),
        (17408, 5120, 2, true, true, false),
    ] {
        assert_eq!(dense_decode_nt4(input, output, k, mul1, m5), expected);
        if expected {
            assert_eq!((output / 16) % 4, 0);
        }
    }
}

/// Conservative M5 dense tile: no padding or changed depth/reduction order.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn dense_qmm_block_rows(rows: i32, input: i32, output: i32, mul1: bool) -> i32 {
    if mul1 && input >= 4096 && (1..65536).contains(&output) && rows >= 128 && rows % 64 == 0 {
        64
    } else {
        32
    }
}

#[test]
fn dense_qmm_large_tile_preserves_ragged_and_small_fallbacks() {
    for (rows, input, output, mul1, expected) in [
        (128, 4096, 5120, true, 64),
        (256, 5120, 17408, true, 64),
        (192, 17408, 5120, true, 64),
        (128, 4096, 65535, true, 64),
        (128, 4096, 65536, true, 32),
        (128, 4096, 248320, true, 32),
        (127, 4096, 5120, true, 32),
        (129, 4096, 5120, true, 32),
        (64, 4096, 5120, true, 32),
        (128, 2048, 5120, true, 32),
        (128, 4096, 5120, false, 32),
        (i32::MIN, 4096, 5120, true, 32),
    ] {
        assert_eq!(dense_qmm_block_rows(rows, input, output, mul1), expected);
    }
}

#[cfg(kani)]
#[kani::proof]
fn dense_qmm_large_tile_never_pads_or_changes_ineligible_shapes() {
    let rows: i32 = kani::any();
    let input: i32 = kani::any();
    let output: i32 = kani::any();
    let mul1: bool = kani::any();
    let block = dense_qmm_block_rows(rows, input, output, mul1);
    assert!(block == 32 || block == 64);
    if block == 64 {
        assert!(rows >= 128 && rows % block == 0);
        assert!(input >= 4096 && output > 0 && output < 65536 && mul1);
    }
    kani::cover!(block == 64);
    kani::cover!(rows == 129 && block == 32);
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

/// Reference EXL3 reduction with two independent rows sharing each decoded tile.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn flash_reference_batch_layout(rows: i32) -> Option<(i32, i32)> {
    if !(2..=8).contains(&rows) {
        return None;
    }
    Some((2, (rows + 1) / 2))
}

#[test]
fn flash_reference_batch_covers_odd_rows_without_extra_outputs() {
    for rows in 2..=8 {
        let (mb, groups) = flash_reference_batch_layout(rows).unwrap();
        assert_eq!(mb, 2);
        assert!(groups * mb >= rows && groups * mb < rows + mb);
    }
    for rows in [i32::MIN, -1, 0, 1, 9, 24, i32::MAX] {
        assert_eq!(flash_reference_batch_layout(rows), None);
    }
}

#[cfg(kani)]
#[kani::proof]
fn flash_reference_batch_has_exact_bounded_coverage() {
    let rows: i32 = kani::any();
    match flash_reference_batch_layout(rows) {
        Some((mb, groups)) => {
            assert!((2..=8).contains(&rows));
            assert_eq!(mb, 2);
            assert!(groups * mb >= rows && groups * mb < rows + mb);
            assert!((1..=4).contains(&groups));
            kani::cover!(rows == 3);
            kani::cover!(rows == 8);
        }
        None => assert!(!(2..=8).contains(&rows)),
    }
}

/// Experimental M3 geometry, restricted to the checkpoint shapes measured on M5.
#[cfg(any(feature = "mlx", test, kani))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn qmv_grouped_batch_layout(
    rows: i32,
    input: i32,
    output: i32,
    k: usize,
    mul1: bool,
    m5: bool,
    enabled: bool,
) -> Option<(i32, i32)> {
    if enabled
        && m5
        && mul1
        && rows == 3
        && input == 5120
        && ((output == 16384 && k == 2) || (output == 34816 && (1..=3).contains(&k)))
    {
        Some((3, 1))
    } else {
        qmv_batch_layout(rows, true, output / 16 >= 1024)
    }
}

#[test]
fn grouped_mb3_requires_opt_in_and_measured_shape() {
    for (output, k) in [(16384, 2), (34816, 1), (34816, 2), (34816, 3)] {
        assert_eq!(
            qmv_grouped_batch_layout(3, 5120, output, k, true, true, true),
            Some((3, 1))
        );
        for (input, mul1, m5, enabled) in [
            (5120, true, true, false),
            (5120, true, false, true),
            (5120, false, true, true),
            (4096, true, true, true),
        ] {
            assert_eq!(
                qmv_grouped_batch_layout(3, input, output, k, mul1, m5, enabled),
                Some((1, 3))
            );
        }
    }
    for rows in [i32::MIN, 1, 2, 4, 8, 23, 24, i32::MAX] {
        assert_eq!(
            qmv_grouped_batch_layout(rows, 5120, 34816, 2, true, true, true),
            qmv_batch_layout(rows, true, true)
        );
    }
    assert_eq!(
        qmv_grouped_batch_layout(3, 5120, 34816, 4, true, true, true),
        Some((1, 3))
    );
    assert_eq!(
        qmv_grouped_batch_layout(3, 5120, 10240, 2, true, true, true),
        Some((1, 3))
    );
}

#[cfg(kani)]
#[kani::proof]
fn grouped_mb3_preserves_rows_and_all_fallbacks() {
    let rows: i32 = kani::any();
    let input: i32 = kani::any();
    let output: i32 = kani::any();
    let k: usize = kani::any();
    let mul1: bool = kani::any();
    let m5: bool = kani::any();
    let enabled: bool = kani::any();
    let actual = qmv_grouped_batch_layout(rows, input, output, k, mul1, m5, enabled);
    let selected = actual == Some((3, 1));
    assert_eq!(
        selected,
        enabled
            && m5
            && mul1
            && rows == 3
            && input == 5120
            && ((output == 16384 && k == 2) || (output == 34816 && (1..=3).contains(&k)))
    );
    if selected {
        assert_eq!(rows, 3);
        assert_eq!(actual.unwrap().0 * actual.unwrap().1, rows);
    } else {
        assert_eq!(actual, qmv_batch_layout(rows, true, output / 16 >= 1024));
    }
    kani::cover!(selected && output == 16384);
    kani::cover!(selected && output == 34816 && k == 1);
    kani::cover!(selected && output == 34816 && k == 2);
    kani::cover!(selected && output == 34816 && k == 3);
    kani::cover!(!enabled && rows == 3 && actual == Some((1, 3)));
    kani::cover!(enabled && rows == 4 && actual == Some((2, 2)));
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

pub(crate) fn absolute_data_range(payload: u64, [start, end]: [u64; 2]) -> Option<[u64; 2]> {
    if start > end {
        return None;
    }
    Some([payload.checked_add(start)?, payload.checked_add(end)?])
}

#[cfg(kani)]
#[kani::proof]
fn tensor_source_interval_is_ordered_and_cannot_wrap() {
    let payload: u64 = kani::any();
    let start: u64 = kani::any();
    let end: u64 = kani::any();
    let result = absolute_data_range(payload, [start, end]);
    assert_eq!(
        result.is_some(),
        start <= end && u128::from(payload) + u128::from(end) <= u128::from(u64::MAX)
    );
    if let Some([first, last]) = result {
        assert!(first <= last && first >= payload);
        assert_eq!(u128::from(first), u128::from(payload) + u128::from(start));
        assert_eq!(u128::from(last), u128::from(payload) + u128::from(end));
        kani::cover!(first == u64::MAX && last == u64::MAX);
        kani::cover!(first == last && first == 0);
    }
    kani::cover!(start > end && result.is_none());
    kani::cover!(payload == u64::MAX && end == 1 && result.is_none());
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
/// Keep small allocator working sets warm; reclaim large unused buffers at idle.
#[cfg(any(feature = "mlx", kani, test))]
pub(crate) fn release_idle_cache(cache_bytes: u64, memory_saver: bool) -> bool {
    cache_bytes > if memory_saver { 0 } else { 512 * 1024 * 1024 }
}

#[cfg(kani)]
#[kani::proof]
fn idle_cache_release_preserves_small_working_sets() {
    let bytes: u64 = kani::any();
    let release = release_idle_cache(bytes, false);
    if release {
        assert!(bytes > 536_870_912);
    } else {
        assert!(bytes <= 536_870_912);
    }
    kani::cover!(release && bytes == u64::MAX);
    kani::cover!(!release && bytes == 536_870_912);
    kani::cover!(release && bytes == 536_870_913);
    kani::cover!(!release && bytes == 0);
}

#[cfg(test)]
#[test]
fn idle_cache_policy_boundaries() {
    for (bytes, normal, saver) in [
        (0, false, false),
        (1, false, true),
        (536_870_912, false, true),
        (536_870_913, true, true),
        (u64::MAX, true, true),
    ] {
        assert_eq!(release_idle_cache(bytes, false), normal);
        assert_eq!(release_idle_cache(bytes, true), saver);
    }
}

#[cfg(kani)]
#[kani::proof]
fn memory_saver_releases_every_unused_buffer() {
    let bytes: u64 = kani::any();
    let saver: bool = kani::any();
    let release = release_idle_cache(bytes, saver);
    assert_eq!(
        release,
        if saver {
            bytes != 0
        } else {
            bytes > 536_870_912
        }
    );
    kani::cover!(saver && bytes == 0 && !release);
    kani::cover!(saver && bytes == 1 && release);
    kani::cover!(!saver && bytes == 536_870_912 && !release);
    kani::cover!(saver && bytes == u64::MAX && release);
}

/// Logical F16 states retained by batch-one LFM2, excluding backing storage,
/// allocator caches and temporary activations. Signed totals match Swift Int.
#[cfg(any(feature = "mlx", test, kani))]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct LfmContextMemory {
    fixed_bytes: i64,
    bytes_per_token: i64,
}

#[cfg(any(feature = "mlx", test, kani))]
impl LfmContextMemory {
    pub(crate) fn attention(self, kv_heads: i32, head_dim: i32) -> Option<Self> {
        if kv_heads <= 0 || head_dim <= 0 {
            return None;
        }
        // Two F16 tensors, each [1, kv_heads, tokens, head_dim].
        let bytes = i64::from(kv_heads)
            .checked_mul(i64::from(head_dim))?
            .checked_mul(4)?;
        Some(Self {
            bytes_per_token: self.bytes_per_token.checked_add(bytes)?,
            ..self
        })
    }

    pub(crate) fn convolution(self, length: i32, dim: i32) -> Option<Self> {
        if length <= 0 || dim <= 0 {
            return None;
        }
        // A F16 history view [1, length - 1, dim].
        let bytes = i64::from(length - 1)
            .checked_mul(i64::from(dim))?
            .checked_mul(2)?;
        Some(Self {
            fixed_bytes: self.fixed_bytes.checked_add(bytes)?,
            ..self
        })
    }

    #[cfg(any(feature = "mlx", test))]
    pub(crate) fn into_json(self) -> serde_json::Value {
        serde_json::json!({
            "fixed_bytes": self.fixed_bytes,
            "draft_bytes_per_token": null,
            "layers": [{"bytes_per_token": self.bytes_per_token, "step": 1, "max_tokens": null}]
        })
    }
}

#[test]
fn lfm_context_memory_matches_independent_tensor_shapes() {
    // LFM2.5-24B-A2B: ten attention and thirty convolution layers.
    let mut profile = LfmContextMemory::default();
    for layer in 0..40 {
        profile = if layer % 4 == 3 {
            profile.attention(8, 64).unwrap()
        } else {
            profile.convolution(3, 2048).unwrap()
        };
    }
    assert_eq!(
        (profile.fixed_bytes, profile.bytes_per_token),
        (245_760, 20_480)
    );
    for tokens in [1, 2, 23, 24, 255, 256, 257, 32_768] {
        // Independent shapes include batch, both K/V tensors and F16 item size.
        let kv_shape = [10_i64, 2, 1, 8, tokens, 64, 2];
        let conv_shape = [30_i64, 1, 2, 2048, 2];
        let expected = kv_shape.iter().product::<i64>() + conv_shape.iter().product::<i64>();
        assert_eq!(
            profile.fixed_bytes + profile.bytes_per_token * tokens,
            expected
        );
    }
    let json = profile.into_json();
    assert_eq!(
        json,
        serde_json::json!({
            "fixed_bytes": 245760, "draft_bytes_per_token": null,
            "layers": [{"bytes_per_token": 20480, "step": 1, "max_tokens": null}]
        })
    );
    let wire =
        serde_json::to_vec(&serde_json::json!({"type":"ready", "context_memory":json})).unwrap();
    let decoded: serde_json::Value = serde_json::from_slice(&wire).unwrap();
    assert_eq!(
        decoded["context_memory"]["layers"][0]["bytes_per_token"],
        20480
    );
}

#[test]
fn lfm_context_memory_supports_empty_history_and_mixed_layer_geometry() {
    let empty = LfmContextMemory::default();
    assert_eq!(empty.convolution(1, 2048), Some(empty));
    let profile = empty
        .attention(1, 2)
        .unwrap()
        .attention(2, 4)
        .unwrap()
        .convolution(4, 16)
        .unwrap();
    assert_eq!((profile.fixed_bytes, profile.bytes_per_token), (96, 40));
    assert_eq!(empty.into_json()["fixed_bytes"], 0);
}

#[test]
fn lfm_context_memory_rejects_invalid_dimensions_without_changing_state() {
    let valid = LfmContextMemory::default()
        .attention(8, 64)
        .unwrap()
        .convolution(3, 2048)
        .unwrap();
    for invalid in [i32::MIN, -1, 0] {
        assert_eq!(valid.attention(invalid, 64), None);
        assert_eq!(valid.attention(8, invalid), None);
        assert_eq!(valid.convolution(invalid, 2048), None);
        assert_eq!(valid.convolution(3, invalid), None);
    }
    assert_eq!((valid.fixed_bytes, valid.bytes_per_token), (8192, 2048));
}

#[test]
fn lfm_context_memory_rejects_product_and_sum_overflow() {
    let empty = LfmContextMemory::default();
    assert_eq!(empty.attention(i32::MAX, i32::MAX), None);
    // The largest single convolution fits in i64; a second one must not wrap.
    let large = empty.convolution(i32::MAX, i32::MAX).unwrap();
    assert!(large.fixed_bytes > i64::MAX / 2);
    assert_eq!(large.convolution(i32::MAX, i32::MAX), None);
    let near_limit = LfmContextMemory {
        fixed_bytes: i64::MAX - 2,
        bytes_per_token: i64::MAX - 4,
    };
    assert_eq!(near_limit.convolution(2, 1).unwrap().fixed_bytes, i64::MAX);
    assert_eq!(near_limit.convolution(2, 2), None);
    assert_eq!(
        near_limit.attention(1, 1).unwrap().bytes_per_token,
        i64::MAX
    );
    assert_eq!(near_limit.attention(1, 2), None);
}

#[test]
fn lfm_context_memory_sampled_dimensions_match_wide_integer_oracle() {
    for first in [i32::MIN, -1, 0, 1, 2, 8, 64, 2048, i32::MAX] {
        for second in [i32::MIN, -1, 0, 1, 2, 8, 64, 2048, i32::MAX] {
            let empty = LfmContextMemory::default();
            let expected_attention = (first > 0 && second > 0)
                .then(|| i128::from(first) * i128::from(second) * 4)
                .and_then(|n| i64::try_from(n).ok());
            let expected_conv = (first > 0 && second > 0)
                .then(|| (i128::from(first) - 1) * i128::from(second) * 2)
                .and_then(|n| i64::try_from(n).ok());
            assert_eq!(
                empty.attention(first, second).map(|p| p.bytes_per_token),
                expected_attention
            );
            assert_eq!(
                empty.convolution(first, second).map(|p| p.fixed_bytes),
                expected_conv
            );
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn lfm_context_attention_matches_f16_shapes_and_rejects_overflow() {
    let total: i64 = kani::any();
    let heads: i32 = kani::any();
    kani::assume(total >= 0); // Every reachable accumulator starts at zero and only adds.
    let profile = LfmContextMemory {
        fixed_bytes: 0,
        bytes_per_token: total,
    };
    // Full i32 head count, actual LFM2.5 head dimension; no loops or stubs.
    let expected = (heads > 0).then(|| i128::from(total) + i128::from(heads) * 256);
    let expected = expected.and_then(|n| i64::try_from(n).ok());
    let actual = profile.attention(heads, 64);
    assert_eq!(actual.map(|p| p.bytes_per_token), expected);
    if let Some(result) = actual {
        assert_eq!(result.fixed_bytes, 0);
    }
    kani::cover!(heads == 8 && total == 0 && actual.is_some());
    kani::cover!(heads == 0 && actual.is_none());
    kani::cover!(heads == 1 && total == i64::MAX && actual.is_none());
}

#[cfg(kani)]
#[kani::proof]
fn lfm_context_conv_matches_f16_history_and_rejects_overflow() {
    let total: i64 = kani::any();
    let length: i32 = kani::any();
    kani::assume(total >= 0);
    let profile = LfmContextMemory {
        fixed_bytes: total,
        bytes_per_token: 0,
    };
    // Full i32 history length, actual LFM2.5 hidden dimension.
    let expected = (length > 0).then(|| i128::from(total) + (i128::from(length) - 1) * 4096);
    let expected = expected.and_then(|n| i64::try_from(n).ok());
    let actual = profile.convolution(length, 2048);
    assert_eq!(actual.map(|p| p.fixed_bytes), expected);
    if let Some(result) = actual {
        assert_eq!(result.bytes_per_token, 0);
    }
    kani::cover!(length == 3 && total == 0 && actual.is_some());
    kani::cover!(length == 1 && total == i64::MAX && actual.is_some());
    kani::cover!(length == 0 && actual.is_none());
    kani::cover!(length == 2 && total == i64::MAX && actual.is_none());
}

#[cfg(kani)]
#[kani::proof]
fn lfm_context_memory_small_dimensions_match_wide_oracle() {
    let total: i64 = kani::any();
    let first: i8 = kani::any();
    let second: i8 = kani::any();
    kani::assume(total >= 0);
    let profile = LfmContextMemory {
        fixed_bytes: total,
        bytes_per_token: total,
    };
    let positive = first > 0 && second > 0;
    let attention = positive
        .then(|| i128::from(total) + i128::from(first) * i128::from(second) * 4)
        .and_then(|n| i64::try_from(n).ok());
    let conv = positive
        .then(|| i128::from(total) + (i128::from(first) - 1) * i128::from(second) * 2)
        .and_then(|n| i64::try_from(n).ok());
    let actual_attention = profile.attention(i32::from(first), i32::from(second));
    let actual_conv = profile.convolution(i32::from(first), i32::from(second));
    assert_eq!(actual_attention.map(|p| p.bytes_per_token), attention);
    assert_eq!(actual_conv.map(|p| p.fixed_bytes), conv);
    if let Some(result) = actual_attention {
        assert_eq!(result.fixed_bytes, total);
    }
    if let Some(result) = actual_conv {
        assert_eq!(result.bytes_per_token, total);
    }
    kani::cover!(first == 1 && second == 1 && total == 0 && actual_attention.is_some());
    kani::cover!(first == 1 && second == 127 && total == i64::MAX && actual_conv.is_some());
    kani::cover!(first == 127 && second == 127 && total == i64::MAX && actual_conv.is_none());
    kani::cover!(first == i8::MIN && actual_attention.is_none() && actual_conv.is_none());
}

/// The observed terminal event, independent of token-count heuristics in clients.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GenerationStopReason {
    Eos,
    MaxTokens,
    ContextFull,
}

impl GenerationStopReason {
    pub fn from_round(eos_observed: bool, context_full: bool) -> Self {
        if eos_observed {
            Self::Eos
        } else if context_full {
            Self::ContextFull
        } else {
            Self::MaxTokens
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "eos" => Some(Self::Eos),
            "max_tokens" => Some(Self::MaxTokens),
            "context_full" => Some(Self::ContextFull),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Eos => "eos",
            Self::MaxTokens => "max_tokens",
            Self::ContextFull => "context_full",
        }
    }

    pub fn openai_finish_reason(self) -> &'static str {
        match self {
            Self::Eos => "stop",
            Self::MaxTokens | Self::ContextFull => "length",
        }
    }
}

#[test]
fn generation_stop_reason_preserves_cause_and_openai_mapping() {
    use GenerationStopReason::{ContextFull, Eos, MaxTokens};
    for (eos, full, expected, finish) in [
        (true, false, Eos, "stop"),
        (true, true, Eos, "stop"),
        (false, false, MaxTokens, "length"),
        (false, true, ContextFull, "length"),
    ] {
        let reason = GenerationStopReason::from_round(eos, full);
        assert_eq!(reason, expected);
        assert_eq!(reason.openai_finish_reason(), finish);
        assert_eq!(GenerationStopReason::parse(reason.as_str()), Some(reason));
    }
    for invalid in [
        "",
        "stop",
        "length",
        "EOS",
        "max_tokens ",
        "context_full\n",
        "cancelled",
        "error",
    ] {
        assert_eq!(GenerationStopReason::parse(invalid), None);
    }
}

#[cfg(kani)]
#[kani::proof]
fn generation_stop_reason_matches_observed_terminal_event() {
    let eos: bool = kani::any();
    let full: bool = kani::any();
    let reason = GenerationStopReason::from_round(eos, full);
    assert_eq!(reason == GenerationStopReason::Eos, eos);
    assert_eq!(reason == GenerationStopReason::ContextFull, !eos && full);
    assert_eq!(reason == GenerationStopReason::MaxTokens, !eos && !full);
    kani::cover!(eos && !full);
    kani::cover!(!eos && full);
    kani::cover!(!eos && !full);
    kani::cover!(eos && full);
}

/// Enable the qualified layout by default; explicit malformed values stay off.
pub fn exl3_lane_order_enabled(value: Option<&str>) -> bool {
    value.is_none_or(|value| value == "1")
}

/// Keep persisted calibration profiles distinct for both activation layouts.
pub fn exl3_lane_order_key(enabled: bool) -> u8 {
    u8::from(enabled)
}

/// Exact inventory geometries screened in Q27-16, with stock large-M fallback.
#[cfg(any(feature = "mlx", test, kani))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn exl3_lane_order(
    rows: i32,
    input: i32,
    output: i32,
    k: usize,
    grouped: bool,
    mul1: bool,
    m5: bool,
    enabled: bool,
) -> bool {
    enabled
        && m5
        && mul1
        && matches!(rows, 1 | 2)
        && match (input, output, grouped) {
            (5120, 16384, true) => k == 2,
            (5120, 34816, true) => (1..=3).contains(&k),
            (5120, 248320, false) => k == 3,
            (5120, 1024, false) => matches!(k, 3 | 4),
            (5120, 12288, false) => matches!(k, 1 | 2),
            (5120, 10240, false) => k == 1,
            (5120, 6144, false) => k == 2,
            (6144, 5120, false) => matches!(k, 2 | 3),
            (17408, 5120, false) => (1..=3).contains(&k),
            _ => false,
        }
}

/// Within each tile16, group four values used by the same SIMD lane.
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn exl3_lane_source_index(index: u32) -> u32 {
    let within = index & 15;
    (index & !15) | ((within & 12) >> 1) | (within & 1) | ((within & 2) << 2)
}

#[test]
fn exl3_lane_order_default_is_on_and_explicit_values_are_strict() {
    for (option, expected) in [
        (None, true),
        (Some(""), false),
        (Some("0"), false),
        (Some("1"), true),
        (Some("01"), false),
        (Some("true"), false),
        (Some("1 "), false),
        (Some("１"), false),
        (Some("1\0"), false),
    ] {
        assert_eq!(exl3_lane_order_enabled(option), expected);
    }
    assert_eq!(exl3_lane_order_key(false), 0);
    assert_eq!(exl3_lane_order_key(true), 1);
}

#[test]
fn exl3_lane_order_matches_all_sixteen_inventory_shapes_and_fallbacks() {
    let inventory = [
        (5120, 16384, 2, true),
        (5120, 34816, 1, true),
        (5120, 34816, 2, true),
        (5120, 34816, 3, true),
        (5120, 248320, 3, false),
        (6144, 5120, 2, false),
        (17408, 5120, 1, false),
        (17408, 5120, 2, false),
        (5120, 1024, 3, false),
        (5120, 12288, 1, false),
        (5120, 12288, 2, false),
        (6144, 5120, 3, false),
        (17408, 5120, 3, false),
        (5120, 1024, 4, false),
        (5120, 10240, 1, false),
        (5120, 6144, 2, false),
    ];
    for (input, output, k, grouped) in inventory {
        for rows in [1, 2] {
            assert!(exl3_lane_order(
                rows, input, output, k, grouped, true, true, true
            ));
        }
        for rows in [i32::MIN, -1, 0, 3, 4, 8, 24, i32::MAX] {
            assert!(!exl3_lane_order(
                rows, input, output, k, grouped, true, true, true
            ));
        }
        for (mul1, m5, enabled) in [
            (false, true, true),
            (true, false, true),
            (true, true, false),
        ] {
            assert!(!exl3_lane_order(
                1, input, output, k, grouped, mul1, m5, enabled
            ));
        }
    }
    for (input, output, k, grouped) in [
        (2048, 5120, 2, false),
        (5120, 1025, 3, false),
        (5120, 16384, 2, false),
        (6144, 5120, 1, false),
        (5120, 248320, 2, false),
        (17408, 5120, 4, false),
        (5120, 34816, 7, true),
        (5120, 34816, usize::MAX, true),
        (i32::MAX, i32::MAX, 2, false),
    ] {
        assert!(!exl3_lane_order(
            1, input, output, k, grouped, true, true, true
        ));
    }
}

#[test]
fn exl3_lane_permutation_matches_independent_table_for_every_inventory_input() {
    let expected = [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15];
    for input in [5120u32, 6144, 17408] {
        let mut seen = vec![false; input as usize];
        for index in 0..input {
            let mapped = exl3_lane_source_index(index);
            assert_eq!(mapped, (index / 16) * 16 + expected[(index % 16) as usize]);
            assert!(mapped < input && !seen[mapped as usize]);
            seen[mapped as usize] = true;
        }
        assert!(seen.into_iter().all(|value| value));
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(10)]
fn exl3_lane_order_default_and_bounded_explicit_values() {
    let bytes: [u8; 4] = kani::any();
    let length = usize::from(kani::any::<u8>() % 5);
    let present: bool = kani::any();
    let decoded = std::str::from_utf8(&bytes[..length]);
    let value = decoded.unwrap_or("");
    let enabled = exl3_lane_order_enabled(present.then_some(value));
    assert_eq!(enabled, !present || (length == 1 && bytes[0] == b'1'));
    assert_eq!(exl3_lane_order_key(enabled), u8::from(enabled));
    kani::cover!(!present && enabled);
    kani::cover!(present && length == 1 && bytes[0] == b'1' && enabled);
    kani::cover!(present && length == 1 && bytes[0] == b'0' && !enabled);
    kani::cover!(present && length == 0 && !enabled);
    kani::cover!(present && decoded.is_err() && !enabled);
    kani::cover!(present && length > 1 && !enabled);
}

#[cfg(kani)]
#[kani::proof]
fn exl3_lane_layout_key_distinguishes_options() {
    let enabled: bool = kani::any();
    let other: bool = kani::any();
    let key = exl3_lane_order_key(enabled);
    assert!(key <= 1);
    assert_eq!(key == exl3_lane_order_key(other), enabled == other);
    kani::cover!(key == 0);
    kani::cover!(key == 1);
}

#[cfg(kani)]
#[kani::proof]
fn exl3_lane_permutation_is_bijective_and_preserves_tile_bounds() {
    let index: u32 = kani::any();
    let other: u32 = kani::any();
    let mapped = exl3_lane_source_index(index);
    assert_eq!(mapped / 16, index / 16);
    assert_eq!(
        exl3_lane_source_index(exl3_lane_source_index(mapped)),
        index
    );
    assert_eq!(mapped == exl3_lane_source_index(other), index == other);
    kani::cover!(index == 2 && mapped == 8);
    kani::cover!(index == u32::MAX && mapped == u32::MAX);
    kani::cover!(index > 15 && mapped != index);
}

#[cfg(kani)]
#[kani::proof]
fn exl3_lane_order_preserves_shape_and_large_m_fallback() {
    let rows: i32 = kani::any();
    let input: i32 = kani::any();
    let output: i32 = kani::any();
    let k: usize = kani::any();
    let grouped: bool = kani::any();
    let mul1: bool = kani::any();
    let m5: bool = kani::any();
    let enabled: bool = kani::any();
    let selected = exl3_lane_order(rows, input, output, k, grouped, mul1, m5, enabled);
    if selected {
        assert!(enabled && m5 && mul1 && (rows == 1 || rows == 2));
        assert!(matches!(input, 5120 | 6144 | 17408));
        assert!(output > 0 && output <= 248320 && output % 128 == 0);
        assert!((1..=4).contains(&k));
        assert!(!grouped || (input == 5120 && matches!(output, 16384 | 34816)));
        let index: u32 = kani::any();
        if index < input as u32 {
            assert!(exl3_lane_source_index(index) < input as u32);
        }
    }
    if rows >= 3 || !enabled || !mul1 || !m5 {
        assert!(!selected);
    }
    kani::cover!(selected && rows == 1 && output == 248320);
    kani::cover!(selected && rows == 2 && grouped && k == 1);
    kani::cover!(selected && input == 6144);
    kani::cover!(selected && input == 17408);
    kani::cover!(rows == 3 && enabled && !selected);
    kani::cover!(!enabled && rows == 1 && !selected);
}
