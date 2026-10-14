//! Stable MoE top-k through the production MLXL3 Metal kernel.
use crate::array::{self, Array, Dtype};
use anyhow::{Context, Result, ensure};

/// Only the validated top-k kernel can construct these device-side routes.
/// Consumers verify the expert count against their weights without reading IDs
/// back to the CPU (which would break a recursive MTP graph).
pub(crate) struct RoutedExperts {
    indices: Array,
    scores: Array,
    experts: i32,
}

impl RoutedExperts {
    pub(crate) fn indices(&self) -> &Array {
        &self.indices
    }

    pub(crate) fn scores(&self) -> &Array {
        &self.scores
    }

    pub(crate) fn experts(&self) -> i32 {
        self.experts
    }
}

pub(crate) fn routes(values: &Array, top_k: usize, normalize: bool) -> Result<RoutedExperts> {
    let (indices, scores) = topk(values, top_k, normalize)?;
    Ok(RoutedExperts {
        indices,
        scores,
        experts: values.shape()[1],
    })
}

pub fn topk(values: &Array, top_k: usize, normalize: bool) -> Result<(Array, Array)> {
    if values.shape().len() == 2
        && values.shape()[1] == 256
        && (1..=4).contains(&values.shape()[0])
        && top_k == 8
        && values.dtype() == Dtype::Float16
    {
        return top8_simd(values, normalize);
    }
    topk_legacy(values, top_k, normalize)
}

fn topk_legacy(values: &Array, top_k: usize, normalize: bool) -> Result<(Array, Array)> {
    let [rows, experts]: [i32; 2] = values
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("router values must have rank 2"))?;
    ensure!(
        rows > 0 && experts > 0 && experts <= 1024,
        "native router accepts positive rows and at most 1024 experts"
    );
    ensure!(
        top_k > 0 && top_k <= experts as usize,
        "invalid router top-k"
    );
    ensure!(
        values.dtype() == Dtype::Float16,
        "native router currently expects FP16 values"
    );
    let score = if normalize { "score / sum" } else { "score" };
    let source = format!(
        r#"
            constexpr uint EXPERTS = {experts}u;
            constexpr uint TOP_K = {top_k}u;
            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;

            threadgroup ulong sort_keys[EXPERTS];
            threadgroup float top_values[TOP_K];

            float value = float(selection[row * EXPERTS + tid]);
            uint bits = value == 0.0f ? 0u : as_type<uint>(value);
            uint monotonic = isnan(value)
                ? 0xffffffffu
                : bits ^ uint((int(bits) >> 31) | int(0x80000000u));
            ulong key = (ulong(monotonic) << 32) | ulong(tid);
            sort_keys[tid] = key;
            threadgroup_barrier(mem_flags::mem_threadgroup);

            int above = 0;
            for (uint other = 0u; other < EXPERTS; ++other) {{
                above += sort_keys[other] > key ? 1 : 0;
            }}
            if (above < int(TOP_K)) {{
                uint slot = TOP_K - 1u - uint(above);
                top_values[slot] = float(values[row * EXPERTS + tid]);
                indices[row * TOP_K + slot] = tid;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            if (tid == 0u) {{
                half sum = half(0);
                for (uint slot = 0u; slot < TOP_K; ++slot) {{
                    sum = half(top_values[slot]) + sum;
                }}
                for (uint slot = 0u; slot < TOP_K; ++slot) {{
                    half score = half(top_values[slot]);
                    scores[row * TOP_K + slot] = {score};
                }}
            }}
        "#
    );
    let mut outputs = array::metal_kernel(
        &format!(
            "mlxl3_rs_router_e{experts}_k{top_k}_n{}_v2",
            u8::from(normalize)
        ),
        &["selection", "values"],
        &["indices", "scores"],
        "",
        &source,
        &[values, values],
        &[vec![rows, top_k as i32], vec![rows, top_k as i32]],
        &[Dtype::UInt32, Dtype::Float16],
        [experts, rows, 1],
        [experts, 1, 1],
    )?;
    ensure!(
        outputs.len() == 2,
        "router kernel returned wrong output count"
    );
    let scores = outputs.pop().expect("checked output count");
    let indices = outputs.pop().expect("checked output count");
    Ok((indices, scores))
}

/// MTPLX's row-owned two-stage selection, adapted to MLXL3's FP16 sort keys.
/// The tombstone key is below every finite/inf/NaN FP16 key. Index breaks ties;
/// the final eight scores retain the legacy ascending half-add/divide order.
fn top8_simd(values: &Array, normalize: bool) -> Result<(Array, Array)> {
    ensure!(
        values.shape().len() == 2 && values.shape()[1] == 256 && values.dtype() == Dtype::Float16,
        "SIMD router requires FP16 rows of 256 experts"
    );
    let rows = values.shape()[0];
    ensure!((1..=4).contains(&rows), "SIMD router requires 1..4 rows");
    let score = if normalize { "score / sum" } else { "score" };
    let source = format!(
        r#"
        uint row = threadgroup_position_in_grid.y;
        uint sg = simdgroup_index_in_threadgroup;
        uint lane = thread_index_in_simdgroup;
        uint index = sg * 32u + lane;
        float value = float(values[row * 256u + index]);
        uint bits = value == 0.0f ? 0u : as_type<uint>(value);
        uint key = isnan(value) ? 0xffffffffu
            : bits ^ uint((int(bits) >> 31) | int(0x80000000u));
        threadgroup uint local_keys[64];
        threadgroup uint local_ids[64];
        threadgroup uint selected[8];
        for (uint rank = 0u; rank < 8u; ++rank) {{
            uint best = simd_max(key);
            uint winner = simd_max(key == best ? index : 0u);
            if (lane == 0u) {{
                local_keys[sg * 8u + rank] = best;
                local_ids[sg * 8u + rank] = winner;
            }}
            if (index == winner) key = 0u;
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0u) {{
            uint k0 = local_keys[lane], k1 = local_keys[lane + 32u];
            uint i0 = local_ids[lane], i1 = local_ids[lane + 32u];
            for (uint rank = 0u; rank < 8u; ++rank) {{
                bool second = k1 > k0 || (k1 == k0 && i1 > i0);
                uint candidate = second ? k1 : k0;
                uint id = second ? i1 : i0;
                uint best = simd_max(candidate);
                uint winner = simd_max(candidate == best ? id : 0u);
                if (lane == 0u) selected[7u - rank] = winner;
                if (id == winner) {{ if (second) k1 = 0u; else k0 = 0u; }}
            }}
            if (lane == 0u) {{
                half sum = half(0);
                for (uint slot = 0u; slot < 8u; ++slot) {{
                    float value = float(values[row * 256u + selected[slot]]);
                    sum = half(isnan(value) ? NAN : value) + sum;
                }}
                for (uint slot = 0u; slot < 8u; ++slot) {{
                    indices[row * 8u + slot] = selected[slot];
                    float value = float(values[row * 256u + selected[slot]]);
                    half score = half(isnan(value) ? NAN : value);
                    scores[row * 8u + slot] = {score};
                }}
            }}
        }}
    "#
    );
    let mut out = array::metal_kernel(
        &format!("mlxl3_rs_router_e256_k8_simd_n{}", u8::from(normalize)),
        &["values"],
        &["indices", "scores"],
        "",
        &source,
        &[values],
        &[vec![rows, 8], vec![rows, 8]],
        &[Dtype::UInt32, Dtype::Float16],
        [256, rows, 1],
        [256, 1, 1],
    )?;
    let scores = out.pop().context("missing SIMD scores")?;
    let indices = out.pop().context("missing SIMD indices")?;
    Ok((indices, scores))
}

pub fn topk_biased(values: &Array, bias: &Array, top_k: usize) -> Result<(Array, Array)> {
    let [rows, experts]: [i32; 2] = values
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("router values must have rank 2"))?;
    ensure!(
        crate::contracts::biased_router_domain(rows, experts, top_k),
        "biased router requires one row, 1..=64 experts and a valid top-k"
    );
    ensure!(
        values.dtype() == Dtype::Float16
            && bias.dtype() == Dtype::Float16
            && bias.shape() == [experts],
        "biased router expects FP16 values and bias"
    );
    let source = format!(
        r#"
            constexpr uint EXPERTS = {experts}u;
            constexpr uint TOP_K = {top_k}u;
            uint lane = thread_index_in_simdgroup;
            half probability = lane < EXPERTS ? probabilities[lane] : half(0);
            float candidate = lane < EXPERTS
                ? float(probability) + float(bias[lane])
                : -INFINITY;
            uint second_expert = lane + 32u;
            float second = second_expert < EXPERTS
                ? float(probabilities[second_expert]) + float(bias[second_expert])
                : -INFINITY;
            uint chosen[TOP_K];
            for (uint rank = 0u; rank < TOP_K; ++rank) {{
                float best = simd_max(max(candidate, second));
                uint possible = min(candidate == best ? lane : 0xffffffffu,
                    second == best ? second_expert : 0xffffffffu);
                uint winner = simd_min(possible);
                if (lane == 0u) chosen[rank] = winner;
                if (lane == winner) candidate = -INFINITY;
                if (second_expert == winner) second = -INFINITY;
            }}
            if (lane == 0u) {{
                for (uint slot = 0u; slot < TOP_K; ++slot) {{
                    uint expert = chosen[TOP_K - 1u - slot];
                    indices[slot] = expert;
                    scores[slot] = probabilities[expert];
                }}
            }}
        "#
    );
    let mut outputs = array::metal_kernel(
        &format!("mlxl3_rs_topk_biased_e{experts}_k{top_k}_v2"),
        &["probabilities", "bias"],
        &["indices", "scores"],
        "",
        &source,
        &[values, bias],
        &[vec![1, top_k as i32], vec![1, top_k as i32]],
        &[Dtype::UInt32, Dtype::Float16],
        [32, 1, 1],
        [32, 1, 1],
    )?;
    ensure!(
        outputs.len() == 2,
        "biased router returned wrong output count"
    );
    let scores = outputs.pop().expect("checked output count");
    let indices = outputs.pop().expect("checked output count");
    Ok((indices, scores))
}

#[allow(clippy::too_many_arguments)]
pub fn grouped_topk_biased(
    logits: &Array,
    bias: &Array,
    top_k: usize,
    groups: usize,
    top_groups: usize,
    scale: f32,
) -> Result<(Array, Array)> {
    let [rows, experts]: [i32; 2] = logits
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("router logits must have rank 2"))?;
    ensure!(
        rows > 0
            && experts > 0
            && experts <= 1024
            && (experts as usize).is_multiple_of(groups)
            && top_k > 0
            && top_k <= experts as usize
            && top_groups > 0
            && top_groups <= groups
            && scale.is_finite()
            && scale > 0.,
        "invalid grouped router dimensions"
    );
    ensure!(
        logits.dtype() == Dtype::Float32
            && bias.dtype() == Dtype::Float32
            && bias.shape() == [experts],
        "grouped router expects FP32 logits and bias"
    );
    let per_group = experts as usize / groups;
    let serial_source = format!(
        r#"
            uint row = thread_position_in_grid.x;
            constexpr uint EXPERTS = {experts}u;
            constexpr uint GROUPS = {groups}u;
            constexpr uint PER_GROUP = {per_group}u;
            constexpr uint TOP_GROUPS = {top_groups}u;
            constexpr uint TOP_K = {top_k}u;
            float probabilities[EXPERTS];
            float routing[EXPERTS];
            float group_score[GROUPS];
            bool allowed[GROUPS];
            uint route_base = row * TOP_K;
            for (uint e = 0u; e < EXPERTS; ++e) {{
                probabilities[e] = probabilities_in[row * EXPERTS + e];
                routing[e] = probabilities[e] + bias[e];
            }}
            for (uint group = 0u; group < GROUPS; ++group) {{
                float first = -INFINITY, second = -INFINITY;
                for (uint local = 0u; local < PER_GROUP; ++local) {{
                    float value = routing[group * PER_GROUP + local];
                    if (value > first) {{ second = first; first = value; }}
                    else if (value > second) second = value;
                }}
                group_score[group] = first + second;
                allowed[group] = false;
            }}
            for (uint rank = 0u; rank < TOP_GROUPS; ++rank) {{
                float best = -INFINITY; uint winner = 0u;
                for (uint group = 0u; group < GROUPS; ++group) {{
                    if (!allowed[group] && group_score[group] > best) {{
                        best = group_score[group]; winner = group;
                    }}
                }}
                allowed[winner] = true;
            }}
            for (uint rank = 0u; rank < TOP_K; ++rank) {{
                float best = -INFINITY; uint winner = 0u;
                for (uint e = 0u; e < EXPERTS; ++e) {{
                    bool used = false;
                    for (uint old = 0u; old < rank; ++old) used |= uint(indices[route_base + old]) == e;
                    if (allowed[e / PER_GROUP] && !used && routing[e] > best) {{
                        best = routing[e]; winner = e;
                    }}
                }}
                indices[route_base + rank] = winner;
                scores[route_base + rank] = probabilities[winner];
            }}
        "#
    );
    let parallel = experts == 128;
    let source = if parallel {
        format!(
            r#"
            uint row = threadgroup_position_in_grid.y;
            uint tid = thread_position_in_threadgroup.x;
            constexpr uint EXPERTS = {experts}u;
            constexpr uint GROUPS = {groups}u;
            constexpr uint PER_GROUP = {per_group}u;
            constexpr uint TOP_GROUPS = {top_groups}u;
            constexpr uint TOP_K = {top_k}u;
            threadgroup float probabilities[EXPERTS];
            threadgroup float routing[EXPERTS];
            threadgroup float group_score[GROUPS];
            threadgroup bool allowed[GROUPS];
            threadgroup bool used[EXPERTS];
            threadgroup float candidate[EXPERTS];
            uint route_base = row * TOP_K;
            probabilities[tid] = probabilities_in[row * EXPERTS + tid];
            routing[tid] = probabilities[tid] + bias[tid];
            used[tid] = false;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < GROUPS) {{
                float first = -INFINITY, second = -INFINITY;
                for (uint local = 0u; local < PER_GROUP; ++local) {{
                    float value = routing[tid * PER_GROUP + local];
                    if (value > first) {{ second = first; first = value; }}
                    else if (value > second) second = value;
                }}
                group_score[tid] = first + second;
                allowed[tid] = false;
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {{
                for (uint rank = 0u; rank < TOP_GROUPS; ++rank) {{
                    float best = -INFINITY; uint winner = 0u;
                    for (uint group = 0u; group < GROUPS; ++group) {{
                        if (!allowed[group] && group_score[group] > best) {{
                            best = group_score[group]; winner = group;
                        }}
                    }}
                    allowed[winner] = true;
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint rank = 0u; rank < TOP_K; ++rank) {{
                candidate[tid] = allowed[tid / PER_GROUP] && !used[tid]
                    ? routing[tid] : -INFINITY;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {{
                    float best = -INFINITY; uint winner = 0u;
                    for (uint e = 0u; e < EXPERTS; ++e) {{
                        if (candidate[e] > best) {{ best = candidate[e]; winner = e; }}
                    }}
                    used[winner] = true;
                    indices[route_base + rank] = winner;
                    scores[route_base + rank] = probabilities[winner];
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }}
            "#
        )
    } else {
        serial_source
    };
    let mut outputs = array::metal_kernel(
        &format!("mlxl3_rs_group_router_e{experts}_g{groups}_tg{top_groups}_k{top_k}_v2"),
        &["probabilities_in", "bias"],
        &["indices", "scores"],
        "",
        &source,
        &[&logits.sigmoid()?, bias],
        &[vec![rows, top_k as i32], vec![rows, top_k as i32]],
        &[Dtype::UInt32, Dtype::Float32],
        if parallel {
            [experts, rows, 1]
        } else {
            [rows, 1, 1]
        },
        if parallel { [experts, 1, 1] } else { [1, 1, 1] },
    )?;
    let scores = outputs.pop().expect("checked router output count");
    let indices = outputs.pop().expect("checked router output count");
    let scores = scores.div(&scores.sum(1, true)?)?.scalar_mul(scale)?;
    Ok((indices, scores))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "physical Apple GPU; LFM2.5 biased 64-expert regression"]
    fn biased_router_matches_independent_sort_through_64_experts() -> Result<()> {
        use half::f16;
        for experts in [1, 2, 31, 32, 33, 63, 64] {
            for fixture in 0..5 {
                let probabilities = (0..experts)
                    .map(|i| {
                        f16::from_f32(match fixture {
                            0 => ((i * 37 % 101) + 1) as f32 / 128.,
                            1 | 2 => 0.5,
                            3 => (i + 1) as f32 / 128.,
                            _ => {
                                if i % 2 == 0 {
                                    0.
                                } else {
                                    -0.
                                }
                            }
                        })
                        .to_bits()
                    })
                    .collect::<Vec<_>>();
                let biases = (0..experts)
                    .map(|i| {
                        f16::from_f32(match fixture {
                            0 => ((i * 13 % 29) - 14) as f32 / 64.,
                            2 if i == experts - 1 => 1.,
                            _ => 0.,
                        })
                        .to_bits()
                    })
                    .collect::<Vec<_>>();
                let values = Array::from_f16_bits(&probabilities, &[1, experts])?;
                let bias = Array::from_f16_bits(&biases, &[experts])?;
                let selection = probabilities
                    .iter()
                    .zip(&biases)
                    .map(|(p, b)| f16::from_bits(*p).to_f32() + f16::from_bits(*b).to_f32())
                    .collect::<Vec<_>>();
                for top_k in [1, 4.min(experts as usize), experts as usize] {
                    let mut order = (0..experts as u32).collect::<Vec<_>>();
                    order.sort_by(|a, b| {
                        selection[*b as usize]
                            .partial_cmp(&selection[*a as usize])
                            .expect("finite routing fixture")
                            .then(a.cmp(b))
                    });
                    order.truncate(top_k);
                    order.reverse();
                    let expected_scores = order
                        .iter()
                        .map(|&index| probabilities[index as usize])
                        .collect::<Vec<_>>();
                    let (indices, scores) = topk_biased(&values, &bias, top_k)?;
                    assert_eq!(indices.shape(), [1, top_k as i32]);
                    assert_eq!(
                        indices.to_u32()?,
                        order,
                        "E={experts}, K={top_k}, fixture={fixture}"
                    );
                    assert_eq!(scores.to_f16_bits()?, expected_scores);
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU; biased router boundary validation"]
    fn biased_router_rejects_invalid_inputs_before_dispatch() -> Result<()> {
        let values = Array::from_f16_bits(&[0x3800; 64], &[1, 64])?;
        let bias = Array::from_f16_bits(&[0; 64], &[64])?;
        for top_k in [0, 65, usize::MAX] {
            assert!(topk_biased(&values, &bias, top_k).is_err());
        }
        let rows = Array::from_f16_bits(&[0x3800; 128], &[2, 64])?;
        assert!(topk_biased(&rows, &bias, 4).is_err());
        let rank = values.reshape(&[64])?;
        assert!(topk_biased(&rank, &bias, 4).is_err());
        let experts = Array::from_f16_bits(&[0x3800; 65], &[1, 65])?;
        let extra_bias = Array::from_f16_bits(&[0; 65], &[65])?;
        assert!(topk_biased(&experts, &extra_bias, 4).is_err());
        assert!(topk_biased(&values.astype(Dtype::Float32)?, &bias, 4).is_err());
        assert!(topk_biased(&values, &bias.astype(Dtype::Float32)?, 4).is_err());
        assert!(topk_biased(&values, &bias.reshape(&[1, 64])?, 4).is_err());
        assert!(topk_biased(&values, &bias.slice(0, 0, 63)?, 4).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU, preregistered MTP-05 router trial"]
    fn simd_top8_matches_legacy_bits_and_profiles_small_rows() -> Result<()> {
        use std::time::Instant;
        for rows in 1..=4 {
            for fixture in 0..12 {
                let bits = (0..rows * 256)
                    .map(|i| match fixture {
                        0 => 0x3800,
                        1 => {
                            if i % 2 == 0 {
                                0
                            } else {
                                0x8000
                            }
                        }
                        2 => [0x7c00, 0xfc00, 0x7e00, 0xfe00, 1, 0x8001, 0x3c00, 0xbc00]
                            [i as usize % 8],
                        _ => (i as u16).wrapping_mul(179).wrapping_add(fixture * 5557),
                    })
                    .collect::<Vec<_>>();
                let values = Array::from_f16_bits(&bits, &[rows, 256])?;
                for normalize in [false, true] {
                    let old = topk_legacy(&values, 8, normalize)?;
                    let new = top8_simd(&values, normalize)?;
                    let dispatched = topk(&values, 8, normalize)?;
                    assert_eq!(dispatched.0.to_u32()?, new.0.to_u32()?);
                    assert_eq!(dispatched.1.to_f16_bits()?, new.1.to_f16_bits()?);
                    assert_eq!(
                        new.0.to_u32()?,
                        old.0.to_u32()?,
                        "IDs rows={rows} fixture={fixture}"
                    );
                    assert_eq!(
                        new.1.to_f16_bits()?,
                        old.1.to_f16_bits()?,
                        "scores rows={rows} fixture={fixture} normalize={normalize}"
                    );
                }
            }
            let values = Array::from_f16_bits(
                &(0..rows * 256)
                    .map(|i| half::f16::from_f32((1 + i % 256) as f32 / 512.).to_bits())
                    .collect::<Vec<_>>(),
                &[rows, 256],
            )?;
            let mut elapsed = [Vec::new(), Vec::new()];
            for repetition in 0..44 {
                for variant in if repetition % 2 == 0 { [0, 1] } else { [1, 0] } {
                    let start = Instant::now();
                    let result = if variant == 0 {
                        topk_legacy(&values, 8, true)?
                    } else {
                        top8_simd(&values, true)?
                    };
                    result.0.eval()?;
                    result.1.eval()?;
                    if repetition >= 4 {
                        elapsed[variant].push(start.elapsed().as_secs_f64() * 1e6);
                    }
                }
            }
            for values in &mut elapsed {
                values.sort_by(f64::total_cmp);
            }
            eprintln!(
                "router rows={rows} legacy_us={} simd_us={} repetitions=40 host+GPU",
                elapsed[0][20], elapsed[1][20]
            );
        }
        let wrong_dtype = Array::from_f32(&[0.; 256], &[1, 256])?;
        assert!(topk(&wrong_dtype, 8, true).is_err());
        assert!(topk(&wrong_dtype.astype(Dtype::Float16)?, 0, true).is_err());
        assert!(topk(&wrong_dtype.astype(Dtype::Float16)?, 257, true).is_err());
        let fallback = Array::from_f16_bits(&[0x3800; 1280], &[5, 256])?;
        assert!(top8_simd(&fallback, true).is_err());
        assert_eq!(
            topk(&fallback, 8, true)?.0.to_u32()?,
            topk_legacy(&fallback, 8, true)?.0.to_u32()?
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires Apple GPU"]
    fn grouped_router_masks_unselected_groups() -> Result<()> {
        let logits = Array::from_f32(&[0.; 8], &[1, 8])?;
        let bias = Array::from_f32(&[0., 0., 0., 0., 1., 2., 3., 4.], &[8])?;
        let (indices, scores) = grouped_topk_biased(&logits, &bias, 2, 2, 1, 2.5)?;
        assert!(indices.to_u32()?.iter().all(|&index| index >= 4));
        assert!((scores.to_f32()?.iter().sum::<f32>() - 2.5).abs() < 0.002);
        Ok(())
    }

    #[test]
    #[ignore = "requires Apple GPU"]
    fn grouped_router_batch_matches_serial() -> Result<()> {
        let logits = (0..4 * 128)
            .map(|index| ((index * 53) % 409) as f32 / 37.)
            .collect::<Vec<_>>();
        let logits = Array::from_f32(&logits, &[4, 128])?;
        let bias = Array::from_f32(
            &(0..128)
                .map(|index| index as f32 / 1000.)
                .collect::<Vec<_>>(),
            &[128],
        )?;
        let (batch_indices, batch_scores) = grouped_topk_biased(&logits, &bias, 8, 8, 4, 2.5)?;
        let batch_indices = batch_indices.to_u32()?;
        let batch_scores = batch_scores.to_f32()?;
        for row in 0..4 {
            let (indices, scores) =
                grouped_topk_biased(&logits.slice(0, row, row + 1)?, &bias, 8, 8, 4, 2.5)?;
            assert_eq!(
                &batch_indices[row as usize * 8..(row as usize + 1) * 8],
                indices.to_u32()?
            );
            assert_eq!(
                &batch_scores[row as usize * 8..(row as usize + 1) * 8],
                scores.to_f32()?
            );
        }
        Ok(())
    }
}
