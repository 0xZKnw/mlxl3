//! Stable MoE top-k through the production MLXL3 Metal kernel.
use crate::array::{self, Array, Dtype};
use anyhow::{Result, ensure};

pub fn topk(values: &Array, top_k: usize, normalize: bool) -> Result<(Array, Array)> {
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

pub fn topk_biased(values: &Array, bias: &Array, top_k: usize) -> Result<(Array, Array)> {
    let [rows, experts]: [i32; 2] = values
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("router values must have rank 2"))?;
    ensure!(
        rows == 1 && experts > 0 && experts <= 32,
        "biased router accepts one row and at most 32 experts"
    );
    ensure!(
        top_k > 0 && top_k <= experts as usize,
        "invalid router top-k"
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
            uint chosen[TOP_K];
            for (uint rank = 0u; rank < TOP_K; ++rank) {{
                float best = simd_max(candidate);
                uint possible = candidate == best ? lane : 0xffffffffu;
                uint winner = simd_min(possible);
                if (lane == 0u) chosen[rank] = winner;
                if (lane == winner) candidate = -INFINITY;
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
        &format!("mlxl3_rs_topk_biased_e{experts}_k{top_k}_v1"),
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
        rows == 1
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
    let source = format!(
        r#"
            if (thread_position_in_grid.x != 0u) return;
            constexpr uint EXPERTS = {experts}u;
            constexpr uint GROUPS = {groups}u;
            constexpr uint PER_GROUP = {per_group}u;
            constexpr uint TOP_GROUPS = {top_groups}u;
            constexpr uint TOP_K = {top_k}u;
            float probabilities[EXPERTS];
            float routing[EXPERTS];
            float group_score[GROUPS];
            bool allowed[GROUPS];
            for (uint e = 0u; e < EXPERTS; ++e) {{
                probabilities[e] = probabilities_in[e];
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
                    for (uint old = 0u; old < rank; ++old) used |= uint(indices[old]) == e;
                    if (allowed[e / PER_GROUP] && !used && routing[e] > best) {{
                        best = routing[e]; winner = e;
                    }}
                }}
                indices[rank] = winner;
                scores[rank] = probabilities[winner];
            }}
        "#
    );
    let mut outputs = array::metal_kernel(
        &format!("mlxl3_rs_group_router_e{experts}_g{groups}_tg{top_groups}_k{top_k}_v1"),
        &["probabilities_in", "bias"],
        &["indices", "scores"],
        "",
        &source,
        &[&logits.sigmoid()?, bias],
        &[vec![1, top_k as i32], vec![1, top_k as i32]],
        &[Dtype::UInt32, Dtype::Float32],
        [1, 1, 1],
        [1, 1, 1],
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
    #[ignore = "requires Apple GPU"]
    fn grouped_router_masks_unselected_groups() -> Result<()> {
        let logits = Array::from_f32(&[0.; 8], &[1, 8])?;
        let bias = Array::from_f32(&[0., 0., 0., 0., 1., 2., 3., 4.], &[8])?;
        let (indices, scores) = grouped_topk_biased(&logits, &bias, 2, 2, 1, 2.5)?;
        assert!(indices.to_u32()?.iter().all(|&index| index >= 4));
        assert!((scores.to_f32()?.iter().sum::<f32>() - 2.5).abs() < 0.002);
        Ok(())
    }
}
