//! Stable one-row MoE top-k through the production MLXL3 Metal kernel.
use crate::array::{self, Array, Dtype};
use anyhow::{Result, ensure};

pub fn topk(values: &Array, top_k: usize, normalize: bool) -> Result<(Array, Array)> {
    let [rows, experts]: [i32; 2] = values
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("router values must have rank 2"))?;
    ensure!(
        rows == 1 && experts > 0 && experts <= 1024,
        "native router accepts one row and at most 1024 experts"
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
        &[vec![1, top_k as i32], vec![1, top_k as i32]],
        &[Dtype::UInt32, Dtype::Float16],
        [experts, 1, 1],
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
