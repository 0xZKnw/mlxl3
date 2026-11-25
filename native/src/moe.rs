//! Qwen decode-only MoE transforms around the mapped EXL3 projections.
use crate::{
    array::{self, Array, Dtype},
    checkpoint::Checkpoint,
    codec::{self, Codebook},
    linear::{checkpoint_arrays, codebook_header, expert_mapped},
};
use anyhow::{Context, Result, ensure};

fn load_workers() -> Result<usize> {
    if let Ok(value) = std::env::var("MLXL3_LOAD_THREADS") {
        let workers = value
            .parse::<usize>()
            .context("MLXL3_LOAD_THREADS must be an integer")?;
        ensure!(
            (1..=32).contains(&workers),
            "MLXL3_LOAD_THREADS must be 1..=32"
        );
        return Ok(workers);
    }
    Ok(std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(8))
}

fn butterfly<'a>(mut source: &'a str, mut target: &'a str, round_after_four: bool) -> String {
    let mut stages = String::new();
    for shift in 0..7 {
        let step = 1 << shift;
        stages += &format!(
            r#"
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float {source}_value_{shift} = {source}[tid];
            float {source}_peer_{shift} = {source}[tid ^ {step}u];
            {target}[tid] = (tid & {step}u)
                ? {source}_peer_{shift} - {source}_value_{shift}
                : {source}_value_{shift} + {source}_peer_{shift};
            "#
        );
        if round_after_four && shift == 3 {
            stages += &format!("{target}[tid] = float(half({target}[tid]));\n");
        }
        std::mem::swap(&mut source, &mut target);
    }
    stages
}

#[derive(Clone, Copy)]
enum GluActivation {
    Silu,
    Gelu,
    /// Inkling 2: `softcap(g, gate_cap) * sigmoid(g) * softcap(u, up_cap)`, `softcap(x, c) = c * tanh(x / c)`.
    SoftcapSilu {
        gate_cap: f32,
        up_cap: f32,
    },
}

impl GluActivation {
    /// Metal expression of the activation from `gate_value` and `up_value`, and the kernel-name tag
    /// that identifies it (every constant baked into the source is also in the name).
    fn metal(self) -> (String, String) {
        match self {
            Self::Silu => (
                "float(half((gate_value / (1.0f + exp(-gate_value))) * up_value))".into(),
                "silu".into(),
            ),
            Self::Gelu => (
                "float(half(half(0.5f * gate_value * (1.0f + tanh(0.7978845608028654f * (gate_value + 0.044715f * gate_value * gate_value * gate_value)))) * half(up_value)))".into(),
                "gelu".into(),
            ),
            Self::SoftcapSilu { gate_cap, up_cap } => (
                format!(
                    "float(half((tanh(gate_value / {gate_cap:?}f) * {gate_cap:?}f) \
                     * (1.0f / (1.0f + exp(-gate_value))) \
                     * (tanh(up_value / {up_cap:?}f) * {up_cap:?}f)))"
                ),
                format!("softcap_g{:08x}_u{:08x}", gate_cap.to_bits(), up_cap.to_bits()),
            ),
        }
    }
}

struct RoutePlan {
    experts: Array,
    tokens: Array,
    inverse: Array,
    table: Array,
    count: Array,
    slots: i32,
    padded: i32,
    max_blocks: i32,
}

fn route_plan(
    selected: &Array,
    rows: i32,
    top_k: i32,
    experts: i32,
    parallel: bool,
) -> Result<RoutePlan> {
    let slots = rows.checked_mul(top_k).context("MoE slot overflow")?;
    ensure!(
        selected.shape() == [slots]
            && selected.dtype() == Dtype::UInt32
            && experts > 0
            && experts <= 256,
        "invalid segmented MoE routes"
    );
    let padded = slots.checked_add(64).context("MoE padding overflow")?;
    let max_blocks = slots
        .checked_div(64)
        .and_then(|n| n.checked_add(experts + 1))
        .context("MoE block table overflow")?;
    let order = selected.argsort()?;
    let planner = if parallel && experts == 256 && top_k == 8 && (64..=256).contains(&rows) {
        include_str!("../shaders/q36_parallel_route_plan.metal")
    } else {
        r#"
            if (tid == 0u) {
                uint next_block = 0u;
                for (uint expert = 0u; expert < EXPERTS; ++expert) {
                    uint lo = 0u, hi = SLOTS;
                    while (lo < hi) {
                        uint mid = (lo + hi) >> 1u;
                        if (selected[order[mid]] < expert) lo = mid + 1u;
                        else hi = mid;
                    }
                    uint begin = lo;
                    hi = SLOTS;
                    while (lo < hi) {
                        uint mid = (lo + hi) >> 1u;
                        if (selected[order[mid]] <= expert) lo = mid + 1u;
                        else hi = mid;
                    }
                    uint count = lo - begin;
                    for (uint offset = 0u; offset < count; offset += 64u) {
                        block_table[next_block] = expert;
                        block_table[MAX_BLOCKS + next_block] = begin + offset;
                        block_table[2u * MAX_BLOCKS + next_block] = min(64u, count - offset);
                        ++next_block;
                    }
                }
                block_count[0] = next_block;
            }
"#
    };
    let source = format!(
        r#"
            constexpr uint SLOTS = {slots}u;
            constexpr uint PADDED = {padded}u;
            constexpr uint TOP_K = {top_k}u;
            constexpr uint EXPERTS = {experts}u;
            constexpr uint MAX_BLOCKS = {max_blocks}u;
            uint tid = thread_position_in_threadgroup.x;

            for (uint i = tid; i < SLOTS; i += 256u) {{
                uint original = order[i];
                sorted_experts[i] = selected[original];
                token_rows[i] = original / TOP_K;
                inverse[original] = i;
            }}
            for (uint i = SLOTS + tid; i < PADDED; i += 256u) {{
                sorted_experts[i] = 0u;
                token_rows[i] = 0u;
            }}

            {planner}
        "#
    );
    let mut outputs = array::metal_kernel(
        &format!("mlxl3_rs_route_plan_r{rows}_k{top_k}_e{experts}_v1"),
        &["selected", "order"],
        &[
            "sorted_experts",
            "token_rows",
            "inverse",
            "block_table",
            "block_count",
        ],
        "",
        &source,
        &[selected, &order],
        &[
            vec![padded],
            vec![padded],
            vec![slots],
            vec![3, max_blocks],
            vec![1],
        ],
        &[
            Dtype::UInt32,
            Dtype::UInt32,
            Dtype::UInt32,
            Dtype::UInt32,
            Dtype::UInt32,
        ],
        [256, 1, 1],
        [256, 1, 1],
    )?;
    Ok(RoutePlan {
        experts: outputs.remove(0),
        tokens: outputs.remove(0),
        inverse: outputs.remove(0),
        table: outputs.remove(0),
        count: outputs.remove(0),
        slots,
        padded,
        max_blocks,
    })
}

fn routed_inputs(x: &Array, scales: &Array, plan: &RoutePlan, input: i32) -> Result<Array> {
    ensure!(
        x.shape().len() == 2
            && x.shape()[1] == input
            && scales.shape().len() == 3
            && scales.shape()[1..] == [2, input]
            && input % 128 == 0,
        "invalid routed prefill input"
    );
    let source = format!(
        r#"
            constexpr uint INPUT_DIMS = {input}u;
            uint tid = thread_position_in_threadgroup.x;
            uint block = threadgroup_position_in_grid.x;
            uint route = threadgroup_position_in_grid.y;
            uint projection = threadgroup_position_in_grid.z;
            uint token = tokens[route];
            uint expert = experts[route];
            uint base = block * 128u + tid * 4u;
            float v[4];
            for (uint r = 0u; r < 4u; ++r) {{
                uint column = base + r;
                v[r] = float(half(half(x[token * INPUT_DIMS + column]) *
                    half(scales[(expert * 2u + projection) * INPUT_DIMS + column])));
            }}
            for (uint h = 1u; h < 4u; h *= 2u) {{
                for (uint i = 0u; i < 2u; ++i) {{
                    uint k = i & (h - 1u);
                    uint j = ((i - k) << 1u) + k;
                    float a = v[j], b = v[j + h];
                    v[j] = a + b;
                    v[j + h] = a - b;
                }}
            }}
            for (uint h = 1u; h < 4u; h *= 2u) {{
                for (uint r = 0u; r < 4u; ++r) {{
                    float peer = simd_shuffle_xor(v[r], h);
                    v[r] = (tid & h) ? peer - v[r] : v[r] + peer;
                }}
            }}
            for (uint r = 0u; r < 4u; ++r) v[r] = float(half(v[r]));
            for (uint h = 4u; h < 32u; h *= 2u) {{
                for (uint r = 0u; r < 4u; ++r) {{
                    float peer = simd_shuffle_xor(v[r], h);
                    v[r] = (tid & h) ? peer - v[r] : v[r] + peer;
                }}
            }}
            for (uint r = 0u; r < 4u; ++r) {{
                out[(projection * uint(tokens_shape[0]) + route) * INPUT_DIMS + base + r] =
                    half(float(half(v[r])) * 0.08838834764831845f);
            }}
        "#
    );
    Ok(array::metal_kernel(
        &format!("mlxl3_rs_routed_input_d{input}_v1"),
        &["x", "scales", "tokens", "experts"],
        &["out"],
        "",
        &source,
        &[x, scales, &plan.tokens, &plan.experts],
        &[vec![2, plan.padded, input]],
        &[Dtype::Float16],
        [input / 4, plan.padded, 2],
        [32, 1, 1],
    )?
    .remove(0))
}

// The mapped half4 reader consumes four values from each 16-element tile.
const LANE_STORE: &str =
    "((column & ~15u) | (column & 1u) | ((column & 6u) << 1u) | ((column & 8u) >> 2u))";

/// INVARIANT-2026-11-25-INK2M-L3-3: the exact expert panel in invariant verifications (experts shared by several
/// rows read once; verification of 8 rows measured 13 % faster, the same bits). On by default;
/// `MLXL3_INKLING_EXPERT_PANEL=0` turns it off.
pub(crate) fn expert_panel_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = EXPERT_PANEL_TEST.with(std::cell::Cell::get) {
        return value;
    }
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE
        .get_or_init(|| panel_setting(std::env::var("MLXL3_INKLING_EXPERT_PANEL").ok().as_deref()))
}

/// The panel setting of `MLXL3_INKLING_EXPERT_PANEL`: on unless "0".
fn panel_setting(value: Option<&str>) -> bool {
    value != Some("0")
}

#[cfg(test)]
thread_local! {
    static EXPERT_PANEL_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// L3-2 A/B in one process: runs `run` with the expert panel on or off.
#[cfg(test)]
pub(crate) fn with_expert_panel<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            EXPERT_PANEL_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(EXPERT_PANEL_TEST.with(|flag| flag.replace(Some(enabled))));
    run()
}

/// DFLASH2-2026-11-24-INK2M-P4-2: grouped TensorOps experts for Inkling calls of 2..=8 rows
/// (`MLXL3_INKLING_EXPERT_TENSOR=1`; off by default: not bit-exact).
pub(crate) fn expert_tensor_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = EXPERT_TENSOR_TEST.with(std::cell::Cell::get) {
        return value;
    }
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| std::env::var("MLXL3_INKLING_EXPERT_TENSOR").as_deref() == Ok("1"))
}

#[cfg(test)]
thread_local! {
    static EXPERT_TENSOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// P4-2 A/B in one process: runs `run` with the grouped TensorOps experts on or off.
#[cfg(test)]
pub(crate) fn with_expert_tensor<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            EXPERT_TENSOR_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(EXPERT_TENSOR_TEST.with(|flag| flag.replace(Some(enabled))));
    run()
}

fn routed_inputs_small(
    x: &Array,
    scales: &Array,
    selected: &Array,
    rows: i32,
    top_k: i32,
    input: i32,
    lane_order: bool,
) -> Result<Array> {
    let slots = rows.checked_mul(top_k).context("MoE slot overflow")?;
    ensure!(
        x.shape() == [rows, input]
            && scales.shape().len() == 3
            && scales.shape()[1..] == [2, input]
            && selected.shape() == [slots]
            && selected.dtype() == Dtype::UInt32
            && input % 128 == 0,
        "invalid small routed input"
    );
    let output_column = if lane_order { LANE_STORE } else { "column" };
    let source = format!(
        r#"
            constexpr uint INPUT_DIMS = {input}u;
            constexpr uint TOP_K = {top_k}u;
            uint tid = thread_position_in_threadgroup.x;
            uint block = threadgroup_position_in_grid.x;
            uint route = threadgroup_position_in_grid.y;
            uint projection = threadgroup_position_in_grid.z;
            uint token = route / TOP_K;
            uint expert = selected[route];
            uint base = block * 128u + tid * 4u;
            float v[4];
            for (uint r = 0u; r < 4u; ++r) {{
                uint column = base + r;
                v[r] = float(half(half(x[token * INPUT_DIMS + column]) *
                    half(scales[(expert * 2u + projection) * INPUT_DIMS + column])));
            }}
            for (uint h = 1u; h < 4u; h *= 2u) {{
                for (uint i = 0u; i < 2u; ++i) {{
                    uint k = i & (h - 1u);
                    uint j = ((i - k) << 1u) + k;
                    float a = v[j], b = v[j + h];
                    v[j] = a + b;
                    v[j + h] = a - b;
                }}
            }}
            for (uint h = 1u; h < 4u; h *= 2u) {{
                for (uint r = 0u; r < 4u; ++r) {{
                    float peer = simd_shuffle_xor(v[r], h);
                    v[r] = (tid & h) ? peer - v[r] : v[r] + peer;
                }}
            }}
            for (uint r = 0u; r < 4u; ++r) v[r] = float(half(v[r]));
            for (uint h = 4u; h < 32u; h *= 2u) {{
                for (uint r = 0u; r < 4u; ++r) {{
                    float peer = simd_shuffle_xor(v[r], h);
                    v[r] = (tid & h) ? peer - v[r] : v[r] + peer;
                }}
            }}
            for (uint r = 0u; r < 4u; ++r) {{
                uint column = base + r;
                out[((route * 2u + projection) * INPUT_DIMS) + {output_column}] =
                    half(float(half(v[r])) * 0.08838834764831845f);
            }}
        "#
    );
    Ok(array::metal_kernel(
        &format!("mlxl3_rs_routed_small_d{input}_k{top_k}_v1"),
        &["x", "scales", "selected"],
        &["out"],
        "",
        &source,
        &[x, scales, selected],
        &[vec![slots * 2, input]],
        &[Dtype::Float16],
        [input / 4, slots, 2],
        [32, 1, 1],
    )?
    .remove(0))
}

#[allow(clippy::too_many_arguments)]
fn expert_segmented_qmm(
    xhat: &Array,
    trellis: &Array,
    plan: &RoutePlan,
    input: i32,
    output: i32,
    experts: i32,
    tile_offset: i32,
    k: usize,
    cb: Codebook,
) -> Result<Array> {
    ensure!(
        array::supports_exl3_tensor_ops()?
            && k != 7
            && input % 16 == 0
            && output % 32 == 0
            && xhat.shape() == [plan.padded, input]
            && trellis.shape()[0] * 16 == input
            && trellis.shape()[2] == (16 * k) as i32,
        "invalid segmented expert QMM"
    );
    let tiles_per_expert = output / 16;
    let tiles_n = trellis.shape()[1];
    ensure!(
        tile_offset >= 0 && tile_offset + experts * tiles_per_expert <= tiles_n,
        "segmented expert QMM weight view exceeds trellis"
    );
    let inverse = codec::permutation_inverse()
        .map(|value| value.to_string())
        .join(",");
    let header = format!(
        "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\nusing namespace mpp;\n{}constant ushort mlxl3_perm_inv[256] = {{{inverse}}};\n#define BM 32u\n#define BN 32u\n#define BK 16u\n#define K_BITS {k}u\n#define PACKED_U32 {}u\n#define INPUT_DIMS {input}u\n#define OUTPUT_DIMS {output}u\n#define TILES_PER_EXPERT {tiles_per_expert}u\n#define TILES_N {tiles_n}u\n#define WEIGHT_TILE_OFFSET {tile_offset}u\n#define MAX_BLOCKS {}u\n",
        codebook_header(cb),
        k * 8,
        plan.max_blocks,
    );
    let shared = expert_qmm_shared_decode();
    let (variant, source, subblocks) = if shared {
        (
            "shared",
            include_str!("../shaders/_expert_qmm_shared_tensor_kernel.metal"),
            1,
        )
    } else {
        (
            "segmented",
            include_str!("../shaders/_expert_qmm_tensor_kernel.metal"),
            2,
        )
    };
    Ok(array::metal_kernel(
        &format!(
            "mlxl3_rs_expert_qmm_{variant}_{input}_{output}_{experts}_{k}_{}_o{tile_offset}_v1",
            cb as u32
        ),
        &["xhat", "trellis", "block_table", "block_count"],
        &["yhat"],
        &header,
        source,
        &[
            xhat,
            &trellis.reshape(&[-1])?.view(Dtype::UInt32)?,
            &plan.table,
            &plan.count,
        ],
        &[vec![plan.slots, output]],
        &[Dtype::Float16],
        [(output / 32) * 32, plan.max_blocks, subblocks],
        [32, 1, 1],
    )?
    .remove(0))
}

/// Decode each expert weight fragment once for both 32-row subblocks of a
/// 64-row segment (bit-identical to the two-subblock kernel). On by default;
/// `MLXL3_EXPERT_QMM_SHARED=0` restores the two-subblock kernel.
fn expert_qmm_shared_decode() -> bool {
    static SHARED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    match EXPERT_QMM_SHARED_OVERRIDE.load(std::sync::atomic::Ordering::Relaxed) {
        0 => false,
        1 => true,
        _ => *SHARED.get_or_init(|| std::env::var("MLXL3_EXPERT_QMM_SHARED").as_deref() != Ok("0")),
    }
}

static EXPERT_QMM_SHARED_OVERRIDE: std::sync::atomic::AtomicU8 =
    std::sync::atomic::AtomicU8::new(2);

/// Test hook: force the shared-decode expert kernel on/off (`None` = env).
#[cfg(test)]
pub(crate) fn set_expert_qmm_shared(value: Option<bool>) {
    EXPERT_QMM_SHARED_OVERRIDE.store(
        value.map_or(2, u8::from),
        std::sync::atomic::Ordering::Relaxed,
    );
}

fn glu_down_input(
    gate_up_raw: &Array,
    gate_up_output_scales: &Array,
    down_input_scales: &Array,
    selected: &Array,
    activation: GluActivation,
    logical_hidden: i32,
    lane_order: bool,
) -> Result<Array> {
    let [rows, hidden]: [i32; 2] = gate_up_raw
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("gate/up output must have rank 2"))?;
    let slots = selected
        .shape()
        .iter()
        .try_fold(1i32, |n, &d| n.checked_mul(d))
        .context("MoE slot count overflow")?;
    ensure!(
        slots > 0 && rows == slots * 2 && hidden > 0 && hidden % 128 == 0,
        "invalid fused SwiGLU dimensions"
    );
    ensure!(
        selected.dtype() == Dtype::UInt32
            && gate_up_output_scales.shape().len() == 3
            && gate_up_output_scales.shape()[1..] == [2, hidden]
            && down_input_scales.shape() == [gate_up_output_scales.shape()[0], hidden],
        "invalid fused SwiGLU scales/routes"
    );
    ensure!(
        logical_hidden > 0 && logical_hidden <= hidden,
        "invalid logical GLU width"
    );
    let output_column = if lane_order { LANE_STORE } else { "column" };
    let gelu = matches!(activation, GluActivation::Gelu);
    let gate = butterfly("gate_a", "gate_b", gelu);
    let up = butterfly("up_a", "up_b", gelu);
    let down = butterfly("down_a", "down_b", gelu);
    let gate_finish = if gelu {
        "float(half(float(half(gate_b[tid])) * HAD_SCALE))"
    } else {
        "gate_b[tid] * HAD_SCALE"
    };
    let up_finish = if gelu {
        "float(half(float(half(up_b[tid])) * HAD_SCALE))"
    } else {
        "up_b[tid] * HAD_SCALE"
    };
    let (activated, tag) = activation.metal();
    let down_finish = if gelu {
        "float(half(down_b[tid]))"
    } else {
        "down_b[tid]"
    };
    let source = format!(
        r#"
            constexpr uint HIDDEN = {hidden}u;
            constexpr float HAD_SCALE = 0.08838834764831845f;
            uint tid = thread_position_in_threadgroup.x;
            uint group = threadgroup_position_in_grid.x;
            uint blocks = HIDDEN / 128u;
            uint slot = group / blocks;
            uint block = group - slot * blocks;
            uint column = block * 128u + tid;
            uint expert = uint(selected[slot]);

            threadgroup float gate_a[128];
            threadgroup float gate_b[128];
            threadgroup float up_a[128];
            threadgroup float up_b[128];
            threadgroup float down_a[128];
            threadgroup float down_b[128];

            gate_a[tid] = float(half(ygu[(slot * 2u) * HIDDEN + column]));
            up_a[tid] = float(half(ygu[(slot * 2u + 1u) * HIDDEN + column]));
            {gate}
            {up}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float gate_value = float(half(
                {gate_finish}
                * float(gu_svh[(expert * 2u) * HIDDEN + column])
            ));
            float up_value = float(half(
                {up_finish}
                * float(gu_svh[(expert * 2u + 1u) * HIDDEN + column])
            ));
            float activated = {activated};
            if (column >= {logical_hidden}u) activated = 0.0f;
            down_a[tid] = float(half(
                activated * float(down_suh[expert * HIDDEN + column])
            ));
            {down}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            xhat[slot * HIDDEN + {output_column}] = half({down_finish} * HAD_SCALE);
        "#
    );
    Ok(array::metal_kernel(
        &format!("mlxl3_rs_moe_glu_down_h{hidden}_{tag}_l{logical_hidden}_v2"),
        &["ygu", "gu_svh", "down_suh", "selected"],
        &["xhat"],
        "",
        &source,
        &[
            gate_up_raw,
            gate_up_output_scales,
            down_input_scales,
            selected,
        ],
        &[vec![slots, hidden]],
        &[Dtype::Float16],
        [
            slots
                .checked_mul(hidden / 128)
                .and_then(|n| n.checked_mul(128))
                .context("SwiGLU grid overflow")?,
            1,
            1,
        ],
        [128, 1, 1],
    )?
    .remove(0))
}

pub fn swiglu_down_input(
    gate_up_raw: &Array,
    gate_up_output_scales: &Array,
    down_input_scales: &Array,
    selected: &Array,
) -> Result<Array> {
    let hidden = *gate_up_raw
        .shape()
        .last()
        .context("gate/up output must have dimensions")?;
    glu_down_input(
        gate_up_raw,
        gate_up_output_scales,
        down_input_scales,
        selected,
        GluActivation::Silu,
        hidden,
        false,
    )
}

pub fn finish_and_reduce(raw: &Array, output_scales: &Array, scores: &Array) -> Result<Array> {
    let [slots, width]: [i32; 2] = raw
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("expert output must have rank 2"))?;
    let [rows, top_k]: [i32; 2] = scores
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("expert scores must have rank 2"))?;
    ensure!(
        width > 0
            && width % 128 == 0
            && slots == rows * top_k
            && output_scales.shape() == [slots, width],
        "invalid expert reduction dimensions"
    );
    raw.astype(Dtype::Float16)?
        .reshape(&[slots, width / 128, 128])?
        .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
        .reshape(&[rows, top_k, width])?
        .mul(&output_scales.reshape(&[rows, top_k, width])?)?
        .mul(&scores.reshape(&[rows, top_k, 1])?)?
        .sum(1, false)
}

pub struct Exl3SwitchGlu {
    gu_trellis: Array,
    gu_suh: Array,
    gu_svh: Array,
    down_trellis: Array,
    down_suh: Array,
    down_svh: Array,
    experts: i32,
    input: i32,
    hidden: i32,
    top_k: i32,
    /// Bits per weight of the gate/up projections (one trellis, so always equal).
    k: usize,
    /// Bits per weight of the down projections; equal to `k` for single-width checkpoints.
    down_k: usize,
    cb: Codebook,
    activation: GluActivation,
    logical_hidden: i32,
}

impl Exl3SwitchGlu {
    pub fn from_checkpoint(
        checkpoint: &Checkpoint,
        prefix: &str,
        experts: i32,
        top_k: i32,
    ) -> Result<Self> {
        Self::from_checkpoint_names(
            checkpoint,
            prefix,
            experts,
            top_k,
            ["gate_proj", "up_proj", "down_proj"],
            GluActivation::Silu,
            None,
        )
    }

    pub fn from_lfm_checkpoint(
        checkpoint: &Checkpoint,
        prefix: &str,
        experts: i32,
        top_k: i32,
    ) -> Result<Self> {
        Self::from_checkpoint_names(
            checkpoint,
            prefix,
            experts,
            top_k,
            ["w1", "w3", "w2"],
            GluActivation::Silu,
            None,
        )
    }

    pub fn from_gemma_checkpoint(
        checkpoint: &Checkpoint,
        prefix: &str,
        experts: i32,
        top_k: i32,
        logical_hidden: i32,
    ) -> Result<Self> {
        Self::from_checkpoint_names(
            checkpoint,
            prefix,
            experts,
            top_k,
            ["gate_proj", "up_proj", "down_proj"],
            GluActivation::Gelu,
            Some(logical_hidden),
        )
    }

    /// Inkling 2 routed experts: `{prefix}.{e}.{gate,up,down}_proj`, softcapped SwiGLU.
    pub fn from_inkling_checkpoint(
        checkpoint: &Checkpoint,
        prefix: &str,
        experts: i32,
        top_k: i32,
        gate_cap: f32,
        up_cap: f32,
    ) -> Result<Self> {
        ensure!(
            gate_cap.is_finite() && gate_cap > 0. && up_cap.is_finite() && up_cap > 0.,
            "invalid Inkling activation softcaps"
        );
        Self::from_checkpoint_names(
            checkpoint,
            prefix,
            experts,
            top_k,
            ["gate_proj", "up_proj", "down_proj"],
            GluActivation::SoftcapSilu { gate_cap, up_cap },
            None,
        )
    }

    fn from_checkpoint_names(
        checkpoint: &Checkpoint,
        prefix: &str,
        experts: i32,
        top_k: i32,
        [gate, up, down]: [&str; 3],
        activation: GluActivation,
        logical_hidden: Option<i32>,
    ) -> Result<Self> {
        ensure!(
            experts > 0
                && top_k > 0
                && top_k <= experts
                && experts as usize <= checkpoint.tensors.len() / 3,
            "invalid expert count or incomplete expert bank"
        );
        let names = |projection: &str| {
            (0..experts)
                .map(|expert| format!("{prefix}.{expert}.{projection}"))
                .collect::<Vec<_>>()
        };
        let gates = names(gate);
        let ups = names(up);
        let downs = names(down);
        let mut k = None;
        let mut down_k = None;
        let mut cb = None;
        for (name, is_down) in gates
            .iter()
            .chain(&ups)
            .map(|name| (name, false))
            .chain(downs.iter().map(|name| (name, true)))
        {
            let trellis = checkpoint
                .tensors
                .get(&format!("{name}.trellis"))
                .with_context(|| format!("missing expert trellis {name}"))?;
            ensure!(
                matches!(trellis.dtype.as_str(), "I16" | "U16"),
                "{name}: expert trellis must contain 16-bit integers"
            );
            let bits = *trellis.shape.last().context("empty expert trellis")? / 16;
            // gate and up share one trellis: one width; down may use its own width
            let width = if is_down { &mut down_k } else { &mut k };
            ensure!(
                width.is_none_or(|value| value == bits),
                "{}",
                if is_down {
                    "mixed down-projection bit widths"
                } else {
                    "mixed gate/up bit widths"
                }
            );
            *width = Some(bits);
            let mode = if checkpoint.tensors.contains_key(&format!("{name}.mul1")) {
                Codebook::Mul1
            } else if checkpoint.tensors.contains_key(&format!("{name}.mcg")) {
                Codebook::Mcg
            } else {
                Codebook::Default
            };
            ensure!(
                cb.is_none_or(|value| value == mode),
                "mixed expert codebooks"
            );
            cb = Some(mode);
        }
        let scale_name = |name: &str, primary: &str, legacy: &str| {
            let key = format!("{name}.{primary}");
            if checkpoint.tensors.contains_key(&key) {
                key
            } else {
                format!("{name}.{legacy}")
            }
        };
        let workers = load_workers()?;
        let trellises = |names: &[String]| -> Result<Vec<Array>> {
            let keys = names
                .iter()
                .map(|name| format!("{name}.trellis"))
                .collect::<Vec<_>>();
            // Signed and unsigned trellis words carry bits, never numeric values.
            checkpoint_arrays(checkpoint, &keys, workers)?
                .into_iter()
                .map(|array| array.view(Dtype::UInt16))
                .collect()
        };
        let mut gu_trellises = trellises(&gates)?;
        gu_trellises.extend(trellises(&ups)?);
        let gu_trellis = Array::concatenate(&gu_trellises.iter().collect::<Vec<_>>(), 1)?;
        let down_trellises = trellises(&downs)?;
        let down_trellis = Array::concatenate(&down_trellises.iter().collect::<Vec<_>>(), 1)?;
        let paired = |left: &[String], right: &[String], primary, legacy| -> Result<Array> {
            let keys = left
                .iter()
                .zip(right)
                .flat_map(|(a, b)| {
                    [
                        scale_name(a, primary, legacy),
                        scale_name(b, primary, legacy),
                    ]
                })
                .collect::<Vec<_>>();
            let scales = checkpoint_arrays(checkpoint, &keys, workers)?;
            let mut rows = Vec::with_capacity(left.len());
            for pair in scales.as_chunks::<2>().0 {
                rows.push(Array::concatenate(&[&pair[0], &pair[1]], 0)?);
            }
            let joined = Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0)?;
            let width = joined.shape()[0] / (experts * 2);
            joined.reshape(&[experts, 2, width])
        };
        let stacked = |names: &[String], primary, legacy| -> Result<Array> {
            let keys = names
                .iter()
                .map(|name| scale_name(name, primary, legacy))
                .collect::<Vec<_>>();
            let scales = checkpoint_arrays(checkpoint, &keys, workers)?;
            let rows = scales
                .iter()
                .map(|scale| scale.reshape(&[1, -1]))
                .collect::<Result<Vec<_>>>()?;
            Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0)
        };
        Self::new_with_activation(
            gu_trellis,
            paired(&gates, &ups, "suh", "su")?,
            paired(&gates, &ups, "svh", "sv")?,
            down_trellis,
            stacked(&downs, "suh", "su")?,
            stacked(&downs, "svh", "sv")?,
            top_k,
            k.context("checkpoint has no experts")?,
            down_k.context("checkpoint has no down projections")?,
            cb.context("checkpoint has no expert codebook")?,
            activation,
            logical_hidden,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        gu_trellis: Array,
        gu_suh: Array,
        gu_svh: Array,
        down_trellis: Array,
        down_suh: Array,
        down_svh: Array,
        top_k: i32,
        k: usize,
        cb: Codebook,
    ) -> Result<Self> {
        Self::new_with_activation(
            gu_trellis,
            gu_suh,
            gu_svh,
            down_trellis,
            down_suh,
            down_svh,
            top_k,
            k,
            k,
            cb,
            GluActivation::Silu,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_activation(
        gu_trellis: Array,
        gu_suh: Array,
        gu_svh: Array,
        down_trellis: Array,
        down_suh: Array,
        down_svh: Array,
        top_k: i32,
        k: usize,
        down_k: usize,
        cb: Codebook,
        activation: GluActivation,
        logical_hidden: Option<i32>,
    ) -> Result<Self> {
        let [experts, projections, input]: [i32; 3] = gu_suh
            .shape()
            .try_into()
            .map_err(|_| anyhow::anyhow!("gate/up input scales must have rank 3"))?;
        let [scale_experts, scale_projections, hidden]: [i32; 3] = gu_svh
            .shape()
            .try_into()
            .map_err(|_| anyhow::anyhow!("gate/up output scales must have rank 3"))?;
        let [gu_tiles, down_tiles, packed] =
            crate::contracts::switch_glu_layout(experts, top_k, input, hidden, k)
                .context("invalid or overflowing SwitchGLU dimensions")?;
        let [_, _, down_packed] =
            crate::contracts::switch_glu_layout(experts, top_k, input, hidden, down_k)
                .context("invalid down-projection bit width")?;
        ensure!(
            projections == 2 && scale_experts == experts && scale_projections == 2,
            "invalid SwitchGLU dimensions"
        );
        ensure!(
            down_suh.shape() == [experts, hidden]
                && down_svh.shape() == [experts, input]
                && gu_trellis.shape() == [input / 16, gu_tiles, packed]
                && down_trellis.shape() == [hidden / 16, down_tiles, down_packed],
            "SwitchGLU trellis/scales disagree"
        );
        ensure!(
            [&gu_trellis, &down_trellis]
                .iter()
                .all(|a| matches!(a.dtype(), Dtype::Int16 | Dtype::UInt16)),
            "expert trellis must contain 16-bit integers"
        );
        for value in [
            &gu_trellis,
            &gu_suh,
            &gu_svh,
            &down_trellis,
            &down_suh,
            &down_svh,
        ] {
            value.eval()?;
        }
        Ok(Self {
            gu_trellis,
            gu_suh,
            gu_svh,
            down_trellis,
            down_suh,
            down_svh,
            experts,
            input,
            hidden,
            top_k,
            k,
            down_k,
            cb,
            activation,
            logical_hidden: logical_hidden.unwrap_or(hidden),
        })
    }

    pub fn forward(&self, x: &Array, selected: &Array, scores: &Array) -> Result<Array> {
        let [rows, input]: [i32; 2] = x
            .shape()
            .try_into()
            .map_err(|_| anyhow::anyhow!("SwitchGLU input must have rank 2"))?;
        ensure!(
            rows > 0
                && input == self.input
                && selected.shape() == [rows, self.top_k]
                && scores.shape() == [rows, self.top_k],
            "invalid SwitchGLU input/routes"
        );
        let slots = rows.checked_mul(self.top_k).context("MoE slot overflow")?;
        let selected = selected.reshape(&[slots])?;
        if matches!(self.activation, GluActivation::Gelu) {
            return self.forward_gelu(x, &selected, scores, rows);
        }
        if rows >= 64
            && self.experts <= 256
            && self.k != 7
            && self.down_k != 7
            && array::supports_exl3_tensor_ops()?
        {
            return self.forward_segmented(x, &selected, scores, rows);
        }
        // the Q36 lane layout is tuned for one width shared by every projection
        let lane_order = self.k == self.down_k
            && crate::contracts::q36_expert_lane_shape(
                [rows, self.input, self.hidden, self.experts, self.top_k],
                self.k,
                self.cb == Codebook::Mcg,
            );
        // INVARIANT-2026-11-24-INK2M-L3-2: Inkling verifications of 2..=8 rows, each expert read once for its rows
        // (the same bits as the per-route kernel)
        if !lane_order
            && (2..=8).contains(&rows)
            && matches!(self.activation, GluActivation::SoftcapSilu { .. })
            && self.k != 7
            && self.down_k != 7
            && expert_panel_enabled()
            && crate::inkling2::invariant_call(rows)
        {
            return self.forward_panel(x, &selected, scores, rows);
        }
        // DFLASH2-2026-11-24-INK2M-P4-2: Inkling calls of 3..=8 rows with the shared experts in TensorOps (opt-in;
        // 2 rows measured slower)
        if !lane_order
            && (3..=8).contains(&rows)
            && matches!(self.activation, GluActivation::SoftcapSilu { .. })
            && self.k != 7
            && self.down_k != 7
            && expert_tensor_enabled()
            && array::supports_exl3_tensor_ops()?
        {
            return self.forward_tensor_hybrid(x, &selected, scores, rows);
        }
        let x_gu = routed_inputs_small(
            x,
            &self.gu_suh,
            &selected,
            rows,
            self.top_k,
            self.input,
            lane_order,
        )?;
        let gate_up = crate::linear::prepared_expert_raw(
            &x_gu,
            &self.gu_trellis,
            &selected,
            self.hidden,
            2,
            self.experts * self.hidden / 16,
            self.k,
            self.cb,
            lane_order,
        )?;
        let down_input = glu_down_input(
            &gate_up,
            &self.gu_svh,
            &self.down_suh,
            &selected,
            self.activation,
            self.logical_hidden,
            lane_order,
        )?;
        let down = crate::linear::prepared_expert_raw(
            &down_input,
            &self.down_trellis,
            &selected,
            self.input,
            1,
            0,
            self.down_k,
            self.cb,
            lane_order,
        )?;
        finish_and_reduce(&down, &self.down_svh.take(&selected, 0)?, scores)
    }

    /// L3-3: `forward` with the experts chosen by several rows read once for those rows (grouped on the GPU by
    /// member count, one launch per count; the other routes on the per-route kernel; inputs in lane order): every
    /// route has the bits of the per-route kernel.
    fn forward_panel(
        &self,
        x: &Array,
        selected: &Array,
        scores: &Array,
        rows: i32,
    ) -> Result<Array> {
        let plan = crate::linear::expert_panel_plan(selected, self.top_k)?;
        let x_gu = routed_inputs_small(
            x,
            &self.gu_suh,
            selected,
            rows,
            self.top_k,
            self.input,
            true,
        )?;
        let gate_up = crate::linear::prepared_expert_plan_raw(
            &x_gu,
            &self.gu_trellis,
            selected,
            &plan,
            self.hidden,
            2,
            self.experts * self.hidden / 16,
            self.k,
            self.cb,
        )?;
        let down_input = glu_down_input(
            &gate_up,
            &self.gu_svh,
            &self.down_suh,
            selected,
            self.activation,
            self.logical_hidden,
            true,
        )?;
        let down = crate::linear::prepared_expert_plan_raw(
            &down_input,
            &self.down_trellis,
            selected,
            &plan,
            self.input,
            1,
            0,
            self.down_k,
            self.cb,
        )?;
        finish_and_reduce(&down, &self.down_svh.take(selected, 0)?, scores)
    }

    /// P4-2 (not bit-exact): `forward` with the experts chosen by >= 2 routes of the call computed by one grouped
    /// TensorOps tile each and the other routes by the current kernel (their bits unchanged); the inputs, the
    /// SwiGLU step and the reduction are the current ones.
    fn forward_tensor_hybrid(
        &self,
        x: &Array,
        selected: &Array,
        scores: &Array,
        rows: i32,
    ) -> Result<Array> {
        let routes = rows * self.top_k;
        let (tile_info, members, shared) = crate::linear::expert_tensor_groups(selected, 2)?;
        let x_gu = routed_inputs_small(
            x,
            &self.gu_suh,
            selected,
            rows,
            self.top_k,
            self.input,
            false,
        )?;
        let stride = self.experts * self.hidden / 16;
        let gate_up = crate::linear::prepared_expert_raw_skip(
            &x_gu,
            &self.gu_trellis,
            selected,
            &shared,
            self.hidden,
            2,
            stride,
            self.k,
            self.cb,
        )?
        .add(&crate::linear::expert_tensor_raw(
            &x_gu,
            &self.gu_trellis,
            &tile_info,
            &members,
            routes,
            self.hidden,
            2,
            stride,
            self.k,
            self.cb,
        )?)?;
        let down_input = glu_down_input(
            &gate_up,
            &self.gu_svh,
            &self.down_suh,
            selected,
            self.activation,
            self.logical_hidden,
            false,
        )?;
        let down = crate::linear::prepared_expert_raw_skip(
            &down_input,
            &self.down_trellis,
            selected,
            &shared,
            self.input,
            1,
            0,
            self.down_k,
            self.cb,
        )?
        .add(&crate::linear::expert_tensor_raw(
            &down_input,
            &self.down_trellis,
            &tile_info,
            &members,
            routes,
            self.input,
            1,
            0,
            self.down_k,
            self.cb,
        )?)?;
        finish_and_reduce(&down, &self.down_svh.take(selected, 0)?, scores)
    }

    pub(crate) fn q36_final_row_bank(&self) -> bool {
        self.k == 2
            && self.down_k == 2
            && self.cb == Codebook::Mcg
            && self.input == 2048
            && self.hidden == 512
            && self.logical_hidden == 512
            && self.experts == 256
            && self.top_k == 8
            && matches!(self.activation, GluActivation::Silu)
    }

    // Same BM32 TensorOps accumulation as a full prefill. Each chosen expert
    // gets one real row; the existing plan supplies its padded input tile.
    pub(crate) fn forward_segmented_last(
        &self,
        x: &Array,
        selected: &Array,
        scores: &Array,
    ) -> Result<Array> {
        ensure!(
            x.shape() == [1, self.input]
                && selected.shape() == [1, self.top_k]
                && scores.shape() == [1, self.top_k],
            "invalid final MoE row"
        );
        self.forward_segmented(x, &selected.reshape(&[self.top_k])?, scores, 1)
    }

    fn forward_segmented(
        &self,
        x: &Array,
        selected: &Array,
        scores: &Array,
        rows: i32,
    ) -> Result<Array> {
        let parallel = self.input == 2048
            && self.hidden == 512
            && self.logical_hidden == 512
            && self.cb == Codebook::Mcg
            && matches!(self.k, 2 | 3)
            && self.down_k == self.k
            && matches!(self.activation, GluActivation::Silu)
            && array::is_m5_gpu()?;
        #[cfg(test)]
        let parallel = parallel && !crate::linear::q36_reference();
        let plan = route_plan(selected, rows, self.top_k, self.experts, parallel)?;
        let prepared = routed_inputs(x, &self.gu_suh, &plan, self.input)?;
        let gate = expert_segmented_qmm(
            &prepared
                .slice(0, 0, 1)?
                .reshape(&[plan.padded, self.input])?,
            &self.gu_trellis,
            &plan,
            self.input,
            self.hidden,
            self.experts,
            0,
            self.k,
            self.cb,
        )?;
        let up = expert_segmented_qmm(
            &prepared
                .slice(0, 1, 2)?
                .reshape(&[plan.padded, self.input])?,
            &self.gu_trellis,
            &plan,
            self.input,
            self.hidden,
            self.experts,
            self.experts * self.hidden / 16,
            self.k,
            self.cb,
        )?;
        let sorted = plan.experts.slice(0, 0, plan.slots)?;
        let gate_up = Array::concatenate(
            &[
                &gate.reshape(&[plan.slots, 1, self.hidden])?,
                &up.reshape(&[plan.slots, 1, self.hidden])?,
            ],
            1,
        )?
        .reshape(&[plan.slots * 2, self.hidden])?;
        let down_input = glu_down_input(
            &gate_up,
            &self.gu_svh,
            &self.down_suh,
            &sorted,
            self.activation,
            self.logical_hidden,
            false,
        )?;
        let down = expert_segmented_qmm(
            &Array::concatenate(
                &[
                    &down_input,
                    &Array::zeros_dtype(&[64, self.hidden], Dtype::Float16)?,
                ],
                0,
            )?,
            &self.down_trellis,
            &plan,
            self.hidden,
            self.input,
            self.experts,
            0,
            self.down_k,
            self.cb,
        )?;
        finish_and_reduce(
            &down.take(&plan.inverse, 0)?,
            &self.down_svh.take(selected, 0)?,
            scores,
        )
    }

    fn forward_gelu(
        &self,
        x: &Array,
        selected: &Array,
        scores: &Array,
        rows: i32,
    ) -> Result<Array> {
        let slots = rows.checked_mul(self.top_k).context("MoE slot overflow")?;
        let gate_up = expert_mapped(
            &x.reshape(&[rows, 1, 1, self.input])?
                .broadcast_to(&[rows, self.top_k, 2, self.input])?
                .reshape(&[slots * 2, self.input])?,
            &self.gu_trellis,
            Some(
                &self
                    .gu_suh
                    .take(selected, 0)?
                    .reshape(&[slots * 2, self.input])?,
            ),
            Some(
                &self
                    .gu_svh
                    .take(selected, 0)?
                    .reshape(&[slots * 2, self.hidden])?,
            ),
            selected,
            self.hidden,
            2,
            self.experts * self.hidden / 16,
            self.k,
            self.cb,
            false,
            false,
        )?
        .reshape(&[slots, 2, self.hidden])?;
        let gate = gate_up.slice(1, 0, 1)?.reshape(&[slots, self.hidden])?;
        let up = gate_up.slice(1, 1, 2)?.reshape(&[slots, self.hidden])?;
        let mut hidden = gate.geglu(&up)?;
        if self.logical_hidden < self.hidden {
            hidden = Array::concatenate(
                &[
                    &hidden.slice(1, 0, self.logical_hidden)?,
                    &Array::zeros_dtype(
                        &[slots, self.hidden - self.logical_hidden],
                        hidden.dtype(),
                    )?,
                ],
                1,
            )?;
        }
        expert_mapped(
            &hidden,
            &self.down_trellis,
            Some(&self.down_suh.take(selected, 0)?),
            Some(&self.down_svh.take(selected, 0)?),
            selected,
            self.input,
            1,
            0,
            self.down_k,
            self.cb,
            false,
            false,
        )?
        .reshape(&[rows, self.top_k, self.input])?
        .mul(&scores.reshape(&[rows, self.top_k, 1])?)?
        .sum(1, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    /// INVARIANT-2026-11-25-INK2M-L3-3: the expert panel runs unless turned off (`MLXL3_INKLING_EXPERT_PANEL=0`).
    #[test]
    fn expert_panel_is_on_unless_zero() {
        assert!(panel_setting(None));
        assert!(!panel_setting(Some("0")));
        assert!(panel_setting(Some("")));
        assert!(panel_setting(Some("true")));
        assert!(panel_setting(Some("1")));
    }

    /// DFLASH2-2026-11-24-INK2M-P4-1 (bench): cost of an expert shared by several rows on the real layer-12 banks.
    /// (1) Calls of T = 2 / 4 / 8 rows x 8 routes (`distinct`, `real` routes captured from a verification of 8 real
    /// tokens, `same`): raw gate/up and down products, current kernel (A) against the grouped exact kernel (G, NT
    /// 1 / 2 / 4), G bit-identical to A. (2) One expert, M = 1..8 rows: full gate and down projections, current
    /// kernel against the transposed TensorOps kernel (numeric gap recorded). Times per call: 16 independent calls
    /// per evaluation, median of 15 after 3 warm-ups.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn inkling_expert_sharing_bench() -> Result<()> {
        use std::time::Instant;
        const CALLS: usize = 16;
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        // real routes: 8 consecutive real tokens verified after a real 64-token prefix
        let real_calls: Vec<Vec<u32>> = {
            let rows: serde_json::Value = serde_json::from_reader(std::fs::File::open(
                std::env::var("MLXL3_INKLING_ROWS")?,
            )?)?;
            let tokens: Vec<u32> = serde_json::from_value(rows["rows"][0]["tokens"].clone())?;
            ensure!(tokens.len() >= 72, "row too short");
            let mut model = crate::inkling2::Inkling2::load(&dir)?;
            let taps = vec![5, 23, 41, model.visit_count()];
            model.forward_tokens(&tokens[..64])?.eval()?;
            crate::inkling2::MOE_ROUTES.with(|c| *c.borrow_mut() = Some(Vec::new()));
            let verified = model.verify(&tokens[64..72], &taps);
            let captured = crate::inkling2::MOE_ROUTES
                .with(|c| c.borrow_mut().take())
                .context("routes")?;
            verified?.0.eval()?;
            ensure!(
                captured.len() == 48,
                "{} MoE calls captured",
                captured.len()
            );
            [6usize, 18, 30, 42]
                .iter()
                .map(|&i| {
                    ensure!(
                        captured[i].0 == 8 && captured[i].1.len() == 64,
                        "captured call shape"
                    );
                    Ok(captured[i].1.clone())
                })
                .collect::<Result<_>>()?
        };
        crate::array::clear_cache()?;
        let checkpoint = crate::checkpoint::inspect(&dir)?;
        let (experts, top_k) = (128i32, 8i32);
        let moe = Exl3SwitchGlu::from_inkling_checkpoint(
            &checkpoint,
            "model.llm.layers.12.mlp.experts",
            experts,
            top_k,
            4.0,
            25.0,
        )?;
        ensure!(
            moe.hidden == 1152 && moe.input == 2304,
            "unexpected Inkling expert shapes"
        );
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let mut random = |rows: i32, cols: i32| -> Result<Array> {
            let v: Vec<f32> = (0..rows * cols)
                .map(|_| ((next() % 2001) as f32 / 1000.0 - 1.0) * 0.5)
                .collect();
            let a = Array::from_f32(&v, &[rows, cols])?.astype(Dtype::Float16)?;
            a.eval()?;
            Ok(a)
        };
        // per-call time of `f` over CALLS independent inputs evaluated together
        let time = |f: &dyn Fn(usize) -> Result<Array>| -> Result<f64> {
            let mut samples = Vec::new();
            for rep in 0..18 {
                let outs = (0..CALLS).map(f).collect::<Result<Vec<_>>>()?;
                let start = Instant::now();
                Array::eval_many(&outs.iter().collect::<Vec<_>>())?;
                if rep >= 3 {
                    samples.push(start.elapsed().as_secs_f64() * 1e3 / CALLS as f64);
                }
            }
            samples.sort_by(f64::total_cmp);
            Ok(samples[samples.len() / 2])
        };
        let bits = |a: &Array| -> Result<Vec<u32>> {
            Ok(a.astype(Dtype::Float32)?
                .to_f32()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        let gu_stride = moe.experts * moe.hidden / 16;
        let mut report = Vec::new();
        // (1) production-shaped calls
        let mut patterns: Vec<(String, usize, Vec<u32>)> = Vec::new();
        for t in [2usize, 4, 8] {
            let mut distinct = Vec::new();
            let mut pick = 0x2545_f491_4f6c_dd1du64.wrapping_add(t as u64);
            while distinct.len() < t * 8 {
                pick = pick
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let e = (pick >> 33) as u32 % experts as u32;
                if !distinct.contains(&e) {
                    distinct.push(e);
                }
            }
            patterns.push((format!("distinct{t}"), t, distinct));
            for (i, call) in real_calls.iter().enumerate() {
                patterns.push((format!("real{t}_{i}"), t, call[..t * 8].to_vec()));
            }
            let first = real_calls[0][..8].to_vec();
            let same: Vec<u32> = (0..t)
                .flat_map(|r| (0..8).map(move |i| (r, i)))
                .map(|(r, i)| first[(i + r) % 8])
                .collect();
            patterns.push((format!("same{t}"), t, same));
        }
        for (name, t, ids) in &patterns {
            let routes = ids.len() as i32;
            let distinct = ids.iter().collect::<std::collections::BTreeSet<_>>().len();
            let xs_gu = (0..CALLS)
                .map(|_| random(routes * 2, moe.input))
                .collect::<Result<Vec<_>>>()?;
            let xs_dn = (0..CALLS)
                .map(|_| random(routes, moe.hidden))
                .collect::<Result<Vec<_>>>()?;
            let selected = Array::from_u32(ids, &[routes])?;
            let a_gu = |i: usize| {
                crate::linear::prepared_expert_raw(
                    &xs_gu[i],
                    &moe.gu_trellis,
                    &selected,
                    moe.hidden,
                    2,
                    gu_stride,
                    moe.k,
                    moe.cb,
                    false,
                )
            };
            let a_dn = |i: usize| {
                crate::linear::prepared_expert_raw(
                    &xs_dn[i],
                    &moe.down_trellis,
                    &selected,
                    moe.input,
                    1,
                    0,
                    moe.down_k,
                    moe.cb,
                    false,
                )
            };
            let g_gu = |i: usize| {
                crate::linear::prepared_expert_grouped_raw(
                    &xs_gu[i],
                    &moe.gu_trellis,
                    ids,
                    experts,
                    moe.hidden,
                    2,
                    gu_stride,
                    moe.k,
                    moe.cb,
                )
            };
            let g_dn = |i: usize| {
                crate::linear::prepared_expert_grouped_raw(
                    &xs_dn[i],
                    &moe.down_trellis,
                    ids,
                    experts,
                    moe.input,
                    1,
                    0,
                    moe.down_k,
                    moe.cb,
                )
            };
            // exactness before speed: G against A on two inputs, for every NT
            for nt in [None, Some(1), Some(2), Some(4)] {
                for i in [0usize, CALLS - 1] {
                    let (ra, rg) = (
                        bits(&a_gu(i)?)?,
                        crate::linear::with_expert_nt(nt, || bits(&g_gu(i)?))?,
                    );
                    ensure!(
                        ra.len() == (routes * 2 * moe.hidden) as usize && ra == rg,
                        "{name}: grouped gate/up differs (NT {nt:?})"
                    );
                    let (ra, rg) = (
                        bits(&a_dn(i)?)?,
                        crate::linear::with_expert_nt(nt, || bits(&g_dn(i)?))?,
                    );
                    ensure!(
                        ra.len() == (routes * moe.input) as usize && ra == rg,
                        "{name}: grouped down differs (NT {nt:?})"
                    );
                }
            }
            let a1 = (time(&a_gu)?, time(&a_dn)?);
            let mut grouped = Vec::new();
            for nt in [1, 2, 4] {
                grouped.push((
                    nt,
                    crate::linear::with_expert_nt(Some(nt), || time(&g_gu))?,
                    crate::linear::with_expert_nt(Some(nt), || time(&g_dn))?,
                ));
            }
            let a2 = (time(&a_gu)?, time(&a_dn)?);
            let current = (a1.0 + a2.0) / 2.0 + (a1.1 + a2.1) / 2.0;
            let best = grouped
                .iter()
                .map(|&(_, gu, dn)| gu + dn)
                .fold(f64::INFINITY, f64::min);
            println!(
                "P4BENCH {name}: rows {t} routes {routes} distinct {distinct} | A gate/up {:.4}/{:.4} down {:.4}/{:.4} ms | G (nt, gate/up, down) {:?} | best G / A {:.3}",
                a1.0,
                a2.0,
                a1.1,
                a2.1,
                grouped,
                best / current
            );
            report.push(serde_json::json!({
                "case": name, "rows": t, "routes": routes, "distinct": distinct,
                "a_gate_up_ms": [a1.0, a2.0], "a_down_ms": [a1.1, a2.1],
                "grouped": grouped.iter().map(|&(nt, gu, dn)| serde_json::json!({"nt": nt, "gate_up_ms": gu, "down_ms": dn})).collect::<Vec<_>>(),
                "best_grouped_over_current": best / current, "bit_identical": true,
            }));
        }
        // (2) one expert, M rows: full projections, current kernel against the transposed TensorOps kernel
        let tensor_ops = crate::array::supports_exl3_tensor_ops()?;
        let e = real_calls[0][0];
        let ei = e as i32;
        let gate_suh = moe
            .gu_suh
            .slice(0, ei, ei + 1)?
            .slice(1, 0, 1)?
            .reshape(&[1, moe.input])?;
        let gate_svh = moe
            .gu_svh
            .slice(0, ei, ei + 1)?
            .slice(1, 0, 1)?
            .reshape(&[1, moe.hidden])?;
        let down_suh = moe
            .down_suh
            .slice(0, ei, ei + 1)?
            .reshape(&[1, moe.hidden])?;
        let down_svh = moe
            .down_svh
            .slice(0, ei, ei + 1)?
            .reshape(&[1, moe.input])?;
        let (lanes_gu, lanes_dn) = (
            crate::linear::lane_source_indices(moe.input)?,
            crate::linear::lane_source_indices(moe.hidden)?,
        );
        Array::eval_many(&[
            &gate_suh, &gate_svh, &down_suh, &down_svh, &lanes_gu, &lanes_dn,
        ])?;
        for m in 1..=8i32 {
            let xs_g = (0..CALLS)
                .map(|_| random(m, moe.input))
                .collect::<Result<Vec<_>>>()?;
            let xs_d = (0..CALLS)
                .map(|_| random(m, moe.hidden))
                .collect::<Result<Vec<_>>>()?;
            let selected = Array::from_u32(&vec![e; m as usize], &[m])?;
            let rep = |a: &Array, cols: i32| a.broadcast_to(&[m, cols])?.astype(Dtype::Float16);
            let (gs, gv, ds, dv) = (
                rep(&gate_suh, moe.input)?,
                rep(&gate_svh, moe.hidden)?,
                rep(&down_suh, moe.hidden)?,
                rep(&down_svh, moe.input)?,
            );
            Array::eval_many(&[&gs, &gv, &ds, &dv])?;
            let cur_g = |i: usize| {
                crate::linear::expert_mapped(
                    &xs_g[i],
                    &moe.gu_trellis,
                    Some(&gs),
                    Some(&gv),
                    &selected,
                    moe.hidden,
                    1,
                    0,
                    moe.k,
                    moe.cb,
                    false,
                    false,
                )
            };
            let cur_d = |i: usize| {
                crate::linear::expert_mapped(
                    &xs_d[i],
                    &moe.down_trellis,
                    Some(&ds),
                    Some(&dv),
                    &selected,
                    moe.input,
                    1,
                    0,
                    moe.down_k,
                    moe.cb,
                    false,
                    false,
                )
            };
            let tiles_gu = moe.gu_trellis.shape()[1];
            let tiles_dn = moe.down_trellis.shape()[1];
            let ten_g = |i: usize| {
                crate::linear::qmm_tensor_lane(
                    &xs_g[i],
                    &moe.gu_trellis,
                    &gate_suh.reshape(&[moe.input])?,
                    &gate_svh.reshape(&[moe.hidden])?,
                    &lanes_gu,
                    moe.input,
                    moe.hidden,
                    moe.k,
                    moe.cb,
                    tiles_gu,
                    ei * (moe.hidden / 16),
                )
            };
            let ten_d = |i: usize| {
                crate::linear::qmm_tensor_lane(
                    &xs_d[i],
                    &moe.down_trellis,
                    &down_suh.reshape(&[moe.hidden])?,
                    &down_svh.reshape(&[moe.input])?,
                    &lanes_dn,
                    moe.hidden,
                    moe.input,
                    moe.down_k,
                    moe.cb,
                    tiles_dn,
                    ei * (moe.input / 16),
                )
            };
            let c1 = (time(&cur_g)?, time(&cur_d)?);
            let tensor = if tensor_ops {
                // numeric gap against the current kernel (F16 words that differ, largest absolute error)
                let gap = |a: &Array, b: &Array| -> Result<(usize, f32, usize, bool)> {
                    let (a, b) = (
                        a.astype(Dtype::Float32)?.to_f32()?,
                        b.astype(Dtype::Float32)?.to_f32()?,
                    );
                    ensure!(a.len() == b.len(), "TensorOps output length");
                    let differ = a
                        .iter()
                        .zip(&b)
                        .filter(|(x, y)| f16::from_f32(**x) != f16::from_f32(**y))
                        .count();
                    let max = a
                        .iter()
                        .zip(&b)
                        .map(|(x, y)| (x - y).abs())
                        .fold(0f32, f32::max);
                    Ok((differ, max, a.len(), b.iter().all(|v| v.is_finite())))
                };
                let gg = gap(&cur_g(0)?, &ten_g(0)?)?;
                let gd = gap(&cur_d(0)?, &ten_d(0)?)?;
                Some((time(&ten_g)?, time(&ten_d)?, gg, gd))
            } else {
                None
            };
            let c2 = (time(&cur_g)?, time(&cur_d)?);
            println!(
                "P4BENCH expert m {m}: current gate {:.4}/{:.4} down {:.4}/{:.4} ms | TensorOps {:?}",
                c1.0, c2.0, c1.1, c2.1, tensor
            );
            report.push(serde_json::json!({
                "case": format!("expert_m{m}"), "rows": m,
                "current_gate_ms": [c1.0, c2.0], "current_down_ms": [c1.1, c2.1],
                "tensor": tensor.map(|(g, d, gg, gd)| serde_json::json!({
                    "gate_ms": g, "down_ms": d,
                    "gate_gap": {"f16_differ": gg.0, "max_abs": gg.1, "values": gg.2, "finite": gg.3},
                    "down_gap": {"f16_differ": gd.0, "max_abs": gd.1, "values": gd.2, "finite": gd.3},
                })),
            }));
        }
        println!("P4BENCH_JSON {}", serde_json::to_string(&report)?);
        Ok(())
    }

    /// INVARIANT-2026-11-24-INK2M-L3-2 oracle: the GPU plan against a host recount (chunks of up to 4 member routes
    /// per expert in first-route order), and the raw gate/up and down products of the plan mode against the
    /// per-route kernel bit for bit, T = 2..=8, routes `distinct`, `same` and pseudo-random draws with repeats; a
    /// mutated plan (last member of full chunks dropped) is detected.
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL"]
    fn inkling_expert_panel_matches_per_route() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let checkpoint = crate::checkpoint::inspect(&dir)?;
        let moe = Exl3SwitchGlu::from_inkling_checkpoint(
            &checkpoint,
            "model.llm.layers.12.mlp.experts",
            128,
            8,
            4.0,
            25.0,
        )?;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut uniform = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let bits = |a: &Array| -> Result<Vec<u32>> {
            Ok(a.astype(Dtype::Float32)?
                .to_f32()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        let stride = moe.experts * moe.hidden / 16;
        let mut cases = 0usize;
        let mut shared_routes = 0usize;
        let mut mutation_detected = false;
        for t in 2..=8usize {
            let mut patterns: Vec<Vec<u32>> = Vec::new();
            // distinct experts
            let mut d = Vec::new();
            while d.len() < t * 8 {
                let e = uniform() % 128;
                if !d.contains(&e) {
                    d.push(e);
                }
            }
            patterns.push(d);
            // every row on the same 8 experts
            let first: Vec<u32> = patterns[0][..8].to_vec();
            patterns.push(
                (0..t)
                    .flat_map(|r| (0..8).map(move |i| (r, i)))
                    .map(|(r, i)| first[(i + r) % 8])
                    .collect(),
            );
            // rows drawing from a small pool (repeats across rows, distinct within a row)
            for pool in [12u32, 24, 48] {
                let mut ids = Vec::new();
                for _ in 0..t {
                    let mut row = Vec::new();
                    while row.len() < 8 {
                        let e = uniform() % pool;
                        if !row.contains(&e) {
                            row.push(e);
                        }
                    }
                    ids.extend(row);
                }
                patterns.push(ids);
            }
            for ids in &patterns {
                let rows = t as i32;
                let routes = rows * 8;
                let selected = Array::from_u32(ids, &[routes])?;
                // host recount of the plan
                let plan = crate::linear::expert_panel_plan(&selected, 8)?;
                let mut lists: [Vec<u32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
                let mut shared = vec![0u32; ids.len()];
                for (q, e) in ids.iter().enumerate() {
                    if ids[..q].contains(e) {
                        continue;
                    }
                    let members: Vec<u32> = (q as u32..ids.len() as u32)
                        .filter(|&u| ids[u as usize] == *e)
                        .collect();
                    let mut at = 0;
                    while members.len() - at >= 2 {
                        let take = (members.len() - at).min(4);
                        lists[take - 2].push(take as u32);
                        lists[take - 2].extend(&members[at..at + take]);
                        for &m in &members[at..at + take] {
                            shared[m as usize] = 1;
                        }
                        at += take;
                    }
                }
                ensure!(plan.max_members == rows.min(4), "panel member bound");
                for (index, expected) in lists.iter_mut().enumerate() {
                    let m = index + 2;
                    expected.resize(ids.len() / m * (m + 1), 0);
                    let got = plan.lists[index].to_u32()?;
                    ensure!(
                        got == *expected,
                        "chunks of {m} differ from the host recount (T {t}): {got:?} vs {expected:?}"
                    );
                }
                ensure!(
                    plan.shared.to_u32()? == shared,
                    "shared flags differ from the host recount (T {t})"
                );
                shared_routes += shared.iter().sum::<u32>() as usize;
                let quads = !lists[2].iter().all(|&v| v == 0);
                // raw products
                let v: Vec<f32> = (0..rows * moe.input)
                    .map(|_| ((uniform() % 2001) as f32 / 1000.0 - 1.0) * 0.5)
                    .collect();
                let x = Array::from_f32(&v, &[rows, moe.input])?.astype(Dtype::Float16)?;
                let natural =
                    routed_inputs_small(&x, &moe.gu_suh, &selected, rows, 8, moe.input, false)?;
                let lane =
                    routed_inputs_small(&x, &moe.gu_suh, &selected, rows, 8, moe.input, true)?;
                let gu = crate::linear::prepared_expert_raw(
                    &natural,
                    &moe.gu_trellis,
                    &selected,
                    moe.hidden,
                    2,
                    stride,
                    moe.k,
                    moe.cb,
                    false,
                )?;
                let gu_plan = crate::linear::prepared_expert_plan_raw(
                    &lane,
                    &moe.gu_trellis,
                    &selected,
                    &plan,
                    moe.hidden,
                    2,
                    stride,
                    moe.k,
                    moe.cb,
                )?;
                let (a, b) = (bits(&gu)?, bits(&gu_plan)?);
                ensure!(
                    a.len() == (routes * 2 * moe.hidden) as usize && a == b,
                    "T {t}: plan gate/up differs from the per-route kernel"
                );
                let dn_natural = glu_down_input(
                    &gu,
                    &moe.gu_svh,
                    &moe.down_suh,
                    &selected,
                    moe.activation,
                    moe.logical_hidden,
                    false,
                )?;
                let dn_lane = glu_down_input(
                    &gu,
                    &moe.gu_svh,
                    &moe.down_suh,
                    &selected,
                    moe.activation,
                    moe.logical_hidden,
                    true,
                )?;
                let dn = crate::linear::prepared_expert_raw(
                    &dn_natural,
                    &moe.down_trellis,
                    &selected,
                    moe.input,
                    1,
                    0,
                    moe.down_k,
                    moe.cb,
                    false,
                )?;
                let dn_plan = crate::linear::prepared_expert_plan_raw(
                    &dn_lane,
                    &moe.down_trellis,
                    &selected,
                    &plan,
                    moe.input,
                    1,
                    0,
                    moe.down_k,
                    moe.cb,
                )?;
                let (a, b) = (bits(&dn)?, bits(&dn_plan)?);
                ensure!(
                    a.len() == (routes * moe.input) as usize && a == b,
                    "T {t}: plan down differs from the per-route kernel"
                );
                cases += 1;
                if !mutation_detected && quads {
                    let mutated = crate::linear::with_expert_group_mutation(|| {
                        crate::linear::expert_panel_plan(&selected, 8)
                    })?;
                    let gu_m = crate::linear::prepared_expert_plan_raw(
                        &lane,
                        &moe.gu_trellis,
                        &selected,
                        &mutated,
                        moe.hidden,
                        2,
                        stride,
                        moe.k,
                        moe.cb,
                    )?;
                    mutation_detected = bits(&gu_m)? != bits(&gu)?;
                }
            }
        }
        ensure!(cases == 35, "{cases} cases");
        ensure!(mutation_detected, "the mutated plan is not detected");
        ensure!(
            shared_routes > 500,
            "only {shared_routes} shared routes exercised"
        );
        println!(
            "L3PANEL all {cases} cases: plan = host recount ({shared_routes} shared routes), products identical to the per-route kernel; mutation detected"
        );
        Ok(())
    }

    /// INVARIANT-2026-11-24-INK2M-L3-1 (bench): on the real layer-12 banks, T = 2 / 4 / 8 rows, routes `real`
    /// (captured from a verification of 8 real tokens), `distinct` and `same`: the current expert kernel in natural
    /// order (A) against the same kernel with the inputs in lane order (bits), and the grouped kernel (shared
    /// experts read once, up to 4 rows) in lane order at NT 1 / 2 / 4 against A (bits, then time per call of the raw
    /// gate/up + down products; 16 calls per evaluation, median of 15).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn inkling_expert_lane_grouped_bench() -> Result<()> {
        use std::time::Instant;
        const CALLS: usize = 16;
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let real_calls: Vec<Vec<u32>> = {
            let rows: serde_json::Value = serde_json::from_reader(std::fs::File::open(
                std::env::var("MLXL3_INKLING_ROWS")?,
            )?)?;
            let tokens: Vec<u32> = serde_json::from_value(rows["rows"][0]["tokens"].clone())?;
            let mut model = crate::inkling2::Inkling2::load(&dir)?;
            let taps = vec![5, 23, 41, model.visit_count()];
            model.forward_tokens(&tokens[..64])?.eval()?;
            crate::inkling2::MOE_ROUTES.with(|c| *c.borrow_mut() = Some(Vec::new()));
            let verified = model.verify(&tokens[64..72], &taps);
            let captured = crate::inkling2::MOE_ROUTES
                .with(|c| c.borrow_mut().take())
                .context("routes")?;
            verified?.0.eval()?;
            ensure!(
                captured.len() == 48,
                "{} MoE calls captured",
                captured.len()
            );
            [6usize, 18, 30, 42]
                .iter()
                .map(|&i| captured[i].1.clone())
                .collect()
        };
        crate::array::clear_cache()?;
        let checkpoint = crate::checkpoint::inspect(&dir)?;
        let moe = Exl3SwitchGlu::from_inkling_checkpoint(
            &checkpoint,
            "model.llm.layers.12.mlp.experts",
            128,
            8,
            4.0,
            25.0,
        )?;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut uniform = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let time = |f: &dyn Fn(usize) -> Result<Array>| -> Result<f64> {
            let mut samples = Vec::new();
            for rep in 0..18 {
                let outs = (0..CALLS).map(f).collect::<Result<Vec<_>>>()?;
                let start = Instant::now();
                Array::eval_many(&outs.iter().collect::<Vec<_>>())?;
                if rep >= 3 {
                    samples.push(start.elapsed().as_secs_f64() * 1e3 / CALLS as f64);
                }
            }
            samples.sort_by(f64::total_cmp);
            Ok(samples[samples.len() / 2])
        };
        let bits = |a: &Array| -> Result<Vec<u32>> {
            Ok(a.astype(Dtype::Float32)?
                .to_f32()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        let stride = moe.experts * moe.hidden / 16;
        let mut report = Vec::new();
        for t in [2usize, 4, 8] {
            let mut patterns: Vec<(String, Vec<u32>)> = real_calls
                .iter()
                .enumerate()
                .map(|(i, c)| (format!("real{t}_{i}"), c[..t * 8].to_vec()))
                .collect();
            let mut distinct = Vec::new();
            while distinct.len() < t * 8 {
                let e = uniform() % 128;
                if !distinct.contains(&e) {
                    distinct.push(e);
                }
            }
            patterns.push((format!("distinct{t}"), distinct));
            let first = real_calls[0][..8].to_vec();
            patterns.push((
                format!("same{t}"),
                (0..t)
                    .flat_map(|r| (0..8).map(move |i| (r, i)))
                    .map(|(r, i)| first[(i + r) % 8])
                    .collect(),
            ));
            for (name, ids) in &patterns {
                let rows = t as i32;
                let routes = rows * 8;
                let selected = Array::from_u32(ids, &[routes])?;
                let xs = (0..CALLS)
                    .map(|_| {
                        let v: Vec<f32> = (0..rows * moe.input)
                            .map(|_| ((uniform() % 2001) as f32 / 1000.0 - 1.0) * 0.5)
                            .collect();
                        let x = Array::from_f32(&v, &[rows, moe.input])?.astype(Dtype::Float16)?;
                        let natural = routed_inputs_small(
                            &x,
                            &moe.gu_suh,
                            &selected,
                            rows,
                            8,
                            moe.input,
                            false,
                        )?;
                        let lane = routed_inputs_small(
                            &x,
                            &moe.gu_suh,
                            &selected,
                            rows,
                            8,
                            moe.input,
                            true,
                        )?;
                        let gu = crate::linear::prepared_expert_raw(
                            &natural,
                            &moe.gu_trellis,
                            &selected,
                            moe.hidden,
                            2,
                            stride,
                            moe.k,
                            moe.cb,
                            false,
                        )?;
                        let dn_natural = glu_down_input(
                            &gu,
                            &moe.gu_svh,
                            &moe.down_suh,
                            &selected,
                            moe.activation,
                            moe.logical_hidden,
                            false,
                        )?;
                        let dn_lane = glu_down_input(
                            &gu,
                            &moe.gu_svh,
                            &moe.down_suh,
                            &selected,
                            moe.activation,
                            moe.logical_hidden,
                            true,
                        )?;
                        Array::eval_many(&[&natural, &lane, &dn_natural, &dn_lane])?;
                        Ok((natural, lane, dn_natural, dn_lane))
                    })
                    .collect::<Result<Vec<_>>>()?;
                let a = |i: usize| -> Result<Array> {
                    let (n, _, dn, _) = &xs[i];
                    let gu = crate::linear::prepared_expert_raw(
                        n,
                        &moe.gu_trellis,
                        &selected,
                        moe.hidden,
                        2,
                        stride,
                        moe.k,
                        moe.cb,
                        false,
                    )?;
                    let down = crate::linear::prepared_expert_raw(
                        dn,
                        &moe.down_trellis,
                        &selected,
                        moe.input,
                        1,
                        0,
                        moe.down_k,
                        moe.cb,
                        false,
                    )?;
                    Array::concatenate(&[&gu.reshape(&[-1])?, &down.reshape(&[-1])?], 0)
                };
                let a_lane = |i: usize| -> Result<Array> {
                    let (_, l, _, dl) = &xs[i];
                    let gu = crate::linear::prepared_expert_raw(
                        l,
                        &moe.gu_trellis,
                        &selected,
                        moe.hidden,
                        2,
                        stride,
                        moe.k,
                        moe.cb,
                        true,
                    )?;
                    let down = crate::linear::prepared_expert_raw(
                        dl,
                        &moe.down_trellis,
                        &selected,
                        moe.input,
                        1,
                        0,
                        moe.down_k,
                        moe.cb,
                        true,
                    )?;
                    Array::concatenate(&[&gu.reshape(&[-1])?, &down.reshape(&[-1])?], 0)
                };
                let g_lane = |i: usize| -> Result<Array> {
                    let (_, l, _, dl) = &xs[i];
                    let gu = crate::linear::prepared_expert_grouped_raw_lane(
                        l,
                        &moe.gu_trellis,
                        ids,
                        128,
                        moe.hidden,
                        2,
                        stride,
                        moe.k,
                        moe.cb,
                        true,
                    )?;
                    let down = crate::linear::prepared_expert_grouped_raw_lane(
                        dl,
                        &moe.down_trellis,
                        ids,
                        128,
                        moe.input,
                        1,
                        0,
                        moe.down_k,
                        moe.cb,
                        true,
                    )?;
                    Array::concatenate(&[&gu.reshape(&[-1])?, &down.reshape(&[-1])?], 0)
                };
                let reference = bits(&a(0)?)?;
                ensure!(!reference.is_empty(), "{name}: empty output");
                let lane_same = bits(&a_lane(0)?)? == reference;
                let mut grouped_same = Vec::new();
                for nt in [1, 2, 4] {
                    grouped_same.push(crate::linear::with_expert_nt(
                        Some(nt),
                        || -> Result<bool> { Ok(bits(&g_lane(0)?)? == reference) },
                    )?);
                }
                let ta = time(&a)?;
                let tl = time(&a_lane)?;
                let mut tg = Vec::new();
                for nt in [1, 2, 4] {
                    tg.push((
                        nt,
                        crate::linear::with_expert_nt(Some(nt), || time(&g_lane))?,
                    ));
                }
                let ta2 = time(&a)?;
                let base = (ta + ta2) / 2.0;
                let best = tg.iter().map(|&(_, v)| v).fold(f64::INFINITY, f64::min);
                println!(
                    "L3LANE {name}: lane order identical {lane_same}, grouped identical {grouped_same:?} | A {ta:.4}/{ta2:.4} A-lane {tl:.4} G-lane (nt, ms) {tg:?} | best G-lane / A {:.3}",
                    best / base
                );
                ensure!(lane_same, "{name}: the lane order changes the bits");
                ensure!(
                    grouped_same.iter().all(|&v| v),
                    "{name}: the grouped kernel changes the bits"
                );
                report.push(serde_json::json!({"case": name, "a_ms": [ta, ta2], "a_lane_ms": tl,
                    "g_lane": tg.iter().map(|&(nt, v)| serde_json::json!({"nt": nt, "ms": v})).collect::<Vec<_>>(),
                    "best_over_a": best / base}));
            }
        }
        println!("L3LANE_JSON {}", serde_json::to_string(&report)?);
        Ok(())
    }

    /// DFLASH2-2026-11-24-INK2M-P4-2 oracle and call cost on the real layer-12 banks, T = 2..=8 rows, routes `real`
    /// (captured from a verification of 8 real tokens), `distinct` and `same`: the grouping against a host recount;
    /// raw gate/up and down products of the hybrid path against the current kernel: rows of unshared routes equal
    /// bit for bit, shared rows within 2e-3 relative L2 and finite, skipped rows zero; the whole `forward` with
    /// the option on against off (relative L2) and its time per call; a mutated grouping (last member dropped) is
    /// detected.
    #[test]
    #[ignore = "requires an Apple M5 GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn inkling_expert_tensor_hybrid_matches_exact() -> Result<()> {
        use std::time::Instant;
        const CALLS: usize = 16;
        ensure!(
            crate::array::supports_exl3_tensor_ops()?,
            "TensorOps unavailable"
        );
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let real_calls: Vec<Vec<u32>> = {
            let rows: serde_json::Value = serde_json::from_reader(std::fs::File::open(
                std::env::var("MLXL3_INKLING_ROWS")?,
            )?)?;
            let tokens: Vec<u32> = serde_json::from_value(rows["rows"][3]["tokens"].clone())?;
            ensure!(tokens.len() >= 72, "row too short");
            let mut model = crate::inkling2::Inkling2::load(&dir)?;
            let taps = vec![5, 23, 41, model.visit_count()];
            model.forward_tokens(&tokens[..64])?.eval()?;
            crate::inkling2::MOE_ROUTES.with(|c| *c.borrow_mut() = Some(Vec::new()));
            let verified = model.verify(&tokens[64..72], &taps);
            let captured = crate::inkling2::MOE_ROUTES
                .with(|c| c.borrow_mut().take())
                .context("routes")?;
            verified?.0.eval()?;
            ensure!(
                captured.len() == 48,
                "{} MoE calls captured",
                captured.len()
            );
            [3usize, 21, 37]
                .iter()
                .map(|&i| captured[i].1.clone())
                .collect()
        };
        crate::array::clear_cache()?;
        let checkpoint = crate::checkpoint::inspect(&dir)?;
        let moe = Exl3SwitchGlu::from_inkling_checkpoint(
            &checkpoint,
            "model.llm.layers.12.mlp.experts",
            128,
            8,
            4.0,
            25.0,
        )?;
        let mut state = 0x5851_f42d_4c95_7f2du64;
        let mut uniform = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let bits = |a: &Array| -> Result<Vec<f32>> { a.astype(Dtype::Float32)?.to_f32() };
        let relative = |a: &[f32], b: &[f32]| -> f64 {
            let num: f64 = a.iter().zip(b).map(|(x, y)| f64::from(x - y).powi(2)).sum();
            let den: f64 = b.iter().map(|y| f64::from(*y).powi(2)).sum();
            (num / den.max(f64::MIN_POSITIVE)).sqrt()
        };
        let time = |f: &dyn Fn(usize) -> Result<Array>| -> Result<f64> {
            let mut samples = Vec::new();
            for rep in 0..18 {
                let outs = (0..CALLS).map(f).collect::<Result<Vec<_>>>()?;
                let start = Instant::now();
                Array::eval_many(&outs.iter().collect::<Vec<_>>())?;
                if rep >= 3 {
                    samples.push(start.elapsed().as_secs_f64() * 1e3 / CALLS as f64);
                }
            }
            samples.sort_by(f64::total_cmp);
            Ok(samples[samples.len() / 2])
        };
        let mut report = Vec::new();
        let mut worst_shared = 0f64;
        for t in 2..=8usize {
            let mut patterns: Vec<(String, Vec<u32>)> = real_calls
                .iter()
                .enumerate()
                .map(|(i, c)| (format!("real{t}_{i}"), c[..t * 8].to_vec()))
                .collect();
            let mut distinct = Vec::new();
            while distinct.len() < t * 8 {
                let e = uniform() % 128;
                if !distinct.contains(&e) {
                    distinct.push(e);
                }
            }
            patterns.push((format!("distinct{t}"), distinct));
            let first = real_calls[0][..8].to_vec();
            patterns.push((
                format!("same{t}"),
                (0..t)
                    .flat_map(|r| (0..8).map(move |i| (r, i)))
                    .map(|(r, i)| first[(i + r) % 8])
                    .collect(),
            ));
            for (name, ids) in &patterns {
                let rows = t as i32;
                let routes = rows * 8;
                let selected = Array::from_u32(ids, &[routes])?;
                // the grouping against a host recount
                let (tile_info, members, shared) =
                    crate::linear::expert_tensor_groups(&selected, 2)?;
                let (info, mem, flags) = (tile_info.to_u32()?, members.to_u32()?, shared.to_u32()?);
                let mut expected_tiles = Vec::new();
                for (r, e) in ids.iter().enumerate() {
                    if ids[..r].contains(e) {
                        continue;
                    }
                    let list: Vec<u32> = (0..ids.len() as u32)
                        .filter(|&q| ids[q as usize] == *e)
                        .collect();
                    if list.len() >= 2 {
                        expected_tiles.push((*e, list));
                    }
                }
                for (q, flag) in flags.iter().enumerate() {
                    let n = ids.iter().filter(|&&e| e == ids[q]).count();
                    ensure!(
                        *flag == u32::from(n >= 2),
                        "{name}: shared flag of route {q}"
                    );
                }
                let tokens = crate::linear::EXPERT_TENSOR_TOKENS as usize;
                for (tile, (e, list)) in expected_tiles.iter().enumerate() {
                    ensure!(
                        info[2 * tile] == *e && info[2 * tile + 1] as usize == list.len(),
                        "{name}: tile {tile}"
                    );
                    ensure!(
                        mem[tile * tokens..tile * tokens + list.len()] == list[..],
                        "{name}: members of tile {tile}"
                    );
                }
                for tile in expected_tiles.len()..info.len() / 2 {
                    ensure!(
                        info[2 * tile + 1] == 0,
                        "{name}: unused tile {tile} is live"
                    );
                }
                let shared_routes = flags.iter().filter(|&&f| f == 1).count();
                // raw products
                let x: Vec<f32> = (0..rows * moe.input)
                    .map(|_| ((uniform() % 2001) as f32 / 1000.0 - 1.0) * 0.5)
                    .collect();
                let x = Array::from_f32(&x, &[rows, moe.input])?.astype(Dtype::Float16)?;
                let x_gu =
                    routed_inputs_small(&x, &moe.gu_suh, &selected, rows, 8, moe.input, false)?;
                let stride = moe.experts * moe.hidden / 16;
                let exact_gu = crate::linear::prepared_expert_raw(
                    &x_gu,
                    &moe.gu_trellis,
                    &selected,
                    moe.hidden,
                    2,
                    stride,
                    moe.k,
                    moe.cb,
                    false,
                )?;
                let skip_gu = crate::linear::prepared_expert_raw_skip(
                    &x_gu,
                    &moe.gu_trellis,
                    &selected,
                    &shared,
                    moe.hidden,
                    2,
                    stride,
                    moe.k,
                    moe.cb,
                )?;
                let tens_gu = crate::linear::expert_tensor_raw(
                    &x_gu,
                    &moe.gu_trellis,
                    &tile_info,
                    &members,
                    routes,
                    moe.hidden,
                    2,
                    stride,
                    moe.k,
                    moe.cb,
                )?;
                let down_input = glu_down_input(
                    &exact_gu,
                    &moe.gu_svh,
                    &moe.down_suh,
                    &selected,
                    moe.activation,
                    moe.logical_hidden,
                    false,
                )?;
                let exact_dn = crate::linear::prepared_expert_raw(
                    &down_input,
                    &moe.down_trellis,
                    &selected,
                    moe.input,
                    1,
                    0,
                    moe.down_k,
                    moe.cb,
                    false,
                )?;
                let skip_dn = crate::linear::prepared_expert_raw_skip(
                    &down_input,
                    &moe.down_trellis,
                    &selected,
                    &shared,
                    moe.input,
                    1,
                    0,
                    moe.down_k,
                    moe.cb,
                )?;
                let tens_dn = crate::linear::expert_tensor_raw(
                    &down_input,
                    &moe.down_trellis,
                    &tile_info,
                    &members,
                    routes,
                    moe.input,
                    1,
                    0,
                    moe.down_k,
                    moe.cb,
                )?;
                let mut worst = 0f64;
                for (label, exact, skip, tens, width, projections) in [
                    (
                        "gate_up",
                        &exact_gu,
                        &skip_gu,
                        &tens_gu,
                        moe.hidden as usize,
                        2usize,
                    ),
                    (
                        "down",
                        &exact_dn,
                        &skip_dn,
                        &tens_dn,
                        moe.input as usize,
                        1usize,
                    ),
                ] {
                    let (e, k, g) = (bits(exact)?, bits(skip)?, bits(tens)?);
                    ensure!(
                        e.len() == routes as usize * projections * width
                            && k.len() == e.len()
                            && g.len() == e.len(),
                        "{name} {label}: output sizes"
                    );
                    for row in 0..routes as usize * projections {
                        let span = row * width..(row + 1) * width;
                        let route = row / projections;
                        if flags[route] == 0 {
                            ensure!(
                                k[span.clone()]
                                    .iter()
                                    .zip(&e[span.clone()])
                                    .all(|(a, b)| a.to_bits() == b.to_bits())
                                    && g[span.clone()].iter().all(|&v| v == 0.0),
                                "{name} {label}: unshared row {row} changed"
                            );
                        } else {
                            ensure!(
                                k[span.clone()].iter().all(|&v| v == 0.0),
                                "{name} {label}: skipped row {row} not zero"
                            );
                            ensure!(
                                g[span.clone()].iter().all(|v| v.is_finite()),
                                "{name} {label}: row {row} not finite"
                            );
                            let rel = relative(&g[span.clone()], &e[span.clone()]);
                            ensure!(
                                rel <= 2e-3,
                                "{name} {label}: shared row {row} relative error {rel:.2e}"
                            );
                            worst = worst.max(rel);
                        }
                    }
                }
                worst_shared = worst_shared.max(worst);
                // whole forward, option off and on, and its time
                let scores: Vec<f32> = (0..routes)
                    .map(|_| (uniform() % 1000) as f32 / 4000.0 + 0.01)
                    .collect();
                let scores = Array::from_f32(&scores, &[rows, 8])?;
                let route_ids = selected.reshape(&[rows, 8])?;
                let off =
                    with_expert_tensor(false, || bits(&moe.forward(&x, &route_ids, &scores)?))?;
                let on = with_expert_tensor(true, || bits(&moe.forward(&x, &route_ids, &scores)?))?;
                ensure!(
                    on.len() == off.len() && on.iter().all(|v| v.is_finite()),
                    "{name}: forward outputs"
                );
                let forward_rel = relative(&on, &off);
                ensure!(
                    forward_rel <= 5e-3,
                    "{name}: forward relative error {forward_rel:.2e}"
                );
                let xs = (0..CALLS)
                    .map(|_| {
                        let v: Vec<f32> = (0..rows * moe.input)
                            .map(|_| ((uniform() % 2001) as f32 / 1000.0 - 1.0) * 0.5)
                            .collect();
                        let a = Array::from_f32(&v, &[rows, moe.input])?.astype(Dtype::Float16)?;
                        a.eval()?;
                        Ok(a)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let run = |i: usize| moe.forward(&xs[i], &route_ids, &scores);
                let a1 = with_expert_tensor(false, || time(&run))?;
                let h1 = with_expert_tensor(true, || time(&run))?;
                let h2 = with_expert_tensor(true, || time(&run))?;
                let a2 = with_expert_tensor(false, || time(&run))?;
                println!(
                    "P4HYBRID {name}: rows {t} shared routes {shared_routes}/{routes} tiles {} | worst shared row {worst:.2e} forward {forward_rel:.2e} | exact {a1:.4}/{a2:.4} hybrid {h1:.4}/{h2:.4} ms ({:.3})",
                    expected_tiles.len(),
                    (h1 + h2) / (a1 + a2)
                );
                report.push(serde_json::json!({
                    "case": name, "rows": t, "shared_routes": shared_routes, "tiles": expected_tiles.len(),
                    "worst_shared_row": worst, "forward_rel": forward_rel,
                    "exact_ms": [a1, a2], "hybrid_ms": [h1, h2], "ratio": (h1 + h2) / (a1 + a2),
                }));
            }
        }
        // mutation: the last member of every tile dropped -> its row is never computed
        {
            let ids = real_calls[0].clone();
            let selected = Array::from_u32(&ids, &[64])?;
            let (tile_info, members, shared) = crate::linear::with_expert_group_mutation(|| {
                crate::linear::expert_tensor_groups(&selected, 2)
            })?;
            let x: Vec<f32> = (0..8 * moe.input)
                .map(|_| ((uniform() % 2001) as f32 / 1000.0 - 1.0) * 0.5)
                .collect();
            let x = Array::from_f32(&x, &[8, moe.input])?.astype(Dtype::Float16)?;
            let x_gu = routed_inputs_small(&x, &moe.gu_suh, &selected, 8, 8, moe.input, false)?;
            let stride = moe.experts * moe.hidden / 16;
            let exact = bits(&crate::linear::prepared_expert_raw(
                &x_gu,
                &moe.gu_trellis,
                &selected,
                moe.hidden,
                2,
                stride,
                moe.k,
                moe.cb,
                false,
            )?)?;
            let hybrid = bits(
                &crate::linear::prepared_expert_raw_skip(
                    &x_gu,
                    &moe.gu_trellis,
                    &selected,
                    &shared,
                    moe.hidden,
                    2,
                    stride,
                    moe.k,
                    moe.cb,
                )?
                .add(&crate::linear::expert_tensor_raw(
                    &x_gu,
                    &moe.gu_trellis,
                    &tile_info,
                    &members,
                    64,
                    moe.hidden,
                    2,
                    stride,
                    moe.k,
                    moe.cb,
                )?)?,
            )?;
            let width = moe.hidden as usize;
            let detected = (0..128).any(|row| {
                relative(
                    &hybrid[row * width..(row + 1) * width],
                    &exact[row * width..(row + 1) * width],
                ) > 2e-3
            });
            ensure!(detected, "the dropped tile member is not detected");
            println!("P4HYBRID mutation detected");
        }
        println!(
            "P4HYBRID_JSON {}",
            serde_json::json!({"worst_shared_row": worst_shared, "cases": report})
        );
        Ok(())
    }

    /// DFLASH2-2026-11-24-INK2M-P4-1b (bench): compute cost of the transposed TensorOps kernel at the scale of one
    /// MoE call: one `qmm_tensor_lane` launch over D adjacent experts of the real layer-12 banks (gate and up:
    /// D x 1152 outputs each; down: D x 2304), M = 2 / 4 / 8 rows, against the current raw expert kernel on calls
    /// of 8 M distinct routes (its cost barely depends on sharing, P4-1). Times per call: 16 independent calls per
    /// evaluation, median of 15 after 3 warm-ups; order current, TensorOps, current.
    #[test]
    #[ignore = "requires an Apple M5 GPU and MLXL3_INKLING_MODEL"]
    fn inkling_expert_tensor_call_bench() -> Result<()> {
        use std::time::Instant;
        const CALLS: usize = 16;
        ensure!(
            crate::array::supports_exl3_tensor_ops()?,
            "TensorOps unavailable"
        );
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let checkpoint = crate::checkpoint::inspect(&dir)?;
        let (experts, top_k) = (128i32, 8i32);
        let moe = Exl3SwitchGlu::from_inkling_checkpoint(
            &checkpoint,
            "model.llm.layers.12.mlp.experts",
            experts,
            top_k,
            4.0,
            25.0,
        )?;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut random = |rows: i32, cols: i32| -> Result<Array> {
            let v: Vec<f32> = (0..rows * cols)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (((state >> 33) as u32 % 2001) as f32 / 1000.0 - 1.0) * 0.5
                })
                .collect();
            let a = Array::from_f32(&v, &[rows, cols])?.astype(Dtype::Float16)?;
            a.eval()?;
            Ok(a)
        };
        let time = |f: &dyn Fn(usize) -> Result<Array>| -> Result<f64> {
            let mut samples = Vec::new();
            for rep in 0..18 {
                let outs = (0..CALLS).map(f).collect::<Result<Vec<_>>>()?;
                let start = Instant::now();
                Array::eval_many(&outs.iter().collect::<Vec<_>>())?;
                if rep >= 3 {
                    samples.push(start.elapsed().as_secs_f64() * 1e3 / CALLS as f64);
                }
            }
            samples.sort_by(f64::total_cmp);
            Ok(samples[samples.len() / 2])
        };
        let bits_of = |a: &Array| -> Result<Vec<u32>> {
            Ok(a.astype(Dtype::Float32)?
                .to_f32()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        let gu_stride = moe.experts * moe.hidden / 16;
        let (tiles_gu, tiles_dn) = (moe.gu_trellis.shape()[1], moe.down_trellis.shape()[1]);
        let lanes_gu = crate::linear::lane_source_indices(moe.input)?;
        let lanes_dn = crate::linear::lane_source_indices(moe.hidden)?;
        let gate_suh = moe
            .gu_suh
            .slice(0, 0, 1)?
            .slice(1, 0, 1)?
            .reshape(&[moe.input])?;
        let down_suh = moe.down_suh.slice(0, 0, 1)?.reshape(&[moe.hidden])?;
        Array::eval_many(&[&lanes_gu, &lanes_dn, &gate_suh, &down_suh])?;
        let mut report = Vec::new();
        for m in [2i32, 4, 8] {
            // current kernel on 8 m distinct routes
            let mut ids = Vec::new();
            let mut pick = 0x2545_f491_4f6c_dd1du64.wrapping_add(m as u64);
            while ids.len() < (m * 8) as usize {
                pick = pick
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let e = (pick >> 33) as u32 % experts as u32;
                if !ids.contains(&e) {
                    ids.push(e);
                }
            }
            let routes = m * 8;
            let selected = Array::from_u32(&ids, &[routes])?;
            let xs_gu = (0..CALLS)
                .map(|_| random(routes * 2, moe.input))
                .collect::<Result<Vec<_>>>()?;
            let xs_dn = (0..CALLS)
                .map(|_| random(routes, moe.hidden))
                .collect::<Result<Vec<_>>>()?;
            let a_gu = |i: usize| {
                crate::linear::prepared_expert_raw(
                    &xs_gu[i],
                    &moe.gu_trellis,
                    &selected,
                    moe.hidden,
                    2,
                    gu_stride,
                    moe.k,
                    moe.cb,
                    false,
                )
            };
            let a_dn = |i: usize| {
                crate::linear::prepared_expert_raw(
                    &xs_dn[i],
                    &moe.down_trellis,
                    &selected,
                    moe.input,
                    1,
                    0,
                    moe.down_k,
                    moe.cb,
                    false,
                )
            };
            let a1 = (time(&a_gu)?, time(&a_dn)?);
            let xs_t = (0..CALLS)
                .map(|_| random(m, moe.input))
                .collect::<Result<Vec<_>>>()?;
            let xs_td = (0..CALLS)
                .map(|_| random(m, moe.hidden))
                .collect::<Result<Vec<_>>>()?;
            let mut tensor = Vec::new();
            let mut single = Vec::new();
            for d in [8i32, 16, 26, 32, 39, 64] {
                if d > m * 8 {
                    continue;
                }
                let gate_svh = moe
                    .gu_svh
                    .slice(0, 0, d)?
                    .slice(1, 0, 1)?
                    .reshape(&[d * moe.hidden])?;
                let up_svh = moe
                    .gu_svh
                    .slice(0, 0, d)?
                    .slice(1, 1, 2)?
                    .reshape(&[d * moe.hidden])?;
                let down_svh = moe.down_svh.slice(0, 0, d)?.reshape(&[d * moe.input])?;
                Array::eval_many(&[&gate_svh, &up_svh, &down_svh])?;
                // gate and up of experts 0..d (adjacent in the bank), one launch each
                let t_gu = |i: usize| -> Result<Array> {
                    let gate = crate::linear::qmm_tensor_lane(
                        &xs_t[i],
                        &moe.gu_trellis,
                        &gate_suh,
                        &gate_svh,
                        &lanes_gu,
                        moe.input,
                        d * moe.hidden,
                        moe.k,
                        moe.cb,
                        tiles_gu,
                        0,
                    )?;
                    let up = crate::linear::qmm_tensor_lane(
                        &xs_t[i],
                        &moe.gu_trellis,
                        &gate_suh,
                        &up_svh,
                        &lanes_gu,
                        moe.input,
                        d * moe.hidden,
                        moe.k,
                        moe.cb,
                        tiles_gu,
                        experts * moe.hidden / 16,
                    )?;
                    Array::concatenate(&[&gate, &up], 1)
                };
                let t_dn = |i: usize| {
                    crate::linear::qmm_tensor_lane(
                        &xs_td[i],
                        &moe.down_trellis,
                        &down_suh,
                        &down_svh,
                        &lanes_dn,
                        moe.hidden,
                        d * moe.input,
                        moe.down_k,
                        moe.cb,
                        tiles_dn,
                        0,
                    )
                };
                let out = t_gu(0)?;
                ensure!(
                    out.shape() == [m, 2 * d * moe.hidden],
                    "TensorOps gate/up shape {:?}",
                    out.shape()
                );
                let dn = t_dn(0)?;
                ensure!(
                    dn.shape() == [m, d * moe.input],
                    "TensorOps down shape {:?}",
                    dn.shape()
                );
                ensure!(
                    out.astype(Dtype::Float32)?
                        .to_f32()?
                        .iter()
                        .all(|v| v.is_finite())
                        && dn
                            .astype(Dtype::Float32)?
                            .to_f32()?
                            .iter()
                            .all(|v| v.is_finite()),
                    "TensorOps outputs not finite"
                );
                // P4-1d: the same compute as gate + up of d experts in one launch (2 d adjacent gate experts)
                let gate2_svh = moe
                    .gu_svh
                    .slice(0, 0, 2 * d)?
                    .slice(1, 0, 1)?
                    .reshape(&[2 * d * moe.hidden])?;
                gate2_svh.eval()?;
                let t_g2 = |i: usize| {
                    crate::linear::qmm_tensor_lane(
                        &xs_t[i],
                        &moe.gu_trellis,
                        &gate_suh,
                        &gate2_svh,
                        &lanes_gu,
                        moe.input,
                        2 * d * moe.hidden,
                        moe.k,
                        moe.cb,
                        tiles_gu,
                        0,
                    )
                };
                if 2 * d <= experts {
                    // P4-1e: the K3 window decode gives the same outputs bit for bit, then its time
                    let plain = crate::linear::with_tensor_lane_k3(false, || bits_of(&t_g2(0)?))?;
                    let window = crate::linear::with_tensor_lane_k3(true, || bits_of(&t_g2(0)?))?;
                    ensure!(
                        plain.len() == (m * 2 * d * moe.hidden) as usize && plain == window,
                        "K3 window decode changes the TensorOps outputs (rows {m}, experts {d})"
                    );
                    single.push((
                        d,
                        crate::linear::with_tensor_lane_k3(false, || time(&t_g2))?,
                        crate::linear::with_tensor_lane_k3(true, || time(&t_g2))?,
                        crate::linear::with_tensor_lane_k3(false, || time(&t_g2))?,
                    ));
                }
                tensor.push((d, time(&t_gu)?, time(&t_dn)?));
            }
            let a2 = (time(&a_gu)?, time(&a_dn)?);
            let current = (a1.0 + a2.0) / 2.0 + (a1.1 + a2.1) / 2.0;
            println!(
                "P4SINGLE rows {m}: one launch over 2 d gate experts (d, ms generic, ms K3 window, ms generic) {single:?}; current gate/up per route {:.5} ms",
                (a1.0 + time(&a_gu)?) / 2.0 / f64::from(routes)
            );
            println!(
                "P4TENSOR rows {m}: current (routes {routes}) gate/up {:.4}/{:.4} down {:.4}/{:.4} ms | TensorOps (experts, gate/up, down) {:?} | ratios {:?}",
                a1.0,
                a2.0,
                a1.1,
                a2.1,
                tensor,
                tensor
                    .iter()
                    .map(|&(d, gu, dn)| (d, (gu + dn) / current))
                    .collect::<Vec<_>>()
            );
            report.push(serde_json::json!({
                "rows": m, "current_gate_up_ms": [a1.0, a2.0], "current_down_ms": [a1.1, a2.1],
                "tensor": tensor.iter().map(|&(d, gu, dn)| serde_json::json!({"experts": d, "gate_up_ms": gu, "down_ms": dn, "over_current": (gu + dn) / current})).collect::<Vec<_>>(),
            }));
        }
        println!("P4TENSOR_JSON {}", serde_json::to_string(&report)?);
        Ok(())
    }

    /// DSPARK-2026-11-23-INK2M-MTPLX-C0 (diagnostic): GPU wall time of each stage of
    /// `Exl3SwitchGlu::forward` on the real Inkling 2 Mini expert banks (layer 12 of `MLXL3_INKLING_MODEL`) for
    /// 1 to 4 rows with distinct random routes, each stage evaluated alone from evaluated inputs (median of
    /// 30 after 5 warm-up runs, rows ascending then descending), and the whole call.
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL"]
    fn inkling_moe_stage_cost() -> Result<()> {
        use std::time::Instant;
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let checkpoint = crate::checkpoint::inspect(&dir)?;
        let (experts, top_k) = (128i32, 8i32);
        let moe = Exl3SwitchGlu::from_inkling_checkpoint(
            &checkpoint,
            "model.llm.layers.12.mlp.experts",
            experts,
            top_k,
            4.0,
            25.0,
        )?;
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };
        let time = |f: &mut dyn FnMut() -> Result<Array>| -> Result<f64> {
            let mut samples = Vec::new();
            for rep in 0..35 {
                let start = Instant::now();
                f()?.eval()?;
                if rep >= 5 {
                    samples.push(start.elapsed().as_secs_f64() * 1e3);
                }
            }
            samples.sort_by(f64::total_cmp);
            Ok(samples[samples.len() / 2])
        };
        // `MLXL3_MOE_PATTERN=same`: every row chooses the same 8 experts (in a rotated order)
        let same = std::env::var("MLXL3_MOE_PATTERN").as_deref() == Ok("same");
        let mut report = std::collections::BTreeMap::<i32, Vec<Vec<f64>>>::new();
        for rows in [1i32, 2, 3, 4, 4, 3, 2, 1] {
            let mut ids = Vec::new();
            let mut first = Vec::new();
            for r in 0..rows {
                let mut row = Vec::new();
                if same && r > 0 {
                    row = (0..top_k as usize)
                        .map(|i| first[(i + r as usize) % top_k as usize])
                        .collect();
                }
                while row.len() < top_k as usize {
                    let e = next() % experts as u32;
                    if !row.contains(&e) {
                        row.push(e);
                    }
                }
                if r == 0 {
                    first = row.clone();
                }
                ids.extend(row);
            }
            let x: Vec<f32> = (0..rows * 2304)
                .map(|_| ((next() % 2001) as f32 / 1000.0 - 1.0) * 0.5)
                .collect();
            let x = Array::from_f32(&x, &[rows, 2304])?.astype(Dtype::Float16)?;
            let scores: Vec<f32> = (0..rows * top_k)
                .map(|_| (next() % 1000) as f32 / 4000.0 + 0.01)
                .collect();
            let scores = Array::from_f32(&scores, &[rows, top_k])?;
            let routes = Array::from_u32(&ids, &[rows, top_k])?;
            let selected = routes.reshape(&[rows * top_k])?;
            Array::eval_many(&[&x, &scores, &routes, &selected])?;
            let x_gu =
                routed_inputs_small(&x, &moe.gu_suh, &selected, rows, top_k, moe.input, false)?;
            let gate_up = crate::linear::prepared_expert_raw(
                &x_gu,
                &moe.gu_trellis,
                &selected,
                moe.hidden,
                2,
                moe.experts * moe.hidden / 16,
                moe.k,
                moe.cb,
                false,
            )?;
            let down_input = glu_down_input(
                &gate_up,
                &moe.gu_svh,
                &moe.down_suh,
                &selected,
                moe.activation,
                moe.logical_hidden,
                false,
            )?;
            let down = crate::linear::prepared_expert_raw(
                &down_input,
                &moe.down_trellis,
                &selected,
                moe.input,
                1,
                0,
                moe.down_k,
                moe.cb,
                false,
            )?;
            Array::eval_many(&[&x_gu, &gate_up, &down_input, &down])?;
            let svh = moe.down_svh.take(&selected, 0)?;
            svh.eval()?;
            let t = [
                time(&mut || {
                    routed_inputs_small(&x, &moe.gu_suh, &selected, rows, top_k, moe.input, false)
                })?,
                time(&mut || {
                    crate::linear::prepared_expert_raw(
                        &x_gu,
                        &moe.gu_trellis,
                        &selected,
                        moe.hidden,
                        2,
                        moe.experts * moe.hidden / 16,
                        moe.k,
                        moe.cb,
                        false,
                    )
                })?,
                time(&mut || {
                    glu_down_input(
                        &gate_up,
                        &moe.gu_svh,
                        &moe.down_suh,
                        &selected,
                        moe.activation,
                        moe.logical_hidden,
                        false,
                    )
                })?,
                time(&mut || {
                    crate::linear::prepared_expert_raw(
                        &down_input,
                        &moe.down_trellis,
                        &selected,
                        moe.input,
                        1,
                        0,
                        moe.down_k,
                        moe.cb,
                        false,
                    )
                })?,
                time(&mut || finish_and_reduce(&down, &svh, &scores))?,
                time(&mut || moe.forward(&x, &routes, &scores))?,
                // an empty evaluation: the fixed cost of one eval round trip
                time(&mut || x.try_clone())?,
            ];
            println!(
                "rows {rows}: inputs {:.3} gate_up {:.3} glu {:.3} down {:.3} reduce {:.3} | forward {:.3} | eval floor {:.3} ms",
                t[0], t[1], t[2], t[3], t[4], t[5], t[6]
            );
            report.entry(rows).or_default().push(t.to_vec());
        }
        println!("MOE_STAGE_COST {}", serde_json::to_string(&report)?);
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU; disposable signed/unsigned EXL3 checkpoint"]
    fn mixed_signed_trellises_preserve_checkpoint_bits() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let file = dir.path().join("experts.safetensors");
        let prefix = "model.layers.0.mlp.experts";
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut words = Vec::new();
        for (index, name) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
            let data = (0..2048)
                .map(|i| {
                    (i as u16)
                        .wrapping_mul(1777)
                        .wrapping_add(0x8001 + index as u16)
                })
                .collect::<Vec<_>>();
            let start = payload.len();
            payload.extend(data.iter().flat_map(|w| w.to_le_bytes()));
            header.insert(format!("{prefix}.0.{name}.trellis"), serde_json::json!({"dtype":if index==1 {"U16"} else {"I16"},"shape":[8,8,32],"data_offsets":[start,payload.len()]}));
            words.push(data);
            for scale in ["suh", "svh"] {
                let start = payload.len();
                payload.extend([f16::ONE.to_le_bytes(); 128].into_iter().flatten());
                header.insert(format!("{prefix}.0.{name}.{scale}"), serde_json::json!({"dtype":"F16","shape":[128],"data_offsets":[start,payload.len()]}));
            }
        }
        let json = serde_json::to_vec(&header)?;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend(json);
        bytes.extend(payload);
        std::fs::write(&file, bytes)?;
        let mut checkpoint = Checkpoint {
            path: dir.path().to_path_buf(),
            model_type: "qwen3_5_moe".into(),
            bits: None,
            size_bytes: 0,
            modules: Vec::new(),
            tensors: crate::checkpoint::read_header(&file)?,
        };
        let layer = Exl3SwitchGlu::from_checkpoint(&checkpoint, prefix, 1, 1)?;
        assert_eq!(layer.gu_trellis.dtype(), Dtype::UInt16);
        assert_eq!(layer.down_trellis.dtype(), Dtype::UInt16);
        let mut expected = Vec::new();
        for row in 0..8 {
            for bank in &words[..2] {
                expected.extend(
                    bank[row * 256..(row + 1) * 256]
                        .iter()
                        .flat_map(|w| w.to_le_bytes()),
                );
            }
        }
        assert_eq!(expected.len(), 8192);
        assert_eq!(layer.gu_trellis.to_bytes()?, expected);
        assert_eq!(
            layer.down_trellis.to_bytes()?,
            words[2]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>()
        );
        // Metadata rejection must happen before loading any tensor payload.
        assert!(Exl3SwitchGlu::from_checkpoint(&checkpoint, prefix, i32::MAX, 1).is_err());
        checkpoint
            .tensors
            .get_mut(&format!("{prefix}.0.gate_proj.trellis"))
            .unwrap()
            .dtype = "BOOL".into();
        let error = Exl3SwitchGlu::from_checkpoint(&checkpoint, prefix, 1, 1)
            .err()
            .context("accepted BOOL checkpoint trellis")?;
        assert!(error.to_string().contains("16-bit integers"));
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU; constructor admission, no invalid shader dispatch"]
    fn switch_glu_rejects_invalid_trellis_before_evaluation() -> Result<()> {
        let make = |gate_dtype, down_dtype, k| {
            Exl3SwitchGlu::new(
                Array::zeros_dtype(&[8, 32, 16 * k as i32], gate_dtype)?,
                Array::from_f16_bits(&[f16::ONE.to_bits(); 512], &[2, 2, 128])?,
                Array::from_f16_bits(&[f16::ONE.to_bits(); 512], &[2, 2, 128])?,
                Array::zeros_dtype(&[8, 16, 16 * k as i32], down_dtype)?,
                Array::from_f16_bits(&[f16::ONE.to_bits(); 256], &[2, 128])?,
                Array::from_f16_bits(&[f16::ONE.to_bits(); 256], &[2, 128])?,
                1,
                k,
                Codebook::Mcg,
            )
        };
        for bad in [
            Dtype::Bool,
            Dtype::UInt8,
            Dtype::Float16,
            Dtype::Float32,
            Dtype::Int32,
        ] {
            assert!(
                make(bad, Dtype::UInt16, 2).is_err(),
                "accepted gate/up {bad:?}"
            );
            assert!(
                make(Dtype::UInt16, bad, 2).is_err(),
                "accepted down {bad:?}"
            );
        }
        for k in [0, 7, 9] {
            assert!(
                make(Dtype::UInt16, Dtype::UInt16, k).is_err(),
                "accepted K={k}"
            );
        }
        for dtype in [Dtype::Int16, Dtype::UInt16] {
            make(dtype, dtype, 2)?;
        }
        Ok(())
    }

    /// Disposable EXL3 expert bank: gate/up at `k` bits, down at `down_k` bits, default codebook,
    /// deterministic words and scales. `up_k` overrides the up width (admission tests).
    fn split_width_checkpoint(
        dir: &std::path::Path,
        prefix: &str,
        experts: usize,
        [input, hidden]: [usize; 2],
        [k, up_k, down_k]: [usize; 3],
    ) -> Result<Checkpoint> {
        let file = dir.join("experts.safetensors");
        let mut header = serde_json::Map::new();
        let mut payload = Vec::new();
        let mut state = 0x9e37_79b9_u32;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for e in 0..experts {
            for (name, rows, cols, bits) in [
                ("gate_proj", input, hidden, k),
                ("up_proj", input, hidden, up_k),
                ("down_proj", hidden, input, down_k),
            ] {
                let words = rows / 16 * (cols / 16) * 16 * bits;
                let start = payload.len();
                for _ in 0..words {
                    payload.extend((next() as u16).to_le_bytes());
                }
                header.insert(
                    format!("{prefix}.{e}.{name}.trellis"),
                    serde_json::json!({"dtype":"U16","shape":[rows / 16, cols / 16, 16 * bits],
                        "data_offsets":[start, payload.len()]}),
                );
                for (scale, len) in [("suh", rows), ("svh", cols)] {
                    let start = payload.len();
                    for _ in 0..len {
                        // signed scales in [0.5, 1.5): the sign carries the Hadamard flip
                        let unit = (next() % 1024) as f32 / 1024.0;
                        let value = if next() & 1 == 0 {
                            0.5 + unit
                        } else {
                            -0.5 - unit
                        };
                        payload.extend(f16::from_f32(value).to_le_bytes());
                    }
                    header.insert(
                        format!("{prefix}.{e}.{name}.{scale}"),
                        serde_json::json!({"dtype":"F16","shape":[len],"data_offsets":[start, payload.len()]}),
                    );
                }
            }
        }
        let json = serde_json::to_vec(&header)?;
        let mut bytes = (json.len() as u64).to_le_bytes().to_vec();
        bytes.extend(json);
        bytes.extend(payload);
        std::fs::write(&file, bytes)?;
        Ok(Checkpoint {
            path: dir.to_path_buf(),
            model_type: "qwen3_5_moe".into(),
            bits: None,
            size_bytes: 0,
            modules: Vec::new(),
            tensors: crate::checkpoint::read_header(&file)?,
        })
    }

    /// Width rules are checked from the safetensors metadata, before any tensor payload is read:
    /// gate and up share one trellis and must agree, down may differ but must be a valid width.
    #[test]
    fn split_widths_are_admitted_only_when_gate_and_up_agree() -> Result<()> {
        let prefix = "model.layers.0.mlp.experts";
        for (widths, expected) in [
            ([3, 4, 3], Some("mixed gate/up bit widths")),
            ([4, 3, 4], Some("mixed gate/up bit widths")),
            ([3, 3, 7], Some("invalid down-projection bit width")),
            ([3, 3, 9], Some("invalid down-projection bit width")),
            (
                [7, 7, 3],
                Some("invalid or overflowing SwitchGLU dimensions"),
            ),
        ] {
            let dir = tempfile::tempdir()?;
            let checkpoint = split_width_checkpoint(dir.path(), prefix, 2, [128, 128], widths)?;
            let error = match Exl3SwitchGlu::from_checkpoint(&checkpoint, prefix, 2, 1) {
                Ok(_) => anyhow::bail!("accepted widths {widths:?}"),
                Err(error) => format!("{error:#}"),
            };
            let expected = expected.context("case without expectation")?;
            ensure!(
                error.contains(expected),
                "widths {widths:?}: expected {expected:?}, got {error:?}"
            );
        }
        // one expert whose down width differs from the others is still a mixed down bank
        let dir = tempfile::tempdir()?;
        let mut checkpoint = split_width_checkpoint(dir.path(), prefix, 2, [128, 128], [3, 3, 4])?;
        checkpoint
            .tensors
            .get_mut(&format!("{prefix}.1.down_proj.trellis"))
            .context("down trellis")?
            .shape = vec![8, 8, 48];
        let error = Exl3SwitchGlu::from_checkpoint(&checkpoint, prefix, 2, 1)
            .err()
            .context("accepted mixed down widths")?;
        ensure!(format!("{error:#}").contains("mixed down-projection bit widths"));
        Ok(())
    }

    /// A bank with gate/up at 3 bits and down at 4 bits against an independent oracle: each routed
    /// expert's three projections through `Exl3Linear`, the activation in FP32, the weighted sum.
    /// Covers the small-M path and the segmented TensorOps path (64 rows or more), SiLU and the
    /// Inkling soft-capped SiLU.
    #[cfg(feature = "mlx")]
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn split_down_width_matches_per_expert_linears() -> Result<()> {
        const EXPERTS: usize = 4;
        const INPUT: i32 = 256;
        const HIDDEN: i32 = 128;
        const TOP_K: i32 = 2;
        let prefix = "model.layers.0.mlp.experts";
        let dir = tempfile::tempdir()?;
        let checkpoint = split_width_checkpoint(
            dir.path(),
            prefix,
            EXPERTS,
            [INPUT as usize, HIDDEN as usize],
            [3, 3, 4],
        )?;
        let linear = |e: usize, name: &str| {
            crate::linear::Exl3Linear::from_checkpoint(&checkpoint, &format!("{prefix}.{e}.{name}"))
        };
        let oracle = (0..EXPERTS)
            .map(|e| {
                Ok([
                    linear(e, "gate_proj")?,
                    linear(e, "up_proj")?,
                    linear(e, "down_proj")?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        let caps = (30.0f32, 20.0f32);
        let mut cases = 0;
        for inkling in [false, true] {
            let bank = if inkling {
                Exl3SwitchGlu::from_inkling_checkpoint(
                    &checkpoint,
                    prefix,
                    EXPERTS as i32,
                    TOP_K,
                    caps.0,
                    caps.1,
                )?
            } else {
                Exl3SwitchGlu::from_checkpoint(&checkpoint, prefix, EXPERTS as i32, TOP_K)?
            };
            ensure!(
                bank.k == 3 && bank.down_k == 4,
                "widths not read from the checkpoint"
            );
            for rows in [1, 5, 72] {
                let values = (0..rows * INPUT)
                    .map(|i| f16::from_f32(((i * 37 % 101) as f32 - 50.0) / 64.0).to_bits())
                    .collect::<Vec<_>>();
                let x = Array::from_f16_bits(&values, &[rows, INPUT])?;
                let routes = (0..rows * TOP_K)
                    .map(|slot| {
                        ((slot / TOP_K) * 3 + (slot % TOP_K) + slot / 7) as u32 % EXPERTS as u32
                    })
                    .collect::<Vec<_>>();
                // two distinct experts per row
                let routes = routes
                    .chunks(2)
                    .flat_map(|pair| {
                        let second = if pair[1] == pair[0] {
                            (pair[0] + 1) % EXPERTS as u32
                        } else {
                            pair[1]
                        };
                        [pair[0], second]
                    })
                    .collect::<Vec<_>>();
                let weights = (0..rows * TOP_K)
                    .map(|slot| f16::from_f32(0.25 + 0.5 * ((slot % 3) as f32) / 2.0).to_bits())
                    .collect::<Vec<_>>();
                let selected = Array::from_u32(&routes, &[rows, TOP_K])?;
                let scores = Array::from_f16_bits(&weights, &[rows, TOP_K])?;
                let actual = bank
                    .forward(&x, &selected, &scores)?
                    .astype(Dtype::Float32)?;
                ensure!(actual.shape() == [rows, INPUT], "bank output shape");
                let actual = actual.to_f32()?;
                let mut expected = vec![0f32; (rows * INPUT) as usize];
                for row in 0..rows {
                    let xr = x.slice(0, row, row + 1)?;
                    for j in 0..TOP_K {
                        let slot = (row * TOP_K + j) as usize;
                        let [gate, up, down] = &oracle[routes[slot] as usize];
                        let g = gate.forward(&xr)?.astype(Dtype::Float32)?;
                        let u = up.forward(&xr)?.astype(Dtype::Float32)?;
                        let h = if inkling {
                            g.scalar_mul(1.0 / caps.0)?
                                .tanh()?
                                .scalar_mul(caps.0)?
                                .mul(&g.sigmoid()?)?
                                .mul(&u.scalar_mul(1.0 / caps.1)?.tanh()?.scalar_mul(caps.1)?)?
                        } else {
                            g.mul(&g.sigmoid()?)?.mul(&u)?
                        };
                        let d = down.forward(&h.astype(Dtype::Float16)?)?.to_f32()?;
                        let w = f16::from_bits(weights[slot]).to_f32();
                        for (i, value) in d.iter().enumerate() {
                            expected[row as usize * INPUT as usize + i] += w * value;
                        }
                    }
                }
                ensure!(
                    actual.iter().chain(&expected).all(|v| v.is_finite()),
                    "non-finite bank or oracle output"
                );
                let scale = expected.iter().fold(0f32, |m, v| m.max(v.abs()));
                let worst = actual
                    .iter()
                    .zip(&expected)
                    .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
                let rel = worst / scale;
                println!(
                    "inkling={inkling} rows={rows}: max |bank - oracle| / max |oracle| = {rel:.2e}"
                );
                ensure!(
                    scale > 0.0 && rel < 2e-2,
                    "split-width bank differs from the oracle: {rel}"
                );
                cases += 1;
            }
        }
        ensure!(cases == 6, "cases run: {cases}");
        Ok(())
    }

    #[cfg(feature = "mlx")]
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn small_routed_input_matches_array_reference() -> Result<()> {
        const EXPERTS: i32 = 16;
        const INPUT: i32 = 256;
        let scale_values = (0..EXPERTS * 2 * INPUT)
            .map(|index| f16::from_f32(0.5 + (index % 29) as f32 / 64.0).to_bits())
            .collect::<Vec<_>>();
        let scales = Array::from_f16_bits(&scale_values, &[EXPERTS, 2, INPUT])?;
        for (rows, top_k) in [(1, 8), (6, 8)] {
            let values = (0..rows * INPUT)
                .map(|index| f16::from_f32(((index % 97) as f32 - 48.0) / 32.0).to_bits())
                .collect::<Vec<_>>();
            let routes = rows * top_k;
            let route_values = (0..routes)
                .map(|route| u32::try_from((route * 7 + 3) % EXPERTS))
                .collect::<Result<Vec<_>, _>>()?;
            let selected = Array::from_u32(&route_values, &[routes])?;
            let x = Array::from_f16_bits(&values, &[rows, INPUT])?;
            let reference = x
                .reshape(&[rows, 1, 1, INPUT])?
                .broadcast_to(&[rows, top_k, 2, INPUT])?
                .reshape(&[routes * 2, INPUT])?
                .mul(&scales.take(&selected, 0)?.reshape(&[routes * 2, INPUT])?)?
                .reshape(&[routes * 2, INPUT / 128, 128])?
                .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
                .reshape(&[routes * 2, INPUT])?;
            let actual = routed_inputs_small(&x, &scales, &selected, rows, top_k, INPUT, false)?;
            assert_eq!(actual.to_f16_bits()?, reference.to_f16_bits()?);
        }
        Ok(())
    }
    #[test]
    #[ignore = "requires physical M5 and 35B"]
    fn q36_parallel_segmented_components() -> Result<()> {
        let ck = crate::checkpoint::inspect(std::path::Path::new(&std::env::var(
            "MLXL3_QWEN_TEST_MODEL",
        )?))?;
        for layer in [0, 17] {
            let bank = Exl3SwitchGlu::from_checkpoint(
                &ck,
                &format!("model.language_model.layers.{layer}.mlp.experts"),
                256,
                8,
            )?;
            for rows in [64, 65, 73, 256] {
                let spread = rows == 73 || rows == 256;
                let ids = (0..rows * 8)
                    .map(|i| {
                        if spread {
                            (i * 37 % 256) as u32
                        } else {
                            (i % 8 * 31) as u32
                        }
                    })
                    .collect::<Vec<_>>();
                let selected = Array::from_u32(&ids, &[rows * 8])?;
                let plan = route_plan(&selected, rows, 8, 256, true)?;

                let table = plan.table.to_u32()?;
                let count = plan.count.to_u32()?;
                let mut start = 0;
                let mut expected = Vec::new();
                let size = 64;
                for expert in 0..256 {
                    let n = ids.iter().filter(|&&v| v == expert).count() as u32;
                    for offset in (0..n).step_by(size) {
                        expected.push((expert, start + offset, (n - offset).min(size as u32)));
                    }
                    start += n;
                }
                assert_eq!(count, [expected.len() as u32]);
                assert_eq!(start, (rows * 8) as u32);
                for (i, (e, offset, n)) in expected.into_iter().enumerate() {
                    let stride = plan.max_blocks as usize;
                    assert_eq!(
                        (table[i], table[stride + i], table[2 * stride + i]),
                        (e, offset, n)
                    );
                }
                for (input, output, bank_offset, weights) in [
                    (2048, 512, 0, &bank.gu_trellis),
                    (2048, 512, 8192, &bank.gu_trellis),
                    (512, 2048, 0, &bank.down_trellis),
                ] {
                    let x = Array::from_f16_bits(
                        &(0..plan.padded * input)
                            .map(|i| {
                                f16::from_f32(((i * 13 % 127) as f32 - 64.0) / 128.0).to_bits()
                            })
                            .collect::<Vec<_>>(),
                        &[plan.padded, input],
                    )?;
                    let run = |parallel| -> Result<Vec<u16>> {
                        let plan = route_plan(&selected, rows, 8, 256, parallel)?;
                        expert_segmented_qmm(
                            &x,
                            weights,
                            &plan,
                            input,
                            output,
                            256,
                            bank_offset,
                            bank.k,
                            bank.cb,
                        )?
                        .to_f16_bits()
                    };
                    let base = run(false)?;
                    let candidate = run(true)?;
                    ensure!(
                        base.len() == (rows * 8 * output) as usize
                            && base.iter().all(|&x| f16::from_bits(x).is_finite())
                            && base == candidate,
                        "segmented mismatch at {rows} layer {layer} offset {bank_offset}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "Qwen 35B checkpoint on a physical Apple GPU"]
    fn q36_expert_lane_matches_canonical_intermediates() -> Result<()> {
        use crate::linear::prepared_expert_raw;
        let path = std::env::var("MLXL3_QWEN_TEST_MODEL")?;
        let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&path))?;
        let mut cases = 0;
        for (layer, k) in [(0, 3), (17, 2)] {
            let bank = Exl3SwitchGlu::from_checkpoint(
                &checkpoint,
                &format!("model.language_model.layers.{layer}.mlp.experts"),
                256,
                8,
            )?;
            assert_eq!(bank.k, k);
            for rows in [1, 2, 4] {
                let x = Array::from_f16_bits(
                    &(0..rows * 2048)
                        .map(|i| f16::from_f32(((i * 17 % 113) as f32 - 55.0) / 50.0).to_bits())
                        .collect::<Vec<_>>(),
                    &[rows, 2048],
                )?;
                let ids = Array::from_u32(
                    &(0..rows * 8)
                        .map(|i| [0, 1, 31, 255, 87, 201, 33, 76][(i % 8) as usize])
                        .collect::<Vec<_>>(),
                    &[rows * 8],
                )?;
                let compare_prepared = |canonical: &Array, permuted: &Array| -> Result<()> {
                    let original = canonical.to_f16_bits()?;
                    let actual = permuted.to_f16_bits()?;
                    assert_eq!(canonical.shape(), permuted.shape());
                    assert!(
                        !original.is_empty()
                            && original.len() == actual.len()
                            && original.len() % 16 == 0
                    );
                    assert!(original.iter().all(|&v| f16::from_bits(v).is_finite()));
                    assert_ne!(original, actual);
                    for (i, value) in actual.into_iter().enumerate() {
                        assert_eq!(
                            value,
                            original[crate::contracts::exl3_lane_source_index(i as u32) as usize]
                        );
                    }
                    Ok(())
                };
                let mut outputs = Vec::new();
                for permuted in [false, true] {
                    let a = routed_inputs_small(&x, &bank.gu_suh, &ids, rows, 8, 2048, permuted)?;
                    let gu = prepared_expert_raw(
                        &a,
                        &bank.gu_trellis,
                        &ids,
                        512,
                        2,
                        8192,
                        k,
                        bank.cb,
                        permuted,
                    )?;
                    let b = glu_down_input(
                        &gu,
                        &bank.gu_svh,
                        &bank.down_suh,
                        &ids,
                        bank.activation,
                        512,
                        permuted,
                    )?;
                    let down = prepared_expert_raw(
                        &b,
                        &bank.down_trellis,
                        &ids,
                        2048,
                        1,
                        0,
                        k,
                        bank.cb,
                        permuted,
                    )?;
                    outputs.push((a, gu, b, down));
                }
                let [old, new] = outputs.as_slice() else {
                    unreachable!()
                };
                compare_prepared(&old.0, &new.0)?;
                compare_prepared(&old.2, &new.2)?;
                for (a, b, expected) in [
                    (&old.1, &new.1, vec![rows * 16, 512]),
                    (&old.3, &new.3, vec![rows * 8, 2048]),
                ] {
                    assert_eq!(a.shape(), expected);
                    assert_eq!(b.shape(), expected);
                    let v = a.to_f32()?;
                    assert!(v.iter().all(|n| n.is_finite()) && v.iter().any(|n| *n != 0.0));
                    assert_eq!(a.to_bytes()?, b.to_bytes()?);
                }
                cases += 1;
            }
        }
        assert_eq!(cases, 6);
        Ok(())
    }
}
