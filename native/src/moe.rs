//! Qwen decode-only MoE transforms around the mapped EXL3 projections.
use crate::{
    array::{self, Array, Dtype},
    checkpoint::Checkpoint,
    codec::{self, Codebook},
    linear::{checkpoint_array, codebook_header, expert_mapped},
};
use anyhow::{Context, Result, ensure};

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

fn route_plan(selected: &Array, rows: i32, top_k: i32, experts: i32) -> Result<RoutePlan> {
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

            if (tid == 0u) {{
                uint next_block = 0u;
                for (uint expert = 0u; expert < EXPERTS; ++expert) {{
                    uint lo = 0u, hi = SLOTS;
                    while (lo < hi) {{
                        uint mid = (lo + hi) >> 1u;
                        if (selected[order[mid]] < expert) lo = mid + 1u;
                        else hi = mid;
                    }}
                    uint begin = lo;
                    hi = SLOTS;
                    while (lo < hi) {{
                        uint mid = (lo + hi) >> 1u;
                        if (selected[order[mid]] <= expert) lo = mid + 1u;
                        else hi = mid;
                    }}
                    uint count = lo - begin;
                    for (uint offset = 0u; offset < count; offset += 64u) {{
                        block_table[next_block] = expert;
                        block_table[MAX_BLOCKS + next_block] = begin + offset;
                        block_table[2u * MAX_BLOCKS + next_block] = min(64u, count - offset);
                        ++next_block;
                    }}
                }}
                block_count[0] = next_block;
            }}
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
        array::is_m5_gpu()?
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
    Ok(array::metal_kernel(
        &format!(
            "mlxl3_rs_expert_qmm_segmented_{input}_{output}_{experts}_{k}_{}_o{tile_offset}_v1",
            cb as u32
        ),
        &["xhat", "trellis", "block_table", "block_count"],
        &["yhat"],
        &header,
        include_str!("../shaders/_expert_qmm_tensor_kernel.metal"),
        &[
            xhat,
            &trellis.reshape(&[-1])?.view(Dtype::UInt32)?,
            &plan.table,
            &plan.count,
        ],
        &[vec![plan.slots, output]],
        &[Dtype::Float16],
        [(output / 32) * 32, plan.max_blocks, 2],
        [32, 1, 1],
    )?
    .remove(0))
}

fn glu_down_input(
    gate_up_raw: &Array,
    gate_up_output_scales: &Array,
    down_input_scales: &Array,
    selected: &Array,
    activation: GluActivation,
    logical_hidden: i32,
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
    let activated = if gelu {
        "float(half(half(0.5f * gate_value * (1.0f + tanh(0.7978845608028654f * (gate_value + 0.044715f * gate_value * gate_value * gate_value)))) * half(up_value)))"
    } else {
        "float(half((gate_value / (1.0f + exp(-gate_value))) * up_value))"
    };
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
            xhat[slot * HIDDEN + column] = half({down_finish} * HAD_SCALE);
        "#
    );
    Ok(array::metal_kernel(
        &format!(
            "mlxl3_rs_moe_glu_down_h{hidden}_{}_l{logical_hidden}_v2",
            if gelu { "gelu" } else { "silu" }
        ),
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
    k: usize,
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

    fn from_checkpoint_names(
        checkpoint: &Checkpoint,
        prefix: &str,
        experts: i32,
        top_k: i32,
        [gate, up, down]: [&str; 3],
        activation: GluActivation,
        logical_hidden: Option<i32>,
    ) -> Result<Self> {
        ensure!(experts > 0, "invalid expert count");
        let names = |projection: &str| {
            (0..experts)
                .map(|expert| format!("{prefix}.{expert}.{projection}"))
                .collect::<Vec<_>>()
        };
        let gates = names(gate);
        let ups = names(up);
        let downs = names(down);
        let all = gates.iter().chain(&ups).chain(&downs);
        let mut k = None;
        let mut cb = None;
        for name in all {
            let trellis = checkpoint
                .tensors
                .get(&format!("{name}.trellis"))
                .with_context(|| format!("missing expert trellis {name}"))?;
            let bits = *trellis.shape.last().context("empty expert trellis")? / 16;
            ensure!(
                k.is_none_or(|value| value == bits),
                "mixed expert bit widths"
            );
            k = Some(bits);
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
        let scale = |name: &str, primary: &str, legacy: &str| {
            let key = format!("{name}.{primary}");
            let key = if checkpoint.tensors.contains_key(&key) {
                key
            } else {
                format!("{name}.{legacy}")
            };
            checkpoint_array(checkpoint, &key)
        };
        let trellises = |names: &[String]| -> Result<Vec<Array>> {
            names
                .iter()
                .map(|name| checkpoint_array(checkpoint, &format!("{name}.trellis")))
                .collect()
        };
        let mut gu_trellises = trellises(&gates)?;
        gu_trellises.extend(trellises(&ups)?);
        let gu_trellis = Array::concatenate(&gu_trellises.iter().collect::<Vec<_>>(), 1)?;
        let down_trellises = trellises(&downs)?;
        let down_trellis = Array::concatenate(&down_trellises.iter().collect::<Vec<_>>(), 1)?;
        let paired = |left: &[String], right: &[String], primary, legacy| -> Result<Array> {
            let mut rows = Vec::with_capacity(left.len());
            for (a, b) in left.iter().zip(right) {
                rows.push(Array::concatenate(
                    &[&scale(a, primary, legacy)?, &scale(b, primary, legacy)?],
                    0,
                )?);
            }
            let joined = Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0)?;
            let width = joined.shape()[0] / (experts * 2);
            joined.reshape(&[experts, 2, width])
        };
        let stacked = |names: &[String], primary, legacy| -> Result<Array> {
            let rows = names
                .iter()
                .map(|name| scale(name, primary, legacy)?.reshape(&[1, -1]))
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
        ensure!(
            experts > 0
                && projections == 2
                && scale_experts == experts
                && scale_projections == 2
                && top_k > 0
                && top_k <= experts
                && input % 128 == 0
                && hidden % 128 == 0,
            "invalid SwitchGLU dimensions"
        );
        ensure!(
            down_suh.shape() == [experts, hidden]
                && down_svh.shape() == [experts, input]
                && gu_trellis.shape() == [input / 16, experts * hidden / 8, (16 * k) as i32]
                && down_trellis.shape() == [hidden / 16, experts * input / 16, (16 * k) as i32],
            "SwitchGLU trellis/scales disagree"
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
        if rows >= 64 && self.experts <= 256 && self.k != 7 && array::is_m5_gpu()? {
            return self.forward_segmented(x, &selected, scores, rows);
        }
        let x_gu = x
            .reshape(&[rows, 1, 1, self.input])?
            .broadcast_to(&[rows, self.top_k, 2, self.input])?
            .reshape(&[slots * 2, self.input])?;
        let gu_input_scales = self
            .gu_suh
            .take(&selected, 0)?
            .reshape(&[slots * 2, self.input])?;
        let gate_up = expert_mapped(
            &x_gu,
            &self.gu_trellis,
            Some(&gu_input_scales),
            None,
            &selected,
            self.hidden,
            2,
            self.experts * self.hidden / 16,
            self.k,
            self.cb,
            false,
            true,
        )?;
        let down_input = glu_down_input(
            &gate_up,
            &self.gu_svh,
            &self.down_suh,
            &selected,
            self.activation,
            self.logical_hidden,
        )?;
        let down = expert_mapped(
            &down_input,
            &self.down_trellis,
            None,
            None,
            &selected,
            self.input,
            1,
            0,
            self.k,
            self.cb,
            true,
            true,
        )?;
        finish_and_reduce(&down, &self.down_svh.take(&selected, 0)?, scores)
    }

    fn forward_segmented(
        &self,
        x: &Array,
        selected: &Array,
        scores: &Array,
        rows: i32,
    ) -> Result<Array> {
        let plan = route_plan(selected, rows, self.top_k, self.experts)?;
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
            self.k,
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
            self.k,
            self.cb,
            false,
            false,
        )?
        .reshape(&[rows, self.top_k, self.input])?
        .mul(&scores.reshape(&[rows, self.top_k, 1])?)?
        .sum(1, false)
    }
}
