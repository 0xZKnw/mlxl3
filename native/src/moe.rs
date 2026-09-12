//! Qwen decode-only MoE transforms around the mapped EXL3 projections.
use crate::{
    array::{self, Array, Dtype},
    checkpoint::Checkpoint,
    codec::Codebook,
    linear::{checkpoint_array, expert_mapped},
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
        ensure!(
            x.shape() == [1, self.input]
                && selected.shape() == [1, self.top_k]
                && scores.shape() == [1, self.top_k],
            "invalid SwitchGLU input/routes"
        );
        let selected = selected.reshape(&[self.top_k])?;
        if matches!(self.activation, GluActivation::Gelu) {
            return self.forward_gelu(x, &selected, scores);
        }
        let x_gu = x
            .reshape(&[1, 1, 1, self.input])?
            .broadcast_to(&[1, self.top_k, 2, self.input])?
            .reshape(&[self.top_k * 2, self.input])?;
        let gu_input_scales = self
            .gu_suh
            .take(&selected, 0)?
            .reshape(&[self.top_k * 2, self.input])?;
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

    fn forward_gelu(&self, x: &Array, selected: &Array, scores: &Array) -> Result<Array> {
        let gate_up = expert_mapped(
            &x.reshape(&[1, 1, 1, self.input])?
                .broadcast_to(&[1, self.top_k, 2, self.input])?
                .reshape(&[self.top_k * 2, self.input])?,
            &self.gu_trellis,
            Some(
                &self
                    .gu_suh
                    .take(selected, 0)?
                    .reshape(&[self.top_k * 2, self.input])?,
            ),
            Some(
                &self
                    .gu_svh
                    .take(selected, 0)?
                    .reshape(&[self.top_k * 2, self.hidden])?,
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
        .reshape(&[self.top_k, 2, self.hidden])?;
        let gate = gate_up
            .slice(1, 0, 1)?
            .reshape(&[self.top_k, self.hidden])?;
        let up = gate_up
            .slice(1, 1, 2)?
            .reshape(&[self.top_k, self.hidden])?;
        let mut hidden = gate.geglu(&up)?;
        if self.logical_hidden < self.hidden {
            hidden = Array::concatenate(
                &[
                    &hidden.slice(1, 0, self.logical_hidden)?,
                    &Array::zeros_dtype(
                        &[self.top_k, self.hidden - self.logical_hidden],
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
        .mul(&scores.reshape(&[self.top_k, 1])?)?
        .sum(0, false)?
        .reshape(&[1, self.input])
    }
}
