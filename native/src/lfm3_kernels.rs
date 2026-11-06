//! Metal fusions for LFM3's shared text backbone (standalone and VL).
use crate::{
    array::{self, Array, Dtype},
    lfm3,
};
use anyhow::{Context, Result, ensure};

#[cfg(test)]
static CONV_DISPATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
#[cfg(test)]
static GATE_DISPATCHES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

pub(crate) fn short_conv_step(
    bcx: &Array,
    state: &Array,
    weight: &Array,
) -> Result<(Array, Array)> {
    let shape = bcx.shape();
    ensure!(
        shape.len() == 3 && shape[0] == 1 && shape[1] == 1 && shape[2] % 3 == 0,
        "invalid LFM3 convolution input"
    );
    let dim = shape[2] / 3;
    let [input, state_values] =
        lfm3::short_conv_sizes(dim).context("LFM3 convolution size overflow")?;
    ensure!(
        shape[2] == input
            && state.shape() == [1, 6, dim]
            && weight.shape() == [dim, 7, 1]
            && [bcx, state, weight]
                .iter()
                .all(|x| x.dtype() == Dtype::Float16),
        "invalid LFM3 convolution layout or dtype"
    );
    #[cfg(test)]
    CONV_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut outputs = array::metal_kernel(
        &format!("mlxl3_lfm3_conv_step7_{dim}_v2"),
        &["bcx", "state", "weight"],
        &["y", "next_state"],
        &format!("constant uint DIM = {dim}u; constant uint STATE_VALUES = {state_values}u;"),
        include_str!("../shaders/lfm3_short_conv.metal"),
        &[bcx, state, weight],
        &[vec![1, 1, dim], vec![1, 6, dim]],
        &[Dtype::Float16, Dtype::Float16],
        [dim, 1, 1],
        [dim.min(256), 1, 1],
    )?;
    let next_state = outputs.remove(1);
    Ok((outputs.remove(0), next_state))
}

pub(crate) fn attention_gate(attention: &Array, gate: &Array) -> Result<Array> {
    let shape = attention.shape();
    ensure!(
        shape.len() == 4 && shape[0] == 1,
        "invalid LFM3 attention rank or batch"
    );
    let (heads, time, dim) = (shape[1], shape[2], shape[3]);
    let [gate_values, elements] =
        lfm3::attention_gate_sizes(time, heads, dim).context("LFM3 attention size overflow")?;
    let hidden = heads
        .checked_mul(dim)
        .context("LFM3 attention width overflow")?;
    ensure!(
        gate.shape() == [1, time, heads]
            && attention.dtype() == Dtype::Float16
            && gate.dtype() == Dtype::Float16,
        "invalid LFM3 attention gate layout or dtype"
    );
    #[cfg(test)]
    GATE_DISPATCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(array::metal_kernel(
        &format!("mlxl3_lfm3_gate_{time}_{heads}_{dim}_v1"),
        &["attention", "gate"], &["y"],
        &format!("constant uint TIME = {time}u; constant uint HEADS = {heads}u; constant uint HEAD_DIM = {dim}u; constant uint GATE_VALUES = {gate_values}u;"),
        include_str!("../shaders/lfm3_attention_gate.metal"), &[attention, gate],
        &[vec![1, time, hidden]], &[Dtype::Float16], [elements, 1, 1], [elements.min(256), 1, 1],
    )?.remove(0))
}

#[cfg(test)]
#[path = "lfm3_kernels_tests.rs"]
mod tests;
