//! One-token Qwen Gated DeltaNet state transition through the production MSL kernel.
use crate::array::{self, Array, Dtype};
use anyhow::{Context, Result, ensure};

pub fn step(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array)> {
    let [batch, time, key_heads, key_dim]: [i32; 4] = q
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("Gated DeltaNet q must have rank 4"))?;
    let [v_batch, v_time, value_heads, value_dim]: [i32; 4] = v
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("Gated DeltaNet v must have rank 4"))?;
    ensure!(
        batch > 0 && batch == v_batch && time == 1 && v_time == 1,
        "native Gated DeltaNet currently accepts one token"
    );
    ensure!(
        key_dim == 128 && value_dim > 0 && value_dim % 8 == 0,
        "packed Gated DeltaNet requires key dim 128 and value dim divisible by 8"
    );
    ensure!(
        key_heads > 0 && value_heads > 0 && value_heads % key_heads == 0,
        "invalid Gated DeltaNet head counts"
    );
    ensure!(k.shape() == q.shape(), "Gated DeltaNet q/k shapes differ");
    ensure!(
        q.dtype() == Dtype::Float16
            && k.dtype() == Dtype::Float16
            && v.dtype() == Dtype::Float16
            && beta.dtype() == Dtype::Float16,
        "native packed Gated DeltaNet expects FP16 q/k/v/beta"
    );
    ensure!(
        g.shape() == [batch, 1, value_heads] && g.dtype() == Dtype::Float32,
        "Gated DeltaNet decay must be FP32 [batch,1,value_heads]"
    );
    ensure!(
        beta.shape() == [batch, 1, value_heads],
        "invalid Gated DeltaNet beta shape"
    );
    ensure!(
        state.shape() == [batch, value_heads, value_dim, key_dim]
            && state.dtype() == Dtype::Float32,
        "Gated DeltaNet state must be FP32 [batch,value_heads,value_dim,128]"
    );
    let header = format!(
        "#define InT half\n#define StT float\n#define T 1\n#define Dk {key_dim}\n#define Dv {value_dim}\n#define Hk {key_heads}\n#define Hv {value_heads}\n"
    );
    let mut outputs = array::metal_kernel(
        &format!("mlxl3_rs_gdn_packed_b{batch}_hk{key_heads}_hv{value_heads}_dv{value_dim}"),
        &["q", "k", "v", "g", "beta", "state_in"],
        &["y", "state_out"],
        &header,
        include_str!("../shaders/gated_delta_packed.metal"),
        &[q, k, v, g, beta, state],
        &[
            vec![batch, 1, value_heads, value_dim],
            state.shape().to_vec(),
        ],
        &[Dtype::Float16, Dtype::Float32],
        [
            32,
            value_dim / 8,
            batch
                .checked_mul(value_heads)
                .context("GDN grid overflow")?,
        ],
        [32, 2, 1],
    )?;
    ensure!(
        outputs.len() == 2,
        "Gated DeltaNet kernel returned wrong output count"
    );
    let state = outputs.pop().context("missing Gated DeltaNet state")?;
    let output = outputs.pop().context("missing Gated DeltaNet output")?;
    Ok((output, state))
}
