//! Qwen Gated DeltaNet state transition through the production MSL kernel.
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
        batch > 0 && batch == v_batch && time > 0 && v_time == time,
        "invalid Gated DeltaNet batch/time dimensions"
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
        g.shape() == [batch, time, value_heads] && g.dtype() == Dtype::Float32,
        "Gated DeltaNet decay must be FP32 [batch,time,value_heads]"
    );
    ensure!(
        beta.shape() == [batch, time, value_heads],
        "invalid Gated DeltaNet beta shape"
    );
    ensure!(
        state.shape() == [batch, value_heads, value_dim, key_dim]
            && state.dtype() == Dtype::Float32,
        "Gated DeltaNet state must be FP32 [batch,value_heads,value_dim,128]"
    );
    let header = format!(
        "#define InT half\n#define StT float\n#define T {time}\n#define Dk {key_dim}\n#define Dv {value_dim}\n#define Hk {key_heads}\n#define Hv {value_heads}\n"
    );
    let mut outputs = array::metal_kernel(
        &format!(
            "mlxl3_rs_gdn_packed_b{batch}_t{time}_hk{key_heads}_hv{value_heads}_dv{value_dim}"
        ),
        &["q", "k", "v", "g", "beta", "state_in"],
        &["y", "state_out"],
        &header,
        include_str!("../shaders/gated_delta_packed.metal"),
        &[q, k, v, g, beta, state],
        &[
            vec![batch, time, value_heads, value_dim],
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

pub fn step_vector(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
) -> Result<(Array, Array)> {
    let [batch, time, heads, dim]: [i32; 4] = q
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("vector Gated DeltaNet q must have rank 4"))?;
    ensure!(
        batch > 0
            && time == 1
            && heads > 0
            && dim == 128
            && k.shape() == q.shape()
            && v.shape() == q.shape()
            && g.shape() == q.shape()
            && beta.shape() == [batch, 1, heads]
            && state.shape() == [batch, heads, dim, dim],
        "invalid vector Gated DeltaNet dimensions"
    );
    ensure!(
        q.dtype() == Dtype::Float16
            && k.dtype() == Dtype::Float16
            && v.dtype() == Dtype::Float16
            && g.dtype() == Dtype::Float32
            && beta.dtype() == Dtype::Float16
            && state.dtype() == Dtype::Float32,
        "invalid vector Gated DeltaNet dtypes"
    );
    let source = r#"
        uint packed = thread_position_in_grid.z;
        uint batch = packed / HEADS;
        uint head = packed - batch * HEADS;
        uint lane = thread_position_in_threadgroup.x;
        uint value_index = thread_position_in_grid.y;
        uint base = lane * 4u;
        ulong vector_base = (ulong(batch) * HEADS + head) * 128u;
        ulong state_base = (ulong(batch) * HEADS * 128u + head * 128u + value_index) * 128u;
        float local[4];
        float memory = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            uint key_index = base + i;
            local[i] = state_in[state_base + key_index] * g[vector_base + key_index];
            memory += local[i] * float(k[vector_base + key_index]);
        }
        memory += simd_shuffle_xor(memory, 1);
        memory += simd_shuffle_xor(memory, 2);
        memory += simd_shuffle_xor(memory, 4);
        memory += simd_shuffle_xor(memory, 8);
        memory += simd_shuffle_xor(memory, 16);
        float delta = (float(v[vector_base + value_index]) - memory)
            * float(beta[(ulong(batch) * HEADS) + head]);
        float output = 0.0f;
        for (uint i = 0u; i < 4u; ++i) {
            uint key_index = base + i;
            local[i] += float(k[vector_base + key_index]) * delta;
            state_out[state_base + key_index] = local[i];
            output += local[i] * float(q[vector_base + key_index]);
        }
        output += simd_shuffle_xor(output, 1);
        output += simd_shuffle_xor(output, 2);
        output += simd_shuffle_xor(output, 4);
        output += simd_shuffle_xor(output, 8);
        output += simd_shuffle_xor(output, 16);
        if (lane == 0u) y[vector_base + value_index] = half(output);
    "#;
    let mut outputs = array::metal_kernel(
        &format!("mlxl3_rs_gdn_vector_b{batch}_h{heads}_d{dim}_v1"),
        &["q", "k", "v", "g", "beta", "state_in"],
        &["y", "state_out"],
        &format!("#define HEADS {heads}u\n"),
        source,
        &[q, k, v, g, beta, state],
        &[q.shape().to_vec(), state.shape().to_vec()],
        &[Dtype::Float16, Dtype::Float32],
        [
            32,
            dim,
            batch.checked_mul(heads).context("GDN grid overflow")?,
        ],
        [32, 4, 1],
    )?;
    let state = outputs
        .pop()
        .context("missing vector Gated DeltaNet state")?;
    let output = outputs
        .pop()
        .context("missing vector Gated DeltaNet output")?;
    Ok((output, state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires Apple GPU"]
    fn vector_step_matches_one_hot_update() -> Result<()> {
        let mut qkv = vec![0u16; 128];
        qkv[0] = half::f16::ONE.to_bits();
        let ones = vec![half::f16::ONE.to_bits(); 128];
        let (output, state) = step_vector(
            &Array::from_f16_bits(&qkv, &[1, 1, 1, 128])?,
            &Array::from_f16_bits(&qkv, &[1, 1, 1, 128])?,
            &Array::from_f16_bits(&ones, &[1, 1, 1, 128])?,
            &Array::from_f32(&vec![1.; 128], &[1, 1, 1, 128])?,
            &Array::from_f16_bits(&[half::f16::ONE.to_bits()], &[1, 1, 1])?,
            &Array::zeros_dtype(&[1, 1, 128, 128], Dtype::Float32)?,
        )?;
        assert!(output.to_f32()?.iter().all(|&value| value == 1.));
        let state = state.to_f32()?;
        assert!((0..128).all(|row| state[row * 128] == 1.));
        Ok(())
    }
}
