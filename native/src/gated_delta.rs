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

#[allow(clippy::too_many_arguments)]
pub fn step_with_gates(
    q: &Array,
    k: &Array,
    v: &Array,
    a: &Array,
    b: &Array,
    a_log: &Array,
    dt_bias: &Array,
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
        batch == 1 && time == 1 && v_batch == batch && v_time == time,
        "fused Gated DeltaNet gates require one decode token"
    );
    ensure!(
        key_dim == 128
            && value_dim > 0
            && value_dim % 8 == 0
            && key_heads > 0
            && value_heads > 0
            && value_heads % key_heads == 0,
        "invalid fused Gated DeltaNet head dimensions"
    );
    ensure!(
        k.shape() == q.shape()
            && a.shape() == [batch, time, value_heads]
            && b.shape() == a.shape()
            && a_log.shape() == [value_heads]
            && dt_bias.shape() == [value_heads]
            && state.shape() == [batch, value_heads, value_dim, key_dim],
        "invalid fused Gated DeltaNet gate shapes"
    );
    ensure!(
        q.dtype() == Dtype::Float16
            && k.dtype() == Dtype::Float16
            && v.dtype() == Dtype::Float16
            && a.dtype() == Dtype::Float16
            && b.dtype() == Dtype::Float16
            && a_log.dtype() == Dtype::Float32
            && dt_bias.dtype() == Dtype::Float16
            && state.dtype() == Dtype::Float32,
        "invalid fused Gated DeltaNet gate dtypes"
    );
    let header = format!(
        "#define MLXL3_GDN_FUSED_GATES 1\n#define InT half\n#define StT float\n#define T 1\n#define Dk {key_dim}\n#define Dv {value_dim}\n#define Hk {key_heads}\n#define Hv {value_heads}\n"
    );
    let mut outputs = array::metal_kernel(
        &format!("mlxl3_rs_gdn_packed_gates_hk{key_heads}_hv{value_heads}_dv{value_dim}"),
        &["q", "k", "v", "a", "b", "a_log", "dt_bias", "state_in"],
        &["y", "state_out"],
        &header,
        include_str!("../shaders/gated_delta_packed.metal"),
        &[q, k, v, a, b, a_log, dt_bias, state],
        &[
            vec![batch, time, value_heads, value_dim],
            state.shape().to_vec(),
        ],
        &[Dtype::Float16, Dtype::Float32],
        [32, value_dim / 8, value_heads],
        [32, 2, 1],
    )?;
    ensure!(
        outputs.len() == 2,
        "fused Gated DeltaNet gates returned wrong output count"
    );
    let state = outputs
        .pop()
        .context("missing fused Gated DeltaNet state")?;
    let output = outputs
        .pop()
        .context("missing fused Gated DeltaNet output")?;
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
    step_vector_impl(q, k, v, g, beta, state, false)
}

pub fn step_vector_with_beta(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta_raw: &Array,
    state: &Array,
) -> Result<(Array, Array)> {
    ensure!(
        q.shape().get(1) == Some(&1),
        "fused vector beta requires one decode token"
    );
    step_vector_impl(q, k, v, g, beta_raw, state, true)
}

fn step_vector_impl(
    q: &Array,
    k: &Array,
    v: &Array,
    g: &Array,
    beta: &Array,
    state: &Array,
    beta_is_raw: bool,
) -> Result<(Array, Array)> {
    let [batch, time, heads, dim]: [i32; 4] = q
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("vector Gated DeltaNet q must have rank 4"))?;
    ensure!(
        batch > 0
            && time > 0
            && heads > 0
            && dim == 128
            && k.shape() == q.shape()
            && v.shape() == q.shape()
            && g.shape() == q.shape()
            && beta.shape() == [batch, time, heads]
            && state.shape() == [batch, heads, dim, dim],
        "invalid vector Gated DeltaNet dimensions"
    );
    ensure!(
        q.dtype() == Dtype::Float16
            && k.dtype() == Dtype::Float16
            && v.dtype() == Dtype::Float16
            && g.dtype() == Dtype::Float32
            && beta.dtype()
                == if beta_is_raw {
                    Dtype::Float32
                } else {
                    Dtype::Float16
                }
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
        ulong state_base = (ulong(batch) * HEADS * 128u + head * 128u + value_index) * 128u;
        float local[4];
        for (uint i = 0u; i < 4u; ++i) {
            local[i] = state_in[state_base + base + i];
        }
        for (uint t = 0u; t < TIME; ++t) {
            ulong vector_base = ((ulong(batch) * TIME + t) * HEADS + head) * 128u;
            float memory = 0.0f;
            for (uint i = 0u; i < 4u; ++i) {
                uint key_index = base + i;
                local[i] *= g[vector_base + key_index];
                memory += local[i] * float(k[vector_base + key_index]);
            }
            memory += simd_shuffle_xor(memory, 1);
            memory += simd_shuffle_xor(memory, 2);
            memory += simd_shuffle_xor(memory, 4);
            memory += simd_shuffle_xor(memory, 8);
            memory += simd_shuffle_xor(memory, 16);
#if MLXL3_GDN_VECTOR_FUSED_BETA
            float beta_value = beta_raw[(ulong(batch) * TIME + t) * HEADS + head];
            float beta_tail = 1.0f / (1.0f + metal::exp(metal::abs(beta_value)));
            half beta_t = half(beta_value < 0.0f ? beta_tail : 1.0f - beta_tail);
#else
            half beta_t = beta[(ulong(batch) * TIME + t) * HEADS + head];
#endif
            float delta = (float(v[vector_base + value_index]) - memory)
                * float(beta_t);
            float output = 0.0f;
            for (uint i = 0u; i < 4u; ++i) {
                uint key_index = base + i;
                local[i] += float(k[vector_base + key_index]) * delta;
                output += local[i] * float(q[vector_base + key_index]);
            }
            output += simd_shuffle_xor(output, 1);
            output += simd_shuffle_xor(output, 2);
            output += simd_shuffle_xor(output, 4);
            output += simd_shuffle_xor(output, 8);
            output += simd_shuffle_xor(output, 16);
            if (lane == 0u) y[vector_base + value_index] = half(output);
        }
        for (uint i = 0u; i < 4u; ++i) {
            state_out[state_base + base + i] = local[i];
        }
    "#;
    let fused_header = if beta_is_raw {
        "#define MLXL3_GDN_VECTOR_FUSED_BETA 1\n"
    } else {
        ""
    };
    let beta_name = if beta_is_raw { "beta_raw" } else { "beta" };
    let mut outputs = array::metal_kernel(
        &format!(
            "mlxl3_rs_gdn_vector{}_b{batch}_t{time}_h{heads}_d{dim}_v2",
            if beta_is_raw { "_raw_beta" } else { "" }
        ),
        &["q", "k", "v", "g", beta_name, "state_in"],
        &["y", "state_out"],
        &format!("{fused_header}#define TIME {time}u\n#define HEADS {heads}u\n"),
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
    fn packed_step_with_gates_matches_graph() -> Result<()> {
        let mut qkv = vec![0u16; 128];
        qkv[0] = half::f16::ONE.to_bits();
        let q = Array::from_f16_bits(&qkv, &[1, 1, 1, 128])?;
        let values = Array::from_f16_bits(&[half::f16::ONE.to_bits(); 8], &[1, 1, 1, 8])?;
        let a = Array::from_f16_bits(&[half::f16::from_f32(-0.75).to_bits()], &[1, 1, 1])?;
        let b = Array::from_f16_bits(&[half::f16::from_f32(0.625).to_bits()], &[1, 1, 1])?;
        let a_log = Array::from_f32(&[-4.25], &[1])?;
        let dt_bias = Array::from_f16_bits(&[half::f16::from_f32(0.125).to_bits()], &[1])?;
        let state = Array::zeros_dtype(&[1, 1, 8, 128], Dtype::Float32)?;
        let zero = Array::from_f16_bits(&[0], &[])?;
        let beta = b.sigmoid()?;
        let softplus = a.add(&dt_bias)?.logaddexp(&zero)?;
        let decay = a_log.exp()?.mul(&softplus)?.negative()?.exp()?;
        let (expected, expected_state) = step(&q, &q, &values, &decay, &beta, &state)?;
        let (actual, actual_state) =
            step_with_gates(&q, &q, &values, &a, &b, &a_log, &dt_bias, &state)?;
        assert_eq!(actual.to_f16_bits()?, expected.to_f16_bits()?);
        assert_eq!(actual_state.to_f32()?, expected_state.to_f32()?);
        Ok(())
    }

    #[test]
    #[ignore = "requires Apple GPU"]
    fn vector_step_with_beta_matches_graph() -> Result<()> {
        let mut qkv = vec![0u16; 128];
        qkv[0] = half::f16::ONE.to_bits();
        let q = Array::from_f16_bits(&qkv, &[1, 1, 1, 128])?;
        let g = Array::from_f32(&[1.0; 128], &[1, 1, 1, 128])?;
        let beta_raw = Array::from_f32(&[0.625], &[1, 1, 1])?;
        let beta = beta_raw.sigmoid()?.astype(Dtype::Float16)?;
        let state = Array::from_f32(&[0.25; 128 * 128], &[1, 1, 128, 128])?;
        let (expected, expected_state) = step_vector(&q, &q, &q, &g, &beta, &state)?;
        let (actual, actual_state) = step_vector_with_beta(&q, &q, &q, &g, &beta_raw, &state)?;
        assert_eq!(actual.to_f16_bits()?, expected.to_f16_bits()?);
        assert_eq!(actual_state.to_f32()?, expected_state.to_f32()?);
        Ok(())
    }

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

    #[test]
    #[ignore = "requires Apple GPU"]
    fn vector_batch_matches_serial_steps() -> Result<()> {
        let mut qkv = vec![0u16; 256];
        qkv[0] = half::f16::ONE.to_bits();
        qkv[128] = half::f16::ONE.to_bits();
        let qkv = Array::from_f16_bits(&qkv, &[1, 2, 1, 128])?;
        let values = Array::from_f16_bits(&vec![half::f16::ONE.to_bits(); 256], &[1, 2, 1, 128])?;
        let decay = Array::from_f32(&[vec![0.8; 128], vec![0.9; 128]].concat(), &[1, 2, 1, 128])?;
        let beta = Array::from_f16_bits(&[half::f16::ONE.to_bits(); 2], &[1, 2, 1])?;
        let initial = Array::zeros_dtype(&[1, 1, 128, 128], Dtype::Float32)?;
        let (batched, batched_state) = step_vector(&qkv, &qkv, &values, &decay, &beta, &initial)?;
        let mut state = initial;
        let mut serial = Vec::new();
        for time in 0..2 {
            let (output, next) = step_vector(
                &qkv.slice(1, time, time + 1)?,
                &qkv.slice(1, time, time + 1)?,
                &values.slice(1, time, time + 1)?,
                &decay.slice(1, time, time + 1)?,
                &beta.slice(1, time, time + 1)?,
                &state,
            )?;
            serial.push(output);
            state = next;
        }
        let serial = Array::concatenate(&serial.iter().collect::<Vec<_>>(), 1)?;
        assert_eq!(batched.to_f16_bits()?, serial.to_f16_bits()?);
        assert_eq!(batched_state.to_f32()?, state.to_f32()?);
        Ok(())
    }
}
