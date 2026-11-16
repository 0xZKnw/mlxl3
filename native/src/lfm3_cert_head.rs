//! LFM3-CERT-HEAD: certified greedy head.
//!
//! The greedy token only depends on the argmax of `log_probs(head(x))`. An
//! int8 copy of the F16 head (one scale per row) is read instead of the head,
//! with a rigorous per-row bound on the difference to the logit computed by
//! the exact head kernel. Rows whose upper bound reaches the best lower bound
//! are recomputed with the exact head arithmetic; when the best exact logit
//! beats every other row by more than `2^-5` the argmax is certified,
//! otherwise the caller falls back to the full head. Certified IDs are
//! therefore the IDs of the current path by construction.
//!
//! Bound: with `v_r` the FP32 dot product of the exact head kernel, `A_r` the
//! FP32 screening dot product and `w_hat = s_r q_r`,
//! `|v_r - A_r| <= |x| (|w_r - w_hat_r| + gamma (|w_r| + |w_hat_r|))`
//! (Cauchy-Schwarz and the standard `gamma_n` summation bound, valid for any
//! summation order). `E_r` stores that coefficient with a factor 2 on gamma
//! and two `1 + 2^-10` inflations, rounded up to f32; `|x|` is inflated by
//! `1 + 2^-10` on the GPU.
//!
//! Margin: `log_probs = fl16(l - c)` with `c >= l*` the log-sum-exp, so
//! `l* - c` lies in `[-ln V, 0]` (plus rounding of `c`), inside `(-16, 16)`
//! where the F16 ulp is at most `2^-7`. A real gap above `2^-5` stays strict
//! after rounding, whatever `c` is, so the argmax is unique and equal to `l*`.

use anyhow::{Context, Result, ensure};

use crate::array::{self, Array, Dtype};

/// Rows screened per simdgroup and simdgroups per threadgroup.
const SCREEN_ROWS: i32 = 8;
const SCREEN_SIMDGROUPS: i32 = 8;
/// Rows of one screening threadgroup.
pub(crate) const SCREEN_TILE: i32 = SCREEN_ROWS * SCREEN_SIMDGROUPS;
/// `2^-5` certification margin plus slack for the f32 subtraction.
const MARGIN: &str = "0.031494140625f";

/// Host-side int8 copy and bound coefficients.
#[derive(Debug)]
pub(crate) struct Prepared {
    pub q: Vec<u8>,
    pub scale: Vec<f32>,
    pub bound: Vec<f32>,
}

pub(crate) fn supported(rows: i32, dims: i32) -> bool {
    rows > 0 && rows % SCREEN_TILE == 0 && dims > 0 && dims % 512 == 0
}

/// `gamma_n = n u / (1 - n u)` for f32 with `n = dims + 2` (products, scale
/// and every reduction order).
pub(crate) fn gamma(dims: usize) -> f64 {
    let nu = (dims as f64 + 2.0) * f64::powi(2.0, -24);
    nu / (1.0 - nu)
}

fn round_up_f32(value: f64) -> f32 {
    let rounded = value as f32;
    if f64::from(rounded) < value {
        f32::from_bits(rounded.to_bits() + 1)
    } else {
        rounded
    }
}

/// Quantizes F16 rows (bits, row-major `[rows, dims]`) to symmetric int8 and
/// computes each row's bound coefficient in f64.
pub(crate) fn prepare_rows(bits: &[u16], dims: usize) -> Result<Prepared> {
    ensure!(
        dims > 0 && bits.len().is_multiple_of(dims),
        "invalid certified head rows"
    );
    let rows = bits.len() / dims;
    let gamma = gamma(dims);
    let inflate = 1.0 + f64::powi(2.0, -10);
    let mut prepared = Prepared {
        q: vec![0; bits.len()],
        scale: Vec::with_capacity(rows),
        bound: Vec::with_capacity(rows),
    };
    let mut values = vec![0f64; dims];
    for (row, (source, q)) in bits
        .chunks_exact(dims)
        .zip(prepared.q.chunks_exact_mut(dims))
        .enumerate()
    {
        let mut amax = 0f64;
        for (value, &bits) in values.iter_mut().zip(source) {
            *value = half::f16::from_bits(bits).to_f64();
            ensure!(value.is_finite(), "certified head row {row} is not finite");
            amax = amax.max(value.abs());
        }
        let scale = if amax > 0.0 {
            (amax / 127.0) as f32
        } else {
            1.0
        };
        ensure!(
            scale.is_finite() && scale > 0.0,
            "certified head scale {row} is invalid"
        );
        let scale64 = f64::from(scale);
        let (mut delta, mut weight, mut approx) = (0f64, 0f64, 0f64);
        for (value, q) in values.iter().zip(q.iter_mut()) {
            let level = (value / scale64).round().clamp(-127.0, 127.0);
            *q = (level as i8) as u8;
            let restored = scale64 * level;
            delta += (value - restored) * (value - restored);
            weight += value * value;
            approx += restored * restored;
        }
        let bound =
            (delta.sqrt() * inflate + 2.0 * gamma * (weight.sqrt() + approx.sqrt())) * inflate;
        let bound = round_up_f32(bound);
        ensure!(bound.is_finite(), "certified head bound {row} is invalid");
        prepared.scale.push(scale);
        prepared.bound.push(bound);
    }
    Ok(prepared)
}

/// GPU copy of `Prepared` for one dense F16 head.
pub(crate) struct CertifiedHead {
    q: Array,
    scale: Array,
    bound: Array,
    rows: i32,
    dims: i32,
}

impl CertifiedHead {
    pub(crate) fn build(weight: &Array) -> Result<Self> {
        let prepared = Self::prepare(weight)?;
        Self::from_prepared(&prepared, weight.shape()[0], weight.shape()[1])
    }

    /// CPU preparation of an F16 `[rows, dims]` head, in chunks and threads.
    pub(crate) fn prepare(weight: &Array) -> Result<Prepared> {
        ensure!(
            weight.dtype() == Dtype::Float16 && weight.shape().len() == 2,
            "certified head requires F16 [rows, dims] weights"
        );
        let (rows, dims) = (weight.shape()[0], weight.shape()[1]);
        ensure!(supported(rows, dims), "unsupported certified head shape");
        let width = usize::try_from(dims)?;
        let mut prepared = Prepared {
            q: Vec::with_capacity(usize::try_from(rows)? * width),
            scale: Vec::with_capacity(usize::try_from(rows)?),
            bound: Vec::with_capacity(usize::try_from(rows)?),
        };
        const CHUNK: i32 = 16384;
        let mut begin = 0;
        while begin < rows {
            let end = (begin + CHUNK).min(rows);
            let bits = weight.slice(0, begin, end)?.to_f16_bits()?;
            let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
            let part = bits.len().div_ceil(threads * width) * width;
            let pieces = std::thread::scope(|scope| {
                bits.chunks(part)
                    .map(|chunk| scope.spawn(move || prepare_rows(chunk, width)))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|handle| handle.join().expect("certified head worker panicked"))
                    .collect::<Result<Vec<_>>>()
            })?;
            for piece in pieces {
                prepared.q.extend_from_slice(&piece.q);
                prepared.scale.extend_from_slice(&piece.scale);
                prepared.bound.extend_from_slice(&piece.bound);
            }
            begin = end;
        }
        Ok(prepared)
    }

    pub(crate) fn from_prepared(prepared: &Prepared, rows: i32, dims: i32) -> Result<Self> {
        ensure!(supported(rows, dims), "unsupported certified head shape");
        let q = Array::from_bytes(&prepared.q, &[rows, dims / 4], Dtype::UInt32)?;
        let scale = Array::from_f32(&prepared.scale, &[rows])?;
        let bound = Array::from_f32(&prepared.bound, &[rows])?;
        for array in [&q, &scale, &bound] {
            array.eval()?;
        }
        Ok(Self {
            q,
            scale,
            bound,
            rows,
            dims,
        })
    }

    #[cfg(test)]
    pub(crate) fn retained_bytes(&self) -> Result<usize> {
        Ok(self.q.byte_len()? + self.scale.byte_len()? + self.bound.byte_len()?)
    }

    fn check_input(&self, x: &Array) -> Result<()> {
        ensure!(
            x.dtype() == Dtype::Float16
                && x.shape().iter().product::<i32>() == self.dims
                && x.shape().last() == Some(&self.dims),
            "certified head expects one F16 row of {} values",
            self.dims
        );
        Ok(())
    }

    /// Kernel 1: `[upper [rows], partial maxima of lower [rows/64], |x| [1]]`.
    pub(crate) fn screen(&self, x: &Array) -> Result<Vec<Array>> {
        self.check_input(x)?;
        let source = format!(
            r#"
                constexpr uint K = {dims}u;
                constexpr uint ITS = K / 512u;
                constexpr uint ROWS = {rows_per_sg}u;
                uint lane = thread_index_in_simdgroup;
                uint sg = simdgroup_index_in_threadgroup;
                uint tg = threadgroup_position_in_grid.x;
                float xv[ITS][16];
                float squares = 0.0f;
                for (uint it = 0u; it < ITS; ++it) {{
                    for (uint j = 0u; j < 16u; ++j) {{
                        float value = float(x[it * 512u + lane * 16u + j]);
                        xv[it][j] = value;
                        squares = fma(value, value, squares);
                    }}
                }}
                squares = simd_sum(squares);
                float norm = metal::precise::sqrt(squares) * 1.0009765625f;
                float lower_max = -INFINITY;
                uint base = (tg * {sgs}u + sg) * ROWS;
                for (uint r = 0u; r < ROWS; ++r) {{
                    uint row = base + r;
                    const device uint4* words =
                        reinterpret_cast<const device uint4*>(q + row * (K / 4u));
                    float acc = 0.0f;
                    for (uint it = 0u; it < ITS; ++it) {{
                        uint4 packed = words[it * 32u + lane];
                        for (uint c = 0u; c < 4u; ++c) {{
                            float4 f = float4(as_type<char4>(packed[c]));
                            acc = fma(f.x, xv[it][c * 4u], acc);
                            acc = fma(f.y, xv[it][c * 4u + 1u], acc);
                            acc = fma(f.z, xv[it][c * 4u + 2u], acc);
                            acc = fma(f.w, xv[it][c * 4u + 3u], acc);
                        }}
                    }}
                    acc = simd_sum(acc);
                    float approx = acc * scale[row];
                    float error = norm * bound[row];
                    if (lane == 0u) {{
                        upper[row] = approx + error;
                    }}
                    lower_max = max(lower_max, approx - error);
                }}
                threadgroup float maxima[{sgs}];
                if (lane == 0u) {{
                    maxima[sg] = lower_max;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (thread_index_in_threadgroup == 0u) {{
                    float best = maxima[0];
                    for (uint s = 1u; s < {sgs}u; ++s) {{
                        best = max(best, maxima[s]);
                    }}
                    partial[tg] = best;
                    if (tg == 0u) {{
                        x_norm[0] = norm;
                    }}
                }}
            "#,
            dims = self.dims,
            rows_per_sg = SCREEN_ROWS,
            sgs = SCREEN_SIMDGROUPS,
        );
        let tiles = self.rows / SCREEN_TILE;
        array::metal_kernel(
            &format!("mlxl3_lfm3_cert_screen_{}_{}_v1", self.rows, self.dims),
            &["x", "q", "scale", "bound"],
            &["upper", "partial", "x_norm"],
            "",
            &source,
            &[x, &self.q, &self.scale, &self.bound],
            &[vec![self.rows], vec![tiles], vec![1]],
            &[Dtype::Float32, Dtype::Float32, Dtype::Float32],
            [tiles * SCREEN_SIMDGROUPS * 32, 1, 1],
            [SCREEN_SIMDGROUPS * 32, 1, 1],
        )
    }

    /// Kernel 2: the best lower bound `[1]`.
    pub(crate) fn threshold(&self, partial: &Array) -> Result<Array> {
        let count = partial.shape().iter().product::<i32>();
        let source = format!(
            r#"
                uint tid = thread_position_in_threadgroup.x;
                float best = -INFINITY;
                for (uint i = tid; i < {count}u; i += 1024u) {{
                    best = max(best, partial[i]);
                }}
                best = simd_max(best);
                threadgroup float maxima[32];
                if (thread_index_in_simdgroup == 0u) {{
                    maxima[simdgroup_index_in_threadgroup] = best;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {{
                    float value = maxima[0];
                    for (uint s = 1u; s < 32u; ++s) {{
                        value = max(value, maxima[s]);
                    }}
                    threshold[0] = value;
                }}
            "#
        );
        Ok(array::metal_kernel(
            &format!("mlxl3_lfm3_cert_threshold_{count}_v1"),
            &["partial"],
            &["threshold"],
            "",
            &source,
            &[partial],
            &[vec![1]],
            &[Dtype::Float32],
            [1024, 1, 1],
            [1024, 1, 1],
        )?
        .remove(0))
    }

    /// Kernel 3: exact F16 logits of every 4-row group holding a candidate
    /// (`upper >= threshold`), `-inf` elsewhere. The arithmetic block is the
    /// one of `Projection::forward_dense_rows_exact` (MLX T1 GEMV order).
    pub(crate) fn exact(
        &self,
        weight: &Array,
        x: &Array,
        upper: &Array,
        threshold: &Array,
    ) -> Result<Array> {
        self.check_input(x)?;
        ensure!(
            weight.shape() == [self.rows, self.dims] && weight.dtype() == Dtype::Float16,
            "certified head weight shape changed"
        );
        let source = format!(
            r#"
                constexpr uint INPUT_DIMS = {input}u;
                uint output = threadgroup_position_in_grid.x * 16u
                    + simdgroup_index_in_threadgroup * 4u;
                uint lane = thread_index_in_simdgroup;
                float limit = threshold[0];
                bool candidate[4];
                bool any = false;
                for (uint m = 0u; m < 4u; ++m) {{
                    candidate[m] = upper[output + m] >= limit;
                    any = any || candidate[m];
                }}
                if (!any) {{
                    if (lane == 0u) {{
                        for (uint m = 0u; m < 4u; ++m) {{
                            y[output + m] = half(-INFINITY);
                        }}
                    }}
                    return;
                }}
                uint row = 0u;
                float result[4] = {{0.0f, 0.0f, 0.0f, 0.0f}};
                uint k = lane * 4u;
                for (uint block = 0u; block < INPUT_DIMS / 128u; ++block) {{
                    float coeff[4];
                    for (uint n = 0u; n < 4u; ++n) {{
                        coeff[n] = float(x[row * INPUT_DIMS + k + n]);
                    }}
                    for (uint m = 0u; m < 4u; ++m) {{
                        for (uint n = 0u; n < 4u; ++n) {{
                            result[m] += float(weight[(output + m) * INPUT_DIMS + k + n])
                                * coeff[n];
                        }}
                    }}
                    k += 128u;
                }}
                for (ushort offset = 16u; offset >= 1u; offset >>= 1u) {{
                    for (uint m = 0u; m < 4u; ++m) {{
                        result[m] += simd_shuffle_down(result[m], offset);
                    }}
                }}
                if (lane == 0u) {{
                    for (uint m = 0u; m < 4u; ++m) {{
                        y[output + m] = candidate[m] ? half(result[m]) : half(-INFINITY);
                    }}
                }}
            "#,
            input = self.dims,
        );
        Ok(array::metal_kernel(
            &format!("mlxl3_lfm3_cert_exact_{}_{}_v1", self.dims, self.rows),
            &["x", "weight", "upper", "threshold"],
            &["y"],
            "",
            &source,
            &[x, weight, upper, threshold],
            &[vec![self.rows]],
            &[Dtype::Float16],
            [(self.rows / 16) * 128, 1, 1],
            [128, 1, 1],
        )?
        .remove(0))
    }

    /// Kernel 4: `[id, certified]` as UInt32.
    pub(crate) fn certify(&self, y: &Array, threshold: &Array, norm: &Array) -> Result<Array> {
        let source = format!(
            r#"
                constexpr uint V = {rows}u;
                uint tid = thread_position_in_threadgroup.x;
                float best = -INFINITY;
                float second = -INFINITY;
                uint index = 0xffffffffu;
                for (uint i = tid; i < V; i += 1024u) {{
                    float value = float(y[i]);
                    if (value > best) {{
                        second = best;
                        best = value;
                        index = i;
                    }} else if (value > second) {{
                        second = value;
                    }}
                }}
                threadgroup float bests[1024];
                threadgroup float seconds[1024];
                threadgroup uint indices[1024];
                bests[tid] = best;
                seconds[tid] = second;
                indices[tid] = index;
                for (uint stride = 512u; stride >= 1u; stride >>= 1u) {{
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid < stride) {{
                        float a = bests[tid], b = bests[tid + stride];
                        uint ia = indices[tid], ib = indices[tid + stride];
                        float sa = seconds[tid], sb = seconds[tid + stride];
                        if (a > b || (a == b && ia < ib)) {{
                            seconds[tid] = max(sa, b);
                        }} else {{
                            bests[tid] = b;
                            indices[tid] = ib;
                            seconds[tid] = max(sb, a);
                        }}
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {{
                    float top = bests[0];
                    float limit = threshold[0];
                    float floor_value = max(
                        seconds[0],
                        limit + fabs(limit) * 0.0009765625f + 9.5367431640625e-07f
                    );
                    bool finite = (as_type<uint>(top) & 0x7f800000u) != 0x7f800000u
                        && (as_type<uint>(limit) & 0x7f800000u) != 0x7f800000u
                        && (as_type<uint>(norm[0]) & 0x7f800000u) != 0x7f800000u
                        && (as_type<uint>(floor_value) & 0x7f800000u) != 0x7f800000u;
                    bool certified = finite && indices[0] < V
                        && top - floor_value > {margin};
                    out[0] = indices[0];
                    out[1] = certified ? 1u : 0u;
                }}
            "#,
            rows = self.rows,
            margin = MARGIN,
        );
        Ok(array::metal_kernel(
            &format!("mlxl3_lfm3_cert_certify_{}_v1", self.rows),
            &["y", "threshold", "norm"],
            &["out"],
            "",
            &source,
            &[y, threshold, norm],
            &[vec![2]],
            &[Dtype::UInt32],
            [1024, 1, 1],
            [1024, 1, 1],
        )?
        .remove(0))
    }

    /// The four kernels; one host read of `[id, certified]`.
    pub(crate) fn greedy(&self, weight: &Array, x: &Array) -> Result<(u32, bool)> {
        let mut screened = self.screen(x)?;
        let norm = screened.pop().context("missing norm")?;
        let partial = screened.pop().context("missing partial maxima")?;
        let upper = screened.pop().context("missing upper bounds")?;
        let threshold = self.threshold(&partial)?;
        let y = self.exact(weight, x, &upper, &threshold)?;
        let out = self.certify(&y, &threshold, &norm)?.to_u32()?;
        ensure!(
            out.len() == 2,
            "certified head returned {} values",
            out.len()
        );
        Ok((out[0], out[1] == 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(seed: u64, rows: usize, dims: usize) -> Vec<u16> {
        let mut state = seed;
        (0..rows * dims)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                // Mix scales, exact zeros and subnormal-sized values.
                let value = match i % 97 {
                    0 => 0.0,
                    1 => unit * 1e-6,
                    _ => unit * (0.02 + (i / dims % 7) as f64 * 0.03),
                };
                half::f16::from_f64(value).to_bits()
            })
            .collect()
    }

    /// Independent f64 oracle of `|w.x - w_hat.x| <= |x| E_r` on random x,
    /// plus the stored quantization levels and scales.
    #[test]
    fn prepared_rows_bound_the_exact_dot_products() {
        let dims = 512;
        let count = 40;
        let mut bits = rows(7, count, dims);
        // One all-zero row and one constant row.
        bits[..dims].fill(0);
        bits[dims..2 * dims].fill(half::f16::from_f32(0.25).to_bits());
        let prepared = prepare_rows(&bits, dims).unwrap();
        assert_eq!(prepared.q.len(), bits.len());
        assert_eq!((prepared.scale.len(), prepared.bound.len()), (count, count));
        assert_eq!(prepared.scale[0], 1.0);
        assert_eq!(prepared.bound[0], 0.0);
        assert!(prepared.q[..dims].iter().all(|&q| q == 0));
        assert!(prepared.q[dims..2 * dims].iter().all(|&q| q as i8 == 127));
        let mut state = 99u64;
        for trial in 0..64 {
            let x: Vec<f64> = (0..dims)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    let unit = (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                    half::f16::from_f64(unit * (1.0 + trial as f64)).to_f64()
                })
                .collect();
            let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            for row in 0..count {
                let range = row * dims..(row + 1) * dims;
                let scale = f64::from(prepared.scale[row]);
                let (mut exact, mut approx) = (0f64, 0f64);
                for ((&w, &q), &x) in bits[range.clone()].iter().zip(&prepared.q[range]).zip(&x) {
                    let level = f64::from(q as i8);
                    assert!((-127.0..=127.0).contains(&level));
                    exact += half::f16::from_bits(w).to_f64() * x;
                    approx += scale * level * x;
                }
                let bound = norm * f64::from(prepared.bound[row]);
                assert!(
                    (exact - approx).abs() <= bound,
                    "row {row}: |{exact} - {approx}| > {bound}"
                );
            }
        }
    }

    #[test]
    fn preparation_rejects_invalid_rows() {
        assert!(prepare_rows(&[0; 10], 4).is_err());
        assert!(prepare_rows(&[0; 8], 0).is_err());
        let infinite = half::f16::INFINITY.to_bits();
        assert!(prepare_rows(&[0, infinite, 0, 0], 4).is_err());
        assert!(supported(128000, 2048));
        assert!(!supported(128001, 2048));
        assert!(!supported(128000, 2000));
        assert!(!supported(0, 2048));
        assert!(gamma(2048) > 2050.0 * f64::powi(2.0, -24));
    }
}
