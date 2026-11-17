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

    /// L1 kernel 2: the `k`-th largest of the per-tile maxima of the lower
    /// bounds `[1]`. Each of those tiles holds a distinct row whose lower
    /// bound reaches the value, so at least `k` rows have an exact logit at
    /// or above it: every row of the true top-`k` has `upper >=` it.
    pub(crate) fn kth_threshold(&self, partial: &Array, k: usize) -> Result<Array> {
        let count = partial.shape().iter().product::<i32>();
        ensure!(
            (1..=SAMPLE_MAX_K).contains(&k) && usize::try_from(count)? >= k,
            "invalid certified top-k request"
        );
        let padded = u32::try_from(count)?.next_power_of_two().max(1024);
        ensure!(padded <= 4096, "certified top-k head has too many tiles");
        let source = format!(
            r#"
                constexpr uint COUNT = {count}u;
                constexpr uint P = {padded}u;
                uint tid = thread_position_in_threadgroup.x;
                threadgroup float data[P];
                for (uint i = tid; i < P; i += 1024u) {{
                    data[i] = i < COUNT ? partial[i] : -INFINITY;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint size = 2u; size <= P; size <<= 1u) {{
                    for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {{
                        for (uint i = tid; i < P; i += 1024u) {{
                            uint j = i ^ stride;
                            if (j > i) {{
                                float a = data[i];
                                float b = data[j];
                                bool descending = (i & size) == 0u;
                                if (descending ? (a < b) : (a > b)) {{
                                    data[i] = b;
                                    data[j] = a;
                                }}
                            }}
                        }}
                        threadgroup_barrier(mem_flags::mem_threadgroup);
                    }}
                }}
                if (tid == 0u) {{
                    threshold[0] = data[{index}u];
                }}
            "#,
            index = k - 1,
        );
        Ok(array::metal_kernel(
            &format!("mlxl3_lfm3_cert_kth_{count}_{k}_v1"),
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

    /// L1 kernel 4: `[count, lse estimate, threshold, |x|, (id, F16 logit)…]`
    /// as UInt32. Candidates use the predicate of `exact`; the log-sum-exp
    /// estimate adds their exact logits to the screening products of the
    /// other rows.
    pub(crate) fn gather(
        &self,
        y: &Array,
        upper: &Array,
        threshold: &Array,
        norm: &Array,
    ) -> Result<Array> {
        let source = format!(
            r#"
                constexpr uint V = {rows}u;
                constexpr uint CAP = {cap}u;
                uint tid = thread_position_in_threadgroup.x;
                threadgroup atomic_uint counter;
                if (tid == 0u) {{
                    atomic_store_explicit(&counter, 0u, memory_order_relaxed);
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                float limit = threshold[0];
                float scale = norm[0];
                float m = -INFINITY;
                float s = 0.0f;
                for (uint i = tid; i < V; i += 1024u) {{
                    bool candidate = upper[i] >= limit;
                    half h = y[i];
                    float v = candidate ? float(h) : upper[i] - scale * bound[i];
                    if (isfinite(v)) {{
                        if (v > m) {{
                            s = s * metal::precise::exp(m - v) + 1.0f;
                            m = v;
                        }} else {{
                            s += metal::precise::exp(v - m);
                        }}
                    }}
                    if (candidate) {{
                        uint slot = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
                        if (slot < CAP) {{
                            out[4u + 2u * slot] = i;
                            out[5u + 2u * slot] = uint(as_type<ushort>(h));
                        }}
                    }}
                }}
                threadgroup float maxima[1024];
                threadgroup float sums[1024];
                maxima[tid] = m;
                sums[tid] = s;
                for (uint stride = 512u; stride >= 1u; stride >>= 1u) {{
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid < stride) {{
                        float ma = maxima[tid];
                        float mb = maxima[tid + stride];
                        float top = max(ma, mb);
                        float total = 0.0f;
                        if (ma > -INFINITY) {{
                            total += sums[tid] * metal::precise::exp(ma - top);
                        }}
                        if (mb > -INFINITY) {{
                            total += sums[tid + stride] * metal::precise::exp(mb - top);
                        }}
                        maxima[tid] = top;
                        sums[tid] = total;
                    }}
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (tid == 0u) {{
                    out[0] = atomic_load_explicit(&counter, memory_order_relaxed);
                    out[1] = as_type<uint>(maxima[0] + metal::precise::log(sums[0]));
                    out[2] = as_type<uint>(limit);
                    out[3] = as_type<uint>(scale);
                }}
            "#,
            rows = self.rows,
            cap = SAMPLE_CAPACITY,
        );
        Ok(array::metal_kernel(
            &format!(
                "mlxl3_lfm3_cert_gather_{}_{}_v1",
                self.rows, SAMPLE_CAPACITY
            ),
            &["y", "upper", "bound", "threshold", "norm"],
            &["out"],
            "",
            &source,
            &[y, upper, &self.bound, threshold, norm],
            &[vec![4 + 2 * SAMPLE_CAPACITY]],
            &[Dtype::UInt32],
            [1024, 1, 1],
            [1024, 1, 1],
        )?
        .remove(0))
    }

    /// L1: exact F16 logits of every row that may belong to the top-`k`,
    /// with the threshold and a log-sum-exp estimate; one host read.
    pub(crate) fn sample(&self, weight: &Array, x: &Array, k: usize) -> Result<SampledRows> {
        let mut screened = self.screen(x)?;
        let norm = screened.pop().context("missing norm")?;
        let partial = screened.pop().context("missing partial maxima")?;
        let upper = screened.pop().context("missing upper bounds")?;
        let threshold = self.kth_threshold(&partial, k)?;
        let y = self.exact(weight, x, &upper, &threshold)?;
        SampledRows::parse(&self.gather(&y, &upper, &threshold, &norm)?.to_u32()?)
    }
}

/// Screening copy kept by a model: int8 (default greedy head and L1) or
/// int4 (L2, greedy only).
pub(crate) enum Screen {
    Int8(CertifiedHead),
    Int4(Int4Head),
}

impl Screen {
    pub(crate) fn greedy(&self, weight: &Array, x: &Array) -> Result<(u32, bool)> {
        match self {
            Self::Int8(head) => head.greedy(weight, x),
            Self::Int4(head) => head.greedy(weight, x),
        }
    }

    pub(crate) fn is_int4(&self) -> bool {
        matches!(self, Self::Int4(_))
    }
}

/// Most candidate rows the sampling head returns before falling back.
pub(crate) const SAMPLE_CAPACITY: i32 = 2048;
/// Largest `top_k` the sampling head certifies.
pub(crate) const SAMPLE_MAX_K: usize = 256;

/// Host copy of the L1 candidates.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SampledRows {
    pub ids: Vec<u32>,
    /// F16 bits of the exact logits, in the order of `ids`.
    pub logits: Vec<u16>,
    pub lse: f32,
    pub threshold: f32,
    /// More candidates than `SAMPLE_CAPACITY`: the list is incomplete.
    pub overflow: bool,
}

impl SampledRows {
    pub(crate) fn parse(out: &[u32]) -> Result<Self> {
        let capacity = usize::try_from(SAMPLE_CAPACITY)?;
        ensure!(
            out.len() == 4 + 2 * capacity,
            "certified sampling head returned {} values",
            out.len()
        );
        let count = usize::try_from(out[0])?;
        let kept = count.min(capacity);
        let pairs = &out[4..4 + 2 * kept];
        Ok(Self {
            ids: pairs.iter().step_by(2).copied().collect(),
            logits: pairs
                .iter()
                .skip(1)
                .step_by(2)
                .map(|&bits| u16::try_from(bits))
                .collect::<std::result::Result<_, _>>()
                .context("certified sampling head returned an invalid logit")?,
            lse: f32::from_bits(out[1]),
            threshold: f32::from_bits(out[2]),
            overflow: count > capacity,
        })
    }
}

/// `2^-5` margin (F16 rounding of the logit and of the log probability).
const SAMPLE_MARGIN: f32 = 0.031_494_14;

/// L1: the full-vocabulary log probabilities `Sampler::distribution`
/// receives, with `-inf` outside the candidates, or `None` when the
/// candidates do not certify the kept set.
///
/// Values are `fl16(l - fl16(lse))`, the arithmetic of `log_probs`; they
/// equal the current path when the estimate rounds to the same F16 log-sum-
/// exp (measured, not proved: F1). Every non-candidate has an exact logit
/// below the threshold, and penalties only lower values, so if the `k`-th
/// best penalized candidate clears that bound by the margin, the kept set is
/// the one of the full row. A tie at the boundary would let the unstable
/// selection of the full row keep another of the tied rows (with a weight
/// that is tiny but not zero), so sampled rows with a boundary tie fall back.
pub(crate) fn certified_values(
    rows: &SampledRows,
    vocab: usize,
    sampler: &crate::sampling::Sampler,
    history: &crate::sampling::History,
) -> Option<Vec<f32>> {
    let k = sampler.certified_k(SAMPLE_MAX_K)?;
    if rows.overflow
        || rows.ids.len() != rows.logits.len()
        || !rows.lse.is_finite()
        || !rows.threshold.is_finite()
    {
        return None;
    }
    let lse = half::f16::from_f32(rows.lse);
    if !lse.is_finite() {
        return None;
    }
    let lse = lse.to_f32();
    let mut values = vec![f32::NEG_INFINITY; vocab];
    for (&id, &bits) in rows.ids.iter().zip(&rows.logits) {
        let logit = half::f16::from_bits(bits).to_f32();
        let slot = values.get_mut(usize::try_from(id).ok()?)?;
        if !logit.is_finite() || *slot != f32::NEG_INFINITY {
            return None;
        }
        *slot = half::f16::from_f32(logit - lse).to_f32();
        if !slot.is_finite() {
            return None;
        }
    }
    let mut penalized = values.clone();
    sampler.apply_penalties(&mut penalized, history, &[]);
    let mut kept: Vec<f32> = rows.ids.iter().map(|&id| penalized[id as usize]).collect();
    kept.sort_unstable_by(|left, right| right.total_cmp(left));
    let boundary = *kept.get(k - 1)?;
    let floor = rows.threshold + rows.threshold.abs() * 0.000_976_562_5 + 9.536_743e-7 - lse;
    if !(boundary.is_finite() && boundary - floor > SAMPLE_MARGIN) {
        return None;
    }
    if !sampler.deterministic() && kept.get(k) == Some(&boundary) {
        return None;
    }
    Some(values)
}

/// L2 (greedy default; `MLXL3_LFM3_CERT_SCREEN=int8` keeps int8): an int4 screen of the F16 head with
/// one F16 scale per 64 weights and the bound of `CertifiedHead`, and a
/// two-stage threshold: the exact logit of the row with the best lower
/// bound, minus `INT4_DELTA`. Certified IDs are the IDs of the full head by
/// the argument of `CertifiedHead::certify`, which holds for any threshold.
pub(crate) struct Int4Head {
    q: Array,
    scale: Array,
    bound: Array,
    rows: i32,
    dims: i32,
    inner: CertifiedHead,
}

/// Weights per int4 scale.
const INT4_GROUP: usize = 64;
/// Distance kept below the provisional exact logit (above the `2^-5`
/// certification margin and the F16 slack of `certify`).
const INT4_DELTA: &str = "0.125f";

/// Host-side int4 copy: two levels per byte (low nibble first, level + 8),
/// F16 scales `[rows, dims / 64]` and bound coefficients.
#[derive(Debug)]
pub(crate) struct Int4Prepared {
    pub q: Vec<u8>,
    pub scale: Vec<u16>,
    pub bound: Vec<f32>,
}

pub(crate) fn prepare_int4_rows(bits: &[u16], dims: usize) -> Result<Int4Prepared> {
    ensure!(
        dims > 0 && dims.is_multiple_of(INT4_GROUP) && bits.len().is_multiple_of(dims),
        "invalid int4 certified head rows"
    );
    let rows = bits.len() / dims;
    let gamma = gamma(dims);
    let inflate = 1.0 + f64::powi(2.0, -10);
    let mut prepared = Int4Prepared {
        q: vec![0; bits.len() / 2],
        scale: Vec::with_capacity(rows * dims / INT4_GROUP),
        bound: Vec::with_capacity(rows),
    };
    for (row, (source, packed)) in bits
        .chunks_exact(dims)
        .zip(prepared.q.chunks_exact_mut(dims / 2))
        .enumerate()
    {
        let (mut delta, mut weight, mut approx) = (0f64, 0f64, 0f64);
        for (group, (values, packed)) in source
            .as_chunks::<INT4_GROUP>()
            .0
            .iter()
            .zip(packed.as_chunks_mut::<{ INT4_GROUP / 2 }>().0)
            .enumerate()
        {
            let mut amax = 0f64;
            for &bits in values {
                let value = half::f16::from_bits(bits).to_f64();
                ensure!(
                    value.is_finite(),
                    "int4 certified head row {row} is not finite"
                );
                amax = amax.max(value.abs());
            }
            // The stored F16 scale is the one the bound uses.
            let scale = if amax > 0.0 {
                half::f16::from_f64(amax / 7.0)
            } else {
                half::f16::ONE
            };
            ensure!(
                scale.is_finite() && scale.to_f64() > 0.0,
                "int4 certified head scale {row}/{group} is invalid"
            );
            let scale64 = scale.to_f64();
            for (index, &bits) in values.iter().enumerate() {
                let value = half::f16::from_bits(bits).to_f64();
                let level = (value / scale64).round().clamp(-7.0, 7.0);
                let nibble = (level as i8 + 8) as u8;
                packed[index / 2] |= if index % 2 == 0 { nibble } else { nibble << 4 };
                let restored = scale64 * level;
                delta += (value - restored) * (value - restored);
                weight += value * value;
                approx += restored * restored;
            }
            prepared.scale.push(scale.to_bits());
        }
        let bound =
            (delta.sqrt() * inflate + 2.0 * gamma * (weight.sqrt() + approx.sqrt())) * inflate;
        let bound = round_up_f32(bound);
        ensure!(
            bound.is_finite(),
            "int4 certified head bound {row} is invalid"
        );
        prepared.bound.push(bound);
    }
    Ok(prepared)
}

impl Int4Head {
    pub(crate) fn build(weight: &Array) -> Result<Self> {
        ensure!(
            weight.dtype() == Dtype::Float16 && weight.shape().len() == 2,
            "certified head requires F16 [rows, dims] weights"
        );
        let (rows, dims) = (weight.shape()[0], weight.shape()[1]);
        ensure!(supported(rows, dims), "unsupported certified head shape");
        let width = usize::try_from(dims)?;
        let mut prepared = Int4Prepared {
            q: Vec::with_capacity(usize::try_from(rows)? * width / 2),
            scale: Vec::with_capacity(usize::try_from(rows)? * width / INT4_GROUP),
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
                    .map(|chunk| scope.spawn(move || prepare_int4_rows(chunk, width)))
                    .collect::<Vec<_>>()
                    .into_iter()
                    .map(|handle| handle.join().expect("int4 certified head worker panicked"))
                    .collect::<Result<Vec<_>>>()
            })?;
            for piece in pieces {
                prepared.q.extend_from_slice(&piece.q);
                prepared.scale.extend_from_slice(&piece.scale);
                prepared.bound.extend_from_slice(&piece.bound);
            }
            begin = end;
        }
        Self::from_prepared(&prepared, rows, dims)
    }

    pub(crate) fn from_prepared(prepared: &Int4Prepared, rows: i32, dims: i32) -> Result<Self> {
        ensure!(supported(rows, dims), "unsupported certified head shape");
        let q = Array::from_bytes(&prepared.q, &[rows, dims / 8], Dtype::UInt32)?;
        let scale = Array::from_f16_bits(&prepared.scale, &[rows, dims / 64])?;
        let bound = Array::from_f32(&prepared.bound, &[rows])?;
        for array in [&q, &scale, &bound] {
            array.eval()?;
        }
        // The exact and certify kernels only use the shape of `inner`.
        let empty = Array::from_f32(&[0.0], &[1])?;
        Ok(Self {
            q,
            scale,
            bound,
            rows,
            dims,
            inner: CertifiedHead {
                q: empty.clone(),
                scale: empty.clone(),
                bound: empty,
                rows,
                dims,
            },
        })
    }

    #[cfg(test)]
    pub(crate) fn retained_bytes(&self) -> Result<usize> {
        Ok(self.q.byte_len()? + self.scale.byte_len()? + self.bound.byte_len()?)
    }

    /// `[upper [rows], best lower per tile [rows/64], its row [rows/64], |x|]`.
    pub(crate) fn screen(&self, x: &Array) -> Result<Vec<Array>> {
        self.inner.check_input(x)?;
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
                uint lower_row = 0xffffffffu;
                uint base = (tg * {sgs}u + sg) * ROWS;
                for (uint r = 0u; r < ROWS; ++r) {{
                    uint row = base + r;
                    const device uint2* words =
                        reinterpret_cast<const device uint2*>(q + row * (K / 8u));
                    const device half* scales = scale + row * (K / 64u);
                    float acc = 0.0f;
                    for (uint it = 0u; it < ITS; ++it) {{
                        uint2 packed = words[it * 32u + lane];
                        float part = 0.0f;
                        for (uint j = 0u; j < 8u; ++j) {{
                            float lo = float(int((packed.x >> (4u * j)) & 0xfu) - 8);
                            float hi = float(int((packed.y >> (4u * j)) & 0xfu) - 8);
                            part = fma(lo, xv[it][j], part);
                            part = fma(hi, xv[it][8u + j], part);
                        }}
                        acc = fma(part, float(scales[it * 8u + lane / 4u]), acc);
                    }}
                    acc = simd_sum(acc);
                    float error = norm * bound[row];
                    if (lane == 0u) {{
                        upper[row] = acc + error;
                    }}
                    float lower = acc - error;
                    if (lower > lower_max) {{
                        lower_max = lower;
                        lower_row = row;
                    }}
                }}
                threadgroup float maxima[{sgs}];
                threadgroup uint indices[{sgs}];
                if (lane == 0u) {{
                    maxima[sg] = lower_max;
                    indices[sg] = lower_row;
                }}
                threadgroup_barrier(mem_flags::mem_threadgroup);
                if (thread_index_in_threadgroup == 0u) {{
                    float best = maxima[0];
                    uint index = indices[0];
                    for (uint s = 1u; s < {sgs}u; ++s) {{
                        if (maxima[s] > best) {{
                            best = maxima[s];
                            index = indices[s];
                        }}
                    }}
                    partial[tg] = best;
                    partial_row[tg] = index;
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
            &format!("mlxl3_lfm3_cert_screen4_{}_{}_v1", self.rows, self.dims),
            &["x", "q", "scale", "bound"],
            &["upper", "partial", "partial_row", "x_norm"],
            "",
            &source,
            &[x, &self.q, &self.scale, &self.bound],
            &[vec![self.rows], vec![tiles], vec![tiles], vec![1]],
            &[
                Dtype::Float32,
                Dtype::Float32,
                Dtype::UInt32,
                Dtype::Float32,
            ],
            [tiles * SCREEN_SIMDGROUPS * 32, 1, 1],
            [SCREEN_SIMDGROUPS * 32, 1, 1],
        )
    }

    /// Exact F16 logit of the row with the best lower bound (arithmetic of
    /// `CertifiedHead::exact`, one simdgroup), minus `INT4_DELTA`.
    fn provisional(
        &self,
        weight: &Array,
        x: &Array,
        partial: &Array,
        partial_row: &Array,
    ) -> Result<Array> {
        let tiles = self.rows / SCREEN_TILE;
        let source = format!(
            r#"
                constexpr uint INPUT_DIMS = {input}u;
                constexpr uint TILES = {tiles}u;
                uint lane = thread_index_in_simdgroup;
                float best = -INFINITY;
                uint index = 0u;
                for (uint i = lane; i < TILES; i += 32u) {{
                    float value = partial[i];
                    if (value > best) {{
                        best = value;
                        index = partial_row[i];
                    }}
                }}
                for (ushort offset = 16u; offset >= 1u; offset >>= 1u) {{
                    float other = simd_shuffle_down(best, offset);
                    uint other_index = simd_shuffle_down(index, offset);
                    if (other > best) {{
                        best = other;
                        index = other_index;
                    }}
                }}
                index = simd_broadcast(index, 0);
                best = simd_broadcast(best, 0);
                if (!(best > -INFINITY) || index >= {rows}u) {{
                    if (lane == 0u) {{
                        threshold[0] = -INFINITY;
                    }}
                    return;
                }}
                uint output = index & ~3u;
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
                    threshold[0] = float(half(result[index - output])) - {delta};
                }}
            "#,
            input = self.dims,
            rows = self.rows,
            delta = INT4_DELTA,
        );
        Ok(array::metal_kernel(
            &format!("mlxl3_lfm3_cert_provisional_{}_{}_v1", self.dims, self.rows),
            &["x", "weight", "partial", "partial_row"],
            &["threshold"],
            "",
            &source,
            &[x, weight, partial, partial_row],
            &[vec![1]],
            &[Dtype::Float32],
            [32, 1, 1],
            [32, 1, 1],
        )?
        .remove(0))
    }

    /// Screen, provisional exact row, exact candidates, certificate; one host
    /// read of `[id, certified]`.
    pub(crate) fn greedy(&self, weight: &Array, x: &Array) -> Result<(u32, bool)> {
        ensure!(
            weight.shape() == [self.rows, self.dims] && weight.dtype() == Dtype::Float16,
            "certified head weight shape changed"
        );
        let mut screened = self.screen(x)?;
        let norm = screened.pop().context("missing norm")?;
        let partial_row = screened.pop().context("missing partial rows")?;
        let partial = screened.pop().context("missing partial maxima")?;
        let upper = screened.pop().context("missing upper bounds")?;
        let threshold = self.provisional(weight, x, &partial, &partial_row)?;
        let y = self.inner.exact(weight, x, &upper, &threshold)?;
        let out = self.inner.certify(&y, &threshold, &norm)?.to_u32()?;
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

    fn next(state: &mut u64) -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        (*state >> 11) as f64 / (1u64 << 53) as f64
    }

    /// L2: independent f64 oracle of the int4 bound, levels and packing.
    #[test]
    fn int4_rows_bound_the_exact_dot_products() {
        let dims = 512;
        let count = 24;
        let mut bits = rows(17, count, dims);
        bits[..dims].fill(0);
        bits[dims..2 * dims].fill(half::f16::from_f32(-0.5).to_bits());
        let prepared = prepare_int4_rows(&bits, dims).unwrap();
        assert_eq!(prepared.q.len(), bits.len() / 2);
        assert_eq!(prepared.scale.len(), count * dims / 64);
        assert_eq!(prepared.bound.len(), count);
        assert_eq!(prepared.bound[0], 0.0);
        assert!(prepared.q[..dims / 2].iter().all(|&b| b == 0x88));
        assert!(prepared.q[dims / 2..dims].iter().all(|&b| b == 0x11));
        let level = |row: usize, k: usize| {
            let byte = prepared.q[(row * dims + k) / 2];
            let nibble = if k.is_multiple_of(2) {
                byte & 15
            } else {
                byte >> 4
            };
            assert!((1..=15).contains(&nibble), "level nibble {nibble}");
            f64::from(nibble) - 8.0
        };
        let mut state = 7u64;
        for trial in 0..48 {
            let x: Vec<f64> = (0..dims)
                .map(|_| {
                    half::f16::from_f64((next(&mut state) - 0.5) * (1.0 + trial as f64)).to_f64()
                })
                .collect();
            let norm = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            for row in 0..count {
                let (mut exact, mut approx) = (0f64, 0f64);
                for (k, &x) in x.iter().enumerate() {
                    let scale =
                        half::f16::from_bits(prepared.scale[row * dims / 64 + k / 64]).to_f64();
                    exact += half::f16::from_bits(bits[row * dims + k]).to_f64() * x;
                    approx += scale * level(row, k) * x;
                }
                let bound = norm * f64::from(prepared.bound[row]);
                assert!(
                    (exact - approx).abs() <= bound,
                    "row {row}: |{exact} - {approx}| > {bound}"
                );
            }
        }
        // Mutation: the bound without the quantization term is too small.
        let weak = (0..count).any(|row| {
            let delta: f64 = (0..dims)
                .map(|k| {
                    let scale =
                        half::f16::from_bits(prepared.scale[row * dims / 64 + k / 64]).to_f64();
                    (half::f16::from_bits(bits[row * dims + k]).to_f64() - scale * level(row, k))
                        .powi(2)
                })
                .sum();
            f64::from(prepared.bound[row]) * 0.01 < delta.sqrt()
        });
        assert!(weak, "the int4 bound has no quantization term");
        assert!(prepare_int4_rows(&bits[..dims], 500).is_err());
        assert!(prepare_int4_rows(&bits[..100], 64).is_err());
        let infinite = half::f16::INFINITY.to_bits();
        assert!(prepare_int4_rows(&[infinite; 64], 64).is_err());
    }

    fn f16(value: f64) -> f32 {
        half::f16::from_f64(value).to_f32()
    }

    /// One synthetic row: F16 logits, an upper bound per row (`>=` the
    /// logit), the candidates of `threshold`, the exact log-sum-exp and the
    /// full log probabilities of `log_probs`.
    struct Row {
        rows: SampledRows,
        full: Vec<f32>,
    }

    fn row(state: &mut u64, vocab: usize, spread: f64, slack: f64, top: usize) -> Row {
        let logits: Vec<f32> = (0..vocab)
            .map(|_| f16((next(state) - 0.5) * spread))
            .collect();
        let upper: Vec<f64> = logits
            .iter()
            .map(|&l| f64::from(l) + next(state) * slack)
            .collect();
        let mut sorted: Vec<f32> = logits.clone();
        sorted.sort_by(|a, b| b.total_cmp(a));
        let threshold = sorted[top.min(vocab - 1)];
        let lse = logits
            .iter()
            .map(|&l| f64::from(l))
            .fold(f64::NEG_INFINITY, |m, l| {
                if m == f64::NEG_INFINITY {
                    l
                } else {
                    m.max(l) + (-(m - l).abs()).exp().ln_1p()
                }
            });
        let c16 = f16(lse);
        let full = logits.iter().map(|&l| f16(f64::from(l - c16))).collect();
        let (ids, logits): (Vec<u32>, Vec<u16>) = (0..vocab)
            .filter(|&i| upper[i] >= f64::from(threshold))
            .map(|i| (i as u32, half::f16::from_f32(logits[i]).to_bits()))
            .unzip();
        Row {
            rows: SampledRows {
                ids,
                logits,
                lse: lse as f32,
                threshold,
                overflow: false,
            },
            full,
        }
    }

    fn canonical(
        values: Vec<f32>,
        sampler: &crate::sampling::Sampler,
        history: &crate::sampling::History,
    ) -> crate::sampling::Distribution {
        sampler
            .distribution(values, history, &[])
            .unwrap()
            .canonical()
    }

    /// L1: whenever `certified_values` accepts a row, the canonical
    /// distribution equals the one of the full row (independent emulation of
    /// `log_probs`); deterministic fuzzing over samplers, penalties and gaps.
    #[test]
    fn certified_values_keep_the_full_distribution() {
        use crate::sampling::{History, Sampler};
        let mut state = 3u64;
        let samplers = [
            Sampler::new(0.2, 80, 1., 1.05, 0.).unwrap(),
            Sampler::new(0.1, 50, 1., 1.1, 0.).unwrap(),
            Sampler::new(0.7, 20, 0.9, 1., 0.3).unwrap(),
            Sampler::new(1.0, 3, 1., 1.3, 0.).unwrap(),
            Sampler::new(0., 40, 1., 1.2, 0.).unwrap(),
        ];
        let (mut accepted, mut rejected) = (0, 0);
        for case in 0..600 {
            let sampler = &samplers[case % samplers.len()];
            let k = sampler.certified_k(SAMPLE_MAX_K).unwrap();
            let vocab = 400 + case % 300;
            let spread = [4.0, 12.0, 40.0][case % 3];
            let slack = [0.05, 0.5, 3.0][case / 3 % 3];
            let top = k + case % 70;
            let row = row(&mut state, vocab, spread, slack, top);
            let history = History::from_tokens(
                &(0..case % 40)
                    .map(|_| (next(&mut state) * vocab as f64) as u32)
                    .collect::<Vec<_>>(),
            );
            match certified_values(&row.rows, vocab, sampler, &history) {
                Some(values) => {
                    accepted += 1;
                    for (index, &value) in values.iter().enumerate() {
                        if value != f32::NEG_INFINITY {
                            assert_eq!(value.to_bits(), row.full[index].to_bits());
                        }
                    }
                    assert_eq!(
                        canonical(values, sampler, &history),
                        canonical(row.full.clone(), sampler, &history),
                        "case {case}"
                    );
                }
                None => rejected += 1,
            }
        }
        assert!(accepted > 100 && rejected > 20, "{accepted} {rejected}");
    }

    #[test]
    fn certified_values_reject_unsafe_rows() {
        use crate::sampling::{History, Sampler};
        let sampler = Sampler::new(0.7, 2, 1., 1., 0.).unwrap();
        let history = History::default();
        let base = SampledRows {
            ids: vec![4, 9, 1],
            logits: [3.0f32, 2.0, 1.0]
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect(),
            lse: 3.5,
            threshold: 0.5,
            overflow: false,
        };
        let values = certified_values(&base, 10, &sampler, &history).unwrap();
        assert_eq!(values[4], f16(f64::from(3.0f32 - f16(3.5))));
        assert_eq!(values.iter().filter(|v| v.is_finite()).count(), 3);
        let reject = |rows: SampledRows, sampler: &Sampler| {
            assert!(certified_values(&rows, 10, sampler, &history).is_none());
        };
        // The 2nd best candidate is too close to the bound of the others.
        reject(
            SampledRows {
                threshold: 1.98,
                ..base.clone()
            },
            &sampler,
        );
        reject(
            SampledRows {
                overflow: true,
                ..base.clone()
            },
            &sampler,
        );
        reject(
            SampledRows {
                ids: vec![4, 4, 1],
                ..base.clone()
            },
            &sampler,
        );
        reject(
            SampledRows {
                ids: vec![4, 10, 1],
                ..base.clone()
            },
            &sampler,
        );
        reject(
            SampledRows {
                lse: f32::NAN,
                ..base.clone()
            },
            &sampler,
        );
        reject(
            SampledRows {
                lse: 1e6,
                ..base.clone()
            },
            &sampler,
        );
        reject(
            SampledRows {
                ids: vec![4, 9],
                ..base.clone()
            },
            &sampler,
        );
        // Fewer than k candidates, top-k 0 and negative presence.
        reject(
            SampledRows {
                ids: vec![4],
                logits: vec![base.logits[0]],
                ..base.clone()
            },
            &sampler,
        );
        reject(base.clone(), &Sampler::new(0.7, 0, 1., 1., 0.).unwrap());
        reject(base.clone(), &Sampler::new(0.7, 2, 1., 1., -0.1).unwrap());
        // A tie at the boundary: rejected when sampled, kept for argmax.
        let tied = SampledRows {
            logits: [3.0f32, 2.0, 2.0]
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect(),
            ..base.clone()
        };
        reject(tied.clone(), &sampler);
        let argmax = Sampler::new(0., 2, 1., 1.1, 0.).unwrap();
        assert!(certified_values(&tied, 10, &argmax, &history).is_some());
        // Even a tie with a negligible weight falls back (the full row may
        // keep the other tied row: GPU oracle, LFM3 T 0.1, weight e^-81).
        let far = SampledRows {
            logits: [30.0f32, 2.0, 2.0]
                .iter()
                .map(|&v| half::f16::from_f32(v).to_bits())
                .collect(),
            lse: 30.0,
            ..base.clone()
        };
        reject(far, &Sampler::new(0.1, 2, 1., 1., 0.).unwrap());
    }

    #[test]
    fn sampled_rows_parse_the_kernel_layout() {
        let capacity = usize::try_from(SAMPLE_CAPACITY).unwrap();
        let mut out = vec![0u32; 4 + 2 * capacity];
        out[0] = 2;
        out[1] = 1.5f32.to_bits();
        out[2] = (-2.0f32).to_bits();
        out[4..8].copy_from_slice(&[7, 0x3c00, 3, 0x4000]);
        let rows = SampledRows::parse(&out).unwrap();
        assert_eq!(rows.ids, [7, 3]);
        assert_eq!(rows.logits, [0x3c00, 0x4000]);
        assert_eq!(
            (rows.lse, rows.threshold, rows.overflow),
            (1.5, -2.0, false)
        );
        out[0] = u32::try_from(capacity).unwrap() + 1;
        let rows = SampledRows::parse(&out).unwrap();
        assert!(rows.overflow && rows.ids.len() == capacity);
        assert!(SampledRows::parse(&out[1..]).is_err());
        out[0] = 1;
        out[5] = 0x1_0000;
        assert!(SampledRows::parse(&out).is_err());
    }
}
