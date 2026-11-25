// INVARIANT-2026-11-24-INK2M-L1L4 (L1): MLX 0.32.2's affine `qmv_fast_impl` / `qmv_impl`
// (mlx/backend/metal/kernels/quantized.h, MIT, Copyright © 2023-2024 Apple Inc.) for ROWS input vectors in one
// threadgroup: each vector follows exactly the operations of the 1-vector kernel (same lane slices, same
// `load_vector` / `qdot` sequences, same `simd_sum`), while a block's weights, scales and biases are read once for
// all the vectors. Header defines: BITS (4 or 8), GROUP, IN_DIMS (K), OUT_DIMS (N), ROWS, FAST (1: qmv_fast,
// 0: qmv). Inputs `w` (u32), `scales`, `biases`, `x` ([ROWS, K], half); output `y` ([ROWS, N], half).
    constexpr int SIMD = 32;
    constexpr int pack_factor = 32 / BITS;
    constexpr int bytes_per_pack = 4;
    constexpr int packs_per_thread = FAST ? 2 : 1;
    constexpr int num_simdgroups = 2;
    constexpr int results_per_simdgroup = 4;
    constexpr int values_per_thread = pack_factor * packs_per_thread;
    constexpr int block_size = values_per_thread * SIMD;
    constexpr int scale_step_per_thread = GROUP / values_per_thread;
    const int in_vec_size = IN_DIMS;
    const int out_vec_size = OUT_DIMS;
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const uint3 tid = threadgroup_position_in_grid;
    typedef float U;
    const int in_vec_size_w = in_vec_size * bytes_per_pack / pack_factor;
    const int in_vec_size_g = in_vec_size / GROUP;
    const int out_row = tid.y * (num_simdgroups * results_per_simdgroup) + simd_gid * results_per_simdgroup;
#if FAST
    const int first_row = out_row;
#else
    if (out_row >= out_vec_size) {
        return;
    }
    const int first_row = min(out_vec_size - results_per_simdgroup, out_row);
#endif
    const device uint8_t* ws = (const device uint8_t*)w + first_row * in_vec_size_w
        + simd_lid * packs_per_thread * bytes_per_pack;
    const device half* sc = scales + first_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device half* bi = biases + first_row * in_vec_size_g + simd_lid / scale_step_per_thread;
    const device half* xs[ROWS];
    for (int m = 0; m < ROWS; ++m) {
        xs[m] = x + m * in_vec_size + simd_lid * values_per_thread;
    }
    thread U x_thread[ROWS][values_per_thread];
    thread U result[ROWS][results_per_simdgroup];
    for (int m = 0; m < ROWS; ++m) {
        for (int r = 0; r < results_per_simdgroup; ++r) {
            result[m][r] = 0;
        }
    }
    int k = 0;
#if FAST
    for (; k < in_vec_size; k += block_size) {
#else
    for (; k < in_vec_size - block_size; k += block_size) {
#endif
        U sum[ROWS];
        for (int m = 0; m < ROWS; ++m) {
            // load_vector<half, float, values_per_thread, BITS>
            U s = 0;
            const device half* xv = xs[m];
            thread U* xt = x_thread[m];
#if BITS == 8
            for (int i = 0; i < values_per_thread; i++) {
                s += xv[i];
                xt[i] = xv[i];
            }
#else
            for (int i = 0; i < values_per_thread; i += 4) {
                s += xv[i] + xv[i + 1] + xv[i + 2] + xv[i + 3];
                xt[i] = xv[i];
                xt[i + 1] = xv[i + 1] / 16.0f;
                xt[i + 2] = xv[i + 2] / 256.0f;
                xt[i + 3] = xv[i + 3] / 4096.0f;
            }
#endif
            sum[m] = s;
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            const device uint8_t* wl = ws + row * in_vec_size_w;
            const device half* sl = sc + row * in_vec_size_g;
            const device half* bl = bi + row * in_vec_size_g;
            U s = sl[0];
            U b = bl[0];
#if BITS == 8 && !FAST
            // qmv path: the weight bytes converted once for all the rows (uint8 -> float is exact); not in the fast
            // path, where the extra registers spill at 8 rows
            U wf[values_per_thread];
            for (int i = 0; i < values_per_thread; i++) {
                wf[i] = wl[i];
            }
#endif
            for (int m = 0; m < ROWS; ++m) {
                // qdot<float, values_per_thread, BITS>
                U accum = 0;
                const thread U* xt = x_thread[m];
#if BITS == 8 && !FAST
                for (int i = 0; i < values_per_thread; i++) {
                    accum += xt[i] * wf[i];
                }
#elif BITS == 8
                for (int i = 0; i < values_per_thread; i++) {
                    accum += xt[i] * wl[i];
                }
#else
                const device uint16_t* wsh = (const device uint16_t*)wl;
                for (int i = 0; i < (values_per_thread / 4); i++) {
                    accum +=
                        (xt[4 * i] * (wsh[i] & 0x000f) +
                         xt[4 * i + 1] * (wsh[i] & 0x00f0) +
                         xt[4 * i + 2] * (wsh[i] & 0x0f00) +
                         xt[4 * i + 3] * (wsh[i] & 0xf000));
                }
#endif
                result[m][row] += s * accum + sum[m] * b;
            }
        }
        ws += block_size * bytes_per_pack / pack_factor;
        sc += block_size / GROUP;
        bi += block_size / GROUP;
        for (int m = 0; m < ROWS; ++m) {
            xs[m] += block_size;
        }
    }
#if !FAST
    const int remaining = clamp(
        static_cast<int>(in_vec_size - k - simd_lid * values_per_thread), 0, values_per_thread);
    if (remaining > 0) {
        U sum[ROWS];
        for (int m = 0; m < ROWS; ++m) {
            // load_vector_safe<half, float, values_per_thread, BITS>(x, x_thread, remaining)
            U s = 0;
            const device half* xv = xs[m];
            thread U* xt = x_thread[m];
#if BITS == 8
            for (int i = 0; i < remaining; i++) {
                s += xv[i];
                xt[i] = xv[i];
            }
#else
            for (int i = 0; i < remaining; i += 4) {
                s += xv[i] + xv[i + 1] + xv[i + 2] + xv[i + 3];
                xt[i] = xv[i];
                xt[i + 1] = xv[i + 1] / 16.0f;
                xt[i + 2] = xv[i + 2] / 256.0f;
                xt[i + 3] = xv[i + 3] / 4096.0f;
            }
#endif
            for (int i = remaining; i < values_per_thread; i++) {
                xt[i] = 0;
            }
            sum[m] = s;
        }
        for (int row = 0; row < results_per_simdgroup; row++) {
            const device uint8_t* wl = ws + row * in_vec_size_w;
            const device half* sl = sc + row * in_vec_size_g;
            const device half* bl = bi + row * in_vec_size_g;
            U s = sl[0];
            U b = bl[0];
            for (int m = 0; m < ROWS; ++m) {
                // qdot_safe<float, values_per_thread, BITS>(w, x_thread, s, b, sum, remaining)
                U accum = 0;
                const thread U* xt = x_thread[m];
#if BITS == 8
                for (int i = 0; i < remaining; i++) {
                    accum += xt[i] * wl[i];
                }
#else
                const device uint16_t* wsh = (const device uint16_t*)wl;
                for (int i = 0; i < (remaining / 4); i++) {
                    accum +=
                        (xt[4 * i] * (wsh[i] & 0x000f) +
                         xt[4 * i + 1] * (wsh[i] & 0x00f0) +
                         xt[4 * i + 2] * (wsh[i] & 0x0f00) +
                         xt[4 * i + 3] * (wsh[i] & 0xf000));
                }
#endif
                result[m][row] += s * accum + sum[m] * b;
            }
        }
    }
#endif
    for (int m = 0; m < ROWS; ++m) {
        for (int row = 0; row < results_per_simdgroup; row++) {
            U total = simd_sum(result[m][row]);
            if (simd_lid == 0) {
                y[m * out_vec_size + first_row + row] = static_cast<half>(total);
            }
        }
    }
