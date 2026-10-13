// GEMV geometry and reduction order: MLX 0.32.2, Apple Inc., MIT license.
// Half weight loads avoid materializing a full float weight matrix per token.
uint lane = thread_index_in_simdgroup;
uint sg = simdgroup_index_in_threadgroup;
uint group_n = sg % BN;
uint output = threadgroup_position_in_grid.x * (BM * 4) + (sg / BN) * 4;
float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#if ROWS == 2
float second[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#endif
uint column = (group_n * 32 + lane) * 4;
for (uint block = 0; block < (K + BN * 128 - 1) / (BN * 128); ++block) {
    float coeff[4];
    for (uint n = 0; n < 4; ++n) {
        coeff[n] = column + n < K ? x[column + n] : 0.0f;
    }
#if ROWS == 2
    float coeff_second[4];
    for (uint n = 0; n < 4; ++n) {
        coeff_second[n] = column + n < K ? x[column + n] : 0.0f;
    }
#endif
    for (uint m = 0; m < 4; ++m) {
        for (uint n = 0; n < 4; ++n) {
            float w = column + n < K ? float(weight[(output + m) * K + column + n]) : 0.0f;
            result[m] = fma(w, coeff[n], result[m]);
#if ROWS == 2
            second[m] = fma(w, coeff_second[n], second[m]);
#endif
        }
    }
    column += BN * 128;
}
for (uint m = 0; m < 4; ++m) {
    for (ushort delta = 16; delta >= 1; delta >>= 1) {
        result[m] += simd_shuffle_down(result[m], delta);
#if ROWS == 2
        second[m] += simd_shuffle_down(second[m], delta);
#endif
    }
}
#if BN > 1
threadgroup float partials[BN * 8];
#if ROWS == 2
threadgroup float partials_second[BN * 8];
#endif
if (lane == 0) {
    for (uint m = 0; m < 4; ++m) partials[group_n * 8 + m] = result[m];
#if ROWS == 2
    for (uint m = 0; m < 4; ++m) partials_second[group_n * 8 + m] = second[m];
#endif
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group_n == 0) {
    for (uint other = 1; other < BN; ++other) {
        for (uint m = 0; m < 4; ++m) result[m] += partials[other * 8 + m];
#if ROWS == 2
        for (uint m = 0; m < 4; ++m) second[m] += partials_second[other * 8 + m];
#endif
    }
}
#endif
if (group_n == 0 && lane == 0) {
    for (uint m = 0; m < 4; ++m) out[output + m] = result[m];
#if ROWS == 2
    for (uint m = 0; m < 4; ++m) out[N + output + m] = second[m];
#endif
}
