// Adapted from MTPLX fused_norm.py, revision 9882703, Apache-2.0.
// Copyright 2026 Youssof Altoukhi. See THIRD_PARTY_NOTICES.md.
// Match MLX's four consecutive values per lane and FP16 rounding points.
uint row = threadgroup_position_in_grid.x;
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint group = simdgroup_index_in_threadgroup;
threadgroup float inverse[1];
threadgroup float sums[32];
half values[8];
float acc = 0.0f;
size_t offset = size_t(row) * size_t(AXIS);
for (uint r = 0; r < AXIS; r += THREADS * 4) {
    for (uint i = 0; i < 4; ++i) {
        uint index = r + lid * 4 + i;
        if (index < AXIS) {
            half value = x[offset + index] + residual[offset + index];
            values[(r / (THREADS * 4)) * 4 + i] = value;
            float f = float(value);
            acc += f * f;
        }
    }
}
acc = simd_sum(acc);
if (group == 0) sums[lane] = 0.0f;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lane == 0) sums[group] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0) {
    acc = simd_sum(sums[lane]);
    if (lane == 0) inverse[0] = metal::precise::rsqrt(acc / float(AXIS) + eps[0]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint r = 0; r < AXIS; r += THREADS * 4) {
    for (uint i = 0; i < 4; ++i) {
        uint index = r + lid * 4 + i;
        if (index < AXIS) {
            half value = values[(r / (THREADS * 4)) * 4 + i];
            h[offset + index] = value;
            normed[offset + index] = weight[index] * half(float(value) * inverse[0]);
        }
    }
}
