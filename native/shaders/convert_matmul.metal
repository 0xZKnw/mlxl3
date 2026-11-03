#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

// Explicit full-FP32 path. Source views retain their element offsets/strides.
kernel void mlxl3_convert_matmul(
    const device float* left [[buffer(0)]],
    const device float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant long* p [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint m = uint(p[0]), n = uint(p[1]), k = uint(p[2]);
    uint row = tile.y * 8, col = tile.x * 8;
    threadgroup float a[64], b[64], c[64];
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (uint depth = 0; depth < k; depth += 8) {
        for (uint i = lane; i < 64; i += 32) {
            uint r = i / 8, s = i % 8;
            a[i] = row + r < m && depth + s < k
                ? left[p[3] + long(row + r) * p[4] + long(depth + s) * p[5]] : 0.0f;
            b[i] = depth + r < k && col + s < n
                ? right[p[6] + long(depth + r) * p[7] + long(col + s) * p[8]] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 ma, mb;
        simdgroup_load(ma, a, 8);
        simdgroup_load(mb, b, 8);
        simdgroup_multiply_accumulate(acc, ma, mb, acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc, c, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lane; i < 64; i += 32) {
        uint r = row + i / 8, s = col + i % 8;
        if (r < m && s < n) output[ulong(r) * n + s] = c[i];
    }
}
