// Q27D-31: fused epilogue of the lane TensorOps product. Same arithmetic as the MLX chain
// sum(splits) -> astype(F16) -> hadamard_transform(1/sqrt(128)) -> mul(svh) -> astype(out):
// sequential split sum from +0 (no addition for one split), MLX 0.32.2 Hadamard-128, F16 svh
// product, conversion to the output type. One threadgroup of 8 threads per (row, 128 block).
// PARTIAL_OFFSET(split, row) addresses [splits, padded, out] (lane) or [rows, splits, out] (QMV).
short i = short(thread_position_in_threadgroup.x);
uint block = threadgroup_position_in_grid.x;
uint row = threadgroup_position_in_grid.y;
uint base = block * 128u;
threadgroup half buf[128];
// Splits and padded rows are read at run time so one library serves every row count.
uint splits = uint(partials_shape[SPLIT_AXIS]);
uint padded_rows = uint(partials_shape[1 - SPLIT_AXIS]);
for (short r = 0; r < 16; ++r) {
    uint column = base + uint(i * 16 + r);
    float value;
    if (splits == 1u) {
        value = partials[PARTIAL_OFFSET(0u, row) + column];
    } else {
        volatile float total = 0.0f;
        for (uint s = 0u; s < splits; ++s) {
            total = total + partials[PARTIAL_OFFSET(s, row) + column];
        }
        value = total;
    }
    buf[i * 16 + r] = half(value);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
mlxl3_hadamard128_mlx(buf, i);
float scale = as_type<float>(uint(SCALE_BITS));
for (short r = 0; r < 16; ++r) {
    uint column = base + uint(i * 16 + r);
    half rotated = half(float(buf[i * 16 + r]) * scale);
    y[ulong(row) * OUTPUT_DIMS + column] = OUT_T(half(rotated * svh[column]));
}
