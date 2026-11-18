// Q27D-31: fused prologue of the lane TensorOps product. Same arithmetic as the MLX chain
// astype(F16) -> mul(suh) -> hadamard_transform(1/sqrt(128)) -> take(lane order) -> zero rows:
// F16 product, MLX 0.32.2 radix-16 then radix-8 butterflies in F32 with F16 storage between
// stages, final half(float(v) * scale). One threadgroup of 8 threads per (row, 128 block).
short i = short(thread_position_in_threadgroup.x);
uint block = threadgroup_position_in_grid.x;
uint out_row = threadgroup_position_in_grid.y;
// Rows are read at run time (x is [rows, input]) so one library serves every row count.
uint matrix_rows = uint(x_shape[0]);
uint row = out_row / GROUPS;
uint group = out_row % GROUPS;
uint base = block * 128u;
threadgroup half buf[128];
if (row >= matrix_rows) {
    for (short r = 0; r < 16; ++r) xhat[ulong(out_row) * INPUT_DIMS + base + uint(i * 16 + r)] = half(0.0h);
    return;
}
for (short r = 0; r < 16; ++r) {
    uint column = base + uint(i * 16 + r);
    buf[i * 16 + r] = half(x[ulong(row) * INPUT_DIMS + column]) * suh[ulong(group) * INPUT_DIMS + column];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
mlxl3_hadamard128_mlx(buf, i);
float scale = as_type<float>(uint(SCALE_BITS));
for (short r = 0; r < 16; ++r) {
    uint j = uint(i * 16 + r);
#if LANE_ORDER
    uint within = j & 15u;
    uint source = (j & ~15u) | ((within & 12u) >> 1u) | (within & 1u) | ((within & 2u) << 2u);
#else
    uint source = j;
#endif
    xhat[ulong(out_row) * INPUT_DIMS + base + j] = half(float(buf[source]) * scale);
}
