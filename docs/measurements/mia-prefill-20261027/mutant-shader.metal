// Shared decoded weights across rows, inspired by Mia's prompt tile; M1 FMA/reduction order is unchanged.
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint row_lane = simdgroup_index_in_threadgroup / MLXL3_QMV_SG;
uint simd = simdgroup_index_in_threadgroup % MLXL3_QMV_SG;
uint tile_group = threadgroup_position_in_grid.x;
uint split = threadgroup_position_in_grid.z;
uint groups_per_row = uint(OUTPUT_TILES);
uint sub = tile_group / groups_per_row;
uint tile_n = tile_group % groups_per_row;
uint route = sub / uint(ROUTING_REPEAT);
uint projection = sub % uint(ROUTING_REPEAT);
uint original = tile_sub[route * MLXL3_QMV_MB + row_lane] * uint(ROUTING_REPEAT) + projection;
uint slot = tile_map[tile_sub[route * MLXL3_QMV_MB]];
uint compact = (route * MLXL3_QMV_MB + row_lane) * uint(ROUTING_REPEAT) + projection;
threadgroup float weights[MLXL3_QMV_SG][256];
threadgroup float partials[MLXL3_QMV_MB][MLXL3_QMV_SG][256];
float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
uint word0[2], word1[2], shifts[2];
for (uint group = 0; group < 2; ++group) {
    int end = int(lane * 8u + group * 4u + 4u) * K + 256 * K;
    int last = (end - 1) / 32;
    word1[group] = uint(last) % uint(PACKED_U32);
    word0[group] = uint(last - 1) % uint(PACKED_U32);
    shifts[group] = uint((last + 1) * 32 - end);
}
uint per_split = (uint(TILES_K) + uint(N_SPLITS) - 1u) / uint(N_SPLITS);
uint begin = split * per_split;
uint end = min(begin + per_split, uint(TILES_K));
// Every row and SIMDgroup reaches both barriers, including a partial final K step.
for (uint base = begin; base < end; base += MLXL3_QMV_SG) {
    uint tile_k = base + simd;
    if (row_lane == 0 && tile_k < end) {
        const device uint* words = trellis +
            ((slot * uint(ROUTING_REPEAT) + projection) * uint(TILES_K) * uint(TILES_N)
             + tile_k * uint(TILES_N) + tile_n) * uint(PACKED_U32);
        for (uint group = 0; group < 2; ++group) {
            ulong merged = (ulong(words[word0[group]]) << 32) | ulong(words[word1[group]]);
            uint window = uint(merged >> shifts[group]);
            uint cw[4] = {
                uint(merged >> (shifts[group] + 3u * MLXL3_K_BITS)) & 0xffffu,
                (window >> (2u * MLXL3_K_BITS)) & 0xffffu,
                (window >> MLXL3_K_BITS) & 0xffffu,
                window & 0xffffu
            };
            for (uint j = 0; j < 4; ++j)
                weights[simd][lane * 8u + group * 4u + j] = mlxl3_decode_codeword(cw[j], CB);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tile_k < end) {
        const device half2* pairs = reinterpret_cast<const device half2*>(
            xhat + original * uint(INPUT_DIMS) + tile_k * 16u);
        half2 pair = pairs[lane & 7u];
        half2 lo = simd_shuffle(pair, ushort(lane & 3u));
        half2 hi = simd_shuffle(pair, ushort((lane & 3u) + 4u));
        float x[4] = {float(lo.x), float(lo.y), float(hi.x), float(hi.y)};
        for (uint j = 0; j < 8; ++j)
            acc[j] = fma(x[j & 3u], weights[simd][lane * 8u + j], acc[j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
for (uint j = 0; j < 8; ++j)
    partials[row_lane][simd][mlxl3_reference_positions[lane * 8u + j]] = acc[j];
threadgroup_barrier(mem_flags::mem_threadgroup);
uint local = tid % (MLXL3_QMV_SG * 32u);
if (local < 16u) {
    float sum = 1024.0f;
    for (uint r = 0; r < 16; ++r)
        for (uint group = 0; group < MLXL3_QMV_SG; ++group)
            sum += partials[row_lane][group][r * 16u + local];
    yhat[split * uint(MLXL3_GROUP_OUTPUT_DIMS)
         + compact * uint(OUTPUT_TILES) * 16u + tile_n * 16u + local] = sum;
}
