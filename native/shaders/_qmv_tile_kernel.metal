// Ported from MLXL3's former Python/Metal prototype.
// See THIRD_PARTY_NOTICES.md for attribution.

            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint tile_n = threadgroup_position_in_grid.x * MLXL3_QMV_NT;
            uint row_base = threadgroup_position_in_grid.y * MLXL3_QMV_MB;
            uint row = row_base;
            uint split = threadgroup_position_in_grid.z;
            static_assert(MLXL3_QMV_MB == 1u || !MLXL3_FUSE_OUTPUT,
                          "fused epilogue supports one row");
            threadgroup float partials[MLXL3_QMV_SG]
                                      [MLXL3_QMV_MB * MLXL3_QMV_NT * 16u];
            threadgroup float ha[128];
            threadgroup float hb[128];

            uint word0[2];
            uint word1[2];
            uint shifts[2];
            for (uint group = 0u; group < 2u; ++group) {
                uint first = lane * 8u + group * 4u;
                int end = int(first + 4u) * K + 256 * K;
                int last_word = (end - 1) / 32;
                word1[group] = uint(last_word) % uint(PACKED_U32);
                word0[group] = uint(last_word - 1) % uint(PACKED_U32);
                shifts[group] = uint((last_word + 1) * 32 - end);
            }

            float acc[MLXL3_QMV_MB][MLXL3_QMV_NT][8];
            for (uint batch_row = 0u; batch_row < MLXL3_QMV_MB; ++batch_row) {
                for (uint output_tile = 0u; output_tile < MLXL3_QMV_NT; ++output_tile) {
                    for (uint j = 0u; j < 8u; ++j) {
                        acc[batch_row][output_tile][j] = 0.0f;
                    }
                }
            }
            uint tiles_per_split =
                (uint(TILES_K) + uint(N_SPLITS) - 1u) / uint(N_SPLITS);
            uint tile_begin = split * tiles_per_split;
            uint tile_end = min(tile_begin + tiles_per_split, uint(TILES_K));
            for (
                uint tile_k = tile_begin + simd;
                tile_k < tile_end;
                tile_k += MLXL3_QMV_SG
            ) {
                float x_values[MLXL3_QMV_MB][4];
                for (uint batch_row = 0u; batch_row < MLXL3_QMV_MB; ++batch_row) {
                    const device half2* x_pairs =
                        reinterpret_cast<const device half2*>(
                            xhat + (row_base + batch_row) * uint(INPUT_DIMS)
                                 + tile_k * 16u
                        );
                    half2 x_pair = x_pairs[lane & 7u];
                    half2 x_pair0 = simd_shuffle(x_pair, ushort(lane & 3u));
                    half2 x_pair1 = simd_shuffle(x_pair, ushort((lane & 3u) + 4u));
                    x_values[batch_row][0] = float(x_pair0.x);
                    x_values[batch_row][1] = float(x_pair0.y);
                    x_values[batch_row][2] = float(x_pair1.x);
                    x_values[batch_row][3] = float(x_pair1.y);
                }
                for (
                    uint output_tile = 0u;
                    output_tile < MLXL3_QMV_NT;
                    ++output_tile
                ) {
                    const device uint* words = trellis +
                        (tile_k * uint(TILES_N) + tile_n + output_tile)
                        * uint(PACKED_U32);
                    for (uint group = 0u; group < 2u; ++group) {
                        ulong merged = (ulong(words[word0[group]]) << 32) |
                                       ulong(words[word1[group]]);
                        uint shift = shifts[group];
                        uint window = uint(merged >> shift);
                        uint cw3 = window & 0xffffu;
                        uint cw2 = (window >> MLXL3_K_BITS) & 0xffffu;
                        uint cw1 = (window >> (2u * MLXL3_K_BITS)) & 0xffffu;
                        uint cw0 = MLXL3_K_BITS <= 5u
                            ? (window >> (3u * MLXL3_K_BITS)) & 0xffffu
                            : uint(merged >> (shift + 3u * MLXL3_K_BITS)) & 0xffffu;
                        uint j = group * 4u;
                        float weight0 = float(mlxl3_decode_codeword(cw0, CB));
                        float weight1 = float(mlxl3_decode_codeword(cw1, CB));
                        float weight2 = float(mlxl3_decode_codeword(cw2, CB));
                        float weight3 = float(mlxl3_decode_codeword(cw3, CB));
                        for (uint batch_row = 0u; batch_row < MLXL3_QMV_MB; ++batch_row) {
                            acc[batch_row][output_tile][j] = fma(
                                x_values[batch_row][0], weight0,
                                acc[batch_row][output_tile][j]
                            );
                            acc[batch_row][output_tile][j + 1u] = fma(
                                x_values[batch_row][1], weight1,
                                acc[batch_row][output_tile][j + 1u]
                            );
                            acc[batch_row][output_tile][j + 2u] = fma(
                                x_values[batch_row][2], weight2,
                                acc[batch_row][output_tile][j + 2u]
                            );
                            acc[batch_row][output_tile][j + 3u] = fma(
                                x_values[batch_row][3], weight3,
                                acc[batch_row][output_tile][j + 3u]
                            );
                        }
                    }
                }
            }

            for (uint batch_row = 0u; batch_row < MLXL3_QMV_MB; ++batch_row) {
              for (uint output_tile = 0u; output_tile < MLXL3_QMV_NT; ++output_tile) {
                // The EXL3 permutation assigns each four-lane quadgroup one
                // output column and gives every lane four rows for columns
                // c and c + 8. Reduce those rows in registers instead of
                // materializing all 256 products in threadgroup memory.
                float column0 = quad_sum(
                    acc[batch_row][output_tile][0] + acc[batch_row][output_tile][1]
                ) + quad_sum(
                    acc[batch_row][output_tile][2] + acc[batch_row][output_tile][3]
                );
                float column1 = quad_sum(
                    acc[batch_row][output_tile][4] + acc[batch_row][output_tile][5]
                ) + quad_sum(
                    acc[batch_row][output_tile][6] + acc[batch_row][output_tile][7]
                );
                if ((lane & 3u) == 0u) {
                    uint column = lane >> 2u;
                    uint base = (batch_row * MLXL3_QMV_NT + output_tile) * 16u;
                    partials[simd][base + column] = column0;
                    partials[simd][base + column + 8u] = column1;
                }
              }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 16u * MLXL3_QMV_NT * MLXL3_QMV_MB) {
                uint batch_row = tid / (16u * MLXL3_QMV_NT);
                uint local = tid % (16u * MLXL3_QMV_NT);
                uint output_tile = local >> 4u;
                uint column = local & 15u;
                uint partial = (batch_row * MLXL3_QMV_NT + output_tile) * 16u + column;
                float sum = 0.0f;
                for (uint group = 0u; group < MLXL3_QMV_SG; ++group) {
                    sum += partials[group][partial];
                }
                if (MLXL3_FUSE_OUTPUT) ha[tid] = float(half(sum));
                else yhat[
                    ((row_base + batch_row) * uint(N_SPLITS) + split) * uint(OUTPUT_DIMS)
                    + (tile_n + output_tile) * 16u + column
                ] = sum;
            }
            if (MLXL3_FUSE_OUTPUT) {
                for (uint shift = 0u; shift < 7u; ++shift) {
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid < 128u) {
                        uint h = 1u << shift;
                        float own = ha[tid], peer = ha[tid ^ h];
                        float value = (tid & h) ? peer - own : own + peer;
                        hb[tid] = shift == 3u ? float(half(value)) : value;
                    }
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    if (tid < 128u) ha[tid] = hb[tid];
                }
                if (tid < 128u) {
                    uint index = tile_n * 16u + tid;
                    half rotated = half(float(half(ha[tid])) * 0.08838834764831845f);
                    yhat[row * uint(OUTPUT_DIMS) + index] =
                        half(rotated * half(svh[index]));
                }
            }
