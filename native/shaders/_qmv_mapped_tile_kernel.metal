// Verbatim source body from src/mlxl3/kernels/qmv.py.
// See THIRD_PARTY_NOTICES.md for attribution and licenses.

            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint tile_group = threadgroup_position_in_grid.x;
            uint local_tile = tile_group * MLXL3_QMV_NT;
            uint split = threadgroup_position_in_grid.z;
            threadgroup float partials[MLXL3_QMV_SG][MLXL3_QMV_NT * 16u];

#if MLXL3_K3_WINDOW_DECODE
            int k3_bit_end = int((lane * 8u + 257u) * 3u + 21u);
            int k3_word_end = (k3_bit_end - 1) / 32;
            uint k3_word1 = uint(k3_word_end) % uint(PACKED_U32);
            uint k3_word0 = uint((k3_bit_end - 21 - 16) / 32) % uint(PACKED_U32);
            uint k3_shift = uint((k3_word_end + 1) * 32 - k3_bit_end);
#else
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
#endif

            float acc[MLXL3_QMV_NT][8];
            for (uint output_tile = 0u; output_tile < MLXL3_QMV_NT; ++output_tile) {
                for (uint j = 0u; j < 8u; ++j) {
                    acc[output_tile][j] = 0.0f;
                }
            }
            uint tiles_per_split =
                (uint(TILES_K) + uint(N_SPLITS) - 1u) / uint(N_SPLITS);
            uint tile_begin = split * tiles_per_split;
            uint tile_end = min(tile_begin + tiles_per_split, uint(TILES_K));
            uint sub;
            uint tile_ns[MLXL3_QMV_NT];
            if (EXPERT_MAP) {
                uint groups_per_row = uint(OUTPUT_TILES) / MLXL3_QMV_NT;
                sub = tile_group / groups_per_row;
                uint group_in_row = tile_group - sub * groups_per_row;
                uint route = sub / uint(ROUTING_REPEAT);
                uint projection = sub - route * uint(ROUTING_REPEAT);
                uint source_base =
                    tile_map[route] * uint(OUTPUT_TILES)
                    + projection * uint(PROJECTION_STRIDE_TILES)
                    + group_in_row * MLXL3_QMV_NT;
                for (
                    uint output_tile = 0u;
                    output_tile < MLXL3_QMV_NT;
                    ++output_tile
                ) {
                    tile_ns[output_tile] = source_base + output_tile;
                }
            } else {
                sub = tile_sub[local_tile];
                for (
                    uint output_tile = 0u;
                    output_tile < MLXL3_QMV_NT;
                    ++output_tile
                ) {
                    uint tile_offset = local_tile + output_tile;
                    tile_ns[output_tile] =
                        IDENTITY_MAP ? tile_offset : tile_map[tile_offset];
                }
            }
            for (
                uint tile_k = tile_begin + simd;
                tile_k < tile_end;
                tile_k += MLXL3_QMV_SG
            ) {
                // Output rows are 128-aligned, so both tiles in a pair use
                // the same transformed activation row.
                const device half2* x_pairs = reinterpret_cast<const device half2*>(
                    xhat + sub * uint(INPUT_DIMS) + tile_k * 16u
                );
                half2 x_pair = x_pairs[lane & 7u];
                half2 x_pair0 = simd_shuffle(x_pair, ushort(lane & 3u));
                half2 x_pair1 = simd_shuffle(x_pair, ushort((lane & 3u) + 4u));
                float x_values[4];
                x_values[0] = float(x_pair0.x);
                x_values[1] = float(x_pair0.y);
                x_values[2] = float(x_pair1.x);
                x_values[3] = float(x_pair1.y);
                for (
                    uint output_tile = 0u;
                    output_tile < MLXL3_QMV_NT;
                    ++output_tile
                ) {
                    const device uint* words = trellis +
                        (tile_k * uint(TILES_N) + tile_ns[output_tile])
                        * uint(PACKED_U32);
#if MLXL3_K3_WINDOW_DECODE
                    ulong merged = (ulong(words[k3_word0]) << 32)
                        | ulong(words[k3_word1]);
                    uint codewords[8];
                    uint window7 = uint(merged >> k3_shift);
                    codewords[7] = window7;
                    codewords[6] = window7 >> 3u;
                    codewords[5] = window7 >> 6u;
                    codewords[4] = window7 >> 9u;
                    uint window3 = uint(merged >> (k3_shift + 12u));
                    codewords[3] = window3;
                    codewords[2] = window3 >> 3u;
                    codewords[1] = window3 >> 6u;
                    codewords[0] = window3 >> 9u;
                    for (uint j = 0u; j < 8u; ++j) {
                        acc[output_tile][j] = fma(
                            x_values[j & 3u],
                            float(mlxl3_decode_codeword(codewords[j], CB)),
                            acc[output_tile][j]
                        );
                    }
#else
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
                        acc[output_tile][j] = fma(
                            x_values[0],
                            float(mlxl3_decode_codeword(cw0, CB)),
                            acc[output_tile][j]
                        );
                        acc[output_tile][j + 1u] = fma(
                            x_values[1],
                            float(mlxl3_decode_codeword(cw1, CB)),
                            acc[output_tile][j + 1u]
                        );
                        acc[output_tile][j + 2u] = fma(
                            x_values[2],
                            float(mlxl3_decode_codeword(cw2, CB)),
                            acc[output_tile][j + 2u]
                        );
                        acc[output_tile][j + 3u] = fma(
                            x_values[3],
                            float(mlxl3_decode_codeword(cw3, CB)),
                            acc[output_tile][j + 3u]
                        );
                    }
#endif
                }
            }

            for (uint output_tile = 0u; output_tile < MLXL3_QMV_NT; ++output_tile) {
                float column0 = quad_sum(
                    acc[output_tile][0] + acc[output_tile][1]
                ) + quad_sum(
                    acc[output_tile][2] + acc[output_tile][3]
                );
                float column1 = quad_sum(
                    acc[output_tile][4] + acc[output_tile][5]
                ) + quad_sum(
                    acc[output_tile][6] + acc[output_tile][7]
                );
                if ((lane & 3u) == 0u) {
                    uint column = lane >> 2u;
                    partials[simd][output_tile * 16u + column] = column0;
                    partials[simd][output_tile * 16u + column + 8u] = column1;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid < 16u * MLXL3_QMV_NT) {
                uint output_tile = tid >> 4u;
                uint column = tid & 15u;
                float sum = 0.0f;
                for (uint group = 0u; group < MLXL3_QMV_SG; ++group) {
                    sum += partials[group][output_tile * 16u + column];
                }
                yhat[
                    split * uint(LOCAL_OUTPUT_DIMS)
                    + (local_tile + output_tile) * 16u + column
                ] = sum;
            }
