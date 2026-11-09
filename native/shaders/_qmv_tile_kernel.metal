// Ported from MLXL3's former Python/Metal prototype.
// See THIRD_PARTY_NOTICES.md for attribution.

#ifndef MLXL3_REFERENCE_ACCUMULATION
#define MLXL3_REFERENCE_ACCUMULATION 0
#endif
#ifndef MLXL3_REFERENCE_MTP_BATCH
#define MLXL3_REFERENCE_MTP_BATCH 0
#endif

            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint tile_n = threadgroup_position_in_grid.x * MLXL3_QMV_NT;
#ifdef MLXL3_INDEX_KEY_TILES
            tile_n += 32u;
#endif
#ifdef MLXL3_INDEPENDENT_QMV_JOBS
            uint row_base = 0u;
#else
            uint row_base = threadgroup_position_in_grid.y * MLXL3_QMV_MB;
#endif
            uint row = row_base;
            uint split = threadgroup_position_in_grid.z;
#if MLXL3_AR30_COOPERATIVE
            const bool cooperative = K >= 2 && K <= 4 && !MLXL3_REFERENCE_MTP_BATCH;
#endif
            static_assert(MLXL3_QMV_MB == 1u || !MLXL3_FUSE_OUTPUT,
                          "fused epilogue supports one row");
            threadgroup float partials[MLXL3_QMV_SG]
                                      [MLXL3_QMV_MB * MLXL3_QMV_NT * 16u];
#if MLXL3_REFERENCE_ACCUMULATION
            threadgroup float reference_partials[MLXL3_QMV_MB][MLXL3_QMV_NT][MLXL3_QMV_SG][256];
#endif
#if MLXL3_Q02_PROFILE
            threadgroup float q02_input_sums[MLXL3_QMV_MB][MLXL3_QMV_SG];
            float q02_sx[MLXL3_QMV_MB]={};
#endif
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
#if MLXL3_AR20_PRELOAD
            for (uint tile_pair = tile_begin + simd; tile_pair < tile_end; tile_pair += 2u * MLXL3_QMV_SG) {
                ulong staged[2][MLXL3_QMV_NT][2];
#if MLXL3_AR20_READER8
                ulong staged_reader[2][MLXL3_QMV_NT];
#endif
                for (uint u = 0u; u < 2u; ++u) {
                    uint tk = tile_pair + u * MLXL3_QMV_SG;
                    if (tk >= tile_end) continue;
                    for (uint ot = 0u; ot < MLXL3_QMV_NT; ++ot) {
                        const device uint* src = trellis + (tk * uint(TILES_N) + tile_n + ot) * uint(PACKED_U32);
#if MLXL3_AR30_COOPERATIVE
                        if (cooperative) {
                            uint owned = ar30_cooperative_load(src, lane, uint(K));
                            for (uint g = 0u; g < 2u; ++g)
                                staged[u][ot][g] = ar30_cooperative_pair(owned, word0[g], word1[g]);
                        } else
#endif
                        {
#if MLXL3_AR20_READER8
                        if (K <= 6) staged_reader[u][ot] = ar20_window8(src, lane, uint(K));
                        else
#endif
                        for (uint g = 0u; g < 2u; ++g)
                            staged[u][ot][g] = (ulong(src[word0[g]]) << 32u) | ulong(src[word1[g]]);
                        }
                    }
                }
                for (uint u = 0u; u < 2u; ++u) {
                    uint tile_k = tile_pair + u * MLXL3_QMV_SG;
                    if (tile_k >= tile_end) break;
#else
            for (
                uint tile_k = tile_begin + simd;
                tile_k < tile_end;
                tile_k += MLXL3_QMV_SG
            ) {
#endif
                float x_values[MLXL3_QMV_MB][4];
                for (uint batch_row = 0u; batch_row < MLXL3_QMV_MB; ++batch_row) {
#if defined(MLXL3_QMV_LANE_ORDER) && MLXL3_QMV_LANE_ORDER
                    // MLXL3_Q27_LANE_INPUT_PERMUTED
                    half4 x_quad = half4(0.0h);
                    if (row_base + batch_row < uint(MLXL3_MATRIX_ROWS)) {
                        const device half4* x_quads =
                            reinterpret_cast<const device half4*>(
                                xhat + (row_base + batch_row) * uint(INPUT_DIMS)
                                     + tile_k * 16u
                            );
                        x_quad = x_quads[lane & 3u];
                    }
                    x_values[batch_row][0] = float(x_quad.x);
                    x_values[batch_row][1] = float(x_quad.y);
                    x_values[batch_row][2] = float(x_quad.z);
                    x_values[batch_row][3] = float(x_quad.w);
#else
                    half2 x_pair = half2(0.0h);
                    if (row_base + batch_row < uint(MLXL3_MATRIX_ROWS)) {
                        const device half2* x_pairs =
                            reinterpret_cast<const device half2*>(
                                xhat + (row_base + batch_row) * uint(INPUT_DIMS)
                                     + tile_k * 16u
                            );
                        x_pair = x_pairs[lane & 7u];
                    }
                    half2 x_pair0 = simd_shuffle(x_pair, ushort(lane & 3u));
                    half2 x_pair1 = simd_shuffle(x_pair, ushort((lane & 3u) + 4u));
                    x_values[batch_row][0] = float(x_pair0.x);
                    x_values[batch_row][1] = float(x_pair0.y);
                    x_values[batch_row][2] = float(x_pair1.x);
                    x_values[batch_row][3] = float(x_pair1.y);
#endif
                }
#if MLXL3_Q02_PROFILE
                if (CB==2u) for(uint br=0;br<MLXL3_QMV_MB;++br) {
                    float v=lane<4u?((x_values[br][0]+x_values[br][1])+(x_values[br][2]+x_values[br][3])):0.0f;
                    q02_sx[br]+=simd_sum(v);
                }
#endif
                for (
                    uint output_tile = 0u;
                    output_tile < MLXL3_QMV_NT;
                    ++output_tile
                ) {
                    const device uint* words = trellis +
                        (tile_k * uint(TILES_N) + tile_n + output_tile)
                        * uint(PACKED_U32);
#if MLXL3_AR30_COOPERATIVE && !MLXL3_AR20_PRELOAD
                    uint owned = cooperative ? ar30_cooperative_load(words, lane, uint(K)) : 0u;
#endif
#if MLXL3_AR20_READER8
                    uint reader_cw[8];
                    if (K <= 6
#if MLXL3_AR30_COOPERATIVE
                        && !cooperative
#endif
                    ) {
#if MLXL3_AR20_PRELOAD
                        ar20_read8(staged_reader[u][output_tile], uint(K), reader_cw);
#else
                        ar20_read8(ar20_window8(words, lane, uint(K)), uint(K), reader_cw);
#endif
                    }
#endif
                    for (uint group = 0u; group < 2u; ++group) {
                        uint cw0, cw1, cw2, cw3;
#if MLXL3_AR20_READER8
                        if (K <= 6
#if MLXL3_AR30_COOPERATIVE
                            && !cooperative
#endif
                        ) {
                            cw0 = reader_cw[group * 4u];
                            cw1 = reader_cw[group * 4u + 1u];
                            cw2 = reader_cw[group * 4u + 2u];
                            cw3 = reader_cw[group * 4u + 3u];
                        } else
#endif
                        {
#if MLXL3_AR20_PRELOAD
                        ulong merged = staged[u][output_tile][group];
#else
#if MLXL3_AR30_COOPERATIVE
                        ulong merged = cooperative
                            ? ar30_cooperative_pair(owned, word0[group], word1[group])
                            : (ulong(words[word0[group]]) << 32) | ulong(words[word1[group]]);
#else
                        ulong merged = (ulong(words[word0[group]]) << 32) |
                                       ulong(words[word1[group]]);
#endif
#endif
                        uint shift = shifts[group];
                        uint window = uint(merged >> shift);
                        cw3 = window & 0xffffu;
                        cw2 = (window >> MLXL3_K_BITS) & 0xffffu;
                        cw1 = (window >> (2u * MLXL3_K_BITS)) & 0xffffu;
                        cw0 = MLXL3_K_BITS <= 5u
                            ? (window >> (3u * MLXL3_K_BITS)) & 0xffffu
                            : uint(merged >> (shift + 3u * MLXL3_K_BITS)) & 0xffffu;
                        }
                        uint j = group * 4u;
                        #if MLXL3_Q02_PROFILE
                        float weight0=CB==2u?mlxl3_q02_weight(cw0,CB):float(mlxl3_decode_codeword(cw0,CB));
#else
                        float weight0 = float(mlxl3_decode_codeword(cw0, CB));
#endif
                        #if MLXL3_Q02_PROFILE
                        float weight1=CB==2u?mlxl3_q02_weight(cw1,CB):float(mlxl3_decode_codeword(cw1,CB));
#else
                        float weight1 = float(mlxl3_decode_codeword(cw1, CB));
#endif
                        #if MLXL3_Q02_PROFILE
                        float weight2=CB==2u?mlxl3_q02_weight(cw2,CB):float(mlxl3_decode_codeword(cw2,CB));
#else
                        float weight2 = float(mlxl3_decode_codeword(cw2, CB));
#endif
                        #if MLXL3_Q02_PROFILE
                        float weight3=CB==2u?mlxl3_q02_weight(cw3,CB):float(mlxl3_decode_codeword(cw3,CB));
#else
                        float weight3 = float(mlxl3_decode_codeword(cw3, CB));
#endif
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

#if MLXL3_AR20_PRELOAD
            }
#endif
#if MLXL3_Q02_PROFILE
            if(lane==0u) for(uint br=0;br<MLXL3_QMV_MB;++br) q02_input_sums[br][simd]=q02_sx[br];
#endif
            for (uint batch_row = 0u; batch_row < MLXL3_QMV_MB; ++batch_row) {
              for (uint output_tile = 0u; output_tile < MLXL3_QMV_NT; ++output_tile) {
#if MLXL3_REFERENCE_ACCUMULATION
                for (uint j = 0u; j < 8u; ++j) {
                    reference_partials[batch_row][output_tile][simd][mlxl3_reference_positions[lane * 8u + j]] = acc[batch_row][output_tile][j];
                }
#else
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
#endif
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
#if MLXL3_REFERENCE_ACCUMULATION
                // Row-then-group order preserves Flash's expert decisions.
                for (uint r = 0u; r < 16u; ++r) {
#if MLXL3_REFERENCE_MTP_BATCH
                    sum += reference_partials[batch_row][output_tile][0][r * 16u + column]
                         + reference_partials[batch_row][output_tile][1][r * 16u + column];
#else
                    for (uint group = 0u; group < MLXL3_QMV_SG; ++group) {
                        sum += reference_partials[batch_row][output_tile][group][r * 16u + column];
                    }
#endif
                }
#else
                for (uint group = 0u; group < MLXL3_QMV_SG; ++group) {
                    sum += partials[group][partial];
                }
#endif
#if MLXL3_Q02_PROFILE
                if(CB==2u) {
                    float sx=0.0f;
                    for(uint sg=0;sg<MLXL3_QMV_SG;++sg) sx+=q02_input_sums[batch_row][sg];
                    // This CTA owns one split: never repeat a global correction in each split.
                    sum=fma(887.0f/131072.0f,sum,(MLXL3_Q02_PROFILE==2?0.01171875f:-3.453125f)*sx);
                }
#endif
                if (MLXL3_FUSE_OUTPUT) ha[tid] = float(half(sum));
                else if (row_base + batch_row < uint(MLXL3_MATRIX_ROWS)) {
                    yhat[
                        ((row_base + batch_row) * uint(N_SPLITS) + split)
                        * uint(OUTPUT_DIMS)
                        + (tile_n + output_tile) * 16u + column
                    ] = sum;
                }
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
