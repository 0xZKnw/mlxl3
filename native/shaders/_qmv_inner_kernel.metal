// Verbatim kernel body from src/mlxl3/kernels/qmv.py::_qmv_inner_kernel.
// See THIRD_PARTY_NOTICES.md for attribution.

            uint tid = thread_index_in_threadgroup;
            uint lane = thread_index_in_simdgroup;
            uint simd = simdgroup_index_in_threadgroup;
            uint n = threadgroup_position_in_grid.x;
            threadgroup float partials[4];
            float acc = 0.0f;

            for (uint row = tid; row < uint(INPUT_DIMS); row += 128u) {
                uint tile_k = row >> 4;
                uint tile_n = n >> 4;
                uint local_rm = ((row & 15u) << 4) | (n & 15u);
                uint t = uint(mlxl3_perm_inv[local_rm]);
                uint packed_per_tile = 16u * uint(K);
                uint words_per_tile = 8u * uint(K);
                uint tile_base =
                    (tile_k * uint(TILES_N) + tile_n) * packed_per_tile;

                uint b0 = t * uint(K) + uint(K) - 16u + 256u * uint(K);
                uint b1 = b0 + 16u;
                uint i0 = b0 >> 5;
                uint i1 = (b1 - 1u) >> 5;
                uint shift = ((i1 + 1u) << 5) - b1;
                uint wi0 = i0 % words_per_tile;
                uint wi1 = i1 % words_per_tile;
                uint a = uint(trellis[tile_base + 2u * wi0]) |
                         (uint(trellis[tile_base + 2u * wi0 + 1u]) << 16);
                uint b = uint(trellis[tile_base + 2u * wi1]) |
                         (uint(trellis[tile_base + 2u * wi1 + 1u]) << 16);
                ulong merged = (ulong(a) << 32) | ulong(b);
                uint word = uint((merged >> shift) & 0xfffful);
                float weight = float(mlxl3_decode_codeword(word, CB));
                acc = fma(float(xhat[row]), weight, acc);
            }

            acc = simd_sum(acc);
            if (lane == 0u) {
                partials[simd] = acc;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float total = tid < 4u ? partials[tid] : 0.0f;
            total = simd_sum(total);
            if (tid == 0u) {
                yhat[n] = total;
            }
