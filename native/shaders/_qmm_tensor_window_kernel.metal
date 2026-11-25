// Q27P-07: prefill EXL3 product on the M5 TensorOps unit. One threadgroup of ROW_GROUP
// simdgroups covers ROW_GROUP row blocks of BM rows. Each depth step decodes the 16 x BN weight
// tile once, by trellis windows like `_qmv_tile_kernel.metal` (one 64-bit window gives the four
// codewords of a lane group), into threadgroup memory in natural (input, column) order; every
// simdgroup then runs the same operation as `_qmm_tensor_kernel.metal` on its row block with the
// tile read by the unit straight from threadgroup memory. Same products in the same order, so
// the outputs are expected bit for bit.
uint column_block = threadgroup_position_in_grid.x;
uint row_group = threadgroup_position_in_grid.y;
uint simdgroup = simdgroup_index_in_threadgroup;
uint group_thread = thread_index_in_threadgroup;
uint row_block = row_group * ROW_GROUP + simdgroup;
// Row blocks are read at run time so one library serves every chunk length.
uint row_blocks = uint(xhat_shape[0]) / BM;
bool active = row_block < row_blocks;
device half* activation = const_cast<device half*>(xhat);
#ifndef STEP_TILES
#define STEP_TILES 1u
#endif
// The native engine consumes one decoded tile at a time. Studio can combine adjacent tiles
// from this same buffer for the image projections with a measured wider operation.
#ifndef OP_K
#define OP_K BK
#endif
#if OP_K < BK || OP_K % BK != 0 || (STEP_TILES * BK) % OP_K != 0 || INPUT_DIMS % OP_K != 0
#error "The TensorOps depth must span whole decoded tiles and divide both the window and input"
#endif
// Q27P-07b: STEP_TILES depth tiles are decoded per barrier, then as many operations run.
threadgroup half shared_tile[2][STEP_TILES * BK * BN];

constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
    BM,
    BN,
    OP_K,
    false,
    false,
    false,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
);
tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;
using left_type = tensor<device half, dextents<int, 2>, tensor_inline>;
using right_type = tensor<threadgroup half, dextents<int, 2>, tensor_inline>;
auto accumulator = operation.get_destination_cooperative_tensor<left_type, right_type, float>();
for (ushort index = 0; index < accumulator.get_capacity(); ++index) {
    if (accumulator.is_valid_element(index)) accumulator[index] = 0.0f;
}

// Window w of a step: column tile w / 64, QMV lane (w / 2) % 32 = 4 j + kb, group w % 2.
// Its codewords 0..3 are the weights of lane-ordered inputs 4 kb + i, i.e. natural inputs
// 2 kb + (i & 1) + 8 (i >> 1) (`exl3_lane_source_index`), at column j + 8 group of the tile.
constexpr uint TILE_WINDOWS = 64u * (BN / 16u);
constexpr uint WINDOWS = TILE_WINDOWS * STEP_TILES;
constexpr uint THREADS = 32u * ROW_GROUP;
constexpr uint PER_THREAD = (WINDOWS + THREADS - 1u) / THREADS;
constexpr uint DEPTH_TILES = INPUT_DIMS / 16u;
uint word0[PER_THREAD], word1[PER_THREAD], shifts[PER_THREAD], slots[PER_THREAD], depth_tile[PER_THREAD];
bool decodes[PER_THREAD];
#pragma unroll
for (uint n = 0u; n < PER_THREAD; ++n) {
    uint window = group_thread + n * THREADS;
    decodes[n] = window < WINDOWS;
    uint k_tile = (window / TILE_WINDOWS) % STEP_TILES;
    uint column_tile = (window >> 6u) % (BN / 16u);
    uint lane = (window >> 1u) & 31u;
    uint group = window & 1u;
    uint first = lane * 8u + group * 4u;
    int end = int(first + 4u) * int(K_BITS) + 256 * int(K_BITS);
    int last_word = (end - 1) / 32;
    uint tile_n = column_block * (BN / 16u) + column_tile + WEIGHT_TILE_OFFSET;
    word1[n] = (k_tile * TILES_N + tile_n) * PACKED_U32 + uint(last_word) % PACKED_U32;
    word0[n] = (k_tile * TILES_N + tile_n) * PACKED_U32 + uint(last_word - 1) % PACKED_U32;
    shifts[n] = uint((last_word + 1) * 32 - end);
    depth_tile[n] = k_tile;
    slots[n] = (k_tile * 16u + 2u * (lane & 3u)) * BN + column_tile * 16u + (lane >> 2u) + 8u * group;
}

uint step = 0u;
for (uint tile_k = 0u; tile_k < DEPTH_TILES; tile_k += STEP_TILES, ++step) {
    threadgroup half* tile = shared_tile[step & 1u];
    const device uint* words = trellis + ulong(tile_k) * TILES_N * PACKED_U32;
    #pragma unroll
    for (uint n = 0u; n < PER_THREAD; ++n) {
#ifdef LAB_SKIP_DECODE
        // Q27P-12 ablation (tests only): decode once, then reuse the tile (wrong outputs).
        if (decodes[n] && tile_k < 2u * STEP_TILES) {
#else
        if (decodes[n] && tile_k + depth_tile[n] < DEPTH_TILES) {
#endif
            ulong merged = (ulong(words[word0[n]]) << 32) | ulong(words[word1[n]]);
            uint shift = shifts[n];
            uint bits = uint(merged >> shift);
            uint cw3 = bits & 0xffffu;
            uint cw2 = (bits >> K_BITS) & 0xffffu;
            uint cw1 = (bits >> (2u * K_BITS)) & 0xffffu;
            uint cw0 = K_BITS <= 5u
                ? (bits >> (3u * K_BITS)) & 0xffffu
                : uint(merged >> (shift + 3u * K_BITS)) & 0xffffu;
            uint slot = slots[n];
            tile[slot] = half(mlxl3_decode_codeword(cw0, 0));
            tile[slot + BN] = half(mlxl3_decode_codeword(cw1, 0));
            tile[slot + 8u * BN] = half(mlxl3_decode_codeword(cw2, 0));
            tile[slot + 9u * BN] = half(mlxl3_decode_codeword(cw3, 0));
        }
    }
    // One barrier per step; the other buffer is rewritten only after the next barrier, when
    // every simdgroup has finished the operations that read it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (active) {
        #pragma unroll
        for (uint part = 0u; part < STEP_TILES; part += OP_K / BK) {
            if (tile_k + part < DEPTH_TILES) {
                auto left = tensor(
                    activation + ulong(row_block * BM) * INPUT_DIMS + (tile_k + part) * 16u,
                    dextents<int, 2>{int(OP_K), int(BM)},
                    array<int, 2>{1, int(INPUT_DIMS)}
                );
                auto right = tensor(tile + part * 16u * BN, dextents<int, 2>{int(BN), int(OP_K)},
                    array<int, 2>{1, int(BN)});
                operation.run(left, right, accumulator);
            }
        }
    }
}

if (active) {
    for (ushort index = 0; index < accumulator.get_capacity(); ++index) {
        if (!accumulator.is_valid_element(index)) continue;
        auto coordinate = accumulator.get_multidimensional_index(index);
        uint output_row = row_block * BM + uint(coordinate[1]);
        uint output_column = column_block * BN + uint(coordinate[0]);
        yhat[ulong(output_row) * OUTPUT_DIMS + output_column] = half(accumulator[index]);
    }
}
