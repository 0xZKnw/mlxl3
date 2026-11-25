// Q27D-17 (not bit-exact): EXL3 product on the M5 TensorOps unit with the weights as the LEFT
// operand. A left cooperative-tensor lane holds four consecutive inputs x columns {j, j+8, ...},
// exactly one QMV lane's codeword groups per 16x16 tile, so each lane decodes four codewords per
// trellis window like `_qmv_tile_kernel.metal`. Activations are lane-ordered (input permutation
// shared with the QMV path) and fed transposed on the right. Partial sums over K are FP32.
#ifndef LANE_COLS
#define LANE_COLS 32u
#endif
#ifndef LANE_DEPTH
#define LANE_DEPTH 16u
#endif
#ifndef LANE_TOKENS
#define LANE_TOKENS 16u
#endif
#ifndef LANE_BUFFERS
#define LANE_BUFFERS 1
#endif
#ifndef LANE_K3_WINDOW
#define LANE_K3_WINDOW 0
#endif
uint lane = thread_index_in_simdgroup;
uint column_block = threadgroup_position_in_grid.x;
uint token_block = threadgroup_position_in_grid.y;
uint split = threadgroup_position_in_grid.z;
device half* activation = const_cast<device half*>(xhat);

constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
    LANE_COLS, LANE_TOKENS, LANE_DEPTH, false, true, false,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;
auto left = operation.get_left_input_cooperative_tensor<half, half, float>();
using left_type = tensor_ops::matmul2d<descriptor, execution_simdgroup>
    ::cooperative_tensor_left_input_t<half, half, float>;
auto accumulator = operation.get_destination_cooperative_tensor<
    left_type, tensor<device half, dextents<int, 2>, tensor_inline>, float>();
for (ushort index = 0; index < accumulator.get_capacity(); ++index) {
    if (accumulator.is_valid_element(index)) accumulator[index] = 0.0f;
}

// Static element map read from the M5 layouts (Q27D-17); the lane's QMV twin is 4*j + kb.
// Recorded for 32x16, 32x32, 64x16, 32x64 and 64x32: element e holds input
// 4kb + (e&3) + 16 kt and column j + 8 g + 16 tn with e = (e&3) + 4 g + 8 kt + 8 KT tn.
uint kb = (lane & 1u) | ((lane >> 2u) & 2u);
uint j = ((lane >> 1u) & 3u) | ((lane >> 2u) & 4u);
uint qmv_lane = 4u * j + kb;
constexpr uint ELEMENTS = LANE_COLS * LANE_DEPTH / 32u;
bool layout_ok = left.get_capacity() == ELEMENTS;
for (ushort index = 0; index < ELEMENTS; ++index) {
    if (!layout_ok || !left.is_valid_element(index)) { layout_ok = false; break; }
    auto coordinate = left.get_multidimensional_index(index);
    uint e = index;
    // General rule of the recorded layouts: e = i + 4 g + 8 kt + 8 KT tn.
    uint want_k = 4u * kb + (e & 3u) + 16u * ((e >> 3u) & (LANE_DEPTH / 16u - 1u));
    uint want_m = j + 8u * ((e >> 2u) & 1u) + 16u * (e / (8u * (LANE_DEPTH / 16u)));
    layout_ok = uint(coordinate[0]) == want_k && uint(coordinate[1]) == want_m;
}
uint word0[2], word1[2], shifts[2];
for (uint group = 0u; group < 2u; ++group) {
    uint first = qmv_lane * 8u + group * 4u;
    int end = int(first + 4u) * int(K_BITS) + 256 * int(K_BITS);
    int last_word = (end - 1) / 32;
    word1[group] = uint(last_word) % PACKED_U32;
    word0[group] = uint(last_word - 1) % PACKED_U32;
    shifts[group] = uint((last_word + 1) * 32 - end);
}

#if LANE_K3_WINDOW
// DFLASH2-2026-11-24-INK2M-P4-1e: K = 3 reads the lane's 8 codewords of a tile from one 64-bit window (the QMV's
// `MLXL3_K3_WINDOW_DECODE` formulas): the same codewords as the two windows of the generic path.
int k3_bit_end = int((qmv_lane * 8u + 257u) * 3u + 21u);
int k3_word_end = (k3_bit_end - 1) / 32;
uint k3_word1 = uint(k3_word_end) % PACKED_U32;
uint k3_word0 = uint((k3_bit_end - 21 - 16) / 32) % PACKED_U32;
uint k3_shift = uint((k3_word_end + 1) * 32 - k3_bit_end);
#endif
constexpr uint K_TILES = LANE_DEPTH / 16u;
constexpr uint N_TILES = LANE_COLS / 16u;
uint steps = INPUT_DIMS / LANE_DEPTH;
uint per_split = (steps + N_SPLITS - 1u) / N_SPLITS;
uint step_begin = split * per_split;
uint step_end = min(step_begin + per_split, steps);
// Q27D-28: LANE_BUFFERS 2 decodes step s + 1 into a second operand while step s runs; the
// operations and their order on the accumulator are unchanged (same values bit for bit).
auto decode_step = [&](thread left_type& left, uint step) {
    #pragma unroll
    for (uint ktile = 0u; ktile < K_TILES; ++ktile) {
        #pragma unroll
        for (uint ntile = 0u; ntile < N_TILES; ++ntile) {
#if LANE_K3_WINDOW
            {
                uint tile_k = step * K_TILES + ktile;
                uint tile_n = column_block * N_TILES + ntile + WEIGHT_TILE_OFFSET;
                const device uint* words = trellis + (ulong(tile_k) * TILES_N + tile_n) * PACKED_U32;
                ulong merged = (ulong(words[k3_word0]) << 32) | ulong(words[k3_word1]);
                uint window7 = uint(merged >> k3_shift);
                uint window3 = uint(merged >> (k3_shift + 12u));
                uint base = 8u * K_TILES * ntile + 8u * ktile;
                left[base + 0u] = half(mlxl3_decode_codeword(window3 >> 9u, 0));
                left[base + 1u] = half(mlxl3_decode_codeword(window3 >> 6u, 0));
                left[base + 2u] = half(mlxl3_decode_codeword(window3 >> 3u, 0));
                left[base + 3u] = half(mlxl3_decode_codeword(window3, 0));
                left[base + 4u] = half(mlxl3_decode_codeword(window7 >> 9u, 0));
                left[base + 5u] = half(mlxl3_decode_codeword(window7 >> 6u, 0));
                left[base + 6u] = half(mlxl3_decode_codeword(window7 >> 3u, 0));
                left[base + 7u] = half(mlxl3_decode_codeword(window7, 0));
            }
            continue;
#endif
            #pragma unroll
            for (uint group = 0u; group < 2u; ++group) {
                uint tile_k = step * K_TILES + ktile;
                uint tile_n = column_block * N_TILES + ntile + WEIGHT_TILE_OFFSET;
                const device uint* words = trellis + (ulong(tile_k) * TILES_N + tile_n) * PACKED_U32;
                ulong merged = (ulong(words[word0[group]]) << 32) | ulong(words[word1[group]]);
                uint shift = shifts[group];
                uint bits = uint(merged >> shift);
                uint cw3 = bits & 0xffffu;
                uint cw2 = (bits >> K_BITS) & 0xffffu;
                uint cw1 = (bits >> (2u * K_BITS)) & 0xffffu;
                uint cw0 = K_BITS <= 5u
                    ? (bits >> (3u * K_BITS)) & 0xffffu
                    : uint(merged >> (shift + 3u * K_BITS)) & 0xffffu;
                uint base = 8u * K_TILES * ntile + 8u * ktile + 4u * group;
                left[base + 0u] = half(mlxl3_decode_codeword(cw0, 0));
                left[base + 1u] = half(mlxl3_decode_codeword(cw1, 0));
                left[base + 2u] = half(mlxl3_decode_codeword(cw2, 0));
                left[base + 3u] = half(mlxl3_decode_codeword(cw3, 0));
            }
        }
    }
};
auto run_step = [&](thread left_type& left, uint step) {
    auto right = tensor(
        activation + ulong(token_block * LANE_TOKENS) * INPUT_DIMS + step * LANE_DEPTH,
        dextents<int, 2>{int(LANE_DEPTH), int(LANE_TOKENS)},
        array<int, 2>{1, int(INPUT_DIMS)}
    );
    operation.run(left, right, accumulator);
};
#if LANE_BUFFERS == 2
auto left_next = operation.get_left_input_cooperative_tensor<half, half, float>();
uint step = step_begin;
if (step < step_end) decode_step(left, step);
for (; step + 1u < step_end; step += 2u) {
    decode_step(left_next, step + 1u);
    run_step(left, step);
    if (step + 2u < step_end) decode_step(left, step + 2u);
    run_step(left_next, step + 1u);
}
if (step < step_end) run_step(left, step);
#else
for (uint step = step_begin; step < step_end; ++step) {
    decode_step(left, step);
    run_step(left, step);
}
#endif

for (ushort index = 0; index < accumulator.get_capacity(); ++index) {
    if (!accumulator.is_valid_element(index)) continue;
    auto coordinate = accumulator.get_multidimensional_index(index);
    uint token = token_block * LANE_TOKENS + uint(coordinate[0]);
    uint column = column_block * LANE_COLS + uint(coordinate[1]);
    // A layout that differs from the recorded one poisons the output instead of mixing weights.
    partials[(ulong(split) * PADDED_ROWS + token) * OUTPUT_DIMS + column] =
        layout_ok ? accumulator[index] : NAN;
}
