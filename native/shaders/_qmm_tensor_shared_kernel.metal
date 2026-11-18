// Q27P-02: prefill EXL3 product on the M5 TensorOps unit with the decoded weight tile shared
// by ROW_GROUP simdgroups. Each simdgroup owns one BM-row block and runs exactly the operations
// of `_qmm_tensor_kernel.metal` (same descriptor, same operands, same order on its accumulator),
// so outputs are expected bit for bit; the BK x BN weight tile of each depth step is decoded once
// per threadgroup into threadgroup memory instead of once per row block.
uint column_block = threadgroup_position_in_grid.x;
uint row_group = threadgroup_position_in_grid.y;
uint simdgroup = simdgroup_index_in_threadgroup;
uint lane_thread = thread_index_in_threadgroup;
uint row_block = row_group * ROW_GROUP + simdgroup;
// Row blocks are read at run time so one library serves every chunk length.
uint row_blocks = uint(xhat_shape[0]) / BM;
bool active = row_block < row_blocks;
device half* activation = const_cast<device half*>(xhat);

constexpr auto descriptor = tensor_ops::matmul2d_descriptor(
    BM,
    BN,
    BK,
    false,
    false,
    false,
    tensor_ops::matmul2d_descriptor::mode::multiply_accumulate
);
tensor_ops::matmul2d<descriptor, execution_simdgroup> operation;
auto right = operation.get_right_input_cooperative_tensor<half, half, float>();
auto first_left = tensor(
    activation + ulong(min(row_block, row_blocks - 1u) * BM) * INPUT_DIMS,
    dextents<int, 2>{int(BK), int(BM)},
    array<int, 2>{1, int(INPUT_DIMS)}
);
using right_type = tensor_ops::matmul2d<descriptor, execution_simdgroup>
    ::cooperative_tensor_right_input_t<half, half, float>;
auto accumulator = operation.get_destination_cooperative_tensor<
    tensor<device half, dextents<int, 2>, tensor_inline>, right_type, float
>();
for (ushort index = 0; index < accumulator.get_capacity(); ++index) {
    if (accumulator.is_valid_element(index)) accumulator[index] = 0.0f;
}

// Decode assignment: element e = thread + j * THREADS of the BK x BN tile, (input row, column)
// = (e / BN, e % BN); same addressing and codeword extraction as the stock kernel.
constexpr uint THREADS = 32u * ROW_GROUP;
constexpr uint PER_THREAD = BK * BN / THREADS;
uint offset0[PER_THREAD], offset1[PER_THREAD], shifts[PER_THREAD];
for (uint j = 0u; j < PER_THREAD; ++j) {
    uint e = lane_thread + j * THREADS;
    uint input_row = e / BN;
    uint output_column = column_block * BN + e % BN;
    uint tile_n = (output_column >> 4u) + WEIGHT_TILE_OFFSET;
    uint local = (input_row & 15u) * 16u + (output_column & 15u);
    uint source = uint(mlxl3_perm_inv[local]);
    uint position = source >> 1u;
    int begin = int(position * 2u * K_BITS + K_BITS) - 16 + int(256u * K_BITS);
    int end = begin + int(K_BITS) + 16;
    uint word0 = uint(begin / 32) % PACKED_U32;
    uint word1 = uint((end - 1) / 32) % PACKED_U32;
    // BK 32 (Q27P-03) spans two 16-deep trellis tiles per step.
    uint tile_base = ((input_row >> 4u) * TILES_N + tile_n) * PACKED_U32;
    offset0[j] = tile_base + word0;
    offset1[j] = tile_base + word1;
    shifts[j] = uint(((end - 1) / 32 + 1) * 32 - end) + ((source & 1u) ? 0u : K_BITS);
}
// Where each lane reads its cooperative right elements in the shared tile.
constexpr uint RIGHT_CAPACITY = BN * BK / 32u;
ushort right_slot[RIGHT_CAPACITY];
for (ushort index = 0; index < RIGHT_CAPACITY; ++index) {
    right_slot[index] = 0xffffu;
    if (index < right.get_capacity() && right.is_valid_element(index)) {
        auto coordinate = right.get_multidimensional_index(index);
        right_slot[index] = ushort(uint(coordinate[1]) * BN + uint(coordinate[0]));
    }
}
threadgroup half shared_tile[2][BK * BN];

uint step = 0u;
for (uint depth = 0u; depth < INPUT_DIMS; depth += BK, ++step) {
    threadgroup half* tile = shared_tile[step & 1u];
    const device uint* words = trellis + ulong(depth / 16u) * TILES_N * PACKED_U32;
    for (uint j = 0u; j < PER_THREAD; ++j) {
        ulong merged = (ulong(words[offset0[j]]) << 32) | ulong(words[offset1[j]]);
        uint codeword = uint(merged >> shifts[j]) & 0xffffu;
        tile[lane_thread + j * THREADS] = half(mlxl3_decode_codeword(codeword, 0));
    }
    // One barrier per step: the other buffer is rewritten only after the next barrier.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (ushort index = 0; index < RIGHT_CAPACITY; ++index) {
        if (right_slot[index] != 0xffffu) right[index] = tile[right_slot[index]];
    }
    if (active) {
        auto left = tensor(
            activation + ulong(row_block * BM) * INPUT_DIMS + depth,
            dextents<int, 2>{int(BK), int(BM)},
            array<int, 2>{1, int(INPUT_DIMS)}
        );
        operation.run(left, right, accumulator);
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
