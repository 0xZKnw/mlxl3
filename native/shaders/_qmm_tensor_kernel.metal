uint column_block = threadgroup_position_in_grid.x;
uint row_block = threadgroup_position_in_grid.y;
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
    activation + ulong(row_block * BM) * INPUT_DIMS,
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

constexpr uint ADDRESS_CAPACITY = BN * BK / 32u;
uint offset0[ADDRESS_CAPACITY], offset1[ADDRESS_CAPACITY];
uint shifts[ADDRESS_CAPACITY];
if (right.get_capacity() <= ADDRESS_CAPACITY) {
    for (ushort index = 0; index < right.get_capacity(); ++index) {
        if (!right.is_valid_element(index)) continue;
        auto coordinate = right.get_multidimensional_index(index);
        uint output_column = column_block * BN + uint(coordinate[0]);
        uint input_row = uint(coordinate[1]);
        uint tile_k = input_row >> 4u;
        uint tile_n = (output_column >> 4u) + WEIGHT_TILE_OFFSET;
        uint local = (input_row & 15u) * 16u + (output_column & 15u);
        uint source = uint(mlxl3_perm_inv[local]);
        uint position = source >> 1u;
        int begin = int(position * 2u * K_BITS + K_BITS)
            - 16 + int(256u * K_BITS);
        int end = begin + int(K_BITS) + 16;
        uint word0 = uint(begin / 32) % PACKED_U32;
        uint word1 = uint((end - 1) / 32) % PACKED_U32;
        uint shift = uint(((end - 1) / 32 + 1) * 32 - end)
            + ((source & 1u) ? 0u : K_BITS);
        offset0[index] = (tile_k * TILES_N + tile_n) * PACKED_U32 + word0;
        offset1[index] = (tile_k * TILES_N + tile_n) * PACKED_U32 + word1;
        shifts[index] = shift;
    }
}

for (uint depth = 0u; depth < INPUT_DIMS; depth += BK) {
    for (ushort index = 0; index < right.get_capacity(); ++index) {
        if (!right.is_valid_element(index)) continue;
        uint word0, word1, shift;
        const device uint* words;
        if (right.get_capacity() <= ADDRESS_CAPACITY) {
            words = trellis + ulong(depth / 16u) * TILES_N * PACKED_U32;
            word0 = offset0[index];
            word1 = offset1[index];
            shift = shifts[index];
        } else {
            auto coordinate = right.get_multidimensional_index(index);
            uint output_column = column_block * BN + uint(coordinate[0]);
            uint input_row = depth + uint(coordinate[1]);
            uint tile_k = input_row >> 4u;
            uint tile_n = (output_column >> 4u) + WEIGHT_TILE_OFFSET;
            uint local = (input_row & 15u) * 16u + (output_column & 15u);
            uint source = uint(mlxl3_perm_inv[local]);
            uint position = source >> 1u;
            int begin = int(position * 2u * K_BITS + K_BITS)
                - 16 + int(256u * K_BITS);
            int end = begin + int(K_BITS) + 16;
            word0 = uint(begin / 32) % PACKED_U32;
            word1 = uint((end - 1) / 32) % PACKED_U32;
            shift = uint(((end - 1) / 32 + 1) * 32 - end)
                + ((source & 1u) ? 0u : K_BITS);
            words = trellis + ulong(tile_k * TILES_N + tile_n) * PACKED_U32;
        }
        ulong merged = (ulong(words[word0]) << 32) | ulong(words[word1]);
        uint codeword = uint(merged >> shift) & 0xffffu;
        right[index] = half(mlxl3_decode_codeword(codeword, 0));
    }
    auto left = tensor(
        activation + ulong(row_block * BM) * INPUT_DIMS + depth,
        dextents<int, 2>{int(BK), int(BM)},
        array<int, 2>{1, int(INPUT_DIMS)}
    );
    operation.run(left, right, accumulator);
}

for (ushort index = 0; index < accumulator.get_capacity(); ++index) {
    if (!accumulator.is_valid_element(index)) continue;
    auto coordinate = accumulator.get_multidimensional_index(index);
    uint output_row = row_block * BM + uint(coordinate[1]);
    uint output_column = column_block * BN + uint(coordinate[0]);
    yhat[ulong(output_row) * OUTPUT_DIMS + output_column] = half(accumulator[index]);
}
