// Affine Q4 (group 64, StorageN=256) M=8 projection for Apple TensorOps.
// The packed ABI and math match IncoAI Splash; MLXL3 owns dispatch/tuning.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include <metal_stdlib>
#include <metal_tensor>

using namespace metal;
using namespace mpp::tensor_ops;

template <class Tensor, class Body>
__attribute__((always_inline)) inline void dflash_visit(
    const thread Tensor &values, const thread Body &body) {
#pragma unroll
  for (ushort index = 0; index < values.get_capacity(); ++index)
    if (values.is_valid_element(index)) body(index);
}

inline void dflash_input_sums(device const bfloat *input, uint input_origin,
                              threadgroup float *sums, uint sum_origin,
                              uint simd_lane, uint simd_group) {
  uint origin = simd_group * DFLASH_INPUT + input_origin + simd_lane;
  float first = simd_sum(float(input[origin]) + float(input[origin + 32]));
  float second =
      simd_sum(float(input[origin + 64]) + float(input[origin + 96]));
  float third =
      simd_sum(float(input[origin + 128]) + float(input[origin + 160]));
  float fourth =
      simd_sum(float(input[origin + 192]) + float(input[origin + 224]));
  if (simd_lane == 0) {
    sums[sum_origin + simd_group] = first;
    sums[sum_origin + 8 + simd_group] = second;
    sums[sum_origin + 16 + simd_group] = third;
    sums[sum_origin + 24 + simd_group] = fourth;
  }
}

template <ushort TileN, bool Pipelined>
inline void dflash_q4_tile(device bfloat *input, device uchar *weights,
                           device bfloat *scales, device bfloat *biases,
                           device bfloat *output,
                           threadgroup float *input_sums, uint output_origin,
                           uint simd_lane, uint simd_group) {
  auto activation = tensor(input, dextents<int, 2>{DFLASH_INPUT, 8},
                           array<int, 2>{1, DFLASH_INPUT});
  constexpr auto descriptor =
      matmul2d_descriptor(8, TileN, 64, false, true, false);
  matmul2d<descriptor, execution_simdgroups<8>> operation;
  auto first_activation = activation.slice<64, 8>(0, 0);
  constexpr uint quant_groups = DFLASH_INPUT / 64;
  uint tile = output_origin / 256;
  uint tile_offset = output_origin % 256;
  device uchar *tile_weights =
      weights + ulong(tile) * quant_groups * 256 * 64 / 2;
  tensor<device uint4b_format, dextents<int, 2>, tensor_inline> first_weight(
      tile_weights + tile_offset * 32, dextents<int, 2>{64, TileN},
      array<int, 2>{1, 64});
  auto first_weight_slice = first_weight.slice<64, TileN>(0, 0);
  auto accumulated = operation.template get_destination_cooperative_tensor<
      decltype(first_activation), decltype(first_weight_slice), float>();
  dflash_visit(accumulated,
               [&](ushort index) { accumulated[index] = 0.0f; });

  dflash_input_sums(input, 0, input_sums, 0, simd_lane, simd_group);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  auto run_group = [&](uint group, thread decltype(accumulated) &partial) {
    uint input_origin = group * 64;
    auto input_slice = activation.slice<64, 8>(input_origin, 0);
    device uchar *group_weights =
        tile_weights + (ulong(group) * 256 + tile_offset) * 64 / 2;
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> weight(
        group_weights, dextents<int, 2>{64, TileN}, array<int, 2>{1, 64});
    auto weight_slice = weight.slice<64, TileN>(0, 0);
    operation.run(input_slice, weight_slice, partial);
  };
  auto finish_group = [&](uint group, thread decltype(accumulated) &partial) {
    dflash_visit(accumulated,
                 [&](ushort index) __attribute__((always_inline)) {
      auto coordinate = accumulated.get_multidimensional_index(index);
      uint row = coordinate[1];
      ulong parameter =
          (ulong(tile) * quant_groups + group) * 256 + tile_offset + coordinate[0];
      uint sum_offset = ((group >> 2) & 1) * 32 + (group & 3) * 8;
      accumulated[index] +=
          partial[index] * float(scales[parameter]) +
          input_sums[sum_offset + row] * float(biases[parameter]);
    });
    if ((group & 3) == 3 && group + 1 < quant_groups) {
      uint next = (group + 1) >> 2;
      dflash_input_sums(input, group * 64 + 64, input_sums,
                        (next & 1) * 32, simd_lane, simd_group);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  };
  if constexpr (Pipelined) {
    uint group = 0;
    for (; group + 1 < quant_groups; group += 2) {
      decltype(accumulated) first, second;
      run_group(group, first);
      run_group(group + 1, second);
      finish_group(group, first);
      finish_group(group + 1, second);
    }
    if (group < quant_groups) {
      decltype(accumulated) partial;
      run_group(group, partial);
      finish_group(group, partial);
    }
  } else {
    for (uint group = 0; group < quant_groups; ++group) {
      decltype(accumulated) partial;
      run_group(group, partial);
      finish_group(group, partial);
    }
  }

  dflash_visit(accumulated, [&](ushort index) {
    auto coordinate = accumulated.get_multidimensional_index(index);
    uint output_index =
        coordinate[1] * DFLASH_OUTPUT + output_origin + coordinate[0];
    output[output_index] = bfloat(accumulated[index]);
  });
  threadgroup_barrier(mem_flags::mem_threadgroup);
}
