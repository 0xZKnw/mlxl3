#include <metal_stdlib>
using namespace metal;

// One SIMD group owns one output. The host binds the selected expert's slice.
kernel void qwen_mtp_affine4(
    device const half *input [[buffer(0)]],
    device const uint *weights [[buffer(1)]],
    device const half *scales [[buffer(2)]],
    device const half *biases [[buffer(3)]],
    device half *output [[buffer(4)]],
    constant uint2 &shape [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint simd [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    uint row = group * 4 + simd;
    if (row >= shape.y) return;
    uint words = row * (shape.x / 8);
    uint groups = row * (shape.x / 64);
    float sum = 0;
    for (uint column = lane; column < shape.x; column += 32) {
        uint nibble = (weights[words + column / 8] >> (4 * (column % 8))) & 15;
        uint index = groups + column / 64;
        float weight = fma(float(nibble), float(scales[index]), float(biases[index]));
        sum = fma(float(input[column]), weight, sum);
    }
    sum = simd_sum(sum);
    if (lane == 0) output[row] = half(sum);
}
