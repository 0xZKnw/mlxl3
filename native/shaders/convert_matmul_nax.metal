// Cooperative 16x32x16 product, using the QSA's qualified M5 NAX primitive.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void mlxl3_convert_matmul_nax(
    const device float* left [[buffer(0)]],
    const device float* right [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant long* p [[buffer(3)]],
    uint2 tile [[threadgroup_position_in_grid]]) {
    uint m = uint(p[0]), n = uint(p[1]), k = uint(p[2]);
    uint row = tile.y * 16, col = tile.x * 32;
    constexpr auto desc = matmul2d_descriptor(16, 32, 16, false, false, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto a = op.get_left_input_cooperative_tensor<float, float, float>();
    auto b = op.get_right_input_cooperative_tensor<float, float, float>();
    auto c = op.get_destination_cooperative_tensor<decltype(a), decltype(b), float>();
    for (ushort i = 0; i < c.get_capacity(); ++i) c[i] = 0.0f;
    for (uint depth = 0; depth < k; depth += 16) {
        for (ushort i = 0; i < a.get_capacity(); ++i) {
            auto ij = a.get_multidimensional_index(i);
            uint r = row + uint(ij[1]), s = depth + uint(ij[0]);
            a[i] = r < m && s < k ? left[p[3] + long(r) * p[4] + long(s) * p[5]] : 0.0f;
        }
        for (ushort i = 0; i < b.get_capacity(); ++i) {
            auto ij = b.get_multidimensional_index(i);
            uint r = depth + uint(ij[1]), s = col + uint(ij[0]);
            b[i] = r < k && s < n ? right[p[6] + long(r) * p[7] + long(s) * p[8]] : 0.0f;
        }
        op.run(a, b, c);
    }
    for (ushort i = 0; i < c.get_capacity(); ++i) {
        auto ij = c.get_multidimensional_index(i);
        uint r = row + uint(ij[1]), s = col + uint(ij[0]);
        if (r < m && s < n) output[ulong(r) * n + s] = c[i];
    }
}
