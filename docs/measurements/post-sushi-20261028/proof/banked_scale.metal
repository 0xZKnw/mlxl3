#include <metal_stdlib>
using namespace metal;
kernel void banked_scale_wrapper(
    device const half* x [[buffer(0)]],
    device const half* bank [[buffer(1)]],
    device const uint* slots [[buffer(2)]],
    device half* out [[buffer(3)]],
    constant int* x_shape [[buffer(4)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {
#include "_flash_banked_scale.metal"
}
