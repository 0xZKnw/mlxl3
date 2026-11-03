// Qwen preparation and state equations reuse the maintained Rust/MLX kernels.
#include <metal_stdlib>
using namespace metal;

kernel void qwen_dense(const device half* x [[buffer(0)]], const device half* weight [[buffer(1)]],
    device half* out [[buffer(2)]], constant uint* extent [[buffer(3)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
#define K HIDDEN
#define BM 1
#define BN 8
#include "flash_hc_gemv.metal"
#undef K
#undef BM
#undef BN
}
inline half qwen_input(const device half* qkv, const device half* history, uint t, uint channel) {
    return t < CONV_HISTORY ? history[t * CONV_DIMS + channel] : qkv[(t - CONV_HISTORY) * CONV_DIMS + channel];
}
inline half mlxl3_gdn_input(const device half* qkv, const device half* conv_state, uint position, uint channel) {
    return qwen_input(qkv, conv_state, position, channel);
}
inline half mlxl3_gdn_conv_silu(const device half* qkv, const device half* history,
    const device half* weight, uint t, uint channel) {
    float sum = 0;
    for (uint i = 0; i < CONV_LENGTH; ++i) sum += float(qwen_input(qkv, history, t+i, channel)) * float(weight[channel*CONV_LENGTH+i]);
    half value = half(sum);
    half tail = half(1) / (half(1) + metal::exp(metal::abs(value)));
    half probability = value < half(0) ? tail : half(1)-tail;
    return value * probability;
}
kernel void qwen_prepare(const device half* qkv [[buffer(0)]], const device half* conv_state [[buffer(1)]],
    const device half* conv_weight [[buffer(2)]], device half* q [[buffer(3)]], device half* k [[buffer(4)]],
    device half* v [[buffer(5)]], device half* state_out [[buffer(6)]], constant uint* unused [[buffer(7)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
    half q_scale = half(1.0f / 128.0f), k_scale = half(0.08838834764831845f);
#define TIME 1
#include "gated_delta_prepare.metal"
#undef TIME
}
kernel void qwen_step(const device half* q [[buffer(0)]], const device half* k [[buffer(1)]],
    const device half* v [[buffer(2)]], const device half* a [[buffer(3)]], const device half* b [[buffer(4)]],
    const device float* a_log [[buffer(5)]], const device half* dt_bias [[buffer(6)]], const device float* state_in [[buffer(7)]],
    device half* y [[buffer(8)]], device float* state_out [[buffer(9)]], constant uint* unused [[buffer(10)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]], uint3 group [[threadgroup_position_in_grid]]) {
    uint3 thread_position_in_grid = uint3(thread_index_in_simdgroup, group.y * 2 + sg, group.z);
#define MLXL3_GDN_FUSED_GATES 1
#define MLXL3_GDN_SAVE_HISTORY 0
#define MLXL3_GDN_SAVE_TAPE 0
#define InT half
#define StT float
#define B 1
#define T 1
#define Dk 128
#define Dv 128
#define Hk KEY_HEADS
#define Hv VALUE_HEADS
#include "gated_delta_packed.metal"
#undef MLXL3_GDN_FUSED_GATES
#undef MLXL3_GDN_SAVE_HISTORY
#undef MLXL3_GDN_SAVE_TAPE
#undef InT
#undef StT
#undef B
#undef T
#undef Dk
#undef Dv
#undef Hk
#undef Hv
}
kernel void qwen_norm_gate(const device half* value [[buffer(0)]], const device half* z [[buffer(1)]],
    const device half* gain [[buffer(2)]], device half* output [[buffer(3)]], constant uint* p [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]], uint head [[threadgroup_position_in_grid]]) {
    #pragma clang fp contract(off)
    float x[4], total=0;
    for (uint j=0;j<4;++j) { x[j]=float(value[head*128+lane*4+j]); total+=x[j]*x[j]; }
    float inverse=precise::rsqrt(simd_sum(total)/128.0f+as_type<float>(p[0]));
    for (uint j=0;j<4;++j) {
        uint at=head*128+lane*4+j;
        volatile half unit=half(x[j]*inverse);
        volatile half normalized=unit*gain[lane*4+j];
        float gate=float(z[at]);
        float tail=1.0f/(1.0f+fast::exp(metal::abs(gate)));
        volatile float probability=gate<0.0f ? tail : 1.0f-tail;
        volatile float activated=gate*probability;
        output[at]=half(activated*float(normalized));
    }
}
