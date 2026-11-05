// Each of the four streams is normalized independently. Accumulate the same four
// squares per lane as MLX's contiguous last-axis reduction.
kernel void hc_write_norm_float(
    const device float* streams [[buffer(0)]], const device void* branch [[buffer(1)]],
    const device float* gate [[buffer(2)]], const device float* gain [[buffer(3)]],
    device float* written [[buffer(4)]], device float* normalized [[buffer(5)]],
    constant long* layout [[buffer(6)]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    #pragma clang fp contract(off)
    threadgroup float partial[20];
    float values[4], sum=0;
    uint base=group*2560+tid*4;
    for(uint j=0;j<4;++j) {
        long b=layout[2]+long((base+j)%2560)*layout[3];
        long g=layout[4]+long(group)*layout[5];
        float delta=layout[6] ? ((const device float*)branch)[b] : float(((const device half*)branch)[b]);
        volatile float product=delta*gate[g];
        volatile float value=streams[layout[0]+long(base+j)*layout[1]]+product;
        values[j]=value;
        written[base+j]=value;
        sum+=values[j]*values[j];
    }
    sum=simd_sum(sum);
    if(lane==0) partial[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        float total=simd_sum(lane<20?partial[lane]:0.0f);
        if(lane==0) partial[0]=rsqrt(total*(1.0f/2560.0f)+as_type<float>(uint(layout[7])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<4;++j) normalized[base+j]=(values[j]*partial[0])*(1.0f+gain[base+j]);
}

template <typename T>
kernel void hc_norm(const device T* x [[buffer(0)]], const device float* gain [[buffer(1)]],
    device T* out [[buffer(2)]], constant long* layout [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]], uint group [[threadgroup_position_in_grid]]) {
    #pragma clang fp contract(off)
    threadgroup float partial[20];
    float values[4];
    float sum=0;
    uint base=group*2560+tid*4;
    for(uint j=0;j<4;++j) {
        values[j]=float(x[layout[0]+long(base+j)*layout[1]]);
        sum+=values[j]*values[j];
    }
    sum=simd_sum(sum);
    if(lane==0) partial[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        float total=simd_sum(lane<20?partial[lane]:0.0f);
        if(lane==0) partial[0]=rsqrt(total*(1.0f/2560.0f)+as_type<float>(uint(layout[2])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=0;j<4;++j) out[base+j]=T((values[j]*partial[0])*(1.0f+gain[base+j]));
}

#undef K
#define HC_GEMV_PARAMS(T) const device T* x [[buffer(0)]], const device half* weight [[buffer(1)]], \
    device T* out [[buffer(2)]], constant uint* dims [[buffer(3)]], \
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], \
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], \
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]
template <typename T>
kernel void hc_narrow(HC_GEMV_PARAMS(T)) {
#define K 10240
#define BM 1
#define BN 8
#include "flash_hc_gemv.metal"
#undef K
#undef BM
#undef BN
}
template <typename T>
kernel void hc_wide(HC_GEMV_PARAMS(T)) {
#define K 320
#define BM 8
#define BN 1
#include "flash_hc_gemv.metal"
#undef K
#undef BM
#undef BN
}

template <typename T> inline T hc_sigmoid(T x) {
    T tail=T(1)/(T(1)+exp(abs(x)));
    return x<T(0)?tail:T(1)-tail;
}
// The compiled activation uses fast exp. Sigmoid before the mix reshape uses the ordinary exponent.
inline half hc_compiled_sigmoid(half x) { return hc_sigmoid(x); }
inline float hc_compiled_sigmoid(float x) {
    float tail=1.0f/(1.0f+fast::exp(abs(x)));
    return x<0.0f?tail:1.0f-tail;
}
// Both activations operate in the source dtype, including the division by four.
template <typename T>
kernel void hc_activate(const device T* x [[buffer(0)]], device T* out [[buffer(1)]],
    constant uint* params [[buffer(2)]], uint col [[thread_position_in_grid]]) {
    if(col>=params[0]) return;
    T y=x[col]/T(4);
    out[col]=(params[1]?T(2):y)*hc_compiled_sigmoid(y);
}
template <typename T>
kernel void hc_mix(const device T* weights [[buffer(0)]], const device T* normalized [[buffer(1)]],
    device T* out [[buffer(2)]], constant uint* dims [[buffer(3)]], uint col [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    T sum=0;
    for(uint stream=0;stream<4;++stream) {
        uint i=stream*2560+col;
        sum+=hc_sigmoid(weights[i])*normalized[i];
    }
    out[col]=sum*T(0.25);
}
template <typename T>
kernel void hc_write(const device T* streams [[buffer(0)]], const device void* branch [[buffer(1)]],
    const device T* gate [[buffer(2)]], device void* out [[buffer(3)]], constant long* layout [[buffer(4)]],
    uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    long b=layout[2]+long(i%2560)*layout[3], g=layout[4]+long(i/2560)*layout[5];
    T residual=streams[layout[0]+long(i)*layout[1]];
    if(layout[6]) {
        ((device float*)out)[i]=float(residual)+((const device float*)branch)[b]*float(gate[g]);
    } else {
        ((device T*)out)[i]=residual+T(((const device half*)branch)[b])*gate[g];
    }
}

#define HC_KERNELS(T, NORM, NARROW, WIDE, ACTIVE, MIX, WRITE) \
template [[host_name(NORM)]] kernel void hc_norm<T>(const device T*,const device float*,device T*,constant long*,uint,uint,uint,uint); \
template [[host_name(NARROW)]] kernel void hc_narrow<T>(HC_GEMV_PARAMS(T)); \
template [[host_name(WIDE)]] kernel void hc_wide<T>(HC_GEMV_PARAMS(T)); \
template [[host_name(ACTIVE)]] kernel void hc_activate<T>(const device T*,device T*,constant uint*,uint); \
template [[host_name(MIX)]] kernel void hc_mix<T>(const device T*,const device T*,device T*,constant uint*,uint); \
template [[host_name(WRITE)]] kernel void hc_write<T>(const device T*,const device void*,const device T*,device void*,constant long*,uint);
HC_KERNELS(half,"hc_norm_half","hc_narrow_half","hc_wide_half","hc_activate_half","hc_mix_half","hc_write_half")
HC_KERNELS(float,"hc_norm_float","hc_narrow_float","hc_wide_float","hc_activate_float","hc_mix_float","hc_write_float")
#undef HC_KERNELS
#undef HC_GEMV_PARAMS

// F32-only exact up GEMV + four-stream mix. Each SIMD group keeps hc_wide's
// K320/BN1 FMA and shuffle sequence; only independent output rows are regrouped.
kernel void hc_up_mix_float(const device float* x [[buffer(0)]],
    const device half* weight [[buffer(1)]], const device float* normalized [[buffer(2)]],
    device float* out [[buffer(3)]], constant uint* dims [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    #pragma clang fp contract(off)
    uint stream = sg / 2;
    uint local = (sg % 2) * 4;
    uint output = stream * 2560 + group * 8 + local;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    uint column = lane * 4;
    for (uint block = 0; block < (320 + 128 - 1) / 128; ++block) {
        float coeff[4];
        for (uint n = 0; n < 4; ++n) coeff[n] = column + n < 320 ? x[column + n] : 0.0f;
        for (uint m = 0; m < 4; ++m) {
            for (uint n = 0; n < 4; ++n) {
                float w = column + n < 320 ? float(weight[(output + m) * 320 + column + n]) : 0.0f;
                result[m] = fma(w, coeff[n], result[m]);
            }
        }
        column += 128;
    }
    for (uint m = 0; m < 4; ++m) {
        for (ushort delta = 16; delta >= 1; delta >>= 1) result[m] += simd_shuffle_down(result[m], delta);
    }
    threadgroup float up[32];
    if (lane == 0) {
        for (uint m = 0; m < 4; ++m) up[stream * 8 + local + m] = float(result[m]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < 8) {
        float sum = 0;
        for (uint s = 0; s < 4; ++s) {
            uint i = s * 2560 + group * 8 + lane;
            sum += hc_sigmoid(up[s * 8 + lane]) * normalized[i];
        }
        out[group * 8 + lane] = sum * 0.25f;
    }
}

// The final group computes injection's four rows, with its own zero-offset output.
// All groups reuse the original narrow GEMV body and reduction order verbatim.
kernel void hc_down_inject_float(const device float* x [[buffer(0)]],
    const device half* down_weight [[buffer(1)]], const device half* inject_weight [[buffer(2)]],
    device float* down_out [[buffer(3)]], device float* inject_out [[buffer(4)]],
    constant uint* dims [[buffer(5)]], uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    const device half* weight = group.x < 80 ? down_weight : inject_weight;
    device float* out = group.x < 80 ? down_out : inject_out;
    uint3 threadgroup_position_in_grid = uint3(group.x < 80 ? group.x : 0, 0, 0);
#define K 10240
#define BM 1
#define BN 8
#include "flash_hc_gemv.metal"
#undef K
#undef BM
#undef BN
}
