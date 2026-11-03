template <uint BM>
kernel void ple_gemv(const device half* x [[buffer(0)]],const device half* weight [[buffer(1)]],device half* out [[buffer(2)]],
    constant uint* extent [[buffer(3)]], uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
#define K 2560
#define BN 1
#include "flash_hc_gemv.metal"
#undef BN
#undef K
}
#define PLE_DENSE(NAME,BM) template [[host_name(NAME)]] kernel void ple_gemv<BM>(const device half*,const device half*,device half*,constant uint*,uint,uint,uint3);
PLE_DENSE("ple_key",8)
PLE_DENSE("ple_value",4)
#undef PLE_DENSE

template <typename T>
kernel void ple_route(const device half* key [[buffer(0)]],const device T* query [[buffer(1)]],device float* raw [[buffer(2)]],
    device float* gate [[buffer(3)]],constant float* params [[buffer(4)]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]],uint stream [[threadgroup_position_in_grid]]) {
    #pragma clang fp contract(off)
    threadgroup T partial[20];
    T sum=0;
    uint base=stream*2560+tid*4;
    for(uint j=0;j<4;++j) sum+=T(key[base+j])*query[base+j];
    sum=simd_sum(sum);
    if(lane==0) partial[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        T total=simd_sum(lane<20?partial[lane]:T(0));
        if(tid==0) {
            float dot=float(total)/params[0];
            raw[stream]=dot;
            float score=sign(dot)*precise::sqrt(max(abs(dot),1.0e-6f));
            gate[stream]=hc_sigmoid(score);
        }
    }
}
kernel void ple_gated(const device half* value [[buffer(0)]],const device float* gate [[buffer(1)]],device float* out [[buffer(2)]],
    constant uint* extent [[buffer(3)]],uint col [[thread_position_in_grid]]) {out[col]=float(value[col%2560])*gate[col/2560];}

template <typename T>
kernel void ple_residual(const device T* streams [[buffer(0)]],const device float* gated [[buffer(1)]],const device float* normed [[buffer(2)]],
    const device float* history [[buffer(3)]],const device half* weight [[buffer(4)]],device float* next [[buffer(5)]],
    device float* conv_raw [[buffer(6)]],device float* convolution [[buffer(7)]],device float* out [[buffer(8)]],
    constant long* layout [[buffer(9)]],uint col [[thread_position_in_grid]]) {
    float acc=0;
    for(uint tap=0;tap<4;++tap) {
        float value=tap<3?history[tap*3*10240+col]:normed[col];
        acc=fma(value,float(weight[col*4+tap]),acc);
    }
    conv_raw[col]=acc;
    float active=acc*hc_compiled_sigmoid(acc);
    convolution[col]=active;
    out[col]=float(streams[layout[0]+long(col)*layout[1]])+(gated[col]+active);
    for(uint t=0;t<9;++t) next[t*10240+col]=t<8?history[(t+1)*10240+col]:normed[col];
}
#define PLE_KERNELS(T,R,W) \
template [[host_name(R)]] kernel void ple_route<T>(const device half*,const device T*,device float*,device float*,constant float*,uint,uint,uint,uint); \
template [[host_name(W)]] kernel void ple_residual<T>(const device T*,const device float*,const device float*,const device float*,const device half*,device float*,device float*,device float*,device float*,constant long*,uint);
PLE_KERNELS(half,"ple_route_half","ple_residual_half")
PLE_KERNELS(float,"ple_route_float","ple_residual_float")
#undef PLE_KERNELS
