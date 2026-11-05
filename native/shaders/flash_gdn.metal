// Flash GDN uses half inputs and emits a float residual branch.
template <typename T>
kernel void gdn_cast_half(const device T* in [[buffer(0)]], device half* out [[buffer(1)]],
    constant long* layout [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    out[i]=half(in[layout[0]+long(i)*layout[1]]);
}
template [[host_name("gdn_from_half")]] kernel void gdn_cast_half<half>(const device half*,device half*,constant long*,uint);
template [[host_name("gdn_from_float")]] kernel void gdn_cast_half<float>(const device float*,device half*,constant long*,uint);

kernel void gdn_dense_gate(const device half* x [[buffer(0)]],const device half* weight [[buffer(1)]],
    device half* out [[buffer(2)]],constant uint* extent [[buffer(3)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
#define K 2560
#define BM 1
#define BN 8
#include "flash_hc_gemv.metal"
#undef K
#undef BM
#undef BN
}
kernel void gdn_prepare_row(const device half* qkv [[buffer(0)]],const device half* history [[buffer(1)]],
    const device half* weight [[buffer(2)]],device half* conv [[buffer(3)]],device half* q [[buffer(4)]],
    device half* k [[buffer(5)]],device half* v [[buffer(6)]],device half* state_out [[buffer(7)]],
    constant uint* extent [[buffer(8)]],uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]]) {
#include "flash_gdn_prepare.metal"
}
kernel void gdn_step_row(const device half* q [[buffer(0)]],const device half* k [[buffer(1)]],
    const device half* v [[buffer(2)]],const device half* a [[buffer(3)]],const device half* b [[buffer(4)]],
    const device float* a_log [[buffer(5)]],const device half* dt_bias [[buffer(6)]],const device float* state_in [[buffer(7)]],
    device half* y [[buffer(8)]],device float* state_out [[buffer(9)]],constant uint* extent [[buffer(10)]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    uint3 thread_position_in_grid=uint3(thread_index_in_simdgroup, group.y*2+sg,group.z);
#define MLXL3_GDN_FUSED_GATES 1
#define MLXL3_GDN_SAVE_HISTORY 0
#define MLXL3_GDN_SAVE_TAPE 0
#define InT half
#define StT float
#define B 1
#define T 1
#define Dk 128
#define Dv 128
#define Hk 16
#define Hv 48
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
kernel void gdn_norm_row(const device half* delta [[buffer(0)]], const device float* gain [[buffer(1)]],
    device float* normed [[buffer(2)]],constant uint* params [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]], uint head [[threadgroup_position_in_grid]]) {
    const uint at=head*128+lane*4;
    float values[4],sum=0;
    for(uint i=0;i<4;++i) {values[i]=float(delta[at+i]);sum+=values[i]*values[i];}
    float inverse=precise::rsqrt(simd_sum(sum)/128.0f+as_type<float>(params[0]));
    for(uint i=0;i<4;++i) normed[at+i]=gain[lane*4+i]*(values[i]*inverse);
}
kernel void gdn_norm_gate_row(const device half* delta [[buffer(0)]], const device float* gain [[buffer(1)]],
    const device half* z [[buffer(2)]], device half* output [[buffer(3)]], constant uint* params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]], uint head [[threadgroup_position_in_grid]]) {
    const uint at=head*128+lane*4;
    float values[4],sum=0;
    for(uint i=0;i<4;++i) {values[i]=float(delta[at+i]);sum+=values[i]*values[i];}
    float inverse=precise::rsqrt(simd_sum(sum)/128.0f+as_type<float>(params[0]));
    for(uint i=0;i<4;++i) {
        // Preserve the F32 materialization between canonical norm and the T17 epilogue.
        volatile float normalized=gain[lane*4+i]*(values[i]*inverse);
        float gate=float(z[at+i]);
        float y=1.0f/(1.0f+metal::precise::exp(metal::abs(gate)));
        volatile float probability=gate<0.0f?y:1.0f-y;
        volatile float product=normalized*probability;
        output[at+i]=half(product);
    }
}
kernel void gdn_gate_row(const device float* normed0 [[buffer(0)]], const device half* z [[buffer(1)]],
    device half* output [[buffer(2)]],constant uint* extent [[buffer(3)]],uint3 thread_position_in_grid [[thread_position_in_grid]]) {
#define ROWS 1
#define WIDTH 6144
#include "flash_gdn_epilogue.metal"
#undef ROWS
#undef WIDTH
}
kernel void gdn_to_float(const device half* in [[buffer(0)]],device float* out [[buffer(1)]],
    constant uint* extent [[buffer(2)]],uint col [[thread_position_in_grid]]) {out[col]=float(in[col]);}

kernel void gdn_joint_ab(const device half* x [[buffer(0)]],const device half* a [[buffer(1)]],
    const device half* b [[buffer(2)]],device half* out_a [[buffer(3)]],device half* out_b [[buffer(4)]],
    constant uint* extent [[buffer(5)]],uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint3 position [[threadgroup_position_in_grid]]) {
    const device half* weight=position.x<12?a:b;
    device half* out=position.x<12?out_a:out_b;
    uint3 threadgroup_position_in_grid=uint3(position.x%12,0,0);
#define K 2560
#define BM 1
#define BN 8
#include "flash_hc_gemv.metal"
#undef K
#undef BM
#undef BN
}

kernel void ar10_gdn_whole(const device half* qkv [[buffer(0)]],const device half* z [[buffer(1)]],
    const device half* a [[buffer(2)]],const device half* b [[buffer(3)]],const device half* history [[buffer(4)]],
    const device half* weight [[buffer(5)]],const device float* a_log [[buffer(6)]],const device half* dt_bias [[buffer(7)]],
    const device float* state_in [[buffer(8)]],device half* history_out [[buffer(9)]],device float* state_out [[buffer(10)]],
    const device float* gain [[buffer(11)]],device half* gated [[buffer(12)]],constant uint* params [[buffer(13)]],
    uint lid [[thread_index_in_threadgroup]],uint h [[threadgroup_position_in_grid]]) {
    const uint sg=lid/32,ln=lid%32,hk=h/3;
    threadgroup half tq[128],tk[128],tv[128],ty[128];
    if(sg<3) {
        const uint channel_head=sg==0?hk:(sg==1?16+hk:32+h);
        half values[4],sum=0.0h;
        for(uint i=0;i<4;++i) {
            uint channel=channel_head*128+ln*4+i;
            float acc=0.0f;
            for(uint t=0;t<4;++t) {
                half x=t<3?history[t*10240+channel]:qkv[channel];
                acc+=float(x)*float(weight[channel*4+t]);
            }
            half x=half(acc);
            half tail=half(1.0h)/(half(1.0h)+metal::exp(metal::abs(x)));
            half sigmoid=x<half(0.0h)?tail:half(1.0h)-tail;
            half value=x*sigmoid;
            values[i]=value;
            volatile half squared=value*value;
            sum+=squared;
            // Shared Q/K heads have exactly one history owner; each V head is unique.
            if(sg==2 || h%3==0) for(uint t=0;t<3;++t)
                history_out[t*10240+channel]=t<2?history[(t+1)*10240+channel]:qkv[channel];
            if(sg==2)tv[ln*4+i]=value;
        }
        if(sg<2) {
            half total=simd_sum(sum);
            half inverse=metal::precise::rsqrt(half(total+half(1.0e-6f)));
            for(uint i=0;i<4;++i) {
                volatile half normalized=values[i]*inverse;
                if(sg==0)tq[ln*4+i]=normalized*half(0.08838834764831845f);
                else tk[ln*4+i]=normalized;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup half* q=tq;
    threadgroup half* k=tk;
    threadgroup half* v=tv;
    threadgroup half* y=ty;
    uint thread_index_in_simdgroup=ln;
    uint3 thread_position_in_grid=uint3(ln,sg,h);
#define MLXL3_GDN_LOCAL_QKV 1
#define MLXL3_GDN_FUSED_GATES 1
#define MLXL3_GDN_SAVE_HISTORY 0
#define MLXL3_GDN_SAVE_TAPE 0
#define InT half
#define StT float
#define B 1
#define T 1
#define Dk 128
#define Dv 128
#define Hk 16
#define Hv 48
#include "gated_delta_packed.metal"
#undef MLXL3_GDN_LOCAL_QKV
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
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        const uint at=h*128+ln*4;
        float values[4],sum=0;
        for(uint i=0;i<4;++i){values[i]=float(ty[ln*4+i]);sum+=values[i]*values[i];}
        float inverse=precise::rsqrt(simd_sum(sum)/128.0f+as_type<float>(params[0]));
        for(uint i=0;i<4;++i) {
            volatile float normalized=gain[ln*4+i]*(values[i]*inverse);
            float gate=float(z[at+i]);
            float p=1.0f/(1.0f+metal::precise::exp(metal::abs(gate)));
            volatile float probability=gate<0.0f?p:1.0f-p;
            volatile float product=normalized*probability;
            gated[at+i]=half(product);
        }
    }
}
