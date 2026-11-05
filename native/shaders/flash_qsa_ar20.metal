#define AR20_CHAIN_ARGS const device half* queries [[buffer(0)]],const device half* keys [[buffer(1)]], \
    const device half* values [[buffer(2)]],device float* maxima [[buffer(3)]],device float* sums [[buffer(4)]], \
    device float* outputs [[buffer(5)]],constant uint* params [[buffer(6)]],uint3 group [[threadgroup_position_in_grid]], \
    uint stripe [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]

kernel void ar20_qsa_chains(AR20_CHAIN_ARGS) {
    const uint f=group.x*4+stripe,chain=f%32,first_head=group.y*12+(f/32)*2;
#define AR20_HPT 2
#define AR20_PREFETCH 0
#include "flash_qsa_chain.metal"
#undef AR20_PREFETCH
#undef AR20_HPT
    for(uint h=0;h<2;++h) {
        uint head=first_head+h;
        if(lane==0) {maxima[head*32+chain]=top[h];sums[head*32+chain]=sum[h];}
        for(uint j=0;j<8;++j) outputs[(head*256+lane*8+j)*32+chain]=o[h][j];
    }
}
kernel void ar20_qsa_chains_prefetch(AR20_CHAIN_ARGS) {
    const uint f=group.x*4+stripe,chain=f%32,first_head=group.y*12+(f/32)*2;
#define AR20_HPT 2
#define AR20_PREFETCH 1
#include "flash_qsa_chain.metal"
#undef AR20_PREFETCH
#undef AR20_HPT
    for(uint h=0;h<2;++h) {
        uint head=first_head+h;
        if(lane==0) {maxima[head*32+chain]=top[h];sums[head*32+chain]=sum[h];}
        for(uint j=0;j<8;++j) outputs[(head*256+lane*8+j)*32+chain]=o[h][j];
    }
}
#undef AR20_CHAIN_ARGS

kernel void ar20_qsa_chain_combine(const device float* maxima [[buffer(0)]],const device float* sums [[buffer(1)]],
    const device float* outputs [[buffer(2)]],device half* out [[buffer(3)]],const device half* qg [[buffer(4)]],
    constant uint* params [[buffer(5)]],uint head [[threadgroup_position_in_grid]],
    uint stripe [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    float m=maxima[head*32+lane],factor=fast::exp(m-simd_max(m));
    float denominator=simd_sum(sums[head*32+lane]*factor);
    float result[8];
    for(uint j=0;j<8;++j) result[j]=simd_sum(outputs[(head*256+stripe*8+j)*32+lane]*factor)/denominator;
    if(lane==0) for(uint j=0;j<8;++j) {
        volatile half attention=half(result[j]);
        out[head*256+stripe*8+j]=params[0]?attention*hc_sigmoid(qg[(head*2+1)*256+stripe*8+j]):attention;
    }
}

kernel void ar20_qsa_prefetch(const device half* queries [[buffer(0)]],const device half* keys [[buffer(1)]],
    const device half* values [[buffer(2)]],device half* out [[buffer(3)]],const device half* qg [[buffer(4)]],
    constant uint* params [[buffer(5)]],uint head [[threadgroup_position_in_grid]],
    uint stripe [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    const uint chain=stripe,first_head=head;
#define AR20_HPT 1
#define AR20_PREFETCH 1
#include "flash_qsa_chain.metal"
#undef AR20_PREFETCH
#undef AR20_HPT
    threadgroup float outputs[1024],maxima[32],sums[32];
    if(lane==0) {maxima[stripe]=top[0];sums[stripe]=sum[0];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m=maxima[lane],factor=fast::exp(m-simd_max(m));
    float denominator=simd_sum(sums[lane]*factor);
    for(uint j=0;j<8;++j) {
        outputs[lane*32+stripe]=o[0][j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[0][j]=simd_sum(outputs[stripe*32+lane]*factor)/denominator;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0) for(uint j=0;j<8;++j) {
        volatile half attention=half(o[0][j]);
        out[head*256+stripe*8+j]=params[2]?attention*hc_sigmoid(qg[(head*2+1)*256+stripe*8+j]):attention;
    }
}
