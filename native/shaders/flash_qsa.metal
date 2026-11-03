// The checkpoint stores contiguous [query, gate] pairs per Q head.
kernel void qsa_norm(const device half* x [[buffer(0)]], const device float* weight [[buffer(1)]],
    device half* out [[buffer(2)]], constant uint* params [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]], uint head [[threadgroup_position_in_grid]]) {
    #pragma clang fp contract(off)
    float v[8], sum=0;
    for(uint j=0;j<8;++j) {
        uint col=(j/4)*128+lane*4+j%4;
        v[j]=float(x[head*params[0]+col]);
        sum+=v[j]*v[j];
    }
    float inverse=rsqrt(simd_sum(sum)/256.0f+as_type<float>(params[1]));
    for(uint j=0;j<8;++j) {
        uint col=(j/4)*128+lane*4+j%4;
        out[head*256+col]=half((v[j]*inverse)*(1.0f+weight[col]));
    }
}

kernel void qsa_freq(device float* out [[buffer(0)]], constant float& theta [[buffer(1)]],
    uint i [[thread_position_in_grid]]) { out[i]=1.0f/pow(theta,float(i)/32.0f); }

kernel void qsa_rope(const device half* x [[buffer(0)]], const device float* inv_freq [[buffer(1)]],
    device half* out [[buffer(2)]], constant uint* position [[buffer(3)]],
    uint3 thread_position_in_grid [[thread_position_in_grid]]) {
#define ROTARY 64
#define WIDTH 256
#define HEADS position[1]
#include "flash_rope.metal"
#undef HEADS
#undef WIDTH
#undef ROTARY
}

// Single-query FP16 SDPA, specialized from MLX 0.32.2, Copyright © 2024 Apple Inc. (MIT).
kernel void qsa_attention(const device half* queries [[buffer(0)]], const device half* keys [[buffer(1)]],
    const device half* values [[buffer(2)]], device half* out [[buffer(3)]],
    constant uint& length [[buffer(4)]], uint head [[threadgroup_position_in_grid]],
    uint stripe [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float outputs[1024], maxima[32], sums[32];
    float q[8], o[8]={};
    for(uint j=0;j<8;++j) q[j]=float(queries[head*256+lane*8+j])/16.0f;
    float top=-FLT_MAX, sum=0;
    for(uint t=stripe;t<length;t+=32) {
        uint base=t*512+(head/12)*256+lane*8;
        float score=0;
        for(uint j=0;j<8;++j) score+=q[j]*float(keys[base+j]);
        score=simd_sum(score);
        float next=max(top,score), factor=fast::exp(top-next), weight=fast::exp(score-next);
        top=next;
        sum=sum*factor+weight;
        for(uint j=0;j<8;++j) o[j]=o[j]*factor+weight*float(values[base+j]);
    }
    if(lane==0) { maxima[stripe]=top; sums[stripe]=sum; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m=maxima[lane];
    float factor=fast::exp(m-simd_max(m));
    float denominator=simd_sum(sums[lane]*factor);
    for(uint j=0;j<8;++j) {
        outputs[lane*32+stripe]=o[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        o[j]=simd_sum(outputs[stripe*32+lane]*factor)/denominator;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0) for(uint j=0;j<8;++j) out[head*256+stripe*8+j]=half(o[j]);
}

kernel void qsa_gate(const device half* attention [[buffer(0)]], const device half* qg [[buffer(1)]],
    device half* gated [[buffer(2)]],constant uint& width [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    gated[i]=attention[i]*hc_sigmoid(qg[(i/width*2+1)*width+i%width]);
}
