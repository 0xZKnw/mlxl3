// The checkpoint stores contiguous [query, gate] pairs per Q head.
kernel void qsa_norm(const device half* x [[buffer(0)]], const device float* weight [[buffer(1)]],
    device half* out [[buffer(2)]], constant uint* params [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]], uint head [[threadgroup_position_in_grid]]) {
    #pragma clang fp contract(off)
    const uint width=params[2];
    float v[8], sum=0;
    for(uint j=0;j<width/32;++j) {
        uint col=(j/4)*128+lane*4+j%4;
        v[j]=float(x[head*params[0]+col]);
        sum+=v[j]*v[j];
    }
    float inverse=rsqrt(simd_sum(sum)/float(width)+as_type<float>(params[1]));
    for(uint j=0;j<width/32;++j) {
        uint col=(j/4)*128+lane*4+j%4;
        out[head*width+col]=half((v[j]*inverse)*(1.0f+weight[col]));
    }
}

kernel void qsa_freq(device float* out [[buffer(0)]], constant float* p [[buffer(1)]],
    uint i [[thread_position_in_grid]]) { out[i]=1.0f/pow(p[0],float(i)/p[1]); }

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

kernel void qsa_index_rope(const device half* x [[buffer(0)]], const device float* inv_freq [[buffer(1)]],
    device half* out [[buffer(2)]], constant uint* position [[buffer(3)]], uint3 thread_position_in_grid [[thread_position_in_grid]]) {
#define ROTARY 32
#define WIDTH 128
#define HEADS position[1]
#include "flash_rope.metal"
#undef HEADS
#undef WIDTH
#undef ROTARY
}

kernel void qsa_pool(const device half* tail [[buffer(0)]], const device half* qk [[buffer(1)]],
    device half* out [[buffer(2)]],constant uint& width [[buffer(3)]],uint i [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    float sum=float(tail[i])+float(tail[width+i])+float(tail[2*width+i])+float(qk[4*width+i]);
    out[i]=half(sum*0.25f);
}

// The stock FP32 matmul collapses the four index heads into M and uses tiled GEMM.
kernel void qsa_index_scores(const device half* q [[buffer(0)]],const device half* blocks [[buffer(1)]],
    device float* out [[buffer(2)]], constant uint& count [[buffer(3)]],uint group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float a[64],b[64],c[64];
    simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8,8>(0.0f);
    for(uint k=0;k<128;k+=8){
        for(uint i=lane;i<64;i+=32){
            uint row=i/8,col=i%8;
            a[i]=(row<4)?float(q[row*128+k+col]):0.0f;
            uint id=group*8+col;
            b[i]=(id<count)?float(blocks[id*128+k+row]):0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 ma,mb;
        simdgroup_load(ma,a,8);simdgroup_load(mb,b,8);
        simdgroup_multiply_accumulate(acc,ma,mb,acc);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    simdgroup_store(acc,c,8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    uint id=group*8+lane;
    if(lane<8 && id<count){
        float sum=0;
        for(uint h=0;h<4;++h) sum+=max(c[h*8+lane],0.0f);
        out[id]=sum*0.08838834764831844f;
    }
}
// Single-query FP16 SDPA, specialized from MLX 0.32.2, Copyright © 2024 Apple Inc. (MIT).
kernel void qsa_attention(const device half* queries [[buffer(0)]], const device half* keys [[buffer(1)]],
    const device half* values [[buffer(2)]], device half* out [[buffer(3)]],
    const device uint* positions [[buffer(4)]], constant uint* params [[buffer(5)]], uint head [[threadgroup_position_in_grid]],
    uint stripe [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float outputs[1024], maxima[32], sums[32];
    float q[8], o[8]={};
    for(uint j=0;j<8;++j) q[j]=float(queries[head*256+lane*8+j])/16.0f;
    float top=-FLT_MAX, sum=0;
    for(uint t=stripe;t<params[0];t+=32) {
        uint base=(params[1]?positions[t]:t)*512+(head/12)*256+lane*8;
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

// Two-pass Flash decode SDPA for Max and Ultra, following Apple MLX 0.32.2 (MIT).
kernel void qsa_attention_partials(const device half* q [[buffer(0)]], const device half* keys [[buffer(1)]],
    const device half* values [[buffer(2)]],const device uint* ids [[buffer(3)]], device half* partial [[buffer(4)]],
    device float* sums [[buffer(5)]],device float* maxima [[buffer(6)]],constant uint* p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    const uint head=group.x, block=group.y, blocks=p[2];
    float query[8],acc[8]={};
    for(uint j=0;j<8;++j) query[j]=float(q[head*256+lane*8+j])/16.0f;
    float top=-FLT_MAX,sum=0;
    for(uint t=block;t<p[0];t+=blocks) {
        uint base=(p[1]?ids[t]:t)*512+(head/12)*256+lane*8;
        float score=0;
        for(uint j=0;j<8;++j) score+=query[j]*float(keys[base+j]);
        score=simd_sum(score);
        float next=max(top,score),factor=fast::exp(top-next),w=fast::exp(score-next);
        sum=sum*factor+w;top=next;
        for(uint j=0;j<8;++j) acc[j]=acc[j]*factor+w*float(values[base+j]);
    }
    uint slot=head*blocks+block;
    if(lane==0){sums[slot]=sum;maxima[slot]=top;}
    for(uint j=0;j<8;++j)partial[slot*256+lane*8+j]=half(acc[j]);
}

kernel void qsa_attention_finish(const device half* partial [[buffer(0)]],const device float* sums [[buffer(1)]],
    const device float* maxima [[buffer(2)]],device half* out [[buffer(3)]],constant uint& blocks [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],uint stripe [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float across[1024];
    sums+=head*blocks;maxima+=head*blocks;partial+=head*blocks*256;
    float top=-FLT_MAX,sum=0,acc[8]={};
    for(uint b=lane;b<blocks;b+=32)top=max(top,maxima[b]);
    top=simd_max(top);
    for(uint b=lane;b<blocks;b+=32)sum+=fast::exp(maxima[b]-top)*sums[b];
    sum=simd_sum(sum);
    for(uint b=stripe;b<blocks;b+=32){
        float factor=fast::exp(maxima[b]-top);
        for(uint j=0;j<8;++j)acc[j]+=factor*float(partial[b*256+lane*8+j]);
    }
    for(uint j=0;j<8;++j){
        across[lane*32+stripe]=acc[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc[j]=simd_sum(across[stripe*32+lane])/sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0)for(uint j=0;j<8;++j)out[head*256+stripe*8+j]=half(acc[j]);
}

inline void ar10_qsa_norm256(const device half* x,const device float* weight,
    threadgroup half* normed,float eps,uint lane) {
    #pragma clang fp contract(off)
    float v[8],sum=0;
    for(uint j=0;j<8;++j) {
        uint col=(j/4)*128+lane*4+j%4;
        v[j]=float(x[col]);sum+=v[j]*v[j];
    }
    float inverse=rsqrt(simd_sum(sum)/256.0f+eps);
    for(uint j=0;j<8;++j) {
        uint col=(j/4)*128+lane*4+j%4;
        normed[col]=half((v[j]*inverse)*(1.0f+weight[col]));
    }
}
kernel void ar10_qsa_prepare(const device half* qg [[buffer(0)]],const device half* k [[buffer(1)]],
    const device float* q_gain [[buffer(2)]],const device float* k_gain [[buffer(3)]],
    const device float* inv_freq [[buffer(4)]],device half* q_out [[buffer(5)]],device half* k_out [[buffer(6)]],
    constant uint* p [[buffer(7)]],uint lane [[thread_index_in_simdgroup]],uint group [[threadgroup_position_in_grid]]) {
    threadgroup half normed[256];
    const bool query=group<24;
    const uint head=query?group:group-24;
    const device half* x=query?qg+head*512:k+head*256;
    device half* out=query?q_out+head*256:k_out+head*256;
    ar10_qsa_norm256(x,query?q_gain:k_gain,normed,as_type<float>(p[1]),lane);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint slot=lane;slot<224;slot+=32) {
        if(slot>=32) {uint pass=64+slot-32;out[pass]=normed[pass];continue;}
        float angle=float(p[0])*float(inv_freq[slot]);
        float c=metal::cos(angle),s=metal::sin(angle);
        float xv=float(normed[slot]),xp=float(normed[slot+32]);
        out[slot]=half(xv*c-xp*s);
        out[slot+32]=half(xp*c+xv*s);
    }
}
