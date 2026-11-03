// RMS boundaries follow MLX 0.32.2 rms_norm.metal, Copyright 2024 Apple Inc. (MIT).
template <typename W>
kernel void model_rms(const device half* x [[buffer(0)]], const device W* gain [[buffer(1)]],
    device half* out [[buffer(2)]],constant long* p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint sg [[simdgroup_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]]) {
    #pragma clang fp contract(off)
    threadgroup float sums[32],inverse;
    const uint width=uint(p[0]);
    float sum=0;
    for(ulong block=ulong(tid)*4;block<width;block+=ulong(threads)*4) {
        for(uint i=0;i<4 && block+i<width;++i) {
            float value=float(x[p[1]+long(row)*p[2]+long(block+i)*p[3]]);
            sum+=value*value;
        }
    }
    sum=simd_sum(sum);
    if(sg==0)sums[lane]=0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0)sums[sg]=sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg==0) {
        float total=simd_sum(sums[lane]);
        if(lane==0)inverse=precise::rsqrt(total/float(width)+as_type<float>(uint(p[6])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(ulong block=ulong(tid)*4;block<width;block+=ulong(threads)*4) {
        for(uint i=0;i<4 && block+i<width;++i) {
            uint col=uint(block+i);
            W normalized=W(float(x[p[1]+long(row)*p[2]+long(col)*p[3]])*inverse);
            out[ulong(row)*width+col]=half(gain[p[4]+long(col)*p[5]]*normalized);
        }
    }
}

// Same four-output/four-read row reduction used by the existing exact dense row path.
template <typename W>
kernel void model_dense(const device half* x [[buffer(0)]],const device W* w [[buffer(1)]],
    device half* out [[buffer(2)]],constant long* p [[buffer(3)]],
    uint2 group [[threadgroup_position_in_grid]],uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    #pragma clang fp contract(off)
    const uint input=uint(p[0]),output=uint(p[1]),first=group.x*16+sg*4;
    float acc[4]={};
    for(ulong block=ulong(lane)*4;block<input;block+=128) {
        for(uint j=0;j<4 && block+j<input;++j) {
            float value=float(x[p[2]+long(group.y)*p[3]+long(block+j)*p[4]]);
            for(uint i=0;i<4;++i)if(first+i<output)
                acc[i]+=value*float(w[p[5]+long(first+i)*p[6]+long(block+j)*p[7]]);
        }
    }
    for(ushort shift=16;shift>0;shift>>=1)for(uint i=0;i<4;++i)acc[i]+=simd_shuffle_down(acc[i],shift);
    if(lane==0)for(uint i=0;i<4;++i)if(first+i<output)out[ulong(group.y)*output+first+i]=half(acc[i]);
}
#define MODEL_RMS_INSTANTIATE(T, NAME) \
template [[host_name(NAME)]] kernel void model_rms<T>(const device half*,const device T*,device half*,constant long*,uint,uint,uint,uint,uint);
MODEL_RMS_INSTANTIATE(half,"model_rms_half")
MODEL_RMS_INSTANTIATE(float,"model_rms_float")
#undef MODEL_RMS_INSTANTIATE
#define MODEL_DENSE_INSTANTIATE(T, NAME) \
template [[host_name(NAME)]] kernel void model_dense<T>(const device half*,const device T*,device half*,constant long*,uint2,uint,uint);
MODEL_DENSE_INSTANTIATE(half,"model_dense_half")
MODEL_DENSE_INSTANTIATE(float,"model_dense_float")
#undef MODEL_DENSE_INSTANTIATE

// Single-query schedule from MLX 0.32.2 SDPA, also used by the qualified Flash QSA.
template <uint D>
kernel void model_gqa(const device half* queries [[buffer(0)]],const device half* keys [[buffer(1)]],
    const device half* values [[buffer(2)]],device half* out [[buffer(3)]],constant long* p [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],uint stripe [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float outputs[1024],maxima[32],sums[32];
    constexpr uint N=D/32;
    float q[N],acc[N]={};
    const uint kv_head=head/uint(p[1]/p[2]);
    for(uint j=0;j<N;++j)q[j]=float(queries[p[3]+long(head*D+lane*N+j)*p[4]])*as_type<float>(uint(p[11]));
    float top=-FLT_MAX,sum=0;
    for(ulong token=stripe;token<ulong(p[0]);token+=32) {
        const long kb=p[5]+long(token)*p[6]+long(kv_head*D+lane*N)*p[7];
        const long vb=p[8]+long(token)*p[9]+long(kv_head*D+lane*N)*p[10];
        float score=0;
        for(uint j=0;j<N;++j)score+=q[j]*float(keys[kb+long(j)*p[7]]);
        score=simd_sum(score);
        float next=max(top,score),factor=fast::exp(top-next),weight=fast::exp(score-next);
        top=next;sum=sum*factor+weight;
        for(uint j=0;j<N;++j)acc[j]=acc[j]*factor+weight*float(values[vb+long(j)*p[10]]);
    }
    if(lane==0){maxima[stripe]=top;sums[stripe]=sum;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m=maxima[lane],factor=fast::exp(m-simd_max(m));
    float denominator=simd_sum(sums[lane]*factor);
    for(uint j=0;j<N;++j){
        outputs[lane*32+stripe]=acc[j];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc[j]=simd_sum(outputs[stripe*32+lane]*factor)/denominator;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(lane==0)for(uint j=0;j<N;++j)out[head*D+stripe*N+j]=half(acc[j]);
}
#define MODEL_GQA_INSTANTIATE(D) \
template [[host_name("model_gqa_" #D)]] kernel void model_gqa<D>(const device half*,const device half*,const device half*,device half*,constant long*,uint,uint,uint);
MODEL_GQA_INSTANTIATE(64)
MODEL_GQA_INSTANTIATE(128)
MODEL_GQA_INSTANTIATE(256)
#undef MODEL_GQA_INSTANTIATE

kernel void model_rope(const device half* x [[buffer(0)]],device half* out [[buffer(1)]],
    constant long* p [[buffer(2)]],uint index [[thread_position_in_grid]]) {
    const uint width=uint(p[1]),rotary=uint(p[2]),slots=width-rotary/2;
    if(ulong(index)>=ulong(p[0])*slots)return;
    const uint head=index/slots,slot=index%slots;
    const long base=p[4]+long(head)*p[5];
    if(slot>=rotary/2) {
        const uint col=slot+rotary/2;
        out[head*width+col]=x[base+long(col)*p[6]];
        return;
    }
    float angle=float(p[3])*metal::exp2(-float(slot)/float(rotary/2)*as_type<float>(uint(p[7])));
    float c=metal::fast::cos(angle),s=metal::fast::sin(angle);
    float a=float(x[base+long(slot)*p[6]]),b=float(x[base+long(slot+rotary/2)*p[6]]);
    out[head*width+slot]=half(a*c-b*s);
    out[head*width+slot+rotary/2]=half(b*c+a*s);
}
