// The scalar switch uses the reference dot-product reduction (one 512-thread block).
template <typename T>
kernel void moe_switch(const device T* x [[buffer(0)]], const device half* weight [[buffer(1)]],
    device T* out [[buffer(2)]], constant long* layout [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float parts[16];
    constexpr uint width=16/sizeof(T);
    float4 acc=0;
    for (uint tile=0;tile<32;tile+=width) {
        uint start=sg*1024+lane*width+tile*32;
        if(start>=2560) continue;
        for(uint j=0;j<width;++j) {
            uint col=start+j;
            acc[j%4]=fma(float(x[layout[0]+long(col)*layout[1]]),float(weight[col]),acc[j%4]);
        }
    }
    float part=simd_sum(acc.x+acc.y+acc.z+acc.w);
    if (lane==0) parts[sg]=part;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg==0) {
        float result=simd_sum(lane<16 ? parts[lane] : 0.0f);
        if (tid!=0) return;
        T g=T(result), tail=T(1)/(T(1)+exp(abs(g)));
        out[0]=g<T(0)?tail:T(1)-tail;
        if (layout[2]) out[1]=g;
    }
}

#define EXPERT_ARG(N) const device half* e##N [[buffer(N)]]
#define EXPERT_ARGS EXPERT_ARG(0), EXPERT_ARG(1), EXPERT_ARG(2), EXPERT_ARG(3), EXPERT_ARG(4), EXPERT_ARG(5), EXPERT_ARG(6), EXPERT_ARG(7), EXPERT_ARG(8), EXPERT_ARG(9)
template <typename T>
kernel void moe_mix(EXPERT_ARGS,
    const device half* shared [[buffer(10)]], const device T* scores [[buffer(11)]],
    const device T* multiplier [[buffer(12)]], device T* out [[buffer(13)]],
    constant uint* extent [[buffer(14)]], uint col [[thread_position_in_grid]]) {
    #pragma clang fp contract(off)
    if(col>=extent[0]) return;
    const device half* experts[10]={e0,e1,e2,e3,e4,e5,e6,e7,e8,e9};
    T sum=0;
    // The stock reduction of [1,10,2560] partitions the route axis into eight stripes.
    for(uint i=0; i<8; ++i) {
        T partial=T(experts[i][col])*scores[i];
        if (i<2) {
            T later=T(experts[i+8][col])*scores[i+8];
            partial=later+partial;
        }
        sum=partial+sum;
    }
    T branch=T(shared[col])*multiplier[0];
    out[col]=sum+branch;
}

#define MOE_KERNELS(T,S,M) \
template [[host_name(S)]] kernel void moe_switch<T>(const device T*,const device half*,device T*,constant long*,uint,uint,uint); \
template [[host_name(M)]] kernel void moe_mix<T>(EXPERT_ARGS,const device half*,const device T*,const device T*,device T*,constant uint*,uint);
MOE_KERNELS(half,"moe_switch_half","moe_mix_half")
MOE_KERNELS(float,"moe_switch_float","moe_mix_float")
#undef MOE_KERNELS
#undef EXPERT_ARGS
#undef EXPERT_ARG
