// Fused routed SwiGLU keeps the maintained float butterfly and activation boundaries.
inline float mapped_partial_sum(const device float* source, uint column, uint width, uint splits) {
    uint stripes=min(splits,8u);
    float sum=0;
    for(uint stripe=0;stripe<stripes;++stripe) {
        float partial=0;
        for(uint split=stripe;split<splits;split+=stripes) partial=source[split*width+column]+partial;
        sum=stripe==0 ? partial : partial+sum;
    }
    return sum;
}
inline void mapped_float_h128(threadgroup float* a,threadgroup float* b,uint tid) {
    for(uint stride=1;stride<=64;stride<<=1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float own=a[tid],peer=a[tid^stride];
        b[tid]=(tid&stride)?peer-own:own+peer;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        a[tid]=b[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
kernel void mapped_glu(const device float* gate_raw [[buffer(0)]],const device float* up_raw [[buffer(1)]],
    const device half* gate_scale [[buffer(2)]],const device half* up_scale [[buffer(3)]],
    const device half* down_scale [[buffer(4)]],device half* prepared [[buffer(5)]],
    constant uint* p [[buffer(6)]],uint tid [[thread_index_in_threadgroup]],uint block [[threadgroup_position_in_grid]]) {
    threadgroup float ga[128],gb[128],ua[128],ub[128],da[128],db[128];
    uint column=block*128+tid;
    ga[tid]=float(half(mapped_partial_sum(gate_raw,column,p[0],p[1])));
    ua[tid]=float(half(mapped_partial_sum(up_raw,column,p[0],p[1])));
    mapped_float_h128(ga,gb,tid);
    mapped_float_h128(ua,ub,tid);
    float gate=float(half(ga[tid]*0.08838834764831845f*float(gate_scale[column])));
    float up=float(half(ua[tid]*0.08838834764831845f*float(up_scale[column])));
    float activated=float(half((gate/(1.0f+exp(-gate)))*up));
    da[tid]=float(half(activated*float(down_scale[column])));
    mapped_float_h128(da,db,tid);
    prepared[column]=half(da[tid]*0.08838834764831845f);
}
kernel void mapped_mix(const device half* expert [[buffer(0)]],const device half* scores [[buffer(1)]],
    device half* out [[buffer(2)]],constant uint* p [[buffer(3)]],uint column [[thread_position_in_grid]]) {
    if(column>=p[0])return;
    uint stripes=min(p[1],8u);
    half sum=0;
    for(uint stripe=0;stripe<stripes;++stripe) {
        half partial=0;
        for(uint route=stripe;route<p[1];route+=stripes) {
            volatile half weighted=expert[route*p[0]+column]*scores[route];
            partial=weighted+partial;
        }
        sum=partial+sum;
    }
    out[column]=sum;
}
