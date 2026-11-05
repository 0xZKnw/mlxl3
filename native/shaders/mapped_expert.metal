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
inline void mapped_glu_block(const device float* gate_raw, const device float* up_raw,
    const device half* gate_scale, const device half* up_scale, const device half* down_scale,
    device half* prepared, uint width, uint splits, uint tid, uint block,
    threadgroup float* ga, threadgroup float* gb, threadgroup float* ua,
    threadgroup float* ub, threadgroup float* da, threadgroup float* db) {
    uint column=block*128+tid;
    ga[tid]=float(half(mapped_partial_sum(gate_raw,column,width,splits)));
    ua[tid]=float(half(mapped_partial_sum(up_raw,column,width,splits)));
    mapped_float_h128(ga,gb,tid);
    mapped_float_h128(ua,ub,tid);
    float gate=float(half(ga[tid]*0.08838834764831845f*float(gate_scale[column])));
    float up=float(half(ua[tid]*0.08838834764831845f*float(up_scale[column])));
    // MLX 0.32.2 custom kernels use MathModeSafe with the default fast FP32 functions.
    // fastMathEnabled(false) also switches the native library to precise functions;
    // name this exponential explicitly to preserve the source checkpoint's rounding.
    float activated=float(half((gate/(1.0f+fast::exp(-gate)))*up));
    da[tid]=float(half(activated*float(down_scale[column])));
    mapped_float_h128(da,db,tid);
    prepared[column]=half(da[tid]*0.08838834764831845f);
}

kernel void mapped_glu(const device float* gate_raw [[buffer(0)]],const device float* up_raw [[buffer(1)]],
    const device half* gate_scale [[buffer(2)]],const device half* up_scale [[buffer(3)]],
    const device half* down_scale [[buffer(4)]],device half* prepared [[buffer(5)]],
    constant uint* p [[buffer(6)]],uint tid [[thread_index_in_threadgroup]],uint block [[threadgroup_position_in_grid]]) {
    threadgroup float ga[128],gb[128],ua[128],ub[128],da[128],db[128];
    mapped_glu_block(gate_raw, up_raw, gate_scale, up_scale, down_scale,
                     prepared, p[0], p[1], tid, block, ga, gb, ua, ub, da, db);
}

kernel void qwen_stage_glu(
    const device float* raw [[buffer(0)]],
    const device QwenStageScales* scales [[buffer(1)]],
    device half* prepared [[buffer(2)]],
    constant uint& splits [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float ga[128],gb[128],ua[128],ub[128],da[128],db[128];
    uint route = group.y;
    uint job_words = splits * 512u;
    const device QwenStageScales& weights = scales[route];
    mapped_glu_block(raw + (route * 2u) * job_words, raw + (route * 2u + 1u) * job_words,
                     weights.sv[0], weights.sv[1], weights.su[2], prepared + route * 512u,
                     512u, splits, tid, group.x, ga, gb, ua, ub, da, db);
}

kernel void qwen_stage_prepare(
    const device half* input [[buffer(0)]],
    const device QwenStageScales* scales [[buffer(1)]],
    device half* prepared [[buffer(2)]],
    constant long2& input_view [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float values[128];
    uint start = group.x * 128u;
    uint job = group.y;
    const device half* su = scales[job / 2u].su[job % 2u];
    for (uint i = lane; i < 128u; i += 64u) {
        uint col = start + i;
        values[i] = float(input[input_view.x + long(col) * input_view.y] * su[col]);
    }
    hadamard128(values, lane);
    for (uint i = lane; i < 128u; i += 64u)
        prepared[job * 2048u + start + i] = half(values[i] * as_type<float>(0x3db504f3u));
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
