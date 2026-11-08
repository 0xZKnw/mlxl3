// Shared verbatim reduction; all 1024 threads must call before consuming the factor.
inline float mtp_hidden_factor(const device float* input, float eps,
    threadgroup float* partial, uint lane, uint sg, uint tid) {
    #pragma clang fp contract(off)
    float sum = 0;
    for (uint base = tid*4; base < 10240; base += 1024*4) {
        for (uint j=0; j<4; ++j) {
            float value = input[base+j];
            sum += value * value;
        }
    }
    sum = simd_sum(sum);
    if (!lane) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!sg) {
        float total = simd_sum(partial[lane]);
        if (!lane) partial[0] = rsqrt(total * (1.0f/10240.0f) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial[0];
}

// Match MLX's row_reduce_looped kernel for a single 10,240-wide FP32 row.
kernel void mtp_hidden_norm(const device float* input [[buffer(0)]], const device float* gain [[buffer(1)]],
    device float* output [[buffer(2)]], constant float& eps [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    threadgroup float partial[32];
    float factor = mtp_hidden_factor(input, eps, partial, lane, sg, tid);
    for (uint col = tid; col < 10240; col += 1024)
        output[col] = (input[col] * factor) * (1.0f + gain[col]);
}

kernel void mtp_combine_streams(const device half* embedding [[buffer(0)]], const device half* hidden [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& width [[buffer(3)]], uint col [[thread_position_in_grid]]) {
    output[col] = embedding[col % width] + hidden[col];
}

kernel void mtp_hidden_factor_only(const device float* input [[buffer(0)]],
    device float* output [[buffer(1)]], constant float& eps [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float partial[32];
    float factor = mtp_hidden_factor(input, eps, partial, lane, sg, tid);
    if (!tid) output[0] = factor;
}

kernel void mtp_hidden_norm_prepare(const device float* input [[buffer(0)]],
    const device float* gain [[buffer(1)]], const device float* factor [[buffer(2)]],
    const device half* suh [[buffer(3)]], device half* output [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    #pragma clang fp contract(off)
    threadgroup float values[128];
    uint base = block * 128u;
    for (uint i=lane; i<128u; i+=64u) {
        uint col = base+i;
        half normalized = half((input[col]*factor[0])*(1.0f+gain[col]));
        values[i] = float(half(normalized*suh[col%2560u]));
    }
    hadamard128(values,lane);
    for (uint i=lane; i<128u; i+=64u)
        output[base+i] = half(values[i]*as_type<float>(0x3db504f3u));
}
