// Match MLX's row_reduce_looped kernel for a single 10,240-wide FP32 row.
kernel void mtp_hidden_norm(const device float* input [[buffer(0)]], const device float* gain [[buffer(1)]],
    device float* output [[buffer(2)]], constant float& eps [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    threadgroup float partial[32];
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
    for (uint col = tid; col < 10240; col += 1024)
        output[col] = (input[col] * partial[0]) * (1.0f + gain[col]);
}

kernel void mtp_combine_streams(const device half* embedding [[buffer(0)]], const device half* hidden [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& width [[buffer(3)]], uint col [[thread_position_in_grid]]) {
    output[col] = embedding[col % width] + hidden[col];
}
