// Whole-row reduction for MTP's FP32 four-stream RMS, using the reference's 128 segments of 80.
kernel void mtp_hidden_norm(const device float* input [[buffer(0)]], const device float* gain [[buffer(1)]],
    device float* output [[buffer(2)]], constant float& eps [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]],
    uint tid [[thread_index_in_threadgroup]]) {
    #pragma clang fp contract(off)
    threadgroup float partial[128];
    for (uint row = sg; row < 128; row += 32) {
        float acc = 0;
        if (lane < 20) {
            for (uint j = 0; j < 4; ++j) {
                float value = input[row*80 + lane*4 + j];
                acc += value * value;
            }
        }
        acc = simd_sum(acc);
        if (!lane) partial[row] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!sg) {
        float acc = 0;
        for (uint j = 0; j < 4; ++j) acc += partial[lane*4+j];
        acc = simd_sum(acc);
        if (!lane) partial[0] = rsqrt(acc * (1.0f/10240.0f) + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint col = tid; col < 10240; col += 1024)
        output[col] = (input[col] * partial[0]) * (1.0f + gain[col]);
}

kernel void mtp_combine_streams(const device half* embedding [[buffer(0)]], const device half* hidden [[buffer(1)]],
    device half* output [[buffer(2)]], constant uint& width [[buffer(3)]], uint col [[thread_position_in_grid]]) {
    output[col] = embedding[col % width] + hidden[col];
}
