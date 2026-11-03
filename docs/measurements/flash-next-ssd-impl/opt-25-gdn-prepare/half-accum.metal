// Flash uses half L2 normalization, unlike the RMS preparation of other Qwen models.
uint lane = thread_index_in_simdgroup;
uint head = threadgroup_position_in_grid.y;
half values[4];
half sum = 0.0h;
for (uint i = 0; i < 4; ++i) {
    uint channel = head * 128 + lane * 4 + i;
    float acc = 0.0f;
    for (uint t = 0; t < 4; ++t) {
        half x = t < 3 ? history[t * 10240 + channel] : qkv[channel];
        acc += float(x) * float(weight[channel * 4 + t]);
    }
    half x = half(acc);
    half tail = half(1.0h) / (half(1.0h) + metal::exp(metal::abs(x)));
    half sigmoid = x < half(0.0h) ? tail : half(1.0h) - tail;
    half value = x * sigmoid;
    values[i] = value;
    conv[channel] = value;
    sum += value * value;
    for (uint t = 0; t < 3; ++t) {
        state_out[t * 10240 + channel] = t < 2 ? history[(t + 1) * 10240 + channel] : qkv[channel];
    }
    if (head >= 32) v[channel - 4096] = value;
}
if (head < 32) {
    half total = simd_sum(sum);
    half inverse = metal::precise::rsqrt(half(total + half(1.0e-6f)));
    for (uint i = 0; i < 4; ++i) {
        uint offset = (head % 16) * 128 + lane * 4 + i;
        half normalized = values[i] * inverse;
        if (head < 16) q[offset] = normalized * half(0.08838834764831845f);
        else k[offset] = normalized;
    }
}
