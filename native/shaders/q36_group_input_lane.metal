constexpr uint INPUT_DIMS = 2048u;
uint tid = thread_position_in_threadgroup.x;
uint block = threadgroup_position_in_grid.x;
uint token = threadgroup_position_in_grid.y;
uint projection = threadgroup_position_in_grid.z;
uint base = block * 128u + tid * 4u;
float v[4];
for (uint r = 0u; r < 4u; ++r) {
    uint column = base + r;
    v[r] = float(half(half(x[token * INPUT_DIMS + column]) *
        half(scales[projection * INPUT_DIMS + column])));
}
for (uint h = 1u; h < 4u; h *= 2u) {
    for (uint i = 0u; i < 2u; ++i) {
        uint k = i & (h - 1u);
        uint j = ((i - k) << 1u) + k;
        float a = v[j], b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
    }
}
for (uint h = 1u; h < 4u; h *= 2u) {
    for (uint r = 0u; r < 4u; ++r) {
        float peer = simd_shuffle_xor(v[r], h);
        v[r] = (tid & h) ? peer - v[r] : v[r] + peer;
    }
}
// MLX H128 rounds the intermediate after the first four butterfly stages.
for (uint r = 0u; r < 4u; ++r) v[r] = float(half(v[r]));
for (uint h = 4u; h < 32u; h *= 2u) {
    for (uint r = 0u; r < 4u; ++r) {
        float peer = simd_shuffle_xor(v[r], h);
        v[r] = (tid & h) ? peer - v[r] : v[r] + peer;
    }
}
for (uint r = 0u; r < 4u; ++r) {
    uint column = base + r;
    out[((token * 2u + projection) * INPUT_DIMS) + ((column & ~15u) | (column & 1u) | ((column & 6u) << 1u) | ((column & 8u) >> 2u))] =
        half(float(half(v[r])) * 0.08838834764831845f);
}
