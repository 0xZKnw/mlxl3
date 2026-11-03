// Stable MoE top-k. Stock and direct kernels supply their own row bindings.
template <typename T>
inline void mlxl3_rank_row(T raw, uint tid, uint experts, uint k, bool normalize,
        threadgroup ulong* keys, threadgroup float* selected,
        device uint* indices, device T* scores) {
    float value = float(raw);
    uint bits = value == 0.0f ? 0u : as_type<uint>(value);
    uint monotonic = isnan(value) ? 0xffffffffu
        : bits ^ uint((int(bits) >> 31) | int(0x80000000u));
    ulong key = (ulong(monotonic) << 32) | ulong(tid);
    keys[tid] = key;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint above = 0;
    for (uint i = 0; i < experts; ++i) above += keys[i] > key;
    if (above < k) {
        uint slot = k - 1 - above;
        indices[slot] = tid;
        selected[slot] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        T sum = T(0);
        if (normalize) {
            for (uint i = 0; i < k; ++i) sum = T(selected[i]) + sum;
        }
        for (uint i = 0; i < k; ++i) {
            T score = T(selected[i]);
            scores[i] = normalize ? score / sum : score;
        }
    }
}
