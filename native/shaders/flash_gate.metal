// Flash 2560 -> 512. Reference arithmetic and selection are documented in opti.md.
template <typename T>
kernel void gate_dot(const device T* x [[buffer(0)]], const device half* weights [[buffer(1)]],
    device T* out [[buffer(2)]], constant long* layout [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    long start = layout[0] + long(group.y) * layout[1];
    float accum = 0;
    for (uint block = 0; block < 2560; block += 128) {
        for (uint j = 0; j < 4; ++j) {
            uint col = block + lane * 4 + j;
            accum = fma(float(weights[group.x * 2560 + col]), float(x[start + long(col) * layout[2]]), accum);
        }
    }
    for (ushort offset = 16; offset > 0; offset /= 2)
        accum += simd_shuffle_down(accum, offset);
    if (lane == 0) out[group.y * 512 + group.x] = T(accum);
}

// Each accumulator retains the reference's 80 FMA chain and shuffle tree.
template <typename T>
kernel void ar30_gate_dot_pair(const device T* x [[buffer(0)]], const device half* weights [[buffer(1)]],
    device T* out [[buffer(2)]], constant long* layout [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    const uint first = group.x * 2;
    if (first >= 512) return;
    long start = layout[0] + long(group.y) * layout[1];
    float a = 0, b = 0;
    for (uint block = 0; block < 2560; block += 128) {
        for (uint j = 0; j < 4; ++j) {
            uint col = block + lane * 4 + j;
            float value = float(x[start + long(col) * layout[2]]);
            a = fma(float(weights[first * 2560 + col]), value, a);
            b = fma(float(weights[(first + 1) * 2560 + col]), value, b);
        }
    }
    for (ushort offset = 16; offset > 0; offset /= 2) {
        a += simd_shuffle_down(a, offset);
        b += simd_shuffle_down(b, offset);
    }
    if (lane == 0) {
        out[group.y * 512 + first] = T(a);
        out[group.y * 512 + first + 1] = T(b);
    }
}

template <typename T>
kernel void gate_softmax(const device T* logits [[buffer(0)]], device T* probabilities [[buffer(1)]],
    constant uint* bindings [[buffer(2)]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint simd [[simdgroup_index_in_threadgroup]],
    uint row [[threadgroup_position_in_grid]]) {
    threadgroup float partial[4];
    const uint start = row*512 + 4*tid;
    float v[4], m = -INFINITY;
    for (uint j=0; j<4; ++j) { v[j] = float(logits[start+j]); m = max(m, v[j]); }
    m = simd_max(m);
    if (lane==0) partial[simd] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd==0) {
        m = simd_max(lane < 4 ? partial[lane] : -INFINITY);
        if (lane==0) partial[0] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = partial[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total=0;
    for (uint j=0; j<4; ++j) { v[j] = fast::exp(v[j]-m); total += v[j]; }
    total = simd_sum(total);
    if (lane==0) partial[simd] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd==0) {
        total = simd_sum(lane < 4 ? partial[lane] : 0.0f);
        if (lane==0) partial[0] = 1.0f/total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j=0; j<4; ++j) probabilities[start+j] = T(v[j]*partial[0]);
}

#define GATE_KERNELS(T, DOT, SOFTMAX) \
template [[host_name(DOT)]] kernel void gate_dot<T>(const device T*, const device half*, device T*, constant long*, uint, uint2); \
template [[host_name(SOFTMAX)]] kernel void gate_softmax<T>(const device T*, device T*, constant uint*, uint, uint, uint, uint);
GATE_KERNELS(half, "gate_dot_half", "gate_softmax_half")
GATE_KERNELS(float, "gate_dot_float", "gate_softmax_float")
#undef GATE_KERNELS

template [[host_name("ar30_gate_dot_pair_half")]] kernel void ar30_gate_dot_pair<half>(const device half*, const device half*, device half*, constant long*, uint, uint2);
template [[host_name("ar30_gate_dot_pair_float")]] kernel void ar30_gate_dot_pair<float>(const device float*, const device half*, device float*, constant long*, uint, uint2);
