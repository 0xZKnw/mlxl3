// EXL3 format and reduction order ported from MLXL3's former Python prototype.
// Upstream attribution: ../../THIRD_PARTY_NOTICES.md.
#include <metal_stdlib>
using namespace metal;

struct Params { uint count; uint k; uint cb; uint rows; uint cols; };

inline half codeword(uint x, uint cb) {
    x &= 0xffffu;
    if (cb < 2) {
        x = cb == 0 ? x * 89226354u + 64248484u : x * 0xCBAC1FEDu;
        x = 0x3B603B60u ^ (x & 0x8FFF8FFFu);
        return as_type<half>(ushort(x)) + as_type<half>(ushort(x >> 16));
    }
    x *= 0x83DCD12Du;
    uint sum = 0x6400u + (x & 255u) + ((x >> 8) & 255u)
        + ((x >> 16) & 255u) + ((x >> 24) & 255u);
    return fma(as_type<half>(ushort(sum)), as_type<half>(ushort(0x1eee)),
        as_type<half>(ushort(0xc931)));
}

inline ushort state(const device ushort* packed, uint tile, uint t, uint k) {
    uint b0 = t * k + k + 256u * k - 16u;
    uint b1 = b0 + 16u;
    uint i0 = (b0 >> 5) % (8u * k);
    uint last = (b1 - 1u) >> 5;
    uint i1 = last % (8u * k);
    uint shift = ((last + 1u) << 5) - b1;
    uint base = tile * 16u * k;
    uint a = uint(packed[base + 2u * i0]) | (uint(packed[base + 2u * i0 + 1u]) << 16);
    uint b = uint(packed[base + 2u * i1]) | (uint(packed[base + 2u * i1 + 1u]) << 16);
    return ushort(((ulong(a) << 32) | ulong(b)) >> shift);
}

kernel void exl3_pack(const device ushort* encoded [[buffer(0)]],
    device ushort* packed [[buffer(1)]], constant Params& p [[buffer(2)]],
    uint elem [[thread_position_in_grid]]) {
    if (elem >= p.count) return;
    uint tile = elem / (16u * p.k);
    uint logical = (elem % (16u * p.k)) ^ 1u;
    uint base = tile * 256u + (logical / p.k) * 16u;
    uint word_idx = logical % p.k;
    ushort word = 0;
    for (uint bit = 0; bit < 16u; ++bit) {
        uint stream_bit = word_idx * 16u + bit;
        word = ushort((word << 1) | ((encoded[base + stream_bit / p.k]
            >> (p.k - 1u - stream_bit % p.k)) & 1u));
    }
    packed[elem] = word;
}

kernel void exl3_decode(const device ushort* packed [[buffer(0)]],
    device ushort* states [[buffer(1)]], device half* values [[buffer(2)]],
    constant Params& p [[buffer(3)]], uint elem [[thread_position_in_grid]]) {
    if (elem >= p.count) return;
    ushort s = state(packed, elem >> 8, elem & 255u, p.k);
    states[elem] = s;
    values[elem] = codeword(s, p.cb);
}

kernel void exl3_codebook(device half* values [[buffer(0)]],
    constant Params& p [[buffer(1)]], uint elem [[thread_position_in_grid]]) {
    if (elem < p.count) values[elem] = codeword(elem, p.cb);
}

// Conservative serialized QMV, matching the existing _qmv_inner_kernel.
// No dense weight matrix is allocated. Tiled/split-K dispatch is a separate port.
inline float qmv_column(const device half* x, const device ushort* packed,
    const device ushort* inv, constant Params& p, threadgroup float* partials,
    uint tid, uint lane, uint simd, uint n) {
    float acc = 0;
    for (uint row = tid; row < p.rows; row += 128u) {
        uint tile = (row >> 4) * (p.cols >> 4) + (n >> 4);
        uint t = inv[((row & 15u) << 4) | (n & 15u)];
        acc = fma(float(x[row]), float(codeword(state(packed, tile, t, p.k), p.cb)), acc);
    }
    acc = simd_sum(acc);
    if (lane == 0) partials[simd] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = simd_sum(tid < 4 ? partials[tid] : 0.0f);
    return total;
}

kernel void exl3_qmv(const device half* x [[buffer(0)]],
    const device ushort* packed [[buffer(1)]], const device ushort* inv [[buffer(2)]],
    device half* y [[buffer(3)]], constant Params& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint n [[threadgroup_position_in_grid]]) {
    threadgroup float partials[4];
    float total = qmv_column(x, packed, inv, p, partials, tid, lane, simd, n);
    if (tid == 0) y[n] = half(total);
}

kernel void flash_k7_qmv(const device half* x [[buffer(0)]],
    const device ushort* packed [[buffer(1)]], const device ushort* inv [[buffer(2)]],
    device float* y [[buffer(3)]], constant Params& p [[buffer(4)]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint simd [[simdgroup_index_in_threadgroup]], uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float partials[4];
    float total = qmv_column(x+group.y*p.rows, packed, inv, p, partials, tid, lane, simd, group.x);
    if (tid == 0) y[group.y*p.cols + group.x] = total;
}

// One thread owns both elements of its butterfly. Each stage completes before
// the next can read it. The two FP16 boundaries reproduce radix-16 then radix-8.
inline void hadamard128(threadgroup float* values, uint lane) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1; stride <= 64; stride <<= 1) {
        uint low = 2 * (lane & ~(stride - 1)) + (lane & (stride - 1));
        float a = values[low], b = values[low + stride];
        float plus = a + b, minus = a - b;
        if (stride == 8 || stride == 64) {
            plus = float(half(plus));
            minus = float(half(minus));
        }
        values[low] = plus;
        values[low + stride] = minus;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void exl3_prepare128(const device half* x [[buffer(0)]],
    const device half* suh [[buffer(1)]], device half* output [[buffer(2)]],
    constant uint& scale_count [[buffer(3)]], uint lane [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
    threadgroup float values[128];
    uint base = block * 128;
    for (uint i = lane; i < 128; i += 64)
        values[i] = float(half(x[base + i] * suh[(base + i) % scale_count]));
    hadamard128(values, lane);
    const float normalization = as_type<float>(0x3db504f3u);
    for (uint i = lane; i < 128; i += 64)
        output[base + i] = half(values[i] * normalization);
}

kernel void flash_finish128(const device float* partials [[buffer(0)]],
    const device half* svh [[buffer(1)]], const device half* bias [[buffer(2)]],
    device half* output [[buffer(3)]], constant Params& p [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    threadgroup float values[128];
    uint base = block * 128, col = base % p.cols;
    uint first = (base / p.cols * p.count) * p.cols + col;
    for (uint i = lane; i < 128; i += 64) {
        float sum = partials[first+i];
        if (p.count == 2u) sum = (partials[first+p.cols+i] + 0.0f) + (sum + 0.0f);
        values[i] = float(half(sum));
    }
    hadamard128(values, lane);
    for (uint i = lane; i < 128; i += 64) {
        half y = half(values[i] * as_type<float>(0x3db504f3u)) * svh[col+i];
        output[base+i] = p.cb ? half(y + bias[col+i]) : y;
    }
}

#include "flash_projection.metal"
