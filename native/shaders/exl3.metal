// EXL3 format and reduction order ported from MLXL3's former Python prototype.
// Upstream attribution: ../../THIRD_PARTY_NOTICES.md.
#include <metal_stdlib>
using namespace metal;

struct Params { uint count; uint k; uint cb; uint rows; uint cols; };

inline half codeword_canonical(uint x, uint cb) {
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

#ifndef MLXL3_AR20_SWAR
#define MLXL3_AR20_SWAR 0
#endif
#ifndef MLXL3_AR20_READER8
#define MLXL3_AR20_READER8 0
#endif
#ifndef MLXL3_AR20_PRELOAD
#define MLXL3_AR20_PRELOAD 0
#endif
#ifndef MLXL3_AR20_H128
#define MLXL3_AR20_H128 0
#endif
inline half codeword(uint x, uint cb) {
#if MLXL3_AR20_SWAR
    if (cb == 2u) {
        x = (x & 0xffffu) * 0x83DCD12Du;
        uint pairs = (x & 0x00ff00ffu) + ((x >> 8u) & 0x00ff00ffu);
        uint sum = 0x6400u + (pairs & 0xffffu) + (pairs >> 16u);
        return fma(as_type<half>(ushort(sum)), as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931)));
    }
#endif
    return codeword_canonical(x, cb);
}

#include "ar20_trellis.metal"

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

// The up projection's half result remains an explicit rounding boundary.
kernel void flash_up_finish_swiglu(const device float* partials [[buffer(0)]],
    const device half* svh [[buffer(1)]], const device half* bias [[buffer(2)]],
    device half* output [[buffer(3)]], const device half* gate [[buffer(4)]],
    constant Params& p [[buffer(5)]],
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
        half up = p.cb ? half(y + bias[col+i]) : y;
        half g = gate[base+i];
        half tail = half(1) / (half(1) + metal::exp(metal::abs(g)));
        half sigmoid = g < half(0) ? tail : half(1) - tail;
        output[base+i] = (g * sigmoid) * up;
    }
}

kernel void flash_swiglu(const device half* gate [[buffer(0)]],
    const device half* up [[buffer(1)]], device half* dest [[buffer(2)]],
    constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    half g = gate[i];
    half tail = half(1) / (half(1) + metal::exp(metal::abs(g)));
    half sigmoid = g < half(0) ? tail : half(1) - tail;
    dest[i] = (g * sigmoid) * up[i];
}

kernel void flash_prepare_view(const device uchar* input [[buffer(0)]],
    const device half* scales [[buffer(1)]], device half* output [[buffer(2)]],
    constant long* p [[buffer(3)]], uint lane [[thread_index_in_threadgroup]],
    uint block [[threadgroup_position_in_grid]]) {
    threadgroup float values[128];
    long start = long(block) * 128, row = start / p[0], col = start % p[0];
    for (uint i=lane; i<128; i+=64) {
        long xi = p[1] + row*p[2] + (col+i)*p[3];
        long si = p[4] + row*p[5] + (col+i)*p[6];
        half x = p[7] ? half(reinterpret_cast<const device float*>(input)[xi])
                      : reinterpret_cast<const device half*>(input)[xi];
        values[i]=float(x * scales[si]);
    }
    hadamard128(values,lane);
    for (uint i=lane; i<128; i+=64)
        output[start+i]=half(values[i]*as_type<float>(0x3db504f3u));
}

// Two different Suh transforms; each keeps the canonical half multiply/H128.
kernel void flash_dual_gu_prepare(const device uchar* input [[buffer(0)]],
    const device half* gate_suh [[buffer(1)]], const device half* up_suh [[buffer(2)]],
    device half* output [[buffer(3)]], constant long* p [[buffer(4)]],
    uint lane [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    threadgroup float gate[128], up[128];
    uint start = block * 128;
    for (uint i = lane; i < 128; i += 64) {
        long xi = p[0] + long(start+i) * p[1];
        half x = p[2] ? half(reinterpret_cast<const device float*>(input)[xi])
                      : reinterpret_cast<const device half*>(input)[xi];
        gate[i] = float(x * gate_suh[start+i]);
        up[i] = float(x * up_suh[start+i]);
    }
    hadamard128(gate, lane);
    hadamard128(up, lane);
    for (uint i = lane; i < 128; i += 64) {
        output[start+i] = half(gate[i] * as_type<float>(0x3db504f3u));
        output[p[3]+start+i] = half(up[i] * as_type<float>(0x3db504f3u));
    }
}

#include "router_topk.h"

template <typename T>
kernel void route_matrix(const device T* values [[buffer(0)]], device uint* indices [[buffer(1)]],
    device T* scores [[buffer(2)]], constant long* p [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    threadgroup ulong keys[1024];
    threadgroup float selected[1024];
    uint k = uint(p[1]);
    long address = p[3] + long(row)*p[4] + long(tid)*p[5];
    mlxl3_rank_row(values[address], tid, uint(p[0]), k, bool(p[2]), keys, selected,
                   indices + row*k, scores + row*k);
}

#define ROUTE_KERNEL(TYPE, NAME) template [[host_name(NAME)]] kernel void route_matrix<TYPE>( \
    const device TYPE*, device uint*, device TYPE*, constant long*, uint, uint);
ROUTE_KERNEL(half, "route_half")
ROUTE_KERNEL(float, "route_float")
#undef ROUTE_KERNEL

#include "flash_gate.metal"

#include "flash_moe.metal"

#include "flash_hc.metal"

#include "flash_gdn.metal"

#include "flash_ple.metal"

#include "flash_qsa.metal"
#include "flash_logsumexp.metal"

#include "flash_mtp.metal"

#include "dense_projection.metal"

#include "mapped_expert.metal"
#include "model_ops.metal"

#include "ar20_kernel_tests.metal"
