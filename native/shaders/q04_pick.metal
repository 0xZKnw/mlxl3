// Parallel picker adapted from beamivalice/sushi@0bb32df5, moe_stream_compute.zig.
// Copyright (c) 2026 Theinruj Toranavikrai; Copyright (c) 2026 David Dalcu.
// MIT license: see ../../docs/licenses/sushi-MIT.txt.
#include <metal_stdlib>
using namespace metal;

template<typename T>
inline void q04_pick(device const T* p, device const T* weights,
    device const uint* ids, device const uint* ready, device const uchar* starved,
    device uint* picked, constant uint* args, uint t, uint lane, uint sg,
    threadgroup bool* taken, threadgroup uint* selected,
    threadgroup uint* order, threadgroup bool* protected_rank,
    threadgroup uint* counts, threadgroup float* red_v, threadgroup int* red_i) {
    for (uint e=t; e<512; e+=128) taken[e]=false;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t==0) {
        counts[0]=0; counts[1]=0;
        for (uint rank=0; rank<10; ++rank) {
            uint e=ids[rank]; selected[rank]=e; taken[e]=true;
            uint above=0;
            for (uint j=0; j<10; ++j)
                above += float(p[ids[j]])>float(p[e]) ||
                    (float(p[ids[j]])==float(p[e]) && ids[j]<e);
            protected_rank[rank]=above<3;
            if (ready[e]) continue;
            uint at=counts[0];
            while (at>0 && float(p[ids[order[at-1]]])<float(p[e])) {
                order[at]=order[at-1]; --at;
            }
            order[at]=rank; ++counts[0];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float factor=as_type<float>(args[0]);
    for (uint i=0; i<counts[0]; ++i) {
        uint rank=order[i], missed=ids[rank];
        if (protected_rank[rank] || starved[missed]>=3 || counts[1]>=args[1] ||
            !(float(weights[rank])>0 && float(weights[rank])<=0.1f) ||
            !(float(p[missed])>0) || !args[2]) continue;
        float best=-INFINITY; int pick=-1;
        for (uint e=t; e<512; e+=128) {
            float v=float(p[e]);
            if (taken[e] || !ready[e] || !(v>=factor*float(p[missed]))) continue;
            if (v>best || (v==best && (pick<0 || int(e)<pick))) { best=v; pick=int(e); }
        }
        for (uint off=16; off>0; off>>=1) {
            float v=simd_shuffle_down(best,off); int e=simd_shuffle_down(pick,off);
            if (lane+off<32 && e>=0 && (pick<0 || v>best || (v==best && e<pick))) {
                best=v; pick=e;
            }
        }
        if (lane==0) { red_v[sg]=best; red_i[sg]=pick; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t==0) {
            float v=-INFINITY; int e=-1;
            for (uint j=0; j<4; ++j)
                if (red_i[j]>=0 && (e<0 || red_v[j]>v || (red_v[j]==v && red_i[j]<e))) {
                    v=red_v[j]; e=red_i[j];
                }
            if (e>=0) { selected[rank]=uint(e); taken[e]=true; ++counts[1]; }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (t<10) picked[t]=selected[t];
}

#define Q04_KERNEL(TYPE, NAME) \
kernel void NAME(device const TYPE* p [[buffer(0)]], device const TYPE* weights [[buffer(1)]], \
    device const uint* ids [[buffer(2)]], device const uint* ready [[buffer(3)]], \
    device const uchar* starved [[buffer(4)]], device uint* picked [[buffer(5)]], \
    constant uint* args [[buffer(6)]], uint t [[thread_index_in_threadgroup]], \
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) { \
    threadgroup bool taken[512], protected_rank[10]; \
    threadgroup uint selected[10], order[10], counts[2]; \
    threadgroup float red_v[4]; threadgroup int red_i[4]; \
    q04_pick(p,weights,ids,ready,starved,picked,args,t,lane,sg,taken,selected,order,protected_rank,counts,red_v,red_i); \
}
Q04_KERNEL(half, q04_pick_half)
Q04_KERNEL(float, q04_pick_float)

// Exact IEEE widening also preserves Half subnormals when the GPU flushes half arithmetic.
kernel void mtp_hidden_widen(device const ushort* x [[buffer(0)]],
    device uint* out [[buffer(1)]], device atomic_uint* invalid [[buffer(2)]],
    constant uint* count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
    if (i>=count[0]) return;
    uint word=x[i], sign=(word&0x8000u)<<16, exponent=(word>>10)&31u, mantissa=word&1023u;
    uint value;
    if (exponent==0) value=as_type<uint>(float(mantissa)*0x1p-24f);
    else if (exponent==31) {
        atomic_fetch_or_explicit(invalid,1u,memory_order_relaxed);
        value=0x7f800000u|(mantissa<<13);
    } else value=((exponent+112u)<<23)|(mantissa<<13);
    out[i]=sign|value;
}
