// PonyExl3 8e7fa6b Metal trellis reference, retaining half costs and both tail-biting passes.
#include <metal_stdlib>
using namespace metal;
#define EDGES (1u << (16u - MLXL3_QUANT_K))
#define KK (1u << MLXL3_QUANT_K)
#define KR_BITS (16u - MLXL3_QUANT_K)
#define H_INF_F 65504.0f

inline float mlxl3_quantize_codeword(uint cw) {
#if MLXL3_QUANT_CB == 0
    uint bits = cw * 89226354u + 64248484u;
#elif MLXL3_QUANT_CB == 1
    uint bits = cw * 0xCBAC1FEDu;
#else
    uint bits = cw * 0x83DCD12Du;
    uint sum = 0x6400u;
    for (uint lane = 0; lane < 4; lane++) sum += (bits >> (8u * lane)) & 0xFFu;
    half value = as_type<half>(ushort(sum & 0xFFFFu));
    half scale = as_type<half>(ushort(0x1EEEu));
    half bias = as_type<half>(ushort(0xC931u));
    return float(value * scale + bias);
#endif
#if MLXL3_QUANT_CB < 2
    half2 values = as_type<half2>((bits & 0x8FFF8FFFu) ^ 0x3B603B60u);
    return float(values.x + values.y);
#endif
}

kernel void mlxl3_quantize_tiles(
    const device float* tiles [[buffer(0)]],
    device float* q_tiles [[buffer(1)]],
    device ushort* indices [[buffer(2)]],
    device ushort* temp_edges [[buffer(3)]],
    device half* temp_costs [[buffer(4)]],
    uint tile [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    const device float* in_tile = tiles + tile * 256u;
    device float* out_tile = q_tiles + tile * 256u;
    device ushort* out_idx = indices + tile * 256u;
    device ushort* edge_ptr = temp_edges + ulong(tile) * 256u * EDGES;
#if MLXL3_QUANT_K < 4
    device half* costs = temp_costs + ulong(tile) * 2u * EDGES;
#define COST(buf, edge) costs[ulong(buf) * EDGES + (edge)]
#define SYNC threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#else
    threadgroup half costs[2][EDGES];
#define COST(buf, edge) costs[buf][edge]
#define SYNC threadgroup_barrier(mem_flags::mem_threadgroup)
#endif
    threadgroup uint sh_pin;
    uint pin = 0;
    bool pinned = false;
    for (uint pass = 0; pass < 2; pass++) {
        uint curr = 0;
        uint roll = pass == 0 ? 128u : 0u;
        for (uint e = tid; e < EDGES; e += 256u) {
            float target = in_tile[roll];
            float best = H_INF_F;
            uint best_pred = 0;
            for (uint fresh = 0; fresh < KK; fresh++) {
                uint cw = (fresh << KR_BITS) | e;
                uint pred = cw >> MLXL3_QUANT_K;
                float err;
                if (pinned && pred != pin) {
                    err = H_INF_F;
                } else {
                    float dh = mlxl3_quantize_codeword(cw) - target;
                    err = dh * dh;
                    if (err > H_INF_F) err = H_INF_F;
                }
                if (err < best) { best = err; best_pred = pred; }
            }
            COST(curr, e) = half(best);
            if (pass == 1 || roll < 128u) edge_ptr[roll * EDGES + e] = ushort(best_pred);
        }
        SYNC;
        for (uint step = 1; step < 256u; step++) {
            uint ri = (step + roll) & 255u;
            uint prev_buf = curr;
            curr = 1u - curr;
            for (uint e = tid; e < EDGES; e += 256u) {
                float target = in_tile[ri];
                float best = H_INF_F;
                uint best_pred = 0;
                for (uint fresh = 0; fresh < KK; fresh++) {
                    uint cw = (fresh << KR_BITS) | e;
                    uint pred = cw >> MLXL3_QUANT_K;
                    float dh = mlxl3_quantize_codeword(cw) - target;
                    float err = fma(dh, dh, float(COST(prev_buf, pred)));
                    if (err > H_INF_F) err = H_INF_F;
                    if (err < best) { best = err; best_pred = pred; }
                }
                COST(curr, e) = half(best);
                if (pass == 1 || ri < 128u) edge_ptr[ulong(ri) * EDGES + e] = ushort(best_pred);
            }
            SYNC;
        }
        if (tid == 0) {
            uint edge = pin;
            if (pass == 0) {
                float best = H_INF_F;
                edge = 0;
                for (uint e = 0; e < EDGES; e++) {
                    float value = float(COST(curr, e));
                    if (value < best) { best = value; edge = e; }
                }
            }
            for (int step_i = 255; step_i >= 0; step_i--) {
                uint ri = (uint(step_i) + roll) & 255u;
                uint prev = uint(edge_ptr[ulong(ri) * EDGES + edge]);
                uint state = (prev << MLXL3_QUANT_K) | edge;
                edge = prev;
                if (pass == 1) {
                    out_idx[ri] = ushort(state);
                    out_tile[ri] = mlxl3_quantize_codeword(state);
                } else if (ri == 0) { break; }
            }
            sh_pin = edge;
        }
        SYNC;
        pin = sh_pin;
        pinned = true;
    }
}
