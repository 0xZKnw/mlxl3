        for (uint t = 0; t < BTC; ++t) {
            const device uint* tile = wbase + (ulong(tk) * OUT_TILES + blk * BTC + t) * ulong(NW);
            const auto ww = exl3_tile_load<N, 0>(tile, qb, ga, lp);
            uint cw[8];
            cw[0] = exl3_tile_cw<N, WMASK, 0, 0>(ww, lp); cw[1] = exl3_tile_cw<N, WMASK, 1, 0>(ww, lp); cw[2] = exl3_tile_cw<N, WMASK, 2, 0>(ww, lp); cw[3] = exl3_tile_cw<N, WMASK, 3, 0>(ww, lp);
            cw[4] = exl3_tile_cw<N, WMASK, 4, 0>(ww, lp); cw[5] = exl3_tile_cw<N, WMASK, 5, 0>(ww, lp); cw[6] = exl3_tile_cw<N, WMASK, 6, 0>(ww, lp); cw[7] = exl3_tile_cw<N, WMASK, 7, 0>(ww, lp);
            for (uint m = 0; m < 8u; ++m) {
                // DEFER: MUL1's byte sum t; the finish applies A and C (C times the row's input sum)
                const float wv = (DEFER != 0) ? exl3_mul1_t(cw[m]) : float(exl3_decode<CB>(cw[m]));
                for (uint r = 0; r < RBC; ++r) a[r][t][m >> 2] = fma(xr[r][m & 3u], wv, a[r][t][m >> 2]);
            }
        }
    }
    for (uint r = 0; r < RBC; ++r) turboexl3_fold<BTC>(a[r], &red[sg][r * 16u * BTC], L);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < RBC * 16u * BTC; i += 128u) {
        const uint r = i / (16u * BTC);
        const uint row = r0 + r;
        if (row < ROWSC) zc[(ulong(ksl) * ulong(ROWSC) + row) * ulong(OUTC) + blk * (16u * BTC) + (i - r * 16u * BTC)] = ((red[0][i] + red[1][i]) + red[2][i]) + red[3][i];
    }
    // arrival: this threadgroup's stores fenced device-wide before its one count; the count that reaches ARR finishes
    // the output block here, every other threadgroup is done (nothing waits)
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        const uint old = atomic_fetch_add_explicit((device atomic_uint*)cnt + (ob * NCHC + chunk), 1u, memory_order_relaxed);
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        arrived_last = (old == ARR - 1u) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (arrived_last == 0u) return;
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    // text_dfinish for this output block, simdgroup rr for the chunk's row rr
    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    for (uint rr = sg; rr < RBC; rr += 4u) {
        const uint row = r0 + rr;
        if (row >= ROWSC) break;
        float xsum = 0.0f;

