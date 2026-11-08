    constexpr uint HIDC = uint(HID);
    constexpr uint TOPKC = uint(TOPK);
    constexpr uint KSC = uint(KS);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint blk = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint k = simdgroup_index_in_threadgroup;
    const uint L = thread_index_in_simdgroup;
    const uint tid = thread_index_in_threadgroup;
    threadgroup float term[TOPKC * 128u];
    const float MUL1_A = float(as_type<half>(ushort(0x1EEEu)));
    const float MUL1_C = 1024.0f * MUL1_A + float(as_type<half>(ushort(0xC931u)));
    const uint rk = row * TOPKC + k;
    const uint e = uint(slots[rk]);
    float v[4];
    for (uint q = 0; q < 4u; ++q) {
        const uint o = blk * 128u + L + 32u * q;
        float t = 0.0f;
        for (uint ks = 0; ks < KSC; ++ks) t += zd[(ulong(ks) * ulong(RKC) + rk) * ulong(HIDC) + o];
        v[q] = t;
    }
    if (DEFER != 0) {
        float hsum = 0.0f;
        for (uint q = 0; q < uint(NBI); ++q) hsum += hsx[ulong(rk) * uint(NBI) + q];
        for (uint q = 0; q < 4u; ++q) v[q] = fma(v[q], MUL1_A, MUL1_C * hsum);
    }
    for (uint q = 0; q < 4u; ++q) v[q] = float(half(v[q]));
    turboexl3_h128r(v, L);
    for (uint q = 0; q < 4u; ++q) {
        const uint o = blk * 128u + L + 32u * q;
        term[k * 128u + L + 32u * q] = float(half(v[q] * HAD_SCALE * float(down_svh[e * HIDC + o])));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128u) {
        float acc = 0.0f;
        for (uint j = 0; j < TOPKC; ++j) acc = fma(float(scores[row * TOPKC + j]), term[j * 128u + tid], acc);
        y[ulong(row) * ulong(HIDC) + blk * 128u + tid] = OUT_T(acc);   // the caller's dtype (bf16 served, f32 in the tests)
    }
