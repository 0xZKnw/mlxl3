    T zf[4];
    for (uint j = 0; j < 4u; ++j) { const uint o = head * 128u + L + 32u * j; zf[j] = T(float(half(v[j] * HAD_SCALE * float(svh[o])))); }
    // the gated norm's RMS over the head as MLX's rms_norm sums it at axis 128 (one simdgroup): lane L sums channels
    // 4L..4L+3 in order, then simd_sum
    uint base = (row * uint(HV) + head) * uint(DV) + L * 4;
    float xs[4];
    float sumsq = 0.0f;
    for (uint i = 0; i < 4; ++i) {
        xs[i] = float(y[base + i]);
        sumsq += xs[i] * xs[i];
    }
    sumsq = simd_sum(sumsq);
    float inv = metal::precise::rsqrt(sumsq / float(DV) + eps);
    // the gated value of channel c = L + 32 j: MLX's rms_norm output times MLX's sigmoid of z by the table, in T; then
    // out_proj's prepare of it (dense text_prep's statements, natural order)
    const uint hbase = (row * uint(HV) + head) * uint(DV);
    for (uint j = 0; j < 4u; ++j) {
        const uint c = L + 32u * j;
        const T normed = norm_w[c] * T(float(y[hbase + c]) * inv);
        const T sig = sigtab[as_type<ushort>(zf[j])];
        const T gated = normed * sig;
        v[j] = float(half(float(gated))) * float(suh[head * 128u + c]);
    }
