        d[q] = float(half(hv)) * float(down_suh[e * INTC + o]);
    }
    turboexl3_h128r(d, L);
    float s[4];
    for (uint q = 0; q < 4u; ++q) {
        const uint i = L + 32u * q;
        const uint r = i & 15u; const uint qq = (r & 7u) >> 1u; const uint mm = (r & 1u) | ((r >> 3u) << 1u);
        const half hh = half(d[q] * HAD_SCALE);
        hd[ulong(rk) * ulong(INTC) + b * 128u + (i & ~15u) + 4u * qq + mm] = hh;
        s[q] = simd_sum(float(hh));
    }
    if (L == 0u) hs[ulong(rk) * ulong(INTC / 128u) + b] = ((s[0] + s[1]) + s[2]) + s[3];
