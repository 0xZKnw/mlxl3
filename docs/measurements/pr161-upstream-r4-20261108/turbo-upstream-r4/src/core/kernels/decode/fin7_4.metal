        for (uint q = 0; q < 4u; ++q) { g[q] = fma(g[q], MUL1_A, MUL1_C * sgx); u[q] = fma(u[q], MUL1_A, MUL1_C * sux); }
    }
    for (uint q = 0; q < 4u; ++q) { g[q] = float(half(g[q])); u[q] = float(half(u[q])); }
    turboexl3_h128r(g, L);
    turboexl3_h128r(u, L);
    float d[4];
    for (uint q = 0; q < 4u; ++q) {
        const uint o = b * 128u + L + 32u * q;
        const float gv = float(half(g[q] * HAD_SCALE * float(gate_svh[e * INTC + o])));
        const float uv = float(half(u[q] * HAD_SCALE * float(up_svh[e * INTC + o])));
