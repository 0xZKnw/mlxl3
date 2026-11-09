    constexpr uint DIMC = uint(DIM);
    constexpr uint NB = DIMC / 128u;
    constexpr uint HDC = uint(HD);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint b = threadgroup_position_in_grid.x;
    const uint row = threadgroup_position_in_grid.y;
    const uint S = threadgroups_per_grid.y;
    const uint L = thread_index_in_simdgroup;
    float v[4];
    for (uint j = 0; j < 4u; ++j) {
        const uint i = b * 128u + L + 32u * j;
        const uint hh = i / HDC;
        const uint d = i - hh * HDC;
        const T xv = attn[(ulong(hh) * S + row) * HDC + d];
        const T gv = gate[ulong(row) * uint(GROW) + hh * uint(GHEAD) + uint(GOFF) + d];
        const T yv = xv * sigtab[as_type<ushort>(gv)];
        v[j] = float(half(float(yv))) * float(suh[i]);
    }
