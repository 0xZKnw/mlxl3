    constexpr uint INTC = uint(INTER);
    constexpr uint KSC = uint(KS);
    constexpr uint NBHC = uint(NBH);
    constexpr float HAD_SCALE = 0.08838834764831845f;
    const uint b = threadgroup_position_in_grid.x;
    const uint rk = threadgroup_position_in_grid.y;
    const uint L = thread_index_in_simdgroup;
