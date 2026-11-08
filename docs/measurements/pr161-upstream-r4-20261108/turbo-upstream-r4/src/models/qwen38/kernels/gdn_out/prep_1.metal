    constexpr float HAD_SCALE = 0.08838834764831845f;
    constexpr uint VD = uint(HV) * uint(DV);
    const uint L = thread_index_in_simdgroup;
    const uint row = threadgroup_position_in_grid.y;
    const uint head = threadgroup_position_in_grid.z * uint(NSG) + simdgroup_index_in_threadgroup;
    // in_proj_z's finish of block `head` (dense text_finish's statements): zf[j] is z at channel head * 128 + L + 32 j
    float v[4];
    for (uint j = 0; j < 4u; ++j) v[j] = float(half(float(zq[row * VD + head * 128u + L + 32u * j])));
