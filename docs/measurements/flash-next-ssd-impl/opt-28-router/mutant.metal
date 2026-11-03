uint row = threadgroup_position_in_grid.y;
uint sg = simdgroup_index_in_threadgroup;
uint lane = thread_index_in_simdgroup;
uint index = sg * 32u + lane;
float value = float(values[row * 512u + index]);
uint bits = value == 0.0f ? 0u : as_type<uint>(value);
uint key = isnan(value) ? 0xffffffffu
    : bits ^ uint((int(bits) >> 31) | int(0x80000000u));
threadgroup uint local_keys[160];
threadgroup uint local_ids[160];
threadgroup uint selected[10];
threadgroup float top_values[10];
for (uint rank = 0u; rank < 10u; ++rank) {
    uint best = simd_max(key);
    uint winner = simd_min(key == best ? index : 0xffffffffu);
    if (lane == 0u) {
        local_keys[sg * 10u + rank] = best;
        local_ids[sg * 10u + rank] = winner;
    }
    if (index == winner) key = 0u;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0u) {
    uint keys[5], ids[5];
    for (uint i = 0u; i < 5u; ++i) {
        keys[i] = local_keys[lane + i * 32u];
        ids[i] = local_ids[lane + i * 32u];
    }
    for (uint rank = 0u; rank < 10u; ++rank) {
        uint candidate = 0u, id = 0u;
        for (uint i = 0u; i < 5u; ++i) {
            if (keys[i] > candidate || (keys[i] == candidate && ids[i] > id)) {
                candidate = keys[i]; id = ids[i];
            }
        }
        uint best = simd_max(candidate);
        uint winner = simd_max(candidate == best ? id : 0u);
        if (lane == 0u) {
            selected[9u - rank] = winner;
            top_values[9u - rank] = float(values[row * 512u + winner]);
        }
        for (uint i = 0u; i < 5u; ++i) {
            if (ids[i] == winner) keys[i] = 0u;
        }
    }
    if (lane == 0u) {
        ARITH sum = ARITH(0);
        for (uint slot = 0u; slot < 10u; ++slot) {
            sum = ARITH(top_values[slot]) + sum;
        }
        for (uint slot = 0u; slot < 10u; ++slot) {
            ARITH score = ARITH(top_values[slot]);
            indices[row * 10u + slot] = selected[slot];
            scores[row * 10u + slot] = NORMALIZE ? score / sum : score;
        }
    }
}
