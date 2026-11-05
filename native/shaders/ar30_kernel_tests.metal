#if MLXL3_AR20_TESTS
kernel void ar30_compiled_cooperative(device uint* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid == 0u) out[0] = MLXL3_AR30_COOPERATIVE;
}

template <bool Mutant>
inline void ar30_test_cooperative_states(const device ushort* packed,
    device ushort* out, uint k, uint lane, uint tile) {
    uint cw[8];
    if (k >= 2u && k <= 4u) {
        const device uint* words = reinterpret_cast<const device uint*>(packed) + tile * 8u * k;
        uint owned = ar30_cooperative_load(words, lane, k);
        for (uint group = 0u; group < 2u; ++group) {
            uint end = (lane * 8u + group * 4u + 4u) * k + 256u * k;
            uint last = (end - 1u) / 32u;
            uint high = (last - 1u) % (8u * k), low = last % (8u * k);
            if (Mutant) high = (high + 1u) % (8u * k);
            ulong merged = ar30_cooperative_pair(owned, high, low);
            uint window = uint(merged >> ((last + 1u) * 32u - end));
            for (uint j = 0u; j < 4u; ++j)
                cw[group * 4u + j] = (window >> ((3u - j) * k)) & 0xffffu;
        }
    } else {
        for (uint j = 0u; j < 8u; ++j) cw[j] = state(packed, tile, lane * 8u + j, k);
    }
    for (uint j = 0u; j < 8u; ++j) out[tile * 256u + lane * 8u + j] = ushort(cw[j]);
}

kernel void ar30_cooperative_states(const device ushort* packed [[buffer(0)]],
    device ushort* out [[buffer(1)]], device ushort* original [[buffer(2)]],
    constant uint& k [[buffer(3)]], uint lane [[thread_index_in_simdgroup]],
    uint tile [[threadgroup_position_in_grid]]) {
    ar30_test_cooperative_states<false>(packed, out, k, lane, tile);
    for (uint j = 0u; j < 8u; ++j)
        original[tile * 256u + lane * 8u + j] = state(packed, tile, lane * 8u + j, k);
}
kernel void ar30_cooperative_wrong_owner(const device ushort* packed [[buffer(0)]],
    device ushort* out [[buffer(1)]], constant uint& k [[buffer(2)]],
    uint lane [[thread_index_in_simdgroup]], uint tile [[threadgroup_position_in_grid]]) {
    ar30_test_cooperative_states<true>(packed, out, k, lane, tile);
}
#endif
