#if MLXL3_AR20_TESTS
kernel void ar33_compiled_codeword_table(device uint* out [[buffer(0)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid == 0u) out[0] = MLXL3_AR33_CODEWORD_TABLE;
}
kernel void ar33_codeword_states(device half* actual [[buffer(0)]],
    device half* canonical [[buffer(1)]], device half* wrong [[buffer(2)]],
    constant uint4& p [[buffer(3)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= p.z) return;
    uint word = tid | p.y;
    actual[tid] = codeword(word, p.x);
    canonical[tid] = codeword_canonical(word, p.x);
#if MLXL3_AR33_CODEWORD_TABLE
    uint mixed = (word & 0xffffu) * 0x83DCD12Du;
    uint sum = (mixed & 255u) + ((mixed >> 8u) & 255u)
        + ((mixed >> 16u) & 255u) + ((mixed >> 24u) & 255u);
    wrong[tid] = as_type<half>(ar33_mul1_table[(sum + 1u) % 1021u]);
#else
    wrong[tid] = codeword_canonical(word, p.x);
#endif
}
kernel void ar33_codeword_entries(device half* actual [[buffer(0)]],
    device half* canonical [[buffer(1)]], constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= count) return;
#if MLXL3_AR33_CODEWORD_TABLE
    actual[tid] = as_type<half>(ar33_mul1_table[tid]);
#else
    actual[tid] = half(0);
#endif
    canonical[tid] = fma(as_type<half>(ushort(0x6400u + tid)),
        as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931)));
}
#endif
