#if MLXL3_AR20_TESTS
kernel void ar20_compiled_bits(device uint* out [[buffer(0)]], uint tid [[thread_position_in_grid]]) {
    if (tid == 0u) out[0] = MLXL3_AR20_SWAR | (MLXL3_AR20_READER8 << 1) | (MLXL3_AR20_PRELOAD << 2) | (MLXL3_AR20_H128 << 3);
}
kernel void ar20_codebook_original(device half* out [[buffer(0)]], uint cw [[thread_position_in_grid]]) {
    if (cw < 65536u) out[cw] = codeword_canonical(cw, 2u);
}
kernel void ar20_codebook_omit_high_byte(device half* out [[buffer(0)]], uint cw [[thread_position_in_grid]]) {
    if (cw >= 65536u) return;
    uint mixed = cw * 0x83DCD12Du;
    uint sum = 0x6400u + (mixed & 255u) + ((mixed >> 8u) & 255u) + ((mixed >> 16u) & 255u);
    out[cw] = fma(as_type<half>(ushort(sum)), as_type<half>(ushort(0x1eee)), as_type<half>(ushort(0xc931)));
}
#endif
