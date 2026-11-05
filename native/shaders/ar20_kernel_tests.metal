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

#if MLXL3_AR20_TESTS
kernel void ar20_reader8_states(const device ushort* packed [[buffer(0)]], device ushort* out [[buffer(1)]],
    constant uint& k [[buffer(2)]], uint lane [[thread_index_in_threadgroup]], uint tile [[threadgroup_position_in_grid]]) {
    uint cw[8];
    if (k <= 6u) ar20_read8(ar20_window8(reinterpret_cast<const device uint*>(packed) + tile * 8u * k, lane, k), k, cw);
    else for (uint j=0u; j<8u; ++j) cw[j] = state(packed, tile, lane*8u+j, k);
    for (uint j=0u; j<8u; ++j) out[tile * 256u + lane*8u+j] = ushort(cw[j]);
}
kernel void ar20_reader8_omit_high_word(const device ushort* packed [[buffer(0)]], device ushort* out [[buffer(1)]],
    constant uint& k [[buffer(2)]], uint lane [[thread_index_in_threadgroup]], uint tile [[threadgroup_position_in_grid]]) {
    const device uint* words = reinterpret_cast<const device uint*>(packed) + tile * 8u * k;
    uint end = (lane*8u+8u)*k+256u*k, last=(end-1u)/32u, shift=(last+1u)*32u-end;
    uint a=words[last%(8u*k)], b=words[(last-1u)%(8u*k)];
    uint low=uint(((ulong(b)<<32u)|ulong(a))>>shift), high=uint(ulong(b)>>shift);
    uint cw[8]; ar20_read8((ulong(high)<<32u)|ulong(low), k, cw);
    for(uint j=0u;j<8u;++j) out[tile*256u+lane*8u+j]=ushort(cw[j]);
}
#endif

#if MLXL3_AR20_TESTS
kernel void ar20_h128_compare(const device float* input [[buffer(0)]], device float* original [[buffer(1)]],
    device float* actual [[buffer(2)]], device float* mutant [[buffer(3)]],
    uint lane [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    threadgroup float old[128], now[128];
    for(uint i=lane;i<128u;i+=64u) { old[i]=input[block*128u+i]; now[i]=old[i]; }
    hadamard128_canonical(old,lane);
    hadamard128(now,lane);
    for(uint i=lane;i<128u;i+=64u) { original[block*128u+i]=old[i]; actual[block*128u+i]=now[i]; }
    if(lane<32u) {
        float v[4];for(uint j=0u;j<4u;++j)v[j]=input[block*128u+lane+32u*j];
        ar20_h128_register(v,lane,false);
        for(uint j=0u;j<4u;++j)mutant[block*128u+lane+32u*j]=v[j];
    }
}
kernel void ar20_h128_routes(const device float* input [[buffer(0)]], device float* original [[buffer(1)]],
    device float* actual [[buffer(2)]],uint tid [[thread_index_in_threadgroup]],uint block [[threadgroup_position_in_grid]]) {
    uint route=tid/64u, lane=tid%64u;
    threadgroup float old[10][128],now[10][128];
    uint base=block*1280u+route*128u;
    for(uint i=lane;i<128u;i+=64u){old[route][i]=input[base+i];now[route][i]=old[route][i];}
    hadamard128_canonical(old[route],lane);
    hadamard128(now[route],lane);
    for(uint i=lane;i<128u;i+=64u){original[base+i]=old[route][i];actual[base+i]=now[route][i];}
}
#endif
