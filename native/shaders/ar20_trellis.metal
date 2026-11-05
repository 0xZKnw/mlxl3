// Same packed windows as the canonical reader, eight states from three wrapped words.
inline ulong ar20_window8(const device uint* words, uint lane, uint k) {
    uint packed = 8u * k;
    uint end = (lane * 8u + 8u) * k + 256u * k;
    uint last = (end - 1u) / 32u;
    uint shift = (last + 1u) * 32u - end;
    uint a = words[last % packed];
    uint b = words[(last - 1u) % packed];
    uint c = words[(last - 2u) % packed];
    uint low = uint(((ulong(b) << 32u) | ulong(a)) >> shift);
    uint high = uint(((ulong(c) << 32u) | ulong(b)) >> shift);
    return (ulong(high) << 32u) | ulong(low);
}
inline void ar20_read8(ulong bits, uint k, thread uint (&cw)[8]) {
    for (uint j = 0u; j < 8u; ++j) cw[j] = uint(bits >> ((7u - j) * k)) & 0xffffu;
}
