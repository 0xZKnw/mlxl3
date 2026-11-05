// Each output word owns its bits; SWAP16 also crosses spans when K is odd.
inline ushort mlxl3_pack_word(const device ushort* encoded, uint k, uint output_word) {
    uint word = output_word ^ 1u;
    uint tile = word / (16u * k);
    uint span = (word % (16u * k)) / k;
    uint begin = (word % k) * 16u;
    uint result = 0u;
    for (uint i = begin / k; i <= (begin + 15u) / k; ++i) {
        uint value = uint(encoded[tile * 256u + span * 16u + i]) & ((1u << k) - 1u);
        uint end = (i + 1u) * k;
        if (end <= begin + 16u) result |= value << (begin + 16u - end);
        else result |= value >> (end - begin - 16u);
    }
    return ushort(result);
}
