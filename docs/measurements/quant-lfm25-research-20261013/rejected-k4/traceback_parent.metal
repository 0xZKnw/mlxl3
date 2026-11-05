// K4 stores the winning fresh nibble; 128 preserves the reference fallback.
inline uint mlxl3_k4_parent(uint edge, uint fresh) {
    return fresh == 128u ? 0u : (fresh << 8u) | (edge >> 4u);
}
