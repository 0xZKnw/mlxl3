// The first warmup row anchors the path; later rows inherit through its chosen parent.
inline ushort mlxl3_warmup_origin(uint parent, uint ri, threadgroup const ushort* previous) {
    return ri == 0u ? ushort(parent) : previous[parent];
}
