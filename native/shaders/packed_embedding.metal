// Reconstruct exact F16 bits, including subnormals, signed zero and NaNs.
uint index = thread_position_in_grid.x;
if (index >= OUTPUT_COUNT) return;
uint row = uint(ids[index / WIDTH]);
if (row >= VOCAB) { output[index] = half(0); return; }
ulong element = ulong(row) * WIDTH + index % WIDTH;
ulong bit = element * 13;
uint shift = uint(bit % 32);
uint high = main[bit / 32] >> shift;
if (shift > 19) high |= main[bit / 32 + 1] << (32 - shift);
uint low = 0;
uint base = offsets[element / 128];
if (base != 0xffffffffu) {
    uint low_bit = uint(element % 128) * 3;
    uint low_shift = low_bit % 32;
    low = tail[base + low_bit / 32] >> low_shift;
    if (low_shift > 29) low |= tail[base + low_bit / 32 + 1] << (32 - low_shift);
}
output[index] = as_type<half>(ushort(((high & 8191u) << 3) | (low & 7u)));
