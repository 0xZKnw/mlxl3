uint position = thread_position_in_grid.x;
if (position >= OUTPUT_COUNT) return;
uint token = uint(ids[position / WIDTH]);
// Production callers validate vocabulary bounds; still avoid device OOB on misuse.
if (token >= VOCAB) { output[position] = ushort(0); return; }
ulong element = ulong(token) * WIDTH + position % WIDTH;
uint block = uint(element / 128), index = uint(element % 128);
uint start = metadata[2 * block], descriptor = metadata[2 * block + 1];
uint base = descriptor & 255, width = (descriptor >> 8) & 15;
uint delta = 0;
if (width) {
    uint bit = index * width, shift = bit % 8;
    delta = payload[start + 128 + bit / 8];
    if (shift + width > 8)
        delta |= uint(payload[start + 128 + bit / 8 + 1]) << 8;
    delta = (delta >> shift) & ((1u << width) - 1);
}
uint sign_mantissa = payload[start + index];
output[position] = ushort((sign_mantissa & 127) | ((sign_mantissa & 128) << 8)
                          | ((base + delta) << 7));
