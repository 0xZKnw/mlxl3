#ifndef MLXL3_LOSSLESS_EMBEDDING_H
#define MLXL3_LOSSLESS_EMBEDDING_H
#include <stdint.h>

// Called only with min <= max, index < 128, width <= 8, and delta < 2^width.
// The byte stream has exactly 16*width exponent bytes, initially zeroed.
static inline uint8_t mlxl3_exponent_width(uint8_t min, uint8_t max) {
  uint32_t span = (uint32_t)max - min;
  uint8_t width = 0;
  while (span) { ++width; span >>= 1; }
  return width;
}
static inline void mlxl3_pack_exponent(uint8_t* stream, uint32_t index,
                                       uint8_t width, uint8_t delta) {
  if (!width) return;
  uint32_t bit = index * width, shift = bit % 8;
  stream[bit / 8] |= (uint8_t)((uint32_t)delta << shift);
  if (shift + width > 8)
    stream[bit / 8 + 1] |= (uint8_t)((uint32_t)delta >> (8 - shift));
}
static inline uint16_t mlxl3_unpack_bf16(const uint8_t* block, uint32_t index,
                                        uint8_t base, uint8_t width) {
  uint32_t delta = 0;
  if (width) {
    uint32_t bit = index * width, shift = bit % 8;
    delta = block[128 + bit / 8];
    if (shift + width > 8)
      delta |= (uint32_t)block[128 + bit / 8 + 1] << 8;
    delta = (delta >> shift) & ((1u << width) - 1);
  }
  uint32_t sign_mantissa = block[index];
  return (uint16_t)((sign_mantissa & 127) | ((sign_mantissa & 128) << 8)
                    | ((base + delta) << 7));
}
#endif
