// Standalone source checks of the exact encoder helpers used by mlx_bridge.cpp.
#include "../ffi/lossless_embedding.h"
#include <assert.h>

#ifdef MLXL3_CBMC
extern uint8_t nondet_u8();
extern uint32_t nondet_u32();
extern "C" void lossless_embedding_width_proof() {
  uint8_t low = nondet_u8(), high = nondet_u8();
  __CPROVER_assume(low <= high);
  uint8_t width = mlxl3_exponent_width(low, high);
  uint32_t span = (uint32_t)high - low;
  assert(width <= 8);
  assert(span < (1u << width));
  if (width) assert(span >= (1u << (width - 1)));
}
extern "C" void lossless_embedding_codec_proof() {
  uint8_t width = nondet_u8(), base = nondet_u8(), delta = nondet_u8();
  uint32_t index = nondet_u32();
  uint8_t sign_mantissa = nondet_u8();
  // These are guaranteed by the real block min/max scan and 128-value loop.
  __CPROVER_assume(width <= 8 && index < 128);
  __CPROVER_assume((uint32_t)delta < (1u << width));
  __CPROVER_assume((uint32_t)base + delta <= 255);
  uint8_t bytes[256] = {};
  bytes[index] = sign_mantissa;
  mlxl3_pack_exponent(bytes + 128, index, width, delta);
  uint16_t expected = (uint16_t)((sign_mantissa & 127) +
      ((uint32_t)(sign_mantissa >> 7) << 15) + ((uint32_t)(base + delta) << 7));
  assert(mlxl3_unpack_bf16(bytes, index, base, width) == expected);
  // Separate --cover branch run checks these edge cases are reachable.
  if (width == 0) assert(delta == 0);
  if (width == 8 && index == 127) assert(bytes[255] == delta);
  if (width == 7 && index == 1) assert((bytes[129] & 63) == (delta >> 1));
}
#else
#include <stdio.h>
int main() {
  unsigned long checked = 0;
  for (uint32_t low = 0; low < 256; ++low) {
    for (uint32_t high = low; high < 256; ++high) {
      uint8_t width = mlxl3_exponent_width((uint8_t)low, (uint8_t)high);
      assert(width <= 8 && high - low < (1u << width));
      if (width) assert(high - low >= (1u << (width - 1)));
    }
  }
  for (uint32_t width = 0; width <= 8; ++width) {
    uint32_t span = (1u << width) - 1;
    assert(mlxl3_exponent_width(0, (uint8_t)span) == width);
    for (uint32_t value = 0; value < 65536; ++value) {
      uint8_t exponent = (uint8_t)((value >> 7) & 255);
      uint8_t base = exponent > span ? (uint8_t)(exponent - span) : 0;
      uint8_t delta = (uint8_t)(exponent - base);
      uint32_t index = (value * 37) % 128;
      uint8_t bytes[258] = {};
      bytes[0] = 0x5a; bytes[257] = 0xa5;
      uint8_t* block = bytes + 1;
      block[index] = (uint8_t)((value & 127) | ((value >> 8) & 128));
      mlxl3_pack_exponent(block + 128, index, (uint8_t)width, delta);
      assert(mlxl3_unpack_bf16(block, index, base, (uint8_t)width) == value);
      assert(bytes[0] == 0x5a && bytes[257] == 0xa5);
      ++checked;
    }
    uint8_t block[256] = {};
    for (uint32_t i = 0; i < 128; ++i) {
      block[i] = (uint8_t)(i * 73);
      mlxl3_pack_exponent(block + 128, i, (uint8_t)width, (uint8_t)((i * 37) & span));
    }
    for (uint32_t i = 0; i < 128; ++i) {
      uint16_t expected = (uint16_t)((block[i] & 127) | ((uint32_t)(block[i] & 128) << 8)
                                    | (((i * 37) & span) << 7));
      assert(mlxl3_unpack_bf16(block, i, 0, (uint8_t)width) == expected);
      ++checked;
    }
  }
  printf("%lu codec roundtrips passed; 9 widths, all 65536 BF16 patterns\n", checked);
}
#endif
