// Derived from IncoAI Splash's Apache-2.0 DFlash selector.
#include <metal_stdlib>
using namespace metal;

inline bool dflash_top_beats(float value, uint token, float other,
                             uint other_token) {
  return value > other || (value == other && token < other_token);
}

inline void dflash_top16_insert(thread float (&values)[16],
                                thread uint (&ids)[16], float value,
                                uint token) {
#pragma clang loop unroll(full)
  for (uint slot = 15; slot > 0; --slot) {
    bool here = dflash_top_beats(value, token, values[slot], ids[slot]);
    bool above =
        dflash_top_beats(value, token, values[slot - 1], ids[slot - 1]);
    values[slot] = here ? (above ? values[slot - 1] : value) : values[slot];
    ids[slot] = here ? (above ? ids[slot - 1] : token) : ids[slot];
  }
  bool top = dflash_top_beats(value, token, values[0], ids[0]);
  values[0] = top ? value : values[0];
  ids[0] = top ? token : ids[0];
}

template <uint Count>
inline void dflash_top_pop(thread float (&values)[Count],
                           thread uint (&ids)[Count]) {
#pragma clang loop unroll(full)
  for (uint slot = 0; slot + 1 < Count; ++slot) {
    values[slot] = values[slot + 1];
    ids[slot] = ids[slot + 1];
  }
  values[Count - 1] = -INFINITY;
  ids[Count - 1] = 0xffffffffu;
}

inline void dflash_simd_best(float value, uint token, thread float &best,
                             thread uint &best_token) {
  best = simd_max(value);
  best_token = simd_min(value == best ? token : 0xffffffffu);
}

inline void dflash_merge_shards(device const uint *partial_ids,
                                device const float *partial_values, uint row,
                                uint lane, thread float &value,
                                thread uint &token) {
  constexpr uint K = 16, Shards = 8, Entries = Shards * K / 32;
  uint origin = row * Shards * K + lane * Entries;
  float values[Entries];
  uint ids[Entries];
  for (uint i = 0; i < Entries; ++i) {
    values[i] = partial_values[origin + i];
    ids[i] = partial_ids[origin + i];
  }
  value = -INFINITY;
  token = 0xffffffffu;
  for (uint rank = 0; rank < K; ++rank) {
    float best;
    uint best_id;
    dflash_simd_best(values[0], ids[0], best, best_id);
    if (lane == rank) {
      value = best;
      token = best_id;
    }
    if (values[0] == best && ids[0] == best_id)
      dflash_top_pop<Entries>(values, ids);
  }
}
