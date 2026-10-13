#ifdef __CPROVER
typedef unsigned int uint;
typedef unsigned short ushort;
extern "C" void __CPROVER_assert(bool, const char*);
#define assert(condition) __CPROVER_assert(condition, #condition)
#else
#include <cassert>
#include <cstdint>
#include <cstdio>
using uint = std::uint32_t;
using ushort = std::uint16_t;
#endif
// Same Metal body; this host view does not model GPU address spaces/scheduling.
#define device
#include "../src/mlxl3_quantizer/pack_trellis.metal"
#undef device

static ushort oracle(const ushort* encoded, uint k, uint output_word) {
    uint word = output_word ^ 1u;
    uint span = word / k;
    uint start = (word % k) * 16u;
    uint result = 0;
    for (uint bit = 0; bit < 16; ++bit) {
        uint position = start + bit;
        uint symbol = uint(encoded[span * 16u + position / k]);
        result = (result << 1u) | ((symbol >> (k - 1u - position % k)) & 1u);
    }
    return ushort(result);
}

#ifdef __CPROVER
extern uint nondet_uint();
extern ushort nondet_ushort();
void proof() {
    ushort encoded[512];
    for (uint i = 0; i < 512; ++i) encoded[i] = nondet_ushort();
    uint k = nondet_uint(), word = nondet_uint();
    __CPROVER_assume(k >= 1u && k <= 8u);
    __CPROVER_assume(word < 32u * k);
#ifdef MLXL3_COVER
    if (k == 1u) assert(false);
    if (k == 4u) assert(false);
    if (k == 8u) assert(false);
    if (word == 0u) assert(false);
    if (word == 32u * k - 1u) assert(false);
    if (word >= 16u * k) assert(false);
#else
    assert(mlxl3_pack_word(encoded, k, word) == oracle(encoded, k, word));
#endif
}
#else
int main() {
    ushort encoded[512] = {};
    for (uint k = 1; k <= 8; ++k) {
        for (uint position = 0; position < 512; ++position) {
            for (uint bit = 0; bit < 16; ++bit) {
                encoded[position] = ushort(1u << bit);
                for (uint word = 0; word < 32u * k; ++word)
                    assert(mlxl3_pack_word(encoded, k, word) == oracle(encoded, k, word));
            }
            encoded[position] = 0;
        }
        for (uint i = 0; i < 512; ++i) encoded[i] = ushort(i * 641u + 319u);
        for (uint word = 0; word < 32u * k; ++word)
            assert(mlxl3_pack_word(encoded, k, word) == oracle(encoded, k, word));
        for (auto& value : encoded) value = 0;
    }
    std::puts("Packing: K=1..8, 2 tiles, every input bit/word; host sanitizer checks passed");
}
#endif
