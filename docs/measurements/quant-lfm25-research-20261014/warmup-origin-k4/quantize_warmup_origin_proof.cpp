// Host address-space abstraction of the shared Metal helper; GPU scheduling is outside this model.
#ifdef __CPROVER
typedef unsigned int uint;
typedef unsigned short ushort;
extern "C" void __CPROVER_assert(bool, const char*);
#define assert(condition) __CPROVER_assert(condition, #condition)
#else
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <random>
using uint = std::uint32_t;
using ushort = std::uint16_t;
#endif
#define threadgroup
#include "warmup_origin.metal"

#ifdef __CPROVER
extern uint nondet_uint();
#ifndef MLXL3_RECURRENCE
void proof() {
    uint parent = nondet_uint(), ri = nondet_uint(), ancestor = nondet_uint();
    __CPROVER_assume(parent < 4096u && ri < 128u && ancestor < 4096u);
    ushort previous[4096];
    previous[parent] = ushort(ancestor);
    const ushort* origin_ptr = 0;
    if (ri) origin_ptr = &previous[0];
    ushort actual = mlxl3_warmup_origin(parent, ri, origin_ptr);
    assert(actual < 4096u);
    assert(actual == (ri ? ancestor : parent));
#ifdef MLXL3_COVER
    if (ri == 0u && parent == 0u) assert(false);
    if (ri == 127u && parent == 4095u) assert(false);
    if (ri == 1u && ancestor == 0u) assert(false);
#endif
}
#else
void recurrence() {
    const uint states = 4, rows = 4;
    ushort parents[rows][states], origins[2][states];
    for (uint r = 0; r < rows; ++r) {
        for (uint e = 0; e < states; ++e) {
            uint parent = nondet_uint();
            __CPROVER_assume(parent < states);
            parents[r][e] = ushort(parent);
            const ushort* origin_ptr = 0;
            if (r) origin_ptr = &origins[(r - 1) % 2][0];
            origins[r % 2][e] = mlxl3_warmup_origin(parent, r, origin_ptr);
        }
    }
    uint edge = nondet_uint();
    __CPROVER_assume(edge < states);
    uint selected = edge;
    for (int r = int(rows) - 1; r >= 0; --r) selected = parents[r][selected];
    assert(origins[(rows - 1) % 2][edge] == selected);
}
#endif
#else
int main() {
    constexpr uint states = 4096, rows = 128;
    std::array<std::array<ushort, states>, rows> parents;
    std::array<std::array<ushort, states>, 2> origins;
    std::mt19937 random(139);
    for (uint variant = 0; variant < 4; ++variant) {
        for (uint r = 0; r < rows; ++r) {
            for (uint e = 0; e < states; ++e) {
                uint parent = variant == 0 ? random() % states
                    : variant == 1 ? 0u : variant == 2 ? states - 1 : e;
                parents[r][e] = ushort(parent);
                origins[r % 2][e] = mlxl3_warmup_origin(parent, r, r ? origins[(r - 1) % 2].data() : nullptr);
            }
        }
        for (uint edge = 0; edge < states; ++edge) {
            uint selected = edge;
            for (int r = int(rows) - 1; r >= 0; --r) selected = parents[r][selected];
            assert(origins[(rows - 1) % 2][edge] == selected);
        }
    }
    std::puts("K4 origins: four 128-step / 4096-state maps equal serial traceback; sanitizer checks passed");
}
#endif
