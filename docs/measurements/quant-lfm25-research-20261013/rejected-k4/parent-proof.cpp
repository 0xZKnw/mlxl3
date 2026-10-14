#ifdef __CPROVER
typedef unsigned int uint;
extern "C" void __CPROVER_assert(bool, const char*);
#define assert(condition) __CPROVER_assert(condition, #condition)
#else
#include <cassert>
#include <cstdint>
#include <cstdio>
using uint = std::uint32_t;
#endif
#include "traceback_parent.metal"

#ifdef __CPROVER
extern uint nondet_uint();
void proof() {
    uint edge = nondet_uint(), fresh = nondet_uint();
    __CPROVER_assume(edge < 4096u);
    __CPROVER_assume(fresh < 16u || fresh == 128u);
#ifdef MLXL3_COVER
    if (fresh == 128u) assert(false);
    if (fresh == 0u && edge < 16u) assert(false);
    if (fresh == 15u && edge == 4095u) assert(false);
#else
    uint parent = mlxl3_k4_parent(edge, fresh);
    assert(parent < 4096u);
    assert(parent == (fresh == 128u ? 0u : ((fresh << 12u) | edge) >> 4u));
#endif
}
#else
int main() {
    for (uint edge = 0; edge < 4096; ++edge) {
        assert(mlxl3_k4_parent(edge, 128) == 0);
        for (uint fresh = 0; fresh < 16; ++fresh) {
            uint parent = mlxl3_k4_parent(edge, fresh);
            assert(parent < 4096);
            assert(parent == ((fresh << 12u) | edge) >> 4u);
        }
    }
    std::puts("K4 parents: all 4096 edges x 17 labels; sanitizer checks passed");
}
#endif
