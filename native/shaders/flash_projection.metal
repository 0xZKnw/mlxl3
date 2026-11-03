// Canonical Flash row-then-group QMV with dimensions bound at submission.
#include "flash_reference_reduction.h"
#define MLXL3_QMV_NT 1u
#define MLXL3_QMV_MB 1u
#define MLXL3_QMV_SG 4u
#define MLXL3_FUSE_OUTPUT 0
#define MLXL3_MATRIX_ROWS p.count
#define MLXL3_K_BITS p.k
#define K int(p.k)
#define CB p.cb
#define PACKED_U32 (8u * p.k)
#define TILES_K (p.rows / 16u)
#define TILES_N (p.cols / 16u)
#define INPUT_DIMS p.rows
#define OUTPUT_DIMS p.cols
#define N_SPLITS grid_size.z
#define mlxl3_decode_codeword(x, mode) codeword(x, mode)

kernel void flash_reference_qmv(const device half* xhat [[buffer(0)]],
    const device uint* trellis [[buffer(1)]], const device half* svh [[buffer(2)]],
    device float* yhat [[buffer(3)]], constant Params& p [[buffer(4)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],
    uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],
    uint3 grid_size [[threadgroups_per_grid]]) {
#include "_qmv_tile_kernel.metal"
}
