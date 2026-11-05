// One directly bound record per grid.z expert. The canonical inner QMV is unchanged.
struct ExpertAxisParams { Params p; uint trellis; uint svh; uint stride; };
#define EXPERT_AXIS_ARGS const device half* prepared [[buffer(0)]], \
    const device uchar* r0 [[buffer(1)]], const device uchar* r1 [[buffer(2)]], \
    const device uchar* r2 [[buffer(3)]], const device uchar* r3 [[buffer(4)]], \
    const device uchar* r4 [[buffer(5)]], const device uchar* r5 [[buffer(6)]], \
    const device uchar* r6 [[buffer(7)]], const device uchar* r7 [[buffer(8)]], \
    const device uchar* r8 [[buffer(9)]], const device uchar* r9 [[buffer(10)]], \
    device float* partial [[buffer(11)]], constant ExpertAxisParams& ep [[buffer(12)]], \
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]], \
    uint thread_index_in_simdgroup [[thread_index_in_simdgroup]], \
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]], \
    uint3 axis_group [[threadgroup_position_in_grid]], uint3 axis_grid [[threadgroups_per_grid]]
#define EXPERT_AXIS_LOCALS \
    const device uchar* records[10] = {r0,r1,r2,r3,r4,r5,r6,r7,r8,r9}; \
    const uint expert = axis_group.z; \
    const device half* xhat = prepared + expert * ep.stride; \
    const device uint* trellis = reinterpret_cast<const device uint*>(records[expert] + ep.trellis); \
    const device half* svh = reinterpret_cast<const device half*>(records[expert] + ep.svh); \
    device float* yhat = partial + expert * uint(OUTPUT_DIMS); \
    constant Params& p = ep.p; \
    uint3 threadgroup_position_in_grid = uint3(axis_group.x,0,0); \
    uint3 grid_size = uint3(axis_grid.x,1,1);

#undef K
#define K 2
#undef CB
#define CB 2u
#undef MLXL3_K_BITS
#define MLXL3_K_BITS 2u
#undef PACKED_U32
#define PACKED_U32 16u
#undef N_SPLITS
#define N_SPLITS 1u
#undef INPUT_DIMS
#define INPUT_DIMS 2560u
#undef OUTPUT_DIMS
#define OUTPUT_DIMS 640u
#undef TILES_K
#define TILES_K 160u
#undef TILES_N
#define TILES_N 40u
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 1u
kernel void flash_axis_k2_2560_640_nt1(EXPERT_AXIS_ARGS) {
    EXPERT_AXIS_LOCALS
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void flash_axis_k2_2560_640_nt2(EXPERT_AXIS_ARGS) {
    EXPERT_AXIS_LOCALS
#include "_qmv_tile_kernel.metal"
}
#undef INPUT_DIMS
#define INPUT_DIMS 640u
#undef OUTPUT_DIMS
#define OUTPUT_DIMS 2560u
#undef TILES_K
#define TILES_K 40u
#undef TILES_N
#define TILES_N 160u
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 1u
kernel void flash_axis_k2_640_2560_nt1(EXPERT_AXIS_ARGS) {
    EXPERT_AXIS_LOCALS
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void flash_axis_k2_640_2560_nt2(EXPERT_AXIS_ARGS) {
    EXPERT_AXIS_LOCALS
#include "_qmv_tile_kernel.metal"
}
#undef EXPERT_AXIS_ARGS
#undef EXPERT_AXIS_LOCALS
#undef INPUT_DIMS
#define INPUT_DIMS p.rows
#undef OUTPUT_DIMS
#define OUTPUT_DIMS p.cols
#undef TILES_K
#define TILES_K (p.rows / 16u)
#undef TILES_N
#define TILES_N (p.cols / 16u)
#undef N_SPLITS
#define N_SPLITS grid_size.z
#undef K
#define K int(p.k)
#undef CB
#define CB p.cb
#undef MLXL3_K_BITS
#define MLXL3_K_BITS p.k
#undef PACKED_U32
#define PACKED_U32 (8u * p.k)
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 1u
