// Two real resident matrices, separate transformed inputs and partial outputs.
#undef K
#define K 4
#undef CB
#define CB 2u
#undef MLXL3_K_BITS
#define MLXL3_K_BITS 4u
#undef PACKED_U32
#define PACKED_U32 32u
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
#undef MLXL3_MATRIX_ROWS
#define MLXL3_MATRIX_ROWS 1u
#undef INPUT_DIMS
#define INPUT_DIMS 2560u
#undef OUTPUT_DIMS
#define OUTPUT_DIMS (bundle_z ? 6144u : 10240u)
#undef TILES_K
#define TILES_K 160u
#undef TILES_N
#define TILES_N (OUTPUT_DIMS / 16u)
#undef N_SPLITS
#define N_SPLITS 1u
kernel void ar20_gdn_resident_qkv_z_nt2(const device half* qkv_input [[buffer(0)]],const device uint* qkv_trellis [[buffer(1)]],
    const device half* z_input [[buffer(2)]],const device uint* z_trellis [[buffer(3)]],
    device float* qkv_partial [[buffer(4)]],device float* z_partial [[buffer(5)]],
    uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],uint thread_index_in_simdgroup [[thread_index_in_simdgroup]],
    uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],uint3 physical_group [[threadgroup_position_in_grid]]) {
    const bool bundle_z=physical_group.x>=320u;
    const device half* xhat=bundle_z?z_input:qkv_input;
    const device uint* trellis=bundle_z?z_trellis:qkv_trellis;
    const device half* svh=xhat;
    device float* yhat=bundle_z?z_partial:qkv_partial;
    const uint3 threadgroup_position_in_grid=uint3(physical_group.x-(bundle_z?320u:0u),0,0);
#include "_qmv_tile_kernel.metal"
}
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
#undef MLXL3_MATRIX_ROWS
#define MLXL3_MATRIX_ROWS p.count
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
