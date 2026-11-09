// Only K/codebook become constants. Dimensions/splits and the canonical body,
// including every accumulation and half boundary, are unchanged.
#undef CB
#define CB 2u
#undef K
#define K 2
#undef MLXL3_K_BITS
#define MLXL3_K_BITS 2u
#undef PACKED_U32
#define PACKED_U32 16u
kernel void flash_static_k2_nt1(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
kernel void flash_verify_static_k2_nt1(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 3u
kernel void flash_verify_static_k2_nt1_mb3(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 1u

#undef K
#define K 3
#undef MLXL3_K_BITS
#define MLXL3_K_BITS 3u
#undef PACKED_U32
#define PACKED_U32 24u
kernel void flash_static_k3_nt1(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
kernel void flash_verify_static_k3_nt1(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 3u
kernel void flash_verify_static_k3_nt1_mb3(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 1u
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void flash_static_k3_nt2(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
kernel void flash_verify_static_k3_nt2(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 3u
kernel void flash_verify_static_k3_nt2_mb3(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 1u

#undef K
#define K 4
#undef MLXL3_K_BITS
#define MLXL3_K_BITS 4u
#undef PACKED_U32
#define PACKED_U32 32u
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 1u
kernel void flash_static_k4_nt1(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
kernel void flash_verify_static_k4_nt1(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 3u
kernel void flash_verify_static_k4_nt1_mb3(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 1u
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void flash_static_k4_nt2(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
kernel void flash_verify_static_k4_nt2(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 3u
kernel void flash_verify_static_k4_nt2_mb3(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 1u

// Fixed K2 shapes retain the exact canonical body and runtime matrix row count.
#undef K
#define K 2
#undef MLXL3_K_BITS
#define MLXL3_K_BITS 2u
#undef PACKED_U32
#define PACKED_U32 16u
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 1u
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
kernel void flash_fixed_k2_2560_640(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
kernel void flash_verify_fixed_k2_2560_640(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 3u
kernel void flash_verify_fixed_k2_2560_640_mb3(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 1u
#define MLXL3_INDEX_KEY_TILES 1
kernel void flash_index_key_qmv(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_INDEX_KEY_TILES

#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void flash_fixed_k2_nt2_2560_640(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 1u
#undef INPUT_DIMS
#define INPUT_DIMS 640u
#undef OUTPUT_DIMS
#define OUTPUT_DIMS 2560u
#undef TILES_K
#define TILES_K 40u
#undef TILES_N
#define TILES_N 160u
kernel void flash_fixed_k2_640_2560(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
kernel void flash_verify_fixed_k2_640_2560(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 3u
kernel void flash_verify_fixed_k2_640_2560_mb3(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 2u
#undef MLXL3_QMV_MB
#define MLXL3_QMV_MB 1u
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void flash_fixed_k2_nt2_640_2560(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 1u
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

// Restore every runtime binding before fc_hidden and subsequent includes.
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
