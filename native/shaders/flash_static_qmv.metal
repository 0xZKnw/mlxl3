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

#undef K
#define K 3
#undef MLXL3_K_BITS
#define MLXL3_K_BITS 3u
#undef PACKED_U32
#define PACKED_U32 24u
kernel void flash_static_k3_nt1(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void flash_static_k3_nt2(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

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
#undef MLXL3_QMV_NT
#define MLXL3_QMV_NT 2u
kernel void flash_static_k4_nt2(FLASH_QMV_ARGUMENTS) {
#include "_qmv_tile_kernel.metal"
}

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
