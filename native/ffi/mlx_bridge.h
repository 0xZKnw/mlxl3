// Private ABI for the experimental LFM capacity cache.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#define MLXL3_NOEXCEPT noexcept
#else
#define MLXL3_NOEXCEPT
#endif

int mlxl3_array_update_slice(void* storage, void* update, int32_t axis,
                            int32_t start, int32_t end) MLXL3_NOEXCEPT;
// Diagnostic only: evaluates, but does not add an owner to MLX storage.
int mlxl3_array_storage_identity(void* array, uint64_t* identity) MLXL3_NOEXCEPT;

#ifdef __cplusplus
}
#endif
#undef MLXL3_NOEXCEPT
