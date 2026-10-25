#include <cstddef>
#include <cstdint>

extern "C" int mlxl3_array_uninitialized(const int32_t*, size_t, int, void**) noexcept;
extern "C" int mlxl3_array_metadata(void*, size_t*, const int32_t**, int*) noexcept;
extern "C" void mlxl3_array_free(void*);
extern "C" void __CPROVER_havoc_object(void*);
extern "C" void __CPROVER_assume(bool);
extern "C" void __CPROVER_assert(bool, const char*);
extern "C" void __CPROVER_cover(bool);

// Link the actual MLX bridge; allocation failures are permitted, GPU is not stubbed.
void proof() {
  int32_t dimension;
  __CPROVER_havoc_object(&dimension);
  __CPROVER_assume(dimension >= -1 && dimension <= 8);
  void* output = nullptr;
  const int status = mlxl3_array_uninitialized(&dimension, 1, 2, &output);
  if (status == 0) {
    size_t rank = 0;
    const int32_t* dims = nullptr;
    int type = -1;
    __CPROVER_assert(dimension >= 0 && output != nullptr, "valid owned allocation");
    __CPROVER_assert(mlxl3_array_metadata(output, &rank, &dims, &type) == 0,
                     "allocated metadata is accessible without reading storage");
    __CPROVER_assert(rank == 1 && dims[0] == dimension && type == 2,
                     "shape and dtype preserved");
    mlxl3_array_free(output);
  } else {
    __CPROVER_assert(output == nullptr, "failure publishes no handle");
  }
  __CPROVER_assert(mlxl3_array_uninitialized(&dimension, 1, 2, nullptr) != 0,
                   "null output rejected");
  __CPROVER_cover(dimension == -1);
  __CPROVER_cover(status == 0);
}
