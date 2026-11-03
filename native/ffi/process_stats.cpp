#include <algorithm>
#include "process_memory.h"

extern "C" int mlxl3_read_process_stats(uint64_t* out, size_t count) noexcept {
  if (!out || count < 9) return -1;
  const auto sample = mlxl3::current_process_memory();
  std::copy(sample.begin(), sample.end(), out);
  return 0;
}
