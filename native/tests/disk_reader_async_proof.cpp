#include "disk_read_batch.h"

extern "C" void __CPROVER_havoc_object(void*);
extern "C" void __CPROVER_assume(bool);
extern "C" void __CPROVER_assert(bool, const char*);
extern "C" void __CPROVER_cover(bool);

// Actual constructor/validator/worker API, without stubbing GCD, atomics or preadv.
void async_prepare_proof() {
  size_t start = 0;
  bool dynamic = false;
  __CPROVER_havoc_object(&start);
  __CPROVER_havoc_object(&dynamic);
  __CPROVER_assume(start <= 8);
  uint8_t bank[24] = {};
  bool rejected = false;
  try {
    mlxl3::DiskReadBatch batch(
        {{-1, 0, {{bank, 8}}}, {-1, 0, {{bank + start, 8}}}, {-1, 0, {{bank + 16, 8}}}},
        2, false, SIZE_MAX, dynamic ? mlxl3::DiskSchedule::Dynamic : mlxl3::DiskSchedule::Static);
    try { batch.finish(); } catch (const std::runtime_error&) {}
  } catch (const std::invalid_argument&) { rejected = true; }
  __CPROVER_assert(rejected == (start < 8), "actual async validator rejects cross-job overlap");
  for (auto byte : bank) __CPROVER_assert(byte == 0, "invalid file descriptors never write bytes");
  __CPROVER_cover(rejected); __CPROVER_cover(!rejected);
  __CPROVER_cover(dynamic && !rejected); __CPROVER_cover(!dynamic && !rejected);
}
