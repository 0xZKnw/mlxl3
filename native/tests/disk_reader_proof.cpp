#include "disk_reader.h"

extern "C" void __CPROVER_havoc_object(void*);
extern "C" void __CPROVER_assume(bool);
extern "C" void __CPROVER_assert(bool, const char*);
extern "C" void __CPROVER_cover(bool);

// Attempt the actual validator/dispatch path; no stubs for preadv or libdispatch.
void proof() {
  size_t start = 0, length = 0, workers = 0;
  __CPROVER_havoc_object(&start);
  __CPROVER_havoc_object(&length);
  __CPROVER_havoc_object(&workers);
  __CPROVER_assume(start <= 8 && length <= 8 && workers >= 1 && workers <= 4);
  uint8_t bank[16] = {};
  std::vector<mlxl3::DiskRead> reads{{-1, 0, {{bank, 8}}},
                                    {-1, 0, {{bank + start, length}}}};
  bool rejected = false;
  try { mlxl3::read_disk_records(reads, workers); }
  catch (const std::invalid_argument&) { rejected = true; }
  catch (const std::runtime_error&) {}
  __CPROVER_assert(rejected == (start < 8 || length == 0), "reject actual overlapping destinations");
  for (auto byte : bank) __CPROVER_assert(byte == 0, "invalid sources do not write");
  __CPROVER_cover(rejected);
  __CPROVER_cover(!rejected);
}
