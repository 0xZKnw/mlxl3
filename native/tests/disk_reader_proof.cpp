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

void copy_proof() {
  size_t start = 0, length = 0;
  __CPROVER_havoc_object(&start);
  __CPROVER_havoc_object(&length);
  __CPROVER_assume(start <= 8 && length <= 8);
  uint8_t source[8], bank[16] = {};
  __CPROVER_havoc_object(&source);
  const auto before = std::vector<uint8_t>(bank, bank + 16);
  bool rejected = false;
  try { mlxl3::copy_disk_record(source, 8, {{bank + start, length}, {bank + 8, 8 - length}}); }
  catch (const std::invalid_argument&) { rejected = true; }
  const bool valid = length > 0 && length < 8 && start + length <= 8;
  __CPROVER_assert(rejected != valid, "copy rejects empty/incomplete segments atomically");
  if (valid) {
    for (size_t i = 0; i < length; ++i) __CPROVER_assert(bank[start + i] == source[i], "first copied segment exact");
    for (size_t i = 0; i < 8 - length; ++i) __CPROVER_assert(bank[8 + i] == source[length + i], "second copied segment exact");
  } else {
    for (size_t i = 0; i < 16; ++i) __CPROVER_assert(bank[i] == before[i], "rejection leaves bank unchanged");
  }
  __CPROVER_cover(valid); __CPROVER_cover(!valid);
}

// Validate the exact prepare-time path before any worker may receive its token.
void persistent_prepare_proof() {
  size_t start = 0, length = 0;
  __CPROVER_havoc_object(&start); __CPROVER_havoc_object(&length);
  __CPROVER_assume(start <= 8 && length <= 8);
  uint8_t bank[16] = {};
  std::vector<mlxl3::DiskRead> reads{{-1, 0, {{bank, 8}, {bank + start, length}}}};
  bool rejected = false;
  try { mlxl3::validate_disk_reads(reads); }
  catch (const std::invalid_argument&) { rejected = true; }
  __CPROVER_assert(rejected == (start < 8 || length == 0), "pending destinations are disjoint");
  for (auto byte : bank) __CPROVER_assert(byte == 0, "preparation performs no writes");
  __CPROVER_assert(reads[0].bytes_read == 0, "preparation has no claimed I/O");
  __CPROVER_cover(rejected); __CPROVER_cover(!rejected);
}
