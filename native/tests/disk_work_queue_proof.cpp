#include "disk_work_queue.h"
#include <cassert>

extern "C" size_t nondet_size_t();
extern "C" void __CPROVER_assume(bool);
extern "C" void __CPROVER_cover(bool);

// Sequential interleavings of claims on the real production queue, up to 8 jobs.
// Not a proof of concurrent atomics, libdispatch, preadv or lease publication.
void proof() {
  const auto count = nondet_size_t();
  const auto workers = nondet_size_t();
  __CPROVER_assume(count >= 1 && count <= 8 && workers >= 1 && workers <= 4 && workers <= count);
  mlxl3::DiskWorkQueue queue(count, workers, mlxl3::DiskSchedule::Dynamic);
  bool seen[8] = {};
  size_t cursor = 0;
  for (size_t i = 0; i < count; ++i) {
    cursor = nondet_size_t();
    __CPROVER_assume(cursor < workers);
    const auto job = queue.claim(cursor);
    assert(job < count && !seen[job]);
    seen[job] = true;
  }
  assert(queue.claim(cursor) == count);
  for (size_t i = 0; i < count; ++i) assert(seen[i]);
  __CPROVER_cover(count == 8 && workers == 4);
  __CPROVER_cover(count == 1 && workers == 1);
}
