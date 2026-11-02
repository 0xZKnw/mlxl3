#include "disk_work_queue.h"

#include <cassert>

extern "C" size_t nondet_size_t();
extern "C" bool nondet_bool();
extern "C" void __CPROVER_assume(bool);
extern "C" void __CPROVER_cover(bool);

// Productive claims in symbolic worker order on the actual queue, <=8 jobs/4 workers.
// The static scheduling oracle is independent of the implementation's cursors.
// No atomic stubs: this does not prove concurrent CAS, GCD, I/O or publication.
void proof() {
  using namespace mlxl3;
  const auto count = nondet_size_t();
  const auto workers = nondet_size_t();
  const auto requested = nondet_bool() ? DiskSchedule::Dynamic : DiskSchedule::Static;
  __CPROVER_assume(count >= 1 && count <= 8 && workers >= 1 && workers <= 4 && workers <= count);
  const auto mode = effective_disk_schedule(requested, count, workers);
  assert((mode == DiskSchedule::Dynamic) ==
         (requested == DiskSchedule::Dynamic && workers > 1 && count > workers));
  DiskWorkQueue queue(count, workers, mode);
  bool seen[8] = {};
  size_t cursors[4] = {0, 1, 2, 3};
  size_t expected[4] = {0, 1, 2, 3};
  for (size_t i = 0; i < count; ++i) {
    const auto worker = nondet_size_t();
    __CPROVER_assume(worker < workers);
    if (mode == DiskSchedule::Static) __CPROVER_assume(expected[worker] < count);
    const auto job = queue.claim(cursors[worker]);
    assert(job < count && !seen[job]);
    if (mode == DiskSchedule::Static) {
      assert(job == expected[worker]);
      assert(job % workers == worker);
      expected[worker] += workers;
    } else {
      assert(job == i);
    }
    seen[job] = true;
  }
  for (size_t worker = 0; worker < workers; ++worker) {
    assert(queue.claim(cursors[worker]) == count);
    assert(queue.claim(cursors[worker]) == count);
  }
  for (size_t i = 0; i < count; ++i) assert(seen[i]);
  __CPROVER_cover(count == 8 && workers == 4 && mode == DiskSchedule::Dynamic);
  __CPROVER_cover(count == 8 && workers == 4 && mode == DiskSchedule::Static);
  __CPROVER_cover(count == 1 && workers == 1 && requested == DiskSchedule::Dynamic &&
                 mode == DiskSchedule::Static);
  __CPROVER_cover(count == workers && workers == 4 && requested == DiskSchedule::Dynamic &&
                 mode == DiskSchedule::Static);
}
