#pragma once
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace mlxl3 {
enum class DiskSchedule { Static, Dynamic };

inline DiskSchedule disk_schedule(const char* value) {
  if (!value || std::strcmp(value, "0") == 0) return DiskSchedule::Static;
  if (std::strcmp(value, "1") == 0) return DiskSchedule::Dynamic;
  throw std::invalid_argument("MLXL3_SSD_DYNAMIC_READERS must be 0 or 1");
}

// Only job assignment changes. Completion, errors and publication remain with
// read_disk_records' synchronous join. Relaxed atomics allocate unique indices;
// no record contents or slot leases are published through this counter.
class DiskWorkQueue {
 public:
  DiskWorkQueue(size_t jobs, size_t workers, DiskSchedule schedule)
      : count_(jobs), workers_(workers), schedule_(schedule) {
    if (!jobs || jobs > 512 || !workers || workers > 4 || workers > jobs)
      throw std::invalid_argument("invalid disk work queue");
  }

  // Static cursor starts at worker id. count_ is a stable exhaustion sentinel.
  size_t claim(size_t& cursor) {
    if (schedule_ == DiskSchedule::Static) {
      const auto job = std::min(cursor, count_);
      cursor = job < count_ ? std::min(job + workers_, count_) : count_;
      return job;
    }
    auto job = next_.load(std::memory_order_relaxed);
    while (job < count_ && !next_.compare_exchange_weak(
        job, job + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {}
    return job;
  }

 private:
  const size_t count_, workers_;
  const DiskSchedule schedule_;
  std::atomic<size_t> next_{0};
};
}  // namespace mlxl3
