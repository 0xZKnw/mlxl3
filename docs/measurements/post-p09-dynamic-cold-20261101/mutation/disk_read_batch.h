#pragma once

#include "disk_reader.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>

namespace mlxl3 {
// OS-only workers. The caller owns destinations/files through join and destroys this on its owner thread.
class DiskReadBatch {
 public:
  explicit DiskReadBatch(std::vector<DiskRead> reads, size_t workers, bool profile = false,
                         size_t held_job = std::numeric_limits<size_t>::max(),
                         DiskSchedule schedule = DiskSchedule::Static)
      : reads_(std::move(reads)),
        schedule_(effective_disk_schedule(schedule, reads_.size(), workers)),
        jobs_(reads_.size(), std::min(workers, reads_.size()), DiskSchedule::Static),
        errors_(reads_.size()),
        states_(std::make_unique<std::atomic<uint8_t>[]>(reads_.size())),
        held_job_(held_job), tail_released_(held_job == std::numeric_limits<size_t>::max()) {
    if (!workers || workers > 4) throw std::invalid_argument("invalid disk worker count");
    validate_disk_reads(reads_);
    if (!tail_released_ && held_job >= reads_.size())
      throw std::invalid_argument("invalid held disk job");
    workers_ = std::min(workers, reads_.size());
    for (size_t i = 0; i < reads_.size(); ++i) states_[i].store(0, std::memory_order_relaxed);
    if (profile) timings_.resize(reads_.size());
    epoch_ = std::chrono::steady_clock::now();
    group_ = dispatch_group_create();
    if (!group_) throw std::bad_alloc();
    // All allocations and worker contexts precede the first callback.
    for (size_t i = 0; i < workers_; ++i) contexts_[i] = {this, i};
    for (size_t i = 0; i < workers_; ++i)
      dispatch_group_async_f(group_, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
                             &contexts_[i], &run_worker);
  }
  DiskReadBatch(const DiskReadBatch&) = delete;
  DiskReadBatch& operator=(const DiskReadBatch&) = delete;
  ~DiskReadBatch() {
    join();
    dispatch_release(group_);
  }

  size_t size() const noexcept { return reads_.size(); }
  DiskSchedule schedule() const noexcept { return schedule_; }

  std::vector<uint8_t> ready() const {
    std::vector<uint8_t> mask(reads_.size());
    for (size_t i = 0; i < mask.size(); ++i)
      mask[i] = states_[i].load(std::memory_order_acquire) == 1;
    return mask;
  }

  std::vector<uint8_t> wait_ready(size_t minimum,
                                std::chrono::milliseconds timeout = std::chrono::seconds(30)) {
    if (!minimum || minimum > 512 || timeout.count() <= 0)
      throw std::invalid_argument("invalid disk readiness wait");
    std::unique_lock<std::mutex> lock(mutex_);
    if (!progress_.wait_for(lock, timeout, [&] {
          return successes_ >= minimum || completed_ == reads_.size() || failed_;
        })) throw std::runtime_error("timed out waiting for disk batch");
    const bool failure = failed_;
    lock.unlock();
    if (failure) finish();  // Drain every job before propagating the canonical error.
    return ready();
  }

  void join() noexcept {
    if (joined_) return;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      tail_released_ = true;
    }
    progress_.notify_all();
    dispatch_group_wait(group_, DISPATCH_TIME_FOREVER);
    joined_ = true;
  }

  void finish() {
    join();
    for (const auto& error : errors_)
      if (error) std::rethrow_exception(error);
  }

  // Non-atomic payloads are visible only after acquiring that job's terminal state.
  uint64_t bytes(size_t job) const {
    if (job >= reads_.size()) throw std::out_of_range("invalid disk job");
    return states_[job].load(std::memory_order_acquire) ? reads_[job].bytes_read : 0;
  }
  const std::vector<DiskTiming>& timings() const {
    if (!joined_) throw std::logic_error("disk timings requested before join");
    return timings_;
  }

 private:
  struct Worker { DiskReadBatch* batch = nullptr; size_t worker = 0; };
  static void run_worker(void* opaque) noexcept {
    auto& context = *static_cast<Worker*>(opaque);
    auto& batch = *context.batch;
    size_t cursor = context.worker;
    for (size_t i = batch.jobs_.claim(cursor); i < batch.reads_.size();
         i = batch.jobs_.claim(cursor)) {
      if (i == batch.held_job_) {
        std::unique_lock<std::mutex> lock(batch.mutex_);
        batch.progress_.wait(lock, [&] { return batch.tail_released_; });
      }
      auto ns = [&] { return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::steady_clock::now() - batch.epoch_).count()); };
      if (!batch.timings_.empty()) batch.timings_[i] = {i, context.worker, ns(), 0, 0, 0};
      try { read_disk_record(batch.reads_[i]); }
      catch (...) { batch.errors_[i] = std::current_exception(); }
      if (!batch.timings_.empty()) {
        auto& timing = batch.timings_[i];
        timing.end_ns = ns(); timing.bytes = batch.reads_[i].bytes_read;
        timing.failed = bool(batch.errors_[i]);
      }
      {
        std::lock_guard<std::mutex> lock(batch.mutex_);
        const bool failure = bool(batch.errors_[i]);
        batch.states_[i].store(failure ? 2 : 1, std::memory_order_release);
        ++batch.completed_;
        if (failure) batch.failed_ = true;
        else ++batch.successes_;
      }
      batch.progress_.notify_all();
    }
  }

  std::vector<DiskRead> reads_;
  const DiskSchedule schedule_;
  DiskWorkQueue jobs_;
  std::vector<std::exception_ptr> errors_;
  std::unique_ptr<std::atomic<uint8_t>[]> states_;
  std::vector<DiskTiming> timings_;
  std::array<Worker, 4> contexts_{};
  size_t workers_ = 0, held_job_, completed_ = 0, successes_ = 0;
  bool tail_released_, failed_ = false, joined_ = false;
  std::mutex mutex_;
  std::condition_variable progress_;
  std::chrono::steady_clock::time_point epoch_;
  dispatch_group_t group_ = nullptr;
};
}  // namespace mlxl3
