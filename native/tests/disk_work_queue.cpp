#include "disk_work_queue.h"

#include <cassert>
#include <iostream>
#include <limits>
#include <thread>
#include <vector>

int main() {
  using namespace mlxl3;
  assert(disk_schedule(nullptr) == DiskSchedule::Dynamic);
  assert(disk_schedule("0") == DiskSchedule::Static);
  assert(disk_schedule("1") == DiskSchedule::Dynamic);
  for (const char* literal : {"", "true", "false", "01", "00", "2", "-1", "+1", " 1", "1 ", "1\n"}) {
    bool rejected = false;
    try { (void)disk_schedule(literal); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }

  size_t policy_cases = 0;
  for (auto requested : {DiskSchedule::Static, DiskSchedule::Dynamic}) {
    for (size_t count = 1; count <= 512; ++count) {
      for (size_t workers = 1; workers <= 4; ++workers) {
        const bool dynamic = requested == DiskSchedule::Dynamic && workers > 1 && count > workers;
        const auto actual = effective_disk_schedule(requested, count, workers);
        assert(actual == (dynamic ? DiskSchedule::Dynamic : DiskSchedule::Static));
        // The bridge selects before clamping idle workers out of the GCD batch.
        assert(effective_disk_schedule(requested, count, std::min(count, workers)) == actual);
        if (workers == 1 || count <= workers) assert(actual == DiskSchedule::Static);
        ++policy_cases;
      }
    }
  }
  assert(policy_cases == 4096);

  size_t cases = 0;
  for (auto mode : {DiskSchedule::Static, DiskSchedule::Dynamic}) {
    for (size_t count = 1; count <= 512; ++count) {
      for (size_t workers = 1; workers <= std::min(count, size_t(4)); ++workers) {
        DiskWorkQueue queue(count, workers, mode);
        std::vector<std::atomic<unsigned>> seen(count);
        for (auto& value : seen) value.store(0, std::memory_order_relaxed);
        std::vector<std::thread> threads;
        for (size_t worker = 0; worker < workers; ++worker) threads.emplace_back([&, worker] {
          size_t cursor = worker;
          for (auto job = queue.claim(cursor); job < count; job = queue.claim(cursor)) {
            if (mode == DiskSchedule::Static) assert(job % workers == worker);
            seen[job].fetch_add(1, std::memory_order_relaxed);
            if ((job + worker) % 3 == 0) std::this_thread::yield();
          }
          for (size_t i = 0; i < 16; ++i) assert(queue.claim(cursor) == count);
        });
        for (auto& thread : threads) thread.join();
        for (const auto& value : seen) assert(value.load(std::memory_order_relaxed) == 1);
        ++cases;
      }
    }
  }
  assert(cases == 4084);

  // A slow worker holding job zero does not pin its remaining static stripe.
  {
    DiskWorkQueue queue(8, 2, DiskSchedule::Dynamic);
    size_t slow = 0, fast = 1;
    assert(queue.claim(slow) == 0);
    for (size_t job = 1; job < 8; ++job) assert(queue.claim(fast) == job);
    assert(queue.claim(fast) == 8 && queue.claim(slow) == 8);
  }
  for (auto mode : {DiskSchedule::Static, DiskSchedule::Dynamic}) {
    DiskWorkQueue queue(512, 4, mode);
    size_t cursor = std::numeric_limits<size_t>::max();
    if (mode == DiskSchedule::Static) {
      assert(queue.claim(cursor) == 512);
    } else {
      for (size_t job = 0; job < 512; ++job) assert(queue.claim(cursor) == job);
    }
    for (size_t i = 0; i < 16; ++i) assert(queue.claim(cursor) == 512);
  }
  for (auto mode : {DiskSchedule::Static, DiskSchedule::Dynamic}) {
    for (size_t count : {size_t(0), size_t(513), std::numeric_limits<size_t>::max()}) {
      bool rejected = false;
      try { DiskWorkQueue invalid(count, 1, mode); }
      catch (const std::invalid_argument&) { rejected = true; }
      assert(rejected);
    }
    for (size_t workers : {size_t(0), size_t(5), size_t(9), std::numeric_limits<size_t>::max()}) {
      bool rejected = false;
      try { DiskWorkQueue invalid(8, workers, mode); }
      catch (const std::invalid_argument&) { rejected = true; }
      assert(rejected);
    }
    bool rejected = false;
    try { DiskWorkQueue invalid(1, 2, mode); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  std::cout << cases << " concurrent disk queue cases passed; "
            << policy_cases << " effective policy cases passed\n";
}
