#include "disk_work_queue.h"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>

int main() {
  using namespace mlxl3;
  assert(disk_schedule(nullptr) == DiskSchedule::Static);
  assert(disk_schedule("0") == DiskSchedule::Static);
  assert(disk_schedule("1") == DiskSchedule::Dynamic);
  for (const char* literal : {"", "true", "01", "2", "-1"}) {
    bool rejected = false;
    try { (void)disk_schedule(literal); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  size_t cases = 0;
  for (auto mode : {DiskSchedule::Static, DiskSchedule::Dynamic}) {
    for (size_t count = 1; count <= 512; ++count) {
      for (size_t workers = 1; workers <= std::min(count, size_t(4)); ++workers) {
        DiskWorkQueue queue(count, workers, mode);
        std::vector<std::atomic<unsigned>> seen(count);
        for (auto& value : seen) value.store(0);
        std::vector<std::thread> threads;
        for (size_t worker = 0; worker < workers; ++worker) threads.emplace_back([&, worker] {
          size_t cursor = worker;
          for (auto job = queue.claim(cursor); job < count; job = queue.claim(cursor)) {
            if (mode == DiskSchedule::Static) assert(job % workers == worker);
            seen[job].fetch_add(1);
            if ((job + worker) % 3 == 0) std::this_thread::yield();
          }
          for (size_t i = 0; i < 16; ++i) assert(queue.claim(cursor) == count);
        });
        for (auto& thread : threads) thread.join();
        for (const auto& value : seen) assert(value.load() == 1);
        ++cases;
      }
    }
  }
  // A slow worker holding job zero does not pin its remaining static stripe.
  DiskWorkQueue queue(8, 2, DiskSchedule::Dynamic);
  size_t slow = 0, fast = 1;
  assert(queue.claim(slow) == 0);
  for (size_t job = 1; job < 8; ++job) assert(queue.claim(fast) == job);
  assert(queue.claim(fast) == 8 && queue.claim(slow) == 8);
  for (size_t count : {0, 513}) {
    bool rejected = false;
    try { DiskWorkQueue invalid(count, 1, DiskSchedule::Dynamic); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  for (size_t workers : {0, 5, 9}) {
    bool rejected = false;
    try { DiskWorkQueue invalid(8, workers, DiskSchedule::Dynamic); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  {
    bool rejected = false;
    try { DiskWorkQueue invalid(1, 2, DiskSchedule::Dynamic); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  std::cout << cases << " concurrent disk queue cases passed\n";
}
