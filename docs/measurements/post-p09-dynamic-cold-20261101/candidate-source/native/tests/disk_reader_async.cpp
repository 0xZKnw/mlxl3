#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <iostream>
#include <sys/uio.h>
#include <unistd.h>
#include <vector>

static std::atomic<bool> inject_partial{false};
static std::atomic<size_t> syscall_count{0};
// Substitute only the syscall boundary: the actual reader loops, offsets and iovecs remain unchanged.
static ssize_t checked_preadv(int fd, const iovec* vectors, int count, off_t offset) {
  if (!inject_partial.load()) return ::preadv(fd, vectors, count, offset);
  if (syscall_count.fetch_add(1) == 0) { errno = EINTR; return -1; }
  std::vector<iovec> short_read;
  size_t remaining = 3;
  for (int i = 0; i < count && remaining; ++i) {
    const auto size = std::min(remaining, vectors[i].iov_len);
    short_read.push_back({vectors[i].iov_base, size});
    remaining -= size;
  }
  return ::preadv(fd, short_read.data(), int(short_read.size()), offset);
}
#define preadv checked_preadv
#include "disk_read_batch.h"
#undef preadv

using Bank = std::array<uint8_t, 16>;
static std::vector<mlxl3::DiskRead> records(int fd, std::vector<Bank>& banks) {
  std::vector<mlxl3::DiskRead> reads;
  for (size_t i = 0; i < banks.size(); ++i) {
    banks[i].fill(0xa5);
    reads.push_back({fd, i * 16, {{banks[i].data(), 2}, {banks[i].data() + 2, 3},
                                 {banks[i].data() + 5, 3}}});
  }
  return reads;
}
static void exact(const std::vector<Bank>& banks, const std::vector<uint8_t>& source) {
  for (size_t i = 0; i < banks.size(); ++i)
    for (size_t j = 0; j < 16; ++j)
      assert(banks[i][j] == (j < 8 ? source[i * 16 + j] : 0xa5));
}

int main() {
  auto* file = std::tmpfile();
  assert(file);
  const int fd = fileno(file);
  std::vector<uint8_t> source(512 * 16);
  for (size_t i = 0; i < source.size(); ++i) source[i] = uint8_t((i * 19 + i / 257) % 251);
  assert(std::fwrite(source.data(), 1, source.size(), file) == source.size());
  assert(std::fflush(file) == 0);
  size_t cases = 0;
  for (auto schedule : {mlxl3::DiskSchedule::Static, mlxl3::DiskSchedule::Dynamic}) {
  for (size_t workers = 1; workers <= 4; ++workers) {
    for (size_t count : {1, 2, 3, 6, 20, 512}) {
      std::vector<Bank> banks(count);
      mlxl3::DiskReadBatch batch(records(fd, banks), workers, true, SIZE_MAX, schedule);
      const bool dynamic = schedule == mlxl3::DiskSchedule::Dynamic && workers > 1 && count > workers;
      assert((batch.schedule() == mlxl3::DiskSchedule::Dynamic) == dynamic);
      const auto mask = batch.wait_ready(2);
      assert(mask.size() == count && size_t(std::count(mask.begin(), mask.end(), 1)) >= std::min(count, size_t(2)));
      batch.finish(); batch.finish();
      const auto complete = batch.ready();
      assert(std::all_of(complete.begin(), complete.end(), [](auto value) { return value == 1; }));
      const auto& timings = batch.timings();
      assert(timings.size() == count);
      for (size_t i = 0; i < count; ++i) {
        assert(batch.bytes(i) == 8 && timings[i].job == i);
        assert(timings[i].worker < std::min(workers, count));
        if (!dynamic) assert(timings[i].worker == i % std::min(workers, count));
        assert(timings[i].end_ns >= timings[i].start_ns && timings[i].bytes == 8 && !timings[i].failed);
      }
      exact(banks, source); ++cases;
    }
  }
  for (size_t held : {size_t(0), size_t(2)}) {
    std::vector<Bank> banks(6);
    mlxl3::DiskReadBatch batch(records(fd, banks), 4, true, held, schedule);
    const auto early = batch.wait_ready(2);
    assert(!early[held] && batch.bytes(held) == 0);
    assert(std::count(early.begin(), early.end(), 1) >= 2);
    for (size_t i = 0; i < early.size(); ++i)
      if (early[i]) assert(std::equal(banks[i].begin(), banks[i].begin() + 8, source.begin() + i * 16));
    batch.finish(); exact(banks, source); ++cases;
  }
  {
    std::vector<Bank> banks(6);
    {
      mlxl3::DiskReadBatch batch(records(fd, banks), 4, false, 0, schedule);
      assert(!batch.wait_ready(2)[0]);
    }  // Drop releases the gate and joins before destination storage is examined/freed.
    exact(banks, source); ++cases;
  }
  for (bool first_truncated : {false, true}) {
    std::vector<Bank> banks(6);
    auto reads = records(fd, banks);
    reads[0].fd = first_truncated ? fd : -1;
    reads[0].offset = first_truncated ? source.size() - 1 : 0;
    reads[3].fd = first_truncated ? -1 : fd;
    reads[3].offset = first_truncated ? 0 : source.size() - 1;
    mlxl3::DiskReadBatch batch(std::move(reads), 4, true, 0, schedule);
    for (size_t repeat = 0; repeat < 2; ++repeat) {
      bool failed = false;
      try { if (!repeat) batch.wait_ready(512); else batch.finish(); }
      catch (const std::runtime_error& error) {
        failed = true;
        assert(std::string(error.what()).find(first_truncated ? "checkpoint truncated" : "preadv failed") != std::string::npos);
      }
      assert(failed);
    }
    const auto mask = batch.ready();
    assert(!mask[0] && !mask[3]);
    for (size_t i : {size_t(1), size_t(2), size_t(4), size_t(5)}) {
      assert(mask[i] && batch.bytes(i) == 8);
      assert(std::equal(banks[i].begin(), banks[i].begin() + 8, source.begin() + i * 16));
    }
    assert(batch.bytes(0) == (first_truncated ? 1 : 0));
    assert(batch.bytes(3) == (first_truncated ? 0 : 1));
    ++cases;
  }
  {
    std::vector<Bank> banks(3);
    {
      mlxl3::DiskReadBatch batch(records(fd, banks), 1, false, 0, schedule);
      bool timed_out = false;
      try { batch.wait_ready(2, std::chrono::milliseconds(1)); }
      catch (const std::runtime_error&) { timed_out = true; }
      assert(timed_out && batch.ready() == std::vector<uint8_t>(3, 0));
      for (size_t minimum : {size_t(0), size_t(513)}) {
        bool rejected = false;
        try { batch.wait_ready(minimum); } catch (const std::invalid_argument&) { rejected = true; }
        assert(rejected); ++cases;
      }
    }
    exact(banks, source); ++cases;
  }
  {
    std::vector<Bank> banks(1);
    auto read = records(fd, banks)[0];
    syscall_count.store(0); inject_partial.store(true);
    mlxl3::read_disk_record(read);
    inject_partial.store(false);
    assert(read.bytes_read == 8 && syscall_count.load() == 4);
    exact(banks, source); ++cases;
  }
  {
    std::vector<Bank> banks(2);
    auto reject = [&](std::vector<mlxl3::DiskRead> reads, size_t workers, size_t held = SIZE_MAX) {
      for (auto& bank : banks) bank.fill(0xa5);
      bool failed = false;
      try { mlxl3::DiskReadBatch batch(std::move(reads), workers, false, held, schedule); }
      catch (const std::exception&) { failed = true; }
      assert(failed);
      for (const auto& bank : banks) assert(std::all_of(bank.begin(), bank.end(), [](auto v) { return v == 0xa5; }));
      ++cases;
    };
    reject(records(fd, banks), 0); reject(records(fd, banks), 5);
    reject(records(fd, banks), 4, 2); reject({}, 4);
    reject({{fd, 0, {{banks[0].data(), 8}}}, {fd, 8, {{banks[0].data() + 7, 8}}}}, 4);
    reject({{fd, UINT64_MAX, {{banks[0].data(), 8}}}}, 4);
    reject({{fd, 0, {{nullptr, 8}}}}, 4);
    reject({{fd, 0, {{banks[0].data(), 0}}}}, 4);
  }
  }
  {
    std::vector<Bank> banks(12);
    mlxl3::DiskReadBatch batch(records(fd, banks), 4, true, 0, mlxl3::DiskSchedule::Dynamic);
    // A blocked worker must not strand jobs 4 and 8 in its former static stripe.
    const auto ready = batch.wait_ready(11, std::chrono::seconds(2));
    assert(!ready[0] && std::count(ready.begin(), ready.end(), 1) == 11);
    assert(batch.bytes(0) == 0);
    batch.finish(); exact(banks, source); ++cases;
  }
  assert(std::fclose(file) == 0);
  std::cout << cases << " async disk reader cases passed\n";
}
