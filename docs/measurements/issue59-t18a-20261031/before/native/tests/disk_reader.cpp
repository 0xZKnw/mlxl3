#include "disk_reader.h"

#include <array>
#include <cassert>
#include <cstdio>
#include <iostream>

int main() {
  auto* file = std::tmpfile();
  assert(file);
  const int fd = fileno(file);
  std::vector<uint8_t> source(512 * 32);
  for (size_t i = 0; i < source.size(); ++i) source[i] = uint8_t(i * 31 + i / 251);
  assert(std::fwrite(source.data(), 1, source.size(), file) == source.size());
  assert(std::fflush(file) == 0);
  size_t cases = 0;
  for (size_t start = 0; start <= 16; ++start) {
    for (size_t length = 0; length <= 16 - start; ++length) {
      std::array<uint8_t, 32> bank;
      bank.fill(0xa5);
      const auto original = bank;
      bool failed = false;
      try { mlxl3::copy_disk_record(source.data(), 8, {{bank.data() + start, length}, {bank.data() + 16, 8 - std::min(length, size_t(8))}}); }
      catch (const std::invalid_argument&) { failed = true; }
      const bool valid = length > 0 && length < 8;
      assert(failed != valid);
      if (valid) {
        for (size_t i = 0; i < length; ++i) assert(bank[start + i] == source[i]);
        for (size_t i = 0; i < 8 - length; ++i) assert(bank[16 + i] == source[length + i]);
      } else { assert(bank == original); }
      ++cases;
    }
  }
  {
    std::array<uint8_t, 16> bank{};
    auto rejects = [&](const uint8_t* input, size_t size, std::vector<iovec> ranges) {
      bank.fill(0xa5); const auto original = bank;
      bool failed = false;
      try { mlxl3::copy_disk_record(input, size, ranges); }
      catch (const std::invalid_argument&) { failed = true; }
      assert(failed && bank == original); ++cases;
    };
    rejects(source.data(), 8, {{bank.data(), 5}, {bank.data() + 4, 3}});
    rejects(source.data(), 8, {{nullptr, 8}});
    rejects(nullptr, 8, {{bank.data(), 8}});
    rejects(bank.data(), 8, {{bank.data() + 1, 8}});
    rejects(source.data(), 0, {{bank.data(), 8}});
    rejects(source.data(), 8, {{bank.data(), 7}});
    rejects(source.data(), 8, {{reinterpret_cast<void*>(UINTPTR_MAX), 8}});
    rejects(reinterpret_cast<uint8_t*>(UINTPTR_MAX), 8, {{bank.data(), 8}});
    rejects(source.data(), 8, {});
    mlxl3::copy_disk_record(source.data(), 8, {{bank.data(), 8}});
    assert(std::equal(bank.begin(), bank.begin() + 8, source.begin())); ++cases;
  }
  for (size_t workers = 1; workers <= 4; ++workers) {
    for (size_t count : {1, 2, 5, 512}) {
      std::vector<std::array<uint8_t, 32>> banks(count);
      for (auto& b : banks) b.fill(0xa5);
      std::vector<mlxl3::DiskRead> reads;
      for (size_t i = 0; i < count; ++i)
        reads.push_back({fd, i * 32, {{banks[i].data() + 2, 9}, {banks[i].data() + 17, 7}}});
      std::vector<mlxl3::DiskTiming> timings;
      mlxl3::read_disk_records(reads, workers, &timings);
      assert(timings.size() == count);
      for (size_t i = 0; i < count; ++i) {
        const auto& t = timings[i];
        assert(t.job == i && t.worker == i % std::min(workers, count));
        assert(t.end_ns >= t.start_ns && t.bytes == 16 && !t.failed);
        if (i >= workers) assert(t.start_ns >= timings[i - workers].end_ns);
      }
      for (const auto& read : reads) assert(read.bytes_read == 16);
      for (size_t i = 0; i < count; ++i)
        for (size_t j = 0; j < 32; ++j) {
          const uint8_t expected = j >= 2 && j < 11 ? source[i * 32 + j - 2]
              : j >= 17 && j < 24 ? source[i * 32 + j - 8] : 0xa5;
          assert(banks[i][j] == expected);
        }
      ++cases;
    }
    std::array<uint8_t, 16> bank{};
    auto rejects = [&](std::vector<mlxl3::DiskRead> reads, size_t threads) {
      bank.fill(0xa5);
      bool failed = false;
      try { mlxl3::read_disk_records(reads, threads); }
      catch (const std::exception&) { failed = true; }
      assert(failed);
      assert(std::all_of(bank.begin(), bank.end(), [](auto x) { return x == 0xa5; }));
      ++cases;
    };
    rejects({{fd, 0, {{bank.data(), 5}, {bank.data() + 4, 2}}}}, workers);
    rejects({{fd, 0, {{bank.data(), 5}}}, {fd, 8, {{bank.data() + 4, 3}}}}, workers);
    rejects({{fd, 0, {{bank.data(), 0}}}}, workers);
    rejects({{fd, 0, {{nullptr, 1}}}}, workers);
    rejects({{fd, UINT64_MAX, {{bank.data(), 1}}}}, workers);
    rejects({{fd, 0, {{reinterpret_cast<void*>(UINTPTR_MAX), 2}}}}, workers);
    rejects({{fd, 0, {}}}, workers);
    rejects({}, workers);
    rejects({{fd, 0, {{bank.data(), 1}}}}, 0);
    rejects({{fd, 0, {{bank.data(), 1}}}}, 5);
    rejects(std::vector<mlxl3::DiskRead>(513, {fd, 0, {{bank.data(), 1}}}), workers);
    rejects({{fd, 0, std::vector<iovec>(65, {bank.data(), 1})}}, workers);

    // A failing record cannot leave sibling readers alive or skip later records.
    for (bool truncated : {false, true}) {
      bank.fill(0xa5);
      std::vector<mlxl3::DiskRead> reads;
      for (size_t i = 0; i < 8; ++i)
        reads.push_back({i == 0 && !truncated ? -1 : fd,
                         i == 0 && truncated ? source.size() - 1 : i * 32,
                         {{bank.data() + i * 2, 2}}});
      bool failed = false;
      std::vector<mlxl3::DiskTiming> timings;
      try { mlxl3::read_disk_records(reads, workers, &timings); }
      catch (const std::runtime_error&) { failed = true; }
      assert(failed && timings.size() == reads.size());
      for (size_t i = 0; i < timings.size(); ++i) {
        assert(timings[i].job == i && timings[i].worker == i % workers);
        assert(timings[i].end_ns >= timings[i].start_ns);
        assert(timings[i].bytes == reads[i].bytes_read && timings[i].failed == (i == 0));
      }
      assert(reads[0].bytes_read == (truncated ? 1 : 0));
      for (size_t i = 1; i < 8; ++i) {
        assert(reads[i].bytes_read == 2);
        assert(bank[i * 2] == source[i * 32]);
        assert(bank[i * 2 + 1] == source[i * 32 + 1]);
      }
      std::vector<mlxl3::DiskRead> retry{{fd, 0, {{bank.data(), bank.size()}}}};
      mlxl3::read_disk_records(retry, workers);
      assert(std::equal(bank.begin(), bank.end(), source.begin()));
      ++cases;
    }
  }
  assert(std::fclose(file) == 0);
  std::cout << cases << " disk reader cases passed\n";
}
