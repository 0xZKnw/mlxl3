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
  for (size_t workers = 1; workers <= 4; ++workers) {
    for (size_t count : {1, 2, 5, 512}) {
      std::vector<std::array<uint8_t, 32>> banks(count);
      for (auto& b : banks) b.fill(0xa5);
      std::vector<mlxl3::DiskRead> reads;
      for (size_t i = 0; i < count; ++i)
        reads.push_back({fd, i * 32, {{banks[i].data() + 2, 9}, {banks[i].data() + 17, 7}}});
      mlxl3::read_disk_records(reads, workers);
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
      try { mlxl3::read_disk_records(reads, workers); }
      catch (const std::runtime_error&) { failed = true; }
      assert(failed);
      for (size_t i = 1; i < 8; ++i) {
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
