#include "kernel_cache_key.h"

#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <new>
#include <tuple>
#include <utility>

namespace {
bool track_allocations = false;
std::size_t allocation_count = 0;
}

void* operator new(std::size_t size) {
  if (track_allocations) ++allocation_count;
  if (void* value = std::malloc(size ? size : 1)) return value;
  throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* value) noexcept { std::free(value); }
void operator delete[](void* value) noexcept { std::free(value); }
#if defined(__cpp_sized_deallocation)
void operator delete(void* value, std::size_t) noexcept { std::free(value); }
void operator delete[](void* value, std::size_t) noexcept { std::free(value); }
#endif

namespace {
using LegacyKey = std::tuple<std::string, std::vector<std::string>,
                             std::vector<std::string>, std::string, std::string>;
using LegacyMap = std::map<LegacyKey, int>;
using ViewMap = std::map<mlxl3::OwnedKernelKey, int, mlxl3::KernelKeyOrder>;

struct Fixture {
  mlxl3::OwnedKernelKey key;
  std::vector<const char*> inputs;
  std::vector<const char*> outputs;

  mlxl3::KernelKeyView view() const {
    return {key.name.c_str(), {inputs.data(), inputs.size()},
            {outputs.data(), outputs.size()}, key.header.c_str(), key.source.c_str()};
  }
};

auto legacy_ref(const mlxl3::OwnedKernelKey& key) {
  return std::tie(key.name, key.inputs, key.outputs, key.header, key.source);
}

std::vector<std::string> copy_names(const std::vector<const char*>& names) {
  std::vector<std::string> result;
  for (const char* name : names) result.emplace_back(name);
  return result;
}

LegacyKey legacy_copy(const Fixture& fixture) {
  return {fixture.key.name.c_str(), copy_names(fixture.inputs), copy_names(fixture.outputs),
          fixture.key.header.c_str(), fixture.key.source.c_str()};
}

std::vector<Fixture> fixtures() {
  std::vector<Fixture> result;
  std::uint32_t random = 0x7541ab89;
  for (int index = 0; index < 256; ++index) {
    mlxl3::OwnedKernelKey key{"kernel_name_longer_than_small_string_storage",
                              {"input", "another_input"}, {"output"},
                              std::string(2048, 'h'), std::string(8192, 's')};
    key.source += std::to_string(index);
    if (index < 32) {
      switch (index % 8) {
        case 0: key.name = ""; break;
        case 1: key.inputs.clear(); break;
        case 2: key.inputs = {"", "a", "aa"}; break;
        case 3: key.outputs.clear(); break;
        case 4: key.outputs = {"a", "aa", ""}; break;
        case 5: key.header = ""; break;
        case 6: key.header = std::string(1, char(0x80)); break;
        case 7: key.source = std::string(1, char(0xff)); break;
      }
    } else if (index < 128) {
      for (int field = 0; field < 5; ++field) {
        random = random * 1664525u + 1013904223u;
        std::string text;
        for (std::uint32_t length = random % 5; length != 0; --length) {
          random = random * 1664525u + 1013904223u;
          text += char(1 + random % 255);
        }
        switch (field) {
          case 0: key.name = text; break;
          case 1: key.inputs = {text, text + "a"}; break;
          case 2: key.outputs = {text}; break;
          case 3: key.header = text; break;
          case 4: key.source = text + std::to_string(index); break;
        }
      }
    }
    result.push_back({std::move(key), {}, {}});
  }
  for (auto& fixture : result) {
    for (const auto& name : fixture.key.inputs) fixture.inputs.push_back(name.c_str());
    for (const auto& name : fixture.key.outputs) fixture.outputs.push_back(name.c_str());
  }
  return result;
}

__attribute__((noinline)) int lookup(const LegacyMap& cache, const Fixture& fixture) {
  auto found = cache.find(legacy_copy(fixture));
  assert(found != cache.end());
  return found->second;
}

__attribute__((noinline)) int lookup(const ViewMap& cache, const Fixture& fixture) {
  auto found = cache.find(fixture.view());
  assert(found != cache.end());
  return found->second;
}

void check(const std::vector<Fixture>& cases) {
  mlxl3::KernelKeyOrder less;
  for (const auto& left : cases) {
    for (const auto& right : cases) {
      bool expected = legacy_ref(left.key) < legacy_ref(right.key);
      assert(less(left.key, right.key) == expected);
      assert(less(left.key, right.view()) == expected);
      assert(less(left.view(), right.key) == expected);
      assert(less(left.view(), right.view()) == expected);
    }
  }
  LegacyMap original;
  ViewMap candidate;
  for (std::size_t index = 0; index < cases.size(); ++index) {
    if (original.size() >= 128) original.erase(original.begin());
    if (candidate.size() >= 128) candidate.erase(candidate.begin());
    original.emplace(legacy_copy(cases[index]), int(index));
    {
      auto ephemeral = cases[index].key;
      candidate.emplace(ephemeral, int(index));
      ephemeral.name.assign(100, 'z');
      ephemeral.inputs.clear();
      ephemeral.header.assign(4000, 'x');
    }
    assert(original.size() == candidate.size());
    auto left = original.begin();
    for (const auto& entry : candidate) {
      assert(left != original.end());
      assert(left->first == legacy_ref(entry.first));
      assert(left->second == entry.second);
      ++left;
    }
    assert(left == original.end());
    for (const auto& fixture : cases) {
      auto old = original.find(legacy_copy(fixture));
      auto now = candidate.find(fixture.view());
      assert((old == original.end()) == (now == candidate.end()));
      if (old != original.end()) assert(old->second == now->second);
    }
  }
}

void benchmark(const std::vector<Fixture>& cases, bool measure) {
  LegacyMap original;
  ViewMap candidate;
  for (int index = 128; index < 256; ++index) {
    original.emplace(legacy_copy(cases[index]), index);
    candidate.emplace(cases[index].key, index);
  }
  volatile std::uint64_t checksum = 0;
  allocation_count = 0;
  track_allocations = true;
  for (int index = 128; index < 256; ++index) checksum += lookup(original, cases[index]);
  track_allocations = false;
  auto old_allocations = allocation_count;
  allocation_count = 0;
  track_allocations = true;
  for (int index = 128; index < 256; ++index) checksum += lookup(candidate, cases[index]);
  track_allocations = false;
  assert(old_allocations > 0 && allocation_count == 0);
  std::cout << "KEY_VIEW_QUALITY {\"ordering_pairs\":65536,\"ordering_combinations\":4,"
               "\"cache_keys\":256,\"capacity\":128,\"old_allocations_128_hits\":"
            << old_allocations << ",\"candidate_allocations_128_hits\":" << allocation_count
            << ",\"checksum\":" << checksum << "}\n";
  if (!measure) return;
  auto run = [&](bool view, int iterations) {
    auto start = std::chrono::steady_clock::now();
    for (int index = 0; index < iterations; ++index) {
      const auto& fixture = cases[128 + index % 128];
      checksum += view ? lookup(candidate, fixture) : lookup(original, fixture);
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
           / iterations;
  };
  run(false, 100000);
  run(true, 100000);
  for (const char* order : {"ABBA", "BAAB"}) {
    for (int pass = 0; pass < 4; ++pass) {
      std::cout << "KEY_VIEW_MICRO {\"order\":\"" << order << "\",\"pass\":" << pass
                << ",\"variant\":\"" << order[pass] << "\",\"iterations\":200000,"
                   "\"seconds_per_lookup\":[";
      for (int sample = 0; sample < 7; ++sample) {
        if (sample) std::cout << ',';
        std::cout.precision(17);
        std::cout << run(order[pass] == 'B', 200000);
      }
      std::cout << "]}\n" << std::flush;
    }
  }
  assert(checksum != 0);
}
}  // namespace

int main(int argc, char** argv) {
  if (argc > 2 || (argc == 2 && std::strcmp(argv[1], "--bench") != 0)) return 2;
  auto cases = fixtures();
  check(cases);
  benchmark(cases, argc == 2);
}
