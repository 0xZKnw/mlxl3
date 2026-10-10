#include "kernel_cache_key.h"

#include <array>
#include <tuple>

extern "C" void __CPROVER_havoc_object(void*);
extern "C" void __CPROVER_assume(bool);
extern "C" void __CPROVER_assert(bool, const char*);
extern "C" void __CPROVER_cover(bool);

namespace {
unsigned char any_byte() {
  unsigned char value = 0;
  __CPROVER_havoc_object(&value);
  return value;
}

struct Probe {
  std::array<std::array<char, 4>, 7> bytes{};
  std::array<const char*, 2> inputs{};
  std::array<const char*, 2> outputs{};
  mlxl3::KernelKeyView view{};
  mlxl3::OwnedKernelKey owned{};
};

void initialize(Probe& probe) {
  for (auto& text : probe.bytes) {
    for (std::size_t index = 0; index < 3; ++index) text[index] = char(any_byte());
    text[3] = 0;
  }
  std::size_t name = any_byte(), header = any_byte(), source = any_byte();
  std::size_t inputs = any_byte(), outputs = any_byte();
  __CPROVER_assume(name <= 3 && header <= 3 && source <= 3);
  __CPROVER_assume(inputs <= 2 && outputs <= 2);
  probe.inputs = {probe.bytes[3].data(), probe.bytes[4].data()};
  probe.outputs = {probe.bytes[5].data(), probe.bytes[6].data()};
  probe.view = {{probe.bytes[0].data(), name}, {probe.inputs.data(), inputs},
                {probe.outputs.data(), outputs}, {probe.bytes[1].data(), header},
                {probe.bytes[2].data(), source}};
  probe.owned.name.assign(probe.bytes[0].data(), name);
  probe.owned.header.assign(probe.bytes[1].data(), header);
  probe.owned.source.assign(probe.bytes[2].data(), source);
  for (std::size_t index = 0; index < inputs; ++index)
    probe.owned.inputs.emplace_back(probe.inputs[index]);
  for (std::size_t index = 0; index < outputs; ++index)
    probe.owned.outputs.emplace_back(probe.outputs[index]);
}
}  // namespace

void proof() {
  Probe left, right;
  initialize(left);
  initialize(right);
  auto tuple = [](const mlxl3::OwnedKernelKey& key) {
    return std::tie(key.name, key.inputs, key.outputs, key.header, key.source);
  };
  bool expected = tuple(left.owned) < tuple(right.owned);
  mlxl3::KernelKeyOrder less;
  __CPROVER_assert(less(left.owned, right.owned) == expected, "owned keys match original tuple");
  __CPROVER_assert(less(left.owned, right.view) == expected, "owned/view match original tuple");
  __CPROVER_assert(less(left.view, right.owned) == expected, "view/owned match original tuple");
  __CPROVER_assert(less(left.view, right.view) == expected, "borrowed keys match original tuple");
  __CPROVER_cover(expected);
  __CPROVER_cover(!expected);
  __CPROVER_cover(left.owned.inputs.empty() && right.owned.inputs.size() == 2);
  __CPROVER_cover(left.owned.name == right.owned.name && left.owned.header != right.owned.header);
}
