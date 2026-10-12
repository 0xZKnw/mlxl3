#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace mlxl3 {

struct KernelNamesView {
  const char* const* data;
  std::size_t count;

  std::size_t size() const noexcept { return count; }
  const char* operator[](std::size_t index) const noexcept { return data[index]; }
};

template<class Text, class Names> struct KernelCacheKey {
  Text name;
  Names inputs;
  Names outputs;
  Text header;
  Text source;
};

using OwnedKernelKey = KernelCacheKey<std::string, std::vector<std::string>>;
using KernelKeyView = KernelCacheKey<std::string_view, KernelNamesView>;

struct KernelKeyOrder {
  using is_transparent = void;

  template<class Left, class Right>
  static int names_compare(const Left& left, const Right& right) noexcept {
    for (std::size_t index = 0; index < left.size() && index < right.size(); ++index) {
      auto order = std::string_view(left[index]).compare(std::string_view(right[index]));
      if (order != 0) return order;
    }
    return left.size() < right.size() ? -1 : left.size() > right.size() ? 1 : 0;
  }

  template<class Left, class Right>
  bool operator()(const Left& left, const Right& right) const noexcept {
    auto order = std::string_view(left.name).compare(std::string_view(right.name));
    if (order != 0) return order < 0;
    order = names_compare(left.inputs, right.inputs);
    if (order != 0) return order < 0;
    order = names_compare(left.outputs, right.outputs);
    if (order != 0) return order < 0;
    order = std::string_view(left.header).compare(std::string_view(right.header));
    if (order != 0) return order < 0;
    return std::string_view(left.source).compare(std::string_view(right.source)) < 0;
  }
};

}  // namespace mlxl3
