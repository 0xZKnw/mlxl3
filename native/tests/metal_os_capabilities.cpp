#include "metal_os_capabilities.h"

#include <array>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>

namespace {
std::size_t checks = 0;
bool check(bool condition, const std::string& context) {
  ++checks;
  if (!condition) std::cerr << "Metal OS capability failure: " << context << '\n';
  return condition;
}
}

int main() {
  for (const auto& [version, expected] : std::array<std::pair<const char*, uint32_t>, 12>{{
      {"26.2", 0}, {"26.2.99", 0}, {"26.3", 1}, {"26.3.1", 1},
      {"26.4", 3}, {"26.4.0", 3}, {"27.0", 3}, {"27.1.3", 3},
      {"25.99.99", 0}, {"0.0", 0}, {"26.4294967295", 3},
      {"4294967295.0.0", 3}}}) {
    const auto actual = mlxl3::metal_os_capabilities(version, std::strlen(version) + 1);
    if (!check(actual && *actual == expected, version)) return 1;
  }
  for (const char* version : {"", "26", "26.", ".3", "26..3", "26.3.",
       "26.3.0.1", "26.3beta", " 26.3", "26.3\n", "-26.3", "26.+3",
       "4294967296.3", "26.4294967296", "26.3.4294967296"}) {
    if (!check(!mlxl3::metal_os_capabilities(version, std::strlen(version) + 1),
               std::string("reject ") + version)) return 1;
  }
  std::array<char, mlxl3::metal_os_version_capacity> buffer{};
  buffer.fill('9');
  if (!check(!mlxl3::metal_os_capabilities(buffer.data(), buffer.size()),
             "unterminated buffer")) return 1;
  buffer.back() = '\0';
  if (!check(!mlxl3::metal_os_capabilities(buffer.data(), buffer.size()),
             "bounded overflowing component")) return 1;
  if (!check(!mlxl3::metal_os_capabilities(buffer.data(), buffer.size() + 1),
             "oversized length rejected before reading")) return 1;
  if (!check(!mlxl3::metal_os_capabilities(nullptr, 64), "null")) return 1;
  if (!check(!mlxl3::metal_os_capabilities("26.3", 0), "zero bytes")) return 1;
  if (!check(!mlxl3::metal_os_capabilities("26.3", 1), "short bytes")) return 1;
  if (!check(!mlxl3::metal_os_capabilities("26.3", 4), "missing final NUL")) return 1;
  if (!check(!mlxl3::metal_os_capabilities("26.3\0junk", 10), "embedded NUL")) return 1;
  for (uint32_t major = 20; major <= 30; ++major) {
    for (uint32_t minor = 0; minor <= 20; ++minor) {
      for (uint32_t patch = 0; patch <= 4; ++patch) {
        const std::string version = std::to_string(major) + "." +
            std::to_string(minor) + "." + std::to_string(patch);
        const uint32_t expected = major < 26 ? 0 : major > 26 ? 3 :
            minor < 3 ? 0 : minor == 3 ? 1 : 3;
        const auto actual = mlxl3::metal_os_capabilities(version.c_str(), version.size() + 1);
        if (!check(actual && *actual == expected, version)) return 1;
      }
    }
  }
  std::cout << checks << " Metal OS parser/capability checks passed\n";
}
