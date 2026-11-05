#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace mlxl3 {

constexpr std::size_t metal_os_version_capacity = 64;

// Parse the bounded, NUL-terminated kern.osproductversion result. Require
// major.minor with an optional patch component; reject incomplete, extra or
// overflowing components rather than accidentally enabling an optional API.
inline std::optional<uint32_t> metal_os_capabilities(
    const char* version, std::size_t bytes) noexcept {
  if (!version || bytes < 2 || bytes > metal_os_version_capacity ||
      version[bytes - 1] != '\0') return std::nullopt;
  std::array<uint32_t, 3> parts{};
  std::size_t part = 0;
  uint32_t value = 0;
  bool digit_seen = false;
  for (std::size_t index = 0; index + 1 < bytes; ++index) {
    const char c = version[index];
    if (c >= '0' && c <= '9') {
      const auto digit = uint32_t(c - '0');
      if (value > (std::numeric_limits<uint32_t>::max() - digit) / 10)
        return std::nullopt;
      value = value * 10 + digit;
      digit_seen = true;
    } else if (c == '.' && digit_seen && part < parts.size() - 1) {
      parts[part++] = value;
      value = 0;
      digit_seen = false;
    } else {
      return std::nullopt;
    }
  }
  if (!digit_seen || part == 0) return std::nullopt;
  parts[part] = value;
  const bool cooperative = parts[0] > 26 || (parts[0] == 26 && parts[1] >= 3);
  const bool native_q4 = parts[0] > 26 || (parts[0] == 26 && parts[1] >= 4);
  return uint32_t(cooperative) | (uint32_t(native_q4) << 1);
}

} // namespace mlxl3
