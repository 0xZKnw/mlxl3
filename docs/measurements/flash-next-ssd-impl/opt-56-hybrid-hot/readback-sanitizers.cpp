// Exercise the actual C ABI under ASan/UBSan; libmlx itself is not instrumented.
#include "../../../../native/ffi/mlx_bridge.cpp"
#include <cassert>
#include <iostream>

void check_read(mx::array& a, const std::vector<uint32_t>& expected) {
  const size_t size = expected.size() * sizeof(uint32_t);
  std::vector<uint8_t> out(size + 2, 0xa5);
  assert(mlxl3_array_copy_bytes(&a, out.data() + 1, size) == 0);
  assert(out.front() == 0xa5 && out.back() == 0xa5);
  assert(!size || std::memcmp(out.data() + 1, expected.data(), size) == 0);
  assert(mlxl3_array_copy_bytes(&a, out.data(), size + 1) != 0);
  assert(mlxl3_array_copy_bytes(&a, nullptr, size) == (size ? -1 : 0));
}

int main(int argc, char** argv) {
  assert(argc == 2 && mlxl3_mlx_init(argv[1]) == 0);
  const std::vector<uint32_t> values{0u, 511u, 0x7fc01234u, 0xffffffffu, 0x80000000u, 7u};
  for (bool scheduled : {false, true}) {
    mx::array source(values.data(), {2, 3}, mx::uint32);
    auto base = mx::copy(source);
    if (scheduled) mx::async_eval({base});
    check_read(base, values);
    auto transposed = mx::transpose(base, {1, 0});
    if (scheduled) mx::async_eval({transposed});
    check_read(transposed, {values[0], values[3], values[1], values[4], values[2], values[5]});
    auto offset = mx::slice(base, {1, 0}, {2, 3});
    if (scheduled) mx::async_eval({offset});
    check_read(offset, {values[3], values[4], values[5]});
    auto empty = mx::slice(base, {0, 0}, {0, 3});
    if (scheduled) mx::async_eval({empty});
    check_read(empty, {});
    auto floats = mx::view(base, mx::float32);
    if (scheduled) mx::async_eval({floats});
    check_read(floats, values);
  }
  std::cout << "READBACK_SANITIZERS cases=10 canaries=true invalid_ranges=true raw_bits=exact\n";
}
