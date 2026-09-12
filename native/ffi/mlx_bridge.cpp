// Thin, exception-contained C ABI over the same MLX backend as the reference.
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include "mlx/array.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/compile.h"
#include "mlx/device.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/version.h"

namespace mx = mlx::core;
static_assert(MLX_VERSION_MAJOR == 0 && MLX_VERSION_MINOR == 32 && MLX_VERSION_PATCH == 2,
              "MLXL3 native requires MLX headers 0.32.2");
namespace {
thread_local std::array<char, 2048> last_error{};
template<class F> int protect(F&& f) noexcept {
  last_error[0] = 0;
  try { f(); return 0; }
  catch (const std::exception& e) {
    std::strncpy(last_error.data(), e.what(), last_error.size() - 1);
  } catch (...) {
    std::strncpy(last_error.data(), "unknown MLX exception", last_error.size() - 1);
  }
  last_error.back() = 0;
  return -1;
}
mx::array& arr(void* p) {
  if (!p) throw std::invalid_argument("null array handle");
  return *static_cast<mx::array*>(p);
}
mx::Dtype dtype(int code) {
  switch (code) {
    case 0: return mx::bool_; case 1: return mx::uint8; case 2: return mx::uint16;
    case 3: return mx::uint32; case 4: return mx::uint64; case 5: return mx::int8;
    case 6: return mx::int16; case 7: return mx::int32; case 8: return mx::int64;
    case 9: return mx::float16; case 10: return mx::float32; case 11: return mx::float64;
    case 12: return mx::bfloat16; case 13: return mx::complex64;
    default: throw std::invalid_argument("invalid dtype");
  }
}
mx::Shape shape(const int32_t* p, size_t n) {
  if (!n) return {};
  if (!p) throw std::invalid_argument("null shape");
  return mx::Shape(p, p + n);
}
size_t byte_count(const mx::Shape& s, mx::Dtype dt) {
  size_t bytes = dt.size();
  for (auto dim : s) {
    if (dim < 0) throw std::invalid_argument("negative tensor dimension");
    if (dim && bytes > std::numeric_limits<size_t>::max() / size_t(dim))
      throw std::overflow_error("tensor size overflow");
    bytes *= size_t(dim);
  }
  return bytes;
}
std::vector<mx::array> arrays(void* const* p, size_t n) {
  std::vector<mx::array> result;
  result.reserve(n);
  for (size_t i = 0; i < n; ++i) result.push_back(arr(p[i]));
  return result;
}
std::vector<std::string> strings(const char* const* p, size_t n) {
  std::vector<std::string> result;
  for (size_t i = 0; i < n; ++i) result.emplace_back(p[i]);
  return result;
}
}

extern "C" {
const char* mlxl3_mlx_error() noexcept { return last_error.data(); }
int mlxl3_mlx_init(const char* metallib) noexcept {
  return protect([&] {
    static std::once_flag initialized;
    std::call_once(initialized, [&] {
      if (std::string(mx::version()) != "0.32.2")
        throw std::runtime_error("loaded MLX library is not version 0.32.2");
      mx::metal::set_metallib_path(metallib);
    });
    mx::set_default_device(mx::Device(mx::Device::gpu));
  });
}
void mlxl3_array_free(void* p) noexcept { delete static_cast<mx::array*>(p); }
int mlxl3_array_clone(void* p, void** out) noexcept {
  return protect([&] { *out = new mx::array(arr(p)); });
}
int mlxl3_array_metadata(void* p, size_t* rank, const int32_t** dims, int* type) noexcept {
  return protect([&] {
    auto& a = arr(p); *rank = a.ndim(); *dims = a.shape().data();
    *type = static_cast<int>(a.dtype().val());
  });
}
int mlxl3_array_from_bytes(const uint8_t* bytes, size_t count, const int32_t* dims,
                          size_t rank, int type, void** out) noexcept {
  return protect([&] {
    auto s = shape(dims, rank); auto dt = dtype(type);
    if (byte_count(s, dt) != count) throw std::invalid_argument("tensor byte count mismatch");
    // Copy into MLX-owned storage; the Rust input slice may be freed immediately.
    struct OwnedBuffer {
      mx::allocator::Buffer value;
      bool transferred = false;
      ~OwnedBuffer() { if (!transferred) mx::allocator::free(value); }
    } buffer{mx::allocator::malloc(count)};
    if (count) std::memcpy(buffer.value.raw_ptr(), bytes, count);
    *out = new mx::array(buffer.value, std::move(s), dt);
    buffer.transferred = true;
  });
}
int mlxl3_array_zeros(const int32_t* dims, size_t rank, int type, void** out) noexcept {
  return protect([&] { auto s = shape(dims, rank); byte_count(s, dtype(type));
    *out = new mx::array(mx::zeros(s, dtype(type))); });
}
int mlxl3_array_eval(void* p) noexcept { return protect([&] { arr(p).eval(); }); }
int mlxl3_array_copy_bytes(void* p, uint8_t* out, size_t count) noexcept {
  return protect([&] {
    auto a = mx::contiguous(arr(p));
    if (a.nbytes() != count) throw std::invalid_argument("output byte count mismatch");
    a.eval(); if (count) std::memcpy(out, a.data<uint8_t>(), count);
  });
}
// Opcodes stay private to array.rs; every operation returns a fresh owned handle.
int mlxl3_array_unary(void* p, int operation, const int32_t* args, size_t nargs,
                     float scalar, int flag, void** out) noexcept {
  return protect([&] {
    auto& a = arr(p);
    auto run = [&]() -> mx::array {
      switch (operation) {
        case 0: return mx::reshape(a, shape(args, nargs));
        case 1: return mx::transpose(a, std::vector<int>(args, args + nargs));
        case 2: {
          if (nargs != 3) throw std::invalid_argument("invalid slice arguments");
          int axis = args[0]; if (axis < 0) axis += a.ndim();
          if (axis < 0 || size_t(axis) >= a.ndim()) throw std::invalid_argument("slice axis out of bounds");
          auto start = a.shape(), stop = a.shape(); std::fill(start.begin(), start.end(), 0);
          start[axis] = args[1]; stop[axis] = args[2]; return mx::slice(a, start, stop);
        }
        case 3: return mx::astype(a, dtype(flag));
        case 4: return mx::hadamard_transform(a, flag ? std::optional<float>(scalar) : std::nullopt);
        case 5: return mx::sigmoid(a);
        case 6: if (nargs == 1) return mx::sum(a, args[0], bool(flag)); break;
        case 7: return mx::view(a, dtype(flag));
        case 8: if (nargs == 2) return mx::fast::rope(a, args[0], false, scalar, 1.0f, args[1]); break;
        case 9: return mx::subtract(a, mx::logsumexp(a, -1, true));
        case 10: return mx::softmax(a, -1, true);
        case 11: return mx::negative(a);
        case 12: return mx::exp(a);
        case 13: return mx::fast::rms_norm(a, std::nullopt, scalar);
        case 14: return mx::multiply(a, mx::sigmoid(a));
      }
      throw std::invalid_argument("invalid unary operation");
    };
    *out = new mx::array(run());
  });
}
int mlxl3_array_binary(void* lhs, void* rhs, int operation, int arg,
                      float scalar, void** out) noexcept {
  return protect([&] {
    auto& a = arr(lhs); auto& b = arr(rhs);
    auto run = [&]() -> mx::array {
      switch (operation) {
        case 0: return mx::add(a, b);
        case 1: return mx::multiply(a, b);
        case 2: return mx::matmul(a, b);
        case 3: return mx::take(a, b, arg);
        case 4: return mx::fast::rms_norm(a, b, scalar);
        case 5: return mx::conv1d(a, b, 1, 0, 1, arg);
        case 6: {
          thread_local auto swiglu = mx::compile([](const std::vector<mx::array>& x) {
            return std::vector<mx::array>{mx::multiply(mx::multiply(x[0], mx::sigmoid(x[0])), x[1])};
          }, true);
          return swiglu({a, b})[0];
        }
        case 7: return mx::logaddexp(a, b);
        case 8: {
          thread_local auto silu = mx::compile([](const std::vector<mx::array>& x) {
            return std::vector<mx::array>{mx::multiply(x[0], mx::sigmoid(x[0]))};
          }, true);
          auto gate = silu({mx::astype(a, mx::float32)})[0];
          auto value = mx::astype(b, mx::float32);
          return mx::astype(mx::multiply(gate, value), a.dtype());
        }
      }
      throw std::invalid_argument("invalid binary operation");
    };
    *out = new mx::array(run());
  });
}
int mlxl3_array_concatenate(void* const* inputs, size_t count, int axis, void** out) noexcept {
  return protect([&] { *out = new mx::array(mx::concatenate(arrays(inputs, count), axis)); });
}
int mlxl3_array_sdpa(void* q, void* k, void* v, float scale, int causal, void** out) noexcept {
  return protect([&] { *out = new mx::array(mx::fast::scaled_dot_product_attention(
    arr(q), arr(k), arr(v), scale, causal ? "causal" : "")); });
}
int mlxl3_metal_kernel(const char* name, const char* const* input_names,
                      const char* const* output_names, const char* header, const char* source,
                      void* const* inputs, size_t ninputs, const int32_t* output_dims,
                      const size_t* output_ranks, const int32_t* output_types, size_t noutputs,
                      const int32_t* grid, const int32_t* threadgroup, void** outputs) noexcept {
  return protect([&] {
    using Key = std::tuple<std::string, std::vector<std::string>, std::vector<std::string>, std::string, std::string>;
    thread_local std::map<Key, mx::fast::CustomKernelFunction> cache;
    Key key{name, strings(input_names, ninputs), strings(output_names, noutputs), header, source};
    auto found = cache.find(key);
    if (found == cache.end()) {
      // ponytail: bounded key-ordered eviction; use LRU only if dynamic factories churn.
      if (cache.size() >= 128) cache.erase(cache.begin());
      found = cache.emplace(key, mx::fast::metal_kernel(name, std::get<1>(key), std::get<2>(key),
        source, header, true, false, mx::CompileOptions{})).first;
    }
    std::vector<mx::Shape> shapes; std::vector<mx::Dtype> types; size_t offset = 0;
    for (size_t i = 0; i < noutputs; ++i) {
      shapes.push_back(shape(output_dims + offset, output_ranks[i])); offset += output_ranks[i];
      types.push_back(dtype(output_types[i])); byte_count(shapes.back(), types.back());
    }
    auto result = found->second(arrays(inputs, ninputs), shapes, types,
      {grid[0], grid[1], grid[2]}, {threadgroup[0], threadgroup[1], threadgroup[2]}, {}, std::nullopt, false, {});
    if (result.size() != noutputs) throw std::runtime_error("custom kernel output count mismatch");
    std::vector<std::unique_ptr<mx::array>> owned; owned.reserve(noutputs);
    for (auto& value : result) owned.push_back(std::make_unique<mx::array>(std::move(value)));
    for (size_t i = 0; i < noutputs; ++i) outputs[i] = owned[i].release();
  });
}
}
