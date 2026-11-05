// Flash EXL3 slot preparation, using the Hadamard implementation from the linked MLX.
#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/ops.h"
#include "mlx/primitives.h"
#include <cstdio>
#include <mutex>

namespace mlxl3 {
namespace fx = mlx::core;

class FlashPrepareProgram final : public fx::Primitive {
 public:
  explicit FlashPrepareProgram(fx::Stream s) : Primitive(s) {
    // Only the stateless primitive survives construction. Never bind a real step here.
    fx::array prototype({1, 128}, fx::float16, nullptr, {});
    hadamard_ = fx::hadamard_transform(prototype, 1.0f / std::sqrt(128.0f), s).primitive_ptr();
  }

  const char* name() const override { return "MLXL3FlashPrepare128"; }
  void eval_cpu(const std::vector<fx::array>&, std::vector<fx::array>&) override {
    throw std::runtime_error("FlashPrepare128 requires Metal");
  }
  void eval_gpu(const std::vector<fx::array>& inputs, std::vector<fx::array>& outputs) override {
    const auto& x = inputs[0];
    const auto& scales = inputs[1];
    auto& device = fx::metal::device(stream().device);
    static constexpr auto source = R"metal(
#include <metal_stdlib>
using namespace metal;
template<typename T>
void product(device const T* x, device const half* s, device half* y, constant long* b, uint i) {
  long row = long(i) / b[0], col = long(i) % b[0];
  y[i] = half(x[row*b[1]+col*b[2]]) * s[row*b[3]+col*b[4]];
}
kernel void prep_half(device const half* x [[buffer(0)]], device const half* s [[buffer(1)]],
                      device half* y [[buffer(2)]], constant long* b [[buffer(3)]],
                      uint i [[thread_position_in_grid]]) { product(x,s,y,b,i); }
kernel void prep_float(device const float* x [[buffer(0)]], device const half* s [[buffer(1)]],
                       device half* y [[buffer(2)]], constant long* b [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) { product(x,s,y,b,i); }
)metal";
    auto lib = device.get_library("mlxl3_flash_prepare128_v1", [] { return std::string(source); });
    auto pipeline = device.get_kernel(x.dtype() == fx::float16 ? "prep_half" : "prep_float", lib);
    fx::array scratch({int(x.size() / 128), 128}, fx::float16, nullptr, {});
    scratch.set_data(fx::allocator::malloc(scratch.nbytes()));
    const std::array<int64_t, 5> bindings = {
      int64_t(x.shape(1)), x.strides()[0], x.strides()[1], scales.strides()[0], scales.strides()[1]};
    auto& encoder = fx::metal::get_command_encoder(stream());
    // Retain every referenced buffer even if encoding the second dispatch throws.
    encoder.add_temporary(x);
    encoder.add_temporary(scales);
    encoder.add_temporary(scratch);
    encoder.set_compute_pipeline_state(pipeline);
    encoder.set_input_array(x, 0);
    encoder.set_input_array(scales, 1);
    encoder.set_output_array(scratch, 2);
    encoder.set_bytes(bindings, 3);
    encoder.dispatch_threads(MTL::Size(x.size(), 1, 1), MTL::Size(128, 1, 1));
    // Same stream and MLX hazard tracking between the producer and consumer.
    hadamard_->eval_gpu({scratch}, outputs);
    static std::once_flag observed;
    std::call_once(observed, [] { std::fputs("MLXL3FlashPrepare128 encoded\n", stderr); });
  }

 private:
  std::shared_ptr<fx::Primitive> hadamard_;
};

struct FlashPrepareCache {
  std::shared_ptr<FlashPrepareProgram> program;
  uint64_t hits = 0, creations = 0, evictions = 0;
};

inline FlashPrepareCache& flash_prepare_cache() {
  thread_local FlashPrepareCache cache;
  return cache;
}

inline fx::array flash_prepare128(const fx::array& x, const fx::array& scales) {
  if (x.ndim() != 2 || x.shape() != scales.shape() || x.shape(0) < 1 || x.shape(0) > 20 ||
      (x.shape(1) != 640 && x.shape(1) != 2560) || scales.dtype() != fx::float16 ||
      (x.dtype() != fx::float16 && x.dtype() != fx::float32))
    throw std::invalid_argument("unsupported FlashPrepare128 inputs");
  auto s = fx::default_stream(fx::default_device());
  if (s.device.type != fx::Device::gpu) throw std::invalid_argument("FlashPrepare128 requires GPU");
  auto& cache = flash_prepare_cache();
  if (!cache.program || !(cache.program->stream() == s)) {
    auto next = std::make_shared<FlashPrepareProgram>(s);
    if (cache.program) ++cache.evictions;
    cache.program = std::move(next);
    ++cache.creations;
  } else {
    ++cache.hits;
  }
  return fx::array({int(x.size())}, fx::float16, cache.program, {x, scales});
}
} // namespace mlxl3
