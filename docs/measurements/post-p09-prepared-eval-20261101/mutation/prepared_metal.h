// MLX 0.32.2 binding order follows Apple Inc.'s MIT-licensed custom_kernel.cpp.
#pragma once

#include "mlx/allocator.h"
#include "mlx/backend/metal/device.h"
#include "mlx/fast_primitives.h"
#include <cstring>
#include <limits>
#include <mutex>
#include <sstream>

namespace mlxl3 {
namespace pmx = mlx::core;

inline bool prepared_metal_option(const char* value) {
  if (!value || std::strcmp(value, "0") == 0) return false;
  if (std::strcmp(value, "1") == 0) return true;
  throw std::invalid_argument("MLXL3_PREPARED_METAL must be 0 or 1");
}

struct PreparedMetalPlan {
  using Signature = std::tuple<pmx::Dtype, bool, bool>;
  std::vector<Signature> inputs;
  std::vector<pmx::Dtype> outputs;
  std::vector<std::tuple<bool, bool, bool>> shape_infos;
  pmx::Device device;
  std::string name, library;
  std::once_flag initialized;
  NS::SharedPtr<MTL::ComputePipelineState> pipeline;

  PreparedMetalPlan(pmx::fast::CustomKernel& kernel,
                    const std::vector<pmx::array>& in,
                    const std::vector<pmx::Dtype>& out)
      : outputs(out), device(kernel.stream().device) {
    auto state = kernel.state();
    if (!std::get<5>(state) || std::get<6>(state) ||
        !std::get<7>(state).empty() || std::get<8>(state) || std::get<9>(state))
      throw std::invalid_argument("unsupported prepared Metal state");
    name = std::get<0>(state);
    std::ostringstream key;
    key << name << '_' << std::hex << std::hash<std::string>{}(std::get<1>(state))
        << '_' << std::dec << std::get<10>(state);
    library = key.str();
    shape_infos = std::move(std::get<4>(state));
    for (const auto& x : in) inputs.push_back(signature(x));
  }

  static Signature signature(const pmx::array& x) {
    return {x.dtype(), x.ndim() == 0, x.size() < 8};
  }

  bool matches(const std::vector<pmx::array>& in,
               const std::vector<pmx::Dtype>& out, pmx::Device d) const {
    if (d != device || in.size() != inputs.size() || out != outputs) return false;
    for (size_t i = 0; i < in.size(); ++i)
      if (signature(in[i]) != inputs[i]) return false;
    return true;
  }
};

class PreparedMetal : public pmx::Primitive {
 public:
  PreparedMetal(std::shared_ptr<pmx::Primitive> original,
                std::shared_ptr<PreparedMetalPlan> plan,
                const int32_t* grid, const int32_t* group)
      : Primitive(original->stream()), original_(std::move(original)),
        plan_(std::move(plan)), grid_(grid[0], grid[1], grid[2]),
        group_(std::min(grid[0], group[0]), std::min(grid[1], group[1]),
               std::min(grid[2], group[2])),
        threads_(uint64_t(group[0]) * group[1] * group[2]) {}

  const char* name() const override { return "MLXL3PreparedMetal"; }
  void eval_cpu(const std::vector<pmx::array>&, std::vector<pmx::array>&) override {
    throw std::runtime_error("Custom kernels only run on GPU.");
  }

  void eval_gpu(const std::vector<pmx::array>& inputs,
                std::vector<pmx::array>& outputs) override {
    bool first = false;
    std::call_once(plan_->initialized, [&] {
      // Let MLX compile the exact source/options and retain its pipeline afterwards.
      original_->eval_gpu(inputs, outputs);
      auto& device = pmx::metal::device(stream().device);
      auto library = device.get_library(plan_->library);
      plan_->pipeline = NS::RetainPtr(device.get_kernel(plan_->name, library));
      first = true;
    });
    if (first) return;
    for (const auto& in : inputs) {
      if (!in.flags().row_contiguous) {
        original_->eval_gpu(inputs, outputs);
        return;
      }
    }
    if (threads_ > plan_->pipeline->maxTotalThreadsPerThreadgroup())
      throw std::invalid_argument("prepared Metal threadgroup exceeds pipeline limit");
    for (auto& out : outputs) out.set_data(pmx::allocator::malloc(out.nbytes()));
    auto& encoder = pmx::metal::get_command_encoder(stream());
    encoder.set_compute_pipeline_state(plan_->pipeline.get());
    int index = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
      const auto& in = inputs[i];
      encoder.set_input_array(in, index++);
      if (in.ndim() == 0) continue;
      const auto& info = plan_->shape_infos[i];
      if (std::get<0>(info)) encoder.set_vector_bytes(in.shape(), in.ndim(), index++);
      if (std::get<1>(info)) encoder.set_vector_bytes(in.strides(), in.ndim(), index++);
      if (std::get<2>(info)) encoder.set_bytes(in.ndim() + 1, index++);
    }
    for (auto& out : outputs) encoder.set_output_array(out, index++);
    encoder.dispatch_threads(grid_, group_);
  }

 private:
  std::shared_ptr<pmx::Primitive> original_;
  std::shared_ptr<PreparedMetalPlan> plan_;
  MTL::Size grid_, group_;
  uint64_t threads_;
};

// Called only on fresh arrays produced by the pinned MLX metal_kernel factory.
inline void prepare_metal(std::vector<pmx::array>& outputs,
                          std::shared_ptr<PreparedMetalPlan>& plan,
                          const std::vector<pmx::Dtype>& types,
                          const int32_t* grid, const int32_t* group) {
  uint64_t threads = 1;
  for (int i = 0; i < 3; ++i) {
    if (grid[i] <= 0 || group[i] <= 0 ||
        threads > std::numeric_limits<uint64_t>::max() / uint64_t(group[i]))
      throw std::invalid_argument("invalid prepared Metal dispatch dimensions");
    threads *= uint64_t(group[i]);
  }
  if (outputs.empty() || !outputs[0].has_primitive() ||
      std::strcmp(outputs[0].primitive().name(), "CustomKernel") != 0)
    throw std::invalid_argument("prepared Metal requires a fresh CustomKernel");
  auto original = outputs[0].primitive_ptr();
  auto& kernel = static_cast<pmx::fast::CustomKernel&>(*original);
  const auto& inputs = outputs[0].inputs();
  if (!plan || !plan->matches(inputs, types, original->stream().device))
    plan = std::make_shared<PreparedMetalPlan>(kernel, inputs, types);
  auto prepared = std::make_shared<PreparedMetal>(original, plan, grid, group);
  for (auto& out : outputs) out.primitive_ptr() = prepared;
}
} // namespace mlxl3
