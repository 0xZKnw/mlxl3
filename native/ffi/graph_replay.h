#pragma once
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
#include "mlx/array.h"

namespace mlxl3 {
// Replay the original primitives without changing their arithmetic or capturing mutable roots.
class GraphReplay {
  using Array = mlx::core::array;
  struct Node {
    std::vector<size_t> inputs, outputs;
    std::vector<mlx::core::Shape> shapes;
    std::vector<mlx::core::Dtype> dtypes;
    std::shared_ptr<mlx::core::Primitive> primitive;
  };
  std::vector<mlx::core::Shape> input_shapes_;
  std::vector<mlx::core::Dtype> input_dtypes_;
  std::vector<std::pair<size_t, Array>> constants_;
  std::vector<Node> nodes_;
  std::vector<size_t> results_;
  size_t slots_ = 0;
  size_t slot() {
    if (slots_ >= 8192) throw std::invalid_argument("replay graph exceeds slot bound");
    return slots_++;
  }
 public:
  GraphReplay(const std::vector<Array>& roots, const std::vector<Array>& outputs) {
    if (roots.empty() || roots.size() > 4 || outputs.empty() || outputs.size() > 4)
      throw std::invalid_argument("replay requires 1..4 inputs and outputs");
    std::map<uintptr_t, size_t> seen;
    for (const auto& a : roots) {
      if (!seen.emplace(a.id(), slot()).second) throw std::invalid_argument("duplicate replay root");
      input_shapes_.push_back(a.shape()); input_dtypes_.push_back(a.dtype());
    }
    std::function<size_t(const Array&, size_t)> visit = [&](const Array& a, size_t depth) -> size_t {
      if (auto it = seen.find(a.id()); it != seen.end()) return it->second;
      if (depth > 256) throw std::invalid_argument("replay graph depth or cycle");
      if (!a.has_primitive()) {
        auto index = slot(); seen.emplace(a.id(), index); constants_.emplace_back(index, a);
        return index;
      }
      Node node;
      for (const auto& input : a.inputs()) node.inputs.push_back(visit(input, depth + 1));
      node.primitive = a.primitive_ptr();
      for (const auto& output : a.outputs()) {
        if (seen.count(output.id())) throw std::invalid_argument("replay sibling crosses root boundary");
        auto index = slot(); seen.emplace(output.id(), index); node.outputs.push_back(index);
        node.shapes.push_back(output.shape()); node.dtypes.push_back(output.dtype());
      }
      nodes_.push_back(std::move(node));
      return seen.at(a.id());
    };
    for (const auto& output : outputs) results_.push_back(visit(output, 0));
  }
  std::vector<Array> apply(const std::vector<Array>& inputs) const {
    if (inputs.size() != input_shapes_.size()) throw std::invalid_argument("replay input count differs");
    std::vector<std::optional<Array>> values(slots_);
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (inputs[i].shape() != input_shapes_[i] || inputs[i].dtype() != input_dtypes_[i])
        throw std::invalid_argument("replay input shape or dtype differs");
      values.at(i) = inputs[i];
    }
    for (const auto& [index, a] : constants_) values.at(index) = a;
    for (const auto& node : nodes_) {
      std::vector<Array> args;
      for (auto index : node.inputs) args.push_back(values.at(index).value());
      auto outputs = Array::make_arrays(node.shapes, node.dtypes, node.primitive, args);
      if (outputs.size() != node.outputs.size()) throw std::logic_error("replay primitive output count differs");
      for (size_t i = 0; i < outputs.size(); ++i) values.at(node.outputs[i]) = std::move(outputs[i]);
    }
    std::vector<Array> outputs;
    for (auto index : results_) outputs.push_back(values.at(index).value());
    return outputs;
  }
};
}  // namespace mlxl3
