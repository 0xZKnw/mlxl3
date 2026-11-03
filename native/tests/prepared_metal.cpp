#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include "prepared_metal.h"
#include "mlx/fast.h"
#include "mlx/ops.h"
#include "mlx/transforms.h"
#include <cassert>
#include <cmath>
#include <iostream>

namespace mx = mlx::core;

int main() {
  assert(!mlxl3::prepared_metal_option(nullptr));
  assert(!mlxl3::prepared_metal_option("0"));
  assert(mlxl3::prepared_metal_option("1"));
  for (const char* value : {"", "01", "true", "-1", "2", "1 "}) {
    bool rejected = false;
    try { mlxl3::prepared_metal_option(value); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  const std::string source =
      "uint i=thread_position_in_grid.x; if(i>=uint(x_shape[0]*x_shape[1])) return; "
      "a[i]=x[i]+x_ndim+x_shape[1]+x_strides[1]; b[i]=x[i]*2;";
  auto factory = mx::fast::metal_kernel("prepared_binding_test", {"x"}, {"a", "b"}, source);
  std::shared_ptr<mlxl3::PreparedMetalPlan> plan;
  std::vector<mx::array> pending;
  std::vector<std::vector<float>> expected;
  size_t cases = 0;
  // Cross the constant/device boundary and keep old graphs alive across plan replacement.
  for (int rows : {1, 2, 4, 8, 16, 32}) {
    for (auto type : {mx::float16, mx::float32}) {
      for (bool strided : {false, true}) {
        for (int revision : {0, 1, 2}) {
          std::vector<float> data(rows * 2);
          for (int i = 0; i < rows * 2; ++i) data[i] = float(i + revision * 13) / 4;
          auto x = mx::array(data.data(), {rows, 2});
          if (strided) x = mx::transpose(mx::reshape(x, {2, rows}));
          x = mx::astype(x, type);
          const std::vector<mx::Dtype> types{type, type};
          auto reference = factory({x}, {{rows, 2}, {rows, 2}}, types,
                                   {rows * 2, 1, 1}, {32, 1, 1}, {}, {}, false, {});
          auto actual = factory({x}, {{rows, 2}, {rows, 2}}, types,
                                {rows * 2, 1, 1}, {32, 1, 1}, {}, {}, false, {});
          int32_t grid[3]{rows * 2, 1, 1}, group[3]{32, 1, 1};
          mlxl3::prepare_metal(actual, plan, types, grid, group);
          assert(std::string(actual[0].primitive().name()) == "MLXL3PreparedMetal");
          mx::eval(reference);
          for (size_t out = 0; out < actual.size(); ++out) {
            auto ref = mx::astype(reference[out], mx::float32);
            ref.eval();
            std::vector<float> oracle(ref.data<float>(), ref.data<float>() + ref.size());
            assert(oracle.size() == data.size());
            for (int i = 0; i < rows * 2; ++i) {
              auto value = data[strided ? (i % 2) * rows + i / 2 : i];
              assert(oracle[i] == (out == 0 ? value + 5 : value * 2));
            }
            pending.push_back(actual[out]);
            expected.push_back(std::move(oracle));
          }
          ++cases;
        }
      }
    }
  }
  // Evaluate after replacing input arrays/plans, then in reverse sibling order.
  for (size_t i = pending.size(); i-- > 0;) {
    auto value = mx::astype(pending[i], mx::float32);
    value.eval();
    assert(value.size() == expected[i].size());
    for (size_t j = 0; j < value.size(); ++j)
      assert(std::isfinite(value.data<float>()[j]) && value.data<float>()[j] == expected[i][j]);
  }
  auto scalar_factory = mx::fast::metal_kernel("prepared_scalar", {"x"}, {"y"}, "y[0]=x+3;");
  std::shared_ptr<mlxl3::PreparedMetalPlan> scalar_plan;
  for (float value : {1.f, 2.f, -3.f}) {
    auto outputs = scalar_factory({mx::array(value)}, {{1}}, {mx::float32},
                                   {1, 1, 1}, {1, 1, 1}, {}, {}, false, {});
    int32_t grid[3]{1, 1, 1};
    mlxl3::prepare_metal(outputs, scalar_plan, {mx::float32}, grid, grid);
    assert(outputs[0].item<float>() == value + 3);
  }
  for (int dimension : {0, -1, INT32_MAX}) {
    auto outputs = scalar_factory({mx::array(1.f)}, {{1}}, {mx::float32},
                                   {1, 1, 1}, {1, 1, 1}, {}, {}, false, {});
    int32_t grid[3]{1, 1, 1}, group[3]{dimension, dimension, dimension};
    bool rejected = false;
    try { mlxl3::prepare_metal(outputs, scalar_plan, {mx::float32}, grid, group); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
  }
  std::cout << "PREPARED_QUALITY cases=" << cases << " outputs=" << pending.size()
            << " scalar=3 invalid=3 exact=true\n";
}
