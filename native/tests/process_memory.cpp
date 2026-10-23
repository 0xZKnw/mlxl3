#include "../ffi/process_memory.h"
#include <cassert>
#include <climits>

int main() {
  rusage_info_v4 usage{};
  task_vm_info_data_t vm{};
  usage.ri_phys_footprint = 101;
  usage.ri_lifetime_max_phys_footprint = 102;
  usage.ri_proc_start_abstime = 103;
  vm.compressed = 201;
  vm.compressed_peak = 202;
  vm.compressed_lifetime = 203;
  vm.decompressions = 301;
  vm.ledger_swapins = 401;
  const mach_msg_type_number_t counts[] = {0, TASK_VM_INFO_REV0_COUNT - 1,
      TASK_VM_INFO_REV0_COUNT, TASK_VM_INFO_REV5_COUNT - 1,
      TASK_VM_INFO_REV5_COUNT, TASK_VM_INFO_REV6_COUNT - 1,
      TASK_VM_INFO_REV6_COUNT, TASK_VM_INFO_COUNT};
  unsigned cases = 0;
  for (auto count : counts) {
    for (bool usage_ok : {false, true}) {
      for (bool vm_ok : {false, true}) {
        auto v = mlxl3::process_memory_values(usage, usage_ok, vm, count, vm_ok);
        const bool compression = vm_ok && count >= TASK_VM_INFO_REV0_COUNT;
        const bool decompressions = vm_ok && count >= TASK_VM_INFO_REV5_COUNT;
        const bool swapins = vm_ok && count >= TASK_VM_INFO_REV6_COUNT;
        assert(v[0] == (usage_ok ? 101 : 0));
        assert(v[1] == (usage_ok ? 102 : 0));
        assert(v[2] == (compression ? 201 : 0));
        assert(v[3] == (compression ? 202 : 0));
        assert(v[4] == (compression ? 203 : 0));
        assert(v[5] == (decompressions ? 301 : 0));
        assert(v[6] == (swapins ? 401 : 0));
        assert(v[7] == (usage_ok ? 103 : 0));
        assert(v[8] == (uint64_t(usage_ok) | (uint64_t(compression) << 1) |
                       (uint64_t(decompressions) << 2) | (uint64_t(swapins) << 3)));
        ++cases;
      }
    }
  }
  vm.decompressions = -1;
  vm.ledger_swapins = -1;
  auto negative = mlxl3::process_memory_values(usage, true, vm, TASK_VM_INFO_COUNT, true);
  assert(negative[5] == 0 && negative[6] == 0 && negative[8] == 3);
  vm.decompressions = INT_MAX;
  vm.ledger_swapins = INT64_MAX;
  auto maximum = mlxl3::process_memory_values(usage, true, vm, TASK_VM_INFO_COUNT, true);
  assert(maximum[5] == INT_MAX && maximum[6] == INT64_MAX && maximum[8] == 15);
  auto zero = mlxl3::process_memory_values({}, true, {}, TASK_VM_INFO_COUNT, true);
  assert(zero[8] == 15);
  for (size_t i = 0; i < 8; ++i) assert(zero[i] == 0);
  assert(cases == 32);
}
