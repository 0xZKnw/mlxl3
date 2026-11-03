#pragma once

#include <array>
#include <cstdint>
#include <mach/mach.h>
#include <sys/resource.h>

namespace mlxl3 {
constexpr size_t memory_stats_values = 12;
constexpr uint64_t memory_usage_valid = 1;
constexpr uint64_t memory_compression_valid = 2;
constexpr uint64_t memory_decompressions_valid = 4;
constexpr uint64_t memory_swapins_valid = 8;

inline std::array<uint64_t, 9> process_memory_values(
    const rusage_info_v4& usage, bool usage_ok, const task_vm_info_data_t& vm,
    mach_msg_type_number_t count, bool vm_ok) {
  std::array<uint64_t, 9> values{};
  auto& valid = values[8];
  if (usage_ok) {
    values[0] = usage.ri_phys_footprint;
    values[1] = usage.ri_lifetime_max_phys_footprint;
    values[7] = usage.ri_proc_start_abstime;
    valid |= memory_usage_valid;
  }
  if (vm_ok && count >= TASK_VM_INFO_REV0_COUNT) {
    values[2] = vm.compressed;
    values[3] = vm.compressed_peak;
    values[4] = vm.compressed_lifetime;
    valid |= memory_compression_valid;
  }
  if (vm_ok && count >= TASK_VM_INFO_REV5_COUNT && vm.decompressions >= 0) {
    values[5] = static_cast<uint64_t>(vm.decompressions);
    valid |= memory_decompressions_valid;
  }
  if (vm_ok && count >= TASK_VM_INFO_REV6_COUNT && vm.ledger_swapins >= 0) {
    values[6] = static_cast<uint64_t>(vm.ledger_swapins);
    valid |= memory_swapins_valid;
  }
  return values;
}
}  // namespace mlxl3
