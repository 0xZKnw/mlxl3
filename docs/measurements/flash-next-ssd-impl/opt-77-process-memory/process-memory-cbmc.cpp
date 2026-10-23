#include "process_memory.h"
#include <cassert>

extern uint64_t nondet_u64();
extern int64_t nondet_i64();
extern int32_t nondet_i32();
extern uint32_t nondet_u32();
extern bool nondet_bool();

int main() {
  rusage_info_v4 usage{};
  task_vm_info_data_t vm{};
  usage.ri_phys_footprint = nondet_u64();
  usage.ri_lifetime_max_phys_footprint = nondet_u64();
  usage.ri_proc_start_abstime = nondet_u64();
  vm.compressed = nondet_u64();
  vm.compressed_peak = nondet_u64();
  vm.compressed_lifetime = nondet_u64();
  vm.decompressions = nondet_i32();
  vm.ledger_swapins = nondet_i64();
  auto count = nondet_u32();
  auto usage_ok = nondet_bool();
  auto vm_ok = nondet_bool();
  const auto values = mlxl3::process_memory_values(usage, usage_ok, vm, count, vm_ok);
  assert(bool(values[8] & 1) == usage_ok);
  assert(bool(values[8] & 2) == (vm_ok && count >= TASK_VM_INFO_REV0_COUNT));
  assert(bool(values[8] & 4) == (vm_ok && count >= TASK_VM_INFO_REV5_COUNT && vm.decompressions >= 0));
  assert(bool(values[8] & 8) == (vm_ok && count >= TASK_VM_INFO_REV6_COUNT && vm.ledger_swapins >= 0));
  assert((values[8] & ~uint64_t(15)) == 0);
  if (values[8] & 1) {
    assert(values[0] == usage.ri_phys_footprint);
    assert(values[1] == usage.ri_lifetime_max_phys_footprint);
    assert(values[7] == usage.ri_proc_start_abstime);
  }
  if (values[8] & 2) {
    assert(values[2] == vm.compressed);
    assert(values[3] == vm.compressed_peak);
    assert(values[4] == vm.compressed_lifetime);
  }
  if (values[8] & 4) assert(values[5] == uint64_t(vm.decompressions));
  if (values[8] & 8) assert(values[6] == uint64_t(vm.ledger_swapins));
}
