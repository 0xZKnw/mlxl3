//! CPU-only contracts for optional Metal API availability.
//!
//! The native bridge reports runtime OS availability, not GPU generation. Keep
//! those separate: M5 identity also controls already qualified QMV layouts.
pub(crate) const COOPERATIVE_INPUTS: u32 = 1;
pub(crate) const NATIVE_Q4: u32 = 2;

pub(crate) fn exl3_tensor_ops(flags: u32, is_m5: bool) -> bool {
    flags & COOPERATIVE_INPUTS != 0 && is_m5
}

pub(crate) fn native_q4(flags: u32) -> bool {
    flags & NATIVE_Q4 != 0
}

#[test]
fn optional_metal_apis_require_their_own_runtime_capability() {
    // macOS 26.2, 26.3 and 26.4 respectively. Unknown flags cannot enable
    // either feature, and a newer OS does not turn an older GPU into M5.
    for (flags, cooperative, q4) in [
        (0, false, false),
        (COOPERATIVE_INPUTS, true, false),
        (COOPERATIVE_INPUTS | NATIVE_Q4, true, true),
        (NATIVE_Q4, false, true),
        (!3, false, false),
        (u32::MAX, true, true),
    ] {
        assert_eq!(exl3_tensor_ops(flags, true), cooperative);
        assert!(!exl3_tensor_ops(flags, false));
        assert_eq!(native_q4(flags), q4);
    }
}

#[cfg(kani)]
#[kani::proof]
fn optional_metal_apis_never_bypass_their_runtime_requirement() {
    let flags: u32 = kani::any();
    let is_m5: bool = kani::any();
    assert_eq!(exl3_tensor_ops(flags, is_m5), flags & 1 != 0 && is_m5);
    assert_eq!(native_q4(flags), flags & 2 != 0);
    kani::cover!(exl3_tensor_ops(flags, is_m5));
    kani::cover!(!exl3_tensor_ops(flags, is_m5) && is_m5);
    kani::cover!(native_q4(flags));
    kani::cover!(!native_q4(flags));
}
