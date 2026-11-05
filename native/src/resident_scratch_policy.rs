//! A retained scratch charge is independent of current process allocation.
use std::cell::Cell;

pub(crate) const CAP: u64 = 32 << 20;

pub(crate) fn reserve(used: u64, bytes: u64) -> Option<u64> {
    let next = used.checked_add(bytes)?;
    (bytes > 0 && next <= CAP).then_some(next)
}

pub(crate) fn eligible(
    enabled: bool,
    generic: bool,
    mtp: bool,
    routed: bool,
    batch_one: bool,
    private: bool,
    unsubmitted: bool,
) -> bool {
    enabled && !generic && !mtp && !routed && batch_one && private && unsubmitted
}

pub(crate) fn reusable(completed: bool, same_command: bool, exact_pair: bool) -> bool {
    completed && !same_command && exact_pair
}

pub(crate) struct Budget {
    pub enabled: Cell<bool>,
    charged: Cell<u64>,
    valid: Cell<bool>,
    #[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
    pub reuses: Cell<u64>,
}

impl Budget {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled: Cell::new(enabled),
            charged: Cell::new(0),
            valid: Cell::new(true),
            #[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
            reuses: Cell::new(0),
        }
    }
    pub fn charged(&self) -> u64 {
        self.charged.get()
    }
    pub fn valid(&self) -> bool {
        self.valid.get()
    }
    pub fn claim(&self, bytes: u64) -> bool {
        if !self.enabled.get() || !self.valid.get() {
            return false;
        }
        if let Some(next) = reserve(self.charged.get(), bytes) {
            self.charged.set(next);
            true
        } else {
            false
        }
    }
    pub fn refund(&self, bytes: u64) {
        if let Some(next) = self.charged.get().checked_sub(bytes) {
            self.charged.set(next);
        } else {
            self.valid.set(false);
        }
    }
}

#[test]
fn budget_keeps_disabled_cap_and_invalid_refund_truthful() {
    let off = Budget::new(false);
    assert!(!off.enabled.get() && !off.claim(1) && off.charged() == 0);
    let on = Budget::new(true);
    assert!(on.claim(CAP) && !on.claim(1));
    on.refund(CAP);
    assert!(on.valid() && on.charged() == 0 && on.claim(1));
    on.refund(2);
    assert!(!on.valid() && on.charged() == 1 && !on.claim(1));
    assert!(!reusable(false, false, true) && !reusable(true, true, true));
    assert!(eligible(true, false, false, false, true, true, true));
    for flags in [
        [false, false, false, false, true, true, true],
        [true, true, false, false, true, true, true],
        [true, false, true, false, true, true, true],
        [true, false, false, true, true, true, true],
        [true, false, false, false, false, true, true],
        [true, false, false, false, true, false, true],
        [true, false, false, false, true, true, false],
    ] {
        assert!(!eligible(
            flags[0], flags[1], flags[2], flags[3], flags[4], flags[5], flags[6]
        ));
    }
    let plan = crate::flash_memory::ResourcePlan::native_ar(1, 4096).unwrap();
    assert_eq!(
        plan.with_resident_scratch().unwrap().extra_bytes(),
        plan.extra_bytes() + CAP
    );
}

#[cfg(kani)]
#[kani::proof]
fn completed_slot_admission_and_reservation_are_bounded() {
    let used: u64 = kani::any();
    let bytes: u64 = kani::any();
    if let Some(next) = reserve(used, bytes) {
        assert!(next <= CAP && next >= used && bytes > 0);
        assert_eq!(next.checked_sub(bytes), Some(used));
    }
    let flags: [bool; 7] = kani::any();
    assert_eq!(
        eligible(
            flags[0], flags[1], flags[2], flags[3], flags[4], flags[5], flags[6]
        ),
        flags == [true, false, false, false, true, true, true]
    );
    let budget = Budget::new(true);
    if budget.claim(used) {
        assert!(budget.charged() == used && used <= CAP);
        if budget.claim(bytes) {
            assert_eq!(budget.charged(), used + bytes);
            budget.refund(bytes);
            assert!(budget.valid() && budget.charged() == used);
            kani::cover!(used == 1 && bytes == 1);
        }
        budget.enabled.set(false);
        assert!(!budget.claim(bytes) && budget.charged() == used);
        budget.enabled.set(true);
        budget.refund(used + 1);
        assert!(!budget.valid() && budget.charged() == used && !budget.claim(bytes));
    } else {
        assert!(budget.valid() && budget.charged() == 0);
    }
    let (complete, same, exact): (bool, bool, bool) = kani::any();
    let reuse = reusable(complete, same, exact);
    assert_eq!(reuse, complete && !same && exact);
    assert!(!reusable(false, true, exact)); // owner is replaced before encoding
    kani::cover!(reuse);
    kani::cover!(!reuse && complete && same);
    kani::cover!(reserve(used, bytes) == Some(CAP));
    kani::cover!(reserve(used, bytes).is_none() && used == CAP && bytes == 1);
    let positions: u16 = kani::any();
    if let Some(plan) = crate::flash_memory::ResourcePlan::native_ar(u32::from(positions), 4096) {
        assert_eq!(
            plan.with_resident_scratch().unwrap().extra_bytes(),
            plan.extra_bytes() + CAP
        );
        kani::cover!(positions == 4096);
    }
}
