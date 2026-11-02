//! Test-only host construction intervals; these are not GPU dispatch timings.

#[cfg(test)]
use std::cell::Cell;

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    Coarse,
    Fine,
}

#[cfg(test)]
pub(crate) const TARGET: &str = "mlxl3_rs_expert_2560_640_2_2_1_4_1_slots1_resident53";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Stats {
    pub(crate) valid: bool,
    pub(crate) calls: u64,
    /// validation, cstrings, marshalling, initialize, ffi, adoption, total
    pub(crate) ns: [u64; 7],
}

impl Default for Stats {
    fn default() -> Self {
        Self {
            valid: true,
            calls: 0,
            ns: [0; 7],
        }
    }
}

impl Stats {
    /// On failure, retain every counter and mark the whole sample invalid.
    pub(crate) fn checked_record(&mut self, delta: [u64; 7]) -> bool {
        if !self.valid {
            return false;
        }
        let Some(calls) = self.calls.checked_add(1) else {
            self.valid = false;
            return false;
        };
        let mut ns = self.ns;
        for (value, addition) in ns.iter_mut().zip(delta) {
            let Some(sum) = value.checked_add(addition) else {
                self.valid = false;
                return false;
            };
            *value = sum;
        }
        self.calls = calls;
        self.ns = ns;
        true
    }
}

#[cfg(test)]
thread_local! {
    static PROFILE: Cell<(Mode, Stats)> = const {
        Cell::new((Mode::Off, Stats { valid: true, calls: 0, ns: [0; 7] }))
    };
}

#[cfg(test)]
pub(crate) fn configure(mode: Mode) {
    PROFILE.set((mode, Stats::default()));
}

#[cfg(test)]
pub(crate) fn mode_for(name: &str) -> Mode {
    if name == TARGET {
        PROFILE.get().0
    } else {
        Mode::Off
    }
}

#[cfg(test)]
pub(crate) fn snapshot() -> Stats {
    PROFILE.get().1
}

#[cfg(test)]
pub(crate) fn record(mut delta: [u64; 7]) -> bool {
    let (mode, mut stats) = PROFILE.get();
    match mode {
        Mode::Off => return false,
        Mode::Coarse => delta[..6].fill(0),
        Mode::Fine => {}
    }
    let recorded = stats.checked_record(delta);
    PROFILE.set((mode, stats));
    recorded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_record_accumulates_all_fields_and_accepts_exact_limits() {
        let mut stats = Stats::default();
        assert!(stats.checked_record([0; 7]));
        assert_eq!(stats.calls, 1);
        assert_eq!(stats.ns, [0; 7]);
        assert!(stats.checked_record([1, 2, 3, 4, 5, 6, 21]));
        assert!(stats.checked_record([7, 6, 5, 4, 3, 2, 27]));
        assert_eq!(stats.calls, 3);
        assert_eq!(stats.ns, [8, 8, 8, 8, 8, 8, 48]);
        assert!(stats.valid);

        let mut limits = Stats {
            valid: true,
            calls: u64::MAX - 1,
            ns: [u64::MAX - 1; 7],
        };
        assert!(limits.checked_record([1; 7]));
        assert_eq!(limits.calls, u64::MAX);
        assert_eq!(limits.ns, [u64::MAX; 7]);
        assert!(limits.valid);
    }

    #[test]
    fn overflow_is_atomic_for_calls_and_each_phase_and_stays_invalid() {
        for overflow in 0..8 {
            let mut stats = Stats {
                valid: true,
                calls: 17,
                ns: [2, 3, 4, 5, 6, 7, 8],
            };
            if overflow == 7 {
                stats.calls = u64::MAX;
            } else {
                stats.ns[overflow] = u64::MAX;
            }
            let before = stats;
            assert!(!stats.checked_record([1; 7]));
            assert_eq!(stats.calls, before.calls);
            assert_eq!(stats.ns, before.ns);
            assert!(!stats.valid);
            let failed = stats;
            assert!(!stats.checked_record([0; 7]));
            assert!(!stats.checked_record([u64::MAX; 7]));
            assert_eq!(stats, failed);
        }
    }

    #[test]
    fn modes_gate_the_exact_target_and_reset_every_counter() {
        for mode in [Mode::Off, Mode::Coarse, Mode::Fine] {
            configure(mode);
            assert_eq!(snapshot(), Stats::default());
            assert_eq!(mode_for(TARGET), mode);
            for name in ["", "mlxl3_rs_expert_2560_640", "other_kernel"] {
                assert_eq!(mode_for(name), Mode::Off);
            }
            assert_eq!(mode_for(&format!("{TARGET}_other")), Mode::Off);
            assert_eq!(mode_for(&format!("x{TARGET}")), Mode::Off);
            assert_eq!(record([1, 2, 3, 4, 5, 6, 21]), mode != Mode::Off);
            let stats = snapshot();
            assert!(stats.valid);
            match mode {
                Mode::Off => assert_eq!(stats, Stats::default()),
                Mode::Coarse => {
                    assert_eq!(stats.calls, 1);
                    assert_eq!(stats.ns, [0, 0, 0, 0, 0, 0, 21]);
                }
                Mode::Fine => {
                    assert_eq!(stats.calls, 1);
                    assert_eq!(stats.ns, [1, 2, 3, 4, 5, 6, 21]);
                }
            }
        }
        PROFILE.set((
            Mode::Fine,
            Stats {
                valid: false,
                calls: u64::MAX,
                ns: [u64::MAX; 7],
            },
        ));
        assert!(!record([0; 7]));
        assert!(!snapshot().valid);
        configure(Mode::Fine);
        assert_eq!(snapshot(), Stats::default());
        assert!(record([0; 7]));
        configure(Mode::Off);
    }

    #[test]
    fn configuration_and_snapshots_are_thread_local() {
        configure(Mode::Fine);
        assert!(record([1, 2, 3, 4, 5, 6, 21]));
        let before = snapshot();
        std::thread::spawn(|| {
            assert_eq!(mode_for(TARGET), Mode::Off);
            assert_eq!(snapshot(), Stats::default());
            configure(Mode::Coarse);
            assert!(record([0, 0, 0, 0, 0, 0, 99]));
            assert_eq!(snapshot().ns[6], 99);
        })
        .join()
        .expect("collector thread completes");
        assert_eq!(mode_for(TARGET), Mode::Fine);
        assert_eq!(snapshot(), before);
        configure(Mode::Off);
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(129)]
fn kernel_build_profile_records_checked_counters_atomically() {
    let before = Stats {
        valid: kani::any(),
        calls: kani::any(),
        ns: kani::any(),
    };
    let delta: [u64; 7] = kani::any();
    let wide_calls = u128::from(before.calls) + 1;
    let mut expected = before.valid && wide_calls <= u128::from(u64::MAX);
    for index in 0..7 {
        expected &= u128::from(before.ns[index]) + u128::from(delta[index]) <= u128::from(u64::MAX);
    }
    let mut actual = before;
    let accepted = actual.checked_record(delta);
    assert_eq!(accepted, expected);
    assert_eq!(actual.valid, expected);
    if accepted {
        assert_eq!(u128::from(actual.calls), wide_calls);
        for index in 0..7 {
            assert_eq!(
                u128::from(actual.ns[index]),
                u128::from(before.ns[index]) + u128::from(delta[index])
            );
        }
    } else {
        assert_eq!(actual.calls, before.calls);
        for index in 0..7 {
            assert_eq!(actual.ns[index], before.ns[index]);
        }
        let later: [u64; 7] = kani::any();
        assert!(!actual.checked_record(later));
        assert!(!actual.valid);
        assert_eq!(actual.calls, before.calls);
        for index in 0..7 {
            assert_eq!(actual.ns[index], before.ns[index]);
        }
    }
    kani::cover!(accepted);
    kani::cover!(!before.valid && !accepted);
    kani::cover!(before.valid && before.calls == u64::MAX && !accepted);
    for index in 0..7 {
        kani::cover!(
            before.valid
                && before.calls < u64::MAX
                && before.ns[index] == u64::MAX
                && delta[index] == 1
                && !accepted
        );
    }
}
