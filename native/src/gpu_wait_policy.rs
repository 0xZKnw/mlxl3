//! The production poll loop uses both a time and an iteration bound.
pub(crate) fn may_spin(enabled: bool, elapsed_ns: u128, polls: u32) -> bool {
    enabled && elapsed_ns < 50_000 && polls < 4096
}

#[test]
fn polling_stops_at_either_bound_and_when_disabled() {
    assert!(may_spin(true, 0, 0));
    assert!(may_spin(true, 49_999, 4095));
    for (time, polls) in [(50_000, 0), (0, 4096), (u128::MAX, u32::MAX)] {
        assert!(!may_spin(true, time, polls));
    }
    assert!(!may_spin(false, 0, 0));
}

#[cfg(kani)]
#[kani::proof]
fn gpu_spin_budget_never_exceeds_either_production_limit() {
    let enabled: bool = kani::any();
    let elapsed: u128 = kani::any();
    let polls: u32 = kani::any();
    let actual = may_spin(enabled, elapsed, polls);
    assert_eq!(actual, enabled && elapsed <= 49_999 && polls <= 4095);
    kani::cover!(actual && elapsed == 49_999 && polls == 4095);
    kani::cover!(!actual && enabled && elapsed == 50_000 && polls == 0);
    kani::cover!(!actual && enabled && elapsed == 0 && polls == 4096);
    kani::cover!(!actual && !enabled && elapsed == 0 && polls == 0);
}

/// Wait for every owned command before retiring any owners, including on error.
/// The callback must return only after that command is terminal.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) fn retire_commands<T, E>(
    commands: &mut Vec<T>,
    mut wait: impl FnMut(&T) -> Result<(), E>,
) -> Result<(), E> {
    let mut failure = None;
    for command in commands.iter() {
        if let Err(error) = wait(command)
            && failure.is_none()
        {
            failure = Some(error);
        }
    }
    commands.clear();
    failure.map_or(Ok(()), Err)
}

#[test]
fn retirement_waits_all_commands_and_does_not_latch_failed_commands() {
    for fail in 0..5 {
        let mut commands = vec![0, 1, 2, 3];
        let mut waited = Vec::new();
        let result = retire_commands(&mut commands, |&id| {
            waited.push(id);
            if id >= fail { Err(id) } else { Ok(()) }
        });
        assert_eq!(waited, vec![0, 1, 2, 3]);
        assert!(commands.is_empty());
        assert_eq!(result, if fail < 4 { Err(fail) } else { Ok(()) });
        assert_eq!(retire_commands(&mut commands, |_| Err(99)), Ok(()));
    }
}

#[test]
fn retirement_keeps_every_owner_alive_until_all_waits_complete() {
    use std::{cell::Cell, rc::Rc};
    struct Owner(Rc<Cell<usize>>);
    impl Drop for Owner {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }
    let released = Rc::new(Cell::new(0));
    let mut commands: Vec<_> = (0..4).map(|_| Owner(released.clone())).collect();
    let mut waited = 0;
    let result = retire_commands(&mut commands, |_| {
        assert_eq!(released.get(), 0);
        waited += 1;
        Err(waited)
    });
    assert_eq!(result, Err(1));
    assert_eq!(waited, 4);
    assert_eq!(released.get(), 4);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(6)]
fn retirement_clears_all_terminal_commands_even_after_error() {
    let count: usize = kani::any();
    kani::assume(count <= 4);
    let failed: [bool; 4] = kani::any();
    let mut commands: Vec<usize> = (0..count).collect();
    let mut waited = 0;
    let result = retire_commands(&mut commands, |&id| {
        assert_eq!(id, waited);
        waited += 1;
        if failed[id] { Err(id) } else { Ok(()) }
    });
    assert_eq!(waited, count);
    assert!(commands.is_empty());
    assert_eq!(result.err(), (0..count).find(|&id| failed[id]));
    kani::cover!(count == 4 && failed[0] && !failed[3]);
}

/// A failed GPU producer makes its row unusable until a fresh model is loaded.
/// Shared with outputs so draining again cannot erase an observed failure.
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
#[derive(Default)]
pub(crate) struct CommandHealth(std::cell::Cell<bool>);

#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
impl CommandHealth {
    pub(crate) fn observe(&self, success: bool) -> bool {
        self.0.set(self.0.get() || !success);
        !self.0.get()
    }
}

#[test]
fn failed_producer_requires_reload_after_all_owners_retire() {
    use std::rc::Rc;
    for failed in 0..4 {
        let row = Rc::new(CommandHealth::default());
        let output = row.clone();
        let mut commands = vec![0, 1, 2, 3];
        let mut waited = Vec::new();
        let result = retire_commands(&mut commands, |&id| {
            waited.push(id);
            if id == failed { Err(id) } else { Ok(()) }
        });
        assert!(!output.observe(result.is_ok()));
        assert_eq!(waited, vec![0, 1, 2, 3]);
        assert!(commands.is_empty());
        let retry: Result<(), ()> = retire_commands(&mut commands, |_| Ok(()));
        assert!(!row.observe(retry.is_ok()));
        assert!(CommandHealth::default().observe(true));
    }
}

#[cfg(kani)]
#[kani::proof]
fn failed_command_health_cannot_be_cleared_by_success() {
    let first: bool = kani::any();
    let second: bool = kani::any();
    let health = CommandHealth::default();
    assert_eq!(health.observe(first), first);
    assert_eq!(health.observe(second), first && second);
    assert_eq!(health.observe(true), first && second);
    kani::cover!(!first && second);
    kani::cover!(first && second);
}
