//! Keep a fallible operation and its diagnostic export in one rollback scope.
pub fn with_rollback<S, P, T, E>(
    state: &mut S,
    before: &P,
    operation: impl FnOnce(&mut S) -> Result<T, E>,
    restore: impl FnOnce(&mut S, &P) -> Result<(), E>,
) -> Result<T, E> {
    let result = operation(state);
    if result.is_err() {
        restore(state, before)?;
    }
    result
}

#[test]
fn export_failure_rolls_back_and_retry_consumes_exactly_once() {
    for fail_run in [false, true] {
        let mut state = (4, vec![1, 2], vec![3]);
        let before = state.clone();
        let result = with_rollback(
            &mut state,
            &before,
            |s| {
                s.0 += 1;
                s.1.push(9);
                if fail_run {
                    return Err("run");
                }
                s.2.push(8);
                Err::<(), _>("export")
            },
            |s, before| {
                *s = before.clone();
                Ok(())
            },
        );
        assert_eq!(result, Err(if fail_run { "run" } else { "export" }));
        assert_eq!(state, before);
        with_rollback(
            &mut state,
            &before,
            |s| {
                s.0 += 1;
                s.1.push(9);
                s.2.push(8);
                Ok::<_, &str>(())
            },
            |s, before| {
                *s = before.clone();
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(state, (5, vec![1, 2, 9], vec![3, 8]));
    }
}

#[cfg(kani)]
#[kani::proof]
fn failed_export_restores_snapshot() {
    let before: u32 = kani::any();
    let changed: u32 = kani::any();
    let failure: bool = kani::any();
    let mut state = before;
    let result = with_rollback(
        &mut state,
        &before,
        |s| {
            *s = changed;
            if failure { Err(()) } else { Ok(()) }
        },
        |s, before| {
            *s = *before;
            Ok(())
        },
    );
    assert_eq!(result.is_err(), failure);
    assert_eq!(state, if failure { before } else { changed });
    kani::cover!(failure && before != changed);
}
