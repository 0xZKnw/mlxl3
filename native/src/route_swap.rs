//! AR diagnostic policy only; not a production or MTP routing contract.
use anyhow::{Context, Result, ensure};

fn recovery(current: u8, missed: bool, swapped: bool) -> u8 {
    if !missed {
        current
    } else if swapped && current < 3 {
        current + 1
    } else {
        0
    }
}

/// Commit already selected IDs without sorting probabilities or choosing again.
pub(crate) fn apply_choices(
    original: &[u32],
    picked: &[u32],
    ready: &[bool; 512],
    starved: &mut [u8; 512],
) -> Option<usize> {
    if original.len() != 10 || picked.len() != 10 {
        return None;
    }
    let mut changed = 0;
    for (rank, (&old, &new)) in original.iter().zip(picked).enumerate() {
        if old >= 512
            || new >= 512
            || original[..rank].contains(&old)
            || picked[..rank].contains(&new)
        {
            return None;
        }
        if old != new {
            if ready[old as usize]
                || !ready[new as usize]
                || original.contains(&new)
                || starved[old as usize] >= 3
            {
                return None;
            }
            changed += 1;
        }
    }
    if changed > 3 {
        return None;
    }
    for (&old, &new) in original.iter().zip(picked) {
        let e = old as usize;
        starved[e] = recovery(starved[e], !ready[e], old != new);
    }
    Some(changed)
}

/// Bounded CPU block picker: retain only the three useful candidates, on the stack.
pub(crate) fn select_fast(
    ids: &[u32],
    p: &[f32],
    weights: &[f32],
    ready: &[bool; 512],
    starved: &mut [u8; 512],
    tau: f32,
    limit: usize,
) -> Result<Vec<(usize, u32)>> {
    ensure!(tau.is_finite() && (0.0..=0.4).contains(&tau) && (1..=3).contains(&limit));
    ensure!(ids.len() == 10 && weights.len() == 10 && p.len() == 512);
    ensure!(
        ids.iter()
            .enumerate()
            .all(|(i, &e)| e < 512 && !ids[..i].contains(&e))
    );
    ensure!(p.iter().chain(weights).all(|v| v.is_finite() && *v >= 0.0));
    let mut protected_order = std::array::from_fn::<_, 10, _>(|i| i);
    protected_order.sort_unstable_by(|&a, &b| {
        p[ids[b] as usize]
            .total_cmp(&p[ids[a] as usize])
            .then(ids[a].cmp(&ids[b]))
    });
    let mut order = std::array::from_fn::<_, 10, _>(|i| i);
    order.sort_by(|&a, &b| p[ids[b] as usize].total_cmp(&p[ids[a] as usize]));
    let mut candidates = [u32::MAX; 3];
    for e in 0..512u32 {
        if !ready[e as usize] || ids.contains(&e) {
            continue;
        }
        for at in 0..3 {
            let prior = candidates[at];
            if prior == u32::MAX
                || p[e as usize] > p[prior as usize]
                || (p[e as usize] == p[prior as usize] && e < prior)
            {
                for j in (at + 1..3).rev() {
                    candidates[j] = candidates[j - 1];
                }
                candidates[at] = e;
                break;
            }
        }
    }
    let mut selected: [u32; 10] = ids.try_into().expect("checked ten original IDs");
    let mut picks = Vec::with_capacity(3);
    for rank in order {
        let e = ids[rank] as usize;
        if ready[e]
            || tau == 0.0
            || protected_order[..3].contains(&rank)
            || starved[e] >= 3
            || picks.len() >= limit
            || !(weights[rank] > 0.0 && weights[rank] <= 0.1 && p[e] > 0.0)
        {
            continue;
        }
        let candidate = candidates[picks.len()];
        if candidate < 512 && p[candidate as usize] >= (1.0 - tau) * p[e] {
            selected[rank] = candidate;
            picks.push((rank, candidate));
        }
    }
    apply_choices(ids, &selected, ready, starved).context("invalid bounded Q04 choices")?;
    Ok(picks)
}

pub(crate) fn select(
    ids: &[u32],
    probabilities: &[f32],
    weights: &[f32],
    ready: &[bool; 512],
    starved: &mut [u8; 512],
    tau: f32,
    limit: usize,
) -> Result<Vec<(usize, u32)>> {
    ensure!(tau.is_finite() && (0.0..=0.4).contains(&tau) && (1..=3).contains(&limit));
    ensure!(ids.len() == 10 && weights.len() == 10 && probabilities.len() == 512);
    ensure!(
        ids.iter()
            .enumerate()
            .all(|(i, &e)| e < 512 && !ids[..i].contains(&e))
    );
    ensure!(probabilities.iter().all(|p| p.is_finite() && *p >= 0.0));
    ensure!(weights.iter().all(|p| p.is_finite() && *p >= 0.0));
    let mut ranked: Vec<_> = (0..10).collect();
    ranked.sort_by(|&a, &b| {
        probabilities[ids[b] as usize]
            .total_cmp(&probabilities[ids[a] as usize])
            .then(ids[a].cmp(&ids[b]))
    });
    let protected = ranked[..3].to_vec();
    let mut order: Vec<_> = (0..10).collect();
    order.sort_by(|&a, &b| {
        probabilities[ids[b] as usize].total_cmp(&probabilities[ids[a] as usize])
    });
    let mut candidates: Vec<_> = (0..512u32)
        .filter(|&e| ready[e as usize] && !ids.contains(&e))
        .collect();
    candidates.sort_by(|&a, &b| {
        probabilities[b as usize]
            .total_cmp(&probabilities[a as usize])
            .then(a.cmp(&b))
    });
    let mut picks = Vec::new();
    for rank in order {
        let e = ids[rank] as usize;
        if ready[e] {
            continue;
        }
        let candidate = candidates.first().copied();
        let swap = tau > 0.0
            && !protected.contains(&rank)
            && starved[e] < 3
            && picks.len() < limit
            && weights[rank] > 0.0
            && weights[rank] <= 0.1
            && probabilities[e] > 0.0
            && candidate
                .is_some_and(|c| probabilities[c as usize] >= (1.0 - tau) * probabilities[e]);
        starved[e] = recovery(starved[e], true, swap);
        if swap {
            picks.push((rank, candidates.remove(0)));
        }
    }
    Ok(picks)
}

#[cfg(kani)]
#[kani::proof]
fn recovery_preserves_hits_and_forces_fourth_miss() {
    let current: u8 = kani::any();
    let missed: bool = kani::any();
    let swapped: bool = kani::any();
    let next = recovery(current, missed, swapped);
    if !missed {
        assert_eq!(next, current);
    } else {
        assert!(next <= 3);
        if swapped && current < 3 {
            assert_eq!(next, current + 1);
        } else {
            assert_eq!(next, 0);
        }
    }
    kani::cover!(!missed && current == 2);
    kani::cover!(missed && swapped && current == 2 && next == 3);
    kani::cover!(missed && current == 3 && next == 0);
}

#[test]
fn selections_only_replace_misses_with_distinct_ready_nonoriginals() -> Result<()> {
    let ids: Vec<_> = (0..10).collect();
    let mut ready = [false; 512];
    for e in [3, 10, 11, 12] {
        ready[e] = true;
    }
    let mut state = [0; 512];
    state[3] = 2;
    assert_eq!(
        select(
            &ids,
            &[1.0 / 512.0; 512],
            &[0.1; 10],
            &ready,
            &mut state,
            0.4,
            3
        )?,
        vec![(4, 10), (5, 11), (6, 12)]
    );
    assert_eq!((state[3], state[4]), (2, 1));
    state[4] = 3;
    let picks = select(
        &ids,
        &[1.0 / 512.0; 512],
        &[0.1; 10],
        &ready,
        &mut state,
        0.2,
        1,
    )?;
    assert_eq!(picks, vec![(5, 10)]);
    assert_eq!(state[4], 0);
    assert!(
        select(
            &ids,
            &[1.0 / 512.0; 512],
            &[0.1; 10],
            &ready,
            &mut state,
            0.0,
            3
        )?
        .is_empty()
    );
    for invalid in [f32::NAN, f32::INFINITY, -0.1, 0.41] {
        assert!(
            select(
                &ids,
                &[1.0 / 512.0; 512],
                &[0.1; 10],
                &ready,
                &mut state,
                invalid,
                1
            )
            .is_err()
        );
    }
    let mut invalid = ids;
    invalid[9] = 512;
    assert!(
        select(
            &invalid,
            &[0.0; 512],
            &[0.1; 10],
            &ready,
            &mut state,
            0.2,
            1
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn bounded_picker_matches_independent_sorted_oracle_and_transitions() -> Result<()> {
    let mut random = 0x5a17u32;
    for case in 0..2048usize {
        let mut next = || {
            random = random.wrapping_mul(1664525).wrapping_add(1013904223);
            random
        };
        let ids: Vec<_> = (0..10).map(|r| (case as u32 + 37 * r) % 512).collect();
        let p: Vec<_> = (0..512).map(|_| (next() % 33) as f32 / 1024.0).collect();
        let weights: Vec<_> = (0..10)
            .map(|_| [0.0, 0.09, 0.1, 0.10001][(next() % 4) as usize])
            .collect();
        let ready = std::array::from_fn(|_| next() % 7 == 0);
        let mut old = std::array::from_fn(|_| (next() % 5) as u8);
        let mut fast = old;
        let tau = [0.0, 0.05, 0.1, 0.2, 0.4][case % 5];
        let a = select(&ids, &p, &weights, &ready, &mut old, tau, 1 + case % 3)?;
        let b = select_fast(&ids, &p, &weights, &ready, &mut fast, tau, 1 + case % 3)?;
        assert_eq!(a, b, "case {case}");
        assert_eq!(old, fast, "counter case {case}");
    }
    let original: Vec<_> = (0..10).collect();
    let mut picked = original.clone();
    picked[3] = 10;
    let mut starved = [2; 512];
    let ready = [false; 512];
    assert!(apply_choices(&original, &picked, &ready, &mut starved).is_none());
    assert_eq!(starved, [2; 512]);
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(13)]
fn selected_ids_commit_only_valid_miss_transitions() {
    let original = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
    let mut selected = original;
    let mut ready = [false; 512];
    let mut state = [0; 512];
    let current: [u8; 10] = kani::any();
    let hits: [bool; 10] = kani::any();
    let swaps: [bool; 10] = kani::any();
    for i in 0..10 {
        state[i] = current[i];
        ready[i] = hits[i];
        ready[i + 10] = true;
        if swaps[i] {
            selected[i] = (i + 10) as u32;
        }
    }
    let before = state;
    let result = apply_choices(&original, &selected, &ready, &mut state);
    let mut count = 0;
    let mut valid = true;
    for i in 0..10 {
        if swaps[i] {
            count += 1;
            valid &= !hits[i] && current[i] < 3;
        }
    }
    assert_eq!(result.is_some(), valid && count <= 3);
    let accepted = result;
    if let Some(changed) = accepted {
        assert_eq!(changed, count);
        for i in 0..10 {
            assert_eq!(
                state[i],
                if hits[i] {
                    current[i]
                } else if swaps[i] {
                    current[i] + 1
                } else {
                    0
                }
            );
        }
    } else {
        for i in 0..10 {
            assert_eq!(state[i], before[i]);
        }
    }
    let other: usize = kani::any();
    kani::assume((10..512).contains(&other));
    assert_eq!(state[other], 0);
    kani::cover!(accepted == Some(3) && hits[0] && current[0] == 2);
    kani::cover!(result.is_none() && swaps[0] && hits[0]);
}
