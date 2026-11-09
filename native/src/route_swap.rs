//! AR diagnostic policy only; not a production or MTP routing contract.
use anyhow::{Result, ensure};

fn recovery(current: u8, missed: bool, swapped: bool) -> u8 {
    if !missed {
        current
    } else if swapped && current < 3 {
        current + 1
    } else {
        0
    }
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
