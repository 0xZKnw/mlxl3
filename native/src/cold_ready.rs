//! Partition route occurrences without changing their order or deduplicating experts.

type Occurrences = Vec<(usize, u32)>;

pub(crate) fn parse_max_rows(value: Option<&str>) -> Option<i32> {
    match value {
        None | Some("2") => Some(2),
        Some("4") => Some(4),
        _ => None,
    }
}

/// Order reads only; callers keep occurrence order for the final reduction.
pub(crate) fn prioritize_jobs(jobs: &mut [(u32, usize)], cold: &[(usize, u32)]) {
    jobs.sort_by_key(|&(_, slot)| {
        std::cmp::Reverse(cold.iter().filter(|&&(_, s)| s as usize == slot).count())
    });
}

#[cfg(test)]
pub(crate) fn job_multiplicities(jobs: &[(u32, usize)], cold: &[(usize, u32)]) -> Vec<usize> {
    jobs.iter()
        .map(|&(_, slot)| cold.iter().filter(|&&(_, s)| s as usize == slot).count())
        .collect()
}

pub(crate) fn partition(cold: &[(usize, u32)], ready_slots: &[u32]) -> (Occurrences, Occurrences) {
    let mut ready = Vec::with_capacity(cold.len());
    let mut waiting = Vec::with_capacity(cold.len());
    for &occurrence in cold {
        if ready_slots.contains(&occurrence.1) {
            ready.push(occurrence);
        } else {
            waiting.push(occurrence);
        }
    }
    (ready, waiting)
}

#[cfg(test)]
mod tests {
    use super::{job_multiplicities, parse_max_rows, partition, prioritize_jobs};

    #[test]
    fn row_limit_options_are_strict_and_default_to_qualified_width() {
        assert_eq!(parse_max_rows(None), Some(2));
        assert_eq!(parse_max_rows(Some("2")), Some(2));
        assert_eq!(parse_max_rows(Some("4")), Some(4));
        for value in ["", "0", "1", "3", "5", "8", "04", "+4", "4 ", " 4", "４"] {
            assert_eq!(parse_max_rows(Some(value)), None, "{value}");
        }
    }

    #[test]
    fn priority_preserves_identity_and_stable_ties_without_reordering_occurrences() {
        // The most frequent slot is deliberately last in the read plan.
        let cold = [(0, 9), (1, 2), (2, 4), (10, 2), (11, 4), (12, 2)];
        let original = cold;
        let mut jobs = [(81, 9), (19, 4), (61, 2), (99, usize::MAX)];
        prioritize_jobs(&mut jobs, &cold);
        assert_eq!(jobs, [(61, 2), (19, 4), (81, 9), (99, usize::MAX)]);
        assert_eq!(job_multiplicities(&jobs, &cold), [3, 2, 1, 0]);
        assert_eq!(cold, original);
        let mut ties = [(17, 4), (18, 2), (19, 8)];
        prioritize_jobs(&mut ties, &[(0, 2), (1, 4), (2, 8)]);
        assert_eq!(ties, [(17, 4), (18, 2), (19, 8)]);
        prioritize_jobs(&mut ties, &[]);
        assert_eq!(ties, [(17, 4), (18, 2), (19, 8)]);
        prioritize_jobs(&mut [], &cold);
    }

    #[test]
    fn all_b2_presence_masks_match_independent_priority_and_partition_oracle() {
        // Four slots, each present zero, once, or in both token rows.
        for pattern in 0..81usize {
            let mut remainder = pattern;
            let mut counts = [0; 4];
            let mut cold = Vec::new();
            for (slot, count) in counts.iter_mut().enumerate() {
                *count = remainder % 3;
                remainder /= 3;
                for row in 0..*count {
                    cold.push((row * 10 + slot, slot as u32));
                }
            }
            let mut jobs = [(100, 0), (101, 1), (102, 2), (103, 3)];
            prioritize_jobs(&mut jobs, &cold);
            let mut expected = Vec::new();
            for count in (0..=2).rev() {
                for (slot, &actual_count) in counts.iter().enumerate() {
                    if actual_count == count {
                        expected.push((100 + slot as u32, slot));
                    }
                }
            }
            assert_eq!(jobs.as_slice(), expected);
            for mask in 0..16 {
                let ready_slots = (0..4).filter(|s| mask & (1 << s) != 0).collect::<Vec<_>>();
                let (ready, waiting) = partition(&cold, &ready_slots);
                assert_eq!(ready.len() + waiting.len(), cold.len());
                assert_eq!(
                    ready,
                    cold.iter()
                        .copied()
                        .filter(|&(_, s)| mask & (1 << s) != 0)
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    waiting,
                    cold.iter()
                        .copied()
                        .filter(|&(_, s)| mask & (1 << s) == 0)
                        .collect::<Vec<_>>()
                );
            }
        }
    }

    #[test]
    fn empty_all_none_and_integer_boundaries() {
        assert_eq!(partition(&[], &[]), (vec![], vec![]));
        assert_eq!(partition(&[], &[0, u32::MAX]), (vec![], vec![]));
        let cold = [(usize::MAX, u32::MAX), (0, 0), (usize::MAX, u32::MAX)];
        assert_eq!(partition(&cold, &[]), (vec![], cold.to_vec()));
        assert_eq!(partition(&cold, &[u32::MAX, 0]), (cold.to_vec(), vec![]));
        assert_eq!(partition(&cold, &[1]), (vec![], cold.to_vec()));
    }

    #[test]
    fn duplicate_b2_experts_keep_each_activation_occurrence() {
        let cold = [(1, 5), (3, 8), (8, 2), (11, 5), (14, 2), (19, 8)];
        assert_eq!(
            partition(&cold, &[2, 5, 2]),
            (
                vec![(1, 5), (8, 2), (11, 5), (14, 2)],
                vec![(3, 8), (19, 8)]
            )
        );
        assert_eq!(
            partition(&cold, &[8]),
            (
                vec![(3, 8), (19, 8)],
                vec![(1, 5), (8, 2), (11, 5), (14, 2)]
            )
        );
    }

    #[test]
    fn twenty_occurrences_match_all_ten_slot_readiness_masks() {
        let cold = (0..20)
            .map(|row| (row, ((row * 7) % 10) as u32))
            .collect::<Vec<_>>();
        for mask in 0..1024u32 {
            let slots = (0..10)
                .filter(|slot| mask & (1 << slot) != 0)
                .collect::<Vec<_>>();
            let (ready, waiting) = partition(&cold, &slots);
            let mut ready_index = 0;
            let mut waiting_index = 0;
            for &occurrence in &cold {
                if mask & (1 << occurrence.1) != 0 {
                    assert_eq!(ready[ready_index], occurrence);
                    ready_index += 1;
                } else {
                    assert_eq!(waiting[waiting_index], occurrence);
                    waiting_index += 1;
                }
            }
            assert_eq!(ready.len(), ready_index);
            assert_eq!(waiting.len(), waiting_index);
            assert_eq!(ready.len() + waiting.len(), cold.len());
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(6)]
fn two_job_priority_keeps_records_and_stable_order() {
    let slots: [u32; 4] = kani::any();
    let ids: [u32; 2] = kani::any();
    let cold = [(0, slots[0]), (1, slots[1]), (10, slots[2]), (11, slots[3])];
    let original = [(ids[0], 0usize), (ids[1], 1usize)];
    let mut jobs = original;
    prioritize_jobs(&mut jobs, &cold);
    let first_count = slots.iter().filter(|&&slot| slot == 0).count();
    let second_count = slots.iter().filter(|&&slot| slot == 1).count();
    if second_count > first_count {
        assert_eq!(jobs, [original[1], original[0]]);
    } else {
        assert_eq!(jobs, original);
    }
    kani::cover!(second_count > first_count);
    kani::cover!(second_count == first_count);
    kani::cover!(first_count > second_count);
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(8)]
fn cold_row_limit_has_only_qualified_and_opt_in_values() {
    let choice: u8 = kani::any();
    let value = match choice {
        0 => None,
        1 => Some("2"),
        2 => Some("4"),
        3 => Some("3"),
        4 => Some("04"),
        5 => Some("4 "),
        6 => Some(""),
        _ => Some("４"),
    };
    let result = parse_max_rows(value);
    assert_eq!(
        result,
        match choice {
            0 | 1 => Some(2),
            2 => Some(4),
            _ => None,
        }
    );
    kani::cover!(result == Some(2));
    kani::cover!(result == Some(4));
    kani::cover!(result.is_none());
}
