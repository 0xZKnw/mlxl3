//! Partition route occurrences without changing their order or deduplicating experts.

pub(crate) fn partition(
    cold: &[(usize, u32)],
    ready_slots: &[u32],
) -> (Vec<(usize, u32)>, Vec<(usize, u32)>) {
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
    use super::partition;

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
#[kani::unwind(8)]
fn cold_partition_preserves_symbolic_occurrences_and_stable_order() {
    let cold: [(usize, u32); 6] = kani::any();
    let slots: [u32; 6] = kani::any();
    let count: u8 = kani::any();
    let ready_count: u8 = kani::any();
    kani::assume(count <= 6 && ready_count <= 6);
    let cold = &cold[..usize::from(count)];
    let slots = &slots[..usize::from(ready_count)];
    let (ready, waiting) = partition(cold, slots);
    let mut ready_index = 0;
    let mut waiting_index = 0;
    for &occurrence in cold {
        let mut member = false;
        for &slot in slots {
            member |= slot == occurrence.1;
        }
        if member {
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
    kani::cover!(cold.is_empty());
    kani::cover!(!cold.is_empty() && ready.is_empty());
    kani::cover!(!cold.is_empty() && waiting.is_empty());
    kani::cover!(!ready.is_empty() && !waiting.is_empty());
    kani::cover!(
        cold.len() >= 2 && cold[0].1 == cold[1].1 && cold[0].0 != cold[1].0 && ready.len() >= 2
    );
    kani::cover!(cold.len() >= 3 && ready.len() == 2 && ready[0] == cold[0] && ready[1] == cold[2]);
}
