//! Partition route occurrences without changing their order or deduplicating experts.

type Occurrences = Vec<(usize, u32)>;

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
