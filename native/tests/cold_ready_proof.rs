//! Manual proof only: impose external memory and wall-clock limits before invoking Kani.
//! See docs/measurements/post-p08-cold-ready-20261101/cpu-verification.md.

#[path = "../src/cold_ready.rs"]
mod cold_ready;

use cold_ready::partition;

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
