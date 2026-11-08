//! A pool is the only owner permitted to renew a retained frame.
pub(crate) const CAPACITY: usize = 9;
pub(crate) const FRAME_BYTES: u64 = 30720 * 2 + 48 * 128 * 128 * 4;

pub(crate) fn reusable_epoch(owners: usize, epoch: u64) -> Option<u64> {
    if owners != 1 {
        return None;
    }
    epoch.checked_add(1)
}

#[test]
fn only_an_unshared_frame_can_advance_its_generation() {
    use std::rc::Rc;
    let owner = Rc::new(());
    assert_eq!(reusable_epoch(Rc::strong_count(&owner), 0), Some(1));
    let external = owner.clone();
    assert_eq!(reusable_epoch(Rc::strong_count(&owner), 0), None);
    drop(external);
    assert_eq!(reusable_epoch(Rc::strong_count(&owner), 9), Some(10));
    assert_eq!(reusable_epoch(1, u64::MAX), None);
    assert_eq!(FRAME_BYTES, 3207168);
}

#[cfg(kani)]
#[kani::proof]
fn generation_renewal_requires_exclusive_pool_ownership_without_wrap() {
    let owners: usize = kani::any();
    let epoch: u64 = kani::any();
    let renewed = reusable_epoch(owners, epoch);
    assert_eq!(renewed.is_some(), owners == 1 && epoch < u64::MAX);
    if let Some(next) = renewed {
        assert!(next > epoch && next - epoch == 1);
    }
    kani::cover!(owners == 2 && renewed.is_none());
    kani::cover!(owners == 1 && epoch == u64::MAX && renewed.is_none());
    kani::cover!(owners == 1 && renewed == Some(1));
}

pub(crate) fn temporal_key(rows: u32) -> Option<usize> {
    match rows {
        2 => Some(0),
        3 | 4 => Some(1),
        _ => None,
    }
}

#[test]
fn temporal_shapes_mask_only_the_b3_tail() {
    for rows in [0, 1, 2, 3, 4, 5, u32::MAX] {
        assert_eq!(
            temporal_key(rows),
            match rows {
                2 => Some(0),
                3 | 4 => Some(1),
                _ => None,
            }
        );
    }
}

#[cfg(kani)]
#[kani::proof]
fn temporal_shape_admission_keeps_rows_and_pipeline_indices_bounded() {
    let rows: u32 = kani::any();
    let key = temporal_key(rows);
    assert_eq!(key.is_some(), rows >= 2 && rows <= 4);
    if let Some(key) = key {
        assert!(key < 2 && rows <= (2 << key));
        assert!(u64::from(rows) * FRAME_BYTES <= 4 * FRAME_BYTES);
    }
    kani::cover!(rows == 3 && key == Some(1));
    kani::cover!(rows == 2 && key == Some(0));
}

pub(crate) fn segment_admitted(pending: usize, failed: bool) -> bool {
    !failed && pending < 2
}

#[cfg(kani)]
#[kani::proof]
fn gpu_segment_admission_never_wraps_or_overcommits_after_a_fault() {
    let pending: usize = kani::any();
    let failed: bool = kani::any();
    let admit = segment_admitted(pending, failed);
    assert_eq!(admit, !failed && pending < 2);
    if admit {
        assert!(pending + 1 <= 2);
    }
    kani::cover!(pending == 1 && admit);
    kani::cover!(pending == 0 && failed && !admit);
}
