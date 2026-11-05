//! Geometry for the paired router keeps the fixed reference output indexing.
pub(crate) fn geometry(enabled: bool, shape: [u32; 2], capability: [u32; 2]) -> Option<[u32; 2]> {
    (enabled
        && crate::contracts::flash_gate_rows(shape[0], shape[1])
        && capability[0] == 32
        && capability[1] >= 32)
        .then_some([256, shape[0]])
}

#[test]
fn paired_router_geometry_keeps_reference_bounds_and_capability() {
    assert_eq!(geometry(true, [1, 2560], [32, 32]), Some([256, 1]));
    assert_eq!(
        geometry(true, [u32::MAX / 512, 2560], [32, u32::MAX]),
        Some([256, u32::MAX / 512])
    );
    for (enabled, shape, cap) in [
        (false, [1, 2560], [32, 32]),
        (true, [0, 2560], [32, 32]),
        (true, [1, 2559], [32, 32]),
        (true, [1, 2561], [32, 32]),
        (true, [u32::MAX / 512 + 1, 2560], [32, 32]),
        (true, [1, 2560], [64, 32]),
        (true, [1, 2560], [32, 31]),
    ] {
        assert_eq!(geometry(enabled, shape, cap), None);
    }
}

#[cfg(kani)]
#[kani::proof]
fn paired_router_actual_geometry_preserves_every_reference_output_bound() {
    let enabled: bool = kani::any();
    let shape: [u32; 2] = kani::any();
    let capability: [u32; 2] = kani::any();
    let actual = geometry(enabled, shape, capability);
    assert_eq!(
        actual.is_some(),
        enabled
            && shape[0] > 0
            && shape[0] <= u32::MAX / 512
            && shape[1] == 2560
            && capability[0] == 32
            && capability[1] >= 32
    );
    if let Some([groups, rows]) = actual {
        assert_eq!(groups * 2, 512);
        assert_eq!(rows, shape[0]);
        assert!(
            (rows - 1)
                .checked_mul(512)
                .unwrap()
                .checked_add(511)
                .is_some()
        );
        let group: u32 = kani::any();
        if group < groups {
            assert!(group * 2 + 1 < 512);
            assert_eq!((group * 2 + 1) / 2, group);
        }
    }
    kani::cover!(actual == Some([256, 1]));
    kani::cover!(actual == Some([256, u32::MAX / 512]));
    kani::cover!(actual.is_none() && shape[1] == 2559);
    kani::cover!(actual.is_none() && capability[1] == 31);
}
