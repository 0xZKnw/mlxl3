//! One command buffer owns each routed expert's two temporary buffers.

pub(crate) const EXTENTS: [[u32; 3]; 3] = [[2560, 640, 640], [2560, 640, 640], [640, 2560, 2560]];
pub(crate) const CAPACITIES: [u64; 2] = [5120, 10240];

pub(crate) fn select(
    routed: bool,
    extents: [[u32; 3]; 3],
    setting: Option<bool>,
) -> Result<bool, ()> {
    if !routed || extents != EXTENTS {
        return Ok(false);
    }
    setting.ok_or(())
}

/// Useful prefixes, never the backing capacity, determine dispatch geometry.
pub(crate) fn fits([inputs, partials, outputs]: [u32; 3], capacities: [u64; 2]) -> bool {
    inputs > 0
        && partials > 0
        && outputs > 0
        && u64::from(inputs) * 2 <= capacities[0]
        && u64::from(partials) * 4 <= capacities[1]
}

#[test]
fn routed_scratch_scope_and_prefix_boundaries() {
    assert_eq!(select(true, EXTENTS, None), Err(()));
    assert_eq!(select(false, EXTENTS, None), Ok(false));
    assert_eq!(select(true, EXTENTS, Some(false)), Ok(false));
    assert_eq!(select(true, EXTENTS, Some(true)), Ok(true));
    for stage in 0..3 {
        for dimension in 0..3 {
            for value in [0, 1, 128, 640, 1280, 2560, 5120, u32::MAX] {
                let mut extents = EXTENTS;
                extents[stage][dimension] = value;
                if extents != EXTENTS {
                    assert_eq!(select(true, extents, None), Ok(false));
                }
            }
        }
    }
    for extents in EXTENTS {
        assert!(fits(extents, CAPACITIES));
        let need = [u64::from(extents[0]) * 2, u64::from(extents[1]) * 4];
        assert!(fits(extents, need));
        assert!(!fits(extents, [need[0] - 1, need[1]]));
        assert!(!fits(extents, [need[0], need[1] - 1]));
    }
    assert!(!fits([0, 640, 640], CAPACITIES));
    assert!(!fits([2560, 0, 640], CAPACITIES));
    assert!(!fits([2560, 640, 0], CAPACITIES));
    assert!(!fits([u32::MAX; 3], CAPACITIES));
    assert!(fits(
        [u32::MAX; 3],
        [u64::from(u32::MAX) * 2, u64::from(u32::MAX) * 4]
    ));
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(40)]
fn native_routed_scratch_selection_preserves_all_fallbacks() {
    let extents: [[u32; 3]; 3] = kani::any();
    let routed: bool = kani::any();
    let setting = match kani::any::<u8>() % 3 {
        0 => None,
        1 => Some(false),
        _ => Some(true),
    };
    let actual = select(routed, extents, setting);
    let eligible = routed && extents == EXTENTS;
    assert_eq!(actual.is_err(), eligible && setting.is_none());
    assert_eq!(actual == Ok(true), eligible && setting == Some(true));
    kani::cover!(actual == Ok(true));
    kani::cover!(actual.is_err());
    kani::cover!(!routed && setting.is_none());
    kani::cover!(routed && extents != EXTENTS && setting.is_none());
}

#[cfg(kani)]
#[kani::proof]
fn native_routed_scratch_prefix_capacities_use_wide_arithmetic() {
    let extents: [u32; 3] = kani::any();
    let capacities: [u64; 2] = kani::any();
    let expected = extents[0] != 0
        && extents[1] != 0
        && extents[2] != 0
        && u128::from(extents[0]) * 2 <= u128::from(capacities[0])
        && u128::from(extents[1]) * 4 <= u128::from(capacities[1]);
    assert_eq!(fits(extents, capacities), expected);
    kani::cover!(fits(extents, capacities) && extents[0] == u32::MAX);
    kani::cover!(!fits(extents, capacities) && extents[0] > 0 && extents[1] > 0 && extents[2] > 0);
}
