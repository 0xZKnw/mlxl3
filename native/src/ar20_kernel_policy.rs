//! Exact bits passed to the native Metal compiler, not a GPU correctness proof.
pub(crate) fn mask(flags: [bool; 4]) -> u32 {
    u32::from(flags[0])
        | (u32::from(flags[1]) << 1)
        | (u32::from(flags[2]) << 2)
        | (u32::from(flags[3]) << 3)
}

pub(crate) fn defines(bits: u32) -> String {
    format!(
        "#define MLXL3_AR20_SWAR {}\n#define MLXL3_AR20_READER8 {}\n#define MLXL3_AR20_PRELOAD {}\n#define MLXL3_AR20_H128 {}\n",
        bits & 1,
        (bits >> 1) & 1,
        (bits >> 2) & 1,
        (bits >> 3) & 1
    )
}

#[test]
fn shader_defines_match_every_actual_kernel_choice() {
    for expected in 0u32..16 {
        let flags = std::array::from_fn(|i| expected & (1 << i) != 0);
        assert_eq!(mask(flags), expected);
        let text = defines(mask(flags));
        let lines = text.lines().collect::<Vec<_>>();
        assert_eq!(lines.len(), 4);
        for (i, name) in ["SWAR", "READER8", "PRELOAD", "H128"].iter().enumerate() {
            assert_eq!(
                lines[i],
                format!("#define MLXL3_AR20_{name} {}", u32::from(flags[i]))
            );
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn native_kernel_mask_preserves_every_actual_compiler_switch() {
    let flags: [bool; 4] = kani::any();
    let bits = mask(flags);
    assert!(bits <= 15);
    assert_eq!(bits & 1 != 0, flags[0]);
    assert_eq!(bits & 2 != 0, flags[1]);
    assert_eq!(bits & 4 != 0, flags[2]);
    assert_eq!(bits & 8 != 0, flags[3]);
    kani::cover!(bits == 0);
    kani::cover!(bits == 15);
    kani::cover!(bits == 5);
    kani::cover!(bits == 10);
}
