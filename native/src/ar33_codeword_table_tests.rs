use crate::{codec, codec::Codebook, flash_support::hc_kernel_flag};
use half::f16;

fn table() -> Vec<u16> {
    include_str!("../shaders/ar33_codeword_table.metal")
        .split_once("= {")
        .unwrap()
        .1
        .split_once('}')
        .unwrap()
        .0
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| u16::from_str_radix(s.strip_prefix("0x").unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn post_hash_table_matches_independent_half_fma_including_both_endpoints() {
    let actual = table();
    assert_eq!(actual.len(), 1021);
    let expected = (0u16..=1020)
        .map(|sum| {
            f16::from_f64(
                f16::from_bits(0x6400 + sum).to_f64() * f16::from_bits(0x1eee).to_f64()
                    + f16::from_bits(0xc931).to_f64(),
            )
            .to_bits()
        })
        .collect::<Vec<_>>();
    assert!(actual.iter().all(|&v| f16::from_bits(v).is_finite()));
    assert_eq!(actual, expected);
    for endpoint in [0, 1020] {
        let mut mutant = actual.clone();
        mutant[endpoint] ^= 1;
        assert_ne!(mutant, expected, "endpoint mutation escaped the oracle");
    }
}

#[test]
fn post_hash_table_covers_every_sixteen_bit_state_without_k_restriction() {
    let table = table();
    for word in 0..=u16::MAX {
        let mixed = u32::from(word).wrapping_mul(0x83dcd12d);
        let sum = mixed
            .to_le_bytes()
            .iter()
            .map(|&v| usize::from(v))
            .sum::<usize>();
        assert!(sum <= 1020);
        assert_eq!(
            table[sum],
            codec::decode_codeword(word, Codebook::Mul1).to_bits()
        );
    }
}

#[test]
fn codeword_table_uses_existing_strict_off_default_m5_policy() {
    for qualified in [false, true] {
        assert_eq!(hc_kernel_flag(None, qualified), Some(false));
        assert_eq!(hc_kernel_flag(Some("0"), qualified), Some(false));
        assert_eq!(hc_kernel_flag(Some("1"), qualified), Some(qualified));
        for invalid in ["", "01", "true", " 1", "1\n", "é"] {
            assert_eq!(hc_kernel_flag(Some(invalid), qualified), None);
        }
    }
}
