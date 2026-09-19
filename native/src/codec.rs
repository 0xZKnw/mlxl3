//! EXL3 on-disk format. See THIRD_PARTY_NOTICES.md for upstream attribution.
use anyhow::{Result, ensure};
use half::f16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
pub enum Codebook {
    Default = 0,
    Mcg = 1,
    Mul1 = 2,
}

impl TryFrom<u32> for Codebook {
    type Error = anyhow::Error;
    fn try_from(value: u32) -> Result<Self> {
        match value {
            0 => Ok(Self::Default),
            1 => Ok(Self::Mcg),
            2 => Ok(Self::Mul1),
            _ => anyhow::bail!("invalid EXL3 codebook {value}"),
        }
    }
}

pub fn check_k(k: usize) -> Result<()> {
    ensure!(valid_k(k), "K must be in 1..=8, got {k}");
    Ok(())
}

fn valid_k(k: usize) -> bool {
    (1..=8).contains(&k)
}

fn pack_source(k: usize, logical: usize, bit: usize) -> (usize, usize) {
    let span = logical / k;
    let stream_bit = (logical % k) * 16 + bit;
    (span * 16 + stream_bit / k, k - 1 - stream_bit % k)
}

fn unpack_window(k: usize, state: usize) -> (usize, usize, usize) {
    let b0 = state * k + k + 256 * k - 16;
    let b1 = b0 + 16;
    let i0 = b0 / 32;
    let i1 = (b1 - 1) / 32;
    (i0, i1, (i1 + 1) * 32 - b1)
}

fn packed_pair(k: usize, word: usize) -> usize {
    2 * (word % (8 * k))
}

pub fn decode_codeword(word: u16, mode: Codebook) -> f16 {
    let x = u32::from(word);
    let bits = match mode {
        Codebook::Default => x.wrapping_mul(89_226_354).wrapping_add(64_248_484),
        Codebook::Mcg => x.wrapping_mul(0xcbac1fed),
        Codebook::Mul1 => {
            let bits = x.wrapping_mul(0x83dcd12d);
            let sum = 0x6400 + (0..4).map(|i| (bits >> (8 * i)) & 255).sum::<u32>();
            // One final FP16 rounding, like the reference CUDA __hfma.
            let h = f16::from_bits(sum as u16).to_f64();
            let inv = f16::from_bits(0x1eee).to_f64();
            let bias = f16::from_bits(0xc931).to_f64();
            return f16::from_f64(h * inv + bias);
        }
    };
    let bits = 0x3b603b60 ^ (bits & 0x8fff8fff);
    f16::from_f32(
        f16::from_bits(bits as u16).to_f32() + f16::from_bits((bits >> 16) as u16).to_f32(),
    )
}

pub fn pack(encoded: &[u16], k: usize) -> Result<Vec<u16>> {
    check_k(k)?;
    ensure!(
        encoded.len().is_multiple_of(256),
        "encoded length must be a multiple of 256"
    );
    Ok(pack_valid(encoded, k))
}

fn pack_valid(encoded: &[u16], k: usize) -> Vec<u16> {
    let mut result = vec![0u16; encoded.len() / 256 * 16 * k];
    for (tile, output) in encoded
        .as_chunks::<256>()
        .0
        .iter()
        .zip(result.chunks_exact_mut(16 * k))
    {
        for logical in 0..16 * k {
            let mut word = 0u16;
            for bit in 0..16 {
                let (index, shift) = pack_source(k, logical, bit);
                word = (word << 1) | ((tile[index] >> shift) & 1);
            }
            output[logical ^ 1] = word;
        }
    }
    result
}

pub fn unpack(packed: &[u16], k: usize) -> Result<Vec<u16>> {
    check_k(k)?;
    ensure!(
        packed.len().is_multiple_of(16 * k),
        "packed length is not a whole tile"
    );
    Ok(unpack_valid(packed, k))
}

fn unpack_valid(packed: &[u16], k: usize) -> Vec<u16> {
    let mut result = Vec::with_capacity(packed.len() / (16 * k) * 256);
    for tile in packed.chunks_exact(16 * k) {
        for t in 0..256 {
            let (i0, i1, shift) = unpack_window(k, t);
            let read = |i: usize| {
                let p = packed_pair(k, i);
                u32::from(tile[p]) | (u32::from(tile[p + 1]) << 16)
            };
            result.push((((u64::from(read(i0)) << 32) | u64::from(read(i1))) >> shift) as u16);
        }
    }
    result
}

pub fn permutation() -> [usize; 256] {
    let mut out = [0; 256];
    for lane in 0..32 {
        let r = (lane % 4) * 2;
        let c = lane / 4;
        for (cg, col) in [c, c + 8].into_iter().enumerate() {
            for (ri, row) in [r, r + 1, r + 8, r + 9].into_iter().enumerate() {
                out[lane * 8 + cg * 4 + ri] = row * 16 + col;
            }
        }
    }
    out
}

pub fn permutation_inverse() -> [usize; 256] {
    let mut out = [0; 256];
    for (index, value) in permutation().into_iter().enumerate() {
        out[value] = index;
    }
    out
}

/// Reference inner matrix in row-major [input, output] order (no rotations).
pub fn reconstruct_inner(
    packed: &[u16],
    k: usize,
    mode: Codebook,
    rows: usize,
    cols: usize,
) -> Result<Vec<f16>> {
    check_k(k)?;
    ensure!(
        rows > 0 && cols > 0 && rows.is_multiple_of(16) && cols.is_multiple_of(16),
        "dimensions must be positive multiples of 16"
    );
    let count = rows
        .checked_mul(cols)
        .ok_or_else(|| anyhow::anyhow!("shape overflow"))?;
    ensure!(
        packed.len() == count / 256 * 16 * k,
        "trellis size does not match shape"
    );
    let states = unpack(packed, k)?;
    let perm = permutation();
    let mut out = vec![f16::ZERO; count];
    for (tile, values) in states.as_chunks::<256>().0.iter().enumerate() {
        let row = tile / (cols / 16) * 16;
        let col = tile % (cols / 16) * 16;
        for (index, state) in values.iter().enumerate() {
            let p = perm[index];
            out[(row + p / 16) * cols + col + p % 16] = decode_codeword(*state, mode);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pack_tail_biting_all_bitrates() {
        let input: Vec<_> = (0..768).map(|i| (i * 391 + 1729) as u16).collect();
        for k in 1..=8 {
            let packed = pack(&input, k).unwrap();
            let decoded = unpack(&packed, k).unwrap();
            let mask = (1 << k) - 1;
            for (actual, source) in decoded.iter().zip(&input) {
                assert_eq!(actual & mask, source & mask);
            }
            assert_eq!(pack(&decoded, k).unwrap(), packed);
        }
        assert!(pack(&input, 0).is_err());
        assert!(pack(&input[..255], 4).is_err());
        assert!(unpack(&[0; 15], 1).is_err());
        assert!(Codebook::try_from(3).is_err());
    }
    #[test]
    fn permutation_is_bijective() {
        let direct = permutation();
        for (i, p) in permutation_inverse().iter().enumerate() {
            assert_eq!(direct[*p], i);
        }
    }
    #[test]
    fn codebooks_are_finite_exhaustively() {
        for mode in [Codebook::Default, Codebook::Mcg, Codebook::Mul1] {
            for word in 0..=u16::MAX {
                assert!(decode_codeword(word, mode).is_finite());
            }
        }
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn bitrate_validation_accepts_exactly_one_through_eight() {
        let k: usize = kani::any();
        assert_eq!(valid_k(k), (1..=8).contains(&k));
        kani::cover!(valid_k(k));
        kani::cover!(!valid_k(k));
    }

    #[kani::proof]
    fn symbolic_pack_indices_stay_inside_one_tile() {
        let k: usize = kani::any();
        let logical: usize = kani::any();
        let bit: usize = kani::any();
        kani::assume((1..=8).contains(&k));
        kani::assume(logical < 16 * k && bit < 16);
        let (index, shift) = pack_source(k, logical, bit);
        assert!(index < 256);
        assert!(shift < k);
        kani::cover!(k == 1);
        kani::cover!(k == 8);
    }

    #[kani::proof]
    fn symbolic_unpack_indices_stay_inside_one_packed_tile() {
        let k: usize = kani::any();
        let state: u8 = kani::any();
        kani::assume((1..=8).contains(&k));
        let (i0, i1, shift) = unpack_window(k, usize::from(state));
        let p0 = packed_pair(k, i0);
        let p1 = packed_pair(k, i1);
        assert!(p0 + 1 < 16 * k);
        assert!(p1 + 1 < 16 * k);
        assert!(shift < 64);
        kani::cover!(k == 1);
        kani::cover!(k == 8);
    }

    #[kani::proof]
    #[kani::unwind(260)]
    fn exl3_permutation_and_inverse_are_bijections() {
        let direct = permutation();
        let inverse = permutation_inverse();
        for index in 0..256 {
            assert!(direct[index] < 256);
            assert!(inverse[index] < 256);
            assert_eq!(direct[inverse[index]], index);
            assert_eq!(inverse[direct[index]], index);
        }
    }
}
