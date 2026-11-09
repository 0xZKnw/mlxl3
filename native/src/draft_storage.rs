//! Compact immutable projection payloads, decoded into unchanged SGEMM operands.
use anyhow::{Context, Result, ensure};
use half::bf16;

#[derive(Clone, Debug)]
pub(crate) enum CompactWeight {
    Bf16(Vec<u16>),
    Affine4 {
        codes: Vec<u8>,
        parameters: Vec<[u16; 2]>,
    },
}
impl CompactWeight {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        ensure!(
            !bytes.is_empty() && bytes.len().is_multiple_of(2),
            "invalid BF16 payload"
        );
        let mut words = Vec::new();
        words.try_reserve_exact(bytes.len() / 2)?;
        for b in bytes.as_chunks::<2>().0 {
            let word = u16::from_le_bytes(*b);
            ensure!(bf16::from_bits(word).is_finite(), "nonfinite draft weight");
            words.push(word);
        }
        Ok(Self::Bf16(words))
    }
    pub fn len(&self) -> usize {
        match self {
            Self::Bf16(v) => v.len(),
            Self::Affine4 { codes, .. } => codes.len() * 2,
        }
    }
    pub fn retained_bytes(&self) -> Result<usize> {
        match self {
            Self::Bf16(v) => v.capacity().checked_mul(2),
            Self::Affine4 { codes, parameters } => parameters
                .capacity()
                .checked_mul(4)
                .and_then(|n| n.checked_add(codes.capacity())),
        }
        .context("compact draft payload overflow")
    }
    pub fn decode(&self) -> Result<Vec<f32>> {
        let mut out = Vec::new();
        out.try_reserve_exact(self.len())?;
        match self {
            Self::Bf16(words) => out.extend(words.iter().map(|&w| bf16::from_bits(w).to_f32())),
            Self::Affine4 { codes, parameters } => {
                ensure!(
                    codes.len() == parameters.len() * 32,
                    "invalid affine draft storage"
                );
                for (bytes, &[s, b]) in codes.as_chunks::<32>().0.iter().zip(parameters) {
                    let (scale, bias) = (bf16::from_bits(s).to_f32(), bf16::from_bits(b).to_f32());
                    for &byte in bytes {
                        out.push(f32::from(byte & 15) * scale + bias);
                        out.push(f32::from(byte >> 4) * scale + bias);
                    }
                }
            }
        }
        ensure!(
            out.iter().all(|v| v.is_finite()),
            "nonfinite decoded draft weight"
        );
        Ok(out)
    }
    pub fn quantize(&mut self) -> Result<()> {
        let Self::Bf16(words) = self else {
            return Ok(());
        };
        ensure!(words.len().is_multiple_of(64), "incomplete affine group");
        let mut codes = Vec::new();
        let mut parameters = Vec::new();
        codes.try_reserve_exact(words.len() / 2)?;
        parameters.try_reserve_exact(words.len() / 64)?;
        for words in words.as_chunks::<64>().0 {
            let group: [f32; 64] = std::array::from_fn(|i| bf16::from_bits(words[i]).to_f32());
            let min = group.iter().copied().fold(f32::MAX, f32::min);
            let max = group.iter().copied().fold(0., f32::max);
            let mut scale = ((max - min) / 15.).max(1e-7);
            let edge = if min.abs() > max.abs() {
                min
            } else {
                scale = -scale;
                max
            };
            let q0 = (edge / scale).round();
            let bias = if q0 == 0. {
                0.
            } else {
                scale = edge / q0;
                edge
            };
            let s = bf16::from_f32(scale);
            let b = bf16::from_f32(bias);
            ensure!(s.is_finite() && b.is_finite(), "invalid affine parameters");
            parameters.push([s.to_bits(), b.to_bits()]);
            for pair in group.as_chunks::<2>().0 {
                let encode = |v: f32| ((v - bias) / scale).round().clamp(0., 15.) as u8;
                codes.push(encode(pair[0]) | (encode(pair[1]) << 4));
            }
        }
        let candidate = Self::Affine4 { codes, parameters };
        // Finitude validation without a full FP32 temporary.
        if let Self::Affine4 { codes, parameters } = &candidate {
            for (chunk, &[s, b]) in codes.as_chunks::<32>().0.iter().zip(parameters) {
                let (s, b) = (bf16::from_bits(s).to_f32(), bf16::from_bits(b).to_f32());
                ensure!(
                    chunk.iter().all(|v| (f32::from(v & 15) * s + b).is_finite()
                        && (f32::from(v >> 4) * s + b).is_finite()),
                    "invalid affine reconstruction"
                );
            }
        }
        *self = candidate;
        Ok(())
    }
}

/// All planes are validated before the first mutation; admitted copies cannot
/// fail, allocate, or clone old planes. Callers stage every producer first.
pub(crate) fn commit_ring(
    planes: &mut [&mut [f32]],
    updates: &[&[f32]],
    start: usize,
    rows: usize,
    width: usize,
) -> Result<()> {
    let writes =
        crate::flash_draft_budget::ring_writes(i32::try_from(start)?, i32::try_from(rows)?)
            .context("invalid ring interval")?;
    let capacity = 2048usize
        .checked_mul(width)
        .context("ring capacity overflow")?;
    let count = rows.checked_mul(width).context("ring update overflow")?;
    ensure!(
        width > 0 && !planes.is_empty() && planes.len() == updates.len(),
        "invalid ring planes"
    );
    ensure!(
        planes.iter().all(|p| p.len() == capacity)
            && updates
                .iter()
                .all(|p| p.len() == count && p.iter().all(|v| v.is_finite())),
        "invalid ring plane storage"
    );
    for (plane, update) in planes.iter_mut().zip(updates) {
        for (source, destination, rows) in writes {
            let (source, destination, rows) = (
                source as usize * width,
                destination as usize * width,
                rows as usize * width,
            );
            plane[destination..destination + rows].copy_from_slice(&update[source..source + rows]);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect()
    }
    #[test]
    fn all_finite_bf16_words_roundtrip_bitwise_with_half_the_dense_payload() -> Result<()> {
        let words: Vec<u16> = (0..=u16::MAX)
            .filter(|&w| bf16::from_bits(w).is_finite())
            .collect();
        let raw: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let weight = CompactWeight::from_bytes(&raw)?;
        let decoded = weight.decode()?;
        assert_eq!(weight.len(), words.len());
        assert_eq!(weight.retained_bytes()?, words.len() * 2);
        for (&actual, &word) in decoded.iter().zip(&words) {
            // BF16 is the high half of the IEEE FP32 representation, including signed zero.
            assert_eq!(actual.to_bits(), u32::from(word) << 16);
        }
        for invalid in [vec![], vec![1], vec![0x80, 0x7f], vec![0x81, 0xff]] {
            assert!(CompactWeight::from_bytes(&invalid).is_err());
        }
        Ok(())
    }
    #[test]
    fn packed_affine_matches_independent_dense_reference_for_every_code() -> Result<()> {
        // Copied mathematical reference, not codec decode/encode helpers. Constant,
        // positive-only, negative-only, mixed, signed-zero and deterministic random groups.
        let mut input = vec![0.; 64];
        input.extend(vec![-0.; 64]);
        input.extend(vec![2.; 64]);
        input.extend(vec![-3.; 64]);
        for offset in [-128., 0., 128.] {
            input.extend((0..64).map(|i| i as f32 + offset));
        }
        let mut seed = 42u32;
        for _ in 0..128 * 64 {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            input.push((seed as i32 as f32) / 16777216.);
        }
        let raw = bytes(&input);
        let mut weight = CompactWeight::from_bytes(&raw)?;
        let canonical: Vec<f32> = raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16))
            .collect();
        let mut expected = Vec::new();
        for group in canonical.as_chunks::<64>().0 {
            let low = group.iter().copied().fold(f32::MAX, |a, b| a.min(b));
            let high = group.iter().copied().fold(0f32, |a, b| a.max(b));
            let delta = ((high - low) / 15.).max(1e-7);
            let (edge, signed_delta) = if low.abs() > high.abs() {
                (low, delta)
            } else {
                (high, -delta)
            };
            let quantized_edge = (edge / signed_delta).round();
            let (scale, bias) = if quantized_edge == 0. {
                (signed_delta, 0.)
            } else {
                (edge / quantized_edge, edge)
            };
            for value in group {
                let code = ((value - bias) / scale).round().clamp(0., 15.);
                expected
                    .push(code * bf16::from_f32(scale).to_f32() + bf16::from_f32(bias).to_f32());
            }
        }
        weight.quantize()?;
        assert_eq!(
            weight.retained_bytes()?,
            canonical.len() / 2 + canonical.len() / 64 * 4
        );
        let actual = weight.decode()?;
        assert_eq!(actual.len(), expected.len());
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let before = actual;
        weight.quantize()?;
        assert_eq!(weight.decode()?, before);
        let mut incomplete = CompactWeight::from_bytes(&bytes(&[1.; 63]))?;
        assert!(incomplete.quantize().is_err());
        assert_eq!(incomplete.decode()?, vec![1.; 63]);
        Ok(())
    }
    #[test]
    fn ring_commit_matches_unbounded_history_without_reallocating_planes() -> Result<()> {
        let width = 3;
        let mut planes = vec![vec![0.; 2048 * width]; 10];
        let pointers: Vec<_> = planes.iter().map(|v| v.as_ptr()).collect();
        let mut histories: Vec<Vec<f32>> = vec![Vec::new(); 10];
        let mut start = 0;
        for rows in [1, 2046, 3, 2048, 2051, 17, 4097] {
            let update: Vec<Vec<f32>> = (0..10)
                .map(|p| {
                    (0..rows * width)
                        .map(|i| (p * 1000000 + start * width + i) as f32)
                        .collect()
                })
                .collect();
            let old_prefix = planes.clone();
            let mut destinations: Vec<_> = planes.iter_mut().map(Vec::as_mut_slice).collect();
            let sources: Vec<_> = update.iter().map(Vec::as_slice).collect();
            commit_ring(&mut destinations, &sources, start, rows, width)?;
            for p in 0..10 {
                histories[p].extend_from_slice(&update[p]);
                let end = start + rows;
                for token in end.saturating_sub(2048)..end {
                    for col in 0..width {
                        assert_eq!(
                            planes[p][token % 2048 * width + col],
                            histories[p][token * width + col]
                        );
                    }
                }
                assert_eq!(pointers[p], planes[p].as_ptr());
            }
            assert_ne!(old_prefix, planes); // A previously copied checkpoint remains independent.
            start += rows;
        }
        Ok(())
    }
    #[test]
    fn malformed_last_plane_and_nonfinite_producer_do_not_commit_earlier_layers() -> Result<()> {
        let mut planes = vec![vec![7.; 2048]; 10];
        let before = planes.clone();
        let mut updates = vec![vec![1.; 2]; 10];
        for fault in 0..4 {
            updates[9] = match fault {
                0 => vec![1.],
                1 => vec![f32::NAN, 1.],
                _ => vec![1.; 2],
            };
            let mut dst: Vec<_> = planes.iter_mut().map(Vec::as_mut_slice).collect();
            if fault == 2 {
                dst[9] = &mut [];
            }
            let src: Vec<_> = updates.iter().map(Vec::as_slice).collect();
            assert!(
                commit_ring(
                    &mut dst,
                    &src,
                    if fault == 3 { usize::MAX } else { 2047 },
                    2,
                    1
                )
                .is_err()
            );
            assert_eq!(planes, before);
        }
        let sources: Vec<_> = updates.iter().map(Vec::as_slice).collect();
        commit_ring(
            &mut planes.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(),
            &sources,
            2047,
            2,
            1,
        )?;
        assert!(
            planes
                .iter()
                .all(|p| p[2047] == 1. && p[0] == 1. && p[1] == 7.)
        );
        Ok(())
    }
}

#[cfg(kani)]
mod proofs {
    use super::*;
    #[kani::proof]
    #[kani::unwind(2050)]
    fn invalid_ring_producer_preserves_all_planes() {
        let start: u16 = kani::any();
        let mut plane = [3.; 2048];
        // A malformed producer is rejected for all starts before any mutation.
        assert!(commit_ring(&mut [&mut plane], &[&[]], usize::from(start), 1, 1).is_err());
        assert!(plane.iter().all(|&v| v == 3.));
        kani::cover!(start == 2047);
    }
}
