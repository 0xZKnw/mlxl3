//! The BF16 operator boundaries used by Flash's raw speculative draft.
use crate::checkpoint::TensorInfo;
use anyhow::{Context, Result, ensure};
use half::bf16;

#[derive(Clone, Debug)]
pub(crate) struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<f32>,
}

use crate::contracts::draft_dense_extent as extent;

pub(crate) fn round(value: f32) -> f32 {
    bf16::from_f32(value).to_f32()
}

impl Matrix {
    pub fn new(rows: usize, cols: usize, values: Vec<f32>) -> Result<Self> {
        ensure!(
            extent(rows, cols) == Some(values.len()) && values.iter().all(|v| v.is_finite()),
            "invalid draft matrix"
        );
        Ok(Self { rows, cols, values })
    }

    pub fn load(info: &TensorInfo) -> Result<Self> {
        ensure!(
            info.dtype == "BF16" && (1..=3).contains(&info.shape.len()),
            "draft weights must be BF16 with rank one to three"
        );
        let cols = *info.shape.last().context("missing draft width")?;
        let rows = info.shape[..info.shape.len() - 1]
            .iter()
            .try_fold(1usize, |n, &dim| n.checked_mul(dim))
            .context("draft weight extent overflow")?;
        let count = extent(rows, cols).context("invalid draft weight extent")?;
        let bytes = info.read_bytes()?;
        ensure!(
            count.checked_mul(2) == Some(bytes.len()),
            "truncated draft weight"
        );
        Self::new(
            rows,
            cols,
            bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| bf16::from_le_bytes(b).to_f32())
                .collect(),
        )
    }

    pub fn row(&self, index: usize) -> Result<Self> {
        ensure!(index < self.rows, "draft row out of bounds");
        Self::new(
            1,
            self.cols,
            self.values[index * self.cols..(index + 1) * self.cols].to_vec(),
        )
    }

    pub fn norm(&self, gain: &Self, width: usize) -> Result<Self> {
        ensure!(
            width > 0 && self.cols.is_multiple_of(width) && gain.rows == 1 && gain.cols == width,
            "invalid draft RMS shape"
        );
        let mut out = self.values.clone();
        for (input, output) in self
            .values
            .chunks_exact(width)
            .zip(out.chunks_exact_mut(width))
        {
            let inv = (input.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-6)
                .sqrt()
                .recip();
            for ((dst, &value), &gain) in output.iter_mut().zip(input).zip(&gain.values) {
                *dst = round((value * inv) * gain);
            }
        }
        Self::new(self.rows, self.cols, out)
    }

    pub fn residual_norm(&self, residual: &Self, gain: &Self) -> Result<(Self, Self)> {
        ensure!(
            self.rows == residual.rows
                && self.cols == residual.cols
                && gain.rows == 1
                && gain.cols == self.cols,
            "invalid draft residual norm shape"
        );
        let sum: Vec<_> = self
            .values
            .iter()
            .zip(&residual.values)
            .map(|(a, b)| a + b)
            .collect();
        let retained: Vec<_> = sum.iter().map(|&v| round(v)).collect();
        let mut normalized = retained.clone();
        for ((raw, rounded), output) in sum
            .chunks_exact(self.cols)
            .zip(retained.chunks_exact(self.cols))
            .zip(normalized.chunks_exact_mut(self.cols))
        {
            let inv = (raw.iter().map(|v| v * v).sum::<f32>() / self.cols as f32 + 1e-6)
                .sqrt()
                .recip();
            for ((dst, &value), &gain) in output.iter_mut().zip(rounded).zip(&gain.values) {
                *dst = round((value * gain) * inv);
            }
        }
        Ok((
            Self::new(self.rows, self.cols, normalized)?,
            Self::new(self.rows, self.cols, retained)?,
        ))
    }

    pub fn linear(&self, weight: &Self) -> Result<Self> {
        ensure!(
            extent(self.rows, self.cols) == Some(self.values.len())
                && extent(weight.rows, weight.cols) == Some(weight.values.len()),
            "invalid draft BLAS storage"
        );
        ensure!(self.cols == weight.cols, "draft projection shape mismatch");
        let count = extent(self.rows, weight.rows).context("draft projection extent overflow")?;
        let mut out = Vec::new();
        out.try_reserve_exact(count)?;
        out.resize(count, 0.);
        #[link(name = "Accelerate", kind = "framework")]
        unsafe extern "C" {
            fn cblas_sgemm(
                order: i32,
                trans_a: i32,
                trans_b: i32,
                m: i32,
                n: i32,
                k: i32,
                alpha: f32,
                a: *const f32,
                lda: i32,
                b: *const f32,
                ldb: i32,
                beta: f32,
                c: *mut f32,
                ldc: i32,
            );
        }
        // Validated dense row-major matrices. Only this call can write the fresh output.
        unsafe {
            cblas_sgemm(
                101,
                111,
                112,
                self.rows as i32,
                weight.rows as i32,
                self.cols as i32,
                1.,
                self.values.as_ptr(),
                self.cols as i32,
                weight.values.as_ptr(),
                weight.cols as i32,
                0.,
                out.as_mut_ptr(),
                weight.rows as i32,
            )
        };
        for value in &mut out {
            *value = round(*value);
        }
        Self::new(self.rows, weight.rows, out)
    }
}

impl Matrix {
    // Affine parameter rule from MLX 0.32.2 quantized.h (MIT), preserving the existing Q4 draft format.
    pub fn affine4_weights(&mut self) -> Result<()> {
        ensure!(
            self.cols.is_multiple_of(64),
            "Q4 draft projection requires complete groups of 64"
        );
        for group in self.values.as_chunks_mut::<64>().0 {
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
            let stored_scale = round(scale);
            let stored_bias = round(bias);
            for value in group {
                let code = ((*value - bias) / scale).round().clamp(0., 15.);
                *value = code * stored_scale + stored_bias;
            }
        }
        ensure!(
            self.values.iter().all(|v| v.is_finite()),
            "invalid Q4 draft affine parameters"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn residual_variance_uses_the_unrounded_sum_from_the_independent_counterexample() -> Result<()>
    {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../docs/measurements/flash-next-ssd-impl/opt-43-mtp-route-head/drafts-speed/residual-norm-counterexample.json"
        ))?;
        let case = &fixture["results"][0];
        let values = |name: &str| -> Vec<f32> {
            (0..2560)
                .map(|i| f32::from_bits(case[name][i % 16].as_u64().unwrap() as u32))
                .collect()
        };
        let x = Matrix::new(1, 2560, values("x_bits"))?;
        let residual = Matrix::new(1, 2560, values("residual_bits"))?;
        let gain = Matrix::new(1, 2560, vec![1.; 2560])?;
        let (actual, _) = x.residual_norm(&residual, &gain)?;
        let expected = values("guide_bits");
        assert_eq!(actual.values, expected);
        assert_ne!(actual.values, values("local_bits"));
        Ok(())
    }

    #[test]
    fn dense_projection_preserves_row_major_transpose_and_bf16_boundaries() -> Result<()> {
        let x = Matrix::new(2, 3, vec![1., 2., 3., -1., 4., 2.])?;
        let w = Matrix::new(2, 3, vec![1., 0., 2., -2., 1., 3.])?;
        let out = x.linear(&w)?;
        assert_eq!((out.rows, out.cols), (2, 2));
        assert_eq!(out.values, vec![7., 9., 3., 12.]);
        assert!(x.linear(&Matrix::new(1, 2, vec![1., 1.])?).is_err());
        assert!(Matrix::new(1, 1, vec![f32::NAN]).is_err());
        for dimensions in [(0, 2), (2, 0), (usize::MAX, 2), (i32::MAX as usize + 1, 1)] {
            assert!(extent(dimensions.0, dimensions.1).is_none());
        }
        let mut invalid = x.clone();
        invalid.values.pop();
        assert!(invalid.linear(&w).is_err());
        let residual = Matrix::new(1, 2, vec![round(1.003), round(1.01)])?;
        let gain = Matrix::new(1, 2, vec![1., 1.])?;
        let (normalized, retained) = gain.residual_norm(&residual, &gain)?;
        assert_eq!(
            retained.values,
            vec![
                round(1. + residual.values[0]),
                round(1. + residual.values[1])
            ]
        );
        assert!(normalized.values.iter().all(|v| v.is_finite()));
        Ok(())
    }
}
