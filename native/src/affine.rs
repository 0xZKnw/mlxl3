//! MLX affine 4-bit projections used by native Qwen MTP sidecars.
//! The backend's QMM/gather-QMM kernels run on every supported Apple GPU.
use crate::{
    array::{Array, Dtype},
    checkpoint::Checkpoint,
    linear::checkpoint_array,
};
use anyhow::{Result, ensure};

pub struct AffineLinear {
    weight: Array,
    scales: Array,
    biases: Array,
    input: i32,
    output: i32,
    experts: Option<i32>,
}

impl AffineLinear {
    pub fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        input: i32,
        output: i32,
        experts: Option<i32>,
    ) -> Result<Self> {
        ensure!(
            input > 0 && input % 64 == 0 && output > 0 && experts.is_none_or(|n| n > 0),
            "invalid affine dimensions"
        );
        let weight = checkpoint_array(checkpoint, &format!("{prefix}.weight"))?;
        let scales =
            checkpoint_array(checkpoint, &format!("{prefix}.scales"))?.astype(Dtype::Float16)?;
        let biases =
            checkpoint_array(checkpoint, &format!("{prefix}.biases"))?.astype(Dtype::Float16)?;
        let mut packed = experts.into_iter().collect::<Vec<_>>();
        packed.extend([output, input / 8]);
        let mut groups = experts.into_iter().collect::<Vec<_>>();
        groups.extend([output, input / 64]);
        ensure!(
            weight.dtype() == Dtype::UInt32
                && weight.shape() == packed
                && scales.shape() == groups
                && biases.shape() == groups,
            "{prefix}: invalid affine weights"
        );
        weight.eval()?;
        scales.eval()?;
        biases.eval()?;
        Ok(Self {
            weight,
            scales,
            biases,
            input,
            output,
            experts,
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        ensure!(
            self.experts.is_none()
                && x.shape().last() == Some(&self.input)
                && x.dtype() == Dtype::Float16,
            "invalid affine input"
        );
        x.affine4(&self.weight, &self.scales, &self.biases, None)
    }

    pub fn gather(&self, x: &Array, indices: &Array) -> Result<Array> {
        ensure!(
            self.experts.is_some()
                && x.shape().last() == Some(&self.input)
                && indices.dtype() == Dtype::UInt32
                && x.dtype() == Dtype::Float16,
            "invalid affine expert input"
        );
        // Router output is checked here too: malformed public calls never reach
        // an unchecked GPU gather. This synchronizes only the small index array.
        ensure!(
            indices
                .to_u32()?
                .iter()
                .all(|&id| id < self.experts.unwrap() as u32),
            "affine expert outside range"
        );
        x.affine4(&self.weight, &self.scales, &self.biases, Some(indices))
    }

    pub fn output_dims(&self) -> i32 {
        self.output
    }

    /// The router owns the range guarantee; callers cannot supply arbitrary IDs.
    /// Public `gather` above retains its checks for external arrays.
    pub(crate) fn gather_routed(
        &self,
        x: &Array,
        routes: &crate::router::RoutedExperts,
    ) -> Result<Array> {
        ensure!(
            self.experts == Some(routes.experts())
                && x.shape().last() == Some(&self.input)
                && x.dtype() == Dtype::Float16,
            "affine routes do not match expert weights/input"
        );
        x.affine4(
            &self.weight,
            &self.scales,
            &self.biases,
            Some(routes.indices()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    #[ignore = "physical Apple GPU"]
    fn routed_gather_matches_checked_and_rejects_wrong_experts() -> Result<()> {
        let layer = AffineLinear {
            weight: Array::from_u32(&[0x76543210; 128], &[2, 8, 8])?,
            scales: Array::from_f16_bits(&[f16::from_f32(0.25).to_bits(); 16], &[2, 8, 1])?,
            biases: Array::from_f16_bits(&[f16::from_f32(-1.0).to_bits(); 16], &[2, 8, 1])?,
            input: 64,
            output: 8,
            experts: Some(2),
        };
        let x = Array::from_f16_bits(&[f16::ONE.to_bits(); 64], &[1, 1, 1, 64])?;
        for k in 1..=2 {
            let probabilities = Array::from_f16_bits(&[0x3400, 0x3a00], &[1, 2])?;
            let routes = crate::router::routes(&probabilities, k, true)?;
            assert_eq!(
                layer.gather_routed(&x, &routes)?.to_f16_bits()?,
                layer.gather(&x, routes.indices())?.to_f16_bits()?
            );
            assert!(
                layer
                    .gather_routed(&x.astype(Dtype::Float32)?, &routes)
                    .is_err()
            );
        }
        let wrong = Array::from_f16_bits(&[0x3400, 0x3800, 0x3a00], &[1, 3])?;
        assert!(
            layer
                .gather_routed(&x, &crate::router::routes(&wrong, 1, true)?)
                .is_err()
        );
        assert!(layer.gather(&x, &Array::from_u32(&[2], &[1, 1])?).is_err());
        assert!(crate::router::routes(&wrong, 0, true).is_err());
        Ok(())
    }
}
