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
}
