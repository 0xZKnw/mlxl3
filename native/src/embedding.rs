//! Reversible BF16 storage. The final conversion uses the canonical MLX cast.
use crate::{
    array::{Array, Dtype, metal_kernel},
    checkpoint::Checkpoint,
    linear::checkpoint_array,
};
use anyhow::{Context, Result, ensure};

pub(crate) enum Embedding {
    Dense(Array),
    Packed {
        shape: [i32; 2],
        payload: Array,
        metadata: Array,
    },
}

impl Embedding {
    pub(crate) fn load(
        checkpoint: &Checkpoint,
        name: &str,
        shape: [i32; 2],
        packed: bool,
    ) -> Result<Self> {
        let weight = checkpoint_array(checkpoint, name)?;
        ensure!(weight.shape() == shape, "invalid embedding shape");
        Self::from_weight(weight, packed)
    }

    pub(crate) fn from_weight(weight: Array, packed: bool) -> Result<Self> {
        ensure!(
            weight.shape().len() == 2
                && weight.shape().iter().all(|&d| d > 0)
                && matches!(
                    weight.dtype(),
                    Dtype::BFloat16 | Dtype::Float16 | Dtype::Float32
                ),
            "embedding requires a nonempty floating-point matrix"
        );
        if packed
            && weight.dtype() == Dtype::BFloat16
            && weight.shape()[1] % 128 == 0
            && let Some([payload, metadata]) = weight.pack_bf16_embedding()?
        {
            return Ok(Self::Packed {
                shape: [weight.shape()[0], weight.shape()[1]],
                payload,
                metadata,
            });
        }
        let weight = weight.astype(Dtype::Float16)?;
        weight.eval()?;
        Ok(Self::Dense(weight))
    }

    pub(crate) fn shape(&self) -> &[i32] {
        match self {
            Self::Dense(a) => a.shape(),
            Self::Packed { shape, .. } => shape,
        }
    }

    pub(crate) fn take(&self, ids: &Array, axis: i32) -> Result<Array> {
        match self {
            Self::Dense(a) => a.take(ids, axis),
            Self::Packed {
                shape,
                payload,
                metadata,
            } => {
                ensure!(
                    axis == 0 && matches!(ids.dtype(), Dtype::Int32 | Dtype::UInt32),
                    "invalid packed embedding IDs"
                );
                let mut output_shape = ids.shape().to_vec();
                output_shape.push(shape[1]);
                let count = output_shape
                    .iter()
                    .try_fold(1i32, |n, &d| n.checked_mul(d))
                    .filter(|&n| n > 0)
                    .context("packed embedding output is empty or too large")?;
                let header = format!(
                    "using namespace metal;\n#define WIDTH {}u\n#define VOCAB {}u\n#define OUTPUT_COUNT {}u\n",
                    shape[1], shape[0], count
                );
                let output = metal_kernel(
                    "mlxl3_lossless_bf16_embedding_v1",
                    &["payload", "metadata", "ids"],
                    &["output"],
                    &header,
                    include_str!("../shaders/lossless_embedding.metal"),
                    &[payload, metadata, ids],
                    &[output_shape],
                    &[Dtype::UInt16],
                    [count, 1, 1],
                    [256, 1, 1],
                )?
                .pop()
                .context("missing packed embedding output")?;
                // A byte view, then the very same cast as half_weight. This also
                // preserves MLX's handling of subnormals, infinities and NaNs.
                output.view(Dtype::BFloat16)?.astype(Dtype::Float16)
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn retained_bytes(&self) -> Result<usize> {
        match self {
            Self::Dense(a) => a.retained_bytes(),
            Self::Packed {
                payload, metadata, ..
            } => payload
                .retained_bytes()?
                .checked_add(metadata.retained_bytes()?)
                .context("embedding size overflow"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "physical Apple GPU; run alone"]
    fn lossless_embedding_preserves_every_bf16_pattern_and_cast() -> Result<()> {
        let bytes = (0..65_536u32)
            .flat_map(|v| (v as u16).to_ne_bytes())
            .collect::<Vec<_>>();
        let raw = Array::from_bytes(&bytes, &[512, 128], Dtype::BFloat16)?;
        let expected = raw.astype(Dtype::Float16)?;
        let packed = Embedding::from_weight(raw.try_clone()?, true)?;
        ensure!(
            matches!(packed, Embedding::Packed { .. }),
            "fixture must compress"
        );
        assert!(packed.retained_bytes()? < raw.retained_bytes()?);
        for dtype in [Dtype::Int32, Dtype::UInt32] {
            for reverse in [false, true] {
                let ids = (0..512)
                    .map(|i| if reverse { 511 - i } else { i })
                    .collect::<Vec<i32>>();
                let ids = Array::from_i32(&ids, &[16, 32])?.astype(dtype)?;
                let actual = packed.take(&ids, 0)?;
                assert_eq!(actual.shape(), &[16, 32, 128]);
                assert_eq!(actual.dtype(), Dtype::Float16);
                assert_eq!(actual.to_bytes()?, expected.take(&ids, 0)?.to_bytes()?);
            }
        }
        let ids = Array::from_i32(&[511, 0, 511, 255], &[4])?;
        assert_eq!(
            packed.take(&ids, 0)?.to_bytes()?,
            expected.take(&ids, 0)?.to_bytes()?
        );
        assert!(packed.take(&ids, 1).is_err());
        assert!(packed.take(&Array::from_f32(&[0.], &[1])?, 0).is_err());
        assert!(
            Embedding::from_weight(Array::zeros_dtype(&[1, 128], Dtype::UInt16)?, true).is_err()
        );
        assert!(Embedding::from_weight(Array::zeros(&[0, 128])?, true).is_err());
        assert!(matches!(
            Embedding::from_weight(Array::zeros(&[1, 127])?, true)?,
            Embedding::Dense(_)
        ));
        // Every block spans all eight exponent bits, making packing unprofitable.
        let bits = (0..128u16)
            .map(|i| if i % 2 == 0 { 0 } else { 0x7f80 })
            .flat_map(u16::to_ne_bytes)
            .collect::<Vec<_>>();
        assert!(matches!(
            Embedding::from_weight(Array::from_bytes(&bits, &[1, 128], Dtype::BFloat16)?, true)?,
            Embedding::Dense(_)
        ));
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU; run alone"]
    fn lossless_embedding_variable_widths_cross_bytes() -> Result<()> {
        let mut bits = Vec::new();
        for width in 0..=8u32 {
            let span = (1u16 << width) - 1;
            let base = if width == 8 { 0 } else { 29 };
            for i in 0..128u16 {
                let delta = if i == 127 { span } else { (i * 37) & span };
                let value = ((base + delta) << 7) | (i & 127) | ((i & 1) << 15);
                bits.extend_from_slice(&value.to_ne_bytes());
            }
        }
        let raw = Array::from_bytes(&bits, &[9, 128], Dtype::BFloat16)?;
        let expected = raw.astype(Dtype::Float16)?;
        let packed = Embedding::from_weight(raw, true)?;
        ensure!(
            matches!(packed, Embedding::Packed { .. }),
            "mixed fixture must compress"
        );
        let ids = Array::from_i32(&[8, 0, 7, 1, 6, 2, 5, 3, 4, 8], &[10])?;
        let actual = packed.take(&ids, 0)?;
        assert_eq!(actual.shape(), &[10, 128]);
        assert_eq!(actual.to_bytes()?, expected.take(&ids, 0)?.to_bytes()?);
        Ok(())
    }
}
