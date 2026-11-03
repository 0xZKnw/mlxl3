use super::{Metal, buffer, projection::submit_encoded, upload};
use crate::{contracts, tensor_layout::MatrixLayout};
use anyhow::{Context, Result, ensure};
use metal::{Buffer, CommandBufferRef, MTLResourceOptions as Storage, MTLSize};

#[derive(Clone, Copy, PartialEq)]
enum FloatType {
    F16,
    F32,
}

/// Initialized immutable storage. Every view retains the underlying MTLBuffer.
pub struct TensorStorage {
    buffer: Buffer,
    elements: usize,
    dtype: FloatType,
}

pub struct MatrixView {
    pub(super) buffer: Buffer,
    pub(super) layout: MatrixLayout,
    dtype: FloatType,
}

impl TensorStorage {
    pub fn matrix(&self, layout: MatrixLayout) -> Result<MatrixView> {
        ensure!(
            layout.fits(self.elements),
            "matrix view outside initialized storage"
        );
        Ok(MatrixView {
            buffer: self.buffer.clone(),
            layout,
            dtype: self.dtype,
        })
    }
}

impl MatrixView {
    pub(super) fn column_scales(buffer: &Buffer, rows: u32, cols: u32) -> Result<Self> {
        ensure!(
            buffer.length() == 2 * u64::from(cols),
            "invalid column scale allocation"
        );
        Ok(Self {
            buffer: buffer.clone(),
            layout: MatrixLayout {
                shape: [rows, cols],
                strides: [0, 1],
                offset: 0,
            },
            dtype: FloatType::F16,
        })
    }
}

impl Metal {
    pub fn f16_storage(&self, bits: &[u16]) -> Result<TensorStorage> {
        Ok(TensorStorage {
            buffer: upload(&self.device, bits)?,
            elements: bits.len(),
            dtype: FloatType::F16,
        })
    }
    pub fn f32_storage(&self, values: &[f32]) -> Result<TensorStorage> {
        Ok(TensorStorage {
            buffer: upload(&self.device, values)?,
            elements: values.len(),
            dtype: FloatType::F32,
        })
    }

    fn view_blocks(&self, x: &MatrixView, scales: &MatrixView) -> Result<u32> {
        let [rows, cols] = x.layout.shape;
        ensure!(
            x.layout.shape == scales.layout.shape && scales.dtype == FloatType::F16,
            "invalid H128 view shape or scale dtype"
        );
        ensure!(
            x.buffer.device().registry_id() == self.device.registry_id()
                && scales.buffer.device().registry_id() == self.device.registry_id(),
            "view belongs to a different Metal device"
        );
        let n = rows.checked_mul(cols).context("H128 view size overflow")?;
        contracts::prepare128_blocks(n as usize, cols as usize, cols as usize)
            .context("H128 view width must be positive and 128-aligned")
    }

    pub(super) fn encode_prepare_view(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        scales: &MatrixView,
        output: &Buffer,
    ) -> Result<()> {
        let blocks = self.view_blocks(x, scales)?;
        ensure!(
            output.length() == u64::from(blocks) * 256 && blocks > 0,
            "invalid H128 view output allocation"
        );
        ensure!(
            self.prepare_view_pipeline
                .max_total_threads_per_threadgroup()
                >= 64,
            "Metal H128 requires 64 threads"
        );
        let binding = [
            i64::from(x.layout.shape[1]),
            x.layout.offset,
            x.layout.strides[0],
            x.layout.strides[1],
            scales.layout.offset,
            scales.layout.strides[0],
            scales.layout.strides[1],
            i64::from(x.dtype == FloatType::F32),
        ];
        Self::encode(
            cmd,
            &self.prepare_view_pipeline,
            &[&x.buffer, &scales.buffer, output],
            binding,
            MTLSize::new(u64::from(blocks), 1, 1),
            64,
        );
        Ok(())
    }

    pub fn prepare_view(&self, x: &MatrixView, scales: &MatrixView) -> Result<Vec<u16>> {
        let blocks = self.view_blocks(x, scales)?;
        if blocks == 0 {
            return Ok(Vec::new());
        }
        submit_encoded(self, |cmd| {
            let output = buffer(
                &self.device,
                u64::from(blocks) * 256,
                Storage::StorageModeShared,
            )?;
            self.encode_prepare_view(cmd, x, scales, &output)?;
            Ok(output)
        })?
        .read()
    }
}
