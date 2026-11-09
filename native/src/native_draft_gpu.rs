//! Experimental compact-weight projection. CPU activation/cache ownership is unchanged.
fn projection_extent(
    rows: usize,
    width: usize,
    input_len: usize,
    outputs: usize,
    weight_width: usize,
    weight_len: usize,
) -> Option<usize> {
    let extent = crate::contracts::draft_dense_extent;
    if width != weight_width
        || extent(rows, width)? != input_len
        || extent(outputs, weight_width)? != weight_len
    {
        return None;
    }
    extent(rows, outputs)
}

#[cfg(all(target_os = "macos", feature = "direct-metal", feature = "chat"))]
pub(crate) use metal_backend::Backend;

#[cfg(all(target_os = "macos", feature = "direct-metal", feature = "chat"))]
mod metal_backend {
    use crate::{
        draft_storage::CompactWeight,
        native_bf16::{Matrix, Weight},
    };
    use anyhow::{Context, Result, ensure};
    use metal::{
        Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, MTLCommandBufferStatus,
        MTLResourceOptions, MTLSize,
    };

    const SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
kernel void draft_projection(device const float* x [[buffer(0)]],
    device const uchar* packed [[buffer(1)]],
    device const ushort2* params [[buffer(2)]],
    device float* out [[buffer(3)]],
    constant uint4& dims [[buffer(4)]], uint2 id [[thread_position_in_grid]]) {
    if (id.x >= dims.y || id.y >= dims.x) return;
    float sum = 0.0f;
    for (uint k = 0; k < dims.z; ++k) {
        ulong index = ulong(id.x) * dims.z + k;
        float w;
        if (dims.w == 0) {
            uint word = uint(packed[index * 2]) | (uint(packed[index * 2 + 1]) << 8);
            w = as_type<float>(word << 16);
        } else {
            uint code = (packed[index / 2] >> ((index & 1) * 4)) & 15;
            ushort2 p = params[index / 64];
            float scale = as_type<float>(uint(p.x) << 16);
            float bias = as_type<float>(uint(p.y) << 16);
            w = float(code) * scale + bias;
        }
        sum = sum + x[ulong(id.y) * dims.z + k] * w;
    }
    uint bits = as_type<uint>(sum);
    uint rounded = (bits + 0x7fffu + ((bits >> 16) & 1u)) & 0xffff0000u;
    // Preserve NaN/Inf so host finitude validation cannot mistake NaN for zero.
    if ((bits & 0x7f800000u) == 0x7f800000u) rounded = bits;
    out[ulong(id.y) * dims.y + id.x] = as_type<float>(rounded);
}
"#;

    pub(crate) struct Backend {
        device: Device,
        queue: CommandQueue,
        pipeline: ComputePipelineState,
        peak_bytes: std::cell::Cell<u64>,
    }

    impl Backend {
        pub fn new() -> Result<Self> {
            objc::rc::autoreleasepool(|| {
                let device =
                    Device::system_default().context("draft GPU requires a Metal device")?;
                let options = CompileOptions::new();
                options.set_fast_math_enabled(false);
                let library = device
                    .new_library_with_source(SOURCE, &options)
                    .map_err(anyhow::Error::msg)?;
                let function = library
                    .get_function("draft_projection", None)
                    .map_err(anyhow::Error::msg)?;
                let pipeline = device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(anyhow::Error::msg)?;
                let queue = device.new_command_queue();
                Ok(Self {
                    device,
                    queue,
                    pipeline,
                    peak_bytes: std::cell::Cell::new(0),
                })
            })
        }

        pub fn peak_bytes(&self) -> u64 {
            self.peak_bytes.get()
        }

        fn upload<T: Copy>(&self, values: &[T]) -> Result<Buffer> {
            let bytes = std::mem::size_of_val(values);
            let buffer = crate::gpu::buffer(
                &self.device,
                u64::try_from(bytes)?,
                MTLResourceOptions::StorageModeShared,
            )?;
            // Fresh exclusively owned shared allocation, no GPU producer or reader yet.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    values.as_ptr().cast::<u8>(),
                    buffer.contents().cast(),
                    bytes,
                );
            }
            Ok(buffer)
        }

        pub fn linear(&self, x: &Matrix, weight: &Weight) -> Result<Matrix> {
            // Headless callers need not provide a pool. All borrowed Metal
            // references and their resource retains retire before we return.
            objc::rc::autoreleasepool(|| {
                let Weight::Compact {
                    rows,
                    cols,
                    payload,
                } = weight
                else {
                    // Small dense operands retain the established CPU implementation.
                    return x.linear(weight.matrix()?.as_ref());
                };
                let extent = crate::contracts::draft_dense_extent;
                ensure!(
                    extent(x.rows, x.cols) == Some(x.values.len())
                        && x.values.iter().all(|v| v.is_finite()),
                    "invalid draft GPU activation"
                );
                ensure!(
                    x.cols == *cols && extent(*rows, *cols) == Some(payload.len()),
                    "invalid draft GPU projection shape"
                );
                let count = super::projection_extent(
                    x.rows,
                    x.cols,
                    x.values.len(),
                    *rows,
                    *cols,
                    payload.len(),
                )
                .context("draft GPU output extent overflow")?;
                let (packed, parameters, format) = match payload {
                    CompactWeight::Bf16(words) => {
                        ensure!(
                            words.iter().all(|&w| half::bf16::from_bits(w).is_finite()),
                            "invalid GPU BF16 weight"
                        );
                        (self.upload(words)?, self.upload(&[[0u16; 2]])?, 0u32)
                    }
                    CompactWeight::Affine4 { codes, parameters } => {
                        ensure!(
                            cols.is_multiple_of(64)
                                && parameters.len().checked_mul(32) == Some(codes.len()),
                            "invalid GPU affine storage"
                        );
                        // Validate actual reconstructed values without an FP32 weight allocation.
                        for (group, &[s, b]) in codes.chunks_exact(32).zip(parameters) {
                            let (s, b) = (
                                half::bf16::from_bits(s).to_f32(),
                                half::bf16::from_bits(b).to_f32(),
                            );
                            ensure!(
                                s.is_finite()
                                    && b.is_finite()
                                    && group
                                        .iter()
                                        .all(|&c| (f32::from(c & 15) * s + b).is_finite()
                                            && (f32::from(c >> 4) * s + b).is_finite()),
                                "invalid GPU affine values"
                            );
                        }
                        (self.upload(codes)?, self.upload(parameters)?, 1u32)
                    }
                };
                let input = self.upload(&x.values)?;
                let output = crate::gpu::buffer(
                    &self.device,
                    u64::try_from(count.checked_mul(4).context("draft GPU bytes overflow")?)?,
                    MTLResourceOptions::StorageModeShared,
                )?;
                let dimensions = [
                    u32::try_from(x.rows)?,
                    u32::try_from(*rows)?,
                    u32::try_from(*cols)?,
                    format,
                ];
                let bytes = [&packed, &parameters, &input, &output]
                    .into_iter()
                    .try_fold(0u64, |sum, buffer| {
                        sum.checked_add(buffer.length())
                            .context("draft GPU ledger overflow")
                    })?;
                self.peak_bytes.set(self.peak_bytes.get().max(bytes));
                let command = self.queue.new_command_buffer();
                let encoder = command.new_compute_command_encoder();
                encoder.set_compute_pipeline_state(&self.pipeline);
                for (index, buffer) in [&input, &packed, &parameters, &output]
                    .into_iter()
                    .enumerate()
                {
                    encoder.set_buffer(index as u64, Some(buffer), 0);
                }
                encoder.set_bytes(
                    4,
                    std::mem::size_of_val(&dimensions) as u64,
                    dimensions.as_ptr().cast(),
                );
                let width = self.pipeline.thread_execution_width();
                encoder.dispatch_threads(
                    MTLSize::new(*rows as u64, x.rows as u64, 1),
                    MTLSize::new(width, 1, 1),
                );
                encoder.end_encoding();
                command.commit();
                command.wait_until_completed();
                ensure!(
                    command.status() == MTLCommandBufferStatus::Completed,
                    "draft GPU projection failed: {:?}",
                    command.status()
                );
                // The sole producer completed successfully; the buffer is local and never shared.
                let values =
                    unsafe { std::slice::from_raw_parts(output.contents().cast::<f32>(), count) }
                        .to_vec();
                Matrix::new(x.rows, *rows, values)
            })
        }
    }
}

#[cfg(test)]
mod admission_tests {
    use super::projection_extent;
    #[test]
    fn production_projection_admission_rejects_invalid_layouts() {
        assert_eq!(projection_extent(2, 64, 128, 3, 64, 192), Some(6));
        for shape in [
            (0, 64, 0, 3, 64, 192),
            (2, 64, 127, 3, 64, 192),
            (2, 64, 128, 3, 63, 189),
            (2, 64, 128, 3, 64, 191),
            (usize::MAX, 1, usize::MAX, 1, 1, 1),
            (1, 1, 1, i32::MAX as usize + 1, 1, i32::MAX as usize + 1),
        ] {
            assert!(
                projection_extent(shape.0, shape.1, shape.2, shape.3, shape.4, shape.5).is_none()
            );
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn draft_gpu_projection_admission_matches_dimensions() {
    let rows = kani::any::<u8>() as usize;
    let width = kani::any::<u8>() as usize;
    let outputs = kani::any::<u8>() as usize;
    let weight_width = kani::any::<u8>() as usize;
    let input_len = kani::any::<u16>() as usize;
    let weight_len = kani::any::<u16>() as usize;
    let result = projection_extent(rows, width, input_len, outputs, weight_width, weight_len);
    let valid = rows != 0
        && width != 0
        && outputs != 0
        && width == weight_width
        && input_len == rows * width
        && weight_len == outputs * width;
    assert_eq!(result.is_some(), valid);
    if let Some(count) = result {
        assert_eq!(count, rows * outputs);
    }
    kani::cover!(result.is_some() && rows > 1 && outputs != width);
    kani::cover!(result.is_none() && width != weight_width);
}

#[cfg(all(test, target_os = "macos", feature = "direct-metal", feature = "chat"))]
mod tests {
    use super::*;
    use crate::{
        draft_storage::CompactWeight,
        native_bf16::{Matrix, Weight},
    };
    use anyhow::Result;

    #[test]
    #[ignore = "requires a physical Apple GPU; explicit compact projection oracle"]
    fn physical_compact_projection_formats_shapes_rounding_and_errors() -> Result<()> {
        let backend = Backend::new()?;
        let values: Vec<f32> = (0..192).map(|i| (i % 7) as f32 - 3.).collect();
        let bytes: Vec<_> = values
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let mut weight = Weight::Compact {
            rows: 3,
            cols: 64,
            payload: CompactWeight::from_bytes(&bytes)?,
        };
        let mut x = Matrix::new(2, 64, (0..128).map(|i| (i % 3) as f32 - 1.).collect())?;
        for affine in [false, true] {
            if affine {
                weight.quantize("fc.weight")?;
            }
            let dense = weight.matrix()?;
            let actual = backend.linear(&x, &weight)?;
            assert_eq!((actual.rows, actual.cols, actual.values.len()), (2, 3, 6));
            // Independent f64 accumulation, exact for these small dyadic operands.
            let expected: Vec<_> = x
                .values
                .chunks_exact(64)
                .flat_map(|a| {
                    dense.values.chunks_exact(64).map(move |b| {
                        crate::native_bf16::round(
                            a.iter()
                                .zip(b)
                                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                                .sum::<f64>() as f32,
                        )
                    })
                })
                .collect();
            assert_eq!(actual.values, expected);
            assert_eq!(actual.values, x.linear(dense.as_ref())?.values);
        }
        let tie = Weight::Compact {
            rows: 2,
            cols: 1,
            payload: CompactWeight::from_bytes(&[0x80, 0x3f, 0x80, 0x3f])?,
        };
        let tied = backend.linear(&Matrix::new(2, 1, vec![1.00390625, 1.01171875])?, &tie)?;
        assert_eq!(tied.values, vec![1., 1., 1.015625, 1.015625]);
        assert!(
            backend
                .linear(&Matrix::new(1, 2, vec![1.; 2])?, &weight)
                .is_err()
        );
        x.values[0] = f32::NAN;
        assert!(backend.linear(&x, &weight).is_err());
        let malformed = Weight::Compact {
            rows: 1,
            cols: 64,
            payload: CompactWeight::Affine4 {
                codes: vec![0; 32],
                parameters: vec![],
            },
        };
        assert!(
            backend
                .linear(&Matrix::new(1, 64, vec![1.; 64])?, &malformed)
                .is_err()
        );
        Ok(())
    }
}
