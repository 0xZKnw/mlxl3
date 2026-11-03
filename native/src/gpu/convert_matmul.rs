//! Owned FP32 matrix views with selectable reference NAX/TF32 or full precision.
use super::{MatrixView, Metal, buffer, upload, view::FloatType};
use crate::tensor_layout::MatrixLayout;
use anyhow::{Context, Result, ensure};
use metal::{
    CommandBufferRef, CompileOptions, ComputePipelineState, Device, DeviceRef,
    MTLCommandBufferStatus, MTLResourceOptions as Storage, MTLSize,
};
use std::sync::atomic::{AtomicBool, Ordering};

#[allow(
    unexpected_cfgs,
    reason = "objc0.2 selector macro checks the legacy cargo-clippy cfg"
)]
fn reference_nax(device: &DeviceRef, full_fp32: bool) -> Result<bool> {
    use objc::{msg_send, runtime::Object, sel, sel_impl};
    use std::ffi::{CStr, c_char};
    // This compatibility flag selects the maintained converter's arithmetic;
    // it does not load or call the MLX runtime.
    let tf32 = std::env::var("MLX_ENABLE_TF32")
        .map_or(true, |v| v.trim().parse::<i32>().unwrap_or(1) != 0);
    if full_fp32 || !tf32 {
        return Ok(false);
    }
    let name = unsafe {
        let architecture: *mut Object = msg_send![device, architecture];
        ensure!(!architecture.is_null(), "missing Metal architecture");
        let name: *mut Object = msg_send![architecture, name];
        ensure!(!name.is_null(), "missing Metal architecture name");
        let text: *const c_char = msg_send![name, UTF8String];
        ensure!(!text.is_null(), "invalid Metal architecture name");
        CStr::from_ptr(text).to_str()?.to_owned()
    };
    let generation = name
        .strip_prefix("applegpu_g")
        .and_then(|v| v.get(..v.len().checked_sub(1)?))
        .and_then(|v| v.parse::<u32>().ok())
        .context("unknown Metal architecture")?;
    if generation < if name.ends_with('p') { 18 } else { 17 } {
        return Ok(false);
    }
    let version = std::process::Command::new("/usr/bin/sw_vers")
        .arg("-productVersion")
        .output()?;
    ensure!(version.status.success(), "could not read macOS version");
    let text = std::str::from_utf8(&version.stdout)?;
    let mut fields = text.trim().split('.');
    let major = fields
        .next()
        .context("missing macOS version")?
        .parse::<u32>()?;
    let minor = fields.next().unwrap_or("0").parse::<u32>()?;
    Ok((major, minor) >= (26, 2))
}

/// Reusable encoder for owned device views. Source weights may be strided;
/// output must not alias either input, and callers retain the views through
/// their own command completion boundary.
pub(crate) struct ConverterMatmul {
    pipeline: ComputePipelineState,
    tile: [u32; 2],
    device: u64,
    pub(crate) arithmetic: &'static str,
}
impl ConverterMatmul {
    #[allow(
        unexpected_cfgs,
        reason = "objc0.2 selector macro checks the legacy cargo-clippy cfg"
    )]
    pub(crate) fn new(device: &DeviceRef, full_fp32: bool) -> Result<Self> {
        use objc::{msg_send, sel, sel_impl};
        let nax = reference_nax(device, full_fp32)?;
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        if nax {
            // metal-rs0.33 predates the SDK's MTLLanguageVersion4_0 enum.
            unsafe {
                let _: () = msg_send![options.as_ref(), setLanguageVersion:0x40000usize];
            }
        }
        let (source, function, tile, arithmetic) = if nax {
            (
                include_str!("../../shaders/convert_matmul_nax.metal"),
                "mlxl3_convert_matmul_nax",
                [16, 32],
                "nax-tf32",
            )
        } else {
            (
                include_str!("../../shaders/convert_matmul.metal"),
                "mlxl3_convert_matmul",
                [8, 8],
                "simdgroup-fp32",
            )
        };
        let library = device
            .new_library_with_source(source, &options)
            .map_err(anyhow::Error::msg)?;
        let function = library
            .get_function(function, None)
            .map_err(anyhow::Error::msg)?;
        let pipeline = device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(anyhow::Error::msg)?;
        ensure!(
            pipeline.max_total_threads_per_threadgroup() >= 32
                && pipeline.thread_execution_width() == 32,
            "unsupported matrix SIMD width"
        );
        Ok(Self {
            pipeline,
            tile,
            device: device.registry_id(),
            arithmetic,
        })
    }

    pub(crate) fn encode(
        &self,
        command: &CommandBufferRef,
        left: &MatrixView,
        right: &MatrixView,
        output: &MatrixView,
    ) -> Result<()> {
        let [m, k] = left.shape();
        let [right_k, n] = right.shape();
        crate::convert_matmul::matrix_plan(m, n, k).context("invalid matrix geometry")?;
        ensure!(
            right_k == k && output.shape() == [m, n],
            "matrix shape mismatch"
        );
        for view in [left, right, output] {
            ensure!(
                view.dtype == FloatType::F32
                    && view.buffer.device().registry_id() == self.device
                    && view.layout.fits((view.buffer.length() / 4) as usize),
                "invalid converter FP32 device view"
            );
        }
        ensure!(
            output.layout.strides == [i64::from(n), 1] && output.layout.offset >= 0,
            "matrix output must be contiguous"
        );
        let a = left.layout;
        let b = right.layout;
        let parameters = [
            i64::from(m),
            i64::from(n),
            i64::from(k),
            a.offset,
            a.strides[0],
            a.strides[1],
            b.offset,
            b.strides[0],
            b.strides[1],
        ];
        Metal::encode_bound(
            command,
            &self.pipeline,
            [
                (&left.buffer, 0),
                (&right.buffer, 0),
                (&output.buffer, output.layout.offset as u64 * 4),
            ]
            .into_iter(),
            parameters,
            MTLSize::new(
                u64::from(n.div_ceil(self.tile[1])),
                u64::from(m.div_ceil(self.tile[0])),
                1,
            ),
            32,
        );
        Ok(())
    }
}

pub fn matmul(
    left: &[f32],
    right: &[f32],
    m: u32,
    n: u32,
    k: u32,
    full_fp32: bool,
    cancelled: &AtomicBool,
) -> Result<(Vec<f32>, &'static str)> {
    let (a, b, count) = crate::convert_matmul::matrix_plan(m, n, k)
        .context("invalid converter matrix dimensions")?;
    ensure!(
        left.len() == a && right.len() == b,
        "matrix input shape mismatch"
    );
    ensure!(
        left.iter().chain(right).all(|v| v.is_finite()),
        "matrix input must be finite"
    );
    ensure!(!cancelled.load(Ordering::Relaxed), "conversion cancelled");
    let device = Device::system_default().context("no Metal device")?;
    let queue = device.new_command_queue();
    let pipeline = ConverterMatmul::new(&device, full_fp32)?;
    let values = objc::rc::autoreleasepool(|| -> Result<Vec<f32>> {
        let left = MatrixView {
            buffer: upload(&device, left)?,
            dtype: FloatType::F32,
            layout: MatrixLayout {
                shape: [m, k],
                offset: 0,
                strides: [i64::from(k), 1],
            },
        };
        let right = MatrixView {
            buffer: upload(&device, right)?,
            dtype: FloatType::F32,
            layout: MatrixLayout {
                shape: [k, n],
                offset: 0,
                strides: [i64::from(n), 1],
            },
        };
        let output = MatrixView {
            buffer: buffer(&device, count as u64 * 4, Storage::StorageModeShared)?,
            dtype: FloatType::F32,
            layout: MatrixLayout {
                shape: [m, n],
                offset: 0,
                strides: [i64::from(n), 1],
            },
        };
        let command = queue.new_command_buffer();
        pipeline.encode(command, &left, &right, &output)?;
        command.commit();
        command.wait_until_completed();
        ensure!(
            command.status() == MTLCommandBufferStatus::Completed,
            "converter matrix failed ({:?})",
            command.status()
        );
        ensure!(!cancelled.load(Ordering::Relaxed), "conversion cancelled");
        Ok(output
            .read_output()?
            .into_iter()
            .map(f32::from_bits)
            .collect())
    })?;
    ensure!(
        values.len() == count && values.iter().all(|v| v.is_finite()),
        "nonfinite matrix result"
    );
    Ok((values, pipeline.arithmetic))
}

#[test]
fn conversion_matrix_rejects_bad_inputs_and_cancel_before_allocating_device() {
    let active = AtomicBool::new(false);
    assert!(matmul(&[], &[], 8, 8, 8, false, &active).is_err());
    assert!(matmul(&[f32::NAN; 64], &[0.; 64], 8, 8, 8, false, &active).is_err());
    let cancelled = AtomicBool::new(true);
    assert!(
        matmul(&[0.; 64], &[0.; 64], 8, 8, 8, false, &cancelled)
            .unwrap_err()
            .to_string()
            .contains("cancelled")
    );
}
