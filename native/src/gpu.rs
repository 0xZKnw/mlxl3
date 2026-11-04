//! Direct Metal ownership. No MLX/Python runtime; synchronous reference API.
use crate::{
    codec::{self, Codebook},
    contracts,
};
use anyhow::{Context, Result, ensure};
use metal::{
    Buffer, CommandBufferRef, CommandQueue, CompileOptions, ComputePipelineState, Device,
    DeviceRef, MTLCommandBufferStatus, MTLResourceOptions as Storage, MTLSize,
    foreign_types::ForeignType,
};

use objc::{msg_send, sel, sel_impl};
pub struct FlashAttentionOutput {
    pub values: MatrixView,
    pub trace: Option<std::collections::BTreeMap<&'static str, Vec<u32>>>,
}

mod model_ops;
mod qsa;
pub use qsa::{FlashQsa, FlashQsaLayer, FlashQsaState};
mod flash;
pub use flash::{FlashMtpHead, FlashMtpSession, FlashNext, FlashSnapshot, FlashTraceTensor};
mod ple;
pub use ple::{FlashPle, FlashPleLayer, FlashPleLayerState, FlashPleState};
mod qwen_gdn;
pub use qwen_gdn::{QwenGdn, QwenGdnState, QwenGdnTrace};
mod gdn;
pub use gdn::{FlashGdn, FlashGdnLayer, FlashGdnOutput, FlashGdnState};
mod hc;
pub use hc::{FlashHc, FlashHcRead};
mod memory;
pub use memory::MemoryLedger;
mod moe;
pub use moe::{FlashMoeRow, MoeOutput};
mod gate;
pub use gate::{FlashGate, FlashGateTrace};
mod router;
pub use router::RouteOutput;
mod view;
pub use view::{MatrixView, TensorStorage};
pub mod convert_matmul;
mod projection;
pub mod quantize;
pub use projection::{
    FlashExpert, FlashProjection, MappedSwitchGlu, ProjectionGroup, ProjectionOutput,
};

#[allow(
    unexpected_cfgs,
    reason = "objc 0.2 selector macro tests the legacy cargo-clippy cfg"
)]
fn buffer(device: &DeviceRef, length: u64, storage: Storage) -> Result<Buffer> {
    ensure!(
        length > 0 && length <= device.max_buffer_length(),
        "invalid Metal buffer length {length}"
    );
    // Metal may return nil on allocation failure. A successful new method returns +1.
    let ptr: *mut objc::runtime::Object =
        unsafe { msg_send![device, newBufferWithLength:length options:storage] };
    ensure!(!ptr.is_null(), "Metal could not allocate {length} bytes");
    Ok(unsafe { Buffer::from_ptr(ptr.cast()) })
}

fn float_values(info: &crate::checkpoint::TensorInfo) -> Result<Vec<f32>> {
    use anyhow::bail;
    use half::{bf16, f16};
    let bytes = info.read_bytes()?;
    Ok(match info.dtype.as_str() {
        "F16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| f16::from_le_bytes(v).to_f32())
            .collect(),
        "BF16" => bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| bf16::from_le_bytes(v).to_f32())
            .collect(),
        "F32" => bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&v| f32::from_le_bytes(v))
            .collect(),
        _ => bail!("expected floating HC weights"),
    })
}
fn checkpoint_float_weight(
    gpu: &Metal,
    c: &crate::checkpoint::Checkpoint,
    name: &str,
    shape: &[usize],
    dtype: view::FloatType,
) -> Result<Buffer> {
    let tensor = c
        .tensors
        .get(name)
        .with_context(|| format!("missing {name}"))?;
    ensure!(tensor.shape == shape, "invalid shape for {name}");
    if let Some(p) = name.strip_suffix(".weight") {
        ensure!(
            !c.tensors.contains_key(&format!("{p}.bias")),
            "unexpected bias for {name}"
        );
    }
    let count = shape
        .iter()
        .try_fold(1usize, |n, width| n.checked_mul(*width))
        .context("weight extent overflow")?;
    let (kind, bytes) = if dtype == view::FloatType::F16 {
        ("F16", 2)
    } else {
        ("F32", 4)
    };
    if tensor.dtype == kind {
        return read_weight(
            &gpu.device,
            tensor,
            (count as u64)
                .checked_mul(bytes)
                .context("weight bytes overflow")?,
        );
    }
    let values = float_values(tensor)?;
    ensure!(values.len() == count, "truncated {name}");
    if dtype == view::FloatType::F16 {
        upload(
            &gpu.device,
            &values
                .into_iter()
                .map(|v| half::f16::from_f32(v).to_bits())
                .collect::<Vec<_>>(),
        )
    } else {
        upload(&gpu.device, &values)
    }
}

fn read_weight(
    device: &DeviceRef,
    tensor: &crate::checkpoint::TensorInfo,
    bytes: u64,
) -> Result<Buffer> {
    ensure!(
        tensor.data_offsets[1].checked_sub(tensor.data_offsets[0]) == Some(bytes),
        "weight byte count mismatch"
    );
    let (file, offset) = tensor.source()?;
    read_buffer(device, file, offset, bytes)
}

fn read_buffer(
    device: &DeviceRef,
    file: &std::fs::File,
    offset: u64,
    bytes: u64,
) -> Result<Buffer> {
    use std::os::unix::fs::FileExt;
    let packed = buffer(device, bytes, Storage::StorageModeShared)?;
    // The inspected inode is pinned. No handle is published before read_exact_at succeeds.
    let dest = unsafe {
        std::slice::from_raw_parts_mut(packed.contents().cast::<u8>(), usize::try_from(bytes)?)
    };
    file.read_exact_at(dest, offset)?;
    Ok(packed)
}

fn upload<T: Copy>(device: &DeviceRef, data: &[T]) -> Result<Buffer> {
    let bytes = std::mem::size_of_val(data);
    let dest = buffer(device, (bytes as u64).max(1), Storage::StorageModeShared)?;
    // The private-to-this-function allocation is shared-storage and unpublished.
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dest.contents().cast(), data.len()) };
    Ok(dest)
}

pub struct Metal {
    device: Device,
    queue: CommandQueue,
    hc_chain: std::cell::Cell<bool>,
    #[cfg(test)]
    helper_commands: std::cell::Cell<usize>,
    #[cfg(test)]
    hc_chain_pairs: std::cell::Cell<usize>,
    pack_pipeline: ComputePipelineState,
    decode_pipeline: ComputePipelineState,
    codebook_pipeline: ComputePipelineState,
    qmv_pipeline: ComputePipelineState,
    prepare128_pipeline: ComputePipelineState,
    flash_qmv_pipeline: ComputePipelineState,
    flash_qmv_nt2: ComputePipelineState,
    flash_static_qmv: [ComputePipelineState; 5],
    flash_static_supported: [bool; 5],
    flash_static_setting: Option<bool>,
    flash_fixed_qmv: Option<[ComputePipelineState; 2]>,
    flash_fixed_setting: Option<bool>,
    #[cfg(test)]
    flash_fixed_submissions: [std::cell::Cell<u64>; 2],
    #[cfg(test)]
    flash_static_submissions: [std::cell::Cell<u64>; 5],
    dense_qmv: [[ComputePipelineState; 3]; 2],
    dense_finish: ComputePipelineState,
    mapped_glu: ComputePipelineState,
    mapped_mix: ComputePipelineState,
    mtp_hidden_qmv: ComputePipelineState,
    mtp_norm: ComputePipelineState,
    mtp_combine: ComputePipelineState,
    finish128_pipeline: ComputePipelineState,
    prepare_view_pipeline: ComputePipelineState,
    swiglu_pipeline: ComputePipelineState,
    k7_pipeline: ComputePipelineState,
    hc: [hc::Pipelines; 2],
    qsa: qsa::Pipelines,
    model_ops: model_ops::Pipelines,
    gdn: gdn::Pipelines,
    logsumexp: ComputePipelineState,
    ple: ple::Pipelines,
    moe_switch: [ComputePipelineState; 2],
    moe_mix: [ComputePipelineState; 2],
    gate_dot: [ComputePipelineState; 2],
    gate_softmax: [ComputePipelineState; 2],
    route_half: ComputePipelineState,
    route_float: ComputePipelineState,
    inverse: Buffer,
    readers: std::cell::OnceCell<crate::ssd_reads::ReadService>,
}

impl Metal {
    fn readers(&self) -> Result<&crate::ssd_reads::ReadService> {
        if self.readers.get().is_none() {
            let count = std::env::var("MLXL3_SSD_READERS")
                .unwrap_or_else(|_| "4".into())
                .parse::<usize>()
                .context("invalid SSD reader count")?;
            self.readers
                .set(crate::ssd_reads::ReadService::new(count)?)
                .map_err(|_| anyhow::anyhow!("native SSD readers already initialized"))?;
        }
        self.readers.get().context("missing native SSD readers")
    }

    pub fn new() -> Result<Self> {
        objc::rc::autoreleasepool(|| {
            let device = Device::system_default().context("no Metal device")?;
            let qualified_m5 = device.name() == "Apple M5";
            let hc_chain = match std::env::var("MLXL3_NATIVE_AR_HC_CHAIN") {
                Ok(value) => crate::flash_support::hc_chain_flag(Some(&value), qualified_m5),
                Err(std::env::VarError::NotPresent) => {
                    crate::flash_support::hc_chain_flag(None, qualified_m5)
                }
                Err(error) => return Err(error.into()),
            }
            .context("MLXL3_NATIVE_AR_HC_CHAIN must be 0 or 1")?;
            // Snapshot once; cold expert records never re-read this environment.
            // Defer malformed-setting errors to eligible projection loads.
            let flash_static_setting =
                if device.name() == "Apple M5" || device.name().starts_with("Apple M5 ") {
                    match std::env::var("MLXL3_NATIVE_FLASH_STATIC_K") {
                        Ok(value) => crate::flash_qmv::flag(Some(&value)),
                        Err(std::env::VarError::NotPresent) => crate::flash_qmv::flag(None),
                        Err(std::env::VarError::NotUnicode(_)) => None,
                    }
                } else {
                    Some(false)
                };
            let flash_fixed_setting = if qualified_m5 {
                match std::env::var("MLXL3_NATIVE_FLASH_FIXED_DIMS") {
                    Ok(value) => crate::flash_qmv::fixed_flag(Some(&value)),
                    Err(std::env::VarError::NotPresent) => crate::flash_qmv::fixed_flag(None),
                    Err(std::env::VarError::NotUnicode(_)) => None,
                }
            } else {
                Some(false)
            };
            let options = CompileOptions::new();
            options.set_fast_math_enabled(false);
            let source = include_str!("../shaders/exl3.metal")
                .replace(
                    "#include \"mapped_expert.metal\"",
                    include_str!("../shaders/mapped_expert.metal"),
                )
                .replace(
                    "#include \"dense_projection.metal\"",
                    include_str!("../shaders/dense_projection.metal"),
                )
                .replace(
                    "#include \"flash_projection.metal\"",
                    include_str!("../shaders/flash_projection.metal"),
                )
                .replace(
                    "#include \"flash_static_qmv.metal\"",
                    include_str!("../shaders/flash_static_qmv.metal"),
                )
                .replace(
                    "#include \"_qmv_tile_kernel.metal\"",
                    include_str!("../shaders/_qmv_tile_kernel.metal"),
                )
                .replace(
                    "#include \"flash_reference_reduction.h\"",
                    include_str!("../shaders/flash_reference_reduction.h"),
                )
                .replace(
                    "#include \"router_topk.h\"",
                    include_str!("../shaders/router_topk.h"),
                );
            let source = source.replace(
                "#include \"model_ops.metal\"",
                include_str!("../shaders/model_ops.metal"),
            );
            let source = source.replace(
                "#include \"flash_gate.metal\"",
                include_str!("../shaders/flash_gate.metal"),
            );
            let source = source.replace(
                "#include \"flash_moe.metal\"",
                include_str!("../shaders/flash_moe.metal"),
            );
            let source = source.replace(
                "#include \"flash_mtp.metal\"",
                include_str!("../shaders/flash_mtp.metal"),
            );
            let source = source.replace(
                "#include \"flash_hc.metal\"",
                include_str!("../shaders/flash_hc.metal"),
            );
            let source = source
                .replace(
                    "#include \"flash_gdn.metal\"",
                    include_str!("../shaders/flash_gdn.metal"),
                )
                .replace(
                    "#include \"flash_gdn_prepare.metal\"",
                    include_str!("../shaders/flash_gdn_prepare.metal"),
                )
                .replace(
                    "#include \"gated_delta_packed.metal\"",
                    include_str!("../shaders/gated_delta_packed.metal"),
                )
                .replace(
                    "#include \"flash_gdn_epilogue.metal\"",
                    include_str!("../shaders/flash_gdn_epilogue.metal"),
                )
                .replace(
                    "#include \"flash_qsa.metal\"",
                    include_str!("../shaders/flash_qsa.metal"),
                )
                .replace(
                    "#include \"flash_rope.metal\"",
                    include_str!("../shaders/flash_rope.metal"),
                )
                .replace(
                    "#include \"flash_ple.metal\"",
                    include_str!("../shaders/flash_ple.metal"),
                )
                .replace(
                    "#include \"flash_hc_gemv.metal\"",
                    include_str!("../shaders/flash_hc_gemv.metal"),
                );
            let source = source.replace(
                "#include \"flash_logsumexp.metal\"",
                include_str!("../shaders/flash_logsumexp.metal"),
            );
            let library = device
                .new_library_with_source(&source, &options)
                .map_err(|e| anyhow::anyhow!("Metal compilation failed: {e}"))?;
            let pipeline = |name: &str| -> Result<_> {
                let function = library
                    .get_function(name, None)
                    .map_err(anyhow::Error::msg)?;
                device
                    .new_compute_pipeline_state_with_function(&function)
                    .map_err(anyhow::Error::msg)
            };
            let pack_pipeline = pipeline("exl3_pack")?;
            let decode_pipeline = pipeline("exl3_decode")?;
            let codebook_pipeline = pipeline("exl3_codebook")?;
            let qmv_pipeline = pipeline("exl3_qmv")?;
            let flash_qmv_pipeline = pipeline("flash_reference_qmv")?;
            let flash_qmv_nt2 = pipeline("flash_reference_qmv_nt2")?;
            let static_pipeline = |key: usize| {
                let spec = &crate::flash_qmv::SPECS[key];
                pipeline(&format!("flash_static_k{}_nt{}", spec.k, spec.nt))
            };
            let flash_static_qmv = [
                static_pipeline(0)?,
                static_pipeline(1)?,
                static_pipeline(2)?,
                static_pipeline(3)?,
                static_pipeline(4)?,
            ];
            let flash_static_supported = flash_static_qmv.each_ref().map(|p| {
                p.thread_execution_width() == 32 && p.max_total_threads_per_threadgroup() >= 128
            });
            let flash_fixed_qmv = if flash_fixed_setting == Some(true) {
                let fixed_pipeline = |shape: [u32; 2]| {
                    pipeline(&format!("flash_fixed_k2_{}_{}", shape[0], shape[1]))
                };
                let states = [
                    fixed_pipeline(crate::flash_qmv::FIXED_SHAPES[0])?,
                    fixed_pipeline(crate::flash_qmv::FIXED_SHAPES[1])?,
                ];
                ensure!(
                    states.iter().all(|p| p.thread_execution_width() == 32
                        && p.max_total_threads_per_threadgroup() >= 128),
                    "unsupported Flash fixed dimensions threadgroup"
                );
                Some(states)
            } else {
                None
            };
            let dense_qmv = [
                [
                    pipeline("dense_qmv_s4_n1")?,
                    pipeline("dense_qmv_s4_n2")?,
                    pipeline("dense_qmv_s4_n4")?,
                ],
                [
                    pipeline("dense_qmv_s8_n1")?,
                    pipeline("dense_qmv_s8_n2")?,
                    pipeline("dense_qmv_s8_n4")?,
                ],
            ];
            let dense_finish = pipeline("dense_finish128")?;
            let mapped_glu = pipeline("mapped_glu")?;
            let mapped_mix = pipeline("mapped_mix")?;
            let k7_pipeline = pipeline("flash_k7_qmv")?;
            let swiglu_pipeline = pipeline("flash_swiglu")?;
            let prepare_view_pipeline = pipeline("flash_prepare_view")?;
            let finish128_pipeline = pipeline("flash_finish128")?;
            let prepare128_pipeline = pipeline("exl3_prepare128")?;
            ensure!(
                qmv_pipeline.thread_execution_width() == 32
                    && qmv_pipeline.max_total_threads_per_threadgroup() >= 128,
                "EXL3 reference QMV requires 32-wide SIMD and 128 threads"
            );
            let perm: Vec<u16> = codec::permutation_inverse()
                .into_iter()
                .map(|i| i as u16)
                .collect();
            let inverse = upload(&device, &perm)?;
            let gate_dot = [pipeline("gate_dot_half")?, pipeline("gate_dot_float")?];
            let gate_softmax = [
                pipeline("gate_softmax_half")?,
                pipeline("gate_softmax_float")?,
            ];
            let moe_switch = [pipeline("moe_switch_half")?, pipeline("moe_switch_float")?];
            let moe_mix = [pipeline("moe_mix_half")?, pipeline("moe_mix_float")?];
            let route_half = pipeline("route_half")?;
            let route_float = pipeline("route_float")?;
            let hc = [
                hc::Pipelines::load(&pipeline, "half")?,
                hc::Pipelines::load(&pipeline, "float")?,
            ];
            let mtp_hidden_qmv = pipeline("flash_mtp_hidden_qmv")?;
            let mtp_norm = pipeline("mtp_hidden_norm")?;
            let mtp_combine = pipeline("mtp_combine_streams")?;
            let qsa = qsa::Pipelines::load(&pipeline)?;
            let gdn = gdn::Pipelines::load(&pipeline)?;
            let logsumexp = pipeline("flash_logsumexp")?;
            let ple = ple::Pipelines::load(&pipeline)?;
            let queue = device.new_command_queue();
            let model_ops = model_ops::Pipelines::load(&pipeline)?;
            Ok(Self {
                device,
                queue,
                hc_chain: std::cell::Cell::new(hc_chain),
                #[cfg(test)]
                helper_commands: std::cell::Cell::new(0),
                #[cfg(test)]
                hc_chain_pairs: std::cell::Cell::new(0),
                pack_pipeline,
                decode_pipeline,
                codebook_pipeline,
                qmv_pipeline,
                prepare128_pipeline,
                flash_qmv_pipeline,
                flash_qmv_nt2,
                flash_static_qmv,
                flash_static_supported,
                flash_static_setting,
                flash_fixed_qmv,
                flash_fixed_setting,
                #[cfg(test)]
                flash_fixed_submissions: std::array::from_fn(|_| std::cell::Cell::new(0)),
                #[cfg(test)]
                flash_static_submissions: std::array::from_fn(|_| std::cell::Cell::new(0)),
                dense_qmv,
                dense_finish,
                mapped_glu,
                mapped_mix,
                mtp_hidden_qmv,
                mtp_norm,
                mtp_combine,
                finish128_pipeline,
                prepare_view_pipeline,
                swiglu_pipeline,
                k7_pipeline,
                hc,
                qsa,
                model_ops,
                gdn,
                logsumexp,
                ple,
                moe_switch,
                moe_mix,
                gate_dot,
                gate_softmax,
                route_half,
                route_float,
                inverse,
                readers: std::cell::OnceCell::new(),
            })
        })
    }

    fn command<T>(&self, encode: impl FnOnce(&CommandBufferRef) -> Result<T>) -> Result<T> {
        objc::rc::autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            #[cfg(test)]
            self.helper_commands.set(self.helper_commands.get() + 1);
            let value = encode(cmd)?;
            cmd.commit();
            cmd.wait_until_completed();
            ensure!(
                cmd.status() == MTLCommandBufferStatus::Completed,
                "Metal command failed ({:?})",
                cmd.status()
            );
            Ok(value)
        })
    }

    pub fn device_name(&self) -> &str {
        self.device.name()
    }

    fn output(&self, count: usize) -> Result<Buffer> {
        let bytes = u64::try_from(count)?
            .checked_mul(2)
            .context("output size overflow")?;
        buffer(&self.device, bytes, Storage::StorageModeShared)
    }

    fn dispatch(
        &self,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        params: [u32; 5],
        groups: usize,
        threads: usize,
    ) -> Result<()> {
        objc::rc::autoreleasepool(|| {
            let command = self.queue.new_command_buffer();
            Self::encode(
                command,
                pipeline,
                buffers,
                params,
                MTLSize::new(groups as u64, 1, 1),
                threads,
            );
            command.commit();
            command.wait_until_completed();
            ensure!(
                command.status() == MTLCommandBufferStatus::Completed,
                "Metal command failed ({:?})",
                command.status()
            );
            Ok(())
        })
    }

    fn encode<T: Copy, const N: usize>(
        command: &CommandBufferRef,
        pipeline: &ComputePipelineState,
        buffers: &[&Buffer],
        params: [T; N],
        groups: MTLSize,
        threads: usize,
    ) {
        Self::encode_bound(
            command,
            pipeline,
            buffers.iter().map(|b| (*b, 0)),
            params,
            groups,
            threads,
        );
    }

    fn encode_bound<'a, T: Copy, const N: usize>(
        command: &CommandBufferRef,
        pipeline: &ComputePipelineState,
        buffers: impl ExactSizeIterator<Item = (&'a Buffer, u64)>,
        params: [T; N],
        groups: MTLSize,
        threads: usize,
    ) {
        let count = buffers.len();
        let encoder = command.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(pipeline);
        for (i, (buffer, offset)) in buffers.enumerate() {
            encoder.set_buffer(i as u64, Some(buffer), offset);
        }
        encoder.set_bytes(
            count as u64,
            std::mem::size_of_val(&params) as u64,
            params.as_ptr().cast(),
        );
        encoder.dispatch_thread_groups(groups, MTLSize::new(threads as u64, 1, 1));
        encoder.end_encoding();
    }

    fn read(buffer: &Buffer, count: usize) -> Vec<u16> {
        // Called only after a successful wait. Shared storage is CPU-coherent,
        // buffer length was allocated from this count, and u16 is POD/aligned.
        unsafe { std::slice::from_raw_parts(buffer.contents().cast::<u16>(), count).to_vec() }
    }

    pub fn pack(&self, encoded: &[u16], k: usize) -> Result<Vec<u16>> {
        codec::check_k(k)?;
        ensure!(
            encoded.len().is_multiple_of(256),
            "expected whole 256-state tiles"
        );
        if encoded.is_empty() {
            return Ok(Vec::new());
        }
        let count =
            contracts::packed_words(encoded.len(), k).context("Metal pack shape overflow")?;
        let n = u32::try_from(count).context("Metal pack index overflow")?;
        ensure!(
            encoded.len() <= u32::MAX as usize,
            "Metal input index overflow"
        );
        let input = upload(&self.device, encoded)?;
        let output = self.output(count)?;
        self.dispatch(
            &self.pack_pipeline,
            &[&input, &output],
            [n, k as u32, 0, 0, 0],
            count.div_ceil(256),
            256,
        )?;
        Ok(Self::read(&output, count))
    }

    pub fn decode(&self, packed: &[u16], k: usize, cb: Codebook) -> Result<(Vec<u16>, Vec<u16>)> {
        codec::check_k(k)?;
        ensure!(
            packed.len().is_multiple_of(16 * k),
            "expected whole packed tiles"
        );
        if packed.is_empty() {
            return Ok((Vec::new(), Vec::new()));
        }
        let count =
            contracts::decoded_states(packed.len(), k).context("Metal decode shape overflow")?;
        let n = u32::try_from(count).context("Metal decode index overflow")?;
        let input = upload(&self.device, packed)?;
        let states = self.output(count)?;
        let values = self.output(count)?;
        self.dispatch(
            &self.decode_pipeline,
            &[&input, &states, &values],
            [n, k as u32, cb as u32, 0, 0],
            count.div_ceil(256),
            256,
        )?;
        Ok((Self::read(&states, count), Self::read(&values, count)))
    }

    pub fn codebook(&self, cb: Codebook) -> Result<Vec<u16>> {
        let out = self.output(65536)?;
        self.dispatch(
            &self.codebook_pipeline,
            &[&out],
            [65536, 0, cb as u32, 0, 0],
            256,
            256,
        )?;
        Ok(Self::read(&out, 65536))
    }

    /// Multiply FP16 activations by per-column or per-element input scales,
    /// then apply normalized H128 to each consecutive block within a row.
    pub fn prepare128(&self, x: &[u16], width: usize, suh: &[u16]) -> Result<Vec<u16>> {
        let blocks = contracts::prepare128_blocks(x.len(), width, suh.len())
            .context("expected complete 128-aligned rows and column or element scales")?;
        if blocks == 0 {
            return Ok(Vec::new());
        }
        ensure!(
            self.prepare128_pipeline.max_total_threads_per_threadgroup() >= 64,
            "Metal H128 requires 64 threads per threadgroup"
        );
        let input = upload(&self.device, x)?;
        let scales = upload(&self.device, suh)?;
        let output = self.output(x.len())?;
        self.dispatch(
            &self.prepare128_pipeline,
            &[&input, &scales, &output],
            [suh.len() as u32, 0, 0, 0, 0],
            blocks as usize,
            64,
        )?;
        Ok(Self::read(&output, x.len()))
    }

    /// Half-bit inputs and outputs; operates in the rotated EXL3 basis.
    pub fn qmv_inner(
        &self,
        x: &[u16],
        packed: &[u16],
        cols: usize,
        k: usize,
        cb: Codebook,
    ) -> Result<Vec<u16>> {
        codec::check_k(k)?;
        let rows = x.len();
        ensure!(
            rows > 0 && cols > 0 && rows.is_multiple_of(16) && cols.is_multiple_of(16),
            "expected positive 16-aligned dimensions"
        );
        let size = contracts::qmv_words(rows, cols, k).context("QMV shape overflow")?;
        ensure!(
            packed.len() == size && size <= u32::MAX as usize,
            "invalid trellis size or Metal index overflow"
        );
        let rows = u32::try_from(rows)?;
        let n = u32::try_from(cols)?;
        let x = upload(&self.device, x)?;
        let packed = upload(&self.device, packed)?;
        let y = self.output(cols)?;
        self.dispatch(
            &self.qmv_pipeline,
            &[&x, &packed, &self.inverse, &y],
            [n, k as u32, cb as u32, rows, n],
            cols,
            128,
        )?;
        Ok(Self::read(&y, cols))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires physical Apple GPU; run cargo test -- --ignored"]
    fn metal_codec_matches_cpu() -> Result<()> {
        let gpu = Metal::new()?;
        for cb in [Codebook::Default, Codebook::Mcg, Codebook::Mul1] {
            let actual = gpu.codebook(cb)?;
            for word in 0..=u16::MAX {
                ensure!(
                    actual[word as usize] == codec::decode_codeword(word, cb).to_bits(),
                    "codebook mismatch {cb:?}/{word}"
                );
            }
            for k in 1..=8 {
                let input: Vec<_> = (0..1024).map(|i| (i * 701 + 33) as u16).collect();
                let packed = gpu.pack(&input, k)?;
                assert_eq!(packed, codec::pack(&input, k)?);
                let (states, values) = gpu.decode(&packed, k, cb)?;
                assert_eq!(states, codec::unpack(&packed, k)?);
                assert_eq!(
                    values,
                    states
                        .iter()
                        .map(|s| codec::decode_codeword(*s, cb).to_bits())
                        .collect::<Vec<_>>()
                );
                assert!(
                    gpu.qmv_inner(&[0; 32], &packed[..4 * 16 * k], 32, k, cb)?
                        .iter()
                        .all(|&bits| bits == 0)
                );
            }
        }
        Ok(())
    }
}
