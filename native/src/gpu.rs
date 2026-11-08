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
type CachedPipeline = crate::pipeline_caps::Pipeline<ComputePipelineState>;
pub struct FlashAttentionOutput {
    pub values: MatrixView,
    pub trace: Option<std::collections::BTreeMap<&'static str, Vec<u32>>>,
}

mod device_hot_cache;
mod greedy;
mod indirect_hot;
mod model_ops;
mod qsa;
pub use qsa::{FlashQsa, FlashQsaLayer, FlashQsaState};
mod flash;
pub use flash::{
    FlashMtpHead, FlashMtpSession, FlashNext, FlashSnapshot, FlashTraceTensor, FlashVerification,
};
mod ple;
pub use ple::{FlashPle, FlashPleLayer, FlashPleLayerState, FlashPleState};
mod qwen_gdn;
pub use qwen_gdn::{QwenGdn, QwenGdnState, QwenGdnTrace};
mod qwen;
pub use qwen::{Qwen35, QwenMtpHead, QwenMtpSession, QwenSnapshot};
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
mod fixed_residency;
mod projection;
pub mod quantize;
pub use projection::{
    FlashExpert, FlashProjection, MappedSwitchGlu, ProjectionGroup, ProjectionOutput,
};

#[cfg(test)]
thread_local! { static BUFFER_REQUESTS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) }; }
#[cfg(test)]
thread_local! {
    static BUFFER_REQUESTED_BYTES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static BUFFER_ALLOCATION_NS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[allow(
    unexpected_cfgs,
    reason = "objc 0.2 selector macro tests the legacy cargo-clippy cfg"
)]
pub(crate) fn buffer(device: &DeviceRef, length: u64, storage: Storage) -> Result<Buffer> {
    #[cfg(test)]
    let observed = crate::flash_profile::active().then(std::time::Instant::now);
    ensure!(
        length > 0 && length <= device.max_buffer_length(),
        "invalid Metal buffer length {length}"
    );
    // Metal may return nil on allocation failure. A successful new method returns +1.
    let ptr: *mut objc::runtime::Object =
        unsafe { msg_send![device, newBufferWithLength:length options:storage] };
    #[cfg(test)]
    BUFFER_REQUESTS.with(|count| count.set(count.get() + 1));
    #[cfg(test)]
    {
        BUFFER_REQUESTED_BYTES.with(|v| v.set(v.get().saturating_add(length)));
        if let Some(start) = observed {
            let elapsed = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
            BUFFER_ALLOCATION_NS.with(|v| v.set(v.get().saturating_add(elapsed)));
            crate::flash_profile::native_allocation(length, elapsed);
        }
    }
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
    #[cfg(test)]
    let _copy = crate::flash_profile::span("native.upload_copy");
    #[cfg(test)]
    if let Some(span) = &_copy {
        span.data(serde_json::json!({"bytes":bytes}));
    }
    // The private-to-this-function allocation is shared-storage and unpublished.
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), dest.contents().cast(), data.len()) };
    Ok(dest)
}

// Terminal errors still reach the caller's existing status checks and drain.
fn wait_command(command: &CommandBufferRef, enabled: bool) -> [u64; 4] {
    let terminal = || {
        matches!(
            command.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        )
    };
    if enabled {
        if terminal() {
            return [1, 0, 0, 0];
        }
        let start = std::time::Instant::now();
        let mut polls = 0;
        while crate::gpu_wait_policy::may_spin(enabled, start.elapsed().as_nanos(), polls) {
            std::hint::spin_loop();
            polls += 1;
            if terminal() {
                return [0, 1, 0, u64::from(polls)];
            }
        }
        command.wait_until_completed();
        return [0, 0, 1, u64::from(polls)];
    }
    command.wait_until_completed();
    [0, 0, 1, 0]
}

pub struct Metal {
    device: Device,
    cached_caps_enabled: bool,
    indirect_hot: Option<indirect_hot::Pipelines>,
    #[cfg(test)]
    ar20_kernel_bits: u32,
    #[cfg(test)]
    ar30_cooperative: bool,
    #[cfg(test)]
    ar30_cooperative_uses: [std::cell::Cell<u64>; 2],
    #[cfg(test)]
    ar33_codeword_table: bool,
    #[cfg(test)]
    ar33_codeword_table_uses: [std::cell::Cell<u64>; 2],
    #[cfg(test)]
    ar20_library: metal::Library,
    queue: CommandQueue,
    hc_chain: std::cell::Cell<bool>,
    hc_gate_chain: std::cell::Cell<bool>,
    hc_gdn_chain: std::cell::Cell<bool>,
    hc_up_mix: std::cell::Cell<bool>,
    gdn_half_elision: std::cell::Cell<bool>,
    gdn_norm_gate: std::cell::Cell<bool>,
    gdn_joint_ab: std::cell::Cell<bool>,
    gdn_whole: std::cell::Cell<bool>,
    gdn_head_gates: std::cell::Cell<bool>,
    gdn_resident_bundle: std::cell::Cell<bool>,
    gdn_resident_qmv: Option<ComputePipelineState>,
    #[cfg(test)]
    gdn_resident_bundle_uses: std::cell::Cell<usize>,
    #[cfg(test)]
    gdn_head_gate_uses: std::cell::Cell<usize>,
    #[cfg(test)]
    gdn_whole_uses: std::cell::Cell<usize>,
    qsa_prepare: std::cell::Cell<bool>,
    qsa_chains: std::cell::Cell<bool>,
    qsa_prefetch: std::cell::Cell<bool>,
    #[cfg(test)]
    qsa_chain_uses: std::cell::Cell<usize>,
    #[cfg(test)]
    qsa_prefetch_uses: std::cell::Cell<usize>,
    qsa_gate_fused: std::cell::Cell<bool>,
    #[cfg(test)]
    qsa_gate_fused_uses: std::cell::Cell<usize>,
    #[cfg(test)]
    qsa_prepare_uses: std::cell::Cell<usize>,
    #[cfg(test)]
    gdn_joint_ab_uses: std::cell::Cell<usize>,
    #[cfg(test)]
    gdn_norm_gate_uses: std::cell::Cell<usize>,
    #[cfg(test)]
    gdn_half_elisions: std::cell::Cell<usize>,
    hc_down_inject: std::cell::Cell<bool>,
    hc_write_norm: std::cell::Cell<bool>,
    #[cfg(test)]
    hc_write_norm_uses: std::cell::Cell<usize>,
    #[cfg(test)]
    hc_up_mix_uses: std::cell::Cell<usize>,
    #[cfg(test)]
    hc_down_inject_uses: std::cell::Cell<usize>,
    hc_workspace_enabled: std::cell::Cell<bool>,
    #[cfg(test)]
    hc_workspace_uses: std::cell::Cell<usize>,
    cross_layer: std::cell::Cell<bool>,
    gdn_route_chain: std::cell::Cell<bool>,
    hot_workspace: std::cell::Cell<bool>,
    qsa_index_key_only: std::cell::Cell<bool>,
    flash_index_key_qmv: Option<ComputePipelineState>,
    #[cfg(test)]
    hot_workspace_reuses: std::cell::Cell<usize>,
    #[cfg(test)]
    index_key_uses: std::cell::Cell<usize>,
    #[cfg(test)]
    gdn_route_uses: std::cell::Cell<usize>,
    moe_deferred: std::cell::Cell<bool>,
    #[cfg(test)]
    moe_deferred_uses: std::cell::Cell<usize>,
    #[cfg(test)]
    moe_deferred_fail: std::cell::Cell<bool>,
    #[cfg(test)]
    cross_layer_triples: std::cell::Cell<usize>,
    #[cfg(test)]
    cross_layer_fail_after: std::cell::Cell<Option<usize>>,
    #[cfg(test)]
    hc_final_writes: std::cell::Cell<usize>,
    #[cfg(test)]
    hc_gdn_pairs: std::cell::Cell<usize>,
    ssd_direct_enabled: bool,
    fixed_residency_enabled: std::cell::Cell<bool>,
    #[cfg(test)]
    ssd_direct_records: std::cell::Cell<u64>,
    #[cfg(test)]
    ssd_direct_read_bytes: std::cell::Cell<u64>,
    #[cfg(test)]
    ssd_legacy_upload_bytes: std::cell::Cell<u64>,
    #[cfg(test)]
    helper_commands: std::cell::Cell<usize>,
    #[cfg(test)]
    helper_commits: std::cell::Cell<usize>,
    #[cfg(test)]
    hc_chain_pairs: std::cell::Cell<usize>,
    #[cfg(test)]
    hc_gate_chain_pairs: std::cell::Cell<usize>,
    #[cfg(test)]
    gate_commands: std::cell::Cell<usize>,
    #[cfg(test)]
    helper_waits: std::cell::Cell<usize>,
    #[cfg(test)]
    route_waits: std::rc::Rc<std::cell::Cell<usize>>,
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
    routed_nt2_pipelines: Option<[ComputePipelineState; 2]>,
    routed_nt2_gu: std::cell::Cell<bool>,
    routed_nt2_down: std::cell::Cell<bool>,
    #[cfg(test)]
    routed_nt2_uses: [std::cell::Cell<u64>; 2],
    flash_fixed_setting: Option<bool>,
    #[cfg(test)]
    flash_fixed_submissions: [std::cell::Cell<u64>; 2],
    flash_expert_scratch_setting: Option<bool>,
    expert_ar_enabled: [std::cell::Cell<bool>; 4],
    spin_wait_enabled: bool,
    #[cfg(test)]
    spin_wait_stats: std::rc::Rc<[std::cell::Cell<u64>; 4]>,
    #[cfg(test)]
    expert_ar_uses: [std::cell::Cell<u64>; 4],
    expert_up_finish: Option<ComputePipelineState>,
    expert_down_prepare: Option<ComputePipelineState>,
    common_executor: bool,
    flash_defaults: bool,
    segment_enabled: bool,
    segment_pending: std::cell::Cell<usize>,
    segment_failed: std::cell::Cell<bool>,
    #[cfg(test)]
    segment_submissions: std::cell::Cell<u64>,
    #[cfg(test)]
    expert_down_prepare_uses: std::cell::Cell<u64>,
    expert_gu_prepare: Option<ComputePipelineState>,
    expert_axis_qmv: Option<[[ComputePipelineState; 2]; 2]>,
    expert_consumer_qmv: Option<[[ComputePipelineState; 2]; 2]>,
    #[cfg(test)]
    expert_consumer_dispatches: std::cell::Cell<u64>,
    #[cfg(test)]
    expert_axis_dispatches: std::cell::Cell<u64>,
    #[cfg(test)]
    flash_scratch_bindings: std::cell::Cell<[[usize; 2]; 3]>,
    #[cfg(test)]
    flash_scratch_uses: std::cell::Cell<usize>,
    #[cfg(test)]
    flash_expert_outputs: std::cell::Cell<[usize; 4]>,
    #[cfg(test)]
    flash_static_submissions: [std::cell::Cell<u64>; 5],
    dense_qmv: [[ComputePipelineState; 3]; 2],
    mapped_qmv: Option<[[ComputePipelineState; 2]; 2]>,
    qwen_expert_stages: Option<[ComputePipelineState; 3]>,
    #[cfg(test)]
    qwen_expert_stage_dispatches: std::cell::Cell<usize>,
    #[cfg(test)]
    mapped_qmv_groups: std::cell::Cell<usize>,
    #[cfg(test)]
    mapped_qmv_singles: std::cell::Cell<usize>,
    dense_finish: ComputePipelineState,
    mapped_glu: ComputePipelineState,
    mapped_mix: ComputePipelineState,
    mtp_hidden_qmv: ComputePipelineState,
    flash_verify_qmv: Option<[ComputePipelineState; 2]>,
    flash_verify_specialized: Option<[ComputePipelineState; 7]>,
    #[cfg(test)]
    flash_verify_specialized_uses: std::cell::Cell<[u64; 7]>,
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
    logsumexp: CachedPipeline,
    device_greedy_pipeline: Option<ComputePipelineState>,
    device_greedy_calls: std::cell::Cell<u64>,
    device_greedy_heads: std::cell::Cell<u64>,
    #[cfg(test)]
    device_greedy_reject_result: std::cell::Cell<bool>,
    ple: ple::Pipelines,
    moe_switch: [CachedPipeline; 2],
    moe_mix: [ComputePipelineState; 2],
    moe_finish_mix: std::cell::Cell<bool>,
    moe_finish_mix_pipeline: Option<ComputePipelineState>,
    #[cfg(test)]
    moe_finish_mix_uses: std::cell::Cell<usize>,
    gate_dot: [ComputePipelineState; 2],
    router_pair: gate::PairedDot,
    resident_scratch: crate::resident_scratch_policy::Budget,
    gate_softmax: [ComputePipelineState; 2],
    route_half: ComputePipelineState,
    route_float: ComputePipelineState,
    inverse: Buffer,
    readers: std::cell::OnceCell<crate::ssd_reads::ReadService>,
}

impl Metal {
    fn completed_record(&self, payload: crate::ssd_reads::ReadPayload) -> Result<Buffer> {
        match payload {
            crate::ssd_reads::ReadPayload::Bytes(bytes) => {
                let backing = upload(&self.device, &bytes)?;
                #[cfg(test)]
                self.ssd_legacy_upload_bytes
                    .set(self.ssd_legacy_upload_bytes.get() + bytes.len() as u64);
                Ok(backing)
            }
            crate::ssd_reads::ReadPayload::Shared(backing) => {
                #[cfg(test)]
                {
                    self.ssd_direct_records
                        .set(self.ssd_direct_records.get() + 1);
                    self.ssd_direct_read_bytes
                        .set(self.ssd_direct_read_bytes.get() + backing.length());
                }
                Ok(backing)
            }
        }
    }
    fn readers(&self) -> Result<&crate::ssd_reads::ReadService> {
        if self.readers.get().is_none() {
            let count = std::env::var("MLXL3_SSD_READERS")
                .unwrap_or_else(|_| "4".into())
                .parse::<usize>()
                .context("invalid SSD reader count")?;
            self.readers
                .set(crate::ssd_reads::ReadService::new_qualified(
                    count,
                    self.device.name() == "Apple M5",
                )?)
                .map_err(|_| anyhow::anyhow!("native SSD readers already initialized"))?;
        }
        self.readers.get().context("missing native SSD readers")
    }

    pub fn new() -> Result<Self> {
        objc::rc::autoreleasepool(|| {
            let device = Device::system_default().context("no Metal device")?;
            let qualified_m5 = device.name() == "Apple M5";
            let flash_defaults = crate::native_executor::defaults(qualified_m5);
            let fixed_residency_enabled = match std::env::var("MLXL3_NATIVE_FIXED_RESIDENCY") {
                Ok(value) => crate::flash_support::hc_kernel_flag(Some(&value), qualified_m5),
                Err(std::env::VarError::NotPresent) => {
                    crate::flash_support::hc_kernel_flag(None, qualified_m5)
                }
                Err(error) => return Err(error.into()),
            }
            .context("MLXL3_NATIVE_FIXED_RESIDENCY must be 0 or 1")?;
            let hc_kernel_setting = |name: &str| -> Result<bool> {
                let value = match std::env::var(name) {
                    Ok(value) => Some(value),
                    Err(std::env::VarError::NotPresent) => None,
                    Err(error) => return Err(error.into()),
                };
                let default = flash_defaults
                    && match name {
                        "MLXL3_NATIVE_FLASH_COMMON_EXECUTOR"
                        | "MLXL3_NATIVE_FLASH_VERIFY_QMV"
                        | "MLXL3_NATIVE_FLASH_VERIFY_SPECIALIZED"
                        | "MLXL3_NATIVE_FLASH_GLU_DOWN"
                        | "MLXL3_NATIVE_FLASH_EXPERT_CONSUMERS"
                        | "MLXL3_NATIVE_FLASH_GDN_TEMPORAL"
                        | "MLXL3_NATIVE_FLASH_GPU_SEGMENTS"
                        | "MLXL3_NATIVE_DEVICE_GREEDY" => true,
                        "MLXL3_NATIVE_HOT_WORKSPACE" => {
                            std::env::var_os("MLXL3_NATIVE_MOE_FINISH_MIX")
                                .is_none_or(|value| value != "1")
                        }
                        _ => false,
                    };
                crate::native_executor::default_flag(value.as_deref(), qualified_m5, default)
                    .with_context(|| format!("{name} must be 0 or 1"))
            };
            let spin_wait_enabled = hc_kernel_setting("MLXL3_NATIVE_GPU_SPIN_WAIT")?;
            let hot_workspace = hc_kernel_setting("MLXL3_NATIVE_HOT_WORKSPACE")?;
            let qsa_index_key_only = hc_kernel_setting("MLXL3_NATIVE_QSA_INDEX_KEY_ONLY")?;
            let ar30_cooperative = hc_kernel_setting("MLXL3_NATIVE_QMV_COOPERATIVE")?;
            let ar33_codeword_table = hc_kernel_setting("MLXL3_NATIVE_CODEWORD_TABLE")?;
            let cached_caps_enabled = hc_kernel_setting("MLXL3_NATIVE_CACHED_CAPS")?;
            let device_resident_route_value =
                match std::env::var("MLXL3_NATIVE_DEVICE_RESIDENT_ROUTE") {
                    Ok(value) => Some(value),
                    Err(std::env::VarError::NotPresent) => None,
                    Err(error) => return Err(error.into()),
                };
            let device_resident_route = crate::device_hot_policy::flag(
                device_resident_route_value.as_deref(),
                qualified_m5,
                qualified_m5
                    && device.argument_buffers_support() == metal::MTLArgumentBuffersTier::Tier2,
            )
            .context("MLXL3_NATIVE_DEVICE_RESIDENT_ROUTE must be 0 or 1")?;
            let ar20_kernel_bits = crate::ar20_kernel_policy::mask([
                hc_kernel_setting("MLXL3_NATIVE_MUL1_SWAR")?,
                hc_kernel_setting("MLXL3_NATIVE_TRELLIS_READER8")?,
                hc_kernel_setting("MLXL3_NATIVE_QMV_K_PRELOAD")?,
                hc_kernel_setting("MLXL3_NATIVE_H128_SIMD")?,
            ]);
            let moe_finish_mix = hc_kernel_setting("MLXL3_NATIVE_MOE_FINISH_MIX")?;
            let ar10_setting = |name| crate::ar10_defaults::setting(name, qualified_m5);
            let hc_up_mix = hc_kernel_setting("MLXL3_NATIVE_HC_UP_MIX")?;
            let hc_write_norm = ar10_setting("MLXL3_NATIVE_HC_WRITE_NORM")?;
            let routed_nt2_gu = ar10_setting("MLXL3_NATIVE_ROUTED_NT2_GU")?;
            let routed_nt2_down = ar10_setting("MLXL3_NATIVE_ROUTED_NT2_DOWN")?;
            let gdn_half_elision = hc_kernel_setting("MLXL3_NATIVE_GDN_HALF_ELISION")?;
            let gdn_norm_gate = hc_kernel_setting("MLXL3_NATIVE_GDN_NORM_GATE")?;
            let gdn_joint_ab = ar10_setting("MLXL3_NATIVE_GDN_JOINT_AB")?;
            let gdn_whole = ar10_setting("MLXL3_NATIVE_GDN_WHOLE")?;
            let gdn_head_gates = hc_kernel_setting("MLXL3_NATIVE_GDN_HEAD_GATES")?;
            let gdn_resident_bundle = hc_kernel_setting("MLXL3_NATIVE_GDN_RESIDENT_BUNDLE")?;
            let qsa_prepare = ar10_setting("MLXL3_NATIVE_QSA_PREPARE")?;
            let qsa_chains = hc_kernel_setting("MLXL3_NATIVE_QSA_CHAINS")?;
            let qsa_prefetch = hc_kernel_setting("MLXL3_NATIVE_QSA_PREFETCH")?;
            let qsa_gate_fused = ar10_setting("MLXL3_NATIVE_QSA_GATE")?;
            let hc_down_inject = hc_kernel_setting("MLXL3_NATIVE_HC_DOWN_INJECT")?;
            // Only qualified L1 consumers defer completion, so use L1's hardware default.
            let moe_deferred = match std::env::var("MLXL3_NATIVE_MOE_DEFERRED") {
                Ok(value) => crate::flash_support::hc_gdn_chain_flag(Some(&value), qualified_m5),
                Err(std::env::VarError::NotPresent) => {
                    crate::flash_support::hc_gdn_chain_flag(None, qualified_m5)
                }
                Err(error) => return Err(error.into()),
            }
            .context("MLXL3_NATIVE_MOE_DEFERRED must be 0 or 1")?;
            let hc_workspace_enabled = match std::env::var("MLXL3_NATIVE_HC_WORKSPACE") {
                Ok(value) => crate::hc_workspace_policy::flag(Some(&value), qualified_m5),
                Err(std::env::VarError::NotPresent) => crate::hc_workspace_policy::flag(
                    Some(if flash_defaults { "1" } else { "0" }),
                    qualified_m5,
                ),
                Err(error) => return Err(error.into()),
            }
            .context("MLXL3_NATIVE_HC_WORKSPACE must be 0 or 1")?;
            let cross_layer = match std::env::var("MLXL3_NATIVE_CROSS_LAYER") {
                Ok(value) => crate::flash_support::hc_gdn_chain_flag(Some(&value), qualified_m5),
                Err(std::env::VarError::NotPresent) => {
                    crate::flash_support::hc_gdn_chain_flag(None, qualified_m5)
                }
                Err(error) => return Err(error.into()),
            }
            .context("MLXL3_NATIVE_CROSS_LAYER must be 0 or 1")?;
            let hc_gdn_chain = match std::env::var("MLXL3_NATIVE_HC_GDN_CHAIN") {
                Ok(value) => crate::flash_support::hc_gdn_chain_flag(Some(&value), qualified_m5),
                Err(std::env::VarError::NotPresent) => {
                    crate::flash_support::hc_gdn_chain_flag(None, qualified_m5)
                }
                Err(error) => return Err(error.into()),
            }
            .context("MLXL3_NATIVE_HC_GDN_CHAIN must be 0 or 1")?;
            let ssd_direct_enabled = match std::env::var("MLXL3_NATIVE_SSD_DIRECT") {
                Ok(value) => crate::ssd_reads::direct_flag(Some(&value), qualified_m5),
                Err(std::env::VarError::NotPresent) => {
                    crate::ssd_reads::direct_flag(None, qualified_m5)
                }
                Err(error) => return Err(error.into()),
            }
            .context("MLXL3_NATIVE_SSD_DIRECT must be 0 or 1")?;
            let hc_chain = match std::env::var("MLXL3_NATIVE_AR_HC_CHAIN") {
                Ok(value) => crate::flash_support::hc_chain_flag(Some(&value), qualified_m5),
                Err(std::env::VarError::NotPresent) => {
                    crate::flash_support::hc_chain_flag(None, qualified_m5)
                }
                Err(error) => return Err(error.into()),
            }
            .context("MLXL3_NATIVE_AR_HC_CHAIN must be 0 or 1")?;
            let hc_gate_chain = match std::env::var("MLXL3_NATIVE_AR_HC_GATE_CHAIN") {
                Ok(value) => crate::flash_support::hc_chain_flag(Some(&value), qualified_m5),
                Err(std::env::VarError::NotPresent) => {
                    crate::flash_support::hc_chain_flag(None, qualified_m5)
                }
                Err(error) => return Err(error.into()),
            }
            .context("MLXL3_NATIVE_AR_HC_GATE_CHAIN must be 0 or 1")?;
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
                    Ok(value) => crate::flash_support::hc_chain_flag(Some(&value), qualified_m5),
                    Err(std::env::VarError::NotPresent) => {
                        crate::flash_support::hc_chain_flag(None, qualified_m5)
                    }
                    Err(std::env::VarError::NotUnicode(_)) => None,
                }
            } else {
                Some(false)
            };
            let flash_expert_scratch_setting = match std::env::var("MLXL3_NATIVE_ROUTED_SCRATCH") {
                Ok(value) => crate::flash_support::hc_chain_flag(Some(&value), qualified_m5),
                Err(std::env::VarError::NotPresent) => {
                    crate::flash_support::hc_chain_flag(None, qualified_m5)
                }
                Err(std::env::VarError::NotUnicode(_)) => None,
            };
            let expert_ar_names = [
                "MLXL3_NATIVE_EXPERT_UP_FINISH",
                "MLXL3_NATIVE_EXPERT_GU_PREPARE",
                "MLXL3_NATIVE_EXPERT_OUTPUT_SLAB",
                "MLXL3_NATIVE_EXPERT_AXIS_QMV",
            ];
            let mut expert_ar_enabled = [false; 4];
            for (index, name) in expert_ar_names.into_iter().enumerate() {
                let value = std::env::var(name);
                expert_ar_enabled[index] = match value {
                    Ok(value) => crate::expert_ar_policy::flag(Some(&value), qualified_m5),
                    Err(std::env::VarError::NotPresent) => {
                        crate::expert_ar_policy::flag(None, qualified_m5)
                    }
                    Err(error) => return Err(error.into()),
                }
                .with_context(|| format!("{name} must be 0 or 1"))?;
            }
            let options = CompileOptions::new();
            options.set_fast_math_enabled(false);
            let source = include_str!("../shaders/exl3.metal")
                .replace(
                    "#include \"indirect_hot.metal\"",
                    include_str!("../shaders/indirect_hot.metal"),
                )
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
                    "#include \"flash_expert_axis.metal\"",
                    include_str!("../shaders/flash_expert_axis.metal"),
                )
                .replace(
                    "#include \"flash_gdn_resident.metal\"",
                    include_str!("../shaders/flash_gdn_resident.metal"),
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
                    "#include \"flash_gdn_whole.metal\"",
                    include_str!("../shaders/flash_gdn_whole.metal"),
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
                    "#include \"flash_qsa_ar20.metal\"",
                    include_str!("../shaders/flash_qsa_ar20.metal"),
                )
                .replace(
                    "#include \"flash_qsa_chain.metal\"",
                    include_str!("../shaders/flash_qsa_chain.metal"),
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
            let source = format!(
                "{}\n#define MLXL3_AR30_COOPERATIVE {}\n#define MLXL3_AR33_CODEWORD_TABLE {}\n#define MLXL3_DEVICE_RESIDENT_ROUTE {}\n#define MLXL3_AR20_TESTS {}\n{}",
                crate::ar20_kernel_policy::defines(ar20_kernel_bits),
                u32::from(ar30_cooperative),
                u32::from(ar33_codeword_table),
                u32::from(device_resident_route),
                u32::from(cfg!(test)),
                source
            )
            .replace(
                r#"#include "ar20_kernel_tests.metal""#,
                include_str!("../shaders/ar20_kernel_tests.metal"),
            );
            let source = source.replace(
                r#"#include "ar20_trellis.metal""#,
                include_str!("../shaders/ar20_trellis.metal"),
            );
            let source = source.replace(
                r#"#include "ar30_kernel_tests.metal""#,
                include_str!("../shaders/ar30_kernel_tests.metal"),
            );
            let source = source
                .replace(
                    r#"#include "ar33_codeword_table.metal""#,
                    include_str!("../shaders/ar33_codeword_table.metal"),
                )
                .replace(
                    r#"#include "ar33_kernel_tests.metal""#,
                    include_str!("../shaders/ar33_kernel_tests.metal"),
                );
            let hot_slots_value = match std::env::var("MLXL3_NATIVE_DEVICE_HOT_SLOTS") {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(error) => return Err(error.into()),
            };
            let hot_slots = crate::device_hot_policy::capacity(hot_slots_value.as_deref())
                .context("MLXL3_NATIVE_DEVICE_HOT_SLOTS must be 53 or 64")?;
            let source = format!("#define MLXL3_DEVICE_HOT_SLOTS {hot_slots}u\n{source}");
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
            let cached_pipeline = |name: &str| -> Result<_> {
                pipeline(name).map(|raw| CachedPipeline::new(raw, cached_caps_enabled))
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
            let routed_nt2_pipelines = if routed_nt2_gu || routed_nt2_down {
                let states = [
                    pipeline("flash_fixed_k2_nt2_2560_640")?,
                    pipeline("flash_fixed_k2_nt2_640_2560")?,
                ];
                ensure!(
                    states.iter().all(|p| p.thread_execution_width() == 32
                        && p.max_total_threads_per_threadgroup() >= 128),
                    "unsupported routed NT2 geometry"
                );
                Some(states)
            } else {
                None
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
            let mapped_qmv = if qualified_m5
                && std::env::var("MLXL3_NATIVE_QWEN_GROUP_EXPERT_QMV").as_deref() == Ok("1")
                && device.argument_buffers_support() == metal::MTLArgumentBuffersTier::Tier2
            {
                Some([
                    [pipeline("mapped_qmv_s4_n2")?, pipeline("mapped_qmv_s4_n4")?],
                    [pipeline("mapped_qmv_s8_n2")?, pipeline("mapped_qmv_s8_n4")?],
                ])
            } else {
                None
            };
            let qwen_expert_stages = if mapped_qmv.is_some()
                && std::env::var("MLXL3_NATIVE_QWEN_GROUP_EXPERT_STAGES").as_deref() == Ok("1")
            {
                Some([
                    pipeline("qwen_stage_prepare")?,
                    pipeline("qwen_stage_glu")?,
                    pipeline("qwen_stage_finish")?,
                ])
            } else {
                None
            };
            let dense_finish = pipeline("dense_finish128")?;
            let mapped_glu = pipeline("mapped_glu")?;
            let mapped_mix = pipeline("mapped_mix")?;
            let k7_pipeline = pipeline("flash_k7_qmv")?;
            let swiglu_pipeline = pipeline("flash_swiglu")?;
            let prepare_view_pipeline = pipeline("flash_prepare_view")?;
            let finish128_pipeline = pipeline("flash_finish128")?;
            let expert_up_finish = expert_ar_enabled[0]
                .then(|| pipeline("flash_up_finish_swiglu"))
                .transpose()?;
            let expert_down_prepare = (hc_kernel_setting("MLXL3_NATIVE_FLASH_GLU_DOWN")?
                && qualified_m5
                && expert_ar_enabled[0])
                .then(|| pipeline("flash_up_finish_swiglu_prepare"))
                .transpose()?;
            let expert_gu_prepare = expert_ar_enabled[1]
                .then(|| pipeline("flash_dual_gu_prepare"))
                .transpose()?;
            let expert_axis_qmv = expert_ar_enabled[3]
                .then(|| -> Result<_> {
                    Ok([
                        [
                            pipeline("flash_axis_k2_2560_640_nt1")?,
                            pipeline("flash_axis_k2_2560_640_nt2")?,
                        ],
                        [
                            pipeline("flash_axis_k2_640_2560_nt1")?,
                            pipeline("flash_axis_k2_640_2560_nt2")?,
                        ],
                    ])
                })
                .transpose()?;
            let gdn_resident_qmv = gdn_resident_bundle
                .then(|| pipeline("ar20_gdn_resident_qkv_z_nt2"))
                .transpose()?;
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
            let router_pair =
                gate::PairedDot::load(&pipeline, hc_kernel_setting("MLXL3_NATIVE_ROUTER_PAIR")?)?;
            let gate_softmax = [
                pipeline("gate_softmax_half")?,
                pipeline("gate_softmax_float")?,
            ];
            let moe_switch = [
                cached_pipeline("moe_switch_half")?,
                cached_pipeline("moe_switch_float")?,
            ];
            let moe_mix = [pipeline("moe_mix_half")?, pipeline("moe_mix_float")?];
            let moe_finish_mix_pipeline = moe_finish_mix
                .then(|| pipeline("moe_finish_mix_float"))
                .transpose()?;
            let route_half = pipeline("route_half")?;
            let route_float = pipeline("route_float")?;
            let hc = [
                hc::Pipelines::load(&cached_pipeline, "half", false, false, false)?,
                hc::Pipelines::load(
                    &cached_pipeline,
                    "float",
                    hc_up_mix,
                    hc_down_inject,
                    hc_write_norm,
                )?,
            ];
            let mtp_hidden_qmv = pipeline("flash_mtp_hidden_qmv")?;
            let flash_verify_qmv = if hc_kernel_setting("MLXL3_NATIVE_FLASH_VERIFY_QMV")? {
                Some([
                    pipeline("flash_verify_qmv_mb2")?,
                    pipeline("flash_verify_qmv_mb2_nt2")?,
                ])
            } else {
                None
            };
            let specialized = hc_kernel_setting("MLXL3_NATIVE_FLASH_VERIFY_SPECIALIZED")?;
            let expert_consumer_qmv = (hc_kernel_setting("MLXL3_NATIVE_FLASH_EXPERT_CONSUMERS")?
                && qualified_m5
                && expert_axis_qmv.is_some()
                && flash_verify_qmv.is_some())
            .then(|| -> Result<_> {
                Ok([
                    [
                        pipeline("flash_consumers_k2_2560_640_nt1")?,
                        pipeline("flash_consumers_k2_2560_640_nt2")?,
                    ],
                    [
                        pipeline("flash_consumers_k2_640_2560_nt1")?,
                        pipeline("flash_consumers_k2_640_2560_nt2")?,
                    ],
                ])
            })
            .transpose()?;
            let flash_verify_specialized = if specialized && flash_verify_qmv.is_some() {
                let states = [
                    pipeline("flash_verify_static_k2_nt1")?,
                    pipeline("flash_verify_static_k3_nt1")?,
                    pipeline("flash_verify_static_k3_nt2")?,
                    pipeline("flash_verify_static_k4_nt1")?,
                    pipeline("flash_verify_static_k4_nt2")?,
                    pipeline("flash_verify_fixed_k2_2560_640")?,
                    pipeline("flash_verify_fixed_k2_640_2560")?,
                ];
                ensure!(
                    states.iter().all(|p| p.thread_execution_width() == 32
                        && p.max_total_threads_per_threadgroup() >= 128),
                    "unsupported specialized MB2 geometry"
                );
                Some(states)
            } else {
                None
            };
            let mtp_norm = pipeline("mtp_hidden_norm")?;
            let mtp_combine = pipeline("mtp_combine_streams")?;
            let qsa = qsa::Pipelines::load(
                &cached_pipeline,
                qsa_prepare,
                qsa_gate_fused,
                qsa_chains,
                qsa_prefetch,
                qsa_index_key_only,
            )?;
            let gdn = gdn::Pipelines::load(
                &cached_pipeline,
                gdn_norm_gate,
                gdn_joint_ab,
                gdn_whole,
                gdn_head_gates,
                hc_kernel_setting("MLXL3_NATIVE_FLASH_GDN_TEMPORAL")? && qualified_m5,
            )?;
            let logsumexp = cached_pipeline("flash_logsumexp")?;
            let device_greedy_pipeline = hc_kernel_setting("MLXL3_NATIVE_DEVICE_GREEDY")?
                .then(|| pipeline("flash_normalized_greedy"))
                .transpose()?;
            let ple = ple::Pipelines::load(&cached_pipeline)?;
            let indirect_hot = indirect_hot::Pipelines::load(
                &device,
                &library,
                &pipeline,
                device_resident_route,
                hot_slots,
            )?;
            let queue = device.new_command_queue();
            let model_ops = model_ops::Pipelines::load(&cached_pipeline)?;
            let flash_index_key_qmv = qsa_index_key_only
                .then(|| pipeline("flash_index_key_qmv"))
                .transpose()?;
            Ok(Self {
                device,
                cached_caps_enabled,
                indirect_hot,
                #[cfg(test)]
                ar20_kernel_bits,
                #[cfg(test)]
                ar30_cooperative,
                #[cfg(test)]
                ar30_cooperative_uses: std::array::from_fn(|_| std::cell::Cell::new(0)),
                #[cfg(test)]
                ar33_codeword_table,
                #[cfg(test)]
                ar33_codeword_table_uses: std::array::from_fn(|_| std::cell::Cell::new(0)),
                #[cfg(test)]
                ar20_library: library,
                queue,
                hc_chain: std::cell::Cell::new(hc_chain),
                hc_gate_chain: std::cell::Cell::new(hc_gate_chain),
                hc_gdn_chain: std::cell::Cell::new(hc_gdn_chain),
                hc_up_mix: std::cell::Cell::new(hc_up_mix),
                gdn_half_elision: std::cell::Cell::new(gdn_half_elision),
                gdn_norm_gate: std::cell::Cell::new(gdn_norm_gate),
                gdn_joint_ab: std::cell::Cell::new(gdn_joint_ab),
                gdn_whole: std::cell::Cell::new(gdn_whole),
                gdn_head_gates: std::cell::Cell::new(gdn_head_gates),
                gdn_resident_bundle: std::cell::Cell::new(gdn_resident_bundle),
                gdn_resident_qmv,
                #[cfg(test)]
                gdn_resident_bundle_uses: std::cell::Cell::new(0),
                #[cfg(test)]
                gdn_head_gate_uses: std::cell::Cell::new(0),
                #[cfg(test)]
                gdn_whole_uses: std::cell::Cell::new(0),
                qsa_prepare: std::cell::Cell::new(qsa_prepare),
                qsa_chains: std::cell::Cell::new(qsa_chains),
                qsa_prefetch: std::cell::Cell::new(qsa_prefetch),
                #[cfg(test)]
                qsa_chain_uses: std::cell::Cell::new(0),
                #[cfg(test)]
                qsa_prefetch_uses: std::cell::Cell::new(0),
                qsa_gate_fused: std::cell::Cell::new(qsa_gate_fused),
                #[cfg(test)]
                qsa_gate_fused_uses: std::cell::Cell::new(0),
                #[cfg(test)]
                qsa_prepare_uses: std::cell::Cell::new(0),
                #[cfg(test)]
                gdn_joint_ab_uses: std::cell::Cell::new(0),
                #[cfg(test)]
                gdn_norm_gate_uses: std::cell::Cell::new(0),
                #[cfg(test)]
                gdn_half_elisions: std::cell::Cell::new(0),
                hc_down_inject: std::cell::Cell::new(hc_down_inject),
                hc_write_norm: std::cell::Cell::new(hc_write_norm),
                #[cfg(test)]
                hc_write_norm_uses: std::cell::Cell::new(0),
                #[cfg(test)]
                hc_up_mix_uses: std::cell::Cell::new(0),
                #[cfg(test)]
                hc_down_inject_uses: std::cell::Cell::new(0),
                hc_workspace_enabled: std::cell::Cell::new(hc_workspace_enabled),
                #[cfg(test)]
                hc_workspace_uses: std::cell::Cell::new(0),
                cross_layer: std::cell::Cell::new(cross_layer),
                hot_workspace: std::cell::Cell::new(hot_workspace),
                qsa_index_key_only: std::cell::Cell::new(qsa_index_key_only),
                flash_index_key_qmv,
                #[cfg(test)]
                hot_workspace_reuses: std::cell::Cell::new(0),
                #[cfg(test)]
                index_key_uses: std::cell::Cell::new(0),
                gdn_route_chain: std::cell::Cell::new(
                    match std::env::var("MLXL3_NATIVE_GDN_ROUTE_CHAIN") {
                        Ok(value) => crate::flash_support::hc_chain_flag(Some(&value), false),
                        Err(std::env::VarError::NotPresent) => Some(qualified_m5),
                        Err(error) => return Err(error.into()),
                    }
                    .context("MLXL3_NATIVE_GDN_ROUTE_CHAIN must be 0 or 1")?,
                ),
                #[cfg(test)]
                gdn_route_uses: std::cell::Cell::new(0),
                moe_deferred: std::cell::Cell::new(moe_deferred),
                #[cfg(test)]
                moe_deferred_uses: std::cell::Cell::new(0),
                #[cfg(test)]
                moe_deferred_fail: std::cell::Cell::new(false),
                #[cfg(test)]
                cross_layer_triples: std::cell::Cell::new(0),
                #[cfg(test)]
                cross_layer_fail_after: std::cell::Cell::new(None),
                #[cfg(test)]
                hc_final_writes: std::cell::Cell::new(0),
                #[cfg(test)]
                hc_gdn_pairs: std::cell::Cell::new(0),
                ssd_direct_enabled,
                fixed_residency_enabled: std::cell::Cell::new(fixed_residency_enabled),
                #[cfg(test)]
                ssd_direct_records: std::cell::Cell::new(0),
                #[cfg(test)]
                ssd_direct_read_bytes: std::cell::Cell::new(0),
                #[cfg(test)]
                ssd_legacy_upload_bytes: std::cell::Cell::new(0),
                #[cfg(test)]
                helper_commands: std::cell::Cell::new(0),
                #[cfg(test)]
                helper_commits: std::cell::Cell::new(0),
                #[cfg(test)]
                hc_chain_pairs: std::cell::Cell::new(0),
                #[cfg(test)]
                hc_gate_chain_pairs: std::cell::Cell::new(0),
                #[cfg(test)]
                gate_commands: std::cell::Cell::new(0),
                #[cfg(test)]
                helper_waits: std::cell::Cell::new(0),
                #[cfg(test)]
                route_waits: std::rc::Rc::new(std::cell::Cell::new(0)),
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
                routed_nt2_pipelines,
                routed_nt2_gu: std::cell::Cell::new(routed_nt2_gu),
                routed_nt2_down: std::cell::Cell::new(routed_nt2_down),
                #[cfg(test)]
                routed_nt2_uses: std::array::from_fn(|_| std::cell::Cell::new(0)),
                flash_fixed_setting,
                #[cfg(test)]
                flash_fixed_submissions: std::array::from_fn(|_| std::cell::Cell::new(0)),
                flash_expert_scratch_setting,
                expert_ar_enabled: expert_ar_enabled.map(std::cell::Cell::new),
                spin_wait_enabled,
                #[cfg(test)]
                spin_wait_stats: std::rc::Rc::new(std::array::from_fn(|_| std::cell::Cell::new(0))),
                #[cfg(test)]
                expert_ar_uses: std::array::from_fn(|_| std::cell::Cell::new(0)),
                expert_up_finish,
                expert_down_prepare,
                flash_defaults,
                common_executor: hc_kernel_setting("MLXL3_NATIVE_FLASH_COMMON_EXECUTOR")?,
                segment_enabled: hc_kernel_setting("MLXL3_NATIVE_FLASH_GPU_SEGMENTS")?,
                segment_pending: std::cell::Cell::new(0),
                segment_failed: std::cell::Cell::new(false),
                #[cfg(test)]
                segment_submissions: std::cell::Cell::new(0),
                #[cfg(test)]
                expert_down_prepare_uses: std::cell::Cell::new(0),
                expert_gu_prepare,
                expert_axis_qmv,
                expert_consumer_qmv,
                #[cfg(test)]
                expert_consumer_dispatches: std::cell::Cell::new(0),
                #[cfg(test)]
                expert_axis_dispatches: std::cell::Cell::new(0),
                #[cfg(test)]
                flash_scratch_bindings: std::cell::Cell::new([[0; 2]; 3]),
                #[cfg(test)]
                flash_scratch_uses: std::cell::Cell::new(0),
                #[cfg(test)]
                flash_expert_outputs: std::cell::Cell::new([0; 4]),
                #[cfg(test)]
                flash_static_submissions: std::array::from_fn(|_| std::cell::Cell::new(0)),
                dense_qmv,
                mapped_qmv,
                qwen_expert_stages,
                #[cfg(test)]
                qwen_expert_stage_dispatches: std::cell::Cell::new(0),
                #[cfg(test)]
                mapped_qmv_groups: std::cell::Cell::new(0),
                #[cfg(test)]
                mapped_qmv_singles: std::cell::Cell::new(0),
                dense_finish,
                mapped_glu,
                mapped_mix,
                mtp_hidden_qmv,
                flash_verify_qmv,
                flash_verify_specialized,
                #[cfg(test)]
                flash_verify_specialized_uses: std::cell::Cell::new([0; 7]),
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
                device_greedy_pipeline,
                device_greedy_calls: std::cell::Cell::new(0),
                device_greedy_heads: std::cell::Cell::new(0),
                #[cfg(test)]
                device_greedy_reject_result: std::cell::Cell::new(false),
                ple,
                moe_switch,
                moe_mix,
                moe_finish_mix: std::cell::Cell::new(moe_finish_mix),
                moe_finish_mix_pipeline,
                #[cfg(test)]
                moe_finish_mix_uses: std::cell::Cell::new(0),
                gate_dot,
                router_pair,
                resident_scratch: crate::resident_scratch_policy::Budget::new(hc_kernel_setting(
                    "MLXL3_NATIVE_RESIDENT_SCRATCH",
                )?),
                gate_softmax,
                route_half,
                route_float,
                inverse,
                readers: std::cell::OnceCell::new(),
            })
        })
    }

    fn wait_command(&self, command: &CommandBufferRef) {
        let stats = wait_command(command, self.spin_wait_enabled);
        #[cfg(test)]
        for (counter, value) in self.spin_wait_stats.iter().zip(stats) {
            counter.set(counter.get() + value);
        }
        #[cfg(not(test))]
        let _ = stats;
    }

    fn command<T>(&self, encode: impl FnOnce(&CommandBufferRef) -> Result<T>) -> Result<T> {
        ensure!(!self.segment_failed.get(), "previous GPU segment failed");
        objc::rc::autoreleasepool(|| {
            let cmd = self.queue.new_command_buffer();
            #[cfg(test)]
            self.helper_commands.set(self.helper_commands.get() + 1);
            #[cfg(test)]
            let encoding = crate::flash_profile::span("native.central_encode");
            let value = encode(cmd)?;
            #[cfg(test)]
            drop(encoding);
            cmd.commit();
            #[cfg(test)]
            {
                self.helper_commits.set(self.helper_commits.get() + 1);
                self.helper_waits.set(self.helper_waits.get() + 1);
            }
            #[cfg(test)]
            let _wait = crate::flash_profile::span("native.central_wait");
            self.wait_command(cmd);
            ensure!(
                cmd.status() == MTLCommandBufferStatus::Completed,
                "Metal command failed ({:?})",
                cmd.status()
            );
            #[cfg(test)]
            crate::flash_profile::native_gpu_interval(cmd, "central");
            Ok(value)
        })
    }

    #[allow(
        unexpected_cfgs,
        reason = "objc 0.2 selector macro tests the legacy cargo-clippy cfg"
    )]
    fn owns_command_queue(&self, cmd: &CommandBufferRef) -> bool {
        // MTLCommandBuffer.commandQueue is a nonnull borrowed MTLCommandQueue.
        let queue: *mut metal::MTLCommandQueue = unsafe { msg_send![cmd, commandQueue] };
        queue == self.queue.as_ptr()
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
            self.wait_command(command);
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
        #[cfg(test)]
        let _copy = crate::flash_profile::span("native.readback_copy");
        #[cfg(test)]
        if let Some(span) = &_copy {
            span.data(serde_json::json!({"bytes":count*2}));
        }
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

#[cfg(test)]
#[path = "gpu/ssd_direct_tests.rs"]
mod ssd_direct_tests;

#[cfg(test)]
#[path = "gpu/ar10_model_tests.rs"]
mod ar10_model_tests;

#[cfg(test)]
#[path = "gpu/ar20_kernel_tests.rs"]
mod ar20_kernel_tests;
#[cfg(test)]
#[path = "gpu/ar33_kernel_tests.rs"]
mod ar33_kernel_tests;

#[cfg(test)]
#[path = "gpu/ar30_kernel_tests.rs"]
mod ar30_kernel_tests;

#[cfg(test)]
#[path = "gpu/ar20_capture_tests.rs"]
mod ar20_capture_tests;

#[cfg(test)]
#[path = "gpu/ar30_greedy_tests.rs"]
mod ar30_greedy_tests;

#[path = "gpu/segment.rs"]
mod segment;

#[cfg(test)]
mod fast_profile_tests;
