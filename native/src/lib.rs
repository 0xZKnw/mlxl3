//! Native migration. This crate does not call Python or embed an interpreter.
#[cfg(feature = "mlx")]
pub mod affine;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod ar10_defaults;
#[cfg(test)]
mod ar33_codeword_table_tests;
#[cfg(feature = "mlx")]
pub mod array;
pub mod checkpoint;
pub mod codec;
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) mod cold_ready;
mod contracts;
pub mod convert_matmul;
pub mod quantize;
pub mod tensor_layout;
pub use contracts::{
    GenerationStopReason, exl3_lane_order_enabled, exl3_lane_order_key, native_prompt_reuse,
    reusable_prefix_len, smallm_kernel_key,
};
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod ar20_kernel_policy;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod ar30_router_policy;
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) mod dec_one;
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) mod dense_epilogue;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod device_hot_policy;
pub mod dflash;
#[cfg(feature = "mlx")]
pub(crate) mod embedding;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod expert_ar_policy;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod fixed_residency_policy;
#[cfg(any(test, kani))]
mod flash_cache_profile;
#[cfg(any(feature = "mlx", all(target_os = "macos", feature = "direct-metal")))]
mod flash_config;
#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal", feature = "chat"),
    test,
    kani
))]
pub(crate) mod flash_draft_budget;
pub(crate) mod flash_draft_metadata;
#[cfg(any(
    all(feature = "mlx", feature = "chat"),
    all(target_os = "macos", feature = "direct-metal", feature = "chat"),
    kani
))]
mod flash_draft_options;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod flash_expert_scratch;
pub mod flash_memory;
#[cfg(any(feature = "mlx", all(target_os = "macos", feature = "direct-metal")))]
pub(crate) mod flash_ngrams;
#[cfg(test)]
pub(crate) mod flash_profile;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod flash_qmv;
pub mod flash_support;
#[cfg(feature = "mlx")]
pub mod gated_delta;
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) mod gdn_epilogue;
#[cfg(feature = "mlx")]
pub mod gemma4;
#[cfg(all(target_os = "macos", feature = "direct-metal"))]
pub mod gpu;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod gpu_greedy_policy;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod gpu_wait_policy;
#[cfg(any(test, kani))]
mod hc_gdn_policy_tests;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod hc_workspace_policy;
pub mod hub;
#[cfg(feature = "mlx")]
pub mod lfm2;
#[cfg(feature = "mlx")]
pub mod linear;
#[cfg(feature = "mlx")]
pub mod ling;
pub mod mcp;
pub mod memory_policy;
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) mod metal_capabilities;
#[cfg(feature = "mlx")]
pub mod moe;
pub mod mtp;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod pipeline_caps;
#[cfg(feature = "mlx")]
pub mod qwen35;
#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal"),
    test,
    kani
))]
mod qwen35_config;
#[cfg(feature = "mlx")]
pub mod qwen4;
pub mod registry;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod resident_scratch_policy;
#[cfg(feature = "mlx")]
pub mod router;
pub mod runtime_stats;
pub mod speculative;
#[cfg(feature = "mlx")]
pub mod ssd_moe;
#[cfg(any(
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal"),
    test,
    kani
))]
pub(crate) mod ssd_prefetch;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
pub(crate) mod ssd_reads;
pub mod streaming;
pub mod structured;
#[cfg(feature = "chat")]
pub mod tokenizer;
pub mod tool_call;
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) mod weighted_epilogue;

#[cfg(all(target_os = "macos", feature = "direct-metal", feature = "chat"))]
mod native_bf16;
#[cfg(all(target_os = "macos", feature = "direct-metal", feature = "chat"))]
pub mod native_draft;
