//! Native migration. This crate does not call Python or embed an interpreter.
#[cfg(feature = "mlx")]
pub mod affine;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod ar10_defaults;
#[cfg(test)]
mod ar33_codeword_table_tests;
#[cfg(feature = "mlx")]
pub mod array;
pub mod backend;
pub mod checkpoint;
pub mod codec;
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) mod cold_ready;
mod contracts;
pub mod convert_matmul;
pub mod image_budget;
#[cfg(any(feature = "mlx", test, kani))]
mod mtp_projection;
pub mod quantize;
pub mod tensor_layout;
pub mod transaction;
pub use contracts::{
    GenerationStopReason, INKLING_MOE_GROUP_MAX, INKLING_PREFILL_SLICE, Q36_MOE_GROUP_MAX,
    Q36_MOE_TAIL_MIN, QWEN_PREFILL_SLICE, QwenPipeline, evict_band, evict_width,
    exl3_lane_order_enabled, exl3_lane_order_key, inkling_moe_group_setting, mtp_evict_setting,
    mtp_post_norm_setting, native_prompt_reuse, prefill_span_end, prefill_span_end_merged,
    q36_head_lane_setting, q36_moe_group_setting, qwen_pipeline, qwen_pipeline_key,
    reusable_prefix_len, smallm_kernel_key,
};
#[cfg(feature = "mlx")]
pub use contracts::{switch_glu_layout, switch_glu_route_rows};
pub mod agent_loop;
pub mod agent_turn;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod ar20_kernel_policy;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod ar30_router_policy;
#[cfg(feature = "chat")]
pub mod d1;
#[cfg(feature = "chat")]
pub mod d1_parsing;
#[cfg(all(feature = "mlx", feature = "chat"))]
pub mod d1_server;
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) mod dec_one;
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) mod dense_epilogue;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod device_hot_policy;
pub mod dflash;
#[cfg(any(
    all(target_os = "macos", feature = "direct-metal", feature = "chat"),
    test,
    kani
))]
mod draft_storage;
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
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod frame_arena;
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
pub mod inkling2;
#[cfg(feature = "mlx")]
pub mod inkling2_dflash2;
#[cfg(feature = "mlx")]
pub mod inkling2_draft;
#[cfg(feature = "mlx")]
pub mod inkling2_lookup;
pub mod inkling2_vision;
#[cfg(feature = "mlx")]
pub mod lfm2;
pub mod lfm3;
pub mod lfm3_audio;
#[cfg(feature = "mlx")]
mod lfm3_cert_head;
#[cfg(feature = "mlx")]
mod lfm3_kernels;
#[cfg(feature = "mlx")]
mod lfm3_vision;
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
mod native_executor;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod pipeline_caps;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod qsa_chain_policy;
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
pub mod qwen_profile;
pub mod registry;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod resident_scratch_policy;
#[cfg(feature = "mlx")]
pub mod router;
#[cfg(any(all(target_os = "macos", feature = "direct-metal"), test, kani))]
mod router_network;
pub mod runtime_stats;
pub mod sampling;
pub mod speculation_plan;
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
#[cfg(feature = "mlx")]
pub mod topk_readback;
#[cfg(any(feature = "mlx", test, kani))]
pub(crate) mod weighted_epilogue;

#[cfg(all(target_os = "macos", feature = "direct-metal", feature = "chat"))]
mod native_bf16;
#[cfg(all(target_os = "macos", feature = "direct-metal", feature = "chat"))]
pub mod native_draft;
#[cfg(any(
    all(target_os = "macos", feature = "direct-metal", feature = "chat"),
    test,
    kani
))]
mod native_draft_gpu;

#[cfg(feature = "chat")]
pub mod vision_input;
