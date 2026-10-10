//! Native migration. This crate does not call Python or embed an interpreter.
#[cfg(feature = "mlx")]
pub mod affine;
#[cfg(feature = "mlx")]
pub mod array;
pub mod checkpoint;
pub mod codec;
mod contracts;
pub use contracts::{reusable_prefix_len, smallm_kernel_key};
pub mod dflash;
#[cfg(feature = "mlx")]
pub(crate) mod embedding;
#[cfg(feature = "mlx")]
pub mod gated_delta;
#[cfg(feature = "mlx")]
pub mod gemma4;
#[cfg(all(target_os = "macos", feature = "direct-metal"))]
pub mod gpu;
pub mod hub;
#[cfg(feature = "mlx")]
pub mod lfm2;
#[cfg(feature = "mlx")]
pub mod linear;
#[cfg(feature = "mlx")]
pub mod ling;
pub mod mcp;
pub mod memory_policy;
#[cfg(feature = "mlx")]
pub mod moe;
pub mod mtp;
#[cfg(feature = "mlx")]
pub mod qwen35;
pub mod registry;
#[cfg(feature = "mlx")]
pub mod router;
pub mod speculative;
pub mod streaming;
#[cfg(feature = "chat")]
pub mod tokenizer;
pub mod tool_call;
