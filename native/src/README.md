# Native engine sources

The Rust package is defined by the [root Cargo.toml](../../Cargo.toml); `lib.rs`, `main.rs`, and `hybrid_main.rs` are its entry points. `array.rs`, `linear.rs`, `codec.rs`, `checkpoint.rs`, and `registry.rs` provide shared primitives and services. `lfm2.rs`, `lfm3.rs`, `qwen35.rs`, `qwen4.rs`, `gemma4.rs`, and `ling.rs` organize model families; `openai.rs`, `mcp.rs`, and `streaming.rs` handle interfaces.

Subdirectories describe specialized modules and tests: [gpu/](gpu/README.md), [ssd_reads/](ssd_reads/README.md), [ssd_moe/](ssd_moe/README.md), [qwen4/](qwen4/README.md), [hub/](hub/README.md), [linear/](linear/README.md), and other neighboring modules. See also the [shaders](../shaders/README.md), [FFI](../ffi/README.md), [native tests](../tests/README.md), and [engine README](../README.md).
