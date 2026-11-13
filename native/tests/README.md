# Native tests

`contracts.rs` exercises CPU contracts; `runtime_gpu.rs` contains checks requiring the appropriate runtime and hardware. The C++ files cover disk reads, queues, Metal capabilities, and memory helpers; some are proof harnesses. `fixtures/` preserves a Qwen counterexample fixture.

Rust targets are declared in the [root Cargo.toml](../../Cargo.toml). CI commands and options are in [rust.yml](../../.github/workflows/rust.yml); physical tests ignored by default must be explicitly selected and are not validated by hosted CI.
