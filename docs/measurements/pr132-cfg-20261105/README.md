# PR132 non-Metal test-counter cfg correction

Base `92feadef`. The two A11 destination counters were `cfg(test)` even when their only writers/readers were absent without macOS/direct-metal. Strict no-default/chat all-target Clippy reproduced the dead-field failure before editing. Four attributes now use `cfg(all(test, target_os = "macos", feature = "direct-metal"))`: two fields and their two initializers. The shared CPU guards and tests remain compiled; there is no inference behavior change or blanket dead-code allowance.

The same strict Clippy command passes after the fix, `cargo fmt --check` passes, and the selected SSD CPU integration tests pass **19/19, 0 ignored**. This compiler/lint reproduction is the regression oracle; an additional test duplicating cfg syntax would not add meaningful coverage. The existing `.github/workflows/rust.yml` Native Rust ubuntu/macOS matrix already runs strict Clippy on pushes and pull requests; no workflow change was necessary.

Kani 0.68.0 / CBMC 6.11.0 rechecked the real, unchanged record-reuse eligibility helper: **1 obligation passed, 5/5 covers**, symbolic booleans and the complete `usize` depth domain, no assumptions/stubs. Wall time was 5.2374 s including compilation; the bounded process group was fully stopped/joined. This property does not establish cfg correctness, Darwin I/O, Metal or concurrency.

All commands, source hashes, statuses and lossless compressed logs are in [report.json](report.json). The target was a private APFS clone; the root target was never modified. No GPU, model, benchmark, Linux cross-build or remote CI was run by this agent. Root was notified that CPU work had ended before metadata archival.
