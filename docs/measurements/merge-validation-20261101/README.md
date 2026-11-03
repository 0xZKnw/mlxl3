# Integration qualification — 1 November 2026

Status: local functional qualification complete; remote proof coverage incomplete.
Published on main through PR64; this is not a release or an installed-app replacement.

PR64 integrates main `f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6`, Cold U
`9880fa4f923c69a13a1cfe09ccb1d61068bd76de`, PR61
`a36ba5077bf386466660d6b478058fbe6880245c`, PR62
`16c765aade25ad6ff85f974aa6da16e3d93ae4df` and PR63
`01c678ebcbb67b52ffbfa7fb98480edeb44400d3`.
The combined runtime source is `b00e51d38afe237bac72c5b798efd3d0cc1bff7e`.
Later verification-script, workflow and evidence changes do not change these
runtime sources. Binary/source SHA-256 identities are in `identity.json`.

PR61 conflicts deliberately retain the locally corrected T17/T18a code and
tests. The discarded T04 runtime is not reintroduced. Its original code and
boundary tests remain in PR61's commit history; the subsequent adapted tests,
negative results and removal are recorded in `opti.md`. PR62's runtime fixes
and PR63's proof partitions are retained. The workflow preserves additional
test-only profiling and cache-aging obligations outside PR63's library inventory.

## Local checks

- `cargo test --locked --features chat`: 276 passed, 10 ignored; two additional
  targets contained zero tests and provide no GPU qualification.
- `cargo fmt --check` and strict `cargo clippy --locked --features mlx,chat
  --all-targets -- -D warnings`: passed.
- `cargo test --locked --release --features mlx,chat --no-run`: built the actual
  FFI, CLI and test binaries in 3m34s. Cargo used one build worker and MLX 0.32.2.
- `MLXL3_SKIP_METAL_CHECK=1 scripts/check-e2e.sh` with Python 3.12: 2,205 passed,
  137 skipped, 10 warnings in 207.10s. Skips include physical GPU/model cases
  and PR63's Linux-only process runner; they are not counted as passed.
- C++17 OS-version parser under ASan/UBSan: 1,190 checks passed.
- `ruff check` and `ruff format --check` on Kani orchestration/tests: passed.
- Ordinary MLX release executions: 216 library tests passed / 191 ignored,
  44 CLI tests passed / 9 ignored. Physical selections are counted separately below.

Desktop execution initially failed under Command Line Tools because its
SwiftUI macro plugin was unavailable. Xcode 27.2 built the app, but strict
checks then rejected the pre-existing `VoiceController.installTap` deprecation
introduced by SDK 27. Switching to SDK 26.5 exposed Swift Build reusing a
27.2 SwiftMath module despite the SDK override. The existing SDK override in
`scripts/check-desktop.sh` now selects the native SwiftPM backend explicitly.
Normal CI without that override is unchanged. The native backend is deprecated
by Xcode 27; this compatibility path will need revisiting when it is removed.
All failed logs are retained. SDK 27 strict audio-API migration is outside scope.

The main Desktop suite passed on SDK 26.5 (including 1,179 mode/Lab,
965 sampling, 114 idle and 229 workspace checks, plus real bridge framing,
oversize-input termination and recovery). Its nested Studios invocation then
failed with the same SDK mismatch. The override was fixed in that caller too,
and the global success message moved after Studios. The affected Studios rerun
passed: 39 Python tests, 41 project/artifact checks, 25 sandbox checks and
29 MCP protocol checks. The successful main suite and affected rerun are
separate executions; the initial whole `check-e2e.sh` invocation did not pass.

## Source-level verification limits

Kani 0.68 checked the actual `metal_capabilities.rs` source with symbolic u32
flags and boolean M5 identity: two assertions passed, all four covers satisfied.
The actual attention and GDN width harnesses retain full i32 domains but each
hit the local 45-second limit without a verdict. A 512 MiB process-group RSS
cap was retained; descendants were stopped and joined. Reports and logs are
included. These are not complete Rust/MLX/Metal proofs.

CBMC 6.11 was attempted on the real C++ parser tests with libc++ headers and
safety/unwinding assertions. It failed parsing libc++ (its C++ dialect support
ends at C++11); no proof was obtained for this C++17 implementation. Swift and
the shell SDK override have no source-level proof; they are checked through
compilation and execution.

## GPU and performance scope

Three affected physical selections were preregistered: Cold component/default
activation, independent full-model oracle, and PR62's DFlash-to-MTP invalid-head
cleanup with unchanged ordinary logits. Plans reuse the original GPU1 anchor
and the explicitly authorized cumulative 64 MiB write / 1 GiB read limits.
Only one GPU process may run, after compilers/provers/Desktop tests finish.

All three selections passed, with zero new swap reads and writes in each run:

- Component/default: 48 cases, byte-exact parity and verified overlap before
  the held I/O tail; 2.789 seconds.
- Independent model oracle: five tokens across all 48 layers, 248,320 logits
  and reset-state checks; maximum absolute error and RMSE both zero,
  176 natural Cold waves; 4.641 seconds.
- DFlash-to-invalid/empty-MTP switch: the draft is retired before head loading
  fails and subsequent ordinary logits are byte-exact; 2.404 seconds.

All GPU processes were joined. Cumulative fixed-anchor writes were 42,450,944
bytes and reads 1,054,834,688 bytes, within the authorized limits. Plan/result
receipts and logs are included; no reanchoring occurred.

No integration throughput benchmark is claimed. The earlier Cold U measurements
and their drift limits remain in [the performance report](../../cold-ready-m5-20261101.md).
Merging compatibility/proof changes does not relabel those timings as measurements
of this combined binary.

## Remote checks

Exact-runtime-head runs: Native Rust `41320660981`, Desktop `41320660984`,
Apple compatibility `41320660986`, conversion `41320660993`.
At inspection, native Linux, Apple runtime/loader, allocator Kani, conversion
and packing-proof jobs passed. Native macOS reached its 20-minute job deadline
during compatibility tests after the MLX bridge build/lint and earlier tests
passed; job 124889952959 contains no failed assertion before cancellation.
The native job deadline is increased to 35 minutes without removing any check
or changing proof deadlines. This adjusted remote run still requires completion.
Apple guard proofs reached their 15-minute deadline without complete proof;
other proof shards and Desktop were still running at this snapshot. These
outcomes are not an all-green CI claim; exact job receipts are included.
The prior PR63 head retains four 600-second proof timeouts; no check was
disabled or reclassified as successful to merge it.

## Publication receipt

Final qualification commit `e4a8e86702ab2b448b253a1042daea946a20f5b4` was
pushed and PR64 merged as `e7c15b18dfdf77b306d6e3311a65636ac3ac6247`
on 1 November 2026 at 15:46 UTC. GitHub reports PR61, PR62 and PR63 merged
through the preserved ancestry; their exact heads are ancestors of main.
The merge tree is identical to the final qualified branch tree.

Main push workflows were inspected: Desktop `41322940775` was in progress;
Native Rust `41322940759`, Apple compatibility `41322940757` and conversion
`41322940850` were queued. The qualification commit's PR workflows had also
started. These are pending checks, not successful results. This final receipt
changes documentation only; no additional model run or speed claim is made.
