# Engine 2.0.3 beta 1 — build and verification, 6 November 2026

Published: [engine-v2.0.3-beta.1](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.3-beta.1). The archive was built from **b8472fc9073c3ff1ec1e314de327167de1d20e6a**, based on main **ed3bfdb8ff45e650fd8347b1a8f00931187284bf**. All three runtime processes report clean source `b8472fc9073c`, version 2.0.3. The later commit **f0d12a090a6222c4a0408f0528a0c8d448e38b94** changes one copied-project test fixture only; the shipped engine bytes were retained.

The original checkout and installed application were preserved. Runtime artifacts were built in an isolated worktree and copied byte-identically into the original checkout's `dist/`. Source is on [PR #151](https://github.com/0xZKnw/mlxl3/pull/151); no merge or installed-app replacement was performed. Desktop beta 7 remains its separately published build 38, bundled with engine 2.0.2.

## Implemented scope

The schema-1 archive contains `mlxl3`, two MLX dylibs, `mlx.metallib`, and its manifest. The executable embeds a separately compiled MLX helper and direct-Metal helper. Selection reads actual checkpoint configuration: only `qwen4_exp` uses native Flash; other models use MLX. The shared Clap CLI preserves aliases, flags, registry positions and image arguments. The wrapper uses exec to preserve process identity and cancellation behavior. Private, byte-verified helper extraction rejects corruption, invalid sizes/names, links and unsafe permissions; tests exercise concurrent installation and reuse.

LFM3-1.7B and LFM3-VL-2B share new F16 attention sigmoid/layout and seven-tap single-token convolution/state/gate fusions. Other shapes/dtypes retain existing operations. The initial prototype incorrectly assumed three taps; its real-model dispatch test failed and its sources/binaries/results were retained under `failed-conv3/`. The corrected K7 implementation explicitly dispatched on both real models and passed the same scalar/original-MLX oracles. No new speed or memory benchmark was run, as requested by the owner.

The ExLlamaV3 semantics reviewed were master 1.6.6 `bd1f519172406a9560bd5c90e61000691bb69a07` and dev `0c325eff053f3bdce1e59cd31bbd858d8d1bc6d0`. These Metal implementations adapt the math; they are not a claimed portable CUDA kernel. LiquidAI model revisions were text `794219432967af4826d8276d370b2d1676d1f870` and VL `15d4ec5cc43c5a6146dc675de0d89f939ec9c2da`.

## Inspected checks

Apple M5 / 24 GiB, macOS 27.2, Rust 1.98.1, MLX 0.32.2. One GPU model at a time, without concurrent compiler/prover work during physical trials. Counts below are separate configurations with overlapping tests, not additive coverage.

| Check | Inspected result |
| --- | --- |
| CPU hybrid suite | 418 passed, 10 ignored |
| Direct/chat suite | 429 passed, 108 ignored |
| Release MLX library | 364 passed, 319 ignored; before final path-only fix |
| Explicit physical kernel/model tests | 5 passed; 48 convolution multi-step shape cases, 72 attention/layout cases, malformed no-dispatch and real dispatch/reset, four checkpoint references |
| Fresh explicit-metallib subprocess regressions | 2 passed on final helper; missing path and directory fail before GPU initialization |
| Metadata/coverage focused suite | 78 passed and 2 subtests |
| Complete affected packaging/converter CI selection after fixture correction | 216 passed locally; Linux-only differences remain platform-specific |
| Strict lint/build | rustfmt, Clippy MLX/chat, direct/chat, CPU dispatcher and actual embedded dispatcher passed; shell syntax, Python Ruff and source inventory passed |
| Signed package | All payload hashes, helper embedding hashes, arm64 identities and ad-hoc signatures inspected |
| Actual Desktop beta-7 updater | Disposable install, relocation, runtime execution and rejection/fallback passed |

The explicit metallib path fix was necessary because extracting a helper into the user cache changes its executable directory. Its old behavior silently used the build machine's MLX installation. A retained old helper reproduced that negative control. The final wrapper selects its own adjacent Metal library; removing that library from a disposable relocated bundle now produces the expected error despite the build-machine MLX installation being available. Five real CLI/bridge/HTTP loader logs confirm `libmlx.dylib` came from the relocated package.

| Independent checkpoint reference | Final logit KL | Image feature relative L2 |
| --- | ---: | ---: |
| Text 4.49 | 0.00005172896 | n/a |
| Text 8 | 0.00040396852 | n/a |
| VL 4 control | 0.00003621023 | 0.00440135 |
| VL 8 calibrated | 0.00009866317 | 0.00443477 |

All four had 128000 finite logits, below the preregistered KL 0.001 control limit, and exact reset/replay. Primitive output tolerance was absolute 0.002 + relative 0.002; convolution state matched the independent F16 scalar oracle bit-for-bit. Real reset/replay also compared all retained states.

The actual CLI compared baseline `96edda9caded653babd72109d34a6523e67b3cd1` (binary SHA-256 `b9f7e6a327ec21edf0bad2c980c555922f5451d1a26985edd0ac13c51c506857`) with the final package on tokens 1,2,3, `--batch --states`, using exactly the same published weights. Text 4.49: KL 0.00011756834, maximum logit difference 0.0703125, 35 finite state arrays, worst relative state L2 0.00863635. VL 4.49: KL 0.00069134690, maximum logit difference 0.1015625, 35 finite state arrays, worst relative state L2 0.00883022. Limits were KL 0.001 and relative state L2 0.01, fixed before execution.

Published 4.49 text/VL bridge journeys passed math, image rejection for text, distinguishable red/blue images, malformed-image rejection and recovery. HTTP passed image JSON, SSE, malformed/remote-image rejection and recovery. Native Flash EXL3 2.05-bpw checkpoint `65c89531` produced four exact independent two-token hashes `9f850af1cbba074b`, rejected one invalid request, cancelled both prefill and decode and recovered exactly; the child was joined and dyld showed Metal with no MLX. Its fixed-slot qualification used cache 3 GiB / 53 slots, four readers, lookahead depth 0. The first invocation omitted that cache override and adaptively chose 14 slots: all generation/cancellation checks passed, but the existing harness's hardcoded 53-slot assertion failed. That original `failed/parity:false` report remains intact; the corrected-protocol result is separate.

## Source-level verification and limits

Kani 0.68.0 / CBMC 6.11.0 checked production CPU helpers: causal window full-u32 (4 obligations, 2 covers), patch index symbolic-u8 dimensions (32, 1), K7 buffers full-i32 (19, 2), attention products full-i32 (17, 2), exact Flash admission up to 16 symbolic bytes and full-usize helper size (67, 2). The actual helper functions were checked, with unwinding assertions retained. These bounded/scalar properties do not prove MLX, Metal execution, allocation, I/O races or complete generation. The full Kani package suite was not completed locally.

CBMC was attempted on the real C++20 MLX bridge with its actual include directory; code 6, unsupported standard-library/header parse. No rewritten surrogate or unjustified stub was used. CrossHair invocation failed because the module is unavailable locally. C++/Metal and Python source-level properties remain unverified. The 319 ignored release-MLX tests were not executed as a full GPU suite; five relevant physical tests were selected explicitly. The final path-only fix was covered by its two fresh regressions and relocated production journeys, not a repeat of the entire preceding library suite.

The mixed published VL checkpoint still has its historical independent feature L2 0.0165145, above the 0.01 qualification limit. CLI parity against the old MLX path and successful color journeys do not remove that discrepancy or prove broad VQA/OCR/video quality. An installed-app GUI image journey and general quality/performance campaigns remain unverified. The archive is ad-hoc signed, not notarized.

## Commands and retained evidence

Commands were run in the isolated engine worktree; the MLX root was the project's pinned MLX 0.32.2 installation. Runtime identities, source hashes, model identities, raw logs and failed attempts are included in the release verification archive.

```sh
cargo fmt --check
cargo clippy --locked --all-targets --features mlx,chat -- -D warnings
cargo clippy --locked --all-targets --features chat -- -D warnings
cargo clippy --locked --all-targets --no-default-features --features hybrid-dispatch -- -D warnings
cargo test --locked --no-default-features --features hybrid-dispatch
cargo test --locked --features chat
cargo test --release --locked --features mlx,chat --lib --test runtime_gpu --no-run --message-format=json
# Execute the emitted release libtest serially; select the five relevant physical tests with --exact --ignored.
cargo kani --lib --no-default-features --harness lfm3_short_conv_buffers_are_bounded
cargo kani --lib --no-default-features --harness lfm3_attention_gate_buffers_are_bounded
cargo kani --lib --no-default-features --harness hybrid_routes_only_exact_flash_identifiers
MLXL3_ENGINE_BACKEND=hybrid scripts/build-engine.sh
MLXL3_TEST_MLX_ENGINE=build/hybrid-helpers/mlx python -m pytest -q tests/test_metallib_override.py
python scripts/check_native_coverage.py
PYTHONPATH=src python -m pytest -q tests/test_package_engine.py tests/test_engine_builder.py tests/test_build_manifest.py tests/test_native_quantizer.py tests/test_native_transforms.py tests/test_native_converter_host.py tests/test_native_capture.py tests/test_native_feedback.py
```

The build also supplied `MLXL3_MLX_ROOT`, `MLXL3_BUILD_PYTHON`, deployment target 26.2 and, for embedded-dispatcher Clippy, both signed helper paths. Final CLI forwards used `--tokens 1,2,3 --states --batch`. Existing qualified bridge/HTTP scripts were retained and adapted to the relocated final archive and the published 4.49-bpw weights. Source hashes and concrete results are linked in [validation.json](measurements/engine-v2.0.3-beta.1/validation.json) and [publication.json](measurements/engine-v2.0.3-beta.1/publication.json).

Remote CI on original b8472fc9 passed profile checks and packing proof, then exposed six copied metadata fixtures omitting the added MLX-VLM notice. The failure was reproduced locally and fixed only in the fixture, with 216 affected tests passing. Corrected-head f0d12a09 workflows have been inspected: profile and the previously failing native-coverage job succeeded; the remaining matrix is still queued/in progress. No complete CI-green claim is made. Original failure logs and corrected-head receipts are retained. Release assets were individually compared against GitHub's SHA-256 digests and sizes; the release was published with `latest=false`.

Credits and complete artifact hashes are in [the English release notes](release-engine-v2.0.3-beta.1.md).

The anonymous asset returned HTTP/2 200 with Content-Length 73029179. An initial checker expected the HTTP/1 phrase `200 OK` and failed; parsing the numeric status correctly passed against the retained original headers. No release bytes changed.
