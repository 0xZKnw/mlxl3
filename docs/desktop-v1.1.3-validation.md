# Desktop v1.1.3 — verification and UI performance

Version 1.1.3, build 18. Production engine behavior is unchanged from v1.1.2
(`a7ba6ab`). Existing unrelated local engine experiments are excluded from this
release. The renderer retains its token fade, syntax highlighter, math/table
views, colors, fonts and scroll controls.

## Changes and contracts

The message copy control writes `ChatMessage.content` directly to the native
pasteboard. It preserves every answer part, including Markdown/code and exact
UTF-8 spelling. It works for user messages, streaming/finished/interrupted
answers and reopened history. Thinking, tool output and source file paths are
not inserted. Empty messages leave the clipboard unchanged.

Markdown chunking and parsing run in a serial actor away from the UI executor.
The renderer has one active preparation and one latest pending snapshot. A
completed prefix may be displayed while more text arrives; replacement or
truncation rejects obsolete results. Leaving the view cancels its work and
invalidates its generation ticket. The newest final snapshot is eventually
displayed without dropping any bridge fragments. Chunks outside the scroll
viewport are loaded lazily, reducing unnecessary graphical surfaces.

The Markdown cache adopts the already validated local AUDIT-05 implementation;
the additional trial measures native rendering and large code fences. It does
not rerun that historical parsing microbenchmark to claim a new gain.

## Tests and source-level verification

Commands run on Apple M5, macOS 27.2 (26B5091g), Swift 6.4, SDK 26.5:

```sh
MLXL3_TEST_PYTHON=/path/to/python3.12 MLXL3_MACOS_SDK=/path/to/MacOSX26.5.sdk PYTHONPATH=src scripts/check-e2e.sh
swift format lint --strict apps/MLXL3Studio/Sources/MLXL3Studio/MessageCopyButton.swift tests/response-ui-check.swift
python3 -m py_compile tests/fake-desktop-engine.py
plutil -lint apps/MLXL3Studio/Resources/Info.plist
cargo fmt --check
cargo clippy --locked --all-targets --features chat -- -D warnings
cargo test --locked --features chat
cargo clippy --locked --all-targets --no-default-features --features chat -- -D warnings
cargo test --locked --no-default-features --features chat
cargo kani --lib --no-default-features --output-format terse
```

All passed. Python: 2 passed, 4 optional skips (missing ponyexl3 and local Ling
model). Rust: 21 unit and 14 integration tests passed in each configuration;
3 / 2 physical model/tokenizer tests remain ignored. Existing vendor warnings
concern deprecated SwiftMath font APIs, Command Line Tools search paths, and
Rust dependency `block` future compatibility.

Desktop coverage includes lifecycle/crashes, persistence ordering, deletion,
file import, history migration, MCP, actual subprocess bridge callbacks,
cancellation/reload, CLI transport, and native AppKit rendering. New checks
exercise exact complete-message copy, empty/fallback/interrupted answers,
tools and reasoning, 56 preparation parity cases, rapid changes,
replacement/truncation, disappearance/reappearance, a 20,000-line streamed
answer, and saving/reopening/copying that answer. Existing streaming checks
compare 4,845 Markdown fragments and 1,194 lexical fragments with full scans,
including Unicode boundaries, escapes, long tables and fences.

A separately compiled mutation copied only five characters. The new check
failed on the expected complete-text/Unicode assertion. The production source
was never changed to this mutation, and the full suite passed on final source.
Attempts to press the SwiftUI control through native accessibility returned
an empty tree in the standalone test host, including with a test window. The
control's native layout and the clipboard action used by its button are
tested; a manual click in the installed app remains unverified here.

Swift 6 compiler checking enforces the actor isolation and Sendable payloads.
There is no configured practical formal verifier for this SwiftUI/AppKit
source. Parser comparisons and asynchronous lifecycle checks are finite tests,
not a proof for every possible document or thread schedule. The Python engine
fixture performs subprocess I/O and is checked through the actual bridge;
it does not fit a pure CrossHair/Nagini proof.

Kani 0.68.0 / CBMC 6.11 verified all 17 existing engine harnesses with no failed
checks. This bounded verification covers the declared codec, checkpoint path,
budget/alignment and speculative acceptance contracts within each harness's
input/unwind bounds. It does not verify Swift, concurrent rendering, filesystem
effects or Metal/MLX floating-point execution. Tests and bounded verification
are not a complete proof of the application.

In particular, the arithmetic harnesses cover pack/decode for 0–65535 tiles
and K=1–8, QMV for 1–255 row/column tiles and K=1–8, arbitrary `u64` data
ranges, and four tensor/array dimensions (`u8`/`i8`) with item sizes 1/2/4/8,
unwind 6. Other bounds remain declared in the existing Rust harnesses. Some
checks are unreachable and `caller_location`/foreign-function warnings remain;
no reachable obligation failed and cover goals were satisfied.

## Performance protocol and evidence

The synthetic fixture is [benchmark.swift](measurements/desktop-v1.1.3/benchmark.swift).
It compiles against actual production Swift sources with `swiftc -O`, presents
`MessagesView` in a private offscreen native window (900 × 700), streams one
line per 50 ms interval, and measures `append` plus `layoutSubtreeIfNeeded`.
Each size/content pair has 2 warmups and 20 measured updates. Timer drift
records lateness beyond the requested 50 ms sleep, not frame rate. Sizes are
nominal 64/256/1024 KiB, prose with Markdown and HTML in an open code fence.

Power: AC, battery 100%; other desktop apps including Deezer and Codex remained
open. Thermal state is not controlled. No language-model process, compiler or
profiler runs concurrently with the corrected final pair. These are UI costs;
there is no measured inference throughput gain or universal FPS guarantee.

Earlier attempts and negative results remain in `opti.md` and the measurement
directory. A 5-second sample of the rejected actor-only candidate showed major
RenderBox/QuartzCore surface allocation cost on the main thread. That process
reported a 4.6 GB physical footprint; final comparative memory is unmeasured.
Moving parsing alone improved code while leaving very slow prose rendering.
Lazy chunk presentation was required for the retained candidate.

Corrected isolated pair, baseline v1.1.2 → retained candidate, milliseconds
per update (median / p95):

| Nominal size | Prose baseline → final | Code baseline → final |
| --- | --- | --- |
| 64 KiB | 17.608 / 32.554 → 0.862 / 6.114 | 2.754 / 2.880 → 0.606 / 0.664 |
| 256 KiB | 78.104 / 88.983 → 3.684 / 6.293 | 8.983 / 9.360 → 1.241 / 1.349 |
| 1 MiB | 66.233 / 284.232 → 0.965 / 8.137 | 34.190 / 34.301 → 3.145 / 3.329 |

The 1 MiB prose maximum falls from 3,460.541 to 74.232 ms; p95 timer drift
falls from 186.709 to 75.199 ms. Earlier final passes had different drift
(26.667 and 76.946 ms), so some pauses and variability remain. This is a
synthetic UI improvement, not a promise of continuous 60 FPS under GPU load.
Evidence: [baseline](measurements/desktop-v1.1.3/baseline-isolated.log),
[final](measurements/desktop-v1.1.3/final-isolated.log). Animation code is
retained; the experiment measures rendering with streaming enabled.

CI, clean release packaging and mounted-DMG checks are recorded separately in
the release validation asset after their completion. Installed-user-app manual
visual QA, GPU-contention FPS, other macOS versions and M1–M4 remain unverified.
