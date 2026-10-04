# File import validation — MLXL3 Desktop v1.1.2, build 17

Validation used an isolated checkout of the previously published `fbfb357`
plus only this change. Existing local inference, optimization, API and UI
changes were preserved and excluded from the release. No optimization
benchmark was run and no performance gain is claimed.

Environment: Apple Silicon/M5, macOS 27.2, Apple Swift 6.4, macOS SDK 26.5,
Rust/Cargo 1.98.1, Kani 0.68.0 and CBMC 6.11.0. The app targets macOS 26.2+.

## Commands and results

- `MLXL3_TEST_PYTHON=/Users/justin/Documents/mix-stq1_0/.venv/bin/python
  MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
  PYTHONPATH=src scripts/check-e2e.sh` — passed. The Python suite passed two
  tests and skipped four optional converter/local-model checks (`ponyexl3`
  and the local Ling checkpoint were unavailable). The desktop suite passed
  lifecycle, file import, timeline, MCP preferences, bridge, incremental
  rendering and CLI transport checks.
- `swift format lint --strict
  apps/MLXL3Studio/Sources/MLXL3Studio/ChatAttachment.swift
  tests/file-import-check.swift` — passed.
- `cargo fmt --check` — passed.
- `cargo clippy --locked --all-targets --features chat -- -D warnings` and
  the same command with `--no-default-features` — passed.
- `cargo test --locked --features chat` — 21 unit tests and 14 integration
  contracts passed; three existing physical GPU/draft/tokenizer checks were
  ignored. With `--no-default-features`, the same 35 tests passed and two
  optional checks were ignored.
- `cargo kani --lib --no-default-features --output-format terse` — 17
  existing harnesses passed, no failures, all requested cover properties
  satisfied. This is bounded model checking of the Rust paths, not proof
  of the SwiftUI feature or the whole application.
- `git diff --check` — passed. The existing desktop CI runs the new file
  import check on pushes and pull requests. Native CI now uses Rust 1.98.1,
  checks both feature configurations, and no longer calls deleted Python
  parity scripts. Its actual GitHub results and the DMG packaging checks
  are recorded with the release assets.

## Import coverage

`tests/file-import-check.swift` calls the production importer, `StudioModel`,
the real subprocess bridge and conversation store, and renders the composer
and message timeline through AppKit/SwiftUI. Test files, history and
preferences are disposable and isolated from personal data.

The checks cover UTF-8, UTF-8 BOM, both UTF-16 byte orders, malformed UTF-8
and UTF-16, binary content, source code, CSV, extensionless text, multipage
PDFs with page references, blank/scanned PDFs, protected PDFs, corrupt PDFs,
missing files, folders, remote URLs, per-file limits, duplicates, eight-file
admission, rejection of a ninth file, and mixed valid/invalid batches.

Admission checks exhaust a finite matrix of 150 cases: existing counts
0–9, aggregate byte counts 0/524287/524288, and incoming sizes
0/1/262143/262144/262145. Encoding checks sample 128 strings with a fixed
seed in two encodings (256 cases). Neither check establishes a general
formal proof.

Integration checks exercise attachment-only sends, removal, per-conversation
drafts, failed imports, error dismissal, switching/deletion around an async
import, follow-up context, compatible legacy history and history reloading
after deleting the original file. The fake engine returns the actual
generation request so omitted document content is observable.

Two temporary mutations were rejected by the tests: removing the lossless
UTF-16 round-trip check and sending only the visible question to the engine.
Both mutations were restored and the focused suite passed again.

## Proof bounds, failures and remaining gaps

Rust arithmetic harnesses check pack/decode inverses for 0–65535 tiles and
K=1–8, QMV shapes for 1–255 row/column tiles and K=1–8, exact data ranges for
arbitrary `u64` values, and four tensor/array dimensions (`u8`/`i8`) with item
sizes 1/2/4/8 and unwind 6. Other existing harnesses retain their own bounds
in source. Kani reports some unreachable checks and unsupported
`caller_location`/foreign-function constructs; none produced a reachable
failure, and cover goals were satisfied. This does not verify GPU execution,
filesystem I/O, subprocess concurrency or PDFKit internals.

No compatible source-level formal verifier is configured for this Swift
application. Swift 6 compilation, runtime checks, finite boundary checks,
seeded tests and integration tests are the evidence for the import feature.

Verification first found Foundation silently accepting an odd-length UTF-16
payload; production now rejects any decode whose encoded bytes differ from
the original payload. A protected-PDF fixture initially omitted the required
owner password and was corrected. Initial sandboxed builds could not write
compiler caches; authorized builds succeeded outside that sandbox.

The previous GitHub native run failed on unused Metal helpers under Linux
and on references to removed Python tests. Local checks also exposed a
macOS CPU-only CLI reference to the absent Metal module and unused request
fields. Feature gates and the codec recovery regression were corrected;
both configurations then passed strict lint and their full test suites.

Computer Use access to the isolated QA application was not approved. The
native file chooser, keyboard shortcut and Finder drag-and-drop therefore
remain unverified by automated UI interaction. Production view compilation,
AppKit rendering and the shared import/send path were tested. Scanned PDFs
do not receive OCR; images and Office documents are intentionally unsupported.
The app remains ad-hoc signed and is not notarized.

Raw local logs are under `/tmp/mlxl3-file-import-*.log`; those paths are
temporary. A compact copy of the successful validation and packaging evidence
is shipped as `validation-v1.1.2.txt` with the release.
