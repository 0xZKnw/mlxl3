# Issue 162: checkpoint download integrity and cancellation

## Scope and provenance

This independent change addresses N04/N05 of [issue 162](https://github.com/0xZKnw/mlxl3/issues/162). It is based on main `28cf4c210490f9a028e9bdda5b1e067e1cdbca13`, not on PR 161. PR 161 was inspected at `ebacd5ecdec128c249607eac264c28a4ba836ec2`; its changed files do not include `native/src/hub.rs` or Cargo dependencies. The shared append-only experiment journal may require ordinary merge reconciliation.

The remaining issue 162 findings are **not closed by this PR**. There is no default speculative-decoding change, model throughput claim, application installation or merge.

## Contract being repaired

Generic EXL3 download staging previously admitted a same-size file without checking payload identity. The selected Hugging Face commit provides LFS SHA-256 or ordinary Git blob identity; the download must retain that identity and check it before reuse and publication. Structural checkpoint validation remains necessary, not a substitute for hashing.

Resume is allowed for a shorter partial. An oversized file or a complete file with incorrect content must not be resumed from its existing offset. Cancellation must be observed even when every staged file is complete, and immediately before the rename that begins publication. A cancellation received after that commit point must not be described as rolling back an already published installation.

## Verification

Cloud Linux results (logs in [the evidence directory](measurements/issue162-download-integrity/)):

- Targeted Hub suite: 12 passed, 0 ignored, including five new regressions using actual curl with disposable file:// transfers. Final rerun after the proof-only addition also passed all 12.
- Checksum-bypass mutation: expected test failure, exit 101. Original implementation restored and suite passed again.
- Full CPU library suite, no default features: 323 passed, 9 ignored.
- Chat library suite, serial: 333 passed, 10 ignored.
- Strict Clippy: library/no-default-features and all targets/chat passed. Formatting and diff whitespace checks passed.
- Native source inventory: 6 tests passed. Kani manifest: 5 passed after adding the new proof name to the explicit additions inventory. The initial missing-pytest attempt and the initial manifest mismatch are retained as logs.
- Independent final source review found no concrete introduced defect; it did not execute tests.

A bounded symbolic Kani harness calls the actual metadata-admission method for LFS/Git selection, presence, ASCII character and lengths 0–65 (unwind 140). It is registered in the proof manifest. Attempting the exact harness returned `no such command: kani`: **not verified**, not a success or counterexample. The formal-proof skill/reference was unavailable. Hash implementations, I/O, signal races and subprocess supervision are not formally proved.

Remaining acceptance gaps: remote HTTP 416, live Hugging Face authentication, real SIGTERM delivery, full run_download registry rollback and crash recovery were not exercised. The curl seam tests are not a full installation E2E test. Existing adversarial staging/symlink races and comprehensive curl supervision are outside this change. Postcommit staging cleanup is best effort with a stderr diagnostic; a valid installed-and-registered model is no longer falsely reported as failed solely because staging cleanup failed.

No user Mac, Apple build, model, GPU, real Hugging Face model download, numerical parity or speed benchmark was used. Hosted CI of the eventual published commit must be checked separately; these local results are not a global CI-green claim.
