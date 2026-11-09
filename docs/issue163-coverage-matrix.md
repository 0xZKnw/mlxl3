# Issue162 — all 42 findings reconciled

Implementation candidate, 2026-11-09. Every card has a concrete change or inherited correction below. This does **not** mean every physical/behavioral acceptance gate has passed. No user Mac, merge, release, installed app or speed claim.

## Integration and validation

First full expansion published as PR163 head b28d33b72fc2e495f8b69611217a6a9e2bf8d798 (tree b998b08c). This follow-up repairs the six Apple source errors and converter test isolation, adds experimental GPU projections, worker read checkpoints, durable install reconciliation and incremental lexical caching. Kani inventory memory failure and two route-proof timeouts remain open; their experimental fixes are excluded from this publication. Published does not mean CI-qualified. Default reference paths remain available; new numerical/performance candidates stay opt-in.

Fresh aggregate:392 CPU library tests passed/10 ignored; strict all-target CPU/chat Clippy passed. Python aggregate221 passed/1 physical skip; packaging77 passed after formatting. Proof-orchestration/source-inventory tests are not Kani proofs. Swift compiler/AppKit/Metal unavailable locally: exact-head hosted macOS compilation and behavioral gates remain required. Physical GPU parity, model continuations, user-interface visuals/IME/accessibility and throughput remain unmeasured.

| Card | Concrete implementation | Validation boundary |
|---|---|---|
| N01 | Stop verification at accepted/rejected/bonus EOS and token budget; no post-terminal forward. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| N02 | Retire every command despite earlier SSD/GPU failure; shared poisoned producer requires reload; NSError diagnostics. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| N03 | Shared cancellation latch plus worker checkpoints and ≤256KiB submissions, logical-window release and drain before retry. Active OS reads cannot be forcibly preempted. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| N04 | Pinned content hashes and verification before reuse/publication; final-file size bounded across redirects. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| N05 | Resumed bounds and durable transaction/receipt/registry identity reconciliation; valid publication preserved after ambiguous fsync and cleanup failures. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| N06 | Image dimension/orientation/aggregate admission before patch allocation; fallible capacity. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| N07 | Controller learns full speculative cycle including commit; separate phase observations preserved. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| N08 | Target forward and trace export share rollback boundary. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| N09 | Public model operations own submission/completion; private pending encoders, checked host extents and compile-fail API gates. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| P01 | Opt-in compact BF16/affine-Q4 storage; optional Metal projection prototype with autorelease scopes and observed buffer peak. Other draft operations stay CPU. No speed claim. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| P02 | Pre-copy prefix admission and validated atomic in-place multi-plane ring commit. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| P03 | Explicit reference/optimized Qwen profile in native/CLI/Desktop, override precedence and effective diagnostics. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| P04 | Share immutable geometry-keyed GDN pipelines per owner, retain separate weights/state. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| P05 | Inherited PLE fix plus diagnostic draft traces omitted on non-trace execution. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| P06 | One prediction dot/top16 command instead of unused intermediate ranking. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| P07 | Opt-in exact sorting-network router candidate and independent all-E/all-k CPU oracle. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| P08 | Opt-in owned persistent converter worker, fixed scratch, strict IDs, atomic results and bounded cancellation/reap. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| D01 | Immutable release identity, beta ordering and artifact binding; conservative legacy handling. | CI-wired Swift fixtures; not executed locally. |
| D02 | Model readiness error no longer globally quarantines healthy engine. | CI-wired Swift fixtures; not executed locally. |
| D03 | Agent branches carry historical workspace checkpoint and policy. | CI-wired Swift fixtures; not executed locally. |
| D04 | Agent regenerate waits for exact MCP request acknowledgement once. | CI-wired Swift fixtures; not executed locally. |
| D05 | Apply preserves executable modes, detects conflicts and rolls back modes. | CI-wired Swift fixtures; not executed locally. |
| D06 | Ordinary sampling profile disables DFlash and gates pending MTP changes. | CI-wired Swift fixtures; not executed locally. |
| D07 | Native context/memory profile preserved through Rust→Swift. | CI-wired Swift fixtures; not executed locally. |
| D08 | Native memory/identity telemetry and native labels retained in UI. | CI-wired Swift fixtures; not executed locally. |
| D09 | Ticketed bounded tuner stop/reap fallback, stale callback guards. | CI-wired Swift fixtures; not executed locally. |
| D10 | Typed guarded quantizer JSON response handling. | CI-wired Swift fixtures; not executed locally. |
| U01 | Reusable actor document index, stable bounded top-k, async send and stale/cancel guards. | CI-wired Swift fixtures; not executed locally. |
| U02 | Mutation-driven snapshot/sidebar caches and filtered-search publication notification. | CI-wired Swift fixtures; not executed locally. |
| U03 | Detached runs plus incremental neutral-chunk lexical cache/state convergence, UTF16/revision/IME guards. Swift parity fixtures await CI. | CI-wired Swift fixtures; not executed locally. |
| U04 | UI resolver/hash work off MainActor with cancellation/tickets; fresh integrity verification retained. | CI-wired Swift fixtures; not executed locally. |
| U05 | Full tool output/copy, Reduce Motion, accurate preview label, send blockers, bounded draft/recency restoration. | CI-wired Swift fixtures; not executed locally. |
| T01 | Owned nonblocking MCP pipes and absolute read/write/cancel deadline. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| T02 | Incremental SSE framing and exact response ID; notifications handled independently. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| T03 | Aggregate response/header/stderr bounds before unlimited buffering. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| T04 | Lock protects MCP read-modify-write transaction and durability. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| T05 | API reports actual backend image support. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| T06 | Bounded canonical uniqueItems with exact wide integers and cancellation. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| T07 | Declared token-ID channel controls, special:false Qwen support, literal ordinary-token markers; explicit retokenizable context contract. | Portable component tests/review in reports; Apple/model qualification remains separate. |
| C01 | Typed-loop RGB fixture inherited from main, rendering criterion preserved. | desktop-preferences passed at33bb6ff; recheck expanded head. |
| C02 | Separate cleanup confirmation and EPERM-safe mandatory reap; unresolved descendants fail closed. | 57 focused tests; recheck expanded CI. |
| C03 | Structured failure diagnostics distinguish timeout, budget, OOM, ICE, unwinding and explicit property evidence; required harness inventory retained. | Classifier/inventory tests; actual proof verdicts remain CI-dependent. |

## Reports and evidence

- [native-reconstruction](issue163-native-reconstruction.md)
- [metal-recovery](issue163-metal-recovery.md)
- [buffer-api-reconstruction](issue163-buffer-api-reconstruction.md)
- [gpu-reconstruction](issue163-gpu-reconstruction.md)
- [draft-storage-reconstruction](issue163-draft-storage-reconstruction.md)
- [converter-reconstruction](issue163-converter-reconstruction.md)
- [desktop-runtime-reconstruction](issue163-desktop-runtime-reconstruction.md)
- [agent-reconstruction](issue163-agent-reconstruction.md)
- [responsiveness-reconstruction](issue163-responsiveness-reconstruction.md)
- [responsiveness-completion](issue163-responsiveness-completion.md)
- [protocol-reconstruction](issue163-protocol-reconstruction.md)

The earlier reconstruction reports retain negative trials and former partial states; completion supplements supersede those implementation gaps without erasing their test limits. C03 does not call an OOM/timeout a runtime counterexample. P01 retains CPU arithmetic and decodes compact projections; no draft-GPU port is claimed. P08 does not change matmul arithmetic. A full physical qualification campaign and 40 tok/s remain unestablished.

## Follow-up qualification

403 portable Rust tests passed, 10 ignored; 52 quantizer tests passed, 1 physical skip; 21 tests also pass with MLX preloaded. Eight inventory/manifest controls pass. New GPU admission, SSD chunk arithmetic and install identity predicates have bounded Kani results in their reports, not full system proofs. Physical GPU/model qualification and updated hosted Swift compilation remain required.

- [CI corrections](issue163-ci-followup.md)
- [GPU projection prototype](issue163-draft-gpu-followup.md)
- [Physical I/O cancellation limits](issue163-physical-io-cancellation.md)
- [Download crash recovery](issue163-download-recovery.md)
- [Incremental lexer](issue163-incremental-lexer.md)

## Functional re-review on 8080c96c

The following additional defects were reproduced and corrected separately from
CI maintenance. This does not close the outstanding physical qualification or
route-proof limits above.

- N03: terminal read, enqueue and timeout errors now abandon the batch before
  return; late completions cannot revive it. Producer-owned buffers survive until
  physical drain. See [native recovery](issue163-functional-native.md).
- N05: partial cleanup may delete the staging intent before the request. Recovery
  now uses the installed receipt only when the existing registry agrees on the
  exact request, path and installation identity. Retained IDs/paths and symlinks
  are checked before staging mutation. This is not race-free containment against
  hostile concurrent ancestor replacement. See [download recovery](issue163-functional-download.md).
- T06: recursive const/enum equality and unique-item hashing/confirmation consume
  an irreversible node/byte work budget. Hash collisions never determine equality.
  Parsing, final serialization and existing length scans are still bounded by
  input sizes rather than interruptible at each byte. See [structured validation](issue163-functional-structured.md).
