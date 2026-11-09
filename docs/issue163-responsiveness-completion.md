# U02–U04 completion supplement, 2026-11-09

This supplements, rather than erases, the earlier partial reconstruction report.
Preregistered in opti.md as PR163-U-COMPLETE-20261109 before changes and checks.
Baseline: current shared PR163 recovery checkout; concurrent runtime/Agent/native
changes preserved. Actual issue162 U02–U04 read through the GitHub connector.

## Implementation and contracts

- U02: ChatMessage publishes after each existing streamRevision mutation, including
  sources and turnContext. Conversation membership changes create a fresh observer
  cache. Value copies retain independent membership after mutation, while shared
  message references notify every observing cache. Weak subscriptions avoid cycles.
  Snapshot access no longer maps identities/revisions or searches every message:
  only dirty and currently streaming indices rebuild their immutable snapshots.
  Streaming processing elapsed times remain refreshed. Array copy-on-write may
  still copy the snapshot pointer/value array when a previous snapshot is retained;
  this is not an append-only journal or a claim of constant-time serialization.
- Sidebar filtering/sorting is cached by query, Agent mode and project. Conversation
  structural/metadata mutations and message revisions invalidate it. Repeated body
  renders use the shared result without rescanning message contents. Mutating the
  conversation list rebuilds conversation subscriptions, not message signatures on
  every render. The existing validated active-message reference and single-read
  history backup from the previous reconstruction are preserved.
- U03: the editor's bounded lexer emits Sendable UTF-16 color runs in a detached
  task; its lexical definitions remain shared with the existing renderer. AppKit
  colors/fonts/storage stay on MainActor. Document reset, edits and styling changes
  cancel old jobs and change an identity ticket. Results require a matching ticket,
  unchanged text and no marked text. Attribute application preserves characters,
  selection and undo, and skips unchanged runs. The editor no longer reconstructs
  a whole attributed string. Multiline lexical context is retained by lexing the
  bounded whole source off-main rather than inventing incomplete line-state rules.
  The 256 KiB bound remains; larger files use plain styling of changed ranges and
  incremental line offsets. Full initial document/font styling remains necessary.
- U04: LocalServer and quantization conversion now resolve on detached workers.
  Server resolution is visibly active/busy, Stop invalidates pending launch, and
  stale jobs cannot launch a process. Quantization rechecks operation/cancellation
  after resolution. CLICommand was already on a background DispatchQueue; bridge
  and updater background verification from the previous reconstruction remain.
  No hash cache is added, and no integrity check is removed. Issue162 explicitly
  permits moving IO off MainActor; its cache conditions apply only if one is added.

## Verification and explicit limits

- `git diff --check`: passed after edits.
- `zsh -n scripts/check-desktop.sh`: passed.
- `swiftc --version`: command not found (127). Compilation, Swift behavioral
  fixtures, Swift mutants, concurrency diagnostics and AppKit integration are
  **not executed**, not passed. Formal-proof-skill and Swift reference are absent
  from the available skill directories/catalog. No formal proof claimed.
- Extended tests/responsiveness-check.swift, already in the real desktop CI gate:
  snapshot copy membership and multiple observers; detached lexer versus existing
  highlighter for Swift/Python/HTML/JavaScript, multiline strings/comments and
  surrogate pairs; full UTF-16 coverage; stale document results; 200/256 KiB and
  2 MiB text/selection, undo and IME; server Stop during resolution and invalid
  port recovery. Styling tests now await the asynchronous production operation.
- Extended desktop-experience-check.swift with identical-query invalidation after
  streaming, message replacement and subsequent mutation of the replacement.
- No new workflow is needed: existing check-desktop/check-e2e is already wired to
  hosted macOS on PR/main. Final exact-head CI must be inspected after publication.
  These fixtures are executable gates, not evidence that they passed locally.
- No user Mac, speed benchmark, physical latency/VoiceOver/visual verification,
  commit, push, release, merge or installed-app change performed by this worker.

Quality: implemented source, macOS gates pending. Performance: unmeasured.
Integration: shared local checkout only. No experiment remains running.

## Independent-review correction: sidebar publication

PR163-U02-NOTIFY-20261109 records a concrete gap in the first cache implementation:
clearing the cached result alone does not notify SwiftUI when an excluded row gains
matching streamed text. The subscriber now publishes StudioModel.objectWillChange
for a nonempty cached search in the active mode/project and coalesces subsequent
mutations until the next render. Empty searches retain their rows without scans or
extra notifications; metadata/membership mutations still invalidate through the
conversation property observer. Subscriptions capture scalar scope values rather
than the conversation/cache, preserving weak ownership and avoiding a cycle.

The experience fixture now observes the production publisher before changing an
excluded message, asserts exactly one notification, checks coalescing, then checks
that an empty search does not publish. It also verifies the filtered result.
`git diff --check` passed after the correction. Swift fixtures and a Swift mutation
check remain unrun because swiftc is unavailable; no macOS validation claimed.
