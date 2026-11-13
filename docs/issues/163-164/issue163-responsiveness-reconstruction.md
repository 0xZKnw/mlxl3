# PR163 responsiveness reconstruction (U01–U05)

## Preregistered protocol, 2026-11-09
Baseline: published 2ddde23 plus preserved concurrent reconstruction. The previous
unpublished implementation/logs are lost and do not validate this source. Read
issue162/annex, opti.md and recovery reports before edits. No speed benchmark.

Implement reusable document passage/term indexing outside MainActor, deterministic
bounded top-k and stale-result guards at send. Preserve citations, PDF pages,
ordering and limits. Reduce active-message scans, share sidebar filtering, reuse
immutable snapshots safely, read persisted history once with corrupt/future-schema
refusal. Editor: attributes only, marked-text protection, bounded work, UTF-16.
U05: full tool output/copy, live Reduce Motion, honest preview network wording,
visible send blockers. Coordinate StudioModel/Domain/runtime ownership.

Parent records protocol in opti before tests. Add actual production Swift harness
cases and negatives; attempt tool availability/source checks. Formal-proof-skill
and its language reference unavailable. No proof claimed. Linux cannot validate
AppKit, IME, VoiceOver, Metal, physical UI latency; no Mac access or performance
claims. Hosted macOS results require exact final commit inspection.

## Reconstructed source and limits

- U01: DocumentSearchWorker owns per-document immutable passage/term entries.
  Content and metadata changes rebuild an entry; removed entries are discarded.
  Search retains at most eight ranked results, preserving score/tie/source order.
  StudioModel send waits asynchronously with an explicit pending state and Stop,
  validates conversation/project, document revision, draft/attachments/model/mode,
  and retains the Agent checkpoint immediately before actual send. Old tickets
  cannot publish after cancellation/navigation. No vector service or new dependency.
- U02: active response reference avoids per-event global scans; deletion's existing
  force-stop clears request identity. Sidebar count/list share one filtered result
  per render. Conversation snapshots reuse immutable message arrays by object
  identity/revision, rebuilding streaming arrays for live elapsed time. Sources and
  turnContext invalidate revisions. Signature scans still exist, as do sidebar
  rescans across renders: this is not a fully revision-indexed history store.
  ConversationStore validates and backs up the same prior bytes from one read,
  retaining corruption/future-schema protection and writer revision serialization.
- U03: recoloring applies attributes, never replaces characters; marked text is
  left alone. User edits update UTF-16 line offsets from replacement ranges rather
  than rescanning characters. Large-file subsequent plain styling covers changed
  ranges; initial styling/font/document changes remain full. Existing <=256 KiB
  lexer is still synchronous/full-document; larger diff files now remain plain
  rather than running an unbounded diff coloring pass. Full incremental lexical
  state/off-main colors and measured latency remain follow-ups, not claimed solved.
- U04: bridge launch now resolves and verifies in a detached task, then checks
  cancellation/generation ticket before starting. Updater off-main identity belongs
  to the runtime reconstruction. No integrity check was removed or cache added;
  other synchronous CLIResolver callers and repeated total hashing remain follow-ups.
- U05: bounded scrolling full tool output with exact copy; Reduce Motion responds
  to preference changes; tool/preview labels disclose possible remote imports;
  composer displays current blockers. updatedAt is optional/backward compatible,
  set at completed turn, sorting unpinned chats without per-token jumps; legacy
  unset dates preserve insertion order. Exact unsent draft/embedded attachments
  restore without rereading source files, clear on send, and use existing deferred
  persistence. Bounds: 64 KiB draft text/conversation, eight attachments, 128 drafts,
  16 MiB aggregate. Oversized drafts remain in memory, are excluded (not truncated)
  from disk with a visible storage warning; ordinary history still saves.

## Validation performed and not performed

- `swiftc --version`: exit 127, command not found. Swift compilation, production
  regression execution and Swift mutant execution are **not run**, not passed.
- `git diff --check`: passed after final edits; concurrent worker changes preserved.
- `zsh -n scripts/check-desktop.sh`: passed.
- Four structural assertions passed: no editor textStorage character replacement,
  no synchronous document search in StudioModel, bounded ranker without global
  candidate sorting, new harness registered. These are not behavioral tests.
- Added tests/responsiveness-check.swift to actual macOS desktop-and-protocol CI
  through check-desktop/check-e2e (PR and main pushes already configured): ranking,
  cache reuse/invalidation/removal, PDF/Unicode, negative limits, generated UTF-16
  edits against full rebuild, immutable snapshots and replaced object identity,
  draft restore/orphans/bounds/corrupt old history/exact backup, 200/256 KiB and
  2 MiB editor text/selection, undo and marked text. Existing workspace integration
  adds cancellation and stale-draft rejection across the real engine fixture and
  awaits document preparation; experience harness covers draft restore/recency.
- Hosted exact-head results are pending parent inspection. AppKit visual behavior,
  accessibility, IME, crash timing and performance remain unverified without macOS.
  No actual Swift checks or formal obligations are reported as passed. No speed
  measurements, Mac use, commit, push, merge, release or installed app changes.

Quality: source implemented, local structural checks only, Swift gates pending.
Performance: unmeasured. Integration: local checkout, not independently published.
