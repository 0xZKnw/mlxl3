# Desktop runtime reconstruction (PR163)

## Preregistered scope, 2026-11-09
Baseline f428945b (published tree 2ddde23), main0fd9fbd1 includes PR161.
Authoritative issue https://github.com/0xZKnw/mlxl3/issues/162 fetched directly.
Reconstruct D01/D02 and D06–D10 against current source; lost prior work/tests are
not validation of this code. Keep download phase work and Agent edits.

Plan: release identity and immutable manifest-bound directories; legacy pointer
migration; executable/protocol-only quarantine; ordinary profiles disable both
speculative modes; native memory/context aliases and truthful metrics; ticketed
Tune cancellation fallback; guarded quantizer response. Preserve signatures and
hashes. Production Swift regression fixtures, packaging Python suite and mutation
where practical; compiler/formal tool availability checked explicitly. No Mac,
GPU, performance benchmark, push or merge. Parent owns shared journal.

## Implemented

- D01: numeric runtime protocol version stays separate from full release identity.
  Beta ordering, immutable tag/archive-digest directories, pointer-bound candidate
  manifest digest, explicit legacy unknown provenance, atomic next-install
  migration, downgrade refusal and refreshed active identity. Packaging accepts
  canonical release tags, emits matching beta asset names, checks them during both
  hash passes, and preserves tags in bundle metadata. Historical bundles without
  release provenance cannot reliably be identified as beta retroactively.
- D02: only explicit protocol mismatch or classified executable launch failure
  disables managed binaries. Model/context/resource/unknown pre-ready exits stay
  local; wrapped POSIX resource launch failures also avoid quarantine. One launch
  retry; disable marker bounds protocol fallback. Runtime-root injection allows
  tests to use isolated stores, never installed engines.
- D06: ordinary profiles disable MTP and DFlash. Sending also waits for MTP
  configuration acknowledgement, matching the existing DFlash pending guard.
- D07/D08: native context-memory alias, optional legacy counters, native allocated
  Metal bytes, backend/runtime/Qwen options retained; display and Lab distinguish
  allocation from MLX peak and use unknown rather than invented zero.
- D09: request/load-scoped 3s Tune cancellation fallback stops and joins the bridge.
  Completion cancels the timer, old tickets cannot stop a new load/Tune, saved
  tuning selection is not overwritten. Unloading resets current UI configuration
  normally; this is not erasure of the stored selection.
- D10: malformed/non-object quantizer probe JSON becomes a caught error, not a trap.

## Checks and remaining validation

- Packaging/build-manifest Python suite: **75 passed** (final run 3.89s).
- Mutation: force numeric-only archive filename, new release archive test fails
  with exit1; restored production and reran the 75-test suite successfully.
- Exact disposable shell used by the new Swift harness: six Linux scenarios pass
  (three model/resource exits, ready, mismatched protocol, ignored SIGUSR1), all
  children reaped. This checks fixture behavior, **not Swift production logic**.
- Added CI-wired `issue163-runtime-contract-check.swift`: real store and bridge
  transitions beta1/beta2/stable, legacy migration, idempotence, same-digest content
  mismatch, manifest/bytes replacement, model-A failure/model-B success, bad
  protocol quarantine, native/legacy telemetry, old cancellation ticket and
  bounded Tune recovery. It enters the post-signature commit boundary with a shell
  fixture; existing signed-archive updater integration remains required.
- Existing speculative-sampling harness now checks actual requests after applying
  a profile with MTP/DFlash active, including pending acknowledgement. Existing
  quantizer worker test now emits array/incomplete/invalid JSON before a good probe.
- `swiftc --version`: command not found (127). Swift compilation, these Swift
  regressions and visual macOS checks are **unexecuted**, not passing. No Mac used.
- formal-proof-skill unavailable; CrossHair attempt fails `No module named crosshair`.
  No formal proof claimed. No GPU/model/performance measurements or integration
  into an installed app. Diff whitespace check passes; parent owns shared journal,
  independent review and exact-head hosted CI. Full issue162 remains broader.

Conservative legacy migration: numeric-only historical pointers may represent a
stable or beta. Same-base beta updates are therefore not offered over unknown
legacy provenance; stable/new-base updates migrate normally. Legacy directories
never mask an equal-base bundled runtime. New fully identified beta1→beta2→stable
transitions work without this ambiguity. This prevents a historical stable from
being silently downgraded to beta; no lost provenance is invented.

Shell validation: first `bash -n scripts/build-engine.sh` failed because that
script uses zsh glob qualifiers (wrong validator, not a new source error).
Correct `zsh -n` for build-engine.sh and check-desktop.sh both pass.

Final packaging/bundle extension: added actual build-info generation tests for
both Metal and MLX prerelease manifests. Final combined suite: **77 passed in
4.53s** (supersedes the 75-test subset above). Both zsh scripts and final diff
whitespace checks pass. New Swift harness remains unexecuted. Disposable
pytest-of-agent output is not part of the deliverable.

## P03 explicit Qwen profile selection

Prerequested extension uses MLXL3_NATIVE_QWEN_PROFILE=reference|optimized. Desktop
preference defaults to reference; unknown persisted choices normalize to reference.
Bridge fills only an absent profile variable and leaves every individual/external
flag unchanged. Settings display requested profile separately from runtime-reported
qwen_profile and effective options, with reload requirement and experimental/no-gain
warning. Missing ready field remains unknown, never inferred from the preference.
Runtime identity/Lab retain the engine report. Production environment-builder
regressions cover default, invalid preference, optimized opt-in and external
profile/individual-flag preservation. Swift tests remain unexecuted on Linux.

P03 native contract confirmed with GPU worker: ready `qwen_profile` plus boolean
`qwen_effective_flags` (moe_join, layer_join, layer_chain, device_kv,
prefill_last_head, group_expert_qmv, group_expert_stages, device_greedy).
Desktop retains the whole sorted dictionary and falls back to historical separate
fields only when that dictionary is absent. Two updated shell-fixture profile
reports pass on Linux; these are not Swift/native integration passes. Reference
remains the default; no optimized profile or greedy path is automatically enabled.
