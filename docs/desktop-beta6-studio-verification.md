# Desktop beta 6 Studio verification

Status: source and signed packages qualified; Desktop beta6 and engine2.0.2-beta.1 published.

The working branch starts at `f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6`. Artifacts Studio and its independent revision workflow are removed. Agent Studio uses a full-window project controller; Chat and Agent share native completed-code cards and an on-demand isolated HTML preview. Saved artifact directories are not deleted.

## Checked production paths

- Real Git import, checkpoints, selected historical file inventories, binary/deleted-file states, UTF-8 code edits, dirty navigation, save conflicts and permission changes.
- Apply requires the exact reviewed commit and unchanged working data, then copies reviewed bytes. Original-project conflicts are checked before any writes.
- Git attributes cannot transform checkpoint bytes or hide ZIP entries. A CRLF/Unicode fixture is checked against the committed data and a ZIP extracted with macOS ditto. Scoped diffs treat `[a].txt` literally rather than selecting `a.txt`.
- Native full-window integration, Code/Diff/History pages, syntax colors, selected-file races and clipped line-number gutters.
- Completed code preserves exact Unicode and newlines in a selectable, read-only native text view. HTML controls remain unavailable through streaming, subsequent tool rounds and interrupted replies. No WebKit is created by merely displaying code.
- Existing transcript verification covers 81 native viewport frames across long prose, growing code and completion, scroll anchors/interpolation and 63 parser/chunk cases. It also exercises the fake bridge, full copy and persisted/reopened history. This does not reproduce or universally resolve the earlier intermittent disappearing-text report.

## Results and commands

All logs are in `docs/measurements/studio-workspace-fix/`.

| Check | Result | Evidence |
|---|---|---|
| Swift build, SDK 26.5 | Passed; inherited linker search-path warnings | inline-build.log/json |
| Native Agent/controller/Git journey | 44 passed, no skipped cases | agent-final2.log/json |
| Existing navigation/lifecycle checks | 1174 passed | affected-agent-inline2.log/json |
| Existing project/permission checks | 39 passed; two removed artifact-history assertions no longer apply | affected-agent-inline2.log/json |
| Full response rendering journey | Passed: 81 frames, 63 parser/chunk cases, 9 formulas in 2 styles, lifecycle/persistence | response-inline2.log/json |
| Format/lint/syntax | swift-format strict on new/rewritten Swift files; zsh -n and git diff --check passed | final checks recorded during release qualification |
| Covered preview lifetime | Passed; real CPU timer starts, covered view removed, previous document cleared | preview-covered-final2.log/json |

The native workspace journey is reproducible through `MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk scripts/check-studio-workspace.sh`. The response and navigation compile/run recipes are the affected sections of `scripts/check-desktop.sh`; tests use the repository fake engines, never a model. Bounded local supervisors retained their command, exit code and deadline in the adjacent JSON receipts.

## Retained failures

The original sheet was captured and its unreviewed-apply bug reproduced before the shared store fix. Earlier candidates also retain: a fixed 800 ms UI wait with no content diagnostic; a Unicode filename mismatch in the unzip oracle (independent zipfile inspection found exact data, so the oracle now extracts with ditto); a test diagnostic warning promoted to an error; empty internal accessibility trees despite a rendered Copy control; and a parallel SwiftPM rebuild changing SwiftMath.o while another harness compiled. The final checks use bounded content waits, native/pixel oracles and sequential builds. None of these failed runs is relabeled a pass.

## Verification scope

These are deterministic integration tests and finite native rendering checks, not a formal proof. No maintained source verifier for this SwiftUI/AppKit implementation is configured; source-level Swift properties remain unproved. No behavioral Rust implementation is changed by the UI work. The engine integration carries main's existing qualifications and their original source identities separately.

No LLM inference, quantization, Metal kernel or WebGL test was run during this preparation. The covered-preview check uses only a disposable HTML page with a CPU JavaScript counter. External CDN execution and physical numerical behavior of the rebuilt runtime remain unverified in this release run. The user's installed app and personal dirty checkout remain untouched.

## Long response correction and final qualification

The strengthened baseline reproduced a persistent missing code tail when a growing block crossed into its internal scroller. C1 gives that scroller a native initial/size anchor, measures its chunks consistently and preserves following until a user gesture suspends it. Thinking uses raw bounded chunks; stable/large text uses native drawing, while the small active suffix keeps the fade. The user confirmed the disappearance did not occur during C1.

The optimized native-bitmap journey passes 30 exact marker/OCR captures across 131,226 bytes of thinking and 262,197 bytes of answer, code, tools, completion and two window sizes. The final optimized bitmap run had zero initial misses and recorded 30 physically visible captures, while its declared domain remains native bitmap/OCR. The earlier C1 run had five initial prose misses resolved after the recorded 2-second settling check; its evidence is retained separately. Optimized physical-compositor attempts remained unavailable because macOS kept the test app inactive/occluded; they remain failed/unverified evidence, not passes. `--require-onscreen` retains those guards. Manual nonoptimized C1 had 30 visible captures and the user's observation.

Final CPU source checks: 141 raw-cache boundary/Unicode cases, 1,194 incremental code cases, 4,845 incremental Markdown cases, 1,179 mode/permission checks, 39 project checks and 44 workspace checks pass. See `desktop-bitmap-final.log/json`. The optimized ordinary response checks from `desktop-final4.log` passed their 786 fade cases, 63 drawing comparisons, 81 viewport frames, HTML lifecycle/Unicode source checks, 9 formulas/2 styles and 63 parser/chunk cases before the separate foreground-availability failure. The production rendering code is unchanged between those checks and the final bitmap run.

Native engine: 284 CPU tests passed, 9 ignored; fmt and strict release Clippy with shipped `mlx,chat` features passed. Kani0.68/CBMC6.11 checks actual pre-draft cost observation for symbolic counters and arguments, no assumptions/stubs, unwind5: 380 obligations pass, 6 internal ZST/fmt checks are unreachable, all4 business covers are satisfied. This is bounded CPU accounting, not proof of readers/concurrency/Metal/inference. The 129 native source/build files match qualified main4c1d byte for byte; Cargo changes only version2.0.1→2.0.2. Main e966 adds profile-report compatibility without changing those algorithms.

The fixture fixes its NSHostingView bounds explicitly because [Apple documents its default window sizing constraints](https://developer.apple.com/documentation/swiftui/nshostingview/sizingoptions). No source-level Swift verifier or fresh GPU/model/quantization/WebGL qualification is claimed.

## Publication

[Desktop beta6/build37](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.0-beta.6) and [engine2.0.2-beta.1](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.2-beta.1) are published as prereleases with latest=false. All eight uploaded asset sizes and GitHub SHA-256 digests match the local artifacts; release bodies match the English notes. The production UpdateManager accepts the anonymous public feed and selects beta6/build37 plus engine2.0.2-beta.1; the Stable channel remains v1.5.1. See `docs/measurements/desktop-v2-beta6/publication-verification.json` and `public-feed-preflight.log`.

The exact-source publication CI snapshot has Profile/Conversion successful, Regression/Desktop and Native running, and Apple compatibility cancelled. Publication and merge without waiting for CI are explicitly authorized; no overall-green claim. Subsequent receipt-only commits do not change the frozen binary source or qualify unexecuted GPU/model paths.

## Frozen release package

Clean source `9df28b6d2a10e0d646f8a36532ae69c88c9ec3e8` produced Desktop2.0.0/build37 and engine2.0.2. Build170.58s; production DMG preflight36→37 passed6.15s and standalone engine install/activate/resolve2.0.1→2.0.2 passed3.13s in disposable storage. Wrong archive digest is rejected without activation; signatures/arm64/protocol/four runtime hashes/source agree, empty registries list0models, temporary roots/mounts are removed. Bundled and standalone runtime manifests are identical. No installed app was replaced. The exact-source CI has profile/conversion successful and other workflows running at publication preparation; no overall-green claim.
