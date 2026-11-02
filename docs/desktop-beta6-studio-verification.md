# Desktop beta 6 Studio verification

Status: UI candidate verified on CPU; release integration and final package qualification in progress.

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
