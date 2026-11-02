# Desktop preferences, notifications and model startup recovery

Adds persisted System/Light/Dark/Midnight blue themes, optional automatic loading and an explicit
startup-model picker. A missing chosen model stays unloaded rather than silently
loading another one. Appearance changes keep the draft, conversation and engine.

Adds opt-in completion notifications, with macOS permission requested on enable.
Only successful background responses notify; content is generic and contains no
conversation text. Cancellation, failures, foreground and duplicate completions
do not notify. Notification errors do not interrupt the engine.

Failed loads can now retry the same model directly, with visible error details
and a Retry button. Loading tasks are cancelled/identified and context is captured
with the selected model so superseded attempts cannot overwrite newer state.
The reported intermittent Flash Next first-load incident was not reproduced:
real direct loads (4096/automatic context) and actual Desktop startup pass. This
PR fixes the independently reproduced retry defect, without claiming to identify
the original intermittent engine failure.

Validation: full Desktop suite, 55 preference and 14 notification/retry assertions,
358 Python tests passed/4 historical skips, original retry-guard mutation detected,
physical Flash Next startup, release bundle signature and runtime hashes.
Formal Swift verification unavailable; CrossHair missing. No numeric/model changes.

Base: `codex/qwen27-lossless-ram` (`72f4744`), including the previous themes commit.
No merge of newer upstream main, installed app replacement or release.
Pre-existing disk-cleanup notes in opti.md remain outside the feature commits.

To test, close the current Desktop and run `./scripts/build-desktop-test.sh --launch`.
The local bundle uses the verified installed engine2.0.2 that supports Flash Next,
with the complete Desktop changes from this branch.

[Preferences](desktop-theme-startup-20261102.md) · [follow-up verification](desktop-notifications-flash-startup-20261102.md).
