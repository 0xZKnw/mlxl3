# Add pinned chats, reading width and remembered per-model MTP

The Desktop now offers Centered/Wide conversation layouts without changing
font size or losing drafts, and pinned chats that stay above recent chats
across restarts and history exports.

MTP has a toolbar control and a shared settings panel. New models default to
OFF; opting in persists for that model across reloads and automatic startup.
Errors preserve the user's intent and expose retry, while opting out stops
automatic preparation. Existing calibration isolation remains unchanged.
Flash Next uses its integrated checkpoint head instead of the incompatible
external-head installer in engine 2.0.2.

Validation and limitations are recorded in
[`desktop-experience-20261102.md`](desktop-experience-20261102.md), with logs,
source identities and screenshots. No numerical kernels, quantization, KV
precision or memory-saving policy changed. No performance gain is claimed.
The local test app retains the verified installed engine 2.0.2.

Create this PR against `codex/desktop-theme-startup` in `HENK0O/mlxl3` to review
only these additions; the branch includes the earlier Desktop settings work.
Base SHA: `bacee9d99f6fc1251b45a42172d117e3e4a13c54`.
