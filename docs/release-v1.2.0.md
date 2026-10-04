# MLXL3 Desktop v1.2.0 — M3, native MTP and independent engine updates

- New full-canvas graphite/platinum **M3** icon, sidebar monogram and menu bar mark.
- Whole-message copy buttons moved below user and assistant messages.
- Native Qwen3.5-family MTP predictor, including Qwen3.6-35B-A3B. One proposal per block, greedy-only in v1; every delivered token is checked against the target. Other sampling settings use ordinary decoding.
- Matching Qwen3.6 affine4/group64 head downloaded on demand from a pinned Hugging Face revision; size, SHA-256, tensor inventory and layout checked before use. Other layouts require their matching head. DFlash is disabled at Desktop relaunch and retained as a CLI experiment.
- Conversation prefix cache retains both target state and MTP KV, without capturing a token outside the reusable prefix.
- Independent engine archives from GitHub, stored outside the signed app. Integrity, protocol, minimum OS/app, architecture and signatures checked before atomic activation. A managed engine that fails before ready falls back to the bundled runtime.
- App updates verify the candidate before replacing the installed app, retain the previous version, and restore it on a caught installation error. App and engine channels can update together.
- Fixed large EXL3 prefill selecting M5 TensorOps on earlier GPUs. Portable path verified against independent rows with M5 TensorOps disabled.

## Requirements and scope

Apple Silicon, macOS **26.2 or newer**. Physical validation: M5 / 24 GB; no physical M1–M4 campaign. The DMG is self-contained, ad-hoc signed, and contains no model weights. It is not Developer ID notarized.

Native MTP target: Qwen3.6-35B-A3B EXL3 2.49 bpw with matching MLX MTP4. Dense Qwen MTP loading has structural tests but no physical dense-model run. Full GPU/state tests and formal CPU contracts do not prove the entire application free of bugs.

[Verification, commands and limits](desktop-v1.2.0-validation.md).
