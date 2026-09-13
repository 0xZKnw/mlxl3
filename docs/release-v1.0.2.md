# MLXL3 Desktop v1.0.2

- Much faster native EXL3 checkpoint loading, especially for large MoE models.
- Batched TensorOps prefill for dense LFM2 models and Gemma 4.
- Keeps the previous serial path for LFM2 MoE and for Gemma contexts beyond its
  validated sliding-attention batch window.
- Includes the current Rust/Metal engine directly in the SwiftUI app and CLI.

On the tested M5, the fixed LFM2 dense workloads improved from 92 to 2,060 tok/s
and from 47 to 837 tok/s prefill, with identical greedy output hashes. Gemma 4
improved from 40 to 142 tok/s on the 82-token workload and from 36 to 127 tok/s
on 565 tokens, also with identical hashes. These measurements are specific to
the listed models, prompts and Mac; no decode improvement is claimed.

The DMG includes the native app, Rust engine and MLX runtime, **not model
weights**. It requires Apple Silicon and macOS 26.2 or later.

**Signing limitation:** the app is ad-hoc signed, not Apple Developer ID signed
or notarized. macOS may require explicit approval in Privacy & Security. Do not
disable Gatekeeper globally; keep the previous app for rollback.
