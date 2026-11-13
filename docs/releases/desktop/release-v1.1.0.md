# MLXL3 Desktop v1.1.0 — DFlash2 for Qwen3.6-35B-A3B

This release adds **experimental, opt-in DFlash2 speculative decoding** to
MLXL3 Desktop for Qwen3.6-35B-A3B EXL3. The native Rust/Metal engine proposes
tokens with Inco AI's draft and verifies them against the loaded target before
streaming them to the chat. The ordinary decoder remains the default.

In **Generation → DFlash2**, choose the folder containing the separate
`draft/` weights and apply the greedy preset. DFlash2 currently requires greedy
sampling (temperature 0 or Top K 1, repetition penalty 1.0). Leave DFlash2 off
for other models or sampling settings; incompatible requests fail visibly rather
than silently changing the chosen sampling behavior.

The previously reported DFlash2 speedups were physical M5 benchmark results,
not promises for Desktop, other prompts or other Macs. The DMG includes the
SwiftUI app, Rust/Metal engine and MLX runtime, **not** the Qwen target or draft
weights. The first speculative request can be slower while Metal compiles its
kernels. See the [README](https://github.com/0xZKnw/mlxl3#readme) for setup and limitations.

**Signing limitation:** this build is ad-hoc signed, not Apple Developer ID
signed or notarized. macOS may require explicit approval in Privacy & Security.
Do not disable Gatekeeper globally; keep the previous app for rollback.
