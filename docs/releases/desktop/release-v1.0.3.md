# MLXL3 Desktop v1.0.3 — Ling 3.0 MCP compatibility

This is the release that makes Ling 3.0 Tiny's native tool-call format work
with MCP in MLXL3 Desktop. Ling could already generate text in v1.0.2, but its
`<tool_call>` output was not understood by the Rust bridge, so Exa and other
configured MCP tools were not called.

- The Rust bridge now parses Ling's `<arg_key>` / `<arg_value>` arguments,
  including JSON numbers, booleans and objects, while retaining the existing
  Gemma, Qwen and LFM tool-call formats.
- Malformed, duplicate or mixed prose/tool payloads remain non-executable.
- No model weights or quantization recipe changed in this release.

The DMG includes the native SwiftUI app, Rust/Metal engine and MLX runtime,
but not model weights. It requires Apple Silicon and macOS 26.2 or later.

**Signing limitation:** the app is ad-hoc signed, not Apple Developer ID signed
or notarized. macOS may require explicit approval in Privacy & Security. Do not
disable Gatekeeper globally; keep the previous app for rollback.
