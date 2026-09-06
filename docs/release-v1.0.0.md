# MLXL3 Desktop v1.0.0

Reliability and polish across the Metal engine, terminal chat and native Mac app.

- Protected chat saves, recovery/export/import, reliable cancellation and engine lifecycle.
- Safer MCP parsing, schema validation, bounded results and preserved tool context.
- Checkpoint/shard validation, cache recovery and model-memory cleanup.
- Smoother thinking/answer streaming, long-code layout, improved Markdown/math and per-chat drafts.
- Resumable variant-only HF downloads, folder relocation and HF sign-in in Settings.
- Build-aware updates, retained app backup and bundled dependency/source manifest.

The DMG includes the engine and its dependencies, **not model weights**.
Requires **Apple Silicon, macOS 26.2+**. CLI and Desktop share the same engine.

**Distribution limitation:** ad-hoc signed, **not Apple Developer ID signed or
notarized**, as in earlier releases. macOS may require explicit approval in
Privacy & Security. Do not disable Gatekeeper globally.

See [validation, compatibility and remaining limits](https://github.com/0xZKnw/mlxl3/blob/v1.0.0/docs/v1-validation.md).
No universal model/device certification or new performance gain is claimed.
