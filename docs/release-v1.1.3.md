# MLXL3 Desktop v1.1.3 — copy messages and smoother streaming

Each message now has a **Copy** button. It copies the complete answer,
including paragraphs, Markdown, formulas and code blocks. It also works
while the model is responding or after stopping generation. Thinking and
internal tool output stay out of the copied answer.

Markdown preparation now runs away from the UI thread. Stable chunks are
reused, and rapid updates keep one preparation running with only the latest
pending text. Replacing a response or leaving the view prevents obsolete
preparations from overwriting its contents. The final response and saved
history retain every received fragment.

The existing token fade, thinking indicator, syntax colors, typography,
tables, math rendering and scrolling controls are preserved.

Build 18 targets Apple Silicon and macOS 26.2 or newer. The DMG includes the
SwiftUI app, Rust/Metal engine and MLX runtime. Signing remains ad-hoc,
without Apple notarization. See `docs/desktop-v1.1.3-validation.md` for the
measurements, test coverage and limits of verification.
