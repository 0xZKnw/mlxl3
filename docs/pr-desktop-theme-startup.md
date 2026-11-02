# Add Desktop themes and optional model loading on startup

Adds persisted System, Light and Dark themes throughout the Desktop interface,
including settings, menu panel, code highlighting and formula tinting. Appearance
changes keep the current draft, conversation and engine alive.

Adds “Load model on startup” in Settings. When off, startup lists available models
but waits for manual loading; when on, it restores the last selected model or
falls back to the first available one. Changes apply to the next launch without
interrupting an active model. Existing dark/auto-load defaults are preserved.

Validation: full Desktop E2E suite, targeted isolated preferences/startup tests,
rendered Light/Dark/System checks, production-source mutation control and release
build. Formal Swift verification is unavailable; model/GPU inference unchanged.
See [verification and limitations](desktop-theme-startup-20261102.md).

Base branch: `codex/qwen27-lossless-ram` at `72f4744`. This PR contains only Desktop
preferences and tests, with no update of the historical base to upstream main.
No app installation or release included. Pre-existing local disk-cleanup notes
in opti.md are preserved outside this commit.
