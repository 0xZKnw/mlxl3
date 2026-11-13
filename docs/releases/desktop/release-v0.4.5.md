# MLXL3 Desktop v0.4.5

- Dark/ivory model library with serif headings, matching the conversation view.
- Installed models: load, reveal in Finder, unregister or move to Trash after confirmation.
- Hugging Face EXL3 search with debouncing, cancellation and session caching.
- Native Markdown model cards; branch/tag and folder/quantification selection with download size.
- Selective downloads pinned to a commit, progress, pause/resume, disk-space checks and per-variant locking. Shared tokenizer/config files are inherited; ambiguous checkpoints are refused.
- Redesigned menu-bar panel: engine/UI memory, active model, context, generation metrics, unload/open/quit controls. New monochrome menu-bar and application icons.
- Short-lived CLI commands drain output before waiting, preventing large model cards from blocking the pipe.

No inference kernels or generation settings are changed. HF search uses at most
60 results; refine a query or paste an exact repository. Public models need no
login; gated/private repositories use the local HF account. Cards do not run
HTML/scripts or load remote badges. Some unusual repository layouts may need
manual folder import. Models are not included in the app bundle.

Checks: Python unit suite, native chat/preferences checks, CLI transport checks
(`tests/cli-command-check.swift`), and native visual QA with live Hub metadata.
Selective transfer/resume tests use isolated tiny fixtures, not multi-GB downloads.
