# MLXL3 Desktop v0.4.6

## Correct Metal memory accounting

- The menu-bar widget now reports the macOS **physical memory footprint** of
  the engine and interface, using `proc_pid_rusage / ri_phys_footprint` instead
  of RSS. GPU allocations are included: Qwen no longer appears to occupy only
  a few hundred megabytes when its Metal buffers use roughly 12 GB.
- Unavailable measurements display a dash instead of a misleading zero.
- The widget distinguishes model loading from the ready state.
- This is an accounting fix, not a reduction in memory use or a change to model
  weights, inference kernels or generation quality.

## Model library and refreshed interface

This release also ships the v0.4.5 work that had only been installed locally:

- Dark/ivory interface, serif headings, redesigned menu-bar panel and monochrome icons.
- Installed-model library: load, reveal in Finder, remove an entry or move a model
  folder to Trash after confirmation.
- Hugging Face EXL3 search, native Markdown model cards, branch/tag and variant
  selection with download size.
- Selective downloads pinned to a commit, progress, pause/resume and disk-space
  checks. Shared tokenizer files are inherited; ambiguous weight layouts are
  refused rather than mixed.
- Large catalogue responses no longer block the CLI output pipe.

## Installation

Use **Settings → Check for updates**, then restart when the download is ready;
or download the Apple Silicon DMG, open it and drag MLXL3 Desktop to Applications.
Requires an Apple Silicon Mac and macOS 26.2+. The engine/runtime are included;
model weights are downloaded separately. The app remains ad-hoc signed, not
Apple-notarized.

Validation includes the Python suite and packaged native checks, including a
64 MiB GPU-private Metal allocation that must increase the reported footprint.
