# MLXL3 Desktop 2.0.30

**Redesigned pages across the app, an image history, a two-model comparison in Lab and a storage review. In Agent mode, a file's content now travels beside the code instead of inside it.**

Desktop **2.0.30 / build 83** bundles MLXL3 engine **2.0.34**: the same files as [`engine-v2.0.34`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.34).

## New

- **Redesigned pages** ([#201](https://github.com/0xZKnw/mlxl3/pull/201)). Every full-window page and standalone mode, except the home screen (redesigned in 2.0.26), now uses the same components: sidebar rows, Back as the sidebar's bottom row, cards, setting rows, buttons, notices and empty states.
  - **Lab**: a daily activity chart, the change against the previous period, per-model share bars, generation rows that expand into their measurements, and the memory MLXL3 uses on this Mac.
  - **Agent Studio**: a compact task bar gives the editor more height, the file tree starts expanded, small text is larger, and versions are a list.
  - **Settings, Server, MCP, Projects, Models, Decision, Image, Quantization Studio**: the same components. Server gives the base URL, the masked API key and the test command a Copy button each; Models moves secondary actions into a "…" menu.
- **Images → History** ([#200](https://github.com/0xZKnw/mlxl3/pull/200)). Browse the PNGs generated on this Mac, newest first, with their prompt, model, size, steps and seed. **Reuse settings** puts them back in the generator and waits for you to start the run. Older images appear too; what was never saved for them stays empty.
- **Lab → Compare** ([#199](https://github.com/0xZKnw/mlxl3/pull/199)). Run one prompt on two installed text models with the same output limit. They are loaded one at a time, so both do not need to fit in memory together. Each answer shows its time to first visible text, decode speed and generated tokens.
- **Models → Manage storage** ([#199](https://github.com/0xZKnw/mlxl3/pull/199)). Installed models, external folders, interrupted downloads and related Hugging Face cache files in one place. Before a removal, the app estimates the space it frees, hard links and APFS clones included. A model's "…" menu and an interrupted download open this review instead of a confirmation dialog.

## Agent mode

- **A file's content no longer has to be written inside the code.** The `javascript` tool takes an optional `text` argument (at most 1 MiB) that the program reads as the variable `text`, so `write("game.py", text)` needs no escaping. Before, a whole file sat in a JavaScript string: one backtick in it ended the string early and the call failed.
- **A program that does not compile now says where.** The error gives the line and its text, or says that a string or bracket is not closed at the end of the code. When the program writes a file, it also points to the `text` argument. It used to be the bare message, for example `SyntaxError: Invalid character: '#'`.
- Code sent between Markdown fences that does not compile reports that error, instead of an unrelated `TypeError`.
- The call's detail in the chat shows the code, then the text.

These three points change what the model is offered; whether a small model uses `text` on its own has not been tested on a model yet (see the limits below).

## Fixed

- **Local server, `/v1/chat/completions`** (engine 2.0.34, [#198](https://github.com/0xZKnw/mlxl3/pull/198), issue [#195](https://github.com/0xZKnw/mlxl3/issues/195)): `reasoning_content` sent back on an assistant turn reaches the chat template, and `"tool_calls": null` or `[]` are accepted as "no tool call". See the [engine notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.34).

## Unchanged

- The generation inspector and the menu bar panel.
- The Windows app (its installer is built and attached by the `windows-desktop` workflow after publication).

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security.

## Verification and limits

- **Agent `text` argument and error messages**: deterministic checks on CPU, 373 sandbox checks and 87 checks through the real MCP server, with files read back byte for byte; 12 temporary mutants of the change were all caught. These ran before the change was rebased on the other pull requests; none of the files they cover changed in the rebase.
- **Not tested on a model**: whether a model picks the `text` argument by itself, and the effect on a "write a large file" task. No GPU run was made for this release.
- **Pages, image history, comparison and storage review**: verified by their authors in their pull requests (linked above), not again for this release.
- **The test suites were not run for this release** at the maintainer's request; the app compiles and the DMG checks below passed.
- The tool definition sent to the model grows by 65 tokens (Qwen3.8 tokenizer) or 81 (Inkling 2 Mini).
- With models that write tool calls as XML parameters, a `.json` file sent through `text` is written with the same data but sorted keys and no formatting.

## Source and download

- Source: [`6836e664b`](https://github.com/0xZKnw/mlxl3/commit/6836e664b) (tag `v2.0.30`).
- `MLXL3-Desktop-v2.0.30-b83-Apple-Silicon.dmg`: 253,730,091 bytes, SHA-256 `5d88fc8a0f0f42ccb0cb08aa19926e4f8ba1e03c899c8cb75c5ee83b66737673`.
- `MLXL3-Desktop-v2.0.30-b83-Apple-Silicon.dmg.sha256`: 110 bytes.
- Bundled engine: MLXL3 engine 2.0.34, the same files as [`engine-v2.0.34`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.34).
- This DMG: `hdiutil verify` valid, `codesign --deep --strict` passed on the app, version 2.0.30 / build 83, `build-info` has commit `6836e664b` with no tracked changes; the five runtime files are identical to the published engine archive.

## Acknowledgements

- [HENK0O](https://github.com/HENK0O) for the image history ([#200](https://github.com/0xZKnw/mlxl3/pull/200)), the Lab comparison and the storage review ([#199](https://github.com/0xZKnw/mlxl3/pull/199)).
- [EnriqueGalindo](https://github.com/EnriqueGalindo) for the report on `/v1/chat/completions` ([#195](https://github.com/0xZKnw/mlxl3/issues/195)).
- [0xZKnw](https://github.com/0xZKnw) for the page redesign ([#201](https://github.com/0xZKnw/mlxl3/pull/201)) and the report, with a screenshot, of the Agent call that failed on a Python game.
- [Unharnessed](https://github.com/unharnessed-agent/Unharnessed) and [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) for the single-tool code mode the Agent follows. Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
