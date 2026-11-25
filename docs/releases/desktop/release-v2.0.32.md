# MLXL3 Desktop 2.0.32

**Agent works on real folders and takes new instructions mid-task, Image gets Qwen Image Turbo, and Inkling 2 Mini reads long prompts up to about 1.5× faster.**

Desktop **2.0.32 / build 85** bundles MLXL3 engine **2.0.36**: the same files as [`engine-v2.0.36`](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.36).

## New

### Agent

- **Native tools by default.** Agent now gives the model flat tools — `read`, `edit`, `write`, `grep`, `glob`, `bash`, `job_output`, `job_list`, `job_kill`, plus `fetch` and `download` when Internet is allowed — the way [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) does. File contents and edits no longer have to be escaped inside JavaScript. Each call still runs in the same sandbox, with the same permissions. The `javascript` tool remains available under **Model tools**. **Existing workspaces switch to the native tools too.** `grep` gains an `include` file-name filter and `glob` a folder.
- **Attach a project.** In **Tools**, choose **Attach files to read…** for reference material or **Work in a folder…** for an editable copy of a project. The conversation, draft and attachments are kept. Click the folder name in the composer to review its permissions.
- **Review the work from the reply that produced it.** **Review N files** shows the changes made by that turn, even if later turns changed or deleted the files. Failed, interrupted and unconfirmed commands are shown separately.
- **Steer a running task.** Type a follow-up and press Return: the current action finishes, the work is checkpointed and your instruction continues the same conversation. Stop takes priority and keeps a queued instruction as a draft. With the experimental direct Metal engine, the instruction is delivered at the end of the turn.

### Image

- **Qwen Image 2.1 Turbo.** Image now offers the 4-bit [Qwen-Image-2.1-Turbo-EXL3](https://huggingface.co/jyohukuchan/Qwen-Image-2.1-Turbo-EXL3) checkpoint first (8 steps), next to Base. Models download in place from a pinned revision, with progress, cancellation and resume; a model you already have is detected and never downloaded again. An earlier choice of Base is kept.
- **Saving is now optional.** A finished image stays in the preview and **Save…** keeps the PNG. To keep every generation and its settings in the gallery, turn on **Settings → Privacy → Automatically save generated images**.
- **Download PNG** in the image history detail exports the original PNG, including for older images without settings.

### Lab

- **Lab → Images** counts the Base and Turbo images finished on this Mac over the last 30 calendar days and all time. Only the generation ID, model variant and day are stored locally.

## Performance

- **Inkling 2 Mini reads long prompts faster** with engine 2.0.36: experts of up to 8 prompt chunks are computed together, with identical output. On an Apple M5 (24 GB), a prompt of about 1,135 tokens went from 179–199 to **269–294 tok/s** in `mlxl3-rs run`, time to first token 5.7–6.3 s → 3.9–4.2 s. Not measured in the app itself; see the [engine notes](https://github.com/0xZKnw/mlxl3/releases/tag/engine-v2.0.36).

## Fixed

- Drafts, attachments, selection and generation settings are restored from backups. Undoing a deletion keeps the replacement chat and any new work. A restored chat can outlive its deleted project. Malformed histories are reported before being written to.
- Agent: a reviewed rename applies both sides and keeps the original file permissions; literal filenames are handled; the sandbox handles the extra global exposed by macOS 27.3. An empty written file is no longer reported as "1 line", and the line naming the file that holds a command's full output is no longer cut.
- Lab keeps a visible answer when its model fills the context during a comparison. Image prompts starting with a dash are passed verbatim.
- Conversion results are published only after the worker validates them and exits successfully. Updater cleanup only removes recognized archive files.
- Server: a voluntary stop is no longer reported as a failed start, and the final diagnostic is kept. Decision: 16-bit PCM WAV imports are limited to five minutes.
- Windows: history import only fills the space left in the library; a failed migration keeps the previous copy.

## Installation

Requires **Apple Silicon and macOS 26.2+**. Download the DMG below, or check for updates in Settings.

The DMG is ad-hoc signed. First launch may require approval in System Settings → Privacy & Security. The Windows installer is built and attached by the `windows-desktop` workflow after publication.

## Verification and limits

- **No test suite was re-run for this release**: the checks cited here are those of the pull requests ([#203](https://github.com/0xZKnw/mlxl3/pull/203), [#204](https://github.com/0xZKnw/mlxl3/pull/204), [#206](https://github.com/0xZKnw/mlxl3/pull/206), [#207](https://github.com/0xZKnw/mlxl3/pull/207), [#208](https://github.com/0xZKnw/mlxl3/pull/208)), run on their own sources.
- Agent folders and mid-task guidance were tested end to end on an M5 with Inkling 2 Mini (disposable folder, network and shell disabled). Turbo and Base renders at 512 px matched reference PNGs byte for byte; 768 and 1024 px were not checked.
- **Native tools were not tried with a model**: whether small local models do better with them than with `javascript` is not measured. Tool definitions cost more prompt tokens (1,587 against 945 with Inkling 2 Mini, with Internet).
- The Windows app was tested on the Mac only.

## Source and download

To be completed after the build.

## Acknowledgements

- [HENK0O](https://github.com/HENK0O) for the Studio fixes ([#203](https://github.com/0xZKnw/mlxl3/pull/203)), Agent folders, review and mid-task guidance ([#204](https://github.com/0xZKnw/mlxl3/pull/204)), and Qwen Image Turbo, model downloads and Lab image counts ([#207](https://github.com/0xZKnw/mlxl3/pull/207)).
- [0xZKnw](https://github.com/0xZKnw) for the native Agent tools ([#206](https://github.com/0xZKnw/mlxl3/pull/206)) and the grouped Inkling prefill ([#208](https://github.com/0xZKnw/mlxl3/pull/208)).
- [jyohukuchan](https://huggingface.co/jyohukuchan) for the Qwen-Image-2.1-Turbo EXL3 checkpoint; [Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1) and [MFLUX](https://github.com/mflux-community/mflux) for the image pipeline; [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) for the native tool design.
- Runtime and app: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [PonyExl3](https://github.com/beamivalice/PonyExl3), [minijinja](https://github.com/mitsuhiko/minijinja).
