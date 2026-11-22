# MLXL3 Engine 2.0.25

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release lets an app **choose the reasoning level of each request** for models whose chat template supports it. It changes no kernel and needs no new model files.

## New in this release

### Reasoning level per request

The bridge accepts an optional `reasoning_effort` field on `generate` requests: `none`, `low`, `medium`, `high`, `xhigh` or `max`.

- `none` turns thinking off: the template receives `enable_thinking = false` (and Inkling 2 receives `reasoning_effort = 0`).
- Any other level turns thinking on and passes the level to the template as `reasoning_effort`.
- Each template accepts its own levels and rejects the others with an error:
  - Inkling 2 Mini: all six (its default is `high`, 0.9);
  - Qwen3.8 27B and Qwen3.8 Flash Next: `low`, `medium` and `xhigh` (its default is `xhigh`), plus `none`.
- Any other value is refused before generation starts.
- Without the field, the template default applies and the prompt is byte-identical to 2.0.24, so older apps behave exactly as before.
- The `ready` event now reports `"reasoning_effort_supported": true`. Desktop 2.0.15 shows its reasoning selector only when it sees this.

The level is set for one request and cleared when the request ends, so the next request starts from the template default.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- Every model that 2.0.24 loads still loads, with the same outputs when no level is requested.
- Desktop 2.0.15 bundles this engine. Earlier Desktop versions work with it unchanged; they never send the new field.

## Verification and limits

- **New tests:**
  - template rendering: the default leaves the template untouched, `none` turns thinking off, each other level reaches the template, and empty, unknown or wrongly cased levels are refused;
  - bridge: the field is optional, each level is set for one request and cleared afterwards, invalid levels and non-string values are refused;
  - a single `javascript` tool call keeps multi-line code (comparisons, quotes, template literals, markup-like text) intact in the Qwen XML, Hermes JSON and Gemma tool-call formats. This is a test only, for Desktop 2.0.15's Agent mode.
- **Local checks on the release source:** Rust formatting, strict Clippy (`-D warnings`) for `mlx,chat` and `direct-metal,chat`, and the MLX source inventory (two counters moved: the new `mlx` feature gates and the test's `console.log`).

  | Suite | Passed | Failed | Ignored |
  |---|---|---|---|
  | All MLX targets | 695 | 0 | 350 |
  | CPU | 618 | 0 | 10 |
  | Direct-Metal | 637 | 0 | 144 |

- **Released archive, extracted into a new directory:**
  - Its four files match the manifest (`engine-v2.0.25`, version 2.0.25, minimum Desktop 1.4.0, minimum macOS 26.2.0).
  - The three Mach-O files pass strict signature and arm64 checks, and `mlx.metallib` is identical to MLX 0.32.2's.
  - `runtime-info` reports 2.0.25 and source `9c36fa4ebbbb`. A second clean build gave a byte-identical archive.
  - At the bridge, with a 230 MB test checkpoint of Inkling 2 (its DSpark drafter off with `MLXL3_INKLING_DSPARK=0`):
    - the `ready` event announced the capability;
    - requests with no level and with `none`, `low` and `max` completed;
    - `ultra` was refused with "unsupported reasoning effort".
  - The same check ran 15 times, from the archive and from the Desktop 2.0.15 DMG. Fourteen runs passed in 1 to 2 s. One stopped answering for 120 s at its third request, with about 1 GB of memory free while another 13 GB model was loaded. It was not reproduced in the next fourteen runs.
- **Not verified, and limits:**
  - No speed was measured: no kernel changed, and a request without the field renders the same prompt as 2.0.24.
  - The levels were checked on the templates of Inkling 2 Mini, Qwen3.8 27B and Qwen3.8 Flash Next stored on the release machine. Answers at each level were not compared on the full models.
  - `scripts/smoke-release.py` did not pass. It checks that a second turn recalls a code given in the first, which the test checkpoint cannot do. The bridge served both turns. The full Inkling 2 Mini was not loaded, because another model was in use on the release machine.
  - Kani was not run on the new code.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for asking for a reasoning selector and for choosing its six levels.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and its graded `reasoning_effort`.
- Thanks to the [Qwen team](https://huggingface.co/Qwen) for the `reasoning_effort` and `enable_thinking` controls of Qwen3.8.
- Thanks to [MiniJinja](https://github.com/mitsuhiko/minijinja), which renders the chat templates, and to [Apple MLX](https://github.com/ml-explore/mlx) and [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the runtime and the EXL3 format.

## Source and assets

Engine source commit `9c36fa4ebbbb254acad97c2f6c118e05db1a4afb`. It is `main` at `442625ec0` (engine 2.0.24) plus the reasoning level change, the Desktop 2.0.15 changes, the versions and the MLX source inventory.

- `MLXL3-Engine-v2.0.25-arm64.tar.gz`: 69,078,048 bytes, SHA-256 `9bfa00ef555f3851af06af7831537ef3686024538404570454e9aba1cdfb1f46`.
- `MLXL3-Engine-v2.0.25-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `5dee798f425d6fe59f48893bbb9030221e4a1ad048881d9d0ae44987955cb46a`.
- `mlxl3`: 15,034,096 bytes, SHA-256 `e97a0892da1735860d47e596cc0a3748f325d5010b45778655be3d46a0b05714`.
- Detailed records: `opti.md`, entry `RELEASE-2026-11-22-ENGINE-2.0.25-DESKTOP-2.0.15`.

- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
