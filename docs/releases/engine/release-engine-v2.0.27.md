# MLXL3 Engine 2.0.27

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release lets **Inkling 2 Mini call tools**, gives **Agent mode a longer, sturdier tool loop**, and turns **DSpark off by default** for Inkling 2 Mini, loading its drafter only when asked.

## New

### Inkling 2 Mini calls tools

Inkling 2 Mini used to answer without tools: its chat template could not render them, and the engine did not read its calls. In Agent mode, the model then wrote a call such as `javascript<|content_invoke_tool_json|>{…}` as plain text, and nothing ran.

- **The template renders.** The template engine's `tojson` filter now accepts `sort_keys`, `separators`, `indent` and `ensure_ascii`, with the output of Python's `json.dumps`, as transformers does. Inkling's template uses them to declare its tools and to write its calls. Without options, `tojson` writes exactly what it wrote before, so other models' prompts are unchanged.
- **Calls are read.** `NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}` calls are parsed, also after the model's own text and when the name field is damaged. They are hidden while they stream.
- **Same prompt as the reference.** With tools, a call and its result, the rendered Inkling prompt is identical, byte for byte, to transformers' `apply_chat_template`.

### A longer, sturdier tool loop

- **`tool_rounds` per request**, from 1 to 64. Requests without it keep 5 rounds; Desktop 2.0.16 asks for 24 in Agent mode.
- **The limit ends in an answer.** When the rounds are used up, the last tool result tells the model to answer from what it has, and one more turn writes that answer. A call made anyway is not run, and the request completes with `stop_reason` `tool_round_limit`. It used to end with an error and lose the turn.
- **Unreadable calls go back to the model.** A cut, malformed or text-mixed call used to fail the whole request. The model now reads "your tool call could not be read; nothing was executed" and can try again. It counts as a round.
- **Long tool output keeps its start and its end.** A result over the context budget (8,192 tokens at most) used to keep only its start. It now keeps the first quarter and the rest from the end, where errors and summaries are.
- **`tool_start` carries the call's arguments** (up to 16 KiB), so Desktop can show the code that ran.
- **`callTimeoutSeconds` per MCP server**, from 1 to 120 seconds. Servers without it keep 12 seconds. The Agent sandbox asks for 43, so a command may use its whole limit.
- `ready` announces `tool_rounds_supported` and `mcp_call_timeout_supported`.

### DSpark off by default, drafter loaded on demand

At temperature 1, the Desktop preset, DSpark did not beat plain decoding on Inkling 2 Mini. It is now off unless a request asks for it:

- A request without `dspark` decodes plainly; `"dspark": true` turns it on for that request.
- The drafter is no longer loaded with the model. The first request with `"dspark": true` loads and warms it up, and a request without DSpark frees it.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- Desktop 2.0.15 works with this engine. Desktop 2.0.16 uses its new Agent features.
- Requests without the new fields behave as with 2.0.26, except that Inkling 2 Mini now gets the tools a request offers, and DSpark is off unless requested.

## Verification and limits

- **Inkling 2 Mini, real model.** The Inkling 2 Mini installed by Desktop (`Inkling-2-Mini-EXL3-3.33bpw`) ran Agent tasks through the bridge built from this source, with the real sandbox and the Desktop 2.0.16 request (32,768 tokens, 24 rounds), at temperature 1:
  - 8 runs with the JavaScript tool: **0 engine errors, 0 unreadable calls, no call markup in any answer**, and the arguments reached all 27 `tool_start` events.
  - Bug fix 2/2, rename across 5 files 2/2, a live `fetch` of a GitHub file 2/2, counting TODOs 2/2. The two TODO answers were right, but the pre-registered check rejected their wording, so its own count is 6/8.
  - With `tool_rounds` 1, the request completed with `stop_reason` `tool_round_limit` and no error.
- **Released archive on the same model:** the model ran 2 JavaScript calls and answered with the right file count. A greedy request with `"dspark": true` ran 57 DSpark blocks; without the field and with `false`, none.
- **Drafter on demand** (bridge, Inkling 2 Mini at revision 8-bit embeddings): 12.30 GB after loading; 12.77 GB without DSpark; 13.31 GB with the drafter loaded; 12.34 GB after `"dspark": false`; reloaded correctly afterwards.
- **Prompts.** Inkling's prompt with tools, a call and its result matches transformers 5.16.1 `apply_chat_template` byte for byte. `tojson` with options matches Python's `json.dumps` on a reference value; without options, it matches the previous filter.
- **Mutations caught:** the parser without the Inkling branch, `tojson` ignoring its options, the previous `tojson` (Inkling's template then fails with "too many arguments"), no paragraph between messages, start-only truncation, an MCP deadline that ignores the config.
- **Local checks on the release source:** `cargo fmt --check`; strict Clippy for `mlx,chat`, `chat` and `direct-metal,chat`; tests: `mlx,chat` **708 passed, 0 failed, 363 ignored**, `chat` **627 / 0 / 12**, `direct-metal,chat` **646 / 0 / 146**; MLX source inventory of 341 files.
- **Released archive, extracted into a new directory:** its four files match the manifest (`engine-v2.0.27`, version 2.0.27, minimum Desktop 1.4.0, minimum macOS 26.2.0); the three Mach-O files pass strict signature and arm64 checks; `mlx.metallib` is identical to MLX 0.32.2's; `runtime-info` reports 2.0.27 and source `ed3898889399`.
- **Not verified:**
  - On a real model: the four separate tools with Inkling, and any Qwen model with the new loop. The release run was shortened on request.
  - Speed: no kernel changed. Inkling 2 Mini now decodes without DSpark unless asked, which measured within a few percent of DSpark at temperature 1.
- **Known limit, not changed here:** Qwen's tool prompt differs from transformers' in spacing (compact JSON versus `json.dumps` defaults). Changing it would change every model's tool prompt, and is left for a measured trial.

## Source and download

- Compiled source `ed38988893997613216b7176a193481cbc0a72e1`: `main` at `d5b5d31e7` (engine 2.0.26, DSpark off by default) merged with the Agent work, and the versions.
- `MLXL3-Engine-v2.0.27-arm64.tar.gz` · **69,127,578 bytes**, SHA-256 `52ca00dcfa0e1fb180d24a39d634f1b42e1fa2af498b052d6c063353625acff1`.
- `MLXL3-Engine-v2.0.27-arm64.tar.gz.sha256` · 100 bytes, SHA-256 `4851f446a3e94fed2501aa34e105d5e61d109fb580428dc2d251595732f98ea5`.
- `mlxl3` · 15,149,328 bytes, SHA-256 `eeddf28cde64018bce8efad02bde85cb856ff4eb6c4c12dc0afa241a1ba766f2`.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for the Inkling tool-call report with its screenshot, the Agent harness requests and the DSpark decision.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini and its chat template, to the [Qwen team](https://huggingface.co/Qwen), and to [Hugging Face transformers](https://github.com/huggingface/transformers), whose `apply_chat_template` is the reference for the prompts.
- Thanks also to [Unharnessed](https://github.com/unharnessed-agent/Unharnessed), whose turn loop inspired the answer at the round limit and the start-and-end truncation.
- Runtime: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (EXL3 format), [PonyExl3](https://github.com/beamivalice/PonyExl3) (Metal EXL3 port), [minijinja](https://github.com/mitsuhiko/minijinja) (templates).
