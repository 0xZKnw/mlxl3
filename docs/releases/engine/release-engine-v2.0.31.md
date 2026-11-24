# MLXL3 Engine 2.0.31

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release takes three ideas from [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) into the **Agent turn loop**:

- When old tool output cannot be shortened enough, the model **summarizes the earlier steps of the task** instead of stopping with a full context.
- A call to a tool the model was not given comes back with **the list of tools it can call**.
- The loop is now **tested without a GPU**, with a scripted model.

## New

### Summarized steps instead of a full context

2.0.30 shortens old tool output when the prompt passes 70% of the context. Some of a long task cannot be shortened, such as many calls whose code stays under 2,000 characters. The context then filled up and the turn ended. In 2.0.31:

- **When it happens:** after pruning, if the prompt still holds more than **75%** of the context, the older exchanges of the current turn are summarized by the model and replaced. A step cut by a full context with nothing left to prune is summarized too, then run once more.
- **What is kept word for word:** the newest exchanges that fit in 20% of the context (always the newest one), the system prompt, the user's request and earlier turns. A call is never separated from its result.
- **What the model writes:** a summary under fixed headings (Progress; Files and code; Commands and results; Errors and fixes; Remaining work; Next step), with exact paths, commands, identifiers and error messages. A previous summary is merged, not repeated. The summary is at most 3,072 tokens, and thinking is closed after 1,024.
- **What the client sees:** the summary is never streamed as an answer. The bridge sends `generation_status` messages: "Summarizing N earlier messages to free context", then "Context summarized: N earlier messages replaced (X to Y tokens)".
- **When it is refused:** an empty summary, a cut one, a summary with a tool call, or one that does not shorten the prompt is dropped. The earlier messages are kept and the loop goes on as in 2.0.30. Another attempt waits until the turn has grown by two exchanges.
- **Where it goes:** at the start of the first kept tool result, the same place as 2.0.30's loop notes, so every chat template accepts it. Pruning never shortens it, and later turns get it through `turn_context`.

### Unknown tools get the remedy

- **Agent mode:** a call to a tool the model was not offered (for example `read_file`, instead of the `read()` function of the `javascript` tool) no longer reaches MCP. The result names the tools the model can call. It also says that functions listed in a tool's description are called inside that tool's input.
- **Chat with project tool restrictions:** a call to a disabled tool used to end the turn with "MCP tool is not allowed in this project". It now gets the same message, and the turn goes on.

### A loop tested without a GPU

The turn loop moved from the bridge binary into the library (`agent_turn`), behind a small host interface (generate a step, count tokens, call a tool, emit an event). Its behaviour is unchanged: same events, texts and conditions. As DeepSeek Harness does with its scripted adapter, 18 tests drive the real loop with a scripted model. They cover:

- direct answers and tool calls;
- unreadable and unknown calls;
- the empty-step reminder, the repeated-call stop and the round limit;
- context-full retries and forced ends of thinking;
- long tool output and cancellation;
- JSON retries and every summary case.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- **Any Desktop version** gets the summary and the unknown-tool message: neither needs a client change. The bridge protocol is unchanged (1); only new status texts are sent.
- [Desktop 2.0.23](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.23) bundles this engine with the matching Agent sandbox and command-output changes.

## Verification and limits

- **Formatting and lint:** `cargo fmt --check` passed. Clippy `-D warnings` reported nothing on `chat`, `direct-metal,chat`, `hybrid-dispatch` and `mlx,chat`.
- **Tests:**
  - `cargo test --features chat`: 673 passed, 0 failed, 12 ignored (the ignored tests need GPU checkpoints, as before).
  - `hybrid-dispatch`: 675 passed, 0 failed, 12 ignored.
- **New tests:** 18 scripted-loop tests and 4 summary-policy tests. 15 temporary mutations of the loop and summary guards were tried, and all were caught. One was caught only after a missing case was added: a summary with text and a tool call.
- **Not verified:**
  - a real Agent task long enough to trigger a summary;
  - the time a summary costs on a local model (roughly 1 to 2 minutes at 20 tokens/s, an estimate, not measured);
  - the quality of summaries written by local models.
  - Kani was not run.

## Artifacts

ARTIFACTS

## Thanks

- Thanks to [0xZKnw](https://github.com/0xZKnw) for asking to take the best of DeepSeek Harness into MLXL3's single-tool Agent.
- [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (MIT), by DeepSeek: its two-level compaction (pruning, then a summary with fixed sections), its code-mode hint for direct tool calls, and its scripted-adapter loop tests are the models for this release. MLXL3's version is sized for small local context windows.
- Runtime: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [minijinja](https://github.com/mitsuhiko/minijinja).
