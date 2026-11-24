# MLXL3 Engine 2.0.30

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release rebuilds the **Agent turn loop**. A long Agent task is no longer stopped by a total token budget, old tool output is shortened instead of overflowing the context, and a call repeated with the same result ends tool use. Inkling 2 Mini now also **decodes faster in Agent mode**: its prompt lookup and GPU draw were switched off whenever tools were offered, and they now run in every Agent step.

## New

### An Agent loop without a total budget

A Desktop Agent turn used to share one budget, 32,768 tokens, across every step and all reasoning. A long task (for example writing a whole game) stopped mid-work with "32,768-token Agent limit reached". 2.0.30 announces `agent_loop_supported` at `ready`. A client that reads it can send:

- `max_tokens: -1`: each step may use the room left in the context, as in Unharnessed, deepseek-harness and opencode.
- `tool_rounds` up to **512** (was 64).

Reasoning is never cut by default.

### Context pruning instead of a full context

Before a step, when the prompt holds more than 70% of the context, older tool exchanges are shortened until it holds 45%. Three levels are applied oldest first, and the loop stops as soon as there is room:

1. A tool output keeps its first 200 and last 400 characters, where a command's result, errors or test verdict usually are.
2. Long arguments of old calls keep their first 500 characters.
3. As a last resort, an output is replaced by a short notice that tells the model to run the call again.

The newest exchange is always kept whole, and so are recent exchanges whose outputs and arguments fit in 20% of the context. System, user and assistant text is never changed.

If a step still runs into the end of the context, it is run once more after pruning.

### A stop for loops and empty steps

- **Repeated calls.** When the same tool is called with the same arguments and returns the same result 5 times in a row, the result tells the model that tools are off and the next step answers without tools. `stop_reason` is `repeated_calls`.
- **Empty steps.** A step that ends with neither text nor a call, after tool results, gets one reminder added to the last result, and the loop goes on.

### Progress during long prefills

While a long prompt is read, the bridge sends a `generation_status` every 5 seconds ("Reading context · n / N tokens"). A client that stops silent turns (Desktop stops an Agent turn after 5 silent minutes) no longer stops a long prefill.

### Faster Inkling 2 Mini decoding in Agent mode

Prompt lookup and the one-step-ahead GPU draw (engine 2.0.28 and 2.0.26) were only used without tools, so every Agent step decoded on the plain path. Both now run in tool steps too. DSpark stays off unless a request asks for it.

### Optional reasoning budget (not used by Desktop)

A request may send `reasoning_budget` (256 to 131,072 tokens per step). When a step's thinking reaches it, the engine closes the thinking block in the model's own format (`</think>`, `<channel|>`, or Inkling 2's `<|end_message|>`), adds a short note, and lets the same message continue from its exact tokens. This follows Qwen's thinking-budget method. Desktop does not send it: reasoning is not cut, and its length is chosen with the reasoning level.

## Performance

Inkling 2 Mini as Desktop installs it, through the bridge, on an Apple M5 (24 GB) on AC power. The Agent request is real, with the sandbox's single JavaScript tool, context 32,768, at the default reasoning level.

**Fast decoding in Agent steps, temperature 1** (Desktop preset). Four tasks (Perl bugfix, multi-file rename, code question, Python feature), one run per task and path:

| Path | Decode, median | Total time | Tasks passed |
|---|---|---|---|
| Old path | 24.1 tok/s | 307.1 s | 4/4 |
| 2.0.30 | **26.7 tok/s** (+10.5%) | 292.6 s | 4/4 |

Texts are sampled, so token counts differ from run to run, and one run per task is indicative only.

**Greedy, the bugfix task**, one run per path:

| Path | Decode | Time | Text |
|---|---|---|---|
| Old path | 23.6 tok/s | 135.7 s | reference |
| GPU draw | 25.1 tok/s | 127.8 s | identical |
| GPU draw + prompt lookup | 27.3 tok/s | 97.1 s | different |

The gain depends on how much of the output repeats earlier text: edits and quoted code gain most.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- **Desktop 2.0.21 and older** do not read `agent_loop_supported`. They keep sending their 32,768-token Agent budget and 24 tool rounds, so the uncapped loop needs a later Desktop version. With this engine they still get:
  - the faster Inkling decoding;
  - pruning;
  - the repeated-call stop;
  - the empty-step reminder;
  - the prefill progress.
- Requests without the new fields behave as before, except for those additions. The tool-round default for chat stays 5.
- **Greedy text with Inkling tools can differ** from 2.0.29 when prompt lookup is on, as already documented for chat in 2.0.28: verified rows are computed in a batch. At temperature 1, the distribution of each token is unchanged. `MLXL3_INKLING_LOOKUP=0` turns lookup off; `MLXL3_INKLING_GPU_DRAW=0` turns the GPU draw off.

## Verification and limits

- **Unit tests, CPU:**
  - Pruning: start and end kept; full clearing only when shortening is not enough; trigger boundary; newest exchange protected; user text untouched; multibyte arguments.
  - The repeated-call counter.
  - The reasoning-budget bounds and closing text per model family.
  - The tool-round bounds (512 accepted, 513 refused).
  - 15 temporary mutations of these guards were tried; 13 were caught, and the 2 that were not change no behaviour.
- **Full suites before the release merge:**
  - `chat` suite: library 531 passed / 12 ignored, binary 38, contracts 17, OpenAI transport 37.
  - Bridge `mlx,chat`: 19 passed / 6 ignored (they need GPU checkpoints).
  - Clippy `-D warnings` on `mlx,chat`, `direct-metal,chat` and `chat`.
- **Greedy parity of the GPU draw in tool steps:** the same Agent turn on the old path and with the GPU draw produced identical event streams: 2,097 events, including thinking, answer, call arguments and tool results, with the same token hash.
- **GPU trials on Inkling 2 Mini**, one run each at temperature 1, so indicative only:
  - Short tasks: bugfix, rename, question and feature passed, with 0 engine errors and 0 unreadable calls.
  - The pruning trial at a 4,096-token context: pruning fired, the turn ended normally, and the task passed after the start-and-end level was added. The first version cleared whole outputs and lost their last lines.
  - A longer generation task (a curses Snake game) was still working when the trial script's own 1,200-second limit stopped it.
- **After merging with 2.0.29:** formatting, Clippy on the three variants, 41 targeted tests and 17 bridge tests were rerun. The full suites were not rerun for this release.
- **Not done:**
  - A summary-based compaction (deepseek-harness, Unharnessed and opencode can summarise a conversation): pruning keeps the start and end of each output instead.
  - Desktop's request with no total budget is not in a published Desktop yet.

## Artifacts

| File | Size | SHA-256 |
|---|---|---|
| `MLXL3-Engine-v2.0.30-arm64.tar.gz` | 69,185,549 bytes | `d4002c085b1bf11532bf96bd7c2e6f0710b7ba879cbf52a2905254decaa5ee75` |
| `MLXL3-Engine-v2.0.30-arm64.tar.gz.sha256` | 100 bytes | `99b67cf13bebe5217c2efcae5b83991835b14a1e56a592d0dcb2fb7bbfa5ad2e` |

Source: `dc20d6693583` (tag `engine-v2.0.30`).

The released archive was extracted into a new directory and checked:
- **Manifest:** its four files match it (`engine-v2.0.30`, version 2.0.30, minimum Desktop 1.4.0, minimum macOS 26.2.0, arm64, bridge protocol 1).
- **Binaries:** the three Mach-O files pass strict signature and arm64 checks, and `mlx.metallib` is identical to MLX 0.32.2's.
- **`runtime-info`:** it reports 2.0.30 from source `dc20d6693583`; `mlxl3` is 15,262,944 bytes, SHA-256 `ffbd7c6a326081c0bd400ec0fd906a7d96b34447e40008aade184bec8a1d1564`.
- **Bridge:** without any environment variable, on the installed Inkling 2 Mini, a real Agent call ran `return list().length`, got `3` and answered "3 files". `"dspark": true` ran 57 DSpark blocks; without the field, and with `false`, none ran.

## Thanks

- [0xZKnw](https://github.com/0xZKnw) reported the Agent turn stopped at 32,768 tokens with a screenshot. They also asked for a harness on the level of Unharnessed, deepseek-harness and opencode, chose not to cut reasoning, and asked for the fast decoding paths in Agent mode.
- [Unharnessed](https://github.com/unharnessed-agent/Unharnessed): no step cap in a turn, and an output budget equal to the room left in the context.
- [deepseek-harness](https://github.com/deepseek-ai/deepseek-harness): head-and-tail pruning of tool results and reminders on repeated identical calls.
- [opencode](https://github.com/anomalyco/opencode): steps unlimited by default, clearing of old tool output, and a guard against identical calls.
- [Qwen](https://qwen.readthedocs.io/en/stable/getting_started/quickstart.html): the thinking-budget method behind the optional `reasoning_budget`.
