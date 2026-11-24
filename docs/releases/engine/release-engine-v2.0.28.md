# MLXL3 Engine 2.0.28

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release makes **Inkling 2 Mini conversations much faster after the first turn**: a new turn no longer reads the whole conversation again. It also adds **prompt lookup**, which speeds up answers that copy earlier text (code, rewrites), and moves Inkling's router tail into one kernel.

## New

### Prompt cache for Inkling 2 Mini

Until now, every Inkling 2 Mini turn prefilled the whole conversation again, and so did every tool round. After a 3,000-token document, a follow-up question waited about 18 seconds before its first token.

- **A turn resumes the previous prompt** at its last 128-token boundary and reads only what comes after it: the previous answer and the new message.
- **Same numbers as before.** The resumed state gives the logits of a full prefill to the bit: 15,604,524 values compared, before and after the 1,024-token sliding window.
- **Bounded memory.** Full-attention visits keep their rows in the live cache, so only the sliding windows and the short-convolution states are saved: about 214 MB, within the existing 256 MiB prompt-cache budget. The cache follows the usual rules: same `conversation_id`, `reuse_prompt_cache`, no memory saver, no images. DSpark requests do not use it.
- `MLXL3_INKLING_PROMPT_CACHE=0` turns it off.

### Prompt lookup for Inkling 2 Mini

Prompt lookup drafts the tokens that followed the latest earlier copy of the text's end: quoted code, a file being edited, a passage being rewritten. It needs no extra weights and no memory.

- **The target checks every proposal** in one verification block. Greedy requests keep the argmax of each verified row; sampled requests use exact point-mass speculative acceptance, so every sampled token keeps the model's distribution.
- **A controller decides each step** from the recent acceptance and the measured block costs. Plain text stays on plain decoding (one step ahead on the GPU, as in 2.0.26); a copy switches to blocks of up to 7 proposals.
- **On by default** for plain Inkling 2 Mini decoding with the Desktop sampling preset (temperature 1, no top-k, top-p 1, no penalties) or greedy. `"lookup": false` in a request, or `MLXL3_INKLING_LOOKUP=0`, turns it off. It is not used with tools (Agent mode) or with DSpark.
- **Greedy text can differ from plain decoding.** Verified rows are computed in a batch, whose rounding differs slightly from one-token steps, so a near tie can resolve the other way, as with DSpark. At temperature 1, the distribution of each token is unchanged.

### Router tail in one kernel

Inkling 2 Mini's router tail for one token runs as one Metal kernel: sigmoid, biased top-8, normalization and scale. It gives the same routes and weights to the bit as the MLX operations it replaces. Its speed gain is not established (−0.13 to −0.17 ms per token in two paired runs); it is kept because it changes no output and costs no memory. `MLXL3_INKLING_P17_OFF=rt` turns it off.

## Performance

Measured through the bridge on an Apple M5 (10-core GPU, 24 GB), on AC power, with the Inkling 2 Mini files Desktop installs (R3, 8-bit embeddings), context 8,192. Each comparison alternates A/B/B/A in separate processes. Sampled texts differ from run to run.

- **Prompt cache**, a document of 2,764 tokens and two follow-up questions, greedy:
  - turn 2 (3,082 tokens, 2,688 resumed): prefill **2.39–2.45 s instead of 17.35–18.03 s**;
  - turn 3 (3,297 tokens, 3,072 resumed): **1.41–1.47 s instead of 18.49–18.50 s**;
  - the same tokens in all four runs; decode speed and memory unchanged (peak +65 MB).
- **Prompt lookup**, temperature 1, decode tok/s with / without:
  - annotating a 6,000-character Rust file: **36.5 and 32.6 / 27.2 and 27.1** (+20 % and +34 %);
  - rewriting a French passage: **30.0 and 45.5 / 28.0 and 28.5** (+7 % and +60 %); greedy 46.3 / 27.9;
  - a 900-token explanation (plain prose): 28.9 and 28.5 / 28.6 and 28.3 (no change).
- These are separate measurements; the gains do not add up. Speed depends on how much of the answer repeats earlier text.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- Desktop 2.0.19 bundles this engine. Older Desktop versions work with it too: the prompt cache and prompt lookup need no new request field.
- Requests behave as with 2.0.27, except that Inkling 2 Mini resumes cached prompts and uses prompt lookup unless asked not to.

## Verification and limits

- **Prompt cache, GPU parity test** on Inkling 2 Mini R3: prompts of 300 and 1,300 tokens, a pipelined GPU decode and a partly accepted verification block, then the next prompt resumed against a prefill from scratch: every chunk's logits and 32 following steps identical to the bit. Two mutations are caught (live convolution states, sliding keys rolled back by row count past the window); a prefix from another cache lineage is refused.
- **Prompt lookup:** on a prompt that repeats a passage, 138 of 138 proposals accepted, and every delivered token equals the argmax of a one-token replay. CPU tests cover the n-gram lookup against a reference scan (400 sequences), the controller, and the request field.
- **Router tail:** 240 cases identical to the bit against the MLX operations (ties, amplitudes up to 40); a reversed sum is caught.
- **Unchanged numbers:** the historical decode fingerprints of Inkling 2 Mini (default, FP16, five contexts up to 8,192, GPU token choice, small model) are identical.
- **Released archive, extracted into a new directory:** its four files match the manifest (`engine-v2.0.28`, version 2.0.28, minimum Desktop 1.4.0, minimum macOS 26.2.0); the three Mach-O files pass strict signature and arm64 checks; `mlx.metallib` is identical to MLX 0.32.2's; `runtime-info` reports 2.0.28 and source `be09e2c0c3b6`.
- **Released archive in the bridge**, without any environment variable:
  - with the installed Inkling 2 Mini, a real tool call: 2 `sandbox.javascript` calls, results `3` and the file list, answer "3 files"; `"dspark": true` ran 57 DSpark blocks, without the field and with `false` none;
  - a 3-turn conversation resumed 2,688 and 3,072 prompt tokens; with `MLXL3_INKLING_LOOKUP=0`, its tokens are identical to those of the measured engine without the router-tail kernel;
  - prompt lookup on (default) against off, same conversation, greedy, A/B/B/A: decode 25.3–26.6 against 24.5–26.4 tok/s (no slowdown on this summary-style answer; the machine was busy with other apps during these runs);
  - a request without the `lookup` field (as Desktop sends it) ran 155 lookup blocks on the code task.
- **Local checks on the release source:** `cargo fmt --check`; strict Clippy for `mlx,chat`, `chat` and `direct-metal,chat`; tests: `mlx,chat` **714 passed, 0 failed, 367 ignored**, `chat` **628 / 0 / 12**, `direct-metal,chat` **647 / 0 / 146**; MLX source inventory of 342 files.
- **Not verified:**
  - the prompt cache at temperature 1 in the bridge (same prefill path), in Agent tool rounds, and beyond 3,300 tokens;
  - this archive's speed: the measurements above used an engine built from the same sources without the router-tail kernel, which changes no output;
  - where the greedy text with prompt lookup first differs from plain decoding.
- **Diagnostic, nothing changed:** Inkling's EXL3 expert kernel keeps 77 % of its time when its data is in cache, so it is mostly bound in the core (trellis decoding), not by memory latency.

## Source and download

- Compiled source `be09e2c0c3b6480e2b06a7217d53ec42f756a539`: `main` at `7350d3085` (Desktop 2.0.18) merged with the Inkling work (`f636e47ee`), and the versions.
- `MLXL3-Engine-v2.0.28-arm64.tar.gz` · **69,142,557 bytes**, SHA-256 `24b22246c42c36e851be434b7ec65144b0dd34f8e9cb53d8c1e50028439c554b`.
- `MLXL3-Engine-v2.0.28-arm64.tar.gz.sha256` · 100 bytes.
- `mlxl3` · 15,170,560 bytes, SHA-256 `4e088591d7378b59b9a7cebb5817d4ba444bf35dd43eb00b498ef620756e9019`.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for pointing at Project Maya and for the prompt-lookup decision.
- Thanks to [Project Maya](https://github.com/mw00/project-maya) by [mw00](https://github.com/mw00) and its contributors, whose Inkling 2 engine (`maya-ik2`, MIT) keeps the conversation's cache between turns and drafts from the prompt with a per-step controller. These ideas were reimplemented here for MLX; no code was taken.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini.
- Runtime: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) (EXL3 format), [PonyExl3](https://github.com/beamivalice/PonyExl3) (Metal EXL3 port), [minijinja](https://github.com/mitsuhiko/minijinja) (templates).
