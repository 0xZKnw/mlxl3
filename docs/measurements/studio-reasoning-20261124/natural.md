# Native reasoning in Studio

All 15 requests returned exactly 888 and ended on the model's end-of-sequence token in a single generation round. Studio selects the effort through the checkpoint's chat template. The request omits the reasoning budget and passes `max_tokens=-1`. The context window is the physical limit on generation. A saved level that the model cannot express falls back to its default.

## M5 observations

One fresh conversation per setting, using the user message `What is 37 times 24? Reply with only the integer.` on an M5 with 24 GiB. The application used the actual local EXL3 checkpoints. The harness called the MLX bridge directly with context 4,096, greedy sampling (temperature 0, top-k 1, top-p 1, repetition penalty 1, presence penalty 0), no tools and prompt-cache reuse disabled. Flash used the optional 5 GiB SSD expert cache and GPU hot cache. Inkling ran with DSPARK disabled. Only one model ran on the GPU at a time.

| Model | Native setting | Streamed thinking, re-encoded tokens | Total sampled tokens |
| --- | --- | ---: | ---: |
| Qwen3.6 | Default | 494 | 499 |
| Qwen3.6 | Off | 0 | 3 |
| Qwen3.6 | On | 494 | 499 |
| Qwen3.8 | Default | 67 | 72 |
| Qwen3.8 | Off | 0 | 3 |
| Qwen3.8 | Low | 58 | 63 |
| Qwen3.8 | Medium | 56 | 61 |
| Qwen3.8 | Extra high | 67 | 72 |
| Inkling | Default | 41 | 48 |
| Inkling | Zero effort | 41 | 48 |
| Inkling | Minimal | 37 | 44 |
| Inkling | Low | 36 | 43 |
| Inkling | Medium | 41 | 48 |
| Inkling | High | 41 | 48 |
| Inkling | Max | 41 | 48 |

The reported thinking count re-encodes the thinking channel with the local tokenizer. The total includes sampled structural tokens. Each request had one context event, one generation round, zero tool calls and EOS with context remaining. Qwen3.6 generated 494 thinking tokens when enabled, passing the previous prototype's 256-token Low threshold. Inkling's native zero setting still produced 41 thinking tokens. The production tokenizer check confirms the zero value is present in its prompt. Low and Medium can finish in either order on an individual question. These samples measure natural completion on a fixed input. Broader accuracy, average token use, throughput, energy and peak memory were outside this study.

## Native controls

Qwen3.6 uses the template's thinking switch. Qwen3.8 accepts Low, Medium and Extra high when thinking is enabled. Inkling maps None, Minimal, Low, Medium, High and Max to 0, 0.1, 0.2, 0.7, 0.9 and 0.99. Its Extra high alias also maps to 0.99. The selector displays one entry for this value. Choosing Model default omits the effort field entirely. An older engine that does not advertise Minimal receives only levels it supports.

## Evidence

Native source `a8baa5ee4059`, engine 2.0.31 MLX, SHA-256 `a8369653bc2af711c88d90a06b19e7d6573c11866946d2ccaadfe855788b9588`. The bundled executable has the same hash. The local measurement file `build/studio-reasoning-qa/natural/a8baa5ee/results.json` alongside the isolated checkout records model, template and tokenizer fingerprints, request payloads, generation outcomes and counts. Its SHA-256 is `91ad21863ac7479d27e6be119b68875d61fdfb55ba7bea1b042ec98293a045fb`. The harness `natural.py` in that study directory hashes to `67e5379973475f58663c1f9b7f9073967883cdaa90604eab69c5d3e743a0f233`.

The final local app uses Studio source `0a89b56d`. The signed Studio executable hashes to `e0cf6e5b809c20524001a4e7e6289d70a8886b150f4a2a3e1dbfeb338d0e2699`. The bundle passed the chat timeline check, code-signature validation and the size and SHA-256 checks on all four engine payloads. Studio tests passed 491 preference assertions and 2,338 mode assertions. The physical Inkling tokenizer check exercised every native setting and the Extra high alias. Both bridge implementations passed their request tests and strict Clippy checks. Kani checked strings of up to eight bytes for the native effort vocabulary, with zero failed obligations and two covered cases. This branch was verified locally.
