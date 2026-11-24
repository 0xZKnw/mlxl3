# Reasoning levels in Studio Chat on an M5

Comparison source `399016ffac36`, based on `d32130c8dfe4`. The app verification below uses the follow-up source `5bb09853dbc1`. Each setting ran once per question on the same Apple M5 with 24 GiB. The requests used a fresh conversation and greedy decoding with temperature 0, top-k 1, repetition penalty 1, no tools and a 4,096-token context. Both MLX and direct Metal used the shipping bridge protocol and local EXL3 weights. The output limit was 1,536 sampled tokens for MLX and 512 for the separate direct Metal check. Flash used the optional 5 GiB SSD expert cache and GPU hot cache for both requests. Inkling ran with DSPARK disabled.

The product question requested only the integer for 37 × 24. The independently computed answer is 888. The remainder question requested the smallest positive multiple of 7 that leaves remainders 1 and 2 modulo 5 and 3 respectively. Its answer is 56. Thinking tokens are re-encoded from the streamed thinking channel. Total tokens are the engine’s sampled count across all rounds, including structural tokens. The injected transition occupies context and is excluded from both counts. The engine reserves at least half of a short output allowance when selecting the initial thinking threshold. It can finish earlier.

| Model | Question | Effort | Thinking tokens | Total tokens | Visible answer | Correct |
| --- | --- | --- | ---: | ---: | --- | --- |
| Qwen3.6 | Remainders | low | 256 | 257 | 7 | No |
| Qwen3.6 | Remainders | high | 768 | 770 | 91 | No |
| Qwen3.6 | Product | low | 256 | 259 | 888 | Yes |
| Qwen3.6 | Product | high | 494 | 499 | 888 | Yes |
| Inkling 2 Mini | Remainders | low | 194 | 201 | 56 | Yes |
| Inkling 2 Mini | Remainders | high | 247 | 254 | 56 | Yes |
| Inkling 2 Mini | Product | low | 36 | 43 | 888 | Yes |
| Inkling 2 Mini | Product | high | 41 | 48 | 888 | Yes |
| Qwen3.6 direct Metal | Product | none | 0 | 3 | 888 | Yes |
| Qwen3.6 direct Metal | Product | low | 256 | 259 | 888 | Yes |
| Qwen3.8 Flash Next | Product | low | 58 | 63 | 888 | Yes |
| Qwen3.8 Flash Next | Product | xhigh | 67 | 72 | 888 | Yes |

## Earlier runs

On the published MLX engine 2.0.31 (`e117e02e8f7b`), Qwen3.6 used all 1,536 tokens on the remainder problem without a visible answer at the default and native High settings. Its existing explicit budget of 256 produced 7 in 257 tokens. The first prototype emitted a duplicate answer and channel delimiter on the product question in Low. On the remainder question it continued reasoning visibly after the cut and consumed 1,278 tokens in Low. The final continuation gives the model a private instruction to conclude and resumes in the appropriate channel.

## Scope and verification

The counts describe individual deterministic requests. Timings, power draw, memory and accuracy across a wider set of questions were outside the scope of this comparison. Agent and structured JSON requests continue to use native model effort. The threshold applies to an individual thinking block. A model that opens another block or starts another tool step can spend additional tokens. After two forced closures in one step the engine allows natural completion.

Studio’s isolated preference and mode checks passed 477 and 2,333 assertions respectively. They exercised per-model selection, reset, legacy engines, mode switches and the fields actually sent at generation. The Qwen3.6 template regression failed against the original code. The CPU chat library passed 568 tests with 12 ignored on the comparison source and 569 with 12 ignored on the final source. The two bridge binaries passed their scoped-effort regression, one test each. Strict Clippy passed on both backends. Kani verified the CPU reasoning budget, context room and separator arithmetic for arbitrary unsigned inputs, with 42 successful checks and 5 reachable covers. Removing the small-context cutoff fails the boundary test. The local Studio bundle passed its timeline and code-signature checks and matched all four recorded engine payload hashes. GitHub CI was not run for this local branch.

## Final app

The subsequent change in `5bb09853dbc1` adjusts only the private wording when the model has tools available. Scripted tool-enabled turns cover both a direct answer and a successful tool call after a thinking cutoff. A separate build of Studio embeds the MLX engine with SHA-256 `bc07425ababc7fd20ad47a6f7eba987e37c9e291d414c291e996628c06321a5a`. Its runtime revision reports `5bb09853dbc1-dirty` because the study journal was still being completed. All executable source matches the commit. Replaying Qwen3.6 Low on the product question using the binary inside the signed app reproduced 888, 256 re-encoded thinking tokens, 259 generated tokens and the original thinking-stream hash. That replay is separate from the 12-run comparison.

## Reproduction

The two user messages were `What is 37 times 24? Reply with only the integer.` and `What is the smallest positive integer that is divisible by 7, leaves a remainder of 1 when divided by 5, and leaves a remainder of 2 when divided by 3? Reply with only the integer.`

Send one `generate` request per table row over `mlxl3 --registry <isolated-registry> bridge <model> --context-length 4096`. Use `reuse_prompt_cache=false`, `mcp_enabled=false`, `top_p=1` and `presence_penalty=0` with the sampling settings above. The requested Low, High and Extra high budgets are 256, 4,096 and 8,192. For Off, send `reasoning_effort="none"` and omit the budget. Compare streamed deltas by phase and read `complete.stats.generated_tokens`.

MLX executable SHA-256 `9f63a7028c87be492b49cadd399367d722aa8f83f1df824c4edf47052f412d9f`. Direct Metal executable SHA-256 `da53a4c54560da180222adf54b6f410de82c0e3363075fd24dc6f61067842971`.

| Installed model | Config SHA-256 | Template SHA-256 | Tokenizer SHA-256 |
| --- | --- | --- | --- |
| Inkling-2-Mini-EXL3-3.33bpw-074cc704 | `7ffdfd7ef86c4837b32fa54c6112c50afdd7b50d050d5b058f93b1ab1daa4b85` | `53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc` | `7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6` |
| Qwen3.6-35B-A3B-exl3-2.49bpw-1bde3c20 | `aca65841e826dc7aadcb2700d84196c29df812f1b7caea5c2b09362ec738aa51` | `e84f32a23fdda27689f868aa4a1a5621f41133e51a48d7f3efcbea2839574259` | `5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42` |
| Qwen3.8-Flash-Next-exl3-84147e95 | `d4a35225fc5236c8a2d64d01fd273f2f41192be5ff43e5c50ef1ef50e28ccda2` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` | `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` |
