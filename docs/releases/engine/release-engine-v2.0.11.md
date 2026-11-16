# MLXL3 Engine 2.0.11

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. It contains everything from the 2.0.11 betas and a faster long-prompt prefill for Qwen3.6-35B-A3B. Weights, quantization, vocabulary and greedy outputs are unchanged: the new prefill path is bit-identical to the previous one.

## New in this release: faster Qwen3.6-35B-A3B prefill

- **Grouped MoE prefill, on by default.** Prefill used to run each 256-token slice through all 40 layers, so the routed experts saw at most 256 rows at a time. The engine now runs the MoE of up to eight slices (2,048 rows) in one call. Attention, Gated DeltaNet, the norms, the router gate and the shared-expert gate still run per 256-token slice, so every rounding-sensitive operation keeps its previous shape. A final partial slice of 64 tokens or more joins the last group. Shorter tails keep the previous path.
- **Prefix cache inside a group.** When a group ends with a partial slice, the reusable prefix-cache checkpoint (the last whole 256-token boundary) is captured inside the group. Its state, logits and MTP head state are the ones the per-slice path would have produced.
- **MTP prefill grouped too.** With a Qwen MTP head loaded, the trunk runs in the same groups. The MTP head is still extended per 256-token chunk, with the same token pairs and the same widths as before.
- **Shared weight decode in the expert kernel, on by default.** The segmented expert matmul decoded the EXL3 trellis weights separately for each 32-row half of a 64-row expert segment. It now decodes each weight fragment once for both halves. Each accumulator receives the same sequence of operations, so outputs are bit-identical.
- **Settings.**
  - `MLXL3_Q36_MOE_GROUP` accepts `1` to `8` (default `8`). `1` restores per-slice prefill. Any other value is rejected when a request starts.
  - `MLXL3_EXPERT_QMM_SHARED=0` restores the previous expert kernel.
  - DFlash2 prefill is not grouped.

## Included from the 2.0.11 betas

- **Beta 1:** single-pass target head for MTP verification widths 3 and 4, progressive layer submission, four-row 35B verification, parallel route planning for segmented prefill, and a smaller final prefill layer. See [2.0.11 beta 1](release-engine-v2.0.11-beta.1.md).
- **Beta 2:** post-norm MTP wiring and confidence-based verify truncation (EVICT), both on by default. Profiles saved by Tune before beta 2 are measured again. See [2.0.11 beta 2](release-engine-v2.0.11-beta.2.md).
- **Beta 3:** Qwen3.8 Flash Next sizes its memory reserve from the Mac's RAM, so it loads on most 16 GiB Macs, and gives a clearer refusal when memory is short. See [2.0.11 beta 3](release-engine-v2.0.11-beta.3.md).

## Performance

Measured on Apple M5 with 24 GiB and macOS 27.2, **on battery**, with Qwen3.6-35B-A3B EXL3 2.49 bpw. At the user's request, testing stopped after the campaigns below.

- **Protocol:** `benchmarks/compare_native.py` runs the same binary in both arms:
  - previous behavior: `MLXL3_Q36_MOE_GROUP=1 MLXL3_EXPERT_QMM_SHARED=0`;
  - new defaults: no variable set.
- **Workload:** one 64-token warmup, then two measured 64-token replies per pass. Greedy decoding, prompt cache off.
- **Rates:** prefill is prompt tokens divided by the time to evaluate them, through the bridge.
- **Parity:** every campaign reported identical tokens, text and history in both arms. With MTP3, both arms accepted the same 43 of 52 drafted tokens.

| Workload | Order | Prefill, previous → 2.0.11 | Time to first token |
|---|---|---:|---:|
| AR, 1,097-token prompt | ABBA | 453.63 → 644.52 tok/s (+42.08%) | 2.421 → 1.703 s (−29.6%) |
| AR, 2,825-token prompt | ABBA | 450.05 → 639.81 tok/s (+42.16%) | 6.280 → 4.418 s (−29.7%) |
| MTP3, 1,097-token prompt | ABBA | 463.22 → 653.62 tok/s (+41.10%) | 2.371 → 1.691 s (−28.7%) |
| MTP3, 1,097-token prompt | BAAB | 457.63 → 644.48 tok/s (+40.83%) | 2.410 → 1.714 s (−28.9%) |
| MTP3, 1,097-token prompt, released archive | ABBA | 458.57 → 653.46 tok/s (+42.50%) | 2.396 → 1.683 s (−29.7%) |

- **Short prompts and decoding:** prompts under 256 tokens take the same path as before. On the short prompt and in decode throughput, the arms differed by −14% to +14% with no consistent sign. The decode path is unchanged, so this is attributed to run-to-run noise on battery (charge fell from 75% to 30% over the campaigns), but a decode change of that size is not excluded by these measurements.
- **Where the gain comes from:** in the step-by-step trials, grouping four slices and merging the tail gave +35% to +39%. Eight slices instead of four added +3% to +5% on 2,825 tokens. The shared-decode kernel added +8% to +11% on top. At the component level, it made the expert matmul 6.7% faster at 1,024 rows and 11.0% faster at 2,048 rows, with no change at 256 rows. These trials used different baselines, so their percentages must not be added together.
- **Run-to-run variation:** absolute rates varied by up to about 8% between campaigns. The gain had the same sign in every campaign and every order.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

The bundled MLX 0.32.2 libraries and Metal library are byte-identical to 2.0.11 beta 3.

## Verification and limits

- **Correctness before speed:** independent GPU oracles compared the new paths with the previous ones on Qwen3.6-35B-A3B, byte for byte.
  - Grouped prefill, 19 cases from 329 to 4,169 tokens with groups of 2 to 8 slices: the last-row logits, last hidden state, all 80 layer states and four continuation tokens were identical.
  - Prefix-cache checkpoints inside a group, 5 cases: restored state and continuation identical.
  - Expert kernel at 64, 100, 256, 1,024 and 2,048 rows on three layers, and full-model runs: identical.
  - MTP prefill, 6 lengths with 4 checkpoints: every residual row, logits, layer states and continuations identical.
- **The oracles catch real defects:** deliberately broken variants were rejected, with millions of differing values. These variants were a router gate over all rows, a second subblock reading the wrong rows, and an MoE call that was not split per slice.
- **Local checks on the release source:**
  - Rust formatting passed.
  - Strict Clippy passed for `mlx,chat` (all targets) and for the CPU build.
  - CPU library: 436 tests passed, 9 ignored.
  - MLX library: 478 tests passed, 260 ignored. The ignored tests are physical-GPU or fixture tests; the relevant GPU tests above ran separately.
  - MLX binaries: 51 tests passed, 9 ignored.
- **Archive check:** the archive below was extracted into a new directory.
  - Its files matched the manifest: release tag `engine-v2.0.11`, minimum Desktop 1.4.0, minimum macOS 26.2.0.
  - The three Mach-O files passed strict signature and arm64 checks.
  - `runtime-info` reports version 2.0.11, MLX, chat and source `5a46dfe683f6`.
  - The extracted engine passed the MTP3 bridge comparison in the table.
- **Not verified:**
  - Kani was not run for the new prefill contracts (the harnesses exist but were not executed for this release).
  - Behavior on 8 or 16 GiB Macs, and the extra transient memory of 2,048-row MoE batches, were not measured.
  - Other EXL3 MoE models (Gemma 4, Ling, LFM2 MoE, Qwen3.8 Flash Next) also use the shared-decode expert kernel when their prefill reaches it. The kernel is bit-identical by construction, but no oracle was run on these models.
  - DFlash2 prefill is not grouped. The Desktop app path and real prefix-cache reuse across turns were not exercised through Desktop.
  - M1–M4 and macOS 26.2 were not tested.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the optimization direction and the release decisions. Thanks to [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime and Metal tensor operations, [Qwen](https://huggingface.co/Qwen) for Qwen3.6-35B-A3B, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format, [PonyExl3](https://github.com/beamivalice/PonyExl3) for the Metal EXL3 port, and [MTPLX](https://github.com/youssofal/mtplx) and [TensorFold](https://github.com/ashhart/TensorFold) for the speculative decoding work this engine builds on.

## Source and assets

Engine source commit `5a46dfe683f6b50476a53a7a64ccbb93cd068e8b`. Its engine code is 2.0.11 beta 3 (source `22ac28026`) plus commits `92e48103a` (grouped MoE prefill and shared expert decode) and `5a46dfe68` (grouped MTP prefill). The other commits since beta 3 change only Desktop and documentation.

- `MLXL3-Engine-v2.0.11-arm64.tar.gz`: 68,120,532 bytes, SHA-256 `919fc7fd2672a443875a13c5498f674edb2405b88299696301279a9f63d8b345`.
- `MLXL3-Engine-v2.0.11-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `b0158bc42a7dfd4787ba239a81e4c73f5b920330c76f99db8b206fa4e1ef52b2`.
- `mlxl3`: 12,766,848 bytes, SHA-256 `8bfe5244503992ba2988d01b242f346e1356eb1207647102026fa1c2a3c41246`.
- Detailed trial records: `opti.md` (entry `Q36-MOE-GROUP-20261116`) and `docs/measurements/q36-moe-group-20261116/`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
