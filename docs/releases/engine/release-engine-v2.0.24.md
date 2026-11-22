# MLXL3 Engine 2.0.24

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release makes **Inkling 2 Mini decode faster**, and the gain grows with the conversation length. It needs no new model files.

- **Keys and values stored in place (no output change).** Each of the model's 48 layer visits per token kept its own attention cache and rebuilt it with a full copy at every token. The caches are now updated in place. Outputs are **identical bit for bit**. Decode is 3 ms faster per token at 1,000 tokens of context and 6.7 ms faster at 8,000.
- **8-bit dense attention projections by default.** The engine now rounds the attention projections to MLX affine 8 bits at load, as it already did for the output head. Decode is about 20% faster. Against an exact reference of the source model, the measured quality cost is within noise. `MLXL3_INKLING_DENSE_BITS=16` restores FP16.
- **Faster first answer.** The engine now warms Inkling 2 up when it loads the model, so the first request no longer compiles kernels.
- **DSpark width tuned for sampling.** The adaptive width now uses costs measured with the 8-bit target. On long sampled answers, DSpark is now level with plain decoding instead of slower.

**Speed on the shipped files** (Apple M5, 10 GPU cores, 24 GB, mains power, `0xzknw/Inkling-2-Mini-EXL3-3.33bpw` at revision `dde31cd0`, greedy decoding, median of 32 steps at each context length). The 2.0.23 path is the same binary with `MLXL3_INKLING_DENSE_BITS=16 MLXL3_INKLING_KV_INPLACE=0`. It reproduces the 2.0.23 logits bit for bit (see Verification).

| Context | 2.0.23 path | 2.0.24 | Change |
|---:|---:|---:|---:|
| 256 tokens | 54.93 / 48.97 ms (19.2 tok/s) | 40.11 / 37.90 ms (25.6 tok/s) | −25% |
| 1,024 | 57.61 / 51.85 ms (18.3 tok/s) | 44.16 / 41.63 ms (23.3 tok/s) | −22% |
| 2,048 | 53.62 / 54.42 ms (18.5 tok/s) | 42.60 / 42.40 ms (23.5 tok/s) | −21% |
| 4,096 | 55.23 / 56.10 ms (18.0 tok/s) | 43.40 / 44.88 ms (22.7 tok/s) | −21% |
| 8,192 | 59.68 / 62.50 ms (16.4 tok/s) | 48.21 / 43.79 ms (21.7 tok/s) | −25% |

- Order: 2.0.24, 2.0.23, 2.0.23, 2.0.24.
- This series is noisy, by up to 6 ms between runs of the same setting. Every 2.0.24 run is faster than every 2.0.23 run at every context length. The tok/s figures are from the mean of the two runs.
- Peak MLX memory is 0.4 to 0.6 GB lower: 12.78–13.36 GB against 13.21–13.99 GB.

Other models are unchanged; see Verification.

## New in this release

### Keys and values stored in place for Inkling 2

Inkling 2 Mini runs its layers 6 to 17 three times per token, so one token makes 48 layer visits. Each visit has its own key/value cache: 40 sliding-window caches of 1,024 positions and 8 full caches. Until now, every token rebuilt every cache with a concatenation of the zero sink row, the past and the new row. That copied each cache in full, once per visit.

The caches now live in preallocated buffers:

- New rows are written with `slice_update`, and MLX reuses the buffer instead of copying it.
- Attention reads a view that holds the sink row, the past and the new rows in the same order as before. The single-query attention kernel reads that view through its strides, without a copy, and picks its kernel from the shapes alone.
- Full caches grow in steps of 256 rows, with one copy per step.
- Sliding caches keep room for 256 more rows after the window. When the window has moved, the row that just left it becomes the zero sink row. When the room runs out, the window is copied once to the start of a new buffer.
- DSpark rollback only moves a logical length.

`MLXL3_INKLING_KV_INPLACE=0` restores the concatenation.

**Speed, harness** (R3 files, 8-bit dense default, greedy, 32 steps per level; order in place, concatenation, concatenation, in place):

| Context | Concatenation | In place | Change |
|---:|---:|---:|---:|
| 256 | 40.09 / 40.24 ms | 38.99 / 39.12 ms | −2.8% |
| 1,024 | 43.89 / 43.88 ms | 40.80 / 40.86 ms | −7.0% |
| 2,048 | 44.82 / 45.21 ms | 41.14 / 41.33 ms | −8.4% |
| 4,096 | 46.94 / 47.19 ms | 41.86 / 43.92 ms | −8.9% |
| 8,192 | 50.85 / 50.69 ms | 43.80 / 44.34 ms | −13.2% |

- Fingerprints of all logits were identical in the four runs at every level.
- Peak MLX memory was 14.15 GB against 13.97 GB at 256 tokens, the same at 2,048 and 4,096, and 14.72 GB against 14.75 GB at 8,192.
- Through the Desktop bridge, on a 1,200-token French essay at temperature 1 followed by a summary, plain decoding ran at 22.16 / 23.59 tok/s in place against 21.83 / 22.40 with the concatenation (+3.4%), and 21.10 / 23.19 against 20.33 / 21.58 on the second turn (+6%). Peak process memory was 14.25 GB against 14.96 GB. Sampled texts differ from run to run, so these are indicative.

### 8-bit dense attention projections by default

The seven attention projections of every layer (q, k, v, r, o, gate, sink) are now rounded at load to MLX affine 8 bits, group 64, like the 8-bit output head since 2.0.19. No file changes.

- Decode: 50.2 → 40.8 ms per token on R3 (−19%, greedy harness). The first and last pairs of that series drifted by 3 to 9%, but the gain held in both orders.
- **Quality** against an exact reference of the source model (12 held-out rows: 4,608 prose positions and 1,536 chat positions, paired by position, 95% bootstrap intervals):

  | Rows | FP16 projections | Difference with 8-bit projections |
  |---|---:|---|
  | Prose (English, French, code) | KL 0.01374 | +0.00034 [−0.00028, +0.00100] |
  | Agentic chat transcript | KL 0.0996 | −0.0048 [−0.0269, +0.0154] |

  Top-1 agreement stays within 2 points per row.
- Against the engine's own FP16 projections, the outputs move by KL 0.0065, with about 98% top-1 agreement.
- `MLXL3_INKLING_DENSE_BITS=16` restores FP16 projections, `MLXL3_INKLING_DENSE_GROUP` selects groups of 32 or 128.

### Warm-up at load

After loading Inkling 2 and its drafter, the engine runs a short throwaway session, then resets both models and clears the MLX cache.

- It covers prefill, a plain decoding step and DSpark widths 0 to 3.
- Before and after the warm-up, the logits and the DSpark tokens are identical bit for bit.
- On an idle machine, the first request's time to first token went from 1.0–1.1 s to 0.39–0.41 s, for 32 prompt tokens. Loading takes about 2 to 5 s longer.
- `MLXL3_INKLING_WARMUP=0` skips it.

### DSpark width policy for sampling

With the 8-bit target, a plain decoding step costs less, so the costs that 2.0.21's width policy assumed were too low.

- When sampling, the policy now uses the measured cost of each width: 1, 1.56, 1.83 and 2.17 steps for widths 0 to 3.
- It probes one width past the best one. It spaces its probes out (8, 16, 32 and 64 blocks) while they do not pay.
- Greedy decoding keeps the 2.0.21 policy: widths and accepted tokens are unchanged.
- On a 1,200-token essay at temperature 1, DSpark ran at 22.60 / 22.92 / 23.00 tok/s against 23.02 / 23.09 / 23.04 tok/s for plain decoding (−0.9%). The previous policy was 10.9% slower than plain decoding there.
- `MLXL3_INKLING_DSPARK_POLICY=p4c` restores the previous policy.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- All Inkling 2 Mini files that 2.0.23 loads still load: the current slim files with 8-bit embeddings (`dde31cd0`), the slim files (`a5763a01`) and R3 (`0a455840`).
- Desktop 2.0.14 needs no update.

## Verification and limits

- **The merge leaves the 2.0.23 path unchanged.** On the shipped files with `MLXL3_INKLING_DENSE_BITS=16`, 128 greedy steps give the 2.0.23 fingerprint, `d582053e10aac428`, with the caches in place and with the concatenation.
- **New defaults**, on the shipped files: fingerprint `064031239bcf7719`, the same in place and with the concatenation. On R3 the fingerprints are those of the measured branch: `6f128d95ef5b9e1e` by default and `b627a40ef61b13ee` with FP16 projections.
- **Keys and values in place, same process and model, against the concatenation.**
  - The test runs a 1,500-token prefill in 128-token chunks, which crosses the 1,024-position window and a buffer compaction. It then runs teacher-forced steps and seven DSpark verifications of 2 to 4 rows with partial commits.
  - On the shipped files, 10,904,508 logits and drafter features are equal bit for bit.
  - A deliberate fault (not rewriting the sink row) changes 9,302,356 of them, so the test catches it.
  - The same test on the tiny model, with a 64-position window and many compactions, gives 10,814,396 equal values.
- **Unchanged checks.** The tiny-model oracle and rollback tests give the same numbers as in 2.0.23. On the shipped files, the warm-up leaves no state, and a width-0 DSpark step equals a plain decoding step bit for bit.
- **Released archive, extracted into a new directory:**
  - Its four files match the manifest (`engine-v2.0.24`, version 2.0.24, minimum Desktop 1.4.0, minimum macOS 26.2.0).
  - The three Mach-O files pass strict signature and arm64 checks, and `mlx.metallib` is identical to MLX 0.32.2's.
  - `runtime-info` reports 2.0.24 and source `2e80d8d70431`.
  - At the bridge, on the shipped files, greedy, temperature 1 and `"dspark": false` requests passed, second turns included, and `scripts/smoke-release.py` passed.
- **Local checks on the release source:**
  - Rust formatting;
  - strict Clippy (`-D warnings`) for `chat`, `mlx,chat` and `direct-metal,chat`;
  - the MLX source inventory.

  | Suite | Passed | Failed | Ignored |
  |---|---|---|---|
  | All MLX targets | 690 | 0 | 350 |
  | CPU | 616 | 0 | 10 |
  | Direct-Metal | 635 | 0 | 144 |

- **Not verified, and limits:**
  - Kani was not run on the new code.
  - The 8-bit projections change the outputs slightly. Their cost was measured on 12 rows, with one agentic transcript, and no task benchmark was run.
  - The bridge ignores the `seed` field of a request and seeds its sampler from the clock, so sampled runs cannot be replayed. Sampled speeds above come from different texts.
  - All speeds come from one machine.
  - The Python suite was not rerun: no Python file changed.
  - The other models' greedy regression and the Desktop suite were not rerun: their code is unchanged.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for suspecting that the loop was slowing Inkling 2 down, which led to the cache fix, and for choosing the 8-bit projections and their group size.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and its DSpark drafter.
- Thanks to [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format and the quantizer used for R3.
- Thanks to [Apple MLX](https://github.com/ml-explore/mlx) for `slice_update` buffer reuse, strided attention reads and affine quantization.
- Thanks to the NInfer project for the spaced probes that the DSpark width policy now uses.

## Source and assets

Engine source commit `2e80d8d7043135640eb4dcf0186f573d9296c745`. It is `main` at `e6962db11` (engine 2.0.23, Desktop 2.0.14), merged with the Inkling 2 branch `claude/inkling2-r3` at `4512ac0ce`, plus the version change, the MLX source inventory and the group size of stored affine weights.

- `MLXL3-Engine-v2.0.24-arm64.tar.gz`: 69,076,820 bytes, SHA-256 `252f606bb040c82127880e4b440d811ec698110059ea8c8679427a4c4f0aaf0d`.
- `MLXL3-Engine-v2.0.24-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `5886f1263822c594f29763015f0717142fd7ae883b6200f8dff39fc3fb793701`.
- `mlxl3`: 15,015,904 bytes, SHA-256 `2eec834b4e05947727650e07be4dee1853b0a0351a9ed0c0fd36a38991655995`.
- Detailed trial records: `opti.md`, entries `SPEED-2026-11-21-INK2M-P6`, `SPEED-2026-11-22-INK2M-P8`, `DSPARK-2026-11-22-INK2M-P10`, `SPEED-2026-11-22-INK2M-P12` and `RELEASE-2026-11-22-ENGINE-2.0.24`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
