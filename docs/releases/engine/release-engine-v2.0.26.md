# MLXL3 Engine 2.0.26

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release makes **Inkling 2 Mini decode faster where the app spends its time**: sampled answers at temperature 1, the Desktop preset. It needs no new model files. Greedy outputs are identical bit for bit to 2.0.25.

- **Tokens chosen on the GPU, one step ahead.** At the app's preset (temperature 1, no top-k), every token used to be drawn on the CPU from all 200,058 vocabulary scores while the GPU waited. The engine now draws it inside the GPU step and submits the next step before it reads the token, so the GPU no longer waits for the CPU between tokens.
- **Four fused decode kernels, same bits.** Four chains of small GPU kernels now run as single kernels. Each one repeats the operations of the MLX kernels it replaces, in the same order, so every logit is unchanged.

**Speed on the shipped files** (Apple M5, 10 GPU cores, 24 GB, mains power, `0xzknw/Inkling-2-Mini-EXL3-3.33bpw` at revision `dde31cd0`, the Desktop bridge as the app drives it, context 8,192). 2.0.25 is the published engine archive.

| Scenario | 2.0.25 | 2.0.26 | Change |
|---|---:|---:|---:|
| 1,200-token French essay, temperature 1, DSpark adaptive (the app's default) | 23.89 / 23.86 tok/s | 26.80 / 28.15 tok/s | **+12% / +18%** |
| Summary of that essay (second turn) | 22.88 / 22.28 | 27.33 / 27.69 | +19% / +24% |
| Same essay, plain decoding (DSpark off) | 23.86 / 23.46 | 27.68 / 27.74 | +16% / +18% |
| Short answers (3 prompts), temperature 1 | 26.7 (mean) | 28.5 (mean) | about +7% |
| Short answers (3 prompts), greedy | 29.5 (mean) | 31.5 (mean) | 0 to +7% |

- Order: 2.0.25, 2.0.26, 2.0.26, 2.0.25 for each scenario. Sampled texts differ from run to run, so each pair compares different texts. Greedy runs produce the same text with both engines.
- Greedy answers gain little: they mostly run DSpark blocks of several rows, which neither change covers. Two slow 2.0.25 greedy runs (25.9 and 28.0 tok/s) make the greedy mean look better than it is.
- Process memory is the same within noise: 14.17–14.30 GB with 2.0.25 and 14.19–14.20 GB with 2.0.26 on the essay.

Other models are unchanged.

## New in this release

### Tokens chosen on the GPU, one step ahead

A diagnostic in this cycle showed that the decode step is limited by the GPU, not by the CPU. Of about 38 ms per token, the calling thread is busy for about 12 ms, plus 2.4 ms to build the graph. The GPU still waited at every token, during the graph construction and, at temperature 1, during the CPU draw over the whole vocabulary.

- The token is now drawn inside the decode graph. Greedy decoding uses the same argmax of the log probabilities as before, so its tokens are identical. Sampling at a plain temperature (top-k 0, top-p 1, no repetition or presence penalty, which is the Desktop preset) uses a Gumbel-max draw: `argmax(log_probs / T + G)`.
- The Gumbel noise is built from 32 random bits per value, so it reaches tokens down to about 26 nats below the best one. MLX's own `random::uniform` stops at about 20 nats. The draw has the same law as the CPU sampler, up to FP32 rounding of the scores.
- The step that consumes a token is built and submitted before the token is read. The GPU computes it while the bridge decodes and streams the previous token. A step submitted after the end-of-sequence token, the token budget or a cancellation is discarded; the next turn starts from a fresh cache, as before.
- This covers plain decoding and DSpark blocks of width 0, which are most of the blocks at temperature 1. Wider DSpark blocks keep their CPU verification.
- Other sampler settings (top-k, top-p, penalties) keep the CPU sampler.
- The bridge already ignored the request `seed` and seeded its sampler from the clock, so no reproducibility is lost. The random stream itself is new.
- `MLXL3_INKLING_GPU_DRAW=0` restores the CPU sampler. `MLXL3_INKLING_PIPELINE=0` keeps the GPU draw with synchronous steps.

On R3, measured alone through the bridge against 2.0.24's behaviour (same engine with `MLXL3_INKLING_GPU_DRAW=0`): +7.5% to +10.5% on the essay with DSpark, +12.6% to +13.4% with plain decoding, and no change for greedy short answers.

### Four fused decode kernels

The step is limited by chains of small kernels that each wait for the previous one, about 3 to 6 µs each. Four chains now run as one kernel each:

- **Depth attention with one query** (88 calls per token): scores, scale, softmax and output in one kernel, without stacking the sources first.
- **Loop message gates** (96 per token): the concatenation, the FP16 cast, the down projection and the cast back in one kernel. The forget and gain projections and the gate tail run in a second kernel, reading the checkpoint's BF16 weights, which are exact in FP32.
- **Router logits** (48 per token): the FP32 cast and the projection in one kernel, reading the BF16 router.
- **Attention output gate** (48 per token): casts, sigmoid and product in one kernel.

A decode step goes from 8,511 to 6,791 MLX operations. Each kernel repeats MLX 0.32.2's kernel for these shapes: the same thread layout, the same partial sums and the same reduction trees. Each is used only where MLX picks that kernel, and falls back to the usual operations elsewhere, for example during prefill. On R3, measured in one process with the fusions turned on and off in alternate rounds, they save 1.81 ms per token in all 8 rounds. Through the bridge, on the essay with DSpark, they add +4.3% to +5.9%.

Two other fusions, the attention mask and the dynamic convolution, were also exact, but their gain stayed below the threshold that was set before measuring, and they are not included.

`MLXL3_INKLING_P17_OFF=1` turns the four fusions off. A list turns off some of them: `f1` for depth attention, `f2` for the gates, `f3` for the router and `f5` for the output gate.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- All Inkling 2 Mini files that 2.0.25 loads still load.
- Desktop 2.0.15 needs no update.
- The fused kernels keep BF16 copies of the gate and router weights next to the FP32 ones: about 42 MB more memory, an estimate from the tensor sizes. Process memory measured the same within noise.

## Verification and limits

- **Same outputs.** On the shipped files, 128 greedy steps give 2.0.24 and 2.0.25's fingerprint `064031239bcf7719` with the fusions on and off. With `MLXL3_INKLING_DENSE_BITS=16`, they give `d582053e10aac428`, the 2.0.23 path, with the caches in place and with the concatenation. On R3 the fingerprints are unchanged: `6f128d95ef5b9e1e`, and `b627a40ef61b13ee` with FP16 projections. On R3, the context profile from 256 to 8,192 tokens, which includes chunked prefill, gives the same five fingerprints as before. Greedy bridge text on the shipped files is identical to 2.0.24: `57fd01f204ef`, with the same token counts and the same second turn.
- **GPU path.** Decoding with the token drawn on the GPU and one step ahead gives the host decode's logits and tokens bit for bit, on R3 and on the shipped files. A wrong token fed at step 10 is detected.
- **DSpark sessions.** With width-0 blocks drawn on the GPU, synchronously or one step ahead, sessions give the host session's greedy tokens, blocks and widths. At temperature 1, synchronous and one-step-ahead sessions agree for one seed stream.
- **Law of the draw.** Over 268,435,456 values from fixed seeds, the noise stays finite, between −2.81 and 21.26. Its CDF matches the Gumbel law within 4σ, and 16 values exceed 16.7, which MLX's uniform noise cannot reach. Over 4 million draws, the token law passes a chi-square test against the exact softmax: 9.27 for 15 degrees of freedom at T = 1, and 17.73 for 13 degrees of freedom at T = 0.7, with a bound of p < 10⁻⁴.
- **Each fused kernel against the operations it replaces.**
  - Depth attention: 144 cases identical bit for bit.
  - Gates: 6 cases identical.
  - Router: 6 cases identical.
  - Output gate: identical.
  - Deliberate faults are detected: a reversed reduction tree, a reversed tail tree, and `fast::exp` instead of `precise::exp`.
- **Bridge.** Generations cut at the token budget and cancelled with `SIGUSR1` are handled, and the greedy answer is the same before and after them. Greedy, temperature 1 and `"dspark": false` requests passed on the shipped files, second turns included. `scripts/smoke-release.py` passed.
- **Released archive, extracted into a new directory:**
  - Its four files match the manifest (`engine-v2.0.26`, version 2.0.26, minimum Desktop 1.4.0, minimum macOS 26.2.0).
  - The three Mach-O files pass strict signature and arm64 checks, and `mlx.metallib` is identical to MLX 0.32.2's.
  - `runtime-info` reports 2.0.26 and source `3b49db8fa880`.
- **Local checks on the release source:**
  - Rust formatting;
  - strict Clippy (`-D warnings`) for `chat`, `mlx,chat` and `direct-metal,chat`;
  - the MLX source inventory.

  | Suite | Passed | Failed | Ignored |
  |---|---|---|---|
  | All MLX targets | 697 | 0 | 361 |
  | CPU | 618 | 0 | 10 |
  | Direct-Metal | 637 | 0 | 144 |

- **Not verified, and limits:**
  - Kani was not run on the new code.
  - Sampled outputs follow the same law as before, checked by statistical tests, not bit for bit: the random stream is new.
  - The fused kernels cover one-row steps only; wider DSpark blocks and prefill use the previous operations.
  - Speeds come from one machine and one run per arm.
  - The Python suite was not rerun: no Python file changed.
  - The other models' greedy regression and the Desktop suite were not rerun: their code is unchanged.

## Acknowledgements

- Thanks to [0xZKnw](https://github.com/0xZKnw) for steering this round towards decode speed and for choosing to fuse kernels without changing a bit.
- Thanks to [Thinking Machines](https://huggingface.co/thinkingmachines) for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and its DSpark drafter.
- Thanks to [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format.
- Thanks to [Apple MLX](https://github.com/ml-explore/mlx) for the gemv, softmax and sigmoid kernels that the fused kernels reproduce, the custom Metal kernel API, and `async_eval`. The one-step-ahead loop follows the same idea as [mlx-lm](https://github.com/ml-explore/mlx-lm)'s generation loop.

## Source and assets

Engine source commit `3b49db8fa8800b8124e0eb431f6599a20e1a10d8`: `main` at `418836fe8` (engine 2.0.25, Desktop 2.0.15), merged with the Inkling 2 decode branch `claude/engine-2.0.24` (`c30cc6cf2`, `7f4e4d241`), plus the version change.

- `MLXL3-Engine-v2.0.26-arm64.tar.gz`: 69,097,549 bytes, SHA-256 `2557a63d06fa2cbfc39bb986c8fa2e306ea75f8318a496b07d56d29519d0d833`.
- `MLXL3-Engine-v2.0.26-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `494e2d30b7b60c248a43b86266c39ff471a2ee302ff1bf3fced8446bd13d87ac`.
- `mlxl3`: 15,071,472 bytes, SHA-256 `249217ad69417b43f0e9d4c15136ebd11c5458277ce689120dd50001b6a1f80c`.
- Detailed trial records: `opti.md`, entries `SPEED-2026-11-22-INK2M-P13`, `SPEED-2026-11-22-INK2M-P14`, `SPEED-2026-11-22-INK2M-P17`, `SPEED-2026-11-22-INK2M-TOTAL` and `RELEASE-2026-11-22-ENGINE-2.0.26`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
