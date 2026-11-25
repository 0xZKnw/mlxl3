# MLXL3 Engine 2.0.32

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release makes speculative decoding pay on **Inkling 2 Mini**, with the retrained block drafter [`incoai/Inkling-2-Mini-DFlash2`](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2):

- **DFlash2 drafter for Inkling 2 Mini**, exact for greedy and sampled decoding, **on by default once the drafter is installed**.
- **Verification rows identical to single-row decoding**, and **experts shared by several rows read once** (same bits, 13% faster on 8 rows).
- A managed, pinned download of the drafter (`mlxl3-rs dflash-draft --target <model>`).

## New

### DFlash2 on Inkling 2 Mini

The drafter proposes up to 7 tokens per block; the target verifies them in one pass. Proposals never change the output law:

- **Greedy:** the speculative output is the single-row output, bit for bit (see below).
- **Sampled (temperature > 0, no top-k / top-p / penalties):** proposals are drawn from the drafter's own distribution and kept with probability min(1, p/q), with a residual draw at the first rejection — the rule used by [IncoAI Splash](https://github.com/incoai/splash). Every token keeps the target's sampling law.
- The drafter always runs on its trained 8-row block and reads the first proposals it was asked for. Running it on a shorter block (the previous behaviour for narrow requests) lowered the acceptance of the first proposal from 0.68–0.69 to 0.54–0.55.
- The number of proposals per block follows the measured acceptance and measured block costs. After a probe whose proposals were all kept, the next block probes again instead of waiting 8 to 64 blocks.

### Exact, cheaper verification

- **Batch-invariant verification:** every row of a 2–8 row verification now has the bits of the same step decoded alone (dense projections, output head, router and message gates got row-exact kernels). Greedy speculative decoding therefore equals plain decoding exactly.
- **Shared experts read once:** when several rows of a verification choose the same expert, its weights are decoded once for those rows (grouped on the GPU by member count, one launch per count). Same bits as the per-route kernel; verification of 8 rows 124.7 → 108.2 ms, 4 rows 73.2 → 69.0 ms (M5, 10-core GPU).

### Default and options

- A bridge request that **omits `dspark`** now speculates when a DFlash2 drafter is available; `"dspark": false` turns it off, `"dspark": true` asks for it as before. With only the official DSpark drafter shipped in the model, an omitted field still means off. `ready` reports `dspark_default`.
- The drafter is looked for in `MLXL3_INKLING_DFLASH2` when set, else in `dflash2/` inside the model folder, `Inkling-2-Mini-DFlash2/` beside it, or the managed download folder.
- `mlxl3-rs dflash-draft --target <Inkling 2 Mini folder>` downloads the pinned drafter (revision `4f9088ea66c21f2b99ed657d1941c306b03e84d2`, `model.safetensors` 989,975,616 bytes, SHA-256 `217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee`).
- Switches: `MLXL3_INKLING_DFLASH2_DEFAULT=0` (omitted field means off), `MLXL3_INKLING_EXPERT_PANEL=0` (per-route experts in verification), `MLXL3_INKLING_INVARIANT=0`, `MLXL3_INKLING_DFLASH2_FULL_BLOCK=0`, `MLXL3_INKLING_DFLASH2_STREAK=0`, `MLXL3_INKLING_DRAFT_RATIO=0`.
- The CLI and the OpenAI-compatible server are unchanged: they do not speculate unless asked.

## Performance

One scenario, measured through the bridge on a MacBook Pro M5 (10-core GPU, 24 GB), Inkling 2 Mini EXL3 3.33 bpw, temperature 1 (the Desktop preset), an English coding prompt, one turn of 1,200 tokens, 4 passes per configuration, on battery:

| Configuration | Decode speed (mean) | Passes |
|---|---|---|
| No drafter (prompt lookup only) | 28.4 tok/s | 28.3–28.5 |
| DFlash2 with the previous width policy | 30.0 tok/s | 28.4–32.0 |
| Full block + measured costs + probes in a row | 34.4 tok/s | 33.4–35.5 |
| **+ shared experts (this release's default)** | **37.6 tok/s** | 32.9–40.2 |

The text drawn differs on every pass, so the spread is large; the last step is established per block (12–15 ms less on blocks of 5–7 proposals) but its tok/s gain is not statistically established. **French prose, long contexts and later turns were not measured**; on text the drafter predicts poorly, the policy falls back to plain steps.

## Also included from `main`

Engine changes merged since 2.0.31, not re-verified for this release beyond the test suites below: DSpark blocks decided on the GPU ([#190](https://github.com/0xZKnw/mlxl3/pull/190)); Flash Next grouped prefill and load-time flag validation ([#185](https://github.com/0xZKnw/mlxl3/pull/185)); the native reasoning-effort protocol with Minimal and model-aware reasoning budgets ([#192](https://github.com/0xZKnw/mlxl3/pull/192)); Qwen Image render time and peak memory at 512 px ([#197](https://github.com/0xZKnw/mlxl3/pull/197)).

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

- **Desktop 2.0.26 and earlier** always send `dspark` explicitly and do not download the drafter: nothing changes for them unless the drafter folder is placed by hand and the DSpark switch is turned on.
- [Desktop 2.0.27](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.27) bundles this engine, downloads the drafter and follows the engine default.
- The bridge protocol is unchanged (1). `dspark` may now be omitted; `null` is still rejected.

## Verification and limits

- **Formatting and lint:** `cargo fmt --check` passed. Clippy `-D warnings` reported nothing on `mlx,chat` and `chat`. The `direct-metal,chat` and `hybrid-dispatch` feature sets were left to CI.
- **Tests** (`mlx,chat`, on the release source): library suite 638 passed, 0 failed, 394 ignored (GPU checkpoints); binaries 68 passed, 0 failed, 15 ignored.
- **Exactness on a physical GPU** (M5, Inkling 2 Mini EXL3 3.33 bpw):
  - verification rows against single-row decoding: 32 of 32 cases bit-identical (logits, drafter features, next step), prefixes 64 and 300, blocks of 2, 3, 5 and 8 rows; with the invariant path off the same test differs in 8 of 8 cases, so it can fail;
  - a greedy DFlash2 session and a DSpark session of 160 tokens: every verified row and every token identical to the plain replay;
  - shared experts: 35 kernel cases (2–8 rows, distinct, identical and repeated routes, 926 shared routes) bit-identical to the per-route kernel, GPU grouping equal to a host recount, a mutated grouping detected;
  - narrow requests read the full block: 64 of 64 anchors equal to the first slots of the 7-proposal request; the old 2-row block matches in only 38 of 64.
- **Sampling law:** the ratio decision was checked against the exact law on synthetic logits (chi-square under the 0.999 bound, a "keep everything" mutation detected), as was the drafter's sampled walk.
- **Drafter fidelity:** an independent float64 NumPy oracle reproduces the candidates and the 7 proposals on three real cases; an independent reading against SGLang `main` found the computation faithful.
- **Managed download:** target selection, folder inspection, pinned metadata (changed repository, revision, size or checksum rejected) are unit-tested; with the pinned files already in a throwaway managed folder, `dflash-draft --target` accepts them after size and SHA-256 checks and rejects a modified `config.json`. **The real 990 MB download was not run.**
- **Bridge, default path:** a request without `dspark` and without any drafter variable ran DFlash2 (`drafter: dflash2`, 875 of 1,415 proposals kept, 37.6 tok/s on the coding prompt at temperature 1).
- **Not verified:**
  - speed on French prose, long contexts, multi-turn chats, or with top-k / top-p (the earlier point rule applies there);
  - the official DSpark drafter's speed with the shared-expert path (its bits were checked);
  - the first-use kernel compilation cost of each block width (paid during decoding, once per width);
  - memory beyond the drafter's weights (about 0.5 GB in 8-bit, loaded at the first speculative request);
  - Kani was not run.

## Artifacts

ARTIFACTS_PLACEHOLDER

## Thanks

- Thanks to [0xZKnw](https://github.com/0xZKnw) for pushing for a drafter that pays on Inkling 2 Mini, for insisting on lossless verification, and for the default-on decision.
- [IncoAI](https://huggingface.co/incoai) for the [Inkling 2 Mini DFlash2 drafter](https://huggingface.co/incoai/Inkling-2-Mini-DFlash2) (Apache-2.0) and for [Splash](https://github.com/incoai/splash): its ratio acceptance rule and its cost-driven choice of verified rows are the models for this path.
- [SGLang](https://github.com/sgl-project/sglang) (`dflash.py`, `dflash_worker_v2.py`), the reference our drafter implementation was checked against.
- [MTPLX](https://github.com/youssofal/MTPLX) by Youssof Al, for showing that verification rows can be made bit-identical to single-row decoding and that shared experts are worth reading once.
- [HENK0O](https://github.com/HENK0O) for the Flash Next, reasoning and image changes merged since 2.0.31 ([#185](https://github.com/0xZKnw/mlxl3/pull/185), [#192](https://github.com/0xZKnw/mlxl3/pull/192), [#197](https://github.com/0xZKnw/mlxl3/pull/197)).
- [Thinking Machines](https://huggingface.co/thinkingmachines) for Inkling 2 Mini. Runtime: [Apple MLX](https://github.com/ml-explore/mlx), [ExLlamaV3](https://github.com/turboderp-org/exllamav3), [minijinja](https://github.com/mitsuhiko/minijinja).
