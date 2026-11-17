# MLXL3 Engine 2.0.14

Stable Rust/MLX engine for Apple Silicon, published independently from Desktop. This release speeds up **sampled** decoding, which is what Desktop uses by default:

- the certified head from 2.0.13 now also serves sampled LFM3 requests;
- greedy LFM3 uses a smaller int4 screen;
- sampled requests on every model read back only the top-k candidates instead of the whole vocabulary.

All three are on by default and keep the sampling law exact: each token is drawn from the same distribution as in 2.0.13. Weights, quantization formats and the bridge protocol are unchanged.

## New in this release

### Certified LFM3 head for sampled requests (L1)

In 2.0.13 the certified head only served greedy requests, so Desktop's default chat temperature did not use it. It now serves sampled LFM3 requests too.

- **How it works.**
  - The int8 copy of the head and its error bounds give an interval for every logit.
  - The engine recomputes exactly, with the arithmetic of the normal head, every row that can still enter the penalized top-k.
  - A step is certified only when those rows decide the kept set and its values. The token is then drawn from exactly the distribution the full head would give.
  - Otherwise, for example on near-ties at the top-k boundary, the step runs the normal head.
- **When it is used:**
  - LFM3 models with a dense F16 head;
  - sampled requests whose penalties can only lower values (repetition penalty at least 1, presence penalty at least 0) with top-k between 1 and 256, which covers both Desktop presets (temperature 0.2, top-k 80, repetition 1.05 and the LFM3 preset temperature 0.1, top-k 50, repetition 1.1);
  - no MTP head, no draft model and no prompt lookup.
- **Setting.** On by default. Set `MLXL3_LFM3_CERT_SAMPLE=0` to disable it.
- **Memory.** Sampled LFM3 sessions now keep the 263 MB int8 copy loaded. We measured +269 MB of GPU memory and +685 MB of process footprint. The cause of the extra process footprint has not been determined.

### Int4 screen for greedy LFM3 (L2)

- Greedy LFM3 requests now screen the head with an int4 copy (one F16 scale per 64 weights, 140 MB) instead of the 263 MB int8 copy.
- Certification uses the same exact recomputation and the same rules, so emitted tokens stay bit-identical to the normal head.
- GPU memory for a greedy LFM3 session drops by about 123 MB compared with 2.0.13.
- Set `MLXL3_LFM3_CERT_SCREEN=int8` to return to the 2.0.13 screen.

### Top-k readback for sampled autoregressive decoding (Q1)

- A Metal kernel finds, for each sampled row, a threshold that at least k′ values reach, and returns only the values at or above it, together with their token IDs. Penalties only lower values, so when the k-th penalized candidate still reaches the threshold (with no tie), these candidates decide the full distribution.
- The host then reads at most 16 KB per row instead of the whole vocabulary (about 1 MB of F32 for Qwen3.6). Rows that cannot be certified (more than 2,048 candidates, a NaN, or a tie at the boundary) fall back to the full row.
- It applies to every model on the plain autoregressive path, for vocabularies up to 262,144 tokens and top-k between 1 and 256, with penalties that can only lower values. Other requests use the full row.
- **Setting.** On by default for autoregressive sampling. Set `MLXL3_TOPK_READBACK=0` to disable it. MTP verification rows keep the 2.0.13 path unless `MLXL3_TOPK_READBACK=1` is set (see Experimental).
- **Reporting.** Sampled bridge completions add `topk_readback_rows: [certified, fallback]`.

## Experimental (off by default)

- `MLXL3_LFM3_EMBED_LAZY=1` keeps the LFM3 embedding table out of resident GPU memory: −524 MB of GPU memory with no measurable speed change. Outputs are bit-identical on the GPU oracle and the bridge.
- `MLXL3_TOPK_READBACK=1` also applies the top-k readback to MTP verification rows. With MTP depth 2 on Qwen3.6-35B it was not faster (−1.7%, within drift).
- `MLXL3_MTP_SAMPLED_DRAFT=1` draws MTP drafts from the head's own distribution (including compact projection heads) instead of taking the head's argmax. Acceptance rose by 0.6 to 3.7 points, but decoding was 3.0% slower in this implementation, because each draft level needs a host synchronization. Bridge stats add `mtp_sampled_draft_levels`.
- `MLXL3_LFM3_LOOKUP=v2` (prompt lookup with an n-gram index, composed with the certified head) is **not usable**: see Known issues.

## Performance

Measured on an Apple M5 MacBook (24 GB) **on battery, discharging from 83% to 53%**. We used the bridge with a 4096-token context, the prompt cache off and 256 generated tokens, on five prompts: copy a paragraph, edit code, write code, French explanation, and a long prompt.

- Each case alternated off/on/on/off, one fresh bridge per pass, with a warm-up and 2 to 3 repetitions.
- Both arms used the same binary; only the environment variable differed.
- Sampled runs use a clock seed, so their texts differ between runs.
- The numbers come from separate experiments and **do not add up**: L1 and L2 never run together (sampled versus greedy).

| Model and mode | Off | On | Decode |
|---|---|---|---:|
| LFM3-1.7B, LFM3 preset (T 0.1, k 50, rep 1.1), L1 | 46.7–49.0 tok/s | 50.4–53.0 tok/s | +7.4% (+5.3 to +8.9%) |
| LFM3-1.7B, Desktop default (T 0.2, k 80, rep 1.05), L1 | 46.8–48.8 tok/s | 48.9–52.0 tok/s | +5.6% (+4.4 to +7.0%) |
| LFM3-1.7B, greedy, L2 (int4 versus int8 screen) | 51.0–53.3 tok/s | 54.0–57.0 tok/s | +6.7% (+6.0 to +7.0%) |
| Qwen3.6-35B-A3B 2.49 bpw, Desktop default, no MTP, Q1 | 48.3–54.2 tok/s | 52.2–57.3 tok/s | +9.8% (+5.9 to +13.8%) |

- **L1 and L2.** On every case, each "on" pass was faster than each "off" pass, and the gains were larger than the drift within each arm (at most 1.1% for L1 and 0.8% for L2).
- **Q1: the size of the gain is uncertain.** The "on" arm was stable (about 57 tok/s), but the "off" arm drifted by up to +11% between its first and last pass on three of the five prompts. The gain exceeded the drift on 4 of 5 prompts. A repeat on AC power has not been done.
- Prefill is unchanged.

## Installation and compatibility

Install the archive through the Desktop engine updater, or extract all its files into the same directory. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. The archive is ad-hoc signed, so first launch may require approval in System Settings → Privacy & Security.

No Desktop update is required. Desktop 2.0.6 and earlier versions work with this engine unchanged; the new stats fields are optional.

## Verification and limits

- **GPU oracles on a physical Apple GPU** (LFM3-1.7B EXL3 8 bpw, each run alone, on the release source `babbc83e2`). All 9 passed:
  - certified int8 head: bits, bounds and greedy IDs; fallback on near-ties; decoding bit-identical to normal decoding (IDs and caches);
  - L1: certified distributions equal to the full-head distributions, and sampled decoding identical to the full head (IDs and caches);
  - L2: int4 bounds and greedy IDs, and int4 decoding identical to normal decoding (IDs and caches);
  - lazy embeddings identical to resident embeddings;
  - top-k readback kernel equal to a CPU reference.
- **CPU checks of the sampling law:** exhaustive checks over small vocabularies for the compact MTP row expansion, and sampled-draft chains against sequential target sampling.
- **Bridge, released binary extracted from the archive, no environment variables** (prompts prose and copy, 160 tokens; base = engine 2.0.13 source):
  - LFM3 greedy: token hashes identical to 2.0.13, with the same certified and fallback counts (154/5 and 158/1). GPU memory was 123 MB lower than with `MLXL3_LFM3_CERT_SCREEN=int8`, which confirms the int4 screen was loaded. The `MLXL3_LFM3_CERT_CHECK=1` debug oracle (full head run on every certified step) found 0 mismatches. `MLXL3_LFM3_CERT_SCREEN=int8` and `MLXL3_LFM3_CERT_HEAD=0` also gave hashes identical to 2.0.13.
  - LFM3 sampled, both Desktop presets: L1 certified 73% to 88% of steps, with 0 check mismatches. The first token, which comes from prefill, used the top-k readback. `MLXL3_LFM3_CERT_SAMPLE=0` turned L1 off while the readback stayed on, and adding `MLXL3_TOPK_READBACK=0` turned both off.
  - Qwen3.6-35B greedy: token hashes identical to 2.0.13, with no readback.
  - Qwen3.6-35B sampled: the readback certified 67% to 85% of rows, and `MLXL3_TOPK_READBACK=0` turned it off.
  - Qwen3.6-35B with MTP depth 2: only the first autoregressive token used the readback, and `MLXL3_TOPK_READBACK=1` extended it to MTP rows.
  - All runs completed with 160 tokens.
- **Released archive, extracted into a new directory:**
  - Its four files matched the manifest: release tag `engine-v2.0.14`, version 2.0.14, minimum Desktop 1.4.0, minimum macOS 26.2.0, no extra files.
  - The three Mach-O files passed strict signature and arm64 checks.
  - `mlx.metallib` is identical to MLX 0.32.2.
  - `runtime-info` reports version 2.0.14, MLX, chat and source `babbc83e21ab`.
- **Local checks on the release source:**
  - Rust formatting passed.
  - Strict Clippy (`-D warnings`) passed for `chat`, `mlx,chat` and `direct-metal,chat`.
  - CPU library: 467 passed, 10 ignored; binary: 38 passed.
  - MLX library: 506 passed, 277 ignored; MLX binary: 59 passed, 10 ignored; all MLX test targets: 620 passed, 291 ignored.
  - Direct-Metal library: 519 passed, 406 ignored; binary: 59 passed, 10 ignored.
  - MLX source inventory check: passed (328 files).
  - The ignored tests are physical-GPU or fixture tests; the relevant GPU tests above ran separately.
- **Not verified:**
  - Kani was not run on the new code, so bounds and arithmetic of the new paths have no machine-checked proof. Small finite domains were checked exhaustively, and the rest is covered by sampled tests and the GPU oracles.
  - Only LFM3-1.7B EXL3 8 bpw and Qwen3.6-35B-A3B EXL3 2.49 bpw were run. Other models use the same top-k readback path on sampled requests but were not run.
  - The speed measurements were taken on battery and not repeated on AC power.
  - A bridge that alternates greedy and sampled LFM3 requests rebuilds the int4 or int8 copy at each switch; the cost was not measured.
  - M1–M4, 8/16 GiB Macs and macOS 26.2 were not tested.

### Known issues

- **Experimental LFM3 prompt lookup.** With `MLXL3_LFM3_LOOKUP=v2`, the copy prompt diverged from token-by-token decoding after 573 characters, with or without the certified head. v2 only changes where proposals come from: the verify path is the one `MLXL3_LFM3_LOOKUP=1` has used since 2.0.13. On that prompt, `=1` made no proposal, so its output was unchanged. The batched verification is therefore not bit-identical in every context, and the exact cause has not been determined. Keep both modes off; they are off by default and Desktop does not enable them.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3, the request and the release decision. Thanks to [Liquid AI](https://huggingface.co/LiquidAI) for the LFM models, [Qwen](https://huggingface.co/Qwen) for Qwen3.6, [Apple MLX](https://github.com/ml-explore/mlx) for the array runtime and custom Metal kernels, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) for the EXL3 format, [PonyExl3](https://github.com/beamivalice/PonyExl3) for the Metal EXL3 port, [MTPLX](https://github.com/youssofal/mtplx) for the MTP speculative decoding approach, and [TensorFold](https://github.com/ashhart/TensorFold) for the compact draft vocabulary.

## Source and assets

Engine source commit `babbc83e21abe014745a7fdc4eed0530c2f2a8d1`. It is `main` at `ee7826155` (engine 2.0.13 plus Desktop 2.0.6 documentation) plus commit `5b5e8bcad` (decode boosts and defaults), `8abdb899c` (MLX source inventory) and `babbc83e2` (version 2.0.14).

- `MLXL3-Engine-v2.0.14-arm64.tar.gz`: 68,212,724 bytes, SHA-256 `94e320a1335bc4ededd39c1ca00dea1227e73395cc87c313200457093515f315`.
- `MLXL3-Engine-v2.0.14-arm64.tar.gz.sha256`: 100 bytes, SHA-256 `b65b5b663cf14ab296beebc7c1ca4674abe866980876717025b20a935d905442`.
- `mlxl3`: 12,980,544 bytes, SHA-256 `b832cc7ce829bf05618fd1e282da27f75f2dc83bc4b85753590d247be38acc5a`.
- Detailed trial records: `opti.md` (entries `IMPL-2026-11-16-DECODE-BOOSTS`, `ABBA-2026-11-17-DECODE-BOOSTS`, `DEFAULTS-2026-11-17-DECODE-BOOSTS`, `RELEASE-ENGINE-2.0.14`) and `docs/measurements/decode-boosts-20261116/`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
