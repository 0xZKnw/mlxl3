# Results

Measured with mlx-serve on one Apple M5 Max (128 GB, macOS 27.0.1), serving [Qwen3.8-Flash-Next](GLOSSARY.md#qwen38)
from the MLX repacks of [turboderp](GLOSSARY.md#turboderp)'s packs (the table in [the README](../README.md#the-packs)).
The context-length tables for K2 to K5 were also in the description of
[ddalcu/mlx-serve#997](https://github.com/ddalcu/mlx-serve/pull/997). The code that served them is the branch
[`qwen38-exl3-full`](https://github.com/davidtai/mlx-serve/tree/qwen38-exl3-full) of davidtai/mlx-serve, which serves
these packs through turboexl3's [quant plugin](GLOSSARY.md#quant-plugin).

## Quality: KL divergence and perplexity (turboderp's)

The quality of the weights themselves, as turboderp measured it with exllamav3 on his files, before the MLX repack (the
repacks hold the same files; these numbers were not measured through mlx-serve). Mean KL divergence D_KL(p_BF16 ‖ p_quant)
and perplexity over a self-generated in-domain trace (14,496 input and 45,588 output tokens); BF16 has a perplexity of
1.38474 on it, and the KLD noise floor is 0.00262. From [his model card](https://huggingface.co/turboderp/Qwen3.8-Flash-Next-exl3):

| pack | turboderp's branch | KLD vs BF16 | perplexity |
|---|---|---:|---:|
| K2 | 2.05bpw_h4_ng4 | 0.0681 | 1.4871 |
| K3 | 3.05bpw_h5_ng5 | 0.0174 | 1.4074 |
| K4 | 4.05bpw_h6_ng6 | 0.0066 | 1.3897 |
| K5 | 5.05bpw_h6_ng6 | 0.0040 | 1.3848 |

His charts (the EXL3 series is his plain quants, whose 2.05 to 5.05 points are K2 to K5; EXL3-SC are his self-calibrated
quants; GGUF, GSQ-RCO and NVFP4 are other formats on the same trace):

![Mean KL divergence against bits per weight, by turboderp](https://cdn-uploads.huggingface.co/production/uploads/6383dc174c48969dcf1b4fce/e3PezJzzDxLGuTs6Am4rg.png)

![Perplexity against bits per weight, by turboderp](https://cdn-uploads.huggingface.co/production/uploads/6383dc174c48969dcf1b4fce/gz1lo7P1z9YCURBcogBQZ.png)

![Per-token KL divergence histogram, by turboderp](https://cdn-uploads.huggingface.co/production/uploads/6383dc174c48969dcf1b4fce/Ut4S8nt-ceHFbBnnfPBPp.png)

The end-to-end check through mlx-serve is the [coding evals](#coding-evals) below.

## Charts

Prefill, decode, peak memory and time to first token from 1K to 256K, every pack and arm in one row per metric, with
the standard mlx-serve setup (the iQ-MLX 4.7 bpw pack) among them; time to first token is drawn as each series' %
difference from that setup. Each chart carries its method in its caption and is drawn from the tables below.

![Every series in one row per metric: prefill, decode, peak memory and time to first token](charts/overview_all_series.png)

Every series as its difference from the standard setup at each length (0 = standard mlx-serve):

![Every series against the standard mlx-serve setup](charts/overview_vs_standard_all.png)

The same data with one column per pack, the standard setup as the reference line in each:

![Prefill, decode, peak memory and time to first token by prompt length, K2 to K5](charts/overview.png)

Each metric at full width, every series: [prefill](charts/all_prefill.png), [decode](charts/all_decode.png),
[peak memory](charts/all_peak_memory.png), [time to first token, in seconds at each length](charts/all_ttft.png).

- [Wall time and answer length](charts/wall_time.png)
- [MTP acceptance: per-draft hit rate and drafts accepted per round](charts/mtp_acceptance.png)
- [K5 with MTP off](charts/mtp_off_k5.png)
- [turboexl3 against Sushi, percent difference](charts/turbo_vs_sushi.png)
- [K5 against the standard setup](charts/vs_standard.png)
- [Coding evals](charts/coding_evals.png)
- [Kernels against Sushi](#kernels-against-sushi), one MoE layer call at a time: [decode](charts/kernel_decode.png), [prefill](charts/kernel_prefill.png)

## Across context lengths

One run per cell. Each prompt is the same fixed text (Python source with one coding task) for every pack and arm, sent as
one user message in Qwen's chat template with thinking on, greedy, seed 1234, at most 1,024 new tokens. Each run gets a
fresh server after a warm-up boot on the same prompt and a 40 °C cool gate. Server flags: `--ctx-size 262144
--max-concurrent 1 --prefix-cache-mem 1GB` and the MTP flags below, with `MLX_SERVE_NOCACHE_LOAD=1 MLX_SERVE_NGRAM_WARM=0
MLX_SERVE_GPU_CEILING_MB=114688 MLX_SERVE_CACHE_LIMIT=2147483648`. The rates come from the server's own request line.

- **K2 to K5** are the 2.05, 3.05, 4.05 and 5.05 packs, run on mlx-serve's tree from before its rewrite (PR 978) with
  turboexl3 inside. "Sushi" is [exact numerics](GLOSSARY.md#numerics) (`MLX_SERVE_BIT_EXACT=1`:
  [Sushi](GLOSSARY.md#sushi)'s routed experts, every reordering path off), and "turboexl3" is the default reordered
  numerics, on one binary. K5 ran on build S21 (turboexl3's kernels as of 4a4ea95, with the host fixes of #997) at
  [MTP](GLOSSARY.md#mtp) depth 4 (`--mtp-max-depth 4 --mtp-typical 0.2`), with turboderp's MTP mixer. K2 to K4 ran on
  the earlier build PR17B at MTP depth 3 and were not re-measured.
- **iQ-MLX 4.7 bpw** is the standard mlx-serve setup, with no turboexl3: upstream main (edebbb3) with only #997's two
  load-time switches, `MLX_SERVE_NOCACHE_LOAD` and `MLX_SERVE_NGRAM_WARM` (binary 7f41623228f7). It runs ddalcu's
  [`Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw`](https://huggingface.co/ddalcu/Qwen3.8-Flash-Next-MLX-Serve-iQ-MLX-4.7bpw)
  (revision a0e89db090a3): affine 4-bit experts (group size 64) calibrated with an importance matrix, the rest in 8-bit,
  75.3 GB without the n-gram table. It ran at mlx-serve's default MTP depth (auto: it ran depth 6) with
  `--mtp-typical 0.2`, one arm, from a copy whose `generation_config.json` gained
  `default_chat_template_kwargs: {"enable_thinking": true}`, which the EXL3 repacks carry, so every column renders the
  same prompt. It replaces ddalcu's uncalibrated mixed 4/8-bit pack, which he flagged as bad. It also ran at the EXL3
  runs' fixed depths, `--mtp-max-depth 3` and `4`, on the same binary and prompts: see
  [the standard setup at fixed MTP depths](#the-standard-setup-at-fixed-mtp-depths).
- `*`: the answer reached the 1,024-token cap. `†`: a single run flagged as noisy (K2 8K, K4 1K and 2K), not re-run. With
  MTP, a decode rate also depends on how many drafts its answer accepts, and the arms give different answers.
- MTP off was measured on K5 only.

Prefill, tok/s (MTP on):

| prompt | tokens | iQ-MLX 4.7 bpw | K2 Sushi | K2 turboexl3 | K3 Sushi | K3 turboexl3 | K4 Sushi | K4 turboexl3 | K5 Sushi | K5 turboexl3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 1,031 | 1,379.7 | 1,648.9 | 1,603.2 | 1,580.5 | 1,572.3 | 1,025.5† | 1,373.0† | 955.7 | 1,093.7 |
| 2K | 2,013 | 1,834.9 | 2,147.0 | 2,114.1 | 2,057.0 | 2,080.1 | 1,916.3† | 1,423.7† | 1,161.2 | 1,625.5 |
| 4K | 4,025 | 2,311.6 | 2,518.1 | 2,554.9 | 2,464.7 | 2,497.5 | 1,938.8 | 2,116.1 | 1,884.1 | 1,987.2 |
| 8K | 8,100 | 2,648.7 | 2,412.1† | 2,809.4† | 2,498.3 | 2,769.8 | 2,488.6 | 2,711.0 | 2,018.0 | 2,555.2 |
| 16K | 16,275 | 2,770.3 | 2,714.5 | 2,771.8 | 2,599.3 | 2,834.5 | 2,538.2 | 2,830.0 | 2,264.2 | 2,711.8 |
| 32K | 32,424 | 2,736.2 | 2,664.4 | 2,767.0 | 2,578.8 | 2,766.3 | 2,492.8 | 2,759.3 | 2,334.2 | 2,674.0 |
| 64K | 64,987 | 2,686.4 | 2,543.3 | 2,680.0 | 2,484.4 | 2,659.6 | 2,485.6 | 2,639.1 | 2,242.2 | 2,550.7 |
| 128K | 129,949 | 2,598.6 | 2,451.4 | 2,544.4 | 2,399.0 | 2,522.2 | 2,398.9 | 2,512.4 | 2,179.0 | 2,500.4 |
| 256K | 258,992 | 2,480.7 | 2,318.8 | 2,362.3 | 2,273.7 | 2,348.0 | 2,270.0 | 2,344.0 | 2,073.1 | 2,329.1 |

Decode, tok/s (MTP on):

| prompt | tokens | iQ-MLX 4.7 bpw | K2 Sushi | K2 turboexl3 | K3 Sushi | K3 turboexl3 | K4 Sushi | K4 turboexl3 | K5 Sushi | K5 turboexl3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1K | 1,031 | 97.1 | 126.2 | 131.5 | 131.8 | 140.0 | 119.7† | 141.0† | 126.8 | 134.9 |
| 2K | 2,013 | 92.4\* | 119.6 | 123.6\* | 126.6 | 131.3\* | 114.2† | 134.8† | 122.0\* | 126.5\* |
| 4K | 4,025 | 100.1 | 113.3 | 125.3 | 124.5 | 135.2 | 116.5 | 132.3 | 123.6 | 138.2 |
| 8K | 8,100 | 102.2 | 118.2† | 127.6† | 128.2 | 134.4 | 116.1 | 135.7 | 123.1 | 126.6 |
| 16K | 16,275 | 101.7 | 110.7 | 120.6 | 126.6 | 132.3 | 112.5 | 137.6 | 122.3 | 124.2\* |
| 32K | 32,424 | 110.3 | 107.5 | 117.8 | 117.8 | 128.5 | 108.7 | 126.4 | 114.3 | 122.8 |
| 64K | 64,987 | 96.6 | 108.5 | 110.9 | 112.0 | 116.6 | 102.5 | 119.5 | 107.8 | 113.6 |
| 128K | 129,949 | 96.6 | 98.9 | 101.3 | 106.3 | 110.4 | 96.8 | 114.3 | 103.6 | 108.9 |
| 256K | 258,992 | 87.4 | 91.8 | 96.0 | 93.6 | 99.4 | 89.6 | 105.5 | 92.3 | 95.2 |

MTP off (`--no-mtp`), tok/s:

| prompt | tokens | prefill, K5 Sushi | prefill, K5 turboexl3 | decode, K5 Sushi | decode, K5 turboexl3 |
|---|---:|---:|---:|---:|---:|
| 1K | 1,031 | 1,014.1 | 1,157.9 | 56.8 | 55.9 |
| 2K | 2,013 | 1,439.9 | 1,578.1 | 56.0\* | 55.0\* |
| 4K | 4,025 | 1,756.9 | 2,173.5 | 56.2\* | 55.1\* |
| 8K | 8,100 | 2,211.2 | 2,609.6 | 55.9 | 54.9 |
| 16K | 16,275 | 2,277.3 | 2,751.4 | 55.8 | 54.9 |
| 32K | 32,424 | 2,290.3 | 2,698.0 | 55.1 | 54.2 |
| 64K | 64,987 | 2,269.8 | 2,604.9 | 54.2 | 53.2 |
| 128K | 129,949 | 2,203.3 | 2,503.1 | 53.1 | 52.0 |
| 256K | 258,992 | 2,094.2 | 2,345.8 | 51.9 | 51.2 |

### The standard setup at fixed MTP depths

The iQ-MLX 4.7 bpw pack on the same binary and prompts, at `--mtp-max-depth 3` and `4` with `--mtp-typical 0.2`, beside
its run at the default depth. It decodes 7 to 30% faster at depth 3 than at its default, so the decode columns above
overstate the EXL3 lead at equal depth.

Decode, tok/s (MTP on):

| prompt | tokens | depth 3 | depth 4 | depth 6 (default) |
|---|---:|---:|---:|---:|
| 1K | 1,031 | 125.9 | 98.2 | 97.1 |
| 2K | 2,013 | 118.4\* | 105.4\* | 92.4\* |
| 4K | 4,025 | 120.9 | 100.3 | 100.1 |
| 8K | 8,100 | 124.5 | 102.7 | 102.2 |
| 16K | 16,275 | 117.5 | 111.9 | 101.7 |
| 32K | 32,424 | 120.2 | 110.8 | 110.3 |
| 64K | 64,987 | 109.7 | 99.3 | 96.6 |
| 128K | 129,949 | 103.6 | 100.8 | 96.6 |
| 256K | 258,992 | 95.6 | 94.7 | 87.4 |
| per-draft acceptance, mean | | 86.1% | 84.9% | 81.5% |
| tokens per round, mean | | 3.47 | 3.71 | 4.13 |

At equal depth, the turboexl3 arms against the standard setup:

- **Depth 3:** K2 is level, from 2.2% slower to 4.4% faster. K3 is 4.0 to 12.6% faster, and K4 5.2 to 17.1% faster.
- **Depth 4:** K5 is 0.5 to 37.8% faster than the standard setup.

The Sushi arms at the same depth: K2 is 10.6% slower to 1.0% faster, K3 2.1% slower to 7.7% faster and K4 3.6 to 9.6%
slower at depth 3, and K5 2.5% slower to 29.1% faster at depth 4. These ranges leave out the runs marked `†`.

## Coding evals

EvalPlus 0.3.1: HumanEval+ (164 problems) and MBPP+ (378), greedy pass@1, seed 1234, EvalPlus's chat prompt,
`reasoning_effort: low` with `max_tokens` 16,384, and MTP with typical acceptance (`--mtp-typical 0.2`) on every setup.
Every answer but one finished on its own: the 4.05 pack's answer to Mbpp/56 reached the 16,384-token cap, and the code
in it still passes the base and the extra tests.

| Setup | HumanEval | HumanEval+ | MBPP | MBPP+ |
|---|---:|---:|---:|---:|
| 2.05 pack, turboexl3 4a4ea95 as the quant plugin of #997's head, reordered numerics, MTP depth 4 | 0.970 (159) | 0.927 (152) | 0.958 (362) | 0.807 (305) |
| 3.05 pack, turboexl3 4a4ea95 as the quant plugin of #997's head, reordered numerics, MTP depth 4 | 0.982 (161) | 0.939 (154) | 0.950 (359) | 0.820 (310) |
| 4.05 pack, turboexl3 4a4ea95 as the quant plugin of #997's head, reordered numerics, MTP depth 4 | 0.988 (162) | 0.963 (158) | 0.958 (362) | 0.796 (301) |
| 5.05 pack, turboexl3 4a4ea95 as the quant plugin of #997's head, reordered numerics, MTP depth 4 | 0.988 (162) | 0.945 (155) | 0.955 (361) | 0.812 (307) |
| iQ-MLX 4.7 bpw (the standard pack), upstream main + the two load switches, MTP auto | 0.988 (162) | 0.945 (155) | 0.958 (362) | 0.815 (308) |
| uncalibrated mixed 4/8-bit, superseded by the iQ pack: upstream main + the two load switches, MTP auto | 0.988 (162) | 0.945 (155) | 0.952 (360) | 0.815 (308) |

The four packs are within two points of the standard pack (iQ-MLX 4.7 bpw) on every suite. The largest gap is the 4.05
pack on MBPP+: 301 against 308 problems, 1.9 points. With one greedy run per setup, gaps of this size are within the
noise: paired by problem against the standard pack, the smallest exact McNemar p-value is 0.14. The 5.05 pack and the
standard pack are within one problem of each other on every suite. Their HumanEval answers differ (90 of 164 solutions
identical) but fail mostly the same problems: 7 of the 9 HumanEval+ failures are shared.

![EvalPlus pass@1 on HumanEval, HumanEval+, MBPP and MBPP+ for the 2.05, 3.05, 4.05 and 5.05 packs and the standard iQ-MLX 4.7 bpw pack](charts/coding_evals.png)

## Kernels against Sushi

turboexl3's routed-expert kernels against [Sushi](GLOSSARY.md#sushi)'s, one MoE layer call at a time, at each
[rate](GLOSSARY.md#rate) from K2 to K8, with no server. Sushi runs in the same test binary, on the same weights, inputs
and routing. The binary is built from turboexl3 f1d82d8 (4a4ea95's library plus the commit that adds these timing arms)
on mlx-serve's host, and its sha256 starts d5b06e858592. Measured on an Apple M5 Max (128 GB, macOS 27.0.1) on
2026-10-28.

![Decode kernels: turboexl3 against Sushi at each trellis rate](charts/kernel_decode.png)

![Prefill kernels: turboexl3 against Sushi at each trellis rate](charts/kernel_prefill.png)

### What each arm runs

- **form 9** (`f9`): the served decode form: form 8's four launches at 1 row (the prepare inside the gate|up launch),
  form 7's five launches from 2 rows, each with 2 output tiles per GEMV threadgroup.
- **form 3** (`f3`; the log's `lib`): the plain five-launch chain. It is the reference for the bytes, and it is not
  served.
- **form 7** (`f7`): form 3 with the finishes' H128 transforms in registers. It is not served.
- **Sushi, decode**: Sushi's decode arm (`sushi_exl3.moe` at decode rows).
- **turboexl3, prefill**: the arm `m64dufiajn4ql2h`, `pfOptions` at the default configuration, called through
  `Prefill.moeT128` as `moePrefill` serves it.
- **Sushi, prefill**: Sushi's prefill arm.

Every decode arm times the routed experts alone. At 2 to 4 rows, mlx-serve's served forward also runs the gated shared
expert, with the router's top-k, inside form 9's launches; these timings leave that work out for both engines.

Captured routing comes from mlx-serve serving the 5.05 pack on a 17,743-token coding prompt, the standard test cell,
which prefills as two 8192-row chunks, then a 1328-row chunk. Each decode call takes consecutive tokens' experts from
one MoE layer's routing over an 8192-row chunk; the prefill timings replay one MoE layer's routing over an 8192-row
chunk, then over the 1328-row chunk. The 40 row-slots of a 4-row decode call hold 28.2 distinct experts on average.
Synthetic routing: each row draws 10 distinct experts at random (38.8 distinct in 40 row-slots, on average). From 2 rows
up, turboexl3 pairs the row-slots that chose the same expert under one threadgroup, which reads and decodes those
weights once.

### Method

Qwen3.8-Flash-Next's MoE layer: hidden 2560, expert width 640, 512 experts, top-10, bf16 activations, MUL1 codebook,
random trellis codes at each rate; one process per rate. Decode: each round runs a chain of 48 dependent calls per arm
(each call's output is the next call's input, and each call routes to fresh experts, so every call reads its weights
from DRAM) and divides by 48; 31 rounds after 2 warm-up rounds. Prefill: each round times 4 back-to-back calls per arm
and divides by 4; 15 rounds after 2 warm-up rounds. The arms take turns in every round, with a rotated start. Before
each decode timing, one call of each arm is compared: forms 7 and 9 gave form 3's bytes in every cell, and turboexl3's
output differs from Sushi's by at most 2.7% of the RMS of Sushi's output (the two engines round in different orders).

### Decode, captured routing

Ratios are Sushi's time over the turboexl3 arm's: above 1, turboexl3 is faster. "median" divides the two arms' median
rounds; "fastest" divides their fastest rounds (the log's own `*_vs_sushi`).

| K | rows | form 9 µs, median | form 9 µs, fastest | Sushi µs, median | Sushi µs, fastest | form 9 ratio, median | form 9 ratio, fastest | form 7 ratio, median | form 3 ratio, median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K2 | 1 | 70.7 | 70.4 | 72.8 | 72.6 | 1.030 | 1.031 | 0.977 | 0.935 |
| K2 | 2 | 113.0 | 112.6 | 119.1 | 119.0 | 1.054 | 1.057 | 0.973 | 0.951 |
| K2 | 3 | 153.7 | 153.3 | 171.8 | 171.5 | 1.118 | 1.119 | 1.029 | 1.009 |
| K2 | 4 | 189.9 | 189.5 | 220.0 | 219.7 | 1.159 | 1.159 | 1.031 | 1.016 |
| K3 | 1 | 74.3 | 74.0 | 74.3 | 74.0 | 1.000 | 1.000 | 0.961 | 0.922 |
| K3 | 2 | 118.3 | 117.9 | 122.0 | 121.6 | 1.031 | 1.031 | 0.970 | 0.951 |
| K3 | 3 | 162.4 | 162.1 | 176.2 | 175.9 | 1.085 | 1.085 | 1.024 | 1.009 |
| K3 | 4 | 199.3 | 199.0 | 225.6 | 225.4 | 1.132 | 1.133 | 1.034 | 1.025 |
| K4 | 1 | 69.9 | 69.4 | 86.2 | 85.8 | 1.233 | 1.236 | 1.179 | 1.124 |
| K4 | 2 | 112.0 | 111.4 | 152.6 | 152.3 | 1.363 | 1.367 | 1.260 | 1.235 |
| K4 | 3 | 152.9 | 152.5 | 216.8 | 216.5 | 1.418 | 1.420 | 1.309 | 1.290 |
| K4 | 4 | 188.0 | 187.5 | 280.6 | 280.3 | 1.493 | 1.495 | 1.336 | 1.324 |
| K5 | 1 | 76.8 | 76.3 | 78.1 | 77.7 | 1.017 | 1.018 | 0.986 | 0.949 |
| K5 | 2 | 123.9 | 123.6 | 129.6 | 129.4 | 1.046 | 1.047 | 0.997 | 0.978 |
| K5 | 3 | 169.6 | 169.3 | 185.3 | 184.8 | 1.093 | 1.092 | 1.042 | 1.027 |
| K5 | 4 | 207.0 | 206.7 | 235.0 | 234.6 | 1.135 | 1.135 | 1.040 | 1.032 |
| K6 | 1 | 82.4 | 82.0 | 83.1 | 82.9 | 1.008 | 1.011 | 0.999 | 0.963 |
| K6 | 2 | 138.1 | 137.8 | 138.8 | 138.5 | 1.005 | 1.005 | 0.986 | 0.971 |
| K6 | 3 | 186.1 | 185.6 | 193.9 | 193.4 | 1.042 | 1.042 | 1.021 | 1.008 |
| K6 | 4 | 231.4 | 224.1 | 252.8 | 241.8 | 1.092 | 1.079 | 1.026 | 1.017 |
| K7 | 1 | 93.4 | 93.1 | 109.0 | 108.4 | 1.167 | 1.164 | 1.162 | 1.125 |
| K7 | 2 | 159.6 | 159.2 | 183.3 | 182.9 | 1.148 | 1.149 | 1.122 | 1.110 |
| K7 | 3 | 212.6 | 212.0 | 259.5 | 259.1 | 1.221 | 1.222 | 1.192 | 1.179 |
| K7 | 4 | 269.9 | 258.9 | 350.6 | 333.7 | 1.299 | 1.289 | 1.213 | 1.208 |
| K8 | 1 | 105.8 | 105.4 | 106.0 | 105.8 | 1.002 | 1.004 | 0.992 | 0.967 |
| K8 | 2 | 178.4 | 178.1 | 173.2 | 172.8 | 0.971 | 0.970 | 0.964 | 0.952 |
| K8 | 3 | 235.7 | 235.3 | 237.2 | 236.9 | 1.006 | 1.007 | 0.996 | 0.986 |
| K8 | 4 | 282.8 | 282.2 | 291.9 | 291.3 | 1.032 | 1.032 | 0.999 | 0.991 |

### Decode, synthetic routing

| K | rows | form 9 µs, median | form 9 µs, fastest | Sushi µs, median | Sushi µs, fastest | form 9 ratio, median | form 9 ratio, fastest | form 7 ratio, median | form 3 ratio, median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K2 | 1 | 70.8 | 70.6 | 73.3 | 73.1 | 1.035 | 1.035 | 0.983 | 0.939 |
| K2 | 2 | 120.4 | 119.9 | 119.6 | 119.4 | 0.993 | 0.996 | 0.905 | 0.883 |
| K2 | 3 | 168.3 | 168.1 | 172.1 | 171.8 | 1.023 | 1.022 | 0.928 | 0.913 |
| K2 | 4 | 217.0 | 216.8 | 221.3 | 221.1 | 1.020 | 1.020 | 0.911 | 0.901 |
| K3 | 1 | 74.6 | 74.4 | 74.9 | 74.7 | 1.004 | 1.004 | 0.969 | 0.929 |
| K3 | 2 | 125.8 | 125.6 | 122.8 | 122.6 | 0.976 | 0.976 | 0.903 | 0.884 |
| K3 | 3 | 178.7 | 178.2 | 177.6 | 177.3 | 0.994 | 0.995 | 0.925 | 0.912 |
| K3 | 4 | 228.5 | 228.2 | 227.5 | 227.1 | 0.996 | 0.995 | 0.913 | 0.905 |
| K4 | 1 | 70.8 | 70.4 | 85.4 | 85.2 | 1.206 | 1.210 | 1.160 | 1.108 |
| K4 | 2 | 120.8 | 120.0 | 151.6 | 151.4 | 1.255 | 1.262 | 1.154 | 1.131 |
| K4 | 3 | 169.2 | 168.9 | 215.5 | 215.3 | 1.274 | 1.275 | 1.165 | 1.149 |
| K4 | 4 | 215.0 | 214.6 | 279.0 | 278.7 | 1.298 | 1.299 | 1.176 | 1.166 |
| K5 | 1 | 79.7 | 79.5 | 81.0 | 80.8 | 1.016 | 1.016 | 1.001 | 0.963 |
| K5 | 2 | 139.5 | 139.2 | 137.9 | 137.6 | 0.989 | 0.989 | 0.948 | 0.933 |
| K5 | 3 | 197.6 | 197.3 | 198.4 | 198.1 | 1.004 | 1.004 | 0.967 | 0.955 |
| K5 | 4 | 248.9 | 248.4 | 251.7 | 251.3 | 1.011 | 1.012 | 0.951 | 0.943 |
| K6 | 1 | 88.1 | 87.8 | 88.5 | 88.4 | 1.005 | 1.007 | 1.001 | 0.965 |
| K6 | 2 | 158.0 | 157.7 | 152.9 | 152.6 | 0.968 | 0.968 | 0.956 | 0.943 |
| K6 | 3 | 224.3 | 224.0 | 219.0 | 218.6 | 0.976 | 0.976 | 0.969 | 0.959 |
| K6 | 4 | 285.8 | 285.2 | 278.9 | 278.5 | 0.976 | 0.977 | 0.953 | 0.946 |
| K7 | 1 | 98.3 | 98.1 | 111.6 | 111.4 | 1.135 | 1.136 | 1.128 | 1.093 |
| K7 | 2 | 181.6 | 181.3 | 193.7 | 193.4 | 1.067 | 1.067 | 1.060 | 1.051 |
| K7 | 3 | 255.8 | 255.3 | 279.8 | 279.2 | 1.094 | 1.094 | 1.080 | 1.070 |
| K7 | 4 | 326.3 | 325.9 | 358.2 | 357.8 | 1.098 | 1.098 | 1.062 | 1.055 |
| K8 | 1 | 111.1 | 110.8 | 109.9 | 109.6 | 0.989 | 0.989 | 0.988 | 0.963 |
| K8 | 2 | 202.6 | 202.2 | 194.9 | 194.5 | 0.962 | 0.962 | 0.954 | 0.943 |
| K8 | 3 | 289.5 | 289.0 | 281.2 | 280.9 | 0.971 | 0.972 | 0.966 | 0.957 |
| K8 | 4 | 370.2 | 369.8 | 364.0 | 363.2 | 0.983 | 0.982 | 0.968 | 0.963 |

### Prefill, captured routing

| K | rows | turboexl3 µs, median | turboexl3 µs, fastest | Sushi µs, median | Sushi µs, fastest | ratio, median | ratio, fastest | turboexl3 TFLOPS, median | Sushi TFLOPS, median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| K2 | 8192 | 19682 | 19338 | 20008 | 19682 | 1.017 | 1.018 | 40.9 | 40.2 |
| K3 | 8192 | 19935 | 19732 | 21159 | 20686 | 1.061 | 1.048 | 40.4 | 38.1 |
| K4 | 8192 | 19799 | 19660 | 20634 | 20329 | 1.042 | 1.034 | 40.7 | 39.0 |
| K5 | 8192 | 20067 | 19964 | 26244 | 25527 | 1.308 | 1.279 | 40.1 | 30.7 |
| K6 | 8192 | 20168 | 19919 | 26692 | 26098 | 1.323 | 1.310 | 39.9 | 30.2 |
| K7 | 8192 | 20773 | 20619 | 27084 | 26682 | 1.304 | 1.294 | 38.8 | 29.7 |
| K8 | 8192 | 20662 | 20448 | 23971 | 23653 | 1.160 | 1.157 | 39.0 | 33.6 |
| K2 | 1328 | 4524 | 4508 | 4859 | 4592 | 1.074 | 1.019 | 28.9 | 26.9 |
| K3 | 1328 | 4729 | 4683 | 5467 | 5150 | 1.156 | 1.100 | 27.6 | 23.9 |
| K4 | 1328 | 4507 | 4486 | 5269 | 4886 | 1.169 | 1.089 | 29.0 | 24.8 |
| K5 | 1328 | 4808 | 4792 | 9450 | 8484 | 1.965 | 1.770 | 27.2 | 13.8 |
| K6 | 1328 | 4987 | 4975 | 9874 | 9415 | 1.980 | 1.892 | 26.2 | 13.2 |
| K7 | 1328 | 5436 | 5426 | 10298 | 9621 | 1.894 | 1.773 | 24.0 | 12.7 |
| K8 | 1328 | 5702 | 5693 | 7390 | 7090 | 1.296 | 1.245 | 22.9 | 17.7 |

### Timing the kernels again

Build turboexl3's tests over a consumer's surface with `-Dtest-filter=timing` (see [the README's
Tests](../README.md#tests)). Run the binary once for each rate, with n = 16 × K:

```sh
# decode: forms 3, 7 and 9, then Sushi; rows 1 to 4; synthetic routing, then the captured routing
TURBOEXL3_GPU=1 TURBOEXL3_DEC_TIME=1 TURBOEXL3_DEC_NS=<n> TURBOEXL3_DEC_ROWS=1,2,3,4 TURBOEXL3_DEC_ROUNDS=31 \
  TURBOEXL3_DEC_INDS=<routing of an 8192-row call> <test binary>
# prefill: the served arm, then Sushi
TURBOEXL3_GPU=1 TURBOEXL3_PF_TIME=1 TURBOEXL3_PF_N=<n> TURBOEXL3_PF_E=512 TURBOEXL3_PF_ARMS=m64dufiajn4ql2h,sushi \
  TURBOEXL3_PF_ROUNDS=15 TURBOEXL3_PF_INDS=<routing of an 8192-row call>,<routing of a 1328-row call> <test binary>
```

A routing file holds the `u32 [rows, 10]` expert indices of one prefill call, as `Config.pf_capture_dir` writes them.
