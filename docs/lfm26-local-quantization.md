# Local LFM2.5-2.6B conversions

## Current status — 2026-09-06

The user paused the four-variant batch and then requested **only 4.0 bpw**.
That conversion is complete: 166/166 projections, no missing source tensors,
all MCG markers present, about 1.76 GB. It is registered as
`lfm2.5-2.6b-4bpw` and published at
https://huggingface.co/0xzknw/LFM2.5-2.6B-EXL3-4bpw
(commit `61393eb7da551ab4af03831e9b69bdfd723a3cbb`).

Validation: WikiText-2 test 2048-token perplexity 73.28046 vs BF16 72.71838;
WinoGrande64 36/64 vs BF16 35/64; French arithmetic, JSON and Python CLI
smoke tests passed. The installed Desktop engine also loaded all 166
projections and produced the correct arithmetic response. KL was not measured.
Logs/results: `build/lfm26-quants/4bpw`. This is limited validation, not an
exhaustive quality assessment. Other precisions remain paused; do not restart
the all-variants script without another user request. No automation is active.
The conversion helpers and optimization report are now included in this repository;
model weights and local benchmark artifacts are not.

## Optional converter setup

Conversion is a developer workflow, separate from Desktop/CLI inference. It
requires the pinned PonyExl3 checkout in addition to MLXL3's normal environment:

```sh
git clone https://github.com/beamivalice/PonyExl3.git references/PonyExl3
git -C references/PonyExl3 checkout 8e7fa6b1556f59fc669e25087903b279b9b0346f
git -C references/PonyExl3 apply ../../scripts/patches/ponyexl3-bounded-conversion-cache.patch
```

Do not reapply the patch to an already patched checkout or replace an existing
development checkout. The LFM adapter installs the optimized search itself;
it does not require changes to Desktop or the inference runtime.

## Original four-variant recipe (paused)

The local converter now enables the exact Metal optimization described in
[Metal quantization optimization](metal-quantization-optimization.md).
Existing published weights are unchanged; the four-variant run remains paused.

Source: `LiquidAI/LFM2.5-2.6B`, revision
`654f9463ce32b05d0429d76fe1f580b27d4c1ac0`.
All 11 source files passed `hf cache verify` on 2026-09-05.

Run/resume sequentially (do not launch two copies):

```sh
bash scripts/quantize_lfm26_all.sh
```

Targets: 4.1, 5, 3 and 8 bpw. Results live in
`models/LFM2.5-2.6B-EXL3-<bits>bpw`; logs, resumable work and quality results
live in `build/lfm26-quants/<bits>bpw`.

The local PonyExl3 adapter quantizes all 166 LFM linear projections with
Metal LDLQ, MCG codebook, 2048 calibration rows from the bundled WikiText-2
excerpt, two 1024-token sequences, sigma regularization 0.025 and one worker.
The 4.1-bpw allocator measures K=4/5, keeping gate/up siblings at the same K.
Other targets use their integer K. Calibration capture is shared between runs.
There is no training or QAT.

BPW describes quantized linear weights, not the entire checkpoint: tied
embeddings, norms and convolution kernels remain unquantized (FP16 export).
Each conversion verifies complete tensor coverage and runs WinoGrande64 plus
a CLI chat smoke test. BF16 reference: `build/lfm26-bf16-winogrande64.json`,
35/64 correct. This small sample is a sanity check, not a capability guarantee.

The pilot exposed missing `.mcg`/`.mul1` tensor markers in Pony's exporter.
The adapter now emits the same int32 marker values as ExLlamaV3; without
them MLXL3 interprets the packed weights with the wrong codebook. A regression
test covers all three codebook choices. Corrected pilot projection errors
against BF16 were 0.70% / 0.96% relative RMS on 32 captured rows, and the
hybrid BF16/two-EXL3-layer model answered the chat smoke test correctly.

Pilot bundles are incomplete and are **not usable standalone models**.
No output is registered, uploaded, or published automatically. Inspect each
completed conversion's evaluation and chat logs before registering it.
