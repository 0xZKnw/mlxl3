# Qwen-Image-2.1 EXL3 on Apple M5

The Studio image worker applies seven packed projections per transformer block across 32 blocks. At 512 × 512 there are 1,024 latent image tokens. The original Metal path decoded a tile of weights separately for each 32-row block. The selected kernel assigns four 64-row blocks to a threadgroup, reads four codewords from each trellis window and decodes four depth tiles per barrier. The threadgroup holds 8 KiB of decoded tiles in a double buffer. Short inputs below 128 rows use the original kernel. Both paths consume the original EXL3 shards and run the same diffusion schedule.

## Reproduce the comparison

Use reference commit `326265f11baeaf1f1beb44cb8070f28f87e3a017` and the candidate working tree in separate directories. In each checkout, use the dependencies pinned in the worker. Stage the candidate with `sh scripts/copy-image21-resources.sh /tmp/image21-candidate`. For the reference, place its original `render.py` and `native/shaders/_qmm_tensor_kernel.metal` together. Run the following command independently for each worker and choose a different output path for every run.

```sh
HF_HUB_OFFLINE=1 /usr/bin/time -l "$PYTHON" "$WORKER" \
  --model "$MODEL_DIR" --size 512 --steps 40 --seed 987406473 \
  --prompt 'A studio photograph of a cobalt blue ceramic teapot on a pale peach tabletop, soft morning light, simple composition.' \
  --output "$OUT_PNG"
```

The local checkpoint manifest SHA-256 is `1f61ed6a36d89530d43946a89a3c45faad7a120acfac357ffd25b131a3f399e0`. The 224 packed projections comprise 150 at 4 bits and 74 at 3 bits. Each timed run starts a new process with the auxiliary files already downloaded. Wall time includes imports, verification of all shards, prompt encoding, denoising, VAE decoding and PNG output. The progress-bar interval covers the diffusion loop. macOS maximum RSS and peak footprint are reported separately. These are process metrics. GPU allocator peak and temperature were not measured.

## Research and decisions

| Technique | Decision on this checkpoint |
|---|---|
| Decode compressed weights into a tile shared by several SIMD groups | Reuse the repository's existing Metal implementation. The original Studio worker used one SIMD group per tile. Comparable tiling and in-kernel dequantization appear in [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/kernels/mul_mm.metal). |
| Decode four trellis codewords from a window and amortize the barrier | Compare the existing window kernel with the shared tile kernel on all six combinations of dimensions and bit width. Both pass the projection and image checks. |
| Widen the output tile to 64 columns | Exact on all covered forms. The weighted projection microbenchmark was slower than the 64-row, 32-column shared tile, particularly for the 4-bit MLP output projection. |
| Decode temporarily to FP16 and call a dense GEMM | Inspired by [ExLlamaV3's reconstruction path](https://github.com/turboderp-org/exllamav3/blob/16a49792a3c93d8432d72e6c4bce800841566577/exllamav3/modules/quant/exl3.py). All 98,304 reconstructed synthetic weights matched. The Metal GEMM gave different results for input width 12,288. Width 4,096 was exact on tested real inputs, with mixed microbenchmark timings and 32–96 MiB of temporary weights. This version stays out of the worker. |
| Fuse the Hadamard transform with channel scaling | Candidate for a follow-up. The historical Q27P-11 experiment on a text model had mixed end-to-end timings. The transform is part of the quantization contract. [QuIP#](https://arxiv.org/abs/2402.04396) |
| Efficient attention, fused normalization and rotary encoding, prompt-prefix K/V | The pinned [MFLUX transformer](https://github.com/mflux-community/mflux/tree/d39c9639e6ac22407a725432cc221a83fab19f1b/src/mflux/models/qwen21/model/qwen21_transformer) already provides these paths. A dedicated attention profile at higher resolution is a separate experiment. [FlashAttention-2](https://arxiv.org/abs/2307.08691) |
| Reuse predictions across diffusion steps | [TeaCache](https://arxiv.org/abs/2411.19108) is approximate. The benchmark executes every requested step. |

## Numerical checks

The production oracle covers 3/4/5 bits, 128 input and 256 output features, and 1/32/35/127/128/129/255/256/257 input rows. All 27 comparisons pass against NumPy (maximum relative L2 error 0.000620, limit 0.004). The 15 cases on the new path match the old path bit for bit when rerun as short chunks. Six malformed inputs are rejected. The kernel screen also checked actual checkpoint projections at 129 and 1,024 rows for all six shape/precision combinations. The two-step pilot compared both denoiser outputs (each 1 × 1,024 × 64 finite values) and all decoded RGB pixels before timing. CrossHair was unavailable. This is physical GPU testing on one M5.

The [Diffusers projection-fusion implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py) motivated an additional CPU audit of the actual checkpoint. Across all 32 blocks, every one of 128 Q/K, Q/V, K/V and MLP projection/gate pairings has a different FP16 input scaling vector (4,096 finite entries per vector). Any future grouped EXL3 kernel must apply each vector before its own Hadamard transform. Reusing one rotated activation would change the model. Apple's [Metal TensorOps example](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4) describes the cooperative tensor mechanism used for matrix multiplication.

## M5 measurements

Each long run generated the 512 × 512 reference teapot at 40 steps. The order was window, original, shared, window again. The time per step is the progress bar’s final average. The last column is the macOS process footprint, in decimal GB.

| Worker in run order | Wall time | Denoising | Time per step | Maximum RSS | Peak footprint |
|---|---:|---:|---:|---:|---:|
| Selected window kernel | 113.28 s | 01:38 | 2.47 s | 8.45 GB | 18.57 GB |
| Original Studio worker | 249.71 s | 03:57 | 5.94 s | 8.12 GB | 18.57 GB |
| Shared tile without windows | 156.98 s | 02:25 | 3.65 s | 8.41 GB | 18.58 GB |
| Selected window kernel, repeated | 133.75 s | 02:03 | 3.08 s | 8.36 GB | 18.56 GB |

The selected worker averaged 123.52 s against 249.71 s for one new baseline run, a 50.5% reduction. The two candidate times differed by 20.47 s. The tested PNG matched the previously published Studio image byte for byte (SHA-256 `7b018ca8fc9418ba2d6659cc6909a0d013dde0baaa12558fdb9fd710c1cd87cd`). The candidate’s maximum RSS increased slightly relative to the single baseline. Peak footprint was comparable. This sample does not establish a sustained gain at other resolutions or temperatures. The battery percentage declined during the session despite pmset reporting AC power. Thermal readings were unavailable.

For the four-step ABBA, the original took 22.36 and 24.21 s, and the shared tile took 16.93 and 19.30 s. In the second four-step CBBC, the window kernel took 16.23 and 18.78 s, and the shared tile took 17.54 and 18.82 s. Each four-step image matched. The exact size of the incremental window-kernel gain is uncertain. See [per-run data, hashes and projection timings](results.json).

The initial 768 px check at one step stopped on a non-finite denoiser output in the original worker. A CPU reproduction with the pinned linear scheduler returned sigmas [NaN, 0] at one step and [1, 0.01999998, 0] at two steps. At one step its terminal rescaling divides zero by zero. The resolution checks therefore use two steps, the prompt “An old oak tree beside a quiet river in autumn light.” and seed 371. The kernel patch preserves scheduler behavior.

At both 768 and 1,024 px the two finite denoiser states and every pixel matched the original worker. These are two-step correctness checks. Sustained 40-step performance was measured only at 512 px. No GitHub CI run was requested for the local branch. The installed app and the public checkpoint were not modified.

Instrumented two-step runs (original to candidate) measured 768 px: 8.47 to 8.56 s/step, 1024 px: 27.21 to 31.38 s/step. These runs synchronize each denoiser result to the CPU for hashing, with one pass per arm. The 1,024 px candidate was slower in that check.
