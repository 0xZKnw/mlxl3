# Qwen-Image-2.1 EXL3 on M5: component lifetimes

Studio runs one Python process per image. The optimized worker registers the pinned MFLUX MemorySaver for this single generation. It drops the text encoder after prompt encoding and the transformer before VAE decoding. The callback runs without an MLX cache limit. The model loader remains reusable by callers that build models for multiple generations.

The earlier study tested changes inside the EXL3 matrix multiplication. This trial uses the exact same shaders, model shards, steps, seeds and tokenizer for both arms. Hugging Face describes freeing large pipeline components by inference phase. On this unified-memory Mac the worker can release the finished component altogether. [Diffusers memory guide](https://huggingface.co/docs/diffusers/optimization/memory) · [Pinned MFLUX implementation](https://github.com/mflux-community/mflux/blob/d39c9639e6ac22407a725432cc221a83fab19f1b/src/mflux/callbacks/instances/memory_saver.py)

| Size | Worker | Wall times (s) | Mean denoising (s/step) | Mean peak footprint (GB) | Mean maximum RSS (GB) |
|---|---|---|---:|---:|---:|
| 1024 | baseline | 59.36 / 69.60 | 16.52 | 19.09 | 9.35 |
| 1024 | candidate | 59.25 / 63.22 | 15.94 | 18.83 | 4.61 |
| 512 | baseline | 17.39 / 16.41 | 2.31 | 18.67 | 8.56 |
| 512 | candidate | 14.82 / 14.68 | 2.25 | 15.35 | 4.61 |

Each pair had identical decoded RGB pixels. Before timing, we compared both finite denoiser outputs at 512 and 1024 px and every pixel to the validated prior worker, and observed both callback releases on the actual execution path. Workloads used the published teapot prompt and seed 987406473 at 512 px and the oak prompt and seed 371 at 1024 px, as in the previous study. MLX 0.32.4, one model at a time, offline auxiliary files. The Mac reported AC power and a discharging battery during testing. No temperature measurement. Each worker ran under macOS /usr/bin/time -l. Maximum RSS and peak footprint are separate counters. The latter is the more relevant measurement of total memory pressure; the RSS change must not be read as an equivalent saving of system memory.

Only the earlier kernel experiment used a 40-step sustained workload. The two repeats in this trial give a screening result for latency. [Raw runs and hashes](phase-memory.json).

## SIMD Hadamard kernel screen

The EXL3 worker rotates 128 values before and after a matrix product. The prototype combines the scale, transform and cast in a single Metal dispatch for each direction. Each SIMD group processes 128 values with four contiguous elements per lane, using the same FP16 rounding boundaries as the pinned MLX operator. Its layout differs from the existing Qwen text-model kernel, which permutes the output. [MLX Hadamard transform](https://ml-explore.github.io/mlx/build/html/python/_autosummary/mlx.core.hadamard_transform.html)

The component oracle checked 18 direction/width/dtype combinations. All were bit-exact, including the padded input path. At 129 and 4,096 rows all six actual projection geometries matched the current worker on finite outputs. At 4,096 rows the six-shape sum of median timings weighted by checkpoint frequencies moved from 13,608.6 to 13,587.9 ms. Three geometries won all three timing pairs, so a separate candidate uses fusion exclusively for those large inputs. The 512 and 768 px image-token counts fall below its threshold. [Per-shape microbenchmarks](hadamard-screen.json)

## Reproducing the memory measurements

Stage the reference worker from commit `441fce82dc19057b068227bb0294f8b1018f597f` and the candidate worker separately using `sh scripts/copy-image21-resources.sh <destination>` in each checkout. Reuse the Python 3.12 environment and model files for both. All measurements use ordinary image generation with local files and an empty output path for each run.

```sh
HF_HUB_OFFLINE=1 /usr/bin/time -l "$PYTHON" "$WORKER" \
  --model "$MODEL_DIR" --size 1024 --steps 3 --seed 371 \
  --prompt 'An old oak tree beside a quiet river in autumn light.' --output "$OUT_PNG"
```

The candidate is qualified separately with synchronized per-step checks. Timed runs execute without that instrumentation. macOS 27.3; M5 with 24 GiB; checkpoint manifest, worker hashes and all warmups are recorded in the JSON. Quality checks on one Mac and sampled prompts do not establish a formal numerical proof. CrossHair was unavailable.

The selective fusion passed the two 1024 px two-step image oracles (oak and teapot), including all four denoiser states. At 1024 px and three steps with the oak prompt, wall times were 76.43 and 89.04 s for phase release alone and 76.20 and 84.56 s for phase release plus the fused transform. Means were 82.74 and 80.38 s. The first pair differed by only 0.23 s, and absolute times drifted substantially. This screen does not establish an incremental end-to-end speedup. The selective kernel is archived as a prototype. The local Studio change uses the upstream component-release callback only. [Image timings and hashes](hadamard-images.json).
