# Review of the consolidated Studio image worker

At 512 × 512 (1,024 image tokens), the corrected worker uses the window kernel. Other row counts use the original 32-row kernel. Two-step qualification matched the two saved finite denoiser states and the complete reference image at both 512 and 1024 px.

## Paired 512 px comparison

In scalar / window / window / scalar order, full generations took 27.84, 18.51, 21.81, 29.97 s (means 28.91 and 20.16 s). Peak process footprint 15.3483 and 15.3489 GB, maximum RSS 4.6189 and 4.6129 GB. All four files and decoded RGB buffers matched the historical 512 px four-step sample. To reproduce the scalar control, override the row-tile selection in EXL3Linear with `block_rows, group = (32, 1)`. Stage both workers separately and use the 512 px teapot reproduction command in [the memory report](README.md#512-px-at-40-steps) with `--steps 4` and a fresh output path per run. The two workers differed only in the kernel used at 1,024 rows. Both performed the same checksum, hardware and finiteness checks and used the same phase-release callback. Each timed run started a fresh process after a two-step warmup per variant. The 1024 px qualification took place before the four timed runs. macOS process counters are in decimal GB. These timings measure four denoising steps.

The 40-step output matched the published M5 teapot byte for byte, SHA-256 `7b018ca8fc9418ba2d6659cc6909a0d013dde0baaa12558fdb9fd710c1cd87cd`. It took 153.19 s with 15.20 GB peak footprint. This was a correctness check on the corrected worker. [Per-run results, prompts, worker/shader hashes and power readings](review-kernels.json).

The physical Mac had an Apple M5 and 24 GiB of unified memory, running macOS 27.3. Studio's uv bootstrap resolved Python 3.12.14, MLX 0.32.4, MFLUX d39c9639, transformers 5.24.0, huggingface_hub 1.33.0, numpy 2.5.5, Pillow 13.0.0 and tokenizers 0.23.3. The package cutoff is 2026-11-22 20:00 UTC. Model files were already present and network access for weight retrieval was disabled. No temperature reading or GPU-only peak was available.

## Eager evaluation of the prompt

The standard, eager, eager and standard workers took 25.98, 27.29, 37.19, 32.26 s (means 29.12 and 32.24 s). Peak process footprint 15.2772 and 15.2722 GB, maximum RSS 4.6187 and 4.8808 GB. All four files and decoded RGB buffers matched the historical 512 px four-step sample. Paired footprint savings 0.0044 and 0.0057 GB. The result did not meet the gate for a 40-step confirmation. Prototype rejected. The two two-step runs also matched the saved finite denoiser states and entire PNG. [Per-run data](review-prompt.json).

The eager prototype added `mx.eval(*model._encode_prompt(args.prompt))` immediately before `generate_image`. Both workers used the existing MFLUX prompt cache. The two paired peak reductions were about 4.4 and 5.7 MB.
