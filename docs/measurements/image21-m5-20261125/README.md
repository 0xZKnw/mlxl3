# Qwen-Image-2.1 EXL3 4bpw on Apple M5

On one 24 GiB M5, the 512 × 512 Studio worker took **101.21 seconds** to generate the reference image in 40 steps. The baseline took **107.68 seconds**. The optimized worker also used **9.36 GB less peak process memory**. Each worker reproduced the published sample PNG byte for byte.

| Order | Worker | Wall time (s) | Denoising (s/step) | Peak footprint (GB) | Max RSS (GB) |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Optimized | 100.21 | 2.37 | 5.987 | 4.339 |
| 2 | Baseline | 107.05 | 2.52 | 15.347 | 4.735 |
| 3 | Baseline | 108.31 | 2.55 | 15.344 | 4.613 |
| 4 | Optimized | 102.20 | 2.40 | 5.987 | 4.235 |

The two adjacent comparisons saved 6.84 and 6.11 seconds. These figures describe a single Mac, prompt, seed and resolution. Each generation ran in a fresh process with the model and auxiliary files already local. Measurements include loading, denoising, VAE decode and PNG writing. Memory columns use decimal GB from macOS `/usr/bin/time -l`. The four runs used battery power (65% at the start, 62% at the end). Temperature was uncontrolled. Absolute timings varied across successive experiments.

## What changed

The packaged worker limits the MLX allocator's cache of released Metal buffers to 512 MiB for 512 px renders. A diagnostic run found that the baseline peaked during VAE decoding. Immediately after decode it retained 13.71 GB of free buffers while MLX reported 1.02 GB of active arrays. The cap alone reduced mean 40-step footprint from 15.35 to 5.89 GB, with wall time moving from 103.74 to 104.94 seconds.

For 1024 image-token rows, the Metal window kernel now processes 32 or 64 input features per multiply for six measured projection shapes and bit widths. It uses the existing decoded tile and float accumulator. The standalone kernel screen reduced the weighted projection time by 9.41%. The standalone 40-step comparison had mixed results. Only the measured combination cleared the full-generation speed and memory criteria. The worker uses the previous path at 768 and 1024 px.

## Correctness and scope

- Baseline `a2060bae795d6ef6fffdcaa9663f19e7197bb842`, macOS 27.3, Python 3.12.14, MLX 0.32.4, MFLUX `d39c9639`, EXL3 recipe A at 3.999249 effective bpw.
- Independent reference image SHA-256 `7b018ca8fc9418ba2d6659cc6909a0d013dde0baaa12558fdb9fd710c1cd87cd`. All four long runs matched both the PNG and decoded RGB. Both tensors from the two-step pilot matched their historical hashes.
- A second prompt about a bicycle, seed 1729, reproduced all four step tensors and the full PNG at four steps.
- The production worker passed 12 CPU tests. The physical Metal oracle covered 36 numerical cases at 3, 4 and 5 bits, rejected six invalid inputs and rejected six non-finite tensors. A separate compilation using the native shader's default depth reproduced 786,432 FP16 values bit for bit across the three bit widths.
- Singleton conditioning rows, caching 4 GiB of reconstructed weights, four-bit word extraction and two changes to tile layout or thread assignment failed their speed criteria. Their results are recorded individually.

The per-run measurements, source hashes, seed, prompts and correctness checks are in the [measurement record](evidence.json). [Research log](../../../opti.md)
