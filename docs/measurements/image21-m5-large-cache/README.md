# Qwen-Image-2.1 allocator cache at 768 and 1024 px on Apple M5

**IMAGE21-M5-LARGECACHE-19, 25 November 2026.** On an Apple M5 with 24 GiB, limiting MLX's free buffer cache to 512 MiB saved **11.65 GB** of peak process footprint at 768 × 768 and **10.62 GB** at 1024 × 1024 in a pair of 40-step renders at each size. At 768 px the capped worker finished 23.87 s earlier. At 1024 px it took 75.10 s longer. At each size both workers produced the same PNG and decoded pixels. [Individual runs and source identities](evidence.json)

The final Studio worker uses the cap automatically at 768 px on this configuration. At 1024 px the default uses MLX's ordinary allocator policy. To trade render time for a smaller footprint, select **Reduce memory use** (**Réduire la mémoire**) in the image settings or pass `--low-memory` to the worker. Both paths use the same checkpoint, projection kernels and diffusion schedule. The existing 512 px setting applies across the supported M5 family. [Worker](../../../apps/MLXL3Studio/Resources/Image21/render.py) · [512 px measurements](../image21-m5-20261125/README.md)

## Workers and environment

| Item | Identity |
| --- | --- |
| Baseline A | Packaged Studio worker from `458098f2709094db97e497f0047f01a886228957`, default allocator policy at 768 and 1024 px |
| Experimental B used in all A/B comparisons | Same revision with a local change calling `mx.set_cache_limit(512 * 1024**2)` before model loading at 768 and 1024 px on an Apple M5 with 24 GiB of physical memory |
| Worker SHA-256, A / B | `febcbd7b3e4e7be327dea414cd8208a2f7555bf2c927ccbf79827c2bbd2adffb` / `edfe9e35c575980e890c12de1b0c61dff7216ed0da949ed14ae36cdbec8d2476` |
| Final packaged worker C | Selects the same 512 MiB cap automatically at 768 px and by explicit choice at 1024 px on the tested configuration. Worker SHA-256 `1bdda0db7f85ad97732aa80d7e66413ee3e300f2bec3454ba02b4de6e122f7d7` |
| Metal resource SHA-256, all three workers | `_qmm_tensor_kernel.metal` = `86628ed8519e83490f0424f3dbede3eefd484b871ff112cb19bbace3670e4e33` and `_qmm_tensor_window_kernel.metal` = `df02356bc223e0b0afa0d05eb31fcd3b42f6806846d9809f391bcdecbebb306e` |
| Hardware and software | Apple M5, 24 GiB unified memory, macOS 27.3, Python 3.12.14, MLX 0.32.4, MFLUX 0.24.0 at `d39c9639e6ac22407a725432cc221a83fab19f1b`, Transformers 5.24.0, NumPy 2.5.5, Pillow 13.0.0, huggingface_hub 1.33.0 |
| Checkpoint | Local Qwen-Image-2.1 EXL3 recipe A, 33 shards, 150 four-bit and 74 three-bit projections, 3.999249 effective bpw. Manifest SHA-256 `1f61ed6a36d89530d43946a89a3c45faad7a120acfac357ffd25b131a3f399e0` |
| Auxiliary snapshot | `mlx-community/Qwen-Image-2.1-mflux-q4` at `746a58556820933a2df5c75887a2570f1ad200c0` |

The production entry point is the Python worker copied into Studio's resources. Both workers load local weights, use MFLUX's normal component-release callbacks and generate an RGB PNG. The memory limit applies to reusable buffers that MLX's allocator holds after release. Live tensors, allocator cache, maximum resident set size (RSS) and macOS peak process footprint are measured separately.

## Historical parity and allocator observations

The oak prompt was **“An old oak tree beside a quiet river in autumn light.”**, seed 371, two diffusion steps, A then B at each size. A pilot wrapper observes the full return value of each denoiser forward call (`Qwen21Transformer.__call__`), materializes it as float32 and checks shape, finiteness and variation before hashing. The two step outputs have distinct hashes. Both workers matched both archived hashes and the complete PNG and RGB hashes at each size. The archive contains separate historical runs of the oak fixture. [QMM-02 oracle](../image21-m5-20261122/results.json)

| Width and height | Forward tensor per step | SHA-256 of decoded RGB | SHA-256 of PNG |
| --- | --- | --- | --- |
| 768 px | 1 × 2,304 × 64, 147,456 values | `ade6f641069e1fa22e5465ae5d24fc9d9aa254e59b6152a44794b5a14e1966c8` | `5032713d2b5a4811696baef1064842157c7f8034e3253d21fa08f41501c3bc93` |
| 1024 px | 1 × 4,096 × 64, 262,144 values | `636a5e67c336ce9e920ae74cc3d224af4fb8eee57ca03a8a5a5b8e573c26e967` | `acd94db8121923fe6349b468a4dbf501d48ab5ec8b415d6c0943c27febe34b04` |

The following observations come from the instrumented two-step runs. All memory columns use decimal GB. The second-step and VAE readings are samples from MLX. The footprint and RSS columns are lifetime process peaks from macOS `/usr/bin/time -l`. All four runs used battery power, from 71% to 69% across both sizes. Temperature was unmeasured.

| Size | Worker | Cache after second forward | Cache before VAE | Cache after VAE | Active after VAE | Peak process footprint | Max RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 768 px | A | 1.280 | 0 | 16.774 | 1.023 | 18.711 | 3.863 |
| 768 px | B | 0.555 | 0 | 0.425 | 1.038 | 7.293 | 4.246 |
| 1024 px | A | 1.605 | 0 | 16.689 | 1.031 | 18.724 | 4.729 |
| 1024 px | B | 0.570 | 0 | 0.377 | 1.057 | 8.110 | 4.236 |

The largest sampled difference in allocator cache occurs after VAE decode. The uninstrumented renders described below provide the end-to-end timing comparison.

## Four- and forty-step protocol

The second prompt uses seed 1729 and the exact text below.

> Photographie prise au crépuscule dans une ruelle enneigée de Strasbourg. Un vélo rouge est appuyé contre une porte en bois bleu foncé. Au premier plan, la neige fraîche porte les traces fines des pneus. Une fenêtre éclairée et sa petite jardinière occupent le tiers supérieur droit. Lumière douce, sans texte.

Each render starts a fresh offline worker on the same GPU with the same checkpoint. The two pilot workers per resolution also warm the system. The four-step screen uses A/B/B/A order. Its two adjacent pairs must each save at least 0.5 decimal GB of peak process footprint, with mean B wall time at most 1.05 times mean A wall time. Only a size passing the screen proceeds to a separate 40-step pair, B then A, under the same criteria. Every timed output must match the other runs at that size and step count in both PNG and decoded RGB. The physical worker's wall time includes Python startup, model loading, denoising, VAE decoding and saving to disk. The denoising progress log reports a separate seconds-per-step estimate.

### Timed results

The four-step rows show the mean of two fresh workers per arm, and list the footprint savings separately for each adjacent pair. The 40-step rows show one fresh worker per arm. All footprint values are decimal GB. [Exact timings, max RSS, denoising rates, hashes and power readings](evidence.json)

| Size | Steps | Wall time A → B (s) | Wall change | Peak footprint A → B (GB) | Savings per pair (GB) | Meets both criteria |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 768 px | 4 | 47.51 → 43.14 | −9.20% | 18.730 → 7.257 | 11.441, 11.506 | Yes |
| 768 px | 40 | 452.17 → 428.30 | −5.28% | 18.654 → 7.000 | 11.654 | Yes |
| 1024 px | 4 | 83.65 → 82.83 | −0.99% | 18.807 → 8.127 | 10.771, 10.589 | Yes |
| 1024 px | 40 | 858.14 → 933.24 | +8.75% | 18.714 → 8.099 | 10.615 | No, wall time exceeded +5% |

All twelve timed runs matched the other runs from their size and step count in PNG and RGB hashes. Timing at 1024 px drifted even within the four-step screen. In run order the workers took 73.81, 78.25, 87.40 and 93.49 s. The first capped worker took 4.44 s longer than its adjacent baseline and the second took 6.09 s less. The timed series ran on battery from 69% to 40%, with temperature unmeasured. The repeated memory savings support the resolution-specific choice on this Mac. One long pair per size and one timed prompt leave sustained latency on other workloads open.

## Verification of the final selection

The final worker passed all 13 CPU tests with zero skips, including 36 resolution, hardware and opt-in combinations exercised through its argument parser and entry point. Five temporary mutations were detected, including an always-on 1024 px cap and a worker ignoring the flag. Ruff, byte compilation and comparison of the four staged package resources passed. The A/B measurement harness separately passed 11 CPU tests covering divergence, incomplete data and process cleanup. [Worker tests](../../../tests/test_image21_worker.py)

CrossHair 0.0.114 ran in a disposable environment. Its PEP 316 counterexample search exercised the actual Python `main` through entry to model loading, with the three accepted resolutions, integer memory capacity, known or missing capacity, base M5 or M5 Pro, and the opt-in boolean. Parsed options, the GPU and allocator calls were supplied by the harness. CrossHair reported “Confirmed over all paths” for C under this model. On B it found the unwanted 1024 px cap with 25,769,803,776 bytes, base M5 and the option off. This check covers the selection and ordering of the allocator call. Actual Metal allocation and image correctness were exercised by the physical runs. [CrossHair harness](check_cache_contract.py)

From the repository root, run the following command in an environment with `crosshair-tool==0.0.114`. To repeat the negative control, set `IMAGE21_PROOF_WORKER` to the path of the packaged B worker before running the command.

```sh
crosshair check --analysis_kind=PEP316 --report_all --per_condition_timeout=45 docs/measurements/image21-m5-large-cache/check_cache_contract.py
```

The Studio Swift fixture compiled with warnings as errors and passed 28 history assertions plus hardware, launch and progress preconditions. The subprocess fixture checks that the selected option reaches the worker only for 1024 px on the eligible machine and that the request captures the setting at launch. The SwiftUI toggle is limited to this configuration. [Swift fixture](../../../tests/image-generation-check.swift) · [Studio controls](../../../apps/MLXL3Studio/Sources/MLXL3Studio/ImageGenerationView.swift)

On the baseline commit, GitHub's packaging check, Python worker tests and Studio image-history test passed. The Regression workflow failed on native coverage and skipped the later Image21 lint step. The Native Rust workflow also failed. Locally the inventory reported 187 pre-existing differences across 12 Rust or Metal files. The reported worker inventory matched the baseline. [Baseline Regression run](https://github.com/0xZKnw/mlxl3/actions/runs/45647450368) · [Baseline Native Rust run](https://github.com/0xZKnw/mlxl3/actions/runs/45647450455)

### Physical checks of the final app

The signed local **MLXL3 Desktop Test** bundle (2.0.31, build 84) contains worker C and the same two Metal shaders used in the comparison. The three final two-step oak pilots reproduced both historical denoiser hashes and the PNG and RGB hashes. The after-decode MLX readings show which cache policy actually ran. [Final worker, bundle and Studio evidence](final-validation.json)

| Final worker setting | Peak process footprint (GB) | MLX allocator cache after VAE (GB) |
| --- | ---: | ---: |
| 768 px, automatic | 7.035 | 0.425 |
| 1024 px, default | 18.706 | 16.843 |
| 1024 px, Reduce memory use | 7.987 | 0.377 |

Studio’s actual Swift image controller launched C, archived the 768 px oak image with its prompt, dimensions, step count and seed, and reopened the gallery successfully. It also cancelled and joined a second worker after that worker reported progress. The test used a fresh gallery and a launcher pointing the interpreter discovery step to the pinned offline Python environment. The initial sandboxed SwiftPM build could not access its cache, and a later build was interrupted before packaging. The resumed release build succeeded and the bundle passed strict, recursive macOS signature verification.

Additional 1024 px, 40-step confirmations of the **final optional setting** stopped after 14 and 37 completed steps when another image renderer started. Both attempts are recorded as interrupted and produced no PNG. The complete 40-step times and matching image hashes above belong to the paired A/B campaign. Physical image parity for C was established in the two-step pilots. The measurements cover the base EXL3 checkpoint. During the campaign the separately installed Studio acquired a different worker with Turbo support. This test bundle is a separate build of the base-model checkout.

### Integration with the updated Studio launcher

The measurement artifacts identify the sources used in the original campaign. For the pull request on `84ca5375`, the 1024 px switch was combined with Studio’s newer `--prompt=<text>` argument handling. The repackaged worker, both shaders and the bundled sample were byte-identical to C. On this base, 13 worker tests and all 6 native-inventory tests passed. The Swift subprocess fixture passed its 29 history assertions and its hardware and launch preconditions, including the literal prompt `--low-memory` with the option both enabled and disabled. An isolated mutation that split the prompt into two arguments failed this fixture. CrossHair repeated the worker-selection check under the model described above. The archived A/B measurements and physical checks apply to their recorded builds.

The Studio image settings show the 1024 px choice unchecked by default. Both screenshots use the bundled example image and isolated preferences. [Default setting at 1120 × 760 pt](qwen-image-memory-1024-normal.png) · [Enabled setting at 820 × 740 pt](qwen-image-memory-1024-enabled.png)

## Reproduction

The parent workspace retains the original A and B packages, six sets of logs, and the individual PNGs under `build/image21-large-cache`. The portable measurement file records all 16 A/B workers, their commands and resource hashes. To stage the final policy for a new render, use a new directory from the final checkout. [Portable measurement file](evidence.json)

```sh
WORK_DIR=$(mktemp -d)
sh "$FINAL_CHECKOUT/scripts/copy-image21-resources.sh" "$WORK_DIR/C"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1
```

The local harness expects the Image21 venv and historical artifacts in sibling campaign directories. From the parent workspace, use a fresh series name for each independent reproduction. Set `SIZE` first to 768 and then to 1024.

```sh
CAMPAIGN=build/image21-large-cache
PYTHON=build/image21-studio/venv/bin/python
"$PYTHON" "$CAMPAIGN/run.py" pilot "$SIZE" --series replay
"$PYTHON" "$CAMPAIGN/run.py" short "$SIZE" --series replay
"$PYTHON" "$CAMPAIGN/run.py" long "$SIZE" --series replay
```

To generate the 1024 px bicycle with the final reduced-memory setting, use the pinned Python environment, the local recipe-A checkpoint, the exact bicycle prompt above and a fresh output path. Omit `--low-memory` to reproduce the default 1024 px policy. At 768 px the worker selects the cap automatically on the tested hardware.

```sh
/usr/bin/time -l "$PYTHON" -I -u "$WORK_DIR/C/render.py" \
  --model "$MODEL_DIR" --size 1024 --steps 40 --seed 1729 \
  --prompt "$PROMPT" --output "$OUT_PNG" --low-memory
```

The local campaign directory also contains `conditional_cache_contract.py` and the CrossHair output, the Python and Swift fixture logs, and the four packaged C resources. [Experiment log](../../../opti.md)
