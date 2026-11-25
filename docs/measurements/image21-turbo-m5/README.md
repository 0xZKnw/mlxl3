# Qwen Image 2.1 Turbo on Apple M5

Studio's **Reduce memory use** option at 1024 × 1024 now also covers Qwen
Image 2.1 Turbo on the 24 GiB Apple M5. On a complete eight-step reference
image, limiting the cache during decoding reduced peak process footprint from
18.705 GB to 8.117 GB. The PNG and decoded pixels matched exactly. The setting
is off by default. A change in typical generation time has not been established.

Studio runs the eight-step Turbo schedule in a fresh Python process for each image.
This study compares changes to two parts of that path. At 1024 × 1024, each of the
eight evaluations runs 224 EXL3 projections over 4096 image tokens. At any size,
the loader reads and hashes 225 local weight files before using them. The
reference is the worker shipped with [the Turbo integration](https://github.com/0xZKnw/mlxl3/pull/205),
commit `21305b5f42eca226dac5f6fe76c990fcadcba859`. Its worker, both Metal
shaders, catalog and packaged example matched the installed Studio resources
byte for byte.

## Reference and protocol

The hardware is an Apple M5 with 10 GPU cores and 24 GiB of unified memory.
The production environment uses Python 3.12.15, MLX and mlx-metal 0.32.4,
and MFLUX `d39c9639e6ac22407a725432cc221a83fab19f1b`.
The installed [4-bit EXL3 checkpoint](https://huggingface.co/jyohukuchan/Qwen-Image-2.1-Turbo-EXL3/tree/f362145b8e8ebc88d6639c445eb9f31beab7c0b4/4bit)
uses the Default codebook for all 224 projections. The upstream model revision
is `fc13ba1986fcb905854dc8de2f52eee395f49b5b`, and the quantization manifest
has SHA-256 `60f7f2bd2f679faf6fa1e3423f1538af8ce0a80940c4b7b9f7cb82e015e87924`.
The auxiliary revision is `746a58556820933a2df5c75887a2570f1ad200c0`.

The physical projection oracle compared 48 synthetic cases against independent
NumPy trellis decoding and Hadamard transforms. All passed, including 12 for the
Turbo codebook. The largest Turbo relative L2 error was 0.000591 rounded up
(limit 0.004). Six invalid inputs and six non-finite tensors were rejected.

The image comparison uses seed 987406473 and the prompt
“A studio photograph of a cobalt blue ceramic teapot on a pale peach tabletop,
soft morning light, simple composition.” A qualifying image run must match the
input latent and output prediction for each of eight steps, as well as the
decoded RGB and PNG. At 512 × 512 the previous Studio Turbo run also supplies an
independent archived PNG, SHA-256
`458f4609cc865111dbd506e55528b6b8dda7ed998aaca4d77903c798255a14ea`.
Comparisons are relative to Studio on this Mac. Fidelity to the original
upstream ROCm/BF16 pipeline was not measured.

## Projection selection at 1024 × 1024

Each row below uses a representative matrix from the real checkpoint, with
4096 input tokens. The times include rotation, compressed matrix multiplication,
output rotation and conversion to the requested dtype.

| Projection (input × output) | Count per step | Reference, ms | Selected, ms | Row tile / SIMD groups / operation depth |
|---|---:|---:|---:|---|
| Attention, 4096 × 4096 | 128 | 50.43 | 34.01 | 64 / 4 / 64 |
| MLP expansion, 4096 × 12288 | 64 | 161.92 | 99.02 | 64 / 4 / 32 |
| MLP contraction, 12288 × 4096 | 32 | 136.36 | 136.36 | Original 32 / 1 / 16 |

The three candidate geometries matched all 251,658,240 BF16 values compared
with the reference. Timing used a separate warmup for every candidate and four
counterbalanced orders of three synchronized calls each. Weighted by the actual
counts, the median projection times sum to 21.18 seconds per step in the reference
and 15.05 seconds with the selected geometries. Individual rounds varied.
The full-worker comparison below determines the shipping decision.

The first 1024 × 1024 Studio baseline completed in 177.80 seconds under state
instrumentation and reached a physical process footprint of 18.705 GB. The
kernel-only candidate matched every input latent and prediction across the
eight steps. During VAE decoding, macOS signaled critical memory pressure.
The supervisor stopped that worker before a PNG was produced. The eight matched
steps establish denoiser parity for this sample. The interrupted render supplies
no end-to-end speed or image result. [Projection samples](projection-screen.json)
and [initial 1024 px run records](initial-1024.json).

## Weight verification at startup

The three implementations authenticated all 225 files, totaling 3,766,700,960
bytes, in manifest order. With the files already in the macOS page cache, the
sequential check took 1.368, 1.397 and 1.372 seconds. Four readers took 0.644,
0.666 and 0.645 seconds. The median saving was 0.727 seconds. Two readers saved
0.507 seconds and missed the 0.7-second threshold for a full-image trial.

Six complete 512 × 512 images matched the archived RGB and PNG. The two
instrumented pilots also matched all sixteen intermediate tensors. The four
fresh uninstrumented workers took 23.75 s (sequential), 24.29 s (four readers),
28.64 s (four readers) and 34.70 s (sequential). The candidate lost 0.54 s in
the first pair and gained 6.06 s in the second. Its peak footprint was 0.380 GB
and 0.195 GB higher, respectively. The reference times drifted by 46%.
The shipping worker keeps the sequential integrity check.
[Per-run verification and image results](verification-512.json).

At the same 512 × 512 resolution, we also tested three direct Metal kernels
with 32, 64 or 128 rows per tile. All nine comparisons matched 62,914,560
reference values. Even the fastest direct geometry took 7.50, 24.56 and
27.42 ms on the three projection shapes. The existing window kernel took
5.55, 17.13 and 17.67 ms. Studio therefore keeps its existing 512 px kernels.
[All 144 timed projection calls](direct-tiles-512.json).

## Decoder memory at 1024 × 1024

The next two candidates limited MLX's cache for released Metal buffers to
512 MiB just for the VAE decode. The worker restored the previous limit after
the finite-pixel check, including when decoding raised an exception. The
denoising steps ran under the original allocator settings.

| Instrumented pilot | Worker time | Peak process footprint | Image |
|---|---:|---:|---|
| Original worker | 177.80 s | 18.705 GB | Reference |
| Decoder cache limit | 161.44 s | 8.117 GB | All intermediate states and pixels matched |
| Decoder cache limit and selected projection kernels | 176.94 s | 8.112 GB | All intermediate states and pixels matched |

The decoder limit on its own qualified for the uninstrumented timing block.
The additional kernels failed the pilot selection criterion. The footprint is
macOS physical process memory in decimal GB. The 512 MiB setting affects freed
MLX buffers and allows larger live tensors when required.

Two attempts to complete a four-run timing block stopped during the first
uncapped run. Another process started during the first attempt. In the second,
all eight steps finished in 3 min 19 s before macOS signaled critical memory
pressure during decoding. Neither attempt produced a PNG. This leaves one
complete baseline pilot for the memory comparison and no completed timing
pair at 1024 px. [Pilots and both interrupted timing blocks](decode-1024.json).

To use the tested setting in Studio, select Turbo, choose 1024 px and enable
**Reduce memory use**. The worker CLI accepts `--variant turbo --size 1024 --steps 8 --low-memory`.
The text encoder, eight denoising steps and checkpoint verification follow the
existing path. Base and Turbo remember independent choices while the view is open.
[French Studio screenshot at 1120 × 772 points](turbo-1024-memoire-fr.png).

## Integration checks

The tested Desktop package was version 2.0.32 (build 85), built from
`2c4c761522c980f1641ba4154a7290fb7b1c298c` on top of
`df1d4411521bd369c62fa5f0a894876f59e0cc15` from the
[Base memory control PR](https://github.com/0xZKnw/mlxl3/pull/209). The 19 Python worker
tests cover 72 CLI configurations. Six inventory tests passed, and the
reviewed MLX call inventory matches all 351 files. The Swift controller and
history fixture passed 78 assertions, and the SwiftUI fixture exercised both
positions of the switch in both variants. Removing the Turbo offering made
the UI regression test fail.

CrossHair 0.0.114 explored the production Python wrapper with a modeled
allocator, an integer previous limit and a decoder that can return or raise.
It confirmed the restoration contract and found a counterexample when restoration
was removed. The actual Metal allocator and its concurrency are outside that
model. [Restoration contract](check_restore.py).

## Packaged worker and Studio gallery

With `--low-memory` enabled, the packaged worker completed all eight steps at
1024 × 1024. Each of the eight incoming latents and eight denoiser predictions
matched the reference and the decoder-only prototype byte for byte. Each tensor
contained 262,144 finite BF16 values, with shape `[1, 4096, 64]`.
The peak physical process footprint was **8.111 GB** (8,111,311,632 bytes),
below the 10 GB qualification threshold. The maximum resident set size was
**4.846 GB** (4,846,354,432 bytes). Both values are macOS process measurements.
The 187.64 s wall time includes state instrumentation and belongs to this
single qualification run. [Final worker and Studio records](final-option.json).

The production Swift image controller then started a fresh offline worker with
the same settings through Studio’s usual uv launcher. It generated a 1,079,260-byte
PNG and automatically saved it to an isolated gallery. A new controller reopened
the gallery and verified the prompt, seed, variant, model name, dimensions and
step count. The saved PNG matched the reference SHA-256
`be53936baaafd930fb930dc27daf3399a8930d552a9a1f721b37e9b03e145c72`.
All 3,145,728 decoded RGB bytes matched as well, SHA-256
`c6ba408c4dd81c3769c8362640b129df5140fdd449ac0ad6c825bdb6a1188d9e`.
The controller run measured image and gallery correctness. Peak memory was
measured separately in the packaged-worker run. [Controller command, settings and output](final-option.json).

The PR’s rebased code commit is `a1d819d38c9eae790978aad611c93b16315917bd`
on base `820d77e6daa23f61de99a980a1a718349bf625cd` of PR #209. The
worker SHA-256 is `fcebbc7e50c0130c950964a7e0a98fd3544cbe6ad8095d4860567763347b2a42`
in both code commits and in both physical runs. Both shaders, the model catalog,
the packaged sample and the Swift image controller source also match byte for
byte. The newer base includes a change to the SwiftUI save-panel code.
The two physical runs above used the package built from `2c4c7615`.
[Per-commit source hashes](final-option.json).

The local test app was rebuilt from `a1d819d3` as Desktop 2.0.33 (build 86)
with the verified 2.0.36 engine. Its signature passed strict verification.
The entire Image21 resource directory matched the physically tested package.
The updated controller and UI fixtures passed, including gallery export and
cancellation. The rebuilt executable has SHA-256
`02f3c2d8fc9a5f3dfa13cd8fd05a921a7cdd0642cb5230c51887bd4d29a917b4`.
