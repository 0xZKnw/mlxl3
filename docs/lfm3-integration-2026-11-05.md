# LFM3 text and vision integration — 2026-11-05

## Delivered locally

LFM3-1.7B and LFM3-VL-2B now load and generate through the native MLXL3
engine. The CLI accepts repeated `--image` inputs, the bridge accepts image
attachments, and the OpenAI-compatible API accepts text/image content parts.
Desktop image import persists the image bytes and sends them to the bridge.
Supported image encodings are PNG, JPEG and WebP; remote image URLs are rejected.

This is a **local source/build change**. The installed Desktop application has
not been replaced; no GitHub push, release or remote CI run was performed.
The checkout already contained substantial unrelated modifications, preserved
throughout. Initial tracked source SHA:
`1ac910d0a40efcad049c6e5a8cf081d770723465` plus the existing dirty worktree.
Pre-edit copies of touched files are in
`build/lfm3-integration-20261105/baseline/`.

The text implementation reuses LFM2 projections/convolutions and adds LFM3
attention gates, full-head partial RoPE, sliding-window masking and sandwich
normalizations. The separate vision path implements patch ordering, axial RoPE,
bidirectional attention, gated depthwise convolution, spatial merge and the
projector. Input validation bounds decoding, images per conversation, resized
pixels and placeholders; JPEG orientation is applied before preprocessing.

## Sources and quantization results

- Text source: [LiquidAI/LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B),
  revision `0dc4268b561499c648c7ba7c9466bf3d8d01517a`.
- Vision source: [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B),
  revision `71d8e745d97c224984e0f20412f532a5ce9dfd3e`.
- Reference/converter: [ExLlamaV3 1.6.6](https://github.com/turboderp-org/exllamav3/tree/bd1f519172406a9560bd5c90e61000691bb69a07),
  PyTorch 2.10 / CUDA 12.8, one Modal H100. Relevant master architecture and
  converter files matched the wheel used for the original exports. Subsequent
  dev changes did not establish a mandatory reconversion bug for this BF16,
  single-GPU campaign.

Teacher-forced text quality used the official FP32 eager source as the oracle:
one original bilingual passage, 499 input tokens, 498 next-token labels,
including positions beyond the 256-token attention window.

| Text candidate | Mean KL, nats/token | Argmax agreement | NLL change |
|---|---:|---:|---:|
| Unquantized ExLlama control | 0.000216 | 99.80% | -0.003236 |
| Original 4 bpw | 0.033835 | 87.98% | +0.026389 |
| 4 bpw, scaled Hessian | 0.033075 | 89.38% | +0.017708 |
| Mixed 4.48927 bpw | 0.018950 | 91.58% | +0.006095 |
| Original 8 bpw | 0.000243 | 99.20% | -0.000567 |

The mixed recipe uses 24 FFN down projections at 5 bits, 57 attention/conv
output projections at 6 bits and 61 remaining projections at 4 bits. Embedding
and separate head remain FP16. **Projection bpw is not whole-file bpw**.
The owner accepted 91.58%; the original 92% goal remains failed in the raw
report. Negative NLL changes near the numerical control are not improvements.

Vision quality used one synthetic image (64×2,048 features, 24 evaluated text
positions) and eight disjoint CIFAR-10 test images for feature checks.
Calibration used 64 CIFAR-10 training photos resized from 32×32 to 256×256 and
16 seeded synthetic patterns, with per-projection activation Hessians.

| VL candidate | Synthetic feature relative L2 | Text KL | Argmax |
|---|---:|---:|---:|
| Original 4 bpw | 0.342358 | 0.300521 | 87.50% |
| Original 8 bpw | 0.030252 | 0.002599 | 100% |
| Calibrated 4 bpw | 0.114349 | 0.204061 | 95.83% |
| Calibrated 8 bpw | 0.010045 | 0.002094 | 100% |
| Mixed-language 4.48927 bpw + calibrated vision 4 | 0.114349 | 0.308227 | 95.83% |
| Jointly calibrated mixed 4.48927 bpw (R4) | 0.114349 | 0.104194 | 95.83% |
| Image-priority 80/20 mixed 4.48927 bpw (R5) | 0.114349 | 0.158174 | 91.67% |

The calibrated 8-bpw held-out photo feature L2 is mean 0.011077 / maximum
0.014111; it passed its prespecified sampled gates. The earlier low-bit VL
recipes failed the KL≤0.10 gate and remain private. The owner later explicitly
accepted R4 at KL0.104194 for publication despite its narrow goal miss; R5 was
rejected. The raw failed-gate status is retained. This is limited numerical
evidence, not a general language/VQA/OCR/video benchmark.

## Authorized Hugging Face publication

The following exact public commits were verified anonymously after upload:

- [LFM3-1.7B-EXL3](https://huggingface.co/0xzknw/LFM3-1.7B-EXL3/tree/794219432967af4826d8276d370b2d1676d1f870):
  new `4.49bpw/`, 1,874,771,227 weight bytes,
  SHA-256 `d40fb15cf81742d0a1e5f913437d56602b93a041101cce990b320f8496a6d5ea`.
  Existing strict 4/8 variants are unchanged. Twenty-two staged files verified.
- [LFM3-VL-2B-EXL3](https://huggingface.co/0xzknw/LFM3-VL-2B-EXL3/tree/97d5195410f269eb73c155d15cf846d694efcdb6):
  recalibrated `8bpw/`, 2,708,140,652 weight bytes,
  SHA-256 `22e6da660378840af9e5d2aaadec96d66018870cfedef8b9509427658c632457`.
  Legacy public 4-bpw remains but is explicitly marked not recommended.
  Seventeen staged files verified.

A later explicit owner instruction authorized the R4 VL candidate despite the
missed 0.10 gate. Published and anonymously verified: [VL mixed 4.49 bpw](https://huggingface.co/0xzknw/LFM3-VL-2B-EXL3/tree/15d4ec5cc43c5a6146dc675de0d89f939ec9c2da/4.49bpw),
commit `15d4ec5cc43c5a6146dc675de0d89f939ec9c2da`, 26 staged files checked. Weight bytes
1,970,205,068; SHA-256
`c7d7d8da15eec844c2333ac99d65388f995267a2df47f4e3ced3261731873654`. The root card and raw report preserve
KL0.104194/95.83% multimodal argmax, the failed initial goal and owner acceptance,
text-only metrics, calibration recipe and the outstanding native low-bit
feature-parity limit. Existing VL8 and legacy VL4 remain unchanged.

Verification checked public visibility, exact commit/head, file sizes, LFS
SHA-256 and byte-identical non-LFS metadata. Cards include recipes, licenses,
results, raw missed goals, limitations and manifests. A directory download of
the calibrated VL8 initially produced wrong weight data despite matching size;
that copy was rejected before inference. Cloud integrity was rechecked, and a
single-file download restored the exact expected SHA. The rejected copy and
transfer logs were retained.

## Verification of the engine and Desktop path

Commands were run from the repository with Rust 1.98.1, MLX 0.32.2, local
Python 3.12 and the macOS 26.5 SDK. Detailed commands/logs and fixtures are in
`build/lfm3-integration-20261105/`; experiment protocols and failures are in
[`opti.md`](../opti.md).

| Check | Observed result |
|---|---|
| `cargo test --locked --features chat` | 45 library + 15 contract tests passed; 3 ignored |
| `cargo test --locked --features mlx,chat -- --test-threads=1` | 71 passed; 58 ignored |
| `cargo clippy --locked --all-targets --features mlx,chat -- -D warnings` | Passed |
| `cargo fmt --all -- --check`, `git diff --check` | Passed at checked source state |
| Release build with `mlx,chat` | Passed |
| Explicit ignored LFM3 physical-GPU tests | Scalar primitive oracles and four original checkpoint comparisons passed |
| Publication-candidate physical-GPU tests | 2 tests passed; 4 real checkpoints, including new text4.49 and calibrated VL8 |
| Real subprocess bridge | Math, two distinguishable image colors, malformed image rejection and recovery passed |
| Real HTTP server | JSON image response, SSE image response, invalid/remote image rejection and recovery passed |
| `pytest` | 49 passed, 1 skipped (Ling model unavailable) |
| `scripts/check-desktop.sh` with SDK26.5 | Full suite passed, including image import, deduplication, Codable persistence and bridge payload |

The first physical-GPU pass found a real F32-mask/F16-QKV incompatibility;
casting the sliding mask to Q's dtype fixed it and the same checks passed.
The initial Swift build under SDK27.2 failed in existing macro/toolchain code;
SDK26.5 passed. Neither failure was erased from the evidence.

Native versus ExLlama final-logit KL for new text4.49 was 0.00007783
(max absolute error 0.0473633); calibrated VL8 was 0.00008256 (max 0.0585938),
with image-feature relative L2 0.00443477. Outputs were nonempty and finite;
two reset/replays were bit exact. These compare the **same quantized weights**,
separately from source-model fidelity. An extra experimental mixed-VL fixture
failed the native feature L2≤0.01 limit at 0.0165145. Its downloaded references
matched cloud hashes; the failure remains open and does not validate that
private checkpoint.

The two targeted Kani harnesses passed on production Rust helpers:

- `cargo kani --lib --no-default-features --harness lfm3_window_is_causal_and_bounded`:
  symbolic `u32` query/key/window; 4 obligations and 2 covers.
- Patch-index harness: symbolic `u8` dimensions/coordinates/merge factor;
  32 obligations and one cover. This bound does not cover all runtime sizes.

The wider Kani run passed 18 harnesses before a pre-existing Flash harness
exceeded the time budget and was interrupted. The standalone CBMC command
was unavailable; a later attempt found Kani's bundled CBMC 6.11.0 and tried
`mlxl3_array_unary` on the actual C++/MLX source with bounds/pointer/overflow
checks and unwind 8. It failed parsing the macOS libc++ headers before any
property was verified; the tool supports at most C++11, while this source
requires C++20. CrossHair for Python was unavailable. Kani does not prove MLX, Metal, the FFI, image decoding,
concurrency or generation. A temporary window-bound mutation was detected by
the CPU regression test, restored and rerun successfully. Ignored tests remain
unexecuted unless explicitly listed above.

CI was inspected and the Rust workflow now includes a pinned macOS MLX
FFI/protocol compile job on PRs and default-branch pushes. Hosted CI does not
replace the physical-GPU checks. No remote run exists for this unpushed state.

## Further low-bit VL research

[MBQ](https://arxiv.org/html/2412.19509v2) identifies different vision/language
token sensitivities during calibration. ExLlama's
[recipe guide](https://github.com/turboderp-org/exllamav3/blob/bd1f519172406a9560bd5c90e61000691bb69a07/doc/optimize.md)
also argues for matching calibration, sensitivity measurement and deployment
inputs. These motivate a multimodal language-trunk calibration rather than
another blind bitrate increase.

An ablation on the unchanged 24-position oracle found:

| Language / vision | Mean KL |
|---|---:|
| Unquantized / unquantized | 0.000280 |
| Unquantized / calibrated 4 | 0.036901 |
| Mixed4.49 / unquantized | 0.297064 |
| Mixed4.49 / calibrated 4 | 0.307972 |

This attributes most sampled error to the quantized language trunk. The completed R4
trial kept the exact R3 bit allocation and calibrated vision tensors, but
captured language activations on 80 image conversations plus 64×1,024 ordinary
text tokens, with balanced scenario weights. Its heuristic visual-token
weighting is **not an implementation of MBQ**. R4 lowered the initial multimodal
KL by about 66%, from 0.308227 to 0.104194, with 95.83% argmax unchanged.
Eight additional photos had mean text KL0.084882. Text-only VL quality on the
499-token passage was KL0.034147/88.18% argmax/ΔNLL+0.021507.

R5 changed only the scenario mass to 80% image conversations and 20% pure
text. It improved the additional-photo mean KL to 0.057116 but worsened the
initial KL to 0.158174 and argmax to 91.67%, and was rejected. R4 remains the
retained recipe. The first R4 launch failed before quantization because the
calibration harness put a `prefer_cpu` embedding on CUDA while its indexed
image vectors were on CPU; respecting the upstream device preference fixed
the harness. Failed run and successful retry evidence were retained. The
weight file is 1,970,205,068 bytes, SHA-256
`c7d7d8da15eec844c2333ac99d65388f995267a2df47f4e3ced3261731873654`.

Published GGUF comparisons were researched separately:
[ngquocvinh](https://huggingface.co/ngquocvinh/LFM3-1.7B-GGUF#fidelity-measurements)
reports Q4_K_M KL0.078081/top1 86.657%, Q5_K_M 0.025549/92.204%,
Q6_K 0.007199/96.066%, Q8_0 0.001638/98.094%;
[bartowski](https://huggingface.co/bartowski/LFM3-1.7B-GGUF#per-tensor-layouts)
reports an optimized Q4_K_M KL0.0767. Their WikiText/context protocol differs
from our original passage and file sizes differ substantially. These results
do not establish a controlled superiority claim. The official QAD GGUF is a
separate quantization-aware-distilled checkpoint.

## Remaining limits

No general VQA/OCR/video accuracy evaluation, no full Kani suite completion,
no C++/Metal proof, no exact Pillow equivalence proof for arbitrary resized
images, no installed-app GUI image journey and no remote CI run. The local
engine/HTTP/bridge and Desktop payload/import boundaries were exercised;
the research mixed-VL parity discrepancy remains unresolved.


### VL GGUF research and stopped direct comparison

[AtomicChat's VL card](https://huggingface.co/AtomicChat/LFM3-VL-2B-GGUF)
reports text-only mean KL0.061/top1 85.8% for AD-Q4_K_M, 0.078/83.8% for
AD-IQ4_XS, 0.024/90.7% for AD-Q5_K_M, and 0.003/96.6% for Q8_0, on a
30-language neutral set. These differ from our bilingual passage and cannot
be compared directly with the multimodal KL0.104194. Bartowski's VL card
explicitly labels its layout KL table as measured on the normal 1.7B model.
[VelociQ8's VL card](https://huggingface.co/VelociQ8/LFM3-VL-2B-Q8A64-GGUF)
reports text-only WikiText-2 KL0.002580 versus llama.cpp Q8_0 0.002686; no image
quality measurement is claimed there.

A direct comparison was prepared using verified, revision-pinned public GGUF
files and llama.cpp b12088 (`d1044bc82712fbf08996080bef6298b24a27e9c7`). The F16
control produced exact evaluated token IDs and finite outputs, with source
FP32 KL9.9496e-6 on 499 text positions and 4.1584e-5 on the 24 image-conditioned
positions. A CAPI media-marker mismatch failed before image evaluation and
was corrected by setting the marker explicitly. The parser's five tests and
an identity-check mutation passed. These controls validate neither Q4 GGUF
fidelity nor superiority of our quantization.

**The owner explicitly stopped direct GGUF testing. No GGUF Q4 was evaluated.**
Only the published model-card comparisons above are available, with differing
protocols. The later instruction to publish the approximately 0.1-KL VL
candidate is explicit owner acceptance independent of a GGUF superiority claim.
