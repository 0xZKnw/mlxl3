# Third-party notices

## Flash Next architecture and SSD integration

`native/src/qwen4.rs` and its independent verification script follow Qwen's
released Flash Next configuration and the MIT-licensed MLX-VLM 0.7.3 text
architecture, including HyperConnections, sparse attention and PLE.
`native/shaders/flash_rope.metal` adapts the text half-split MRoPE body from
MLX-VLM `models/rope_utils.py` under the same MIT license:
<https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen4_exp>.
Copyright © 2025 Prince Canuma. The MIT license is provided in
`LICENSES/MLX-VLM-MIT.txt`. Existing EXL3 kernels retain their PonyExl3
attribution and Apache-2.0 license.

The SSD design was compared with the user's `localqwen3.8flash` project and
Sushi <https://github.com/beamivalice/sushi> at
`eaaca12c5508e6a24a01671384a9a41089370058` (HC, QSA, EXL3 and n-gram kernels).
Sushi kernel source has not been copied; its compiled bank/n-gram format and
optional approximate expert selection are not used by this implementation.

MLXL3 is an independent implementation that adapts algorithms and kernel
structures from the following projects.


MLXL3 is an independent implementation that adapts algorithms and kernel
structures from the following projects.

## MTPLX

Source: <https://github.com/youssofal/MTPLX>, revision
`9882703f3105363ddc37eca9f97aa09a1d387112`, Apache-2.0
(`LICENSES/Apache-2.0.txt`). Qwen MTP pre-normalization residual feedback,
device-side recursive drafting, accepted-history KV repair and warm candidate
tuning informed the independent Rust implementation. EXL3 target kernels and
exact greedy verification remain MLXL3 implementations. Upstream performance
numbers are not MLXL3 measurements.

The optional full-vocabulary affine4 draft projection and output-chunked
conversion in `scripts/experimental-draft-head.py` are independently implemented,
informed by `mtplx/draft_lm_head.py`. EXL3 reconstruction/rotations are MLXL3's
implementation; no upstream timing is treated as a local result.

The optional `native/shaders/mtp_add_rms.metal` prototype adapts
`mtplx/kernels/fused_norm.py` at that revision, preserving the rounded
residual and MLX RMSNorm reduction order. Copyright 2026 Youssof Altoukhi,
Apache-2.0. Its 5120-wide variant keeps both groups of lane values in
registers; eligibility is limited to FP16 Qwen MTP rows1..4. The app's
Settings displays "Powered by MTPLX" with the source link.

`native/src/router.rs` adapts the two-stage SIMD row-owned top-8 structure
from `mtplx/qwen_row_owned_router.py` to FP16 monotone keys, legacy tie order
and half accumulation. Copyright MTPLX contributors; Apache-2.0. The small
EXL3 verify batching uses MLXL3's own trellis kernels and arithmetic.

## OpenBMB LES

Source: <https://github.com/OpenBMB/llama.cpp-omni>, revision
`a64932ec615e1463158129feaf2f0752d0c07a2e`, especially
`docs/les/lookahead-expert-scheduling.md`, `src/models/qwen35moe.cpp`, and
`src/llama-moe-slot.cpp`. Cross-layer routing prediction and expert loading
overlapped with current-layer computation informed MLXL3's independent
experimental Rust implementation. It uses the existing EXL3 banks and
checked readers, preserving actual routing and reduction order. OpenBMB's
quantization and DSpark results are separate from MLXL3 measurements.

## Strata

Sources: <https://github.com/Niko1221/Strata>, revision
`cbfd5b926b794b356195d3cb600191b0c3567540`, and the distinct MLX project
<https://github.com/yibie/strata-mlx>, revision
`295c96594dd79cbd080052da8740b1a8e44d7d76`.
Their GPU expert planning, grouped weight reuse, streamed-cache scheduling
and host/GPU overlap informed the independently implemented Flash Next
resident-routing experiment. The comparison of implemented and outstanding
mechanisms is recorded in `docs/flash-next-ssd-strata-56-2026-10-20.md`.
MLXL3 preserves its EXL3 arithmetic and routing contract; Strata's source
and performance results are references, not bundled kernels or local timings.

## TensorFold

Source: <https://github.com/ashhart/TensorFold>, revision
`cb2ebf0540f42604e2759b2ddef497861e928248`, Apache-2.0
(`LICENSES/Apache-2.0.txt`). Its draft-only vocabulary pruning and grouped target-layer submission
informed independent Q4 row selection, native token-ID remapping and MLX
graph submission. The public
`src/tensorfold/families/qwen4_exp/cuda/draft_vocab.txt` list was used for local
measurements; neither that list nor upstream implementation code is bundled.
The target retains its full vocabulary and exact verification.

## Sushi

Source: <https://github.com/beamivalice/sushi>, revision
`cac6a1ed9284bcd13609eaf80ceed74e0db6cea3`, MIT (`LICENSE` at that revision).
The bounded prompt-lookup and cross-line matching concepts informed an
independent Rust implementation. Proposals still use MLXL3's exact target
verification and accepted-history KV repair. No Sushi code or kernel is bundled;
upstream throughput is not an MLXL3 measurement.

## IncoAI Splash

Source: <https://github.com/inco-ai/splash>.

MLXL3 reads Splash's public DFlash 2 draft package ABI and uses its published
geometry as a compatibility reference. Runtime kernels remain independently
benchmarked MLXL3 implementations and may differ from Splash. Splash is
licensed under the Apache License, Version 2.0; a copy is provided in
`LICENSES/Apache-2.0.txt`.

## SGLang DFlash2 and DSpark

The experimental Flash Next draft tests in `native/src/flash_draft.rs` and
the independent NumPy oracle in `scripts/check_flash_drafts.py` implement
the DFlash2 and vanilla DSpark equations documented by SGLang, revision
`e00930c5489053f26d86b179cee0d087f846acbb`:
<https://github.com/sgl-project/sglang/tree/e00930c5489053f26d86b179cee0d087f846acbb/python/sglang/srt/models>.
SGLang is licensed under Apache-2.0 (`LICENSES/Apache-2.0.txt`).
The locally downloaded IncoAI DFlash2 and shadowpa0327 DSpark checkpoints
are not bundled; their respective model cards document their weight licenses.

## ExLlamaV3

Source: <https://github.com/turboderp-org/exllamav3>, pinned during development
at `ca5270c4b842876ddbe9a28594fbb6eac516cdf2`.

The EXL3 serialized format, trellis codec, procedural codebooks, permutation,
Hadamard reconstruction, and CUDA kernel algorithms are derived from
ExLlamaV3.

MIT License

Copyright (c) 2025 Turboderp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## PonyExl3

Source: <https://github.com/beamivalice/PonyExl3>, pinned during development at
`8e7fa6b1556f59fc669e25087903b279b9b0346f`.

Copyright 2026 Theinruj Toranavikrai

The native Metal QMV/QMM shaders incorporate modified and independently
integrated kernel structures from PonyExl3. MLXL3 changes the
dispatch API, compile-time specialization, permutation embedding, split-K
policy, row crossover, output layout, and integration/tests. PonyExl3 is
licensed under the Apache License, Version 2.0; a copy is provided in
`LICENSES/Apache-2.0.txt`.

The optional converter in `src/mlxl3_quantizer/metal.py` also adapts PonyExl3's Metal
trellis-search source. MLXL3 modifies predecessor sharing, exact codebook
lookups, backpointer storage, search ordering, synchronization and scratch
batching. The bounded allocator-cache modification is distributed separately
in `scripts/patches/ponyexl3-bounded-conversion-cache.patch` under the same
Apache-2.0 terms. These conversion helpers are not used by inference.

## MLX Swift LM

Source: <https://github.com/ml-explore/mlx-swift-lm>, pinned during development
at `5694a2f6705f7c8b9cf195f29ec6d05938d42d22`.

The stable single-row MoE router top-k algorithm in the native Metal runtime is
an adaptation of MLX Swift LM's `MoERouterTopK.swift`. MLXL3 adds its
own runtime dispatch and EXL3 integration. MLX Swift LM is licensed under the
MIT License.

Copyright (c) 2024 ml-explore

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## MLX and MLX LM

Sources: <https://github.com/ml-explore/mlx>, version 0.32.2, and
<https://github.com/ml-explore/mlx-lm>, version 0.32.0.

The packed Gated DeltaNet Metal kernel in
`native/shaders/gated_delta_packed.metal` is carried from MLX LM for exact
behavior during the native Rust migration. MLX LM is licensed under the MIT
License. The dense verification gate in `native/src/lfm2.rs` reproduces MLX's
GEMV reduction geometry so multi-row verification remains bit-exact; MLX is
also licensed under the MIT License. The mixed HC GEMV in
`native/shaders/flash_hc_gemv.metal` adapts the same MLX 0.32.2 GEMV
geometry and accumulation order, with FP16 weight loads and FP32 outputs.

Copyright © 2023 Apple Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
