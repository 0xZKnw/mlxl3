# Shared native model operators

This component supplies strided F16-output RMS normalization, dense row
projection, single-query grouped attention and half-split RoPE for the native
quantized Qwen/LFM blocks. All inputs retain their Metal storage. The encoders
check geometry, device ownership and allocation extents before submission.

[checks.json](checks.json) records source and binary hashes, commands, numerical
results and limits. The production encoder is exercised by a release libtest on
the physical M5 with 24 GiB, macOS 27.2 build 26B123. Its independent oracle runs
MLX 0.32.2 in a separate process and exits before native execution.

The final matrix has 62 numerical cases and four explicit unsupported-shape
rejections. Every one of the 82,149 numerical FP16 words is finite and identical.
Inputs use offsets and negative column strides. It covers RMS widths
64/128/2048/2560 with F16/F32 gains and one/three rows, dense F16 weights at
2048→1/32/64/256 with one/three rows, F32 weights with one row, GQA head widths
64/128/256 at prefixes 1/4/31/32/33/257, and full/partial RoPE at positions
0/1/4096/131071. Every output is explicitly F16, including promoted F32-gain RMS.

F32-weight dense batches remain unsupported. The first attempted rowwise
implementation disagreed with the oracle's batched GEMM. The native API now
rejects this new combination before allocation. The old failure stays in
physical-04 and raw F32/BF16 teacher support remains open. This component does
not remove a previously supported product path.

Reproduce with a direct `--features chat` release libtest and
`python tests/physical_model_ops.py --test-binary TEST_BINARY --output NEW_DIRECTORY`.
The driver enforces deadlines, checks the exact selected-test count, rejects
missing or unequal outputs, audits loaded libraries and keeps parity false on
failure. Nine deterministic CPU driver tests cover its failures and cleanup.
The generated local fixture is hashed; its large raw JSON is not committed.
The generator, fixed seed and exact oracle version are committed.

The three Rust suites pass 304/311/347 tests with 8/26/247 ignored, respectively
CPU/direct/MLX compatibility. Strict Clippy, formatting, native release, Metal
AIR and Python checks pass. Kani with Z3 checks the real GQA width helper on
three arbitrary u32 inputs, without assumptions: 221 obligations, eight standard
intrinsic checks unreachable and both covers reached, with no failure. The
earlier CaDiCaL run timed out at 60 seconds. Full-library Kani reproduces the
existing macOS compiler ICE. CrossHair finds no checkable process-driver
functions. These checks do not prove Metal arithmetic, concurrency, allocation
or whole-model generation. Other Apple hardware and complete family integration
remain separate qualification work. Throughput is not measured.

The retained failures identify the corrected boundaries: mixed-gain RMS promotes
its intermediate dtype; standard fast RoPE uses host-rounded log2(base), exp2
frequencies and fast trigonometry. These follow the tagged upstream
[RMS frontend](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/fast.cpp),
[RMS kernel](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/rms_norm.metal)
and [RoPE kernel](https://github.com/ml-explore/mlx/blob/v0.32.2/mlx/backend/metal/kernels/rope.metal).
