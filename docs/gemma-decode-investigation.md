# Gemma 4 EXL3 decode investigation

Local Gemma 4 26B-A4B, EXL3 3.54 bpw, Apple M5. CLI only; no desktop rebuild.
Experiments compare warm greedy streaming of the same prompt, capped at 192
generated tokens, alternating baseline/candidate order across five pairs.
The baseline includes the previously added padded gate/up grouping and compiled
MLP/router graphs. Thus these results are incremental, not gains versus the
original Gemma implementation.

## Findings

- Compiling Experts alone: median paired change -2.0%; not retained.
- Four versus eight SIMD groups for mapped expert projections: negligible gain
  on the first pair, different greedy tokens. Abandoned.
- Compiling the entire post-attention feedforward block: median paired change
  -2.5%; not retained.
- A scaled-Hadamard kernel with the original eight-thread/shared-memory layout
  was slower; abandoned.
- SIMD-only scaled-Hadamard using 32 lanes, with a compiled Python entry point:
  24 exact transform tests passed; all five greedy token sequences identical.
  Median paired speedup 2.22%. Retained as opt-in, not enabled by default.

Final CLI pairs, tokens/s:

| Pair | Baseline | SIMD scaled Hadamard |
| --- | ---: | ---: |
| 0 | 30.944 | 34.342 |
| 1 | 44.766 | 45.760 |
| 2 | 44.924 | 44.244 |
| 3 | 41.824 | 42.131 |
| 4 | 40.658 | 44.095 |

Timing variation is large. Median speeds alone (41.82 and 44.10) exaggerate the
typical paired improvement. No universal or large gain is established.

## Why these paths matter

The instrumented, uncompiled decode path invokes 186 input and 186 output
scaled-Hadamard transformations per token. These counts are not Metal dispatch
counts: the reference helper includes multiple MLX operations and compilation
can change dispatch counts. Explicit synchronization inflates timing; the
profile is stage attribution, not a pure GPU trace.

The vocabulary head has shape 2816 by 262144, with a 6-bit trellis containing
553,648,128 bytes. Its complete evaluation reads a substantial amount of data
per token. Reducing vocabulary or changing quantization is outside the
lossless requirement.

MLX's 128-element FP16 Hadamard uses radix-16 then radix-8, rounding to FP16 after
each radix before normalization. A mathematically equivalent seven-stage float
butterfly does not preserve these roundings. The experimental SIMD kernel does.

Next useful measurement: a Metal GPU capture to distinguish dispatch overhead,
integer EXL3 decode cost and memory stalls. Do not infer GPU utilization or
bandwidth saturation from synchronized Python timings alone.
