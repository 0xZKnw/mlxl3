# Ling 3.0 Tiny conversion

The Ling adapter uses MLX-LM's native Bailing V3 KDA/MLA implementation and
MLXL3's existing EXL3 linear and grouped MoE kernels. No remote model Python
is executed, and no speculative decoding or training is involved.

Source: `inclusionAI/Ling-3.0-tiny`, revision
`e3a47d5b986e7141b6efd62597d598ebb392060d`.

```sh
.venv/bin/hf download inclusionAI/Ling-3.0-tiny \
  --revision e3a47d5b986e7141b6efd62597d598ebb392060d \
  --local-dir models/source/Ling-3.0-tiny-HF
.venv/bin/python scripts/quantize_ling.py \
  --in-dir models/source/Ling-3.0-tiny-HF \
  --out-dir models/Ling-3.0-tiny-EXL3-4bpw \
  --work-dir build/ling-4bpw --bits 4 --head-bits 4 \
  --calibration-rows 2048 --calibration-seq-len 1024 \
  --max-workers 2 --search-backend metal
```

Requires the optional pinned PonyExl3 checkout and consolidated patch described
in the [LFM conversion guide](lfm26-local-quantization.md). Apply that patch
once to a clean checkout; it contains the Ling discovery/export, bounded
conversion and Metal-search changes. The checkout is not needed for inference.

The complete plan contains 9,031 eligible linear projections, including 8,832
routed-expert projections. Embeddings, routers, convolution kernels, unsupported
small-width projections, and fused MLA `kv_b_proj` stay unquantized. Original
FP32 tensors remain FP32. “4 bpw” describes the quantized matrices, not every
parameter or the total file size.

Calibration runs BF16 source layers one at a time over two independent
1,024-token sequences, with 128-token execution chunks and recurrent/KV state
preserved within each sequence. All experts use the same observed normalized
MoE input rows; each down projection uses its own expert's BF16 SwiGLU outputs.
This is common-input calibration, not routing-conditioned expert sampling.
No expert borrows another expert's down activations.

For the pinned Ling source at uniform K4/MCG, the adapter reuses the 0.908
expert global scale established by 1,552 completed searches from the same
recipe. A 128-expert layer measured 127.3 seconds for its 384 projections;
its mean inner relative RMS was 0.0195% lower than the historical individually
searched weights (worst individual change +0.332%). Down activations are formed
in batches of eight and remain byte-identical after FP16 capture. The complete
model took about **92 minutes** on the measured M5 under changing power/thermal
conditions, rather than the earlier 55–60 minute pilot estimate. Its held-out
WikiText-2 perplexity was **21.66936** versus **21.05421** for BF16 on the same
2,048 tokens and 256-token windows. The model was registered locally as
`ling3.0-tiny-4bpw`; it has not been published. The scale shortcut's own
quality effect cannot be isolated without a full comparison against individual
scale searches.

Activations are persisted as NumPy files and candidate measurement is split
into groups of 32. This bounds the converter's otherwise eager pending list.
The recipe, candidates and LDLQ arithmetic remain unchanged across groups.
Work files require substantial disk space; keep the work directory until
conversion and validation are complete so interrupted stages can resume.

Validation before publication: complete tensor inventory, strict EXL3 load,
held-out perplexity versus BF16 on identical tokens, and CLI generation.
Conversion success alone is not a quality result.

The layerwise BF16 baseline uses `benchmarks/eval_ling_source.py`. Compare with
`benchmarks/eval_checkpoint_perplexity.py --chunk-size 128` on the EXL3 model,
using the same test text, token count and 256-token windows.

An 18-projection routed-expert pilot on this Mac took 14.82 s with one worker,
13.52 s with two and 14.08 s with four. All trellis and scale tensors matched
exactly. This is a small local scheduling check, not a whole-model speedup claim.

Ling's `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`
format is supported by the shared CLI/desktop bridge parser, including
split streaming markers. Existing MCP enablement and schema checks still apply.
