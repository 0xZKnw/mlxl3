The source-forward producer and host consumer use this schema. The consumer is implemented; raw source model forward/capture is still an integration gate. None of this receipt alone establishes numerical parity of a model.

```
mlxl3 capture MODEL --tokens segments.json --modules modules.json \
  --max-rows N --output DIR --capture-dtype float16 [--logits]
```

The segments file is a nonempty array of nonempty token-ID arrays. A new session starts for each segment. Modules are exact checkpoint projection names, with expected input widths supplied independently by the host. The observer sees actual projection inputs; SwiGLU down-projection inputs are captured after the activation. Raw BF16 source computation preserves BF16 operator boundaries and must not be replaced by the F16 graph used for existing quantized checkpoints.

`manifest.json` schema 1 has `status="complete"` only after closing all payloads and atomically publishing the directory. Its `source` equals the caller's `{path, config_sha256, weights:[{name,size,mtime_ns}]}` identity, and `request` equals `{tokens_sha256, modules_sha256, max_rows, capture_dtype, logits}`. The two hashes identify the exact request files, not a reserialized equivalent.

`projections` contains exactly one record per requested module:

```json
{
  "path": "projection-00001.f32",
  "shape": [128, 2560],
  "storage_dtype": "float32",
  "original_dtype": "bfloat16",
  "observed_rows": 2048,
  "sha256": "<64 lowercase hex characters>"
}
```

Payloads are contiguous little-endian FP32. Projection inputs first round to the requested `float16` or `float32` capture dtype, then expand to FP32 storage. The consumer verifies finite values and exact FP16 expansion when requested. `original_dtype` is `float16`, `bfloat16`, or `float32` and describes the observed source tensor before capture rounding. Stored rows equal `min(observed_rows,max_rows)`, preserving the original first-N policy. Cold projections remain explicit `[0,input_width]` records with `path=null, sha256=null`; an adapter can then apply its maintained expert-coverage policy without hiding missing observations.

`processed_segment_lengths` attests the processed input prefix. Every completed segment has its full requested length. Early stop or a partial last segment is permitted only when all projection caps are full and logits were not requested. With logits enabled all requested tokens are processed, and the `logits` record has the same path/shape/storage/original-dtype/hash fields, shape `[sum(segment_lengths),checkpoint_vocab_size]`, and raw pre-softmax FP32-expanded values in segment/token order. Teacher rows stream to disk; they need not be resident together. Without logits requested, that record is absent or null.

The host refuses partial/foreign receipts, duplicate JSON keys, shape/dtype/count mismatches, nonfinite values, bad hashes, aliasing payloads, directory traversal, symlinks and later corruption. It opens calibration arrays lazily as read-only memmaps, one active module at a time. Native MoE all-expert filling and routed-floor policy require the additional actual shared inputs and route observations discussed with the model implementation; they must not be synthesized from quantized F16 captures.
