# Existing Flash mechanisms in the independent native engine

Audit before the HEADONLY prototype, based on `opti.md`, the issue54 HEADONLY
screen1/screen2/pooled reports, issue59 T01 notes and POST-P09 T01B/DEC-ONE history.
This is a source audit, not a performance measurement.

| Mechanism | Native state at cache base be43ee15 | Decision |
| --- | --- | --- |
| Prepared MLX source/header and generated-kernel cache (T01) | Native Metal owns pipelines and checkpoint projections; there is no MLX source/header generation in its inference loop. | No wrapper port. |
| DFlash prefill HEADONLY | `Session::extend_prefill` eagerly calls the target head and reads 248,320 half logits for every row, retaining only the last. | Port only the unused-head elimination; keep token-major trunk/capture/append order. |
| AR HEADONLY | `FlashNext::prefill_inner` already runs hidden-only prefix rows and one final head per chunk. | Preserve AR behavior. |
| MTP last-two prefill heads | Native `FlashMtpSession::extend_prefill` still calls target forward for each row. | Separate follow-up after DFlash qualification; no MTP change here. |
| Device greedy selection | Native reads full logits, computes normalized probabilities and selects on the host; stock retains normalized FP16 tie semantics. | Defer; no old isolated host-readback gain established, and raw-logit argmax is not equivalent. |
| Prefix snapshots | Native target snapshots clone owned Metal views; DFlash snapshots preserve pending-row ownership. The production cache stores genuine logits after each completed suffix. | Preserve ownership and one required final head per call; no full-state-copy optimization assumed. |
| QSA append/ring, prompt reuse and SSD reads | Native implementations exist, with cache/mode and IO work owned by another agent. | Keep unchanged in this port. |
| P01 layer-major prefill / T01B / DEC-ONE variants | Historical model-level results were rejected, withdrawn or inconclusive under their recorded protocols. | Do not relabel them as a new proven native boost. |

The old HEADONLY pooled TTFT changes were −4.210% MAIN, −3.827% HOLD and
−5.074% LONG. Its inverse-order strict global gate was false, and conditions
changed between pairs. These measurements justify investigating the mechanism;
they do not establish its gain on this eager native runtime. Native speed remains
unmeasured until the preregistered same-binary campaign passes correction first.

The actual native cache caller passes the entire nonempty DFlash suffix to
`prefill`/`extend_prefill`. An empty suffix uses saved genuine logits. Each call's
last row must therefore compute its own head, even if that row is an intermediate
checkpoint of a longer conversation.
