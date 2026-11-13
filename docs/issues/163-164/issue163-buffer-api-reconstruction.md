# Issue 162 N09: public GPU result admission

## Preregistration and scope

2026-11-09, shared PR163 recovery tree. Parent preregistered the initial
buffer audit and approved this narrower revision before tests: preserve the
internal no-wait encoding path, replace the four public command-taking model
encoders with checked synchronous APIs, and qualify host byte-range admission.
No Mac, device benchmark, push or performance claim.

The earlier broad migration proposal in `issue163-native-reconstruction.md`
was based on treating every internal MatrixView as a public pending capability.
Inspection found a smaller external hole: four public `encode_model_*`
methods accepted a caller command and returned a host-readable view before
submission. Raw Buffer+count helpers are already private in this baseline.
The chosen correction is not just visibility: public `model_rms`,
`model_dense`, `model_gqa`, and `model_rope` retain checked shape/dtype/device
admission, allocate shared output, submit their own command, wait and reject
failed completion before returning. They take no caller command or completion
token. The internal encoder paths retain their existing same-queue batching.

## Producer and caller audit

- `TensorStorage` fields and `MatrixView` backing/layout/dtype are private to
  the GPU implementation. Public storage comes from copied F16/F32 uploads.
  `matrix` admits all reachable indices using `MatrixLayout::fits`; `row`
  retains the same backing and admitted offset. Neither accepts a raw buffer.
- `ProjectionOutput` retains its command and raw buffer privately. `read`
  waits and checks success. Projection bundle and expert bank `forward`
  return views through `Metal::command`; their encoders are internal.
- `RouteOutput` keeps pending scores nonpublic; `completed_scores` waits and
  checks its own command. `MoeOutput` keeps values private and checks every
  producer plus retained Hot producer failures before `completed`/`read`.
- `FlashHc::read/write` use `Metal::command`; pending routes complete before
  publication. Workspace-reused normalized/down/active/up views stay internal;
  published mixed/gate allocations are separate. Its forward-MoE path calls
  the experts' completion gate before the final write.
- GDN, QSA and PLE public forward paths complete their own commands before
  exposing values; layer forward paths use the deferred owner's `complete`.
  QwenGdn forward uses `Metal::command`. Public traces copy after completion.
- Flash hidden state is published after completed layer/head paths, with
  rollback on failure. Verification prefix states are retained after cohort
  completion; clones/rows do not manufacture a different producer token.
- CLI and numerical diagnostic `read_output` calls consume the above public
  completed results. Internal diagnostic reads stay behind existing command
  or owner completion. No new waits were added to their encode-side paths.

Four external compile-fail API examples reject early command-based results;
the Apple compatibility workflow explicitly runs these doctests. These are
not claimed executed on Linux where the entire Metal module is cfg-disabled.

## Host pointer boundary

`MatrixLayout::host_byte_range` is the actual production readback admission.
It requires F16/F32 element width, nonnegative aligned offset, contiguous
columns (and rows for multirow views), aligned allocation size, checked byte
arithmetic, allocation containment, and `isize::MAX` pointer bounds. Empty
ranges return without constructing a raw slice. The unsafe slice creation
is confined to retained shared immutable backing, copies immediately and
does not return a borrowed host pointer. Row views now read their actual
offset instead of requiring the entire allocation. Trace export retains its
stricter exact-allocation admission.

Tests enumerate small layouts, strides, offsets, element widths and allocation
sizes against an independent integer oracle; cover row aliases, truncation and
extreme arithmetic. A full-domain Kani harness targets this real helper.
The independent model-operator physical fixture now calls the public wrappers;
an additional ignored physical test covers RMS, identity dense, one-key GQA,
position-zero RoPE, bad gain shape, row readback after parent view drop.

## Results and limits

- Targeted portable layout suite: 6 passed, 0 ignored. Removing allocation
  containment caused both new host-range tests to fail (truncated row admitted
  and empty range outside allocation); the guard was restored.
- Final restored layout suite: 6 passed, 0 ignored (0.03s). Portable strict
  all-target Clippy with chat passed (6.82s); scoped rustfmt check passed.
- Kani attempted with `cargo kani --lib --no-default-features --harness
  host_read_range_is_aligned_bounded_and_exact`: unavailable, `no such command:
  kani`. Harness retained; no proof claimed.
- Requested formal-proof skill and language reference were absent from the
  available catalog and repository `.agents`; repository proof rules followed.
- First shell attempt lacked Cargo on PATH; the next needed sandbox write
  permission for formatting/build outputs. Authorized escalated run succeeded.
- Metal compilation, the four external doctests, physical numerical tests,
  failed-device execution and actual GPU concurrency remain **unverified** on
  this Linux worker. CPU range tests do not prove GPU completion or numerics.
  No complete issue-162 or device qualification claim follows from this patch.
