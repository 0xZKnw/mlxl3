# P08: owned persistent tile conversion worker

## Scope and status

Reconstructed against the current PR163 checkout on 2026-11-09, following the
P08 finding in [issue162](https://github.com/0xZKnw/mlxl3/issues/162). Preregistered
as PR163-P08-20261109 in `opti.md` before implementation. Earlier lost local
implementations or historical Metal results do not validate this source.

The candidate is implemented locally and **opt-in**, using
`MLXL3_NATIVE_CONVERTER_WORKER=1` with the existing `MLXL3_NATIVE_ENGINE`.
The default remains the original one-process-per-batch implementation. Neither
configuration changes the shader, codebooks, two-pass tail-biting arithmetic,
rounding, tile ordering or packing algorithm. Matmul remains unchanged; no
arbitrary K splitting or NAX/TF32 arithmetic change is introduced.

## Ownership and contracts

`converter_tiles` owns one child for one complete call. A cancellable process-wide
lock serializes converter calls. It starts `quantize-worker ROOT --k K --codebook
CB --max-scratch-bytes 268435456` and sends canonical sequential decimal u64 IDs
over stdin. Each request is at most 21 bytes including newline. The native reader
reads at most 22 bytes before rejecting an oversized/incomplete frame. Replays,
zero/leading-zero IDs, overflow, alternate paths and out-of-order IDs are rejected.
Configuration and paths cannot be supplied through individual requests.

One `TileQuantizer` owns exactly one device, queue and pipeline. Per-request
autorelease pools and the existing per-launch pool release transient resources.
No model cache or unbounded PSO cache is introduced. Requests are strictly
single-flight; their input is `ROOT/ID/tiles.f32` and result is `ROOT/ID/result`.
Native admission requires a complete finite input fitting one scratch launch,
before materializing its payload. Scratch is capped at 256 MiB. Maximum tile
counts remain 31 at K2, 127 at K4 and 2032 at K8, so the largest request input is
2,080,768 bytes. Host vectors, serialized output and GPU input/output buffers add
bounded overhead; **256 MiB is not a claim about total RSS**. The caller's full
input/output arrays remain proportional to the original conversion size.

The native writer retains the existing exclusive atomic directory publication
and report/data formats. A response envelope contains the request ID and report.
Python checks the ID, schema, dimensions, exact configuration, directory entries,
file types/sizes, report equality and finite decoded values before returning any
result. A rejected request never returns partially converted arrays. Temporary
native result directories are private staging artifacts, not final model output.

Nonblocking pipes and selector polling impose a 300-second absolute per-request
deadline by default, including initialization/response wait; callers of the lower
level helper can select a smaller positive finite timeout. Each output stream is
limited to 64 KiB in user-space per request, with bounded kernel pipes and no
unbounded log file. Cancellation is checked every polling cycle and while waiting
for the converter lock. Error, cancellation and normal completion all kill/reap
the owned child before deleting its root. The worker is never reused after an
error. Hybrid CLI routing execs the native backend in the same PID.

## Verification and limitations

The formal-proof skill and language references were unavailable in this cloud
environment. Real-source Kani harness
`quantize::worker_request_ids_never_replay_or_overflow` uses a symbolic 21-byte
buffer, symbolic prefix length 0..21 and symbolic previous u64; successful parsing
must advance by exactly one without overflow. Unwind bound is 24, with coverage
for ID1. The existing scratch-plan harness remains. Kani invocation failed because
`cargo kani` is unavailable. CrossHair invocation likewise failed because its
module is absent. These properties are **not formally verified** here.

- Initial Python collection failed before executing tests because NumPy was
  absent. Installed the CI-pinned NumPy2.5.2 and safetensors0.8.0 into the existing
  test venv. Initial Ruff import/context warnings were corrected.
- First focused Python run: 69 passed, one explicit physical test skipped.
- Expanded converter suite: **157 passed, one skipped**, 7.97s; final repetition
  after the post-validation cancellation checkpoint: **157 passed, one skipped**,
  12.66s. It covers
  existing converter/transforms/capture/feedback boundaries, 21 K/codebook worker
  reuse combinations, real subprocess crash, silence, stdout/stderr floods,
  malformed/foreign replies, partial payload, cancellation after success,
  cancellable lock contention, fatal startup failures and fresh-worker recovery.
  Fake output verifies transport/lifecycle, not Metal numerical parity.
- CPU Rust `cargo test --locked --no-default-features --features chat --lib
  quantize::`: **2 passed**, 396 filtered; parser and scratch boundary checks.
- Strict CPU/chat Clippy across all targets passed in 11.49s, including the new
  CLI variant. Python Ruff lint/format and targeted Rust formatting passed.
- In a separate Python process, replaced only the response-ID predicate in the
  loaded module with `False`. The foreign-response test failed with DID NOT RAISE,
  detecting the mutation. Repository source was never changed. A fresh process
  rerun of the original test passed. This is a finite regression oracle, not proof.
- The hardware gate in `tests/test_native_quantizer.py` is deliberately skipped
  unless `MLXL3_PHYSICAL_WORKER_ENGINE` is set. It runs two requests for every
  K2..8/codebook combination against the independent frozen #98/#102/#107 corpus,
  comparing decoded bits and states exactly. It has **not run** on this source.

Commands and diagnostics are in `docs/measurements/pr163-converter/`. Existing
native-engine CI includes the edited Python files and regression test; hosted CI
for the final combined commit has not been inspected here. CPU checks do not
compile or qualify the macOS-only Metal implementation. Apple compilation,
physical corpus/packing, cancellation under active Metal load, repeated-request
RSS and spawn/compilation/transport/compute timings remain required. No speedup,
peak-memory measurement, installed application change, commit or push is claimed.
