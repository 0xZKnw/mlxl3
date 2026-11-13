# Release validation — Desktop beta 11 and engine 2.0.5 beta 1

The Desktop DMG and engine archive were built on Apple Silicon from clean source
commit `a3cffdcaa7bafb513eb127c15319dc3c0a1f5046`, with MLX 0.32.2 and the
macOS 26.5 SDK. That source includes PR #163 and the MLX-only v2 policy from
`f3f181e805d25c7af95e908a3ec49b95389b46f2`. Subsequent PR changes add benchmark
evidence, its runner and CI configuration; they do not change the shipped
Desktop or engine executable source.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `MLXL3-Desktop-v2.0.0-beta.11-b42-Apple-Silicon.dmg` | 252,774,697 | `a52665c603060e59bc120b73a189c05bdc17a601de090dc1351dc821ed246002` |
| `MLXL3-Engine-v2.0.5-beta.1-arm64.tar.gz` | 67,863,224 | `c2de4f7081823d8779f64619cb39ec3d5b0c9a3d8c1aad992eecefc0dc17c1cf` |

The engine executable in both artifacts has SHA-256
`7f5ea49e6768ffddf0ad10e39dfb2cae48410789ce92842d4d1a42e7660338a0` and
reports `backend=mlx`, `version=2.0.5` and commit `a3cffdcaa7ba`. The Desktop
build manifest records a clean checkout, version 2.0.0 and build 42.

## Local checks

- The DMG was mounted read-only and detached. Its app signature and arm64
  architecture were checked; all bundled runtime files were compared by size
  and SHA-256 with the extracted archive's manifest.
- The actual beta engine archive was installed and selected in a disposable
  engine store through `EngineRuntimeStore.install`; `runtime-info` and `list`
  succeeded. The actual DMG passed the application's updater preflight. Neither
  test replaces the installed Desktop app or the user's engine.
- The packaged quantizer accepted a real local EXL3 checkpoint `probe`, checked
  its header and correctly reported it as already quantized. Its interface is
  JSON on standard input; `--help` is not a supported command.
- A real two-turn Heretic NX session preserved and recalled a test code. The
  actual Desktop → bridge → MLX engine → LFM3-VL path distinguished red and
  blue fixture images. This is a narrow integration check, not broad VQA.
- The six model measurements, checkpoint identities, prompt, per-run metrics,
  and bounded numerical comparisons are in
  [`v2-models-20261111`](../../measurements/v2-models-20261111/).
- After the MLX-policy merge, CPU/chat Rust tests passed: 424 library tests,
  10 ignored, plus 37, 17, 37 and 1 tests in the remaining suites. Rustfmt and
  strict CPU/chat Clippy passed. Engine packaging/builder tests: 84 passed and
  13 native/hybrid packaging cases intentionally skipped under the v2 policy.
- Relevant Desktop checks passed: 72 image/history/bridge cases, 257 workspace
  and catalogue cases, 48 Desktop experience cases, 24 French/English MTP
  layouts, 100 background speech-authorization callbacks and concurrent audio
  cleanup. The settings/notification/logo/response UI suite passed. A full
  Desktop run was interrupted when later user-requested changes arrived, and
  was not rerun in full after the final MLX-policy merge.
- The benchmark runner's four tests check complete runs, partial failures,
  non-finite metrics and the checked-in summary against individual runs.
  Four Metal API compile-fail doctests passed with `direct-metal` explicitly
  enabled for that CI check; the release engine itself remains MLX-only.

## Proof and CI limits

A Kani run on the unchanged LFM3 Hub support predicate checked 33 obligations
and four reachable covers, with unwind 20 and four symbolic text/vision and
known/unknown combinations. This bounded check does not prove JSON parsing,
Swift, Metal, MLX, I/O or the complete application. The earlier attempted
symbolic JSON proof was interrupted.

Two existing Flash route proof harnesses timed out after 600 seconds each on a
previous PR run; there was no reported counterexample. The Flash route function
and those harnesses are unchanged by the LFM3/UI work. A separate Apple CI job
exposed a feature-selection error after the default backend changed; its local
four-doctest check passed with the explicit feature correction. The owner asked
not to wait for the final remote CI run, so remote completion is not asserted.
The full earlier Python suite passed 2,760 tests with 156 skips before the last
MLX-policy integration; the relevant packaging and added tests were rerun as
listed above.

All signatures are ad hoc and the DMG is not notarized. Numerical and vision
checks cover the listed finite inputs only; they do not guarantee model quality
or performance for other prompts, images, hardware or memory pressure.
