# Apple Silicon stability pass — 2026-10-30

Base: `f3f2c1d5e564cacf1b2bf2675bf972c8fc4f3ea6` (`main`). Branch:
`fix/apple-silicon-stability`. Independent of the disabled experiments in PR61.
This is a bounded corrective pass, without a throughput or universal reliability
claim. The final CI and local receipt are recorded below after verification.

## Defects and production contracts

| Trigger on the base | Correction and regression |
| --- | --- |
| M5 on macOS 26.2 reaches a ≥24-row EXL3 TensorOps dispatch; it uses cooperative tensor inputs introduced in 26.3. | Query runtime OS availability independently of GPU identity. Keep the existing portable QMV batching below 26.3. Gate direct and routed TensorOps entry points. CPU flag contracts and a native no-GPU query compare against `sw_vers`. |
| A packed Qwen3.6/3.8 DFlash draft uses native Metal int4 tensors on macOS <26.4. | Advertise that mode/Tune as unavailable in bridge readiness and reject direct weight/projection loading before file reads or GPU work, with an actionable version error. No silent requantization or alternative arithmetic. Simulated older-OS production-loader tests cover both boundaries. |
| Qwen attention/GDN dimensions overflow signed projection widths; nonpositive normalization epsilon or RoPE theta can reach model work. | Checked width arithmetic shared by layer constructors, with whole-config rejection before resident weights. Independent wide-integer boundary oracles and real loader/error-path tests. |
| A broken engine streams an unbounded JSON line into Desktop. | Bound each frame to 8,000,000 bytes excluding newline, including fragmented/unterminated frames. Clear it, report one bounded error, retire and join the engine. Tests allow large batches of small frames, reject stale failures and recover after a SIGTERM-ignoring fixture. |
| A direct `generate` switches DFlash→MTP while the old draft remains owned during new-head loading (historical issue45 A2). | Centralize draft retirement in the existing MTP preparer for all three callers, before path validation/reuse/allocation. An ignored physical regression checks failed-head cleanup and complete finite AR logits against a reset baseline. Physical peak memory is unmeasured. |
| Desktop Agent regression expects the old two-argument helper launch. | Assert the real three-argument contract including the persistent permission envelope outside the editable task files, and isolate permissions across tasks. No production Agent permissions were weakened. |

Apple documents [cooperative matmul inputs in 26.3 and integer tensors in 26.4](https://developer.apple.com/videos/play/tech-talks/111432/).
TensorOps itself is portable across M1–M5; this pass retains the repository's
existing M5-only EXL3 policy. It does not disable every TensorOps kernel on older
chips. The shader math, EXL3 quantization, weights, KV/recurrent commit rules and
experimental defaults are unchanged.

## Support requirements, separate from test coverage

| Configuration | Source contract in this branch | Physical coverage in this cloud pass |
| --- | --- | --- |
| Desktop/release engine | arm64 Apple Silicon; release manifest/deployment target macOS 26.2+, Swift tools 6.2, pinned MLX 0.32.2 | None; cloud is x86_64 Linux |
| M1, M2, M3, M4, including available Pro/Max/Ultra variants | Existing portable EXL3 path, regardless of newer OS; optional APIs still need their own OS version | Not executed |
| M5 and available family variants, macOS 26.2 | Portable EXL3 path; packed native-Q4 DFlash rejected | Not executed |
| M5 family, macOS 26.3 | Existing cooperative-input EXL3 TensorOps path on eligible shapes; packed native-Q4 DFlash rejected | Not executed |
| M1–M5, macOS 26.4+ | Packed Qwen3.6/3.8 DFlash API eligible; model/shape/greedy constraints still apply; M5 EXL3 selection remains separate | Not executed |
| Intel Mac, macOS below26.2 | Outside the shipped arm64 release contract | Not supported by this PR |
| Any RAM size | Checkpoint, context, resident weights, scratch, draft and available system memory must fit. SSD experts do not make resident state free. | No physical memory/OOM campaign |

The loader accepts supported layouts of `lfm2`, `lfm2_moe`, `qwen3_5`,
`qwen3_5_moe`, `qwen4_exp`, `gemma4`, and `bailing_hybrid`; a family name alone
does not establish tensor/layout compatibility. Mapped EXL3 K7 remains rejected.
Flash's automatic expert policy reserves 8 GiB of estimated available memory and
requires at least ten slots per each of48 layers; it can reject small-memory
systems. An explicit cache override is not proof that the whole model fits.
No RAM threshold or cache reserve is relaxed here.

## Hardware validation protocol before promotion

Record machine/GPU name, generation/variant/core count, RAM, macOS build, Xcode/
SDK, MLX version, source SHA, dirty state, binary/checkpoint hashes, options,
context/output budgets and known memory pressure. Use disposable registries and
fixtures. Run one inference process at a time; stop/join it before changing
models. Numeric checks precede any optional speed measurement.

1. Cover at least one physical M1/M2/M3/M4/M5, then available Pro/Max/Ultra
   variants, especially the smallest-RAM machine and a high-core-count GPU.
   Include macOS 26.2, 26.3 and26.4 boundaries where hardware supports them.
   Mark unavailable combinations explicitly. Hosted macOS builds cover no
   physical generation by themselves.
2. Build the shipped variant with pinned dependencies:

   ```sh
   cargo fmt --check
   cargo clippy --locked --all-targets --features mlx,chat -- -D warnings
   cargo test --locked --features mlx,chat
   cargo build --locked --release --features mlx,chat
   scripts/check-e2e.sh
   ```

   Set `MLXL3_MLX_ROOT` to the installed pinned package. Hosted regression CI
   explicitly sets `MLXL3_SKIP_METAL_CHECK=1`; a physical run must not carry that
   override. Some GPU oracles are ignored and must be selected explicitly.
3. Run the existing synthetic portable projection oracle alone:

   ```sh
   MLXL3_DISABLE_TENSOR_OPS=1 cargo test --locked --features mlx,chat --test runtime_gpu portable_exl3_prefill_matches_independent_single_rows -- --exact --ignored --test-threads=1
   ```

   Require exactly one executed test, zero failures and zero ignored tests in
   its result. It checks K1..8 and row boundaries23/24/25/46/47/256/513 plus
   grouped projections against single-row results. Then separately exercise
   eligible native paths on newer systems; disabling TensorOps does not test
   their implementation.
4. Use the repository's independent model/logit/state fixtures for each affected
   family and actual bridge/CLI. Include prompt lengths1/23/24/25, routed rows
   around63/64, KV wrap/context limits, reset, cancellation and retry. For Flash,
   include recurrent state, rollback, SSD read/truncation failure, MTP and
   DFlash; short contexts alone do not cover long-QSA selection.
5. On 26.2/26.3 assert packed native-Q4 rejection is explicit and ordinary
   generation still succeeds. On26.4+ test the packed draft with actual
   independent logits/token checks. Test raw Flash affine-Q4 separately. Test
   direct DFlash→MTP→ordinary generation and failed draft/head loading, then
   compare resumed output with a clean greedy reference.

   The new physical failed-head regression is selected with:

   ```sh
   cargo test --locked --bin mlxl3-rs --features mlx,chat tests::mtp_preparation_retires_flash_draft_before_invalid_head_is_read -- --exact --ignored --test-threads=1
   ```

   It requires local `MLXL3_FLASH_MODEL` and `MLXL3_FLASH_DRAFT_MODEL`. It checks
   failed-path retirement and AR recovery; it does not load a successful MTP
   head or replace the complete successful mode-switch campaign.
6. On low RAM, inspect idle/load/turn/switch/cancel RSS, physical footprint, MLX
   active/cache bytes and system swap separately; verify clean admission errors
   and process recovery. Do not deliberately overcommit the user's machine.
   Exercise small supported models and realistic contexts first. Record whether
   a failure is a loader error, process termination or system pressure.

Each fixture/selector must exist, execute a nonzero number of checks and reject
empty/nonfinite/truncated outputs. A timeout is incomplete, not parity. Existing
Kani CPU harnesses and any new guard proof do not establish Metal numerics,
allocator behavior, concurrent SSD safety or all-generation compatibility.

## Verification receipt

Cloud host: x86_64 Linux, Rust1.94.0, Python3.12, pytest8.4.2,
NumPy2.3.5, safetensors0.8.0, Ruff0.16.5, Modal1.5.5 in a disposable venv.
The production pins/manifests were not changed to match this host.

- `cargo fmt --check`, strict Clippy `--locked --no-default-features
  --all-targets --features chat -- -D warnings`, CPU build and complete CPU test
  suite passed: **248 passed,6 ignored** (162lib,35bin,16contracts,35transport).
  The Linux `runtime_gpu` target selected **zero** tests, which validates no GPU.
- Guard tests include **8,192** independent wide-integer boundary evaluations.
  Separate temporary mutations bypassing the OS requirement and replacing
  attention/GDN checked arithmetic with wrapping were detected. This is dynamic
  regression testing, not a formal proof.
- Python process/mode/memory/admission/package regression selection: **335
  passed**. The broader portable run passed **1,842**, skipped67 and deselected2;
  four modules with initial collection errors were explicitly excluded. A
  separate follow-up of the two Modal modules after installing its declared
  dependency passed **35**, skipped8 for unavailable Torch. The two Ling MLX
  modules remain unexecuted. This is not
  a complete macOS Python result. The first unrestricted collection had four
  dependency errors; the first portable attempt had24failures/41errors due to
  missing Modal/MLX/MLX-LM, with1,772passed/74skipped. Installing the declared
  Modal development dependency resolved its failures; the remaining two MLX
  cases were explicitly deselected for the final portable run. Failures are
  retained in the evidence logs.
- C++ existing kernel-cache test passed under ASan+UBSan; its counters are a
  regression check, not a new performance result. Python fake-engine smoke
  emitted exactly8,000,001 unterminated bytes, ignoredSIGTERM and was killed/
  joined. Fixture `py_compile` and Ruff F/E9 checks passed.
- All current Rust Apple ARM64 targets with `mlx,chat` passed frontend type
  checking and strict Clippy through an isolated temporary manifest: original
  source paths, build.rs disabled, tokenizers fancy-regex replacing native cross
  build dependencies **only there**. This does not compile/link the shipped FFI
  or validate Metal. Initial Linux MLX check failures on platform/configuration
  assumptions are retained. True native compilation is delegated to macOS CI.
- Targeted `cargo kani` and CBMC attempts were **unavailable**, not successful
  proofs; symbolic harnesses remain and the three new selectors are in CI.
  `formal-proof-skill` and language references were searched in the cloud
  filesystem and executor skill catalog but were absent. Swift/macOS SDK,
  `libproc.h` and physical Apple GPU/checkpoints are absent locally. Native FFI
  syntax validation, Swift tests and the new physical mode-switch test were not
  executed on Linux.

The [local evidence directory](measurements/apple-silicon-stability-20261030/)
retains commands/results and executable-source hashes. The PR records CI runs
for its exact final SHA separately; pending/failed/infrastructure-stopped jobs
must not be presented as a green result. CI now executes four new MLX-feature
error-path/query tests explicitly and rejects zero selections, plus the existing
Desktop parser/lifecycle/Agent checks. Hosted jobs skip physical Metal inference.

## Remaining risks in priority order

1. Execute the above physical matrix. No Linux or hosted build can establish
   shader availability, fidelity or memory behavior on every Mac M.
2. Qualify long-context Flash QSA, all supported model layouts and mode-switch
   memory pressure with independent state oracles; default short-prompt testing
   does not cover them.
3. Retain the existing packed DFlash sanitizer/numerical caveats in README;
   this PR does not resolve the historical Q4/MPP sanitizer report.
4. The visible recent PR/issue history was checked, including PR61 and issues
   45/52/59/60. Unpublished Desktop changes on Justin's Mac are unavailable here;
   compare the two Desktop/fixture diffs before integrating with that checkout.
   No installed app, merge or release is included.
