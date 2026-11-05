# AR30 — Flash Next native AR qualification, 2026-11-05

Four native adaptations passed component, full-model and bridge correctness gates.
All four remain **experimental OFF**; this campaign does not establish 11–12 tok/s
or authorize default activation. The accepted thirteen AR10 settings remain ON
by default, and all twelve AR20 experiments remain OFF. Generation is AR only,
without MTP/DFlash. No installed app replacement, merge or release occurred.

[Issue 135](https://github.com/0xZKnw/mlxl3/issues/135) and the
[candidate matrix](matrix.md) preceded new tests/prototypes. Baseline was
`2035348319a76bd505ad6991ccdd510345225568`. Original dirty workspaces and model
files were preserved; work used isolated checkouts.

## Implemented scope and research reserves

| ID | Implemented adaptation | Explicit opt-in flag | Source handoff |
|---|---|---|---|
| C01 | Cooperative packed EXL3 K2/3/4 word ownership and SIMD shuffles; original codewords/FMA/reduction and unsupported-rate fallback retained | `MLXL3_NATIVE_QMV_COOPERATIVE` | [C01](c01-source.md) |
| C02 | Two router outputs per SIMD share input loads; canonical 80 FMA/lane and shuffle order retained | `MLXL3_NATIVE_ROUTER_PAIR` | [C02](C02-source.md) |
| C03 | Native normalized-Half GPU greedy selection; Private head and scalar readback, transaction/rollback preserved | `MLXL3_NATIVE_DEVICE_GREEDY` | [C03](C03-source.md) |
| C04 | Bounded completed-CB cache of private resident prepared/partial buffers; destinations remain fresh | `MLXL3_NATIVE_RESIDENT_SCRATCH` | [C04 and corrected accounting](C04-design.md) |

Flags use strict exact-M5 qualification and default OFF. Source handoffs preserve
their original pre-execution status; the completed receipts below supersede
their historical “not run” descriptions.

The pinned [kernel](research-kernels.md), [runtime](research-runtime.md) and
[attention](research-attention.md) reports distinguish implemented upstream
mechanisms from locally measured gains. They cover TurboEXL3, ExLlamaV3, MLX,
oMLX/MLX-LM, Strata, llama.cpp, TensorFold and additional native Metal engines.
Upstream throughput headlines are not native Flash Next evidence.
C05 capability-check caching, C06 aligned/dense QSA and C07 resident encoder
reuse remain conditional/unimplemented. C08 extraction metadata and C09 exact
SIMD ownership remain reserves; the [C09 mapping/cost design](c09-design.md)
is source analysis, not an executable prototype, checked proof or measured gain.
Canceled P04/expert-encoder work, HC L3–L5 and drafting were not resumed.

The [fresh intrusive profile](profile/report.json) motivated the scope:
3,639.7 allocation requests/row, 155.62 MB requested and 18.33 ms host allocation API
time; MoE 46.87 ms/GDN 31.06 ms GPU composites overlap. These instrumented figures
are neither release throughput nor additive recoverable time.

## Frozen identity and replay prerequisites

[freeze.json](freeze.json) records 577 source/helper/model-metadata hashes,
source `afb9fde3137a5c968ad54d0a4feeb7bcff414cee` **plus its recorded dirty
source state**, rustc 1.98.1/LLVM 22.1.8, and macOS deployment target 26.2.
Qualified execution used Apple M5; other devices/contexts are not covered here.

| Frozen artifact | SHA-256 |
|---|---|
| direct-cli | `f24201e3e5ca145f94397a90cf9edaa746d81c86102c67d2d51350a7d3e734f0` |
| native-libtest | `b284888627ad24d29bfa3f84c51ab9ec5dabb5206caacef1e2f482a40baa4f9b` |

Model: `/Users/justin/Documents/mix-stq1_0/models/Qwen3.8-Flash-Next-exl3-205-65c89531`,
checkpoint `65c895314393431c09050b2e04e250836b3a6eb4`, recorded quantization
`EXL3 avg2.05/H4/MTP2`. Metadata hashes identify the existing checkpoint;
no new full-weight rehash/download was performed.

Replay requires existing independent fixtures, not freshly generated candidate
oracles. Resident metadata/stock are reused from
`/Users/justin/.codex/worktrees/issue75-native-flash-gpu-ports/mix-stq1_0/build/issue75nt2/physical-02/`;
routed cases/stock from
`/Users/justin/.codex/worktrees/issue75-native-flash-ar-boost/mix-stq1_0/docs/measurements/issue75-native-expert-encoder-20261104/quality-01/routed-scratch/`.
The controller copies resident fixture metadata into a private output directory.
The retained full-model oracle is
`../flash-ar20-20261105/quiet-01/quality/oracle.json`, SHA-256
`171bea949c7c16f20c1e1affa26b70602676c73620fcbb380d430a7587835351`.

[run.py.txt](run.py.txt) reuses audited process/marker/CLI helpers. The freeze
pins all helper paths and bytes, including basis.py `7405384a…`,
components-basis.py `de424bda…`, run.py `15160c31…`, historical cli_short.py
`e48d4123…` and benchmark-flash-bridge.py `8560057f…`; full digests are in
the freeze. Historical absolute helper/fixture paths must exist for replay.
Missing prerequisites must fail explicitly, not yield an empty parity pass.

## CPU checks and bounded verification

[Final CPU review](cpu/final-review.json): fmt and strict all-target Clippy
for no-direct/chat, direct/chat and mlx/chat PASS; Direct suite **396 PASS,
91 ignored**, with two zero-test binaries explicitly excluded from coverage.
Native inventory 276 files and portable 10 tests+2 subtests PASS. The initial
large_enum_variant Clippy failure is preserved in [cpu-all.json](cpu/cpu-all.json);
boxing the two large layer variants repaired it, and affected checks were rerun.
Release CLI/libtest builds passed before exclusive GPU execution.

[First Kani receipt](cpu/kani-first.json): Kani 0.68/CBMC 6.11, three real CPU
harnesses PASS, 394 reported obligations (387 SUCCESS/7 internal std UNREACHABLE),
15/15 covers. Domains include literal flag UTF8 bytes 0..4/unwind 10, full symbolic
router shape/capability bounds and greedy admission/readback values.
[C04 receipt/log](cpu/kani-scratch.log): one real Budget/guard/ResourcePlan
harness PASS, 387 reported obligations (381 SUCCESS/6 internal std UNREACHABLE),
6/6 covers; full-u64 claims/refunds, admission/ownership booleans and u16 positions
at context 4096, no loops/assumptions/stubs. Individual unreachable checks were
reviewed. These are bounded CPU checks, not proofs of Metal, completion,
concurrency, allocator correctness or complete generation. The unrelated full
Kani suite was not rerun locally; exact-head hosted CI is reported separately.
CrossHair was unavailable; no source-verifier proof of Metal SIMD is claimed.

## Explicit physical and bridge qualification

The ordinary CPU suite leaves physical tests ignored. Subsequently
**8/8 selected physical/bridge phases PASS**, all children joined, before speed.
[quality/report.json](quality/report.json) records exact commands, environments,
actual counters, marker checks and timings; [controller](run.py.txt) retains
bounded process-group cleanup and truthful failure reporting.

| Phase | Nonempty independent evidence |
|---|---|
| C01 windows | 2,300,928 state words, 8,960 bit-basis vectors; K2/3/4, ring wrap, K5–8 fallback, three wrong-owner mutants detected |
| C01 resident | 48 frozen cases/738,304 finite Half words, full stock-binary equality, 44 actual cooperative dispatches |
| C01 routed | 20,480 finite Half words vs frozen stock, 24 actual cooperative dispatches |
| C02 router | 10 cases, 11,264 logits+11,264 probabilities, 220 IDs+scores, 10 paired uses; strides/ties/wrong-row mutant/rejection |
| C03 greedy | 11 full 248,320-wide distributions, 8 valid/3 invalid; independent CPU f64+canonical Half oracle, raw-argmax mutant caught |
| C04 scratch | 8 outputs/81,920 Half words, 2 reuses/46,080 B; poison, retained destinations, pending/same-CB fallback, failure/abandon/refund |
| Common model | 57,606,144 state words, 993,280 logits, 81,920 hidden bytes, 84 initial states; exact independent archive/continuation, invalid-scalar rollback/recovery |
| Real CLI | 2 token golden `9f850af1cbba074b`, no MLX dylib, shutdown/join PASS |

The common model observed 336 router uses, 936 scratch reuses and 5,821,440 charged
bytes: 157 slots/314 buffers, including the head admitted during late rollback.
QSA output projections are Shared and excluded; [dated C04 correction](C04-design.md)
preserves the earlier overestimate and reconciles every byte. The 32 MiB cap is
conservative charge, separate from actual cache backings and process RAM.
Numerical equality/counter checks do not establish a speed benefit.

## Final short speed evaluation

Same MAIN32 protocol (32 generated/31 decode, one warm+one measured request,
context4096/cache3GiB/readers4/lookahead0/prefix0/plain AR), with compiler/prover
processes stopped. [Evaluation receipt](evaluation.json) preserves all36 complete
golden warm/measured outputs; every process group joined. No swap abort or cooling timer.

| Campaign | Samples in order (tok/s) | Decision |
|---|---|---|
| [First screen](speed/report.json) | OFF7.582779; C01+C02 8.854881; C03+C04 7.243154; all5.543504; OFF8.361276 | Strong drift; no boost established |
| [Old/new confirmation](confirm-01/report.json) | old9.839609; newOFF9.672806; C01+C02 8.940462/8.741837; old9.628187 | NewOFF inside old control range; kernel composition loses against both controls |
| [Standalone attribution](split-01/report.json) | OFF10.409299; C02 10.323522; C04 10.355911; C03 10.247368; C01 9.384326; OFF10.253031 | C01 regresses8–10%; C02/C03/C04 remain within control noise |
| [Compatible composition](compose-01/report.json) | C02+C03+C04 9.911238; closingOFF9.345471, precedingOFF10.253031 | Candidate lies between drifting controls; nonconclusive |

**No new default is promoted and11–12 tok/s was not achieved.** C01's stable
regression is retained as a negative result; the other correct adaptations stay
opt-in because no speed gain is established. This is not evidence that every
upstream mechanism is ineffective or that the Mac's temperature caused drift.
C09 remains a source design only: it needs a different SIMD mapping and cannot
naively compose with C01.

The user now requests deeper profiling of the actual retained ~10.4tok/s path
before further optimization research. AR31 is separately preregistered in
`opti.md`: production-path fidelity, observer overhead, aligned CPU/GPU/SSD
clocks and a critical-path/overlap account. Existing native8's191.8ms intrusive
row is not a96ms production budget. No AR31 model capture is included here.
