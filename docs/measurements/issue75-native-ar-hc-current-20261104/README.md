# Native AR HC command chain — current-source qualification

Correctness is validated for this fixed native Flash point. Performance is not
reported in this commit; the root is running the preregistered isolated screen.
Integration is local: `MLXL3_NATIVE_AR_HC_CHAIN` still defaults to `0`, with no
installed-app replacement, main merge or release.

The production source is `f026dcd3710de58f129e4c2a6e3a1ca5187d6eef`, composed on
`717b92312e1c12a1a31386608dffcd78e3c0f916` (current main plus qualified default
NT2). The only subsequent executable edit is three lines in the ignored AR
fixture. Production CLI SHA-256 is
`1e282c471a8e525191d69d5e2b1c7b11b21923dd3d747654d5c0d5f99a8608c4`;
corrected libtest SHA-256 is
`2e9c7341e034fc76c8245c64b01f86691a7ab87c4c35624b81af848d395f0c2c`.
See [source receipt](source-receipt.json), [corrected CPU receipt](fixture-fix-cpu.json)
and [latest quality summary](quality-summary.json).

The chain puts the existing attention HC write and following MLP HC read into
one command buffer. Shaders, dispatch order, casts, buffers and arithmetic are
unchanged. `written` and the owned read intermediates survive through the final
completion wait; owner checks and completion status remain checked. This removes
one helper submission/wait at each of 48 layer boundaries. These counts do not
establish a throughput gain.

## Preserved failure and corrected fixture

The [initial campaign](physical-result.json) remains failed with `parity: false`.
The HC component and six independent frozen writes passed, then the AR test
failed at prefix 2: helper commands were 242→206 rather than the expected
48-command reduction. Prefix 0 had already passed 242→194 and its full states.
The remaining three phases were not run in that campaign.

[The source audit](failure-adaptation-audit.json) identifies a fixture asymmetry:
QSA snapshots retain a shared slab whose published cursor advances monotonically.
The control extended it before the candidate restored the older prefix. At a
nonempty restored prefix, each of 12 QSA layers therefore issued an extra copy
command only for the candidate. The correction primes and restores the snapshot
before measuring the control, so both branches have the same copy-on-write cost.
The exact 48-pair/48-command assertions and every numeric comparison are retained.
No production counter or operator was changed to hide the failure.

## Physical checks

Model: Qwen3.8-Flash-Next EXL3, checkpoint
`65c895314393431c09050b2e04e250836b3a6eb4`, average 2.05 bits/weight. Apple M5,
24 GiB, macOS 27.2 (26B123). HC enabled, NT2 enabled, dynamic readers disabled;
3 GiB cache, four demand readers. The independent target fixture requires
lookahead 1; the budget fixture uses its existing 0.6 GiB correctness cache.
Neither exception is a speed comparison. AC 100% was recorded at phase boundaries;
temperature was not measured. Root coordinated exclusive GPU execution with
compilers, provers and other owned inference drained.

| Check | Result | Process time |
| --- | --- | ---: |
| HC component, reused unchanged | 4 cases; 51,216 finite F32 words exact; actual commands 2→1; strides, F16/F32 branches, errors, recovery, owner | 1.48 s |
| Frozen independent writes, reused unchanged | 6 cases; 61,440 finite F32 output words exact; manifest `726e7e2a…` matches prior provenance | 0.77 s |
| Corrected full AR | Prefixes 0/2/9; 86,409,216 finite F32 state words, logits and hidden exact; 48 joint pairs and commands saved per forward; rollback/recovery | 8.66 s |
| Archived independent target | 84 states / 28,803,072 finite F32 words; 496,640 half logits; reset, prefill, continuation, nonzero prefetch and drain | 3.28 s |
| Production AR bridge | Four completions matching independent golden hash; rejection, context/delta cancellation and exact recovery; native allocation ledger | 61.01 s |
| Production AR budget | Two golden completions; 8 GiB request cap; memory-saver cleanup, ledger and rejection recovery | 25.77 s |

Corrected full-AR command counts are 242→194 at prefix 0 and 254→206 at prefixes
2 and 9, confirming both the original asymmetry and the retained 48-command
contract. The two earlier HC oracles are reused only because HC implementation,
GPU runtime, shaders and the production CLI are unchanged. Each Rust physical
test selected exactly one test, with none ignored within that selection. The
budget fixture body is byte-identical to its compiled source and explicitly
invokes the current CLI. Speculative configuration/generation requests in that
fixture are rejected before loading; no MTP or DFlash inference was performed.

[The corrected campaign](quality-02/physical-result.json) records four successful
remaining phases, with every owned process joined and process group verified
empty. Loaded-image audits require Metal and reject MLX/JACCL. Its
[production bridge report](quality-02/ar-bridge.json) preserves the nonempty
golden/text/cache checks. Original and corrected stdout/stderr are retained in
compressed, byte-verifiable form under `logs/`; [the manifest](log-manifest.json)
records both compressed and original hashes.

## CPU checks and verification limits

After the three-line fixture correction, format, native inventory (246 files),
strict direct Clippy and the complete relevant direct suite pass: 341 passed,
41 ignored. Those ignored tests are not silently counted as executed; the
physical selections above are separate. Release libtest rebuild completed in
52.91 seconds. Previous CPU-only/no-chat/MLX-compatibility checks remain recorded
because production code and those paths did not change.

The [actual corrected test-body Kani attempt](fixture-fix-kani.json) used a
disposable source copy, exposing only fixture/counter visibility and an explicit
proof wrapper, without Metal or filesystem stubs. It reached the known compiler
ICE at `intrinsics.rs:243:17` after 9.88 seconds, before obligations. This provides
no proof of the GPU, ownership, allocator, queue, concurrency or fixture. The
current flag harness's 90-second timeout and earlier 187/97-second interruptions
and ICE are retained, with no successful verdict invented.

The relevant physical plans and verification commands are in
[physical-plan.json](physical-plan.json),
[physical-resume-plan.json](physical-resume-plan.json) and
[fixture-fix-verification-plan.json](fixture-fix-verification-plan.json).
No full-generation performance, composition gain or default activation follows
from these correctness checks.
