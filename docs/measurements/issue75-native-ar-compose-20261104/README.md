# Native Flash K/CB and HC composition

Correctness is qualified for the composed candidate. Its performance pilot is
pending in this source checkpoint; no isolated gains are added or presented as
the composition result. The candidate is local and has not replaced an installed
app, merged main or produced a release.

Base is PR117 head `b56df620f2c8e4bc18ceff0285c86a4af21b6460`. Existing HC runtime
and its retained failed/corrected qualification were cherry-picked without
changing shaders. The default change enables HC only for the exact measured
device name `Apple M5`; other device names retain default off. Explicit
`MLXL3_NATIVE_AR_HC_CHAIN=0` rolls back, `1` enables the experimental path elsewhere,
and malformed settings remain errors. Static-K shaders, selector and projection
source are byte-identical to PR117.

The new runtime changes are only the hardware-aware HC default and its caller
wiring. The existing command-chain implementation retains its buffers through
the final wait, checks the Metal owner and completion status, and preserves every
shader, cast and accumulation. Test-only activation markers expose existing
fields/counters; they add no product setting or release instrumentation.

## Source and verification

[freeze.json](freeze.json) identifies every changed source and frozen artifact.
Both executable artifacts have verified macOS minimum 26.2 / SDK 27.2:

- CLI SHA-256 `942dbeee8ccb2c5907c2b071ae071707139e17f20fbeda4ecf326603f8d68d90`.
- Libtest SHA-256 `425387869ada59192fac9edff50547782d8572c9504dba90c34fa8ad99c37527`.

The private Cargo target was APFS-cloned and package-cleaned. Formatting,
inventory (246 files), three strict Clippy configurations and full CPU/direct/
compatibility suites pass: 333/342/378 passed, 9/41/259 ignored. Python has six
passes and 44 Linux-specific skips; two portable inventory-policy checks also
pass when invoked directly. After adding the requested activation marker,
strict direct Clippy, the direct library suite (253 passed / 37 ignored) and the
release libtest were repeated. [CPU receipts](cpu-result.json) and
[affected marker checks](marker-cpu-result.json) distinguish these stages.

[The bounded Kani check](hc-default-proof.json) verifies the actual HC flag helper
with arbitrary hardware/presence booleans and valid UTF-8 strings of 0–4 bytes,
all byte patterns, no assumptions and unwind 10 with unwinding assertions.
All 237 checks pass, with seven covers satisfied. The seven unreachable checks
are listed individually and belong to standard-library formatting, UTF-8 and
alignment internals; source assertions and required branches are exercised.
A byte-exact extraction of the actual helper and unit test rejects the mutation
that restores an unconditional default off, then passes the original
([receipt](flag-mutation.json)). This mutation check is a test, not a proof.
The earlier HC 90-second timeout and GPU-path compiler ICE remain preserved in
the prior HC qualification; this bounded helper result does not prove Metal,
queues, ownership, concurrency or complete generation.

## Composed physical qualification

Model: Qwen3.8-Flash-Next EXL3, checkpoint
`65c895314393431c09050b2e04e250836b3a6eb4`, average 2.05 bits/weight, Apple M5
24 GiB / macOS 27.2. HC and Static-K environment variables were absent, exercising
the real M5 defaults, with NT2 fixed to 1. The normal cache is 3 GiB / four demand
readers / lookahead 0. The existing independent-state fixture requires lookahead
1; the existing budget fixture uses 0.6 GiB. These are correctness exceptions,
not performance comparisons.

| Gate | Result | Process time |
| --- | --- | ---: |
| HC full AR, prefixes 0/2/9 | 86,409,216 finite F32 state words, hidden and logits exact; 48 joint pairs and 48 helper commands removed per forward; rollback/recovery | 7.57 s |
| Archived independent target | 84 states / 28,803,072 finite F32 words and 496,640 half logits exact; reset/prefill/continuation and prefetch drain | 3.54 s |
| Actual AR bridge | Four golden completions, malformed-message rejection, context/delta cancellation and exact recovery | 56.16 s |
| Actual AR budget | Two golden completions, memory-saver cleanup, native ledger and rejection recovery | 23.54 s |

The independent fixture strictly reports and checks activation from the same
actual Metal instance: initial HC is `true`, HC joint-pair count is **192**, and
Static-K submissions are **[5760, 0, 0, 480, 723]**. Thus equal values cannot hide
a silent fallback. Both K2 and resident K4 execute. Every selected Rust physical
test runs exactly one test, with zero ignored within the selection. Existing
component/shader oracles are reused because their source is unchanged; no stock
model was loaded again.

All four phases pass in 90.85 seconds. Each owned process is joined and its
process group verified empty; loaded-image audits require Metal and reject
MLX/JACCL. No MTP or DFlash inference is performed. See [quality summary](quality-summary.json),
[plan](quality-01/plan.json), [raw result](quality-01/report.json) and
[production bridge result](quality-01/ar-bridge.json). Compressed logs and their
original/compressed hashes are in [log-manifest.json](log-manifest.json).

## Performance scope

The earlier isolated HC [short screen](isolated-hc/hc-chain-summary.json) and its
four raw reports are retained separately: MAIN median 5.1009→5.2409 tok/s and
HOLD 5.1664→5.3508 tok/s, two measurements per variant/prompt, unchanged native
Metal allocation. That binary had HC default off and Static-K off; those results
do not validate the composed default candidate or a sum of gains.

The [preregistered composed pilot](protocol.json) compares one fixed final binary
with A = Static-K 0 / HC 0 against B = Static-K 1 / HC 1, NT2 1, ABBA, MAIN/HOLD,
32 tokens, one warmup and one measurement per prompt/process, 3 GiB cache,
four readers, lookahead 0 and prompt reuse off. Root owns that isolated GPU
campaign. Sustained, long-context, speculative-mode and installed-app performance
remain unmeasured here.
