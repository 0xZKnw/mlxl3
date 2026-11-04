# Opt-in native HC-to-gate command grouping

Base merged main `21cfc9ad`. The source change appends unchanged gate dot,
softmax and normalized top-10 to the accepted HC write/read command. All HC
and gate intermediates remain owned until its final wait/status check. The
same Metal instance and queue are required. Route IDs, DemandPlan, SSD demand
and hot/W1/W2 waves remain in their existing consumer.

`MLXL3_NATIVE_AR_HC_GATE_CHAIN` is snapshotted once, defaults off, accepts only
literal 0/1 and rejects malformed/non-Unicode settings. Selection additionally
requires HC active, depth0 and no prefetch pool. Every other path falls back.
There is no GPU quality or performance verdict yet. No app or release changed.

## CPU checks actually executed

The APFS-cloned private target was package-cleaned. See `cpu-result.json` for
exact argv/environment/durations and `logs/` plus its SHA manifest. Formatting,
source inventory246, three strict Clippy configurations and full CPU/direct/
compatibility suites pass:334/343/379 tests,9/45/263 ignored. Python inventory
checks:6 pass,44 Linux skips; the two portable partition/required-name tests
were also called directly and pass. Ruff check/format of the changed Python
inventory test passes. Fresh CLI and libtest were built with deployment26.2;
`freeze.json` records hashes, paths, sizes and SDK27.2. A review found the
new ignored owner fixture needed cache capacity10 rather than1; this test-only
precondition was corrected before GPU execution, direct strict Clippy reran,
and the final release libtest was built afterwards. No GPU test has run here.

The actual eligibility helper Kani check passes81 checks and5 covers in3.39s,
with arbitrary3 booleans/full usize depth, no assumptions and no unreachable
obligations. It also checks absent parser input with defaultfalse. The reused
strict parser implementation is unchanged from the earlier UTF8<=4/unwind10
proof; this is not a fresh proof of unbounded strings. The actual unit test
rejects a temporary depth-guard inversion; extraction/mutation evidence is a
test, not a proof. See `kani-selection.json` and `mutation.json`.

The actual new full-AR GPU fixture was selected by an explicit Kani wrapper
in a disposable source copy, without Metal/FS stubs. Preparation errors
E0425/E0599/E0609 are retained separately. The final current direct-metal/Kani
context still cannot compile the pre-existing `flash_profile.rs:455` proof
wrapper calling a function marked only `#[test]`; there is no GPU/source
property verdict. This pre-existing blocker and older HC ICE/timeouts remain
visible; no unrelated proof source was repaired or weakened.

## Physical checks pending central composition

`physical-plan.json` contains exact new test names, bounded commands and
strict nonempty/finite/count requirements. Root will substitute the frozen
composed executable identities before launch. The new depth0 archived oracle
requires192 actual joints and zero standalone gate commands; the depth1 old
oracle requires zero joints and positive standalone commands. Full AR compares
prefixes0/2/9 and86,409,216 finite F32 state words/hidden/logits. Expected source
mechanism:48 gate commands disappear while helper commands and wait API calls
remain equal; no wait-call or timing saving is inferred. Analytic F32/F16
ties/strides, foreign owner/queue, dropped command/recovery, route lifetime,
existing SSD faults, real bridge cancellation/recovery and budget are explicit.
Two independent source reviews found no concrete defect; they are not proofs.
