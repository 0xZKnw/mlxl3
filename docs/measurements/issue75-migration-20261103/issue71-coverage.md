# Issue71 — audited teacher-forced coverage screen

Status: complete for these two windows. The six-policy offline replay preserves
all original captures and the five-policy intermediate summary. Production B4
has been restored byte for byte: 76 native files / 168 identity entries; the
two transient native test modules were removed. The production CLI is unchanged.

[Issue #71](https://github.com/0xZKnw/mlxl3/issues/71) proposes the initial
four-input shape: anchor, two children, and a grandchild under the first child.
That shape was implemented faithfully. It imposes depth two and maximum L=3;
stock B4 has depth three and maximum L=4. This is an important adaptation limit.

The fixed protocol used MAIN87/HOLD82, 64 greedy teacher IDs per prompt, 16
consecutive anchors at positions32..47, actual draft widths1/2/3, Q4/ring,
context4096, and the integrated B4 baseline. There were two GPU processes and
96 actual pool queries. Both processes completed and were joined; the recorded
GPU interval added zero swapin or swapout bytes. These historical guard results
do not authorize another run after subsequent machine activity.

L below is 16 emitted tokens / number of simulated cycles, including an ordinary
AR cycle when one token remains. These are coverage counts, not throughput.

| Policy | MAIN L (cycles) | HOLD L (cycles) |
|---|---:|---:|
| chain B2 | 2.0000 (8) | 1.7778 (9) |
| chain B3 | 2.6667 (6) | 2.6667 (6) |
| chain B4 | 4.0000 (4) | 3.2000 (5) |
| fixed tree, pool W2 | 2.6667 (6) | 2.6667 (6) |
| fixed tree, pool W3 | 2.6667 (6) | 2.6667 (6) |
| frontier4, pool W3 | 4.0000 (4) | 3.2000 (5) |

The final frontier ablation chooses three nodes by the best available joint
prefix score, the sum of local log-softmax scores from the actual stock pool.
It expands sixteen successors only under a selected parent, limits depth to
three, and breaks ties by depth then lexicographic rank path. The constructor
has no future/reference argument. This pairwise score is an ordering heuristic;
it is not a calibrated target acceptance probability or trained Weaver.

| Frontier topology, all 16 prefixes | MAIN | HOLD |
|---|---:|---:|
| chain | 14 | 12 |
| fork | 2 | 3 |
| star | 0 | 1 |

Against B4 at every captured prefix, frontier has equal matched length at16/16
MAIN prefixes, and equal14 / smaller2 / larger0 on HOLD. Its visited cycle
boundaries still emit the same16 tokens in the same4/5 cycles as B4. Adaptive
allocation therefore removes the aggregate deficit of the fixed fork here,
mostly by choosing a chain. It shows no coverage advantage over B4 in this
small screen. A tree-wide rejection would be unsupported.

Follow-up inspection: every actually visited full-width frontier cycle is
exactly the stock B4 chain, including IDs and parents: MAIN32/36/40/44 and
HOLD32/36/39/42. The final HOLD46 cycle uses the same reduced W1 stock tail
in both policies. All nine visited cycles therefore match stock chain
execution in the simulation; none actually branches. The equal aggregate L
is an expected degeneration to B4, not independent evidence about a branching
verifier. Keeping the complete B4 path and adding one alternative needs at
least five target inputs including the anchor; that variant is unmeasured.
See chain-degeneration-inspection.json. No new GPU run was performed.

Two adaptation corrections were made prospectively, using only archived pools:

- The initial simulator clipped a full-width policy at the terminal budget.
  Production reduces the query to min(width, remaining−1), and different widths
  can predict different IDs. Replay now uses the actual narrower captured pool;
  remaining1 emits one AR token. A tree uses its full topology only when at
  least four outputs remain, then a stock narrower chain. The original L values
  happen to be unchanged, but terminal accepted/bonus classification changes.
- A depth-two fixed tree needs only poolW2. Rebuilding it from the real W2 pool
  changes tree tokens at2/16 MAIN and5/16 HOLD prefixes, without changing total L.
  The final preregistered frontier ablation then tests the fixed node allocation.

The original shape rescues one alternative-root error on MAIN and none on HOLD.
It loses matched depth versus B4 at12/16 MAIN and7/16 HOLD prefixes; a forced
branch often spends a node where the correct chain can continue. Conditional
stock top16 coverage and the remaining diagnostics are retained in summary.json.

Source audit found the BF16 head, top16 IDs, codebook edges, score layout and
rank tie-break aligned with stock selection: all96 CPU reconstructions equal
actual select_logits. Pool shapes and all scores are finite. Each query group
leaves draft KV bytes and offset unchanged; policies were persisted before the
next teacher ID was exposed. Anchor y[j] is already emitted and unconsumed;
future begins at y[j+1], and the accepted prefix plus one bonus advances to j+L.
The first32 teacher IDs match the external AR32 hash.

One limit remains: the teacher appends true captures one row at a time, whereas
production commits accepted captures in groups of1..4 through affine4. This
probe does not independently prove equality across those append schedules. It
measures canonical teacher-prefix coverage. There is also no full path-conditioned
trained draft, target tree verifier, tree SSD union measurement, GPU-tree cost,
or tree throughput measurement. Python uses stable binary64 exp/log over exported
f32 scores for the offline frontier; its softmax rounding is not GPU parity.

Evidence and replay:

- coverage-MAIN.json / coverage-HOLD.json: immutable physical captures and original
  provisional simulations, never overwritten by corrected analysis.
- summary-five-policies.json: preserved intermediate analysis.
- summary.json: audited six-policy results, histories, limits, receipts and hashes.
- audit.py: exact offline replay command recorded in summary.json; no model import.
  It checks three synthetic frontier cases (confident chain, flat siblings, mixed
  fork), ancestor/budget invariants, all96 stock pool reconstructions, and exact
  equality with the five preserved simulations. The first replay refused only a
  history-label mismatch; audit-first-attempt.txt preserves that correction.
- CPU probe tests: 5 passed, 0 failed. Kani:323 successful checks,0 failed,
  1 unreachable standard-library formatting assertion,5/5 covers, unwind5.
  This bounds the original pure Rust accounting helper; it does not prove MLX,
  Metal, tree execution, concurrency, or the Python floating-point allocation.
  CrossHair is unavailable for that Python replay.
- gpu-plan-result.json / cpu-tests-result.json / kani-final-result.json /
  identity.json / restoration.json preserve commands, binaries, guards and scope.
- README-preparation.md preserves the initial preparation description, including
  assumptions superseded by this audit. commands.json contains planned commands,
  not execution receipts.

No GPU rerun followed these offline corrections. No tree kernel port, default
activation, runtime gain, or conclusion about trained Weaver follows from this
bounded screen.
