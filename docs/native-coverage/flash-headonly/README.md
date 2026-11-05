# Native DFlash prefill final heads

Native DFlash previously computed and downloaded 248,320 target logits for each
prompt row, then discarded intermediate vectors. With the existing
`MLXL3_FLASH_PREFILL_HEADS=1` default, only the last row of each nonempty prefill
or restored-suffix call computes its target head. `0` restores the eager path.
Every token-major trunk, capture, draft append and genuine final checkpoint
logit remains intact. AR, MTP and verification retain their previous behavior.

The [protocol](protocol.json) was recorded before edits. The
[source audit](audit.md) distinguishes this missing native mechanism from already
ported, rejected and unrelated historical paths.

## Verification

The [CPU receipt](cpu-receipt.json) records 330 passed / 9 ignored CPU tests and
339 / 38 direct Metal tests. Four strict Clippy configurations, Rust formatting
and the reviewed source inventory pass. The real-source Kani helper passes
61 checks and five covers for arbitrary usize row/length and both flags, without
assumptions. An eager-head mutation fails the intended assertion. This establishes
the CPU selection policy, not Metal or the whole generator. The 44 Linux-only
orchestration tests remain skipped locally; their exact obligation-set check was
also invoked directly and passed.

The [exclusive physical replay](quality-report.json) checks lengths 1/2/3/9:
actual target-head submissions N→1, 248,320 finite final half logits per case,
hidden, ten draft KV matrices and 115,212,288 finite F32 continuation words match
the eager control exactly. Paired suffix restore, errors and cancellation pass.
The final recovery vector assertion in that unit covers length/offset only.

The [actual bridge replay](bridge/report.json) passes cold/full-prefix requests
and independent hashes in AR/MTP/DFlash. Its malformed/error/cancellation/recovery
cases are AR cases. The runner's internal 0.6 GiB cache is a correctness setting,
not a timing comparison with the 3 GiB screen. Logs are retained byte-exact in
gzip with provenance hashes. No MLX/JACCL loads were observed.

## Short timing screen

The [complete summary](screen/summary.json), [executed arms](screen/arms.json) and
four raw reports preserve all eight warmups and eight measured responses: ABBA,
32 tokens, MAIN/HOLD, two measured samples per variant/prompt, 3 GiB requested
cache, 53 slots × 48 layers, four readers, lookahead zero and prompt reuse off.
The release CLI is the same uninstrumented SHA in all arms. AC power stayed at
100%; thermals were not measured.

| Prompt | Median TTFT control→candidate | Interpretation |
| --- | --- | --- |
| MAIN | 18.9162→18.5184 s (−2.103%) | Small observation consistent in both pairs. |
| HOLD | 17.7662→17.5547 s (−1.191%) | Inconclusive: pair changes +0.541%/−2.948%. |

No decode gain is established; HOLD control decode drifts 9.79%. Metal allocation
is unchanged at 6,608,617,472 bytes. Process footprint/peak and logical compression
are reported separately. There is no 64-token confirmation, long-context or
combined-port timing, and no universal or sustained boost claim.

Retain default 1 for the qualified elimination of unused work, with 0 as rollback.
The user then prioritized AR only: no further DFlash/MTP experiment is planned.
This PR publishes the already qualified slice; it does not install an app, merge
or release a product.
