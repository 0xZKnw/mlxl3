# B09/B11/B12 targeted CPU proof window

One bounded90-second CPU window,85.12seconds actually used including setup/compilation and complete process-group cleanup. Kani0.68.0/CBMC6.11.0. Commands, sourceSHA and full source copies are in report.json; stdout/stderr are lossless guards.log.gz with raw/compressedSHA.

- **B09 PASS:** real finish_mix_eligible and finish_view_valid helpers,27checks/4covers,0failures,0.1587seconds solver. Symbolic booleans/usize capability and full fouru64 offset/length domains, no assumptions/stubs. These are CPU admission/range properties, not a numerical or GPU proof.
- **B12 TIMEOUT, UNVERIFIED:** real SLRU queue implementation at capacities10/11, six fixed valid promotions (demotion exercised), then four symbolicu8slot/boolhit transitions and optional clear; unwind40 and assertions retained. The solver had no verdict/counterexample when the shared window expired. The source domain was not reduced to obtain success. The observed solver process was around4.8GiBRSS at49seconds; this is not a peak guarantee.
- **B11 NOT STARTED, UNVERIFIED:** the selection included its actual flag/inode guard harness, but Kani executes selected names lexicographically. The SLRU timeout occurred before the B11 solver started; translation/selection is not proof. Run it alone in a later coordinated CPU window if desired.

All owned subprocesses were stopped/joined and PGID19399 no longer existed before notifying root. No model, GPU, replay, tests or other proof was run by this agent in this window. Kernel math, Metal lifetime, Darwinfcntl effects, OS reads and concurrency remain outside these CPU properties. The existing source gates/replay still require root qualification.

## Separate B11 follow-up

The original not-started state above is preserved. A later parent-authorized isolated 30 s CPU window completed B11 alone: 40 checks passed, 4/4 covers, total 2.6353 s, all children joined. See [the separate report](b11-isolated/report.json) and [scope](b11-isolated/README.md). B12 remains timeout/unverified; its domain was not changed or rerun.
