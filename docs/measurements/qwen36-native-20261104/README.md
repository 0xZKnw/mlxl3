# Qwen3.6 native Metal validation receipts

Historical base: `28f1a84e5371775205aeace22f4bf391a301a884` plus the Qwen port. The source and artifact hashes identify the exact local versions checked; these receipts do not qualify a later merge automatically. See `integration/` for checks after merging upstream main `dd91e12d7baecd63bb008d8893026fd283c42a5b`.

`published-logs.json` maps original local log names to the compressed logs committed under `logs/`, including unsuccessful attempts. The old absolute paths in receipts are provenance, not portable installation paths. `source-final-sha256.json`, `delivery.json` and `artifacts.json` refer to the earlier frozen artifacts. The initial diagnostic libtest executable was overwritten during localization; its hash/log survives, not its bytes. Later diagnostic binaries remain local.

Small numerical/bridge/operator reports, full-model manifest, checkpoint hashes and host/prover logs are included. The 249 MiB independent raw model oracle, model weights, compiled binaries, compiler caches, diagnostic source patches/buffers and raw event streams remain local. They have not been deleted. The focused 512-value counterexample is included in `native/tests/fixtures/qwen35-glu-counterexample.json` and runs against the real mapped shader.

No throughput or memory improvement is claimed. Ignored physical tests are selected explicitly in the recorded GPU runs; other ignored tests remain unexecuted. The successful Kani cursor check covers its real scalar bounds/publication contract, not Metal execution, memory ownership across drivers, concurrent behavior or complete model correctness. The broader geometry repeat was interrupted; CrossHair was unavailable.
