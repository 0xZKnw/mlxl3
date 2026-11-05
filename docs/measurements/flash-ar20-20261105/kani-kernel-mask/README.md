# AR20 native compiler mask Kani check

Source: `df245b8f3b894bfc93942d9a39cb73243fecf263` (hashes in `report.json`). Target-private, offline, one job; root CPU-only grant. No GPU or other tests ran.

```sh
CARGO_TARGET_DIR=build/ar20-kernel-mask-kani-target CARGO_BUILD_JOBS=1 CARGO_NET_OFFLINE=true cargo kani --lib --no-default-features -Z stubbing --exact --harness ar20_kernel_policy::native_kernel_mask_preserves_every_actual_compiler_switch --jobs 1 --output-format regular
```

Kani 0.68.0 / CBMC 6.11.0. Completed in 26.77 seconds including versions/compilation (90-second process-group deadline); solver 0.439 seconds. One harness passed: 216 successful obligations, six unreachable Kani/std internal obligations, zero failed; four of four covers satisfied. No assumptions. The real `mask` implementation preserves every flag over four symbolic booleans (all 16 masks), stays within four bits, and covers masks 0, 15, 5, 10. All mask assertions were reachable and successful.

The six unreachable checks concern ZST pointer offsets and std formatting; unsupported `caller_location`/foreign-function warnings did not become reachable. Compile warnings also include existing unused paths, Kani register_tool and existing semicolon future compatibility. The child group was joined and absent after exit; source hashes remained unchanged.

This is bounded source model checking of the CPU mask, not proof of Metal arithmetic, source defines formatting, shader compilation, the environment parser, allocation, concurrency or model generation. The separate CPU defines regression and physical gates remain for root qualification. Complete raw output and version logs are retained.

Raw `.log` files are preserved losslessly in adjacent `.log.gz` files; run `gzip -dk *.log.gz` here to restore report paths. See [log-archive.json](log-archive.json) for raw/compressed hashes.
