# Prepared AR20 native coverage inventory

**Prepared for root integration; check-mode validation is pending.** Generated with root's unchanged `scripts/check_native_coverage.py --root ROOT --inventory` (exit0), reviewed against its current full reference. Root source/manifest/README were not edited. No Cargo, prover, model, GPU or benchmark was run.

The complete inventory moves from 258 to 272 files: 14 new inventory-bearing files, 9 changed files, 0 removed files, 76 lexical counter changes. There are 218 distinct Rust configuration keys and 112 distinct literal shader/kernel references; 75 files remain in the PonyEXL3 sub-bundle. `mlx-sites.previous.json` preserves root's old reference byte-for-byte; `mlx-sites.prepared.json` is the complete candidate replacement, not a patch. [Full reviewed delta](delta.txt), [hashes/categories](report.json).

The delta includes inherited AR10 sites A01–A12, including A10 joint a/b fields/shader and the dormant-PSO corrections, as well as AR20. Entries introduced by the PR132 test-counter cfg repair are lexically unchanged because the guarded field names do not enter this inventory. No prior entry was dropped merely because a test or option was disabled.

Review of all 76 differences:

- 39 configuration occurrences identify existing AR10 flags, twelve AR20 flags, physical fixtures and the offline capture/report paths; they are not loaded MLX modules.
- 7 shader literals are embedded native Metal headers/kernels (including the AR10 whole-GDN/joint a/b and expert-axis sources); no external `.metallib`/MLX dependency is introduced.
- The remaining 30 counter changes are native helper/test sites or lexical homonyms: `std::array::from_fn`, Option/Result `.transpose`, MatrixView `.shape`, frame `.finish`, SSD job `.run`/`.ready`/`.start`, Option `.take`, File `.try_clone`, pointer `.add`, scalar Half `.to_f32` and iterator `.sum`. The checker intentionally retains those counts. No new MLX FFI symbol, Python MLX import/call, MLX dylib/wheel dependency or removed migration site appears in the reviewed delta.

The checker CLI supports `--root` and `--inventory` only, with no temporary-reference option. Per root instruction, neither normal check-mode against a swapped reference nor the mini test suite was run before integration. The existing four regression tests already exercise new/removed sites, aliases, file additions, method/FFI accounting and finite dictionary boundaries; the checker was not changed. After copying the prepared manifest to `docs/native-coverage/mlx-sites.json`, root should run `python3 scripts/check_native_coverage.py` and `python3 -m unittest discover -s tests -p test_native_coverage.py`, then inspect CI for that exact commit. A lexical inventory pass is not a runtime dependency proof.

Integration artifacts:

- `mlx-sites.prepared.json` → `docs/native-coverage/mlx-sites.json`.
- `native-coverage-README.prepared.md` → `docs/native-coverage/README.md` (full copy with the historical counts corrected and AR20 status appended).
- `attribution-and-options.prepared.md` is a section for root's final AR20 README; its relative evidence links are relative to this preparation folder and must be shifted one level if pasted into the AR20 folder. Speed selection/results remain root-owned.

All 12 candidates are experimental OFF by default. B01 and B12 are excluded from the current AR speed composition for documented reasons; quality qualification is distinct from speed or integration. No app/merge/release claim is made.
