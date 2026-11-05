# DEFAULT-AR10 lossless log archive

The two closed evidence directories and the parent controller driver are indexed in manifest.json. Every raw log/JSONL SHA-256 and size, compressed SHA-256 and size, readable JSON and controller snapshot hash is recorded. Every gzip was reconstructed and compared byte-for-byte; all original files remain present locally. CPU/Kani/build warnings and failures, if present, are preserved unchanged.

From repository root, restore one log with `gzip -dc GZIP_PATH > ORIGINAL_PATH` after creating its parent directory, then compare the raw SHA-256 in manifest.json. stage-paths.txt lists this archive plus readable reports/snapshots; exclude-paths.txt lists raw counterparts retained locally. No staging/commit, code, opti.md, benchmark, CPU suite, prover or GPU run was performed by this archive action. The five-phase qualification report remains the authoritative outcome; two short MAIN32 samples do not establish a general speed gain.
