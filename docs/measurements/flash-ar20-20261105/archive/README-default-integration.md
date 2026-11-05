# Final root default integration archive

root-final-integration.json indexes the closed five-phase default integration, six CPU/build/style logs, native libtest JSONL, controller driver, readable freeze and source-proof-reuse metadata. Every raw/compressed SHA-256 and size is recorded. Every gzip was decompressed and compared byte-for-byte; all originals remain local. All previous top-level archive documents remain byte-identical and are bound by their hashes.

Use stage-paths-default-integration.txt alongside earlier stage lists; exclude-paths-default-integration.txt lists raw counterparts retained locally. Restore one log from repository root with `gzip -dc GZIP_PATH > ORIGINAL_PATH` after creating its parent directory, then verify the raw SHA-256 in the manifest. Human-readable JSON reports remain authoritative. This archive action reran no tests, build, benchmark, prover or GPU operation, and changed no code, controller or opti.md. No staging or commit was performed.
