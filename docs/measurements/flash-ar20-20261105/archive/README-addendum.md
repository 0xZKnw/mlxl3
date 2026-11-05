# Final closed-campaign addendum

This adds the now-closed kernels-confirmation campaign, final CPU logs, all three Kani-manifest regression attempts (two failures followed by success), and five controller driver logs. The initial manifest.json and its original staging/exclusion lists remain unchanged. fmt-final.log was already archived identically in the initial manifest and is referenced there, not duplicated.

addendum-final.json records the initial manifest hash, every new raw/gzip SHA-256 and size, retained readable metadata and the preserved failure/success sequence. Every new gzip was round-tripped byte-for-byte and all original local files remain present. A first create-new collision with the already archived format log stopped safely before writing the addendum; this attempt is recorded, original bytes retained. No code, controller, Rust, opti.md, prover, compiler, linter or GPU run was changed/launched.

Use stage-paths-addendum.txt for this addendum only, alongside the original stage-paths.txt. exclude-paths-addendum.txt lists raw logs retained locally. Decompress with `gzip -dc GZIP_PATH > ORIGINAL_PATH` from repository root after creating its parent directory, then compare the manifest SHA. Human-readable kernels report JSON remains staged directly.

Parent reports final no-direct strict Clippy PASS, 379 tests PASS/10 ignored, format PASS, and 7 pytest PASS/16 subtests plus Ruff check/format PASS. No command was rerun for this archive. The empty format log is preserved but is not itself treated as an independent exit-status receipt.
