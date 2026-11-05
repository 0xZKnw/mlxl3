# Closed AR20 evidence archive

Only profile, fast, components-01, quality, main32 and confirmation were archived, plus existing cpu-fast/cpu-common/proof-fast logs. kernels-confirmation was live and was not read, changed or included. Every original local file remains present. No benchmark, prover, compiler, linter or GPU job was launched.

manifest.json records each raw path, size and SHA-256 and each compressed path, size and SHA-256. Every new gzip was decompressed and compared byte-for-byte; source hashes were checked again. profile/events.json.gz was already compressed and is indexed with both compressed and decoded hashes. Human-readable reports, oracle and conditions JSON stay readable.

Restore one raw log from repository root using `gzip -dc COMPRESSED_PATH > ORIGINAL_PATH` (create its parent directory first). Verify compressed and reconstructed bytes against manifest.json. Raw fixture JSON/binaries are intentionally excluded from staging. Restore all 51 resident fixture files using `tar -xzf docs/measurements/flash-ar20-20261105/components-fixtures.tar.gz` from repository root. Its index lists every member/hash and archive/tar hashes; full reconstruction was checked, and all 48 binary outputs matched their stock Half words before archiving. Tar metadata is normalized, file contents are unchanged.

root-proofs/source-verdict.json binds the existing B01/B10 Kani log to byte-identical fast/common helpers, harnesses and callee. Terse log: 2 assertions, 7 covers, 2 successful harnesses. Exact invocation argv and process-group join receipt were not serialized in that raw log; these fields are explicitly absent, not invented. This is CPU guard evidence, not Metal or model proof.

stage-paths.txt lists exactly the archival scope files to stage; exclude-paths.txt lists raw counterparts retained locally. These lists do not cover unrelated source/research files. The whole live kernels-confirmation subtree remains excluded until root closes it. No file was staged or committed.
