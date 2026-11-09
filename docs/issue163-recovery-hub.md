# PR163 recovery: checkpoint transfers

Reconstructed on 2026-11-09 from the current recovery checkout. Older unpublished
sources and logs were unavailable; they are not evidence for this patch.

## Corrected behavior

- The final download limit uses `RLIMIT_FSIZE` in the curl child's Unix `pre_exec`
  hook, not curl `--max-filesize`. Intermediate 302/307 response bodies may be
  larger than the final checkpoint. The child limit bounds the complete output
  file, including a resumed prefix. The parent process limit is unchanged.
- Metadata response limits remain in place; pinned identities and streaming
  SHA-256/Git blob SHA-1 checks remain mandatory. File-limit failures include an
  explicit diagnostic, other subprocess failures include their exit status.
- Successful registration records `completed.json` before best-effort staging
  cleanup. Pending discovery skips only a well-formed matching installation
  identity (ID, repo, commit, folder, name, destination, size). Invalid/foreign
  markers do not hide pending work. A retry removes the marker under the existing
  exclusive download lock before metadata requests or other fallible work.
- Additive `download_with_phase` and `resume_with_phase` APIs report `download`,
  `verify`, `publication`, and `install`. Existing two-argument callbacks remain
  supported by wrappers. Verification is reported even when byte progress has
  already reached 100%; installation is reported before registry installation.

## New evidence

`docs/measurements/issue163-recovery-hub/` contains fresh setup/check logs.
The HTTP fixture exercises the production checkpoint helper and checksum with
302 and 307, each initial and resumed (`he` to `hello`), with 128-byte redirect
bodies and a five-byte final payload. Separate 200/206 oversize cases assert an
error and on-disk length no larger than five. Threads have shutdown signals,
read/write deadlines, and joined teardown. Staging tests cover retained success,
malformed/foreign markers, and same-identity retry invalidation.

Mutation: restoring curl `--max-filesize` made the redirect test fail with curl
error 63. The production source was restored, and the tests rerun.

Kani was attempted for `checkpoint_digest_family_and_syntax_are_required`; the
command is not installed. The harness now has 65 independently symbolic ASCII
bytes rather than a single repeated character. No formal proof is claimed.
Initial Clippy diagnostics for unnecessary unwraps and a target-dependent
conversion were corrected; the initial diagnostic log is retained.

## Limits

No Mac, Swift runtime, model, Metal or speed benchmark was used. These checks do
not prove crash consistency of the filesystem or registry transaction. A process
crash between registration and marker persistence, or failure to write the marker
and clean staging, can still leave discoverable residual staging. Markers are not
authoritative evidence that the installed files remain intact. Cross-platform
Unix compilation and real Hugging Face authentication/network behavior still need
qualification; Linux tests do not constitute that qualification. HTTP metadata
caps and generic download timeout/transport design were not removed or redesigned.

## Final checks

- `cargo test --no-default-features --lib`: 350 passed, 9 ignored, 0 failures;
  38.66 seconds, recorded in `cpu-suite.log`.
- Hub selection: 23 passed, 0 ignored, 18.74 seconds; final restored-code run
  also recorded in `tests-final.log`.
- `cargo clippy --all-targets --no-default-features --features chat -- -D warnings`:
  passed, 28.97 seconds, including the additive phase API's CLI callers.
- `cargo fmt --all` and `git diff --check` on owned sources passed.
- Kani attempt failed because cargo-kani is absent; not a proof or counterexample.

Commands used the official Rust minimal installation and isolated cloud target
`/tmp/pr163-hub-target`. No former environment's results were reused.
