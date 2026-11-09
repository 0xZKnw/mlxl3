# Hub integrity checks, 2026-11-08

Linux cloud workspace, no user Mac, model or GPU used. Commands use
CARGO_HOME=/tmp/mlxl3-cargo, RUSTUP_HOME=/tmp/mlxl3-rustup and the independent
CARGO_TARGET_DIR=/tmp/mlxl3-issue162-target.

- `cargo test --no-default-features --lib hub::tests --quiet`: 12 passed, 0 ignored,
  320 filtered out. Five new tests exercise content identity, resumable file
  admission, metadata parsing, commit cancellation and the production transfer
  helper using actual curl with disposable file:// fixtures.
- Mutation: replaced `Ok(actual.eq_ignore_ascii_case(expected))` with unconditional
  true in `RepoFile::matches`; the same-length corruption test failed as expected
  (exit 101). Original source restored, complete hub suite rerun successfully.
- `cargo clippy --no-default-features --lib -- -D warnings`: passed.
- `cargo fmt --all` and `git diff --check` on owned code/dependency files: passed.

Hash fixtures use independent standard SHA-256("hello") and Git blob("hello")
identities. Tests cover short partial retention, corrupt complete and oversized
restart, SHA mismatch, late callback cancellation, no rename on cancellation and
successful publication. They do not exercise remote HTTP 416, registry failure
rollback, live HF authentication, process death/crash recovery, Apple builds or
models. No performance result or full installation E2E validation is claimed.

The formal-proof skill was not present in the exposed skill catalog or local
skills directory. Kani availability/attempt is recorded by the integrating worker;
no formal verification of I/O, hashing, process supervision or signal races is
claimed by these tests.
