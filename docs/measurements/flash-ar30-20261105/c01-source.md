# C01 source handoff

Base `574e4a8c4bed718a2b8983bb16e44e2a48cce9d6`, isolated branch
`codex/ar30-cooperative`. Issue135 and the root matrix predate implementation.
Source only; not committed or numerically/performance qualified at handoff.

`MLXL3_NATIVE_QMV_COOPERATIVE` uses the existing strict experimental flag parser:
absent/0 OFF,1 ON only on exact AppleM5, invalid/non-Unicode errors. It does not
change the13 accepted defaults or any AR20 option. The resulting immutable
`MLXL3_AR30_COOPERATIVE` define modifies the shared native QMV header for K2/3/4
only, excluding MTP's special reduction. Other K and the OFF source retain the
previous reader. Every lane owns at most one unique packed u32 and participates
in both shuffles per window. No repack, output change, extra buffer, encoder,
barrier or production PSO. Existing per-lane metadata, codeword decoder, Half
materialization, eight FMA chains and final reduction are unchanged. C01 takes
priority over reader8 for K2–4; preload stages the same cooperative windows if
explicitly composed. Neither prior option is enabled by this change.

Production callers are resident/static/fixed/dense projections and the routed
single-record/expert-axis PSO families that include `_qmv_tile_kernel.metal`.
Test-build counters `Metal.ar30_cooperative_uses[0,1]` increment only after actual
resident/routed dispatch encoding. These count dispatches, not bytes or speed.
The compiled flag is independently read from the Metal library, not inferred
only from the environment. The existing common capture reports it and both
counters, refuses an ON reference capture and requires positive per-forward
deltas for both paths when C01 is selected. No additional model test/load was
added: the parent owns one common composition gate.

Prepared component entry points, not yet executed:

- `gpu::ar30_kernel_tests::physical_ar30_cooperative_words_preserve_every_window_bit`:
  K2..8, every single packed bit as a basis vector, plus four fixed tile patterns,
  all lanes/eight windows/ring wrap.2,300,928 state words;8960 basis vectors.
  Cooperative helper outputs agree with **both** canonical Metal `state()` and
  independent CPU `codec::unpack`. A wrong-owner-index mutant must fail for each
  supported K2/3/4. This is a finite exhaustive bit-basis check of a linear,
  data-independent extraction, not enumeration of every packed tile or proof of
  Metal execution. K5..8 exercise the test reader's explicit canonical fallback;
  production fallback coverage additionally comes from the resident fixture set.
  Marker `AR30_COOPERATIVE_WINDOWS`.
- `gpu::projection::ar20_projection_tests::physical_ar30_cooperative_preserves_frozen_routed_stock`:
  existing frozen stock records, F16/F32 inputs and complete finite2560-word
  outputs, requires actual routed dispatches. Marker `AR30_COOPERATIVE_ROUTED`.
- `gpu::projection::physical_flash_resident_nt2`: existing independent full
  fixture matrix and output files unchanged, additional ON-only compiled-flag
  and resident-dispatch assertion/marker `AR30_COOPERATIVE_RESIDENT`.

There are three lazily created test PSOs (compiled flag, candidate+canonical
windows, wrong-owner mutant), and zero added production PSOs. All physical tests
remain explicitly ignored and exclusive; the parent must copy fixture metadata
into a private directory as before to preserve the original frozen captures.

Rust introduces no new host eligibility/mapping algorithm: selection reuses
`flash_support::hc_kernel_flag`; numerical mapping remains Metal. The relevant
existing actual parser harness is
`flash_support::hc_kernel_flags_are_off_by_default_and_require_exact_m5_literals`
(symbolic presence/M5/UTF8 bytes0..4, unwind10). Attempt it after grant; it does
not prove SIMD behavior, device buffers or kernel arithmetic. A verifier for the
actual Metal SIMD context is not available in the existing proof setup; do not
replace it with a rewritten CPU function and call that a Metal proof.

Source formatting completed with `/Users/justin/.cargo/bin/cargo fmt --all` and
`git diff --check`. Initial `cargo fmt --all` could not resolve cargo from PATH;
no compilation was attempted and the explicit binary resolved the formatting
step. No Cargo build/test/Clippy, Kani/CBMC, GPU test, model/weight read, benchmark
or push has run. Native-coverage inventory must be refreshed honestly after
the three agents' source union (`gpu.rs` gains one config key and one shader
include); no claim that new GPU coverage has already passed. Existing PR/push
CI includes Rust/direct-Metal builds, native inventory and the parser harness.
