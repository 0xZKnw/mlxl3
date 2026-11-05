# Expert prototype source handoff (A03–A06)

Base118a1b74, isolated branch codex/flash-ar10-experts. Source commits:46fa2b52 A03, b4a8e9a3 A04,7e19fc16 A05,b0a9162c A06,1f2c5321 QA distinct-Suh correction,8392a52d extracted finish binding correction. All four flags OFF, exact AppleM5 only. No compile/proof/GPU was run by this agent before handoff; formatting and diff-space checks passed. Root composes A02 dynamic NT2 selection into `FlashExpert::axis_nt` before common tests; four expert-axis fixed shape×NT PSOs already exist.

Flags in `Metal.expert_ar_enabled` and test-only `expert_ar_uses`, order:

1. `MLXL3_NATIVE_EXPERT_UP_FINISH`
2. `MLXL3_NATIVE_EXPERT_GU_PREPARE`
3. `MLXL3_NATIVE_EXPERT_OUTPUT_SLAB`
4. `MLXL3_NATIVE_EXPERT_AXIS_QMV`

The first/second/fourth usage counters count expert evaluations actually encoded on those paths; the third counts freshly allocated model contribution slabs (48 expected per successful untraced model row). `expert_axis_dispatches` counts the actual grouped QMV encode calls (three per nonempty ready cohort). These diagnostics do not claim GPU completion by themselves. Canonical standalone projection/static counters remain separate.

## Indispensable physical gates (not executed here)

Use the shared build's actual libtest, not an old executable. Set `MLXL3_ROUTED_SCRATCH_FIXTURES` to the existing independent real-record fixture directory containing `cases.json` and `stock.json`. The retained original is `/Users/justin/.codex/worktrees/issue75-native-flash-ar-round2/mix-stq1_0/build/ar-round2/quality-01/routed-scratch`; four routed cases0/1/0/1, distinct inputs, seed75114, final shared case. New gates do not rewrite this directory.

- `gpu::projection::routed_scratch_tests::physical_expert_ar_epilogues_match_independent_stock --exact --ignored --nocapture`: four A03/A04 combinations ×four records ×two strided dtypes;81,920 stock output words; both prepared prefixes vs canonical prepare_view; guaranteed distinct synthetic Suh and swapped-binding detection/restoration; ten reordered down output prefixes, duplicate identities,25,600 stock words, owner drop before commit and allocation request counts. Marker `NATIVE_EXPERT_AR_COMPONENT`.
- `gpu::projection::routed_scratch_tests::physical_expert_axis_matches_independent_stock_and_ready_cohorts --exact --ignored --nocapture`:48cohorts1/2/10×four records×two strided dtypes×A03/A04OFF/ON;532,480external stock words; distinct/permuted/duplicate records on a common input against a separately captured canonical branch; poisoned25,600-word slab; exact3QMV/cohort; invalid cohorts and uncommitted late error/recovery. Marker `NATIVE_EXPERT_AXIS_COMPONENT`.

Then root's common model test must check full finite logits/hidden/84 states plus continuation, actual A03/A04/A05/A06 branch counts, cold readiness/cohort activity and rollback/error cleanup. A05 trace remains legacy; A06 trace/shared/batch/otherformat fallback remains legacy. Real CLI golden/budget/prefix gate precedes MAIN32. No scalar or shader benchmark replaces those model checks.

## CPU verification prepared, not executed here

Real production helpers in `expert_ar_policy.rs` have deterministic tests and four Kani harnesses, manifest+CI regression list updated. `cargo test --lib --no-default-features expert_ar_`; actual-harness Kani commands use `cargo kani --lib --no-default-features --harness expert_ar_policy::<name>` under a private target. Names: expert_ar_prefixes_are_bounded_disjoint_and_never_overflow, expert_ar_selection_requires_exact_actual_record_extents, expert_axis_admits_only_nonempty_bounded_ready_tables, expert_ar_flags_are_strict_off_and_require_qualified_hardware. Domains: full usize/u32/u64 for table/offsets and boolean/routed extents; flag validUTF8≤4bytes/unwind10. Keep individual obligations/covers/unsupported/timeouts visible. No assumptions or shader rewrite stands in for Metal proof.

These CPU checks cannot prove MSL arithmetic, GPU scheduling, resource retention, I/O or concurrency. Metal source is the actual canonical body plus new glue; C++ CBMC cannot be claimed to verify Metal builtins/threadgroup semantics. Existing PR/main CI Kani inventory remains applicable; hosted CI does not execute Apple physical gates.

## Small source inspection

`suh-inspection.json` records only two fixture experts' real Suh payload comparison (20,480bytes read, no GPU/broad scan). Both differ strongly:2544 and2539 of2560halfwords. This does not assert all experts differ, nor is distinctness required for correctness. No per-encode memcmp or single-transform shortcut was introduced.

Memory and overlap tradeoffs: A04 doubles prepared capacity for the live expert; A06 uses a fresh cohort slab for prepared/partial, so tracked WAR dependencies and host vector bookkeeping may change overlap. A05 reduces allocation requests but preserves payload bytes. No milliseconds or throughput percentage is inferred from structural counts. Every result remains unverified until the shared build and physical gates execute.
