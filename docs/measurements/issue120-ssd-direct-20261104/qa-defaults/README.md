# P01 qualified M5 default verification

The final actual flag helper is checked with arbitrary hardware qualification and valid UTF8 strings of length0..4, plus absence, unwind10. Kani0.68/CBMC6.11 passes240checks and7covers, no assumptions/stubs; seven std fmt/align/UTF8 guards are unreachable, no functional assertion. Explicit0/1 and malformed values keep their prior contract; absence follows qualified hardware. Extent and eligibility proofs are reused unchanged.

A standalone test contains exact production direct_flag/hc_chain_flag bodies and the exact current None assertion. Original passes0; changing only the default argument from qualified_m5 to false fails101; the same original executable passes0 again. The worktree helper and qualified binaries were never mutated. This verifies the small helper regression, not complete model generation, OS memory initialization, resource ownership or concurrent interleavings.

Physical fixtures capture initial_direct_enabled immediately after Metal::new, requiretrue before test overrides, and emit it in both expert/model branch markers. Actual flag absence and real backend counters still require the coordinator's final physical qualification. No final GPU run or throughput measurement is claimed by this receipt.
