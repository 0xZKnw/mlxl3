# Qualified P13 default: CPU policy verification

The only production change is the default argument in hc_gdn_chain_flag: absence now follows qualified hardware; explicit0rolls back, explicit1still requires exact M5 qualification, malformed values remain errors. GDN/HC S2 code, allocations, shader/math and lifetime adaptation are unchanged. Earlier defaultOFF receipts in../qa remain intact.

Three targeted unit tests pass, strict no-default CPU Clippy all-targets passes, the full no-default suite passes336tests/8ignored, and the exact Kani manifest policy passes2tests. Exact commands, limits, timing and process joins are retained in cpu-report.json. Ignored tests were not run by this suite.

Both real helper Kani harnesses pass: eligibility43checks/8covers over complete u32shapes and boolean guards, qualified default/literals240checks/7covers over absence/hardware/presence and valid UTF8 strings up to4bytes, unwind10. No assumptions or stubs. Seven std fmt/align/UTF8 guards are unreachable; no functional assertion, and all15covers are satisfied. Kani0.68/CBMC6.11, Rust1.98.1. This is bounded CPU source verification, not Metal, OS, concurrency, actual device-name mapping or complete generation proof.

A disposable standalone copies the actual three helper bodies and uses the actual current policy module. Changing only the default qualified argument back tofalse makes the hardware-default unit fail101. Original passes0 and its same executable passes0 again; worktree source and frozen binaries were never mutated. Sources and logs are retained without compiled mutation executables.

All owned CPU/prover/test processes joined before handback. Actual flag absence and active joints still require separate final GPU qualification. Prior S2 numeric/timing artifacts remain distinct; this archive neither reruns a GPU nor claims a new performance result.
