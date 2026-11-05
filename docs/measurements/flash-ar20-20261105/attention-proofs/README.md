# B02/B04 CPU guard verification

Kani0.68.0 / CBMC6.11.0, actual Rust production helpers and harnesses in native/src/flash_support.rs, source commitb53427c4. Two targeted harnesses pass **641 checks /7covers**, no failures or timeouts. First combined command9.64s; one same-harness B04 regular diagnostic3.10s inspected the six unreachable checks. All child groups joined after each command; no GPU or other tests were run in this grant.

B02:68checks/4covers, symbolic booleans, keysu32 and all four pipeline capabilityu32s, no assumptions/stubs. Verifies dense short/private/trace guards and both actual pipeline capabilities. It does not model shader addressing, floats or execution.

B04:573checks/3covers, all boolean/fullu32 shape/format/split/capability values, unwind40 for fixed array comparisons, no assumptions/stubs. Verifies exact live pair admission, bias/residency/NT2/split1 and complete canonical output group counts. Six unreachable properties were inspected in resident-regular.log.gz: check28 std::fmt shift overflow, checks29/31/32/36/37 ptr_offset_from<()> zero-sized-type offset/div/mod model guards. None is a functional assertion or requested coverage point. No complete GPU/allocator/concurrency proof is claimed.

Commands, versions, budgets and process results are in report.json/regular-report.json; helper/harness verbatim function copies and SHA256 are in source-functions.txt/json for comparison after integration. These receipts are tied to the actual agent source; root must check those function hashes after cherry-picking instead of treating the entire root file as identical. Logs are losslessgzip with raw/compressed hashes in logs.json. Runtime math/physical components/bridge and performance remain parent-owned, not inferred from these CPU checks.
