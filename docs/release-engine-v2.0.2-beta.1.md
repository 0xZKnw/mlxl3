# MLXL3 Engine 2.0.2-beta.1

Prerelease `engine-v2.0.2-beta.1`, internal engine version **2.0.2**, published with `latest=false`. This runtime is also bundled with Desktop **2.0.0-beta.6 / build 37**. Release source: **PENDING_SOURCE_SHA**.

## Features

- Include the qualified Flash-Next Cold expert overlap and dynamic SSD readers from main [`4c1d5ecffbd0a04e14385db5c286b947a634f73a`](https://github.com/0xZKnw/mlxl3/commit/4c1d5ecffbd0a04e14385db5c286b947a634f73a). Ready Cold work can start while remaining reads finish; dynamic readers distribute the pending reads across the existing workers.
- Include the Flash-Next MTP prefill change that computes the final two required heads instead of every prompt head. The existing state, continuation and output contracts are retained.
- `MLXL3_SSD_COLD_READY`, `MLXL3_SSD_DYNAMIC_READERS` and `MLXL3_FLASH_PREFILL_HEADS` default to `1`; explicit `0` overrides remain available. These defaults do not enable every experimental Flash option.

## Performance

The following are **historical measurements of their recorded sources**, not fresh measurements or physical qualification of this release package. Apple M5 / 24 GB, MLX 0.32.2, Flash Next EXL3 2.05 bpw (`65c89531`), 3 GiB expert cache and four readers were used. The tested profile also enabled Hot/Wide/Unique/Overlap; Hot/Wide/Unique are **not enabled by default**. Other models and settings may behave differently.

Dynamic readers composed with Cold U showed aggregate short decode changes of **+0.535% AR, +0.960% MTP and +1.030% DFlash**, against Static readers with Cold U. Short MTP prefill time increased **0.700%** in the complete sample; this tradeoff was explicitly accepted. See the [campaign and retained samples](https://github.com/0xZKnw/mlxl3/blob/main/docs/measurements/post-p09-dynamic-cold-20261101/README.md) and [default qualification](https://github.com/0xZKnw/mlxl3/blob/main/docs/measurements/post-p09-dynamic-default-20261101/README.md).

The MTP prefill-head campaign observed about **4.8–4.9% less short prefill time** and **2.81% less long prefill time**. Long decode was **0.738% lower** in its small C-A sample; this is retained uncertainty, not a universal decode improvement. Its source and results are recorded in [the P09 report](https://github.com/0xZKnw/mlxl3/blob/main/docs/measurements/decode-profile-p09-20261101/published-issue-body-final.md). Results from different campaigns are not added together, and no universal speedup or RAM saving is claimed.

## Fixes

- Include the integrated macOS capability checks and draft-mode transition cleanup qualified with the Cold overlap runtime. An invalid MTP-head load retires the preceding DFlash draft before reporting failure.
- Preserve read completion, error recovery, canonical expert ordering and bounded ownership through the qualified Cold paths.

## Installation and compatibility

Apple Silicon, macOS **26.2 or later**, MLX **0.32.2**. The archive contains the native executable and its runtime libraries, not model weights. Engine and Desktop versions remain independent; internal **2.0.2** advances beyond the **2.0.1** engine bundled with beta 5.

Prepared Metal, Cold Shared, DEC-ONE, MB3/MB4 reference projections, Cold priority and routed-index caching remain **experimental and OFF by default**. Cold-ready width remains **2** by default; width **4** is opt-in. Their presence in the source does not establish a delivered speed benefit. No new experimental flag is enabled for this release.

## Verification and limitations

Current CPU/build/package checks: **PENDING_VERIFICATION_RESULTS**. Exact-source GitHub CI status: **PENDING_CI_STATUS**.

No inference, quantization, Metal/WebGL execution or new physical benchmark was run for this release preparation, as requested. Earlier component, state/logit, CLI and concurrency qualifications are preserved with their original identities; they do not establish physical parity or performance of this newly built package. See the [integrated runtime qualification](https://github.com/0xZKnw/mlxl3/blob/main/docs/measurements/merge-validation-20261101/README.md).

Historical Kani attempts for the affected Cold paths encountered a compiler internal error; CBMC could not parse the required C++ headers. These formal properties remain unverified. Tests, sanitizers and finite physical checks are not complete proofs. Newly guarded experiments have deferred numerical/performance validation and remain disabled.

## Artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| PENDING_ENGINE_MANIFEST_FILENAME | PENDING_BYTES | `PENDING_SHA256` |
| PENDING_ENGINE_ARCHIVE_FILENAME | PENDING_BYTES | `PENDING_SHA256` |

Package provenance, signatures, architecture and runtime-file hashes: **PENDING_PACKAGE_RECEIPT**.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3 and the integration recorded in [PR64](https://github.com/0xZKnw/mlxl3/pull/64), [PR68](https://github.com/0xZKnw/mlxl3/pull/68) and [PR69](https://github.com/0xZKnw/mlxl3/pull/69); [Apple MLX](https://github.com/ml-explore/mlx) for tensor execution; [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [PonyExl3](https://github.com/beamivalice/PonyExl3) for EXL3 work; and [IncoAI / Splash](https://github.com/incoai/splash) for the DFlash2 draft. Upstream notices remain included in the runtime package.
