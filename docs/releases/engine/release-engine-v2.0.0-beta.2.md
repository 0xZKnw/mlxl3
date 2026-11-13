# MLXL3 Engine 2.0.0-beta.2

Prerelease `engine-v2.0.0-beta.2`, latest=false, internal version 2.0.0. Built from [`0842599c24461b3395b4d2d2bd9bf617eb739798`](https://github.com/0xZKnw/mlxl3/commit/0842599c24461b3395b4d2d2bd9bf617eb739798).

## New features and fixes

The dflash-draft command recognizes the Flash Next checkpoint and prepares its compatible RAW draft, separate from the 35B/27B packages. It validates the config, 81 BF16 tensors, sizes and pinned SHA256 hashes before atomically publishing the cache. Cache reuse, resume and cancellation are retained. CPU geometry validation is shared with the existing loader; no Array calculation or kernel changed.

## Performance

No new tokens/s gain or GPU campaign. The qualified local copy was reused with zero network download after full hashing.

## Installation and compatibility

Apple Silicon/macOS 26.2+, MLX 0.32.2. Use Desktop 2.0.0-beta.2/build 32 to receive the button and the preference for the bundled engine as well. The internal version 2.0.0 remains numeric: a beta1 engine already reporting 2.0.0 is not offered a standalone update; Desktop beta2 bundles the fix. No model is included in the archive.

## Verification and limitations

30 targeted tests passed/2 ignored, fmt, CPU/MLX Clippy and MLX compilation; the real CLI inspected the local draft, rejected invalid files/targets and verified the fully hashed managed cache. Kani attempts on two contracts stopped with an internal compiler error before obligations were generated: they remain unverified. No new numerical GPU test; existing calculations are retained. Package and CI checks: signatures, arm64, four runtime hashes, relocated CLI/cache and read-only DMG checks passed. At the recorded CI snapshot for the final source, three workflows were still running; CI was not declared fully green. An inherited absolute RPATH remains present, but it was not used by the libraries loaded from the package.

Final identities verified: bundle and engine `0842599c2446`, built without local modifications. Rejected intermediate attempts remain in the evidence; the legacy dense-RAM advice has been corrected for Flash SSD.

## Artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| MLXL3-Engine-v2.0.0-beta.2-arm64.json | 13588 | `9b33019725439e64fa33483967dd696c3da25bef2af1d410faaea0261e96c1b6` |
| MLXL3-Engine-v2.0.0-beta.2-arm64.tar.gz | 67254401 | `8922476474341ffd3db73f586067b29ea04f56bb81e216fe6c4e96e25b2d6b9a` |

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the integration, [IncoAI](https://github.com/incoai/splash) for the DFlash2 draft, [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [PonyExl3](https://github.com/beamivalice/PonyExl3) for EXL3, and [MLX](https://github.com/ml-explore/mlx) for the runtime.
