# MLXL3 Desktop 2.0.0-beta.2 — build 32

Prerelease: `v2.0.0-beta.2`, latest=false; internal Desktop/engine version 2.0.0. Built from [`0842599c24461b3395b4d2d2bd9bf617eb739798`](https://github.com/0xZKnw/mlxl3/commit/0842599c24461b3395b4d2d2bd9bf617eb739798). Models and conversations remain separate from the application.

## New features

- **Download the compatible draft** button for Flash Next: about 1.2 GB, a pinned IncoAI revision, SHA256 integrity checks, caching, resume, progress and cancellation. Local folder selection remains available; the downloaded path is saved after the engine acknowledges it.
- **Stable / Beta** selection in Settings → Updates. Stable is the default; Beta enables prereleases. Changing channels clears the previous update offer; Search starts the check, without an automatic downgrade.
- Four square cards in Models → Discover: Qwen3.6 35B-A3B 2.49 bpw, Qwen3.8 27B 2.00 bpw, Flash Next 2.05 bpw and the public LFM2.5 24B-A2B v2. Links use the revisions recovered from local download records. Local LFM v3/CUDA conversions are separate from this v2.

## Fixes

Unavailable Tune support for Flash Next no longer suggests a missing engine update; Auto/2/7 remain selectable. The bundled engine takes priority over an older managed engine with the same version, so beta1 cannot mask the beta2 fix.

## Performance

No new throughput gain is claimed, and no GPU benchmark was repeated for this update. The inference calculations and experimental settings from beta1 are retained.

## Installation and compatibility

Apple Silicon, macOS 26.2+, ad-hoc signing without notarization. Install this DMG, or select Beta and then Search in Desktop 1.5.1. The corrected engine ships with Desktop beta2: standalone engine updates do not distinguish two tags sharing the internal version 2.0.0. The target checkpoint and draft must still be downloaded separately if absent.

## Verification and limitations

Targeted checks: 30 native tests passed/2 ignored, formatting/Clippy on CPU and MLX, the real CLI and SHA256-verified RAW cache; 9 download scenarios and 8 local/legacy scenarios; two updater suites including 15 signed disposable application cases; strict Swift 6 compilation and 229 workspace checks. No new GPU generation was run. Kani was attempted on two real contracts but hit an internal compilation error before generating obligations: these contracts remain unverified. These tests are not a complete proof. Package and CI checks: signatures, arm64, four runtime hashes, relocated CLI/cache and read-only DMG checks passed. At the recorded CI snapshot for the final source, three workflows were still running; CI was not declared fully green. An inherited absolute RPATH remains present, but it was not used by the libraries loaded from the package.

Final identities verified: bundle and engine `0842599c2446`, built without local modifications. Rejected intermediate attempts remain in the evidence; the legacy dense-RAM advice has been corrected for Flash SSD.

## Artifacts

| File | Bytes | SHA-256 |
|---|---:|---|
| MLXL3-Desktop-v2.0.0-beta.2-b32-Apple-Silicon.dmg | 74117391 | `83999f5e83d051ab55503d604e8804617d5a0eae84b2116b2572a545f140096d` |
| MLXL3-v2.0.0-beta.2-validation.json | 3307 | `2f85b719e00ad9174e1c6e9c1f732eb8a6f2389737e9a93d527371e077b81ef0` |
| MLXL3-v2.0.0-beta.2-verification.tar.gz | 9732 | `c45948a9bb071d8baa86e48685375ea5ab6e915f4f7993d2a6bddc352ef03d7a` |

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for MLXL3 and the LFM conversions, [IncoAI/Splash](https://github.com/incoai/splash) for DFlash2, [UnstableLlama](https://huggingface.co/UnstableLlama) and [turboderp](https://huggingface.co/turboderp) for the EXL3 checkpoints, and [MLX](https://github.com/ml-explore/mlx) for the Apple Silicon runtime.
