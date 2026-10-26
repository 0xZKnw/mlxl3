# MLXL3 Desktop 1.5.1 — build 33

## New features

A **Stable / Beta** selector is now available in update settings. Stable is the initial setting; the selection is saved for future checks. The Beta channel makes compatible MLXL3 prereleases available.

Desktop and engine releases are still selected separately. This stable release updates only the Desktop; no engine 1.5.1 is published. Filenames, GitHub URLs and SHA-256 hashes are still checked before installation.

## Fixes

Results from an older check cannot replace the current channel's results after a channel change or cancellation. An external engine with exactly the same version as the bundled engine no longer masks the new bundle's engine.

## Performance

No new throughput or memory gain is claimed. This release starts from stable 1.5.0 and ports only update channels and their guards; the stable 1.5.0 engine code is unchanged.

## Installation and compatibility

Artifact: **MLXL3-Desktop-v1.5.1-b33-Apple-Silicon.dmg**, Apple Silicon arm64, macOS **26.2+**, bundled engine **1.5.0**, MLX **0.32.2**. Ad-hoc signing without Apple notarization.

Choosing a channel does not install an update or downgrade the application. To return from a beta to a lower stable version, install it manually and keep a backup of your data. Selecting a channel does not move models or history.

## Verification and limitations

Both targeted Swift 6 compilations with warnings treated as errors passed, along with the selector/updater and installer suites: channel selection and persistence, invalid tags/assets/URLs/hashes, cancellation, hostile archives, activation/fallback and 15 signed compatibility/integrity cases. The first installer test interruption inside the sandbox is preserved; the same binary passed outside the sandbox without a code change.

The [targeted report](desktop-v1.5.1-verification.md) distinguishes these checks from the final package. The official package was built from [`b7b3a8a136e496a8de77b89f704da2f590851836`](https://github.com/0xZKnw/mlxl3/commit/b7b3a8a136e496a8de77b89f704da2f590851836): deep/strict signature verification, arm64, four engine hashes, the real 1.5.0 CLI and exact read-only DMG comparison all passed (11 commands). At the recorded CI snapshot for this SHA, conversion had passed and other jobs were still running. The unchanged 1.5.0 engine code was rebuilt for this bundle; no standalone 1.5.1 archive was published. No new generation, speed measurement or formal Swift proof is claimed.

## Acknowledgements

- [@0xZKnw](https://github.com/0xZKnw), for requesting the Stable/Beta selector and validating the product on Mac.
- [MLX](https://github.com/ml-explore/mlx), the retained local compute engine; [SwiftMath](https://github.com/mgriebling/SwiftMath), the retained math renderer. The stable release's licenses and other attributions remain in the repository.

## Artifacts and source

Package source: `b7b3a8a136e496a8de77b89f704da2f590851836`.

| File | Bytes | SHA-256 |
|---|---:|---|
| MLXL3-Desktop-v1.5.1-b33-Apple-Silicon.dmg | 73809680 | `19ea82857a04690cd04de6fbd8da88860330f90426368bbac4e6ba7588826253` |
| MLXL3-v1.5.1-validation.json | 1592 | `71f71f8f7ce0683bb020e965cd02629c3f4233a07c344d14d8f0c4431dfdd5b8` |
| MLXL3-v1.5.1-verification.tar.gz | 5056 | `61af918451f190d8845e9813b6bc9854c0ff9ce3dd9fa4a167fc9083b2f6b2d8` |
