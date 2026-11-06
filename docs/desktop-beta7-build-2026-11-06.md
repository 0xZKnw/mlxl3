# Desktop beta 7 local build — 2026-11-06

Desktop source `accabf48a8a6c2a57fb1daa05703adc7f41be330`, branch `codex/desktop-v2-beta7`, based on main `ed3bfdb8ff45e650fd8347b1a8f00931187284bf`. The original workspace and installed build37 were preserved. Production change for this preparation: Info.plist build37→38. Three regression fixtures were updated for already integrated behavior; no application Swift behavior was changed.

## Package and provenance

- `MLXL3-Desktop-v2.0.0-beta.7-b38-Apple-Silicon.dmg`: **247050083 bytes**, SHA-256 `431e7de3f8a1d6df0aad4944c9817b6230dc86d56dc1844fb523268f3a9e3385`.
- Reused published beta6 DMG SHA-256 `234ba954e167b7cce6e0eed7464c2f798cb960f94aaaf10c0893d03945fc7070`, checked against GitHub release metadata before extraction.
- Engine **2.0.2**, commit `9df28b6d2a10`, bridge1, MLX/chat enabled, aarch64. All four payload files plus engine.json remain identical, including modes. All3,050 converter regular-file/link records remain identical.
- New Swift release executable and build38 plist installed over a disposable copy of that published application. Only the outer app was signed; nested engine/converter signatures and payload bytes were retained. `build-info.json` records separate Desktop and reused-payload sources.
- Final DMG verified, mounted read-only, deep/strict signature and arm64 checked. Final mounted executable passed both real self-check flags. The production updater accepted internal2.0.0/build37→38, validated runtime hashes and ran the isolated list probe. No installed application was replaced.

## Commands and results

Executed with Swift6.4.2, Python3.12.14, SDK26.5 on AppleM5/24GiB/macOS27.2.

```sh
swift build --configuration release --package-path apps/MLXL3Studio --sdk /Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk
python -m pytest -q tests/test_build_manifest.py tests/test_copy_test_runtime.py tests/test_package_engine.py
MLXL3_MACOS_SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk MLXL3_TEST_PYTHON=/Users/justin/Documents/mix-stq1_0/.venv/bin/python scripts/check-desktop.sh
codesign --force --sign - --no-strict 'dist/MLXL3 Desktop.app'
codesign --verify --deep --strict 'dist/MLXL3 Desktop.app'
hdiutil create -volname 'MLXL3 Desktop' -srcfolder STAGE -format UDZO -imagekey zlib-level=9 DIST_DMG
hdiutil verify DIST_DMG
```

The full Desktop command stopped at stale assertions. Its unchanged successful groups were retained; the affected checks and the remaining groups were rerun using the exact production script sections saved in `build/desktop-beta7/check-*.zsh`. Final groups: `desktop-full-final.log` through preferences/notifications/logo, `desktop-rest-rerun.log` through sampling/experience/text-size, `layout-final.log`, then `desktop-after-layout.log` through the final E2E success. This is a completed segmented suite, not a claim that the initial command passed uninterrupted.

Package Python:94 passed/0 skipped. Studio Python:39 passed/0 skipped. CPU/native groups include240 memory oracle cases,49 composer states,1179 mode checks,114 idle assertions,62 preference assertions,17 notification assertions,25 logo checks,969 sampling assertions/216 states,35 experience assertions,51 text-size assertions/13 positions,16 layout renders,15 updater cases,22 transfers,34 HTML checks,150 import boundaries/256 encoding samples,229 workspace checks,141 thinking boundaries,1194 code fragments,4845 Markdown fragments,63 response parity cases,30 native long-response bitmap/OCR frames,300 download cases,64 tuning grids,39 Studio checks and44 full-window workspace checks. These different units are not summed into a fabricated test count. Bridge, sandbox, real command cleanup, cancellation, malformed data and error recovery passed. Network CDNs, conversion and model inference were not exercised.

The real DMG preflight harness is the existing updater application check combined with production UpdateManager, changing only its real-image currentVersion/currentBuild to2.0.0/37. Generated source, compiler command evidence, fixture logs and successful final updater log are retained. No user application, model registry or active engine pointer was modified.

## Retained failures and corrections

- System Python3.9 lacked tomllib. The retry used project Python3.12.
- Current build_manifest.py requires today's generated notices and rejected the unchanged published beta6 manifest. Rather than mutate the engine, provenance was assembled after independent byte/mode/link comparisons against the digest-verified beta6 DMG. The original engine notices remain intact; the receipt explicitly identifies reuse.
- Hardening fixture expected an empty Flash MTP head and an old shared model path. It now checks the actual target checkpoint, its saved preference and preservation of the other model's head. Initial failures remain in `desktop-full.log` and `desktop-full-rerun.log`.
- Sampling fixture assumed global MTP activation before metadata and a global OFF on model switch. It now tests remembered per-model intent separately from actual activation, preservation of another model's ON choice, shared MTP controls and unchanged ordinary sampling. The original/retry failures and the969-assertion success are retained.
- Initial remainder launcher omitted idle_sources; fixed in the saved operational launcher before sampling ran. Its error log is retained.
- The layout bound130 included newly expanded tuning guidance. Native captures confirm all depth segments are readable; the bound is160 without results,370 with results. All16 cases pass. A temporary copy restoring the historical labeled picker fails four cases under these bounds; production UI source was never mutated. Both selected captures were inspected.
- Nonfatal SDK/linker/deprecation, CoreGraphics, sandbox warnings are preserved in the relevant logs. Earlier candidate DMGs remain in first-package/second-package; their hashes do not attest to the delivered artifact.

## Limits and delivery

No Swift source verifier is available. No formal proof is claimed. No Rust/Metal engine changes or numerical inference checks belong to this Desktop preparation. The full unrelated engine/Python suites were not run for Desktop; later engine work is separate. Existing CI runs Desktop checks on PRs/main pushes, but no remote run exists for this local source. No push, merge, publication, release or installation was performed.

The delivered DMG, SHA-256 file, validation JSON and verification archive were copied byte-identically into `/Users/justin/Documents/mix-stq1_0/dist/`. Full local evidence remains at `/Users/justin/.codex/worktrees/desktop-v2-beta7/mix-stq1_0/build/desktop-beta7`. Release notes are in `docs/release-v2.0.0-beta.7.md`.

## Authorized publication

After the completed local build, the owner explicitly requested GitHub publication. The Desktop prerelease `v2.0.0-beta.7` targets the exact application source `accabf48a8a6c2a57fb1daa05703adc7f41be330`; these build notes are on `codex/desktop-v2-beta7`. The validation JSON and verification archive preserve the earlier local-build snapshot. No engine artifact is included. No exact-source GitHub CI run exists at publication preparation; the local results above are the available validation, not a green remote CI claim.

Publication completed and read back on2026-11-06 at01:22:54UTC: [Desktop beta7](https://github.com/0xZKnw/mlxl3/releases/tag/v2.0.0-beta.7). The public tag resolves to the exact application source above. All four uploaded asset sizes and SHA-256 digests match the local files; anonymous DMG HEAD returns200 with the exact byte length. The production updater parser selects Desktop beta7/build38, preserves stable1.5.1 and still selects engine2.0.2-beta.1. No installed app or engine was changed.
