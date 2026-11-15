# MLXL3 Engine 2.0.11 beta 3

Independent engine prerelease for Apple Silicon. It is 2.0.11 beta 2 plus one fix to how Qwen3.8 Flash Next sizes its memory reserve. Desktop 2.0.2 bundles this engine. Decoding, MTP, DFlash2 and the defaults introduced in beta 2 are unchanged.

## Fixes

- **Flash Next no longer refuses to load on most 16 GiB Macs.** The automatic SSD expert cache kept a fixed 8 GiB reserve, sized on a 24 GiB Mac. Every Mac therefore needed 8.56 GiB of free plus file-cache memory before Flash Next could load. The engine itself needs about 4.68 GiB outside the expert cache. The reserve is now 5 GiB for the engine plus a system margin of one eighth of physical RAM, kept between 1 and 3 GiB.

  | Physical RAM | Reserve, beta 2 → beta 3 | Minimum available memory, beta 2 → beta 3 |
  |---|---:|---:|
  | 8 GiB | 8 → 6 GiB | 8.56 → 6.56 GiB |
  | 16 GiB | 8 → 7 GiB | 8.56 → 7.56 GiB |
  | 18 GiB | 8 → 7.25 GiB | 8.56 → 7.81 GiB |
  | 24 GiB or more | 8 → 8 GiB | unchanged |

  The 6 GiB cache ceiling, the 10-slot minimum, the kernels and the `MLXL3_SSD_CACHE_GIB` override are unchanged.
- **Clearer refusal.** When memory is still too low, the error states the memory needed, the memory macOS reports as available, the reserve, the Mac's RAM and the `MLXL3_SSD_CACHE_GIB` workaround.

## Performance

No performance change is claimed. The beta 2 measurements are not repeated for this build and do not apply as a new gain.

## Installation and compatibility

Install the archive through the Desktop engine updater on the beta channel, or extract its files together. It requires Apple Silicon and macOS 26.2 or later, and its manifest requires Desktop 1.4.0 or later. Desktop 2.0.2 already contains this engine. The archive is ad-hoc signed.

## Verification and limits

- The user asked to skip the remaining tests for this release. Only the checks below ran.
- The memory reserve change was checked by its author on the same Rust source (the `native/` tree is byte-identical):
  - Rust formatting and strict Clippy passed for `chat`, `direct-metal,chat` and `mlx,chat`;
  - the CPU `chat` suite passed, including 448 library tests with 10 ignored;
  - Kani 0.68 verified the two reserve harnesses (183 and 111 checks, 0 failed, all covers reached).
- The release build compiled the MLX engine, signed it and ran it (`mlxl3 list --json`) inside the Desktop 2.0.2 bundle.
- The archive's files were checked against its manifest (see Source and assets).
- **Not verified:**
  - loading Flash Next on the GPU with this build, its logits and the Desktop bridge path (the checkpoint was not on the test Mac);
  - behavior on a real 8 or 16 GiB Mac;
  - the code that reads physical RAM under Kani (a known Kani limitation on macOS system calls; a test compares it with `sysctl`);
  - the beta 2 qualification (CLI logits, bridge hashes, MTP depths, DFlash2, updater) was not repeated on this archive;
  - memory used by long contexts (KV cache) is not part of the reserve, before or after this change.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for relaying the Flash Next loading report and for the release decisions, and to the user who reported the refusal. Thanks also to [Apple MLX](https://github.com/ml-explore/mlx), [Qwen](https://huggingface.co/Qwen), [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [PonyExl3](https://github.com/beamivalice/PonyExl3) for the runtime, models and EXL3 format.

## Source and assets

Engine source commit `22ac280263d7d6cbdd7c7f96bc4ca67b26d714e5` (Desktop 2.0.2 source). Its engine code is beta 2 (`3fe7b089`) plus the memory reserve fix `0fb4abb3`; the other commits change only Desktop, tests and documentation.

- `MLXL3-Engine-v2.0.11-beta.3-arm64.tar.gz`: 68,101,230 bytes, SHA-256 `3908fe934e8482bb6879f07508b85d667a3942e56cef7e0451aa892b87e0df98`.
- `MLXL3-Engine-v2.0.11-beta.3-arm64.tar.gz.sha256`: 107 bytes.
- The archive was extracted into a new directory. Its files matched the manifest (release tag `engine-v2.0.11-beta.3`, minimum Desktop 1.4.0, minimum macOS 26.2.0), and the three Mach-O files passed strict signature and arm64 checks. `runtime-info` reports version 2.0.11, MLX, chat and source `22ac280263d7`.
- Every extracted file is byte-identical to the runtime bundled in Desktop 2.0.2. The MLX libraries and Metal library are byte-identical to beta 2; `mlxl3` is 12,727,904 bytes, SHA-256 `b2dab05bbc438a711ef2649f505fadb35e073acdcdd84e38723a10a36f4bd655`.
- Published with `latest=false`, so the Desktop DMG remains the default GitHub download.
