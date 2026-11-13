# MLXL3 Engine 2.0.8-beta.1

## Changes

Qwen3.6-35B-A3B now automatically uses a compact, draft-only Q4 vocabulary projection when MTP is enabled. The engine builds it natively on first use and reuses a verified cache. No Python conversion or additional model download is required. MTP1 remains the default depth; fixed depths 2 and 3 remain available.

The target keeps its original EXL3 weights and complete 248,320-token vocabulary. Every proposed token is verified by the original target. Explicitly supplied draft projections take precedence. Other models and unsupported head layouts retain their previous behavior.

## Fixes and cache behavior

Cache identity includes the source head contents, vocabulary and conversion version. Writers are serialized and only complete results are published. Corrupt caches are rebuilt; unavailable caches fall back to standard MTP with a diagnostic. Cancellation during preparation publishes no partial projection and can be retried.

The engine package includes TensorFold's vocabulary attribution and Apache-2.0 terms. Unconfirmed upstream kernel and adaptive-budget prototypes are excluded from this release.

## Performance

The previous qualification on Apple M5 / 24 GiB, MLX 0.32.2 and Qwen3.6-35B-A3B EXL3 2.49 bpw compared stock and compact MTP with identical greedy 128-token workloads, context 4096 and prompt reuse disabled. The reverse-order confirmation measured MTP1 **64.76→72.57 tok/s** for French and **64.47→69.98 tok/s** for code. The first order also favored compact projection, with wider dispersion. These are workload-specific decode measurements, not universal gains. MTP1 remained faster than fixed depths 2/3 on these prompts.

The final isolated release receives its own correctness and packaged-runtime checks. Final artifact identities and measurements are recorded in the companion validation report. Startup preparation is separate from decode; the previous local head-loading check measured 3.69 s on first use and 1.08 s with cache, outside target-model loading.

## Installation and compatibility

Independent Apple Silicon engine prerelease for macOS 26.2+, MLX backend. Install the complete archive through the Desktop engine updater or extract all its components together. MTP must still be enabled to use this optimization.

The Desktop DMG is unchanged. Publication uses prerelease=true and latest=false to preserve the Desktop update channel. Signatures are ad hoc, not notarized. Physical performance was measured on M5; other supported Apple GPUs have not received this performance campaign.

## Verification and known limits

Validation covers native generated tensor equality, target token/cache/state parity, fixed MTP depths 1/2/3, explicit overrides, cache reuse/corruption, cancellation/retry and cache-unavailable fallback. Source-level Kani verification concerns checked chunk/offset arithmetic only. It does not prove MLX, Metal, filesystem transactions, concurrent processes or model quality for arbitrary inputs. The available CBMC cannot parse the actual modern libc++/MLX headers. Hosted CI is reported by exact commit and separately from physical GPU checks.

See the companion validation report for commands, test counts, source SHA, archive size/checksum, component hashes, relocation/signature checks and remaining verification limits.

## Acknowledgements

Thanks to [0xZKnw](https://github.com/0xZKnw) for the measurements and optimization requests; [TensorFold](https://github.com/ashhart/TensorFold) contributors for the Apache-2.0 draft vocabulary and draft-only pruning design; [MTPLX](https://github.com/youssofal/mtplx) contributors for the retained MTP architecture; [Apple MLX](https://github.com/ml-explore/mlx) for native affine quantization and inference kernels; [Qwen](https://huggingface.co/Qwen) for the model; and [ExLlamaV3](https://github.com/turboderp-org/exllamav3) and [PonyExl3](https://github.com/beamivalice/PonyExl3) for EXL3 foundations.
