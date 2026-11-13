# Engine 2.0.7-beta.1 validation

- Source built: `b05e6579081c5bcae25238f49f32e6b6feef7d6d`. [PR #171](https://github.com/0xZKnw/mlxl3/pull/171) is merged. Its later main synchronization changes Desktop/documentation only; native sources, dependencies and engine build scripts match the built source.
- Runtime: 2.0.7, MLX backend, aarch64, release; embedded source b05e6579081c.
- Archive: `MLXL3-Engine-v2.0.7-beta.1-arm64.tar.gz`, **67873627 bytes**, SHA-256 `f33b40edd6710d22ee3b6b56846b690b65794beb25a58e7c33e260332fc6ffc4`.
- Manifest SHA-256: `fd0d4d5f8817a438df1c746de91905edfab99fd8f6b8cf6fbd127e891ce79001`. All component sizes/hashes, archive member inventory and ad-hoc signatures checked after extraction. Relocated runtime-info passed.
- Format and strict all-target MLX/chat Clippy passed. Focused physical loader regression: default and explicit 1 MiB override, each 1 passed / 0 ignored / 686 filtered, live F32 arrays exact. The test executable was built before the version-only bump from 2.0.6 to 2.0.7.
- Packaged 2.0.7 bridge smoke: eight generated tokens, MTP active, nonempty output, default allocator cache 78727256 bytes, clean shutdown and child reaped. No allocator override was present. This is a functional smoke, not a speed benchmark.
- Packaging tests: 52 passed. Earlier transport/driver checks: 69 passed; experiment oracle self-checks: 3 passed.
- Setting qualification: the existing 2.0.6 runtime produced identical hashes for 1,032 MTP tokens with default/512 MiB settings; completion footprint 18.505929 -> 14.020153 GB. Three sequential target snapshots each contained 248,320 finite F16 logits and 80 nonempty finite state arrays; default/512 output bytes were equal via hashes. These are finite checks, not a proof of all contexts or the complete application.
- Kani 0.68 actual-helper attempt failed before verification with its aarch64 intrinsic ICE (`intrinsics.rs:243`, I32 output assertion). No property was proved. CrossHair on the experiment comparator was inconclusive.
- MTP timing campaign is incomplete: three measured generations per setting, 43.56 -> 45.09 tokens/s medians; a shutdown sampling race was recorded as a controller failure. Reverse controls are absent. No confirmed speed gain. DFlash2 benchmarks and full test suites were explicitly omitted at the owner's request.
- Hosted CI was inspected for the exact source and merged commit; statuses are recorded separately. Pending, cancelled or failed jobs are not a green CI result. Physical GPU checks do not come from hosted CI.

This release changes only the Qwen MoE unused allocator-buffer default. It does not alter weights, quantization, context, sampler or speculative algorithms. The Desktop DMG is unchanged; GitHub publication uses prerelease=true and latest=false.
