# MLXL3 engine 1.4.3

This release makes DFlash2 setup/tuning available for Qwen3.6-35B-A3B and Qwen3.8-27B, and integrates the cache and optional execution features from [#31](https://github.com/0xZKnw/mlxl3/pull/31), [#32](https://github.com/0xZKnw/mlxl3/pull/32) and [#33](https://github.com/0xZKnw/mlxl3/pull/33).

## Features and fixes

- Pinned DFlash2 drafts, resumable downloads with size/SHA-256 checks, target-family validation, explicit load/release and NDJSON Tune support. The target verifies every proposed token.
- Tune compares baseline and three DFlash2 policies, checks exact token/history parity, excludes invalid results and saves a distinct runtime identity. Allocation policy and optional kernels participate in MTP calibration identity.
- Prefill checkpoints count complete retained buffers, draft/head state, logits and CPU history. Conditional compaction copies wasteful views exactly without changing rollback arithmetic.
- Kernel-cache lookup borrows all five key fields on hits, owning them only on insertion. Ordering, collision and eviction tests use an independent tuple oracle.
- Context-memory profiles feed the Desktop recommendation. Memory saving also applies to DFlash2 Tune and clears checkpoints/unused allocator buffers at request boundaries.

## Options and defaults

| Option | Default / scope |
| --- | --- |
| DFlash2 | User activation required; greedy, one draft family per target; Auto is 2/5 proposals. |
| MLXL3_PROMPT_CACHE_MIB=0..4096 | 256 MiB whole prefill checkpoint budget; 0 disables checkpoints, not all RAM. |
| MLXL3_CACHE_COMPACTION=0 | Conditional compaction is ON by default; 0 disables it. No per-block rollback compaction. |
| MLXL3_ALLOCATOR_CACHE_MIB=0..4096 | Normal MLX reuse by default; explicit cap concerns unused buffers only. |
| MLXL3_MTP_ADAPTIVE=1 | Experimental, OFF. D0..3 with measured wall-cost windows, cold-sample exclusion and hysteresis; Tune uses fixed depths. |
| MLXL3_EMBEDDINGS_PACKED=1 | Experimental, OFF. Lossless F16 bit packing, with dense fallback when packing saves no allocation. |
| MLXL3_EXPERIMENTAL_DENSE_MLP_BATCH=1 / MLXL3_EXPERIMENTAL_GROUPED_MB3=1 | Experimental, OFF. Additional qualified Qwen3.8-27B/M5 dispatch shapes, unchanged shader arithmetic. |
| MLXL3_DFLASH_CONTEXT_COPY=1 | Experimental, OFF. 35B Auto only; recent context proposes tokens that still require target verification. |

Targeted Metal capture requires explicit request/path variables; see [the memory report](adaptive-memory-qwen-20261009.md). Advanced GPU counters remain unmeasured without full Xcode.

## Performance scope

There is no new confirmed global tokens/s boost or established advantage over MTP. The archived key-lookup CPU microbenchmark improved roughly 25%; the production 35B timing comparison failed its drift criterion despite exact outputs. Historical short-context 27B compaction/packing reduced RAM by 566–654 MB under its documented options; this is not a global saving and is not added to other experiments. [Protocols and limits](dflash2-exact-optimisations-2026-10-08.md).

Rejected Q4/GPU and KV8 prototypes are not delivered. Default full-attention KV precision remains F16. Experimental options are explicit rather than automatically selected from inconclusive timing campaigns.

## Verification and limits

[Release validation](release-v1.4.3-validation.md) records Rust formatting/strict Clippy, the shipped mlx/chat build and suites, Python/Swift checks, bounded Kani domains and individual physical tests. Physical 35B checks cover exact state/logit/cache parity, MTP D0↔D1..3 transitions, packed embeddings, compaction and production DFlash2 generation/Tune/cancellation.

New local 27B GPU tests are intentionally excluded. Formal CPU verification does not prove MLX/Metal/FFI, allocation or concurrency; source-verifier gaps, timeouts and inherited direct-Q4 fixture failure are retained in the reports. No universal quality or speed claim follows from these bounded checks.

## Install and compatibility

Download **MLXL3-Engine-v1.4.3-arm64.tar.gz** with its JSON manifest through Desktop's engine updater. Requires Desktop 1.4.0+, Apple Silicon, macOS 26.2+, bridge protocol 1 and MLX 0.32.2. Desktop 1.4.3 is required for the new UI. Restart after an engine update. The archive includes runtime libraries, manifests, hashes and notices; no model weights.

This engine release uses latest=false. Desktop 1.4.3/build25 bundles the same engine and remains the latest updater release. Exact source SHA, artifact sizes and SHA-256 accompany the release and its manifests.

## Thanks

Thanks to [@HENK0O](https://github.com/HENK0O) for [#31](https://github.com/0xZKnw/mlxl3/pull/31) and [#32](https://github.com/0xZKnw/mlxl3/pull/32), and [@0xZKnw](https://github.com/0xZKnw) for integration. Upstream research used [MLX](https://github.com/ml-explore/mlx), [DFlash](https://github.com/z-lab/dflash), [IncoAI](https://huggingface.co/IncoAI), [Splash](https://huggingface.co/Goekdeniz-Guelmez), [dflash-mlx](https://github.com/bstnxbt/dflash-mlx), [mlx-node](https://github.com/mlx-node/mlx-node), [MTPLX](https://github.com/youssofal/MTPLX) and [TensorFold](https://github.com/ashhart/TensorFold). No upstream benchmark is substituted for local measurements.
