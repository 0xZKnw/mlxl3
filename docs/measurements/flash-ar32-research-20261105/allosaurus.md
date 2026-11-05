# Allosaurus v0.2 — primary-source AR audit, 2026-11-05

**Allosaurus v0.2 is the OsaurusAI quantized Qwen3.8-Flash-Next model, not a
separate engine.** Upstream reports genuine **greedy AR34.0→56.1tok/s**, but on
M5 Max/JANGHT with resident experts. It is not evidence of56tok/s for our
M5/EXL3/SSD-expert engine. No new drop-in native boost was established.

Read AR32 preregistration, AR30 matrix/runtime/attention/kernel reports and
native82fff paths first. GitHub connector plus primary web/Hub metadata and
README reads only; hf-cli skill loaded. No weights, clone, prototype, test,
benchmark, compiler, GPU or profiling. Profiling remains canceled; Brave open.

## Identity and pinned sources

| Object | Exact pin / primary source |
|---|---|
| Allosaurus-v0.2-125B-A6B-JANGHT1.8 | HF `40567ada0ffaa073fd8a297757c7a58102f6b166`; [model card](https://huggingface.co/OsaurusAI/Allosaurus-v0.2-125B-A6B-JANGHT1.8/blob/40567ada0ffaa073fd8a297757c7a58102f6b166/README.md) |
| Minimum runtime Osaurus0.25.24 | [Release2026-10-20](https://github.com/osaurus-ai/osaurus/releases/tag/0.25.24), app commit `78232d434856a561a9db473937e76e8485f904c9`; [lockfile](https://github.com/osaurus-ai/osaurus/blob/78232d434856a561a9db473937e76e8485f904c9/Packages/OsaurusCore/Package.resolved) pins engine `afa53392e86c7d6b3a4a3e2cc3e95c08a285951c` |
| Main runtime adaptation | [vmlx-swift PR613](https://github.com/osaurus-ai/vmlx-swift/pull/613), merge `8b1e19e157d9748eaa95d016aa560bad03bdcaca`; [app3125](https://github.com/osaurus-ai/osaurus/pull/3125) |
| AR-fallback lookahead follow-up | [vmlx-swift PR614](https://github.com/osaurus-ai/vmlx-swift/pull/614), merge `934dbeff3db92da2591c89992033859e2ec289b4`; [app3129](https://github.com/osaurus-ai/osaurus/pull/3129) |
| Current source observed | App `ed00d4aa1958056445bf7549289b498e0b2b1f4b`; its [lockfile](https://github.com/osaurus-ai/osaurus/blob/ed00d4aa1958056445bf7549289b498e0b2b1f4b/Packages/OsaurusCore/Package.resolved) and engine main both pin `625d43292d29bb964f413c90b95f700ef7db929d` |

The [Osaurus model listing](https://osaurus.ai/models) dates the announcement
November3, while Hub creation is October19 and minimum runtime release October20.
These are distinct events; there is no invented engine “v0.2” tag, nor a claim
that0.25.24 is today's latest app release. Current-main DFlash work is excluded.

## Speed attestation and compatibility

The card describes text-only JANGHT1.8: approximately27.2GiB resident weights
and54GiB n-gram table on SSD, versus our SSD-streamed EXL3 routed experts.
Quantization changes outputs: teacher-forced bf16 top-1 agreement is75% for
v0.2 versus80.82% for v0.1. This is not an exact oracle for our checkpoint.

PR613 reports **greedy AR34.0→56.1tok/s** (Python reference52.2), M5 Max with a
shared GPU. The card specifies short-prompt greedy; token/context counts are
not given alongside that row. It is an aggregate author A/B, not isolated proof
of each change. The48GB policy was emulated on a128GB host; reviewed evidence
explicitly leaves real48GB qualification open. Our M5/24GiB, EXL3,3GiB cache,
MAIN32/31decode observations cannot be compared causally to those numbers.

The separate27–36% tool-turn claim is **Auto MTP versus Off**, T0.45, two
interleaved repetitions under pressure emulation. It is excluded from AR gains.
The PRs preserve noisy prose, refusals/incomplete answers and clock variation.

## What actually changed, and what already exists natively

HC mixers previously missed the MLX compiled path because their affine metadata
was f16, not bf16. [Qwen4Exp.swift397–489](https://github.com/osaurus-ai/vmlx-swift/blob/934dbeff3db92da2591c89992033859e2ec289b4/Libraries/MLXVLM/Models/Qwen4Exp.swift#L397-L489)
adds a guarded compiled region; [mix651–661](https://github.com/osaurus-ai/vmlx-swift/blob/934dbeff3db92da2591c89992033859e2ec289b4/Libraries/MLXVLM/Models/Qwen4Exp.swift#L651-L661)
is its real caller. Shared experts get the same correction in
[Qwen35.swift483–538](https://github.com/osaurus-ai/vmlx-swift/blob/8b1e19e157d9748eaa95d016aa560bad03bdcaca/Libraries/MLXVLM/Models/Qwen35.swift#L483-L538),
called by `compiledSharedExpert` at2775–2788. These repair MLX generic-launch
chains; native already has explicit Metal HC/GDN/A06/finish paths. The source's
rough9ms mixer penalty is not9ms available locally. MLX compile is not a new
native kernel; resident encoder reuse remains old AR30-C07, not a new idea.

Its4-output injection uses F32 product+sum at [468–473](https://github.com/osaurus-ai/vmlx-swift/blob/934dbeff3db92da2591c89992033859e2ec289b4/Libraries/MLXVLM/Models/Qwen4Exp.swift#L468-L473),
with a160KiB/mixer F32 weight copy. Native HC already partitions K over eight
SIMD groups and has canonical FMA/shuffle/group-add order in
`native/shaders/flash_hc_gemv.metal:1–40`. Generic multiply-reduce changes that
floating-point program and overlaps excluded HC L3/L5: **no transplant/retest**.
Environment caching is already achieved by native Metal::new snapshots; device
mask ranges are not a missing dense host mask in native QSA. C05 PSO capability
caching is an existing reserve. Native fixed-weight residency also already
exists (`gpu/fixed_residency.rs`); full-model MLX wire-budget repairs are not a
new3GiB expert-cache optimization. No OS/wire-limit change is proposed.

## Three transferable families, without additive gain claims

1. **Exact codeword lookup — strengthens historical AR30-K7 only.**
   [JANGHRoutedFastDecode29–118](https://github.com/osaurus-ai/vmlx-swift/blob/8b1e19e157d9748eaa95d016aa560bad03bdcaca/Libraries/MLXLMCommon/JANGHRoutedFastDecode.swift#L29-L118)
   uses immutable exact-bit polynomial levels; actual guarded caller/admission is
   [JANGHRelu2RoutedExpertLayer109–153](https://github.com/osaurus-ai/vmlx-swift/blob/8b1e19e157d9748eaa95d016aa560bad03bdcaca/Libraries/MLXLMCommon/JANGHRelu2RoutedExpertLayer.swift#L109-L153).
   This is adjacent JANGH/ReLU2 code, **not Allosaurus v0.2's JANGT path** and
   not an attributed cause of56.1tok/s. Native cb2's1021-entry post-hash Half
   table is already [AR30-K7](../flash-ar30-20261105/research-kernels.md#follow-up-k7--small-post-hash-codeword-table-explicitly-a-new-inference).
   Preserve hash/byte sum/canonical Half result and every downstream FMA; require
   all65536states/all codebooks against an independent canonical GPU oracle,
   real projections/full states/CLI. This differs from invalid PERF89's8-entry
   table, but cache traffic may still lose. New evidence, not a new measured gain.

2. **One-step plain-AR scheduling — real, but a larger conditional design.**
   [Evaluate.swift3361–3399](https://github.com/osaurus-ai/vmlx-swift/blob/934dbeff3db92da2591c89992033859e2ec289b4/Libraries/MLXLMCommon/Evaluate.swift#L3361-L3399)
   builds/async-submits next lazy output before materializing the previous token.
   PR614 restores that principle inside AR fallback; the MTP iterator itself
   remains excluded. Native router/SSD embedding/PLE reads require concrete IDs,
   so a whole lazy graph cannot simply cross these boundaries. A pending-row
   design would need exact producer ownership, output/RNG/EOS order, rollback
   and trimming of an un-emitted cache row. No standalone native gain is attested;
   this is not permission to resume MTP/DFlash or canceled P04.

3. **Durable hybrid prefixes — missing capability for TTFT, not decode TPS.**
   [PrefixCacheInvariants](https://github.com/osaurus-ai/vmlx-swift/blob/8b1e19e157d9748eaa95d016aa560bad03bdcaca/Libraries/MLXLMCommon/Cache/PrefixCacheInvariants.swift)
   and [SSM companion fingerprint repair1054–1081](https://github.com/osaurus-ai/vmlx-swift/blob/8b1e19e157d9748eaa95d016aa560bad03bdcaca/Libraries/MLXLMCommon/Cache/SSMCompanionDiskStore.swift#L1054-L1081)
   have real app3125 restore evidence. The Foundation URL-cache bug is not
   asserted in Rust. Native `native_prompt_cache.rs` retains one RAM snapshot;
   SSD persistence would require exact GDN/QSA/PLE/hidden/logit/token/compute-key
   state, corruption checks, immutable bounded async writes and joined shutdown.
   This helps repeated-turn prefill; MAIN32 prefix-cache0 cannot benefit.

**Decision:** no newly justified zero-arithmetic-change native boost from the
headline. Keep the existing small exact-table reserve, larger conditional AR
scheduling, and a separately labeled TTFT extension. No default promotion,
11–12tok/s promise or renewed profiling; root owns any later issue/preregistration
and independent correctness gates before prototypes.
