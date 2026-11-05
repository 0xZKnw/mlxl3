# Flash Next AR: JitEngine, latest Strata and Allosaurus v0.2

Source audit completed on 5 November 2026, following the user's explicit
cancellation of AR31 profiling. The three requested projects were examined at
immutable revisions, including real production callers and their numerical and
memory contracts. No new engine code, model download, optimization trial or
speed benchmark was run. The original dirty checkout and the stopped AR31 work
remain preserved. Brave stays open.

**Result:** these sources provide useful scheduling and kernel references, but
no newly established, directly transferable native AR speedup. The retained
engine is `82fff5a54c91efd2a3f3d34f75a83aa5251785cc`, accepted thirteen defaults
ON and AR20/AR30 experiments OFF. Historical MAIN32 controls of10.4093/10.2530
tok/s remain historical; this research does not establish11–12tok/s.

## Verified projects

| Source | Exact revision and evidence | What matters locally |
|---|---|---|
| [JitEngine audit](jitengine.md) | `skydiscover-ai/jitengine` `2c34508d617800ab7448a69e14a6dd551aec156e`; separate M2/M4Pro Flash Next engines | M4Pro32.9tok/s includes MTP. The24GB M2 engine has genuine AR evidence, but a different affine4-bit checkpoint, GPU, expert cache and comparison session. |
| [Strata audit](strata.md) | Main `9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e`; latest stablev0.2.11 targets our prior pin `c3b3f5035506f3e966f1b4b7b188aeaabae92f1d` | The three intervening commits change funding/docs/Docker packaging, with zero inference-code delta. Its latest release supplies no new Metal optimization relative to our audit. |
| [Allosaurus audit](allosaurus.md) | Model `OsaurusAI/Allosaurus-v0.2-125B-A6B-JANGHT1.8` at `40567ada0ffaa073fd8a297757c7a58102f6b166`; vmlx-swift changes pinned in the report | The reported34.0→56.1tok/s is genuine upstream greedy AR evidence on M5Max/JANGHT, with27.2GiB resident weights and a different memory protocol. It is not our M5/24GiB EXL3 result. |

The reports distinguish version tags, source commits, shipped packages, author
attestations and inferred adaptations. No wrapper, MTP speedup, changed
quantization or previously tested mechanism is counted as a new native boost.

## Candidate decisions

| Mechanism | Decision and necessary boundary |
|---|---|
| JitEngine device slot map, generation checks, selective expert repair and one-layer lookahead | The strongest new implementation reference for AR scheduling. It anticipates work inside the same token, rather than drafting future tokens. A native adaptation must retain our EXL3 kernels and complete canonical route sum, protect leased slots, repair remapped generations, and roll back successor GDN/QSA/PLE state. Existing Sushi low-hit evidence makes a large port conditional. |
| JitEngine cycle-aware expected-next-use eviction | Real past-route EMA64 plus layer distance, not an oracle. Layer distance is constant within our per-layer caches; the distinctive benefit needs a shared pool or changing layer quotas. Historical global-pool/SLRU results remain relevant. An offline opportunity check precedes a runtime revival. |
| JitEngine bounded PLE issue/collect | Its only actual upstream caller is MTP verify2, so there is no deployed solo-AR attestation. A known-token native adaptation is possible, but our repeated MAIN32 warmup already populates the persistent PLE row cache; this is not an immediate warm-decode speed candidate. |
| Small exact post-hash EXL3 codeword table | An adjacent Osaurus JANGH decoder supplies further evidence for the lookup mechanism. The native1021-entry/2042-byte factorization remains the old, unimplemented AR30-K7 reserve, not a new Allosaurus speed result. An exhaustive65536-state GPU bit oracle is required before projection/model/CLI timing. A lookup may cost more than the removed Half FMA. |
| Osaurus compiled HC/shared-expert regions and environment caching | Repairs to a mixed-metadata MLX path, mostly already bypassed by our explicit native kernels and load-time policy snapshots. Its F32 inject product/reduce changes the floating-point program and overlaps excluded HC work; no direct port. |
| Osaurus asynchronous plain-AR token pipeline | Real scheduling reference, but a whole lazy graph cannot simply cross our CPU router, embedding and PLE readbacks. A pending-row API would need ownership, EOS/stop/output-order and abandoned-state contracts. No standalone native gain is established. |
| Osaurus durable hybrid prefix cache | A potential interactive TTFT feature, requiring complete hybrid-state identity and corruption handling. Prefix cache is disabled in MAIN32, so this cannot raise its decode tok/s. |
| Strata cached CPU dispatch descriptions | A useful existing reserve, already AR30 runtimeR5. Rebind and validate owned buffers, dimensions and positions before emission; keep current command/encoder boundaries. No new release or isolated performance evidence justifies relabeling it as a fresh boost. |
| Strata segmented GPU/CPU I/O graph, arena classes and shared encoders | Already examined, already local in part, or incompatible with the demanded lifetime/error contract. No renewed P04/L3/L5 trial or unbounded GPU wait behind SSD I/O. |

### Why PLE is not the immediate MAIN32 choice

This is a source deduction, not a newly measured cache counter. In
`gpu/ple.rs::reset`, only convolution history and prior-token context are reset;
`Ngrams.cache` survives. The standard fresh-process warm+measured deterministic
MAIN32 requests repeat the same87-token prompt and31decode forwards. At most
16×118=1888 row lookups per PLE layer are needed to warm those keys, below its
16384-row capacity. The second identical trajectory therefore has no new PLE
misses for a prefetch to hide. Cold or genuinely new prompts are a different
question, with contention and failure cleanup still to validate.

## Recommended next work

Keep a small exact-codeword-table trial as a bounded possible kernel experiment,
with its historical identity and uncertain benefit explicit. Treat generation-
checked AR scheduling as a separate, larger transaction design. The current
sources do not justify another broad flag sweep or automatic activation.
Before either prototype: record its exact hypothesis/matrix in `opti.md` and
issue135, validate independent numerical/state/error oracles and actual path
counts, then use the requested short32generated/31decode controls. No new trial
was launched under this source-audit entry.

Documentation checks cover immutable commit syntax, local links, and an exact
unchanged engine source tree relative to82fff. No executable change requires a
new build, numeric GPU test or Kani run for this increment; none was presented
as performed. Upstream reported tests were read, not rerun locally. The
cancelled profiling trace does not support a complete millisecond attribution.

Thanks to SkyDiscover/JitEngine, Niko1221/Strata, and OsaurusAI/vmlx-swift for the
public implementations and explicit performance protocols. Their pinned source
links and the earlier local negative trials are retained in the three reports.
