# AR43 —64-slot T05 + early SSD enqueue; qualified P02 default

Base04d42828 (AR42/PR157), above P02/PR156. This implements the remaining P01/P05 mechanisms prioritized from #147–#152/#150. P03/P08 remain OFF after their neutral screen. No MTP/DFlash.

P02 now defaults ON for exact Apple M5; existing structural guards, hardware/Tier2 admission, export fallback and explicit0/1 override remain. This default is asserted without an override in the physical model gate and used without an override by every release CLI arm.

P01 extends the immutable bank/snapshot admission to exactly53 or64 slots. Metal always compiles these kernels from embedded source: the same selected capacity is inserted as a shader macro and passed to the host pipelines. Binding validation requires that exact capacity; no external metallib/encoded-length guess. Retirement uses a checked slot<capacity≤64 before shifting. No72/80/107 support. The53 bank remains default; opt-in64 uses MLXL3_NATIVE_DEVICE_HOT_SLOTS=64 together with a cache budget yielding64 slots/layer (3.625GiB for this checkpoint). Changing the table alone never expands the user's cache budget.11extra records×48layers=658950336B (~0.614GiB) of maximum expert record capacity; this is not process RAM. A mismatched cache/bank uses the existing fallback.

P05 validates all source descriptors, count/owners and extents before any Shared job starts, then optionally passes a lazy fallible allocation iterator to the existing bounded read core. First job can enqueue before later destinations allocate. Partial allocation failure drains already queued jobs through the existing deadline; timed-out workers keep their file/destination ownership and block conflicting new foreground reads. Split/scheduling/record reuse semantics preserved. MLXL3_NATIVE_EARLY_READS initially remains opt-in for the comparison.

## Focused verification

5 device_hot_policy +21 SSD CPU tests passed,5 physical tests skipped in that CPU selection. The new partial-failure/retry test uses a disposable real file, holds the first worker, observes its active ticket before creating the second destination, injects an error and requires no remaining ticket before retry. A temporary collect-before-enqueue mutation makes it fail; restored source passes. Early Shared integration checks real bytes, two reused/two fresh buffers, EOF/retry, invalid owner extent fallback and invalid plan rejection before allocation.

Three explicitly selected DEBUG physical gates pass1test/0ignored/452filtered each; all children join. The64-bank gate compares all64 bindings/163840 finite Half oracle words and281600 total contribution words across11hit/miss cases,16actual H submissions, wrong-pointer mutation, invalid/duplicate IDs, stale generations, retired slot63 and rejection of64, retained/unsubmitted consumers and publication rejection. The model gate checks two tokens: full248320 logits,40960B hidden,84continuation tensors/case and57606144 finite state words,36actual P02 uses/token with64slots, error/rollback/retry/export restoration. Independent initial logit digest/first archived state tensor; other state comparisons use the canonical baseline. No new long-context/model-family coverage.

Kani0.68 actual CPU implementation: capacity/admission/contribution/retirement/reserve85checks0failed,6covers; Shared plan1check0failed,2covers. Symbolic full usize domains, no unwinding failures; reserve is ten retired records plus≤2MiB auxiliary. These are CPU contracts, not Metal/concurrency/allocator proofs. Standalone CBMC absent as recorded in AR42. Debug gates compile actual Metal source; release CLI/golden verifies the shipped numerical path. No redundant release-libtest rebuild/general GPU suite per user's request.

Strict Clippy --features chat --all-targets and --no-default-features --lib, formatting, release CLI, metadata/inventory checks. One metadata run caught a stale count from the new invalid-plan test; reference updated after review and failure preserved. Workflows rust.yml/check.yml cover PR and main, manifest includes the new proof. CI of exact published head is inspected separately; existing parent failures are not claimed resolved.

## Protocol and artifacts

M5/24GiB, Qwen3.8-Flash-Next EXL3avg2.05/H4 checkpoint65c89531, MAIN87,128generated/127decode,ctx4096,T0/topk1/penalty1,prefix0,4readers/depth0,dynamic0. Existing13AR defaults+AR2+T05; P02 qualified default, P03/P08OFF. One warmup+one measure per arm:53slots/3GiB/earlyOFF, then64slots/3.625GiB/earlyON, then53control. The candidate changes two mechanisms and extra cache capacity; no isolated P05 speed claim or sum with AR41/AR42. No simultaneous compiler/prover/model, OS changes, cooling pause, swap abort or app closures; temperature unmeasured.

[Quality freeze](quality-plan.json), [three physical gates](quality.json), [quality supervisor](quality.py.txt), [speed freeze](speed-plan.json), [speed supervisor](speed.py.txt). Source files, binary/model metadata, fixture header and selected record payload fingerprints recorded. No whole42GB weight hash. No installed runtime/app replaced, merge or release.

## Audit requested before publication

No implementation fault or silent T05 fallback found. The first model gate lacked explicit positive read-path counters; test-only assertions now require48 actual T05 submissions per token,239 early batches and positive T05 misses, while preserving full states. Its receipt is [audit/quality.json](audit/quality.json). No release code changed in this audit.

| Check | Observed result |
|---|---|
| T05 at53 and64 |6096/6096 expected H submissions in each127-step decode;4572/4572 expected P02 uses, in warmup and measured phases |
| Real measured decode misses |18401 at53 →16117 at64 (-12.412%), same full128 tokens/text/golden hash |
| First host read call |P05 ON36500ns, before last allocation83042ns; OFF395625ns, after last allocation305709ns |
| Metal allocation |6614138880 →7280246784B (+635.25MiB) |
| Process footprint |7275780024 /7944690024 /7272142728B; candidate +639.66MiB vs mean controls |

The I/O microcheck reads all10 full disposable records and verifies every byte; timestamps are host entry immediately before read_exact_at and after each allocation. One OFF-then-ON observation, no gates/sleep, cache/allocator order effects present. It confirms overlap opportunity and actual observed ordering, not an isolated real-model TPS gain or physical NAND start. [Micro](audit4/quality.json).

The count test uses the production native prefill/reset/forward plus the exact CLI log_probs→greedy_log_prob_id path, MAIN87, warm128+measured128,2 processes at53/64. Positive H/P02 counters exclude silent fallback throughout decode. Diagnostic DEBUG execution has no event/kernel profiler, and its throughput is not compared to release. Independent full128 golden hash f6b5f974c6db2e7a and exact token/text equality across all4 phases. [Final counters](audit4/counts-r3/quality.json). Initial two attempts stopped before decode because their test helper required DEVICE-greedy, which the frozen CLI does not enable; failures remain parityfalse/joinedtrue in audit4 and counts-r2. Corrected test did not enable any extra optimization.

Release compressed logical memory602 /983 /242MiB, lifetime process decompressions20593 /45672 /249. Process ledger swap-ins0 in all three samples; this says nothing about global swap. More compression is compatible with pressure offsetting saved reads, but does not prove its causal contribution. No thermal measurement. The extra allocation matches528 additional records rounded to16KiB pages (inference from the byte delta); payload capacity alone is658950336B.

## Release screen decision

10.13229 control /10.0857164+early /10.01931control tok/s. Pooled controls 10.07548; candidate +0.102%, drift -1.115%. Six full128 outputs match golden, all children joined. No established new speed gain:64-bank and P05 remain experimental, no extra default cache RAM. The pair does not reject either option individually. P02 alone was qualified in AR41 and now defaults ON for exact M5; P03/P08 remain OFF. No second speed campaign after test-only audit changes.

Kani85+1 checks/8 covers,26 CPU tests (5physical skipped in that selection), initial3physical gates plus1 model-counter audit and3 final observation processes, each selected test1pass/0ignored. Formatting/strict direct-chat Clippy/release CLI/11metadata+2subtests. CPU library Clippy passes; broad no-default-features/all-targets additionally exposed a pre-existing chat-only vision_input reference in openai_transport/main, retained in logs, outside this change. No full suites, Metal/concurrency proof, new model family coverage or app installation.

Published for review in [PR158](https://github.com/0xZKnw/mlxl3/pull/158), stacked on PR157/156. Implementation/report commit1c7a1c3f; this publication note changes documentation only. Engine/app installation unchanged. No trials remain in progress.
