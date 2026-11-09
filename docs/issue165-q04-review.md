# Q04 review — 2026-11-09

Historical review of the first replay, before the fixes and online prototypes below. Its findings and original verdict are retained; see the [final experiment report](measurements/issue165-q04-20261109/README.md) for the later implementation, quality checks and throughput measurements.

Q04 currently captures exact production routes and replays a conservative fixed-bank policy. It does not substitute experts in the production engine. This review compares that diagnostic with the source idea, rather than treating it as a finished implementation.

## Source comparison

- Sushi issue revision: [`904b0f6e`, `substituteMisses`](https://github.com/beamivalice/sushi/blob/904b0f6e4819b5dea627806f90d75d0b5c2ae7ba/src/expert_stream.zig#L529-L578).
- Latest examined revision: [`0bb32df5`, same function](https://github.com/beamivalice/sushi/blob/0bb32df5d1359f17a46eb2b904bbe1d9a8fd99ae/src/expert_stream.zig#L523-L572). The function bodies are identical in these two revisions.
- The [real grouped-cache caller](https://github.com/beamivalice/sushi/blob/0bb32df5d1359f17a46eb2b904bbe1d9a8fd99ae/src/expert_stream.zig#L2906-L2938) builds readiness from occupied, ready cache slots. Substitutions retain the original mixture weight.
- The [score conversion](https://github.com/beamivalice/sushi/blob/904b0f6e4819b5dea627806f90d75d0b5c2ae7ba/src/moe_stream_compute.zig#L50-L78) uses F32 logits for softmax and log-sigmoid for sigmoid routers. The threshold is a log-score gap `ln(1−tau)`.

Sushi processes only actual misses, in descending original score order. It can replace several misses with the highest-scoring available experts outside the original selection. An expert deferred three times must be loaded on its next miss. Cached originals do not reset this counter. Its GPU twin does not make this a different selection policy.

Our banks of 16/32, protection of the three highest scores, one replacement maximum, original weight cap 0.10 and tolerance at most 0.05 are deliberate conservative adaptations. Their smaller opportunity is not evidence of a faulty Sushi algorithm. Selecting protected experts by scores rather than array positions is correct for our native top-k ordering.

## Findings before drawing conclusions

### 1. The replay can spend its sole replacement on an existing hit

**Code:** `scripts/replay_native_route_swap.py:17–34`, called at line 115.

**Case:** with original IDs 0–9, uniform probabilities, bank `{10}` and ready experts `{3,10}`, `choose_swap` returns `(rank=3, expert=10)`. Expert 3 was already ready. With bank 32 and tolerance 0.05 in the real trace, 1,374 of 2,830 substitutions were baseline hits; another 476 substitutions chose a candidate absent from baseline ready slots.

**Consequence:** these substitutions do not directly avoid an observed miss, and the one-replacement limit can exclude a genuine miss in that row. This is a mismatch with Sushi's miss-only selection. In the preregistered fixed-bank experiment it is explicitly counted as wasted work, not silently included as a read saving.

**Smallest correction for the evaluation:** distinguish a Sushi-style observed-ready diagnostic from the fixed-bank experiment. Do not feed instantaneous physical readiness directly into a future MTP target: that target needs deterministic logical residency and accepted-prefix state updates. If a bank is pinned, replay its capacity cost and the remaining cache, rather than assuming the baseline cache is unchanged.

### 2. Starvation recovery has a different contract

**Code:** `scripts/replay_native_route_swap.py:125–126`.

**Case:** the actual `analyse` routine was exercised with a synthetic complete 48-layer trace. Expert 16 is first substituted, then appears as an unmodified cache hit, then misses again. The counter before these rows is `[0,1,0]`; Sushi preserves `[0,1,1]` because hits are not processed by its miss loop.

**Consequence:** the current counter does not impose Sushi's limit on accumulated deferred misses. A hit really executes the original expert, so this difference alone does not demonstrate a quality regression; it is a different recovery policy that the present tests do not distinguish.

**Smallest correction:** either retain the upstream miss-only counter transitions, or explicitly specify and test the alternative recovery contract. In production MTP, preserve and roll back this counter with the accepted prefix.

### 3. Frozen baseline readiness cannot predict net cache traffic

**Code:** `scripts/replay_native_route_swap.py:112–129`.

**Case:** if the baseline loads original expert X and the simulated policy replaces it, a later baseline snapshot can list X as ready even though the simulated policy never loaded it. Pinning a bank also removes slots available to other experts.

**Consequence:** the reported 1.35% / 2.72% at bank 32 and tolerances 0.025 / 0.05 are gross fractions of baseline logical cold reads with an eligible substitution, not net read savings or throughput gains. The existing report already states this limitation. They cannot reject Q04 generally or validate the final policy.

**Smallest correction:** evaluate a defined cache trajectory, with residency updates and capacity accounting, before deciding whether a production prototype is worthwhile. Do not rerun the unchanged GPU capture merely to repair this offline diagnostic.

## Verification and limits

- Reused existing capture: 22,752 routes, 384 full 248,320-value Half logit rows exactly equal to the frozen exact reference. This validates the diagnostic capture on that corpus, not approximate routing quality.
- Review command: `.venv/bin/python -m pytest -q tests/test_native_route_swap_replay.py` using the existing project Python 3.12. Result: **14 passed**, none skipped. Their passing result does not cover the two source-semantic differences above.
- Additional synthetic checks execute the real `choose_swap` and `analyse` functions, with a temporary 48-layer trace; no GPU job. Evidence: `build/issue165-20261108/q04-review-20261109.json`.
- Reviewed replay SHA-256: `70357899a9a21bc25e719dae4beae86cb46c67faee88c7ac3f1b0b340f7f2755`.
- No runtime code changed by this review. No speed measurement, approximate-target quality measurement, full cache simulation, upstream test execution, or formal proof of the Python replay. Comparing probability ratios is mathematically related to Sushi's logit gap, but bit equality at F32 threshold boundaries has not been verified.

**Verdict:** keep the engine exact. Repair/clarify the diagnostic and counter semantics before judging Q04's profitability. The original idea has neither been rejected nor validated by this replay.
