# suffix lookup + moce-aware multi-token verification on streamed moe

## recommendation

start with **opt-in, greedy-only suffix lookup, minimum match 4, at most 3 drafts, and windows that never cross a four-cell qsa block boundary**. do not enable unrestricted `T <= 8` in serving yet: the existing qsa raw-key tail has a reproducible rejected-draft rollback defect. stronger moce on speculative rows is a separate **approximate-model experiment**, not lossless speculative decoding.

artifacts:

- `/tmp/sol_spec_decode/01_suffix_serve_boundary.patch`: serving support and shared safe-window/prefix helpers; no metal changes.
- `/tmp/sol_spec_decode/02_experimental_draft_moce.patch`: separately opt-in per-draft routing knobs, matching fused-router support, and optional root-only prefetch.
- `/tmp/sol_spec_decode/suffix_moce.patch`: combined patch; paths relative to `vendor/strata`.
- `/tmp/sol_spec_decode/tree/`: exact proposed source files.
- `/tmp/sol_spec_decode/test_suffix_window.cpp`, `tests.log`, `compile-commands.json`, `*-compile.log`: cpu-only validation evidence.
- `/tmp/sol_spec_decode/original/`, `source-sha256.json`: exact audited worktree snapshots.

**validation:** all three changed c++ translation units compile successfully to objects with the existing build's flags; `git apply --check` succeeds. the actual suffix drafter and extracted actual `slop4` functions pass cpu tests. the qsa-tail cpu model reproduces 736 failures among unrestricted cases and passes all 320 boundary-capped cases. **metal compilation, gpu execution, end-to-end serving, model quality, gdn numerical parity and performance are unverified.** `xcrun --find metal` fails: the metal compiler is not installed. no gpu/ssd benchmark was run; no subagents were spawned. stage 2 is not release-ready until metal compilation and model tests are done.

## scope and source pin

research date: 2026-10-17. strata revision: `296b59d2f73c8f04a805df29c2fa97655adace78`; the checkout contains local work, so hashes/snapshots above—not the revision alone—identify the audited bytes. no tracked file under `vendor/strata` was edited. original line references below refer to those snapshots.

`src/core/spec.hpp` does not exist in this checkout. the actual implementation is `include/strata/spec/suffix_drafter.hpp` and `src/spec/suffix_drafter.cpp`. `metal/src/strata_window_cli.cpp` contains no mtp entry point; this patch enables suffix/prompt lookup, not an mtp model. the engine has a separate mtp layer implementation beginning at `window.cpp:2279`.

## 2025–2026 arxiv sweep

primary arxiv abstract pages were downloaded and retained in `/tmp/sol_spec_decode/<id>.html`, in addition to `web_search`. the closest paper, acceptmoe, was also inspected in full html (`AcceptMoE-full.html`). speedups below are authors' claims, not reproduced measurements and emphatically not m5 forecasts.

| paper | relevant result / implication |
| --- | --- |
| [specmoeoff, 2508.21706](https://arxiv.org/abs/2508.21706), 2025 | increases useful work per transferred expert; roofline-based cpu/gpu scheduling and speculation tuning; up to 2.5x decode throughput. host-memory offload differs from nvme and apple unified memory. |
| [sp-moe, 2510.10302](https://arxiv.org/abs/2510.10302), 2025 | identifies expert-union expansion during multi-token verification; predictive prefetch with bounded depth and pipelining. claimed 1.07–3.5x tpot speedup. supports limiting prefetch rather than indiscriminately multiplying lookahead. |
| [moe-speq, 2511.14102](https://arxiv.org/abs/2511.14102), 2025 | uses an on-device draft model to predict expert demand; amortization roofline governor. abstract reports at most 2.34x on phi-moe. prediction is not proof of future acceptance. |
| [accelerate speculative decoding with sparse computation in verification, 2512.21911](https://arxiv.org/abs/2512.21911), 2025 | sparsifies attention/ffn/moe verification and reuses retrieval across drafts/layers; explicitly reports efficiency–accuracy tradeoffs. not evidence that modified verification is distribution-preserving. |
| [moe-spec, 2602.16052](https://arxiv.org/abs/2602.16052), 2026 | caps verification expert eligibility; 10–30% throughput over eagle-3 at comparable quality, with tighter budgets trading accuracy for latency. closest fixed-budget alternative to moce. |
| [specmoe, 2604.10152](https://arxiv.org/abs/2604.10152), 2026 | self-assisted speculative decoding for cpu-offloaded moe; up to 4.30x throughput. distinct from specmoeoff. does not establish that a suffix drafter on iq2_xs will share enough experts. |
| [evict, 2605.00342](https://arxiv.org/abs/2605.00342), 2026 | trims uneconomic draft-tree prefixes before target verification; described as lossless. claims average 1.21x over eagle-3. a better conceptual fit when preserving the target is mandatory. |
| [ecospec, 2607.12696](https://arxiv.org/abs/2607.12696), 2026 | draft selection includes predicted marginal expert cost while leaving target verification unchanged; up to 1.62x. maps naturally to acceptance/cold-expert-aware draft-length selection. |
| [acceptmoe, 2608.02989](https://arxiv.org/abs/2608.02989), 2026 | **direct match to this proposal**: commitment-weighted router demand, root top-k anchors, entropy/effective-rank-sized eligible expert set, residency-aware pruning. explicitly acknowledges changed model distribution; mean accuracy -0.27 percentage points, claimed 2.06x throughput with physical expert offload and 73.6–77.1% less transfer. not proof of safety for this model/persona. |
| [suffixdecoding, 2411.04975v3](https://arxiv.org/abs/2411.04975), revised 2025 / neurips 2025 | original id is 2024, not a new 2025 paper. adaptive model-free suffix-tree lookup for repetitive agent/code workloads; up to 5.3x reported. strata's four-way trigram index plus backward suffix extension is simpler; do not attribute those numbers to this implementation. |

ranking: suffix lookup first because there are no draft weights or extra gpu forward passes; evict/ecospec-style length selection next if exact target semantics matter; acceptmoe-style verification eligibility only if a measured quality tradeoff is acceptable. mtp adds draft compute, its own qsa caches, hidden-row alignment and model-loading costs, so it is not the zero-cost next step.

## code audit: run / finish / commit

### token contract and acceptance

`window.cpp:1478–1628` takes inputs `[root, draft1, ...]`, runs all layers, and returns **next-token predictions for each input row**, not predictions for the input row itself. if output row `i` equals input row `i+1`, that next input may be committed. first mismatch at `i` means commit `i+1` inputs and emit outputs `0..i`; output `i` becomes the unprocessed root for the next window. the last output is the bonus token when every draft matches.

`strata_window_cli.cpp:877–919` already implements that greedy prefix test. `serve_loop:590–617` always runs `T=1`. the patch passes `--suffix-draft` into serving and preserves the old sampled/default branch. only `temperature == 0` uses batched greedy verification. it commits only the verified input prefix, adds precisely those inputs to `cached_ids`, advances `pos` by the committed count, emits the corresponding output prefix, and adds only emitted outputs to suffix history. eos terminates the prefix before later draft inputs are committed. remaining output length and the engine's strict context-margin check constrain the proposal length. lookup is rebuilt from the final request ids after scr/pad/prefix processing, not from an old request's divergent history.

sampling is deliberately not enabled: accepting equality against greedy argmax at nonzero temperature is wrong. distribution-correct speculative sampling needs target/draft probabilities (or deterministic proposals with the corresponding acceptance rule), residual resampling, and sequential repeat/presence-penalty histories. none is provided by this patch.

### gdn recurrence and lazy fold

`window.cpp:1138–1159`: `T=1` commits in `gdn_step`; ordinary `T>1` does not mutate the committed recurrence, retaining projected per-layer `qkvL_/alL_/beL_/zL_` for replay. with `STRATA_METAL_PARTIAL=1`, a head window commits root row 0 in the step; headless prompt windows commit all rows.

`finish:1922` flips `hp_` for `T=1` or partial commit. `commit:2233–2260` validates `1 <= keep <= last_T_`; normal mode folds `keep` rows, partial head mode folds only `keep-1`, partial prompt mode refuses partial rejection. `fold:1942–1963` uses offset 0 normally or 1 in partial mode, recurrence commit=1 and the old/new convolution-history parity. lazy mode stores `pend_fold_=nf` and advances parity; the next run prepends the fold using `1-hp_` before new layer projections overwrite its saved inputs. graph-cache keys include pending-fold count, parity, `T`, head, selection mode and mrope. eager mode folds immediately. checkpoint/session save flushes pending fold before copying recurrence/history.

this supports rejected-prefix replay by construction, but it is **not a numerical parity test**. never call commit twice for one run or run again before committing; the api does not provide a transaction/idempotency guard. the new serve branch calls commit exactly once.

### ple history

`run:1493–1502` derives all lookup rows from a local copy of `ss.ple_prev`, advancing locally through window inputs; no speculative update leaks into `ss`. `ple_conv_add` (`metal/src/engine/msl/ple.metal:75–90`) reads committed `pleHist_` for negative tap offsets and current window `normW_` for prior nonnegative offsets. `commit:2262–2274` shifts only the first `keep` normalized rows into `pleHist_`, and advances `ss.ple_prev` only over committed input tokens. rejected rows remain scratch. the unused cpu `ple_hist_`/`last_norm_` helper is not the active gpu history path.

### slop4

`finish:1874–1920` steps actual input tokens from `slop_at(pos0-1)`, writes a state per absolute position, and applies penalties to each corresponding output-logit row. the greedy outputs used by the suffix verifier therefore already include anti-slop v4. `commit` does not truncate `slop_`, and that alone is **not** a rejected-draft bug: the next run starts from state at `pos0+keep-1` and overwrites later rejected states. the cpu test uses the actual extracted `slop4::step` code and confirms this restart invariant for every `T=1..8`, every kept prefix, newline/indent/eos and corrected continuations.

separate pre-existing caveat: checkpoint restore does not restore `slop_`. after different branches overwrite the same prefix positions, restoring an old conversation checkpoint may need a token rescan. that is outside this suffix patch; the above invariant concerns immediate rejected-draft continuation, not arbitrary checkpoint history.

### qsa: real rollback defect and conservative containment

positional quantized k/v cells beyond the kept prefix are harmless if rewritten before causal attention reads them. the indexer's **four-slot raw-key tail is not positional**.

`qsa.metal:101–120` writes every speculative input raw key to `tail[(cell&3)*128]`; when `(cell&3)==3`, it pools all four slots. `commit` never restores that tail. counterexample:

1. committed cells end at 4; start `pos0=5`, `T=4` (cells 5,6,7,8).
2. speculative cell 8 overwrites tail slot 0, destroying the raw key for committed cell 4.
3. reject the first draft; keep only input cell 5.
4. rerun cells 6 and 7. cell 7 pools `[raw8_rejected,raw5,raw6,raw7]`, not `[raw4,raw5,raw6,raw7]`.

this may only affect model output once sparse selection matters (>2051 cells), but it corrupts indexer state earlier. the bug exists even at `T=4`, not only at 8.

containment is `T <= 4-(pos0&3)`. no speculative write can wrap onto an earlier committed slot. rejected slots within the current block are rewritten before that block completes; a prematurely written pool for the incomplete block is ignored because `qsa_score:163–176` uses `dead` for the current tail block, then recomputes the completed pool when corrected cell 3 runs. all 320 capped cpu cases pass; 736 unrestricted cases fail in the same exhaustive model. this is a structural scalar-key model of the shader, not gpu execution.

for unrestricted `T<=8`, the engine needs per-layer tail rollback: snapshot each qsa tail before run and retain per-row raw indexer keys, then restore/replay only `keep` rows before another run/checkpoint/session save. preserve/recompute the relevant pooled tail block as needed. simply copying final tail or shrinking the logical kv length does not recover overwritten committed keys. mtp qsa tails need the same treatment. the patch does not pretend to fix unrestricted engine callers; it confines both suffix callers to the safe region.

## moe / ssd behavior and experimental routing patch

`moe_route` (`moe.metal:158–249`) routes each token independently in simdgroup `sg=t`, then `plan_body` groups the **union** of experts across rows. current protected rank and moce bonus are the same for every row. protected top-j are selected first; resident scores are multiplied by `exp(bonus)` afterwards; selected weights undo that multiplier and renormalize actual probabilities. `protect=3, bonus=0.85` is already an approximate routing policy, not the natural target's exact top-10.

`ExpertStream::at_layer:725–730` unions all `T*topk` ids, pins all wanted resident/in-flight slots first, submits every missing expert urgently, and waits for all required reads (`810–818`). it cannot know a draft will later be rejected. `ahead:654–714` similarly considers every row, merges expert demand and sorts by strongest priority; both normal and split-turn `after_layer` call it. a rejected draft can therefore incur synchronous loads **and** evict useful experts/read ahead unnecessarily. `window_commit` does not rewind the expert cache or io counters.

stage 2 adds independent function constants 28/29, avoiding existing 24/25/26/27. it updates **both** `moe.metal` and duplicated fused `router.metal` behavior; changing only `moe_route` silently misses `STRATA_METAL_ROUTE=1`. pipeline-cache tags include the new constants. defaults leave current row policy unchanged. optional settings:

```text
STRATA_METAL_MOCE_PROTECT=3
STRATA_METAL_MOCE_BONUS=0.85
STRATA_METAL_SPEC_MOCE_PROTECT=1
STRATA_METAL_SPEC_MOCE_BONUS=1.2
STRATA_METAL_SPEC_ROOT_PREFETCH=1
```

these are **candidate experiment settings, not measured recommendations**. draft bonus is finite and accepted only within [0,20]. `sg=0` keeps the configured base routing; `sg>0` uses draft knobs when base moce is active (`a.pad!=0`). draft overrides disable the optional intermediate rank-1 bonus on draft rows to avoid conflicting protection semantics. no knobs force the root to natural top-10. `SPEC_ROOT_PREFETCH` defaults off; when on, only row 0 contributes to `ahead` for small windows, while synchronous loads still honor every selected row. it may increase late waits if most drafts are accepted.

**a finite larger bonus cannot bound cold misses.** arbitrarily stronger logits can still select cold experts. a hard bound requires an eligible expert set or budget, e.g. root anchors plus ready-resident experts, and a fallback if fewer than top-k are eligible. an acceptmoe-like implementation would first synchronize root ids, retain all root anchors, then restrict draft selection before plan generation. a threadgroup barrier is necessary; concurrent simdgroups must not read incomplete root ids. the present patch deliberately offers a smaller, auditable finite-bias experiment, not that hard-budget algorithm.

**root preservation is not exact decoding quality.** accepted row 1 may match row 0's predicted token while its own hidden state and next-token logits were computed with different experts; committing it preserves the approximate trajectory. even with identical moce knobs on all rows, speculative loads/evictions change future-layer residency and can alter root routing versus serial `T=1`. true target equivalence requires natural/cache-independent target routing or an explicitly defined equivalent cache snapshot policy, and actual numerical parity tests. acceptance rate alone cannot certify quality when the verifier itself changes.

## systems/math decision rule

let `D=T-1`, conditional draft acceptance probabilities be `a_i`, and `q_i=prod_{j=1..i} a_j`, with `q_0=1`. expected committed inputs/emitted outputs per window are

`E[keep] = sum_{i=0..D} q_i`.

with constant acceptance 0.7 and `T=4`, `E[keep]=2.533`; `T=3` gives 2.19. the supplied serial baseline 44.4 ms/token requires a four-row window to finish below about **112.5 ms** just to break even at 0.7 acceptance, including cpu drafting, io, replay/fold and amortized compilation. this is a threshold, not a prediction. a minimum match of 4 does **not** mathematically imply 70% acceptance: identical four-token suffixes can have unrelated continuations.

at layer l let selected expert sets be `E_l,t`, ready residents `R_l` and `U_l=union_t E_l,t`. synchronous cold demand scales with `|U_l \ R_l|`, while expert-weight gpu traffic scales with union/reuse, not just accepted tokens. uniform independent top-10 out of 512 would give `E|U_l| = 512*(1-(1-10/512)^T)`, roughly 38.8 experts for `T=4` versus 10 for `T=1`: almost no useful sharing. real same-prefix tokens may correlate much more, but that must be measured. dense weights can be shared by a multi-row matvec, so not all 38 ms of the supplied gpu time scales with expert union. with only 6 ms exposed io, eliminating that wait alone caps speedup at `44.4/38 ~= 1.17x`; more gain needs actual gpu/dense-weight amortization.

a useful length controller chooses the largest candidate prefix whose marginal expected outputs exceed marginal cost at the current throughput:

`delta_E_keep / delta_C_window > E_keep / C_window`.

estimate acceptance by match-length / position bucket and cost by `T`, context, incremental cold expert union, prefetch lateness and fold policy. reject a longer draft before issuing its io. suffix lookup itself does not expose target expert routes before verification, so a cold-union predictor needs recorded history or a cheap router predictor, not a claim of zero-cost exact foresight. an initial controller can use observed per-T window time and committed count; avoid treating rejected-token count as total io waste when experts were shared with accepted rows.

## release gate / next experiment (not run here)

1. apply stage 1 only in an isolated build; keep `--suffix-draft 0` as default. stage 2 needs the missing metal compiler and parity between fused/unfused routers, base-only vs draft knobs, streamed/nonstreamed, `T=1` vs `T>1`.
2. after the other m5 benchmark finishes, run greedy repeated-code/edit and freeform prose separately. log proposal coverage, match length, offered/kept drafts, `T` histogram, mean and tail window latency, accepted tokens/sec, union experts/layer, extra cold experts beyond root, misses, prefetched bytes, late waits, and total model bytes read. separate pipeline warmup from steady state.
3. force rejection at every position for `T=2..4`, all pos modulo 4, both partial modes and lazy/eager fold; compare next logits, gdn recurrence/convolution history, `pleHist_`, `ss.ple_prev`, kv/indexer state and slop penalties with serial continuation. test contexts below and above 2051, context margin, eos at every output position and max_new boundaries.
4. exercise prefix resume, divergent requests, scr pads, save/restore and checkpoint-ring restoration. detect existing slop checkpoint caveat separately. do not let a matching generated token mask different committed hidden state.
5. only then compare stage 2 against stage 1 with identical prompts/seeds/cache initialization and quality probes. a throughput win without those checks is not a safe serving win.

bottom line: there is a real no-draft-model avenue here, but unrestricted rollback is already falsified and stronger verification routing is not lossless. the split patch keeps those two facts visible instead of hiding them behind an attractive acceptance percentage.
