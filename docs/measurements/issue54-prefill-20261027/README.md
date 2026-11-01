# Issue 54 — Flash Next prefill

Updated 2026-10-27. **P01A/B and P01L rejected and removed; HEADONLY kept with default1 and final verification complete.** No push, app replacement or release by this task.

## P01B: layer-major prefill, removed

Native Rust/MLX/Metal prototype; layer blocks2/4/8/32/128, internal tiles4, final-only GDN scan, existing grouped expert execution, causal captures and row1 draft appends. Base7978fceac30d991a8c06d72114fc932b2443f555; model EXL3 `65c89531`, draft Q4/ring `a22a1744`, M5 standard24GiB/MLX0.32.2. Cache3GiB/53slots, allocator64MiB, fixed DFlashB2, context4096. Full environment and source/binary hashes: p01b/source-initial.json, engine.json, memory-diagnostics-source.json and runtime-identity.json. All original evidence retained.

140 physical differential cases passed (14lengths ×5chunks ×AR/DFlash): nonempty finite248320logits, full target states,12800captures/token and draftKV with shape/dtype, continuation3, cancellation/reset/recovery and errors. Two isolated capture/reset mutations failed as expected; correct sources restored. Lib182passed/156ignored; main44passed/8ignored; CPU247passed/6ignored; supervisor16passed. Bridge recovery14requests,4cancellations,2expectederrors,8completions; bridge quality17tokens on87/82/246prompts inAR/DFlash matched hashes/text/counts. r2 reports validated with processes/readers joined and zero new Swapouts. Physical tests are sampled validation, not a GPU proof.

Kani parser: symbolicUTF8 length0..3, unwind5,232obligations/0failed/8unreachable,4/4covers; bounded CPU parser check only. Real MLX Kani ICE intrinsics.rs:243 before obligations: unverified. Format/strictClippy/releaseMLX+chat and MLX withoutchat built. GPU primitive oracle A is recorded separately, not attributed as B end-to-end proof.

### Speed: both orders H/C and C/H

One warmup+3measures/prompt/arm; same frozen production binary;128output tokens, exact texts/hashes/DFlash counters throughout. All36measured responses retained. Ranges and both orders in [results.json](p01b/rejected-final/results.json). Pooled medians6/variant:

| Prompt | TTFT H→C | Complete H→C | Decode change |
|---|---|---|---|
|87tokens|8.875→6.667s|19.880→18.187s|−4.56%|
|82tokens|8.412→6.252s|19.958→18.284s|−3.83%|
|246tokens|24.624→17.969s|36.691→30.506s|−3.41%|

MLX allocation equal; process peak grew only about88MiB in first pair, while compressed logical pages increased about2GiB (not equivalent to2GiB extra allocation). This effect and decode slowdown repeat in reverse order; cause not established. Original decode/compression gates fail. User waived variability veto, then explicitly required removal for decode slowdown: **rejected, native files restored**, patch/full sources retained in p01b/rejected-final. No claim that all original qualification gates passed.

### Interruptions retained

B1 model group stopped after70cases with132300800bytes new Swapout; r2 resumed only after user freedRAM with fresh anchor. R2 first baselineLONG varied4.576%TTFT; a two-warmup r3 retry was interrupted after user waived the variability veto, partial data excluded as preregistered. All r2 samples kept. The512-budget diagnostic was stopped/reaped at user request; candidate never launched, no diagnostic gain claimed. Earlier A interruptions and wide pytest/unittest runner failures are detailed in [previous-readme.md](p01b/rejected-final/previous-readme.md); unrelated missingponyexl3 failures do not validate or invalidate this native change.

## HEADONLY: prefill gain retained

Only the final output head is needed during prefill. Token-major layer/expert order, recurrence and casts remain unchanged. MLXL3_FLASH_PREFILL_HEADS=0|1 now defaults to1 in local code;0 is the original path. MTP prefill keeps its per-token behavior. No app replacement/push/release.

Two orders H/C and C/H, one warmup+3measures/prompt/arm,128outputs:36measured+12warmup responses. Every hash/text/count and DFlash expert-demand byte count matched. Zero new Swapouts, all processes/readers joined. Pooled medians6/arm, M5/EXL3/cache3GiB/fixedDFlashB2, as above:

| Prompt | TTFT reduction | Complete response reduction | Decode change |
|---|---|---|---|
|87tokens|4.210%|2.509%|+1.079%|
|82tokens|3.827%|2.107%|+0.955%|
|246tokens|5.074%|4.008%|+1.778%|

[All ranges and medians](headonly/pooled.json). Per-order TTFT reductions2.883..6.328%; decode −1.015..+4.102%, descriptive rather than a universal guarantee. Strict original global gate passed first pair and failed reverse: HOLD complete0.958% was below1%, LONG decode−1.015% below−1%. Those false statuses and variability remain intact in screen1/2. User explicitly accepted the variation and positive aggregate; decision retains the robust prefill reduction. RAM conditions changed between pairs after user closed apps. No percentages from different campaigns added.

Quality on source/binaryv1:18differential cases (9lengths×AR/DFlash), full248320logits/semantic states/draftKV metadata+bytes/continuation,12800captures per row; count of actual heads N→1; errors/cancellation/recovery. Two head/capture mutations failed as expected. Native package276passed/166ignored; CPU246passed/6ignored. Real bridge14recovery requests and4quality arms with3warmup+3responses17tokens each matched prior goldens. Independent trace fixture:5tokens,240layer checks,10GDN checks(only layers0/1 available),5full-logit checks plus reset, all nonempty and exact. Native test passed; wrapper label incorrectly required GDN46 and failed. Failure retained; [scope verification](headonly/trace-nonvacuity.json), no fabricated oracle for the other34GDN layers.

Commands: cargo fmt --all; MLXL3_MLX_ROOT=<pinned mlx package> cargo clippy --features mlx,chat --all-targets -- -D warnings; cargo build --release --features mlx,chat; cargo test --release --features mlx,chat --no-run; cargo test --no-default-features --features chat. Ignored physical function final_prefill_head_preserves_states_draft_and_continuation selected explicitly/alone in native lib; exact commands/options and supervision in headonly/*-plan.json. Kani --lib --tests --features mlx,chat --harness actual_final_prefill_head_contract hit compiler ICE intrinsics.rs:243 before obligations: no GPU proof. Python supervisors/validators reuse existing source; CrossHair unavailable. Source/binary hashes in headonly/source-v1.json, preserved sources/patch. Final default-without-variable bridge verification is included in the P01L source campaign.

## P01L: scalar work ordered by layer, rejected and removed

A different mechanism from rejected P01B: layer-major blocks≤128, **every projection/GDN/MoE call remains scalar**. No new neural compute kernel or grouped expert reservation. All LFU keys/counts/ages/clock/pending state and expert I/O were compared with serial replay from the same initial metadata. Captures concatenated without arithmetic changes; draft appends remained row1. No-fetch path only. Its MLXL3_FLASH_PREFILL_LAYERS flag and implementation are now removed. Cancellation sampled after2rows and after a committed128-token block, with reset/recovery checked.

Build/strictClippyMLX+chat passed; CPU246passed/6ignored; MLX withoutchat compiled with existing warnings. Real Kani used a symbolic U32token and2booleans on2tokens(shapes/offsets/rejection), but compiler ICE prevented verification. Sources/binaries p01l/source-v1.json/prototype-v1.patch. All36comparisons passed (9lengths×AR/DF×2candidates): full logits/states/captures/KV/continuation, exact LFU metadata and I/O, head counts, errors/reset/recovery. LFU-clock and wrong-capture mutants failed as intended. Native package276passed/166ignored; bridge14recovery requests and four17-token quality arms matched HEADONLY goldens. Every supervised child/reader joined, zero new Swapouts.

Screen H=HEADONLY1/LAYERS0, C=HEADONLY1/LAYERS1,128outputs, one warmup+3measures/prompt/arm:

| Prompt | TTFT H→C | Complete H→C | Decode change |
|---|---|---|---|
|87tokens|8.827→9.969s (+12.935% time)|20.205→21.666s (+7.231%)|−3.394%|
|82tokens|8.554→9.437s (+10.328% time)|20.630→21.815s (+5.742%)|−2.433%|
|246tokens|23.989→26.865s (+11.991% time)|37.040→39.515s (+6.682%)|−0.998%|

[Full screen and ranges](p01l/screen1.json), favourable=false, stable=false. All hashes/text/counts/expert-demand bytes matched, zero new Swapouts. No reverse order was launched after this negative screen. Extra per-row synchronization is plausible but was not isolated as the cause. Full sources/patch/hashes are preserved in p01l/rejected-final. No claim that LFU equality diagnoses the earlier P01B decode regression.

## Final retained code

Only three native files differ from7978fce: qwen4, flash_draft and flash_draft_runtime. Final production calculations match measured HEADONLYv1; qwen4 changes only its missing-env default to1, flash_draft is byte-identical, runtime changes only its cfgKani proof body. main/ssd_moe/flash_support/gated_delta are restored to HEAD. Source/binary identities: [final/source.json](headonly/final/source.json); retained patch and exact frozen-binary commands in the same directory.

After removal: fmt/strictClippyMLX+chat/release build/test builds and MLX withoutchat passed. Final normal suites: lib181/156, main44/8, contracts16/0, OpenAI35/0, runtime_gpu0/2 (passed/ignored), total276/166. Selected physical test additionally passed12differential cases (lengths1/2/7/8/9/17×AR/DF), full logits/states/captures/KV/continuation and head counts/errors. Both real bridge defaults ran without HEADS or LAYERS variables,3warmups+3responses per mode, and matched all original explicitHEADS1 goldens including non-timing DFlash profile/demand bytes. Eight supervised processes joined with zero new Swapouts. [Inspected results](headonly/final/verification-inspection.json). The first inspection script mishandled valid AR dflash_profile=null and stopped before writing a report; its error is preserved, corrected reader passed without rerunning inference.

Final Kani attempted the real implementation with allU32tokens/headsbool on two tokens, checking shape/offset or invalid-token rejection; compiler ICE occurs before obligations. No proof established. The original18case validation, independent fixture scope and negative controls above remain tied to unchanged production calculations. No new speed claim from final17-token quality runs. AR and other models have no measured speed gain from this campaign; quoted speed numbers are for fixed DFlashB2.

CI PR/pushmain reviewed and existing shipped MLX checks cover compilation/lint; no remote CI run covers unpublished local changes. App published by the other chat does not contain these experiments. TensorFold research is complete: [sources, distinct candidates and historical exclusions](../tensorfold-20261027/README.md). No TensorFold prototype or benchmark was launched.
