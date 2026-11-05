# Flash Next AR34: device-resident routing and qualified default

Issue [#138](https://github.com/0xZKnw/mlxl3/issues/138), following the immutable-source research in [#137](https://github.com/0xZKnw/mlxl3/pull/137). T01/T02 and the negative T04 replay are published separately in [draft #139](https://github.com/0xZKnw/mlxl3/pull/139). All #138 candidates have now received their specified implementation or opportunity gate; a source gate is not a runtime speed test.

T05 submits an owned HC/router producer P, then H performs the actual GPU lookup against an immutable 53-record argument buffer before CPU route readback. Ready contributions retain their original positions; demand misses use fresh Shared destinations while H owns the old records. The canonical EXL3 kernels, Half rounding and full eight-stripe-plus-two route sum are preserved. This is intralayer AR scheduling, without a future token or provisional successor.

The user requested default activation after the controlled screen. It is ON only on qualified Apple M5/Tier2 hardware. `MLXL3_NATIVE_DEVICE_RESIDENT_ROUTE=0` disables it; invalid values are rejected. Cache53, depth0, no-prefetch, NT2, format and ownership guards remain. Other hardware keeps the reference path. The thirteen previously accepted settings remain unchanged. No MTP/DFlash, canceled profiling, P04 or HC L3/L5 work was resumed. Brave and Deezer stayed open.

## Controlled short screen

Apple M5/24 GiB, macOS27.2, deployment26.2; checkpoint `65c895314393431c09050b2e04e250836b3a6eb4`, Qwen3.8-Flash-Next EXL3avg2.05/H4/MTP2 weights used as plain AR. MAIN87 prompt, context4096, cache3GiB, four readers, depth0, prefix0, temperature0/topk1/penalty1. Each arm has one warm and one measured request. Generated32/128 correspond to decode31/127; their rates are not pooled.

| Generated tokens | OFF start | T05 ON | OFF end | Observed increase against controls |
|---|---:|---:|---:|---:|
|32|8.7118 tok/s|9.0383 tok/s|8.7496 tok/s|3.30–3.75%|
|128|8.8893 tok/s|9.0013 tok/s|8.8414 tok/s|1.26–1.81%|

Controls drift +0.43% and −0.54%. Every warm/measured output matches the full independent golden text/hash. These are small positive observations, not confidence intervals or proof of a universal speedup. The default-only retained binary subsequently measured **9.2732 tok/s** on128, with the same full golden. Its difference from the screen is not an additional optimization gain. Historical10.3–10.4 figures belong to other measurements.

`freeze.json` pins274 inputs and experimental CLI `83ee43130fe02bcc5af00c76c700fe749081f8116ce16856f6b9f5a39bfef34b`; `default/freeze.json` pins277 inputs and default CLI `b9e7adb899aaa5339d9570d44987571d174c0567cd181ebab1a355d79d09d9cc`. These builds originate from `01da8f01` plus the explicitly recorded dirty implementation, native baseline `82fff5a5`. The final retained composition merges main `f5118ad897071bbdb858d1e609dd3719addca60a`; see below.

## Correction and ownership

Five explicitly selected physical Rust tests plus the real CLI passed. They cover P's12814 finite F32 words/10IDs, 53 actual record bindings, 15 H submissions, 256000 compared Half words, 0/1/9/10 hits, malformed/duplicate IDs, stale generations, remapping and an effective wrong-pointer mutation. Real fresh SSD readback checks1,248,012 bytes and the old backing. H publication errors remain latched across repeated completion; abandonment drains F then H, including an unattached/unsubmitted consumer.

The independent-process model oracle checks57,606,144 state words,993,280 logits,81,920 hidden bytes and84 initial states, continuation and producer failure rollback. Actual H counters are positive, including mixed hits/misses. The short numerical model fixture does not fill the cache enough to retire an old record; component ownership tests and the full32/128 generation goldens cover different parts of that boundary. Default qualification repeats the real model and CLI with the T05 environment variable **absent**, requires that fact in the model marker, and verifies positive H activity.

One active transaction per Metal owner bounds retired records at ten; admission reserves14,577,272 bytes, including2MiB of auxiliary headroom. Typed read owners protect content from A11 reuse. H belongs to HotOutput, never FlashMoeRow::last. HotOutput retains F before F is committed and drains F then H before releasing owners and the active charge. F success never hides an H error.

Review found and fixed missing autorelease pools and abandonment ownership. A direct Drop on public MoeOutput broke field moves; the final private consumer guard preserves the public API. Initial Metal syntax, Clippy, controller mock and per-slot-generation test failures remain archived. Synthetic rejection after Completed is not an induced Metal driver error.

## Verification and limits

- Before default promotion:400 CPU tests passed/95 ignored; five physical Rust tests were then selected explicitly. Two zero-test binaries are not counted. Three Kani harnesses passed211 obligations/10 covers.
- Default-only change: four affected CPU tests passed/two physical tests ignored/396 filtered. The400-test suite was not rerun for this parser-only promotion. Strict Clippy direct/chat and mlx/chat, release CLI/libtest, formatting and inventory checks passed.
- Four Kani harnesses on the retained default source passed460 reported obligations:453SUCCESS, seven internal UTF8 UNREACHABLE,19 covers. Parser inputs are symbolic UTF8 length0..4 with unwind10 and no assumptions; admission/identity/offset/peak inputs retain full u32/u64/usize domains. This bounded CPU verification does not prove Metal, the allocator, SSD or concurrent execution.
- Fifteen default controller tests pass, including the eleven inherited checks. Native inventory is updated to280 interesting files. Linux orchestration/process tests were skipped on this Mac; two portable inventory assertions were invoked directly. CrossHair is unavailable in the qualified Python3.12 environment; its diagnostic is retained. MLX numerical inference and unrelated physical model tests were not rerun.
- Hosted CI is checked on the final published commit; local results do not imply hosted success. No installed application, merge or release is part of this work.

Commands are retained verbatim in CPU JSON receipts and raw logs. Principal commands: `cargo clippy --locked --all-targets --features direct-metal,chat -- -D warnings`, the same with `mlx,chat`, `cargo test --locked --features chat`, `cargo test --locked --features chat --lib device_hot -- --test-threads=1`, and `cargo kani --lib --no-default-features --jobs 1` with the four fully named harnesses in the receipt. Release builds use `MACOSX_DEPLOYMENT_TARGET=26.2`; Metal OFF and ON/test helpers were compiled with per-command Xcode DEVELOPER_DIR.

## Evidence

- [Evaluation](evaluation.json), [prototype freeze](freeze.json), [default freeze](default/freeze.json).
- [Prototype archive receipt](prototype-archive-receipt.json):274 source pins and73 raw files, roundtrip checked. Four source versions were restored by reversing default-only edits into temporary files and matched the original freeze hashes exactly.
- [Default archive receipt](default/archive-receipt.json):277 source pins reconstructed from the prototype archive plus eight changed files, and26 raw files, all roundtrip checked. No full-weight archive or new full-weight rehash was performed; five model metadata pins and the checkpoint identity are retained.

The new [issue #140](https://github.com/0xZKnw/mlxl3/issues/140) prioritizes vectorizing the remaining five stages of **cold Ready cohorts** (AR1), then QSA append/attention boundaries (AR2). T05 already batches the hot path; it must not be sold again as AR1. No #140 prototype or benchmark was started here.

## Retained composition with current main

Main `f5118ad8` was merged without replacing its Qwen additions. Only the journal conflicted; both histories were preserved. Executable changes merged automatically; shared-file additions affect Qwen module wiring, method visibility and a Qwen test, while the T05 arithmetic and ownership code remain unchanged.

The final composition at merge `2108f5e7` passed **402 CPU tests /97 ignored**, strict direct/chat and mlx/chat Clippy, formatting, four Kani harnesses (460 reported obligations/453SUCCESS/seven internal UTF8 UNREACHABLE/19 covers), Metal ON/test compilation and native inventory282. The independent model and real CLI were rerun without the T05 variable, followed by one128-token warm+measured qualification: **9.0630 tok/s**, full golden parity and all children joined. This final run is not a fresh OFF/ON/OFF comparison. Other model GPU tests remain unexecuted by this campaign.

[`main/freeze.json`](main/freeze.json) pins280 sources and final CLI `3126e70f536696ae60449afbc3fc0838627d705fcc19f08e1c1a2195bbff36a5`; libtest `2f9b632c92c5628c97a0e58f947e80291c10a62fcae30723ed6e386fcca481eb`. [`main/archive-receipt.json`](main/archive-receipt.json) verifies the final source delta and raw evidence against the previous two immutable archives. No older measurement is relabeled as this binary.
