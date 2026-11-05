# Issue 140 — AR2: QSA index append and attention

AR2 encodes the indexed QSA append and dense attention in one command buffer. It is enabled by default on the qualified Apple M5. `MLXL3_NATIVE_QSA_APPEND_CHAIN=0` restores the separate boundary; other hardware defaults to OFF. Invalid setting values remain errors. T05 and the thirteen accepted defaults are preserved; AR1's runtime is absent and AR3/AR4 remain OFF.

The join applies only with an indexer, without trace export, and with a post-append length of at most **2051**. Length 2052 retains CPU sparse selection and its synchronization. Unindexed and trace calls keep their former path. Growth and copy-on-write copies remain outside the joined command; they are not counted as eliminated submissions. No numerical kernel, attention selection, SSD scheduling, speculative decoding or model weight changes.

The staged operation owns its input, index, slab and command. It checks the original state owner, offset and published highwater before encoding, and publishes only after the entire command completes. Dropping a submitted pending operation joins it without publishing. Unsubmitted commands are never waited on. The old prefix survives growth, retained snapshots, stale prepares, failure and retry.

## Throughput screen

Apple M5 / 24 GiB / macOS 27.2, deployment target 26.2; `Qwen3.8-Flash-Next-exl3-205-65c89531`, checkpoint `65c895314393431c09050b2e04e250836b3a6eb4`, EXL3 average 2.05 bits/H4/MTP2 weights, **plain AR without MTP or DFlash**.

MAIN prompt: 87 tokens, 128 generated/127 decode tokens, context 4096, expert cache 3 GiB, four readers, depth 0, prompt cache 0, temperature 0/top-k 1/penalty 1. Each arm has one warmup and one measured request. Brave and Deezer remain open. No other inference, compiler or prover runs during GPU measurement; no forced cooling, swap stop or OS setting change.

| Arm | Decode tok/s |
|---|---:|
| OFF start | 8.846880962431834 |
| AR2 ON | 9.379397977324716 |
| OFF end | 9.187287097728767 |

The candidate is above both fresh controls: an observed **2.09–6.02%** increase in this short screen. The controls drift upward by about 3.85%; these are single measured samples, not confidence intervals or a kernel-time attribution. All six warmup/measured outputs match the complete independent golden128 text and hash. [Raw screen](speed128/report.json), [evaluation](evaluation-a.json).

The final default composition is qualified separately, with the setting absent, using model/CLI gates and one MAIN128 arm. Its result is **8.8451 tok/s**, recorded in [default/speed128/report.json](default/speed128/report.json); it is not an additional controlled speedup.

## Correctness and actual selection

[The original quality gate](quality/report.json) has three selected phases, all joined:

- The physical QSA test covers 27 cases at lengths 3/4/32/33/1023/1024/2047–2052. It compares 12,800 F32 output words against the independently captured attention reference, then nonempty finite outputs and over one million persistent Half words against the unchanged separate path at synthetic boundaries. The historical reference's five short contexts are independent; the long synthetic boundary comparison is OFF versus ON, not an independent upstream long-context capture.
- Trace, missing indexer, sparse 2052, growth, fork, retained snapshot, prepare→forward→restore stale preparation, encode failure, post-terminal publication rejection, submitted/unsubmitted abandonment and retry are exercised. Actual counters are `[35 completed, 34 published, 2 abandoned]`, including intentional failures. No incomplete case is promoted to parity.
- The model checks the historical independent oracle: 84 initial trace states, 57,606,144 state words, 993,280 logits and 81,920 hidden bytes across the established continuation/rollback protocol. Trace remains a fallback; actual normal rows independently require exactly `[12 completed,12 published,0 abandoned]`. The model's total is `[72,72,0]`. This does not claim that the 84 transient tensors remain available after a normal call.
- The real CLI checks two complete golden tokens, the direct engine with no MLX library loaded, shutdown and child joining. The throughput screen then checks full MAIN128 outputs.

[Default qualification](default/quality/report.json) reruns the affected model and CLI gates with the variable absent, requiring `qsa_append_default_environment_unset:true` and the same real per-row counters. The numerical component is unchanged by default selection and its original 27 cases are retained rather than rerun as a new experiment.

## CPU checks, proofs and CI

The final default build passes **412 CPU tests / 100 ignored**, strict Clippy for `direct-metal,chat` and `mlx,chat`, release CLI/libtest builds, formatting and the native inventory. **19 controller tests** exercise flag/default selection, strict marker validation, missing/false/nonfinite/duplicate evidence, nonzero actual deltas and existing bounded process cleanup. **11 portable inventory tests plus two subtests** pass. Exact commands and logs: [CPU checks](default/cpu/checks.json), [reviewed results](default/cpu/reviewed-summary.json).

Three Kani harnesses exercise the actual production policy: indexed/dense/CPU-free eligibility, terminal owner/state publication and abandoned-submission guards, plus the default parser. Results: **251 reported obligations: 244 SUCCESS, seven internal UTF8 UNREACHABLE; 15 covers satisfied**. Guard inputs span all `u32`/boolean values, with no assumptions or harness loops. The parser checks symbolic UTF8 byte lengths 0–4 with unwind 10 and no assumptions; longer strings are outside that bounded check. The baseline harnesses are preserved and the new ones are registered in the dynamically partitioned CI inventory. [Kani results](default/cpu/kani.log).

These are CPU policy model checks and physical regression tests, not a proof of Metal, the driver, the allocator or concurrent SSD execution. CrossHair is unavailable in the venv; its diagnostic is retained in [cpu/crosshair.log](cpu/crosshair.log), with no Python proof claimed. Other models and unqualified hardware were not physically rerun. Hosted CI must be reported for the exact publication commit; local success alone is not a green CI claim.

## Source identities and preserved failures

Current main `6030540b5573185a4bf8b13207011872d94d0045` is integrated, including PR141/T05, PR139's experimental OFF options and Desktop changes. Only the affected Flash path is qualified here; the incoming Desktop work is not claimed as newly tested.

| Composition | Source pins | CLI SHA-256 | libtest SHA-256 |
|---|---:|---|---|
| Explicit ON screen | 300 | `5e06c0d67843eb082fb764dc9a21ca67b12cc5b927be49d3ac3a55c66578c698` | `95bf7c8155af3a380725dea2e08ab6bbf0fa0a3ea89086a7a80e7a322b586576` |
| Retained default | 303 | `725fc7e18dce92ed8b16ced1679df8574872f7e595350cfcad5d8b045edcd4bd` | `45f2c7009048150e2d4d3ba6a54c3d026322c644264494c1efd1f529f622f007` |

The [explicit-ON freeze](freeze.json) and [default freeze](default/freeze.json) pin native sources, workflows, helpers, historical oracles/goldens and eleven attention reference files. Five checkpoint metadata files are pinned; no new full-weight rehash is claimed. The first source state and raw evidence were archived and roundtrip checked before changing default selection: [source archive](source-a.tar.gz), [source index](source-a-index.json), [raw archive](raw-a.tar.gz), [raw index](raw-a-index.json). The final default state is archived separately before publication: [sources](default/source-default.tar.gz), [source index](default/source-default-index.json), [raw evidence](default/raw-default.tar.gz), [raw index](default/raw-default-index.json), [roundtrip/evaluation](default/evaluation-default.json).

Failures remain visible: the first Clippy run rejected one fixture's modulo style; the default controller's first lint rejected a lambda/with style; a reused Kani supervisor preflight lacked an `artifacts` field and failed before launching a prover. Their original reports/logs are retained; subsequent success does not overwrite them. No GPU correctness divergence or timeout occurred in the completed gates. The installed app is not replaced by this engine PR, and no release or merge is performed here.
