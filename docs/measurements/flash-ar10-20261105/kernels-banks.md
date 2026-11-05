# AR10: EXL3 kernels and expert banks

Source-only research, 2026-11-05. Local checkout: `be17b6316dbddc00a95f223a409304fc5857de42` (PR130). Read `opti.md`, including `AR10-RESEARCH-20261105`, the earlier TensorFold/llama audit, native routed-scratch adaptation, canceled expert-encoder work, and old EXL3 tiling/fusion experiments before inspecting candidates. No executable changes, compilation, proof, inference, or timing was performed for this note.

Current qualified defaults include L1 and the deferred MoE consumer. The previous five UAR mechanisms are excluded here: MoE wait movement, internal GDN cast elision, GDN norm/gate fusion, shared SSD worker queue, and fixed-weight residency. HC L3/L5, GPU greedy P04, speculative decoding, and the canceled one-encoder experiment are not proposed again.

## Pinned primary sources

The GitHub branch endpoints returned these heads during this research (not yesterday's heads):

| Official repository | Branch | Exact head | Commit time UTC |
| --- | --- | --- | --- |
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) | master | `c55b094880bb70dacac0b50c6653790650e8dd2d` | 2026-11-05 08:38:08 |
| [ashhart/TensorFold](https://github.com/ashhart/TensorFold) | main | `9200a3064153b8041930f61f5ba92102d4bea692` | 2026-11-05 08:51:26 |

TensorFold's identity was checked against its [official guide](https://tensorfold.dev/guide), which links that repository. TensorFold's affine six-bit/BF16 experts are **not EXL3**; its organization is useful evidence, not a numerically compatible kernel implementation. llama.cpp's quantized formats also differ from EXL3.

## Four distinct missing mechanisms

The ordering below favors a bounded first prototype. The largest structural change is last. These are structural opportunities, not predicted throughput improvements; no current profile isolates the time recoverable by any of them. Counts assume the target's 48 MoE layers and ten distinct routed experts per row, exclude the shared expert, and are upper bounds if routes are deduplicated.

### 1. Finish the up projection and apply SwiGLU in one kernel

**Upstream mechanism.** TensorFold computes gate/up accumulators and applies activation directly to their rounded values in its [expert shader, lines 66–76](https://github.com/ashhart/TensorFold/blob/9200a3064153b8041930f61f5ba92102d4bea692/zig/src/families/flashnext/replay_expert_source.zig#L66). Its line 76 explicitly converts both accumulated results through BF16 before the activation/product. This demonstrates that fusion need not erase an intermediate rounding boundary.

**Actual local gap.** `native/src/gpu/projection.rs:1104–1121` separately finishes gate and up, then calls `encode_swiglu`. `encode_with_scratch` allocates a fresh half destination at line 439 and dispatches the finish at lines 488–515. `native/shaders/exl3.metal:136–152` performs split accumulation, half rounding, H128, normalization, SVH and optional bias; lines 157–166 then read both stored half outputs for SwiGLU. `projection.rs:915–933` allocates a third buffer for that activation. This is independent of norm/gate fusion in GDN.

**Smallest trial.** Routed K2, batch one only: leave gate unchanged; specialize the **up finish** to read the already completed gate half, execute exactly the existing finish arithmetic, create the exact same half up value, and store the exact current half SwiGLU result. Reuse the existing partial scratch; down preparation follows that store. Leave every QMV, split, scale, bias, FMA order, sigmoid formula, and half rounding intact. No two-partial-buffer scheme is needed. Unmatched forms and trace exports use the old path.

**Structural effect:** one half640 buffer (1,280 bytes requested) and one dispatch per routed expert disappear: up to **480 allocation requests and 480 dispatches/row**, 614,400 requested buffer bytes/row. This is cumulative allocation traffic, not peak process RAM. Gate, activated, and down outputs remain distinct. Smaller messages and allocations can still lose to register pressure or kernel occupancy; no isolated time is known.

**Indispensable oracle.** Compare the real checkpoint expert component against existing independently frozen stock outputs and CPU half/finish oracles, for F16/F32 input views, strides, nonzero scales, variable inputs, signed zero, finite extreme values and both split configurations actually admitted. A mutation omitting the intermediate half up must fail on a constructed boundary input. Then full logits/hidden/84 states plus continuation, actual selected-branch counts, and one real CLI golden before MAIN32. Avoid copying TensorFold's BF16 formula or changing our `abs`/tail sigmoid.

### 2. Prepare gate and up input transforms together, retaining two outputs

**Upstream mechanism.** TensorFold [shader lines 60–73](https://github.com/ashhart/TensorFold/blob/9200a3064153b8041930f61f5ba92102d4bea692/zig/src/families/flashnext/replay_expert_source.zig#L60) visits one activation range for both gate and up; its [SX variant comment, lines 166–167](https://github.com/ashhart/TensorFold/blob/9200a3064153b8041930f61f5ba92102d4bea692/zig/src/families/flashnext/replay_expert_source.zig#L166) explicitly describes shared input conversion. This is a mechanism to adapt, not an applicable affine kernel.

**Actual local gap.** The same `Activation` is passed twice in `projection.rs:1104–1117`. Each call runs `prepare128_pipeline` / `encode_prepare_view` independently at lines 405–421, with its own Suh. Our native expert scratch already reuses one prepared buffer gate→up→down; it does not calculate two differently scaled transforms together.

**Smallest trial.** One dual-output preparation kernel, admitted only for the two real routed K2 input transforms. Load the activation once, multiply by each distinct Suh, execute two unchanged H128 transforms, and write two disjoint half prefixes. Bind their respective offsets to the unchanged gate/up QMV kernels. Keep partial scratch reuse and all output kernels unchanged. The prepared backing can have two bounded segments; no persistent pool, inter-expert alias or new cache policy. Geometry must still derive from explicit extents, never enlarged `Buffer.length`.

**Structural effect:** at most **480 fewer preparation dispatches/row**. It adds one 5,120-byte prepared prefix per live expert relative to the current single-pair scratch. This is therefore not an automatic memory win. H128 working storage/register pressure and longer overlap can erase the saved dispatch. The old source audit of two scratch pairs alone did not test this new dual-transform kernel, and its WAR/overlap caveat remains relevant.

**Indispensable oracle.** Both actual prepared prefixes must match two independent old preparations bit for bit across view dtypes/strides, changing data and distinct Suh. Poison both output regions, swap gate/up Suh in a deliberate mutation, verify wrong capacity/overlap/owner is rejected, and exercise several experts in the same command buffer. Reuse routed expert and full-state model/CLI oracles before MAIN32. This trial can be isolated from candidate 1.

### 3. Store routed down contributions in one canonical output slab

**Upstream mechanism.** llama.cpp's one-token-compatible [ID matvec dispatch, lines 3119–3137](https://github.com/ggml-org/llama.cpp/blob/c55b094880bb70dacac0b50c6653790650e8dd2d/ggml/src/ggml-metal/ggml-metal-ops.cpp#L3119) writes one destination. The [kernel, lines 3329–3354](https://github.com/ggml-org/llama.cpp/blob/c55b094880bb70dacac0b50c6653790650e8dd2d/ggml/src/ggml-metal/kernels/mul_mv.metal#L3329) derives a canonical route's destination offset separately from the selected weight ID. TensorFold likewise passes one `act`/`y` output for all expert pairs in [Run.experts, lines 432–446](https://github.com/ashhart/TensorFold/blob/9200a3064153b8041930f61f5ba92102d4bea692/zig/src/families/flashnext/replay_run.zig#L432).

**Actual local gap.** `gpu/moe.rs:320–346` separately encodes each expert and stores its down `Buffer` in a `BTreeMap`; line 387 onwards reconstructs canonical bindings for the mix. `FlashExpert::encode` returns a separately allocated down output (`projection.rs:1122–1128`). `native/shaders/flash_moe.metal:28–57` binds ten half pointers. A useful existing local precedent is the resident `MappedSwitchGlu` stack at `projection.rs:1409` and its checked route offset / finish destination at lines 1464–1483; that class still loops over experts and has different fused arithmetic, so it is not already batched EXL3 and must not replace this arithmetic wholesale.

**Smallest trial.** Allocate one fresh `[10,2560]` half slab per layer call. Allow only the routed down finish to target the checked byte prefix corresponding to its canonical route; mix reads those prefixes while preserving its exact existing striped reduction and half boundaries. Keep every expert dispatch, readiness wave, cache publication, and shared expert output unchanged. Handle duplicate slot IDs by mapping canonical occurrences explicitly; never let cold completion order choose the output offset. Do not introduce reuse across rows, generic allocators or HC workspace.

**Structural effect:** replace ten 5,120-byte buffers with one 51,200-byte buffer: up to **432 fewer buffer allocation requests/row**. Total contribution payload bytes are unchanged. One destination slab can be retained through several producers and the final mix; it must outlive every submitted command, including an error before mix. No causal allocator time is attributed to it yet. New offsets can affect hazards and cache behavior.

**Indispensable oracle.** Exercise reordered cold completions, duplicate IDs, all hot/mixed/all cold routes, one failed read and error after an earlier producer commit. Check all ten nonempty contribution prefixes and final mix against independent stock values, not only final text; poison unused regions. Preserve the exact loop and `(later + partial)` ordering in the current mix. Verify actual buffer request reduction, output size/finite values, full-state continuation and CLI golden, then MAIN32. This can be measured without true batched compute.

### 4. Dispatch EXL3 QMV across the expert axis, rather than a host loop

**Upstream mechanism.** llama.cpp dispatches the vector path with `ne123 = ne20 * ne21` at [ops lines 3126–3137](https://github.com/ggml-org/llama.cpp/blob/c55b094880bb70dacac0b50c6653790650e8dd2d/ggml/src/ggml-metal/ggml-metal-ops.cpp#L3126); its [ID kernel lines 3329–3354](https://github.com/ggml-org/llama.cpp/blob/c55b094880bb70dacac0b50c6653790650e8dd2d/ggml/src/ggml-metal/kernels/mul_mv.metal#L3329) uses grid.z for the selected expert, with an independent logical destination index. TensorFold's [one-row path](https://github.com/ashhart/TensorFold/blob/9200a3064153b8041930f61f5ba92102d4bea692/zig/src/families/flashnext/replay_run.zig#L429) dispatches gate/up and down over all `pairs`; this is separate from its multirow-only grouped branch at lines 408–427. Its [FSR storage lines 46–69](https://github.com/ashhart/TensorFold/blob/9200a3064153b8041930f61f5ba92102d4bea692/zig/src/families/flashnext/fsr_storage.zig#L46) uses nine component slabs, a logical→slot map and strides. That entire bank layout is not a small prerequisite we should blindly port.

**Actual local gap.** `moe.rs:320–346` calls `FlashExpert::encode_view` per hot expert and per cold wave member. Each routed expert currently performs prepare/QMV/finish for gate, prepare/QMV/finish for up, SwiGLU, prepare/QMV/finish for down (`projection.rs:1080–1136`): ten dispatches even with routed scratch and static/fixed PSOs active. Thus the target can issue 4,800 routed-expert dispatches/row. Static K/dimension specialization and scratch reuse reduce other overhead but leave that host/expert loop intact.

**Smallest end-to-end prototype, still medium complexity.** Batch only already CPU-complete experts belonging to the same hot/cold wave, with a bounded table of up to ten direct buffer bindings plus checked offsets. Add an expert grid dimension to the existing K2 body while keeping each output's inner accumulation/reduction order, Suh/Svh, split, and half boundaries identical. Initially keep gate/up separate, no affine conversion, no new SSD layout or speculative reads. A partial wave remains a smaller batch; do not delay SSD demand to fill ten. Keep legacy fallback for other formats, shared expert, shapes, trace and unsupported devices.

This is a **new compute-grid kernel**, not the canceled experiment that placed the same ten dispatches in one encoder. It should change actual dispatch counts as well as host calls. Moving buffer pointers into an argument table may require explicit resource declarations/lifetime retention; indirect pointer reachability is not automatically equivalent to the present direct bindings. This is the largest candidate and should not block small independent trials.

**Indispensable oracle.** Reuse frozen real EXL3 component data, compare per-expert and mixed results for batch sizes 1/2/10, permutations, duplicate IDs, different Suh/Svh, both view dtypes/strides and nonzero offsets. Verify wrong ID/bounds/argument-table mutation is detected. Actual dispatch/table-use counters must reject a legacy fallback that merely gives matching values. Then full logits/84 states/continuation for mixed hot/cold waves, real CLI golden and budget cleanup before MAIN32. CPU proof can cover bounds/table offsets, not Metal pointer residency or concurrency.

## Secondary idea, not a fourth strong performance claim

**Read-only fixed weights with untracked hazard mode.** TensorFold [replay.zig:24](https://github.com/ashhart/TensorFold/blob/9200a3064153b8041930f61f5ba92102d4bea692/zig/src/families/flashnext/replay.zig#L24) uses shared|untracked, and [allocateOptions:166–179](https://github.com/ashhart/TensorFold/blob/9200a3064153b8041930f61f5ba92102d4bea692/zig/src/families/flashnext/replay_run.zig#L166) applies those options. Our `gpu.rs:61–80` allocator receives storage mode only; fixed weight `read_weight`/`read_buffer` at lines 164–190 publishes only after CPU read completion, and `checkpoint_float_weight` at line 113 can also use synchronous upload. No current native untracked path was found. This differs from the existing residency set, which controls residency, not hazard tracking.

[Apple's untracked documentation](https://developer.apple.com/documentation/metal/mtlhazardtrackingmode/untracked) makes the application responsible for correct synchronization; [device-created resources default to tracked](https://developer.apple.com/documentation/metal/resource-synchronization). A bounded version would touch only provably immutable fixed weights, initialized synchronously before use. Keep every SSD backing, recurrent state, activation, output, snapshot and scratch tracked. Reuse fixed-weight enumeration; do not add a global synchronization framework. Benefits would concern driver bookkeeping and are completely unmeasured; it could do nothing. Any GPU-writing loader or alias invalidates this narrow safety argument and must fall back. Therefore rank below the four concrete dispatch/allocation candidates.

## Duplicates and non-candidates deliberately excluded

- Generic FlashAttention: current QSA already has online score/normalization/weighted-value kernels; no missing score tensor was found to remove.
- Native routed NT2: routed K2 currently remains NT1, but the journal has extensive old NT2/tiling experiments. This note does not count an unqualified repeat as a new idea; a separate exact history/shape audit is required before proposing it.
- Whole affine/BF16 expert port, changed quantization, SIMD-group MMA or altered sum order: not an exact minimal EXL3 optimization and not justified by these source snippets.
- TensorFold grouped multirow reuse: guarded by `rows > 1`; no claimed AR-one-row benefit.
- Global cache replacement/pooling: earlier POOL04 and scratch/overlap limitations remain; none of these four asks for that policy change.

No percentages, benchmark results, proof results or measured savings are claimed. The current default-only MAIN32 result and current profile remain root-owned artifacts; cumulative request/dispatch counts above are derived from the actual callers, not converted into milliseconds or added to existing gains.
