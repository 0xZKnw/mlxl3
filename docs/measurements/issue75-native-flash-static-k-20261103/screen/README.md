# Native Flash Next K/codebook screen — 2026-11-04

The isolated K/codebook specialization is retained after a matched, uninstrumented AR32 screen. NT2 is enabled in every arm; HC chaining, dynamic readers and lookahead are disabled. No quantization, routing or accumulation order was changed.

| Prompt | Control median | Specialized median | Change | Paired changes | Control drift |
| --- | ---: | ---: | ---: | --- | ---: |
| MAIN | 4.963007 tok/s | 5.441811 tok/s | +9.647% | +9.932%, +9.368% | +1.602% |
| HOLD | 5.036959 tok/s | 5.522159 tok/s | +9.633% | +9.896%, +9.364% | −2.033% |

Each candidate sample exceeds both controls for its prompt. Median TTFT falls17.6223→16.2179s on MAIN (−7.969%) and17.0688→15.0621s on HOLD (−11.756%). These observations belong to this screen and must not be added to gains from older binaries or protocols.

The four processes use explicit flags0/1/1/0 on the same release SHA256 `d2ff7420acae31ca0123e5ccff9d12bec5198c47947a739c2c51e016c3b04c8e`, qualified source `b8f0323313b046456fa887f54fc12f913613a49b`. For each process: one warmup and one measured response on each fixed prompt,32 tokens, context4096, temperature0/top-k1/repetition penalty1, prefix reuse disabled,3GiB requested cache,53 slots across each of48layers (3,174,942,528B effective capacity),4 readers. All8 warmups and8 measured responses have identical full text and token hashes per prompt, exactly32 generated/31 decode tokens, and consistent positive timings. Every native child and benchmark reader is joined.

Native Metal allocated bytes are6,608,617,472B in all eight measured responses. This is distinct from process footprint, lifetime peak and OS compression; those raw values are retained in [the summary](static-k-summary.json) and each arm report. AC power remains100%; thermal state is not measured. Background desktop activity was observed, with no other inference/compiler/prover during the campaign.

Before timing, both explicit settings pass the independent48-case component matrix across NT2 off/on/unconfigured, with4,429,824 finite half words exact in total. All five real specialized PSOs are exercised. Full target oracles at both settings match84 states/28,803,072finite F32 words and496,640logits each; budget/replay and the real AR golden/error/context+delta cancellation/recovery path pass. These tests establish the exercised fixed point, not a complete Metal or concurrency proof.

See [the preregistered protocol](protocol.json), [all arm outcomes](static-k-screen.json), and [paired results with separate memory measures](static-k-summary.json). The unchanged benchmark harness SHA256 is8560057f3bcf8e1c722d9cdd20885413beb9f5134a4b07c283de42b94cc3bdf8; its77 existing regressions were previously run.

Only two measured samples per variant/prompt are available, with32-token outputs. Sustained throughput, long context, MTP/DFlash, startup cost relative to the old binary and the installed app remain unmeasured. Default activation and its newly compiled executable require a separate qualification; these explicit-flag results are preserved rather than relabeled as default-path measurements.

The separate HC candidate is still gated after a command-count fixture failure. Its initial component/independent-write passes and failed full-model report remain retained; no HC speed gain is claimed.
