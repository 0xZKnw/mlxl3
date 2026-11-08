# Issue164 — native executor implementation

Local branch `codex/issue164-native-executor`, based on `b5dfaa1b`. No push, release or installed application replacement. Speed campaign deferred until composition is complete at the user’s request. Correctness checks remain required.

| Lot | Items | State | Evidence / next gate |
|---|---|---|---|
| L0 | R3 baseline, separate R4 reference | Integrated locally | `b252363a`; exact archived native patch `017e9461`; original R3/R4 frozen binaries retained |
| L1 | N01 specialized MB2, N03 fused down preparation | Implemented locally, opt-in | N01 Kani 142 / 5 covers; resident physical 11,538,432 exact words. N03 independent 128,000 prepared words; component + composed prefix checks pass |
| L2 | N02 experts × consumers | Implemented locally, opt-in | Component dispatch assertions and B2–B4 every-prefix model oracle pass |
| L3 | N05 sliding reads; separate N13 priority | N05 implemented locally, opt-in; N13 pending | Sliding whole-record credits; 22 CPU tests and bounded Kani; composed model gate passes |
| L4 | N04 temporal GDN; N06 segments; bounded N09 arena | Implemented locally, opt-in | GDN temporal B2/B4 (B3 masked), first joined HC/router segment, 9-frame arena; composed every-prefix physical gate passes |
| L5 | N07 common plan/transaction; N08 QSA/PLE; N14 memory admission | N07 partial; N14 implemented locally; N08 not started | Common B1/Bn contracts, owned Hot/workspace/deferred/finish alternatives and budget admission pass physical gates. Complete redesign and grouped QSA/PLE remain pending |
| L6 | N10 draft debt; N11 real D0–D3 session; conditional N12 | Pending | Target exactness, recursive hidden provenance, full transition costs |
| L7 | N15 prefill; conditional N16/N17/N18 | Pending | No automatic promotion of approximations or unprofiled hardware changes |

Detailed protocol, failures and raw evidence: `opti.md` and local `build/issue164-20261108/`. This file tracks implementation, not a claim that the complete issue is qualified or that its speed target has been reached.

## Verified local checkpoint

The complete issue is **not implemented**. The remaining lots have not been started. N01–N05 and the bounded frame arena N09 have local implementations; N06 has its first GDN→HC/router segment, and N07 has common contracts and the owned Hot/workspace/deferred/finish paths. N14 connects conservative prospective admission with the live memory ledger. All new switches default to off.

Release library: **365 passed / 133 ignored**. Strict Clippy passes. Physical B2/B3/B4 every-prefix oracles pass with finish/mix+workspace+deferred and with Hot+workspace+deferred, including actual dispatch counters, all retained states, continuations, drop and late errors. The budget gate verifies rejection before mutation and refund of draft capacity. Three CPU Kani harnesses cover plans, bounded Hot owners and reserve arithmetic; these do not prove the GPU, allocator or concurrency.

N08, N10, N11, N13 and N15 remain unimplemented. N12/N16/N17/N18 are conditional experiments requiring later profiles. The remaining N06 scheduling stages and the full N07 executor redesign are also pending. The old Hot+finish/mix combination still fails closed; predictive prefetch guards remain. No speed campaign, gain claim, push, release or application replacement.

The final native CLI builds with `direct-metal,chat`. A short production bridge correctness check confirms identical AR/MTP text and token hash (`c10b2a835cb93ad6`), including warmups. Its incidental throughput statistics are not a performance qualification. Manifest/inventory checks: 11 passed / 44 Linux-only skipped / 2 subtests passed; Ruff passes. CrossHair is unavailable for the proof-manifest expectation data. Full Kani, the remaining ignored physical suite and new Metal debug validation are not executed at this checkpoint. Local artifact identities and receipts are in `build/issue164-20261108/final-identity.json`.
