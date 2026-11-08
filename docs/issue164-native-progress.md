# Issue164 — native executor implementation

Local branch `codex/issue164-native-executor`, based on `b5dfaa1b`. No push, release or installed application replacement. Speed campaign deferred until composition is complete at the user’s request. Correctness checks remain required.

| Lot | Items | State | Evidence / next gate |
|---|---|---|---|
| L0 | R3 baseline, separate R4 reference | Integrated locally | `b252363a`; exact archived native patch `017e9461`; original R3/R4 frozen binaries retained |
| L1 | N01 specialized MB2, N03 fused down preparation | Implemented locally, opt-in | N01 Kani 142 / 5 covers; resident physical 11,538,432 exact words. N03 independent 128,000 prepared words; component + composed prefix checks pass |
| L2 | N02 experts × consumers | Implemented locally, opt-in | Component dispatch assertions and B2–B4 every-prefix model oracle pass |
| L3 | N05 sliding reads; separate N13 priority | N05 implemented locally, opt-in; N13 pending | Sliding whole-record credits; 22 CPU tests and bounded Kani; composed model gate passes |
| L4 | N04 temporal GDN; N06 segments; bounded N09 arena | Pending | Preserve every prefix state and external snapshots |
| L5 | N07 common plan/transaction; N08 QSA/PLE; N14 memory admission | Pending | Keep B1 optimizations, connect actual resource accounting |
| L6 | N10 draft debt; N11 real D0–D3 session; conditional N12 | Pending | Target exactness, recursive hidden provenance, full transition costs |
| L7 | N15 prefill; conditional N16/N17/N18 | Pending | No automatic promotion of approximations or unprofiled hardware changes |

Detailed protocol, failures and raw evidence: `opti.md` and local `build/issue164-20261108/`. This file tracks implementation, not a claim that the complete issue is qualified or that its speed target has been reached.
