# Native Flash GPU reuse audit

Baseline: main `599628120c95df45b00a2076ffe466331a9c94f2`. Preregistrations: `I75-FLASH-GPU-REUSE-AUDIT-20261103` and `I75-FLASH-RESIDENT-NT2-20261103` in opti.md. No throughput result is established by this audit.

| Historical path | Actual native status | Decision |
| --- | --- | --- |
| Canonical Flash SG4 and row→group accumulation | Present in `_qmv_tile_kernel.metal` and `flash_reference_reduction.h` | Preserve unchanged |
| MTP `fc_hidden` MB4/SG2/two splits | Present in `flash_mtp_hidden_qmv`, guarded shape2560×2560 | Preserve NT1; do not widen generic reduction |
| opt20 HC half weights / FP32 arithmetic | Present in `gpu/hc.rs`, `flash_hc*.metal` | No new F32-weight promotion |
| H128 input cast/scale/preparation | Native `prepare128` and view preparation retain half boundaries, radix16 then8 | Already present; do not advertise as a new fusion |
| opt67 stable PSO specialization | Native fixed PSOs and submission-time Params | Present by construction; native does not recreate an MLX kernel factory per route count |
| Old hot-masked route compaction/grouping | Native CPU demand plan submits selected routes; AR is one target row | No blind compaction port; cross-token grouping requires a distinct batch path |
| FLASHSSD-2026-10-12-10 resident NT2 | Missing: ordinary native Flash hardcodes NT1 | First small isolated candidate, default OFF |
| opt46 expert-slot NT2 | Rejected: AR MAIN+0.523%/HOLD−0.637%; final MTP HOLD−5.576% with16% drift | Keep routed K2 path unchanged |
| opt54Q scratch relayout / register decode | Component quality passed, no consistent useful gain; baseline restored | Do not resurrect |
| ISSUE59 T04 SwiGLU→down scale fusion | Removed/rejected: AR−1.6413%, MTP+0.432%, DFlash+0.942%, drift | No validated native boost to reuse |
| opt15/15B MTP pair MB2 | Rejected at the historical3% threshold; source restored | No validated missing boost |
| ISSUE66/67 CODE01/02 MB3/4 | Source prepared/default OFF; physical controls not executed | Unqualified; no throughput claim |

Historical resident NT2 evidence is retained in `docs/measurements/flash-next-ssd-impl/opt-10-nt2/`: lossless prototype patch, micro-summary.json, quality-execution.json and bridge-execution.json. All eight matrices were numerically exact; measured component latency reductions ranged0.35–16.59%. Macro generation was interrupted/nonconclusive with drift above10%. These are component results, are not additive and do not establish a generation gain.

The actual local Flash205 checkpoint has K4/MUL1 for shared-down, QKV, Z, GDN-out, QSA-q, QSA-out and lm_head; `mtp.fc_embedding` is K3/MUL1. Its model-average2.05bpw does not identify these resident matrices. The native port is gated by those actual K/codebook headers, M5 hardware, complete128-wide outputs and a strict default-OFF flag. It reuses the canonical tile body unchanged, while `fc_hidden`, generic EXL3 and routed record constructors remain excluded.

The new reason for one controlled trial is the standalone native command/runtime path. Root coordinates subsequent quality gates and release ABBA at3GiB cache,4 readers,lookahead0,prompt reuse OFF. Historical32/256MiB swap termination and3% stop rules are not copied; current user instructions and the common new protocol govern the campaign. The earlier0.6GiB quick5tok/s run is not a baseline for this comparison.
