# Current HC kernels on accepted L1

Numerical quality passed before speed: 16 actual model cases with independent 84-state initial archive, 460,849,152 finite F32 state values and 7,946,240 logits compared bit exactly. Each untraced control and candidate used 34 L1 triples; continuation explicitly disabled experimental paths and used zero triples. Active cases required 97 up-mix uses and/or 96 down-inject uses. Workspace and extra SSD wave were disabled. One real CLI completion generated the golden two-token hash, passed the existing ledger check and loaded no MLX; all children were joined. Exact commands and results are in [quality/report.json](quality/report.json).

The exploratory order extended the earlier A1–L2–workspace series with L3, L4, then a closing L1 A2. It was preregistered before this extension, not retrospectively presented as a balanced experiment. MAIN32 only, one warmup plus one measurement per new process. A1 is historical, not simultaneous. No HOLD, ABBA, interpolation of drift, additive gains, installed-app update or statistically established speed improvement is claimed.

| Variant | Decode tok/s | vs A1 | vs A2 |
| --- | ---: | ---: | ---: |
| Historical L1 A1 | 6.561890 | — | — |
| L1 + L3 up-mix | 6.653878 | +1.4018% | +0.5882% |
| L1 + L4 down-inject | 6.524045 | −0.5767% | −1.3745% |
| Closing L1 A2 | 6.614966 | +0.8089% | — |

All eight full text/token-hash outputs, including historical A1 warmup and measurement, matched. L3 and L4 allocated 6,608,617,472 Metal bytes each. The endpoint drift is observed only. [performance/summary.json](performance/summary.json) retains timings and comparisons; raw reports are adjacent. The copied controllers are exact sources used, with their adaptation-specific static and negative checks deferred during the user-prioritized speed window.

Artifacts: current CLI `b4899081478369ae575dff4863d65570efb9909b384fdbbf6330a93a2e660110`, current libtest `7382bddbfb23538485bdb654e63319ab1513494189e67fd112e18f1e50df1296`, accepted L1 reference CLI `9e2c538c85851159bdb53a054e97485c6accfff8260ebc7f858aaa2fa5b9758c`. Root freeze and CPU/proof receipts remain in the parent directory. No source or binary changed during these runs. Flags and exact Python argv are retained in the copied controllers; the existing qualified CLI-short runner was reused with explicit current-root PYTHONPATH.
