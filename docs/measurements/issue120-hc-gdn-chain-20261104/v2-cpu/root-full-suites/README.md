# Root full-suite qualification of P13 S2

All recorded root checks pass: format, inventory250, strict MLX/chat all-target Clippy, complete direct/chat suite352passed/53ignored and MLX/chat suite388passed/271ignored. Counts are summed from every final test-result line, including integration/doc targets; counts.json preserves each summary. Exact commands, time bounds and process/PGID joins are in report.json. These logs were copied and compressed without rerunning a check.

Ignored tests remain ignored in these automatic suites. The separately selected two physical Rust fixtures and CLI are reported in../../quality-v2; their execution is not folded into these totals. Prior no-default CPU336/8 and policy Kani283checks/15covers are explicitly reused unchanged in the root receipt, not rerun here. They do not prove Metal retention, OS or concurrency. Static's existing v2-cpu format/direct-Clippy/source-diff archive remains untouched.
