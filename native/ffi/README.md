# C++ interface and MLX bridge

`mlx_bridge.cpp` connects the Rust engine to MLX operations; neighboring headers define contracts for disk reads, work queues, kernel caching, Metal capabilities, and process memory. `process_stats.cpp` supplements process statistics. Rust declarations and callers are under [native/src/](../src/README.md), with C++ checks in [native/tests/](../tests/README.md).
