# Native GPU implementation

The engine's Metal modules cover memory and views, projection, attention/QSA, GDN, MoE, routing, MTP, prefetch, and model operations. [gpu.rs](../gpu.rs) declares the modules, types, and kernel selection; Metal sources are in [shaders/](../../shaders/README.md).

The [moe/](moe/README.md) and [qwen/](qwen/README.md) subdirectories supplement their corresponding modules. This directory's `*_tests.rs` files include targeted checks; read their annotations before running physical GPU tests.
