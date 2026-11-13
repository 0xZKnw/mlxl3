# Shared MoE expert on GPU

[shared_block.rs](shared_block.rs) handles shared-expert rows with immutable weights; it does not mutate the router, cache, or state. It supplements [moe.rs](../moe.rs) and [moe_block.rs](../moe_block.rs); related kernels are in [shaders/](../../../shaders/README.md).
