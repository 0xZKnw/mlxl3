# Sources du moteur natif

Le paquet Rust est défini par le [Cargo.toml racine](../../Cargo.toml) ; `lib.rs`, `main.rs` et `hybrid_main.rs` sont ses entrées. `array.rs`, `linear.rs`, `codec.rs`, `checkpoint.rs` et `registry.rs` portent des primitives et services partagés. `lfm2.rs`, `lfm3.rs`, `qwen35.rs`, `qwen4.rs`, `gemma4.rs` et `ling.rs` organisent les familles de modèles ; `openai.rs`, `mcp.rs` et `streaming.rs` traitent les interfaces.

Les sous-dossiers documentent les modules spécialisés et leurs tests : [gpu/](gpu/README.md), [ssd_reads/](ssd_reads/README.md), [ssd_moe/](ssd_moe/README.md), [qwen4/](qwen4/README.md), [hub/](hub/README.md), [linear/](linear/README.md) et les autres modules voisins. Voir aussi les [shaders](../shaders/README.md), la [FFI](../ffi/README.md), les [tests natifs](../tests/README.md) et le [README moteur](../README.md).
