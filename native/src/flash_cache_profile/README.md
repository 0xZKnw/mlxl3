# Flash cache replay

[profile53.rs](profile53.rs) replays accesses from cold to warm; [shared.rs](shared.rs) and [shared/](shared/README.md) evaluate shared-slot policies offline. The parent module is [flash_cache_profile.rs](../flash_cache_profile.rs). These simulations do not reconstruct all physical cache conditions or model GPU buffers and in-flight leases.
