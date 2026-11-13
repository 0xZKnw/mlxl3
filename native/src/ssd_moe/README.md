# MoE with SSD-backed experts

[policy.rs](policy.rs) resolves read, preparation, and residency policies; [grouped_micro.rs](grouped_micro.rs) and [row_oracle.rs](row_oracle.rs) supplement the parent [ssd_moe.rs](../ssd_moe.rs) module. Tests cover indices, prefetch, shared experts, and cold scenarios. The read transport is documented in [ssd_reads/](../ssd_reads/README.md).
