//! Wire types shared by the resident inference backends.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct MemoryStats {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub native_metal_allocated_bytes: Option<u64>,
    pub mlx_active_bytes: u64,
    pub mlx_cache_bytes: u64,
    pub mlx_peak_bytes: u64,
    pub process_footprint_bytes: Option<u64>,
    pub process_lifetime_peak_bytes: Option<u64>,
    pub process_id: u32,
    pub process_start_abstime: Option<u64>,
    pub process_compressed_logical_bytes: Option<u64>,
    pub process_compressed_peak_logical_bytes: Option<u64>,
    pub process_compressed_lifetime_logical_bytes: Option<u64>,
    /// Counts pages, saturating at i32::MAX; it does not count disk reads.
    pub process_decompressions: Option<u64>,
    /// Logical bytes consumed from swapped-in segments, not SSD bytes or swap-outs.
    pub process_ledger_swapins_logical_bytes: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct FlashDraftProfile {
    pub draft_seconds: f64,
    pub verification_seconds: f64,
    pub commit_cache_seconds: f64,
    pub verified_rows: usize,
    pub drafted_rows: usize,
    pub drafted_widths: [usize; 8],
    pub verified_widths: [usize; 8],
    pub expert_demand_bytes: u64,
    /// Submitted, completed, used, reused, timeouts, wait ns, read bytes, late demands.
    pub ssd_prefetch: [u64; 8],
}

#[cfg(any(feature = "mlx", all(target_os = "macos", feature = "direct-metal")))]
impl MemoryStats {
    pub(crate) fn from_raw(values: [u64; 12], process_id: u32) -> Self {
        let field = |index: usize, mask: u64| (values[11] & mask != 0).then_some(values[index]);
        Self {
            native_metal_allocated_bytes: None,
            mlx_active_bytes: values[0],
            mlx_cache_bytes: values[1],
            mlx_peak_bytes: values[2],
            process_footprint_bytes: field(3, 1),
            process_lifetime_peak_bytes: field(4, 1),
            process_id,
            process_start_abstime: field(10, 1),
            process_compressed_logical_bytes: field(5, 2),
            process_compressed_peak_logical_bytes: field(6, 2),
            process_compressed_lifetime_logical_bytes: field(7, 2),
            process_decompressions: field(8, 4),
            process_ledger_swapins_logical_bytes: field(9, 8),
        }
    }
}
