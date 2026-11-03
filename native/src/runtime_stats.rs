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

#[cfg(any(
    kani,
    feature = "mlx",
    all(target_os = "macos", feature = "direct-metal")
))]
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

#[cfg(any(kani, all(target_os = "macos", feature = "direct-metal")))]
impl MemoryStats {
    pub(crate) fn native(host: [u64; 9], metal_bytes: u64, pid: u32) -> Self {
        let mut raw = [0; 12];
        raw[3..].copy_from_slice(&host);
        Self {
            native_metal_allocated_bytes: Some(metal_bytes),
            ..Self::from_raw(raw, pid)
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(13)]
fn native_memory_stats_separate_metal_and_process_validity() {
    let host: [u64; 9] = kani::any();
    let bytes: u64 = kani::any();
    let pid: u32 = kani::any();
    let stats = MemoryStats::native(host, bytes, pid);
    assert_eq!(stats.native_metal_allocated_bytes, Some(bytes));
    assert_eq!(
        (
            stats.mlx_active_bytes,
            stats.mlx_cache_bytes,
            stats.mlx_peak_bytes
        ),
        (0, 0, 0)
    );
    assert_eq!(stats.process_id, pid);
    assert_eq!(
        stats.process_footprint_bytes,
        (host[8] & 1 != 0).then_some(host[0])
    );
    assert_eq!(
        stats.process_compressed_logical_bytes,
        (host[8] & 2 != 0).then_some(host[2])
    );
    assert_eq!(
        stats.process_decompressions,
        (host[8] & 4 != 0).then_some(host[5])
    );
    assert_eq!(
        stats.process_ledger_swapins_logical_bytes,
        (host[8] & 8 != 0).then_some(host[6])
    );
    kani::cover!(bytes != 0 && host[8] == 0);
    kani::cover!(bytes != 0 && host[8] == 15);
}
