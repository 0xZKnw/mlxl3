use anyhow::{Context, Result, ensure};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct ResolvedPolicy {
    pub(crate) dynamic_readers_requested: bool,
    pub(crate) native_prepare_requested: bool,
    pub(crate) native_prepare: bool,
    pub(crate) force_lru: bool,
    pub(crate) resident_gates: bool,
    pub(crate) overlap: bool,
    pub(crate) readers: usize,
    pub(crate) gpu_hot: bool,
    pub(crate) gpu_hot_wide: bool,
    pub(crate) unique_cohorts: bool,
    pub(crate) shared_exl3: bool,
    pub(crate) stable_expert_kernels: bool,
    pub(crate) shared_batch: bool,
    pub(crate) weighted_epilogue: bool,
    pub(crate) cold_ready: bool,
    pub(crate) cold_ready_rows: i32,
    pub(crate) cold_priority: bool,
    pub(crate) cold_shared: bool,
    pub(crate) routed_index_cache: bool,
    pub(crate) prefetch_pipeline: bool,
    pub(crate) predictor_half: bool,
}

impl ResolvedPolicy {
    pub(crate) fn resolve(target_layer: bool, overlap: bool) -> Result<Self> {
        let control = |name, default: &str| -> Result<String> {
            match std::env::var(name) {
                Ok(value) => Ok(value),
                Err(std::env::VarError::NotPresent) => Ok(default.into()),
                Err(error) => Err(error.into()),
            }
        };
        let native_prepare_requested =
            crate::flash_support::parse_gpu_hot_wide(&control("MLXL3_FLASH_NATIVE_PREPARE", "0")?)
                .context("MLXL3_FLASH_NATIVE_PREPARE must be 0 or 1")?;
        let native_prepare = native_prepare_requested && target_layer && crate::array::is_m5_gpu()?;
        let force_lru =
            crate::flash_support::parse_lru_override(&control("MLXL3_SSD_CACHE_POLICY", "auto")?)
                .context("MLXL3_SSD_CACHE_POLICY must be auto or lru")?;
        let predictor_half = crate::flash_support::parse_predictor_half(&control(
            "MLXL3_SSD_PREDICTOR_DTYPE",
            "f32",
        )?)
        .context("MLXL3_SSD_PREDICTOR_DTYPE must be f32 or f16")?;
        let resident_gates = match control("MLXL3_SSD_RESIDENT_GATES", "0")?.as_str() {
            "0" => false,
            "1" => true,
            _ => anyhow::bail!("MLXL3_SSD_RESIDENT_GATES must be 0 or 1"),
        };
        let gpu_hot = match control("MLXL3_SSD_GPU_HOT", "0")?.as_str() {
            "0" => false,
            "1" => target_layer,
            _ => anyhow::bail!("MLXL3_SSD_GPU_HOT must be 0 or 1"),
        };
        let gpu_hot_wide =
            crate::flash_support::parse_gpu_hot_wide(&control("MLXL3_SSD_GPU_HOT_WIDE", "0")?)
                .context("MLXL3_SSD_GPU_HOT_WIDE must be 0 or 1")?;
        let unique_cohorts =
            crate::flash_support::parse_gpu_hot_wide(&control("MLXL3_SSD_UNIQUE_COHORT", "0")?)
                .context("MLXL3_SSD_UNIQUE_COHORT must be 0 or 1")?;
        let shared_exl3 =
            crate::flash_support::parse_gpu_hot_wide(&control("MLXL3_SSD_SHARED_EXL3", "0")?)
                .context("MLXL3_SSD_SHARED_EXL3 must be 0 or 1")?;
        let stable_expert_kernels = crate::flash_support::parse_gpu_hot_wide(&control(
            "MLXL3_SSD_STABLE_EXPERT_KERNELS",
            "0",
        )?)
        .context("MLXL3_SSD_STABLE_EXPERT_KERNELS must be 0 or 1")?;
        let shared_batch =
            crate::flash_support::parse_gpu_hot_wide(&control("MLXL3_SSD_SHARED_BATCH", "0")?)
                .context("MLXL3_SSD_SHARED_BATCH must be 0 or 1")?;
        let weighted_epilogue =
            crate::flash_support::parse_gpu_hot_wide(&control("MLXL3_SSD_WEIGHTED_EPILOGUE", "0")?)
                .context("MLXL3_SSD_WEIGHTED_EPILOGUE must be 0 or 1")?;
        let cold_ready =
            crate::flash_support::parse_gpu_hot_wide(&control("MLXL3_SSD_COLD_READY", "1")?)
                .context("MLXL3_SSD_COLD_READY must be 0 or 1")?;
        let cold_ready_rows =
            crate::cold_ready::parse_max_rows(Some(&control("MLXL3_SSD_COLD_READY_ROWS", "2")?))
                .context("MLXL3_SSD_COLD_READY_ROWS must be 2 or 4")?;
        let cold_priority =
            crate::flash_support::parse_gpu_hot_wide(&control("MLXL3_SSD_COLD_PRIORITY", "0")?)
                .context("MLXL3_SSD_COLD_PRIORITY must be 0 or 1")?;
        let cold_shared =
            crate::flash_support::parse_gpu_hot_wide(&control("MLXL3_SSD_COLD_SHARED", "0")?)
                .context("MLXL3_SSD_COLD_SHARED must be 0 or 1")?;
        let routed_index_cache = crate::flash_support::parse_gpu_hot_wide(&control(
            "MLXL3_SSD_ROUTED_INDEX_CACHE",
            "0",
        )?)
        .context("MLXL3_SSD_ROUTED_INDEX_CACHE must be 0 or 1")?;
        let dynamic_readers_requested =
            crate::flash_support::parse_gpu_hot_wide(&control("MLXL3_SSD_DYNAMIC_READERS", "1")?)
                .context("MLXL3_SSD_DYNAMIC_READERS must be 0 or 1")?;
        let prefetch_pipeline =
            crate::flash_support::parse_gpu_hot_wide(&control("MLXL3_SSD_PREFETCH_PIPELINE", "0")?)
                .context("MLXL3_SSD_PREFETCH_PIPELINE must be 0 or 1")?;
        let readers = std::env::var("MLXL3_SSD_READERS")
            .unwrap_or_else(|_| "4".into())
            .parse::<usize>()
            .context("invalid SSD reader count")?;
        ensure!((1..=4).contains(&readers), "SSD readers must be in 1..=4");
        Ok(Self {
            dynamic_readers_requested,
            native_prepare_requested,
            native_prepare,
            force_lru,
            resident_gates,
            overlap,
            readers,
            gpu_hot,
            gpu_hot_wide,
            unique_cohorts,
            shared_exl3,
            stable_expert_kernels,
            shared_batch,
            weighted_epilogue,
            cold_ready,
            cold_ready_rows,
            cold_priority,
            cold_shared,
            routed_index_cache,
            prefetch_pipeline,
            predictor_half,
        })
    }
}
