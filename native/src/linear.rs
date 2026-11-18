//! Rust dispatch of the existing MLXL3 serialized QMV shaders through native MLX.
use crate::{
    array::{self, Array, Dtype},
    checkpoint::Checkpoint,
    codec::{self, Codebook, reference_reduction_header},
};
use anyhow::{Context, Result, ensure};

#[cfg(test)]
thread_local! {
    static USE_Q36_REFERENCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}
#[cfg(test)]
pub(crate) fn q36_reference() -> bool {
    USE_Q36_REFERENCE.with(std::cell::Cell::get)
}
#[cfg(test)]
pub(crate) fn with_q36_reference<T>(run: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            USE_Q36_REFERENCE.with(|v| v.set(self.0));
        }
    }
    let _restore = Restore(USE_Q36_REFERENCE.with(|v| v.replace(true)));
    run()
}

fn grouped_mb3_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = GROUPED_MB3_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MLXL3_EXPERIMENTAL_GROUPED_MB3").as_deref() == Ok("1"))
}

#[cfg(test)]
thread_local! {
    static GROUPED_MB3_TEST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static REFERENCE_MB34_REQUESTS_TEST: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
}

#[cfg(test)]
pub(crate) fn with_grouped_mb3_test<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            GROUPED_MB3_TEST_OVERRIDE.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(GROUPED_MB3_TEST_OVERRIDE.with(|flag| flag.replace(Some(enabled))));
    run()
}

fn dense_decode_nt4_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = DENSE_DECODE_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MLXL3_DENSE_DECODE_NT4").as_deref() != Ok("0"))
}

#[cfg(test)]
thread_local! {
    static DENSE_DECODE_TEST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Test-only override: compare geometries with one model and the same GPU clock.
#[cfg(test)]
pub(crate) fn with_dense_decode_nt4_test<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DENSE_DECODE_TEST_OVERRIDE.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(DENSE_DECODE_TEST_OVERRIDE.with(|flag| flag.replace(Some(enabled))));
    run()
}

fn lane_order_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = LANE_ORDER_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::contracts::exl3_lane_order_enabled(
            std::env::var_os("MLXL3_EXL3_LANE_ORDER")
                .as_deref()
                .map(|value| value.to_str().unwrap_or("")),
        )
    })
}

#[cfg(test)]
thread_local! {
    static LANE_ORDER_TEST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Load under true to prepare indices; this override can then select stock on the same weights.
#[cfg(test)]
pub(crate) fn with_lane_order_test<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            LANE_ORDER_TEST_OVERRIDE.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(LANE_ORDER_TEST_OVERRIDE.with(|flag| flag.replace(Some(enabled))));
    run()
}

/// Q27D-23: threshold the engine binary applies when `MLXL3_DENSE_TENSOR_ROWS` is absent
/// (0 = exact kernels). The library and its tests keep the exact default.
static DENSE_TENSOR_PRODUCT_DEFAULT: std::sync::atomic::AtomicI32 =
    std::sync::atomic::AtomicI32::new(0);

/// Called once by the engine binary before any model loads.
pub fn set_dense_tensor_product_default(rows: i32) {
    if (2..=16).contains(&rows) {
        DENSE_TENSOR_PRODUCT_DEFAULT.store(rows, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Q27D-16 (not bit-exact): first verification width sent to the M5 TensorOps kernel.
pub fn dense_tensor_rows_threshold() -> Result<Option<i32>> {
    #[cfg(test)]
    if let Some(value) = DENSE_TENSOR_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return Ok(value);
    }
    // Resolved once: this runs for every projection, so no per-call OS capability query.
    static SETTING: std::sync::OnceLock<Option<Option<i32>>> = std::sync::OnceLock::new();
    let setting = *SETTING.get_or_init(|| {
        let value = std::env::var("MLXL3_DENSE_TENSOR_ROWS").ok();
        let fallback = DENSE_TENSOR_PRODUCT_DEFAULT.load(std::sync::atomic::Ordering::Relaxed);
        if value.is_none() && fallback > 0 {
            return Some(Some(fallback)).map(|t: Option<i32>| {
                t.filter(|_| array::supports_exl3_tensor_ops().unwrap_or(false))
            });
        }
        crate::contracts::dense_tensor_rows_setting(value.as_deref()).map(|threshold| {
            threshold.filter(|_| array::supports_exl3_tensor_ops().unwrap_or(false))
        })
    });
    setting.context("MLXL3_DENSE_TENSOR_ROWS must be 0, off or an integer in 2..=16")
}

#[cfg(test)]
thread_local! {
    static DENSE_TENSOR_TEST_OVERRIDE: std::cell::Cell<Option<Option<i32>>> = const { std::cell::Cell::new(None) };
}

/// Test-only override of the TensorOps verification threshold.
#[cfg(test)]
pub(crate) fn with_dense_tensor_rows<T>(threshold: Option<i32>, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<Option<i32>>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DENSE_TENSOR_TEST_OVERRIDE.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(DENSE_TENSOR_TEST_OVERRIDE.with(|flag| flag.replace(Some(threshold))));
    run()
}

/// Q27D-21: whole-block verification in TensorOps mode; `MLXL3_DENSE_TENSOR_BLOCK=0` disables it.
pub fn dense_tensor_block_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MLXL3_DENSE_TENSOR_BLOCK").as_deref() != Ok("0"))
}

/// Q27D-04 rollback switch for the shared-decode verification layouts.
pub fn dense_shared_enabled() -> bool {
    #[cfg(test)]
    if let Some(enabled) = DENSE_SHARED_TEST_OVERRIDE.with(std::cell::Cell::get) {
        return enabled;
    }
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        crate::contracts::dense_shared_verify_enabled(
            std::env::var_os("MLXL3_DENSE_SHARED_VERIFY")
                .as_deref()
                .map(|value| value.to_str().unwrap_or("")),
        )
    })
}

#[cfg(test)]
thread_local! {
    static DENSE_SHARED_TEST_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Test-only override: stock and shared layouts on the same loaded weights.
#[cfg(test)]
pub(crate) fn with_dense_shared_test<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DENSE_SHARED_TEST_OVERRIDE.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(DENSE_SHARED_TEST_OVERRIDE.with(|flag| flag.replace(Some(enabled))));
    run()
}

fn lane_order_indices(
    input: i32,
    output: i32,
    k: usize,
    cb: Codebook,
    grouped: bool,
) -> Result<Option<Array>> {
    if !crate::contracts::exl3_lane_order(
        1,
        input,
        output,
        k,
        grouped,
        cb == Codebook::Mul1,
        array::is_m5_gpu()?,
        lane_order_enabled(),
    ) {
        return Ok(None);
    }
    let values = (0..input)
        .map(|index| i32::try_from(crate::contracts::exl3_lane_source_index(index as u32)))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let indices = Array::from_i32(&values, &[input])?;
    indices.eval()?;
    Ok(Some(indices))
}

fn lane_order_selected(indices: Option<&Array>, matrix_rows: i32) -> bool {
    indices.is_some()
        && lane_order_enabled()
        && (matches!(matrix_rows, 1 | 2)
            || crate::contracts::dense_shared_rows(matrix_rows, true, dense_shared_enabled()))
}

/// Q27D-33: fused QMV prologue/epilogue for the exact paths, opt-in (`MLXL3_QMV_FUSED_GLUE=1`):
/// same values bit for bit, no bridge gain established. F16, BF16 and F32 activations only.
fn qmv_fused_glue(dtype: Dtype) -> bool {
    let supported = matches!(dtype, Dtype::Float16 | Dtype::BFloat16 | Dtype::Float32);
    #[cfg(test)]
    if let Some(enabled) = QMV_FUSED_GLUE_TEST.with(std::cell::Cell::get) {
        return enabled && supported;
    }
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    supported && *VALUE.get_or_init(|| std::env::var("MLXL3_QMV_FUSED_GLUE").as_deref() == Ok("1"))
}

#[cfg(test)]
thread_local! {
    static QMV_FUSED_GLUE_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_qmv_fused_glue<T>(value: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            QMV_FUSED_GLUE_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(QMV_FUSED_GLUE_TEST.with(|flag| flag.replace(Some(value))));
    run()
}

fn lane_order_input(
    xhat: Array,
    indices: Option<&Array>,
    matrix_rows: i32,
    groups: i32,
    input: i32,
) -> Result<(Array, bool)> {
    if let Some(indices) = indices.filter(|_| lane_order_selected(indices, matrix_rows)) {
        let shape = xhat.shape().to_vec();
        let value = xhat
            .reshape(&[matrix_rows, groups, input])?
            .take(indices, 2)?
            .reshape(&shape)?;
        Ok((value, true))
    } else {
        Ok((xhat, false))
    }
}

/// Strict `MLXL3_Q36_HEAD_LANE` rollback switch, read only for the qualified head shape.
fn q36_head_lane_enabled() -> Result<bool> {
    const MESSAGE: &str = "MLXL3_Q36_HEAD_LANE must be 0 or 1";
    let value = match std::env::var("MLXL3_Q36_HEAD_LANE") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error).context(MESSAGE),
    };
    crate::contracts::q36_head_lane_setting(value.as_deref()).context(MESSAGE)
}

#[cfg(test)]
thread_local! {
    static Q36_HEAD_LANE_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static Q36_HEAD_LANE_USES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Run with the prepared head lane path on (or stock on the same weights) and count its uses.
#[cfg(test)]
pub(crate) fn with_q36_head_lane_test<T>(enabled: bool, run: impl FnOnce() -> T) -> (T, usize) {
    struct Restore(Option<bool>, usize);
    impl Drop for Restore {
        fn drop(&mut self) {
            Q36_HEAD_LANE_TEST.with(|flag| flag.set(self.0));
            Q36_HEAD_LANE_USES.with(|uses| uses.set(self.1));
        }
    }
    let _restore = Restore(
        Q36_HEAD_LANE_TEST.with(|flag| flag.replace(Some(enabled))),
        Q36_HEAD_LANE_USES.with(|uses| uses.replace(0)),
    );
    let result = run();
    (result, Q36_HEAD_LANE_USES.with(std::cell::Cell::get))
}

/// Lane gather of the 35B head input, built once at load (Q36-HEAD-LANE-P24).
fn q36_head_lane_indices(input: i32, output: i32, k: usize, cb: Codebook) -> Result<Option<Array>> {
    if !crate::contracts::q36_head_lane_shape(
        input,
        output,
        k,
        cb == Codebook::Mcg,
        array::is_m5_gpu()?,
    ) || !q36_head_lane_enabled()?
    {
        return Ok(None);
    }
    let values = (0..input as u32)
        .map(|index| i32::try_from(crate::contracts::exl3_lane_source_index(index)))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let indices = Array::from_i32(&values, &[input])?;
    indices.eval()?;
    Ok(Some(indices))
}

pub(crate) fn checkpoint_dtype(dtype: &str, name: &str) -> Result<Dtype> {
    Ok(match dtype {
        "F16" => Dtype::Float16,
        "BF16" => Dtype::BFloat16,
        "F32" => Dtype::Float32,
        "U16" => Dtype::UInt16,
        "I16" => Dtype::Int16,
        "U32" => Dtype::UInt32,
        "I32" => Dtype::Int32,
        "BOOL" => Dtype::Bool,
        other => anyhow::bail!("unsupported model tensor dtype {other} ({name})"),
    })
}

pub fn checkpoint_array(checkpoint: &Checkpoint, name: &str) -> Result<Array> {
    let info = checkpoint
        .tensors
        .get(name)
        .with_context(|| format!("missing tensor {name}"))?;
    let dtype = checkpoint_dtype(&info.dtype, name)?;
    let shape = info
        .shape
        .iter()
        .map(|&n| i32::try_from(n))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if dtype == Dtype::Bool {
        return Array::from_bytes(&info.read_bytes()?, &shape, dtype);
    }
    let (file, offset) = info.source()?;
    Array::from_file(file, offset, &shape, dtype)
        .with_context(|| format!("cannot load checkpoint tensor {name}"))
}

pub fn checkpoint_arrays(
    checkpoint: &Checkpoint,
    names: &[String],
    workers: usize,
) -> Result<Vec<Array>> {
    let infos = names
        .iter()
        .map(|name| {
            checkpoint
                .tensors
                .get(name)
                .with_context(|| format!("missing tensor {name}"))
        })
        .collect::<Result<Vec<_>>>()?;
    let dtypes = infos
        .iter()
        .zip(names)
        .map(|(info, name)| checkpoint_dtype(&info.dtype, name))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        dtypes.iter().all(|&dtype| dtype != Dtype::Bool),
        "batched boolean tensors are unsupported"
    );
    let shapes = infos
        .iter()
        .map(|info| {
            info.shape
                .iter()
                .map(|&n| i32::try_from(n))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(Into::into)
        })
        .collect::<Result<Vec<_>>>()?;
    let sources = infos
        .iter()
        .map(|info| info.source())
        .collect::<Result<Vec<_>>>()?;
    let files = sources.iter().map(|&(file, _)| file).collect::<Vec<_>>();
    let offsets = sources
        .iter()
        .map(|&(_, offset)| offset)
        .collect::<Vec<_>>();
    Array::from_files(&files, &offsets, &shapes, &dtypes, workers)
}

pub struct Exl3Linear {
    trellis: Array,
    suh: Array,
    svh: Array,
    bias: Option<Array>,
    k: usize,
    cb: Codebook,
    rows: i32,
    cols: i32,
    simdgroups: i32,
    reference_accumulation: bool,
    mtp_hidden_batch: bool,
    flash_batch: bool,
    reference_mb34: bool,
    dense_epilogue: bool,
    dec_one: bool,
    lane_indices: Option<Array>,
    head_lane_indices: Option<Array>,
    lfm3_prefill: [bool; 2],
}

impl Exl3Linear {
    pub(crate) fn weight_arrays(&self) -> impl Iterator<Item = &Array> {
        [&self.trellis, &self.suh, &self.svh]
            .into_iter()
            .chain(&self.bias)
            .chain(&self.lane_indices)
            .chain(&self.head_lane_indices)
    }

    pub fn from_checkpoint(checkpoint: &Checkpoint, prefix: &str) -> Result<Self> {
        let dense_epilogue = if checkpoint.model_type == "qwen4_exp" {
            let setting = match std::env::var("MLXL3_FLASH_DENSE_EPILOGUE") {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(error) => return Err(error.into()),
            };
            crate::dense_epilogue::enabled_setting(setting.as_deref())
                .context("MLXL3_FLASH_DENSE_EPILOGUE must be 0 or 1")?
        } else {
            false
        };
        let reference_mb34 = if checkpoint.model_type == "qwen4_exp" {
            let setting = match std::env::var("MLXL3_FLASH_REFERENCE_MB34") {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(error) => return Err(error.into()),
            };
            crate::dense_epilogue::enabled_setting(setting.as_deref())
                .context("MLXL3_FLASH_REFERENCE_MB34 must be 0 or 1")?
        } else {
            false
        };
        let reference_mb34_roles = if checkpoint.model_type == "qwen4_exp" {
            let setting = match std::env::var("MLXL3_FLASH_REFERENCE_MB34_ROLES") {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(error) => return Err(error.into()),
            };
            crate::contracts::flash_reference_mb34_roles(setting.as_deref())
                .context("MLXL3_FLASH_REFERENCE_MB34_ROLES must be qkv, z, out or all")?
        } else {
            crate::contracts::FlashReferenceMb34Roles::Qkv
        };
        let trellis = checkpoint_array(checkpoint, &format!("{prefix}.trellis"))?;
        let dec_one = if checkpoint.model_type == "qwen4_exp" {
            let setting = match std::env::var("MLXL3_FLASH_DEC_ONE") {
                Ok(value) => Some(value),
                Err(std::env::VarError::NotPresent) => None,
                Err(error) => return Err(error.into()),
            };
            crate::dense_epilogue::enabled_setting(setting.as_deref())
                .context("MLXL3_FLASH_DEC_ONE must be 0 or 1")?
                && crate::dec_one::target_scope(prefix)
        } else {
            false
        };
        let scale = |primary, legacy| {
            let key = format!("{prefix}.{primary}");
            checkpoint_array(
                checkpoint,
                if checkpoint.tensors.contains_key(&key) {
                    &key
                } else {
                    return checkpoint_array(checkpoint, &format!("{prefix}.{legacy}"));
                },
            )
        };
        let cb = if checkpoint.tensors.contains_key(&format!("{prefix}.mul1")) {
            Codebook::Mul1
        } else if checkpoint.tensors.contains_key(&format!("{prefix}.mcg")) {
            Codebook::Mcg
        } else {
            Codebook::Default
        };
        let bias_key = format!("{prefix}.bias");
        let bias = if checkpoint.tensors.contains_key(&bias_key) {
            Some(checkpoint_array(checkpoint, &bias_key)?.astype(Dtype::Float16)?)
        } else {
            None
        };
        let k = *trellis
            .shape()
            .last()
            .context("trellis must have dimensions")? as usize
            / 16;
        let mut linear = Self::new(
            trellis,
            scale("suh", "su")?,
            scale("svh", "sv")?,
            bias,
            k,
            cb,
        )?;
        let qualified = crate::lfm3::m5_text8(
            checkpoint.model_type == "lfm3",
            checkpoint.bits,
            array::is_m5_gpu()?,
        ) && linear.k == 8
            && linear.cb == Codebook::Mcg;
        linear.lfm3_prefill = lfm3_prefill_options(qualified);
        if checkpoint.model_type == "qwen4_exp" {
            linear.dense_epilogue = dense_epilogue;
            linear.dec_one = dec_one;
            linear.simdgroups = 4;
            linear.reference_accumulation = true;
            // Flash keeps its reference layout; lane order was qualified on other models.
            linear.lane_indices = None;
            linear.head_lane_indices = None;
            linear.flash_batch = crate::flash_support::multi_token_verifier()?;
            linear.reference_mb34 = reference_mb34
                && if reference_mb34_roles == crate::contracts::FlashReferenceMb34Roles::Qkv {
                    crate::contracts::flash_reference_mb34_scope(prefix, linear.rows, linear.cols)
                } else {
                    crate::contracts::flash_reference_mb34_role_scope(
                        prefix,
                        linear.rows,
                        linear.cols,
                        reference_mb34_roles,
                    )
                };
            linear.mtp_hidden_batch = prefix == "mtp.fc_hidden";
            if linear.mtp_hidden_batch {
                ensure!(
                    linear.rows == 2560 && linear.cols == 2560,
                    "unsupported MTP hidden projection"
                );
                linear.simdgroups = 2;
            }
        }
        Ok(linear)
    }

    pub fn new(
        trellis: Array,
        suh: Array,
        svh: Array,
        bias: Option<Array>,
        k: usize,
        cb: Codebook,
    ) -> Result<Self> {
        codec::check_k(k)?;
        let shape = trellis.shape();
        ensure!(
            shape.len() == 3 && shape[2] == (16 * k) as i32 && shape.iter().all(|&n| n > 0),
            "invalid trellis shape"
        );
        let rows = shape[0].checked_mul(16).context("row overflow")?;
        let cols = shape[1].checked_mul(16).context("column overflow")?;
        ensure!(
            rows % 128 == 0 && cols % 128 == 0,
            "public EXL3 projections require 128-aligned dimensions"
        );
        ensure!(
            suh.shape() == [rows] && svh.shape() == [cols],
            "EXL3 scales have wrong dimensions"
        );
        ensure!(
            bias.as_ref().is_none_or(|b| b.shape() == [cols]),
            "EXL3 bias has wrong dimensions"
        );
        ensure!(
            matches!(trellis.dtype(), Dtype::UInt16 | Dtype::Int16),
            "EXL3 trellis must contain 16-bit integers"
        );
        let trellis = trellis.view(Dtype::UInt16)?;
        let simdgroups = if array::is_m5_gpu()? { 8 } else { 4 };
        // Materialize each loaded matrix so temporary CPU buffers/casts can be
        // released promptly instead of retaining an entire model load graph.
        trellis.eval()?;
        suh.eval()?;
        svh.eval()?;
        if let Some(bias) = &bias {
            bias.eval()?;
        }
        Ok(Self {
            trellis,
            suh,
            svh,
            bias,
            k,
            cb,
            rows,
            cols,
            simdgroups,
            reference_accumulation: false,
            mtp_hidden_batch: false,
            flash_batch: false,
            reference_mb34: false,
            dense_epilogue: false,
            dec_one: false,
            lane_indices: lane_order_indices(rows, cols, k, cb, false)?,
            head_lane_indices: q36_head_lane_indices(rows, cols, k, cb)?,
            lfm3_prefill: [false; 2],
        })
    }

    pub(crate) fn lfm3_grouped_prefill_enabled(&self) -> bool {
        self.lfm3_prefill[1]
    }

    /// Lane-ordered stock projection: eligible for the shared-decode verify layouts.
    pub(crate) fn shared_rows_ready(&self) -> bool {
        self.lane_indices.is_some()
            && lane_order_enabled()
            && self.head_lane_indices.is_none()
            && !self.reference_accumulation
            && !self.mtp_hidden_batch
            && self.k != 7
    }

    pub fn input_dims(&self) -> i32 {
        self.rows
    }
    pub fn output_dims(&self) -> i32 {
        self.cols
    }

    fn splits(&self, input_tiles: i32, output_tiles: i32) -> i32 {
        if self.reference_accumulation {
            crate::flash_support::reference_splits(input_tiles as u32, output_tiles as u32) as i32
        } else {
            split_count(input_tiles, output_tiles)
        }
    }

    fn try_dense_epilogue(
        &self,
        x: &Array,
        partials: &Array,
        rows: i32,
        splits: i32,
    ) -> Result<Option<Array>> {
        if !self.dense_epilogue
            || !self.reference_accumulation
            || self.k != 4
            || self.cb != Codebook::Mul1
            || self.rows != 6144
            || self.cols != 2560
            || self.simdgroups != 4
            || splits != 2
            || !(1..=2).contains(&rows)
            || !matches!(x.dtype(), Dtype::Float16 | Dtype::Float32)
            || self.bias.is_some()
        {
            return Ok(None);
        }
        let Some(value) = crate::dense_epilogue::apply(partials, &self.svh, rows)? else {
            return Ok(None);
        };
        let mut shape = x.shape().to_vec();
        *shape
            .last_mut()
            .context("empty dense epilogue input shape")? = self.cols;
        Ok(Some(value.astype(x.dtype())?.reshape(&shape)?))
    }

    fn try_dec_one(
        &self,
        x: &Array,
        xhat: &Array,
        words: &Array,
        rows: i32,
        header: &str,
    ) -> Result<Option<Array>> {
        if !self.dec_one
            || !self.reference_accumulation
            || self.mtp_hidden_batch
            || self.k != 4
            || self.cb != Codebook::Mul1
            || self.rows != 6144
            || self.cols != 2560
            || self.simdgroups != 4
            || self.splits(384, 160) != 2
            || !(1..=2).contains(&rows)
            || !matches!(x.dtype(), Dtype::Float16 | Dtype::Float32)
            || self.svh.dtype() != Dtype::Float16
            || self.bias.is_some()
        {
            return Ok(None);
        }
        let value = crate::dec_one::apply(xhat, words, &self.svh, rows, header)?;
        let mut shape = x.shape().to_vec();
        *shape.last_mut().context("empty DEC-ONE input shape")? = self.cols;
        Ok(Some(value.astype(x.dtype())?.reshape(&shape)?))
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let elements = x
            .shape()
            .iter()
            .try_fold(1i64, |count, &dimension| {
                count.checked_mul(i64::from(dimension))
            })
            .context("linear input size overflow")?;
        ensure!(
            x.shape().last() == Some(&self.rows) && elements % i64::from(self.rows) == 0,
            "invalid EXL3 linear input for {} inputs",
            self.rows
        );
        let matrix_rows = i32::try_from(elements / i64::from(self.rows))?;
        ensure!(matrix_rows > 0, "EXL3 linear input cannot be empty");
        ensure!(
            !self.mtp_hidden_batch || matrix_rows == 4,
            "MTP hidden projection requires four streams"
        );
        if self.reference_accumulation && !self.mtp_hidden_batch && matrix_rows > 1 {
            if self.flash_batch
                && self.k != 7
                && crate::contracts::flash_reference_batch_layout(matrix_rows).is_some()
            {
                return self.forward_qmv_batch(x, matrix_rows);
            }
            let matrix = x.reshape(&[matrix_rows, self.rows])?;
            let rows = (0..matrix_rows)
                .map(|row| self.forward(&matrix.slice(0, row, row + 1)?))
                .collect::<Result<Vec<_>>>()?;
            let mut shape = x.shape().to_vec();
            *shape.last_mut().context("empty EXL3 input shape")? = self.cols;
            return Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0)?.reshape(&shape);
        }
        if crate::contracts::use_tensor_ops(matrix_rows, array::supports_exl3_tensor_ops()?) {
            if self.lfm3_prefill[0]
                && x.dtype() == Dtype::Float16
                && x.shape() == [1, matrix_rows, self.rows]
                && crate::lfm3::reconstructed_prefill_allowed(
                    matrix_rows,
                    self.rows,
                    self.cols,
                    self.k,
                )
            {
                return lfm3_prefill::reconstruct_mm(self, x);
            }
            return self.forward_qmm_tensor(x);
        }
        if matrix_rows >= 24 {
            let matrix = x.reshape(&[matrix_rows, self.rows])?;
            let mut batches = Vec::new();
            for start in (0..matrix_rows).step_by(23) {
                batches.push(self.forward(&matrix.slice(
                    0,
                    start,
                    start.saturating_add(23).min(matrix_rows),
                )?)?);
            }
            let mut shape = x.shape().to_vec();
            *shape.last_mut().context("empty EXL3 input shape")? = self.cols;
            return Array::concatenate(&batches.iter().collect::<Vec<_>>(), 0)?.reshape(&shape);
        }
        if crate::contracts::dense_tensor_rows(
            matrix_rows,
            self.shared_rows_ready(),
            dense_tensor_rows_threshold()?,
        ) {
            if let Some(indices) = self.lane_indices.as_ref().filter(|_| tensor_lane_enabled()) {
                let output = qmm_tensor_lane(
                    x,
                    &self.trellis,
                    &self.suh,
                    &self.svh,
                    indices,
                    self.rows,
                    self.cols,
                    self.k,
                    self.cb,
                    self.cols / 16,
                    0,
                )?;
                return match &self.bias {
                    Some(bias) => output.add(bias),
                    None => Ok(output),
                };
            }
            return self.forward_qmm_tensor(x);
        }
        if matrix_rows > 1 {
            if let Some((first, _)) = crate::contracts::dense_shared_chunks(
                matrix_rows,
                self.shared_rows_ready(),
                dense_shared_enabled(),
            ) {
                let matrix = x.reshape(&[matrix_rows, self.rows])?;
                let head = self.forward_qmv_batch(&matrix.slice(0, 0, first)?, first)?;
                let tail = self.forward_qmv_batch(
                    &matrix.slice(0, first, matrix_rows)?,
                    matrix_rows - first,
                )?;
                let mut shape = x.shape().to_vec();
                *shape.last_mut().context("empty EXL3 batch shape")? = self.cols;
                return Array::concatenate(&[&head, &tail], 0)?.reshape(&shape);
            }
            if self.k != 7 {
                return self.forward_qmv_batch(x, matrix_rows);
            }
            let matrix = x.reshape(&[matrix_rows, self.rows])?;
            let rows = (0..matrix_rows)
                .map(|row| self.forward(&matrix.slice(0, row, row + 1)?))
                .collect::<Result<Vec<_>>>()?;
            let mut shape = x.shape().to_vec();
            *shape.last_mut().context("empty EXL3 batch shape")? = self.cols;
            return Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0)?.reshape(&shape);
        }
        let fused = self.fused_glue(x.dtype());
        let (xhat, lane_order) = if fused {
            let lane = lane_order_selected(self.lane_indices.as_ref(), 1);
            (
                glue_prologue(x, &self.suh, 1, 1, self.rows, lane, 1)?.reshape(&[self.rows])?,
                lane,
            )
        } else {
            let xhat = x
                .astype(Dtype::Float16)?
                .reshape(&[1, self.rows])?
                .mul(&self.suh.astype(Dtype::Float16)?)?
                .reshape(&[1, self.rows / 128, 128])?
                .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
                .reshape(&[self.rows])?;
            lane_order_input(xhat, self.lane_indices.as_ref(), 1, 1, self.rows)?
        };
        let input_tiles = self.rows / 16;
        let output_tiles = self.cols / 16;
        let mut header = codebook_header(self.cb)
            + reference_reduction_header(self.reference_accumulation)
            + &format!("\n#define MLXL3_QMV_LANE_ORDER {}\n", u8::from(lane_order));
        let output = if self.k == 7 {
            let inverse = codec::permutation_inverse()
                .map(|v| v.to_string())
                .join(",");
            header += &format!(
                "\nconstant ushort mlxl3_perm_inv[256] = {{{inverse}}};\n#define K 7\n#define CB {}\n#define INPUT_DIMS {}\n#define TILES_N {}\n",
                self.cb as u32, self.rows, output_tiles
            );
            array::metal_kernel(
                &format!(
                    "mlxl3_rs_inner_{}_{}_{}",
                    self.rows, self.cols, self.cb as u32
                ),
                &["xhat", "trellis"],
                &["yhat"],
                &header,
                include_str!("../shaders/_qmv_inner_kernel.metal"),
                &[&xhat, &self.trellis],
                &[vec![self.cols]],
                &[Dtype::Float32],
                [
                    self.cols.checked_mul(128).context("QMV grid overflow")?,
                    1,
                    1,
                ],
                [128, 1, 1],
            )?
            .remove(0)
        } else {
            let splits = self.splits(input_tiles, output_tiles);
            let nt = if self.reference_accumulation {
                1
            } else if dense_decode_nt4_enabled()
                && crate::contracts::dense_decode_nt4(
                    self.rows,
                    self.cols,
                    self.k,
                    self.cb == Codebook::Mul1,
                    array::is_m5_gpu()?,
                )
            {
                4
            } else if output_tiles >= 1024 {
                if output_tiles % 2 == 0 { 2 } else { 1 }
            } else if output_tiles % 4 == 0 {
                4
            } else if output_tiles % 2 == 0 {
                2
            } else {
                1
            };
            header += &format!(
                "\n#define MLXL3_QMV_NT {nt}u\n#define MLXL3_QMV_MB 1u\n#define MLXL3_MATRIX_ROWS 1u\n#define MLXL3_QMV_SG {}u\n#define MLXL3_K_BITS {}u\n#define MLXL3_FUSE_OUTPUT 0\n#define K {}\n#define CB {}\n#define PACKED_U32 {}\n#define INPUT_DIMS {}\n#define TILES_K {input_tiles}\n#define TILES_N {output_tiles}\n#define N_SPLITS {splits}\n#define OUTPUT_DIMS {}\n",
                self.simdgroups,
                self.k,
                self.k,
                self.cb as u32,
                self.k * 8,
                self.rows,
                self.cols
            );
            let words = self.trellis.reshape(&[-1])?.view(Dtype::UInt32)?;
            if let Some(output) = self.try_dec_one(x, &xhat, &words, 1, &header)? {
                return Ok(output);
            }
            let partials = array::metal_kernel(
                &format!(
                    "mlxl3_rs_tile_{}_{}_{}_{}_{}_{}_{}",
                    self.rows, self.cols, self.k, self.cb as u32, nt, self.simdgroups, splits
                ),
                &["xhat", "trellis", "svh"],
                &["yhat"],
                &header,
                include_str!("../shaders/_qmv_tile_kernel.metal"),
                &[&xhat, &words, &self.svh],
                &[vec![splits, self.cols]],
                &[Dtype::Float32],
                [
                    (output_tiles / nt)
                        .checked_mul(self.simdgroups * 32)
                        .context("QMV grid overflow")?,
                    1,
                    splits,
                ],
                [self.simdgroups * 32, 1, 1],
            )?
            .remove(0);
            if let Some(output) = self.try_dense_epilogue(x, &partials, 1, splits)? {
                return Ok(output);
            }
            if fused {
                let mut shape = x.shape().to_vec();
                *shape.last_mut().context("empty input shape")? = self.cols;
                let y = glue_epilogue(
                    &partials.reshape(&[splits, 1, self.cols])?,
                    &self.svh,
                    splits,
                    1,
                    1,
                    self.cols,
                    x.dtype(),
                    false,
                )?
                .reshape(&shape)?;
                return match &self.bias {
                    Some(bias) => y.add(bias),
                    None => Ok(y),
                };
            }
            if splits == 1 {
                partials.reshape(&[self.cols])?
            } else {
                partials.sum(0, false)?
            }
        };
        let y = output
            .astype(Dtype::Float16)?
            .reshape(&[1, self.cols / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[self.cols])?
            .mul(&self.svh.astype(Dtype::Float16)?)?;
        let mut shape = x.shape().to_vec();
        *shape.last_mut().context("empty input shape")? = self.cols;
        let y = y.astype(x.dtype())?.reshape(&shape)?;
        match &self.bias {
            Some(bias) => y.add(bias),
            None => Ok(y),
        }
    }

    /// Q27D-33: exact QMV paths that may use the fused prologue/epilogue.
    fn fused_glue(&self, dtype: Dtype) -> bool {
        self.k != 7
            && !self.reference_accumulation
            && !self.mtp_hidden_batch
            && !self.flash_batch
            && qmv_fused_glue(dtype)
    }

    /// Q36-HEAD-LANE-P24: H2 (three rows) and F3 (four rows) read the head input in lane order.
    fn q36_head_lane_layout(&self, matrix_rows: i32, splits: i32) -> Option<(i32, i32, i32)> {
        #[cfg(test)]
        let active = Q36_HEAD_LANE_TEST
            .with(std::cell::Cell::get)
            .unwrap_or(true)
            && !q36_reference();
        #[cfg(not(test))]
        let active = true;
        let prepared = active
            && self.head_lane_indices.is_some()
            && !self.reference_accumulation
            && !self.mtp_hidden_batch;
        crate::contracts::q36_head_lane_layout(matrix_rows, prepared, splits, self.simdgroups)
    }

    fn forward_qmv_batch(&self, x: &Array, matrix_rows: i32) -> Result<Array> {
        ensure!(
            self.k != 7 && (2..24).contains(&matrix_rows),
            "batch QMV requires 2 to 23 rows"
        );
        let input_tiles = self.rows / 16;
        let output_tiles = self.cols / 16;
        // The released MTP FC uses Pony's four-row GEMM: two groups and two splits.
        let splits = if self.mtp_hidden_batch {
            2
        } else {
            self.splits(input_tiles, output_tiles)
        };
        let head_lane = self.q36_head_lane_layout(matrix_rows, splits);
        let fused = head_lane.is_none() && self.fused_glue(x.dtype());
        let (xhat, lane_order) = if fused {
            let lane = lane_order_selected(self.lane_indices.as_ref(), matrix_rows);
            let xhat = glue_prologue(x, &self.suh, matrix_rows, matrix_rows, self.rows, lane, 1)?;
            (xhat, lane)
        } else {
            let xhat = x
                .astype(Dtype::Float16)?
                .reshape(&[matrix_rows, self.rows])?
                .mul(&self.suh.astype(Dtype::Float16)?)?
                .reshape(&[matrix_rows, self.rows / 128, 128])?
                .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
                .reshape(&[matrix_rows, self.rows])?;
            lane_order_input(xhat, self.lane_indices.as_ref(), matrix_rows, 1, self.rows)?
        };
        let (xhat, lane_order) = match (head_lane, &self.head_lane_indices) {
            (Some(_), Some(indices)) => (xhat.take(indices, 1)?, true),
            _ => (xhat, lane_order),
        };
        // Q27D-04: the lane-ordered rows of one verify block share each decoded tile.
        let shared = lane_order
            && self.shared_rows_ready()
            && crate::contracts::dense_shared_rows(matrix_rows, true, dense_shared_enabled());
        let use_nt4 = matrix_rows >= 8 && output_tiles % 4 == 0 && array::is_m5_gpu()?;
        let mut nt = if use_nt4 {
            4
        } else if output_tiles >= 1024 {
            if output_tiles % 2 == 0 { 2 } else { 1 }
        } else if output_tiles % 4 == 0 {
            4
        } else if output_tiles % 2 == 0 {
            2
        } else {
            1
        };
        let (mb, batch_groups) = if self.mtp_hidden_batch {
            ensure!(
                matrix_rows == 4,
                "MTP hidden projection requires four streams"
            );
            nt = 1;
            (4, 1)
        } else if self.reference_accumulation {
            nt = 1;
            crate::contracts::flash_reference_mb34_layout(
                matrix_rows,
                self.reference_mb34
                    && self.simdgroups == 4
                    && crate::contracts::flash_reference_mb34_shape(self.rows, self.cols),
            )
            .context("unsupported Flash reference batch")?
        } else if let Some((mb, groups, head_nt)) = head_lane {
            nt = head_nt;
            (mb, groups)
        } else if shared {
            nt = 1;
            (matrix_rows, 1)
        } else {
            crate::contracts::qmv_batch_layout(matrix_rows, false, output_tiles >= 1024)
                .context("invalid batch QMV rows")?
        };
        if mb == 2 {
            nt = nt.min(2);
        }
        if mb == 3 {
            nt = 1;
        }
        #[cfg(test)]
        if head_lane.is_some() {
            Q36_HEAD_LANE_USES.with(|uses| uses.set(uses.get() + 1));
        }
        #[cfg(test)]
        if self.reference_accumulation && !self.mtp_hidden_batch && matches!(mb, 3 | 4) {
            REFERENCE_MB34_REQUESTS_TEST.with(|counts| {
                let mut next = counts.get();
                next[(mb - 3) as usize] += 1;
                counts.set(next);
            });
        }
        let header = codebook_header(self.cb)
            + reference_reduction_header(self.reference_accumulation)
            + &format!(
                "\n#define MLXL3_REFERENCE_MTP_BATCH {}\n",
                u8::from(self.mtp_hidden_batch)
            )
            + &format!("\n#define MLXL3_QMV_LANE_ORDER {}\n", u8::from(lane_order))
            + &format!(
                "\n#define MLXL3_QMV_NT {nt}u\n#define MLXL3_QMV_MB {mb}u\n#define MLXL3_MATRIX_ROWS {matrix_rows}u\n#define MLXL3_QMV_SG {}u\n#define MLXL3_K_BITS {}u\n#define MLXL3_FUSE_OUTPUT 0\n#define K {}\n#define CB {}\n#define PACKED_U32 {}\n#define INPUT_DIMS {input}\n#define TILES_K {input_tiles}\n#define TILES_N {output_tiles}\n#define N_SPLITS {splits}\n#define OUTPUT_DIMS {output}\n",
                self.simdgroups,
                self.k,
                self.k,
                self.cb as u32,
                self.k * 8,
                input = self.rows,
                output = self.cols,
            );
        let words = self.trellis.reshape(&[-1])?.view(Dtype::UInt32)?;
        if let Some(output) = self.try_dec_one(x, &xhat, &words, matrix_rows, &header)? {
            return Ok(output);
        }
        let partials = array::metal_kernel(
            &format!(
                "mlxl3_rs_tile_batch_{}_{}_{}_{}_{}_{}_{}_{}_{}{}",
                self.rows,
                self.cols,
                self.k,
                self.cb as u32,
                matrix_rows,
                nt,
                mb,
                self.simdgroups,
                splits,
                // MLX caches libraries by name; the lane header must not reuse a stock one.
                if head_lane.is_some() {
                    "_q36lane"
                } else if shared {
                    "_shared"
                } else {
                    ""
                }
            ),
            &["xhat", "trellis", "svh"],
            &["yhat"],
            &header,
            include_str!("../shaders/_qmv_tile_kernel.metal"),
            &[&xhat, &words, &self.svh],
            &[vec![matrix_rows, splits, self.cols]],
            &[Dtype::Float32],
            [
                (output_tiles / nt)
                    .checked_mul(self.simdgroups * 32)
                    .context("QMV grid overflow")?,
                batch_groups,
                splits,
            ],
            [self.simdgroups * 32, 1, 1],
        )?
        .remove(0);
        if let Some(output) = self.try_dense_epilogue(x, &partials, matrix_rows, splits)? {
            return Ok(output);
        }
        if fused {
            let mut shape = x.shape().to_vec();
            *shape.last_mut().context("empty input shape")? = self.cols;
            let y = glue_epilogue(
                &partials,
                &self.svh,
                splits,
                matrix_rows,
                matrix_rows,
                self.cols,
                x.dtype(),
                true,
            )?
            .reshape(&shape)?;
            return match &self.bias {
                Some(bias) => y.add(bias),
                None => Ok(y),
            };
        }
        let output = if splits == 1 {
            partials.reshape(&[matrix_rows, self.cols])?
        } else {
            partials.sum(1, false)?
        };
        let y = output
            .astype(Dtype::Float16)?
            .reshape(&[matrix_rows, self.cols / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[matrix_rows, self.cols])?
            .mul(&self.svh.astype(Dtype::Float16)?)?;
        let mut shape = x.shape().to_vec();
        *shape.last_mut().context("empty input shape")? = self.cols;
        let y = y.astype(x.dtype())?.reshape(&shape)?;
        match &self.bias {
            Some(bias) => y.add(bias),
            None => Ok(y),
        }
    }

    fn forward_qmm_tensor(&self, x: &Array) -> Result<Array> {
        let output = qmm_tensor(
            x,
            &self.trellis,
            &self.suh,
            &self.svh,
            self.rows,
            self.cols,
            self.k,
            self.cb,
            self.cols / 16,
            0,
        )?;
        match &self.bias {
            Some(bias) => output.add(bias),
            None => Ok(output),
        }
    }
}

/// Q27D-17 (not bit-exact): TensorOps product with the weights on the left, decoded four
/// codewords per trellis window in QMV lane order; activations permuted by `lane_indices`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn qmm_tensor_lane(
    x: &Array,
    trellis: &Array,
    suh: &Array,
    svh: &Array,
    lane_indices: &Array,
    input_dims: i32,
    output_dims: i32,
    k: usize,
    cb: Codebook,
    weight_tiles_n: i32,
    weight_tile_offset: i32,
) -> Result<Array> {
    ensure!(
        array::supports_exl3_tensor_ops()?,
        "TensorOps QMM requires Apple M5 and macOS 26.3 or newer"
    );
    ensure!(
        (1..=8).contains(&k)
            && k != 7
            && input_dims % 16 == 0
            && output_dims % 32 == 0
            && weight_tile_offset >= 0
            && weight_tile_offset + output_dims / 16 <= weight_tiles_n
            && lane_indices.shape() == [input_dims],
        "lane TensorOps QMM dimensions are not tiled"
    );
    let elements = x
        .shape()
        .iter()
        .try_fold(1i64, |count, &dimension| {
            count.checked_mul(i64::from(dimension))
        })
        .context("QMM input size overflow")?;
    ensure!(
        x.shape().last() == Some(&input_dims) && elements % i64::from(input_dims) == 0,
        "invalid lane TensorOps QMM input"
    );
    let matrix_rows = i32::try_from(elements / i64::from(input_dims))?;
    ensure!(matrix_rows >= 1, "TensorOps QMM requires rows");
    let tokens = tensor_lane_tokens(matrix_rows);
    let padded_rows = (matrix_rows + tokens - 1) / tokens * tokens;
    let fused = tensor_lane_fused();
    let xhat = if fused {
        lane_prologue_fused(x, suh, matrix_rows, padded_rows, input_dims)?
    } else {
        lane_prologue(x, suh, lane_indices, matrix_rows, padded_rows, input_dims)?
    };
    let (lane_cols, lane_depth) = tensor_lane_shape();
    let (lane_cols, lane_depth) = if output_dims % lane_cols == 0 && input_dims % lane_depth == 0 {
        (lane_cols, lane_depth)
    } else {
        (32, 16)
    };
    let groups = output_dims / lane_cols;
    let splits = match tensor_lane_target() {
        Some((target, limit)) => {
            crate::contracts::tensor_lane_splits_for(groups, input_dims / lane_depth, target, limit)
        }
        None => crate::contracts::tensor_lane_splits(groups, input_dims / lane_depth),
    };
    let buffers = tensor_lane_buffers();
    let header = format!(
        "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\nusing namespace mpp;\n{}#define K_BITS {k}u\n#define PACKED_U32 {}u\n#define INPUT_DIMS {input_dims}u\n#define OUTPUT_DIMS {output_dims}u\n#define TILES_N {weight_tiles_n}u\n#define WEIGHT_TILE_OFFSET {weight_tile_offset}u\n#define N_SPLITS {splits}u\n#define PADDED_ROWS {padded_rows}u\n#define LANE_COLS {lane_cols}u\n#define LANE_DEPTH {lane_depth}u\n#define LANE_TOKENS {tokens}u\n#define LANE_BUFFERS {buffers}\n",
        codebook_header(cb),
        k * 8,
    );
    let words = trellis.reshape(&[-1])?.view(Dtype::UInt32)?;
    let partials = array::metal_kernel(
        &format!(
            "mlxl3_rs_qmm_tensor_lane_{input_dims}_{output_dims}_{k}_{}_s{weight_tiles_n}_o{weight_tile_offset}_p{padded_rows}_k{splits}_c{lane_cols}_d{lane_depth}_t{tokens}{}{}",
            cb as u32,
            if buffers == 2 { "_b2" } else { "" },
            tensor_lane_source_tag()
        ),
        &["xhat", "trellis"],
        &["partials"],
        &header,
        &tensor_lane_source(),
        &[&xhat, &words],
        &[vec![splits, padded_rows, output_dims]],
        &[Dtype::Float32],
        [groups.checked_mul(32).context("QMM grid overflow")?, padded_rows / tokens, splits],
        [32, 1, 1],
    )?
    .remove(0);
    let output = if fused {
        lane_epilogue_fused(
            &partials,
            svh,
            splits,
            matrix_rows,
            padded_rows,
            output_dims,
            x.dtype(),
        )?
    } else {
        lane_epilogue(
            &partials,
            svh,
            splits,
            matrix_rows,
            padded_rows,
            output_dims,
        )?
        .astype(x.dtype())?
    };
    let mut shape = x.shape().to_vec();
    *shape.last_mut().context("empty QMM input shape")? = output_dims;
    output.reshape(&shape)
}

/// Q27D-31: fused lane prologue/epilogue (`MLXL3_TENSOR_LANE_FUSED=0` restores the MLX chains).
fn tensor_lane_fused() -> bool {
    #[cfg(test)]
    if let Some(value) = TENSOR_LANE_FUSED_TEST.with(std::cell::Cell::get) {
        return value;
    }
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| std::env::var("MLXL3_TENSOR_LANE_FUSED").as_deref() != Ok("0"))
}

#[cfg(test)]
thread_local! {
    static TENSOR_LANE_FUSED_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_tensor_lane_fused<T>(value: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TENSOR_LANE_FUSED_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(TENSOR_LANE_FUSED_TEST.with(|flag| flag.replace(Some(value))));
    run()
}

/// MLX 0.32.2 `hadamard_transform` of 128 F16 values held in threadgroup memory by eight
/// threads (radix 16 then radix 8, F32 butterflies, F16 between stages), as in
/// `flash_dense_epilogue.metal`. The caller applies the final scale.
const LANE_HADAMARD_HEADER: &str = r#"
inline void mlxl3_hadamard128_mlx(threadgroup half* buf, short i) {
    float x[16];
    #pragma clang loop unroll(full)
    for (short r = 0; r < 16; ++r) x[r] = float(buf[i * 16 + r]);
    short h = 1;
    #pragma clang loop unroll(full)
    for (short stage = 0; stage < 4; ++stage) {
        #pragma clang loop unroll(full)
        for (short pair = 0; pair < 8; ++pair) {
            short k = pair & (h - 1);
            short j = ((pair - k) << 1) + k;
            float a = x[j], b = x[j + h];
            x[j] = a + b;
            x[j + h] = a - b;
        }
        h <<= 1;
    }
    #pragma clang loop unroll(full)
    for (short r = 0; r < 16; ++r) buf[i * 16 + r] = half(x[r]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma clang loop unroll(full)
    for (short t = 0; t < 2; ++t) {
        short index = i + t * 8;
        #pragma clang loop unroll(full)
        for (short r = 0; r < 8; ++r) x[r] = float(buf[index + 16 * r]);
        h = 1;
        #pragma clang loop unroll(full)
        for (short stage = 0; stage < 3; ++stage) {
            #pragma clang loop unroll(full)
            for (short pair = 0; pair < 4; ++pair) {
                short k = pair & (h - 1);
                short j = ((pair - k) << 1) + k;
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
            h <<= 1;
        }
        #pragma clang loop unroll(full)
        for (short r = 0; r < 8; ++r) buf[index + 16 * r] = half(x[r]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
"#;

fn lane_metal_type(dtype: Dtype) -> Result<&'static str> {
    Ok(match dtype {
        Dtype::Float16 => "half",
        Dtype::BFloat16 => "bfloat16_t",
        Dtype::Float32 => "float",
        other => anyhow::bail!("lane TensorOps glue does not support {other:?}"),
    })
}

/// Q27D-31: one dispatch for `lane_prologue` (same values bit for bit).
pub(crate) fn lane_prologue_fused(
    x: &Array,
    suh: &Array,
    matrix_rows: i32,
    padded_rows: i32,
    input_dims: i32,
) -> Result<Array> {
    glue_prologue(x, suh, matrix_rows, padded_rows, input_dims, true, 1)
}

/// Q27D-31/33: fused `astype(F16) * suh -> Hadamard-128 -> [lane order] -> zero rows`.
/// With `groups` > 1, `suh` is `[groups, input]` and the output rows are `row * groups + g`
/// (the grouped QMV layout `[rows, groups, input]`).
pub(crate) fn glue_prologue(
    x: &Array,
    suh: &Array,
    matrix_rows: i32,
    padded_rows: i32,
    input_dims: i32,
    lane_order: bool,
    groups: i32,
) -> Result<Array> {
    let suh_elements = suh.shape().iter().map(|&d| i64::from(d)).product::<i64>();
    ensure!(
        input_dims % 128 == 0
            && matrix_rows >= 1
            && padded_rows >= matrix_rows
            && (1..=8).contains(&groups)
            && (groups == 1 || padded_rows == matrix_rows)
            && suh_elements == i64::from(groups) * i64::from(input_dims),
        "invalid fused prologue geometry"
    );
    lane_metal_type(x.dtype())?;
    let suh = if suh.dtype() == Dtype::Float16 {
        suh.try_clone()?
    } else {
        suh.astype(Dtype::Float16)?
    };
    let x = x.reshape(&[matrix_rows, input_dims])?;
    let header = format!(
        "{LANE_HADAMARD_HEADER}#define INPUT_DIMS {input_dims}u\n#define SCALE_BITS {}u\n#define LANE_ORDER {}\n#define GROUPS {groups}u\n",
        (1.0 / 128.0f32.sqrt()).to_bits(),
        u8::from(lane_order)
    );
    Ok(array::metal_kernel(
        &format!(
            "mlxl3_rs_glue_prologue_{input_dims}_{:?}_l{}_g{groups}",
            x.dtype(),
            u8::from(lane_order)
        ),
        &["x", "suh"],
        &["xhat"],
        &header,
        include_str!("../shaders/_lane_prologue.metal"),
        &[&x, &suh],
        &[vec![padded_rows * groups, input_dims]],
        &[Dtype::Float16],
        [
            (input_dims / 128)
                .checked_mul(8)
                .context("prologue grid overflow")?,
            padded_rows * groups,
            1,
        ],
        [8, 1, 1],
    )?
    .remove(0))
}

/// Q27D-31: one dispatch for `lane_epilogue` followed by the output conversion.
pub(crate) fn lane_epilogue_fused(
    partials: &Array,
    svh: &Array,
    splits: i32,
    matrix_rows: i32,
    padded_rows: i32,
    output_dims: i32,
    output: Dtype,
) -> Result<Array> {
    glue_epilogue(
        partials,
        svh,
        splits,
        matrix_rows,
        padded_rows,
        output_dims,
        output,
        false,
    )
}

/// Q27D-31/33: fused `sum(splits) -> F16 -> Hadamard-128 -> * svh -> astype(output)`; the
/// partials are `[splits, padded, out]` (lane kernel) or `[rows, splits, out]` (`row_major`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn glue_epilogue(
    partials: &Array,
    svh: &Array,
    splits: i32,
    matrix_rows: i32,
    padded_rows: i32,
    output_dims: i32,
    output: Dtype,
    row_major: bool,
) -> Result<Array> {
    let expected = if row_major {
        vec![matrix_rows, splits, output_dims]
    } else {
        vec![splits, padded_rows, output_dims]
    };
    ensure!(
        output_dims % 128 == 0
            && (1..=64).contains(&splits)
            && matrix_rows >= 1
            && padded_rows >= matrix_rows
            && partials.shape() == expected.as_slice()
            && (partials.dtype() == Dtype::Float32
                || (partials.dtype() == Dtype::Float16 && splits == 1)),
        "invalid fused epilogue geometry"
    );
    let out_type = lane_metal_type(output)?;
    let svh = if svh.dtype() == Dtype::Float16 {
        svh.try_clone()?
    } else {
        svh.astype(Dtype::Float16)?
    };
    let offset = if row_major {
        "#define SPLIT_AXIS 1\n#define PARTIAL_OFFSET(s, row) ((ulong(row) * splits + ulong(s)) * OUTPUT_DIMS)"
    } else {
        "#define SPLIT_AXIS 0\n#define PARTIAL_OFFSET(s, row) ((ulong(s) * padded_rows + ulong(row)) * OUTPUT_DIMS)"
    };
    let header = format!(
        "{LANE_HADAMARD_HEADER}#define OUTPUT_DIMS {output_dims}u\n#define OUT_T {out_type}\n#define SCALE_BITS {}u\n{offset}\n",
        (1.0 / 128.0f32.sqrt()).to_bits()
    );
    Ok(array::metal_kernel(
        &format!(
            "mlxl3_rs_glue_epilogue_{output_dims}_{out_type}_r{}{}",
            u8::from(row_major),
            if partials.dtype() == Dtype::Float16 {
                "_h"
            } else {
                ""
            }
        ),
        &["partials", "svh"],
        &["y"],
        &header,
        include_str!("../shaders/_lane_epilogue.metal"),
        &[partials, &svh],
        &[vec![matrix_rows, output_dims]],
        &[output],
        [
            (output_dims / 128)
                .checked_mul(8)
                .context("epilogue grid overflow")?,
            matrix_rows,
            1,
        ],
        [8, 1, 1],
    )?
    .remove(0))
}

/// Lane TensorOps prologue: F16, input signs, Hadamard-128, lane order, zero rows to the tile.
pub(crate) fn lane_prologue(
    x: &Array,
    suh: &Array,
    lane_indices: &Array,
    matrix_rows: i32,
    padded_rows: i32,
    input_dims: i32,
) -> Result<Array> {
    let xhat = x
        .astype(Dtype::Float16)?
        .reshape(&[matrix_rows, input_dims])?
        .mul(&suh.astype(Dtype::Float16)?)?
        .reshape(&[matrix_rows, input_dims / 128, 128])?
        .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
        .reshape(&[matrix_rows, input_dims])?
        .take(lane_indices, 1)?;
    if matrix_rows == padded_rows {
        Ok(xhat)
    } else {
        Array::concatenate(
            &[
                &xhat,
                &Array::zeros_dtype(&[padded_rows - matrix_rows, input_dims], Dtype::Float16)?,
            ],
            0,
        )
    }
}

/// Lane TensorOps epilogue: sum of the K splits, F16, Hadamard-128 and output signs (F16).
pub(crate) fn lane_epilogue(
    partials: &Array,
    svh: &Array,
    splits: i32,
    matrix_rows: i32,
    padded_rows: i32,
    output_dims: i32,
) -> Result<Array> {
    let raw = if splits == 1 {
        partials.reshape(&[padded_rows, output_dims])?
    } else {
        partials.sum(0, false)?
    }
    .slice(0, 0, matrix_rows)?;
    raw.astype(Dtype::Float16)?
        .reshape(&[matrix_rows, output_dims / 128, 128])?
        .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
        .reshape(&[matrix_rows, output_dims])?
        .mul(&svh.astype(Dtype::Float16)?)
}

/// Q27D-17 tuning: `MLXL3_TENSOR_LANE_SPLITS=target,limit` (simdgroups aimed at, split cap).
fn tensor_lane_target() -> Option<(i32, i32)> {
    #[cfg(test)]
    if let Some(value) = TENSOR_LANE_TARGET_TEST.with(std::cell::Cell::get) {
        return Some(value);
    }
    static VALUE: std::sync::OnceLock<Option<(i32, i32)>> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        let text = std::env::var("MLXL3_TENSOR_LANE_SPLITS").ok()?;
        let (target, limit) = text.split_once(',')?;
        let (target, limit) = (target.parse().ok()?, limit.parse().ok()?);
        ((1..=1 << 20).contains(&target) && (1..=64).contains(&limit)).then_some((target, limit))
    })
}

#[cfg(test)]
thread_local! {
    static TENSOR_LANE_TARGET_TEST: std::cell::Cell<Option<(i32, i32)>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_tensor_lane_target<T>(value: (i32, i32), run: impl FnOnce() -> T) -> T {
    struct Restore(Option<(i32, i32)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TENSOR_LANE_TARGET_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(TENSOR_LANE_TARGET_TEST.with(|flag| flag.replace(Some(value))));
    run()
}

/// Lane kernel source; tests may apply ablation edits (timing attribution only).
fn tensor_lane_source() -> String {
    let source = include_str!("../shaders/_qmm_tensor_lane_kernel.metal").to_owned();
    #[cfg(test)]
    {
        let edit = TENSOR_LANE_EDIT_TEST.with(|cell| cell.borrow().clone());
        if let Some(edit) = edit {
            return lane_ablation(&source, &edit);
        }
    }
    source
}

fn tensor_lane_source_tag() -> String {
    #[cfg(test)]
    if let Some(edit) = TENSOR_LANE_EDIT_TEST.with(|cell| cell.borrow().clone()) {
        return format!("_lab_{}", edit.replace('+', "_"));
    }
    String::new()
}

#[cfg(test)]
thread_local! {
    static TENSOR_LANE_EDIT_TEST: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_tensor_lane_edit<T>(edit: &str, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<String>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TENSOR_LANE_EDIT_TEST.with(|cell| *cell.borrow_mut() = self.0.take());
        }
    }
    let _restore =
        Restore(TENSOR_LANE_EDIT_TEST.with(|cell| cell.borrow_mut().replace(edit.to_owned())));
    run()
}

/// Ablations of the lane kernel: outputs are wrong by design.
#[cfg(test)]
fn lane_ablation(source: &str, edit: &str) -> String {
    let mut text = source.to_owned();
    for part in edit.split('+') {
        match part {
            "nodecode" => {
                for i in 0..4 {
                    text = text.replace(
                        &format!("left[base + {i}u] = half(mlxl3_decode_codeword(cw{i}, 0));"),
                        &format!("left[base + {i}u] = as_type<half>(ushort(cw{i} & 0x3bffu));"),
                    );
                }
            }
            "noload" => {
                text = text.replace(
                    "ulong merged = (ulong(words[word0[group]]) << 32) | ulong(words[word1[group]]);",
                    "ulong merged = ulong((tile_k * 2654435761u) ^ (lane * 40503u) ^ (ntile * 977u) ^ (group * 7919u)) << 13;",
                );
            }
            "norun" => {
                // Keep the decoded operand live through the accumulator so nothing is elided.
                text = text.replace(
                    "    operation.run(left, right, accumulator);",
                    "    for (ushort e = 0; e < accumulator.get_capacity(); ++e) if (accumulator.is_valid_element(e)) accumulator[e] += float(left[e % ELEMENTS]);",
                );
            }
            "norightload" => {
                text = text.replace(
                    "        activation + ulong(token_block * LANE_TOKENS) * INPUT_DIMS + step * LANE_DEPTH,",
                    "        activation + ulong(token_block * LANE_TOKENS) * INPUT_DIMS,",
                );
            }
            _ => {}
        }
    }
    text
}

/// Q27D-28: `MLXL3_TENSOR_LANE_BUFFERS=2` double-buffers the decoded left operand (same values).
fn tensor_lane_buffers() -> i32 {
    #[cfg(test)]
    if let Some(value) = TENSOR_LANE_BUFFERS_TEST.with(std::cell::Cell::get) {
        return value;
    }
    static VALUE: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *VALUE.get_or_init(
        || match std::env::var("MLXL3_TENSOR_LANE_BUFFERS").as_deref() {
            Ok("2") => 2,
            _ => 1,
        },
    )
}

#[cfg(test)]
thread_local! {
    static TENSOR_LANE_BUFFERS_TEST: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_tensor_lane_buffers<T>(value: i32, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<i32>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TENSOR_LANE_BUFFERS_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(TENSOR_LANE_BUFFERS_TEST.with(|flag| flag.replace(Some(value))));
    run()
}

/// Q27D-22: token tile of the lane kernel: `MLXL3_TENSOR_LANE_TOKENS` = 16 (default) or 8,
/// 8 only applies to blocks of at most eight rows.
fn tensor_lane_tokens(rows: i32) -> i32 {
    #[cfg(test)]
    if let Some(value) = TENSOR_LANE_TOKENS_TEST.with(std::cell::Cell::get) {
        return if rows <= 8 { value } else { 16 };
    }
    static VALUE: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    let tokens =
        *VALUE.get_or_init(
            || match std::env::var("MLXL3_TENSOR_LANE_TOKENS").as_deref() {
                Ok("8") => 8,
                _ => 16,
            },
        );
    if rows <= 8 { tokens } else { 16 }
}

#[cfg(test)]
thread_local! {
    static TENSOR_LANE_TOKENS_TEST: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_tensor_lane_tokens<T>(value: i32, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<i32>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TENSOR_LANE_TOKENS_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(TENSOR_LANE_TOKENS_TEST.with(|flag| flag.replace(Some(value))));
    run()
}

/// Q27D-17 tuning: `MLXL3_TENSOR_LANE_SHAPE` = `32x32` (default), `32x16` or `64x16`.
fn tensor_lane_shape() -> (i32, i32) {
    #[cfg(test)]
    if let Some(value) = TENSOR_LANE_SHAPE_TEST.with(std::cell::Cell::get) {
        return value;
    }
    static VALUE: std::sync::OnceLock<(i32, i32)> = std::sync::OnceLock::new();
    // Q27D-17 lab 8: 32 columns x 32 inputs per operation was the fastest shape.
    *VALUE.get_or_init(
        || match std::env::var("MLXL3_TENSOR_LANE_SHAPE").as_deref() {
            Ok("32x16") => (32, 16),
            Ok("64x16") => (64, 16),
            Ok("32x64") => (32, 64),
            Ok("64x32") => (64, 32),
            _ => (32, 32),
        },
    )
}

#[cfg(test)]
thread_local! {
    static TENSOR_LANE_SHAPE_TEST: std::cell::Cell<Option<(i32, i32)>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_tensor_lane_shape<T>(value: (i32, i32), run: impl FnOnce() -> T) -> T {
    struct Restore(Option<(i32, i32)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            TENSOR_LANE_SHAPE_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(TENSOR_LANE_SHAPE_TEST.with(|flag| flag.replace(Some(value))));
    run()
}

/// `MLXL3_DENSE_TENSOR_KERNEL`: `lane` (default) uses `qmm_tensor_lane` when lane indices exist.
fn tensor_lane_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MLXL3_DENSE_TENSOR_KERNEL").as_deref() != Ok("stock"))
}

#[allow(clippy::too_many_arguments)]
fn qmm_tensor(
    x: &Array,
    trellis: &Array,
    suh: &Array,
    svh: &Array,
    input_dims: i32,
    output_dims: i32,
    k: usize,
    cb: Codebook,
    weight_tiles_n: i32,
    weight_tile_offset: i32,
) -> Result<Array> {
    ensure!(
        array::supports_exl3_tensor_ops()?,
        "TensorOps QMM requires Apple M5 and macOS 26.3 or newer"
    );
    // Q27P-04 lab: tile shape of the shared prefill kernel (tests only).
    #[cfg(test)]
    let tile = PREFILL_TILE_TEST.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let tile: Option<(i32, i32)> = None;
    let block_columns = tile.map_or(32, |(_, columns)| columns);
    let block_depth = 16;
    ensure!(
        input_dims % block_depth == 0
            && output_dims % block_columns == 0
            && weight_tile_offset >= 0
            && weight_tile_offset + output_dims / 16 <= weight_tiles_n,
        "TensorOps QMM dimensions are not tiled"
    );
    let elements = x
        .shape()
        .iter()
        .try_fold(1i64, |count, &dimension| {
            count.checked_mul(i64::from(dimension))
        })
        .context("QMM input size overflow")?;
    ensure!(
        x.shape().last() == Some(&input_dims) && elements % i64::from(input_dims) == 0,
        "invalid TensorOps QMM input"
    );
    let matrix_rows = i32::try_from(elements / i64::from(input_dims))?;
    ensure!(matrix_rows >= 1, "TensorOps QMM requires rows");
    static DENSE_PREFILL_M64: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *DENSE_PREFILL_M64
        .get_or_init(|| std::env::var("MLXL3_DENSE_PREFILL_M64").as_deref() != Ok("0"));
    // Q27D-15: one 16-row tile below the production threshold (verification rows).
    let block_rows = if let Some((rows, _)) = tile {
        rows
    } else if matrix_rows <= 16 {
        16
    } else {
        crate::contracts::dense_qmm_block_rows(
            matrix_rows,
            input_dims,
            output_dims,
            enabled && cb == Codebook::Mul1,
        )
    };
    let padded_rows = ((matrix_rows + block_rows - 1) / block_rows) * block_rows;
    // Q27P-11: one dispatch for the prologue and one for the epilogue (same values).
    let fused = dense_prefill_fused()
        && matrix_rows > 16
        && input_dims % 128 == 0
        && output_dims % 128 == 0
        && matches!(x.dtype(), Dtype::Float16 | Dtype::BFloat16 | Dtype::Float32);
    let finish = |raw: Array| -> Result<Array> {
        let output = if fused {
            glue_epilogue(
                &raw.reshape(&[1, padded_rows, output_dims])?,
                svh,
                1,
                matrix_rows,
                padded_rows,
                output_dims,
                x.dtype(),
                false,
            )?
        } else {
            raw.slice(0, 0, matrix_rows)?
                .reshape(&[matrix_rows, output_dims / 128, 128])?
                .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
                .reshape(&[matrix_rows, output_dims])?
                .mul(&svh.astype(Dtype::Float16)?)?
                .astype(x.dtype())?
        };
        let mut shape = x.shape().to_vec();
        *shape.last_mut().context("empty QMM input shape")? = output_dims;
        output.reshape(&shape)
    };
    let xhat = if fused {
        glue_prologue(x, suh, matrix_rows, padded_rows, input_dims, false, 1)?
    } else {
        let xhat = x
            .astype(Dtype::Float16)?
            .reshape(&[matrix_rows, input_dims])?
            .mul(&suh.astype(Dtype::Float16)?)?
            .reshape(&[matrix_rows, input_dims / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[matrix_rows, input_dims])?;
        if matrix_rows == padded_rows {
            xhat
        } else {
            Array::concatenate(
                &[
                    &xhat,
                    &Array::zeros_dtype(&[padded_rows - matrix_rows, input_dims], Dtype::Float16)?,
                ],
                0,
            )?
        }
    };
    let inverse = codec::permutation_inverse()
        .map(|value| value.to_string())
        .join(",");
    let header = format!(
        "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\nusing namespace mpp;\n{}constant ushort mlxl3_perm_inv[256] = {{{inverse}}};\n#define BM {block_rows}u\n#define BN {block_columns}u\n#define BK {block_depth}u\n#define K_BITS {k}u\n#define PACKED_U32 {}u\n#define INPUT_DIMS {input_dims}u\n#define OUTPUT_DIMS {output_dims}u\n#define TILES_N {weight_tiles_n}u\n#define WEIGHT_TILE_OFFSET {weight_tile_offset}u\n",
        codebook_header(cb),
        k * 8,
    );
    let words = trellis.reshape(&[-1])?.view(Dtype::UInt32)?;
    let row_blocks = padded_rows / block_rows;
    if let Some(group) = dense_prefill_shared_group(row_blocks) {
        // Q27P-03 (opt-in, no gain): 32-deep operations.
        let depth = dense_prefill_depth(input_dims);
        // Q27P-07: window decode and a threadgroup-memory right operand (same operations).
        let window = depth == 16 && k != 7 && dense_prefill_window();
        let header = if depth == 32 {
            header.replace("#define BK 16u\n", "#define BK 32u\n")
        } else {
            header
        };
        let step_tiles = if window {
            dense_prefill_step_tiles()
        } else {
            1
        };
        #[cfg(test)]
        let skip_decode = window && PREFILL_SKIP_DECODE_TEST.with(std::cell::Cell::get);
        #[cfg(not(test))]
        let skip_decode = false;
        let header = format!(
            "{header}#define STEP_TILES {step_tiles}u\n{}",
            if skip_decode {
                "#define LAB_SKIP_DECODE 1\n"
            } else {
                ""
            }
        );
        let raw = array::metal_kernel(
            &format!(
                "mlxl3_rs_qmm_tensor_{}_{input_dims}_{output_dims}_{k}_{}_s{weight_tiles_n}_o{weight_tile_offset}_bm{block_rows}_bn{block_columns}_g{group}_d{depth}_t{step_tiles}",
                if skip_decode {
                    "window_nodecode"
                } else if window {
                    "window"
                } else {
                    "shared"
                },
                cb as u32
            ),
            &["xhat", "trellis"],
            &["yhat"],
            &format!("{header}#define ROW_GROUP {group}u\n"),
            if window {
                include_str!("../shaders/_qmm_tensor_window_kernel.metal")
            } else {
                include_str!("../shaders/_qmm_tensor_shared_kernel.metal")
            },
            &[&xhat, &words],
            &[vec![padded_rows, output_dims]],
            &[Dtype::Float16],
            [
                (output_dims / block_columns)
                    .checked_mul(32 * group)
                    .context("QMM grid overflow")?,
                (row_blocks + group - 1) / group,
                1,
            ],
            [32 * group, 1, 1],
        )?
        .remove(0);
        return finish(raw);
    }
    let raw = array::metal_kernel(
            &format!(
                "mlxl3_rs_qmm_tensor_{input_dims}_{output_dims}_{k}_{}_s{weight_tiles_n}_o{weight_tile_offset}_bm{block_rows}",
                cb as u32
            ),
            &["xhat", "trellis"],
            &["yhat"],
            &header,
            include_str!("../shaders/_qmm_tensor_kernel.metal"),
            &[&xhat, &words],
            &[vec![padded_rows, output_dims]],
            &[Dtype::Float16],
            [
                (output_dims / block_columns)
                    .checked_mul(32)
                    .context("QMM grid overflow")?,
                padded_rows / block_rows,
                1,
            ],
            [32, 1, 1],
        )?
        .remove(0);
    finish(raw)
}

/// Q27P-11: `MLXL3_DENSE_PREFILL_FUSED=1` fuses the prologue and the epilogue of the prefill
/// TensorOps product (same values bit for bit as the MLX chains).
fn dense_prefill_fused() -> bool {
    #[cfg(test)]
    if let Some(value) = DENSE_PREFILL_FUSED_TEST.with(std::cell::Cell::get) {
        return value;
    }
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| std::env::var("MLXL3_DENSE_PREFILL_FUSED").as_deref() == Ok("1"))
}

#[cfg(test)]
thread_local! {
    static DENSE_PREFILL_FUSED_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_dense_prefill_fused<T>(value: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DENSE_PREFILL_FUSED_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(DENSE_PREFILL_FUSED_TEST.with(|flag| flag.replace(Some(value))));
    run()
}

/// Q27P-02: up to `MLXL3_DENSE_PREFILL_SHARED` = 2, 4 (default) or 8 row blocks share each
/// decoded weight tile in the prefill TensorOps kernel (same operations per row block, so the
/// same values bit for bit); `0` keeps one row block per threadgroup.
fn dense_prefill_shared_group(row_blocks: i32) -> Option<i32> {
    let limit = {
        #[cfg(test)]
        let setting = DENSE_PREFILL_SHARED_TEST.with(std::cell::Cell::get);
        #[cfg(not(test))]
        let setting: Option<i32> = None;
        static VALUE: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
        match setting {
            Some(value) => value,
            None => *VALUE.get_or_init(|| {
                match std::env::var("MLXL3_DENSE_PREFILL_SHARED").as_deref() {
                    Ok("0") => 0,
                    Ok("2") => 2,
                    Ok("8") => 8,
                    Ok("16") => 16,
                    _ => 4,
                }
            }),
        }
    };
    (limit >= 2 && row_blocks >= 2).then(|| {
        let mut group = 2;
        while group < limit && group < row_blocks {
            group *= 2;
        }
        group
    })
}

#[cfg(test)]
pub(crate) fn with_prefill_tile<T>(value: (i32, i32), run: impl FnOnce() -> T) -> T {
    struct Restore(Option<(i32, i32)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PREFILL_TILE_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(PREFILL_TILE_TEST.with(|flag| flag.replace(Some(value))));
    run()
}

#[cfg(test)]
thread_local! {
    static PREFILL_TILE_TEST: std::cell::Cell<Option<(i32, i32)>> = const { std::cell::Cell::new(None) };
    static DENSE_PREFILL_SHARED_TEST: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
    static DENSE_PREFILL_DEPTH_TEST: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
}

/// Q27P-07: `MLXL3_DENSE_PREFILL_WINDOW=1` selects the window-decode prefill kernel whose right
/// operand is read from threadgroup memory (same operations as the shared kernel).
fn dense_prefill_window() -> bool {
    #[cfg(test)]
    if let Some(value) = DENSE_PREFILL_WINDOW_TEST.with(std::cell::Cell::get) {
        return value;
    }
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| std::env::var("MLXL3_DENSE_PREFILL_WINDOW").as_deref() == Ok("1"))
}

/// Q27P-07b: depth tiles decoded per threadgroup barrier in the window prefill kernel
/// (`MLXL3_DENSE_PREFILL_STEP_TILES` = 1, 2, 4 or 8; same operations in the same order).
fn dense_prefill_step_tiles() -> i32 {
    #[cfg(test)]
    if let Some(value) = DENSE_PREFILL_STEP_TEST.with(std::cell::Cell::get) {
        return value;
    }
    static VALUE: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *VALUE.get_or_init(
        || match std::env::var("MLXL3_DENSE_PREFILL_STEP_TILES").as_deref() {
            Ok("2") => 2,
            Ok("4") => 4,
            Ok("8") => 8,
            _ => 1,
        },
    )
}

#[cfg(test)]
pub(crate) fn with_dense_prefill_step_tiles<T>(value: i32, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<i32>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DENSE_PREFILL_STEP_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(DENSE_PREFILL_STEP_TEST.with(|flag| flag.replace(Some(value))));
    run()
}

#[cfg(test)]
pub(crate) fn with_prefill_skip_decode<T>(run: impl FnOnce() -> T) -> T {
    struct Restore;
    impl Drop for Restore {
        fn drop(&mut self) {
            PREFILL_SKIP_DECODE_TEST.with(|flag| flag.set(false));
        }
    }
    PREFILL_SKIP_DECODE_TEST.with(|flag| flag.set(true));
    let _restore = Restore;
    run()
}

#[cfg(test)]
thread_local! {
    static PREFILL_SKIP_DECODE_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static DENSE_PREFILL_STEP_TEST: std::cell::Cell<Option<i32>> = const { std::cell::Cell::new(None) };
    static DENSE_PREFILL_WINDOW_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

#[cfg(test)]
pub(crate) fn with_dense_prefill_window<T>(value: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DENSE_PREFILL_WINDOW_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(DENSE_PREFILL_WINDOW_TEST.with(|flag| flag.replace(Some(value))));
    run()
}

/// Q27P-03: `MLXL3_DENSE_PREFILL_DEPTH=32` uses 32-deep TensorOps operations in the shared
/// prefill kernel (not bit-exact: the unit's accumulation order changes). Default 16.
fn dense_prefill_depth(input_dims: i32) -> i32 {
    #[cfg(test)]
    let setting = DENSE_PREFILL_DEPTH_TEST.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let setting: Option<i32> = None;
    static VALUE: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    let depth = match setting {
        Some(value) => value,
        None => *VALUE.get_or_init(|| {
            if std::env::var("MLXL3_DENSE_PREFILL_DEPTH").as_deref() == Ok("32") {
                32
            } else {
                16
            }
        }),
    };
    if depth == 32 && input_dims % 32 == 0 {
        32
    } else {
        16
    }
}

#[cfg(test)]
pub(crate) fn with_dense_prefill_depth<T>(value: i32, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<i32>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DENSE_PREFILL_DEPTH_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(DENSE_PREFILL_DEPTH_TEST.with(|flag| flag.replace(Some(value))));
    run()
}

#[cfg(test)]
pub(crate) fn with_dense_prefill_shared<T>(value: i32, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<i32>);
    impl Drop for Restore {
        fn drop(&mut self) {
            DENSE_PREFILL_SHARED_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(DENSE_PREFILL_SHARED_TEST.with(|flag| flag.replace(Some(value))));
    run()
}

/// Ragged projections with independent input scales, packed once at load time.
pub struct Exl3Group {
    trellis: Array,
    suh: Array,
    svh: Array,
    tile_sub: Array,
    identity: Array,
    widths: Vec<i32>,
    biases: Vec<Option<Array>>,
    rows: i32,
    cols: i32,
    k: usize,
    cb: Codebook,
    simdgroups: i32,
    lane_indices: Option<Array>,
    lfm3_prefill: bool,
}

impl Exl3Group {
    pub fn compatible(linears: &[&Exl3Linear]) -> bool {
        linears.first().is_some_and(|first| {
            linears.len() >= 2
                && first.k != 7
                && linears
                    .iter()
                    .all(|l| l.rows == first.rows && l.k == first.k && l.cb == first.cb)
        })
    }

    pub fn new(linears: Vec<Exl3Linear>) -> Result<Self> {
        ensure!(
            Self::compatible(&linears.iter().collect::<Vec<_>>()),
            "incompatible grouped projections"
        );
        let first = &linears[0];
        let rows = first.rows;
        let k = first.k;
        let cb = first.cb;
        let simdgroups = if first.simdgroups == 8 && k == 2 {
            4
        } else {
            first.simdgroups
        };
        let widths: Vec<_> = linears.iter().map(|l| l.cols).collect();
        let cols = widths
            .iter()
            .try_fold(0i32, |sum, n| sum.checked_add(*n))
            .context("group width overflow")?;
        let trellis =
            Array::concatenate(&linears.iter().map(|l| &l.trellis).collect::<Vec<_>>(), 1)?;
        let scales = linears
            .iter()
            .map(|l| l.suh.reshape(&[1, rows]))
            .collect::<Result<Vec<_>>>()?;
        let suh = Array::concatenate(&scales.iter().collect::<Vec<_>>(), 0)?;
        let svh = Array::concatenate(&linears.iter().map(|l| &l.svh).collect::<Vec<_>>(), 0)?;
        let indices: Vec<u8> = widths
            .iter()
            .enumerate()
            .flat_map(|(index, width)| std::iter::repeat_n(index as u32, (width / 16) as usize))
            .flat_map(u32::to_ne_bytes)
            .collect();
        let tile_sub = Array::from_bytes(&indices, &[cols / 16], Dtype::UInt32)?;
        let identity = Array::from_bytes(&0u32.to_ne_bytes(), &[1], Dtype::UInt32)?;
        for array in [&trellis, &suh, &svh, &tile_sub] {
            array.eval()?;
        }
        let lfm3_prefill = linears.iter().all(|v| v.lfm3_prefill[1]);
        let biases = linears.into_iter().map(|l| l.bias).collect();
        let lane_indices =
            if widths.as_slice() == [10240, 6144] || widths.as_slice() == [17408, 17408] {
                lane_order_indices(rows, cols, k, cb, true)?
            } else {
                None
            };
        Ok(Self {
            trellis,
            suh,
            svh,
            tile_sub,
            identity,
            widths,
            biases,
            rows,
            cols,
            k,
            cb,
            simdgroups,
            lane_indices,
            lfm3_prefill,
        })
    }

    /// Lane-ordered stock group: eligible for the shared-decode verify layouts.
    pub(crate) fn shared_rows_ready(&self) -> bool {
        self.lane_indices.is_some() && lane_order_enabled()
    }

    pub fn forward(&self, x: &Array) -> Result<Vec<Array>> {
        let elements = x
            .shape()
            .iter()
            .try_fold(1i64, |count, &n| count.checked_mul(i64::from(n)))
            .context("grouped EXL3 input size overflow")?;
        ensure!(
            x.shape().last() == Some(&self.rows) && elements % i64::from(self.rows) == 0,
            "invalid grouped EXL3 input"
        );
        let matrix_rows = i32::try_from(elements / i64::from(self.rows))?;
        ensure!(matrix_rows > 0, "grouped EXL3 input cannot be empty");
        if crate::contracts::use_tensor_ops(matrix_rows, array::supports_exl3_tensor_ops()?)
            || crate::contracts::dense_tensor_rows(
                matrix_rows,
                self.shared_rows_ready(),
                dense_tensor_rows_threshold()?,
            )
        {
            if self.lfm3_prefill
                && matrix_rows <= 256
                && x.shape() == [1, matrix_rows, self.rows]
                && x.dtype() == Dtype::Float16
                && crate::lfm3::grouped_prefill_layout(matrix_rows, self.rows, &self.widths)
                    .is_some()
            {
                return lfm3_prefill::grouped_qmm(self, x);
            }
            let mut tile_cursor = 0;
            let mut scale_cursor = 0;
            let mut outputs = Vec::with_capacity(self.widths.len());
            let lane = self
                .lane_indices
                .as_ref()
                .filter(|_| matrix_rows < 24 && tensor_lane_enabled());
            for (index, (width, bias)) in self.widths.iter().zip(&self.biases).enumerate() {
                let suh = self
                    .suh
                    .slice(0, index as i32, index as i32 + 1)?
                    .reshape(&[self.rows])?;
                let svh = self.svh.slice(0, scale_cursor, scale_cursor + width)?;
                if let Some(indices) = lane {
                    let output = qmm_tensor_lane(
                        x,
                        &self.trellis,
                        &suh,
                        &svh,
                        indices,
                        self.rows,
                        *width,
                        self.k,
                        self.cb,
                        self.cols / 16,
                        tile_cursor,
                    )?;
                    outputs.push(match bias {
                        Some(value) => output.add(value)?,
                        None => output,
                    });
                    tile_cursor += width / 16;
                    scale_cursor += width;
                    continue;
                }
                let output = qmm_tensor(
                    x,
                    &self.trellis,
                    &self
                        .suh
                        .slice(0, index as i32, index as i32 + 1)?
                        .reshape(&[self.rows])?,
                    &self.svh.slice(0, scale_cursor, scale_cursor + width)?,
                    self.rows,
                    *width,
                    self.k,
                    self.cb,
                    self.cols / 16,
                    tile_cursor,
                )?;
                outputs.push(match bias {
                    Some(value) => output.add(value)?,
                    None => output,
                });
                tile_cursor += width / 16;
                scale_cursor += width;
            }
            return Ok(outputs);
        }
        if matrix_rows >= 24 {
            let matrix = x.reshape(&[matrix_rows, self.rows])?;
            let mut batches: Vec<Vec<Array>> = self.widths.iter().map(|_| Vec::new()).collect();
            for start in (0..matrix_rows).step_by(23) {
                for (batch, value) in batches.iter_mut().zip(self.forward(&matrix.slice(
                    0,
                    start,
                    start.saturating_add(23).min(matrix_rows),
                )?)?) {
                    batch.push(value);
                }
            }
            return batches
                .into_iter()
                .zip(&self.widths)
                .map(|(batch, &width)| {
                    let mut shape = x.shape().to_vec();
                    *shape.last_mut().context("empty grouped input shape")? = width;
                    Array::concatenate(&batch.iter().collect::<Vec<_>>(), 0)?.reshape(&shape)
                })
                .collect();
        }
        if let Some((first, _)) = crate::contracts::dense_shared_chunks(
            matrix_rows,
            self.shared_rows_ready(),
            dense_shared_enabled(),
        ) {
            let matrix = x.reshape(&[matrix_rows, self.rows])?;
            let head = self.forward_qmv_batch(&matrix.slice(0, 0, first)?, first)?;
            let tail =
                self.forward_qmv_batch(&matrix.slice(0, first, matrix_rows)?, matrix_rows - first)?;
            return head
                .iter()
                .zip(&tail)
                .zip(&self.widths)
                .map(|((head, tail), &width)| {
                    let mut shape = x.shape().to_vec();
                    *shape.last_mut().context("empty grouped input shape")? = width;
                    Array::concatenate(&[head, tail], 0)?.reshape(&shape)
                })
                .collect();
        }
        if matrix_rows > 1 {
            return self.forward_qmv_batch(x, matrix_rows);
        }
        let groups = i32::try_from(self.widths.len())?;
        let fused = qmv_fused_glue(x.dtype()) && groups <= 8;
        let (xhat, lane_order) = if fused {
            let lane = lane_order_selected(self.lane_indices.as_ref(), 1);
            (
                glue_prologue(x, &self.suh, 1, 1, self.rows, lane, groups)?.reshape(&[-1])?,
                lane,
            )
        } else {
            let xhat = x
                .astype(Dtype::Float16)?
                .reshape(&[1, self.rows])?
                .mul(&self.suh.astype(Dtype::Float16)?)?
                .reshape(&[groups, self.rows / 128, 128])?
                .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
                .reshape(&[-1])?;
            lane_order_input(xhat, self.lane_indices.as_ref(), 1, groups, self.rows)?
        };
        let tiles = self.cols / 16;
        let splits = split_count(self.rows / 16, tiles);
        let nt = if dense_decode_nt4_enabled()
            && crate::contracts::dense_decode_nt4(
                self.rows,
                self.cols,
                self.k,
                self.cb == Codebook::Mul1,
                array::is_m5_gpu()?,
            ) {
            4
        } else if tiles >= 1024 {
            if tiles % 2 == 0 { 2 } else { 1 }
        } else if tiles % 4 == 0 {
            4
        } else if tiles % 2 == 0 {
            2
        } else {
            1
        };
        let header = codebook_header(self.cb)
            + &format!("\n#define MLXL3_QMV_LANE_ORDER {}\n", u8::from(lane_order))
            + &format!(
                "\n#define MLXL3_QMV_NT {nt}u\n#define MLXL3_QMV_SG {sg}u\n#define MLXL3_K_BITS {k}u\n#define MLXL3_K3_WINDOW_DECODE 0\n#define K {k}\n#define CB {cb}\n#define PACKED_U32 {words}\n#define INPUT_DIMS {rows}\n#define TILES_K {kt}\n#define TILES_N {tiles}\n#define N_SPLITS {splits}\n#define LOCAL_OUTPUT_DIMS {cols}\n#define IDENTITY_MAP 1\n#define EXPERT_MAP 0\n#define OUTPUT_TILES {tiles}\n#define ROUTING_REPEAT 1\n#define PROJECTION_STRIDE_TILES 0\n",
                sg = self.simdgroups,
                k = self.k,
                cb = self.cb as u32,
                words = self.k * 8,
                rows = self.rows,
                kt = self.rows / 16,
                cols = self.cols
            );
        let words = self.trellis.reshape(&[-1])?.view(Dtype::UInt32)?;
        let partials = array::metal_kernel(
            "mlxl3_rs_grouped",
            &["xhat", "trellis", "tile_map", "tile_sub"],
            &["yhat"],
            &header,
            include_str!("../shaders/_qmv_mapped_tile_kernel.metal"),
            &[&xhat, &words, &self.identity, &self.tile_sub],
            &[vec![splits, self.cols]],
            &[Dtype::Float32],
            [
                (tiles / nt)
                    .checked_mul(self.simdgroups * 32)
                    .context("grouped grid overflow")?,
                1,
                splits,
            ],
            [self.simdgroups * 32, 1, 1],
        )?
        .remove(0);
        let output = if fused {
            glue_epilogue(
                &partials.reshape(&[splits, 1, self.cols])?,
                &self.svh,
                splits,
                1,
                1,
                self.cols,
                x.dtype(),
                false,
            )?
            .reshape(&[self.cols])?
        } else {
            let yhat = if splits == 1 {
                partials.reshape(&[self.cols])?
            } else {
                partials.sum(0, false)?
            };
            yhat.astype(Dtype::Float16)?
                .reshape(&[1, self.cols / 128, 128])?
                .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
                .reshape(&[self.cols])?
                .mul(&self.svh.astype(Dtype::Float16)?)?
                .astype(x.dtype())?
        };
        let mut begin = 0;
        let mut outputs = Vec::new();
        for (width, bias) in self.widths.iter().zip(&self.biases) {
            let mut shape = x.shape().to_vec();
            *shape.last_mut().context("empty grouped input")? = *width;
            let y = output.slice(0, begin, begin + width)?.reshape(&shape)?;
            outputs.push(match bias {
                Some(b) => y.add(b)?,
                None => y,
            });
            begin += width;
        }
        Ok(outputs)
    }

    // The four-row 35B verifier reads half4 values in SIMD lane order.
    fn small_batch_input(&self, x: &Array, matrix_rows: i32, groups: i32) -> Result<(Array, bool)> {
        let qualified = crate::contracts::q36_qkv_lane(
            matrix_rows,
            self.rows,
            self.k,
            self.cb == Codebook::Mcg,
            self.widths.as_slice(),
        );
        #[cfg(test)]
        let qualified = qualified && !q36_reference();
        if qualified && array::is_m5_gpu()? {
            let out = array::metal_kernel(
                "mlxl3_rs_q36_group_input_lane",
                &["x", "scales"],
                &["out"],
                "",
                include_str!("../shaders/q36_group_input_lane.metal"),
                &[
                    &x.astype(Dtype::Float16)?.reshape(&[matrix_rows, 2048])?,
                    &self.suh.astype(Dtype::Float16)?,
                ],
                &[vec![matrix_rows * groups * 2048]],
                &[Dtype::Float16],
                [512, matrix_rows, 2],
                [32, 1, 1],
            )?
            .remove(0);
            return Ok((out, true));
        }
        if qmv_fused_glue(x.dtype()) && groups <= 8 {
            let lane = lane_order_selected(self.lane_indices.as_ref(), matrix_rows);
            let xhat = glue_prologue(
                x,
                &self.suh,
                matrix_rows,
                matrix_rows,
                self.rows,
                lane,
                groups,
            )?
            .reshape(&[-1])?;
            return Ok((xhat, lane));
        }
        let xhat = x
            .astype(Dtype::Float16)?
            .reshape(&[matrix_rows, 1, self.rows])?
            .mul(
                &self
                    .suh
                    .astype(Dtype::Float16)?
                    .reshape(&[1, groups, self.rows])?,
            )?
            .reshape(&[matrix_rows * groups, self.rows / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[-1])?;
        lane_order_input(
            xhat,
            self.lane_indices.as_ref(),
            matrix_rows,
            groups,
            self.rows,
        )
    }

    fn forward_qmv_batch(&self, x: &Array, matrix_rows: i32) -> Result<Vec<Array>> {
        ensure!(
            (2..24).contains(&matrix_rows),
            "grouped batch QMV requires 2 to 23 rows"
        );
        let groups = i32::try_from(self.widths.len())?;
        let (xhat, lane_order) = self.small_batch_input(x, matrix_rows, groups)?;
        let tiles = self.cols / 16;
        let splits = split_count(self.rows / 16, tiles);
        let mut nt = if tiles >= 1024 {
            if tiles % 2 == 0 { 2 } else { 1 }
        } else if tiles % 4 == 0 {
            4
        } else if tiles % 2 == 0 {
            2
        } else {
            1
        };
        let (mb, batch_groups) = crate::contracts::qmv_grouped_batch_layout(
            matrix_rows,
            self.rows,
            self.cols,
            self.k,
            self.cb == Codebook::Mul1,
            array::is_m5_gpu()?,
            grouped_mb3_enabled(),
        )
        .context("invalid grouped batch QMV rows")?;
        // Q27D-04: the lane-ordered rows of one verify block share each decoded tile.
        let shared = lane_order
            && self.shared_rows_ready()
            && crate::contracts::dense_shared_rows(matrix_rows, true, dense_shared_enabled());
        let (mb, batch_groups) = if shared {
            (matrix_rows, 1)
        } else {
            (mb, batch_groups)
        };
        if mb == 2 {
            nt = nt.min(2);
        }
        if mb >= 3 {
            nt = 1;
        }
        let header = codebook_header(self.cb)
            + &format!("\n#define MLXL3_QMV_LANE_ORDER {}\n", u8::from(lane_order))
            + &format!(
                "\n#define MLXL3_QMV_NT {nt}u\n#define MLXL3_QMV_MB {mb}u\n#define MLXL3_MATRIX_ROWS {matrix_rows}u\n#define MLXL3_QMV_SG {sg}u\n#define MLXL3_K_BITS {k}u\n#define MLXL3_K3_WINDOW_DECODE 0\n#define MLXL3_BATCH_ROWS 1\n#define GROUPS {groups}\n#define K {k}\n#define CB {cb}\n#define PACKED_U32 {words}\n#define INPUT_DIMS {rows}\n#define TILES_K {kt}\n#define TILES_N {tiles}\n#define N_SPLITS {splits}\n#define LOCAL_OUTPUT_DIMS {cols}\n#define IDENTITY_MAP 1\n#define EXPERT_MAP 0\n#define OUTPUT_TILES {tiles}\n#define ROUTING_REPEAT 1\n#define PROJECTION_STRIDE_TILES 0\n",
                sg = self.simdgroups,
                k = self.k,
                cb = self.cb as u32,
                words = self.k * 8,
                rows = self.rows,
                kt = self.rows / 16,
                cols = self.cols,
            );
        let words = self.trellis.reshape(&[-1])?.view(Dtype::UInt32)?;
        let partials = array::metal_kernel(
            &format!(
                "mlxl3_rs_grouped_batch_{}_{}_{}_{}_{}_{}_{}_{}{}",
                self.rows,
                self.cols,
                self.k,
                self.cb as u32,
                nt,
                mb,
                self.simdgroups,
                splits,
                // MLX caches libraries by name; the lane header must not reuse a stock one.
                if shared { "_shared" } else { "" },
            ),
            &["xhat", "trellis", "tile_map", "tile_sub"],
            &["yhat"],
            &header,
            include_str!("../shaders/_qmv_mapped_tile_kernel.metal"),
            &[&xhat, &words, &self.identity, &self.tile_sub],
            &[vec![matrix_rows, splits, self.cols]],
            &[Dtype::Float32],
            [
                (tiles / nt)
                    .checked_mul(self.simdgroups * 32)
                    .context("grouped batch grid overflow")?,
                batch_groups,
                splits,
            ],
            [self.simdgroups * 32, 1, 1],
        )?
        .remove(0);
        let output = if qmv_fused_glue(x.dtype()) && groups <= 8 {
            glue_epilogue(
                &partials,
                &self.svh,
                splits,
                matrix_rows,
                matrix_rows,
                self.cols,
                x.dtype(),
                true,
            )?
        } else {
            let yhat = if splits == 1 {
                partials.reshape(&[matrix_rows, self.cols])?
            } else {
                partials.sum(1, false)?
            };
            yhat.astype(Dtype::Float16)?
                .reshape(&[matrix_rows, self.cols / 128, 128])?
                .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
                .reshape(&[matrix_rows, self.cols])?
                .mul(&self.svh.astype(Dtype::Float16)?)?
                .astype(x.dtype())?
        };
        let mut begin = 0;
        let mut outputs = Vec::with_capacity(self.widths.len());
        for (width, bias) in self.widths.iter().zip(&self.biases) {
            let mut shape = x.shape().to_vec();
            *shape.last_mut().context("empty grouped batch input")? = *width;
            let y = output.slice(1, begin, begin + width)?.reshape(&shape)?;
            outputs.push(match bias {
                Some(value) => y.add(value)?,
                None => y,
            });
            begin += width;
        }
        Ok(outputs)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn expert_mapped(
    x: &Array,
    trellis: &Array,
    suh: Option<&Array>,
    svh: Option<&Array>,
    selected: &Array,
    output_dims: i32,
    projections_per_route: i32,
    projection_stride_tiles: i32,
    k: usize,
    cb: Codebook,
    input_pretransformed: bool,
    return_raw: bool,
) -> Result<Array> {
    expert_mapped_layout(
        x,
        trellis,
        suh,
        svh,
        selected,
        output_dims,
        projections_per_route,
        projection_stride_tiles,
        k,
        cb,
        input_pretransformed,
        return_raw,
        false,
        false,
        None,
        false,
        false,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn prepared_expert_raw(
    x: &Array,
    bank: &Array,
    routes: &Array,
    width: i32,
    projections: i32,
    projection_stride: i32,
    k: usize,
    cb: Codebook,
    lane_order: bool,
) -> Result<Array> {
    expert_mapped_layout(
        x,
        bank,
        None,
        None,
        routes,
        width,
        projections,
        projection_stride,
        k,
        cb,
        true,
        true,
        false,
        false,
        None,
        false,
        false,
        lane_order,
    )
}

struct GroupedRoutes {
    origins: Vec<(i32, Array)>,
    restore: Vec<u32>,
}

/// CPU-origin indices validated once, before any expert graph is submitted.
pub(crate) struct ExpertSlots {
    indices: Array,
    capacity: i32,
    masked: bool,
    groups: Option<GroupedRoutes>,
}

impl ExpertSlots {
    pub(crate) fn new(ids: &[u32], capacity: i32) -> Result<Self> {
        ensure!(
            crate::flash_support::slots_fit(ids, capacity),
            "expert slot outside bank"
        );
        Ok(Self {
            indices: Array::from_u32(ids, &[i32::try_from(ids.len())?])?,
            capacity,
            masked: false,
            groups: None,
        })
    }

    pub(crate) fn grouped(ids: &[u32], capacity: i32) -> Result<Self> {
        ensure!(
            ids.len() <= 80 && capacity <= 512,
            "invalid grouped expert slots"
        );
        let mut slots = Self::new(ids, capacity)?;
        if ids.len() < 16 {
            return Ok(slots);
        }
        let chunks = crate::flash_support::expert_group_chunks(ids)
            .context("invalid grouped expert route count")?;
        if !crate::flash_support::expert_grouping_worthwhile(ids.len(), chunks) {
            return Ok(slots);
        }
        let groups = crate::flash_support::expert_route_groups(ids, capacity)
            .context("invalid grouped expert slots")?;
        let mut origins = Vec::new();
        let mut restore = vec![0; ids.len()];
        let mut offset = 0;
        for (width, rows) in groups.iter().enumerate() {
            if rows.is_empty() {
                continue;
            }
            for &row in rows {
                restore[row as usize] = offset;
                offset += 1;
            }
            origins.push((
                width as i32 + 1,
                Array::from_u32(rows, &[rows.len() as i32])?,
            ));
        }
        slots.groups = Some(GroupedRoutes { origins, restore });
        Ok(slots)
    }

    pub(crate) fn indices(&self) -> &Array {
        &self.indices
    }

    #[cfg(test)]
    pub(crate) fn shares_weights(&self) -> bool {
        self.groups.is_some()
    }

    pub(crate) fn residents(selected: &Array, map: &[u32; 512], capacity: i32) -> Result<Self> {
        let rows = selected
            .shape()
            .iter()
            .try_fold(1i32, |n, &s| n.checked_mul(s))
            .context("resident route shape overflow")?;
        ensure!(selected.dtype() == Dtype::UInt32 && (1..=80).contains(&rows));
        ensure!(
            (1..=512).contains(&capacity)
                && map.iter().all(|&s| s == u32::MAX || s < capacity as u32),
            "invalid resident slot map"
        );
        let indices = array::metal_kernel(
            "mlxl3_resident_slots", &["ids", "slot_map"], &["slots"], "",
            "uint i=thread_position_in_grid.x; if(i>=uint(ids_shape[0])) return; uint id=ids[i]; slots[i]=id<512u ? slot_map[id] : 0xffffffffu;",
            &[&selected.reshape(&[-1])?, &Array::from_u32(map, &[512])?],
            &[vec![rows]], &[Dtype::UInt32], [rows,1,1], [32,1,1],
        )?.remove(0);
        Ok(Self {
            indices,
            capacity,
            masked: true,
            groups: None,
        })
    }

    pub(crate) fn gather_scales(&self, bank: &Array) -> Result<Array> {
        ensure!(
            bank.shape().len() == 3
                && bank.shape()[0] == self.capacity
                && bank.dtype() == Dtype::Float16,
            "invalid expert scale bank"
        );
        if !self.masked {
            return bank.take(&self.indices, 0);
        }
        let shape = bank.shape();
        let width = shape[1]
            .checked_mul(shape[2])
            .filter(|&n| n > 0)
            .context("invalid expert scale width")?;
        let count = self.indices.shape()[0]
            .checked_mul(width)
            .context("expert scale grid overflow")?;
        Ok(array::metal_kernel(
            "mlxl3_resident_scales", &["bank", "slots"], &["out"],
            &format!("#define WIDTH {width}u\n#define CAPACITY {}u\n", self.capacity),
            "uint i=thread_position_in_grid.x; if(i>=uint(slots_shape[0])*WIDTH) return; uint slot=slots[i/WIDTH]; if(slot<CAPACITY) out[i]=bank[slot*WIDTH+i%WIDTH]; else out[i]=half(0);",
            &[&bank.reshape(&[-1])?, &self.indices],
            &[vec![self.indices.shape()[0],shape[1],shape[2]]], &[Dtype::Float16], [count,1,1], [128,1,1],
        )?.remove(0))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn expert_slots(
    x: &Array,
    trellis: &Array,
    suh: &Array,
    svh: &Array,
    selected: &ExpertSlots,
    output_dims: i32,
    projections_per_route: i32,
    k: usize,
    cb: Codebook,
    stable_shapes: bool,
    native_prepare: bool,
) -> Result<Array> {
    ensure!(
        trellis.shape().len() == 5 && trellis.shape()[0] == selected.capacity,
        "expert slots were validated for a different bank capacity"
    );
    expert_mapped_layout(
        x,
        trellis,
        Some(suh),
        Some(svh),
        selected.indices(),
        output_dims,
        projections_per_route,
        0,
        k,
        cb,
        false,
        false,
        true,
        selected.masked,
        selected.groups.as_ref(),
        stable_shapes,
        native_prepare,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn expert_mapped_layout(
    x: &Array,
    trellis: &Array,
    suh: Option<&Array>,
    svh: Option<&Array>,
    selected: &Array,
    output_dims: i32,
    projections_per_route: i32,
    projection_stride_tiles: i32,
    k: usize,
    cb: Codebook,
    input_pretransformed: bool,
    return_raw: bool,
    slot_major: bool,
    masked: bool,
    groups: Option<&GroupedRoutes>,
    stable_shapes: bool,
    native_prepare: bool,
    lane_order: bool,
) -> Result<Array> {
    codec::check_k(k)?;
    ensure!(k != 7, "mapped K=7 is not supported");
    let [rows, input_dims]: [i32; 2] = x
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("expert QMV input must have rank 2"))?;
    let shape = trellis.shape();
    let valid_trellis = if slot_major {
        shape.len() == 5
            && shape[0] > 0
            && shape[1] == projections_per_route
            && shape[2].checked_mul(16) == Some(input_dims)
            && shape[3].checked_mul(16) == Some(output_dims)
            && shape[4] == (16 * k) as i32
    } else {
        shape.len() == 3
            && shape[0].checked_mul(16) == Some(input_dims)
            && shape[2] == (16 * k) as i32
    };
    ensure!(valid_trellis, "invalid expert trellis");
    let tiles_k = shape[if slot_major { 2 } else { 0 }];
    let tiles_n = shape[if slot_major { 3 } else { 1 }];
    ensure!(
        output_dims > 0 && output_dims % 128 == 0,
        "expert output width must be 128-aligned"
    );
    ensure!(
        projections_per_route > 0
            && selected
                .shape()
                .iter()
                .map(|&n| i64::from(n))
                .product::<i64>()
                * i64::from(projections_per_route)
                == i64::from(rows),
        "expert route count does not match input rows"
    );
    ensure!(
        selected.dtype() == Dtype::UInt32
            && (input_pretransformed || suh.is_some_and(|s| s.shape() == x.shape())),
        "invalid expert routes or input scales"
    );
    ensure!(
        return_raw || svh.is_some_and(|s| s.shape() == [rows, output_dims]),
        "invalid expert output scales"
    );
    let xhat = if input_pretransformed {
        x.astype(Dtype::Float16)?.reshape(&[-1])?
    } else if native_prepare
        && crate::flash_support::native_prepare_shape(rows, input_dims)
        && matches!(x.dtype(), Dtype::Float16 | Dtype::Float32)
        && suh.is_some_and(|scale| scale.dtype() == Dtype::Float16)
    {
        x.flash_prepare128(suh.expect("validated input scales"))?
    } else {
        x.astype(Dtype::Float16)?
            .mul(
                &suh.expect("validated input scales")
                    .astype(Dtype::Float16)?,
            )?
            .reshape(&[rows, input_dims / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[-1])?
    };
    #[cfg(test)]
    crate::flash_profile::stage("exl3_input_scale_hadamard", &xhat)?;
    let output_tiles = output_dims / 16;
    let local_tiles = rows
        .checked_mul(output_tiles)
        .context("expert tile count overflow")?;
    let splits = if slot_major {
        crate::flash_support::reference_splits(tiles_k as u32, output_tiles as u32) as i32
    } else {
        split_count(tiles_k, local_tiles)
    };
    let m5 = array::is_m5_gpu()?;
    let use_nt4 = m5 && rows >= 64 && output_tiles % 4 == 0;
    let mut nt = if slot_major {
        1
    } else if use_nt4 {
        4
    } else if local_tiles >= 1024 {
        if local_tiles % 2 == 0 { 2 } else { 1 }
    } else if local_tiles % 4 == 0 {
        4
    } else if local_tiles % 2 == 0 {
        2
    } else {
        1
    };
    if output_tiles % nt != 0 {
        nt = 1;
    }
    let simdgroups = if slot_major || (m5 && k == 2) {
        4
    } else if m5 {
        8
    } else {
        4
    };
    ensure!(
        !masked || slot_major,
        "masked experts require slot-major banks"
    );
    ensure!(
        groups.is_none() || (slot_major && !masked && k == 2 && matches!(cb, Codebook::Mul1)),
        "grouped experts require unmasked K2 MUL1 slot-major banks"
    );
    let masked_header = if masked {
        format!(
            "\n#define MLXL3_MASKED_SLOTS 1\n#define MLXL3_SLOT_CAPACITY {}u\n",
            shape[0]
        )
    } else {
        String::new()
    };
    let local_output_dims = if stable_shapes {
        ensure!(
            slot_major && selected.shape().len() == 1,
            "dynamic strides require flat slot routes"
        );
        crate::flash_support::expert_output_stride(
            selected.shape()[0] as u32,
            projections_per_route as u32,
            output_dims as u32,
        )
        .context("dynamic expert output stride overflow")?;
        "mlxl3_route_output_dims".to_owned()
    } else {
        (local_tiles * 16).to_string()
    };
    let kernel_source = include_str!("../shaders/_qmv_mapped_tile_kernel.metal");
    // MLX detects shape bindings in the body, not in the header.
    let stable_source = stable_shapes.then(|| format!("uint {} = uint({}_shape[0]) * uint(ROUTING_REPEAT) * uint(OUTPUT_TILES) * 16u;\n{kernel_source}",
        if groups.is_some() { "mlxl3_group_output_dims" } else { "mlxl3_route_output_dims" },
        if groups.is_some() { "tile_sub" } else { "tile_map" },
    ));
    let header = codebook_header(cb)
        + if lane_order {
            "\n#define MLXL3_QMV_LANE_ORDER 1\n"
        } else {
            ""
        }
        + &masked_header
        + reference_reduction_header(slot_major)
        + &format!(
            "\n#define MLXL3_QMV_NT {nt}u\n#define MLXL3_QMV_SG {simdgroups}u\n#define MLXL3_K_BITS {k}u\n#define MLXL3_K3_WINDOW_DECODE {}\n#define K {k}\n#define CB {}\n#define PACKED_U32 {}\n#define INPUT_DIMS {input_dims}\n#define TILES_K {}\n#define TILES_N {}\n#define N_SPLITS {splits}\n#define LOCAL_OUTPUT_DIMS {}\n#define IDENTITY_MAP 0\n#define EXPERT_MAP 1\n#define OUTPUT_TILES {output_tiles}\n#define ROUTING_REPEAT {projections_per_route}\n#define PROJECTION_STRIDE_TILES {projection_stride_tiles}\n#define MLXL3_SLOT_MAJOR {}\n",
            u8::from(k == 3),
            cb as u32,
            k * 8,
            tiles_k,
            tiles_n,
            local_output_dims,
            u8::from(slot_major),
        );
    let words = trellis.reshape(&[-1])?.view(Dtype::UInt32)?;
    let name = format!(
        "mlxl3_rs_expert_{}_{}_{}_{}_{}_{}_{}_slots{}{}",
        input_dims,
        output_dims,
        k,
        cb as u32,
        nt,
        simdgroups,
        splits,
        u8::from(slot_major),
        if masked {
            format!("_resident{}", shape[0])
        } else {
            String::new()
        },
    );
    let dispatch = |group: Option<(i32, &Array)>| -> Result<Array> {
        let (route_rows, work_tiles) = group.map_or((rows, local_tiles), |(width, origins)| {
            let grouped_rows = origins.shape()[0] * projections_per_route;
            (grouped_rows, grouped_rows / width * output_tiles)
        });
        let grouped_header = group.map(|(width, _)| {
            let stride = if stable_shapes {
                "mlxl3_group_output_dims".to_owned()
            } else {
                format!("{}u", route_rows * output_dims)
            };
            format!("{header}\n#define MLXL3_GROUPED_ROUTES 1\n#define MLXL3_QMV_MB {width}u\n#define MLXL3_GROUP_OUTPUT_DIMS {stride}\n")
        });
        #[cfg(test)]
        crate::flash_profile::kernel_signature(
            stable_shapes,
            &group.map_or_else(|| name.clone(), |(width, _)| format!("{name}_group{width}")),
            grouped_header.as_deref().unwrap_or(&header),
            stable_source.as_deref().unwrap_or(kernel_source),
        );
        Ok(array::metal_kernel(
            &group.map_or_else(|| name.clone(), |(width, _)| format!("{name}_group{width}")),
            &["xhat", "trellis", "tile_map", "tile_sub"],
            &["yhat"],
            grouped_header.as_deref().unwrap_or(&header),
            stable_source.as_deref().unwrap_or(kernel_source),
            &[
                &xhat,
                &words,
                selected,
                group.map_or(selected, |(_, origins)| origins),
            ],
            &[vec![splits, route_rows, output_dims]],
            &[Dtype::Float32],
            [
                (work_tiles / nt)
                    .checked_mul(simdgroups * 32)
                    .context("expert QMV grid overflow")?,
                1,
                splits,
            ],
            [simdgroups * 32, 1, 1],
        )?
        .remove(0))
    };
    let partials = if let Some(groups) = groups {
        let parts = groups
            .origins
            .iter()
            .map(|(width, origins)| dispatch(Some((*width, origins))))
            .collect::<Result<Vec<_>>>()?;
        let joined = if parts.len() == 1 {
            parts[0].try_clone()?
        } else {
            Array::concatenate(&parts.iter().collect::<Vec<_>>(), 1)?
        };
        let restore = groups
            .restore
            .iter()
            .flat_map(|&row| {
                (0..projections_per_route as u32)
                    .map(move |projection| row * projections_per_route as u32 + projection)
            })
            .collect::<Vec<_>>();
        joined.take(&Array::from_u32(&restore, &[rows])?, 1)?
    } else {
        dispatch(None)?
    };
    #[cfg(test)]
    crate::flash_profile::stage("exl3_trellis_qmv", &partials)?;
    let yhat = if splits == 1 {
        partials.reshape(&[rows, output_dims])?
    } else {
        partials.sum(0, false)?
    };
    #[cfg(test)]
    crate::flash_profile::stage("exl3_split_reduce", &yhat)?;
    if return_raw {
        return Ok(yhat);
    }
    let output = yhat
        .astype(Dtype::Float16)?
        .reshape(&[rows, output_dims / 128, 128])?
        .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
        .reshape(&[rows, output_dims])?
        .mul(svh.expect("validated output scales"))?
        .astype(x.dtype())?;
    #[cfg(test)]
    crate::flash_profile::stage("exl3_output_hadamard_scale", &output)?;
    Ok(output)
}

fn split_count(input_tiles: i32, output_tiles: i32) -> i32 {
    crate::contracts::exl3_projection_splits((input_tiles as u32) * 16, (output_tiles as u32) * 16)
        .expect("validated EXL3 widths") as i32
}

pub(crate) fn codebook_header(cb: Codebook) -> String {
    let body = match cb {
        Codebook::Default => {
            "uint bits=x*89226354u+64248484u; bits=0x3B603B60u^(bits&0x8FFF8FFFu); half2 v=as_type<half2>(bits); return float(v.x+v.y);"
        }
        Codebook::Mcg => {
            "uint bits=x*0xCBAC1FEDu; bits=0x3B603B60u^(bits&0x8FFF8FFFu); half2 v=as_type<half2>(bits); return float(v.x+v.y);"
        }
        Codebook::Mul1 => {
            "uint bits=x*0x83DCD12Du; uint pairs=(bits&0x00ff00ffu)+((bits>>8)&0x00ff00ffu); uint sum=0x6400u+(pairs&0xffffu)+(pairs>>16); half value=as_type<half>(ushort(sum)); half inv=as_type<half>(ushort(0x1EEEu)); half bias=as_type<half>(ushort(0xC931u)); return float(value*inv+bias);"
        }
    };
    format!("inline float mlxl3_decode_codeword(uint x,int unused_cb) {{ x &= 0xffffu; {body} }}\n")
}

#[cfg(test)]
#[path = "linear/dense_epilogue_tests.rs"]
mod dense_epilogue_tests;

#[cfg(test)]
#[path = "linear/prepared_tests.rs"]
mod prepared_tests;

#[cfg(test)]
#[path = "linear/reference_mb34_tests.rs"]
mod reference_mb34_tests;

#[cfg(test)]
#[path = "linear/q27_kernel_lab.rs"]
mod q27_kernel_lab;

#[cfg(test)]
mod tests {
    #[cfg(kani)]
    #[kani::proof]
    fn q4_projection_reference_entry() {
        assert!(q4_draft_matches_exl3_rotation_reference().is_ok());
    }

    #[test]
    #[ignore = "physical Apple GPU and Flash checkpoint; run alone"]
    fn flash_reference_batches_match_independent_single_rows() -> anyhow::Result<()> {
        use super::*;
        let checkpoint =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut cases = 0;
        for name in [
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
            "model.language_model.layers.0.linear_attn.in_proj_z",
            "model.language_model.layers.0.linear_attn.out_proj",
            "lm_head",
        ] {
            let mut linear = Exl3Linear::from_checkpoint(&checkpoint, name)?;
            ensure!(linear.reference_accumulation && linear.simdgroups == 4 && linear.k != 7);
            for rows in [2, 3, 4, 8, 5, 6, 7, 1, 9] {
                for dtype in [Dtype::Float16, Dtype::Float32] {
                    for strided in [false, true] {
                        let value = |r: i32, c: i32| ((r * 37 + c * 13) % 997) as f32 / 1024. - 0.5;
                        let x = if strided {
                            let values = (0..linear.rows)
                                .flat_map(|c| (0..rows).map(move |r| value(r, c)))
                                .collect::<Vec<_>>();
                            Array::from_f32(&values, &[linear.rows, rows])?
                                .astype(dtype)?
                                .transpose(&[1, 0])?
                        } else {
                            let values = (0..rows)
                                .flat_map(|r| (0..linear.rows).map(move |c| value(r, c)))
                                .collect::<Vec<_>>();
                            Array::from_f32(&values, &[rows, linear.rows])?.astype(dtype)?
                        };
                        linear.flash_batch = false;
                        let expected = (0..rows)
                            .map(|r| linear.forward(&x.slice(0, r, r + 1)?))
                            .collect::<Result<Vec<_>>>()?;
                        let expected = Array::concatenate(&expected.iter().collect::<Vec<_>>(), 0)?;
                        linear.flash_batch = true;
                        let actual = linear.forward(&x)?;
                        ensure!(actual.shape() == [rows, linear.cols]);
                        let values = actual.astype(Dtype::Float32)?.to_f32()?;
                        ensure!(
                            values.len() == (rows * linear.cols) as usize
                                && values.iter().all(|v| v.is_finite())
                        );
                        ensure!(
                            actual.to_bytes()? == expected.to_bytes()?,
                            "reference batch differs {name}/B{rows}/{dtype:?}/strided={strided}"
                        );
                        if rows > 1 {
                            ensure!(
                                actual.slice(0, 0, 1)?.to_bytes()?
                                    != expected.slice(0, 1, 2)?.to_bytes()?,
                                "vacuous row oracle"
                            );
                        }
                        cases += 1;
                    }
                }
            }
            assert!(
                linear
                    .forward(&Array::zeros_dtype(&[0, linear.rows], Dtype::Float16)?)
                    .is_err()
            );
            assert!(
                linear
                    .forward(&Array::zeros_dtype(&[2, linear.rows - 1], Dtype::Float16)?)
                    .is_err()
            );
        }
        assert_eq!(cases, 144);
        println!("FLASH_REFERENCE_BATCH cases={cases} parity=byte_exact");
        Ok(())
    }

    #[cfg(kani)]
    #[kani::proof]
    fn actual_flash_reference_batch_entry() {
        flash_reference_batches_match_independent_single_rows().unwrap();
    }

    #[test]
    #[ignore = "physical Apple GPU; real Flash slot shapes, dtypes, strides and wrong-slot oracle"]
    fn flash_slot_shapes_match_resident_reference_for_every_route() -> anyhow::Result<()> {
        use super::*;
        let cases = [
            vec![9u32],
            vec![0, 1],
            vec![0, 1, 0, 2],
            (0..10).map(|i| i % 5).collect(),
            (0..20).map(|i| i % 10).collect(),
            (0..30).map(|i| i % 10).collect(),
            (0..40).map(|i| i % 10).collect(),
            (0..50).map(|i| i % 10).collect(),
            (0..60).map(|i| i % 10).collect(),
            (0..70).map(|i| i % 10).collect(),
            (0..80).map(|i| i % 10).collect(),
        ];
        let mut checked = 0;
        let mut masked_checked = 0;
        let mut grouped_checked = 0;
        crate::flash_profile::start(false)?;
        for stable_shapes in [false, true] {
            for (input, output, projections) in [(2560usize, 640usize, 2usize), (640, 2560, 1)] {
                let words = input / 16 * (output / 16) * 32;
                let packed = (0..10 * projections * words)
                    .map(|i| (i as u16).wrapping_mul(1777))
                    .collect::<Vec<_>>();
                let bank = Array::from_u16(
                    &packed,
                    &[
                        10,
                        projections as i32,
                        (input / 16) as i32,
                        (output / 16) as i32,
                        32,
                    ],
                )?;
                for dtype in [Dtype::Float16, Dtype::Float32] {
                    for strided in [false, true] {
                        for routes in &cases {
                            let rows = routes.len() * projections;
                            let value = |row: usize, col: usize| {
                                ((row * 37 + col * 13) % 997) as f32 / 1024. - 0.5
                            };
                            let values = (0..rows)
                                .flat_map(|r| (0..input).map(move |c| value(r, c)))
                                .collect::<Vec<_>>();
                            let x = if strided {
                                let transposed = (0..input)
                                    .flat_map(|c| (0..rows).map(move |r| value(r, c)))
                                    .collect::<Vec<_>>();
                                Array::from_f32(&transposed, &[input as i32, rows as i32])?
                                    .astype(dtype)?
                                    .transpose(&[1, 0])?
                            } else {
                                Array::from_f32(&values, &[rows as i32, input as i32])?
                                    .astype(dtype)?
                            };
                            let scales = |width: usize| {
                                (0..rows * width)
                                    .map(|i| {
                                        half::f16::from_f32(0.5 + (i % 11) as f32 / 32.).to_bits()
                                    })
                                    .collect::<Vec<_>>()
                            };
                            let su = scales(input);
                            let sv = scales(output);
                            let suh = Array::from_f16_bits(&su, &[rows as i32, input as i32])?;
                            let svh = Array::from_f16_bits(&sv, &[rows as i32, output as i32])?;
                            let before = Array::host_readbacks_for_test();
                            let actual = expert_slots(
                                &x,
                                &bank,
                                &suh,
                                &svh,
                                &ExpertSlots::new(routes, 10)?,
                                output as i32,
                                projections as i32,
                                2,
                                Codebook::Mul1,
                                stable_shapes,
                                false,
                            )?;
                            assert_eq!(Array::host_readbacks_for_test(), before);
                            assert_eq!(actual.shape(), [rows as i32, output as i32]);
                            let host = actual.to_f32()?;
                            assert_eq!(host.len(), rows * output);
                            assert!(host.iter().all(|v| v.is_finite()));
                            let grouped_reads = Array::host_readbacks_for_test();
                            let grouped_slots = ExpertSlots::grouped(routes, 10)?;
                            grouped_checked += usize::from(grouped_slots.shares_weights());
                            let grouped = expert_slots(
                                &x,
                                &bank,
                                &suh,
                                &svh,
                                &grouped_slots,
                                output as i32,
                                projections as i32,
                                2,
                                Codebook::Mul1,
                                stable_shapes,
                                false,
                            )?;
                            assert_eq!(Array::host_readbacks_for_test(), grouped_reads);
                            assert_eq!(grouped.shape(), actual.shape());
                            assert_eq!(
                                grouped.to_bytes()?,
                                actual.to_bytes()?,
                                "shared EXL3 changed a route: {routes:?}"
                            );
                            if routes.len() == 20 {
                                let mut wrong = ExpertSlots::grouped(routes, 10)?;
                                let origins = &mut wrong.groups.as_mut().unwrap().origins;
                                let rows = &mut origins
                                    .iter_mut()
                                    .find(|(width, _)| *width == 2)
                                    .unwrap()
                                    .1;
                                let mut indices = rows.to_u32()?;
                                indices[0] = 1;
                                *rows = Array::from_u32(&indices, &[20])?;
                                let changed = expert_slots(
                                    &x,
                                    &bank,
                                    &suh,
                                    &svh,
                                    &wrong,
                                    output as i32,
                                    projections as i32,
                                    2,
                                    Codebook::Mul1,
                                    stable_shapes,
                                    false,
                                )?;
                                assert_ne!(
                                    changed.to_bytes()?,
                                    actual.to_bytes()?,
                                    "wrong shared origin was not detected"
                                );
                            }
                            for (route, &slot) in routes.iter().enumerate() {
                                for projection in 0..projections {
                                    let row = route * projections + projection;
                                    let index = slot as usize * projections + projection;
                                    let mut reference = Exl3Linear::new(
                                        Array::from_u16(
                                            &packed[index * words..(index + 1) * words],
                                            &[(input / 16) as i32, (output / 16) as i32, 32],
                                        )?,
                                        Array::from_f16_bits(
                                            &su[row * input..(row + 1) * input],
                                            &[input as i32],
                                        )?,
                                        Array::from_f16_bits(
                                            &sv[row * output..(row + 1) * output],
                                            &[output as i32],
                                        )?,
                                        None,
                                        2,
                                        Codebook::Mul1,
                                    )?;
                                    reference.simdgroups = 4;
                                    reference.reference_accumulation = true;
                                    let expected = reference.forward(
                                        &Array::from_f32(
                                            &values[row * input..(row + 1) * input],
                                            &[1, input as i32],
                                        )?
                                        .astype(dtype)?,
                                    )?;
                                    assert_eq!(
                                        actual.slice(0, row as i32, row as i32 + 1)?.to_bytes()?,
                                        expected.to_bytes()?,
                                        "input={input},output={output},dtype={dtype:?},strided={strided},row={row},slot={slot}"
                                    );
                                }
                            }
                            let ordinary = actual.to_bytes()?;
                            let row_bytes = ordinary.len() / rows;
                            for mask in 0..3 {
                                let mut map = [u32::MAX; 512];
                                for id in 0..10u32 {
                                    if mask == 2 || (mask == 1 && id % 2 == 0) {
                                        map[id as usize] = id;
                                    }
                                }
                                let slots = ExpertSlots::residents(
                                    &Array::from_u32(routes, &[routes.len() as i32])?,
                                    &map,
                                    10,
                                )?;
                                let masked = expert_slots(
                                    &x,
                                    &bank,
                                    &suh,
                                    &svh,
                                    &slots,
                                    output as i32,
                                    projections as i32,
                                    2,
                                    Codebook::Mul1,
                                    stable_shapes,
                                    false,
                                )?;
                                assert_eq!(masked.shape(), actual.shape());
                                let values = masked.to_f32()?;
                                let bytes = masked.to_bytes()?;
                                assert_eq!(values.len(), rows * output);
                                assert!(values.iter().all(|v| v.is_finite()));
                                for row in 0..rows {
                                    if map[routes[row / projections] as usize] == u32::MAX {
                                        assert!(
                                            values[row * output..(row + 1) * output]
                                                .iter()
                                                .all(|&v| v == 0.)
                                        );
                                    } else {
                                        assert_eq!(
                                            bytes[row * row_bytes..(row + 1) * row_bytes],
                                            ordinary[row * row_bytes..(row + 1) * row_bytes],
                                            "resident row differs: mask={mask},row={row}"
                                        );
                                    }
                                }
                                masked_checked += 1;
                            }
                            if routes.len() == 1 {
                                let wrong = expert_slots(
                                    &x,
                                    &bank,
                                    &suh,
                                    &svh,
                                    &ExpertSlots::new(&[0], 10)?,
                                    output as i32,
                                    projections as i32,
                                    2,
                                    Codebook::Mul1,
                                    stable_shapes,
                                    false,
                                )?;
                                assert_ne!(
                                    actual.to_bytes()?,
                                    wrong.to_bytes()?,
                                    "wrong-slot control was not detected"
                                );
                            }
                            if routes.len() == 20 {
                                for (invalid, capacity) in [
                                    (vec![], 10),
                                    (vec![0; 81], 10),
                                    (vec![10; 20], 10),
                                    (vec![u32::MAX; 20], 10),
                                    (vec![0; 20], 0),
                                    (vec![0; 20], 513),
                                ] {
                                    assert!(ExpertSlots::grouped(&invalid, capacity).is_err());
                                }
                            }
                            checked += 1;
                        }
                    }
                }
            }
        }
        let events = serde_json::to_value(crate::flash_profile::finish()?)?;
        let mut keys: [std::collections::BTreeSet<String>; 2] = Default::default();
        for event in events
            .as_array()
            .unwrap()
            .iter()
            .filter(|e| e["name"] == "expert_kernel_signature")
        {
            let data = &event["data"];
            keys[usize::from(data["stable"].as_bool().unwrap())]
                .insert(serde_json::to_string(&data["key"])?);
        }
        ensure!(
            !keys[0].is_empty() && !keys[1].is_empty() && keys[1].len() < keys[0].len(),
            "stable expert headers did not reduce actual factory keys"
        );
        println!(
            "FLASH_STABLE_KERNEL_KEYS baseline={} stable={} equality=full_metadata no_timing_claim=true",
            keys[0].len(),
            keys[1].len()
        );
        assert_eq!(checked, 176);
        assert_eq!(masked_checked, 528);
        assert_eq!(grouped_checked, 112);
        println!(
            "FLASH_SHARED_SLOT_SHAPES cases={checked} enabled={grouped_checked} fallback=64 widths=1..4 routes=1..80 wrong_origin_controls=16 parity=byte_exact no_added_host_readback=true"
        );
        println!("FLASH_SLOT_SHAPES cases={checked} wrong_slot_controls=16 parity=byte_exact");
        println!(
            "FLASH_MASKED_SLOT_SHAPES cases={masked_checked} masks=all_cold,mixed,all_hot parity=byte_exact"
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU; resident lookup must not gather missing or invalid experts"]
    fn gpu_resident_plans_and_scale_gathers_preserve_bounds_and_strides() -> anyhow::Result<()> {
        use super::*;
        let mut map = [u32::MAX; 512];
        for (id, slot) in map.iter_mut().enumerate() {
            if id % 17 == 0 {
                *slot = (id * 7 % 10) as u32;
            }
        }
        map[511] = 9;
        let bits = (0..10 * 2 * 128)
            .map(|i| half::f16::from_f32((i as f32 - 300.) / 1024.).to_bits())
            .collect::<Vec<_>>();
        let all_ids = (0..512u32).chain([512, u32::MAX]).collect::<Vec<_>>();
        let mut cases = 0;
        for strided_bank in [false, true] {
            let bank = if strided_bank {
                let storage = bits
                    .iter()
                    .flat_map(|&b| [b, half::f16::NAN.to_bits()])
                    .collect::<Vec<_>>();
                Array::from_f16_bits(&storage, &[10, 2, 128, 2])?
                    .slice(3, 0, 1)?
                    .reshape(&[10, 2, 128])?
            } else {
                Array::from_f16_bits(&bits, &[10, 2, 128])?
            };
            for strided_ids in [false, true] {
                for ids in all_ids.chunks(80) {
                    let selected = if strided_ids {
                        let storage = ids
                            .iter()
                            .flat_map(|&id| [id, u32::MAX])
                            .collect::<Vec<_>>();
                        Array::from_u32(&storage, &[ids.len() as i32, 2])?
                            .slice(1, 0, 1)?
                            .reshape(&[-1])?
                    } else {
                        Array::from_u32(ids, &[ids.len() as i32])?
                    };
                    let before = Array::host_readbacks_for_test();
                    let slots = ExpertSlots::residents(&selected, &map, 10)?;
                    let scales = slots.gather_scales(&bank)?;
                    assert_eq!(Array::host_readbacks_for_test(), before);
                    let expected = ids
                        .iter()
                        .map(|&id| map.get(id as usize).copied().unwrap_or(u32::MAX))
                        .collect::<Vec<_>>();
                    assert_eq!(slots.indices().to_u32()?, expected);
                    let bytes = scales.to_bytes()?;
                    assert_eq!(scales.shape(), [ids.len() as i32, 2, 128]);
                    assert_eq!(bytes.len(), ids.len() * 256 * 2);
                    assert!(scales.to_f32()?.iter().all(|v| v.is_finite()));
                    let expected_bits = expected
                        .iter()
                        .flat_map(|&slot| {
                            if slot == u32::MAX {
                                vec![0u16; 256]
                            } else {
                                bits[slot as usize * 256..(slot as usize + 1) * 256].to_vec()
                            }
                        })
                        .flat_map(u16::to_le_bytes)
                        .collect::<Vec<_>>();
                    assert_eq!(bytes, expected_bits);
                    cases += 1;
                }
            }
        }
        let selected = Array::from_u32(&[511], &[1])?;
        for cap in [0, 513] {
            assert!(ExpertSlots::residents(&selected, &map, cap).is_err());
        }
        let mut invalid = map;
        invalid[511] = 10;
        assert!(ExpertSlots::residents(&selected, &invalid, 10).is_err());
        for n in [0, 81] {
            assert!(
                ExpertSlots::residents(&Array::from_u32(&vec![0; n], &[n as i32])?, &map, 10)
                    .is_err()
            );
        }
        assert!(ExpertSlots::residents(&selected.astype(Dtype::Int32)?, &map, 10).is_err());
        let mut wrong = map;
        wrong[511] = 0;
        assert_ne!(
            ExpertSlots::residents(&selected, &wrong, 10)?
                .indices()
                .to_u32()?,
            vec![9]
        );
        assert_eq!(cases, 28);
        println!(
            "FLASH_GPU_RESIDENT_PLAN cases={cases} ids=514 invalid_controls=6 wrong_map=true parity=byte_exact"
        );
        Ok(())
    }

    #[cfg(kani)]
    #[kani::proof]
    fn actual_flash_slots_preserve_reference_outputs() {
        flash_slot_shapes_match_resident_reference_for_every_route().unwrap();
    }

    #[test]
    #[ignore = "physical Apple GPU; exact SSD slot layout versus resident EXL3"]
    fn streamed_slots_match_resident_exl3_and_reject_invalid_routes() -> anyhow::Result<()> {
        use super::*;
        let (slots, projections, width, k) = (3usize, 2usize, 128usize, 2usize);
        let packed = (0..slots * projections * 8 * 8 * 32)
            .map(|i| (i as u16).wrapping_mul(1777))
            .collect::<Vec<_>>();
        let bank = Array::from_u16(&packed, &[3, 2, 8, 8, 32])?;
        let input = (0..width)
            .map(|i| half::f16::from_f32((i as f32 - 61.) / 79.).to_bits())
            .collect::<Vec<_>>();
        for routes in [vec![2u32], vec![0, 2], vec![2, 0, 1, 2, 1, 0, 2, 2, 0, 1]] {
            let rows = (routes.len() * projections) as i32;
            let x = Array::from_f16_bits(&input.repeat(rows as usize), &[rows, 128])?;
            let scales = vec![half::f16::from_f32(0.5).to_bits(); rows as usize * width];
            let suh = Array::from_f16_bits(&scales, &[rows, 128])?;
            let svh = Array::from_f16_bits(&scales, &[rows, 128])?;
            let before = Array::host_readbacks_for_test();
            let actual = expert_slots(
                &x,
                &bank,
                &suh,
                &svh,
                &ExpertSlots::new(&routes, 3)?,
                128,
                2,
                k,
                Codebook::Mul1,
                false,
                false,
            )?;
            assert_eq!(
                Array::host_readbacks_for_test(),
                before,
                "validated slots were read back from MLX"
            );
            let actual = actual.to_f16_bits()?;
            assert_eq!(actual.len(), rows as usize * width);
            for (route, &slot) in routes.iter().enumerate() {
                for projection in 0..projections {
                    let index = slot as usize * projections + projection;
                    let mut reference = Exl3Linear::new(
                        Array::from_u16(&packed[index * 2048..(index + 1) * 2048], &[8, 8, 32])?,
                        Array::from_f16_bits(&scales[..width], &[128])?,
                        Array::from_f16_bits(&scales[..width], &[128])?,
                        None,
                        k,
                        Codebook::Mul1,
                    )?;
                    // Match the Flash checkpoint policy; new() retains the legacy policy.
                    reference.simdgroups = 4;
                    reference.reference_accumulation = true;
                    let expected = reference
                        .forward(&Array::from_f16_bits(&input, &[1, 128])?)?
                        .to_f16_bits()?;
                    let start = (route * 2 + projection) * width;
                    assert_eq!(
                        &actual[start..start + width],
                        expected,
                        "slot={slot}, projection={projection}"
                    );
                }
            }
        }
        let x = Array::from_f16_bits(&input.repeat(2), &[2, 128])?;
        let scales = Array::from_f16_bits(&vec![half::f16::ONE.to_bits(); 256], &[2, 128])?;
        for (ids, capacity) in [
            (&[][..], 3),
            (&[3][..], 3),
            (&[u32::MAX][..], 3),
            (&[0][..], 0),
        ] {
            assert!(ExpertSlots::new(ids, capacity).is_err());
        }
        assert!(
            expert_slots(
                &x,
                &bank,
                &scales,
                &scales,
                &ExpertSlots::new(&[0], 2)?,
                128,
                2,
                k,
                Codebook::Mul1,
                false,
                false,
            )
            .is_err()
        );
        Ok(())
    }
    #[test]
    #[ignore = "requires disposable draft Q4 projection and physical Apple GPU"]
    fn q4_draft_matches_exl3_rotation_reference() -> anyhow::Result<()> {
        use crate::{affine::AffineLinear, checkpoint::Checkpoint};
        use std::path::Path;
        let root = std::env::var("MLXL3_MTP_TEST_FULL_HEAD")
            .unwrap_or_else(|_| "build/mtp-followup/q4-head".into());
        let root = Path::new(&root);
        let reference: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(root.join("reference.json"))?)?;
        let model = std::env::var("MLXL3_MTP_TEST_MODEL")
            .unwrap_or_else(|_| "models/Qwen3.6-35B-A3B-EXL3-2.49bpw".into());
        let original = crate::checkpoint::inspect(Path::new(&model))?;
        let exl3 = super::Exl3Linear::from_checkpoint(&original, "lm_head")?;
        let (hidden, vocab) = (exl3.input_dims(), exl3.output_dims());
        assert!(hidden > 0 && vocab >= 4096);
        let checkpoint = Checkpoint {
            path: root.to_owned(),
            model_type: "draft_head_q4".into(),
            bits: Some(4.),
            size_bytes: 0,
            modules: Vec::new(),
            tensors: crate::checkpoint::read_header(&root.join("draft_head_q4.safetensors"))?,
        };
        let q4 = AffineLinear::load(&checkpoint, "lm_head", hidden, vocab, None)?;
        let mut compact = None;
        if let Some(path) = std::env::var_os("MLXL3_MTP_TEST_COMPACT_HEAD") {
            let path = std::path::PathBuf::from(path);
            let metadata: serde_json::Value =
                serde_json::from_reader(std::fs::File::open(path.join("draft_head_q4.json"))?)?;
            let ids = metadata["token_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|id| u32::try_from(id.as_u64().unwrap()).unwrap())
                .collect::<Vec<_>>();
            assert!(!ids.is_empty() && ids.len() < vocab as usize);
            assert!(
                ids.iter().all(|&id| id < vocab as u32)
                    && ids.windows(2).all(|pair| pair[0] < pair[1])
            );
            let checkpoint = Checkpoint {
                path: path.clone(),
                model_type: "draft_head_q4".into(),
                bits: Some(4.),
                size_bytes: 0,
                modules: Vec::new(),
                tensors: crate::checkpoint::read_header(&path.join("draft_head_q4.safetensors"))?,
            };
            compact = Some((
                AffineLinear::load(
                    &checkpoint,
                    "lm_head",
                    hidden,
                    i32::try_from(ids.len())?,
                    None,
                )?,
                ids,
            ));
        }
        let inputs = reference["inputs"].as_array().unwrap();
        let logits = reference["logits"].as_array().unwrap();
        assert_eq!(inputs.len(), 3, "missing numerical fixtures");
        assert_eq!(inputs.len(), logits.len());
        for (input, logits) in inputs.iter().zip(logits) {
            let input = input
                .as_array()
                .unwrap()
                .iter()
                .map(|v| half::f16::from_f32(v.as_f64().unwrap() as f32).to_bits())
                .collect::<Vec<_>>();
            let expected = logits
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect::<Vec<_>>();
            assert_eq!(input.len(), hidden as usize);
            assert_eq!(expected.len(), 4096);
            assert!(expected.iter().all(|v| v.is_finite()));
            let input = crate::array::Array::from_f16_bits(&input, &[1, hidden])?;
            let actual = exl3.forward(&input)?.to_f16_bits()?;
            let quantized = q4.forward(&input)?.to_f16_bits()?;
            assert_eq!(actual.len(), vocab as usize);
            assert_eq!(quantized.len(), actual.len());
            assert!(
                actual
                    .iter()
                    .chain(&quantized)
                    .all(|&b| half::f16::from_bits(b).is_finite())
            );
            let energy = expected.iter().map(|v| v * v).sum::<f64>();
            assert!(energy.is_finite() && energy > 0.);
            let error = |values: &[u16]| {
                values
                    .iter()
                    .zip(&expected)
                    .map(|(&b, &v)| (half::f16::from_bits(b).to_f64() - v).powi(2))
                    .sum::<f64>()
                    / energy
            };
            let reconstruction = error(&actual).sqrt();
            let quantization = error(&quantized).sqrt();
            eprintln!(
                "head relative output RMSE: reconstructed={reconstruction}, affine4={quantization}"
            );
            assert!(
                reconstruction < 0.01,
                "EXL3 rotations/scales reconstructed incorrectly"
            );
            assert!(quantization < 0.2, "poor draft quantization");
            if let Some((head, ids)) = &compact {
                let values = head.forward(&input)?.to_f16_bits()?;
                let selected = ids
                    .iter()
                    .map(|&id| quantized[id as usize])
                    .collect::<Vec<_>>();
                assert_eq!(values.len(), ids.len());
                assert_eq!(
                    values, selected,
                    "compact projection differs from full Q4 rows"
                );
                eprintln!(
                    "compact projection: {} finite rows exactly match full Q4",
                    ids.len()
                );
            }
        }
        Ok(())
    }
    use super::*;

    #[test]
    fn lane_order_override_restores_and_stays_thread_local() {
        assert_eq!(LANE_ORDER_TEST_OVERRIDE.with(std::cell::Cell::get), None);
        with_lane_order_test(false, || {
            assert!(!lane_order_enabled());
            with_lane_order_test(true, || assert!(lane_order_enabled()));
            assert!(!lane_order_enabled());
            std::thread::spawn(|| {
                assert_eq!(LANE_ORDER_TEST_OVERRIDE.with(std::cell::Cell::get), None);
            })
            .join()
            .unwrap();
            assert!(
                std::panic::catch_unwind(|| {
                    with_lane_order_test(true, || panic!("temporary test failure"));
                })
                .is_err()
            );
            assert!(!lane_order_enabled());
        });
        assert_eq!(LANE_ORDER_TEST_OVERRIDE.with(std::cell::Cell::get), None);
    }

    #[cfg(kani)]
    #[kani::proof]
    fn lane_order_real_array_api() {
        let bits: u16 = kani::any();
        let x = Array::from_f16_bits(&[bits; 16], &[16]).unwrap();
        let ids = Array::from_i32(
            &[0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15],
            &[16],
        )
        .unwrap();
        let result = with_lane_order_test(true, || lane_order_input(x, Some(&ids), 1, 1, 16));
        assert_eq!(result.unwrap().0.shape(), [16]);
    }

    #[test]
    #[ignore = "requires physical M5 and 35B"]
    fn q36_qkv_lane_components() -> Result<()> {
        let ck = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_QWEN_TEST_MODEL")?))?;
        let bundle = Exl3Group::new(
            ["in_proj_qkv", "in_proj_z"]
                .into_iter()
                .map(|name| {
                    Exl3Linear::from_checkpoint(
                        &ck,
                        &format!("model.language_model.layers.0.linear_attn.{name}"),
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        )?;
        for rows in 1..=5 {
            let x = Array::from_f16_bits(
                &(0..rows * 2048)
                    .map(|i| f16::from_f32(((i * 37 % 251) as f32 - 124.) / 128.).to_bits())
                    .collect::<Vec<_>>(),
                &[rows, 2048],
            )?;
            let (reference, old_layout) =
                with_q36_reference(|| bundle.small_batch_input(&x, rows, 2))?;
            let (result, layout) = bundle.small_batch_input(&x, rows, 2)?;
            assert!(!old_layout);
            assert_eq!(layout, rows == 4);
            let original = reference.to_f16_bits()?;
            let result = result.to_f16_bits()?;
            assert_eq!(original.len(), (rows * 4096) as usize);
            assert_eq!(original.len(), result.len());
            assert!(original.iter().all(|&v| f16::from_bits(v).is_finite()));
            if layout {
                assert_ne!(original, result);
            }
            for (i, v) in result.into_iter().enumerate() {
                assert_eq!(
                    v,
                    original[if layout {
                        crate::contracts::exl3_lane_source_index(i as u32) as usize
                    } else {
                        i
                    }]
                );
            }
            let run = || -> Result<Vec<Vec<u16>>> {
                bundle.forward(&x)?.iter().map(Array::to_f16_bits).collect()
            };
            let base = with_q36_reference(run)?;
            assert_eq!(
                base.iter().map(Vec::len).collect::<Vec<_>>(),
                vec![rows as usize * 8192, rows as usize * 4096]
            );
            assert!(
                base.iter()
                    .flatten()
                    .all(|&v| f16::from_bits(v).is_finite())
            );
            ensure!(base == run()?, "QKV+Z mismatch at width {rows}");
        }
        Ok(())
    }

    #[test]
    #[ignore = "physical M5, MLXL3_QWEN_TEST_MODEL; run alone"]
    fn lane_order_real_projections_match_stock_and_reject_mutated_index() -> Result<()> {
        let checkpoint =
            crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_QWEN_TEST_MODEL")?))?;
        let mut cases = 0;
        for prefix in [
            "lm_head",
            "model.language_model.layers.0.mlp.down_proj",
            "model.language_model.layers.0.linear_attn.in_proj_z",
        ] {
            let mut linear =
                with_lane_order_test(true, || Exl3Linear::from_checkpoint(&checkpoint, prefix))?;
            ensure!(
                linear.lane_indices.is_some(),
                "fixture missing prepared lane indices"
            );
            for rows in [1, 2, 3, 4] {
                let bits = (0..rows * linear.rows)
                    .map(|i| f16::from_f32(((i * 37 % 251) as f32 - 125.0) / 256.0).to_bits())
                    .collect::<Vec<_>>();
                let x = Array::from_f16_bits(&bits, &[rows, linear.rows])?;
                let expected = with_lane_order_test(false, || linear.forward(&x))?.to_f16_bits()?;
                let actual = with_lane_order_test(true, || linear.forward(&x))?.to_f16_bits()?;
                ensure!(
                    expected.len() == (rows * linear.cols) as usize
                        && !expected.is_empty()
                        && expected
                            .iter()
                            .chain(&actual)
                            .all(|&v| f16::from_bits(v).is_finite()),
                    "invalid lane projection values"
                );
                ensure!(
                    expected == actual,
                    "lane projection differs for {prefix} M={rows}"
                );
                if prefix == "lm_head" && rows == 1 {
                    let correct = linear.lane_indices.take().context("missing lane indices")?;
                    let mut ids = (0..linear.rows)
                        .map(|i| crate::contracts::exl3_lane_source_index(i as u32) as i32)
                        .collect::<Vec<_>>();
                    ids.swap(0, 2);
                    linear.lane_indices = Some(Array::from_i32(&ids, &[linear.rows])?);
                    let corrupted =
                        with_lane_order_test(true, || linear.forward(&x))?.to_f16_bits()?;
                    ensure!(
                        corrupted.len() == expected.len()
                            && corrupted.iter().all(|&v| f16::from_bits(v).is_finite())
                            && corrupted != expected,
                        "mutation escaped lane oracle"
                    );
                    let differing = corrupted
                        .iter()
                        .zip(&expected)
                        .filter(|(a, b)| a != b)
                        .count();
                    eprintln!("lane mutation detected: {differing} finite FP16 values");
                    linear.lane_indices = Some(correct);
                    ensure!(
                        with_lane_order_test(true, || linear.forward(&x))?.to_f16_bits()?
                            == expected,
                        "restored index differs"
                    );
                }
                cases += 1;
            }
        }
        for (left, right) in [("in_proj_qkv", "in_proj_z"), ("gate_proj", "up_proj")] {
            let branch = if left == "in_proj_qkv" {
                "linear_attn"
            } else {
                "mlp"
            };
            let group = with_lane_order_test(true, || {
                Exl3Group::new(vec![
                    Exl3Linear::from_checkpoint(
                        &checkpoint,
                        &format!("model.language_model.layers.0.{branch}.{left}"),
                    )?,
                    Exl3Linear::from_checkpoint(
                        &checkpoint,
                        &format!("model.language_model.layers.0.{branch}.{right}"),
                    )?,
                ])
            })?;
            ensure!(
                group.lane_indices.is_some(),
                "fixture missing grouped lane indices"
            );
            for rows in [1, 2, 3, 4] {
                let bits = (0..rows * group.rows)
                    .map(|i| f16::from_f32(((i * 53 % 251) as f32 - 125.0) / 256.0).to_bits())
                    .collect::<Vec<_>>();
                let x = Array::from_f16_bits(&bits, &[rows, group.rows])?;
                let expected = with_lane_order_test(false, || group.forward(&x))?;
                let actual = with_lane_order_test(true, || group.forward(&x))?;
                ensure!(
                    expected.len() == 2 && actual.len() == 2,
                    "missing grouped outputs"
                );
                for ((left, right), &width) in expected.iter().zip(&actual).zip(&group.widths) {
                    let a = left.to_f16_bits()?;
                    let b = right.to_f16_bits()?;
                    ensure!(
                        a.len() == (rows * width) as usize
                            && !a.is_empty()
                            && a.iter().chain(&b).all(|&v| f16::from_bits(v).is_finite()),
                        "invalid grouped lane values"
                    );
                    ensure!(
                        a == b,
                        "grouped lane projection differs for {branch} M={rows}"
                    );
                }
                cases += 1;
            }
        }
        ensure!(cases == 20, "missing lane projection comparisons");
        Ok(())
    }

    #[test]
    fn grouped_mb3_override_restores_and_stays_thread_local() {
        assert_eq!(GROUPED_MB3_TEST_OVERRIDE.with(std::cell::Cell::get), None);
        with_grouped_mb3_test(false, || {
            assert!(!grouped_mb3_enabled());
            with_grouped_mb3_test(true, || assert!(grouped_mb3_enabled()));
            assert!(!grouped_mb3_enabled());
            std::thread::spawn(|| {
                assert_eq!(GROUPED_MB3_TEST_OVERRIDE.with(std::cell::Cell::get), None);
            })
            .join()
            .unwrap();
            assert!(
                std::panic::catch_unwind(|| {
                    with_grouped_mb3_test(true, || panic!("temporary test failure"));
                })
                .is_err()
            );
            assert!(!grouped_mb3_enabled());
        });
        assert_eq!(GROUPED_MB3_TEST_OVERRIDE.with(std::cell::Cell::get), None);
    }

    #[test]
    fn dense_decode_test_override_restores_and_stays_thread_local() {
        assert_eq!(DENSE_DECODE_TEST_OVERRIDE.with(std::cell::Cell::get), None);
        with_dense_decode_nt4_test(false, || {
            assert!(!dense_decode_nt4_enabled());
            with_dense_decode_nt4_test(true, || {
                assert!(dense_decode_nt4_enabled());
            });
            assert!(!dense_decode_nt4_enabled());
            let other = std::thread::spawn(|| {
                assert_eq!(DENSE_DECODE_TEST_OVERRIDE.with(std::cell::Cell::get), None);
            });
            other.join().unwrap();
            let panicked = std::panic::catch_unwind(|| {
                with_dense_decode_nt4_test(true, || panic!("temporary benchmark failure"));
            });
            assert!(panicked.is_err());
            assert!(!dense_decode_nt4_enabled());
        });
        assert_eq!(DENSE_DECODE_TEST_OVERRIDE.with(std::cell::Cell::get), None);
    }
    use half::f16;
    use std::{path::Path, time::Instant};

    fn serial_rows(linear: &Exl3Linear, x: &Array) -> Result<Array> {
        let rows = x.shape()[0];
        let outputs = (0..rows)
            .map(|row| linear.forward(&x.slice(0, row, row + 1)?))
            .collect::<Result<Vec<_>>>()?;
        Array::concatenate(&outputs.iter().collect::<Vec<_>>(), 0)
    }

    fn serial_group_rows(group: &Exl3Group, x: &Array) -> Result<Vec<Array>> {
        let rows = (0..x.shape()[0])
            .map(|row| group.forward(&x.slice(0, row, row + 1)?))
            .collect::<Result<Vec<_>>>()?;
        (0..group.widths.len())
            .map(|projection| {
                Array::concatenate(
                    &rows.iter().map(|row| &row[projection]).collect::<Vec<_>>(),
                    0,
                )
            })
            .collect()
    }

    #[test]
    #[ignore = "requires local Qwen checkpoint and Apple M5 GPU"]
    fn batch_qmv_eight_rows_matches_serial() -> Result<()> {
        let checkpoint =
            crate::checkpoint::inspect(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        let linear = Exl3Linear::from_checkpoint(
            &checkpoint,
            "model.language_model.layers.0.linear_attn.in_proj_qkv",
        )?;
        let input = |rows| {
            let values = (0..rows * linear.rows)
                .map(|index| f16::from_f32(((index % 251) as f32 - 125.0) / 128.0).to_bits())
                .collect::<Vec<_>>();
            Array::from_f16_bits(&values, &[rows, linear.rows])
        };
        for rows in [2, 4, 16, 23] {
            let x = input(rows)?;
            assert_eq!(
                serial_rows(&linear, &x)?.to_f16_bits()?,
                linear.forward(&x)?.to_f16_bits()?,
                "batch QMV differs at M={rows}"
            );
        }
        let x = input(8)?;

        let serial = serial_rows(&linear, &x)?;
        let batched = linear.forward(&x)?;
        let serial_bits = serial.to_f16_bits()?;
        let batched_bits = batched.to_f16_bits()?;
        let mismatches = serial_bits
            .iter()
            .zip(&batched_bits)
            .filter(|(left, right)| left != right)
            .count();
        let max_abs = serial_bits
            .iter()
            .zip(&batched_bits)
            .map(|(&left, &right)| {
                (f16::from_bits(left).to_f32() - f16::from_bits(right).to_f32()).abs()
            })
            .fold(0.0f32, f32::max);

        for _ in 0..2 {
            serial_rows(&linear, &x)?.eval()?;
            linear.forward(&x)?.eval()?;
        }
        let repeats = 5;
        let start = Instant::now();
        for _ in 0..repeats {
            serial_rows(&linear, &x)?.eval()?;
        }
        let serial_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(repeats);
        let start = Instant::now();
        for _ in 0..repeats {
            linear.forward(&x)?.eval()?;
        }
        let batched_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(repeats);
        eprintln!(
            "M=8 projection: serial={serial_ms:.3}ms batch_qmv={batched_ms:.3}ms speedup={:.2}x mismatches={mismatches}/{} max_abs={max_abs}",
            serial_ms / batched_ms,
            serial_bits.len()
        );
        assert_eq!(serial.shape(), batched.shape());
        assert_eq!(mismatches, 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen checkpoint and Apple M5 GPU"]
    fn batch_grouped_qmv_shared_rows_match_serial() -> Result<()> {
        let checkpoint =
            crate::checkpoint::inspect(Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        let group = Exl3Group::new(vec![
            Exl3Linear::from_checkpoint(
                &checkpoint,
                "model.language_model.layers.0.linear_attn.in_proj_qkv",
            )?,
            Exl3Linear::from_checkpoint(
                &checkpoint,
                "model.language_model.layers.0.linear_attn.in_proj_z",
            )?,
        ])?;
        for matrix_rows in [6, 8] {
            let values = (0..matrix_rows * group.rows)
                .map(|index| f16::from_f32(((index % 251) as f32 - 125.0) / 128.0).to_bits())
                .collect::<Vec<_>>();
            let x = Array::from_f16_bits(&values, &[matrix_rows, group.rows])?;
            let expected = serial_group_rows(&group, &x)?;
            let actual = group.forward(&x)?;
            for (index, (left, right)) in expected.iter().zip(&actual).enumerate() {
                assert_eq!(
                    left.to_f16_bits()?,
                    right.to_f16_bits()?,
                    "grouped batch M={matrix_rows} output {index} differs"
                );
            }
        }
        let matrix_rows = 8;
        let values = (0..matrix_rows * group.rows)
            .map(|index| f16::from_f32(((index % 251) as f32 - 125.0) / 128.0).to_bits())
            .collect::<Vec<_>>();
        let x = Array::from_f16_bits(&values, &[matrix_rows, group.rows])?;
        for _ in 0..2 {
            for value in serial_group_rows(&group, &x)? {
                value.eval()?;
            }
            for value in group.forward(&x)? {
                value.eval()?;
            }
        }
        let repeats = 5;
        let start = Instant::now();
        for _ in 0..repeats {
            for value in serial_group_rows(&group, &x)? {
                value.eval()?;
            }
        }
        let serial_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(repeats);
        let start = Instant::now();
        for _ in 0..repeats {
            for value in group.forward(&x)? {
                value.eval()?;
            }
        }
        let batched_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(repeats);
        eprintln!(
            "Qwen grouped QMV M=8: serial={serial_ms:.3}ms batch={batched_ms:.3}ms speedup={:.2}x",
            serial_ms / batched_ms
        );
        Ok(())
    }

    #[test]
    fn q36_head_lane_override_restores_and_counts_only_its_scope() {
        assert_eq!(Q36_HEAD_LANE_TEST.with(std::cell::Cell::get), None);
        let ((), outer) = with_q36_head_lane_test(true, || {
            Q36_HEAD_LANE_USES.with(|uses| uses.set(uses.get() + 1));
            let ((), inner) = with_q36_head_lane_test(false, || {
                assert_eq!(Q36_HEAD_LANE_TEST.with(std::cell::Cell::get), Some(false));
                Q36_HEAD_LANE_USES.with(|uses| uses.set(uses.get() + 2));
            });
            assert_eq!(inner, 2);
            assert_eq!(Q36_HEAD_LANE_TEST.with(std::cell::Cell::get), Some(true));
        });
        assert_eq!(outer, 1);
        assert_eq!(Q36_HEAD_LANE_TEST.with(std::cell::Cell::get), None);
        let panicked = std::panic::catch_unwind(|| {
            with_q36_head_lane_test(true, || panic!("temporary test failure"))
        });
        assert!(panicked.is_err());
        assert_eq!(Q36_HEAD_LANE_TEST.with(std::cell::Cell::get), None);
        assert_eq!(Q36_HEAD_LANE_USES.with(std::cell::Cell::get), 0);
    }

    /// Q36-HEAD-LANE-P24 component oracle: the real 35B lm_head with the production
    /// lane path (M=3 H2, M=4 F3) and with stock, both against serial M=1, bit for bit.
    #[test]
    #[ignore = "requires physical M5 and the local 35B (MLXL3_QWEN_TEST_MODEL); run alone"]
    fn q36_head_lane_real_head_matches_serial_and_rejects_mutated_index() -> Result<()> {
        const OUTPUT: usize = 248_320;
        ensure!(array::is_m5_gpu()?, "requires M5");
        let checkpoint =
            crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_QWEN_TEST_MODEL")?))?;
        let mut head = Exl3Linear::from_checkpoint(&checkpoint, "lm_head")?;
        ensure!(
            head.rows == 2048
                && head.cols == OUTPUT as i32
                && head.k == 6
                && head.cb == Codebook::Mcg
                && head.simdgroups == 8
                && head.splits(head.rows / 16, head.cols / 16) == 1
                && !head.reference_accumulation
                && !head.mtp_hidden_batch
                && head.lane_indices.is_none(),
            "unexpected 35B head"
        );
        let prepared = head
            .head_lane_indices
            .as_ref()
            .context("head lane indices were not prepared (MLXL3_Q36_HEAD_LANE set to 0?)")?
            .to_bytes()?;
        let expected = (0..2048u32)
            .flat_map(|i| (crate::contracts::exl3_lane_source_index(i) as i32).to_le_bytes())
            .collect::<Vec<_>>();
        ensure!(prepared == expected, "prepared head lane gather is wrong");
        let input = |rows: i32, seed: usize| {
            let values = (0..(rows * 2048) as usize)
                .map(|i| {
                    f16::from_f32((((i * 37 + seed * 101) % 251) as f32 - 125.0) / 128.0).to_bits()
                })
                .collect::<Vec<_>>();
            Array::from_f16_bits(&values, &[rows, 2048])
        };
        let checked = |value: Array, rows: i32| -> Result<Vec<u16>> {
            let bits = value.to_f16_bits()?;
            ensure!(
                value.shape() == [rows, OUTPUT as i32]
                    && bits.len() == rows as usize * OUTPUT
                    && bits.iter().all(|&v| f16::from_bits(v).is_finite()),
                "invalid head output at M={rows}"
            );
            Ok(bits)
        };
        let mut exact = 0;
        for seed in 0..3 {
            for rows in 1..=6 {
                let x = input(rows, 1700 + seed)?;
                x.eval()?;
                let (serial, serial_uses) =
                    with_q36_head_lane_test(true, || serial_rows(&head, &x));
                let serial = checked(serial?, rows)?;
                let (stock, stock_uses) = with_q36_head_lane_test(false, || head.forward(&x));
                let stock = checked(stock?, rows)?;
                let (lane, lane_uses) = with_q36_head_lane_test(true, || head.forward(&x));
                let lane = checked(lane?, rows)?;
                let selected = usize::from(matches!(rows, 3 | 4));
                ensure!(
                    serial_uses == 0 && stock_uses == 0 && lane_uses == selected,
                    "head lane activation at M={rows}: serial {serial_uses}, stock {stock_uses}, lane {lane_uses}"
                );
                ensure!(
                    stock == serial,
                    "stock head differs from serial at M={rows}"
                );
                ensure!(
                    lane == serial,
                    "lane head differs from serial at M={rows} (seed {seed})"
                );
                exact += 1;
                eprintln!("q36 head lane seed={seed} M={rows} uses={lane_uses} exact");
            }
        }
        // A wrong gather must be visible to the oracle on both selected widths.
        let correct = head
            .head_lane_indices
            .take()
            .context("missing head lane indices")?;
        let mut ids = (0..2048u32)
            .map(|i| crate::contracts::exl3_lane_source_index(i) as i32)
            .collect::<Vec<_>>();
        ids.swap(0, 2);
        head.head_lane_indices = Some(Array::from_i32(&ids, &[2048])?);
        for rows in [3, 4] {
            let x = input(rows, 1900)?;
            let serial = checked(serial_rows(&head, &x)?, rows)?;
            let (mutant, uses) = with_q36_head_lane_test(true, || head.forward(&x));
            let mutant = checked(mutant?, rows)?;
            ensure!(
                uses == 1 && mutant != serial,
                "mutated head lane gather not detected at M={rows}"
            );
            eprintln!("q36 head lane mutated gather detected at M={rows}");
        }
        head.head_lane_indices = Some(correct);
        for rows in [3, 4] {
            let x = input(rows, 1900)?;
            let serial = checked(serial_rows(&head, &x)?, rows)?;
            let (lane, uses) = with_q36_head_lane_test(true, || head.forward(&x));
            ensure!(
                uses == 1 && checked(lane?, rows)? == serial,
                "restored head lane differs at M={rows}"
            );
        }
        ensure!(exact == 18, "incomplete head lane matrix");
        eprintln!("q36 head lane component: {exact}/18 exact, 2/2 mutants detected, restored");
        Ok(())
    }
}

#[path = "linear/lfm3_prefill.rs"]
mod lfm3_prefill;
#[cfg(test)]
#[path = "linear/lfm3_prefill_tests.rs"]
mod lfm3_prefill_tests;
#[cfg(test)]
thread_local! {
    static PREFILL_OVERRIDE: std::cell::Cell<Option<[bool; 2]>> = const { std::cell::Cell::new(None) };
    static PREFILL_DISPATCHES: std::cell::Cell<[usize; 2]> = const { std::cell::Cell::new([0; 2]) };
}

pub(crate) fn lfm3_prefill_options(qualified: bool) -> [bool; 2] {
    #[cfg(test)]
    if let Some(flags) = PREFILL_OVERRIDE.get() {
        return flags.map(|flag| qualified && flag);
    }
    ["MLXL3_LFM3_PREFILL_GEMM", "MLXL3_LFM3_PREFILL_GROUPED"].map(|name| {
        match std::env::var(name) {
            Ok(value) => crate::lfm3::prefill_selected(qualified, Some(&value)),
            Err(std::env::VarError::NotPresent) => crate::lfm3::prefill_selected(qualified, None),
            Err(std::env::VarError::NotUnicode(_)) => false,
        }
    })
}
