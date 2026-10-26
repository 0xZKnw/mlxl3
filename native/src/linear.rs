//! Rust dispatch of the existing MLXL3 serialized QMV shaders through native MLX.
use crate::{
    array::{self, Array, Dtype},
    checkpoint::Checkpoint,
    codec::{self, Codebook, reference_reduction_header},
};
use anyhow::{Context, Result, ensure};

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

fn lane_order_input(
    xhat: Array,
    indices: Option<&Array>,
    matrix_rows: i32,
    groups: i32,
    input: i32,
) -> Result<(Array, bool)> {
    if let Some(indices) = indices.filter(|_| lane_order_enabled() && matches!(matrix_rows, 1 | 2))
    {
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

fn checkpoint_dtype(dtype: &str, name: &str) -> Result<Dtype> {
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
    lane_indices: Option<Array>,
}

impl Exl3Linear {
    pub fn from_checkpoint(checkpoint: &Checkpoint, prefix: &str) -> Result<Self> {
        let trellis = checkpoint_array(checkpoint, &format!("{prefix}.trellis"))?;
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
        if checkpoint.model_type == "qwen4_exp" {
            linear.simdgroups = 4;
            linear.reference_accumulation = true;
            // Flash keeps its reference layout; lane order was qualified on other models.
            linear.lane_indices = None;
            linear.flash_batch = crate::flash_support::multi_token_verifier()?;
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
            lane_indices: lane_order_indices(rows, cols, k, cb, false)?,
        })
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
        if crate::contracts::use_tensor_ops(matrix_rows, array::is_m5_gpu()?) {
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
        if matrix_rows > 1 {
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
        let xhat = x
            .astype(Dtype::Float16)?
            .reshape(&[1, self.rows])?
            .mul(&self.suh.astype(Dtype::Float16)?)?
            .reshape(&[1, self.rows / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[self.rows])?;
        let (xhat, lane_order) =
            lane_order_input(xhat, self.lane_indices.as_ref(), 1, 1, self.rows)?;
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

    fn forward_qmv_batch(&self, x: &Array, matrix_rows: i32) -> Result<Array> {
        ensure!(
            self.k != 7 && (2..24).contains(&matrix_rows),
            "batch QMV requires 2 to 23 rows"
        );
        let xhat = x
            .astype(Dtype::Float16)?
            .reshape(&[matrix_rows, self.rows])?
            .mul(&self.suh.astype(Dtype::Float16)?)?
            .reshape(&[matrix_rows, self.rows / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[matrix_rows, self.rows])?;
        let (xhat, lane_order) =
            lane_order_input(xhat, self.lane_indices.as_ref(), matrix_rows, 1, self.rows)?;
        let input_tiles = self.rows / 16;
        let output_tiles = self.cols / 16;
        // The released MTP FC uses Pony's four-row GEMM: two groups and two splits.
        let splits = if self.mtp_hidden_batch {
            2
        } else {
            self.splits(input_tiles, output_tiles)
        };
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
            crate::contracts::flash_reference_batch_layout(matrix_rows)
                .context("unsupported Flash reference batch")?
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
        let partials = array::metal_kernel(
            &format!(
                "mlxl3_rs_tile_batch_{}_{}_{}_{}_{}_{}_{}_{}_{}",
                self.rows,
                self.cols,
                self.k,
                self.cb as u32,
                matrix_rows,
                nt,
                mb,
                self.simdgroups,
                splits
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
    ensure!(array::is_m5_gpu()?, "TensorOps QMM requires Apple M5");
    let block_columns = 32;
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
    ensure!(matrix_rows >= 24, "TensorOps QMM requires at least 24 rows");
    static DENSE_PREFILL_M64: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let enabled = *DENSE_PREFILL_M64
        .get_or_init(|| std::env::var("MLXL3_DENSE_PREFILL_M64").as_deref() != Ok("0"));
    let block_rows = crate::contracts::dense_qmm_block_rows(
        matrix_rows,
        input_dims,
        output_dims,
        enabled && cb == Codebook::Mul1,
    );
    let padded_rows = ((matrix_rows + block_rows - 1) / block_rows) * block_rows;
    let xhat = x
        .astype(Dtype::Float16)?
        .reshape(&[matrix_rows, input_dims])?
        .mul(&suh.astype(Dtype::Float16)?)?
        .reshape(&[matrix_rows, input_dims / 128, 128])?
        .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
        .reshape(&[matrix_rows, input_dims])?;
    let xhat = if matrix_rows == padded_rows {
        xhat
    } else {
        Array::concatenate(
            &[
                &xhat,
                &Array::zeros_dtype(&[padded_rows - matrix_rows, input_dims], Dtype::Float16)?,
            ],
            0,
        )?
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
        .remove(0)
        .slice(0, 0, matrix_rows)?;
    let output = raw
        .reshape(&[matrix_rows, output_dims / 128, 128])?
        .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
        .reshape(&[matrix_rows, output_dims])?
        .mul(&svh.astype(Dtype::Float16)?)?
        .astype(x.dtype())?;
    let mut shape = x.shape().to_vec();
    *shape.last_mut().context("empty QMM input shape")? = output_dims;
    output.reshape(&shape)
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
        })
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
        if crate::contracts::use_tensor_ops(matrix_rows, array::is_m5_gpu()?) {
            let mut tile_cursor = 0;
            let mut scale_cursor = 0;
            let mut outputs = Vec::with_capacity(self.widths.len());
            for (index, (width, bias)) in self.widths.iter().zip(&self.biases).enumerate() {
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
        if matrix_rows > 1 {
            return self.forward_qmv_batch(x, matrix_rows);
        }
        let groups = i32::try_from(self.widths.len())?;
        let xhat = x
            .astype(Dtype::Float16)?
            .reshape(&[1, self.rows])?
            .mul(&self.suh.astype(Dtype::Float16)?)?
            .reshape(&[groups, self.rows / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[-1])?;
        let (xhat, lane_order) =
            lane_order_input(xhat, self.lane_indices.as_ref(), 1, groups, self.rows)?;
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
        let yhat = if splits == 1 {
            partials.reshape(&[self.cols])?
        } else {
            partials.sum(0, false)?
        };
        let output = yhat
            .astype(Dtype::Float16)?
            .reshape(&[1, self.cols / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[self.cols])?
            .mul(&self.svh.astype(Dtype::Float16)?)?
            .astype(x.dtype())?;
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

    fn forward_qmv_batch(&self, x: &Array, matrix_rows: i32) -> Result<Vec<Array>> {
        ensure!(
            (2..24).contains(&matrix_rows),
            "grouped batch QMV requires 2 to 23 rows"
        );
        let groups = i32::try_from(self.widths.len())?;
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
        let (xhat, lane_order) = lane_order_input(
            xhat,
            self.lane_indices.as_ref(),
            matrix_rows,
            groups,
            self.rows,
        )?;
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
        if mb == 2 {
            nt = nt.min(2);
        }
        if mb == 3 {
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
                "mlxl3_rs_grouped_batch_{}_{}_{}_{}_{}_{}_{}_{}",
                self.rows, self.cols, self.k, self.cb as u32, nt, mb, self.simdgroups, splits,
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
        let yhat = if splits == 1 {
            partials.reshape(&[matrix_rows, self.cols])?
        } else {
            partials.sum(1, false)?
        };
        let output = yhat
            .astype(Dtype::Float16)?
            .reshape(&[matrix_rows, self.cols / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[matrix_rows, self.cols])?
            .mul(&self.svh.astype(Dtype::Float16)?)?
            .astype(x.dtype())?;
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
        if let Some(profile) = crate::flash_profile::span("expert_kernel_signature") {
            profile.data(serde_json::json!({
                "stable": stable_shapes,
                "key": [group.map_or_else(|| name.clone(), |(width, _)| format!("{name}_group{width}")),
                    grouped_header.as_deref().unwrap_or(&header).to_owned(),
                    stable_source.as_deref().unwrap_or(kernel_source).to_owned()],
            }));
        }
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
    let target = if output_tiles <= 64 {
        (output_tiles / 2).max(32)
    } else {
        output_tiles.clamp(32, 256)
    };
    let mut splits = 1;
    while input_tiles / (splits * 2) >= target {
        splits *= 2;
    }
    splits
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

    #[test]
    #[ignore = "physical Apple GPU; all sixty real QSA projections, independent scalar oracle"]
    fn flash_qsa_batches_match_independent_single_rows() -> anyhow::Result<()> {
        use super::*;
        let checkpoint =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut names = Vec::new();
        for layer in 0..48 {
            let prefix = format!("model.language_model.layers.{layer}.self_attn");
            if checkpoint
                .tensors
                .contains_key(&format!("{prefix}.q_proj.trellis"))
            {
                for name in [
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "indexer.index_qk_proj",
                ] {
                    names.push(format!("{prefix}.{name}"));
                }
            }
        }
        ensure!(names.len() == 60, "did not select all QSA projections");
        let mut cases = 0;
        for name in &names {
            let linear = Exl3Linear::from_checkpoint(&checkpoint, name)?;
            ensure!(
                linear.reference_accumulation
                    && linear.flash_batch
                    && linear.k != 7
                    && linear.cb == Codebook::Mul1
            );
            ensure!(
                linear.k
                    == if name.ends_with("index_qk_proj") {
                        2
                    } else {
                        4
                    }
            );
            for dtype in [Dtype::Float16, Dtype::Float32] {
                for strided in [false, true] {
                    let value = |r: i32, c: i32| ((r * 37 + c * 13) % 997) as f32 / 1024. - 0.5;
                    let x = if strided {
                        let values = (0..linear.rows)
                            .flat_map(|c| (0..2).map(move |r| value(r, c)))
                            .collect::<Vec<_>>();
                        Array::from_f32(&values, &[linear.rows, 2])?
                            .astype(dtype)?
                            .transpose(&[1, 0])?
                    } else {
                        let values = (0..2)
                            .flat_map(|r| (0..linear.rows).map(move |c| value(r, c)))
                            .collect::<Vec<_>>();
                        Array::from_f32(&values, &[2, linear.rows])?.astype(dtype)?
                    };
                    let first = linear.forward(&x.slice(0, 0, 1)?)?;
                    let second = linear.forward(&x.slice(0, 1, 2)?)?;
                    let expected = Array::concatenate(&[&first, &second], 0)?.to_bytes()?;
                    let actual = linear.forward(&x)?;
                    let values = actual.to_f32()?;
                    ensure!(
                        actual.shape() == [2, linear.cols]
                            && values.len() == (2 * linear.cols) as usize
                            && values.iter().all(|v| v.is_finite())
                    );
                    ensure!(
                        actual.to_bytes()? == expected,
                        "QSA MB2 differs: {name} {dtype:?} strided={strided}"
                    );
                    ensure!(
                        first.to_bytes()? != second.to_bytes()?,
                        "identical rows give a vacuous oracle"
                    );
                    cases += 1;
                }
            }
        }
        println!("QSA_MB2_ORACLE projections=60 cases={cases} parity=byte_exact finite=true");
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
}
