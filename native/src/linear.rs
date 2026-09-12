//! Rust dispatch of the existing MLXL3 serialized QMV shaders through native MLX.
use crate::{
    array::{self, Array, Dtype},
    checkpoint::Checkpoint,
    codec::{self, Codebook},
};
use anyhow::{Context, Result, ensure};

pub fn checkpoint_array(checkpoint: &Checkpoint, name: &str) -> Result<Array> {
    let info = checkpoint
        .tensors
        .get(name)
        .with_context(|| format!("missing tensor {name}"))?;
    let dtype = match info.dtype.as_str() {
        "F16" => Dtype::Float16,
        "BF16" => Dtype::BFloat16,
        "F32" => Dtype::Float32,
        "U16" => Dtype::UInt16,
        "I16" => Dtype::Int16,
        "U32" => Dtype::UInt32,
        "I32" => Dtype::Int32,
        "BOOL" => Dtype::Bool,
        other => anyhow::bail!("unsupported model tensor dtype {other} ({name})"),
    };
    let shape = info
        .shape
        .iter()
        .map(|&n| i32::try_from(n))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Array::from_bytes(&info.read_bytes()?, &shape, dtype)
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
        Self::new(
            trellis,
            scale("suh", "su")?,
            scale("svh", "sv")?,
            bias,
            k,
            cb,
        )
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
        })
    }

    pub fn input_dims(&self) -> i32 {
        self.rows
    }
    pub fn output_dims(&self) -> i32 {
        self.cols
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
        if matrix_rows >= 24 {
            return self.forward_qmm_tensor(x);
        }
        ensure!(matrix_rows == 1, "native QMM requires at least 24 rows");
        let xhat = x
            .astype(Dtype::Float16)?
            .reshape(&[1, self.rows])?
            .mul(&self.suh.astype(Dtype::Float16)?)?
            .reshape(&[1, self.rows / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[self.rows])?;
        let input_tiles = self.rows / 16;
        let output_tiles = self.cols / 16;
        let mut header = codebook_header(self.cb);
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
            let splits = split_count(input_tiles, output_tiles);
            let nt = if output_tiles >= 1024 {
                if output_tiles % 2 == 0 { 2 } else { 1 }
            } else if output_tiles % 4 == 0 {
                4
            } else if output_tiles % 2 == 0 {
                2
            } else {
                1
            };
            header += &format!(
                "\n#define MLXL3_QMV_NT {nt}u\n#define MLXL3_QMV_SG {}u\n#define MLXL3_K_BITS {}u\n#define MLXL3_FUSE_OUTPUT 0\n#define K {}\n#define CB {}\n#define PACKED_U32 {}\n#define TILES_K {input_tiles}\n#define TILES_N {output_tiles}\n#define N_SPLITS {splits}\n#define OUTPUT_DIMS {}\n",
                self.simdgroups,
                self.k,
                self.k,
                self.cb as u32,
                self.k * 8,
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
    let block_rows = 32;
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
                "mlxl3_rs_qmm_tensor_{input_dims}_{output_dims}_{k}_{}_s{weight_tiles_n}_o{weight_tile_offset}",
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
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Vec<Array>> {
        let elements = x.shape().iter().map(|&n| i64::from(n)).product::<i64>();
        ensure!(
            x.shape().last() == Some(&self.rows) && elements % i64::from(self.rows) == 0,
            "invalid grouped EXL3 input"
        );
        let matrix_rows = i32::try_from(elements / i64::from(self.rows))?;
        if matrix_rows >= 24 {
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
        ensure!(matrix_rows == 1, "grouped QMV expects one token");
        let groups = i32::try_from(self.widths.len())?;
        let xhat = x
            .astype(Dtype::Float16)?
            .reshape(&[1, self.rows])?
            .mul(&self.suh.astype(Dtype::Float16)?)?
            .reshape(&[groups, self.rows / 128, 128])?
            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
            .reshape(&[-1])?;
        let tiles = self.cols / 16;
        let splits = split_count(self.rows / 16, tiles);
        let nt = if tiles >= 1024 {
            if tiles % 2 == 0 { 2 } else { 1 }
        } else if tiles % 4 == 0 {
            4
        } else if tiles % 2 == 0 {
            2
        } else {
            1
        };
        let header = codebook_header(self.cb)
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
    codec::check_k(k)?;
    ensure!(k != 7, "mapped K=7 is not supported");
    let [rows, input_dims]: [i32; 2] = x
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("expert QMV input must have rank 2"))?;
    ensure!(
        trellis.shape().len() == 3
            && trellis.shape()[0] * 16 == input_dims
            && trellis.shape()[2] == (16 * k) as i32,
        "invalid expert trellis"
    );
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
    let output_tiles = output_dims / 16;
    let local_tiles = rows
        .checked_mul(output_tiles)
        .context("expert tile count overflow")?;
    let splits = split_count(trellis.shape()[0], local_tiles);
    let mut nt = if local_tiles >= 1024 {
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
    let m5 = array::is_m5_gpu()?;
    let simdgroups = if m5 && k == 2 {
        4
    } else if m5 {
        8
    } else {
        4
    };
    let header = codebook_header(cb)
        + &format!(
            "\n#define MLXL3_QMV_NT {nt}u\n#define MLXL3_QMV_SG {simdgroups}u\n#define MLXL3_K_BITS {k}u\n#define MLXL3_K3_WINDOW_DECODE {}\n#define K {k}\n#define CB {}\n#define PACKED_U32 {}\n#define INPUT_DIMS {input_dims}\n#define TILES_K {}\n#define TILES_N {}\n#define N_SPLITS {splits}\n#define LOCAL_OUTPUT_DIMS {}\n#define IDENTITY_MAP 0\n#define EXPERT_MAP 1\n#define OUTPUT_TILES {output_tiles}\n#define ROUTING_REPEAT {projections_per_route}\n#define PROJECTION_STRIDE_TILES {projection_stride_tiles}\n",
            u8::from(k == 3),
            cb as u32,
            k * 8,
            trellis.shape()[0],
            trellis.shape()[1],
            local_tiles * 16,
        );
    let words = trellis.reshape(&[-1])?.view(Dtype::UInt32)?;
    let partials = array::metal_kernel(
        &format!(
            "mlxl3_rs_expert_{}_{}_{}_{}_{}_{}_{}",
            input_dims, output_dims, k, cb as u32, nt, simdgroups, splits
        ),
        &["xhat", "trellis", "tile_map", "tile_sub"],
        &["yhat"],
        &header,
        include_str!("../shaders/_qmv_mapped_tile_kernel.metal"),
        &[&xhat, &words, selected, selected],
        &[vec![splits, rows, output_dims]],
        &[Dtype::Float32],
        [
            (local_tiles / nt)
                .checked_mul(simdgroups * 32)
                .context("expert QMV grid overflow")?,
            1,
            splits,
        ],
        [simdgroups * 32, 1, 1],
    )?
    .remove(0);
    let yhat = if splits == 1 {
        partials.reshape(&[rows, output_dims])?
    } else {
        partials.sum(0, false)?
    };
    if return_raw {
        return Ok(yhat);
    }
    yhat.astype(Dtype::Float16)?
        .reshape(&[rows, output_dims / 128, 128])?
        .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
        .reshape(&[rows, output_dims])?
        .mul(svh.expect("validated output scales"))?
        .astype(x.dtype())
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

fn codebook_header(cb: Codebook) -> String {
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
