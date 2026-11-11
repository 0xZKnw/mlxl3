//! Qualified LFM3 prefill alternatives; flags are off by default.
use super::*;

pub(super) fn reconstruct(linear: &Exl3Linear) -> Result<Array> {
    let elements = crate::lfm3::reconstruction_elements(linear.rows, linear.cols, linear.k)
        .context("unqualified reconstruction geometry")?;
    ensure!(
        linear.cb == Codebook::Mcg,
        "reconstruction screen requires MCG8"
    );
    let words = linear.trellis.reshape(&[-1])?.view(Dtype::UInt32)?;
    ensure!(
        words.shape() == [elements / 4],
        "invalid reconstruction trellis length"
    );
    let inverse = codec::permutation_inverse()
        .map(|v| v.to_string())
        .join(",");
    let header = codebook_header(linear.cb)
        + &format!(
            "\nconstant ushort mlxl3_perm_inv[256] = {{{inverse}}};\n#define OUTPUT_DIMS {}u\n",
            linear.cols
        );
    Ok(array::metal_kernel(
        &format!("mlxl3_lfm3_reconstruct_{}_{}_v1", linear.rows, linear.cols),
        &["trellis"],
        &["weights"],
        &header,
        include_str!("../../shaders/lfm3_reconstruct.metal"),
        &[&words],
        &[vec![linear.rows, linear.cols]],
        &[Dtype::Float16],
        [elements, 1, 1],
        [256, 1, 1],
    )?
    .remove(0))
}

pub(super) fn reconstruct_mm(linear: &Exl3Linear, x: &Array) -> Result<Array> {
    ensure!(
        x.shape().len() == 3
            && x.shape()[0] == 1
            && x.shape()[2] == linear.rows
            && x.shape()[1] >= 24
            && matches!(x.dtype(), Dtype::Float16 | Dtype::Float32)
    );
    #[cfg(test)]
    PREFILL_DISPATCHES.with(|v| {
        let mut counts = v.get();
        counts[0] += 1;
        v.set(counts);
    });
    let rows = x.shape()[1];
    let transformed = x
        .astype(Dtype::Float16)?
        .reshape(&[rows, linear.rows])?
        .mul(&linear.suh.astype(Dtype::Float16)?)?
        .reshape(&[rows, linear.rows / 128, 128])?
        .hadamard_transform(Some(1. / 128.0_f32.sqrt()))?
        .reshape(&[rows, linear.rows])?;
    // Reconstruct on every call: caching expanded weights would change the memory tradeoff.
    let raw = transformed.matmul(&reconstruct(linear)?)?;
    let output = raw
        .reshape(&[rows, linear.cols / 128, 128])?
        .hadamard_transform(Some(1. / 128.0_f32.sqrt()))?
        .reshape(&[1, rows, linear.cols])?
        .mul(&linear.svh.astype(Dtype::Float16)?)?
        .astype(x.dtype())?;
    match &linear.bias {
        Some(bias) => output.add(bias),
        None => Ok(output),
    }
}

pub(super) fn grouped_qmm(group: &Exl3Group, x: &Array) -> Result<Vec<Array>> {
    ensure!(
        x.shape().len() == 3
            && x.shape()[0] == 1
            && x.shape()[2] == group.rows
            && group.k == 8
            && group.cb == Codebook::Mcg
            && x.dtype() == Dtype::Float16
    );
    let rows = x.shape()[1];
    let [padded, output, groups] =
        crate::lfm3::grouped_prefill_layout(rows, group.rows, &group.widths)
            .context("unqualified grouped QMM geometry")?;
    #[cfg(test)]
    PREFILL_DISPATCHES.with(|v| {
        let mut counts = v.get();
        counts[1] += 1;
        v.set(counts);
    });
    let transformed = x
        .reshape(&[1, rows, group.rows])?
        .mul(&group.suh.reshape(&[groups, 1, group.rows])?)?
        .reshape(&[groups, rows, group.rows / 128, 128])?
        .hadamard_transform(Some(1. / 128.0_f32.sqrt()))?
        .reshape(&[groups, rows, group.rows])?;
    let transformed = if rows == padded {
        transformed
    } else {
        Array::concatenate(
            &[
                &transformed,
                &Array::zeros_dtype(&[groups, padded - rows, group.rows], Dtype::Float16)?,
            ],
            1,
        )?
    };
    let inverse = codec::permutation_inverse()
        .map(|v| v.to_string())
        .join(",");
    let end0 = group.widths[0] / 32;
    let end1 = if groups == 3 {
        (group.widths[0] + group.widths[1]) / 32
    } else {
        output / 32
    };
    let header = format!(
        "#include <metal_tensor>\n#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\nusing namespace metal;\nusing namespace mpp;\n{}constant ushort mlxl3_perm_inv[256] = {{{inverse}}};\n#define BM 32u\n#define BN 32u\n#define BK 16u\n#define K_BITS 8u\n#define PACKED_U32 64u\n#define INPUT_DIMS {}u\n#define OUTPUT_DIMS {output}u\n#define TILES_N {}u\n#define WEIGHT_TILE_OFFSET 0u\n#define PADDED_ROWS {padded}u\n#define GROUP0_END {end0}u\n#define GROUP1_END {end1}u\n",
        codebook_header(group.cb),
        group.rows,
        group.cols / 16
    );
    let words = group.trellis.reshape(&[-1])?.view(Dtype::UInt32)?;
    let raw = array::metal_kernel(
        &format!("mlxl3_lfm3_grouped_qmm_{groups}_{padded}_v1"),
        &["xhat", "trellis"],
        &["yhat"],
        &header,
        include_str!("../../shaders/lfm3_grouped_qmm.metal"),
        &[&transformed, &words],
        &[vec![padded, output]],
        &[Dtype::Float16],
        [output, padded / 32, 1],
        [32, 1, 1],
    )?
    .remove(0)
    .slice(0, 0, rows)?;
    let combined = raw
        .reshape(&[rows, output / 128, 128])?
        .hadamard_transform(Some(1. / 128.0_f32.sqrt()))?
        .reshape(&[1, rows, output])?
        .mul(&group.svh)?;
    let mut start = 0;
    group
        .widths
        .iter()
        .zip(&group.biases)
        .map(|(&width, bias)| {
            let value = combined.slice(2, start, start + width)?;
            start += width;
            match bias {
                Some(bias) => value.add(bias),
                None => Ok(value),
            }
        })
        .collect()
}
