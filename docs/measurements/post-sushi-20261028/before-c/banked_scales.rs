use super::*;

fn scale_rows(x: &Array, bank: &Array, slots: &ExpertSlots) -> Result<Array> {
    let input: [i32; 2] = x
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("banked scale input must have rank two"))?;
    let shape: [i32; 3] = bank
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("banked scales must have rank three"))?;
    let [routes]: [i32; 1] = slots
        .indices
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("banked slots must be flat"))?;
    ensure!(
        x.dtype() == Dtype::Float16
            && bank.dtype() == Dtype::Float16
            && slots.indices.dtype() == Dtype::UInt32,
        "invalid banked scale dtypes"
    );
    let count =
        crate::flash_support::flash_banked_scale_count(input, shape, routes, slots.capacity)
            .context("invalid banked scale geometry")?;
    #[cfg(test)]
    let _profile = crate::flash_profile::span("exl3_banked_scale_build");
    Ok(array::metal_kernel(
        "mlxl3_flash_banked_scale_product",
        &["x", "bank", "slots"],
        &["out"],
        &format!(
            "#define WIDTH {}u\n#define PROJECTIONS {}u\n#define CAPACITY {}u\n",
            shape[2], shape[1], slots.capacity
        ),
        include_str!("../../shaders/_flash_banked_scale.metal"),
        &[&x.reshape(&[-1])?, &bank.reshape(&[-1])?, &slots.indices],
        &[input.to_vec()],
        &[Dtype::Float16],
        [count, 1, 1],
        [128, 1, 1],
    )?
    .remove(0))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn project(
    x: &Array,
    trellis: &Array,
    suh: &Array,
    svh: &Array,
    slots: &ExpertSlots,
    output_dims: i32,
    projections: i32,
    k: usize,
    cb: Codebook,
    stable: bool,
) -> Result<Array> {
    let [rows, input_dims]: [i32; 2] = x
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("expert QMV input must have rank two"))?;
    ensure!(
        trellis.shape().len() == 5 && trellis.shape()[0] == slots.capacity,
        "expert slots were validated for a different bank capacity"
    );
    ensure!(
        slots.groups.is_none() && matches!(x.dtype(), Dtype::Float16 | Dtype::Float32),
        "banked projection requires flat floating-point routes"
    );
    ensure!(
        suh.dtype() == Dtype::Float16
            && svh.dtype() == Dtype::Float16
            && suh.shape() == [slots.capacity, projections, input_dims]
            && svh.shape() == [slots.capacity, projections, output_dims],
        "invalid banked projection scales"
    );
    let [routes]: [i32; 1] = slots
        .indices
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("banked slots must be flat"))?;
    ensure!(
        crate::flash_support::flash_banked_scale_count(
            [rows, input_dims],
            [slots.capacity, projections, input_dims],
            routes,
            slots.capacity
        )
        .is_some()
            && crate::flash_support::flash_banked_scale_count(
                [rows, output_dims],
                [slots.capacity, projections, output_dims],
                routes,
                slots.capacity
            )
            .is_some(),
        "invalid banked projection geometry"
    );
    let transformed = scale_rows(&x.astype(Dtype::Float16)?, suh, slots)?
        .reshape(&[rows, input_dims / 128, 128])?
        .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
        .reshape(&[rows, input_dims])?;
    let raw = expert_mapped_layout(
        &transformed,
        trellis,
        None,
        None,
        &slots.indices,
        output_dims,
        projections,
        0,
        k,
        cb,
        true,
        true,
        true,
        slots.masked,
        None,
        stable,
    )?;
    let rotated = raw
        .astype(Dtype::Float16)?
        .reshape(&[rows, output_dims / 128, 128])?
        .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
        .reshape(&[rows, output_dims])?;
    let output = scale_rows(&rotated, svh, slots)?.astype(x.dtype())?;
    #[cfg(test)]
    crate::flash_profile::stage("exl3_output_hadamard_scale", &output)?;
    Ok(output)
}

#[cfg(test)]
mod tests;
