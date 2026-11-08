//! One ready wave, sharing fixed-record QMV across experts and token consumers.
use super::*;

impl FlashExpert<'_> {
    pub(in crate::gpu) fn encode_consumer_cohort(
        cmd: &CommandBufferRef,
        x: &MatrixView,
        experts: &[&Self],
        destinations: &[Vec<(&Buffer, u64)>],
    ) -> Result<()> {
        let rows = x.shape()[0];
        crate::expert_ar_policy::consumer_grid(experts.len(), rows)
            .context("invalid expert/consumer grid")?;
        ensure!(
            destinations.len() == experts.len()
                && destinations
                    .iter()
                    .all(|outputs| outputs.len() == rows as usize),
            "incomplete expert/consumer destinations"
        );
        let gpu = experts[0].gate.metal;
        ensure!(
            gpu.expert_consumer_qmv.is_some() && x.shape()[1] == 2560,
            "unsupported expert/consumer grid"
        );
        for expert in experts {
            ensure!(
                std::ptr::eq(gpu, expert.gate.metal)
                    && crate::expert_ar_policy::row_fusions_eligible(
                        expert.routed,
                        expert.expert_extents(expert.gate.view_extents(x)?)?
                    ),
                "invalid expert/consumer extents"
            );
        }
        let n = experts.len() as u64;
        let input_words = rows * 2560;
        let dual = gpu.expert_ar_enabled[1].get();
        let fused = gpu.expert_ar_enabled[0].get();
        let down_fused = fused && gpu.expert_down_prepare.is_some();
        let stride = input_words * if dual { 2 } else { 1 };
        let prepared = buffer(
            &gpu.device,
            n * u64::from(stride) * 2,
            Storage::StorageModePrivate,
        )?;
        let partial = buffer(
            &gpu.device,
            n * u64::from(rows) * 2560 * 4,
            Storage::StorageModePrivate,
        )?;
        for (index, expert) in experts.iter().enumerate() {
            for row in 0..rows {
                let x = x.row(row)?;
                let offset = index as u64 * u64::from(stride) * 2 + u64::from(row) * 5120;
                if dual {
                    Metal::encode_bound(
                        cmd,
                        gpu.expert_gu_prepare
                            .as_ref()
                            .context("dual preparation missing")?,
                        [
                            (&x.buffer, 0),
                            (&expert.gate.suh, expert.gate.offsets.suh),
                            (&expert.up.suh, expert.up.offsets.suh),
                            (&prepared, offset),
                        ]
                        .into_iter(),
                        [
                            x.layout.offset,
                            x.layout.strides[1],
                            i64::from(x.dtype == super::super::view::FloatType::F32),
                            i64::from(input_words),
                        ],
                        MTLSize::new(20, 1, 1),
                        64,
                    );
                    #[cfg(test)]
                    gpu.expert_ar_uses[1].set(gpu.expert_ar_uses[1].get() + 1);
                } else {
                    expert
                        .gate
                        .encode_input_into(cmd, Activation::View(&x), &prepared, offset)?;
                }
            }
        }
        Self::encode_axis_qmv(cmd, experts, 0, &prepared, 0, [stride, rows], &partial)?;
        let mut gates = Vec::with_capacity(experts.len());
        for (index, expert) in experts.iter().enumerate() {
            let gate = buffer(
                &gpu.device,
                u64::from(rows) * 1280,
                Storage::StorageModePrivate,
            )?;
            expert.gate.encode_finish(
                cmd,
                &partial,
                index as u64 * u64::from(rows) * 2560,
                &gate,
                0,
                rows * 640,
                None,
                None,
            )?;
            gates.push(gate);
        }
        if !dual {
            for (index, expert) in experts.iter().enumerate() {
                for row in 0..rows {
                    expert.up.encode_input_into(
                        cmd,
                        Activation::View(&x.row(row)?),
                        &prepared,
                        index as u64 * u64::from(stride) * 2 + u64::from(row) * 5120,
                    )?;
                }
            }
        }
        Self::encode_axis_qmv(
            cmd,
            experts,
            1,
            &prepared,
            if dual { u64::from(input_words) * 2 } else { 0 },
            [stride, rows],
            &partial,
        )?;
        for (index, expert) in experts.iter().enumerate() {
            let offset = index as u64 * u64::from(rows) * 1280;
            if down_fused {
                expert.up.encode_finish(
                    cmd,
                    &partial,
                    index as u64 * u64::from(rows) * 2560,
                    &prepared,
                    offset,
                    rows * 640,
                    Some(&gates[index]),
                    Some((&expert.down.suh, expert.down.offsets.suh)),
                )?;
            } else {
                let up = buffer(
                    &gpu.device,
                    u64::from(rows) * 1280,
                    Storage::StorageModePrivate,
                )?;
                expert.up.encode_finish(
                    cmd,
                    &partial,
                    index as u64 * u64::from(rows) * 2560,
                    &up,
                    0,
                    rows * 640,
                    fused.then_some(&gates[index]),
                    None,
                )?;
                let activated = if fused {
                    up
                } else {
                    gpu.encode_swiglu(cmd, &gates[index], &up, Storage::StorageModePrivate)?
                };
                Metal::encode_bound(
                    cmd,
                    &gpu.prepare128_pipeline,
                    [
                        (&activated, 0),
                        (&expert.down.suh, expert.down.offsets.suh),
                        (&prepared, offset),
                    ]
                    .into_iter(),
                    [640u32],
                    MTLSize::new(u64::from(rows * 5), 1, 1),
                    64,
                );
            }
            #[cfg(test)]
            if fused {
                gpu.expert_ar_uses[0].set(gpu.expert_ar_uses[0].get() + 1);
            }
        }
        Self::encode_axis_qmv(cmd, experts, 2, &prepared, 0, [rows * 640, rows], &partial)?;
        for (index, expert) in experts.iter().enumerate() {
            for (row, &(dest, offset)) in destinations[index].iter().enumerate() {
                expert.down.encode_finish(
                    cmd,
                    &partial,
                    (index as u64 * u64::from(rows) + row as u64) * 10240,
                    dest,
                    offset,
                    2560,
                    None,
                    None,
                )?;
            }
        }
        #[cfg(test)]
        gpu.expert_ar_uses[3].set(gpu.expert_ar_uses[3].get() + n * u64::from(rows));
        Ok(())
    }
}
