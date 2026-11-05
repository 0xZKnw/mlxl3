//! Grouped Qwen MoE stages with disjoint per-job regions in command-owned buffers.
use super::{
    Buffer, CommandBufferRef, Context, FlashProjection, MTLSize, MappedSwitchGlu, MatrixView,
    Metal, Result, Storage, buffer, dispatch_mapped_qmv, ensure, upload,
};
use crate::qwen35_config::{QWEN_STAGE_SHAPES, native_qwen_qmv_binding, native_qwen_stage_offsets};

fn address(buffer: &Buffer, offset: u64, bytes: u64, align: u64) -> Result<u64> {
    native_qwen_qmv_binding(buffer.gpu_address(), offset, bytes, buffer.length(), align)
        .context("invalid Qwen grouped stage binding")
}

fn encode_qmv(
    bank: &MappedSwitchGlu<'_>,
    cmd: &CommandBufferRef,
    ids: &[u32],
    prepared: &Buffer,
    partials: &Buffer,
    down: bool,
) -> Result<()> {
    let gpu = bank.gpu;
    let [jobs, input, output] = QWEN_STAGE_SHAPES[usize::from(down)];
    let plan = if down { bank.down_plan } else { bank.gu_plan };
    let mut descriptors = Vec::with_capacity(jobs as usize);
    let mut reads: Vec<&metal::ResourceRef> = Vec::with_capacity(1 + 2 * jobs as usize);
    reads.push(prepared);
    let mut first: Option<&FlashProjection<'_>> = None;
    for (route, &id) in ids.iter().enumerate() {
        let e = &bank.experts[id as usize];
        let gu = [&e.gate, &e.up];
        let only_down = [&e.down];
        let projections: &[&FlashProjection<'_>] = if down { &only_down } else { &gu };
        for (part, &p) in projections.iter().enumerate() {
            let baseline = *first.get_or_insert(p);
            ensure!(
                [p.input, p.output] == [input, output] && (p.k, p.cb) == (baseline.k, baseline.cb),
                "incompatible Qwen grouped stage projections"
            );
            let job = u32::try_from(route * projections.len() + part)?;
            let [input_at, raw_at] =
                native_qwen_stage_offsets(job, down).context("invalid Qwen job")?;
            descriptors.push([
                address(prepared, input_at, u64::from(input) * 2, 4)?,
                address(
                    &p.packed,
                    p.offsets.trellis,
                    u64::from(input) * u64::from(output) * u64::from(p.k) / 8,
                    4,
                )?,
                address(&p.svh, p.offsets.svh, u64::from(output) * 2, 2)?,
                address(partials, raw_at, u64::from(output) * 4, 4)?,
            ]);
            reads.push(&p.packed);
            reads.push(&p.svh);
        }
    }
    ensure!(
        descriptors.len() == jobs as usize,
        "incomplete Qwen grouped stage"
    );
    dispatch_mapped_qmv(
        gpu,
        cmd,
        first.context("empty Qwen grouped stage")?,
        plan,
        &descriptors,
        [&reads, &[partials]],
    )
}

pub(super) fn encode(
    bank: &MappedSwitchGlu<'_>,
    cmd: &CommandBufferRef,
    x: &MatrixView,
    ids: &[u32],
    stack: &MatrixView,
) -> Result<()> {
    let gpu = bank.gpu;
    let [prepare, glu, finish] = gpu
        .qwen_expert_stages
        .as_ref()
        .context("Qwen grouped stage pipelines unavailable")?;
    ensure!(
        ids.len() == 8
            && x.layout.fits(usize::try_from(x.buffer.length() / 2)?)
            && [prepare, glu, finish]
                .iter()
                .all(|p| p.thread_execution_width() == 32
                    && p.max_total_threads_per_threadgroup() >= 128),
        "invalid Qwen grouped stage input"
    );

    let mut arguments = [[0; 6]; 8];
    let mut read_su: Vec<&metal::ResourceRef> = Vec::with_capacity(16);
    let mut read_glu: Vec<&metal::ResourceRef> = Vec::with_capacity(24);
    let mut read_finish: Vec<&metal::ResourceRef> = Vec::with_capacity(8);
    for (row, &id) in arguments.iter_mut().zip(ids) {
        let e = &bank.experts[id as usize];
        for (i, p) in [&e.gate, &e.up, &e.down].into_iter().enumerate() {
            row[i] = address(&p.suh, p.offsets.suh, u64::from(p.input) * 2, 2)?;
            row[3 + i] = address(&p.svh, p.offsets.svh, u64::from(p.output) * 2, 2)?;
        }
        read_su.push(&e.gate.suh);
        read_su.push(&e.up.suh);
        read_glu.push(&e.gate.svh);
        read_glu.push(&e.up.svh);
        read_glu.push(&e.down.suh);
        read_finish.push(&e.down.svh);
    }
    let arguments = upload(&gpu.device, &arguments)?;
    let allocate = |down: bool, raw: bool| {
        let [jobs, input, output] = QWEN_STAGE_SHAPES[usize::from(down)];
        let bytes = if raw { output * 4 } else { input * 2 };
        buffer(
            &gpu.device,
            u64::from(jobs) * u64::from(bytes),
            Storage::StorageModePrivate,
        )
    };
    let prepared = allocate(false, false)?;
    let gu_partials = allocate(false, true)?;
    let activated = allocate(true, false)?;
    let down_partials = allocate(true, true)?;

    dispatch(
        gpu,
        cmd,
        prepare,
        [&x.buffer, &arguments, &prepared],
        &read_su,
        [x.layout.offset, x.layout.strides[1]],
        [16, 16],
        64,
    );
    encode_qmv(bank, cmd, ids, &prepared, &gu_partials, false)?;
    dispatch(
        gpu,
        cmd,
        glu,
        [&gu_partials, &arguments, &activated],
        &read_glu,
        bank.gu_plan[0],
        [4, 8],
        128,
    );
    encode_qmv(bank, cmd, ids, &activated, &down_partials, true)?;
    dispatch(
        gpu,
        cmd,
        finish,
        [&down_partials, &arguments, &stack.buffer],
        &read_finish,
        bank.down_plan[0],
        [16, 8],
        64,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn dispatch<T: Copy>(
    _gpu: &Metal,
    cmd: &CommandBufferRef,
    pipeline: &metal::ComputePipelineState,
    buffers: [&Buffer; 3],
    reads: &[&metal::ResourceRef],
    params: T,
    [x, y]: [u64; 2],
    threads: u64,
) {
    let encoder = cmd.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (slot, buffer) in buffers.into_iter().enumerate() {
        encoder.set_buffer(slot as u64, Some(buffer), 0);
    }
    encoder.use_resources(reads, metal::MTLResourceUsage::Read);
    encoder.set_bytes(
        3,
        std::mem::size_of::<T>() as u64,
        std::ptr::from_ref(&params).cast(),
    );
    encoder.dispatch_thread_groups(MTLSize::new(x, y, 1), MTLSize::new(threads, 1, 1));
    encoder.end_encoding();
    #[cfg(test)]
    _gpu.qwen_expert_stage_dispatches
        .set(_gpu.qwen_expert_stage_dispatches.get() + 1);
}
