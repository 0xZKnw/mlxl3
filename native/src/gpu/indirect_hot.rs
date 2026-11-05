//! Intralayer GPU cache lookup. The canonical projection/reduction bodies are unchanged.
use super::{device_hot_cache::ResidentSnapshot, *};
use std::{cell::Cell, rc::Rc};

pub(super) struct Pipelines {
    stages: [ComputePipelineState; 5],
    argument: metal::ArgumentEncoder,
    dummy: Buffer,
    active: Rc<Cell<bool>>,
    #[cfg(test)]
    submissions: Cell<u64>,
    #[cfg(test)]
    wrong_binding: Cell<bool>,
    #[cfg(test)]
    reject_result: Cell<bool>,
}

impl Pipelines {
    pub(super) fn load(
        device: &DeviceRef,
        library: &metal::Library,
        pipeline: &impl Fn(&str) -> Result<ComputePipelineState>,
        enabled: bool,
    ) -> Result<Option<Self>> {
        if !enabled {
            return Ok(None);
        }
        let stages = [
            pipeline("device_hot_gu_prepare")?,
            pipeline("device_hot_qmv_gu")?,
            pipeline("device_hot_qmv_down")?,
            pipeline("device_hot_finish")?,
            pipeline("device_hot_down_prepare")?,
        ];
        let function = library
            .get_function("device_hot_qmv_gu", None)
            .map_err(anyhow::Error::msg)?;
        let argument = function.new_argument_encoder(1);
        let length = argument.encoded_length();
        let alignment = argument.alignment();
        if length == 0
            || length > 65536
            || !alignment.is_power_of_two()
            || stages.iter().any(|p| {
                p.thread_execution_width() != 32 || p.max_total_threads_per_threadgroup() < 128
            })
        {
            return Ok(None);
        }
        Ok(Some(Self {
            stages,
            argument,
            dummy: buffer(
                device,
                crate::flash_support::EXPERT_BYTES,
                Storage::StorageModeShared,
            )?,
            active: Rc::new(Cell::new(false)),
            #[cfg(test)]
            submissions: Cell::new(0),
            #[cfg(test)]
            wrong_binding: Cell::new(false),
            #[cfg(test)]
            reject_result: Cell::new(false),
        }))
    }
}

struct Active(Rc<Cell<bool>>);
impl Drop for Active {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

/// H is deliberately not inserted into FlashMoeRow::last. Dropping it releases
/// its retained references after the final consumer has checked every producer.
pub(super) struct HotOutput {
    command: metal::CommandBuffer,
    final_consumer: Option<metal::CommandBuffer>,
    snapshot: ResidentSnapshot,
    slab: Buffer,
    _resources: Vec<Buffer>,
    _pipelines: [ComputePipelineState; 5],
    #[cfg(test)]
    reject_result: bool,
    _active: Active,
}

impl HotOutput {
    /// The MoE caller creates F on H's queue, attaches it, then commits it.
    /// Keeping F here preserves public MoeOutput field moves while bounding abandonment.
    pub(super) fn retain_consumer(&mut self, command: &CommandBufferRef) -> Result<()> {
        ensure!(
            self.final_consumer.is_none(),
            "resident consumer already attached"
        );
        ensure!(
            command.status() == MTLCommandBufferStatus::NotEnqueued,
            "resident consumer must be attached before submission"
        );
        ensure!(
            !std::ptr::eq::<CommandBufferRef>(self.command.as_ref(), command),
            "resident producer cannot be its own consumer"
        );
        self.final_consumer = Some(command.to_owned());
        Ok(())
    }
    #[cfg(test)]
    pub(super) fn inject_terminal_failure(&mut self) {
        self.reject_result = true;
    }
    pub(super) fn snapshot(&self) -> &ResidentSnapshot {
        &self.snapshot
    }

    /// GPU-only view: callers enqueue consumption on the same queue after H.
    pub(super) fn contribution(&self, route: usize) -> Result<(&Buffer, u64)> {
        let offset = crate::device_hot_policy::contribution_offset(route)
            .context("invalid resident contribution route")?;
        Ok((&self.slab, offset))
    }

    pub(super) fn completed(&self) -> Result<()> {
        ensure!(
            self.command.status() != MTLCommandBufferStatus::NotEnqueued,
            "resident producer was never submitted"
        );
        if !matches!(
            self.command.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        ) {
            self.command.wait_until_completed();
        }
        ensure!(
            self.command.status() == MTLCommandBufferStatus::Completed,
            "resident producer failed ({:?})",
            self.command.status()
        );
        #[cfg(test)]
        ensure!(!self.reject_result, "injected resident publication failure");
        Ok(())
    }
}

impl Drop for HotOutput {
    fn drop(&mut self) {
        if let Some(consumer) = &self.final_consumer
            && matches!(
                consumer.status(),
                MTLCommandBufferStatus::Committed | MTLCommandBufferStatus::Scheduled
            )
        {
            consumer.wait_until_completed();
        }
        if !matches!(
            self.command.status(),
            MTLCommandBufferStatus::NotEnqueued
                | MTLCommandBufferStatus::Completed
                | MTLCommandBufferStatus::Error
        ) {
            self.command.wait_until_completed();
        }
    }
}

// Every encoder declares its indirect reads; binding the argument buffer alone
// does not tell Metal which of the separate record allocations it can access.
#[allow(clippy::too_many_arguments)]
fn encode<T: Copy, const N: usize>(
    command: &CommandBufferRef,
    pipeline: &ComputePipelineState,
    bindings: &[(&Buffer, u64)],
    params: [T; N],
    groups: MTLSize,
    threads: u64,
    resources: &[&metal::ResourceRef],
) {
    let encoder = command.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pipeline);
    for (index, (buffer, offset)) in bindings.iter().enumerate() {
        encoder.set_buffer(index as u64, Some(buffer), *offset);
    }
    encoder.use_resources(resources, metal::MTLResourceUsage::Read);
    encoder.set_bytes(
        bindings.len() as u64,
        std::mem::size_of_val(&params) as u64,
        params.as_ptr().cast(),
    );
    encoder.dispatch_thread_groups(groups, MTLSize::new(threads, 1, 1));
    encoder.end_encoding();
}

impl Metal {
    pub(super) fn device_resident_route_enabled(&self) -> bool {
        self.indirect_hot.is_some()
            && self.routed_nt2_gu.get()
            && self.routed_nt2_down.get()
            && self.expert_ar_enabled[0].get()
            && self.expert_ar_enabled[1].get()
            && self.expert_ar_enabled[3].get()
    }

    pub(super) fn device_resident_route_available(&self) -> bool {
        self.device_resident_route_enabled()
            && self.indirect_hot.as_ref().is_some_and(|p| !p.active.get())
    }

    #[cfg(test)]
    pub(super) fn device_hot_submissions(&self) -> u64 {
        self.indirect_hot
            .as_ref()
            .map_or(0, |p| p.submissions.get())
    }

    pub(super) fn submit_resident_hot(
        &self,
        input: &MatrixView,
        ids: &Buffer,
        snapshot: ResidentSnapshot,
    ) -> Result<HotOutput> {
        self.submit_resident_hot_inner(input, ids, snapshot, Storage::StorageModePrivate)
    }

    #[cfg(test)]
    pub(super) fn submit_resident_hot_for_test(
        &self,
        input: &MatrixView,
        ids: &Buffer,
        snapshot: ResidentSnapshot,
    ) -> Result<HotOutput> {
        self.submit_resident_hot_inner(input, ids, snapshot, Storage::StorageModeShared)
    }

    fn submit_resident_hot_inner(
        &self,
        input: &MatrixView,
        ids: &Buffer,
        snapshot: ResidentSnapshot,
        storage: Storage,
    ) -> Result<HotOutput> {
        objc::rc::autoreleasepool(|| {
            ensure!(
                self.device_resident_route_available(),
                "resident route is disabled, busy or unsupported"
            );
            ensure!(
                input.dtype == view::FloatType::F32
                    && input.shape() == [1, 2560]
                    && input.layout.offset == 0
                    && input.layout.strides[1] == 1
                    && input.buffer.length() >= 10240
                    && ids.length() == 40
                    && input.buffer.device().registry_id() == self.device.registry_id()
                    && ids.device().registry_id() == self.device.registry_id(),
                "invalid resident route input/ID buffer"
            );
            ensure!(
                snapshot.records().len() == 53,
                "resident snapshot must contain53slots"
            );
            for &slot in snapshot.map() {
                ensure!(
                    slot == u32::MAX || slot < 53 && snapshot.records()[slot as usize].is_some(),
                    "resident map names an empty or out-of-range slot"
                );
            }
            for record in snapshot.records().iter().flatten() {
                ensure!(
                    record.length() == crate::flash_support::EXPERT_BYTES
                        && record.device().registry_id() == self.device.registry_id(),
                    "invalid resident record owner"
                );
            }
            let p = self
                .indirect_hot
                .as_ref()
                .context("resident pipelines missing")?;
            let auxiliary = 281600u64
                .checked_add(2048)
                .and_then(|n| n.checked_add(p.argument.encoded_length()))
                .and_then(|n| n.checked_add(crate::flash_support::EXPERT_BYTES))
                .context("resident auxiliary budget overflow")?;
            let peak = crate::device_hot_policy::checked_peak(10, auxiliary)
                .context("resident transaction exceeds reserved budget")?;
            ensure!(
                peak <= crate::device_hot_policy::RESERVE_BYTES,
                "resident transaction exceeds admission reserve"
            );
            p.active.set(true);
            let active = Active(p.active.clone());
            let argument = buffer(
                &self.device,
                p.argument.encoded_length(),
                Storage::StorageModeShared,
            )?;
            p.argument.set_argument_buffer(&argument, 0);
            let records = snapshot
                .records()
                .iter()
                .map(|r| r.as_ref().unwrap_or(&p.dummy).clone())
                .collect::<Vec<_>>();
            for (slot, record) in records.iter().enumerate() {
                #[cfg(test)]
                let record = if p.wrong_binding.get() {
                    &records[(slot + 1) % 53]
                } else {
                    record
                };
                p.argument.set_buffer(slot as u64, record, 0);
            }
            let resources = records
                .iter()
                .map(|r| -> &metal::ResourceRef { r })
                .collect::<Vec<_>>();
            let map = upload(&self.device, snapshot.map())?;
            let prepared = buffer(&self.device, 102400, Storage::StorageModePrivate)?;
            let partial = buffer(&self.device, 102400, Storage::StorageModePrivate)?;
            let gate = buffer(&self.device, 12800, Storage::StorageModePrivate)?;
            let activated = buffer(&self.device, 12800, Storage::StorageModePrivate)?;
            let slab = buffer(&self.device, 51200, storage)?;
            let offsets = (0..3)
                .map(|i| {
                    crate::flash_support::record_offsets(i)
                        .context("invalid resident record offsets")
                })
                .collect::<Result<Vec<_>>>()?;
            let command = self.queue.new_command_buffer().to_owned();
            encode(
                &command,
                &p.stages[0],
                &[
                    (&input.buffer, 0),
                    (&argument, 0),
                    (ids, 0),
                    (&map, 0),
                    (&prepared, 0),
                ],
                [offsets[1].suh, offsets[2].suh],
                MTLSize::new(20, 10, 1),
                64,
                &resources,
            );
            for (stage, width, offset) in [(1usize, 640u32, 0u64), (2, 640, 5120), (0, 2560, 0)] {
                if stage == 0 {
                    encode(
                        &command,
                        &p.stages[4],
                        &[
                            (&activated, 0),
                            (&argument, 0),
                            (ids, 0),
                            (&map, 0),
                            (&prepared, 0),
                        ],
                        [offsets[0].suh],
                        MTLSize::new(5, 10, 1),
                        64,
                        &resources,
                    );
                }
                let geometry = if stage == 0 {
                    [640u32, 2560]
                } else {
                    [2560u32, 640]
                };
                encode(
                    &command,
                    &p.stages[if stage == 0 { 2 } else { 1 }],
                    &[
                        (&prepared, offset),
                        (&argument, 0),
                        (ids, 0),
                        (&map, 0),
                        (&partial, 0),
                    ],
                    [
                        1u32,
                        2,
                        2,
                        geometry[0],
                        geometry[1],
                        u32::try_from(offsets[stage].trellis)?,
                        u32::try_from(offsets[stage].svh)?,
                        5120,
                    ],
                    MTLSize::new(
                        u64::from(
                            contracts::flash_reference_groups(width, 2)
                                .context("invalid resident QMV groups")?,
                        ),
                        1,
                        10,
                    ),
                    128,
                    &resources,
                );
                let dest = match stage {
                    1 => &gate,
                    2 => &activated,
                    _ => &slab,
                };
                encode(
                    &command,
                    &p.stages[3],
                    &[
                        (&partial, 0),
                        (&argument, 0),
                        (ids, 0),
                        (&map, 0),
                        (dest, 0),
                        (&gate, 0),
                    ],
                    [
                        width,
                        u32::try_from(offsets[stage].svh)?,
                        u32::from(stage == 2),
                        0,
                    ],
                    MTLSize::new(u64::from(width / 128), 10, 1),
                    64,
                    &resources,
                );
            }
            drop(resources);
            let mut owned = records;
            owned.extend([
                argument,
                map,
                prepared,
                partial,
                gate,
                activated,
                input.buffer.clone(),
                ids.clone(),
            ]);
            let pipelines = p.stages.clone();
            command.commit();
            #[cfg(test)]
            p.submissions.set(p.submissions.get() + 1);
            Ok(HotOutput {
                command,
                final_consumer: None,
                snapshot,
                slab,
                _resources: owned,
                _pipelines: pipelines,
                #[cfg(test)]
                reject_result: p.reject_result.get(),
                _active: active,
            })
        })
    }
}

#[cfg(test)]
#[path = "ar34_indirect_hot_tests.rs"]
mod tests;
