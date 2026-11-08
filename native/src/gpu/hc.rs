use super::CachedPipeline as ComputePipelineState;
use super::{MatrixView, Metal, view::FloatType};
use crate::checkpoint::Checkpoint;
#[cfg(test)]
use anyhow::bail;
use anyhow::{Context, Result, ensure};
#[cfg(test)]
use half::bf16;
#[cfg(test)]
use half::f16;
use metal::{
    Buffer, CommandBuffer, MTLCommandBufferStatus, MTLResourceOptions as Storage, MTLSize,
};

#[path = "hc_workspace.rs"]
mod workspace;
use std::{cell::RefCell, rc::Rc};

#[cfg(test)]
thread_local! {
    pub(super) static FAIL_NEXT_PENDING_HC_ROUTE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(super) struct Pipelines {
    norm: ComputePipelineState,
    narrow: ComputePipelineState,
    wide: ComputePipelineState,
    activate: ComputePipelineState,
    mix: ComputePipelineState,
    write: ComputePipelineState,
    up_mix: Option<ComputePipelineState>,
    down_inject: Option<ComputePipelineState>,
    write_norm: Option<ComputePipelineState>,
}
impl Pipelines {
    pub(super) fn load(
        pipeline: &impl Fn(&str) -> Result<ComputePipelineState>,
        suffix: &str,
        up_mix: bool,
        down_inject: bool,
        write_norm: bool,
    ) -> Result<Self> {
        let stage = |name: &str| pipeline(&format!("hc_{name}_{suffix}"));
        Ok(Self {
            norm: stage("norm")?,
            narrow: stage("narrow")?,
            wide: stage("wide")?,
            activate: stage("activate")?,
            mix: stage("mix")?,
            write: stage("write")?,
            up_mix: (suffix == "float" && up_mix)
                .then(|| stage("up_mix"))
                .transpose()?,
            down_inject: (suffix == "float" && down_inject)
                .then(|| stage("down_inject"))
                .transpose()?,
            write_norm: (suffix == "float" && write_norm)
                .then(|| pipeline("hc_write_norm_float"))
                .transpose()?,
        })
    }
}

pub struct FlashHc<'a> {
    pub(super) gpu: &'a Metal,
    gain: Buffer,
    down: Buffer,
    up: Buffer,
    inject: Option<Buffer>,
    eps: f32,
    workspace: Rc<RefCell<workspace::Cache>>,
}

pub struct FlashHcRead {
    pub mixed: MatrixView,
    pub gate: Option<MatrixView>,
    pub trace: Option<FlashHcTrace>,
}

// These views remain owned until the enclosing command has completed.
pub(super) struct ReadBuffers {
    normalized: MatrixView,
    down: MatrixView,
    active: MatrixView,
    up: Option<MatrixView>,
    _raw_gate: Option<MatrixView>,
    mixed: MatrixView,
    gate: Option<MatrixView>,
    _workspace: Option<workspace::Lease>,
}
impl ReadBuffers {
    pub(super) fn injection_gate(&self) -> Result<&MatrixView> {
        self.gate.as_ref().context("missing HC injection gate")
    }
    pub(super) fn mixed(&self) -> &MatrixView {
        &self.mixed
    }
    pub(super) fn finish(self, trace: bool) -> Result<FlashHcRead> {
        let Self {
            normalized,
            down,
            active,
            up,
            _raw_gate,
            mixed,
            gate,
            _workspace,
        } = self;
        Ok(FlashHcRead {
            mixed,
            gate,
            trace: trace
                .then(|| -> Result<_> {
                    Ok(FlashHcTrace {
                        normalized: normalized.read_output()?,
                        down: down.read_output()?,
                        active: active.read_output()?,
                        up: up
                            .context("HC trace requires materialized up")?
                            .read_output()?,
                    })
                })
                .transpose()?,
        })
    }
}

/// GPU-only borrowing is valid on the producer queue; host publication waits for P.
pub(super) struct PendingHcRoute<'a> {
    gpu: &'a Metal,
    command: CommandBuffer,
    submitted: bool,
    buffers: Option<(MatrixView, ReadBuffers, super::gate::GateBuffers)>,
    #[cfg(test)]
    injected_failure: bool,
}

impl PendingHcRoute<'_> {
    pub(super) fn mixed(&self) -> &MatrixView {
        &self
            .buffers
            .as_ref()
            .expect("owned pending HC route")
            .1
            .mixed
    }

    pub(super) fn routes(&self) -> &super::RouteOutput {
        &self
            .buffers
            .as_ref()
            .expect("owned pending HC route")
            .2
            .routes
    }

    fn drain(&self) {
        let terminal = matches!(
            self.command.status(),
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        );
        if crate::device_hot_policy::pending_route_needs_wait(self.submitted, terminal) {
            #[cfg(test)]
            self.gpu.helper_waits.set(self.gpu.helper_waits.get() + 1);
            self.gpu.wait_command(&self.command);
        }
    }

    pub(super) fn complete(mut self) -> Result<(MatrixView, FlashHcRead, super::RouteOutput)> {
        #[cfg(test)]
        let _scope = crate::flash_profile::span("native.route.pending_complete");
        self.drain();
        ensure!(
            crate::device_hot_policy::pending_route_publishable(
                self.submitted,
                self.command.status() == MTLCommandBufferStatus::Completed,
            ),
            "pending HC route command failed ({:?})",
            self.command.status()
        );
        #[cfg(test)]
        ensure!(!self.injected_failure, "injected pending HC route failure");
        #[cfg(test)]
        crate::flash_profile::native_gpu_interval(&self.command, "pending-route");
        let (written, read, gate) = self.buffers.take().context("missing pending HC route")?;
        Ok((written, read.finish(false)?, gate.routes))
    }
}

impl Drop for PendingHcRoute<'_> {
    fn drop(&mut self) {
        // Drain before the owned views/workspace are released, including error exits.
        self.drain();
    }
}

#[derive(serde::Serialize)]
pub struct FlashHcTrace {
    normalized: Vec<u32>,
    down: Vec<u32>,
    active: Vec<u32>,
    up: Vec<u32>,
}

fn groups(n: u32) -> MTLSize {
    MTLSize::new(u64::from(n), 1, 1)
}

impl Metal {
    pub(super) fn encode_stream_norm(
        &self,
        cmd: &metal::CommandBufferRef,
        x: &MatrixView,
        gain: &Buffer,
        eps: f32,
        storage: Storage,
    ) -> Result<MatrixView> {
        self.encode_stream_norm_into(cmd, x, gain, eps, storage, None)
    }
    fn encode_stream_norm_into(
        &self,
        cmd: &metal::CommandBufferRef,
        x: &MatrixView,
        gain: &Buffer,
        eps: f32,
        storage: Storage,
        destination: Option<&MatrixView>,
    ) -> Result<MatrixView> {
        let [rows, width] = x.shape();
        ensure!(
            rows == 1
                && matches!(width, 2560 | 10240)
                && x.buffer.device().registry_id() == self.device.registry_id()
                && gain.length() == u64::from(width) * 4,
            "invalid Flash stream normalization"
        );
        let pipeline = &self.hc[usize::from(x.dtype == FloatType::F32)].norm;
        ensure!(
            pipeline.thread_execution_width() == 32
                && pipeline.max_total_threads_per_threadgroup() >= 640,
            "HC needs SIMD32 and 640 threads for normalization"
        );
        let normalized = if let Some(view) = destination {
            ensure!(
                view.shape() == [1, width]
                    && view.dtype == x.dtype
                    && view.buffer.device().registry_id() == self.device.registry_id()
                    && view.layout.offset == 0
                    && view.layout.strides == [i64::from(width), 1],
                "invalid HC normalized workspace"
            );
            view.clone()
        } else {
            MatrixView::output(self, x.dtype, 1, width, storage)?
        };
        Metal::encode(
            cmd,
            pipeline,
            &[&x.buffer, gain, &normalized.buffer],
            [
                x.layout.offset,
                x.layout.strides[1],
                i64::from(eps.to_bits()),
            ],
            groups(width / 2560),
            640,
        );
        Ok(normalized)
    }
}
impl<'a> FlashHc<'a> {
    pub fn load(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        eps: f32,
        combine: bool,
    ) -> Result<Self> {
        ensure!(
            c.model_type == "qwen4_exp" && eps.is_finite() && eps > 0.,
            "invalid Flash HC configuration"
        );
        let weight = |part: &str, shape: &[usize], dtype| {
            super::checkpoint_float_weight(gpu, c, &format!("{prefix}.{part}.weight"), shape, dtype)
        };
        let dense = |part: &str, n, k| weight(part, &[n, k], FloatType::F16);
        Ok(Self {
            gpu,
            eps,
            workspace: Rc::default(),
            gain: weight("hc_norm", &[10240], FloatType::F32)?,
            down: dense("input_mix_weight_down", 320, 10240)?,
            up: dense("input_mix_weight_up", 10240, 320)?,
            inject: combine
                .then(|| dense("block_inject_weight", 4, 10240))
                .transpose()?,
        })
    }

    fn pipeline(&self, x: &MatrixView, width: u32) -> Result<&Pipelines> {
        ensure!(
            x.shape() == [1, width]
                && x.buffer.device().registry_id() == self.gpu.device.registry_id(),
            "invalid Flash HC input shape or device"
        );
        Ok(&self.gpu.hc[usize::from(x.dtype == FloatType::F32)])
    }

    pub fn read(&self, x: &MatrixView, trace: bool) -> Result<FlashHcRead> {
        #[cfg(test)]
        let _scope = crate::flash_profile::span("native.hc.read_separate");
        self.gpu
            .command(|cmd| self.encode_read(cmd, x, trace))?
            .finish(trace)
    }

    pub(super) fn encode_after_write(
        &self,
        cmd: &metal::CommandBufferRef,
        streams: &MatrixView,
        branch: &MatrixView,
        gate: &MatrixView,
    ) -> Result<(MatrixView, ReadBuffers)> {
        let gpu = self.gpu;
        gpu.check_hc_write(streams, branch, gate)?;
        if !crate::flash_support::hc_write_norm_eligible(
            gpu.hc_write_norm.get(),
            streams.dtype == FloatType::F32,
            gpu.hc_workspace_enabled.get(),
        ) {
            let written = gpu.encode_hc_write(cmd, streams, branch, gate)?;
            let read = self.encode_read(cmd, &written, false)?;
            return Ok((written, read));
        }
        let pipeline = gpu.hc[1]
            .write_norm
            .as_ref()
            .context("HC write/norm PSO was not loaded")?;
        ensure!(
            pipeline.thread_execution_width() == 32
                && pipeline.max_total_threads_per_threadgroup() >= 640,
            "unsupported HC write/norm geometry"
        );
        #[cfg(test)]
        let scope = crate::flash_profile::span("native.hc.encode_write");
        let written =
            MatrixView::output(gpu, FloatType::F32, 1, 10240, Storage::StorageModeShared)?;
        let normalized = {
            #[cfg(test)]
            let _scope = crate::flash_profile::span("native.hc.workspace.norm");
            MatrixView::output(gpu, FloatType::F32, 1, 10240, Storage::StorageModePrivate)?
        };
        Metal::encode(
            cmd,
            pipeline,
            &[
                &streams.buffer,
                &branch.buffer,
                &gate.buffer,
                &self.gain,
                &written.buffer,
                &normalized.buffer,
            ],
            [
                streams.layout.offset,
                streams.layout.strides[1],
                branch.layout.offset,
                branch.layout.strides[1],
                gate.layout.offset,
                gate.layout.strides[1],
                i64::from(branch.dtype == FloatType::F32),
                i64::from(self.eps.to_bits()),
            ],
            groups(4),
            640,
        );
        #[cfg(test)]
        {
            gpu.hc_write_norm_uses.set(gpu.hc_write_norm_uses.get() + 1);
            drop(scope);
        }
        let read = self.encode_read_prepared(cmd, &written, false, Some(normalized))?;
        Ok((written, read))
    }

    pub(super) fn encode_read(
        &self,
        cmd: &metal::CommandBufferRef,
        x: &MatrixView,
        trace: bool,
    ) -> Result<ReadBuffers> {
        self.encode_read_prepared(cmd, x, trace, None)
    }

    fn encode_read_prepared(
        &self,
        cmd: &metal::CommandBufferRef,
        x: &MatrixView,
        trace: bool,
        prepared_norm: Option<MatrixView>,
    ) -> Result<ReadBuffers> {
        #[cfg(test)]
        let _scope = crate::flash_profile::span("native.hc.encode_read");
        let p = self.pipeline(x, 10240)?;
        let storage = if trace {
            Storage::StorageModeShared
        } else {
            Storage::StorageModePrivate
        };
        let fused_up = crate::contracts::flash_hc_kernel_eligible(
            self.gpu.hc_up_mix.get(),
            trace,
            x.dtype == FloatType::F32,
        );
        let workspace = if self.gpu.hc_workspace_enabled.get() && !trace {
            workspace::Cache::acquire(&self.workspace, self.gpu, cmd, x.dtype, !fused_up)?
        } else {
            None
        };
        let cached = workspace.as_ref().map(|(_, views)| views);
        let output = |w, mode| MatrixView::output(self.gpu, x.dtype, 1, w, mode);
        #[cfg(test)]
        let norm_scope = crate::flash_profile::span("native.hc.workspace.norm");
        let normalized = if let Some(normalized) = prepared_norm {
            ensure!(
                !trace
                    && cached.is_none()
                    && normalized.shape() == [1, 10240]
                    && normalized.dtype == x.dtype
                    && normalized.layout.offset == 0
                    && normalized.layout.strides[1] == 1
                    && normalized.buffer.device().registry_id() == self.gpu.device.registry_id(),
                "invalid precomputed HC normalization"
            );
            normalized
        } else {
            self.gpu.encode_stream_norm_into(
                cmd,
                x,
                &self.gain,
                self.eps,
                storage,
                cached.map(|v| &v.normalized),
            )?
        };
        #[cfg(test)]
        drop(norm_scope);
        let gemv = |v: &MatrixView, weight: &Buffer, n: u32| -> Result<MatrixView> {
            #[cfg(test)]
            let _scope = crate::flash_profile::span(if n == 4 {
                "native.hc.result.gate_gemv"
            } else {
                "native.hc.workspace.gemv"
            });
            let width = v.shape()[1];
            let (count, threads) = crate::contracts::flash_hc_dispatch(width, n)
                .context("invalid Flash HC matrix dimensions")?;
            let pipeline = if width == 320 { &p.wide } else { &p.narrow };
            ensure!(
                pipeline.thread_execution_width() == 32
                    && pipeline.max_total_threads_per_threadgroup() >= u64::from(threads),
                "unsupported Flash HC GEMV geometry"
            );
            let out = match (n, cached) {
                (320, Some(v)) => v.down.clone(),
                (10240, Some(v)) => v.up.as_ref().context("missing HC up workspace")?.clone(),
                _ => output(n, storage)?,
            };
            Metal::encode(
                cmd,
                pipeline,
                &[&v.buffer, weight, &out.buffer],
                [n],
                groups(count),
                threads as usize,
            );
            Ok(out)
        };
        let activate = |v: &MatrixView, gate: bool| -> Result<MatrixView> {
            #[cfg(test)]
            let _scope = crate::flash_profile::span(if gate {
                "native.hc.result.gate_activation"
            } else {
                "native.hc.workspace.activation"
            });
            let n = v.shape()[1];
            let a = if !gate && let Some(v) = cached {
                v.active.clone()
            } else {
                output(
                    n,
                    if gate {
                        Storage::StorageModeShared
                    } else {
                        storage
                    },
                )?
            };
            Metal::encode(
                cmd,
                &p.activate,
                &[&v.buffer, &a.buffer],
                [n, u32::from(gate)],
                groups(n.div_ceil(32)),
                32,
            );
            Ok(a)
        };
        let fused_down = self.inject.is_some()
            && crate::contracts::flash_hc_kernel_eligible(
                self.gpu.hc_down_inject.get(),
                trace,
                x.dtype == FloatType::F32,
            );
        let (down, raw_gate) = if fused_down {
            let pipeline = p
                .down_inject
                .as_ref()
                .context("HC down/inject pipeline not loaded")?;
            let (count, threads) = crate::contracts::flash_hc_down_inject_dispatch(10240, 320, 4)
                .context("invalid joint HC dimensions")?;
            ensure!(
                pipeline.thread_execution_width() == 32
                    && pipeline.max_total_threads_per_threadgroup() >= u64::from(threads),
                "unsupported joint HC geometry"
            );
            let down = match cached {
                Some(v) => v.down.clone(),
                None => output(320, storage)?,
            };
            let gate = output(4, storage)?;
            Metal::encode(
                cmd,
                pipeline,
                &[
                    &normalized.buffer,
                    &self.down,
                    self.inject.as_ref().context("missing HC injection")?,
                    &down.buffer,
                    &gate.buffer,
                ],
                [320u32, 4],
                groups(count),
                threads as usize,
            );
            #[cfg(test)]
            self.gpu
                .hc_down_inject_uses
                .set(self.gpu.hc_down_inject_uses.get() + 1);
            (down, Some(gate))
        } else {
            (gemv(&normalized, &self.down, 320)?, None)
        };
        let active = activate(&down, false)?;
        let up = if fused_up {
            None
        } else {
            Some(gemv(&active, &self.up, 10240)?)
        };
        #[cfg(test)]
        let mixed_scope = crate::flash_profile::span("native.hc.result.mixed");
        let mixed = output(2560, Storage::StorageModeShared)?;
        #[cfg(test)]
        drop(mixed_scope);
        if fused_up {
            let pipeline = p.up_mix.as_ref().context("HC up/mix pipeline not loaded")?;
            let (count, threads) = crate::contracts::flash_hc_up_mix_dispatch(320, 10240, 2560)
                .context("invalid fused HC dimensions")?;
            ensure!(
                pipeline.thread_execution_width() == 32
                    && pipeline.max_total_threads_per_threadgroup() >= u64::from(threads),
                "unsupported fused HC geometry"
            );
            Metal::encode(
                cmd,
                pipeline,
                &[&active.buffer, &self.up, &normalized.buffer, &mixed.buffer],
                [2560u32],
                groups(count),
                threads as usize,
            );
            #[cfg(test)]
            self.gpu
                .hc_up_mix_uses
                .set(self.gpu.hc_up_mix_uses.get() + 1);
        } else {
            let up = up.as_ref().context("missing HC up output")?;
            Metal::encode(
                cmd,
                &p.mix,
                &[&up.buffer, &normalized.buffer, &mixed.buffer],
                [2560u32],
                groups(80),
                32,
            );
        }
        let gate = match raw_gate.as_ref() {
            Some(raw) => Some(activate(raw, true)?),
            None => self
                .inject
                .as_ref()
                .map(|w| activate(&gemv(&normalized, w, 4)?, true))
                .transpose()?,
        };
        Ok(ReadBuffers {
            normalized,
            down,
            active,
            up,
            _raw_gate: raw_gate,
            mixed,
            gate,
            _workspace: workspace.map(|(lease, _)| lease),
        })
    }

    pub(super) fn read_after_write(
        &self,
        writer: &FlashHc,
        streams: &MatrixView,
        branch: &MatrixView,
        gate: &MatrixView,
    ) -> Result<(MatrixView, FlashHcRead)> {
        ensure!(
            self.inject.is_some(),
            "Flash MLP requires an injection gate"
        );
        if !self.gpu.hc_chain.get() {
            let written = writer.write(streams, branch, gate)?;
            let read = self.read(&written, false)?;
            return Ok((written, read));
        }
        #[cfg(test)]
        let _scope = crate::flash_profile::span("native.hc.chain_write_read");
        ensure!(
            std::ptr::eq(self.gpu, writer.gpu),
            "HC chain requires the same Metal instance"
        );
        let (written, read) = self.gpu.command(|cmd| {
            let (written, read) = self.encode_after_write(cmd, streams, branch, gate)?;
            #[cfg(test)]
            self.gpu
                .hc_chain_pairs
                .set(self.gpu.hc_chain_pairs.get() + 1);
            Ok((written, read))
        })?;
        Ok((written, read.finish(false)?))
    }

    pub(super) fn read_after_write_with_gate(
        &self,
        writer: &FlashHc,
        streams: &MatrixView,
        branch: &MatrixView,
        gate: &MatrixView,
        moe: &super::FlashMoeRow,
    ) -> Result<(MatrixView, FlashHcRead, super::RouteOutput)> {
        #[cfg(test)]
        let _scope = crate::flash_profile::span("native.hc.chain_write_read_gate");
        ensure!(
            self.inject.is_some(),
            "Flash MLP requires an injection gate"
        );
        ensure!(moe.joint_gate_eligible(0), "ineligible joint HC gate");
        ensure!(
            std::ptr::eq(self.gpu, writer.gpu),
            "HC chain requires the same Metal instance"
        );
        let (written, read, gate) = self.gpu.command(|cmd| {
            let (written, read) = self.encode_after_write(cmd, streams, branch, gate)?;
            let gate = moe.encode_gate(self.gpu, cmd, &read.mixed)?;
            #[cfg(test)]
            {
                self.gpu
                    .hc_chain_pairs
                    .set(self.gpu.hc_chain_pairs.get() + 1);
                self.gpu
                    .hc_gate_chain_pairs
                    .set(self.gpu.hc_gate_chain_pairs.get() + 1);
            }
            Ok((written, read, gate))
        })?;
        Ok((written, read.finish(false)?, gate.routes))
    }

    pub(super) fn submit_after_write_with_gate(
        &self,
        writer: &FlashHc,
        streams: &MatrixView,
        branch: &MatrixView,
        gate: &MatrixView,
        moe: &super::FlashMoeRow,
    ) -> Result<PendingHcRoute<'a>> {
        ensure!(moe.device_hot_gate_eligible(0), "resident producer is busy");
        self.submit_after_write_with_gate_on(
            self.gpu.queue.new_command_buffer().to_owned(),
            writer,
            streams,
            branch,
            gate,
            moe,
        )
    }

    /// The private caller creates this command on `self.gpu.queue`.
    pub(super) fn submit_after_write_with_gate_on(
        &self,
        command: CommandBuffer,
        writer: &FlashHc,
        streams: &MatrixView,
        branch: &MatrixView,
        gate: &MatrixView,
        moe: &super::FlashMoeRow,
    ) -> Result<PendingHcRoute<'a>> {
        #[cfg(test)]
        let scope = crate::flash_profile::span("native.hc.chain_write_read_gate");
        #[cfg(test)]
        if let Some(span) = &scope {
            span.data(serde_json::json!({"submission_only":true}));
        }
        objc::rc::autoreleasepool(|| {
            ensure!(
                self.inject.is_some()
                    && !self.gpu.hc_workspace_enabled.get()
                    && streams.dtype == FloatType::F32
                    && moe.device_hot_gate_eligible_for_encoding(0),
                "ineligible pending HC route"
            );
            ensure!(
                std::ptr::eq(self.gpu, writer.gpu),
                "HC chain requires the same Metal instance"
            );
            ensure!(
                command.status() == MTLCommandBufferStatus::NotEnqueued,
                "pending HC route requires an unsubmitted command"
            );
            #[cfg(test)]
            self.gpu
                .helper_commands
                .set(self.gpu.helper_commands.get() + 1);
            let mut pending = PendingHcRoute {
                gpu: self.gpu,
                command,
                submitted: false,
                buffers: None,
                #[cfg(test)]
                injected_failure: FAIL_NEXT_PENDING_HC_ROUTE.with(|failure| failure.replace(false)),
            };
            let (written, read) =
                self.encode_after_write(&pending.command, streams, branch, gate)?;
            let routes = moe.encode_gate(self.gpu, &pending.command, &read.mixed)?;
            pending.buffers = Some((written, read, routes));
            pending.command.commit();
            pending.submitted = true;
            #[cfg(test)]
            {
                let gpu = self.gpu;
                gpu.helper_commits.set(gpu.helper_commits.get() + 1);
                gpu.hc_chain_pairs.set(gpu.hc_chain_pairs.get() + 1);
                gpu.hc_gate_chain_pairs
                    .set(gpu.hc_gate_chain_pairs.get() + 1);
            }
            Ok(pending)
        })
    }

    /// Read the four streams, run the ordinary demand-loaded MLP and inject its residual.
    pub fn forward_moe(
        &self,
        streams: &MatrixView,
        moe: &mut super::FlashMoeRow,
        trace: bool,
    ) -> Result<(MatrixView, super::MoeOutput, FlashHcRead)> {
        self.forward_moe_lookahead(streams, moe, trace, None, 0)
    }
    pub(super) fn forward_moe_lookahead<'b>(
        &self,
        streams: &MatrixView,
        moe: &mut super::FlashMoeRow<'b>,
        trace: bool,
        next: Option<&mut super::FlashMoeRow<'b>>,
        depth: usize,
    ) -> Result<(MatrixView, super::MoeOutput, FlashHcRead)> {
        ensure!(
            self.inject.is_some(),
            "Flash MLP requires an injection gate"
        );
        let read = self.read(streams, trace)?;
        self.forward_moe_read(streams, moe, read, next, depth)
    }
    pub(super) fn forward_moe_read<'b>(
        &self,
        streams: &MatrixView,
        moe: &mut super::FlashMoeRow<'b>,
        read: FlashHcRead,
        next: Option<&mut super::FlashMoeRow<'b>>,
        depth: usize,
    ) -> Result<(MatrixView, super::MoeOutput, FlashHcRead)> {
        let experts = moe.lookahead(&read.mixed, next, depth)?;
        let written = self.write(
            streams,
            experts.completed()?,
            read.gate.as_ref().context("missing MLP injection")?,
        )?;
        Ok((written, experts, read))
    }
    pub fn write(
        &self,
        streams: &MatrixView,
        branch: &MatrixView,
        gate: &MatrixView,
    ) -> Result<MatrixView> {
        #[cfg(test)]
        let _scope = crate::flash_profile::span("native.hc.write_separate");
        self.gpu
            .command(|cmd| self.encode_write(cmd, streams, branch, gate))
    }
    pub(super) fn encode_write(
        &self,
        cmd: &metal::CommandBufferRef,
        streams: &MatrixView,
        branch: &MatrixView,
        gate: &MatrixView,
    ) -> Result<MatrixView> {
        self.gpu.encode_hc_write(cmd, streams, branch, gate)
    }
}

impl Metal {
    fn check_hc_write(
        &self,
        streams: &MatrixView,
        branch: &MatrixView,
        gate: &MatrixView,
    ) -> Result<()> {
        for (view, width) in [(streams, 10240), (branch, 2560), (gate, 4)] {
            ensure!(
                view.shape() == [1, width]
                    && view.buffer.device().registry_id() == self.device.registry_id(),
                "invalid Flash HC input shape or device"
            );
        }
        ensure!(
            gate.dtype == streams.dtype,
            "injection gate dtype disagrees with its streams"
        );
        Ok(())
    }

    pub(super) fn encode_hc_write(
        &self,
        cmd: &metal::CommandBufferRef,
        streams: &MatrixView,
        branch: &MatrixView,
        gate: &MatrixView,
    ) -> Result<MatrixView> {
        #[cfg(test)]
        let _scope = crate::flash_profile::span("native.hc.encode_write");
        self.check_hc_write(streams, branch, gate)?;
        let p = &self.hc[usize::from(streams.dtype == FloatType::F32)];
        let dtype = if branch.dtype == FloatType::F32 {
            FloatType::F32
        } else {
            streams.dtype
        };
        let out = MatrixView::output(self, dtype, 1, 10240, Storage::StorageModeShared)?;
        Metal::encode(
            cmd,
            &p.write,
            &[&streams.buffer, &branch.buffer, &gate.buffer, &out.buffer],
            [
                streams.layout.offset,
                streams.layout.strides[1],
                branch.layout.offset,
                branch.layout.strides[1],
                gate.layout.offset,
                gate.layout.strides[1],
                i64::from(branch.dtype == FloatType::F32),
            ],
            groups(320),
            32,
        );
        Ok(out)
    }
}

impl FlashHc<'_> {
    pub(super) fn record_memory(&self, backings: &mut crate::gpu::memory::Backings) -> Result<()> {
        for buffer in [&self.gain, &self.down, &self.up]
            .into_iter()
            .chain(&self.inject)
        {
            backings.record("fixed_weights", buffer)?;
        }
        self.workspace.borrow().record_memory(backings)
    }
    pub(super) fn release_workspace(&self) -> Result<()> {
        self.workspace.borrow_mut().clear()
    }
}

#[cfg(test)]
#[path = "hc_workspace_tests.rs"]
mod workspace_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_layout::MatrixLayout;
    use serde_json::Value;

    #[test]
    #[ignore = "physical AR34 pending producer ownership, abandonment and exact canonical HC/router; exclusive GPU"]
    fn physical_ar34_pending_hc_route_drains_before_release_and_rejects_publication_failure()
    -> Result<()> {
        let gpu = Metal::new()?;
        let waits = gpu.helper_waits.get();
        let command = gpu.queue.new_command_buffer().to_owned();
        let unsubmitted = || PendingHcRoute {
            gpu: &gpu,
            command: command.clone(),
            submitted: false,
            buffers: None,
            injected_failure: false,
        };
        drop(unsubmitted());
        ensure!(command.status() == MTLCommandBufferStatus::NotEnqueued);
        ensure!(unsubmitted().complete().is_err());
        ensure!(gpu.helper_waits.get() == waits);

        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let writer = FlashHc::load(
            &gpu,
            &c,
            "model.language_model.layers.0.attn_hyper_connection",
            1e-6,
            true,
        )?;
        let reader = FlashHc::load(
            &gpu,
            &c,
            "model.language_model.layers.0.mlp_hyper_connection",
            1e-6,
            true,
        )?;
        let moe =
            super::super::FlashMoeRow::load(&gpu, &c, "model.language_model.layers.0.mlp", 53)?;
        ensure!(moe.device_hot_gate_eligible(0));
        let row = |n| {
            gpu.f32_storage(
                &(0..n)
                    .map(|i| ((i * 17 % 97) as f32 - 48.) / 128.)
                    .collect::<Vec<_>>(),
            )?
            .matrix(MatrixLayout {
                shape: [1, n as u32],
                strides: [n as i64, 1],
                offset: 0,
            })
        };
        let streams = row(10240)?;
        let branch = row(2560)?;
        let gate = row(4)?;
        let (written, read, routes) =
            reader.read_after_write_with_gate(&writer, &streams, &branch, &gate, &moe)?;
        let expected = [
            written.read_output()?,
            read.mixed.read_output()?,
            read.gate
                .context("canonical injection gate")?
                .read_output()?,
            routes.completed_scores()?.read_output()?,
        ];
        let expected_ids = routes.indices()?;
        ensure!(
            expected.iter().map(Vec::len).eq([10240, 2560, 4, 10])
                && expected
                    .iter()
                    .flatten()
                    .all(|v| f32::from_bits(*v).is_finite())
                && expected_ids.len() == 10
        );

        let commits = gpu.helper_commits.get();
        gpu.hc_workspace_enabled.set(true);
        ensure!(
            reader
                .submit_after_write_with_gate(&writer, &streams, &branch, &gate, &moe)
                .is_err()
        );
        ensure!(gpu.helper_commits.get() == commits);
        gpu.hc_workspace_enabled.set(false);

        let before_waits = gpu.helper_waits.get();
        let pending =
            reader.submit_after_write_with_gate(&writer, &streams, &branch, &gate, &moe)?;
        ensure!(gpu.helper_waits.get() == before_waits);
        ensure!(pending.routes().device_indices().length() == 40);
        let retained = pending.mixed().clone();
        let producer = pending.command.clone();
        drop(pending);
        ensure!(producer.status() == MTLCommandBufferStatus::Completed);
        ensure!(retained.read_output()? == expected[1]);

        FAIL_NEXT_PENDING_HC_ROUTE.with(|failure| failure.set(true));
        let failed =
            reader.submit_after_write_with_gate(&writer, &streams, &branch, &gate, &moe)?;
        let failed_command = failed.command.clone();
        ensure!(failed.complete().is_err());
        ensure!(failed_command.status() == MTLCommandBufferStatus::Completed);

        let pending =
            reader.submit_after_write_with_gate(&writer, &streams, &branch, &gate, &moe)?;
        let (written, read, routes) = pending.complete()?;
        let actual = [
            written.read_output()?,
            read.mixed.read_output()?,
            read.gate.context("pending injection gate")?.read_output()?,
            routes.completed_scores()?.read_output()?,
        ];
        ensure!(actual == expected && routes.indices()? == expected_ids);
        eprintln!(
            "AR34_PENDING_HC_ROUTE {{\"parity\":true,\"finite_f32_words\":12814,\"route_ids\":10,\"unsubmitted\":true,\"workspace_rejected_before_commit\":true,\"abandoned\":true,\"retained\":true,\"publication_failure\":true,\"recovery\":true,\"all_producers_joined\":true}}"
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical HC write/read chain, exact values and actual command count"]
    fn joint_write_read_keeps_buffers_values_and_rejects_invalid_inputs() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        let writer = FlashHc::load(
            &gpu,
            &c,
            "model.language_model.layers.0.attn_hyper_connection",
            1e-6,
            true,
        )?;
        let reader = FlashHc::load(
            &gpu,
            &c,
            "model.language_model.layers.0.mlp_hyper_connection",
            1e-6,
            true,
        )?;
        let row = |width: usize, half: bool, strided: bool| -> Result<MatrixView> {
            let stride = if strided { 2 } else { 1 };
            let offset = i64::from(strided);
            let values: Vec<_> = (0..width * stride + strided as usize)
                .map(|i| ((i * 17 % 97) as f32 - 48.) / 128.)
                .collect();
            let data = if half {
                gpu.f16_storage(
                    &values
                        .iter()
                        .map(|&v| f16::from_f32(v).to_bits())
                        .collect::<Vec<_>>(),
                )?
            } else {
                gpu.f32_storage(&values)?
            };
            data.matrix(MatrixLayout {
                shape: [1, width as u32],
                strides: [(width * stride) as i64, stride as i64],
                offset,
            })
        };
        let mut words = 0;
        for strided in [false, true] {
            for half_branch in [false, true] {
                let streams = row(10240, false, strided)?;
                let branch = row(2560, half_branch, strided)?;
                let gate = row(4, false, strided)?;
                gpu.hc_chain.set(false);
                let before = gpu.helper_commands.get();
                let (written, read) = reader.read_after_write(&writer, &streams, &branch, &gate)?;
                ensure!(gpu.helper_commands.get() - before == 2);
                let expected = [
                    written.export_bytes(&gpu)?,
                    read.mixed.export_bytes(&gpu)?,
                    read.gate
                        .context("missing eager gate")?
                        .export_bytes(&gpu)?,
                ];
                gpu.hc_chain.set(true);
                let before = gpu.helper_commands.get();
                let pairs = gpu.hc_chain_pairs.get();
                let (written, read) = reader.read_after_write(&writer, &streams, &branch, &gate)?;
                ensure!(
                    gpu.helper_commands.get() - before == 1
                        && gpu.hc_chain_pairs.get() - pairs == 1
                );
                let actual = [
                    written.export_bytes(&gpu)?,
                    read.mixed.export_bytes(&gpu)?,
                    read.gate
                        .context("missing joint gate")?
                        .export_bytes(&gpu)?,
                ];
                for ((a, b), count) in actual.iter().zip(&expected).zip([10240, 2560, 4]) {
                    ensure!(
                        a == b
                            && a.len() == count * 4
                            && a.as_chunks::<4>()
                                .0
                                .iter()
                                .all(|v| f32::from_le_bytes(*v).is_finite())
                    );
                    words += count;
                }
                let bad = row(3, false, false)?;
                let pairs = gpu.hc_chain_pairs.get();
                ensure!(
                    reader
                        .read_after_write(&writer, &streams, &branch, &bad)
                        .is_err()
                        && gpu.hc_chain_pairs.get() == pairs
                );
                let (again, _) = reader.read_after_write(&writer, &streams, &branch, &gate)?;
                ensure!(again.export_bytes(&gpu)? == expected[0]);
            }
        }
        ensure!(words == 51_216);
        let other = Metal::new()?;
        other.hc_chain.set(true);
        let foreign = FlashHc::load(
            &other,
            &c,
            "model.language_model.layers.0.mlp_hyper_connection",
            1e-6,
            true,
        )?;
        ensure!(
            foreign
                .read_after_write(
                    &writer,
                    &row(10240, false, false)?,
                    &row(2560, false, false)?,
                    &row(4, false, false)?
                )
                .is_err()
        );
        eprintln!(
            "HC chain:4cases,51216 finite F32 words exact, actual helper commands2->1, strided/mixed dtype/error/recovery/owner checks passed"
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical joint HC gate owner rejection before commit and recovery"]
    fn joint_hc_gate_rejects_foreign_moe_before_commit_and_recovers() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        gpu.hc_chain.set(true);
        gpu.hc_gate_chain.set(true);
        let writer = FlashHc::load(
            &gpu,
            &c,
            "model.language_model.layers.0.attn_hyper_connection",
            1e-6,
            true,
        )?;
        let reader = FlashHc::load(
            &gpu,
            &c,
            "model.language_model.layers.0.mlp_hyper_connection",
            1e-6,
            true,
        )?;
        let moe =
            super::super::FlashMoeRow::load(&gpu, &c, "model.language_model.layers.0.mlp", 10)?;
        let view = |n| {
            gpu.f32_storage(&vec![0.0; n])?.matrix(MatrixLayout {
                shape: [1, n as u32],
                strides: [n as i64, 1],
                offset: 0,
            })
        };
        let streams = view(10240)?;
        let branch = view(2560)?;
        let gate = view(4)?;
        let (expected, read) = reader.read_after_write(&writer, &streams, &branch, &gate)?;
        let expected_mixed = read.mixed.read_output()?;
        let other = Metal::new()?;
        other.hc_chain.set(true);
        other.hc_gate_chain.set(true);
        let foreign =
            super::super::FlashMoeRow::load(&other, &c, "model.language_model.layers.0.mlp", 10)?;
        let commits = gpu.helper_commits.get();
        ensure!(
            reader
                .read_after_write_with_gate(&writer, &streams, &branch, &gate, &foreign)
                .is_err()
        );
        ensure!(gpu.helper_commits.get() == commits);
        let (actual, read, routing) =
            reader.read_after_write_with_gate(&writer, &streams, &branch, &gate, &moe)?;
        ensure!(
            actual.export_bytes(&gpu)? == expected.export_bytes(&gpu)?
                && read.mixed.read_output()? == expected_mixed
        );
        ensure!(
            routing.indices()?.len() == 10
                && routing
                    .completed_scores()?
                    .read_output()?
                    .iter()
                    .all(|&b| f32::from_bits(b).is_finite())
        );
        Ok(())
    }

    #[test]
    fn hc_weights_decode_supported_storage_and_reject_integer_weights() -> Result<()> {
        use std::io::Write;
        let values = [-0.0f32, -1.5, 0.125, 32.0];
        for dtype in ["F16", "BF16", "F32", "I32"] {
            let data: Vec<u8> = match dtype {
                "F16" => values
                    .iter()
                    .flat_map(|&x| f16::from_f32(x).to_le_bytes())
                    .collect(),
                "BF16" => values
                    .iter()
                    .flat_map(|&x| bf16::from_f32(x).to_le_bytes())
                    .collect(),
                _ => values.iter().flat_map(|&x| x.to_le_bytes()).collect(),
            };
            let header = serde_json::to_vec(
                &serde_json::json!({"weight":{"dtype":dtype,"shape":[4],"data_offsets":[0,data.len()]}}),
            )?;
            let mut file = tempfile::NamedTempFile::new()?;
            file.write_all(&(header.len() as u64).to_le_bytes())?;
            file.write_all(&header)?;
            file.write_all(&data)?;
            let tensors = crate::checkpoint::read_header(file.path())?;
            let decoded = super::super::float_values(&tensors["weight"]);
            if dtype == "I32" {
                assert!(decoded.is_err());
            } else {
                assert_eq!(
                    decoded?.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
                    values.map(f32::to_bits)
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "independent Flash HC write capture and physical Apple GPU"]
    fn frozen_residual_writes_preserve_mixed_precision_rounding() -> Result<()> {
        let root = std::path::PathBuf::from(
            std::env::var_os("MLXL3_HC_FROZEN").context("missing frozen HC writes")?,
        );
        let c = crate::checkpoint::inspect(std::path::Path::new(
            &std::env::var_os("MLXL3_HC_MODEL").context("missing model")?,
        ))?;
        let spec: Value = serde_json::from_slice(&std::fs::read(root.join("reference.json"))?)?;
        ensure!(spec["status"] == "complete");
        let cases = spec["hc_write"].as_array().context("missing writes")?;
        ensure!(cases.len() == 6);
        let gpu = Metal::new()?;
        let hc = FlashHc::load(
            &gpu,
            &c,
            "model.language_model.layers.0.mlp_hyper_connection",
            1e-6,
            true,
        )?;
        for case in cases {
            let values = |key: &str, n: u32| -> Result<Vec<f32>> {
                let d = &case[key];
                ensure!(d["count"].as_u64() == Some(u64::from(n)));
                let data = std::fs::read(root.join(d["file"].as_str().context("missing data")?))?;
                ensure!(data.len() == n as usize * 4);
                Ok(data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|&v| f32::from_le_bytes(v))
                    .collect())
            };
            let view = |key: &str, n: u32, half: bool| -> Result<MatrixView> {
                let bits = values(key, n)?;
                let storage = if half {
                    gpu.f16_storage(
                        &bits
                            .iter()
                            .map(|&v| f16::from_f32(v).to_bits())
                            .collect::<Vec<_>>(),
                    )?
                } else {
                    gpu.f32_storage(&bits)?
                };
                storage.matrix(MatrixLayout {
                    shape: [1, n],
                    strides: [i64::from(n), 1],
                    offset: 0,
                })
            };
            let half = match case["branch_dtype"].as_str() {
                Some("mlx.core.float16") => true,
                Some("mlx.core.float32") => false,
                _ => bail!("unexpected branch dtype"),
            };
            let streams = view("streams", 10240, false)?;
            let branch = view("branch", 2560, half)?;
            let gate = view("gate", 4, false)?;
            ensure!(hc.write(&streams, &gate, &branch).is_err());
            let output = hc.write(&streams, &branch, &gate)?.read_output()?;
            let expected = values("output", 10240)?;
            ensure!(expected.iter().all(|v| v.is_finite()));
            assert_eq!(
                output,
                expected.into_iter().map(f32::to_bits).collect::<Vec<_>>()
            );
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "hc_kernel_tests.rs"]
mod hc_kernel_tests;

#[cfg(test)]
#[path = "hc_write_norm_tests.rs"]
mod hc_write_norm_tests;
