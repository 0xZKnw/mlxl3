use super::CachedPipeline as ComputePipelineState;
use super::{FlashAttentionOutput, FlashProjection, MatrixView, Metal, view::FloatType};
use crate::{checkpoint::Checkpoint, flash_support::qsa_append, qsa_chain_policy};
use anyhow::{Context, Result, ensure};
use metal::{
    Buffer, CommandBuffer, CommandBufferRef, MTLCommandBufferStatus,
    MTLResourceOptions as Storage, MTLSize,
};
use std::{cell::Cell, collections::BTreeMap, rc::Rc};

pub(super) struct Pipelines {
    index_rope: ComputePipelineState,
    pool: ComputePipelineState,
    scores: ComputePipelineState,
    nax: std::cell::OnceCell<ComputePipelineState>,
    norm: ComputePipelineState,
    prepare: Option<ComputePipelineState>,
    freq: ComputePipelineState,
    rope: ComputePipelineState,
    attention: ComputePipelineState,
    gated_attention: Option<ComputePipelineState>,
    partials: ComputePipelineState,
    finish: ComputePipelineState,
    gate: ComputePipelineState,
    chains: Option<ComputePipelineState>,
    chains_prefetch: Option<ComputePipelineState>,
    chain_combine: Option<ComputePipelineState>,
    prefetch: Option<ComputePipelineState>,
}
impl Pipelines {
    pub(super) fn load(
        p: &impl Fn(&str) -> Result<ComputePipelineState>,
        prepare: bool,
        gate: bool,
        chains: bool,
        prefetch: bool,
    ) -> Result<Self> {
        Ok(Self {
            index_rope: p("qsa_index_rope")?,
            pool: p("qsa_pool")?,
            scores: p("qsa_index_scores")?,
            nax: std::cell::OnceCell::new(),
            norm: p("qsa_norm")?,
            prepare: prepare.then(|| p("ar10_qsa_prepare")).transpose()?,
            freq: p("qsa_freq")?,
            rope: p("qsa_rope")?,
            attention: p("qsa_attention")?,
            gated_attention: gate.then(|| p("ar10_qsa_attention_gate")).transpose()?,
            partials: p("qsa_attention_partials")?,
            finish: p("qsa_attention_finish")?,
            gate: p("qsa_gate")?,
            chains: chains.then(|| p("ar20_qsa_chains")).transpose()?,
            chains_prefetch: (chains && prefetch)
                .then(|| p("ar20_qsa_chains_prefetch"))
                .transpose()?,
            chain_combine: chains.then(|| p("ar20_qsa_chain_combine")).transpose()?,
            prefetch: prefetch.then(|| p("ar20_qsa_prefetch")).transpose()?,
        })
    }
}
struct Slab {
    keys: Buffer,
    values: Buffer,
    blocks: Option<Buffer>,
    capacity: u32,
    published: Cell<u32>,
}
impl Slab {
    fn new(gpu: &Metal, capacity: u32, indexed: bool) -> Result<Rc<Self>> {
        let alloc = || {
            super::buffer(
                &gpu.device,
                u64::from(capacity) * 1024,
                Storage::StorageModePrivate,
            )
        };
        Ok(Rc::new(Self {
            keys: alloc()?,
            values: alloc()?,
            blocks: indexed
                .then(|| {
                    super::buffer(
                        &gpu.device,
                        u64::from(capacity.div_ceil(4)) * 256,
                        Storage::StorageModePrivate,
                    )
                })
                .transpose()?,
            capacity,
            published: Cell::new(0),
        }))
    }
}

#[derive(Clone)]
pub struct FlashQsaState {
    owner: Rc<()>,
    slab: Rc<Slab>,
    offset: u32,
    tail: Option<Buffer>,
}
struct Index<'a> {
    nax: Option<ComputePipelineState>,
    qk: FlashProjection<'a>,
    q_norm: Buffer,
    k_norm: Buffer,
    freq: Buffer,
}
struct IndexStep {
    tail: Option<Buffer>,
    positions: Option<Vec<u32>>,
    trace: BTreeMap<&'static str, Vec<u32>>,
}
struct EncodedIndex {
    tail: Option<Buffer>,
    score_view: Option<MatrixView>,
    qk: MatrixView,
    block: Option<MatrixView>,
}
impl Index<'_> {
    fn append(
        &self,
        gpu: &Metal,
        slab: &Slab,
        state: &FlashQsaState,
        x: &MatrixView,
        eps: f32,
        trace: bool,
    ) -> Result<IndexStep> {
        let encoded = gpu.command(|cmd| self.encode(gpu, cmd, slab, state, x, (eps, trace)))?;
        self.finish(encoded, state.offset, trace)
    }
    fn encode(
        &self,
        gpu: &Metal,
        cmd: &CommandBufferRef,
        slab: &Slab,
        state: &FlashQsaState,
        x: &MatrixView,
        domain: (f32, bool),
    ) -> Result<EncodedIndex> {
        let (eps, trace) = domain;
        let offset = state.offset;
        let rows = offset % 4;
        let count = (offset + 1) / 4;
        let blocks = slab.blocks.as_ref().context("missing QSA block slab")?;
        let storage = if trace {
            Storage::StorageModeShared
        } else {
            Storage::StorageModePrivate
        };
        let half = |n| MatrixView::output(gpu, FloatType::F16, 1, n, storage);
        let groups = |n: u32| MTLSize::new(u64::from(n), 1, 1);
        let qk = self.qk.encode_row(cmd, x, storage)?;
        let prepare = |src: &MatrixView, gain: &Buffer, heads: u32, pos: u32| -> Result<_> {
            let norm = half(heads * 128)?;
            Metal::encode(
                cmd,
                &gpu.qsa.norm,
                &[&src.buffer, gain, &norm.buffer],
                [128, eps.to_bits(), 128],
                groups(heads),
                32,
            );
            let rotated = half(heads * 128)?;
            Metal::encode(
                cmd,
                &gpu.qsa.index_rope,
                &[&norm.buffer, &self.freq, &rotated.buffer],
                [pos, heads],
                groups((heads * 112).div_ceil(32)),
                32,
            );
            Ok(rotated)
        };
        let (tail, block) = if rows == 3 {
            let old = state.tail.as_ref().context("missing three index rows")?;
            let pool = half(128)?;
            Metal::encode(
                cmd,
                &gpu.qsa.pool,
                &[old, &qk.buffer, &pool.buffer],
                [128u32],
                groups(4),
                32,
            );
            let block = prepare(&pool, &self.k_norm, 1, offset - 3)?;
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer(&block.buffer, 0, blocks, u64::from(count - 1) * 256, 256);
            blit.end_encoding();
            (None, Some(block))
        } else {
            let tail = super::buffer(&gpu.device, u64::from(rows + 1) * 256, storage)?;
            let blit = cmd.new_blit_command_encoder();
            if rows > 0 {
                let old = state.tail.as_ref().context("missing partial index tail")?;
                blit.copy_from_buffer(old, 0, &tail, 0, u64::from(rows) * 256);
            }
            blit.copy_from_buffer(&qk.buffer, 1024, &tail, u64::from(rows) * 256, 256);
            blit.end_encoding();
            (Some(tail), None)
        };
        let scores = if count > 512 {
            let queries = prepare(&qk, &self.q_norm, 4, offset)?;
            let scores =
                MatrixView::output(gpu, FloatType::F32, 1, count, Storage::StorageModeShared)?;
            Metal::encode(
                cmd,
                self.nax.as_ref().unwrap_or(&gpu.qsa.scores),
                &[&queries.buffer, blocks, &scores.buffer],
                [count],
                groups(count.div_ceil(if self.nax.is_some() { 32 } else { 8 })),
                32,
            );
            Some(scores)
        } else {
            None
        };
        Ok(EncodedIndex {
            tail,
            score_view: scores,
            qk,
            block,
        })
    }
    fn finish(&self, encoded: EncodedIndex, offset: u32, trace: bool) -> Result<IndexStep> {
        let EncodedIndex {
            tail,
            score_view,
            qk,
            block,
        } = encoded;
        let mut stages = BTreeMap::new();
        let positions = score_view
            .map(|s| -> Result<Vec<u32>> {
                let values = s.read_output()?;
                let scores = values
                    .iter()
                    .copied()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>();
                let ids = crate::flash_support::sparse_qsa_positions(&scores, offset + 1)
                    .context("invalid QSA block scores or count")?;
                if trace {
                    stages.insert("index_scores", values);
                    stages.insert("selected_positions", ids.clone());
                }
                Ok(ids)
            })
            .transpose()?;
        if trace {
            stages.insert("index_qk", qk.read_output()?);
            if let Some(b) = block {
                stages.insert("index_block", b.read_output()?);
            }
            if let Some(t) = &tail {
                stages.insert(
                    "index_tail",
                    Metal::read(t, t.length() as usize / 2)
                        .into_iter()
                        .map(u32::from)
                        .collect(),
                );
            }
        }
        Ok(IndexStep {
            tail,
            positions,
            trace: stages,
        })
    }
}

#[allow(
    unexpected_cfgs,
    reason = "objc 0.2 selector macro checks the legacy cargo-clippy cfg"
)]
impl Metal {
    fn qsa_architecture(&self) -> Result<String> {
        use objc::{msg_send, runtime::Object, sel, sel_impl};
        use std::ffi::{CStr, c_char};
        let dev = self.device.as_ref();
        // MTLDevice retains its MTLArchitecture and the immutable name for the device lifetime.
        let name = unsafe {
            let arch: *mut Object = msg_send![dev, architecture];
            ensure!(!arch.is_null(), "missing Metal architecture");
            let name: *mut Object = msg_send![arch, name];
            ensure!(!name.is_null(), "missing Metal architecture name");
            let utf8: *const c_char = msg_send![name, UTF8String];
            ensure!(!utf8.is_null(), "invalid Metal architecture string");
            CStr::from_ptr(utf8).to_str()?
        };
        Ok(name.to_owned())
    }
}

type AttentionStep = (MatrixView, Option<[(&'static str, MatrixView); 8]>);

pub(super) struct PreparedQsa {
    before: FlashQsaState,
    published: u32,
    slab: Rc<Slab>,
    at: u64,
}

pub(super) struct PendingQsa {
    command: CommandBuffer,
    payload: Option<(PreparedQsa, Option<EncodedIndex>, AttentionStep, MatrixView)>,
    #[cfg(test)]
    reject_publication: bool,
}
impl Drop for PendingQsa {
    fn drop(&mut self) {
        let status = self.command.status();
        let committed = matches!(
            status,
            MTLCommandBufferStatus::Committed
                | MTLCommandBufferStatus::Scheduled
                | MTLCommandBufferStatus::Completed
                | MTLCommandBufferStatus::Error
        );
        let terminal = matches!(
            status,
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error
        );
        if qsa_chain_policy::submitted_needs_wait(committed, terminal) {
            self.command.wait_until_completed();
        }
        #[cfg(test)]
        if committed && self.payload.is_some() {
            QSA_APPEND_CHAIN_COUNTS.with(|counts| {
                let mut c = counts.get();
                c[2] += 1;
                counts.set(c);
            });
        }
    }
}

#[cfg(test)]
thread_local! {
    pub(super) static QSA_APPEND_CHAIN_COUNTS: Cell<[u64;3]> = const { Cell::new([0;3]) };
}

pub struct FlashQsa<'a> {
    arch_class: u8,
    append_chain_enabled: bool,
    #[cfg(test)]
    fail_after_index_encode: Cell<bool>,
    #[cfg(test)]
    fail_after_append: Cell<bool>,
    index: Option<Index<'a>>,
    gpu: &'a Metal,
    q: FlashProjection<'a>,
    k: FlashProjection<'a>,
    v: FlashProjection<'a>,
    out: FlashProjection<'a>,
    q_norm: Buffer,
    k_norm: Buffer,
    freq: Buffer,
    eps: f32,
    capacity: u32,
    state: FlashQsaState,
}
impl Metal {
    #[allow(
        unexpected_cfgs,
        reason = "objc 0.2 selector macro checks the legacy cargo-clippy cfg"
    )]
    fn qsa_nax_pipeline(&self, arch: &str) -> Result<Option<ComputePipelineState>> {
        use objc::{msg_send, sel, sel_impl};
        let generation = arch
            .strip_prefix("applegpu_g")
            .and_then(|s| s.get(..s.len().checked_sub(1)?))
            .and_then(|s| s.parse::<u32>().ok())
            .context("unknown Metal GPU architecture")?;
        if generation < if arch.ends_with('p') { 18 } else { 17 } {
            return Ok(None);
        }
        if let Some(p) = self.qsa.nax.get() {
            return Ok(Some(p.clone()));
        }
        let version = std::process::Command::new("/usr/bin/sw_vers")
            .arg("-productVersion")
            .output()?;
        ensure!(version.status.success(), "could not read macOS version");
        let text = std::str::from_utf8(&version.stdout)?.trim();
        let mut parts = text.split('.');
        let major = parts
            .next()
            .context("missing macOS major version")?
            .parse::<u32>()?;
        let minor = parts.next().unwrap_or("0").parse::<u32>()?;
        if (major, minor) < (26, 2) {
            return Ok(None);
        }
        let options = metal::CompileOptions::new();
        options.set_fast_math_enabled(false);
        // metal-rs 0.33 does not expose the SDK's MTLLanguageVersion4_0 enum yet.
        unsafe {
            let _: () = msg_send![options.as_ref(),setLanguageVersion:0x40000usize];
        }
        let lib = self
            .device
            .new_library_with_source(include_str!("../../shaders/flash_qsa_nax.metal"), &options)
            .map_err(anyhow::Error::msg)?;
        let f = lib
            .get_function("qsa_index_scores_nax", None)
            .map_err(anyhow::Error::msg)?;
        let p = self
            .device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(anyhow::Error::msg)?;
        let p = ComputePipelineState::new(p, self.cached_caps_enabled);
        ensure!(
            p.thread_execution_width() == 32,
            "unsupported Flash NAX SIMD width"
        );
        self.qsa
            .nax
            .set(p.clone())
            .map_err(|_| anyhow::anyhow!("duplicate QSA pipeline initialization"))?;
        Ok(Some(p))
    }
}
impl<'a> FlashQsa<'a> {
    /// Below the sparse selection threshold every past position contributes, including the current token.
    pub fn load(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        eps: f32,
        theta: f32,
        capacity: u32,
    ) -> Result<Self> {
        ensure!(
            eps.is_finite() && eps > 0. && theta.is_finite() && theta > 0.,
            "invalid Flash QSA epsilon or theta"
        );
        ensure!(
            (1..=crate::flash_support::FLASH_MAX_CONTEXT).contains(&capacity),
            "Flash QSA context must be within 1..=262144"
        );
        ensure!(
            gpu.qsa.attention.thread_execution_width() == 32
                && gpu.qsa.attention.max_total_threads_per_threadgroup() >= 1024
                && gpu.qsa.norm.thread_execution_width() == 32,
            "unsupported Flash QSA threadgroup"
        );
        let proj = |name: &str, dims| -> Result<_> {
            let p = FlashProjection::load(gpu, c, &format!("{prefix}.{name}"))?;
            ensure!(
                p.dimensions() == dims,
                "invalid Flash QSA projection {name}"
            );
            Ok(p)
        };
        let weight = |name| {
            super::checkpoint_float_weight(
                gpu,
                c,
                &format!("{prefix}.{name}.weight"),
                &[256],
                FloatType::F32,
            )
        };
        let arch = if capacity >= 1024 {
            gpu.qsa_architecture()?
        } else {
            String::new()
        };
        let arch_class = arch.as_bytes().last().copied().unwrap_or(b'g');
        let frequencies = |n: u32| -> Result<Buffer> {
            let f = super::buffer(&gpu.device, 128, Storage::StorageModePrivate)?;
            gpu.command(|cmd| {
                Metal::encode(
                    cmd,
                    &gpu.qsa.freq,
                    &[&f],
                    [theta, n as f32],
                    MTLSize::new(1, 1, 1),
                    32,
                );
                Ok(())
            })?;
            Ok(f)
        };
        let indexed = capacity >= 2052 || prefix == "mtp.layers.0.self_attn";
        let index = if indexed {
            let gain = |name| {
                super::checkpoint_float_weight(
                    gpu,
                    c,
                    &format!("{prefix}.indexer.{name}.weight"),
                    &[128],
                    FloatType::F32,
                )
            };
            Some(Index {
                nax: if capacity >= 2052 {
                    gpu.qsa_nax_pipeline(&arch)?
                } else {
                    None
                },
                qk: proj("indexer.index_qk_proj", [2560, 640])?,
                q_norm: gain("q_layernorm")?,
                k_norm: gain("k_layernorm")?,
                freq: frequencies(16)?,
            })
        } else {
            None
        };
        let append_chain_enabled = match std::env::var("MLXL3_NATIVE_QSA_APPEND_CHAIN") {
            Ok(value) => qsa_chain_policy::setting(Some(&value), gpu.device.name() == "Apple M5"),
            Err(std::env::VarError::NotPresent) => {
                qsa_chain_policy::setting(None, gpu.device.name() == "Apple M5")
            }
            Err(error) => return Err(error.into()),
        }
        .context("MLXL3_NATIVE_QSA_APPEND_CHAIN must be 0 or 1")?;
        Ok(Self {
            arch_class,
            append_chain_enabled,
            #[cfg(test)]
            fail_after_index_encode: Cell::new(false),
            #[cfg(test)]
            fail_after_append: Cell::new(false),
            gpu,
            q: proj("q_proj", [2560, 12288])?,
            k: proj("k_proj", [2560, 512])?,
            v: proj("v_proj", [2560, 512])?,
            out: proj("o_proj", [6144, 2560])?,
            q_norm: weight("q_norm")?,
            k_norm: weight("k_norm")?,
            freq: frequencies(32)?,
            index,
            eps,
            capacity,
            state: FlashQsaState {
                owner: Rc::new(()),
                slab: Slab::new(gpu, capacity.min(32), indexed)?,
                tail: None,
                offset: 0,
            },
        })
    }
    pub fn snapshot(&self) -> FlashQsaState {
        self.state.clone()
    }
    pub fn restore(&mut self, s: &FlashQsaState) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&s.owner, &self.state.owner),
            "QSA state belongs to a different layer"
        );
        self.state = s.clone();
        Ok(())
    }
    pub fn reset(&mut self) {
        self.state.offset = 0;
        self.state.tail = None;
    }

    pub fn forward(&mut self, x: &MatrixView, trace: bool) -> Result<FlashAttentionOutput> {
        let gpu = self.gpu;
        ensure!(
            x.shape() == [1, 2560] && x.buffer.device().registry_id() == gpu.device.registry_id(),
            "invalid Flash QSA input shape or device"
        );
        let prepared = self.prepare_append()?;
        let offset = prepared.before.offset;
        if qsa_chain_policy::append_chain_eligible(
            self.append_chain_enabled,
            self.index.is_some(),
            trace,
            offset,
        ) {
            let pending = gpu.command(|cmd| self.encode_for_consumer(cmd, x, prepared))?;
            return self.finish(pending);
        }
        let index = self
            .index
            .as_ref()
            .map(|index| index.append(gpu, &prepared.slab, &prepared.before, x, self.eps, trace))
            .transpose()?;
        let positions = index
            .as_ref()
            .and_then(|step| step.positions.as_deref())
            .map(|ids| super::upload(&gpu.device, ids))
            .transpose()?;
        let selected = index
            .as_ref()
            .and_then(|step| step.positions.as_ref())
            .map_or(offset + 1, |ids| ids.len() as u32);
        let attention = gpu.command(|cmd| {
            self.encode_attention(cmd, x, &prepared, positions.as_ref(), selected, trace)
        })?;
        self.publish(prepared, index, attention)
    }
    pub(super) fn prepare_append(&self) -> Result<PreparedQsa> {
        let gpu = self.gpu;
        let offset = self.state.offset;
        let prior = &self.state.slab;
        let published = prior.published.get();
        let (at, fork) = qsa_append(offset, published, self.capacity)
            .context("Flash QSA dense context exhausted")?;
        let size = crate::flash_support::qsa_slab_capacity(offset, prior.capacity, self.capacity)
            .context("invalid Flash QSA slab extent")?;
        let slab = if fork || size != prior.capacity {
            let target = Slab::new(gpu, size, self.index.is_some())?;
            if at > 0 {
                gpu.command(|cmd| {
                    let blit = cmd.new_blit_command_encoder();
                    blit.copy_from_buffer(&prior.keys, 0, &target.keys, 0, at);
                    blit.copy_from_buffer(&prior.values, 0, &target.values, 0, at);
                    if let (Some(from), Some(to)) = (&prior.blocks, &target.blocks)
                        && offset >= 4
                    {
                        blit.copy_from_buffer(from, 0, to, 0, u64::from(offset / 4) * 256);
                    }
                    blit.end_encoding();
                    Ok(())
                })?;
            }
            target
        } else {
            prior.clone()
        };
        Ok(PreparedQsa {
            before: self.state.clone(),
            published,
            slab,
            at,
        })
    }
    pub(super) fn encode_for_consumer(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        prepared: PreparedQsa,
    ) -> Result<PendingQsa> {
        let gpu = self.gpu;
        ensure!(
            x.shape() == [1, 2560] && x.buffer.device().registry_id() == gpu.device.registry_id(),
            "invalid Flash QSA input shape or device"
        );
        ensure!(
            self.owns_prepared(&prepared)
                && qsa_chain_policy::inline_eligible(
                    self.index.is_some(),
                    false,
                    prepared.before.offset
                ),
            "QSA inline append requires unchanged CPU-free state"
        );
        ensure!(
            gpu.owns_command_queue(cmd) && cmd.status() == MTLCommandBufferStatus::NotEnqueued,
            "QSA encoder requires an unsubmitted command on its owner queue"
        );
        let index = self
            .index
            .as_ref()
            .map(|index| {
                index.encode(
                    gpu,
                    cmd,
                    &prepared.slab,
                    &prepared.before,
                    x,
                    (self.eps, false),
                )
            })
            .transpose()?;
        #[cfg(test)]
        ensure!(
            !self.fail_after_index_encode.replace(false),
            "injected QSA failure after index encoding"
        );
        let attention =
            self.encode_attention(cmd, x, &prepared, None, prepared.before.offset + 1, false)?;
        Ok(PendingQsa {
            command: cmd.to_owned(),
            payload: Some((prepared, index, attention, x.clone())),
            #[cfg(test)]
            reject_publication: self.fail_after_append.replace(false),
        })
    }
    fn owns_prepared(&self, prepared: &PreparedQsa) -> bool {
        self.state.offset == prepared.before.offset
            && Rc::ptr_eq(&self.state.owner, &prepared.before.owner)
            && Rc::ptr_eq(&self.state.slab, &prepared.before.slab)
            && prepared.before.slab.published.get() == prepared.published
    }
    pub(super) fn finish(&mut self, mut pending: PendingQsa) -> Result<FlashAttentionOutput> {
        let prepared = &pending
            .payload
            .as_ref()
            .context("missing staged QSA append")?
            .0;
        ensure!(
            qsa_chain_policy::publication_matches(
                self.state.offset,
                prepared.before.offset,
                self.owns_prepared(prepared),
                pending.command.status() == MTLCommandBufferStatus::Completed
            ),
            "QSA state cannot publish before its whole command completes"
        );
        #[cfg(test)]
        {
            QSA_APPEND_CHAIN_COUNTS.with(|counts| {
                let mut c = counts.get();
                c[0] += 1;
                counts.set(c);
            });
            ensure!(
                !pending.reject_publication,
                "injected QSA failure after append completion"
            );
        }
        let (prepared, encoded_index, attention, _input) = pending.payload.take().unwrap();
        let index = encoded_index
            .map(|encoded| {
                self.index
                    .as_ref()
                    .context("missing staged QSA index")?
                    .finish(encoded, prepared.before.offset, false)
            })
            .transpose()?;
        let output = self.publish(prepared, index, attention)?;
        #[cfg(test)]
        QSA_APPEND_CHAIN_COUNTS.with(|counts| {
            let mut c = counts.get();
            c[1] += 1;
            counts.set(c);
        });
        Ok(output)
    }
    fn encode_attention(
        &self,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        prepared: &PreparedQsa,
        positions: Option<&Buffer>,
        selected: u32,
        trace: bool,
    ) -> Result<AttentionStep> {
        let gpu = self.gpu;
        let offset = prepared.before.offset;
        let at = prepared.at;
        let slab = &prepared.slab;
        let storage = if trace {
            Storage::StorageModeShared
        } else {
            Storage::StorageModePrivate
        };
        let half = |n| MatrixView::output(gpu, FloatType::F16, 1, n, storage);
        let groups = |n| MTLSize::new(n, 1, 1);
        let qg = self.q.encode_row(cmd, x, storage)?;
        let k = self.k.encode_row(cmd, x, storage)?;
        let v = self.v.encode_row(cmd, x, storage)?;
        let prepare = |input: &MatrixView, gain: &Buffer, heads: u32, stride: u32| -> Result<_> {
            let norm = half(heads * 256)?;
            Metal::encode(
                cmd,
                &gpu.qsa.norm,
                &[&input.buffer, gain, &norm.buffer],
                [stride, self.eps.to_bits(), 256],
                groups(u64::from(heads)),
                32,
            );
            let rope = half(heads * 256)?;
            Metal::encode(
                cmd,
                &gpu.qsa.rope,
                &[&norm.buffer, &self.freq, &rope.buffer],
                [offset, heads],
                groups(u64::from(heads) * 7),
                32,
            );
            Ok((norm, rope))
        };
        let (qn, qr, kn, kr) = if crate::flash_support::gdn_private_epilogue_eligible(
            gpu.qsa_prepare.get(),
            trace,
            x.dtype == FloatType::F32,
        ) {
            let qr = half(6144)?;
            let kr = half(512)?;
            Metal::encode(
                cmd,
                gpu.qsa
                    .prepare
                    .as_ref()
                    .context("QSA prepare pipeline not initialized")?,
                &[
                    &qg.buffer,
                    &k.buffer,
                    &self.q_norm,
                    &self.k_norm,
                    &self.freq,
                    &qr.buffer,
                    &kr.buffer,
                ],
                [offset, self.eps.to_bits()],
                groups(26),
                32,
            );
            #[cfg(test)]
            gpu.qsa_prepare_uses.set(gpu.qsa_prepare_uses.get() + 1);
            (None, qr, None, kr)
        } else {
            let (qn, qr) = prepare(&qg, &self.q_norm, 24, 512)?;
            let (kn, kr) = prepare(&k, &self.k_norm, 2, 256)?;
            (Some(qn), qr, Some(kn), kr)
        };
        let blit = cmd.new_blit_command_encoder();
        blit.copy_from_buffer(&kr.buffer, 0, &slab.keys, at, 1024);
        blit.copy_from_buffer(&v.buffer, 0, &slab.values, at, 1024);
        blit.end_encoding();
        let attention = half(6144)?;
        let ids = positions.unwrap_or(&qr.buffer);
        let sparse = u32::from(positions.is_some());
        let fused_gate = crate::flash_support::ar10_qsa_gate_eligible(
            gpu.qsa_gate_fused.get(),
            trace,
            x.dtype == FloatType::F32,
            selected,
            positions.is_some(),
        );
        if gpu.encode_ar20_qsa(
            cmd,
            [
                &qr.buffer,
                &slab.keys,
                &slab.values,
                &attention.buffer,
                &qg.buffer,
            ],
            selected,
            [trace, x.dtype == FloatType::F32, positions.is_some()],
            fused_gate,
        )? {
        } else if let Some(blocks) =
            crate::flash_support::qsa_attention_blocks(self.arch_class, selected)
        {
            let partial = half(24 * blocks * 256)?;
            let scratch = || {
                super::buffer(
                    &gpu.device,
                    u64::from(24 * blocks) * 4,
                    Storage::StorageModePrivate,
                )
            };
            let sums = scratch()?;
            let maxima = scratch()?;
            Metal::encode(
                cmd,
                &gpu.qsa.partials,
                &[
                    &qr.buffer,
                    &slab.keys,
                    &slab.values,
                    ids,
                    &partial.buffer,
                    &sums,
                    &maxima,
                ],
                [selected, sparse, blocks],
                MTLSize::new(24, u64::from(blocks), 1),
                32,
            );
            Metal::encode(
                cmd,
                &gpu.qsa.finish,
                &[&partial.buffer, &sums, &maxima, &attention.buffer],
                [blocks],
                groups(24),
                1024,
            );
        } else if fused_gate {
            Metal::encode(
                cmd,
                gpu.qsa
                    .gated_attention
                    .as_ref()
                    .context("QSA gated pipeline not initialized")?,
                &[
                    &qr.buffer,
                    &slab.keys,
                    &slab.values,
                    &attention.buffer,
                    ids,
                    &qg.buffer,
                ],
                [selected, sparse],
                groups(24),
                1024,
            );
            #[cfg(test)]
            gpu.qsa_gate_fused_uses
                .set(gpu.qsa_gate_fused_uses.get() + 1);
        } else {
            Metal::encode(
                cmd,
                &gpu.qsa.attention,
                &[&qr.buffer, &slab.keys, &slab.values, &attention.buffer, ids],
                [selected, sparse],
                groups(24),
                1024,
            );
        }
        let gated = if fused_gate {
            attention.clone()
        } else {
            let gated = half(6144)?;
            Metal::encode(
                cmd,
                &gpu.qsa.gate,
                &[&attention.buffer, &qg.buffer, &gated.buffer],
                [256u32],
                groups(192),
                32,
            );
            gated
        };
        let out = self
            .out
            .encode_row(cmd, &gated, Storage::StorageModeShared)?;
        let values = if x.dtype == FloatType::F32 {
            let f = MatrixView::output(gpu, FloatType::F32, 1, 2560, Storage::StorageModeShared)?;
            Metal::encode(
                cmd,
                &gpu.gdn.to_float,
                &[&out.buffer, &f.buffer],
                [2560u32],
                groups(80),
                32,
            );
            f
        } else {
            out
        };
        Ok((
            values,
            if trace {
                Some([
                    ("q_proj", qg),
                    ("q_norm", qn.context("trace Q norm missing")?),
                    ("q_rope", qr),
                    ("k_proj", k),
                    ("k_norm", kn.context("trace K norm missing")?),
                    ("k_rope", kr),
                    ("v_proj", v),
                    ("gated", gated),
                ])
            } else {
                None
            },
        ))
    }
    fn publish(
        &mut self,
        prepared: PreparedQsa,
        index: Option<IndexStep>,
        attention: AttentionStep,
    ) -> Result<FlashAttentionOutput> {
        let offset = prepared.before.offset;
        let slab = prepared.slab;
        let (values, steps) = attention;
        let mut trace = steps
            .map(|items| {
                items
                    .into_iter()
                    .map(|(name, v)| Ok((name, v.read_output()?)))
                    .collect::<Result<BTreeMap<_, _>>>()
            })
            .transpose()?;
        if let Some(index) = index {
            if let Some(t) = &mut trace {
                t.extend(index.trace);
            }
            self.state.tail = index.tail;
        }
        slab.published.set(offset + 1);
        self.state.slab = slab;
        self.state.offset = offset + 1;
        Ok(FlashAttentionOutput { values, trace })
    }
}

pub struct FlashQsaLayer<'a> {
    attention: FlashQsa<'a>,
    pub(super) residual: super::gdn::LayerResidual<'a>,
}
impl<'a> FlashQsaLayer<'a> {
    pub fn load(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        eps: f32,
        theta: f32,
        capacity: u32,
        slots: usize,
    ) -> Result<Self> {
        Ok(Self {
            attention: FlashQsa::load(
                gpu,
                c,
                &format!("{prefix}.self_attn"),
                eps,
                theta,
                capacity,
            )?,
            residual: super::gdn::LayerResidual::load(gpu, c, prefix, eps, slots)?,
        })
    }
    pub(super) fn offset(&self) -> u32 {
        self.attention.state.offset
    }
    #[cfg(test)]
    pub(super) fn retained_half_words(&self) -> Result<BTreeMap<&'static str, Vec<u16>>> {
        let state = &self.attention.state;
        let gpu = self.attention.gpu;
        let n = state.offset as usize;
        let mut data = BTreeMap::new();
        let read = |buffer: &Buffer, words: usize| -> Result<Vec<u16>> {
            let bytes = u64::try_from(words)?
                .checked_mul(2)
                .context("cache extent overflow")?;
            ensure!(bytes > 0 && bytes <= buffer.length());
            let out = super::buffer(&gpu.device, bytes, Storage::StorageModeShared)?;
            gpu.command(|cmd| {
                let blit = cmd.new_blit_command_encoder();
                blit.copy_from_buffer(buffer, 0, &out, 0, bytes);
                blit.end_encoding();
                Ok(())
            })?;
            Ok(Metal::read(&out, words))
        };
        if n > 0 {
            for (key, buf) in [("keys", &state.slab.keys), ("values", &state.slab.values)] {
                let interleaved = read(buf, n * 512)?;
                let mut ordered = Vec::with_capacity(n * 512);
                for h in 0..2 {
                    for row in interleaved.as_chunks::<512>().0 {
                        ordered.extend_from_slice(&row[h * 256..(h + 1) * 256]);
                    }
                }
                data.insert(key, ordered);
            }
        }
        if let Some(tail) = &state.tail {
            data.insert("index_tail", read(tail, n % 4 * 128)?);
        }
        if n >= 4
            && let Some(blocks) = &state.slab.blocks
        {
            data.insert("blocks", read(blocks, n / 4 * 128)?);
        }
        Ok(data)
    }
    pub(super) fn extend(&mut self, streams: &MatrixView) -> Result<()> {
        let mixed = self.residual.attention_input(streams)?;
        self.attention.forward(&mixed, false)?;
        Ok(())
    }
    pub fn snapshot(&self) -> FlashQsaState {
        self.attention.snapshot()
    }
    pub(super) fn prefix_position(&self) -> u32 {
        self.attention.state.offset
    }
    pub(super) fn owns_prefix(&self, state: &FlashQsaState) -> bool {
        Rc::ptr_eq(&self.attention.state.owner, &state.owner)
    }
    pub fn restore(&mut self, s: &FlashQsaState) -> Result<()> {
        self.attention.restore(s)
    }
    pub fn reset(&mut self) {
        self.attention.reset();
    }
    pub fn forward(&mut self, streams: &MatrixView) -> Result<MatrixView> {
        self.forward_lookahead(streams, None, 0)
    }
    pub(super) fn forward_lookahead(
        &mut self,
        streams: &MatrixView,
        next: Option<&mut super::FlashMoeRow<'a>>,
        depth: usize,
    ) -> Result<MatrixView> {
        self.forward_deferred(streams, next, depth, false)?
            .complete(self.attention.gpu)
    }
    pub(super) fn forward_deferred(
        &mut self,
        streams: &MatrixView,
        next: Option<&mut super::FlashMoeRow<'a>>,
        depth: usize,
        defer_write: bool,
    ) -> Result<super::gdn::LayerOutput> {
        let before = self.snapshot();
        let out = self.residual.forward(
            streams,
            |x| self.attention.forward(x, false),
            next,
            depth,
            defer_write,
        );
        if out.is_err() {
            self.restore(&before)?;
        }
        out
    }
}

impl FlashQsaState {
    pub(super) fn record_memory(
        &self,
        backings: &mut crate::gpu::memory::Backings,
        category: &'static str,
    ) -> Result<()> {
        for buffer in [&self.slab.keys, &self.slab.values] {
            backings.record(category, buffer)?;
        }
        if let Some(blocks) = &self.slab.blocks {
            backings.record(category, blocks)?;
        }
        if let Some(tail) = &self.tail {
            backings.record(category, tail)?;
        }
        Ok(())
    }
    pub fn retained_bytes(&self) -> Result<u64> {
        let mut backings = crate::gpu::memory::Backings::default();
        self.record_memory(&mut backings, "prompt_cache")?;
        backings.total()
    }
}
impl FlashQsaLayer<'_> {
    pub(super) fn record_memory(&self, backings: &mut crate::gpu::memory::Backings) -> Result<()> {
        let q = &self.attention;
        for projection in [&q.q, &q.k, &q.v, &q.out] {
            projection.record_memory(backings)?;
        }
        for buffer in [&q.q_norm, &q.k_norm, &q.freq] {
            backings.record("fixed_weights", buffer)?;
        }
        if let Some(index) = &q.index {
            index.qk.record_memory(backings)?;
            for buffer in [&index.q_norm, &index.k_norm, &index.freq] {
                backings.record("fixed_weights", buffer)?;
            }
        }
        backings.record("qsa_kv", &q.state.slab.keys)?;
        backings.record("qsa_kv", &q.state.slab.values)?;
        if let Some(buffer) = &q.state.slab.blocks {
            backings.record("qsa_index", buffer)?;
        }
        if let Some(buffer) = &q.state.tail {
            backings.record("qsa_index", buffer)?;
        }
        self.residual.record_memory(backings)
    }
    pub(super) fn release_idle_state(&mut self) -> Result<()> {
        self.residual.release_idle_state()?;
        let attention = &mut self.attention;
        for projection in [&attention.q, &attention.k, &attention.v, &attention.out] {
            projection.release_idle_scratch()?;
        }
        if let Some(index) = &attention.index {
            index.qk.release_idle_scratch()?;
        }
        let slab = Slab::new(
            attention.gpu,
            attention.capacity.min(32),
            attention.index.is_some(),
        )?;
        attention.state.slab = slab;
        attention.reset();
        Ok(())
    }
    pub(super) fn shrink(&mut self, slots: usize) -> Result<()> {
        self.residual.shrink(slots)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_layout::MatrixLayout;

    #[test]
    #[ignore = "physical Flash QSA retained branches and full context"]
    fn divergent_suffix_preserves_both_snapshots_and_rejects_full_context() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_QSA_MODEL")?))?;
        let gpu = Metal::new()?;
        let layer = |prefix| FlashQsa::load(&gpu, &c, prefix, 1e-6, 1e7, 3);
        let mut a = layer("model.language_model.layers.15.self_attn")?;
        let other = layer("model.language_model.layers.35.self_attn")?;
        let row = |t| {
            gpu.f32_storage(
                &(0..2560)
                    .map(|i| ((i * 17 + t * 83) % 257 - 128) as f32 / 237.)
                    .collect::<Vec<_>>(),
            )?
            .matrix(MatrixLayout {
                shape: [1, 2560],
                strides: [2560, 1],
                offset: 0,
            })
        };
        let (x, y, z) = (row(0)?, row(1)?, row(2)?);
        let first = a.forward(&x, false)?.values;
        let prefix = a.snapshot();
        a.forward(&y, false)?;
        let xy = a.snapshot();
        let xyz = a.forward(&z, false)?.values;
        ensure!(
            a.forward(&x, false)
                .err()
                .context("accepted full context")?
                .to_string()
                .contains("exhausted")
        );
        ensure!(a.restore(&other.snapshot()).is_err());
        a.restore(&prefix)?;
        a.forward(&z, false)?;
        let xzy = a.forward(&y, false)?.values;
        a.restore(&xy)?;
        ensure!(a.forward(&z, false)?.values.read_output()? == xyz.read_output()?);
        a.reset();
        ensure!(a.forward(&x, false)?.values.read_output()? == first.read_output()?);
        a.forward(&z, false)?;
        ensure!(
            a.forward(&y, false)?.values.read_output()? == xzy.read_output()?
                && xzy.read_output()? != xyz.read_output()?
        );
        Ok(())
    }
}

#[cfg(test)]
#[test]
#[ignore = "physical long-context QSA comparing sparse selections and complete cached state"]
fn long_qsa_replays_oracle_and_checks_cache() -> Result<()> {
    use crate::tensor_layout::MatrixLayout;
    use serde_json::json;
    use std::path::{Path, PathBuf};
    let model = std::env::var("MLXL3_QSA_MODEL")?;
    let root = PathBuf::from(std::env::var("MLXL3_QSA_ORACLE_DIR")?);
    let output = PathBuf::from(std::env::var("MLXL3_QSA_REPORT")?);
    let mut sink = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    let c = crate::checkpoint::inspect(Path::new(&model))?;
    let gpu = Metal::new()?;
    let mut qsa = FlashQsa::load(
        &gpu,
        &c,
        "model.language_model.layers.15.self_attn",
        1e-6,
        1e7,
        4096,
    )?;
    if let Ok(class) = std::env::var("MLXL3_QSA_TEST_CLASS") {
        ensure!(matches!(class.as_str(), "s" | "d"));
        qsa.arch_class = class.as_bytes()[0];
    }
    if std::env::var_os("MLXL3_QSA_TEST_CLASSIC").is_some() {
        qsa.index.as_mut().context("missing index")?.nax = None;
    }
    let mut rows = Vec::new();
    let mut total = 0usize;
    let mut mismatches = 0usize;
    let evaluation = (|| -> Result<()> {
        for t in 0..4096 {
            let n = t + 1;
            let sample = crate::flash_support::qsa_probe_checkpoint(n);
            let x = gpu
                .f32_storage(&crate::flash_support::qsa_probe_row(t))?
                .matrix(MatrixLayout {
                    shape: [1, 2560],
                    strides: [0, 1],
                    offset: 0,
                })?;
            let branch = matches!(n, 32 | 33 | 2049 | 2052 | 2053);
            let prefix = branch.then(|| qsa.snapshot());
            if n == 33 {
                // Inject a mismatched projection after allocating the larger slab.
                std::mem::swap(&mut qsa.index.as_mut().unwrap().qk, &mut qsa.out);
                let error = qsa
                    .forward(&x, false)
                    .err()
                    .context("invalid index projection accepted")?;
                std::mem::swap(&mut qsa.index.as_mut().unwrap().qk, &mut qsa.out);
                ensure!(error.to_string().contains("projection view width"));
                let before = prefix.as_ref().unwrap();
                ensure!(
                    qsa.state.offset == before.offset
                        && Rc::ptr_eq(&before.slab, &qsa.state.slab)
                        && before.slab.published.get() == before.offset
                );
            }
            let y = qsa.forward(&x, sample)?;
            if let Some(prefix) = prefix {
                let original = qsa.snapshot();
                let original_out = y.values.read_output()?;
                qsa.restore(&prefix)?;
                let alternate = gpu
                    .f32_storage(&crate::flash_support::qsa_probe_row(t + 7))?
                    .matrix(x.layout)?;
                let alt = qsa.forward(&alternate, true)?.values.read_output()?;
                ensure!(alt != original_out);
                let alt_state = qsa.snapshot();
                qsa.restore(&prefix)?;
                ensure!(qsa.forward(&x, true)?.values.read_output()? == original_out);
                qsa.restore(&alt_state)?;
                ensure!(qsa.state.offset == n);
                qsa.restore(&original)?;
            }
            if !sample {
                continue;
            }
            let mut stages = y.trace.context("missing long QSA trace")?;
            stages.retain(|key, _| key.starts_with("index_") || *key == "selected_positions");
            stages.insert("out", y.values.read_output()?);
            if n == 4096 {
                let slab = &qsa.state.slab;
                for (name, buffer, elements) in [
                    ("keys", &slab.keys, n * 512),
                    ("values", &slab.values, n * 512),
                    (
                        "blocks",
                        slab.blocks.as_ref().context("missing blocks")?,
                        n / 4 * 128,
                    ),
                ] {
                    let copy = super::buffer(
                        &gpu.device,
                        u64::from(elements) * 2,
                        Storage::StorageModeShared,
                    )?;
                    gpu.command(|cmd| {
                        let blit = cmd.new_blit_command_encoder();
                        blit.copy_from_buffer(buffer, 0, &copy, 0, copy.length());
                        blit.end_encoding();
                        Ok(())
                    })?;
                    stages.insert(
                        name,
                        Metal::read(&copy, elements as usize)
                            .into_iter()
                            .map(u32::from)
                            .collect(),
                    );
                }
            }
            ensure!(
                stages.contains_key("index_qk")
                    && stages.contains_key(if n % 4 == 0 {
                        "index_block"
                    } else {
                        "index_tail"
                    })
            );
            ensure!(
                stages.contains_key("selected_positions") == (n / 4 > 512)
                    && stages.contains_key("index_scores") == (n / 4 > 512)
            );
            for (key, actual) in stages {
                let data = std::fs::read(root.join(format!("{n}-{key}.bin")))?;
                let wide = matches!(key, "index_scores" | "selected_positions" | "out");
                let expected: Vec<u32> = if wide {
                    data.as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| u32::from_le_bytes(*b))
                        .collect()
                } else {
                    data.as_chunks::<2>()
                        .0
                        .iter()
                        .map(|b| u32::from(u16::from_le_bytes(*b)))
                        .collect()
                };
                ensure!(
                    !actual.is_empty()
                        && actual.len() == expected.len()
                        && data.len() == expected.len() * if wide { 4 } else { 2 },
                    "invalid length at {n}/{key}"
                );
                let finite = |&v: &u32| {
                    if key == "selected_positions" {
                        v < n
                    } else if wide {
                        f32::from_bits(v).is_finite()
                    } else {
                        half::f16::from_bits(v as u16).is_finite()
                    }
                };
                ensure!(
                    actual.iter().all(finite) && expected.iter().all(finite),
                    "invalid value at {n}/{key}"
                );
                let mut bad = actual
                    .iter()
                    .zip(&expected)
                    .enumerate()
                    .filter(|(_, pair)| pair.0 != pair.1);
                let first = bad.next().map(|(i, (&a, &b))| (i, a, b));
                let differences = bad.count() + usize::from(first.is_some());
                total += actual.len();
                mismatches += differences;
                rows.push(json!({"length":n,"name":key,"elements":actual.len(),"mismatches":differences,"first":first}));
            }
        }
        ensure!(
            total > 4000000 && mismatches == 0,
            "{mismatches} unequal words over {total} comparisons"
        );
        Ok(())
    })();
    serde_json::to_writer_pretty(
        &mut sink,
        &json!({"status":if evaluation.is_ok(){"verified"}else{"failed"},"error":evaluation.as_ref().err().map(ToString::to_string),"rows":rows,"comparisons":total,"mismatches":mismatches}),
    )?;
    evaluation
}

#[cfg(test)]
#[path = "ar10_qsa_tests.rs"]
mod ar10_qsa_tests;

impl Metal {
    pub(super) fn encode_ar20_qsa(
        &self,
        cmd: &metal::CommandBufferRef,
        b: [&Buffer; 5],
        keys: u32,
        domain: [bool; 3],
        fused_gate: bool,
    ) -> Result<bool> {
        let [trace, fp32, sparse] = domain;
        if !crate::flash_support::ar10_qsa_gate_eligible(true, trace, fp32, keys, sparse) {
            return Ok(false);
        }
        let p = &self.qsa;
        let prefetch = self.qsa_prefetch.get();
        if self.qsa_chains.get() {
            let producer = if prefetch {
                p.chains_prefetch.as_ref()
            } else {
                p.chains.as_ref()
            };
            if let (Some(producer), Some(combine)) = (producer, p.chain_combine.as_ref())
                && crate::flash_support::ar20_qsa_chains_eligible(
                    true,
                    trace,
                    fp32,
                    keys,
                    sparse,
                    [
                        producer.thread_execution_width() as u32,
                        producer.max_total_threads_per_threadgroup() as u32,
                        combine.thread_execution_width() as u32,
                        combine.max_total_threads_per_threadgroup() as u32,
                    ],
                )
            {
                let scratch = |n| super::buffer(&self.device, n, Storage::StorageModePrivate);
                let maxima = scratch(3072)?;
                let sums = scratch(3072)?;
                let outputs = scratch(786432)?;
                Metal::encode(
                    cmd,
                    producer,
                    &[b[0], b[1], b[2], &maxima, &sums, &outputs],
                    [keys],
                    MTLSize::new(48, 2, 1),
                    128,
                );
                Metal::encode(
                    cmd,
                    combine,
                    &[&maxima, &sums, &outputs, b[3], b[4]],
                    [u32::from(fused_gate)],
                    MTLSize::new(24, 1, 1),
                    1024,
                );
                #[cfg(test)]
                {
                    self.qsa_chain_uses.set(self.qsa_chain_uses.get() + 1);
                    if prefetch {
                        self.qsa_prefetch_uses.set(self.qsa_prefetch_uses.get() + 1);
                    }
                    if fused_gate {
                        self.qsa_gate_fused_uses
                            .set(self.qsa_gate_fused_uses.get() + 1);
                    }
                }
                return Ok(true);
            }
        }
        if prefetch
            && let Some(pipeline) = p.prefetch.as_ref()
            && pipeline.thread_execution_width() == 32
            && pipeline.max_total_threads_per_threadgroup() >= 1024
        {
            Metal::encode(
                cmd,
                pipeline,
                &b,
                [keys, 0, u32::from(fused_gate)],
                MTLSize::new(24, 1, 1),
                1024,
            );
            #[cfg(test)]
            {
                self.qsa_prefetch_uses.set(self.qsa_prefetch_uses.get() + 1);
                if fused_gate {
                    self.qsa_gate_fused_uses
                        .set(self.qsa_gate_fused_uses.get() + 1);
                }
            }
            return Ok(true);
        }
        Ok(false)
    }
}
#[cfg(test)]
#[path = "ar20_qsa_tests.rs"]
mod ar20_qsa_tests;

#[cfg(test)]
mod ar36_tests {
    use super::*;
    use crate::tensor_layout::MatrixLayout;
    use half::f16;
    use std::path::{Path, PathBuf};

    fn seed(layer: &mut FlashQsa<'_>, offset: u32) -> Result<()> {
        let gpu = layer.gpu;
        let capacity = offset
            .max(1)
            .next_power_of_two()
            .max(32)
            .min(layer.capacity);
        let slab = Slab::new(gpu, capacity, layer.index.is_some())?;
        let words = |count: usize, salt: usize| {
            (0..count)
                .map(|i| f16::from_f32(((i * 17 + salt) % 257) as f32 / 237. - 0.5).to_bits())
                .collect::<Vec<_>>()
        };
        if offset > 0 {
            let keys = super::super::upload(&gpu.device, &words(offset as usize * 512, 3))?;
            let values = super::super::upload(&gpu.device, &words(offset as usize * 512, 11))?;
            let blocks = (offset >= 4 && layer.index.is_some())
                .then(|| super::super::upload(&gpu.device, &words((offset / 4) as usize * 128, 19)))
                .transpose()?;
            gpu.command(|cmd| {
                let blit = cmd.new_blit_command_encoder();
                for (from, to) in [(&keys, &slab.keys), (&values, &slab.values)] {
                    blit.copy_from_buffer(from, 0, to, 0, from.length());
                }
                if let Some(from) = &blocks {
                    blit.copy_from_buffer(from, 0, slab.blocks.as_ref().unwrap(), 0, from.length());
                }
                blit.end_encoding();
                Ok(())
            })?;
        }
        layer.state.tail = (offset % 4 > 0 && layer.index.is_some())
            .then(|| super::super::upload(&gpu.device, &words((offset % 4) as usize * 128, 29)))
            .transpose()?;
        slab.published.set(offset);
        layer.state.slab = slab;
        layer.state.offset = offset;
        Ok(())
    }

    fn retained(gpu: &Metal, state: &FlashQsaState) -> Result<BTreeMap<&'static str, Vec<u16>>> {
        let mut result = BTreeMap::new();
        let n = state.offset;
        let mut copy = |name, source: &Buffer, words: u32| -> Result<()> {
            if words == 0 {
                return Ok(());
            }
            let bytes = u64::from(words) * 2;
            ensure!(bytes <= source.length());
            let target = super::super::buffer(&gpu.device, bytes, Storage::StorageModeShared)?;
            gpu.command(|cmd| {
                let blit = cmd.new_blit_command_encoder();
                blit.copy_from_buffer(source, 0, &target, 0, bytes);
                blit.end_encoding();
                Ok(())
            })?;
            let values = Metal::read(&target, words as usize);
            ensure!(
                values.len() == words as usize
                    && values.iter().all(|&v| f16::from_bits(v).is_finite())
            );
            result.insert(name, values);
            Ok(())
        };
        copy("keys", &state.slab.keys, n * 512)?;
        copy("values", &state.slab.values, n * 512)?;
        if let Some(blocks) = &state.slab.blocks {
            copy("blocks", blocks, n / 4 * 128)?;
        }
        if let Some(tail) = &state.tail {
            copy("tail", tail, n % 4 * 128)?;
        }
        Ok(result)
    }

    fn input(gpu: &Metal, fp32: bool, salt: usize) -> Result<MatrixView> {
        let values = (0..2560)
            .map(|i| ((i * 17 + salt) % 257) as f32 / 237. - 0.5)
            .collect::<Vec<_>>();
        if fp32 {
            let mut backing = vec![8.; 2562];
            for (i, value) in values.into_iter().enumerate() {
                backing[2560 - i] = value;
            }
            gpu.f32_storage(&backing)?.matrix(MatrixLayout {
                shape: [1, 2560],
                offset: 2560,
                strides: [0, -1],
            })
        } else {
            let mut backing = vec![f16::from_f32(8.).to_bits(); 5123];
            for (i, value) in values.into_iter().enumerate() {
                backing[1 + 2 * i] = f16::from_f32(value).to_bits();
            }
            gpu.f16_storage(&backing)?.matrix(MatrixLayout {
                shape: [1, 2560],
                offset: 1,
                strides: [0, 2],
            })
        }
    }

    fn check_pair(
        control: &mut FlashQsa<'_>,
        candidate: &mut FlashQsa<'_>,
        x: &MatrixView,
        length: u32,
        trace: bool,
    ) -> Result<usize> {
        seed(control, length - 1)?;
        seed(candidate, length - 1)?;
        let gpu = control.gpu;
        let grows = usize::from(length > control.state.slab.capacity);
        let before = gpu.helper_commits.get();
        let expected = control.forward(x, trace)?;
        ensure!(
            gpu.helper_commits.get() - before
                == grows + if control.index.is_some() { 2 } else { 1 }
        );
        let before = gpu.helper_commits.get();
        let count = QSA_APPEND_CHAIN_COUNTS.with(Cell::get);
        let actual = candidate.forward(x, trace)?;
        let chained = candidate.index.is_some() && !trace && length <= 2051;
        ensure!(
            gpu.helper_commits.get() - before
                == grows
                    + if chained || candidate.index.is_none() {
                        1
                    } else {
                        2
                    }
        );
        let after = QSA_APPEND_CHAIN_COUNTS.with(Cell::get);
        ensure!(
            after[0] - count[0] == u64::from(chained) && after[1] - count[1] == u64::from(chained)
        );
        let words = expected.values.read_output()?;
        ensure!(words.len() == 2560 && words.iter().any(|&v| v != 0));
        ensure!(words.iter().all(|&v| if x.dtype == FloatType::F32 {
            f32::from_bits(v).is_finite()
        } else {
            f16::from_bits(v as u16).is_finite()
        }));
        ensure!(actual.values.read_output()? == words && actual.trace == expected.trace);
        let a = retained(gpu, &control.state)?;
        let b = retained(gpu, &candidate.state)?;
        ensure!(a == b && !a.is_empty());
        ensure!(
            control.state.offset == length
                && candidate.state.offset == length
                && candidate.state.slab.published.get() == length
        );
        Ok(a.values().map(Vec::len).sum())
    }

    fn frozen_words(root: &Path, descriptor: &serde_json::Value) -> Result<Vec<u32>> {
        let name = descriptor["file"].as_str().context("frozen QSA filename")?;
        ensure!(Path::new(name).components().count() == 1);
        let bytes = std::fs::read(root.join(name))?;
        let count = descriptor["count"].as_u64().context("frozen QSA extent")?;
        ensure!(
            count == 2560
                && descriptor["shape"] == serde_json::json!([1, 1, 2560])
                && bytes.len() == 2560 * 4
        );
        let words = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        ensure!(words.iter().all(|&v| f32::from_bits(v).is_finite()));
        Ok(words)
    }

    #[test]
    #[ignore = "physical AR36 indexed append/attention boundaries, independent frozen evaluation, state/fork/grow and abandonment; exclusive GPU"]
    fn physical_ar36_qsa_append_chain_preserves_canonical_states_and_rollbacks() -> Result<()> {
        let model = std::env::var("MLXL3_QSA_MODEL")?;
        let c = crate::checkpoint::inspect(Path::new(&model))?;
        let gpu = Metal::new()?;
        let prefix = "model.language_model.layers.15.self_attn";
        let mut control = FlashQsa::load(&gpu, &c, prefix, 1e-6, 1e7, 4096)?;
        let mut candidate = FlashQsa::load(&gpu, &c, prefix, 1e-6, 1e7, 4096)?;
        ensure!(
            candidate.append_chain_enabled && candidate.index.is_some(),
            "explicit QSA_APPEND_CHAIN=1 on M5 required"
        );
        control.append_chain_enabled = false;
        let initial = QSA_APPEND_CHAIN_COUNTS.with(Cell::get);
        let contexts = [3, 4, 32, 33, 1023, 1024, 2047, 2048, 2049, 2050, 2051, 2052];
        let mut state_words = 0usize;
        for fp32 in [false, true] {
            let x = input(&gpu, fp32, 83)?;
            for length in contexts {
                state_words += check_pair(&mut control, &mut candidate, &x, length, false)?;
            }
        }
        let x = input(&gpu, true, 83)?;
        for length in [4, 2051] {
            state_words += check_pair(&mut control, &mut candidate, &x, length, true)?;
        }
        let indexes = (control.index.take(), candidate.index.take());
        state_words += check_pair(&mut control, &mut candidate, &x, 3, false)?;
        control.index = indexes.0;
        candidate.index = indexes.1;

        seed(&mut control, 3)?;
        seed(&mut candidate, 3)?;
        let expected = control.forward(&x, false)?.values.read_output()?;
        let stale = candidate.prepare_append()?;
        let prefix_state = candidate.snapshot();
        ensure!(candidate.forward(&x, false)?.values.read_output()? == expected);
        let retained_suffix = candidate.snapshot();
        let suffix_words = retained(&gpu, &retained_suffix)?;
        candidate.restore(&prefix_state)?;
        let unsubmitted = gpu.queue.new_command_buffer().to_owned();
        ensure!(
            candidate
                .encode_for_consumer(&unsubmitted, &x, stale)
                .is_err()
        );
        ensure!(unsubmitted.status() == MTLCommandBufferStatus::NotEnqueued);
        ensure!(retained(&gpu, &retained_suffix)? == suffix_words);
        seed(&mut candidate, 3)?;
        let before = candidate.snapshot();
        let before_words = retained(&gpu, &before)?;
        candidate.fail_after_index_encode.set(true);
        let commits = gpu.helper_commits.get();
        ensure!(candidate.forward(&x, false).is_err() && gpu.helper_commits.get() == commits);
        ensure!(candidate.state.offset == 3 && Rc::ptr_eq(&candidate.state.slab, &before.slab));
        candidate.fail_after_append.set(true);
        ensure!(candidate.forward(&x, false).is_err());
        ensure!(
            candidate.state.offset == 3
                && before.slab.published.get() == 3
                && retained(&gpu, &before)? == before_words
        );
        ensure!(candidate.forward(&x, false)?.values.read_output()? == expected);
        let original = candidate.snapshot();
        let original_words = retained(&gpu, &original)?;
        candidate.restore(&before)?;
        let y = input(&gpu, true, 181)?;
        let alternate = candidate.forward(&y, false)?.values.read_output()?;
        ensure!(alternate != expected && !Rc::ptr_eq(&candidate.state.slab, &original.slab));
        let alternate_state = candidate.snapshot();
        candidate.restore(&before)?;
        ensure!(candidate.forward(&x, false)?.values.read_output()? == expected);
        ensure!(retained(&gpu, &original)? == original_words);
        candidate.restore(&alternate_state)?;
        control
            .restore(&before)
            .err()
            .context("foreign QSA owner accepted")?;
        seed(&mut control, 3)?;
        control.forward(&y, false)?;
        let z = input(&gpu, true, 227)?;
        ensure!(
            candidate.forward(&z, false)?.values.read_output()?
                == control.forward(&z, false)?.values.read_output()?
        );
        ensure!(retained(&gpu, &candidate.state)? == retained(&gpu, &control.state)?);

        for submitted in [false, true] {
            seed(&mut candidate, 3)?;
            let before = candidate.snapshot();
            let previous = retained(&gpu, &before)?;
            let command = gpu.queue.new_command_buffer().to_owned();
            let pending =
                candidate.encode_for_consumer(&command, &x, candidate.prepare_append()?)?;
            if submitted {
                command.commit();
            }
            drop(pending);
            ensure!(
                command.status()
                    == if submitted {
                        MTLCommandBufferStatus::Completed
                    } else {
                        MTLCommandBufferStatus::NotEnqueued
                    }
            );
            ensure!(
                candidate.state.offset == 3
                    && before.slab.published.get() == 3
                    && retained(&gpu, &before)? == previous
            );
            ensure!(candidate.forward(&x, false)?.values.read_output()? == expected);
        }

        // Independent evaluation capture predates this encoder refactor. Long
        // boundary states above are synthetic and are not an upstream long run.
        let root = PathBuf::from(std::env::var("MLXL3_FLASH_ATTENTION_REFERENCE")?);
        let reference: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("reference.json"))?)?;
        ensure!(reference["status"] == "complete" && reference["mode"] == "eval");
        let rows = reference["attention"]["15"]
            .as_array()
            .context("missing independent QSA reference")?;
        ensure!(rows.len() == 5);
        candidate.reset();
        let mut independent_words = 0;
        for row in rows {
            let words = frozen_words(&root, &row["input"])?;
            let input = gpu
                .f32_storage(&words.into_iter().map(f32::from_bits).collect::<Vec<_>>())?
                .matrix(MatrixLayout {
                    shape: [1, 2560],
                    strides: [2560, 1],
                    offset: 0,
                })?;
            let expected = frozen_words(&root, &row["stages"]["out"])?;
            ensure!(candidate.forward(&input, false)?.values.read_output()? == expected);
            independent_words += expected.len();
        }
        let after = QSA_APPEND_CHAIN_COUNTS.with(Cell::get);
        let counts = std::array::from_fn::<_, 3, _>(|i| after[i] - initial[i]);
        ensure!(
            counts[0] > 0
                && counts[1] > 0
                && counts[2] >= 2
                && independent_words == 12800
                && state_words > 1_000_000
        );
        eprintln!(
            "AR36_QSA_APPEND_CHAIN {}",
            serde_json::json!({"parity":true,"boundary_cases":27,"contexts":contexts,"independent_output_words":independent_words,"synthetic_state_half_words":state_words,"actual_counts":counts,"trace_fallback":true,"unindexed_fallback":true,"sparse_2052_fallback":true,"grow":true,"fork":true,"retained_snapshot":true,"stale_prepared_rejected":true,"encode_failure":true,"post_append_publication_failure":true,"retry":true,"unsubmitted_abandon":true,"submitted_abandon":true,"all_commands_joined":true})
        );
        Ok(())
    }
}
