use super::{FlashAttentionOutput, FlashProjection, MatrixView, Metal, view::FloatType};
use crate::{checkpoint::Checkpoint, flash_support::qsa_append};
use anyhow::{Context, Result, ensure};
use metal::{Buffer, ComputePipelineState, MTLResourceOptions as Storage, MTLSize};
use std::{cell::Cell, collections::BTreeMap, rc::Rc};

pub(super) struct Pipelines {
    index_rope: ComputePipelineState,
    pool: ComputePipelineState,
    scores: ComputePipelineState,
    nax: std::cell::OnceCell<ComputePipelineState>,
    norm: ComputePipelineState,
    freq: ComputePipelineState,
    rope: ComputePipelineState,
    attention: ComputePipelineState,
    partials: ComputePipelineState,
    finish: ComputePipelineState,
    gate: ComputePipelineState,
}
impl Pipelines {
    pub(super) fn load(p: &impl Fn(&str) -> Result<ComputePipelineState>) -> Result<Self> {
        Ok(Self {
            index_rope: p("qsa_index_rope")?,
            pool: p("qsa_pool")?,
            scores: p("qsa_index_scores")?,
            nax: std::cell::OnceCell::new(),
            norm: p("qsa_norm")?,
            freq: p("qsa_freq")?,
            rope: p("qsa_rope")?,
            attention: p("qsa_attention")?,
            partials: p("qsa_attention_partials")?,
            finish: p("qsa_attention_finish")?,
            gate: p("qsa_gate")?,
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
        let (tail, score_view, qk, block) = gpu.command(|cmd| {
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
            Ok((tail, scores, qk, block))
        })?;
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

pub struct FlashQsa<'a> {
    arch_class: u8,
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
        Ok(Self {
            arch_class,
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
        let offset = self.state.offset;
        let prior = &self.state.slab;
        let (at, fork) = qsa_append(offset, prior.published.get(), self.capacity)
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
        let index = self
            .index
            .as_ref()
            .map(|index| index.append(gpu, &slab, &self.state, x, self.eps, trace))
            .transpose()?;
        let positions = index
            .as_ref()
            .and_then(|s| s.positions.as_deref())
            .map(|ids| super::upload(&gpu.device, ids))
            .transpose()?;
        let selected = index
            .as_ref()
            .and_then(|s| s.positions.as_ref())
            .map_or(offset + 1, |ids| ids.len() as u32);
        let storage = if trace {
            Storage::StorageModeShared
        } else {
            Storage::StorageModePrivate
        };
        let half = |n| MatrixView::output(gpu, FloatType::F16, 1, n, storage);
        let groups = |n| MTLSize::new(n, 1, 1);
        let (values, steps) = gpu.command(|cmd| {
            let qg = self.q.encode_row(cmd, x, storage)?;
            let k = self.k.encode_row(cmd, x, storage)?;
            let v = self.v.encode_row(cmd, x, storage)?;
            let prepare =
                |input: &MatrixView, gain: &Buffer, heads: u32, stride: u32| -> Result<_> {
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
            let (qn, qr) = prepare(&qg, &self.q_norm, 24, 512)?;
            let (kn, kr) = prepare(&k, &self.k_norm, 2, 256)?;
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer(&kr.buffer, 0, &slab.keys, at, 1024);
            blit.copy_from_buffer(&v.buffer, 0, &slab.values, at, 1024);
            blit.end_encoding();
            let attention = half(6144)?;
            let ids = positions.as_ref().unwrap_or(&qr.buffer);
            let sparse = u32::from(positions.is_some());
            if let Some(blocks) =
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
            let gated = half(6144)?;
            Metal::encode(
                cmd,
                &gpu.qsa.gate,
                &[&attention.buffer, &qg.buffer, &gated.buffer],
                [256u32],
                groups(192),
                32,
            );
            let out = self
                .out
                .encode_row(cmd, &gated, Storage::StorageModeShared)?;
            let values = if x.dtype == FloatType::F32 {
                let f =
                    MatrixView::output(gpu, FloatType::F32, 1, 2560, Storage::StorageModeShared)?;
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
                trace.then_some([
                    ("q_proj", qg),
                    ("q_norm", qn),
                    ("q_rope", qr),
                    ("k_proj", k),
                    ("k_norm", kn),
                    ("k_rope", kr),
                    ("v_proj", v),
                    ("gated", gated),
                ]),
            ))
        })?;
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
