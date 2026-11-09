//! Resident Qwen3.6-35B-A3B using only owned Metal resources.
use super::{
    FlashProjection, FlashTraceTensor, MappedSwitchGlu, MatrixView, Metal, ProjectionGroup,
    QwenGdn, QwenGdnState, RouteOutput, checkpoint_float_weight, float_values, view::FloatType,
};
use crate::{
    checkpoint::{Checkpoint, TensorInfo},
    qwen35_config::{
        NATIVE_BLOCK_MAX, native_block_end, native_chain_next, native_retained_prefix,
    },
    tensor_layout::MatrixLayout,
};
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use metal::{CommandBufferRef, MTLResourceOptions as Storage};
use std::rc::Rc;

mod kv;
use kv::DeviceKv;
mod mtp;
pub use mtp::{QwenMtpHead, QwenMtpSession};

const HIDDEN: u32 = 2048;
const VOCAB: u32 = 248320;
const EPS: f32 = 1e-6;

fn matrix(gpu: &Metal, values: &[u16], rows: u32, cols: u32) -> Result<MatrixView> {
    ensure!(
        values.len() == rows as usize * cols as usize,
        "invalid Qwen matrix extent"
    );
    gpu.f16_storage(values)?.matrix(MatrixLayout {
        shape: [rows, cols],
        strides: [i64::from(cols), 1],
        offset: 0,
    })
}
fn words(view: &MatrixView) -> Result<Vec<u16>> {
    ensure!(view.dtype == FloatType::F16, "expected Qwen half output");
    Ok(view.read_output()?.into_iter().map(|v| v as u16).collect())
}
fn norm(gpu: &Metal, c: &Checkpoint, name: &str, width: u32) -> Result<MatrixView> {
    let t = c
        .tensors
        .get(name)
        .with_context(|| format!("missing {name}"))?;
    ensure!(t.shape == [width as usize], "invalid {name} shape");
    let values = float_values(t)?
        .into_iter()
        .map(|v| {
            // +1 happens in the source dtype, before the maintained half cast.
            let v = match t.dtype.as_str() {
                "BF16" => bf16::from_f32(v + 1.).to_f32(),
                "F16" => f16::from_f32(v + 1.).to_f32(),
                _ => v + 1.,
            };
            f16::from_f32(v).to_bits()
        })
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|&b| f16::from_bits(b).is_finite()),
        "nonfinite Qwen norm"
    );
    matrix(gpu, &values, 1, width)
}
fn rms(gpu: &Metal, x: &MatrixView, gain: &MatrixView) -> Result<MatrixView> {
    gpu.command(|cmd| encode_rms(gpu, cmd, x, gain))
}
fn encode_rms(
    gpu: &Metal,
    cmd: &CommandBufferRef,
    x: &MatrixView,
    gain: &MatrixView,
) -> Result<MatrixView> {
    gpu.encode_model_rms(cmd, x, gain, EPS, Storage::StorageModeShared)
}

enum Projection<'a> {
    Exl3(FlashProjection<'a>),
    Dense(MatrixView),
}
impl<'a> Projection<'a> {
    fn load(gpu: &'a Metal, c: &Checkpoint, prefix: &str, input: u32, output: u32) -> Result<Self> {
        if c.tensors.contains_key(&format!("{prefix}.trellis")) {
            let p = FlashProjection::load_exl3(gpu, c, prefix)?;
            ensure!(
                p.dimensions() == [input, output],
                "invalid {prefix} dimensions"
            );
            Ok(Self::Exl3(p))
        } else {
            let weight = checkpoint_float_weight(
                gpu,
                c,
                &format!("{prefix}.weight"),
                &[output as usize, input as usize],
                FloatType::F16,
            )?;
            Ok(Self::Dense(MatrixView {
                buffer: weight,
                dtype: FloatType::F16,
                layout: MatrixLayout {
                    shape: [output, input],
                    strides: [i64::from(input), 1],
                    offset: 0,
                },
            }))
        }
    }
    fn forward(&self, gpu: &Metal, x: &MatrixView) -> Result<MatrixView> {
        gpu.command(|cmd| self.encode(gpu, cmd, x))
    }

    fn encode(&self, gpu: &Metal, cmd: &CommandBufferRef, x: &MatrixView) -> Result<MatrixView> {
        self.encode_storage(gpu, cmd, x, Storage::StorageModeShared)
    }

    fn encode_storage(
        &self,
        gpu: &Metal,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        storage: Storage,
    ) -> Result<MatrixView> {
        match self {
            Self::Exl3(p) => p.encode_rows(cmd, x, storage),
            Self::Dense(w) => gpu.encode_model_dense(cmd, x, w, storage),
        }
    }
}

#[derive(Clone)]
struct Kv {
    keys: Vec<u16>,
    values: Vec<u16>,
}
#[derive(Clone)]
enum AttentionCache {
    Host(Rc<Kv>),
    Device(DeviceKv),
}

impl AttentionCache {
    fn host(&self) -> Result<&Kv> {
        match self {
            Self::Host(kv) => Ok(kv),
            Self::Device(_) => anyhow::bail!("expected host Qwen KV"),
        }
    }

    fn device(&self) -> Result<&DeviceKv> {
        match self {
            Self::Device(kv) => Ok(kv),
            Self::Host(_) => anyhow::bail!("expected device Qwen KV"),
        }
    }

    fn retained_bytes(&self) -> usize {
        match self {
            Self::Host(kv) => (kv.keys.capacity() + kv.values.capacity()) * 2,
            Self::Device(kv) => kv.retained_bytes(),
        }
    }

    fn trace_words(&self) -> [std::borrow::Cow<'_, [u16]>; 2] {
        use std::borrow::Cow;
        match self {
            Self::Host(kv) => [Cow::Borrowed(&kv.keys), Cow::Borrowed(&kv.values)],
            Self::Device(kv) => kv.words().map(Cow::Owned),
        }
    }
}

struct AttentionInput {
    query: MatrixView,
    key: MatrixView,
    value: MatrixView,
    gate: MatrixView,
}

impl AttentionInput {
    fn from_projections(qkv: Vec<MatrixView>) -> Result<Self> {
        let [mut query, mut key, value]: [MatrixView; 3] = qkv
            .try_into()
            .map_err(|_| anyhow::anyhow!("missing native Qwen QKV projections"))?;
        // Gate is the second half of each head, not of the whole projection.
        let mut gate = query.clone();
        query.layout = MatrixLayout {
            shape: [16, 256],
            strides: [512, 1],
            offset: 0,
        };
        gate.layout = MatrixLayout {
            offset: 256,
            ..query.layout
        };
        key.layout = MatrixLayout {
            shape: [2, 256],
            strides: [256, 1],
            offset: 0,
        };
        Ok(Self {
            query,
            key,
            value,
            gate,
        })
    }

    fn encode_rope(&mut self, gpu: &Metal, cmd: &CommandBufferRef, offset: u32) -> Result<()> {
        self.query = gpu.encode_model_rope(
            cmd,
            &self.query,
            16,
            256,
            64,
            10000000.,
            offset,
            Storage::StorageModeShared,
        )?;
        self.key = gpu.encode_model_rope(
            cmd,
            &self.key,
            2,
            256,
            64,
            10000000.,
            offset,
            Storage::StorageModeShared,
        )?;
        Ok(())
    }

    fn append(&self, prior: Option<&Kv>, offset: u32) -> Result<Rc<Kv>> {
        let mut next = prior.cloned().unwrap_or(Kv {
            keys: Vec::new(),
            values: Vec::new(),
        });
        ensure!(
            next.keys.len() == offset as usize * 512 && next.values.len() == next.keys.len(),
            "Qwen KV position mismatch"
        );
        next.keys.extend(words(&self.key)?);
        next.values.extend(words(&self.value)?);
        Ok(Rc::new(next))
    }

    fn encode_attend(&self, gpu: &Metal, cmd: &CommandBufferRef, kv: &Kv) -> Result<MatrixView> {
        let length = u32::try_from(kv.keys.len() / 512)?;
        let keys = matrix(gpu, &kv.keys, length, 512)?;
        let values = matrix(gpu, &kv.values, length, 512)?;
        self.encode_attend_views(gpu, cmd, &[keys, values])
    }

    fn encode_attend_views(
        &self,
        gpu: &Metal,
        cmd: &CommandBufferRef,
        [keys, values]: &[MatrixView; 2],
    ) -> Result<MatrixView> {
        let mut attended = gpu.encode_model_gqa(
            cmd,
            &self.query,
            keys,
            values,
            16,
            2,
            256,
            Storage::StorageModeShared,
        )?;
        attended.layout = MatrixLayout {
            shape: [16, 256],
            strides: [256, 1],
            offset: 0,
        };
        Ok(attended)
    }
}

fn flatten_attention(mut x: MatrixView) -> MatrixView {
    x.layout = MatrixLayout {
        shape: [1, 4096],
        strides: [4096, 1],
        offset: 0,
    };
    x
}

struct Attention<'a> {
    qkv: ProjectionGroup<'a>,
    output: Projection<'a>,
    q_norm: MatrixView,
    k_norm: MatrixView,
    kv: Option<AttentionCache>,
    device_limit: Option<u32>,
}
impl<'a> Attention<'a> {
    fn load(
        gpu: &'a Metal,
        c: &Checkpoint,
        prefix: &str,
        device_limit: Option<u32>,
    ) -> Result<Self> {
        let qkv = ProjectionGroup::load(
            gpu,
            c,
            &[
                format!("{prefix}.q_proj"),
                format!("{prefix}.k_proj"),
                format!("{prefix}.v_proj"),
            ],
        )?;
        ensure!(
            qkv.input_width() == HIDDEN && qkv.output_widths() == [8192, 512, 512],
            "invalid Qwen QKV"
        );
        Ok(Self {
            qkv,
            output: Projection::load(gpu, c, &format!("{prefix}.o_proj"), 4096, HIDDEN)?,
            q_norm: norm(gpu, c, &format!("{prefix}.q_norm.weight"), 256)?,
            k_norm: norm(gpu, c, &format!("{prefix}.k_norm.weight"), 256)?,
            kv: None,
            device_limit,
        })
    }
    fn forward(&mut self, gpu: &Metal, x: &MatrixView, offset: u32) -> Result<MatrixView> {
        let mut input = AttentionInput::from_projections(self.qkv.forward(x)?)?;
        input.query = rms(gpu, &input.query, &self.q_norm)?;
        input.key = rms(gpu, &input.key, &self.k_norm)?;
        gpu.command(|cmd| input.encode_rope(gpu, cmd, offset))?;
        let kv = input.append(
            self.kv.as_ref().map(AttentionCache::host).transpose()?,
            offset,
        )?;
        let attended = gpu.command(|cmd| input.encode_attend(gpu, cmd, &kv))?;
        let gated = flatten_attention(gpu.model_binary(&attended, &input.gate, 2)?);
        let result = self.output.forward(gpu, &gated)?;
        self.kv = Some(AttentionCache::Host(kv));
        Ok(result)
    }

    fn encode_input(
        &self,
        gpu: &Metal,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        offset: u32,
    ) -> Result<AttentionInput> {
        let mut input = AttentionInput::from_projections(self.qkv.encode_row(
            cmd,
            x,
            Storage::StorageModeShared,
        )?)?;
        input.query = encode_rms(gpu, cmd, &input.query, &self.q_norm)?;
        input.key = encode_rms(gpu, cmd, &input.key, &self.k_norm)?;
        input.encode_rope(gpu, cmd, offset)?;
        Ok(input)
    }

    fn encode_output(
        &self,
        gpu: &Metal,
        cmd: &CommandBufferRef,
        input: &AttentionInput,
        kv: &Kv,
    ) -> Result<MatrixView> {
        let attended = input.encode_attend(gpu, cmd, kv)?;
        self.encode_gated_output(gpu, cmd, input, &attended)
    }

    fn encode_gated_output(
        &self,
        gpu: &Metal,
        cmd: &CommandBufferRef,
        input: &AttentionInput,
        attended: &MatrixView,
    ) -> Result<MatrixView> {
        let gated = flatten_attention(gpu.encode_model_binary(cmd, attended, &input.gate, 2)?);
        self.output.encode(gpu, cmd, &gated)
    }
}

struct Moe<'a> {
    gate: Projection<'a>,
    experts: MappedSwitchGlu<'a>,
    shared: ProjectionGroup<'a>,
    down: Projection<'a>,
    multiplier: Projection<'a>,
}
impl<'a> Moe<'a> {
    fn load(gpu: &'a Metal, c: &Checkpoint, prefix: &str) -> Result<Self> {
        let shared = ProjectionGroup::load(
            gpu,
            c,
            &[
                format!("{prefix}.shared_expert.gate_proj"),
                format!("{prefix}.shared_expert.up_proj"),
            ],
        )?;
        ensure!(
            shared.input_width() == HIDDEN && shared.output_widths() == [512, 512],
            "invalid shared expert"
        );
        Ok(Self {
            gate: Projection::load(gpu, c, &format!("{prefix}.gate"), HIDDEN, 256)?,
            experts: MappedSwitchGlu::load(
                gpu,
                c,
                &format!("{prefix}.experts"),
                256,
                8,
                ["gate_proj", "up_proj", "down_proj"],
            )?,
            shared,
            down: Projection::load(
                gpu,
                c,
                &format!("{prefix}.shared_expert.down_proj"),
                512,
                HIDDEN,
            )?,
            multiplier: Projection::load(
                gpu,
                c,
                &format!("{prefix}.shared_expert_gate"),
                HIDDEN,
                1,
            )?,
        })
    }
    fn forward(&self, gpu: &Metal, x: &MatrixView, join: bool) -> Result<MatrixView> {
        if join {
            return self.forward_joined(gpu, x);
        }
        let logits = self.gate.forward(gpu, x)?;
        let probabilities = gpu.model_softmax(&logits)?;
        let routes = gpu.route(&probabilities, 8, true)?;
        let ids = routes.indices()?;
        let routed = self.experts.forward(x, &ids, routes.completed_scores()?)?;
        let inputs = self.shared.forward(x)?;
        let activated = gpu.command(|cmd| Self::encode_activation(gpu, cmd, &inputs))?;
        let shared = self.down.forward(gpu, &activated)?;
        let mult = self.multiplier.forward(gpu, x)?;
        let shared = gpu.model_binary(&shared, &mult, 2)?;
        gpu.model_binary(&routed, &shared, 0)
    }

    fn forward_joined(&self, gpu: &Metal, x: &MatrixView) -> Result<MatrixView> {
        let routes = gpu.command(|cmd| self.encode_route(gpu, cmd, x))?;
        let ids = routes.indices()?;
        gpu.command(|cmd| self.encode_experts(gpu, cmd, x, &routes, &ids))
    }

    fn encode_route(
        &self,
        gpu: &Metal,
        cmd: &CommandBufferRef,
        x: &MatrixView,
    ) -> Result<RouteOutput> {
        let logits = self.gate.encode(gpu, cmd, x)?;
        let probabilities = gpu.encode_model_softmax(cmd, &logits)?;
        gpu.encode_route(cmd, &probabilities, 8, true)
    }

    fn encode_experts(
        &self,
        gpu: &Metal,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        routes: &RouteOutput,
        ids: &[u32],
    ) -> Result<MatrixView> {
        let routed = self.experts.encode_row(cmd, x, ids, &routes.scores)?;
        let inputs = self.shared.encode_row(cmd, x, Storage::StorageModeShared)?;
        let activated = Self::encode_activation(gpu, cmd, &inputs)?;
        let shared = self.down.encode(gpu, cmd, &activated)?;
        let mult = self.multiplier.encode(gpu, cmd, x)?;
        let shared = gpu.encode_model_binary(cmd, &shared, &mult, 2)?;
        gpu.encode_model_binary(cmd, &routed, &shared, 0)
    }

    fn encode_activation(
        gpu: &Metal,
        cmd: &CommandBufferRef,
        inputs: &[MatrixView],
    ) -> Result<MatrixView> {
        let buffer = gpu.encode_swiglu(
            cmd,
            &inputs[0].buffer,
            &inputs[1].buffer,
            Storage::StorageModeShared,
        )?;
        Ok(MatrixView {
            buffer,
            dtype: FloatType::F16,
            layout: MatrixLayout {
                shape: [1, 512],
                strides: [512, 1],
                offset: 0,
            },
        })
    }
}
enum Mixer<'a> {
    Gdn(Box<QwenGdn<'a>>),
    Attention(Box<Attention<'a>>),
}
struct Layer<'a> {
    input_norm: MatrixView,
    post_norm: MatrixView,
    mixer: Mixer<'a>,
    mlp: Moe<'a>,
}
struct RoutedRow {
    residual: MatrixView,
    post: MatrixView,
    routes: RouteOutput,
}

struct RoutedBlock {
    rows: Vec<RoutedRow>,
    states: Vec<LayerState>,
}

enum MixerBlock {
    Routed(RoutedBlock),
    Attention(Vec<AttentionInput>),
}

impl Layer<'_> {
    fn encode_route(
        &self,
        gpu: &Metal,
        cmd: &CommandBufferRef,
        x: &MatrixView,
        attention: &MatrixView,
    ) -> Result<RoutedRow> {
        let residual = gpu.encode_model_binary(cmd, x, attention, 0)?;
        let post = encode_rms(gpu, cmd, &residual, &self.post_norm)?;
        let routes = self.mlp.encode_route(gpu, cmd, &post)?;
        Ok(RoutedRow {
            residual,
            post,
            routes,
        })
    }

    fn encode_mixer_block(
        &self,
        gpu: &Metal,
        cmd: &CommandBufferRef,
        rows: &[MatrixView],
        offset: u32,
    ) -> Result<MixerBlock> {
        match &self.mixer {
            Mixer::Gdn(gdn) => {
                let mut previous = gdn.snapshot();
                let mut routed = Vec::with_capacity(rows.len());
                let mut states = Vec::with_capacity(rows.len());
                for x in rows {
                    let input = encode_rms(gpu, cmd, x, &self.input_norm)?;
                    let (next, attention) = gdn.encode_step(cmd, &input, previous.as_ref())?;
                    routed.push(self.encode_route(gpu, cmd, x, &attention)?);
                    previous = Some(next);
                    states.push(LayerState::Gdn(previous.clone()));
                }
                Ok(MixerBlock::Routed(RoutedBlock {
                    rows: routed,
                    states,
                }))
            }
            Mixer::Attention(attention) => {
                if let Some(limit) = attention.device_limit {
                    let mut previous = attention
                        .kv
                        .as_ref()
                        .map(AttentionCache::device)
                        .transpose()?
                        .cloned();
                    let mut routed = Vec::with_capacity(rows.len());
                    let mut states = Vec::with_capacity(rows.len());
                    for (i, x) in rows.iter().enumerate() {
                        let position = offset + i as u32;
                        let normalized = encode_rms(gpu, cmd, x, &self.input_norm)?;
                        let input = attention.encode_input(gpu, cmd, &normalized, position)?;
                        let kv = DeviceKv::append(
                            gpu,
                            cmd,
                            previous.as_ref(),
                            [&input.key, &input.value],
                            position,
                            limit,
                        )?;
                        let attended = input.encode_attend_views(gpu, cmd, &kv.views())?;
                        let output = attention.encode_gated_output(gpu, cmd, &input, &attended)?;
                        routed.push(self.encode_route(gpu, cmd, x, &output)?);
                        states.push(LayerState::Attention(Some(AttentionCache::Device(
                            kv.clone(),
                        ))));
                        previous = Some(kv);
                    }
                    return Ok(MixerBlock::Routed(RoutedBlock {
                        rows: routed,
                        states,
                    }));
                }
                let inputs = rows
                    .iter()
                    .enumerate()
                    .map(|(i, x)| {
                        let input = encode_rms(gpu, cmd, x, &self.input_norm)?;
                        attention.encode_input(gpu, cmd, &input, offset + i as u32)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(MixerBlock::Attention(inputs))
            }
        }
    }

    fn complete_mixer_block(
        &self,
        gpu: &Metal,
        block: MixerBlock,
        rows: &[MatrixView],
        offset: u32,
    ) -> Result<RoutedBlock> {
        match (block, &self.mixer) {
            (MixerBlock::Routed(block), _) => Ok(block),
            (MixerBlock::Attention(inputs), Mixer::Attention(attention)) => {
                let mut previous = match &attention.kv {
                    Some(AttentionCache::Host(kv)) => Some(kv.clone()),
                    None => None,
                    Some(AttentionCache::Device(_)) => anyhow::bail!("expected host Qwen KV"),
                };
                let mut cache = Vec::with_capacity(inputs.len());
                for (i, input) in inputs.iter().enumerate() {
                    let next = input.append(previous.as_deref(), offset + i as u32)?;
                    previous = Some(next.clone());
                    cache.push(next);
                }
                let rows = gpu.command(|cmd| {
                    rows.iter()
                        .zip(&inputs)
                        .zip(&cache)
                        .map(|((x, input), kv)| {
                            let output = attention.encode_output(gpu, cmd, input, kv)?;
                            self.encode_route(gpu, cmd, x, &output)
                        })
                        .collect::<Result<Vec<_>>>()
                })?;
                let states = cache
                    .into_iter()
                    .map(|kv| LayerState::Attention(Some(AttentionCache::Host(kv))))
                    .collect();
                Ok(RoutedBlock { rows, states })
            }
            _ => anyhow::bail!("Qwen mixer block belongs to another layer kind"),
        }
    }

    fn encode_expert_block(
        &self,
        gpu: &Metal,
        cmd: &CommandBufferRef,
        rows: &[RoutedRow],
        indices: &[Vec<u32>],
    ) -> Result<Vec<MatrixView>> {
        rows.iter()
            .zip(indices)
            .map(|(row, ids)| {
                let mlp = self
                    .mlp
                    .encode_experts(gpu, cmd, &row.post, &row.routes, ids)?;
                gpu.encode_model_binary(cmd, &row.residual, &mlp, 0)
            })
            .collect()
    }
}

#[derive(Clone)]
enum LayerState {
    Gdn(Option<QwenGdnState>),
    Attention(Option<AttentionCache>),
}

#[derive(Clone)]
pub struct QwenSnapshot {
    owner: Rc<()>,
    offset: u32,
    states: Vec<LayerState>,
}
impl QwenSnapshot {
    pub fn position(&self) -> u32 {
        self.offset
    }
    pub fn retained_bytes(&self) -> usize {
        // Fixed GDN payload and actual KV backing, including shared snapshots conservatively.
        self.states
            .iter()
            .map(|s| match s {
                LayerState::Gdn(Some(_)) => 32 * 128 * 128 * 4 + 3 * 8192 * 2,
                LayerState::Attention(Some(kv)) => kv.retained_bytes(),
                _ => 0,
            })
            .sum()
    }
}

struct StagedBlock {
    rows: Vec<MatrixView>,
    prefixes: Vec<QwenSnapshot>,
    trace: Vec<Vec<u16>>,
    greedy: Option<metal::Buffer>,
}

/// Holds a staged verification against an exclusively borrowed session.
/// Dropping the block leaves its starting position and states live.
pub struct QwenVerification<'model, 'gpu> {
    model: &'model mut Qwen35<'gpu>,
    prefixes: Vec<QwenSnapshot>,
    rows: Vec<MatrixView>,
    logits: Vec<Vec<u16>>,
}

impl QwenVerification<'_, '_> {
    pub fn logits(&self) -> &[Vec<u16>] {
        &self.logits
    }

    pub fn commit(self, retained: usize) -> Result<()> {
        let index = native_retained_prefix(retained, self.prefixes.len())
            .context("Qwen commit exceeds verified block")?;
        if let Some(index) = index {
            self.model.restore(&self.prefixes[index])?;
        }
        Ok(())
    }
}

// MTP is an independent strict opt-in, not part of the Qwen optimization profile.
fn opt_in(name: &str) -> Result<bool> {
    match std::env::var(name) {
        Ok(value) => crate::flash_support::hc_chain_flag(Some(&value), false),
        Err(std::env::VarError::NotPresent) => crate::flash_support::hc_chain_flag(None, false),
        Err(error) => return Err(error.into()),
    }
    .with_context(|| format!("{name} must be 0 or 1"))
}

pub struct Qwen35<'a> {
    gpu: &'a Metal,
    profile: crate::qwen_profile::QwenProfile,
    embed: TensorInfo,
    head: Projection<'a>,
    norm: MatrixView,
    layers: Vec<Layer<'a>>,
    join_moe: bool,
    join_layers: bool,
    chain_layers: bool,
    prefill_last_head: bool,
    device_greedy: bool,
    #[cfg(test)]
    projected_rows: std::cell::Cell<usize>,
    context: u32,
    offset: u32,
    owner: Rc<()>,
}
impl<'a> Qwen35<'a> {
    pub fn load(gpu: &'a Metal, c: &Checkpoint, context: u32) -> Result<Self> {
        let profile = crate::qwen_profile::QwenProfile::from_environment()?;
        let [
            join_moe,
            join_layers,
            chain_layers,
            _,
            prefill_last_head,
            group_expert_qmv,
            group_expert_stages,
            device_greedy,
        ] = profile.effective(gpu.device_greedy_available(), true);
        let root: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(c.path.join("config.json"))?)?;
        let config = &root["text_config"];
        ensure!(
            c.model_type == "qwen3_5_moe" && root["model_type"] == "qwen3_5_moe",
            "native Qwen requires qwen3_5_moe"
        );
        ensure!(
            crate::qwen35_config::attention_widths(16, 2, 256) == Some([8192, 512, 4096])
                && crate::qwen35_config::gated_delta_widths(16, 32, 128, 128)
                    == Some([2048, 4096, 8192]),
            "invalid native Qwen geometry contract"
        );
        for (name, expected) in [
            ("hidden_size", 2048),
            ("num_hidden_layers", 40),
            ("num_experts", 256),
            ("num_experts_per_tok", 8),
            ("moe_intermediate_size", 512),
            ("shared_expert_intermediate_size", 512),
            ("vocab_size", 248320),
            ("num_attention_heads", 16),
            ("num_key_value_heads", 2),
            ("head_dim", 256),
            ("linear_num_key_heads", 16),
            ("linear_num_value_heads", 32),
            ("linear_key_head_dim", 128),
            ("linear_value_head_dim", 128),
            ("linear_conv_kernel_dim", 4),
        ] {
            ensure!(
                config[name].as_u64() == Some(expected),
                "unsupported native Qwen {name}"
            );
        }
        ensure!(
            config["tie_word_embeddings"] == false
                && config["attention_bias"] == false
                && config["partial_rotary_factor"].as_f64() == Some(0.25)
                && config["rms_norm_eps"].as_f64().map(|v| v as f32) == Some(EPS)
                && config["rope_parameters"]["rope_theta"].as_f64() == Some(10000000.),
            "unsupported native Qwen numerical config"
        );
        let kinds = config["layer_types"]
            .as_array()
            .context("missing Qwen layer types")?;
        ensure!(
            kinds.len() == 40
                && kinds.iter().enumerate().all(|(i, k)| k
                    == if i % 4 == 3 {
                        "full_attention"
                    } else {
                        "linear_attention"
                    }),
            "unsupported Qwen layer layout"
        );
        let limit = config["max_position_embeddings"]
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .filter(|&v| v > 0)
            .context("invalid Qwen context")?;
        let context = if context == 0 { limit } else { context };
        ensure!(
            context > 0 && context <= limit,
            "native Qwen context exceeds checkpoint limit"
        );
        let device_limit =
            (join_layers && profile.flags[3] && context <= crate::flash_support::FLASH_MAX_CONTEXT)
                .then_some(context);
        let embed = c
            .tensors
            .get("model.language_model.embed_tokens.weight")
            .context("missing Qwen embeddings")?
            .clone();
        ensure!(
            embed.shape == [VOCAB as usize, HIDDEN as usize]
                && matches!(embed.dtype.as_str(), "BF16" | "F16"),
            "invalid Qwen embedding"
        );
        let head = Projection::load(gpu, c, "lm_head", HIDDEN, VOCAB)?;
        let norm = norm(gpu, c, "model.language_model.norm.weight", HIDDEN)?;
        let mut layers = Vec::with_capacity(40);
        for i in 0..40 {
            let prefix = format!("model.language_model.layers.{i}");
            let mixer = if i % 4 == 3 {
                Mixer::Attention(Box::new(Attention::load(
                    gpu,
                    c,
                    &format!("{prefix}.self_attn"),
                    device_limit,
                )?))
            } else {
                Mixer::Gdn(Box::new(QwenGdn::load(
                    gpu,
                    c,
                    &format!("{prefix}.linear_attn"),
                    HIDDEN,
                    16,
                    32,
                    4,
                    EPS,
                )?))
            };
            let mut mlp = Moe::load(gpu, c, &format!("{prefix}.mlp"))?;
            mlp.experts.enable_grouped_qmv(group_expert_qmv);
            mlp.experts.enable_grouped_stages(group_expert_stages);
            layers.push(Layer {
                input_norm: norm_fn(gpu, c, &format!("{prefix}.input_layernorm.weight"))?,
                post_norm: norm_fn(gpu, c, &format!("{prefix}.post_attention_layernorm.weight"))?,
                mixer,
                mlp,
            });
        }
        let model = Self {
            gpu,
            profile,
            embed,
            head,
            norm,
            layers,
            join_moe,
            join_layers,
            chain_layers,
            prefill_last_head,
            device_greedy,
            #[cfg(test)]
            projected_rows: std::cell::Cell::new(0),
            context,
            offset: 0,
            owner: Rc::new(()),
        };
        eprintln!(
            "Native Qwen profile={} effective={}",
            model.profile_name(),
            model.effective_flags()
        );
        Ok(model)
    }
    pub fn profile_name(&self) -> &'static str {
        self.profile.name
    }

    pub fn effective_flags(&self) -> serde_json::Value {
        serde_json::json!({
            "moe_join": self.join_moe, "layer_join": self.join_layers,
            "layer_chain": self.chain_layers, "device_kv": self.device_kv_enabled(),
            "prefill_last_head": self.prefill_last_head,
            "group_expert_qmv": self.grouped_expert_qmv_enabled(),
            "group_expert_stages": self.grouped_expert_stages_enabled(),
            "device_greedy": self.device_greedy,
        })
    }

    pub fn context_limit(&self) -> u32 {
        self.context
    }
    pub fn offset(&self) -> u32 {
        self.offset
    }
    pub fn memory_stats(&self) -> Result<crate::runtime_stats::MemoryStats> {
        self.gpu.memory_stats()
    }
    pub fn device_greedy_enabled(&self) -> bool {
        self.device_greedy
    }

    pub fn device_greedy_eligible(
        &self,
        temperature: f32,
        top_k: usize,
        penalty: f32,
        speculative: bool,
    ) -> bool {
        crate::gpu_greedy_policy::eligible(
            self.device_greedy,
            temperature,
            top_k,
            penalty,
            speculative,
        )
    }

    pub fn device_greedy_counts(&self) -> [u64; 2] {
        [
            self.gpu.device_greedy_calls.get(),
            self.gpu.device_greedy_heads.get(),
        ]
    }

    pub fn greedy_from_logits(&self, logits: &[u16]) -> Result<u32> {
        Self::check_logits(logits, 1)?;
        self.gpu.device_greedy_from_logits(logits)
    }

    pub fn forward_greedy(&mut self, token: u32) -> Result<u32> {
        ensure!(
            self.device_greedy,
            "Qwen device greedy is disabled or unsupported"
        );
        let staged = self.stage_block_inner(&[token], false, true)?;
        let selected = staged.greedy.context("missing Qwen device greedy result")?;
        let index = self.gpu.read_device_greedy(&selected)?;
        self.restore(&staged.prefixes[0])?;
        let heads = &self.gpu.device_greedy_heads;
        heads.set(heads.get().saturating_add(1));
        Ok(index)
    }

    pub fn log_probs(&self, logits: &[u16]) -> Result<Vec<f32>> {
        self.gpu.flash_log_probs(logits)
    }
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Mixer::Gdn(g) => g.reset(),
                Mixer::Attention(a) => a.kv = None,
            }
        }
        self.offset = 0;
    }
    pub fn snapshot(&self) -> QwenSnapshot {
        QwenSnapshot {
            owner: self.owner.clone(),
            offset: self.offset,
            states: self
                .layers
                .iter()
                .map(|l| match &l.mixer {
                    Mixer::Gdn(g) => LayerState::Gdn(g.snapshot()),
                    Mixer::Attention(a) => LayerState::Attention(a.kv.clone()),
                })
                .collect(),
        }
    }
    pub fn restore(&mut self, s: &QwenSnapshot) -> Result<()> {
        ensure!(
            Rc::ptr_eq(&self.owner, &s.owner)
                && s.states.len() == self.layers.len()
                && s.offset <= self.context,
            "foreign Qwen snapshot"
        );
        for (l, state) in self.layers.iter_mut().zip(&s.states) {
            match (&mut l.mixer, state) {
                (Mixer::Gdn(g), LayerState::Gdn(s)) => g.restore(s.clone())?,
                (Mixer::Attention(a), LayerState::Attention(s)) => a.kv = s.clone(),
                _ => anyhow::bail!("invalid Qwen snapshot layer"),
            }
        }
        self.offset = s.offset;
        Ok(())
    }
    pub fn device_kv_enabled(&self) -> bool {
        self.layers
            .iter()
            .any(|layer| matches!(&layer.mixer, Mixer::Attention(a) if a.device_limit.is_some()))
    }

    pub fn grouped_expert_qmv_enabled(&self) -> bool {
        self.layers
            .iter()
            .all(|layer| layer.mlp.experts.grouped_qmv_enabled())
    }

    pub fn grouped_expert_stages_enabled(&self) -> bool {
        self.layers
            .iter()
            .all(|layer| layer.mlp.experts.grouped_stages_enabled())
    }

    pub fn prefill_last_head_enabled(&self) -> bool {
        self.prefill_last_head
    }

    /// Cancellation is polled between these bounded chunks by the bridge.
    pub fn prefill_chunk_size(&self) -> usize {
        if self.join_layers {
            NATIVE_BLOCK_MAX
        } else {
            1
        }
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>> {
        self.prefill_cancellable(tokens, || false)
    }

    pub fn prefill_cancellable(
        &mut self,
        tokens: &[u32],
        cancelled: impl Fn() -> bool,
    ) -> Result<Vec<u16>> {
        ensure!(
            !tokens.is_empty()
                && tokens.iter().all(|&t| t < VOCAB)
                && tokens.len() <= (self.context - self.offset) as usize,
            "invalid Qwen prefill tokens or context"
        );
        let before = self.snapshot();
        let result = (|| {
            let mut logits = None;
            let chunks = tokens.chunks(self.prefill_chunk_size());
            let count = chunks.len();
            for (i, chunk) in chunks.enumerate() {
                ensure!(!cancelled(), "generation cancelled");
                if self.prefill_last_head {
                    let staged = self.stage_block(chunk, false)?;
                    if crate::flash_support::prefill_head_required(i, count, true) == Some(true) {
                        let last = staged
                            .rows
                            .last()
                            .context("missing Qwen final prompt row")?;
                        logits = self.project_block(std::slice::from_ref(last))?.0.pop();
                    }
                    self.restore(
                        staged
                            .prefixes
                            .last()
                            .context("missing Qwen prompt state")?,
                    )?;
                } else if chunk.len() == 1 {
                    logits = Some(self.forward(chunk[0])?);
                } else {
                    let mut verified = self.verify_block(chunk)?;
                    logits = verified.logits.pop();
                    verified.commit(chunk.len())?;
                }
            }
            ensure!(!cancelled(), "generation cancelled");
            logits.context("empty Qwen prefill")
        })();
        if result.is_err() {
            self.restore(&before)?;
        }
        result
    }
    pub fn forward(&mut self, token: u32) -> Result<Vec<u16>> {
        Ok(self.run(token, false)?.0)
    }
    pub fn forward_trace(&mut self, token: u32) -> Result<(Vec<u16>, Vec<FlashTraceTensor>)> {
        self.forward_trace_export(token, Self::state_trace)
    }
    fn forward_trace_export(
        &mut self,
        token: u32,
        export: impl FnOnce(&Self) -> Result<Vec<FlashTraceTensor>>,
    ) -> Result<(Vec<u16>, Vec<FlashTraceTensor>)> {
        // Keep rollback state until fallible private-to-shared export completes.
        let before = self.snapshot();
        crate::transaction::with_rollback(
            self,
            &before,
            |model| {
                let (logits, _) = model.run(token, false)?;
                Ok((logits, export(model)?))
            },
            Self::restore,
        )
    }
    pub fn layer_trace(&mut self, token: u32) -> Result<(Vec<u16>, Vec<Vec<u16>>)> {
        self.run(token, true)
    }
    fn embedding(&self, token: u32) -> Result<(Vec<u16>, MatrixView)> {
        let (file, base) = self.embed.source()?;
        let mut bytes = vec![0; HIDDEN as usize * 2];
        use std::os::unix::fs::FileExt;
        let at = base
            .checked_add(u64::from(token) * u64::from(HIDDEN) * 2)
            .context("Qwen embedding offset overflow")?;
        file.read_exact_at(&mut bytes, at)?;
        let bits = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| {
                if self.embed.dtype == "BF16" {
                    f16::from_f32(bf16::from_le_bytes(b).to_f32()).to_bits()
                } else {
                    u16::from_le_bytes(b)
                }
            })
            .collect::<Vec<_>>();
        let x = matrix(self.gpu, &bits, 1, HIDDEN)?;
        Ok((bits, x))
    }

    pub fn verify_block(&mut self, tokens: &[u32]) -> Result<QwenVerification<'_, 'a>> {
        let block = self.stage_block(tokens, false)?;
        let (logits, _) = self.project_block(&block.rows)?;
        Ok(QwenVerification {
            model: self,
            prefixes: block.prefixes,
            rows: block.rows,
            logits,
        })
    }

    fn stage_block(&self, tokens: &[u32], trace: bool) -> Result<StagedBlock> {
        self.stage_block_inner(tokens, trace, false)
    }

    fn stage_block_inner(
        &self,
        tokens: &[u32],
        trace: bool,
        select_greedy: bool,
    ) -> Result<StagedBlock> {
        ensure!(
            !select_greedy || (tokens.len() == 1 && !trace),
            "invalid Qwen scalar head request"
        );
        let end = native_block_end(tokens, VOCAB, self.offset, self.context)
            .context("Qwen block must fit the context and contain 1 to 4 valid tokens")?;
        let mut prefixes = (self.offset + 1..=end)
            .map(|offset| QwenSnapshot {
                owner: self.owner.clone(),
                offset,
                states: Vec::with_capacity(self.layers.len()),
            })
            .collect::<Vec<_>>();
        let mut rows = Vec::with_capacity(tokens.len());
        let mut traces = Vec::new();
        for &token in tokens {
            let (bits, x) = self.embedding(token)?;
            if trace {
                traces.push(bits);
            }
            rows.push(x);
        }
        let gpu = self.gpu;
        let mut pending = None;
        let mut greedy = None;
        for (index, layer) in self.layers.iter().enumerate() {
            let block = match pending.take() {
                Some(block) => block,
                None => {
                    gpu.command(|cmd| layer.encode_mixer_block(gpu, cmd, &rows, self.offset))?
                }
            };
            let RoutedBlock {
                rows: routed,
                states,
            } = layer.complete_mixer_block(gpu, block, &rows, self.offset)?;
            let indices = routed
                .iter()
                .map(|row| row.routes.indices())
                .collect::<Result<Vec<_>>>()?;
            rows = gpu.command(|cmd| {
                let output = layer.encode_expert_block(gpu, cmd, &routed, &indices)?;
                if let Some(next) =
                    native_chain_next(index, self.layers.len(), self.chain_layers && !trace)
                {
                    pending = Some(self.layers[next].encode_mixer_block(
                        gpu,
                        cmd,
                        &output,
                        self.offset,
                    )?);
                }
                if select_greedy && index + 1 == self.layers.len() {
                    let normalized = encode_rms(gpu, cmd, &output[0], &self.norm)?;
                    let logits = self.head.encode_storage(
                        gpu,
                        cmd,
                        &normalized,
                        Storage::StorageModePrivate,
                    )?;
                    greedy = Some(gpu.encode_normalized_greedy(cmd, &logits.buffer, true)?);
                }
                Ok(output)
            })?;
            for (prefix, state) in prefixes.iter_mut().zip(states) {
                prefix.states.push(state);
            }
            if trace {
                for row in &rows {
                    traces.push(words(row)?);
                }
            }
        }
        Ok(StagedBlock {
            rows,
            prefixes,
            trace: traces,
            greedy,
        })
    }

    fn project_block(&self, rows: &[MatrixView]) -> Result<(Vec<Vec<u16>>, MatrixView)> {
        let gpu = self.gpu;
        let (logits, norm) = gpu.command(|cmd| {
            let norm = if let [single] = rows {
                encode_rms(gpu, cmd, single, &self.norm)?
            } else {
                let row_bytes = u64::from(HIDDEN) * 2;
                let packed = MatrixView::output(
                    gpu,
                    FloatType::F16,
                    u32::try_from(rows.len())?,
                    HIDDEN,
                    Storage::StorageModeShared,
                )?;
                let normalized = rows
                    .iter()
                    .map(|x| encode_rms(gpu, cmd, x, &self.norm))
                    .collect::<Result<Vec<_>>>()?;
                let blit = cmd.new_blit_command_encoder();
                for (i, source) in normalized.iter().enumerate() {
                    blit.copy_from_buffer(
                        &source.buffer,
                        0,
                        &packed.buffer,
                        i as u64 * row_bytes,
                        row_bytes,
                    );
                }
                blit.end_encoding();
                packed
            };
            Ok((self.head.encode(gpu, cmd, &norm)?, norm))
        })?;
        let values = words(&logits)?;
        Self::check_logits(&values, rows.len())?;
        #[cfg(test)]
        self.projected_rows
            .set(self.projected_rows.get() + rows.len());
        let logits = if rows.len() == 1 {
            vec![values]
        } else {
            values
                .as_chunks::<{ VOCAB as usize }>()
                .0
                .iter()
                .map(|row| row.to_vec())
                .collect()
        };
        Ok((logits, norm))
    }

    fn check_logits(values: &[u16], rows: usize) -> Result<()> {
        ensure!(
            values.len() == rows * VOCAB as usize
                && values.iter().all(|&b| f16::from_bits(b).is_finite()),
            "nonfinite or truncated Qwen logits"
        );
        Ok(())
    }

    fn run(&mut self, token: u32, trace: bool) -> Result<(Vec<u16>, Vec<Vec<u16>>)> {
        let end = crate::qwen35_config::native_step_end(token, VOCAB, self.offset, self.context)
            .context("Qwen token outside vocabulary or context full")?;
        if self.join_layers {
            let mut staged = self.stage_block(&[token], trace)?;
            let (mut logits, normalized) = self.project_block(&staged.rows)?;
            if trace {
                staged.trace.push(words(&normalized)?);
            }
            let output = logits.pop().context("missing Qwen logits")?;
            self.restore(&staged.prefixes[0])?;
            return Ok((output, staged.trace));
        }
        let before = self.snapshot();
        let result = (|| {
            let (bits, mut x) = self.embedding(token)?;
            let mut traces = Vec::new();
            if trace {
                traces.push(bits);
            }
            for l in &mut self.layers {
                let normalized = rms(self.gpu, &x, &l.input_norm)?;
                let attention = match &mut l.mixer {
                    Mixer::Gdn(g) => g.forward(&normalized)?,
                    Mixer::Attention(a) => a.forward(self.gpu, &normalized, self.offset)?,
                };
                let hidden = self.gpu.model_binary(&x, &attention, 0)?;
                let post = rms(self.gpu, &hidden, &l.post_norm)?;
                let mlp = l.mlp.forward(self.gpu, &post, self.join_moe)?;
                x = self.gpu.model_binary(&hidden, &mlp, 0)?;
                if trace {
                    traces.push(words(&x)?);
                }
            }
            let normalized = rms(self.gpu, &x, &self.norm)?;
            if trace {
                traces.push(words(&normalized)?);
            }
            let logits = words(&self.head.forward(self.gpu, &normalized)?)?;
            Self::check_logits(&logits, 1)?;
            Ok((logits, traces))
        })();
        if result.is_err() {
            self.restore(&before)?;
        } else {
            self.offset = end;
        }
        result
    }
    pub fn state_trace(&self) -> Result<Vec<FlashTraceTensor>> {
        let mut out = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            match &l.mixer {
                Mixer::Gdn(g) => {
                    if let Some(s) = g.states()? {
                        out.push(FlashTraceTensor {
                            name: format!("layer.{i}.conv"),
                            shape: vec![1, 3, 8192],
                            data: s.conv_state.iter().flat_map(|b| b.to_le_bytes()).collect(),
                        });
                        out.push(FlashTraceTensor {
                            name: format!("layer.{i}.recurrent"),
                            shape: vec![1, 32, 128, 128],
                            data: s.state.iter().flat_map(|b| b.to_le_bytes()).collect(),
                        });
                    }
                }
                Mixer::Attention(a) => {
                    if let Some(kv) = &a.kv {
                        for (name, data) in ["keys", "values"].into_iter().zip(kv.trace_words()) {
                            ensure!(
                                data.len() == self.offset as usize * 512,
                                "Qwen traced KV extent differs from cursor"
                            );
                            // Export in maintained [batch,kv_heads,time,dim] order.
                            let mut bytes = Vec::with_capacity(data.len() * 2);
                            for head in 0..2 {
                                for t in 0..self.offset as usize {
                                    for d in 0..256 {
                                        bytes.extend_from_slice(
                                            &data[t * 512 + head * 256 + d].to_le_bytes(),
                                        );
                                    }
                                }
                            }
                            out.push(FlashTraceTensor {
                                name: format!("layer.{i}.{name}"),
                                shape: vec![1, 2, self.offset, 256],
                                data: bytes,
                            });
                        }
                    }
                }
            }
        }
        Ok(out)
    }
}
fn norm_fn(gpu: &Metal, c: &Checkpoint, name: &str) -> Result<MatrixView> {
    norm(gpu, c, name, HIDDEN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "physical Qwen checkpoint: compare the two real MoE submission schedules"]
    fn joins_qwen_moe_commands_without_changing_outputs() -> Result<()> {
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_QWEN_TEST_MODEL")?))?;
        let gpu = Metal::new()?;
        let input = (0..HIDDEN)
            .map(|i| f16::from_f32(((i * 17 + 11) % 113) as f32 / 64. - 1.).to_bits())
            .collect::<Vec<_>>();
        let x = matrix(&gpu, &input, 1, HIDDEN)?;
        let bad = matrix(&gpu, &[0; 32], 1, 32)?;
        for layer in [0, 35] {
            let moe = Moe::load(
                &gpu,
                &c,
                &format!("model.language_model.layers.{layer}.mlp"),
            )?;
            let forward = |joined, expected_commits| -> Result<Vec<u16>> {
                let before = gpu.helper_commits.get();
                let output = moe.forward(&gpu, &x, joined)?;
                ensure!(
                    gpu.helper_commits.get() - before == expected_commits,
                    "unexpected Qwen MoE helper submissions on layer {layer}"
                );
                ensure!(
                    output.shape() == [1, HIDDEN],
                    "invalid Qwen MoE output shape"
                );
                let bits = words(&output)?;
                ensure!(
                    bits.len() == HIDDEN as usize
                        && bits.iter().all(|&bit| f16::from_bits(bit).is_finite()),
                    "empty, truncated or nonfinite Qwen MoE output"
                );
                Ok(bits)
            };
            let reference = forward(false, 9)?;
            let before = gpu.helper_commits.get();
            ensure!(
                moe.forward(&gpu, &bad, true).is_err(),
                "invalid input accepted"
            );
            ensure!(
                gpu.helper_commits.get() == before,
                "invalid input submitted"
            );
            ensure!(
                forward(true, 2)? == reference,
                "Qwen MoE differs on layer {layer}"
            );
        }
        println!("Qwen MoE: 2 layers, 4096 exact finite FP16 words, helper submissions 9 -> 2");
        Ok(())
    }
    fn check_oracle_state(
        model: &Qwen35<'_>,
        root: &Path,
        expected: &serde_json::Value,
    ) -> Result<usize> {
        let trace = model.state_trace()?;
        let mut compared = 0usize;
        let states = expected["states"]
            .as_array()
            .context("missing oracle states")?;
        ensure!(
            trace.len() == 80 && states.len() == 80,
            "incomplete state coverage"
        );
        for (actual, expected) in trace.iter().zip(states) {
            let shape: Vec<u32> = serde_json::from_value(expected["shape"].clone())?;
            let count = shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
                .context("state overflow")?;
            let item = if actual.name.ends_with("recurrent") {
                4
            } else {
                2
            };
            ensure!(
                actual.name == expected["name"]
                    && actual.shape == shape
                    && count > 0
                    && actual.data.len() == count * item,
                "invalid state {}",
                actual.name
            );
            let finite = if item == 4 {
                actual
                    .data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|b| f32::from_le_bytes(*b).is_finite())
            } else {
                actual
                    .data
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .all(|b| f16::from_le_bytes(*b).is_finite())
            };
            ensure!(finite, "nonfinite state {}", actual.name);
            ensure!(
                actual.data
                    == std::fs::read(
                        root.join(expected["file"].as_str().context("missing state file")?)
                    )?,
                "Qwen state differs at offset {}: {}",
                model.offset(),
                actual.name
            );
            compared += count;
        }
        Ok(compared)
    }

    #[test]
    #[ignore = "physical M5 and Qwen checkpoint: strict scalar selection and rollback"]
    fn qwen_device_greedy_preserves_ties_and_prefix_on_failure() -> Result<()> {
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_QWEN_TEST_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut model = Qwen35::load(&gpu, &c, 8)?;
        assert!(model.device_greedy_eligible(0., 1, 1., false));
        assert!(!model.device_greedy_eligible(0., 1, 1.1, false));
        assert!(!model.device_greedy_eligible(0., 1, 1., true));

        let select = |words: &[u16]| -> Result<u32> {
            let raw = crate::gpu::upload(&gpu.device, words)?;
            let selected = gpu.command(|cmd| gpu.encode_normalized_greedy(cmd, &raw, true))?;
            gpu.read_device_greedy(&selected)
        };
        let mut tie = vec![0; VOCAB as usize];
        tie[1] = f16::from_f32(0.001).to_bits();
        assert_eq!(select(&tie)?, 0);
        for invalid in [f16::NEG_INFINITY, f16::INFINITY, f16::NAN] {
            tie[12345] = invalid.to_bits();
            assert!(select(&tie).is_err());
        }

        model.prefill(&[63611, 222460])?;
        let prefix = model.snapshot();
        let before = gpu.helper_commits.get();
        let logits = model.forward(177571)?;
        let full_commits = gpu.helper_commits.get() - before;
        let expected = crate::speculative::greedy_log_prob_id(&model.log_probs(&logits)?)
            .context("missing reference token")?;
        let states = model.state_trace()?;
        model.restore(&prefix)?;
        gpu.device_greedy_reject_result.set(true);
        let rejected = model.forward_greedy(177571);
        gpu.device_greedy_reject_result.set(false);
        assert!(rejected.is_err());
        assert_eq!(model.offset(), 2);
        let before = gpu.helper_commits.get();
        assert_eq!(model.forward_greedy(177571)?, expected);
        assert_eq!(gpu.helper_commits.get() - before + 1, full_commits);
        assert_eq!(model.offset(), 3);
        assert_eq!(model.device_greedy_counts()[1], 1);
        let actual = model.state_trace()?;
        assert_eq!(actual.len(), 80);
        assert_eq!(states.len(), 80);
        for (a, e) in actual.iter().zip(states) {
            assert_eq!(a.name, e.name);
            assert_eq!(a.shape, e.shape);
            assert!(!a.data.is_empty());
            assert_eq!(a.data, e.data);
        }
        println!("Qwen strict greedy ties, nonfinite rejection, rollback and one fewer command");
        Ok(())
    }

    #[test]
    #[ignore = "physical Qwen checkpoint: joined/nonjoined export failure and retry"]
    fn qwen_failed_trace_export_preserves_state_and_continuation() -> Result<()> {
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_QWEN_TEST_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut model = Qwen35::load(&gpu, &c, 16)?;
        for joined in [false, true] {
            model.reset();
            model.join_layers = joined;
            model.prefill(&[63611, 222460])?;
            let prefix = model.snapshot();
            let before = model.state_trace()?;
            let expected = model.forward_trace(177571)?;
            let continuation = model.forward(644)?;
            model.restore(&prefix)?;
            let failed =
                model.forward_trace_export(177571, |_| anyhow::bail!("injected export failure"));
            assert!(failed.is_err());
            assert_eq!(model.offset(), 2);
            let same = |a: &[FlashTraceTensor], b: &[FlashTraceTensor]| {
                assert!(!a.is_empty());
                assert_eq!(a.len(), b.len());
                for (a, b) in a.iter().zip(b) {
                    assert_eq!(a.name, b.name);
                    assert_eq!(a.shape, b.shape);
                    assert_eq!(a.data, b.data);
                }
            };
            same(&model.state_trace()?, &before);
            let actual = model.forward_trace(177571)?;
            assert_eq!(actual.0, expected.0);
            same(&actual.1, &expected.1);
            assert_eq!(model.forward(644)?, continuation);
        }
        Ok(())
    }

    #[test]
    #[ignore = "physical Qwen checkpoint: final prefill head and rollback"]
    fn qwen_prefill_projects_only_the_last_row_and_preserves_resume() -> Result<()> {
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_QWEN_TEST_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut model = Qwen35::load(&gpu, &c, 16)?;
        ensure!(model.join_layers, "test requires joined Qwen");
        let tokens = [63611, 222460, 177571, 644, 52543, 13, 436, 82458, 15];
        let same_state = |model: &Qwen35<'_>, expected: Vec<FlashTraceTensor>| -> Result<()> {
            let actual = model.state_trace()?;
            assert_eq!(actual.len(), 80);
            assert_eq!(expected.len(), 80);
            for (a, e) in actual.iter().zip(expected) {
                assert_eq!(a.name, e.name);
                assert_eq!(a.shape, e.shape);
                assert!(!a.data.is_empty());
                assert_eq!(a.data, e.data);
            }
            Ok(())
        };
        for width in [1, 2, 4, 5, 9] {
            model.reset();
            model.prefill_last_head = false;
            let before = model.projected_rows.get();
            let expected = model.prefill(&tokens[..width])?;
            assert_eq!(model.projected_rows.get() - before, width);
            let expected_state = model.state_trace()?;
            model.reset();
            model.prefill_last_head = true;
            let before = model.projected_rows.get();
            let actual = model.prefill(&tokens[..width])?;
            assert_eq!(model.projected_rows.get() - before, 1);
            assert_eq!(actual.len(), VOCAB as usize);
            assert!(actual.iter().all(|&word| f16::from_bits(word).is_finite()));
            assert_eq!(actual, expected);
            same_state(&model, expected_state)?;
        }
        model.reset();
        model.prefill(&tokens[..2])?;
        let at_two = model.snapshot();
        let states = model.state_trace()?;
        let before = model.projected_rows.get();
        let calls = std::cell::Cell::new(0);
        let cancelled = model.prefill_cancellable(&tokens[2..], || {
            calls.set(calls.get() + 1);
            calls.get() > 1
        });
        assert_eq!(cancelled.unwrap_err().to_string(), "generation cancelled");
        assert_eq!(calls.get(), 2);
        assert_eq!(model.offset(), 2);
        assert_eq!(model.projected_rows.get(), before);
        same_state(&model, states)?;
        assert!(model.prefill(&[tokens[2], VOCAB]).is_err());
        assert!(model.prefill(&[0; 15]).is_err());
        assert_eq!(model.offset(), 2);
        let recovered = model.prefill(&tokens[2..])?;
        model.restore(&at_two)?;
        model.prefill_last_head = false;
        let expected = model.prefill(&tokens[2..])?;
        assert_eq!(recovered, expected);
        assert_eq!(model.offset(), 9);
        println!(
            "Qwen final-head rows verified at 1/2/4/5/9, exact final states and cancellation rollback"
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical Qwen model: exact verification and every accepted prefix"]
    fn native_block_verifies_and_commits_only_the_selected_prefix() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("MLXL3_QWEN_NATIVE_ORACLE")?);
        let status: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("status.json"))?)?;
        ensure!(
            status["status"] == "captured",
            "incomplete independent oracle"
        );
        let steps: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(steps.len() == 4, "incomplete oracle steps");
        let tokens = steps
            .iter()
            .map(|step| -> Result<u32> {
                Ok(u32::try_from(
                    step["token"].as_u64().context("missing oracle token")?,
                )?)
            })
            .collect::<Result<Vec<_>>>()?;
        let oracle = (0..4)
            .map(|i| std::fs::read(root.join(format!("{i}-logits.bin"))))
            .collect::<std::io::Result<Vec<_>>>()?;
        let equal = |index: usize, words: &[u16]| {
            words.len() == VOCAB as usize
                && oracle[index].len() == VOCAB as usize * 2
                && words.iter().all(|&v| f16::from_bits(v).is_finite())
                && words
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .eq(oracle[index].iter().copied())
        };
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_QWEN_TEST_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut model = Qwen35::load(&gpu, &c, 4)?;
        let device_layers = model
            .layers
            .iter()
            .filter(|layer| matches!(&layer.mixer, Mixer::Attention(a) if a.device_limit.is_some()))
            .count();
        let submissions = (if model.chain_layers { 52 } else { 91 }) - device_layers;
        let mut compared_states = 0usize;
        let mut compared_logits = 0usize;
        for width in 2..=4 {
            for retained in 0..=width {
                model.reset();
                let before = gpu.helper_commits.get();
                let block = model.verify_block(&tokens[..width])?;
                ensure!(
                    gpu.helper_commits.get() - before == submissions,
                    "unexpected Qwen grouped submission count"
                );
                ensure!(block.logits().len() == width, "missing verification rows");
                for (i, row) in block.logits().iter().enumerate() {
                    ensure!(
                        equal(i, row),
                        "logits differ at block width {width} row {i}"
                    );
                    compared_logits += row.len();
                }
                block.commit(retained)?;
                ensure!(
                    model.offset() as usize == retained,
                    "verification committed another prefix"
                );
                if retained == 0 {
                    ensure!(
                        model.state_trace()?.is_empty(),
                        "zero prefix published state"
                    );
                } else {
                    compared_states += check_oracle_state(&model, &root, &steps[retained - 1])?;
                }
                if retained < 4 {
                    ensure!(
                        equal(retained, &model.forward(tokens[retained])?),
                        "continuation differs"
                    );
                    compared_states += check_oracle_state(&model, &root, &steps[retained])?;
                }
            }
        }
        model.reset();
        model.forward(tokens[0])?;
        ensure!(model.verify_block(&[]).is_err() && model.verify_block(&[VOCAB]).is_err());
        ensure!(
            model.verify_block(&tokens).is_err(),
            "over-context block accepted"
        );
        drop(model.verify_block(&tokens[1..3])?);
        ensure!(
            model.verify_block(&tokens[1..3])?.commit(3).is_err(),
            "over-commit accepted"
        );
        ensure!(
            model.offset() == 1,
            "failed or dropped verification published state"
        );
        compared_states += check_oracle_state(&model, &root, &steps[0])?;
        let suffix = model.verify_block(&tokens[1..])?;
        for (i, row) in suffix.logits().iter().enumerate() {
            ensure!(equal(i + 1, row), "nonzero-offset block differs");
            compared_logits += row.len();
        }
        suffix.commit(3)?;
        compared_states += check_oracle_state(&model, &root, &steps[3])?;
        println!(
            "Qwen verified M2/M3/M4 with every accepted prefix, {compared_logits} exact finite logits and {compared_states} exact finite state words, {submissions} submissions per block"
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical Qwen model and independent MLX oracle: full logits/layers/states and transactions"]
    fn complete_session_matches_independent_mlx_and_restores_branches() -> Result<()> {
        let root = std::path::PathBuf::from(std::env::var("MLXL3_QWEN_NATIVE_ORACLE")?);
        let status: serde_json::Value =
            serde_json::from_slice(&std::fs::read(root.join("status.json"))?)?;
        ensure!(
            status["status"] == "captured",
            "incomplete independent oracle"
        );
        let steps: Vec<serde_json::Value> =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
        ensure!(steps.len() == 4, "incomplete oracle steps");
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_QWEN_TEST_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut model = Qwen35::load(&gpu, &c, 4)?;
        let empty = model.snapshot();
        let mut compared = 0usize;
        let mut branch = None;
        for (step, expected) in steps.iter().enumerate() {
            let token = expected["token"].as_u64().context("missing oracle token")? as u32;
            let (logits, layers) = model.layer_trace(token)?;
            let reference = std::fs::read(root.join(format!("{step}-logits.bin")))?;
            ensure!(
                logits.len() == 248320
                    && reference.len() == 248320 * 2
                    && logits.iter().all(|&v| f16::from_bits(v).is_finite()),
                "invalid native logits extent"
            );
            let logits_equal = logits.iter().flat_map(|v| v.to_le_bytes()).eq(reference);
            ensure!(layers.len() == 42, "missing Qwen layers");
            for (i, layer) in layers.iter().enumerate() {
                let reference = std::fs::read(root.join(format!("{step}-layer-{i}.bin")))?;
                ensure!(
                    layer.len() == 2048
                        && reference.len() == 4096
                        && layer.iter().all(|&v| f16::from_bits(v).is_finite()),
                    "invalid layer {i}"
                );
                if !layer
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .eq(reference.iter().copied())
                {
                    let actual = root
                        .parent()
                        .context("missing oracle parent")?
                        .join(format!("failed-{step}-layer-{i}.bin"));
                    std::fs::write(
                        actual,
                        layer
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )?;
                    let bad = layer
                        .iter()
                        .zip(reference.as_chunks::<2>().0)
                        .filter(|(a, b)| a.to_le_bytes() != **b)
                        .count();
                    anyhow::bail!("Qwen layer differs at step {step} layer {i}: {bad} words");
                }
            }
            ensure!(logits_equal, "Qwen logits differ at step {step}");
            compared += check_oracle_state(&model, &root, expected)?;
            if step == 1 {
                branch = Some(model.snapshot());
            }
            let before = model.snapshot();
            ensure!(model.forward(VOCAB).is_err(), "invalid token accepted");
            ensure!(
                model.offset() == before.position(),
                "rejected token advanced position"
            );
        }
        let final_state = model.state_trace()?;
        ensure!(
            model.forward(0).is_err() && model.offset() == 4,
            "full context advanced"
        );
        model.restore(&branch.context("missing branch")?)?;
        model.forward(42)?;
        model.forward(11)?;
        let again = model.state_trace()?;
        ensure!(
            again
                .iter()
                .zip(&final_state)
                .all(|(a, b)| a.name == b.name && a.shape == b.shape && a.data == b.data),
            "restored branch differs"
        );
        model.restore(&empty)?;
        ensure!(
            model.offset() == 0 && model.state_trace()?.is_empty(),
            "empty snapshot did not clear state"
        );
        let logits = model.prefill(&[248044, 1, 42, 11])?;
        ensure!(
            logits
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .eq(std::fs::read(root.join("3-logits.bin"))?),
            "prefill differs"
        );
        let before = model.snapshot();
        ensure!(model.prefill(&[]).is_err() && model.offset() == before.position());
        model.reset();
        ensure!(
            model.prefill(&[1, VOCAB]).is_err()
                && model.offset() == 0
                && model.state_trace()?.is_empty()
        );
        // Foreign owner check occurs before restoring any layer, without loading another GPU model.
        let mut foreign = empty.clone();
        foreign.owner = Rc::new(());
        ensure!(model.restore(&foreign).is_err() && model.offset() == 0);
        eprintln!(
            "Qwen native checked 4 steps, 993280 logits, 168 layers, {compared} finite state words; branch/reset/error/prefill passed"
        );
        Ok(())
    }
    use std::path::Path;
}
