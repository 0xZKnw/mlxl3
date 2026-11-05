//! Resident Qwen3.6-35B-A3B using only owned Metal resources.
use super::{
    FlashProjection, FlashTraceTensor, MappedSwitchGlu, MatrixView, Metal, ProjectionGroup,
    QwenGdn, QwenGdnState, checkpoint_float_weight, float_values, view::FloatType,
};
use crate::{
    checkpoint::{Checkpoint, TensorInfo},
    tensor_layout::MatrixLayout,
};
use anyhow::{Context, Result, ensure};
use half::{bf16, f16};
use metal::MTLResourceOptions as Storage;
use std::rc::Rc;

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
    gpu.command(|cmd| gpu.encode_model_rms(cmd, x, gain, EPS, Storage::StorageModeShared))
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
        match self {
            Self::Exl3(p) => gpu.command(|cmd| p.encode_row(cmd, x, Storage::StorageModeShared)),
            Self::Dense(w) => {
                gpu.command(|cmd| gpu.encode_model_dense(cmd, x, w, Storage::StorageModeShared))
            }
        }
    }
}

#[derive(Clone)]
struct Kv {
    keys: Vec<u16>,
    values: Vec<u16>,
}
struct Attention<'a> {
    qkv: ProjectionGroup<'a>,
    output: Projection<'a>,
    q_norm: MatrixView,
    k_norm: MatrixView,
    kv: Option<Rc<Kv>>,
}
impl<'a> Attention<'a> {
    fn load(gpu: &'a Metal, c: &Checkpoint, prefix: &str) -> Result<Self> {
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
        })
    }
    fn forward(&mut self, gpu: &Metal, x: &MatrixView, offset: u32) -> Result<MatrixView> {
        let qkv = self.qkv.forward(x)?;
        let mut query = qkv[0].clone();
        query.layout = MatrixLayout {
            shape: [16, 256],
            strides: [512, 1],
            offset: 0,
        };
        let q = rms(gpu, &query, &self.q_norm)?;
        let mut key = qkv[1].clone();
        key.layout = MatrixLayout {
            shape: [2, 256],
            strides: [256, 1],
            offset: 0,
        };
        let k = rms(gpu, &key, &self.k_norm)?;
        let (q, k) = gpu.command(|cmd| {
            Ok((
                gpu.encode_model_rope(
                    cmd,
                    &q,
                    16,
                    256,
                    64,
                    10000000.,
                    offset,
                    Storage::StorageModeShared,
                )?,
                gpu.encode_model_rope(
                    cmd,
                    &k,
                    2,
                    256,
                    64,
                    10000000.,
                    offset,
                    Storage::StorageModeShared,
                )?,
            ))
        })?;
        let mut kv = self.kv.as_deref().cloned().unwrap_or(Kv {
            keys: Vec::new(),
            values: Vec::new(),
        });
        ensure!(
            kv.keys.len() == offset as usize * 512 && kv.values.len() == kv.keys.len(),
            "Qwen KV position mismatch"
        );
        kv.keys.extend(words(&k)?);
        kv.values.extend(words(&qkv[2])?);
        let keys = matrix(gpu, &kv.keys, offset + 1, 512)?;
        let values = matrix(gpu, &kv.values, offset + 1, 512)?;
        let attended = gpu.command(|cmd| {
            gpu.encode_model_gqa(
                cmd,
                &q,
                &keys,
                &values,
                16,
                2,
                256,
                Storage::StorageModeShared,
            )
        })?;
        // Gate is the second half of EACH head, not the second half of the vector.
        let mut gate = qkv[0].clone();
        gate.layout = MatrixLayout {
            shape: [16, 256],
            strides: [512, 1],
            offset: 256,
        };
        let mut attended = attended;
        attended.layout = MatrixLayout {
            shape: [16, 256],
            strides: [256, 1],
            offset: 0,
        };
        let mut gated = gpu.model_binary(&attended, &gate, 2)?;
        gated.layout = MatrixLayout {
            shape: [1, 4096],
            strides: [4096, 1],
            offset: 0,
        };
        let result = self.output.forward(gpu, &gated)?;
        self.kv = Some(Rc::new(kv));
        Ok(result)
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
    fn forward(&self, gpu: &Metal, x: &MatrixView) -> Result<MatrixView> {
        let logits = self.gate.forward(gpu, x)?;
        let probabilities = gpu.model_softmax(&logits)?;
        let routes = gpu.route(&probabilities, 8, true)?;
        let ids = routes.indices()?;
        let routed = self.experts.forward(x, &ids, routes.completed_scores()?)?;
        let inputs = self.shared.forward(x)?;
        let activated = gpu.command(|cmd| {
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
        })?;
        let shared = self.down.forward(gpu, &activated)?;
        let mult = self.multiplier.forward(gpu, x)?;
        let shared = gpu.model_binary(&shared, &mult, 2)?;
        gpu.model_binary(&routed, &shared, 0)
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
#[derive(Clone)]
enum LayerState {
    Gdn(Option<QwenGdnState>),
    Attention(Option<Rc<Kv>>),
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
        // Fixed GDN payload and exact initialized host KV, including shared snapshots conservatively.
        self.states
            .iter()
            .map(|s| match s {
                LayerState::Gdn(Some(_)) => 32 * 128 * 128 * 4 + 3 * 8192 * 2,
                LayerState::Attention(Some(kv)) => (kv.keys.capacity() + kv.values.capacity()) * 2,
                _ => 0,
            })
            .sum()
    }
}

pub struct Qwen35<'a> {
    gpu: &'a Metal,
    embed: TensorInfo,
    head: Projection<'a>,
    norm: MatrixView,
    layers: Vec<Layer<'a>>,
    context: u32,
    offset: u32,
    owner: Rc<()>,
}
impl<'a> Qwen35<'a> {
    pub fn load(gpu: &'a Metal, c: &Checkpoint, context: u32) -> Result<Self> {
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
                )?))
            } else {
                Mixer::Gdn(Box::new(QwenGdn::from_checkpoint(
                    gpu,
                    c,
                    &format!("{prefix}.linear_attn"),
                )?))
            };
            layers.push(Layer {
                input_norm: norm_fn(gpu, c, &format!("{prefix}.input_layernorm.weight"))?,
                post_norm: norm_fn(gpu, c, &format!("{prefix}.post_attention_layernorm.weight"))?,
                mixer,
                mlp: Moe::load(gpu, c, &format!("{prefix}.mlp"))?,
            });
        }
        Ok(Self {
            gpu,
            embed,
            head,
            norm,
            layers,
            context,
            offset: 0,
            owner: Rc::new(()),
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
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<u16>> {
        ensure!(
            !tokens.is_empty()
                && tokens.iter().all(|&t| t < VOCAB)
                && tokens.len() <= (self.context - self.offset) as usize,
            "invalid Qwen prefill tokens or context"
        );
        let before = self.snapshot();
        let result = (|| {
            let mut logits = None;
            for &t in tokens {
                logits = Some(self.forward(t)?);
            }
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
        let (logits, _) = self.run(token, false)?;
        Ok((logits, self.state_trace()?))
    }
    pub fn layer_trace(&mut self, token: u32) -> Result<(Vec<u16>, Vec<Vec<u16>>)> {
        self.run(token, true)
    }
    fn run(&mut self, token: u32, trace: bool) -> Result<(Vec<u16>, Vec<Vec<u16>>)> {
        let end = crate::qwen35_config::native_step_end(token, VOCAB, self.offset, self.context)
            .context("Qwen token outside vocabulary or context full")?;
        let before = self.snapshot();
        let result = (|| {
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
            let mut x = matrix(self.gpu, &bits, 1, HIDDEN)?;
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
                let mlp = l.mlp.forward(self.gpu, &post)?;
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
            ensure!(
                logits.len() == VOCAB as usize
                    && logits.iter().all(|&b| f16::from_bits(b).is_finite()),
                "nonfinite or truncated Qwen logits"
            );
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
                        for (name, data) in [("keys", &kv.keys), ("values", &kv.values)] {
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
            let trace = model.state_trace()?;
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
                    "Qwen state differs at step {step}: {}",
                    actual.name
                );
                compared += count;
            }
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
