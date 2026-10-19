//! Flash Next text inference with exact EXL3 experts and disk-backed PLE.
//! Architecture: Qwen / mlx-vlm (Apple, MIT); see THIRD_PARTY_NOTICES.md.
use crate::{
    array::{self, Array, Dtype},
    checkpoint::{Checkpoint, TensorInfo},
    codec::{self, Codebook},
    flash_support, gated_delta,
    lfm2::{Projection, half_weight},
    linear::checkpoint_array,
    ssd_moe::SsdMoe,
    ssd_prefetch::ReaderPool,
};
use anyhow::{Context, Result, ensure};
use half::f16;
use serde_json::Value;
use std::{
    cell::RefCell,
    collections::{BTreeMap, VecDeque},
    fs::File,
    os::unix::fs::FileExt,
    path::PathBuf,
    rc::Rc,
    sync::atomic::Ordering,
};

const H: i32 = 2560;
const HC: i32 = 4;
const WIDTH: i32 = H * HC;
const EOS: u32 = 248044;

struct Norm {
    gain: Array,
    width: i32,
    groups: i32,
    eps: f32,
    zero_centered: bool,
}

impl Norm {
    fn load(
        c: &Checkpoint,
        name: &str,
        width: i32,
        groups: i32,
        eps: f32,
        zero_centered: bool,
    ) -> Result<Self> {
        let raw = checkpoint_array(c, name)?;
        ensure!(
            raw.shape() == [width * groups]
                && matches!(
                    raw.dtype(),
                    Dtype::Float16 | Dtype::BFloat16 | Dtype::Float32
                ),
            "invalid Flash norm {name}"
        );
        let raw = raw.astype(Dtype::Float32)?;
        let gain = raw;
        gain.eval()?;
        Ok(Self {
            gain,
            width,
            groups,
            eps,
            zero_centered,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        ensure!(
            x.shape().last() == Some(&(self.width * self.groups)),
            "invalid Flash norm input shape"
        );
        let mut shape = x.shape().to_vec();
        if self.groups > 1 {
            *shape.last_mut().context("empty Flash norm input")? = self.groups;
            shape.push(self.width);
        }
        let grouped = x.reshape(&shape)?;
        let value = if self.zero_centered {
            let gain = if self.groups > 1 {
                self.gain.reshape(&[self.groups, self.width])?
            } else {
                self.gain.try_clone()?
            };
            grouped.qwen4_rms_norm(&gain, self.eps)?
        } else {
            grouped.rms_norm(&self.gain, self.eps)?
        };
        value.reshape(x.shape())
    }
}

struct Hc {
    norm: Norm,
    down: Projection,
    up: Projection,
    inject: Option<Projection>,
}
impl Hc {
    fn load(c: &Checkpoint, p: &str, eps: f32, combine: bool) -> Result<Self> {
        Ok(Self {
            norm: Norm::load(c, &format!("{p}.hc_norm.weight"), H, HC, eps, true)?,
            down: Projection::load(c, &format!("{p}.input_mix_weight_down"), WIDTH, 320, false)?,
            up: Projection::load(c, &format!("{p}.input_mix_weight_up"), 320, WIDTH, false)?,
            inject: combine
                .then(|| Projection::load(c, &format!("{p}.block_inject_weight"), WIDTH, HC, false))
                .transpose()?,
        })
    }
    fn read(&self, streams: &Array) -> Result<(Array, Option<Array>)> {
        let normalized = self.norm.forward(&streams.reshape(&[1, 1, WIDTH])?)?;
        let weights = self.up.forward_flash_hc(
            &self
                .down
                .forward_flash_hc(&normalized)?
                .qwen4_hc_activate()?,
        )?;
        let mixed = weights.qwen4_hc_mix(&normalized)?.reshape(&[1, H])?;
        let injection = self
            .inject
            .as_ref()
            .map(|p| {
                p.forward_flash_hc(&normalized)?
                    .qwen4_hc_gate()?
                    .reshape(&[1, HC])
            })
            .transpose()?;
        Ok((mixed, injection))
    }
    fn write(streams: &Array, branch: &Array, gate: &Array) -> Result<Array> {
        streams.add(
            &branch
                .reshape(&[1, 1, H])?
                .mul(&gate.reshape(&[1, HC, 1])?)?
                .reshape(&[1, WIDTH])?,
        )
    }
}

struct Gdn {
    #[cfg(test)]
    stages: Vec<(String, Array)>,
    qkv: Projection,
    z: Projection,
    a: Projection,
    b: Projection,
    out: Projection,
    conv: Array,
    a_log: Array,
    dt_bias: Array,
    norm: Norm,
    state: Option<Array>,
    history: Option<Array>,
}
impl Gdn {
    fn load(c: &Checkpoint, p: &str, eps: f32) -> Result<Self> {
        let proj =
            |name: &str, output| Projection::load(c, &format!("{p}.{name}"), H, output, false);
        Ok(Self {
            #[cfg(test)]
            stages: Vec::new(),
            qkv: proj("in_proj_qkv", 10240)?,
            z: proj("in_proj_z", 6144)?,
            a: proj("in_proj_a", 48)?,
            b: proj("in_proj_b", 48)?,
            out: Projection::load(c, &format!("{p}.out_proj"), 6144, H, false)?,
            conv: half_weight(c, &format!("{p}.conv1d.weight"), Some(&[10240, 1, 4]))?
                .transpose(&[0, 2, 1])?,
            a_log: checkpoint_array(c, &format!("{p}.A_log"))?.astype(Dtype::Float32)?,
            dt_bias: half_weight(c, &format!("{p}.dt_bias"), Some(&[48]))?,
            norm: Norm::load(c, &format!("{p}.norm.weight"), 128, 1, eps, false)?,
            state: None,
            history: None,
        })
    }
    fn forward(&mut self, x: &Array) -> Result<Array> {
        #[cfg(test)]
        self.stages.clear();
        macro_rules! stage {
            ($name:expr, $value:expr) => {
                #[cfg(test)]
                self.stages.push(($name.into(), $value.try_clone()?));
            };
        }
        let half_input = x.astype(Dtype::Float16)?;
        let x = &half_input;
        let qkv = self
            .qkv
            .forward(x)?
            .astype(Dtype::Float16)?
            .reshape(&[1, 1, 10240])?;
        stage!("qkv", qkv);
        let history = match &self.history {
            Some(h) => h.try_clone()?,
            None => Array::zeros_dtype(&[1, 3, 10240], Dtype::Float16)?,
        };
        let input = Array::concatenate(&[&history, &qkv], 1)?;
        self.history = Some(input.slice(1, 1, 4)?);
        let conv = input
            .astype(Dtype::Float32)?
            .mul(&self.conv.transpose(&[2, 1, 0])?.astype(Dtype::Float32)?)?
            .sum(1, true)?
            .astype(Dtype::Float16)?
            .silu()?;
        stage!("conv", conv);
        let q = conv.slice(2, 0, 2048)?.reshape(&[1, 1, 16, 128])?;
        let k = conv.slice(2, 2048, 4096)?.reshape(&[1, 1, 16, 128])?;
        let v = conv.slice(2, 4096, 10240)?.reshape(&[1, 1, 48, 128])?;
        let norm = |value: &Array, scale: f32| -> Result<Array> {
            let eps = Array::from_f16_bits(&[f16::from_f32(1e-6).to_bits()], &[])?;
            let inverse = value.mul(value)?.sum(3, true)?.add(&eps)?.rsqrt()?;
            value.mul(&inverse)?.scalar_mul(scale)
        };
        let q = norm(&q, 1. / 128f32.sqrt())?;
        let k = norm(&k, 1.)?;
        stage!("q", q);
        stage!("k", k);
        stage!("v", v);
        let state = match &self.state {
            Some(s) => s.try_clone()?,
            None => Array::zeros_dtype(&[1, 48, 128, 128], Dtype::Float32)?,
        };
        let a = self
            .a
            .forward(x)?
            .astype(Dtype::Float16)?
            .reshape(&[1, 1, 48])?;
        let b = self
            .b
            .forward(x)?
            .astype(Dtype::Float16)?
            .reshape(&[1, 1, 48])?;
        stage!("a", a);
        stage!("b", b);
        let (value, state) =
            gated_delta::step_with_gates(&q, &k, &v, &a, &b, &self.a_log, &self.dt_bias, &state)?;
        stage!("delta", value);
        self.state = Some(state);
        let z = self
            .z
            .forward(x)?
            .astype(Dtype::Float16)?
            .reshape(&[1, 1, 48, 128])?;
        stage!("z", z);
        let normed = self.norm.forward(&value)?;
        stage!("normed", normed);
        let gated = normed
            .astype(Dtype::Float32)?
            .mul(&z.astype(Dtype::Float32)?.sigmoid()?)?
            .astype(Dtype::Float16)?;
        stage!("gated", gated);
        self.out
            .forward(&gated.reshape(&[1, 6144])?)?
            .astype(Dtype::Float32)
    }
    fn reset(&mut self) {
        self.state = None;
        self.history = None;
    }
}

struct Qsa {
    #[cfg(test)]
    stages: Vec<(String, Array)>,
    q: Projection,
    k: Projection,
    v: Projection,
    out: Projection,
    index: Projection,
    q_norm: Norm,
    k_norm: Norm,
    iq_norm: Norm,
    ik_norm: Norm,
    keys: Option<Array>,
    values: Option<Array>,
    index_tail: Option<Array>,
    blocks: Option<Array>,
    offset: i32,
    frequencies: [Array; 2],
}

struct QsaState {
    keys: Option<Array>,
    values: Option<Array>,
    index_tail: Option<Array>,
    blocks: Option<Array>,
    offset: i32,
}

fn clone_array(value: &Option<Array>) -> Result<Option<Array>> {
    value.as_ref().map(Array::try_clone).transpose()
}

impl Qsa {
    fn snapshot(&self) -> Result<QsaState> {
        Ok(QsaState {
            keys: clone_array(&self.keys)?,
            values: clone_array(&self.values)?,
            index_tail: clone_array(&self.index_tail)?,
            blocks: clone_array(&self.blocks)?,
            offset: self.offset,
        })
    }
    fn restore(&mut self, state: QsaState) {
        self.keys = state.keys;
        self.values = state.values;
        self.index_tail = state.index_tail;
        self.blocks = state.blocks;
        self.offset = state.offset;
    }
    fn load(c: &Checkpoint, p: &str, eps: f32, rope: f32) -> Result<Self> {
        let proj =
            |name: &str, output| Projection::load(c, &format!("{p}.{name}"), H, output, false);
        let base = Array::from_f32(&[rope], &[])?;
        let frequencies = [32, 64].map(|dimensions| -> Result<Array> {
            let exponents = (0..dimensions / 2)
                .map(|i| (2 * i) as f32 / dimensions as f32)
                .collect::<Vec<_>>();
            let inverse = Array::from_f32(&[1.], &[])?
                .div(&base.pow(&Array::from_f32(&exponents, &[dimensions / 2])?)?)?;
            inverse.eval()?;
            Ok(inverse)
        });
        let [small, large] = frequencies;
        let frequencies = [small?, large?];
        Ok(Self {
            #[cfg(test)]
            stages: Vec::new(),
            q: proj("q_proj", 12288)?,
            k: proj("k_proj", 512)?,
            v: proj("v_proj", 512)?,
            out: Projection::load(c, &format!("{p}.o_proj"), 6144, H, false)?,
            index: proj("indexer.index_qk_proj", 640)?,
            q_norm: Norm::load(c, &format!("{p}.q_norm.weight"), 256, 1, eps, true)?,
            k_norm: Norm::load(c, &format!("{p}.k_norm.weight"), 256, 1, eps, true)?,
            iq_norm: Norm::load(
                c,
                &format!("{p}.indexer.q_layernorm.weight"),
                128,
                1,
                eps,
                true,
            )?,
            ik_norm: Norm::load(
                c,
                &format!("{p}.indexer.k_layernorm.weight"),
                128,
                1,
                eps,
                true,
            )?,
            keys: None,
            values: None,
            index_tail: None,
            blocks: None,
            offset: 0,
            frequencies,
        })
    }
    fn rope(&self, x: &Array, dimensions: i32, offset: i32) -> Result<Array> {
        let [batch, heads, time, width]: [i32; 4] = x
            .shape()
            .try_into()
            .map_err(|_| anyhow::anyhow!("invalid Flash RoPE rank"))?;
        ensure!(
            batch == 1
                && time == 1
                && heads > 0
                && width >= dimensions
                && offset >= 0
                && x.dtype() == Dtype::Float16,
            "invalid Flash RoPE input"
        );
        let frequencies = match dimensions {
            32 => &self.frequencies[0],
            64 => &self.frequencies[1],
            _ => anyhow::bail!("unsupported Flash rotary width"),
        };
        let work = heads
            .checked_mul(width - dimensions / 2)
            .context("Flash RoPE grid overflow")?;
        array::metal_kernel(
            &format!("mlxl3_flash_mrope_r{dimensions}_h{heads}_w{width}"),
            &["x", "inv_freq", "position"],
            &["out"],
            &format!(
                "#define ROTARY {dimensions}
#define WIDTH {width}
#define HEADS {heads}
"
            ),
            include_str!("../shaders/flash_rope.metal"),
            &[x, frequencies, &Array::from_i32(&[offset], &[1])?],
            &[x.shape().to_vec()],
            &[Dtype::Float16],
            [work, 1, 1],
            [256, 1, 1],
        )?
        .pop()
        .context("missing Flash RoPE output")
    }
    fn append(old: &Option<Array>, value: &Array, axis: i32) -> Result<Array> {
        match old {
            Some(old) => Array::concatenate(&[old, value], axis),
            None => value.try_clone(),
        }
    }
    fn forward(&mut self, x: &Array) -> Result<Array> {
        #[cfg(test)]
        self.stages.clear();
        macro_rules! stage {
            ($name:expr, $value:expr) => {
                #[cfg(test)]
                self.stages.push(($name.into(), $value.try_clone()?));
            };
        }
        let offset = self.offset;
        let qk = self.append_cache(x)?;
        let qg = self
            .q
            .forward(x)?
            .astype(Dtype::Float16)?
            .reshape(&[1, 1, 24, 512])?;
        let gate = qg.slice(3, 256, 512)?.reshape(&[1, 6144])?;
        stage!("q_proj", qg);
        let query_norm = self.q_norm.forward(&qg.slice(3, 0, 256)?)?;
        stage!("q_norm", query_norm);
        let query = self.rope(&query_norm.transpose(&[0, 2, 1, 3])?, 64, offset)?;
        stage!("q_rope", query);
        let keys = self.keys.as_ref().context("missing QSA keys")?;
        let values = self.values.as_ref().context("missing QSA values")?;
        let count = (offset + 1) / 4;
        let (attn_keys, attn_values) = if count > 512 {
            let iq = self.rope(
                &self
                    .iq_norm
                    .forward(&qk.slice(2, 0, 4)?)?
                    .transpose(&[0, 2, 1, 3])?,
                32,
                offset,
            )?;
            let scores = iq.astype(Dtype::Float32)?.matmul(
                &self
                    .blocks
                    .as_ref()
                    .context("missing QSA index blocks")?
                    .astype(Dtype::Float32)?
                    .transpose(&[0, 1, 3, 2])?,
            )?;
            let reduced=array::metal_kernel(&format!("mlxl3_flash_qsa_scores_{count}"),&["scores"],&["out"],&format!("#define BLOCKS {count}\n"),
                "uint i=thread_position_in_grid.x;if(i<BLOCKS){float value=0;for(uint h=0;h<4;h++)value+=max(scores[h*BLOCKS+i],0.0f);out[i]=value*0.08838834764831844f;}",
                &[&scores],&[vec![count]],&[Dtype::Float32],[count,1,1],[32,1,1])?.remove(0);
            let mut blocks = reduced.argsort()?.slice(0, count - 512, count)?.to_u32()?;
            blocks.sort_unstable();
            let mut tokens = Vec::with_capacity(2051);
            for block in blocks {
                for step in 0..4 {
                    tokens.push(block * 4 + step);
                }
            }
            tokens.extend((count * 4..offset + 1).map(|i| i as u32));
            let selected = Array::from_u32(&tokens, &[tokens.len() as i32])?;
            (keys.take(&selected, 2)?, values.take(&selected, 2)?)
        } else {
            (keys.try_clone()?, values.try_clone()?)
        };
        let attention = Array::sdpa(&query, &attn_keys, &attn_values, 1. / 16., false)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, 6144])?;
        let gated = attention.mul(&gate.sigmoid()?)?;
        stage!("gated", gated);
        self.out.forward(&gated)?.astype(x.dtype())
    }

    fn append_cache(&mut self, x: &Array) -> Result<Array> {
        macro_rules! stage {
            ($name:expr, $value:expr) => {
                #[cfg(test)]
                self.stages.push(($name.into(), $value.try_clone()?));
            };
        }
        let qk = self
            .index
            .forward(x)?
            .astype(Dtype::Float16)?
            .reshape(&[1, 1, 5, 128])?;
        let raw = qk.slice(2, 4, 5)?.reshape(&[1, 1, 128])?;
        let tail = Self::append(&self.index_tail, &raw, 1)?;
        if (self.offset + 1) % 4 == 0 {
            let pool = tail
                .astype(Dtype::Float32)?
                .sum(1, true)?
                .scalar_mul(0.25)?
                .astype(Dtype::Float16)?;
            let block = self.rope(
                &self.ik_norm.forward(&pool)?.reshape(&[1, 1, 1, 128])?,
                32,
                self.offset - 3,
            )?;
            self.blocks = Some(Self::append(&self.blocks, &block, 2)?);
            self.index_tail = None;
        } else {
            self.index_tail = Some(tail);
        }
        let raw_key = self
            .k
            .forward(x)?
            .astype(Dtype::Float16)?
            .reshape(&[1, 1, 2, 256])?;
        stage!("k_proj", raw_key);
        let key_norm = self.k_norm.forward(&raw_key)?;
        stage!("k_norm", key_norm);
        let key = self.rope(&key_norm.transpose(&[0, 2, 1, 3])?, 64, self.offset)?;
        stage!("k_rope", key);
        let value = self
            .v
            .forward(x)?
            .astype(Dtype::Float16)?
            .reshape(&[1, 1, 2, 256])?;
        stage!("v_proj", value);
        let value = value.transpose(&[0, 2, 1, 3])?;
        self.keys = Some(Self::append(&self.keys, &key, 2)?);
        self.values = Some(Self::append(&self.values, &value, 2)?);
        self.offset = self.offset.checked_add(1).context("QSA offset exhausted")?;
        Ok(qk)
    }
    fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.index_tail = None;
        self.blocks = None;
        self.offset = 0;
    }
}

struct Ngrams {
    shards: Vec<TensorInfo>,
    sizes: [u64; 16],
    offsets: [u64; 16],
    multipliers: [u64; 3],
    bias: Vec<f16>,
    cache: BTreeMap<u64, Vec<u16>>,
    queue: VecDeque<u64>,
    previous: [u32; 2],
    read_bytes: u64,
}
impl Ngrams {
    fn load(c: &Checkpoint, p: &str) -> Result<Self> {
        let integers = |name: &str, n: usize| -> Result<Vec<u64>> {
            let tensor = c
                .tensors
                .get(&format!("{p}.{name}"))
                .context("missing PLE hash metadata")?;
            ensure!(
                tensor.dtype == "I64" && tensor.shape == [n],
                "invalid PLE hash metadata"
            );
            Ok(tensor
                .read_bytes()?
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b))
                .collect())
        };
        let sizes: [u64; 16] = integers("head_vocab_sizes", 16)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("wrong PLE heads"))?;
        let offsets: [u64; 16] = integers("head_offsets", 16)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("wrong PLE offsets"))?;
        let multipliers: [u64; 3] = integers("layer_multipliers", 3)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("wrong PLE multipliers"))?;
        let mut end = 0;
        for head in 0..16 {
            ensure!(
                offsets[head] == end && sizes[head] > 0 && sizes[head] <= i64::MAX as u64,
                "invalid PLE head range"
            );
            end = end
                .checked_add(sizes[head])
                .context("PLE row count overflow")?;
        }
        ensure!(
            end <= 320001536 && end > 320001408,
            "unsupported PLE row count"
        );
        let bias = checkpoint_array(c, &format!("{p}.head_bias"))?;
        ensure!(
            bias.shape() == [16, 160] && bias.dtype() == Dtype::Float16,
            "invalid PLE head bias"
        );
        let bias = bias
            .to_f16_bits()?
            .into_iter()
            .map(f16::from_bits)
            .collect();
        let mut shards = Vec::with_capacity(128);
        for index in 0..128 {
            let tensor = c
                .tensors
                .get(&format!("{p}.shard_{index}.trellis"))
                .context("missing PLE trellis shard")?;
            ensure!(
                tensor.dtype == "I16" && tensor.shape == [2500012, 41],
                "unsupported PLE row format"
            );
            shards.push(tensor.clone());
        }
        Ok(Self {
            shards,
            sizes,
            offsets,
            multipliers,
            bias,
            cache: BTreeMap::new(),
            queue: VecDeque::new(),
            previous: [EOS; 2],
            read_bytes: 0,
        })
    }
    fn forward(&mut self, token: u32) -> Result<Array> {
        let history = flash_support::segment_tokens(token, self.previous, EOS);
        let mut output = Vec::with_capacity(2560);
        for head in 0..16 {
            let row = flash_support::ngram_row(
                history,
                self.multipliers,
                2 + head / 8,
                self.offsets[head],
                self.sizes[head],
            )
            .context("invalid PLE row")?;
            if !self.cache.contains_key(&row) {
                let shard = &self.shards[(row / 2500012) as usize];
                let (file, base) = shard.source()?;
                let offset = flash_support::row_offset(base, row % 2500012, 2500012, 82)
                    .context("PLE offset overflow")?;
                let mut raw = [0u8; 82];
                file.read_exact_at(&mut raw, offset)?;
                let words = raw
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| u16::from_le_bytes(*b))
                    .collect();
                if self.cache.len() == 16384
                    && let Some(old) = self.queue.pop_front()
                {
                    self.cache.remove(&old);
                }
                self.cache.insert(row, words);
                self.queue.push_back(row);
                self.read_bytes += 82;
            }
            let words = &self.cache[&row];
            let scale = f16::from_bits(words[0]).to_f32();
            ensure!(scale.is_finite(), "nonfinite PLE row scale");
            for i in 0..160 {
                let mut state = 0u16;
                for j in 0..4 {
                    let symbol = (i + 160 - j) % 160;
                    state |= ((words[1 + symbol / 4] >> (4 * (symbol % 4))) & 15) << (j * 4);
                }
                let value = f16::from_f32(
                    codec::decode_codeword(state, Codebook::Mul1).to_f32() * scale
                        + self.bias[head * 160 + i].to_f32(),
                );
                ensure!(value.is_finite(), "nonfinite PLE embedding");
                output.push(value.to_bits());
            }
        }
        self.previous = [token, self.previous[0]];
        Array::from_f16_bits(&output, &[1, H])
    }
}

struct Ple {
    #[cfg(test)]
    stages: Vec<(String, Array)>,
    rows: Ngrams,
    key: Projection,
    value: Projection,
    nk: Norm,
    nq: Norm,
    nc: Norm,
    conv: Array,
    history: Option<Array>,
}
impl Ple {
    fn load(c: &Checkpoint, p: &str, eps: f32) -> Result<Self> {
        Ok(Self {
            #[cfg(test)]
            stages: Vec::new(),
            rows: Ngrams::load(c, &format!("{p}.ple_embedding.ngram_embedding"))?,
            key: Projection::load(c, &format!("{p}.key_proj"), H, WIDTH, false)?,
            value: Projection::load(c, &format!("{p}.value_proj"), H, H, false)?,
            nk: Norm::load(c, &format!("{p}.norm_key.weight"), H, HC, eps, true)?,
            nq: Norm::load(c, &format!("{p}.norm_query.weight"), H, HC, eps, true)?,
            nc: Norm::load(c, &format!("{p}.norm_conv.weight"), H, HC, eps, true)?,
            conv: half_weight(c, &format!("{p}.conv1d.weight"), Some(&[WIDTH, 1, 4]))?
                .transpose(&[1, 2, 0])?,
            history: None,
        })
    }
    fn forward(&mut self, streams: &Array, token: u32) -> Result<Array> {
        #[cfg(test)]
        self.stages.clear();
        macro_rules! stage {
            ($name:expr, $value:expr) => {
                #[cfg(test)]
                self.stages.push(($name.into(), $value.try_clone()?));
            };
        }
        let embedding = self.rows.forward(token)?;
        stage!("embedding", embedding);
        let key = self
            .nk
            .forward(&self.key.forward(&embedding)?)?
            .reshape(&[1, HC, H])?;
        let query = self.nq.forward(streams)?.reshape(&[1, HC, H])?;
        stage!("key", key);
        stage!("query", query);
        let raw = key
            .mul(&query)?
            .sum(2, true)?
            .div(&Array::from_f32(&[(H as f32).sqrt()], &[])?)?;
        let gate = raw.qwen4_ple_gate()?;
        stage!("raw", raw);
        stage!("gate", gate);
        let gated = self
            .value
            .forward(&embedding)?
            .astype(Dtype::Float32)?
            .reshape(&[1, 1, H])?
            .mul(&gate)?
            .reshape(&[1, WIDTH])?;
        stage!("gated", gated);
        let normed = self.nc.forward(&gated)?.reshape(&[1, 1, WIDTH])?;
        stage!("normed", normed);
        let old = match &self.history {
            Some(h) => h.try_clone()?,
            None => Array::zeros(&[1, 9, WIDTH])?,
        };
        let input = Array::concatenate(&[&old, &normed], 1)?;
        self.history = Some(input.slice(1, 1, 10)?);
        let indices = Array::from_u32(&[0, 3, 6, 9], &[4])?;
        let conv_raw = input
            .take(&indices, 1)?
            .conv1d(&self.conv.transpose(&[2, 1, 0])?, WIDTH)?
            .reshape(&[1, WIDTH])?;
        stage!("conv_raw", conv_raw);
        let convolution = conv_raw.compiled_silu()?;
        stage!("convolution", convolution);
        streams.add(&gated.add(&convolution)?)
    }
    fn reset(&mut self) {
        self.rows.previous = [EOS; 2];
        self.history = None;
    }
}

enum Attention {
    Gdn(Box<Gdn>),
    Qsa(Box<Qsa>),
}

enum AttentionState {
    Gdn {
        history: Option<Array>,
        state: Option<Array>,
    },
    Qsa(QsaState),
}
struct LayerState {
    attention: AttentionState,
    ple: Option<(Option<Array>, [u32; 2])>,
}
pub(crate) struct Layer {
    #[cfg(test)]
    stages: Vec<(String, Array)>,
    attn_hc: Hc,
    mlp_hc: Hc,
    attention: Attention,
    moe: SsdMoe,
    ple: Option<Ple>,
}
impl Layer {
    #[allow(clippy::too_many_arguments)]
    fn load(
        c: &Checkpoint,
        p: &str,
        linear: bool,
        ple: bool,
        eps: f32,
        rope: f32,
        slots: usize,
        overlap: bool,
    ) -> Result<Self> {
        Ok(Self {
            #[cfg(test)]
            stages: Vec::new(),
            attn_hc: Hc::load(c, &format!("{p}.attn_hyper_connection"), eps, true)?,
            mlp_hc: Hc::load(c, &format!("{p}.mlp_hyper_connection"), eps, true)?,
            attention: if linear {
                Attention::Gdn(Box::new(Gdn::load(c, &format!("{p}.linear_attn"), eps)?))
            } else {
                Attention::Qsa(Box::new(Qsa::load(
                    c,
                    &format!("{p}.self_attn"),
                    eps,
                    rope,
                )?))
            },
            moe: SsdMoe::load(c, &format!("{p}.mlp"), slots, overlap)?,
            ple: ple
                .then(|| Ple::load(c, &format!("{p}.ple"), eps))
                .transpose()?,
        })
    }
    fn forward(&mut self, streams: &Array, token: u32) -> Result<Array> {
        self.forward_lookahead(streams, token, None, 0)
    }
    fn forward_lookahead(
        &mut self,
        streams: &Array,
        token: u32,
        next: Option<&mut SsdMoe>,
        depth: usize,
    ) -> Result<Array> {
        let (streams, mixed, gate) = self.prepare(streams, token)?;
        let output = self.moe.forward_lookahead(&mixed, next, depth)?;
        #[cfg(test)]
        self.stages.push(("moe".into(), output.try_clone()?));
        Hc::write(&streams, &output, &gate)
    }
    fn prepare(&mut self, streams: &Array, token: u32) -> Result<(Array, Array, Array)> {
        #[cfg(test)]
        self.stages.clear();
        let streams = match &mut self.ple {
            Some(ple) => ple.forward(streams, token)?,
            None => streams.try_clone()?,
        };
        let (mixed, gate) = self.attn_hc.read(&streams)?;
        #[cfg(test)]
        {
            self.stages.push(("attn_mixed".into(), mixed.try_clone()?));
            self.stages.push((
                "attn_gate".into(),
                gate.as_ref().context("missing gate")?.try_clone()?,
            ));
        }
        let branch = match &mut self.attention {
            Attention::Gdn(gdn) => gdn.forward(&mixed)?,
            Attention::Qsa(qsa) => qsa.forward(&mixed)?,
        };
        let streams = Hc::write(
            &streams,
            &branch,
            &gate.context("missing attention injection")?,
        )?;
        #[cfg(test)]
        self.stages.push(("attention".into(), branch.try_clone()?));
        let (mixed, gate) = self.mlp_hc.read(&streams)?;
        #[cfg(test)]
        {
            self.stages.push(("mlp_mixed".into(), mixed.try_clone()?));
            self.stages.push((
                "mlp_gate".into(),
                gate.as_ref().context("missing gate")?.try_clone()?,
            ));
        }
        Ok((streams, mixed, gate.context("missing MoE injection")?))
    }
    fn snapshot(&self) -> Result<LayerState> {
        Ok(LayerState {
            attention: match &self.attention {
                Attention::Gdn(g) => AttentionState::Gdn {
                    history: clone_array(&g.history)?,
                    state: clone_array(&g.state)?,
                },
                Attention::Qsa(q) => AttentionState::Qsa(q.snapshot()?),
            },
            ple: self
                .ple
                .as_ref()
                .map(|p| -> Result<_> { Ok((clone_array(&p.history)?, p.rows.previous)) })
                .transpose()?,
        })
    }
    fn restore(&mut self, state: LayerState) -> Result<()> {
        match (&mut self.attention, state.attention) {
            (Attention::Gdn(g), AttentionState::Gdn { history, state }) => {
                g.history = history;
                g.state = state;
            }
            (Attention::Qsa(q), AttentionState::Qsa(state)) => q.restore(state),
            _ => anyhow::bail!("invalid Flash attention snapshot"),
        }
        if let Some((history, previous)) = state.ple {
            let ple = self.ple.as_mut().context("invalid Flash PLE snapshot")?;
            ple.history = history;
            ple.rows.previous = previous;
        }
        Ok(())
    }
    fn forward_pair(
        &mut self,
        streams: &[Array; 2],
        tokens: [u32; 2],
        next: Option<&mut SsdMoe>,
        depth: usize,
    ) -> Result<([Array; 2], LayerState)> {
        let a = self.prepare(&streams[0], tokens[0])?;
        let first = self.snapshot()?;
        let b = self.prepare(&streams[1], tokens[1])?;
        let output =
            self.moe
                .forward_lookahead(&Array::concatenate(&[&a.1, &b.1], 0)?, next, depth)?;
        Ok((
            [
                Hc::write(&a.0, &output.slice(0, 0, 1)?, &a.2)?,
                Hc::write(&b.0, &output.slice(0, 1, 2)?, &b.2)?,
            ],
            first,
        ))
    }
    fn reset(&mut self) {
        match &mut self.attention {
            Attention::Gdn(gdn) => gdn.reset(),
            Attention::Qsa(qsa) => qsa.reset(),
        };
        if let Some(ple) = &mut self.ple {
            ple.reset();
        }
    }
}

pub struct FlashNext {
    path: PathBuf,
    embed: TensorInfo,
    layers: Vec<Layer>,
    mixer: Hc,
    head: Projection,
    offset: i32,
    context: i32,
    pub last_hidden: Option<Array>,
    verification: Option<PairState>,
    lookahead_depth: usize,
    lookahead_pool: Option<Rc<RefCell<ReaderPool>>>,
}
struct PairState {
    base: i32,
    layers: Vec<LayerState>,
    hidden: Array,
}
impl FlashNext {
    pub fn from_checkpoint(c: &Checkpoint) -> Result<Self> {
        ensure!(c.model_type == "qwen4_exp", "expected qwen4_exp");
        let lookahead_depth = match std::env::var("MLXL3_SSD_LOOKAHEAD_DEPTH") {
            Ok(value) => value
                .parse::<usize>()
                .context("invalid SSD lookahead depth")?,
            Err(std::env::VarError::NotPresent) => 0,
            Err(error) => return Err(error.into()),
        };
        ensure!(
            lookahead_depth <= 10,
            "SSD lookahead depth must be in 0..=10"
        );
        let root: Value = serde_json::from_reader(File::open(c.path.join("config.json"))?)?;
        let t = &root["text_config"];
        for (key, value) in [
            ("hidden_size", 2560),
            ("hc_count", 4),
            ("hc_lowrank", 320),
            ("num_hidden_layers", 48),
            ("num_experts", 512),
            ("num_experts_per_tok", 10),
            ("moe_intermediate_size", 640),
            ("head_dim", 256),
            ("num_attention_heads", 24),
            ("num_key_value_heads", 2),
            ("linear_key_head_dim", 128),
            ("linear_value_head_dim", 128),
            ("linear_num_key_heads", 16),
            ("linear_num_value_heads", 48),
            ("linear_conv_kernel_dim", 4),
            ("indexer_n_heads", 4),
            ("indexer_kv_heads", 1),
            ("indexer_head_dim", 128),
            ("indexer_compress_ratio", 4),
            ("indexer_budget", 2048),
            ("eos_token_id", 248044),
            ("vocab_size", 248320),
            ("ple_embed_dim", 2560),
            ("ngram_size", 3),
            ("heads_per_ngram", 8),
            ("ple_conv_kernel_size", 4),
        ] {
            ensure!(
                t[key].as_i64() == Some(value),
                "unsupported Flash Next {key}"
            );
        }
        ensure!(
            t["output_gate_type"] == "sigmoid" && t["ple_layer_ids"] == serde_json::json!([2]),
            "unsupported Flash PLE/gate configuration"
        );
        let eps = t["rms_norm_eps"]
            .as_f64()
            .context("missing Flash epsilon")? as f32;
        let rope = t["rope_parameters"]["rope_theta"]
            .as_f64()
            .context("missing Flash RoPE")? as f32;
        let context = i32::try_from(
            t["max_position_embeddings"]
                .as_u64()
                .context("missing context limit")?,
        )?;
        ensure!(
            eps.is_finite()
                && eps > 0.
                && rope.is_finite()
                && rope > 0.
                && context > 0
                && t["rope_parameters"]["partial_rotary_factor"] == 0.25,
            "invalid Flash numerical configuration"
        );
        let budget = match std::env::var("MLXL3_SSD_CACHE_GIB") {
            Ok(value) => Some(
                value
                    .parse::<f64>()
                    .context("invalid MLXL3_SSD_CACHE_GIB")?,
            ),
            Err(std::env::VarError::NotPresent) => None,
            Err(error) => return Err(error.into()),
        };
        let slots = if let Some(budget) = budget {
            ensure!(
                budget.is_finite() && (0.6..=12.).contains(&budget),
                "SSD expert cache budget must be 0.6..12 GiB"
            );
            ((budget * (1u64 << 30) as f64 / (48. * flash_support::EXPERT_BYTES as f64)).floor()
                as usize)
                .clamp(10, 512)
        } else {
            #[cfg(target_os = "macos")]
            let available = flash_support::available_memory_bytes()?;
            #[cfg(not(target_os = "macos"))]
            let available = 14 << 30;
            let slots = flash_support::automatic_cache_slots(available).context(
                "not enough system memory for Flash's minimum SSD cache and memory reserve",
            )?;
            eprintln!(
                "Flash Next: auto cache {:.2} GiB ({slots} slots/layer), estimated available {:.2} GiB",
                slots as f64 * 48. * flash_support::EXPERT_BYTES as f64 / (1u64 << 30) as f64,
                available as f64 / (1u64 << 30) as f64
            );
            slots
        };
        crate::array::set_default_cache_limit(512 << 20)?;
        let overlap = match std::env::var("MLXL3_SSD_OVERLAP") {
            Ok(value) => match value.as_str() {
                "0" => false,
                "1" => true,
                _ => anyhow::bail!("MLXL3_SSD_OVERLAP must be 0 or 1"),
            },
            Err(std::env::VarError::NotPresent) => true,
            Err(error) => return Err(error.into()),
        };
        let embed = c
            .tensors
            .get("model.language_model.embed_tokens.weight")
            .context("missing token embedding")?
            .clone();
        ensure!(
            embed.dtype == "BF16" && embed.shape == [248320, 2560],
            "unsupported token embedding"
        );
        let types = t["layer_types"].as_array().context("missing layer types")?;
        ensure!(types.len() == 48, "invalid layer count");
        let mut layers = Vec::with_capacity(48);
        for (i, kind) in types.iter().enumerate() {
            let linear = i % 4 != 3;
            ensure!(
                *kind
                    == if linear {
                        "linear_attention"
                    } else {
                        "full_attention"
                    },
                "invalid Flash layer type"
            );
            eprintln!(
                "Flash Next: loading layer {}/48, {} exact expert slots",
                i + 1,
                slots
            );
            layers.push(Layer::load(
                c,
                &format!("model.language_model.layers.{i}"),
                linear,
                i == 1,
                eps,
                rope,
                slots,
                overlap,
            )?);
        }
        let lookahead_pool = if lookahead_depth > 0 {
            let pool = Rc::new(RefCell::new(ReaderPool::new()?));
            for (index, layer) in layers.iter_mut().enumerate() {
                layer.moe.prefetcher = Some((index, Rc::clone(&pool)));
            }
            Some(pool)
        } else {
            None
        };
        let mixer = Hc::load(c, "model.language_model.hyper_connection_mixer", eps, false)?;
        let head = Projection::load(c, "lm_head", H, 248320, false)?;
        Ok(Self {
            path: c.path.clone(),
            embed,
            layers,
            mixer,
            head,
            offset: 0,
            context,
            last_hidden: None,
            verification: None,
            lookahead_depth,
            lookahead_pool,
        })
    }
    pub fn embedding(&self, token: u32) -> Result<Array> {
        ensure!(token < 248320, "token outside Flash vocabulary");
        let (file, base) = self.embed.source()?;
        let offset = flash_support::row_offset(base, u64::from(token), 248320, 5120)
            .context("token embedding offset overflow")?;
        Array::from_file(file, offset, &[1, H], Dtype::BFloat16)?.astype(Dtype::Float16)
    }
    pub fn forward(&mut self, token: u32) -> Result<Array> {
        Ok(self.forward_inner(token, false)?.0)
    }
    pub fn forward_trace(&mut self, token: u32) -> Result<(Array, Vec<(String, Array)>)> {
        self.forward_inner(token, true)
    }
    fn forward_inner(
        &mut self,
        token: u32,
        capture: bool,
    ) -> Result<(Array, Vec<(String, Array)>)> {
        ensure!(
            self.verification.is_none(),
            "uncommitted Flash verification"
        );
        ensure!(self.offset < self.context, "Flash context exhausted");
        let mut trace = Vec::new();
        let mut streams = self.streams(token)?;
        for i in 0..self.layers.len() {
            let (current, rest) = self.layers.split_at_mut(i + 1);
            let layer = &mut current[i];
            let next = (self.lookahead_depth > 0)
                .then(|| rest.first_mut().map(|l| &mut l.moe))
                .flatten();
            streams = layer.forward_lookahead(&streams, token, next, self.lookahead_depth)?;
            if capture {
                trace.push((format!("layer_{i}"), streams.try_clone()?));
                if let Attention::Gdn(gdn) = &layer.attention
                    && let Some(state) = &gdn.state
                {
                    trace.push((format!("gdn_state_{i}"), state.try_clone()?));
                }
            }
        }
        let logits = self.logits(&streams)?;
        self.drain()?;
        self.last_hidden = Some(streams);
        self.offset += 1;
        Ok((logits, trace))
    }
    fn streams(&self, token: u32) -> Result<Array> {
        self.embedding(token)?
            .astype(Dtype::Float32)?
            .reshape(&[1, 1, H])?
            .broadcast_to(&[1, HC, H])?
            .reshape(&[1, WIDTH])
    }
    fn logits(&self, streams: &Array) -> Result<Array> {
        let (mixed, _) = self.mixer.read(streams)?;
        let logits = self.head.forward(&mixed.astype(Dtype::Float16)?)?;
        logits.eval()?;
        Ok(logits)
    }
    fn drain(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.moe.drain()?;
            match &layer.attention {
                Attention::Gdn(g) => {
                    if let Some(state) = &g.state {
                        state.eval()?;
                    }
                }
                Attention::Qsa(q) => {
                    if let Some(blocks) = &q.blocks {
                        blocks.eval()?;
                    }
                }
            }
        }
        Ok(())
    }
    pub fn verify_pair(&mut self, tokens: [u32; 2]) -> Result<(Array, Array)> {
        ensure!(
            self.verification.is_none(),
            "uncommitted Flash verification"
        );
        ensure!(
            self.context - self.offset >= 2 && tokens.iter().all(|&t| t < 248320),
            "invalid Flash verification tokens or context"
        );
        let base = self.offset;
        let mut streams = [self.streams(tokens[0])?, self.streams(tokens[1])?];
        let mut states = Vec::with_capacity(self.layers.len());
        for i in 0..self.layers.len() {
            let (current, rest) = self.layers.split_at_mut(i + 1);
            let next = (self.lookahead_depth > 0)
                .then(|| rest.first_mut().map(|l| &mut l.moe))
                .flatten();
            let (output, first) =
                current[i].forward_pair(&streams, tokens, next, self.lookahead_depth)?;
            streams = output;
            states.push(first);
        }
        let logits =
            Array::concatenate(&[&self.logits(&streams[0])?, &self.logits(&streams[1])?], 0)?;
        let hidden = Array::concatenate(&[&streams[0], &streams[1]], 0)?;
        self.drain()?;
        self.verification = Some(PairState {
            base,
            layers: states,
            hidden: streams[0].try_clone()?,
        });
        self.last_hidden = Some(streams[1].try_clone()?);
        self.offset += 2;
        Ok((logits, hidden))
    }
    pub fn commit_pair(&mut self, retained: i32) -> Result<()> {
        ensure!(
            crate::contracts::gdn_tape_prefix(2, retained).is_some(),
            "invalid Flash verification commit"
        );
        let state = self
            .verification
            .as_ref()
            .context("no Flash verification to commit")?;
        let committed = flash_support::pair_commit_offset(state.base, self.offset, retained)
            .context("stale Flash verification")?;
        ensure!(
            state.layers.len() == self.layers.len(),
            "invalid Flash layer snapshot"
        );
        let state = self.verification.take().expect("validated verification");
        if retained == 1 {
            for (layer, saved) in self.layers.iter_mut().zip(state.layers) {
                layer.restore(saved)?;
            }
            self.last_hidden = Some(state.hidden);
        }
        self.offset = committed;
        Ok(())
    }
    pub fn forward_tokens(&mut self, tokens: &[u32]) -> Result<Array> {
        ensure!(
            !tokens.is_empty()
                && tokens.len() <= usize::try_from(self.context - self.offset)?
                && tokens.iter().all(|&t| t < 248320),
            "invalid Flash token batch"
        );
        let mut last = None;
        for &token in tokens {
            last = Some(self.forward(token)?);
        }
        last.context("empty Flash token batch")
    }
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
        self.offset = 0;
        self.last_hidden = None;
        self.verification = None;
    }
    pub fn context_limit(&self) -> i32 {
        self.context
    }
    pub fn offset(&self) -> i32 {
        self.offset
    }
    pub fn io_stats(&self) -> (u64, u64, u64) {
        let bytes = self
            .lookahead_pool
            .as_ref()
            .map_or(0, |p| p.borrow().counters.bytes.load(Ordering::Relaxed));
        self.layers.iter().fold((0, 0, bytes), |(h, m, b), l| {
            (
                h + l.moe.hits,
                m + l.moe.misses,
                b.saturating_add(l.moe.read_bytes),
            )
        })
    }
}

pub struct FlashMtpHead {
    #[cfg(test)]
    stages: Vec<(String, Array)>,
    embedding_norm: Norm,
    hidden_norm: Norm,
    embedding_fc: Projection,
    hidden_fc: Projection,
    layer: Layer,
    mixer: Hc,
}
impl FlashMtpHead {
    pub fn load(c: &Checkpoint, target: &FlashNext) -> Result<Self> {
        ensure!(
            c.path == target.path && c.model_type == "qwen4_exp",
            "Flash MTP requires the target checkpoint"
        );
        let config: Value = serde_json::from_reader(File::open(c.path.join("config.json"))?)?;
        ensure!(
            config["text_config"]["mtp_num_hidden_layers"] == 1,
            "Flash MTP requires one native draft layer"
        );
        let eps = target.mixer.norm.eps;
        let rope = config["text_config"]["rope_parameters"]["rope_theta"]
            .as_f64()
            .context("missing MTP RoPE")? as f32;
        let mut layer = Layer::load(c, "mtp.layers.0", false, false, eps, rope, 512, true)?;
        layer.moe.warm_all()?;
        Ok(Self {
            #[cfg(test)]
            stages: Vec::new(),
            embedding_norm: Norm::load(c, "mtp.pre_fc_norm_embedding.weight", H, 1, eps, true)?,
            hidden_norm: Norm::load(c, "mtp.pre_fc_norm_hidden.weight", WIDTH, 1, eps, true)?,
            embedding_fc: Projection::load(c, "mtp.fc_embedding", H, H, false)?,
            hidden_fc: Projection::load(c, "mtp.fc_hidden", H, H, false)?,
            layer,
            mixer: Hc::load(c, "mtp.hyper_connection_mixer", eps, false)?,
        })
    }
    fn mixed(&self, target: &FlashNext, hidden: &Array, token: u32) -> Result<Array> {
        ensure!(
            hidden.shape() == [1, WIDTH],
            "invalid Flash MTP hidden shape"
        );
        let embedding = target.embedding(token)?.reshape(&[1, 1, H])?;
        let embedding = self
            .embedding_fc
            .forward(&self.embedding_norm.forward(&embedding)?)?;
        let streams = self
            .hidden_norm
            .forward(&hidden.reshape(&[1, 1, WIDTH])?)?
            .reshape(&[1, 1, HC, H])?;
        let projected = self.hidden_fc.forward(&streams)?.astype(Dtype::Float16)?;
        embedding
            .reshape(&[1, 1, 1, H])?
            .add(&projected)?
            .reshape(&[1, WIDTH])
    }
    pub fn forward(&mut self, target: &FlashNext, hidden: &Array, token: u32) -> Result<Array> {
        let streams = self.mixed(target, hidden, token)?;
        let output = self.layer.forward(&streams, token)?;
        let (mixed, _) = self.mixer.read(&output)?;
        #[cfg(test)]
        {
            self.stages = vec![("mixed".into(), streams.try_clone()?)];
            self.stages.extend(
                self.layer
                    .stages
                    .iter()
                    .map(|(name, value)| Ok((name.clone(), value.try_clone()?)))
                    .collect::<Result<Vec<_>>>()?,
            );
            self.stages.push(("layer".into(), output.try_clone()?));
            self.stages.push(("output".into(), mixed.try_clone()?));
        }
        let logits = target.head.forward(&mixed.astype(Dtype::Float16)?)?;
        logits.eval()?;
        self.layer.moe.drain()?;
        self.eval_cache()?;
        Ok(logits)
    }
    pub fn extend_cache(&mut self, target: &FlashNext, hidden: &Array, token: u32) -> Result<()> {
        let streams = self.mixed(target, hidden, token)?;
        let (input, _) = self.layer.attn_hc.read(&streams)?;
        let Attention::Qsa(qsa) = &mut self.layer.attention else {
            anyhow::bail!("invalid MTP attention");
        };
        qsa.append_cache(&input)?;
        self.eval_cache()
    }
    fn eval_cache(&self) -> Result<()> {
        let Attention::Qsa(qsa) = &self.layer.attention else {
            anyhow::bail!("invalid MTP attention");
        };
        for array in [&qsa.keys, &qsa.values, &qsa.index_tail, &qsa.blocks]
            .into_iter()
            .flatten()
        {
            array.eval()?;
        }
        Ok(())
    }
    pub fn reset(&mut self) {
        self.layer.reset();
    }
    pub fn offset(&self) -> i32 {
        match &self.layer.attention {
            Attention::Qsa(q) => q.offset,
            Attention::Gdn(_) => 0,
        }
    }
}

#[derive(Default)]
pub struct FlashMtpSession {
    pending: VecDeque<u32>,
    pub proposed: usize,
    pub accepted: usize,
    pub blocks: usize,
    #[cfg(test)]
    profile_seconds: [f64; 4],
}
impl FlashMtpSession {
    pub fn advance(
        &mut self,
        target: &mut FlashNext,
        head: &mut FlashMtpHead,
        anchor: u32,
        context_limit: usize,
        output_remaining: usize,
    ) -> Result<u32> {
        ensure!(
            output_remaining > 0 && anchor < 248320,
            "invalid Flash MTP budget or token"
        );
        if let Some(token) = self.pending.pop_front() {
            return Ok(token);
        }
        let remaining = context_limit
            .checked_sub(usize::try_from(target.offset())?)
            .context("Flash MTP context exhausted")?;
        let width = crate::speculative::mtp_width(1, remaining, output_remaining)
            .context("Flash MTP context or output exhausted")?;
        #[cfg(test)]
        let step_start = std::time::Instant::now();
        let result = self.step(target, head, anchor, width);
        if result.is_err() {
            self.pending.clear();
            target.reset();
            head.reset();
        }
        #[cfg(test)]
        {
            self.profile_seconds[3] += step_start.elapsed().as_secs_f64();
        }
        result
    }
    fn step(
        &mut self,
        target: &mut FlashNext,
        head: &mut FlashMtpHead,
        anchor: u32,
        width: usize,
    ) -> Result<u32> {
        ensure!(
            target.offset() > 0 && head.offset() == target.offset() - 1,
            "Flash MTP cache is not aligned with target"
        );
        let hidden = target
            .last_hidden
            .as_ref()
            .context("Flash MTP needs a prefilled target")?
            .try_clone()?;
        if width == 0 {
            head.extend_cache(target, &hidden, anchor)?;
            return target
                .forward(anchor)?
                .chat_greedy_ids()?
                .first()
                .copied()
                .context("missing Flash fallback token");
        }
        #[cfg(test)]
        let head_start = std::time::Instant::now();
        let proposal = head
            .forward(target, &hidden, anchor)?
            .chat_greedy_ids()?
            .first()
            .copied()
            .context("missing Flash MTP proposal")?;
        #[cfg(test)]
        {
            self.profile_seconds[0] += head_start.elapsed().as_secs_f64();
        }
        #[cfg(test)]
        let verify_start = std::time::Instant::now();
        let (logits, hidden) = target.verify_pair([anchor, proposal])?;
        let tokens = logits.chat_greedy_ids()?;
        #[cfg(test)]
        {
            self.profile_seconds[1] += verify_start.elapsed().as_secs_f64();
        }
        #[cfg(test)]
        let repair_start = std::time::Instant::now();
        let acceptance = crate::speculative::greedy_accept(&[proposal], &tokens)
            .context("invalid Flash MTP verification")?;
        let accepted = acceptance.accepted_draft_tokens;
        target.commit_pair(1 + accepted as i32)?;
        if accepted > 0 {
            head.extend_cache(target, &hidden.slice(0, 0, 1)?, proposal)?;
        }
        #[cfg(test)]
        {
            self.profile_seconds[2] += repair_start.elapsed().as_secs_f64();
        }
        if accepted > 0 {
            self.pending.push_back(proposal);
        }
        self.pending.push_back(acceptance.target_token);
        self.proposed += 1;
        self.accepted += accepted;
        self.blocks += 1;
        self.pending
            .pop_front()
            .context("missing Flash MTP target token")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
    fn state_bytes(model: &FlashNext) -> Result<Vec<Vec<u8>>> {
        fn tensor(value: &Option<Array>) -> Result<Vec<u8>> {
            let Some(value) = value else {
                return Ok(vec![0]);
            };
            let data = value.to_bytes()?;
            ensure!(!data.is_empty(), "empty semantic state");
            ensure!(
                value
                    .astype(Dtype::Float32)?
                    .to_f32()?
                    .iter()
                    .all(|v| v.is_finite()),
                "nonfinite semantic state"
            );
            let mut bytes = vec![1];
            for dim in value.shape() {
                bytes.extend(dim.to_le_bytes());
            }
            bytes.extend(data);
            Ok(bytes)
        }
        let mut result = vec![
            model.offset.to_le_bytes().to_vec(),
            tensor(&model.last_hidden)?,
        ];
        for layer in &model.layers {
            match &layer.attention {
                Attention::Gdn(g) => {
                    result.push(tensor(&g.history)?);
                    result.push(tensor(&g.state)?);
                }
                Attention::Qsa(q) => result.extend(qsa_bytes(q)?),
            }
            if let Some(p) = &layer.ple {
                result.push(tensor(&p.history)?);
                result.push(
                    p.rows
                        .previous
                        .into_iter()
                        .flat_map(u32::to_le_bytes)
                        .collect(),
                );
            }
        }
        Ok(result)
    }
    #[test]
    #[ignore = "physical Apple GPU and Flash checkpoint; LES full logits, states and commits"]
    fn flash_lookahead_preserves_full_logits_states_and_pair_commits() -> Result<()> {
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut model = FlashNext::from_checkpoint(&c)?;
        let tokens = [EOS, 78768, 11, 7, 5, 19];
        let pool = Rc::new(RefCell::new(ReaderPool::new()?));
        for (index, layer) in model.layers.iter_mut().enumerate() {
            layer.moe.prefetcher = Some((index, Rc::clone(&pool)));
        }
        model.lookahead_pool = Some(Rc::clone(&pool));
        model.lookahead_depth = 0;
        let mut logits = Vec::new();
        let mut states = Vec::new();
        for token in tokens {
            let value = model.forward(token)?;
            ensure!(value.shape() == [1, 248320]);
            let host = value.to_f32()?;
            ensure!(host.len() == 248320 && host.iter().all(|v| v.is_finite()));
            logits.push(value.to_bytes()?);
            states.push(state_bytes(&model)?);
        }
        for half in [false, true] {
            for depth in [0, 2, 4, 8, 10] {
                for retained in [1, 2] {
                    model.reset();
                    model.lookahead_depth = depth;
                    for layer in &mut model.layers {
                        layer.moe.cache_for_test_invalidate();
                        layer.moe.predictor_half = half;
                    }
                    let before = pool.borrow().counters.submitted.load(Ordering::Relaxed);
                    for step in 0..3 {
                        ensure!(model.forward(tokens[step])?.to_bytes()? == logits[step]);
                        ensure!(state_bytes(&model)? == states[step]);
                    }
                    let (pair, _) = model.verify_pair([tokens[3], tokens[4]])?;
                    let mut expected = logits[3].clone();
                    expected.extend(&logits[4]);
                    ensure!(pair.shape() == [2, 248320] && pair.to_bytes()? == expected);
                    ensure!(state_bytes(&model)? == states[4]);
                    model.commit_pair(retained)?;
                    ensure!(state_bytes(&model)? == states[2 + retained as usize]);
                    let next = 3 + retained as usize;
                    ensure!(model.forward(tokens[next])?.to_bytes()? == logits[next]);
                    ensure!(state_bytes(&model)? == states[next]);
                    let after = pool.borrow().counters.submitted.load(Ordering::Relaxed);
                    ensure!(if depth > 0 {
                        after > before
                    } else {
                        after == before
                    });
                    pool.borrow_mut().wait_for_test()?;
                    pool.borrow_mut().poll();
                    let pool = pool.borrow();
                    let counts = &pool.counters;
                    eprintln!(
                        "LES41_DIAGNOSTIC {}",
                        serde_json::json!({
                            "predictor_half":half,"depth":depth,"commit":retained,"submitted":counts.submitted.load(Ordering::Relaxed),
                            "completed":counts.completed.load(Ordering::Relaxed),"used":counts.used.load(Ordering::Relaxed),
                            "discarded":counts.discarded.load(Ordering::Relaxed),"late_needed":counts.late_needed.load(Ordering::Relaxed),
                            "failed":counts.failed.load(Ordering::Relaxed),"read_bytes":counts.bytes.load(Ordering::Relaxed)
                        })
                    );
                }
            }
        }
        Ok(())
    }

    #[cfg(kani)]
    #[kani::proof]
    fn actual_lookahead_model_path_checks_tokens_and_output_shape() {
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL").unwrap()))
            .unwrap();
        let mut model = FlashNext::from_checkpoint(&c).unwrap();
        model.lookahead_depth = if kani::any::<bool>() { 2 } else { 4 };
        let token = kani::any::<u32>();
        match model.forward(token) {
            Ok(logits) => {
                assert!(token < 248320);
                assert_eq!(logits.shape(), [1, 248320]);
                kani::cover!(token == 0);
            }
            Err(_) => {
                kani::cover!(token >= 248320);
            }
        }
    }
    fn qsa_bytes(qsa: &Qsa) -> Result<Vec<Vec<u8>>> {
        let mut bytes = vec![qsa.offset.to_le_bytes().to_vec()];
        for value in [&qsa.keys, &qsa.values, &qsa.index_tail, &qsa.blocks] {
            let mut item = vec![u8::from(value.is_some())];
            if let Some(value) = value {
                ensure!(
                    value
                        .astype(Dtype::Float32)?
                        .to_f32()?
                        .iter()
                        .all(|v| v.is_finite()),
                    "nonfinite QSA state"
                );
                for dim in value.shape() {
                    item.extend(dim.to_le_bytes());
                }
                item.extend(value.to_bytes()?);
            }
            bytes.push(item);
        }
        Ok(bytes)
    }
    #[cfg(feature = "chat")]
    #[test]
    #[ignore = "physical Apple GPU; isolated natural decode diagnostic, not release throughput"]
    fn flash_light_profile_preserves_natural_decode() -> Result<()> {
        use std::time::Instant;
        let mtp = match std::env::var("MLXL3_FLASH_PROFILE_MTP")?.as_str() {
            "0" => false,
            "1" => true,
            _ => anyhow::bail!("invalid profile mode"),
        };
        let label = std::env::var("MLXL3_FLASH_PROFILE_PROMPT")?;
        let prompt = match label.as_str() {
            "MAIN" => {
                "Explique en français, avec un exemple concret, comment un cache LFU évite les lectures SSD inutiles pendant l’inférence d’un modèle MoE."
            }
            "HOLD" => {
                "Écris une fonction Python merge_intervals qui fusionne les intervalles qui se chevauchent, puis explique sa complexité et donne un exemple."
            }
            _ => anyhow::bail!("invalid profile prompt"),
        };
        let path = std::env::var("MLXL3_FLASH_MODEL")?;
        let c = crate::checkpoint::inspect(Path::new(&path))?;
        let tokenizer = crate::tokenizer::ChatTokenizer::load(Path::new(&path))?;
        let tokens = tokenizer.encode(&tokenizer.render(&[crate::tokenizer::Message {
            role: "user".into(),
            content: prompt.into(),
        }])?)?;
        let mut target = FlashNext::from_checkpoint(&c)?;
        let mut head = if mtp {
            Some(FlashMtpHead::load(&c, &target)?)
        } else {
            None
        };
        ensure!(!tokens.is_empty() && tokens.len() + 64 <= target.context_limit() as usize);
        for phase in ["warmup", "measured"] {
            target.reset();
            if let Some(head) = &mut head {
                head.reset();
            }
            let mut logits = None;
            for &token in &tokens {
                if let (Some(head), Some(hidden)) = (&mut head, &target.last_hidden) {
                    head.extend_cache(&target, hidden, token)?;
                }
                logits = Some(target.forward(token)?);
            }
            let mut anchor = logits.context("empty prefill")?.chat_greedy_ids()?[0];
            if let Some(pool) = &target.lookahead_pool {
                let mut pool = pool.borrow_mut();
                pool.wait_for_test()?;
                pool.poll();
                let c = &pool.counters;
                for counter in [
                    &c.submitted,
                    &c.completed,
                    &c.used,
                    &c.discarded,
                    &c.failed,
                    &c.late_needed,
                    &c.bytes,
                ] {
                    counter.store(0, Ordering::Relaxed);
                }
            }
            for layer in &mut target.layers {
                let moe = &mut layer.moe;
                moe.profile_seconds = [0.0; 2];
                moe.prefetch_profile = [0; 4];
                moe.import_seconds = 0.0;
                moe.overlap_hot_rows = 0;
                moe.prefetched = 0;
                moe.hits = 0;
                moe.misses = 0;
                moe.read_bytes = 0;
            }
            let mut session = FlashMtpSession::default();
            let mut output = vec![anchor];
            let start = Instant::now();
            for step in 1..64 {
                anchor = if let Some(head) = &mut head {
                    session.advance(&mut target, head, anchor, tokens.len() + 63, 64 - step)?
                } else {
                    target.forward(anchor)?.chat_greedy_ids()?[0]
                };
                output.push(anchor);
            }
            target.drain()?;
            let decode_seconds = start.elapsed().as_secs_f64();
            ensure!(output.len() == 64 && output.iter().all(|&id| id < 248320));
            ensure!(session.pending.is_empty() && target.offset() as usize == tokens.len() + 63);
            if let Some(head) = &head {
                ensure!(head.offset() == target.offset() - 1);
            }
            let logits = target.logits(
                target
                    .last_hidden
                    .as_ref()
                    .context("missing final hidden")?,
            )?;
            ensure!(logits.shape() == [1, 248320]);
            let host = logits.to_f32()?;
            ensure!(host.len() == 248320 && host.iter().all(|v| v.is_finite()));
            let prefetch = if let Some(pool) = &target.lookahead_pool {
                let mut pool = pool.borrow_mut();
                pool.wait_for_test()?;
                let c = &pool.counters;
                Some(serde_json::json!({
                    "submitted":c.submitted.load(Ordering::Relaxed), "completed":c.completed.load(Ordering::Relaxed),
                    "used":c.used.load(Ordering::Relaxed), "discarded":c.discarded.load(Ordering::Relaxed),
                    "failed":c.failed.load(Ordering::Relaxed), "late_needed":c.late_needed.load(Ordering::Relaxed),
                    "read_bytes":c.bytes.load(Ordering::Relaxed)
                }))
            } else {
                None
            };
            let (hits, misses, bytes) = target.io_stats();
            let mut counts = [0u64; 6];
            let mut clocks = [0.0f64; 3];
            for layer in &target.layers {
                let moe = &layer.moe;
                for (sum, v) in counts.iter_mut().zip(
                    moe.prefetch_profile
                        .into_iter()
                        .chain([moe.prefetched, moe.overlap_hot_rows]),
                ) {
                    *sum += v;
                }
                for (sum, v) in clocks
                    .iter_mut()
                    .zip(moe.profile_seconds.into_iter().chain([moe.import_seconds]))
                {
                    *sum += v;
                }
            }
            ensure!(
                decode_seconds.is_finite()
                    && decode_seconds > 0.0
                    && clocks
                        .iter()
                        .chain(&session.profile_seconds)
                        .all(|v| v.is_finite() && *v >= 0.0)
            );
            println!(
                "SSD_LIGHT_PROFILE {}",
                serde_json::json!({
                    "status":"completed_diagnostic_awaiting_external_oracle", "phase":phase, "prompt":label, "mtp":mtp,
                    "lookahead_depth":target.lookahead_depth, "tokens":output, "text":tokenizer.decode(&output)?,
                    "logit_count":host.len(), "logits_finite":true, "prompt_tokens":tokens.len(),
                    "mtp_proposed_tokens":session.proposed, "mtp_accepted_tokens":session.accepted, "mtp_blocks":session.blocks,
                    "hits":hits, "misses":misses, "total_read_bytes":bytes, "prefetch":prefetch,
                    "candidate_scanned_hot_admitted_imported_overlap_rows":counts,
                    "route_fill_import_seconds":clocks, "mtp_head_verify_repair_total_seconds":session.profile_seconds,
                    "decode_seconds":decode_seconds, "memory":crate::array::memory_stats(false)?,
                    "limits":"test clocks; one prompt/mode per process; no full-state copies; external H20 hash/text oracle required; not release throughput"
                })
            );
        }
        Ok(())
    }

    #[cfg(all(kani, feature = "chat"))]
    #[kani::proof]
    fn physical_light_profile_preserves_real_path() {
        flash_light_profile_preserves_natural_decode().unwrap();
    }

    #[cfg(feature = "chat")]
    #[test]
    #[ignore = "physical Apple GPU; intrusive diagnostic, not a throughput benchmark"]
    fn flash_decode_cost_profile_preserves_tokens_logits_and_states() -> Result<()> {
        use std::time::Instant;
        const STEPS: usize = 32;
        fn prefill(
            target: &mut FlashNext,
            mut head: Option<&mut FlashMtpHead>,
            tokens: &[u32],
        ) -> Result<u32> {
            target.reset();
            if let Some(head) = &mut head {
                head.reset();
            }
            let mut logits = None;
            for &token in tokens {
                if let (Some(head), Some(hidden)) = (&mut head, &target.last_hidden) {
                    head.extend_cache(target, hidden, token)?;
                }
                logits = Some(target.forward(token)?);
            }
            Ok(logits.context("empty profile prompt")?.chat_greedy_ids()?[0])
        }
        fn clear_clocks(target: &mut FlashNext) {
            for layer in &mut target.layers {
                layer.moe.profile_seconds = [0.0; 2];
            }
        }
        fn counters(target: &FlashNext, before: (u64, u64, u64)) -> Result<Value> {
            let (hits, misses, bytes) = target.io_stats();
            let clocks = target.layers.iter().fold([0.0; 2], |mut sum, layer| {
                for (total, value) in sum.iter_mut().zip(layer.moe.profile_seconds) {
                    *total += value;
                }
                sum
            });
            ensure!(clocks.iter().all(|v| v.is_finite() && *v >= 0.0));
            Ok(serde_json::json!({
                "hits": hits.checked_sub(before.0).context("hit counter reversed")?,
                "misses": misses.checked_sub(before.1).context("miss counter reversed")?,
                "logical_read_bytes": bytes.checked_sub(before.2).context("byte counter reversed")?,
                "route_frontier_seconds": clocks[0], "fill_seconds": clocks[1]
            }))
        }
        let path = std::env::var("MLXL3_FLASH_MODEL")?;
        let checkpoint = crate::checkpoint::inspect(Path::new(&path))?;
        let tokenizer = crate::tokenizer::ChatTokenizer::load(Path::new(&path))?;
        let mut target = FlashNext::from_checkpoint(&checkpoint)?;
        let mut head = FlashMtpHead::load(&checkpoint, &target)?;
        for (label, prompt) in [
            (
                "MAIN",
                "Explique en français, avec un exemple concret, comment un cache LFU évite les lectures SSD inutiles pendant l’inférence d’un modèle MoE.",
            ),
            (
                "HOLD",
                "Écris une fonction Python merge_intervals qui fusionne les intervalles qui se chevauchent, puis explique sa complexité et donne un exemple.",
            ),
        ] {
            let rendered = tokenizer.render(&[crate::tokenizer::Message {
                role: "user".into(),
                content: prompt.into(),
            }])?;
            let tokens = tokenizer.encode(&rendered)?;
            ensure!(!tokens.is_empty() && tokens.len() + STEPS <= target.context_limit() as usize);
            let mut anchor = prefill(&mut target, None, &tokens)?;
            for _ in 0..STEPS {
                anchor = target.forward(anchor)?.chat_greedy_ids()?[0];
            }
            let mut anchor = prefill(&mut target, None, &tokens)?;
            clear_clocks(&mut target);
            let before = target.io_stats();
            let mut golden = Vec::new();
            let mut golden_logits = Vec::new();
            let mut native_seconds = 0.0;
            for _ in 0..STEPS {
                let start = Instant::now();
                let logits = target.forward(anchor)?;
                anchor = logits.chat_greedy_ids()?[0];
                native_seconds += start.elapsed().as_secs_f64();
                golden.push(anchor);
                golden_logits.push(logits.to_bytes()?);
            }
            let expected_state = state_bytes(&target)?;
            let native = counters(&target, before)?;
            let mut anchor = prefill(&mut target, None, &tokens)?;
            clear_clocks(&mut target);
            let before = target.io_stats();
            let mut layers = vec![[0.0; 3]; target.layers.len()];
            let mut final_seconds = 0.0;
            for step in 0..STEPS {
                let mut streams = target.streams(anchor)?;
                streams.eval()?;
                for (layer, times) in target.layers.iter_mut().zip(&mut layers) {
                    let start = Instant::now();
                    let (residual, mixed, gate) = layer.prepare(&streams, anchor)?;
                    mixed.eval()?;
                    residual.eval()?;
                    gate.eval()?;
                    match &layer.attention {
                        Attention::Gdn(g) => {
                            if let Some(state) = &g.state {
                                state.eval()?;
                            }
                        }
                        Attention::Qsa(q) => {
                            if let Some(blocks) = &q.blocks {
                                blocks.eval()?;
                            }
                        }
                    }
                    times[0] += start.elapsed().as_secs_f64();
                    let start = Instant::now();
                    let moe = layer.moe.forward(&mixed)?;
                    moe.eval()?;
                    times[1] += start.elapsed().as_secs_f64();
                    let start = Instant::now();
                    streams = Hc::write(&residual, &moe, &gate)?;
                    streams.eval()?;
                    times[2] += start.elapsed().as_secs_f64();
                }
                let start = Instant::now();
                let logits = target.logits(&streams)?;
                target.drain()?;
                target.last_hidden = Some(streams);
                target.offset += 1;
                anchor = logits.chat_greedy_ids()?[0];
                final_seconds += start.elapsed().as_secs_f64();
                ensure!(
                    anchor == golden[step],
                    "profile token differs {label}/{step}"
                );
                ensure!(
                    logits.to_bytes()? == golden_logits[step],
                    "profile logits differ {label}/{step}"
                );
            }
            ensure!(
                state_bytes(&target)? == expected_state,
                "profile states differ {label}"
            );
            let intrusive = counters(&target, before)?;
            let mut anchor = prefill(&mut target, Some(&mut head), &tokens)?;
            let mut session = FlashMtpSession::default();
            for step in 0..STEPS {
                anchor = session.advance(
                    &mut target,
                    &mut head,
                    anchor,
                    tokens.len() + STEPS,
                    STEPS - step,
                )?;
            }
            let mut anchor = prefill(&mut target, Some(&mut head), &tokens)?;
            let mut session = FlashMtpSession::default();
            clear_clocks(&mut target);
            let before = target.io_stats();
            let start = Instant::now();
            for (step, &expected) in golden.iter().enumerate() {
                anchor = session.advance(
                    &mut target,
                    &mut head,
                    anchor,
                    tokens.len() + STEPS,
                    STEPS - step,
                )?;
                ensure!(
                    anchor == expected,
                    "profile MTP token differs {label}/{step}"
                );
                ensure!(
                    head.offset() == target.offset() - 1,
                    "profile head alignment differs"
                );
            }
            let mtp_seconds = start.elapsed().as_secs_f64();
            ensure!(session.pending.is_empty() && target.offset() as usize == tokens.len() + STEPS);
            ensure!(
                state_bytes(&target)? == expected_state,
                "profile MTP target states differ {label}"
            );
            ensure!(
                layers
                    .iter()
                    .flatten()
                    .chain(&session.profile_seconds)
                    .all(|v| v.is_finite() && *v >= 0.0)
            );
            println!(
                "FLASH_COST_PROFILE {}",
                serde_json::json!({
                    "status":"validated_diagnostic", "prompt":label, "steps":STEPS,
                    "prompt_tokens":tokens.len(), "native_seconds":native_seconds,
                    "native":native, "intrusive":intrusive, "intrusive_layer_seconds":layers,
                    "intrusive_final_seconds":final_seconds, "mtp_seconds":mtp_seconds,
                    "mtp":counters(&target, before)?, "mtp_head_verify_repair_total_seconds":session.profile_seconds,
                    "mtp_proposed":session.proposed, "mtp_accepted":session.accepted,
                    "mtp_blocks":session.blocks, "tokens":golden,
                    "parity":"all32 full logits + all target states, MTP tokens/target states",
                    "limits":"test-only clocks, intrusive evals, one32-step diagnostic, head resident in every phase; not release throughput or exclusive GPU time"
                })
            );
        }
        Ok(())
    }
    #[cfg(all(kani, feature = "chat"))]
    #[kani::proof]
    fn physical_profile_preserves_real_path() {
        flash_decode_cost_profile_preserves_tokens_logits_and_states().unwrap();
    }
    #[test]
    #[ignore = "physical Apple GPU and full Flash checkpoint; greedy session and cache boundaries"]
    fn flash_mtp_session_preserves_greedy_sequence_target_and_head_states() -> Result<()> {
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut target = FlashNext::from_checkpoint(&c)?;
        let mut head = FlashMtpHead::load(&c, &target)?;
        fn prefill(target: &mut FlashNext, head: &mut FlashMtpHead) -> Result<u32> {
            target.reset();
            head.reset();
            let mut logits = None;
            for token in [EOS, 78768, 11] {
                if let Some(hidden) = &target.last_hidden {
                    head.extend_cache(target, hidden, token)?;
                }
                logits = Some(target.forward(token)?);
            }
            Ok(logits.context("empty fixture")?.chat_greedy_ids()?[0])
        }
        fn head_bytes(head: &FlashMtpHead) -> Result<Vec<Vec<u8>>> {
            let Attention::Qsa(q) = &head.layer.attention else {
                anyhow::bail!("invalid head")
            };
            qsa_bytes(q)
        }
        let (mut proposed, mut accepted) = (0, 0);
        for budget in [1, 2, 3, 17] {
            let mut anchor = prefill(&mut target, &mut head)?;
            let mut golden = Vec::new();
            for _ in 0..budget {
                head.extend_cache(
                    &target,
                    target.last_hidden.as_ref().context("missing hidden")?,
                    anchor,
                )?;
                anchor = target.forward(anchor)?.chat_greedy_ids()?[0];
                golden.push(anchor);
            }
            let target_expected = state_bytes(&target)?;
            let head_expected = head_bytes(&head)?;
            let mut anchor = prefill(&mut target, &mut head)?;
            let mut session = FlashMtpSession::default();
            let before = (state_bytes(&target)?, head_bytes(&head)?);
            ensure!(
                session
                    .advance(&mut target, &mut head, anchor, 3 + budget, 0)
                    .is_err(),
                "empty output budget accepted"
            );
            ensure!(
                session
                    .advance(&mut target, &mut head, 248320, 3 + budget, budget)
                    .is_err(),
                "invalid anchor accepted"
            );
            ensure!(
                before == (state_bytes(&target)?, head_bytes(&head)?),
                "invalid request changed state"
            );
            for (index, &expected) in golden.iter().enumerate() {
                anchor =
                    session.advance(&mut target, &mut head, anchor, 3 + budget, budget - index)?;
                ensure!(
                    anchor == expected,
                    "MTP token differs at budget={budget}, step={index}"
                );
                ensure!(
                    head.offset() == target.offset() - 1,
                    "cache alignment differs"
                );
                ensure!(target.offset() <= (3 + budget) as i32, "context exceeded");
            }
            ensure!(
                session.pending.is_empty(),
                "finished budget left output pending"
            );
            ensure!(
                target_expected == state_bytes(&target)?,
                "session target state differs, budget={budget}"
            );
            ensure!(
                head_expected == head_bytes(&head)?,
                "session head state differs, budget={budget}"
            );
            println!(
                "MTP session budget={budget}, proposed={}, accepted={}, blocks={}",
                session.proposed, session.accepted, session.blocks
            );
            proposed += session.proposed;
            accepted += session.accepted;
        }
        ensure!(
            accepted > 0 && proposed > accepted,
            "fixture did not cover acceptance and rejection"
        );
        let anchor = prefill(&mut target, &mut head)?;
        head.reset();
        ensure!(
            FlashMtpSession::default()
                .advance(&mut target, &mut head, anchor, 64, 4)
                .is_err(),
            "misaligned cache accepted"
        );
        ensure!(
            target.offset() == 0 && head.offset() == 0 && target.last_hidden.is_none(),
            "failed session did not reset target/head"
        );
        Ok(())
    }
    #[test]
    #[ignore = "physical Apple GPU and full Flash checkpoint; exact two-token transactions"]
    fn flash_pair_verification_preserves_serial_logits_and_all_states() -> Result<()> {
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut model = FlashNext::from_checkpoint(&c)?;
        let tokens = [EOS, 1, 42, EOS, 7, 248319, 11];
        for prefix in 0..=4 {
            model.reset();
            for &token in &tokens[..prefix] {
                model.forward(token)?;
            }
            let first = model.forward(tokens[prefix])?.to_bytes()?;
            let first_state = state_bytes(&model)?;
            let second = model.forward(tokens[prefix + 1])?.to_bytes()?;
            let second_state = state_bytes(&model)?;
            let next = model.forward(tokens[prefix + 2])?.to_bytes()?;
            let next_state = state_bytes(&model)?;
            for retained in [1, 2] {
                model.reset();
                for &token in &tokens[..prefix] {
                    model.forward(token)?;
                }
                let before = state_bytes(&model)?;
                ensure!(
                    model.verify_pair([248320, 7]).is_err(),
                    "invalid verify token accepted"
                );
                ensure!(
                    before == state_bytes(&model)?,
                    "invalid verification changed state"
                );
                let (logits, hidden) = model.verify_pair([tokens[prefix], tokens[prefix + 1]])?;
                ensure!(
                    logits.shape() == [2, 248320] && hidden.shape() == [2, WIDTH],
                    "invalid verification shapes"
                );
                ensure!(
                    logits.slice(0, 0, 1)?.to_bytes()? == first
                        && logits.slice(0, 1, 2)?.to_bytes()? == second,
                    "paired logits differ at prefix {prefix}"
                );
                ensure!(
                    model.commit_pair(0).is_err() && model.commit_pair(3).is_err(),
                    "invalid commit accepted"
                );
                ensure!(
                    model.forward(7).is_err() && model.verify_pair([1, 2]).is_err(),
                    "pending transaction bypassed"
                );
                model.commit_pair(retained)?;
                ensure!(
                    state_bytes(&model)?
                        == if retained == 1 {
                            &first_state
                        } else {
                            &second_state
                        }
                        .clone(),
                    "retained states differ, prefix={prefix}, retained={retained}"
                );
                ensure!(
                    model.commit_pair(retained).is_err(),
                    "duplicate commit accepted"
                );
                let token = tokens[prefix + retained as usize];
                let output = model.forward(token)?.to_bytes()?;
                ensure!(
                    output == if retained == 1 { &second } else { &next }.clone(),
                    "next token differs after commit"
                );
                ensure!(
                    state_bytes(&model)?
                        == if retained == 1 {
                            &second_state
                        } else {
                            &next_state
                        }
                        .clone(),
                    "next states differ after commit"
                );
                let context = model.context;
                model.context = model.offset + 1;
                let before = state_bytes(&model)?;
                ensure!(
                    model.verify_pair([1, 2]).is_err(),
                    "context overflow accepted"
                );
                ensure!(
                    before == state_bytes(&model)?,
                    "context rejection changed state"
                );
                model.context = context;
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "physical Apple GPU and independent Flash MTP fixture"]
    fn flash_mtp_head_matches_reference_and_cache_only() -> Result<()> {
        let path = std::env::var("MLXL3_FLASH_MTP_REFERENCE")?;
        let root = Path::new(&path);
        let data: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
        ensure!(
            data["status"] == "complete" && data["mode"] == "eval",
            "invalid MTP oracle"
        );
        let steps = data["mtp_steps"].as_array().context("missing MTP steps")?;
        ensure!(
            (4..=31).contains(&steps.len()),
            "insufficient MTP oracle coverage"
        );
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let target = FlashNext::from_checkpoint(&c)?;
        let mut head = FlashMtpHead::load(&c, &target)?;
        for rows in [1, 2, 5] {
            ensure!(
                head.hidden_fc.forward(&Array::zeros(&[rows, H])?).is_err(),
                "invalid MTP FC stream count accepted"
            );
        }
        for (index, case) in steps.iter().enumerate() {
            let hidden = input(root, &case["input"], WIDTH)?;
            let token = u32::try_from(case["token"].as_u64().context("missing MTP token")?)?;
            let Attention::Qsa(q) = &head.layer.attention else {
                unreachable!()
            };
            let before = q.snapshot()?;
            let logits = head.forward(&target, &hidden, token)?;
            for (name, array) in head
                .stages
                .iter()
                .chain(std::iter::once(&("logits".into(), logits.try_clone()?)))
            {
                check(
                    root,
                    &case[name],
                    array,
                    &format!("MTP/{index}/{name}"),
                    0.01,
                )?;
                ensure!(
                    array.astype(Dtype::Float32)?.to_bytes()?
                        == input(root, &case[name], array.shape().iter().product())?.to_bytes()?,
                    "MTP {name} differs at step {index}"
                );
            }
            ensure!(
                logits.chat_greedy_ids()?
                    == [case["argmax"].as_u64().context("missing MTP argmax")? as u32],
                "MTP argmax differs"
            );
            let Attention::Qsa(q) = &head.layer.attention else {
                unreachable!()
            };
            for (name, array) in [
                ("keys", &q.keys),
                ("values", &q.values),
                ("index_tail", &q.index_tail),
            ] {
                if let Some(array) = array {
                    check(
                        root,
                        &case[name],
                        array,
                        &format!("MTP/{index}/{name}"),
                        0.003,
                    )?;
                    ensure!(
                        array.astype(Dtype::Float32)?.to_bytes()?
                            == input(root, &case[name], array.shape().iter().product())?
                                .to_bytes()?,
                        "MTP cache {name} differs"
                    );
                } else {
                    ensure!(case[name].is_null(), "MTP cache missing");
                }
            }
            let expected = qsa_bytes(q)?;
            let Attention::Qsa(q) = &mut head.layer.attention else {
                unreachable!()
            };
            q.restore(before);
            head.extend_cache(&target, &hidden, token)?;
            let Attention::Qsa(q) = &head.layer.attention else {
                unreachable!()
            };
            ensure!(
                expected == qsa_bytes(q)?,
                "cache-only differs from full head at {index}"
            );
            let before = qsa_bytes(q)?;
            ensure!(
                head.extend_cache(&target, &hidden, 248320).is_err(),
                "invalid MTP token accepted"
            );
            ensure!(
                head.extend_cache(&target, &Array::zeros(&[1, H])?, token)
                    .is_err(),
                "invalid MTP hidden accepted"
            );
            let Attention::Qsa(q) = &head.layer.attention else {
                unreachable!()
            };
            ensure!(before == qsa_bytes(q)?, "rejected MTP input changed cache");
        }
        head.reset();
        ensure!(head.offset() == 0, "MTP reset offset wrong");
        Ok(())
    }
    fn input(root: &Path, entry: &Value, width: i32) -> Result<Array> {
        let bytes =
            std::fs::read(root.join(entry["file"].as_str().context("missing reference input")?))?;
        ensure!(
            bytes.len() == width as usize * 4,
            "invalid reference input length"
        );
        Array::from_bytes(&bytes, &[1, width], Dtype::Float32)
    }
    #[test]
    #[ignore = "physical Apple GPU and independent Pony projection fixture"]
    fn flash_projections_match_upstream_rounding() -> Result<()> {
        let root = std::env::var("MLXL3_FLASH_PROJECTION_REFERENCE")?;
        let root = Path::new(&root);
        let data: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
        ensure!(data["status"] == "complete", "projection oracle incomplete");
        let cases = data["projections"]
            .as_array()
            .context("missing projections")?;
        ensure!(cases.len() == 20, "insufficient projection coverage");
        let checkpoint =
            crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        for (index, case) in cases.iter().enumerate() {
            let width = i32::try_from(case["input_width"].as_i64().context("missing width")?)?;
            let out = i32::try_from(
                case["output_width"]
                    .as_i64()
                    .context("missing output width")?,
            )?;
            let projection = Projection::load(
                &checkpoint,
                case["prefix"].as_str().context("missing projection")?,
                width,
                out,
                false,
            )?;
            let dtype = if case["dtype"] == "mlx.core.float16" {
                Dtype::Float16
            } else {
                Dtype::Float32
            };
            let x = input(root, &case["input"], width)?.astype(dtype)?;
            let y = projection.forward(&x)?;
            check(
                root,
                &case["output"],
                &y,
                &format!("projection/{index}"),
                1e-5,
            )?;
            ensure!(
                y.astype(Dtype::Float32)?.to_bytes()?
                    == input(root, &case["output"], out)?.to_bytes()?,
                "projection {index} differs bitwise"
            );
        }
        Ok(())
    }
    #[test]
    #[ignore = "physical Apple GPU and independent upstream HC fixture"]
    fn flash_hc_write_matches_upstream_rounding() -> Result<()> {
        let root = std::env::var("MLXL3_FLASH_NORM_REFERENCE")?;
        let root = Path::new(&root);
        let data: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
        ensure!(data["status"] == "complete", "HC oracle incomplete");
        let cases = data["hc_write"].as_array().context("missing HC cases")?;
        ensure!(cases.len() == 6, "insufficient HC coverage");
        for (index, case) in cases.iter().enumerate() {
            let dtype = if case["branch_dtype"] == "mlx.core.float16" {
                Dtype::Float16
            } else {
                Dtype::Float32
            };
            let streams = input(root, &case["streams"], WIDTH)?;
            let branch = input(root, &case["branch"], H)?.astype(dtype)?;
            let gate = input(root, &case["gate"], HC)?;
            let actual = Hc::write(&streams, &branch, &gate)?.to_bytes()?;
            let expected = input(root, &case["output"], WIDTH)?.to_bytes()?;
            ensure!(actual == expected, "HC write differs bitwise at {index}");
        }
        Ok(())
    }
    #[test]
    #[ignore = "physical Apple GPU and independent upstream norm fixture"]
    fn flash_norms_match_upstream_rounding() -> Result<()> {
        let root = std::env::var("MLXL3_FLASH_NORM_REFERENCE")?;
        let root = Path::new(&root);
        let data: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
        ensure!(data["status"] == "complete", "norm oracle incomplete");
        let cases = data["cases"].as_array().context("missing norm cases")?;
        ensure!(cases.len() == 40, "insufficient norm coverage");
        let checkpoint =
            crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        for (index, case) in cases.iter().enumerate() {
            let width = i32::try_from(case["width"].as_i64().context("missing width")?)?;
            let groups = i32::try_from(case["groups"].as_i64().context("missing groups")?)?;
            let norm = Norm::load(
                &checkpoint,
                case["weight"].as_str().context("missing weight")?,
                width,
                groups,
                1e-6,
                case["centered"]
                    .as_bool()
                    .context("missing centered flag")?,
            )?;
            let dtype = match case["dtype"].as_str() {
                Some("mlx.core.float16") => Dtype::Float16,
                Some("mlx.core.float32") => Dtype::Float32,
                other => anyhow::bail!("unsupported oracle dtype {other:?}"),
            };
            let rows = i32::try_from(case["rows"].as_i64().context("missing rows")?)?;
            let x = input(root, &case["input"], rows * width * groups)?
                .reshape(&[rows, width * groups])?
                .astype(dtype)?;
            let y = norm.forward(&x)?;
            check(root, &case["output"], &y, &format!("norm/{index}"), 1e-5)?;
            let expected = input(root, &case["output"], rows * width * groups)?;
            ensure!(
                y.astype(Dtype::Float32)?.to_bytes()? == expected.to_bytes()?,
                "norm {index} differs bitwise"
            );
        }
        Ok(())
    }
    #[test]
    #[ignore = "physical Apple GPU and independent QSA-stage fixture"]
    fn flash_qsa_stages_match_upstream() -> Result<()> {
        let root = std::env::var("MLXL3_FLASH_ATTENTION_REFERENCE")?;
        let root = Path::new(&root);
        let data: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
        ensure!(
            data["status"] == "complete" && data["mode"] == "eval",
            "QSA oracle incomplete or wrong mode"
        );
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        for index in [15, 35] {
            let steps = data["attention"][index.to_string()]
                .as_array()
                .context("missing QSA probe")?;
            ensure!((5..=32).contains(&steps.len()), "insufficient QSA coverage");
            let mut qsa = Qsa::load(
                &c,
                &format!("model.language_model.layers.{index}.self_attn"),
                1e-6,
                1e7,
            )?;
            for shape in [&[1, 256][..], &[1, 0, 1, 256], &[1, 2, 2, 256]] {
                let x = Array::zeros_dtype(shape, Dtype::Float16)?;
                ensure!(qsa.rope(&x, 64, 0).is_err(), "invalid RoPE shape accepted");
            }
            let x = Array::zeros_dtype(&[1, 2, 1, 256], Dtype::Float16)?;
            ensure!(
                qsa.rope(&x, 64, -1).is_err(),
                "negative RoPE offset accepted"
            );
            ensure!(
                qsa.rope(&x, 65, 0).is_err(),
                "invalid rotary width accepted"
            );
            ensure!(
                qsa.rope(&x.astype(Dtype::Float32)?, 64, 0).is_err(),
                "invalid RoPE dtype accepted"
            );
            ensure!(
                qsa.offset == 0 && qsa.keys.is_none(),
                "rejected RoPE changed cache"
            );
            for (step, case) in steps.iter().enumerate() {
                let output = qsa.forward(&input(root, &case["input"], H)?)?;
                for (name, value) in qsa
                    .stages
                    .iter()
                    .chain(std::iter::once(&("out".into(), output)))
                {
                    let reference = &case["stages"][name];
                    check(
                        root,
                        reference,
                        value,
                        &format!("QSA/{index}/{step}/{name}"),
                        f64::INFINITY,
                    )?;
                    let expected = input(root, reference, value.shape().iter().product())?;
                    ensure!(
                        value.astype(Dtype::Float32)?.to_bytes()? == expected.to_bytes()?,
                        "QSA stage differs bitwise at layer {index}/step {step}/{name}"
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "physical Apple GPU and independent PLE-stage fixture"]
    fn flash_ple_stages_match_upstream() -> Result<()> {
        let root = std::env::var("MLXL3_FLASH_ATTENTION_REFERENCE")?;
        let root = Path::new(&root);
        let data: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
        ensure!(
            data["status"] == "complete" && data["mode"] == "eval",
            "PLE oracle incomplete or wrong mode"
        );
        let steps = data["attention"]["1"]
            .as_array()
            .context("missing PLE probe")?;
        ensure!((5..=32).contains(&steps.len()), "insufficient PLE coverage");
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut ple = Ple::load(&c, "model.language_model.layers.1.ple", 1e-6)?;
        for (step, case) in steps.iter().enumerate() {
            let output = ple.forward(
                &input(root, &case["input"], WIDTH)?,
                u32::try_from(case["token"].as_u64().context("missing PLE token")?)?,
            )?;
            for (name, value) in ple
                .stages
                .iter()
                .chain(std::iter::once(&("out".into(), output)))
            {
                let reference = &case["stages"][name];
                check(
                    root,
                    reference,
                    value,
                    &format!("PLE/{step}/{name}"),
                    f64::INFINITY,
                )?;
                let expected = input(root, reference, value.shape().iter().product())?;
                ensure!(
                    value.astype(Dtype::Float32)?.to_bytes()? == expected.to_bytes()?,
                    "PLE stage differs bitwise at step {step}/{name}"
                );
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "physical Apple GPU and independent attention-stage fixture"]
    fn flash_attention_stages_match_upstream() -> Result<()> {
        let root = std::env::var("MLXL3_FLASH_ATTENTION_REFERENCE")?;
        let root = Path::new(&root);
        let data: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
        ensure!(
            data["status"] == "complete" && data["mode"] == "eval",
            "attention oracle incomplete or wrong mode"
        );
        let steps = data["attention"]["2"]
            .as_array()
            .context("missing GDN probe")?;
        ensure!(
            (5..=32).contains(&steps.len()),
            "insufficient attention coverage"
        );
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut gdn = Gdn::load(&c, "model.language_model.layers.2.linear_attn", 1e-6)?;
        for (step, case) in steps.iter().enumerate() {
            let output = gdn.forward(&input(root, &case["input"], H)?)?;
            for (name, value) in gdn
                .stages
                .iter()
                .chain(std::iter::once(&("out".into(), output)))
            {
                let reference = &case["stages"][name];
                check(
                    root,
                    reference,
                    value,
                    &format!("GDN/{step}/{name}"),
                    f64::INFINITY,
                )?;
                let expected = input(root, reference, value.shape().iter().product())?;
                ensure!(
                    value.astype(Dtype::Float32)?.to_bytes()? == expected.to_bytes()?,
                    "attention stage differs bitwise at step {step}/{name}"
                );
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "physical M5 and independent per-component Flash fixture"]
    fn flash_components_match_reference_on_identical_inputs() -> Result<()> {
        let root = std::env::var("MLXL3_FLASH_REFERENCE")?;
        let root = Path::new(&root);
        let data: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
        ensure!(
            data["status"] == "complete" && data["mode"] == "eval",
            "oracle incomplete or wrong mode"
        );
        let checkpoint =
            crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut model = FlashNext::from_checkpoint(&checkpoint)?;
        let steps = data["steps"].as_array().context("missing steps")?;
        ensure!(steps.len() >= 5, "insufficient component coverage");
        for (step, values) in steps.iter().enumerate() {
            for (index, layer) in model.layers.iter_mut().enumerate() {
                let component = &values["components"][index];
                for (name, hc) in [
                    ("attn_hyper_connection", &layer.attn_hc),
                    ("mlp_hyper_connection", &layer.mlp_hc),
                ] {
                    let reference = &component[name];
                    let (mixed, gate) = hc.read(&input(root, &reference["input"], WIDTH)?)?;
                    check(
                        root,
                        &reference["mixed"],
                        &mixed,
                        &format!("s{step}/l{index}/{name}"),
                        0.005,
                    )?;
                    check(
                        root,
                        &reference["injection"],
                        &gate.context("missing gate")?,
                        &format!("s{step}/l{index}/{name}/gate"),
                        0.005,
                    )?;
                    ensure!(
                        mixed.to_bytes()? == input(root, &reference["mixed"], H)?.to_bytes()?,
                        "HC mixed differs bitwise at s{step}/l{index}/{name}"
                    );
                }
                let name = match &layer.attention {
                    Attention::Gdn(_) => "linear_attn",
                    Attention::Qsa(_) => "self_attn",
                };
                let reference = &component[name];
                let x = input(root, &reference["input"], H)?;
                let output = match &mut layer.attention {
                    Attention::Gdn(gdn) => gdn.forward(&x)?,
                    Attention::Qsa(qsa) => qsa.forward(&x)?,
                };
                check(
                    root,
                    &reference["output"],
                    &output,
                    &format!("s{step}/l{index}/{name}"),
                    0.005,
                )?;
                ensure!(
                    output.to_bytes()? == input(root, &reference["output"], H)?.to_bytes()?,
                    "attention differs bitwise at s{step}/l{index}/{name}"
                );
                let reference = &component["mlp"];
                let x = input(root, &reference["input"], H)?;
                let mut selected = layer.moe.route(&x)?.0.to_u32()?;
                selected.sort_unstable();
                let expected = data["steps"][step]["route_ids"][index]
                    .as_array()
                    .context("missing reference route cohort")?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .context("invalid reference route")
                            .and_then(|v| Ok(u32::try_from(v)?))
                    })
                    .collect::<Result<Vec<_>>>()?;
                ensure!(selected == expected, "router differs at s{step}/l{index}");
                let output = layer.moe.forward(&x)?;
                check(
                    root,
                    &reference["output"],
                    &output,
                    &format!("s{step}/l{index}/moe"),
                    0.005,
                )?;
                layer.moe.drain()?;
            }
        }
        Ok(())
    }
    fn check(
        root: &Path,
        entry: &Value,
        actual: &Array,
        label: &str,
        tolerance: f64,
    ) -> Result<()> {
        let expected =
            std::fs::read(root.join(entry["file"].as_str().context("missing fixture")?))?;
        ensure!(expected.len().is_multiple_of(4), "invalid oracle bytes");
        let expected = expected
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect::<Vec<_>>();
        let actual = actual.astype(Dtype::Float32)?.to_f32()?;
        ensure!(
            !actual.is_empty()
                && actual.len() == expected.len()
                && actual.len() == entry["count"].as_u64().unwrap_or(0) as usize,
            "oracle count mismatch for {label}"
        );
        ensure!(
            actual.iter().chain(&expected).all(|x| x.is_finite()),
            "nonfinite oracle values for {label}"
        );
        let energy = expected.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>();
        let error = actual
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| f64::from(a - b).powi(2))
            .sum::<f64>();
        let relative = (error / energy.max(1e-30)).sqrt();
        let max = actual
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!(
            "{label}: count={}, relative_rmse={relative:.6}, max_abs={max:.6}",
            actual.len()
        );
        ensure!(
            !tolerance.is_finite() || relative < tolerance,
            "reference differs at {label}: relative RMSE {relative}"
        );
        Ok(())
    }
    #[test]
    #[ignore = "physical M5, full pinned checkpoint and independent mlx-vlm/PonyExl3 fixture"]
    fn flash_matches_independent_reference_states_and_logits() -> Result<()> {
        let root = std::env::var("MLXL3_FLASH_REFERENCE")?;
        let root = Path::new(&root);
        let data: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
        ensure!(
            data["status"] == "complete" && data["mode"] == "eval",
            "oracle incomplete or wrong mode"
        );
        let steps = data["steps"].as_array().context("missing oracle steps")?;
        ensure!(steps.len() >= 5, "empty or insufficient oracle coverage");
        let checkpoint =
            crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut model = FlashNext::from_checkpoint(&checkpoint)?;
        for (i, step) in steps.iter().enumerate() {
            let (logits, trace) = model.forward_trace(u32::try_from(
                step["token"].as_u64().context("invalid token")?,
            )?)?;
            let mut same_routes = true;
            for (index, layer) in model.layers.iter().enumerate() {
                let mut actual = layer.moe.route_ids.clone();
                actual.sort_unstable();
                let expected = step["route_ids"][index]
                    .as_array()
                    .context("missing routes")?
                    .iter()
                    .map(|v| {
                        v.as_u64()
                            .and_then(|v| u32::try_from(v).ok())
                            .context("invalid route")
                    })
                    .collect::<Result<Vec<_>>>()?;
                if actual != expected {
                    eprintln!(
                        "trajectory route mismatch s{i}/l{index}: native={actual:?}, oracle={expected:?}"
                    );
                    same_routes = false;
                }
            }
            for (name, value) in trace {
                let entry = if let Some(index) = name.strip_prefix("layer_") {
                    &step["layers"][index.parse::<usize>()?]
                } else {
                    &step[&name]
                };
                if !entry.is_null() {
                    check(
                        root,
                        entry,
                        &value,
                        &format!("s{i}/{name}"),
                        if name.starts_with("layer_") {
                            f64::INFINITY
                        } else {
                            0.003
                        },
                    )?;
                    if name.starts_with("gdn_state_") {
                        ensure!(
                            value.astype(Dtype::Float32)?.to_bytes()?
                                == input(root, entry, 786432)?.to_bytes()?,
                            "coupled recurrent state differs bitwise at s{i}/{name}"
                        );
                    }
                }
            }
            check(
                root,
                &step["logits"],
                &logits,
                &format!("s{i}/logits"),
                0.01,
            )?;
            ensure!(
                logits.astype(Dtype::Float32)?.to_bytes()?
                    == input(root, &step["logits"], 248320)?.to_bytes()?,
                "full logits differ bitwise at s{i}"
            );
            ensure!(same_routes, "full trajectory routing differs at s{i}");
            let values = logits.astype(Dtype::Float32)?.to_f32()?;
            let chosen = values
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .context("empty logits")?
                .0;
            ensure!(
                chosen == step["argmax"].as_u64().context("missing argmax")? as usize,
                "oracle argmax differs at step {i}"
            );
        }
        let first = steps[0]["token"].as_u64().unwrap() as u32;
        model.reset();
        check(
            root,
            &steps[0]["logits"],
            &model.forward(first)?,
            "reset/logits",
            0.01,
        )?;
        ensure!(model.offset() == 1, "reset offset wrong");
        ensure!(model.forward_tokens(&[]).is_err(), "empty forward accepted");
        ensure!(
            model.forward(248320).is_err(),
            "invalid vocabulary accepted"
        );
        Ok(())
    }
}
