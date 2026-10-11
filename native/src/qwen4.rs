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
};
use anyhow::{Context, Result, ensure};
use half::f16;
use serde_json::Value;
use std::{
    collections::{BTreeMap, VecDeque},
    fs::File,
    os::unix::fs::FileExt,
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
        let weights = self
            .up
            .forward(&self.down.forward(&normalized)?.qwen4_hc_activate()?)?;
        let mixed = weights.qwen4_hc_mix(&normalized)?.reshape(&[1, H])?;
        let injection = self
            .inject
            .as_ref()
            .map(|p| p.forward(&normalized)?.qwen4_hc_gate()?.reshape(&[1, HC]))
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
impl Qsa {
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
        let qg = self
            .q
            .forward(x)?
            .astype(Dtype::Float16)?
            .reshape(&[1, 1, 24, 512])?;
        let gate = qg.slice(3, 256, 512)?.reshape(&[1, 6144])?;
        stage!("q_proj", qg);
        let query_norm = self.q_norm.forward(&qg.slice(3, 0, 256)?)?;
        stage!("q_norm", query_norm);
        let query = self.rope(&query_norm.transpose(&[0, 2, 1, 3])?, 64, self.offset)?;
        stage!("q_rope", query);
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
        let keys = self.keys.as_ref().context("missing QSA keys")?;
        let values = self.values.as_ref().context("missing QSA values")?;
        let count = (self.offset + 1) / 4;
        let (attn_keys, attn_values) = if count > 512 {
            let iq = self.rope(
                &self
                    .iq_norm
                    .forward(&qk.slice(2, 0, 4)?)?
                    .transpose(&[0, 2, 1, 3])?,
                32,
                self.offset,
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
            tokens.extend((count * 4..self.offset + 1).map(|i| i as u32));
            let selected = Array::from_u32(&tokens, &[tokens.len() as i32])?;
            (keys.take(&selected, 2)?, values.take(&selected, 2)?)
        } else {
            (keys.try_clone()?, values.try_clone()?)
        };
        let attention = Array::sdpa(&query, &attn_keys, &attn_values, 1. / 16., false)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, 6144])?;
        self.offset = self.offset.checked_add(1).context("QSA offset exhausted")?;
        let gated = attention.mul(&gate.sigmoid()?)?;
        stage!("gated", gated);
        self.out.forward(&gated)?.astype(Dtype::Float32)
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
pub(crate) struct Layer {
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
        let streams = match &mut self.ple {
            Some(ple) => ple.forward(streams, token)?,
            None => streams.try_clone()?,
        };
        let (mixed, gate) = self.attn_hc.read(&streams)?;
        let branch = match &mut self.attention {
            Attention::Gdn(gdn) => gdn.forward(&mixed)?,
            Attention::Qsa(qsa) => qsa.forward(&mixed)?,
        };
        let streams = Hc::write(
            &streams,
            &branch,
            &gate.context("missing attention injection")?,
        )?;
        let (mixed, gate) = self.mlp_hc.read(&streams)?;
        Hc::write(
            &streams,
            &self.moe.forward(&mixed)?,
            &gate.context("missing MoE injection")?,
        )
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
    embed: TensorInfo,
    layers: Vec<Layer>,
    mixer: Hc,
    head: Projection,
    offset: i32,
    context: i32,
    pub last_hidden: Option<Array>,
}
impl FlashNext {
    pub fn from_checkpoint(c: &Checkpoint) -> Result<Self> {
        ensure!(c.model_type == "qwen4_exp", "expected qwen4_exp");
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
            Ok(value) => value
                .parse::<f64>()
                .context("invalid MLXL3_SSD_CACHE_GIB")?,
            Err(std::env::VarError::NotPresent) => 8.,
            Err(error) => return Err(error.into()),
        };
        ensure!(
            budget.is_finite() && (0.6..=12.).contains(&budget),
            "SSD expert cache budget must be 0.6..12 GiB"
        );
        let slots = ((budget * (1u64 << 30) as f64 / (48. * flash_support::EXPERT_BYTES as f64))
            .floor() as usize)
            .clamp(10, 512);
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
        let mixer = Hc::load(c, "model.language_model.hyper_connection_mixer", eps, false)?;
        let head = Projection::load(c, "lm_head", H, 248320, false)?;
        Ok(Self {
            embed,
            layers,
            mixer,
            head,
            offset: 0,
            context,
            last_hidden: None,
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
        ensure!(self.offset < self.context, "Flash context exhausted");
        let mut trace = Vec::new();
        let mut streams = self
            .embedding(token)?
            .astype(Dtype::Float32)?
            .reshape(&[1, 1, H])?
            .broadcast_to(&[1, HC, H])?
            .reshape(&[1, WIDTH])?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            streams = layer.forward(&streams, token)?;
            if capture {
                trace.push((format!("layer_{i}"), streams.try_clone()?));
                if let Attention::Gdn(gdn) = &layer.attention
                    && let Some(state) = &gdn.state
                {
                    trace.push((format!("gdn_state_{i}"), state.try_clone()?));
                }
            }
        }
        let (mixed, _) = self.mixer.read(&streams)?;
        let logits = self.head.forward(&mixed.astype(Dtype::Float16)?)?;
        logits.eval()?;
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
        self.last_hidden = Some(streams);
        self.offset += 1;
        Ok((logits, trace))
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
    }
    pub fn context_limit(&self) -> i32 {
        self.context
    }
    pub fn offset(&self) -> i32 {
        self.offset
    }
    pub fn io_stats(&self) -> (u64, u64, u64) {
        self.layers.iter().fold((0, 0, 0), |(h, m, b), l| {
            (h + l.moe.hits, m + l.moe.misses, b + l.moe.read_bytes)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;
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
