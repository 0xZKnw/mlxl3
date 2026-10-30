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

#[cfg(feature = "chat")]
#[path = "flash_draft.rs"]
mod draft;

/// Inclusive host timings; verification includes its existing GPU completion waits.
#[cfg(feature = "chat")]
#[derive(Clone, Copy, Debug, Default, serde::Serialize)]
pub struct FlashDraftProfile {
    pub draft_seconds: f64,
    pub verification_seconds: f64,
    pub commit_cache_seconds: f64,
    pub verified_rows: usize,
    pub drafted_rows: usize,
    pub drafted_widths: [usize; 8],
    pub verified_widths: [usize; 8],
    pub expert_demand_bytes: u64,
    /// Submitted, completed, used, reused, timeouts, wait ns, read bytes, late demands.
    pub ssd_prefetch: [u64; 8],
}

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
        #[cfg(test)]
        let _profile = crate::flash_profile::span("hc_read_build");
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
        #[cfg(test)]
        crate::flash_profile::stage("hc_read_completion", &mixed)?;
        Ok((mixed, injection))
    }
    fn write(streams: &Array, branch: &Array, gate: &Array) -> Result<Array> {
        #[cfg(test)]
        let _profile = crate::flash_profile::span("hc_write_build");
        streams.add(
            &branch
                .reshape(&[1, 1, H])?
                .mul(&gate.reshape(&[1, HC, 1])?)?
                .reshape(&[1, WIDTH])?,
        )
    }
}

fn verification_projection(projection: &Projection, input: &Array) -> Result<Array> {
    if projection.is_exl3() {
        return projection.forward(input);
    }
    let rows = (0..input.shape()[0])
        .map(|row| projection.forward(&input.slice(0, row, row + 1)?))
        .collect::<Result<Vec<_>>>()?;
    Array::concatenate(&rows.iter().collect::<Vec<_>>(), 0)
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
    fused_epilogue: bool,
    #[cfg(test)]
    fused_epilogue_calls: std::cell::Cell<usize>,
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
            fused_epilogue: match std::env::var("MLXL3_FLASH_GDN_EPILOGUE") {
                Ok(value) => flash_support::parse_gpu_hot_wide(&value)
                    .context("MLXL3_FLASH_GDN_EPILOGUE must be 0 or 1")?,
                Err(std::env::VarError::NotPresent) => false,
                Err(error) => return Err(error.into()),
            },
            #[cfg(test)]
            fused_epilogue_calls: std::cell::Cell::new(0),
        })
    }
    fn forward(&mut self, x: &Array) -> Result<Array> {
        #[cfg(test)]
        let _profile = crate::flash_profile::span("gdn_single_build");
        #[cfg(test)]
        self.stages.clear();
        macro_rules! stage {
            ($name:expr, $value:expr) => {
                #[cfg(test)]
                {
                    crate::flash_profile::stage($name, &$value)?;
                    self.stages.push(($name.into(), $value.try_clone()?));
                }
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
        let (_conv, q, k, v) = self.convolve(&qkv)?;
        stage!("conv", _conv);
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
        let gated = self.gate_output(&normed, &z)?;
        stage!("gated", gated);
        self.out
            .forward(&gated.reshape(&[1, 6144])?)?
            .astype(Dtype::Float32)
    }
    fn gate_output(&self, normed: &Array, z: &Array) -> Result<Array> {
        if self.fused_epilogue
            && flash_support::gdn_epilogue_shape(normed.shape())
            && normed.shape() == z.shape()
            && normed.dtype() == Dtype::Float16
            && z.dtype() == Dtype::Float16
        {
            #[cfg(test)]
            self.fused_epilogue_calls
                .set(self.fused_epilogue_calls.get() + 1);
            return normed.qwen4_gdn_epilogue(z);
        }
        normed
            .astype(Dtype::Float32)?
            .mul(&z.astype(Dtype::Float32)?.sigmoid()?)?
            .astype(Dtype::Float16)
    }
    fn convolve(&mut self, qkv: &Array) -> Result<(Array, Array, Array, Array)> {
        #[cfg(test)]
        let _profile = crate::flash_profile::span("gdn_convolve_norm_build");
        let history = match &self.history {
            Some(h) => h.try_clone()?,
            None => Array::zeros_dtype(&[1, 3, 10240], Dtype::Float16)?,
        };
        let input = Array::concatenate(&[&history, qkv], 1)?;
        self.history = Some(input.slice(1, 1, 4)?);
        let conv = input
            .astype(Dtype::Float32)?
            .mul(&self.conv.transpose(&[2, 1, 0])?.astype(Dtype::Float32)?)?
            .sum(1, true)?
            .astype(Dtype::Float16)?
            .silu()?;
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
        Ok((conv, q, k, v))
    }

    fn forward_rows(&mut self, inputs: &[Array]) -> Result<(Vec<Array>, Vec<AttentionState>)> {
        #[cfg(test)]
        let _profile = crate::flash_profile::span("gdn_rows");
        ensure!(
            (2..=8).contains(&inputs.len()),
            "invalid GDN verification width"
        );
        #[cfg(test)]
        self.stages.clear();
        let rows = inputs.len() as i32;
        let input =
            Array::concatenate(&inputs.iter().collect::<Vec<_>>(), 0)?.astype(Dtype::Float16)?;
        #[cfg(test)]
        let projection = crate::flash_profile::span("gdn_qkv_build");
        let qkv = verification_projection(&self.qkv, &input)?
            .astype(Dtype::Float16)?
            .reshape(&[1, rows, 10240])?;
        #[cfg(test)]
        {
            drop(projection);
            crate::flash_profile::stage("gdn_rows_qkv_completion", &qkv)?;
        }
        let mut prepared = Vec::with_capacity(inputs.len());
        let mut histories = Vec::with_capacity(inputs.len() - 1);
        for row in 0..rows {
            let (_, q, k, v) = self.convolve(&qkv.slice(1, row, row + 1)?)?;
            prepared.push([q, k, v]);
            if row + 1 < rows {
                histories.push(clone_array(&self.history)?);
            }
        }
        let cat = |column: usize| {
            Array::concatenate(&prepared.iter().map(|p| &p[column]).collect::<Vec<_>>(), 1)
        };
        let q = cat(0)?;
        let k = cat(1)?;
        let v = cat(2)?;
        #[cfg(test)]
        {
            crate::flash_profile::stage("gdn_rows_q_completion", &q)?;
            crate::flash_profile::stage("gdn_rows_k_completion", &k)?;
            crate::flash_profile::stage("gdn_rows_v_completion", &v)?;
        }
        #[cfg(test)]
        let gates_profile = crate::flash_profile::span("gdn_gates_build");
        let a = verification_projection(&self.a, &input)?
            .astype(Dtype::Float16)?
            .reshape(&[1, rows, 48])?;
        let b = verification_projection(&self.b, &input)?
            .astype(Dtype::Float16)?
            .reshape(&[1, rows, 48])?;
        #[cfg(test)]
        {
            drop(gates_profile);
            crate::flash_profile::stage("gdn_rows_a_completion", &a)?;
            crate::flash_profile::stage("gdn_rows_b_completion", &b)?;
        }
        let initial = match &self.state {
            Some(state) => state.try_clone()?,
            None => Array::zeros_dtype(&[1, 48, 128, 128], Dtype::Float32)?,
        };
        let (value, final_state, tape) = gated_delta::step_with_gates_tape(
            &q,
            &k,
            &v,
            &a,
            &b,
            &self.a_log,
            &self.dt_bias,
            &initial,
        )?;
        #[cfg(test)]
        crate::flash_profile::stage("gdn_rows_scan_completion", &value)?;
        self.state = Some(final_state);
        let tape = Rc::new(tape);
        let snapshots = histories
            .into_iter()
            .enumerate()
            .map(|(row, history)| AttentionState::GdnTape {
                history,
                tape: Rc::clone(&tape),
                retained: row as i32 + 1,
            })
            .collect();
        let z = verification_projection(&self.z, &input)?
            .astype(Dtype::Float16)?
            .reshape(&[1, rows, 48, 128])?;
        #[cfg(test)]
        crate::flash_profile::stage("gdn_rows_z_completion", &z)?;
        #[cfg(test)]
        let post_profile = crate::flash_profile::span("gdn_norm_gate_build");
        let gated = if self.fused_epilogue {
            // Keep exactly the native norm call per row. Only the epilogue is batched.
            let normed = (0..rows)
                .map(|row| self.norm.forward(&value.slice(1, row, row + 1)?))
                .collect::<Result<Vec<_>>>()?;
            let normed = Array::concatenate(&normed.iter().collect::<Vec<_>>(), 1)?;
            self.gate_output(&normed, &z)?.reshape(&[rows, 6144])?
        } else {
            let mut gated = Vec::with_capacity(inputs.len());
            for row in 0..rows {
                let normed = self.norm.forward(&value.slice(1, row, row + 1)?)?;
                gated.push(
                    self.gate_output(&normed, &z.slice(1, row, row + 1)?)?
                        .reshape(&[1, 6144])?,
                );
            }
            Array::concatenate(&gated.iter().collect::<Vec<_>>(), 0)?
        };
        #[cfg(test)]
        self.stages.push(("gated".into(), gated.try_clone()?));
        #[cfg(test)]
        drop(post_profile);
        #[cfg(test)]
        crate::flash_profile::stage("gdn_rows_norm_gate_completion", &gated)?;
        let output = verification_projection(&self.out, &gated)?.astype(Dtype::Float32)?;
        #[cfg(test)]
        crate::flash_profile::stage("gdn_rows_out_completion", &output)?;
        Ok((
            (0..rows)
                .map(|row| output.slice(0, row, row + 1))
                .collect::<Result<Vec<_>>>()?,
            snapshots,
        ))
    }
    fn reset(&mut self) {
        self.state = None;
        self.history = None;
    }
}

#[cfg(test)]
#[path = "flash_boundary_tests.rs"]
mod boundary_tests;

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
        let _profile = crate::flash_profile::span("qsa_build");
        #[cfg(test)]
        self.stages.clear();
        macro_rules! stage {
            ($name:expr, $value:expr) => {
                #[cfg(test)]
                {
                    crate::flash_profile::stage($name, &$value)?;
                    self.stages.push(($name.into(), $value.try_clone()?));
                }
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
                {
                    crate::flash_profile::stage($name, &$value)?;
                    self.stages.push(($name.into(), $value.try_clone()?));
                }
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
        let _profile = crate::flash_profile::span("ple_build");
        #[cfg(test)]
        self.stages.clear();
        macro_rules! stage {
            ($name:expr, $value:expr) => {
                #[cfg(test)]
                {
                    crate::flash_profile::stage($name, &$value)?;
                    self.stages.push(($name.into(), $value.try_clone()?));
                }
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
    GdnTape {
        history: Option<Array>,
        tape: Rc<gated_delta::RecurrentTape>,
        retained: i32,
    },
    Qsa(QsaState),
}
struct LayerState {
    attention: AttentionState,
    ple: Option<(Option<Array>, [u32; 2])>,
}
type PreparedRow = (Array, Array, Array);
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
        let written = Hc::write(&streams, &output, &gate)?;
        #[cfg(test)]
        crate::flash_profile::stage("mlp_hc_write", &written)?;
        Ok(written)
    }
    fn prepare(&mut self, streams: &Array, token: u32) -> Result<PreparedRow> {
        #[cfg(test)]
        let _prepare_profile = crate::flash_profile::span("prepare");
        #[cfg(test)]
        self.stages.clear();
        let streams = match &mut self.ple {
            Some(ple) => ple.forward(streams, token)?,
            None => streams.try_clone()?,
        };
        #[cfg(test)]
        crate::flash_profile::stage("ple_output", &streams)?;
        let (mixed, gate) = self.attn_hc.read(&streams)?;
        #[cfg(test)]
        {
            crate::flash_profile::stage("attn_hc_read", &mixed)?;
            crate::flash_profile::stage("attn_hc_gate", gate.as_ref().context("missing gate")?)?;
        }
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
        #[cfg(test)]
        crate::flash_profile::stage("attention_out_projection", &branch)?;
        let streams = Hc::write(
            &streams,
            &branch,
            &gate.context("missing attention injection")?,
        )?;
        #[cfg(test)]
        self.stages.push(("attention".into(), branch.try_clone()?));
        #[cfg(test)]
        crate::flash_profile::stage("attn_hc_write", &streams)?;
        let (mixed, gate) = self.mlp_hc.read(&streams)?;
        #[cfg(test)]
        {
            crate::flash_profile::stage("mlp_hc_read", &mixed)?;
            crate::flash_profile::stage("mlp_hc_gate", gate.as_ref().context("missing gate")?)?;
        }
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
    fn prepare_rows(
        &mut self,
        streams: &[Array],
        tokens: &[u32],
        multi: bool,
    ) -> Result<(Vec<PreparedRow>, Vec<LayerState>)> {
        #[cfg(test)]
        let _profile = crate::flash_profile::span("prepare_rows");
        ensure!(
            streams.len() == tokens.len() && (1..=8).contains(&tokens.len()),
            "invalid Flash preparation rows"
        );
        if !multi || tokens.len() == 1 || matches!(self.attention, Attention::Qsa(_)) {
            let mut prepared = Vec::with_capacity(tokens.len());
            let mut states = Vec::with_capacity(tokens.len() - 1);
            for (row, (&token, stream)) in tokens.iter().zip(streams).enumerate() {
                prepared.push(self.prepare(stream, token)?);
                if row + 1 < tokens.len() {
                    states.push(self.snapshot()?);
                }
            }
            return Ok((prepared, states));
        }
        let mut residuals = Vec::with_capacity(tokens.len());
        let mut mixed = Vec::with_capacity(tokens.len());
        let mut gates = Vec::with_capacity(tokens.len());
        let mut ple_states = Vec::with_capacity(tokens.len() - 1);
        for (row, (&token, stream)) in tokens.iter().zip(streams).enumerate() {
            let residual = match &mut self.ple {
                Some(ple) => ple.forward(stream, token)?,
                None => stream.try_clone()?,
            };
            let (input, gate) = self.attn_hc.read(&residual)?;
            residuals.push(residual);
            mixed.push(input);
            gates.push(gate.context("missing attention injection")?);
            if row + 1 < tokens.len() {
                ple_states.push(
                    self.ple
                        .as_ref()
                        .map(|p| -> Result<_> { Ok((clone_array(&p.history)?, p.rows.previous)) })
                        .transpose()?,
                );
            }
        }
        let Attention::Gdn(gdn) = &mut self.attention else {
            unreachable!()
        };
        let (branches, snapshots) = gdn.forward_rows(&mixed)?;
        let mut prepared = Vec::with_capacity(tokens.len());
        for ((residual, branch), gate) in residuals.iter().zip(&branches).zip(&gates) {
            let residual = Hc::write(residual, branch, gate)?;
            let (input, gate) = self.mlp_hc.read(&residual)?;
            prepared.push((residual, input, gate.context("missing MoE injection")?));
        }
        Ok((
            prepared,
            snapshots
                .into_iter()
                .zip(ple_states)
                .map(|(attention, ple)| LayerState { attention, ple })
                .collect(),
        ))
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
            (
                Attention::Gdn(g),
                AttentionState::GdnTape {
                    history,
                    tape,
                    retained,
                },
            ) => {
                let state = tape.replay(retained)?;
                g.history = history;
                g.state = Some(state);
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
        multi: bool,
    ) -> Result<([Array; 2], LayerState)> {
        #[cfg(test)]
        let _pair_profile = crate::flash_profile::span("forward_pair");
        let (prepared, mut states) = self.prepare_rows(streams, &tokens, multi)?;
        let [a, b]: [PreparedRow; 2] = prepared
            .try_into()
            .map_err(|_| anyhow::anyhow!("incomplete Flash pair preparation"))?;
        let first = states.pop().context("missing Flash pair snapshot")?;
        let output =
            self.moe
                .forward_lookahead(&Array::concatenate(&[&a.1, &b.1], 0)?, next, depth)?;
        let result = [
            Hc::write(&a.0, &output.slice(0, 0, 1)?, &a.2)?,
            Hc::write(&b.0, &output.slice(0, 1, 2)?, &b.2)?,
        ];
        #[cfg(test)]
        for value in &result {
            crate::flash_profile::stage("mlp_hc_write", value)?;
        }
        Ok((result, first))
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
    multi_token: bool,
    lookahead_depth: usize,
    lookahead_pool: Option<Rc<RefCell<ReaderPool>>>,
    #[cfg(feature = "chat")]
    dflash: Option<Box<draft::runtime::Session>>,
}
struct PairState {
    base: i32,
    layers: Vec<LayerState>,
    hidden: Array,
}
impl FlashNext {
    pub fn from_checkpoint(c: &Checkpoint) -> Result<Self> {
        ensure!(c.model_type == "qwen4_exp", "expected qwen4_exp");
        let multi_token = flash_support::multi_token_verifier()?;
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
        let layer_slots = match std::env::var("MLXL3_SSD_LAYER_SLOTS") {
            Ok(value) => flash_support::parse_layer_slots(&value, slots).context(
                "MLXL3_SSD_LAYER_SLOTS requires 48 capacities in 20..512 with exactly the current total cache budget",
            )?,
            Err(std::env::VarError::NotPresent) => [slots; 48],
            Err(error) => return Err(error.into()),
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
                layer_slots[i]
            );
            layers.push(Layer::load(
                c,
                &format!("model.language_model.layers.{i}"),
                linear,
                i == 1,
                eps,
                rope,
                layer_slots[i],
                overlap,
            )?);
            #[cfg(test)]
            assert_eq!(layers[i].moe.slots_for_test(), layer_slots[i]);
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
            multi_token,
            lookahead_depth,
            lookahead_pool,
            #[cfg(feature = "chat")]
            dflash: None,
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
        #[cfg(test)]
        crate::flash_profile::stage("embedding_streams", &streams)?;
        for i in 0..self.layers.len() {
            #[cfg(test)]
            let _layer_profile = crate::flash_profile::span(&format!("layer_{i}"));
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
        #[cfg(test)]
        let _profile = crate::flash_profile::span("logits");
        let (mixed, _) = self.mixer.read(streams)?;
        #[cfg(test)]
        crate::flash_profile::stage("final_hc_read", &mixed)?;
        let logits = self.head.forward(&mixed.astype(Dtype::Float16)?)?;
        #[cfg(test)]
        let _eval = crate::flash_profile::span("lm_head_eval_inclusive");
        logits.eval()?;
        Ok(logits)
    }
    fn drain(&mut self) -> Result<()> {
        #[cfg(test)]
        let _profile = crate::flash_profile::span("drain");
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
        #[cfg(test)]
        let _profile = crate::flash_profile::span("verify_pair");
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
            #[cfg(test)]
            let _layer_profile = crate::flash_profile::span(&format!("layer_{i}"));
            let (current, rest) = self.layers.split_at_mut(i + 1);
            let next = (self.lookahead_depth > 0)
                .then(|| rest.first_mut().map(|l| &mut l.moe))
                .flatten();
            let (output, first) = current[i].forward_pair(
                &streams,
                tokens,
                next,
                self.lookahead_depth,
                self.multi_token,
            )?;
            streams = output;
            states.push(first);
        }
        let logits = if self.multi_token {
            let input = streams
                .iter()
                .map(|stream| -> Result<Array> {
                    self.mixer.read(stream)?.0.astype(Dtype::Float16)
                })
                .collect::<Result<Vec<_>>>()?;
            let logits = verification_projection(
                &self.head,
                &Array::concatenate(&input.iter().collect::<Vec<_>>(), 0)?,
            )?;
            logits.eval()?;
            logits
        } else {
            Array::concatenate(&[&self.logits(&streams[0])?, &self.logits(&streams[1])?], 0)?
        };
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
        #[cfg(test)]
        let _profile = crate::flash_profile::span("commit_pair");
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
        #[cfg(feature = "chat")]
        if let Some(session) = &mut self.dflash {
            session.reset();
        }
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
    #[cfg(feature = "chat")]
    fn prefetch_stats(&self) -> [u64; 8] {
        self.lookahead_pool.as_ref().map_or([0; 8], |pool| {
            let pool = pool.borrow();
            let c = &pool.counters;
            [
                &c.submitted,
                &c.completed,
                &c.used,
                &c.demand_reused,
                &c.demand_timeouts,
                &c.demand_wait_ns,
                &c.bytes,
                &c.late_needed,
            ]
            .map(|c| c.load(Ordering::Relaxed))
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
            #[cfg(test)]
            let _profile = crate::flash_profile::span("mtp_tail_fallback");
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
        #[cfg(test)]
        let head_profile = crate::flash_profile::span("mtp_head_proposal");
        let proposal = head
            .forward(target, &hidden, anchor)?
            .chat_greedy_ids()?
            .first()
            .copied()
            .context("missing Flash MTP proposal")?;
        #[cfg(test)]
        drop(head_profile);
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
            #[cfg(test)]
            let _profile = crate::flash_profile::span("mtp_head_cache_extend");
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
    pub(super) fn state_bytes(model: &FlashNext) -> Result<Vec<Vec<u8>>> {
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
    #[cfg(kani)]
    #[kani::proof]
    #[kani::unwind(100)]
    fn actual_paired_verification_checks_tokens_shapes_and_commit() {
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL").unwrap()))
            .unwrap();
        let mut model = FlashNext::from_checkpoint(&c).unwrap();
        let tokens = kani::any::<[u32; 2]>();
        let retained = if kani::any::<bool>() { 1 } else { 2 };
        match model.verify_pair(tokens) {
            Ok((logits, hidden)) => {
                assert!(tokens.iter().all(|&t| t < 248320));
                assert_eq!(logits.shape(), [2, 248320]);
                assert_eq!(hidden.shape(), [2, WIDTH]);
                assert_eq!(model.offset(), 2);
                model.commit_pair(retained).unwrap();
                assert_eq!(model.offset(), retained);
                kani::cover!(retained == 1);
                kani::cover!(retained == 2);
            }
            Err(_) => {
                kani::cover!(tokens.iter().any(|&t| t >= 248320));
            }
        }
    }
    pub(super) fn qsa_bytes(qsa: &Qsa) -> Result<Vec<Vec<u8>>> {
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
    #[test]
    #[ignore = "physical Apple GPU; original route/head component profile, not release throughput"]
    fn flash_route_head_component_profile() -> Result<()> {
        use std::time::Instant;
        let started = Instant::now();
        let snapshot = |phase: &str| -> Result<()> {
            let output = std::process::Command::new("vm_stat").output()?;
            ensure!(output.status.success(), "VM snapshot failed");
            println!(
                "MTP43_VM {}",
                serde_json::json!({
                    "phase":phase,"seconds":started.elapsed().as_secs_f64(),
                    "raw":String::from_utf8(output.stdout)?
                })
            );
            Ok(())
        };
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut model = FlashNext::from_checkpoint(&c)?;
        snapshot("loaded")?;
        model.forward(EOS)?;
        let first = model
            .last_hidden
            .as_ref()
            .context("missing first hidden")?
            .try_clone()?;
        model.forward(7)?;
        let second = model
            .last_hidden
            .as_ref()
            .context("missing second hidden")?
            .try_clone()?;
        first.eval()?;
        second.eval()?;
        let x = Array::from_f32(
            &(0..2 * H)
                .map(|i| ((i * 17 + 13) % 257) as f32 / 257. - 0.5)
                .collect::<Vec<_>>(),
            &[2, H],
        )?;
        x.eval()?;
        let inputs = [x.slice(0, 0, 1)?, x.slice(0, 1, 2)?];
        for input in &inputs {
            input.eval()?;
        }
        ensure!(model.layers.len() == 48);
        let mut checked_gate_logits = 0;
        for layer in &model.layers {
            for input in &inputs {
                let logits = layer.moe.gate_for_test().forward(input)?;
                ensure!(logits.shape() == [1, 512] && logits.dtype() == Dtype::Float32);
                let values = logits.to_f32()?;
                ensure!(values.len() == 512 && values.iter().all(|v| v.is_finite()));
                checked_gate_logits += values.len();
                let probabilities = logits.softmax_precise()?.to_f32()?;
                ensure!(
                    probabilities.len() == 512
                        && probabilities.iter().all(|v| v.is_finite() && *v >= 0.)
                );
                let mut order = (0..512usize).collect::<Vec<_>>();
                order.sort_by(|&a, &b| {
                    probabilities[a]
                        .total_cmp(&probabilities[b])
                        .then(a.cmp(&b))
                });
                let expected = &order[502..];
                let (indices, weights) = layer.moe.route(input)?;
                ensure!(
                    indices.to_u32()? == expected.iter().map(|&id| id as u32).collect::<Vec<_>>()
                );
                let weights = weights.to_f32()?;
                ensure!(weights.len() == 10 && weights.iter().all(|v| v.is_finite()));
                let total = expected.iter().map(|&id| probabilities[id]).sum::<f32>();
                ensure!(total.is_finite() && total > 0.);
                for (&weight, &id) in weights.iter().zip(expected) {
                    ensure!((weight - probabilities[id] / total).abs() <= 2e-6);
                }
            }
        }
        for streams in [&first, &second] {
            let logits = model.logits(streams)?;
            ensure!(logits.shape() == [1, 248320]);
            let values = logits.to_f32()?;
            ensure!(values.len() == 248320 && values.iter().all(|v| v.is_finite()));
        }
        snapshot("inputs_ready")?;
        let routes = || -> Result<()> {
            for layer in &model.layers {
                let a = layer.moe.route(&inputs[0])?;
                let b = layer.moe.route(&inputs[1])?;
                let ids = Array::concatenate(&[&a.0, &b.0], 0)?.to_u32()?;
                ensure!(ids.len() == 20);
                std::hint::black_box(ids);
            }
            Ok(())
        };
        let heads = || -> Result<()> {
            std::hint::black_box(model.logits(&first)?);
            std::hint::black_box(model.logits(&second)?);
            Ok(())
        };
        for _ in 0..2 {
            routes()?;
            heads()?;
        }
        snapshot("warmup_done")?;
        let mut route_samples = Vec::new();
        let mut head_samples = Vec::new();
        snapshot("route_begin")?;
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..16 {
                routes()?;
            }
            route_samples.push(start.elapsed().as_secs_f64() / 16.);
        }
        snapshot("route_end")?;
        snapshot("head_begin")?;
        for _ in 0..5 {
            let start = Instant::now();
            for _ in 0..16 {
                heads()?;
            }
            head_samples.push(start.elapsed().as_secs_f64() / 16.);
        }
        snapshot("head_end")?;
        ensure!(
            checked_gate_logits == 49152 && route_samples.len() == 5 && head_samples.len() == 5
        );
        ensure!(
            route_samples
                .iter()
                .chain(&head_samples)
                .all(|v| v.is_finite() && *v > 0.)
        );
        let memory = crate::array::memory_stats(false)?;
        let peak = memory
            .process_lifetime_peak_bytes
            .context("missing process peak")?;
        ensure!(peak > 0 && peak <= 10 << 30);
        println!(
            "MTP43_COMPONENT {}",
            serde_json::json!({
                "status":"completed_original_components","gate_layers":48,"gate_logit_count":checked_gate_logits,
                "head_logit_count":496640,"warmup_sweeps":2,"samples":5,"sweeps_per_sample":16,
                "route_pair_48_layers_seconds":route_samples,"head_pair_seconds":head_samples,
                "memory":memory,"limits":"materialized inputs, original two M1 gates and heads; wall-clock components, not release throughput"
            })
        );
        Ok(())
    }
    #[cfg(feature = "chat")]
    #[test]
    #[ignore = "physical Apple GPU; experiment47 cold/warm route capture, not throughput"]
    fn flash_cache_calibration_captures_prefill_decode_and_identical_states() -> Result<()> {
        use crate::flash_cache_profile::{CALIBRATION_PROMPT, CacheTrace, TracePhase};
        fn counts(target: &FlashNext) -> Vec<[u64; 3]> {
            target
                .layers
                .iter()
                .map(|l| [l.moe.hits, l.moe.misses, l.moe.read_bytes])
                .collect()
        }
        fn difference(after: &[[u64; 3]], before: &[[u64; 3]]) -> Vec<[u64; 3]> {
            after
                .iter()
                .zip(before)
                .map(|(a, b)| {
                    std::array::from_fn(|i| a[i].checked_sub(b[i]).expect("monotone I/O counter"))
                })
                .collect()
        }
        fn routes(target: &FlashNext) -> Result<Vec<Vec<u32>>> {
            target
                .layers
                .iter()
                .map(|layer| {
                    let ids = layer.moe.route_ids.clone();
                    ensure!(ids.len() == 10 && ids.iter().all(|&id| id < 512));
                    let mut unique = ids.clone();
                    unique.sort_unstable();
                    unique.dedup();
                    ensure!(unique.len() == 10);
                    Ok(ids)
                })
                .collect()
        }
        ensure!(std::env::var("MLXL3_SSD_CACHE_GIB")? == "3");
        ensure!(std::env::var("MLXL3_SSD_CACHE_POLICY")? == "auto");
        ensure!(
            std::env::var_os("MLXL3_SSD_LAYER_SLOTS").is_none(),
            "calibration trace requires uniform cache"
        );
        let output = std::env::var("MLXL3_FLASH_CACHE_TRACE")?;
        let marker = File::create_new(&output)?;
        serde_json::to_writer(marker, &serde_json::json!({"status":"incomplete"}))?;
        let path = std::env::var("MLXL3_FLASH_MODEL")?;
        let checkpoint = crate::checkpoint::inspect(Path::new(&path))?;
        let tokenizer = crate::tokenizer::ChatTokenizer::load(Path::new(&path))?;
        let prompt_ids = tokenizer.encode(&tokenizer.render(&[crate::tokenizer::Message {
            role: "user".into(),
            content: CALIBRATION_PROMPT.into(),
        }])?)?;
        ensure!((1..=512).contains(&prompt_ids.len()));
        let mut target = FlashNext::from_checkpoint(&checkpoint)?;
        ensure!(target.lookahead_depth == 0 && target.layers.len() == 48);
        let mut trace = CacheTrace {
            status: "incomplete".into(),
            slots: 53,
            prompt: CALIBRATION_PROMPT.into(),
            thinking: "default_on".into(),
            prompt_ids,
            phases: Vec::new(),
        };
        let mut first_state = None;
        let mut first_logits = None;
        for _ in 0..2 {
            target.reset();
            let before = counts(&target);
            let mut captured = Vec::new();
            let mut logits = None;
            for &token in &trace.prompt_ids {
                logits = Some(target.forward(token)?);
                captured.push(routes(&target)?);
            }
            let after_prefill = counts(&target);
            let mut next = logits
                .context("empty calibration prefill")?
                .chat_greedy_ids()?[0];
            let mut tokens = vec![next];
            for _ in 1..128 {
                next = target.forward(next)?.chat_greedy_ids()?[0];
                captured.push(routes(&target)?);
                tokens.push(next);
            }
            target.drain()?;
            ensure!(target.offset() as usize == trace.prompt_ids.len() + 127);
            let logits = target.logits(
                target
                    .last_hidden
                    .as_ref()
                    .context("missing calibration hidden")?,
            )?;
            ensure!(logits.shape() == [1, 248320]);
            let values = logits.to_f32()?;
            ensure!(values.len() == 248320 && values.iter().all(|v| v.is_finite()));
            let bytes = logits.to_bytes()?;
            let state = state_bytes(&target)?;
            ensure!(state.len() > 100 && state.iter().all(|buffer| !buffer.is_empty()));
            if let (Some(first), Some(first_values)) = (&first_state, &first_logits) {
                ensure!(
                    state == *first && bytes == *first_values,
                    "warm cache changed calibration states/logits"
                );
            } else {
                first_state = Some(state.clone());
                first_logits = Some(bytes);
            }
            trace.phases.push(TracePhase {
                routes: captured,
                text: tokenizer.decode(&tokens)?,
                tokens,
                prefill_counts: difference(&after_prefill, &before),
                decode_counts: difference(&counts(&target), &after_prefill),
                logit_count: values.len(),
                logits_finite: true,
                state_buffers: state.len(),
            });
        }
        ensure!(
            trace.phases[0].tokens == trace.phases[1].tokens
                && trace.phases[0].routes == trace.phases[1].routes
        );
        trace.status = "complete".into();
        trace.validate()?;
        let bridge: Value = serde_json::from_reader(File::open(std::env::var(
            "MLXL3_FLASH_CACHE_BRIDGE_REFERENCE",
        )?)?)?;
        ensure!(
            bridge["status"] == "validated" && bridge["tokens"] == 128 && bridge["repeats"] == 1
        );
        ensure!(
            bridge["mode"] == "baseline"
                && bridge["expert_cache_gib"] == 3
                && bridge["model"] == path
        );
        ensure!(bridge["prompts"]["CAL"] == CALIBRATION_PROMPT);
        for (phase, key) in trace.phases.iter().zip(["warmup", "runs"]) {
            let reference = bridge[key]
                .as_array()
                .context("missing calibration bridge runs")?;
            ensure!(reference.len() == 1 && reference[0]["stats"]["generated_tokens"] == 128);
            let hash = phase
                .tokens
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .fold(0xcbf29ce484222325u64, |h, byte| {
                    (h ^ u64::from(byte)).wrapping_mul(0x100000001b3)
                });
            ensure!(
                reference[0]["text"] == phase.text
                    && reference[0]["token_hash"] == format!("{hash:016x}"),
                "calibration differs from production NT1 bridge"
            );
        }
        std::fs::write(output, serde_json::to_vec_pretty(&trace)?)?;
        println!(
            "FLASH_CACHE_TRACE phases=2 tokens_per_phase=128 layers=48 shape=10 status=complete"
        );
        Ok(())
    }

    #[cfg(all(kani, feature = "chat"))]
    #[kani::proof]
    fn actual_flash_cache_capture_preserves_all_routes_and_states() {
        flash_cache_calibration_captures_prefill_decode_and_identical_states().unwrap();
    }

    #[cfg(feature = "chat")]
    #[test]
    #[ignore = "physical Apple GPU; experiment49D repeated MAIN cache/I/O diagnostic, not throughput"]
    fn flash_repeated_main_exposes_cache_and_timing_variance() -> Result<()> {
        use crate::flash_support::{EXPERT_BYTES, ExpertCache};
        use std::time::{Instant, SystemTime, UNIX_EPOCH};
        fn counts(target: &FlashNext) -> Vec<[u64; 3]> {
            target
                .layers
                .iter()
                .map(|l| [l.moe.hits, l.moe.misses, l.moe.read_bytes])
                .collect()
        }
        fn clocks(target: &FlashNext) -> [f64; 3] {
            let mut out = [0.; 3];
            for layer in &target.layers {
                out[0] += layer.moe.profile_seconds[0];
                out[1] += layer.moe.profile_seconds[1];
                out[2] += layer.moe.import_seconds;
            }
            out
        }
        fn difference(a: &[[u64; 3]], b: &[[u64; 3]]) -> Result<Vec<[u64; 3]>> {
            a.iter()
                .zip(b)
                .map(|(a, b)| {
                    Ok([
                        a[0].checked_sub(b[0]).context("backward hit count")?,
                        a[1].checked_sub(b[1]).context("backward miss count")?,
                        a[2].checked_sub(b[2]).context("backward byte count")?,
                    ])
                })
                .collect()
        }
        fn routes(target: &FlashNext) -> Result<Vec<Vec<u32>>> {
            target
                .layers
                .iter()
                .map(|layer| {
                    let ids = layer.moe.route_ids.clone();
                    ensure!(ids.len() == 10 && ids.iter().all(|&i| i < 512));
                    let unique = ids
                        .iter()
                        .copied()
                        .collect::<std::collections::BTreeSet<_>>();
                    ensure!(unique.len() == 10);
                    Ok(ids)
                })
                .collect()
        }
        ensure!(std::env::var("MLXL3_SSD_CACHE_GIB")? == "3");
        ensure!(std::env::var("MLXL3_SSD_RESIDENT_GATES")? == "0");
        ensure!(
            std::env::var("MLXL3_SSD_CACHE_POLICY")? == "auto"
                && std::env::var_os("MLXL3_SSD_LAYER_SLOTS").is_none()
        );
        let output = std::env::var("MLXL3_FLASH_VARIANCE_TRACE")?;
        serde_json::to_writer(
            File::create_new(&output)?,
            &serde_json::json!({"status":"incomplete"}),
        )?;
        let reference: Value = serde_json::from_reader(File::open(std::env::var(
            "MLXL3_FLASH_VARIANCE_REFERENCE",
        )?)?)?;
        ensure!(
            reference["status"] == "validated"
                && reference["mode"] == "baseline"
                && reference["tokens"] == 128
                && reference["repeats"] == 2
                && reference["expert_cache_gib"] == 3
        );
        let prompt = reference["prompts"]["MAIN"]
            .as_str()
            .context("missing MAIN prompt")?;
        let path = std::env::var("MLXL3_FLASH_MODEL")?;
        let c = crate::checkpoint::inspect(Path::new(&path))?;
        let tokenizer = crate::tokenizer::ChatTokenizer::load(Path::new(&path))?;
        let prompt_ids = tokenizer.encode(&tokenizer.render(&[crate::tokenizer::Message {
            role: "user".into(),
            content: prompt.into(),
        }])?)?;
        ensure!(!prompt_ids.is_empty());
        let mut target = FlashNext::from_checkpoint(&c)?;
        ensure!(
            target.lookahead_depth == 0
                && target.layers.iter().all(|l| l.moe.slots_for_test() == 53)
        );
        let mut phases = Vec::new();
        let mut first_state = None;
        let mut first_logits = None;
        let mut all_routes = Vec::new();
        let mut all_counts = Vec::new();
        for phase in 0..3 {
            target.reset();
            let before = counts(&target);
            let mut trace = Vec::new();
            let mut logits = None;
            for &token in &prompt_ids {
                logits = Some(target.forward(token)?);
                trace.push(routes(&target)?);
            }
            let pref = counts(&target);
            let before_clocks = clocks(&target);
            let mut anchor = logits.context("empty prefill")?.chat_greedy_ids()?[0];
            let mut tokens = vec![anchor];
            let mut steps = Vec::new();
            let start = Instant::now();
            for step in 1..128 {
                let c0 = counts(&target);
                let t0 = clocks(&target);
                let epoch = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs_f64();
                let token_start = Instant::now();
                anchor = target.forward(anchor)?.chat_greedy_ids()?[0];
                let seconds = token_start.elapsed().as_secs_f64();
                ensure!(seconds.is_finite() && seconds > 0.);
                let c1 = counts(&target);
                let t1 = clocks(&target);
                let delta = difference(&c1, &c0)?;
                let sum: [u64; 3] = std::array::from_fn(|i| delta.iter().map(|d| d[i]).sum());
                steps.push(serde_json::json!({"step":step,"epoch":epoch,"seconds":seconds,"hits_misses_bytes":sum,"route_fill_import_seconds":std::array::from_fn::<_,3,_>(|i| t1[i]-t0[i])}));
                tokens.push(anchor);
                trace.push(routes(&target)?);
            }
            target.drain()?;
            let seconds = start.elapsed().as_secs_f64();
            ensure!(tokens.len() == 128 && target.offset() as usize == prompt_ids.len() + 127);
            let hash = tokens
                .iter()
                .flat_map(|t| t.to_le_bytes())
                .fold(0xcbf29ce484222325u64, |h, b| {
                    (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
                });
            let text = tokenizer.decode(&tokens)?;
            let expected = if phase == 0 {
                &reference["warmup"][0]
            } else {
                &reference["runs"][phase - 1]
            };
            ensure!(
                !text.is_empty()
                    && expected["text"] == text
                    && expected["token_hash"] == format!("{hash:016x}")
            );
            let logits = target.logits(target.last_hidden.as_ref().context("missing hidden")?)?;
            ensure!(logits.shape() == [1, 248320]);
            let values = logits.to_f32()?;
            ensure!(values.len() == 248320 && values.iter().all(|v| v.is_finite()));
            let bytes = logits.to_bytes()?;
            let state = state_bytes(&target)?;
            ensure!(state.len() > 100 && state.iter().all(|v| !v.is_empty()));
            if let (Some(old_state), Some(old_logits)) = (&first_state, &first_logits) {
                ensure!(
                    state == *old_state && bytes == *old_logits,
                    "repeated cache changed states/logits"
                );
            } else {
                first_state = Some(state);
                first_logits = Some(bytes);
            }
            let after = counts(&target);
            let t1 = clocks(&target);
            let decode = difference(&after, &pref)?;
            let prefill = difference(&pref, &before)?;
            phases.push(serde_json::json!({"phase":phase,"tokens":tokens,"token_hash":format!("{hash:016x}"),"text":text,"prefill_counts":prefill,"decode_counts":decode,"phase_wall_seconds":seconds,"route_fill_import_seconds":std::array::from_fn::<_,3,_>(|i|t1[i]-before_clocks[i]),"steps":steps,"memory":crate::array::memory_stats(false)?}));
            all_counts.push([prefill, decode]);
            all_routes.push(trace);
        }
        ensure!(
            all_routes[0] == all_routes[1] && all_routes[1] == all_routes[2],
            "same tokens changed route trace"
        );
        let mut replay = (0..48)
            .map(|_| ExpertCache::new(53))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for phase in 0..3 {
            let mut expected = [vec![[0u64; 3]; 48], vec![[0u64; 3]; 48]];
            ensure!(all_routes[phase].len() == prompt_ids.len() + 127);
            for (step, layers) in all_routes[phase].iter().enumerate() {
                ensure!(layers.len() == 48);
                for (layer, ids) in layers.iter().enumerate() {
                    let routes = replay[layer].reserve(ids)?;
                    let misses = routes.iter().filter(|(_, hit)| !hit).count() as u64;
                    let stats = &mut expected[usize::from(step >= prompt_ids.len())][layer];
                    stats[0] += 10 - misses;
                    stats[1] += misses;
                    stats[2] += misses * EXPERT_BYTES;
                }
            }
            ensure!(expected == all_counts[phase], "CPU cache replay differs");
        }
        std::fs::write(
            output,
            serde_json::to_vec_pretty(
                &serde_json::json!({"status":"complete_diagnostic_not_release_throughput","prompt_ids":prompt_ids,"phases":phases,"routes":all_routes,"cpu_cache_replay_exact":true,"logit_count":248320,"states_logits_byte_exact":true,"limits":"inclusive route/fill/import clocks overlap; original release run lacks contemporaneous I/O counters"}),
            )?,
        )?;
        println!(
            "FLASH_VARIANCE_TRACE phases=3 tokens=128 routes=identical cpu_replay=exact states_logits=byte_exact status=complete"
        );
        Ok(())
    }

    #[cfg(all(kani, feature = "chat"))]
    #[kani::proof]
    fn actual_repeated_flash_profile_preserves_routes_and_states() {
        flash_repeated_main_exposes_cache_and_timing_variance().unwrap();
    }

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

#[cfg(all(test, feature = "chat"))]
mod profile;
