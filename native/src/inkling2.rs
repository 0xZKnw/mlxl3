//! Native Inkling 2 Mini (`inkling2_loop`) inference: EXL3 routed experts, dense FP16 projections,
//! FP32 residual stream, depth attention and loop messages.
//!
//! Reference: Transformers PR #51050 (`modeling_inkling2_loop.py` at 6173a2f6) and mlx-lm PR #2086.
//! The 24 decoder layers are visited 48 times per token (layers `core_start..core_end` run
//! `num_passes` times with shared weights); every visit owns its KV cache and convolution states.
use crate::{
    array::{Array, Dtype},
    checkpoint::{self, Checkpoint},
    lfm2::{Projection, half_weight},
    linear::checkpoint_array,
    moe::Exl3SwitchGlu,
    router,
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use std::{
    fs::File,
    path::Path,
    time::{Duration, Instant},
};

/// Accumulated `(x^T x, rows)` per `(layer, slot)`.
type HessianMap = std::collections::BTreeMap<(usize, u8), (Array, u64)>;

thread_local! {
    /// SPEED-2026-11-22-INK2M-P8: when set, `Attention::forward` adds x^T x (FP32) of the inputs of the dense
    /// projections per (layer, slot): 0 = hq (q, r, gate, sink), 1 = hk (k), 2 = hv (v), 3 = input of o.
    static DENSE_HESSIANS: std::cell::RefCell<Option<HessianMap>> = const { std::cell::RefCell::new(None) };
}

/// Routes of one MoE call: rows and the `[rows, top_k]` expert ids.
#[cfg(test)]
pub(crate) type MoeRoutes = Vec<(usize, Vec<u32>)>;

#[cfg(test)]
thread_local! {
    /// SPEED-2026-11-22-INK2M-P11 D3 (tests only): when set, every MoE call appends its routes `[rows, top_k]`.
    pub(crate) static MOE_ROUTES: std::cell::RefCell<Option<MoeRoutes>> = const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn capture_routes(selected: &Array) -> Result<()> {
    MOE_ROUTES.with(|cell| {
        if let Some(routes) = cell.borrow_mut().as_mut() {
            let rows = usize::try_from(selected.shape()[0])?;
            routes.push((rows, selected.to_u32()?));
        }
        Ok(())
    })
}

#[cfg(not(test))]
#[inline(always)]
fn capture_routes(_selected: &Array) -> Result<()> {
    Ok(())
}

/// INVARIANT-2026-11-24-INK2M-L1L4: verifications of 2..=16 rows compute every row as the 1-row path does, so a
/// verification row has the bits of the AR step (on by default; `MLXL3_INKLING_INVARIANT=0` restores the batched
/// products, whose rows differ from the AR steps by rounding).
pub(crate) fn invariant_rows() -> bool {
    #[cfg(test)]
    if let Some(value) = INVARIANT_TEST.with(std::cell::Cell::get) {
        return value;
    }
    static VALUE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| std::env::var("MLXL3_INKLING_INVARIANT").as_deref() != Ok("0"))
}

/// The rows of `x` (`[rows, ...]`) through `f` one at a time, concatenated: the 1-row arithmetic for each row.
fn per_row(x: &Array, f: impl Fn(&Array) -> Result<Array>) -> Result<Array> {
    let rows = x.shape()[0];
    let outs = (0..rows)
        .map(|r| f(&x.slice(0, r, r + 1)?))
        .collect::<Result<Vec<_>>>()?;
    Array::concatenate(&outs.iter().collect::<Vec<_>>(), 0)
}

thread_local! {
    /// Set while a verification (`run` without commit) builds its graph: the invariant rows apply only there, so
    /// the prefill and the AR steps keep their bits.
    static VERIFY_SCOPE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether a call of `rows` rows takes the invariant per-row path: a verification of 2..=16 rows with the option.
pub(crate) fn invariant_call(rows: i32) -> bool {
    (2..=16).contains(&rows) && VERIFY_SCOPE.with(std::cell::Cell::get) && invariant_rows()
}

/// The verification scope while it lives (the previous value restored on drop, also on error).
struct VerifyScope(bool);

impl VerifyScope {
    fn enter(verify: bool) -> Self {
        Self(VERIFY_SCOPE.with(|flag| flag.replace(verify)))
    }
}

impl Drop for VerifyScope {
    fn drop(&mut self) {
        VERIFY_SCOPE.with(|flag| flag.set(self.0));
    }
}

/// INVARIANT-2026-11-24-INK2M-L1L4 (L1): `x @ dequant(w)^T` for the rows of `x` (`[M, K]`, M <= 16) in one launch,
/// each row with exactly the arithmetic of MLX 0.32.2's 1-row affine product (`qmv_fast` when N % 8 == 0 and K is a
/// multiple of the fast block, `qmv` otherwise), the weights read once per block for all rows; `None` outside
/// that domain (other bits or dtypes, N < 8, K not a multiple of the group).
fn qmv_rows(
    x: &Array,
    w: &Array,
    scales: &Array,
    biases: &Array,
    group: i32,
    bits: i32,
) -> Result<Option<Array>> {
    let &[rows, k] = x.shape() else {
        return Ok(None);
    };
    let &[n, groups] = scales.shape() else {
        return Ok(None);
    };
    if !matches!(bits, 4 | 8)
        || !(1..=16).contains(&rows)
        || n < 8
        || groups * group != k
        || !matches!(group, 32 | 64 | 128)
        || scales.dtype() != Dtype::Float16
        || biases.dtype() != Dtype::Float16
        || biases.shape() != scales.shape()
        || w.dtype() != Dtype::UInt32
        || w.shape() != [n, k * bits / 32]
    {
        return Ok(None);
    }
    let fast = n % 8 == 0 && k % (if bits == 8 { 256 } else { 512 }) == 0;
    if !fast && k % (32 / bits * 32) != 0 && k < 32 / bits * 32 {
        return Ok(None);
    }
    let blocks = (n + 7) / 8;
    let header = format!(
        "#define BITS {bits}\n#define GROUP {group}\n#define IN_DIMS {k}\n#define OUT_DIMS {n}\n#define ROWS {rows}\n#define FAST {}\n",
        u8::from(fast)
    );
    let out = crate::array::metal_kernel(
        &format!(
            "mlxl3_inkling_qmv_rows_b{bits}_g{group}_k{k}_n{n}_r{rows}_f{}_v1",
            u8::from(fast)
        ),
        &["w", "scales", "biases", "x"],
        &["y"],
        &header,
        include_str!("../shaders/inkling_qmv_rows.metal"),
        &[w, scales, biases, &x.astype(Dtype::Float16)?],
        &[vec![rows, n]],
        &[Dtype::Float16],
        [64, blocks, 1],
        [64, 1, 1],
    )?
    .remove(0);
    Ok(Some(out))
}

#[cfg(test)]
thread_local! {
    static INVARIANT_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Runs `run` with the invariant rows on or off (tests).
#[cfg(test)]
pub(crate) fn with_invariant_rows<T>(enabled: bool, run: impl FnOnce() -> T) -> T {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            INVARIANT_TEST.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(INVARIANT_TEST.with(|flag| flag.replace(Some(enabled))));
    run()
}

#[cfg(test)]
thread_local! {
    /// INVARIANT-2026-11-24-INK2M-L1L4 (tests only): when set, the forward appends (stage, value) at every stage
    /// point, rows on axis 0.
    pub(crate) static STAGES: std::cell::RefCell<Option<Vec<(&'static str, Array)>>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(test)]
fn stage(label: &'static str, value: &Array) -> Result<()> {
    STAGES.with(|cell| {
        if let Some(stages) = cell.borrow_mut().as_mut() {
            stages.push((label, value.try_clone()?));
        }
        Ok(())
    })
}

#[cfg(not(test))]
#[inline(always)]
fn stage(_label: &'static str, _value: &Array) -> Result<()> {
    Ok(())
}

fn capture_dense_input(layer: usize, slot: u8, x: &Array) -> Result<()> {
    DENSE_HESSIANS.with(|cell| {
        let mut cell = cell.borrow_mut();
        let Some(map) = cell.as_mut() else {
            return Ok(());
        };
        let rows = x.shape()[0];
        let x = x.astype(Dtype::Float32)?.reshape(&[rows, -1])?;
        let xtx = x.transpose(&[1, 0])?.matmul(&x)?;
        let entry = match map.remove(&(layer, slot)) {
            Some((h, n)) => (h.add(&xtx)?, n + rows as u64),
            None => (xtx, rows as u64),
        };
        map.insert((layer, slot), entry);
        Ok(())
    })
}

/// Group size of the MLX affine copies made at load.
pub(crate) const AFFINE_GROUP: i32 = 64;

pub(crate) fn bits_from_env(name: &str, default: i32) -> Result<i32> {
    let bits = std::env::var(name)
        .ok()
        .map(|v| v.parse::<i32>())
        .transpose()?
        .unwrap_or(default);
    ensure!(matches!(bits, 4 | 8 | 16), "{name} must be 4, 8 or 16");
    Ok(bits)
}

/// Subset of the attention projections (`q`, `k`, `v`, `r`, `o`, `gate`, `sink`) as a bit set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DenseSet(u8);

impl DenseSet {
    const NAMES: [&'static str; 7] = ["q", "k", "v", "r", "o", "gate", "sink"];
    pub const ALL: Self = Self(0x7f);

    /// `q,o,gate` or `q+o+gate`.
    pub fn parse(list: &str) -> Result<Self> {
        let mut set = 0u8;
        for name in list.split([',', '+']).filter(|n| !n.is_empty()) {
            let index = Self::NAMES
                .iter()
                .position(|n| *n == name)
                .with_context(|| format!("unknown Inkling projection {name}"))?;
            set |= 1 << index;
        }
        ensure!(set != 0, "empty Inkling projection set");
        Ok(Self(set))
    }

    fn contains(self, name: &str) -> bool {
        Self::NAMES
            .iter()
            .position(|n| *n == name)
            .is_some_and(|i| self.0 & (1 << i) != 0)
    }
}

/// Run-time precision of the dense weights that are not EXL3: 16 keeps the checkpoint tensors; 8 or 4
/// replaces them by MLX affine copies made at load (lossy). Default: 8-bit group-64 attention projections
/// (SPEED-2026-11-22-INK2M-P8: no measurable KL change against the exact reference on 12 rows, about -19%
/// decode time on R3; group 32 was not faster to read and not measurably better) and an 8-bit group-64
/// head (SPEED-2026-11-21-INK2M-P3: KL 1e-5 against the FP16 head).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Precision {
    /// q, k, v, r, o, gate and sink projections of every attention.
    pub dense_bits: i32,
    /// Output head (unpadded vocabulary rows).
    pub head_bits: i32,
    /// Projections that take `dense_bits`; the others stay FP16.
    pub dense_only: DenseSet,
    /// Affine group of the dense copies: 32, 64 or 128 (the head always uses `AFFINE_GROUP`).
    pub dense_group: i32,
}

impl Precision {
    /// Every dense weight as stored in the checkpoint (FP16): the reference of the numerical checks.
    pub const FULL: Self = Self {
        dense_bits: 16,
        head_bits: 16,
        dense_only: DenseSet::ALL,
        dense_group: AFFINE_GROUP,
    };

    /// What generation uses unless the environment says otherwise.
    pub const DEFAULT: Self = Self {
        dense_bits: 8,
        head_bits: 8,
        dense_only: DenseSet::ALL,
        dense_group: AFFINE_GROUP,
    };

    /// `MLXL3_INKLING_DENSE_BITS` (default 8), `MLXL3_INKLING_HEAD_BITS` (default 8), each 4, 8 or 16, and
    /// `MLXL3_INKLING_DENSE_GROUP` (default 64): 32, 64 or 128.
    pub fn from_env() -> Result<Self> {
        Self::from_env_or(Self::DEFAULT)
    }

    /// The environment overrides `default` field by field.
    pub fn from_env_or(default: Self) -> Result<Self> {
        Ok(Self {
            dense_bits: bits_from_env("MLXL3_INKLING_DENSE_BITS", default.dense_bits)?,
            head_bits: bits_from_env("MLXL3_INKLING_HEAD_BITS", default.head_bits)?,
            dense_only: match std::env::var("MLXL3_INKLING_DENSE_ONLY") {
                Ok(list) => DenseSet::parse(&list)?,
                Err(_) => default.dense_only,
            },
            dense_group: match std::env::var("MLXL3_INKLING_DENSE_GROUP") {
                Ok(group) => group
                    .parse()
                    .context("MLXL3_INKLING_DENSE_GROUP must be 32, 64 or 128")?,
                Err(_) => default.dense_group,
            },
        })
    }

    fn validate(self) -> Result<Self> {
        ensure!(
            matches!(self.dense_bits, 4 | 8 | 16) && matches!(self.head_bits, 4 | 8 | 16),
            "Inkling precision bits must be 4, 8 or 16"
        );
        ensure!(
            matches!(self.dense_group, 32 | 64 | 128),
            "Inkling dense group must be 32, 64 or 128"
        );
        Ok(self)
    }
}

/// A dense projection: the checkpoint FP16 tensor, or an MLX affine copy of it.
pub(crate) enum Linear {
    Dense(Projection),
    Affine {
        w: Array,
        scales: Array,
        biases: Array,
        bits: i32,
        group: i32,
    },
}

impl Linear {
    pub(crate) fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        input: i32,
        output: i32,
        bits: i32,
    ) -> Result<Self> {
        if let Some(stored) = Self::stored(checkpoint, prefix, input, output)? {
            ensure!(
                stored.bits() == bits,
                "{prefix} is stored as {}-bit affine weights; {bits}-bit weights were requested",
                stored.bits()
            );
            return Ok(stored);
        }
        let dense = Projection::load(checkpoint, prefix, input, output, false)?;
        if bits >= 16 {
            return Ok(Self::Dense(dense));
        }
        Self::quantized(dense.dense_weight().context("dense weight")?, bits)
    }

    /// The MLX affine copy stored in the checkpoint as `{prefix}.weight` (U32 `[output, input * bits / 32]`)
    /// with `{prefix}.scales` and `{prefix}.biases` (F16 `[output, input / 64]`), exactly what `quantized`
    /// produces at load; `None` when the checkpoint has no `{prefix}.scales` (dense weight).
    pub(crate) fn stored(
        checkpoint: &Checkpoint,
        prefix: &str,
        input: i32,
        output: i32,
    ) -> Result<Option<Self>> {
        if !checkpoint.tensors.contains_key(&format!("{prefix}.scales")) {
            return Ok(None);
        }
        ensure!(
            input > 0 && output > 0 && input % AFFINE_GROUP == 0,
            "{prefix}: invalid affine dimensions {output}x{input}"
        );
        let w = checkpoint_array(checkpoint, &format!("{prefix}.weight"))?;
        let scales = checkpoint_array(checkpoint, &format!("{prefix}.scales"))?;
        let biases = checkpoint_array(checkpoint, &format!("{prefix}.biases"))?;
        let packed = if w.shape().len() == 2 {
            w.shape()[1]
        } else {
            0
        };
        let bits = [4, 8]
            .into_iter()
            .find(|&b| i64::from(packed) * 32 == i64::from(b) * i64::from(input))
            .with_context(|| format!("{prefix}: packed width {packed} is not 4 or 8 bits"))?;
        let groups = [output, input / AFFINE_GROUP];
        ensure!(
            w.dtype() == Dtype::UInt32
                && w.shape()[0] == output
                && scales.dtype() == Dtype::Float16
                && biases.dtype() == Dtype::Float16
                && scales.shape() == groups
                && biases.shape() == groups,
            "{prefix}: invalid stored affine weights"
        );
        Array::eval_many(&[&w, &scales, &biases])?;
        Ok(Some(Self::Affine {
            w,
            scales,
            biases,
            bits,
            group: AFFINE_GROUP,
        }))
    }

    /// 16 for the dense FP16 weight, otherwise the affine width.
    pub(crate) fn bits(&self) -> i32 {
        match self {
            Self::Dense(_) => 16,
            Self::Affine { bits, .. } => *bits,
        }
    }

    /// `weight`: `[output, input]`, `input` a multiple of the group size.
    pub(crate) fn quantized(weight: &Array, bits: i32) -> Result<Self> {
        Self::quantized_group(weight, bits, AFFINE_GROUP)
    }

    pub(crate) fn quantized_group(weight: &Array, bits: i32, group: i32) -> Result<Self> {
        let shape = weight.shape();
        ensure!(
            shape.len() == 2
                && matches!(group, 32 | 64 | 128)
                && shape[1] % group == 0
                && matches!(bits, 4 | 8),
            "unsupported affine projection {shape:?} at {bits} bits, group {group}"
        );
        let (w, scales, biases) = weight
            .astype(Dtype::Float16)?
            .quantize_affine(group, bits)?;
        Array::eval_many(&[&w, &scales, &biases])?;
        Ok(Self::Affine {
            w,
            scales,
            biases,
            bits,
            group,
        })
    }

    /// SPEED-2026-11-22-INK2M-P8: precomputed MLX affine 8-bit tensors (`{prefix}.weight` packed u32,
    /// `{prefix}.scales` / `{prefix}.biases` FP16, `mx.quantize` layout) instead of a copy made at load.
    fn from_sidecar(sidecar: &Checkpoint, prefix: &str, input: i32, output: i32) -> Result<Self> {
        let w = checkpoint_array(sidecar, &format!("{prefix}.weight"))?;
        let scales = checkpoint_array(sidecar, &format!("{prefix}.scales"))?;
        let biases = checkpoint_array(sidecar, &format!("{prefix}.biases"))?;
        let groups = scales.shape().get(1).copied().unwrap_or(0);
        ensure!(
            groups > 0 && input % groups == 0,
            "{prefix}: side-car scales {:?} do not divide {input} inputs",
            scales.shape()
        );
        let group = input / groups;
        ensure!(
            matches!(group, 32 | 64 | 128)
                && w.dtype() == Dtype::UInt32
                && w.shape() == [output, input / 4]
                && scales.dtype() == Dtype::Float16
                && biases.dtype() == Dtype::Float16
                && scales.shape() == [output, groups]
                && biases.shape() == [output, groups],
            "{prefix}: side-car tensors {:?} {:?} {:?} are not 8-bit affine [{output}, {input}]",
            w.shape(),
            scales.shape(),
            biases.shape()
        );
        Array::eval_many(&[&w, &scales, &biases])?;
        Ok(Self::Affine {
            w,
            scales,
            biases,
            bits: 8,
            group,
        })
    }

    pub(crate) fn forward(&self, x: &Array) -> Result<Array> {
        if ablated("dense")
            && let Self::Dense(p) = self
        {
            // same output shape from one row of the weight: no full read of the matrix
            let w = p.dense_weight().context("dense weight")?;
            let rows = x.shape()[0];
            return x
                .slice(1, 0, 1)?
                .astype(Dtype::Float16)?
                .matmul(&w.slice(1, 0, 1)?.reshape(&[1, w.shape()[0]])?)?
                .reshape(&[rows, w.shape()[0]]);
        }
        if x.shape().len() == 2 && invariant_call(x.shape()[0]) {
            if let Self::Affine {
                w,
                scales,
                biases,
                bits,
                group,
            } = self
                && let Some(out) = qmv_rows(x, w, scales, biases, *group, *bits)?
            {
                return Ok(out);
            }
            return per_row(x, |row| self.forward_rows(row));
        }
        self.forward_rows(x)
    }

    fn forward_rows(&self, x: &Array) -> Result<Array> {
        match self {
            Self::Dense(p) => p.forward(x),
            Self::Affine {
                w,
                scales,
                biases,
                bits,
                group,
            } => x
                .astype(Dtype::Float16)?
                .quantized_matmul(w, scales, biases, *group, *bits),
        }
    }
}

/// `MLXL3_INKLING_EMBED_BITS`, if set (4, 8 or 16).
fn embed_bits_requested() -> Result<Option<i32>> {
    match std::env::var("MLXL3_INKLING_EMBED_BITS") {
        Ok(_) => Ok(Some(bits_from_env("MLXL3_INKLING_EMBED_BITS", 16)?)),
        Err(_) => Ok(None),
    }
}

/// Token embedding table `[vocab, D]`: the checkpoint FP16 table, or an MLX affine copy stored in the checkpoint
/// (SIZE-2026-11-22-INK2M-EMBED8) whose requested rows are dequantized on lookup.
pub(crate) enum Embeddings {
    Dense(Array),
    Affine {
        w: Array,
        scales: Array,
        biases: Array,
        bits: i32,
    },
}

impl Embeddings {
    /// `requested`: `MLXL3_INKLING_EMBED_BITS`. On a stored copy it must be absent or equal to the stored width.
    /// On a dense table, 4 or 8 rounds it at load (evaluation option of SIZE-2026-11-21-INK2M-SLIM part B,
    /// element for element the table a stored copy gives); absent or 16 keeps it.
    pub(crate) fn load(
        checkpoint: &Checkpoint,
        vocab: i32,
        hidden: i32,
        requested: Option<i32>,
    ) -> Result<Self> {
        if let Some(Linear::Affine {
            w,
            scales,
            biases,
            bits,
            group: AFFINE_GROUP,
        }) = Linear::stored(checkpoint, "model.llm.embed", hidden, vocab)?
        {
            ensure!(
                requested.is_none_or(|r| r == bits),
                "this checkpoint stores {bits}-bit embeddings; MLXL3_INKLING_EMBED_BITS={} is unavailable",
                requested.unwrap_or(16)
            );
            return Ok(Self::Affine {
                w,
                scales,
                biases,
                bits,
            });
        }
        let table = half_weight(checkpoint, "model.llm.embed.weight", Some(&[vocab, hidden]))?;
        match requested {
            Some(bits) if bits < 16 => {
                let (w, scales, biases) = table.quantize_affine(AFFINE_GROUP, bits)?;
                let rounded = Array::dequantize_affine(&w, &scales, &biases, AFFINE_GROUP, bits)?;
                rounded.eval()?;
                Ok(Self::Dense(rounded))
            }
            _ => Ok(Self::Dense(table)),
        }
    }

    /// FP16 rows `[n, D]` of the token ids.
    pub(crate) fn rows(&self, ids: &Array) -> Result<Array> {
        match self {
            Self::Dense(table) => table.take(ids, 0),
            Self::Affine {
                w,
                scales,
                biases,
                bits,
            } => Array::dequantize_affine(
                &w.take(ids, 0)?,
                &scales.take(ids, 0)?,
                &biases.take(ids, 0)?,
                AFFINE_GROUP,
                *bits,
            ),
        }
    }
}

#[derive(Deserialize)]
struct RootConfig {
    model_type: String,
    text_config: TextConfig,
}

#[derive(Deserialize)]
struct LoopConfig {
    core_start: usize,
    core_end: usize,
    num_passes: usize,
    gate_rank: i32,
    #[serde(default)]
    is_varloop: bool,
}

#[derive(Deserialize)]
struct TextConfig {
    /// Affine bits of the dense attention projections (16 = FP16), set from `Precision` at load.
    #[serde(skip)]
    dense_bits: i32,
    /// Names of the attention projections that take `dense_bits` (the others stay FP16).
    #[serde(skip)]
    dense_only: DenseSet,
    /// Affine group of the dense copies (`MLXL3_INKLING_DENSE_GROUP`, default 64; P8 option).
    #[serde(skip)]
    dense_group: i32,
    /// Precomputed 8-bit dense tensors (`MLXL3_INKLING_DENSE_SIDECAR`; P8 option).
    #[serde(skip)]
    dense_sidecar: Option<std::sync::Arc<Checkpoint>>,
    hidden_size: i32,
    num_hidden_layers: usize,
    vocab_size: i32,
    unpadded_vocab_size: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    swa_head_dim: i32,
    swa_num_attention_heads: i32,
    swa_num_key_value_heads: i32,
    d_rel: i32,
    rel_extent: i32,
    log_scaling_n_floor: f32,
    log_scaling_alpha: f32,
    rms_norm_eps: f32,
    qk_l2_clip_threshold: f32,
    rel_l2_clip_threshold: f32,
    #[serde(default)]
    local_layer_ids: Vec<usize>,
    /// Transformers configs: `hybrid_sliding` / `hybrid` per layer; takes precedence over `local_layer_ids`.
    #[serde(default)]
    layer_types: Option<Vec<String>>,
    #[serde(alias = "conv_kernel_size")]
    sconv_kernel_size: i32,
    final_logit_softcapping: f32,
    sliding_window_size: i32,
    n_layers_per_attn_res_block: usize,
    n_routed_experts: i32,
    num_experts_per_tok: usize,
    n_shared_experts: i32,
    #[serde(alias = "moe_intermediate_size")]
    intermediate_size: i32,
    mlp_gate_softcap: f32,
    mlp_up_softcap: f32,
    route_scale: f32,
    use_gate_bias: bool,
    #[serde(alias = "max_position_embeddings")]
    model_max_length: i32,
    loop_config: LoopConfig,
}

impl TextConfig {
    fn is_sliding(&self, layer: usize) -> Result<bool> {
        match &self.layer_types {
            Some(types) => match types.get(layer).map(String::as_str) {
                Some("hybrid_sliding") => Ok(true),
                Some("hybrid") => Ok(false),
                other => anyhow::bail!("unsupported Inkling layer type {other:?} at layer {layer}"),
            },
            None => Ok(self.local_layer_ids.contains(&layer)),
        }
    }
}

const L2_EPS: f32 = 1e-6;

fn f32_weight(checkpoint: &Checkpoint, name: &str, shape: &[i32]) -> Result<Array> {
    let weight = checkpoint_array(checkpoint, name)?;
    ensure!(
        matches!(
            weight.dtype(),
            Dtype::Float16 | Dtype::Float32 | Dtype::BFloat16
        ),
        "{name}: expected floating-point weights"
    );
    ensure!(
        weight.shape() == shape,
        "{name}: shape {:?}, expected {shape:?}",
        weight.shape()
    );
    let weight = weight.astype(Dtype::Float32)?;
    weight.eval()?;
    Ok(weight)
}

/// Dense FP16 projection stored as `[output, input]` under an arbitrary tensor name.
struct Dense {
    /// `[input, output]` FP16.
    weight_t: Array,
}

impl Dense {
    fn load(checkpoint: &Checkpoint, name: &str, output: i32, input: i32) -> Result<Self> {
        let weight_t = half_weight(checkpoint, name, Some(&[output, input]))?.transpose(&[1, 0])?;
        weight_t.eval()?;
        Ok(Self { weight_t })
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        // INVARIANT-2026-11-24-INK2M-L1L4: MLX's matmul changes kernels with the row count
        if x.shape().len() == 2 && invariant_call(x.shape()[0]) {
            return per_row(x, |row| row.astype(Dtype::Float16)?.matmul(&self.weight_t));
        }
        x.astype(Dtype::Float16)?.matmul(&self.weight_t)
    }
}

fn scalar(value: f32) -> Result<Array> {
    Array::from_f32(&[value], &[1])
}

/// Compiled (fused) subgraphs, bit-identical to the unfused op sequences; `MLXL3_INKLING_UNFUSED=1` keeps the
/// unfused path for A/B checks.
fn fused() -> bool {
    static FUSED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FUSED.get_or_init(|| std::env::var_os("MLXL3_INKLING_UNFUSED").is_none())
}

/// Graph simplifications and single-kernel steps of SPEED-2026-11-21-INK2M-P4 (same bits as the plain op
/// sequences): on with the fused path, `MLXL3_INKLING_P4_OFF=1` turns only them off for A/B checks.
fn lean() -> bool {
    static LEAN: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LEAN.get_or_init(|| fused() && std::env::var_os("MLXL3_INKLING_P4_OFF").is_none())
}

/// Timing-only ablations for the test harness (`MLXL3_INKLING_ABLATE=experts,head,...`): the named part is
/// replaced by a trivial computation of the same shape so its cost can be read as a time difference. The outputs
/// are meaningless. Compiled out of non-test builds.
#[cfg(test)]
fn ablated(part: &str) -> bool {
    static PARTS: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    PARTS
        .get_or_init(|| {
            std::env::var("MLXL3_INKLING_ABLATE")
                .map(|v| v.split(',').map(str::to_string).collect())
                .unwrap_or_default()
        })
        .iter()
        .any(|p| p == part)
}

#[cfg(not(test))]
#[inline(always)]
fn ablated(_part: &str) -> bool {
    false
}

/// `(a + b) * scale`
fn add_scaled(a: &Array, b: &Array, scale: f32) -> Result<Array> {
    if fused() {
        a.add_scaled(b, scale)
    } else {
        a.add(b)?.scalar_mul(scale)
    }
}

/// `x * min(threshold / (|x| + 1e-6), 1)` over the last axis, after an optional per-element scale.
fn l2_clip(x: &Array, scale: Option<&Array>, threshold: f32) -> Result<Array> {
    if fused() {
        return match scale {
            Some(scale) => x.l2_clip_scaled(scale, threshold),
            None => x.l2_clip(threshold),
        };
    }
    let x = x.astype(Dtype::Float32)?;
    let x = match scale {
        Some(scale) => x.mul(scale)?,
        None => x,
    };
    let last = x.shape().len() as i32 - 1;
    let norm = x.mul(&x)?.sum(last, true)?.sqrt()?.scalar_add(L2_EPS)?;
    x.mul(&scalar(threshold)?.div(&norm)?.clamp_max(1.0)?)
}

/// Attention of learned queries over the depth sources of each token.
/// `sources`: N arrays `[T, D]` (FP32, already RMS-normalized, also the values); `queries`: `[Q, D]`.
/// Returns `[T, Q, D]` FP32.
fn depth_attention(sources: &[&Array], queries: &Array) -> Result<Array> {
    if let Some(out) = depth_attend_one(sources, queries)? {
        return Ok(out);
    }
    depth_attend(&depth_stack(sources)?, queries)
}

/// Fusion `step` of SPEED-2026-11-22-INK2M-P17 (same bits as the op sequence it replaces): on with the lean
/// path; `MLXL3_INKLING_P17_OFF=1` turns all of them off, `MLXL3_INKLING_P17_OFF=f1,f2` the listed ones.
fn p17(step: &str) -> bool {
    #[cfg(test)]
    if let Some(off) = P17_TEST_OFF.with(|c| c.borrow().clone()) {
        return lean() && !off.split(',').any(|s| s == step || s == "all");
    }
    static OFF: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let off = OFF.get_or_init(|| std::env::var("MLXL3_INKLING_P17_OFF").ok());
    lean()
        && match off.as_deref() {
            None => true,
            Some("1") | Some("all") => false,
            Some(list) => !list.split(',').any(|s| s == step),
        }
}

#[cfg(test)]
thread_local! {
    /// Mutation for the component test: the score reduction tree runs in the opposite order.
    static DEPTH_ONE_MUTATE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Mutation for the gate component test: the tail's reduction tree runs in the opposite order.
    static GATE_TAIL_MUTATE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Mutation for the router tail component test: the score sum runs in the opposite order.
    static ROUTER_TAIL_MUTATE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// Paired A/B timing (`fusion_ab_profile`): P17 steps turned off on this thread (`Some("")` = none off).
    static P17_TEST_OFF: std::cell::RefCell<Option<String>> = const { std::cell::RefCell::new(None) };
}

/// INVARIANT-2026-11-24-INK2M-L1L4 (L2): a 1-row kernel source applied to row `mrow = threadgroup_position_in_grid.y`
/// of `[rows, ...]` buffers by offsetting its buffer indices only (`edits`: (from, to) text replacements), so
/// every row runs the exact operations of the 1-row kernel.
fn rows_source(source: &str, edits: &[(&str, &str)]) -> Result<String> {
    let mut out = format!("    const int mrow = int(threadgroup_position_in_grid.y);\n{source}");
    for (from, to) in edits {
        ensure!(out.contains(from), "row edit `{from}` does not apply");
        out = out.replace(from, to);
    }
    Ok(out)
}

impl Moe {
    /// L2: router logits, selected experts and scores of every row of `x` (`[rows, hidden]` F16) with the 1-row
    /// kernels (F3 logits, RT tail) launched once over the rows; `None` outside their domain.
    fn route_rows(&self, x: &Array) -> Result<Option<(Array, Array, Array)>> {
        let Some(w) = &self.router_bf16 else {
            return Ok(None);
        };
        if !p17("f3") || !p17("rt") {
            return Ok(None);
        }
        let &[experts, hid] = w.shape() else {
            return Ok(None);
        };
        let &[rows, xh] = x.shape() else {
            return Ok(None);
        };
        if xh != hid
            || x.dtype() != Dtype::Float16
            || experts != 128
            || hid <= 64
            || hid < 16 * experts
            || self.bias.shape() != [128]
            || self.bias.dtype() != Dtype::Float32
            || self.top_k != 8
            || !(self.scale.is_finite() && self.scale > 0.0)
        {
            return Ok(None);
        }
        let key = [
            i64::from(experts),
            i64::from(hid),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let logits_kernel =
            crate::array::prepared_kernel("inkling_router_logits_rows", key, || {
                let source = rows_source(
                    &ROUTER_SOURCE.replace("HID", &hid.to_string()),
                    &[
                        ("x[bn + t]", &format!("x[mrow * {hid} + bn + t]")),
                        (
                            "out[out_row + tm]",
                            &format!("out[mrow * {experts} + out_row + tm]"),
                        ),
                    ],
                )
                .expect("router rows source");
                crate::array::PreparedKernel::new(
                    &format!("inkling_router_logits_rows_{experts}_{hid}"),
                    &["x", "w"],
                    &["out"],
                    "",
                    &source,
                )
            })?;
        let logits = logits_kernel
            .run(
                &[x, w],
                &[&[rows, experts]],
                &[Dtype::Float32],
                [256 * (experts / 4), rows, 1],
                [256, 1, 1],
            )?
            .pop()
            .context("router logits rows")?;
        let bits = self.scale.to_bits();
        let tail_key = [i64::from(bits), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let tail_kernel =
            crate::array::prepared_kernel("inkling_router_tail_rows", tail_key, || {
                let source = rows_source(
                    &ROUTER_TAIL_SOURCE
                        .replace(
                            "SUM_ORDER",
                            "for (uint r = 0u; r < TOP_K; ++r) { total = chosen[r] + total; }",
                        )
                        .replace("EXPERTS", "128u")
                        .replace("TOP_K", "8u")
                        .replace("SCALE", &format!("as_type<float>({bits}u)")),
                    &[
                        (
                            "float x = logits[tid];",
                            "float x = logits[mrow * 128 + tid];",
                        ),
                        (
                            "indices[rank] = winner;",
                            "indices[mrow * 8 + rank] = winner;",
                        ),
                        ("scores[rank] =", "scores[mrow * 8 + rank] ="),
                    ],
                )
                .expect("router tail rows source");
                crate::array::PreparedKernel::new(
                    &format!("inkling_router_tail_rows_{bits:08x}"),
                    &["logits", "bias"],
                    &["indices", "scores"],
                    "",
                    &source,
                )
            })?;
        let mut out = tail_kernel.run(
            &[&logits, &self.bias],
            &[&[rows, 8], &[rows, 8]],
            &[Dtype::UInt32, Dtype::Float32],
            [128, rows, 1],
            [128, 1, 1],
        )?;
        let scores = out.pop().context("router tail rows scores")?;
        let selected = out.pop().context("router tail rows indices")?;
        Ok(Some((logits, selected, scores)))
    }
}

impl MessageGate {
    /// L2: the F2 gate of every row (`[rows, hid]` F32 inputs) with the two 1-row kernels launched once over the
    /// rows; `None` outside their domain.
    fn forward_fused_rows(
        &self,
        hidden: &Array,
        hidden_norm: &Array,
        message_norm: &Array,
        message: &Array,
    ) -> Result<Option<Array>> {
        let (Some(forget), Some(gain)) = (&self.forget_bf16, &self.gain_bf16) else {
            return Ok(None);
        };
        if !p17("f2") || !fused() {
            return Ok(None);
        }
        let &[hid, rank] = forget.shape() else {
            return Ok(None);
        };
        let rows = hidden.shape()[0];
        let f32_rows = |a: &Array| a.shape() == [rows, hid] && a.dtype() == Dtype::Float32;
        let down = &self.down.weight_t;
        if rank != 64
            || hid % 32 != 0
            || 2 * hid < 16 * rank
            || !(f32_rows(hidden)
                && f32_rows(hidden_norm)
                && f32_rows(message_norm)
                && f32_rows(message))
            || down.shape() != [2 * hid, rank]
            || down.dtype() != Dtype::Float16
        {
            return Ok(None);
        }
        let key = [
            i64::from(hid),
            i64::from(rank),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let down_kernel = crate::array::prepared_kernel("inkling_gate_down_rows", key, || {
            let source = rows_source(
                GATE_DOWN_SOURCE,
                &[
                    (
                        "k < HID ? hn[k] : mn[k - HID]",
                        "k < HID ? hn[mrow * HID + k] : mn[mrow * HID + k - HID]",
                    ),
                    (
                        "out[out_row + tm] =",
                        &format!("out[mrow * {rank} + out_row + tm] ="),
                    ),
                ],
            )
            .map(|source| source.replace("HID", &hid.to_string()))
            .expect("gate down rows source");
            crate::array::PreparedKernel::new(
                &format!("inkling_gate_down_rows_{hid}_{rank}"),
                &["hn", "mn", "w"],
                &["out"],
                "",
                &source,
            )
        })?;
        let weights = down.transpose(&[1, 0])?;
        let states = down_kernel
            .run(
                &[hidden_norm, message_norm, &weights],
                &[&[rows, rank]],
                &[Dtype::Float32],
                [256 * (rank / 4), rows, 1],
                [256, 1, 1],
            )?
            .pop()
            .context("gate states rows")?;
        let tail_kernel = crate::array::prepared_kernel("inkling_gate_tail_rows", key, || {
            let source = rows_source(
                &GATE_TAIL_SOURCE.replace("RANK", &rank.to_string()),
                &[
                    (
                        "v[t] = s[bn + t];",
                        &format!("v[t] = s[mrow * {rank} + bn + t];"),
                    ),
                    (
                        "const float hv = h[row];",
                        &format!("const float hv = h[mrow * {hid} + row];"),
                    ),
                    (
                        "const float c = tg * m[row];",
                        &format!("const float c = tg * m[mrow * {hid} + row];"),
                    ),
                    ("out[row] = o;", &format!("out[mrow * {hid} + row] = o;")),
                ],
            )
            .expect("gate tail rows source");
            crate::array::PreparedKernel::new(
                &format!("inkling_gate_tail_rows_{hid}_{rank}"),
                &["s", "f", "g", "h", "m"],
                &["out"],
                "",
                &source,
            )
        })?;
        let out = tail_kernel
            .run(
                &[&states, forget, gain, hidden, message],
                &[&[rows, hid]],
                &[Dtype::Float32],
                [32 * (hid / 32), rows, 1],
                [32, 1, 1],
            )?
            .pop()
            .context("gate output rows")?;
        Ok(Some(out))
    }
}

/// SPEED-INK2M-P17 F1: one token, one query (`[1, D]` FP32) over `N` sources (`[1, D]` FP32) in one kernel.
/// It repeats MLX 0.32.2's kernels for `depth_attend` at these shapes, thread layout and operation order
/// included: scores `gemv_float32_bm1_bn8_sm1_sn32_tm*_tn4`, the scale `multiply`, `block_softmax_float32`,
/// output `gemv_t_float32_bm1_bn16_sm8_sn4_tm4_tn4`. Threadgroups of 8 simdgroups, one per 256 output columns.
const DEPTH_ONE_SOURCE: &str = r#"
    const int lane = int(thread_index_in_simdgroup);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int group = int(threadgroup_position_in_grid.x);
    const device float* src[NSRC] = { SRCLIST };
    threadgroup float w[NSRC];
    // scores: one simdgroup per source row; each lane keeps the partial sums of the 8 simdgroups of
    // gemv (bn 8, sn 32, tn 4) in the same order, then the same trees and the same final sum
    for (int r = sg; r < NSRC; r += 8) {
        float part[8];
        for (int g = 0; g < 8; ++g) {
            float result = 0.0f;
            int bn = (32 * g + lane) * 4;
            for (int i = 0; i < DIMS / 1024; ++i) {
                float v[4];
                float m[4];
                for (int t = 0; t < 4; ++t) { v[t] = q[bn + t]; }
                for (int t = 0; t < 4; ++t) { m[t] = src[r][bn + t]; }
                for (int t = 0; t < 4; ++t) { result += m[t] * v[t]; }
                bn += 1024;
            }
            if (DIMS % 1024 > 0) {
                float v[4];
                float m[4];
                if (bn + 4 <= DIMS) {
                    for (int t = 0; t < 4; ++t) { v[t] = q[bn + t]; }
                } else {
                    for (int t = 0; t < 4; ++t) { v[t] = bn + t < DIMS ? q[bn + t] : 0.0f; }
                }
                if (bn + 4 <= DIMS) {
                    for (int t = 0; t < 4; ++t) { m[t] = src[r][bn + t]; }
                } else {
                    for (int t = 0; t < 4; ++t) { m[t] = bn + t < DIMS ? src[r][bn + t] : 0.0f; }
                }
                for (int t = 0; t < 4; ++t) { result += m[t] * v[t]; }
            }
            SHUFFLE_TREE
            part[g] = result;
        }
        if (lane == 0) {
            float acc = part[0];
            for (int g = 1; g < 8; ++g) { acc += part[g]; }
            w[r] = acc * as_type<float>(SCALEBITSu);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // softmax: block_softmax_float32, one simdgroup, 4 values per lane
    if (sg == 0) {
        const int base = lane * 4;
        float ld[4];
        if (base + 4 <= NSRC) {
            for (int i = 0; i < 4; ++i) { ld[i] = w[base + i]; }
        } else {
            for (int i = 0; i < 4; ++i) { ld[i] = (base + i) < NSRC ? w[base + i] : -INFINITY; }
        }
        float maxval = -FLT_MAX;
        for (int i = 0; i < 4; ++i) { maxval = (maxval < ld[i]) ? ld[i] : maxval; }
        maxval = simd_max(maxval);
        maxval = simd_max(lane == 0 ? maxval : -INFINITY);
        float normalizer = 0.0f;
        for (int i = 0; i < 4; ++i) {
            float exp_x = metal::fast::exp(ld[i] - maxval);
            ld[i] = exp_x;
            normalizer += exp_x;
        }
        normalizer = simd_sum(normalizer);
        normalizer = simd_sum(lane == 0 ? normalizer : 0.0f);
        normalizer = 1 / normalizer;
        if (base + 4 <= NSRC) {
            for (int i = 0; i < 4; ++i) { w[base + i] = ld[i] * normalizer; }
        } else {
            for (int i = 0; i < 4; ++i) {
                if ((base + i) < NSRC) { w[base + i] = ld[i] * normalizer; }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // output: gemv_t (bn 16, sm 8, sn 4, tm 4, tn 4), 256 columns per threadgroup in two passes
    const int thr_m = lane / 4;
    const int thr_n = lane % 4;
    const int bm = thr_m * 4;
    for (int pass = 0; pass < 2; ++pass) {
        const int col = group * 256 + pass * 128 + (4 * sg + thr_n) * 4;
        float res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (col < DIMS) {
            for (int tm = 0; tm < 4 && bm + tm < NSRC; ++tm) {
                float vc = w[bm + tm];
                float inter[4];
                for (int tn = 0; tn < 4; ++tn) { inter[tn] = src[bm + tm][col + tn]; }
                for (int tn = 0; tn < 4; ++tn) { res[tn] += vc * inter[tn]; }
            }
        }
        for (int tn = 0; tn < 4; ++tn) {
            for (ushort sm = 4; sm >= 1; sm >>= 1) { res[tn] += simd_shuffle_down(res[tn], ushort(4 * sm)); }
        }
        if (thr_m == 0 && col < DIMS) {
            for (int tn = 0; tn < 4; ++tn) { out[col + tn] = res[tn]; }
        }
    }
"#;

/// `depth_attend` of one token and one query in one kernel, or `None` outside that domain (several rows or
/// queries, 32 sources or more, D below 16 per source or 64, another dtype, the timing ablation).
fn depth_attend_one(sources: &[&Array], queries: &Array) -> Result<Option<Array>> {
    if !p17("f1") || ablated("depth") {
        return Ok(None);
    }
    let count = sources.len();
    let &[1, dims] = queries.shape() else {
        return Ok(None);
    };
    // MLX picks the bn 8 score gemv only when D > 64 and D >= 16 * sources
    if count == 0
        || count >= 32
        || dims <= 64
        || dims < 16 * count as i32
        || dims % 4 != 0
        || queries.dtype() != Dtype::Float32
        || sources
            .iter()
            .any(|s| s.shape() != [1, dims] || s.dtype() != Dtype::Float32)
    {
        return Ok(None);
    }
    #[cfg(test)]
    let mutate = DEPTH_ONE_MUTATE.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let mutate = false;
    let scale = (dims as f32).powf(-0.5);
    let names: Vec<String> = (0..count)
        .map(|i| format!("s{i}"))
        .chain(["q".to_string()])
        .collect();
    let key = [
        count as i64,
        i64::from(dims),
        i64::from(mutate),
        i64::from(scale.to_bits()),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    let kernel = crate::array::prepared_kernel("inkling_depth_one", key, || {
        let tree = if mutate {
            "for (ushort s = 1; s <= 16; s <<= 1) { result += simd_shuffle_down(result, s); }"
        } else {
            "for (ushort s = 16; s >= 1; s >>= 1) { result += simd_shuffle_down(result, s); }"
        };
        let source = DEPTH_ONE_SOURCE
            .replace("NSRC", &count.to_string())
            .replace("DIMS", &dims.to_string())
            .replace("SRCLIST", &names[..count].join(", "))
            .replace("SCALEBITS", &format!("0x{:08x}", scale.to_bits()))
            .replace("SHUFFLE_TREE", tree);
        let refs: Vec<&str> = names.iter().map(String::as_str).collect();
        crate::array::PreparedKernel::new(
            &format!(
                "inkling_depth_one_{count}_{dims}{}",
                if mutate { "_m" } else { "" }
            ),
            &refs,
            &["out"],
            "",
            &source,
        )
    })?;
    let mut inputs: Vec<&Array> = sources.to_vec();
    inputs.push(queries);
    // one threadgroup per 256 output columns, as `gemv_t` (each recomputes the scores and the softmax)
    let groups = (dims + 255) / 256;
    let mut outputs = kernel.run(
        &inputs,
        &[&[1, 1, dims]],
        &[Dtype::Float32],
        [256 * groups, 1, 1],
        [256, 1, 1],
    )?;
    Ok(Some(outputs.pop().context("depth attention output")?))
}

/// `[T, N, D]` stack of the depth sources.
fn depth_stack(sources: &[&Array]) -> Result<Array> {
    ensure!(!sources.is_empty(), "depth attention needs a source");
    let [time, dims]: [i32; 2] = sources[0]
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("depth sources must have rank 2"))?;
    let stacked = sources
        .iter()
        .map(|s| s.reshape(&[time, 1, dims]))
        .collect::<Result<Vec<_>>>()?;
    Array::concatenate(&stacked.iter().collect::<Vec<_>>(), 1)
}

fn depth_attend(stacked: &Array, queries: &Array) -> Result<Array> {
    let [time, sources, dims]: [i32; 3] = stacked
        .shape()
        .try_into()
        .map_err(|_| anyhow::anyhow!("depth stack must have rank 3"))?;
    let count = queries.shape()[0];
    if ablated("depth") {
        return stacked
            .slice(1, sources - 1, sources)?
            .broadcast_to(&[time, count, dims]);
    }
    // one query: both transposes only move a size-1 axis, so a reshape gives the same array
    let single = count == 1 && lean();
    let queries_t = if single {
        queries.reshape(&[dims, 1])?
    } else {
        queries.transpose(&[1, 0])?
    };
    let scores = stacked
        .matmul(&queries_t)?
        .scalar_mul((dims as f32).powf(-0.5))?;
    let weights = if single {
        scores.reshape(&[time, 1, sources])?
    } else {
        scores.transpose(&[0, 2, 1])?
    }
    .softmax_precise()?;
    weights.matmul(stacked)
}

/// `[T, H, D]` -> `[1, H, T, D]`; with one row the transpose only moves a size-1 axis.
fn heads_first(x: &Array, time: i32, heads: i32, dims: i32) -> Result<Array> {
    if time == 1 && lean() {
        x.reshape(&[1, heads, 1, dims])
    } else {
        x.transpose(&[1, 0, 2])?.reshape(&[1, heads, time, dims])
    }
}

/// Depthwise causal short convolution over a `[T, C]` FP32 sequence with its `K - 1` previous rows.
struct ShortConv {
    /// `[C, K, 1]` FP32 (MLX conv1d layout).
    weight: Array,
    /// `[1]` FP32: `1 / sqrt(K + 1)`.
    scale: Array,
    channels: i32,
    kernel: i32,
}

/// One thread per window element `(channel, row)`: copies the window `[K - 1 + T, C]` (state rows, then the
/// inputs as FP32) and, for the first `T` rows, writes `(conv + x) * scale`. The taps are accumulated exactly as
/// MLX's `depthwise_conv_1d` kernel does (`acc = 0; acc += in * w` in tap order), then the residual add and the
/// scale as the two separate operations they replace, so the result is the same bits as the op sequence.
const SHORT_CONV_SOURCE: &str = r#"
    uint c = thread_position_in_grid.x;
    uint row = thread_position_in_grid.y;
    int K = w_shape[1];
    int K1 = K - 1;
    int T = x_shape[0];
    int C = x_shape[1];
    float v;
    if (int(row) < K1) {
        v = state[row * C + c];
    } else {
        v = static_cast<float>(x[(row - K1) * C + c]);
    }
    window[row * C + c] = v;
    if (int(row) < T) {
        float acc = 0.0;
        for (int i = 0; i < K; ++i) {
            int r = int(row) + i;
            float in_v;
            if (r < K1) {
                in_v = state[r * C + c];
            } else {
                in_v = static_cast<float>(x[(r - K1) * C + c]);
            }
            acc += in_v * w[c * K + i];
        }
        float sum = acc + static_cast<float>(x[row * C + c]);
        out[row * C + c] = sum * scale[0];
    }
"#;

impl ShortConv {
    fn load(checkpoint: &Checkpoint, name: &str, channels: i32, kernel: i32) -> Result<Self> {
        let weight = f32_weight(checkpoint, name, &[channels, 1, kernel])?
            .transpose(&[0, 2, 1])?
            .reshape(&[channels, kernel, 1])?;
        weight.eval()?;
        Ok(Self {
            weight,
            scale: scalar(1.0 / ((kernel + 1) as f32).sqrt())?,
            channels,
            kernel,
        })
    }

    /// Returns `((conv + x) / sqrt(K + 1), window)`; the window `[K - 1 + T, C]` holds the next states.
    fn forward(&self, x: &Array, state: Option<&Array>) -> Result<(Array, Array)> {
        if ablated("conv") {
            let rows = x.shape()[0] + self.kernel - 1;
            return Ok((
                x.astype(Dtype::Float32)?,
                Array::zeros_dtype(&[rows, self.channels], Dtype::Float32)?,
            ));
        }
        if lean() {
            self.forward_kernel(x, state)
        } else {
            self.forward_ops(x, state)
        }
    }

    /// The whole step in one Metal kernel (same bits as `forward_ops`).
    fn forward_kernel(&self, x: &Array, state: Option<&Array>) -> Result<(Array, Array)> {
        let time = x.shape()[0];
        ensure!(
            x.shape() == [time, self.channels]
                && matches!(x.dtype(), Dtype::Float16 | Dtype::Float32),
            "short convolution input"
        );
        let zeros;
        let state = match state {
            Some(state) => state,
            None => {
                zeros = Array::zeros_dtype(&[self.kernel - 1, self.channels], Dtype::Float32)?;
                &zeros
            }
        };
        ensure!(
            state.shape() == [self.kernel - 1, self.channels] && state.dtype() == Dtype::Float32,
            "short convolution state"
        );
        let rows = time + self.kernel - 1;
        let kernel = crate::array::prepared_kernel("inkling_short_conv", [0; 16], || {
            crate::array::PreparedKernel::new(
                "inkling_short_conv",
                &["x", "state", "w", "scale"],
                &["out", "window"],
                "",
                SHORT_CONV_SOURCE,
            )
        })?;
        let mut outputs = kernel.run(
            &[x, state, &self.weight, &self.scale],
            &[&[time, self.channels], &[rows, self.channels]],
            &[Dtype::Float32, Dtype::Float32],
            [self.channels, rows, 1],
            [self.channels.min(256), 1, 1],
        )?;
        let window = outputs.pop().context("short convolution window")?;
        let out = outputs.pop().context("short convolution output")?;
        Ok((out, window))
    }

    fn forward_ops(&self, x: &Array, state: Option<&Array>) -> Result<(Array, Array)> {
        let time = x.shape()[0];
        let x = x.astype(Dtype::Float32)?;
        let window = with_state(&x, state, self.kernel, self.channels)?;
        let conv = window
            .reshape(&[1, time + self.kernel - 1, self.channels])?
            .conv1d(&self.weight, self.channels)?
            .reshape(&[time, self.channels])?;
        let out = add_scaled(&conv, &x, 1.0 / ((self.kernel + 1) as f32).sqrt())?;
        Ok((out, window))
    }
}

fn with_state(x: &Array, state: Option<&Array>, kernel: i32, channels: i32) -> Result<Array> {
    let zeros;
    let state = match state {
        Some(state) => state,
        None => {
            zeros = Array::zeros_dtype(&[kernel - 1, channels], Dtype::Float32)?;
            &zeros
        }
    };
    Array::concatenate(&[state, x], 0)
}

/// Convolution state after the first `rows` inputs of a `[K - 1 + T, C]` window.
fn state_after(window: &Array, kernel: i32, rows: i32) -> Result<Array> {
    window.slice(0, rows, rows + kernel - 1)
}

/// Short convolution whose taps are static per-channel weights plus per-token weights projected
/// from the projection input, shared within each key/value head (full-attention layers).
struct DynamicConv {
    /// `[H, C / H, K]` FP32, tap index = backward distance.
    static_weight: Array,
    weight_proj: Dense,
    heads: i32,
    channels: i32,
    kernel: i32,
}

impl DynamicConv {
    fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        hidden: i32,
        heads: i32,
        channels: i32,
        kernel: i32,
    ) -> Result<Self> {
        let static_weight = f32_weight(
            checkpoint,
            &format!("{prefix}.static_w"),
            &[kernel, channels],
        )?
        .transpose(&[1, 0])?
        .reshape(&[heads, channels / heads, kernel])?;
        static_weight.eval()?;
        Ok(Self {
            static_weight,
            weight_proj: Dense::load(
                checkpoint,
                &format!("{prefix}.weight_proj"),
                heads * kernel,
                hidden,
            )?,
            heads,
            channels,
            kernel,
        })
    }

    fn forward(&self, x: &Array, input: &Array, state: Option<&Array>) -> Result<(Array, Array)> {
        let time = x.shape()[0];
        if ablated("conv") {
            return Ok((
                x.astype(Dtype::Float32)?,
                Array::zeros_dtype(&[time + self.kernel - 1, self.channels], Dtype::Float32)?,
            ));
        }
        let per_head = self.channels / self.heads;
        let weights = self
            .weight_proj
            .forward(input)?
            .astype(Dtype::Float32)?
            .reshape(&[time, self.heads, 1, self.kernel])?
            .add(&self.static_weight)?;
        let x = x.astype(Dtype::Float32)?;
        let window = with_state(&x, state, self.kernel, self.channels)?;
        let mut out: Option<Array> = None;
        for lag in 0..self.kernel {
            let start = self.kernel - 1 - lag;
            let shifted = window
                .slice(0, start, start + time)?
                .reshape(&[time, self.heads, per_head])?;
            let tap = weights
                .slice(3, lag, lag + 1)?
                .reshape(&[time, self.heads, per_head])?;
            let term = shifted.mul(&tap)?;
            out = Some(match out {
                Some(acc) => acc.add(&term)?,
                None => term,
            });
        }
        let out = out
            .context("empty dynamic convolution")?
            .reshape(&[time, self.channels])?;
        let out = add_scaled(&out, &x, 1.0 / ((2 * self.kernel + 1) as f32).sqrt())?;
        Ok((out, window))
    }
}

enum KvConv {
    Static { k: ShortConv, v: ShortConv },
    Dynamic { k: DynamicConv, v: DynamicConv },
}

/// Positions shared by every visit of one attention kind during a forward call.
struct MaskPlan {
    /// `[T * S]` flat indices `row * R + min(distance, R - 1)` into the per-row bias profiles.
    gather: Array,
    /// `[T, S]` 1 where `0 <= distance < R`.
    bias_valid: Array,
    /// `[T, S]` 0 for visible keys, -inf otherwise.
    visible: Array,
    /// Every `bias_valid` entry is 1 / every `visible` entry is 0: the multiply / add are identities.
    all_valid: bool,
    all_visible: bool,
    time: i32,
    keys: i32,
}

impl MaskPlan {
    /// Queries at positions `start + i` (i < time) attend keys at positions `first_key..first_key + keys`.
    fn new(
        start: i32,
        time: i32,
        first_key: i32,
        keys: i32,
        extent: i32,
        window: Option<i32>,
    ) -> Result<Self> {
        let count = (time as usize)
            .checked_mul(keys as usize)
            .context("attention mask overflow")?;
        let (mut gather, mut valid, mut visible) = (
            Vec::with_capacity(count),
            Vec::with_capacity(count),
            Vec::with_capacity(count),
        );
        for row in 0..time {
            for key in 0..keys {
                let distance = start + row - (first_key + key);
                let seen = distance >= 0 && window.is_none_or(|w| distance < w);
                gather.push(row * extent + distance.clamp(0, extent - 1));
                valid.push(if (0..extent).contains(&distance) {
                    1.
                } else {
                    0.
                });
                visible.push(if seen { 0. } else { f32::NEG_INFINITY });
            }
        }
        Ok(Self {
            all_valid: valid.iter().all(|&v| v == 1.),
            all_visible: visible.iter().all(|&v| v == 0.),
            gather: Array::from_i32(&gather, &[time * keys])?,
            bias_valid: Array::from_f32(&valid, &[time, keys])?,
            visible: Array::from_f32(&visible, &[time, keys])?,
            time,
            keys,
        })
    }
}

#[derive(Default)]
struct VisitState {
    /// Cached keys/values FP16. Lean layout (default): `[1, Hkv, S, D]`, the real rows only, kept as a view
    /// of the last attended array so a sliding trim copies nothing. `MLXL3_INKLING_UNFUSED=1`:
    /// `[1, Hkv, 1 + S, D]` with the zero sink row first, trimmed by a copy.
    keys: Option<Array>,
    values: Option<Array>,
    conv: [Option<Array>; 4],
    /// Last call, not committed yet: keys/values with every row of the call (no sliding trim), the row count
    /// before the call, and the four `[K - 1 + T, C]` convolution windows.
    pending_kv: Option<(Array, Array, i32)>,
    pending_conv: [Option<Array>; 4],
    /// In-place K/V storage (SPEED-INK2M-P12); `keys`/`values` are then views of it.
    kv_buffer: Option<KvBuffer>,
}

/// `[1, Hkv, cap, D]` FP16 buffers: rows `[base, base + len)` hold the committed keys/values, row `zero` is
/// known to be zero. An attended call reads the view `[base - 1, base + len + T)`: the zero sink row, the
/// past and the chunk, the same rows in the same order as the concatenation.
struct KvBuffer {
    keys: Array,
    values: Array,
    base: i32,
    len: i32,
    zero: i32,
}

const KV_STEP: i32 = 256;

/// Buffer rows for a call that needs `needed` rows (sink, past, chunk of `time` rows). Full visits grow by
/// steps of 256; sliding visits keep room for a 256-row (or larger) chunk after a full window.
fn kv_capacity(needed: i32, window: Option<i32>, time: i32) -> Option<i32> {
    let rows = match window {
        Some(window) => needed.max(window.checked_add(time.max(KV_STEP))?),
        None => needed,
    };
    if needed < 1 || time < 1 || rows < needed {
        return None;
    }
    rows.checked_add(KV_STEP - 1).map(|r| r / KV_STEP * KV_STEP)
}

#[cfg(test)]
thread_local! {
    static KV_INPLACE_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    /// Mutation for the parity test: skip the sink rewrite of sliding visits.
    static KV_SKIP_SINK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// SPEED-2026-11-23-INK2M-MAYA PC mutations of `resume_prefix`: 1 keeps the live convolution windows,
    /// 2 keeps the live sliding K/V instead of the saved ones.
    static PREFIX_MUTATION: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
    /// SPEED-INK2M-P13: `run` submits by `async_eval` and returns without waiting for the GPU.
    static ASYNC_SUBMIT: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// SPEED-INK2M-P13: thread CPU time spent in `run` (construction, evaluation) since the last take.
    static RUN_CPU: std::cell::Cell<(Duration, Duration)> =
        const { std::cell::Cell::new((Duration::ZERO, Duration::ZERO)) };
}

/// CPU time of the calling thread (zero if the clock is unavailable).
#[cfg(test)]
fn thread_cpu() -> Duration {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `t` is a valid out-pointer for the duration of the call.
    if unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &mut t) } != 0 {
        return Duration::ZERO;
    }
    Duration::new(
        t.tv_sec.max(0) as u64,
        t.tv_nsec.clamp(0, 999_999_999) as u32,
    )
}

/// Evaluate the outputs of one `run` (tests may submit them asynchronously instead).
fn submit(pending: &[&Array]) -> Result<()> {
    #[cfg(test)]
    if ASYNC_SUBMIT.with(std::cell::Cell::get) {
        return Array::async_eval_all(pending);
    }
    Array::eval_many(pending)
}

/// K/V stored in place (`MLXL3_INKLING_KV_INPLACE=0` keeps the concatenation); lean layout only.
fn kv_inplace() -> bool {
    #[cfg(test)]
    if let Some(enabled) = KV_INPLACE_OVERRIDE.with(std::cell::Cell::get) {
        return enabled && lean();
    }
    static INPLACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *INPLACE.get_or_init(|| {
        lean() && std::env::var("MLXL3_INKLING_KV_INPLACE").map_or(true, |v| v != "0")
    })
}

impl VisitState {
    /// Keep the first `rows` rows of the pending call. Sliding visits then keep the sink row and the last
    /// `window - 1` real keys.
    fn commit(&mut self, rows: i32, calls: i32, window: Option<i32>, kernel: i32) -> Result<()> {
        let (keys, values, before) = self.pending_kv.take().context("no pending Inkling call")?;
        if let Some(buffer) = &mut self.kv_buffer {
            // in place: only the logical rows move; same kept rows as the trims below
            drop((keys, values));
            ensure!(
                before == buffer.len + 1 && (1..=calls).contains(&rows),
                "Inkling in-place cache and pending call disagree"
            );
            let total = buffer.len + rows;
            match window {
                Some(window) if total > window - 1 => {
                    buffer.base += total - (window - 1);
                    buffer.len = window - 1;
                }
                _ => buffer.len = total,
            }
            ensure!(buffer.len > 0, "Inkling cache would be empty");
            let end = buffer.base + buffer.len;
            self.keys = Some(buffer.keys.slice(2, buffer.base, end)?);
            self.values = Some(buffer.values.slice(2, buffer.base, end)?);
            for i in 0..4 {
                let window = self.pending_conv[i]
                    .take()
                    .context("missing pending convolution")?;
                self.conv[i] = Some(state_after(&window, kernel, rows)?);
            }
            return Ok(());
        }
        let cut = |a: Array| -> Result<Array> {
            if rows == calls {
                Ok(a)
            } else {
                a.slice(2, 0, before + rows)
            }
        };
        let (keys, values) = (cut(keys)?, cut(values)?);
        let total = keys.shape()[2];
        if lean() {
            // drop the sink row (index 0) and, on sliding visits, everything but the last window - 1 rows:
            // a view of the attended array, same rows in the same order as the copying trim below
            let from = match window {
                Some(window) if total - 1 > window - 1 => total - (window - 1),
                _ => 1,
            };
            ensure!(from < total, "Inkling cache would be empty");
            self.keys = Some(keys.slice(2, from, total)?);
            self.values = Some(values.slice(2, from, total)?);
            for i in 0..4 {
                let window = self.pending_conv[i]
                    .take()
                    .context("missing pending convolution")?;
                self.conv[i] = Some(state_after(&window, kernel, rows)?);
            }
            return Ok(());
        }
        let keep = match window {
            Some(window) if total - 1 > window - 1 => Some(total - (window - 1)),
            _ => None,
        };
        let trim = |a: &Array| -> Result<Array> {
            match keep {
                Some(from) => {
                    Array::concatenate(&[&a.slice(2, 0, 1)?, &a.slice(2, from, total)?], 2)
                }
                None => a.try_clone(),
            }
        };
        self.keys = Some(trim(&keys)?);
        self.values = Some(trim(&values)?);
        for i in 0..4 {
            let window = self.pending_conv[i]
                .take()
                .context("missing pending convolution")?;
            self.conv[i] = Some(state_after(&window, kernel, rows)?);
        }
        Ok(())
    }
}

struct Attention {
    layer: usize,
    q: Linear,
    k: Linear,
    v: Linear,
    r: Linear,
    o: Linear,
    gate: Linear,
    sink: Linear,
    rel_proj: Array,
    q_scale: Array,
    rel_scale: Array,
    conv: KvConv,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    d_rel: i32,
    extent: i32,
    window: Option<i32>,
    log_scaling: Option<(f32, f32)>,
    qk_clip: f32,
    rel_clip: f32,
    eps: f32,
    /// `[1, Hkv, 1, D]` FP16 zeros: the sink row written into in-place buffers.
    zero_row: Array,
}

impl Attention {
    fn load(checkpoint: &Checkpoint, layer: usize, sliding: bool, c: &TextConfig) -> Result<Self> {
        let prefix = format!("model.llm.layers.{layer}.attn");
        let (heads, kv_heads, head_dim) = if sliding {
            (
                c.swa_num_attention_heads,
                c.swa_num_key_value_heads,
                c.swa_head_dim,
            )
        } else {
            (c.num_attention_heads, c.num_key_value_heads, c.head_dim)
        };
        ensure!(
            heads > 0 && kv_heads > 0 && heads % kv_heads == 0 && head_dim > 0,
            "invalid Inkling attention heads"
        );
        let extent = if sliding {
            c.sliding_window_size
        } else {
            c.rel_extent
        };
        let hidden = c.hidden_size;
        let q_dim = heads * head_dim;
        let kv_dim = kv_heads * head_dim;
        let proj = |role: &str, name: &str, input: i32, output: i32| {
            let prefix = format!("{prefix}.{name}");
            let bits = if c.dense_only.contains(role) {
                c.dense_bits
            } else {
                16
            };
            match (&c.dense_sidecar, bits) {
                (Some(sidecar), 8) => Linear::from_sidecar(sidecar, &prefix, input, output),
                (_, 16) | (None, _) if c.dense_group == AFFINE_GROUP || bits >= 16 => {
                    Linear::load(checkpoint, &prefix, input, output, bits)
                }
                _ => {
                    let dense = Projection::load(checkpoint, &prefix, input, output, false)?;
                    Linear::quantized_group(
                        dense.dense_weight().context("dense weight")?,
                        bits,
                        c.dense_group,
                    )
                }
            }
        };
        let conv = if sliding {
            KvConv::Static {
                k: ShortConv::load(
                    checkpoint,
                    &format!("{prefix}.k_sconv.weight"),
                    kv_dim,
                    c.sconv_kernel_size,
                )?,
                v: ShortConv::load(
                    checkpoint,
                    &format!("{prefix}.v_sconv.weight"),
                    kv_dim,
                    c.sconv_kernel_size,
                )?,
            }
        } else {
            KvConv::Dynamic {
                k: DynamicConv::load(
                    checkpoint,
                    &format!("{prefix}.k_sconv"),
                    hidden,
                    kv_heads,
                    kv_dim,
                    c.sconv_kernel_size,
                )?,
                v: DynamicConv::load(
                    checkpoint,
                    &format!("{prefix}.v_sconv"),
                    hidden,
                    kv_heads,
                    kv_dim,
                    c.sconv_kernel_size,
                )?,
            }
        };
        Ok(Self {
            layer,
            q: proj("q", "wq_du", hidden, q_dim)?,
            k: proj("k", "wk_dv", hidden, kv_dim)?,
            v: proj("v", "wv_dv", hidden, kv_dim)?,
            r: proj("r", "wr_du", hidden, heads * c.d_rel)?,
            o: proj("o", "wo_ud", q_dim, hidden)?,
            gate: proj("gate", "gate_attn", hidden, q_dim)?,
            sink: proj("sink", "attn_sink_proj", hidden, heads)?,
            rel_proj: f32_weight(
                checkpoint,
                &format!("{prefix}.rel_logits_proj.proj"),
                &[c.d_rel, extent],
            )?,
            q_scale: f32_weight(
                checkpoint,
                &format!("{prefix}.q_l2_clip.scale"),
                &[heads, head_dim],
            )?,
            rel_scale: f32_weight(
                checkpoint,
                &format!("{prefix}.rel_l2_clip.scale"),
                &[heads, c.d_rel],
            )?,
            conv,
            heads,
            kv_heads,
            head_dim,
            d_rel: c.d_rel,
            extent,
            window: sliding.then_some(c.sliding_window_size),
            log_scaling: (!sliding).then_some((c.log_scaling_alpha, c.log_scaling_n_floor)),
            qk_clip: c.qk_l2_clip_threshold,
            rel_clip: c.rel_l2_clip_threshold,
            eps: c.rms_norm_eps,
            zero_row: {
                let zero = Array::zeros_dtype(&[1, kv_heads, 1, head_dim], Dtype::Float16)?;
                zero.eval()?;
                zero
            },
        })
    }

    /// `tau = 1 + alpha * log(max((position + 1) / n_floor, 1))` per query row, `[T, 1, 1]`.
    fn tau(&self, start: i32, time: i32) -> Result<Option<Array>> {
        let Some((alpha, floor)) = self.log_scaling else {
            return Ok(None);
        };
        let values = (0..time)
            .map(|i| 1.0 + alpha * (((start + i + 1) as f32 / floor).max(1.0)).ln())
            .collect::<Vec<_>>();
        Ok(Some(Array::from_f32(&values, &[time, 1, 1])?))
    }

    /// `hq`, `hk`, `hv`: normalized FP16 inputs `[T, hidden]`. Returns the attention output `[T, hidden]` FP16.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hq: &Array,
        hk: &Array,
        hv: &Array,
        start: i32,
        plan: &MaskPlan,
        state: &mut VisitState,
    ) -> Result<Array> {
        let time = hq.shape()[0];
        capture_dense_input(self.layer, 0, hq)?;
        capture_dense_input(self.layer, 1, hk)?;
        capture_dense_input(self.layer, 2, hv)?;
        let k_proj = self.k.forward(hk)?;
        let v_proj = self.v.forward(hv)?;
        stage("a.k_proj", &k_proj)?;
        stage("a.v_proj", &v_proj)?;
        let (k, v, k_state, v_state) = match &self.conv {
            KvConv::Static { k, v } => {
                let (ko, ks) = k.forward(&k_proj, state.conv[0].as_ref())?;
                let (vo, vs) = v.forward(&v_proj, state.conv[1].as_ref())?;
                (ko, vo, ks, vs)
            }
            KvConv::Dynamic { k, v } => {
                let (ko, ks) = k.forward(&k_proj, hk, state.conv[0].as_ref())?;
                let (vo, vs) = v.forward(&v_proj, hv, state.conv[1].as_ref())?;
                (ko, vo, ks, vs)
            }
        };
        stage("a.k_conv", &k)?;
        stage("a.v_conv", &v)?;
        let tau = self.tau(start, time)?;
        let q_proj = self.q.forward(hq)?;
        let r_proj = self.r.forward(hq)?;
        stage("a.q_proj", &q_proj)?;
        stage("a.r_proj", &r_proj)?;
        let mut q = l2_clip(
            &q_proj.reshape(&[time, self.heads, self.head_dim])?,
            Some(&self.q_scale),
            self.qk_clip,
        )?;
        let mut rel = l2_clip(
            &r_proj.reshape(&[time, self.heads, self.d_rel])?,
            Some(&self.rel_scale),
            self.rel_clip,
        )?;
        if let Some(tau) = &tau {
            q = q.mul(tau)?;
            rel = rel.mul(tau)?;
        }
        stage("a.q", &q)?;
        stage("a.rel", &rel)?;
        let k = l2_clip(
            &k.reshape(&[time, self.kv_heads, self.head_dim])?,
            None,
            self.qk_clip,
        )?;
        let v = v
            .reshape(&[time, self.kv_heads, self.head_dim])?
            .rms_norm_without_weight(self.eps)?;
        stage("a.k", &k)?;
        stage("a.v", &v)?;
        let q = heads_first(&q.astype(Dtype::Float16)?, time, self.heads, self.head_dim)?;
        let k = heads_first(
            &k.astype(Dtype::Float16)?,
            time,
            self.kv_heads,
            self.head_dim,
        )?;
        let v = heads_first(
            &v.astype(Dtype::Float16)?,
            time,
            self.kv_heads,
            self.head_dim,
        )?;
        let (keys, values) = self.append(state, &k, &v, start)?;
        ensure!(
            keys.shape()[2] == plan.keys + 1 && plan.time == time,
            "Inkling attention cache and mask disagree"
        );
        // Bias profiles [T, H, R] -> per (head, query, key) through the distance gather
        let lean = lean();
        let one = time == 1 && lean;
        if ablated("attn") {
            // keep the projections and the cache append; drop bias, mask and attention
            state.pending_conv[0] = Some(k_state);
            state.pending_conv[1] = Some(v_state);
            let gate = self.gate.forward(hq)?;
            let _ = (&rel, &keys, &values);
            return self.o.forward(
                &q.reshape(&[time, self.heads * self.head_dim])?
                    .add(&gate.scalar_mul(0.0)?)?,
            );
        }
        let profiles = rel.matmul(&self.rel_proj)?;
        stage("a.profiles", &profiles)?;
        let profiles = if one {
            profiles.reshape(&[self.heads, self.extent])?
        } else {
            profiles
                .transpose(&[1, 0, 2])?
                .reshape(&[self.heads, time * self.extent])?
        };
        let sink_raw = self.sink.forward(hq)?;
        stage("a.sink", &sink_raw)?;
        let mut bias = if ablated("bias") {
            Array::zeros_dtype(&[self.heads, time, plan.keys], Dtype::Float32)?
        } else {
            profiles
                .take(&plan.gather, 1)?
                .reshape(&[self.heads, time, plan.keys])?
        };
        // x * 1 and x + 0 are exact identities (a -0 bias would become +0, which the attention scores
        // absorb identically); skip them when the whole mask says so
        if !(lean && plan.all_valid) {
            bias = bias.mul(&plan.bias_valid)?;
        }
        if !(lean && plan.all_visible) {
            bias = bias.add(&plan.visible)?;
        }
        let sinks = sink_raw.astype(Dtype::Float32)?;
        let sinks = if one {
            sinks.reshape(&[self.heads, 1, 1])?
        } else {
            sinks.transpose(&[1, 0])?.reshape(&[self.heads, time, 1])?
        };
        let mask = Array::concatenate(&[&sinks, &bias], 2)?
            .reshape(&[1, self.heads, time, plan.keys + 1])?
            .astype(Dtype::Float16)?;
        self.finish(
            hq, &q, &keys, &values, &mask, time, one, state, k_state, v_state,
        )
    }

    /// Masked attention, output gate and output projection (shared by the plain and F4 mask paths).
    #[allow(clippy::too_many_arguments)]
    fn finish(
        &self,
        hq: &Array,
        q: &Array,
        keys: &Array,
        values: &Array,
        mask: &Array,
        time: i32,
        one: bool,
        state: &mut VisitState,
        k_state: Array,
        v_state: Array,
    ) -> Result<Array> {
        let attended = Array::sdpa_mask(q, keys, values, (self.head_dim as f32).powf(-0.5), mask)?;
        let attended = if one {
            attended.reshape(&[1, self.heads * self.head_dim])?
        } else {
            attended
                .reshape(&[self.heads, time, self.head_dim])?
                .transpose(&[1, 0, 2])?
                .reshape(&[time, self.heads * self.head_dim])?
        };
        stage("a.sdpa", &attended)?;
        let gate_raw = self.gate.forward(hq)?;
        stage("a.gate_raw", &gate_raw)?;
        // SPEED-INK2M-P17 F5: one row: casts, sigmoid and product in one kernel
        let gated = if one
            && p17("f5")
            && attended.dtype() == Dtype::Float16
            && gate_raw.dtype() == Dtype::Float16
        {
            attention_gate_one(&attended, &gate_raw)?
        } else {
            let gate = gate_raw.astype(Dtype::Float32)?.sigmoid()?;
            attended
                .astype(Dtype::Float32)?
                .mul(&gate)?
                .astype(Dtype::Float16)?
        };
        state.pending_conv[0] = Some(k_state);
        state.pending_conv[1] = Some(v_state);
        capture_dense_input(self.layer, 3, &gated)?;
        stage("a.gated", &gated)?;
        self.o.forward(&gated)
    }

    /// Appends the chunk and returns the keys/values it attends (sink row, past, chunk). Sliding
    /// layers then keep only the sink row and the last `window - 1` real keys for the next call.
    fn append(
        &self,
        state: &mut VisitState,
        k: &Array,
        v: &Array,
        start: i32,
    ) -> Result<(Array, Array)> {
        if kv_inplace() {
            return self.append_inplace(state, k, v, start);
        }
        if lean() {
            // lean layout: the cache holds the real rows only; one concatenation builds [sink, past, chunk]
            let zero = Array::zeros_dtype(&[1, self.kv_heads, 1, self.head_dim], Dtype::Float16)?;
            let (keys, values, before) = match (&state.keys, &state.values) {
                (Some(keys), Some(values)) => (
                    Array::concatenate(&[&zero, keys, k], 2)?,
                    Array::concatenate(&[&zero, values, v], 2)?,
                    keys.shape()[2] + 1,
                ),
                _ => {
                    ensure!(start == 0, "Inkling cache must start at position zero");
                    (
                        Array::concatenate(&[&zero, k], 2)?,
                        Array::concatenate(&[&zero, v], 2)?,
                        1,
                    )
                }
            };
            state.pending_kv = Some((keys.try_clone()?, values.try_clone()?, before));
            return Ok((keys, values));
        }
        let (keys, values) = match (&state.keys, &state.values) {
            (Some(keys), Some(values)) => (
                Array::concatenate(&[keys, k], 2)?,
                Array::concatenate(&[values, v], 2)?,
            ),
            _ => {
                ensure!(start == 0, "Inkling cache must start at position zero");
                let zero =
                    Array::zeros_dtype(&[1, self.kv_heads, 1, self.head_dim], Dtype::Float16)?;
                (
                    Array::concatenate(&[&zero, k], 2)?,
                    Array::concatenate(&[&zero, v], 2)?,
                )
            }
        };
        let before = state.keys.as_ref().map_or(1, |k| k.shape()[2]);
        state.pending_kv = Some((keys.try_clone()?, values.try_clone()?, before));
        Ok((keys, values))
    }

    /// `append` with the keys/values written into a per-visit buffer: returns views of it with the same rows
    /// as the lean concatenation `[sink, past, chunk]`.
    fn append_inplace(
        &self,
        state: &mut VisitState,
        k: &Array,
        v: &Array,
        start: i32,
    ) -> Result<(Array, Array)> {
        let time = k.shape()[2];
        ensure!(
            time >= 1 && v.shape()[2] == time,
            "Inkling cache chunk is empty"
        );
        // drop the views of the previous buffer so MLX can update it in place
        state.keys = None;
        state.values = None;
        let previous = state.kv_buffer.take();
        if previous.is_none() {
            ensure!(start == 0, "Inkling cache must start at position zero");
        }
        let (base, len) = previous.as_ref().map_or((1, 0), |b| (b.base, b.len));
        let end = base
            .checked_add(len)
            .and_then(|e| e.checked_add(time))
            .context("Inkling cache overflow")?;
        let mut buffer = match previous {
            Some(b) if b.keys.shape()[2] >= end => b,
            previous => {
                // new buffer: zero sink row at 0, the committed rows copied once to [1, 1 + len)
                let needed = len
                    .checked_add(time)
                    .and_then(|n| n.checked_add(1))
                    .context("Inkling cache overflow")?;
                let capacity =
                    kv_capacity(needed, self.window, time).context("Inkling cache overflow")?;
                let shape = [1, self.kv_heads, capacity, self.head_dim];
                let mut keys = Array::zeros_dtype(&shape, Dtype::Float16)?;
                let mut values = Array::zeros_dtype(&shape, Dtype::Float16)?;
                if let Some(old) = previous.filter(|old| old.len > 0) {
                    let to = old.base + old.len;
                    keys = keys.slice_update(&old.keys.slice(2, old.base, to)?, &[0, 0, 1, 0])?;
                    values =
                        values.slice_update(&old.values.slice(2, old.base, to)?, &[0, 0, 1, 0])?;
                }
                KvBuffer {
                    keys,
                    values,
                    base: 1,
                    len,
                    zero: 0,
                }
            }
        };
        #[cfg(test)]
        let skip_sink = KV_SKIP_SINK.with(std::cell::Cell::get);
        #[cfg(not(test))]
        let skip_sink = false;
        if buffer.zero != buffer.base - 1 && !skip_sink {
            // the row just before the window left it: it becomes the sink row
            let at = [0, 0, buffer.base - 1, 0];
            buffer.keys = buffer.keys.slice_update(&self.zero_row, &at)?;
            buffer.values = buffer.values.slice_update(&self.zero_row, &at)?;
            buffer.zero = buffer.base - 1;
        }
        let at = [0, 0, buffer.base + buffer.len, 0];
        buffer.keys = buffer.keys.slice_update(k, &at)?;
        buffer.values = buffer.values.slice_update(v, &at)?;
        let end = buffer.base + buffer.len + time;
        let keys = buffer.keys.slice(2, buffer.base - 1, end)?;
        let values = buffer.values.slice(2, buffer.base - 1, end)?;
        state.pending_kv = Some((keys.try_clone()?, values.try_clone()?, buffer.len + 1));
        state.kv_buffer = Some(buffer);
        Ok((keys, values))
    }
}

/// `(1 - tanh(forget(s))) * h + tanh(gain(s)) * m`, `s = down([rms(h), rms(m)])`.
struct MessageGate {
    down: Dense,
    /// `[rank, hidden]` FP32 (transposed for matmul).
    forget_t: Array,
    gain_t: Array,
    /// SPEED-INK2M-P17 F2: the checkpoint's `[hidden, rank]` BF16 tensors (exact in FP32), read by the fused
    /// gate tail; `None` for another dtype.
    forget_bf16: Option<Array>,
    gain_bf16: Option<Array>,
    eps: f32,
}

impl MessageGate {
    fn load(
        checkpoint: &Checkpoint,
        prefix: &str,
        hidden: i32,
        rank: i32,
        eps: f32,
    ) -> Result<Self> {
        let t = |name: &str| -> Result<Array> {
            let w = f32_weight(checkpoint, &format!("{prefix}.{name}"), &[hidden, rank])?
                .transpose(&[1, 0])?;
            w.eval()?;
            Ok(w)
        };
        let bf16 = |name: &str| -> Result<Option<Array>> {
            let w = checkpoint_array(checkpoint, &format!("{prefix}.{name}"))?;
            if w.dtype() != Dtype::BFloat16 || w.shape() != [hidden, rank] {
                return Ok(None);
            }
            w.eval()?;
            Ok(Some(w))
        };
        Ok(Self {
            down: Dense::load(
                checkpoint,
                &format!("{prefix}.down_weight"),
                rank,
                2 * hidden,
            )?,
            forget_t: t("forget_weight")?,
            gain_t: t("gain_weight")?,
            forget_bf16: bf16("forget_weight")?,
            gain_bf16: bf16("gain_weight")?,
            eps,
        })
    }

    fn forward(&self, hidden: &Array, message: &Array) -> Result<Array> {
        self.forward_normed(hidden, &hidden.rms_norm_without_weight(self.eps)?, message)
    }

    /// `hidden_norm`: `rms(hidden)`, shared by the gates that read the same hidden state.
    fn forward_normed(
        &self,
        hidden: &Array,
        hidden_norm: &Array,
        message: &Array,
    ) -> Result<Array> {
        if ablated("gates") {
            return hidden.try_clone();
        }
        // INVARIANT-2026-11-24-INK2M-L1L4: every row through the 1-row gate (its fused kernels over the rows, or
        // one row at a time outside their domain)
        let rows = hidden.shape()[0];
        if invariant_call(rows) {
            let message_norm = message.rms_norm_without_weight(self.eps)?;
            if let Some(out) =
                self.forward_fused_rows(hidden, hidden_norm, &message_norm, message)?
            {
                return Ok(out);
            }
            let outs = (0..rows)
                .map(|r| {
                    self.forward_normed(
                        &hidden.slice(0, r, r + 1)?,
                        &hidden_norm.slice(0, r, r + 1)?,
                        &message.slice(0, r, r + 1)?,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            return Array::concatenate(&outs.iter().collect::<Vec<_>>(), 0);
        }
        let message_norm = message.rms_norm_without_weight(self.eps)?;
        if let Some(out) = self.forward_fused(hidden, hidden_norm, &message_norm, message)? {
            return Ok(out);
        }
        let states = Array::concatenate(&[hidden_norm, &message_norm], 1)?;
        let states = self.down.forward(&states)?.astype(Dtype::Float32)?;
        if fused() {
            return Array::message_gate(&states, &self.forget_t, &self.gain_t, hidden, message);
        }
        let forget = states.matmul(&self.forget_t)?.tanh()?;
        let gain = states.matmul(&self.gain_t)?.tanh()?;
        hidden.sub(&forget.mul(hidden)?)?.add(&gain.mul(message)?)
    }
}

/// SPEED-INK2M-P17 F5: `half(float(attended) * sigmoid(float(gate)))` in one kernel, with MLX's `Sigmoid`
/// functor (`1 / (1 + exp(|x|))`, then `y` or `1 - y`) and one statement per op.
const ATTENTION_GATE_SOURCE: &str = r#"
    const uint i = thread_position_in_grid.x;
    if (i < SIZE) {
        const float x = static_cast<float>(gate[i]);
        const float ax = metal::abs(x);
        const float e = metal::precise::exp(ax);
        const float d = 1 + e;
        const float y = 1 / d;
        const float s = (x < 0) ? y : 1 - y;
        const float a = static_cast<float>(attended[i]);
        const float p = a * s;
        out[i] = static_cast<half>(p);
    }
"#;

fn attention_gate_one(attended: &Array, gate: &Array) -> Result<Array> {
    let size = attended.shape().iter().product::<i32>();
    ensure!(
        attended.shape().len() == 2 && attended.shape()[0] == 1 && gate.shape() == attended.shape(),
        "attention gate inputs"
    );
    let key = [i64::from(size), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let kernel = crate::array::prepared_kernel("inkling_attention_gate", key, || {
        crate::array::PreparedKernel::new(
            &format!("inkling_attention_gate_{size}"),
            &["attended", "gate"],
            &["out"],
            "",
            &ATTENTION_GATE_SOURCE.replace("SIZE", &format!("{size}u")),
        )
    })?;
    kernel
        .run(
            &[attended, gate],
            &[attended.shape()],
            &[Dtype::Float16],
            [(size + 255) / 256 * 256, 1, 1],
            [256, 1, 1],
        )?
        .pop()
        .context("attention gate")
}

/// SPEED-INK2M-P17 F2a: `down(half([h_norm, m_norm]))` cast back to FP32 in one kernel, as MLX 0.32.2's
/// `concatenate`, `astype`, `gemv_float16_bm1_bn8_sm1_sn32_tm4_tn4` and `astype` (16 threadgroups of 8
/// simdgroups, 4 rows each; same thread layout and operation order).
const GATE_DOWN_SOURCE: &str = r#"
    const int lane = int(thread_index_in_simdgroup);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int group = int(threadgroup_position_in_grid.x);
    threadgroup float part[8 * 8];
    const int out_row = group * 4;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int bn = (32 * sg + lane) * 4;
    for (int i = 0; i < (2 * HID) / 1024; ++i) {
        float v[4];
        for (int t = 0; t < 4; ++t) {
            const int k = bn + t;
            v[t] = static_cast<float>(static_cast<half>(k < HID ? hn[k] : mn[k - HID]));
        }
        for (int tm = 0; tm < 4; ++tm) {
            half inter[4];
            for (int t = 0; t < 4; ++t) { inter[t] = w[(out_row + tm) * (2 * HID) + bn + t]; }
            for (int t = 0; t < 4; ++t) { result[tm] += inter[t] * v[t]; }
        }
        bn += 1024;
    }
    if ((2 * HID) % 1024 > 0) {
        float v[4];
        for (int t = 0; t < 4; ++t) {
            const int k = bn + t;
            v[t] = k < 2 * HID ? static_cast<float>(static_cast<half>(k < HID ? hn[k] : mn[k - HID])) : 0.0f;
        }
        for (int tm = 0; tm < 4; ++tm) {
            half inter[4];
            for (int t = 0; t < 4; ++t) {
                inter[t] = bn + t < 2 * HID ? w[(out_row + tm) * (2 * HID) + bn + t] : half(0);
            }
            for (int t = 0; t < 4; ++t) { result[tm] += inter[t] * v[t]; }
        }
    }
    for (int tm = 0; tm < 4; ++tm) {
        for (ushort sn = 16; sn >= 1; sn >>= 1) { result[tm] += simd_shuffle_down(result[tm], sn); }
    }
    if (lane == 0) {
        for (int tm = 0; tm < 4; ++tm) { part[sg * 8 + tm] = result[tm]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        for (int g = 1; g < 8; ++g) {
            for (int tm = 0; tm < 4; ++tm) { result[tm] += part[g * 8 + tm]; }
        }
        for (int tm = 0; tm < 4; ++tm) {
            out[out_row + tm] = static_cast<float>(static_cast<half>(result[tm]));
        }
    }
"#;

/// SPEED-INK2M-P17 F2b: `(h - tanh(s @ F) * h) + tanh(s @ G) * m` in one kernel, as MLX 0.32.2's two
/// `gemv_float32_bm1_bn1_sm8_sn4_tm4_tn4` (K = 64: one simdgroup per threadgroup, 32 rows) and the compiled
/// tail (one statement per op); F and G are read in BF16, exact in FP32.
const GATE_TAIL_SOURCE: &str = r#"
    const int lane = int(thread_index_in_simdgroup);
    const int group = int(threadgroup_position_in_grid.x);
    const int thr_m = lane / 4;
    const int thr_n = lane % 4;
    const int out_row = group * 32 + thr_m * 4;
    float rf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float rg[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int bn = thr_n * 4;
    for (int i = 0; i < RANK / 16; ++i) {
        float v[4];
        for (int t = 0; t < 4; ++t) { v[t] = s[bn + t]; }
        for (int tm = 0; tm < 4; ++tm) {
            float inter[4];
            for (int t = 0; t < 4; ++t) { inter[t] = static_cast<float>(f[(out_row + tm) * RANK + bn + t]); }
            for (int t = 0; t < 4; ++t) { rf[tm] += inter[t] * v[t]; }
        }
        for (int tm = 0; tm < 4; ++tm) {
            float inter[4];
            for (int t = 0; t < 4; ++t) { inter[t] = static_cast<float>(g[(out_row + tm) * RANK + bn + t]); }
            for (int t = 0; t < 4; ++t) { rg[tm] += inter[t] * v[t]; }
        }
        bn += 16;
    }
    for (int tm = 0; tm < 4; ++tm) {
        for (ushort sn = 2; sn >= 1; sn >>= 1) { rf[tm] += simd_shuffle_down(rf[tm], sn); }
    }
    for (int tm = 0; tm < 4; ++tm) {
        for (ushort sn = 2; sn >= 1; sn >>= 1) { rg[tm] += simd_shuffle_down(rg[tm], sn); }
    }
    if (thr_n == 0) {
        for (int tm = 0; tm < 4; ++tm) {
            const int row = out_row + tm;
            const float hv = h[row];
            const float tf = metal::precise::tanh(rf[tm]);
            const float a = tf * hv;
            const float b = hv - a;
            const float tg = metal::precise::tanh(rg[tm]);
            const float c = tg * m[row];
            const float o = b + c;
            out[row] = o;
        }
    }
"#;

impl MessageGate {
    /// SPEED-INK2M-P17 F2: the gate of one token in two kernels (same bits), or `None` outside their domain
    /// (several rows, other shapes or dtypes, BF16 tensors missing, the ablation, `MLXL3_INKLING_P17_OFF`).
    fn forward_fused(
        &self,
        hidden: &Array,
        hidden_norm: &Array,
        message_norm: &Array,
        message: &Array,
    ) -> Result<Option<Array>> {
        let (Some(forget), Some(gain)) = (&self.forget_bf16, &self.gain_bf16) else {
            return Ok(None);
        };
        if !p17("f2") || !fused() {
            return Ok(None);
        }
        let &[hid, rank] = forget.shape() else {
            return Ok(None);
        };
        let f32_row = |a: &Array| a.shape() == [1, hid] && a.dtype() == Dtype::Float32;
        let down = &self.down.weight_t;
        // MLX picks the bn 8 down gemv only when 2 hid >= 16 * rank (and > 64)
        if rank != 64
            || hid % 32 != 0
            || 2 * hid < 16 * rank
            || !(f32_row(hidden)
                && f32_row(hidden_norm)
                && f32_row(message_norm)
                && f32_row(message))
            || down.shape() != [2 * hid, rank]
            || down.dtype() != Dtype::Float16
        {
            return Ok(None);
        }
        #[cfg(test)]
        let mutate = GATE_TAIL_MUTATE.with(std::cell::Cell::get);
        #[cfg(not(test))]
        let mutate = false;
        let key = [
            i64::from(hid),
            i64::from(rank),
            i64::from(mutate),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let down_kernel = crate::array::prepared_kernel("inkling_gate_down", key, || {
            crate::array::PreparedKernel::new(
                &format!("inkling_gate_down_{hid}_{rank}"),
                &["hn", "mn", "w"],
                &["out"],
                "",
                &GATE_DOWN_SOURCE.replace("HID", &hid.to_string()),
            )
        })?;
        // `weight_t` is the `[2 hid, rank]` transposed view of the `[rank, 2 hid]` rows the gemv reads
        let rows = down.transpose(&[1, 0])?;
        let states = down_kernel
            .run(
                &[hidden_norm, message_norm, &rows],
                &[&[1, rank]],
                &[Dtype::Float32],
                [256 * (rank / 4), 1, 1],
                [256, 1, 1],
            )?
            .pop()
            .context("gate states")?;
        let tail_kernel = crate::array::prepared_kernel("inkling_gate_tail", key, || {
            let source = GATE_TAIL_SOURCE.replace("RANK", &rank.to_string());
            let source = if mutate {
                source.replace(
                    "for (ushort sn = 2; sn >= 1; sn >>= 1) { rf[tm]",
                    "for (ushort sn = 1; sn <= 2; sn <<= 1) { rf[tm]",
                )
            } else {
                source
            };
            crate::array::PreparedKernel::new(
                &format!(
                    "inkling_gate_tail_{hid}_{rank}{}",
                    if mutate { "_m" } else { "" }
                ),
                &["s", "f", "g", "h", "m"],
                &["out"],
                "",
                &source,
            )
        })?;
        let out = tail_kernel
            .run(
                &[&states, forget, gain, hidden, message],
                &[&[1, hid]],
                &[Dtype::Float32],
                [32 * (hid / 32), 1, 1],
                [32, 1, 1],
            )?
            .pop()
            .context("gate output")?;
        Ok(Some(out))
    }
}

struct Gates {
    q: MessageGate,
    k: MessageGate,
    v: MessageGate,
    mlp: MessageGate,
}

enum InputNorms {
    Shared(Array),
    Split { q: Array, k: Array, v: Array },
}

struct Moe {
    /// `[hidden, experts]` FP32.
    router_t: Array,
    /// SPEED-INK2M-P17 F3: the checkpoint's `[experts, hidden]` BF16 router (exact in FP32); `None` for another
    /// dtype.
    router_bf16: Option<Array>,
    bias: Array,
    experts: Exl3SwitchGlu,
    top_k: usize,
    scale: f32,
}

/// SPEED-INK2M-P17 F3: router logits `float(x) @ router_t` in one kernel, as MLX 0.32.2's `astype` and
/// `gemv_float32_bm1_bn8_sm1_sn32_tm4_tn4` (one threadgroup of 8 simdgroups per 4 experts; same thread
/// layout and operation order); the router is read in BF16, exact in FP32.
const ROUTER_SOURCE: &str = r#"
    const int lane = int(thread_index_in_simdgroup);
    const int sg = int(simdgroup_index_in_threadgroup);
    const int group = int(threadgroup_position_in_grid.x);
    threadgroup float part[8 * 8];
    const int out_row = group * 4;
    float result[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    int bn = (32 * sg + lane) * 4;
    for (int i = 0; i < HID / 1024; ++i) {
        float v[4];
        for (int t = 0; t < 4; ++t) { v[t] = static_cast<float>(x[bn + t]); }
        for (int tm = 0; tm < 4; ++tm) {
            float inter[4];
            for (int t = 0; t < 4; ++t) { inter[t] = static_cast<float>(w[(out_row + tm) * HID + bn + t]); }
            for (int t = 0; t < 4; ++t) { result[tm] += inter[t] * v[t]; }
        }
        bn += 1024;
    }
    if (HID % 1024 > 0) {
        float v[4];
        for (int t = 0; t < 4; ++t) { v[t] = bn + t < HID ? static_cast<float>(x[bn + t]) : 0.0f; }
        for (int tm = 0; tm < 4; ++tm) {
            float inter[4];
            for (int t = 0; t < 4; ++t) {
                inter[t] = bn + t < HID ? static_cast<float>(w[(out_row + tm) * HID + bn + t]) : 0.0f;
            }
            for (int t = 0; t < 4; ++t) { result[tm] += inter[t] * v[t]; }
        }
    }
    for (int tm = 0; tm < 4; ++tm) {
        for (ushort sn = 16; sn >= 1; sn >>= 1) { result[tm] += simd_shuffle_down(result[tm], sn); }
    }
    if (lane == 0) {
        for (int tm = 0; tm < 4; ++tm) { part[sg * 8 + tm] = result[tm]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane == 0) {
        for (int g = 1; g < 8; ++g) {
            for (int tm = 0; tm < 4; ++tm) { result[tm] += part[g * 8 + tm]; }
        }
        for (int tm = 0; tm < 4; ++tm) { out[out_row + tm] = result[tm]; }
    }
"#;

/// SPEED-2026-11-23-INK2M-MAYA RT: the router tail of one row in one kernel — MLX 0.32.2's `Sigmoid`
/// (`1 / (1 + exp(|x|))`, `x < 0 ? y : 1 - y`, precise exp and division), the biased top-k of
/// `router::grouped_topk_biased` (128 experts, one group: same threads, same comparisons), then the
/// scores' sum in the order of MLX's row reduction of 8 values, the division and the scale.
const ROUTER_TAIL_SOURCE: &str = r#"
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float probabilities[EXPERTS];
    threadgroup float routing[EXPERTS];
    threadgroup bool used[EXPERTS];
    threadgroup float candidate[EXPERTS];
    threadgroup float chosen[TOP_K];
    float x = logits[tid];
    float y = metal::precise::divide(1.0f, 1.0f + metal::precise::exp(metal::abs(x)));
    float p = (x < 0.0f) ? y : 1.0f - y;
    probabilities[tid] = p;
    routing[tid] = p + bias[tid];
    used[tid] = false;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint rank = 0u; rank < TOP_K; ++rank) {
        candidate[tid] = !used[tid] ? routing[tid] : -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float best = -INFINITY; uint winner = 0u;
            for (uint e = 0u; e < EXPERTS; ++e) {
                if (candidate[e] > best) { best = candidate[e]; winner = e; }
            }
            used[winner] = true;
            indices[rank] = winner;
            chosen[rank] = probabilities[winner];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) {
        float total = 0.0f;
        SUM_ORDER
        for (uint rank = 0u; rank < TOP_K; ++rank) {
            scores[rank] = metal::precise::divide(chosen[rank], total) * SCALE;
        }
    }
"#;

impl Moe {
    /// SPEED-2026-11-23-INK2M-MAYA RT: `(indices [1, k] u32, scores [1, k] f32)` of one row of router logits
    /// in one kernel, or `None` outside its domain (several rows, not 128 experts, `MLXL3_INKLING_P17_OFF`
    /// with `rt`).
    fn router_tail_fused(&self, logits: &Array) -> Result<Option<(Array, Array)>> {
        if !p17("rt") || logits.shape() != [1, 128] || logits.dtype() != Dtype::Float32 {
            return Ok(None);
        }
        if self.bias.shape() != [128] || self.bias.dtype() != Dtype::Float32 || self.top_k != 8 {
            return Ok(None);
        }
        router_tail_kernel(logits, &self.bias, self.scale).map(Some)
    }
}

/// The RT kernel (see `ROUTER_TAIL_SOURCE`) for 128 experts and top-8.
fn router_tail_kernel(logits: &Array, bias: &Array, scale: f32) -> Result<(Array, Array)> {
    ensure!(
        scale.is_finite() && scale > 0.0,
        "router tail needs a positive scale"
    );
    #[cfg(test)]
    let mutate = ROUTER_TAIL_MUTATE.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let mutate = false;
    // MLX's row reduction of 8 contiguous values: sequential from zero, `op(value, total)`
    let order = if mutate {
        "for (int r = int(TOP_K) - 1; r >= 0; --r) { total = chosen[r] + total; }"
    } else {
        "for (uint r = 0u; r < TOP_K; ++r) { total = chosen[r] + total; }"
    };
    let bits = scale.to_bits();
    let key = [
        i64::from(bits),
        i64::from(mutate),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ];
    let kernel = crate::array::prepared_kernel("inkling_router_tail", key, || {
        crate::array::PreparedKernel::new(
            &format!("inkling_router_tail_{bits:08x}_{}", u8::from(mutate)),
            &["logits", "bias"],
            &["indices", "scores"],
            "",
            &ROUTER_TAIL_SOURCE
                .replace("SUM_ORDER", order)
                .replace("EXPERTS", "128u")
                .replace("TOP_K", "8u")
                .replace("SCALE", &format!("as_type<float>({bits}u)")),
        )
    })?;
    let mut out = kernel.run(
        &[logits, bias],
        &[&[1, 8], &[1, 8]],
        &[Dtype::UInt32, Dtype::Float32],
        [128, 1, 1],
        [128, 1, 1],
    )?;
    let scores = out.pop().context("router tail scores")?;
    let indices = out.pop().context("router tail indices")?;
    Ok((indices, scores))
}

impl Moe {
    /// SPEED-INK2M-P17 F3: logits `[1, experts]` FP32 of one FP16 row, or `None` outside the kernel's domain
    /// (several rows, other dtypes or shapes, MLX would pick another gemv, `MLXL3_INKLING_P17_OFF`).
    fn router_logits_fused(&self, x: &Array) -> Result<Option<Array>> {
        let Some(w) = &self.router_bf16 else {
            return Ok(None);
        };
        if !p17("f3") {
            return Ok(None);
        }
        let &[experts, hid] = w.shape() else {
            return Ok(None);
        };
        // MLX picks the bn 8 gemv only when hidden > 64 and hidden >= 16 * experts
        if x.shape() != [1, hid]
            || x.dtype() != Dtype::Float16
            || experts % 4 != 0
            || hid <= 64
            || hid < 16 * experts
        {
            return Ok(None);
        }
        let key = [
            i64::from(experts),
            i64::from(hid),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        let kernel = crate::array::prepared_kernel("inkling_router_logits", key, || {
            crate::array::PreparedKernel::new(
                &format!("inkling_router_logits_{experts}_{hid}"),
                &["x", "w"],
                &["out"],
                "",
                &ROUTER_SOURCE.replace("HID", &hid.to_string()),
            )
        })?;
        let out = kernel
            .run(
                &[x, w],
                &[&[1, experts]],
                &[Dtype::Float32],
                [256 * (experts / 4), 1, 1],
                [256, 1, 1],
            )?
            .pop()
            .context("router logits")?;
        Ok(Some(out))
    }
}

impl Moe {
    /// Router logits `[rows, experts]`, selected experts and scores `[rows, top_k]`.
    fn route(&self, x: &Array) -> Result<(Array, Array, Array)> {
        let logits = match self.router_logits_fused(x)? {
            Some(logits) => logits,
            None => x.astype(Dtype::Float32)?.matmul(&self.router_t)?,
        };
        let (selected, scores) = match self.router_tail_fused(&logits)? {
            Some(routes) => routes,
            None => router::grouped_topk_biased(&logits, &self.bias, self.top_k, 1, 1, self.scale)?,
        };
        Ok((logits, selected, scores))
    }

    fn forward(&self, x: &Array) -> Result<Array> {
        if ablated("moe") {
            return x.scalar_mul(0.0);
        }
        let (logits, selected, scores) = if invariant_call(x.shape()[0]) {
            // INVARIANT-2026-11-24-INK2M-L1L4 (L2): every row routed by the 1-row router (its kernels over the
            // rows, or one row at a time outside their domain)
            match self.route_rows(x)? {
                Some(routes) => routes,
                None => {
                    let rows = x.shape()[0];
                    let mut parts = (Vec::new(), Vec::new(), Vec::new());
                    for r in 0..rows {
                        let (l, s, c) = self.route(&x.slice(0, r, r + 1)?)?;
                        parts.0.push(l);
                        parts.1.push(s);
                        parts.2.push(c);
                    }
                    let join =
                        |v: &Vec<Array>| Array::concatenate(&v.iter().collect::<Vec<_>>(), 0);
                    (join(&parts.0)?, join(&parts.1)?, join(&parts.2)?)
                }
            }
        } else {
            self.route(x)?
        };
        stage("m.logits", &logits)?;
        stage("m.selected", &selected)?;
        stage("m.scores", &scores)?;
        capture_routes(&selected)?;
        if ablated("experts") {
            // keep the router in the graph, drop the expert kernels
            return x
                .scalar_mul(0.0)?
                .add(
                    &scores
                        .sum(1, true)?
                        .astype(Dtype::Float16)?
                        .scalar_mul(0.0)?,
                )?
                .add(
                    &selected
                        .astype(Dtype::Float16)?
                        .sum(1, true)?
                        .scalar_mul(0.0)?,
                );
        }
        self.experts
            .forward(&x.astype(Dtype::Float16)?, &selected, &scores)
    }
}

struct Layer {
    attention_query: Option<Array>,
    mlp_query: Array,
    norms: InputNorms,
    post_attention_norm: Array,
    attention: Attention,
    post_attention_conv: ShortConv,
    post_mlp_conv: ShortConv,
    moe: Moe,
    gates: Option<Gates>,
    block_boundary: bool,
    sliding: bool,
    eps: f32,
}

impl Layer {
    fn load(checkpoint: &Checkpoint, index: usize, c: &TextConfig) -> Result<Self> {
        let prefix = format!("model.llm.layers.{index}");
        let hidden = c.hidden_size;
        let sliding = c.is_sliding(index)?;
        let attention_query = if index == 0 {
            None
        } else {
            let name = format!("model.llm.attn_res.attn_queries.{}", index - 1);
            let shape = if sliding {
                vec![1, hidden]
            } else {
                vec![3, hidden]
            };
            let raw = checkpoint_array(checkpoint, &name)?;
            ensure!(
                raw.shape().iter().product::<i32>() == shape.iter().product::<i32>(),
                "{name}: unexpected depth query shape {:?}",
                raw.shape()
            );
            let q = raw.astype(Dtype::Float32)?.reshape(&shape)?;
            q.eval()?;
            Some(q)
        };
        ensure!(
            sliding || index > 0,
            "full attention at layer 0 is unsupported"
        );
        let mlp_query = f32_weight(
            checkpoint,
            &format!("model.llm.attn_res.mlp_queries.{index}"),
            &[hidden],
        )?
        .reshape(&[1, hidden])?;
        let norm =
            |name: &str| f32_weight(checkpoint, &format!("{prefix}.{name}.weight"), &[hidden]);
        let norms = if sliding {
            InputNorms::Shared(norm("attn_norm")?)
        } else {
            InputNorms::Split {
                q: norm("q_input_norm")?,
                k: norm("k_input_norm")?,
                v: norm("v_input_norm")?,
            }
        };
        let looped = (c.loop_config.core_start..c.loop_config.core_end).contains(&index);
        let gates = if looped {
            let g = |name: &str| {
                MessageGate::load(
                    checkpoint,
                    &format!("{prefix}.controllers.{name}"),
                    hidden,
                    c.loop_config.gate_rank,
                    c.rms_norm_eps,
                )
            };
            Some(Gates {
                q: g("attn")?,
                k: g("k")?,
                v: g("v")?,
                mlp: g("mlp")?,
            })
        } else {
            None
        };
        let router = f32_weight(
            checkpoint,
            &format!("{prefix}.mlp.gate.weight"),
            &[c.n_routed_experts, hidden],
        )?
        .transpose(&[1, 0])?;
        router.eval()?;
        let router_bf16 = {
            let w = checkpoint_array(checkpoint, &format!("{prefix}.mlp.gate.weight"))?;
            if w.dtype() == Dtype::BFloat16 && w.shape() == [c.n_routed_experts, hidden] {
                w.eval()?;
                Some(w)
            } else {
                None
            }
        };
        let bias = f32_weight(
            checkpoint,
            &format!("{prefix}.mlp.gate.bias"),
            &[c.n_routed_experts],
        )?;
        Ok(Self {
            attention_query,
            mlp_query,
            norms,
            post_attention_norm: norm("mlp_norm")?,
            attention: Attention::load(checkpoint, index, sliding, c)?,
            post_attention_conv: ShortConv::load(
                checkpoint,
                &format!("{prefix}.attn_sconv.weight"),
                hidden,
                c.sconv_kernel_size,
            )?,
            post_mlp_conv: ShortConv::load(
                checkpoint,
                &format!("{prefix}.mlp_sconv.weight"),
                hidden,
                c.sconv_kernel_size,
            )?,
            moe: Moe {
                router_t: router,
                router_bf16,
                bias,
                experts: Exl3SwitchGlu::from_inkling_checkpoint(
                    checkpoint,
                    &format!("{prefix}.mlp.experts"),
                    c.n_routed_experts,
                    c.num_experts_per_tok as i32,
                    c.mlp_gate_softcap,
                    c.mlp_up_softcap,
                )?,
                top_k: c.num_experts_per_tok,
                scale: c.route_scale,
            },
            gates,
            block_boundary: index.is_multiple_of(c.n_layers_per_attn_res_block),
            sliding,
            eps: c.rms_norm_eps,
        })
    }

    /// One visit. `hidden`: layer input `[T, D]` FP32; `blocks`: normalized block states.
    /// `messages`: `[T, 4, D]` FP32 in the loop passes after the first.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        hidden: &Array,
        blocks: &mut Vec<Array>,
        messages: Option<&Array>,
        start: i32,
        plan: &MaskPlan,
        state: &mut VisitState,
    ) -> Result<Array> {
        let time = hidden.shape()[0];
        let dims = hidden.shape()[1];
        stage("in", hidden)?;
        let normed_input = hidden.rms_norm_without_weight(self.eps)?;
        stage("normed_input", &normed_input)?;
        let inputs: Vec<Array> = match &self.attention_query {
            Some(queries) => {
                let mut sources: Vec<&Array> = blocks.iter().collect();
                sources.push(&normed_input);
                let out = depth_attention(&sources, queries)?;
                (0..queries.shape()[0])
                    .map(|i| out.slice(1, i, i + 1)?.reshape(&[time, dims]))
                    .collect::<Result<_>>()?
            }
            None => vec![hidden.try_clone()?],
        };
        let message = |i: i32| -> Result<Option<Array>> {
            messages
                .map(|m| m.slice(1, i, i + 1)?.reshape(&[time, dims]))
                .transpose()
        };
        let (hq, hk, hv) = match &self.norms {
            InputNorms::Shared(weight) => {
                let base = &inputs[0];
                match (&self.gates, messages) {
                    (Some(g), Some(_)) => {
                        // the three gates read the same hidden state: normalize it once
                        let base_norm = base.rms_norm_without_weight(self.eps)?;
                        let gate = |g: &MessageGate, i: i32| -> Result<Array> {
                            let message = message(i)?.context("message")?;
                            if lean() {
                                g.forward_normed(base, &base_norm, &message)
                            } else {
                                g.forward(base, &message)
                            }?
                            .rms_norm(weight, self.eps)?
                            .astype(Dtype::Float16)
                        };
                        (gate(&g.q, 0)?, gate(&g.k, 1)?, gate(&g.v, 2)?)
                    }
                    _ => {
                        let n = base.rms_norm(weight, self.eps)?;
                        if lean() {
                            // one cast for the three identical inputs
                            let n = n.astype(Dtype::Float16)?;
                            (n.try_clone()?, n.try_clone()?, n)
                        } else {
                            (
                                n.astype(Dtype::Float16)?,
                                n.astype(Dtype::Float16)?,
                                n.astype(Dtype::Float16)?,
                            )
                        }
                    }
                }
            }
            InputNorms::Split { q, k, v } => {
                ensure!(inputs.len() == 3, "full attention needs three depth inputs");
                let (iq, ik, iv) = match (&self.gates, messages) {
                    (Some(g), Some(_)) => (
                        g.q.forward(&inputs[0], &message(0)?.context("message")?)?,
                        g.k.forward(&inputs[1], &message(1)?.context("message")?)?,
                        g.v.forward(&inputs[2], &message(2)?.context("message")?)?,
                    ),
                    _ => (
                        inputs[0].try_clone()?,
                        inputs[1].try_clone()?,
                        inputs[2].try_clone()?,
                    ),
                };
                (
                    iq.rms_norm(q, self.eps)?.astype(Dtype::Float16)?,
                    ik.rms_norm(k, self.eps)?.astype(Dtype::Float16)?,
                    iv.rms_norm(v, self.eps)?.astype(Dtype::Float16)?,
                )
            }
        };
        stage("dinput0", &inputs[0])?;
        stage("hq", &hq)?;
        stage("hk", &hk)?;
        stage("hv", &hv)?;
        let attended = self.attention.forward(&hq, &hk, &hv, start, plan, state)?;
        stage("attention", &attended)?;
        let (attended, conv_state) = self
            .post_attention_conv
            .forward(&attended, state.conv[2].as_ref())?;
        stage("post_attention_conv", &attended)?;
        state.pending_conv[2] = Some(conv_state);
        let residual = if self.block_boundary {
            blocks.push(normed_input);
            attended
        } else {
            hidden.add(&attended)?
        };
        let mut sources: Vec<&Array> = blocks.iter().collect();
        let normed = residual.rms_norm_without_weight(self.eps)?;
        sources.push(&normed);
        let mut mlp_input = depth_attention(&sources, &self.mlp_query)?.reshape(&[time, dims])?;
        if let (Some(g), Some(_)) = (&self.gates, messages) {
            mlp_input = g
                .mlp
                .forward(&mlp_input, &message(3)?.context("message")?)?;
        }
        stage("residual", &residual)?;
        stage("mlp_input", &mlp_input)?;
        let moe_in = mlp_input
            .rms_norm(&self.post_attention_norm, self.eps)?
            .astype(Dtype::Float16)?;
        stage("moe_in", &moe_in)?;
        let mlp = self.moe.forward(&moe_in)?;
        stage("moe_out", &mlp)?;
        let (mlp, conv_state) = self.post_mlp_conv.forward(&mlp, state.conv[3].as_ref())?;
        stage("post_mlp_conv", &mlp)?;
        state.pending_conv[3] = Some(conv_state);
        let out = residual.add(&mlp)?;
        stage("out", &out)?;
        Ok(out)
    }
}

/// Input tokens of one `run`: host ids, or one id still on the GPU (`[1]` integers).
#[derive(Clone, Copy)]
enum Ids<'a> {
    Host(&'a [u32]),
    Device(&'a Array),
}

/// SPEED-INK2M-P14: a token choice computed on the GPU inside the decode graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DrawRule {
    /// argmax of the log probabilities: the bridge's greedy rule (`Array::chat_greedy_ids`).
    Greedy,
    /// One draw from softmax(log probabilities / temperature) over the whole vocabulary (Gumbel-max).
    Softmax(f32),
}

impl DrawRule {
    /// The rule giving the same law as `sampler`, or `None` where the CPU sampler must stay (penalties,
    /// top-k, top-p, a non-positive or non-finite temperature).
    pub fn for_sampler(sampler: &crate::sampling::Sampler) -> Option<Self> {
        if sampler.is_greedy() {
            return Some(Self::Greedy);
        }
        (sampler.top_k == 0
            && sampler.top_p >= 1.0
            && sampler.repetition_penalty == 1.0
            && sampler.presence_penalty == 0.0
            && sampler.temperature.is_finite()
            && sampler.temperature > 0.0)
            .then_some(Self::Softmax(sampler.temperature))
    }
}

/// One token per row of `[rows, vocab]` logits under `rule`: `[rows]` u32, lazy. Softmax rows add standard
/// Gumbel noise (`Array::gumbel_noise`, seeded by `seed`) to log_probs / temperature: each row is one draw
/// from softmax(log_probs / temperature), up to FP32 rounding of the scores.
pub fn draw_rows(logits: &Array, rule: DrawRule, seed: u64) -> Result<Array> {
    ensure!(
        logits.shape().len() == 2,
        "draw_rows expects [rows, vocab] logits"
    );
    let log_probs = logits.log_probs()?;
    match rule {
        DrawRule::Greedy => log_probs.argmax(),
        DrawRule::Softmax(temperature) => {
            ensure!(
                temperature.is_finite() && temperature > 0.0,
                "a softmax draw needs a positive finite temperature"
            );
            let noise = Array::gumbel_noise(log_probs.shape(), seed)?;
            let scores = if temperature == 1.0 {
                log_probs
            } else {
                log_probs.scalar_mul(1.0 / temperature)?
            };
            scores.add(&noise)?.argmax()
        }
    }
}

/// The token after the last row of `logits` (`[1, rows, vocab]` or `[rows, vocab]`) under `rule`: `[1]`
/// u32, lazy.
pub fn draw_token(logits: &Array, rule: DrawRule, seed: u64) -> Result<Array> {
    let shape = logits.shape();
    ensure!(
        shape.len() >= 2 && shape[..shape.len() - 2].iter().all(|&d| d == 1),
        "draw_token expects [1, rows, vocab] logits"
    );
    let (rows, vocab) = (shape[shape.len() - 2], shape[shape.len() - 1]);
    ensure!(rows >= 1 && vocab >= 1, "draw_token needs at least one row");
    let last = logits.reshape(&[rows, vocab])?.slice(0, rows - 1, rows)?;
    draw_rows(&last, rule, seed)
}

/// DSPARK-2026-11-23-INK2M-MTPLX-A: threads of one row reduction in `decide_block` (one threadgroup a row).
const DECIDE_THREADS: i32 = 1024;

/// Per verified row: argmax over the vocabulary of `scores * inv_t + noise` excluding the row's proposal
/// (the bonus row excludes nothing), the log-sum-exp of `scores * inv_t` and `scores * inv_t` at the
/// proposal. One threadgroup of `DECIDE_THREADS` a row; ties go to the lowest index.
const DECIDE_ROWS_SOURCE: &str = r#"
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint excluded = row + 1u < uint(ROWS) ? proposals[row] : 0xffffffffu;
    const float it = inv_t[0];
    const device float* s = scores + size_t(row) * uint(VOCAB);
    const device float* g = noise + size_t(row) * uint(VOCAB);
    float m = -INFINITY;
    float acc = 0.0f;
    float bv = -INFINITY;
    uint bi = 0xffffffffu;
    for (uint j = tid; j < uint(VOCAB); j += uint(THREADS)) {
        const float x = s[j] * it;
        if (x > m) {
            acc = acc * metal::exp(m - x) + 1.0f;
            m = x;
        } else if (x > -INFINITY) {
            acc += metal::exp(x - m);
        }
        if (j != excluded) {
            const float k = x + g[j];
            if (k > bv || bi == 0xffffffffu) {
                bv = k;
                bi = j;
            }
        }
    }
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        const float om = simd_shuffle_down(m, offset);
        const float oa = simd_shuffle_down(acc, offset);
        const float ov = simd_shuffle_down(bv, offset);
        const uint oi = simd_shuffle_down(bi, offset);
        if (lane + offset < 32u) {
            const float mm = metal::max(m, om);
            acc = (mm == -INFINITY) ? 0.0f : acc * metal::exp(m - mm) + oa * metal::exp(om - mm);
            m = mm;
            if (oi != 0xffffffffu && (bi == 0xffffffffu || ov > bv || (ov == bv && oi < bi))) {
                bv = ov;
                bi = oi;
            }
        }
    }
    threadgroup float sm[32];
    threadgroup float sa[32];
    threadgroup float sv[32];
    threadgroup uint si[32];
    if (lane == 0u) {
        sm[sg] = m;
        sa[sg] = acc;
        sv[sg] = bv;
        si[sg] = bi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tm = sm[0];
        float ta = sa[0];
        float tv = sv[0];
        uint ti = si[0];
        for (uint i = 1u; i < uint(THREADS) / 32u; ++i) {
            const float mm = metal::max(tm, sm[i]);
            ta = (mm == -INFINITY) ? 0.0f : ta * metal::exp(tm - mm) + sa[i] * metal::exp(sm[i] - mm);
            tm = mm;
            if (si[i] != 0xffffffffu && (ti == 0xffffffffu || sv[i] > tv || (sv[i] == tv && si[i] < ti))) {
                tv = sv[i];
                ti = si[i];
            }
        }
        best[row] = ti;
        lse[row] = tm + metal::log(ta);
        at[row] = excluded < uint(VOCAB) ? s[excluded] * it : -INFINITY;
    }
"#;

/// The block's decision in one thread: proposal i is kept when log U_i < log p_i(x_i); at the first
/// rejection the token is that row's draw without the proposal, after every proposal the bonus row's draw.
/// `packet` = [accepted, token, proposals...].
const DECIDE_SOFTMAX_SOURCE: &str = r#"
    uint accepted = uint(PROPOSALS);
    for (uint i = 0u; i < uint(PROPOSALS); ++i) {
        if (!(metal::log(coins[i]) < at[i] - lse[i])) {
            accepted = i;
            break;
        }
    }
    packet[0] = accepted;
    packet[1] = best[accepted];
    for (uint i = 0u; i < uint(PROPOSALS); ++i) {
        packet[2u + i] = proposals[i];
    }
"#;

/// Greedy decision: proposals kept while they equal the argmax of their row, then the next row's argmax.
const DECIDE_GREEDY_SOURCE: &str = r#"
    uint accepted = uint(PROPOSALS);
    for (uint i = 0u; i < uint(PROPOSALS); ++i) {
        if (top[i] != proposals[i]) {
            accepted = i;
            break;
        }
    }
    packet[0] = accepted;
    packet[1] = top[accepted];
    for (uint i = 0u; i < uint(PROPOSALS); ++i) {
        packet[2u + i] = proposals[i];
    }
"#;

/// DSPARK-2026-11-23-INK2M-MTPLX-A: decision of a DSpark block on the GPU. `logits` `[1, 1 + k, vocab]` from
/// `verify_device` of the anchor and the `k` deterministic `proposals` (`[k]` u32 on the GPU); returns the
/// lazy packet `[2 + k]` u32 = accepted count, the target token after them, then the proposals.
///
/// Greedy: the bridge's rule (`speculative::greedy_accept` over `Array::chat_greedy_ids`, same operations).
/// Softmax(T): the point-mass rule of `Sampler::accept_with` for the law softmax(log_probs / T): proposal i
/// is accepted with probability p_i(x_i) (uniform from `uniform_seed`), the token after a rejection is drawn
/// from p_i without x_i, the token after every proposal from the bonus row (Gumbel-max, noise from
/// `noise_seed`). Every token has the law of an ordinary draw from the same rows, up to FP32 rounding; the
/// draws of a given seed differ from the CPU path's.
pub fn decide_block(
    logits: &Array,
    proposals: &Array,
    rule: DrawRule,
    noise_seed: u64,
    uniform_seed: u64,
) -> Result<Array> {
    let shape = logits.shape();
    ensure!(
        shape.len() == 3 && shape[0] == 1 && shape[1] >= 2 && shape[2] >= 2,
        "decide_block expects [1, 1 + k, vocab] logits with k >= 1"
    );
    let (rows, vocab) = (shape[1], shape[2]);
    let k = rows - 1;
    ensure!(
        proposals.shape() == [k] && proposals.dtype() == Dtype::UInt32,
        "decide_block expects [k] u32 proposals"
    );
    let log_probs = logits.reshape(&[rows, vocab])?.log_probs()?;
    let defines = format!("#define ROWS {rows}\n#define PROPOSALS {k}\n#define VOCAB {vocab}\n");
    let packet = match rule {
        DrawRule::Greedy => {
            return decide_equal(
                &log_probs.argmax()?.astype(Dtype::UInt32)?,
                proposals,
                &defines,
            );
        }
        DrawRule::Softmax(temperature) => {
            ensure!(
                temperature.is_finite() && temperature > 0.0,
                "a softmax decision needs a positive finite temperature"
            );
            let noise = Array::gumbel_noise(&[rows, vocab], noise_seed)?;
            let inv_t = Array::from_f32(&[1.0 / temperature], &[1])?;
            let mut reduced = crate::array::metal_kernel(
                &format!("mlxl3_inkling_dspark_decide_rows_v1_r{rows}_v{vocab}"),
                &["scores", "noise", "proposals", "inv_t"],
                &["best", "lse", "at"],
                &format!("{defines}#define THREADS {DECIDE_THREADS}\n"),
                DECIDE_ROWS_SOURCE,
                &[&log_probs, &noise, proposals, &inv_t],
                &[vec![rows], vec![rows], vec![rows]],
                &[Dtype::UInt32, Dtype::Float32, Dtype::Float32],
                [DECIDE_THREADS, rows, 1],
                [DECIDE_THREADS, 1, 1],
            )?;
            ensure!(reduced.len() == 3, "missing decision rows");
            let at = reduced.pop().context("at")?;
            let lse = reduced.pop().context("lse")?;
            let best = reduced.pop().context("best")?;
            let coins = Array::random_uniform(&[k], uniform_seed)?;
            crate::array::metal_kernel(
                &format!("mlxl3_inkling_dspark_decide_softmax_v1_k{k}"),
                &["best", "lse", "at", "coins", "proposals"],
                &["packet"],
                &defines,
                DECIDE_SOFTMAX_SOURCE,
                &[&best, &lse, &at, &coins, proposals],
                &[vec![2 + k]],
                &[Dtype::UInt32],
                [1, 1, 1],
                [1, 1, 1],
            )?
        }
    }
    .pop()
    .context("missing DSpark decision packet")?;
    Ok(packet)
}

/// The packet of a block whose verified rows chose `top` (`[1 + k]` u32): proposals kept while they equal
/// their row's choice, then the next row's choice (`DECIDE_GREEDY_SOURCE`).
fn decide_equal(top: &Array, proposals: &Array, defines: &str) -> Result<Array> {
    let k = proposals.shape()[0];
    ensure!(
        top.shape() == [k + 1] && top.dtype() == Dtype::UInt32,
        "decide_equal expects [1 + k] u32 choices"
    );
    crate::array::metal_kernel(
        &format!("mlxl3_inkling_dspark_decide_greedy_v1_k{k}"),
        &["top", "proposals"],
        &["packet"],
        defines,
        DECIDE_GREEDY_SOURCE,
        &[top, proposals],
        &[vec![2 + k]],
        &[Dtype::UInt32],
        [1, 1, 1],
        [1, 1, 1],
    )?
    .pop()
    .context("missing DSpark decision packet")
}

/// DSPARK-2026-11-24-INK2M-MAYA2-D2: the coupled decision of a sampled DSpark block. `noise` `[1 + k, vocab]`
/// is the block's standard Gumbel noise, the one the drafter drew its proposals with (`coupled_pick`). Row i
/// chooses `t_i = argmax(log_probs_i / T + noise_i)`, one draw from softmax(log_probs_i / T) (the operations
/// of `draw_rows`); proposals are kept while they equal their row's draw and the token after them is the
/// next row's draw. Row i's logits depend on the noise of rows < i only (through the proposals before it), so
/// every delivered token has the law of an ordinary draw from its row, whatever the drafter proposed. Returns
/// the lazy packet `[2 + k]` u32 of `decide_block`.
pub fn decide_coupled(
    logits: &Array,
    proposals: &Array,
    temperature: f32,
    noise: &Array,
) -> Result<Array> {
    let shape = logits.shape();
    ensure!(
        shape.len() == 3 && shape[0] == 1 && shape[1] >= 2 && shape[2] >= 2,
        "decide_coupled expects [1, 1 + k, vocab] logits with k >= 1"
    );
    let (rows, vocab) = (shape[1], shape[2]);
    let k = rows - 1;
    ensure!(
        proposals.shape() == [k] && proposals.dtype() == Dtype::UInt32,
        "decide_coupled expects [k] u32 proposals"
    );
    ensure!(
        noise.shape() == [rows, vocab] && noise.dtype() == Dtype::Float32,
        "decide_coupled expects [1 + k, vocab] FP32 noise"
    );
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "a coupled decision needs a positive finite temperature"
    );
    let log_probs = logits.reshape(&[rows, vocab])?.log_probs()?;
    let scores = if temperature == 1.0 {
        log_probs
    } else {
        log_probs.scalar_mul(1.0 / temperature)?
    };
    #[cfg(test)]
    if COUPLE_MUTATION.with(std::cell::Cell::get) {
        // mutation: every proposal is kept, the drafter's draws are delivered as the target's
        let top = Array::concatenate(
            &[
                proposals,
                &scores
                    .add(noise)?
                    .argmax()?
                    .astype(Dtype::UInt32)?
                    .slice(0, k, k + 1)?,
            ],
            0,
        )?;
        let defines =
            format!("#define ROWS {rows}\n#define PROPOSALS {k}\n#define VOCAB {vocab}\n");
        return decide_equal(&top, proposals, &defines);
    }
    let top = scores.add(noise)?.argmax()?.astype(Dtype::UInt32)?;
    let defines = format!("#define ROWS {rows}\n#define PROPOSALS {k}\n#define VOCAB {vocab}\n");
    decide_equal(&top, proposals, &defines)
}

#[cfg(test)]
thread_local! {
    /// DFLASH2-2026-11-24-INK2M-S3 test mutation: `decide_ratio` keeps every proposal.
    pub(crate) static RATIO_MUTATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Per verified row of a ratio block: argmax over the vocabulary of log r(v) + noise(v), r = max(p - q, 0) on
/// the row's candidates and r = p elsewhere (the residual of the rejection of its proposal; the bonus row has no
/// candidates: r = p), the plain draw argmax(log p + noise) and log p at the proposal. `log_probs` normalized at
/// the temperature. One threadgroup of `DECIDE_THREADS` a row; ties go to the lowest index.
const RATIO_ROWS_SOURCE: &str = r#"
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const bool drafted = row < uint(PROPOSALS);
    uint cand[CANDIDATES];
    float qc[CANDIDATES];
    for (uint c = 0u; c < uint(CANDIDATES); ++c) {
        cand[c] = drafted ? candidates[row * uint(CANDIDATES) + c] : 0xffffffffu;
        qc[c] = drafted ? q[row * uint(CANDIDATES) + c] : 0.0f;
    }
    const device float* s = log_probs + size_t(row) * uint(VOCAB);
    const device float* g = noise + size_t(row) * uint(VOCAB);
    float rv = -INFINITY;
    uint ri = 0xffffffffu;
    float pv = -INFINITY;
    uint pi = 0xffffffffu;
    for (uint j = tid; j < uint(VOCAB); j += uint(THREADS)) {
        const float lp = s[j];
        const float key = lp + g[j];
        if (pi == 0xffffffffu || key > pv) {
            pv = key;
            pi = j;
        }
        float r = lp;
        for (uint c = 0u; c < uint(CANDIDATES); ++c) {
            if (cand[c] == j) {
                const float d = metal::exp(lp) - qc[c];
                r = d > 0.0f ? metal::log(d) : -INFINITY;
            }
        }
        const float rk = r + g[j];
        if (rk > -INFINITY && (ri == 0xffffffffu || rk > rv)) {
            rv = rk;
            ri = j;
        }
    }
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        const float orv = simd_shuffle_down(rv, offset);
        const uint ori = simd_shuffle_down(ri, offset);
        const float opv = simd_shuffle_down(pv, offset);
        const uint opi = simd_shuffle_down(pi, offset);
        if (lane + offset < 32u) {
            if (ori != 0xffffffffu && (ri == 0xffffffffu || orv > rv || (orv == rv && ori < ri))) {
                rv = orv;
                ri = ori;
            }
            if (opi != 0xffffffffu && (pi == 0xffffffffu || opv > pv || (opv == pv && opi < pi))) {
                pv = opv;
                pi = opi;
            }
        }
    }
    threadgroup float srv[32];
    threadgroup uint sri[32];
    threadgroup float spv[32];
    threadgroup uint spi[32];
    if (lane == 0u) {
        srv[sg] = rv;
        sri[sg] = ri;
        spv[sg] = pv;
        spi[sg] = pi;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tr = srv[0];
        uint tri = sri[0];
        float tp = spv[0];
        uint tpi = spi[0];
        for (uint i = 1u; i < uint(THREADS) / 32u; ++i) {
            if (sri[i] != 0xffffffffu && (tri == 0xffffffffu || srv[i] > tr || (srv[i] == tr && sri[i] < tri))) {
                tr = srv[i];
                tri = sri[i];
            }
            if (spi[i] != 0xffffffffu && (tpi == 0xffffffffu || spv[i] > tp || (spv[i] == tp && spi[i] < tpi))) {
                tp = spv[i];
                tpi = spi[i];
            }
        }
        residual[row] = tri;
        plain[row] = tpi;
        at[row] = drafted ? s[proposals[row]] : -INFINITY;
    }
"#;

/// The ratio block's decision in one thread: proposal i is kept when U_i q_i(x_i) < p_i(x_i) (probability
/// min(1, p / q)); at the first rejection the token is that row's residual draw (its plain draw if the residual is
/// numerically empty), after every proposal the bonus row's draw. `packet` = [accepted, token, proposals...].
const RATIO_DECIDE_SOURCE: &str = r#"
    uint accepted = uint(PROPOSALS);
    for (uint i = 0u; i < uint(PROPOSALS); ++i) {
        if (RATIO_KEEP_ALL == 0 && !(metal::log(coins[i]) + metal::log(chosen[i]) < at[i])) {
            accepted = i;
            break;
        }
    }
    uint token = plain[accepted];
    if (accepted < uint(PROPOSALS) && residual[accepted] != 0xffffffffu) {
        token = residual[accepted];
    }
    packet[0] = accepted;
    packet[1] = token;
    for (uint i = 0u; i < uint(PROPOSALS); ++i) {
        packet[2u + i] = proposals[i];
    }
"#;

/// DFLASH2-2026-11-24-INK2M-S3: the ratio rule of IncoAI Splash for a sampled block (speculative sampling of a
/// chain of draws): proposal i, drawn from the drafter's law q_i over its candidates (`candidates` `[k, C]`, `q`
/// `[k, C]`, `chosen` = q_i(x_i) `[k]`; row i conditioned on the proposals before it), is kept with probability
/// min(1, p_i(x_i) / q_i(x_i)) (`uniform_seed`); at the first rejection the token is drawn from the residual
/// max(p_i - q_i, 0) (`noise_seed`), after every proposal from the bonus row. p_i = softmax(logits_i / T) of
/// `logits` `[1, 1 + k, vocab]`. Every delivered token has the law of an ordinary draw from its row, whatever the
/// drafter proposed, provided the proposals were drawn from `q`. Returns the lazy packet `[2 + k]` u32 of
/// `decide_block`.
#[allow(clippy::too_many_arguments)]
pub fn decide_ratio(
    logits: &Array,
    proposals: &Array,
    candidates: &Array,
    q: &Array,
    chosen: &Array,
    temperature: f32,
    noise_seed: u64,
    uniform_seed: u64,
) -> Result<Array> {
    let shape = logits.shape();
    ensure!(
        shape.len() == 3 && shape[0] == 1 && shape[1] >= 2 && shape[2] >= 2,
        "decide_ratio expects [1, 1 + k, vocab] logits with k >= 1"
    );
    let (rows, vocab) = (shape[1], shape[2]);
    let k = rows - 1;
    let c = candidates.shape().get(1).copied().unwrap_or(0);
    ensure!(
        proposals.shape() == [k] && proposals.dtype() == Dtype::UInt32,
        "decide_ratio expects [k] u32 proposals"
    );
    ensure!(
        (1..=32).contains(&c)
            && candidates.shape() == [k, c]
            && candidates.dtype() == Dtype::UInt32
            && q.shape() == [k, c]
            && q.dtype() == Dtype::Float32
            && chosen.shape() == [k]
            && chosen.dtype() == Dtype::Float32,
        "decide_ratio expects [k, C] u32 candidates, [k, C] FP32 q and [k] FP32 chosen (C <= 32)"
    );
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "a ratio decision needs a positive finite temperature"
    );
    let scaled = logits.reshape(&[rows, vocab])?.astype(Dtype::Float32)?;
    let scaled = if temperature == 1.0 {
        scaled
    } else {
        scaled.scalar_mul(1.0 / temperature)?
    };
    let log_probs = scaled.log_probs()?;
    let noise = Array::gumbel_noise(&[rows, vocab], noise_seed)?;
    let defines = format!(
        "#define ROWS {rows}\n#define PROPOSALS {k}\n#define VOCAB {vocab}\n#define CANDIDATES {c}\n"
    );
    let mut reduced = crate::array::metal_kernel(
        &format!("mlxl3_inkling_dflash2_ratio_rows_v1_r{rows}_v{vocab}_c{c}"),
        &["log_probs", "noise", "candidates", "q", "proposals"],
        &["residual", "plain", "at"],
        &format!("{defines}#define THREADS {DECIDE_THREADS}\n"),
        RATIO_ROWS_SOURCE,
        &[&log_probs, &noise, candidates, q, proposals],
        &[vec![rows], vec![rows], vec![rows]],
        &[Dtype::UInt32, Dtype::UInt32, Dtype::Float32],
        [DECIDE_THREADS, rows, 1],
        [DECIDE_THREADS, 1, 1],
    )?;
    ensure!(reduced.len() == 3, "missing ratio decision rows");
    let at = reduced.pop().context("at")?;
    let plain = reduced.pop().context("plain")?;
    let residual = reduced.pop().context("residual")?;
    let coins = Array::random_uniform(&[k], uniform_seed)?;
    #[cfg(test)]
    let keep_all = RATIO_MUTATION.with(std::cell::Cell::get);
    #[cfg(not(test))]
    let keep_all = false;
    crate::array::metal_kernel(
        &format!(
            "mlxl3_inkling_dflash2_ratio_decide_v1_k{k}_m{}",
            u8::from(keep_all)
        ),
        &["residual", "plain", "at", "coins", "chosen", "proposals"],
        &["packet"],
        &format!("{defines}#define RATIO_KEEP_ALL {}\n", u8::from(keep_all)),
        RATIO_DECIDE_SOURCE,
        &[&residual, &plain, &at, &coins, chosen, proposals],
        &[vec![2 + k]],
        &[Dtype::UInt32],
        [1, 1, 1],
        [1, 1, 1],
    )?
    .pop()
    .context("missing ratio decision packet")
}

/// DSPARK-2026-11-24-INK2M-MAYA2-D2: the drafter's coupled proposal at one position: `[1]` u32
/// `argmax(logits / T + noise[..V])` over its `[1, V]` logits, with the noise row `[1, V_target]` the target
/// draws that position with (`decide_coupled`).
pub fn coupled_pick(logits: &Array, temperature: f32, noise_row: &Array) -> Result<Array> {
    let shape = logits.shape();
    ensure!(
        shape.len() == 2 && shape[0] == 1,
        "coupled_pick expects [1, vocab] logits"
    );
    ensure!(
        noise_row.shape().len() == 2
            && noise_row.shape()[0] == 1
            && noise_row.shape()[1] >= shape[1],
        "coupled_pick noise narrower than the draft vocabulary"
    );
    ensure!(
        temperature.is_finite() && temperature > 0.0,
        "a coupled proposal needs a positive finite temperature"
    );
    let scores = logits.astype(Dtype::Float32)?;
    let scores = if temperature == 1.0 {
        scores
    } else {
        scores.scalar_mul(1.0 / temperature)?
    };
    let noise = if noise_row.shape()[1] == shape[1] {
        noise_row.try_clone()?
    } else {
        noise_row.slice(1, 0, shape[1])?
    };
    scores.add(&noise)?.argmax()?.astype(Dtype::UInt32)
}

#[cfg(test)]
thread_local! {
    /// D2 oracle mutation: the coupled decision keeps every proposal.
    pub(crate) static COUPLE_MUTATION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Seeds of successive GPU draws: splitmix64 of a Weyl sequence started at `seed`.
#[derive(Clone, Debug)]
pub struct DrawSeeds(u64);

impl DrawSeeds {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_seed(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// SPEED-INK2M-P14: plain decoding with every token chosen on the GPU. When `pipelined`, the step that
/// consumes a token is built and submitted before the token is read, so the GPU computes it while the
/// caller handles the token (one step ahead).
pub struct DeviceDecode {
    rule: DrawRule,
    seeds: DrawSeeds,
    pipelined: bool,
    /// The next token to deliver, on the GPU.
    next: Array,
    /// The token after `next`, when the step consuming `next` was already submitted.
    after: Option<Array>,
}

impl DeviceDecode {
    /// Draws the first token from the prompt logits (`[1, rows, vocab]`, last row used).
    pub fn start(
        logits: &Array,
        rule: DrawRule,
        seeds: DrawSeeds,
        pipelined: bool,
    ) -> Result<Self> {
        let mut seeds = seeds;
        let next = draw_token(logits, rule, seeds.next_seed())?;
        if pipelined {
            next.async_eval()?;
        }
        Ok(Self {
            rule,
            seeds,
            pipelined,
            next,
            after: None,
        })
    }

    /// Delivers the next token. `more`: a token will be delivered after this one, so the step consuming
    /// it runs (exactly when the plain loop would call `forward` with it). Without `more`, `target` is not
    /// touched.
    pub fn next_token(&mut self, target: &mut Inkling2, more: bool) -> Result<u32> {
        if more && self.pipelined && self.after.is_none() {
            self.after = Some(self.step(target)?);
        }
        let token = single_token(&self.next)?;
        if more {
            let after = match self.after.take() {
                Some(after) => after,
                None => self.step(target)?,
            };
            self.next = after;
        }
        Ok(token)
    }

    /// The committed decode step of `next`, and the token drawn from its logits.
    fn step(&mut self, target: &mut Inkling2) -> Result<Array> {
        let (logits, _) = target.forward_device(&self.next, None, self.pipelined)?;
        let token = draw_token(&logits, self.rule, self.seeds.next_seed())?;
        if self.pipelined {
            token.async_eval()?;
        }
        Ok(token)
    }
}

/// The value of a `[1]` token array (waits for it).
pub(crate) fn single_token(token: &Array) -> Result<u32> {
    let values = token.to_u32()?;
    ensure!(values.len() == 1, "expected one drawn token");
    Ok(values[0])
}

/// Time spent in `run` since the last `take_profile`: graph construction (CPU) and evaluation.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunProfile {
    pub calls: u64,
    pub build: Duration,
    pub eval: Duration,
}

/// Default MLX buffer-cache limit while Inkling 2 is loaded (APP-F1).
pub const INKLING_CACHE_LIMIT: u64 = 1 << 30;

pub struct Inkling2 {
    profile: RunProfile,
    /// Rows of an uncommitted `verify` call.
    pending: Option<i32>,
    kernel: i32,
    embeddings: Embeddings,
    embed_norm: Array,
    norm: Array,
    final_query: Array,
    message_queries: Array,
    /// Dense FP16 head (padded rows); `None` when the checkpoint stores only the affine copy.
    head: Option<Projection>,
    /// Affine copy of the unpadded head rows when `precision.head_bits < 16`.
    head_q: Option<Linear>,
    precision: Precision,
    layers: Vec<Layer>,
    visits: Vec<usize>,
    states: Vec<VisitState>,
    core: (usize, usize),
    passes: usize,
    vocab: i32,
    unpadded_vocab: i32,
    hidden: i32,
    eps: f32,
    softcap: f32,
    window: i32,
    extent: i32,
    context_limit: i32,
    offset: i32,
    /// SPEED-2026-11-23-INK2M-MAYA PC: lineage of the cache rows, renewed by every `reset` (process-wide
    /// unique), so a saved prefix only resumes the state it was taken from.
    epoch: u64,
    /// Vision encoder: absent from the checkpoint, loadable on the first image, or loaded.
    vision: VisionSlot,
    /// Image features of the current prompt: `(first position, rows [n, D] FP32)` in position order. They
    /// replace the embeddings of those positions; `reset` and `prepare_images` clear them.
    images: Vec<(i32, Array)>,
}

/// Where the vision encoder of an Inkling checkpoint is.
enum VisionSlot {
    Absent,
    /// The checkpoint directory; loaded by the first `prepare_images` with pictures.
    Available(std::path::PathBuf),
    Loaded(Box<crate::inkling2_vision::Vision>),
}

/// Process-wide source of `Inkling2::epoch` values.
fn next_epoch() -> u64 {
    static EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// SPEED-2026-11-23-INK2M-MAYA PC: the committed state at a prefill chunk boundary, kept for the next turn
/// of the conversation (bridge prompt cache). Full visits stored in place keep only their row count: their
/// rows `[1, 1 + len)` are never rewritten afterwards (fixed zero sink at row 0, growth copies the rows),
/// so the live buffers still hold them when the next turn resumes. Every other visit keeps references of
/// its arrays at the capture point (MLX copies a buffer on the next write while it is referenced here).
pub struct InklingPrefix {
    epoch: u64,
    offset: i32,
    visits: Vec<PrefixVisit>,
}

struct PrefixVisit {
    kv: PrefixKv,
    conv: [Option<Array>; 4],
}

enum PrefixKv {
    /// Full visit stored in place: the committed row count at the capture point.
    Rows(i32),
    /// Keys/values views and the in-place buffer (if any) at the capture point.
    Saved {
        keys: Option<Array>,
        values: Option<Array>,
        buffer: Option<(Array, Array, i32, i32, i32)>,
    },
}

impl InklingPrefix {
    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Bytes this prefix keeps alive beyond the live state (saved buffers and convolution windows).
    pub fn retained_bytes(&self) -> Result<usize> {
        let mut bytes = 0usize;
        for visit in &self.visits {
            if let PrefixKv::Saved {
                keys,
                values,
                buffer,
            } = &visit.kv
            {
                let arrays: Vec<&Array> = match buffer {
                    Some((k, v, ..)) => vec![k, v],
                    None => [keys, values].into_iter().flatten().collect(),
                };
                for a in arrays {
                    bytes = bytes
                        .checked_add(a.retained_bytes()?)
                        .context("Inkling prefix size overflow")?;
                }
            }
            for a in visit.conv.iter().flatten() {
                bytes = bytes
                    .checked_add(a.retained_bytes()?)
                    .context("Inkling prefix size overflow")?;
            }
        }
        Ok(bytes)
    }
}

impl Inkling2 {
    pub fn load(path: &Path) -> Result<Self> {
        let checkpoint = checkpoint::inspect(path)?;
        Self::from_checkpoint(&checkpoint)
    }

    pub fn load_with(path: &Path, precision: Precision) -> Result<Self> {
        Self::from_checkpoint_with(&checkpoint::inspect(path)?, precision)
    }

    /// Precision from the environment (`Precision::from_env`): FP16 projections and an 8-bit head by default.
    pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
        Self::from_checkpoint_with(checkpoint, Precision::from_env()?)
    }

    pub fn from_checkpoint_with(checkpoint: &Checkpoint, precision: Precision) -> Result<Self> {
        // APP-F1: without a limit the MLX buffer cache grew by 3.4-3.9 GB during a long generation
        // (`MLXL3_ALLOCATOR_CACHE_MIB` still overrides it)
        crate::array::set_default_cache_limit(INKLING_CACHE_LIMIT)?;
        let precision = precision.validate()?;
        let root: RootConfig =
            serde_json::from_reader(File::open(checkpoint.path.join("config.json"))?)?;
        ensure!(
            root.model_type == "inkling2_loop",
            "expected Inkling 2 (inkling2_loop)"
        );
        let mut c = root.text_config;
        c.dense_bits = precision.dense_bits;
        c.dense_only = precision.dense_only;
        c.dense_group = precision.dense_group;
        // the side-car replaces the 8-bit copies only: an FP16 load (the reference of the checks) ignores it
        c.dense_sidecar = match std::env::var_os("MLXL3_INKLING_DENSE_SIDECAR") {
            Some(path) if c.dense_bits == 8 => {
                let path = Path::new(&path);
                Some(std::sync::Arc::new(Checkpoint {
                    path: path.to_path_buf(),
                    model_type: "inkling2-dense-sidecar".into(),
                    bits: Some(8.0),
                    size_bytes: std::fs::metadata(path)?.len(),
                    modules: Vec::new(),
                    tensors: checkpoint::read_header(path)
                        .with_context(|| format!("dense side-car {}", path.display()))?,
                }))
            }
            _ => None,
        };
        let lc = &c.loop_config;
        ensure!(
            c.hidden_size > 0
                && c.num_hidden_layers > 0
                && lc.core_start < lc.core_end
                && lc.core_end <= c.num_hidden_layers
                && lc.num_passes >= 1
                && !lc.is_varloop
                && c.n_shared_experts == 0
                && c.use_gate_bias
                && c.sconv_kernel_size > 1
                && c.unpadded_vocab_size > 0
                && c.unpadded_vocab_size <= c.vocab_size
                && c.n_layers_per_attn_res_block > 0
                && c.final_logit_softcapping > 0.0
                && c.rel_extent > 0
                && c.sliding_window_size > 0
                && c.intermediate_size > 0
                && c.intermediate_size % 128 == 0,
            "unsupported Inkling 2 configuration"
        );
        let hidden = c.hidden_size;
        let embeddings =
            Embeddings::load(checkpoint, c.vocab_size, hidden, embed_bits_requested()?)?;
        let embed_norm = f32_weight(checkpoint, "model.llm.embed_norm.weight", &[hidden])?;
        let norm = f32_weight(checkpoint, "model.llm.norm.weight", &[hidden])?;
        let final_query = f32_weight(checkpoint, "model.llm.attn_res.output_query", &[hidden])?
            .reshape(&[1, hidden])?;
        // Checkpoint: [query, mlp] rows and [key, value] rows per loop layer -> [loop, 4 (q, k, v, mlp), D]
        let loop_layers = (lc.core_end - lc.core_start) as i32;
        let qm = f32_weight(
            checkpoint,
            "model.llm.middle_cycle_cross_pass_queries",
            &[2 * loop_layers, hidden],
        )?
        .reshape(&[loop_layers, 2, hidden])?;
        let kv = f32_weight(
            checkpoint,
            "model.llm.middle_cycle_cross_pass_kv_queries",
            &[2 * loop_layers, hidden],
        )?
        .reshape(&[loop_layers, 2, hidden])?;
        let message_queries =
            Array::concatenate(&[&qm.slice(1, 0, 1)?, &kv, &qm.slice(1, 1, 2)?], 1)?;
        message_queries.eval()?;
        // SIZE-2026-11-21-INK2M-SLIM: a checkpoint may store the affine head copy itself (unpadded rows)
        let (head, head_q) = match Linear::stored(
            checkpoint,
            "model.llm.unembed",
            hidden,
            c.unpadded_vocab_size,
        )? {
            Some(stored) => {
                ensure!(
                    stored.bits() == precision.head_bits,
                    "this checkpoint stores a {}-bit output head; MLXL3_INKLING_HEAD_BITS={} is unavailable",
                    stored.bits(),
                    precision.head_bits
                );
                (None, Some(stored))
            }
            None => {
                let head =
                    Projection::load(checkpoint, "model.llm.unembed", hidden, c.vocab_size, false)?;
                let head_q = if precision.head_bits < 16 {
                    Some(Linear::quantized(
                        &head.dense_weight().context("dense Inkling head")?.slice(
                            0,
                            0,
                            c.unpadded_vocab_size,
                        )?,
                        precision.head_bits,
                    )?)
                } else {
                    None
                };
                (Some(head), head_q)
            }
        };
        let layers = (0..c.num_hidden_layers)
            .map(|i| Layer::load(checkpoint, i, &c).with_context(|| format!("Inkling layer {i}")))
            .collect::<Result<Vec<_>>>()?;
        let mut visits: Vec<usize> = (0..lc.core_start).collect();
        for _ in 0..lc.num_passes {
            visits.extend(lc.core_start..lc.core_end);
        }
        visits.extend(lc.core_end..c.num_hidden_layers);
        let states = visits.iter().map(|_| VisitState::default()).collect();
        Ok(Self {
            profile: RunProfile::default(),
            pending: None,
            kernel: c.sconv_kernel_size,
            embeddings,
            embed_norm,
            norm,
            final_query,
            message_queries,
            head,
            head_q,
            precision,
            layers,
            visits,
            states,
            core: (lc.core_start, lc.core_end),
            passes: lc.num_passes,
            vocab: c.vocab_size,
            unpadded_vocab: c.unpadded_vocab_size,
            hidden,
            eps: c.rms_norm_eps,
            softcap: c.final_logit_softcapping,
            window: c.sliding_window_size,
            extent: c.rel_extent,
            context_limit: c.model_max_length,
            offset: 0,
            epoch: next_epoch(),
            // `MLXL3_INKLING_VISION=0` keeps the model text only
            vision: if crate::inkling2_vision::Vision::present(checkpoint)
                && std::env::var("MLXL3_INKLING_VISION").as_deref() != Ok("0")
            {
                VisionSlot::Available(checkpoint.path.clone())
            } else {
                VisionSlot::Absent
            },
            images: Vec::new(),
        })
    }

    /// The checkpoint ships the vision encoder: `prepare_images` accepts pictures.
    pub fn supports_images(&self) -> bool {
        !matches!(self.vision, VisionSlot::Absent)
    }

    /// Encodes `images` and places their features at the image placeholders of `tokens`, the whole
    /// prompt from position 0: the `i`-th run of placeholders must hold exactly the `i`-th image's tokens.
    /// The next forward calls read those rows instead of the placeholder embeddings. An empty list only
    /// clears the previous prompt's features.
    pub fn prepare_images(
        &mut self,
        tokens: &[u32],
        images: &[crate::lfm3::PreparedImage],
    ) -> Result<()> {
        self.images.clear();
        if images.is_empty() {
            return Ok(());
        }
        ensure!(
            self.offset == 0,
            "Inkling images need a prompt prefilled from its start"
        );
        let runs = crate::inkling2_vision::placeholder_runs(tokens);
        ensure!(
            runs.len() == images.len()
                && runs
                    .iter()
                    .zip(images)
                    .all(|(&(_, length), image)| length == image.token_count()),
            "image placeholders of the prompt do not match its {} images",
            images.len()
        );
        if let VisionSlot::Available(path) = &self.vision {
            let path = path.clone();
            let checkpoint = checkpoint::inspect(&path)?;
            let root: serde_json::Value =
                serde_json::from_reader(File::open(path.join("config.json"))?)?;
            let vision = crate::inkling2_vision::Vision::load(&checkpoint, &root)
                .context("loading the Inkling vision encoder")?;
            ensure!(
                vision.text_hidden() == self.hidden,
                "Inkling vision output does not match the language model"
            );
            self.vision = VisionSlot::Loaded(Box::new(vision));
        }
        let VisionSlot::Loaded(vision) = &self.vision else {
            anyhow::bail!("this Inkling checkpoint has no vision encoder");
        };
        let mut placed = Vec::with_capacity(images.len());
        for (&(first, length), image) in runs.iter().zip(images) {
            let features = vision.forward(image)?.astype(Dtype::Float32)?;
            features.eval()?;
            ensure!(
                features.shape() == [i32::try_from(length)?, self.hidden],
                "unexpected Inkling image feature shape"
            );
            placed.push((i32::try_from(first)?, features));
        }
        self.images = placed;
        Ok(())
    }

    /// `hidden` `[T, D]` of positions `[start, start + T)` with the rows of prepared images in place.
    fn place_images(&self, mut hidden: Array, start: i32) -> Result<Array> {
        let end = start + hidden.shape()[0];
        for (first, rows) in &self.images {
            let (from, to) = ((*first).max(start), (first + rows.shape()[0]).min(end));
            if from < to {
                hidden = hidden.slice_update(
                    &rows.slice(0, from - first, to - first)?,
                    &[from - start, 0],
                )?;
            }
        }
        Ok(hidden)
    }

    pub fn context_limit(&self) -> i32 {
        self.context_limit
    }

    pub fn take_profile(&mut self) -> RunProfile {
        std::mem::take(&mut self.profile)
    }

    pub fn reset(&mut self) {
        self.offset = 0;
        self.pending = None;
        self.epoch = next_epoch();
        self.images.clear();
        for state in &mut self.states {
            *state = VisitState::default();
        }
    }

    /// SPEED-2026-11-23-INK2M-MAYA PC: the committed state now (no uncommitted verify call), to resume a
    /// later prompt that starts with the same `offset()` tokens.
    pub fn prefix_snapshot(&self) -> Result<InklingPrefix> {
        ensure!(
            self.pending.is_none() && self.states.iter().all(|s| s.pending_kv.is_none()),
            "Inkling prefix taken during an uncommitted call"
        );
        let mut visits = Vec::with_capacity(self.states.len());
        for (visit, state) in self.states.iter().enumerate() {
            let full = self.layers[self.visits[visit]].attention.window.is_none();
            let kv = match &state.kv_buffer {
                Some(buffer) if full && buffer.base == 1 && buffer.zero == 0 => {
                    PrefixKv::Rows(buffer.len)
                }
                buffer => PrefixKv::Saved {
                    keys: state.keys.as_ref().map(Array::try_clone).transpose()?,
                    values: state.values.as_ref().map(Array::try_clone).transpose()?,
                    buffer: buffer
                        .as_ref()
                        .map(|b| -> Result<_> {
                            Ok((
                                b.keys.try_clone()?,
                                b.values.try_clone()?,
                                b.base,
                                b.len,
                                b.zero,
                            ))
                        })
                        .transpose()?,
                },
            };
            // the windows are views of the last chunk's input: keep only their rows (same values)
            let mut conv: [Option<Array>; 4] = Default::default();
            for (saved, live) in conv.iter_mut().zip(&state.conv) {
                *saved = live.as_ref().map(Array::compact_for_cache).transpose()?;
            }
            visits.push(PrefixVisit { kv, conv });
        }
        Ok(InklingPrefix {
            epoch: self.epoch,
            offset: self.offset,
            visits,
        })
    }

    /// SPEED-2026-11-23-INK2M-MAYA PC: back to `prefix`, taken earlier from this same state lineage (no
    /// `reset` since). Returns `false`, leaving the state untouched, when it cannot resume (another
    /// lineage, fewer rows than the prefix, a layout change); the caller then prefills from scratch.
    pub fn resume_prefix(&mut self, prefix: InklingPrefix) -> Result<bool> {
        if prefix.epoch != self.epoch
            || prefix.offset > self.offset
            || prefix.visits.len() != self.states.len()
        {
            return Ok(false);
        }
        for (state, saved) in self.states.iter().zip(&prefix.visits) {
            if let PrefixKv::Rows(len) = saved.kv {
                let Some(buffer) = &state.kv_buffer else {
                    return Ok(false);
                };
                if buffer.base != 1 || buffer.zero != 0 || buffer.len < len || len < 1 {
                    return Ok(false);
                }
            }
        }
        self.pending = None;
        for (state, saved) in self.states.iter_mut().zip(prefix.visits) {
            state.pending_kv = None;
            state.pending_conv = Default::default();
            match saved.kv {
                PrefixKv::Rows(len) => {
                    let buffer = state.kv_buffer.as_mut().context("checked buffer")?;
                    buffer.len = len;
                    state.keys = Some(buffer.keys.slice(2, 1, 1 + len)?);
                    state.values = Some(buffer.values.slice(2, 1, 1 + len)?);
                }
                PrefixKv::Saved {
                    keys,
                    values,
                    buffer,
                } => {
                    // mutation 2: only the row count of the live buffer goes back (wrong once the window moved)
                    #[cfg(test)]
                    if PREFIX_MUTATION.with(std::cell::Cell::get) == 2
                        && let (Some(live), Some((_, _, _, len, _))) =
                            (state.kv_buffer.as_mut(), buffer.as_ref())
                    {
                        live.len = *len;
                        state.keys = Some(live.keys.slice(2, live.base, live.base + len)?);
                        state.values = Some(live.values.slice(2, live.base, live.base + len)?);
                        state.conv = saved.conv;
                        continue;
                    }
                    state.keys = keys;
                    state.values = values;
                    state.kv_buffer = buffer.map(|(keys, values, base, len, zero)| KvBuffer {
                        keys,
                        values,
                        base,
                        len,
                        zero,
                    });
                }
            }
            #[cfg(test)]
            if PREFIX_MUTATION.with(std::cell::Cell::get) == 1 {
                continue;
            }
            state.conv = saved.conv;
        }
        self.offset = prefix.offset;
        Ok(true)
    }

    pub fn forward(&mut self, token: u32) -> Result<Array> {
        self.forward_tokens(&[token])
    }

    /// Logits `[1, unpadded_vocab]` FP32 of the last token.
    pub fn forward_tokens(&mut self, tokens: &[u32]) -> Result<Array> {
        match self.run(Ids::Host(tokens), false, None, true, false) {
            Ok((logits, _)) => Ok(logits),
            Err(error) => {
                self.reset();
                Err(error.context("Inkling forward failed; its cache was reset"))
            }
        }
    }

    /// Logits `[T, unpadded_vocab]` FP32 of every token (parity checks).
    pub fn forward_all(&mut self, tokens: &[u32]) -> Result<Array> {
        match self.run(Ids::Host(tokens), true, None, true, false) {
            Ok((logits, _)) => Ok(logits),
            Err(error) => {
                self.reset();
                Err(error.context("Inkling forward failed; its cache was reset"))
            }
        }
    }

    /// Normalized token embeddings `[n, D]` FP32 (`embed_norm(embed(ids))`).
    pub fn embed_tokens(&self, ids: &Array) -> Result<Array> {
        self.embeddings
            .rows(ids)?
            .astype(Dtype::Float32)?
            .rms_norm(&self.embed_norm, self.eps)
    }

    /// Unsoftcapped vocabulary scores `[n, unpadded_vocab]` of FP16 hidden rows.
    pub fn head_scores(&self, hidden: &Array) -> Result<Array> {
        let hidden = hidden.astype(Dtype::Float16)?;
        if ablated("head") {
            let rows = hidden.shape()[0];
            return hidden
                .slice(1, 0, 1)?
                .broadcast_to(&[rows, self.unpadded_vocab]);
        }
        match (&self.head_q, &self.head) {
            (Some(head), _) => head.forward(&hidden),
            (None, Some(head)) => head.forward(&hidden)?.slice(1, 0, self.unpadded_vocab),
            (None, None) => anyhow::bail!("Inkling model without an output head"),
        }
    }

    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// `[vocab, D]` FP16 output head (padded rows included); `None` for a checkpoint that stores only the
    /// affine copy.
    pub fn head_weight(&self) -> Option<&Array> {
        self.head.as_ref().and_then(Projection::dense_weight)
    }

    /// Padded vocabulary rows of the embedding table (and of the dense head when there is one).
    pub fn padded_vocab(&self) -> i32 {
        self.vocab
    }

    pub fn logit_softcap(&self) -> f32 {
        self.softcap
    }

    pub fn unpadded_vocab(&self) -> i32 {
        self.unpadded_vocab
    }

    pub fn hidden_size(&self) -> i32 {
        self.hidden
    }

    /// Number of decoder-layer visits per token (the draft feature indices refer to these).
    pub fn visit_count(&self) -> usize {
        self.visits.len()
    }

    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// Speculative verification: logits `[1, T, vocab]` of every row and the draft features `[T, taps * D]`
    /// (FP32 residual stream before each tapped visit, then the final normalized hidden for the index equal to
    /// `visit_count`). Nothing is committed until `commit`.
    pub fn verify(&mut self, tokens: &[u32], taps: &[usize]) -> Result<(Array, Array)> {
        ensure!(
            !taps.is_empty()
                && taps.windows(2).all(|w| w[0] < w[1])
                && *taps.last().unwrap_or(&0) == self.visits.len(),
            "Inkling draft taps must be increasing and end at the visit count"
        );
        match self.run(Ids::Host(tokens), true, Some(taps), false, false) {
            Ok((logits, features)) => Ok((logits, features.context("features")?)),
            Err(error) => {
                self.reset();
                Err(error.context("Inkling verify failed; its cache was reset"))
            }
        }
    }

    /// DSPARK-2026-11-23-INK2M-MTPLX-A: `verify` of ids still on the GPU (`[n]` integers below the unpadded
    /// vocabulary, e.g. the anchor followed by the drafter's proposals), submitted without waiting: the same
    /// graph as `verify` of the same ids, so the same logits and features.
    pub fn verify_device(&mut self, ids: &Array, taps: &[usize]) -> Result<(Array, Array)> {
        ensure!(
            !taps.is_empty()
                && taps.windows(2).all(|w| w[0] < w[1])
                && *taps.last().unwrap_or(&0) == self.visits.len(),
            "Inkling draft taps must be increasing and end at the visit count"
        );
        match self.run(Ids::Device(ids), true, Some(taps), false, true) {
            Ok((logits, features)) => Ok((logits, features.context("features")?)),
            Err(error) => {
                self.reset();
                Err(error.context("Inkling verify failed; its cache was reset"))
            }
        }
    }

    /// Committed forward of `tokens` that also captures the drafter features of every row: the logits of
    /// the last row only (`[1, 1, vocab]`) and the features `[T, taps * D]`. Prefill of a DSpark session.
    pub fn forward_with_features(
        &mut self,
        tokens: &[u32],
        taps: &[usize],
    ) -> Result<(Array, Array)> {
        ensure!(
            !taps.is_empty()
                && taps.windows(2).all(|w| w[0] < w[1])
                && *taps.last().unwrap_or(&0) == self.visits.len(),
            "Inkling draft taps must be increasing and end at the visit count"
        );
        match self.run(Ids::Host(tokens), false, Some(taps), true, false) {
            Ok((logits, features)) => Ok((logits, features.context("features")?)),
            Err(error) => {
                self.reset();
                Err(error.context("Inkling forward failed; its cache was reset"))
            }
        }
    }

    /// SPEED-INK2M-P14: committed decode step of one token still on the GPU (`[1]` integer id, e.g. a
    /// `draw_token` result, below the unpadded vocabulary). Logits `[1, 1, vocab]` FP32 and, with `taps`,
    /// the drafter features `[1, taps * D]`. With `asynchronous`, the step is submitted without waiting.
    pub fn forward_device(
        &mut self,
        id: &Array,
        taps: Option<&[usize]>,
        asynchronous: bool,
    ) -> Result<(Array, Option<Array>)> {
        if let Some(taps) = taps {
            ensure!(
                !taps.is_empty()
                    && taps.windows(2).all(|w| w[0] < w[1])
                    && *taps.last().unwrap_or(&0) == self.visits.len(),
                "Inkling draft taps must be increasing and end at the visit count"
            );
        }
        match self.run(Ids::Device(id), false, taps, true, asynchronous) {
            Ok(outputs) => Ok(outputs),
            Err(error) => {
                self.reset();
                Err(error.context("Inkling forward failed; its cache was reset"))
            }
        }
    }

    /// Keep the first `rows` rows of the last `verify` call (1 <= rows <= its length).
    pub fn commit(&mut self, rows: usize) -> Result<()> {
        let calls = self.pending.context("no pending Inkling verify")?;
        let rows = i32::try_from(rows).context("commit rows")?;
        ensure!(
            rows >= 1 && rows <= calls,
            "commit rows outside the verified call"
        );
        let result = self.commit_rows(rows, calls).and_then(|()| {
            let mut pending = Vec::new();
            for state in &self.states {
                pending.extend([&state.keys, &state.values].into_iter().flatten());
                pending.extend(state.conv.iter().flatten());
            }
            Array::eval_many(&pending)
        });
        if let Err(error) = result {
            self.reset();
            return Err(error.context("Inkling commit failed; its cache was reset"));
        }
        Ok(())
    }

    fn commit_rows(&mut self, rows: i32, calls: i32) -> Result<()> {
        for (visit, state) in self.states.iter_mut().enumerate() {
            if ablated("loop") && state.pending_kv.is_none() {
                continue;
            }
            let window = self.layers[self.visits[visit]].attention.window;
            state.commit(rows, calls, window, self.kernel)?;
        }
        self.offset += rows;
        self.pending = None;
        Ok(())
    }

    pub fn eval_state(&self) -> Result<()> {
        for state in &self.states {
            for a in [&state.keys, &state.values].into_iter().flatten() {
                a.eval()?;
            }
            for a in state.conv.iter().flatten() {
                a.eval()?;
            }
        }
        Ok(())
    }

    fn run(
        &mut self,
        ids: Ids<'_>,
        all: bool,
        taps: Option<&[usize]>,
        commit: bool,
        asynchronous: bool,
    ) -> Result<(Array, Option<Array>)> {
        // INVARIANT-2026-11-24-INK2M-L1L4: the invariant rows apply to verifications (no commit) only
        let _verify_scope = VerifyScope::enter(!commit);
        let started = Instant::now();
        #[cfg(test)]
        let cpu_started = thread_cpu();
        ensure!(
            self.pending.is_none(),
            "Inkling has an uncommitted verify call"
        );
        let time = match ids {
            Ids::Host(tokens) => {
                ensure!(!tokens.is_empty(), "Inkling needs at least one token");
                ensure!(
                    tokens.iter().all(|&t| t < self.vocab as u32),
                    "Inkling token is outside vocabulary"
                );
                i32::try_from(tokens.len()).context("Inkling batch is too large")?
            }
            // DSPARK-2026-11-23-INK2M-MTPLX-A: a verify block may also be a device array of ids
            Ids::Device(id) => {
                ensure!(
                    id.shape().len() == 1 && id.shape()[0] >= 1,
                    "Inkling device tokens must have shape [n], n >= 1"
                );
                id.shape()[0]
            }
        };
        let start = self.offset;
        let end = start
            .checked_add(time)
            .context("Inkling position overflow")?;
        ensure!(end <= self.context_limit, "Inkling context is full");
        // a device token comes from `draw_token` or the drafter's argmax (below the unpadded vocabulary by
        // construction)
        let ids = match ids {
            Ids::Host(tokens) => Array::from_i32(
                &tokens.iter().map(|&t| t as i32).collect::<Vec<_>>(),
                &[time],
            )?,
            Ids::Device(id) => id.astype(Dtype::Int32)?,
        };
        let hidden = self.embed_tokens(&ids)?;
        // image placeholders read their encoder rows (text-only prompts keep the embeddings as they are)
        let hidden = if self.images.is_empty() {
            hidden
        } else {
            self.place_images(hidden, start)?
        };
        // Sliding visits keep at most window - 1 past keys; full visits keep everything
        let sliding_first = (start - (self.window - 1)).max(0);
        let sliding = MaskPlan::new(
            start,
            time,
            sliding_first,
            end - sliding_first,
            self.window,
            Some(self.window),
        )?;
        let full = MaskPlan::new(start, time, 0, end, self.extent, None)?;
        let (core_start, core_end) = self.core;
        let mut blocks: Vec<Array> = Vec::new();
        let mut hidden = hidden;
        let mut visit = 0usize;
        let mut features: Vec<Array> = Vec::new();
        let tapped = |visit: usize| taps.is_some_and(|t| t.contains(&visit));
        let layers = &self.layers;
        let states = &mut self.states;
        let mut run_layer = |layer: usize,
                             hidden: &Array,
                             blocks: &mut Vec<Array>,
                             messages: Option<&Array>,
                             visit: usize|
         -> Result<Array> {
            let l = &layers[layer];
            let plan = if l.sliding { &sliding } else { &full };
            l.forward(hidden, blocks, messages, start, plan, &mut states[visit])
        };
        for layer in 0..core_start {
            if tapped(visit) {
                features.push(hidden.try_clone()?);
            }
            hidden = run_layer(layer, &hidden, &mut blocks, None, visit)?;
            visit += 1;
        }
        let pre_hidden = hidden.try_clone()?;
        let pre_blocks = blocks
            .iter()
            .map(Array::try_clone)
            .collect::<Result<Vec<_>>>()?;
        // timing-only ablation: one pass of the loop (meaningless outputs)
        let passes = if ablated("loop") { 1 } else { self.passes };
        for pass in 0..passes {
            let bank = if pass == 0 {
                None
            } else {
                let previous: Vec<Array> = blocks[pre_blocks.len() + 1..]
                    .iter()
                    .map(Array::try_clone)
                    .collect::<Result<_>>()?;
                let partial = hidden.rms_norm_without_weight(self.eps)?;
                let stacked = if lean() {
                    let mut sources: Vec<&Array> = previous.iter().collect();
                    sources.push(&partial);
                    Some(depth_stack(&sources)?)
                } else {
                    None
                };
                hidden = pre_hidden.try_clone()?;
                blocks = pre_blocks
                    .iter()
                    .map(Array::try_clone)
                    .collect::<Result<_>>()?;
                Some((previous, partial, stacked))
            };
            for (i, layer) in (core_start..core_end).enumerate() {
                let messages = match &bank {
                    Some((previous, partial, stacked)) => {
                        let queries = self
                            .message_queries
                            .slice(0, i as i32, i as i32 + 1)?
                            .reshape(&[4, self.hidden])?;
                        Some(match stacked {
                            Some(stacked) => depth_attend(stacked, &queries)?,
                            None => {
                                let mut sources: Vec<&Array> = previous.iter().collect();
                                sources.push(partial);
                                depth_attention(&sources, &queries)?
                            }
                        })
                    }
                    None => None,
                };
                if tapped(visit) {
                    features.push(hidden.try_clone()?);
                }
                hidden = run_layer(layer, &hidden, &mut blocks, messages.as_ref(), visit)?;
                visit += 1;
            }
        }
        for layer in core_end..layers.len() {
            if tapped(visit) {
                features.push(hidden.try_clone()?);
            }
            hidden = run_layer(layer, &hidden, &mut blocks, None, visit)?;
            visit += 1;
        }
        ensure!(
            visit == self.visits.len() || ablated("loop"),
            "Inkling visit count mismatch"
        );
        let partial = hidden.rms_norm_without_weight(self.eps)?;
        let mut sources: Vec<&Array> = blocks.iter().collect();
        sources.push(&partial);
        let hidden = depth_attention(&sources, &self.final_query)?
            .reshape(&[time, self.hidden])?
            .rms_norm(&self.norm, self.eps)?;
        let features = match taps {
            Some(taps) => {
                features.push(hidden.try_clone()?);
                ensure!(
                    features.len() == taps.len(),
                    "Inkling draft taps not all reached"
                );
                Some(Array::concatenate(&features.iter().collect::<Vec<_>>(), 1)?)
            }
            None => None,
        };
        let hidden = if all {
            hidden
        } else {
            hidden.slice(0, time - 1, time)?
        };
        let rows = hidden.shape()[0];
        let logits = self
            .head_scores(&hidden)?
            .astype(Dtype::Float32)?
            .softcap(self.softcap)?;
        // Same layout as the other chat models: [1, rows, vocab]
        let logits = logits.reshape(&[1, rows, self.unpadded_vocab])?;
        if commit {
            self.pending = Some(time);
            self.commit_rows(time, time)?;
        } else {
            self.pending = Some(time);
        }
        let mut pending = vec![&logits];
        pending.extend(features.iter());
        for state in &self.states {
            if commit {
                pending.extend([&state.keys, &state.values].into_iter().flatten());
                pending.extend(state.conv.iter().flatten());
            } else {
                if let Some((k, v, _)) = &state.pending_kv {
                    pending.extend([k, v]);
                }
                pending.extend(state.pending_conv.iter().flatten());
            }
        }
        static GRAPH_STATS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if *GRAPH_STATS.get_or_init(|| std::env::var_os("MLXL3_INKLING_GRAPH_STATS").is_some()) {
            eprintln!(
                "inkling graph rows={time}\n{}",
                Array::graph_stats(&pending)?
            );
        }
        let built = Instant::now();
        #[cfg(test)]
        let cpu_built = thread_cpu();
        if asynchronous {
            Array::async_eval_all(&pending)?;
        } else {
            submit(&pending)?;
        }
        let done = Instant::now();
        #[cfg(test)]
        {
            let cpu_done = thread_cpu();
            RUN_CPU.with(|cell| {
                let (build, eval) = cell.get();
                cell.set((
                    build + cpu_built.saturating_sub(cpu_started),
                    eval + cpu_done.saturating_sub(cpu_built),
                ));
            });
        }
        self.profile.calls += 1;
        self.profile.build += built - started;
        self.profile.eval += done - built;
        debug_assert!(commit || self.offset == start);
        let _ = end;
        Ok((logits, features))
    }
}

/// Test and export helper: writes `tensors` (U32, F16 or F32, row-major) as one safetensors file.
#[cfg(test)]
pub(crate) fn write_safetensors(path: &Path, tensors: &[(String, &Array)]) -> Result<()> {
    use std::io::Write;
    let mut header = serde_json::Map::new();
    let mut payloads = Vec::with_capacity(tensors.len());
    let mut offset = 0u64;
    for (name, array) in tensors {
        let dtype = match array.dtype() {
            Dtype::UInt32 => "U32",
            Dtype::Float16 => "F16",
            Dtype::Float32 => "F32",
            other => anyhow::bail!("{name}: cannot write dtype {other:?}"),
        };
        let bytes = array.to_bytes()?;
        let end = offset + bytes.len() as u64;
        let previous = header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": array.shape(), "data_offsets": [offset, end]}),
        );
        ensure!(previous.is_none(), "duplicate tensor {name}");
        offset = end;
        payloads.push(bytes);
    }
    let mut json = serde_json::to_vec(&serde_json::Value::Object(header))?;
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut file = std::io::BufWriter::new(File::create(path)?);
    file.write_all(&(json.len() as u64).to_le_bytes())?;
    file.write_all(&json)?;
    for bytes in payloads {
        file.write_all(&bytes)?;
    }
    file.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stored_checkpoint(dir: &Path, tensors: &[(String, &Array)]) -> Result<Checkpoint> {
        let path = dir.join("model.safetensors");
        write_safetensors(&path, tensors)?;
        Ok(Checkpoint {
            path: dir.to_path_buf(),
            model_type: "inkling2_loop".into(),
            bits: None,
            size_bytes: std::fs::metadata(&path)?.len(),
            modules: Vec::new(),
            tensors: checkpoint::read_header(&path)?,
        })
    }

    /// SIZE-2026-11-21-INK2M-SLIM: an affine copy written to a checkpoint and read back by `Linear::stored`
    /// is the copy `Linear::quantized` made (same bytes, same matmul bits); every malformed variant and a
    /// width other than the stored one are refused; a checkpoint without scales stays dense.
    #[test]
    fn stored_affine_weights_round_trip_and_refusals() -> Result<()> {
        let (output, input) = (24, 192);
        let weight = Array::from_f32(
            &pseudo_random((output * input) as usize, 7, 1.0),
            &[output, input],
        )?
        .astype(Dtype::Float16)?;
        let x = Array::from_f32(&pseudo_random(3 * input as usize, 11, 1.0), &[3, input])?;
        let dir = tempfile::tempdir()?;
        for bits in [4, 8] {
            let Linear::Affine {
                w, scales, biases, ..
            } = Linear::quantized(&weight, bits)?
            else {
                anyhow::bail!("quantized returned a dense weight");
            };
            let named = |p: &str| -> Vec<(String, &Array)> {
                vec![
                    (format!("{p}.weight"), &w),
                    (format!("{p}.scales"), &scales),
                    (format!("{p}.biases"), &biases),
                ]
            };
            let ckpt = stored_checkpoint(dir.path(), &named("p"))?;
            let stored = Linear::stored(&ckpt, "p", input, output)?.context("stored copy")?;
            assert_eq!(stored.bits(), bits);
            let Linear::Affine {
                w: sw,
                scales: ss,
                biases: sb,
                ..
            } = &stored
            else {
                anyhow::bail!("stored copy is dense");
            };
            assert_eq!(sw.to_bytes()?, w.to_bytes()?);
            assert_eq!(ss.to_bytes()?, scales.to_bytes()?);
            assert_eq!(sb.to_bytes()?, biases.to_bytes()?);
            let fresh = Linear::quantized(&weight, bits)?
                .forward(&x)?
                .to_f16_bits()?;
            assert_eq!(stored.forward(&x)?.to_f16_bits()?, fresh);
            // the requested width must be the stored one
            assert!(Linear::load(&ckpt, "p", input, output, bits).is_ok());
            for other in [4, 8, 16].into_iter().filter(|&b| b != bits) {
                let error = Linear::load(&ckpt, "p", input, output, other)
                    .err()
                    .context("other width accepted")?;
                assert!(error.to_string().contains("stored as"), "{error}");
            }
            // wrong logical dimensions
            assert!(Linear::stored(&ckpt, "p", input, output + 1).is_err());
            assert!(Linear::stored(&ckpt, "p", input * 2, output).is_err());
            assert!(Linear::stored(&ckpt, "p", 100, output).is_err());
            // F32 scales, missing biases
            let scales32 = scales.astype(Dtype::Float32)?;
            let bad = stored_checkpoint(
                dir.path(),
                &[
                    ("p.weight".into(), &w),
                    ("p.scales".into(), &scales32),
                    ("p.biases".into(), &biases),
                ],
            )?;
            assert!(Linear::stored(&bad, "p", input, output).is_err());
            let missing = stored_checkpoint(
                dir.path(),
                &[("p.weight".into(), &w), ("p.scales".into(), &scales)],
            )?;
            assert!(Linear::stored(&missing, "p", input, output).is_err());
        }
        // no scales: dense path, nothing stored
        let dense = stored_checkpoint(dir.path(), &[("p.weight".into(), &weight)])?;
        assert!(Linear::stored(&dense, "p", input, output)?.is_none());
        Ok(())
    }

    /// SIZE-2026-11-22-INK2M-EMBED8 exporter (not a check): writes `embed.safetensors` into the new directory
    /// `MLXL3_INKLING_SLIM_OUT` with the 8-bit affine copy of the FP16 embedding table of `MLXL3_INKLING_MODEL`,
    /// computed as `Embeddings::load` does for the measured option (`half_weight`, then `quantize_affine`).
    #[test]
    #[ignore = "writes files; requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_SLIM_OUT"]
    fn export_slim_embeddings() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let out = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_SLIM_OUT")?);
        std::fs::create_dir(&out).context("MLXL3_INKLING_SLIM_OUT must not exist yet")?;
        let checkpoint = checkpoint::inspect(&dir)?;
        let root: RootConfig = serde_json::from_reader(File::open(dir.join("config.json"))?)?;
        let (vocab, hidden) = (root.text_config.vocab_size, root.text_config.hidden_size);
        ensure!(
            Linear::stored(&checkpoint, "model.llm.embed", hidden, vocab)?.is_none(),
            "the source already stores an affine embedding table"
        );
        let table = half_weight(
            &checkpoint,
            "model.llm.embed.weight",
            Some(&[vocab, hidden]),
        )?;
        let (w, scales, biases) = table.quantize_affine(AFFINE_GROUP, 8)?;
        Array::eval_many(&[&w, &scales, &biases])?;
        write_safetensors(
            &out.join("embed.safetensors"),
            &[
                ("model.llm.embed.weight".into(), &w),
                ("model.llm.embed.scales".into(), &scales),
                ("model.llm.embed.biases".into(), &biases),
            ],
        )?;
        let report = serde_json::json!({"trial": "SIZE-2026-11-22-INK2M-EMBED8", "source": dir,
            "prefix": "model.llm.embed", "bits": 8, "group": AFFINE_GROUP, "rows": vocab, "hidden": hidden});
        std::fs::write(
            out.join("export.json"),
            serde_json::to_string_pretty(&report)?,
        )?;
        println!("{report}");
        Ok(())
    }

    /// SIZE-2026-11-22-INK2M-EMBED8: a stored 8-bit table gives, row for row, the bits of the dense table rounded
    /// at load (the measured option); another requested width is refused; a dense table is unchanged by default.
    #[test]
    fn stored_embeddings_equal_the_rounded_table() -> Result<()> {
        let (vocab, hidden) = (40, 128);
        let table = Array::from_f32(
            &pseudo_random((vocab * hidden) as usize, 9, 1.0),
            &[vocab, hidden],
        )?
        .astype(Dtype::Float16)?;
        let (w, scales, biases) = table.quantize_affine(AFFINE_GROUP, 8)?;
        let dir = tempfile::tempdir()?;
        let stored = stored_checkpoint(
            dir.path(),
            &[
                ("model.llm.embed.weight".into(), &w),
                ("model.llm.embed.scales".into(), &scales),
                ("model.llm.embed.biases".into(), &biases),
            ],
        )?;
        let dense_dir = tempfile::tempdir()?;
        let dense = stored_checkpoint(
            dense_dir.path(),
            &[("model.llm.embed.weight".into(), &table)],
        )?;
        let ids = Array::from_i32(&[0, 39, 7, 7, 21], &[5])?;
        let from_stored = Embeddings::load(&stored, vocab, hidden, None)?.rows(&ids)?;
        let explicit = Embeddings::load(&stored, vocab, hidden, Some(8))?.rows(&ids)?;
        let rounded = Embeddings::load(&dense, vocab, hidden, Some(8))?.rows(&ids)?;
        assert_eq!(from_stored.dtype(), Dtype::Float16);
        assert_eq!(from_stored.shape(), [5, hidden]);
        assert_eq!(from_stored.to_f16_bits()?, rounded.to_f16_bits()?);
        assert_eq!(explicit.to_f16_bits()?, rounded.to_f16_bits()?);
        for other in [4, 16] {
            let error = Embeddings::load(&stored, vocab, hidden, Some(other))
                .err()
                .context("other width accepted")?;
            assert!(
                error.to_string().contains("stores 8-bit embeddings"),
                "{error}"
            );
        }
        // dense table: unchanged without a request or with 16, rounded only on request
        let plain = table.take(&ids, 0)?.to_f16_bits()?;
        assert_eq!(
            Embeddings::load(&dense, vocab, hidden, None)?
                .rows(&ids)?
                .to_f16_bits()?,
            plain
        );
        assert_eq!(
            Embeddings::load(&dense, vocab, hidden, Some(16))?
                .rows(&ids)?
                .to_f16_bits()?,
            plain
        );
        assert_ne!(rounded.to_f16_bits()?, plain);
        // a stored copy of the wrong vocabulary size is refused
        assert!(Embeddings::load(&stored, vocab + 1, hidden, None).is_err());
        Ok(())
    }

    /// SIZE-2026-11-21-INK2M-SLIM part B: the dequantized table equals the stored copy decoded row by row
    /// (`take` then `dequantize`), and its error stays within half a quantization step.
    #[test]
    fn dequantized_embedding_rows_match_row_lookup() -> Result<()> {
        let (rows, dim) = (40, 128);
        let values = pseudo_random((rows * dim) as usize, 5, 1.0);
        let table = Array::from_f32(&values, &[rows, dim])?.astype(Dtype::Float16)?;
        let (w, scales, biases) = table.quantize_affine(AFFINE_GROUP, 8)?;
        let full = Array::dequantize_affine(&w, &scales, &biases, AFFINE_GROUP, 8)?;
        assert_eq!(full.dtype(), Dtype::Float16);
        assert_eq!(full.shape(), [rows, dim]);
        let ids = Array::from_i32(&[3, 0, 39, 3, 17], &[5])?;
        let lookup = Array::dequantize_affine(
            &w.take(&ids, 0)?,
            &scales.take(&ids, 0)?,
            &biases.take(&ids, 0)?,
            AFFINE_GROUP,
            8,
        )?;
        assert_eq!(full.take(&ids, 0)?.to_f16_bits()?, lookup.to_f16_bits()?);
        let restored = full.to_f32()?;
        let half = table.to_f32()?;
        let steps = scales.to_f32()?;
        for (i, (a, b)) in restored.iter().zip(&half).enumerate() {
            let step = steps[i / AFFINE_GROUP as usize];
            // half a step, plus the FP16 rounding of the scale, the bias and the result
            let bound = 0.75 * step.abs() + 4e-3 * b.abs().max(1.0);
            assert!(a.is_finite() && (a - b).abs() <= bound, "{i}: {a} vs {b}");
        }
        Ok(())
    }

    #[test]
    fn precision_defaults_and_projection_sets() -> Result<()> {
        // generation default: 8-bit group-64 projections, 8-bit head; numerical reference: everything exact
        assert_eq!(
            (
                Precision::DEFAULT.dense_bits,
                Precision::DEFAULT.head_bits,
                Precision::DEFAULT.dense_group
            ),
            (8, 8, 64)
        );
        assert_eq!(Precision::FULL.dense_group, AFFINE_GROUP);
        for bad in [0, 16, 48, 256] {
            let precision = Precision {
                dense_group: bad,
                ..Precision::DEFAULT
            };
            assert!(precision.validate().is_err(), "dense group {bad} accepted");
        }
        assert_eq!(
            (Precision::FULL.dense_bits, Precision::FULL.head_bits),
            (16, 16)
        );
        assert_eq!(Precision::DEFAULT.dense_only, DenseSet::ALL);
        Precision::DEFAULT.validate()?;
        for bad in [0, 2, 3, 5, 12, 32] {
            let precision = Precision {
                head_bits: bad,
                ..Precision::FULL
            };
            assert!(precision.validate().is_err(), "head bits {bad} accepted");
            let precision = Precision {
                dense_bits: bad,
                ..Precision::FULL
            };
            assert!(precision.validate().is_err(), "dense bits {bad} accepted");
        }
        let set = DenseSet::parse("q+o,gate")?;
        for name in ["q", "o", "gate"] {
            assert!(set.contains(name));
        }
        for name in ["k", "v", "r", "sink", "head", ""] {
            assert!(!set.contains(name));
        }
        assert_eq!(DenseSet::parse("q,k,v,r,o,gate,sink")?, DenseSet::ALL);
        assert!(DenseSet::parse("").is_err());
        assert!(DenseSet::parse("q,unknown").is_err());
        Ok(())
    }

    /// Deterministic values in roughly [-scale, scale] with a wide spread of magnitudes.
    fn pseudo_random(count: usize, seed: u64, scale: f32) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..count)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
                let magnitude = 10f32.powi(((state >> 8) % 5) as i32 - 3);
                unit * magnitude * scale
            })
            .collect()
    }

    /// The fused short-convolution kernel against the op sequence it replaces: same bits for the output and
    /// for the window (next state), for 1 to 300 rows, FP16 and FP32 inputs, with and without a state.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn short_conv_kernel_matches_ops() -> Result<()> {
        let kernel = 4;
        let mut checked = 0usize;
        for channels in [256, 1024, 2304] {
            let weight = Array::from_f32(
                &pseudo_random((channels * kernel) as usize, 7 + channels as u64, 0.7),
                &[channels, kernel, 1],
            )?;
            let conv = ShortConv {
                weight,
                scale: scalar(1.0 / ((kernel + 1) as f32).sqrt())?,
                channels,
                kernel,
            };
            for time in [1, 2, 3, 7, 26, 128, 300] {
                for dtype in [Dtype::Float16, Dtype::Float32] {
                    for with_state in [false, true] {
                        let seed = (channels * 1000 + time * 10 + i32::from(with_state)) as u64;
                        let x = Array::from_f32(
                            &pseudo_random((time * channels) as usize, seed, 50.0),
                            &[time, channels],
                        )?
                        .astype(dtype)?;
                        let state = Array::from_f32(
                            &pseudo_random(((kernel - 1) * channels) as usize, seed + 1, 3.0),
                            &[kernel - 1, channels],
                        )?;
                        let state = with_state.then_some(&state);
                        let (out, window) = conv.forward_kernel(&x, state)?;
                        let (expected, expected_window) = conv.forward_ops(&x, state)?;
                        ensure!(
                            out.shape() == [time, channels]
                                && window.shape() == [time + kernel - 1, channels]
                                && out.dtype() == Dtype::Float32
                                && window.dtype() == Dtype::Float32,
                            "kernel output layout"
                        );
                        let bits = |a: &Array| -> Result<Vec<u32>> {
                            Ok(a.to_f32()?.iter().map(|x| x.to_bits()).collect())
                        };
                        let (out, expected) = (bits(&out)?, bits(&expected)?);
                        ensure!(
                            out.len() == (time * channels) as usize && out == expected,
                            "output bits differ: channels {channels} rows {time} {dtype:?} state {with_state}"
                        );
                        ensure!(
                            bits(&window)? == bits(&expected_window)?,
                            "window bits differ: channels {channels} rows {time} {dtype:?} state {with_state}"
                        );
                        ensure!(
                            out.iter().any(|&b| b != 0)
                                && out.iter().all(|&b| f32::from_bits(b).is_finite()),
                            "degenerate comparison"
                        );
                        checked += 1;
                    }
                }
            }
        }
        ensure!(checked == 84, "expected 84 cases, ran {checked}");
        Ok(())
    }

    #[test]
    fn mask_plan_distances_window_and_extent() -> Result<()> {
        // two queries at positions 5 and 6 over keys 3..7, window 3, extent 2
        let plan = MaskPlan::new(5, 2, 3, 4, 2, Some(3))?;
        assert_eq!(
            plan.gather.astype(Dtype::Float32)?.to_f32()?,
            vec![1., 1., 0., 0., 3., 3., 3., 2.]
        );
        assert_eq!(
            plan.bias_valid.to_f32()?,
            vec![0., 1., 1., 0., 0., 0., 1., 1.]
        );
        let seen: Vec<bool> = plan
            .visible
            .to_f32()?
            .iter()
            .map(|x| x.is_finite())
            .collect();
        assert_eq!(seen, vec![true, true, true, false, false, true, true, true]);
        assert!(MaskPlan::new(0, 1, 0, 1, 4, None).is_ok());
        Ok(())
    }

    fn read_f32(path: &Path) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path)?;
        let (words, rest) = bytes.as_chunks::<4>();
        ensure!(rest.is_empty(), "truncated oracle file");
        Ok(words.iter().map(|w| f32::from_le_bytes(*w)).collect())
    }

    /// Relative RMSE of the vocabulary-centered logits, per position.
    fn centered_rel_rmse(actual: &[f32], expected: &[f32]) -> f64 {
        let mean = |v: &[f32]| v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
        let (ma, me) = (mean(actual), mean(expected));
        let (mut num, mut den) = (0f64, 0f64);
        for (&a, &e) in actual.iter().zip(expected) {
            let (a, e) = (a as f64 - ma, e as f64 - me);
            num += (a - e) * (a - e);
            den += e * e;
        }
        (num / den.max(1e-30)).sqrt()
    }

    fn argmax(v: &[f32]) -> usize {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    }

    /// Engine logits against a Transformers oracle written by the conversion driver
    /// (`oracle_to_bin.py`): one pass, chunked prefill and token-by-token decode.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ORACLE"]
    fn matches_oracle_logits() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let oracle = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_ORACLE")?);
        let limit: f64 = std::env::var("MLXL3_INKLING_MAX_RELRMSE")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(0.05);
        let chunk: usize = std::env::var("MLXL3_INKLING_CHUNK")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(7);
        let manifest: serde_json::Value =
            serde_json::from_reader(File::open(oracle.join("manifest.json"))?)?;
        let mut model = Inkling2::load_with(
            Path::new(&model_dir),
            Precision::from_env_or(Precision::FULL)?,
        )?;
        let mut worst = 0f64;
        let mut report = Vec::new();
        for (index, row) in manifest["rows"]
            .as_array()
            .context("rows")?
            .iter()
            .enumerate()
        {
            let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone())?;
            let tail = row["tail"].as_u64().context("tail")? as usize;
            let vocab = row["vocab"].as_u64().context("vocab")? as usize;
            let expected = read_f32(&oracle.join(row["file"].as_str().context("file")?))?;
            ensure!(
                expected.len() == tail * vocab && tail <= tokens.len(),
                "row {index}: bad oracle"
            );
            let start = tokens.len() - tail;
            let mut modes: Vec<(&str, Vec<Vec<f32>>)> = Vec::new();
            // one pass
            model.reset();
            let all = model.forward_all(&tokens)?;
            ensure!(
                all.shape() == [1, tokens.len() as i32, vocab as i32],
                "row {index}: logits shape"
            );
            let values = all.to_f32()?;
            modes.push((
                "full",
                (start..tokens.len())
                    .map(|t| values[t * vocab..(t + 1) * vocab].to_vec())
                    .collect(),
            ));
            // chunked prefill
            model.reset();
            let mut rows_out = Vec::new();
            for piece in tokens.chunks(chunk) {
                let out = model.forward_all(piece)?.to_f32()?;
                ensure!(
                    out.len() == piece.len() * vocab,
                    "row {index}: chunk logits"
                );
                rows_out.extend(out.chunks_exact(vocab).map(<[f32]>::to_vec));
            }
            modes.push(("chunked", rows_out[start..].to_vec()));
            // prefill then decode the tail token by token
            model.reset();
            let mut decoded = Vec::new();
            if start > 0 {
                let out = model.forward_all(&tokens[..start])?.to_f32()?;
                ensure!(out.len() == start * vocab, "row {index}: prefill logits");
            }
            for &token in &tokens[start..] {
                let out = model.forward(token)?;
                ensure!(
                    out.shape() == [1, 1, vocab as i32],
                    "row {index}: decode shape"
                );
                decoded.push(out.to_f32()?);
            }
            modes.push(("decode", decoded));
            if let Ok(dump) = std::env::var("MLXL3_INKLING_DUMP") {
                let dir = std::path::PathBuf::from(dump);
                std::fs::create_dir_all(&dir)?;
                for (mode, actual) in &modes {
                    let bytes: Vec<u8> = actual
                        .iter()
                        .flatten()
                        .flat_map(|x| x.to_le_bytes())
                        .collect();
                    std::fs::write(dir.join(format!("row{index}.{mode}.f32")), bytes)?;
                }
            }
            for (mode, actual) in &modes {
                let mut max_rel = 0f64;
                let mut agree = 0usize;
                for (t, logits) in actual.iter().enumerate() {
                    let want = &expected[t * vocab..(t + 1) * vocab];
                    ensure!(
                        logits.iter().all(|x| x.is_finite()),
                        "row {index} {mode}: non-finite logits"
                    );
                    max_rel = max_rel.max(centered_rel_rmse(logits, want));
                    agree += usize::from(argmax(logits) == argmax(want));
                }
                worst = worst.max(max_rel);
                report.push(
                    serde_json::json!({"row": index, "mode": mode, "positions": actual.len(),
                    "max_rel_rmse": max_rel, "argmax_agree": agree as f64 / actual.len() as f64}),
                );
            }
        }
        println!("{}", serde_json::to_string_pretty(&report)?);
        ensure!(
            worst <= limit,
            "max centered relative RMSE {worst} > {limit}"
        );
        Ok(())
    }

    fn fnv(hash: &mut u64, bytes: &[u8]) {
        for &b in bytes {
            *hash ^= u64::from(b);
            *hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    }

    fn quantiles(mut v: Vec<f64>) -> (f64, f64, f64) {
        v.sort_by(f64::total_cmp);
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize];
        (q(0.25), q(0.5), q(0.75))
    }

    /// Decode timing harness: fixed prompt, greedy decode, per-step construction/evaluation time and a
    /// FNV-1a fingerprint of every decoded FP32 logit and token (lossless-change gate).
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL"]
    fn decode_profile() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let var = |name: &str, default: usize| -> Result<usize> {
            Ok(std::env::var(name)
                .ok()
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(default))
        };
        let steps = var("MLXL3_INKLING_STEPS", 128)?;
        let reps = var("MLXL3_INKLING_REPS", 3)?;
        let prompt = profile_prompt()?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let argmax_of = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i as u32)
        };
        let once = |model: &mut Inkling2, steps: usize| -> Result<serde_json::Value> {
            model.reset();
            model.take_profile();
            let t = Instant::now();
            let logits = model.forward_tokens(&prompt)?;
            let prefill = t.elapsed().as_secs_f64() * 1e3;
            let mut next = argmax_of(&logits.to_f32()?);
            let (mut total, mut build, mut eval) = (Vec::new(), Vec::new(), Vec::new());
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            let mut tokens = Vec::new();
            model.take_profile();
            for _ in 0..steps {
                let t = Instant::now();
                let logits = model.forward(next)?;
                total.push(t.elapsed().as_secs_f64() * 1e3);
                let p = model.take_profile();
                build.push(p.build.as_secs_f64() * 1e3);
                eval.push(p.eval.as_secs_f64() * 1e3);
                let values = logits.to_f32()?;
                for x in &values {
                    fnv(&mut hash, &x.to_le_bytes());
                }
                next = argmax_of(&values);
                fnv(&mut hash, &next.to_le_bytes());
                tokens.push(next);
            }
            let (t25, t50, t75) = quantiles(total);
            let (_, b50, _) = quantiles(build);
            let (_, e50, _) = quantiles(eval);
            Ok(
                serde_json::json!({"prompt_tokens": prompt.len(), "prefill_ms": prefill, "steps": steps,
                "decode_ms_p25": t25, "decode_ms_median": t50, "decode_ms_p75": t75,
                "build_ms_median": b50, "eval_ms_median": e50, "decode_tok_s": 1000.0 / t50,
                "fingerprint": format!("{hash:016x}"), "tokens_head": &tokens[..tokens.len().min(16)]}),
            )
        };
        once(&mut model, 16)?;
        let runs = (0..reps)
            .map(|_| once(&mut model, steps))
            .collect::<Result<Vec<_>>>()?;
        println!("{}", serde_json::to_string_pretty(&runs)?);
        ensure!(
            runs.iter()
                .all(|r| r["fingerprint"] == runs[0]["fingerprint"]),
            "decode is not deterministic across repetitions"
        );
        Ok(())
    }

    /// Prompt of the decode harnesses: first row of `MLXL3_INKLING_ORACLE`, else a fixed 4-token prompt.
    fn profile_prompt() -> Result<Vec<u32>> {
        Ok(match std::env::var("MLXL3_INKLING_ORACLE") {
            Ok(dir) => {
                let manifest: serde_json::Value =
                    serde_json::from_reader(File::open(Path::new(&dir).join("manifest.json"))?)?;
                serde_json::from_value(manifest["rows"][0]["tokens"].clone())?
            }
            Err(_) => vec![200002, 200004, 34182, 279],
        })
    }

    /// `decode_profile` fingerprint of decoded logit rows: FNV-1a of each FP32 row, then of its argmax.
    fn rows_fingerprint(rows: &[Vec<f32>]) -> (u64, Vec<u32>) {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut argmax = Vec::with_capacity(rows.len());
        for row in rows {
            for x in row {
                fnv(&mut hash, &x.to_le_bytes());
            }
            let best = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i as u32);
            fnv(&mut hash, &best.to_le_bytes());
            argmax.push(best);
        }
        (hash, argmax)
    }

    fn take_run_cpu() -> (Duration, Duration) {
        RUN_CPU.with(|cell| cell.replace((Duration::ZERO, Duration::ZERO)))
    }

    /// SPEED-INK2M-P13: CPU/GPU split of a decode step and the ceiling of a one-step-ahead pipeline. A
    /// greedy pass fixes the fed tokens; they are replayed synchronously (`sync`) and with step i submitted
    /// before the logits of step i - 1 are read (`ahead`). Every mode must hash like the greedy pass.
    /// `MLXL3_INKLING_PIPE_MUTATE=1` feeds a wrong token at step 10 of the `ahead` replays (must fail).
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL"]
    fn decode_pipeline_profile() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let var = |name: &str, default: usize| -> Result<usize> {
            Ok(std::env::var(name)
                .ok()
                .map(|v| v.parse())
                .transpose()?
                .unwrap_or(default))
        };
        let steps = var("MLXL3_INKLING_STEPS", 96)?;
        let reps = var("MLXL3_INKLING_REPS", 3)?;
        let mutate = std::env::var("MLXL3_INKLING_PIPE_MUTATE").is_ok_and(|v| v == "1");
        ensure!(steps > 10, "MLXL3_INKLING_STEPS must exceed 10");
        let prompt = profile_prompt()?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let vocab = usize::try_from(model.unpadded_vocab())?;
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let median = |v: Vec<f64>| quantiles(v).1;
        // Greedy reference: the fed tokens and the rows they produce.
        let greedy = |model: &mut Inkling2, steps: usize| -> Result<(Vec<u32>, Vec<Vec<f32>>)> {
            model.reset();
            let first = model.forward_tokens(&prompt)?.to_f32()?;
            let mut fed = vec![rows_fingerprint(&[first]).1[0]];
            let mut rows = Vec::with_capacity(steps);
            for _ in 0..steps {
                let row = model.forward(*fed.last().context("fed")?)?.to_f32()?;
                fed.push(rows_fingerprint(std::slice::from_ref(&row)).1[0]);
                rows.push(row);
            }
            fed.pop();
            Ok((fed, rows))
        };
        let replay =
            |model: &mut Inkling2, fed: &[u32], ahead: bool| -> Result<serde_json::Value> {
                model.reset();
                model.forward_tokens(&prompt)?.to_f32()?;
                model.take_profile();
                take_run_cpu();
                let mut rows = Vec::with_capacity(fed.len());
                let (mut step, mut build, mut eval, mut read) = (vec![], vec![], vec![], vec![]);
                let (mut build_cpu, mut eval_cpu) = (vec![], vec![]);
                ASYNC_SUBMIT.with(|f| f.set(ahead));
                let started = Instant::now();
                let result = (|| -> Result<()> {
                    let mut in_flight: Option<Array> = None;
                    let mut mark = Instant::now();
                    for (i, &token) in fed.iter().enumerate() {
                        let token = if mutate && ahead && i == 10 {
                            (token + 1) % vocab as u32
                        } else {
                            token
                        };
                        let logits = model.forward(token)?;
                        let p = model.take_profile();
                        let (b, e) = take_run_cpu();
                        build.push(ms(p.build));
                        eval.push(ms(p.eval));
                        build_cpu.push(ms(b));
                        eval_cpu.push(ms(e));
                        let t = Instant::now();
                        if let Some(previous) = in_flight.replace(logits) {
                            rows.push(previous.to_f32()?);
                        }
                        if !ahead {
                            rows.push(in_flight.take().context("logits")?.to_f32()?);
                        }
                        read.push(ms(t.elapsed()));
                        step.push(ms(mark.elapsed()));
                        mark = Instant::now();
                    }
                    if let Some(last) = in_flight.take() {
                        rows.push(last.to_f32()?);
                    }
                    Ok(())
                })();
                let wall = started.elapsed();
                ASYNC_SUBMIT.with(|f| f.set(false));
                result?;
                ensure!(
                    rows.len() == fed.len(),
                    "replay rows {} != {}",
                    rows.len(),
                    fed.len()
                );
                ensure!(
                    rows.iter()
                        .all(|r| r.len() == vocab && r.iter().all(|x| x.is_finite())),
                    "replay rows have a wrong width or non-finite logits"
                );
                let (hash, argmax) = rows_fingerprint(&rows);
                let diverged = (1..fed.len()).find(|&i| argmax[i - 1] != fed[i]);
                Ok(
                    serde_json::json!({"mode": if ahead {"ahead"} else {"sync"}, "steps": fed.len(),
                "step_ms_median": median(step), "step_ms_mean": ms(wall) / fed.len() as f64,
                "build_ms_median": median(build), "build_cpu_ms_median": median(build_cpu),
                "eval_ms_median": median(eval), "eval_cpu_ms_median": median(eval_cpu),
                "read_ms_median": median(read), "fingerprint": format!("{hash:016x}"),
                "first_divergent_feed": diverged}),
                )
            };
        greedy(&mut model, 16)?;
        let (fed, reference) = greedy(&mut model, steps)?;
        let (reference_hash, _) = rows_fingerprint(&reference);
        let reference_fp = format!("{reference_hash:016x}");
        ensure!(
            reference
                .iter()
                .all(|r| r.len() == vocab && r.iter().all(|x| x.is_finite())),
            "greedy rows have a wrong width or non-finite logits"
        );
        drop(reference);
        let mut runs = Vec::new();
        for rep in 0..reps {
            let order = if rep % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for ahead in order {
                runs.push(replay(&mut model, &fed, ahead)?);
            }
        }
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "greedy_fingerprint": reference_fp, "prompt_tokens": prompt.len(),
                "fed_head": &fed[..fed.len().min(16)], "runs": runs}))?
        );
        ensure!(
            runs.iter()
                .all(|r| r["fingerprint"] == reference_fp.as_str()),
            "a replay differs from the greedy pass"
        );
        Ok(())
    }

    /// SPEED-INK2M-P17 F1: the one-query depth kernel gives the MLX op sequence bit for bit for 1-24 sources,
    /// D = 512 / 2 304, random rows, equal rows (equal scores) and scores 48 apart (saturated softmax); a
    /// reversed reduction tree is detected; shapes outside its domain keep the op sequence.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn depth_attend_one_matches_ops() -> Result<()> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let (mut cases, mut caught) = (0, 0);
        for dims in [512i32, 2304] {
            for count in 1..=24usize {
                for kind in 0..3 {
                    let first: Vec<f32> = (0..dims).map(|_| uniform()).collect();
                    let rows: Vec<Vec<f32>> = (0..count)
                        .map(|r| match kind {
                            1 => first.clone(),
                            2 => first.iter().map(|x| x * (1 + r) as f32).collect(),
                            _ if r == 0 => first.clone(),
                            _ => (0..dims).map(|_| uniform()).collect(),
                        })
                        .collect();
                    let q: Vec<f32> = if kind == 2 {
                        first.iter().map(|x| x * 3.0).collect()
                    } else {
                        (0..dims).map(|_| uniform()).collect()
                    };
                    let sources = rows
                        .iter()
                        .map(|r| Array::from_f32(r, &[1, dims]))
                        .collect::<Result<Vec<_>>>()?;
                    let refs: Vec<&Array> = sources.iter().collect();
                    let query = Array::from_f32(&q, &[1, dims])?;
                    let expected = depth_attend(&depth_stack(&refs)?, &query)?;
                    ensure!(expected.shape() == [1, 1, dims], "reference shape");
                    let expected = expected.to_f32()?;
                    let got = depth_attend_one(&refs, &query)?.context("inside the domain")?;
                    ensure!(got.shape() == [1, 1, dims], "kernel shape");
                    let got = got.to_f32()?;
                    ensure!(
                        expected.len() == dims as usize && expected.iter().all(|x| x.is_finite()),
                        "reference values"
                    );
                    let same = |a: &[f32], b: &[f32]| {
                        a.len() == b.len()
                            && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                    };
                    ensure!(
                        same(&expected, &got),
                        "depth kernel differs: {count} sources, D {dims}, kind {kind}"
                    );
                    DEPTH_ONE_MUTATE.with(|m| m.set(true));
                    let mutant = depth_attend_one(&refs, &query);
                    DEPTH_ONE_MUTATE.with(|m| m.set(false));
                    let mutant = mutant?.context("mutant")?.to_f32()?;
                    caught += usize::from(!same(&expected, &mutant));
                    cases += 1;
                }
            }
        }
        println!("depth kernel: {cases} cases identical, mutant differs in {caught}");
        ensure!(caught > 0, "the reversed reduction tree went unnoticed");
        // outside the domain: two queries, two rows, 32 sources, FP16
        let row = Array::from_f32(&vec![0.5; 128], &[1, 128])?;
        let two = Array::from_f32(&vec![0.5; 256], &[2, 128])?;
        ensure!(
            depth_attend_one(&[&row, &row], &two)?.is_none(),
            "two queries"
        );
        ensure!(depth_attend_one(&[&two, &two], &row)?.is_none(), "two rows");
        ensure!(depth_attend_one(&[&row; 32], &row)?.is_none(), "32 sources");
        ensure!(
            depth_attend_one(&[&row; 9], &row)?.is_none(),
            "D < 16 sources (another MLX gemv)"
        );
        let half = row.astype(Dtype::Float16)?;
        ensure!(depth_attend_one(&[&half], &row)?.is_none(), "FP16 source");
        Ok(())
    }

    /// SPEED-INK2M-P17 F1 diagnostic: GPU time of 96 chained one-query depth attentions (8 sources, D 2 304;
    /// the last source of each call is the previous output), MLX op sequence against the fused kernel.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn depth_kernel_latency() -> Result<()> {
        const CALLS: usize = 96;
        let dims = 2304i32;
        let rows: Vec<Array> = (0..7)
            .map(|r| {
                let v: Vec<f32> = (0..dims)
                    .map(|i| (((i * 31 + r * 17) % 97) as f32 - 48.0) / 97.0)
                    .collect();
                Array::from_f32(&v, &[1, dims])
            })
            .collect::<Result<_>>()?;
        let q: Vec<f32> = (0..dims).map(|i| ((i % 13) as f32 - 6.0) / 13.0).collect();
        let query = Array::from_f32(&q, &[1, dims])?;
        Array::eval_many(&rows.iter().chain([&query]).collect::<Vec<_>>())?;
        let chain = |fused: bool| -> Result<f64> {
            let mut last = rows[0].try_clone()?;
            let mut outs = Vec::new();
            for _ in 0..CALLS {
                let mut sources: Vec<&Array> = rows.iter().collect();
                sources.push(&last);
                let out = if fused {
                    depth_attend_one(&sources, &query)?.context("fused")?
                } else {
                    depth_attend(&depth_stack(&sources)?, &query)?
                };
                last = out.reshape(&[1, dims])?;
                outs.push(last.try_clone()?);
            }
            let t = Instant::now();
            Array::eval_many(&outs.iter().collect::<Vec<_>>())?;
            Ok(t.elapsed().as_secs_f64() * 1e3)
        };
        for _ in 0..3 {
            chain(false)?;
            chain(true)?;
        }
        let (mut ops, mut fused) = (Vec::new(), Vec::new());
        for i in 0..20 {
            if i % 2 == 0 {
                ops.push(chain(false)?);
                fused.push(chain(true)?);
            } else {
                fused.push(chain(true)?);
                ops.push(chain(false)?);
            }
        }
        let (o, f) = (quantiles(ops).1, quantiles(fused).1);
        println!(
            "depth chain of {CALLS}: ops {o:.3} ms ({:.2} us/call), fused {f:.3} ms ({:.2} us/call)",
            o * 1e3 / CALLS as f64,
            f * 1e3 / CALLS as f64
        );
        Ok(())
    }

    /// SPEED-INK2M-P17 F2: the fused message gate (down kernel + tail kernel) gives the op sequence bit for
    /// bit: hidden 512 / 2 304, rank 64, three magnitudes (tanh saturated in the last); a reversed tail tree
    /// is detected.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn gate_kernels_match_ops() -> Result<()> {
        let mut state = 0x2545_f491_4f6c_dd1du64;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let rank = 64;
        let (mut cases, mut caught) = (0, 0);
        for hid in [512i32, 2304] {
            for scale in [0.05f32, 0.5, 4.0] {
                let mut values =
                    |n: i32, s: f32| (0..n).map(|_| uniform() * s).collect::<Vec<f32>>();
                let down = Array::from_f32(&values(rank * 2 * hid, scale * 0.1), &[rank, 2 * hid])?
                    .astype(Dtype::Float16)?;
                let forget = Array::from_f32(&values(hid * rank, scale), &[hid, rank])?
                    .astype(Dtype::BFloat16)?;
                let gain = Array::from_f32(&values(hid * rank, scale), &[hid, rank])?
                    .astype(Dtype::BFloat16)?;
                let hidden = Array::from_f32(&values(hid, 2.0), &[1, hid])?;
                let message = Array::from_f32(&values(hid, 2.0), &[1, hid])?;
                let transposed = |w: &Array| -> Result<Array> {
                    let w = w.astype(Dtype::Float32)?.transpose(&[1, 0])?;
                    w.eval()?;
                    Ok(w)
                };
                let weight_t = down.transpose(&[1, 0])?;
                Array::eval_many(&[&weight_t, &forget, &gain, &hidden, &message])?;
                let gate = |bf16: bool| -> Result<MessageGate> {
                    Ok(MessageGate {
                        down: Dense {
                            weight_t: weight_t.try_clone()?,
                        },
                        forget_t: transposed(&forget)?,
                        gain_t: transposed(&gain)?,
                        forget_bf16: bf16.then(|| forget.try_clone()).transpose()?,
                        gain_bf16: bf16.then(|| gain.try_clone()).transpose()?,
                        eps: 1e-5,
                    })
                };
                let (fused_gate, plain_gate) = (gate(true)?, gate(false)?);
                let expected = plain_gate.forward(&hidden, &message)?;
                ensure!(expected.shape() == [1, hid], "reference shape");
                let expected = expected.to_f32()?;
                let got = fused_gate.forward(&hidden, &message)?;
                ensure!(got.shape() == [1, hid], "fused shape");
                let got = got.to_f32()?;
                ensure!(expected.iter().all(|x| x.is_finite()), "reference values");
                let same = |a: &[f32], b: &[f32]| {
                    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
                };
                ensure!(
                    same(&expected, &got),
                    "fused gate differs: hidden {hid}, scale {scale}"
                );
                GATE_TAIL_MUTATE.with(|m| m.set(true));
                let mutant = fused_gate.forward(&hidden, &message);
                GATE_TAIL_MUTATE.with(|m| m.set(false));
                caught += usize::from(!same(&expected, &mutant?.to_f32()?));
                cases += 1;
            }
        }
        println!("gate kernels: {cases} cases identical, mutant differs in {caught}");
        ensure!(caught > 0, "the reversed tail tree went unnoticed");
        Ok(())
    }

    /// SPEED-INK2M-P17 F3: the fused router logits give `float(x) @ router_t` bit for bit (2 304 x 128 and
    /// 2 048 x 128, three magnitudes); a swapped reduction order is detected through the plain op sum.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn router_kernel_matches_ops() -> Result<()> {
        let mut state = 0x5851_f42d_4c95_7f2du64;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let mut cases = 0;
        for (experts, hid) in [(128i32, 2304i32), (128, 2048)] {
            for scale in [0.02f32, 0.3, 3.0] {
                let w: Vec<f32> = (0..experts * hid).map(|_| uniform() * scale).collect();
                let w = Array::from_f32(&w, &[experts, hid])?.astype(Dtype::BFloat16)?;
                let router_t = w.astype(Dtype::Float32)?.transpose(&[1, 0])?;
                let x: Vec<f32> = (0..hid).map(|_| uniform() * 2.0).collect();
                let x = Array::from_f32(&x, &[1, hid])?.astype(Dtype::Float16)?;
                Array::eval_many(&[&w, &router_t, &x])?;
                let moe_bias = Array::zeros(&[experts])?;
                let expected = x.astype(Dtype::Float32)?.matmul(&router_t)?;
                let kernel = crate::array::prepared_kernel(
                    "inkling_router_logits_test",
                    [
                        i64::from(experts),
                        i64::from(hid),
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ],
                    || {
                        crate::array::PreparedKernel::new(
                            &format!("inkling_router_logits_{experts}_{hid}"),
                            &["x", "w"],
                            &["out"],
                            "",
                            &ROUTER_SOURCE.replace("HID", &hid.to_string()),
                        )
                    },
                )?;
                let got = kernel
                    .run(
                        &[&x, &w],
                        &[&[1, experts]],
                        &[Dtype::Float32],
                        [256 * (experts / 4), 1, 1],
                        [256, 1, 1],
                    )?
                    .pop()
                    .context("logits")?;
                let _ = moe_bias;
                let (expected, got) = (expected.to_f32()?, got.to_f32()?);
                ensure!(
                    expected.len() == experts as usize && expected.iter().all(|v| v.is_finite()),
                    "reference logits"
                );
                ensure!(
                    expected
                        .iter()
                        .zip(&got)
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "router kernel differs: {experts} x {hid}, scale {scale}"
                );
                // sensitivity: the same products summed in plain order differ somewhere
                let xs = x.astype(Dtype::Float32)?.to_f32()?;
                let ws = w.astype(Dtype::Float32)?.to_f32()?;
                let plain: Vec<f32> = (0..experts as usize)
                    .map(|e| {
                        (0..hid as usize)
                            .fold(0f32, |acc, k| acc + ws[e * hid as usize + k] * xs[k])
                    })
                    .collect();
                ensure!(
                    plain
                        .iter()
                        .zip(&got)
                        .any(|(a, b)| a.to_bits() != b.to_bits()),
                    "a plain sequential sum matches too: the check cannot see the order"
                );
                cases += 1;
            }
        }
        println!("router kernel: {cases} cases identical");
        Ok(())
    }

    /// SPEED-INK2M-P17 paired timing: one process, the model loaded once, `MLXL3_INKLING_AB_ROUNDS` rounds (8)
    /// of A (all P17 steps on) and B (`MLXL3_INKLING_AB_OFF` off) in alternating order, 96 one-step-ahead
    /// steps each; every pass must hash like the greedy reference.
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL"]
    fn fusion_ab_profile() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let off = std::env::var("MLXL3_INKLING_AB_OFF")?;
        let rounds: usize = std::env::var("MLXL3_INKLING_AB_ROUNDS")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(8);
        let steps = 96usize;
        let prompt = profile_prompt()?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        // greedy reference with every step on
        P17_TEST_OFF.with(|c| *c.borrow_mut() = Some(String::new()));
        model.reset();
        let mut logits = model.forward_tokens(&prompt)?;
        let (mut fed, mut reference) = (Vec::new(), Vec::new());
        for _ in 0..steps {
            let token = logits.chat_greedy_ids()?[0];
            fed.push(token);
            logits = model.forward(token)?;
            reference.push(logits.to_f32()?);
        }
        let reference = rows_fingerprint(&reference).0;
        let pass = |model: &mut Inkling2, off: &str| -> Result<(f64, u64)> {
            P17_TEST_OFF.with(|c| *c.borrow_mut() = Some(off.to_string()));
            model.reset();
            model.forward_tokens(&prompt)?.to_f32()?;
            ASYNC_SUBMIT.with(|f| f.set(true));
            let result = (|| -> Result<(Vec<f64>, Vec<Vec<f32>>)> {
                let (mut times, mut rows) = (Vec::new(), Vec::new());
                let mut in_flight: Option<Array> = None;
                let mut mark = Instant::now();
                for &token in &fed {
                    let logits = model.forward(token)?;
                    if let Some(previous) = in_flight.replace(logits) {
                        rows.push(previous.to_f32()?);
                    }
                    times.push(mark.elapsed().as_secs_f64() * 1e3);
                    mark = Instant::now();
                }
                if let Some(last) = in_flight.take() {
                    rows.push(last.to_f32()?);
                }
                Ok((times, rows))
            })();
            ASYNC_SUBMIT.with(|f| f.set(false));
            let (times, rows) = result?;
            ensure!(rows.len() == steps, "pass rows");
            Ok((quantiles(times).1, rows_fingerprint(&rows).0))
        };
        pass(&mut model, "")?;
        pass(&mut model, &off)?;
        let (mut a, mut b, mut diffs) = (Vec::new(), Vec::new(), Vec::new());
        for round in 0..rounds {
            let order = if round % 2 == 0 {
                ["", off.as_str()]
            } else {
                [off.as_str(), ""]
            };
            let mut times = [0f64; 2];
            for arm in order {
                let (median, hash) = pass(&mut model, arm)?;
                ensure!(
                    hash == reference,
                    "round {round}: pass with off={arm:?} differs from the reference"
                );
                times[usize::from(!arm.is_empty())] = median;
            }
            a.push(times[0]);
            b.push(times[1]);
            diffs.push(times[1] - times[0]);
        }
        P17_TEST_OFF.with(|c| *c.borrow_mut() = None);
        let wins = diffs.iter().filter(|d| **d > 0.0).count();
        println!(
            "{}",
            serde_json::json!({"off": off, "rounds": rounds, "reference": format!("{reference:016x}"),
                "a_ms": a, "b_ms": b, "b_minus_a_ms": diffs, "median_a": quantiles(a.clone()).1,
                "median_b": quantiles(b.clone()).1, "median_diff": quantiles(diffs.clone()).1,
                "b_slower_rounds": wins})
        );
        Ok(())
    }

    /// SPEED-INK2M-P17 F5: the attention output gate kernel gives the op sequence bit for bit (3 072 values,
    /// magnitudes 0.1 / 4 / 40, saturated sigmoid included).
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn attention_gate_kernel_matches_ops() -> Result<()> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ 0x1234;
        let mut uniform = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let same = |a: &[f32], b: &[f32]| {
            a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
        };
        for scale in [0.1f32, 4.0, 40.0] {
            let size = 3072;
            let a: Vec<f32> = (0..size).map(|_| uniform() * 8.0).collect();
            let g: Vec<f32> = (0..size).map(|_| uniform() * scale).collect();
            let attended = Array::from_f32(&a, &[1, size])?.astype(Dtype::Float16)?;
            let gate = Array::from_f32(&g, &[1, size])?.astype(Dtype::Float16)?;
            let expected = attended
                .astype(Dtype::Float32)?
                .mul(&gate.astype(Dtype::Float32)?.sigmoid()?)?
                .astype(Dtype::Float16)?;
            let got = attention_gate_one(&attended, &gate)?;
            let (e, g) = (
                expected.astype(Dtype::Float32)?.to_f32()?,
                got.astype(Dtype::Float32)?.to_f32()?,
            );
            ensure!(e.iter().all(|x| x.is_finite()), "gate reference");
            ensure!(same(&e, &g), "gate kernel differs: scale {scale}");
        }
        println!("attention gate kernel: identical");
        Ok(())
    }

    /// SPEED-INK2M-P14: the GPU rule exists exactly for greedy samplers and plain tempered softmax.
    #[test]
    fn draw_rule_follows_the_sampler() -> Result<()> {
        use crate::sampling::Sampler;
        let rule = |t, k, p, r, q| -> Result<Option<DrawRule>> {
            Ok(DrawRule::for_sampler(&Sampler::new(t, k, p, r, q)?))
        };
        assert_eq!(
            DrawRule::for_sampler(&Sampler::greedy()),
            Some(DrawRule::Greedy)
        );
        assert_eq!(rule(0.0, 0, 1.0, 1.0, 0.0)?, Some(DrawRule::Greedy));
        assert_eq!(rule(0.8, 1, 1.0, 1.0, 0.0)?, Some(DrawRule::Greedy));
        assert_eq!(rule(1.0, 0, 1.0, 1.0, 0.0)?, Some(DrawRule::Softmax(1.0)));
        assert_eq!(rule(0.7, 0, 1.0, 1.0, 0.0)?, Some(DrawRule::Softmax(0.7)));
        // anything the GPU draw does not reproduce stays on the CPU sampler
        assert_eq!(rule(1.0, 40, 1.0, 1.0, 0.0)?, None);
        assert_eq!(rule(1.0, 0, 0.95, 1.0, 0.0)?, None);
        assert_eq!(rule(1.0, 0, 1.0, 1.1, 0.0)?, None);
        assert_eq!(rule(1.0, 0, 1.0, 1.0, 0.5)?, None);
        assert_eq!(rule(0.0, 0, 1.0, 1.1, 0.0)?, None);
        assert_eq!(rule(0.0, 0, 1.0, 1.0, -0.5)?, None);
        // seeds: deterministic, distinct
        let (mut a, mut b) = (DrawSeeds::new(7), DrawSeeds::new(7));
        let first: Vec<u64> = (0..1000).map(|_| a.next_seed()).collect();
        assert!(
            first
                .iter()
                .zip((0..1000).map(|_| b.next_seed()))
                .all(|(x, y)| *x == y)
        );
        let distinct: std::collections::HashSet<u64> = first.iter().copied().collect();
        assert_eq!(distinct.len(), first.len());
        assert_ne!(DrawSeeds::new(8).next_seed(), first[0]);
        Ok(())
    }

    /// SPEED-INK2M-P14: `Array::gumbel_noise` is finite, within its bounds, follows the Gumbel CDF and reaches
    /// values `random::uniform` cannot give (G > 16.64), all from fixed seeds.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn gumbel_noise_follows_the_gumbel_law_and_tail() -> Result<()> {
        const BATCH: i32 = 1 << 23;
        const BATCHES: u64 = 32;
        let points = [-1.0f64, 0.0, 1.0, 2.0, 4.0, 8.0];
        let mut below = [0u64; 6];
        let (mut lowest, mut highest, mut tail, mut cdf_n) =
            (f32::INFINITY, f32::NEG_INFINITY, 0u64, 0u64);
        for batch in 0..BATCHES {
            let noise = Array::gumbel_noise(&[BATCH], 1000 + batch)?.to_f32()?;
            ensure!(noise.len() == BATCH as usize, "noise length");
            for &g in &noise {
                ensure!(g.is_finite(), "non-finite Gumbel noise");
                lowest = lowest.min(g);
                highest = highest.max(g);
                tail += u64::from(g > 16.7);
            }
            if batch < 2 {
                cdf_n += noise.len() as u64;
                for (count, &x) in below.iter_mut().zip(&points) {
                    *count += noise.iter().filter(|&&g| f64::from(g) <= x).count() as u64;
                }
            }
        }
        println!(
            "gumbel noise: min {lowest} max {highest} tail(>16.7) {tail} of {}",
            BATCHES * BATCH as u64
        );
        ensure!(
            lowest >= -2.82 && highest <= 22.88,
            "noise outside [-2.82, 22.88]"
        );
        ensure!(tail >= 1, "no noise above 16.7: the high tail is missing");
        for (count, &x) in below.iter().zip(&points) {
            let p = (-(-x).exp()).exp();
            let observed = *count as f64 / cdf_n as f64;
            let sigma = (p * (1.0 - p) / cdf_n as f64).sqrt();
            ensure!(
                (observed - p).abs() <= 4.0 * sigma + 1e-9,
                "CDF at {x}: {observed} against {p} (sigma {sigma})"
            );
        }
        // same seed, same noise; another seed, other noise
        let a = Array::gumbel_noise(&[64], 5)?.to_f32()?;
        ensure!(
            a == Array::gumbel_noise(&[64], 5)?.to_f32()?,
            "same seed, different noise"
        );
        ensure!(
            a != Array::gumbel_noise(&[64], 6)?.to_f32()?,
            "the seed does not change the noise"
        );
        Ok(())
    }

    /// SPEED-INK2M-P14: `draw_rows` follows softmax(logits / T) (Pearson chi-square over 4 M independent
    /// rows, bins of expectation below 5 merged, p < 1e-4 bound by Wilson-Hilferty), and its greedy rule is
    /// `chat_greedy_ids`.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn draw_rows_follow_the_tempered_softmax() -> Result<()> {
        const ROWS: i32 = 1 << 20;
        let logits = [
            3.0f32, 2.5, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0, 1.5,
            0.5,
        ];
        let vocab = logits.len() as i32;
        let row: Vec<f32> = (0..ROWS).flat_map(|_| logits).collect();
        let batch = Array::from_f32(&row, &[ROWS, vocab])?;
        for (temperature, seed) in [(1.0f32, 21u64), (0.7, 22)] {
            let mut counts = vec![0f64; logits.len()];
            let mut draws_total = 0f64;
            for part in 0..4u64 {
                let draws = draw_rows(&batch, DrawRule::Softmax(temperature), seed * 10 + part)?
                    .to_u32()?;
                ensure!(draws.len() == ROWS as usize, "one draw per row");
                for &token in &draws {
                    ensure!(
                        (token as usize) < logits.len(),
                        "draw outside the vocabulary"
                    );
                    counts[token as usize] += 1.0;
                }
                draws_total += draws.len() as f64;
            }
            let weights: Vec<f64> = logits
                .iter()
                .map(|&v| (f64::from(v) / f64::from(temperature)).exp())
                .collect();
            let total: f64 = weights.iter().sum();
            // merge the least likely bins until every bin expects at least 5 draws
            let mut order: Vec<usize> = (0..logits.len()).collect();
            order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]));
            let (mut bins, mut rest) = (Vec::new(), (0f64, 0f64));
            for &i in &order {
                let expected = weights[i] / total * draws_total;
                if expected >= 5.0 {
                    bins.push((counts[i], expected));
                } else {
                    rest = (rest.0 + counts[i], rest.1 + expected);
                }
            }
            if rest.1 > 0.0 {
                bins.push(rest);
            }
            let chi2: f64 = bins.iter().map(|(c, e)| (c - e).powi(2) / e).sum();
            let dof = (bins.len() - 1) as f64;
            let z = 3.719f64;
            let bound = dof * (1.0 - 2.0 / (9.0 * dof) + z * (2.0 / (9.0 * dof)).sqrt()).powi(3);
            println!(
                "T={temperature} bins={} chi2={chi2:.2} bound={bound:.2}",
                bins.len()
            );
            ensure!(
                chi2 < bound,
                "draws do not follow softmax(logits / T): chi2 {chi2} >= {bound}"
            );
        }
        // greedy: the bridge's rule on random rows, including ties
        let mut values: Vec<f32> = (0..64 * 1000)
            .map(|i| ((i * 7919) % 1013) as f32 * 0.01)
            .collect();
        values[5] = values[3];
        let rows = Array::from_f32(&values, &[64, 1000])?;
        ensure!(
            draw_rows(&rows, DrawRule::Greedy, 0)?.to_u32()? == rows.chat_greedy_ids()?,
            "greedy draw differs from chat_greedy_ids"
        );
        let three = Array::from_f32(&values[..2000], &[1, 2, 1000])?;
        ensure!(
            draw_token(&three, DrawRule::Greedy, 0)?.to_u32()?
                == three.slice(1, 1, 2)?.chat_greedy_ids()?,
            "draw_token does not draw from the last row"
        );
        Ok(())
    }

    /// SPEED-INK2M-P14: decoding with the token on the GPU, one step ahead, gives the bridge's host decode
    /// bit for bit (greedy: every logit row and token), the pipelined and synchronous device decodes agree
    /// at T = 1 for one seed stream, and a wrong fed token is detected. `MLXL3_INKLING_STEPS` steps.
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL"]
    fn device_decode_matches_the_host_decode() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let steps: usize = std::env::var("MLXL3_INKLING_STEPS")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(96);
        ensure!(steps > 10, "MLXL3_INKLING_STEPS must exceed 10");
        let prompt = profile_prompt()?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let vocab = usize::try_from(model.unpadded_vocab())?;
        // host decode, the bridge's greedy rule: rows and fed tokens
        model.reset();
        let mut logits = model.forward_tokens(&prompt)?;
        let (mut host_rows, mut host_fed) = (Vec::new(), Vec::new());
        for _ in 0..steps {
            let token = logits.chat_greedy_ids()?[0];
            host_fed.push(token);
            logits = model.forward(token)?;
            host_rows.push(logits.to_f32()?);
        }
        let (host_hash, _) = rows_fingerprint(&host_rows);
        // device decode, one step ahead, rows read back (`mutate`: wrong token fed at step 10)
        let device_rows =
            |model: &mut Inkling2, mutate: bool| -> Result<(Vec<Vec<f32>>, Vec<u32>)> {
                model.reset();
                let prompt_logits = model.forward_tokens(&prompt)?;
                let mut next = draw_token(&prompt_logits, DrawRule::Greedy, 0)?;
                next.async_eval()?;
                let (mut rows, mut fed) = (Vec::new(), Vec::new());
                let mut in_flight: Option<Array> = None;
                for i in 0..steps {
                    let input = if mutate && i == 10 {
                        Array::from_u32(&[(single_token(&next)? + 1) % vocab as u32], &[1])?
                    } else {
                        next.try_clone()?
                    };
                    let (step_logits, _) = model.forward_device(&input, None, true)?;
                    let after = draw_token(&step_logits, DrawRule::Greedy, 0)?;
                    after.async_eval()?;
                    fed.push(single_token(&input)?);
                    if let Some(previous) = in_flight.replace(step_logits) {
                        rows.push(previous.to_f32()?);
                    }
                    next = after;
                }
                if let Some(last) = in_flight.take() {
                    rows.push(last.to_f32()?);
                }
                Ok((rows, fed))
            };
        let (rows, fed) = device_rows(&mut model, false)?;
        ensure!(
            rows.len() == steps
                && rows
                    .iter()
                    .all(|r| r.len() == vocab && r.iter().all(|x| x.is_finite())),
            "device rows have a wrong count, width or non-finite logits"
        );
        let (device_hash, _) = rows_fingerprint(&rows);
        println!(
            "host {host_hash:016x} device {device_hash:016x} fed head {:?}",
            &fed[..16]
        );
        ensure!(
            fed == host_fed,
            "device greedy tokens differ from the host decode"
        );
        ensure!(
            device_hash == host_hash,
            "device logits differ from the host decode"
        );
        let (mutated, _) = device_rows(&mut model, true)?;
        ensure!(
            rows_fingerprint(&mutated).0 != host_hash,
            "a wrong fed token went unnoticed"
        );
        // the production decoder: greedy = host tokens; T = 1 pipelined = synchronous for one seed stream
        let decode = |model: &mut Inkling2, rule: DrawRule, pipelined: bool| -> Result<Vec<u32>> {
            model.reset();
            let prompt_logits = model.forward_tokens(&prompt)?;
            let mut decode =
                DeviceDecode::start(&prompt_logits, rule, DrawSeeds::new(42), pipelined)?;
            (0..steps)
                .map(|i| decode.next_token(model, i + 1 < steps))
                .collect()
        };
        ensure!(
            decode(&mut model, DrawRule::Greedy, true)? == host_fed,
            "DeviceDecode greedy differs from the host decode"
        );
        let sampled = decode(&mut model, DrawRule::Softmax(1.0), true)?;
        ensure!(
            sampled == decode(&mut model, DrawRule::Softmax(1.0), false)?,
            "pipelined and synchronous T = 1 decodes differ for one seed stream"
        );
        ensure!(
            sampled.iter().all(|&t| (t as usize) < vocab),
            "sampled token outside the vocabulary"
        );
        ensure!(sampled != host_fed, "T = 1 decode equals the greedy decode");
        Ok(())
    }

    /// Real tokens of `MLXL3_INKLING_ROWS` (`rows.json` of the local reference) end to end.
    fn text_tokens() -> Result<Vec<u32>> {
        let path = std::env::var("MLXL3_INKLING_ROWS")?;
        let rows: serde_json::Value = serde_json::from_reader(File::open(path)?)?;
        let mut tokens = Vec::new();
        for row in rows["rows"].as_array().context("rows")? {
            let row: Vec<u32> = serde_json::from_value(row["tokens"].clone())?;
            tokens.extend(row);
        }
        ensure!(!tokens.is_empty(), "no tokens in MLXL3_INKLING_ROWS");
        Ok(tokens)
    }

    #[test]
    fn kv_capacity_rounds_and_keeps_room() {
        // full visits: the needed rows rounded up to 256
        assert_eq!(kv_capacity(1, None, 1), Some(256));
        assert_eq!(kv_capacity(256, None, 1), Some(256));
        assert_eq!(kv_capacity(257, None, 1), Some(512));
        assert_eq!(kv_capacity(1 + 1200 + 128, None, 128), Some(1536));
        // sliding visits: a full window plus a 256-row chunk, or the chunk itself when larger
        assert_eq!(kv_capacity(2, Some(1024), 1), Some(1280));
        assert_eq!(kv_capacity(1 + 1023 + 128, Some(1024), 128), Some(1280));
        assert_eq!(kv_capacity(1 + 1023 + 1200, Some(1024), 1200), Some(2304));
        assert_eq!(kv_capacity(1 + 63 + 1, Some(64), 1), Some(512));
        for needed in 1..3000 {
            for (window, time) in [(None, 1), (None, 128), (Some(64), 1), (Some(1024), 4)] {
                let cap = kv_capacity(needed, window, time).unwrap();
                let floor = window.map_or(needed, |w| needed.max(w + time.max(256)));
                assert!(cap >= floor && cap % 256 == 0 && cap < floor + 256);
            }
        }
        // invalid or overflowing requests are refused
        assert_eq!(kv_capacity(0, None, 1), None);
        assert_eq!(kv_capacity(1, None, 0), None);
        assert_eq!(kv_capacity(i32::MAX, None, 1), None);
        assert_eq!(kv_capacity(2, Some(i32::MAX), 1), None);
    }

    /// Decode cost against the context (SPEED-INK2M-P12 D2): prefill of real tokens by 128-row chunks up
    /// to each level of `MLXL3_INKLING_CONTEXTS`, then `MLXL3_INKLING_STEPS` greedy steps there. FNV-1a of
    /// every logit row read (prefill chunks and steps) for bitwise A/B checks.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn decode_context_profile() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let contexts: Vec<usize> = std::env::var("MLXL3_INKLING_CONTEXTS")
            .unwrap_or_else(|_| "256,1024,2048,4096,8192".into())
            .split(',')
            .map(str::parse)
            .collect::<Result<_, _>>()?;
        let steps: usize = std::env::var("MLXL3_INKLING_STEPS")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(32);
        ensure!(
            !contexts.is_empty() && contexts.windows(2).all(|w| w[0] < w[1]) && steps > 0,
            "contexts must increase and steps must be positive"
        );
        let text = text_tokens()?;
        ensure!(
            text.len() >= contexts[contexts.len() - 1],
            "not enough tokens for the last context"
        );
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let argmax_of = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map_or(0, |(i, _)| i as u32)
        };
        // warm-up of the chunk and step shapes
        model.forward_tokens(&text[..128])?;
        for i in 0..8 {
            model.forward(text[128 + i])?;
        }
        model.reset();
        crate::array::memory_stats(true)?;
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut fed = 0usize;
        let mut points = Vec::new();
        for &context in &contexts {
            let t = Instant::now();
            let mut last = None;
            while (model.offset() as usize) < context {
                let n = (context - model.offset() as usize).min(128);
                let logits = model.forward_tokens(&text[fed..fed + n])?.to_f32()?;
                fed += n;
                ensure!(
                    logits.len() == model.unpadded_vocab() as usize
                        && logits.iter().all(|x| x.is_finite()),
                    "invalid prefill logits"
                );
                for x in &logits {
                    fnv(&mut hash, &x.to_le_bytes());
                }
                last = Some(logits);
            }
            let prefill = t.elapsed().as_secs_f64() * 1e3;
            let mut next = argmax_of(&last.context("no prefill chunk for this context")?);
            model.take_profile();
            let (mut total, mut build, mut eval) = (Vec::new(), Vec::new(), Vec::new());
            for _ in 0..steps {
                let t = Instant::now();
                let logits = model.forward(next)?;
                total.push(t.elapsed().as_secs_f64() * 1e3);
                let p = model.take_profile();
                build.push(p.build.as_secs_f64() * 1e3);
                eval.push(p.eval.as_secs_f64() * 1e3);
                let values = logits.to_f32()?;
                ensure!(
                    values.iter().all(|x| x.is_finite()),
                    "invalid decode logits"
                );
                for x in &values {
                    fnv(&mut hash, &x.to_le_bytes());
                }
                next = argmax_of(&values);
                fnv(&mut hash, &next.to_le_bytes());
            }
            let (t25, t50, t75) = quantiles(total);
            let (_, b50, _) = quantiles(build);
            let (_, e50, _) = quantiles(eval);
            let memory = crate::array::memory_stats(false)?;
            points.push(serde_json::json!({"context": context, "prefill_ms": prefill, "steps": steps,
                "decode_ms_p25": t25, "decode_ms_median": t50, "decode_ms_p75": t75,
                "build_ms_median": b50, "eval_ms_median": e50, "fingerprint": format!("{hash:016x}"),
                "mlx_peak_bytes": memory.mlx_peak_bytes, "mlx_active_bytes": memory.mlx_active_bytes,
                "kv_inplace": kv_inplace()}));
        }
        println!("{}", serde_json::to_string_pretty(&points)?);
        Ok(())
    }

    /// K/V in place against the concatenation (SPEED-INK2M-P12 K1), same process and model: chunked
    /// prefill past the sliding window and buffer compactions, teacher-forced steps, verify calls with
    /// partial commits. Logits and features must be equal to the bit; skipping the sink rewrite must not.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn kv_inplace_keeps_the_bits() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let prefix: usize = std::env::var("MLXL3_INKLING_PREFIX")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(1500);
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let vocab = model.unpadded_vocab() as u32;
        let text: Vec<u32> = text_tokens()?.into_iter().map(|t| t % vocab).collect();
        ensure!(text.len() >= prefix + 64, "not enough tokens");
        let taps = vec![1, model.visit_count()];
        let script = |model: &mut Inkling2, inplace: bool, skip_sink: bool| -> Result<Vec<f32>> {
            KV_INPLACE_OVERRIDE.with(|f| f.set(Some(inplace)));
            KV_SKIP_SINK.with(|f| f.set(skip_sink));
            model.reset();
            let mut out = Vec::new();
            let mut fed = 0usize;
            while fed < prefix {
                let n = (prefix - fed).min(128);
                out.extend(model.forward_tokens(&text[fed..fed + n])?.to_f32()?);
                fed += n;
            }
            for _ in 0..6 {
                out.extend(model.forward(text[fed])?.to_f32()?);
                fed += 1;
            }
            for (rows, keep) in [(2, 1), (3, 3), (4, 2), (4, 4), (3, 1), (2, 2), (4, 1)] {
                let (logits, features) = model.verify(&text[fed..fed + rows], &taps)?;
                out.extend(logits.to_f32()?);
                out.extend(features.astype(Dtype::Float32)?.to_f32()?);
                model.commit(keep)?;
                fed += keep;
                for _ in 0..2 {
                    out.extend(model.forward(text[fed])?.to_f32()?);
                    fed += 1;
                }
            }
            KV_INPLACE_OVERRIDE.with(|f| f.set(None));
            KV_SKIP_SINK.with(|f| f.set(false));
            Ok(out)
        };
        let concatenated = script(&mut model, false, false)?;
        let inplace = script(&mut model, true, false)?;
        let mutated = script(&mut model, true, true)?;
        ensure!(
            !concatenated.is_empty() && concatenated.iter().all(|x| x.is_finite()),
            "empty or non-finite reference"
        );
        ensure!(
            inplace.len() == concatenated.len() && mutated.len() == concatenated.len(),
            "value counts differ"
        );
        let differ = |a: &[f32], b: &[f32]| {
            a.iter()
                .zip(b)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count()
        };
        let (same, mutation) = (
            differ(&concatenated, &inplace),
            differ(&concatenated, &mutated),
        );
        println!(
            "values {} in-place differing {same} mutation differing {mutation}",
            concatenated.len()
        );
        ensure!(same == 0, "in-place K/V changed {same} values");
        ensure!(mutation > 0, "the sink mutation was not detected");
        Ok(())
    }

    /// SPEED-2026-11-23-INK2M-MAYA RT: the fused router tail gives the routes and scores of
    /// `router::grouped_topk_biased` (which applies the sigmoid) to the bit (random logits of several amplitudes, ties, the
    /// real biases and scales); a reversed score sum must differ somewhere.
    #[test]
    #[ignore = "requires an Apple GPU"]
    fn router_tail_kernel_matches_ops() -> Result<()> {
        let mut state = 0x1234_5678u32;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state >> 8) as f32 / (1u32 << 24) as f32
        };
        let (mut cases, mut mutant_differs) = (0usize, 0usize);
        for case in 0..240 {
            let amplitude = [0.05f32, 1.0, 4.0, 12.0, 40.0][case % 5];
            let mut logits: Vec<f32> = (0..128)
                .map(|_| (uniform() - 0.5) * 2.0 * amplitude)
                .collect();
            if case % 7 == 0 {
                // ties: repeated values across experts
                for e in 0..128 {
                    logits[e] = logits[e % 9];
                }
            }
            let bias: Vec<f32> = (0..128)
                .map(|_| {
                    if case % 3 == 0 {
                        0.0
                    } else {
                        (uniform() - 0.5) * 0.2
                    }
                })
                .collect();
            let scale = [8.0f32, 2.5, 1.0][case % 3];
            let x = Array::from_f32(&logits, &[1, 128])?;
            let b = Array::from_f32(&bias, &[128])?;
            let (want_ids, want_scores) = router::grouped_topk_biased(&x, &b, 8, 1, 1, scale)?;
            let (ids, scores) = router_tail_kernel(&x, &b, scale)?;
            ensure!(
                ids.to_u32()? == want_ids.to_u32()?,
                "case {case}: routes differ"
            );
            let same = scores
                .to_f32()?
                .iter()
                .zip(want_scores.to_f32()?)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            ensure!(same, "case {case}: scores differ");
            ROUTER_TAIL_MUTATE.with(|m| m.set(true));
            let mutant = router_tail_kernel(&x, &b, scale);
            ROUTER_TAIL_MUTATE.with(|m| m.set(false));
            let (_, mutant) = mutant?;
            if mutant
                .to_f32()?
                .iter()
                .zip(want_scores.to_f32()?)
                .any(|(a, b)| a.to_bits() != b.to_bits())
            {
                mutant_differs += 1;
            }
            cases += 1;
        }
        println!(
            "router tail: {cases} cases identical to the bit, reversed sum differs in {mutant_differs}"
        );
        ensure!(
            mutant_differs > 0,
            "the reversed-sum mutation was not detected"
        );
        Ok(())
    }

    /// SPEED-2026-11-23-INK2M-MAYA ED: is the EXL3 expert kernel bound by memory or by the core? Chains of
    /// 48 dependent expert calls (gate/up + down, T = 1) on the layer-6 bank with (a) 8 distinct pseudo-random
    /// experts per call, (b) the 8 routes on one expert (3 MB, cache resident), and (c) a 576 MiB FP16 sum as
    /// the read-bandwidth reference; median of 20 after 3 warm-ups, order a/b/c/c/b/a.
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL"]
    fn expert_kernel_bound_profile() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let model = Inkling2::load(Path::new(&model_dir))?;
        let moe = &model.layers[6].moe;
        let hidden = model.hidden;
        let x0 = Array::random_uniform(&[1, hidden], 11)?
            .scalar_add(-0.5)?
            .scalar_mul(2.0)?
            .astype(Dtype::Float16)?;
        x0.eval()?;
        let scores = Array::from_f32(&[1.0; 8], &[1, 8])?;
        let mut state = 0x9e37_79b9u32;
        let mut varied = Vec::new();
        for _ in 0..48 {
            let mut ids: Vec<u32> = Vec::new();
            while ids.len() < 8 {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                let e = state % 128;
                if !ids.contains(&e) {
                    ids.push(e);
                }
            }
            varied.push(Array::from_u32(&ids, &[1, 8])?);
        }
        let same = Array::from_u32(&[5; 8], &[1, 8])?;
        let chain = |routes: &dyn Fn(usize) -> Array| -> Result<f64> {
            let mut times = Vec::new();
            for rep in 0..23 {
                let started = Instant::now();
                let mut x = x0.try_clone()?;
                for call in 0..48 {
                    let out = moe.experts.forward(&x, &routes(call), &scores)?;
                    x = out.scalar_mul(0.0)?.astype(Dtype::Float16)?.add(&x0)?;
                }
                x.eval()?;
                if rep >= 3 {
                    times.push(started.elapsed().as_secs_f64() * 1e3);
                }
            }
            times.sort_by(f64::total_cmp);
            Ok(times[times.len() / 2])
        };
        let big = Array::random_uniform(&[302_000_000], 3)?.astype(Dtype::Float16)?;
        big.eval()?;
        let copy = || -> Result<f64> {
            let mut times = Vec::new();
            for rep in 0..23 {
                let started = Instant::now();
                big.sum(0, false)?.eval()?;
                if rep >= 3 {
                    times.push(started.elapsed().as_secs_f64() * 1e3);
                }
            }
            times.sort_by(f64::total_cmp);
            Ok(times[times.len() / 2])
        };
        let a1 = chain(&|call| varied[call].try_clone().unwrap())?;
        let b1 = chain(&|_| same.try_clone().unwrap())?;
        let c1 = copy()?;
        let c2 = copy()?;
        let b2 = chain(&|_| same.try_clone().unwrap())?;
        let a2 = chain(&|call| varied[call].try_clone().unwrap())?;
        let (a, b, c) = ((a1 + a2) / 2.0, (b1 + b2) / 2.0, (c1 + c2) / 2.0);
        let bytes = 302_000_000f64 * 2.0;
        println!(
            "{}",
            serde_json::json!({
                "a_varied_ms": [a1, a2], "b_one_expert_ms": [b1, b2], "c_sum_ms": [c1, c2],
                "a_per_call_us": a / 48.0 * 1e3, "b_per_call_us": b / 48.0 * 1e3,
                "b_over_a": b / a, "read_gbps": bytes / (c / 1e3) / 1e9,
                "a_gbps_at_24mb_per_call": 24e6 * 48.0 / (a / 1e3) / 1e9,
            })
        );
        Ok(())
    }

    /// SPEED-2026-11-23-INK2M-MAYA PC: a prompt resumed from the prefix saved at its last 128-row boundary
    /// (after the turn decoded on the pipelined GPU path and verified a partial block) gives the logits of a
    /// prefill from scratch to the bit, for every chunk and 32 following steps, before and past the sliding
    /// window; the two mutations (live convolution windows, live sliding K/V) must differ, and a prefix of
    /// another lineage must be refused.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn prompt_cache_resume_matches_cold_prefill() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let vocab = model.unpadded_vocab() as u32;
        let text: Vec<u32> = text_tokens()?.into_iter().map(|t| t % vocab).collect();
        let taps = vec![model.visit_count()];
        let prefill = |model: &mut Inkling2,
                       tokens: &[u32],
                       from: usize,
                       out: &mut Vec<f32>,
                       keep: bool,
                       capture: Option<usize>|
         -> Result<Option<InklingPrefix>> {
            let mut fed = from;
            let mut saved = None;
            while fed < tokens.len() {
                let n = (tokens.len() - fed).min(128);
                let logits = model.forward_tokens(&tokens[fed..fed + n])?;
                if keep {
                    out.extend(logits.to_f32()?);
                }
                fed += n;
                if capture == Some(fed) {
                    saved = Some(model.prefix_snapshot()?);
                }
            }
            Ok(saved)
        };
        let mut total = 0usize;
        for first in [300usize, 1300] {
            let boundary = first / 128 * 128;
            ensure!(text.len() >= first + 200 + 32, "not enough tokens");
            // turn 1: prefill with the capture, then a pipelined GPU decode and a partial verify block
            model.reset();
            let mut scratch = Vec::new();
            let prefix = prefill(
                &mut model,
                &text[..first],
                0,
                &mut scratch,
                false,
                Some(boundary),
            )?
            .context("no prefix captured")?;
            ensure!(prefix.offset() as usize == boundary, "prefix offset");
            // the bridge keeps it only within the default prompt cache budget
            let retained = prefix.retained_bytes()?;
            println!("first {first}: prefix retains {retained} bytes");
            ensure!(
                retained <= crate::memory_policy::DEFAULT_PROMPT_CACHE_BYTES,
                "the prefix exceeds the default prompt cache budget"
            );
            let logits = model.forward_tokens(&text[first..first + 1])?;
            let mut decode =
                DeviceDecode::start(&logits, DrawRule::Greedy, DrawSeeds::new(7), true)?;
            let mut answer = vec![text[first]];
            for _ in 0..40 {
                answer.push(decode.next_token(&mut model, true)?);
            }
            drop(decode);
            let block = [text[first + 3], text[first + 4], text[first + 5]];
            model.verify(&block, &taps)?;
            model.commit(1)?;
            let live_epoch = model.epoch;
            let mut prompt = text[..first].to_vec();
            prompt.extend(&answer);
            prompt.extend(&text[first + 1..first + 200]);
            let steps = &text[first + 200..first + 232];
            let script = |model: &mut Inkling2, resumed: bool| -> Result<Vec<f32>> {
                let mut out = Vec::new();
                prefill(model, &prompt, boundary, &mut out, true, None)?;
                ensure!(
                    model.offset() as usize == prompt.len() || !resumed,
                    "offset"
                );
                for &t in steps {
                    out.extend(model.forward(t)?.to_f32()?);
                }
                let (logits, _) = model.verify(&steps[..4], &taps)?;
                out.extend(logits.to_f32()?);
                model.commit(4)?;
                let _ = resumed;
                Ok(out)
            };
            // resumed (twice more for the mutations, each from a fresh turn-1 state)
            let rerun = |model: &mut Inkling2, mutation: u8| -> Result<Vec<f32>> {
                model.reset();
                let mut scratch = Vec::new();
                let prefix = prefill(
                    model,
                    &text[..first],
                    0,
                    &mut scratch,
                    false,
                    Some(boundary),
                )?
                .context("no prefix captured")?;
                let logits = model.forward_tokens(&text[first..first + 1])?;
                let mut decode =
                    DeviceDecode::start(&logits, DrawRule::Greedy, DrawSeeds::new(7), true)?;
                for _ in 0..40 {
                    decode.next_token(model, true)?;
                }
                drop(decode);
                model.verify(&block, &taps)?;
                model.commit(1)?;
                PREFIX_MUTATION.with(|m| m.set(mutation));
                let resumed = model.resume_prefix(prefix);
                PREFIX_MUTATION.with(|m| m.set(0));
                ensure!(resumed?, "the prefix did not resume");
                ensure!(model.offset() as usize == boundary, "resumed offset");
                script(model, true)
            };
            ensure!(model.resume_prefix(prefix)?, "the prefix did not resume");
            ensure!(model.epoch == live_epoch, "resume changed the lineage");
            let resumed = script(&mut model, true)?;
            model.reset();
            let mut cold = Vec::new();
            prefill(&mut model, &prompt[..boundary], 0, &mut cold, false, None)?;
            let cold = {
                let mut tail = script(&mut model, false)?;
                cold.append(&mut tail);
                cold
            };
            let conv_mutant = rerun(&mut model, 1)?;
            let sliding_mutant = rerun(&mut model, 2)?;
            // a prefix of another lineage is refused and leaves the state as it was
            model.reset();
            let mut scratch = Vec::new();
            let stale = prefill(&mut model, &text[..256], 0, &mut scratch, false, Some(256))?
                .context("no prefix captured")?;
            model.reset();
            model.forward_tokens(&text[..64])?;
            ensure!(!model.resume_prefix(stale)?, "a stale prefix resumed");
            ensure!(model.offset() == 64, "a refused prefix changed the state");
            ensure!(
                !cold.is_empty() && cold.iter().all(|x| x.is_finite()),
                "empty or non-finite reference"
            );
            ensure!(
                resumed.len() == cold.len()
                    && conv_mutant.len() == cold.len()
                    && sliding_mutant.len() == cold.len(),
                "value counts differ"
            );
            let differ = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count()
            };
            let (same, conv, sliding) = (
                differ(&cold, &resumed),
                differ(&cold, &conv_mutant),
                differ(&cold, &sliding_mutant),
            );
            println!(
                "first {first} boundary {boundary} prompt {} values {} resumed differing {same} \
                 conv mutation differing {conv} sliding mutation differing {sliding}",
                prompt.len(),
                cold.len()
            );
            ensure!(same == 0, "the resumed prompt changed {same} values");
            ensure!(conv > 0, "the convolution mutation was not detected");
            // before the window moves, going back by row count is also exact (nothing to detect)
            ensure!(
                first + 42 < 1024 || sliding > 0,
                "the sliding K/V mutation was not detected"
            );
            total += cold.len();
        }
        println!("prompt cache parity: {total} values identical to the bit");
        Ok(())
    }

    /// Speculative commit mechanics on a model where batched and single-row numerics agree (tiny export with
    /// decisive routing): verify blocks with arbitrary proposals, commit partial prefixes, cross the sliding
    /// window, then compare the next-token logits with a teacher-forced AR pass over the committed tokens.
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL (tiny stable export)"]
    fn commit_rollback_matches_teacher_forced_ar() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let mut model = Inkling2::load_with(
            Path::new(&model_dir),
            Precision::from_env_or(Precision::FULL)?,
        )?;
        let taps = vec![1, 5, model.visit_count()];
        let prompt: Vec<u32> = (0..9).map(|i| 1000 + 37 * i).collect();
        let mut committed = prompt.clone();
        model.reset();
        let (_, features) = model.verify(&prompt, &taps)?;
        assert_eq!(features.shape(), [9, 3 * model.hidden_size()]);
        model.commit(prompt.len())?;
        let mut next = 5000u32;
        let mut step = 0u32;
        while committed.len() < 150 {
            let width = 1 + (step % 7) as usize;
            let mut block = vec![next];
            block.extend((0..width).map(|i| 7000 + 13 * step + i as u32));
            let keep = 1 + (step as usize * 5) % (width + 1); // 1..=width+1 rows
            let (logits, features) = model.verify(&block, &taps)?;
            assert_eq!(
                logits.shape(),
                [1, block.len() as i32, model.unpadded_vocab()]
            );
            assert_eq!(
                features.shape(),
                [block.len() as i32, 3 * model.hidden_size()]
            );
            model.commit(keep)?;
            committed.extend(&block[..keep]);
            assert_eq!(model.offset() as usize, committed.len());
            next = 3000 + step * 11;
            step += 1;
        }
        // an uncommitted verify must be refused before the next call
        model.verify(&[next], &taps)?;
        assert!(model.forward(next).is_err());
        model.reset();
        model.forward_tokens(&committed[..prompt.len()])?;
        for &t in &committed[prompt.len()..] {
            model.forward(t)?;
        }
        let ar = model.forward(next)?.to_f32()?;
        // replay the speculative schedule (same rows committed, same order)
        model.reset();
        let (_, _) = model.verify(&prompt, &taps)?;
        model.commit(prompt.len())?;
        let mut cursor = prompt.len();
        let mut next2 = 5000u32;
        let mut step = 0u32;
        while cursor < committed.len() {
            let width = 1 + (step % 7) as usize;
            let mut block = vec![next2];
            block.extend((0..width).map(|i| 7000 + 13 * step + i as u32));
            let keep = 1 + (step as usize * 5) % (width + 1);
            model.verify(&block, &taps)?;
            model.commit(keep)?;
            cursor += keep;
            next2 = 3000 + step * 11;
            step += 1;
        }
        let spec = model.forward(next)?.to_f32()?;
        let rel = centered_rel_rmse(&spec, &ar);
        println!(
            "committed {} tokens over {step} blocks, rel {rel}",
            committed.len()
        );
        ensure!(
            rel < 0.05,
            "speculative state differs from teacher-forced AR: {rel}"
        );
        Ok(())
    }

    /// SPEED-INK2M-P11 D3 (diagnostic): share of duplicated expert routes inside verification blocks of
    /// T = 2..4 consecutive real tokens (distinct experts per MoE call over T * top_k), against random routes.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn verify_expert_overlap() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let taps: Vec<usize> = vec![5, 23, 41, model.visit_count()];
        let mut report = Vec::new();
        for t in 2..=4usize {
            let (mut slots, mut distinct, mut calls) = (0usize, 0usize, 0usize);
            for row in rows["rows"].as_array().context("rows")? {
                let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone())?;
                let prefix = 64.min(tokens.len().saturating_sub(8 * t + 1));
                ensure!(prefix >= 16, "row too short");
                model.reset();
                model.forward_tokens(&tokens[..prefix])?.eval()?;
                for block in tokens[prefix..].chunks_exact(t).take(8) {
                    MOE_ROUTES.with(|c| *c.borrow_mut() = Some(Vec::new()));
                    let (logits, _) = model.verify(block, &taps)?;
                    logits.eval()?;
                    let routes = MOE_ROUTES
                        .with(|c| c.borrow_mut().take())
                        .context("routes")?;
                    model.commit(t)?;
                    for (r, ids) in routes {
                        ensure!(r == t && ids.len() % t == 0, "unexpected MoE call shape");
                        let set: std::collections::BTreeSet<u32> = ids.iter().copied().collect();
                        slots += ids.len();
                        distinct += set.len();
                        calls += 1;
                    }
                }
            }
            ensure!(calls > 0, "no MoE call captured");
            // random top-k routes: expected distinct experts of t independent draws of 8 among 128
            let (k, e) = (8.0f64, 128.0f64);
            let random = e * (1.0 - (1.0 - k / e).powi(t as i32)) / (k * t as f64);
            report.push(serde_json::json!({
                "rows": t, "calls": calls,
                "distinct_share": distinct as f64 / slots as f64,
                "duplicated_share": 1.0 - distinct as f64 / slots as f64,
                "random_distinct_share": random,
            }));
        }
        println!("{}", serde_json::to_string(&report)?);
        Ok(())
    }

    /// DSPARK-2026-11-23-INK2M-MTPLX-B0 (diagnostic): wall time of a T = 4 verification after the same real
    /// 64-token prefix for blocks whose rows share few or many experts (`synthetic` ids, `real` consecutive
    /// tokens, one `repeat`ed token), with the share of duplicated routes of each block measured in a separate
    /// untimed pass, and the T = 1 verification of the first real token as the reference.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn verify_cost_by_overlap() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let tokens: Vec<u32> = serde_json::from_value(rows["rows"][0]["tokens"].clone())?;
        ensure!(tokens.len() >= 68, "row too short");
        let prefix = &tokens[..64];
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let taps: Vec<usize> = vec![5, 23, 41, model.visit_count()];
        let blocks: Vec<(&str, Vec<u32>)> = vec![
            ("synthetic", vec![2000, 2017, 2034, 2051]),
            ("real", tokens[64..68].to_vec()),
            ("repeat", vec![tokens[64]; 4]),
            ("single", vec![tokens[64]]),
        ];
        let mut duplicated = std::collections::BTreeMap::new();
        for (name, block) in &blocks {
            model.reset();
            model.forward_tokens(prefix)?.eval()?;
            MOE_ROUTES.with(|c| *c.borrow_mut() = Some(Vec::new()));
            let (logits, _) = model.verify(block, &taps)?;
            logits.eval()?;
            let routes = MOE_ROUTES
                .with(|c| c.borrow_mut().take())
                .context("routes")?;
            let (mut slots, mut distinct) = (0usize, 0usize);
            for (_, ids) in routes {
                slots += ids.len();
                distinct += ids.iter().collect::<std::collections::BTreeSet<_>>().len();
            }
            ensure!(slots > 0, "no MoE call captured");
            duplicated.insert(*name, 1.0 - distinct as f64 / slots as f64);
        }
        let mut times: std::collections::BTreeMap<&str, Vec<f64>> = Default::default();
        let order = [0usize, 1, 2, 3, 3, 2, 1, 0];
        for _round in 0..2 {
            for &i in &order {
                let (name, block) = &blocks[i];
                for rep in 0..10 {
                    model.reset();
                    model.forward_tokens(prefix)?.eval()?;
                    let start = Instant::now();
                    model.verify(block, &taps)?;
                    let ms = start.elapsed().as_secs_f64() * 1e3;
                    if rep > 0 {
                        times.entry(name).or_default().push(ms);
                    }
                }
            }
        }
        let report: Vec<serde_json::Value> = blocks
            .iter()
            .map(|(name, block)| {
                let (q1, median, q3) = quantiles(times[name].clone());
                serde_json::json!({
                    "block": name, "rows": block.len(), "duplicated_share": duplicated[name],
                    "verify_ms_median": median, "verify_ms_q1": q1, "verify_ms_q3": q3,
                })
            })
            .collect();
        println!("VERIFY_COST_BY_OVERLAP {}", serde_json::to_string(&report)?);
        Ok(())
    }

    /// DFLASH2-2026-11-24-INK2M-P4-2 quality (TENSOR-C1 protocol, 8-row blocks): rows 0..12 of `MLXL3_INKLING_ROWS`
    /// (en / fr / code / chat x 3), 64 tokens of prefix, then 48 teacher-forced verifications of 8 rows (384
    /// positions per row); an exact pass keeps its FP32 log-probs on the GPU, then the grouped TensorOps experts
    /// (`MLXL3_INKLING_EXPERT_TENSOR` path) on the same tokens: per position KL(p_exact || p_hybrid), argmax
    /// disagreement and the log-likelihood of the next token (perplexity). Control: a second exact pass of row 0
    /// gives KL 0 and no disagreement.
    #[test]
    #[ignore = "requires an Apple M5 GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn expert_tensor_verify_quality() -> Result<()> {
        const PREFIX: usize = 64;
        const BLOCKS: usize = 48;
        const T: usize = 8;
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let taps: Vec<usize> = vec![5, 23, 41, model.visit_count()];
        let vocab = model.unpadded_vocab();
        let mut report = Vec::new();
        let (mut all_kl, mut all_max, mut all_dis, mut all_n) = (0f64, 0f64, 0usize, 0usize);
        let (mut all_nll_e, mut all_nll_h) = (0f64, 0f64);
        for index in 0..12usize {
            let name = rows["rows"][index]["name"]
                .as_str()
                .unwrap_or("?")
                .to_owned();
            let tokens: Vec<u32> = serde_json::from_value(rows["rows"][index]["tokens"].clone())?;
            let tokens: Vec<u32> = tokens.into_iter().map(|t| t % vocab as u32).collect();
            ensure!(tokens.len() > PREFIX + BLOCKS * T, "row {name} too short");
            let mut pass = |hybrid: bool| -> Result<Vec<Array>> {
                crate::moe::with_expert_tensor(hybrid, || -> Result<Vec<Array>> {
                    model.reset();
                    model.forward_tokens(&tokens[..PREFIX])?.eval()?;
                    let mut out = Vec::with_capacity(BLOCKS);
                    for b in 0..BLOCKS {
                        let block = &tokens[PREFIX + b * T..PREFIX + (b + 1) * T];
                        let (logits, _) = model.verify(block, &taps)?;
                        model.commit(T)?;
                        let lp = logits
                            .reshape(&[T as i32, -1])?
                            .slice(1, 0, vocab)?
                            .astype(Dtype::Float32)?
                            .log_probs()?;
                        lp.eval()?;
                        out.push(lp);
                    }
                    Ok(out)
                })
            };
            let exact = pass(false)?;
            let compare = |other: &[Array]| -> Result<(Vec<f64>, usize, f64, f64)> {
                let (mut kls, mut dis, mut nll_e, mut nll_h) = (Vec::new(), 0usize, 0f64, 0f64);
                for (b, (p, q)) in exact.iter().zip(other).enumerate() {
                    let kl = p.exp()?.mul(&p.sub(q)?)?.sum(1, false)?.to_f32()?;
                    ensure!(
                        kl.len() == T && kl.iter().all(|v| v.is_finite()),
                        "KL values"
                    );
                    kls.extend(kl.iter().map(|&v| f64::from(v.max(0.0))));
                    let (ap, aq) = (p.argmax()?.to_u32()?, q.argmax()?.to_u32()?);
                    ensure!(ap.len() == T && aq.len() == T, "argmax shapes");
                    dis += ap.iter().zip(&aq).filter(|(a, b)| a != b).count();
                    let next: Vec<u32> = (0..T)
                        .map(|i| i as u32 * vocab as u32 + tokens[PREFIX + b * T + i + 1])
                        .collect();
                    let idx = Array::from_u32(&next, &[T as i32])?;
                    let le = p.reshape(&[-1])?.take(&idx, 0)?.to_f32()?;
                    let lh = q.reshape(&[-1])?.take(&idx, 0)?.to_f32()?;
                    nll_e -= le.iter().map(|&v| f64::from(v)).sum::<f64>();
                    nll_h -= lh.iter().map(|&v| f64::from(v)).sum::<f64>();
                }
                Ok((kls, dis, nll_e, nll_h))
            };
            if index == 0 {
                let control = pass(false)?;
                let (kls, dis, _, _) = compare(&control)?;
                let max = kls.iter().cloned().fold(0f64, f64::max);
                println!(
                    "P4QUALITY control {name}: exact vs exact KL max {max:.3e}, disagreements {dis}"
                );
                ensure!(
                    max == 0.0 && dis == 0,
                    "the exact passes differ: the harness is not deterministic"
                );
            }
            let hybrid = pass(true)?;
            let (kls, dis, nll_e, nll_h) = compare(&hybrid)?;
            let n = kls.len();
            let mean = kls.iter().sum::<f64>() / n as f64;
            let max = kls.iter().cloned().fold(0f64, f64::max);
            let (ppl_e, ppl_h) = ((nll_e / n as f64).exp(), (nll_h / n as f64).exp());
            println!(
                "P4QUALITY {name}: positions {n} KL mean {mean:.3e} max {max:.3e} disagreements {dis} perplexity {ppl_e:.4} -> {ppl_h:.4}"
            );
            report.push(serde_json::json!({
                "row": name, "positions": n, "kl_mean": mean, "kl_max": max, "disagreements": dis,
                "perplexity_exact": ppl_e, "perplexity_hybrid": ppl_h,
            }));
            all_kl += kls.iter().sum::<f64>();
            all_max = all_max.max(max);
            all_dis += dis;
            all_n += n;
            all_nll_e += nll_e;
            all_nll_h += nll_h;
        }
        println!(
            "P4QUALITY total: positions {all_n} KL mean {:.3e} max {all_max:.3e} disagreements {all_dis} ({:.2} %) perplexity {:.4} -> {:.4}",
            all_kl / all_n as f64,
            100.0 * all_dis as f64 / all_n as f64,
            (all_nll_e / all_n as f64).exp(),
            (all_nll_h / all_n as f64).exp()
        );
        println!("P4QUALITY_JSON {}", serde_json::to_string(&report)?);
        Ok(())
    }

    /// INVARIANT-2026-11-24-INK2M-L1L4 (L1 probe): for the dense projections of layer 0, the head, the gates'
    /// `Dense` and a dynamic convolution's `weight_proj`, at M = 2 / 4 / 8 rows: the rows as a batch of 1-row
    /// products (`[M, 1, K]`) against the 1-row products (bits), and the time per call of the current batched
    /// product, the per-row loop and the `[M, 1, K]` form (16 calls per evaluation, median of 15).
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL"]
    fn invariant_batched_vector_probe() -> Result<()> {
        const CALLS: usize = 16;
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let model = Inkling2::load(Path::new(&model_dir))?;
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut random = |rows: i32, cols: i32| -> Result<Array> {
            let v: Vec<f32> = (0..rows * cols)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (((state >> 33) as u32 % 2001) as f32 / 1000.0 - 1.0) * 0.5
                })
                .collect();
            let a = Array::from_f32(&v, &[rows, cols])?.astype(Dtype::Float16)?;
            a.eval()?;
            Ok(a)
        };
        let time = |f: &dyn Fn(usize) -> Result<Array>| -> Result<f64> {
            let mut samples = Vec::new();
            for rep in 0..18 {
                let outs = (0..CALLS).map(f).collect::<Result<Vec<_>>>()?;
                let start = Instant::now();
                Array::eval_many(&outs.iter().collect::<Vec<_>>())?;
                if rep >= 3 {
                    samples.push(start.elapsed().as_secs_f64() * 1e3 / CALLS as f64);
                }
            }
            samples.sort_by(f64::total_cmp);
            Ok(samples[samples.len() / 2])
        };
        let bits = |a: &Array| -> Result<Vec<u32>> {
            Ok(a.astype(Dtype::Float32)?
                .to_f32()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        // every distinct 8-bit product shape of the model: the attention projections of all layers, the head
        let mut linears: Vec<(String, &Linear)> = Vec::new();
        for (i, layer) in model.layers.iter().enumerate() {
            let a = &layer.attention;
            for (label, l) in [
                ("q", &a.q),
                ("k", &a.k),
                ("v", &a.v),
                ("r", &a.r),
                ("o", &a.o),
                ("gate", &a.gate),
                ("sink", &a.sink),
            ] {
                linears.push((format!("l{i}.{label}"), l));
            }
        }
        linears.push((
            "head".to_owned(),
            model.head_q.as_ref().context("8-bit head")?,
        ));
        let mut seen = std::collections::BTreeSet::new();
        let mut report = Vec::new();
        let mut checked = 0usize;
        for (label, linear) in &linears {
            let Linear::Affine {
                w,
                scales,
                biases,
                bits: b,
                group,
            } = linear
            else {
                continue;
            };
            let k = scales.shape()[1] * *group;
            let n = scales.shape()[0];
            let fresh = seen.insert((k, n, *b, *group));
            for m in 1..=8i32 {
                let x = random(m, k)?;
                let single = per_row(&x, |r| linear.forward_rows(r))?;
                let twin = qmv_rows(&x, w, scales, biases, *group, *b)?
                    .context("twin outside its domain")?;
                ensure!(twin.shape() == single.shape(), "{label} m {m}: twin shape");
                let (t, s1) = (bits(&twin)?, bits(&single)?);
                ensure!(t.len() == (m * n) as usize, "{label} m {m}: twin size");
                let differ = t.iter().zip(&s1).filter(|(a, b)| a != b).count();
                ensure!(
                    differ == 0,
                    "{label} (K {k}, N {n}, {b} bits, group {group}) m {m}: {differ} values differ from the 1-row product"
                );
                checked += 1;
                if fresh && matches!(m, 2 | 4 | 8) {
                    let xs = (0..CALLS)
                        .map(|_| random(m, k))
                        .collect::<Result<Vec<_>>>()?;
                    let current = |i: usize| linear.forward_rows(&xs[i]);
                    let rows = |i: usize| per_row(&xs[i], |r| linear.forward_rows(r));
                    let twin_f = |i: usize| -> Result<Array> {
                        qmv_rows(&xs[i], w, scales, biases, *group, *b)?.context("twin")
                    };
                    let (tc, ts, tt) = (time(&current)?, time(&rows)?, time(&twin_f)?);
                    println!(
                        "L1TWIN {label} K {k} N {n}: m {m} identical | current {tc:.4} per-row {ts:.4} twin {tt:.4} ms"
                    );
                    report.push(
                        serde_json::json!({"product": label, "k": k, "n": n, "rows": m,
                        "current_ms": tc, "per_row_ms": ts, "twin_ms": tt}),
                    );
                }
            }
        }
        println!("L1TWIN all {checked} (product, rows) cases identical to the 1-row products");
        println!("L1TWIN_JSON {}", serde_json::to_string(&report)?);
        Ok(())
    }

    /// INVARIANT-2026-11-24-INK2M-L1L4 oracle: with `MLXL3_INKLING_INVARIANT`, every row of `verify` (logits and
    /// tap features) equals the AR step on the same state bit for bit, and so does the step after `commit` (the
    /// committed states): rows en0 / fr0 / code0 / chat0, prefixes of 64 and 300 real tokens, blocks of 2, 3, 5
    /// and 8 tokens. The same comparison with the option off must find differences (the harness can fail).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn verify_matches_ar_bitwise() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let vocab = model.unpadded_vocab() as u32;
        let taps: Vec<usize> = vec![5, 23, 41, model.visit_count()];
        let bits = |a: &Array| -> Result<Vec<u32>> {
            Ok(a.astype(Dtype::Float32)?
                .to_f32()?
                .iter()
                .map(|v| v.to_bits())
                .collect())
        };
        let mut checked = 0usize;
        let mut baseline_differs = 0usize;
        for index in [0usize, 3, 6, 9] {
            let name = rows["rows"][index]["name"]
                .as_str()
                .unwrap_or("?")
                .to_owned();
            let tokens: Vec<u32> = serde_json::from_value(rows["rows"][index]["tokens"].clone())?;
            let tokens: Vec<u32> = tokens.into_iter().map(|t| t % vocab).collect();
            for prefix in [64usize, 300] {
                for t in [2usize, 3, 5, 8] {
                    let block = &tokens[prefix..prefix + t];
                    let after = tokens[prefix + t];
                    // AR: one forward per token, then the next step
                    model.reset();
                    model.forward_tokens(&tokens[..prefix])?.eval()?;
                    let mut ar_logits = Vec::new();
                    let mut ar_features = Vec::new();
                    for &token in block {
                        let (l, f) = model.forward_with_features(&[token], &taps)?;
                        ar_logits.push(bits(&l)?);
                        ar_features.push(bits(&f)?);
                    }
                    let ar_next = bits(&model.forward(after)?)?;
                    // (logit rows, feature rows, the step after commit), as bits
                    type VerifyBits = (Vec<Vec<u32>>, Vec<Vec<u32>>, Vec<u32>);
                    let run = |model: &mut Inkling2, invariant: bool| -> Result<VerifyBits> {
                        with_invariant_rows(invariant, || {
                            model.reset();
                            model.forward_tokens(&tokens[..prefix])?.eval()?;
                            let (logits, features) = model.verify(block, &taps)?;
                            let rows_l = (0..t as i32)
                                .map(|j| {
                                    bits(&logits.reshape(&[t as i32, -1])?.slice(0, j, j + 1)?)
                                })
                                .collect::<Result<Vec<_>>>()?;
                            let rows_f = (0..t as i32)
                                .map(|j| bits(&features.slice(0, j, j + 1)?))
                                .collect::<Result<Vec<_>>>()?;
                            model.commit(t)?;
                            let next = bits(&model.forward(after)?)?;
                            Ok((rows_l, rows_f, next))
                        })
                    };
                    let (vl, vf, vn) = run(&mut model, true)?;
                    for j in 0..t {
                        ensure!(
                            !vl[j].is_empty() && vl[j].len() == ar_logits[j].len(),
                            "{name} prefix {prefix} T {t}: logits row {j} size"
                        );
                        ensure!(
                            vl[j] == ar_logits[j],
                            "{name} prefix {prefix} T {t}: logits row {j} differ from AR"
                        );
                        ensure!(
                            vf[j] == ar_features[j],
                            "{name} prefix {prefix} T {t}: features row {j} differ from AR"
                        );
                    }
                    ensure!(
                        vn == ar_next,
                        "{name} prefix {prefix} T {t}: the step after commit differs from AR"
                    );
                    checked += 1;
                    if t == 8 {
                        let (bl, _, _) = run(&mut model, false)?;
                        baseline_differs += usize::from((0..t).any(|j| bl[j] != ar_logits[j]));
                    }
                    println!(
                        "INVARIANT {name} prefix {prefix} T {t}: verify == AR (logits, features, next step)"
                    );
                }
            }
        }
        ensure!(checked == 32, "{checked} cases checked");
        ensure!(
            baseline_differs > 0,
            "the option-off baseline never differs: the comparison cannot fail"
        );
        println!(
            "INVARIANT all {checked} cases identical; option off differs in {baseline_differs}/8 cases at T = 8"
        );
        Ok(())
    }

    /// INVARIANT-2026-11-24-INK2M-L1L4 cost: median wall time of `verify` (evaluated) for T = 1..=8 after a real
    /// 64-token prefix, option off / on / on / off (7 timed repetitions after 2 warm-ups per cell).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn invariant_verify_cost() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let tokens: Vec<u32> = serde_json::from_value(rows["rows"][6]["tokens"].clone())?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let vocab = model.unpadded_vocab() as u32;
        let tokens: Vec<u32> = tokens.into_iter().map(|t| t % vocab).collect();
        let taps: Vec<usize> = vec![5, 23, 41, model.visit_count()];
        let mut cells = Vec::new();
        for t in 1..=8usize {
            let mut arms = Vec::new();
            for invariant in [false, true, true, false] {
                let mut times = Vec::new();
                with_invariant_rows(invariant, || -> Result<()> {
                    for rep in 0..9 {
                        model.reset();
                        model.forward_tokens(&tokens[..64])?.eval()?;
                        let block = &tokens[64 + rep..64 + rep + t];
                        let start = Instant::now();
                        let (logits, _) = model.verify(block, &taps)?;
                        logits.eval()?;
                        let ms = start.elapsed().as_secs_f64() * 1e3;
                        model.commit(t)?;
                        if rep >= 2 {
                            times.push(ms);
                        }
                    }
                    Ok(())
                })?;
                arms.push(quantiles(times).1);
            }
            println!(
                "INVCOST rows {t}: off {:.2} / {:.2} invariant {:.2} / {:.2} ms",
                arms[0], arms[3], arms[1], arms[2]
            );
            cells.push(serde_json::json!({"rows": t, "off_ms": [arms[0], arms[3]], "invariant_ms": [arms[1], arms[2]]}));
        }
        println!("INVCOST_JSON {}", serde_json::to_string(&cells)?);
        Ok(())
    }

    /// INVARIANT-2026-11-24-INK2M-L3-2 cost: median wall time of `verify` (evaluated) for T = 1..=8 after a real
    /// 64-token prefix: batched products (invariant off), invariant without the expert panel, invariant with it;
    /// order off / no panel / panel / panel / no panel / off, 7 timed repetitions after 2 warm-ups per cell.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn expert_panel_verify_cost() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let tokens: Vec<u32> = serde_json::from_value(rows["rows"][6]["tokens"].clone())?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let vocab = model.unpadded_vocab() as u32;
        let tokens: Vec<u32> = tokens.into_iter().map(|t| t % vocab).collect();
        let taps: Vec<usize> = vec![5, 23, 41, model.visit_count()];
        let mut cells = Vec::new();
        for t in 1..=8usize {
            let mut arms = std::collections::BTreeMap::<&str, Vec<f64>>::new();
            for (label, invariant, panel) in [
                ("off", false, false),
                ("invariant", true, false),
                ("panel", true, true),
                ("panel", true, true),
                ("invariant", true, false),
                ("off", false, false),
            ] {
                let mut times = Vec::new();
                with_invariant_rows(invariant, || {
                    crate::moe::with_expert_panel(panel, || -> Result<()> {
                        for rep in 0..9 {
                            model.reset();
                            model.forward_tokens(&tokens[..64])?.eval()?;
                            let block = &tokens[64 + rep..64 + rep + t];
                            let start = Instant::now();
                            let (logits, _) = model.verify(block, &taps)?;
                            logits.eval()?;
                            let ms = start.elapsed().as_secs_f64() * 1e3;
                            model.commit(t)?;
                            if rep >= 2 {
                                times.push(ms);
                            }
                        }
                        Ok(())
                    })
                })?;
                arms.entry(label).or_default().push(quantiles(times).1);
            }
            println!(
                "PANELCOST rows {t}: off {:.2?} invariant {:.2?} panel {:.2?} ms",
                arms["off"], arms["invariant"], arms["panel"]
            );
            cells.push(serde_json::json!({"rows": t, "off_ms": arms["off"], "invariant_ms": arms["invariant"], "panel_ms": arms["panel"]}));
        }
        println!("PANELCOST_JSON {}", serde_json::to_string(&cells)?);
        Ok(())
    }

    /// INVARIANT-2026-11-24-INK2M-L1L4 (D): stage by stage, row j of an 8-token `verify` against AR step j (same
    /// prefix, same states): for every stage occurrence in execution order, the rows whose bits differ and the
    /// largest absolute difference; prints every differing stage of the first visits and the first differing
    /// stage of each visit (`MLXL3_INKLING_ROWS` row 6, prefix 64; `INVARIANT_BLOCK` rows, default 8).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn verify_stage_divergence() -> Result<()> {
        let t: usize = std::env::var("INVARIANT_BLOCK")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let tokens: Vec<u32> = serde_json::from_value(rows["rows"][6]["tokens"].clone())?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let vocab = model.unpadded_vocab() as u32;
        let tokens: Vec<u32> = tokens.into_iter().map(|t| t % vocab).collect();
        let taps: Vec<usize> = vec![5, 23, 41, model.visit_count()];
        let capture = |run: &mut dyn FnMut() -> Result<()>| -> Result<Vec<(&'static str, Array)>> {
            STAGES.with(|c| *c.borrow_mut() = Some(Vec::new()));
            let result = run();
            let stages = STAGES.with(|c| c.borrow_mut().take()).context("stages")?;
            result?;
            Ok(stages)
        };
        model.reset();
        model.forward_tokens(&tokens[..64])?.eval()?;
        let block = tokens[64..64 + t].to_vec();
        let invariant = std::env::var("INVARIANT_MODE").as_deref() != Ok("0");
        let verified = with_invariant_rows(invariant, || {
            capture(&mut || {
                let (logits, features) = model.verify(&block, &taps)?;
                Array::eval_many(&[&logits, &features])
            })
        })?;
        model.reset();
        model.forward_tokens(&tokens[..64])?.eval()?;
        let mut steps = Vec::new();
        for &token in &block {
            steps.push(capture(&mut || model.forward(token)?.eval())?);
        }
        ensure!(
            steps.iter().all(|s| s.len() == verified.len()),
            "stage counts differ: verify {} AR {:?}",
            verified.len(),
            steps.iter().map(Vec::len).collect::<Vec<_>>()
        );
        let mut visit = 0usize;
        let mut first_by_visit: Vec<(usize, &'static str)> = Vec::new();
        let mut printed = 0usize;
        let mut differing_labels = std::collections::BTreeMap::<&'static str, usize>::new();
        for (index, (label, batched)) in verified.iter().enumerate() {
            if *label == "in" && index > 0 {
                visit += 1;
            }
            let mut rows_differ = 0usize;
            let mut max = 0f32;
            for (j, step) in steps.iter().enumerate() {
                ensure!(
                    step[index].0 == *label,
                    "stage order differs at {index}: {label} vs {}",
                    step[index].0
                );
                let a = batched
                    .slice(0, j as i32, j as i32 + 1)?
                    .astype(Dtype::Float32)?
                    .to_f32()?;
                let b = step[index].1.astype(Dtype::Float32)?.to_f32()?;
                ensure!(
                    a.len() == b.len(),
                    "{label}: shapes differ ({} vs {})",
                    a.len(),
                    b.len()
                );
                let differ = a
                    .iter()
                    .zip(&b)
                    .filter(|(x, y)| x.to_bits() != y.to_bits())
                    .count();
                if differ > 0 {
                    rows_differ += 1;
                    max = max.max(
                        a.iter()
                            .zip(&b)
                            .map(|(x, y)| (x - y).abs())
                            .fold(0f32, f32::max),
                    );
                }
            }
            if rows_differ > 0 {
                *differing_labels.entry(label).or_default() += 1;
                if first_by_visit.last().is_none_or(|&(v, _)| v != visit) {
                    first_by_visit.push((visit, label));
                }
                if printed < 40 {
                    println!(
                        "STAGEDIFF visit {visit} {label}: {rows_differ}/{t} rows differ, max {max:.3e}"
                    );
                    printed += 1;
                }
            }
        }
        println!("STAGEFIRST {first_by_visit:?}");
        println!("STAGELABELS {differing_labels:?}");
        Ok(())
    }

    /// DIAG-2026-11-24-INK2M-VERIFY-INVARIANCE: which components give a row of an 8-row call different bits from the
    /// same row computed alone: the dense projections of layer 0's attention, the head, the MoE of layer 12 (router
    /// and experts), and the whole verification (row j of an 8-token `verify` against AR step j).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn verify_row_invariance_by_component() -> Result<()> {
        const T: i32 = 8;
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let tokens: Vec<u32> = serde_json::from_value(rows["rows"][6]["tokens"].clone())?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let vocab = model.unpadded_vocab() as u32;
        let tokens: Vec<u32> = tokens.into_iter().map(|t| t % vocab).collect();
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut random = |rows: i32, cols: i32| -> Result<Array> {
            let v: Vec<f32> = (0..rows * cols)
                .map(|_| {
                    state = state
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (((state >> 33) as u32 % 2001) as f32 / 1000.0 - 1.0) * 0.5
                })
                .collect();
            Array::from_f32(&v, &[rows, cols])?.astype(Dtype::Float16)
        };
        // (values that differ in bits, values, largest absolute difference) of a batched call against row by row
        let gap = |batched: &Array, single: &[Array]| -> Result<(usize, usize, f32)> {
            let b = batched.astype(Dtype::Float32)?.to_f32()?;
            let s = Array::concatenate(&single.iter().collect::<Vec<_>>(), 0)?
                .astype(Dtype::Float32)?
                .to_f32()?;
            ensure!(
                b.len() == s.len() && !b.is_empty(),
                "invariance shapes differ"
            );
            let differ = b
                .iter()
                .zip(&s)
                .filter(|(x, y)| x.to_bits() != y.to_bits())
                .count();
            let max = b
                .iter()
                .zip(&s)
                .map(|(x, y)| (x - y).abs())
                .fold(0f32, f32::max);
            Ok((differ, b.len(), max))
        };
        let input_of = |l: &Linear| -> Result<i32> {
            Ok(match l {
                Linear::Dense(p) => p.dense_weight().context("dense weight")?.shape()[1],
                Linear::Affine { scales, group, .. } => scales.shape()[1] * *group,
            })
        };
        let mut report = Vec::new();
        let attention = &model.layers[0].attention;
        for (label, linear) in [
            ("q", &attention.q),
            ("k", &attention.k),
            ("v", &attention.v),
            ("r", &attention.r),
            ("o", &attention.o),
            ("gate", &attention.gate),
            ("sink", &attention.sink),
        ] {
            let x = random(T, input_of(linear)?)?;
            let batched = linear.forward(&x)?;
            let single = (0..T)
                .map(|i| linear.forward(&x.slice(0, i, i + 1)?))
                .collect::<Result<Vec<_>>>()?;
            let (differ, n, max) = gap(&batched, &single)?;
            println!("INVARIANCE dense {label}: {differ}/{n} values differ, max {max:.3e}");
            report.push(serde_json::json!({"component": format!("dense_{label}"), "differ": differ, "values": n, "max": max}));
        }
        {
            let x = random(T, model.hidden)?;
            let batched = model.head_scores(&x)?;
            let single = (0..T)
                .map(|i| model.head_scores(&x.slice(0, i, i + 1)?))
                .collect::<Result<Vec<_>>>()?;
            let (differ, n, max) = gap(&batched, &single)?;
            println!("INVARIANCE head: {differ}/{n} values differ, max {max:.3e}");
            report.push(
                serde_json::json!({"component": "head", "differ": differ, "values": n, "max": max}),
            );
        }
        {
            let moe = &model.layers[12].moe;
            let x = random(T, model.hidden)?;
            let batched = moe.forward(&x)?;
            let single = (0..T)
                .map(|i| moe.forward(&x.slice(0, i, i + 1)?))
                .collect::<Result<Vec<_>>>()?;
            let (differ, n, max) = gap(&batched, &single)?;
            println!(
                "INVARIANCE moe (router + experts): {differ}/{n} values differ, max {max:.3e}"
            );
            report.push(
                serde_json::json!({"component": "moe", "differ": differ, "values": n, "max": max}),
            );
        }
        {
            let taps: Vec<usize> = vec![5, 23, 41, model.visit_count()];
            model.reset();
            model.forward_tokens(&tokens[..64])?.eval()?;
            let (logits, _) = model.verify(&tokens[64..64 + T as usize], &taps)?;
            let batched = logits.reshape(&[T, -1])?;
            batched.eval()?;
            model.reset();
            model.forward_tokens(&tokens[..64])?.eval()?;
            let mut single = Vec::new();
            for i in 0..T as usize {
                let l = model.forward(tokens[64 + i])?.reshape(&[1, -1])?;
                l.eval()?;
                single.push(l);
            }
            for i in 0..T {
                let (differ, n, max) = gap(
                    &batched.slice(0, i, i + 1)?,
                    &single[i as usize..i as usize + 1],
                )?;
                println!(
                    "INVARIANCE verify row {i} vs AR step {i}: {differ}/{n} logits differ, max {max:.3e}"
                );
                report.push(serde_json::json!({"component": format!("verify_row{i}"), "differ": differ, "values": n, "max": max}));
            }
        }
        println!("INVARIANCE_JSON {}", serde_json::to_string(&report)?);
        Ok(())
    }

    /// DFLASH2-2026-11-24-INK2M-P4-2c reference: the protocol of `expert_tensor_verify_quality` between the exact
    /// verification in 8-row blocks and the exact step-by-step decoding (one `forward` per token, teacher forcing):
    /// the gap of the exact batched path itself, to compare with the hybrid's.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn verify_batching_reference_quality() -> Result<()> {
        const PREFIX: usize = 64;
        const BLOCKS: usize = 48;
        const T: usize = 8;
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let taps: Vec<usize> = vec![5, 23, 41, model.visit_count()];
        let vocab = model.unpadded_vocab();
        let mut report = Vec::new();
        let (mut all_kl, mut all_max, mut all_dis, mut all_n) = (0f64, 0f64, 0usize, 0usize);
        let (mut all_nll_v, mut all_nll_a) = (0f64, 0f64);
        for index in 0..12usize {
            let name = rows["rows"][index]["name"]
                .as_str()
                .unwrap_or("?")
                .to_owned();
            let tokens: Vec<u32> = serde_json::from_value(rows["rows"][index]["tokens"].clone())?;
            let tokens: Vec<u32> = tokens.into_iter().map(|t| t % vocab as u32).collect();
            ensure!(tokens.len() > PREFIX + BLOCKS * T, "row {name} too short");
            let lp = |logits: &Array, rows: i32| -> Result<Array> {
                let lp = logits
                    .reshape(&[rows, -1])?
                    .slice(1, 0, vocab)?
                    .astype(Dtype::Float32)?
                    .log_probs()?;
                lp.eval()?;
                Ok(lp)
            };
            model.reset();
            model.forward_tokens(&tokens[..PREFIX])?.eval()?;
            let mut verified = Vec::with_capacity(BLOCKS);
            for b in 0..BLOCKS {
                let (logits, _) =
                    model.verify(&tokens[PREFIX + b * T..PREFIX + (b + 1) * T], &taps)?;
                model.commit(T)?;
                verified.push(lp(&logits, T as i32)?);
            }
            model.reset();
            model.forward_tokens(&tokens[..PREFIX])?.eval()?;
            let mut stepped = Vec::with_capacity(BLOCKS);
            for b in 0..BLOCKS {
                let mut block = Vec::with_capacity(T);
                for i in 0..T {
                    block.push(lp(&model.forward(tokens[PREFIX + b * T + i])?, 1)?);
                }
                let joined = Array::concatenate(&block.iter().collect::<Vec<_>>(), 0)?;
                joined.eval()?;
                stepped.push(joined);
            }
            let (mut kls, mut dis, mut nll_v, mut nll_a) = (Vec::new(), 0usize, 0f64, 0f64);
            for (b, (p, q)) in stepped.iter().zip(&verified).enumerate() {
                ensure!(p.shape() == q.shape(), "log-prob shapes differ");
                let kl = p.exp()?.mul(&p.sub(q)?)?.sum(1, false)?.to_f32()?;
                ensure!(
                    kl.len() == T && kl.iter().all(|v| v.is_finite()),
                    "KL values"
                );
                kls.extend(kl.iter().map(|&v| f64::from(v.max(0.0))));
                let (ap, aq) = (p.argmax()?.to_u32()?, q.argmax()?.to_u32()?);
                dis += ap.iter().zip(&aq).filter(|(a, b)| a != b).count();
                let next: Vec<u32> = (0..T)
                    .map(|i| i as u32 * vocab as u32 + tokens[PREFIX + b * T + i + 1])
                    .collect();
                let idx = Array::from_u32(&next, &[T as i32])?;
                nll_a -= p
                    .reshape(&[-1])?
                    .take(&idx, 0)?
                    .to_f32()?
                    .iter()
                    .map(|&v| f64::from(v))
                    .sum::<f64>();
                nll_v -= q
                    .reshape(&[-1])?
                    .take(&idx, 0)?
                    .to_f32()?
                    .iter()
                    .map(|&v| f64::from(v))
                    .sum::<f64>();
            }
            let n = kls.len();
            let mean = kls.iter().sum::<f64>() / n as f64;
            let max = kls.iter().cloned().fold(0f64, f64::max);
            println!(
                "P4REFERENCE {name}: positions {n} KL(step || verify8) mean {mean:.3e} max {max:.3e} disagreements {dis} perplexity step {:.4} verify8 {:.4}",
                (nll_a / n as f64).exp(),
                (nll_v / n as f64).exp()
            );
            report.push(serde_json::json!({"row": name, "kl_mean": mean, "kl_max": max, "disagreements": dis,
                "perplexity_step": (nll_a / n as f64).exp(), "perplexity_verify8": (nll_v / n as f64).exp()}));
            all_kl += kls.iter().sum::<f64>();
            all_max = all_max.max(max);
            all_dis += dis;
            all_n += n;
            all_nll_v += nll_v;
            all_nll_a += nll_a;
        }
        println!(
            "P4REFERENCE total: positions {all_n} KL mean {:.3e} max {all_max:.3e} disagreements {all_dis} ({:.2} %) perplexity step {:.4} verify8 {:.4}",
            all_kl / all_n as f64,
            100.0 * all_dis as f64 / all_n as f64,
            (all_nll_a / all_n as f64).exp(),
            (all_nll_v / all_n as f64).exp()
        );
        println!("P4REFERENCE_JSON {}", serde_json::to_string(&report)?);
        Ok(())
    }

    /// DFLASH2-2026-11-24-INK2M-P4-2 cost: median wall time of `verify` (evaluated) for T = 1..=8 rows after the same
    /// real 64-token prefix, exact / hybrid / hybrid / exact, 7 timed repetitions after 2 warm-ups per cell.
    #[test]
    #[ignore = "requires an Apple M5 GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn expert_tensor_verify_cost() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let tokens: Vec<u32> = serde_json::from_value(rows["rows"][6]["tokens"].clone())?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let vocab = model.unpadded_vocab() as u32;
        let tokens: Vec<u32> = tokens.into_iter().map(|t| t % vocab).collect();
        let taps: Vec<usize> = vec![5, 23, 41, model.visit_count()];
        let mut cells = Vec::new();
        for t in 1..=8usize {
            let mut arms = Vec::new();
            for hybrid in [false, true, true, false] {
                let mut times = Vec::new();
                crate::moe::with_expert_tensor(hybrid, || -> Result<()> {
                    for rep in 0..9 {
                        model.reset();
                        model.forward_tokens(&tokens[..64])?.eval()?;
                        let block = &tokens[64 + rep..64 + rep + t];
                        let start = Instant::now();
                        let (logits, _) = model.verify(block, &taps)?;
                        logits.eval()?;
                        let ms = start.elapsed().as_secs_f64() * 1e3;
                        model.commit(t)?;
                        if rep >= 2 {
                            times.push(ms);
                        }
                    }
                    Ok(())
                })?;
                let (_, median, _) = quantiles(times);
                arms.push(median);
            }
            println!(
                "P4COST rows {t}: exact {:.2} / {:.2} hybrid {:.2} / {:.2} ms",
                arms[0], arms[3], arms[1], arms[2]
            );
            cells.push(serde_json::json!({"rows": t, "exact_ms": [arms[0], arms[3]], "hybrid_ms": [arms[1], arms[2]]}));
        }
        println!("P4COST_JSON {}", serde_json::to_string(&cells)?);
        Ok(())
    }

    /// Verification cost by row count (diagnostic): median wall time of `verify` for T = 1..8 rows.
    #[test]
    #[ignore = "requires an Apple GPU and MLXL3_INKLING_MODEL"]
    fn verify_cost_by_rows() -> Result<()> {
        let model_dir = std::env::var("MLXL3_INKLING_MODEL")?;
        let mut model = Inkling2::load(Path::new(&model_dir))?;
        let taps: Vec<usize> = vec![5, 23, 41, model.visit_count()];
        let prompt: Vec<u32> = (0..26).map(|i| 1000 + 37 * i).collect();
        let mut rows = Vec::new();
        for t in 1..=8usize {
            let mut times = Vec::new();
            for rep in 0..6 {
                model.reset();
                model.forward_tokens(&prompt)?;
                let block: Vec<u32> = (0..t as u32).map(|i| 2000 + 17 * i + rep).collect();
                let start = Instant::now();
                model.verify(&block, &taps)?;
                times.push(start.elapsed().as_secs_f64() * 1e3);
                model.commit(t)?;
            }
            let (_, median, _) = quantiles(times[1..].to_vec());
            rows.push(serde_json::json!({"rows": t, "verify_ms_median": median}));
        }
        println!("{}", serde_json::to_string(&rows)?);
        Ok(())
    }

    /// Quality of the affine dense/head variants against the FP16 engine itself: KL(FP16 || variant) and top-1
    /// agreement on the tail positions of fixed token rows. One model alive at a time.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL and MLXL3_INKLING_ROWS"]
    fn dense_bits_kl() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let variants: Vec<Precision> = std::env::var("MLXL3_INKLING_VARIANTS")
            .unwrap_or_else(|_| "8-8,4-8,8-4,4-4".into())
            .split(',')
            .map(|v| {
                let (bits, only) = match v.split_once('@') {
                    Some((bits, only)) => (bits, DenseSet::parse(only)?),
                    None => (v, DenseSet::ALL),
                };
                let (d, h) = bits
                    .split_once('-')
                    .context("variant must be DENSE-HEAD[@proj+proj]")?;
                Ok(Precision {
                    dense_bits: d.parse()?,
                    head_bits: h.parse()?,
                    dense_only: only,
                    dense_group: Precision::from_env_or(Precision::FULL)?.dense_group,
                })
            })
            .collect::<Result<_>>()?;
        let rows: Vec<(String, Vec<u32>, usize)> = rows["rows"]
            .as_array()
            .context("rows")?
            .iter()
            .map(|r| {
                Ok((
                    r["name"].as_str().context("name")?.to_string(),
                    serde_json::from_value(r["tokens"].clone())?,
                    r["tail"].as_u64().context("tail")? as usize,
                ))
            })
            .collect::<Result<_>>()?;
        // log-probabilities [tail, vocab] FP32 of every row
        let run = |precision: Precision| -> Result<Vec<Vec<f32>>> {
            let mut model = Inkling2::load_with(&dir, precision)?;
            ensure!(model.precision() == precision, "precision not applied");
            let vocab = model.unpadded_vocab() as usize;
            rows.iter()
                .map(|(name, tokens, tail)| {
                    ensure!(*tail >= 1 && *tail <= tokens.len(), "{name}: bad tail");
                    model.reset();
                    let rows = tokens.len() as i32;
                    let values = model
                        .forward_all(tokens)?
                        .reshape(&[rows, vocab as i32])?
                        .slice(0, rows - *tail as i32, rows)?
                        .log_probs()?
                        .to_f32()?;
                    ensure!(
                        values.len() == tail * vocab && values.iter().all(|x| x.is_finite()),
                        "{name}: empty or non-finite log-probabilities"
                    );
                    Ok(values)
                })
                .collect()
        };
        let argmax = |row: &[f32]| {
            row.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |best, (i, &x)| {
                    if x > best.1 { (i, x) } else { best }
                })
                .0
        };
        let reference = run(Precision::FULL)?;
        let mut report = Vec::new();
        for precision in variants {
            let actual = run(precision)?;
            let (mut total_kl, mut total_same, mut total_positions) = (0f64, 0usize, 0usize);
            let mut per_row = Vec::new();
            for (((name, _, tail), reference), actual) in rows.iter().zip(&reference).zip(&actual) {
                let vocab = reference.len() / tail;
                let (mut sum, mut worst, mut same) = (0f64, 0f64, 0usize);
                for (r, a) in reference
                    .chunks_exact(vocab)
                    .zip(actual.chunks_exact(vocab))
                {
                    let kl: f64 = r
                        .iter()
                        .zip(a)
                        .map(|(&r, &a)| (r as f64).exp() * (r as f64 - a as f64))
                        .sum();
                    sum += kl;
                    worst = worst.max(kl);
                    same += usize::from(argmax(r) == argmax(a));
                }
                per_row.push(serde_json::json!({"row": name, "positions": tail, "kl_mean": sum / *tail as f64,
                    "kl_max": worst, "top1": same as f64 / *tail as f64}));
                total_kl += sum;
                total_same += same;
                total_positions += tail;
            }
            report.push(serde_json::json!({
                "dense_bits": precision.dense_bits, "head_bits": precision.head_bits,
                "dense_only": format!("{:?}", precision.dense_only),
                "positions": total_positions, "kl_mean": total_kl / total_positions as f64,
                "top1": total_same as f64 / total_positions as f64, "rows": per_row,
            }));
        }
        println!("{}", serde_json::to_string_pretty(&report)?);
        Ok(())
    }

    /// SPEED-2026-11-22-INK2M-P8: input Hessians H = sum x^T x (FP32) of the dense projections on the rows of
    /// `MLXL3_INKLING_ROWS`, written as raw little-endian f32 `h_{layer}_{slot}.f32` plus `index.json` under
    /// `MLXL3_INKLING_HESSIAN_OUT` (slots: 0 hq, 1 hk, 2 hv, 3 input of o).
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_INKLING_ROWS and MLXL3_INKLING_HESSIAN_OUT"]
    fn dense_hessians() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let out = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_HESSIAN_OUT")?);
        std::fs::create_dir_all(&out)?;
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let rows: Vec<Vec<u32>> = rows["rows"]
            .as_array()
            .context("rows")?
            .iter()
            .map(|r| Ok(serde_json::from_value(r["tokens"].clone())?))
            .collect::<Result<_>>()?;
        ensure!(!rows.is_empty(), "no calibration rows");
        let mut model = Inkling2::load_with(&dir, Precision::FULL)?;
        let layers = model.layers.len();
        DENSE_HESSIANS.with(|c| *c.borrow_mut() = Some(Default::default()));
        let mut tokens = 0usize;
        for (i, row) in rows.iter().enumerate() {
            ensure!(!row.is_empty(), "row {i} is empty");
            model.reset();
            let logits = model.forward_all(row)?;
            DENSE_HESSIANS.with(|c| -> Result<()> {
                let c = c.borrow();
                let map = c.as_ref().context("capture")?;
                let mut all: Vec<&Array> = map.values().map(|(h, _)| h).collect();
                all.push(&logits);
                Array::eval_many(&all)
            })?;
            tokens += row.len();
            eprintln!("row {i}: {} tokens", row.len());
        }
        let map = DENSE_HESSIANS
            .with(|c| c.borrow_mut().take())
            .context("capture")?;
        ensure!(
            map.len() == layers * 4,
            "{} Hessians for {layers} layers",
            map.len()
        );
        let mut index = Vec::new();
        for ((layer, slot), (h, n)) in &map {
            let dim = h.shape()[0];
            ensure!(h.shape() == [dim, dim], "Hessian shape {:?}", h.shape());
            let values = h.to_f32()?;
            ensure!(
                values.len() == (dim * dim) as usize && values.iter().all(|x| x.is_finite()),
                "layer {layer} slot {slot}: empty or non-finite Hessian"
            );
            let bytes: Vec<u8> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(out.join(format!("h_{layer}_{slot}.f32")), bytes)?;
            index.push(serde_json::json!({"layer": layer, "slot": slot, "dim": dim, "rows": n}));
        }
        std::fs::write(
            out.join("index.json"),
            serde_json::to_string_pretty(
                &serde_json::json!({"tokens": tokens, "rows": rows.len(), "hessians": index}),
            )?,
        )?;
        Ok(())
    }

    /// DIAG-2026-11-21-INK2M-LOCALREF. With `MLXL3_INKLING_EVAL_WRITE=1`: writes the logits of every
    /// `stride`-th position of each row (`MLXL3_INKLING_ROWS`) as FP32 under `MLXL3_INKLING_REF_DUMP`.
    /// Otherwise: KL(reference || model) and top-1 agreement of the loaded model against those files.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL, MLXL3_INKLING_ROWS and MLXL3_INKLING_REF_DUMP"]
    fn eval_against_reference() -> Result<()> {
        let dir = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let dump = std::path::PathBuf::from(std::env::var("MLXL3_INKLING_REF_DUMP")?);
        let write = std::env::var_os("MLXL3_INKLING_EVAL_WRITE").is_some();
        let stride: i32 = std::env::var("MLXL3_INKLING_STRIDE")
            .ok()
            .map(|v| v.parse())
            .transpose()?
            .unwrap_or(4);
        let only: Option<String> = std::env::var("MLXL3_INKLING_EVAL_ROWS").ok();
        let rows: serde_json::Value =
            serde_json::from_reader(File::open(std::env::var("MLXL3_INKLING_ROWS")?)?)?;
        let mut model = Inkling2::load_with(&dir, Precision::from_env_or(Precision::FULL)?)?;
        let vocab = model.unpadded_vocab();
        if write {
            std::fs::create_dir_all(&dump)?;
        }
        let mut report = Vec::new();
        for row in rows["rows"].as_array().context("rows")? {
            let name = row["name"].as_str().context("name")?;
            if only
                .as_ref()
                .is_some_and(|list| !list.split(',').any(|n| n == name))
            {
                continue;
            }
            let domain = row["domain"].as_str().unwrap_or(name);
            let tokens: Vec<u32> = serde_json::from_value(row["tokens"].clone())?;
            let time = tokens.len() as i32;
            ensure!(
                stride >= 1 && time >= stride,
                "{name}: row shorter than the stride"
            );
            let positions: Vec<i32> = (stride - 1..time).step_by(stride as usize).collect();
            let count = positions.len() as i32;
            model.reset();
            let logits = model
                .forward_all(&tokens)?
                .reshape(&[time, vocab])?
                .take(&Array::from_i32(&positions, &[count])?, 0)?;
            let file = dump.join(format!("{name}.f32"));
            if write {
                let values = logits.to_f32()?;
                ensure!(
                    values.len() == (count * vocab) as usize
                        && values.iter().all(|x| x.is_finite()),
                    "{name}: empty or non-finite reference logits"
                );
                std::fs::write(
                    &file,
                    values
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<u8>>(),
                )?;
                report.push(serde_json::json!({"row": name, "positions": count, "written": file}));
                continue;
            }
            let bytes =
                std::fs::read(&file).with_context(|| format!("reference {}", file.display()))?;
            ensure!(
                bytes.len() == (count * vocab) as usize * 4,
                "{name}: reference file does not match the row"
            );
            let reference = Array::from_bytes(&bytes, &[count, vocab], Dtype::Float32)?;
            drop(bytes);
            let lr = reference.log_probs()?;
            let lm = logits.log_probs()?;
            let kl = lr.exp()?.mul(&lr.sub(&lm)?)?.sum(1, false)?.to_f32()?;
            let same: Vec<u32> = reference.argmax()?.reshape(&[-1])?.to_u32()?;
            let ours: Vec<u32> = logits.argmax()?.reshape(&[-1])?.to_u32()?;
            ensure!(
                kl.len() == count as usize && kl.iter().all(|x| x.is_finite()),
                "{name}: invalid KL"
            );
            let agree = same.iter().zip(&ours).filter(|(a, b)| a == b).count();
            report.push(serde_json::json!({
                "row": name, "domain": domain, "positions": count,
                "kl_mean": kl.iter().map(|&x| x as f64).sum::<f64>() / count as f64,
                "kl_max": kl.iter().cloned().fold(0f32, f32::max),
                "top1": agree as f64 / count as f64, "kl": kl,
            }));
        }
        ensure!(!report.is_empty(), "no row evaluated");
        println!("{}", serde_json::to_string(&report)?);
        Ok(())
    }
}
