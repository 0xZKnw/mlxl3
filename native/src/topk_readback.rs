//! Q1-TOPK-READBACK: sampled rows read back only the log probabilities that
//! may enter the penalized top-k instead of the whole vocabulary. On by
//! default for autoregressive sampling (`MLXL3_TOPK_READBACK=0` disables
//! it); MTP rows use it only with `MLXL3_TOPK_READBACK=1`.
//!
//! One threadgroup per row takes the maxima of 64-value tiles, sorts them
//! (bitonic, in threadgroup memory) and sets `threshold` to the `k'`-th
//! largest tile maximum: at least `k'` values reach it. Every value at or
//! above it is compacted with its token. Every other value is below the
//! threshold, and penalties only lower values, so when the k-th penalized
//! candidate still reaches the threshold (and has no tie) the candidates
//! decide the full-row selection; `Sampler::sparse_distribution` checks
//! that and the caller falls back to the full row otherwise.

use crate::array::{self, Array, Dtype};
use crate::sampling::{Distribution, History, Sampler};
use anyhow::{Context, Result, ensure};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Candidates kept per row; more means fallback.
pub const CAPACITY: usize = 2048;
const TILE: usize = 64;
const MAX_TILES: usize = 4096;
const HEADER: usize = 4;

/// Rows served from the readback and rows that fell back, process-wide.
static CERTIFIED: AtomicUsize = AtomicUsize::new(0);
static FALLBACKS: AtomicUsize = AtomicUsize::new(0);

/// `[certified rows, fallback rows]` since process start.
pub fn counters() -> [usize; 2] {
    [
        CERTIFIED.load(Ordering::Relaxed),
        FALLBACKS.load(Ordering::Relaxed),
    ]
}

/// Autoregressive gate: on unless the flag is set to anything but `1`.
pub fn ar_enabled_by(flag: Option<&[u8]>) -> bool {
    matches!(flag, None | Some(b"1"))
}

/// MTP gate: strict opt-in, exactly `1`.
pub fn mtp_enabled_by(flag: Option<&[u8]>) -> bool {
    flag == Some(b"1")
}

/// `MLXL3_TOPK_READBACK` through `gate` (each caller caches it once).
fn flag_is(gate: fn(Option<&[u8]>) -> bool) -> bool {
    let flag = std::env::var_os("MLXL3_TOPK_READBACK");
    gate(flag.as_deref().map(std::ffi::OsStr::as_encoded_bytes))
}

/// Readback for autoregressive sampled rows (`select_token`).
pub fn ar_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| flag_is(ar_enabled_by))
}

/// Readback for MTP verification and fallback rows.
pub fn mtp_enabled() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| flag_is(mtp_enabled_by))
}

/// Tile count of a vocabulary, if the kernel supports it.
pub fn tiles(vocab: usize) -> Option<usize> {
    let tiles = vocab.div_ceil(TILE);
    (vocab > 0 && tiles <= MAX_TILES).then_some(tiles)
}

/// Tile rank of the threshold: room for `k` values after `span` penalized
/// tokens drop out, plus one to see a boundary tie; bounded by the tiles.
pub fn tile_rank(k: usize, span: usize, tiles: usize) -> usize {
    k.saturating_add(1)
        .saturating_add(span)
        .min(k.saturating_mul(2).saturating_add(64))
        .min(tiles)
        .max(1)
}

/// Candidates of one row as decoded from the kernel output.
#[derive(Clone, Debug, PartialEq)]
pub struct RowCandidates {
    pub candidates: Vec<(u32, f32)>,
    pub threshold: f32,
    /// More values than `CAPACITY` reached the threshold.
    pub overflow: bool,
    /// The row holds a NaN.
    pub nan: bool,
}

impl RowCandidates {
    /// Decodes `[count, threshold, nan, 0, (id, f32 bits)…]`.
    pub fn parse(words: &[u32], vocab: usize) -> Result<Self> {
        ensure!(words.len() == HEADER + 2 * CAPACITY, "invalid top-k row");
        let count = words[0] as usize;
        let overflow = count > CAPACITY;
        let kept = if overflow { 0 } else { count };
        let candidates = words[HEADER..HEADER + 2 * kept]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| (pair[0], f32::from_bits(pair[1])))
            .collect::<Vec<_>>();
        ensure!(
            candidates.iter().all(|c| (c.0 as usize) < vocab),
            "top-k candidate outside the vocabulary"
        );
        Ok(Self {
            candidates,
            threshold: f32::from_bits(words[1]),
            overflow,
            nan: words[2] != 0,
        })
    }

    /// The row's distribution if the candidates certify it.
    pub fn distribution(
        &self,
        sampler: &Sampler,
        history: &History,
        extra: &[u32],
    ) -> Result<Option<Distribution>> {
        if self.overflow || self.nan {
            return Ok(None);
        }
        sampler.sparse_distribution(&self.candidates, self.threshold, history, extra)
    }
}

/// Kernel output `[rows, 4 + 2·CAPACITY]` (UInt32) for `log_probs`
/// `[…, vocab]` of any floating dtype; `rank` is the threshold tile rank.
pub fn top_rows(log_probs: &Array, rank: usize) -> Result<Array> {
    let shape = log_probs.shape();
    let vocab = usize::try_from(*shape.last().context("log probs without shape")?)?;
    let rows = shape[..shape.len() - 1].iter().product::<i32>();
    let tiles = tiles(vocab).context("vocabulary too large for top-k readback")?;
    ensure!(
        rows > 0 && (1..=tiles).contains(&rank),
        "invalid top-k readback request"
    );
    let padded = tiles.next_power_of_two().max(1024);
    let dtype = match log_probs.dtype() {
        Dtype::Float16 => "f16",
        Dtype::BFloat16 => "bf16",
        Dtype::Float32 => "f32",
        other => anyhow::bail!("unsupported log-prob dtype {other:?}"),
    };
    let source = format!(
        r#"
            constexpr uint V = {vocab}u;
            constexpr uint TILES = {tiles}u;
            constexpr uint P = {padded}u;
            constexpr uint CAP = {cap}u;
            constexpr uint WORDS = {words}u;
            uint tid = thread_position_in_threadgroup.x;
            uint lane = thread_index_in_simdgroup;
            uint group = simdgroup_index_in_threadgroup;
            uint row = threadgroup_position_in_grid.x;
            uint base = row * V;
            device uint* o = out + row * WORDS;
            threadgroup float data[P];
            threadgroup atomic_uint counter;
            threadgroup atomic_uint nan_seen;
            if (tid == 0u) {{
                atomic_store_explicit(&counter, 0u, memory_order_relaxed);
                atomic_store_explicit(&nan_seen, 0u, memory_order_relaxed);
            }}
            for (uint i = TILES + tid; i < P; i += 1024u) {{
                data[i] = -INFINITY;
            }}
            bool nan_here = false;
            for (uint t = group; t < TILES; t += 32u) {{
                uint a = t * 64u + lane;
                uint b = a + 32u;
                float va = a < V ? float(lp[base + a]) : -INFINITY;
                float vb = b < V ? float(lp[base + b]) : -INFINITY;
                nan_here = nan_here || isnan(va) || isnan(vb);
                va = isnan(va) ? -INFINITY : va;
                vb = isnan(vb) ? -INFINITY : vb;
                float m = simd_max(max(va, vb));
                if (lane == 0u) {{
                    data[t] = m;
                }}
            }}
            if (nan_here) {{
                atomic_store_explicit(&nan_seen, 1u, memory_order_relaxed);
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint size = 2u; size <= P; size <<= 1u) {{
                for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {{
                    for (uint i = tid; i < P; i += 1024u) {{
                        uint j = i ^ stride;
                        if (j > i) {{
                            float x = data[i];
                            float y = data[j];
                            bool descending = (i & size) == 0u;
                            if (descending ? (x < y) : (x > y)) {{
                                data[i] = y;
                                data[j] = x;
                            }}
                        }}
                    }}
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }}
            }}
            float limit = data[{index}u];
            for (uint i = tid; i < V; i += 1024u) {{
                float v = float(lp[base + i]);
                if (v >= limit) {{
                    uint slot = atomic_fetch_add_explicit(&counter, 1u, memory_order_relaxed);
                    if (slot < CAP) {{
                        o[4u + 2u * slot] = i;
                        o[5u + 2u * slot] = as_type<uint>(v);
                    }}
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {{
                o[0] = atomic_load_explicit(&counter, memory_order_relaxed);
                o[1] = as_type<uint>(limit);
                o[2] = atomic_load_explicit(&nan_seen, memory_order_relaxed);
                o[3] = 0u;
            }}
        "#,
        cap = CAPACITY,
        words = HEADER + 2 * CAPACITY,
        index = rank - 1,
    );
    Ok(array::metal_kernel(
        &format!("mlxl3_topk_rows_{dtype}_{vocab}_{rank}_v1"),
        &["lp"],
        &["out"],
        "",
        &source,
        &[log_probs],
        &[vec![rows, i32::try_from(HEADER + 2 * CAPACITY)?]],
        &[Dtype::UInt32],
        [1024 * rows, 1, 1],
        [1024, 1, 1],
    )?
    .remove(0))
}

/// Target distributions of the rows of `logits` for one sampled step or
/// speculative verification, canonical (`Distribution::canonical`), from
/// the readback when certified and from the full row otherwise.
pub struct ReadbackRows<'a> {
    logits: &'a Array,
    rows: Vec<RowCandidates>,
    full: Option<Vec<Vec<f32>>>,
}

impl<'a> ReadbackRows<'a> {
    /// `extra` holds every token later rows may add to the penalty history
    /// (the proposals of a verification).
    pub fn new(
        logits: &'a Array,
        sampler: &Sampler,
        history: &History,
        extra: &[u32],
    ) -> Result<Self> {
        let vocab = usize::try_from(*logits.shape().last().context("logits without shape")?)?;
        let mut readback = Self {
            logits,
            rows: Vec::new(),
            full: None,
        };
        let (Some(k), Some(tiles)) = (
            sampler.certified_k(crate::sampling::SPARSE_MAX_K),
            tiles(vocab),
        ) else {
            return Ok(readback);
        };
        let rank = tile_rank(k, sampler.penalty_span(history, extra), tiles);
        let words = top_rows(&logits.log_probs()?, rank)?.to_u32()?;
        readback.rows = words
            .chunks(HEADER + 2 * CAPACITY)
            .map(|row| RowCandidates::parse(row, vocab))
            .collect::<Result<_>>()?;
        Ok(readback)
    }

    pub fn len(&self) -> usize {
        if self.rows.is_empty() {
            self.logits.shape()[..self.logits.shape().len() - 1]
                .iter()
                .product::<i32>() as usize
        } else {
            self.rows.len()
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn distribution(
        &mut self,
        row: usize,
        sampler: &Sampler,
        history: &History,
        extra: &[u32],
    ) -> Result<Distribution> {
        if let Some(candidates) = self.rows.get(row)
            && let Some(distribution) = candidates.distribution(sampler, history, extra)?
        {
            CERTIFIED.fetch_add(1, Ordering::Relaxed);
            return Ok(distribution);
        }
        FALLBACKS.fetch_add(1, Ordering::Relaxed);
        if self.full.is_none() {
            self.full = Some(crate::sampling::log_prob_rows(self.logits)?);
        }
        let values = self
            .full
            .as_ref()
            .and_then(|rows| rows.get(row))
            .context("missing sampled row")?
            .clone();
        Ok(sampler.distribution(values, history, extra)?.canonical())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_and_shapes() {
        assert!(mtp_enabled_by(Some(b"1")));
        for flag in [None, Some(&b"0"[..]), Some(b""), Some(b"true"), Some(b" 1")] {
            assert!(!mtp_enabled_by(flag), "{flag:?}");
        }
        for flag in [None, Some(&b"1"[..])] {
            assert!(ar_enabled_by(flag), "{flag:?}");
        }
        for flag in [
            Some(&b"0"[..]),
            Some(b""),
            Some(b"true"),
            Some(b" 1"),
            Some(b"off"),
        ] {
            assert!(!ar_enabled_by(flag), "{flag:?}");
        }
        assert_eq!(tiles(248_320), Some(3880));
        assert_eq!(tiles(128_000), Some(2000));
        assert_eq!(tiles(65), Some(2));
        assert_eq!(tiles(MAX_TILES * TILE), Some(MAX_TILES));
        assert_eq!(tiles(MAX_TILES * TILE + 1), None);
        assert_eq!(tiles(0), None);
        assert_eq!(tile_rank(80, 20, 3880), 101);
        assert_eq!(tile_rank(80, 5000, 3880), 224);
        assert_eq!(tile_rank(80, 0, 50), 50);
        assert_eq!(tile_rank(256, usize::MAX, usize::MAX), 576);
        assert_eq!(tile_rank(0, 0, 0), 1);
    }

    #[test]
    fn parse_reads_the_kernel_layout() -> Result<()> {
        let mut words = vec![0u32; HEADER + 2 * CAPACITY];
        words[0] = 2;
        words[1] = (-3.5f32).to_bits();
        words[4] = 7;
        words[5] = (-1.25f32).to_bits();
        words[6] = 9;
        words[7] = (-3.5f32).to_bits();
        let row = RowCandidates::parse(&words, 10)?;
        assert_eq!(row.candidates, vec![(7, -1.25), (9, -3.5)]);
        assert_eq!(row.threshold, -3.5);
        assert!(!row.overflow && !row.nan);
        words[2] = 1;
        assert!(RowCandidates::parse(&words, 10)?.nan);
        words[0] = CAPACITY as u32 + 1;
        let row = RowCandidates::parse(&words, 10)?;
        assert!(row.overflow && row.candidates.is_empty());
        assert_eq!(
            row.distribution(
                &Sampler::new(0.2, 80, 1., 1.05, 0.)?,
                &History::default(),
                &[]
            )?,
            None
        );
        words[0] = 2;
        assert!(RowCandidates::parse(&words, 9).is_err());
        assert!(RowCandidates::parse(&words[1..], 10).is_err());
        Ok(())
    }

    /// Independent CPU reference of the kernel contract for one row.
    fn reference(values: &[f32], rank: usize) -> (f32, Vec<(u32, f32)>, bool) {
        let mut maxima: Vec<f32> = values
            .chunks(TILE)
            .map(|tile| {
                tile.iter()
                    .copied()
                    .filter(|v| !v.is_nan())
                    .fold(f32::NEG_INFINITY, f32::max)
            })
            .collect();
        maxima.sort_unstable_by(|a, b| b.total_cmp(a));
        let threshold = maxima[rank - 1];
        let mut candidates: Vec<(u32, f32)> = values
            .iter()
            .enumerate()
            .filter(|(_, v)| **v >= threshold)
            .map(|(i, v)| (i as u32, *v))
            .collect();
        candidates.sort_unstable_by_key(|c| c.0);
        (threshold, candidates, values.iter().any(|v| v.is_nan()))
    }

    fn synthetic(rows: usize, vocab: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..rows * vocab)
            .map(|i| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                match (state >> 32) % 997 {
                    0 => f32::NEG_INFINITY,
                    1..=40 => -2.5 - (i % 7) as f32 * 0.25,
                    _ => -((state >> 40) as f32 / 1_048_576.0) - 0.01,
                }
            })
            .collect()
    }

    /// Q1 GPU oracle: kernel thresholds, candidate sets and values equal the
    /// CPU reference bit for bit (F16, BF16, F32; ties, -inf, NaN, partial
    /// last tile, real LFM3 rows when MLXL3_LFM3_MODEL is set), and the
    /// readback distributions equal the canonical full-row distributions.
    #[test]
    #[ignore = "physical Apple GPU; Q1-TOPK-READBACK kernel oracle"]
    fn topk_rows_match_the_cpu_reference() -> Result<()> {
        let samplers = [
            Sampler::new(0.2, 80, 1., 1.05, 0.)?,
            Sampler::new(0.1, 50, 1., 1.1, 0.)?,
            Sampler::new(0.7, 20, 0.95, 1., 0.3)?,
            Sampler::new(1.0, 256, 1., 1., 0.)?,
        ];
        let history =
            History::from_tokens(&(0..30).map(|i| i * 4099 % 128_000).collect::<Vec<u32>>());
        let mut inputs: Vec<(String, Array)> = Vec::new();
        for (vocab, seed) in [(248_320usize, 1u64), (128_000, 2), (1_000, 3), (65, 4)] {
            let rows = 4;
            let mut values = synthetic(rows, vocab, seed);
            if vocab == 1_000 {
                values[17] = f32::NAN;
            }
            let raw = Array::from_f32(&values, &[1, rows as i32, vocab as i32])?;
            for dtype in [Dtype::Float32, Dtype::Float16, Dtype::BFloat16] {
                inputs.push((
                    format!("synthetic_{vocab}_{dtype:?}"),
                    raw.astype(dtype)?.log_probs()?,
                ));
            }
        }
        if let Ok(path) = std::env::var("MLXL3_LFM3_MODEL") {
            let checkpoint = crate::checkpoint::inspect(std::path::Path::new(&path))?;
            let mut model = crate::lfm3::Lfm3::from_checkpoint(&checkpoint)?;
            let prompt: Vec<u32> = (0..40).map(|i| 1000 + i * 7919 % 60_000).collect();
            let logits = model.language.forward(&prompt[..32])?;
            inputs.push(("lfm3_prefill".into(), logits.log_probs()?));
            for &token in &prompt[32..] {
                let logits = model.language.forward(&[token])?;
                inputs.push((format!("lfm3_{token}"), logits.log_probs()?));
            }
        }
        let (mut rows_checked, mut certified, mut fallbacks, mut mutation_caught) = (0, 0, 0, 0);
        for (name, log_probs) in &inputs {
            log_probs.eval()?;
            let vocab = *log_probs.shape().last().context("shape")? as usize;
            // The kernel reads `log_probs` as given: the reference reads the
            // same values (no second normalization).
            let flat = log_probs.to_f32()?;
            ensure!(!flat.is_empty() && flat.len() % vocab == 0, "{name}: shape");
            let full: Vec<Vec<f32>> = flat.chunks(vocab).map(<[f32]>::to_vec).collect();
            let tiles = tiles(vocab).context("tiles")?;
            for rank in [
                1,
                tile_rank(80, 20, tiles),
                tile_rank(256, 30, tiles),
                tiles,
            ] {
                let words = top_rows(log_probs, rank)?.to_u32()?;
                ensure!(
                    words.len() == full.len() * (HEADER + 2 * CAPACITY),
                    "{name}: output size"
                );
                for (row, values) in full.iter().enumerate() {
                    let parsed = RowCandidates::parse(
                        &words[row * (HEADER + 2 * CAPACITY)..][..HEADER + 2 * CAPACITY],
                        vocab,
                    )?;
                    let (threshold, mut expected, nan) = reference(values, rank);
                    ensure!(
                        parsed.threshold.to_bits() == threshold.to_bits(),
                        "{name} rank {rank} row {row}: threshold {} vs {threshold}",
                        parsed.threshold
                    );
                    ensure!(parsed.nan == nan, "{name} row {row}: NaN flag");
                    ensure!(
                        parsed.overflow == (expected.len() > CAPACITY),
                        "{name} row {row}: overflow flag"
                    );
                    rows_checked += 1;
                    if parsed.overflow {
                        fallbacks += 1;
                        continue;
                    }
                    let mut got = parsed.candidates.clone();
                    got.sort_unstable_by_key(|c| c.0);
                    ensure!(
                        got.len() == expected.len()
                            && got
                                .iter()
                                .zip(&expected)
                                .all(|(a, b)| a.0 == b.0 && a.1.to_bits() == b.1.to_bits()),
                        "{name} rank {rank} row {row}: candidates differ ({} vs {})",
                        got.len(),
                        expected.len()
                    );
                    if nan {
                        continue;
                    }
                    for sampler in &samplers {
                        let reference = sampler
                            .distribution(values.clone(), &history, &[])?
                            .canonical();
                        match parsed.distribution(sampler, &history, &[])? {
                            Some(distribution) => {
                                certified += 1;
                                ensure!(
                                    distribution == reference,
                                    "{name} rank {rank} row {row}: distribution differs"
                                );
                            }
                            None => fallbacks += 1,
                        }
                    }
                    // Mutation: drop the best candidate; a certified answer must then be wrong.
                    if expected.len() > 1 {
                        let best = (0..expected.len())
                            .max_by(|&a, &b| expected[a].1.total_cmp(&expected[b].1))
                            .context("best")?;
                        expected.remove(best);
                        if let Some(distribution) =
                            samplers[0].sparse_distribution(&expected, threshold, &history, &[])?
                        {
                            let reference = samplers[0]
                                .distribution(values.clone(), &history, &[])?
                                .canonical();
                            mutation_caught += usize::from(distribution != reference);
                        }
                    }
                }
            }
        }
        eprintln!(
            "Q1_ORACLE inputs={} rows={rows_checked} certified={certified} fallbacks={fallbacks} mutation_caught={mutation_caught}",
            inputs.len()
        );
        ensure!(
            certified > 0 && mutation_caught > 0,
            "oracle did not exercise the readback"
        );
        Ok(())
    }
}
