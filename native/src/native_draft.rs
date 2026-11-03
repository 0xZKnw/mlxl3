//! Raw Flash draft backbone with explicit native BF16 boundaries.
use crate::{
    flash_draft_metadata::{self, Kind},
    native_bf16::{Matrix, round},
};
use anyhow::{Context, Result, ensure};
use std::{collections::BTreeMap, path::Path};

const H: usize = 2560;

pub(crate) struct Draft {
    pub kind: Kind,
    weights: BTreeMap<String, Matrix>,
    pub cache: Vec<(Matrix, Matrix)>,
    pub offset: usize,
    cache_capacity: usize,
}
impl Draft {
    pub fn load(path: &Path, kind: Kind) -> Result<Self> {
        let raw = flash_draft_metadata::inspect(path, kind)?;
        let weights = raw
            .tensors
            .iter()
            .map(|(key, info)| Ok((key.clone(), Matrix::load(info)?)))
            .collect::<Result<_>>()?;
        Ok(Self {
            kind,
            weights,
            cache: Vec::new(),
            offset: 0,
            cache_capacity: 0,
        })
    }
    fn quantize(&mut self) -> Result<()> {
        for (name, weight) in &mut self.weights {
            if name.ends_with(".weight") && weight.rows > 1 && !name.ends_with("base_kernel") {
                weight.affine4_weights()?;
            }
        }
        Ok(())
    }
    fn weight(&self, name: &str) -> Result<&Matrix> {
        self.weights
            .get(name)
            .with_context(|| format!("missing draft weight {name}"))
    }
    fn linear(&self, x: &Matrix, prefix: &str) -> Result<Matrix> {
        x.linear(self.weight(&format!("{prefix}.weight"))?)
    }
    fn norm(&self, x: &Matrix, name: &str, width: usize) -> Result<Matrix> {
        x.norm(self.weight(name)?, width)
    }
    fn rope(&self, mut x: Matrix, start: usize) -> Result<Matrix> {
        ensure!(
            x.cols.is_multiple_of(128)
                && start.checked_add(x.rows).is_some_and(|end| end <= 262144),
            "invalid draft rotary interval"
        );
        for row in 0..x.rows {
            for head in 0..x.cols / 128 {
                let at = row * x.cols + head * 128;
                for dim in 0..64 {
                    let angle = (start + row) as f32 / self.kind.rope().powf(dim as f32 / 64.);
                    let (sin, cos) = angle.sin_cos();
                    let a = x.values[at + dim];
                    let b = x.values[at + dim + 64];
                    x.values[at + dim] = round(a * cos - b * sin);
                    x.values[at + dim + 64] = round(a * sin + b * cos);
                }
            }
        }
        Ok(x)
    }
    pub fn append(&mut self, captured: &Matrix, start: usize) -> Result<()> {
        ensure!(
            captured.cols == 5 * H
                && start == self.offset
                && start
                    .checked_add(captured.rows)
                    .is_some_and(|end| end <= 262144),
            "invalid native draft append"
        );
        let context = self.norm(&self.linear(captured, "fc")?, "hidden_norm.weight", H)?;
        let mut cache = Vec::with_capacity(5);
        for layer in 0..5 {
            let p = format!("layers.{layer}.self_attn");
            let keys = self.rope(
                self.norm(
                    &self.linear(&context, &format!("{p}.k_proj"))?,
                    &format!("{p}.k_norm.weight"),
                    128,
                )?,
                start,
            )?;
            let values = self.linear(&context, &format!("{p}.v_proj"))?;
            let combine = |new: Matrix, old: Option<&Matrix>| -> Result<Matrix> {
                if self.cache_capacity != 0 {
                    let mut values =
                        old.map_or_else(|| vec![0.; 2048 * 1024], |old| old.values.clone());
                    let writes = crate::flash_draft_budget::ring_writes(
                        i32::try_from(start)?,
                        i32::try_from(new.rows)?,
                    )
                    .context("invalid native draft ring interval")?;
                    for (source, destination, rows) in writes {
                        let (source, destination, rows) =
                            (source as usize, destination as usize, rows as usize);
                        values[destination * 1024..(destination + rows) * 1024]
                            .copy_from_slice(&new.values[source * 1024..(source + rows) * 1024]);
                    }
                    return Matrix::new(2048, 1024, values);
                }
                let rows = old.map_or(0, |x| x.rows) + new.rows;
                let skip = if self.kind == Kind::Dflash2 {
                    rows.saturating_sub(2048)
                } else {
                    0
                };
                let mut joined = Vec::new();
                joined.try_reserve_exact((rows - skip).max(self.cache_capacity) * 1024)?;
                if let Some(old) = old {
                    joined.extend_from_slice(&old.values[skip.min(old.rows) * 1024..]);
                }
                let new_skip = skip.saturating_sub(old.map_or(0, |x| x.rows));
                joined.extend_from_slice(&new.values[new_skip * 1024..]);
                Matrix::new(rows - skip, 1024, joined)
            };
            cache.push((
                combine(keys, self.cache.get(layer).map(|(k, _)| k))?,
                combine(values, self.cache.get(layer).map(|(_, v)| v))?,
            ));
        }
        self.cache = cache;
        self.offset += captured.rows;
        Ok(())
    }
    fn convolution(
        &self,
        x: &Matrix,
        dynamic: &Matrix,
        prefix: &str,
        side: usize,
    ) -> Result<Matrix> {
        ensure!(
            x.cols == H && dynamic.rows == x.rows && dynamic.cols == 640 && side < 2,
            "invalid native draft convolution"
        );
        let base = self.weight(&format!("{prefix}.base_kernel"))?;
        ensure!(
            base.rows == 4 && base.cols == H,
            "invalid native convolution kernel"
        );
        let mut out = vec![0.; x.values.len()];
        for row in 0..x.rows {
            for col in 0..H {
                let coefficient = |tap: usize| {
                    round(
                        dynamic.values[row * 640 + side * 320 + tap * 160 + col / 16]
                            + base.values[(side * 2 + tap) * H + col],
                    )
                };
                let current = round(x.values[row * H + col] * coefficient(0));
                let previous = if row > 0 {
                    round(x.values[(row - 1) * H + col] * coefficient(1))
                } else {
                    0.
                };
                out[row * H + col] = round(current + previous);
            }
        }
        Matrix::new(x.rows, H, out)
    }
    fn sublayer_input(&self, x: &Matrix, prefix: &str) -> Result<(Matrix, Option<Matrix>)> {
        if self.kind == Kind::Dflash2 {
            let dynamic = self.linear(x, &format!("{prefix}.kernel_projection"))?;
            Ok((self.convolution(x, &dynamic, prefix, 0)?, Some(dynamic)))
        } else {
            Ok((x.clone(), None))
        }
    }
    fn sublayer_output(&self, x: Matrix, dynamic: Option<&Matrix>, prefix: &str) -> Result<Matrix> {
        if let Some(dynamic) = dynamic {
            self.convolution(&x, dynamic, prefix, 1)
        } else {
            Ok(x)
        }
    }
    fn attention(&self, x: &Matrix, layer: usize) -> Result<Matrix> {
        let p = format!("layers.{layer}.self_attn");
        let q = self.rope(
            self.norm(
                &self.linear(x, &format!("{p}.q_proj"))?,
                &format!("{p}.q_norm.weight"),
                128,
            )?,
            self.offset,
        )?;
        let k = self.rope(
            self.norm(
                &self.linear(x, &format!("{p}.k_proj"))?,
                &format!("{p}.k_norm.weight"),
                128,
            )?,
            self.offset,
        )?;
        let v = self.linear(x, &format!("{p}.v_proj"))?;
        let (old_k, old_v) = self
            .cache
            .get(layer)
            .context("missing native draft KV cache")?;
        let past = if self.cache_capacity != 0 {
            self.offset.min(2048)
        } else {
            old_k.rows
        };
        let total = past + x.rows;
        let mut out = vec![0.; x.rows * 4096];
        for row in 0..x.rows {
            for head in 0..32 {
                let group = head / 4;
                let first = if self.kind == Kind::Dflash2 {
                    (past + row + 1).saturating_sub(2048).min(past)
                } else {
                    0
                };
                let mut scores = Vec::with_capacity(total - first);
                for token in first..total {
                    let keys = if token < past {
                        &old_k.values[(if self.cache_capacity != 0 {
                            (self.offset - past + token) % 2048
                        } else {
                            token
                        }) * 1024
                            + group * 128..][..128]
                    } else {
                        &k.values[(token - past) * 1024 + group * 128..][..128]
                    };
                    let query = &q.values[row * 4096 + head * 128..][..128];
                    scores.push(
                        query.iter().zip(keys).map(|(q, k)| q * k).sum::<f32>()
                            * 128f32.sqrt().recip(),
                    );
                }
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denominator = 0.;
                for score in &mut scores {
                    *score = (*score - max).exp();
                    denominator += *score;
                }
                for dim in 0..128 {
                    let mut value = 0.;
                    for (index, score) in scores.iter().enumerate() {
                        let token = first + index;
                        let source = if token < past {
                            old_v.values[(if self.cache_capacity != 0 {
                                (self.offset - past + token) % 2048
                            } else {
                                token
                            }) * 1024
                                + group * 128
                                + dim]
                        } else {
                            v.values[(token - past) * 1024 + group * 128 + dim]
                        };
                        value += score / denominator * source;
                    }
                    out[row * 4096 + head * 128 + dim] = round(value);
                }
            }
        }
        self.linear(&Matrix::new(x.rows, 4096, out)?, &format!("{p}.o_proj"))
    }
    pub fn forward(&self, input: &Matrix, proposals: usize) -> Result<(Matrix, Vec<Matrix>)> {
        ensure!(
            input.rows == self.kind.rows(proposals)? as usize && input.cols == H && self.offset > 0,
            "invalid native draft input"
        );
        let mut hidden = input.clone();
        let mut normalized = self.norm(input, "layers.0.input_layernorm.weight", H)?;
        let mut trace = Vec::new();
        for layer in 0..5 {
            let p = format!("layers.{layer}");
            let (x, dynamic) = self.sublayer_input(&normalized, &format!("{p}.attention_conv"))?;
            let attention = self.sublayer_output(
                self.attention(&x, layer)?,
                dynamic.as_ref(),
                &format!("{p}.attention_conv"),
            )?;
            let (normalized_mlp, residual) = attention.residual_norm(
                &hidden,
                self.weight(&format!("{p}.post_attention_layernorm.weight"))?,
            )?;
            let (x, dynamic) = self.sublayer_input(&normalized_mlp, &format!("{p}.mlp_conv"))?;
            let gate = self.linear(&x, &format!("{p}.mlp.gate_proj"))?;
            let up = self.linear(&x, &format!("{p}.mlp.up_proj"))?;
            let values = gate
                .values
                .iter()
                .zip(&up.values)
                .map(|(&gate, up)| {
                    // MLX's ordinary BF16 sigmoid, multiply and product each round.
                    let sigmoid = round(1. / (1. + (-gate).exp()));
                    round(round(sigmoid * gate) * up)
                })
                .collect();
            let activated = Matrix::new(gate.rows, gate.cols, values)?;
            let mlp = self.sublayer_output(
                self.linear(&activated, &format!("{p}.mlp.down_proj"))?,
                dynamic.as_ref(),
                &format!("{p}.mlp_conv"),
            )?;
            let next_norm = if layer == 4 {
                "norm.weight".to_owned()
            } else {
                format!("layers.{}.input_layernorm.weight", layer + 1)
            };
            (normalized, hidden) = mlp.residual_norm(&residual, self.weight(&next_norm)?)?;
            trace.push(hidden.clone());
        }
        Ok((normalized, trace))
    }
}

#[cfg(test)]
#[test]
#[ignore = "raw pinned Flash draft weights and independent CPU NumPy fixture"]
fn native_backbone_matches_independent_numpy_layers_and_kv() -> Result<()> {
    let root = std::path::PathBuf::from(std::env::var("MLXL3_FLASH_DRAFT_ORACLE")?);
    let report: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("oracle.json"))?)?;
    let kind = Kind::parse(&std::env::var("MLXL3_FLASH_DRAFT_KIND")?)?;
    ensure!(
        report["status"] == "complete_cpu_oracle"
            && report["residual_norm"] == "sglang-3c560089-fp32-variance-bf16-residual"
    );
    let read = |name: &str| -> Result<Matrix> {
        let entry = &report["arrays"][name];
        let rows = entry["shape"][0].as_u64().context("oracle rows")? as usize;
        let shape = entry["shape"].as_array().context("oracle shape")?;
        let cols = shape
            .iter()
            .skip(1)
            .try_fold(1usize, |n, v| n.checked_mul(v.as_u64()? as usize))
            .context("oracle extent")?;
        let bytes = std::fs::read(root.join(entry["file"].as_str().context("oracle file")?))?;
        ensure!(rows.checked_mul(cols).and_then(|n| n.checked_mul(4)) == Some(bytes.len()));
        Matrix::new(
            rows,
            cols,
            bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect(),
        )
    };
    let mut draft = Draft::load(Path::new(&std::env::var("MLXL3_FLASH_DRAFT_MODEL")?), kind)?;
    draft.append(&read("captured")?, 0)?;
    let (output, layers) = draft.forward(
        &read("input")?,
        report["proposals"].as_u64().context("oracle proposals")? as usize,
    )?;
    let check = |name: &str, actual: &Matrix| -> Result<()> {
        let expected = read(name)?;
        ensure!(actual.values.len() == expected.values.len());
        let max = actual
            .values
            .iter()
            .zip(&expected.values)
            .map(|(a, b)| (a - b).abs())
            .fold(0., f32::max);
        let energy = expected
            .values
            .iter()
            .map(|&v| f64::from(v).powi(2))
            .sum::<f64>()
            .max(1e-30);
        let relative = (actual
            .values
            .iter()
            .zip(&expected.values)
            .map(|(&a, &b)| (f64::from(a) - f64::from(b)).powi(2))
            .sum::<f64>()
            / energy)
            .sqrt();
        eprintln!(
            "native draft {name} {} values, max_abs={max}, relative_rmse={relative}",
            actual.values.len()
        );
        ensure!(
            relative <= 0.02,
            "native draft {name} relative RMSE {relative}"
        );
        Ok(())
    };
    check("hidden", &output)?;
    for (i, layer) in layers.iter().enumerate() {
        check(&format!("layer-{i}"), layer)?;
    }
    // The fixture stores [1,heads,tokens,dim]. The native cache is token-major.
    for (i, (k, v)) in draft.cache.iter().enumerate() {
        for (name, actual) in [("keys", k), ("values", v)] {
            let expected = read(&format!("{name}-{i}"))?;
            ensure!(expected.values.len() == actual.values.len());
            let mut max = 0f32;
            let mut error = 0f64;
            let mut energy = 0f64;
            for token in 0..actual.rows {
                for head in 0..8 {
                    for dim in 0..128 {
                        let a = actual.values[token * 1024 + head * 128 + dim];
                        let b = expected.values[head * actual.rows * 128 + token * 128 + dim];
                        max = max.max((a - b).abs());
                        error += (f64::from(a) - f64::from(b)).powi(2);
                        energy += f64::from(b).powi(2);
                    }
                }
            }
            let relative = (error / energy.max(1e-30)).sqrt();
            eprintln!(
                "native draft {name}-{i} {} values, max_abs={max}, relative_rmse={relative}",
                actual.values.len()
            );
            ensure!(
                relative <= 0.01,
                "cache {name}-{i} relative RMSE {relative}"
            );
        }
    }
    Ok(())
}

impl Draft {
    pub fn proposals(
        &self,
        target: &crate::gpu::FlashNext,
        anchor: u32,
        count: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let rows = self.kind.rows(count)? as usize;
        let mut values = Vec::with_capacity(rows * H);
        for row in 0..rows {
            values.extend(
                target.draft_embedding(if self.kind == Kind::Dflash2 && row == 0 {
                    anchor
                } else {
                    self.kind.mask()
                })?,
            );
        }
        let (hidden, _) = self.forward(&Matrix::new(rows, H, values)?, count)?;
        let first = usize::from(self.kind == Kind::Dflash2);
        let selected = Matrix::new(count, H, hidden.values[first * H..].to_vec())?;
        self.select_logits(&target.draft_logits(&selected)?, &selected, anchor)
    }
    fn select_logits(
        &self,
        logits: &[Vec<f32>],
        hidden: &Matrix,
        anchor: u32,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        ensure!(
            logits.len() == hidden.rows
                && (1..=7).contains(&hidden.rows)
                && hidden.cols == H
                && anchor < 248320
                && logits
                    .iter()
                    .all(|row| row.len() == 248320 && row.iter().all(|v| v.is_finite())),
            "invalid native selector input"
        );
        let mut previous = anchor;
        let mut tokens = Vec::with_capacity(hidden.rows);
        let mut confidence = Vec::with_capacity(hidden.rows);
        if self.kind == Kind::Dflash2 {
            let projected = self.linear(hidden, "candidate_selector.hidden_projection")?;
            let predecessors = self.weight("candidate_selector.predecessor_codebook")?;
            let successors = self.weight("candidate_selector.successor_codebook")?;
            for (row, logits) in logits.iter().enumerate() {
                let mut ids: Vec<_> = (0..248320usize).collect();
                let order =
                    |&a: &usize, &b: &usize| logits[b].total_cmp(&logits[a]).then(a.cmp(&b));
                ids.select_nth_unstable_by(16, order);
                ids.truncate(16);
                ids.sort_by(order);
                let latent: Vec<_> = predecessors
                    .row(previous as usize)?
                    .values
                    .iter()
                    .zip(&projected.values[row * 256..][..256])
                    .map(|(a, b)| round(a * b))
                    .collect();
                let candidates = Matrix::new(
                    16,
                    256,
                    ids.iter()
                        .flat_map(|&id| successors.values[id * 256..(id + 1) * 256].iter().copied())
                        .collect(),
                )?;
                let edge = Matrix::new(1, 256, latent)?.linear(&candidates)?;
                let scores: Vec<_> = edge
                    .values
                    .iter()
                    .zip(&ids)
                    .map(|(edge, &id)| edge + logits[id])
                    .collect();
                let best = crate::speculative::greedy_log_prob_id(&scores)
                    .context("invalid native candidate scores")?
                    as usize;
                let max = scores[best];
                let denominator = scores.iter().map(|value| (*value - max).exp()).sum::<f32>();
                previous = ids[best] as u32;
                tokens.push(previous);
                confidence.push(denominator.recip());
            }
        } else {
            let w1 = self.weight("markov_head.markov_w1.weight")?;
            let w2 = self.weight("markov_head.markov_w2.weight")?;
            let gain = self.weight("confidence_head.proj.weight")?;
            let bias = self.weight("confidence_head.proj.bias")?;
            for (row, logits) in logits.iter().enumerate() {
                let latent = w1.row(previous as usize)?;
                let correction = latent.linear(w2)?;
                let scores: Vec<_> = logits
                    .iter()
                    .zip(&correction.values)
                    .map(|(a, b)| round(a + b))
                    .collect();
                previous = crate::speculative::greedy_log_prob_id(&scores)
                    .context("invalid native Markov scores")?;
                let value = hidden.values[row * H..(row + 1) * H]
                    .iter()
                    .chain(&latent.values)
                    .zip(&gain.values)
                    .map(|(x, w)| x * w)
                    .sum::<f32>()
                    + bias.values[0];
                tokens.push(previous);
                confidence.push(1. / (1. + (-value).exp()));
            }
        }
        ensure!(
            confidence
                .iter()
                .all(|p| p.is_finite() && (0. ..=1.).contains(p))
        );
        Ok((tokens, confidence))
    }
}

struct State {
    draft: Draft,
    options: crate::flash_draft_options::Options,
    pending: std::collections::VecDeque<u32>,
    pending_anchor: Option<u32>,
    pending_row: Option<crate::gpu::MatrixView>,
    pending_offset: u32,
    stats: crate::runtime_stats::FlashDraftProfile,
    proposed: usize,
    accepted: usize,
    blocks: usize,
    verify_budget: crate::flash_draft_budget::VerifyBudget,
    pre_budget: crate::flash_draft_budget::PreDraftBudget,
}

/// Owns the native draft and commits only tokens verified by the target model.
pub struct Session {
    checkpoint: std::path::PathBuf,
    state: std::cell::RefCell<State>,
}
impl Session {
    pub fn load(path: &Path, mode: usize, target: &crate::gpu::FlashNext) -> Result<Self> {
        let options = crate::flash_draft_options::Options::read(mode)?;
        let mut draft = Draft::load(path, Kind::Dflash2)?;
        if options.q4 {
            draft.quantize()?;
        }
        // The same two-write interval contract drives the CPU ring and the reference GPU ring.
        if options.ring {
            draft.cache_capacity = 2048;
        }
        Ok(Self {
            checkpoint: target.checkpoint_path().to_path_buf(),
            state: std::cell::RefCell::new(State {
                draft,
                options,
                pending: Default::default(),
                pending_anchor: None,
                pending_row: None,
                pending_offset: 0,
                stats: Default::default(),
                proposed: 0,
                accepted: 0,
                blocks: 0,
                verify_budget: Default::default(),
                pre_budget: Default::default(),
            }),
        })
    }
    pub fn reset(&self) {
        let mut state = self.state.borrow_mut();
        state.draft.cache.clear();
        state.draft.offset = 0;
        state.pending.clear();
        state.pending_anchor = None;
        state.pending_row = None;
    }
    pub fn stats(&self) -> (usize, usize, usize) {
        let s = self.state.borrow();
        (s.proposed, s.accepted, s.blocks)
    }
    pub fn profile(&self) -> crate::runtime_stats::FlashDraftProfile {
        self.state.borrow().stats
    }
    pub fn width(&self) -> usize {
        self.state.borrow().options.width
    }
    pub fn prefill(
        &self,
        target: &mut crate::gpu::FlashNext,
        tokens: &[u32],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<Vec<u16>> {
        ensure!(
            target.checkpoint_path() == self.checkpoint
                && !tokens.is_empty()
                && tokens.len() < target.context_limit() as usize
                && tokens.iter().all(|&v| v < 248320),
            "invalid native draft prefill"
        );
        target.reset();
        self.reset();
        let mut state = self.state.borrow_mut();
        state.stats = Default::default();
        state.proposed = 0;
        state.accepted = 0;
        state.blocks = 0;
        let operation = (|| {
            let mut last = None;
            for &token in tokens {
                ensure!(
                    !cancelled.load(std::sync::atomic::Ordering::Relaxed),
                    "generation cancelled"
                );
                let (logits, captured) =
                    target.forward_captured(token, &state.draft.kind.captures())?;
                let offset = state.draft.offset;
                state.draft.append(&captured, offset)?;
                last = Some(logits);
            }
            last.context("empty native draft prefill")
        })();
        if operation.is_err() {
            target.reset();
            state.draft.cache.clear();
            state.draft.offset = 0;
        } else {
            state.pending_row = target.last_hidden().cloned();
            state.pending_offset = target.offset();
        }
        operation
    }
    pub fn advance(
        &self,
        target: &mut crate::gpu::FlashNext,
        anchor: u32,
        context: usize,
        remaining: usize,
    ) -> Result<u32> {
        ensure!(
            anchor < 248320
                && remaining > 0
                && context <= target.context_limit() as usize
                && target.checkpoint_path() == self.checkpoint,
            "invalid native draft advance"
        );
        let mut state = self.state.borrow_mut();
        ensure!(
            state.pending_offset == target.offset()
                && state
                    .pending_row
                    .as_ref()
                    .zip(target.last_hidden())
                    .is_some_and(|(a, b)| a.shares_storage(b)),
            "native draft belongs to a different target prefix"
        );
        if !state.pending.is_empty() {
            ensure!(
                state.pending_anchor == Some(anchor)
                    && state.pending_offset == target.offset()
                    && state
                        .pending_row
                        .as_ref()
                        .zip(target.last_hidden())
                        .is_some_and(|(a, b)| a.shares_storage(b)),
                "pending draft token belongs to a different prefix"
            );
            let token = state
                .pending
                .pop_front()
                .context("empty native pending draft")?;
            state.pending_anchor = Some(token);
            return Ok(token);
        }
        ensure!(
            target.offset() as usize == state.draft.offset && (target.offset() as usize) < context,
            "native target and draft prefix differ"
        );
        let available = context - target.offset() as usize;
        let mut width = state
            .options
            .width
            .min(remaining.saturating_sub(1))
            .min(available.saturating_sub(1));
        if state.options.precap {
            width = state.pre_budget.choose(width);
        }
        let started = std::time::Instant::now();
        let (mut proposals, confidence) = if width > 0 {
            state.draft.proposals(target, anchor, width)?
        } else {
            (Vec::new(), Vec::new())
        };
        if state.options.adaptive && !confidence.is_empty() {
            let selected = state
                .verify_budget
                .choose(&confidence)
                .context("invalid native draft budget")?;
            proposals.truncate(selected);
        }
        let selected = proposals.len();
        let draft_seconds = started.elapsed().as_secs_f64();
        let before = target.snapshot();
        let verify_start = std::time::Instant::now();
        let operation = (|| {
            let mut captures = Vec::new();
            let mut emitted = Vec::new();
            let mut current = anchor;
            for proposed in proposals
                .iter()
                .copied()
                .map(Some)
                .chain(std::iter::once(None))
            {
                let (logits, captured) =
                    target.forward_captured(current, &state.draft.kind.captures())?;
                captures.extend(captured.values);
                let next = crate::speculative::greedy_log_prob_id(&target.log_probs(&logits)?)
                    .context("invalid native verification logits")?;
                if proposed == Some(next) {
                    emitted.push(next);
                    current = next;
                } else {
                    emitted.push(next);
                    break;
                }
            }
            let rows = captures.len() / 12800;
            let offset = state.draft.offset;
            state
                .draft
                .append(&Matrix::new(rows, 12800, captures)?, offset)?;
            Ok::<_, anyhow::Error>((emitted, rows))
        })();
        let (emitted, rows, commit_seconds) = match operation {
            Ok(v) => v,
            Err(error) => {
                target.restore(&before)?;
                return Err(error);
            }
        };
        let accepted = rows - 1;
        let seconds = (verify_start.elapsed().as_secs_f64()-commit_seconds).max(0.);
        state.proposed += selected;
        state.accepted += accepted;
        state.blocks += 1;
        state.stats.draft_seconds += draft_seconds;
        state.stats.verification_seconds += seconds;
        state.stats.commit_cache_seconds += commit_seconds;
        state.stats.drafted_rows += width;
        state.stats.verified_rows += rows;
        state.stats.drafted_widths[width] += 1;
        state.stats.verified_widths[selected] += 1;
        if state.options.adaptive && selected > 0 {
            let _ = state.verify_budget.observe(
                &confidence[..selected],
                accepted,
                seconds.max(f64::MIN_POSITIVE),
            );
        }
        if state.options.precap && width > 0 {
            let _ = state.pre_budget.observe(
                width,
                emitted.len(),
                (((seconds + draft_seconds) * 1e9) as u64).max(1),
            );
        }
        state.pending = emitted.into();
        state.pending_offset = target.offset();
        state.pending_row = target.last_hidden().cloned();
        let token = state
            .pending
            .pop_front()
            .context("native draft emitted no target token")?;
        state.pending_anchor = Some(token);
        Ok(token)
    }
}

#[cfg(test)]
#[test]
#[ignore = "physical Flash target, raw DFlash2 weights and native transaction/recovery"]
fn speculative_session_preserves_ordinary_tokens_states_and_failure_recovery() -> Result<()> {
    use crate::gpu::{FlashNext, Metal};
    use std::sync::atomic::{AtomicBool, Ordering};
    let checkpoint = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    let mut target = FlashNext::load(&gpu, &checkpoint, 64)?;
    let draft = Session::load(
        Path::new(&std::env::var("MLXL3_FLASH_DRAFT_MODEL")?),
        1,
        &target,
    )?;
    let prompt = [248044, 78768, 11];
    let cancelled = AtomicBool::new(false);
    let greedy = |target: &FlashNext, logits: &[u16]| -> Result<u32> {
        crate::speculative::greedy_log_prob_id(&target.log_probs(logits)?)
            .context("invalid test distribution")
    };
    for budget in [1, 2, 3, 17] {
        target.reset();
        let logits = target.prefill(&prompt)?;
        let mut next = greedy(&target, &logits)?;
        let mut expected = Vec::new();
        for index in 0..budget {
            expected.push(next);
            if index + 1 < budget {
                let logits = target.forward(next)?;
                next = greedy(&target, &logits)?;
            }
        }
        let hidden = target
            .last_hidden()
            .context("missing ordinary hidden")?
            .export_bytes(&gpu)?;
        let continuation = target.forward(7)?;
        let logits = draft.prefill(&mut target, &prompt, &cancelled)?;
        let mut next = greedy(&target, &logits)?;
        let mut actual = Vec::new();
        for index in 0..budget {
            actual.push(next);
            if index + 1 < budget {
                if !draft.state.borrow().pending.is_empty() {
                    ensure!(
                        draft
                            .advance(&mut target, (next + 1) % 248320, 64, budget - index - 1)
                            .is_err()
                    );
                }
                next = draft.advance(&mut target, next, 64, budget - index - 1)?;
            }
        }
        ensure!(
            actual == expected,
            "native DFlash changes greedy tokens at budget {budget}"
        );
        ensure!(target.offset() as usize == prompt.len() + budget - 1);
        ensure!(
            target
                .last_hidden()
                .context("missing speculative hidden")?
                .export_bytes(&gpu)?
                == hidden
        );
        ensure!(
            target.forward(7)? == continuation,
            "native DFlash changes continuation logits"
        );
        if budget==17 { let (proposed,accepted,_)=draft.stats();ensure!(accepted>0 && proposed>accepted,"fixture must cover acceptance and rejection"); }
        eprintln!(
            "native DFlash budget={budget}, counters={:?}, tokens={actual:?}",
            draft.stats()
        );
    }
    cancelled.store(true, Ordering::Relaxed);
    ensure!(draft.prefill(&mut target, &prompt, &cancelled).is_err());
    ensure!(target.offset() == 0 && draft.state.borrow().draft.offset == 0);
    cancelled.store(false, Ordering::Relaxed);
    let logits = draft.prefill(&mut target, &prompt, &cancelled)?;
    let anchor = greedy(&target, &logits)?;
    ensure!(draft.advance(&mut target, anchor, 64, 0).is_err());
    let before = target.snapshot();
    let expected_next = target.forward(7)?;
    target.restore(&before)?;
    let caches = draft.state.borrow().draft.cache.clone();
    let weight = draft
        .state
        .borrow_mut()
        .draft
        .weights
        .remove("fc.weight")
        .context("missing FC test weight")?;
    ensure!(
        draft.advance(&mut target, anchor, 64, 3).is_err(),
        "failed append was accepted"
    );
    ensure!(
        draft
            .state
            .borrow()
            .draft
            .cache
            .iter()
            .zip(&caches)
            .all(|((a, b), (c, d))| a.values == c.values && b.values == d.values)
    );
    ensure!(
        target.forward(7)? == expected_next,
        "failed append did not restore target"
    );
    target.restore(&before)?;
    draft
        .state
        .borrow_mut()
        .draft
        .weights
        .insert("fc.weight".into(), weight);
    let resumed = draft.advance(&mut target, anchor, 64, 3)?;
    ensure!(resumed < 248320);
    target.reset();
    ensure!(draft.advance(&mut target, resumed, 64, 1).is_err());
    Ok(())
}
