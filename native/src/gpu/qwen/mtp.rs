//! Greedy MTP drafting with an independently verified target prefix.
use super::{
    AttentionInput, Checkpoint, CommandBufferRef, Context, FloatType, HIDDEN, Kv, MatrixView,
    Metal, Moe, Qwen35, Rc, Result, Storage, VOCAB, encode_rms, ensure, flatten_attention, norm,
    opt_in, words,
};
use crate::speculative::{greedy_accept, greedy_log_prob_id, mtp_width};
use metal::{Buffer, CompileOptions, ComputePipelineState, MTLSize};
use std::path::Path;

const ATTN: &str = "layers.0.self_attn";
const MLP: &str = "layers.0.mlp";

struct Affine4 {
    weights: Buffer,
    scales: Buffer,
    biases: Buffer,
    input: u32,
    output: u32,
    experts: u32,
}

impl Affine4 {
    fn load(
        gpu: &Metal,
        checkpoint: &Checkpoint,
        name: &str,
        input: u32,
        output: u32,
        experts: u32,
    ) -> Result<Self> {
        // inspect() already checks the affine tensor shapes and source dtypes.
        let words = u64::from(input / 8) * u64::from(output) * u64::from(experts);
        let groups = u64::from(input / 64) * u64::from(output) * u64::from(experts);
        let packed = &checkpoint.tensors[&format!("{name}.weight")];
        let param = |suffix: &str| {
            let name = format!("{name}.{suffix}");
            let shape = &checkpoint.tensors[&name].shape;
            crate::gpu::checkpoint_float_weight(gpu, checkpoint, &name, shape, FloatType::F16)
        };
        let this = Self {
            weights: crate::gpu::read_weight(&gpu.device, packed, words * 4)?,
            scales: param("scales")?,
            biases: param("biases")?,
            input,
            output,
            experts,
        };
        ensure!(
            this.scales.length() == groups * 2 && this.biases.length() == groups * 2,
            "invalid MTP quantization parameters"
        );
        Ok(this)
    }

    fn encode(
        &self,
        gpu: &Metal,
        cmd: &CommandBufferRef,
        pipeline: &ComputePipelineState,
        x: &MatrixView,
        expert: u32,
    ) -> Result<MatrixView> {
        let out = MatrixView::output(
            gpu,
            FloatType::F16,
            1,
            self.output,
            Storage::StorageModeShared,
        )?;
        self.encode_at(cmd, pipeline, x, expert, &out, 0)?;
        Ok(out)
    }

    fn encode_at(
        &self,
        cmd: &CommandBufferRef,
        pipeline: &ComputePipelineState,
        x: &MatrixView,
        expert: u32,
        out: &MatrixView,
        row: u32,
    ) -> Result<()> {
        ensure!(
            expert < self.experts
                && x.dtype == FloatType::F16
                && x.layout.shape == [1, self.input]
                && x.layout.offset == 0
                && x.layout.strides[1] == 1
                && out.dtype == FloatType::F16
                && out.layout.shape[1] == self.output
                && out.layout.offset == 0
                && out.layout.strides == [i64::from(self.output), 1]
                && row < out.layout.shape[0],
            "invalid MTP projection input or output"
        );
        let elements = u64::from(self.input) * u64::from(self.output);
        let [packed_at, groups_at] = crate::qwen35_config::native_affine4_offsets(
            self.input,
            self.output,
            expert,
            self.experts,
        )
        .context("MTP projection extent exceeds the shader range")?;
        let out_at = u64::from(row) * u64::from(self.output) * 2;
        ensure!(
            packed_at + elements / 2 <= self.weights.length()
                && groups_at + elements / 32 <= self.scales.length()
                && groups_at + elements / 32 <= self.biases.length()
                && u64::from(self.input) * 2 <= x.buffer.length()
                && out_at + u64::from(self.output) * 2 <= out.buffer.length(),
            "MTP projection exceeds buffer storage"
        );
        Metal::encode_bound(
            cmd,
            pipeline,
            [
                (&x.buffer, 0),
                (&self.weights, packed_at),
                (&self.scales, groups_at),
                (&self.biases, groups_at),
                (&out.buffer, out_at),
            ]
            .into_iter(),
            [self.input, self.output],
            MTLSize::new(u64::from(self.output.div_ceil(4)), 1, 1),
            128,
        );
        Ok(())
    }
}

fn pipeline(gpu: &Metal) -> Result<ComputePipelineState> {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = gpu
        .device
        .new_library_with_source(
            include_str!("../../../shaders/qwen_mtp_affine.metal"),
            &options,
        )
        .map_err(anyhow::Error::msg)?;
    let function = library
        .get_function("qwen_mtp_affine4", None)
        .map_err(anyhow::Error::msg)?;
    let pipeline = gpu
        .device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(anyhow::Error::msg)?;
    ensure!(
        pipeline.thread_execution_width() == 32
            && pipeline.max_total_threads_per_threadgroup() >= 128,
        "unsupported MTP SIMD geometry"
    );
    Ok(pipeline)
}

pub struct QwenMtpHead {
    owner: Rc<()>,
    affine: ComputePipelineState,
    fc: Affine4,
    qkv: [Affine4; 3],
    attn_out: Affine4,
    router: Affine4,
    experts: [Affine4; 3],
    shared: [Affine4; 3],
    multiplier: Affine4,
    embedding_norm: MatrixView,
    hidden_norm: MatrixView,
    input_norm: MatrixView,
    post_norm: MatrixView,
    output_norm: MatrixView,
    q_norm: MatrixView,
    k_norm: MatrixView,
}

impl Qwen35<'_> {
    pub fn mtp_available(&self) -> Result<bool> {
        Ok(self.prefill_chunk_size() > 1 && opt_in("MLXL3_NATIVE_QWEN_MTP")?)
    }
}

impl QwenMtpHead {
    pub fn load(path: &Path, target: &Qwen35<'_>) -> Result<Self> {
        ensure!(
            target.mtp_available()?,
            "native Qwen MTP requires both Qwen opt-ins"
        );
        let (checkpoint, layout) = crate::mtp::inspect(path)?;
        ensure!(
            layout.managed_head(40) == Some(crate::mtp::ManagedHead::Qwen36Moe),
            "MTP head does not match the native Qwen3.6-35B-A3B target"
        );
        let gpu = target.gpu;
        let affine = |name: &str, input, output, experts| {
            Affine4::load(gpu, &checkpoint, name, input, output, experts)
        };
        let expert = |name: &str, count| -> Result<[Affine4; 3]> {
            let prefix = format!("{MLP}.{name}");
            Ok([
                affine(&format!("{prefix}.gate_proj"), HIDDEN, 512, count)?,
                affine(&format!("{prefix}.up_proj"), HIDDEN, 512, count)?,
                affine(&format!("{prefix}.down_proj"), 512, HIDDEN, count)?,
            ])
        };
        let gain = |name| {
            Ok::<_, anyhow::Error>(MatrixView {
                buffer: crate::gpu::checkpoint_float_weight(
                    gpu,
                    &checkpoint,
                    name,
                    &[HIDDEN as usize],
                    FloatType::F16,
                )?,
                dtype: FloatType::F16,
                layout: crate::tensor_layout::MatrixLayout {
                    shape: [1, HIDDEN],
                    strides: [i64::from(HIDDEN), 1],
                    offset: 0,
                },
            })
        };
        Ok(Self {
            owner: target.owner.clone(),
            affine: pipeline(gpu)?,
            fc: affine("fc", HIDDEN * 2, HIDDEN, 1)?,
            qkv: [
                affine(&format!("{ATTN}.q_proj"), HIDDEN, 8192, 1)?,
                affine(&format!("{ATTN}.k_proj"), HIDDEN, 512, 1)?,
                affine(&format!("{ATTN}.v_proj"), HIDDEN, 512, 1)?,
            ],
            attn_out: affine(&format!("{ATTN}.o_proj"), 4096, HIDDEN, 1)?,
            router: affine(&format!("{MLP}.gate"), HIDDEN, 256, 1)?,
            experts: expert("switch_mlp", 256)?,
            shared: expert("shared_expert", 1)?,
            multiplier: affine(&format!("{MLP}.shared_expert_gate"), HIDDEN, 1, 1)?,
            embedding_norm: gain("pre_fc_norm_embedding.weight")?,
            hidden_norm: gain("pre_fc_norm_hidden.weight")?,
            input_norm: gain("layers.0.input_layernorm.weight")?,
            post_norm: gain("layers.0.post_attention_layernorm.weight")?,
            output_norm: gain("norm.weight")?,
            q_norm: norm(gpu, &checkpoint, &format!("{ATTN}.q_norm.weight"), 256)?,
            k_norm: norm(gpu, &checkpoint, &format!("{ATTN}.k_norm.weight"), 256)?,
        })
    }

    fn encode_input(
        &self,
        target: &Qwen35<'_>,
        hidden: &MatrixView,
        token: u32,
        offset: u32,
    ) -> Result<(MatrixView, AttentionInput)> {
        let gpu = target.gpu;
        ensure!(
            token < VOCAB && offset < target.context_limit(),
            "invalid MTP input position or token"
        );
        let (_, embedding) = target.embedding(token)?;
        gpu.command(|cmd| {
            let embedding = encode_rms(gpu, cmd, &embedding, &self.embedding_norm)?;
            let hidden = encode_rms(gpu, cmd, hidden, &self.hidden_norm)?;
            let pair = MatrixView::output(
                gpu,
                FloatType::F16,
                1,
                HIDDEN * 2,
                Storage::StorageModeShared,
            )?;
            let row_bytes = u64::from(HIDDEN) * 2;
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer(&embedding.buffer, 0, &pair.buffer, 0, row_bytes);
            blit.copy_from_buffer(&hidden.buffer, 0, &pair.buffer, row_bytes, row_bytes);
            blit.end_encoding();
            let mixed = self.fc.encode(gpu, cmd, &self.affine, &pair, 0)?;
            let normalized = encode_rms(gpu, cmd, &mixed, &self.input_norm)?;
            let qkv = self
                .qkv
                .iter()
                .map(|projection| projection.encode(gpu, cmd, &self.affine, &normalized, 0))
                .collect::<Result<Vec<_>>>()?;
            let mut input = AttentionInput::from_projections(qkv)?;
            input.query = encode_rms(gpu, cmd, &input.query, &self.q_norm)?;
            input.key = encode_rms(gpu, cmd, &input.key, &self.k_norm)?;
            input.encode_rope(gpu, cmd, offset)?;
            Ok((mixed, input))
        })
    }

    fn append(
        &self,
        target: &Qwen35<'_>,
        hidden: &MatrixView,
        token: u32,
        offset: u32,
        prior: Option<&Kv>,
    ) -> Result<Rc<Kv>> {
        self.encode_input(target, hidden, token, offset)?
            .1
            .append(prior, offset)
    }

    fn predict(
        &self,
        target: &Qwen35<'_>,
        hidden: &MatrixView,
        token: u32,
        offset: u32,
        prior: Option<&Kv>,
    ) -> Result<(u32, MatrixView, Rc<Kv>)> {
        let gpu = target.gpu;
        let (mixed, input) = self.encode_input(target, hidden, token, offset)?;
        let kv = input.append(prior, offset)?;
        let (residual, normalized, routes) = gpu.command(|cmd| {
            let attention = input.encode_attend(gpu, cmd, &kv)?;
            let gated =
                flatten_attention(gpu.encode_model_binary(cmd, &attention, &input.gate, 2)?);
            let output = self.attn_out.encode(gpu, cmd, &self.affine, &gated, 0)?;
            let residual = gpu.encode_model_binary(cmd, &mixed, &output, 0)?;
            let normalized = encode_rms(gpu, cmd, &residual, &self.post_norm)?;
            let logits = self.router.encode(gpu, cmd, &self.affine, &normalized, 0)?;
            let probabilities = gpu.encode_model_softmax(cmd, &logits)?;
            Ok((
                residual,
                normalized,
                gpu.encode_route(cmd, &probabilities, 8, true)?,
            ))
        })?;
        let ids = routes.indices()?;
        let (hidden, logits) = gpu.command(|cmd| {
            let stack =
                MatrixView::output(gpu, FloatType::F16, 8, HIDDEN, Storage::StorageModeShared)?;
            let [gate, up, down] = &self.experts;
            for (i, id) in ids.into_iter().enumerate() {
                let gate = gate.encode(gpu, cmd, &self.affine, &normalized, id)?;
                let up = up.encode(gpu, cmd, &self.affine, &normalized, id)?;
                let activated = Moe::encode_activation(gpu, cmd, &[gate, up])?;
                down.encode_at(cmd, &self.affine, &activated, id, &stack, u32::try_from(i)?)?;
            }
            let routed =
                MatrixView::output(gpu, FloatType::F16, 1, HIDDEN, Storage::StorageModeShared)?;
            Metal::encode(
                cmd,
                &gpu.mapped_mix,
                &[&stack.buffer, &routes.scores.buffer, &routed.buffer],
                [HIDDEN, 8],
                MTLSize::new(u64::from(HIDDEN / 128), 1, 1),
                128,
            );
            let [gate, up, down] = &self.shared;
            let gate = gate.encode(gpu, cmd, &self.affine, &normalized, 0)?;
            let up = up.encode(gpu, cmd, &self.affine, &normalized, 0)?;
            let activated = Moe::encode_activation(gpu, cmd, &[gate, up])?;
            let shared = down.encode(gpu, cmd, &self.affine, &activated, 0)?;
            let mult = self
                .multiplier
                .encode(gpu, cmd, &self.affine, &normalized, 0)?;
            let shared = gpu.encode_model_binary(cmd, &shared, &mult, 2)?;
            let mlp = gpu.encode_model_binary(cmd, &routed, &shared, 0)?;
            let hidden = gpu.encode_model_binary(cmd, &residual, &mlp, 0)?;
            let normalized = encode_rms(gpu, cmd, &hidden, &self.output_norm)?;
            Ok((hidden, target.head.encode(gpu, cmd, &normalized)?))
        })?;
        let logits = words(&logits)?;
        Qwen35::check_logits(&logits, 1)?;
        Ok((select(target, &logits)?, hidden, kv))
    }
}

fn select(target: &Qwen35<'_>, logits: &[u16]) -> Result<u32> {
    greedy_log_prob_id(&target.log_probs(logits)?).context("invalid Qwen log probabilities")
}

/// One request's exact draft cache and trunk residual.
pub struct QwenMtpSession {
    kv: Option<Rc<Kv>>,
    hidden: MatrixView,
    depth: usize,
    stats: [usize; 4],
    proposed: usize,
    accepted: usize,
}

impl QwenMtpSession {
    pub fn prefill(
        target: &mut Qwen35<'_>,
        head: &QwenMtpHead,
        tokens: &[u32],
        depth: usize,
        cancelled: impl Fn() -> bool,
    ) -> Result<(Vec<u16>, Self)> {
        ensure!(
            Rc::ptr_eq(&target.owner, &head.owner),
            "MTP belongs to a different target"
        );
        ensure!(
            (1..=3).contains(&depth),
            "native Qwen MTP depth must be 1 to 3"
        );
        ensure!(
            !tokens.is_empty() && tokens.len() <= target.context_limit() as usize,
            "invalid native MTP prompt length"
        );
        target.reset();
        let mut kv = None;
        let mut last = None;
        for chunk in tokens.chunks(target.prefill_chunk_size()) {
            ensure!(!cancelled(), "generation cancelled");
            let base = target.offset();
            let staged = target.stage_block(chunk, false)?;
            for (i, row) in staged.rows.iter().enumerate() {
                let offset = base + u32::try_from(i)?;
                if let Some(&next) = tokens.get(offset as usize + 1) {
                    ensure!(!cancelled(), "generation cancelled");
                    kv = Some(head.append(target, row, next, offset, kv.as_deref())?);
                } else {
                    let (mut logits, _) = target.project_block(std::slice::from_ref(row))?;
                    last = Some((logits.remove(0), row.clone()));
                }
            }
            target.restore(staged.prefixes.last().context("empty native MTP block")?)?;
        }
        let (logits, hidden) = last.context("missing final native MTP prompt row")?;
        Ok((
            logits,
            Self {
                kv,
                hidden,
                depth,
                stats: [0; 4],
                proposed: 0,
                accepted: 0,
            },
        ))
    }

    pub fn advance(
        &mut self,
        target: &mut Qwen35<'_>,
        head: &QwenMtpHead,
        anchor: u32,
        limit: u32,
        remaining: usize,
        eos: &[u32],
    ) -> Result<Vec<u32>> {
        ensure!(
            Rc::ptr_eq(&target.owner, &head.owner),
            "MTP belongs to a different target"
        );
        let offset = target.offset();
        let previous = offset
            .checked_sub(1)
            .context("MTP needs a prefilled target")?;
        let context = limit.min(target.context_limit()).saturating_sub(offset) as usize;
        let width_limit = mtp_width(self.depth, context, remaining)
            .context("native MTP request is out of budget")?;
        let mut inputs = vec![anchor];
        let mut first_cache = None;
        let mut draft_cache = self.kv.clone();
        let mut hidden = self.hidden.clone();
        for i in 0..width_limit {
            let (token, residual, next) = head.predict(
                target,
                &hidden,
                inputs[i],
                previous + u32::try_from(i)?,
                draft_cache.as_deref(),
            )?;
            if i == 0 {
                first_cache = Some(next.clone());
            }
            if eos.contains(&token) {
                break;
            }
            draft_cache = Some(next);
            hidden = residual;
            inputs.push(token);
        }
        let width = inputs.len() - 1;
        let block = target.verify_block(&inputs)?;
        let targets = block
            .logits()
            .iter()
            .map(|row| select(block.model, row))
            .collect::<Result<Vec<_>>>()?;
        let decision =
            greedy_accept(&inputs[1..], &targets).context("invalid native MTP acceptance")?;
        let accepted = decision.accepted_draft_tokens;
        let mut committed_cache = first_cache;
        if committed_cache.is_none() {
            committed_cache = Some(head.append(
                block.model,
                &self.hidden,
                anchor,
                previous,
                self.kv.as_deref(),
            )?);
        }
        for i in 0..accepted {
            committed_cache = Some(head.append(
                block.model,
                &block.rows[i],
                inputs[i + 1],
                offset + u32::try_from(i)?,
                committed_cache.as_deref(),
            )?);
        }
        let last = block.rows[accepted].clone();
        block.commit(1 + accepted)?;
        self.kv = committed_cache;
        self.hidden = last;
        self.stats[width] += 1;
        self.proposed += width;
        self.accepted += accepted;
        let mut out = inputs[1..1 + accepted].to_vec();
        out.push(decision.target_token);
        Ok(out)
    }

    pub fn stats(&self) -> (usize, usize, usize, Option<[usize; 4]>) {
        (
            self.proposed,
            self.accepted,
            self.stats[1..].iter().sum(),
            Some(self.stats),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    #[ignore = "requires a physical Metal device"]
    fn native_affine4_matches_independent_unpack_and_expert_oracle() -> Result<()> {
        let gpu = Metal::new()?;
        let pipe = pipeline(&gpu)?;
        const INPUT: u32 = 128;
        const OUTPUT: u32 = 6;
        const EXPERTS: u32 = 3;
        let count = (INPUT * OUTPUT * EXPERTS) as usize;
        let values = (0..count)
            .map(|i| ((i * 11 + i / 113) % 16) as u32)
            .collect::<Vec<_>>();
        let packed = values
            .as_chunks::<8>()
            .0
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .fold(0, |out, (i, &q)| out | (q << (4 * i)))
            })
            .collect::<Vec<_>>();
        let scales = (0..count / 64)
            .map(|i| ((i % 7) + 1) as f32 / 32.)
            .collect::<Vec<_>>();
        let biases = (0..count / 64)
            .map(|i| (i as f32 % 9. - 4.) / 8.)
            .collect::<Vec<_>>();
        let x = (0..INPUT)
            .map(|i| (i as f32 % 13. - 6.) / 16.)
            .collect::<Vec<_>>();
        let to_half = |x: &[f32]| {
            x.iter()
                .map(|&v| f16::from_f32(v).to_bits())
                .collect::<Vec<_>>()
        };
        let row = super::super::matrix(&gpu, &to_half(&x), 1, INPUT)?;
        let affine = Affine4 {
            input: INPUT,
            output: OUTPUT,
            experts: EXPERTS,
            weights: crate::gpu::upload(&gpu.device, &packed)?,
            scales: crate::gpu::upload(&gpu.device, &to_half(&scales))?,
            biases: crate::gpu::upload(&gpu.device, &to_half(&biases))?,
        };
        let output = gpu.command(|cmd| {
            let result = MatrixView::output(
                &gpu,
                FloatType::F16,
                EXPERTS,
                OUTPUT,
                Storage::StorageModeShared,
            )?;
            for expert in 0..EXPERTS {
                affine.encode_at(cmd, &pipe, &row, expert, &result, EXPERTS - 1 - expert)?;
            }
            ensure!(
                affine
                    .encode_at(cmd, &pipe, &row, EXPERTS, &result, 0)
                    .is_err(),
                "invalid expert was accepted"
            );
            ensure!(
                affine
                    .encode_at(cmd, &pipe, &row, 0, &result, EXPERTS)
                    .is_err(),
                "invalid output row was accepted"
            );
            Ok(result)
        })?;
        assert_eq!(output.shape(), [EXPERTS, OUTPUT]);
        let result = words(&output)?;
        assert_eq!(result.len(), (EXPERTS * OUTPUT) as usize);
        for (i, actual) in result.into_iter().enumerate() {
            let out = i % OUTPUT as usize;
            let expert = EXPERTS as usize - 1 - i / OUTPUT as usize;
            let base = (expert * OUTPUT as usize + out) * INPUT as usize;
            let expected: f32 = x
                .iter()
                .enumerate()
                .map(|(column, &v)| {
                    let j = base + column;
                    v * (values[j] as f32 * scales[j / 64] + biases[j / 64])
                })
                .sum();
            let actual = f16::from_bits(actual).to_f32();
            assert!(
                actual.is_finite() && (actual - f16::from_f32(expected).to_f32()).abs() <= 0.001
            );
        }
        Ok(())
    }
}
