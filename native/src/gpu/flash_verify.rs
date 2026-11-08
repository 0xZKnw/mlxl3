//! Conservative layer-major Flash verification, with explicit prefix ownership.
//!
//! The default stages scalar canonical kernels by layer. An explicit SSD cohort
//! option groups stateless MoE reads after owned temporal preparation. Neither
//! path promises throughput or a multirow EXL3 kernel.
use super::{FlashNext, FlashSnapshot, MatrixView};
use anyhow::{Context, Result, ensure};
use std::sync::atomic::{AtomicBool, Ordering};

/// Exclusively borrows the target while verified prefixes are available.
/// The target has already been restored to its starting prefix before this is
/// returned. Drop, `commit(0)` and an invalid commit therefore publish no prefix.
/// Cache residency and allocation high-water marks are not transactional state.
#[must_use = "a staged block must be committed explicitly; dropping it keeps the original prefix"]
pub struct BlockTxn<'model, 'gpu> {
    model: &'model mut FlashNext<'gpu>,
    prefixes: Vec<FlashSnapshot>,
    logits: Vec<Vec<u16>>,
    publications: Vec<super::super::qsa::VerificationPublication>,
}

pub type FlashVerification<'model, 'gpu> = BlockTxn<'model, 'gpu>;

impl BlockTxn<'_, '_> {
    pub fn logits(&self) -> &[Vec<u16>] {
        &self.logits
    }

    pub(super) fn greedy(&mut self, retained: usize, device: bool) -> Result<u32> {
        let index = crate::qwen35_config::native_retained_prefix(retained, self.prefixes.len())
            .flatten()
            .context("invalid verification head row")?;
        let hidden = self.prefixes[index]
            .hidden
            .as_ref()
            .context("missing verification hidden row")?;
        let mixed = self.model.mixer.read(hidden, false)?.mixed;
        #[cfg(test)]
        {
            self.model.target_head_calls += 1;
        }
        if device {
            self.model
                .gpu
                .device_greedy_head(&self.model.head, &mixed, false)
                .map(|value| value.0)
        } else {
            let logits = self.model.head.submit_fp16_view(&mixed)?.read()?;
            self.model.greedy_id(&logits)
        }
    }

    /// Hidden state after `retained` input rows, for repairing the draft using
    /// verified target activations. Zero has no row inside this block.
    pub fn prefix_hidden(&self, retained: usize) -> Option<&MatrixView> {
        let index = crate::qwen35_config::native_retained_prefix(retained, self.prefixes.len())??;
        self.prefixes.get(index)?.hidden.as_ref()
    }

    pub fn commit(mut self, retained: usize) -> Result<()> {
        let index = crate::qwen35_config::native_retained_prefix(retained, self.prefixes.len())
            .context("Flash commit exceeds verified block")?;
        if let Some(index) = index {
            self.model.restore(&self.prefixes[index])?;
            self.prefixes.clear();
            let mut marks = self.publications.iter_mut();
            for layer in &self.model.layers {
                if let super::Layer::Qsa(layer) = layer {
                    marks
                        .next()
                        .expect("one QSA publication mark per layer")
                        .retain(layer);
                }
            }
        }
        Ok(())
    }
}

impl<'gpu> FlashNext<'gpu> {
    /// Stage one through four input tokens in layer-major order. No prefix is
    /// published until the returned handle is consumed with `commit`.
    pub fn stage_block(&mut self, tokens: &[u32]) -> Result<FlashVerification<'_, 'gpu>> {
        self.stage_block_cancellable(tokens, &AtomicBool::new(false))
    }

    pub fn stage_block_cancellable(
        &mut self,
        tokens: &[u32],
        cancelled: &AtomicBool,
    ) -> Result<FlashVerification<'_, 'gpu>> {
        self.stage_block_heads(tokens, cancelled, true)
    }

    pub(super) fn stage_block_headless(
        &mut self,
        tokens: &[u32],
        cancelled: &AtomicBool,
    ) -> Result<FlashVerification<'_, 'gpu>> {
        self.stage_block_heads(tokens, cancelled, false)
    }

    fn stage_block_heads(
        &mut self,
        tokens: &[u32],
        cancelled: &AtomicBool,
        heads: bool,
    ) -> Result<FlashVerification<'_, 'gpu>> {
        let end =
            crate::qwen35_config::native_block_end(tokens, 248_320, self.offset, self.context)
                .context("Flash block must fit the context and contain 1 to 4 valid tokens")?;
        // Admit all owned prefixes and prospective scratch before temporal mutation.
        ensure!(
            self.layer_plans
                .iter()
                .all(|plan| plan.admits(tokens.len())),
            "native block layer plan rejected the shape"
        );
        self.admit_native_block(tokens.len())?;
        // Scalar lookahead assumes the next scheduled consumer is the next
        // layer, not another row of this layer. Keep it out of this prototype.
        ensure!(
            self.lookahead_depth == 0 && self.prefetcher.is_none(),
            "Flash block verification requires SSD lookahead disabled"
        );
        if self.verify_ssd_cohorts {
            // Preflight every layer before preparing any temporal state.
            // Each MoE repeats this check defensively before doing I/O.
            for layer in &mut self.layers {
                ensure!(
                    layer.moe().block_eligible(),
                    "unsupported settings for Flash SSD cohorts"
                );
            }
        }
        ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
        self.ensure_fixed_residency()?;
        let before = self.snapshot();
        let publications = self
            .layers
            .iter()
            .filter_map(|layer| match layer {
                super::Layer::Qsa(layer) => Some(layer.verification_publication()),
                _ => None,
            })
            .collect();
        let operation = self.stage_block_inner(tokens, end, cancelled, heads);
        // All snapshots here were made by this same target and its layers.
        // Restore even after a late layer/head/cancellation error. A successful
        // handle never needs fallible GPU work in Drop.
        if operation.is_err() {
            // Retire every producer, even if an earlier layer reports a GPU fault.
            for layer in &mut self.layers {
                let _ = layer.moe().drain();
            }
        }
        self.restore(&before)?;
        let (prefixes, logits) = operation?;
        Ok(BlockTxn {
            model: self,
            prefixes,
            logits,
            publications,
        })
    }

    fn stage_block_inner(
        &mut self,
        tokens: &[u32],
        end: u32,
        cancelled: &AtomicBool,
        heads: bool,
    ) -> Result<(Vec<FlashSnapshot>, Vec<Vec<u16>>)> {
        let mut prefixes = (self.offset + 1..=end)
            .map(|offset| FlashSnapshot {
                owner: self.owner.clone(),
                offset,
                layers: Vec::with_capacity(self.layers.len()),
                hidden: None,
            })
            .collect::<Vec<_>>();
        let mut rows = tokens
            .iter()
            .map(|&token| self.streams(token))
            .collect::<Result<Vec<_>>>()?;
        let gpu = self.gpu;
        for (index, layer) in self.layers.iter_mut().enumerate() {
            if self.verify_ssd_cohorts {
                ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
                let projected = layer.prepare_block_inputs(&rows)?;
                ensure!(
                    projected
                        .as_ref()
                        .is_none_or(|rows_prepared| rows_prepared.len() == rows.len()),
                    "incomplete GDN block projections"
                );
                let mut projected = projected.map(Vec::into_iter);
                let mut prepared = Vec::with_capacity(rows.len());
                for ((row, &token), prefix) in rows.iter().zip(tokens).zip(&mut prefixes) {
                    ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
                    prepared.push(
                        layer
                            .prepare_moe(row, token, projected.as_mut().and_then(Iterator::next))
                            .with_context(|| format!("preparing Flash cohort layer {index}"))?,
                    );
                    // Only attention/PLE carries temporal state. MoE and final
                    // HC write cannot alter this canonical per-row prefix.
                    prefix.layers.push(layer.snapshot());
                }
                let mixed = prepared
                    .iter()
                    .map(|row| row.mixed.clone())
                    .collect::<Vec<_>>();
                let routing = prepared.iter_mut().map(|row| row.routing.take()).collect();
                let experts = layer
                    .moe()
                    .forward_block_routed_cancellable(&mixed, routing, cancelled)?;
                ensure!(
                    experts.len() == rows.len(),
                    "incomplete Flash cohort outputs"
                );
                rows = gpu.command(|cmd| {
                    prepared
                        .iter()
                        .zip(&experts)
                        .map(|(row, experts)| {
                            ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
                            layer.finish_prepared(cmd, row, experts)
                        })
                        .collect::<Result<Vec<_>>>()
                })?;
                for output in &experts {
                    output.completed()?;
                }
                #[cfg(test)]
                ensure!(
                    gpu.cross_layer_fail_after.get() != Some(index),
                    "injected late verification layer failure {index}"
                );
                continue;
            }
            for ((row, &token), prefix) in rows.iter_mut().zip(tokens).zip(&mut prefixes) {
                ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
                // Complete each write before retaining its row. Cross-layer
                // deferred writes cannot safely cross another temporal row.
                *row = layer
                    .forward(row, token, None, 0, false)
                    .with_context(|| format!("verifying Flash layer {index}"))?
                    .complete(gpu)?;
                #[cfg(test)]
                ensure!(
                    gpu.cross_layer_fail_after.get() != Some(index),
                    "injected late verification layer failure {index}"
                );
                prefix.layers.push(layer.snapshot());
            }
        }
        let mut logits = Vec::with_capacity(rows.len());
        for (row, prefix) in rows.into_iter().zip(&mut prefixes) {
            ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
            if !heads {
                prefix.hidden = Some(row);
                continue;
            }
            let mixed = self.mixer.read(&row, false)?.mixed;
            #[cfg(test)]
            {
                self.target_head_calls += 1;
            }
            let values = self.head.submit_fp16_view(&mixed)?.read()?;
            ensure!(
                values.len() == 248_320
                    && values
                        .iter()
                        .all(|&value| half::f16::from_bits(value).is_finite()),
                "nonfinite or incomplete Flash verification logits"
            );
            prefix.hidden = Some(row);
            logits.push(values);
        }
        ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
        Ok((prefixes, logits))
    }
}

#[cfg(test)]
mod tests {
    use super::super::Layer;
    use super::*;
    use crate::gpu::{FlashTraceTensor, Metal};
    use std::collections::BTreeMap;

    fn retained_state(model: &FlashNext<'_>) -> Result<BTreeMap<String, Vec<u8>>> {
        let mut state = BTreeMap::new();
        for (index, layer) in model.layers.iter().enumerate() {
            if let Some(recurrent) = layer.recurrent() {
                state.insert(format!("gdn_{index}"), recurrent.export_bytes(model.gpu)?);
            }
            if let Layer::Qsa(layer) = layer {
                state.insert(
                    format!("offset_{index}"),
                    layer.offset().to_le_bytes().to_vec(),
                );
                for (name, words) in layer.retained_half_words()? {
                    state.insert(
                        format!("qsa_{index}_{name}"),
                        words.into_iter().flat_map(u16::to_le_bytes).collect(),
                    );
                }
            }
        }
        ensure!(!state.is_empty());
        Ok(state)
    }

    fn same_trace(actual: &[FlashTraceTensor], expected: &[FlashTraceTensor]) -> Result<()> {
        ensure!(!actual.is_empty() && actual.len() == expected.len());
        for (a, b) in actual.iter().zip(expected) {
            ensure!(
                a.name == b.name && a.shape == b.shape && a.data == b.data && !a.data.is_empty(),
                "verification continuation changed {}",
                a.name
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "physical Flash checkpoint; exclusive Apple GPU; full scalar-vs-block logits/state oracle"]
    fn layer_major_verification_matches_scalar_prefixes_and_continuations() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut model = FlashNext::load(&gpu, &c, 64)?;
        // Prefix lengths 1..4 exercise every raw QSA tail alignment; width
        // 1..4 exercises all commit points without trusting generated proposals.
        for prefix_len in 1..=4 {
            model.reset();
            model.prefill(&[248044, 78768, 11, 7][..prefix_len])?;
            let root = model.snapshot();
            for width in 1..=4 {
                let tokens = &[1, 2, 3, 4][..width];
                let mut expected_logits = Vec::new();
                let mut scalar_prefixes = vec![root.clone()];
                for &token in tokens {
                    let logits = model.forward(token)?;
                    ensure!(
                        logits.len() == 248_320
                            && logits.iter().all(|&v| half::f16::from_bits(v).is_finite())
                    );
                    expected_logits.push(logits);
                    scalar_prefixes.push(model.snapshot());
                }
                model.restore(&root)?;
                for (keep, scalar_prefix) in scalar_prefixes.iter().enumerate().take(width + 1) {
                    model.restore(scalar_prefix)?;
                    let expected_state = retained_state(&model)?;
                    let expected_hidden = model
                        .last_hidden()
                        .context("missing scalar hidden")?
                        .export_bytes(&gpu)?;
                    let mut expected_continuation = Vec::new();
                    for token in [7, 8, 9, 10] {
                        expected_continuation.push(model.forward_trace(token)?);
                    }
                    model.restore(&root)?;
                    let block = model.stage_block(tokens)?;
                    ensure!(block.logits() == expected_logits);
                    ensure!(
                        block.prefix_hidden(0).is_none()
                            && block.prefix_hidden(width + 1).is_none()
                    );
                    if keep > 0 {
                        ensure!(
                            block
                                .prefix_hidden(keep)
                                .context("missing verified hidden")?
                                .export_bytes(&gpu)?
                                == expected_hidden
                        );
                    }
                    block.commit(keep)?;
                    ensure!(model.offset() == root.offset + keep as u32);
                    ensure!(retained_state(&model)? == expected_state);
                    ensure!(
                        model
                            .last_hidden()
                            .context("missing committed hidden")?
                            .export_bytes(&gpu)?
                            == expected_hidden
                    );
                    for (token, (wanted_logits, wanted_trace)) in
                        [7, 8, 9, 10].into_iter().zip(expected_continuation)
                    {
                        let (actual_logits, actual_trace) = model.forward_trace(token)?;
                        ensure!(actual_logits == wanted_logits);
                        same_trace(&actual_trace, &wanted_trace)?;
                    }
                }
                model.restore(&root)?;
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "physical Flash checkpoint; exclusive Apple GPU; staging rollback and cancellation"]
    fn dropped_invalid_failed_and_cancelled_blocks_preserve_target() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut model = FlashNext::load(&gpu, &c, 32)?;
        model.prefill(&[248044, 78768, 11])?;
        let root = model.snapshot();
        let root_state = retained_state(&model)?;
        let expected = model.forward_trace(7)?;
        model.restore(&root)?;
        drop(model.stage_block(&[1, 2])?);
        ensure!(model.offset() == root.offset);
        ensure!(retained_state(&model)? == root_state);
        let actual = model.forward_trace(7)?;
        ensure!(actual.0 == expected.0);
        same_trace(&actual.1, &expected.1)?;
        model.restore(&root)?;
        ensure!(model.stage_block(&[1, 2])?.commit(3).is_err());
        for tokens in [&[][..], &[248_320][..], &[1, 2, 3, 4, 5][..]] {
            ensure!(model.stage_block(tokens).is_err());
            ensure!(model.offset() == root.offset);
        }
        let cancelled = AtomicBool::new(true);
        ensure!(model.stage_block_cancellable(&[1, 2], &cancelled).is_err());
        gpu.cross_layer_fail_after.set(Some(3));
        let failed = model.stage_block(&[1, 2]).is_err();
        gpu.cross_layer_fail_after.set(None);
        ensure!(failed && model.offset() == root.offset);
        ensure!(retained_state(&model)? == root_state);
        let actual = model.forward_trace(7)?;
        ensure!(actual.0 == expected.0);
        same_trace(&actual.1, &expected.1)?;
        Ok(())
    }

    #[test]
    #[ignore = "physical checkpoint; lazy heads, exact GPU selectors and external QSA future snapshots"]
    fn lazy_heads_and_publication_retirement_preserve_external_snapshots() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        let mut model = FlashNext::load(&gpu, &c, 64)?;
        model.prefill(&[248044, 78768, 11])?;
        let root = model.snapshot();
        let root_state = retained_state(&model)?;
        let first_logits = model.forward(1)?;
        let first = model.greedy_id(&first_logits)?;
        let retained = model.snapshot();
        let expected_retained_state = retained_state(&model)?;
        let second_logits = model.forward(2)?;
        let second = model.greedy_id(&second_logits)?;
        let future = model.snapshot();
        let future_state = retained_state(&model)?;
        let continuation = model.forward(7)?;
        {
            model.restore(&root)?;
            for device in [false, true] {
                if device && !gpu.device_greedy_available() {
                    continue;
                }
                model.restore(&root)?;
                let heads_before = model.target_head_calls;
                let mut staged = model.stage_block_headless(&[1, 2], &AtomicBool::new(false))?;
                ensure!(staged.logits().is_empty());
                ensure!(staged.greedy(0, device).is_err() && staged.greedy(3, device).is_err());
                ensure!(staged.greedy(1, device)? == first);
                ensure!(staged.greedy(2, device)? == second);
                staged.commit(1)?;
                ensure!(model.target_head_calls - heads_before == 2);
                ensure!(retained_state(&model)? == expected_retained_state);
                ensure!(model.forward(2)? == second_logits);
                model.restore(&future)?;
                ensure!(retained_state(&model)? == future_state);
                ensure!(model.forward(7)? == continuation);
            }
        }
        model.restore(&root)?;
        drop(model.stage_block_headless(&[1, 2], &AtomicBool::new(false))?);
        ensure!(retained_state(&model)? == root_state);
        drop(retained);
        for keep in 0..=2 {
            model.reset();
            model.prefill(&[248044, 78768, 11])?;
            let identities = model
                .layers
                .iter()
                .filter_map(|layer| match layer {
                    Layer::Qsa(layer) => Some(layer.verification_slab().0),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let calls = model.target_head_calls;
            let mut staged = model.stage_block_headless(&[1, 2], &AtomicBool::new(false))?;
            if keep == 1 {
                ensure!(staged.greedy(1, false)? == first);
            }
            staged.commit(keep)?;
            ensure!(model.target_head_calls - calls == usize::from(keep == 1));
            for (layer, identity) in model
                .layers
                .iter()
                .filter_map(|layer| match layer {
                    Layer::Qsa(layer) => Some(layer),
                    _ => None,
                })
                .zip(&identities)
            {
                ensure!(layer.verification_slab() == (*identity, 3 + keep as u32));
            }
            model.forward(7)?;
            for (layer, identity) in model
                .layers
                .iter()
                .filter_map(|layer| match layer {
                    Layer::Qsa(layer) => Some(layer),
                    _ => None,
                })
                .zip(&identities)
            {
                ensure!(layer.verification_slab().0 == *identity);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "physical Flash checkpoint; SSD cohort temporal preparation and prefix rollback"]
    fn ssd_cohort_verification_matches_scalar_and_rejects_unsafe_workspace() -> Result<()> {
        let c =
            crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        // Run with cohort-compatible flags externally. Do not hide unsupported
        // settings by mutating them during load or silently selecting fallback.
        let mut model = FlashNext::load(&gpu, &c, 64)?;
        ensure!(gpu.hc_chain.get() && gpu.hc_gate_chain.get());
        ensure!(
            model
                .layers
                .iter_mut()
                .all(|layer| layer.moe().block_eligible())
        );
        model.prefill(&[248044, 78768, 11])?;
        let root = model.snapshot();
        let root_state = retained_state(&model)?;
        for width in 2..=4 {
            let tokens = &[1, 2, 3, 4][..width];
            for keep in 0..=width {
                model.verify_ssd_cohorts = false;
                let reference = model.stage_block(tokens)?;
                let logits = reference.logits().to_vec();
                reference.commit(keep)?;
                let expected_state = retained_state(&model)?;
                let expected_hidden = model
                    .last_hidden()
                    .context("missing scalar hidden")?
                    .export_bytes(&gpu)?;
                let mut continuation = Vec::new();
                for token in [7, 8, 9, 10] {
                    continuation.push(model.forward_trace(token)?);
                }
                model.restore(&root)?;
                model.verify_ssd_cohorts = true;
                let joint_before = gpu.hc_gate_chain_pairs.get();
                let gdn_before = crate::gpu::FlashGdnLayer::block_projection_count();
                let temporal_before = crate::gpu::FlashGdnLayer::temporal_batch_count();
                let segments_before = gpu.segment_submissions.get();
                let hot_before = gpu.device_hot_submissions();
                let finish_before = gpu.moe_finish_mix_uses.get();
                let workspace_before = gpu.hc_workspace_uses.get();
                let standalone_before = gpu.gate_commands.get();
                let expected_joint = width * model.layers.len();
                let specialized_before = gpu.flash_verify_specialized_uses.get();
                let cohort = model.stage_block(tokens)?;
                if gpu.common_executor && gpu.device_resident_route_enabled() {
                    ensure!(
                        gpu.device_hot_submissions() > hot_before
                            && gpu.device_resident_route_available(),
                        "private Hot producers were not executed or retired"
                    );
                }
                if gpu.common_executor && gpu.moe_finish_mix.get() {
                    ensure!(
                        gpu.moe_finish_mix_uses.get() > finish_before,
                        "block finish/mix was not executed"
                    );
                }
                if gpu.common_executor && gpu.hc_workspace_enabled.get() {
                    ensure!(
                        gpu.hc_workspace_uses.get() > workspace_before,
                        "owned HC workspace leases were not executed"
                    );
                }
                if gpu.gdn.temporal.is_some() && gpu.gdn_whole.get() {
                    let delta = crate::gpu::FlashGdnLayer::temporal_batch_count() - temporal_before;
                    ensure!(delta > 0, "temporal GDN kernel was not reached");
                    eprintln!("temporal GDN batches: {delta}");
                }
                if gpu.segment_enabled {
                    ensure!(
                        gpu.segment_submissions.get() > segments_before
                            && gpu.segment_pending.get() == 0,
                        "GDN/HC/router segment was not retired"
                    );
                }
                if gpu.flash_verify_specialized.is_some()
                    && crate::flash_qmv::shared_verify_enabled(gpu.flash_verify_qmv.is_some())
                {
                    let counts = gpu.flash_verify_specialized_uses.get();
                    let delta =
                        std::array::from_fn::<_, 7, _>(|i| counts[i] - specialized_before[i]);
                    eprintln!("MB2 specialized PSOs: width={width}, keep={keep}, counts={delta:?}");
                    ensure!(
                        delta.iter().any(|&count| count > 0),
                        "no specialized MB2 projection was reached"
                    );
                }
                if crate::flash_qmv::shared_verify_enabled(gpu.flash_verify_qmv.is_some()) {
                    eprintln!(
                        "GDN projection batches: width={width}, keep={keep}, layers={}",
                        crate::gpu::FlashGdnLayer::block_projection_count() - gdn_before
                    );
                    ensure!(
                        crate::gpu::FlashGdnLayer::block_projection_count() > gdn_before,
                        "GDN target projections did not use actual token batching"
                    );
                }
                ensure!(
                    gpu.hc_gate_chain_pairs.get() - joint_before == expected_joint
                        && gpu.gate_commands.get() == standalone_before,
                    "SSD cohort lost the canonical joint HC/router producer"
                );
                ensure!(cohort.logits() == logits && logits.len() == width);
                cohort.commit(keep)?;
                ensure!(retained_state(&model)? == expected_state);
                ensure!(
                    model
                        .last_hidden()
                        .context("missing cohort hidden")?
                        .export_bytes(&gpu)?
                        == expected_hidden
                );
                for (token, (logits, trace)) in [7, 8, 9, 10].into_iter().zip(continuation) {
                    let (actual_logits, actual_trace) = model.forward_trace(token)?;
                    ensure!(actual_logits == logits);
                    same_trace(&actual_trace, &trace)?;
                }
                model.restore(&root)?;
            }
        }
        let joint_enabled = gpu.hc_gate_chain.replace(false);
        let fallback = (|| -> Result<()> {
            model.verify_ssd_cohorts = false;
            let reference = model.stage_block(&[1, 2])?;
            let logits = reference.logits().to_vec();
            drop(reference);
            model.verify_ssd_cohorts = true;
            let joint_before = gpu.hc_gate_chain_pairs.get();
            let standalone_before = gpu.gate_commands.get();
            let cohort = model.stage_block(&[1, 2])?;
            ensure!(cohort.logits() == logits);
            ensure!(gpu.hc_gate_chain_pairs.get() == joint_before);
            ensure!(gpu.gate_commands.get() > standalone_before);
            drop(cohort);
            ensure!(retained_state(&model)? == root_state);
            Ok(())
        })();
        gpu.hc_gate_chain.set(joint_enabled);
        fallback?;
        // Reject workspace retention before preparing even the first layer.
        gpu.hc_workspace_enabled.set(true);
        let rejected = model.stage_block(&[1, 2]).is_err();
        gpu.hc_workspace_enabled.set(false);
        ensure!(rejected != gpu.common_executor && model.offset == root.offset);
        ensure!(retained_state(&model)? == root_state);
        gpu.cross_layer_fail_after.set(Some(3));
        let failed = model.stage_block(&[1, 2]).is_err();
        gpu.cross_layer_fail_after.set(None);
        ensure!(failed && model.offset == root.offset);
        ensure!(retained_state(&model)? == root_state);
        drop(model.stage_block(&[1, 2])?);
        ensure!(retained_state(&model)? == root_state && model.offset == root.offset);
        Ok(())
    }
}
