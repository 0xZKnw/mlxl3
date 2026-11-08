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
pub struct FlashVerification<'model, 'gpu> {
    model: &'model mut FlashNext<'gpu>,
    prefixes: Vec<FlashSnapshot>,
    logits: Vec<Vec<u16>>,
}

impl FlashVerification<'_, '_> {
    pub fn logits(&self) -> &[Vec<u16>] {
        &self.logits
    }

    /// Hidden state after `retained` input rows, for repairing the draft using
    /// verified target activations. Zero has no row inside this block.
    pub fn prefix_hidden(&self, retained: usize) -> Option<&MatrixView> {
        let index = crate::qwen35_config::native_retained_prefix(retained, self.prefixes.len())??;
        self.prefixes.get(index)?.hidden.as_ref()
    }

    pub fn commit(self, retained: usize) -> Result<()> {
        let index = crate::qwen35_config::native_retained_prefix(retained, self.prefixes.len())
            .context("Flash commit exceeds verified block")?;
        if let Some(index) = index {
            self.model.restore(&self.prefixes[index])?;
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
        let end =
            crate::qwen35_config::native_block_end(tokens, 248_320, self.offset, self.context)
                .context("Flash block must fit the context and contain 1 to 4 valid tokens")?;
        // A block can retain multiple full recurrent frames and QSA slab forks.
        // Do not bypass the existing AR-only memory admission until its
        // prospective speculative accounting has been qualified.
        ensure!(
            crate::flash_memory::native_speculation_permitted(self.request_limit),
            "Flash request memory budget supports AR only"
        );
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
                ensure!(layer.moe().block_eligible(), "unsupported settings for Flash SSD cohorts");
            }
        }
        ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
        self.ensure_fixed_residency()?;
        let before = self.snapshot();
        let operation = self.stage_block_inner(tokens, end, cancelled);
        // All snapshots here were made by this same target and its layers.
        // Restore even after a late layer/head/cancellation error. A successful
        // handle never needs fallible GPU work in Drop.
        self.restore(&before)?;
        let (prefixes, logits) = operation?;
        Ok(FlashVerification {
            model: self,
            prefixes,
            logits,
        })
    }

    fn stage_block_inner(
        &mut self,
        tokens: &[u32],
        end: u32,
        cancelled: &AtomicBool,
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
                let mut prepared = Vec::with_capacity(rows.len());
                for ((row, &token), prefix) in rows.iter().zip(tokens).zip(&mut prefixes) {
                    ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
                    prepared.push(layer.prepare_moe(row, token)
                        .with_context(|| format!("preparing Flash cohort layer {index}"))?);
                    // Only attention/PLE carries temporal state. MoE and final
                    // HC write cannot alter this canonical per-row prefix.
                    prefix.layers.push(layer.snapshot());
                }
                let mixed = prepared.iter().map(|row| row.mixed.clone()).collect::<Vec<_>>();
                let experts = layer.moe().forward_block_cancellable(&mixed, cancelled)?;
                ensure!(experts.len() == rows.len(), "incomplete Flash cohort outputs");
                for ((row, prepared), experts) in rows.iter_mut().zip(prepared).zip(experts) {
                    ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
                    *row = layer.finish_prepared(prepared, experts)?;
                }
                #[cfg(test)]
                ensure!(gpu.cross_layer_fail_after.get() != Some(index), "injected late verification layer failure {index}");
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
    #[ignore = "physical Flash checkpoint; SSD cohort temporal preparation and prefix rollback"]
    fn ssd_cohort_verification_matches_scalar_and_rejects_unsafe_workspace() -> Result<()> {
        let c = crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let gpu = Metal::new()?;
        // Run with cohort-compatible flags externally. Do not hide unsupported
        // settings by mutating them during load or silently selecting fallback.
        let mut model = FlashNext::load(&gpu, &c, 64)?;
        ensure!(model.layers.iter_mut().all(|layer| layer.moe().block_eligible()));
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
                let expected_hidden = model.last_hidden().context("missing scalar hidden")?.export_bytes(&gpu)?;
                let mut continuation = Vec::new();
                for token in [7, 8, 9, 10] {
                    continuation.push(model.forward_trace(token)?);
                }
                model.restore(&root)?;
                model.verify_ssd_cohorts = true;
                let cohort = model.stage_block(tokens)?;
                ensure!(cohort.logits() == logits && logits.len() == width);
                cohort.commit(keep)?;
                ensure!(retained_state(&model)? == expected_state);
                ensure!(model.last_hidden().context("missing cohort hidden")?.export_bytes(&gpu)? == expected_hidden);
                for (token, (logits, trace)) in [7, 8, 9, 10].into_iter().zip(continuation) {
                    let (actual_logits, actual_trace) = model.forward_trace(token)?;
                    ensure!(actual_logits == logits);
                    same_trace(&actual_trace, &trace)?;
                }
                model.restore(&root)?;
            }
        }
        // Reject workspace retention before preparing even the first layer.
        gpu.hc_workspace_enabled.set(true);
        let rejected = model.stage_block(&[1, 2]).is_err();
        gpu.hc_workspace_enabled.set(false);
        ensure!(rejected && model.offset == root.offset);
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
