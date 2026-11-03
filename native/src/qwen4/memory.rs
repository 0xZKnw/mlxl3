//! Memory policy runs at load/request boundaries, outside the token loop.
use super::*;
use crate::flash_memory::{ResourcePlan, fits, shrink_slots};

pub(super) fn request_limit() -> Result<Option<u64>> {
    match std::env::var("MLXL3_FLASH_REQUEST_MIB") {
        Ok(text) => Ok(Some(
            text.parse::<u64>()
                .ok()
                .filter(|n| *n > 0)
                .and_then(|n| n.checked_mul(1 << 20))
                .context("MLXL3_FLASH_REQUEST_MIB requires a positive integer MiB budget")?,
        )),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.into()),
    }
}

fn observed(memory: array::MemoryStats) -> Result<u64> {
    Ok(memory
        .process_footprint_bytes
        .context("Flash admission requires process footprint")?
        .max(
            memory
                .mlx_active_bytes
                .checked_add(memory.mlx_cache_bytes)
                .context("MLX allocation accounting overflow")?,
        ))
}

pub(super) fn admit_load(limit: Option<u64>, slots: usize) -> Result<()> {
    if let Some(limit) = limit {
        let bytes = (slots as u64)
            .checked_mul(flash_support::EXPERT_BYTES)
            .context("Flash bank size overflow")?;
        ensure!(
            fits(observed(array::memory_stats(false)?)?, bytes, limit),
            "Flash memory budget too small for expert banks"
        );
    }
    Ok(())
}

#[derive(Default)]
pub(super) struct Backings {
    seen: std::collections::HashSet<usize>,
    bytes: BTreeMap<&'static str, u64>,
}

impl Backings {
    pub(super) fn record(&mut self, category: &'static str, array: &Array) -> Result<()> {
        let (key, bytes) = array.backing()?;
        if self.seen.insert(key) {
            let total = self.bytes.entry(category).or_default();
            *total = total
                .checked_add(bytes)
                .context("Flash backing size overflow")?;
        }
        Ok(())
    }

    fn qsa(&mut self, state: &QsaState, snapshot: bool) -> Result<()> {
        for (category, value) in [
            ("qsa_kv", &state.keys),
            ("qsa_kv", &state.values),
            ("qsa_index", &state.index_tail),
            ("qsa_index", &state.blocks),
        ] {
            if let Some(array) = value {
                self.record(if snapshot { "snapshot" } else { category }, array)?;
            }
        }
        Ok(())
    }
}

#[derive(serde::Serialize)]
pub struct MemoryLedger {
    request_limit_bytes: Option<u64>,
    program_cache: array::FlashProgramCache,
    expert_slots_by_layer: Vec<usize>,
    expert_bank_payload_bytes: u64,
    // One independently allocated backing per bank. Snapshot aliases are excluded.
    expert_bank_backing_bytes: u64,
    fixed_weight_backing_bytes: u64,
    fixed_weight_arrays_not_ready: usize,
    state_backing_bytes: BTreeMap<&'static str, u64>,
    ple_host_row_payload_bytes: usize,
    mlx_active_other_bytes: Option<u64>,
    memory: array::MemoryStats,
}

impl Hc {
    fn collect_weights<'a>(&'a self, out: &mut Vec<&'a Array>) {
        out.push(&self.norm.gain);
        for p in [&self.down, &self.up].into_iter().chain(&self.inject) {
            p.collect_weights(out);
        }
    }
}
impl Layer {
    fn collect_weights<'a>(&'a self, out: &mut Vec<&'a Array>) {
        self.attn_hc.collect_weights(out);
        self.mlp_hc.collect_weights(out);
        self.moe.collect_weights(out);
        match &self.attention {
            Attention::Gdn(g) => {
                let w = &g.weights;
                out.extend([&w.conv, &w.a_log, &w.dt_bias, &w.norm.gain]);
                out.extend(&w.prepared_conv);
                for p in [&w.qkv, &w.z, &w.a, &w.b, &w.out] {
                    p.collect_weights(out);
                }
            }
            Attention::Qsa(q) => {
                let w = &q.weights;
                out.extend([
                    &w.q_norm.gain,
                    &w.k_norm.gain,
                    &w.iq_norm.gain,
                    &w.ik_norm.gain,
                ]);
                out.extend(&w.frequencies);
                for p in [&w.q, &w.k, &w.v, &w.out, &w.index] {
                    p.collect_weights(out);
                }
            }
        }
        if let Some(p) = &self.ple {
            out.extend([&p.nk.gain, &p.nq.gain, &p.nc.gain, &p.conv]);
            p.key.collect_weights(out);
            p.value.collect_weights(out);
        }
    }
}
impl FlashNext {
    /// Called after the bridge's idle synchronization on the inference thread.
    pub fn clear_idle_program_cache(&self) -> Result<()> {
        array::flash_program_cache(true)?;
        Ok(())
    }

    pub fn request_memory_limit(&self) -> Option<u64> {
        self.request_limit
    }

    pub fn memory_ledger(&self, mtp: Option<&FlashMtpHead>) -> Result<MemoryLedger> {
        let slots: Vec<_> = self.layers.iter().map(|l| l.moe.capacity()).collect();
        let banks = self.layers.iter().try_fold(0u64, |total, l| {
            total
                .checked_add(l.moe.bank_backing_bytes()?)
                .context("Flash banks overflow")
        })?;
        let mut backings = Backings::default();
        let mut ple_host_row_payload_bytes = 0;
        for layer in &self.layers {
            match &layer.attention {
                Attention::Gdn(g) => {
                    for a in [&g.state, &g.history].into_iter().flatten() {
                        backings.record("gdn", a)?;
                    }
                }
                Attention::Qsa(q) => backings.qsa(&q.state, false)?,
            }
            if let Some(ple) = &layer.ple {
                if let Some(a) = &ple.history {
                    backings.record("ple", a)?;
                }
                ple_host_row_payload_bytes += ple.rows.cache.len() * 82;
            }
        }
        if let Some(a) = &self.last_hidden {
            backings.record("last_hidden", a)?;
        }
        if let Some(pair) = &self.verification {
            backings.record("snapshot", &pair.hidden)?;
            for layer in &pair.layers {
                match &layer.attention {
                    AttentionState::Gdn { state, history } => {
                        for a in [state, history].into_iter().flatten() {
                            backings.record("snapshot", a)?;
                        }
                    }
                    AttentionState::GdnTape { history, tape, .. } => {
                        if let Some(a) = history {
                            backings.record("snapshot", a)?;
                        }
                        for a in tape.buffers() {
                            backings.record("tape", a)?;
                        }
                    }
                    AttentionState::Qsa(q) => backings.qsa(q, true)?,
                }
                if let Some((Some(a), _)) = &layer.ple {
                    backings.record("snapshot", a)?;
                }
            }
        }
        #[cfg(feature = "chat")]
        if let Some(session) = &self.dflash {
            session.record_memory(&mut backings)?;
        }
        let mut weights = Vec::new();
        self.assets.head.collect_weights(&mut weights);
        self.assets.mixer.collect_weights(&mut weights);
        for layer in &self.layers {
            layer.collect_weights(&mut weights);
        }
        let mut fixed_weight_arrays_not_ready = 0;
        for weight in weights {
            if weight.is_available()? {
                backings.record("fixed_weights", weight)?;
            } else {
                fixed_weight_arrays_not_ready += 1;
            }
        }
        if let Some(head) = mtp {
            let mut weights = vec![&head.embedding_norm.gain, &head.hidden_norm.gain];
            head.mixer.collect_weights(&mut weights);
            head.layer.collect_weights(&mut weights);
            head.embedding_fc.collect_weights(&mut weights);
            head.hidden_fc.collect_weights(&mut weights);
            for weight in weights {
                if weight.is_available()? {
                    backings.record("mtp_weights", weight)?;
                }
            }
            let Attention::Qsa(q) = &head.layer.attention else {
                anyhow::bail!("invalid Flash MTP attention")
            };
            for a in [
                &q.state.keys,
                &q.state.values,
                &q.state.index_tail,
                &q.state.blocks,
            ]
            .into_iter()
            .flatten()
            {
                backings.record("mtp_kv_index", a)?;
            }
            backings
                .bytes
                .insert("mtp_expert_banks", head.layer.moe.bank_backing_bytes()?);
        }
        let memory = array::memory_stats(false)?;
        let owned = backings
            .bytes
            .values()
            .try_fold(banks, |t, b| t.checked_add(*b))
            .context("Flash live bytes overflow")?;
        Ok(MemoryLedger {
            request_limit_bytes: self.request_limit,
            program_cache: array::flash_program_cache(false)?,
            expert_bank_payload_bytes: slots.iter().sum::<usize>() as u64
                * flash_support::EXPERT_BYTES,
            expert_slots_by_layer: slots,
            expert_bank_backing_bytes: banks,
            fixed_weight_backing_bytes: backings.bytes.remove("fixed_weights").unwrap_or(0),
            fixed_weight_arrays_not_ready,
            state_backing_bytes: backings.bytes,
            ple_host_row_payload_bytes,
            mlx_active_other_bytes: memory.mlx_active_bytes.checked_sub(owned),
            memory,
        })
    }

    pub fn runtime_manifest(&self) -> Result<Value> {
        Ok(serde_json::json!({
            "target_policy":self.layers[0].moe.policy,"lookahead_depth":self.lookahead_depth,
            "multi_token":self.multi_token,"prefill_heads":self.prefill_heads,
            "memory":self.memory_ledger(None)?,
        }))
    }

    /// Called once before a new AR prompt is replayed. Never resize in a verification transaction.
    pub fn admit_request(&mut self, positions: usize) -> Result<Option<ResourcePlan>> {
        let Some(limit) = self.request_limit else {
            return Ok(None);
        };
        ensure!(
            positions <= self.assets.context as usize,
            "request exceeds Flash context"
        );
        let plan =
            ResourcePlan::ar(u32::try_from(positions)?).context("invalid Flash request length")?;
        ensure!(
            self.verification.is_none(),
            "cannot resize during Flash verification"
        );
        #[cfg(feature = "chat")]
        ensure!(
            self.dflash.is_none(),
            "Flash request memory budget supports AR only"
        );
        // Flash replays the prompt for each new AR turn. No snapshot is visible to another caller.
        self.drain()?;
        if let Some(pool) = &self.lookahead_pool {
            pool.borrow_mut().wait_for_idle()?;
        }
        self.reset();
        array::synchronize()?;
        array::clear_cache()?;
        let extra = plan.extra_bytes();
        let mut base = observed(array::memory_stats(false)?)?;
        let reclaimable: u64 = self
            .layers
            .iter()
            .map(|l| (l.moe.capacity() - 10) as u64 * flash_support::EXPERT_BYTES)
            .sum();
        ensure!(
            fits(base.saturating_sub(reclaimable), extra, limit),
            "Flash request exceeds memory budget at minimum cache"
        );
        for layer in &mut self.layers {
            if fits(base, extra, limit) {
                break;
            }
            let slots = shrink_slots(
                layer.moe.capacity(),
                base.saturating_add(extra).saturating_sub(limit),
            );
            if slots == layer.moe.capacity() {
                continue;
            }
            // Both old and new banks are live while allocating a replacement.
            ensure!(
                fits(base, slots as u64 * flash_support::EXPERT_BYTES, limit),
                "Flash memory budget has no headroom to replace expert banks"
            );
            layer.moe.shrink(slots)?;
            array::clear_cache()?;
            base = observed(array::memory_stats(false)?)?;
        }
        ensure!(
            fits(base, extra, limit),
            "Flash request exceeds measured memory budget"
        );
        Ok(Some(plan))
    }
}

#[cfg(all(test, feature = "chat"))]
mod tests {
    use super::*;

    #[test]
    #[ignore = "physical Apple GPU and Flash checkpoint; admission, backing aliases, reset and reuse"]
    fn flash_request_budget_and_live_backings() -> Result<()> {
        let path = PathBuf::from(std::env::var("MLXL3_FLASH_MODEL")?);
        let checkpoint = crate::checkpoint::inspect(&path)?;
        let mut target = FlashNext::from_checkpoint(&checkpoint)?;
        let ledger = target.memory_ledger(None)?;
        assert_eq!(ledger.expert_slots_by_layer.len(), 48);
        assert!(ledger.fixed_weight_backing_bytes > 0);
        assert!(ledger.expert_bank_backing_bytes >= ledger.expert_bank_payload_bytes);
        let before_capacity = ledger.expert_slots_by_layer;
        let token = 1234;
        let expected = target.forward(token)?.to_bytes()?;
        assert!(!expected.is_empty());
        let occupied = target.memory_ledger(None)?;
        assert!(occupied.state_backing_bytes["gdn"] >= 36 * 48 * 128 * 128 * 4);
        assert!(occupied.state_backing_bytes["qsa_kv"] >= 24576);
        let state = target.layers[3].snapshot()?;
        let mut backings = Backings::default();
        let Attention::Qsa(q) = &target.layers[3].attention else {
            unreachable!()
        };
        backings.qsa(&q.state, false)?;
        let old = backings.bytes.clone();
        let AttentionState::Qsa(snapshot) = &state.attention else {
            unreachable!()
        };
        backings.qsa(snapshot, true)?;
        assert_eq!(backings.bytes, old);
        drop(state);

        target.request_limit = Some(1);
        assert!(target.admit_request(17).is_err());
        assert_eq!(
            target.memory_ledger(None)?.expert_slots_by_layer,
            before_capacity
        );
        assert_eq!(target.offset(), 0);
        assert!(target.last_hidden.is_none());
        assert!(FlashMtpHead::load(&checkpoint, &target).is_err());
        assert!(target.configure_dflash2(Some(&path), 1).is_err());
        target.request_limit = Some(u64::MAX);
        assert!(target.admit_request(17)?.is_some());
        assert_eq!(target.forward(token)?.to_bytes()?, expected);
        let _ = target.verify_pair([token, token])?;
        assert!(target.admit_request(17).is_err());
        target.reset();
        array::synchronize()?;
        array::clear_cache()?;
        let base = observed(array::memory_stats(false)?)?;
        let extra = ResourcePlan::ar(17).unwrap().extra_bytes();
        target.request_limit = Some(base + extra - (24 << 20));
        assert!(target.admit_request(17)?.is_some());
        let shrunk = target.memory_ledger(None)?;
        assert!(shrunk.expert_slots_by_layer.iter().sum::<usize>() < before_capacity.iter().sum());
        assert!(
            shrunk
                .expert_slots_by_layer
                .iter()
                .all(|slots| *slots >= 10)
        );
        assert_eq!(target.forward(token)?.to_bytes()?, expected);
        target.request_limit = None;
        let head = FlashMtpHead::load(&checkpoint, &target)?;
        let ledger = target.memory_ledger(Some(&head))?;
        assert!(ledger.state_backing_bytes["mtp_weights"] > 0);
        assert!(
            ledger.state_backing_bytes["mtp_expert_banks"] >= 512 * flash_support::EXPERT_BYTES
        );
        println!(
            "FLASH_MEMORY original={before_capacity:?} after={:?} base={base} extra={extra}",
            shrunk.expert_slots_by_layer
        );
        Ok(())
    }
}
