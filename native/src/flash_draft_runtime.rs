use super::*;
use crate::flash_draft_budget::{PreDraftBudget, VerifyBudget};
use std::sync::atomic::AtomicBool;

use crate::flash_draft_options::Options;

#[cfg(kani)]
#[kani::proof]
fn fixed_width_options_reach_the_real_reader() {
    let result = Options::read(1);
    kani::cover!(result.is_ok());
}

pub(in crate::qwen4) struct Session {
    draft: Draft,
    path: PathBuf,
    identity: [(u64, std::time::SystemTime); 2],
    options: Options,
    budget: VerifyBudget,
    pre_budget: PreDraftBudget,
    pending: VecDeque<u32>,
    proposed: usize,
    accepted: usize,
    blocks: usize,
    seconds: f64,
    profile: FlashDraftProfile,
}
impl Session {
    pub(in crate::qwen4) fn record_memory(
        &self,
        ledger: &mut crate::qwen4::memory::Backings,
    ) -> Result<()> {
        for array in self.draft.weights.values() {
            ledger.record("draft_weights", array)?;
        }
        for (w, s, b) in self.draft.quantized.values() {
            for array in [w, s, b] {
                ledger.record("draft_weights", array)?;
            }
        }
        for (k, v) in self.draft.cache.iter().flatten() {
            ledger.record("draft_kv", k)?;
            ledger.record("draft_kv", v)?;
        }
        Ok(())
    }

    pub(in crate::qwen4) fn reset(&mut self) {
        self.draft.reset();
        crate::flash_draft_budget::reset_request_budgets(&mut self.budget, &mut self.pre_budget);
        self.pending.clear();
        self.proposed = 0;
        self.accepted = 0;
        self.blocks = 0;
        self.seconds = 0.;
        self.profile = FlashDraftProfile::default();
    }
    fn advance(&mut self, target: &mut FlashNext, anchor: u32, remaining: usize) -> Result<u32> {
        #[cfg(test)]
        let _profile = crate::flash_profile::span("dflash_advance");
        ensure!(
            anchor < 248320 && remaining > 0,
            "invalid DFlash anchor or budget"
        );
        if let Some(token) = self.pending.pop_front() {
            return Ok(token);
        }
        ensure!(
            remaining > 0
                && target.offset == self.draft.offset
                && target.offset < target.assets.context,
            "DFlash target and cache positions differ"
        );
        let mut count = self
            .options
            .width
            .min(remaining - 1)
            .min((target.assets.context - target.offset - 1) as usize);
        if self.options.precap {
            count = self.pre_budget.choose(count);
        }
        if count == 0 {
            let before = target.io_stats().2;
            let id = target.forward(anchor)?.chat_greedy_ids()?[0];
            self.profile.expert_demand_bytes = flash_support::draft_demand_bytes(
                self.profile.expert_demand_bytes,
                before,
                target.io_stats().2,
            )
            .context("DFlash demand byte counter reversed or exhausted")?;
            return Ok(id);
        }
        let demand_before = target.io_stats().2;
        let prefetch_before = target.prefetch_stats();
        let started = std::time::Instant::now();
        #[cfg(test)]
        let draft_profile = crate::flash_profile::span("dflash_proposal_inclusive");
        let input = target.draft_input(anchor, Kind::Dflash2, count)?;
        ensure!(input.shape() == [count as i32 + 1, H]);
        let (hidden, _) = self.draft.forward(&input, count)?;
        let (mut proposals, confidence) =
            self.draft
                .select(&target.assets.head, &hidden, count, anchor)?;
        ensure!(
            proposals.len() == count && confidence.len() == count,
            "incomplete DFlash selection"
        );
        let selected = if self.options.adaptive {
            self.budget
                .choose(&confidence)
                .context("invalid DFlash confidence")?
        } else {
            count
        };
        proposals.truncate(selected);
        let tokens = std::iter::once(anchor)
            .chain(proposals.iter().copied())
            .collect::<Vec<_>>();
        let start = target.offset;
        let draft_seconds = started.elapsed().as_secs_f64();
        #[cfg(test)]
        drop(draft_profile);
        #[cfg(test)]
        let verification_profile = crate::flash_profile::span("dflash_verification_inclusive");
        let verifying = std::time::Instant::now();
        let verification = target.verify_draft(&tokens, Kind::Dflash2)?;
        let ids = verification.parts.logits.chat_greedy_ids()?;
        let verification_seconds = verifying.elapsed().as_secs_f64();
        #[cfg(test)]
        drop(verification_profile);
        let acceptance = crate::speculative::greedy_accept(&proposals, &ids)
            .context("invalid DFlash verification")?;
        let retained = acceptance.accepted_draft_tokens + 1;
        #[cfg(test)]
        let _commit_profile = crate::flash_profile::span("dflash_commit_cache_build");
        let committing = std::time::Instant::now();
        let captured = verification.parts.captured.slice(0, 0, retained as i32)?;
        verification.commit(retained)?;
        self.draft.append(&captured, start)?;
        let commit_cache_seconds = committing.elapsed().as_secs_f64();
        self.pending.extend(
            proposals[..acceptance.accepted_draft_tokens]
                .iter()
                .copied(),
        );
        self.pending.push_back(acceptance.target_token);
        let seconds = started.elapsed().as_secs_f64();
        if self.options.adaptive {
            self.budget
                .observe(
                    &confidence[..selected],
                    acceptance.accepted_draft_tokens,
                    seconds,
                )
                .context("invalid DFlash budget observation")?;
        }
        if self.options.precap {
            self.pre_budget
                .observe(
                    count,
                    retained,
                    u64::try_from(started.elapsed().as_nanos())?,
                )
                .context("invalid or exhausted pre-draft budget")?;
        }
        self.profile.drafted_rows += count + 1;
        self.profile.drafted_widths[count] += 1;
        self.profile.verified_widths[selected] += 1;
        self.profile.draft_seconds += draft_seconds;
        self.profile.verification_seconds += verification_seconds;
        self.profile.commit_cache_seconds += commit_cache_seconds;
        self.profile.verified_rows += selected + 1;
        for ((total, before), after) in self
            .profile
            .ssd_prefetch
            .iter_mut()
            .zip(prefetch_before)
            .zip(target.prefetch_stats())
        {
            *total = flash_support::draft_demand_bytes(*total, before, after)
                .context("DFlash prefetch counter reversed or exhausted")?;
        }
        self.profile.expert_demand_bytes = flash_support::draft_demand_bytes(
            self.profile.expert_demand_bytes,
            demand_before,
            target.io_stats().2,
        )
        .context("DFlash demand byte counter reversed or exhausted")?;
        self.proposed += selected;
        self.accepted += acceptance.accepted_draft_tokens;
        self.blocks += 1;
        self.seconds += seconds;
        self.pending
            .pop_front()
            .context("empty verified DFlash block")
    }
}
impl FlashNext {
    pub fn configure_dflash2(&mut self, path: Option<&Path>, mode: usize) -> Result<()> {
        let Some(path) = path else {
            self.dflash = None;
            return Ok(());
        };
        ensure!(
            self.request_limit.is_none(),
            "Flash request memory budget supports AR only"
        );
        let options = Options::read(mode)?;
        let path = path.canonicalize()?;
        let identify = |name: &str| -> Result<_> {
            let m = std::fs::metadata(path.join(name))?;
            Ok((m.len(), m.modified()?))
        };
        let identity = [identify("model.safetensors")?, identify("config.json")?];
        if self
            .dflash
            .as_ref()
            .is_some_and(|s| s.path == path && s.identity == identity && s.options == options)
        {
            return Ok(());
        }
        self.reset();
        self.dflash = None;
        array::synchronize()?;
        array::clear_cache()?;
        let draft = Draft::load_options(&path, Kind::Dflash2, options.q4, options.ring)?;
        self.dflash = Some(Box::new(Session {
            draft,
            path,
            identity,
            options,
            budget: VerifyBudget::default(),
            pre_budget: PreDraftBudget::default(),
            pending: VecDeque::new(),
            proposed: 0,
            accepted: 0,
            blocks: 0,
            seconds: 0.,
            profile: FlashDraftProfile::default(),
        }));
        Ok(())
    }
    pub fn dflash2_active(&self) -> bool {
        self.dflash.is_some()
    }
    pub fn dflash2_proposals(&self) -> Option<usize> {
        self.dflash.as_ref().map(|s| s.options.width)
    }
    pub fn dflash2_stats(&self) -> Option<(usize, usize, usize, f64)> {
        self.dflash
            .as_ref()
            .map(|s| (s.proposed, s.accepted, s.blocks, s.seconds))
    }
    pub fn dflash2_profile(&self) -> Option<FlashDraftProfile> {
        self.dflash.as_ref().map(|s| s.profile)
    }
    #[cfg(test)]
    pub(in crate::qwen4) fn dflash2_state_for_profile(&self) -> Result<Vec<Vec<u8>>> {
        let session = self.dflash.as_ref().context("missing profiled draft")?;
        ensure!(
            session.draft.cache.len() == 5 && session.pending.is_empty(),
            "incomplete final draft state"
        );
        let mut state = vec![serde_json::to_vec(&(
            session.draft.offset,
            session.draft.circular,
            &session.pending,
            session.options.q4,
            session.options.ring,
            session.options.adaptive,
            session.options.precap,
            session.options.width,
        ))?];
        for pair in &session.draft.cache {
            let (key, value) = pair.as_ref().context("missing draft KV")?;
            for tensor in [key, value] {
                state.push(crate::qwen4::tests::tensor_state_bytes(tensor)?);
            }
        }
        Ok(state)
    }
    pub fn prefill_dflash2(&mut self, tokens: &[u32], cancelled: &AtomicBool) -> Result<Array> {
        self.reset();
        let mut session = self
            .dflash
            .take()
            .context("DFlash draft is not configured")?;
        let result = (|| {
            ensure!(
                !tokens.is_empty() && tokens.len() < self.assets.context as usize,
                "invalid DFlash prompt"
            );
            let mut last = None;
            for (index, &token) in tokens.iter().enumerate() {
                ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
                let (logits, captured) = if self.prefill_heads && index + 1 < tokens.len() {
                    self.forward_captured_optional(token, Kind::Dflash2, false)?
                } else {
                    let (logits, captured) = self.forward_captured(token, Kind::Dflash2)?;
                    (Some(logits), captured)
                };
                session.draft.append(&captured, self.offset - 1)?;
                last = logits;
            }
            last.context("empty DFlash prompt")
        })();
        if result.is_err() {
            self.reset();
            session.reset();
        }
        self.dflash = Some(session);
        result
    }
    pub fn advance_dflash2(&mut self, anchor: u32, remaining: usize) -> Result<u32> {
        let mut session = self
            .dflash
            .take()
            .context("DFlash draft is not configured")?;
        let result = session.advance(self, anchor, remaining);
        if result.is_err() {
            self.reset();
            session.reset();
        }
        self.dflash = Some(session);
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "physical Apple GPU and Flash/DFlash checkpoints; isolated serial prefill qualification"]
    fn final_prefill_head_preserves_states_draft_and_continuation() -> Result<()> {
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut target = FlashNext::from_checkpoint(&c)?;
        target.configure_dflash2(Some(Path::new(&std::env::var("MLXL3_FLASH_DRAFT")?)), 1)?;
        let lengths = std::env::var("MLXL3_PREFILL_TEST_LENGTHS")
            .unwrap_or_else(|_| "1,2,7,8,9,17".into())
            .split(',')
            .map(str::parse::<usize>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(!lengths.is_empty() && lengths.iter().all(|&n| (1..=512).contains(&n)));
        let cancel = AtomicBool::new(false);
        let saved = tempfile::tempdir()?;
        let mut cases = 0;
        for &length in &lengths {
            let tokens = std::iter::once(EOS)
                .chain((1..length).map(|i| i as u32))
                .collect::<Vec<_>>();
            for dflash in [false, true] {
                let mut expected_logits = Vec::new();
                let mut expected_ids = Vec::new();
                for optimized in [false, true] {
                    target.reset();
                    target.prefill_heads = optimized;
                    if length <= 17 {
                        crate::flash_profile::start(false)?;
                    }
                    let result = if dflash {
                        target.prefill_dflash2(&tokens, &cancel)
                    } else {
                        target.forward_tokens(&tokens)
                    };
                    if length <= 17 {
                        let trace = serde_json::to_value(crate::flash_profile::finish()?)?;
                        let heads = trace
                            .as_array()
                            .unwrap()
                            .iter()
                            .filter(|e| e["name"] == "logits")
                            .count();
                        ensure!(
                            heads == if optimized { 1 } else { length },
                            "unused heads executed"
                        );
                    }
                    let logits = result?;
                    let values = logits.to_f32()?;
                    ensure!(values.len() == 248320 && values.iter().all(|v| v.is_finite()));
                    if optimized {
                        ensure!(logits.to_bytes()? == expected_logits, "final logits differ");
                    } else {
                        expected_logits = logits.to_bytes()?;
                    }
                    ensure!(disk_state(&target, saved.path(), "prefill", !optimized)? > 0);
                    if dflash {
                        let draft = &target.dflash.as_ref().unwrap().draft;
                        ensure!(draft.offset == length as i32 && draft.cache.len() == 5);
                        for (index, pair) in draft.cache.iter().enumerate() {
                            let (k, v) = pair.as_ref().context("missing draft KV")?;
                            for (part, value) in [k, v].into_iter().enumerate() {
                                ensure!(
                                    value.shape().len() == 4
                                        && value.shape().iter().all(|&n| n > 0)
                                );
                                let values = value.to_f32()?;
                                ensure!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
                                let metadata = serde_json::to_vec(&(
                                    value.shape(),
                                    format!("{:?}", value.dtype()),
                                ))?;
                                for (suffix, bytes) in
                                    [("meta", metadata), ("data", value.to_bytes()?)]
                                {
                                    let file =
                                        saved.path().join(format!("kv-{index}-{part}-{suffix}"));
                                    if optimized {
                                        ensure!(std::fs::read(file)? == bytes, "draft KV differs");
                                    } else {
                                        std::fs::write(file, bytes)?;
                                    }
                                }
                            }
                        }
                    }
                    let mut id = logits.chat_greedy_ids()?[0];
                    for (index, remaining) in (1..=3).rev().enumerate() {
                        id = if dflash {
                            target.advance_dflash2(id, remaining)?
                        } else {
                            target.forward(id)?.chat_greedy_ids()?[0]
                        };
                        if optimized {
                            ensure!(id == expected_ids[index], "continuation differs");
                        } else {
                            expected_ids.push(id);
                        }
                    }
                    disk_state(&target, saved.path(), "continued", !optimized)?;
                }
                cases += 1;
            }
            println!("FINAL_HEAD_LENGTH length={length} cases={cases} parity=true");
        }
        // Check the captured values themselves, independently of draft cache equality.
        let mut captures = Vec::new();
        for head in [true, false] {
            target.reset();
            for index in 0..17 {
                let (logits, captured) =
                    target.forward_captured_optional(index, Kind::Dflash2, head)?;
                ensure!(logits.is_some() == head && captured.shape() == [1, 12800]);
                let values = captured.astype(Dtype::Float32)?.to_f32()?;
                ensure!(values.len() == 12800 && values.iter().all(|v| v.is_finite()));
                if head {
                    captures.push(captured.to_bytes()?);
                } else {
                    ensure!(
                        captured.to_bytes()? == captures[index as usize],
                        "captures differ"
                    );
                }
            }
            disk_state(&target, saved.path(), "captures", head)?;
        }
        target.reset();
        for invalid in [vec![], vec![EOS, 248320]] {
            ensure!(target.forward_tokens(&invalid).is_err() && target.offset == 0);
            ensure!(target.prefill_dflash2(&invalid, &cancel).is_err() && target.offset == 0);
            ensure!(
                target
                    .dflash
                    .as_ref()
                    .unwrap()
                    .draft
                    .cache
                    .iter()
                    .all(Option::is_none)
            );
        }
        ensure!(
            target
                .prefill_dflash2(&[EOS, 7], &AtomicBool::new(true))
                .is_err()
        );
        ensure!(target.offset == 0);
        ensure!(target.prefill_dflash2(&[EOS, 7], &cancel)?.shape() == [1, 248320]);
        target.reset();
        target.assets.context = 4;
        ensure!(target.prefill_dflash2(&[EOS; 4], &cancel).is_err());
        ensure!(target.forward_tokens(&[EOS; 4])?.shape() == [1, 248320]);
        ensure!(target.forward_tokens(&[7]).is_err());
        ensure!(cases == lengths.len() * 2);
        println!(
            "FINAL_HEAD_PARITY lengths={lengths:?} cases={cases} full_logits_states_captures_draft_continuation=true head_count=true errors=true"
        );
        Ok(())
    }

    #[cfg(kani)]
    #[kani::proof]
    fn actual_final_prefill_head_contract() {
        let token: u32 = kani::any();
        let heads: bool = kani::any();
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL").unwrap()))
            .unwrap();
        let mut target = FlashNext::from_checkpoint(&c).unwrap();
        target.prefill_heads = heads;
        let result = target.forward_tokens(&[EOS, token]);
        if token < 248320 {
            assert_eq!(result.unwrap().shape(), [1, 248320]);
            assert_eq!(target.offset, 2);
        } else {
            assert!(result.is_err());
            assert_eq!(target.offset, 0);
        }
        kani::cover!(token == 7 && heads);
        kani::cover!(token == EOS && !heads);
        kani::cover!(token >= 248320);
    }

    #[test]
    #[ignore = "physical Apple GPU and Flash/DFlash checkpoints; real terminal fallback; run alone"]
    fn runtime_profile_counts_terminal_target_reads() -> Result<()> {
        let checkpoint =
            crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut target = FlashNext::from_checkpoint(&checkpoint)?;
        let draft = PathBuf::from(std::env::var("MLXL3_FLASH_DRAFT")?);
        target.configure_dflash2(Some(&draft), 1)?;
        target.prefill_dflash2(&[EOS], &AtomicBool::new(false))?;
        let before = target.io_stats().2;
        let id = target.advance_dflash2(7, 1)?;
        let bytes = target
            .io_stats()
            .2
            .checked_sub(before)
            .context("counter reversed")?;
        let profile = target
            .dflash2_profile()
            .context("missing runtime profile")?;
        ensure!(
            bytes > 0 && bytes % flash_support::EXPERT_BYTES == 0,
            "vacuous read fixture"
        );
        ensure!(profile.expert_demand_bytes == bytes && profile.verified_rows == 0);
        ensure!(target.dflash2_stats() == Some((0, 0, 0, 0.)));
        ensure!(id < 248320 && target.offset() == 2);
        target.reset();
        ensure!(target.dflash2_profile().unwrap().expert_demand_bytes == 0);
        println!("FLASH_DRAFT_FALLBACK_PROFILE bytes={bytes} parity=true");
        Ok(())
    }
    #[test]
    fn invalid_modes_are_rejected_before_loading_a_checkpoint() {
        for mode in [0, 4, usize::MAX] {
            assert!(Options::read(mode).is_err());
        }
    }
    #[test]
    #[ignore = "isolated process environment; no checkpoint or GPU model"]
    fn fixed_width_environment_reaches_the_production_option_reader() -> Result<()> {
        let expected = std::env::var("MLXL3_EXPECTED_FIXED_WIDTH")?;
        let actual = Options::read(1);
        if expected == "error" {
            ensure!(actual.is_err(), "invalid fixed-width options accepted");
        } else {
            let actual = actual?;
            ensure!(actual.width == expected.parse::<usize>()? && !actual.adaptive);
        }
        Ok(())
    }
    #[test]
    #[ignore = "isolated environment; production pre-draft option parser"]
    fn pre_draft_environment_reaches_real_options() -> Result<()> {
        let expected = std::env::var("MLXL3_EXPECTED_PRECAP")?;
        let mode: usize = std::env::var("MLXL3_EXPECTED_MODE")?.parse()?;
        let result = Options::read(mode);
        if expected == "error" {
            ensure!(result.is_err());
        } else {
            let options = result?;
            ensure!(options.precap == (expected == "1"));
            if options.precap {
                ensure!(options.adaptive && options.width == 3 && mode == 1);
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU, pinned target and DFlash2 checkpoint; run alone"]
    fn splash_runtime_preserves_serial_tokens_and_states() -> Result<()> {
        check_runtime_parity(
            &[1, 2, 3, 8, 17, 64],
            &[
                (false, false, false, 3, false),
                (true, false, false, 3, false),
                (true, true, false, 3, false),
                (true, true, true, 7, false),
                (true, true, false, 1, false),
                (true, true, false, 7, false),
                (true, true, true, 3, true),
            ],
        )
    }

    #[test]
    #[ignore = "physical Apple GPU; 512 tokens, pre-draft adaptation and independent serial states"]
    fn pre_draft_long_generation_preserves_serial_tokens_and_states() -> Result<()> {
        check_runtime_parity(&[512], &[(true, true, true, 3, true)])
    }

    #[test]
    #[ignore = "physical Apple GPU, target/draft checkpoints, COLD_READY_ROWS=4 and tuned Hot; run alone"]
    fn wide_cold_runtime_preserves_serial_tokens_and_states() -> Result<()> {
        for (name, expected) in [
            ("MLXL3_SSD_COLD_READY_ROWS", "4"),
            ("MLXL3_SSD_GPU_HOT", "1"),
            ("MLXL3_SSD_GPU_HOT_WIDE", "1"),
            ("MLXL3_FLASH_MULTI_TOKEN", "1"),
        ] {
            ensure!(
                std::env::var(name).as_deref() == Ok(expected),
                "test requires {name}={expected}"
            );
        }
        // B2 is the unchanged control; budgets exercise partial and complete B3/B4.
        check_runtime_parity(
            &[1, 2, 3, 4, 8, 17],
            &[
                (true, true, false, 1, false),
                (true, true, false, 2, false),
                (true, true, false, 3, false),
            ],
        )
    }

    #[cfg(kani)]
    #[kani::proof]
    fn actual_pre_draft_runtime_preserves_serial_generation() {
        pre_draft_long_generation_preserves_serial_tokens_and_states().unwrap();
    }

    fn check_runtime_parity(
        budgets: &[usize],
        variants: &[(bool, bool, bool, usize, bool)],
    ) -> Result<()> {
        let target_path = PathBuf::from(std::env::var("MLXL3_FLASH_MODEL")?);
        let draft_path = PathBuf::from(std::env::var("MLXL3_FLASH_DRAFT")?);
        let checkpoint = crate::checkpoint::inspect(&target_path)?;
        let mut target = FlashNext::from_checkpoint(&checkpoint)?;
        let prompt = prompt_with_thinking(&target_path, true)?;
        let cancel = AtomicBool::new(false);
        let root = tempfile::tempdir()?;
        let mut oracles = Vec::new();
        for &budget in budgets {
            target.reset();
            let mut logits = target.forward_tokens(&prompt)?;
            let mut tokens = Vec::new();
            for index in 0..budget {
                let id = logits.chat_greedy_ids()?[0];
                tokens.push(id);
                if index + 1 < budget {
                    logits = target.forward(id)?;
                }
            }
            let mut states = crate::qwen4::tests::state_bytes(&target)?;
            ensure!(!states.is_empty() && states.iter().all(|s| !s.is_empty()));
            for (i, s) in states.drain(..).enumerate() {
                std::fs::write(root.path().join(format!("{budget}-{i}")), s)?;
            }
            oracles.push((budget, tokens));
        }
        let mut records = Vec::new();
        for &(q4, ring, adaptive, width, precap) in variants {
            target.reset();
            target.dflash = None;
            array::synchronize()?;
            array::clear_cache()?;
            let draft = Draft::load_options(&draft_path, Kind::Dflash2, q4, ring)?;
            if q4 {
                ensure!(draft.quantized.len() == 47 && draft.weights.len() > 2);
            }
            target.dflash = Some(Box::new(Session {
                draft,
                path: draft_path.clone(),
                identity: [(0, std::time::UNIX_EPOCH); 2],
                options: Options {
                    q4,
                    ring,
                    adaptive,
                    precap,
                    width,
                },
                budget: VerifyBudget::default(),
                pre_budget: PreDraftBudget::default(),
                pending: VecDeque::new(),
                proposed: 0,
                accepted: 0,
                blocks: 0,
                seconds: 0.,
                profile: FlashDraftProfile::default(),
            }));
            for (budget, expected) in &oracles {
                let logits = target.prefill_dflash2(&prompt, &cancel)?;
                let mut tokens = vec![logits.chat_greedy_ids()?[0]];
                while tokens.len() < *budget {
                    let next =
                        target.advance_dflash2(*tokens.last().unwrap(), budget - tokens.len())?;
                    tokens.push(next);
                }
                ensure!(
                    &tokens == expected,
                    "DFlash tokens differ Q4={q4}/ring={ring}/adaptive={adaptive}/width={width}/budget={budget}"
                );
                ensure!(
                    target.offset() as usize == prompt.len() + budget - 1,
                    "DFlash commit position differs"
                );
                let states = crate::qwen4::tests::state_bytes(&target)?;
                for (i, s) in states.iter().enumerate() {
                    ensure!(
                        *s == std::fs::read(root.path().join(format!("{budget}-{i}")))?,
                        "DFlash state {i} differs at {budget}"
                    );
                }
                let (proposed, accepted, blocks, seconds) = target.dflash2_stats().unwrap();
                let profile = target.dflash2_profile().unwrap();
                ensure!(profile.verified_rows == proposed + blocks);
                ensure!(profile.drafted_widths.iter().sum::<usize>() == blocks);
                ensure!(profile.verified_widths.iter().sum::<usize>() == blocks);
                ensure!(
                    profile.drafted_rows
                        == profile
                            .drafted_widths
                            .iter()
                            .enumerate()
                            .map(|(w, &n)| (w + 1) * n)
                            .sum::<usize>()
                );
                ensure!(
                    profile.verified_rows
                        == profile
                            .verified_widths
                            .iter()
                            .enumerate()
                            .map(|(w, &n)| (w + 1) * n)
                            .sum::<usize>()
                );
                if precap {
                    ensure!(profile.drafted_widths[4..].iter().all(|&n| n == 0));
                    if *budget >= 17 {
                        ensure!(
                            profile.drafted_widths[1..=3].iter().all(|&n| n > 0),
                            "pre-draft scheduler did not reach all three shapes"
                        );
                    }
                }
                ensure!(
                    [
                        profile.draft_seconds,
                        profile.verification_seconds,
                        profile.commit_cache_seconds
                    ]
                    .iter()
                    .all(|s| s.is_finite() && *s >= 0.)
                );
                ensure!(
                    profile.draft_seconds
                        + profile.verification_seconds
                        + profile.commit_cache_seconds
                        <= seconds + 1e-6
                );
                ensure!(
                    accepted <= proposed
                        && profile.expert_demand_bytes % flash_support::EXPERT_BYTES == 0
                );
                if blocks > 0 {
                    ensure!(
                        profile.verified_rows >= blocks * 2
                            && profile.verification_seconds > 0.
                            && profile.commit_cache_seconds > 0.
                    );
                }
                records.push(serde_json::json!({"profile":profile,"q4":q4,"ring":ring,"adaptive":adaptive,"width":width,"precap":precap,"budget":budget,"states":states.len(),"stats":target.dflash2_stats()}));
            }
            cancel.store(true, Ordering::Relaxed);
            ensure!(target.prefill_dflash2(&prompt, &cancel).is_err() && target.offset() == 0);
            ensure!(target.dflash2_profile().unwrap().verified_rows == 0);
            cancel.store(false, Ordering::Relaxed);
            ensure!(
                target
                    .prefill_dflash2(&prompt[..2], &cancel)?
                    .shape()
                    .last()
                    == Some(&248320)
            );
            ensure!(target.advance_dflash2(248320, 2).is_err() && target.offset() == 0);
        }
        target.configure_dflash2(None, 1)?;
        ensure!(!target.dflash2_active());
        println!(
            "FLASH_SPLASH_RUNTIME {}",
            serde_json::json!({"status":"passed","cases":records,"memory":array::memory_stats(false)?})
        );
        Ok(())
    }
}
