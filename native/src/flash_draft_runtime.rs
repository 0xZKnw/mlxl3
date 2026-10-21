use super::*;
use crate::flash_draft_budget::VerifyBudget;
use std::sync::atomic::AtomicBool;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Options {
    q4: bool,
    ring: bool,
    adaptive: bool,
    width: usize,
}
impl Options {
    fn read(mode: usize) -> Result<Self> {
        ensure!((1..=3).contains(&mode), "DFlash mode must be 1, 2 or 3");
        let flag = |name: &str| -> Result<bool> {
            match std::env::var(name) {
                Ok(v) if v == "0" => Ok(false),
                Ok(v) if v == "1" => Ok(true),
                Err(std::env::VarError::NotPresent) => Ok(true),
                _ => anyhow::bail!("{name} must be 0 or 1"),
            }
        };
        let adaptive_requested = flag("MLXL3_FLASH_DRAFT_ADAPTIVE")?;
        let adaptive = adaptive_requested && mode == 1;
        let width = match std::env::var("MLXL3_FLASH_DRAFT_WIDTH") {
            Ok(value) => crate::flash_support::fixed_draft_width(&value, adaptive_requested)
                .context(
                    "MLXL3_FLASH_DRAFT_WIDTH requires 1..7 and MLXL3_FLASH_DRAFT_ADAPTIVE=0",
                )?,
            Err(std::env::VarError::NotPresent) => {
                if adaptive || mode == 3 {
                    7
                } else if mode == 2 {
                    2
                } else {
                    3
                }
            }
            Err(_) => anyhow::bail!("MLXL3_FLASH_DRAFT_WIDTH must be valid Unicode"),
        };
        Ok(Self {
            q4: flag("MLXL3_FLASH_DRAFT_Q4")?,
            ring: flag("MLXL3_FLASH_DRAFT_RING")?,
            adaptive,
            width,
        })
    }
}

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
    pending: VecDeque<u32>,
    proposed: usize,
    accepted: usize,
    blocks: usize,
    seconds: f64,
    profile: FlashDraftProfile,
}
impl Session {
    pub(in crate::qwen4) fn reset(&mut self) {
        self.draft.reset();
        self.budget = VerifyBudget::default();
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
            remaining > 0 && target.offset == self.draft.offset && target.offset < target.context,
            "DFlash target and cache positions differ"
        );
        let count = self
            .options
            .width
            .min(remaining - 1)
            .min((target.context - target.offset - 1) as usize);
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
        let started = std::time::Instant::now();
        #[cfg(test)]
        let draft_profile = crate::flash_profile::span("dflash_proposal_inclusive");
        let input = target.draft_input(anchor, Kind::Dflash2, count)?;
        let (hidden, _) = self.draft.forward(&input, count)?;
        let (mut proposals, confidence) =
            self.draft.select(&target.head, &hidden, count, anchor)?;
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
        self.profile.draft_seconds += draft_seconds;
        self.profile.verification_seconds += verification_seconds;
        self.profile.commit_cache_seconds += commit_cache_seconds;
        self.profile.verified_rows += selected + 1;
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
    pub fn prefill_dflash2(&mut self, tokens: &[u32], cancelled: &AtomicBool) -> Result<Array> {
        self.reset();
        let mut session = self
            .dflash
            .take()
            .context("DFlash draft is not configured")?;
        let result = (|| {
            ensure!(
                !tokens.is_empty() && tokens.len() < self.context as usize,
                "invalid DFlash prompt"
            );
            let mut last = None;
            for &token in tokens {
                ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
                let (logits, captured) = self.forward_captured(token, Kind::Dflash2)?;
                session.draft.append(&captured, self.offset - 1)?;
                last = Some(logits);
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
    #[ignore = "physical Apple GPU, pinned target and DFlash2 checkpoint; run alone"]
    fn splash_runtime_preserves_serial_tokens_and_states() -> Result<()> {
        let target_path = PathBuf::from(std::env::var("MLXL3_FLASH_MODEL")?);
        let draft_path = PathBuf::from(std::env::var("MLXL3_FLASH_DRAFT")?);
        let checkpoint = crate::checkpoint::inspect(&target_path)?;
        let mut target = FlashNext::from_checkpoint(&checkpoint)?;
        let prompt = prompt_with_thinking(&target_path, true)?;
        let cancel = AtomicBool::new(false);
        let root = tempfile::tempdir()?;
        let mut oracles = Vec::new();
        for budget in [1, 2, 3, 17, 64] {
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
        for (q4, ring, adaptive, width) in [
            (false, false, false, 3),
            (true, false, false, 3),
            (true, true, false, 3),
            (true, true, true, 7),
            (true, true, false, 1),
            (true, true, false, 7),
        ] {
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
                    width,
                },
                budget: VerifyBudget::default(),
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
                records.push(serde_json::json!({"profile":profile,"q4":q4,"ring":ring,"adaptive":adaptive,"width":width,"budget":budget,"states":states.len(),"stats":target.dflash2_stats()}));
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
