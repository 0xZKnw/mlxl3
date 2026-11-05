use super::*;
use crate::flash_draft_budget::{PreDraftBudget, VerifyBudget};
use std::sync::atomic::AtomicBool;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Options {
    q4: bool,
    ring: bool,
    adaptive: bool,
    precap: bool,
    width: usize,
}
impl Options {
    fn read(mode: usize) -> Result<Self> {
        ensure!((1..=3).contains(&mode), "DFlash mode must be 1, 2 or 3");
        let flag = |name: &str, default: bool| -> Result<bool> {
            match std::env::var(name) {
                Ok(v) if v == "0" => Ok(false),
                Ok(v) if v == "1" => Ok(true),
                Err(std::env::VarError::NotPresent) => Ok(default),
                _ => anyhow::bail!("{name} must be 0 or 1"),
            }
        };
        let adaptive_requested = flag("MLXL3_FLASH_DRAFT_ADAPTIVE", true)?;
        let precap_requested = flag("MLXL3_FLASH_DRAFT_PRECAP", false)?;
        ensure!(
            !precap_requested || adaptive_requested,
            "MLXL3_FLASH_DRAFT_PRECAP requires MLXL3_FLASH_DRAFT_ADAPTIVE=1"
        );
        let adaptive = adaptive_requested && mode == 1;
        let precap = precap_requested && adaptive;
        let width = match std::env::var("MLXL3_FLASH_DRAFT_WIDTH") {
            Ok(value) => crate::flash_support::fixed_draft_width(&value, adaptive_requested)
                .context(
                    "MLXL3_FLASH_DRAFT_WIDTH requires 1..7 and MLXL3_FLASH_DRAFT_ADAPTIVE=0",
                )?,
            Err(std::env::VarError::NotPresent) => {
                if precap {
                    3
                } else if adaptive || mode == 3 {
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
            q4: flag("MLXL3_FLASH_DRAFT_Q4", true)?,
            ring: flag("MLXL3_FLASH_DRAFT_RING", true)?,
            adaptive,
            precap,
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
    pre_budget: PreDraftBudget,
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
        self.pre_budget = PreDraftBudget::default();
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
        let mut count = self
            .options
            .width
            .min(remaining - 1)
            .min((target.context - target.offset - 1) as usize);
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
            for part in tokens.chunks(self.prefill_chunk) {
                ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
                let start = self.offset;
                let (logits, captured) = if self.prefill_chunk == 1 {
                    self.forward_captured(part[0], Kind::Dflash2)?
                } else {
                    let (logits, captured) =
                        self.forward_prefill_block(part, &Kind::Dflash2.captures(), cancelled)?;
                    (logits, captured.context("missing prefill captures")?)
                };
                for row in 0..part.len() {
                    ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
                    session.draft.append(
                        &captured.slice(0, row as i32, row as i32 + 1)?,
                        start + row as i32,
                    )?;
                }
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
    #[ignore = "physical Apple GPU, target/draft; isolated exact prefill block qualification"]
    fn prefill_blocks_preserve_full_states_captures_draft_and_continuation() -> Result<()> {
        let c = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut target = FlashNext::from_checkpoint(&c)?;
        target.configure_dflash2(Some(Path::new(&std::env::var("MLXL3_FLASH_DRAFT")?)), 1)?;
        let lengths = std::env::var("MLXL3_PREFILL_TEST_LENGTHS")
            .unwrap_or_else(|_| "1,2,3,7,8,9,17".into())
            .split(',')
            .map(str::parse::<usize>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(
            !lengths.is_empty() && lengths.iter().all(|&n| (1..=512).contains(&n)),
            "invalid test lengths"
        );
        let cancel = AtomicBool::new(false);
        let saved = tempfile::tempdir()?;
        let mut cases = 0;
        for &length in &lengths {
            target.reset();
            target.prefill_chunk = 1;
            let tokens = std::iter::once(EOS)
                .chain((1..length).map(|i| i as u32))
                .collect::<Vec<_>>();
            let mut captures = Vec::new();
            let mut final_logits = None;
            for (index, &token) in tokens.iter().enumerate() {
                let (logits, captured) = target.forward_captured(token, Kind::Dflash2)?;
                target
                    .dflash
                    .as_mut()
                    .unwrap()
                    .draft
                    .append(&captured, index as i32)?;
                captures.push(captured);
                final_logits = Some(logits);
            }
            let logits = final_logits.context("empty reference logits")?;
            let values = logits.to_f32()?;
            ensure!(values.len() == 248320 && values.iter().all(|v| v.is_finite()));
            let expected_logits = logits.to_bytes()?;
            let captures = Array::concatenate(&captures.iter().collect::<Vec<_>>(), 0)?;
            ensure!(
                captures.shape() == [length as i32, 12800] && captures.dtype() == Dtype::BFloat16
            );
            let values = captures.astype(Dtype::Float32)?.to_f32()?;
            ensure!(values.len() == length * 12800 && values.iter().all(|v| v.is_finite()));
            let expected_captures = captures.to_bytes()?;
            ensure!(disk_state(&target, saved.path(), "prefill", true)? > 0);
            for (layer, pair) in target
                .dflash
                .as_ref()
                .unwrap()
                .draft
                .cache
                .iter()
                .enumerate()
            {
                let (k, v) = pair.as_ref().context("reference draft cache absent")?;
                for value in [k, v] {
                    ensure!(value.shape().len() == 4 && value.shape().iter().all(|&n| n > 0));
                    let values = value.to_f32()?;
                    ensure!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
                }
                std::fs::write(
                    saved.path().join(format!("meta{layer}")),
                    serde_json::to_vec(&(
                        k.shape(),
                        format!("{:?}", k.dtype()),
                        v.shape(),
                        format!("{:?}", v.dtype()),
                    ))?,
                )?;
                std::fs::write(saved.path().join(format!("k{layer}")), k.to_bytes()?)?;
                std::fs::write(saved.path().join(format!("v{layer}")), v.to_bytes()?)?;
            }
            let mut id = logits.chat_greedy_ids()?[0];
            let mut expected = Vec::new();
            for remaining in (1..=3).rev() {
                id = target.advance_dflash2(id, remaining)?;
                expected.push(id);
            }
            disk_state(&target, saved.path(), "continued", true)?;
            for chunk in [2, 4, 8, 32, 128] {
                for dflash in [false, true] {
                    target.reset();
                    target.prefill_chunk = chunk;
                    target.prefill_captures = dflash.then(Vec::new);
                    let logits = if dflash {
                        target.prefill_dflash2(&tokens, &cancel)?
                    } else {
                        target.prefill_tokens(&tokens, &cancel)?
                    };
                    ensure!(
                        logits.to_bytes()? == expected_logits,
                        "prefill logits differ length={length} chunk={chunk} draft={dflash}"
                    );
                    disk_state(&target, saved.path(), "prefill", false)?;
                    if dflash {
                        let captures = target
                            .prefill_captures
                            .take()
                            .context("missing capture trace")?;
                        ensure!(captures.len() == length.div_ceil(chunk));
                        ensure!(captures.iter().enumerate().all(|(i, c)| c.shape()
                            == [(length - i * chunk).min(chunk) as i32, 12800]
                            && c.dtype() == Dtype::BFloat16));
                        ensure!(
                            Array::concatenate(&captures.iter().collect::<Vec<_>>(), 0)?
                                .to_bytes()?
                                == expected_captures,
                            "prefill captures differ"
                        );
                        ensure!(target.dflash.as_ref().unwrap().draft.offset == length as i32);
                        for (layer, pair) in target
                            .dflash
                            .as_ref()
                            .unwrap()
                            .draft
                            .cache
                            .iter()
                            .enumerate()
                        {
                            let (k, v) = pair.as_ref().context("candidate draft cache absent")?;
                            ensure!(
                                serde_json::to_vec(&(
                                    k.shape(),
                                    format!("{:?}", k.dtype()),
                                    v.shape(),
                                    format!("{:?}", v.dtype())
                                ))? == std::fs::read(saved.path().join(format!("meta{layer}")))?
                            );
                            ensure!(
                                k.to_bytes()?
                                    == std::fs::read(saved.path().join(format!("k{layer}")))?
                            );
                            ensure!(
                                v.to_bytes()?
                                    == std::fs::read(saved.path().join(format!("v{layer}")))?
                            );
                        }
                    }
                    let mut id = logits.chat_greedy_ids()?[0];
                    for (remaining, &want) in (1..=3).rev().zip(&expected) {
                        id = if dflash {
                            target.advance_dflash2(id, remaining)?
                        } else {
                            target.forward(id)?.chat_greedy_ids()?[0]
                        };
                        ensure!(id == want, "continuation differs");
                    }
                    disk_state(&target, saved.path(), "continued", false)?;
                    cases += 1;
                    println!(
                        "PREFILL_CASE {}",
                        serde_json::json!({"length":length,"chunk":chunk,"dflash":dflash,"memory":array::memory_stats(false)?})
                    );
                }
            }
            println!("PREFILL_LENGTH length={length} cases={cases} parity=true");
        }
        target.reset();
        target.prefill_chunk = 128;
        for dflash in [false, true] {
            target.reset();
            disk_state(&target, saved.path(), "reset", true)?;
            target.prefill_cancel_after = Some(2);
            cancel.store(false, Ordering::Relaxed);
            let result = if dflash {
                target.prefill_dflash2(&[EOS; 129], &cancel)
            } else {
                target.prefill_tokens(&[EOS; 129], &cancel)
            };
            ensure!(
                result.is_err() && cancel.load(Ordering::Relaxed),
                "mid-block cancellation missed"
            );
            disk_state(&target, saved.path(), "reset", false)?;
            ensure!(
                target.dflash.as_ref().unwrap().draft.offset == 0
                    && target
                        .dflash
                        .as_ref()
                        .unwrap()
                        .draft
                        .cache
                        .iter()
                        .all(Option::is_none)
            );
            target.prefill_cancel_after = None;
            cancel.store(false, Ordering::Relaxed);
            let logits = if dflash {
                target.prefill_dflash2(&[EOS, 7], &cancel)?
            } else {
                target.prefill_tokens(&[EOS, 7], &cancel)?
            };
            ensure!(logits.shape() == [1, 248320] && target.offset == 2);
        }
        target.reset();
        target.prefill_chunk = 4;
        for invalid in [vec![], vec![EOS, 1, 2, 3, 4, 248320]] {
            ensure!(target.prefill_tokens(&invalid, &cancel).is_err() && target.offset == 0);
            ensure!(target.prefill_dflash2(&invalid, &cancel).is_err());
            ensure!(target.offset == 0 && target.dflash.as_ref().unwrap().draft.offset == 0);
        }
        ensure!(
            target
                .prefill_dflash2(&[EOS, 7], &AtomicBool::new(true))
                .is_err()
        );
        ensure!(
            target.offset == 0
                && target
                    .dflash
                    .as_ref()
                    .unwrap()
                    .draft
                    .cache
                    .iter()
                    .all(Option::is_none)
        );
        target.context = 4;
        ensure!(target.prefill_dflash2(&[EOS; 4], &cancel).is_err());
        ensure!(target.forward_tokens(&[EOS; 4])?.shape() == [1, 248320]);
        ensure!(target.forward_tokens(&[7]).is_err());
        ensure!(cases == lengths.len() * 10);
        println!(
            "PREFILL_BLOCK_PARITY lengths={lengths:?} cases={cases} capture_values_per_token=12800 logits=248320 states_draft_continuation=byte_exact errors=true"
        );
        Ok(())
    }

    #[cfg(kani)]
    #[kani::proof]
    fn actual_commit_total_prefill_checks_shapes_and_errors() {
        prefill_blocks_preserve_full_states_captures_draft_and_continuation().unwrap();
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
