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
        let adaptive = flag("MLXL3_FLASH_DRAFT_ADAPTIVE")? && mode == 1;
        Ok(Self {
            q4: flag("MLXL3_FLASH_DRAFT_Q4")?,
            ring: flag("MLXL3_FLASH_DRAFT_RING")?,
            adaptive,
            width: if adaptive || mode == 3 {
                7
            } else if mode == 2 {
                2
            } else {
                3
            },
        })
    }
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
    }
    fn advance(&mut self, target: &mut FlashNext, anchor: u32, remaining: usize) -> Result<u32> {
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
            return Ok(target.forward(anchor)?.chat_greedy_ids()?[0]);
        }
        let started = std::time::Instant::now();
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
        let verification = target.verify_draft(&tokens, Kind::Dflash2)?;
        let ids = verification.parts.logits.chat_greedy_ids()?;
        let acceptance = crate::speculative::greedy_accept(&proposals, &ids)
            .context("invalid DFlash verification")?;
        let retained = acceptance.accepted_draft_tokens + 1;
        let captured = verification.parts.captured.slice(0, 0, retained as i32)?;
        verification.commit(retained)?;
        self.draft.append(&captured, start)?;
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
    fn invalid_modes_are_rejected_before_loading_a_checkpoint() {
        for mode in [0, 4, usize::MAX] {
            assert!(Options::read(mode).is_err());
        }
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
        for (q4, ring, adaptive) in [
            (false, false, false),
            (true, false, false),
            (true, true, false),
            (true, true, true),
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
                    width: if adaptive { 7 } else { 3 },
                },
                budget: VerifyBudget::default(),
                pending: VecDeque::new(),
                proposed: 0,
                accepted: 0,
                blocks: 0,
                seconds: 0.,
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
                    "DFlash tokens differ Q4={q4}/ring={ring}/adaptive={adaptive}/budget={budget}"
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
                records.push(serde_json::json!({"q4":q4,"ring":ring,"adaptive":adaptive,"budget":budget,"states":states.len(),"stats":target.dflash2_stats()}));
            }
            cancel.store(true, Ordering::Relaxed);
            ensure!(target.prefill_dflash2(&prompt, &cancel).is_err() && target.offset() == 0);
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
