use super::tests::{qsa_bytes, state_bytes, tensor_state_bytes};
use super::*;

// The previous bridge loop is the oracle; it always evaluates every target head.
fn reference(
    target: &mut FlashNext,
    head: &mut FlashMtpHead,
    tokens: &[u32],
    cancelled: &AtomicBool,
) -> Result<Array> {
    target.reset();
    head.reset();
    let mut logits = None;
    for &token in tokens {
        if cancelled.load(Ordering::Relaxed) {
            target.reset();
            head.reset();
            anyhow::bail!("generation cancelled");
        }
        if let Some(hidden) = &target.last_hidden {
            head.extend_cache(target, hidden, token)?;
        }
        logits = Some(target.forward(token)?);
    }
    logits.context("empty Flash MTP prefill")
}

fn snapshot(target: &FlashNext, head: &FlashMtpHead) -> Result<Vec<Vec<u8>>> {
    let mut result = state_bytes(target)?;
    let Attention::Qsa(qsa) = &head.layer.attention else {
        anyhow::bail!("invalid MTP attention");
    };
    result.extend(qsa_bytes(qsa)?);
    Ok(result)
}

fn continuation(
    target: &mut FlashNext,
    head: &mut FlashMtpHead,
    logits: &Array,
) -> Result<(Vec<u32>, [usize; 3])> {
    let mut session = FlashMtpSession::default();
    let mut token = logits.chat_greedy_ids()?[0];
    let context = target.offset() as usize + 3;
    let mut output = Vec::new();
    for remaining in (1..=3).rev() {
        token = session.advance(target, head, token, context, remaining)?;
        output.push(token);
    }
    ensure!(session.pending.is_empty());
    Ok((output, [session.proposed, session.accepted, session.blocks]))
}

#[test]
#[ignore = "physical Apple GPU and full Flash checkpoint; run alone"]
fn mtp_final_prefill_head_matches_old_bridge_states_and_errors() -> Result<()> {
    let c = crate::checkpoint::inspect(&PathBuf::from(std::env::var("MLXL3_FLASH_MODEL")?))?;
    let mut target = FlashNext::from_checkpoint(&c)?;
    let mut head = FlashMtpHead::load(&c, &target)?;
    let _stages = crate::flash_profile::StageRetention::new(false);
    let cancel = AtomicBool::new(false);
    for length in [1, 2, 7, 8, 9, 17] {
        let tokens: Vec<_> = std::iter::once(EOS).chain(1..length as u32).collect();
        let logits = reference(&mut target, &mut head, &tokens, &cancel)?;
        ensure!(logits.shape() == [1, 248320]);
        let expected_logits = tensor_state_bytes(&logits)?;
        let expected_state = snapshot(&target, &head)?;
        let expected_continuation = continuation(&mut target, &mut head, &logits)?;
        let continued_state = snapshot(&target, &head)?;
        for optimized in [false, true] {
            target.prefill_heads = optimized;
            crate::flash_profile::start(false)?;
            let result = target.prefill_mtp(&mut head, &tokens, &cancel);
            let events = serde_json::to_value(crate::flash_profile::finish()?)?;
            let count = |name: &str| {
                events
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|e| e["name"] == name)
                    .count()
            };
            ensure!(count("logits") == if optimized { 1 } else { length });
            ensure!(count("prefill_step") == length);
            ensure!(count("prefill_head_extend") == length - 1);
            let logits = result?;
            ensure!(logits.shape() == [1, 248320]);
            ensure!(
                tensor_state_bytes(&logits)? == expected_logits,
                "logits differ"
            );
            ensure!(
                snapshot(&target, &head)? == expected_state,
                "prefill state differs"
            );
            ensure!(target.offset() == length as i32 && head.offset() == length as i32 - 1);
            ensure!(continuation(&mut target, &mut head, &logits)? == expected_continuation);
            ensure!(
                snapshot(&target, &head)? == continued_state,
                "continued state differs"
            );
        }
        println!("MTP_HEAD_LENGTH length={length} logits_states_continuation_counts_exact=true");
    }
    for (tokens, cancelled, context) in [
        (vec![], false, 4096),
        (vec![248320], false, 4096),
        (vec![EOS, 248320], false, 4096),
        (vec![EOS, 7], true, 4096),
        (vec![EOS, 7, 8], false, 2),
    ] {
        target.context = context;
        let cancelled = AtomicBool::new(cancelled);
        let error = reference(&mut target, &mut head, &tokens, &cancelled).unwrap_err();
        let expected = snapshot(&target, &head)?;
        for optimized in [false, true] {
            target.prefill_heads = optimized;
            let actual = target
                .prefill_mtp(&mut head, &tokens, &cancelled)
                .unwrap_err();
            ensure!(actual.to_string() == error.to_string(), "error changed");
            ensure!(
                snapshot(&target, &head)? == expected,
                "failure state changed"
            );
            let logits = target.prefill_mtp(&mut head, &[EOS, 7], &cancel)?;
            ensure!(logits.shape() == [1, 248320]);
            ensure!(
                target.offset() == 2 && head.offset() == 1,
                "recovery failed"
            );
        }
    }
    println!("MTP_HEAD_PARITY lengths=6 settings=2 errors=5 recovery=true parity=true");
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn actual_mtp_prefill_contract() {
    let token: u32 = kani::any();
    let optimized: bool = kani::any();
    let cancelled: bool = kani::any();
    let c = crate::checkpoint::inspect(&PathBuf::from(std::env::var("MLXL3_FLASH_MODEL").unwrap()))
        .unwrap();
    let mut target = FlashNext::from_checkpoint(&c).unwrap();
    let mut head = FlashMtpHead::load(&c, &target).unwrap();
    target.prefill_heads = optimized;
    let result = target.prefill_mtp(&mut head, &[EOS, token], &AtomicBool::new(cancelled));
    if cancelled {
        assert!(result.is_err());
        assert_eq!(target.offset(), 0);
        assert_eq!(head.offset(), 0);
    } else if token < 248320 {
        assert_eq!(result.unwrap().shape(), [1, 248320]);
        assert_eq!(target.offset(), 2);
        assert_eq!(head.offset(), 1);
    } else {
        assert!(result.is_err());
        assert_eq!(target.offset(), 1);
        assert_eq!(head.offset(), 0);
    }
    kani::cover!(cancelled);
    kani::cover!(!cancelled && token == 7 && optimized);
    kani::cover!(!cancelled && token >= 248320);
}
