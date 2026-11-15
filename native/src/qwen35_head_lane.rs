//! Q36-HEAD-LANE-P24: production `verify_mtp` with the lm_head lane layout (H2 at
//! three rows, F3 at four) against the serial decode loop, flag on and off, all M.
use super::tests::state_bytes;
use super::*;
use crate::linear::with_q36_head_lane_test;
use crate::tokenizer::{ChatTokenizer, Message};
use std::{fs, path::PathBuf};

const MAX_ROWS: usize = 4;
const VOCAB: usize = 248_320;
const HIDDEN: usize = 2048;
const DECODE_WARMUP: usize = 64;

#[test]
#[ignore = "physical Apple M5 GPU, local 35B, MLXL3_VERIFY_FIXTURES and MLXL3_HEAD_LANE_REPORT; run alone"]
fn q36_head_lane_verify_matches_serial_for_every_width() -> Result<()> {
    let output = PathBuf::from(std::env::var("MLXL3_HEAD_LANE_REPORT")?);
    ensure!(!output.exists(), "preserve previous report");
    let mut report = serde_json::json!({"status":"running","parity":false,
        "scope":"Production verify_mtp, M=1..4, head lane flag 0 and 1, against the serial forward loop (logits, MTP hidden, 80 states)",
        "decode_warmup_tokens":DECODE_WARMUP,"checks":[]});
    let save = |value: &serde_json::Value| -> Result<()> {
        let temporary = output.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(value)?)?;
        fs::rename(temporary, &output)?;
        Ok(())
    };
    save(&report)?;
    let result: Result<()> = (|| {
        let path = PathBuf::from(std::env::var("MLXL3_QWEN_TEST_MODEL")?);
        let fixtures = PathBuf::from(std::env::var("MLXL3_VERIFY_FIXTURES")?);
        ensure!(crate::array::is_m5_gpu()?, "requires M5");
        let tokenizer = ChatTokenizer::load(&path)?;
        let mut model = Qwen35Moe::load(&path)?;
        ensure!(
            model.layers.len() == 40
                && model.vocab as usize == VOCAB
                && model.mtp_layout.hidden_size == HIDDEN as i32
                && model.mtp_pipeline,
            "requires qualified 35B with the production MTP pipeline"
        );
        let cases = [
            (
                "fr",
                "Explique en français, en trois points concis, pourquoi le ciel est bleu.".into(),
                27,
            ),
            (
                "code",
                fs::read_to_string(fixtures.join("code-prompt.txt"))?,
                43,
            ),
            (
                "long",
                fs::read_to_string(fixtures.join("long-prompt.txt"))?,
                1097,
            ),
        ];
        for (name, prompt, expected_length) in cases {
            let rendered = tokenizer.render(&[Message {
                role: "user".into(),
                content: prompt,
            }])?;
            let tokens = tokenizer.encode(&rendered)?;
            ensure!(tokens.len() == expected_length, "workload changed: {name}");
            model.reset();
            let mut anchor = model.forward_mtp(&tokens)?.0.chat_greedy_ids()?[0];
            for _ in 0..DECODE_WARMUP {
                anchor = model.forward(anchor)?.chat_greedy_ids()?[0];
            }
            let before = model.snapshot()?;
            let mut forced = Vec::new();
            let mut serial_logits = Vec::new();
            let mut serial_hidden = Vec::new();
            let mut serial_states = Vec::new();
            for _ in 0..MAX_ROWS {
                forced.push(anchor);
                let (logits, uses) = with_q36_head_lane_test(true, || model.forward(anchor));
                let logits = logits?;
                ensure!(uses == 0, "serial decode used the head lane path");
                let bits = logits.to_f16_bits()?;
                ensure!(
                    bits.len() == VOCAB && bits.iter().all(|&v| f16::from_bits(v).is_finite()),
                    "invalid serial logits"
                );
                serial_logits.extend(bits);
                let hidden = model.mtp_hidden()?.to_f16_bits()?;
                ensure!(
                    hidden.len() == HIDDEN && hidden.iter().all(|&v| f16::from_bits(v).is_finite()),
                    "invalid serial hidden"
                );
                serial_hidden.extend(hidden);
                let states = state_bytes(&model)?;
                ensure!(
                    states.len() == 80 && states.iter().all(|state| !state.is_empty()),
                    "invalid serial states"
                );
                serial_states.push(states);
                anchor = logits.chat_greedy_ids()?[0];
            }
            for rows in 1..=MAX_ROWS {
                let verified = &forced[..rows];
                for on in [false, true] {
                    model.restore(before.clone())?;
                    let (observed, uses) =
                        with_q36_head_lane_test(on, || model.verify_mtp(verified));
                    let (logits, raw) = observed?;
                    let logits_ok = logits.shape() == [1, rows as i32, VOCAB as i32]
                        && logits.to_f16_bits()? == serial_logits[..rows * VOCAB];
                    let hidden_ok = raw.shape() == [1, rows as i32, HIDDEN as i32]
                        && raw.to_f16_bits()? == serial_hidden[..rows * HIDDEN];
                    model.commit_dflash_verification(rows, rows)?;
                    let states_ok = state_bytes(&model)? == serial_states[rows - 1];
                    let expected = on && matches!(rows, 3 | 4);
                    report["checks"]
                        .as_array_mut()
                        .unwrap()
                        .push(serde_json::json!({
                        "prompt":name,"rows":rows,"flag":u8::from(on),"head_lane_uses":uses,
                        "logits":logits_ok,"hidden":hidden_ok,"states":states_ok}));
                    save(&report)?;
                    ensure!(
                        logits_ok && hidden_ok && states_ok,
                        "verify differs from serial: {name} M={rows} flag={on}"
                    );
                    ensure!(
                        (uses > 0) == expected,
                        "unexpected head lane activation {name} M={rows} flag={on}: {uses}"
                    );
                    eprintln!("q36 head lane verify {name} M={rows} flag={on} uses={uses} exact");
                }
            }
        }
        ensure!(
            report["checks"].as_array().unwrap().len() == 3 * MAX_ROWS * 2,
            "incomplete parity matrix"
        );
        report["parity"] = serde_json::json!(true);
        Ok(())
    })();
    match &result {
        Ok(()) => report["status"] = serde_json::json!("complete"),
        Err(error) => {
            report["status"] = serde_json::json!("failed");
            report["parity"] = serde_json::json!(false);
            report["error"] = serde_json::json!(format!("{error:#}"));
        }
    }
    save(&report)?;
    result
}
