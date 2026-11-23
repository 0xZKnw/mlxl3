//! Turn-loop policies of the Agent mode (AGENT-2026-11-23-BOUCLE): a per-step reasoning
//! budget closed by the model's own marker, context pruning of old tool exchanges, and a
//! stop for calls that repeat with the same result. Pure functions: the bridge loop in
//! `main.rs` applies them, the tests below pin them without a model.
use anyhow::{Result, ensure};
use serde_json::Value;
use std::hash::{Hash, Hasher};

/// Smallest and largest `reasoning_budget` a request may ask for (tokens per step).
pub const MIN_REASONING_BUDGET: usize = 256;
pub const MAX_REASONING_BUDGET: usize = 131_072;
/// Forced ends of thinking in one step; past them the step thinks freely to its end.
pub const FORCED_CLOSES_PER_STEP: usize = 2;
/// Identical calls with identical results in a row before the loop stops running tools.
pub const REPEATED_CALL_LIMIT: usize = 5;
/// Pruning starts above this share of the context and frees down to the target.
pub const PRUNE_TRIGGER_PERCENT: usize = 70;
pub const PRUNE_TARGET_PERCENT: usize = 45;
/// The most recent tool outputs, up to this share of the context, are never pruned.
pub const PRUNE_PROTECT_PERCENT: usize = 20;
/// Call arguments longer than this (characters, per string) may be shortened once old.
pub const LONG_ARGUMENT_CHARS: usize = 2_000;
/// Characters kept at the start of a shortened argument.
pub const KEPT_ARGUMENT_CHARS: usize = 500;
/// Characters kept at the start and at the end of a shortened tool output: the end holds a
/// command's result, its errors or a test's verdict.
pub const KEPT_OUTPUT_HEAD_CHARS: usize = 200;
pub const KEPT_OUTPUT_TAIL_CHARS: usize = 400;

const PRUNED_OUTPUT: &str = "[Earlier tool output removed to free context";
const SHORTENED_OUTPUT: &str = "characters of this earlier tool output removed to free context";
const PRUNED_ARGUMENT: &str = "[... ";

/// The requested per-step reasoning budget, validated; omitted means no budget.
pub fn reasoning_budget(requested: Option<usize>) -> Result<Option<usize>> {
    if let Some(budget) = requested {
        ensure!(
            (MIN_REASONING_BUDGET..=MAX_REASONING_BUDGET).contains(&budget),
            "reasoning_budget must be {MIN_REASONING_BUDGET} to {MAX_REASONING_BUDGET}"
        );
    }
    Ok(requested)
}

/// The budget a step gets: at most half of the room left, so that after a forced end of
/// thinking the model still has room to act.
pub fn step_reasoning_budget(budget: usize, available: usize) -> usize {
    budget.min(available / 2).max(1)
}

/// Words written for the model when its thinking reaches the budget (also shown as thinking):
/// the early stop of Qwen's thinking budget ("Considering the limited time by the user, I have
/// to give the solution based on the thinking directly now."), turned toward an action.
pub const FORCED_CLOSE_NOTE: &str = "\n\nConsidering the limited time, I have to act now based on my thinking so far: take the next concrete step, then adjust from its result.\n";

/// Text that ends a thinking block cut by its budget, in the model's own voice, followed
/// by the marker that closes this family's block (from `ThinkingSplitter::thinking_close`).
/// After Inkling 2's `<|end_message|>` the next message is opened so the model writes an
/// answer or a call instead of a second thinking message.
pub fn forced_close(close: &str) -> String {
    let note = FORCED_CLOSE_NOTE;
    match close {
        "<|end_message|>" => format!("{note}<|end_message|><|message_model|>"),
        "</think>" => format!("{note}</think>\n\n"),
        other => format!("{note}{other}"),
    }
}

/// What the model reads after a step ended with nothing: no visible text and no call.
pub const EMPTY_STEP_NUDGE: &str = "[You stopped without writing a reply. Continue the task: call the tool for the next step, or write your final answer.]";

/// Note added to the last result when identical calls keep returning the same thing.
pub fn repeated_call_note(count: usize) -> String {
    format!(
        "[This exact call returned the same result {count} times in a row, so tools are off for the rest of this turn: answer now from what you have.]"
    )
}

/// Counts consecutive calls that have the same tool, arguments and result.
#[derive(Default)]
pub struct RepeatGuard {
    last: Option<u64>,
    count: usize,
}

impl RepeatGuard {
    /// Records one executed call and returns how many identical ones ran in a row.
    pub fn observe(
        &mut self,
        name: &str,
        arguments: &Value,
        result: &str,
        is_error: bool,
    ) -> usize {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        (name, arguments.to_string(), result, is_error).hash(&mut hasher);
        let key = hasher.finish();
        if self.last == Some(key) {
            self.count += 1;
        } else {
            self.last = Some(key);
            self.count = 1;
        }
        self.count
    }

    /// True once the limit is reached.
    pub fn exhausted(&self) -> bool {
        self.count >= REPEATED_CALL_LIMIT
    }
}

/// What a pruning pass changed.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Pruned {
    /// Tool outputs shortened to their start and end, or replaced by a short notice
    /// (each output counted once per pass).
    pub outputs: usize,
    /// Call arguments shortened.
    pub arguments: usize,
    /// Estimated tokens freed (content tokens, template overhead not counted).
    pub freed: usize,
}

impl Pruned {
    pub fn is_empty(&self) -> bool {
        self.outputs == 0 && self.arguments == 0
    }
}

/// Frees context when the prompt holds more than [`PRUNE_TRIGGER_PERCENT`] of it, until the
/// estimate is at most [`PRUNE_TARGET_PERCENT`], in three levels applied oldest first:
/// (1) old tool outputs shortened to their start and end, (2) long arguments of old calls
/// shortened, (3) old outputs replaced by a short notice. The newest exchange (last call and
/// its results) and the most recent ones whose outputs and arguments fit in
/// [`PRUNE_PROTECT_PERCENT`] of the context are kept whole. System, user and assistant text
/// are never changed. `count` returns the token count of a text.
pub fn prune_context(
    dialogue: &mut [Value],
    prompt_tokens: usize,
    context_limit: usize,
    count: &mut dyn FnMut(&str) -> Result<usize>,
) -> Result<Pruned> {
    let mut pruned = Pruned::default();
    if prompt_tokens * 100 <= context_limit * PRUNE_TRIGGER_PERCENT {
        return Ok(pruned);
    }
    let target = context_limit * PRUNE_TARGET_PERCENT / 100;
    let protect = context_limit * PRUNE_PROTECT_PERCENT / 100;
    // Messages from `kept_from` on stay as they are: walking back from the end, the newest
    // exchange always, then older messages while their prunable tokens fit in `protect`.
    let mut kept_from = dialogue.len();
    let mut recent = 0usize;
    let mut newest_kept = false;
    for (index, message) in dialogue.iter().enumerate().rev() {
        let calls = message["role"] == "assistant" && message.get("tool_calls").is_some();
        let tokens = prunable_tokens(message, count)?;
        if newest_kept && recent + tokens > protect {
            break;
        }
        recent += tokens;
        kept_from = index;
        newest_kept |= calls;
    }
    let over = |pruned: &Pruned| prompt_tokens.saturating_sub(pruned.freed) > target;
    let mut touched = vec![false; kept_from];
    // (1) start and end of each old output
    for (index, message) in dialogue[..kept_from].iter_mut().enumerate() {
        if !over(&pruned) {
            return Ok(pruned);
        }
        if message["role"] != "tool" {
            continue;
        }
        let Some(text) = message["content"].as_str() else {
            continue;
        };
        if text.starts_with(PRUNED_OUTPUT) || text.contains(SHORTENED_OUTPUT) {
            continue;
        }
        let chars = text.chars().count();
        if chars <= KEPT_OUTPUT_HEAD_CHARS + KEPT_OUTPUT_TAIL_CHARS {
            continue;
        }
        let head: String = text.chars().take(KEPT_OUTPUT_HEAD_CHARS).collect();
        let tail: String = text.chars().skip(chars - KEPT_OUTPUT_TAIL_CHARS).collect();
        let shortened = format!(
            "{head}\n[... {} {SHORTENED_OUTPUT}; run the call again to see all of it ...]\n{tail}",
            chars - KEPT_OUTPUT_HEAD_CHARS - KEPT_OUTPUT_TAIL_CHARS
        );
        let before = count(text)?;
        let after = count(&shortened)?;
        if after >= before {
            continue;
        }
        message["content"] = Value::String(shortened);
        touched[index] = true;
        pruned.outputs += 1;
        pruned.freed += before - after;
    }
    for message in dialogue[..kept_from].iter_mut() {
        if !over(&pruned) {
            break;
        }
        if message["role"] != "assistant" {
            continue;
        }
        let Some(calls) = message.get_mut("tool_calls").and_then(Value::as_array_mut) else {
            continue;
        };
        for call in calls {
            let Some(arguments) = call
                .get_mut("function")
                .and_then(|function| function.get_mut("arguments"))
            else {
                continue;
            };
            let mut strings: Vec<&mut Value> = match arguments {
                Value::Object(map) => map.values_mut().filter(|v| v.is_string()).collect(),
                Value::String(_) => vec![arguments],
                _ => Vec::new(),
            };
            for value in strings.iter_mut() {
                let Some(text) = value.as_str() else { continue };
                let chars = text.chars().count();
                if chars <= LONG_ARGUMENT_CHARS {
                    continue;
                }
                let kept: String = text.chars().take(KEPT_ARGUMENT_CHARS).collect();
                let shortened = format!(
                    "{kept}\n{PRUNED_ARGUMENT}{} characters of this earlier call removed to free context ...]",
                    chars - KEPT_ARGUMENT_CHARS
                );
                let before = count(text)?;
                let after = count(&shortened)?;
                **value = Value::String(shortened);
                pruned.arguments += 1;
                pruned.freed += before.saturating_sub(after);
            }
        }
    }
    // (3) last resort: old outputs replaced by a notice
    for (index, message) in dialogue[..kept_from].iter_mut().enumerate() {
        if !over(&pruned) {
            break;
        }
        if message["role"] != "tool" {
            continue;
        }
        let Some(text) = message["content"].as_str() else {
            continue;
        };
        if text.starts_with(PRUNED_OUTPUT) {
            continue;
        }
        let before = count(text)?;
        let notice =
            format!("{PRUNED_OUTPUT} ({before} tokens). Run the call again if you still need it.]");
        let after = count(&notice)?;
        if after >= before {
            continue;
        }
        message["content"] = Value::String(notice);
        if !std::mem::replace(&mut touched[index], true) {
            pruned.outputs += 1;
        }
        pruned.freed += before - after;
    }
    Ok(pruned)
}

/// Tokens pruning could remove from a message: a tool output, or the string arguments of
/// an assistant's calls.
fn prunable_tokens(message: &Value, count: &mut dyn FnMut(&str) -> Result<usize>) -> Result<usize> {
    if message["role"] == "tool" {
        return message["content"].as_str().map_or(Ok(0), &mut *count);
    }
    let mut tokens = 0;
    for call in message["tool_calls"].as_array().into_iter().flatten() {
        match &call["function"]["arguments"] {
            Value::Object(map) => {
                for text in map.values().filter_map(Value::as_str) {
                    tokens += count(text)?;
                }
            }
            Value::String(text) => tokens += count(text)?,
            _ => {}
        }
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One token per four bytes, rounded up: independent of any model.
    fn bytes(text: &str) -> Result<usize> {
        Ok(text.len().div_ceil(4))
    }

    fn tool(text: &str) -> Value {
        json!({"role":"tool", "name":"javascript", "content":text})
    }

    fn call(code: &str) -> Value {
        json!({"role":"assistant", "content":"", "tool_calls":[
            {"type":"function", "function":{"name":"javascript", "arguments":{"code":code}}}
        ]})
    }

    #[test]
    fn reasoning_budget_is_optional_and_bounded() {
        assert_eq!(reasoning_budget(None).unwrap(), None);
        assert_eq!(reasoning_budget(Some(256)).unwrap(), Some(256));
        assert_eq!(reasoning_budget(Some(131_072)).unwrap(), Some(131_072));
        assert!(reasoning_budget(Some(255)).is_err());
        assert!(reasoning_budget(Some(131_073)).is_err());
        assert!(reasoning_budget(Some(0)).is_err());
        assert_eq!(step_reasoning_budget(8_192, 100_000), 8_192);
        assert_eq!(step_reasoning_budget(8_192, 6_000), 3_000);
        assert_eq!(step_reasoning_budget(8_192, 1), 1);
        assert_eq!(step_reasoning_budget(8_192, 0), 1);
    }

    #[test]
    fn forced_close_ends_with_each_family_marker() {
        let qwen = forced_close("</think>");
        assert!(qwen.starts_with("\n\nConsidering the limited time, I have to act now"));
        assert!(qwen.ends_with("adjust from its result.\n</think>\n\n"));
        assert!(forced_close("<channel|>").ends_with("result.\n<channel|>"));
        assert!(
            forced_close("<|end_message|>").ends_with("result.\n<|end_message|><|message_model|>")
        );
        // the note itself never contains a marker
        let note = qwen.trim_end_matches("</think>\n\n");
        assert!(!note.contains('<') && !note.contains('>'));
    }

    #[test]
    fn repeat_guard_counts_identical_calls_in_a_row() {
        let mut guard = RepeatGuard::default();
        let code = json!({"code":"run('ls')"});
        for expected in 1..REPEATED_CALL_LIMIT {
            assert_eq!(guard.observe("javascript", &code, "a\nb", false), expected);
            assert!(!guard.exhausted());
        }
        assert_eq!(
            guard.observe("javascript", &code, "a\nb", false),
            REPEATED_CALL_LIMIT
        );
        assert!(guard.exhausted());
        // any difference starts a new run: result, error flag, arguments, tool
        let mut guard = RepeatGuard::default();
        guard.observe("javascript", &code, "a", false);
        assert_eq!(guard.observe("javascript", &code, "b", false), 1);
        assert_eq!(guard.observe("javascript", &code, "b", true), 1);
        assert_eq!(
            guard.observe("javascript", &json!({"code":"1"}), "b", true),
            1
        );
        assert_eq!(guard.observe("other", &json!({"code":"1"}), "b", true), 1);
        assert_eq!(guard.observe("other", &json!({"code":"1"}), "b", true), 2);
        assert!(repeated_call_note(5).contains("5 times in a row"));
    }

    #[test]
    fn prune_does_nothing_up_to_the_trigger_and_acts_just_above() {
        let dialogue = vec![
            call("a"),
            tool(&"1".repeat(4_000)),
            call("b"),
            tool("recent"),
        ];
        // exactly 70 % of the context: untouched
        let mut at = dialogue.clone();
        assert!(
            prune_context(&mut at, 700, 1_000, &mut bytes)
                .unwrap()
                .is_empty()
        );
        assert_eq!(at, dialogue);
        // one token more: the old output goes
        let mut above = dialogue.clone();
        let pruned = prune_context(&mut above, 701, 1_000, &mut bytes).unwrap();
        assert_eq!(pruned.outputs, 1);
        assert_ne!(above, dialogue);
    }

    /// The output `text` shortened by level 1: its start, a marker, its end.
    fn shortened(text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        let head: String = chars[..KEPT_OUTPUT_HEAD_CHARS].iter().collect();
        let tail: String = chars[chars.len() - KEPT_OUTPUT_TAIL_CHARS..]
            .iter()
            .collect();
        format!(
            "{head}\n[... {} characters of this earlier tool output removed to free context; run the call again to see all of it ...]\n{tail}",
            chars.len() - 600
        )
    }

    #[test]
    fn prune_shortens_old_outputs_oldest_first_and_keeps_the_recent_ones() {
        // context 10 000: trigger above 7 000, target 4 500, protected 2 000 tokens
        let system = json!({"role":"system", "content":"s".repeat(8_000)});
        let user = json!({"role":"user", "content":"u".repeat(8_000)});
        let first = format!("{}END: maple\n", "1".repeat(7_988));
        let mut dialogue = vec![
            system.clone(),
            user.clone(),
            call("a"),
            tool(&first), // 2 000 tokens
            call("b"),
            tool(&"2".repeat(8_000)), // 2 000 tokens: 1 000 + 1 + 750 + 1 + 2 000 > 2 000
            call("c"),
            tool(&"3".repeat(3_000)), // 750 tokens, protected
            call("d"),
            tool(&"4".repeat(4_000)), // 1 000 tokens, newest, protected
        ];
        let pruned = prune_context(&mut dialogue, 8_000, 10_000, &mut bytes).unwrap();
        // 3 500 to free: both old outputs keep their start and end (about 1 800 freed each)
        assert_eq!(pruned.outputs, 2);
        assert_eq!(pruned.arguments, 0);
        assert!(pruned.freed >= 3_500, "{pruned:?}");
        assert_eq!(dialogue[3]["content"], shortened(&first));
        assert!(
            dialogue[3]["content"]
                .as_str()
                .unwrap()
                .ends_with("END: maple\n")
        );
        assert_eq!(dialogue[5]["content"], shortened(&"2".repeat(8_000)));
        assert_eq!(dialogue[7]["content"], "3".repeat(3_000));
        assert_eq!(dialogue[9]["content"], "4".repeat(4_000));
        assert_eq!(dialogue[0], system);
        assert_eq!(dialogue[1], user);
        // the prompt measured again is under the trigger: nothing more to do
        let again = dialogue.clone();
        let remeasured = 8_000 - pruned.freed;
        let second = prune_context(&mut dialogue, remeasured, 10_000, &mut bytes).unwrap();
        assert!(second.is_empty());
        assert_eq!(dialogue, again);
    }

    #[test]
    fn prune_clears_outputs_only_when_shortening_is_not_enough() {
        let mut dialogue = vec![
            call("a"),
            tool(&"1".repeat(8_000)),
            call("b"),
            tool(&"2".repeat(8_000)),
            call("c"),
            tool(&"3".repeat(400)), // newest, kept
        ];
        // 9 900 of 10 000: shortening both frees about 3 650, the target needs 5 400
        let pruned = prune_context(&mut dialogue, 9_900, 10_000, &mut bytes).unwrap();
        assert_eq!(pruned.outputs, 2, "each output counted once");
        for index in [1, 3] {
            let text = dialogue[index]["content"].as_str().unwrap();
            assert!(
                text.starts_with("[Earlier tool output removed to free context ("),
                "{text}"
            );
        }
        assert_eq!(dialogue[5]["content"], "3".repeat(400));
        // an output shortened earlier is not shortened again, only cleared
        let mut once = vec![
            call("a"),
            tool(&shortened(&"1".repeat(8_000))),
            call("b"),
            tool(&"x".repeat(8_000)), // the newest exchange uses up the protected share
        ];
        let quiet = prune_context(&mut once, 7_100, 10_000, &mut bytes).unwrap();
        assert_eq!(quiet.outputs, 1);
        assert!(once[1]["content"].as_str().unwrap().starts_with("[Earlier"));
    }

    #[test]
    fn prune_stops_once_the_target_is_reached() {
        let mut dialogue = vec![
            call("a"),
            tool(&"1".repeat(16_000)), // 4 000 tokens: enough on its own
            call("b"),
            tool(&"2".repeat(16_000)),
            call("c"),
            tool(&"3".repeat(400)),
        ];
        let pruned = prune_context(&mut dialogue, 8_000, 10_000, &mut bytes).unwrap();
        assert_eq!(pruned.outputs, 1);
        assert_eq!(dialogue[1]["content"], shortened(&"1".repeat(16_000)));
        assert_eq!(dialogue[3]["content"], "2".repeat(16_000));
    }

    #[test]
    fn prune_always_keeps_the_newest_exchange_even_when_it_is_large() {
        let mut dialogue = vec![
            call("a"),
            tool(&"1".repeat(400)),
            call("b"),
            tool(&"2".repeat(36_000)), // 9 000 tokens, far beyond the protected share
        ];
        let pruned = prune_context(&mut dialogue, 9_500, 10_000, &mut bytes).unwrap();
        assert_eq!(dialogue[3]["content"], "2".repeat(36_000));
        assert_eq!(
            dialogue[2]["tool_calls"][0]["function"]["arguments"]["code"],
            "b"
        );
        assert_eq!(pruned.outputs, 1);
        // without any call the newest exchange is unknown: nothing is pruned
        let mut bare = vec![tool(&"1".repeat(16_000)), tool("x")];
        assert!(
            prune_context(&mut bare, 9_500, 10_000, &mut bytes)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn prune_shortens_long_arguments_of_old_calls_after_the_outputs() {
        let long = "w".repeat(6_000);
        let mut dialogue = vec![
            call(&long),
            tool("ok"),
            json!({"role":"assistant", "content":"", "tool_calls":[
                {"type":"function", "function":{"name":"javascript", "arguments":long.clone()}}
            ]}),
            tool("ok"),
            call(&long), // after the newest kept output's call: kept
            tool(&"n".repeat(400)),
        ];
        let pruned = prune_context(&mut dialogue, 9_000, 10_000, &mut bytes).unwrap();
        assert_eq!(pruned.outputs, 0, "short outputs are not worth a notice");
        assert_eq!(pruned.arguments, 2);
        let object = dialogue[0]["tool_calls"][0]["function"]["arguments"]["code"]
            .as_str()
            .unwrap();
        assert!(object.starts_with(&"w".repeat(500)));
        assert!(
            object.ends_with(
                "\n[... 5500 characters of this earlier call removed to free context ...]"
            )
        );
        let string = dialogue[2]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert_eq!(string, object);
        assert_eq!(
            dialogue[4]["tool_calls"][0]["function"]["arguments"]["code"],
            long
        );
        assert_eq!(dialogue[1]["content"], "ok");
    }

    #[test]
    fn prune_keeps_text_and_counts_multibyte_arguments_by_character() {
        let long = "é".repeat(2_500);
        let mut dialogue = vec![
            json!({"role":"assistant", "content":"é".repeat(9_000)}),
            call(&long),
            tool("ok"),
            call("x"),
            tool(&"n".repeat(8_000)), // 2 000 tokens: the protected share is used up
        ];
        prune_context(&mut dialogue, 9_000, 10_000, &mut bytes).unwrap();
        assert_eq!(dialogue[0]["content"], "é".repeat(9_000));
        let code = dialogue[1]["tool_calls"][0]["function"]["arguments"]["code"]
            .as_str()
            .unwrap();
        assert!(code.starts_with(&"é".repeat(500)));
        assert!(code.contains("2000 characters of this earlier call"));
    }
}
