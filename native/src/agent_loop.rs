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
pub fn step_reasoning_budget(budget: usize, output_slots: usize) -> Option<usize> {
    (output_slots >= 2).then(|| budget.min(output_slots / 2))
}

#[cfg(kani)]
#[kani::proof]
fn reasoning_cut_reserves_answer_tokens() {
    let budget: usize = kani::any();
    let slots: usize = kani::any();
    if let Some(limit) = step_reasoning_budget(budget, slots) {
        assert!(limit <= budget);
        assert!(limit <= slots - limit);
        assert!(limit == budget || limit == slots / 2);
        kani::cover!(limit == budget && limit < slots / 2);
        kani::cover!(slots == 2 && limit == 1);
    } else {
        assert!(slots < 2);
        kani::cover!(slots == 1);
    }

    let thought: usize = kani::any();
    let separator: usize = kani::any();
    let output_left: usize = kani::any();
    let cut = cut_thinking_now(thought, budget, slots, separator, output_left);
    if cut {
        assert!(slots > separator && output_left > 0);
        assert!(thought >= budget || slots - separator == 1);
        kani::cover!(thought < budget && slots == 3 && separator == 2);
    } else if output_left > 0 && slots > separator {
        assert!(thought < budget && slots - separator > 1);
        kani::cover!(separator == 2 && slots == 4);
    }
}

/// A cut resumes the same assistant message. The model gets a private transition
/// in its own channel format; only text it actually generates is streamed.
pub fn reasoning_continuation(close: &str, summary: bool, tools: bool, compact: bool) -> String {
    let note = if compact {
        ""
    } else if summary {
        SUMMARY_CLOSE_NOTE
    } else if tools {
        "\n\nI will answer the request now, using a tool only if needed.\n"
    } else {
        "\n\nConsidering the limited time, I have to give my final answer in the requested format now.\n"
    };
    match close {
        "<|end_message|>" if summary || !tools => {
            format!("{note}<|end_message|><|message_model|><|content_text|>")
        }
        "<|end_message|>" => format!("{note}<|end_message|><|message_model|>"),
        "</think>" => format!("{note}</think>\n\n"),
        other => format!("{note}{other}"),
    }
}

/// Keep a context slot for at least one sampled answer token in the next round.
/// An injected separator consumes context but is not part of the output quota.
pub fn cut_thinking_now(
    thought: usize,
    limit: usize,
    context_left: usize,
    separator: usize,
    output_left: usize,
) -> bool {
    output_left > 0
        && context_left > separator
        && (thought >= limit || context_left - separator == 1)
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
        if text.starts_with(PRUNED_OUTPUT)
            || text.contains(SHORTENED_OUTPUT)
            || text.starts_with(SUMMARY_MARKER)
        {
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
        let before_arguments = pruned.arguments;
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
        // AGENT-2026-11-26-LIVE-CACHE: the step's own text still holds the whole arguments
        if pruned.arguments > before_arguments
            && let Some(written) = message.as_object_mut()
        {
            written.remove("raw");
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
        if text.starts_with(PRUNED_OUTPUT) || text.starts_with(SUMMARY_MARKER) {
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

/// AGENT-2026-11-24-DSH-PISTES, P3 (after DeepSeek Harness's `compaction-basic`, sized for
/// small windows): after pruning, a prompt still above this share of the context has the
/// older exchanges of the current turn summarized by the model.
pub const COMPACT_TRIGGER_PERCENT: usize = 75;
/// The newest exchanges kept word for word, up to this share of the context (the newest one
/// always).
pub const COMPACT_KEEP_PERCENT: usize = 20;
/// Tokens the summary may generate, thinking included.
pub const SUMMARY_MAX_TOKENS: usize = 3072;
/// Thinking allowed before the summary is written (then closed like a step's).
pub const SUMMARY_REASONING_BUDGET: usize = 1024;
/// Start of the text that carries a summary; pruning never shortens it.
pub const SUMMARY_MARKER: &str = "[Summary of the earlier steps of this task";
/// What the model is asked, after the exchanges to summarize (dsh's fixed sections, fewer).
pub const SUMMARY_REQUEST: &str = "[The context is almost full: your steps above will be removed. Write a summary of them that lets you finish the task without them. Do not call any tool. Use these headings, each with a few short bullets (\"none\" if empty): Progress; Files and code (paths, functions, what you changed, exact names); Commands and results (what you ran, statuses, key output); Errors and fixes; Remaining work; Next step. Keep exact paths, commands, identifiers, numbers and error messages. If an earlier summary is above, merge it instead of repeating it. Write only the summary.]";
/// Ends a summary's thinking cut by its budget.
pub const SUMMARY_CLOSE_NOTE: &str = "\n\nI have to write the summary now.\n";

/// The exchanges of the current turn (messages from `start`) to summarize: every exchange
/// (an assistant message and the results or notes that follow it) except the newest ones
/// that fit in [`COMPACT_KEEP_PERCENT`] of the context, the newest always kept. The cut is
/// always before an assistant message, so a call is never separated from its result, and
/// the kept part has a result or note to carry the summary. None: nothing to summarize.
pub fn compaction_range(
    dialogue: &[Value],
    start: usize,
    context_limit: usize,
    count: &mut dyn FnMut(&str) -> Result<usize>,
) -> Result<Option<std::ops::Range<usize>>> {
    let starts: Vec<usize> = (start..dialogue.len())
        .filter(|&index| dialogue[index]["role"] == "assistant")
        .collect();
    let (Some(&first), Some(&newest)) = (starts.first(), starts.last()) else {
        return Ok(None);
    };
    if first != start || starts.len() < 2 {
        return Ok(None);
    }
    let keep = context_limit * COMPACT_KEEP_PERCENT / 100;
    let mut cut = newest;
    let mut kept = 0;
    for message in &dialogue[newest..] {
        kept += message_tokens(message, count)?;
    }
    for pair in starts.windows(2).rev() {
        let (older, newer) = (pair[0], pair[1]);
        if newer != cut || older == first {
            break;
        }
        let mut tokens = 0;
        for message in &dialogue[older..newer] {
            tokens += message_tokens(message, count)?;
        }
        if kept + tokens > keep {
            break;
        }
        kept += tokens;
        cut = older;
    }
    let carried = dialogue[cut..]
        .iter()
        .any(|message| message["role"] != "assistant" && message["content"].is_string());
    Ok((cut > start && carried).then_some(start..cut))
}

/// The prompt of the summary step: the conversation up to the end of `range`, then the
/// request (a prefix of the current prompt, so the prompt cache serves it).
pub fn summary_dialogue(dialogue: &[Value], range: &std::ops::Range<usize>) -> Vec<Value> {
    let mut request = dialogue[..range.end].to_vec();
    request.push(serde_json::json!({"role":"user", "content":SUMMARY_REQUEST}));
    request
}

/// Replaces `range` by `summary`, written at the start of the first result or note after it
/// (no new message: every chat template accepts it, like the B5/B6 notes). False when no
/// kept message can carry it.
pub fn apply_summary(
    dialogue: &mut Vec<Value>,
    range: std::ops::Range<usize>,
    summary: &str,
) -> bool {
    let Some(carrier) = dialogue[range.end..]
        .iter()
        .position(|message| message["role"] != "assistant" && message["content"].is_string())
        .map(|offset| range.end + offset)
    else {
        return false;
    };
    let text = dialogue[carrier]["content"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    dialogue[carrier]["content"] = Value::String(format!(
        "{SUMMARY_MARKER}, written by you to free context:]\n{}\n[End of the summary. Result of the call above:]\n\n{text}",
        summary.trim()
    ));
    dialogue.drain(range);
    true
}

/// AGENT-2026-11-26-V21, A: start of the text that carries the summary of earlier turns, at
/// the head of the first user message kept.
pub const HISTORY_SUMMARY_MARKER: &str = "[Summary of the earlier conversation";
/// What the model is asked after the turns to summarize.
pub const HISTORY_SUMMARY_REQUEST: &str = "[The context is almost full: the conversation above will be removed. Write a summary of it that lets you continue without it. Do not call any tool. Use these headings, each with a few short bullets (\"none\" if empty): User requests and decisions; Work done (files, functions, exact names, what changed); Commands and results (what was run, statuses, key output); Errors and fixes; Open points and next steps. Keep exact paths, commands, identifiers, numbers and error messages. If an earlier summary is above, merge it instead of repeating it. Write only the summary.]";

/// True for the request that ends a summary step (of this turn's steps or of earlier turns).
pub fn is_summary_request(content: &Value) -> bool {
    content == SUMMARY_REQUEST || content == HISTORY_SUMMARY_REQUEST
}

/// Where earlier turns may be cut for a summary: the user messages with text among the
/// messages the client sent (`..end`), after the first message of the conversation. Newest
/// first. A cut is always before a user message, so a turn is summarized whole.
fn history_cuts(dialogue: &[Value], end: usize) -> (usize, Vec<usize>) {
    let first = dialogue
        .iter()
        .position(|message| message["role"] != "system")
        .unwrap_or(dialogue.len());
    let cuts = (first + 1..end.min(dialogue.len()))
        .rev()
        .filter(|&index| {
            dialogue[index]["role"] == "user" && dialogue[index]["content"].is_string()
        })
        .collect();
    (first, cuts)
}

/// The earlier turns to summarize when nothing of the current turn can be: from the first
/// message of the conversation up to a user message, keeping word for word the newest turns
/// that fit in [`COMPACT_KEEP_PERCENT`] of the context (the last user message always). `fits`
/// says whether the summary step of a candidate cut fits the context: an older cut is tried
/// until one does. None: a single turn, or no cut whose summary step fits.
pub fn history_range(
    dialogue: &[Value],
    end: usize,
    context_limit: usize,
    count: &mut dyn FnMut(&str) -> Result<usize>,
    fits: &mut dyn FnMut(usize) -> Result<bool>,
) -> Result<Option<std::ops::Range<usize>>> {
    let (first, cuts) = history_cuts(dialogue, end);
    let Some(&newest) = cuts.first() else {
        return Ok(None);
    };
    let keep = context_limit * COMPACT_KEEP_PERCENT / 100;
    let mut kept = 0;
    for message in &dialogue[newest..end.min(dialogue.len())] {
        kept += message_tokens(message, count)?;
    }
    let mut chosen = 0usize;
    for (position, pair) in cuts.windows(2).enumerate() {
        let (newer, older) = (pair[0], pair[1]);
        let mut tokens = 0;
        for message in &dialogue[older..newer] {
            tokens += message_tokens(message, count)?;
        }
        if kept + tokens > keep {
            break;
        }
        kept += tokens;
        chosen = position + 1;
    }
    // The preferred cut first, then older ones: each summarizes less and asks for less room.
    for &cut in &cuts[chosen..] {
        if fits(cut)? {
            return Ok(Some(first..cut));
        }
    }
    Ok(None)
}

/// The prompt of the summary step of earlier turns: the conversation up to the cut, then
/// the request (a prefix of the current prompt, so the prompt cache serves it).
pub fn history_summary_dialogue(dialogue: &[Value], cut: usize) -> Vec<Value> {
    let mut request = dialogue[..cut].to_vec();
    request.push(serde_json::json!({"role":"user", "content":HISTORY_SUMMARY_REQUEST}));
    request
}

/// Replaces `range` by `summary`, written at the head of the user message that follows it.
/// Returns that head exactly as written (the client stores it and sends it again), or None
/// when that message cannot carry it.
pub fn apply_history_summary(
    dialogue: &mut Vec<Value>,
    range: std::ops::Range<usize>,
    summary: &str,
) -> Option<String> {
    let carrier = dialogue.get(range.end)?;
    if carrier["role"] != "user" {
        return None;
    }
    let text = carrier["content"].as_str()?.to_owned();
    let prefix = format!(
        "{HISTORY_SUMMARY_MARKER}, written by you to free context:]\n{}\n[End of the summary. The conversation continues with the user's message below.]\n\n",
        summary.trim()
    );
    dialogue[range.end]["content"] = Value::String(format!("{prefix}{text}"));
    dialogue.drain(range);
    Some(prefix)
}

/// Tokens of a message's text: its content and its calls' string arguments.
fn message_tokens(message: &Value, count: &mut dyn FnMut(&str) -> Result<usize>) -> Result<usize> {
    let content = message["content"].as_str().map_or(Ok(0), &mut *count)?;
    let arguments = if message["role"] == "tool" {
        0
    } else {
        prunable_tokens(message, count)?
    };
    Ok(content + arguments)
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

    /// System, user, then `n` exchanges whose code is `size` characters (one token each 4).
    fn turn_with(n: usize, size: usize) -> Vec<Value> {
        let mut dialogue = vec![
            json!({"role":"system", "content":"sys"}),
            json!({"role":"user", "content":"task"}),
        ];
        for index in 0..n {
            dialogue.push(call(&format!("{index}{}", "c".repeat(size))));
            dialogue.push(tool(&format!("result {index}")));
        }
        dialogue
    }

    /// System, then `turns` exchanges of `size` characters each (question and answer), then
    /// a short new question.
    fn conversation(turns: usize, size: usize) -> Vec<Value> {
        let mut dialogue = vec![json!({"role":"system", "content":"sys"})];
        for index in 0..turns {
            dialogue
                .push(json!({"role":"user", "content":format!("q{index}{}", "u".repeat(size))}));
            dialogue.push(
                json!({"role":"assistant", "content":format!("a{index}{}", "a".repeat(size))}),
            );
        }
        dialogue.push(json!({"role":"user", "content":"next"}));
        dialogue
    }

    #[test]
    fn earlier_turns_are_cut_before_a_user_message_and_the_newest_are_kept() -> Result<()> {
        // 5 turns of 2 x 1 001 tokens. 20 % of 30 000 keeps the two newest turns and the
        // new question; 20 % of 12 000 only one; the new question alone otherwise.
        let dialogue = conversation(5, 4_000);
        let end = dialogue.len();
        let mut always = |_: usize| Ok(true);
        for (limit, cut) in [(30_000, 7), (12_000, 9), (5_000, 11)] {
            let range = history_range(&dialogue, end, limit, &mut bytes, &mut always)?.unwrap();
            assert_eq!(range, 1..cut, "context {limit}");
            assert_eq!(dialogue[range.end]["role"], "user");
            assert_eq!(dialogue[range.end - 1]["role"], "assistant");
        }
        // A summary step that does not fit at the preferred cut is tried one turn earlier,
        // down to the oldest turn alone; none fitting gives nothing.
        let mut tried = Vec::new();
        let range = history_range(&dialogue, end, 5_000, &mut bytes, &mut |cut| {
            tried.push(cut);
            Ok(cut <= 5)
        })?;
        assert_eq!((range, tried), (Some(1..5), vec![11, 9, 7, 5]));
        assert_eq!(
            history_range(&dialogue, end, 5_000, &mut bytes, &mut |_| Ok(false))?,
            None
        );
        // One turn, or only the question: nothing to give up. Without a system message the
        // first message is index 0.
        assert_eq!(
            history_range(&conversation(0, 10), 2, 5_000, &mut bytes, &mut always)?,
            None
        );
        let bare = conversation(2, 4_000)[1..].to_vec();
        assert_eq!(
            history_range(&bare, bare.len(), 5_000, &mut bytes, &mut always)?,
            Some(0..4)
        );
        // Messages after `end` (this turn's exchanges) are never part of it.
        let mut running = conversation(2, 4_000);
        running.push(call("x"));
        running.push(tool("out"));
        running.push(json!({"role":"user", "content":"a note of this turn"}));
        assert_eq!(
            history_range(&running, 6, 5_000, &mut bytes, &mut always)?,
            Some(1..5)
        );
        Ok(())
    }

    #[test]
    fn a_history_summary_heads_the_first_kept_user_message() {
        let mut dialogue = conversation(2, 40);
        let kept = dialogue[3]["content"].as_str().unwrap().to_owned();
        let prefix = apply_history_summary(&mut dialogue, 1..3, "  - did A\n").unwrap();
        assert_eq!(
            prefix,
            format!(
                "{HISTORY_SUMMARY_MARKER}, written by you to free context:]\n- did A\n[End of the summary. The conversation continues with the user's message below.]\n\n"
            )
        );
        assert_eq!(dialogue.len(), 4);
        assert_eq!(dialogue[1]["content"], format!("{prefix}{kept}"));
        assert_eq!(dialogue[1]["role"], "user");
        // An assistant message, or a user message with images, cannot carry it: nothing changes.
        let mut wrong = conversation(2, 40);
        let before = wrong.clone();
        assert!(apply_history_summary(&mut wrong, 1..2, "s").is_none());
        wrong[3]["content"] = json!([{"type":"text", "text":"q"}]);
        assert!(apply_history_summary(&mut wrong, 1..3, "s").is_none());
        wrong[3] = before[3].clone();
        assert_eq!(wrong, before);
        assert!(
            is_summary_request(&json!(SUMMARY_REQUEST))
                && is_summary_request(&json!(HISTORY_SUMMARY_REQUEST))
        );
        assert!(
            !is_summary_request(&json!("Write a summary")) && !is_summary_request(&Value::Null)
        );
    }

    #[test]
    fn compaction_keeps_the_newest_exchanges_that_fit_and_never_splits_a_call() -> Result<()> {
        // 6 exchanges of 1 003 tokens (code and result): 20 % of 11 000 keeps the two newest,
        // 20 % of 10 000 only the newest.
        let dialogue = turn_with(6, 4_000);
        assert_eq!(
            compaction_range(&dialogue, 2, 11_000, &mut bytes)?,
            Some(2..10)
        );
        assert_eq!(dialogue[10]["role"], "assistant");
        assert_eq!(
            compaction_range(&dialogue, 2, 10_000, &mut bytes)?,
            Some(2..12)
        );
        // The newest exchange stays even when it alone exceeds the share.
        let range = compaction_range(&turn_with(3, 40_000), 2, 10_000, &mut bytes)?;
        assert_eq!(range, Some(2..6));
        // One exchange, or a turn that does not start with the model's message: nothing.
        assert_eq!(
            compaction_range(&turn_with(1, 4_000), 2, 10_000, &mut bytes)?,
            None
        );
        assert_eq!(
            compaction_range(&turn_with(4, 4_000), 1, 10_000, &mut bytes)?,
            None
        );
        // Small exchanges all fit in the kept share except the first, always summarized.
        assert_eq!(
            compaction_range(&turn_with(4, 10), 2, 10_000, &mut bytes)?,
            Some(2..4)
        );
        Ok(())
    }

    #[test]
    fn a_summary_replaces_its_range_inside_the_first_kept_result() {
        let mut dialogue = turn_with(4, 10);
        let request = summary_dialogue(&dialogue, &(2..6));
        assert_eq!(request.len(), 7);
        assert_eq!(
            request[6],
            json!({"role":"user", "content":SUMMARY_REQUEST})
        );
        assert!(apply_summary(&mut dialogue, 2..6, "  - did A\n- did B \n"));
        assert_eq!(dialogue.len(), 6);
        assert_eq!(dialogue[2], call(&format!("2{}", "c".repeat(10))));
        assert_eq!(
            dialogue[3]["content"],
            format!(
                "{SUMMARY_MARKER}, written by you to free context:]\n- did A\n- did B\n[End of the summary. Result of the call above:]\n\nresult 2"
            )
        );
        assert_eq!(dialogue[5]["content"], "result 3");
        // Nothing to carry it: unchanged.
        let mut bare = turn_with(2, 10);
        bare.truncate(5);
        let before = bare.clone();
        assert!(!apply_summary(&mut bare, 2..4, "x"));
        assert_eq!(bare, before);
    }

    #[test]
    fn pruning_never_shortens_a_summary() -> Result<()> {
        let summary = format!("{SUMMARY_MARKER}, written by you:]\n{}", "s".repeat(8_000));
        let mut dialogue = turn_with(1, 10);
        dialogue[3] = tool(&summary);
        for index in 0..4 {
            dialogue.push(call(&format!("{index}")));
            dialogue.push(tool(&"o".repeat(8_000)));
        }
        let used = dialogue
            .iter()
            .map(|m| bytes(&m.to_string()).unwrap())
            .sum();
        let pruned = prune_context(&mut dialogue, used, 9_000, &mut bytes)?;
        assert!(pruned.outputs > 0, "other old outputs were pruned");
        assert_eq!(dialogue[3]["content"], summary.as_str());
        Ok(())
    }

    #[test]
    fn continuation_uses_the_right_channel_and_can_fit_a_short_context() {
        assert_eq!(
            reasoning_continuation("</think>", false, false, true),
            "</think>\n\n"
        );
        assert_eq!(
            reasoning_continuation("<channel|>", false, true, true),
            "<channel|>"
        );
        assert_eq!(
            reasoning_continuation("<|end_message|>", false, false, true),
            "<|end_message|><|message_model|><|content_text|>"
        );
        assert_eq!(
            reasoning_continuation("<|end_message|>", false, true, true),
            "<|end_message|><|message_model|>"
        );
        assert!(
            reasoning_continuation("<|end_message|>", true, true, false)
                .ends_with("<|message_model|><|content_text|>")
        );
        assert!(cut_thinking_now(1, 2, 3, 2, 3));
        assert!(!cut_thinking_now(1, 2, 4, 2, 3));
        assert!(!cut_thinking_now(2, 2, 2, 2, 2));
        assert!(!cut_thinking_now(2, 2, 3, 2, 0));
    }

    #[test]
    fn reasoning_budget_is_optional_and_bounded() {
        assert_eq!(reasoning_budget(None).unwrap(), None);
        assert_eq!(reasoning_budget(Some(256)).unwrap(), Some(256));
        assert_eq!(reasoning_budget(Some(131_072)).unwrap(), Some(131_072));
        assert!(reasoning_budget(Some(255)).is_err());
        assert!(reasoning_budget(Some(131_073)).is_err());
        assert!(reasoning_budget(Some(0)).is_err());
        assert_eq!(step_reasoning_budget(8_192, 100_000), Some(8_192));
        assert_eq!(step_reasoning_budget(8_192, 6_000), Some(3_000));
        assert_eq!(step_reasoning_budget(8_192, 1), None);
        assert_eq!(step_reasoning_budget(8_192, 0), None);
    }

    #[test]
    fn studio_thinking_targets_cut_distinctly_and_leave_an_answer_slot() {
        // A deterministic decoder remains in its thinking channel for 6,144 sampled
        // tokens unless the production policy requests a closure sooner.
        for (request, expected) in [
            (None, None),
            (Some(256), Some(256)),
            (Some(1024), Some(1024)),
            (Some(4096), Some(4096)),
        ] {
            let limit = reasoning_budget(request)
                .unwrap()
                .and_then(|budget| step_reasoning_budget(budget, 16_384));
            let closed_at = limit.and_then(|limit| {
                (1..=6144).find(|&thought| {
                    cut_thinking_now(thought, limit, 16_384 - thought, 2, 16_384 - thought)
                })
            });
            assert_eq!(closed_at, expected);
        }

        let limit = step_reasoning_budget(4096, 768).unwrap();
        assert_eq!(limit, 384);
        assert!(!cut_thinking_now(383, limit, 385, 2, 385));
        assert!(cut_thinking_now(384, limit, 384, 2, 384));
        assert!(cut_thinking_now(2, limit, 3, 2, 10));
        assert!(!cut_thinking_now(2, limit, 2, 2, 10));
        assert!(!cut_thinking_now(limit, limit, 384, 2, 0));
    }

    #[test]
    fn forced_close_ends_with_each_family_marker() {
        let qwen = reasoning_continuation("</think>", false, true, false);
        assert_eq!(
            qwen,
            "\n\nI will answer the request now, using a tool only if needed.\n</think>\n\n"
        );
        let chat = reasoning_continuation("</think>", false, false, false);
        assert!(chat.contains("final answer in the requested format"));
        assert!(chat.ends_with("</think>\n\n"));
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

    /// AGENT-2026-11-26-LIVE-CACHE: a step whose arguments are shortened loses its own text (it
    /// still holds the whole arguments); a step left as it is keeps it.
    #[test]
    fn prune_drops_the_own_text_of_a_step_it_shortens() {
        let long = "w".repeat(6_000);
        let mut old = call(&long);
        old["raw"] = json!(format!("thinking {long}"));
        let mut short = call("tiny");
        short["raw"] = json!("thinking tiny");
        let mut recent = call(&long);
        recent["raw"] = json!(format!("thinking {long}"));
        let mut dialogue = vec![
            old,
            tool("ok"),
            short,
            tool("ok"),
            recent, // after the newest kept output's call: kept
            tool(&"n".repeat(400)),
        ];
        let pruned = prune_context(&mut dialogue, 9_000, 10_000, &mut bytes).unwrap();
        assert_eq!(pruned.arguments, 1);
        assert!(dialogue[0].get("raw").is_none());
        assert_eq!(dialogue[2]["raw"], "thinking tiny");
        assert_eq!(dialogue[4]["raw"], format!("thinking {long}"));
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
