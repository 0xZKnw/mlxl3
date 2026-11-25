//! The turn loop of the chat bridge and the Agent mode (AGENT-2026-11-24-DSH-PISTES, P1):
//! generate a step, run its tool calls, feed the results back, until the model answers.
//! The loop only sees a [`Host`] (the model, its tokenizer, the MCP servers and the event
//! stream), so the policies of `agent_loop` can be driven by a scripted model in CPU tests,
//! as DeepSeek Harness tests its loop with a scripted adapter. `main.rs` is the real host.
use crate::GenerationStopReason;
use crate::agent_loop;
use crate::mcp::ToolResult;
use crate::structured::{self, Schema};
use crate::tool_call;
use anyhow::{Context, Result, bail};
use serde_json::{Value, json};
use std::sync::atomic::{AtomicBool, Ordering};

/// What one generated step gives the loop.
#[derive(Clone, Debug)]
pub struct Step {
    pub stop_reason: GenerationStopReason,
    /// The full generated text (thinking, answer and calls), as the cache saw it.
    pub raw: String,
    /// The visible answer.
    pub answer: String,
    /// The text that may hold tool calls.
    pub tool_text: String,
    /// The generated token ids (a forced end of thinking continues from them).
    pub generated: Vec<u32>,
    /// The step stopped at its reasoning budget; holds the exact text to resume the model.
    pub reasoning_cut: Option<String>,
    /// Prompt and generated tokens in the context after this step.
    pub context_used: usize,
    pub generated_tokens: usize,
    pub token_hash: String,
}

/// One step to generate.
pub struct StepRequest<'a> {
    pub dialogue: &'a [Value],
    pub tools: &'a [Value],
    pub max_tokens: i64,
    /// Tokens already generated in this assistant message (a forced end of thinking).
    pub continuation: &'a [u32],
    pub reasoning_budget: Option<usize>,
    /// Nothing is streamed to the client (JSON output).
    pub buffered: bool,
}

/// What the loop needs from the engine.
pub trait Host {
    /// Generates one step; the host streams its deltas and keeps its statistics.
    fn generate(&mut self, request: StepRequest<'_>) -> Result<Step>;
    /// Tokens of the prompt the next step would render.
    fn prompt_tokens(&mut self, dialogue: &[Value], tools: &[Value]) -> Result<usize>;
    fn encode(&mut self, text: &str) -> Result<Vec<u32>>;
    fn decode(&mut self, tokens: &[u32]) -> Result<String>;
    /// The MCP server of a tool, for the client's events.
    fn server(&self, tool: &str) -> Option<String>;
    fn call_tool(&mut self, name: &str, arguments: Value) -> ToolResult;
    fn emit(&mut self, event: Value) -> Result<()>;
    fn cancellation(&self) -> &AtomicBool;
    fn yield_requested(&self) -> bool;
    /// The prompt cache no longer matches what the next step renders.
    fn forget_prompt_cache(&mut self);
    /// The model state after a cancellation (the bridge resets it before failing).
    fn reset_model(&mut self);
    /// The JSON answer is about to be shown (end-to-end time to first text).
    fn answer_shown(&mut self);
}

/// The fixed settings of a turn.
pub struct Turn<'a> {
    pub request_id: &'a str,
    pub context_limit: usize,
    pub max_tokens: i64,
    /// Rounds that may run tools (JSON output keeps its three validation attempts).
    pub tool_rounds: usize,
    pub reasoning_budget: Option<usize>,
    pub schema: Option<&'a Schema>,
    pub allowed_tools: Option<&'a [String]>,
    /// Messages the client sent; later ones are this turn's exchanges.
    pub original_length: usize,
}

/// How the turn ended, for the bridge's `complete` event.
#[derive(Debug, PartialEq)]
pub struct TurnEnd {
    /// What the client keeps as the assistant message (the validated JSON, or the answer).
    pub assistant: String,
    pub raw: String,
    pub token_hash: String,
    pub context_full: bool,
    pub stop_reason: &'static str,
    /// This turn's exchanges and final message, when tools ran.
    pub turn_context: Option<String>,
    /// The JSON answer was validated and shown.
    pub json_answer: bool,
    pub rounds: usize,
}

/// Steering lets a running MCP call finish. A simultaneous Stop wins at the next boundary.
#[derive(Debug, PartialEq, Eq)]
enum ToolBoundary {
    Continue,
    Yield,
    Cancel,
}

fn tool_boundary(cancelled: bool, yield_requested: bool) -> ToolBoundary {
    if cancelled {
        ToolBoundary::Cancel
    } else if yield_requested {
        ToolBoundary::Yield
    } else {
        ToolBoundary::Continue
    }
}

#[cfg(kani)]
#[kani::proof]
fn queued_guidance_never_overrides_stop_or_starts_a_tool() {
    let cancelled = kani::any();
    let queued = kani::any();
    let action = tool_boundary(cancelled, queued);
    assert_eq!(action == ToolBoundary::Cancel, cancelled);
    assert_eq!(action == ToolBoundary::Continue, !cancelled && !queued);
    assert_eq!(action == ToolBoundary::Yield, !cancelled && queued);
    kani::cover!(cancelled && queued && action == ToolBoundary::Cancel);
    kani::cover!(action == ToolBoundary::Yield);
    kani::cover!(action == ToolBoundary::Continue);
}

/// This turn's exchanges as the client stores them (`turn_context`): only assistant and
/// tool messages. Keep the same tool failures the UI saw. The parser's private repair notes
/// are not user messages. A note can also carry the summary of earlier, compacted exchanges.
fn turn_history(transcript: &[Value]) -> Vec<Value> {
    let mut history = Vec::new();
    for message in transcript {
        if message["role"] == "user" {
            history.pop(); // the preceding malformed assistant call
            history.push(json!({
                "role":"assistant", "content":"", "tool_calls":[{"type":"function",
                    "function":{"name":"invalid_tool_call", "arguments":{}}}]
            }));
            history.push(json!({
                "role":"tool", "name":"invalid_tool_call", "content":message["content"]
            }));
        } else {
            history.push(message.clone());
        }
    }
    history
}

/// Carried by the error of a turn that stopped (Stop, an engine error) after tools had run:
/// the exchanges that were finished, so the client can give them back to the model in the
/// next turn instead of an answer with no trace of its calls. Displays as the error itself.
#[derive(Debug)]
pub struct Interrupted {
    /// Same form as [`TurnEnd::turn_context`].
    pub turn_context: String,
    message: String,
}

impl std::fmt::Display for Interrupted {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.message)
    }
}

/// The finished exchanges of the turn `error` stopped, when tools had run.
pub fn interrupted_context(error: &anyhow::Error) -> Option<&str> {
    error
        .downcast_ref::<Interrupted>()
        .map(|interrupted| interrupted.turn_context.as_str())
}

/// The bridge's event for a turn that returned `error`: `cancelled` after a Stop, `error`
/// otherwise, with the turn's finished exchanges as `turn_context` when tools had run.
pub fn failure_event(request_id: &str, cancelled: bool, error: &anyhow::Error) -> Value {
    let mut event = if cancelled {
        json!({"type":"cancelled", "request_id":request_id})
    } else {
        json!({"type":"error", "request_id":request_id, "message":error.to_string()})
    };
    if let Some(turn_context) = interrupted_context(error) {
        event["turn_context"] = json!(turn_context);
    }
    event
}

/// What an interrupted turn leaves for the next one: every call that got its result, in
/// order, closed by an empty assistant message (as a paused turn is). A call that was written
/// but has no result is dropped, so no template ever sees a call without its answer. None
/// when no tool result exists: the client then keeps the visible text, as before.
fn interrupted_history(transcript: &[Value]) -> Option<String> {
    let mut history = turn_history(transcript);
    if let Some(last) = history
        .iter()
        .rposition(|message| message.get("tool_calls").is_some())
    {
        let results = history.len() - last - 1;
        let calls = history[last]["tool_calls"].as_array_mut()?;
        calls.truncate(results);
        if calls.is_empty() {
            let text = history[last]["content"].as_str().unwrap_or_default();
            if text.trim().is_empty() {
                history.truncate(last);
            } else {
                history[last] = json!({"role":"assistant", "content":text});
            }
        }
    }
    if !history.iter().any(|message| message["role"] == "tool") {
        return None;
    }
    if history
        .last()
        .is_some_and(|message| message["role"] == "tool")
    {
        history.push(json!({"role":"assistant", "content":""}));
    }
    serde_json::to_string(&history).ok()
}

fn yielded_turn(
    host: &mut dyn Host,
    transcript: &[Value],
    answer: String,
    token_hash: String,
    rounds: usize,
) -> Result<TurnEnd> {
    host.forget_prompt_cache();
    let history = turn_history(transcript);
    Ok(TurnEnd {
        raw: answer.clone(),
        assistant: answer,
        token_hash,
        context_full: false,
        stop_reason: "agent_yield",
        turn_context: Some(serde_json::to_string(&history)?),
        json_answer: false,
        rounds,
    })
}

/// Runs one turn on `dialogue` with `tools` offered to the model. An error raised after tool
/// results carries them ([`interrupted_context`]).
pub fn run_turn(
    host: &mut dyn Host,
    mut dialogue: Vec<Value>,
    tools: &[Value],
    turn: &Turn<'_>,
) -> Result<TurnEnd> {
    run_steps(host, &mut dialogue, tools, turn).map_err(|error| {
        // JSON output has no tools: its user messages are validation retries, not call notes.
        let transcript = if turn.schema.is_some() {
            &[]
        } else {
            dialogue.get(turn.original_length..).unwrap_or_default()
        };
        match interrupted_history(transcript) {
            Some(turn_context) => {
                let message = error.to_string();
                error.context(Interrupted {
                    turn_context,
                    message,
                })
            }
            None => error,
        }
    })
}

fn run_steps(
    host: &mut dyn Host,
    dialogue: &mut Vec<Value>,
    tools: &[Value],
    turn: &Turn<'_>,
) -> Result<TurnEnd> {
    let request_id = turn.request_id;
    let context_limit = turn.context_limit;
    let schema = turn.schema;
    // `tool_rounds` rounds may run tools; one more round answers from their results, so
    // the limit ends in an answer, not in an error. (JSON output has no tools: it keeps its
    // three validation attempts.)
    let tool_rounds = if schema.is_some() {
        tool_call::DEFAULT_TOOL_ROUNDS
    } else {
        turn.tool_rounds
    };
    // AGENT-2026-11-23-BOUCLE: the round that answers without tools; identical calls that keep
    // returning the same result bring it forward (B5).
    let mut last_round = tool_rounds;
    let mut repeated = false;
    let mut repeats = agent_loop::RepeatGuard::default();
    // One step is one assistant message. A forced end of thinking (B2) continues the same
    // message from its exact tokens; the parts written so far are kept here.
    let mut continuation: Vec<u32> = Vec::new();
    let mut step_raw = String::new();
    let mut step_answer = String::new();
    let mut step_tool_text = String::new();
    let mut forced_closes = 0usize;
    let reasoning_budget = turn.reasoning_budget.filter(|_| schema.is_none());
    let mut nudged = false;
    let mut context_retry = false;
    let mut generated_total = 0usize;
    // Prompt tokens of the next round, from the last one (pruning measures only near the trigger).
    let mut prompt_estimate: Option<usize> = None;
    // P2: what the model may call, by the names it was offered.
    let offered: Vec<&str> = tools
        .iter()
        .filter_map(|tool| tool["function"]["name"].as_str())
        .filter(|name| structured::tool_allowed(name, turn.allowed_tools))
        .collect();
    // P3: dialogue length at the last summary that could not be made; another is tried only
    // once the turn has grown by two exchanges.
    let mut compaction_failed_at: Option<usize> = None;
    let mut round = 0usize;
    loop {
        // B4: old tool exchanges make room before the context runs out.
        let prunable = dialogue
            .iter()
            .any(|message| message["role"] == "tool" || message.get("tool_calls").is_some());
        if continuation.is_empty()
            && prunable
            && prompt_estimate.is_none_or(|tokens| tokens * 100 > context_limit * 60)
        {
            let used = host.prompt_tokens(dialogue, tools)?;
            let pruned = agent_loop::prune_context(dialogue, used, context_limit, &mut |text| {
                Ok(host.encode(text)?.len())
            })?;
            if !pruned.is_empty() {
                host.emit(json!({
                    "type":"generation_status", "request_id":request_id, "phase":"prefill",
                    "text":format!(
                        "Freeing context: {} old tool outputs and {} long call arguments shortened",
                        pruned.outputs, pruned.arguments
                    )
                }))?;
            }
            // P3: what pruning could not free is summarized.
            let left = used.saturating_sub(pruned.freed);
            if left * 100 > context_limit * agent_loop::COMPACT_TRIGGER_PERCENT
                && compaction_allowed(compaction_failed_at, dialogue.len())
                && !compact(host, dialogue, tools, turn, &mut generated_total, left)?
            {
                compaction_failed_at = Some(dialogue.len());
            }
        }
        host.emit(json!({
            "type":"generation_status", "request_id":request_id,
            "phase":"prefill", "text":if !continuation.is_empty() {
                "Preparing the response"
            } else if round == 0 {
                "Preparing context"
            } else {
                "Reading MCP results"
            }
        }))?;
        let round_max_tokens = structured::remaining_tokens(turn.max_tokens, generated_total)
            .context(if schema.is_some() {
                "JSON token budget exhausted"
            } else {
                "Generation token budget exhausted"
            })?;
        let output = host.generate(StepRequest {
            dialogue,
            tools,
            max_tokens: round_max_tokens,
            continuation: &continuation,
            reasoning_budget: reasoning_budget
                .filter(|_| forced_closes < agent_loop::FORCED_CLOSES_PER_STEP),
            buffered: schema.is_some(),
        })?;
        generated_total += output.generated_tokens;
        prompt_estimate = Some(output.context_used);
        // The model step has ended. A queued instruction can take over before a retry,
        // another reasoning segment or the first tool; an explicit Stop takes priority.
        if host.yield_requested() {
            if host.cancellation().load(Ordering::Relaxed) {
                host.reset_model();
                bail!("generation cancelled");
            }
            let answer = step_answer + &output.answer;
            dialogue.push(json!({"role":"assistant", "content":answer}));
            return yielded_turn(
                host,
                &dialogue[turn.original_length..],
                answer,
                output.token_hash,
                round,
            );
        }
        if let Some(text) = output.reasoning_cut {
            forced_closes += 1;
            step_raw.push_str(&output.raw);
            step_raw.push_str(&text);
            step_answer.push_str(&output.answer);
            step_tool_text.push_str(&output.tool_text);
            continuation.extend_from_slice(&output.generated);
            continuation.extend(host.encode(&text)?);
            continue;
        }
        let raw = std::mem::take(&mut step_raw) + &output.raw;
        let answer = std::mem::take(&mut step_answer) + &output.answer;
        let tool_text = std::mem::take(&mut step_tool_text) + &output.tool_text;
        continuation.clear();
        forced_closes = 0;
        let validated_json = if let Some(schema) = schema {
            match schema.validate_cancellable(&answer, host.cancellation()) {
                Ok(answer) => Some(answer),
                Err(error) if round < 2 => {
                    dialogue.push(json!({"role":"assistant", "content":raw}));
                    dialogue.push(json!({"role":"user", "content":format!("The previous response failed JSON validation: {error}. Return only valid JSON conforming to the schema.")}));
                    host.forget_prompt_cache();
                    round += 1;
                    continue;
                }
                Err(error) => return Err(error.context("JSON validation failed after 3 attempts")),
            }
        } else {
            None
        };
        // A call the parser cannot read goes back to the model instead of ending the turn.
        let (calls, unreadable) = if tools.is_empty() {
            (Vec::new(), None)
        } else {
            match tool_call::parse_answer(&tool_text) {
                Ok(calls) => (calls, None),
                Err(error) => (Vec::new(), Some(error)),
            }
        };
        let context_full = output.context_used >= context_limit;
        if context_full && !tools.is_empty() && !context_retry {
            // B4: a step cut by the end of the context runs once more after old tool exchanges
            // made room (its partial output is dropped).
            let pruned =
                agent_loop::prune_context(dialogue, context_limit, context_limit, &mut |text| {
                    Ok(host.encode(text)?.len())
                })?;
            if !pruned.is_empty() {
                context_retry = true;
                prompt_estimate = None;
                host.emit(json!({
                    "type":"generation_status", "request_id":request_id, "phase":"prefill",
                    "text":format!(
                        "Context full: {} old tool outputs and {} long call arguments shortened, retrying the step",
                        pruned.outputs, pruned.arguments
                    )
                }))?;
                continue;
            }
            // P3: nothing left to prune: summarize the older exchanges, then retry once.
            if compaction_allowed(compaction_failed_at, dialogue.len()) {
                if compact(
                    host,
                    dialogue,
                    tools,
                    turn,
                    &mut generated_total,
                    context_limit,
                )? {
                    context_retry = true;
                    prompt_estimate = None;
                    continue;
                }
                compaction_failed_at = Some(dialogue.len());
            }
        }
        let final_round = round == last_round;
        // B6: a step that ends with nothing after tool results gets one reminder, added to the
        // last result (no new message: every chat template accepts it).
        if calls.is_empty()
            && unreadable.is_none()
            && !context_full
            && !final_round
            && !nudged
            && output.stop_reason == GenerationStopReason::Eos
            && answer.trim().is_empty()
            && let Some(last) = dialogue
                .last_mut()
                .filter(|message| message["role"] == "tool")
            && let Some(text) = last["content"].as_str().map(str::to_owned)
        {
            nudged = true;
            last["content"] = json!(format!("{text}\n\n{}", agent_loop::EMPTY_STEP_NUDGE));
            continue;
        }
        if (calls.is_empty() && unreadable.is_none()) || context_full || final_round {
            // At the limit nothing more runs: the turn ends on what the model wrote.
            let limited = final_round && (!calls.is_empty() || unreadable.is_some());
            let assistant = validated_json.clone().unwrap_or_else(|| answer.clone());
            if let Some(answer) = &validated_json {
                host.answer_shown();
                host.emit(
                    json!({"type":"delta", "request_id":request_id,"phase":"answer","text":answer}),
                )?;
            }
            let turn_context = if round == 0 {
                None
            } else {
                let mut transcript = dialogue[turn.original_length..].to_vec();
                transcript.push(json!({"role":"assistant", "content":raw}));
                Some(serde_json::to_string(&transcript)?)
            };
            let stop_reason = match (limited, repeated) {
                (true, true) => "repeated_calls",
                (true, false) => "tool_round_limit",
                (false, _) => output.stop_reason.as_str(),
            };
            return Ok(TurnEnd {
                assistant,
                raw,
                token_hash: output.token_hash,
                context_full,
                stop_reason,
                turn_context,
                json_answer: validated_json.is_some(),
                rounds: round,
            });
        }
        context_retry = false;
        if let Some(error) = unreadable {
            let call_id = format!("mcp-{round}-{}-invalid", dialogue.len());
            let text = tool_call::unreadable_call_message(&error);
            host.emit(json!({
                "type":"tool_start", "request_id":request_id,
                "tool_call_id":call_id, "tool_name":"invalid_tool_call", "server_name":null
            }))?;
            host.emit(json!({
                "type":"tool_result", "request_id":request_id,
                "tool_call_id":call_id, "tool_name":"invalid_tool_call",
                "server_name":null, "text":text, "is_error":true
            }))?;
            let note = if round + 1 == last_round {
                format!("{text}\n\n{}", tool_call::round_limit_note(tool_rounds))
            } else {
                text
            };
            dialogue.push(json!({"role":"assistant", "content":tool_text}));
            dialogue.push(json!({"role":"user", "content":note}));
            round += 1;
            continue;
        }
        // A bare tool name that a single offered tool carries is that tool: the call runs and
        // the dialogue keeps the full name.
        let calls: Vec<tool_call::ToolCall> = calls
            .into_iter()
            .map(|mut call| {
                if let Some(full) = tool_call::offered_name(&call.name, &offered) {
                    call.name = full.to_owned();
                }
                call
            })
            .collect();
        let assistant_index = dialogue.len();
        dialogue.push(json!({"role":"assistant", "content":answer, "tool_calls":[]}));
        let mut calls = calls.into_iter();
        // Check before the first call, between individual calls in a batch, and after the
        // final result. The transcript contains only calls that were actually dispatched.
        loop {
            match tool_boundary(
                host.cancellation().load(Ordering::Relaxed),
                host.yield_requested(),
            ) {
                ToolBoundary::Cancel => {
                    host.reset_model();
                    bail!("generation cancelled");
                }
                ToolBoundary::Yield => {
                    let ran_tools = dialogue.len() > assistant_index + 1;
                    if ran_tools {
                        dialogue.push(json!({"role":"assistant", "content":""}));
                    } else {
                        dialogue[assistant_index] = json!({"role":"assistant", "content":answer});
                    }
                    return yielded_turn(
                        host,
                        &dialogue[turn.original_length..],
                        answer,
                        output.token_hash,
                        round + usize::from(ran_tools),
                    );
                }
                ToolBoundary::Continue => {}
            }
            let Some(call) = calls.next() else { break };
            dialogue[assistant_index]["tool_calls"].as_array_mut().expect("tool_calls is initialized as an array")
                .push(json!({"type":"function", "function":{"name":call.name, "arguments":call.arguments}}));
            let server = host.server(&call.name);
            let call_id = format!("mcp-{round}-{}-{}", dialogue.len(), call.name);
            host.emit(json!({
                "type":"tool_start", "request_id":request_id,
                "tool_call_id":call_id, "tool_name":call.name,
                "server_name":server,
                "arguments":tool_call::shown_arguments(&call.arguments)
            }))?;
            let arguments = call.arguments.clone();
            // P2: a tool that was not offered (a hallucinated name, a project-disabled tool)
            // goes back to the model with the remedy instead of reaching MCP or ending the turn.
            let mut result = if offered.contains(&call.name.as_str()) {
                host.call_tool(&call.name, call.arguments)
            } else {
                ToolResult {
                    text: tool_call::unknown_tool_message(&call.name, &offered),
                    is_error: true,
                }
            };
            let budget = 8192usize.min((context_limit / 8).max(256));
            let result_tokens = host.encode(&result.text)?;
            if result_tokens.len() > budget {
                // Start and end: a command's errors and summary come last.
                let (head, tail) = tool_call::kept_token_ranges(result_tokens.len(), budget, 32);
                let omitted = result_tokens.len() - head.len() - tail.len();
                result.text = format!(
                    "{}\n[... {omitted} tokens of tool output omitted to fit the context ...]\n{}",
                    host.decode(&result_tokens[head])?,
                    host.decode(&result_tokens[tail])?
                );
            }
            if let Some(tokens) = &mut prompt_estimate {
                *tokens += result_tokens.len().min(budget) + 32;
            }
            // B5: the same call returning the same thing again and again ends the tool use.
            if !repeated
                && repeats.observe(&call.name, &arguments, &result.text, result.is_error)
                    >= agent_loop::REPEATED_CALL_LIMIT
            {
                repeated = true;
                last_round = round + 1;
                result.text = format!(
                    "{}\n\n{}",
                    result.text,
                    agent_loop::repeated_call_note(agent_loop::REPEATED_CALL_LIMIT)
                );
            }
            host.emit(json!({
                "type":"tool_result", "request_id":request_id,
                "tool_call_id":call_id, "tool_name":call.name,
                "server_name":server, "text":result.text,
                "is_error":result.is_error
            }))?;
            dialogue.push(json!({
                "role":"tool", "name":call.name, "content":result.text
            }));
        }
        if round + 1 == last_round
            && !repeated
            && let Some(content) = dialogue
                .last_mut()
                .and_then(|message| message.get_mut("content"))
        {
            let text = content.as_str().unwrap_or_default().to_owned();
            *content = json!(format!(
                "{text}\n\n{}",
                tool_call::round_limit_note(tool_rounds)
            ));
        }
        round += 1;
    }
}

/// A summary is tried again only after a failed one once the turn grew by two exchanges.
fn compaction_allowed(failed_at: Option<usize>, length: usize) -> bool {
    failed_at.is_none_or(|failed| length >= failed + 4)
}

/// P3: summarizes the older exchanges of this turn with the model (nothing streamed) and
/// replaces them by the summary. False, with the dialogue unchanged, when there is nothing to
/// summarize, the budget is short, or the summary is empty, cut, a tool call, or not shorter
/// than what it replaces (`used`: the prompt's tokens now).
fn compact(
    host: &mut dyn Host,
    dialogue: &mut Vec<Value>,
    tools: &[Value],
    turn: &Turn<'_>,
    generated_total: &mut usize,
    used: usize,
) -> Result<bool> {
    let request_id = turn.request_id;
    if tools.is_empty() {
        return Ok(false);
    }
    let remaining = structured::remaining_tokens(turn.max_tokens, *generated_total);
    if remaining.is_none_or(|left| left != -1 && left < 2 * agent_loop::SUMMARY_MAX_TOKENS as i64) {
        return Ok(false);
    }
    let Some(range) = agent_loop::compaction_range(
        dialogue,
        turn.original_length,
        turn.context_limit,
        &mut |text| Ok(host.encode(text)?.len()),
    )?
    else {
        return Ok(false);
    };
    let request = agent_loop::summary_dialogue(dialogue, &range);
    if host.prompt_tokens(&request, tools)? + agent_loop::SUMMARY_MAX_TOKENS > turn.context_limit {
        return Ok(false);
    }
    host.emit(json!({
        "type":"generation_status", "request_id":request_id, "phase":"prefill",
        "text":format!("Summarizing {} earlier messages to free context", range.len())
    }))?;
    let mut continuation: Vec<u32> = Vec::new();
    let mut answer = String::new();
    let mut tool_text = String::new();
    let mut generated = 0usize;
    let mut cuts = 0usize;
    let finished = loop {
        let step = host.generate(StepRequest {
            dialogue: &request,
            tools,
            max_tokens: (agent_loop::SUMMARY_MAX_TOKENS - generated) as i64,
            continuation: &continuation,
            reasoning_budget: (cuts < agent_loop::FORCED_CLOSES_PER_STEP)
                .then_some(agent_loop::SUMMARY_REASONING_BUDGET),
            buffered: true,
        })?;
        generated += step.generated_tokens;
        *generated_total += step.generated_tokens;
        answer.push_str(&step.answer);
        tool_text.push_str(&step.tool_text);
        if let Some(text) = step.reasoning_cut.as_deref()
            && generated < agent_loop::SUMMARY_MAX_TOKENS
        {
            cuts += 1;
            continuation.extend_from_slice(&step.generated);
            continuation.extend(host.encode(text)?);
            continue;
        }
        break step.stop_reason == GenerationStopReason::Eos && step.reasoning_cut.is_none();
    };
    let called = tool_call::parse_answer(&tool_text).map_or(true, |calls| !calls.is_empty());
    let summary = answer.trim();
    let mut candidate = dialogue.clone();
    let accepted = finished
        && !called
        && !summary.is_empty()
        && agent_loop::apply_summary(&mut candidate, range.clone(), summary)
        && host.prompt_tokens(&candidate, tools)? < used;
    if !accepted {
        host.emit(json!({
            "type":"generation_status", "request_id":request_id, "phase":"prefill",
            "text":"The summary could not be used; the earlier messages are kept"
        }))?;
        return Ok(false);
    }
    let after = host.prompt_tokens(&candidate, tools)?;
    *dialogue = candidate;
    host.emit(json!({
        "type":"generation_status", "request_id":request_id, "phase":"prefill",
        "text":format!(
            "Context summarized: {} earlier messages replaced ({used} to {after} tokens)",
            range.len()
        )
    }))?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    //! The loop driven by a scripted model (DeepSeek Harness's `MockAdapter` approach): each
    //! test lists the steps the model writes and the tool results, then checks the events,
    //! the dialogues the model was given and how the turn ended. One token per byte.
    use super::*;
    use std::collections::VecDeque;

    struct Scripted {
        raw: String,
        answer: String,
        tool_text: String,
        stop: GenerationStopReason,
        cut: Option<&'static str>,
        /// Context after the step; by default the prompt bytes plus the raw text.
        context_used: Option<usize>,
    }

    fn answer(text: &str) -> Scripted {
        Scripted {
            raw: text.into(),
            answer: text.into(),
            tool_text: text.into(),
            stop: GenerationStopReason::Eos,
            cut: None,
            context_used: None,
        }
    }

    fn call(code: &str) -> Scripted {
        let text = format!(
            "<tool_call>{}</tool_call>",
            json!({"name":"javascript", "arguments":{"code":code}})
        );
        Scripted {
            raw: text.clone(),
            answer: String::new(),
            tool_text: text,
            stop: GenerationStopReason::Eos,
            cut: None,
            context_used: None,
        }
    }

    struct Recorded {
        dialogue: Vec<Value>,
        max_tokens: i64,
        continuation: Vec<u32>,
        reasoning_budget: Option<usize>,
        buffered: bool,
    }

    /// The simulated MCP server: a result for each call.
    type Results = Box<dyn FnMut(&str, &Value) -> ToolResult>;

    struct ScriptHost {
        steps: VecDeque<Scripted>,
        /// What the model writes when asked for a summary (P3).
        summaries: VecDeque<Scripted>,
        results: Results,
        requests: Vec<Recorded>,
        events: Vec<Value>,
        calls: Vec<(String, Value)>,
        cancelled: AtomicBool,
        /// Cancel once this many tools ran.
        cancel_after: Option<usize>,
        /// The step with this number (from 1) fails in the engine instead of generating.
        fail_step: Option<usize>,
        yield_after: Option<usize>,
        resets: usize,
        forgets: usize,
        shown: usize,
    }

    impl ScriptHost {
        fn new(steps: Vec<Scripted>) -> Self {
            Self {
                steps: steps.into(),
                summaries: VecDeque::new(),
                results: Box::new(|_, arguments| ToolResult {
                    text: format!("ran {}", arguments["code"].as_str().unwrap_or("")),
                    is_error: false,
                }),
                requests: Vec::new(),
                events: Vec::new(),
                calls: Vec::new(),
                cancelled: AtomicBool::new(false),
                cancel_after: None,
                fail_step: None,
                yield_after: None,
                resets: 0,
                forgets: 0,
                shown: 0,
            }
        }

        fn statuses(&self) -> Vec<String> {
            self.events
                .iter()
                .filter(|event| event["type"] == "generation_status")
                .map(|event| event["text"].as_str().unwrap().to_owned())
                .collect()
        }

        fn kinds(&self) -> Vec<String> {
            self.events
                .iter()
                .map(|event| event["type"].as_str().unwrap().to_owned())
                .collect()
        }
    }

    fn bytes(dialogue: &[Value]) -> usize {
        serde_json::to_string(dialogue).unwrap().len()
    }

    impl Host for ScriptHost {
        fn generate(&mut self, request: StepRequest<'_>) -> Result<Step> {
            self.requests.push(Recorded {
                dialogue: request.dialogue.to_vec(),
                max_tokens: request.max_tokens,
                continuation: request.continuation.to_vec(),
                reasoning_budget: request.reasoning_budget,
                buffered: request.buffered,
            });
            if self.fail_step == Some(self.requests.len()) {
                bail!("Metal device lost");
            }
            let summary = request.buffered
                && request
                    .dialogue
                    .last()
                    .is_some_and(|message| message["content"] == agent_loop::SUMMARY_REQUEST);
            let step = if summary {
                self.summaries
                    .pop_front()
                    .expect("a summary was asked but the script has none")
            } else {
                self.steps
                    .pop_front()
                    .expect("the model was asked for more steps than the script has")
            };
            Ok(Step {
                stop_reason: step.stop,
                context_used: step
                    .context_used
                    .unwrap_or(bytes(request.dialogue) + step.raw.len()),
                generated_tokens: step.raw.len(),
                generated: step.raw.bytes().map(u32::from).collect(),
                raw: step.raw,
                answer: step.answer,
                tool_text: step.tool_text,
                reasoning_cut: step.cut.map(|close| {
                    agent_loop::reasoning_continuation(
                        close,
                        summary,
                        !request.tools.is_empty(),
                        false,
                    )
                }),
                token_hash: format!("hash-{}", self.requests.len()),
            })
        }

        fn prompt_tokens(&mut self, dialogue: &[Value], _: &[Value]) -> Result<usize> {
            Ok(bytes(dialogue))
        }

        fn encode(&mut self, text: &str) -> Result<Vec<u32>> {
            Ok(text.bytes().map(u32::from).collect())
        }

        fn decode(&mut self, tokens: &[u32]) -> Result<String> {
            Ok(
                String::from_utf8_lossy(&tokens.iter().map(|&t| t as u8).collect::<Vec<_>>())
                    .into_owned(),
            )
        }

        fn server(&self, tool: &str) -> Option<String> {
            (tool == "javascript").then(|| "sandbox".to_owned())
        }

        fn call_tool(&mut self, name: &str, arguments: Value) -> ToolResult {
            self.calls.push((name.to_owned(), arguments.clone()));
            if self.cancel_after == Some(self.calls.len()) {
                self.cancelled.store(true, Ordering::Relaxed);
            }
            (self.results)(name, &arguments)
        }

        fn emit(&mut self, event: Value) -> Result<()> {
            self.events.push(event);
            Ok(())
        }

        fn yield_requested(&self) -> bool {
            self.yield_after == Some(self.calls.len())
        }

        fn cancellation(&self) -> &AtomicBool {
            &self.cancelled
        }

        fn forget_prompt_cache(&mut self) {
            self.forgets += 1;
        }

        fn reset_model(&mut self) {
            self.resets += 1;
        }

        fn answer_shown(&mut self) {
            self.shown += 1;
        }
    }

    fn tools() -> Vec<Value> {
        vec![json!({"type":"function", "function":{"name":"javascript", "parameters":{}}})]
    }

    fn dialogue() -> Vec<Value> {
        vec![
            json!({"role":"system", "content":"You are an agent."}),
            json!({"role":"user", "content":"Fix the bug."}),
        ]
    }

    fn turn(context_limit: usize, tool_rounds: usize) -> Turn<'static> {
        Turn {
            request_id: "r1",
            context_limit,
            max_tokens: -1,
            tool_rounds,
            reasoning_budget: None,
            schema: None,
            allowed_tools: None,
            original_length: 2,
        }
    }

    #[test]
    fn a_direct_answer_ends_the_turn_without_tools() -> Result<()> {
        let mut host = ScriptHost::new(vec![answer("Done.")]);
        let end = run_turn(&mut host, dialogue(), &tools(), &turn(100_000, 8))?;
        assert_eq!(
            end,
            TurnEnd {
                assistant: "Done.".into(),
                raw: "Done.".into(),
                token_hash: "hash-1".into(),
                context_full: false,
                stop_reason: "eos",
                turn_context: None,
                json_answer: false,
                rounds: 0,
            }
        );
        assert_eq!(host.statuses(), ["Preparing context"]);
        assert_eq!(host.requests.len(), 1);
        assert_eq!(host.requests[0].max_tokens, -1);
        assert!(!host.requests[0].buffered && host.calls.is_empty());
        Ok(())
    }

    #[test]
    fn a_call_runs_its_tool_and_the_result_goes_back_to_the_model() -> Result<()> {
        let mut host = ScriptHost::new(vec![call("return 1"), answer("It returns 1.")]);
        let end = run_turn(&mut host, dialogue(), &tools(), &turn(100_000, 8))?;
        assert_eq!(
            host.calls,
            [("javascript".to_owned(), json!({"code":"return 1"}))]
        );
        assert_eq!(
            host.kinds(),
            [
                "generation_status",
                "tool_start",
                "tool_result",
                "generation_status"
            ]
        );
        assert_eq!(host.events[1]["tool_call_id"], "mcp-0-3-javascript");
        assert_eq!(host.events[1]["server_name"], "sandbox");
        assert_eq!(host.events[2]["text"], "ran return 1");
        assert_eq!(
            host.statuses(),
            ["Preparing context", "Reading MCP results"]
        );
        let second = &host.requests[1].dialogue;
        assert_eq!(second.len(), 4);
        assert_eq!(
            second[2],
            json!({"role":"assistant", "content":"", "tool_calls":[{"type":"function", "function":{"name":"javascript", "arguments":{"code":"return 1"}}}]})
        );
        assert_eq!(
            second[3],
            json!({"role":"tool", "name":"javascript", "content":"ran return 1"})
        );
        assert_eq!(end.rounds, 1);
        assert_eq!(end.stop_reason, "eos");
        let context: Vec<Value> = serde_json::from_str(end.turn_context.as_deref().unwrap())?;
        assert_eq!(context.len(), 3);
        assert_eq!(
            context[2],
            json!({"role":"assistant", "content":"It returns 1."})
        );
        Ok(())
    }

    #[test]
    fn an_unreadable_call_is_explained_to_the_model_and_nothing_runs() -> Result<()> {
        let mut broken = call("x");
        broken.tool_text = "<tool_call>{\"name\":".into();
        let mut host = ScriptHost::new(vec![broken, answer("Sorry.")]);
        let end = run_turn(&mut host, dialogue(), &tools(), &turn(100_000, 8))?;
        assert!(host.calls.is_empty());
        assert_eq!(host.events[1]["tool_name"], "invalid_tool_call");
        assert_eq!(host.events[2]["is_error"], true);
        let second = &host.requests[1].dialogue;
        assert_eq!(
            second[2],
            json!({"role":"assistant", "content":"<tool_call>{\"name\":"})
        );
        assert_eq!(second[3]["role"], "user");
        assert!(
            second[3]["content"]
                .as_str()
                .unwrap()
                .starts_with("Your tool call could not be read (unfinished tool call; nothing executed); nothing was executed.")
        );
        assert_eq!((end.rounds, end.stop_reason), (1, "eos"));
        Ok(())
    }

    #[test]
    fn a_tool_that_was_not_offered_gets_the_remedy_and_never_runs() -> Result<()> {
        let mut wrong = call("x");
        wrong.tool_text = format!(
            "<tool_call>{}</tool_call>",
            json!({"name":"read_file", "arguments":{"path":"a.txt"}})
        );
        let mut host = ScriptHost::new(vec![wrong, answer("ok")]);
        let end = run_turn(&mut host, dialogue(), &tools(), &turn(100_000, 8))?;
        assert!(host.calls.is_empty(), "MCP never sees an unknown tool");
        let expected = "There is no tool named `read_file`; nothing was executed. The tools you can call are: `javascript`. Functions listed in a tool's description are called inside that tool's input, not as tools.";
        assert_eq!(host.events[2]["text"], expected);
        assert_eq!(host.events[2]["is_error"], true);
        assert_eq!(host.events[1]["server_name"], Value::Null);
        assert_eq!(host.requests[1].dialogue[3]["content"], expected);
        assert_eq!((end.rounds, end.stop_reason), (1, "eos"));
        // A tool the project disabled is not offered either: the turn goes on (it used to
        // end with "MCP tool is not allowed in this project").
        let mut host = ScriptHost::new(vec![call("1"), answer("ok")]);
        let allowed = vec!["other".to_owned()];
        let mut settings = turn(100_000, 8);
        settings.allowed_tools = Some(&allowed);
        let end = run_turn(&mut host, dialogue(), &tools(), &settings)?;
        assert!(host.calls.is_empty());
        assert_eq!(
            host.events[2]["text"],
            "There is no tool named `javascript`; nothing was executed. No tool can be called now: answer without tools."
        );
        assert_eq!(end.stop_reason, "eos");
        Ok(())
    }

    #[test]
    fn a_bare_tool_name_runs_the_only_offered_tool_that_ends_with_it() -> Result<()> {
        let offered = vec![
            json!({"type":"function", "function":{"name":"sandbox.javascript", "parameters":{}}}),
            json!({"type":"function", "function":{"name":"exa.web_search_exa", "parameters":{}}}),
        ];
        // `call` writes the bare name `javascript`.
        let mut host = ScriptHost::new(vec![call("return 1"), answer("ok")]);
        let end = run_turn(&mut host, dialogue(), &offered, &turn(100_000, 8))?;
        assert_eq!(host.calls.len(), 1, "the call reaches MCP");
        assert_eq!(host.calls[0].0, "sandbox.javascript");
        assert_eq!(host.events[1]["tool_name"], "sandbox.javascript");
        assert_eq!(host.events[2]["is_error"], false);
        // The dialogue keeps the full name: the next step sees the right example.
        assert_eq!(
            host.requests[1].dialogue[2]["tool_calls"][0]["function"]["name"],
            "sandbox.javascript"
        );
        assert_eq!((end.rounds, end.stop_reason), (1, "eos"));
        // Two tools with that last name: nothing is chosen, the model gets the list.
        let twins = vec![
            json!({"type":"function", "function":{"name":"a.javascript", "parameters":{}}}),
            json!({"type":"function", "function":{"name":"b.javascript", "parameters":{}}}),
        ];
        let mut host = ScriptHost::new(vec![call("return 1"), answer("ok")]);
        run_turn(&mut host, dialogue(), &twins, &turn(100_000, 8))?;
        assert!(host.calls.is_empty());
        assert_eq!(host.events[2]["is_error"], true);
        Ok(())
    }

    #[test]
    fn an_empty_step_after_a_result_is_nudged_once() -> Result<()> {
        let mut host = ScriptHost::new(vec![call("a"), answer(""), answer(""), answer("never")]);
        let end = run_turn(&mut host, dialogue(), &tools(), &turn(100_000, 8))?;
        assert_eq!(host.requests.len(), 3, "one reminder, then the turn ends");
        assert_eq!(
            host.requests[2].dialogue[3]["content"],
            format!("ran a\n\n{}", agent_loop::EMPTY_STEP_NUDGE)
        );
        assert_eq!((end.assistant.as_str(), end.rounds), ("", 1));
        assert_eq!(host.steps.len(), 1);
        Ok(())
    }

    #[test]
    fn identical_calls_with_identical_results_stop_the_tools_after_five() -> Result<()> {
        let mut steps: Vec<_> = (0..6).map(|_| call("same")).collect();
        steps.push(answer("unused"));
        let mut host = ScriptHost::new(steps);
        let end = run_turn(&mut host, dialogue(), &tools(), &turn(1_000_000, 100))?;
        assert_eq!(host.calls.len(), 5, "the sixth call is not executed");
        let last = host.requests[5].dialogue.last().unwrap()["content"]
            .as_str()
            .unwrap()
            .to_owned();
        assert_eq!(
            last,
            format!(
                "ran same\n\n{}",
                agent_loop::repeated_call_note(agent_loop::REPEATED_CALL_LIMIT)
            )
        );
        assert_eq!((end.stop_reason, end.rounds), ("repeated_calls", 5));
        Ok(())
    }

    #[test]
    fn the_round_limit_warns_on_the_last_result_and_ends_without_running_more() -> Result<()> {
        let mut host = ScriptHost::new(vec![call("1"), call("2"), call("3")]);
        let end = run_turn(&mut host, dialogue(), &tools(), &turn(1_000_000, 2))?;
        assert_eq!(host.calls.len(), 2);
        assert_eq!(
            host.requests[2].dialogue.last().unwrap()["content"],
            format!("ran 2\n\n{}", tool_call::round_limit_note(2))
        );
        assert_eq!((end.stop_reason, end.rounds), ("tool_round_limit", 2));
        Ok(())
    }

    #[test]
    fn a_full_context_prunes_old_outputs_and_retries_the_step_once() -> Result<()> {
        let big = "x".repeat(4_000);
        let mut full = call("cut");
        full.context_used = Some(10_000);
        let mut again = call("cut again");
        again.context_used = Some(10_000);
        let mut host = ScriptHost::new(vec![call("1"), call("2"), call("3"), full, again]);
        let mut sizes = vec![big.clone(), big.clone(), "small".to_owned()].into_iter();
        host.results = Box::new(move |_, _| ToolResult {
            text: sizes.next().unwrap(),
            is_error: false,
        });
        let end = run_turn(&mut host, dialogue(), &tools(), &turn(10_000, 50))?;
        assert_eq!(
            host.requests.len(),
            5,
            "the cut step ran once more, then the turn ended"
        );
        assert!(
            host.statuses()
                .iter()
                .any(|status| status.starts_with("Context full:")
                    && status.ends_with("retrying the step"))
        );
        let retried = &host.requests[4].dialogue;
        assert!(
            retried[3]["content"].as_str().unwrap().len() < big.len(),
            "oldest output pruned"
        );
        assert!(end.context_full);
        assert_eq!(host.calls.len(), 3, "nothing ran after the context filled");
        Ok(())
    }

    #[test]
    fn a_reasoning_cut_continues_the_same_message_from_its_exact_tokens() -> Result<()> {
        let mut thinking = answer("");
        thinking.raw = "<think>long".into();
        thinking.cut = Some("</think>");
        let mut host = ScriptHost::new(vec![thinking, answer("Done.")]);
        let mut settings = turn(100_000, 8);
        settings.reasoning_budget = Some(512);
        let end = run_turn(&mut host, dialogue(), &tools(), &settings)?;
        let close = agent_loop::reasoning_continuation("</think>", false, true, false);
        let mut expected: Vec<u32> = "<think>long".bytes().map(u32::from).collect();
        expected.extend(close.bytes().map(u32::from));
        assert_eq!(host.requests[1].continuation, expected);
        assert_eq!(host.requests[0].reasoning_budget, Some(512));
        assert!(host.events.iter().all(|event| event["phase"] != "thinking"));
        assert_eq!(host.statuses()[1], "Preparing the response");
        assert_eq!(end.raw, format!("<think>long{close}Done."));
        assert_eq!(end.rounds, 0);
        Ok(())
    }

    #[test]
    fn chat_can_still_call_a_tool_after_a_budgeted_thinking_block() -> Result<()> {
        let mut thinking = answer("");
        thinking.cut = Some("<|end_message|>");
        let mut host = ScriptHost::new(vec![thinking, call("lookup"), answer("42")]);
        let mut settings = turn(100_000, 8);
        settings.reasoning_budget = Some(256);
        let end = run_turn(&mut host, dialogue(), &tools(), &settings)?;
        assert_eq!(end.assistant, "42");
        assert_eq!(host.calls.len(), 1);
        let header = host.encode("<|end_message|><|message_model|>")?;
        assert!(host.requests[1].continuation.ends_with(&header));
        assert!(host.events.iter().all(|event| event["phase"] != "thinking"));
        Ok(())
    }

    #[test]
    fn chat_reasoning_cut_continues_with_answer_and_no_injected_thinking_text() -> Result<()> {
        let mut thinking = answer("");
        thinking.raw = "x".into();
        thinking.cut = Some("</think>");
        let mut host = ScriptHost::new(vec![thinking, answer("42")]);
        let mut settings = turn(1024, 8);
        settings.reasoning_budget = Some(256);
        let end = run_turn(&mut host, dialogue(), &[], &settings)?;
        let close = agent_loop::reasoning_continuation("</think>", false, false, false);
        assert_eq!(end.raw, format!("x{close}42"));
        assert_eq!(end.assistant, "42");
        assert_eq!(host.requests[0].reasoning_budget, Some(256));
        assert_eq!(
            host.requests[1].continuation,
            format!("x{close}")
                .bytes()
                .map(u32::from)
                .collect::<Vec<_>>()
        );
        assert!(host.events.iter().all(|event| event["phase"] != "thinking"));
        Ok(())
    }

    #[test]
    fn after_two_forced_closes_the_step_thinks_freely() -> Result<()> {
        let mut steps = Vec::new();
        for _ in 0..2 {
            let mut thinking = answer("");
            thinking.raw = "t".into();
            thinking.cut = Some("</think>");
            steps.push(thinking);
        }
        steps.push(answer("ok"));
        let mut host = ScriptHost::new(steps);
        let mut settings = turn(100_000, 8);
        settings.reasoning_budget = Some(256);
        run_turn(&mut host, dialogue(), &tools(), &settings)?;
        let budgets: Vec<_> = host.requests.iter().map(|r| r.reasoning_budget).collect();
        assert_eq!(budgets, [Some(256), Some(256), None]);
        Ok(())
    }

    #[test]
    fn a_long_tool_output_keeps_its_start_and_end() -> Result<()> {
        let mut host = ScriptHost::new(vec![call("log"), answer("ok")]);
        let output = format!(
            "{}{}{}",
            "A".repeat(600),
            "M".repeat(2_000),
            "Z".repeat(600)
        );
        let text = output.clone();
        host.results = Box::new(move |_, _| ToolResult {
            text: text.clone(),
            is_error: false,
        });
        // Budget: 8192.min(8000 / 8) = 1000 tokens (bytes here).
        run_turn(&mut host, dialogue(), &tools(), &turn(8_000, 8))?;
        let shown = host.events[2]["text"].as_str().unwrap();
        let (head, tail) = tool_call::kept_token_ranges(output.len(), 1000, 32);
        let omitted = output.len() - head.len() - tail.len();
        assert_eq!(
            shown,
            format!(
                "{}\n[... {omitted} tokens of tool output omitted to fit the context ...]\n{}",
                &output[head], &output[tail]
            )
        );
        assert!(shown.starts_with('A') && shown.ends_with('Z'));
        Ok(())
    }

    #[test]
    fn guidance_yields_between_individual_calls_and_replays_only_finished_tools() -> Result<()> {
        for count in 0..=2 {
            let mut batch = call("first");
            batch.tool_text += &call("second").tool_text;
            let mut host = ScriptHost::new(vec![batch]);
            host.yield_after = Some(count);
            let end = run_turn(&mut host, dialogue(), &tools(), &turn(100_000, 8))?;
            assert_eq!(end.stop_reason, "agent_yield");
            assert_eq!(host.calls.len(), count);
            assert_eq!(host.requests.len(), 1);
            assert_eq!(host.resets, 0);
            assert_eq!(host.forgets, 1);
            let history: Vec<Value> = serde_json::from_str(end.turn_context.as_deref().unwrap())?;
            assert_eq!(history.len(), if count == 0 { 1 } else { count + 2 });
            assert_eq!(
                history[0]["tool_calls"].as_array().map_or(0, Vec::len),
                count
            );
            for (index, expected) in ["first", "second"][..count].iter().enumerate() {
                assert_eq!(history[index + 1]["content"], format!("ran {expected}"));
            }
            assert!(
                history
                    .last()
                    .is_some_and(|m| m["role"] == "assistant" && m.get("tool_calls").is_none())
            );
        }
        Ok(())
    }

    #[test]
    fn guidance_handles_unreadable_calls_without_replaying_synthetic_users() -> Result<()> {
        let malformed = "<tool_call>{\"name\":";
        for count in [0, 1] {
            let mut broken = call("");
            broken.tool_text = malformed.into();
            let mut host = ScriptHost::new(vec![broken, call("read")]);
            host.yield_after = Some(count);
            let end = run_turn(&mut host, dialogue(), &tools(), &turn(100_000, 8))?;
            let history: Vec<Value> = serde_json::from_str(end.turn_context.as_deref().unwrap())?;
            assert_eq!(host.calls.len(), count);
            assert_eq!(host.requests.len(), count + 1);
            assert!(
                history
                    .iter()
                    .all(|m| matches!(m["role"].as_str(), Some("assistant" | "tool")))
            );
            assert!(!history.iter().any(|m| m["content"] == malformed));
            if count == 1 {
                assert_eq!(history[1]["name"], "invalid_tool_call");
                assert_eq!(history[1]["content"], host.events[2]["text"]);
                assert_eq!(history[3]["content"], "ran read");
            }
        }
        Ok(())
    }

    #[test]
    fn simultaneous_stop_and_guidance_cancel_at_model_step_end() {
        let mut broken = call("");
        broken.tool_text = "<tool_call>{\"name\":".into();
        for step in [answer("finished"), broken, call("write")] {
            let mut host = ScriptHost::new(vec![step]);
            host.yield_after = Some(0);
            host.cancelled.store(true, Ordering::Relaxed);
            let error = run_turn(&mut host, dialogue(), &tools(), &turn(100_000, 8)).unwrap_err();
            assert_eq!(error.to_string(), "generation cancelled");
            assert_eq!(host.resets, 1);
            assert!(host.calls.is_empty());
            assert_eq!(host.requests.len(), 1);
        }
    }

    #[test]
    fn a_cancellation_between_calls_stops_the_turn_and_resets_the_model() {
        let two = format!(
            "<tool_call>{}</tool_call><tool_call>{}</tool_call>",
            json!({"name":"javascript", "arguments":{"code":"1"}}),
            json!({"name":"javascript", "arguments":{"code":"2"}})
        );
        let mut step = call("");
        step.tool_text = two;
        let mut host = ScriptHost::new(vec![step]);
        host.cancel_after = Some(1);
        let error = run_turn(&mut host, dialogue(), &tools(), &turn(100_000, 8)).unwrap_err();
        assert_eq!(error.to_string(), "generation cancelled");
        assert_eq!((host.calls.len(), host.resets), (1, 1));
    }

    /// What `validated_bridge_messages` and every chat template need from a stored turn: only
    /// assistant and tool messages, each call followed by its result, an assistant message last.
    fn well_formed(history: &[Value]) {
        let mut owed = 0usize;
        for message in history {
            match message["role"].as_str() {
                Some("assistant") => {
                    assert_eq!(owed, 0, "a call has no result: {history:?}");
                    owed = message["tool_calls"].as_array().map_or(0, Vec::len);
                    assert!(
                        message.get("tool_calls").is_none() || owed > 0,
                        "an empty call list: {history:?}"
                    );
                }
                Some("tool") => {
                    assert!(owed > 0, "a result without its call: {history:?}");
                    owed -= 1;
                }
                other => panic!("role {other:?} in a stored turn"),
            }
        }
        assert_eq!(owed, 0, "the last call has no result: {history:?}");
        assert_eq!(history.last().unwrap()["role"], "assistant");
    }

    fn interrupted(error: &anyhow::Error) -> Option<Vec<Value>> {
        let history: Vec<Value> = serde_json::from_str(interrupted_context(error)?).unwrap();
        well_formed(&history);
        Some(history)
    }

    #[test]
    fn a_turn_stopped_between_calls_keeps_the_calls_that_finished() {
        let two = format!(
            "<tool_call>{}</tool_call><tool_call>{}</tool_call>",
            json!({"name":"javascript", "arguments":{"code":"1"}}),
            json!({"name":"javascript", "arguments":{"code":"2"}})
        );
        let mut step = call("");
        step.answer = "I run both.".into();
        step.tool_text = two;
        let mut host = ScriptHost::new(vec![call("0"), step]);
        host.cancel_after = Some(2);
        let error = run_turn(&mut host, dialogue(), &tools(), &turn(100_000, 8)).unwrap_err();
        // The client still reads the same error.
        assert_eq!(error.to_string(), "generation cancelled");
        assert_eq!((host.calls.len(), host.resets), (2, 1));
        let event = failure_event("r1", true, &error);
        assert_eq!(event["type"], "cancelled");
        assert_eq!(event["turn_context"].as_str(), interrupted_context(&error));
        assert!(event.get("message").is_none());
        let call = |code: &str| json!({"type":"function", "function":{"name":"javascript", "arguments":{"code":code}}});
        assert_eq!(
            interrupted(&error).unwrap(),
            [
                json!({"role":"assistant", "content":"", "tool_calls":[call("0")]}),
                json!({"role":"tool", "name":"javascript", "content":"ran 0"}),
                // the second call of this message never ran: it is not in the history
                json!({"role":"assistant", "content":"I run both.", "tool_calls":[call("1")]}),
                json!({"role":"tool", "name":"javascript", "content":"ran 1"}),
                json!({"role":"assistant", "content":""}),
            ]
        );
    }

    #[test]
    fn an_engine_error_after_tool_results_keeps_them() {
        let mut host = ScriptHost::new(vec![call("a"), call("b")]);
        host.fail_step = Some(3);
        let error = run_turn(&mut host, dialogue(), &tools(), &turn(100_000, 8)).unwrap_err();
        assert_eq!(error.to_string(), "Metal device lost");
        let event = failure_event("r1", false, &error);
        assert_eq!(
            (&event["type"], &event["message"]),
            (&json!("error"), &json!("Metal device lost"))
        );
        assert_eq!(event["turn_context"].as_str(), interrupted_context(&error));
        let history = interrupted(&error).unwrap();
        assert_eq!(history.len(), 5);
        assert_eq!(history[1]["content"], "ran a");
        assert_eq!(history[3]["content"], "ran b");
        assert_eq!(history[4], json!({"role":"assistant", "content":""}));
        // The next turn is given exactly these messages after the user's request.
        let mut next = dialogue();
        next.extend(history);
        next.push(json!({"role":"user", "content":"continue"}));
        let mut resumed = ScriptHost::new(vec![answer("Done.")]);
        let mut settings = turn(100_000, 8);
        settings.original_length = next.len();
        run_turn(&mut resumed, next.clone(), &tools(), &settings).unwrap();
        assert_eq!(resumed.requests[0].dialogue, next);
    }

    #[test]
    fn a_turn_stopped_before_any_result_carries_no_history() {
        // Stop during the first step, an engine failure in it, and Stop before the first call.
        let mut failing = ScriptHost::new(vec![]);
        failing.fail_step = Some(1);
        let error = run_turn(&mut failing, dialogue(), &tools(), &turn(100_000, 8)).unwrap_err();
        assert_eq!(error.to_string(), "Metal device lost");
        assert!(interrupted_context(&error).is_none());
        let mut step = call("write");
        step.answer = "Writing the file.".into();
        let mut stopped = ScriptHost::new(vec![step]);
        stopped.cancelled.store(true, Ordering::Relaxed);
        let error = run_turn(&mut stopped, dialogue(), &tools(), &turn(100_000, 8)).unwrap_err();
        assert_eq!(error.to_string(), "generation cancelled");
        assert!(stopped.calls.is_empty());
        assert!(interrupted_context(&error).is_none());
        // Without tool results the events are exactly the ones older clients know.
        assert_eq!(
            failure_event("r1", true, &error),
            json!({"type":"cancelled", "request_id":"r1"})
        );
        assert_eq!(
            failure_event("r1", false, &error),
            json!({"type":"error", "request_id":"r1", "message":"generation cancelled"})
        );
    }

    #[test]
    fn an_unreadable_call_before_the_stop_is_stored_as_the_ui_showed_it() {
        let mut broken = call("");
        broken.tool_text = "<tool_call>{\"name\":".into();
        let mut host = ScriptHost::new(vec![call("a"), broken]);
        host.fail_step = Some(3);
        let error = run_turn(&mut host, dialogue(), &tools(), &turn(100_000, 8)).unwrap_err();
        let history = interrupted(&error).unwrap();
        assert_eq!(history.len(), 5);
        assert_eq!(
            history[2]["tool_calls"][0]["function"]["name"],
            "invalid_tool_call"
        );
        assert_eq!(history[3]["name"], "invalid_tool_call");
        assert!(
            history[3]["content"]
                .as_str()
                .unwrap()
                .contains("could not be read")
        );
    }

    #[test]
    fn an_interrupted_history_never_holds_a_call_without_its_result() {
        // Every shape the loop can be stopped in: `before` finished exchanges, then a message
        // with `calls` calls of which `results` got a result, with or without text.
        let call = |index: usize| json!({"type":"function", "function":{"name":"javascript", "arguments":{"code":index.to_string()}}});
        let result = |index: usize| json!({"role":"tool", "name":"javascript", "content":format!("ran {index}")});
        for before in 0..3 {
            for calls in 0..4 {
                for results in 0..=calls {
                    for text in ["", "Next."] {
                        let mut transcript = Vec::new();
                        for index in 0..before {
                            transcript.push(json!({"role":"assistant", "content":"", "tool_calls":[call(index)]}));
                            transcript.push(result(index));
                        }
                        transcript.push(json!({
                            "role":"assistant", "content":text,
                            "tool_calls":(0..calls).map(|index| call(10 + index)).collect::<Vec<_>>()
                        }));
                        transcript.extend((0..results).map(|index| result(10 + index)));
                        let stored = interrupted_history(&transcript);
                        let finished = before + results;
                        assert_eq!(stored.is_some(), finished > 0, "{before} {calls} {results}");
                        let Some(stored) = stored else { continue };
                        let history: Vec<Value> = serde_json::from_str(&stored).unwrap();
                        well_formed(&history);
                        let kept: Vec<_> = history
                            .iter()
                            .filter(|message| message["role"] == "tool")
                            .map(|message| message["content"].as_str().unwrap().to_owned())
                            .collect();
                        let expected: Vec<_> = (0..before)
                            .chain((0..results).map(|index| 10 + index))
                            .map(|index| format!("ran {index}"))
                            .collect();
                        assert_eq!(kept, expected, "every finished result, in order");
                        // Text written before calls that never ran is kept as plain text.
                        let texts = history.iter().filter(|m| m["content"] == "Next.").count();
                        assert_eq!(texts, usize::from(!text.is_empty()));
                    }
                }
            }
        }
        // A turn without tools (JSON retries) and an empty one carry nothing.
        assert!(interrupted_history(&[]).is_none());
        assert!(interrupted_history(&[json!({"role":"assistant", "content":"nope"})]).is_none());
    }

    #[test]
    fn invalid_json_is_retried_twice_then_fails() -> Result<()> {
        let schema = Schema::parse(r#"{"type":"object","required":["a"]}"#)?;
        let mut host = ScriptHost::new(vec![answer("nope"), answer("{}"), answer(r#"{"a":1}"#)]);
        let mut settings = turn(100_000, 8);
        settings.schema = Some(&schema);
        let end = run_turn(&mut host, dialogue(), &[], &settings)?;
        assert_eq!(host.forgets, 2);
        assert!(host.requests.iter().all(|request| request.buffered));
        assert_eq!(host.requests[2].dialogue.len(), 6);
        assert!(end.json_answer && host.shown == 1);
        assert_eq!(host.events.last().unwrap()["phase"], "answer");
        let mut failing = ScriptHost::new(vec![answer("a"), answer("b"), answer("c")]);
        let error = run_turn(&mut failing, dialogue(), &[], &settings).unwrap_err();
        assert!(
            error
                .to_string()
                .starts_with("JSON validation failed after 3 attempts")
        );
        // The validation retries of a JSON answer are not tool history.
        assert!(interrupted_context(&error).is_none());
        Ok(())
    }

    #[test]
    fn a_finite_budget_is_shared_by_the_steps_and_exhaustion_is_an_error() {
        let mut host = ScriptHost::new(vec![call("1234567890"), answer("x")]);
        let mut settings = turn(100_000, 8);
        let first_step = call("1234567890").raw.len() as i64;
        settings.max_tokens = first_step;
        let error = run_turn(&mut host, dialogue(), &tools(), &settings).unwrap_err();
        assert_eq!(error.to_string(), "Generation token budget exhausted");
        assert_eq!(host.requests[0].max_tokens, first_step);
    }

    /// `n` calls whose code (1 900 characters) pruning cannot shorten, with tiny results.
    fn heavy(n: usize) -> ScriptHost {
        let mut steps: Vec<_> = (0..n)
            .map(|i| call(&format!("{i}{}", "c".repeat(1_900))))
            .collect();
        steps.push(answer("done"));
        let mut host = ScriptHost::new(steps);
        host.results = Box::new(|_, _| ToolResult {
            text: "ok".into(),
            is_error: false,
        });
        host
    }

    fn summary_requests(host: &ScriptHost) -> Vec<&Recorded> {
        host.requests
            .iter()
            .filter(|r| r.dialogue.last().unwrap()["content"] == agent_loop::SUMMARY_REQUEST)
            .collect()
    }

    fn carries_summary(dialogue: &[Value], text: &str) -> bool {
        dialogue.iter().any(|message| {
            message["role"] == "tool"
                && message["content"].as_str().is_some_and(|content| {
                    content.starts_with(agent_loop::SUMMARY_MARKER) && content.contains(text)
                })
        })
    }

    #[test]
    fn old_exchanges_are_summarized_when_pruning_cannot_free_enough() -> Result<()> {
        let mut host = heavy(8);
        host.summaries.push_back(answer("- did steps 0 to 5"));
        let end = run_turn(&mut host, dialogue(), &tools(), &turn(20_000, 50))?;
        let asked = summary_requests(&host);
        assert_eq!(asked.len(), 1);
        let request = asked[0];
        assert!(request.buffered && request.max_tokens == agent_loop::SUMMARY_MAX_TOKENS as i64);
        assert_eq!(
            request.reasoning_budget,
            Some(agent_loop::SUMMARY_REASONING_BUDGET)
        );
        // Its prompt is the dialogue up to the cut (six exchanges), then the request.
        assert_eq!(request.dialogue.len(), 2 + 12 + 1);
        let next = host
            .requests
            .iter()
            .position(|r| std::ptr::eq(r, request))
            .unwrap()
            + 1;
        let after = &host.requests[next].dialogue;
        assert_eq!(
            after.len(),
            2 + 4,
            "two newest exchanges kept, the rest summarized"
        );
        assert!(carries_summary(after, "- did steps 0 to 5"));
        assert_eq!(after[..2], dialogue()[..]);
        let statuses = host.statuses();
        assert!(statuses.contains(&"Summarizing 12 earlier messages to free context".to_owned()));
        assert!(
            statuses
                .iter()
                .any(|s| s.starts_with("Context summarized: 12 earlier messages replaced"))
        );
        assert_eq!((end.assistant.as_str(), end.stop_reason), ("done", "eos"));
        let context: Vec<Value> = serde_json::from_str(end.turn_context.as_deref().unwrap())?;
        assert!(
            carries_summary(&context, "- did steps 0 to 5"),
            "the next turns see it too"
        );
        assert_eq!(host.calls.len(), 8);
        Ok(())
    }

    #[test]
    fn guidance_after_compaction_preserves_replayable_tool_history() -> Result<()> {
        let mut host = heavy(12);
        host.summaries.push_back(answer("- changed settings"));
        host.yield_after = Some(9);
        let end = run_turn(&mut host, dialogue(), &tools(), &turn(20_000, 50))?;
        assert_eq!((end.stop_reason, host.calls.len()), ("agent_yield", 9));
        let context: Vec<Value> = serde_json::from_str(end.turn_context.as_deref().unwrap())?;
        assert!(carries_summary(&context, "- changed settings"));
        assert!(
            context
                .iter()
                .all(|m| matches!(m["role"].as_str(), Some("assistant" | "tool")))
        );
        Ok(())
    }

    #[test]
    fn an_unusable_summary_leaves_the_dialogue_unchanged_and_waits_before_retrying() -> Result<()> {
        let mut cut = answer("- half");
        cut.stop = GenerationStopReason::MaxTokens;
        // Text and a call: the model did not just summarize.
        let mut talking = call("ls");
        talking.answer = "- notes".into();
        for bad in [call("ls"), talking, answer("   "), cut] {
            let mut host = heavy(9);
            host.summaries.push_back(bad);
            let end = run_turn(&mut host, dialogue(), &tools(), &turn(20_000, 50))?;
            assert_eq!(
                summary_requests(&host).len(),
                1,
                "one exchange later is too soon to retry"
            );
            assert!(host.statuses().contains(
                &"The summary could not be used; the earlier messages are kept".to_owned()
            ));
            assert!(
                host.requests
                    .iter()
                    .all(|r| !carries_summary(&r.dialogue, ""))
            );
            assert_eq!(end.assistant, "done");
        }
        Ok(())
    }

    #[test]
    fn a_summary_whose_thinking_is_cut_is_written_after_the_summary_note() -> Result<()> {
        let mut thinking = answer("");
        thinking.raw = "<think>x".into();
        thinking.cut = Some("</think>");
        let mut host = heavy(8);
        host.summaries.extend([thinking, answer("- s")]);
        run_turn(&mut host, dialogue(), &tools(), &turn(20_000, 50))?;
        let asked = summary_requests(&host);
        assert_eq!(asked.len(), 2);
        let mut expected: Vec<u32> = "<think>x".bytes().map(u32::from).collect();
        expected.extend(
            agent_loop::reasoning_continuation("</think>", true, true, false)
                .bytes()
                .map(u32::from),
        );
        assert_eq!(asked[1].continuation, expected);
        assert_eq!(
            asked[1].max_tokens,
            (agent_loop::SUMMARY_MAX_TOKENS - "<think>x".len()) as i64
        );
        assert!(
            host.requests
                .iter()
                .any(|r| carries_summary(&r.dialogue, "- s"))
        );
        Ok(())
    }

    #[test]
    fn a_full_context_with_nothing_to_prune_is_summarized_and_the_step_retried() -> Result<()> {
        let mut host = heavy(6);
        let done = host.steps.pop_back().unwrap();
        let mut full = call("cut");
        full.context_used = Some(20_000);
        host.steps.extend([full, done]);
        host.summaries.push_back(answer("- six steps"));
        let end = run_turn(&mut host, dialogue(), &tools(), &turn(20_000, 50))?;
        assert_eq!(summary_requests(&host).len(), 1);
        let last = &host.requests.last().unwrap().dialogue;
        assert!(carries_summary(last, "- six steps"));
        assert_eq!(host.calls.len(), 6, "the cut step's call never ran");
        assert!(!end.context_full && end.assistant == "done");
        Ok(())
    }
}
