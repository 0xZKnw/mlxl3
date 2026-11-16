//! One user turn on top of the bridge: validated JSON replies and the MCP / Agent tool loop.
//!
//! The Mac engine runs these loops in Rust next to MLX (`native/src/main.rs`
//! `bridge_generate`, origin/main v2.0.4). The Windows engine is a Python ExLlamaV3 worker, so
//! the same loops run here, in the app, and drive the worker one generation round at a time.
//! Rules kept from the Mac:
//! - JSON: no constrained decoding; the answer is buffered, validated by [`Schema`], and retried
//!   with feedback for at most 3 attempts sharing one token budget. The delivered text is the
//!   compact re-serialized JSON.
//! - Tools: at most 5 generation rounds; a call is executed only when the answer starts with a
//!   tool marker (prose mixed with a call executes nothing); a tool outside `allowed` is a hard
//!   error; long results are truncated to a context-relative budget.
//! - JSON and tools are exclusive.
//!
//! Events keep the bridge shape (`generation_status`, `delta`, `done`, `cancelled`, `error`)
//! plus `tool_start` / `tool_result`, all carrying the turn's `request_id`.

use crate::mcp::ToolResult;
use crate::protocol::{Message, Sampling};
use crate::server::Engine;
use crate::structured::{Schema, remaining_tokens, tool_allowed};
use crate::tool_call;
use anyhow::{Context, Result, bail, ensure};
use serde_json::{Value, json};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::{Duration, Instant};

pub const MAX_TOOL_ROUNDS: usize = 5;
pub const JSON_ATTEMPTS: usize = 3;
/// Same idle limit as the server: a round that reports nothing for this long fails.
pub const ROUND_IDLE: Duration = Duration::from_secs(600);
const POLL: Duration = Duration::from_millis(50);
const CANCEL_GRACE: Duration = Duration::from_secs(5);

/// Executes tool calls for the loop. Implemented by the MCP manager and the Agent workspace.
pub trait ToolBox: Send + Sync + 'static {
    /// Server that owns `name` (shown as "MCP · server").
    fn server_of(&self, name: &str) -> String;
    fn call(&self, name: &str, arguments: Value, cancelled: &AtomicBool) -> ToolResult;
}

pub struct TurnRequest {
    pub messages: Vec<Message>,
    pub sampling: Sampling,
    /// Raw JSON schema text; validated before any round runs.
    pub schema: Option<String>,
    /// OpenAI function tools offered to the model.
    pub tools: Vec<Value>,
    /// Project allow-list of public tool names; `None` = every offered tool.
    pub allowed: Option<Vec<String>>,
    /// Context size in tokens, for the tool-result budget (0 = unknown).
    pub context_limit: usize,
}

pub struct Turn {
    pub events: Receiver<Value>,
    pub cancel: Arc<AtomicBool>,
}

/// Tool results longer than this many tokens are cut (Mac: min(8192, max(ctx/8, 256)) tokens).
/// Without a tokenizer here, 4 bytes stand for one token.
pub fn tool_budget_bytes(context_limit: usize) -> usize {
    let tokens = if context_limit == 0 {
        8192
    } else {
        (context_limit / 8).clamp(256, 8192)
    };
    tokens * 4
}

pub fn truncate_result(text: &str, budget: usize) -> String {
    if text.len() <= budget {
        return text.to_owned();
    }
    let mut end = budget;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}\n[Tool output truncated to fit context]", &text[..end])
}

/// Mac `Schema::instruction` appended to the system message (or a new one at index 0).
fn with_instruction(messages: &[Message], instruction: &str) -> Vec<Message> {
    let mut out = messages.to_vec();
    match out.first_mut() {
        Some(first) if first.role == "system" => {
            first.content = if first.content.trim().is_empty() {
                instruction.to_owned()
            } else {
                format!("{}\n\n{instruction}", first.content)
            };
        }
        _ => out.insert(0, Message::text("system", instruction)),
    }
    out
}

/// Reasoning written before the answer (`<think>…</think>`), kept visible in JSON mode.
fn reasoning_prefix(raw: &str) -> &str {
    match raw.rfind("</think>") {
        Some(end) => &raw[..end + "</think>".len()],
        None => "",
    }
}

/// Visible deltas of a tools round. Reasoning (`…</think>`) is shown as it streams; the answer
/// after it goes through the Mac [`tool_call::StreamFilter`], which hides tool markup. Templates
/// that open `<think>` in the prompt only stream the closing tag, so both filters are needed.
struct VisibleText {
    acc: String,
    fed: usize,
    split: bool,
    reasoning: tool_call::StreamFilter,
    answer: tool_call::StreamFilter,
}

const THINK_END: &str = "</think>";

impl VisibleText {
    fn new() -> Self {
        Self {
            acc: String::new(),
            fed: 0,
            split: false,
            reasoning: tool_call::StreamFilter::new(),
            answer: tool_call::StreamFilter::new(),
        }
    }

    fn feed(&mut self, text: &str) -> Vec<String> {
        if self.split {
            return self.answer.feed(text);
        }
        self.acc.push_str(text);
        if let Some(position) = self.acc.find(THINK_END) {
            let cut = position + THINK_END.len();
            let mut out = self.reasoning.feed(&self.acc[self.fed..cut]);
            self.split = true;
            let rest = self.acc[cut..].to_owned();
            out.extend(self.answer.feed(&rest));
            return out;
        }
        // Hold back a possible partial "</think>" at the end of the chunk.
        let mut safe = self.acc.len();
        for keep in (1..THINK_END.len()).rev() {
            if self.acc.len() >= keep
                && self.acc.is_char_boundary(self.acc.len() - keep)
                && THINK_END.starts_with(&self.acc[self.acc.len() - keep..])
            {
                safe = self.acc.len() - keep;
                break;
            }
        }
        let out = self.reasoning.feed(&self.acc[self.fed..safe]);
        self.fed = safe;
        out
    }

    fn is_preparing_tool(&self) -> bool {
        if self.split {
            self.answer.is_preparing_tool()
        } else {
            self.reasoning.is_preparing_tool()
        }
    }

    fn finish(&mut self) -> Vec<String> {
        if self.split {
            return self.answer.finish();
        }
        let rest = self.acc[self.fed..].to_owned();
        let mut out = self.reasoning.feed(&rest);
        out.extend(self.reasoning.finish());
        out
    }
}

struct Round {
    raw: String,
    stats: Value,
}

enum Outcome {
    Done(Round),
    Cancelled,
    Failed(String),
}

struct Driver<'a> {
    engine: &'a dyn Engine,
    id: String,
    cancel: &'a AtomicBool,
    out: &'a Sender<Value>,
}

impl Driver<'_> {
    fn emit(&self, mut event: Value) {
        event["request_id"] = self.id.clone().into();
        let _ = self.out.send(event);
    }

    /// One generation on the worker. `visible` decides what reaches the UI as deltas.
    fn round(
        &self,
        index: usize,
        messages: &[Message],
        sampling: &Sampling,
        tools: &[Value],
        mut visible: impl FnMut(&str) -> Vec<Value>,
    ) -> Result<Outcome> {
        let round_id = if index == 0 {
            format!("{}~", self.id)
        } else {
            format!("{}~{index}", self.id)
        };
        let events = self.engine.submit(&round_id, messages, sampling, tools)?;
        let mut raw = String::new();
        let mut last = Instant::now();
        let mut cancel_sent: Option<Instant> = None;
        loop {
            if self.cancel.load(Ordering::Relaxed) && cancel_sent.is_none() {
                self.engine.cancel(&round_id);
                cancel_sent = Some(Instant::now());
            }
            if cancel_sent.is_some_and(|at| at.elapsed() > CANCEL_GRACE) {
                return Ok(Outcome::Cancelled);
            }
            let event = match events.recv_timeout(POLL) {
                Ok(event) => event,
                Err(RecvTimeoutError::Timeout) => {
                    ensure!(last.elapsed() < ROUND_IDLE, "the engine stopped responding");
                    continue;
                }
                Err(RecvTimeoutError::Disconnected) => bail!("the engine exited"),
            };
            last = Instant::now();
            match event["type"].as_str().unwrap_or_default() {
                "delta" => {
                    let text = event["text"].as_str().unwrap_or_default();
                    raw.push_str(text);
                    for event in visible(text) {
                        self.emit(event);
                    }
                }
                "generation_status" => {
                    let mut status = event.clone();
                    status.as_object_mut().map(|o| o.remove("request_id"));
                    self.emit(status);
                }
                "done" => {
                    let mut stats = event.clone();
                    if let Some(object) = stats.as_object_mut() {
                        object.remove("type");
                        object.remove("request_id");
                    }
                    return Ok(Outcome::Done(Round { raw, stats }));
                }
                "cancelled" => return Ok(Outcome::Cancelled),
                "error" => {
                    return Ok(Outcome::Failed(
                        event["message"]
                            .as_str()
                            .unwrap_or("engine error")
                            .to_owned(),
                    ));
                }
                _ => {}
            }
        }
    }
}

fn add_stats(total: &mut Value, round: &Value) {
    if total.is_null() {
        *total = round.clone();
        return;
    }
    for key in [
        "generated_tokens",
        "decode_seconds",
        "wall_seconds",
        "prefill_seconds",
    ] {
        if let (Some(a), Some(b)) = (total[key].as_f64(), round[key].as_f64()) {
            total[key] = if key == "generated_tokens" {
                json!((a + b) as u64)
            } else {
                json!(a + b)
            };
        }
    }
    for key in ["stop_reason", "prompt_tokens", "cached_tokens"] {
        if !round[key].is_null() {
            total[key] = round[key].clone();
        }
    }
}

fn generated(stats: &Value) -> usize {
    stats["generated_tokens"].as_u64().unwrap_or(0) as usize
}

fn finish(driver: &Driver, mut stats: Value, extra: Value) {
    if !stats.is_object() {
        stats = json!({});
    }
    stats["type"] = "done".into();
    if let Some(extra) = extra.as_object() {
        for (key, value) in extra {
            stats[key] = value.clone();
        }
    }
    driver.emit(stats);
}

fn run_json(driver: &Driver, request: &TurnRequest, schema: &Schema) -> Result<()> {
    let mut dialogue = with_instruction(&request.messages, &schema.instruction());
    let mut total = Value::Null;
    let mut used = 0usize;
    for attempt in 0..JSON_ATTEMPTS {
        let remaining = remaining_tokens(request.sampling.max_tokens, used)
            .context("JSON token budget exhausted")?;
        let sampling = Sampling {
            max_tokens: remaining,
            ..request.sampling.clone()
        };
        if attempt > 0 {
            driver
                .emit(json!({"type":"generation_status","phase":"prefill","text":"Retrying JSON"}));
        }
        // Answers are buffered: only the validated JSON is delivered.
        let round = match driver.round(attempt, &dialogue, &sampling, &[], |_| vec![])? {
            Outcome::Done(round) => round,
            Outcome::Cancelled => {
                driver.emit(json!({"type":"cancelled"}));
                return Ok(());
            }
            Outcome::Failed(message) => bail!(message),
        };
        used += generated(&round.stats);
        add_stats(&mut total, &round.stats);
        let answer = tool_call::assistant_context(&round.raw);
        match schema.validate_cancellable(answer, driver.cancel) {
            Ok(canonical) => {
                let prefix = reasoning_prefix(&round.raw);
                let text = if prefix.is_empty() {
                    canonical
                } else {
                    format!("{prefix}\n{canonical}")
                };
                driver.emit(json!({"type":"delta","text":text}));
                finish(
                    driver,
                    total,
                    json!({"tool_rounds":0,"schema":"valid","attempts":attempt + 1}),
                );
                return Ok(());
            }
            Err(error) if attempt + 1 < JSON_ATTEMPTS => {
                dialogue.push(Message::text("assistant", round.raw));
                dialogue.push(Message::text(
                    "user",
                    format!(
                        "The previous response failed JSON validation: {error}. Return only valid JSON conforming to the schema."
                    ),
                ));
            }
            Err(error) => bail!("JSON validation failed after {JSON_ATTEMPTS} attempts: {error}"),
        }
    }
    unreachable!("the last attempt returns")
}

fn run_tools(driver: &Driver, request: &TurnRequest, toolbox: &dyn ToolBox) -> Result<()> {
    let tools: Vec<Value> = request
        .tools
        .iter()
        .filter(|tool| {
            tool["function"]["name"]
                .as_str()
                .is_some_and(|name| tool_allowed(name, request.allowed.as_deref()))
        })
        .cloned()
        .collect();
    let mut dialogue = request.messages.clone();
    let original = dialogue.len();
    let mut total = Value::Null;
    let mut used = 0usize;
    let budget = tool_budget_bytes(request.context_limit);
    for round_index in 0..MAX_TOOL_ROUNDS {
        let remaining = remaining_tokens(request.sampling.max_tokens, used)
            .context("token budget exhausted")?;
        let sampling = Sampling {
            max_tokens: remaining,
            ..request.sampling.clone()
        };
        if round_index > 0 {
            driver.emit(
                json!({"type":"generation_status","phase":"prefill","text":"Reading MCP results"}),
            );
        }
        let mut filter = VisibleText::new();
        let mut preparing = false;
        let outcome = driver.round(round_index, &dialogue, &sampling, &tools, |text| {
            let mut events: Vec<Value> = filter
                .feed(text)
                .into_iter()
                .filter(|piece| !piece.is_empty())
                .map(|piece| json!({"type":"delta","text":piece}))
                .collect();
            if filter.is_preparing_tool() && !preparing {
                preparing = true;
                events.push(
                    json!({"type":"generation_status","phase":"tool","text":"Preparing tool call"}),
                );
            }
            events
        })?;
        let round = match outcome {
            Outcome::Done(round) => round,
            Outcome::Cancelled => {
                driver.emit(json!({"type":"cancelled"}));
                return Ok(());
            }
            Outcome::Failed(message) => bail!(message),
        };
        for piece in filter.finish() {
            if !piece.is_empty() {
                driver.emit(json!({"type":"delta","text":piece}));
            }
        }
        used += generated(&round.stats);
        add_stats(&mut total, &round.stats);
        let calls = if tools.is_empty() {
            vec![]
        } else {
            tool_call::parse(&round.raw)?
        };
        let context_full = round.stats["stop_reason"] == "max_new_tokens";
        if calls.is_empty() || context_full {
            let turn_context = (round_index > 0).then(|| {
                let mut steps: Vec<Value> = dialogue[original..]
                    .iter()
                    .map(|m| serde_json::to_value(m).unwrap_or_default())
                    .collect();
                steps.push(json!({"role":"assistant","content":round.raw}));
                Value::Array(steps)
            });
            finish(
                driver,
                total,
                json!({"tool_rounds": round_index, "turn_context": turn_context}),
            );
            return Ok(());
        }
        dialogue.push(Message {
            tool_calls: calls
                .iter()
                .map(|call| json!({"type":"function","function":{"name":call.name,"arguments":call.arguments}}))
                .collect(),
            ..Message::text("assistant", tool_call::without_calls(&round.raw))
        });
        for call in calls {
            let call_id = format!("mcp-{round_index}-{}-{}", dialogue.len(), call.name);
            let server = toolbox.server_of(&call.name);
            driver.emit(json!({"type":"tool_start","tool_call_id":call_id,"tool_name":call.name,"server_name":server}));
            ensure!(
                tool_allowed(&call.name, request.allowed.as_deref())
                    && tools
                        .iter()
                        .any(|t| t["function"]["name"] == call.name.as_str()),
                "MCP tool is not allowed in this project: {}",
                call.name
            );
            let result = toolbox.call(&call.name, call.arguments, driver.cancel);
            if driver.cancel.load(Ordering::Relaxed) {
                driver.emit(json!({"type":"cancelled"}));
                return Ok(());
            }
            let text = truncate_result(&result.text, budget);
            driver.emit(json!({"type":"tool_result","tool_call_id":call_id,"tool_name":call.name,"server_name":server,"text":text,"is_error":result.is_error}));
            dialogue.push(Message {
                name: Some(call.name.clone()),
                ..Message::text("tool", text)
            });
        }
    }
    bail!("the model exceeded the limit of {MAX_TOOL_ROUNDS} consecutive MCP tool rounds")
}

/// Check what can fail before inference (Mac: the bridge checks the schema first).
pub fn admit(request: &TurnRequest) -> Result<Option<Schema>> {
    let schema = match request.schema.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(text) => Some(Schema::parse(text)?),
    };
    ensure!(
        schema.is_none() || request.tools.is_empty(),
        "JSON output and MCP must be used separately"
    );
    ensure!(!request.messages.is_empty(), "conversation is empty");
    request.sampling.validate()?;
    Ok(schema)
}

/// Run a turn on the calling thread; events (tagged `request_id`) go to `out`.
pub fn run(
    engine: &dyn Engine,
    toolbox: Option<&dyn ToolBox>,
    request_id: &str,
    request: TurnRequest,
    schema: Option<Schema>,
    cancel: &AtomicBool,
    out: &Sender<Value>,
) {
    let driver = Driver {
        engine,
        id: request_id.to_owned(),
        cancel,
        out,
    };
    let result = match (&schema, toolbox) {
        (Some(schema), _) => run_json(&driver, &request, schema),
        (None, Some(toolbox)) if !request.tools.is_empty() => run_tools(&driver, &request, toolbox),
        (None, _) => run_tools(
            &driver,
            &TurnRequest {
                tools: vec![],
                ..request
            },
            &NoTools,
        ),
    };
    if let Err(error) = result {
        driver.emit(json!({"type":"error","message":format!("{error:#}")}));
    }
}

/// Run a turn on its own thread. Errors before the first round are returned directly.
pub fn start(
    engine: Arc<dyn Engine>,
    toolbox: Option<Arc<dyn ToolBox>>,
    request_id: &str,
    request: TurnRequest,
) -> Result<Turn> {
    ensure!(!request_id.is_empty(), "request id is empty");
    let schema = admit(&request)?;
    let (out, events) = channel();
    let cancel = Arc::new(AtomicBool::new(false));
    let flag = cancel.clone();
    let id = request_id.to_owned();
    std::thread::spawn(move || {
        run(
            engine.as_ref(),
            toolbox.as_deref(),
            &id,
            request,
            schema,
            &flag,
            &out,
        );
    });
    Ok(Turn { events, cancel })
}

struct NoTools;

impl ToolBox for NoTools {
    fn server_of(&self, _: &str) -> String {
        String::new()
    }
    fn call(&self, name: &str, _: Value, _: &AtomicBool) -> ToolResult {
        ToolResult {
            text: format!("Unknown MCP tool: {name}"),
            is_error: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// Replies with scripted rounds and records what each round received.
    type Seen = (String, Vec<Message>, Sampling, usize);

    struct Scripted {
        replies: Mutex<Vec<Vec<Value>>>,
        seen: Mutex<Vec<Seen>>,
        cancelled: Mutex<Vec<String>>,
        open: Mutex<Vec<Sender<Value>>>,
    }

    impl Scripted {
        fn new(rounds: Vec<Vec<Value>>) -> Arc<Self> {
            Arc::new(Self {
                replies: Mutex::new(rounds.into_iter().rev().collect()),
                seen: Mutex::new(vec![]),
                cancelled: Mutex::new(vec![]),
                open: Mutex::new(vec![]),
            })
        }
    }

    impl Engine for Scripted {
        fn model(&self) -> Option<String> {
            Some("m".into())
        }
        fn submit(
            &self,
            request_id: &str,
            messages: &[Message],
            sampling: &Sampling,
            tools: &[Value],
        ) -> Result<Receiver<Value>> {
            self.seen.lock().unwrap().push((
                request_id.into(),
                messages.to_vec(),
                sampling.clone(),
                tools.len(),
            ));
            let (tx, rx) = channel();
            let round = self
                .replies
                .lock()
                .unwrap()
                .pop()
                .context("no scripted round")?;
            for event in round {
                tx.send(event).unwrap();
            }
            // Like the app's route table, the sender lives until the round ends.
            self.open.lock().unwrap().push(tx);
            Ok(rx)
        }
        fn cancel(&self, request_id: &str) {
            self.cancelled.lock().unwrap().push(request_id.into());
        }
    }

    fn delta(text: &str) -> Value {
        json!({"type":"delta","text":text})
    }
    fn done(tokens: u64) -> Value {
        json!({"type":"done","generated_tokens":tokens,"prompt_tokens":5,"stop_reason":"eos"})
    }

    struct Echo(Mutex<Vec<(String, Value)>>);
    impl ToolBox for Echo {
        fn server_of(&self, name: &str) -> String {
            name.split('.').next().unwrap_or_default().into()
        }
        fn call(&self, name: &str, arguments: Value, _: &AtomicBool) -> ToolResult {
            self.0
                .lock()
                .unwrap()
                .push((name.into(), arguments.clone()));
            ToolResult {
                text: format!("result of {name}"),
                is_error: false,
            }
        }
    }

    fn collect(turn: Turn) -> Vec<Value> {
        let mut out = vec![];
        while let Ok(event) = turn.events.recv_timeout(Duration::from_secs(5)) {
            let last = matches!(event["type"].as_str(), Some("done" | "error" | "cancelled"));
            out.push(event);
            if last {
                break;
            }
        }
        out
    }

    fn request(schema: Option<&str>, tools: Vec<Value>, max_tokens: i64) -> TurnRequest {
        TurnRequest {
            messages: vec![
                Message::text("system", "be brief"),
                Message::text("user", "hi"),
            ],
            sampling: Sampling {
                max_tokens,
                ..Sampling::default()
            },
            schema: schema.map(str::to_owned),
            tools,
            allowed: None,
            context_limit: 4096,
        }
    }

    fn tool(name: &str) -> Value {
        json!({"type":"function","function":{"name":name,"parameters":{"type":"object"}}})
    }

    #[test]
    fn plain_turn_streams_and_finishes() {
        let engine = Scripted::new(vec![vec![delta("Hel"), delta("lo"), done(2)]]);
        let events = collect(start(engine.clone(), None, "r1", request(None, vec![], -1)).unwrap());
        let text: String = events
            .iter()
            .filter(|e| e["type"] == "delta")
            .map(|e| e["text"].as_str().unwrap())
            .collect();
        assert_eq!(text, "Hello");
        let last = events.last().unwrap();
        assert_eq!(
            (last["type"].as_str(), last["request_id"].as_str()),
            (Some("done"), Some("r1"))
        );
        assert_eq!(last["tool_rounds"], 0);
        assert!(last["turn_context"].is_null());
        assert_eq!(engine.seen.lock().unwrap()[0].0, "r1~");
    }

    #[test]
    fn json_is_buffered_validated_and_retried_on_one_budget() {
        let schema = r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"]}"#;
        let engine = Scripted::new(vec![
            vec![delta("Sure: {\"a\": 1}"), done(30)],
            vec![delta("<think>fix</think>{ \"a\" : 2 }"), done(20)],
        ]);
        let events = collect(
            start(
                engine.clone(),
                None,
                "j",
                request(Some(schema), vec![], 100),
            )
            .unwrap(),
        );
        let deltas: Vec<&str> = events
            .iter()
            .filter(|e| e["type"] == "delta")
            .map(|e| e["text"].as_str().unwrap())
            .collect();
        assert_eq!(
            deltas,
            ["<think>fix</think>\n{\"a\":2}"],
            "only the validated JSON is delivered"
        );
        let last = events.last().unwrap();
        assert_eq!(
            (
                last["type"].as_str(),
                last["schema"].as_str(),
                last["attempts"].as_u64()
            ),
            (Some("done"), Some("valid"), Some(2))
        );
        assert_eq!(last["generated_tokens"], 50);
        let seen = engine.seen.lock().unwrap();
        assert!(
            seen[0].1[0]
                .content
                .starts_with("be brief\n\nReturn only one JSON value")
        );
        assert_eq!(
            (seen[0].2.max_tokens, seen[1].2.max_tokens),
            (100, 70),
            "shared token budget"
        );
        assert_eq!(seen[1].1.len(), 4);
        assert!(
            seen[1].1[3]
                .content
                .starts_with("The previous response failed JSON validation")
        );
    }

    #[test]
    fn json_fails_after_three_attempts_and_rejects_bad_schemas() {
        let schema = r#"{"type":"object"}"#;
        let engine = Scripted::new(vec![vec![delta("no"), done(1)]; 3]);
        let events =
            collect(start(engine.clone(), None, "j", request(Some(schema), vec![], -1)).unwrap());
        let last = events.last().unwrap();
        assert_eq!(last["type"], "error");
        assert!(
            last["message"]
                .as_str()
                .unwrap()
                .starts_with("JSON validation failed after 3 attempts")
        );
        assert_eq!(engine.seen.lock().unwrap().len(), 3);
        // Exhausted budget stops before a fourth round can start.
        let engine = Scripted::new(vec![vec![delta("no"), done(10)]]);
        let events = collect(start(engine, None, "j", request(Some(schema), vec![], 10)).unwrap());
        assert!(
            events.last().unwrap()["message"]
                .as_str()
                .unwrap()
                .contains("budget exhausted")
        );
        let engine = Scripted::new(vec![]);
        for bad in [r#"[1]"#, r#"{"pattern":"x"}"#, "not json"] {
            assert!(
                start(engine.clone(), None, "j", request(Some(bad), vec![], -1)).is_err(),
                "{bad}"
            );
        }
        assert!(
            start(
                engine,
                None,
                "j",
                request(Some(schema), vec![tool("a.b")], -1)
            )
            .is_err()
        );
    }

    #[test]
    fn tool_loop_runs_calls_and_reports_context() {
        let engine = Scripted::new(vec![
            vec![
                delta("<think>plan</think>"),
                delta(
                    "<tool_call>{\"name\":\"fs.read\",\"arguments\":{\"path\":\"a\"}}</tool_call>",
                ),
                done(9),
            ],
            vec![delta("It says hi."), done(4)],
        ]);
        let toolbox = Arc::new(Echo(Mutex::new(vec![])));
        let events = collect(
            start(
                engine.clone(),
                Some(toolbox.clone()),
                "t",
                request(None, vec![tool("fs.read")], -1),
            )
            .unwrap(),
        );
        let kinds: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert!(
            kinds.contains(&"tool_start") && kinds.contains(&"tool_result"),
            "{kinds:?}"
        );
        let visible: String = events
            .iter()
            .filter(|e| e["type"] == "delta")
            .map(|e| e["text"].as_str().unwrap())
            .collect();
        assert!(
            !visible.contains("<tool_call>"),
            "markup is hidden: {visible}"
        );
        assert!(visible.ends_with("It says hi."));
        let result = events.iter().find(|e| e["type"] == "tool_result").unwrap();
        assert_eq!(
            (result["server_name"].as_str(), result["text"].as_str()),
            (Some("fs"), Some("result of fs.read"))
        );
        assert_eq!(
            toolbox.0.lock().unwrap()[0],
            ("fs.read".into(), json!({"path":"a"}))
        );
        let last = events.last().unwrap();
        assert_eq!(
            (
                last["tool_rounds"].as_u64(),
                last["generated_tokens"].as_u64()
            ),
            (Some(1), Some(13))
        );
        let context = last["turn_context"].as_array().unwrap();
        assert_eq!(context.len(), 3);
        assert_eq!(context[0]["tool_calls"][0]["function"]["name"], "fs.read");
        assert_eq!(
            (context[1]["role"].as_str(), context[1]["name"].as_str()),
            (Some("tool"), Some("fs.read"))
        );
        let seen = engine.seen.lock().unwrap();
        assert_eq!((seen[0].3, seen[1].1.len()), (1, 4));
        assert_eq!(seen[1].0, "t~1");
    }

    #[test]
    fn tool_rules_match_the_mac() {
        // Prose before a marker executes nothing.
        let engine = Scripted::new(vec![vec![
            delta("Let me check. <tool_call>{\"name\":\"fs.read\",\"arguments\":{}}</tool_call>"),
            done(3),
        ]]);
        let toolbox = Arc::new(Echo(Mutex::new(vec![])));
        let events = collect(
            start(
                engine,
                Some(toolbox.clone()),
                "t",
                request(None, vec![tool("fs.read")], -1),
            )
            .unwrap(),
        );
        assert_eq!(events.last().unwrap()["type"], "done");
        assert!(toolbox.0.lock().unwrap().is_empty());
        // A tool outside the project allow-list is a hard error and is never offered.
        let engine = Scripted::new(vec![vec![
            delta("<tool_call>{\"name\":\"fs.read\",\"arguments\":{}}</tool_call>"),
            done(3),
        ]]);
        let mut limited = request(None, vec![tool("fs.read"), tool("web.search")], -1);
        limited.allowed = Some(vec!["web.search".into()]);
        let events = collect(start(engine.clone(), Some(toolbox.clone()), "t", limited).unwrap());
        assert!(
            events.last().unwrap()["message"]
                .as_str()
                .unwrap()
                .contains("not allowed in this project: fs.read")
        );
        assert_eq!(engine.seen.lock().unwrap()[0].3, 1);
        assert!(toolbox.0.lock().unwrap().is_empty());
        // Five rounds of calls is the limit.
        let call = vec![
            delta("<tool_call>{\"name\":\"fs.read\",\"arguments\":{}}</tool_call>"),
            done(1),
        ];
        let engine = Scripted::new(vec![call; 5]);
        let events = collect(
            start(
                engine,
                Some(toolbox.clone()),
                "t",
                request(None, vec![tool("fs.read")], -1),
            )
            .unwrap(),
        );
        assert!(
            events.last().unwrap()["message"]
                .as_str()
                .unwrap()
                .contains("limit of 5 consecutive")
        );
        assert_eq!(toolbox.0.lock().unwrap().len(), 5);
        // Unfinished calls are an error, nothing executed.
        let engine = Scripted::new(vec![vec![delta("<tool_call>{\"name\":"), done(1)]]);
        let events = collect(
            start(
                engine,
                Some(toolbox),
                "t",
                request(None, vec![tool("fs.read")], -1),
            )
            .unwrap(),
        );
        assert_eq!(events.last().unwrap()["type"], "error");
    }

    #[test]
    fn visible_text_hides_markup_after_reasoning_at_any_chunk_boundary() {
        let raw = "<think>a</think>\n<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>";
        let opened = "plan</think><tool_call>{}</tool_call>";
        let plain = "Just text";
        for (text, expected) in [
            (raw, "<think>a</think>"),
            (opened, "plan</think>"),
            (plain, plain),
        ] {
            for split in 1..text.len() {
                if !text.is_char_boundary(split) {
                    continue;
                }
                let mut visible = VisibleText::new();
                let mut shown: String = visible.feed(&text[..split]).concat();
                shown += &visible.feed(&text[split..]).concat();
                shown += &visible.finish().concat();
                assert_eq!(shown.trim_end(), expected, "split at {split} of {text:?}");
            }
        }
    }

    #[test]
    fn engine_errors_cancel_and_truncation() {
        let engine = Scripted::new(vec![vec![
            delta("x"),
            json!({"type":"error","message":"oom"}),
        ]]);
        let events = collect(start(engine, None, "e", request(None, vec![], -1)).unwrap());
        assert_eq!(events.last().unwrap()["message"], "oom");
        // A cancelled turn forwards the request to the current round and reports it.
        let engine = Scripted::new(vec![vec![delta("x")]]);
        let turn = start(engine.clone(), None, "c", request(None, vec![], -1)).unwrap();
        turn.cancel.store(true, Ordering::Relaxed);
        std::thread::sleep(Duration::from_millis(200));
        assert_eq!(engine.cancelled.lock().unwrap().as_slice(), ["c~"]);
        assert_eq!(tool_budget_bytes(0), 32768);
        assert_eq!(tool_budget_bytes(1024), 1024);
        assert_eq!(tool_budget_bytes(1 << 20), 32768);
        assert_eq!(
            truncate_result("héllo", 2),
            "h\n[Tool output truncated to fit context]"
        );
        assert_eq!(truncate_result("ok", 2), "ok");
    }
}
