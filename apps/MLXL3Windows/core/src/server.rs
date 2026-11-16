//! Server mode: an OpenAI-compatible chat API on the model loaded in the app.
//!
//! A port of the Mac `mlxl3 serve` surface (`native/src/openai.rs`): `GET /health`,
//! `GET /v1/models`, `POST /v1/chat/completions` (JSON or SSE). Requests share the app's
//! resident engine, which runs one generation at a time and queues the rest. Listening
//! outside loopback requires an API key. Plain std networking, one request per connection.

use crate::protocol::{Message, Sampling};
use crate::turn;
use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const MAX_BODY: usize = 16 * 1024 * 1024;
pub const MAX_HEAD: usize = 32 * 1024;
pub const MAX_CONNECTIONS: usize = 16;
/// Longest silence accepted from the engine while a request is queued or generating.
pub const EVENT_TIMEOUT: Duration = Duration::from_secs(600);

/// The app side: the loaded model and its resident engine.
pub trait Engine: Send + Sync + 'static {
    /// Name of the loaded model, `None` when nothing is loaded.
    fn model(&self) -> Option<String>;
    /// Queue a generation; bridge events for `request_id` arrive on the receiver.
    fn submit(
        &self,
        request_id: &str,
        messages: &[Message],
        sampling: &Sampling,
        tools: &[Value],
    ) -> Result<Receiver<Value>>;
    fn cancel(&self, request_id: &str);
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub api_key: String,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 8080,
            api_key: String::new(),
        }
    }
}

impl Config {
    pub fn address(&self) -> Result<SocketAddr> {
        let ip: IpAddr = self
            .host
            .trim()
            .parse()
            .context("the host must be an IP address")?;
        ensure!(
            ip.is_loopback() || !self.api_key.is_empty(),
            "an API key is required when listening outside loopback"
        );
        ensure!(
            self.api_key.chars().all(|c| c.is_ascii_graphic()),
            "the API key must be printable ASCII without spaces"
        );
        Ok(SocketAddr::new(ip, self.port))
    }
}

static NEXT: AtomicU64 = AtomicU64::new(1);

/// Mac `response_format` mapping: text → none, json_object → `{"type":"object"}`, json_schema →
/// its schema (validated up front). The answer is validated after generation, not constrained.
pub fn response_schema(input: &Value) -> Result<Option<String>, String> {
    let format = &input["response_format"];
    let schema = match format["type"].as_str() {
        _ if format.is_null() => return Ok(None),
        Some("text") => return Ok(None),
        Some("json_object") => r#"{"type":"object"}"#.to_owned(),
        Some("json_schema") => {
            let schema = &format["json_schema"]["schema"];
            if schema.is_null() {
                return Err("response_format.json_schema.schema is required".into());
            }
            schema.to_string()
        }
        _ => return Err("Unsupported response_format".into()),
    };
    crate::structured::Schema::parse(&schema).map_err(|e| format!("{e:#}"))?;
    Ok(Some(schema))
}

/// Translate an OpenAI chat request into bridge messages and sampling. Options the engine
/// cannot honour are rejected rather than silently ignored, like the Mac server.
pub fn chat_request(input: &Value, model: &str) -> Result<(Vec<Message>, Sampling), String> {
    let object = input.as_object().ok_or("request must be an object")?;
    for key in object.keys() {
        if !matches!(
            key.as_str(),
            "model"
                | "messages"
                | "stream"
                | "stream_options"
                | "max_completion_tokens"
                | "max_tokens"
                | "n_predict"
                | "temperature"
                | "top_k"
                | "repeat_penalty"
                | "repetition_penalty"
                | "top_p"
                | "min_p"
                | "frequency_penalty"
                | "presence_penalty"
                | "stop"
                | "n"
                | "response_format"
                | "seed"
                | "logit_bias"
                | "tools"
                | "tool_choice"
        ) {
            return Err(format!("{key} is not supported by MLXL3"));
        }
    }
    if input["model"].as_str() != Some(model) {
        return Err("Model does not match this session".into());
    }
    let bad_stream_options = input.get("stream_options").is_some_and(|value| {
        value.as_object().is_none_or(|options| {
            options
                .iter()
                .any(|(key, value)| key != "include_usage" || !value.is_boolean())
        })
    });
    if input.get("stream").is_some_and(|value| !value.is_boolean()) || bad_stream_options {
        return Err("Invalid streaming options".into());
    }
    if input
        .get("tools")
        .is_some_and(|v| !v.is_null() && v.as_array().is_none_or(|t| !t.is_empty()))
        || input
            .get("tool_choice")
            .is_some_and(|c| !c.is_null() && c != "none")
    {
        return Err("OpenAI tool calls are not supported by MLXL3".into());
    }
    if input
        .get("stop")
        .is_some_and(|stop| !stop.is_null() && stop != &json!([]))
    {
        return Err("Stop sequences are not supported by MLXL3".into());
    }
    for key in ["frequency_penalty", "presence_penalty"] {
        if input
            .get(key)
            .is_some_and(|v| !v.is_null() && v.as_f64() != Some(0.0))
        {
            return Err(format!("{key} is not supported by MLXL3"));
        }
    }
    if input
        .get("n")
        .is_some_and(|v| !v.is_null() && v.as_u64() != Some(1))
    {
        return Err("n is not supported by MLXL3".into());
    }
    response_schema(input)?;
    if input.get("logit_bias").is_some_and(|v| !v.is_null()) {
        return Err("Requested output constraint is not supported by MLXL3".into());
    }
    let raw = input["messages"]
        .as_array()
        .ok_or("messages must be an array")?;
    if raw.is_empty() {
        return Err("messages must not be empty".into());
    }
    let mut messages = Vec::with_capacity(raw.len());
    for message in raw {
        let role = message["role"].as_str().ok_or("message role is missing")?;
        if !matches!(role, "system" | "user" | "assistant" | "tool")
            || message.get("tool_calls").is_some()
        {
            return Err("MLXL3 requires text messages without tool calls".into());
        }
        let mut images = vec![];
        let content = match &message["content"] {
            Value::String(text) => text.clone(),
            Value::Array(parts) => {
                let mut text = String::new();
                for part in parts {
                    match part["type"].as_str() {
                        Some("text") => {
                            text.push_str(part["text"].as_str().ok_or("text part without text")?)
                        }
                        Some("image_url") if role == "user" => {
                            let url = part["image_url"]["url"]
                                .as_str()
                                .ok_or("image_url without url")?;
                            images.push(data_url_image(url)?);
                        }
                        _ => return Err("unsupported message content part".into()),
                    }
                }
                text
            }
            Value::Null if role == "assistant" => String::new(),
            _ => return Err("message content must be text".into()),
        };
        messages.push(Message {
            images,
            ..Message::text(role, content)
        });
    }
    if messages.last().is_some_and(|m| m.role == "assistant") {
        return Err("the last message must not be an assistant message".into());
    }
    let raw_max = input
        .get("max_completion_tokens")
        .or_else(|| input.get("max_tokens"))
        .or_else(|| input.get("n_predict"));
    let max_tokens = match raw_max {
        Some(Value::Null) | None => -1,
        Some(value) => value.as_i64().ok_or("max_tokens must be an integer")?,
    };
    if max_tokens != -1 && max_tokens <= 0 {
        return Err("max_tokens must be positive".into());
    }
    let number = |key: &str, default: f64| -> Result<f64, String> {
        match input.get(key) {
            None | Some(Value::Null) => Ok(default),
            Some(value) => value
                .as_f64()
                .ok_or_else(|| "Invalid MLXL3 sampling parameters".to_string()),
        }
    };
    let top_k = match input.get("top_k") {
        None | Some(Value::Null) => 0,
        Some(value) => value
            .as_u64()
            .and_then(|k| u32::try_from(k).ok())
            .ok_or("Invalid MLXL3 sampling parameters")?,
    };
    let penalty = match input.get("repeat_penalty").filter(|v| !v.is_null()) {
        Some(_) => number("repeat_penalty", 1.0)?,
        None => number("repetition_penalty", 1.0)?,
    };
    let seed = match input.get("seed") {
        None | Some(Value::Null) => None,
        Some(value) => Some(
            value
                .as_u64()
                .ok_or("seed must be a non-negative integer")?,
        ),
    };
    let sampling = Sampling {
        max_tokens,
        temperature: number("temperature", 0.0)?,
        top_k,
        top_p: number("top_p", 1.0)?,
        min_p: number("min_p", 0.0)?,
        repetition_penalty: penalty,
        enable_thinking: None,
        seed,
    };
    if !(0.0..=2.0).contains(&sampling.temperature) {
        return Err("Invalid MLXL3 sampling parameters".into());
    }
    sampling.validate().map_err(|e| e.to_string())?;
    Ok((messages, sampling))
}

/// Inline images only (`data:image/png;base64,...`); remote URLs are refused, as on the Mac.
fn data_url_image(url: &str) -> Result<String, String> {
    let rest = url
        .strip_prefix("data:")
        .ok_or("only data: image URLs are supported")?;
    let (meta, data) = rest.split_once(',').ok_or("malformed data URL")?;
    if !matches!(
        meta,
        "image/png;base64" | "image/jpeg;base64" | "image/webp;base64"
    ) {
        return Err("images must be base64 PNG, JPEG or WebP".into());
    }
    Ok(data.to_owned())
}

/// Streams `<think>` content apart from the answer, across arbitrary chunk boundaries.
#[derive(Default)]
pub struct ThinkSplitter {
    inside: bool,
    pending: String,
}

impl ThinkSplitter {
    /// Returns `(thinking, answer)` text that is now certain.
    pub fn push(&mut self, chunk: &str) -> (String, String) {
        self.pending.push_str(chunk);
        let (mut thinking, mut answer) = (String::new(), String::new());
        loop {
            let tag = if self.inside { "</think>" } else { "<think>" };
            if let Some(at) = self.pending.find(tag) {
                let before: String = self.pending.drain(..at + tag.len()).collect();
                let text = &before[..at];
                if self.inside {
                    thinking.push_str(text)
                } else {
                    answer.push_str(text)
                }
                self.inside = !self.inside;
                continue;
            }
            // Keep a suffix that could still become the tag.
            let keep = (1..tag.len())
                .rev()
                .find(|n| self.pending.ends_with(&tag[..*n]))
                .unwrap_or(0);
            let cut = self.pending.len() - keep;
            let text: String = self.pending.drain(..cut).collect();
            if self.inside {
                thinking.push_str(&text)
            } else {
                answer.push_str(&text)
            }
            return (thinking, answer);
        }
    }

    pub fn finish(&mut self) -> (String, String) {
        let rest = std::mem::take(&mut self.pending);
        if self.inside {
            (rest, String::new())
        } else {
            (String::new(), rest)
        }
    }
}

pub struct Request {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Reads one HTTP/1.1 request. Errors carry the status to answer with.
pub fn read_request(stream: &mut impl Read) -> Result<Request, (u16, String)> {
    let mut reader = BufReader::new(stream.take((MAX_HEAD + MAX_BODY + 1) as u64));
    let mut head = Vec::new();
    loop {
        let mut line = Vec::new();
        let n = reader
            .read_until(b'\n', &mut line)
            .map_err(|e| (400, e.to_string()))?;
        if n == 0 {
            return Err((400, "incomplete request".into()));
        }
        head.extend_from_slice(&line);
        if head.len() > MAX_HEAD {
            return Err((431, "request headers are too large".into()));
        }
        if line == b"\r\n" || line == b"\n" {
            break;
        }
    }
    let head = String::from_utf8(head).map_err(|_| (400, "headers are not UTF-8".to_string()))?;
    let mut lines = head.lines();
    let mut start = lines.next().unwrap_or_default().split(' ');
    let (method, target) = (
        start.next().unwrap_or_default().to_owned(),
        start.next().unwrap_or_default(),
    );
    let path = target.split('?').next().unwrap_or_default().to_owned();
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_owned(), v.trim().to_owned()))
        .collect();
    let mut request = Request {
        method,
        path,
        headers,
        body: vec![],
    };
    if request.header("transfer-encoding").is_some() {
        return Err((
            411,
            "chunked request bodies are not supported; send Content-Length".into(),
        ));
    }
    let length = match request.header("content-length") {
        Some(value) => value
            .parse::<usize>()
            .map_err(|_| (400, "invalid Content-Length".to_string()))?,
        None => 0,
    };
    if length > MAX_BODY {
        return Err((413, "Request body exceeds 16 MiB".into()));
    }
    request.body = vec![0; length];
    reader
        .read_exact(&mut request.body)
        .map_err(|_| (400, "incomplete request body".to_string()))?;
    Ok(request)
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        411 => "Length Required",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        _ => "Error",
    }
}

fn write_json(out: &mut impl Write, status: u16, value: &Value) -> std::io::Result<()> {
    let body = value.to_string();
    write!(
        out,
        "HTTP/1.1 {status} {}\r\ncontent-type: application/json\r\ncontent-length: {}\r\naccess-control-allow-origin: *\r\nconnection: close\r\n\r\n{body}",
        reason(status),
        body.len()
    )?;
    out.flush()
}

fn error_body(message: &str) -> Value {
    json!({"error": {"message": message, "type": "invalid_request_error"}})
}

fn same_key(given: &[u8], expected: &[u8]) -> bool {
    // Length-independent comparison time for keys of equal length.
    given.len() == expected.len()
        && given
            .iter()
            .zip(expected)
            .fold(0u8, |acc, (a, b)| acc | (a ^ b))
            == 0
}

fn usage(stats: &Value) -> Value {
    let prompt = stats["prompt_tokens"].as_u64().unwrap_or(0);
    let completion = stats["generated_tokens"].as_u64().unwrap_or(0);
    json!({"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt.saturating_add(completion)})
}

fn chunk(id: &str, created: u64, model: &str, delta: Value, finish: Value) -> Value {
    json!({"id": id, "object": "chat.completion.chunk", "created": created, "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]})
}

/// Handles one connection: returns `(status, path)` for the log.
pub fn serve_connection(
    stream: &mut (impl Read + Write),
    engine: &dyn Engine,
    api_key: &str,
) -> (u16, String) {
    let request = match read_request(stream) {
        Ok(request) => request,
        Err((status, message)) => {
            let _ = write_json(stream, status, &error_body(&message));
            return (status, String::new());
        }
    };
    let path = request.path.clone();
    let status = respond(stream, &request, engine, api_key);
    (status, path)
}

fn send(out: &mut dyn Write, status: u16, value: Value) -> u16 {
    let mut buffer = Vec::new();
    let _ = write_json(&mut buffer, status, &value);
    let _ = out.write_all(&buffer).and_then(|()| out.flush());
    status
}

fn respond(out: &mut impl Write, request: &Request, engine: &dyn Engine, api_key: &str) -> u16 {
    if request.method == "OPTIONS" {
        let _ = out.write_all(b"HTTP/1.1 204 No Content\r\naccess-control-allow-origin: *\r\naccess-control-allow-methods: GET, POST, OPTIONS\r\naccess-control-allow-headers: authorization, content-type\r\ncontent-length: 0\r\nconnection: close\r\n\r\n");
        return 204;
    }
    if request.method == "GET" && request.path == "/health" {
        return send(out, 200, json!({"status": "ok"}));
    }
    if !api_key.is_empty() {
        let given = request
            .header("authorization")
            .and_then(|v| v.strip_prefix("Bearer "))
            .unwrap_or("");
        if !same_key(given.as_bytes(), api_key.as_bytes()) {
            return send(out, 401, error_body("Invalid API key"));
        }
    }
    let Some(model) = engine.model() else {
        return send(out, 503, error_body("No model is loaded in MLXL3"));
    };
    if request.method == "GET" && request.path == "/v1/models" {
        return send(
            out,
            200,
            json!({"object": "list", "data": [{"id": model, "object": "model", "owned_by": "mlxl3"}]}),
        );
    }
    if request.method != "POST" || request.path != "/v1/chat/completions" {
        return send(out, 404, error_body("Unsupported endpoint"));
    }
    let input: Value = match serde_json::from_slice(&request.body) {
        Ok(input) => input,
        Err(error) => return send(out, 400, error_body(&error.to_string())),
    };
    let (messages, sampling) = match chat_request(&input, &model) {
        Ok(parsed) => parsed,
        Err(error) => return send(out, 400, error_body(&error)),
    };
    let request_id = format!("srv-{}", NEXT.fetch_add(1, Ordering::Relaxed));
    let request = turn::TurnRequest {
        messages,
        sampling,
        schema: response_schema(&input).ok().flatten(),
        tools: vec![],
        allowed: None,
        context_limit: 0,
    };
    let schema = match turn::admit(&request) {
        Ok(schema) => schema,
        Err(error) => return send(out, 400, error_body(&format!("{error:#}"))),
    };
    let cancel = AtomicBool::new(false);
    let (sender, events) = mpsc::channel();
    std::thread::scope(|scope| {
        scope.spawn(|| {
            turn::run(engine, None, &request_id, request, schema, &cancel, &sender);
            drop(sender);
        });
        let code = stream_reply(out, &input, &model, &request_id, &events, &cancel);
        cancel.store(true, Ordering::Relaxed);
        code
    })
}

fn stream_reply(
    out: &mut impl Write,
    input: &Value,
    model: &str,
    request_id: &str,
    events: &Receiver<Value>,
    cancel: &AtomicBool,
) -> u16 {
    let id = format!("chatcmpl-{request_id}");
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let stream = input["stream"].as_bool().unwrap_or(false);
    let include_usage = input["stream_options"]["include_usage"]
        .as_bool()
        .unwrap_or(false);
    let mut splitter = ThinkSplitter::default();
    let (mut answer, mut thinking) = (String::new(), String::new());
    if stream {
        let head = "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncache-control: no-cache\r\naccess-control-allow-origin: *\r\nconnection: close\r\n\r\n";
        if out.write_all(head.as_bytes()).is_err() {
            cancel.store(true, Ordering::Relaxed);
            return 200;
        }
        let _ = sse(
            out,
            &chunk(
                &id,
                created,
                model,
                json!({"role": "assistant"}),
                Value::Null,
            ),
        );
    }
    let outcome = loop {
        let event = match events.recv_timeout(EVENT_TIMEOUT) {
            Ok(event) => event,
            Err(RecvTimeoutError::Timeout) => {
                cancel.store(true, Ordering::Relaxed);
                break Err("the engine stopped responding".to_owned());
            }
            Err(RecvTimeoutError::Disconnected) => break Err("the engine exited".to_owned()),
        };
        match event["type"].as_str() {
            Some("delta") => {
                let (think, text) = splitter.push(event["text"].as_str().unwrap_or(""));
                thinking.push_str(&think);
                answer.push_str(&text);
                if stream {
                    let mut sent = Ok(());
                    if !think.is_empty() {
                        sent = sse(
                            out,
                            &chunk(
                                &id,
                                created,
                                model,
                                json!({"reasoning_content": think}),
                                Value::Null,
                            ),
                        );
                    }
                    if sent.is_ok() && !text.is_empty() {
                        sent = sse(
                            out,
                            &chunk(&id, created, model, json!({"content": text}), Value::Null),
                        );
                    }
                    if sent.is_err() {
                        // Client went away: free the engine for the next request.
                        cancel.store(true, Ordering::Relaxed);
                        return 200;
                    }
                }
            }
            Some("done") => break Ok(event),
            Some("cancelled") => break Err("generation cancelled".to_owned()),
            Some("error") => {
                break Err(event["message"]
                    .as_str()
                    .unwrap_or("generation failed")
                    .to_owned());
            }
            _ => {}
        }
    };
    let (think, text) = splitter.finish();
    thinking.push_str(&think);
    answer.push_str(&text);
    let finish = |stats: &Value| {
        if stats["stop_reason"] == "max_new_tokens" {
            "length"
        } else {
            "stop"
        }
    };
    if stream {
        match outcome {
            Ok(stats) => {
                if !think.is_empty() {
                    let _ = sse(
                        out,
                        &chunk(
                            &id,
                            created,
                            model,
                            json!({"reasoning_content": think}),
                            Value::Null,
                        ),
                    );
                }
                if !text.is_empty() {
                    let _ = sse(
                        out,
                        &chunk(&id, created, model, json!({"content": text}), Value::Null),
                    );
                }
                let _ = sse(
                    out,
                    &chunk(&id, created, model, json!({}), json!(finish(&stats))),
                );
                if include_usage {
                    let _ = sse(
                        out,
                        &json!({"id": id, "object": "chat.completion.chunk", "created": created,
                        "model": model, "choices": [], "usage": usage(&stats)}),
                    );
                }
                let _ = out
                    .write_all(b"data: [DONE]\n\n")
                    .and_then(|()| out.flush());
            }
            Err(message) => {
                let _ = sse(
                    out,
                    &json!({"error": {"message": message, "type": "server_error"}}),
                );
            }
        }
        return 200;
    }
    match outcome {
        Ok(stats) => send(
            out,
            200,
            json!({
                "id": id, "object": "chat.completion", "created": created, "model": model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer.trim_start(),
                    "reasoning_content": thinking}, "finish_reason": finish(&stats)}],
                "usage": usage(&stats),
            }),
        ),
        Err(message) => send(
            out,
            502,
            json!({"error": {"message": message, "type": "server_error"}}),
        ),
    }
}

fn sse(out: &mut impl Write, value: &Value) -> std::io::Result<()> {
    out.write_all(format!("data: {value}\n\n").as_bytes())?;
    out.flush()
}

/// A running server; dropping it stops accepting connections.
pub struct Handle {
    pub address: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl Handle {
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Handle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Bind and serve until stopped. `log` receives one line per request (method-free, no body).
pub fn start(
    config: &Config,
    engine: Arc<dyn Engine>,
    log: Arc<dyn Fn(String) + Send + Sync>,
) -> Result<Handle> {
    let address = config.address()?;
    let listener =
        TcpListener::bind(address).with_context(|| format!("cannot listen on {address}"))?;
    listener.set_nonblocking(true)?;
    let address = listener.local_addr()?;
    let stop = Arc::new(AtomicBool::new(false));
    let active = Arc::new(AtomicUsize::new(0));
    let key = Arc::new(Mutex::new(config.api_key.clone()));
    let flag = stop.clone();
    let thread = std::thread::Builder::new()
        .name("mlxl3-server".into())
        .spawn(move || {
            while !flag.load(Ordering::SeqCst) {
                let (mut stream, peer) = match listener.accept() {
                    Ok(accepted) => accepted,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(Duration::from_millis(50));
                        continue;
                    }
                    Err(_) => continue,
                };
                let _ = stream.set_nonblocking(false);
                let _ = stream.set_read_timeout(Some(Duration::from_secs(30)));
                if active.fetch_add(1, Ordering::SeqCst) >= MAX_CONNECTIONS {
                    active.fetch_sub(1, Ordering::SeqCst);
                    let _ = write_json(&mut stream, 503, &error_body("too many connections"));
                    continue;
                }
                let (engine, log, active, key) =
                    (engine.clone(), log.clone(), active.clone(), key.clone());
                std::thread::spawn(move || {
                    let api_key = key.lock().unwrap().clone();
                    let (status, path) = serve_connection(&mut stream, engine.as_ref(), &api_key);
                    log(format!("{} {status} {path}", peer.ip()));
                    let _ = stream.shutdown(std::net::Shutdown::Both);
                    active.fetch_sub(1, Ordering::SeqCst);
                });
            }
        })?;
    Ok(Handle {
        address,
        stop,
        thread: Some(thread),
    })
}

/// Sends a request to a local server and returns the raw response (tests and the UI self-check).
pub fn local_request(address: SocketAddr, raw: &[u8]) -> Result<String> {
    let mut stream = TcpStream::connect_timeout(&address, Duration::from_secs(5))?;
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.write_all(raw)?;
    let mut response = String::new();
    stream.read_to_string(&mut response)?;
    Ok(response)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    /// Echoes the last user message word by word, like the worker's fake backend.
    struct Fake {
        model: Option<String>,
        cancelled: Mutex<Vec<String>>,
        script: Option<Vec<Value>>,
    }

    impl Fake {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                model: Some("m".into()),
                cancelled: Mutex::new(vec![]),
                script: None,
            })
        }
    }

    impl Engine for Fake {
        fn model(&self) -> Option<String> {
            self.model.clone()
        }
        fn submit(
            &self,
            request_id: &str,
            messages: &[Message],
            sampling: &Sampling,
            _tools: &[Value],
        ) -> Result<Receiver<Value>> {
            let (tx, rx) = mpsc::channel();
            let events = self.script.clone().unwrap_or_else(|| {
                let last = &messages.last().unwrap().content;
                let mut events: Vec<Value> = last.split(' ').enumerate()
                    .map(|(i, w)| json!({"type": "delta", "request_id": request_id, "text": if i == 0 { w.to_owned() } else { format!(" {w}") }}))
                    .collect();
                let reason = if sampling.max_tokens == 1 { "max_new_tokens" } else { "eos" };
                events.push(json!({"type": "done", "prompt_tokens": 3, "generated_tokens": 2, "stop_reason": reason}));
                events
            });
            for event in events {
                tx.send(event).unwrap();
            }
            Ok(rx)
        }
        fn cancel(&self, request_id: &str) {
            self.cancelled.lock().unwrap().push(request_id.to_owned());
        }
    }

    fn body(model: &str, extra: Value) -> Value {
        let mut value =
            json!({"model": model, "messages": [{"role": "user", "content": "hello world"}]});
        value
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        value
    }

    #[test]
    fn requests_map_to_bridge_and_unsupported_options_fail() {
        let (messages, sampling) = chat_request(
            &body(
                "m",
                json!({"max_tokens": 5, "temperature": 0.7, "top_p": 0.9, "seed": 3, "stop": []}),
            ),
            "m",
        )
        .unwrap();
        assert_eq!(messages, vec![Message::text("user", "hello world")]);
        assert_eq!(
            (
                sampling.max_tokens,
                sampling.temperature,
                sampling.top_p,
                sampling.seed
            ),
            (5, 0.7, 0.9, Some(3))
        );
        let parts = json!({"model": "m", "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is "}, {"type": "text", "text": "this"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]}]});
        let (messages, _) = chat_request(&parts, "m").unwrap();
        assert_eq!(
            (messages[0].content.as_str(), messages[0].images.clone()),
            ("what is this", vec!["QUJD".to_owned()])
        );
        for (bad, why) in [
            (body("other", json!({})), "Model does not match"),
            (body("m", json!({"stream": "yes"})), "streaming"),
            (
                body("m", json!({"stream_options": {"x": true}})),
                "streaming",
            ),
            (
                body("m", json!({"tools": [{"type": "function"}]})),
                "tool calls",
            ),
            (body("m", json!({"tool_choice": "auto"})), "tool calls"),
            (body("m", json!({"stop": ["x"]})), "Stop sequences"),
            (body("m", json!({"n": 2})), "n is not"),
            (
                body("m", json!({"presence_penalty": 0.5})),
                "presence_penalty",
            ),
            (
                body("m", json!({"response_format": {"type": "regex"}})),
                "Unsupported response_format",
            ),
            (
                body(
                    "m",
                    json!({"response_format": {"type": "json_schema", "json_schema": {}}}),
                ),
                "schema is required",
            ),
            (
                body(
                    "m",
                    json!({"response_format": {"type": "json_schema", "json_schema": {"schema": {"pattern": "x"}}}}),
                ),
                "unsupported JSON schema keyword",
            ),
            (body("m", json!({"logit_bias": {"1": 2}})), "constraint"),
            (body("m", json!({"max_tokens": 0})), "positive"),
            (body("m", json!({"max_tokens": 1.5})), "integer"),
            (body("m", json!({"temperature": 2.5})), "sampling"),
            (body("m", json!({"temperature": "hot"})), "sampling"),
            (body("m", json!({"top_p": 0})), "top-p"),
            (body("m", json!({"seed": -1})), "seed"),
            (body("m", json!({"unknown": 1})), "unknown is not supported"),
            (json!({"model": "m", "messages": []}), "empty"),
            (
                json!({"model": "m", "messages": [{"role": "assistant", "content": "x"}]}),
                "last message",
            ),
            (
                json!({"model": "m", "messages": [{"role": "robot", "content": "x"}]}),
                "text messages",
            ),
            (
                json!({"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "https://e.com/a.png"}}]}]}),
                "data:",
            ),
            (
                json!({"model": "m", "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/gif;base64,AA"}}]}]}),
                "PNG",
            ),
            (json!([1]), "object"),
        ] {
            let error = chat_request(&bad, "m").unwrap_err();
            assert!(error.contains(why), "{bad} -> {error}");
        }
    }

    #[test]
    fn think_splitter_handles_split_tags() {
        let mut splitter = ThinkSplitter::default();
        let mut thinking = String::new();
        let mut answer = String::new();
        for piece in ["<thi", "nk>pl", "an</th", "ink>\nAns", "wer <", "b>"] {
            let (t, a) = splitter.push(piece);
            thinking += &t;
            answer += &a;
        }
        let (t, a) = splitter.finish();
        thinking += &t;
        answer += &a;
        assert_eq!(
            (thinking.as_str(), answer.as_str()),
            ("plan", "\nAnswer <b>")
        );
        let mut open = ThinkSplitter::default();
        assert_eq!(
            open.push("<think>never closed <"),
            ("never closed ".into(), String::new())
        );
        assert_eq!(open.finish(), ("<".into(), String::new()));
    }

    fn call(engine: &dyn Engine, key: &str, raw: &str) -> (u16, String) {
        let mut io = Duplex {
            input: std::io::Cursor::new(raw.as_bytes().to_vec()),
            output: vec![],
        };
        let (status, _) = serve_connection(&mut io, engine, key);
        (status, String::from_utf8(io.output).unwrap())
    }

    struct Duplex {
        input: std::io::Cursor<Vec<u8>>,
        output: Vec<u8>,
    }
    impl Read for Duplex {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.input.read(buf)
        }
    }
    impl Write for Duplex {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.output.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    fn post(body: &Value, key: Option<&str>) -> String {
        let body = body.to_string();
        let auth = key
            .map(|k| format!("Authorization: Bearer {k}\r\n"))
            .unwrap_or_default();
        format!(
            "POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n{auth}Content-Length: {}\r\n\r\n{body}",
            body.len()
        )
    }

    #[test]
    fn routes_enforce_the_key_and_answer_like_openai() {
        let engine = Fake::new();
        assert_eq!(
            call(engine.as_ref(), "k", "GET /health HTTP/1.1\r\n\r\n").0,
            200
        );
        assert_eq!(
            call(engine.as_ref(), "k", "GET /v1/models HTTP/1.1\r\n\r\n").0,
            401
        );
        assert_eq!(
            call(
                engine.as_ref(),
                "k",
                "GET /v1/models HTTP/1.1\r\nAuthorization: Bearer x\r\n\r\n"
            )
            .0,
            401
        );
        let (status, text) = call(
            engine.as_ref(),
            "k",
            "GET /v1/models HTTP/1.1\r\nauthorization: Bearer k\r\n\r\n",
        );
        assert_eq!(status, 200);
        assert!(text.contains(r#""id":"m""#), "{text}");
        assert_eq!(
            call(engine.as_ref(), "", "GET /nope HTTP/1.1\r\n\r\n").0,
            404
        );
        assert_eq!(
            call(
                engine.as_ref(),
                "",
                "OPTIONS /v1/chat/completions HTTP/1.1\r\n\r\n"
            )
            .0,
            204
        );

        let (status, text) = call(
            engine.as_ref(),
            "k",
            &post(&body("m", json!({})), Some("k")),
        );
        assert_eq!(status, 200);
        let json: Value = serde_json::from_str(text.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(json["choices"][0]["message"]["content"], "hello world");
        assert_eq!(json["choices"][0]["finish_reason"], "stop");
        assert_eq!(json["usage"]["total_tokens"], 5);
        let (_, text) = call(
            engine.as_ref(),
            "",
            &post(&body("m", json!({"max_tokens": 1})), None),
        );
        assert!(text.contains(r#""finish_reason":"length""#));

        let (status, text) = call(
            engine.as_ref(),
            "",
            &post(
                &body(
                    "m",
                    json!({"stream": true, "stream_options": {"include_usage": true}}),
                ),
                None,
            ),
        );
        assert_eq!(status, 200);
        assert!(text.contains("content-type: text/event-stream"));
        let events: Vec<&str> = text
            .split("\r\n\r\n")
            .nth(1)
            .unwrap()
            .split("\n\n")
            .filter(|e| !e.is_empty())
            .collect();
        assert_eq!(events.last(), Some(&"data: [DONE]"));
        assert!(events[0].contains(r#""role":"assistant""#));
        assert!(events.iter().any(|e| e.contains(r#""content":" world""#)));
        assert!(events.iter().any(|e| e.contains(r#""usage""#)));

        let (status, text) = call(
            engine.as_ref(),
            "",
            &post(&body("m", json!({"stop": ["x"]})), None),
        );
        assert_eq!(status, 400);
        assert!(text.contains("Stop sequences"));
        assert_eq!(
            call(
                engine.as_ref(),
                "",
                "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 3\r\n\r\n{x}"
            )
            .0,
            400
        );
        assert_eq!(
            call(
                engine.as_ref(),
                "",
                &format!(
                    "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
                    MAX_BODY + 1
                )
            )
            .0,
            413
        );
        assert_eq!(
            call(
                engine.as_ref(),
                "",
                "POST /v1/chat/completions HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n"
            )
            .0,
            411
        );
        assert_eq!(
            call(
                engine.as_ref(),
                "",
                "POST /v1/chat/completions HTTP/1.1\r\nContent-Length: 10\r\n\r\n{}"
            )
            .0,
            400
        );
        let huge = format!(
            "GET /health HTTP/1.1\r\nX: {}\r\n\r\n",
            "a".repeat(MAX_HEAD)
        );
        assert_eq!(call(engine.as_ref(), "", &huge).0, 431);

        let unloaded = Arc::new(Fake {
            model: None,
            cancelled: Mutex::new(vec![]),
            script: None,
        });
        assert_eq!(
            call(unloaded.as_ref(), "", "GET /v1/models HTTP/1.1\r\n\r\n").0,
            503
        );
    }

    #[test]
    fn engine_errors_and_thinking_reach_the_client() {
        let failing = Arc::new(Fake {
            model: Some("m".into()),
            cancelled: Mutex::new(vec![]),
            script: Some(vec![
                json!({"type": "delta", "text": "par"}),
                json!({"type": "error", "message": "boom"}),
            ]),
        });
        let (status, text) = call(failing.as_ref(), "", &post(&body("m", json!({})), None));
        assert_eq!(status, 502);
        assert!(text.contains("boom"));
        let (_, text) = call(
            failing.as_ref(),
            "",
            &post(&body("m", json!({"stream": true})), None),
        );
        assert!(text.contains(r#""type":"server_error""#) && !text.contains("[DONE]"));
        let thinking = Arc::new(Fake {
            model: Some("m".into()),
            cancelled: Mutex::new(vec![]),
            script: Some(vec![
                json!({"type": "delta", "text": "<think>hmm</think>\n\nYes"}),
                json!({"type": "done"}),
            ]),
        });
        let (_, text) = call(thinking.as_ref(), "", &post(&body("m", json!({})), None));
        let json: Value = serde_json::from_str(text.split("\r\n\r\n").nth(1).unwrap()).unwrap();
        assert_eq!(
            (
                json["choices"][0]["message"]["content"].as_str(),
                json["choices"][0]["message"]["reasoning_content"].as_str()
            ),
            (Some("Yes"), Some("hmm"))
        );
        let closed = Arc::new(Fake {
            model: Some("m".into()),
            cancelled: Mutex::new(vec![]),
            script: Some(vec![]),
        });
        let (status, text) = call(closed.as_ref(), "", &post(&body("m", json!({})), None));
        assert_eq!(status, 502);
        assert!(text.contains("exited"));
    }

    #[test]
    fn config_requires_a_key_off_loopback() {
        assert!(Config::default().address().is_ok());
        let open = Config {
            host: "0.0.0.0".into(),
            ..Config::default()
        };
        assert!(open.address().unwrap_err().to_string().contains("API key"));
        assert!(
            Config {
                api_key: "secret".into(),
                ..open.clone()
            }
            .address()
            .is_ok()
        );
        assert!(
            Config {
                api_key: "a b".into(),
                ..Config::default()
            }
            .address()
            .is_err()
        );
        assert!(
            Config {
                host: "localhost".into(),
                ..Config::default()
            }
            .address()
            .is_err()
        );
        assert!(
            Config {
                host: "::1".into(),
                ..Config::default()
            }
            .address()
            .is_ok()
        );
    }

    #[test]
    fn a_real_socket_serves_and_stops() {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let sink = logs.clone();
        let handle = start(
            &Config {
                port: 0,
                ..Config::default()
            },
            Fake::new(),
            Arc::new(move |line| sink.lock().unwrap().push(line)),
        )
        .unwrap();
        let response =
            local_request(handle.address, post(&body("m", json!({})), None).as_bytes()).unwrap();
        assert!(response.starts_with("HTTP/1.1 200 OK"), "{response}");
        assert!(response.contains("hello world"));
        let address = handle.address;
        handle.stop();
        std::thread::sleep(Duration::from_millis(100));
        assert!(TcpStream::connect_timeout(&address, Duration::from_millis(300)).is_err());
        let lines = logs.lock().unwrap();
        assert_eq!(lines.len(), 1);
        assert!(lines[0].ends_with("200 /v1/chat/completions"), "{lines:?}");
    }
}
