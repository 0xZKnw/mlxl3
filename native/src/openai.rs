//! Loopback OpenAI Chat Completions adapter for the resident MLXL3 bridge.

use anyhow::{Context, Result, bail};
use hyper::{
    Body, Request, Response, Server, StatusCode,
    body::{Bytes, HttpBody},
    service::{make_service_fn, service_fn},
};
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    net::{IpAddr, SocketAddr, TcpListener},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader},
    process::{ChildStdin, ChildStdout, Command},
    sync::Mutex,
};

static NEXT_REQUEST: AtomicU64 = AtomicU64::new(1);

struct Bridge {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

struct State {
    model: String,
    api_key: String,
    bridge_pid: u32,
    bridge: Mutex<Bridge>,
}

pub async fn serve(
    registry: Option<&Path>,
    model: &str,
    host: &str,
    port: u16,
    api_key: &str,
    context_length: i32,
) -> Result<()> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .context("invalid listen address")?;
    if !matches!(addr.ip(), IpAddr::V4(ip) if ip.is_loopback())
        && !matches!(addr.ip(), IpAddr::V6(ip) if ip.is_loopback())
        && api_key.is_empty()
    {
        bail!("an API key is required when listening outside loopback");
    }
    let listener = TcpListener::bind(addr).context("cannot bind OpenAI API")?;
    let actual_port = listener.local_addr()?.port();

    let executable = std::env::current_exe()?;
    let mut command = Command::new(executable);
    if let Some(registry) = registry {
        command.arg("--registry").arg(registry);
    }
    let mut child = command
        .arg("bridge")
        .arg(model)
        .arg("--context-length")
        .arg(context_length.to_string())
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .context("cannot start resident bridge")?;
    let bridge_pid = child.id().context("bridge has no PID")?;
    let stdin = child.stdin.take().context("bridge stdin unavailable")?;
    let mut stdout = BufReader::new(child.stdout.take().context("bridge stdout unavailable")?);
    if let Some(stderr) = child.stderr.take() {
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                eprintln!("mlxl3 bridge: {line}");
            }
        });
    }
    let ready = tokio::time::timeout(std::time::Duration::from_secs(180), async {
        let mut line = String::new();
        loop {
            line.clear();
            if stdout.read_line(&mut line).await? == 0 {
                bail!("bridge exited before readiness");
            }
            let event: Value = serde_json::from_str(&line)?;
            match event["type"].as_str() {
                Some("ready") => return Ok::<(), anyhow::Error>(()),
                Some("error") => bail!(
                    "{}",
                    event["message"].as_str().unwrap_or("model load failed")
                ),
                _ => {}
            }
        }
    })
    .await
    .context("model load timed out")?;
    ready?;

    let state = Arc::new(State {
        model: model.to_owned(),
        api_key: api_key.to_owned(),
        bridge_pid,
        bridge: Mutex::new(Bridge { stdin, stdout }),
    });
    let service = make_service_fn(move |_| {
        let state = state.clone();
        async move { Ok::<_, Infallible>(service_fn(move |request| handle(request, state.clone()))) }
    });
    let server = Server::from_tcp(listener)?.serve(service);
    println!(
        "{}",
        json!({"type":"ready","model":model,"port":actual_port})
    );
    use std::io::Write;
    std::io::stdout().flush()?;

    #[cfg(unix)]
    {
        let mut term = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            result = server => result?,
            _ = tokio::signal::ctrl_c() => {},
            _ = term.recv() => {},
        }
    }
    #[cfg(not(unix))]
    tokio::select! {
        result = server => result?,
        _ = tokio::signal::ctrl_c() => {},
    }
    cancel(bridge_pid);
    let _ = child.kill().await;
    let _ = child.wait().await;
    Ok(())
}

fn cancel(pid: u32) {
    #[cfg(unix)]
    unsafe {
        libc::kill(pid as i32, libc::SIGUSR1);
    }
    #[cfg(not(unix))]
    let _ = pid;
}

fn json_response(status: StatusCode, value: Value) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("access-control-allow-origin", "*")
        .body(Body::from(value.to_string()))
        .expect("valid response")
}

fn error(status: StatusCode, message: impl Into<String>) -> Response<Body> {
    json_response(
        status,
        json!({"error":{"message":message.into(),"type":"invalid_request_error"}}),
    )
}

async fn handle(request: Request<Body>, state: Arc<State>) -> Result<Response<Body>, Infallible> {
    if request.method() == hyper::Method::OPTIONS {
        return Ok(Response::builder()
            .status(StatusCode::NO_CONTENT)
            .header("access-control-allow-origin", "*")
            .header("access-control-allow-methods", "GET, POST, OPTIONS")
            .header(
                "access-control-allow-headers",
                "authorization, content-type",
            )
            .body(Body::empty())
            .expect("valid preflight response"));
    }
    if request.method() == hyper::Method::GET && request.uri().path() == "/health" {
        return Ok(json_response(StatusCode::OK, json!({"status":"ok"})));
    }
    if !state.api_key.is_empty()
        && request
            .headers()
            .get("authorization")
            .and_then(|value| value.to_str().ok())
            != Some(format!("Bearer {}", state.api_key).as_str())
    {
        return Ok(error(StatusCode::UNAUTHORIZED, "Invalid API key"));
    }
    if request.method() == hyper::Method::GET && request.uri().path() == "/v1/models" {
        return Ok(json_response(
            StatusCode::OK,
            json!({"object":"list","data":[
                {"id":state.model,"object":"model","owned_by":"mlxl3"}
            ]}),
        ));
    }
    if request.method() != hyper::Method::POST || request.uri().path() != "/v1/chat/completions" {
        return Ok(error(StatusCode::NOT_FOUND, "Unsupported endpoint"));
    }
    let mut incoming = request.into_body();
    let mut body = Vec::new();
    while let Some(chunk) = incoming.data().await {
        match chunk {
            Ok(chunk) if chunk.len() <= (16 * 1024 * 1024usize).saturating_sub(body.len()) => {
                body.extend_from_slice(&chunk)
            }
            Ok(_) => {
                return Ok(error(
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "Request body exceeds 16 MiB",
                ));
            }
            Err(err) => return Ok(error(StatusCode::BAD_REQUEST, err.to_string())),
        }
    }
    let input: Value = match serde_json::from_slice(&body) {
        Ok(input) => input,
        Err(err) => return Ok(error(StatusCode::BAD_REQUEST, err.to_string())),
    };
    let bridge_request = match bridge_request(&input, &state.model) {
        Ok(request) => request,
        Err(err) => return Ok(error(StatusCode::BAD_REQUEST, err)),
    };
    let id = format!(
        "chatcmpl-{}-{}",
        std::process::id(),
        NEXT_REQUEST.fetch_add(1, Ordering::Relaxed)
    );
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let stream = input["stream"].as_bool().unwrap_or(false);
    let include_usage = input["stream_options"]["include_usage"]
        .as_bool()
        .unwrap_or(false);
    if stream {
        let (sender, body) = Body::channel();
        tokio::spawn(async move {
            let mut sender = Some(sender);
            let result = generate(&state, bridge_request, &id, created, &mut sender).await;
            if let Some(mut sender) = sender {
                match result {
                    Ok(output) => {
                        let finish = if output.context_full {
                            "length"
                        } else {
                            "stop"
                        };
                        let _ = send_sse(
                            &mut sender,
                            json!({"id":id,"object":"chat.completion.chunk",
                            "created":created,"model":state.model,
                            "choices":[{"index":0,"delta":{},"finish_reason":finish}]}),
                        )
                        .await;
                        if include_usage {
                            let _ = send_sse(&mut sender, json!({"id":id,"object":"chat.completion.chunk",
                                "created":created,"model":state.model,"choices":[],"usage":usage(&output.stats)})).await;
                        }
                        let _ = sender.send_data(Bytes::from("data: [DONE]\n\n")).await;
                    }
                    Err(err) => {
                        let _ = send_sse(
                            &mut sender,
                            json!({"error":{"message":err.to_string(),"type":"server_error"}}),
                        )
                        .await;
                    }
                }
            }
        });
        return Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .header("access-control-allow-origin", "*")
            .body(body)
            .expect("valid stream response"));
    }
    let result = generate(&state, bridge_request, &id, created, &mut None).await;
    Ok(match result {
        Ok(output) => json_response(
            StatusCode::OK,
            json!({
                "id":id,"object":"chat.completion","created":created,"model":state.model,
                "choices":[{"index":0,"message":{"role":"assistant","content":output.answer,
                    "reasoning_content":output.thinking},
                    "finish_reason":if output.context_full {"length"} else {"stop"}}],
                "usage":usage(&output.stats)
            }),
        ),
        Err(err) => error(StatusCode::BAD_GATEWAY, err.to_string()),
    })
}

fn bridge_request(input: &Value, model: &str) -> Result<Value, String> {
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
    if input.get("stream").is_some_and(|value| !value.is_boolean())
        || input
            .get("stream_options")
            .is_some_and(|value| !value.is_object())
        || input
            .get("stream_options")
            .and_then(Value::as_object)
            .is_some_and(|options| {
                options
                    .iter()
                    .any(|(key, value)| key != "include_usage" || !value.is_boolean())
            })
    {
        return Err("Invalid streaming options".into());
    }
    if input.get("tools").is_some_and(|value| !value.is_array())
        || input["tools"]
            .as_array()
            .is_some_and(|tools| !tools.is_empty())
        || input
            .get("tool_choice")
            .is_some_and(|choice| !choice.is_null() && choice != "none")
    {
        return Err("OpenAI tool calls are not supported by MLXL3".into());
    }
    if input
        .get("stop")
        .is_some_and(|stop| !stop.is_null() && stop != &json!([]))
    {
        return Err("Stop sequences are not supported by MLXL3".into());
    }
    for (key, default) in [
        ("top_p", 1.0),
        ("frequency_penalty", 0.0),
        ("presence_penalty", 0.0),
    ] {
        if input
            .get(key)
            .is_some_and(|value| !value.is_null() && value.as_f64() != Some(default))
        {
            return Err(format!("{key} is not supported by MLXL3"));
        }
    }
    if input
        .get("n")
        .is_some_and(|value| !value.is_null() && value.as_u64() != Some(1))
    {
        return Err("n is not supported by MLXL3".into());
    }
    if input
        .get("response_format")
        .is_some_and(|value| !value.is_null() && value["type"] != "text")
        || input.get("seed").is_some_and(|value| !value.is_null())
        || input
            .get("logit_bias")
            .is_some_and(|value| !value.is_null())
    {
        return Err("Requested output constraint is not supported by MLXL3".into());
    }
    let raw_messages = input["messages"]
        .as_array()
        .ok_or("messages must be an array")?;
    if raw_messages.is_empty() {
        return Err("messages must not be empty".into());
    }
    let mut messages = Vec::with_capacity(raw_messages.len());
    for message in raw_messages {
        let role = message["role"].as_str().ok_or("message role is missing")?;
        if !matches!(role, "system" | "user" | "assistant" | "tool")
            || message.get("tool_calls").is_some()
        {
            return Err("MLXL3 requires text messages without tool calls".into());
        }
        let content = match &message["content"] {
            Value::String(text) => text.clone(),
            Value::Array(parts) => {
                let mut text = String::new();
                for part in parts {
                    if part["type"] != "text" {
                        return Err("MLXL3 accepts text only".into());
                    }
                    text.push_str(part["text"].as_str().ok_or("text part is missing text")?);
                }
                text
            }
            _ => return Err("MLXL3 accepts text only".into()),
        };
        messages.push(json!({"role":role,"content":content}));
    }
    let raw_max_tokens = input
        .get("max_completion_tokens")
        .or_else(|| input.get("max_tokens"))
        .or_else(|| input.get("n_predict"));
    let max_tokens = match raw_max_tokens {
        Some(value) => value.as_i64().ok_or("max_tokens must be an integer")?,
        None => -1,
    };
    if max_tokens != -1 && max_tokens <= 0 {
        return Err("max_tokens must be positive".into());
    }
    if input
        .get("temperature")
        .is_some_and(|value| !value.is_null() && !value.is_number())
        || input
            .get("top_k")
            .is_some_and(|value| !value.is_null() && value.as_u64().is_none())
        || input
            .get("repeat_penalty")
            .is_some_and(|value| !value.is_null() && !value.is_number())
        || input
            .get("repetition_penalty")
            .is_some_and(|value| !value.is_null() && !value.is_number())
    {
        return Err("Invalid MLXL3 sampling parameters".into());
    }
    let temperature = input["temperature"].as_f64().unwrap_or(0.0);
    let top_k = input["top_k"].as_u64().unwrap_or(0);
    let penalty = input["repeat_penalty"]
        .as_f64()
        .or_else(|| input["repetition_penalty"].as_f64())
        .unwrap_or(1.0);
    if !(0.0..=2.0).contains(&temperature) || penalty <= 0.0 {
        return Err("Invalid MLXL3 sampling parameters".into());
    }
    Ok(
        json!({"type":"generate","request_id":NEXT_REQUEST.fetch_add(1, Ordering::Relaxed).to_string(),
        "messages":messages,"max_tokens":max_tokens,"temperature":temperature,
        "top_k":top_k,"repetition_penalty":penalty}),
    )
}

struct Output {
    answer: String,
    thinking: String,
    stats: Value,
    context_full: bool,
}

async fn send_sse(sender: &mut hyper::body::Sender, value: Value) -> Result<(), hyper::Error> {
    sender
        .send_data(Bytes::from(format!("data: {value}\n\n")))
        .await
}

async fn generate(
    state: &State,
    request: Value,
    id: &str,
    created: u64,
    sender: &mut Option<hyper::body::Sender>,
) -> Result<Output, String> {
    // The resident bridge serves one generation at a time.
    let mut bridge = state.bridge.lock().await;
    if let Some(sender) = sender.as_mut() {
        send_sse(sender, json!({"id":id,"object":"chat.completion.chunk","created":created,
            "model":state.model,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}))
            .await.map_err(|err| err.to_string())?;
    }
    bridge
        .stdin
        .write_all(format!("{request}\n").as_bytes())
        .await
        .map_err(|err| err.to_string())?;
    bridge.stdin.flush().await.map_err(|err| err.to_string())?;
    let mut answer = String::new();
    let mut thinking = String::new();
    let mut line = String::new();
    let mut disconnected = false;
    loop {
        line.clear();
        if bridge
            .stdout
            .read_line(&mut line)
            .await
            .map_err(|err| err.to_string())?
            == 0
        {
            return Err("MLXL3 bridge exited during generation".into());
        }
        let event: Value = serde_json::from_str(&line).map_err(|err| err.to_string())?;
        if event["request_id"] != request["request_id"] {
            continue;
        }
        match event["type"].as_str() {
            Some("delta") => {
                let text = event["text"].as_str().unwrap_or("");
                let is_thinking = event["phase"] == "thinking";
                if is_thinking {
                    thinking.push_str(text);
                } else {
                    answer.push_str(text);
                }
                if let Some(sender) = sender.as_mut().filter(|_| !disconnected) {
                    let delta = if is_thinking {
                        json!({"reasoning_content":text})
                    } else {
                        json!({"content":text})
                    };
                    if send_sse(sender, json!({"id":id,"object":"chat.completion.chunk","created":created,
                        "model":state.model,"choices":[{"index":0,"delta":delta,"finish_reason":null}]})).await.is_err() {
                        cancel(state.bridge_pid);
                        disconnected = true;
                    }
                }
            }
            Some("complete") => {
                return Ok(Output {
                    answer,
                    thinking,
                    stats: event["stats"].clone(),
                    context_full: event["context_full"].as_bool().unwrap_or(false),
                });
            }
            Some("error") => {
                return Err(event["message"]
                    .as_str()
                    .unwrap_or("generation failed")
                    .into());
            }
            Some("cancelled") => return Err("generation cancelled".into()),
            _ => {}
        }
    }
}

fn usage(stats: &Value) -> Value {
    let prompt = stats["prompt_tokens"].as_u64().unwrap_or(0);
    let completion = stats["generated_tokens"].as_u64().unwrap_or(0);
    json!({"prompt_tokens":prompt,"completion_tokens":completion,"total_tokens":total_tokens(prompt, completion)})
}

fn total_tokens(prompt: u64, completion: u64) -> u64 {
    prompt.saturating_add(completion)
}

#[cfg(kani)]
mod proofs {
    use super::total_tokens;

    #[kani::proof]
    fn usage_total_never_wraps() {
        let prompt: u64 = kani::any();
        let completion: u64 = kani::any();
        let total = total_tokens(prompt, completion);
        assert!(total >= prompt && total >= completion);
        if let Some(exact) = prompt.checked_add(completion) {
            assert_eq!(total, exact);
        } else {
            assert_eq!(total, u64::MAX);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Bridge, State, bridge_request, handle};
    use hyper::{Body, Request, StatusCode, body::to_bytes};
    use serde_json::json;
    use std::sync::Arc;
    use tokio::{io::BufReader, process::Command, sync::Mutex};

    #[test]
    fn text_request_maps_to_bridge_and_unsupported_inputs_fail() {
        let request = bridge_request(&json!({"model":"small","messages":[{"role":"user",
            "content":[{"type":"text","text":"Hello"},{"type":"text","text":" world"}]}],"max_tokens":12}), "small").unwrap();
        assert_eq!(request["messages"][0]["content"], "Hello world");
        assert_eq!(request["max_tokens"], 12);
        assert!(
            bridge_request(
                &json!({"model":"small","messages":[{"role":"user",
            "content":[{"type":"image_url","image_url":{"url":"data:"}}]}]}),
                "small"
            )
            .is_err()
        );
        assert!(
            bridge_request(
                &json!({"model":"small","messages":[{"role":"user","content":"Hi"}],
            "tools":[{"type":"function"}]}),
                "small"
            )
            .is_err()
        );
        assert!(
            bridge_request(
                &json!({"model":"small","messages":[{"role":"user","content":"Hi"}],
            "top_p":0.5}),
                "small"
            )
            .is_err()
        );
        assert!(
            bridge_request(
                &json!({"model":"small","messages":[{"role":"user","content":"Hi"}],
            "top_p":1}),
                "small"
            )
            .is_ok()
        );
    }

    #[test]
    fn malformed_and_silently_ignored_options_are_rejected() {
        let base = json!({"model":"small","messages":[{"role":"user","content":"Hi"}]});
        for (key, value) in [
            ("min_p", json!(0.1)),
            ("max_tokens", json!("many")),
            ("max_tokens", json!(0)),
            ("temperature", json!(3)),
            ("repeat_penalty", json!("high")),
            ("stream", json!("true")),
            ("stream_options", json!({"include_usage":"yes"})),
            ("tools", json!({"type":"function"})),
        ] {
            let mut request = base.clone();
            request[key] = value;
            assert!(bridge_request(&request, "small").is_err(), "accepted {key}");
        }
        assert!(bridge_request(&json!([]), "small").is_err());
    }

    #[tokio::test]
    async fn http_routes_enforce_session_key_and_validate_before_inference() {
        let mut child = Command::new("/bin/cat")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        let state = Arc::new(State {
            model: "small".into(),
            api_key: "secret".into(),
            bridge_pid: child.id().unwrap(),
            bridge: Mutex::new(Bridge {
                stdin: child.stdin.take().unwrap(),
                stdout: BufReader::new(child.stdout.take().unwrap()),
            }),
        });
        let health = handle(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
            state.clone(),
        )
        .await
        .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        let preflight = handle(
            Request::builder()
                .method("OPTIONS")
                .uri("/v1/chat/completions")
                .header("origin", "http://tauri.localhost")
                .header(
                    "access-control-request-headers",
                    "authorization,content-type",
                )
                .body(Body::empty())
                .unwrap(),
            state.clone(),
        )
        .await
        .unwrap();
        assert_eq!(preflight.status(), StatusCode::NO_CONTENT);
        assert_eq!(preflight.headers()["access-control-allow-origin"], "*");
        assert_eq!(
            preflight.headers()["access-control-allow-headers"],
            "authorization, content-type"
        );
        let unauthorized = handle(
            Request::builder()
                .uri("/v1/models")
                .body(Body::empty())
                .unwrap(),
            state.clone(),
        )
        .await
        .unwrap();
        assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
        let models = handle(
            Request::builder()
                .uri("/v1/models")
                .header("authorization", "Bearer secret")
                .body(Body::empty())
                .unwrap(),
            state.clone(),
        )
        .await
        .unwrap();
        assert_eq!(models.status(), StatusCode::OK);
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(
                &to_bytes(models.into_body()).await.unwrap()
            )
            .unwrap()["data"][0]["id"],
            "small"
        );
        let invalid = handle(Request::builder().method("POST").uri("/v1/chat/completions")
            .header("authorization", "Bearer secret")
            .body(Body::from(json!({"model":"small","messages":[{"role":"user","content":"Hi"}],"min_p":0.1}).to_string())).unwrap(), state).await.unwrap();
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        child.kill().await.unwrap();
        child.wait().await.unwrap();
    }
}
