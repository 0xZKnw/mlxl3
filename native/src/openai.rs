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
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    task::Poll,
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::{Mutex, OwnedSemaphorePermit, Semaphore, oneshot},
    time::{Instant, timeout, timeout_at},
};

#[path = "openai_io.rs"]
mod transport;

const MAX_BODY: usize = 16 * 1024 * 1024;
const MAX_EVENT: usize = 64 * 1024 * 1024;
const MAX_OUTPUT: usize = 16 * 1024 * 1024;

static NEXT_REQUEST: AtomicU64 = AtomicU64::new(1);

struct Bridge {
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

#[derive(Clone, Copy)]
struct Limits {
    body_bytes: usize,
    event_bytes: usize,
    output_bytes: usize,
    body: Duration,
    write: Duration,
    event: Duration,
    generation: Duration,
    send: Duration,
    drain: Duration,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            body_bytes: MAX_BODY,
            event_bytes: MAX_EVENT,
            output_bytes: MAX_OUTPUT,
            body: Duration::from_secs(10),
            write: Duration::from_secs(10),
            event: Duration::from_secs(300),
            generation: Duration::from_secs(1800),
            send: Duration::from_secs(10),
            drain: Duration::from_secs(10),
        }
    }
}

struct State {
    model: String,
    api_key: String,
    bridge_info: Value,
    bridge: Mutex<Bridge>,
    child: Arc<Mutex<Child>>,
    admission: Arc<Semaphore>,
    healthy: AtomicBool,
    limits: Limits,
}

impl State {
    async fn bridge_alive(&self) -> bool {
        if !self.healthy.load(Ordering::Acquire) {
            return false;
        }
        if !matches!(self.child.lock().await.try_wait(), Ok(None)) {
            self.healthy.store(false, Ordering::Release);
            return false;
        }
        true
    }

    async fn request_cancel(&self) {
        // try_wait can reap a dead PID. Keep identity and signal under the same lock.
        let mut child = self.child.lock().await;
        if matches!(child.try_wait(), Ok(None)) {
            if let Some(pid) = child.id() {
                cancel(pid);
            }
        } else {
            self.healthy.store(false, Ordering::Release);
        }
    }

    async fn invalidate(&self) {
        self.healthy.store(false, Ordering::Release);
        if let Err(error) = cleanup_child(&self.child, self.limits.drain).await {
            eprintln!("MLXL3 bridge cleanup failed: {error:#}");
        }
    }
}

fn listen_addr(host: &str, port: u16, api_key: &str) -> Result<SocketAddr> {
    let ip = if let Some(inner) = host.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
        IpAddr::V6(inner.parse().context("invalid IPv6 listen address")?)
    } else {
        host.parse::<IpAddr>().context("invalid listen address")?
    };
    if !ip.is_loopback() && api_key.is_empty() {
        bail!("an API key is required when listening outside loopback");
    }
    Ok(SocketAddr::new(ip, port))
}

fn shutdown_signal() -> Result<impl std::future::Future<Output = ()>> {
    #[cfg(unix)]
    {
        // Register both handlers before spawning the model process.
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        let mut interrupt =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::interrupt())?;
        Ok(async move {
            tokio::select! {
                _ = terminate.recv() => {},
                _ = interrupt.recv() => {},
            }
        })
    }
    #[cfg(not(unix))]
    Ok(async {
        let _ = tokio::signal::ctrl_c().await;
    })
}

async fn cleanup_child(child: &Mutex<Child>, wait: Duration) -> Result<()> {
    let mut child = child.lock().await;
    if matches!(child.try_wait(), Ok(Some(_))) {
        return Ok(());
    }
    // A concurrent exit may make kill fail; wait still determines whether it was reaped.
    let _ = child.start_kill();
    timeout(wait, child.wait())
        .await
        .context("bridge cleanup timed out; child reap is unconfirmed")?
        .context("cannot reap bridge child")?;
    Ok(())
}

async fn serve_resident(
    child: Arc<Mutex<Child>>,
    listener: TcpListener,
    model: &str,
    api_key: &str,
    limits: Limits,
    startup_timeout: Duration,
    shutdown: impl std::future::Future<Output = ()>,
) -> Result<()> {
    let stderr = child.lock().await.stderr.take().map(|stderr| {
        tokio::spawn(async move {
            let mut reader = BufReader::new(stderr);
            let mut bytes = Vec::new();
            loop {
                bytes.clear();
                match (&mut reader)
                    .take(64 * 1024)
                    .read_until(b'\n', &mut bytes)
                    .await
                {
                    Ok(0) | Err(_) => break,
                    Ok(_) => eprintln!("mlxl3 bridge: {}", String::from_utf8_lossy(&bytes)),
                }
            }
        })
    });
    let result: Result<()> = async {
        let (stdin, stdout) = {
            let mut process = child.lock().await;
            (process.stdin.take().context("bridge stdin unavailable")?,
                process.stdout.take().context("bridge stdout unavailable")?)
        };
        let mut stdout = BufReader::new(stdout);
        let mut shutdown = std::pin::pin!(shutdown);
        let ready = tokio::select! {
            _ = &mut shutdown => return Ok(()),
            result = timeout(startup_timeout, async {
                let mut line = Vec::new();
                loop {
                    let event = read_event(&mut stdout, &mut line, limits.event_bytes)
                        .await.map_err(anyhow::Error::msg)?;
                    match event["type"].as_str() {
                        Some("ready") => return Ok::<_, anyhow::Error>(event),
                        Some("error") => bail!("{}", event["message"].as_str().unwrap_or("model load failed")),
                        _ => {}
                    }
                }
            }) => result.context("model load timed out")??,
        };
        let actual_port = listener.local_addr()?.port();
        let state = Arc::new(State {
            model: model.to_owned(),
            api_key: api_key.to_owned(),
            bridge_info: bridge_capabilities(&ready),
            bridge: Mutex::new(Bridge { stdin, stdout }),
            child: child.clone(),
            admission: Arc::new(Semaphore::new(1)),
            healthy: AtomicBool::new(true),
            limits,
        });
        let service = make_service_fn(move |_| {
            let state = state.clone();
            async move { Ok::<_, Infallible>(service_fn(move |request| handle(request, state.clone()))) }
        });
        let server = Server::builder(transport::Incoming::new(listener, limits.send)?)
            .http1_header_read_timeout(limits.body)
            .serve(service);
        println!("{}", json!({"type":"ready","model":model,"port":actual_port}));
        use std::io::Write;
        std::io::stdout().flush()?;
        tokio::select! {
            result = server => result.context("OpenAI HTTP server failed"),
            _ = &mut shutdown => Ok(()),
        }
    }.await;
    // Every startup/server exit passes here, including malformed ready and shutdown during load.
    let cleanup = cleanup_child(&child, limits.drain).await;
    let stderr_cleanup: Result<()> = if let Some(mut stderr) = stderr {
        match timeout(limits.drain, &mut stderr).await {
            Ok(result) => result.context("bridge stderr task failed"),
            Err(_) => {
                stderr.abort();
                let _ = stderr.await;
                Err(anyhow::anyhow!(
                    "bridge stderr drainage timed out; task aborted and joined"
                ))
            }
        }
    } else {
        Ok(())
    };
    if let Err(error) = cleanup {
        eprintln!("MLXL3 bridge cleanup failed: {error:#}");
        return Err(result
            .err()
            .unwrap_or(error)
            .context("bridge cleanup was not confirmed"));
    }
    if let Err(error) = stderr_cleanup {
        eprintln!("MLXL3 stderr cleanup failed: {error:#}");
        return Err(result
            .err()
            .unwrap_or(error)
            .context("bridge stderr did not drain normally"));
    }
    result
}

#[cfg_attr(not(all(feature = "mlx", feature = "chat")), allow(dead_code))]
pub async fn serve(
    registry: Option<&Path>,
    model: &str,
    host: &str,
    port: u16,
    api_key: &str,
    context_length: i32,
) -> Result<()> {
    let addr = listen_addr(host, port, api_key)?;
    let listener = TcpListener::bind(addr).context("cannot bind OpenAI API")?;
    listener.set_nonblocking(true)?;
    let shutdown = shutdown_signal()?;
    let executable = std::env::current_exe()?;
    let mut command = Command::new(executable);
    if let Some(registry) = registry {
        command.arg("--registry").arg(registry);
    }
    let child = command
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
    serve_resident(
        Arc::new(Mutex::new(child)),
        listener,
        model,
        api_key,
        Limits::default(),
        Duration::from_secs(180),
        shutdown,
    )
    .await
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
        .body(Body::from(value.to_string()))
        .expect("valid response")
}

fn bounded_error(message: &str) -> &str {
    let mut end = message.len().min(4096);
    while !message.is_char_boundary(end) {
        end -= 1;
    }
    &message[..end]
}

fn error(status: StatusCode, message: impl Into<String>) -> Response<Body> {
    let message = message.into();
    json_response(
        status,
        json!({"error":{"message":bounded_error(&message),"type":"invalid_request_error"}}),
    )
}

fn bridge_capabilities(ready: &Value) -> Value {
    let mut public = serde_json::Map::new();
    for key in [
        "mtp_supported",
        "dflash_supported",
        "memory_saver_supported",
    ] {
        if let Some(value) = ready[key].as_bool() {
            public.insert(key.into(), json!(value));
        }
    }
    for key in ["bridge_protocol", "model_context_limit"] {
        if let Some(value) = ready[key].as_u64().filter(|n| *n <= i32::MAX as u64) {
            public.insert(key.into(), json!(value));
        }
    }
    for key in ["runtime_version", "mlx_version"] {
        if let Some(value) = ready[key].as_str().filter(|s| {
            !s.is_empty()
                && s.len() <= 64
                && s.bytes()
                    .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'+' | b'-'))
        }) {
            public.insert(key.into(), json!(value));
        }
    }
    Value::Object(public)
}

fn api_capabilities(state: &State) -> Value {
    json!({"schema":1,"model":state.model,
        "api":{"chat_completions":true,"stream":true,"stream_usage":true,"text_only":true,
            "sampling":{"temperature":true,"top_k":true,"repetition_penalty":true,
                "top_p":false,"frequency_penalty":false,"presence_penalty":false},
            "stop_sequences":false,"tool_calls":false,"json_schema":false,"images":false,
            "seed":false,"logit_bias":false,"multiple_choices":false,"mtp":false,"dflash":false},
        "bridge":state.bridge_info})
}

fn loopback_authority(authority: &str) -> bool {
    if authority
        .bytes()
        .any(|b| matches!(b, b'@' | b'/' | b'?' | b'#'))
    {
        return false;
    }
    let (host, port) = if let Some(rest) = authority.strip_prefix('[') {
        let Some((host, suffix)) = rest.split_once(']') else {
            return false;
        };
        if host.parse::<std::net::Ipv6Addr>().is_err() {
            return false;
        }
        if suffix.is_empty() {
            (host, None)
        } else if let Some(port) = suffix.strip_prefix(':') {
            (host, Some(port))
        } else {
            return false;
        }
    } else if let Some((host, port)) = authority.split_once(':') {
        (host, Some(port))
    } else {
        (authority, None)
    };
    if port.is_some_and(|port| {
        port.is_empty() || !port.bytes().all(|b| b.is_ascii_digit()) || port.parse::<u16>().is_err()
    }) {
        return false;
    }
    host.eq_ignore_ascii_case("localhost")
        || host.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
}

fn origin_allowed(origin: &str) -> bool {
    let Some((scheme, authority)) = origin.split_once("://") else {
        return false;
    };
    (scheme.eq_ignore_ascii_case("http") || scheme.eq_ignore_ascii_case("https"))
        && loopback_authority(authority)
}

async fn handle(request: Request<Body>, state: Arc<State>) -> Result<Response<Body>, Infallible> {
    let hosts: Vec<_> = request.headers().get_all("host").iter().collect();
    if hosts.len() > 1
        || (state.api_key.is_empty()
            && hosts
                .first()
                .is_some_and(|value| !value.to_str().is_ok_and(loopback_authority)))
    {
        return Ok(error(
            StatusCode::FORBIDDEN,
            "Host is not allowed without an API key",
        ));
    }
    let origins: Vec<_> = request.headers().get_all("origin").iter().collect();
    let origin = match origins.as_slice() {
        [] => None,
        [value] if value.to_str().is_ok_and(origin_allowed) => Some((*value).clone()),
        _ => return Ok(error(StatusCode::FORBIDDEN, "Origin is not allowed")),
    };
    let mut response = handle_inner(request, state).await;
    response
        .headers_mut()
        .insert("vary", hyper::header::HeaderValue::from_static("Origin"));
    if let Some(origin) = origin {
        response
            .headers_mut()
            .insert("access-control-allow-origin", origin);
    }
    Ok(response)
}

async fn handle_inner(request: Request<Body>, state: Arc<State>) -> Response<Body> {
    let path = request.uri().path();
    if request.method() == hyper::Method::OPTIONS {
        if !matches!(
            path,
            "/health" | "/ready" | "/v1/models" | "/v1/capabilities" | "/v1/chat/completions"
        ) {
            return error(StatusCode::NOT_FOUND, "Unsupported endpoint");
        }
        if request
            .headers()
            .get("access-control-request-method")
            .is_some_and(|method| method != "GET" && method != "POST")
            || request
                .headers()
                .get("access-control-request-headers")
                .is_some_and(|value| {
                    value.to_str().map_or(true, |headers| {
                        headers.split(',').any(|name| {
                            !matches!(
                                name.trim().to_ascii_lowercase().as_str(),
                                "authorization" | "content-type"
                            )
                        })
                    })
                })
        {
            return error(StatusCode::BAD_REQUEST, "Unsupported preflight request");
        }
        return Response::builder()
            .status(StatusCode::NO_CONTENT)
            .header("access-control-allow-methods", "GET, POST, OPTIONS")
            .header(
                "access-control-allow-headers",
                "authorization, content-type",
            )
            .body(Body::empty())
            .expect("valid preflight response");
    }
    if request.method() == hyper::Method::GET && matches!(path, "/health" | "/ready") {
        let healthy = state.bridge_alive().await;
        let busy = state.admission.available_permits() == 0;
        let ready = healthy && !busy;
        let status = if path == "/health" || ready {
            StatusCode::OK
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        };
        return json_response(
            status,
            json!({"status":if path == "/health" {"ok"} else if ready {"ready"} else {"unavailable"}, "ready":ready,"busy":busy,"bridge_healthy":healthy}),
        );
    }
    if !state.api_key.is_empty() && request.headers().get_all("authorization").iter().count() != 1 {
        return error(StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    if !state.api_key.is_empty()
        && request
            .headers()
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            != Some(format!("Bearer {}", state.api_key).as_str())
    {
        return error(StatusCode::UNAUTHORIZED, "Invalid API key");
    }
    if request.method() == hyper::Method::GET && path == "/v1/capabilities" {
        return json_response(StatusCode::OK, api_capabilities(&state));
    }
    if request.method() == hyper::Method::GET && path == "/v1/models" {
        return json_response(
            StatusCode::OK,
            json!({"object":"list","data":[{"id":state.model,"object":"model","owned_by":"mlxl3"}]}),
        );
    }
    if request.method() != hyper::Method::POST || path != "/v1/chat/completions" {
        return error(StatusCode::NOT_FOUND, "Unsupported endpoint");
    }
    if !state.bridge_alive().await {
        return error(
            StatusCode::SERVICE_UNAVAILABLE,
            "MLXL3 bridge is unavailable",
        );
    }
    let permit = match state.admission.clone().try_acquire_owned() {
        Ok(permit) => permit,
        Err(_) => {
            let mut response = error(StatusCode::TOO_MANY_REQUESTS, "MLXL3 is busy; retry later");
            response
                .headers_mut()
                .insert("retry-after", hyper::header::HeaderValue::from_static("1"));
            return response;
        }
    };
    if request.body().size_hint().lower() > state.limits.body_bytes as u64 {
        let mut response = error(StatusCode::PAYLOAD_TOO_LARGE, "Request body exceeds 16 MiB");
        response.headers_mut().insert(
            "connection",
            hyper::header::HeaderValue::from_static("close"),
        );
        return response;
    }
    let mut incoming = request.into_body();
    let body = timeout(state.limits.body, async {
        let mut bytes = Vec::new();
        while let Some(chunk) = incoming.data().await {
            let chunk = chunk.map_err(|err| (StatusCode::BAD_REQUEST, err.to_string()))?;
            if chunk.len() > state.limits.body_bytes.saturating_sub(bytes.len()) {
                return Err((
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "Request body exceeds 16 MiB".into(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    })
    .await;
    let bytes = match body {
        Ok(Ok(body)) => body,
        other => {
            let (status, message) = match other {
                Ok(Err(error)) => error,
                _ => (StatusCode::REQUEST_TIMEOUT, "Request body timed out".into()),
            };
            let mut response = error(status, message);
            response.headers_mut().insert(
                "connection",
                hyper::header::HeaderValue::from_static("close"),
            );
            return response;
        }
    };
    let input: Value = match serde_json::from_slice(&bytes) {
        Ok(input) => input,
        Err(err) => return error(StatusCode::BAD_REQUEST, err.to_string()),
    };
    let request = match bridge_request(&input, &state.model) {
        Ok(request) => request,
        Err(err) => return error(StatusCode::BAD_REQUEST, err),
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
    if input["stream"].as_bool().unwrap_or(false) {
        let include_usage = input["stream_options"]["include_usage"]
            .as_bool()
            .unwrap_or(false);
        let (sender, body) = Body::channel();
        tokio::spawn(async move {
            let _permit = permit;
            let mut sender = Some(sender);
            let result = generate(&state, request, &id, created, &mut sender, &mut None).await;
            if let Some(mut sender) = sender {
                let final_value = match result {
                    Ok(output) => {
                        let value = json!({"id":id,"object":"chat.completion.chunk","created":created,"model":state.model,
                            "choices":[{"index":0,"delta":{},"finish_reason":output.reason.openai_finish_reason()}]});
                        if send_sse(&mut sender, value, state.limits.send)
                            .await
                            .is_err()
                        {
                            return;
                        }
                        if include_usage {
                            Some(
                                json!({"id":id,"object":"chat.completion.chunk","created":created,"model":state.model,"choices":[],"usage":usage(&output.stats)}),
                            )
                        } else {
                            None
                        }
                    }
                    Err(err) => {
                        let _ = send_sse(
                            &mut sender,
                            json!({"error":{"message":bounded_error(&err),"type":"server_error"}}),
                            state.limits.send,
                        )
                        .await;
                        return;
                    }
                };
                if let Some(value) = final_value
                    && send_sse(&mut sender, value, state.limits.send)
                        .await
                        .is_err()
                {
                    return;
                }
                let _ = timeout(
                    state.limits.send,
                    sender.send_data(Bytes::from("data: [DONE]\n\n")),
                )
                .await;
            }
        });
        return Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(body)
            .expect("valid stream response");
    }
    let (reply, receiver) = oneshot::channel();
    let worker_state = state.clone();
    let worker_id = id.clone();
    tokio::spawn(async move {
        let mut reply = Some(reply);
        let result = generate(
            &worker_state,
            request,
            &worker_id,
            created,
            &mut None,
            &mut reply,
        )
        .await;
        if let Some(reply) = reply {
            let _ = reply.send(result.map(|output| (output, permit)));
        }
    });
    match receiver.await {
        Ok(Ok((output, permit))) => completion_response(
            json!({"id":id,"object":"chat.completion","created":created,"model":state.model,
            "choices":[{"index":0,"message":{"role":"assistant","content":output.answer,"reasoning_content":output.thinking},
                "finish_reason":output.reason.openai_finish_reason()}],"usage":usage(&output.stats)}),
            permit,
            state.limits.send,
        ),
        Ok(Err(err)) => error(StatusCode::BAD_GATEWAY, err),
        Err(_) => error(StatusCode::BAD_GATEWAY, "MLXL3 generation worker stopped"),
    }
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
    let response_schema = match input
        .get("response_format")
        .filter(|value| !value.is_null())
    {
        None => None,
        Some(format) if format["type"] == "text" => None,
        Some(format) if format["type"] == "json_object" => Some(r#"{"type":"object"}"#.to_string()),
        Some(format) if format["type"] == "json_schema" => {
            let schema = format["json_schema"]
                .get("schema")
                .ok_or("response_format.json_schema.schema is required")?;
            let text = serde_json::to_string(schema).map_err(|error| error.to_string())?;
            mlxl3_native::structured::Schema::parse(&text).map_err(|error| error.to_string())?;
            Some(text)
        }
        _ => return Err("Unsupported response_format".into()),
    };
    if input.get("seed").is_some_and(|value| !value.is_null())
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
    // The bridge consumes f32: reject overflow/underflow before any prefill starts.
    let bridge_penalty = penalty as f32;
    if !(0.0..=2.0).contains(&temperature) || !bridge_penalty.is_finite() || bridge_penalty <= 0.0 {
        return Err("Invalid MLXL3 sampling parameters".into());
    }
    Ok(
        json!({"type":"generate","request_id":NEXT_REQUEST.fetch_add(1, Ordering::Relaxed).to_string(),
        "messages":messages,"max_tokens":max_tokens,"temperature":temperature,
        "top_k":top_k,"repetition_penalty":penalty,"response_schema":response_schema}),
    )
}

struct Output {
    answer: String,
    thinking: String,
    stats: TokenUsage,
    reason: mlxl3_native::GenerationStopReason,
}

struct TokenUsage {
    prompt: u64,
    completion: u64,
}

impl TokenUsage {
    fn new(prompt: u64, completion: u64, maximum: Option<u64>) -> Option<Self> {
        if prompt == 0 || maximum.is_some_and(|maximum| completion > maximum) {
            return None;
        }
        Some(Self { prompt, completion })
    }

    fn parse(stats: &Value, maximum: Option<u64>) -> Option<Self> {
        Self::new(
            stats["prompt_tokens"].as_u64()?,
            stats["generated_tokens"].as_u64()?,
            maximum,
        )
    }
}

type Reply = oneshot::Sender<Result<(Output, OwnedSemaphorePermit), String>>;

async fn send_bytes(
    sender: &mut hyper::body::Sender,
    bytes: &[u8],
    wait: Duration,
) -> Result<(), String> {
    let deadline = Instant::now() + wait;
    // Bound each producer-channel frame; Hyper and the socket have their own buffers.
    for chunk in bytes.chunks(16 * 1024) {
        if Instant::now() >= deadline {
            return Err("HTTP client write timed out".into());
        }
        timeout_at(deadline, sender.send_data(Bytes::copy_from_slice(chunk)))
            .await
            .map_err(|_| "HTTP client write timed out".to_owned())?
            .map_err(|err| err.to_string())?;
    }
    Ok(())
}

async fn send_sse(
    sender: &mut hyper::body::Sender,
    value: Value,
    wait: Duration,
) -> Result<(), String> {
    send_bytes(sender, format!("data: {value}\n\n").as_bytes(), wait).await
}

fn completion_response(
    value: Value,
    permit: OwnedSemaphorePermit,
    wait: Duration,
) -> Response<Body> {
    let (mut sender, body) = Body::channel();
    tokio::spawn(async move {
        let _permit = permit;
        if send_bytes(&mut sender, value.to_string().as_bytes(), wait)
            .await
            .is_err()
        {
            sender.abort();
        }
    });
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(body)
        .expect("valid completion response")
}

async fn response_closed(sender: &mut Option<hyper::body::Sender>, reply: &mut Option<Reply>) {
    if let Some(sender) = sender {
        // Hyper0.14 registers the close waker in poll_ready, even when writable.
        std::future::poll_fn(|cx| match sender.poll_ready(cx) {
            Poll::Ready(Err(_)) => Poll::Ready(()),
            _ => Poll::Pending,
        })
        .await;
    } else if let Some(reply) = reply {
        reply.closed().await;
    } else {
        std::future::pending::<()>().await;
    }
}

async fn read_event(
    reader: &mut BufReader<ChildStdout>,
    line: &mut Vec<u8>,
    maximum: usize,
) -> Result<Value, String> {
    let remaining = maximum
        .checked_sub(line.len())
        .ok_or("Bridge event exceeds 64 MiB")?;
    // read_until is cancellation safe: partial JSON survives into cancellation drainage.
    let n = (&mut *reader)
        .take(remaining as u64 + 1)
        .read_until(b'\n', line)
        .await
        .map_err(|err| err.to_string())?;
    if n == 0 {
        return Err("MLXL3 bridge exited during generation".into());
    }
    if line.len() > maximum || line.last() != Some(&b'\n') {
        return Err("Bridge event exceeds 64 MiB or is truncated".into());
    }
    let event =
        serde_json::from_slice(line).map_err(|err| format!("Invalid MLXL3 bridge event: {err}"))?;
    line.clear();
    Ok(event)
}

async fn cancel_and_drain(
    state: &State,
    bridge: &mut Bridge,
    request_id: &Value,
    line: &mut Vec<u8>,
) {
    state.request_cancel().await;
    let deadline = Instant::now() + state.limits.drain;
    loop {
        if Instant::now() >= deadline {
            state.invalidate().await;
            return;
        }
        match timeout_at(
            deadline,
            read_event(&mut bridge.stdout, line, state.limits.event_bytes),
        )
        .await
        {
            Ok(Ok(event)) => {
                if &event["request_id"] != request_id {
                    continue;
                }
                if matches!(
                    event["type"].as_str(),
                    Some("complete" | "cancelled" | "error")
                ) {
                    return;
                }
                // Only this ack follows the flag reset; never flood signals while draining deltas.
                if event["type"] == "generation_mode" {
                    state.request_cancel().await;
                }
            }
            failure => {
                eprintln!("MLXL3 cancellation drainage failed: {failure:?}");
                state.invalidate().await;
                return;
            }
        }
    }
}

async fn generate(
    state: &State,
    request: Value,
    id: &str,
    created: u64,
    sender: &mut Option<hyper::body::Sender>,
    reply: &mut Option<Reply>,
) -> Result<Output, String> {
    let schema = request["response_schema"]
        .as_str()
        .map(mlxl3_native::structured::Schema::parse)
        .transpose()
        .map_err(|error| error.to_string())?;
    // The resident bridge serves one generation at a time.
    let mut bridge = state.bridge.lock().await;
    if reply.as_ref().is_some_and(Reply::is_closed) {
        return Err("HTTP client disconnected".into());
    }
    if let Some(sender) = sender.as_mut() {
        send_sse(
            sender,
            json!({"id":id,"object":"chat.completion.chunk","created":created,"model":state.model,
            "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}),
            state.limits.send,
        )
        .await?;
    }
    let deadline = Instant::now() + state.limits.generation;
    let written = timeout(state.limits.write, async {
        bridge
            .stdin
            .write_all(format!("{request}\n").as_bytes())
            .await?;
        bridge.stdin.flush().await
    })
    .await;
    if !matches!(written, Ok(Ok(()))) {
        state.invalidate().await;
        return Err("MLXL3 bridge write failed or timed out".into());
    }
    let mut answer = String::new();
    let mut thinking = String::new();
    let mut line = Vec::new();
    loop {
        if Instant::now() >= deadline {
            cancel_and_drain(state, &mut bridge, &request["request_id"], &mut line).await;
            return Err("MLXL3 generation timed out".into());
        }
        let event_deadline = (Instant::now() + state.limits.event).min(deadline);
        let event = tokio::select! {
            _ = response_closed(sender, reply) => {
                cancel_and_drain(state, &mut bridge, &request["request_id"], &mut line).await;
                return Err("HTTP client disconnected".into());
            }
            event = timeout_at(event_deadline, read_event(&mut bridge.stdout, &mut line, state.limits.event_bytes)) => event,
        };
        let event = match event {
            Ok(Ok(event)) => event,
            Ok(Err(err)) => {
                state.invalidate().await;
                return Err(err);
            }
            Err(_) => {
                cancel_and_drain(state, &mut bridge, &request["request_id"], &mut line).await;
                return Err("MLXL3 generation timed out".into());
            }
        };
        if event["request_id"] != request["request_id"] {
            continue;
        }
        match event["type"].as_str() {
            Some("delta") => {
                let Some(text) = event["text"].as_str() else {
                    state.invalidate().await;
                    return Err("Invalid bridge delta".into());
                };
                if text.len()
                    > state
                        .limits
                        .output_bytes
                        .saturating_sub(answer.len() + thinking.len())
                {
                    cancel_and_drain(state, &mut bridge, &request["request_id"], &mut line).await;
                    return Err("MLXL3 response exceeds 16 MiB".into());
                }
                let is_thinking = event["phase"] == "thinking";
                if is_thinking {
                    thinking.push_str(text);
                } else {
                    answer.push_str(text);
                }
                if let Some(sender) = sender.as_mut().filter(|_| schema.is_none()) {
                    let delta = if is_thinking {
                        json!({"reasoning_content":text})
                    } else {
                        json!({"content":text})
                    };
                    if let Err(err) = send_sse(sender, json!({"id":id,"object":"chat.completion.chunk","created":created,
                        "model":state.model,"choices":[{"index":0,"delta":delta,"finish_reason":null}]}), state.limits.send).await {
                        cancel_and_drain(state, &mut bridge, &request["request_id"], &mut line).await;
                        return Err(err);
                    }
                }
            }
            Some("complete") => {
                let Some(reason) = event["stop_reason"]
                    .as_str()
                    .and_then(mlxl3_native::GenerationStopReason::parse)
                else {
                    state.invalidate().await;
                    return Err("Missing or invalid bridge stop_reason".into());
                };
                let Some(stats) =
                    TokenUsage::parse(&event["stats"], request["max_tokens"].as_u64())
                else {
                    state.invalidate().await;
                    return Err("Missing or invalid bridge token usage".into());
                };
                if let Some(schema) = &schema {
                    answer = schema
                        .validate(&answer)
                        .map_err(|error| error.to_string())?;
                    thinking.clear();
                    if let Some(sender) = sender.as_mut() {
                        send_sse(sender, json!({"id":id,"object":"chat.completion.chunk","created":created,
                            "model":state.model,"choices":[{"index":0,"delta":{"content":answer},"finish_reason":null}]}), state.limits.send)
                            .await.map_err(|error| error.to_string())?;
                    }
                }
                return Ok(Output {
                    answer,
                    thinking,
                    stats,
                    reason,
                });
            }
            Some("error") => {
                return Err(bounded_error(
                    event["message"].as_str().unwrap_or("generation failed"),
                )
                .into());
            }
            Some("cancelled") => return Err("generation cancelled".into()),
            _ => {}
        }
    }
}

fn usage(stats: &TokenUsage) -> Value {
    json!({"prompt_tokens":stats.prompt,"completion_tokens":stats.completion,
        "total_tokens":total_tokens(stats.prompt, stats.completion)})
}

fn total_tokens(prompt: u64, completion: u64) -> u64 {
    prompt.saturating_add(completion)
}

#[cfg(kani)]
mod proofs {
    use super::total_tokens;

    #[kani::proof]
    #[kani::unwind(64)]
    fn origin_suffix_cannot_escape_loopback_authority() {
        let first: u8 = kani::any();
        let second: u8 = kani::any();
        let mut bytes = *b"http://localhost00";
        bytes[16] = first;
        bytes[17] = second;
        if let Ok(origin) = std::str::from_utf8(&bytes) {
            let allowed = super::origin_allowed(origin);
            assert_eq!(allowed, first == b':' && second.is_ascii_digit());
            kani::cover!(first == b':' && second == b'8' && allowed);
            kani::cover!(first == b'/' && second == b'x' && !allowed);
        }
    }

    #[kani::proof]
    fn validated_usage_respects_budget_without_total_overflow() {
        let prompt: u64 = kani::any();
        let completion: u64 = kani::any();
        let maximum: Option<u64> = kani::any();
        let usage = super::TokenUsage::new(prompt, completion, maximum);
        assert_eq!(
            usage.is_some(),
            prompt > 0 && maximum.is_none_or(|limit| completion <= limit)
        );
        if let Some(usage) = usage {
            let total = total_tokens(usage.prompt, usage.completion);
            assert!(total >= usage.prompt && total >= usage.completion);
            if let Some(exact) = prompt.checked_add(completion) {
                assert_eq!(total, exact);
            } else {
                assert_eq!(total, u64::MAX);
            }
        }
        kani::cover!(prompt > 0 && completion == 0 && maximum == Some(1));
        kani::cover!(prompt == 0 && completion > 0);
        kani::cover!(maximum.is_some_and(|limit| completion > limit));
    }

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
    use super::{Bridge, Limits, State, bridge_request, handle};
    use hyper::{Body, Request, StatusCode, body::to_bytes};
    use serde_json::json;
    use std::sync::{Arc, atomic::AtomicBool};
    use tokio::{
        io::BufReader,
        process::Command,
        sync::{Mutex, Semaphore},
    };

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
    fn structured_requests_validate_schema_before_inference() {
        let mut request = json!({"model":"small","messages":[{"role":"user","content":"Hi"}],
            "response_format":{"type":"json_schema","json_schema":{"name":"result","strict":true,
                "schema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"],"additionalProperties":false}}}});
        let mapped = bridge_request(&request, "small").unwrap();
        assert!(
            mapped["response_schema"]
                .as_str()
                .unwrap()
                .contains("additionalProperties")
        );
        request["response_format"] = json!({"type":"json_object"});
        assert_eq!(
            bridge_request(&request, "small").unwrap()["response_schema"],
            r#"{"type":"object"}"#
        );
        for format in [
            json!({"type":"json_schema"}),
            json!({"type":"json_schema","json_schema":{"schema":{"$ref":"x"}}}),
            json!({"type":"unknown"}),
        ] {
            request["response_format"] = format;
            assert!(bridge_request(&request, "small").is_err());
        }
    }

    #[tokio::test]
    async fn structured_http_never_publishes_invalid_bridge_output() {
        // Exercise real pipes and HTTP/SSE with a deliberately nonconforming bridge.
        for (text, valid) in [
            (r#"{"name":"Ada"}"#, true),
            (r#"{"wrong":"Ada"}"#, false),
            ("", false),
            (r#"{"name":"Ada""#, false),
        ] {
            for stream in [false, true] {
                let script = r#"import json,sys
for line in sys.stdin:
    request=json.loads(line)
    print(json.dumps(dict(type='delta', request_id=request['request_id'], phase='answer', text=sys.argv[1])),flush=True)
    print(json.dumps(dict(type='complete', request_id=request['request_id'], stop_reason='eos', stats={'prompt_tokens':2,'generated_tokens':1})),flush=True)
"#;
                let mut child = Command::new("python3")
                    .arg("-u")
                    .arg("-c")
                    .arg(script)
                    .arg(text)
                    .stdin(std::process::Stdio::piped())
                    .stdout(std::process::Stdio::piped())
                    .kill_on_drop(true)
                    .spawn()
                    .unwrap();
                let state = Arc::new(State {
                    model: "small".into(),
                    api_key: String::new(),
                    bridge_info: json!({}),
                    bridge: Mutex::new(Bridge {
                        stdin: child.stdin.take().unwrap(),
                        stdout: BufReader::new(child.stdout.take().unwrap()),
                    }),
                    child: Arc::new(Mutex::new(child)),
                    admission: Arc::new(Semaphore::new(1)),
                    healthy: AtomicBool::new(true),
                    limits: Limits::default(),
                });
                let input = json!({"model":"small","messages":[{"role":"user","content":"Hello"}],"stream":stream,
                    "response_format":{"type":"json_schema","json_schema":{"schema":{"type":"object","required":["name"]}}}});
                let response = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    handle(
                        Request::builder()
                            .method("POST")
                            .uri("/v1/chat/completions")
                            .body(Body::from(input.to_string()))
                            .unwrap(),
                        state.clone(),
                    ),
                )
                .await
                .unwrap()
                .unwrap();
                assert_eq!(
                    response.status(),
                    if stream || valid {
                        StatusCode::OK
                    } else {
                        StatusCode::BAD_GATEWAY
                    }
                );
                let body = tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    to_bytes(response.into_body()),
                )
                .await
                .unwrap()
                .unwrap();
                let body = String::from_utf8(body.to_vec()).unwrap();
                if valid {
                    assert!(body.contains("Ada"));
                } else {
                    assert!(body.contains("error"));
                    assert!(!body.contains("wrong"));
                    assert!(!body.contains("Ada"));
                    assert!(!body.contains("[DONE]"));
                }
                state.invalidate().await;
            }
        }
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
            bridge_info: json!({}),

            bridge: Mutex::new(Bridge {
                stdin: child.stdin.take().unwrap(),
                stdout: BufReader::new(child.stdout.take().unwrap()),
            }),
            child: Arc::new(Mutex::new(child)),
            admission: Arc::new(Semaphore::new(1)),
            healthy: AtomicBool::new(true),
            limits: Limits::default(),
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
                .header("origin", "http://localhost:3000")
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
        assert_eq!(
            preflight.headers()["access-control-allow-origin"],
            "http://localhost:3000"
        );
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
            .body(Body::from(json!({"model":"small","messages":[{"role":"user","content":"Hi"}],"min_p":0.1}).to_string())).unwrap(), state.clone()).await.unwrap();
        assert_eq!(invalid.status(), StatusCode::BAD_REQUEST);
        state.invalidate().await;
    }

    #[tokio::test]
    async fn finish_reason_honors_explicit_budget() {
        let script = r#"import json,sys
for line in sys.stdin:
 r=json.loads(line)
 print(json.dumps({'type':'complete','request_id':r['request_id'],'stop_reason':'max_tokens','context_full':False,'stats':{'prompt_tokens':2,'generated_tokens':1}}),flush=True)
"#;
        let mut child = Command::new("python3")
            .args(["-u", "-c", script])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let state = Arc::new(State {
            model: "small".into(),
            api_key: String::new(),
            bridge_info: json!({}),

            bridge: Mutex::new(Bridge {
                stdin: child.stdin.take().unwrap(),
                stdout: BufReader::new(child.stdout.take().unwrap()),
            }),
            child: Arc::new(Mutex::new(child)),
            admission: Arc::new(Semaphore::new(1)),
            healthy: AtomicBool::new(true),
            limits: Limits::default(),
        });
        let response = handle(Request::builder().method("POST").uri("/v1/chat/completions")
            .body(Body::from(json!({"model":"small","messages":[{"role":"user","content":"Hi"}],"max_tokens":1}).to_string())).unwrap(),state.clone()).await.unwrap();
        let payload: serde_json::Value =
            serde_json::from_slice(&to_bytes(response.into_body()).await.unwrap()).unwrap();
        state.invalidate().await;
        assert_eq!(payload["choices"][0]["finish_reason"], "length");
    }
}

#[cfg(test)]
#[path = "openai_tests.rs"]
mod transport_tests;
