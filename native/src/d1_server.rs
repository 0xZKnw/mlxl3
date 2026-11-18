//! Resident entry points for d1 decision models: a JSON-lines bridge for MLXL3 Desktop and
//! a loopback HTTP API (`POST /v1/decide`). One model, one request at a time.
use crate::d1::{Json, runtime::Engine};
use anyhow::{Context, Result, bail};
use hyper::{
    Body, Request, Response, Server, StatusCode,
    body::HttpBody,
    service::{make_service_fn, service_fn},
};
use serde_json::{Value, json};
use std::{
    convert::Infallible,
    io::{BufRead, Write},
    net::{IpAddr, SocketAddr},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
        mpsc,
    },
    time::{Duration, Instant},
};
use tokio::sync::oneshot;

/// Largest request body or bridge line: pictures and clips travel inline as data: URLs.
pub const MAX_REQUEST: usize = 96 * 1024 * 1024;
const QUEUE: usize = 8;

/// One reply line. The answer is written as the ordered `Json` it is: going through
/// `serde_json::Value` would sort its keys and lose the questions' order.
fn answer_line(id: &Value, result: Result<Json>, started: Instant) -> String {
    let elapsed = started.elapsed().as_secs_f64() * 1000.;
    match result.and_then(|answer| Ok(serde_json::to_string(&answer)?)) {
        Ok(answer) => format!(
            r#"{{"type":"result","id":{},"result":{answer},"elapsed_ms":{}}}"#,
            id,
            json!(elapsed)
        ),
        Err(error) => json!({"type": "error", "id": id, "message": format!("{error:#}"), "elapsed_ms": elapsed}).to_string(),
    }
}

/// `{"id": ..., "request": {...}}` per line in, one `result` or `error` line out per request.
/// A line that is not JSON gets an error with a null id; the bridge keeps serving.
pub fn run_bridge(
    path: &Path,
    name: &str,
    input: impl BufRead,
    mut output: impl Write,
) -> Result<()> {
    let mut engine = match Engine::load(path, name) {
        Ok(engine) => engine,
        Err(error) => {
            writeln!(
                output,
                "{}",
                json!({"type": "error", "id": Value::Null, "message": format!("{error:#}")})
            )?;
            output.flush()?;
            return Err(error);
        }
    };
    let mut ready = engine.capabilities();
    ready["type"] = json!("ready");
    writeln!(output, "{ready}")?;
    output.flush()?;
    let mut reader = input;
    let mut line = Vec::new();
    loop {
        line.clear();
        let read = std::io::Read::take(reader.by_ref(), MAX_REQUEST as u64 + 1)
            .read_until(b'\n', &mut line)?;
        if read == 0 {
            return Ok(());
        }
        let started = Instant::now();
        if line.len() > MAX_REQUEST {
            // Drain the rest of the oversized line before answering it.
            while !line.ends_with(b"\n") {
                line.clear();
                if reader.read_until(b'\n', &mut line)? == 0 {
                    break;
                }
            }
            writeln!(
                output,
                "{}",
                json!({"type": "error", "id": Value::Null, "message": "request exceeds 96 MiB"})
            )?;
            output.flush()?;
            continue;
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        let reply = match serde_json::from_slice::<Value>(&line) {
            Err(error) => json!({"type": "error", "id": Value::Null, "message": format!("invalid JSON: {error}")}).to_string(),
            Ok(message) => {
                let id = message.get("id").cloned().unwrap_or(Value::Null);
                if message.get("type").and_then(Value::as_str) == Some("ping") {
                    json!({"type": "pong", "id": id}).to_string()
                } else {
                    let result = message
                        .get("request")
                        .context(r#"a bridge line is {"id": ..., "request": {"state": ..., "questions": {...}}}"#)
                        .and_then(|request| Ok(serde_json::from_value::<Json>(request.clone())?))
                        .and_then(|request| engine.decide(&request, true, None));
                    answer_line(&id, result, started)
                }
            }
        };
        writeln!(output, "{reply}")?;
        output.flush()?;
    }
}

struct Job {
    request: Json,
    reply: oneshot::Sender<Result<Json>>,
}

struct State {
    jobs: mpsc::SyncSender<Job>,
    pending: AtomicUsize,
    api_key: String,
    capabilities: Value,
    name: String,
}

fn json_response(status: StatusCode, value: Value) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .header("access-control-allow-origin", "*")
        .body(Body::from(value.to_string()))
        .expect("valid JSON response")
}

fn error(status: StatusCode, message: impl Into<String>) -> Response<Body> {
    json_response(
        status,
        json!({"error": {"message": message.into(), "type": "invalid_request_error"}}),
    )
}

/// Whether a request carries exactly the bearer key (no key configured: always).
fn authorized(headers: &hyper::HeaderMap, key: &str) -> bool {
    if key.is_empty() {
        return true;
    }
    let mut values = headers.get_all("authorization").iter();
    match (values.next(), values.next()) {
        (Some(value), None) => value.to_str().ok() == Some(format!("Bearer {key}").as_str()),
        _ => false,
    }
}

async fn handle(request: Request<Body>, state: Arc<State>) -> Result<Response<Body>, Infallible> {
    let path = request.uri().path().to_owned();
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
    if request.method() == hyper::Method::GET && path == "/health" {
        return Ok(json_response(
            StatusCode::OK,
            json!({"status": "ok", "pending": state.pending.load(Ordering::Acquire)}),
        ));
    }
    if !authorized(request.headers(), &state.api_key) {
        return Ok(error(StatusCode::UNAUTHORIZED, "Invalid API key"));
    }
    match (request.method().clone(), path.as_str()) {
        (hyper::Method::GET, "/v1/models") => {
            return Ok(json_response(
                StatusCode::OK,
                json!({"object": "list", "data": [{"id": state.name, "object": "model", "owned_by": "mlxl3", "kind": "decision"}]}),
            ));
        }
        (hyper::Method::GET, "/v1/capabilities") => {
            return Ok(json_response(StatusCode::OK, state.capabilities.clone()));
        }
        (hyper::Method::POST, "/v1/decide") => {}
        _ => {
            return Ok(error(
                StatusCode::NOT_FOUND,
                "Unsupported endpoint; use POST /v1/decide",
            ));
        }
    }
    let mut body = request.into_body();
    let mut bytes = Vec::new();
    let read = tokio::time::timeout(Duration::from_secs(60), async {
        while let Some(chunk) = body.data().await {
            let chunk = chunk.map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
            if bytes.len() + chunk.len() > MAX_REQUEST {
                return Err((
                    StatusCode::PAYLOAD_TOO_LARGE,
                    "Request body exceeds 96 MiB".to_owned(),
                ));
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(())
    })
    .await;
    match read {
        Ok(Ok(())) => {}
        Ok(Err((status, message))) => return Ok(error(status, message)),
        Err(_) => return Ok(error(StatusCode::REQUEST_TIMEOUT, "Request body timed out")),
    }
    let request: Json = match serde_json::from_slice(&bytes) {
        Ok(request) => request,
        Err(e) => return Ok(error(StatusCode::BAD_REQUEST, format!("invalid JSON: {e}"))),
    };
    if state.pending.fetch_add(1, Ordering::AcqRel) >= QUEUE {
        state.pending.fetch_sub(1, Ordering::AcqRel);
        let mut response = error(StatusCode::TOO_MANY_REQUESTS, "MLXL3 is busy; retry later");
        response
            .headers_mut()
            .insert("retry-after", hyper::header::HeaderValue::from_static("1"));
        return Ok(response);
    }
    let started = Instant::now();
    let (reply, answer) = oneshot::channel();
    let sent = state.jobs.try_send(Job { request, reply });
    let result = match sent {
        Ok(()) => answer
            .await
            .unwrap_or_else(|_| Err(anyhow::anyhow!("the decision worker stopped"))),
        Err(_) => Err(anyhow::anyhow!("the decision worker is unavailable")),
    };
    state.pending.fetch_sub(1, Ordering::AcqRel);
    Ok(match result {
        Ok(Json::Object(mut entries)) => {
            entries.push(("model".into(), Json::Str(state.name.clone())));
            entries.push((
                "elapsed_ms".into(),
                Json::Float(started.elapsed().as_secs_f64() * 1000.),
            ));
            match serde_json::to_string(&Json::Object(entries)) {
                Ok(body) => Response::builder()
                    .status(StatusCode::OK)
                    .header("content-type", "application/json")
                    .header("access-control-allow-origin", "*")
                    .body(Body::from(body))
                    .expect("valid JSON response"),
                Err(e) => error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
            }
        }
        Ok(_) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "the decision is not an object",
        ),
        Err(e) => error(StatusCode::BAD_REQUEST, format!("{e:#}")),
    })
}

fn listen_addr(host: &str, port: u16, api_key: &str) -> Result<SocketAddr> {
    let ip: IpAddr = host
        .trim_start_matches('[')
        .trim_end_matches(']')
        .parse()
        .context("invalid listen address")?;
    if !ip.is_loopback() && api_key.is_empty() {
        bail!("an API key is required when listening outside loopback");
    }
    Ok(SocketAddr::new(ip, port))
}

/// Loads the model on its own thread, then serves `POST /v1/decide` until SIGINT/SIGTERM.
/// Prints `{"type": "ready", "port": ...}` once listening, like `mlxl3-rs serve`.
pub fn serve(path: PathBuf, name: String, host: &str, port: u16, api_key: String) -> Result<()> {
    let address = listen_addr(host, port, &api_key)?;
    let listener = std::net::TcpListener::bind(address).context("cannot bind the decision API")?;
    listener.set_nonblocking(true)?;
    let (jobs, queue) = mpsc::sync_channel::<Job>(QUEUE);
    let (loaded, ready) = mpsc::channel::<Result<Value>>();
    let worker_name = name.clone();
    std::thread::Builder::new()
        .name("d1-decide".into())
        .spawn(move || {
            let mut engine = match Engine::load(&path, &worker_name) {
                Ok(engine) => {
                    let _ = loaded.send(Ok(engine.capabilities()));
                    engine
                }
                Err(error) => {
                    let _ = loaded.send(Err(error));
                    return;
                }
            };
            while let Ok(job) = queue.recv() {
                let result = engine.decide(&job.request, false, None);
                let _ = job.reply.send(result);
            }
        })?;
    let capabilities = ready
        .recv()
        .context("the decision worker exited while loading")??;
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async move {
        let actual = listener.local_addr()?.port();
        let state = Arc::new(State { jobs, pending: AtomicUsize::new(0), api_key, capabilities, name: name.clone() });
        let service = make_service_fn(move |_| {
            let state = state.clone();
            async move { Ok::<_, Infallible>(service_fn(move |request| handle(request, state.clone()))) }
        });
        let server = Server::from_tcp(listener)?
            .http1_header_read_timeout(Duration::from_secs(10))
            .serve(service);
        println!("{}", json!({"type": "ready", "model": name, "port": actual, "decision": true}));
        std::io::stdout().flush()?;
        let mut terminate = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())?;
        tokio::select! {
            result = server => result.context("decision HTTP server failed"),
            _ = tokio::signal::ctrl_c() => Ok(()),
            _ = terminate.recv() => Ok(()),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bearer_keys_are_exact() {
        let mut headers = hyper::HeaderMap::new();
        assert!(authorized(&headers, ""));
        assert!(!authorized(&headers, "k"));
        headers.insert("authorization", "Bearer k".parse().unwrap());
        assert!(authorized(&headers, "k"));
        assert!(!authorized(&headers, "kk"));
        headers.append("authorization", "Bearer k".parse().unwrap());
        assert!(!authorized(&headers, "k"));
    }

    #[test]
    fn network_listening_needs_a_key() {
        assert!(listen_addr("0.0.0.0", 8000, "").is_err());
        assert!(listen_addr("0.0.0.0", 8000, "k").is_ok());
        assert!(listen_addr("127.0.0.1", 8000, "").is_ok());
        assert!(listen_addr("[::1]", 8000, "").is_ok());
        assert!(listen_addr("nonsense", 8000, "k").is_err());
    }
}
