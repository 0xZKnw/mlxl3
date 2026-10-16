//! CPU-only HTTP/process fixtures. Never loads a model or calls MLX.
use super::*;
use hyper::{Client, Method, body::to_bytes};
use std::process::Stdio;
use tokio::{net::TcpStream, task::JoinHandle};

const FAKE: &str = r#"
import json,os,signal,sys,time
cancelled=False
log=sys.argv[1]
def record(value):
 with open(log,'a') as f: f.write(json.dumps(value)+'\n')
def stop(*args):
 global cancelled
 cancelled=True
signal.signal(signal.SIGUSR1,stop)
for line in sys.stdin:
 request=json.loads(line)
 cancelled=False
 mode=request['messages'][0]['content']
 record({'request':mode})
 rid=request['request_id']
 def emit(kind,**kwargs):
  data=(json.dumps({'type':kind,'request_id':rid,**kwargs})+'\n').encode()
  while data:
   try: written=os.write(sys.stdout.fileno(),data)
   except InterruptedError: continue
   data=data[written:]
 if mode=='eof': sys.exit(0)
 if mode=='invalid': print('not-json',flush=True); continue
 if mode=='oversize': print('x'*5000,flush=True); continue
 if mode=='partial':
  sys.stdout.write(json.dumps({'type':'cancelled','request_id':rid})[:-1]);sys.stdout.flush()
  record({'partial':True})
  while not cancelled: time.sleep(.002)
  record({'signal':'cancel'})
  print('}',flush=True)
  continue
 if mode=='ignore':
  while True: time.sleep(.01)
 if mode=='flood':
  while True: emit('generation_mode')
 if mode=='wait':
  emit('generation_mode')
  while not cancelled: time.sleep(.002)
  record({'signal':'cancel'})
  emit('delta',text='obsolete',phase='answer')
  emit('cancelled')
  continue
 if mode=='silent':
  while not cancelled: time.sleep(.002)
  record({'signal':'cancel'})
  emit('cancelled')
  continue
 if mode=='backpressure':
  for i in range(100):
   if cancelled: break
   emit('delta',text='x'*1000,phase='answer')
  if cancelled: record({'signal':'cancel'});emit('cancelled');continue
 if mode=='large': emit('delta',text='x'*100000,phase='answer')
 if mode=='tcp-large': emit('delta',text='x'*(12*1024*1024),phase='answer')
 if mode=='output-limit': emit('delta',text='x'*3000,phase='answer')
 if mode=='error': emit('error',message='fixture error');continue
 if mode=='big-error': emit('error',message='é'*6000);continue
 if mode.startswith('stats-'):
  stats={'prompt_tokens':2,'generated_tokens':1}
  if mode=='stats-null': stats=None
  if mode=='stats-bool': stats['generated_tokens']=True
  if mode=='stats-prompt-zero': stats['prompt_tokens']=0
  if mode=='stats-prompt-missing': del stats['prompt_tokens']
  if mode=='stats-generated-negative': stats['generated_tokens']=-1
  if mode=='stats-generated-over': stats['generated_tokens']=3
  if mode=='stats-zero': stats['generated_tokens']=0
  emit('complete',stop_reason='eos',stats=stats)
  continue
 emit('delta',text='fresh:'+mode,phase='answer')
 reason=mode if mode in ('eos','max_tokens','context_full','unknown') else 'eos'
 extra={} if mode=='missing' else {'stop_reason':reason}
 emit('complete',context_full=False,stats={'prompt_tokens':2,'generated_tokens':1},**extra)
"#;

struct Fixture {
    state: Arc<State>,
    folder: tempfile::TempDir,
    address: SocketAddr,
    shutdown: Option<oneshot::Sender<()>>,
    server: Option<JoinHandle<()>>,
}

impl Fixture {
    async fn new() -> Self {
        Self::with_limits(Limits {
            body: Duration::from_millis(100),
            write: Duration::from_millis(100),
            event: Duration::from_millis(300),
            generation: Duration::from_secs(2),
            send: Duration::from_millis(100),
            drain: Duration::from_millis(300),
            body_bytes: 4096,
            event_bytes: 4096,
            output_bytes: 2048,
        })
        .await
    }

    async fn with_limits(limits: Limits) -> Self {
        Self::with_key(limits, "secret").await
    }

    async fn with_key(limits: Limits, api_key: &str) -> Self {
        Self::with_program(limits, api_key, FAKE).await
    }

    async fn with_program(limits: Limits, api_key: &str, program: &str) -> Self {
        let folder = tempfile::tempdir().unwrap();
        let log = folder.path().join("events.jsonl");
        let mut child = Command::new("python3")
            .args(["-u", "-c", program, log.to_str().unwrap()])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(
                std::fs::File::create(folder.path().join("stderr.txt")).unwrap(),
            ))
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let state = Arc::new(State {
            model: "small".into(),
            api_key: api_key.into(),
            bridge_info: bridge_capabilities(
                &json!({"runtime_version":"1.4.5","bridge_protocol":1,"model_context_limit":131072,"mtp_supported":true,"dflash_supported":false,"runtime_executable":"/private/secret","api_key":"secret"}),
            ),

            bridge: Mutex::new(Bridge {
                stdin: child.stdin.take().unwrap(),
                stdout: BufReader::new(child.stdout.take().unwrap()),
            }),
            child: Arc::new(Mutex::new(child)),
            admission: Arc::new(Semaphore::new(1)),
            healthy: AtomicBool::new(true),
            limits,
        });
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let service_state = state.clone();
        let service = make_service_fn(move |_| {
            let state = service_state.clone();
            async move { Ok::<_, Infallible>(service_fn(move |request| handle(request, state.clone()))) }
        });
        let (shutdown, receiver) = oneshot::channel();
        let server = Server::builder(transport::Incoming::new(listener, limits.send).unwrap())
            .http1_header_read_timeout(limits.body)
            .serve(service)
            .with_graceful_shutdown(async {
                let _ = receiver.await;
            });
        let server = tokio::spawn(async {
            let _ = server.await;
        });
        Self {
            state,
            folder,
            address,
            shutdown: Some(shutdown),
            server: Some(server),
        }
    }

    fn records(&self) -> String {
        std::fs::read_to_string(self.folder.path().join("events.jsonl")).unwrap_or_default()
    }

    async fn wait_for(&self, needle: &str) {
        timeout(Duration::from_secs(3), async {
            while !self.records().contains(needle) {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("fake bridge did not reach expected state");
    }

    async fn released(&self) {
        timeout(Duration::from_secs(3), async {
            while self.state.admission.available_permits() == 0 {
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        })
        .await
        .expect("admission was not released");
    }

    async fn call(&self, mode: &str, stream: bool) -> Response<Body> {
        handle(completion_request(mode, stream), self.state.clone())
            .await
            .unwrap()
    }

    async fn raw(&self, mode: &str, stream: bool) -> TcpStream {
        let mut socket = TcpStream::connect(self.address).await.unwrap();
        let body = completion_body(mode, stream);
        socket.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
        socket
    }

    async fn finish(mut self) {
        self.state.invalidate().await;
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(mut server) = self.server.take()
            && timeout(Duration::from_secs(2), &mut server).await.is_err()
        {
            server.abort();
            let _ = server.await;
        }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(server) = self.server.take() {
            server.abort();
        }
        // Assertion failures also kill/reap the owned fake child before dropping the runtime.
        if let Ok(mut child) = self.state.child.try_lock() {
            let _ = child.start_kill();
            let deadline = std::time::Instant::now() + Duration::from_secs(1);
            while matches!(child.try_wait(), Ok(None)) && std::time::Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(1));
            }
        }
    }
}

fn completion_body(mode: &str, stream: bool) -> String {
    json!({"model":"small","messages":[{"role":"user","content":mode}],"max_tokens":1,
        "stream":stream,"stream_options":{"include_usage":true}})
    .to_string()
}

fn completion_request(mode: &str, stream: bool) -> Request<Body> {
    Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("authorization", "Bearer secret")
        .body(Body::from(completion_body(mode, stream)))
        .unwrap()
}

async fn assert_fresh(fixture: &Fixture) {
    let response = fixture.call("eos", false).await;
    assert_eq!(
        response.status(),
        StatusCode::OK,
        "records={} stderr={}",
        fixture.records(),
        std::fs::read_to_string(fixture.folder.path().join("stderr.txt")).unwrap_or_default()
    );
    let value: Value =
        serde_json::from_slice(&to_bytes(response.into_body()).await.unwrap()).unwrap();
    assert_eq!(value["choices"][0]["message"]["content"], "fresh:eos");
    assert_eq!(value["choices"][0]["finish_reason"], "stop");
    fixture.released().await;
}

#[tokio::test]
async fn explicit_terminal_reason_controls_streaming_and_nonstreaming() {
    let fixture = Fixture::new().await;
    for (reason, finish) in [
        ("eos", "stop"),
        ("max_tokens", "length"),
        ("context_full", "length"),
    ] {
        for stream in [false, true] {
            let response = fixture.call(reason, stream).await;
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = to_bytes(response.into_body()).await.unwrap();
            let body = String::from_utf8(bytes.to_vec()).unwrap();
            if stream {
                assert!(
                    body.contains(&format!("\"finish_reason\":\"{finish}\"")),
                    "{body}"
                );
                assert!(body.contains("data: [DONE]"));
                assert!(body.contains("\"total_tokens\":3"));
            } else {
                let value: Value = serde_json::from_str(&body).unwrap();
                assert_eq!(value["choices"][0]["finish_reason"], finish);
            }
            fixture.released().await;
        }
    }
    fixture.finish().await;
}

#[tokio::test]
async fn missing_unknown_eof_invalid_or_oversize_bridge_is_not_reused() {
    for mode in ["missing", "unknown", "eof", "invalid", "oversize"] {
        let fixture = Fixture::new().await;
        let response = fixture.call(mode, false).await;
        assert_eq!(response.status(), StatusCode::BAD_GATEWAY, "{mode}");
        assert!(!fixture.state.healthy.load(Ordering::Acquire), "{mode}");
        assert_eq!(
            fixture.call("eos", false).await.status(),
            StatusCode::SERVICE_UNAVAILABLE
        );
        let ready = handle(
            Request::builder()
                .uri("/ready")
                .body(Body::empty())
                .unwrap(),
            fixture.state.clone(),
        )
        .await
        .unwrap();
        assert_eq!(ready.status(), StatusCode::SERVICE_UNAVAILABLE);
        let health = handle(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
            fixture.state.clone(),
        )
        .await
        .unwrap();
        assert_eq!(health.status(), StatusCode::OK);
        fixture.finish().await;
    }
}

#[tokio::test]
async fn admission_rejects_before_reading_slow_body_and_recovers_after_timeout() {
    let fixture = Fixture::new().await;
    let (_sender, body) = Body::channel();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("authorization", "Bearer secret")
        .body(body)
        .unwrap();
    let state = fixture.state.clone();
    let first = tokio::spawn(async move { handle(request, state).await.unwrap() });
    timeout(Duration::from_secs(1), async {
        while fixture.state.admission.available_permits() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    let (_sender2, body2) = Body::channel();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("authorization", "Bearer secret")
        .body(body2)
        .unwrap();
    let rejected = timeout(
        Duration::from_millis(50),
        handle(request, fixture.state.clone()),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(rejected.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(rejected.headers()["retry-after"], "1");
    let busy = handle(
        Request::builder()
            .uri("/ready")
            .body(Body::empty())
            .unwrap(),
        fixture.state.clone(),
    )
    .await
    .unwrap();
    assert_eq!(busy.status(), StatusCode::SERVICE_UNAVAILABLE);
    let response = first.await.unwrap();
    assert_eq!(response.status(), StatusCode::REQUEST_TIMEOUT);
    assert_eq!(response.headers()["connection"], "close");
    assert!(fixture.records().is_empty());
    assert_fresh(&fixture).await;
    fixture.finish().await;
}

#[tokio::test]
async fn oversized_and_failed_http_bodies_never_reach_bridge() {
    let fixture = Fixture::new().await;
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("authorization", "Bearer secret")
        .body(Body::from(vec![b'x'; 4097]))
        .unwrap();
    assert_eq!(
        handle(request, fixture.state.clone())
            .await
            .unwrap()
            .status(),
        StatusCode::PAYLOAD_TOO_LARGE
    );
    let (sender, body) = Body::channel();
    sender.abort();
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/chat/completions")
        .header("authorization", "Bearer secret")
        .body(body)
        .unwrap();
    assert_eq!(
        handle(request, fixture.state.clone())
            .await
            .unwrap()
            .status(),
        StatusCode::BAD_REQUEST
    );
    assert!(fixture.records().is_empty());
    assert_fresh(&fixture).await;
    fixture.finish().await;
}

#[tokio::test]
async fn tcp_disconnect_before_first_delta_drains_then_next_request_is_clean() {
    for stream in [false, true] {
        for mode in ["wait", "partial"] {
            let limits = Limits {
                event: Duration::from_secs(10),
                generation: Duration::from_secs(10),
                drain: Duration::from_millis(200),
                ..Limits::default()
            };
            let fixture = Fixture::with_limits(limits).await;
            let socket = fixture.raw(mode, stream).await;
            fixture
                .wait_for(if mode == "partial" {
                    "\"partial\": true"
                } else {
                    "\"request\": \"wait\""
                })
                .await;
            drop(socket);
            timeout(
                Duration::from_millis(500),
                fixture.wait_for("\"signal\": \"cancel\""),
            )
            .await
            .expect("disconnect was not detected before the ten-second event deadline");
            fixture.released().await;
            assert!(
                fixture.state.healthy.load(Ordering::Acquire),
                "{mode}/{stream}"
            );
            assert_fresh(&fixture).await;
            fixture.finish().await;
        }
    }
}

#[tokio::test]
async fn disconnect_or_timeout_without_terminal_marks_bridge_unhealthy() {
    let fixture = Fixture::new().await;
    let socket = fixture.raw("ignore", false).await;
    fixture.wait_for("\"request\": \"ignore\"").await;
    drop(socket);
    fixture.released().await;
    assert!(!fixture.state.healthy.load(Ordering::Acquire));
    assert_eq!(
        fixture.call("eos", false).await.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    fixture.finish().await;
}

#[tokio::test]
async fn silent_bridge_timeout_cancels_and_drains_a_terminal() {
    let fixture = Fixture::new().await;
    assert_eq!(
        fixture.call("silent", false).await.status(),
        StatusCode::BAD_GATEWAY
    );
    fixture.wait_for("\"signal\": \"cancel\"").await;
    fixture.released().await;
    assert_fresh(&fixture).await;
    fixture.finish().await;
}

#[tokio::test]
async fn backpressure_and_output_limit_are_bounded_and_do_not_signal_next_request() {
    let limits = Limits {
        output_bytes: 1024 * 1024,
        send: Duration::from_millis(100),
        drain: Duration::from_millis(500),
        ..Limits::default()
    };
    let fixture = Fixture::with_limits(limits).await;
    let response = fixture.call("backpressure", true).await;
    let mut body = response.into_body();
    assert!(body.data().await.unwrap().is_ok()); // Consume role, then stop polling.
    fixture.wait_for("\"request\": \"backpressure\"").await;
    timeout(Duration::from_secs(1), fixture.released())
        .await
        .expect("HTTP backpressure did not release admission within its deadline");
    drop(body);
    assert_fresh(&fixture).await;
    fixture.finish().await;
    let fixture = Fixture::new().await;
    let response = fixture.call("output-limit", false).await;
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    fixture.released().await;
    assert_fresh(&fixture).await;
    fixture.finish().await;
}

#[tokio::test]
async fn health_checks_reap_an_idle_dead_child() {
    let fixture = Fixture::new().await;
    fixture.state.child.lock().await.kill().await.unwrap();
    let ready = handle(
        Request::builder()
            .uri("/ready")
            .body(Body::empty())
            .unwrap(),
        fixture.state.clone(),
    )
    .await
    .unwrap();
    assert_eq!(ready.status(), StatusCode::SERVICE_UNAVAILABLE);
    fixture.finish().await;
}

#[tokio::test]
async fn origins_and_authentication_agree_for_preflight_errors_and_real_http() {
    let fixture = Fixture::new().await;
    let client = Client::new();
    for origin in [
        None,
        Some("http://localhost:3000"),
        Some("https://127.0.0.1"),
        Some("http://[::1]:8080"),
    ] {
        let mut request = Request::builder()
            .uri(format!("http://{}/v1/models", fixture.address))
            .header("authorization", "Bearer secret");
        if let Some(origin) = origin {
            request = request.header("origin", origin);
        }
        let response = client
            .request(request.body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("access-control-allow-origin")
                .map(|v| v.to_str().unwrap()),
            origin
        );
        assert_eq!(response.headers()["vary"], "Origin");
        to_bytes(response.into_body()).await.unwrap();
    }
    for origin in [
        "null",
        "https://evil.example",
        "http://localhost.evil",
        "http://localhost:bad",
        "http://localhost/path",
        "http://user@localhost",
        "http://127.1",
        "file://localhost",
    ] {
        for method in [Method::GET, Method::OPTIONS] {
            let response = handle(
                Request::builder()
                    .method(method)
                    .uri("/v1/models")
                    .header("origin", origin)
                    .header("authorization", "Bearer secret")
                    .body(Body::empty())
                    .unwrap(),
                fixture.state.clone(),
            )
            .await
            .unwrap();
            assert_eq!(response.status(), StatusCode::FORBIDDEN, "{origin}");
            assert!(
                response
                    .headers()
                    .get("access-control-allow-origin")
                    .is_none()
            );
        }
    }
    for auth in [None, Some("secret"), Some("Bearer wrong")] {
        let mut request = Request::builder()
            .uri("/v1/models")
            .header("origin", "http://localhost");
        if let Some(auth) = auth {
            request = request.header("authorization", auth);
        }
        let response = handle(request.body(Body::empty()).unwrap(), fixture.state.clone())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(
            response.headers()["access-control-allow-origin"],
            "http://localhost"
        );
    }
    let response = handle(
        Request::builder()
            .method(Method::OPTIONS)
            .uri("/v1/chat/completions")
            .header("origin", "http://localhost")
            .header("access-control-request-method", "POST")
            .header(
                "access-control-request-headers",
                "authorization, content-type",
            )
            .body(Body::empty())
            .unwrap(),
        fixture.state.clone(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::NO_CONTENT);
    assert_eq!(
        response.headers()["access-control-allow-origin"],
        "http://localhost"
    );
    drop(client);
    fixture.finish().await;
}

#[test]
fn loopback_origin_parser_rejects_authority_confusion() {
    for origin in [
        "http://localhost",
        "https://localhost:65535",
        "http://127.0.0.1:0",
        "http://[::1]",
    ] {
        assert!(origin_allowed(origin));
    }
    for origin in [
        "http://localhost:",
        "http://localhost:65536",
        "http://localhost:1/",
        "http://localhost?x",
        "http://localhost#x",
        "http://[::1]evil",
        "http://localhost:1@evil",
    ] {
        assert!(!origin_allowed(origin));
    }
}

#[tokio::test]
async fn capabilities_describe_validator_and_hide_private_ready_fields() {
    let fixture = Fixture::new().await;
    let unauthorized = handle(
        Request::builder()
            .uri("/v1/capabilities")
            .body(Body::empty())
            .unwrap(),
        fixture.state.clone(),
    )
    .await
    .unwrap();
    assert_eq!(unauthorized.status(), StatusCode::UNAUTHORIZED);
    let response = handle(
        Request::builder()
            .uri("/v1/capabilities")
            .header("authorization", "Bearer secret")
            .body(Body::empty())
            .unwrap(),
        fixture.state.clone(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let caps: Value =
        serde_json::from_slice(&to_bytes(response.into_body()).await.unwrap()).unwrap();
    assert_eq!(caps["bridge"]["mtp_supported"], true);
    assert_eq!(caps["api"]["mtp"], false);
    assert_eq!(caps["bridge"]["runtime_version"], "1.4.5");
    assert!(caps["bridge"].get("runtime_executable").is_none());
    assert!(caps["bridge"].get("api_key").is_none());
    for (flag, field, value) in [
        ("stop_sequences", "stop", json!(["halt"])),
        ("tool_calls", "tools", json!([{"type":"function"}])),
        (
            "json_schema",
            "response_format",
            json!({"type":"json_schema"}),
        ),
        ("seed", "seed", json!(1)),
        ("logit_bias", "logit_bias", json!({"1":1})),
        ("multiple_choices", "n", json!(2)),
        ("mtp", "mtp", json!(true)),
        ("dflash", "dflash2", json!(true)),
    ] {
        assert_eq!(caps["api"][flag], false);
        let mut body: Value = serde_json::from_str(&completion_body("eos", false)).unwrap();
        body[field] = value;
        assert!(bridge_request(&body, "small").is_err(), "{flag}");
    }
    for (field, value) in [
        ("top_p", json!(0.5)),
        ("frequency_penalty", json!(1)),
        ("presence_penalty", json!(1)),
    ] {
        assert_eq!(caps["api"]["sampling"][field], false);
        let mut body: Value = serde_json::from_str(&completion_body("eos", false)).unwrap();
        body[field] = value;
        assert!(bridge_request(&body, "small").is_err());
    }
    let mut body: Value = serde_json::from_str(&completion_body("eos", true)).unwrap();
    body["temperature"] = json!(0.7);
    body["top_k"] = json!(37);
    body["repetition_penalty"] = json!(1.12);
    assert!(bridge_request(&body, "small").is_ok());
    for field in ["temperature", "top_k", "repetition_penalty"] {
        assert_eq!(caps["api"]["sampling"][field], true);
    }
    for field in ["chat_completions", "stream", "stream_usage", "text_only"] {
        assert_eq!(caps["api"][field], true);
    }
    body["messages"] =
        json!([{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:"}}]}]);
    assert_eq!(caps["api"]["images"], false);
    assert!(bridge_request(&body, "small").is_err());
    fixture.finish().await;
}

#[test]
fn bridge_metadata_whitelist_rejects_bad_types_and_private_strings() {
    let info = bridge_capabilities(
        &json!({"runtime_version":"/private/secret","mlx_version":"x".repeat(65),
        "mtp_supported":"true","dflash_supported":false,"memory_saver_supported":true,
        "bridge_protocol":-1,"model_context_limit":2147483648u64,"runtime_executable":"hidden"}),
    );
    assert_eq!(
        info,
        json!({"dflash_supported":false,"memory_saver_supported":true})
    );
}

#[tokio::test]
async fn host_validation_is_required_only_for_keyless_sessions() {
    let fixture = Fixture::with_key(Limits::default(), "").await;
    for host in [
        None,
        Some("localhost:8080"),
        Some("LOCALHOST"),
        Some("127.42.8.3:1"),
        Some("[::1]:65535"),
    ] {
        let mut request = Request::builder().uri("/v1/models");
        if let Some(host) = host {
            request = request.header("host", host);
        }
        assert_eq!(
            handle(request.body(Body::empty()).unwrap(), fixture.state.clone())
                .await
                .unwrap()
                .status(),
            StatusCode::OK
        );
    }
    for host in [
        "evil.example",
        "localhost:bad",
        "127.0.0.1:65536",
        "user@localhost",
        "localhost/path",
        "localhost?x",
        "localhost#x",
        "::1",
        "[localhost]",
        "[127.0.0.1]",
    ] {
        let request = Request::builder()
            .uri("/health")
            .header("host", host)
            .body(Body::empty())
            .unwrap();
        assert_eq!(
            handle(request, fixture.state.clone())
                .await
                .unwrap()
                .status(),
            StatusCode::FORBIDDEN,
            "{host}"
        );
    }
    let duplicate = Request::builder()
        .uri("/health")
        .header("host", "localhost")
        .header("host", "evil.example")
        .body(Body::empty())
        .unwrap();
    assert_eq!(
        handle(duplicate, fixture.state.clone())
            .await
            .unwrap()
            .status(),
        StatusCode::FORBIDDEN
    );
    fixture.finish().await;
    let fixture = Fixture::new().await;
    let response = handle(
        Request::builder()
            .uri("/v1/models")
            .header("host", "custom.lan:1234")
            .header("authorization", "Bearer secret")
            .body(Body::empty())
            .unwrap(),
        fixture.state.clone(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let response = handle(
        Request::builder()
            .uri("/v1/models")
            .header("host", "custom.lan:1234")
            .body(Body::empty())
            .unwrap(),
        fixture.state.clone(),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    fixture.finish().await;
}

#[test]
fn penalties_representable_only_in_f64_are_rejected_before_prefill() {
    for penalty in [1e300, 1e-300] {
        let mut input: Value = serde_json::from_str(&completion_body("eos", false)).unwrap();
        input["repetition_penalty"] = json!(penalty);
        assert!(bridge_request(&input, "small").is_err(), "{penalty}");
    }
}

#[tokio::test]
async fn a_partial_bridge_write_is_bounded_and_poisons_the_session() {
    let limits = Limits {
        write: Duration::from_millis(30),
        drain: Duration::from_millis(200),
        ..Limits::default()
    };
    let fixture = Fixture::with_program(limits, "secret", "import time; time.sleep(30)").await;
    let response = timeout(
        Duration::from_secs(1),
        fixture.call(&"x".repeat(512 * 1024), false),
    )
    .await
    .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(!fixture.state.healthy.load(Ordering::Acquire));
    assert_eq!(
        fixture.call("eos", false).await.status(),
        StatusCode::SERVICE_UNAVAILABLE
    );
    fixture.finish().await;
}

#[tokio::test]
async fn nonstreaming_backpressure_keeps_admission_bounded_until_abort() {
    let limits = Limits {
        send: Duration::from_millis(100),
        ..Limits::default()
    };
    let fixture = Fixture::with_limits(limits).await;
    let response = fixture.call("large", false).await;
    assert_eq!(response.status(), StatusCode::OK);
    let mut body = response.into_body();
    let first = body.data().await.unwrap().unwrap();
    assert!(first.len() <= 16 * 1024);
    assert_eq!(fixture.state.admission.available_permits(), 0);
    fixture.released().await;
    assert!(fixture.state.healthy.load(Ordering::Acquire));
    drop(body);
    assert_fresh(&fixture).await;
    fixture.finish().await;
}

#[tokio::test]
async fn slow_headers_close_without_claiming_generation_admission() {
    use tokio::io::AsyncReadExt;
    let fixture = Fixture::new().await;
    let mut socket = TcpStream::connect(fixture.address).await.unwrap();
    socket
        .write_all(b"POST /v1/chat/completions HTTP/1.1\r\nHost:")
        .await
        .unwrap();
    let mut response = Vec::new();
    timeout(Duration::from_secs(1), socket.read_to_end(&mut response))
        .await
        .unwrap()
        .unwrap();
    assert!(response.is_empty() || String::from_utf8_lossy(&response).contains("408"));
    assert_eq!(fixture.state.admission.available_permits(), 1);
    assert!(fixture.records().is_empty());
    fixture.finish().await;
}

#[tokio::test]
async fn continuous_stale_events_cannot_extend_generation_deadline() {
    let limits = Limits {
        generation: Duration::from_millis(50),
        drain: Duration::from_millis(100),
        ..Limits::default()
    };
    let fixture = Fixture::with_limits(limits).await;
    let response = timeout(Duration::from_secs(1), fixture.call("flood", false))
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    assert!(!fixture.state.healthy.load(Ordering::Acquire));
    fixture.finish().await;
}

#[tokio::test]
async fn malformed_usage_is_a_protocol_error_without_false_completion() {
    for mode in [
        "stats-null",
        "stats-bool",
        "stats-prompt-zero",
        "stats-prompt-missing",
        "stats-generated-negative",
        "stats-generated-over",
    ] {
        for stream in [false, true] {
            let fixture = Fixture::new().await;
            let response = fixture.call(mode, stream).await;
            assert_eq!(
                response.status(),
                if stream {
                    StatusCode::OK
                } else {
                    StatusCode::BAD_GATEWAY
                },
                "{mode}"
            );
            let text =
                String::from_utf8(to_bytes(response.into_body()).await.unwrap().to_vec()).unwrap();
            assert!(text.contains("error"), "{mode}: {text}");
            assert!(
                !text.contains("\"finish_reason\":\"stop\""),
                "{mode}: {text}"
            );
            assert!(
                !text.contains("\"finish_reason\":\"length\""),
                "{mode}: {text}"
            );
            assert!(!fixture.state.healthy.load(Ordering::Acquire), "{mode}");
            let ready = handle(
                Request::builder()
                    .uri("/ready")
                    .body(Body::empty())
                    .unwrap(),
                fixture.state.clone(),
            )
            .await
            .unwrap();
            assert_eq!(ready.status(), StatusCode::SERVICE_UNAVAILABLE);
            fixture.finish().await;
        }
    }
    let fixture = Fixture::new().await;
    let response = fixture.call("stats-zero", false).await;
    assert_eq!(response.status(), StatusCode::OK);
    let value: Value =
        serde_json::from_slice(&to_bytes(response.into_body()).await.unwrap()).unwrap();
    assert_eq!(value["usage"]["completion_tokens"], 0);
    assert_eq!(value["choices"][0]["finish_reason"], "stop");
    fixture.finish().await;
}

#[tokio::test]
async fn bridge_errors_have_a_bounded_valid_utf8_http_body() {
    for stream in [false, true] {
        let fixture = Fixture::with_limits(Limits {
            event_bytes: 65536,
            ..Limits::default()
        })
        .await;
        let response = fixture.call("big-error", stream).await;
        assert_eq!(
            response.status(),
            if stream {
                StatusCode::OK
            } else {
                StatusCode::BAD_GATEWAY
            }
        );
        let text =
            String::from_utf8(to_bytes(response.into_body()).await.unwrap().to_vec()).unwrap();
        assert!(text.len() < 5000, "error response was {} bytes", text.len());
        assert!(text.contains("é"));
        assert!(!text.contains("finish_reason\":\"stop"));
        fixture.released().await;
        assert_fresh(&fixture).await;
        fixture.finish().await;
    }
}

#[test]
fn listen_addresses_accept_loopback_ipv4_and_ipv6_without_dns() {
    for host in ["127.0.0.1", "127.99.1.2", "::1", "[::1]"] {
        let address = listen_addr(host, 0, "").unwrap();
        assert!(address.ip().is_loopback());
        assert_eq!(address.port(), 0);
        assert_eq!(listen_addr(host, 65535, "").unwrap().port(), 65535);
    }
    for host in ["::", "[::]", "192.0.2.1"] {
        assert!(listen_addr(host, 8080, "").is_err());
        assert!(listen_addr(host, 8080, "secret").is_ok());
    }
    for host in [
        "localhost",
        "::1]",
        "[::1",
        "[127.0.0.1]",
        "[::1]:80",
        "::1/path",
        "127.0.0.1:80",
        "",
    ] {
        assert!(listen_addr(host, 8080, "secret").is_err(), "{host}");
    }
}

async fn startup_child(program: &str) -> Arc<Mutex<Child>> {
    Arc::new(Mutex::new(
        Command::new("python3")
            .args(["-u", "-c", program])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .unwrap(),
    ))
}

fn startup_listener() -> TcpListener {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    listener
}

#[tokio::test]
async fn startup_timeout_invalid_ready_and_missing_pipes_reap_the_child() {
    for (program, message, wait) in [
        ("import time; time.sleep(30)", "model load timed out", 30),
        (
            "import time; print('bad-json',flush=True); time.sleep(30)",
            "Invalid MLXL3 bridge event",
            1000,
        ),
        (
            "import time; print('{\"type\":\"error\",\"message\":\"fixture load failure\"}',flush=True); time.sleep(30)",
            "fixture load failure",
            1000,
        ),
    ] {
        let child = startup_child(program).await;
        let result = timeout(
            Duration::from_secs(2),
            serve_resident(
                child.clone(),
                startup_listener(),
                "small",
                "",
                Limits {
                    drain: Duration::from_millis(200),
                    ..Limits::default()
                },
                Duration::from_millis(wait),
                std::future::pending(),
            ),
        )
        .await
        .unwrap();
        assert!(format!("{:#}", result.unwrap_err()).contains(message));
        assert!(child.lock().await.try_wait().unwrap().is_some());
    }
    let child = startup_child("import time; time.sleep(30)").await;
    drop(child.lock().await.stdin.take());
    let result = serve_resident(
        child.clone(),
        startup_listener(),
        "small",
        "",
        Limits::default(),
        Duration::from_secs(1),
        std::future::pending(),
    )
    .await;
    assert!(
        result
            .unwrap_err()
            .to_string()
            .contains("stdin unavailable")
    );
    assert!(child.lock().await.try_wait().unwrap().is_some());
}

#[tokio::test]
async fn shutdown_after_ready_reaps_child_and_joins_stderr() {
    let child=startup_child("import sys,time; print('{\"type\":\"ready\"}',flush=True);sys.stderr.write('fixture stderr\\n');sys.stderr.flush();time.sleep(30)").await;
    let listener = startup_listener();
    let address = listener.local_addr().unwrap();
    let (stop, stopped) = oneshot::channel();
    let process = child.clone();
    let server = tokio::spawn(async move {
        serve_resident(
            process,
            listener,
            "small",
            "",
            Limits::default(),
            Duration::from_secs(1),
            async {
                let _ = stopped.await;
            },
        )
        .await
    });
    timeout(Duration::from_secs(2), async {
        loop {
            if let Ok(response) = Client::new()
                .get(format!("http://{address}/ready").parse().unwrap())
                .await
            {
                assert_eq!(response.status(), StatusCode::OK);
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
    stop.send(()).unwrap();
    timeout(Duration::from_secs(2), server)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert!(child.lock().await.try_wait().unwrap().is_some());
}

#[cfg(unix)]
#[tokio::test]
async fn sigterm_during_startup_kills_and_reaps_bridge() {
    const SWITCH: &str = "MLXL3_API_SIGNAL_FIXTURE";
    if let Some(folder) = std::env::var_os(SWITCH) {
        let folder = std::path::PathBuf::from(folder);
        let shutdown = shutdown_signal().unwrap();
        let pid_path = folder.join("bridge-pid");
        let program = format!(
            "import os,time;open({:?},'w').write(str(os.getpid()));time.sleep(30)",
            pid_path
        );
        let child = startup_child(&program).await;
        serve_resident(
            child.clone(),
            startup_listener(),
            "small",
            "",
            Limits {
                drain: Duration::from_millis(300),
                ..Limits::default()
            },
            Duration::from_secs(10),
            shutdown,
        )
        .await
        .unwrap();
        assert!(child.lock().await.try_wait().unwrap().is_some());
        std::fs::write(folder.join("reaped"), b"yes").unwrap();
        return;
    }
    let folder = tempfile::tempdir().unwrap();
    let mut fixture = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "openai::transport_tests::sigterm_during_startup_kills_and_reaps_bridge",
            "--nocapture",
        ])
        .env(SWITCH, folder.path())
        .stdout(Stdio::null())
        .stderr(Stdio::from(
            std::fs::File::create(folder.path().join("stderr.log")).unwrap(),
        ))
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    timeout(Duration::from_secs(3), async {
        while !folder.path().join("bridge-pid").exists() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
    assert_eq!(
        unsafe { libc::kill(fixture.id().unwrap() as i32, libc::SIGTERM) },
        0
    );
    let status = timeout(Duration::from_secs(2), fixture.wait())
        .await
        .unwrap()
        .unwrap();
    assert!(
        status.success(),
        "{}",
        std::fs::read_to_string(folder.path().join("stderr.log")).unwrap()
    );
    assert_eq!(std::fs::read(folder.path().join("reaped")).unwrap(), b"yes");
}

#[cfg(unix)]
#[tokio::test]
async fn a_real_nonreading_tcp_client_is_closed_before_it_resumes_reading() {
    use std::os::fd::AsRawFd;
    let fixture = Fixture::with_limits(Limits {
        send: Duration::from_secs(2),
        ..Limits::default()
    })
    .await;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let mut socket = TcpStream::connect(address).await.unwrap();
    let receive: i32 = 4096;
    assert_eq!(
        unsafe {
            libc::setsockopt(
                socket.as_raw_fd(),
                libc::SOL_SOCKET,
                libc::SO_RCVBUF,
                (&receive as *const i32).cast(),
                std::mem::size_of_val(&receive) as libc::socklen_t,
            )
        },
        0
    );
    let state = fixture.state.clone();
    let mut server = tokio::spawn(async move {
        let (stream, _) = listener.accept().await.unwrap();
        let io = transport::DeadlineIo::new(stream, Duration::from_millis(150));
        hyper::server::conn::Http::new()
            .serve_connection(
                io,
                service_fn(move |request| handle(request, state.clone())),
            )
            .await
    });
    let body = completion_body("tcp-large", false);
    socket.write_all(format!("POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer secret\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",body.len()).as_bytes()).await.unwrap();
    // No HTTP byte is read: completion of the connection proves transport closure,
    // unlike a Body::Sender abort that Hyper may only see after flushing its buffer.
    let result = timeout(Duration::from_secs(3), &mut server).await;
    if result.is_err() {
        server.abort();
        let _ = server.await;
        panic!("stalled TCP connection survived its deadline");
    }
    let error = result.unwrap().unwrap().unwrap_err();
    let mut source: Option<&(dyn std::error::Error + 'static)> = Some(&error);
    let mut timed_out = false;
    while let Some(error) = source {
        timed_out |= error
            .downcast_ref::<std::io::Error>()
            .is_some_and(|e| e.kind() == std::io::ErrorKind::TimedOut);
        source = error.source();
    }
    assert!(timed_out, "{error:?}");
    drop(socket);
    fixture.released().await;
    assert_fresh(&fixture).await;
    fixture.finish().await;
}
