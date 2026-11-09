//! Minimal MCP client shared by the native CLI and desktop bridge.
use anyhow::{Context, Result, bail, ensure};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::fd::AsRawFd,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio},
    sync::atomic::{AtomicBool, Ordering},
    thread,
    time::{Duration, Instant},
};

const PROTOCOL_VERSION: &str = "2025-11-25";
const MAX_RESPONSE: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ServerConfig {
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default = "enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default)]
    pub headers: BTreeMap<String, String>,
}

fn enabled() -> bool {
    true
}

#[derive(Deserialize, Serialize)]
struct ConfigFile {
    version: u32,
    #[serde(rename = "mcpServers")]
    servers: BTreeMap<String, ServerConfig>,
}

#[derive(Clone, Debug)]
pub struct Tool {
    pub server: String,
    pub name: String,
    pub public_name: String,
    pub description: String,
    pub input_schema: Value,
}

#[derive(Debug)]
pub struct ToolResult {
    pub text: String,
    pub is_error: bool,
}

fn builtin_exa() -> ServerConfig {
    ServerConfig {
        command: String::new(),
        args: Vec::new(),
        env: BTreeMap::new(),
        cwd: None,
        enabled: true,
        url: Some("https://mcp.exa.ai/mcp".into()),
        headers: BTreeMap::new(),
    }
}

pub fn config_path(registry: &Path) -> PathBuf {
    registry.with_file_name("mcp.json")
}

fn write_config(path: &Path, payload: &ConfigFile) -> Result<()> {
    let parent = path.parent().unwrap_or(Path::new("."));
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temporary, payload)?;
    temporary.write_all(b"\n")?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .context("saving MCP configuration")?;
    File::open(parent)?.sync_all()?;
    Ok(())
}

fn read_config_unlocked(path: &Path) -> Result<ConfigFile> {
    if !path.exists() {
        let payload = ConfigFile {
            version: 1,
            servers: BTreeMap::from([("exa".into(), builtin_exa())]),
        };
        write_config(path, &payload)?;
        return Ok(payload);
    }
    let mut payload: ConfigFile =
        serde_json::from_reader(File::open(path)?).context("invalid MCP configuration")?;
    ensure!(
        payload.version == 1,
        "unsupported MCP configuration version"
    );
    payload
        .servers
        .entry("exa".into())
        .or_insert_with(builtin_exa);
    validate_configs(&payload.servers)?;
    Ok(payload)
}

fn validate_configs(configs: &BTreeMap<String, ServerConfig>) -> Result<()> {
    for (name, config) in configs {
        ensure!(
            valid_server_name(name.as_bytes()),
            "invalid MCP server name {name:?}"
        );
        match (&config.url, config.command.is_empty()) {
            (Some(url), true) => validate_url(url)?,
            (None, false) => {}
            _ => bail!("MCP server {name:?} requires either a URL or a command"),
        }
        ensure!(
            config
                .headers
                .iter()
                .all(|(key, value)| !key.contains(['\r', '\n', ':'])
                    && !value.contains(['\r', '\n'])),
            "MCP server {name:?} has invalid headers"
        );
    }
    Ok(())
}

fn valid_server_name(name: &[u8]) -> bool {
    !name.is_empty()
        && name
            .iter()
            .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(c))
}

fn validate_url(url: &str) -> Result<()> {
    let uri: hyper::Uri = url.parse().context("invalid MCP URL")?;
    let host = uri.host().context("MCP URL has no host")?;
    ensure!(
        uri.authority()
            .is_some_and(|authority| !authority.as_str().contains('@')),
        "MCP URLs must not contain credentials"
    );
    ensure!(
        uri.scheme_str() == Some("https")
            || (uri.scheme_str() == Some("http")
                && matches!(host, "localhost" | "127.0.0.1" | "[::1]" | "::1")),
        "MCP endpoints require HTTPS (HTTP is allowed for loopback)"
    );
    Ok(())
}

// Own the lock over first creation and the entire read-modify-persist transaction.
fn update_config<T>(
    registry: &Path,
    change: impl FnOnce(&mut ConfigFile) -> Result<T>,
) -> Result<T> {
    let path = config_path(registry);
    fs::create_dir_all(path.parent().unwrap_or(Path::new(".")))?;
    let lock = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path.with_extension("lock"))?;
    lock.lock_exclusive()?;
    let mut payload = read_config_unlocked(&path)?;
    let result = change(&mut payload)?;
    validate_configs(&payload.servers)?;
    write_config(&path, &payload)?;
    Ok(result)
}

pub fn list_servers(registry: &Path) -> Result<BTreeMap<String, ServerConfig>> {
    update_config(registry, |payload| Ok(payload.servers.clone()))
}

pub fn add_server(registry: &Path, name: &str, command: &str, args: Vec<String>) -> Result<()> {
    let config = ServerConfig {
        command: command.into(),
        args,
        env: BTreeMap::new(),
        cwd: None,
        enabled: true,
        url: None,
        headers: BTreeMap::new(),
    };
    update_config(registry, |payload| {
        payload.servers.insert(name.into(), config);
        Ok(())
    })
}

pub fn remove_server(registry: &Path, name: &str) -> Result<()> {
    update_config(registry, |payload| {
        ensure!(
            payload.servers.contains_key(name),
            "unknown MCP server {name:?}"
        );
        if name == "exa" {
            payload.servers.insert(
                name.into(),
                ServerConfig {
                    enabled: false,
                    ..builtin_exa()
                },
            );
        } else {
            payload.servers.remove(name);
        }
        Ok(())
    })
}

enum Client {
    Http(HttpClient),
    Stdio(StdioClient),
}

impl Client {
    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        match self {
            Self::Http(client) => client.request(method, params),
            Self::Stdio(client) => client.request(method, params),
        }
    }

    fn request_cancellable(
        &mut self,
        method: &str,
        params: Value,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        match self {
            Self::Http(client) => client.request_cancellable(method, params, cancelled),
            Self::Stdio(client) => client.request_cancellable(method, params, cancelled),
        }
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        match self {
            Self::Http(client) => client.notify(method, params),
            Self::Stdio(client) => client.notify(method, params),
        }
    }
}

const CALL_TIMEOUT: Duration = Duration::from_secs(12);
const POLL_INTERVAL: Duration = Duration::from_millis(5);
const MAX_HEADERS: usize = 64 * 1024;

fn nonblocking(fd: &impl AsRawFd) -> Result<()> {
    // Pipes are exclusively owned by this transport; no shared descriptor flags.
    let flags = unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_GETFL) };
    ensure!(flags >= 0, "reading pipe flags failed");
    ensure!(
        unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFL, flags | libc::O_NONBLOCK) } >= 0,
        "setting nonblocking pipe failed"
    );
    Ok(())
}

fn check_deadline(deadline: Instant, cancelled: &AtomicBool) -> Result<()> {
    ensure!(!cancelled.load(Ordering::Relaxed), "MCP request cancelled");
    ensure!(Instant::now() < deadline, "MCP request timed out");
    Ok(())
}

// No reader/writer threads or unbounded queues. Each iteration does bounded work,
// including draining stderr, so noisy peers cannot postpone deadline checks.
struct StdioClient {
    child: Child,
    stdin: Option<ChildStdin>,
    stdout: ChildStdout,
    stderr: ChildStderr,
    pending: Vec<u8>,
    next_id: u64,
}

impl StdioClient {
    fn from_command(command: &mut Command) -> Result<Self> {
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0);
        let mut child = command.spawn().context("starting MCP transport")?;
        let stdin = child.stdin.take().context("MCP process has no stdin")?;
        let stdout = child.stdout.take().context("MCP process has no stdout")?;
        let stderr = child.stderr.take().context("MCP process has no stderr")?;
        let client = Self {
            child,
            stdin: Some(stdin),
            stdout,
            stderr,
            pending: Vec::new(),
            next_id: 1,
        };
        nonblocking(client.stdin.as_ref().unwrap())?;
        nonblocking(&client.stdout)?;
        nonblocking(&client.stderr)?;
        Ok(client)
    }

    fn connect(config: &ServerConfig) -> Result<Self> {
        let mut command = Command::new(&config.command);
        command.args(&config.args).envs(&config.env);
        if let Some(cwd) = &config.cwd {
            command.current_dir(cwd);
        }
        Self::from_command(&mut command)
    }

    fn drain_stderr(&mut self) {
        let mut bytes = [0; 4096];
        let _ = self.stderr.read(&mut bytes);
    }

    fn send(&mut self, payload: &Value, deadline: Instant, cancelled: &AtomicBool) -> Result<()> {
        let mut bytes = serde_json::to_vec(payload)?;
        ensure!(bytes.len() <= MAX_RESPONSE, "MCP request exceeds 8 MiB");
        bytes.push(b'\n');
        let mut offset = 0;
        while offset < bytes.len() {
            check_deadline(deadline, cancelled)?;
            self.drain_stderr();
            match self
                .stdin
                .as_mut()
                .context("MCP stdin closed")?
                .write(&bytes[offset..])
            {
                Ok(0) => bail!("MCP stdin closed"),
                Ok(n) => offset += n,
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(POLL_INTERVAL)
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error.into()),
            }
        }
        Ok(())
    }

    fn read_chunk(&mut self, deadline: Instant, cancelled: &AtomicBool) -> Result<Option<Vec<u8>>> {
        check_deadline(deadline, cancelled)?;
        self.drain_stderr();
        let mut bytes = [0; 4096];
        match self.stdout.read(&mut bytes) {
            Ok(0) => Ok(None),
            Ok(n) => Ok(Some(bytes[..n].to_vec())),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(POLL_INTERVAL);
                Ok(Some(Vec::new()))
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => Ok(Some(Vec::new())),
            Err(error) => Err(error.into()),
        }
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.send(
            &json!({"jsonrpc":"2.0", "method":method, "params":params}),
            Instant::now() + CALL_TIMEOUT,
            &AtomicBool::new(false),
        )
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.request_cancellable(method, params, &AtomicBool::new(false))
    }

    fn request_cancellable(
        &mut self,
        method: &str,
        params: Value,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        self.request_until(method, params, cancelled, Instant::now() + CALL_TIMEOUT)
    }

    fn request_until(
        &mut self,
        method: &str,
        params: Value,
        cancelled: &AtomicBool,
        deadline: Instant,
    ) -> Result<Value> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .context("MCP request ID exhausted")?;
        self.send(
            &json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params}),
            deadline,
            cancelled,
        )?;
        let mut received = 0usize;
        loop {
            check_deadline(deadline, cancelled)?;
            if let Some(end) = self.pending.iter().position(|byte| *byte == b'\n') {
                let line: Vec<_> = self.pending.drain(..=end).collect();
                let message: Value =
                    serde_json::from_slice(&line).context("invalid MCP JSON message")?;
                if message.get("method").is_some() {
                    if message.get("id").is_some() {
                        self.send(&json!({"jsonrpc":"2.0", "id":message["id"], "error":{"code":-32601,"message":"Client method not supported"}}), deadline, cancelled)?;
                    }
                    continue;
                }
                if message.get("id").and_then(Value::as_u64) == Some(id) {
                    return rpc_result(message);
                }
                continue;
            }
            let chunk = self
                .read_chunk(deadline, cancelled)?
                .context("MCP stdout closed before response")?;
            received = received
                .checked_add(chunk.len())
                .context("MCP byte budget overflow")?;
            ensure!(
                received <= MAX_RESPONSE,
                "MCP response exceeds 8 MiB aggregate budget"
            );
            self.pending.extend(chunk);
        }
    }
}

impl Drop for StdioClient {
    fn drop(&mut self) {
        // Kill descendants that inherited the pipe too. Never wait for their EOF.
        unsafe {
            libc::kill(-(self.child.id() as i32), libc::SIGKILL);
        }
        let _ = self.child.kill();
        let deadline = Instant::now() + Duration::from_secs(1);
        while Instant::now() < deadline {
            match self.child.try_wait() {
                Ok(Some(_)) | Err(_) => return,
                Ok(None) => thread::sleep(POLL_INTERVAL),
            }
        }
        // The OS may delay reaping a process stuck in kernel I/O; no unbounded wait.
    }
}

struct HttpClient {
    config: ServerConfig,
    session_id: Option<String>,
    protocol_version: String,
    next_id: u64,
}

impl HttpClient {
    fn connect(config: &ServerConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            session_id: None,
            protocol_version: PROTOCOL_VERSION.into(),
            next_id: 1,
        })
    }

    fn exchange(
        &mut self,
        payload: &Value,
        response_expected: bool,
        cancelled: &AtomicBool,
    ) -> Result<Option<Value>> {
        self.exchange_until(
            payload,
            response_expected,
            cancelled,
            Instant::now() + CALL_TIMEOUT,
        )
    }

    fn exchange_until(
        &mut self,
        payload: &Value,
        response_expected: bool,
        cancelled: &AtomicBool,
        deadline: Instant,
    ) -> Result<Option<Value>> {
        check_deadline(deadline, cancelled)?;
        let mut headers = self.config.headers.clone();
        headers.insert("Content-Type".into(), "application/json".into());
        headers.insert("Expect".into(), String::new());
        headers.insert(
            "Accept".into(),
            "application/json, text/event-stream".into(),
        );
        headers.insert("User-Agent".into(), "MLXL3-Desktop/1.0.0".into());
        headers.insert("MCP-Protocol-Version".into(), self.protocol_version.clone());
        if let Some(session) = &self.session_id {
            headers.insert("MCP-Session-Id".into(), session.clone());
        }
        let mut config_file = tempfile::NamedTempFile::new()?;
        writeln!(
            config_file,
            "url = \"{}\"",
            curl_escape(self.config.url.as_deref().unwrap_or_default())
        )?;
        writeln!(
            config_file,
            "request = \"POST\"\nsilent\nshow-error\nmax-time = \"12\"\ndata-binary = \"@-\""
        )?;
        for (key, value) in headers {
            writeln!(
                config_file,
                "header = \"{}: {}\"",
                curl_escape(&key),
                curl_escape(&value)
            )?;
        }
        config_file.flush()?;
        let mut command = Command::new("/usr/bin/curl");
        command.args([
            "--config",
            config_file
                .path()
                .to_str()
                .context("temporary path is not UTF-8")?,
            "--include",
            "--no-buffer",
            "--suppress-connect-headers",
        ]);
        let mut transport = StdioClient::from_command(&mut command)?;
        transport.send(payload, deadline, cancelled)?;
        transport.stdin.take();
        let id = payload.get("id").and_then(Value::as_u64);
        let mut headers = Vec::new();
        let mut headers_done = false;
        let mut sse = false;
        let mut parser = SseParser::default();
        let mut body = Vec::new();
        let mut received = 0usize;
        let mut header_bytes = 0usize;
        'receive: loop {
            let Some(mut chunk) = transport.read_chunk(deadline, cancelled)? else {
                loop {
                    check_deadline(deadline, cancelled)?;
                    if let Some(status) = transport.child.try_wait()? {
                        ensure!(status.success(), "MCP HTTP transport failed");
                        break;
                    }
                    thread::sleep(POLL_INTERVAL);
                }
                ensure!(headers_done, "MCP HTTP headers truncated");
                if !response_expected {
                    return Ok(None);
                }
                ensure!(!sse, "MCP stream ended without a matching response");
                let message: Value =
                    serde_json::from_slice(&body).context("invalid MCP HTTP JSON")?;
                ensure!(
                    message.get("id").and_then(Value::as_u64) == id,
                    "MCP server returned a mismatched response"
                );
                return Ok(Some(message));
            };
            if !headers_done {
                headers.extend_from_slice(&chunk);
                loop {
                    let Some(end) = headers.windows(4).position(|w| w == b"\r\n\r\n") else {
                        ensure!(
                            headers.len() + header_bytes <= MAX_HEADERS,
                            "MCP HTTP headers exceed 64 KiB"
                        );
                        continue 'receive;
                    };
                    header_bytes += end + 4;
                    ensure!(
                        header_bytes <= MAX_HEADERS,
                        "MCP HTTP headers exceed 64 KiB"
                    );
                    let text =
                        std::str::from_utf8(&headers[..end]).context("invalid HTTP headers")?;
                    let status = text
                        .lines()
                        .next()
                        .and_then(|line| line.split_whitespace().nth(1))
                        .context("missing HTTP status")?;
                    if matches!(status, "100" | "102" | "103") {
                        headers.drain(..end + 4);
                        continue;
                    }
                    ensure!(
                        status.starts_with('2'),
                        "MCP HTTP request failed ({status})"
                    );
                    if payload["method"] == "initialize"
                        && let Some(session) = header_value(text, "mcp-session-id")
                    {
                        ensure!(
                            session.bytes().all(|c| (0x21..=0x7e).contains(&c)),
                            "invalid MCP session ID"
                        );
                        self.session_id = Some(session.into());
                    }
                    sse = header_value(text, "content-type").is_some_and(|v| {
                        v.split(';')
                            .next()
                            .is_some_and(|v| v.trim().eq_ignore_ascii_case("text/event-stream"))
                    });
                    chunk = headers.split_off(end + 4);
                    headers.clear();
                    headers_done = true;
                    if !response_expected {
                        return Ok(None);
                    }
                    break;
                }
            }
            received = received
                .checked_add(chunk.len())
                .context("MCP byte budget overflow")?;
            ensure!(
                received <= MAX_RESPONSE,
                "MCP response exceeds 8 MiB aggregate budget"
            );
            if sse {
                for message in parser.feed(&chunk)? {
                    if message.get("method").is_some() {
                        if let Some(request_id) = message.get("id") {
                            self.exchange_until(
                                &json!({"jsonrpc":"2.0", "id":request_id,
                            "error":{"code":-32601,"message":"Client method not supported"}}),
                                false,
                                cancelled,
                                deadline,
                            )?;
                        }
                        continue;
                    }
                    if message.get("id").and_then(Value::as_u64) == id {
                        return Ok(Some(message));
                    }
                }
            } else {
                body.extend_from_slice(&chunk);
            }
        }
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.exchange(
            &json!({"jsonrpc":"2.0", "method":method, "params":params}),
            false,
            &AtomicBool::new(false),
        )?;
        Ok(())
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.request_cancellable(method, params, &AtomicBool::new(false))
    }

    fn request_cancellable(
        &mut self,
        method: &str,
        params: Value,
        cancelled: &AtomicBool,
    ) -> Result<Value> {
        let id = self.next_id;
        self.next_id = self
            .next_id
            .checked_add(1)
            .context("MCP request ID exhausted")?;
        let message = self
            .exchange(
                &json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params}),
                true,
                cancelled,
            )?
            .context("MCP server returned no response")?;
        ensure!(
            message.get("id").and_then(Value::as_u64) == Some(id),
            "MCP server returned a mismatched response"
        );
        rpc_result(message)
    }
}

fn curl_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn header_value<'a>(headers: &'a str, key: &str) -> Option<&'a str> {
    headers.lines().rev().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        name.eq_ignore_ascii_case(key).then(|| value.trim())
    })
}

#[derive(Default)]
struct SseParser {
    line: Vec<u8>,
    data: String,
}

impl SseParser {
    fn feed(&mut self, bytes: &[u8]) -> Result<Vec<Value>> {
        let mut out = Vec::new();
        for byte in bytes {
            if *byte != b'\n' {
                ensure!(
                    self.line.len() + self.data.len() < MAX_RESPONSE,
                    "MCP event exceeds 8 MiB"
                );
                self.line.push(*byte);
                continue;
            }
            if self.line.last() == Some(&b'\r') {
                self.line.pop();
            }
            let line = std::str::from_utf8(&self.line).context("MCP response is not UTF-8")?;
            if line.is_empty() {
                if !self.data.is_empty() {
                    out.push(
                        serde_json::from_str(self.data.trim_end_matches('\n'))
                            .context("invalid MCP SSE JSON")?,
                    );
                    self.data.clear();
                }
            } else if let Some(data) = line.strip_prefix("data:") {
                self.data.push_str(data.strip_prefix(' ').unwrap_or(data));
                self.data.push('\n');
            }
            self.line.clear();
        }
        Ok(out)
    }
}

#[cfg(test)]
fn parse_http_body(body: &[u8]) -> Result<Value> {
    if let Ok(value) = serde_json::from_slice(body) {
        return Ok(value);
    }
    SseParser::default()
        .feed(body)?
        .into_iter()
        .find(|v| v.get("id").is_some() && v.get("method").is_none())
        .context("MCP stream ended without a JSON response")
}

fn rpc_result(message: Value) -> Result<Value> {
    if let Some(error) = message.get("error") {
        bail!(
            "MCP server: {}",
            error
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("request failed")
        );
    }
    message
        .get("result")
        .cloned()
        .filter(Value::is_object)
        .context("MCP server returned an invalid result")
}

pub struct Manager {
    clients: BTreeMap<String, Client>,
    pub tools: BTreeMap<String, Tool>,
    pub errors: BTreeMap<String, String>,
    pub enabled: bool,
}

impl Manager {
    pub fn disabled() -> Self {
        Self {
            clients: BTreeMap::new(),
            tools: BTreeMap::new(),
            errors: BTreeMap::new(),
            enabled: false,
        }
    }

    pub fn set_enabled(&mut self, registry: &Path, enabled: bool, refresh: bool) {
        if self.enabled == enabled && !refresh {
            return;
        }
        *self = Self::disabled();
        self.enabled = enabled;
        if enabled && let Err(error) = self.connect(registry) {
            self.errors
                .insert("configuration".into(), error.to_string());
        }
    }

    pub fn connect(&mut self, registry: &Path) -> Result<()> {
        self.enabled = true;
        for (server, config) in list_servers(registry)? {
            if !config.enabled {
                continue;
            }
            match connect_server(&config) {
                Ok((client, raw_tools)) => {
                    self.clients.insert(server.clone(), client);
                    for raw in raw_tools {
                        let Some(name) = raw.get("name").and_then(Value::as_str) else {
                            continue;
                        };
                        let public_name = unique_name(&server, name, &self.tools);
                        self.tools.insert(
                            public_name.clone(),
                            Tool {
                                server: server.clone(),
                                name: name.into(),
                                public_name,
                                description: raw
                                    .get("description")
                                    .and_then(Value::as_str)
                                    .map(|text| format!("[{server}] {text}"))
                                    .unwrap_or_else(|| {
                                        format!("Tool {name} from MCP server {server}.")
                                    }),
                                input_schema: raw
                                    .get("inputSchema")
                                    .filter(|value| value.is_object())
                                    .cloned()
                                    .unwrap_or_else(|| json!({"type":"object","properties":{}})),
                            },
                        );
                    }
                }
                Err(error) => {
                    self.errors.insert(server, error.to_string());
                }
            }
        }
        Ok(())
    }

    pub fn server_count(&self) -> usize {
        self.clients.len()
    }

    pub fn chat_tools(&self) -> Vec<Value> {
        self.tools.values().map(|tool| json!({"type":"function","function":{"name":tool.public_name,"description":tool.description,"parameters":tool.input_schema}})).collect()
    }

    pub fn call(&mut self, public_name: &str, arguments: Value) -> ToolResult {
        self.call_cancellable(public_name, arguments, &AtomicBool::new(false))
    }

    pub fn call_cancellable(
        &mut self,
        public_name: &str,
        arguments: Value,
        cancelled: &AtomicBool,
    ) -> ToolResult {
        let Some(tool) = self.tools.get(public_name).cloned() else {
            return ToolResult {
                text: format!("Unknown MCP tool: {public_name}"),
                is_error: true,
            };
        };
        let Some(client) = self.clients.get_mut(&tool.server) else {
            return ToolResult {
                text: format!("MCP server unavailable: {}", tool.server),
                is_error: true,
            };
        };
        if !arguments.is_object() {
            return ToolResult {
                text: "Tool arguments must be an object".into(),
                is_error: true,
            };
        }
        match client.request_cancellable(
            "tools/call",
            json!({"name":tool.name,"arguments":arguments}),
            cancelled,
        ) {
            Ok(result) => tool_result(&result),
            Err(error) => {
                self.clients.remove(&tool.server);
                ToolResult {
                    text: error.to_string(),
                    is_error: true,
                }
            }
        }
    }
}

fn connect_server(config: &ServerConfig) -> Result<(Client, Vec<Value>)> {
    let mut client = if config.url.is_some() {
        Client::Http(HttpClient::connect(config)?)
    } else {
        Client::Stdio(StdioClient::connect(config)?)
    };
    let initialized = client.request("initialize", json!({"protocolVersion":PROTOCOL_VERSION,"capabilities":{},"clientInfo":{"name":"MLXL3 Desktop","version":"1.0.0"}}))?;
    if let Some(version) = initialized.get("protocolVersion").and_then(Value::as_str)
        && let Client::Http(http) = &mut client
    {
        http.protocol_version = version.into();
    }
    client.notify("notifications/initialized", json!({}))?;
    let mut tools = Vec::new();
    let mut tools_bytes = 0usize;
    let mut cursor = None;
    for _ in 0..128 {
        let result = client.request(
            "tools/list",
            cursor
                .as_ref()
                .map_or_else(|| json!({}), |cursor| json!({"cursor":cursor})),
        )?;
        // Retained metadata is bounded across pages, not just each wire reply.
        tools_bytes = tools_bytes
            .checked_add(serde_json::to_vec(&result)?.len())
            .context("MCP tool metadata budget overflow")?;
        ensure!(
            tools_bytes <= MAX_RESPONSE,
            "MCP tool metadata exceeds 8 MiB aggregate budget"
        );
        tools.extend(
            result
                .get("tools")
                .and_then(Value::as_array)
                .context("MCP server returned an invalid tools list")?
                .iter()
                .filter(|tool| tool.is_object())
                .cloned(),
        );
        ensure!(
            tools.len() <= 4096,
            "MCP tool pagination exceeded its limit"
        );
        cursor = result
            .get("nextCursor")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .map(str::to_owned);
        if cursor.is_none() {
            return Ok((client, tools));
        }
    }
    bail!("MCP tool pagination exceeded its limit")
}

fn unique_name(server: &str, tool: &str, existing: &BTreeMap<String, Tool>) -> String {
    let base: String = format!("{server}.{tool}")
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || ".-_".contains(c) {
                c
            } else {
                '_'
            }
        })
        .take(128)
        .collect();
    if !existing.contains_key(&base) {
        return base;
    }
    (2..)
        .map(|index| format!("{}_{index}", &base[..base.len().min(124)]))
        .find(|name| !existing.contains_key(name))
        .expect("finite MCP tools")
}

fn tool_result(result: &Value) -> ToolResult {
    let mut parts = Vec::new();
    if let Some(structured) = result.get("structuredContent") {
        parts.push(structured.to_string());
    }
    if let Some(content) = result.get("content").and_then(Value::as_array) {
        for block in content {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        parts.push(text.into());
                    }
                }
                Some("resource_link") => {
                    if let Some(uri) = block.get("uri").and_then(Value::as_str) {
                        parts.push(format!("Resource: {uri}"));
                    }
                }
                Some("image" | "audio") => parts.push(format!(
                    "[{} content returned by MCP server]",
                    block["type"].as_str().unwrap_or("media")
                )),
                _ => {}
            }
        }
    }
    parts.dedup();
    ToolResult {
        text: if parts.is_empty() {
            "Tool completed without text output.".into()
        } else {
            parts.join("\n")
        },
        is_error: result
            .get("isError")
            .and_then(Value::as_bool)
            .unwrap_or(false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn http_fixture(response: Vec<u8>, keep_open: bool) -> Result<Value> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let (done, finish) = std::sync::mpsc::channel();
        let server = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(3);
            let mut socket = loop {
                if let Ok((socket, _)) = listener.accept() {
                    break socket;
                }
                assert!(
                    Instant::now() < deadline,
                    "HTTP test client did not connect"
                );
                thread::sleep(POLL_INTERVAL);
            };
            // Accepted sockets inherit O_NONBLOCK from the listener on some
            // platforms (e.g. macOS); write_all would fail with WouldBlock.
            socket.set_nonblocking(false).unwrap();
            socket
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            socket
                .set_write_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = [0; 4096];
            let _ = socket.read(&mut request).unwrap();
            let _ = socket.write_all(&response);
            if keep_open {
                let _ = finish.recv_timeout(Duration::from_secs(2));
            }
        });
        let mut client = HttpClient::connect(&ServerConfig {
            url: Some(format!("http://{address}/mcp")),
            ..builtin_exa()
        })
        .unwrap();
        let started = Instant::now();
        let result = client.request("tools/list", json!({}));
        if keep_open {
            assert!(
                started.elapsed() < Duration::from_secs(1),
                "client waited for SSE EOF"
            );
        }
        let _ = done.send(());
        server.join().unwrap();
        result
    }

    #[test]
    fn http_consumes_matching_sse_before_eof_and_checks_budgets() {
        let response = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n: hi\r\n\r\ndata: {\"method\":\"notifications/progress\"}\r\n\r\ndata: {\"id\":99,\"result\":{}}\r\n\r\ndata: {\"id\":1,\"result\":{\"ok\":true}}\r\n\r\n";
        assert_eq!(
            http_fixture(response.to_vec(), true).unwrap(),
            json!({"ok":true})
        );
        assert_eq!(http_fixture(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"id\":1,\"result\":{}}".to_vec(), false).unwrap(), json!({}));
        assert!(http_fixture(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: 10000\r\n\r\n{\"id\":1,\"result\":{}}".to_vec(), false).is_err());
        let mut oversized = b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n".to_vec();
        oversized.extend(vec![b' '; MAX_RESPONSE + 1]);
        assert!(
            http_fixture(oversized, false)
                .unwrap_err()
                .to_string()
                .contains("aggregate budget")
        );
        let mut oversized_header = b"HTTP/1.1 200 OK\r\nX-Test: ".to_vec();
        oversized_header.extend(vec![b'x'; MAX_HEADERS]);
        oversized_header.extend_from_slice(b"\r\n\r\n");
        assert!(
            http_fixture(oversized_header, false)
                .unwrap_err()
                .to_string()
                .contains("headers exceed")
        );
        assert!(
            http_fixture(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\ndata: {\"id\":1"
                    .to_vec(),
                false
            )
            .is_err()
        );
    }

    #[test]
    fn sse_fragmented_notifications_crlf_and_truncation() {
        let expected = json!({"jsonrpc":"2.0","id":9,"result":{"ok":true}});
        let input = format!(
            ": comment\r\n\r\ndata: {{\"method\":\"notifications/progress\"}}\r\n\r\ndata: {expected}\r\n\r\n"
        );
        for cut in 0..=input.len() {
            let mut parser = SseParser::default();
            let mut values = parser.feed(&input.as_bytes()[..cut]).unwrap();
            values.extend(parser.feed(&input.as_bytes()[cut..]).unwrap());
            assert_eq!(values.len(), 2);
            assert_eq!(values[1], expected);
        }
        assert!(
            SseParser::default()
                .feed(b"data: {\"id\":9}")
                .unwrap()
                .is_empty()
        );
        assert!(SseParser::default().feed(b"data: invalid\n\n").is_err());
    }

    fn python_client(source: &str) -> StdioClient {
        let mut command = Command::new("python3");
        command.args(["-u", "-c", source]);
        StdioClient::from_command(&mut command).unwrap()
    }

    #[test]
    fn nonreading_stdin_and_noisy_stderr_have_absolute_deadlines() {
        let mut client =
            python_client("import time,sys; sys.stderr.write('x'*200000); time.sleep(10)");
        let started = Instant::now();
        let result = client.send(
            &json!({"data":"x".repeat(1024*1024)}),
            started + Duration::from_millis(60),
            &AtomicBool::new(false),
        );
        assert!(result.unwrap_err().to_string().contains("timed out"));
        drop(client);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn late_result_is_rejected_by_the_same_absolute_deadline() {
        let mut client = python_client(
            "import sys,time; sys.stdin.readline(); time.sleep(.2); print('{\"id\":1,\"result\":{}}')",
        );
        assert!(
            client
                .request_until(
                    "tools/call",
                    json!({}),
                    &AtomicBool::new(false),
                    Instant::now() + Duration::from_millis(50)
                )
                .unwrap_err()
                .to_string()
                .contains("timed out")
        );
    }

    #[test]
    fn notifications_do_not_reset_request_deadline() {
        let mut client = python_client(
            "import sys,time; sys.stdin.readline();\nwhile True: print('{\"method\":\"notifications/progress\"}'); time.sleep(.001)",
        );
        let started = Instant::now();
        assert!(
            client
                .request_until(
                    "tools/call",
                    json!({}),
                    &AtomicBool::new(false),
                    started + Duration::from_millis(60)
                )
                .unwrap_err()
                .to_string()
                .contains("timed out")
        );
        drop(client);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn cancellation_interrupts_active_notification_stream() {
        let mut client = python_client(
            "import sys,time; sys.stdin.readline();\nwhile True: print('{\"method\":\"notifications/progress\"}'); time.sleep(.001)",
        );
        let cancelled = AtomicBool::new(false);
        let started = Instant::now();
        thread::scope(|scope| {
            scope.spawn(|| {
                thread::sleep(Duration::from_millis(60));
                cancelled.store(true, Ordering::Relaxed);
            });
            assert!(
                client
                    .request_cancellable("tools/call", json!({}), &cancelled)
                    .unwrap_err()
                    .to_string()
                    .contains("cancelled")
            );
        });
        drop(client);
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn stdio_rejects_aggregate_notification_overflow() {
        let mut client = python_client(
            "import sys; sys.stdin.readline();\nwhile True: print('{\"method\":\"noise\",\"data\":\"'+'x'*65536+'\"}')",
        );
        assert!(
            client
                .request("tools/call", json!({}))
                .unwrap_err()
                .to_string()
                .contains("aggregate budget")
        );
    }

    #[test]
    fn concurrent_configuration_transactions_preserve_updates() {
        let root = tempfile::tempdir().unwrap();
        let registry = root.path().join("models.json");
        let barrier = std::sync::Barrier::new(8);
        thread::scope(|scope| {
            for index in 0..8 {
                let registry = &registry;
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    add_server(registry, &format!("server{index}"), "true", vec![]).unwrap();
                });
            }
        });
        assert_eq!(list_servers(&registry).unwrap().len(), 9);
        let barrier = std::sync::Barrier::new(2);
        thread::scope(|scope| {
            scope.spawn(|| {
                barrier.wait();
                remove_server(&registry, "server0").unwrap();
            });
            scope.spawn(|| {
                barrier.wait();
                add_server(&registry, "new", "true", vec![]).unwrap();
            });
        });
        let servers = list_servers(&registry).unwrap();
        assert_eq!(servers.len(), 9);
        assert!(servers.contains_key("new"));
        assert!(!servers.contains_key("server0"));
    }

    #[test]
    fn mcp_url_validation_matches_the_desktop_transports() {
        for url in [
            "https://example.com/mcp",
            "http://127.0.0.1:9000/mcp",
            "http://localhost/mcp",
            "http://[::1]:9000/mcp",
        ] {
            assert!(super::validate_url(url).is_ok(), "rejected {url}");
        }
        for url in [
            "http://example.com/mcp",
            "https://",
            "file:///tmp/mcp",
            "https://user:password@example.com/mcp",
        ] {
            assert!(super::validate_url(url).is_err(), "accepted {url}");
        }
    }

    #[test]
    fn parses_json_and_sse_responses() {
        let expected = json!({"jsonrpc":"2.0","id":1,"result":{}});
        assert_eq!(
            parse_http_body(expected.to_string().as_bytes()).unwrap(),
            expected
        );
        let sse = format!("event: message\ndata: {}\n\n", expected);
        assert_eq!(parse_http_body(sse.as_bytes()).unwrap(), expected);
        assert!(validate_url("http://example.com/mcp").is_err());
        assert!(validate_url("http://127.0.0.1:9999/mcp").is_ok());
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    #[kani::unwind(10)]
    fn mcp_server_names_are_nonempty_and_shell_neutral() {
        let bytes: [u8; 8] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= bytes.len());
        let name = &bytes[..len];
        let expected = !name.is_empty()
            && name
                .iter()
                .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(byte));
        assert_eq!(valid_server_name(name), expected);
        kani::cover!(valid_server_name(name));
        kani::cover!(!valid_server_name(name));
    }
}
