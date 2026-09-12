//! Minimal MCP client shared by the native CLI and desktop bridge.
use anyhow::{Context, Result, bail, ensure};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::mpsc::{self, Receiver},
    thread,
    time::Duration,
};

const PROTOCOL_VERSION: &str = "2025-11-25";
const MAX_RESPONSE: u64 = 8 * 1024 * 1024;

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
    let lock = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path.with_extension("lock"))?;
    lock.lock_exclusive()?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temporary, payload)?;
    temporary.write_all(b"\n")?;
    temporary.as_file().sync_all()?;
    temporary
        .persist(path)
        .context("saving MCP configuration")?;
    Ok(())
}

fn read_config(registry: &Path) -> Result<ConfigFile> {
    let path = config_path(registry);
    if !path.exists() {
        let payload = ConfigFile {
            version: 1,
            servers: BTreeMap::from([("exa".into(), builtin_exa())]),
        };
        write_config(&path, &payload)?;
        return Ok(payload);
    }
    let mut payload: ConfigFile =
        serde_json::from_reader(File::open(&path)?).context("invalid MCP configuration")?;
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
            !name.is_empty()
                && name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c)),
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

fn validate_url(url: &str) -> Result<()> {
    if url.starts_with("https://") {
        ensure!(url.len() > 8, "MCP HTTPS URL has no host");
        return Ok(());
    }
    let host = url
        .strip_prefix("http://")
        .and_then(|rest| rest.split(['/', ':']).next())
        .unwrap_or_default();
    ensure!(
        matches!(host, "localhost" | "127.0.0.1" | "[::1]"),
        "MCP endpoints require HTTPS (HTTP is allowed for loopback)"
    );
    Ok(())
}

pub fn list_servers(registry: &Path) -> Result<BTreeMap<String, ServerConfig>> {
    Ok(read_config(registry)?.servers)
}

pub fn add_server(registry: &Path, name: &str, command: &str, args: Vec<String>) -> Result<()> {
    let mut payload = read_config(registry)?;
    let config = ServerConfig {
        command: command.into(),
        args,
        env: BTreeMap::new(),
        cwd: None,
        enabled: true,
        url: None,
        headers: BTreeMap::new(),
    };
    payload.servers.insert(name.into(), config);
    validate_configs(&payload.servers)?;
    write_config(&config_path(registry), &payload)
}

pub fn remove_server(registry: &Path, name: &str) -> Result<()> {
    let mut payload = read_config(registry)?;
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
    write_config(&config_path(registry), &payload)
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

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        match self {
            Self::Http(client) => client.notify(method, params),
            Self::Stdio(client) => client.notify(method, params),
        }
    }
}

struct StdioClient {
    child: Child,
    stdin: ChildStdin,
    messages: Receiver<std::result::Result<Value, String>>,
    next_id: u64,
}

impl StdioClient {
    fn connect(config: &ServerConfig) -> Result<Self> {
        let mut command = Command::new(&config.command);
        command
            .args(&config.args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        command.envs(&config.env);
        if let Some(cwd) = &config.cwd {
            command.current_dir(cwd);
        }
        let mut child = command
            .spawn()
            .with_context(|| format!("starting {}", config.command))?;
        let stdin = child.stdin.take().context("MCP process has no stdin")?;
        let stdout = child.stdout.take().context("MCP process has no stdout")?;
        if let Some(stderr) = child.stderr.take() {
            thread::spawn(move || {
                for line in BufReader::new(stderr).lines() {
                    if line.is_err() {
                        break;
                    }
                }
            });
        }
        let (sender, messages) = mpsc::sync_channel(128);
        thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                let mut line = String::new();
                match reader.by_ref().take(MAX_RESPONSE + 1).read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) if line.len() as u64 > MAX_RESPONSE => {
                        let _ = sender.send(Err("MCP response exceeds 8 MiB".into()));
                        break;
                    }
                    Ok(_) => {
                        if let Ok(value) = serde_json::from_str(&line)
                            && sender.send(Ok(value)).is_err()
                        {
                            break;
                        }
                    }
                    Err(error) => {
                        let _ = sender.send(Err(error.to_string()));
                        break;
                    }
                }
            }
        });
        Ok(Self {
            child,
            stdin,
            messages,
            next_id: 1,
        })
    }

    fn send(&mut self, payload: &Value) -> Result<()> {
        ensure!(self.child.try_wait()?.is_none(), "MCP server exited");
        serde_json::to_writer(&mut self.stdin, payload)?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        Ok(())
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.send(&json!({"jsonrpc":"2.0", "method":method, "params":params}))
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        self.send(&json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params}))?;
        loop {
            let message = self
                .messages
                .recv_timeout(Duration::from_secs(12))
                .context("MCP request timed out")?
                .map_err(anyhow::Error::msg)?;
            if message.get("method").is_some() && message.get("id").is_some() {
                self.send(&json!({"jsonrpc":"2.0", "id":message["id"], "error":{"code":-32601,"message":"Client method not supported"}}))?;
                continue;
            }
            if message.get("id").and_then(Value::as_u64) != Some(id) {
                continue;
            }
            return rpc_result(message);
        }
    }
}

impl Drop for StdioClient {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
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

    fn exchange(&mut self, payload: &Value, response_expected: bool) -> Result<Option<Value>> {
        let mut headers = self.config.headers.clone();
        headers.insert("Content-Type".into(), "application/json".into());
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
        let header_file = tempfile::NamedTempFile::new()?;
        let body_file = tempfile::NamedTempFile::new()?;
        let mut child = Command::new("/usr/bin/curl")
            .args([
                "--config",
                config_file
                    .path()
                    .to_str()
                    .context("temporary path is not UTF-8")?,
                "--dump-header",
                header_file
                    .path()
                    .to_str()
                    .context("temporary path is not UTF-8")?,
                "--output",
                body_file
                    .path()
                    .to_str()
                    .context("temporary path is not UTF-8")?,
                "--write-out",
                "%{http_code}",
            ])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        serde_json::to_writer(child.stdin.take().context("curl has no stdin")?, payload)?;
        let output = child.wait_with_output()?;
        let status = String::from_utf8_lossy(&output.stdout);
        ensure!(
            output.status.success() && status.starts_with('2'),
            "MCP HTTP request failed ({status}): {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let response_headers = fs::read_to_string(header_file.path())?;
        if payload.get("method") == Some(&Value::String("initialize".into()))
            && let Some(session) = header_value(&response_headers, "mcp-session-id")
        {
            ensure!(
                session.bytes().all(|c| (0x21..=0x7e).contains(&c)),
                "invalid MCP session ID"
            );
            self.session_id = Some(session.into());
        }
        if !response_expected {
            return Ok(None);
        }
        let body = fs::read(body_file.path())?;
        ensure!(
            body.len() as u64 <= MAX_RESPONSE,
            "MCP response exceeds 8 MiB"
        );
        let value = parse_http_body(&body)?;
        Ok(Some(value))
    }

    fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.exchange(
            &json!({"jsonrpc":"2.0", "method":method, "params":params}),
            false,
        )?;
        Ok(())
    }

    fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let message = self
            .exchange(
                &json!({"jsonrpc":"2.0", "id":id, "method":method, "params":params}),
                true,
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

fn parse_http_body(body: &[u8]) -> Result<Value> {
    if let Ok(value) = serde_json::from_slice(body) {
        return Ok(value);
    }
    let text = std::str::from_utf8(body).context("MCP response is not UTF-8")?;
    for event in text.split("\n\n") {
        let data = event
            .lines()
            .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
            .collect::<Vec<_>>()
            .join("\n");
        if !data.is_empty()
            && let Ok(value) = serde_json::from_str(&data)
        {
            return Ok(value);
        }
    }
    bail!("MCP stream ended without a JSON response")
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
        match client.request(
            "tools/call",
            json!({"name":tool.name,"arguments":arguments}),
        ) {
            Ok(result) => tool_result(&result),
            Err(error) => ToolResult {
                text: error.to_string(),
                is_error: true,
            },
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
    let mut cursor = None;
    for _ in 0..128 {
        let result = client.request(
            "tools/list",
            cursor
                .as_ref()
                .map_or_else(|| json!({}), |cursor| json!({"cursor":cursor})),
        )?;
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
