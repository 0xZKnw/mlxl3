//! Local tokenizer and text-only chat templates. Never downloads model assets.
use anyhow::{Context, Result, ensure};
use minijinja::{Environment, Error, ErrorKind, context};
use serde::Serialize;
use serde_json::Value;
use std::{fs, path::Path};
use tokenizers::Tokenizer;

#[derive(Clone, Debug, Serialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Levels the bridge accepts as `reasoning_effort` (`none` turns thinking off). Each
/// template supports its own subset: Inkling 2 all of them, Qwen3.8 none/low/medium/xhigh.
pub const REASONING_EFFORTS: [&str; 6] = ["none", "low", "medium", "high", "xhigh", "max"];

pub struct ChatTokenizer {
    tokenizer: Tokenizer,
    environment: Environment<'static>,
    bos_token: String,
    eos_token: String,
    eos_ids: Vec<u32>,
    control_tokens: std::collections::BTreeMap<u32, String>,
}

/// A content part as `vision_input` prepares it for the chat template: `{"type": "text", "text": ...}`
/// or an image slot `{"type": "image"}` (the pixels stay outside the template).
fn prepared_part(part: &Value) -> bool {
    let Some(fields) = part.as_object() else {
        return false;
    };
    match part["type"].as_str() {
        Some("text") => fields.len() == 2 && part["text"].is_string(),
        Some("image") => fields.len() == 1,
        _ => false,
    }
}

#[test]
fn prepared_parts_are_text_or_image_slots_only() {
    use serde_json::json;
    assert!(prepared_part(&json!({"type":"text","text":"a"})));
    assert!(prepared_part(&json!({"type":"image"})));
    for bad in [
        json!({"type":"text"}),
        json!({"type":"text","text":1}),
        json!({"type":"text","text":"a","extra":0}),
        json!({"type":"image","image_url":{"url":"data:image/png;base64,AAAA"}}),
        json!({"type":"image_url","image_url":{"url":"x"}}),
        json!("text"),
        json!(null),
    ] {
        assert!(!prepared_part(&bad), "{bad}");
    }
}

fn json_object(path: &Path) -> Result<Value> {
    let value: Value = serde_json::from_slice(&fs::read(path)?)
        .with_context(|| format!("invalid JSON in {}", path.display()))?;
    ensure!(
        value.is_object(),
        "{} must be a JSON object",
        path.display()
    );
    Ok(value)
}

fn special_token(config: &Value, name: &str, tokenizer: &Tokenizer) -> Result<Option<String>> {
    let Some(value) = config.get(name).filter(|value| !value.is_null()) else {
        return Ok(None);
    };
    let content = if value.is_object() {
        value.get("content")
    } else {
        Some(value)
    }
    .and_then(Value::as_str)
    .with_context(|| format!("{name} must be a string or an object with string content"))?;
    ensure!(!content.is_empty(), "{name} must not be empty");
    ensure!(
        tokenizer.token_to_id(content).is_some(),
        "{name} {content:?} is absent from tokenizer vocabulary"
    );
    Ok(Some(content.to_owned()))
}

fn eos_ids(value: &Value, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    let values = value
        .as_array()
        .map(Vec::as_slice)
        .unwrap_or(std::slice::from_ref(value));
    ensure!(
        !values.is_empty(),
        "eos_token_id must not be an empty array"
    );
    let mut ids = Vec::new();
    for value in values {
        let id = value
            .as_u64()
            .and_then(|id| u32::try_from(id).ok())
            .context("eos_token_id must contain unsigned 32-bit integers")?;
        ensure!(
            tokenizer.id_to_token(id).is_some(),
            "EOS token {id} is absent from tokenizer vocabulary"
        );
        if !ids.contains(&id) {
            ids.push(id);
        }
    }
    Ok(ids)
}

// Jinja's generation extension only tracks training masks. An always-true
// block preserves rendering and the original whitespace-control delimiters.
fn inference_template(source: &str) -> String {
    let mut output = String::with_capacity(source.len());
    let mut rest = source;
    let mut raw = false;
    while let Some(start) = rest.find('{') {
        output.push_str(&rest[..start]);
        rest = &rest[start..];
        let close = if rest.starts_with("{#") {
            "#}"
        } else if rest.starts_with("{{") {
            "}}"
        } else if rest.starts_with("{%") {
            "%}"
        } else {
            output.push('{');
            rest = &rest[1..];
            continue;
        };
        // Do not mistake delimiters inside a quoted Jinja string for tag ends.
        let mut quote = None;
        let mut escaped = false;
        let mut end = None;
        for (offset, c) in rest[2..].char_indices() {
            let index = offset + 2;
            if quote.is_none() && rest[index..].starts_with(close) {
                end = Some(index + 2);
                break;
            }
            if close == "#}" {
                continue;
            }
            if escaped {
                escaped = false;
            } else if quote.is_some() && c == '\\' {
                escaped = true;
            } else if quote == Some(c) {
                quote = None;
            } else if quote.is_none() && matches!(c, '\'' | '"') {
                quote = Some(c);
            }
        }
        let Some(end) = end else {
            break;
        };
        let tag = &rest[..end];
        if close == "%}" {
            let body = &tag[2..tag.len() - 2];
            let keyword = body.trim().trim_matches(['-', '+']).trim();
            let replacement = match keyword {
                "raw" => {
                    raw = true;
                    None
                }
                "endraw" => {
                    raw = false;
                    None
                }
                "generation" if !raw => Some("if true"),
                "endgeneration" if !raw => Some("endif"),
                _ => None,
            };
            if let Some(replacement) = replacement {
                let offset = 2 + body.find(keyword).expect("keyword belongs to tag body");
                output.push_str(&tag[..offset]);
                output.push_str(replacement);
                output.push_str(&tag[offset + keyword.len()..]);
            } else {
                output.push_str(tag);
            }
        } else {
            output.push_str(tag);
        }
        rest = &rest[end..];
    }
    output.push_str(rest);
    output
}

impl ChatTokenizer {
    pub fn load(path: &Path) -> Result<Self> {
        let mut tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
            .map_err(|error| anyhow::anyhow!("loading tokenizer.json: {error}"))?;
        tokenizer
            .with_truncation(None)
            .map_err(|error| anyhow::anyhow!("disabling tokenizer truncation: {error}"))?;
        tokenizer.with_padding(None);
        let config = json_object(&path.join("tokenizer_config.json"))?;
        let bos_token = special_token(&config, "bos_token", &tokenizer)?.unwrap_or_default();
        let eos_token = special_token(&config, "eos_token", &tokenizer)?.unwrap_or_default();
        let mut stop_ids = None;
        for filename in ["generation_config.json", "config.json"] {
            let config_path = path.join(filename);
            if !config_path.exists() {
                continue;
            }
            let model_config = json_object(&config_path)?;
            if let Some(value) = model_config
                .get("eos_token_id")
                .filter(|value| !value.is_null())
            {
                stop_ids = Some(eos_ids(value, &tokenizer)?);
                break;
            }
        }
        let eos_ids = stop_ids
            .or_else(|| tokenizer.token_to_id(&eos_token).map(|id| vec![id]))
            .context("model has no usable EOS token")?;
        let template_path = path.join("chat_template.jinja");
        let template = match fs::read_to_string(&template_path) {
            Ok(template) => template,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => config
                .get("chat_template")
                .and_then(Value::as_str)
                .context("missing chat_template.jinja or string chat_template")?
                .to_owned(),
            Err(error) => return Err(error).context("reading chat_template.jinja"),
        };
        ensure!(!template.trim().is_empty(), "chat template is empty");
        let mut environment = Environment::new();
        environment.set_trim_blocks(true);
        environment.set_lstrip_blocks(true);
        environment
            .set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        environment.add_function(
            "raise_exception",
            |message: String| -> std::result::Result<String, Error> {
                Err(Error::new(ErrorKind::InvalidOperation, message))
            },
        );
        environment.add_filter("tojson", python_tojson);
        environment.add_function("strftime_now", |format: String| {
            if format == "%Y-%m-%d" {
                Ok(time::OffsetDateTime::now_utc().date().to_string())
            } else {
                Err(Error::new(
                    ErrorKind::InvalidOperation,
                    "unsupported date format",
                ))
            }
        });
        environment
            .add_template_owned("chat", inference_template(&template))
            .context("compiling chat template")?;
        let control_tokens = tokenizer
            .get_added_tokens_decoder()
            .into_iter()
            .filter(|(_, token)| {
                matches!(
                    token.content.as_str(),
                    "<think>"
                        | "</think>"
                        | "<|channel>"
                        | "<|channel>thought"
                        | "<channel|>"
                        // Inkling 2 message structure (see streaming::ThinkingSplitter)
                        | "<|message_model|>"
                        | "<|content_thinking|>"
                        | "<|content_text|>"
                        | "<|end_message|>"
                )
            })
            .map(|(id, token)| (id, token.content))
            .collect();
        Ok(Self {
            control_tokens,
            tokenizer,
            environment,
            bos_token,
            eos_token,
            eos_ids,
        })
    }

    pub fn render(&self, messages: &[Message]) -> Result<String> {
        self.render_values(&serde_json::to_value(messages)?, None)
    }

    pub fn render_values(&self, messages: &Value, tools: Option<&[Value]>) -> Result<String> {
        self.render_values_options(messages, tools, None, None)
    }

    pub fn render_values_with_thinking(
        &self,
        messages: &Value,
        tools: Option<&[Value]>,
        thinking: bool,
    ) -> Result<String> {
        self.render_values_options(messages, tools, Some(thinking), None)
    }

    /// Renders with a requested reasoning level: `None` keeps the template default, `none`
    /// turns thinking off, any other level enables thinking with `reasoning_effort` set to it
    /// (the template rejects a level it does not support).
    pub fn render_values_with_effort(
        &self,
        messages: &Value,
        tools: Option<&[Value]>,
        effort: Option<&str>,
    ) -> Result<String> {
        match effort {
            None => self.render_values_options(messages, tools, None, None),
            Some(level) => {
                ensure!(
                    REASONING_EFFORTS.contains(&level),
                    "unsupported reasoning effort {level:?}"
                );
                if level == "none" {
                    self.render_values_options(messages, tools, Some(false), None)
                } else {
                    self.render_values_options(messages, tools, Some(true), Some(level))
                }
            }
        }
    }

    fn render_values_options(
        &self,
        messages: &Value,
        tools: Option<&[Value]>,
        thinking: Option<bool>,
        effort: Option<&str>,
    ) -> Result<String> {
        let messages = messages.as_array().context("messages must be an array")?;
        ensure!(!messages.is_empty(), "chat requires at least one message");
        for message in messages {
            let role = message
                .get("role")
                .and_then(Value::as_str)
                .context("chat message requires a role")?;
            ensure!(
                matches!(role, "system" | "user" | "assistant" | "tool"),
                "unsupported role {role:?}"
            );
            // strings, or the parts `vision_input` keeps for Inkling 2 (text parts and image slots)
            ensure!(
                message
                    .get("content")
                    .is_some_and(|content| content.is_string()
                        || content.as_array().is_some_and(
                            |parts| !parts.is_empty() && parts.iter().all(prepared_part)
                        )),
                "chat message requires string content"
            );
        }
        let values = context! {
            messages => messages,
            bos_token => &self.bos_token,
            eos_token => &self.eos_token,
            add_generation_prompt => true,
            preserve_thinking => true,
            tools => tools,
            documents => Option::<bool>::None,
        };
        let values = if let Some(thinking) = thinking {
            // Inkling 2 reads `reasoning_effort` (0 = no thinking, default 0.9); Qwen3.8 reads
            // low/medium/xhigh; other templates ignore it
            if let Some(effort) = effort.filter(|_| thinking) {
                context! { enable_thinking => thinking, reasoning_effort => effort, ..values }
            } else if thinking {
                context! { enable_thinking => thinking, ..values }
            } else {
                context! { enable_thinking => thinking, reasoning_effort => 0.0, ..values }
            }
        } else {
            values
        };
        self.environment
            .get_template("chat")?
            .render(values)
            .context("rendering chat template")
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer
            .encode(text, false)
            .map(|encoded| encoded.get_ids().to_vec())
            .map_err(|error| anyhow::anyhow!("encoding prompt: {error}"))
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|error| anyhow::anyhow!("decoding tokens: {error}"))
    }

    /// Only tokenizer-declared delimiter IDs are channel controls. Added-token
    /// `special` controls decode skipping, not the model protocol. The same bytes
    /// assembled from ordinary tokens are literal text.
    pub fn control_marker(&self, id: u32) -> Option<&str> {
        self.control_tokens.get(&id).map(String::as_str)
    }

    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// Gives the caller access to decode_stream(false) for incremental output.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

fn json_error(error: impl std::fmt::Display) -> Error {
    Error::new(ErrorKind::InvalidOperation, error.to_string())
}

/// `tojson` as transformers defines it (`json.dumps`, `ensure_ascii=False`). Without options
/// it keeps the engine's compact output, byte for byte, for every template; with
/// `sort_keys`, `separators`, `indent` or `ensure_ascii` (Inkling 2 declares its tools with
/// `sort_keys=true, separators=(",", ":")`) it writes what Python writes.
fn python_tojson(
    value: minijinja::Value,
    options: minijinja::value::Kwargs,
) -> std::result::Result<String, Error> {
    if options.args().next().is_none() {
        return serde_json::to_string(&value).map_err(json_error);
    }
    let sort_keys = options.get::<Option<bool>>("sort_keys")?.unwrap_or(false);
    let ensure_ascii = options
        .get::<Option<bool>>("ensure_ascii")?
        .unwrap_or(false);
    let indent = options.get::<Option<usize>>("indent")?;
    let separators = options.get::<Option<minijinja::Value>>("separators")?;
    options.assert_all_used()?;
    let (item, key) = match separators.filter(|pair| !pair.is_none()) {
        Some(pair) => {
            let parts = pair
                .try_iter()?
                .map(|part| part.as_str().map(str::to_owned))
                .collect::<Option<Vec<_>>>();
            match parts.as_deref() {
                Some([item, key]) => (item.clone(), key.clone()),
                _ => return Err(json_error("tojson separators must be two strings")),
            }
        }
        None => (
            if indent.is_some() { "," } else { ", " }.to_owned(),
            ": ".to_owned(),
        ),
    };
    let mut out = Vec::new();
    let formatter = PythonJson {
        item: &item,
        key: &key,
        indent,
        level: 0,
        has_value: false,
        ensure_ascii,
    };
    let mut serializer = serde_json::Serializer::with_formatter(&mut out, formatter);
    if sort_keys {
        // serde_json maps are ordered by key (UTF-8 bytes: code point order, as Python sorts).
        serde_json::to_value(&value)
            .map_err(json_error)?
            .serialize(&mut serializer)
            .map_err(json_error)?;
    } else {
        value.serialize(&mut serializer).map_err(json_error)?;
    }
    String::from_utf8(out).map_err(json_error)
}

/// `json.dumps` layout: item and key separators, optional indentation, optional ASCII.
struct PythonJson<'a> {
    item: &'a str,
    key: &'a str,
    indent: Option<usize>,
    level: usize,
    has_value: bool,
    ensure_ascii: bool,
}

impl PythonJson<'_> {
    fn newline<W: ?Sized + std::io::Write>(&self, writer: &mut W) -> std::io::Result<()> {
        if let Some(width) = self.indent {
            writer.write_all(b"\n")?;
            writer.write_all(" ".repeat(width * self.level).as_bytes())?;
        }
        Ok(())
    }
}

impl serde_json::ser::Formatter for PythonJson<'_> {
    fn begin_array<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.level += 1;
        self.has_value = false;
        writer.write_all(b"[")
    }

    fn end_array<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.level -= 1;
        if self.has_value {
            self.newline(writer)?;
        }
        writer.write_all(b"]")
    }

    fn begin_array_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(self.item.as_bytes())?;
        }
        self.newline(writer)
    }

    fn end_array_value<W: ?Sized + std::io::Write>(&mut self, _: &mut W) -> std::io::Result<()> {
        self.has_value = true;
        Ok(())
    }

    fn begin_object<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.level += 1;
        self.has_value = false;
        writer.write_all(b"{")
    }

    fn end_object<W: ?Sized + std::io::Write>(&mut self, writer: &mut W) -> std::io::Result<()> {
        self.level -= 1;
        if self.has_value {
            self.newline(writer)?;
        }
        writer.write_all(b"}")
    }

    fn begin_object_key<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(self.item.as_bytes())?;
        }
        self.newline(writer)
    }

    fn begin_object_value<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(self.key.as_bytes())
    }

    fn end_object_value<W: ?Sized + std::io::Write>(&mut self, _: &mut W) -> std::io::Result<()> {
        self.has_value = true;
        Ok(())
    }

    fn write_string_fragment<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        fragment: &str,
    ) -> std::io::Result<()> {
        if !self.ensure_ascii {
            return writer.write_all(fragment.as_bytes());
        }
        for character in fragment.chars() {
            if character.is_ascii() {
                writer.write_all(&[character as u8])?;
            } else {
                for unit in character.encode_utf16(&mut [0; 2]) {
                    write!(writer, "\\u{unit:04x}")?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    /// `tojson` against Python's `json.dumps` (outputs computed with Python 3, embedded).
    #[test]
    fn tojson_options_match_python_json_dumps() {
        let mut environment = Environment::new();
        environment.add_filter("tojson", python_tojson);
        let value = r#"{"name": "sandbox.javascript", "description": "Run JS é \"q\" \u2028", "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}, "n": [1, 2.5, true, none, []], "e": {}}"#;
        let render = |filter: &str| {
            let source = format!("{{{{ {value} | {filter} }}}}");
            environment.render_str(&source, context! {}).unwrap()
        };
        let python = [
            (
                "tojson(sort_keys=true, separators=(\",\", \":\"))",
                "{\"description\":\"Run JS é \\\"q\\\" \u{2028}\",\"e\":{},\"n\":[1,2.5,true,null,[]],\"name\":\"sandbox.javascript\",\"parameters\":{\"properties\":{\"code\":{\"type\":\"string\"}},\"required\":[\"code\"],\"type\":\"object\"}}",
            ),
            // minijinja (no `preserve_order`) iterates maps by key, so an unsorted dump equals
            // Python's sorted one; what these cases check is the layout.
            (
                "tojson(separators=(\",\", \":\"))",
                "{\"description\":\"Run JS é \\\"q\\\" \u{2028}\",\"e\":{},\"n\":[1,2.5,true,null,[]],\"name\":\"sandbox.javascript\",\"parameters\":{\"properties\":{\"code\":{\"type\":\"string\"}},\"required\":[\"code\"],\"type\":\"object\"}}",
            ),
            (
                "tojson(sort_keys=true)",
                "{\"description\": \"Run JS é \\\"q\\\" \u{2028}\", \"e\": {}, \"n\": [1, 2.5, true, null, []], \"name\": \"sandbox.javascript\", \"parameters\": {\"properties\": {\"code\": {\"type\": \"string\"}}, \"required\": [\"code\"], \"type\": \"object\"}}",
            ),
            (
                "tojson(indent=2)",
                "{\n  \"description\": \"Run JS é \\\"q\\\" \u{2028}\",\n  \"e\": {},\n  \"n\": [\n    1,\n    2.5,\n    true,\n    null,\n    []\n  ],\n  \"name\": \"sandbox.javascript\",\n  \"parameters\": {\n    \"properties\": {\n      \"code\": {\n        \"type\": \"string\"\n      }\n    },\n    \"required\": [\n      \"code\"\n    ],\n    \"type\": \"object\"\n  }\n}",
            ),
            (
                "tojson(ensure_ascii=true, sort_keys=true, separators=(\",\", \":\"))",
                "{\"description\":\"Run JS \\u00e9 \\\"q\\\" \\u2028\",\"e\":{},\"n\":[1,2.5,true,null,[]],\"name\":\"sandbox.javascript\",\"parameters\":{\"properties\":{\"code\":{\"type\":\"string\"}},\"required\":[\"code\"],\"type\":\"object\"}}",
            ),
        ];
        for (filter, expected) in python {
            assert_eq!(render(filter), expected, "{filter}");
        }
        // Without options: the engine's compact output, exactly what it wrote before.
        let mut legacy = Environment::new();
        legacy.add_filter("tojson", |value: minijinja::Value| {
            serde_json::to_string(&value).map_err(json_error)
        });
        let source = format!("{{{{ {value} | tojson }}}}");
        assert_eq!(
            render("tojson"),
            legacy.render_str(&source, context! {}).unwrap()
        );
        let scalar = environment
            .render_str(
                "{{ \"sandbox.javascript\" | tojson(sort_keys=true, separators=(\",\", \":\")) }}",
                context! {},
            )
            .unwrap();
        assert_eq!(scalar, "\"sandbox.javascript\"");
        for invalid in [
            "tojson(separators=(\",\",))",
            "tojson(separators=7)",
            "tojson(colour=1)",
        ] {
            let source = format!("{{{{ 1 | {invalid} }}}}");
            assert!(
                environment.render_str(&source, context! {}).is_err(),
                "{invalid}"
            );
        }
    }

    /// The tool loop's prompt against transformers' `apply_chat_template(..., tools=...)`:
    /// tool declarations, a call with parsed arguments and its result, byte for byte.
    #[test]
    #[ignore = "requires MLXL3_TOKENIZER_MODEL and local Python transformers; no download"]
    fn local_transformers_tool_prompt_matches() -> Result<()> {
        let model = std::env::var("MLXL3_TOKENIZER_MODEL").context("set MLXL3_TOKENIZER_MODEL")?;
        let tokenizer = ChatTokenizer::load(Path::new(&model))?;
        let tools = json!([{"type":"function","function":{
            "name":"sandbox.javascript","description":"Run JavaScript é \"q\"",
            "parameters":{"type":"object","properties":{"code":{"type":"string","maxLength":32768}},
                "required":["code"],"additionalProperties":false}}}]);
        let messages = json!([
            {"role":"system","content":"You are a local coding agent."},
            {"role":"user","content":"List the files 🦀."},
            {"role":"assistant","content":"Je liste.","tool_calls":[{"type":"function","function":{
                "name":"sandbox.javascript","arguments":{"code":"return list()"}}}]},
            {"role":"tool","name":"sandbox.javascript","content":"Result:\n[\"a.txt\"]"}
        ]);
        let output = std::process::Command::new(std::env::var("MLXL3_TEST_PYTHON").unwrap_or("python3".into()))
            .env("HF_HUB_OFFLINE", "1").env("TRANSFORMERS_OFFLINE", "1")
            .arg("-c").arg(r#"
import json, sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=False)
print(json.dumps(t.apply_chat_template(json.loads(sys.argv[2]), tools=json.loads(sys.argv[3]), tokenize=False, add_generation_prompt=True)))
"#).arg(&model).arg(messages.to_string()).arg(tools.to_string()).output()?;
        ensure!(
            output.status.success(),
            "transformers failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let reference: String = serde_json::from_slice(&output.stdout)?;
        let tools = tools.as_array().context("tools")?.clone();
        assert_eq!(tokenizer.render_values(&messages, Some(&tools))?, reference);
        Ok(())
    }

    /// The real Inkling 2 template with the engine's tool loop messages: tools declared, the
    /// call rendered in its native format from parsed arguments, the result under its name.
    #[test]
    #[ignore = "requires MLXL3_INKLING_MODEL (an Inkling 2 checkpoint folder); no GPU"]
    fn inkling_template_renders_the_tool_loop() -> Result<()> {
        let path = std::env::var("MLXL3_INKLING_MODEL").context("MLXL3_INKLING_MODEL")?;
        let tokenizer = ChatTokenizer::load(Path::new(&path))?;
        let tools = [serde_json::json!({"type":"function","function":{
            "name":"sandbox.javascript","description":"Run JavaScript",
            "parameters":{"type":"object","properties":{"code":{"type":"string"}},"required":["code"]}}})];
        let messages = serde_json::json!([
            {"role":"system","content":"You are a local coding agent."},
            {"role":"user","content":"List the files."},
            {"role":"assistant","content":"Je liste.","tool_calls":[{"type":"function","function":{
                "name":"sandbox.javascript","arguments":{"code":"return list()"}}}]},
            {"role":"tool","name":"sandbox.javascript","content":"Result:\n[\"a.txt\"]"}
        ]);
        let rendered = tokenizer.render_values(&messages, Some(&tools))?;
        assert!(rendered.contains("<|message_system|>tool_declare<|content_xml|>[{\"description\":\"Run JavaScript\",\"name\":\"sandbox.javascript\""), "{rendered}");
        assert!(rendered.contains("<|message_model|><|content_text|>Je liste.<|end_message|><|message_model|>sandbox.javascript<|content_invoke_tool_json|>{\"name\":\"sandbox.javascript\",\"args\":{\"code\":\"return list()\"}}<|end_message|><|content_model_end_sampling|>"), "{rendered}");
        assert!(rendered.contains("<|message_tool|>sandbox.javascript<|content_text|>Result:\n[\"a.txt\"]<|end_message|>"), "{rendered}");
        // What the model writes back is what the parser reads.
        let call = r#"sandbox.javascript<|content_invoke_tool_json|>{"name":"sandbox.javascript","args":{"code":"return list()"}}"#;
        let parsed = crate::tool_call::parse_answer(call)?;
        assert_eq!(parsed[0].name, "sandbox.javascript");
        assert_eq!(
            parsed[0].arguments,
            serde_json::json!({"code":"return list()"})
        );
        Ok(())
    }

    use super::*;
    use serde_json::json;

    fn fixture(template: &str) -> tempfile::TempDir {
        let root = tempfile::tempdir().unwrap();
        let tokenizer = json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": [
                {"id":0,"content":"<bos>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false},
                {"id":1,"content":"<eos>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}
            ],
            "normalizer": null, "pre_tokenizer": {"type":"WhitespaceSplit"},
            "post_processor": null, "decoder": null,
            "model": {"type":"WordLevel", "vocab":{"<bos>":0,"<eos>":1,"<unk>":2,"hello":3,"<think>":4},"unk_token":"<unk>"}
        });
        fs::write(root.path().join("tokenizer.json"), tokenizer.to_string()).unwrap();
        fs::write(
            root.path().join("tokenizer_config.json"),
            json!({
                "bos_token":{"content":"<bos>"},"eos_token":"<eos>","chat_template":template
            })
            .to_string(),
        )
        .unwrap();
        root
    }

    #[test]
    fn actual_tokenizer_distinguishes_literal_and_declared_marker_ids() {
        use crate::streaming::{Channel, ThinkingSplitter};
        // Qwen3 declares these added delimiters with special:false: the flag
        // controls skip_special_tokens, not their role in the chat protocol.
        for special in [false, true] {
            let root = fixture("{{ messages[0].content }}");
            let path = root.path().join("tokenizer.json");
            let mut json: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            json["decoder"] = json!({"type":"Fuse"});
            for (token, id) in [("<", 5), ("think", 6), (">", 7), ("</think>", 8)] {
                json["model"]["vocab"][token] = json!(id);
            }
            fs::write(&path, json.to_string()).unwrap();
            let plain = ChatTokenizer::load(root.path()).unwrap();
            assert_eq!(plain.control_marker(4), None);
            for (content, id) in [("<think>", 4), ("</think>", 8)] {
                json["added_tokens"].as_array_mut().unwrap().push(json!({
                    "id":id,"content":content,"special":special,"single_word":false,
                    "lstrip":false,"rstrip":false,"normalized":false
                }));
            }
            fs::write(path, json.to_string()).unwrap();
            let tokenizer = ChatTokenizer::load(root.path()).unwrap();
            assert_eq!(
                tokenizer.control_marker(4),
                Some("<think>"),
                "special={special}"
            );
            assert_eq!(tokenizer.control_marker(8), Some("</think>"));
            assert_eq!(tokenizer.decode(&[5, 6, 7]).unwrap(), "<think>");
            let mut decoder = tokenizer.tokenizer().decode_stream(false);
            let mut splitter = ThinkingSplitter::new("");
            let mut fragments = Vec::new();
            for id in [4, 3, 8, 3, 5, 6, 7] {
                if let Some(text) = decoder.step(id).unwrap() {
                    fragments.extend(splitter.feed_token(&text, tokenizer.control_marker(id)));
                }
            }
            fragments.extend(splitter.finish());
            let content = |channel| {
                fragments
                    .iter()
                    .filter(|f| f.channel == channel)
                    .map(|f| f.text.as_str())
                    .collect::<String>()
            };
            assert_eq!(content(Channel::Thinking), "hello");
            assert_eq!(content(Channel::Answer), "hello<think>");
        }
    }

    #[test]
    fn reasoning_effort_reaches_the_template_or_turns_thinking_off() {
        let root = fixture(
            "{{ enable_thinking is undefined }}|{{ enable_thinking }}|{{ reasoning_effort | default('unset') }}|{{ messages[0].content }}",
        );
        let tokenizer = ChatTokenizer::load(root.path()).unwrap();
        let messages = json!([{"role":"user","content":"hi"}]);
        let render = |effort| tokenizer.render_values_with_effort(&messages, None, effort);
        assert_eq!(render(None).unwrap(), "True||unset|hi", "template default");
        assert_eq!(render(Some("none")).unwrap(), "False|False|0.0|hi");
        for level in ["low", "medium", "high", "xhigh", "max"] {
            assert_eq!(
                render(Some(level)).unwrap(),
                format!("False|True|{level}|hi")
            );
        }
        for invalid in ["", "minimal", "HIGH", "0.5", "ultra"] {
            assert!(render(Some(invalid)).is_err(), "{invalid:?}");
        }
    }

    #[test]
    fn explicit_thinking_preserves_undefined_default_and_other_context() {
        let root = fixture(
            "{{ enable_thinking is undefined }}|{{ enable_thinking is true }}|{{ enable_thinking is false }}|{{ messages[0].content }}|{{ tools[0].name }}|{{ add_generation_prompt }}",
        );
        let tokenizer = ChatTokenizer::load(root.path()).unwrap();
        let messages = json!([{"role":"user","content":"hello"}]);
        let tools = [json!({"name":"tool"})];
        assert_eq!(
            tokenizer.render_values(&messages, Some(&tools)).unwrap(),
            "True|False|False|hello|tool|True"
        );
        assert_eq!(
            tokenizer
                .render_values_with_thinking(&messages, Some(&tools), true)
                .unwrap(),
            "False|True|False|hello|tool|True"
        );
        assert_eq!(
            tokenizer
                .render_values_with_thinking(&messages, Some(&tools), false)
                .unwrap(),
            "False|False|True|hello|tool|True"
        );
        assert!(
            tokenizer
                .render_values_with_thinking(&json!([]), None, false)
                .is_err()
        );
        assert!(
            tokenizer
                .render_values_with_thinking(&json!([{"role":"invalid","content":"x"}]), None, true)
                .is_err()
        );
    }

    #[test]
    fn generation_blocks_preserve_whitespace_and_python_methods() {
        let root = fixture(
            "{{ bos_token }} \n{%- generation -%}\n{{ messages[0].content.strip().upper() }}\n{%- endgeneration -%}\n{{ eos_token }}",
        );
        let tokenizer = ChatTokenizer::load(root.path()).unwrap();
        assert_eq!(
            tokenizer
                .render(&[Message {
                    role: "user".into(),
                    content: " hello ".into()
                }])
                .unwrap(),
            "<bos>HELLO<eos>"
        );
        assert_eq!(tokenizer.encode("hello").unwrap(), [3]);
        assert_eq!(tokenizer.decode(&[0, 4, 1]).unwrap(), "<bos> <think> <eos>");
        assert_eq!(tokenizer.eos_ids(), [1]);
        assert!(tokenizer.render(&[]).is_err());
        assert!(
            tokenizer
                .render(&[Message {
                    role: "invalid".into(),
                    content: "result".into()
                }])
                .is_err()
        );
    }

    #[test]
    fn renders_tools_and_tool_results() {
        let root = fixture(
            "{{ strftime_now('%Y-%m-%d') }}|{{ tools | tojson }}|{{ messages[1].role }}:{{ messages[1].content }}",
        );
        let tokenizer = ChatTokenizer::load(root.path()).unwrap();
        let messages = json!([
            {"role":"user","content":"search"},
            {"role":"tool","name":"exa.search","content":"result"}
        ]);
        let tools = vec![json!({"type":"function","function":{"name":"exa.search"}})];
        let rendered = tokenizer.render_values(&messages, Some(&tools)).unwrap();
        assert!(rendered.contains("exa.search"));
        assert!(rendered.ends_with("tool:result"));
    }

    #[test]
    fn literals_comments_and_raw_blocks_are_not_rewritten() {
        for template in [
            "{{ '{% generation %}' }}",
            "{# {% generation %} #}",
            "{% raw %}{% generation %}text{% endgeneration %}{% endraw %}",
            "{% set x = '%}' %}{{ x }}",
        ] {
            assert_eq!(inference_template(template), template);
        }
        assert_eq!(
            inference_template("a{% generation %} b {% endgeneration %}c"),
            "a{% if true %} b {% endif %}c"
        );
    }

    #[test]
    fn external_template_and_generation_eos_take_precedence() {
        let root = fixture("fallback");
        fs::write(root.path().join("chat_template.jinja"), "external").unwrap();
        fs::write(root.path().join("config.json"), r#"{"eos_token_id":0}"#).unwrap();
        fs::write(
            root.path().join("generation_config.json"),
            r#"{"eos_token_id":[1,4,1]}"#,
        )
        .unwrap();
        let tokenizer = ChatTokenizer::load(root.path()).unwrap();
        assert_eq!(tokenizer.eos_ids(), [1, 4]);
        assert_eq!(
            tokenizer
                .render(&[Message {
                    role: "user".into(),
                    content: "hello".into()
                }])
                .unwrap(),
            "external"
        );
        fs::write(
            root.path().join("generation_config.json"),
            r#"{"eos_token_id":999}"#,
        )
        .unwrap();
        assert!(ChatTokenizer::load(root.path()).is_err());
    }

    #[test]
    fn malformed_special_tokens_are_rejected() {
        let root = fixture("{{ bos_token }}");
        for token in [
            json!(23),
            json!({"content":false}),
            json!(""),
            json!("<missing>"),
        ] {
            fs::write(
                root.path().join("tokenizer_config.json"),
                json!({
                    "bos_token":token,"eos_token":"<eos>","chat_template":"{{ bos_token }}"
                })
                .to_string(),
            )
            .unwrap();
            assert!(ChatTokenizer::load(root.path()).is_err());
        }
    }

    #[test]
    #[ignore = "requires MLXL3_TOKENIZER_MODEL and local Python transformers; no download"]
    fn local_transformers_prompt_and_ids_match() -> Result<()> {
        let model = std::env::var("MLXL3_TOKENIZER_MODEL").context("set MLXL3_TOKENIZER_MODEL")?;
        let tokenizer = ChatTokenizer::load(Path::new(&model))?;
        let messages = vec![
            Message {
                role: "system".into(),
                content: "Réponds brièvement.".into(),
            },
            Message {
                role: "user".into(),
                content: "Salut 🦀 !".into(),
            },
            Message {
                role: "assistant".into(),
                content: "<think>Réfléchissons.</think>Bonjour !".into(),
            },
            Message {
                role: "user".into(),
                content: "Et 2 + 2 ?".into(),
            },
        ];
        let output = std::process::Command::new(std::env::var("MLXL3_TEST_PYTHON").unwrap_or("python3".into()))
            .env("HF_HUB_OFFLINE", "1").env("TRANSFORMERS_OFFLINE", "1")
            .arg("-c").arg(r#"
import json, sys
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True, trust_remote_code=False)
prompt = t.apply_chat_template(json.loads(sys.argv[2]), tokenize=False, add_generation_prompt=True, preserve_thinking=True)
ids = t.encode(prompt, add_special_tokens=False)
print(json.dumps({'prompt': prompt, 'ids': ids, 'decoded': t.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)}))
"#).arg(&model).arg(serde_json::to_string(&messages)?).output()?;
        ensure!(
            output.status.success(),
            "transformers failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let reference: Value = serde_json::from_slice(&output.stdout)?;
        let prompt = tokenizer.render(&messages)?;
        let ids = tokenizer.encode(&prompt)?;
        assert_eq!(prompt, reference["prompt"].as_str().unwrap());
        assert_eq!(json!(ids), reference["ids"]);
        assert_eq!(
            tokenizer.decode(&ids)?,
            reference["decoded"].as_str().unwrap()
        );
        // Check every stopping position, including partial byte-fallback emoji.
        let ids = tokenizer.encode("Voilà 🦀 !")?;
        let mut stream = tokenizer.tokenizer().decode_stream(false);
        let mut emitted = String::new();
        for (position, id) in ids.iter().enumerate() {
            if let Some(text) = stream.step(*id).map_err(|e| anyhow::anyhow!("{e}"))? {
                emitted.push_str(&text);
            }
            assert!(tokenizer.decode(&ids[..=position])?.starts_with(&emitted));
        }
        Ok(())
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(100)]
fn actual_flash_thinking_template_preserves_selected_mode() {
    let root = std::env::var("MLXL3_FLASH_MODEL").unwrap();
    let tokenizer = ChatTokenizer::load(Path::new(&root)).unwrap();
    let thinking: bool = kani::any();
    let messages = serde_json::json!([{"role":"user","content":"hello"}]);
    if let Ok(text) = tokenizer.render_values_with_thinking(&messages, None, thinking) {
        assert!(text.ends_with(if thinking {
            "<think>\n"
        } else {
            "<think>\n\n</think>\n\n"
        }));
        kani::cover!(thinking);
        kani::cover!(!thinking);
    }
}
