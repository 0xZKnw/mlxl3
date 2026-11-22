//! Safe parsers for the tool-call formats emitted by supported chat templates.
use anyhow::{Context, Result, ensure};
use serde_json::{Map, Value};

#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

const PYTHON_OPEN: &str = "<|tool_call_start|>";
const PYTHON_CLOSE: &str = "<|tool_call_end|>";
const GEMMA_OPEN: &str = "<|tool_call>";
const GEMMA_CLOSE: &str = "<tool_call|>";
const JSON_OPEN: &str = "<tool_call>";
const JSON_CLOSE: &str = "</tool_call>";
/// Inkling 2: `<|message_model|>NAME<|content_invoke_tool_json|>{"name":…,"args":{…}}<|end_message|>`.
/// The splitter removes the message structure, so the answer text holds `NAME<marker>{json}`.
const INKLING_INVOKE: &str = "<|content_invoke_tool_json|>";

pub fn parse(response: &str) -> Result<Vec<ToolCall>> {
    parse_answer(assistant_context(response))
}

/// Parse an already channel-separated answer, without lexical reasoning removal.
pub fn parse_answer(response: &str) -> Result<Vec<ToolCall>> {
    let response = response.trim();
    if response.contains(INKLING_INVOKE) {
        return parse_inkling_calls(response);
    }
    if response.starts_with(PYTHON_OPEN) {
        return parse_blocks(response, PYTHON_OPEN, PYTHON_CLOSE, parse_python_calls);
    }
    if response.starts_with(GEMMA_OPEN) {
        return parse_blocks(response, GEMMA_OPEN, GEMMA_CLOSE, parse_gemma_calls);
    }
    if response.starts_with(JSON_OPEN) {
        return parse_blocks(response, JSON_OPEN, JSON_CLOSE, parse_json_calls);
    }
    Ok(Vec::new())
}

pub fn without_calls(response: &str) -> String {
    let mut text = assistant_context(response).trim().to_owned();
    for (open, close) in [
        (PYTHON_OPEN, PYTHON_CLOSE),
        (GEMMA_OPEN, GEMMA_CLOSE),
        (JSON_OPEN, JSON_CLOSE),
    ] {
        while let Some(start) = text.find(open) {
            let Some(end) = text[start + open.len()..].find(close) else {
                break;
            };
            text.replace_range(start..start + open.len() + end + close.len(), "");
        }
    }
    text.trim().to_owned()
}

fn assistant_context(response: &str) -> &str {
    if let Some((_, answer)) = response.rsplit_once("<channel|>") {
        return answer;
    }
    if let Some((_, answer)) = response.rsplit_once("</think>") {
        return answer;
    }
    response
}

fn parse_blocks(
    mut input: &str,
    open: &str,
    close: &str,
    parser: fn(&str) -> Result<Vec<ToolCall>>,
) -> Result<Vec<ToolCall>> {
    let mut calls = Vec::new();
    while !input.trim().is_empty() {
        input = input.trim_start();
        let body = input
            .strip_prefix(open)
            .context("ordinary text mixed with a tool call; nothing executed")?;
        let end = body
            .find(close)
            .context("unfinished tool call; nothing executed")?;
        ensure!(end <= 65_536, "tool payload too large");
        calls.extend(parser(&body[..end])?);
        input = &body[end + close.len()..];
    }
    ensure!(!calls.is_empty(), "tool marker contained no calls");
    Ok(calls)
}

/// Characters of a public MCP tool name (`server.tool`, see `mcp::unique_name`).
fn tool_name_char(character: char) -> bool {
    character.is_ascii_alphanumeric() || "._-".contains(character)
}

/// Byte offset where the tool-name characters ending `text` start.
fn trailing_name_start(text: &str) -> usize {
    text.char_indices()
        .rev()
        .take_while(|(_, character)| tool_name_char(*character))
        .last()
        .map_or(text.len(), |(index, _)| index)
}

/// Inkling 2 calls, one or more, after optional text. That text is the model's own
/// `<|content_text|>` message (a separate message in this format), never part of a call;
/// text after a call is refused like in the other formats. The JSON `name` names the tool
/// (the header must end with it); a damaged JSON still runs when its `"args":{…}` is
/// readable, under the header name.
fn parse_inkling_calls(response: &str) -> Result<Vec<ToolCall>> {
    let mut calls = Vec::new();
    let mut rest = response;
    while let Some(position) = rest.find(INKLING_INVOKE) {
        let before = &rest[..position];
        let body = &rest[position + INKLING_INVOKE.len()..];
        let mut values = serde_json::Deserializer::from_str(body).into_iter::<Value>();
        let parsed = values.next();
        let (name, arguments, end) = match parsed {
            Some(Ok(Value::Object(mut call))) => {
                let end = values.byte_offset();
                let name = match call.remove("name") {
                    Some(Value::String(name)) => {
                        ensure!(
                            before.ends_with(&name),
                            "Inkling tool call name differs from its header"
                        );
                        name
                    }
                    None => before[trailing_name_start(before)..].to_owned(),
                    Some(_) => anyhow::bail!("Inkling tool call name must be a string"),
                };
                let arguments = call
                    .remove("args")
                    .unwrap_or_else(|| Value::Object(Map::new()));
                (name, arguments, end)
            }
            Some(Err(error)) if error.is_eof() => {
                anyhow::bail!("unfinished tool call; nothing executed")
            }
            None => anyhow::bail!("unfinished tool call; nothing executed"),
            _ => recover_inkling_arguments(before, body)?,
        };
        ensure!(end <= 65_536, "tool payload too large");
        ensure!(
            arguments.is_object(),
            "Inkling tool arguments must be an object"
        );
        ensure!(
            !name.is_empty() && name.len() <= 256 && name.chars().all(tool_name_char),
            "invalid Inkling tool name"
        );
        let prose = &before[..before.len() - name.len()];
        ensure!(
            calls.is_empty() || prose.trim().is_empty(),
            "ordinary text mixed with a tool call; nothing executed"
        );
        calls.push(ToolCall { name, arguments });
        rest = &body[end..];
    }
    ensure!(
        rest.trim().is_empty(),
        "ordinary text mixed with a tool call; nothing executed"
    );
    ensure!(!calls.is_empty(), "tool marker contained no calls");
    Ok(calls)
}

/// `{…"args":{…}}` whose other fields do not parse (seen: `{"name"""javascript","args":…}`).
fn recover_inkling_arguments(before: &str, body: &str) -> Result<(String, Value, usize)> {
    let malformed = || anyhow::anyhow!("malformed Inkling tool call; nothing executed");
    ensure!(body.trim_start().starts_with('{'), malformed());
    let key = body.find("\"args\"").ok_or_else(malformed)?;
    let after_key = &body[key + "\"args\"".len()..];
    let colon = after_key
        .trim_start()
        .strip_prefix(':')
        .ok_or_else(malformed)?;
    let start = body.len() - colon.len();
    let mut values = serde_json::Deserializer::from_str(colon).into_iter::<Value>();
    let arguments = match values.next() {
        Some(Ok(value)) => value,
        Some(Err(error)) if error.is_eof() => {
            anyhow::bail!("unfinished tool call; nothing executed")
        }
        _ => return Err(malformed()),
    };
    let after_value = start + values.byte_offset();
    let closing = body[after_value..].trim_start();
    ensure!(closing.starts_with('}'), malformed());
    let end = body.len() - closing.len() + 1;
    let name = before[trailing_name_start(before)..].to_owned();
    Ok((name, arguments, end))
}

fn parse_json_calls(body: &str) -> Result<Vec<ToolCall>> {
    if let Some(function) = body.trim().strip_prefix("<function=") {
        return parse_xml_call(function);
    }
    if !body.trim().starts_with('{') && !body.trim().starts_with('[') {
        return parse_ling_call(body);
    }
    let value: Value = serde_json::from_str(body.trim()).context("malformed JSON tool call")?;
    let values = value.as_array().cloned().unwrap_or_else(|| vec![value]);
    values.into_iter().map(tool_from_json).collect()
}

fn parse_ling_call(body: &str) -> Result<Vec<ToolCall>> {
    let body = body.trim();
    let name_end = body
        .find(|c: char| c.is_whitespace() || c == '<')
        .unwrap_or(body.len());
    let name = &body[..name_end];
    ensure!(
        !name.is_empty()
            && name.len() <= 256
            && name
                .bytes()
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == b'_')
            && name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"_.:-".contains(&c)),
        "invalid Ling tool name"
    );
    let mut remaining = body[name_end..].trim();
    let mut arguments = Map::new();
    while !remaining.is_empty() {
        let key_body = remaining
            .strip_prefix("<arg_key>")
            .context("malformed Ling tool argument")?;
        let (key, rest) = key_body
            .split_once("</arg_key>")
            .context("unfinished Ling tool argument key")?;
        let key = key.trim();
        ensure!(
            !key.is_empty() && !key.contains('<') && !key.contains('>'),
            "invalid Ling tool argument key"
        );
        let value_body = rest
            .trim_start()
            .strip_prefix("<arg_value>")
            .context("missing Ling tool argument value")?;
        let (raw, rest) = value_body
            .split_once("</arg_value>")
            .context("unfinished Ling tool argument value")?;
        let raw = raw.trim();
        let value = serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.into()));
        ensure!(
            arguments.insert(key.into(), value).is_none(),
            "duplicate Ling tool argument"
        );
        remaining = rest.trim();
    }
    Ok(vec![ToolCall {
        name: name.into(),
        arguments: Value::Object(arguments),
    }])
}

fn tool_from_json(value: Value) -> Result<ToolCall> {
    let function = value.get("function").unwrap_or(&value);
    let name = function
        .get("name")
        .and_then(Value::as_str)
        .context("tool call has no function name")?;
    let mut arguments = function
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    if let Some(encoded) = arguments.as_str() {
        arguments = serde_json::from_str(encoded).context("malformed encoded tool arguments")?;
    }
    ensure!(arguments.is_object(), "tool arguments must be an object");
    Ok(ToolCall {
        name: name.into(),
        arguments,
    })
}

fn parse_xml_call(body: &str) -> Result<Vec<ToolCall>> {
    let (name, rest) = body
        .split_once('>')
        .context("malformed function tool call")?;
    let content = rest
        .strip_suffix("</function>")
        .context("unfinished function tool call")?;
    let mut arguments = Map::new();
    let mut remaining = content.trim();
    while !remaining.is_empty() {
        let parameter = remaining
            .strip_prefix("<parameter=")
            .context("malformed function parameter")?;
        let (key, value) = parameter
            .split_once('>')
            .context("malformed function parameter")?;
        let end = value
            .find("</parameter>")
            .context("unfinished function parameter")?;
        let raw = value[..end].trim();
        arguments.insert(
            key.trim().into(),
            serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.into())),
        );
        remaining = value[end + "</parameter>".len()..].trim();
    }
    Ok(vec![ToolCall {
        name: name.trim().into(),
        arguments: Value::Object(arguments),
    }])
}

fn parse_python_calls(body: &str) -> Result<Vec<ToolCall>> {
    let mut parser = LiteralParser::new(body);
    parser.expect('[')?;
    let mut calls = Vec::new();
    loop {
        parser.space();
        if parser.consume(']') {
            break;
        }
        let name = parser.identifier(true)?;
        parser.expect('(')?;
        let mut arguments = Map::new();
        loop {
            parser.space();
            if parser.consume(')') {
                break;
            }
            let key = parser.identifier(false)?;
            parser.expect('=')?;
            ensure!(
                arguments.insert(key, parser.value(0)?).is_none(),
                "duplicate tool argument"
            );
            parser.space();
            if parser.consume(')') {
                break;
            }
            parser.expect(',')?;
        }
        calls.push(ToolCall {
            name,
            arguments: Value::Object(arguments),
        });
        parser.space();
        if parser.consume(']') {
            break;
        }
        parser.expect(',')?;
    }
    parser.space();
    ensure!(
        parser.done(),
        "unexpected text after Python-style tool calls"
    );
    ensure!(!calls.is_empty(), "expected a list of calls");
    Ok(calls)
}

fn parse_gemma_calls(body: &str) -> Result<Vec<ToolCall>> {
    let mut input = body.trim();
    let mut calls = Vec::new();
    while !input.is_empty() {
        input = input.trim_start_matches([' ', '\r', '\n', '\t', ',']);
        let rest = input
            .strip_prefix("call:")
            .context("malformed Gemma tool call")?;
        let name_end = rest.find('{').context("Gemma tool call has no arguments")?;
        let name = rest[..name_end].trim();
        ensure!(
            !name.is_empty()
                && name
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c)),
            "invalid Gemma tool name"
        );
        let mut parser = LiteralParser::new(&rest[name_end..]);
        let arguments = parser.value(0)?;
        ensure!(
            arguments.is_object(),
            "Gemma tool arguments must be an object"
        );
        let consumed = rest[name_end..].len() - parser.rest().len();
        calls.push(ToolCall {
            name: name.into(),
            arguments,
        });
        input = &rest[name_end + consumed..];
    }
    ensure!(!calls.is_empty(), "Gemma marker contained no calls");
    Ok(calls)
}

struct LiteralParser<'a> {
    input: &'a str,
    position: usize,
}

impl<'a> LiteralParser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, position: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.input[self.position..]
    }

    fn done(&self) -> bool {
        self.position == self.input.len()
    }

    fn space(&mut self) {
        while self.peek().is_some_and(char::is_whitespace) {
            self.bump();
        }
    }

    fn peek(&self) -> Option<char> {
        self.rest().chars().next()
    }

    fn bump(&mut self) -> Option<char> {
        let next = self.peek()?;
        self.position += next.len_utf8();
        Some(next)
    }

    fn consume(&mut self, expected: char) -> bool {
        self.space();
        if self.peek() == Some(expected) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, expected: char) -> Result<()> {
        ensure!(self.consume(expected), "expected {expected:?} in tool call");
        Ok(())
    }

    fn identifier(&mut self, dotted: bool) -> Result<String> {
        self.space();
        let start = self.position;
        while self.peek().is_some_and(|c| {
            c.is_ascii_alphanumeric() || c == '_' || c == '-' || (dotted && c == '.')
        }) {
            self.bump();
        }
        ensure!(self.position > start, "expected identifier in tool call");
        Ok(self.input[start..self.position].into())
    }

    fn value(&mut self, depth: usize) -> Result<Value> {
        ensure!(depth < 32, "tool arguments are nested too deeply");
        self.space();
        if self.rest().starts_with("<|\"|>") {
            self.position += "<|\"|>".len();
            let end = self
                .rest()
                .find("<|\"|>")
                .context("unfinished Gemma string")?;
            let value = self.rest()[..end].to_owned();
            self.position += end + "<|\"|>".len();
            return Ok(Value::String(value));
        }
        match self.peek().context("missing tool argument value")? {
            '\'' | '"' => self.string().map(Value::String),
            '[' => self.array(']', depth),
            '(' => self.array(')', depth),
            '{' => self.object(depth),
            _ => self.atom(),
        }
    }

    fn string(&mut self) -> Result<String> {
        let quote = self.bump().expect("caller checked quote");
        let mut output = String::new();
        loop {
            let character = self.bump().context("unfinished tool string")?;
            match character {
                c if c == quote => return Ok(output),
                '\\' => {
                    let escaped = self.bump().context("unfinished string escape")?;
                    output.push(match escaped {
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        '\\' => '\\',
                        '\'' => '\'',
                        '"' => '"',
                        other => other,
                    });
                }
                other => output.push(other),
            }
        }
    }

    fn array(&mut self, close: char, depth: usize) -> Result<Value> {
        self.bump();
        let mut values = Vec::new();
        loop {
            self.space();
            if self.consume(close) {
                break;
            }
            values.push(self.value(depth + 1)?);
            self.space();
            if self.consume(close) {
                break;
            }
            self.expect(',')?;
        }
        Ok(Value::Array(values))
    }

    fn object(&mut self, depth: usize) -> Result<Value> {
        self.bump();
        let mut values = Map::new();
        loop {
            self.space();
            if self.consume('}') {
                break;
            }
            let key = if matches!(self.peek(), Some('\'' | '"')) {
                self.string()?
            } else {
                self.identifier(false)?
            };
            self.expect(':')?;
            ensure!(
                values.insert(key, self.value(depth + 1)?).is_none(),
                "duplicate object key"
            );
            self.space();
            if self.consume('}') {
                break;
            }
            self.expect(',')?;
        }
        Ok(Value::Object(values))
    }

    fn atom(&mut self) -> Result<Value> {
        let start = self.position;
        while self
            .peek()
            .is_some_and(|c| !c.is_whitespace() && !",]}):".contains(c))
        {
            self.bump();
        }
        let atom = &self.input[start..self.position];
        match atom {
            "True" | "true" => Ok(Value::Bool(true)),
            "False" | "false" => Ok(Value::Bool(false)),
            "None" | "null" => Ok(Value::Null),
            _ => {
                serde_json::from_str(atom).with_context(|| format!("invalid tool literal {atom:?}"))
            }
        }
    }
}

pub struct StreamFilter {
    buffer: String,
    closer: &'static str,
    inside: bool,
    passthrough: bool,
    /// Inkling 2 format: a call may follow the model's text; from it on, nothing is shown.
    inkling: bool,
}

impl Default for StreamFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamFilter {
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            closer: JSON_CLOSE,
            inside: false,
            passthrough: false,
            inkling: false,
        }
    }

    /// Filter for prompts that declare tools in the Inkling 2 format.
    pub fn inkling() -> Self {
        Self {
            inkling: true,
            ..Self::new()
        }
    }

    /// Inkling: show text up to `NAME<|content_invoke_tool_json|>`, holding back a trailing
    /// name (or a partial marker) until the next text tells whether a call starts there.
    fn feed_inkling(&mut self, visible: &mut Vec<String>) {
        if self.inside {
            self.buffer.clear();
            return;
        }
        if let Some(position) = self.buffer.find(INKLING_INVOKE) {
            let before = &self.buffer[..position];
            let shown = before[..trailing_name_start(before)].to_owned();
            if !shown.is_empty() {
                visible.push(shown);
            }
            self.buffer.clear();
            self.inside = true;
            return;
        }
        let partial = (1..INKLING_INVOKE.len())
            .rev()
            .find(|&length| self.buffer.ends_with(&INKLING_INVOKE[..length]))
            .unwrap_or(0);
        let held = trailing_name_start(&self.buffer[..self.buffer.len() - partial]);
        if held > 0 {
            visible.push(self.buffer[..held].to_owned());
            self.buffer.drain(..held);
        }
    }

    pub fn feed(&mut self, text: &str) -> Vec<String> {
        self.buffer.push_str(text);
        let mut visible = Vec::new();
        if self.inkling {
            self.feed_inkling(&mut visible);
            return visible;
        }
        loop {
            if self.buffer.is_empty() {
                break;
            }
            if self.passthrough {
                visible.push(std::mem::take(&mut self.buffer));
                break;
            }
            if self.inside {
                if let Some(position) = self.buffer.find(self.closer) {
                    self.buffer.drain(..position + self.closer.len());
                    self.inside = false;
                } else {
                    retain_marker_suffix(&mut self.buffer, self.closer);
                    break;
                }
                continue;
            }
            let trimmed = self.buffer.trim_start();
            let leading = self.buffer.len() - trimmed.len();
            let marker = [
                (JSON_OPEN, JSON_CLOSE),
                (GEMMA_OPEN, GEMMA_CLOSE),
                (PYTHON_OPEN, PYTHON_CLOSE),
            ]
            .into_iter()
            .find(|(open, _)| trimmed.starts_with(open));
            if let Some((open, close)) = marker {
                self.buffer.drain(..leading + open.len());
                self.closer = close;
                self.inside = true;
            } else if trimmed.is_empty()
                || [JSON_OPEN, GEMMA_OPEN, PYTHON_OPEN]
                    .iter()
                    .any(|marker| marker.starts_with(trimmed))
            {
                break;
            } else {
                self.passthrough = true;
            }
        }
        visible
    }

    pub fn is_preparing_tool(&self) -> bool {
        self.inside
    }

    pub fn finish(&mut self) -> Vec<String> {
        let visible = if self.inside || self.buffer.is_empty() {
            Vec::new()
        } else {
            vec![std::mem::take(&mut self.buffer)]
        };
        self.buffer.clear();
        visible
    }
}

fn retain_marker_suffix(text: &mut String, marker: &str) {
    let keep = marker_suffix_start(text.as_bytes(), marker.as_bytes());
    text.drain(..keep);
}

fn marker_suffix_start(text: &[u8], marker: &[u8]) -> usize {
    (0..=text.len())
        .find(|&index| marker.starts_with(&text[index..]))
        .unwrap_or(text.len())
}

/// Tool rounds a chat request may run before its final answer (older clients: 5).
pub const DEFAULT_TOOL_ROUNDS: usize = 5;
/// Upper bound a request may ask for (the Agent mode asks for more than the chat).
pub const MAX_TOOL_ROUNDS: usize = 64;
/// Largest `tool_start` arguments shown as JSON; longer ones become a text prefix.
pub const SHOWN_ARGUMENTS_BYTES: usize = 16 * 1024;

/// The requested round limit, validated; omitted means [`DEFAULT_TOOL_ROUNDS`].
pub fn tool_rounds(requested: Option<usize>) -> Result<usize> {
    let rounds = requested.unwrap_or(DEFAULT_TOOL_ROUNDS);
    ensure!(
        (1..=MAX_TOOL_ROUNDS).contains(&rounds),
        "tool_rounds must be 1 to {MAX_TOOL_ROUNDS}"
    );
    Ok(rounds)
}

/// Appended to the last tool result before the final round: the model answers instead of
/// the run ending on an error.
pub fn round_limit_note(rounds: usize) -> String {
    format!(
        "[Tool-round limit of {rounds} reached: answer now from these results, without calling a tool.]"
    )
}

/// What the model reads when its tool call could not be parsed; nothing was executed.
pub fn unreadable_call_message(error: &anyhow::Error) -> String {
    format!(
        "Your tool call could not be read ({error}); nothing was executed. Send one complete tool call and nothing else, or answer without tools."
    )
}

/// Token ranges kept from a tool result longer than `budget`: the first quarter and the
/// rest from its end (errors and summaries come last), `reserve` tokens left for a marker.
pub fn kept_token_ranges(
    len: usize,
    budget: usize,
    reserve: usize,
) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
    if len <= budget {
        return (0..len, len..len);
    }
    let room = budget.saturating_sub(reserve).max(2);
    let head = room / 4;
    let tail = room - head;
    (0..head, len - tail..len)
}

/// Arguments shown with `tool_start`: the value itself, or the first
/// [`SHOWN_ARGUMENTS_BYTES`] of its JSON text (on a character boundary).
pub fn shown_arguments(arguments: &Value) -> Value {
    let text = arguments.to_string();
    if text.len() <= SHOWN_ARGUMENTS_BYTES {
        return arguments.clone();
    }
    let mut end = SHOWN_ARGUMENTS_BYTES;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    Value::String(format!("{}…", &text[..end]))
}

#[cfg(test)]
mod tests {
    #[test]
    fn parses_inkling_tool_calls() {
        let m = super::INKLING_INVOKE;
        let call = |name: &str, args: serde_json::Value| ToolCall {
            name: name.into(),
            arguments: args,
        };
        let single = format!(
            r#"sandbox.javascript{m}{{"name":"sandbox.javascript","args":{{"code":"return list()"}}}}"#
        );
        assert_eq!(
            parse_answer(&single).unwrap(),
            [call(
                "sandbox.javascript",
                serde_json::json!({"code": "return list()"})
            )]
        );
        // The model's own message first, then the call (a paragraph apart, see the splitter).
        let prose = format!("Je liste les fichiers.\n\n{single}");
        assert_eq!(parse_answer(&prose).unwrap().len(), 1);
        let glued = format!("the files{single}");
        assert_eq!(parse_answer(&glued).unwrap()[0].name, "sandbox.javascript");
        let two =
            format!(r#"a.b{m}{{"name":"a.b","args":{{}}}}c.d{m}{{"name":"c.d","args":{{"x":1}}}}"#);
        assert_eq!(
            parse_answer(&two).unwrap(),
            [
                call("a.b", serde_json::json!({})),
                call("c.d", serde_json::json!({"x": 1}))
            ]
        );
        // The user's screenshot: the name field is damaged, `args` is readable.
        let damaged = format!(
            r#"javascript{m}{{"name"""javascript","args":{{"code":"return await list();"}}}}"#
        );
        assert_eq!(
            parse_answer(&damaged).unwrap(),
            [call(
                "javascript",
                serde_json::json!({"code": "return await list();"})
            )]
        );
        let nameless = format!(r#"x.y{m}{{"args":{{"q":"z"}}}}"#);
        assert_eq!(
            parse_answer(&nameless).unwrap(),
            [call("x.y", serde_json::json!({"q": "z"}))]
        );
        let argless = format!(r#"x{m}{{"name":"x"}}"#);
        assert_eq!(
            parse_answer(&argless).unwrap(),
            [call("x", serde_json::json!({}))]
        );
        // A marker inside the code is data, not a second call.
        let nested = format!(r#"x{m}{{"name":"x","args":{{"code":"'{m}'"}}}}"#);
        assert_eq!(
            parse_answer(&nested).unwrap()[0].arguments["code"],
            format!("'{m}'")
        );
        for (input, error) in [
            (format!("{single} done"), "ordinary text"),
            (
                format!(
                    r#"a{m}{{"name":"a","args":{{}}}}text between b{m}{{"name":"b","args":{{}}}}"#
                ),
                "ordinary text",
            ),
            (
                format!(r#"javascript{m}{{"name":"javascript","args":{{"code":"ret"#),
                "unfinished",
            ),
            (format!("javascript{m}"), "unfinished"),
            (
                format!(r#"other{m}{{"name":"javascript","args":{{}}}}"#),
                "differs from its header",
            ),
            (
                format!(r#"x{m}{{"name":"x","args":"code"}}"#),
                "must be an object",
            ),
            (
                format!(r#"x{m}{{"name":7,"args":{{}}}}"#),
                "must be a string",
            ),
            (
                format!(r#"{m}{{"args":{{}}}}"#),
                "invalid Inkling tool name",
            ),
            (format!("x{m}not json"), "malformed"),
            (
                format!(
                    r#"x{m}{{"name":"x","args":{{"code":"{}"}}}}"#,
                    "a".repeat(70_000)
                ),
                "too large",
            ),
        ] {
            let message = parse_answer(&input).unwrap_err().to_string();
            assert!(message.contains(error), "{input:.80}: {message}");
        }
    }

    #[test]
    fn inkling_filter_hides_calls_at_every_chunk_boundary() {
        let m = super::INKLING_INVOKE;
        for (input, shown) in [
            (
                format!(
                    r#"Je liste.{}sandbox.javascript{m}{{"name":"sandbox.javascript","args":{{}}}}"#,
                    "\n\n"
                ),
                "Je liste.\n\n",
            ),
            (
                format!(r#"javascript{m}{{"name"""javascript","args":{{}}}}"#),
                "",
            ),
            (
                "Réponse sans outil, javascript compris.".to_owned(),
                "Réponse sans outil, javascript compris.",
            ),
            (format!("x{}", &m[..10]), "x<|content_"),
        ] {
            let boundaries: Vec<usize> = input
                .char_indices()
                .map(|(index, _)| index)
                .chain([input.len()])
                .collect();
            for &split in &boundaries {
                let mut filter = StreamFilter::inkling();
                let mut visible = filter.feed(&input[..split]).concat();
                visible += &filter.feed(&input[split..]).concat();
                visible += &filter.finish().concat();
                assert_eq!(visible, shown, "{input:?} split at {split}");
            }
            let mut filter = StreamFilter::inkling();
            let mut visible = String::new();
            for character in input.chars() {
                visible += &filter.feed(&character.to_string()).concat();
            }
            visible += &filter.finish().concat();
            assert_eq!(visible, shown, "{input:?} one character at a time");
        }
        let mut filter = StreamFilter::inkling();
        filter.feed(&format!("x{m}{{"));
        assert!(
            filter.is_preparing_tool(),
            "a started call reports its preparation"
        );
    }

    #[test]
    fn tool_round_limit_is_optional_and_bounded() {
        assert_eq!(super::tool_rounds(None).unwrap(), 5);
        assert_eq!(super::tool_rounds(Some(1)).unwrap(), 1);
        assert_eq!(super::tool_rounds(Some(64)).unwrap(), 64);
        assert!(super::tool_rounds(Some(0)).is_err());
        assert!(super::tool_rounds(Some(65)).is_err());
        assert!(super::round_limit_note(24).contains("limit of 24 reached"));
        let message = super::unreadable_call_message(&anyhow::anyhow!("unfinished tool call"));
        assert!(
            message.contains("(unfinished tool call)") && message.contains("nothing was executed")
        );
    }

    #[test]
    fn long_tool_output_keeps_its_start_and_end() {
        assert_eq!(super::kept_token_ranges(100, 100, 32), (0..100, 100..100));
        let (head, tail) = super::kept_token_ranges(10_000, 8192, 32);
        assert_eq!(head, 0..2040);
        assert_eq!(tail, 3880..10_000);
        assert_eq!(head.len() + tail.len() + 32, 8192);
        // Every length and budget: in bounds, ordered, disjoint, within the budget.
        for len in 0..300 {
            for budget in 1..120 {
                let (head, tail) = super::kept_token_ranges(len, budget, 16);
                assert!(
                    head.start == 0 && head.end <= tail.start && tail.end == len,
                    "{len} {budget}"
                );
                if len > budget {
                    assert!(
                        head.len() + tail.len() <= budget.max(2) && !tail.is_empty(),
                        "{len} {budget}"
                    );
                } else {
                    assert_eq!((head.len(), tail.len()), (len, 0));
                }
            }
        }
    }

    #[test]
    fn tool_start_arguments_are_bounded() {
        let small = serde_json::json!({"code": "return 1"});
        assert_eq!(super::shown_arguments(&small), small);
        let code = "é".repeat(20_000);
        let shown = super::shown_arguments(&serde_json::json!({ "code": code }));
        let text = shown.as_str().unwrap();
        assert!(text.starts_with(r#"{"code":"éé"#) && text.ends_with('…'));
        assert!(text.len() <= super::SHOWN_ARGUMENTS_BYTES + '…'.len_utf8());
    }

    use super::*;
    use serde_json::json;

    #[test]
    fn tool_preparation_is_visible_for_every_marker_chunk_boundary() {
        for (open, close) in [
            (JSON_OPEN, JSON_CLOSE),
            (GEMMA_OPEN, GEMMA_CLOSE),
            (PYTHON_OPEN, PYTHON_CLOSE),
        ] {
            for split in 0..=open.len() {
                let mut filter = StreamFilter::new();
                assert!(!filter.is_preparing_tool());
                filter.feed(&open[..split]);
                assert_eq!(filter.is_preparing_tool(), split == open.len());
                filter.feed(&open[split..]);
                assert!(filter.is_preparing_tool());
                assert!(filter.feed("file contents\n").is_empty());
                assert!(filter.is_preparing_tool());
                filter.feed(close);
                assert!(!filter.is_preparing_tool());
            }
        }
        let mut filter = StreamFilter::new();
        assert_eq!(filter.feed("ordinary answer"), ["ordinary answer"]);
        assert!(!filter.is_preparing_tool());
    }

    #[test]
    fn parses_supported_calls_without_executing_prose() {
        assert_eq!(
            parse("<think>x</think><|tool_call_start|>[exa.search(query='Rust', n=3, flags=[True, None])]<|tool_call_end|>").unwrap(),
            [ToolCall { name: "exa.search".into(), arguments: json!({"query":"Rust","n":3,"flags":[true,null]}) }]
        );
        assert_eq!(
            parse("<|tool_call>call:search{query:<|\"|>M5 Metal<|\"|>,n:5}<tool_call|>").unwrap()
                [0]
            .arguments,
            json!({"query":"M5 Metal","n":5})
        );
        assert!(
            parse("prose <tool_call>{\"name\":\"danger\"}</tool_call>")
                .unwrap()
                .is_empty()
        );

        let mut filter = StreamFilter::new();
        assert!(filter.feed("<|tool_call_sta").is_empty());
        assert!(filter.feed("rt|>[search(q='x')]").is_empty());
        assert!(filter.feed("<|tool_call_end|>").is_empty());
        assert!(filter.finish().is_empty());
    }

    #[test]
    fn javascript_tool_code_reaches_the_tool_intact() {
        // Agent mode's single `javascript` tool: multi-line code with comparisons, quotes,
        // backticks and markup-like text must arrive unchanged in every supported format.
        let code = "const files = list();\nif (files.length < 3 && files[0] > \"a\") {\n  console.log(`${files.length} <files> & 'x'`);\n}\nreturn read('src/a.js');";
        let qwen = format!(
            "<think>plan</think>\n\n<tool_call>\n<function=sandbox.javascript>\n<parameter=code>\n{code}\n</parameter>\n</function>\n</tool_call>"
        );
        assert_eq!(
            parse(&qwen).unwrap(),
            [ToolCall {
                name: "sandbox.javascript".into(),
                arguments: json!({ "code": code })
            }]
        );
        let hermes = format!(
            "<tool_call>{}</tool_call>",
            json!({"name": "sandbox.javascript", "arguments": {"code": code}})
        );
        assert_eq!(parse(&hermes).unwrap()[0].arguments["code"], code);
        let gemma =
            format!("<|tool_call>call:sandbox.javascript{{code:<|\"|>{code}<|\"|>}}<tool_call|>");
        assert_eq!(parse(&gemma).unwrap()[0].arguments["code"], code);
        // XML values that are valid JSON are decoded; the sandbox turns them back into text.
        let numeric = parse(
            "<tool_call>\n<function=sandbox.javascript>\n<parameter=code>\n42\n</parameter>\n</function>\n</tool_call>",
        )
        .unwrap();
        assert_eq!(numeric[0].arguments, json!({ "code": 42 }));
    }

    #[test]
    fn parses_ling_calls_and_rejects_malformed_arguments() {
        let response = "<think>searching</think><tool_call>exa.web_search_exa\n<arg_key>query</arg_key>\n<arg_value>GPT-6 Astra statistics</arg_value><arg_key>numResults</arg_key>\n<arg_value>10</arg_value>\n</tool_call>";
        assert_eq!(
            parse(response).unwrap(),
            [ToolCall {
                name: "exa.web_search_exa".into(),
                arguments: json!({"query":"GPT-6 Astra statistics","numResults":10}),
            }]
        );
        assert_eq!(without_calls(response), "");
        assert!(parse("<tool_call>exa.search<arg_key>q</arg_key><arg_value>x</arg_value><arg_key>q</arg_key><arg_value>y</arg_value></tool_call>").is_err());
        assert!(parse("<tool_call>exa.search<arg_key>q</arg_key></tool_call>").is_err());
        assert!(
            parse("prose <tool_call>exa.search</tool_call>")
                .unwrap()
                .is_empty()
        );

        let mut filter = StreamFilter::new();
        assert!(filter.feed("<tool_ca").is_empty());
        assert!(filter.feed("ll>exa.search<arg_key>q</arg_key>").is_empty());
        assert!(
            filter
                .feed("<arg_value>x</arg_value></tool_call>")
                .is_empty()
        );
        assert!(filter.finish().is_empty());
    }

    #[test]
    fn tool_call_filter_is_independent_of_chunk_boundary() {
        for input in [
            "<tool_call>{}</tool_call>",
            "<|tool_call>call:x{}<tool_call|>",
            "<|tool_call_start|>[x()]<|tool_call_end|>",
        ] {
            for split in 0..=input.len() {
                let mut filter = StreamFilter::new();
                assert!(filter.feed(&input[..split]).is_empty());
                assert!(filter.feed(&input[split..]).is_empty());
                assert!(filter.finish().is_empty());
            }
        }
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn tool_preparation_flag_matches_hidden_state() {
        let inside: bool = kani::any();
        let passthrough: bool = kani::any();
        let filter = StreamFilter {
            inside,
            passthrough,
            ..StreamFilter::new()
        };
        assert_eq!(filter.is_preparing_tool(), inside);
        kani::cover!(inside);
        kani::cover!(!inside);
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn marker_suffix_is_the_longest_possible_prefix() {
        const MARKER: &[u8] = b"</tool>";
        let bytes: [u8; 8] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= bytes.len());
        let text = &bytes[..len];
        let keep = marker_suffix_start(text, MARKER);
        assert!(keep <= text.len());
        assert!(MARKER.starts_with(&text[keep..]));
        for earlier in 0..keep {
            assert!(!MARKER.starts_with(&text[earlier..]));
        }
        kani::cover!(keep == 0);
        kani::cover!(keep == text.len());
    }
}
