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

pub fn parse(response: &str) -> Result<Vec<ToolCall>> {
    let response = assistant_context(response).trim();
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

fn parse_json_calls(body: &str) -> Result<Vec<ToolCall>> {
    if let Some(function) = body.trim().strip_prefix("<function=") {
        return parse_xml_call(function);
    }
    let value: Value = serde_json::from_str(body.trim()).context("malformed JSON tool call")?;
    let values = value.as_array().cloned().unwrap_or_else(|| vec![value]);
    values.into_iter().map(tool_from_json).collect()
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
        }
    }

    pub fn feed(&mut self, text: &str) -> Vec<String> {
        self.buffer.push_str(text);
        let mut visible = Vec::new();
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
    let keep = text
        .char_indices()
        .map(|(index, _)| index)
        .chain(std::iter::once(text.len()))
        .find(|&index| marker.starts_with(&text[index..]))
        .unwrap_or(text.len());
    text.drain(..keep);
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

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
}
