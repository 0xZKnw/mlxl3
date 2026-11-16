//! Bridge protocol v1 as seen from the app (requests out, events in).

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

pub const BRIDGE_PROTOCOL: u64 = 1;

/// One event line from the engine. The payload is kept whole so the UI sees
/// every field the worker reports (statistics, device, versions).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Event {
    #[serde(rename = "type")]
    pub kind: String,
    pub request_id: Option<String>,
    pub payload: Value,
}

impl Event {
    pub fn is_terminal_for_request(&self) -> bool {
        matches!(self.kind.as_str(), "done" | "cancelled" | "error")
            && self.request_id.as_deref().is_some_and(|id| !id.is_empty())
    }

    pub fn is_fatal(&self) -> bool {
        self.kind == "error" && self.payload.get("fatal").and_then(Value::as_bool) == Some(true)
    }
}

pub fn parse_event(line: &[u8]) -> Result<Event> {
    let text = std::str::from_utf8(line).context("engine event is not UTF-8")?;
    let payload: Value = serde_json::from_str(text).context("engine event is not JSON")?;
    let object = payload
        .as_object()
        .context("engine event is not a JSON object")?;
    let kind = object
        .get("type")
        .and_then(Value::as_str)
        .context("engine event has no string type")?
        .to_owned();
    let request_id = match object.get("request_id") {
        None | Some(Value::Null) => None,
        Some(Value::String(id)) => Some(id.clone()),
        Some(_) => bail!("engine event request_id is not a string"),
    };
    if kind == "ready" {
        let protocol = object.get("bridge_protocol").and_then(Value::as_u64);
        ensure!(
            protocol == Some(BRIDGE_PROTOCOL),
            "engine speaks bridge protocol {protocol:?}, the app expects {BRIDGE_PROTOCOL}"
        );
    }
    Ok(Event {
        kind,
        request_id,
        payload,
    })
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// User-tunable sampling settings. Validated here so a bad value fails in the
/// UI before reaching the engine (the worker validates again).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sampling {
    pub max_tokens: i64,
    pub temperature: f64,
    pub top_k: u32,
    pub top_p: f64,
    pub min_p: f64,
    pub repetition_penalty: f64,
    pub enable_thinking: Option<bool>,
}

impl Default for Sampling {
    fn default() -> Self {
        Self {
            max_tokens: -1,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            enable_thinking: None,
        }
    }
}

impl Sampling {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.max_tokens == -1 || self.max_tokens > 0,
            "max tokens must be -1 or positive"
        );
        ensure!(
            (0.0..=5.0).contains(&self.temperature),
            "temperature must be between 0 and 5"
        );
        ensure!(
            self.top_p > 0.0 && self.top_p <= 1.0,
            "top-p must be in (0, 1]"
        );
        ensure!((0.0..1.0).contains(&self.min_p), "min-p must be in [0, 1)");
        ensure!(
            (0.5..=3.0).contains(&self.repetition_penalty),
            "repetition penalty must be between 0.5 and 3"
        );
        Ok(())
    }
}

pub fn generate_request(
    request_id: &str,
    messages: &[Message],
    sampling: &Sampling,
) -> Result<Vec<u8>> {
    ensure!(!request_id.is_empty(), "request id is empty");
    ensure!(!messages.is_empty(), "conversation is empty");
    ensure!(
        messages.last().is_some_and(|m| m.role != "assistant"),
        "the last message must not be from the assistant"
    );
    sampling.validate()?;
    let mut value = serde_json::to_value(sampling)?;
    let object = value
        .as_object_mut()
        .expect("sampling serializes to an object");
    if sampling.enable_thinking.is_none() {
        object.remove("enable_thinking");
    }
    object.insert("type".into(), "generate".into());
    object.insert("request_id".into(), request_id.into());
    object.insert("messages".into(), serde_json::to_value(messages)?);
    line(&value)
}

pub fn control_request(kind: &str, request_id: &str) -> Result<Vec<u8>> {
    ensure!(
        matches!(kind, "ping" | "cancel" | "shutdown"),
        "unknown control request {kind}"
    );
    line(&serde_json::json!({"type": kind, "request_id": request_id}))
}

fn line(value: &Value) -> Result<Vec<u8>> {
    let mut bytes = serde_json::to_vec(value)?;
    debug_assert!(!bytes.contains(&b'\n'), "serde_json escapes newlines");
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_events() {
        let event = parse_event(br#"{"type":"delta","request_id":"r","text":"a\nb"}"#).unwrap();
        assert_eq!(
            (event.kind.as_str(), event.request_id.as_deref()),
            ("delta", Some("r"))
        );
        assert!(!event.is_terminal_for_request());
        assert!(
            parse_event(br#"{"type":"done","request_id":"r"}"#)
                .unwrap()
                .is_terminal_for_request()
        );
        assert!(
            !parse_event(br#"{"type":"error","request_id":""}"#)
                .unwrap()
                .is_terminal_for_request()
        );
        assert!(
            parse_event(br#"{"type":"error","fatal":true,"message":"x"}"#)
                .unwrap()
                .is_fatal()
        );
        assert!(parse_event(br#"{"type":"ready","bridge_protocol":1}"#).is_ok());
    }

    #[test]
    fn rejects_bad_events() {
        for line in [
            &b"not json"[..],
            b"[1]",
            b"{}",
            br#"{"type":3}"#,
            br#"{"type":"delta","request_id":1}"#,
            br#"{"type":"ready","bridge_protocol":2}"#,
            br#"{"type":"ready"}"#,
            b"\xff",
        ] {
            assert!(parse_event(line).is_err(), "{line:?}");
        }
    }

    #[test]
    fn generate_request_shape() {
        let messages = vec![Message {
            role: "user".into(),
            content: "a\nb".into(),
        }];
        let bytes = generate_request("r1", &messages, &Sampling::default()).unwrap();
        assert_eq!(bytes.iter().filter(|b| **b == b'\n').count(), 1);
        let value: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["type"], "generate");
        assert_eq!(value["messages"][0]["content"], "a\nb");
        assert_eq!(value["max_tokens"], -1);
        assert!(value.get("enable_thinking").is_none());
        let thinking = Sampling {
            enable_thinking: Some(false),
            ..Sampling::default()
        };
        let value: Value =
            serde_json::from_slice(&generate_request("r", &messages, &thinking).unwrap()).unwrap();
        assert_eq!(value["enable_thinking"], false);
    }

    #[test]
    fn generate_request_validation() {
        let user = vec![Message {
            role: "user".into(),
            content: "x".into(),
        }];
        let assistant = vec![Message {
            role: "assistant".into(),
            content: "x".into(),
        }];
        assert!(generate_request("", &user, &Sampling::default()).is_err());
        assert!(generate_request("r", &[], &Sampling::default()).is_err());
        assert!(generate_request("r", &assistant, &Sampling::default()).is_err());
        for bad in [
            Sampling {
                max_tokens: 0,
                ..Sampling::default()
            },
            Sampling {
                temperature: f64::NAN,
                ..Sampling::default()
            },
            Sampling {
                temperature: 5.5,
                ..Sampling::default()
            },
            Sampling {
                top_p: 0.0,
                ..Sampling::default()
            },
            Sampling {
                min_p: 1.0,
                ..Sampling::default()
            },
            Sampling {
                repetition_penalty: f64::INFINITY,
                ..Sampling::default()
            },
        ] {
            assert!(generate_request("r", &user, &bad).is_err(), "{bad:?}");
        }
        assert!(control_request("pause", "r").is_err());
        assert_eq!(
            control_request("cancel", "r").unwrap(),
            b"{\"request_id\":\"r\",\"type\":\"cancel\"}\n"
        );
    }
}
