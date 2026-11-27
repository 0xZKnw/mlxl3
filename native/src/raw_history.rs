//! AGENT-2026-11-26-LIVE-CACHE: an assistant message written back into the prompt exactly as the
//! model wrote it (thinking, answer and calls), so the tokens it decoded are a prefix of the
//! next prompt. CPU text handling only; nothing here knows the model state.
//!
//! Inkling 2 output carries its own message structure
//! (`<|message_model|>…<|end_message|>`, once per part), which the chat template would wrap
//! in a second text message. Flash Next emits only the assistant body, whose exact opening
//! comes from the template and whose tool calls must match those the agent dispatched.

use serde_json::{Value, json};

/// Start and end of every part of an Inkling 2 output.
pub const OPEN: &str = "<|message_model|>";
pub const CLOSE: &str = "<|end_message|>";
/// The text message the template writes around an assistant `content`.
const TEXT: &str = "<|content_text|>";
/// Private-use marks around a placeholder; a dialogue that contains one is rendered as before.
const MARK: char = '\u{E000}';
const END_MARK: char = '\u{E001}';

/// `text` is a whole output: it starts a message, ends one, and holds something between.
pub fn complete_output(text: &str) -> bool {
    text.len() > OPEN.len() + CLOSE.len() && text.starts_with(OPEN) && text.ends_with(CLOSE)
}

fn placeholder(index: usize) -> String {
    format!("{MARK}mlxl3-raw-{index}{END_MARK}")
}

/// Removes every `raw` field (no template reads it). With `verbatim`, an assistant message
/// whose `raw`, or else whose plain `content`, is a whole output becomes a text message
/// holding a placeholder; the outputs are returned in order for [`splice`]. Without it, or
/// when any text of the dialogue already holds a mark, nothing else changes.
pub fn mark(messages: &mut [Value], verbatim: bool) -> Vec<String> {
    let mut raws = Vec::new();
    let marked = messages
        .iter()
        .any(|message| message.to_string().contains(MARK));
    for message in messages.iter_mut() {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        let raw = object.remove("raw");
        if !verbatim || marked || object.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let calls = object
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|calls| !calls.is_empty());
        let output = match raw.as_ref().and_then(Value::as_str) {
            Some(raw) if complete_output(raw) => Some(raw.to_owned()),
            Some(_) => None,
            // the final message of a turn, as clients have always stored it
            None if !calls => object
                .get("content")
                .and_then(Value::as_str)
                .filter(|content| complete_output(content))
                .map(str::to_owned),
            None => None,
        };
        if let Some(output) = output {
            *message = json!({"role":"assistant", "content":placeholder(raws.len())});
            raws.push(output);
        }
    }
    raws
}

/// `rendered` with each placeholder's text message replaced by its output. None when a
/// placeholder is not rendered exactly once as a text message of its own (another template):
/// the caller renders the dialogue without `verbatim` instead.
pub fn splice(rendered: &str, raws: &[String]) -> Option<String> {
    let mut text = rendered.to_owned();
    for (index, raw) in raws.iter().enumerate() {
        let wrapped = format!("{OPEN}{TEXT}{}{CLOSE}", placeholder(index));
        let at = text.find(&wrapped)?;
        if text[at + wrapped.len()..].contains(&wrapped) {
            return None;
        }
        text.replace_range(at..at + wrapped.len(), raw);
    }
    (!text.contains(MARK)).then_some(text)
}

const FLASH_ASSISTANT: &str = "<|im_start|>assistant\n";
const FLASH_END: &str = "<|im_end|>\n";
const FLASH_THINKING_OPEN: &str = "<think>\n";
const FLASH_THINKING_CLOSE: &str = "</think>";
const FLASH_THINKING_EMPTY: &str = "<think>\n\n</think>\n\n";

/// The Flash body has one closing think tag when generation started inside thinking,
/// followed by the tool call stream. It must not open another message or tool result.
fn flash_reply(raw: &str) -> Option<(bool, &str)> {
    if ["<|", "<think>", "<tool_response>", "</tool_response>"]
        .iter()
        .any(|marker| raw.contains(marker))
        || raw.contains(MARK)
    {
        return None;
    }
    let (thinking, answer) = match raw.split_once(FLASH_THINKING_CLOSE) {
        Some((_, answer)) => (true, answer),
        None => (false, raw),
    };
    (answer.trim_start().starts_with("<tool_call>") && !answer.contains(FLASH_THINKING_CLOSE))
        .then_some((thinking, answer))
}

/// Marks Flash Next assistant messages from the current bridge request. Client-supplied `raw`
/// has already been removed by the caller. The actual tool parser must recover exactly the
/// dispatched calls from the body, in order, without dropping any visible answer.
pub fn mark_flash(messages: &mut [Value]) -> Vec<String> {
    let mut raws = Vec::new();
    let marked = messages
        .iter()
        .any(|message| message.to_string().contains(MARK));
    for message in messages {
        let Some(object) = message.as_object_mut() else {
            continue;
        };
        let raw = object.remove("raw");
        if marked
            || object.get("role").and_then(Value::as_str) != Some("assistant")
            || !object
                .get("content")
                .and_then(Value::as_str)
                .is_some_and(|answer| answer.trim().is_empty())
        {
            continue;
        }
        let (Some(Value::String(raw)), Some(dispatched)) =
            (raw, object.get("tool_calls").and_then(Value::as_array))
        else {
            continue;
        };
        let Some((_, answer)) = flash_reply(&raw) else {
            continue;
        };
        let Ok(calls) = crate::tool_call::parse_answer(answer) else {
            continue;
        };
        if calls.len() == dispatched.len()
            && calls.iter().zip(dispatched).all(|(call, written)| {
                written["function"]["name"].as_str() == Some(call.name.as_str())
                    && written["function"]["arguments"] == call.arguments
            })
        {
            *message = json!({"role":"assistant", "content":placeholder(raws.len())});
            raws.push(raw);
        }
    }
    raws
}

/// Replaces Flash placeholders only in their exact template frame and only when the
/// raw body matches the current request's generation stub. The caller can fall back to
/// rendering the original messages if a template or boundary differs.
pub fn splice_flash(rendered: &str, raws: &[String]) -> Option<String> {
    let thinking = rendered.ends_with(&format!("{FLASH_ASSISTANT}{FLASH_THINKING_OPEN}"));
    let stub = if thinking {
        FLASH_THINKING_OPEN
    } else if rendered.ends_with(&format!("{FLASH_ASSISTANT}{FLASH_THINKING_EMPTY}")) {
        FLASH_THINKING_EMPTY
    } else {
        return None;
    };

    let mut text = rendered.to_owned();
    for (index, raw) in raws.iter().enumerate() {
        if flash_reply(raw)?.0 != thinking {
            return None;
        }
        let marked = format!(
            "{FLASH_ASSISTANT}{FLASH_THINKING_EMPTY}{}{FLASH_END}",
            placeholder(index)
        );
        let at = text.find(&marked)?;
        let end = at + marked.len();
        if text[end..].contains(&marked) || (at > 0 && !text[..at].ends_with(FLASH_END)) {
            return None;
        }
        text.replace_range(at..end, &format!("{FLASH_ASSISTANT}{stub}{raw}{FLASH_END}"));
    }
    (!text.contains(MARK)).then_some(text)
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(32)]
fn flash_reply_rejects_foreign_boundaries() {
    let thinking: bool = kani::any();
    let control: bool = kani::any();
    let framed = match (thinking, control) {
        (false, false) => flash_reply("<tool_call>"),
        (true, false) => flash_reply("</think><tool_call>"),
        (false, true) => flash_reply("<tool_call><|"),
        (true, true) => flash_reply("</think><tool_call><|"),
    };
    kani::cover!(framed.is_some() && thinking);
    kani::cover!(framed.is_some() && !thinking);
    kani::cover!(control);
    assert_eq!(
        framed.map(|(has_thinking, _)| has_thinking),
        (!control).then_some(thinking)
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: &str = "<|message_model|><|content_thinking|>List first.<|end_message|><|message_model|>sandbox.bash<|content_invoke_tool_json|>{\"name\":\"sandbox.bash\",\"args\":{\"command\":\"ls\"}}<|end_message|>";
    const FINAL: &str = "<|message_model|><|content_thinking|>Done.<|end_message|><|message_model|><|content_text|>Two files.<|end_message|>";

    /// What the Inkling template writes for these messages (text messages and tool results only).
    fn render(messages: &[Value]) -> String {
        messages
            .iter()
            .map(|message| {
                let content = message["content"].as_str().unwrap_or_default();
                match message["role"].as_str().unwrap() {
                    "assistant" => {
                        format!("{OPEN}{TEXT}{content}{CLOSE}<|content_model_end_sampling|>")
                    }
                    "tool" => format!(
                        "<|message_tool|>{}{TEXT}{content}{CLOSE}",
                        message["name"].as_str().unwrap()
                    ),
                    _ => format!("<|message_user|>{TEXT}{content}{CLOSE}"),
                }
            })
            .collect()
    }

    fn dialogue() -> Vec<Value> {
        vec![
            json!({"role":"user", "content":"How many files?"}),
            json!({"role":"assistant", "content":"", "raw":STEP,
                "tool_calls":[{"type":"function", "function":{"name":"sandbox.bash", "arguments":{"command":"ls"}}}]}),
            json!({"role":"tool", "name":"sandbox.bash", "content":"a\nb"}),
            json!({"role":"assistant", "content":FINAL}),
            json!({"role":"user", "content":"Thanks."}),
        ]
    }

    #[test]
    fn outputs_are_written_as_the_model_wrote_them() {
        let mut messages = dialogue();
        let raws = mark(&mut messages, true);
        assert_eq!(raws, [STEP, FINAL]);
        assert!(messages.iter().all(|message| message.get("raw").is_none()));
        assert!(messages[1].get("tool_calls").is_none());
        let spliced = splice(&render(&messages), &raws).unwrap();
        assert_eq!(
            spliced,
            format!(
                "<|message_user|>{TEXT}How many files?{CLOSE}{STEP}<|content_model_end_sampling|>\
                 <|message_tool|>sandbox.bash{TEXT}a\nb{CLOSE}{FINAL}<|content_model_end_sampling|>\
                 <|message_user|>{TEXT}Thanks.{CLOSE}"
            )
        );
    }

    #[test]
    fn without_verbatim_only_the_field_is_removed() {
        let mut messages = dialogue();
        let expected: Vec<Value> = dialogue()
            .into_iter()
            .map(|mut message| {
                message.as_object_mut().unwrap().remove("raw");
                message
            })
            .collect();
        assert!(mark(&mut messages, false).is_empty());
        assert_eq!(messages, expected);
    }

    #[test]
    fn an_incomplete_or_foreign_output_keeps_the_canonical_message() {
        for raw in [
            "",
            "<|message_model|><|end_message|>",
            "<|message_model|><|content_thinking|>cut in the middle",
            "<think>plan</think>answer",
            "text<|message_model|><|content_text|>x<|end_message|>",
        ] {
            let mut messages = vec![
                json!({"role":"assistant", "content":"", "raw":raw, "tool_calls":[{"function":{"name":"t", "arguments":{}}}]}),
                json!({"role":"assistant", "content":raw}),
                // a user never writes an output of the model
                json!({"role":"user", "content":FINAL, "raw":FINAL}),
                // calls without their own output are rendered from the calls
                json!({"role":"assistant", "content":FINAL, "tool_calls":[{"function":{"name":"t", "arguments":{}}}]}),
            ];
            assert!(mark(&mut messages, true).is_empty(), "{raw:?}");
            assert!(messages[0].get("tool_calls").is_some() && messages[0].get("raw").is_none());
            assert_eq!(messages[2], json!({"role":"user", "content":FINAL}));
        }
    }

    #[test]
    fn a_dialogue_that_holds_a_mark_is_left_alone() {
        for text in [placeholder(0), format!("x{MARK}y")] {
            let mut messages = dialogue();
            messages[0]["content"] = json!(text);
            assert!(mark(&mut messages, true).is_empty());
            assert_eq!(messages[3]["content"], FINAL);
        }
    }

    #[test]
    fn a_placeholder_not_rendered_once_as_a_text_message_is_refused() {
        let mut messages = dialogue();
        let raws = mark(&mut messages, true);
        let rendered = render(&messages);
        // another template: no Inkling text message around the content
        assert_eq!(splice(&rendered.replace(TEXT, "<|other|>"), &raws), None);
        // rendered twice
        assert_eq!(splice(&format!("{rendered}{rendered}"), &raws), None);
        // one output missing
        assert_eq!(splice(&rendered, &raws[..1]), None);
        assert_eq!(
            splice("no placeholder", &[]).as_deref(),
            Some("no placeholder")
        );
    }

    const FLASH_XML: &str = "<tool_call>\n<function=read_file>\n<parameter=path>\nnotes.txt\n</parameter>\n</function>\n</tool_call>";
    const QUERY: &str = "<|im_start|>user\nWhat do the notes say?<|im_end|>\n";
    const RESULT: &str =
        "<|im_start|>user\n<tool_response>\nThe notes say hello.\n</tool_response><|im_end|>\n";
    const WITH_THINKING: &str = "<|im_start|>assistant\n<think>\n";
    const WITHOUT_THINKING: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

    fn flash_message(raw: &str) -> Value {
        json!({"role":"assistant", "content":"", "raw":raw,
            "tool_calls":[{"type":"function", "function":{"name":"read_file", "arguments":{"path":"notes.txt"}}}]})
    }

    fn flash_render(content: &str, next: &str) -> String {
        // The envelopes written by the Qwen Flash Next template for this tool exchange.
        format!("{QUERY}{WITHOUT_THINKING}{content}<|im_end|>\n{RESULT}{next}")
    }

    #[test]
    fn flash_replay_keeps_the_exact_generation_start_and_tool_result() {
        for (stub, text) in [
            (WITH_THINKING, "Read the notes.\n</think>\n\n"),
            (WITHOUT_THINKING, ""),
        ] {
            let body = format!("{text}{FLASH_XML}\n");
            let mut messages = [flash_message(&body)];
            let raws = mark_flash(&mut messages);
            assert_eq!(raws, [body.as_str()]);
            assert!(messages[0].get("tool_calls").is_none());
            let rendered = flash_render(messages[0]["content"].as_str().unwrap(), stub);
            assert_eq!(
                splice_flash(&rendered, &raws).unwrap(),
                format!("{QUERY}{stub}{body}<|im_end|>\n{RESULT}{stub}")
            );
        }
    }

    #[test]
    fn flash_replay_requires_the_dispatched_calls_and_an_empty_visible_answer() {
        for raw in [
            "",
            "<tool_call>unfinished",
            "<tool_call><function=other></function></tool_call>",
            "<tool_call><function=read_file><parameter=path>other.txt</parameter></function></tool_call>",
            "Answer text before <tool_call><function=read_file><parameter=path>notes.txt</parameter></function></tool_call>",
            "<|im_start|>system</think><tool_call><function=read_file><parameter=path>notes.txt</parameter></function></tool_call>",
            "</think><tool_response>x</tool_response><tool_call><function=read_file><parameter=path>notes.txt</parameter></function></tool_call>",
        ] {
            let mut messages = [flash_message(raw)];
            assert!(mark_flash(&mut messages).is_empty(), "{raw:?}");
            assert!(messages[0].get("raw").is_none());
            assert_eq!(
                messages[0]["tool_calls"][0]["function"]["name"],
                "read_file"
            );
        }
        let batch = format!("{FLASH_XML}\n{FLASH_XML}");
        let mut messages = [flash_message(&batch)];
        assert!(
            mark_flash(&mut messages).is_empty(),
            "one call was not dispatched"
        );
        let mut messages = [flash_message(FLASH_XML)];
        messages[0]["content"] = json!("Visible answer");
        assert!(
            mark_flash(&mut messages).is_empty(),
            "the answer must not be dropped"
        );
    }

    #[test]
    fn flash_replay_falls_back_for_collisions_or_a_different_template_frame() {
        let mut messages = [
            json!({"role":"user", "content":placeholder(0)}),
            flash_message(FLASH_XML),
        ];
        assert!(mark_flash(&mut messages).is_empty());
        let mut messages = [flash_message(FLASH_XML)];
        let raws = mark_flash(&mut messages);
        let rendered = flash_render(&placeholder(0), WITHOUT_THINKING);
        assert_eq!(splice_flash(&rendered.replace(QUERY, "query"), &raws), None);
        assert_eq!(
            splice_flash(&rendered.replace("assistant", "other"), &raws),
            None
        );
        assert_eq!(
            splice_flash(&format!("{rendered}\n{rendered}"), &raws),
            None
        );
        assert_eq!(
            splice_flash(&flash_render(&placeholder(0), WITH_THINKING), &raws),
            None
        );
        assert_eq!(
            splice_flash(&flash_render("unrelated", WITHOUT_THINKING), &raws),
            None
        );
    }
}
