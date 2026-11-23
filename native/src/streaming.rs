//! Incremental reasoning separation, including Gemma channel markers.
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Channel {
    Answer,
    Thinking,
}

#[derive(Debug, PartialEq, Serialize)]
pub struct Fragment {
    pub channel: Channel,
    pub text: String,
}

pub struct ThinkingSplitter {
    mode: Channel,
    buffer: String,
    drop_newline: bool,
    channel_header: Option<String>,
    /// Inkling 2: answer text was written, then a new `<|message_model|>` began; the next
    /// answer text starts a paragraph (a tool call header must not stick to that text).
    answer_written: bool,
    separator_pending: bool,
}

impl ThinkingSplitter {
    pub fn new(prompt: &str) -> Self {
        let prompt = prompt.trim_end();
        Self {
            mode: if prompt.ends_with("<think>") || prompt.ends_with("<|channel>thought") {
                Channel::Thinking
            } else {
                Channel::Answer
            },
            buffer: String::new(),
            drop_newline: false,
            channel_header: None,
            answer_written: false,
            separator_pending: false,
        }
    }

    fn fragment(&mut self, text: String, out: &mut Vec<Fragment>) {
        if text.is_empty() {
            return;
        }
        let text = if self.drop_newline {
            self.drop_newline = false;
            text.strip_prefix("\r\n")
                .or_else(|| text.strip_prefix('\n'))
                .unwrap_or(&text)
                .to_owned()
        } else {
            text
        };
        if !text.is_empty() {
            if self.mode == Channel::Answer {
                if std::mem::take(&mut self.separator_pending) {
                    out.push(Fragment {
                        channel: Channel::Answer,
                        text: "\n\n".into(),
                    });
                }
                self.answer_written = true;
            }
            out.push(Fragment {
                channel: self.mode,
                text,
            });
        }
    }

    /// Production token-aware path. `control` must come from the tokenizer's
    /// declared delimiter-ID table, never from searching decoded text.
    pub fn feed_token(&mut self, text: &str, control: Option<&str>) -> Vec<Fragment> {
        let Some(control) = control.filter(|marker| text.ends_with(marker)) else {
            return self.feed_literal(text);
        };
        let mut out = self.feed_literal(&text[..text.len() - control.len()]);
        if let Some(header) = self.channel_header.take() {
            self.fragment(format!("<|channel>{header}"), &mut out);
        }
        match control {
            "<think>" | "<|channel>thought" => {
                self.mode = Channel::Thinking;
                self.drop_newline = true;
            }
            "</think>" | "<channel|>" if self.mode == Channel::Thinking => {
                self.mode = Channel::Answer;
                self.drop_newline = true;
            }
            "<|channel>" => self.channel_header = Some(String::new()),
            // Inkling 2: <|message_model|><|content_thinking|>...<|end_message|>
            // <|message_model|><|content_text|>answer. The message and content headers are structure,
            // not text; <|end_message|> closes a thinking block.
            "<|content_thinking|>" => {
                self.mode = Channel::Thinking;
                self.drop_newline = false;
            }
            "<|end_message|>" => {
                self.mode = Channel::Answer;
                self.drop_newline = false;
            }
            "<|message_model|>" => self.separator_pending |= self.answer_written,
            "<|content_text|>" => {}
            _ => self.fragment(control.to_owned(), &mut out),
        }
        out
    }

    pub fn feed_literal(&mut self, text: &str) -> Vec<Fragment> {
        let mut out = Vec::new();
        let combined;
        let text = if !self.buffer.is_empty() {
            combined = std::mem::take(&mut self.buffer) + text;
            combined.as_str()
        } else {
            text
        };
        if self.drop_newline && text == "\r" {
            self.buffer.push('\r');
            return out;
        }
        if let Some(mut header) = self.channel_header.take() {
            header.push_str(text);
            if header.len() < "thought".len() && "thought".starts_with(&header) {
                self.channel_header = Some(header);
                return out;
            }
            if let Some(tail) = header.strip_prefix("thought") {
                self.mode = Channel::Thinking;
                self.drop_newline = true;
                self.fragment(tail.to_owned(), &mut out);
            } else {
                self.fragment(format!("<|channel>{header}"), &mut out);
            }
        } else {
            self.fragment(text.to_owned(), &mut out);
        }
        out
    }

    // Legacy text-only CLI splitter: lexical markers are its explicit input
    // protocol. Model generation uses feed_token instead.
    pub fn feed(&mut self, text: &str) -> Vec<Fragment> {
        self.buffer.push_str(text);
        let mut out = Vec::new();
        while !self.buffer.is_empty() {
            // Wait for a possible LF; finish() still preserves a lone CR.
            if self.drop_newline && self.buffer == "\r" {
                break;
            }
            let markers = match self.mode {
                Channel::Thinking => ["</think>", "<channel|>"],
                Channel::Answer => ["<think>", "<|channel>thought"],
            };
            if let Some((position, marker)) = markers
                .iter()
                .filter_map(|m| self.buffer.find(m).map(|p| (p, *m)))
                .min()
            {
                let prefix = self.buffer[..position].to_owned();
                self.fragment(prefix, &mut out);
                self.buffer.drain(..position + marker.len());
                self.mode = if self.mode == Channel::Answer {
                    Channel::Thinking
                } else {
                    Channel::Answer
                };
                self.drop_newline = true;
            } else {
                let keep = markers
                    .iter()
                    .map(|marker| {
                        (1..marker.len())
                            .rev()
                            .find(|&n| self.buffer.ends_with(&marker[..n]))
                            .unwrap_or(0)
                    })
                    .max()
                    .unwrap_or(0);
                let end = self.buffer.len() - keep;
                let prefix: String = self.buffer.drain(..end).collect();
                self.fragment(prefix, &mut out);
                break;
            }
        }
        out
    }

    pub fn finish(&mut self) -> Vec<Fragment> {
        let mut out = Vec::new();
        if let Some(header) = self.channel_header.take() {
            self.fragment(format!("<|channel>{header}"), &mut out);
        }
        let tail = std::mem::take(&mut self.buffer);
        self.fragment(tail, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn literal_markers_preserve_every_chunk_boundary() {
        let input = "En XML, `</think>` est littéral. <think>code<channel|> </think>";
        for cut in input
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(input.len()))
        {
            let mut splitter = ThinkingSplitter::new("");
            let mut out = splitter.feed_token(&input[..cut], None);
            out.extend(splitter.feed_token(&input[cut..], None));
            out.extend(splitter.finish());
            assert!(out.iter().all(|f| f.channel == Channel::Answer));
            assert_eq!(out.into_iter().map(|f| f.text).collect::<String>(), input);
        }
        let mut splitter = ThinkingSplitter::new("");
        assert!(splitter.feed_token("<think>", Some("<think>")).is_empty());
        assert_eq!(
            splitter.feed_token("reason", None)[0].channel,
            Channel::Thinking
        );
        assert!(splitter.feed_token("</think>", Some("</think>")).is_empty());
        assert_eq!(
            splitter.feed_token("answer", None)[0].channel,
            Channel::Answer
        );
    }

    #[test]
    fn markers_split_at_every_character() {
        for (open, close) in [("<think>", "</think>"), ("<|channel>thought", "<channel|>")] {
            let input = format!("{open}\nréfléchis 🦀{close}\nbonjour");
            let mut parser = ThinkingSplitter::new("");
            let mut result = Vec::new();
            for c in input.chars() {
                result.extend(parser.feed(&c.to_string()));
            }
            result.extend(parser.finish());
            let text = |mode| {
                result
                    .iter()
                    .filter(|f| f.channel == mode)
                    .map(|f| f.text.as_str())
                    .collect::<String>()
            };
            assert_eq!(text(Channel::Thinking), "réfléchis 🦀");
            assert_eq!(text(Channel::Answer), "bonjour");
        }
    }
    #[test]
    fn prefilled_and_incomplete_markers() {
        let mut parser = ThinkingSplitter::new("assistant\n<think>\n");
        let result = parser.feed("thinking</think>\nanswer<thi");
        assert_eq!(result[0].channel, Channel::Thinking);
        assert_eq!(result[1].text, "answer");
        assert_eq!(parser.finish()[0].text, "<thi");
    }

    #[test]
    fn crlf_is_independent_of_chunk_boundaries() {
        fn contents(parts: &[&str]) -> (String, String) {
            let mut parser = ThinkingSplitter::new("");
            let mut fragments = Vec::new();
            for part in parts {
                fragments.extend(parser.feed(part));
            }
            fragments.extend(parser.finish());
            let text = |mode| {
                fragments
                    .iter()
                    .filter(|f| f.channel == mode)
                    .map(|f| f.text.as_str())
                    .collect::<String>()
            };
            (text(Channel::Thinking), text(Channel::Answer))
        }
        for (open, close) in [("<think>", "</think>"), ("<|channel>thought", "<channel|>")] {
            let input = format!("{open}\r\nréfléchis 🦀{close}\r\nbonjour");
            let expected = ("réfléchis 🦀".into(), "bonjour".into());
            assert_eq!(contents(&[&input]), expected);
            for index in input.char_indices().map(|(i, _)| i).chain([input.len()]) {
                assert_eq!(contents(&[&input[..index], &input[index..]]), expected);
            }
            let characters: Vec<String> = input.chars().map(|c| c.to_string()).collect();
            assert_eq!(
                contents(&characters.iter().map(String::as_str).collect::<Vec<_>>()),
                expected
            );
        }
    }

    #[test]
    fn inkling_messages_after_answer_text_start_a_paragraph() {
        let stream: [(&str, Option<&str>); 8] = [
            ("<|message_model|>", Some("<|message_model|>")),
            ("<|content_thinking|>", Some("<|content_thinking|>")),
            ("Plan.", None),
            ("<|end_message|>", Some("<|end_message|>")),
            ("<|message_model|>", Some("<|message_model|>")),
            ("<|content_text|>", Some("<|content_text|>")),
            ("Je liste.", None),
            ("<|end_message|>", Some("<|end_message|>")),
        ];
        let tail: [(&str, Option<&str>); 3] = [
            ("<|message_model|>", Some("<|message_model|>")),
            ("sandbox.javascript", None),
            ("<|content_invoke_tool_json|>{}", None),
        ];
        let mut parser = ThinkingSplitter::new("");
        let mut answer = String::new();
        let mut thinking = String::new();
        for (text, control) in stream.into_iter().chain(tail) {
            for fragment in parser.feed_token(text, control) {
                match fragment.channel {
                    Channel::Thinking => thinking.push_str(&fragment.text),
                    Channel::Answer => answer.push_str(&fragment.text),
                }
            }
        }
        assert_eq!(thinking, "Plan.");
        assert_eq!(
            answer,
            "Je liste.\n\nsandbox.javascript<|content_invoke_tool_json|>{}"
        );
    }

    #[test]
    fn inkling_message_markers_split_thinking_and_answer() {
        // token-by-token stream as the bridge feeds it: text pieces plus the declared control tokens
        let stream: [(&str, Option<&str>); 9] = [
            ("<|message_model|>", Some("<|message_model|>")),
            ("<|content_thinking|>", Some("<|content_thinking|>")),
            ("The user asks", None),
            (" in French.", None),
            ("<|end_message|>", Some("<|end_message|>")),
            ("<|message_model|>", Some("<|message_model|>")),
            ("<|content_text|>", Some("<|content_text|>")),
            ("Le ciel", None),
            (" est bleu.", None),
        ];
        let mut parser =
            ThinkingSplitter::new("<|message_system|><|content_text|>x<|end_message|>");
        let mut thinking = String::new();
        let mut answer = String::new();
        for (text, control) in stream {
            for fragment in parser.feed_token(text, control) {
                match fragment.channel {
                    Channel::Thinking => thinking.push_str(&fragment.text),
                    Channel::Answer => answer.push_str(&fragment.text),
                }
            }
        }
        for fragment in parser.finish() {
            answer.push_str(&fragment.text);
        }
        assert_eq!(thinking, "The user asks in French.");
        assert_eq!(answer, "Le ciel est bleu.");
        // without thinking (reasoning effort 0) the answer keeps every character
        let mut parser = ThinkingSplitter::new("");
        let mut answer = String::new();
        for (text, control) in [
            ("<|message_model|>", Some("<|message_model|>")),
            ("<|content_text|>", Some("<|content_text|>")),
            ("\nBonjour", None),
        ] {
            for fragment in parser.feed_token(text, control) {
                assert_eq!(fragment.channel, Channel::Answer);
                answer.push_str(&fragment.text);
            }
        }
        assert_eq!(answer, "\nBonjour");
        // a marker text without the control id is ordinary text (never searched in decoded text)
        let mut parser = ThinkingSplitter::new("");
        let out = parser.feed_token("<|content_thinking|>", None);
        assert_eq!(
            out,
            [Fragment {
                channel: Channel::Answer,
                text: "<|content_thinking|>".into()
            }]
        );
    }

    #[test]
    fn unfinished_cr_is_preserved() {
        let mut parser = ThinkingSplitter::new("");
        assert!(parser.feed("<think>\r").is_empty());
        assert!(parser.feed("").is_empty());
        assert_eq!(
            parser.finish(),
            [Fragment {
                channel: Channel::Thinking,
                text: "\r".into()
            }]
        );
        assert!(parser.finish().is_empty());

        let mut parser = ThinkingSplitter::new("");
        assert!(parser.feed("<think>\r").is_empty());
        assert_eq!(
            parser.feed("x"),
            [Fragment {
                channel: Channel::Thinking,
                text: "\rx".into()
            }]
        );
    }
}
