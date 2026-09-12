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
            out.push(Fragment {
                channel: self.mode,
                text,
            });
        }
    }

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
        let tail = std::mem::take(&mut self.buffer);
        self.fragment(tail, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
