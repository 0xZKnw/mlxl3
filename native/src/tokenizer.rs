//! Local tokenizer and text-only chat templates. Never downloads model assets.
use anyhow::{Context, Result, bail, ensure};
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

pub struct ChatTokenizer {
    tokenizer: Tokenizer,
    environment: Environment<'static>,
    bos_token: String,
    eos_token: String,
    eos_ids: Vec<u32>,
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
        environment.add_filter("tojson", |value: minijinja::Value| {
            serde_json::to_string(&value)
                .map_err(|error| Error::new(ErrorKind::InvalidOperation, error.to_string()))
        });
        environment
            .add_template_owned("chat", inference_template(&template))
            .context("compiling chat template")?;
        Ok(Self {
            tokenizer,
            environment,
            bos_token,
            eos_token,
            eos_ids,
        })
    }

    pub fn render(&self, messages: &[Message]) -> Result<String> {
        ensure!(!messages.is_empty(), "chat requires at least one message");
        for message in messages {
            if !matches!(message.role.as_str(), "system" | "user" | "assistant") {
                bail!(
                    "unsupported role {:?}: native chat currently supports text messages without tools",
                    message.role
                );
            }
        }
        self.environment
            .get_template("chat")?
            .render(context! {
                messages => messages,
                bos_token => &self.bos_token,
                eos_token => &self.eos_token,
                add_generation_prompt => true,
                preserve_thinking => true,
                tools => Option::<bool>::None,
                documents => Option::<bool>::None,
            })
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

    pub fn eos_ids(&self) -> &[u32] {
        &self.eos_ids
    }

    /// Gives the caller access to decode_stream(false) for incremental output.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

#[cfg(test)]
mod tests {
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
                    role: "tool".into(),
                    content: "result".into()
                }])
                .is_err()
        );
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
