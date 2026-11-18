//! Liquid d1 decision models: one forward pass, answers read at slots, no generated token.
//! Prompt layout, slots and readouts follow the reference code published with
//! LiquidAI/d1-2B (revision a6bce774): `prompt.py`, `pointer.py`, `box.py`, `grid.py`,
//! `calibration.py`, `api.py`, `attention.py::pack_set`.
// Offsets are bytes inside this module; spans leave it as characters, like the reference.
use anyhow::{Context, Result, bail, ensure};
use serde::{
    Deserialize, Serialize,
    de::{MapAccess, SeqAccess, Visitor},
    ser::{SerializeMap, SerializeSeq},
};
use std::{collections::HashSet, fmt, path::Path};
use tokenizers::Tokenizer;

pub const IM_START: &str = "<|im_start|>";
pub const IM_END: &str = "<|im_end|>";
const BREAK: char = '\u{200b}';
/// A spoken question's place in its suffix until its clip's markup replaces it.
pub const SPOKEN: &str = "\u{e000}";
const LEAD: &str = "<think></think>";
const HEADER: &str = "QUESTION:\n";
const EDGES: [&str; 4] = ["top", "bottom", "left", "right"];
const INSTANCES: usize = 16;
const MAX_INSTANCES: usize = 64;
const TRAINED_SET_SIZE: usize = 128;
/// One pass holds every key of the sequence; the sliding layers build a T x T mask.
pub const MAX_TOKENS: usize = 8192;

// ------------------------------------------------------------------ ordered JSON

/// JSON that keeps object order: question order and a state's key order are part of the prompt.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(text) => Some(text),
            _ => None,
        }
    }
    fn as_f64(&self) -> Option<f64> {
        match *self {
            Json::Int(v) => Some(v as f64),
            Json::UInt(v) => Some(v as f64),
            Json::Float(v) => Some(v),
            _ => None,
        }
    }
    fn truthy(&self) -> bool {
        match self {
            Json::Null => false,
            Json::Bool(v) => *v,
            Json::Int(v) => *v != 0,
            Json::UInt(v) => *v != 0,
            Json::Float(v) => *v != 0.,
            Json::Str(v) => !v.is_empty(),
            Json::Array(v) => !v.is_empty(),
            Json::Object(v) => !v.is_empty(),
        }
    }
    fn object(entries: Vec<(&str, Json)>) -> Json {
        Json::Object(
            entries
                .into_iter()
                .map(|(k, v)| (k.to_owned(), v))
                .collect(),
        )
    }
}

impl<'de> Deserialize<'de> for Json {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Json;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("a JSON value")
            }
            fn visit_unit<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_none<E>(self) -> Result<Json, E> {
                Ok(Json::Null)
            }
            fn visit_bool<E>(self, v: bool) -> Result<Json, E> {
                Ok(Json::Bool(v))
            }
            fn visit_i64<E>(self, v: i64) -> Result<Json, E> {
                Ok(Json::Int(v))
            }
            fn visit_u64<E>(self, v: u64) -> Result<Json, E> {
                Ok(i64::try_from(v).map_or(Json::UInt(v), Json::Int))
            }
            fn visit_f64<E>(self, v: f64) -> Result<Json, E> {
                Ok(Json::Float(v))
            }
            fn visit_str<E>(self, v: &str) -> Result<Json, E> {
                Ok(Json::Str(v.to_owned()))
            }
            fn visit_string<E>(self, v: String) -> Result<Json, E> {
                Ok(Json::Str(v))
            }
            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Json, A::Error> {
                let mut items = Vec::new();
                while let Some(item) = seq.next_element()? {
                    items.push(item);
                }
                Ok(Json::Array(items))
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Json, A::Error> {
                let mut entries: Vec<(String, Json)> = Vec::new();
                while let Some((key, value)) = map.next_entry::<String, Json>()? {
                    // Python's dict: a repeated key keeps its first place and its last value.
                    match entries.iter_mut().find(|(k, _)| *k == key) {
                        Some(entry) => entry.1 = value,
                        None => entries.push((key, value)),
                    }
                }
                Ok(Json::Object(entries))
            }
        }
        deserializer.deserialize_any(V)
    }
}

impl Serialize for Json {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Json::Null => serializer.serialize_unit(),
            Json::Bool(v) => serializer.serialize_bool(*v),
            Json::Int(v) => serializer.serialize_i64(*v),
            Json::UInt(v) => serializer.serialize_u64(*v),
            Json::Float(v) => serializer.serialize_f64(*v),
            Json::Str(v) => serializer.serialize_str(v),
            Json::Array(items) => {
                let mut seq = serializer.serialize_seq(Some(items.len()))?;
                for item in items {
                    seq.serialize_element(item)?;
                }
                seq.end()
            }
            Json::Object(entries) => {
                let mut map = serializer.serialize_map(Some(entries.len()))?;
                for (key, value) in entries {
                    map.serialize_entry(key, value)?;
                }
                map.end()
            }
        }
    }
}

/// Python's `repr(float)` as `json.dumps` writes it.
fn python_float(value: f64) -> String {
    if value.is_nan() {
        return "NaN".into();
    }
    if value.is_infinite() {
        return if value > 0. { "Infinity" } else { "-Infinity" }.into();
    }
    if value == 0. {
        return if value.is_sign_negative() {
            "-0.0"
        } else {
            "0.0"
        }
        .into();
    }
    let text = format!("{:e}", value.abs());
    let (mantissa, exponent) = text.split_once('e').expect("LowerExp has an exponent");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let point = exponent.parse::<i32>().expect("decimal exponent") + 1;
    let sign = if value < 0. { "-" } else { "" };
    let count = digits.len() as i32;
    if point <= -4 || point > 16 {
        let tail = if count > 1 {
            format!(".{}", &digits[1..])
        } else {
            String::new()
        };
        let e = point - 1;
        return format!(
            "{sign}{}{tail}e{}{:02}",
            &digits[..1],
            if e < 0 { '-' } else { '+' },
            e.abs()
        );
    }
    if point <= 0 {
        format!("{sign}0.{}{digits}", "0".repeat((-point) as usize))
    } else if point >= count {
        format!("{sign}{digits}{}.0", "0".repeat((point - count) as usize))
    } else {
        format!(
            "{sign}{}.{}",
            &digits[..point as usize],
            &digits[point as usize..]
        )
    }
}

/// Python's `f"{value:g}"`.
fn python_g(value: f64) -> String {
    if value == 0. {
        return if value.is_sign_negative() { "-0" } else { "0" }.into();
    }
    if !value.is_finite() {
        return if value.is_nan() {
            "nan"
        } else if value > 0. {
            "inf"
        } else {
            "-inf"
        }
        .into();
    }
    let text = format!("{:.5e}", value);
    let (mantissa, exponent) = text.split_once('e').expect("LowerExp has an exponent");
    let exponent: i32 = exponent.parse().expect("decimal exponent");
    let strip = |s: String| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            s
        }
    };
    if (-4..6).contains(&exponent) {
        strip(format!("{:.*}", (5 - exponent).max(0) as usize, value))
    } else {
        format!(
            "{}e{}{:02}",
            strip(mantissa.to_owned()),
            if exponent < 0 { '-' } else { '+' },
            exponent.abs()
        )
    }
}

fn python_string(text: &str, out: &mut String) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

fn json_pointer(path: &str, key: &str) -> String {
    format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"))
}

/// `json.dumps(value, ensure_ascii=False, indent=2)`, and where each string value sits (quotes excluded).
fn python_dump(
    value: &Json,
    level: usize,
    path: &str,
    out: &mut String,
    strings: &mut Vec<(usize, usize, String)>,
) {
    match value {
        Json::Null => out.push_str("null"),
        Json::Bool(v) => out.push_str(if *v { "true" } else { "false" }),
        Json::Int(v) => out.push_str(&v.to_string()),
        Json::UInt(v) => out.push_str(&v.to_string()),
        Json::Float(v) => out.push_str(&python_float(*v)),
        Json::Str(v) => {
            let start = out.len() + 1;
            python_string(v, out);
            strings.push((start, out.len() - 1, path.to_owned()));
        }
        Json::Array(items) if items.is_empty() => out.push_str("[]"),
        Json::Object(entries) if entries.is_empty() => out.push_str("{}"),
        Json::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                out.push_str(if index == 0 { "\n" } else { ",\n" });
                out.push_str(&"  ".repeat(level + 1));
                python_dump(item, level + 1, &format!("{path}/{index}"), out, strings);
            }
            out.push('\n');
            out.push_str(&"  ".repeat(level));
            out.push(']');
        }
        Json::Object(entries) => {
            out.push('{');
            for (index, (key, item)) in entries.iter().enumerate() {
                out.push_str(if index == 0 { "\n" } else { ",\n" });
                out.push_str(&"  ".repeat(level + 1));
                python_string(key, out);
                out.push_str(": ");
                python_dump(item, level + 1, &json_pointer(path, key), out, strings);
            }
            out.push('\n');
            out.push_str(&"  ".repeat(level));
            out.push('}');
        }
    }
}

fn python_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}
fn blank(text: &str) -> bool {
    text.chars().all(python_space)
}
fn python_strip(text: &str) -> &str {
    text.trim_matches(python_space)
}

// ------------------------------------------------------------------ questions

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadKind {
    Text,
    Number,
    Date,
}
impl ReadKind {
    fn name(self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Number => "number",
            Self::Date => "date",
        }
    }
    pub fn cap(self) -> usize {
        if self == Self::Text { 512 } else { 32 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub low: f64,
    pub high: f64,
    pub size: usize,
}
impl Grid {
    fn new(low: f64, high: f64, size: usize) -> Result<Self> {
        ensure!(
            size >= 2 && low < high,
            "a grid needs two values or more, low to high"
        );
        Ok(Self { low, high, size })
    }
    fn step(&self) -> f64 {
        (self.high - self.low) / (self.size - 1) as f64
    }
    fn values(&self) -> Vec<f64> {
        let step = self.step();
        (0..self.size - 1)
            .map(|i| self.low + i as f64 * step)
            .chain([self.high])
            .collect()
    }
    /// The median, and the central 80 percent interval read at `level`'s quantiles.
    fn read(&self, probabilities: &[f64], level: impl Fn(f64) -> f64) -> (f64, f64, f64) {
        let total = fsum(probabilities.iter().copied());
        let values = self.values();
        let at = [level(0.1), level(0.5), level(0.9)];
        let mut out = [None; 3];
        let mut mass = 0.;
        for (value, p) in values.iter().zip(probabilities) {
            mass += p / total;
            for (slot, quantile) in out.iter_mut().zip(at) {
                if slot.is_none() && mass >= quantile {
                    *slot = Some(*value);
                }
            }
        }
        let last = *values.last().expect("grid has values");
        (
            out[0].unwrap_or(last),
            out[1].unwrap_or(last),
            out[2].unwrap_or(last),
        )
    }
}

/// Neumaier's compensated sum, close to Python's `math.fsum` on probability vectors.
fn fsum(values: impl Iterator<Item = f64>) -> f64 {
    let (mut sum, mut carry) = (0f64, 0f64);
    for value in values {
        let next = sum + value;
        carry += if sum.abs() >= value.abs() {
            (sum - next) + value
        } else {
            (value - next) + sum
        };
        sum = next;
    }
    sum + carry
}

#[derive(Clone, Debug, PartialEq)]
pub enum Question {
    Noul {
        instructions: String,
        criteria: Option<(String, String)>,
    },
    Choice {
        instructions: String,
        criteria: Vec<(String, Option<String>)>,
        multiple: bool,
    },
    Score {
        instructions: String,
        criteria: Vec<String>,
    },
    Read {
        instructions: String,
        kind: ReadKind,
        unit: Option<String>,
    },
    Estimate {
        instructions: String,
        grid: Grid,
        unit: Option<String>,
    },
    Box {
        instructions: String,
    },
    Boxes {
        instructions: String,
        max: Option<usize>,
    },
}

impl Question {
    /// A caller's question in the API's JSON, refused with what to change.
    pub fn parse(q: &Json, settings: &Settings) -> Result<Self> {
        ensure!(
            matches!(q, Json::Object(_)),
            r#"a question is an object such as {{"type": "noul", "instructions": "Is it urgent?"}}"#
        );
        let kind = q.get("type").and_then(Json::as_str).unwrap_or("");
        ensure!(
            [
                "noul", "choice", "score", "text", "number", "date", "box", "boxes"
            ]
            .contains(&kind),
            r#""type" is one of noul, choice, score, text, number, date, box, boxes, not {kind:?}"#
        );
        let instructions = match q.get("instructions") {
            Some(Json::Str(text)) if !blank(text) => text.clone(),
            Some(spoken @ Json::Object(_))
                if spoken.get("type") == Some(&Json::Str("audio".into())) =>
            {
                SPOKEN.to_owned()
            }
            Some(Json::Object(_)) => bail!(
                r#"spoken instructions are an audio clip: {{"type": "audio", "path" | "url": ...}}"#
            ),
            _ => bail!(r#""instructions" is the question as text"#),
        };
        let criteria = q.get("criteria").filter(|c| **c != Json::Null);
        let multiple = q.get("multiple").is_some_and(Json::truthy);
        ensure!(
            !multiple || kind == "choice",
            r#""multiple" belongs to a choice"#
        );
        ensure!(
            q.get("max").is_none() || kind == "boxes",
            r#""max" belongs to boxes"#
        );
        ensure!(
            q.get("step").is_none() || q.get("range").is_some(),
            r#""step" needs a "range""#
        );
        ensure!(
            q.get("range").is_none() || kind == "number",
            r#"an estimate is {{"type": "number", "range": [low, high]}}"#
        );
        let unit = match q.get("unit") {
            None | Some(Json::Null) => None,
            Some(Json::Str(unit)) => Some(unit.clone()),
            Some(_) => bail!(r#""unit" is text"#),
        };
        Ok(match kind {
            "noul" => {
                let criteria = match criteria {
                    None => None,
                    Some(Json::Object(entries)) if entries.is_empty() => None,
                    Some(c @ Json::Object(_)) => {
                        let side = |name: &str| match c.get(name) {
                            None | Some(Json::Null) => Ok("None".to_owned()),
                            Some(Json::Str(text)) => Ok(text.clone()),
                            Some(_) => bail!(
                                r#"a noul's criteria are text: {{"true": "...", "false": "..."}}"#
                            ),
                        };
                        Some((side("true")?, side("false")?))
                    }
                    Some(_) => bail!(
                        r#"a noul's "criteria", when given, is {{"true": "what yes means", "false": "what no means"}}"#
                    ),
                };
                Self::Noul {
                    instructions,
                    criteria,
                }
            }
            "choice" => {
                let Some(Json::Object(entries)) = criteria else {
                    bail!(
                        r#"a choice's "criteria" names the options: {{"billing": "Charges and refunds", "shipping": "Delivery"}}"#
                    )
                };
                ensure!(!entries.is_empty(), "choice with no options");
                let criteria = entries
                    .iter()
                    .map(|(name, text)| match text {
                        Json::Null => Ok((name.clone(), None)),
                        Json::Str(text) => Ok((name.clone(), Some(text.clone()))),
                        _ => bail!("option {name:?}: its description is text or null"),
                    })
                    .collect::<Result<Vec<_>>>()?;
                Self::Choice {
                    instructions,
                    criteria,
                    multiple,
                }
            }
            "score" => {
                let levels = match criteria {
                    Some(Json::Array(items)) if (2..=10).contains(&items.len()) => items
                        .iter()
                        .map(|i| i.as_str().map(str::to_owned))
                        .collect::<Option<Vec<_>>>(),
                    _ => None,
                };
                Self::Score {
                    instructions,
                    criteria: levels.context(r#"a score's "criteria" lists 2 to 10 levels, lowest first: ["Can wait", "Today", "Now"]"#)?,
                }
            }
            "number" if q.get("range").is_some() => {
                let bounds = match q.get("range") {
                    Some(Json::Array(b)) if b.len() == 2 => (b[0].as_f64(), b[1].as_f64()),
                    _ => (None, None),
                };
                let (Some(low), Some(high)) = bounds else {
                    bail!(
                        r#"an estimate is {{"type": "number", "range": [low, high]}}, with an optional "step""#
                    )
                };
                let step = match q.get("step") {
                    None | Some(Json::Null) => None,
                    Some(step) => Some(step.as_f64().context(r#""step" is a number"#)?),
                };
                let divides = |step: f64| {
                    let count = (high - low) / step;
                    step > 0.
                        && (count - count.round()).abs()
                            <= 1e-9 * count.abs().max(count.round().abs())
                };
                ensure!(
                    low < high && step.is_none_or(divides),
                    "an estimate's range runs low to high and its step, above 0, divides it"
                );
                let size = step.map_or(128, |step| {
                    (((high - low) / step).round() as usize + 1).min(128)
                });
                Self::Estimate {
                    instructions,
                    grid: Grid::new(low, high, size)?,
                    unit,
                }
            }
            "text" => Self::Read {
                instructions,
                kind: ReadKind::Text,
                unit,
            },
            "number" => Self::Read {
                instructions,
                kind: ReadKind::Number,
                unit,
            },
            "date" => Self::Read {
                instructions,
                kind: ReadKind::Date,
                unit,
            },
            "box" => Self::Box { instructions },
            _ => {
                let most = if settings.prompt == 1 {
                    INSTANCES
                } else {
                    MAX_INSTANCES
                };
                let max = match q.get("max") {
                    None => None,
                    Some(Json::Int(n)) if (1..=most as i64).contains(n) => Some(*n as usize),
                    Some(_) => bail!(r#""max" belongs to boxes, a whole number from 1 to {most}"#),
                };
                Self::Boxes { instructions, max }
            }
        })
    }
    fn instructions(&self) -> &str {
        match self {
            Self::Noul { instructions, .. }
            | Self::Choice { instructions, .. }
            | Self::Score { instructions, .. }
            | Self::Read { instructions, .. }
            | Self::Estimate { instructions, .. }
            | Self::Box { instructions }
            | Self::Boxes { instructions, .. } => instructions,
        }
    }
    /// A typed answer keeps its raw probabilities: its own table calibrates it.
    pub fn typed(&self) -> bool {
        !matches!(
            self,
            Self::Noul { .. }
                | Self::Score { .. }
                | Self::Choice {
                    multiple: false,
                    ..
                }
        )
    }
    pub fn needs_picture(&self) -> bool {
        matches!(self, Self::Box { .. } | Self::Boxes { .. })
    }
    fn bucket(&self) -> String {
        match self {
            Self::Noul { .. } => "noul".into(),
            Self::Score { criteria, .. } => format!("score:{}", criteria.len()),
            Self::Choice { criteria, .. } => match criteria.len() {
                0..=3 => "choice:small",
                4..=8 => "choice:mid",
                9..=26 => "choice:large",
                _ => "choice:inventory",
            }
            .into(),
            _ => "choice:inventory".into(),
        }
    }
}

// ------------------------------------------------------------------ checkpoint settings

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub codes: u8,
    pub numbers: bool,
    pub set_size: usize,
    pub max_pixels: u64,
    pub interp_edges: bool,
    pub prompt: u8,
}
impl Settings {
    /// The `d1_*` record of a checkpoint's config; `None` for a checkpoint that is not a decision model.
    pub fn from_config(config: &serde_json::Value) -> Result<Option<Self>> {
        let Some(attention) = config.get("d1_attention") else {
            return Ok(None);
        };
        ensure!(
            attention == "set",
            "this engine serves d1 checkpoints trained with `set` attention only"
        );
        let set_size = config["d1_set_size"].as_u64().unwrap_or(4) as usize;
        ensure!(
            (2..=TRAINED_SET_SIZE).contains(&set_size),
            "d1_set_size must be between 2 and {TRAINED_SET_SIZE}"
        );
        let codes = config["d1_codes"].as_u64().unwrap_or(1);
        let prompt = config["d1_prompt"].as_u64().unwrap_or(1);
        ensure!(
            matches!(codes, 1 | 2) && matches!(prompt, 1 | 2),
            "unsupported d1 code scheme or prompt version"
        );
        let edges = config["d1_box_edges"].as_str().unwrap_or("median");
        ensure!(
            matches!(edges, "median" | "interp"),
            "unsupported d1_box_edges {edges:?}"
        );
        Ok(Some(Self {
            codes: codes as u8,
            numbers: config["d1_numbers"].as_bool().unwrap_or(false),
            set_size,
            max_pixels: config["d1_vision_max_pixels"]
                .as_u64()
                .unwrap_or(1024 * 1024),
            interp_edges: edges == "interp",
            prompt: prompt as u8,
        }))
    }
}

/// `calibration.json`: a temperature for option scores, monotone tables for typed probabilities.
#[derive(Clone, Debug, Default, Deserialize)]
pub struct Calibration {
    #[serde(default)]
    temperatures: std::collections::BTreeMap<String, f64>,
    #[serde(default = "one")]
    default: f64,
    #[serde(default)]
    tables: std::collections::BTreeMap<String, Vec<[f64; 2]>>,
}
fn one() -> f64 {
    1.
}
impl Calibration {
    pub fn load(path: &Path) -> Result<Self> {
        let calibration: Self =
            serde_json::from_slice(&std::fs::read(path).context("reading calibration.json")?)
                .context("invalid calibration.json")?;
        let finite = |t: &f64| t.is_finite() && *t > 0.;
        ensure!(
            finite(&calibration.default)
                && calibration.temperatures.values().all(finite)
                && calibration
                    .tables
                    .values()
                    .flatten()
                    .flatten()
                    .all(|v| v.is_finite()),
            "calibration.json holds a non-finite or non-positive value"
        );
        Ok(calibration)
    }
    pub fn neutral() -> Self {
        Self {
            default: 1.,
            ..Self::default()
        }
    }
    fn temperature(&self, q: &Question) -> f64 {
        self.temperatures
            .get(&q.bucket())
            .copied()
            .unwrap_or(self.default)
    }
    fn through(&self, name: &str, value: f64) -> f64 {
        through(value, self.tables.get(name).map(Vec::as_slice))
    }
    fn level(&self, name: &str, share: f64) -> f64 {
        match self.tables.get(name) {
            Some(table) => through(
                share,
                Some(&table.iter().map(|[x, y]| [*y, *x]).collect::<Vec<_>>()),
            ),
            None => share,
        }
    }
}

/// `value` through increasing (raw, calibrated) knots, linear between them, flat past the ends.
fn through(value: f64, knots: Option<&[[f64; 2]]>) -> f64 {
    let Some(knots) = knots.filter(|k| !k.is_empty()) else {
        return value;
    };
    let index = knots.partition_point(|k| k[0] < value);
    if index == 0 {
        knots[0][1]
    } else if index == knots.len() {
        knots[index - 1][1]
    } else {
        let ([a, pa], [b, pb]) = (knots[index - 1], knots[index]);
        pa + (value - a) * (pb - pa) / (b - a)
    }
}

// ------------------------------------------------------------------ tokenizer view

/// What one answer position reads: vocabulary tokens grouped by option, or a state position.
#[derive(Clone, Debug, PartialEq)]
pub enum Readout {
    Tokens {
        groups: Vec<Vec<u32>>,
        temperature: f64,
    },
    Start,
    End,
}

/// Token ids and their byte offsets in the encoded text.
type Encoded = (Vec<u32>, Vec<(usize, usize)>);
/// A packed sequence: ids, state length, each token's question, each question's slots.
type Packed = (Vec<u32>, usize, Vec<i32>, Vec<Vec<usize>>);
/// A read's text, its character span and its JSON path.
pub type ReadValue = (String, [usize; 2], Option<String>);

pub struct Prompter<'a> {
    tokenizer: &'a Tokenizer,
    specials: Vec<String>,
    bos: String,
    pub settings: Settings,
    pub calibration: Calibration,
}

impl<'a> Prompter<'a> {
    pub fn new(
        tokenizer: &'a Tokenizer,
        bos: &str,
        settings: Settings,
        calibration: Calibration,
    ) -> Self {
        let mut specials: Vec<String> = tokenizer
            .get_added_tokens_decoder()
            .into_values()
            .filter(|t| t.special && t.content.chars().count() > 1 && !t.content.contains(BREAK))
            .map(|t| t.content)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        specials.sort();
        specials.sort_by_key(|name| std::cmp::Reverse(name.chars().count()));
        Self {
            tokenizer,
            specials,
            bos: bos.to_owned(),
            settings,
            calibration,
        }
    }
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        Ok(self.encode_offsets(text)?.0)
    }
    fn encode_offsets(&self, text: &str) -> Result<Encoded> {
        let encoded = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("encoding prompt: {e}"))?;
        Ok((encoded.get_ids().to_vec(), encoded.get_offsets().to_vec()))
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| anyhow::anyhow!("decoding tokens: {e}"))
    }
    fn single(&self, text: &str) -> Result<Option<u32>> {
        let ids = self.encode(text)?;
        Ok((ids.len() == 1).then(|| ids[0]))
    }
    /// Distinct single-token forms of `texts`, in order.
    fn ids(&self, texts: &[&str]) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for text in texts {
            if let Some(id) = self.single(text)?.filter(|id| !out.contains(id)) {
                out.push(id);
            }
        }
        Ok(out)
    }

    /// `text` with a zero-width space after the first character of every special token it spells,
    /// and for each byte of the result the byte of `text` it came from.
    fn plain_offsets(&self, text: &str) -> (String, Vec<usize>) {
        let mut text = text.to_owned();
        let mut map: Vec<usize> = (0..=text.len()).collect();
        loop {
            let mut out = String::with_capacity(text.len());
            let mut next = Vec::with_capacity(map.len());
            let (mut at, mut found) = (0, false);
            while at < text.len() {
                let rest = &text[at..];
                let width = rest.chars().next().expect("inside the text").len_utf8();
                let special = self
                    .specials
                    .iter()
                    .find(|name| rest.starts_with(name.as_str()));
                out.push_str(&rest[..width]);
                next.extend(&map[at..at + width]);
                if let Some(name) = special {
                    found = true;
                    out.push(BREAK);
                    next.extend([map[at + width]; 3]);
                    out.push_str(&rest[width..name.len()]);
                    next.extend(&map[at + width..at + name.len()]);
                    at += name.len();
                } else {
                    at += width;
                }
            }
            next.push(map[text.len()]);
            (text, map) = (out, next);
            if !found {
                return (text, map);
            }
        }
    }
    fn plain(&self, text: &str) -> String {
        self.plain_offsets(text).0
    }

    // -------------------------------------------------------------- option codes

    fn option_codes(&self, labels: &[&str]) -> Vec<String> {
        let labs: Vec<&str> = labels.iter().map(|l| python_strip(l)).collect();
        let letter = |l: &&str| {
            let mut chars = l.chars();
            matches!((chars.next(), chars.next()), (Some(c), None) if c.is_alphabetic())
        };
        let letters = || (b'A'..=b'Z').map(|c| (c as char).to_string());
        if !labs.is_empty() && labs.iter().all(letter) {
            labs.iter().map(|l| (*l).to_owned()).collect()
        } else if labs.len() <= 26 {
            letters().take(labs.len()).collect()
        } else if self.settings.codes == 2 {
            letters()
                .chain(letters().flat_map(|a| letters().map(move |b| format!("{a}{b}"))))
                .take(labs.len())
                .collect()
        } else {
            (0..labs.len()).map(|i| format!("{i:02}")).collect()
        }
    }

    /// Every label's distinct single-token code and its token.
    fn aliases(&self, labels: &[&str]) -> Result<Vec<(String, u32)>> {
        let upper = || (b'A'..=b'Z').map(|c| (c as char).to_string());
        let pool = || {
            upper()
                .chain((0..100).map(|i| format!("{i:02}")))
                .chain((b'a'..=b'z').map(|c| (c as char).to_string()))
                .chain((0..200).map(|i| format!("#{i}")))
                .chain(upper().flat_map(move |a| upper().map(move |b| format!("{a}{b}"))))
        };
        let mut out: Vec<(String, u32)> = Vec::new();
        let take = |raw: &str, out: &mut Vec<(String, u32)>| -> Result<bool> {
            match self.single(raw)? {
                Some(id) if out.iter().all(|(_, used)| *used != id) => {
                    out.push((raw.to_owned(), id));
                    Ok(true)
                }
                _ => Ok(false),
            }
        };
        for code in self.option_codes(labels) {
            if take(&code, &mut out)? {
                continue;
            }
            let mut placed = false;
            for raw in pool() {
                if take(&raw, &mut out)? {
                    placed = true;
                    break;
                }
            }
            ensure!(
                placed,
                "no single-token alias left for {} options",
                labels.len()
            );
        }
        Ok(out)
    }

    /// Each grid value times `scale` as one token; `None` when a value is no integer or takes several.
    fn number_ids(&self, grid: &Grid, scale: f64) -> Result<Option<Vec<Vec<u32>>>> {
        let mut groups = Vec::with_capacity(grid.size);
        for value in grid.values() {
            let scaled = value * scale;
            let n = scaled.round_ties_even();
            if (n - scaled).abs() > (1e-9 * n.abs().max(scaled.abs())).max(1e-6) {
                return Ok(None);
            }
            let text = format!("{}", n as i64);
            let group = self.ids(&[&text, &format!(" {text}")])?;
            if group.is_empty() || self.single(&text)?.is_none() {
                return Ok(None);
            }
            groups.push(group);
        }
        Ok(Some(groups))
    }

    fn edge_grid(&self) -> Result<(Grid, f64)> {
        let thousandths = Grid::new(0., 0.999, 1000)?;
        ensure!(
            self.settings.numbers && self.number_ids(&thousandths, 1000.)?.is_some(),
            "this engine reads box edges as the tokenizer's numbers only"
        );
        Ok((thousandths, 1000.))
    }

    fn choice_groups(&self, labels: &[&str]) -> Result<Vec<Vec<u32>>> {
        let mut groups = Vec::new();
        for (code, id) in self.aliases(labels)? {
            let mut group = vec![id];
            group.extend(
                self.ids(&[&format!(" {code}")])?
                    .into_iter()
                    .filter(|i| *i != id),
            );
            groups.push(group);
        }
        ensure!(!groups.is_empty(), "choice with no options");
        Ok(groups)
    }
    fn noul_groups(&self) -> Result<Vec<Vec<u32>>> {
        let (yes, no) = (
            self.ids(&["yes", "Yes", "YES"])?,
            self.ids(&["no", "No", "NO"])?,
        );
        ensure!(
            !yes.is_empty() && !no.is_empty(),
            "tokenizer has no single-token yes/no"
        );
        Ok(vec![yes, no])
    }
    fn digit_groups(&self, levels: usize) -> Result<Vec<Vec<u32>>> {
        (0..levels)
            .map(|i| {
                let group = self.ids(&[&i.to_string()])?;
                ensure!(
                    !group.is_empty(),
                    "score with {levels} levels needs single-token digits"
                );
                Ok(group)
            })
            .collect()
    }
    /// A grid's slot: the tokenizer's numbers where the checkpoint reads them so, else digit or option codes.
    fn grid_groups(&self, grid: &Grid, scale: f64) -> Result<(Vec<Vec<u32>>, bool)> {
        if self.settings.numbers
            && let Some(groups) = self.number_ids(grid, scale)?
        {
            return Ok((groups, true));
        }
        if grid.size <= 10 {
            return Ok((self.digit_groups(grid.size)?, false));
        }
        let labels: Vec<String> = (0..grid.size).map(|i| i.to_string()).collect();
        Ok((
            self.choice_groups(&labels.iter().map(String::as_str).collect::<Vec<_>>())?,
            false,
        ))
    }

    /// Each answer position's readout, in question order.
    pub fn slots(&self, q: &Question) -> Result<Vec<Readout>> {
        let raw = |groups| Readout::Tokens {
            groups,
            temperature: 1.,
        };
        let box_slots = || -> Result<Vec<Readout>> {
            let (grid, scale) = self.edge_grid()?;
            let edge = self.grid_groups(&grid, scale)?.0;
            Ok(std::iter::once(raw(self.noul_groups()?))
                .chain((0..4).map(|_| raw(edge.clone())))
                .collect())
        };
        let temperature = self.calibration.temperature(q);
        Ok(match q {
            Question::Read { .. } => vec![Readout::Start, Readout::End],
            Question::Estimate { grid, .. } => vec![raw(self.grid_groups(grid, 1.)?.0)],
            Question::Box { .. } => box_slots()?,
            Question::Boxes { max, .. } => {
                let block = box_slots()?;
                (0..max.unwrap_or(INSTANCES))
                    .flat_map(|_| block.clone())
                    .collect()
            }
            Question::Choice {
                criteria,
                multiple: true,
                ..
            } => {
                let groups = self.noul_groups()?;
                criteria.iter().map(|_| raw(groups.clone())).collect()
            }
            Question::Choice { criteria, .. } => {
                let labels: Vec<&str> = criteria.iter().map(|(name, _)| name.as_str()).collect();
                vec![Readout::Tokens {
                    groups: self.choice_groups(&labels)?,
                    temperature,
                }]
            }
            Question::Noul { .. } => vec![Readout::Tokens {
                groups: self.noul_groups()?,
                temperature,
            }],
            Question::Score { criteria, .. } => {
                vec![Readout::Tokens {
                    groups: self.digit_groups(criteria.len())?,
                    temperature,
                }]
            }
        })
    }

    // -------------------------------------------------------------- rendering

    fn edge_legend(&self) -> Result<String> {
        self.edge_grid()?;
        Ok("Each edge is a number from 0 to 999, in thousandths of the picture: top and bottom from its top edge, \
            left and right from its left edge."
            .into())
    }

    fn question_block(&self, q: &Question) -> Result<String> {
        let said = q.instructions();
        Ok(match q {
            Question::Box { .. } => format!(
                "{said}\n\nPresence is yes or no. {} Reply with yes or no at present, then one number at top, bottom, left and right.",
                self.edge_legend()?
            ),
            Question::Boxes { .. } => format!(
                "{said}\n\nEvery one of them, {}in reading order (left edge, then top). Present k is yes when there are at least k. {} Reply with yes or no at each present, then its top, bottom, left and right.",
                if self.settings.prompt == 1 {
                    format!("{INSTANCES} at most, ")
                } else {
                    String::new()
                },
                self.edge_legend()?
            ),
            Question::Estimate { grid, unit, .. } => {
                let unit = unit.as_ref().map_or(String::new(), |u| format!(" {u}"));
                let (low, high, step) = (
                    python_g(grid.low),
                    python_g(grid.high),
                    python_g(grid.step()),
                );
                if self.grid_groups(grid, 1.)?.1 {
                    format!(
                        "{said}\n\nEstimate from {low} to {high}{unit}, in steps of {step}. Reply with the number only."
                    )
                } else {
                    let (first, last) = if grid.size <= 10 {
                        ("0".to_owned(), (grid.size - 1).to_string())
                    } else {
                        let labels: Vec<String> = (0..grid.size).map(|i| i.to_string()).collect();
                        let codes =
                            self.aliases(&labels.iter().map(String::as_str).collect::<Vec<_>>())?;
                        (codes[0].0.clone(), codes[codes.len() - 1].0.clone())
                    };
                    format!(
                        "{said}\n\nEstimate from {low} to {high}{unit} on {} equally spaced values, step {step}: {first} is {low}, {last} is {high}. Reply with the code only.",
                        grid.size
                    )
                }
            }
            Question::Read { kind, unit, .. } => {
                let unit = match unit {
                    Some(unit) if *kind == ReadKind::Number && !unit.is_empty() => {
                        format!(" (unit: {unit})")
                    }
                    _ => String::new(),
                };
                format!(
                    "{said}\n\nRead the {}{unit} from the state, or nothing if it is absent.",
                    kind.name()
                )
            }
            Question::Choice {
                criteria,
                multiple: true,
                ..
            } => {
                let lines: Vec<String> = criteria
                    .iter()
                    .map(|(name, text)| {
                        let mut line = String::new();
                        python_string(name, &mut line);
                        format!(
                            "{line}: {}",
                            text.as_deref().filter(|t| !t.is_empty()).unwrap_or(name)
                        )
                    })
                    .collect();
                format!(
                    "{said}\n\nOptions:\n{}\n\nReply with yes or no for each option.",
                    lines.join("\n")
                )
            }
            Question::Choice { criteria, .. } => {
                let labels: Vec<&str> = criteria.iter().map(|(name, _)| name.as_str()).collect();
                let lines: Vec<String> = self
                    .aliases(&labels)?
                    .iter()
                    .zip(criteria)
                    .map(|((code, _), (label, text))| {
                        match text.as_deref().filter(|t| !t.is_empty()) {
                            Some(text) => format!("{code} {text}"),
                            None => format!("{code} {}", label.replace('_', " ")),
                        }
                    })
                    .collect();
                format!(
                    "{said}\n\nOptions:\n{}\n\nReply with the option code only.",
                    lines.join("\n")
                )
            }
            Question::Noul { criteria, .. } => {
                let extra = criteria
                    .as_ref()
                    .map_or(String::new(), |(yes, no)| format!("\nYes: {yes}\nNo: {no}"));
                format!("{said}{extra}\n\nReply with yes or no only.")
            }
            Question::Score { criteria, .. } => {
                let legend: Vec<String> = criteria
                    .iter()
                    .enumerate()
                    .map(|(i, name)| format!("{i} {name}"))
                    .collect();
                format!(
                    "{said}\n\n{}\n\nReply with a single digit 0-{} only.",
                    legend.join("\n"),
                    criteria.len() - 1
                )
            }
        })
    }

    /// The state as the block before `QUESTION:`, and its string values' places when it is JSON.
    fn state_block(&self, state: &Json) -> (String, Vec<(usize, usize, String)>) {
        let mut strings = Vec::new();
        let mut text = match state {
            Json::Str(text) => text.clone(),
            other => {
                let mut out = String::new();
                python_dump(other, 0, "", &mut out, &mut strings);
                out
            }
        };
        text.push_str("\n\n");
        (text, strings)
    }

    /// The state as the prompt renders it, without the blank line after it (a read's context).
    pub fn state_text(&self, state: Option<&Json>) -> String {
        state.map_or(String::new(), |state| {
            let mut text = self.state_block(state).0;
            text.truncate(text.len() - 2);
            text
        })
    }

    /// Everything before the question: the media's markup at the head of the user turn, then the state.
    pub fn prefix(&self, state: Option<&Json>, media: &str) -> String {
        let body = state.map_or(String::new(), |state| {
            format!("{}\n{HEADER}", self.plain(&self.state_block(state).0))
        });
        format!("{}{IM_START}user\n{media}{body}", self.bos)
    }

    /// The question and the assistant header up to the answer slots, and each slot's token offset from the last token.
    pub fn suffix(&self, q: &Question) -> Result<(String, Vec<isize>)> {
        let body = self.question_block(q)?;
        let head = format!("{}{IM_END}\n{IM_START}assistant\n{LEAD}", self.plain(&body));
        let blocks = |max: &Option<usize>| max.unwrap_or(INSTANCES);
        match q {
            Question::Read { .. } => self.marked(head, &["start".into(), "end".into()], ""),
            Question::Box { .. } => self.marked(
                head,
                &std::iter::once("present")
                    .chain(EDGES)
                    .map(str::to_owned)
                    .collect::<Vec<_>>(),
                "",
            ),
            Question::Boxes { max, .. } => {
                let names: Vec<String> = (1..=blocks(max))
                    .flat_map(|k| {
                        std::iter::once("present")
                            .chain(EDGES)
                            .map(move |name| format!("{name} {k}"))
                    })
                    .collect();
                let after = if blocks(max) < INSTANCES || self.settings.prompt > 1 {
                    "\n"
                } else {
                    ""
                };
                self.marked(head, &names, after)
            }
            Question::Choice {
                criteria,
                multiple: true,
                ..
            } => {
                let names: Vec<String> = criteria
                    .iter()
                    .map(|(name, _)| {
                        let mut out = String::new();
                        python_string(name, &mut out);
                        out
                    })
                    .collect();
                self.marked(head, &names, "")
            }
            _ => Ok((head, vec![0])),
        }
    }

    fn marked(
        &self,
        mut text: String,
        names: &[String],
        after: &str,
    ) -> Result<(String, Vec<isize>)> {
        ensure!(
            !names.is_empty()
                && names.iter().collect::<HashSet<_>>().len() == names.len()
                && names.iter().all(|n| !n.is_empty() && !n.contains('\n')),
            "slot names must be distinct nonempty single lines"
        );
        let mut ends = Vec::with_capacity(names.len());
        for (index, name) in names.iter().enumerate() {
            if index > 0 {
                text.push('\n');
            }
            text.push_str(&self.plain(name));
            text.push(':');
            ends.push(text.len());
        }
        if !after.is_empty() {
            let ids = self.encode(&format!("{text}{after}"))?;
            if python_strip_end(&self.decode(&ids[ids.len() - 1..])?).ends_with(':') {
                text.push_str(after);
            }
        }
        let (ids, offsets) = self.encode_offsets(&text)?;
        let mut slots = Vec::with_capacity(ends.len());
        for chars in ends {
            let mut matching = offsets
                .iter()
                .enumerate()
                .filter(|(_, (start, stop))| *start < chars && chars <= *stop);
            let (Some((index, (_, stop))), None) = (matching.next(), matching.next()) else {
                bail!("a slot marker must have an exact colon token position")
            };
            ensure!(
                blank(&text[chars..*stop])
                    && python_strip_end(&self.decode(&ids[index..=index])?).ends_with(':'),
                "a slot marker token must end with its colon and optional whitespace"
            );
            slots.push(index as isize - ids.len() as isize + 1);
        }
        Ok((text, slots))
    }

    /// `set`'s sequence for one state: the prefix and first question tokenized as its lone prompt is,
    /// each further question its own group. Returns ids, the state's length, each token's question
    /// (1..N, -1 on the state) and each question's answer positions.
    fn pack(
        &self,
        prefix: &str,
        suffixes: &[String],
        header: &str,
        offsets: &[Vec<isize>],
    ) -> Result<Packed> {
        ensure!(
            !suffixes.is_empty() && suffixes.len() == offsets.len(),
            "a question set cannot be empty"
        );
        let positions = |own: &[isize], length: usize, end: usize| -> Result<Vec<usize>> {
            ensure!(
                own.last() == Some(&0)
                    && own.windows(2).all(|w| w[0] < w[1])
                    && own.iter().all(|o| -(length as isize) < *o),
                "slots must be increasing positions inside their question, ending at its last token"
            );
            Ok(own.iter().map(|o| (end as isize + o) as usize).collect())
        };
        let mut ids = self.encode(&format!("{prefix}{}", suffixes[0]))?;
        let first = self.encode(&suffixes[0])?.len();
        ensure!(
            first <= ids.len(),
            "the first question does not close its prompt"
        );
        let state = ids.len() - first;
        let mut question = vec![-1; state];
        question.resize(ids.len(), 1);
        let mut ends = vec![positions(&offsets[0], first, ids.len() - 1)?];
        for (k, suffix) in suffixes.iter().enumerate().skip(1) {
            let own = self.encode(&format!("{header}{suffix}"))?;
            ids.extend(&own);
            question.resize(ids.len(), k as i32 + 1);
            ends.push(positions(&offsets[k], own.len(), ids.len() - 1)?);
        }
        Ok((ids, state, question, ends))
    }

    /// A request's sequences: up to `set_size` questions after their state, each read at its slots.
    pub fn sequences(
        &self,
        state: Option<&Json>,
        questions: &[Question],
        media: &str,
    ) -> Result<Vec<Sequence>> {
        self.sequences_with_voices(state, questions, media, &vec![None; questions.len()])
    }

    /// `sequences`, each spoken question's clip markup in place of its instructions.
    pub fn sequences_with_voices(
        &self,
        state: Option<&Json>,
        questions: &[Question],
        media: &str,
        voices: &[Option<String>],
    ) -> Result<Vec<Sequence>> {
        ensure!(
            voices.len() == questions.len(),
            "one voice entry per question"
        );
        ensure!(
            !questions.is_empty(),
            "a request needs at least one question"
        );
        let prefix = self.prefix(state, media);
        let header = if state.is_some() { HEADER } else { "" };
        let mut suffixes = Vec::with_capacity(questions.len());
        let mut offsets = Vec::with_capacity(questions.len());
        for (q, voice) in questions.iter().zip(voices) {
            let (text, slots) = self.suffix(q)?;
            suffixes.push(match voice {
                Some(markup) => {
                    ensure!(
                        text.contains(SPOKEN),
                        "a spoken question renders no instructions"
                    );
                    text.replace(SPOKEN, markup)
                }
                None => text,
            });
            offsets.push(slots);
        }
        let mut out = Vec::new();
        for first in (0..questions.len()).step_by(self.settings.set_size) {
            let last = (first + self.settings.set_size).min(questions.len());
            let (ids, state_len, question, ends) = self.pack(
                &prefix,
                &suffixes[first..last],
                header,
                &offsets[first..last],
            )?;
            ensure!(
                ids.len() <= MAX_TOKENS,
                "a decision sequence holds {} tokens; this engine reads at most {MAX_TOKENS}",
                ids.len()
            );
            let positions = if questions[first..last]
                .iter()
                .any(|q| matches!(q, Question::Read { .. }))
            {
                Some(self.state_positions(
                    state,
                    &prefix,
                    &format!("{prefix}{}", suffixes[first]),
                )?)
            } else {
                None
            };
            let mut slots = Vec::new();
            for (k, q) in questions[first..last].iter().enumerate() {
                let readouts = self.slots(q)?;
                ensure!(
                    readouts.len() == ends[k].len(),
                    "a question's slots and readouts differ"
                );
                slots.extend(
                    readouts
                        .into_iter()
                        .zip(&ends[k])
                        .map(|(readout, end)| Slot {
                            question: first + k,
                            position: *end,
                            readout,
                        }),
                );
            }
            out.push(Sequence {
                ids,
                state_len,
                question,
                slots,
                positions,
            });
        }
        Ok(out)
    }

    /// Which tokens of a prompt are text of the state, and where: template, picture and question tokens are not.
    fn state_positions(
        &self,
        state: Option<&Json>,
        prefix: &str,
        full: &str,
    ) -> Result<StatePositions> {
        ensure!(
            full.starts_with(prefix),
            "the tokenized prompt must start with the state prefix"
        );
        let (body, strings) =
            state.map_or((String::new(), Vec::new()), |state| self.state_block(state));
        let (escaped, map) = self.plain_offsets(&body);
        let (start, text) = match state {
            None => (prefix.len(), String::new()),
            Some(_) => {
                let tail = format!("{escaped}\n{HEADER}");
                ensure!(
                    prefix.ends_with(&tail),
                    "state rendering differs from the prompt prefix"
                );
                (prefix.len() - tail.len(), body[..body.len() - 2].to_owned())
            }
        };
        let boundaries = if state.is_some() { map } else { vec![0] };
        let stop = start + boundaries.partition_point(|b| *b <= text.len()) - 1;
        let (_, offsets) = self.encode_offsets(full)?;
        let spans: Vec<Option<(usize, usize)>> = offsets
            .iter()
            .map(|&(a, b)| {
                let (lo, hi) = (a.max(start), b.min(stop));
                if lo < hi && blank(&full[a..lo]) && blank(&full[hi..b]) {
                    let (left, right) = (boundaries[lo - start], boundaries[hi - start]);
                    (left < right && right <= text.len() && !blank(&text[left..right]))
                        .then_some((left, right))
                } else {
                    None
                }
            })
            .collect();
        ensure!(
            spans.first().is_some_and(Option::is_none),
            "position 0 must be a template token for the nothing readout"
        );
        let paths = if matches!(state, Some(Json::Str(_)) | None) {
            Vec::new()
        } else {
            strings
        };
        Ok(StatePositions { text, spans, paths })
    }
}

fn python_strip_end(text: &str) -> &str {
    text.trim_end_matches(python_space)
}

#[derive(Clone, Debug)]
pub struct Slot {
    pub question: usize,
    pub position: usize,
    pub readout: Readout,
}

#[derive(Clone, Debug)]
pub struct Sequence {
    pub ids: Vec<u32>,
    pub state_len: usize,
    pub question: Vec<i32>,
    pub slots: Vec<Slot>,
    pub positions: Option<StatePositions>,
}

#[derive(Clone, Debug)]
pub struct StatePositions {
    pub text: String,
    pub spans: Vec<Option<(usize, usize)>>,
    pub paths: Vec<(usize, usize, String)>,
}

impl StatePositions {
    /// The keys a read's pointer scores: position 0 (nothing), then the state's text tokens.
    pub fn pointer_mask(&self) -> Vec<bool> {
        let valid: Vec<bool> = self
            .spans
            .iter()
            .enumerate()
            .map(|(i, s)| i == 0 || s.is_some())
            .collect();
        let last = valid.iter().rposition(|v| *v).unwrap_or(0);
        valid[..=last].to_vec()
    }
    pub fn characters(&self, byte: usize) -> usize {
        self.text[..byte].chars().count()
    }
    /// The text between two token positions, trimmed, and clipped to the one JSON string it sits in.
    pub fn read(&self, first: usize, last: usize, cap: usize) -> Result<Option<ReadValue>> {
        if first == 0 && last == 0 {
            return Ok(None);
        }
        ensure!(
            0 < first && first <= last && last < self.spans.len() && last - first < cap,
            "invalid pointer endpoints"
        );
        let (Some(a), Some(b)) = (self.spans[first], self.spans[last]) else {
            bail!("pointer leaves the state text")
        };
        let (mut start, mut end) = (a.0, b.1.max(a.0));
        while start < end && self.text[start..].chars().next().is_some_and(python_space) {
            start += self.text[start..]
                .chars()
                .next()
                .expect("checked")
                .len_utf8();
        }
        while end > start
            && self.text[..end]
                .chars()
                .next_back()
                .is_some_and(python_space)
        {
            end -= self.text[..end]
                .chars()
                .next_back()
                .expect("checked")
                .len_utf8();
        }
        let held: Vec<_> = self
            .paths
            .iter()
            .filter(|(s, e, _)| *s < end && start < *e)
            .collect();
        if let [(s, e, path)] = held.as_slice() {
            let (start, end) = (start.max(*s), end.min(*e));
            let escaped = &self.text[start..end.max(start)];
            let text = serde_json::from_str::<String>(&format!("\"{escaped}\""))
                .unwrap_or_else(|_| escaped.to_owned());
            return Ok(Some((
                text,
                [self.characters(start), self.characters(end.max(start))],
                Some(path.clone()),
            )));
        }
        Ok(Some((
            self.text[start..end].to_owned(),
            [self.characters(start), self.characters(end)],
            None,
        )))
    }
}

// ------------------------------------------------------------------ readouts

fn softmax(scores: &[f64]) -> Vec<f64> {
    let top = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scores.iter().map(|s| (s - top).exp()).collect();
    let total: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / total).collect()
}

/// Option probabilities from the log-probabilities at a slot: each option scores its best form.
pub fn probabilities(
    groups: &[Vec<u32>],
    temperature: f64,
    logz: impl Fn(u32) -> f64,
) -> Result<Vec<f64>> {
    let scores: Vec<f64> = groups
        .iter()
        .map(|g| {
            g.iter()
                .map(|id| logz(*id))
                .fold(f64::NEG_INFINITY, f64::max)
                / temperature
        })
        .collect();
    ensure!(
        !scores.is_empty() && scores.iter().all(|s| s.is_finite()),
        "non-finite option score"
    );
    Ok(softmax(&scores))
}

/// A read's span from its start and end scores over the state positions (`-inf` on the others):
/// (first, last) token positions, (0, 0) for nothing, and P(found).
pub fn span(start: &[f64], end: &[f64], cap: usize) -> Result<((usize, usize), f64)> {
    ensure!(
        !start.is_empty() && start.len() == end.len(),
        "empty pointer row"
    );
    let log_softmax = |row: &[f64]| -> Result<Vec<f64>> {
        let top = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        ensure!(
            top.is_finite() && row.iter().all(|v| !v.is_nan()),
            "non-finite pointer scores"
        );
        let total = row.iter().map(|v| (v - top).exp()).sum::<f64>().ln() + top;
        Ok(row.iter().map(|v| v - total).collect())
    };
    let (start, end) = (log_softmax(start)?, log_softmax(end)?);
    let found = -start[0].exp_m1();
    if start.len() == 1 || found < 0.5 {
        return Ok(((0, 0), found));
    }
    let cap = cap.min(start.len() - 1);
    let mut best = (f64::NEG_INFINITY, 0, 0);
    for (first, opening) in start.iter().enumerate().skip(1) {
        let (mut top, mut at) = (f64::NEG_INFINITY, first);
        for (last, value) in end.iter().enumerate().skip(first).take(cap) {
            if *value > top {
                (top, at) = (*value, last);
            }
        }
        if opening + top > best.0 {
            best = (opening + top, first, at);
        }
    }
    ensure!(best.1 > 0, "no readable span");
    Ok(((best.1, best.2), found))
}

/// What the engine read for one question, before it is shaped as an answer.
#[derive(Clone, Debug)]
pub enum Reading {
    Options(Vec<f64>),
    Blocks(Vec<Vec<f64>>),
    Span {
        found: f64,
        value: Option<(String, [usize; 2], Option<String>)>,
    },
}

fn number(value: f64) -> Json {
    Json::Float(value)
}
fn labelled(names: impl Iterator<Item = String>, values: &[f64]) -> Json {
    Json::Object(
        names
            .zip(values)
            .map(|(name, p)| (name, number(*p)))
            .collect(),
    )
}
fn first_max(values: &[f64]) -> usize {
    let mut best = 0;
    for (index, value) in values.iter().enumerate() {
        if *value > values[best] {
            best = index;
        }
    }
    best
}

/// Radical inverses in bases 2, 3, 5, 7: the same 512 draws every call.
fn halton(index: usize) -> [f64; 4] {
    [2usize, 3, 5, 7].map(|base| {
        let (mut i, mut value, mut scale) = (index, 0f64, 1f64);
        while i > 0 {
            scale /= base as f64;
            value += (i % base) as f64 * scale;
            i /= base;
        }
        value
    })
}

fn ordered(a: f64, b: f64, step: f64) -> (f64, f64) {
    if a < b {
        return (a, b);
    }
    let half = step / 2.;
    let middle = ((a + b) / 2.).max(half).min(1. - half);
    (middle - half, middle + half)
}
fn rectangle(edges: [f64; 4], step: f64) -> [f64; 4] {
    let (top, bottom) = ordered(edges[0], edges[1], step);
    let (left, right) = ordered(edges[2], edges[3], step);
    [top, bottom, left, right]
}
fn iou(a: [f64; 4], b: [f64; 4]) -> f64 {
    let area = |r: [f64; 4]| (r[1] - r[0]) * (r[3] - r[2]);
    let overlap =
        (a[1].min(b[1]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[2].max(b[2])).max(0.);
    overlap / (area(a) + area(b) - overlap)
}

fn edge_grid_of(distribution: &[f64]) -> Result<Grid> {
    if distribution.len() == 1000 {
        Grid::new(0., 0.999, 1000)
    } else {
        Grid::new(0., 1., 128)
    }
}

/// The box at the four edge distributions' medians, and the share of boxes drawn from them that overlap it.
fn box_of(edges: &[Vec<f64>], interp: bool) -> Result<([f64; 4], f64)> {
    ensure!(
        edges.len() == 4
            && edges
                .iter()
                .all(|e| e.len() == edges[0].len() && e.len() >= 2),
        "a box reads four edges"
    );
    let grid = edge_grid_of(&edges[0])?;
    let (values, step) = (grid.values(), grid.step());
    let mut at = [0f64; 4];
    for (slot, row) in at.iter_mut().zip(edges) {
        *slot = if interp {
            let total = fsum(row.iter().copied());
            let mut mass = 0.;
            let mut crossing = grid.high;
            for (value, p) in values.iter().zip(row) {
                let share = p / total;
                if share > 0. && mass + share >= 0.5 {
                    crossing = (value - step / 2. + (0.5 - mass) / share * step)
                        .max(grid.low)
                        .min(grid.high);
                    break;
                }
                mass += share;
            }
            crossing
        } else {
            grid.read(row, |q| q).1
        };
    }
    let rect = rectangle(at, step);
    let cdfs: Vec<Vec<f64>> = edges
        .iter()
        .map(|row| {
            let total = fsum(row.iter().copied());
            row.iter()
                .scan(0., |mass, p| {
                    *mass += p / total;
                    Some(*mass)
                })
                .collect()
        })
        .collect();
    let hits = (1..=512)
        .filter(|&i| {
            let point = halton(i);
            let mut drawn = [0f64; 4];
            for ((slot, cdf), u) in drawn.iter_mut().zip(&cdfs).zip(point) {
                *slot = values[cdf.partition_point(|c| *c < u).min(grid.size - 1)];
            }
            iou(rectangle(drawn, step), rect) >= 0.5
        })
        .count();
    Ok((rect, hits as f64 / 512.))
}
fn box_json(rect: [f64; 4]) -> Json {
    Json::Object(
        EDGES
            .iter()
            .zip(rect)
            .map(|(name, v)| ((*name).to_owned(), number(v)))
            .collect(),
    )
}

/// One question's answer in the API's shape.
/// A parsed number as the reference's Python value serializes: an integer when integral.
fn parsed_number(value: crate::d1_parsing::Number) -> Json {
    use crate::d1_parsing::Number;
    match value {
        Number::Int(v) => i64::try_from(v)
            .map(Json::Int)
            .or_else(|_| u64::try_from(v).map(Json::UInt))
            .unwrap_or(Json::Float(v as f64)),
        // Beyond 128 bits: JSON readers take it as a double anyway.
        Number::BigInt(digits) => digits.parse::<f64>().map_or(Json::Null, Json::Float),
        Number::Float(v) => Json::Float(v),
    }
}

/// One question's answer in the API's shape. `context` is the rendered state, which the
/// strict number and date readers use to settle decimal marks and day/month order.
pub fn answer(
    q: &Question,
    reading: Reading,
    calibration: &Calibration,
    interp_edges: bool,
    context: &str,
) -> Result<Json> {
    Ok(match (q, reading) {
        (Question::Noul { .. }, Reading::Options(p)) => Json::object(vec![
            ("type", Json::Str("noul".into())),
            ("noul", number(p[0])),
        ]),
        (
            Question::Choice {
                criteria,
                multiple: true,
                ..
            },
            Reading::Options(p),
        ) => {
            let p: Vec<f64> = p
                .iter()
                .map(|p| calibration.through("multiple", *p))
                .collect();
            let chosen = criteria
                .iter()
                .zip(&p)
                .filter(|(_, p)| **p >= 0.5)
                .map(|((name, _), _)| Json::Str(name.clone()))
                .collect();
            Json::object(vec![
                ("type", Json::Str("choice".into())),
                ("choices", Json::Array(chosen)),
                (
                    "probabilities",
                    labelled(criteria.iter().map(|(name, _)| name.clone()), &p),
                ),
            ])
        }
        (Question::Choice { criteria, .. }, Reading::Options(p)) => {
            let pick = first_max(&p);
            Json::object(vec![
                ("type", Json::Str("choice".into())),
                ("choice", Json::Str(criteria[pick].0.clone())),
                ("confidence", number(p[pick])),
                (
                    "probabilities",
                    labelled(criteria.iter().map(|(name, _)| name.clone()), &p),
                ),
            ])
        }
        (Question::Score { criteria, .. }, Reading::Options(p)) => {
            let total: f64 = p.iter().sum();
            let total = if total == 0. { 1. } else { total };
            let mut mass = 0.;
            let level = p
                .iter()
                .position(|p| {
                    mass += p / total;
                    mass >= 0.5
                })
                .unwrap_or(p.len() - 1);
            Json::object(vec![
                ("type", Json::Str("score".into())),
                ("score", number(level as f64)),
                ("confidence", number(p[first_max(&p)])),
                (
                    "probabilities",
                    labelled((0..p.len()).map(|i| i.to_string()), &p),
                ),
                (
                    "legend",
                    Json::Object(
                        criteria
                            .iter()
                            .enumerate()
                            .map(|(i, text)| (i.to_string(), Json::Str(text.clone())))
                            .collect(),
                    ),
                ),
            ])
        }
        (Question::Estimate { grid, unit, .. }, Reading::Options(p)) => {
            let (p10, median, p90) = grid.read(&p, |share| calibration.level("estimate", share));
            Json::object(vec![
                ("type", Json::Str("number".into())),
                ("number", number(median)),
                ("interval", Json::Array(vec![number(p10), number(p90)])),
                ("unit", unit.clone().map_or(Json::Null, Json::Str)),
            ])
        }
        (Question::Read { kind, unit, .. }, Reading::Span { found, value }) => {
            let found = calibration.through(&format!("found:{}", kind.name()), found);
            let value = value.filter(|_| found >= 0.5);
            let text = value
                .as_ref()
                .map_or(Json::Null, |(text, _, _)| Json::Str(text.clone()));
            let span = value.as_ref().map_or(Json::Null, |(_, span, _)| {
                Json::Array(span.iter().map(|v| Json::UInt(*v as u64)).collect())
            });
            let written = value.as_ref().map(|(text, _, _)| text.as_str());
            let mut unit_out = unit.clone();
            let parsed = match (kind, written) {
                (ReadKind::Number, Some(written)) => {
                    unit_out = unit_out.or_else(|| crate::d1_parsing::written_unit(written));
                    crate::d1_parsing::parse_number(written, context, unit.as_deref())
                        .map_or(Json::Null, parsed_number)
                }
                (ReadKind::Date, Some(written)) => crate::d1_parsing::parse_date(written, context)
                    .0
                    .map_or(Json::Null, Json::Str),
                _ => text.clone(),
            };
            let mut out = vec![
                ("type", Json::Str(kind.name().into())),
                (kind.name(), parsed),
                ("span", span),
                ("found", number(found)),
            ];
            if *kind != ReadKind::Text {
                out.push(("text", text));
            }
            if let Some((_, _, Some(path))) = &value {
                out.push(("path", Json::Str(path.clone())));
            }
            if *kind == ReadKind::Number {
                out.push(("unit", unit_out.map_or(Json::Null, Json::Str)));
            }
            Json::object(out)
        }
        (Question::Box { .. }, Reading::Blocks(p)) => {
            ensure!(
                p.len() == 5 && p[0].len() == 2,
                "a box reads a presence and four edges"
            );
            let found = calibration.through("found:box", p[0][0]);
            if found < 0.5 {
                Json::object(vec![
                    ("type", Json::Str("box".into())),
                    ("box", Json::Null),
                    ("found", number(found)),
                    ("confidence", number(0.)),
                ])
            } else {
                let (rect, confidence) = box_of(&p[1..], interp_edges)?;
                Json::object(vec![
                    ("type", Json::Str("box".into())),
                    ("box", box_json(rect)),
                    ("found", number(found)),
                    ("confidence", number(calibration.through("box", confidence))),
                ])
            }
        }
        (Question::Boxes { .. }, Reading::Blocks(p)) => {
            ensure!(
                !p.is_empty() && p.len() % 5 == 0,
                "boxes read blocks of a presence and four edges"
            );
            let mut below = 1f64;
            let presence: Vec<f64> = p
                .chunks(5)
                .map(|block| {
                    below = below.min(calibration.through("found:boxes", block[0][0]));
                    below
                })
                .collect();
            let mut boxes = Vec::new();
            for (block, found) in p.chunks(5).zip(&presence) {
                if *found < 0.5 {
                    break;
                }
                let (rect, confidence) = box_of(&block[1..], interp_edges)?;
                boxes.push((rect, *found, calibration.through("boxes", confidence)));
            }
            boxes.sort_by(|a, b| {
                (a.0[2], a.0[0])
                    .partial_cmp(&(b.0[2], b.0[0]))
                    .expect("finite edges")
            });
            let count = boxes.len();
            Json::object(vec![
                ("type", Json::Str("boxes".into())),
                (
                    "boxes",
                    Json::Array(
                        boxes
                            .into_iter()
                            .map(|(rect, found, confidence)| {
                                Json::object(vec![
                                    ("box", box_json(rect)),
                                    ("found", number(found)),
                                    ("confidence", number(confidence)),
                                ])
                            })
                            .collect(),
                    ),
                ),
                ("count", Json::UInt(count as u64)),
                ("at_least", Json::Bool(count > 0 && count == presence.len())),
                (
                    "presence",
                    Json::Array(presence.into_iter().map(number).collect()),
                ),
            ])
        }
        _ => bail!("a question was read with another question's slots"),
    })
}

/// A request as the bridge and CLI take it: `{"state": ..., "questions": {name: question}}`.
pub struct Request {
    pub state: Option<Json>,
    pub names: Vec<String>,
    pub questions: Vec<Question>,
    /// Each question's spoken instructions (an audio media object), if any.
    pub spoken: Vec<Option<Json>>,
}
impl Request {
    pub fn parse(value: &Json, settings: &Settings) -> Result<Self> {
        ensure!(
            matches!(value, Json::Object(_)),
            r#"a request is {{"state": ..., "questions": {{name: question}}}}"#
        );
        let Some(Json::Object(entries)) = value.get("questions") else {
            bail!(
                r#"questions are an object of name to question, {{"refund": {{"type": "noul", "instructions": "Is it a refund request?"}}}}"#
            )
        };
        ensure!(!entries.is_empty(), "a request needs at least one question");
        let mut questions = Vec::with_capacity(entries.len());
        for (name, q) in entries {
            questions
                .push(Question::parse(q, settings).with_context(|| format!("question {name:?}"))?);
        }
        Ok(Self {
            state: value.get("state").filter(|s| **s != Json::Null).cloned(),
            names: entries.iter().map(|(name, _)| name.clone()).collect(),
            questions,
            spoken: entries
                .iter()
                .map(|(_, q)| {
                    q.get("instructions")
                        .filter(|i| matches!(i, Json::Object(_)))
                        .cloned()
                })
                .collect(),
        })
    }
}

/// The mask a question token's query takes on full-attention layers: 0 where it sees the key
/// (the state, its own question), `-inf` elsewhere. State queries see every key and need none.
pub fn question_mask(state_len: usize, question: &[i32]) -> Vec<f32> {
    let total = question.len();
    let mut mask = Vec::with_capacity((total - state_len) * total);
    for q in state_len..total {
        for k in 0..total {
            mask.push(if k < state_len || question[q] == question[k] {
                0.
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    mask
}

/// The markup of pictures at the head of the user turn, as the checkpoint's processor expands
/// them: each picture's placeholder once per projected feature, between the vision markers.
pub fn picture_markup(
    tokenizer: &Tokenizer,
    config: &serde_json::Value,
    counts: &[usize],
) -> Result<String> {
    let token = |key: &str, fallback: &str| -> Result<String> {
        match config.get(key).and_then(serde_json::Value::as_u64) {
            Some(id) => tokenizer
                .id_to_token(id as u32)
                .with_context(|| format!("tokenizer has no {key}")),
            None => {
                ensure!(
                    tokenizer.token_to_id(fallback).is_some(),
                    "tokenizer has no {fallback}"
                );
                Ok(fallback.to_owned())
            }
        }
    };
    let (start, end) = (
        token("vision_start_token_id", "<|vision_start|>")?,
        token("vision_end_token_id", "<|vision_end|>")?,
    );
    let image = token("image_token_id", "<|image|>")?;
    Ok(counts
        .iter()
        .map(|count| format!("{start}{}{end}", image.repeat(*count)))
        .collect())
}

/// The bos token a checkpoint's tokenizer config names, as text.
pub fn bos_token(model: &Path) -> Result<String> {
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(model.join("tokenizer_config.json"))
            .context("reading tokenizer_config.json")?,
    )?;
    Ok(match &config["bos_token"] {
        serde_json::Value::String(text) => text.clone(),
        serde_json::Value::Object(token) => token
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("")
            .to_owned(),
        _ => String::new(),
    })
}

#[cfg(feature = "mlx")]
pub mod runtime {
    //! The decision pass on MLX: one forward per sequence, slots read off its hidden states.
    use super::*;
    use crate::{
        array::{Array, Dtype},
        lfm3::{Lfm3, PreparedImage},
        lfm3_audio::Clip,
    };

    fn dump(folder: &Path, name: &str, values: &[f32]) -> Result<()> {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(folder.join(name), bytes).with_context(|| format!("writing {name}"))
    }

    /// One media item of a request: a picture or an audio clip, decoded.
    pub enum Media {
        Image(PreparedImage),
        Audio(Clip),
    }

    /// A media object `{"type": "image" | "audio", "path" | "url"}`; local paths only when allowed.
    pub fn load_media(item: &Json, settings: &Settings, allow_paths: bool) -> Result<Media> {
        let kind = match item.get("type") {
            Some(Json::Str(kind)) => kind.as_str(),
            _ => bail!(r#"a media item is {{"type": "image" | "audio", "path" | "url": ...}}"#),
        };
        let (path, url) = (item.get("path"), item.get("url"));
        let bytes_of = |path: &str| -> Result<Vec<u8>> {
            ensure!(
                allow_paths,
                "this server reads media from data: URLs only, not local paths"
            );
            let file = std::fs::File::open(path).with_context(|| format!("opening {path}"))?;
            ensure!(file.metadata()?.is_file(), "{path} is not a file");
            let mut bytes = Vec::new();
            use std::io::Read;
            file.take(64 * 1024 * 1024 + 1).read_to_end(&mut bytes)?;
            ensure!(bytes.len() <= 64 * 1024 * 1024, "{path} exceeds 64 MiB");
            Ok(bytes)
        };
        match kind {
            "image" => {
                let url = match (path, url) {
                    (Some(Json::Str(file)), None) => {
                        ensure!(
                            allow_paths,
                            "this server reads media from data: URLs only, not local paths"
                        );
                        crate::vision_input::file_data_url(Path::new(file))
                            .with_context(|| format!("reading picture {file}"))?
                    }
                    (None, Some(Json::Str(url))) => url.clone(),
                    _ => bail!(r#"a picture gives its "path" or a data "url""#),
                };
                Ok(Media::Image(crate::vision_input::decode_image_capped(
                    &url,
                    settings.max_pixels,
                )?))
            }
            "audio" => {
                let bytes = match (path, url) {
                    (Some(Json::Str(file)), None) => bytes_of(file)?,
                    (None, Some(Json::Str(url))) => {
                        let (mime, bytes) = crate::lfm3_audio::data_url(url)?;
                        ensure!(
                            matches!(
                                mime.as_str(),
                                "audio/wav" | "audio/x-wav" | "audio/wave" | "audio/vnd.wave"
                            ),
                            "audio data: URLs must be WAV (audio/wav), not {mime}"
                        );
                        bytes
                    }
                    _ => bail!(r#"an audio clip gives its "path" or a data "url""#),
                };
                Ok(Media::Audio(Clip::from_wav(&bytes)?))
            }
            "video" => bail!("video is not supported by this engine's decision path"),
            other => bail!("unknown media type {other:?}"),
        }
    }

    /// How a checkpoint's processor writes media in a prompt.
    struct Markup {
        image: String,
        audio: Option<String>,
        start: String,
        end: String,
        vision_markers: bool,
        audio_markers: bool,
        image_id: u32,
        audio_id: Option<u32>,
    }

    impl Markup {
        fn load(
            model: &Path,
            config: &serde_json::Value,
            tokenizer: &Tokenizer,
            audio: bool,
        ) -> Result<Self> {
            let processor: serde_json::Value =
                match std::fs::read(model.join("processor_config.json")) {
                    Ok(bytes) => {
                        serde_json::from_slice(&bytes).context("invalid processor_config.json")?
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                        serde_json::json!({})
                    }
                    Err(error) => return Err(error.into()),
                };
            let text =
                |key: &str, fallback: &str| processor[key].as_str().unwrap_or(fallback).to_owned();
            let id = |token: &str| {
                tokenizer
                    .token_to_id(token)
                    .with_context(|| format!("tokenizer has no {token}"))
            };
            let image = text("image_token", "<|image|>");
            let image_id = id(&image)?;
            if let Some(expected) = config["image_token_id"].as_u64() {
                ensure!(
                    u64::from(image_id) == expected,
                    "image token disagrees with the config"
                );
            }
            let audio_token = audio.then(|| text("audio_token", "<|audio|>"));
            let audio_id = audio_token.as_deref().map(id).transpose()?;
            if let (Some(found), Some(expected)) = (audio_id, config["audio_token_id"].as_u64()) {
                ensure!(
                    u64::from(found) == expected,
                    "audio token disagrees with the config"
                );
            }
            let (start, end) = (
                text("vision_start_token", "<|vision_start|>"),
                text("vision_end_token", "<|vision_end|>"),
            );
            id(&start)?;
            id(&end)?;
            Ok(Self {
                image,
                audio: audio_token,
                start,
                end,
                vision_markers: processor["vision_markers"].as_bool().unwrap_or(true),
                audio_markers: processor["audio_markers"].as_bool().unwrap_or(true),
                image_id,
                audio_id,
            })
        }
        fn wrap(&self, token: &str, count: usize, markers: bool) -> String {
            let body = token.repeat(count);
            if markers {
                format!("{}{body}{}", self.start, self.end)
            } else {
                body
            }
        }
        fn of(&self, media: &Media) -> Result<String> {
            Ok(match media {
                Media::Image(image) => {
                    self.wrap(&self.image, image.token_count(), self.vision_markers)
                }
                Media::Audio(clip) => self.wrap(
                    self.audio
                        .as_deref()
                        .context("this checkpoint takes no audio")?,
                    clip.token_count(),
                    self.audio_markers,
                ),
            })
        }
    }

    /// A loaded decision model: weights, tokenizer, checkpoint settings and calibration.
    pub struct Engine {
        model: Lfm3,
        tokenizer: Tokenizer,
        bos: String,
        settings: Settings,
        calibration: Calibration,
        markup: Markup,
        pub name: String,
    }

    impl Engine {
        pub fn load(path: &Path, name: &str) -> Result<Self> {
            let checkpoint = crate::checkpoint::inspect(path)?;
            let config: serde_json::Value =
                serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
            let settings = Settings::from_config(&config)?
                .context("this checkpoint records no d1 decision settings (d1_attention); it is not a decision model")?;
            let calibration = path.join("calibration.json");
            let calibration = if calibration.is_file() {
                Calibration::load(&calibration)?
            } else {
                Calibration::neutral()
            };
            let tokenizer = Tokenizer::from_file(path.join("tokenizer.json"))
                .map_err(|error| anyhow::anyhow!("loading tokenizer.json: {error}"))?;
            let model = Lfm3::from_checkpoint(&checkpoint)?;
            let markup = Markup::load(path, &config, &tokenizer, model.supports_audio())?;
            Ok(Self {
                bos: bos_token(path)?,
                model,
                tokenizer,
                settings,
                calibration,
                markup,
                name: name.to_owned(),
            })
        }

        /// What the model takes, as the bridge and server announce it.
        pub fn capabilities(&self) -> serde_json::Value {
            serde_json::json!({
                "model": self.name,
                "decision": true,
                "images": self.model.supports_images(),
                "audio": self.model.supports_audio(),
                "spoken_questions": self.model.supports_audio(),
                "video": false,
                "set_size": self.settings.set_size,
                "max_tokens": MAX_TOKENS,
                "question_types": ["noul", "choice", "score", "text", "number", "date", "box", "boxes"],
            })
        }

        /// One request `{"state", "questions", "media"}`, answered in the API's shape.
        pub fn decide(
            &mut self,
            value: &Json,
            allow_paths: bool,
            trace: Option<&Path>,
        ) -> Result<Json> {
            let request = Request::parse(value, &self.settings)?;
            let media = match value.get("media") {
                None | Some(Json::Null) => Vec::new(),
                Some(Json::Array(items)) => items
                    .iter()
                    .enumerate()
                    .map(|(i, item)| {
                        load_media(item, &self.settings, allow_paths)
                            .with_context(|| format!("media item {i}"))
                    })
                    .collect::<Result<Vec<_>>>()?,
                Some(_) => bail!(r#""media" is a list of {{"type": "image" | "audio", ...}}"#),
            };
            ensure!(
                media.len() <= 8,
                "a decision request takes at most 8 media items"
            );
            let mut voices = Vec::with_capacity(request.questions.len());
            for (name, spoken) in request.names.iter().zip(&request.spoken) {
                voices.push(match spoken {
                    None => None,
                    Some(item) => match load_media(item, &self.settings, allow_paths)
                        .with_context(|| format!("question {name:?}"))?
                    {
                        Media::Audio(clip) => Some(clip),
                        Media::Image(_) => {
                            bail!("question {name:?}: spoken instructions are an audio clip")
                        }
                    },
                });
            }
            let images = media
                .iter()
                .filter(|m| matches!(m, Media::Image(_)))
                .count();
            let audio = media.len() - images + voices.iter().flatten().count();
            ensure!(
                images == 0 || self.model.supports_images(),
                "this checkpoint takes no picture"
            );
            ensure!(
                audio == 0 || self.model.supports_audio(),
                "this checkpoint takes no audio (use d1-Omni)"
            );
            ensure!(
                images == 1 || !request.questions.iter().any(Question::needs_picture),
                "a box or boxes question reads exactly one picture in `media`; this request has {images}"
            );
            ensure!(
                request.state.is_some() || !media.is_empty(),
                "a request without media needs a state"
            );
            let markup = media
                .iter()
                .map(|m| self.markup.of(m))
                .collect::<Result<String>>()?;
            let spoken: Vec<Option<String>> = voices
                .iter()
                .map(|v| {
                    v.as_ref()
                        .map(|clip| self.markup.of(&Media::Audio(clip.clone())))
                        .transpose()
                })
                .collect::<Result<_>>()?;
            let prompter = Prompter::new(
                &self.tokenizer,
                &self.bos,
                self.settings.clone(),
                self.calibration.clone(),
            );
            let sequences = prompter.sequences_with_voices(
                request.state.as_ref(),
                &request.questions,
                &markup,
                &spoken,
            )?;
            // Each item's features once; each sequence takes the state's media, then its own spoken questions.
            let mut image_features = Vec::new();
            let mut state_audio = Vec::new();
            for item in &media {
                match item {
                    Media::Image(image) => image_features.push(self.model.image_features(image)?),
                    Media::Audio(clip) => state_audio.push(self.model.audio_features(clip)?),
                }
            }
            if let Some(folder) = trace {
                // Each clip's stages in request order (state clips, then spoken questions), for parity checks.
                let clips: Vec<&Clip> = media
                    .iter()
                    .filter_map(|m| match m {
                        Media::Audio(clip) => Some(clip),
                        Media::Image(_) => None,
                    })
                    .chain(voices.iter().flatten())
                    .collect();
                let mut listed = Vec::new();
                for (i, clip) in clips.iter().enumerate() {
                    dump(
                        folder,
                        &format!("clip{i}_mel.f32"),
                        &crate::lfm3_audio::log_mel(clip),
                    )?;
                    let (encoded, projected) = self.model.audio_features_f32(clip)?;
                    dump(folder, &format!("clip{i}_encoded.f32"), &encoded.to_f32()?)?;
                    dump(
                        folder,
                        &format!("clip{i}_projected.f32"),
                        &projected.to_f32()?,
                    )?;
                    listed.push(serde_json::json!({"samples": clip.samples.len(), "frames": clip.frames(), "tokens": clip.token_count()}));
                }
                std::fs::write(folder.join("clips.json"), serde_json::to_vec(&listed)?)?;
            }
            let voice_features: Vec<Option<Array>> = voices
                .iter()
                .map(|v| {
                    v.as_ref()
                        .map(|clip| self.model.audio_features(clip))
                        .transpose()
                })
                .collect::<Result<_>>()?;
            let mut placements = Vec::with_capacity(sequences.len());
            for sequence in &sequences {
                let mut audio: Vec<&Array> = state_audio.iter().collect();
                let mut asked: Vec<usize> =
                    sequence.slots.iter().map(|slot| slot.question).collect();
                asked.dedup();
                audio.extend(asked.iter().filter_map(|q| voice_features[*q].as_ref()));
                let mut placed = Vec::new();
                for (token, features) in [
                    (
                        Some(self.markup.image_id),
                        image_features.iter().collect::<Vec<_>>(),
                    ),
                    (self.markup.audio_id, audio),
                ] {
                    let runs = token.map_or_else(Vec::new, |token| runs_of(&sequence.ids, token));
                    ensure!(
                        runs.len() == features.len(),
                        "media placeholders and items do not match"
                    );
                    for ((start, length), features) in runs.into_iter().zip(features) {
                        ensure!(
                            features.shape()[0] as usize == length,
                            "media features and placeholders differ in length"
                        );
                        placed.push((start, features.try_clone()?));
                    }
                }
                placements.push(placed);
            }
            let result = decide_prepared(
                &mut self.model,
                &prompter,
                &request,
                &sequences,
                placements,
                trace,
            );
            self.model.reset();
            result
        }
    }

    /// Runs of `token` in `ids`: (start, length), in order.
    fn runs_of(ids: &[u32], token: u32) -> Vec<(usize, usize)> {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for (i, id) in ids.iter().enumerate() {
            if *id == token {
                match runs.last_mut() {
                    Some((start, length)) if *start + *length == i => *length += 1,
                    _ => runs.push((i, 1)),
                }
            }
        }
        runs
    }

    fn decide_prepared(
        model: &mut Lfm3,
        prompter: &Prompter,
        request: &Request,
        sequences: &[Sequence],
        placements: Vec<Vec<(usize, Array)>>,
        trace: Option<&Path>,
    ) -> Result<Json> {
        let mut options: Vec<Vec<Vec<f64>>> = vec![Vec::new(); request.questions.len()];
        let mut spans: Vec<Option<Reading>> = vec![None; request.questions.len()];
        let mut placements = placements.into_iter();
        let mut read = 0usize;
        let mut traced = Vec::new();
        for (index, sequence) in sequences.iter().enumerate() {
            read += sequence.ids.len();
            let time = sequence.ids.len() as i32;
            let mask = question_mask(sequence.state_len, &sequence.question);
            model.reset();
            model.place_features(
                placements
                    .next()
                    .context("a sequence has no media placement")?,
            )?;
            let hidden = model.decide_hidden(&sequence.ids, sequence.state_len, &mask)?;
            let dim = hidden.shape()[2];
            let flat = hidden.reshape(&[time, dim])?;
            let positions: Vec<i32> = sequence.slots.iter().map(|s| s.position as i32).collect();
            ensure!(
                positions.iter().all(|p| (0..time).contains(p)),
                "answer slot outside its sequence"
            );
            let rows = flat.take(&Array::from_i32(&positions, &[positions.len() as i32])?, 0)?;
            let code: Vec<usize> = (0..sequence.slots.len())
                .filter(|i| matches!(sequence.slots[*i].readout, Readout::Tokens { .. }))
                .collect();
            let reads: Vec<usize> = (0..sequence.slots.len())
                .filter(|i| !code.contains(i))
                .collect();
            let mut entry = vec![
                (
                    "ids",
                    Json::Array(
                        sequence
                            .ids
                            .iter()
                            .map(|i| Json::UInt(u64::from(*i)))
                            .collect(),
                    ),
                ),
                ("state_len", Json::UInt(sequence.state_len as u64)),
                (
                    "question",
                    Json::Array(
                        sequence
                            .question
                            .iter()
                            .map(|q| Json::Int(i64::from(*q)))
                            .collect(),
                    ),
                ),
                (
                    "slots",
                    Json::Array(
                        sequence
                            .slots
                            .iter()
                            .map(|slot| {
                                let mut out = vec![
                                    ("question", Json::UInt(slot.question as u64)),
                                    ("end", Json::UInt(slot.position as u64)),
                                ];
                                match &slot.readout {
                                    Readout::Tokens {
                                        groups,
                                        temperature,
                                    } => out.extend([
                                        ("kind", Json::Str("tokens".into())),
                                        ("temperature", Json::Float(*temperature)),
                                        (
                                            "groups",
                                            Json::Array(
                                                groups
                                                    .iter()
                                                    .map(|g| {
                                                        Json::Array(
                                                            g.iter()
                                                                .map(|i| Json::UInt(u64::from(*i)))
                                                                .collect(),
                                                        )
                                                    })
                                                    .collect(),
                                            ),
                                        ),
                                    ]),
                                    Readout::Start => out.push(("kind", Json::Str("start".into()))),
                                    Readout::End => out.push(("kind", Json::Str("end".into()))),
                                }
                                Json::object(out)
                            })
                            .collect(),
                    ),
                ),
            ];
            if let Some(folder) = trace {
                dump(folder, &format!("s{index}_hidden.f32"), &flat.to_f32()?)?;
                dump(
                    folder,
                    &format!("s{index}_slot_hidden.f32"),
                    &rows.to_f32()?,
                )?;
            }

            // Code slots: log-probabilities over the vocabulary, only the option tokens brought back.
            let mut tokens: Vec<u32> = Vec::new();
            for slot in &sequence.slots {
                if let Readout::Tokens { groups, .. } = &slot.readout {
                    tokens.extend(groups.iter().flatten());
                }
            }
            tokens.sort_unstable();
            tokens.dedup();
            let mut scores = Vec::new();
            if !code.is_empty() {
                let picked = rows.take(
                    &Array::from_i32(
                        &code.iter().map(|i| *i as i32).collect::<Vec<_>>(),
                        &[code.len() as i32],
                    )?,
                    0,
                )?;
                let logz = model
                    .project_rows(&picked)?
                    .astype(Dtype::Float32)?
                    .log_probs()?;
                ensure!(
                    logz.shape() == [code.len() as i32, logz.shape()[1]] && logz.shape()[1] > 0,
                    "empty slot logits"
                );
                ensure!(
                    tokens.iter().all(|t| (*t as i32) < logz.shape()[1]),
                    "option token outside the vocabulary"
                );
                if let Some(folder) = trace {
                    dump(folder, &format!("s{index}_logz.f32"), &logz.to_f32()?)?;
                }
                scores = logz
                    .take(
                        &Array::from_i32(
                            &tokens.iter().map(|t| *t as i32).collect::<Vec<_>>(),
                            &[tokens.len() as i32],
                        )?,
                        1,
                    )?
                    .to_f32()?;
                ensure!(
                    scores.len() == code.len() * tokens.len()
                        && scores.iter().all(|v| v.is_finite()),
                    "non-finite slot log-probabilities"
                );
            }
            for (row, slot_index) in code.iter().enumerate() {
                let slot = &sequence.slots[*slot_index];
                let Readout::Tokens {
                    groups,
                    temperature,
                } = &slot.readout
                else {
                    unreachable!()
                };
                let at = |id: u32| {
                    f64::from(
                        scores[row * tokens.len()
                            + tokens.binary_search(&id).expect("gathered token")],
                    )
                };
                options[slot.question].push(probabilities(groups, *temperature, at)?);
            }

            // Read slots: each (start, end) pair against the state's positions, sqrt(D)-scaled dot products.
            if !reads.is_empty() {
                let map = sequence
                    .positions
                    .as_ref()
                    .context("a read needs its state positions")?;
                let valid = map.pointer_mask();
                ensure!(
                    valid.len() <= sequence.ids.len() && reads.len().is_multiple_of(2),
                    "invalid pointer layout"
                );
                let queries = rows
                    .take(
                        &Array::from_i32(
                            &reads.iter().map(|i| *i as i32).collect::<Vec<_>>(),
                            &[reads.len() as i32],
                        )?,
                        0,
                    )?
                    .astype(Dtype::Float32)?;
                let keys = flat
                    .slice(0, 0, valid.len() as i32)?
                    .astype(Dtype::Float32)?;
                let raw = queries
                    .matmul(&keys.transpose(&[1, 0])?)?
                    .scalar_mul((dim as f32).powf(-0.5))?
                    .to_f32()?;
                ensure!(
                    raw.len() == reads.len() * valid.len() && raw.iter().all(|v| v.is_finite()),
                    "non-finite pointer scores"
                );
                if let Some(folder) = trace {
                    dump(folder, &format!("s{index}_positions.f32"), &raw)?;
                }
                let row = |r: usize| -> Vec<f64> {
                    raw[r * valid.len()..(r + 1) * valid.len()]
                        .iter()
                        .zip(&valid)
                        .map(|(v, keep)| {
                            if *keep {
                                f64::from(*v)
                            } else {
                                f64::NEG_INFINITY
                            }
                        })
                        .collect()
                };
                for (pair, slots) in reads.chunks(2).enumerate() {
                    let (start, end) = (&sequence.slots[slots[0]], &sequence.slots[slots[1]]);
                    ensure!(
                        start.readout == Readout::Start
                            && end.readout == Readout::End
                            && start.question == end.question,
                        "a read is a start then an end"
                    );
                    let Question::Read { kind, .. } = &request.questions[start.question] else {
                        bail!("pointer slots on another question")
                    };
                    let ((first, last), found) =
                        span(&row(2 * pair), &row(2 * pair + 1), kind.cap())?;
                    spans[start.question] = Some(Reading::Span {
                        found,
                        value: map.read(first, last, kind.cap())?,
                    });
                }
                entry.push(("pointer_width", Json::UInt(valid.len() as u64)));
                entry.push((
                    "offsets",
                    Json::Array(
                        map.spans
                            .iter()
                            .map(|span| match span {
                                Some((a, b)) => Json::Array(vec![
                                    Json::UInt(map.text[..*a].chars().count() as u64),
                                    Json::UInt(map.text[..*b].chars().count() as u64),
                                ]),
                                None => Json::Null,
                            })
                            .collect(),
                    ),
                ));
            }
            entry.push((
                "tokens",
                Json::Array(tokens.iter().map(|t| Json::UInt(u64::from(*t))).collect()),
            ));
            traced.push(Json::object(entry));
        }
        let context = prompter.state_text(request.state.as_ref());
        let mut answers = Vec::with_capacity(request.questions.len());
        for (index, (name, q)) in request.names.iter().zip(&request.questions).enumerate() {
            let read = std::mem::take(&mut options[index]);
            let reading = match q {
                Question::Read { .. } => spans[index].take().context("a read was not answered")?,
                Question::Box { .. } | Question::Boxes { .. } => Reading::Blocks(read),
                Question::Choice { multiple: true, .. } => {
                    Reading::Options(read.iter().map(|p| p[0]).collect())
                }
                _ => {
                    ensure!(
                        read.len() == 1,
                        "question {name:?} was read {} times",
                        read.len()
                    );
                    Reading::Options(read.into_iter().next().expect("one reading"))
                }
            };
            ensure!(
                q.typed() || matches!(reading, Reading::Options(_)),
                "option question without options"
            );
            answers.push((
                name.clone(),
                answer(
                    q,
                    reading,
                    &prompter.calibration,
                    prompter.settings.interp_edges,
                    &context,
                )?,
            ));
        }
        if let Some(folder) = trace {
            std::fs::write(
                folder.join("trace.json"),
                serde_json::to_vec(&Json::Array(traced))?,
            )?;
        }
        Ok(Json::object(vec![
            ("answers", Json::Object(answers)),
            (
                "usage",
                Json::object(vec![
                    ("input_tokens", Json::UInt(read as u64)),
                    ("output_tokens", Json::UInt(0)),
                ]),
            ),
        ]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats_print_as_python_does() {
        for (value, repr) in [
            (12.5, "12.5"),
            (1.0, "1.0"),
            (0.1, "0.1"),
            (1e16, "1e+16"),
            (1e15, "1000000000000000.0"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (1.5e-7, "1.5e-07"),
            (-2.5, "-2.5"),
            (123456789.123, "123456789.123"),
            (1e22, "1e+22"),
            (5e-324, "5e-324"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (2019.0, "2019.0"),
        ] {
            assert_eq!(python_float(value), repr);
        }
        for (value, repr) in [
            (0., "0"),
            (10., "10"),
            (0.5, "0.5"),
            (0.999, "0.999"),
            (100., "100"),
            (1e6, "1e+06"),
            (123456., "123456"),
            (1234567., "1.23457e+06"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (0.007874015748031496, "0.00787402"),
            (-3.25, "-3.25"),
            (1. / 3., "0.333333"),
        ] {
            assert_eq!(python_g(value), repr);
        }
    }

    #[test]
    fn json_renders_as_python_indent_two() {
        let state: Json = serde_json::from_str(
            r#"{"b": {"name": "Zoë \"Zed\"\n", "since": 2019, "x": 12.50, "e": [], "o": {}}, "a": [1, "p/q~r", null, true], "b2": false}"#,
        )
        .unwrap();
        let (mut out, mut strings) = (String::new(), Vec::new());
        python_dump(&state, 0, "", &mut out, &mut strings);
        assert_eq!(
            out,
            "{\n  \"b\": {\n    \"name\": \"Zoë \\\"Zed\\\"\\n\",\n    \"since\": 2019,\n    \"x\": 12.5,\n    \"e\": [],\n    \"o\": {}\n  },\n  \"a\": [\n    1,\n    \"p/q~r\",\n    null,\n    true\n  ],\n  \"b2\": false\n}"
        );
        assert_eq!(
            strings
                .iter()
                .map(|(s, e, p)| (&out[*s..*e], p.as_str()))
                .collect::<Vec<_>>(),
            vec![("Zoë \\\"Zed\\\"\\n", "/b/name"), ("p/q~r", "/a/1")]
        );
        let repeated: Json = serde_json::from_str(r#"{"a": 1, "b": 2, "a": 3}"#).unwrap();
        assert_eq!(
            repeated,
            Json::Object(vec![("a".into(), Json::Int(3)), ("b".into(), Json::Int(2))])
        );
    }

    #[test]
    fn calibration_tables_interpolate_and_invert() {
        let knots = [[0.1, 0.0], [0.5, 0.4], [0.9, 1.0]];
        assert_eq!(through(0.05, Some(&knots)), 0.0);
        assert_eq!(through(0.95, Some(&knots)), 1.0);
        assert!((through(0.3, Some(&knots)) - 0.2).abs() < 1e-12);
        assert!((through(0.5, Some(&knots)) - 0.4).abs() < 1e-12);
        assert_eq!(through(0.7, None), 0.7);
        let calibration = Calibration {
            tables: [("estimate".to_owned(), knots.to_vec())].into(),
            default: 1.,
            ..Default::default()
        };
        assert!((calibration.level("estimate", 0.2) - 0.3).abs() < 1e-12);
        assert_eq!(calibration.level("missing", 0.2), 0.2);
    }

    #[test]
    fn spans_follow_the_reference_rule() {
        let inf = f64::NEG_INFINITY;
        // nothing: the start's mass sits on position 0
        assert_eq!(span(&[5., 0., 0.], &[0., 0., 0.], 512).unwrap().0, (0, 0));
        let ((first, last), found) = span(&[-5., 4., 0., inf], &[-5., 0., 3., inf], 512).unwrap();
        assert_eq!((first, last), (1, 2));
        assert!(found > 0.99);
        // the end never precedes the start, and the cap bounds the span
        assert_eq!(
            span(&[-9., 0., 5., 0.], &[-9., 2., 0., 1.], 512).unwrap().0,
            (2, 3)
        );
        assert_eq!(
            span(&[-9., 20., 0., 0., 0.], &[-9., 0., 1., 0., 9.], 2)
                .unwrap()
                .0,
            (1, 2)
        );
        assert_eq!(
            span(&[-9., 20., 0., 0., 0.], &[-9., 0., 1., 0., 9.], 512)
                .unwrap()
                .0,
            (1, 4)
        );
        assert!(span(&[], &[], 4).is_err());
        assert!(span(&[f64::NAN, 0.], &[0., 0.], 4).is_err());
    }

    #[test]
    fn question_tokens_see_their_state_and_themselves() {
        let question = [-1, -1, 1, 1, 2];
        let mask = question_mask(2, &question);
        let seen: Vec<bool> = mask.iter().map(|m| *m == 0.).collect();
        assert_eq!(
            seen,
            [
                true, true, true, true, false, true, true, true, true, false, true, true, false,
                false, true
            ]
        );
        assert!(question_mask(2, &[-1, -1]).is_empty());
    }

    #[test]
    fn answers_take_the_api_shape() {
        let calibration = Calibration::neutral();
        let settings = Settings {
            codes: 2,
            numbers: true,
            set_size: 16,
            max_pixels: 1,
            interp_edges: false,
            prompt: 1,
        };
        let parse = |text: &str| Question::parse(&serde_json::from_str(text).unwrap(), &settings);
        let score =
            parse(r#"{"type": "score", "instructions": "x", "criteria": ["low", "mid", "high"]}"#)
                .unwrap();
        let out = answer(
            &score,
            Reading::Options(vec![0.2, 0.35, 0.45]),
            &calibration,
            false,
            "",
        )
        .unwrap();
        assert_eq!(out.get("score"), Some(&Json::Float(1.)));
        assert_eq!(out.get("confidence"), Some(&Json::Float(0.45)));
        let estimate =
            parse(r#"{"type": "number", "instructions": "x", "range": [0, 10], "step": 1}"#)
                .unwrap();
        let mut p = vec![0.; 11];
        (p[2], p[3], p[7]) = (0.2, 0.5, 0.3);
        let out = answer(&estimate, Reading::Options(p), &calibration, false, "").unwrap();
        assert_eq!(out.get("number"), Some(&Json::Float(3.)));
        assert_eq!(
            out.get("interval"),
            Some(&Json::Array(vec![Json::Float(2.), Json::Float(7.)]))
        );
        for bad in [
            r#"{"type": "essay", "instructions": "x"}"#,
            r#"{"type": "noul", "instructions": " "}"#,
            r#"{"type": "choice", "instructions": "x", "criteria": ["a", "b"]}"#,
            r#"{"type": "score", "instructions": "x", "criteria": ["only"]}"#,
            r#"{"type": "number", "instructions": "x", "range": [5, 1]}"#,
            r#"{"type": "number", "instructions": "x", "range": [0, 1], "step": 0.3}"#,
            r#"{"type": "noul", "instructions": "x", "multiple": true}"#,
            r#"{"type": "boxes", "instructions": "x", "max": 17}"#,
            r#"{"type": "text", "instructions": "x", "step": 1}"#,
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
        assert!(answer(&score, Reading::Blocks(vec![]), &calibration, false, "").is_err());
    }

    #[test]
    fn boxes_read_medians_and_reading_order() {
        let calibration = Calibration::neutral();
        let edge = |at: usize| {
            let mut row = vec![0.; 1000];
            row[at] = 1.;
            row
        };
        let block = |present: f64, e: [usize; 4]| {
            std::iter::once(vec![present, 1. - present])
                .chain(e.map(edge))
                .collect::<Vec<_>>()
        };
        let q = Question::Box {
            instructions: "x".into(),
        };
        let out = answer(
            &q,
            Reading::Blocks(block(0.9, [100, 300, 200, 600])),
            &calibration,
            false,
            "",
        )
        .unwrap();
        assert_eq!(
            out.get("box").and_then(|b| b.get("right")),
            Some(&Json::Float(0. + 600. * (0.999 / 999.)))
        );
        assert_eq!(out.get("confidence"), Some(&Json::Float(1.)));
        assert_eq!(
            answer(
                &q,
                Reading::Blocks(block(0.2, [1, 2, 3, 4])),
                &calibration,
                false,
                "",
            )
            .unwrap()
            .get("box"),
            Some(&Json::Null)
        );
        let many = Question::Boxes {
            instructions: "x".into(),
            max: Some(3),
        };
        let blocks = [
            block(0.9, [10, 50, 700, 800]),
            block(0.8, [10, 50, 100, 200]),
            block(0.1, [1, 2, 3, 4]),
        ]
        .concat();
        let out = answer(&many, Reading::Blocks(blocks), &calibration, false, "").unwrap();
        assert_eq!(out.get("count"), Some(&Json::UInt(2)));
        assert_eq!(out.get("at_least"), Some(&Json::Bool(false)));
        let Some(Json::Array(boxes)) = out.get("boxes") else {
            panic!()
        };
        assert!(
            boxes[0].get("box").unwrap().get("left").unwrap().as_f64()
                < boxes[1].get("box").unwrap().get("left").unwrap().as_f64()
        );
        // crossed edges become one bin about their middle
        let (low, high) = ordered(0.5, 0.5, 0.001);
        assert!((low - 0.4995).abs() < 1e-12 && (high - 0.5005).abs() < 1e-12);
    }
}
