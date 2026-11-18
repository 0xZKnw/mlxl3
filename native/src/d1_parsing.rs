//! Strict readings of written values for the d1 decision model: no arithmetic, unit conversion or
//! invented date components.
//!
//! Port of `parsing.py` from `LiquidAI/d1-2B@a6bce774e6dde338dcc3036ceb5d91f78bf4c19e`
//! (`written_unit`, `parse_number`, `parse_date`). The port keeps CPython 3.12 semantics where they
//! are observable:
//!
//! * `re` classes are Unicode-aware and follow CPython's Unicode 15.0 database: `\d` is `Nd`, `\w`
//!   is `L|N|_` (CPython's `isalnum() or "_"`), `\s` is `str.isspace()` (which, unlike Rust's
//!   `White_Space`, includes U+001C..U+001F), and `\b` is a `\w`/non-`\w` transition. `re.I`
//!   additionally lets `i` match `ı`/`İ`, as CPython's case-insensitive matching does.
//! * `str.strip`/`str.split` use `str.isspace()`, `str.casefold` is full case folding, and the date
//!   fold is `NFKD(casefold(text))` without combining marks.
//! * Numbers go through `decimal.Decimal` semantics: exact parsing, then a product rounded to the
//!   default context (28 significant digits, `ROUND_HALF_EVEN`, `Emax = 999999`, subnormal
//!   rounding at `Etiny`); integral results are integers, the others `float(value)`.
//! * Dates follow `datetime.date` validity and `datetime.fromisoformat(...).isoformat()` for the
//!   written time.
//!
//! Where CPython raises instead of returning (a `KeyError` on a case-folded marker such as `ınr`,
//! `decimal.Overflow` past `Emax`), the Rust functions return `None`.

use std::cell::OnceCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::LazyLock;

use fancy_regex::{Captures, NoExpand, Regex, RegexBuilder};
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::canonical_combining_class;

/// A value `parse_number` reads: Python returns an `int` when the `Decimal` value is integral,
/// otherwise a `float`.
#[derive(Clone, Debug, PartialEq)]
pub enum Number {
    /// An integral value that fits in an `i128`.
    Int(i128),
    /// An integral value outside the `i128` range, as Python's `str(int)` (optional `-`, digits).
    BigInt(String),
    /// A non-integral value, `float(Decimal)` (correctly rounded).
    Float(f64),
}

/// CPython raised an exception (or a regex search exceeded its backtracking budget).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Raised;

type R<T> = Result<T, Raised>;

// ---------------------------------------------------------------------------------------------
// CPython character semantics
// ---------------------------------------------------------------------------------------------

/// CPython `\d` (`Nd`) restricted to the Unicode 15.0 repertoire of CPython 3.12.
const D: &str = r"[\p{Nd}&&\p{Age:V15_0}]";
/// CPython `\w`: `str.isalnum()` or `_`, i.e. `L|N|_` in Unicode 15.0.
macro_rules! py_word_class {
    () => {
        r"[\p{L}\p{N}_&&\p{Age:V15_0}]"
    };
}
const W: &str = py_word_class!();
/// CPython `\b`, independent of case-insensitivity like in `re`.
const B: &str = concat!(
    "(?-i:(?<=",
    py_word_class!(),
    ")(?!",
    py_word_class!(),
    ")|(?<!",
    py_word_class!(),
    ")(?=",
    py_word_class!(),
    "))"
);
/// CPython `\s` (`str.isspace()`).
const S: &str = r"[\t\n\x0B\x0C\r\x1C-\x1F \x{85}\x{A0}\x{1680}\x{2000}-\x{200A}\x{2028}\x{2029}\x{202F}\x{205F}\x{3000}]";
/// CPython `$` without `re.M`: the end, or before a final newline.
const END: &str = r"(?=\n?\z)";

const PY_SPACE: &[char] = &[
    '\t', '\n', '\u{b}', '\u{c}', '\r', '\u{1c}', '\u{1d}', '\u{1e}', '\u{1f}', ' ', '\u{85}',
    '\u{a0}', '\u{1680}', '\u{2000}', '\u{2001}', '\u{2002}', '\u{2003}', '\u{2004}', '\u{2005}',
    '\u{2006}', '\u{2007}', '\u{2008}', '\u{2009}', '\u{200a}', '\u{2028}', '\u{2029}', '\u{202f}',
    '\u{205f}', '\u{3000}',
];

fn py_isspace(c: char) -> bool {
    PY_SPACE.contains(&c)
}

fn py_strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

fn py_rstrip(s: &str) -> &str {
    s.trim_end_matches(py_isspace)
}

/// `str.split()` without arguments.
fn py_split(s: &str) -> Vec<&str> {
    s.split(py_isspace).filter(|w| !w.is_empty()).collect()
}

/// Python `len(str)`: code points.
fn clen(s: &str) -> usize {
    s.chars().count()
}

/// The decimal digit value of a CPython `Nd` character (`unicodedata.decimal`).
fn nd_value(c: char) -> Option<u32> {
    let cp = u32::from(c);
    let i = ND_ZEROS.partition_point(|&z| z <= cp);
    let zero = *ND_ZEROS.get(i.checked_sub(1)?)?;
    (cp - zero < 10).then_some(cp - zero)
}

/// Python `int(s)` for a string of `Nd` digits, saturating (values this large are rejected by the
/// callers' range checks either way).
fn py_int(s: &str) -> u64 {
    s.chars().fold(0u64, |acc, c| {
        acc.saturating_mul(10)
            .saturating_add(u64::from(nd_value(c).unwrap_or(0)))
    })
}

fn all_nd(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| nd_value(c).is_some())
}

/// `str.casefold()`: full case folding.
fn py_casefold(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match CASEFOLD_EXCEPTIONS.binary_search_by_key(&c, |&(k, _)| k) {
            Ok(i) => out.push_str(CASEFOLD_EXCEPTIONS[i].1),
            Err(_) => out.extend(c.to_lowercase()),
        }
    }
    out
}

/// `_fold`: `NFKD(casefold(text))` without the characters `unicodedata.combining` marks.
fn fold(s: &str) -> String {
    py_casefold(s)
        .as_str()
        .nfkd()
        .filter(|&c| canonical_combining_class(c) == 0)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Regex helpers
// ---------------------------------------------------------------------------------------------

/// Translates a CPython pattern: `\d`, `\w`, `\s`, `\b`, `\uXXXX` and a bare `$`.
fn py(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len() * 2);
    let mut chars = pattern.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => match chars.next() {
                Some('d') => out.push_str(D),
                Some('w') => out.push_str(W),
                Some('s') => out.push_str(S),
                Some('b') => out.push_str(B),
                Some('u') => {
                    let hex: String = chars.by_ref().take(4).collect();
                    out.push_str(&format!("\\x{{{hex}}}"));
                }
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            },
            '$' => out.push_str(END),
            _ => out.push(c),
        }
    }
    out
}

/// A literal for a `re.I` pattern: escaped, with `i` also matching `ı` and `İ` as in CPython.
fn ci(literal: &str) -> String {
    literal
        .chars()
        .map(|c| match c {
            'i' | 'I' => "[iıİ]".to_owned(),
            _ => fancy_regex::escape(&c.to_string()).into_owned(),
        })
        .collect()
}

fn compile(pattern: &str) -> Regex {
    RegexBuilder::new(pattern)
        // CPython's `re` has no backtracking budget; these patterns stay near-linear.
        .backtrack_limit(usize::MAX)
        .build()
        .unwrap_or_else(|e| panic!("d1_parsing pattern {pattern:?}: {e}"))
}

/// `re.fullmatch`.
fn compile_full(pattern: &str) -> Regex {
    compile(&format!(r"^(?:{pattern})\z"))
}

fn is_match(re: &Regex, s: &str) -> R<bool> {
    re.is_match(s).map_err(|_| Raised)
}

fn captures<'t>(re: &Regex, s: &'t str) -> R<Option<Captures<'t>>> {
    re.captures(s).map_err(|_| Raised)
}

/// `re.findall` for a pattern without groups.
fn find_all<'t>(re: &Regex, s: &'t str) -> R<Vec<&'t str>> {
    re.find_iter(s)
        .map(|m| m.map(|m| m.as_str()).map_err(|_| Raised))
        .collect()
}

/// `[m[group] for m in re.finditer(...)]` for a group that always participates.
fn group_all<'t>(re: &Regex, s: &'t str, group: usize) -> R<Vec<&'t str>> {
    re.captures_iter(s)
        .map(|caps| {
            caps.map(|caps| caps.get(group).map_or("", |m| m.as_str()))
                .map_err(|_| Raised)
        })
        .collect()
}

/// `re.sub(pattern, replacement, s)` with a literal replacement.
fn sub(re: &Regex, s: &str, replacement: &str) -> R<String> {
    re.try_replacen(s, 0, NoExpand(replacement))
        .map(|s| s.into_owned())
        .map_err(|_| Raised)
}

fn group<'t>(caps: &Captures<'t>, i: usize) -> &'t str {
    caps.get(i).map_or("", |m| m.as_str())
}

/// `_one`: the one value evidence settles on.
fn one<T: Copy + Ord>(values: &BTreeSet<T>) -> Option<T> {
    if values.len() == 1 {
        values.first().copied()
    } else {
        None
    }
}

// ---------------------------------------------------------------------------------------------
// Tables (built in the dict order of parsing.py, which fixes the regex alternation order)
// ---------------------------------------------------------------------------------------------

const MONTH_NAMES: [&str; 12] = [
    "january janvier januar enero gennaio janeiro januari stycznia styczen января",
    "february fevrier februar febrero febbraio fevereiro februari lutego luty февраля",
    "march mars marz marzo marco maart marca марта",
    "april avril abril aprile kwietnia kwiecien апреля",
    "may mai mayo maggio maio mei maj maja мая",
    "june juin juni junio giugno junho czerwca czerwiec июня",
    "july juillet juli julio luglio julho lipca lipiec июля",
    "august aout agosto augustus augusti sierpnia sierpien августа",
    "september septembre septiembre settembre setembro wrzesnia wrzesien сентября",
    "october octobre oktober octubre ottobre outubro pazdziernika pazdziernik октября",
    "november novembre noviembre novembro listopada listopad ноября",
    "december decembre dezember diciembre dicembre dezembro grudnia grudzien декабря",
];

const WEEKDAY_NAMES: &str = "monday tuesday wednesday thursday friday saturday sunday mon tue \
    tues wed thu thur thurs fri sat sun lundi mardi mercredi jeudi vendredi samedi dimanche montag \
    dienstag mittwoch donnerstag freitag samstag sonntag lunes martes miercoles jueves viernes \
    sabado domingo lunedi martedi mercoledi giovedi venerdi sabato domenica segunda terca quarta \
    quinta sexta";

/// Currencies by the decimal mark their amounts are written with: `""` for none (whole units),
/// `None` for both in use.
const CURRENCIES: [(Option<&str>, &[&str]); 4] = [
    (
        Some("."),
        &[
            "$", "£", "₹", "USD", "GBP", "CAD", "AUD", "NZD", "INR", "Rs.", "Rs", "MXN", "CNY",
            "RMB", "元", "HKD", "SGD",
        ],
    ),
    (
        Some(","),
        &[
            "R$", "BRL", "₽", "RUB", "zł", "PLN", "kr", "SEK", "NOK", "DKK", "₺", "TRY",
        ],
    ),
    (
        Some(""),
        &[
            "¥", "￥", "円", "₩", "₫", "Rp.", "Rp", "IDR", "JPY", "KRW", "VND", "CLP",
        ],
    ),
    (None, &["€", "EUR", "CHF"]),
];

/// Scale words and their power of ten (German "Billion" is 10^12: left out).
const SCALE_WORDS: &[(&str, u32)] = &[
    ("thousand", 3),
    ("mil", 3),
    ("tausend", 3),
    ("тыс", 3),
    ("千", 3),
    ("万", 4),
    ("萬", 4),
    ("lakh", 5),
    ("lakhs", 5),
    ("crore", 7),
    ("crores", 7),
    ("million", 6),
    ("millions", 6),
    ("mio", 6),
    ("mln", 6),
    ("millón", 6),
    ("millones", 6),
    ("milhão", 6),
    ("milhões", 6),
    ("milione", 6),
    ("milioni", 6),
    ("млн", 6),
    ("亿", 8),
    ("億", 8),
    ("billion", 9),
    ("billions", 9),
    ("milliard", 9),
    ("milliards", 9),
    ("milliarde", 9),
    ("milliarden", 9),
    ("mrd", 9),
    ("miliardo", 9),
    ("miliardi", 9),
    ("млрд", 9),
    ("trillion", 12),
];

/// Stable sort by code-point length, longest first (`sorted(..., key=len, reverse=True)`).
fn longest_first<'a>(items: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
    let mut items: Vec<&str> = items.into_iter().collect();
    items.sort_by_key(|s| std::cmp::Reverse(clen(s)));
    items
}

struct Tables {
    /// `_NAMES`: casefolded marker -> marker as written in the table.
    names: HashMap<String, &'static str>,
    /// `_MARK`: casefolded marker -> decimal mark.
    marks: HashMap<String, Option<&'static str>>,
    scales: HashMap<&'static str, u32>,
    months: HashMap<String, u32>,
    weekdays: HashSet<&'static str>,
    currency: String,
    /// The `_CURRENCY` alternatives, in alternation order.
    currency_markers: Vec<&'static str>,
    month_alternation: String,
}

static TABLES: LazyLock<Tables> = LazyLock::new(|| {
    let mut names: HashMap<String, &'static str> = HashMap::new();
    let mut name_order: Vec<String> = Vec::new();
    let mut marks = HashMap::new();
    for (mark, markers) in CURRENCIES {
        for &marker in markers {
            let key = py_casefold(marker);
            if names.insert(key.clone(), marker).is_none() {
                name_order.push(key.clone());
            }
            marks.insert(key, mark);
        }
    }
    let currency_markers = longest_first(name_order.iter().map(|key| names[key]));
    let currency = currency_markers
        .iter()
        .map(|&c| {
            let first = c.chars().next().unwrap_or(' ');
            if first.is_ascii_alphabetic() {
                format!("{B}{}(?![a-zıİ])", ci(c))
            } else {
                ci(c)
            }
        })
        .collect::<Vec<_>>()
        .join("|");

    // Python dict semantics: a new key is appended, an existing key keeps its position.
    fn insert(
        months: &mut HashMap<String, u32>,
        order: &mut Vec<String>,
        name: String,
        month: u32,
    ) {
        if months.insert(name.clone(), month).is_none() {
            order.push(name);
        }
    }
    let mut months: HashMap<String, u32> = HashMap::new();
    let mut month_order: Vec<String> = Vec::new();
    for (i, line) in MONTH_NAMES.iter().enumerate() {
        for name in line.split_whitespace() {
            insert(&mut months, &mut month_order, name.to_owned(), i as u32 + 1);
        }
    }
    let mut abbreviations: Vec<(String, BTreeSet<u32>)> = Vec::new();
    let snapshot: Vec<(String, u32)> = month_order
        .iter()
        .map(|name| (name.clone(), months[name]))
        .collect();
    for (name, month) in snapshot {
        let abbreviation: String = name.chars().take(3).collect();
        match abbreviations.iter_mut().find(|(a, _)| *a == abbreviation) {
            Some((_, set)) => {
                set.insert(month);
            }
            None => abbreviations.push((abbreviation, BTreeSet::from([month]))),
        }
    }
    for (abbreviation, set) in abbreviations {
        if let Some(month) = one(&set) {
            insert(&mut months, &mut month_order, abbreviation, month);
        }
    }
    for (name, month) in [("janv", 1), ("fevr", 2), ("juil", 7), ("sept", 9)] {
        insert(&mut months, &mut month_order, name.to_owned(), month);
    }
    let month_alternation = longest_first(month_order.iter().map(String::as_str))
        .into_iter()
        .map(|m| fancy_regex::escape(m).into_owned())
        .collect::<Vec<_>>()
        .join("|");

    Tables {
        names,
        marks,
        scales: SCALE_WORDS.iter().copied().collect(),
        months,
        weekdays: WEEKDAY_NAMES.split_whitespace().collect(),
        currency: format!("(?i:{currency})"),
        currency_markers,
        month_alternation,
    }
});

// ---------------------------------------------------------------------------------------------
// Compiled patterns
// ---------------------------------------------------------------------------------------------

/// `_CURRENCY` as written, kept as the reference the fast scanner is tested against.
#[cfg(test)]
static CURRENCY: LazyLock<Regex> = LazyLock::new(|| compile(&TABLES.currency));
static CURRENCY_FULL: LazyLock<Regex> = LazyLock::new(|| compile_full(&TABLES.currency));
static CURRENCY_SPLIT: LazyLock<Regex> = LazyLock::new(|| {
    let c = &TABLES.currency;
    compile_full(&format!("({c}){S}*(.*)|(.*?){S}*({c})"))
});
static SCALE: LazyLock<Regex> = LazyLock::new(|| {
    let words = longest_first(SCALE_WORDS.iter().map(|&(w, _)| w))
        .into_iter()
        .map(ci)
        .collect::<Vec<_>>()
        .join("|");
    compile(&format!(r"(?i){S}*({words})\.?{END}"))
});
static SUFFIX: LazyLock<Regex> = LazyLock::new(|| compile(&py(r"(?i)(?<=\d)(k|m|bn|b)$")));
static COUNTED: LazyLock<Regex> = LazyLock::new(|| {
    compile(&py(
        r"(?i)\b(?:money|pr[iıİ]ce|[iıİ]nvo[iıİ]ce|revenue|cost|count|quant[iıİ]ty|populat[iıİ]on|people|[iıİ]tems|un[iıİ]ts|users|downloads)\b",
    ))
});
static GROUPED: LazyLock<Regex> = LazyLock::new(|| {
    compile(&py(
        r"(?<![\d.,])\d{1,3}(?:([,.   '’])\d{3})+(?!\1)([.,])\d+(?![\d.,])",
    ))
});
static SHORT: LazyLock<Regex> =
    LazyLock::new(|| compile(&py(r"(?<![\d.,])\d+([.,])\d{1,2}(?![\d.,])")));
static REPEATED: LazyLock<Regex> =
    LazyLock::new(|| compile(&py(r"(?<![\d.,])\d{1,3}(?:([.,])\d{3}){2,}(?![\d.,])")));
static LOCALE: LazyLock<Regex> = LazyLock::new(|| {
    compile(&py(
        r#"\b(?:locale|language|lang)["']?\s*[=:]\s*["']?(en|fr|de|es|it|pt|nl|pl|ru|sv|ja|zh|ko)(?:[-_][a-z]{2})?\b"#,
    ))
});
static XML_DECLARATION: LazyLock<Regex> = LazyLock::new(|| compile(r"<\?[^>]*\?>"));
static NUMBER_SHAPE: LazyLock<Regex> = LazyLock::new(|| compile_full(&py(r"\d[\d., '’]*")));
static SPACE_GROUPING: LazyLock<Regex> =
    LazyLock::new(|| compile_full(&py(r"(\d{1,3}(?:[ '’]\d{3})+)([.,]\d+)?")));
static LAKH_GROUPING: LazyLock<Regex> =
    LazyLock::new(|| compile_full(&py(r"\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?")));
static RUPEE: LazyLock<Regex> =
    LazyLock::new(|| compile(&py(r"(?i)₹|\b[iıİ]NR\b|\bRs\b|lakh|crore")));
static POINT_DOT: LazyLock<Regex> = LazyLock::new(|| compile_full(&py(r"\d{1,3}(?:,\d{3})+\.\d+")));
static POINT_COMMA: LazyLock<Regex> =
    LazyLock::new(|| compile_full(&py(r"\d{1,3}(?:\.\d{3})+,\d+")));

static CJK_DATE: LazyLock<Regex> = LazyLock::new(|| {
    compile_full(&py(
        r"(\d{4})\s*[年년]\s*(\d{1,2})\s*[月월]\s*(\d{1,2})\s*[日일]",
    ))
});
const NUMERIC_DATE_PATTERN: &str = r"(?<!\d)(\d{1,4})([/.-])(\d{1,2})\2(\d{2,4})(?!\d)";
static NUMERIC_DATE: LazyLock<Regex> = LazyLock::new(|| compile(&py(NUMERIC_DATE_PATTERN)));
static NUMERIC_DATE_FULL: LazyLock<Regex> =
    LazyLock::new(|| compile_full(&py(NUMERIC_DATE_PATTERN)));
static WORD_DATE: LazyLock<Regex> = LazyLock::new(|| {
    let m = &TABLES.month_alternation;
    compile(&py(&format!(
        r"\b(?:\d{{1,2}}\w*[ .]+(?:de\s+)?(?:{m})[ ,.]+(?:de\s+)?\d{{4}}|(?:{m})[ .]+\d{{1,2}}\w*[ ,.]+\d{{4}})\b"
    )))
});
static TIMED: LazyLock<Regex> = LazyLock::new(|| {
    compile_full(&py(
        r"(.+?)[T\s]+(\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:\d{2})?)",
    ))
});
static ORDINAL: LazyLock<Regex> = LazyLock::new(|| compile(&py(r"(?<=\d)(?:st|nd|rd|th|er)\b")));
static ORDINAL_MARK: LazyLock<Regex> = LazyLock::new(|| compile(&py(r"(?<=\d)\.?(?:o|a|°)\b")));
static FILLER: LazyLock<Regex> = LazyLock::new(|| compile(&py(r"\b(?:de|of|the|le|el|il|den)\b")));

// A declared numeric date format, in English, French (jj/mm/aaaa), German (tt.mm.jjjj), Spanish
// or Italian (gg/mm/aaaa).
const DAY: &str = r"(?:dd?|jj?|tt?|gg?|day|jour|tag|dia|giorno)";
const MONTH: &str = r"(?:mm?|month|mois|monat|mes|mese)";
const YEAR: &str = r"(?:yy(?:yy)?|aa(?:aa)?|jj(?:jj)?|year|annee|jahr|ano|anno)";
static DMY_FORMAT: LazyLock<Regex> = LazyLock::new(|| {
    compile(&py(&format!(
        r"\b{DAY}\s*[/.-]\s*{MONTH}\s*[/.-]\s*{YEAR}\b"
    )))
});
static MDY_FORMAT: LazyLock<Regex> = LazyLock::new(|| {
    compile(&py(&format!(
        r"\b{MONTH}\s*[/.-]\s*{DAY}\s*[/.-]\s*{YEAR}\b"
    )))
});
static DATE_LOCALE: LazyLock<Regex> = LazyLock::new(|| {
    compile(&py(
        r#"\b(?:locale|language|lang|langue|sprache|idioma|lingua)["']?\s*[=:]\s*["']?(en(?:[-_](?:us|gb|au|ca))?|fr(?:[-_]fr)?|de(?:[-_]de)?|es(?:[-_]es)?|it(?:[-_]it)?)\b"#,
    ))
});

// ---------------------------------------------------------------------------------------------
// `_CURRENCY` scanning
// ---------------------------------------------------------------------------------------------

// The backtracking VM needs microseconds per character for `_CURRENCY` (its `\b` and lookahead
// make it "fancy"), and contexts can be long documents. The scanner below runs the same ordered
// alternation: a plain-regex prefilter finds the leftmost position where any marker literal
// matches, then the alternatives are tried in order at that position with their `\b` and
// `(?![a-z])` checks, as CPython's backtracking does.

struct CurrencyAlternative {
    /// `^(?i:marker)` with CPython's `i`/`ı`/`İ` equivalence.
    literal: regex::Regex,
    /// `\b...(?![a-z])` around markers starting with an ASCII letter.
    bounded: bool,
}

static CURRENCY_ALTERNATIVES: LazyLock<Vec<CurrencyAlternative>> = LazyLock::new(|| {
    TABLES
        .currency_markers
        .iter()
        .map(|&c| CurrencyAlternative {
            literal: regex::Regex::new(&format!("^(?i:{})", ci(c))).expect("currency literal"),
            bounded: c.chars().next().is_some_and(|f| f.is_ascii_alphabetic()),
        })
        .collect()
});
static CURRENCY_PREFILTER: LazyLock<regex::Regex> = LazyLock::new(|| {
    let literals: Vec<String> = TABLES.currency_markers.iter().map(|c| ci(c)).collect();
    regex::Regex::new(&format!("(?i:{})", literals.join("|"))).expect("currency prefilter")
});
static PY_WORD: LazyLock<regex::Regex> =
    LazyLock::new(|| regex::Regex::new(&format!("^{W}$")).expect("word class"));

/// CPython `\w` for one character.
fn py_is_word(c: char) -> bool {
    PY_WORD.is_match(c.encode_utf8(&mut [0u8; 4]))
}

/// `(?i:[a-z])` in CPython: ASCII letters, KELVIN SIGN, LONG S, DOTLESS I and DOTTED CAPITAL I.
fn ci_ascii_letter(c: char) -> bool {
    c.is_ascii_alphabetic() || matches!(c, '\u{212a}' | '\u{17f}' | '\u{131}' | '\u{130}')
}

/// The end of the `_CURRENCY` match starting exactly at `at`, if any.
fn currency_at(text: &str, at: usize) -> Option<usize> {
    let rest = &text[at..];
    CURRENCY_ALTERNATIVES.iter().find_map(|alternative| {
        let end = at + alternative.literal.find(rest)?.end();
        if alternative.bounded {
            // The marker starts with a letter, so `\b` means "not after a word character".
            if text[..at].chars().next_back().is_some_and(py_is_word)
                || text[end..].chars().next().is_some_and(ci_ascii_letter)
            {
                return None;
            }
        }
        Some(end)
    })
}

/// `re.findall(_CURRENCY, text)`.
fn currency_find_all(text: &str) -> Vec<&str> {
    let mut found = Vec::new();
    let mut pos = 0;
    while let Some(candidate) = CURRENCY_PREFILTER.find_at(text, pos) {
        let start = candidate.start();
        match currency_at(text, start) {
            Some(end) => {
                found.push(&text[start..end]);
                pos = end;
            }
            None => pos = start + text[start..].chars().next().map_or(1, char::len_utf8),
        }
    }
    found
}

/// `re.search(_CURRENCY, text)`.
fn currency_search(text: &str) -> bool {
    let mut pos = 0;
    while let Some(candidate) = CURRENCY_PREFILTER.find_at(text, pos) {
        let start = candidate.start();
        if currency_at(text, start).is_some() {
            return true;
        }
        pos = start + text[start..].chars().next().map_or(1, char::len_utf8);
    }
    false
}

// ---------------------------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------------------------

/// Only an explicitly written percent or currency marker; a currency sign stays a sign.
pub fn written_unit(text: &str) -> Option<String> {
    if py_rstrip(text).ends_with('%') {
        return Some("%".to_owned());
    }
    let markers = currency_find_all(text);
    if markers.len() != 1 {
        return None;
    }
    // A marker whose casefold is not in the table (`ınr`) raises `KeyError` in Python.
    TABLES
        .names
        .get(&py_casefold(markers[0]))
        .map(|&name| name.to_owned())
}

/// The decimal marks of the currencies `text` writes (`""` for whole units); a currency of both
/// says nothing.
fn currency_marks(text: &str) -> R<BTreeSet<&'static str>> {
    let mut marks = BTreeSet::new();
    for marker in currency_find_all(text) {
        match TABLES.marks.get(&py_casefold(marker)) {
            None => return Err(Raised),
            Some(None) => {}
            Some(Some(mark)) => {
                marks.insert(*mark);
            }
        }
    }
    Ok(marks)
}

fn mark_of(s: &str) -> &'static str {
    if s == "." { "." } else { "," }
}

/// The decimal mark (`""` when amounts have none), from the first evidence that speaks: the
/// span's own currency, the document's other numbers, the document's currencies, a declared
/// locale.
fn decimal_mark(written: &str, context: &str) -> R<Option<&'static str>> {
    let context = sub(&XML_DECLARATION, context, "")?;
    let mut numbers = BTreeSet::new();
    for m in group_all(&GROUPED, &context, 2)? {
        numbers.insert(mark_of(m));
    }
    for m in group_all(&SHORT, &context, 1)? {
        numbers.insert(mark_of(m));
    }
    for m in group_all(&REPEATED, &context, 1)? {
        numbers.insert(if m == "." { "," } else { "." });
    }
    let mut locales = BTreeSet::new();
    for lang in group_all(&LOCALE, &py_casefold(&context), 1)? {
        locales.insert(if matches!(lang, "en" | "ja" | "zh" | "ko") {
            "."
        } else {
            ","
        });
    }
    // Python builds the whole evidence tuple before picking from it.
    let evidence = [
        currency_marks(written)?,
        numbers,
        currency_marks(&context)?,
        locales,
    ];
    Ok(evidence
        .iter()
        .find(|marks| !marks.is_empty())
        .and_then(one))
}

/// `_digits`: the number as plain digits and a point.
fn digits(text: &str, written: &str, context: &str) -> R<Option<String>> {
    if !is_match(&NUMBER_SHAPE, text)? {
        return Ok(None);
    }
    let groups_only = |c: char| matches!(c, ' ' | '\'' | '\u{2019}');
    if text.contains(groups_only) {
        // A space or an apostrophe only groups.
        let Some(caps) = captures(&SPACE_GROUPING, text)? else {
            return Ok(None);
        };
        let mut plain: String = group(&caps, 1)
            .chars()
            .filter(|&c| !groups_only(c))
            .collect();
        plain.push_str(group(&caps, 2));
        return digits(&plain, written, context);
    }
    if is_match(&LAKH_GROUPING, text)? && is_match(&RUPEE, &format!("{written}\n{context}"))? {
        return Ok(Some(text.replace(',', "")));
    }
    let (dot, comma) = (text.rfind('.'), text.rfind(','));
    if let (Some(dot), Some(comma)) = (dot, comma) {
        let (point, grouping, shape) = if dot > comma {
            ('.', ',', &*POINT_DOT)
        } else {
            (',', '.', &*POINT_COMMA)
        };
        if !is_match(shape, text)? {
            return Ok(None);
        }
        return Ok(Some(text.replace(grouping, "").replace(point, ".")));
    }
    let mark = match (dot, comma) {
        (None, None) => return Ok(Some(text.to_owned())),
        (Some(_), _) => '.',
        _ => ',',
    };
    let parts: Vec<&str> = text.split(mark).collect();
    if parts.iter().any(|p| p.is_empty()) {
        return Ok(None);
    }
    let point = decimal_mark(written, context)?;
    let mark_str = if mark == '.' { "." } else { "," };
    if parts.len() > 2 || point.is_some_and(|p| p != mark_str) {
        // The mark groups: complete threes.
        let complete = (1..=3).contains(&clen(parts[0])) && parts[1..].iter().all(|p| clen(p) == 3);
        return Ok(complete.then(|| parts.concat()));
    }
    if point.is_none() && clen(parts[1]) == 3 && clen(parts[0]) <= 3 {
        // 1,250: a group or a decimal, unsettled.
        return Ok(None);
    }
    Ok(Some(parts.join(".")))
}

const DECIMAL_PREC: usize = 28;
const DECIMAL_EMAX: i64 = 999_999;
const DECIMAL_EMIN: i64 = -999_999;
const DECIMAL_ETINY: i64 = DECIMAL_EMIN - DECIMAL_PREC as i64 + 1;

/// Drops the `n` lowest digits of `coef` (most significant first, no leading zeros), rounding half
/// to even; `coef` becomes empty for zero.
fn round_off(coef: &mut Vec<u8>, exp: &mut i64, n: usize) {
    *exp += n as i64;
    let len = coef.len();
    if n > len {
        // Below half a unit of the kept position.
        coef.clear();
        return;
    }
    let dropped = coef.split_off(len - n);
    let up = match dropped.first() {
        None => false,
        Some(&first) if first > 5 => true,
        Some(&first) if first < 5 => false,
        Some(_) => dropped[1..].iter().any(|&d| d != 0) || coef.last().is_some_and(|d| d % 2 == 1),
    };
    if up {
        let mut i = coef.len();
        loop {
            if i == 0 {
                coef.insert(0, 1);
                break;
            }
            i -= 1;
            if coef[i] == 9 {
                coef[i] = 0;
            } else {
                coef[i] += 1;
                break;
            }
        }
    }
    while coef.first() == Some(&0) {
        coef.remove(0);
    }
}

/// `Decimal(sign + digits) * 10**pow10`, then `int(value)` when integral, else `float(value)`.
fn decimal_number(negative: bool, digits: &str, pow10: u32) -> R<Option<Number>> {
    let (int_part, frac_part) = digits.split_once('.').unwrap_or((digits, ""));
    let mut coef: Vec<u8> = Vec::with_capacity(digits.len() + pow10 as usize);
    for c in int_part.chars().chain(frac_part.chars()) {
        coef.push(nd_value(c).ok_or(Raised)? as u8);
    }
    let leading = coef.iter().take_while(|&&d| d == 0).count();
    coef.drain(..leading);
    if coef.is_empty() {
        // Zero is integral, and int(-0) == 0.
        return Ok(Some(Number::Int(0)));
    }
    let mut exp = -(clen(frac_part) as i64);
    coef.extend(std::iter::repeat_n(0, pow10 as usize));
    let adjusted = coef.len() as i64 - 1 + exp;
    if adjusted < DECIMAL_EMIN && exp < DECIMAL_ETINY {
        let shift = (DECIMAL_ETINY - exp) as usize;
        round_off(&mut coef, &mut exp, shift);
    } else if coef.len() > DECIMAL_PREC {
        let excess = coef.len() - DECIMAL_PREC;
        round_off(&mut coef, &mut exp, excess);
        if coef.len() > DECIMAL_PREC {
            // 99..9 rounded up to 10^28: one more exact trailing zero goes.
            coef.pop();
            exp += 1;
        }
    }
    if coef.is_empty() {
        return Ok(Some(Number::Int(0)));
    }
    if coef.len() as i64 - 1 + exp > DECIMAL_EMAX {
        // decimal.Overflow is trapped by the default context.
        return Err(Raised);
    }
    let sign = if negative { "-" } else { "" };
    let fraction_digits = usize::try_from(-exp).unwrap_or(0);
    let integral = fraction_digits == 0
        || (fraction_digits < coef.len()
            && coef[coef.len() - fraction_digits..].iter().all(|&d| d == 0));
    let text = |digits: &[u8]| -> String { digits.iter().map(|&d| char::from(b'0' + d)).collect() };
    if integral {
        let mut int = text(&coef[..coef.len() - fraction_digits]);
        if exp > 0 {
            int.extend(std::iter::repeat_n('0', exp as usize));
        }
        let int = format!("{sign}{int}");
        return Ok(Some(match int.parse::<i128>() {
            Ok(value) => Number::Int(value),
            Err(_) => Number::BigInt(int),
        }));
    }
    let value: f64 = format!("{sign}{}e{exp}", text(&coef))
        .parse()
        .map_err(|_| Raised)?;
    Ok(value.is_finite().then_some(Number::Float(value)))
}

/// The written magnitude, percent points included: its currency and percent sign set apart, its
/// scale word applied; no implicit unit and no conversion. `None` when the writing does not
/// settle one value.
pub fn parse_number(text: &str, context: &str, unit: Option<&str>) -> Option<Number> {
    parse_number_impl(text, context, unit).ok().flatten()
}

fn parse_number_impl(text: &str, context: &str, unit: Option<&str>) -> R<Option<Number>> {
    let mut context = context.to_owned();
    if let Some(unit) = unit.filter(|u| !u.is_empty())
        && is_match(&CURRENCY_FULL, py_strip(unit))?
    {
        context.push('\n');
        context.push_str(unit);
    }
    let written = text;
    let mut text = py_strip(text)
        .replace('\u{2212}', "-")
        .replace(['\u{a0}', '\u{202f}'], " ");
    if let Some(stripped) = text.strip_suffix('%') {
        text = stripped.to_owned();
    }
    text = py_rstrip(&text).to_owned();
    let negative = text.starts_with('(') && text.ends_with(')');
    if negative {
        text = if text.len() >= 2 {
            py_strip(&text[1..text.len() - 1]).to_owned()
        } else {
            String::new()
        };
    }
    let mut sign = None;
    let mut currency = false;
    for _ in 0..2 {
        // One currency and one sign, in either order.
        if !currency {
            let rest = captures(&CURRENCY_SPLIT, &text)?.map(|caps| {
                let rest = if caps.get(1).is_some() {
                    group(&caps, 2)
                } else {
                    group(&caps, 3)
                };
                py_strip(rest).to_owned()
            });
            if let Some(rest) = rest {
                currency = true;
                text = rest;
            }
        }
        if sign.is_none()
            && let Some(c @ ('-' | '+')) = text.chars().next()
        {
            sign = Some(c);
            text = py_strip(&text[1..]).to_owned();
        }
    }
    if negative && sign.is_some() {
        return Ok(None);
    }
    // `None` stands for Python's `scale == 0` (German "Billion").
    let mut pow10 = Some(0);
    let scale_word = captures(&SCALE, &text)?.map(|caps| {
        (
            caps.get(0).map_or(0, |m| m.start()),
            group(&caps, 1).to_owned(),
        )
    });
    if let Some((start, word)) = scale_word {
        pow10 = if word == "Billion" {
            None
        } else {
            // `SCALES[word.lower()]` raises `KeyError` for `ı`/`İ`/`ſ` spellings.
            Some(
                *TABLES
                    .scales
                    .get(word.to_lowercase().as_str())
                    .ok_or(Raised)?,
            )
        };
        text.truncate(start);
    } else if let Some((start, suffix)) = captures(&SUFFIX, &text)?.map(|caps| {
        (
            caps.get(0).map_or(0, |m| m.start()),
            group(&caps, 1).to_lowercase(),
        )
    }) {
        // Attached k, M, B: a magnitude only beside money or a count.
        if !(currency_search(&format!("{written}\n{context}")) || is_match(&COUNTED, &context)?) {
            return Ok(None);
        }
        pow10 = Some(match suffix.as_str() {
            "k" => 3,
            "m" => 6,
            "b" | "bn" => 9,
            _ => return Err(Raised),
        });
        text.truncate(start);
    }
    let digits = digits(&text, written, &context)?;
    let (Some(digits), Some(pow10)) = (digits, pow10) else {
        return Ok(None);
    };
    decimal_number(negative || sign == Some('-'), &digits, pow10)
}

// ---------------------------------------------------------------------------------------------
// Dates
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Order {
    Dmy,
    Mdy,
}

/// The context, folded and its date order computed once, when a date needs them.
struct DateContext<'a> {
    context: &'a str,
    folded: OnceCell<String>,
    order: OnceCell<R<Option<Order>>>,
}

impl DateContext<'_> {
    fn folded(&self) -> &str {
        self.folded.get_or_init(|| fold(self.context))
    }

    fn order(&self) -> R<Option<Order>> {
        *self.order.get_or_init(|| date_order(self))
    }
}

/// `datetime.date(year, month, day).isoformat()`, `None` on `ValueError`.
fn iso_date(year: u64, month: u64, day: u64) -> Option<String> {
    if !(1..=9999).contains(&year) || !(1..=12).contains(&month) {
        return None;
    }
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days = match month {
        2 if leap => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    };
    (1..=days)
        .contains(&day)
        .then(|| format!("{year:04}-{month:02}-{day:02}"))
}

/// Declared format/locale first, then the state's other unambiguous numeric dates; conflicts stay
/// unresolved.
fn date_order(ctx: &DateContext<'_>) -> R<Option<Order>> {
    let folded = ctx.folded();
    let mut explicit = BTreeSet::new();
    if is_match(&DMY_FORMAT, folded)? {
        explicit.insert(Order::Dmy);
    }
    if is_match(&MDY_FORMAT, folded)? {
        explicit.insert(Order::Mdy);
    }
    if !explicit.is_empty() {
        return Ok(one(&explicit));
    }
    let locales = group_all(&DATE_LOCALE, folded, 1)?;
    if !locales.is_empty() {
        let orders: BTreeSet<Order> = locales
            .iter()
            .map(|locale| match locale.replace('_', "-").as_str() {
                "en" | "en-us" => Order::Mdy,
                _ => Order::Dmy,
            })
            .collect();
        return Ok(one(&orders));
    }
    let mut orders = BTreeSet::new();
    for caps in NUMERIC_DATE.captures_iter(ctx.context) {
        let caps = caps.map_err(|_| Raised)?;
        let (a, b, year) = (
            py_int(group(&caps, 1)),
            py_int(group(&caps, 3)),
            py_int(group(&caps, 4)),
        );
        if clen(group(&caps, 4)) != 4 || a > 31 || b > 31 || a < 1 || b < 1 {
            continue;
        }
        for (day, month, order) in [(a, b, Order::Dmy), (b, a, Order::Mdy)] {
            if iso_date(year, month, day).is_none() {
                continue;
            }
            if (a > 12) != (b > 12) {
                orders.insert(order);
            }
        }
    }
    Ok(one(&orders))
}

/// `(year, month, day)` for `a`/`b`/year: day first, month first, or the one the context's order
/// says.
fn candidates(year: u64, a: u64, b: u64, order: Option<Order>) -> Vec<(u64, u64, u64)> {
    let (dmy, mdy) = ((year, b, a), (year, a, b));
    match order {
        None => vec![dmy, mdy],
        Some(Order::Dmy) => vec![dmy],
        Some(Order::Mdy) => vec![mdy],
    }
}

/// `datetime.fromisoformat(date + "T" + clock).isoformat()` (CPython 3.12), `None` on
/// `ValueError`. `clock` has the written shape `HH:MM[:SS[.f+]][Z|±HH:MM]`.
fn from_isoformat(date: &str, clock: &str) -> Option<String> {
    // A date that already carries a time (a nested timed span) leaves a second "T" the time
    // parser rejects; CPython also parses ASCII digits only.
    if date.len() != 10 || !clock.is_ascii() {
        return None;
    }
    let b = clock.as_bytes();
    let mut i = 0;
    let two = |i: &mut usize| -> Option<u32> {
        let d = b.get(*i..*i + 2)?;
        if !d.iter().all(u8::is_ascii_digit) {
            return None;
        }
        *i += 2;
        Some(u32::from(d[0] - b'0') * 10 + u32::from(d[1] - b'0'))
    };
    let hour = two(&mut i)?;
    (b.get(i) == Some(&b':')).then_some(())?;
    i += 1;
    let minute = two(&mut i)?;
    let mut second = 0;
    let mut microsecond = 0;
    if b.get(i) == Some(&b':') {
        i += 1;
        second = two(&mut i)?;
        if b.get(i) == Some(&b'.') {
            i += 1;
            let start = i;
            while b.get(i).is_some_and(u8::is_ascii_digit) {
                i += 1;
            }
            let fraction = &b[start..i];
            if fraction.is_empty() {
                return None;
            }
            // Up to six digits are read; the others are truncated.
            let kept = &fraction[..fraction.len().min(6)];
            microsecond = kept
                .iter()
                .fold(0u32, |acc, d| acc * 10 + u32::from(d - b'0'))
                * 10u32.pow(6 - kept.len() as u32);
        }
    }
    let offset = match b.get(i) {
        None => None,
        Some(&b'Z') if i + 1 == b.len() => Some(0i64),
        Some(&sign @ (b'+' | b'-')) => {
            i += 1;
            let tz_hour = two(&mut i)?;
            (b.get(i) == Some(&b':')).then_some(())?;
            i += 1;
            let tz_minute = two(&mut i)?;
            if i != b.len() {
                return None;
            }
            // The offset minutes are not range checked; only |offset| < 24h is.
            let seconds = i64::from(tz_hour) * 3600 + i64::from(tz_minute) * 60;
            if seconds >= 86_400 {
                return None;
            }
            Some(if sign == b'-' { -seconds } else { seconds })
        }
        Some(_) => return None,
    };
    if hour > 23 || minute > 59 || second > 59 {
        return None;
    }
    let mut out = format!("{date}T{hour:02}:{minute:02}:{second:02}");
    if microsecond != 0 {
        out.push_str(&format!(".{microsecond:06}"));
    }
    if let Some(offset) = offset {
        let sign = if offset < 0 { '-' } else { '+' };
        let offset = offset.abs();
        out.push_str(&format!(
            "{sign}{:02}:{:02}",
            offset / 3600,
            offset % 3600 / 60
        ));
    }
    Some(out)
}

/// ISO value and unresolved-ambiguity flag; an invalid or unrecognised span is `(None, false)`.
pub fn parse_date(text: &str, context: &str) -> (Option<String>, bool) {
    let ctx = DateContext {
        context,
        folded: OnceCell::new(),
        order: OnceCell::new(),
    };
    parse_date_impl(text, &ctx).unwrap_or((None, false))
}

fn parse_date_impl(text: &str, ctx: &DateContext<'_>) -> R<(Option<String>, bool)> {
    let text = py_strip(text);
    if text.is_empty() {
        return Ok((None, false));
    }
    // Keep an explicitly written time and timezone, with no default time or timezone.
    if let Some(caps) = captures(&TIMED, text)? {
        let (date, ambiguous) = parse_date_impl(group(&caps, 1), ctx)?;
        let Some(date) = date else {
            return Ok((None, ambiguous));
        };
        let mut clock = group(&caps, 2).to_owned();
        if clock.chars().nth(1) == Some(':') {
            clock.insert(0, '0');
        }
        return Ok((from_isoformat(&date, &clock), false));
    }
    if let Some(caps) = captures(&NUMERIC_DATE_FULL, text)? {
        let (first, last) = (group(&caps, 1), group(&caps, 4));
        let (a, b, c) = (py_int(first), py_int(group(&caps, 3)), py_int(last));
        let candidates = if clen(first) == 4 {
            vec![(a, b, c)]
        } else if clen(last) == 4 {
            candidates(c, a, b, ctx.order()?)
        } else {
            // A century is never invented: another unambiguous complete date must establish it.
            if clen(last) != 2 {
                return Ok((None, false));
            }
            let mut complete: Vec<&str> = Vec::new();
            for caps in NUMERIC_DATE.captures_iter(ctx.context) {
                let caps = caps.map_err(|_| Raised)?;
                if clen(group(&caps, 1)) == 4 || clen(group(&caps, 4)) == 4 {
                    complete.push(group(&caps, 0));
                }
            }
            complete.extend(find_all(&WORD_DATE, ctx.folded())?);
            let mut centuries = BTreeSet::new();
            for candidate in complete {
                if let (Some(value), false) = parse_date_impl(candidate, ctx)? {
                    centuries.insert(py_int(&value[..4]) / 100 * 100);
                }
            }
            let Some(century) = one(&centuries) else {
                return Ok((None, false));
            };
            candidates(century + c, a, b, ctx.order()?)
        };
        let values: BTreeSet<String> = candidates
            .into_iter()
            .filter_map(|(year, month, day)| iso_date(year, month, day))
            .collect();
        if values.len() > 1 {
            return Ok((None, true));
        }
        return Ok((values.into_iter().next(), false));
    }
    if let Some(caps) = captures(&CJK_DATE, text)? {
        let [year, month, day] = [1, 2, 3].map(|i| py_int(group(&caps, i)));
        return Ok((iso_date(year, month, day), false));
    }
    let words = fold(text);
    let words = sub(&ORDINAL, &words, "")?;
    let words = sub(&ORDINAL_MARK, &words, "")?;
    let words = sub(&FILLER, &words, " ")?;
    let words = words.replace([',', '.'], " ");
    let mut words = py_split(&words);
    if let Some(first) = words.first().copied()
        && TABLES
            .weekdays
            .contains(first.strip_suffix("-feira").unwrap_or(first))
    {
        // Tue, 20 Oct 2026; segunda-feira, 20 de ...
        words.remove(0);
    }
    let [first, second, year] = words[..] else {
        return Ok((None, false));
    };
    let (month, day) = if let Some(&month) = TABLES.months.get(first) {
        (month, second)
    } else if let Some(&month) = TABLES.months.get(second) {
        (month, first)
    } else {
        return Ok((None, false));
    };
    // `str.isdigit()` admits digits `int()` then rejects with the same `(None, False)`.
    if !all_nd(day) || !all_nd(year) || clen(year) != 4 {
        return Ok((None, false));
    }
    Ok((iso_date(py_int(year), u64::from(month), py_int(day)), false))
}

// ---------------------------------------------------------------------------------------------
// Unicode tables from CPython
// ---------------------------------------------------------------------------------------------

// Generated from CPython 3.12.14, unicodedata 15.0.0.
const CASEFOLD_EXCEPTIONS: &[(char, &str)] = &[
    ('\u{b5}', "\u{3bc}"),
    ('\u{df}', "\u{73}\u{73}"),
    ('\u{149}', "\u{2bc}\u{6e}"),
    ('\u{17f}', "\u{73}"),
    ('\u{1f0}', "\u{6a}\u{30c}"),
    ('\u{345}', "\u{3b9}"),
    ('\u{390}', "\u{3b9}\u{308}\u{301}"),
    ('\u{3b0}', "\u{3c5}\u{308}\u{301}"),
    ('\u{3c2}', "\u{3c3}"),
    ('\u{3d0}', "\u{3b2}"),
    ('\u{3d1}', "\u{3b8}"),
    ('\u{3d5}', "\u{3c6}"),
    ('\u{3d6}', "\u{3c0}"),
    ('\u{3f0}', "\u{3ba}"),
    ('\u{3f1}', "\u{3c1}"),
    ('\u{3f5}', "\u{3b5}"),
    ('\u{587}', "\u{565}\u{582}"),
    ('\u{13a0}', "\u{13a0}"),
    ('\u{13a1}', "\u{13a1}"),
    ('\u{13a2}', "\u{13a2}"),
    ('\u{13a3}', "\u{13a3}"),
    ('\u{13a4}', "\u{13a4}"),
    ('\u{13a5}', "\u{13a5}"),
    ('\u{13a6}', "\u{13a6}"),
    ('\u{13a7}', "\u{13a7}"),
    ('\u{13a8}', "\u{13a8}"),
    ('\u{13a9}', "\u{13a9}"),
    ('\u{13aa}', "\u{13aa}"),
    ('\u{13ab}', "\u{13ab}"),
    ('\u{13ac}', "\u{13ac}"),
    ('\u{13ad}', "\u{13ad}"),
    ('\u{13ae}', "\u{13ae}"),
    ('\u{13af}', "\u{13af}"),
    ('\u{13b0}', "\u{13b0}"),
    ('\u{13b1}', "\u{13b1}"),
    ('\u{13b2}', "\u{13b2}"),
    ('\u{13b3}', "\u{13b3}"),
    ('\u{13b4}', "\u{13b4}"),
    ('\u{13b5}', "\u{13b5}"),
    ('\u{13b6}', "\u{13b6}"),
    ('\u{13b7}', "\u{13b7}"),
    ('\u{13b8}', "\u{13b8}"),
    ('\u{13b9}', "\u{13b9}"),
    ('\u{13ba}', "\u{13ba}"),
    ('\u{13bb}', "\u{13bb}"),
    ('\u{13bc}', "\u{13bc}"),
    ('\u{13bd}', "\u{13bd}"),
    ('\u{13be}', "\u{13be}"),
    ('\u{13bf}', "\u{13bf}"),
    ('\u{13c0}', "\u{13c0}"),
    ('\u{13c1}', "\u{13c1}"),
    ('\u{13c2}', "\u{13c2}"),
    ('\u{13c3}', "\u{13c3}"),
    ('\u{13c4}', "\u{13c4}"),
    ('\u{13c5}', "\u{13c5}"),
    ('\u{13c6}', "\u{13c6}"),
    ('\u{13c7}', "\u{13c7}"),
    ('\u{13c8}', "\u{13c8}"),
    ('\u{13c9}', "\u{13c9}"),
    ('\u{13ca}', "\u{13ca}"),
    ('\u{13cb}', "\u{13cb}"),
    ('\u{13cc}', "\u{13cc}"),
    ('\u{13cd}', "\u{13cd}"),
    ('\u{13ce}', "\u{13ce}"),
    ('\u{13cf}', "\u{13cf}"),
    ('\u{13d0}', "\u{13d0}"),
    ('\u{13d1}', "\u{13d1}"),
    ('\u{13d2}', "\u{13d2}"),
    ('\u{13d3}', "\u{13d3}"),
    ('\u{13d4}', "\u{13d4}"),
    ('\u{13d5}', "\u{13d5}"),
    ('\u{13d6}', "\u{13d6}"),
    ('\u{13d7}', "\u{13d7}"),
    ('\u{13d8}', "\u{13d8}"),
    ('\u{13d9}', "\u{13d9}"),
    ('\u{13da}', "\u{13da}"),
    ('\u{13db}', "\u{13db}"),
    ('\u{13dc}', "\u{13dc}"),
    ('\u{13dd}', "\u{13dd}"),
    ('\u{13de}', "\u{13de}"),
    ('\u{13df}', "\u{13df}"),
    ('\u{13e0}', "\u{13e0}"),
    ('\u{13e1}', "\u{13e1}"),
    ('\u{13e2}', "\u{13e2}"),
    ('\u{13e3}', "\u{13e3}"),
    ('\u{13e4}', "\u{13e4}"),
    ('\u{13e5}', "\u{13e5}"),
    ('\u{13e6}', "\u{13e6}"),
    ('\u{13e7}', "\u{13e7}"),
    ('\u{13e8}', "\u{13e8}"),
    ('\u{13e9}', "\u{13e9}"),
    ('\u{13ea}', "\u{13ea}"),
    ('\u{13eb}', "\u{13eb}"),
    ('\u{13ec}', "\u{13ec}"),
    ('\u{13ed}', "\u{13ed}"),
    ('\u{13ee}', "\u{13ee}"),
    ('\u{13ef}', "\u{13ef}"),
    ('\u{13f0}', "\u{13f0}"),
    ('\u{13f1}', "\u{13f1}"),
    ('\u{13f2}', "\u{13f2}"),
    ('\u{13f3}', "\u{13f3}"),
    ('\u{13f4}', "\u{13f4}"),
    ('\u{13f5}', "\u{13f5}"),
    ('\u{13f8}', "\u{13f0}"),
    ('\u{13f9}', "\u{13f1}"),
    ('\u{13fa}', "\u{13f2}"),
    ('\u{13fb}', "\u{13f3}"),
    ('\u{13fc}', "\u{13f4}"),
    ('\u{13fd}', "\u{13f5}"),
    ('\u{1c80}', "\u{432}"),
    ('\u{1c81}', "\u{434}"),
    ('\u{1c82}', "\u{43e}"),
    ('\u{1c83}', "\u{441}"),
    ('\u{1c84}', "\u{442}"),
    ('\u{1c85}', "\u{442}"),
    ('\u{1c86}', "\u{44a}"),
    ('\u{1c87}', "\u{463}"),
    ('\u{1c88}', "\u{a64b}"),
    ('\u{1e96}', "\u{68}\u{331}"),
    ('\u{1e97}', "\u{74}\u{308}"),
    ('\u{1e98}', "\u{77}\u{30a}"),
    ('\u{1e99}', "\u{79}\u{30a}"),
    ('\u{1e9a}', "\u{61}\u{2be}"),
    ('\u{1e9b}', "\u{1e61}"),
    ('\u{1e9e}', "\u{73}\u{73}"),
    ('\u{1f50}', "\u{3c5}\u{313}"),
    ('\u{1f52}', "\u{3c5}\u{313}\u{300}"),
    ('\u{1f54}', "\u{3c5}\u{313}\u{301}"),
    ('\u{1f56}', "\u{3c5}\u{313}\u{342}"),
    ('\u{1f80}', "\u{1f00}\u{3b9}"),
    ('\u{1f81}', "\u{1f01}\u{3b9}"),
    ('\u{1f82}', "\u{1f02}\u{3b9}"),
    ('\u{1f83}', "\u{1f03}\u{3b9}"),
    ('\u{1f84}', "\u{1f04}\u{3b9}"),
    ('\u{1f85}', "\u{1f05}\u{3b9}"),
    ('\u{1f86}', "\u{1f06}\u{3b9}"),
    ('\u{1f87}', "\u{1f07}\u{3b9}"),
    ('\u{1f88}', "\u{1f00}\u{3b9}"),
    ('\u{1f89}', "\u{1f01}\u{3b9}"),
    ('\u{1f8a}', "\u{1f02}\u{3b9}"),
    ('\u{1f8b}', "\u{1f03}\u{3b9}"),
    ('\u{1f8c}', "\u{1f04}\u{3b9}"),
    ('\u{1f8d}', "\u{1f05}\u{3b9}"),
    ('\u{1f8e}', "\u{1f06}\u{3b9}"),
    ('\u{1f8f}', "\u{1f07}\u{3b9}"),
    ('\u{1f90}', "\u{1f20}\u{3b9}"),
    ('\u{1f91}', "\u{1f21}\u{3b9}"),
    ('\u{1f92}', "\u{1f22}\u{3b9}"),
    ('\u{1f93}', "\u{1f23}\u{3b9}"),
    ('\u{1f94}', "\u{1f24}\u{3b9}"),
    ('\u{1f95}', "\u{1f25}\u{3b9}"),
    ('\u{1f96}', "\u{1f26}\u{3b9}"),
    ('\u{1f97}', "\u{1f27}\u{3b9}"),
    ('\u{1f98}', "\u{1f20}\u{3b9}"),
    ('\u{1f99}', "\u{1f21}\u{3b9}"),
    ('\u{1f9a}', "\u{1f22}\u{3b9}"),
    ('\u{1f9b}', "\u{1f23}\u{3b9}"),
    ('\u{1f9c}', "\u{1f24}\u{3b9}"),
    ('\u{1f9d}', "\u{1f25}\u{3b9}"),
    ('\u{1f9e}', "\u{1f26}\u{3b9}"),
    ('\u{1f9f}', "\u{1f27}\u{3b9}"),
    ('\u{1fa0}', "\u{1f60}\u{3b9}"),
    ('\u{1fa1}', "\u{1f61}\u{3b9}"),
    ('\u{1fa2}', "\u{1f62}\u{3b9}"),
    ('\u{1fa3}', "\u{1f63}\u{3b9}"),
    ('\u{1fa4}', "\u{1f64}\u{3b9}"),
    ('\u{1fa5}', "\u{1f65}\u{3b9}"),
    ('\u{1fa6}', "\u{1f66}\u{3b9}"),
    ('\u{1fa7}', "\u{1f67}\u{3b9}"),
    ('\u{1fa8}', "\u{1f60}\u{3b9}"),
    ('\u{1fa9}', "\u{1f61}\u{3b9}"),
    ('\u{1faa}', "\u{1f62}\u{3b9}"),
    ('\u{1fab}', "\u{1f63}\u{3b9}"),
    ('\u{1fac}', "\u{1f64}\u{3b9}"),
    ('\u{1fad}', "\u{1f65}\u{3b9}"),
    ('\u{1fae}', "\u{1f66}\u{3b9}"),
    ('\u{1faf}', "\u{1f67}\u{3b9}"),
    ('\u{1fb2}', "\u{1f70}\u{3b9}"),
    ('\u{1fb3}', "\u{3b1}\u{3b9}"),
    ('\u{1fb4}', "\u{3ac}\u{3b9}"),
    ('\u{1fb6}', "\u{3b1}\u{342}"),
    ('\u{1fb7}', "\u{3b1}\u{342}\u{3b9}"),
    ('\u{1fbc}', "\u{3b1}\u{3b9}"),
    ('\u{1fbe}', "\u{3b9}"),
    ('\u{1fc2}', "\u{1f74}\u{3b9}"),
    ('\u{1fc3}', "\u{3b7}\u{3b9}"),
    ('\u{1fc4}', "\u{3ae}\u{3b9}"),
    ('\u{1fc6}', "\u{3b7}\u{342}"),
    ('\u{1fc7}', "\u{3b7}\u{342}\u{3b9}"),
    ('\u{1fcc}', "\u{3b7}\u{3b9}"),
    ('\u{1fd2}', "\u{3b9}\u{308}\u{300}"),
    ('\u{1fd3}', "\u{3b9}\u{308}\u{301}"),
    ('\u{1fd6}', "\u{3b9}\u{342}"),
    ('\u{1fd7}', "\u{3b9}\u{308}\u{342}"),
    ('\u{1fe2}', "\u{3c5}\u{308}\u{300}"),
    ('\u{1fe3}', "\u{3c5}\u{308}\u{301}"),
    ('\u{1fe4}', "\u{3c1}\u{313}"),
    ('\u{1fe6}', "\u{3c5}\u{342}"),
    ('\u{1fe7}', "\u{3c5}\u{308}\u{342}"),
    ('\u{1ff2}', "\u{1f7c}\u{3b9}"),
    ('\u{1ff3}', "\u{3c9}\u{3b9}"),
    ('\u{1ff4}', "\u{3ce}\u{3b9}"),
    ('\u{1ff6}', "\u{3c9}\u{342}"),
    ('\u{1ff7}', "\u{3c9}\u{342}\u{3b9}"),
    ('\u{1ffc}', "\u{3c9}\u{3b9}"),
    ('\u{ab70}', "\u{13a0}"),
    ('\u{ab71}', "\u{13a1}"),
    ('\u{ab72}', "\u{13a2}"),
    ('\u{ab73}', "\u{13a3}"),
    ('\u{ab74}', "\u{13a4}"),
    ('\u{ab75}', "\u{13a5}"),
    ('\u{ab76}', "\u{13a6}"),
    ('\u{ab77}', "\u{13a7}"),
    ('\u{ab78}', "\u{13a8}"),
    ('\u{ab79}', "\u{13a9}"),
    ('\u{ab7a}', "\u{13aa}"),
    ('\u{ab7b}', "\u{13ab}"),
    ('\u{ab7c}', "\u{13ac}"),
    ('\u{ab7d}', "\u{13ad}"),
    ('\u{ab7e}', "\u{13ae}"),
    ('\u{ab7f}', "\u{13af}"),
    ('\u{ab80}', "\u{13b0}"),
    ('\u{ab81}', "\u{13b1}"),
    ('\u{ab82}', "\u{13b2}"),
    ('\u{ab83}', "\u{13b3}"),
    ('\u{ab84}', "\u{13b4}"),
    ('\u{ab85}', "\u{13b5}"),
    ('\u{ab86}', "\u{13b6}"),
    ('\u{ab87}', "\u{13b7}"),
    ('\u{ab88}', "\u{13b8}"),
    ('\u{ab89}', "\u{13b9}"),
    ('\u{ab8a}', "\u{13ba}"),
    ('\u{ab8b}', "\u{13bb}"),
    ('\u{ab8c}', "\u{13bc}"),
    ('\u{ab8d}', "\u{13bd}"),
    ('\u{ab8e}', "\u{13be}"),
    ('\u{ab8f}', "\u{13bf}"),
    ('\u{ab90}', "\u{13c0}"),
    ('\u{ab91}', "\u{13c1}"),
    ('\u{ab92}', "\u{13c2}"),
    ('\u{ab93}', "\u{13c3}"),
    ('\u{ab94}', "\u{13c4}"),
    ('\u{ab95}', "\u{13c5}"),
    ('\u{ab96}', "\u{13c6}"),
    ('\u{ab97}', "\u{13c7}"),
    ('\u{ab98}', "\u{13c8}"),
    ('\u{ab99}', "\u{13c9}"),
    ('\u{ab9a}', "\u{13ca}"),
    ('\u{ab9b}', "\u{13cb}"),
    ('\u{ab9c}', "\u{13cc}"),
    ('\u{ab9d}', "\u{13cd}"),
    ('\u{ab9e}', "\u{13ce}"),
    ('\u{ab9f}', "\u{13cf}"),
    ('\u{aba0}', "\u{13d0}"),
    ('\u{aba1}', "\u{13d1}"),
    ('\u{aba2}', "\u{13d2}"),
    ('\u{aba3}', "\u{13d3}"),
    ('\u{aba4}', "\u{13d4}"),
    ('\u{aba5}', "\u{13d5}"),
    ('\u{aba6}', "\u{13d6}"),
    ('\u{aba7}', "\u{13d7}"),
    ('\u{aba8}', "\u{13d8}"),
    ('\u{aba9}', "\u{13d9}"),
    ('\u{abaa}', "\u{13da}"),
    ('\u{abab}', "\u{13db}"),
    ('\u{abac}', "\u{13dc}"),
    ('\u{abad}', "\u{13dd}"),
    ('\u{abae}', "\u{13de}"),
    ('\u{abaf}', "\u{13df}"),
    ('\u{abb0}', "\u{13e0}"),
    ('\u{abb1}', "\u{13e1}"),
    ('\u{abb2}', "\u{13e2}"),
    ('\u{abb3}', "\u{13e3}"),
    ('\u{abb4}', "\u{13e4}"),
    ('\u{abb5}', "\u{13e5}"),
    ('\u{abb6}', "\u{13e6}"),
    ('\u{abb7}', "\u{13e7}"),
    ('\u{abb8}', "\u{13e8}"),
    ('\u{abb9}', "\u{13e9}"),
    ('\u{abba}', "\u{13ea}"),
    ('\u{abbb}', "\u{13eb}"),
    ('\u{abbc}', "\u{13ec}"),
    ('\u{abbd}', "\u{13ed}"),
    ('\u{abbe}', "\u{13ee}"),
    ('\u{abbf}', "\u{13ef}"),
    ('\u{fb00}', "\u{66}\u{66}"),
    ('\u{fb01}', "\u{66}\u{69}"),
    ('\u{fb02}', "\u{66}\u{6c}"),
    ('\u{fb03}', "\u{66}\u{66}\u{69}"),
    ('\u{fb04}', "\u{66}\u{66}\u{6c}"),
    ('\u{fb05}', "\u{73}\u{74}"),
    ('\u{fb06}', "\u{73}\u{74}"),
    ('\u{fb13}', "\u{574}\u{576}"),
    ('\u{fb14}', "\u{574}\u{565}"),
    ('\u{fb15}', "\u{574}\u{56b}"),
    ('\u{fb16}', "\u{57e}\u{576}"),
    ('\u{fb17}', "\u{574}\u{56d}"),
];
const ND_ZEROS: &[u32] = &[
    0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66,
    0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090, 0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90,
    0x1B50, 0x1BB0, 0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50, 0xABF0, 0xFF10,
    0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136, 0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650,
    0x116C0, 0x11730, 0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60, 0x16AC0,
    0x16B50, 0x1D7CE, 0x1D7D8, 0x1D7E2, 0x1D7EC, 0x1D7F6, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950,
    0x1FBF0,
];

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const ORACLE: &str = include_str!("../tests/fixtures/d1_parse_oracle.json");

    fn same(a: &Option<Number>, b: &Option<Number>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(Number::Int(x)), Some(Number::Int(y))) => x == y,
            (Some(Number::BigInt(x)), Some(Number::BigInt(y))) => x == y,
            (Some(Number::Float(x)), Some(Number::Float(y))) => x.to_bits() == y.to_bits(),
            _ => false,
        }
    }

    fn oracle_number(value: &Value) -> Option<Number> {
        match value {
            Value::Null => None,
            Value::Number(n) if n.is_i64() || n.is_u64() => Some(Number::Int(
                n.as_i64()
                    .map(i128::from)
                    .or_else(|| n.as_u64().map(i128::from))
                    .expect("integer"),
            )),
            Value::Number(n) => Some(Number::Float(n.as_f64().expect("float"))),
            other => panic!("unexpected oracle number {other:?}"),
        }
    }

    fn string(value: &Value) -> Option<String> {
        match value {
            Value::Null => None,
            Value::String(s) => Some(s.clone()),
            other => panic!("unexpected oracle string {other:?}"),
        }
    }

    #[test]
    fn oracle_cases_match_python() {
        let oracle: Value = serde_json::from_str(ORACLE).expect("oracle JSON");
        let numbers = oracle["numbers"].as_array().expect("numbers");
        let dates = oracle["dates"].as_array().expect("dates");
        assert_eq!((numbers.len(), dates.len()), (69, 51), "oracle case count");
        let mut failures = Vec::new();
        let mut checked = 0;
        for case in numbers {
            let text = case["text"].as_str().expect("text");
            let context = case["context"].as_str().expect("context");
            let unit = string(&case["unit"]);
            let expected = oracle_number(&case["number"]);
            let got = parse_number(text, context, unit.as_deref());
            if !same(&got, &expected) {
                failures.push(format!(
                    "parse_number({text:?}, {context:?}, {unit:?}) = {got:?}, Python {expected:?}"
                ));
            }
            let expected_unit = string(&case["written_unit"]);
            let got_unit = written_unit(text);
            if got_unit != expected_unit {
                failures.push(format!(
                    "written_unit({text:?}) = {got_unit:?}, Python {expected_unit:?}"
                ));
            }
            checked += 1;
        }
        for case in dates {
            let text = case["text"].as_str().expect("text");
            let context = case["context"].as_str().expect("context");
            let expected = (
                string(&case["date"]),
                case["ambiguous"].as_bool().expect("ambiguous"),
            );
            let got = parse_date(text, context);
            if got != expected {
                failures.push(format!(
                    "parse_date({text:?}, {context:?}) = {got:?}, Python {expected:?}"
                ));
            }
            checked += 1;
        }
        assert_eq!(checked, 120);
        assert!(
            failures.is_empty(),
            "{} oracle mismatches:\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    // The tests below are not oracle-backed: their expectations come from reading parsing.py,
    // with CPython 3.12 stdlib behaviour (Decimal, datetime.fromisoformat) checked locally.

    fn int(v: i128) -> Option<Number> {
        Some(Number::Int(v))
    }

    fn float(v: f64) -> Option<Number> {
        Some(Number::Float(v))
    }

    fn check_number(text: &str, context: &str, unit: Option<&str>, expected: Option<Number>) {
        let got = parse_number(text, context, unit);
        assert!(
            same(&got, &expected),
            "parse_number({text:?}, {context:?}, {unit:?}) = {got:?}, expected {expected:?}"
        );
    }

    #[test]
    fn empty_and_blank_inputs() {
        for text in ["", "   ", "\u{1c}\u{1f}", "%", "()", "(", "$", "-", "+-1"] {
            check_number(text, "", None, None);
        }
        assert_eq!(parse_date("", ""), (None, false));
        assert_eq!(parse_date(" \t\n", ""), (None, false));
        assert_eq!(written_unit(""), None);
        assert_eq!(written_unit("12 % "), Some("%".to_owned()));
        // str.strip() removes U+001C..U+001F, which Rust's trim keeps.
        check_number("\u{1c}42\u{1f}", "", None, int(42));
        check_number("(-5)", "", None, None);
        check_number("-0", "", None, int(0));
        check_number("-0.5", "", None, float(-0.5));
    }

    #[test]
    fn huge_digit_strings_follow_decimal_context() {
        // Decimal(...) * 1 rounds to 28 significant digits (ROUND_HALF_EVEN).
        check_number(
            &"1".repeat(40),
            "",
            None,
            Some(Number::BigInt(format!(
                "{}{}",
                "1".repeat(28),
                "0".repeat(12)
            ))),
        );
        check_number(
            "12345678901234567890123456789.5",
            "",
            None,
            int(12_345_678_901_234_567_890_123_456_790),
        );
        check_number(
            &format!("1{}.5 trillion", "0".repeat(30)),
            "",
            None,
            Some(Number::BigInt(format!("1{}", "0".repeat(42)))),
        );
        check_number(
            &format!("-{}", "9".repeat(29)),
            "",
            None,
            int(-100_000_000_000_000_000_000_000_000_000),
        );
        // "0.000" is an unsettled group-or-decimal like "1,250"; four places settle it.
        check_number("0.000", "", None, None);
        check_number("0.0000", "", None, int(0));
        // Exact halves round to even; above half rounds up.
        check_number(
            &format!("{}25", "1".repeat(27)),
            "",
            None,
            int(11_111_111_111_111_111_111_111_111_120),
        );
        check_number(
            &format!("{}35", "1".repeat(27)),
            "",
            None,
            int(11_111_111_111_111_111_111_111_111_140),
        );
        check_number(
            &format!("{}251", "1".repeat(27)),
            "",
            None,
            int(111_111_111_111_111_111_111_111_111_300),
        );
        check_number("007", "", None, int(7));
        check_number(
            &format!("{}.25", "3".repeat(300)),
            "",
            None,
            Some(Number::BigInt(format!(
                "{}{}",
                "3".repeat(28),
                "0".repeat(272)
            ))),
        );
        // Million-digit spans go straight to the Decimal step (the span regexes are linear but
        // slow in debug builds). Below Etiny the value rounds to 0E-1000026, which is integral.
        assert_eq!(
            decimal_number(false, &format!("0.{}1", "0".repeat(1_000_030)), 0),
            Ok(Some(Number::Int(0)))
        );
        assert_eq!(
            decimal_number(true, &format!("0.{}5", "0".repeat(1_000_024)), 0),
            Ok(Some(Number::Float(-0.0)))
        );
        // Past Emax CPython raises decimal.Overflow; the port returns None.
        assert_eq!(
            decimal_number(false, &format!("1{}", "0".repeat(999_999)), 0),
            Ok(Some(Number::BigInt(format!("1{}", "0".repeat(999_999)))))
        );
        assert_eq!(
            decimal_number(false, &format!("1{}", "0".repeat(1_000_000)), 0),
            Err(Raised)
        );
    }

    /// Deterministic differential check of the fast `_CURRENCY` scanner against the pattern
    /// compiled as written; not oracle-backed.
    #[test]
    fn currency_scanner_matches_reference_pattern() {
        let alphabet: Vec<char> =
            "UuSsDdIiNnRrKkEeCcHhFfPpLlZzłŁ$.£€¥￥₹元円₽₺ ,0123456789_éıİſ\u{212a}\n-a"
                .chars()
                .collect();
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) as usize
        };
        let mut texts: Vec<String> = vec![
            String::new(),
            "USD".into(),
            "xUSD".into(),
            "USDx".into(),
            "USD1".into(),
            "Rs.5".into(),
            "Rs5".into(),
            "R$ 1".into(),
            "zł".into(),
            "_EUR".into(),
            "éCHF".into(),
            "ınr İNR ſek \u{212a}RW".into(),
            "1 234,50 EUR\n€".into(),
        ];
        for _ in 0..20_000 {
            let len = next() % 12;
            texts.push(
                (0..len)
                    .map(|_| alphabet[next() % alphabet.len()])
                    .collect(),
            );
        }
        let oracle: Value = serde_json::from_str(ORACLE).unwrap();
        for case in oracle["numbers"]
            .as_array()
            .unwrap()
            .iter()
            .chain(oracle["dates"].as_array().unwrap())
        {
            texts.push(case["text"].as_str().unwrap().to_owned());
            texts.push(case["context"].as_str().unwrap().to_owned());
        }
        let mut hits = 0;
        for text in &texts {
            let reference = find_all(&CURRENCY, text).unwrap();
            assert_eq!(currency_find_all(text), reference, "{text:?}");
            assert_eq!(
                currency_search(text),
                CURRENCY.is_match(text).unwrap(),
                "{text:?}"
            );
            hits += reference.len();
        }
        assert!(
            hits > 1_000,
            "the generated texts must exercise matches ({hits})"
        );
    }

    #[test]
    fn non_ascii_digits() {
        check_number("٣٤٥", "", None, int(345));
        check_number("１２３", "", None, int(123));
        check_number("٣,٥", "", None, float(3.5));
        check_number("²", "", None, None);
        assert_eq!(
            parse_date("١٣/٠٤/٢٠٢٦", ""),
            (Some("2026-04-13".to_owned()), false)
        );
        assert_eq!(parse_date("٠٣/٠٤/٢٠٢٦", ""), (None, true));
        assert_eq!(
            parse_date("１２ février ２０２６", ""),
            (Some("2026-02-12".to_owned()), false)
        );
        // fromisoformat only reads ASCII digits.
        assert_eq!(parse_date("2026-04-03T١٠:٣٠", ""), (None, false));
    }

    #[test]
    fn invalid_and_boundary_dates() {
        let none = (None, false);
        assert_eq!(parse_date("30/02/2026", ""), none);
        assert_eq!(parse_date("Feb 30, 2026", ""), none);
        assert_eq!(parse_date("2026年2月30日", ""), none);
        assert_eq!(parse_date("0000-01-01", ""), none);
        assert_eq!(parse_date("29/02/2023", ""), none);
        assert_eq!(
            parse_date("29/02/2024", ""),
            (Some("2024-02-29".to_owned()), false)
        );
        assert_eq!(parse_date("03/04/026", ""), none);
        assert_eq!(parse_date("2026-04-03T24:00", ""), none);
        assert_eq!(parse_date("2026-04-03 10:30 11:00", ""), none);
        assert_eq!(
            parse_date("2026-04-03 10:30+05:99", ""),
            (Some("2026-04-03T10:30:00+06:39".to_owned()), false)
        );
        assert_eq!(parse_date("2026-04-03 10:30+24:00", ""), none);
        assert_eq!(
            parse_date("2026-04-03 10:30-00:00", ""),
            (Some("2026-04-03T10:30:00+00:00".to_owned()), false)
        );
        assert_eq!(
            parse_date("2026-04-03 10:30:00.1234567", ""),
            (Some("2026-04-03T10:30:00.123456".to_owned()), false)
        );
        assert_eq!(
            parse_date("2026-04-03 7:05:09.5-05:30", ""),
            (Some("2026-04-03T07:05:09.500000-05:30".to_owned()), false)
        );
        assert_eq!(
            parse_date("le 1er mai 2026", ""),
            (Some("2026-05-01".to_owned()), false)
        );
        assert_eq!(
            parse_date("0003 april 2026", ""),
            (Some("2026-04-03".to_owned()), false)
        );
    }

    #[test]
    fn case_insensitive_markers_follow_cpython() {
        // `ſ` casefolds to `s`, so `ſek` is SEK and its comma is the decimal mark.
        assert_eq!(written_unit("1,5 ſek"), Some("SEK".to_owned()));
        check_number("1,5 ſek", "", None, float(1.5));
        // `ınr` matches under re.I but its casefold is no table key: Python raises KeyError.
        assert_eq!(written_unit("100 ınr"), None);
        check_number("100 ınr", "", None, int(100));
        check_number("1,5 ınr", "", None, None);
        check_number("10 BILLION", "", None, int(10_000_000_000));
        check_number("₹1,00,00,000.50", "", None, float(10_000_000.5));
        check_number("1,234", "LANG=EN", None, int(1234));
        // The span's own currency speaks before the document's numbers.
        check_number("1,234 kr", "Total 1.50", None, float(1.234));
        check_number("1,234", "Total 1.50", None, int(1234));
    }

    fn hash_update(h: u128, s: &str) -> u128 {
        const M: u128 = (1 << 61) - 1;
        let h = s.chars().fold(h, |h, c| (h * 131 + u128::from(c) + 1) % M);
        (h * 131 + 0x11_0001) % M
    }

    /// Checksums from CPython 3.12.14 (unicodedata 15.0.0) over every assigned non-surrogate code
    /// point; not d1-oracle-backed.
    #[test]
    fn unicode_tables_match_cpython_3_12() {
        let assigned = Regex::new(r"^[\P{Cn}&&\p{Age:V15_0}]$").unwrap();
        let word = Regex::new(&format!("^{W}$")).unwrap();
        let digit = Regex::new(&format!("^{D}$")).unwrap();
        let space = Regex::new(&format!("^{S}$")).unwrap();
        let (mut domain, mut words, mut word_sum, mut digits, mut digit_sum) =
            (0u64, 0u64, 0u128, 0u64, 0u128);
        let (mut lower, mut casefold, mut folded) = (0u128, 0u128, 0u128);
        let mut buf = [0u8; 4];
        for c in (0..=0x10_FFFFu32).filter_map(char::from_u32) {
            let s = &*c.encode_utf8(&mut buf);
            assert_eq!(space.is_match(s).unwrap(), py_isspace(c), "{c:?}");
            assert_eq!(digit.is_match(s).unwrap(), nd_value(c).is_some(), "{c:?}");
            if !assigned.is_match(s).unwrap() {
                continue;
            }
            domain += 1;
            if word.is_match(s).unwrap() {
                words += 1;
                word_sum += u128::from(c);
            }
            if nd_value(c).is_some() {
                digits += 1;
                digit_sum += u128::from(c);
            }
            lower = hash_update(lower, &c.to_lowercase().collect::<String>());
            casefold = hash_update(casefold, &py_casefold(s));
            folded = hash_update(folded, &fold(s));
        }
        assert_eq!(domain, 286_719);
        assert_eq!((words, word_sum), (137_936, 14_877_327_126));
        assert_eq!((digits, digit_sum), (680, 32_783_620));
        assert_eq!(lower, 2_278_034_122_857_789_683);
        assert_eq!(casefold, 1_502_656_408_192_499_160);
        assert_eq!(folded, 1_828_125_694_926_860_095);
    }
}
