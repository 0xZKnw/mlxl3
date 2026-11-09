//! Fail-closed JSON output validation. Unsupported schema assertions are errors.
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    sync::atomic::{AtomicBool, Ordering},
};

const MAX_SCHEMA: usize = 64 * 1024;
const MAX_OUTPUT: usize = 1024 * 1024;
const VALIDATION_BYTE_CHUNK: usize = 256;

pub fn within_bounds(value: usize, minimum: Option<usize>, maximum: Option<usize>) -> bool {
    minimum.is_none_or(|min| value >= min) && maximum.is_none_or(|max| value <= max)
}

pub fn tool_allowed(name: &str, allowed: Option<&[String]>) -> bool {
    allowed.is_none_or(|names| names.iter().any(|item| item == name))
}

pub fn remaining_tokens(limit: i64, generated: usize) -> Option<i64> {
    if limit == -1 {
        return Some(-1);
    }
    limit
        .checked_sub(i64::try_from(generated).ok()?)
        .filter(|remaining| limit > 0 && *remaining > 0)
}

pub struct Schema(Value);

impl Schema {
    pub fn parse(text: &str) -> Result<Self> {
        ensure!(text.len() <= MAX_SCHEMA, "JSON schema exceeds 64 KiB");
        let value: Value = serde_json::from_str(text).context("invalid JSON schema")?;
        check_schema(&value, 0)?;
        Ok(Self(value))
    }

    pub fn instruction(&self) -> String {
        format!(
            "Return only one JSON value conforming to this schema. No Markdown fences or commentary. Schema: {}",
            self.0
        )
    }

    pub fn validate(&self, text: &str) -> Result<String> {
        self.validate_cancellable(text, &AtomicBool::new(false))
    }

    pub fn validate_cancellable(&self, text: &str, cancelled: &AtomicBool) -> Result<String> {
        ensure!(
            !text.trim().is_empty() && text.len() <= MAX_OUTPUT,
            "JSON output is empty or exceeds 1 MiB"
        );
        let value: Value =
            serde_json::from_str(text).context("model did not return complete, valid JSON")?;
        let mut budget = ValidationBudget {
            remaining: 1_000_000,
            exhausted: false,
            cancelled,
        };
        let valid = matches_schema_bounded(&self.0, &value, 0, &mut budget);
        ensure!(
            !budget.exhausted,
            "JSON validation work budget exceeded or cancelled"
        );
        ensure!(valid, "model JSON does not conform to the requested schema");
        Ok(serde_json::to_string(&value)?)
    }
}

fn check_schema(schema: &Value, depth: usize) -> Result<()> {
    ensure!(depth <= 32, "JSON schema nesting exceeds 32");
    if schema.is_boolean() {
        return Ok(());
    }
    let object = schema
        .as_object()
        .context("schema must be an object or boolean")?;
    for (key, value) in object {
        match key.as_str() {
            "$schema" | "title" | "description" => {
                ensure!(value.is_string(), "{key} must be a string");
            }
            "default" | "examples" => {}
            "type" => {
                let valid = |kind: &Value| {
                    matches!(
                        kind.as_str(),
                        Some(
                            "object"
                                | "array"
                                | "string"
                                | "integer"
                                | "number"
                                | "boolean"
                                | "null"
                        )
                    )
                };
                ensure!(
                    valid(value)
                        || value
                            .as_array()
                            .is_some_and(|types| !types.is_empty() && types.iter().all(valid)),
                    "invalid schema type"
                );
            }
            "properties" => {
                let properties = value.as_object().context("properties must be an object")?;
                ensure!(properties.len() <= 256, "too many schema properties");
                for child in properties.values() {
                    check_schema(child, depth + 1)?;
                }
            }
            "required" => {
                ensure!(
                    value.as_array().is_some_and(
                        |items| items.len() <= 256 && items.iter().all(Value::is_string)
                    ),
                    "required must be an array of property names"
                );
            }
            "items" | "additionalProperties" => {
                check_schema(value, depth + 1)?;
            }
            "minLength" | "maxLength" | "minItems" | "maxItems" | "minProperties"
            | "maxProperties" => {
                ensure!(
                    value
                        .as_u64()
                        .and_then(|n| usize::try_from(n).ok())
                        .is_some(),
                    "{key} must be a non-negative integer"
                );
            }
            "minimum" | "maximum" | "exclusiveMinimum" | "exclusiveMaximum" => {
                ensure!(
                    value
                        .as_f64()
                        .is_some_and(|n| n.is_finite() && n.abs() <= 9_007_199_254_740_991.0),
                    "{key} requires a finite number within exact integer range"
                );
            }
            "enum" => {
                ensure!(
                    value.as_array().is_some_and(|items| !items.is_empty()),
                    "enum must be a nonempty array"
                );
            }
            "const" => {}
            "allOf" | "anyOf" | "oneOf" => {
                let alternatives = value
                    .as_array()
                    .context("schema alternatives must be an array")?;
                ensure!(
                    !alternatives.is_empty() && alternatives.len() <= 32,
                    "schema needs 1..32 alternatives"
                );
                for child in alternatives {
                    check_schema(child, depth + 1)?;
                }
            }
            "not" => {
                check_schema(value, depth + 1)?;
            }
            "uniqueItems" => {
                ensure!(value.is_boolean(), "uniqueItems must be a boolean");
            }
            _ => bail!("unsupported JSON schema keyword: {key}"),
        }
    }
    Ok(())
}

fn length_matches(schema: &Value, count: usize, min: &str, max: &str) -> bool {
    within_bounds(
        count,
        schema[min].as_u64().map(|n| n as usize),
        schema[max].as_u64().map(|n| n as usize),
    )
}

fn json_equal(left: &Value, right: &Value, budget: &mut ValidationBudget<'_>) -> bool {
    if !budget.spend() {
        return false;
    }
    if let (Some(a), Some(b)) = (left.as_number(), right.as_number()) {
        if let (Some(a), Some(b)) = (a.as_i64(), b.as_i64()) {
            return a == b;
        }
        if let (Some(a), Some(b)) = (a.as_u64(), b.as_u64()) {
            return a == b;
        }
        if let Some(integer) = a.as_i64() {
            return b.as_f64().is_some_and(|number| {
                number.fract() == 0.0 && number as i128 == i128::from(integer)
            });
        }
        if let Some(integer) = a.as_u64() {
            return b.as_f64().is_some_and(|number| {
                number >= 0.0 && number.fract() == 0.0 && number as u128 == u128::from(integer)
            });
        }
        if b.is_i64() || b.is_u64() {
            return json_equal(right, left, budget);
        }
        return a.as_f64() == b.as_f64();
    }
    match (left, right) {
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| json_equal(a, b, budget))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                // serde_json::Map is a sorted BTreeMap (no preserve_order feature).
                // Zip avoids unaccounted variable-length key lookups.
                && a.iter().zip(b).all(|((ka, a), (kb, b))| {
                    bytes_equal(ka.as_bytes(), kb.as_bytes(), budget) && json_equal(a, b, budget)
                })
        }
        (Value::String(a), Value::String(b)) => bytes_equal(a.as_bytes(), b.as_bytes(), budget),
        (Value::Null, Value::Null) => true,
        (Value::Bool(a), Value::Bool(b)) => a == b,
        _ => false,
    }
}

fn bytes_equal(left: &[u8], right: &[u8], budget: &mut ValidationBudget<'_>) -> bool {
    left.len() == right.len()
        && left
            .chunks(VALIDATION_BYTE_CHUNK)
            .zip(right.chunks(VALIDATION_BYTE_CHUNK))
            .all(|(a, b)| budget.spend() && a == b)
}

struct ValidationBudget<'a> {
    remaining: usize,
    exhausted: bool,
    cancelled: &'a AtomicBool,
}

impl ValidationBudget<'_> {
    fn spend(&mut self) -> bool {
        if self.exhausted || self.remaining == 0 || self.cancelled.load(Ordering::Relaxed) {
            self.exhausted = true;
            return false;
        }
        self.remaining -= 1;
        true
    }
}

// Canonical numeric identity preserves exact integers and normalizes integral
// floats only within the exact signed/unsigned conversion range. Hashes are an
// index, never an equality oracle: every collision is checked with json_equal.
fn canonical_number(number: &serde_json::Number) -> serde_json::Number {
    if number.is_f64() {
        let n = number.as_f64().expect("JSON numbers are finite");
        if n.fract() == 0.0
            && (-9_223_372_036_854_775_808.0..18_446_744_073_709_551_616.0).contains(&n)
        {
            return if n < 0.0 {
                (n as i64).into()
            } else {
                (n as u64).into()
            };
        }
    }
    number.clone()
}

fn hash_bytes(bytes: &[u8], hash: &mut Sha256, budget: &mut ValidationBudget<'_>) -> Option<()> {
    hash.update(bytes.len().to_le_bytes());
    for chunk in bytes.chunks(VALIDATION_BYTE_CHUNK) {
        if !budget.spend() {
            return None;
        }
        hash.update(chunk);
    }
    Some(())
}

fn hash_value(value: &Value, hash: &mut Sha256, budget: &mut ValidationBudget<'_>) -> Option<()> {
    if !budget.spend() {
        return None;
    }
    match value {
        Value::Null => hash.update([0]),
        Value::Bool(flag) => hash.update([1, u8::from(*flag)]),
        Value::Number(number) => {
            hash.update([2]);
            hash_bytes(
                canonical_number(number).to_string().as_bytes(),
                hash,
                budget,
            )?;
        }
        Value::String(text) => {
            hash.update([3]);
            hash_bytes(text.as_bytes(), hash, budget)?;
        }
        Value::Array(items) => {
            hash.update([4]);
            hash.update(items.len().to_le_bytes());
            for item in items {
                hash_value(item, hash, budget)?;
            }
        }
        Value::Object(items) => {
            hash.update([5]);
            hash.update(items.len().to_le_bytes());
            for (key, item) in items {
                hash_bytes(key.as_bytes(), hash, budget)?;
                hash_value(item, hash, budget)?;
            }
        }
    }
    Some(())
}

fn insert_unique<'a>(
    seen: &mut BTreeMap<[u8; 32], Vec<&'a Value>>,
    key: [u8; 32],
    value: &'a Value,
    budget: &mut ValidationBudget<'_>,
) -> bool {
    let bucket = seen.entry(key).or_default();
    for previous in bucket.iter() {
        if json_equal(previous, value, budget) || budget.exhausted {
            return false;
        }
    }
    bucket.push(value);
    true
}

#[cfg(kani)]
fn matches_schema(schema: &Value, value: &Value, depth: usize) -> bool {
    let cancelled = AtomicBool::new(false);
    let mut budget = ValidationBudget {
        remaining: 1_000_000,
        exhausted: false,
        cancelled: &cancelled,
    };
    matches_schema_bounded(schema, value, depth, &mut budget) && !budget.exhausted
}

fn matches_schema_bounded(
    schema: &Value,
    value: &Value,
    depth: usize,
    budget: &mut ValidationBudget<'_>,
) -> bool {
    if !budget.spend() || depth > 32 {
        return false;
    }
    if let Some(flag) = schema.as_bool() {
        return flag;
    }
    let Some(object) = schema.as_object() else {
        return false;
    };
    if let Some(types) = object.get("type") {
        let matches = |kind: &Value| match kind.as_str() {
            Some("object") => value.is_object(),
            Some("array") => value.is_array(),
            Some("string") => value.is_string(),
            Some("boolean") => value.is_boolean(),
            Some("null") => value.is_null(),
            Some("number") => value.is_number(),
            Some("integer") => {
                value.is_i64() || value.is_u64() || value.as_f64().is_some_and(|n| n.fract() == 0.0)
            }
            _ => false,
        };
        if !types
            .as_array()
            .map_or_else(|| matches(types), |items| items.iter().any(matches))
        {
            return false;
        }
    }
    if object
        .get("const")
        .is_some_and(|expected| !json_equal(expected, value, budget))
    {
        return false;
    }
    if let Some(items) = object.get("enum") {
        let Some(items) = items.as_array() else {
            return false;
        };
        let mut matched = false;
        for expected in items {
            if json_equal(expected, value, budget) {
                matched = true;
                break;
            }
            if budget.exhausted {
                return false;
            }
        }
        if !matched {
            return false;
        }
    }
    for (keyword, mode) in [("allOf", 0), ("anyOf", 1), ("oneOf", 2)] {
        if let Some(alternatives) = schema[keyword].as_array() {
            let count = alternatives
                .iter()
                .filter(|child| matches_schema_bounded(child, value, depth + 1, budget))
                .count();
            if (mode == 0 && count != alternatives.len())
                || (mode == 1 && count == 0)
                || (mode == 2 && count != 1)
            {
                return false;
            }
        }
    }
    if object
        .get("not")
        .is_some_and(|child| matches_schema_bounded(child, value, depth + 1, budget))
    {
        return false;
    }
    if let Some(number) = value.as_f64() {
        let has_bounds = ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"]
            .iter()
            .any(|key| object.contains_key(*key));
        if has_bounds && number.abs() > 9_007_199_254_740_991.0 {
            return false;
        }
        if schema["minimum"].as_f64().is_some_and(|min| number < min)
            || schema["maximum"].as_f64().is_some_and(|max| number > max)
            || schema["exclusiveMinimum"]
                .as_f64()
                .is_some_and(|min| number <= min)
            || schema["exclusiveMaximum"]
                .as_f64()
                .is_some_and(|max| number >= max)
        {
            return false;
        }
    }
    if let Some(text) = value.as_str() {
        return length_matches(schema, text.chars().count(), "minLength", "maxLength");
    }
    if let Some(items) = value.as_array() {
        if !length_matches(schema, items.len(), "minItems", "maxItems") {
            return false;
        }
        if schema["uniqueItems"] == true {
            let mut seen = BTreeMap::new();
            for item in items {
                let mut hash = Sha256::new();
                if hash_value(item, &mut hash, budget).is_none()
                    || !insert_unique(&mut seen, hash.finalize().into(), item, budget)
                {
                    return false;
                }
            }
        }
        if let Some(child) = object.get("items")
            && !items
                .iter()
                .all(|item| matches_schema_bounded(child, item, depth + 1, budget))
        {
            return false;
        }
    }
    if let Some(properties) = value.as_object() {
        if !length_matches(schema, properties.len(), "minProperties", "maxProperties") {
            return false;
        }
        if schema["required"].as_array().is_some_and(|names| {
            names
                .iter()
                .any(|name| !properties.contains_key(name.as_str().unwrap_or("")))
        }) {
            return false;
        }
        for (name, item) in properties {
            if let Some(child) = schema["properties"].get(name) {
                if !matches_schema_bounded(child, item, depth + 1, budget) {
                    return false;
                }
            } else if let Some(child) = object.get("additionalProperties")
                && !matches_schema_bounded(child, item, depth + 1, budget)
            {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn enum_comparison_work_is_bounded_in_production() {
        let candidates: Vec<_> = (0..256).map(|i| format!("candidate-{i:03}")).collect();
        let constraint = serde_json::json!({"items": {"enum": candidates}});
        let output = serde_json::to_string(&vec!["candidate-255"; 5_000]).unwrap();
        for constraint in [constraint.clone(), serde_json::json!({"not": constraint})] {
            let schema = Schema::parse(&constraint.to_string()).unwrap();
            let Err(error) = schema.validate(&output) else {
                panic!("comparison work must be bounded");
            };
            assert!(error.to_string().contains("work budget"), "{error}");
        }
    }

    #[test]
    fn comparisons_and_hashing_charge_bounded_byte_chunks() {
        let cancelled = AtomicBool::new(false);
        for size in [0usize, 1, 255, 256, 257, 513] {
            let value = Value::String("x".repeat(size));
            let cost = 1 + size.div_ceil(VALIDATION_BYTE_CHUNK);
            for available in [cost - 1, cost] {
                let mut budget = ValidationBudget {
                    remaining: available,
                    exhausted: false,
                    cancelled: &cancelled,
                };
                assert_eq!(json_equal(&value, &value, &mut budget), available == cost);
                assert_eq!(budget.exhausted, available != cost);
                assert_eq!(budget.remaining, 0);
                let mut budget = ValidationBudget {
                    remaining: available,
                    exhausted: false,
                    cancelled: &cancelled,
                };
                assert_eq!(
                    hash_value(&value, &mut Sha256::new(), &mut budget).is_some(),
                    available == cost
                );
                assert_eq!(budget.exhausted, available != cost);
                assert_eq!(budget.remaining, 0);
            }
        }
        let value = serde_json::json!({"x".repeat(257): [null, true]});
        // Object, two key chunks, array, null and bool: six units.
        for available in [5, 6] {
            let mut budget = ValidationBudget {
                remaining: available,
                exhausted: false,
                cancelled: &cancelled,
            };
            assert_eq!(json_equal(&value, &value, &mut budget), available == 6);
            assert_eq!(budget.exhausted, available != 6);
            let mut budget = ValidationBudget {
                remaining: available,
                exhausted: false,
                cancelled: &cancelled,
            };
            assert_eq!(
                hash_value(&value, &mut Sha256::new(), &mut budget).is_some(),
                available == 6
            );
            assert_eq!(budget.exhausted, available != 6);
        }
    }

    #[test]
    fn equality_cancellation_is_sticky_and_const_spends_children() {
        let cancelled = AtomicBool::new(false);
        let mut budget = ValidationBudget {
            remaining: 10,
            exhausted: false,
            cancelled: &cancelled,
        };
        assert!(json_equal(&Value::Null, &Value::Null, &mut budget));
        cancelled.store(true, Ordering::Relaxed);
        assert!(!json_equal(&Value::Null, &Value::Null, &mut budget));
        assert_eq!(budget.remaining, 9);
        cancelled.store(false, Ordering::Relaxed);
        assert!(!budget.spend());
        assert_eq!(budget.remaining, 9);
        for schema in [
            serde_json::json!({"const": [null, true]}),
            serde_json::json!({"not": {"const": [null, true]}}),
        ] {
            let mut budget = ValidationBudget {
                remaining: 3,
                exhausted: false,
                cancelled: &cancelled,
            };
            let _ =
                matches_schema_bounded(&schema, &serde_json::json!([null, true]), 0, &mut budget);
            assert!(budget.exhausted, "{schema}");
        }
        let mut budget = ValidationBudget {
            remaining: 10,
            exhausted: false,
            cancelled: &cancelled,
        };
        cancelled.store(true, Ordering::Relaxed);
        assert!(hash_value(&Value::Null, &mut Sha256::new(), &mut budget).is_none());
        assert!(budget.exhausted);
    }

    #[test]
    fn digest_collisions_require_exact_equality_and_share_the_budget() {
        let cancelled = AtomicBool::new(false);
        let mut budget = ValidationBudget {
            remaining: 100,
            exhausted: false,
            cancelled: &cancelled,
        };
        let first = serde_json::json!({"n": 1});
        let other = serde_json::json!({"n": 2});
        let equivalent = serde_json::json!({"n": 1.0});
        let mut seen = BTreeMap::new();
        let forced_digest = [0; 32];
        assert!(insert_unique(&mut seen, forced_digest, &first, &mut budget));
        assert!(insert_unique(&mut seen, forced_digest, &other, &mut budget));
        assert!(!insert_unique(
            &mut seen,
            forced_digest,
            &equivalent,
            &mut budget
        ));
        assert_eq!(seen[&forced_digest].len(), 2);
        budget.remaining = 0;
        assert!(!insert_unique(
            &mut seen,
            forced_digest,
            &other,
            &mut budget
        ));
        assert!(budget.exhausted);
    }

    #[test]
    fn bounded_numeric_equality_and_hash_identity_agree() {
        let cases = [
            ("1", "1.0", true),
            ("0", "-0.0", true),
            ("-1", "-1.0", true),
            ("1.5", "1.5", true),
            ("9007199254740992", "9007199254740992.0", true),
            ("9007199254740993", "9007199254740992.0", false),
            ("9223372036854775807", "9223372036854775808.0", false),
            ("9223372036854775808", "9223372036854775808.0", true),
            ("-9223372036854775808", "-9223372036854775808.0", true),
            ("18446744073709551615", "18446744073709551616.0", false),
            ("18446744073709551615", "-1", false),
            ("18446744073709551616.0", "18446744073709551616.0", true),
            (
                r#"{"a":[1,0],"b":true}"#,
                r#"{"b":true,"a":[1.0,-0.0]}"#,
                true,
            ),
            (r#"{"a":[1,2]}"#, r#"{"a":[1,3]}"#, false),
            (r#""1""#, "1", false),
        ];
        for (left, right, equal) in cases {
            for (left, right) in [(left, right), (right, left)] {
                let constant = Schema::parse(&format!(r#"{{"const":{left}}}"#)).unwrap();
                assert_eq!(constant.validate(right).is_ok(), equal, "{left}, {right}");
                let enumeration = Schema::parse(&format!(r#"{{"enum":[{left}]}}"#)).unwrap();
                assert_eq!(
                    enumeration.validate(right).is_ok(),
                    equal,
                    "{left}, {right}"
                );
                let unique = Schema::parse(r#"{"uniqueItems":true}"#).unwrap();
                assert_eq!(
                    unique.validate(&format!("[{left},{right}]")).is_ok(),
                    !equal,
                    "{left}, {right}"
                );
            }
        }
    }

    #[test]
    fn large_unique_arrays_and_exact_numeric_keys() {
        let schema = Schema::parse(r#"{"uniqueItems":true}"#).unwrap();
        let values: Vec<_> = (0..100_000).collect();
        assert!(
            schema
                .validate(&serde_json::to_string(&values).unwrap())
                .is_ok()
        );
        for duplicate in [
            "[1,1.0]",
            "[0,-0.0]",
            "[{\"a\":[1]}, {\"a\":[1.0]}]",
            "[9007199254740992,9007199254740992.0]",
        ] {
            assert!(schema.validate(duplicate).is_err(), "{duplicate}");
        }
        for distinct in [
            "[9007199254740992,9007199254740993]",
            "[9007199254740993,9007199254740992.0]",
            "[18446744073709551615,18446744073709551616.0]",
        ] {
            assert!(schema.validate(distinct).is_ok(), "{distinct}");
        }
        assert!(
            schema
                .validate_cancellable("[1,2]", &AtomicBool::new(true))
                .is_err()
        );
        // Exhaustion under `not` must not turn an interrupted false into success.
        let nested = Schema::parse(r#"{"not":{"uniqueItems":true}}"#).unwrap();
        assert!(
            nested
                .validate_cancellable("[1,2]", &AtomicBool::new(true))
                .is_err()
        );
    }

    #[test]
    fn objects_fail_closed_and_preserve_unicode() {
        let schema = Schema::parse(r#"{"type":"object","properties":{"answer":{"type":"string","minLength":1,"maxLength":2},"count":{"type":"integer","minimum":0,"maximum":2}},"required":["answer","count"],"additionalProperties":false}"#).unwrap();
        assert_eq!(
            schema.validate(r#" {"answer":"é🦀","count":2} "#).unwrap(),
            r#"{"answer":"é🦀","count":2}"#
        );
        for invalid in [
            "",
            "{}",
            r#"{"answer":"","count":1}"#,
            r#"{"answer":"abc","count":1}"#,
            r#"{"answer":"a","count":-1}"#,
            r#"{"answer":"a","count":1.5}"#,
            r#"{"answer":"a","count":2,"extra":0}"#,
            "```json\n{}\n```",
            "{}{}",
            r#"{"answer":"a","count":1"#,
        ] {
            assert!(schema.validate(invalid).is_err(), "{invalid}");
        }
    }
    #[test]
    fn arrays_alternatives_and_nullable_values() {
        let schema = Schema::parse(r#"{"type":"array","minItems":1,"maxItems":3,"uniqueItems":true,"items":{"anyOf":[{"type":"null"},{"type":"integer","exclusiveMinimum":0,"exclusiveMaximum":3}]}}"#).unwrap();
        for valid in ["[null]", "[1,2,null]"] {
            assert!(schema.validate(valid).is_ok());
        }
        for invalid in ["[]", "[1,1]", "[0]", "[3]", "[1,2,null,3]", "[false]"] {
            assert!(schema.validate(invalid).is_err());
        }
        let schema = Schema::parse(r#"{"oneOf":[{"type":"number"},{"type":"integer"}]}"#).unwrap();
        assert!(schema.validate("1").is_err());
        assert!(schema.validate("1.5").is_ok());
        let schema =
            Schema::parse(r#"{"allOf":[{"type":["string","null"]},{"not":{"const":"bad"}}]}"#)
                .unwrap();
        assert!(schema.validate("null").is_ok());
        assert!(schema.validate(r#""bad""#).is_err());
    }
    #[test]
    fn numeric_equivalence_is_enforced_recursively() {
        let schema = Schema::parse(r#"{"type":"array","uniqueItems":true}"#).unwrap();
        assert!(schema.validate("[1,1.0]").is_err());
        assert!(schema.validate(r#"[{"n":1},{"n":1.0}]"#).is_err());
        assert!(
            schema
                .validate("[9007199254740993,9007199254740992.0]")
                .is_ok()
        );
        let schema = Schema::parse(r#"{"enum":[{"n":1}]}"#).unwrap();
        assert!(schema.validate(r#"{"n":1.0}"#).is_ok());
    }

    #[test]
    fn unsupported_or_malformed_schemas_fail_before_inference() {
        for invalid in [
            r##"{"$ref":"#"}"##,
            r#"{"type":"invalid"}"#,
            r#"{"type":[]}"#,
            r#"{"pattern":".*"}"#,
            r#"{"required":true}"#,
            r#"{"minimum":"0"}"#,
            r#"{"items":[]}"#,
            r#"{"anyOf":[]}"#,
            r#"{"maxLength":-1}"#,
            r#"{"properties":{"x":{"format":"email"}}}"#,
            r#"{"minimum":9007199254740992}"#,
        ] {
            assert!(Schema::parse(invalid).is_err(), "{invalid}");
        }
        assert!(Schema::parse(&" ".repeat(MAX_SCHEMA + 1)).is_err());
        let schema = Schema::parse("{}").unwrap();
        assert!(schema.validate(&" ".repeat(MAX_OUTPUT + 1)).is_err());
    }
    #[test]
    fn project_permissions_are_exact_and_default_deny() {
        assert!(tool_allowed("server.read", None));
        assert!(!tool_allowed("server.read", Some(&[])));
        let allowed = vec!["server.read".into()];
        assert!(tool_allowed("server.read", Some(&allowed)));
        for name in ["server.write", "server.read.more", "SERVER.read", ""] {
            assert!(!tool_allowed(name, Some(&allowed)));
        }
    }
    #[test]
    fn structured_retries_share_one_token_budget() {
        for (limit, generated, expected) in [
            (-1, usize::MAX, Some(-1)),
            (0, 0, None),
            (-2, 0, None),
            (5, 0, Some(5)),
            (5, 4, Some(1)),
            (5, 5, None),
            (5, 6, None),
            (1, usize::MAX, None),
            (i64::MIN, 1, None),
        ] {
            assert_eq!(remaining_tokens(limit, generated), expected);
        }
    }
    #[test]
    fn length_boundaries_exhaustive_small_domain() {
        for size in 0..=32 {
            for min in 0..=32 {
                for max in 0..=32 {
                    assert_eq!(
                        within_bounds(size, Some(min), Some(max)),
                        (min..=max).contains(&size)
                    );
                }
            }
        }
    }
}

#[cfg(kani)]
mod proofs {
    #[kani::proof]
    fn validation_budget_never_wraps_or_accepts_cancellation() {
        let remaining: usize = kani::any();
        let cancelled_value: bool = kani::any();
        let exhausted: bool = kani::any();
        let cancelled = AtomicBool::new(cancelled_value);
        let mut budget = ValidationBudget {
            remaining,
            exhausted,
            cancelled: &cancelled,
        };
        let accepted = budget.spend();
        assert_eq!(accepted, remaining > 0 && !cancelled_value && !exhausted);
        assert_eq!(budget.exhausted, !accepted);
        assert_eq!(
            budget.remaining,
            if accepted { remaining - 1 } else { remaining }
        );
        kani::cover!(accepted);
        kani::cover!(!accepted);
    }

    #[kani::proof]
    #[kani::unwind(260)]
    fn byte_comparison_spends_each_chunk() {
        let remaining: u8 = kani::any();
        let cancelled_value: bool = kani::any();
        let different: bool = kani::any();
        let cancelled = AtomicBool::new(cancelled_value);
        let mut budget = ValidationBudget {
            remaining: usize::from(remaining),
            exhausted: false,
            cancelled: &cancelled,
        };
        let left = [0; VALIDATION_BYTE_CHUNK + 1];
        let mut right = left;
        right[VALIDATION_BYTE_CHUNK] = u8::from(different);
        let equal = bytes_equal(&left, &right, &mut budget);
        assert_eq!(equal, remaining >= 2 && !cancelled_value && !different);
        assert_eq!(budget.exhausted, remaining < 2 || cancelled_value);
        assert_eq!(
            budget.remaining,
            if cancelled_value {
                usize::from(remaining)
            } else {
                usize::from(remaining.saturating_sub(2))
            }
        );
        kani::cover!(equal);
        kani::cover!(!equal && !budget.exhausted);
        kani::cover!(budget.exhausted);
    }

    use super::*;
    #[kani::proof]
    fn structured_retry_budget_never_restarts_or_wraps() {
        let limit: i64 = kani::any();
        let generated: usize = kani::any();
        let expected = if limit == -1 {
            Some(-1)
        } else if limit > 0 && (generated as u128) < (limit as u128) {
            Some(limit - generated as i64)
        } else {
            None
        };
        assert_eq!(remaining_tokens(limit, generated), expected);
        kani::cover!(limit > 0 && generated > 0 && expected.is_some());
        kani::cover!(limit > 0 && generated as u128 >= limit as u128);
        kani::cover!(limit == -1 && generated == usize::MAX);
    }
    #[kani::proof]
    fn boolean_schema_contract() {
        let allow: bool = kani::any();
        assert_eq!(matches_schema(&Value::Bool(allow), &Value::Null, 0), allow);
        kani::cover!(allow);
        kani::cover!(!allow);
    }

    #[kani::proof]
    fn structured_length_bounds() {
        let value: usize = kani::any();
        let min: usize = kani::any();
        let max: usize = kani::any();
        assert_eq!(
            within_bounds(value, Some(min), Some(max)),
            value >= min && value <= max
        );
        assert!(within_bounds(value, None, None));
        assert_eq!(within_bounds(value, Some(min), None), value >= min);
        assert_eq!(within_bounds(value, None, Some(max)), value <= max);
        kani::cover!(value == min && min == max);
        kani::cover!(value < min);
        kani::cover!(value > max);
    }
    #[kani::proof]
    #[kani::unwind(16)]
    fn project_permission_boundary() {
        let a: bool = kani::any();
        let b: bool = kani::any();
        let allowed = vec![if a { "read".into() } else { "write".into() }];
        let name = if b { "read" } else { "write" };
        assert_eq!(tool_allowed(name, Some(&allowed)), a == b);
        assert!(!tool_allowed(name, Some(&[])));
        kani::cover!(a == b);
        kani::cover!(a != b);
    }
}
