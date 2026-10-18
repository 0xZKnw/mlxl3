//! Fail-closed JSON output validation. Unsupported schema assertions are errors.
use anyhow::{Context, Result, bail, ensure};
use serde_json::Value;

const MAX_SCHEMA: usize = 64 * 1024;
const MAX_OUTPUT: usize = 1024 * 1024;

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
        ensure!(
            !text.trim().is_empty() && text.len() <= MAX_OUTPUT,
            "JSON output is empty or exceeds 1 MiB"
        );
        let value: Value =
            serde_json::from_str(text).context("model did not return complete, valid JSON")?;
        ensure!(
            matches_schema(&self.0, &value, 0),
            "model JSON does not conform to the requested schema"
        );
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

fn json_equal(left: &Value, right: &Value) -> bool {
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
            return json_equal(right, left);
        }
        return a.as_f64() == b.as_f64();
    }
    match (left, right) {
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| json_equal(a, b))
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(key, a)| b.get(key).is_some_and(|b| json_equal(a, b)))
        }
        _ => left == right,
    }
}

fn matches_schema(schema: &Value, value: &Value, depth: usize) -> bool {
    if depth > 32 {
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
        .is_some_and(|expected| !json_equal(expected, value))
        || object.get("enum").is_some_and(|items| {
            !items
                .as_array()
                .is_some_and(|items| items.iter().any(|expected| json_equal(expected, value)))
        })
    {
        return false;
    }
    for (keyword, mode) in [("allOf", 0), ("anyOf", 1), ("oneOf", 2)] {
        if let Some(alternatives) = schema[keyword].as_array() {
            let count = alternatives
                .iter()
                .filter(|child| matches_schema(child, value, depth + 1))
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
        .is_some_and(|child| matches_schema(child, value, depth + 1))
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
        if schema["uniqueItems"] == true
            && items
                .iter()
                .enumerate()
                .any(|(i, item)| items[..i].iter().any(|previous| json_equal(previous, item)))
        {
            return false;
        }
        if let Some(child) = object.get("items")
            && !items
                .iter()
                .all(|item| matches_schema(child, item, depth + 1))
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
                if !matches_schema(child, item, depth + 1) {
                    return false;
                }
            } else if let Some(child) = object.get("additionalProperties")
                && !matches_schema(child, item, depth + 1)
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
