//! Test-only passive route tracing and the physical oracle for registered LFM-01.
use super::{FeedForward, Lfm2, Operator, PrefillStats};
use crate::array::{Array, Dtype};
use anyhow::{Context, Result, ensure};
use half::f16;
use std::{cell::RefCell, collections::BTreeMap, path::PathBuf};

const MOE_LAYERS: usize = 38;
const TOP_K: usize = 4;
const EXPERTS: u32 = 64;
const MICROBLOCK: usize = 8;
const CONTINUATION: usize = 17;

#[derive(Clone, Debug, PartialEq, Eq)]
struct RouteRecord {
    rows: usize,
    ids: Vec<u32>,
    scores: Vec<u16>,
}

impl RouteRecord {
    fn from_parts(
        shape: &[i32],
        dtype: Dtype,
        score_shape: &[i32],
        score_dtype: Dtype,
        ids: Vec<u32>,
        scores: Vec<u16>,
    ) -> Result<Self> {
        ensure!(
            shape.len() == 2
                && (1..=MICROBLOCK as i32).contains(&shape[0])
                && shape[1] == TOP_K as i32
                && score_shape == shape
                && dtype == Dtype::UInt32
                && score_dtype == Dtype::Float16,
            "route capture requires UInt32/F16 rows1..8, top4, and matching shapes"
        );
        let record = Self {
            rows: usize::try_from(shape[0])?,
            ids,
            scores,
        };
        record.validate()?;
        Ok(record)
    }

    fn validate(&self) -> Result<()> {
        let values = self
            .rows
            .checked_mul(TOP_K)
            .context("route count overflow")?;
        ensure!(
            (1..=MICROBLOCK).contains(&self.rows)
                && self.ids.len() == values
                && self.scores.len() == values,
            "incomplete or invalid route capture"
        );
        ensure!(
            self.ids.iter().all(|&id| id < EXPERTS),
            "expert ID outside0..64"
        );
        ensure!(
            self.scores
                .iter()
                .all(|&bits| f16::from_bits(bits).is_finite()),
            "nonfinite captured route affinity"
        );
        for ids in self.ids.as_chunks::<TOP_K>().0 {
            for left in 0..TOP_K {
                ensure!(
                    !ids[left + 1..].contains(&ids[left]),
                    "duplicate expert ID in captured top4"
                );
            }
        }
        Ok(())
    }
}

thread_local! {
    static ROUTES: RefCell<Option<Vec<RouteRecord>>> = const { RefCell::new(None) };
}

/// Called only after the actual routing normalization; inactive tracing reads no arrays.
pub(super) fn record_routes(selected: &Array, scores: &Array) -> Result<()> {
    if !ROUTES.with(|slot| slot.borrow().is_some()) {
        return Ok(());
    }
    ensure!(
        selected.dtype() == Dtype::UInt32 && scores.dtype() == Dtype::Float16,
        "actual routing dtype changed"
    );
    ensure!(
        selected.shape().len() == 2
            && (1..=MICROBLOCK as i32).contains(&selected.shape()[0])
            && selected.shape()[1] == TOP_K as i32
            && scores.shape() == selected.shape(),
        "actual routing shape changed before readback"
    );
    let record = RouteRecord::from_parts(
        selected.shape(),
        selected.dtype(),
        scores.shape(),
        scores.dtype(),
        selected.to_u32()?,
        scores.to_f16_bits()?,
    )?;
    ROUTES.with(|slot| {
        slot.borrow_mut()
            .as_mut()
            .context("route capture disappeared")?
            .push(record);
        Ok(())
    })
}

struct RouteTrace {
    active: bool,
}

impl RouteTrace {
    fn start() -> Result<Self> {
        ROUTES.with(|slot| {
            let mut slot = slot.borrow_mut();
            ensure!(slot.is_none(), "nested route capture");
            *slot = Some(Vec::new());
            Ok(Self { active: true })
        })
    }

    fn finish(mut self) -> Result<Vec<RouteRecord>> {
        let records = ROUTES.with(|slot| slot.borrow_mut().take());
        self.active = false;
        records.context("route capture was not active")
    }
}

impl Drop for RouteTrace {
    fn drop(&mut self) {
        if self.active {
            ROUTES.with(|slot| *slot.borrow_mut() = None);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct RouteRow {
    ids: [u32; TOP_K],
    scores: [u16; TOP_K],
}

#[derive(Clone, Copy)]
enum TraceOrder {
    Serial,
    Microblocks,
}

fn canonical_routes(
    records: &[RouteRecord],
    tokens: usize,
    order: TraceOrder,
) -> Result<Vec<RouteRow>> {
    ensure!(tokens > 0, "empty route comparison");
    let values = tokens
        .checked_mul(MOE_LAYERS)
        .context("trace count overflow")?;
    let width = match order {
        TraceOrder::Serial => 1,
        TraceOrder::Microblocks => MICROBLOCK,
    };
    let blocks = tokens.div_ceil(width);
    let calls = blocks
        .checked_mul(MOE_LAYERS)
        .context("trace call overflow")?;
    ensure!(
        records.len() == calls,
        "missing or extra actual router calls"
    );
    let mut result = vec![None; values];
    let mut offset = 0;
    for block in records.as_chunks::<MOE_LAYERS>().0 {
        let rows = width.min(tokens - offset);
        for (layer, record) in block.iter().enumerate() {
            record.validate()?;
            ensure!(
                record.rows == rows,
                "router rows disagree with the actual microblock"
            );
            for row in 0..rows {
                let start = row * TOP_K;
                let index = (offset + row) * MOE_LAYERS + layer;
                ensure!(
                    result[index].is_none(),
                    "duplicate canonical route position"
                );
                result[index] = Some(RouteRow {
                    ids: record.ids[start..start + TOP_K].try_into()?,
                    scores: record.scores[start..start + TOP_K].try_into()?,
                });
            }
        }
        offset += rows;
    }
    ensure!(offset == tokens, "missing captured tokens");
    result
        .into_iter()
        .map(|row| row.context("unfilled canonical route position"))
        .collect()
}

fn check_stats(stats: &PrefillStats, tokens: usize) -> Result<()> {
    ensure!(tokens > 0, "empty prefill statistics");
    let full = tokens / MICROBLOCK;
    let tail = tokens % MICROBLOCK;
    let grouped_blocks = full + usize::from(tail > 1);
    ensure!(
        stats.tokens == tokens
            && stats.blocks == full + usize::from(tail != 0)
            && stats.max_block == tokens.min(MICROBLOCK)
            && stats.router_rows
                == tokens
                    .checked_mul(MOE_LAYERS)
                    .context("router rows overflow")?
            && stats.grouped_moe_calls
                == grouped_blocks
                    .checked_mul(MOE_LAYERS)
                    .context("grouped calls overflow")?,
        "prefill counters do not establish execution of the registered microblocks"
    );
    Ok(())
}

#[derive(Debug, PartialEq, Eq)]
struct StateSnapshot {
    shape: Vec<i32>,
    dtype: Dtype,
    bytes: Vec<u8>,
}

type States = BTreeMap<String, StateSnapshot>;

fn finite_logits(model: &Lfm2, logits: &Array) -> Result<Vec<u16>> {
    ensure!(
        logits.dtype() == Dtype::Float16 && logits.shape() == [1, 1, model.vocab_size],
        "wrong full-vocabulary logit shape or dtype"
    );
    let bits = logits.to_f16_bits()?;
    ensure!(
        !bits.is_empty()
            && bits.len() == usize::try_from(model.vocab_size)?
            && bits.iter().all(|&value| f16::from_bits(value).is_finite()),
        "empty, incomplete or nonfinite full-vocabulary logits"
    );
    Ok(bits)
}

fn snapshot_states(model: &Lfm2, position: usize) -> Result<States> {
    ensure!(
        position > 0 && model.position() == i32::try_from(position)?,
        "wrong model position"
    );
    let mut expected = BTreeMap::new();
    for (index, layer) in model.layers.iter().enumerate() {
        match &layer.operator {
            Operator::Attention(_) => {
                for suffix in ["keys", "values"] {
                    expected.insert(
                        format!("model.layers.{index}.{suffix}"),
                        vec![1, 8, i32::try_from(position)?, 64],
                    );
                }
            }
            Operator::Conv(_) => {
                expected.insert(format!("model.layers.{index}.conv_state"), vec![1, 2, 2048]);
            }
        }
    }
    let arrays = model.state_arrays();
    ensure!(
        expected.len() == 50 && arrays.len() == 50,
        "expected all50 V2 cache states"
    );
    let mut result = BTreeMap::new();
    for (name, array) in arrays {
        let shape = expected.get(&name).context("unexpected cache state name")?;
        ensure!(
            array.shape() == shape.as_slice() && array.dtype() == Dtype::Float16,
            "wrong cache {name} shape or dtype"
        );
        let count = shape
            .iter()
            .try_fold(2usize, |bytes, &dim| {
                bytes.checked_mul(usize::try_from(dim).ok()?)
            })
            .context("cache byte size overflow")?;
        let bytes = array.to_bytes()?;
        ensure!(
            bytes.len() == count
                && bytes.as_chunks::<2>().0.iter().all(|pair| {
                    f16::from_bits(u16::from_ne_bytes([pair[0], pair[1]])).is_finite()
                }),
            "incomplete or nonfinite cache {name}"
        );
        ensure!(
            result
                .insert(
                    name,
                    StateSnapshot {
                        shape: shape.clone(),
                        dtype: array.dtype(),
                        bytes
                    }
                )
                .is_none(),
            "duplicate cache state name"
        );
    }
    Ok(result)
}

fn same_logits(expected: &[u16], actual: &[u16], stage: &str) -> Result<()> {
    ensure!(
        !expected.is_empty() && expected.len() == actual.len(),
        "{stage}: logit count changed"
    );
    let first = expected
        .iter()
        .zip(actual)
        .position(|(left, right)| left != right);
    ensure!(
        first.is_none(),
        "{stage}: full-vocabulary F16 logits differ at{first:?}"
    );
    Ok(())
}

fn same_states(expected: &States, actual: &States, stage: &str) -> Result<()> {
    ensure!(
        expected.len() == 50 && expected.keys().eq(actual.keys()),
        "{stage}: cache names changed"
    );
    for (name, expected) in expected {
        let actual = &actual[name];
        ensure!(
            expected.shape == actual.shape && expected.dtype == actual.dtype,
            "{stage}: cache layout changed for{name}"
        );
        let first = expected
            .bytes
            .iter()
            .zip(&actual.bytes)
            .position(|(left, right)| left != right);
        ensure!(
            expected.bytes.len() == actual.bytes.len() && first.is_none(),
            "{stage}: cache{name} differs at{first:?}"
        );
    }
    Ok(())
}

fn validate_v2(model: &Lfm2) -> Result<()> {
    ensure!(
        model.layers.len() == 40 && model.vocab_size == 128000,
        "the physical oracle requires LFM24B V2"
    );
    let mut moe = 0;
    let mut attention = 0;
    let mut conv = 0;
    for layer in &model.layers {
        if let FeedForward::Moe {
            gate, top_k, dim, ..
        } = &layer.feed_forward
        {
            ensure!(
                *top_k == TOP_K
                    && *dim == 2048
                    && gate.logical_input == 2048
                    && gate.logical_output == 64,
                "wrong V2 router geometry"
            );
            moe += 1;
        }
        match &layer.operator {
            Operator::Attention(layer) => {
                ensure!(
                    layer.kv_heads == 8 && layer.head_dim == 64,
                    "wrong V2 attention geometry"
                );
                attention += 1;
            }
            Operator::Conv(layer) => {
                ensure!(
                    layer.dim == 2048 && layer.length == 3,
                    "wrong V2 convolution geometry"
                );
                conv += 1;
            }
        }
    }
    ensure!(
        moe == MOE_LAYERS && attention == 10 && conv == 30,
        "incomplete V2 layer panel"
    );
    Ok(())
}

fn load_v2() -> Result<Lfm2> {
    let path = std::env::var_os("MLXL3_LFM_TEST_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/Users/justin/Documents/mix-stq1_0/models/LFM2.5-24B-A2B-EXL3-4bpw-v2")
        });
    let model =
        Lfm2::load(&path).with_context(|| format!("load physical V2 fixture{}", path.display()))?;
    validate_v2(&model)?;
    ensure!(
        model.supports_moe_exact_prefill(),
        "V2 checkpoint is not eligible for the prototype"
    );
    Ok(model)
}

fn tokens(count: usize) -> Vec<u32> {
    (0..count)
        .map(|position| ((position * 73 + 101) % 125017) as u32)
        .collect()
}

fn seed_prefix(model: &mut Lfm2, prefix: &[u32]) -> Result<()> {
    model.reset();
    ensure!(
        model.position() == 0 && model.state_arrays().is_empty(),
        "reset kept cache state"
    );
    for &token in prefix {
        let logits = model.forward(&[token])?;
        finite_logits(model, &logits)?;
    }
    Ok(())
}

#[derive(Debug)]
struct StepSnapshot {
    logits: Vec<u16>,
    states: States,
}

fn continuation(
    model: &mut Lfm2,
    input: &[u32],
    initial: usize,
) -> Result<(Vec<StepSnapshot>, Vec<RouteRow>)> {
    ensure!(
        input.len() == CONTINUATION,
        "expected17 forced decode steps"
    );
    let trace = RouteTrace::start()?;
    let mut steps = Vec::new();
    for (index, &token) in input.iter().enumerate() {
        let logits = model.forward(&[token])?;
        steps.push(StepSnapshot {
            logits: finite_logits(model, &logits)?,
            states: snapshot_states(model, initial + index + 1)?,
        });
    }
    let routes = canonical_routes(&trace.finish()?, input.len(), TraceOrder::Serial)?;
    Ok((steps, routes))
}

fn parity_case(model: &mut Lfm2, length: usize, prefix: usize) -> Result<serde_json::Value> {
    let count = prefix
        .checked_add(length)
        .and_then(|n| n.checked_add(CONTINUATION))
        .context("test input overflow")?;
    let input = tokens(count);
    let (before, tail) = input.split_at(prefix);
    let (prompt, decode) = tail.split_at(length);
    seed_prefix(model, before)?;
    let trace = RouteTrace::start()?;
    let mut reference_logits = None;
    for &token in prompt {
        let logits = model.forward(&[token])?;
        reference_logits = Some(finite_logits(model, &logits)?);
    }
    let reference_routes = canonical_routes(&trace.finish()?, length, TraceOrder::Serial)?;
    let reference_logits = reference_logits.context("empty reference prefill")?;
    let reference_states = snapshot_states(model, prefix + length)?;
    let (reference_decode, reference_decode_routes) = continuation(model, decode, prefix + length)?;

    seed_prefix(model, before)?;
    let trace = RouteTrace::start()?;
    let (logits, stats) = model.prefill_moe_exact(prompt)?;
    check_stats(&stats, length)?;
    let actual_logits = finite_logits(model, &logits)?;
    let actual_routes = canonical_routes(&trace.finish()?, length, TraceOrder::Microblocks)?;
    same_logits(&reference_logits, &actual_logits, "prefill terminal")?;
    same_states(
        &reference_states,
        &snapshot_states(model, prefix + length)?,
        "prefill terminal",
    )?;
    ensure!(
        reference_routes == actual_routes,
        "prefill routes or affinities differ"
    );
    let (actual_decode, actual_decode_routes) = continuation(model, decode, prefix + length)?;
    for (index, (expected, actual)) in reference_decode.iter().zip(&actual_decode).enumerate() {
        same_logits(&expected.logits, &actual.logits, &format!("decode{index}"))?;
        same_states(&expected.states, &actual.states, &format!("decode{index}"))?;
    }
    ensure!(
        reference_decode.len() == CONTINUATION
            && actual_decode.len() == CONTINUATION
            && reference_decode_routes == actual_decode_routes,
        "forced continuation is incomplete or changes routes"
    );
    model.reset();
    ensure!(
        model.position() == 0 && model.state_arrays().is_empty(),
        "final reset kept cache state"
    );
    Ok(serde_json::json!({
        "status":"complete", "parity":true, "tokens":length, "prefix":prefix,
        "terminal_logits":actual_logits.len(), "states":reference_states.len(),
        "route_rows":actual_routes.len(), "forced_decode_steps":actual_decode.len(),
        "stats":{"tokens":stats.tokens,"blocks":stats.blocks,"max_block":stats.max_block,
            "router_rows":stats.router_rows,"grouped_moe_calls":stats.grouped_moe_calls},
        "scope":"terminal full vocabulary and17forced continuations, not every prefill position",
        "performance":"not measured"
    }))
}

fn selected_cases() -> Result<(Vec<usize>, Vec<usize>)> {
    let lengths = match std::env::var("MLXL3_LFM_TEST_LENGTH") {
        Ok(value) => {
            let value: usize = value.parse().context("invalid MLXL3_LFM_TEST_LENGTH")?;
            ensure!((1..=600).contains(&value), "test length must be1..600");
            vec![value]
        }
        Err(std::env::VarError::NotPresent) => {
            vec![1, 2, 3, 4, 5, 6, 7, 8, 23, 24, 25, 63, 64, 65, 128, 266]
        }
        Err(error) => return Err(error.into()),
    };
    let prefixes = match std::env::var("MLXL3_LFM_TEST_PREFIX") {
        Ok(value) => {
            let value: usize = value.parse().context("invalid MLXL3_LFM_TEST_PREFIX")?;
            ensure!(value == 0 || value == 17, "test prefix must be0or17");
            vec![value]
        }
        Err(std::env::VarError::NotPresent) => vec![0, 17],
        Err(error) => return Err(error.into()),
    };
    Ok((lengths, prefixes))
}

fn check_rejected_inputs(model: &mut Lfm2) -> Result<()> {
    seed_prefix(model, &[101, 174])?;
    let expected = snapshot_states(model, 2)?;
    let trace = RouteTrace::start()?;
    for invalid in [Vec::new(), vec![u32::MAX]] {
        ensure!(
            model.prefill_moe_exact(&invalid).is_err(),
            "invalid prefill input accepted"
        );
        same_states(
            &expected,
            &snapshot_states(model, 2)?,
            "rejected prefill input",
        )?;
    }
    let original_limit = model.context_limit;
    model.context_limit = original_limit.min(4096);
    let remaining = usize::try_from(model.context_limit - model.position())?;
    let overfull = vec![247; remaining + 1];
    ensure!(
        overfull.len() <= 4096,
        "oversized context rejection fixture"
    );
    let overflow = model.prefill_moe_exact(&overfull);
    model.context_limit = original_limit;
    ensure!(overflow.is_err(), "prefill exceeded the context boundary");
    same_states(
        &expected,
        &snapshot_states(model, 2)?,
        "full context rejection",
    )?;
    ensure!(
        trace.finish()?.is_empty(),
        "rejected prefill executed a router"
    );
    model.reset();
    ensure!(
        model.position() == 0 && model.state_arrays().is_empty(),
        "rejection reset failed"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and exact LFM24B V2; one selected case per60s supervision"]
fn lfm_moe_microblocks_match_serial_logits_routes_states() -> Result<()> {
    let (lengths, prefixes) = selected_cases()?;
    let mut model = load_v2()?;
    check_rejected_inputs(&mut model)?;
    for prefix in prefixes {
        for &length in &lengths {
            match parity_case(&mut model, length, prefix) {
                Ok(report) => println!("{report}"),
                Err(error) => {
                    model.reset();
                    println!(
                        "{}",
                        serde_json::json!({"status":"failed","parity":false,"tokens":length,"prefix":prefix,"error":format!("{error:#}"),"performance":"not measured"})
                    );
                    return Err(error);
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and LFM24B V2; passive source/candidate128 control"]
fn lfm_moe_route_capture_is_passive() -> Result<()> {
    let mut model = load_v2()?;
    let input = tokens(128);
    let mut expected = None;
    for &token in &input {
        let logits = model.forward(&[token])?;
        expected = Some(finite_logits(&model, &logits)?);
    }
    let expected = expected.context("missing source logits")?;
    let expected_states = snapshot_states(&model, input.len())?;
    model.reset();
    let trace = RouteTrace::start()?;
    let mut actual = None;
    for &token in &input {
        let logits = model.forward(&[token])?;
        actual = Some(finite_logits(&model, &logits)?);
    }
    let records = trace.finish()?;
    let routes = canonical_routes(&records, input.len(), TraceOrder::Serial)?;
    same_logits(
        &expected,
        &actual.context("missing captured source logits")?,
        "passive source128",
    )?;
    same_states(
        &expected_states,
        &snapshot_states(&model, input.len())?,
        "passive source128",
    )?;
    ensure!(
        records.len() == 128 * MOE_LAYERS && routes.len() == 128 * MOE_LAYERS,
        "empty passive source trace"
    );

    model.reset();
    let (uncaptured, uncaptured_stats) = model.prefill_moe_exact(&input)?;
    check_stats(&uncaptured_stats, input.len())?;
    let uncaptured = finite_logits(&model, &uncaptured)?;
    let uncaptured_states = snapshot_states(&model, input.len())?;
    model.reset();
    let trace = RouteTrace::start()?;
    let (captured, captured_stats) = model.prefill_moe_exact(&input)?;
    check_stats(&captured_stats, input.len())?;
    let captured = finite_logits(&model, &captured)?;
    let candidate_routes =
        canonical_routes(&trace.finish()?, input.len(), TraceOrder::Microblocks)?;
    same_logits(&uncaptured, &captured, "passive candidate128")?;
    same_states(
        &uncaptured_states,
        &snapshot_states(&model, input.len())?,
        "passive candidate128",
    )?;
    ensure!(
        uncaptured_stats == captured_stats && candidate_routes.len() == 128 * MOE_LAYERS,
        "candidate capture changed dispatches or produced an empty trace"
    );
    model.reset();
    ensure!(
        model.position() == 0 && model.state_arrays().is_empty(),
        "passive control reset failed"
    );
    println!(
        "{}",
        serde_json::json!({
            "status":"complete", "parity":true, "tokens":128, "states":expected_states.len(),
            "captured_calls":records.len(), "route_rows":routes.len(),
            "candidate_route_rows":candidate_routes.len(), "candidate_grouped_moe_calls":captured_stats.grouped_moe_calls,
            "scope":"source and candidate, route capture ON/OFF, not source-candidate parity",
            "performance":"not measured"
        })
    );
    Ok(())
}

fn fixture_row(token: usize, layer: usize) -> RouteRow {
    let base = ((token * 11 + layer * 7) % 61) as u32;
    RouteRow {
        ids: [base, base + 1, base + 2, base + 3],
        scores: [
            0x3000 + ((token * MOE_LAYERS + layer) % 0x1000) as u16,
            0x3400,
            0x3800,
            0x3c00,
        ],
    }
}

fn fixture_trace(tokens: usize, width: usize) -> Vec<RouteRecord> {
    let mut records = Vec::new();
    for start in (0..tokens).step_by(width) {
        let rows = width.min(tokens - start);
        for layer in 0..MOE_LAYERS {
            let mut ids = Vec::new();
            let mut scores = Vec::new();
            for token in start..start + rows {
                let row = fixture_row(token, layer);
                ids.extend(row.ids);
                scores.extend(row.scores);
            }
            records.push(RouteRecord { rows, ids, scores });
        }
    }
    records
}

#[test]
fn route_canonicalization_matches_independent_token_layer_fixture() -> Result<()> {
    for tokens in [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 15, 16, 17, 23, 24, 25, 63, 64, 65, 128, 266,
    ] {
        let expected = (0..tokens)
            .flat_map(|token| (0..MOE_LAYERS).map(move |layer| fixture_row(token, layer)))
            .collect::<Vec<_>>();
        assert_eq!(
            canonical_routes(&fixture_trace(tokens, 1), tokens, TraceOrder::Serial)?,
            expected
        );
        assert_eq!(
            canonical_routes(
                &fixture_trace(tokens, MICROBLOCK),
                tokens,
                TraceOrder::Microblocks
            )?,
            expected
        );
    }
    Ok(())
}

#[test]
fn route_canonicalization_rejects_incomplete_rows_and_counts() {
    assert!(canonical_routes(&[], 0, TraceOrder::Serial).is_err());
    assert!(canonical_routes(&[], usize::MAX, TraceOrder::Microblocks).is_err());
    assert!(canonical_routes(&[], 1, TraceOrder::Serial).is_err());
    let good = fixture_trace(9, MICROBLOCK);
    let mut short = good.clone();
    short.pop();
    assert!(canonical_routes(&short, 9, TraceOrder::Microblocks).is_err());
    let mut extra = good.clone();
    extra.push(good[0].clone());
    assert!(canonical_routes(&extra, 9, TraceOrder::Microblocks).is_err());
    let mut rows = good.clone();
    rows[1] = fixture_trace(1, 1)[0].clone();
    assert!(canonical_routes(&rows, 9, TraceOrder::Microblocks).is_err());
    let mut incomplete = good;
    incomplete[0].scores.pop();
    assert!(canonical_routes(&incomplete, 9, TraceOrder::Microblocks).is_err());
}

#[test]
fn route_capture_refuses_bad_shapes_dtypes_ids_and_nonfinite_scores() {
    let row = fixture_row(0, 0);
    for shape in [
        vec![],
        vec![1],
        vec![0, 4],
        vec![-1, 4],
        vec![9, 4],
        vec![1, 3],
    ] {
        assert!(
            RouteRecord::from_parts(
                &shape,
                Dtype::UInt32,
                &shape,
                Dtype::Float16,
                row.ids.to_vec(),
                row.scores.to_vec()
            )
            .is_err()
        );
    }
    for (dtype, score_dtype, score_shape) in [
        (Dtype::Int32, Dtype::Float16, vec![1, 4]),
        (Dtype::UInt32, Dtype::Float32, vec![1, 4]),
        (Dtype::UInt32, Dtype::Float16, vec![4, 1]),
    ] {
        assert!(
            RouteRecord::from_parts(
                &[1, 4],
                dtype,
                &score_shape,
                score_dtype,
                row.ids.to_vec(),
                row.scores.to_vec()
            )
            .is_err()
        );
    }
    for ids in [vec![0, 1, 2], vec![0, 1, 2, 64], vec![0, 1, 1, 3]] {
        assert!(
            RouteRecord::from_parts(
                &[1, 4],
                Dtype::UInt32,
                &[1, 4],
                Dtype::Float16,
                ids,
                row.scores.to_vec()
            )
            .is_err()
        );
    }
    for scores in [
        vec![0x3000; 3],
        vec![0x3000, 0x3400, 0x3800, 0x7e00],
        vec![0x3000, 0x3400, 0x3800, 0x7c00],
    ] {
        assert!(
            RouteRecord::from_parts(
                &[1, 4],
                Dtype::UInt32,
                &[1, 4],
                Dtype::Float16,
                row.ids.to_vec(),
                scores
            )
            .is_err()
        );
    }
}

#[test]
fn a_valid_route_order_mutation_is_detected_by_the_fixture_oracle() -> Result<()> {
    let mut trace = fixture_trace(9, MICROBLOCK);
    let expected = canonical_routes(&trace, 9, TraceOrder::Microblocks)?;
    trace[0].ids.swap(0, 1);
    trace[0].validate()?;
    assert_ne!(
        canonical_routes(&trace, 9, TraceOrder::Microblocks)?,
        expected
    );
    Ok(())
}

#[test]
fn route_trace_scope_refuses_nesting_and_restores_after_error() -> Result<()> {
    let trace = RouteTrace::start()?;
    assert!(RouteTrace::start().is_err());
    assert!(trace.finish()?.is_empty());
    let interrupted = || -> Result<()> {
        let _trace = RouteTrace::start()?;
        anyhow::bail!("injected trace interruption")
    };
    assert!(interrupted().is_err());
    assert!(ROUTES.with(|slot| slot.borrow().is_none()));
    Ok(())
}

#[test]
fn prefill_counters_require_real_grouped_execution() -> Result<()> {
    for (tokens, blocks, grouped, maximum) in [
        (1, 1, 0, 1),
        (2, 1, 38, 2),
        (8, 1, 38, 8),
        (9, 2, 38, 8),
        (16, 2, 76, 8),
        (17, 3, 76, 8),
        (23, 3, 114, 8),
    ] {
        let mut stats = PrefillStats {
            tokens,
            blocks,
            grouped_moe_calls: grouped,
            router_rows: MOE_LAYERS * tokens,
            max_block: maximum,
        };
        check_stats(&stats, tokens)?;
        stats.router_rows -= 1;
        assert!(check_stats(&stats, tokens).is_err());
    }
    Ok(())
}

#[test]
fn microblock_eligibility_requires_v2_config_and_every_mapped_projection() -> Result<()> {
    use crate::checkpoint::{Checkpoint, TensorInfo};
    let mut config = serde_json::json!({
        "model_type":"lfm2_moe", "vocab_size":128000, "hidden_size":2048,
        "max_position_embeddings":131072, "num_hidden_layers":40,
        "num_attention_heads":32, "num_key_value_heads":8, "norm_eps":0.00001,
        "conv_bias":false, "conv_L_cache":3, "intermediate_size":11776,
        "moe_intermediate_size":1536, "num_experts":64, "num_experts_per_tok":4,
        "num_dense_layers":2, "norm_topk_prob":true, "use_expert_bias":true,
        "routed_scaling_factor":1.0
    });
    let good: super::Config = serde_json::from_value(config.clone())?;
    assert!(super::exact_moe_prefill_config(&good, 10, true));
    assert!(!super::exact_moe_prefill_config(&good, 9, true));
    assert!(!super::exact_moe_prefill_config(&good, 10, false));
    for (key, value) in [
        ("model_type", serde_json::json!("lfm2")),
        ("hidden_size", serde_json::json!(2047)),
        ("moe_intermediate_size", serde_json::json!(1664)),
        ("num_experts", serde_json::json!(32)),
        ("num_experts_per_tok", serde_json::json!(8)),
        ("num_hidden_layers", serde_json::json!(39)),
        ("num_dense_layers", serde_json::json!(1)),
        ("num_attention_heads", serde_json::json!(16)),
        ("num_key_value_heads", serde_json::json!(4)),
        ("conv_L_cache", serde_json::json!(4)),
        ("vocab_size", serde_json::json!(65536)),
        ("use_expert_bias", serde_json::json!(false)),
        ("norm_topk_prob", serde_json::json!(false)),
        ("routed_scaling_factor", serde_json::json!(2.0)),
    ] {
        let original = config[key].take();
        config[key] = value;
        let invalid: super::Config = serde_json::from_value(config.clone())?;
        assert!(
            !super::exact_moe_prefill_config(&invalid, 10, true),
            "accepted changed{key}"
        );
        config[key] = original;
    }
    // Headers only: no payloads, MLX arrays or GPU calls are used by this fixture.
    let info = |dtype: &str, shape: &[usize]| -> Result<TensorInfo> {
        Ok(serde_json::from_value(
            serde_json::json!({"dtype":dtype,"shape":shape,"data_offsets":[0,0]}),
        )?)
    };
    let marker = info("BOOL", &[])?;
    let mut checkpoint = Checkpoint {
        path: PathBuf::new(),
        model_type: "lfm2_moe".into(),
        bits: Some(4.0),
        size_bytes: 0,
        modules: Vec::new(),
        tensors: BTreeMap::new(),
    };
    for layer in 2..40 {
        let prefix = format!("model.layers.{layer}.feed_forward");
        checkpoint
            .tensors
            .insert(format!("{prefix}.gate.weight"), info("F16", &[64, 2048])?);
        for expert in 0..64 {
            for (projection, shape) in [
                ("w1", [128, 96, 64]),
                ("w3", [128, 96, 64]),
                ("w2", [96, 128, 64]),
            ] {
                let base = format!("{prefix}.experts.{expert}.{projection}");
                checkpoint
                    .tensors
                    .insert(format!("{base}.trellis"), info("U16", &shape)?);
                checkpoint
                    .tensors
                    .insert(format!("{base}.mcg"), marker.clone());
            }
        }
    }
    assert!(super::exact_moe_projection_metadata(&checkpoint));
    checkpoint
        .modules
        .push("model.layers.39.feed_forward.gate".into());
    assert!(!super::exact_moe_projection_metadata(&checkpoint));
    checkpoint.modules.clear();
    let last = "model.layers.39.feed_forward.experts.63.w2";
    let key = format!("{last}.trellis");
    let original = checkpoint.tensors.remove(&key).unwrap();
    assert!(!super::exact_moe_projection_metadata(&checkpoint));
    checkpoint
        .tensors
        .insert(key.clone(), info("U16", &[96, 128, 112])?);
    assert!(!super::exact_moe_projection_metadata(&checkpoint));
    checkpoint
        .tensors
        .insert(key.clone(), info("F32", &[96, 128, 64])?);
    assert!(!super::exact_moe_projection_metadata(&checkpoint));
    checkpoint.tensors.insert(key, original);
    checkpoint.tensors.insert(format!("{last}.mul1"), marker);
    assert!(!super::exact_moe_projection_metadata(&checkpoint));
    checkpoint.tensors.remove(&format!("{last}.mul1"));
    let marker = checkpoint.tensors.remove(&format!("{last}.mcg")).unwrap();
    assert!(!super::exact_moe_projection_metadata(&checkpoint));
    checkpoint.tensors.insert(format!("{last}.mcg"), marker);
    assert!(super::exact_moe_projection_metadata(&checkpoint));
    Ok(())
}
