use super::*;
use crate::{checkpoint, flash_support::EXPERT_BYTES, tensor_layout::MatrixLayout};
use anyhow::{Context, Result, ensure};
use half::f16;

fn counters(gpu: &Metal) -> [u64; 3] {
    [
        gpu.ssd_direct_records.get(),
        gpu.ssd_direct_read_bytes.get(),
        gpu.ssd_legacy_upload_bytes.get(),
    ]
}
fn valid_counts(enabled: bool, actual: [u64; 3], bytes: u64) -> bool {
    bytes > 0
        && if enabled {
            actual[0] > 0
                && actual[0].checked_mul(EXPERT_BYTES) == Some(bytes)
                && actual[1] == bytes
                && actual[2] == 0
        } else {
            actual == [0, 0, bytes]
        }
}
#[test]
fn direct_io_counter_oracle_rejects_disabled_empty_and_copied_paths() {
    let n = EXPERT_BYTES;
    assert!(valid_counts(true, [1, n, 0], n));
    assert!(valid_counts(false, [0, 0, n], n));
    for actual in [
        [0, 0, 0],
        [0, 0, n],
        [1, 0, 0],
        [1, n, 1],
        [1, n - 1, 0],
        [u64::MAX, n, 0],
    ] {
        assert!(!valid_counts(true, actual, n));
    }
    assert!(!valid_counts(true, [1, n, 0], 0));
    assert!(!valid_counts(false, [1, n, 0], n));
}

#[test]
#[ignore = "physical Flash expert OFF/ON direct SSD, cold/hot outputs; run alone"]
fn direct_ssd_experts_match_legacy_cold_hot_outputs_and_real_counters() -> Result<()> {
    let checkpoint =
        checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let mut reference = None;
    let mut observed = Vec::new();
    for enabled in [false, true] {
        let mut gpu = Metal::new()?;
        let initial_direct_enabled = gpu.ssd_direct_enabled;
        ensure!(
            initial_direct_enabled,
            "qualified Apple M5 direct SSD setting required"
        );
        // Test override precedes every model/queue operation; the production snapshot is immutable.
        gpu.ssd_direct_enabled = enabled;
        let mut layer =
            FlashMoeRow::load(&gpu, &checkpoint, "model.language_model.layers.0.mlp", 10)?;
        let row = (0..2560)
            .map(|i| ((i % 73) as f32 - 36.) / 53.)
            .collect::<Vec<_>>();
        let input = gpu.f32_storage(&row)?.matrix(MatrixLayout {
            shape: [1, 2560],
            strides: [0, 1],
            offset: 0,
        })?;
        let cold = layer.submit(&input)?;
        let cold_values = cold.read()?;
        ensure!(
            cold_values.len() == 2560
                && cold_values.iter().all(|&v| f32::from_bits(v).is_finite())
                && cold_values.iter().any(|&v| v != 0)
        );
        ensure!(cold.hits == 0 && cold.read_bytes == 10 * EXPERT_BYTES && cold.indices.len() == 10);
        let after_cold = counters(&gpu);
        ensure!(
            valid_counts(enabled, after_cold, cold.read_bytes),
            "cold backend did not execute the claimed path"
        );
        let hot = layer.submit(&input)?;
        ensure!(
            hot.hits == 10
                && hot.read_bytes == 0
                && hot.indices == cold.indices
                && hot.read()? == cold_values
        );
        ensure!(counters(&gpu) == after_cold && gpu.readers()?.in_flight() == 0);
        if let Some((ids, values)) = &reference {
            ensure!(*ids == cold.indices && *values == cold_values);
        } else {
            reference = Some((cold.indices.clone(), cold_values));
        }
        observed.push(serde_json::json!({"enabled":enabled,"initial_direct_enabled":initial_direct_enabled,"cold_bytes":cold.read_bytes,"hot_bytes":hot.read_bytes,"counters":after_cold,"finite_words":2560}));
    }
    eprintln!(
        "NATIVE_SSD_DIRECT_EXPERT {}",
        serde_json::json!({"branches":observed,"parity":true,"independent_route_bytes":"owned IO fixture separately validates exact packed records"})
    );
    Ok(())
}

pub(super) fn digest(bytes: &[u8]) -> Result<String> {
    use std::io::Write;
    let mut file = tempfile::NamedTempFile::new()?;
    file.write_all(bytes)?;
    let output = std::process::Command::new("/usr/bin/shasum")
        .args(["-a", "256"])
        .arg(file.path())
        .output()?;
    ensure!(output.status.success(), "state digest failed");
    let text = String::from_utf8(output.stdout)?;
    let hash = text
        .split_whitespace()
        .next()
        .context("missing state hash")?;
    ensure!(hash.len() == 64 && hash.bytes().all(|b| b.is_ascii_hexdigit()));
    Ok(hash.to_owned())
}
pub(super) fn check_tensors(states: &[FlashTraceTensor]) -> Result<usize> {
    ensure!(states.len() == 84, "incomplete state oracle");
    let mut total = 0usize;
    for state in states {
        let elements = state
            .shape
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b as usize))
            .context("state extent overflow")?;
        ensure!(
            elements > 0
                && state.data.len() == elements * 4
                && state
                    .data
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|b| f32::from_le_bytes(*b).is_finite())
        );
        total += elements;
    }
    ensure!(total == 28_803_072);
    Ok(total)
}
pub(super) fn check_initial(logits: &[u16], states: &[FlashTraceTensor]) -> Result<()> {
    let reference: serde_json::Value = serde_json::from_str(include_str!(
        "../../../docs/native-coverage/flash-state/initial.json"
    ))?;
    let expected = reference["step_tensors"]
        .as_array()
        .context("missing independent state oracle")?;
    ensure!(expected.len() == 84 && states.len() == 84);
    let words = logits
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect::<Vec<_>>();
    ensure!(digest(&words)? == "2f78a4d78cb113666fa3eda97cf115abe293bd428e593bc54400744eae422445");
    for (actual, expected) in states.iter().zip(expected) {
        ensure!(
            actual.name == expected["name"]
                && serde_json::to_value(&actual.shape)? == expected["shape"]
                && digest(&actual.data)? == expected["sha256"],
            "independent state differs: {}",
            actual.name
        );
    }
    Ok(())
}

struct ModelReference {
    rows: Vec<(Vec<u16>, Vec<FlashTraceTensor>)>,
    continuation: Vec<u16>,
    hidden: Vec<u8>,
}

#[test]
#[ignore = "physical Flash full states/logits OFF/ON direct SSD and archived initial oracle; run alone"]
fn direct_ssd_preserves_archived_full_states_logits_and_continuation() -> Result<()> {
    let checkpoint =
        checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let mut reference: Option<ModelReference> = None;
    let mut branches: Vec<serde_json::Value> = Vec::new();
    let mut words = 0usize;
    for enabled in [false, true] {
        let mut gpu = Metal::new()?;
        let initial_direct_enabled = gpu.ssd_direct_enabled;
        ensure!(
            initial_direct_enabled,
            "qualified Apple M5 direct SSD setting required"
        );
        let residency_requested = gpu.fixed_residency_enabled.get();
        if residency_requested {
            gpu.fixed_residency_enabled.set(enabled);
        }
        gpu.ssd_direct_enabled = enabled;
        let mut model = FlashNext::load(&gpu, &checkpoint, 64)?;
        let residency = model.fixed_residency_diagnostics();
        let fixed_bytes =
            serde_json::to_value(model.memory_ledger(None)?)?["fixed_weight_backing_bytes"]
                .as_u64()
                .context("missing fixed weight ledger")?;
        if residency_requested {
            ensure!(
                if enabled {
                    residency[0] > 0 && residency[1] == fixed_bytes
                } else {
                    residency == [0, 0]
                },
                "residency did not exercise the selected actual SDK path"
            );
        }
        ensure!(
            model.prefetcher.is_none(),
            "direct fixture requires depth0 without background prefetch"
        );
        let mut actual = Vec::new();
        for (step, token) in [248044, 1].into_iter().enumerate() {
            let (logits, states) = model.forward_trace(token)?;
            ensure!(
                logits.len() == 248320 && logits.iter().all(|&v| f16::from_bits(v).is_finite())
            );
            let count = check_tensors(&states)?;
            if step == 0 {
                check_initial(&logits, &states)?;
            } else {
                ensure!(
                    digest(
                        &logits
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect::<Vec<_>>()
                    )? == "833ef923345928bdfb2ea39847a722c568cb709feaa225e4476ba5f6daac3815"
                );
            }
            if let Some(expected) = &reference {
                ensure!(
                    expected.rows[step].0 == logits && expected.rows[step].1.len() == states.len()
                );
                for (left, right) in expected.rows[step].1.iter().zip(&states) {
                    ensure!(
                        left.name == right.name
                            && left.shape == right.shape
                            && left.data == right.data,
                        "direct changed {}",
                        left.name
                    );
                }
                words += count;
            }
            if reference.is_none() {
                actual.push((logits, states));
            }
        }
        ensure!(model.offset() == 2 && model.snapshot().position() == 2);
        let hidden = model
            .last_hidden()
            .context("missing hidden")?
            .export_bytes(&gpu)?;
        ensure!(
            hidden.len() == 40960
                && hidden
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .all(|b| f32::from_le_bytes(*b).is_finite())
        );
        let prefix_before_continuation = model.snapshot();
        let continuation = model.forward(42)?;
        ensure!(
            continuation.len() == 248320
                && continuation.iter().all(|&v| f16::from_bits(v).is_finite())
                && model.offset() == 3
        );
        if let Some(expected) = &reference {
            ensure!(expected.continuation == continuation && expected.hidden == hidden);
        }
        let counts = counters(&gpu);
        let read_bytes = if enabled { counts[1] } else { counts[2] };
        ensure!(valid_counts(enabled, counts, read_bytes) && gpu.readers()?.in_flight() == 0);
        if enabled {
            ensure!(
                branches[0]["counters"][2].as_u64() == Some(read_bytes),
                "direct changed demanded record bytes"
            );
        }
        let completed_offset = model.offset();
        if residency_requested {
            model.release_idle_state()?;
            ensure!(model.fixed_residency_diagnostics() == [0, 0]);
            model.restore(&prefix_before_continuation)?;
            ensure!(
                model.forward(42)? == continuation,
                "residency idle clear changed retained-prefix replay"
            );
            let resumed = model.fixed_residency_diagnostics();
            ensure!(if enabled {
                resumed[0] == residency[0] && resumed[1] == fixed_bytes
            } else {
                resumed == [0, 0]
            });
            let after = serde_json::to_value(model.memory_ledger(None)?)?;
            ensure!(after["fixed_weight_backing_bytes"].as_u64() == Some(fixed_bytes));
            model.release_idle_state()?;
            ensure!(model.fixed_residency_diagnostics() == [0, 0]);
        }
        branches.push(serde_json::json!({"enabled":enabled,"initial_direct_enabled":initial_direct_enabled,"fixed_residency_requested":residency_requested,"residency_count":residency[0],"residency_bytes":residency[1],"residency_idle_resume":residency_requested,"fixed_weight_bytes":fixed_bytes,"counters":counts,"rows":3,"trace_rows":2,"offset":completed_offset,"snapshot_position_before_continuation":2}));
        if reference.is_none() {
            reference = Some(ModelReference {
                rows: actual,
                continuation,
                hidden,
            });
        }
    }
    ensure!(words == 57_606_144);
    eprintln!(
        "NATIVE_SSD_DIRECT_MODEL {}",
        serde_json::json!({"branches":branches,"compared_state_words":words,"compared_logits":744960,"initial_archive_states":84,"finite":true,"parity":true})
    );
    Ok(())
}
