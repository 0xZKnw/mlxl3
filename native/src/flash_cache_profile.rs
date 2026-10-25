//! Offline experiment47; no runtime cache policy is selected here.

#[path = "flash_cache_profile/profile53.rs"]
mod profile53;

fn allocate_levels(costs: &[Vec<u32>], budget: usize) -> Option<Vec<usize>> {
    let layers = costs.len();
    let levels = costs.first()?.len();
    if !(1..=48).contains(&layers)
        || !(1..=124).contains(&levels)
        || costs.iter().any(|row| row.len() != levels)
        || budget > layers * (levels - 1)
    {
        return None;
    }
    let mut previous = vec![None; budget + 1];
    previous[0] = Some((0u64, 0u64));
    let mut parents = vec![vec![usize::MAX; budget + 1]; layers];
    for (layer, row) in costs.iter().enumerate() {
        let mut next = vec![None; budget + 1];
        for (level, &misses) in row.iter().enumerate().take(budget + 1) {
            let distance = (20 + level * 4).abs_diff(53) as u64;
            for used in level..=budget {
                if let Some((cost, deviation)) = previous[used - level] {
                    let candidate = (cost + u64::from(misses), deviation + distance * distance);
                    if next[used].is_none_or(|old| candidate < old) {
                        next[used] = Some(candidate);
                        parents[layer][used] = level;
                    }
                }
            }
        }
        previous = next;
    }
    previous[budget]?;
    let mut remaining = budget;
    let mut result = vec![0; layers];
    for layer in (0..layers).rev() {
        let level = parents[layer][remaining];
        remaining = remaining.checked_sub(level)?;
        result[layer] = level;
    }
    (remaining == 0).then_some(result)
}

#[cfg(test)]
pub(crate) const CALIBRATION_PROMPT: &str =
    "Explain the difference between CPU caches and virtual memory. Give a numerical example.";

#[cfg(test)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct CacheTrace {
    pub status: String,
    pub slots: usize,
    pub prompt: String,
    pub thinking: String,
    pub prompt_ids: Vec<u32>,
    pub phases: Vec<TracePhase>,
}

#[cfg(test)]
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct TracePhase {
    pub routes: Vec<Vec<Vec<u32>>>,
    pub tokens: Vec<u32>,
    pub text: String,
    pub prefill_counts: Vec<[u64; 3]>,
    pub decode_counts: Vec<[u64; 3]>,
    pub logit_count: usize,
    pub logits_finite: bool,
    pub state_buffers: usize,
}

#[cfg(test)]
impl CacheTrace {
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        use crate::flash_support::EXPERT_BYTES;
        use anyhow::ensure;
        ensure!(self.status == "complete" && self.slots == 53);
        ensure!(self.prompt == CALIBRATION_PROMPT && self.thinking == "default_on");
        ensure!((1..=512).contains(&self.prompt_ids.len()));
        ensure!(self.prompt_ids.iter().all(|&id| id < 248320) && self.phases.len() == 2);
        let prefill = self.prompt_ids.len();
        for phase in &self.phases {
            ensure!(phase.tokens.len() == 128 && phase.tokens.iter().all(|&t| t < 248320));
            ensure!(!phase.text.is_empty() && phase.routes.len() == prefill + 127);
            ensure!(
                phase.logit_count == 248320 && phase.logits_finite && phase.state_buffers > 100
            );
            ensure!(phase.prefill_counts.len() == 48 && phase.decode_counts.len() == 48);
            for layers in &phase.routes {
                ensure!(layers.len() == 48);
                for ids in layers {
                    ensure!(ids.len() == 10 && ids.iter().all(|&id| id < 512));
                    let mut unique = ids.clone();
                    unique.sort_unstable();
                    unique.dedup();
                    ensure!(unique.len() == 10);
                }
            }
            for (steps, counts) in [
                (prefill, &phase.prefill_counts),
                (127, &phase.decode_counts),
            ] {
                for &[hits, misses, bytes] in counts {
                    ensure!(hits <= steps as u64 * 10 && misses == steps as u64 * 10 - hits);
                    ensure!(bytes == misses * EXPERT_BYTES);
                }
            }
        }
        let [first, second] = self.phases.as_slice() else {
            unreachable!()
        };
        ensure!(
            first.tokens == second.tokens
                && first.text == second.text
                && first.routes == second.routes
        );
        Ok(())
    }

    fn replay_layer(&self, layer: usize, slots: usize) -> anyhow::Result<Vec<[[u64; 3]; 2]>> {
        use crate::flash_support::{EXPERT_BYTES, ExpertCache};
        let mut cache = ExpertCache::new(slots)?;
        self.phases
            .iter()
            .map(|phase| {
                let mut counts = [[0u64; 3]; 2];
                for (step, layers) in phase.routes.iter().enumerate() {
                    let mapped = cache.reserve(&layers[layer])?;
                    let misses = mapped.iter().filter(|(_, hit)| !hit).count() as u64;
                    let count = &mut counts[usize::from(step >= self.prompt_ids.len())];
                    count[0] += 10 - misses;
                    count[1] += misses;
                    count[2] += misses * EXPERT_BYTES;
                }
                Ok(counts)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::{Result, ensure};

    #[test]
    fn allocation_matches_exhaustive_assignments_and_rejects_invalid_budgets() {
        for seed in 0..32u32 {
            let costs = (0..3)
                .map(|layer| {
                    (0..4)
                        .map(|q| (seed * 37 + layer * 19 + q * q * 11) % 53)
                        .collect()
                })
                .collect::<Vec<Vec<u32>>>();
            for budget in 0..=9 {
                let actual = allocate_levels(&costs, budget).unwrap();
                let mut candidates = Vec::new();
                for a in 0..4 {
                    for b in 0..4 {
                        for c in 0..4 {
                            if a + b + c == budget {
                                let levels = [a, b, c];
                                let cost =
                                    (0..3).map(|i| u64::from(costs[i][levels[i]])).sum::<u64>();
                                let deviation = levels
                                    .iter()
                                    .map(|q| (20 + 4 * q).abs_diff(53).pow(2))
                                    .sum::<usize>();
                                candidates.push((cost, deviation, [c, b, a]));
                            }
                        }
                    }
                }
                candidates.sort_unstable();
                assert_eq!(
                    actual,
                    candidates[0].2.into_iter().rev().collect::<Vec<_>>()
                );
            }
        }
        assert_eq!(allocate_levels(&[], 0), None);
        assert_eq!(allocate_levels(&[vec![]], 0), None);
        assert_eq!(allocate_levels(&[vec![1], vec![2, 3]], 0), None);
        assert_eq!(allocate_levels(&vec![vec![1]; 49], 0), None);
        assert_eq!(allocate_levels(&[vec![1; 125]], 0), None);
        assert_eq!(
            allocate_levels(&vec![vec![u32::MAX; 2]; 2], usize::MAX),
            None
        );
        assert_eq!(
            allocate_levels(&vec![vec![u32::MAX; 2]; 2], 1),
            Some(vec![1, 0])
        );
    }

    fn fixture() -> CacheTrace {
        let cold = TracePhase {
            routes: vec![vec![(0..10).collect(); 48]; 128],
            tokens: vec![7; 128],
            text: "known text".into(),
            prefill_counts: vec![[0, 10, 10 * crate::flash_support::EXPERT_BYTES]; 48],
            decode_counts: vec![[1270, 0, 0]; 48],
            logit_count: 248320,
            logits_finite: true,
            state_buffers: 110,
        };
        let mut warm = cold.clone();
        warm.prefill_counts.fill([10, 0, 0]);
        CacheTrace {
            status: "complete".into(),
            slots: 53,
            prompt: CALIBRATION_PROMPT.into(),
            thinking: "default_on".into(),
            prompt_ids: vec![1],
            phases: vec![cold, warm],
        }
    }

    #[test]
    fn trace_validation_and_real_cache_replay_cover_cold_warm_and_corruption() -> Result<()> {
        let original = fixture();
        original.validate()?;
        for slots in [20, 24, 53, 512] {
            let replay = original.replay_layer(0, slots)?;
            assert_eq!(
                replay,
                vec![
                    [original.phases[0].prefill_counts[0], [1270, 0, 0]],
                    [[10, 0, 0], [1270, 0, 0]]
                ]
            );
        }
        for mutation in 0..12 {
            let mut bad = original.clone();
            match mutation {
                0 => bad.status = "incomplete".into(),
                1 => bad.phases.clear(),
                2 => bad.phases[0].routes.pop().map(|_| ()).unwrap(),
                3 => bad.phases[0].routes[0][0][0] = 512,
                4 => bad.phases[0].routes[0][0][0] = 1,
                5 => bad.phases[0].decode_counts[0][1] = 1,
                6 => bad.phases[0].tokens.clear(),
                7 => bad.phases[0].logits_finite = false,
                8 => bad.phases[0].text.clear(),
                9 => bad.phases[1].tokens[0] = 8,
                10 => bad.prompt_ids.clear(),
                11 => bad.phases[0].prefill_counts[0][2] = 1,
                _ => unreachable!(),
            }
            assert!(bad.validate().is_err(), "accepted mutation {mutation}");
        }
        Ok(())
    }

    fn write_allocation(input: &std::path::Path, output: &std::path::Path) -> Result<()> {
        use std::fs;
        let marker = fs::File::create_new(output)?;
        serde_json::to_writer(marker, &serde_json::json!({"status":"incomplete"}))?;
        ensure!((1..=16 << 20).contains(&fs::metadata(input)?.len()));
        let trace: CacheTrace = serde_json::from_slice(&fs::read(input)?)?;
        trace.validate()?;
        let mut curves = Vec::new();
        let mut baseline = 0u64;
        for layer in 0..48 {
            let actual = trace.replay_layer(layer, 53)?;
            for (phase, counts) in trace.phases.iter().zip(&actual) {
                ensure!(
                    counts == &[phase.prefill_counts[layer], phase.decode_counts[layer]],
                    "53-slot replay differs at layer {layer}"
                );
            }
            baseline += actual[1][1][1];
            curves.push(
                (20..=512)
                    .step_by(4)
                    .map(|slots| Ok(u32::try_from(trace.replay_layer(layer, slots)?[1][1][1])?))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
        let levels = allocate_levels(&curves, (48 * 53 - 48 * 20) / 4).expect("valid fixed budget");
        let slots = levels.iter().map(|q| 20 + 4 * q).collect::<Vec<_>>();
        ensure!(slots.iter().sum::<usize>() == 48 * 53);
        let predicted = levels
            .iter()
            .enumerate()
            .map(|(i, &q)| u64::from(curves[i][q]))
            .sum::<u64>();
        let improvement = if baseline == 0 {
            0.0
        } else {
            100.0 * (baseline as f64 - predicted as f64) / baseline as f64
        };
        fs::write(
            output,
            serde_json::to_vec_pretty(&serde_json::json!({
                "status":"complete_prediction_not_speed", "baseline_slots":53, "total_slots":48*53,
                "calibration_prompt":CALIBRATION_PROMPT, "curve_min":20, "curve_step":4, "curve_max":512,
                "decode_miss_curves":curves, "selected_slots":slots, "baseline_decode_record_loads":baseline,
                "predicted_decode_record_loads":predicted, "predicted_read_reduction_percent":improvement,
                "eligible_for_runtime_trial":improvement >= 1.0, "replayed_baseline_counts":"exact_both_phases_all_layers",
                "physical_io":"not measured", "confirmed_tps_gain":false
            }))?,
        )?;
        println!(
            "FLASH_CACHE_ALLOCATION baseline_records={baseline} predicted_records={predicted} total_slots=2544 prediction_not_speed=true"
        );
        Ok(())
    }

    #[test]
    fn allocation_files_reject_partial_false_counts_and_overwrite() -> Result<()> {
        use std::fs;
        let directory = tempfile::tempdir()?;
        let input = directory.path().join("trace.json");
        for case in ["truncated", "false_counts"] {
            let output = directory.path().join(format!("{case}.json"));
            if case == "truncated" {
                fs::write(&input, "{")?;
            } else {
                let mut trace = fixture();
                trace.phases[0].prefill_counts.fill([10, 0, 0]);
                trace.validate()?;
                fs::write(&input, serde_json::to_vec(&trace)?)?;
            }
            assert!(write_allocation(&input, &output).is_err());
            let marker: serde_json::Value = serde_json::from_slice(&fs::read(&output)?)?;
            assert_eq!(marker["status"], "incomplete");
            assert!(write_allocation(&input, &output).is_err());
        }
        fs::write(&input, serde_json::to_vec(&fixture())?)?;
        let output = directory.path().join("complete.json");
        write_allocation(&input, &output)?;
        let bytes = fs::read(&output)?;
        let report: serde_json::Value = serde_json::from_slice(&bytes)?;
        assert_eq!(report["status"], "complete_prediction_not_speed");
        assert_eq!(report["baseline_decode_record_loads"], 0);
        assert_eq!(report["predicted_decode_record_loads"], 0);
        assert_eq!(report["eligible_for_runtime_trial"], false);
        assert_eq!(report["confirmed_tps_gain"], false);
        assert_eq!(
            report["selected_slots"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .sum::<u64>(),
            2544
        );
        assert!(write_allocation(&input, &output).is_err());
        assert_eq!(bytes, fs::read(&output)?);
        Ok(())
    }

    #[test]
    #[ignore = "offline experiment47; requires a complete physical calibration trace, no GPU work"]
    fn replay_physical_trace_and_allocate_fixed_expert_budget() -> Result<()> {
        use std::path::Path;
        write_allocation(
            Path::new(&std::env::var("MLXL3_FLASH_CACHE_TRACE")?),
            Path::new(&std::env::var("MLXL3_FLASH_CACHE_ALLOCATION")?),
        )
    }
}

#[cfg(kani)]
mod verification {
    use super::*;
    #[kani::proof]
    #[kani::unwind(5)]
    fn cache_allocation_is_optimal_and_exact_on_two_layers_two_levels() {
        let values: [u32; 4] = kani::any();
        let budget: usize = kani::any();
        let costs = vec![values[..2].to_vec(), values[2..].to_vec()];
        let result = allocate_levels(&costs, budget);
        let missing = result.is_none();
        if budget > 2 {
            assert!(result.is_none());
        } else {
            let selected = result.unwrap();
            assert_eq!(selected.len(), 2);
            assert!(selected.iter().all(|&q| q < 2));
            assert_eq!(selected[0] + selected[1], budget);
            let actual = u64::from(costs[0][selected[0]]) + u64::from(costs[1][selected[1]]);
            let expected = match budget {
                0 => u64::from(values[0]) + u64::from(values[2]),
                1 => (u64::from(values[0]) + u64::from(values[3]))
                    .min(u64::from(values[1]) + u64::from(values[2])),
                2 => u64::from(values[1]) + u64::from(values[3]),
                _ => unreachable!(),
            };
            assert_eq!(actual, expected);
            kani::cover!(budget == 0);
            kani::cover!(budget == 1);
            kani::cover!(budget == 2);
        }
        kani::cover!(budget == usize::MAX && missing);
    }
}
