//! Offline replay from cold, then warm; not reconstruction of the original LFU state.

fn missing_jobs(mapped: &[(usize, bool)]) -> Option<usize> {
    let mut seen = [false; 512];
    let mut jobs = 0;
    for &(slot, hit) in mapped {
        let seen = seen.get_mut(slot)?;
        if !hit && !*seen {
            *seen = true;
            jobs += 1;
        }
    }
    Some(jobs)
}

#[cfg(test)]
mod tests {
    use super::missing_jobs;
    use crate::flash_support::ExpertCache;
    use anyhow::{Context, Result, ensure};
    use serde::Deserialize;
    use std::{collections::BTreeSet, fs, path::Path};

    #[derive(Clone, Deserialize, serde::Serialize)]
    struct Layer {
        prefill: Vec<Vec<u32>>,
        decode: Vec<Vec<u32>>,
    }

    #[derive(Clone, Deserialize, serde::Serialize)]
    struct Case {
        name: String,
        layers: Vec<Layer>,
    }

    fn validate(cases: &[Case]) -> Result<()> {
        ensure!(cases.len() == 4);
        let mut names = BTreeSet::new();
        for case in cases {
            ensure!(["OFF-MAIN", "MTP-MAIN", "OFF-HOLD", "MTP-HOLD"].contains(&case.name.as_str()));
            ensure!(names.insert(case.name.as_str()) && case.layers.len() == 48);
            for layer in &case.layers {
                ensure!(!layer.prefill.is_empty() && !layer.decode.is_empty());
                ensure!(layer.prefill.iter().all(|ids| ids.len() == 10));
                ensure!(
                    !case.name.starts_with("OFF") || layer.decode.iter().all(|ids| ids.len() == 10)
                );
                ensure!(layer.prefill.len() == case.layers[0].prefill.len());
                ensure!(layer.decode.len() == case.layers[0].decode.len());
                for ids in layer.prefill.iter().chain(&layer.decode) {
                    ensure!([10, 20].contains(&ids.len()) && ids.iter().all(|&id| id < 512));
                    ensure!(
                        ids.chunks(10)
                            .all(|row| row.iter().collect::<BTreeSet<_>>().len() == 10)
                    );
                }
            }
        }
        Ok(())
    }

    fn replay(layer: &Layer, slots: usize) -> Result<u32> {
        let mut cache = ExpertCache::new(slots)?;
        let mut jobs = 0usize;
        for generation in 0..2 {
            for (decode, cohorts) in [(false, &layer.prefill), (true, &layer.decode)] {
                for ids in cohorts {
                    let mapped = cache.reserve(ids)?;
                    if generation == 1 && decode {
                        jobs = jobs
                            .checked_add(missing_jobs(&mapped).context("invalid cache slot")?)
                            .context("job overflow")?;
                    }
                }
            }
        }
        Ok(u32::try_from(jobs)?)
    }

    fn allocate(cases: &[Case]) -> Result<serde_json::Value> {
        validate(cases)?;
        let mut all_curves = Vec::new();
        let mut training = vec![vec![0u32; 124]; 48];
        for case in cases {
            let curves = case
                .layers
                .iter()
                .map(|layer| {
                    (20..=512)
                        .step_by(4)
                        .map(|slots| replay(layer, slots))
                        .collect::<Result<Vec<_>>>()
                })
                .collect::<Result<Vec<_>>>()?;
            if case.name.ends_with("MAIN") {
                for (total, curve) in training.iter_mut().zip(&curves) {
                    for (a, b) in total.iter_mut().zip(curve) {
                        *a = a.checked_add(*b).context("cost overflow")?;
                    }
                }
            }
            all_curves.push(curves);
        }
        let levels = super::super::allocate_levels(&training, (48 * 53 - 48 * 20) / 4)
            .context("invalid allocation")?;
        let slots = levels.iter().map(|q| 20 + q * 4).collect::<Vec<_>>();
        ensure!(slots.iter().sum::<usize>() == 2544);
        let mut eligible = true;
        let mut results = Vec::new();
        for (case, curves) in cases.iter().zip(all_curves) {
            let baseline = case
                .layers
                .iter()
                .map(|l| replay(l, 53))
                .collect::<Result<Vec<_>>>()?;
            let selected = levels
                .iter()
                .enumerate()
                .map(|(i, &q)| curves[i][q])
                .collect::<Vec<_>>();
            let before = baseline.iter().map(|&n| u64::from(n)).sum::<u64>();
            let after = selected.iter().map(|&n| u64::from(n)).sum::<u64>();
            let reduction = if before > 0 {
                100. * (before as f64 - after as f64) / before as f64
            } else {
                0.
            };
            eligible &= reduction >= 1.;
            results.push(serde_json::json!({"name":case.name,"training":case.name.ends_with("MAIN"),"baseline_jobs":before,"selected_jobs":after,"reduction_percent":reduction,"baseline_by_layer":baseline,"selected_by_layer":selected,"curves":curves}));
        }
        Ok(
            serde_json::json!({"status":"complete_prediction_not_speed","initial_state":"cold; replay two complete generations; score second decode","original_profile53_hit_parity":"not asserted: initial LFU keys/frequencies unavailable","total_slots":2544,"selected_slots":slots,"cases":results,"eligible_for_runtime_trial":eligible,"confirmed_tps_gain":false}),
        )
    }

    fn write_report(input: &Path, output: &Path) -> Result<()> {
        serde_json::to_writer(
            fs::File::create_new(output)?,
            &serde_json::json!({"status":"incomplete"}),
        )?;
        let result = (|| {
            ensure!((1..=16 << 20).contains(&fs::metadata(input)?.len()));
            let cases: Vec<Case> = serde_json::from_slice(&fs::read(input)?)?;
            allocate(&cases)
        })();
        let value = match &result {
            Ok(value) => value.clone(),
            Err(error) => {
                serde_json::json!({"status":"failed","error":format!("{error:#}"),"confirmed_tps_gain":false})
            }
        };
        fs::write(output, serde_json::to_vec_pretty(&value)?)?;
        result.map(|_| ())
    }

    #[test]
    fn real_cache_replay_counts_unique_jobs_and_warm_generations() -> Result<()> {
        let mapped = [(1, false), (1, false), (2, true), (3, false), (2, false)];
        assert_eq!(missing_jobs(&mapped), Some(3));
        assert_eq!(missing_jobs(&[]), Some(0));
        for invalid in [512, usize::MAX] {
            for hit in [false, true] {
                assert_eq!(missing_jobs(&[(invalid, hit)]), None);
            }
        }
        for seed in 0..256usize {
            let entries = (0..20)
                .map(|i| ((seed * 37 + i * i * 11) % 512, (seed + i) % 3 == 0))
                .collect::<Vec<_>>();
            let expected = entries
                .iter()
                .filter(|(_, hit)| !hit)
                .map(|(slot, _)| *slot)
                .collect::<BTreeSet<_>>()
                .len();
            assert_eq!(missing_jobs(&entries), Some(expected));
        }
        let layer = Layer {
            prefill: vec![(0..10).collect()],
            decode: vec![(10..20).collect(), (10..20).chain(10..20).collect()],
        };
        assert_eq!(replay(&layer, 20)?, 0);
        assert_eq!(replay(&layer, 10)?, 10);
        let directory = tempfile::tempdir()?;
        let input = directory.path().join("input.json");
        let output = directory.path().join("out.json");
        fs::write(&input, "[")?;
        assert!(write_report(&input, &output).is_err());
        let previous = fs::read(&output)?;
        assert_eq!(
            serde_json::from_slice::<serde_json::Value>(&previous)?["status"],
            "failed"
        );
        assert!(write_report(&input, &output).is_err());
        assert_eq!(fs::read(&output)?, previous);
        for fault in [0, 1, 2, 3, 4, 5] {
            let mut cases = ["OFF-MAIN", "MTP-MAIN", "OFF-HOLD", "MTP-HOLD"].map(|name| Case {
                name: name.into(),
                layers: vec![layer.clone(); 48],
            });
            for case in &mut cases {
                if case.name.starts_with("OFF") {
                    for layer in &mut case.layers {
                        layer.decode.retain(|ids| ids.len() == 10);
                    }
                }
            }
            validate(&cases)?;
            match fault {
                0 => cases[0].layers.pop(),
                1 => {
                    cases[0].layers[0].decode.clear();
                    None
                }
                2 => {
                    cases[0].layers[0].prefill[0][0] = 512;
                    None
                }
                3 => {
                    cases[0].layers[0].decode[0][1] = 10;
                    None
                }
                4 => {
                    cases[0].name = "OFF-HOLD".into();
                    None
                }
                _ => {
                    cases[0].layers[0].prefill[0].pop();
                    None
                }
            };
            assert!(validate(&cases).is_err(), "accepted invalid trace {fault}");
        }
        Ok(())
    }

    #[test]
    fn held_out_routes_cannot_fit_the_profile_and_successful_files_are_preserved() -> Result<()> {
        let layer = Layer {
            prefill: vec![(0..10).collect()],
            decode: vec![(10..20).collect()],
        };
        let mut cases = ["OFF-MAIN", "MTP-MAIN", "OFF-HOLD", "MTP-HOLD"].map(|name| Case {
            name: name.into(),
            layers: vec![layer.clone(); 48],
        });
        let original = allocate(&cases)?;
        assert_eq!(original["eligible_for_runtime_trial"], false);
        for layer in &mut cases[2].layers {
            layer.decode = vec![(10..20).collect(); 8];
        }
        // Only one held-out layer has a larger union; training must stay untouched.
        cases[2].layers[0].decode = (0..8)
            .map(|i| (20 + i * 10..30 + i * 10).collect())
            .collect();
        let changed = allocate(&cases)?;
        assert_eq!(original["selected_slots"], changed["selected_slots"]);
        assert!(changed["cases"][2]["baseline_jobs"].as_u64().unwrap() > 0);
        let directory = tempfile::tempdir()?;
        let input = directory.path().join("routes.json");
        let output = directory.path().join("report.json");
        fs::write(&input, serde_json::to_vec(&cases)?)?;
        write_report(&input, &output)?;
        let bytes = fs::read(&output)?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        assert_eq!(value["status"], "complete_prediction_not_speed");
        assert_eq!(
            value["selected_slots"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| n.as_u64().unwrap())
                .sum::<u64>(),
            2544
        );
        assert!(write_report(&input, &output).is_err());
        assert_eq!(bytes, fs::read(&output)?);
        Ok(())
    }

    #[test]
    #[ignore = "CPU-only experiment55, requires validated53 routes and fresh output"]
    fn replay_profile53_routes_with_constant_budget() -> Result<()> {
        write_report(
            Path::new(&std::env::var("MLXL3_CACHE53_ROUTES")?),
            Path::new(&std::env::var("MLXL3_CACHE53_OUTPUT")?),
        )?;
        println!("FLASH_CACHE53 complete_prediction_not_speed total_slots=2544");
        Ok(())
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(22)]
fn profiled_missing_jobs_match_independent_slot_bitset() {
    let mapped: [(usize, bool); 20] = kani::any();
    let len: usize = kani::any();
    kani::assume(len <= 20);
    let mut bits = [0u64; 8];
    for &(slot, hit) in &mapped[..len] {
        kani::assume(slot < 512);
        if !hit {
            bits[slot / 64] |= 1u64 << (slot % 64);
        }
    }
    let expected = bits.iter().map(|b| b.count_ones() as usize).sum::<usize>();
    let actual = missing_jobs(&mapped[..len]);
    assert_eq!(actual, Some(expected));
    kani::cover!(len == 20 && actual == Some(20));
    kani::cover!(len == 20 && actual == Some(1));
    kani::cover!(len == 0 && actual == Some(0));
}
