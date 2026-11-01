//! Reconstruct the physical AR trace before comparing shared demand-only caches.
use super::{Policy, ReplayPool};
use crate::flash_support::{EXPERT_BYTES, ExpertCache};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{fs, path::Path};

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    ids: [u32; 10],
    slots_hit: [(usize, bool); 10],
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Phase {
    name: String,
    before: Vec<[u64; 3]>,
    after_prefill: Vec<[u64; 3]>,
    after: Vec<[u64; 3]>,
    prefill: Vec<Observation>,
    decode: Vec<Observation>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Case {
    name: String,
    prompt_tokens: usize,
    phases: Vec<Phase>,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Capture {
    status: String,
    initial_generations: usize,
    cases: Vec<Case>,
}

impl Capture {
    fn validate(&self) -> Result<()> {
        ensure!(self.status == "complete_ar_routes" && self.initial_generations == 2);
        ensure!(self.cases.len() == 2);
        for (case, name) in self.cases.iter().zip(["MAIN", "HOLD"]) {
            ensure!(case.name == name && (1..=512).contains(&case.prompt_tokens));
            ensure!(case.phases.len() == 3);
            for (phase, name) in case.phases.iter().zip(["trace_1", "trace_2", "trace_3"]) {
                ensure!(phase.name == name);
                ensure!(phase.prefill.len() == 48 * case.prompt_tokens);
                ensure!(phase.decode.len() == 48 * 127);
                for counts in [&phase.before, &phase.after_prefill, &phase.after] {
                    ensure!(counts.len() == 48);
                }
                for obs in phase.prefill.iter().chain(&phase.decode) {
                    ensure!(obs.ids.iter().all(|id| *id < 512));
                    let mut sorted = obs.ids;
                    sorted.sort_unstable();
                    ensure!(sorted.windows(2).all(|w| w[0] != w[1]));
                    ensure!(obs.slots_hit.iter().all(|(slot, _)| *slot < 53));
                }
            }
        }
        Ok(())
    }
}

fn observe(
    cache: &mut ExpertCache,
    observed: &Observation,
    counters: &mut [u64; 3],
    check: bool,
) -> Result<()> {
    let mapped = cache.reserve(&observed.ids)?;
    ensure!(
        !check || mapped == observed.slots_hit,
        "physical slot/hit mismatch"
    );
    let misses = mapped.iter().filter(|(_, hit)| !hit).count() as u64;
    for (total, delta) in counters
        .iter_mut()
        .zip([10 - misses, misses, misses * EXPERT_BYTES])
    {
        *total = total
            .checked_add(delta)
            .ok_or_else(|| anyhow::anyhow!("counter overflow"))?;
    }
    Ok(())
}

fn reconstruct(case: &Case) -> Result<()> {
    let mut caches = (0..48)
        .map(|_| ExpertCache::new(53))
        .collect::<Result<Vec<_>, _>>()?;
    let mut counters = vec![[0u64; 3]; 48];
    for _ in 0..2 {
        for part in [&case.phases[0].prefill, &case.phases[0].decode] {
            for (i, observed) in part.iter().enumerate() {
                observe(&mut caches[i % 48], observed, &mut counters[i % 48], false)?;
            }
        }
    }
    for phase in &case.phases {
        ensure!(
            counters == phase.before,
            "initial/cumulative cache state mismatch"
        );
        for (part, expected) in [
            (&phase.prefill, &phase.after_prefill),
            (&phase.decode, &phase.after),
        ] {
            for (i, observed) in part.iter().enumerate() {
                observe(&mut caches[i % 48], observed, &mut counters[i % 48], true)?;
            }
            ensure!(counters == *expected, "phase/layer cache counters mismatch");
        }
    }
    Ok(())
}

fn analyze(capture: &Capture) -> Result<serde_json::Value> {
    capture.validate()?;
    // No candidate is scored when even one observed physical slot fails reconstruction.
    for case in &capture.cases {
        reconstruct(case)?;
    }
    let mut variants = Vec::new();
    for (policy, floor, local) in [
        (Policy::Lfu, 0, false),
        (Policy::Lfu, 20, false),
        (Policy::Clock, 0, false),
        (Policy::Clock, 20, false),
        (Policy::Clock, 0, true),
    ] {
        let mut cases = Vec::new();
        let mut eligible = !local;
        for case in &capture.cases {
            let mut pools = (0..if local { 48 } else { 1 })
                .map(|_| {
                    ReplayPool::new(
                        if local { 1 } else { 48 },
                        if local { 53 } else { 2544 },
                        floor,
                        policy,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            let mut phases = Vec::new();
            for generation in 0usize..5 {
                let phase = &case.phases[generation.saturating_sub(2)];
                let mut jobs = [0u64; 2];
                for (part, observations) in [&phase.prefill, &phase.decode].into_iter().enumerate()
                {
                    for (i, observed) in observations.iter().enumerate() {
                        let layer = i % 48;
                        let pool = &mut pools[if local { layer } else { 0 }];
                        let mapped = pool.reserve(if local { 0 } else { layer }, &observed.ids)?;
                        jobs[part] += mapped.iter().filter(|(_, hit)| !hit).count() as u64;
                    }
                }
                if generation >= 2 {
                    let baseline: [u64; 2] = std::array::from_fn(|part| {
                        let (before, after) = if part == 0 {
                            (&phase.before, &phase.after_prefill)
                        } else {
                            (&phase.after_prefill, &phase.after)
                        };
                        before.iter().zip(after).map(|(b, a)| a[1] - b[1]).sum()
                    });
                    let reductions = std::array::from_fn::<_, 2, _>(|part| {
                        if baseline[part] == 0 {
                            0.
                        } else {
                            100. * (baseline[part] as f64 - jobs[part] as f64)
                                / baseline[part] as f64
                        }
                    });
                    eligible &= reductions[1] >= 1.;
                    phases.push(serde_json::json!({"phase":phase.name,"baseline_prefill_decode_jobs":baseline,"candidate_prefill_decode_jobs":jobs,"read_reduction_percent":reductions}));
                }
            }
            let resident = if local {
                pools.iter().map(|p| p.resident[0]).collect()
            } else {
                pools[0].resident.clone()
            };
            ensure!(
                resident.iter().sum::<usize>() <= 2544,
                "wrong-size final pool"
            );
            cases.push(serde_json::json!({"prompt":case.name,"phases":phases,"slots_by_layer_after":resident}));
        }
        variants.push(serde_json::json!({"policy":format!("{policy:?}"),"floor":floor,"local_53_control":local,"eligible_for_runtime_trial":eligible,"cases":cases}));
    }
    Ok(
        serde_json::json!({"status":"complete_simulation_not_speed","baseline_counts_and_every_slot_hit":"exact_all_three_phases_both_prompts","initial_state":"empty; two reconstructed full generations before three observed phases","total_slots":2544,"expert_bank_bytes":2544*EXPERT_BYTES,"in_flight_gpu_leases":"not modeled; demand-only drained consumers","confirmed_tps_gain":false,"variants":variants}),
    )
}

fn write_report(input: &Path, output: &Path) -> Result<()> {
    write_report_with(input, output, analyze)
}

fn write_report_with(
    input: &Path,
    output: &Path,
    analyzer: fn(&Capture) -> Result<serde_json::Value>,
) -> Result<()> {
    serde_json::to_writer(
        fs::File::create_new(output)?,
        &serde_json::json!({"status":"incomplete","confirmed_tps_gain":false}),
    )?;
    let result = (|| {
        ensure!(
            (1..=32 << 20).contains(&fs::metadata(input)?.len()),
            "invalid capture size"
        );
        let capture: Capture = serde_json::from_slice(&fs::read(input)?)?;
        analyzer(&capture)
    })();
    let report = match &result {
        Ok(value) => value.clone(),
        Err(error) => {
            serde_json::json!({"status":"failed","confirmed_tps_gain":false,"error":format!("{error:#}")})
        }
    };
    fs::write(output, serde_json::to_vec_pretty(&report)?)?;
    result.map(|_| ())
}

fn analyze_arc(capture: &Capture) -> Result<serde_json::Value> {
    capture.validate()?;
    for case in &capture.cases {
        reconstruct(case)?;
    }
    let mut cases = Vec::new();
    let mut eligible = true;
    for case in &capture.cases {
        let mut cache = super::arc::ReplayArc::new(48, 2544)?;
        let mut phases = Vec::new();
        for generation in 0usize..5 {
            let phase = &case.phases[generation.saturating_sub(2)];
            let mut jobs = [0u64; 2];
            for (part, observations) in [&phase.prefill, &phase.decode].into_iter().enumerate() {
                for (i, observed) in observations.iter().enumerate() {
                    jobs[part] += cache
                        .reserve(i % 48, &observed.ids)?
                        .iter()
                        .filter(|(_, hit)| !hit)
                        .count() as u64;
                }
            }
            if generation >= 2 {
                let baseline: [u64; 2] = std::array::from_fn(|part| {
                    let (before, after) = if part == 0 {
                        (&phase.before, &phase.after_prefill)
                    } else {
                        (&phase.after_prefill, &phase.after)
                    };
                    before.iter().zip(after).map(|(b, a)| a[1] - b[1]).sum()
                });
                let reductions = std::array::from_fn::<_, 2, _>(|part| {
                    if baseline[part] == 0 {
                        0.
                    } else {
                        100. * (baseline[part] as f64 - jobs[part] as f64) / baseline[part] as f64
                    }
                });
                eligible &= reductions[1] >= 1.;
                phases.push(serde_json::json!({"phase":phase.name,"baseline_prefill_decode_jobs":baseline,"candidate_prefill_decode_jobs":jobs,"read_reduction_percent":reductions,"target_recent":cache.target,"t1_t2_b1_b2_lengths":cache.sizes()}));
            }
        }
        let mut resident = [0usize; 48];
        for key in cache.slots.iter().flatten() {
            resident[key.layer] += 1;
        }
        cases.push(serde_json::json!({"prompt":case.name,"phases":phases,"slots_by_layer_after":resident.to_vec(),"metadata_allocated_bytes_after":cache.metadata_allocation_bytes()}));
    }
    Ok(
        serde_json::json!({"status":"complete_simulation_not_speed","baseline_counts_and_every_slot_hit":"exact_all_three_phases_both_prompts","initial_state":"empty; two reconstructed full generations before three observed phases","policy":"ARC_global_complete_batch_protection","primary_source":"https://www.usenix.org/legacy/event/fast03/tech/full_papers/megiddo/megiddo.pdf#page=10","total_slots":2544,"expert_bank_bytes":2544*EXPERT_BYTES,"ghost_expert_payload_bytes":0,"eligible_for_runtime_trial":eligible,"in_flight_gpu_leases":"not modeled; demand-only drained consumers","confirmed_tps_gain":false,"cases":cases}),
    )
}

#[test]
#[ignore = "CPU-only ARC on current AR observations; requires fresh report"]
fn current_ar_traces_compare_arc_after_exact_physical_reconstruction() -> Result<()> {
    write_report_with(
        Path::new(&std::env::var("MLXL3_CURRENT_POOL_CAPTURE")?),
        Path::new(&std::env::var("MLXL3_CURRENT_POOL_REPORT")?),
        analyze_arc,
    )
}

#[test]
#[ignore = "CPU-only current AR pool replay; requires physical observations and fresh report"]
fn current_ar_traces_reconstruct_exact_cache_before_pool_scoring() -> Result<()> {
    write_report(
        Path::new(&std::env::var("MLXL3_CURRENT_POOL_CAPTURE")?),
        Path::new(&std::env::var("MLXL3_CURRENT_POOL_REPORT")?),
    )
}

#[test]
fn observations_reject_wrong_slots_hits_and_counter_overflow() -> Result<()> {
    for fault in 0..4 {
        let mut cache = ExpertCache::new(53)?;
        let mut observed = Observation {
            ids: std::array::from_fn(|i| i as u32),
            slots_hit: std::array::from_fn(|i| (i, false)),
        };
        let mut counters = [0; 3];
        match fault {
            0 => observed.slots_hit[0].0 = 1,
            1 => observed.slots_hit[0].1 = true,
            2 => counters[1] = u64::MAX,
            _ => (),
        }
        assert_eq!(
            observe(&mut cache, &observed, &mut counters, true).is_ok(),
            fault == 3
        );
        if fault == 3 {
            assert_eq!(counters, [0, 10, 10 * EXPERT_BYTES]);
        }
    }
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("input.json");
    for (i, text) in [
        "{",
        "{}",
        "{\"status\":\"incomplete\",\"initial_generations\":2,\"cases\":[]}",
    ]
    .into_iter()
    .enumerate()
    {
        fs::write(&input, text)?;
        let output = temp.path().join(i.to_string());
        assert!(write_report(&input, &output).is_err());
        let original = fs::read(&output)?;
        let value: serde_json::Value = serde_json::from_slice(&original)?;
        assert_eq!(value["status"], "failed");
        assert_eq!(value["confirmed_tps_gain"], false);
        assert!(write_report(&input, &output).is_err());
        assert_eq!(fs::read(&output)?, original);
    }
    Ok(())
}

fn fixture() -> Capture {
    let observed = Observation {
        ids: std::array::from_fn(|i| i as u32),
        slots_hit: std::array::from_fn(|i| (i, true)),
    };
    Capture {
        status: "complete_ar_routes".into(),
        initial_generations: 2,
        cases: ["MAIN", "HOLD"]
            .map(|name| Case {
                name: name.into(),
                prompt_tokens: 1,
                phases: (0..3)
                    .map(|phase| {
                        let hits = 1280 * (2 + phase as u64) - 10;
                        Phase {
                            name: format!("trace_{}", phase + 1),
                            before: vec![[hits, 10, 10 * EXPERT_BYTES]; 48],
                            after_prefill: vec![[hits + 10, 10, 10 * EXPERT_BYTES]; 48],
                            after: vec![[hits + 1280, 10, 10 * EXPERT_BYTES]; 48],
                            prefill: vec![observed.clone(); 48],
                            decode: vec![observed.clone(); 48 * 127],
                        }
                    })
                    .collect(),
            })
            .to_vec(),
    }
}

#[test]
fn reconstruction_checks_all_observations_counts_and_shapes_before_reporting() -> Result<()> {
    let valid = fixture();
    valid.validate()?;
    for case in &valid.cases {
        reconstruct(case)?;
    }
    for fault in 0..13 {
        let mut capture = valid.clone();
        match fault {
            0 => capture.status = "incomplete".into(),
            1 => capture.initial_generations = 3,
            2 => capture.cases[1].name = "MAIN".into(),
            3 => capture.cases[0].prompt_tokens = 513,
            4 => {
                capture.cases[0].phases.pop();
            }
            5 => capture.cases[0].phases[0].name = "trace_2".into(),
            6 => {
                capture.cases[0].phases[0].prefill.pop();
            }
            7 => {
                capture.cases[0].phases[0].before.pop();
            }
            8 => capture.cases[0].phases[0].decode[0].ids[0] = 512,
            9 => capture.cases[0].phases[0].decode[0].ids[0] = 1,
            10 => capture.cases[0].phases[0].decode[0].slots_hit[0].0 = 53,
            11 => capture.cases[1].phases[2].after[47][1] += 1,
            _ => capture.cases[1].phases[2].decode[48 * 127 - 1].slots_hit[0].1 = false,
        }
        assert!(analyze(&capture).is_err(), "accepted fault {fault}");
        assert!(analyze_arc(&capture).is_err(), "ARC accepted fault {fault}");
    }
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("trace.json");
    let output = temp.path().join("report.json");
    fs::write(&input, serde_json::to_vec(&valid)?)?;
    write_report(&input, &output)?;
    let bytes = fs::read(&output)?;
    let report: serde_json::Value = serde_json::from_slice(&bytes)?;
    assert_eq!(report["status"], "complete_simulation_not_speed");
    assert_eq!(report["confirmed_tps_gain"], false);
    let variants = report["variants"].as_array().unwrap();
    assert_eq!(variants.len(), 5);
    for variant in variants {
        assert_eq!(variant["eligible_for_runtime_trial"], false);
        for case in variant["cases"].as_array().unwrap() {
            let resident = case["slots_by_layer_after"].as_array().unwrap();
            assert_eq!(resident.len(), 48);
            assert!(resident.iter().all(|n| n == 10));
            for phase in case["phases"].as_array().unwrap() {
                assert_eq!(phase["read_reduction_percent"], serde_json::json!([0., 0.]));
                assert_eq!(
                    phase["baseline_prefill_decode_jobs"],
                    serde_json::json!([0, 0])
                );
                assert_eq!(
                    phase["candidate_prefill_decode_jobs"],
                    serde_json::json!([0, 0])
                );
            }
        }
    }
    let arc = analyze_arc(&valid)?;
    assert_eq!(arc["eligible_for_runtime_trial"], false);
    assert_eq!(arc["ghost_expert_payload_bytes"], 0);
    assert_eq!(arc["expert_bank_bytes"], 2544 * EXPERT_BYTES);
    for case in arc["cases"].as_array().unwrap() {
        assert!(
            case["slots_by_layer_after"]
                .as_array()
                .unwrap()
                .iter()
                .all(|n| n == 10)
        );
        for phase in case["phases"].as_array().unwrap() {
            assert_eq!(
                phase["candidate_prefill_decode_jobs"],
                serde_json::json!([0, 0])
            );
            assert_eq!(
                phase["t1_t2_b1_b2_lengths"],
                serde_json::json!([0, 480, 0, 0])
            );
        }
    }
    assert!(write_report(&input, &output).is_err());
    assert_eq!(fs::read(&output)?, bytes);
    // A mismatch must leave an explicit failed report, never a partial validation.
    let mut invalid = valid;
    invalid.cases[1].phases[2].after[47][1] += 1;
    fs::write(&input, serde_json::to_vec(&invalid)?)?;
    let failed = temp.path().join("failed.json");
    assert!(write_report(&input, &failed).is_err());
    let report: serde_json::Value = serde_json::from_slice(&fs::read(failed)?)?;
    assert_eq!(report["status"], "failed");
    assert_eq!(report["confirmed_tps_gain"], false);
    assert!(report.get("variants").is_none());
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(55)]
fn physical_reconstruction_cannot_accept_corrupt_first_slot_or_hit() {
    let mut cache = ExpertCache::new(53).unwrap();
    let slot: usize = kani::any();
    let hit: bool = kani::any();
    let mut observed = Observation {
        ids: std::array::from_fn(|i| i as u32),
        slots_hit: std::array::from_fn(|i| (i, false)),
    };
    observed.slots_hit[0] = (slot, hit);
    let mut counters = [0; 3];
    let accepted = observe(&mut cache, &observed, &mut counters, true).is_ok();
    assert!(accepted == (slot == 0 && !hit));
    if accepted {
        assert!(counters == [0, 10, 10 * EXPERT_BYTES]);
    }
    kani::cover!(accepted);
    kani::cover!(!accepted && slot >= 53);
    kani::cover!(!accepted && slot == 0 && hit);
}
