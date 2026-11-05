//! Offline shared-slot screening; no GPU buffers or in-flight leases are modeled.
mod arc;
mod current;
use super::CacheTrace;
use crate::flash_support::{CacheError, EXPERT_BYTES};
use anyhow::{Result, ensure};
use std::{fs, path::Path};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Policy {
    Lfu,
    Clock,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    layer: usize,
    expert: u32,
}

fn aged_frequency(count: u32, source_epoch: u64, epoch: u64) -> u32 {
    count
        .checked_shr(epoch.saturating_sub(source_epoch).min(32) as u32)
        .unwrap_or(0)
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReplayPool {
    keys: Vec<Option<Key>>,
    counts: Vec<u32>,
    ages: Vec<u64>,
    referenced: Vec<bool>,
    resident: Vec<usize>,
    layer_clocks: Vec<u64>,
    clock: u64,
    hand: usize,
    floor: usize,
    policy: Policy,
}

impl ReplayPool {
    fn new(layers: usize, slots: usize, floor: usize, policy: Policy) -> Result<Self, CacheError> {
        if !(1..=48).contains(&layers)
            || slots == 0
            || slots > layers * 512
            || floor > 512
            || floor * layers > slots
        {
            return Err(CacheError::Capacity);
        }
        Ok(Self {
            keys: vec![None; slots],
            counts: vec![0; layers * 512],
            ages: vec![0; slots],
            referenced: vec![false; slots],
            resident: vec![0; layers],
            layer_clocks: vec![0; layers],
            clock: 0,
            hand: 0,
            floor,
            policy,
        })
    }

    fn reserve(&mut self, layer: usize, ids: &[u32]) -> Result<Vec<(usize, bool)>, CacheError> {
        if layer >= self.resident.len()
            || ids.is_empty()
            || ids.len() > 80
            || ids.iter().any(|&id| id >= 512)
        {
            return Err(CacheError::InvalidIds);
        }
        let mut unique = Vec::new();
        for &id in ids {
            let p = unique
                .iter()
                .position(|&key| key >= id)
                .unwrap_or(unique.len());
            if unique.get(p) != Some(&id) {
                unique.insert(p, id);
            }
        }
        // Room for the entire batch plus every other layer's minimum is guaranteed.
        if unique.len() + self.floor * (self.resident.len() - 1) > self.keys.len() {
            return Err(CacheError::Capacity);
        }
        let clock = self.clock.checked_add(1).ok_or(CacheError::Clock)?;
        let layer_clock = self.layer_clocks[layer]
            .checked_add(1)
            .ok_or(CacheError::Clock)?;
        self.clock = clock;
        self.layer_clocks[layer] = layer_clock;
        if layer_clock.is_multiple_of(16) {
            for count in &mut self.counts[layer * 512..(layer + 1) * 512] {
                *count /= 2;
            }
        }
        let epoch = self.layer_clocks.iter().copied().max().unwrap_or(0) / 16;
        let eligible = |key: Option<Key>, resident: &[usize]| {
            key.is_none_or(|k| {
                !(k.layer == layer && unique.contains(&k.expert))
                    && (k.layer == layer || resident[k.layer] > self.floor)
            })
        };
        let mut mapped = Vec::with_capacity(unique.len());
        for &expert in &unique {
            let key = Key { layer, expert };
            let present = self.keys.iter().position(|&k| k == Some(key));
            let slot = if let Some(slot) = present {
                slot
            } else if self.policy == Policy::Lfu {
                (0..self.keys.len())
                    .filter(|&i| eligible(self.keys[i], &self.resident))
                    .min_by_key(|&i| {
                        let count = self.keys[i].map_or(0, |k| {
                            aged_frequency(
                                self.counts[k.layer * 512 + k.expert as usize],
                                self.layer_clocks[k.layer] / 16,
                                epoch,
                            )
                        });
                        (self.keys[i].is_some(), count, self.ages[i], i)
                    })
                    .ok_or(CacheError::NoSlot)?
            } else {
                let mut chosen = None;
                for _ in 0..2 * self.keys.len() {
                    let i = self.hand;
                    self.hand = (self.hand + 1) % self.keys.len();
                    if eligible(self.keys[i], &self.resident) {
                        if self.keys[i].is_none() || !self.referenced[i] {
                            chosen = Some(i);
                            break;
                        }
                        self.referenced[i] = false;
                    }
                }
                chosen.ok_or(CacheError::NoSlot)?
            };
            if present.is_none() {
                if let Some(old) = self.keys[slot] {
                    self.resident[old.layer] -= 1;
                }
                self.resident[layer] += 1;
            }
            self.keys[slot] = Some(key);
            self.ages[slot] = clock;
            self.referenced[slot] = true;
            let count = &mut self.counts[layer * 512 + expert as usize];
            *count = count.saturating_add(1);
            mapped.push((expert, slot, present.is_some()));
        }
        Ok(ids
            .iter()
            .map(|id| {
                let &(_, slot, hit) = mapped
                    .iter()
                    .find(|&&(key, _, _)| key == *id)
                    .expect("reserved key");
                (slot, hit)
            })
            .collect())
    }
}

fn analyze(trace: &CacheTrace) -> Result<serde_json::Value> {
    trace.validate()?;
    let baseline = (0..48)
        .map(|layer| trace.replay_layer(layer, 53))
        .collect::<Result<Vec<_>>>()?;
    for (layer, phases) in baseline.iter().enumerate() {
        for (phase, counts) in phases.iter().enumerate() {
            ensure!(
                counts[0] == trace.phases[phase].prefill_counts[layer]
                    && counts[1] == trace.phases[phase].decode_counts[layer],
                "historical baseline mismatch phase={phase} layer={layer}"
            );
        }
    }
    let baseline_jobs = trace.phases[1]
        .decode_counts
        .iter()
        .map(|c| c[1])
        .sum::<u64>();
    ensure!(baseline_jobs > 0, "empty miss denominator");
    let mut cases = Vec::new();
    for policy in [Policy::Lfu, Policy::Clock] {
        for floor in [0, 20] {
            let mut pool = ReplayPool::new(48, 2544, floor, policy)?;
            let mut phases = Vec::new();
            for phase in &trace.phases {
                let mut counts = vec![vec![[0u64; 3]; 48]; 2];
                for (step, layers) in phase.routes.iter().enumerate() {
                    for (layer, ids) in layers.iter().enumerate() {
                        let mapped = pool.reserve(layer, ids)?;
                        let misses = mapped.iter().filter(|(_, hit)| !hit).count() as u64;
                        let c = &mut counts[usize::from(step >= trace.prompt_ids.len())][layer];
                        c[0] += 10 - misses;
                        c[1] += misses;
                        c[2] += misses * EXPERT_BYTES;
                    }
                }
                phases.push(counts);
            }
            let jobs = phases[1][1].iter().map(|c| c[1]).sum::<u64>();
            let reduction = 100. * (baseline_jobs as f64 - jobs as f64) / baseline_jobs as f64;
            cases.push(serde_json::json!({"policy":format!("{policy:?}"),"floor":floor,"phases_prefill_decode_layers":phases,"second_decode_jobs":jobs,"read_reduction_percent":reduction,"eligible_for_current_trace_validation":reduction>=1.0,"slots_by_layer_after":pool.resident}));
        }
    }
    Ok(
        serde_json::json!({"status":"complete_simulation_not_speed","baseline_counts":"exact_both_phases_all_layers","initial_state":"cold; two full generations, cache and frequencies retained","model":"historical scalar calibration47; current AR/MTP/DFlash not qualified","total_slots":2544,"expert_bank_bytes":2544*EXPERT_BYTES,"baseline_second_decode_jobs":baseline_jobs,"in_flight_gpu_leases":"not modeled; no runtime port validated","confirmed_tps_gain":false,"cases":cases}),
    )
}

fn write_report(input: &Path, output: &Path) -> Result<()> {
    serde_json::to_writer(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(output)?,
        &serde_json::json!({"status":"incomplete","confirmed_tps_gain":false}),
    )?;
    let result = (|| {
        ensure!(
            (1..=8 << 20).contains(&fs::metadata(input)?.len()),
            "invalid trace size"
        );
        let trace: CacheTrace = serde_json::from_slice(&fs::read(input)?)?;
        analyze(&trace)
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

#[test]
#[ignore = "offline calibration47 replay; CPU only, explicit input/output required"]
fn replay_exact_calibration_with_shared_slots() -> Result<()> {
    write_report(
        Path::new(&std::env::var("MLXL3_POOL_TRACE")?),
        Path::new(&std::env::var("MLXL3_POOL_REPORT")?),
    )
}

#[test]
fn shared_pool_preserves_labels_floors_batches_errors_and_layer_aging() -> Result<()> {
    for policy in [Policy::Lfu, Policy::Clock] {
        let mut pool = ReplayPool::new(2, 4, 1, policy)?;
        assert_eq!(pool.reserve(0, &[0, 1])?, vec![(0, false), (1, false)]);
        assert_eq!(pool.reserve(0, &[0])?, vec![(0, true)]);
        assert_eq!(pool.reserve(1, &[0, 1])?, vec![(2, false), (3, false)]);
        let slot = pool.reserve(1, &[2])?[0].0;
        assert_eq!(slot, if policy == Policy::Lfu { 1 } else { 0 });
        assert_eq!(pool.resident, vec![1, 3]);
        for id in 3..20 {
            pool.reserve(1, &[id])?;
        }
        assert_eq!(pool.resident, vec![1, 3]);
        for (layer, ids) in [
            (2, vec![0]),
            (0, vec![]),
            (0, vec![512]),
            (0, vec![0; 81]),
            (0, vec![0, 1, 2, 3]),
        ] {
            let saved = pool.clone();
            assert!(pool.reserve(layer, &ids).is_err());
            assert_eq!(pool, saved);
        }
        let mut pool = ReplayPool::new(1, 2, 0, policy)?;
        pool.reserve(0, &[3, 4])?;
        let values = pool.reserve(0, &[2, 3, 2])?;
        assert!(!values[0].1 && values[1].1 && !values[2].1);
        assert_eq!(values[0], values[2]);
        assert_ne!(values[0].0, values[1].0);
        pool.clock = u64::MAX;
        let saved = pool.clone();
        assert_eq!(pool.reserve(0, &[0]), Err(CacheError::Clock));
        assert_eq!(pool, saved);
    }
    let mut pool = ReplayPool::new(2, 3, 1, Policy::Lfu)?;
    for _ in 0..16 {
        pool.reserve(0, &[0])?;
    }
    pool.reserve(1, &[0])?;
    assert_eq!(pool.counts[0], 8);
    assert_eq!(
        (pool.layer_clocks[0], pool.layer_clocks[1], pool.clock),
        (16, 1, 17)
    );
    for (layers, slots, floor) in [
        (0, 1, 0),
        (49, 1, 0),
        (1, 0, 0),
        (1, 513, 0),
        (2, 3, 2),
        (1, 1, usize::MAX),
    ] {
        assert!(ReplayPool::new(layers, slots, floor, Policy::Lfu).is_err());
    }
    Ok(())
}

#[test]
fn global_lfu_compares_frequencies_at_the_same_decay_epoch() -> Result<()> {
    let mut pool = ReplayPool::new(2, 2, 0, Policy::Lfu)?;
    for _ in 0..15 {
        pool.reserve(1, &[0])?;
    }
    for _ in 0..15 {
        pool.reserve(0, &[0])?;
    }
    let selected = pool.reserve(0, &[1])?;
    assert_eq!(
        selected,
        vec![(0, false)],
        "equally frequent older L1 should be evicted"
    );
    assert!(pool.keys.contains(&Some(Key {
        layer: 0,
        expert: 0
    })));
    assert!(!pool.keys.contains(&Some(Key {
        layer: 1,
        expert: 0
    })));
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn global_frequency_aging_matches_wide_arithmetic_for_all_counters() {
    let count: u32 = kani::any();
    let source_epoch: u64 = kani::any();
    let epoch: u64 = kani::any();
    let shift = epoch.saturating_sub(source_epoch).min(32);
    let actual = aged_frequency(count, source_epoch, epoch);
    let expected = u64::from(count) / (1u64 << shift);
    assert!(u64::from(actual) == expected && actual <= count);
    kani::cover!(shift == 32 && count > 0);
    kani::cover!(shift == 0 && count == u32::MAX);
    kani::cover!(shift == 31 && count == u32::MAX);
    kani::cover!(epoch < source_epoch && count > 0);
}

#[test]
fn shared_pool_streams_match_independent_membership_and_capacity_invariants() -> Result<()> {
    for policy in [Policy::Lfu, Policy::Clock] {
        for floor in [0, 2] {
            for seed in 0..16u32 {
                let mut pool = ReplayPool::new(4, 16, floor, policy)?;
                let mut random = seed + 1;
                for step in 0..128 {
                    let layer = step % 4;
                    let mut ids = Vec::new();
                    for _ in 0..1 + step % 5 {
                        random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                        ids.push(random % 23);
                    }
                    let expected_hits = ids
                        .iter()
                        .map(|&expert| pool.keys.contains(&Some(Key { layer, expert })))
                        .collect::<Vec<_>>();
                    let old_counts = pool.resident.clone();
                    let mapped = pool.reserve(layer, &ids)?;
                    for (i, (&expert, &(slot, hit))) in ids.iter().zip(&mapped).enumerate() {
                        assert_eq!(hit, expected_hits[i]);
                        assert_eq!(pool.keys[slot], Some(Key { layer, expert }));
                    }
                    let keys = pool
                        .keys
                        .iter()
                        .flatten()
                        .map(|k| (k.layer, k.expert))
                        .collect::<std::collections::BTreeSet<_>>();
                    assert_eq!(keys.len(), pool.keys.iter().flatten().count());
                    assert!(keys.len() <= 16);
                    for (layer, &old) in old_counts.iter().enumerate() {
                        assert_eq!(
                            pool.resident[layer],
                            keys.iter().filter(|k| k.0 == layer).count()
                        );
                        assert!(pool.resident[layer] >= old.min(floor));
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn clock_victims_match_independent_rotating_queue_with_complete_demand_pins() -> Result<()> {
    use std::collections::VecDeque;
    for seed in 0..32u32 {
        let mut pool = ReplayPool::new(4, 16, 0, Policy::Clock)?;
        let mut queue = (0..16)
            .map(|slot| (slot, None::<Key>, false))
            .collect::<VecDeque<_>>();
        let mut random = seed + 1;
        for step in 0..256 {
            let layer = step % 4;
            let ids = (0..1 + step % 8)
                .map(|_| {
                    random = random.wrapping_mul(1664525).wrapping_add(1013904223);
                    random % 23
                })
                .collect::<Vec<_>>();
            let mut unique = ids.clone();
            unique.sort_unstable();
            unique.dedup();
            for &expert in &unique {
                let key = Key { layer, expert };
                if let Some(entry) = queue.iter_mut().find(|e| e.1 == Some(key)) {
                    entry.2 = true;
                    continue;
                }
                let mut selected = false;
                for _ in 0..32 {
                    let mut entry = queue.pop_front().unwrap();
                    if entry
                        .1
                        .is_some_and(|k| k.layer == layer && unique.contains(&k.expert))
                    {
                        queue.push_back(entry);
                        continue;
                    }
                    if entry.2 {
                        entry.2 = false;
                        queue.push_back(entry);
                        continue;
                    }
                    entry.1 = Some(key);
                    entry.2 = true;
                    queue.push_back(entry);
                    selected = true;
                    break;
                }
                assert!(selected);
            }
            pool.reserve(layer, &ids)?;
            assert_eq!(pool.hand, queue.front().unwrap().0);
            for &(slot, key, referenced) in &queue {
                assert_eq!(pool.keys[slot], key);
                assert_eq!(pool.referenced[slot], referenced);
            }
        }
    }
    Ok(())
}

#[test]
fn shared_report_rejects_invalid_incomplete_input_and_preserves_existing_files() -> Result<()> {
    let temp = tempfile::tempdir()?;
    let input = temp.path().join("input.json");
    for (name, text) in [
        ("truncated", "{"),
        ("empty", "{}"),
        ("incomplete", "{\"status\":\"incomplete\"}"),
    ] {
        fs::write(&input, text)?;
        let output = temp.path().join(name);
        assert!(write_report(&input, &output).is_err());
        let bytes = fs::read(&output)?;
        let value: serde_json::Value = serde_json::from_slice(&bytes)?;
        ensure!(value["status"] == "failed" && value["confirmed_tps_gain"] == false);
        assert!(write_report(&input, &output).is_err());
        assert_eq!(fs::read(&output)?, bytes);
    }
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(8)]
fn pooled_reservations_keep_batch_slots_distinct_and_bounded() {
    let floor = usize::from(kani::any::<bool>());
    let policy = if kani::any::<bool>() {
        Policy::Lfu
    } else {
        Policy::Clock
    };
    let mut pool = ReplayPool::new(2, 3, floor, policy).unwrap();
    let a: u16 = kani::any();
    let b: u16 = kani::any();
    let layer = usize::from(kani::any::<bool>());
    let saved = pool.clone();
    let first = pool.reserve(layer, &[u32::from(a)]);
    assert!(first.is_ok() == (a < 512));
    if a >= 512 {
        let index = usize::from(kani::any::<u16>()) % 1024;
        assert!(pool.counts[index] == saved.counts[index]);
        assert!(pool.keys == saved.keys && pool.ages == saved.ages);
        assert!(pool.referenced == saved.referenced && pool.resident == saved.resident);
        assert!(pool.layer_clocks == saved.layer_clocks && pool.clock == saved.clock);
        assert!(
            pool.hand == saved.hand && pool.floor == saved.floor && pool.policy == saved.policy
        );
    }
    let ids = [u32::from(a % 512), u32::from(b % 512)];
    for current in [0, 1, layer] {
        let mapped = pool.reserve(current, &ids).unwrap();
        assert!(mapped.len() == 2 && mapped[0].0 < 3 && mapped[1].0 < 3);
        assert!((mapped[0].0 == mapped[1].0) == (ids[0] == ids[1]));
        for i in 0..2 {
            assert!(
                pool.keys[mapped[i].0]
                    == Some(Key {
                        layer: current,
                        expert: ids[i]
                    })
            );
        }
        assert!(pool.resident.iter().sum::<usize>() <= 3);
        kani::cover!(mapped[0].1);
        kani::cover!(!mapped[0].1 && floor == 1);
    }
    kani::cover!(a >= 512);
}
