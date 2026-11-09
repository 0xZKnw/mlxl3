// Review probe: uses the actual planner, reservations, publications and LFU.
#![allow(dead_code)]
#[path = "/Users/justin/.codex/worktrees/pr161-speed-20261108/mix-stq1_0/native/src/flash_support.rs"]
mod flash_support;
#[path = "/Users/justin/.codex/worktrees/pr161-speed-20261108/mix-stq1_0/native/src/speculation_plan.rs"]
mod speculation_plan;

use flash_support::{DemandPlan, ExpertCache};
use std::{cell::RefCell, rc::Rc};

fn reserve_and_publish(cache: &Rc<RefCell<ExpertCache>>, ids: &[u32]) -> usize {
    let plan = DemandPlan::reserve(cache, ids).unwrap();
    let reads = plan.reads.len();
    for &lease in &plan.reads {
        assert!(plan.publish(lease));
    }
    drop(plan);
    assert!(cache.borrow().ready_slot_snapshot().iter().flatten().count() <= cache.borrow().capacity());
    reads
}

fn scenario(slots: usize, shift: u16, block: bool) -> (usize, usize, Vec<u32>) {
    let routes = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(|id| id + shift),
        [0, 1, 2, 3, 4, 10, 11, 12, 13, 14].map(|id| id + shift),
    ];
    let cache = Rc::new(RefCell::new(ExpertCache::new(slots).unwrap()));
    let mut reads = 0;
    if block {
        let resident = Vec::new();
        let plan = speculation_plan::resident_expert_cohorts(&routes, slots, &resident).unwrap();
        for wave in &plan.waves {
            let ids = plan.cohorts[wave.clone()].iter().map(|entry| u32::from(entry.expert)).collect::<Vec<_>>();
            reads += reserve_and_publish(&cache, &ids);
        }
    } else {
        for row in routes {
            reads += reserve_and_publish(&cache, &row.map(u32::from));
        }
    }
    let mut resident = cache.borrow().ready_slot_snapshot().iter().flatten().map(|&(id, _)| id).collect::<Vec<_>>();
    resident.sort_unstable();
    let next_reads = reserve_and_publish(&cache, &routes[1].map(u32::from));
    (reads, next_reads, resident)
}

#[test]
fn reported_lfu_retention_counterexample_is_real() {
    for shift in [0, 1, 255, 497] {
        let scalar = scenario(11, shift, false);
        let block = scenario(11, shift, true);
        assert_eq!(scalar.0, 15);
        assert_eq!(block.0, 15);
        assert_eq!(scalar.1, 0);
        assert_eq!(block.1, 4);
        assert_eq!(block.2, (4 + u32::from(shift)..=14 + u32::from(shift)).collect::<Vec<_>>());
        println!("slots=11 shift={shift} scalar={scalar:?} block={block:?}");
    }
    for slots in [15, 20, 53, 90] {
        for block in [false, true] {
            let result = scenario(slots, 0, block);
            assert_eq!((result.0, result.1), (15, 0));
            println!("slots={slots} block={block} result={result:?}");
        }
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(18)]
fn lfu_counterexample_shifted_ids() {
    let shift = u16::from(kani::any::<u8>());
    let scalar = scenario(11, shift, false);
    let block = scenario(11, shift, true);
    assert_eq!((scalar.0, block.0, scalar.1, block.1), (15, 15, 0, 4));
    kani::cover!(shift == 0);
    kani::cover!(shift == 255);
}
