//! Per-layer 50/50 probation/protected SLRU; cache leases stay in ExpertCache.
#[derive(Debug)]
pub(super) struct Slru {
    slots: usize,
    probation: Vec<usize>,
    protected: Vec<usize>,
    #[cfg(test)]
    reservations: usize,
}
impl Slru {
    pub(super) fn new(slots: usize) -> Self {
        Self {
            slots,
            probation: (0..slots).collect(),
            protected: Vec::with_capacity(slots / 2 + 1),
            #[cfg(test)]
            reservations: 0,
        }
    }
    pub(super) fn clear(&mut self) {
        self.probation.clear();
        self.probation.extend(0..self.slots);
        self.protected.clear();
    }
    pub(super) fn ordered(&self) -> impl Iterator<Item = usize> + '_ {
        self.probation.iter().chain(&self.protected).copied()
    }
    pub(super) fn touch(&mut self, slot: usize, hit: bool) -> bool {
        if slot >= self.slots {
            return false;
        }
        self.probation.retain(|&i| i != slot);
        self.protected.retain(|&i| i != slot);
        if hit {
            self.protected.push(slot);
            if self.protected.len() > self.slots / 2 {
                let oldest = self.protected.remove(0);
                self.probation.push(oldest);
            }
        } else {
            self.probation.push(slot);
        }
        true
    }
    pub(super) fn record_reservation(&mut self) {
        #[cfg(test)]
        {
            self.reservations += 1;
        }
    }
    #[cfg(test)]
    pub(super) fn reservations(&self) -> usize {
        self.reservations
    }
    #[cfg(any(test, kani))]
    fn valid(&self) -> bool {
        self.probation.len() + self.protected.len() == self.slots
            && self.protected.len() <= self.slots / 2
            && (0..self.slots).all(|slot| self.ordered().filter(|&s| s == slot).count() == 1)
    }
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(40)]
fn slru_four_actual_promotions_preserve_every_slot_and_protected_capacity() {
    let capacity = 10 + usize::from(kani::any::<bool>());
    let mut slru = Slru::new(capacity);
    for _ in 0..4 {
        let slot: u8 = kani::any();
        let hit: bool = kani::any();
        assert_eq!(
            slru.touch(usize::from(slot), hit),
            usize::from(slot) < capacity
        );
        assert!(slru.valid());
    }
    let clear: bool = kani::any();
    if clear {
        slru.clear();
    }
    assert!(slru.valid());
    kani::cover!(capacity == 10 && clear);
    kani::cover!(capacity == 11 && !slru.protected.is_empty());
    kani::cover!(slru.protected.is_empty());
}

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;
    struct Oracle {
        keys: Vec<Option<u32>>,
        protected: Vec<bool>,
        times: Vec<usize>,
        tick: usize,
    }
    impl Oracle {
        fn new(n: usize) -> Self {
            Self {
                keys: vec![None; n],
                protected: vec![false; n],
                times: (0..n).collect(),
                tick: n,
            }
        }
        fn touch(&mut self, slot: usize, hit: bool) {
            self.tick += 1;
            self.times[slot] = self.tick;
            self.protected[slot] = hit;
            if self.protected.iter().filter(|&&p| p).count() > self.keys.len() / 2 {
                let old = (0..self.keys.len())
                    .filter(|&i| self.protected[i])
                    .min_by_key(|&i| self.times[i])
                    .unwrap();
                self.tick += 1;
                self.times[old] = self.tick;
                self.protected[old] = false;
            }
        }
        fn reserve(&mut self, ids: &[u32]) -> Vec<(usize, bool)> {
            let mut unique = ids.to_vec();
            unique.sort_unstable();
            unique.dedup();
            let mut touched = Vec::new();
            for &id in ids {
                if !touched.contains(&id) {
                    touched.push(id);
                    if let Some(slot) = self.keys.iter().position(|&k| k == Some(id)) {
                        self.touch(slot, true);
                    }
                }
            }
            let mut mapped = Vec::new();
            for id in unique {
                let present = self.keys.iter().position(|&k| k == Some(id));
                let slot = present.unwrap_or_else(|| {
                    (0..self.keys.len())
                        .filter(|&i| self.keys[i].is_none_or(|k| !ids.contains(&k)))
                        .min_by_key(|&i| {
                            (self.keys[i].is_some(), self.protected[i], self.times[i], i)
                        })
                        .unwrap()
                });
                if present.is_none() {
                    self.keys[slot] = Some(id);
                    self.touch(slot, false);
                }
                mapped.push((id, slot, present.is_some()));
            }
            ids.iter()
                .map(|id| {
                    let &(_, s, h) = mapped.iter().find(|(k, _, _)| k == id).unwrap();
                    (s, h)
                })
                .collect()
        }
    }
    #[test]
    fn slru_matches_independent_timestamp_oracle_with_churn_and_duplicates() -> Result<()> {
        for capacity in [10, 11, 53] {
            let mut cache = ExpertCache::new(capacity)?;
            cache.enable_slru();
            let mut oracle = Oracle::new(capacity);
            for row in 0..1200 {
                let ids = (0..10)
                    .map(|i| {
                        if i < 4 {
                            (i + (row / 200) * 17) % 512
                        } else {
                            (row * 13 + i * 23) % 512
                        }
                    })
                    .map(|x| x as u32)
                    .collect::<Vec<_>>();
                assert_eq!(cache.reserve(&ids)?, oracle.reserve(&ids));
                assert!(cache.slru.as_ref().unwrap().valid());
            }
            assert_eq!(cache.slru_reservations(), 1200);
            cache.invalidate();
            assert!(cache.slru_enabled() && cache.keys.iter().all(Option::is_none));
            assert!(cache.slru.as_ref().unwrap().valid());
            cache.enable_predictive_lru();
            assert!(!cache.slru_enabled());
        }
        Ok(())
    }
    #[test]
    fn slru_keeps_pending_generations_and_recovers_after_abandonment() -> Result<()> {
        use std::{cell::RefCell, rc::Rc};
        let mut cache = ExpertCache::new(10)?;
        cache.enable_slru();
        let cache = Rc::new(RefCell::new(cache));
        let old = DemandPlan::reserve(&cache, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])?;
        assert_eq!(old.reads.len(), 10);
        assert!(
            cache
                .borrow()
                .resident_slot_map()?
                .iter()
                .all(|&x| x == u32::MAX)
        );
        assert!(DemandPlan::reserve(&cache, &[10]).is_err());
        drop(old);
        let next = DemandPlan::reserve(&cache, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])?;
        for &lease in &next.reads {
            assert!(next.publish(lease));
        }
        assert!(cache.borrow().slru.as_ref().unwrap().valid());
        drop(next);
        let pending = DemandPlan::reserve(&cache, &[10, 11])?;
        let lease = pending.reads[0];
        drop(pending);
        let current = DemandPlan::reserve(&cache, &[10, 11])?;
        assert!(!cache.borrow_mut().finish_demand(lease, true));
        for &lease in &current.reads {
            assert!(current.publish(lease));
        }
        drop(current);
        assert_eq!(cache.borrow().available_slots(), 10);
        cache.borrow_mut().invalidate();
        assert!(cache.borrow().slru.as_ref().unwrap().valid());
        Ok(())
    }
}
