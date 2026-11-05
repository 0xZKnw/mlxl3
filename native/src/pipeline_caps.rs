//! Immutable pipeline ownership keeps capability snapshots tied to their PSO.
use std::ops::Deref;

// Values must stay immutable for the retained object's lifetime (as a Metal PSO does).
pub(crate) trait Properties {
    fn thread_execution_width(&self) -> u64;
    fn max_total_threads_per_threadgroup(&self) -> u64;
}

#[derive(Clone)]
pub(crate) struct Pipeline<P> {
    inner: P,
    cached: Option<[u64; 2]>,
}

impl<P: Properties> Pipeline<P> {
    pub(crate) fn new(inner: P, enabled: bool) -> Self {
        let cached = enabled.then(|| {
            #[cfg(test)]
            {
                record(2);
                record(3);
            }
            [
                inner.thread_execution_width(),
                inner.max_total_threads_per_threadgroup(),
            ]
        });
        Self { inner, cached }
    }

    pub(crate) fn thread_execution_width(&self) -> u64 {
        match self.cached {
            Some([width, _]) => {
                #[cfg(test)]
                record(0);
                width
            }
            None => {
                #[cfg(test)]
                record(2);
                self.inner.thread_execution_width()
            }
        }
    }

    pub(crate) fn max_total_threads_per_threadgroup(&self) -> u64 {
        match self.cached {
            Some([_, max]) => {
                #[cfg(test)]
                record(1);
                max
            }
            None => {
                #[cfg(test)]
                record(3);
                self.inner.max_total_threads_per_threadgroup()
            }
        }
    }
}

#[cfg(test)]
thread_local! {
    static READS: std::cell::Cell<[u64; 4]> = const { std::cell::Cell::new([0; 4]) };
}

#[cfg(test)]
fn record(index: usize) {
    READS.with(|reads| {
        let mut value = reads.get();
        value[index] = value[index].saturating_add(1);
        reads.set(value);
    });
}

/// [cached width, cached maximum, live width, live maximum], current thread only.
#[cfg(test)]
pub(crate) fn stats() -> [u64; 4] {
    READS.with(std::cell::Cell::get)
}

impl<P> Deref for Pipeline<P> {
    type Target = P;

    fn deref(&self) -> &P {
        &self.inner
    }
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
impl Properties for metal::ComputePipelineState {
    fn thread_execution_width(&self) -> u64 {
        metal::ComputePipelineStateRef::thread_execution_width(self)
    }

    fn max_total_threads_per_threadgroup(&self) -> u64 {
        metal::ComputePipelineStateRef::max_total_threads_per_threadgroup(self)
    }
}

#[cfg(test)]
mod tests {
    use super::{Pipeline, Properties, stats};
    use std::{cell::Cell, ops::Deref, rc::Rc};

    #[derive(Clone)]
    struct Observed {
        values: [u64; 2],
        calls: Rc<[Cell<u32>; 2]>,
    }

    impl Properties for Observed {
        fn thread_execution_width(&self) -> u64 {
            self.calls[0].set(self.calls[0].get() + 1);
            self.values[0]
        }

        fn max_total_threads_per_threadgroup(&self) -> u64 {
            self.calls[1].set(self.calls[1].get() + 1);
            self.values[1]
        }
    }

    fn observed(values: [u64; 2]) -> Observed {
        Observed {
            values,
            calls: Rc::new([Cell::new(0), Cell::new(0)]),
        }
    }

    #[test]
    fn disabled_pipeline_queries_remain_lazy_and_per_access() {
        let raw = observed([32, 640]);
        let calls = raw.calls.clone();
        let pipeline = Pipeline::new(raw, false);
        assert_eq!([calls[0].get(), calls[1].get()], [0, 0]);
        assert_eq!(pipeline.thread_execution_width(), 32);
        assert_eq!(pipeline.thread_execution_width(), 32);
        assert_eq!([calls[0].get(), calls[1].get()], [2, 0]);
        assert_eq!(pipeline.max_total_threads_per_threadgroup(), 640);
        assert_eq!([calls[0].get(), calls[1].get()], [2, 1]);
    }

    #[test]
    fn enabled_pipeline_and_clone_read_each_owned_capability_only_once() {
        let raw = observed([32, 1024]);
        let calls = raw.calls.clone();
        let pipeline = Pipeline::new(raw, true);
        assert_eq!([calls[0].get(), calls[1].get()], [1, 1]);
        let before = stats();
        for p in [&pipeline, &pipeline.clone()] {
            assert_eq!(p.thread_execution_width(), 32);
            assert_eq!(p.max_total_threads_per_threadgroup(), 1024);
            assert_eq!(p.deref().values, [32, 1024]);
        }
        assert_eq!([calls[0].get(), calls[1].get()], [1, 1]);
        let after = stats();
        assert_eq!(after[0] - before[0], 2);
        assert_eq!(after[1] - before[1], 2);
        assert_eq!(&after[2..], &before[2..]);
    }

    #[test]
    fn capability_boundaries_keep_existing_rejection_and_dynamic_thresholds() {
        for enabled in [false, true] {
            for (values, required, accepted) in [
                ([32, 640], 640, true),
                ([32, 639], 640, false),
                ([31, 1024], 640, false),
                ([64, 1024], 640, false),
                ([0, 0], 32, false),
                ([32, u64::MAX], u64::MAX, true),
                ([32, u64::MAX - 1], u64::MAX, false),
                ([u64::MAX, u64::MAX], 32, false),
            ] {
                let pipeline = Pipeline::new(observed(values), enabled);
                assert_eq!(
                    pipeline.thread_execution_width() == 32
                        && pipeline.max_total_threads_per_threadgroup() >= required,
                    accepted
                );
                assert_eq!(pipeline.deref().values, values);
            }
        }
    }

    #[test]
    fn counter_deltas_distinguish_construction_cached_access_and_live_access() {
        for enabled in [false, true] {
            let before = stats();
            let pipeline = Pipeline::new(observed([32, 640]), enabled);
            let constructed = stats();
            let initial_queries = u64::from(enabled);
            assert_eq!(constructed[0..2], before[0..2]);
            assert_eq!(constructed[2] - before[2], initial_queries);
            assert_eq!(constructed[3] - before[3], initial_queries);
            let clone = pipeline.clone();
            assert_eq!(stats(), constructed);
            assert_eq!(clone.thread_execution_width(), 32);
            assert_eq!(clone.max_total_threads_per_threadgroup(), 640);
            let after = stats();
            assert_eq!(after[0] - constructed[0], u64::from(enabled));
            assert_eq!(after[1] - constructed[1], u64::from(enabled));
            assert_eq!(after[2] - constructed[2], u64::from(!enabled));
            assert_eq!(after[3] - constructed[3], u64::from(!enabled));
        }
    }
}

#[cfg(kani)]
mod proofs {
    use super::{Pipeline, Properties};

    #[derive(Clone, Copy)]
    struct Immutable([u64; 2]);

    impl Properties for Immutable {
        fn thread_execution_width(&self) -> u64 {
            self.0[0]
        }

        fn max_total_threads_per_threadgroup(&self) -> u64 {
            self.0[1]
        }
    }

    #[kani::proof]
    fn cached_pipeline_preserves_full_width_and_dynamic_capability_guards() {
        let enabled: bool = kani::any();
        let values: [u64; 2] = kani::any();
        let required_width: u64 = kani::any();
        let required_threads: u64 = kani::any();
        let pipeline = Pipeline::new(Immutable(values), enabled);
        let clone = pipeline.clone();
        assert_eq!(pipeline.thread_execution_width(), values[0]);
        assert_eq!(pipeline.max_total_threads_per_threadgroup(), values[1]);
        assert_eq!(clone.thread_execution_width(), values[0]);
        assert_eq!(clone.max_total_threads_per_threadgroup(), values[1]);
        let accepted = pipeline.thread_execution_width() == required_width
            && pipeline.max_total_threads_per_threadgroup() >= required_threads;
        if accepted {
            assert_eq!(values[0], required_width);
            assert!(values[1] >= required_threads);
        } else {
            assert!(values[0] != required_width || values[1] < required_threads);
        }
        kani::cover!(enabled && accepted && required_threads == 640);
        kani::cover!(!enabled && accepted && required_threads == 1024);
        kani::cover!(enabled && !accepted && values[1] == 639 && required_threads == 640);
        kani::cover!(enabled && values[0] == u64::MAX && values[1] == u64::MAX);
        kani::cover!(!enabled && values == [0, 0] && !accepted);
    }
}
