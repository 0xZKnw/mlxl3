//! Test-only host intervals. Fenced samples include dispatch and waiting, not exclusive GPU time.
use anyhow::{Result, ensure};
use serde::Serialize;
use serde_json::Value;
use std::{cell::RefCell, time::Instant};

#[derive(Serialize)]
pub(crate) struct Event {
    name: String,
    parent: Option<usize>,
    start_ns: u64,
    end_ns: u64,
    data: Value,
}
struct Trace {
    epoch: Instant,
    fenced: bool,
    stack: Vec<usize>,
    events: Vec<Event>,
}
thread_local! {
    static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
}
pub(crate) struct Span(usize);
pub(crate) fn start(fenced: bool) -> Result<()> {
    TRACE.with_borrow_mut(|trace| {
        ensure!(trace.is_none(), "profile already active");
        *trace = Some(Trace {
            epoch: Instant::now(),
            fenced,
            stack: Vec::new(),
            events: Vec::new(),
        });
        Ok(())
    })
}
pub(crate) fn span(name: &str) -> Option<Span> {
    TRACE.with_borrow_mut(|trace| {
        let trace = trace.as_mut()?;
        let index = trace.events.len();
        trace.events.push(Event {
            name: name.into(),
            parent: trace.stack.last().copied(),
            start_ns: trace
                .epoch
                .elapsed()
                .as_nanos()
                .try_into()
                .expect("profile duration fits u64"),
            end_ns: 0,
            data: Value::Null,
        });
        trace.stack.push(index);
        Some(Span(index))
    })
}
impl Span {
    pub(crate) fn data(&self, data: Value) {
        TRACE.with_borrow_mut(|trace| {
            trace.as_mut().expect("active span").events[self.0].data = data
        });
    }
}
impl Drop for Span {
    fn drop(&mut self) {
        TRACE.with_borrow_mut(|trace| {
            let trace = trace.as_mut().expect("active span");
            assert_eq!(
                trace.stack.pop(),
                Some(self.0),
                "profile spans must be nested"
            );
            trace.events[self.0].end_ns = trace
                .epoch
                .elapsed()
                .as_nanos()
                .try_into()
                .expect("profile duration fits u64");
        });
    }
}
pub(crate) fn finish() -> Result<Vec<Event>> {
    TRACE.with_borrow_mut(|trace| {
        ensure!(
            trace.as_ref().is_some_and(|t| t.stack.is_empty()),
            "profile inactive or spans open"
        );
        Ok(trace.take().expect("validated trace").events)
    })
}
#[cfg(feature = "mlx")]
pub(crate) fn stage(name: &str, value: &crate::array::Array) -> Result<()> {
    let fenced = TRACE.with_borrow(|trace| trace.as_ref().is_some_and(|t| t.fenced));
    if fenced {
        let event = span(name).expect("active fenced trace");
        value.eval()?;
        event.data(serde_json::json!({"kind":"completion_wall", "shape":value.shape(), "dtype":format!("{:?}",value.dtype())}));
    }
    Ok(())
}
pub(crate) fn fenced() -> bool {
    TRACE.with_borrow(|trace| trace.as_ref().is_some_and(|t| t.fenced))
}

#[test]
fn nested_intervals_are_complete_and_disabled_is_a_noop() -> Result<()> {
    ensure!(span("disabled").is_none());
    start(false)?;
    ensure!(start(true).is_err());
    {
        let outer = span("outer").unwrap();
        ensure!(finish().is_err());
        {
            let inner = span("inner").unwrap();
            inner.data(serde_json::json!({"rows":2}));
        }
        outer.data(serde_json::json!({"rows":1}));
    }
    let events = finish()?;
    ensure!(events.len() == 2 && events[0].parent.is_none() && events[1].parent == Some(0));
    ensure!(events[0].start_ns <= events[1].start_ns && events[1].end_ns <= events[0].end_ns);
    ensure!(events[1].end_ns >= events[1].start_ns && events[1].data["rows"] == 2);
    ensure!(finish().is_err() && span("disabled").is_none());
    Ok(())
}

#[cfg(not(feature = "mlx"))]
#[test]
fn modes_remain_thread_local() -> Result<()> {
    start(true)?;
    ensure!(fenced());
    std::thread::spawn(|| {
        assert!(!fenced());
        assert!(span("other thread").is_none());
    })
    .join()
    .unwrap();
    finish()?;
    ensure!(!fenced());
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn real_profile_scope_lifecycle() {
    nested_intervals_are_complete_and_disabled_is_a_noop().unwrap();
}
