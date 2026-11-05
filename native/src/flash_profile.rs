//! Test-only host intervals. Fenced samples include dispatch and waiting, not exclusive GPU time.
use anyhow::{Result, ensure};
use serde::Serialize;
use serde_json::Value;
use std::{
    cell::{Cell, RefCell},
    marker::PhantomData,
    rc::Rc,
    time::Instant,
};

#[derive(Serialize)]
pub(crate) struct Event {
    name: String,
    parent: Option<usize>,
    start_ns: u64,
    end_ns: u64,
    data: Value,
    allocation_requests: u64,
    allocation_requested_bytes: u64,
    allocation_api_ns: u64,
}
struct Trace {
    epoch: Instant,
    fenced: bool,
    stack: Vec<usize>,
    events: Vec<Event>,
    signatures: Vec<KernelSignature>,
}
#[derive(PartialEq)]
struct KernelSignature {
    stable: bool,
    name: String,
    header: String,
    source: String,
}
fn intern_signature(
    signatures: &mut Vec<KernelSignature>,
    stable: bool,
    name: &str,
    header: &str,
    source: &str,
) -> (usize, bool) {
    if let Some(index) = signatures.iter().position(|key| {
        key.stable == stable && key.name == name && key.header == header && key.source == source
    }) {
        return (index, false);
    }
    let index = signatures.len();
    signatures.push(KernelSignature {
        stable,
        name: name.into(),
        header: header.into(),
        source: source.into(),
    });
    (index, true)
}

// A request builds a lazy graph; only a GPU capture establishes actual dispatches.
pub(crate) fn kernel_signature(stable: bool, name: &str, header: &str, source: &str) {
    let key = TRACE.with_borrow_mut(|trace| {
        trace
            .as_mut()
            .map(|trace| intern_signature(&mut trace.signatures, stable, name, header, source))
    });
    if let Some((index, fresh)) = key {
        if fresh {
            span("expert_kernel_signature")
                .expect("active trace")
                .data(serde_json::json!({"id":index,"stable":stable,"key":[name,header,source]}));
        }
        span("expert_kernel_request")
            .expect("active trace")
            .data(serde_json::json!({"signature":index}));
    }
}
thread_local! {
    static TRACE: RefCell<Option<Trace>> = const { RefCell::new(None) };
    static RETAIN_TEST_STAGES: Cell<bool> = const { Cell::new(true) };
}
pub(crate) struct Span(usize);

pub(crate) fn retain_test_stages() -> bool {
    RETAIN_TEST_STAGES.get()
}

pub(crate) struct StageRetention {
    previous: bool,
    _thread: PhantomData<Rc<()>>,
}
impl StageRetention {
    pub(crate) fn new(enabled: bool) -> Self {
        Self {
            previous: RETAIN_TEST_STAGES.replace(enabled),
            _thread: PhantomData,
        }
    }
}
impl Drop for StageRetention {
    fn drop(&mut self) {
        RETAIN_TEST_STAGES.set(self.previous);
    }
}

#[test]
fn stage_retention_is_scoped_restored_after_panic_and_thread_local() {
    assert!(retain_test_stages());
    {
        let _off = StageRetention::new(false);
        assert!(!retain_test_stages());
        {
            let _on = StageRetention::new(true);
            assert!(retain_test_stages());
        }
        assert!(!retain_test_stages());
        let mut retained = Vec::new();
        if retain_test_stages() {
            retained.push(1);
        }
        assert!(retained.is_empty());
        std::thread::spawn(|| {
            assert!(retain_test_stages());
            {
                let _off = StageRetention::new(false);
                assert!(!retain_test_stages());
            }
            assert!(retain_test_stages());
        })
        .join()
        .unwrap();
        assert!(!retain_test_stages());
    }
    assert!(retain_test_stages());
    let failure = std::panic::catch_unwind(|| {
        let _off = StageRetention::new(false);
        assert!(!retain_test_stages());
        panic!("disposable stage-retention regression");
    });
    assert!(failure.is_err() && retain_test_stages());
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(33)]
fn stage_retention_restores_symbolic_nested_scopes() {
    let outer: bool = kani::any();
    let inner: bool = kani::any();
    assert!(retain_test_stages());
    {
        let _outer = StageRetention::new(outer);
        assert!(retain_test_stages() == outer);
        {
            let _inner = StageRetention::new(inner);
            assert!(retain_test_stages() == inner);
        }
        assert!(retain_test_stages() == outer);
    }
    assert!(retain_test_stages());
    kani::cover!(!outer && !inner);
    kani::cover!(!outer && inner);
    kani::cover!(outer && !inner);
    kani::cover!(outer && inner);
}

#[derive(Debug, PartialEq, Serialize)]
pub(crate) struct ReadbackSample {
    pub(crate) wait_ns: u64,
    pub(crate) copy_ns: u64,
    pub(crate) contiguous_build_ns: u64,
    pub(crate) bytes: u64,
    pub(crate) ready_before: bool,
    pub(crate) direct: bool,
    pub(crate) total_ns: u64,
}
pub(crate) fn validate_readback_sample(raw: [u64; 8], bytes: usize) -> Option<ReadbackSample> {
    if raw[0] != 1 || raw[5] > 1 || raw[6] > 1 || u64::try_from(bytes).ok()? != raw[4] {
        return None;
    }
    let measured = raw[1].checked_add(raw[2])?.checked_add(raw[3])?;
    if measured > raw[7] || (raw[6] == 1 && raw[3] != 0) {
        return None;
    }
    Some(ReadbackSample {
        wait_ns: raw[1],
        copy_ns: raw[2],
        contiguous_build_ns: raw[3],
        bytes: raw[4],
        ready_before: raw[5] == 1,
        direct: raw[6] == 1,
        total_ns: raw[7],
    })
}

pub(crate) fn start(fenced: bool) -> Result<()> {
    TRACE.with_borrow_mut(|trace| {
        ensure!(trace.is_none(), "profile already active");
        *trace = Some(Trace {
            epoch: Instant::now(),
            fenced,
            stack: Vec::new(),
            events: Vec::new(),
            signatures: Vec::new(),
        });
        #[cfg(feature = "mlx")]
        crate::array::readback_profile_enable(true);
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
            allocation_requests: 0,
            allocation_requested_bytes: 0,
            allocation_api_ns: 0,
        });
        trace.stack.push(index);
        Some(Span(index))
    })
}
pub(crate) fn active() -> bool {
    TRACE.with_borrow(Option::is_some)
}

// Only the innermost event owns a request; ancestor scopes must not count it again.
pub(crate) fn native_allocation(bytes: u64, api_ns: u64) {
    TRACE.with_borrow_mut(|trace| {
        let Some(trace) = trace.as_mut() else { return };
        let Some(&index) = trace.stack.last() else {
            return;
        };
        let event = &mut trace.events[index];
        event.allocation_requests = event.allocation_requests.saturating_add(1);
        event.allocation_requested_bytes = event.allocation_requested_bytes.saturating_add(bytes);
        event.allocation_api_ns = event.allocation_api_ns.saturating_add(api_ns);
    });
}

#[test]
fn native_allocations_belong_only_to_the_innermost_active_scope() -> Result<()> {
    native_allocation(99, 99);
    start(false)?;
    native_allocation(99, 99);
    {
        let outer = span("outer").unwrap();
        native_allocation(10, 20);
        {
            let _inner = span("inner").unwrap();
            native_allocation(30, 40);
            native_allocation(50, 60);
        }
        outer.data(serde_json::json!({"preserved":true}));
        native_allocation(70, 80);
    }
    let events = finish()?;
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].data["preserved"], true);
    assert_eq!(
        (
            events[0].allocation_requests,
            events[0].allocation_requested_bytes,
            events[0].allocation_api_ns
        ),
        (2, 80, 100)
    );
    assert_eq!(
        (
            events[1].allocation_requests,
            events[1].allocation_requested_bytes,
            events[1].allocation_api_ns
        ),
        (2, 80, 100)
    );
    assert_eq!(events[1].parent, Some(0));
    assert!(!active());
    start(false)?;
    {
        let _scope = span("saturation").unwrap();
        native_allocation(u64::MAX, u64::MAX);
        native_allocation(1, 1);
    }
    let events = finish()?;
    assert_eq!(events[0].allocation_requests, 2);
    assert_eq!(events[0].allocation_requested_bytes, u64::MAX);
    assert_eq!(events[0].allocation_api_ns, u64::MAX);
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(33)]
fn native_allocation_attributes_two_symbolic_requests_exclusively() {
    let outer_bytes: u64 = kani::any();
    let outer_ns: u64 = kani::any();
    let inner_bytes: u64 = kani::any();
    let inner_ns: u64 = kani::any();
    start(false).unwrap();
    {
        let _outer = span("outer").unwrap();
        native_allocation(outer_bytes, outer_ns);
        {
            let _inner = span("inner").unwrap();
            native_allocation(inner_bytes, inner_ns);
        }
    }
    let events = finish().unwrap();
    assert_eq!(events.len(), 2);
    assert_eq!(events[0].allocation_requests, 1);
    assert_eq!(events[0].allocation_requested_bytes, outer_bytes);
    assert_eq!(events[0].allocation_api_ns, outer_ns);
    assert_eq!(events[1].allocation_requests, 1);
    assert_eq!(events[1].allocation_requested_bytes, inner_bytes);
    assert_eq!(events[1].allocation_api_ns, inner_ns);
    kani::cover!(outer_bytes == 0 && inner_bytes == u64::MAX);
    kani::cover!(outer_ns == u64::MAX && inner_ns == 0);
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
#[allow(
    unexpected_cfgs,
    reason = "objc 0.2 selector macro tests legacy cargo-clippy cfg"
)]
pub(crate) fn native_gpu_interval(command: &metal::CommandBufferRef, owner: &str) {
    use metal::foreign_types::ForeignTypeRef;
    use objc::{msg_send, sel, sel_impl};
    if let Some(event) = span("native.gpu_interval") {
        let start: f64 = unsafe { msg_send![command, GPUStartTime] };
        let end: f64 = unsafe { msg_send![command, GPUEndTime] };
        event.data(serde_json::json!({"owner":owner,"command_address":command.as_ptr() as usize,"gpu_start_seconds":start,"gpu_end_seconds":end,"valid":start.is_finite() && end.is_finite() && start>0.0 && end>=start,"clock":"MTLCommandBuffer host seconds; no mapping to CPU Instant epoch recorded"}));
    }
}

#[test]
fn native_observer_active_tracks_the_existing_lifecycle() -> Result<()> {
    assert!(!active());
    start(false)?;
    assert!(active());
    {
        let row = span("row").unwrap();
        row.data(serde_json::json!({"token":1}));
        let _child = span("head").unwrap();
    }
    let events = finish()?;
    assert!(!active() && events.len() == 2);
    assert!(events[0].parent.is_none() && events[1].parent == Some(0));
    assert!(events.iter().all(|e| e.end_ns >= e.start_ns));
    Ok(())
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
        let events = trace.take().expect("validated trace").events;
        #[cfg(feature = "mlx")]
        crate::array::readback_profile_enable(false);
        Ok(events)
    })
}

#[test]
fn signatures_are_unique_but_every_request_is_counted() -> Result<()> {
    kernel_signature(false, "disabled", "", "");
    start(false)?;
    let source = "source".repeat(2048);
    let root = span("root").unwrap();
    let keys = [
        (false, "name", "header", source.as_str()),
        (true, "name", "header", source.as_str()),
        (false, "other", "header", source.as_str()),
        (false, "name", "other", source.as_str()),
        (false, "name", "header", "other"),
    ];
    for &(stable, name, header, source) in &keys {
        kernel_signature(stable, name, header, source);
    }
    for _ in 0..500 {
        kernel_signature(false, "name", "header", &source);
    }
    drop(root);
    let events = finish()?;
    let signatures = events
        .iter()
        .filter(|e| e.name == "expert_kernel_signature")
        .collect::<Vec<_>>();
    let requests = events
        .iter()
        .filter(|e| e.name == "expert_kernel_request")
        .collect::<Vec<_>>();
    assert_eq!(signatures.len(), keys.len());
    assert_eq!(requests.len(), 505);
    for (index, event) in signatures.iter().enumerate() {
        assert_eq!(event.data["id"], index);
        assert_eq!(event.data["stable"], keys[index].0);
        assert_eq!(
            event.data["key"],
            serde_json::json!([keys[index].1, keys[index].2, keys[index].3])
        );
    }
    for (index, event) in requests.iter().enumerate() {
        assert_eq!(event.data["signature"], if index < 5 { index } else { 0 });
    }
    assert!(serde_json::to_vec(&events)?.len() < 100 * source.len());
    start(false)?;
    kernel_signature(false, "name", "header", &source);
    let events = finish()?;
    assert_eq!(events[0].data["id"], 0);
    Ok(())
}

#[test]
fn readback_sample_rejects_incomplete_corrupt_and_overflowing_records() -> Result<()> {
    assert_eq!(
        validate_readback_sample([1, 10, 5, 0, 40, 0, 1, 16], 40)
            .expect("valid direct sample")
            .bytes,
        40
    );
    assert!(
        !validate_readback_sample([1, 11, 5, 3, 40, 1, 0, 20], 40)
            .expect("valid fallback")
            .direct
    );
    assert!(validate_readback_sample([1, 0, 0, 0, 0, 1, 1, 0], 0).is_some());
    for raw in [
        [0, 10, 5, 0, 40, 0, 1, 16],
        [1, 10, 5, 0, 40, 2, 1, 16],
        [1, 10, 5, 0, 40, 0, 2, 16],
        [1, 10, 5, 0, 41, 0, 1, 16],
        [1, 10, 5, 0, 40, 0, 1, 14],
        [1, 10, 5, 1, 40, 0, 1, 16],
        [1, u64::MAX, 1, 0, 40, 0, 1, u64::MAX],
        [1, 0, u64::MAX, 1, 40, 0, 0, u64::MAX],
    ] {
        assert!(
            validate_readback_sample(raw, 40).is_none(),
            "accepted {raw:?}"
        );
    }
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn readback_profile_accepts_exactly_valid_bounded_durations() {
    let raw: [u64; 8] = kani::any();
    let bytes: usize = kani::any();
    let result = validate_readback_sample(raw, bytes);
    let total = u128::from(raw[1]) + u128::from(raw[2]) + u128::from(raw[3]);
    let expected = raw[0] == 1
        && raw[5] <= 1
        && raw[6] <= 1
        && u128::from(raw[4]) == bytes as u128
        && total <= u128::from(raw[7])
        && (raw[6] == 0 || raw[3] == 0);
    assert_eq!(result.is_some(), expected);
    if let Some(sample) = result {
        assert_eq!(sample.bytes as u128, bytes as u128);
        assert_eq!(sample.wait_ns, raw[1]);
        assert_eq!(sample.copy_ns, raw[2]);
        assert_eq!(sample.ready_before, raw[5] == 1);
        kani::cover!(bytes == 0 && sample.direct);
        kani::cover!(bytes == 40 && !sample.ready_before && sample.direct);
        kani::cover!(!sample.direct && sample.contiguous_build_ns > 0);
    }
    kani::cover!(!expected && total > u128::from(u64::MAX));
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(8)]
fn signature_interning_preserves_identity_for_three_requests() {
    let names = ["n", "m"];
    let headers = ["h", "j"];
    let sources = ["s", "t"];
    let mut keys = [(false, "", "", ""); 3];
    for key in &mut keys {
        *key = (
            kani::any(),
            names[usize::from(kani::any::<bool>())],
            headers[usize::from(kani::any::<bool>())],
            sources[usize::from(kani::any::<bool>())],
        );
    }
    let mut signatures = Vec::new();
    let mut ids = [0; 3];
    for i in 0..3 {
        ids[i] = intern_signature(&mut signatures, keys[i].0, keys[i].1, keys[i].2, keys[i].3).0;
        assert!(ids[i] < signatures.len() && signatures.len() <= i + 1);
        for j in 0..=i {
            assert_eq!(ids[i] == ids[j], keys[i] == keys[j]);
        }
    }
    kani::cover!(ids[0] == ids[1] && ids[1] == ids[2]);
    kani::cover!(ids[0] != ids[1] && ids[1] != ids[2] && ids[0] != ids[2]);
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

/// The fused command ends before residual HC/router/MoE work starts. Validate the
/// actual trace's ancestry, including indirect descendants; central intervals stay valid.
pub(crate) fn validate_cross_layer_ancestry(events: &[Event]) -> Result<()> {
    for (index, event) in events.iter().enumerate() {
        let protected = matches!(
            event.name.as_str(),
            "native.moe_wait" | "native.hc.chain_write_read_gate"
        ) || event.name.starts_with("native.moe.")
            || (event.name == "native.gpu_interval" && event.data["owner"] == "moe");
        let mut ancestor = event.parent;
        while let Some(parent) = ancestor {
            ensure!(parent < index, "invalid profile parent index");
            let upstream = &events[parent];
            ensure!(
                !protected || upstream.name != "native.hc_gdn.cross_layer",
                "MoE/residual event {} is inside HC/GDN triple",
                event.name
            );
            ensure!(
                upstream.parent.is_none_or(|next| next < parent),
                "cyclic profile ancestry"
            );
            ancestor = upstream.parent;
        }
    }
    Ok(())
}

#[test]
fn cross_layer_ancestry_rejects_leaked_residuals_and_keeps_central_intervals() -> Result<()> {
    fn record(name: &str, owner: Option<&str>, leaked: bool) -> Result<Vec<Event>> {
        start(false)?;
        let row = span("native.row");
        let mut triple = span("native.hc_gdn.cross_layer");
        {
            let central = span("native.gpu_interval");
            central
                .as_ref()
                .unwrap()
                .data(serde_json::json!({"owner":"central"}));
        }
        {
            let _write = span("native.hc.encode_write");
        }
        if !leaked {
            drop(triple.take());
        }
        let events = if leaked {
            let nested = span("native.command");
            {
                let target = span(name);
                if let Some(owner) = owner {
                    target
                        .as_ref()
                        .unwrap()
                        .data(serde_json::json!({"owner":owner}));
                }
            }
            drop(nested);
            drop(triple);
            drop(row);
            finish()?
        } else {
            {
                let nested = span("native.command");
                {
                    let target = span(name);
                    if let Some(owner) = owner {
                        target
                            .as_ref()
                            .unwrap()
                            .data(serde_json::json!({"owner":owner}));
                    }
                }
                drop(nested);
            }
            drop(row);
            finish()?
        };
        Ok(events)
    }
    for (name, owner) in [
        ("native.moe_wait", None),
        ("native.hc.chain_write_read_gate", None),
        ("native.gpu_interval", Some("moe")),
        ("native.moe.expert", None),
    ] {
        let mut valid = record(name, owner, false)?;
        validate_cross_layer_ancestry(&valid)?;
        assert!(validate_cross_layer_ancestry(&record(name, owner, true)?).is_err());
        let parent = valid[0].parent;
        valid[0].parent = Some(0);
        assert!(validate_cross_layer_ancestry(&valid).is_err());
        valid[0].parent = parent;
        let last = valid.len() - 1;
        valid[last].parent = Some(usize::MAX);
        assert!(validate_cross_layer_ancestry(&valid).is_err());
    }
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(64)]
fn cross_layer_ancestry_checks_actual_two_event_parent_guards() {
    let triple: bool = kani::any();
    let protected: bool = kani::any();
    let root_parent: Option<usize> = kani::any();
    let child_parent: Option<usize> = kani::any();
    let event = |name: &str, parent| Event {
        name: name.into(),
        parent,
        start_ns: 0,
        end_ns: 0,
        data: Value::Null,
        allocation_requests: 0,
        allocation_requested_bytes: 0,
        allocation_api_ns: 0,
    };
    let events = [
        event(
            if triple {
                "native.hc_gdn.cross_layer"
            } else {
                "native.row"
            },
            root_parent,
        ),
        event(
            if protected {
                "native.moe_wait"
            } else {
                "native.hc.encode_write"
            },
            child_parent,
        ),
    ];
    let actual = validate_cross_layer_ancestry(&events).is_ok();
    let expected = root_parent.is_none()
        && (child_parent.is_none() || child_parent == Some(0))
        && !(child_parent == Some(0) && protected && triple);
    assert_eq!(actual, expected);
    kani::cover!(actual && triple && !protected && child_parent == Some(0));
    kani::cover!(
        !actual && triple && protected && child_parent == Some(0) && root_parent.is_none()
    );
    kani::cover!(!actual && root_parent == Some(0));
    kani::cover!(!actual && child_parent == Some(usize::MAX));
    kani::cover!(actual && triple && protected && child_parent.is_none());
}
