//! Bounded owned disk reads. Workers own their entire unpublished destination.
use crate::{
    checkpoint::TensorInfo,
    flash_support::{EXPERT_BYTES, SlotLease},
};
use anyhow::{Context, Result, ensure};
use std::{
    cell::Cell,
    fs::File,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::{self, Receiver, Sender, SyncSender},
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};

#[cfg(test)]
use std::os::unix::fs::FileExt;

#[path = "ssd_reads/cancellable_io.rs"]
mod cancellable_io;
use cancellable_io::ReadCancellation;

pub(crate) enum ReadPayload {
    Bytes(Vec<u8>),
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    Shared(metal::Buffer),
}

impl ReadPayload {
    #[cfg(test)]
    fn legacy_bytes(&self) -> &[u8] {
        match self {
            Self::Bytes(bytes) => bytes,
            #[cfg(all(target_os = "macos", feature = "direct-metal"))]
            Self::Shared(_) => panic!("legacy bytes expected"),
        }
    }
}

#[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
pub(crate) fn direct_flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    crate::flash_support::hc_chain_flag(value, qualified_m5).map(|enabled| enabled && qualified_m5)
}

#[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
pub(crate) fn direct_read_eligible(enabled: bool, depth: usize, prefetcher: bool) -> bool {
    enabled && depth == 0 && !prefetcher
}

#[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
pub(crate) fn direct_destination_valid(bytes: usize, length: u64, shared: bool) -> bool {
    bytes > 0 && bytes <= EXPERT_BYTES as usize && length == bytes as u64 && shared
}

#[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
pub(crate) fn extra_wave_flag(value: Option<&str>, qualified_m5: bool) -> Option<bool> {
    crate::flash_support::hc_chain_flag(value, false).map(|enabled| enabled && qualified_m5)
}

#[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
pub(crate) fn extra_wave(enabled: bool, remaining: usize, extra_used: bool) -> bool {
    enabled && !extra_used && remaining >= 3
}

pub(crate) fn ready_minimum_valid(minimum: usize, remaining: usize) -> bool {
    minimum <= remaining && (minimum > 0 || remaining == 0)
}

#[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
pub(crate) fn shared_plan_valid(count: usize, owners: usize, extents_valid: bool) -> bool {
    count <= 10 && count == owners && extents_valid
}
#[cfg(kani)]
#[kani::proof]
fn early_shared_plan_checks_all_structural_preconditions() {
    let count: usize = kani::any();
    let owners: usize = kani::any();
    let extents: bool = kani::any();
    assert_eq!(
        shared_plan_valid(count, owners, extents),
        count <= 10 && count == owners && extents
    );
    kani::cover!(count == 10 && shared_plan_valid(count, owners, extents));
    kani::cover!(count == 0 && shared_plan_valid(count, owners, extents));
}
pub(crate) fn common_queue_flag(value: Option<&str>) -> Option<bool> {
    crate::flash_support::hc_chain_flag(value, false)
}

pub(crate) fn record_reuse_eligible(
    enabled: bool,
    depth: usize,
    prefetcher: bool,
    completed: bool,
) -> bool {
    enabled && depth == 0 && !prefetcher && completed
}

pub(crate) fn isolated_record_ranges(
    enabled: bool,
    records: usize,
    workers: usize,
    bytes: usize,
    offset: u64,
) -> Option<[[usize; 2]; 2]> {
    if !enabled
        || records != 1
        || !(2..=4).contains(&workers)
        || bytes < 8192
        || bytes > EXPERT_BYTES as usize
        || offset.checked_add(bytes as u64).is_none()
    {
        return None;
    }
    let middle = bytes / 2 / 4096 * 4096;
    Some([[0, middle], [middle, bytes - middle]])
}

pub(crate) fn claim_record_part(claimed: u8, index: usize) -> Option<u8> {
    if claimed > 3 || index >= 2 || claimed & (1 << index) != 0 {
        None
    } else {
        Some(claimed | (1 << index))
    }
}

#[cfg(test)]
#[derive(Default)]
pub(crate) struct TestReadGate(std::sync::Mutex<bool>, std::sync::Condvar);
#[cfg(test)]
impl TestReadGate {
    fn wait(&self) {
        let released = self.0.lock().unwrap();
        let (released, _) = self
            .1
            .wait_timeout_while(released, Duration::from_secs(5), |v| !*v)
            .unwrap();
        assert!(*released, "test read gate was not released");
    }
    pub(crate) fn release(&self) {
        *self.0.lock().unwrap() = true;
        self.1.notify_all();
    }
}

pub(crate) struct Completed {
    lease: SlotLease,
    result: Result<ReadPayload>,
    #[cfg(test)]
    timing: (Instant, Instant),
}
#[cfg(test)]
impl Completed {
    fn new(lease: SlotLease, result: Result<Vec<u8>>) -> Self {
        Self {
            lease,
            result: result.map(ReadPayload::Bytes),
            #[cfg(test)]
            timing: (Instant::now(), Instant::now()),
        }
    }
}

struct Ticket(Arc<AtomicUsize>);
impl Drop for Ticket {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Release);
    }
}
#[cfg(test)]
type SharedTrace = Arc<Mutex<Vec<(&'static str, Instant)>>>;
struct Read {
    cancellation: ReadCancellation,
    #[cfg(test)]
    shared_trace: Option<SharedTrace>,
    lease: SlotLease,
    file: File,
    offset: u64,
    bytes: usize,
    payload: ReadPayload,
    result: Sender<Completed>,
    ticket: Ticket,
    #[cfg(test)]
    gate: Option<Arc<std::sync::Barrier>>,
    #[cfg(test)]
    wave_gate: Option<Arc<TestReadGate>>,
}
impl Read {
    fn run(self) {
        let Read {
            cancellation,
            #[cfg(test)]
            shared_trace,
            lease,
            file,
            offset,
            bytes,
            mut payload,
            result: sender,
            ticket,
            #[cfg(test)]
            gate,
            #[cfg(test)]
            wave_gate,
        } = self;
        #[cfg(all(test, not(all(target_os = "macos", feature = "direct-metal"))))]
        let _ = shared_trace;
        #[cfg(test)]
        let started = Instant::now();
        #[cfg(test)]
        if let Some(gate) = gate {
            gate.wait();
        }
        #[cfg(test)]
        if let Some(gate) = wave_gate {
            gate.wait();
        }
        let result = (|| -> Result<ReadPayload> {
            cancellation.check()?;
            match &mut payload {
                ReadPayload::Bytes(destination) => {
                    destination.try_reserve_exact(bytes)?;
                    destination.resize(bytes, 0);
                    cancellation.read_exact_at(&file, destination, offset)?;
                }
                #[cfg(all(target_os = "macos", feature = "direct-metal"))]
                ReadPayload::Shared(destination) => {
                    ensure!(
                        direct_destination_valid(
                            bytes,
                            destination.length(),
                            destination.storage_mode() == metal::MTLStorageMode::Shared
                        ),
                        "invalid owned SSD destination"
                    );
                    ensure!(!destination.contents().is_null(), "null SSD destination");
                    // This job has exclusive CPU-write/future-GPU access: fresh,
                    // or retired after Completed with all old slot views dropped.
                    // Old Completed command buffers cannot execute again.
                    let target = unsafe {
                        std::slice::from_raw_parts_mut(destination.contents().cast::<u8>(), bytes)
                    };
                    #[cfg(test)]
                    if let Some(trace) = &shared_trace {
                        trace.lock().unwrap().push(("pread", Instant::now()));
                    }
                    cancellation.read_exact_at(&file, target, offset)?;
                }
            }
            Ok(payload)
        })();
        drop(ticket);
        let result = Completed {
            lease,
            result,
            #[cfg(test)]
            timing: (started, Instant::now()),
        };
        let _ = sender.send(result);
    }
}

#[cfg(test)]
thread_local! { static CAMPAIGN_READS: std::cell::Cell<Option<(u64,u64)>> = const { std::cell::Cell::new(None) }; }
#[cfg(all(test, target_os = "macos", feature = "direct-metal", feature = "chat"))]
pub(crate) fn campaign_reads(enabled: bool) -> (u64, u64) {
    CAMPAIGN_READS
        .replace(enabled.then_some((0, 0)))
        .unwrap_or_default()
}

enum ReadWork {
    Whole(Read),
    Part {
        record: Arc<SplitRecord>,
        index: usize,
        file: File,
    },
}
impl ReadWork {
    fn run(self) {
        match self {
            Self::Whole(read) => read.run(),
            Self::Part {
                record,
                index,
                file,
            } => record.run_part(index, file),
        }
    }
}

struct SplitState {
    read: Option<Read>,
    remaining: usize,
    claimed: u8,
    errors: [Option<anyhow::Error>; 2],
    #[cfg(test)]
    started: Option<Instant>,
}
struct SplitRecord {
    state: Mutex<SplitState>,
    // The state owns the allocation until both jobs finish; no unsafe Send impl.
    address: usize,
    ranges: [[usize; 2]; 2],
    #[cfg(test)]
    gates: Option<[Arc<TestReadGate>; 2]>,
    #[cfg(test)]
    finished: Arc<AtomicUsize>,
    #[cfg(test)]
    started_parts: Arc<AtomicUsize>,
}
impl SplitRecord {
    fn new(
        mut read: Read,
        ranges: [[usize; 2]; 2],
        #[cfg(test)] gates: Option<[Arc<TestReadGate>; 2]>,
        #[cfg(test)] finished: Arc<AtomicUsize>,
        #[cfg(test)] started_parts: Arc<AtomicUsize>,
    ) -> Result<Arc<Self>> {
        ensure!(
            ranges[0] == [0, ranges[1][0]]
                && ranges[0][1] > 0
                && ranges[1][1] > 0
                && ranges[1][0].checked_add(ranges[1][1]) == Some(read.bytes),
            "invalid isolated record ranges"
        );
        read.cancellation.check()?;
        let address = match &mut read.payload {
            ReadPayload::Bytes(bytes) => {
                bytes.try_reserve_exact(read.bytes)?;
                bytes.resize(read.bytes, 0);
                bytes.as_mut_ptr() as usize
            }
            #[cfg(all(target_os = "macos", feature = "direct-metal"))]
            ReadPayload::Shared(buffer) => {
                ensure!(
                    direct_destination_valid(
                        read.bytes,
                        buffer.length(),
                        buffer.storage_mode() == metal::MTLStorageMode::Shared
                    ),
                    "invalid split Shared destination"
                );
                buffer.contents() as usize
            }
        };
        ensure!(address != 0, "null split destination");
        Ok(Arc::new(Self {
            state: Mutex::new(SplitState {
                read: Some(read),
                remaining: 2,
                claimed: 0,
                errors: [None, None],
                #[cfg(test)]
                started: None,
            }),
            address,
            ranges,
            #[cfg(test)]
            gates,
            #[cfg(test)]
            finished,
            #[cfg(test)]
            started_parts,
        }))
    }
    fn run_part(&self, index: usize, file: File) {
        {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let Some(claimed) = claim_record_part(state.claimed, index) else {
                return;
            };
            state.claimed = claimed;
        }
        #[cfg(test)]
        {
            let (gate, wave) = {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                state.started.get_or_insert_with(Instant::now);
                let read = state.read.as_ref().unwrap();
                (read.gate.clone(), read.wave_gate.clone())
            };
            if index == 0 {
                if let Some(gate) = gate {
                    gate.wait();
                }
                if let Some(gate) = wave {
                    gate.wait();
                }
            }
        }
        let [begin, length] = self.ranges[index];
        let (offset, cancellation) = {
            let state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            let read = state.read.as_ref().unwrap();
            (read.offset, read.cancellation.clone())
        };
        let result = (|| -> Result<()> {
            let at = offset
                .checked_add(begin as u64)
                .context("split offset overflow")?;
            #[cfg(test)]
            {
                // The observer/gate is in the IO region after all mutex guards
                // were released: holding one part must not block the other.
                self.started_parts.fetch_add(1, Ordering::Release);
                if let Some(gates) = &self.gates {
                    gates[index].wait();
                }
            }
            // The only two jobs own disjoint, nonempty ranges covering the owned
            // allocation exactly. The Arc retains it until both writes return.
            let target = unsafe {
                std::slice::from_raw_parts_mut((self.address as *mut u8).add(begin), length)
            };
            cancellation.read_exact_at(&file, target, at)?;
            Ok(())
        })();
        let completed = {
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.errors[index] = result.err();
            state.remaining -= 1;
            if state.remaining == 0 {
                let read = state.read.take().unwrap();
                let error = state.errors.iter_mut().find_map(Option::take);
                Some((read, error))
            } else {
                None
            }
        };
        if let Some((read, error)) = completed {
            #[cfg(test)]
            let started = self
                .state
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .started
                .unwrap();
            let Read {
                lease,
                payload,
                result: sender,
                ticket,
                ..
            } = read;
            let result = match error {
                Some(error) => Err(error),
                None => Ok(payload),
            };
            drop(ticket);
            let _ = sender.send(Completed {
                lease,
                result,
                #[cfg(test)]
                timing: (started, Instant::now()),
            });
        }
        #[cfg(test)]
        self.finished.fetch_add(1, Ordering::Release);
    }
}

#[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
fn readahead_eligible(enabled: bool, qualified_m5: bool) -> bool {
    enabled && qualified_m5
}

#[cfg(any(test, kani, all(target_os = "macos", feature = "direct-metal")))]
fn readahead_identity_matches(inspected: [u64; 2], opened: [u64; 2]) -> bool {
    inspected == opened
}

#[test]
fn dedicated_readahead_requires_qualified_source_and_exact_inode() {
    for e in [false, true] {
        for q in [false, true] {
            assert_eq!(readahead_eligible(e, q), e && q);
        }
    }
    assert!(readahead_identity_matches([1, 2], [1, 2]));
    assert!(!readahead_identity_matches([1, 2], [2, 1]));
}

#[cfg(kani)]
#[kani::proof]
fn dedicated_readahead_never_accepts_unqualified_source_or_replaced_inode() {
    let e: bool = kani::any();
    let q: bool = kani::any();
    let a: [u64; 2] = kani::any();
    let b: [u64; 2] = kani::any();
    assert_eq!(readahead_eligible(e, q), e && q);
    assert_eq!(
        readahead_identity_matches(a, b),
        a[0] == b[0] && a[1] == b[1]
    );
    kani::cover!(readahead_eligible(e, q) && readahead_identity_matches(a, b));
    kani::cover!(e && !q);
    kani::cover!(a[0] != b[0] && a[1] == b[1]);
    kani::cover!(a[0] == b[0] && a[1] != b[1]);
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
#[path = "ssd_reads/readahead.rs"]
mod readahead;

pub(crate) struct ReadService {
    #[cfg(test)]
    io_probe: std::cell::RefCell<Option<Arc<cancellable_io::ReadProbe>>>,
    cancelled: Arc<AtomicBool>,
    draining: Arc<AtomicBool>,
    #[cfg(test)]
    shared_trace: std::cell::RefCell<Option<SharedTrace>>,
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    early_reads: bool,
    #[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
    pub(crate) early_batches: Cell<usize>,
    pub(crate) window_enabled: bool,
    window_active: Cell<bool>,
    next: Cell<usize>,
    senders: Vec<SyncSender<ReadWork>>,
    workers: Vec<JoinHandle<()>>,
    dynamic: bool,
    split_enabled: bool,
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    readahead_files: Option<readahead::ExpertFiles>,
    active: Arc<AtomicUsize>,
    #[cfg(test)]
    pub(crate) dynamic_submissions: Cell<usize>,
    #[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
    pub(crate) reused_destinations: Cell<usize>,
    #[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
    pub(crate) fresh_destinations: Cell<usize>,
    #[cfg(test)]
    pub(crate) split_records: Cell<usize>,
    #[cfg(test)]
    pub(crate) split_pieces: Cell<usize>,
    #[cfg(test)]
    split_gates: std::cell::RefCell<Option<[Arc<TestReadGate>; 2]>>,
    #[cfg(test)]
    pub(crate) split_finished: Arc<AtomicUsize>,
    #[cfg(test)]
    pub(crate) split_started: Arc<AtomicUsize>,
    #[cfg(test)]
    held_first: std::cell::RefCell<Option<Arc<TestReadGate>>>,
    #[cfg(test)]
    gate: std::cell::RefCell<Option<Arc<std::sync::Barrier>>>,
    #[cfg(test)]
    wave_gates: std::cell::RefCell<Option<[Arc<TestReadGate>; 2]>>,
}
impl ReadService {
    #[cfg(any(test, all(target_os = "macos", feature = "direct-metal")))]
    pub(crate) fn set_cancellation(&mut self, cancelled: Arc<AtomicBool>) {
        self.cancelled = cancelled;
    }
    pub(crate) fn new(readers: usize) -> Result<Self> {
        Self::new_with_split_default(readers, false)
    }
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    pub(crate) fn new_qualified(readers: usize, qualified_m5: bool) -> Result<Self> {
        if qualified_m5 {
            Self::new_with_split_default(readers, true)
        } else {
            Self::new(readers)
        }
    }
    #[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
    pub(crate) fn split_requested(&self) -> bool {
        self.split_enabled
    }
    fn new_with_split_default(readers: usize, qualified_default: bool) -> Result<Self> {
        let dynamic = match std::env::var("MLXL3_SSD_DYNAMIC_READERS") {
            Ok(value) => common_queue_flag(Some(&value)),
            Err(std::env::VarError::NotPresent) => common_queue_flag(None),
            Err(error) => return Err(error.into()),
        }
        .context("MLXL3_SSD_DYNAMIC_READERS must be 0 or 1")?;
        let split_enabled = match std::env::var("MLXL3_NATIVE_READ_SPLIT") {
            Ok(value) => crate::ar10_defaults::split_flag(Some(&value), qualified_default),
            Err(std::env::VarError::NotPresent) => {
                crate::ar10_defaults::split_flag(None, qualified_default)
            }
            Err(error) => return Err(error.into()),
        }
        .context("MLXL3_NATIVE_READ_SPLIT must be 0 or 1")?;
        let window_enabled = match std::env::var("MLXL3_NATIVE_FLASH_READ_WINDOW") {
            Ok(value) => crate::ar10_defaults::flag(Some(&value), qualified_default),
            Err(std::env::VarError::NotPresent) => {
                #[cfg(all(target_os = "macos", feature = "direct-metal"))]
                let default = crate::native_executor::defaults(qualified_default);
                #[cfg(not(all(target_os = "macos", feature = "direct-metal")))]
                let default = false;
                Some(default)
            }
            Err(error) => return Err(error.into()),
        }
        .context("MLXL3_NATIVE_FLASH_READ_WINDOW must be 0 or 1")?;
        let mut service = Self::new_with_options(readers, dynamic, split_enabled)?;
        service.window_enabled = window_enabled;
        Ok(service)
    }
    #[cfg(test)]
    fn new_with_schedule(readers: usize, dynamic: bool) -> Result<Self> {
        Self::new_with_options(readers, dynamic, false)
    }
    fn new_with_options(readers: usize, dynamic: bool, split_enabled: bool) -> Result<Self> {
        ensure!((1..=4).contains(&readers), "SSD readers must be in 1..=4");
        let mut service = Self {
            #[cfg(test)]
            io_probe: std::cell::RefCell::new(None),
            cancelled: Arc::new(AtomicBool::new(false)),
            draining: Arc::new(AtomicBool::new(false)),
            #[cfg(test)]
            shared_trace: std::cell::RefCell::new(None),
            #[cfg(all(target_os = "macos", feature = "direct-metal"))]
            early_reads: match std::env::var("MLXL3_NATIVE_EARLY_READS") {
                Ok(value) => common_queue_flag(Some(&value)),
                Err(std::env::VarError::NotPresent) => Some(false),
                Err(error) => return Err(error.into()),
            }
            .context("MLXL3_NATIVE_EARLY_READS must be 0 or 1")?,
            #[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
            early_batches: Cell::new(0),
            window_enabled: false,
            window_active: Cell::new(false),
            next: Cell::new(0),
            senders: Vec::new(),
            workers: Vec::new(),
            dynamic,
            split_enabled,
            #[cfg(all(target_os = "macos", feature = "direct-metal"))]
            readahead_files: readahead::ExpertFiles::from_env()?,
            active: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            dynamic_submissions: Cell::new(0),
            #[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
            reused_destinations: Cell::new(0),
            #[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
            fresh_destinations: Cell::new(0),
            #[cfg(test)]
            split_records: Cell::new(0),
            #[cfg(test)]
            split_pieces: Cell::new(0),
            #[cfg(test)]
            split_gates: std::cell::RefCell::new(None),
            #[cfg(test)]
            split_finished: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            split_started: Arc::new(AtomicUsize::new(0)),
            #[cfg(test)]
            held_first: std::cell::RefCell::new(None),
            #[cfg(test)]
            gate: std::cell::RefCell::new(None),
            #[cfg(test)]
            wave_gates: std::cell::RefCell::new(None),
        };
        if dynamic {
            let (sender, receiver) = mpsc::sync_channel::<ReadWork>(10);
            service.senders.push(sender);
            let receiver = Arc::new(Mutex::new(receiver));
            for index in 0..readers {
                let receiver = Arc::clone(&receiver);
                service.workers.push(
                    std::thread::Builder::new()
                        .name(format!("native-ssd-{index}"))
                        .spawn(move || {
                            loop {
                                let next = {
                                    let queue = receiver.lock().unwrap_or_else(|e| e.into_inner());
                                    queue.recv()
                                };
                                let Ok(read) = next else { break };
                                // The queue lock never covers IO or destination ownership.
                                read.run();
                            }
                        })?,
                );
            }
        } else {
            for index in 0..readers {
                let (sender, receiver) = mpsc::sync_channel::<ReadWork>(10);
                service.senders.push(sender);
                service.workers.push(
                    std::thread::Builder::new()
                        .name(format!("native-ssd-{index}"))
                        .spawn(move || {
                            while let Ok(read) = receiver.recv() {
                                read.run();
                            }
                        })?,
                );
            }
        }
        Ok(service)
    }
    #[cfg(test)]
    pub(crate) fn set_test_gate(&self, gate: Option<Arc<std::sync::Barrier>>) {
        *self.gate.borrow_mut() = gate;
    }
    #[cfg(test)]
    pub(crate) fn release_test_gate(&self) {
        if let Some(gate) = self.gate.borrow_mut().take() {
            gate.wait();
        }
    }
    #[cfg(test)]
    pub(crate) fn set_wave_test_gates(&self) -> [Arc<TestReadGate>; 2] {
        let gates = [
            Arc::new(TestReadGate::default()),
            Arc::new(TestReadGate::default()),
        ];
        *self.wave_gates.borrow_mut() = Some(gates.clone());
        gates
    }
    #[cfg(test)]
    pub(crate) fn release_wave_test_gate(&self, final_tail: bool) {
        let mut gates = self.wave_gates.borrow_mut();
        if let Some(gates) = gates.as_ref() {
            gates[0].release();
            if final_tail {
                gates[1].release();
            }
        }
        if final_tail {
            *gates = None;
        }
    }
    #[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
    pub(crate) fn readahead_counts(&self) -> [usize; 2] {
        self.readahead_files
            .as_ref()
            .map_or([0; 2], |files| files.counts())
    }
    pub(crate) fn in_flight(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }
    pub(crate) fn submit(
        &self,
        records: impl IntoIterator<Item = (SlotLease, TensorInfo)>,
    ) -> Result<ReadBatch> {
        self.start(self.sources(records, false)?, Duration::from_secs(30))
    }
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    pub(crate) fn submit_shared(
        &self,
        records: impl IntoIterator<Item = (SlotLease, TensorInfo)>,
        device: &metal::DeviceRef,
    ) -> Result<ReadBatch> {
        self.start_shared(
            self.sources(records, device.name() == "Apple M5")?,
            Duration::from_secs(30),
            device,
        )
    }
    /// Device-Hot retains old cache owners: this API accepts no reusable destination.
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    pub(crate) fn submit_shared_fresh(
        &self,
        records: impl IntoIterator<Item = (SlotLease, TensorInfo)>,
        device: &metal::DeviceRef,
    ) -> Result<ReadBatch> {
        self.submit_shared(records, device)
    }
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    fn start_shared(
        &self,
        reads: Vec<(SlotLease, File, u64, usize)>,
        limit: Duration,
        device: &metal::DeviceRef,
    ) -> Result<ReadBatch> {
        let count = reads.len();
        self.start_shared_reusing(reads, (0..count).map(|_| None), limit, device)
    }
    /// Owners come only from private, retired cache slots after all old GPU uses
    /// completed and every old projection view was dropped by FlashMoeRow.
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    pub(crate) fn submit_shared_reusing(
        &self,
        records: impl IntoIterator<Item = (SlotLease, TensorInfo, Option<metal::Buffer>)>,
        device: &metal::DeviceRef,
    ) -> Result<ReadBatch> {
        let (records, owners): (Vec<_>, Vec<_>) = records
            .into_iter()
            .map(|(lease, record, owner)| ((lease, record), owner))
            .unzip();
        self.start_shared_reusing(
            self.sources(records, device.name() == "Apple M5")?,
            owners,
            Duration::from_secs(30),
            device,
        )
    }
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    fn start_shared_reusing(
        &self,
        reads: Vec<(SlotLease, File, u64, usize)>,
        owners: impl IntoIterator<Item = Option<metal::Buffer>, IntoIter: ExactSizeIterator>,
        limit: Duration,
        device: &metal::DeviceRef,
    ) -> Result<ReadBatch> {
        self.validate_start(reads.len(), false)?;
        let owners = owners.into_iter();
        ensure!(
            shared_plan_valid(
                reads.len(),
                owners.len(),
                reads
                    .iter()
                    .all(|r| direct_destination_valid(r.3, r.3 as u64, true))
            ),
            "invalid shared read plan"
        );
        let reads = reads
            .into_iter()
            .zip(owners)
            .map(|((lease, file, offset, bytes), owner)| {
                let destination = self.owned_destination(device, owner, bytes)?;
                Ok((lease, file, offset, bytes, ReadPayload::Shared(destination)))
            });
        let allow_split = self.split_enabled && device.name() == "Apple M5";
        if self.early_reads {
            #[cfg(test)]
            if reads.len() > 0 {
                self.early_batches.set(self.early_batches.get() + 1);
            }
            return self.start_fallible_payloads(reads, limit, false, allow_split);
        }
        self.start_payloads(
            reads.collect::<Result<Vec<_>>>()?.into_iter(),
            limit,
            false,
            self.split_enabled && device.name() == "Apple M5",
        )
    }
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    fn owned_destination(
        &self,
        device: &metal::DeviceRef,
        owner: Option<metal::Buffer>,
        bytes: usize,
    ) -> Result<metal::Buffer> {
        ensure!(
            direct_destination_valid(bytes, bytes as u64, true),
            "invalid owned SSD destination extent"
        );
        let reusable = owner.filter(|buffer| {
            buffer.device().registry_id() == device.registry_id()
                && direct_destination_valid(
                    bytes,
                    buffer.length(),
                    buffer.storage_mode() == metal::MTLStorageMode::Shared,
                )
        });
        let destination = if let Some(buffer) = reusable {
            #[cfg(test)]
            self.reused_destinations
                .set(self.reused_destinations.get() + 1);
            buffer
        } else {
            let buffer = crate::gpu::buffer(
                device,
                bytes as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )?;
            #[cfg(test)]
            self.fresh_destinations
                .set(self.fresh_destinations.get() + 1);
            buffer
        };
        #[cfg(test)]
        if let Some(trace) = self.shared_trace.borrow().as_ref() {
            trace.lock().unwrap().push(("allocation", Instant::now()));
        }
        Ok(destination)
    }
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    pub(crate) fn submit_background(
        &self,
        lease: SlotLease,
        record: TensorInfo,
    ) -> Result<ReadBatch> {
        self.start_mode(
            self.sources([(lease, record)], false)?,
            Duration::from_secs(30),
            true,
        )
    }
    fn sources(
        &self,
        records: impl IntoIterator<Item = (SlotLease, TensorInfo)>,
        qualified_m5: bool,
    ) -> Result<Vec<(SlotLease, File, u64, usize)>> {
        let mut reads = Vec::new();
        for (lease, record) in records {
            // The inspected record descriptor is its first constituent tensor.
            let (file, offset) = record.source()?;
            let end = offset
                .checked_add(EXPERT_BYTES)
                .context("Flash record offset overflow")?;
            ensure!(
                end <= file.metadata()?.len(),
                "checkpoint record was truncated after inspection"
            );
            #[cfg(all(target_os = "macos", feature = "direct-metal"))]
            let handle = if let Some(files) = self
                .readahead_files
                .as_ref()
                .filter(|_| readahead_eligible(true, qualified_m5))
            {
                files.source(&record.file, file)?
            } else {
                file.try_clone()?
            };
            #[cfg(not(all(target_os = "macos", feature = "direct-metal")))]
            let handle = {
                let _ = qualified_m5;
                file.try_clone()?
            };
            reads.push((lease, handle, offset, EXPERT_BYTES as usize));
        }
        Ok(reads)
    }
    fn start(
        &self,
        reads: Vec<(SlotLease, File, u64, usize)>,
        limit: Duration,
    ) -> Result<ReadBatch> {
        self.start_mode(reads, limit, false)
    }
    fn start_mode(
        &self,
        reads: Vec<(SlotLease, File, u64, usize)>,
        limit: Duration,
        background: bool,
    ) -> Result<ReadBatch> {
        let reads = reads.into_iter().map(|(lease, file, offset, bytes)| {
            (lease, file, offset, bytes, ReadPayload::Bytes(Vec::new()))
        });
        self.start_payloads(reads, limit, background, false)
    }
    fn validate_start(&self, count: usize, background: bool) -> Result<()> {
        ensure!(
            !self.cancelled.load(Ordering::Acquire),
            "generation cancelled"
        );
        if self.draining.load(Ordering::Acquire) {
            ensure!(
                self.in_flight() == 0,
                "cancelled native SSD reads are still draining"
            );
            self.draining.store(false, Ordering::Release);
        }
        ensure!(
            !self.window_active.get(),
            "native SSD window still owns the reader pool"
        );
        ensure!(count <= 10, "native SSD demand exceeds one AR cohort");
        ensure!(
            !background || (count == 1 && self.in_flight() < 4),
            "native prefetch exceeds four readers"
        );
        ensure!(
            background || self.active.load(Ordering::Acquire) == 0,
            "previous native SSD reads are still draining"
        );
        Ok(())
    }
    fn start_payloads(
        &self,
        reads: impl ExactSizeIterator<Item = (SlotLease, File, u64, usize, ReadPayload)>,
        limit: Duration,
        background: bool,
        allow_split: bool,
    ) -> Result<ReadBatch> {
        self.start_fallible_payloads(reads.map(Ok), limit, background, allow_split)
    }
    fn start_fallible_payloads(
        &self,
        reads: impl ExactSizeIterator<Item = Result<(SlotLease, File, u64, usize, ReadPayload)>>,
        limit: Duration,
        background: bool,
        allow_split: bool,
    ) -> Result<ReadBatch> {
        self.validate_start(reads.len(), background)?;
        let (result, receiver) = mpsc::channel();
        let mut batch = ReadBatch {
            io_cancelled: Arc::new(AtomicBool::new(false)),
            receiver,
            cancelled: Arc::clone(&self.cancelled),
            draining: Arc::clone(&self.draining),
            abandoned: false,
            remaining: 0,
            deadline: Instant::now() + limit,
            #[cfg(test)]
            trace: Vec::new(),
            #[cfg(test)]
            wait_started: None,
        };
        self.enqueue_payloads(&mut batch, &result, reads, background, allow_split)?;
        Ok(batch)
    }
    fn enqueue_payloads(
        &self,
        batch: &mut ReadBatch,
        result: &Sender<Completed>,
        reads: impl ExactSizeIterator<Item = Result<(SlotLease, File, u64, usize, ReadPayload)>>,
        background: bool,
        allow_split: bool,
    ) -> Result<()> {
        let outcome = self.enqueue_payloads_inner(batch, result, reads, background, allow_split);
        terminal_batch_result(outcome, || batch.abandon())
    }
    fn enqueue_payloads_inner(
        &self,
        batch: &mut ReadBatch,
        result: &Sender<Completed>,
        reads: impl ExactSizeIterator<Item = Result<(SlotLease, File, u64, usize, ReadPayload)>>,
        background: bool,
        allow_split: bool,
    ) -> Result<()> {
        let count = reads.len();
        for item in reads {
            batch.check_cancelled()?;
            let (lease, file, offset, bytes, payload) = item?;
            self.active.fetch_add(1, Ordering::Relaxed);
            let read = Read {
                cancellation: ReadCancellation {
                    requested: Arc::clone(&batch.cancelled),
                    stopped: Arc::clone(&batch.io_cancelled),
                    #[cfg(test)]
                    probe: self.io_probe.borrow().clone(),
                },
                #[cfg(test)]
                shared_trace: self.shared_trace.borrow().clone(),
                lease,
                file,
                offset,
                bytes,
                payload,
                result: result.clone(),
                ticket: Ticket(self.active.clone()),
                #[cfg(test)]
                gate: (batch.remaining + 1 == count)
                    .then(|| self.gate.borrow().clone())
                    .flatten(),
                #[cfg(test)]
                wave_gate: self
                    .held_first
                    .borrow()
                    .clone()
                    .filter(|_| batch.remaining == 0)
                    .or_else(|| {
                        self.wave_gates
                            .borrow()
                            .as_ref()
                            .and_then(|gates| match batch.remaining {
                                0 => None,
                                1 | 2 => Some(gates[0].clone()),
                                _ => Some(gates[1].clone()),
                            })
                    }),
            };
            if let Some(ranges) = isolated_record_ranges(
                allow_split && self.split_enabled && !background,
                count,
                self.workers.len(),
                bytes,
                offset,
            ) {
                let files = [read.file.try_clone()?, read.file.try_clone()?];
                let record = SplitRecord::new(
                    read,
                    ranges,
                    #[cfg(test)]
                    self.split_gates.borrow().clone(),
                    #[cfg(test)]
                    Arc::clone(&self.split_finished),
                    #[cfg(test)]
                    Arc::clone(&self.split_started),
                )?;
                // One logical record, including partially enqueued failure paths.
                batch.remaining += 1;
                for (index, file) in files.into_iter().enumerate() {
                    self.enqueue(ReadWork::Part {
                        record: Arc::clone(&record),
                        index,
                        file,
                    })?;
                    #[cfg(test)]
                    self.split_pieces.set(self.split_pieces.get() + 1);
                }
                #[cfg(test)]
                self.split_records.set(self.split_records.get() + 1);
            } else {
                self.enqueue(ReadWork::Whole(read))?;
                batch.remaining += 1;
            }
        }
        Ok(())
    }
    fn enqueue(&self, work: ReadWork) -> Result<()> {
        #[cfg(test)]
        let bytes = match &work {
            ReadWork::Whole(read) => read.bytes as u64,
            ReadWork::Part { record, index, .. } => record.ranges[*index][1] as u64,
        };
        let worker = self.next.get();
        self.next.set((worker + 1) % self.workers.len());
        self.senders[if self.dynamic { 0 } else { worker }]
            .try_send(work)
            .map_err(|_| anyhow::anyhow!("native SSD reader unavailable"))?;
        #[cfg(test)]
        if let Some((total, jobs)) = CAMPAIGN_READS.get() {
            CAMPAIGN_READS.set(Some((
                total.checked_add(bytes).expect("campaign byte overflow"),
                jobs.checked_add(1).expect("campaign job overflow"),
            )));
        }
        #[cfg(test)]
        if self.dynamic {
            self.dynamic_submissions
                .set(self.dynamic_submissions.get() + 1);
        }
        Ok(())
    }
}
impl Drop for ReadService {
    fn drop(&mut self) {
        // Disconnect the common queue before joining any idle receiver.
        self.senders.clear();
        let workers = std::mem::take(&mut self.workers);
        for thread in workers {
            if self.active.load(Ordering::Acquire) == 0 {
                let _ = thread.join();
            }
            // A timed-out read owns its file and destination through physical completion.
        }
    }
}

/// Stop a failed batch before its caller unwinds or can request another result.
/// Preserve the original error rather than replacing it with cleanup status.
fn terminal_batch_result<T, E>(
    result: std::result::Result<T, E>,
    stop: impl FnOnce(),
) -> std::result::Result<T, E> {
    if result.is_err() {
        stop();
    }
    result
}

fn latch_cancellation(abandoned: &mut bool, requested: bool) -> bool {
    *abandoned |= requested;
    *abandoned
}

#[cfg(kani)]
#[kani::proof]
fn batch_cancellation_is_irreversible_across_flag_reset() {
    let initial: bool = kani::any();
    let requested: bool = kani::any();
    let mut abandoned = initial;
    let observed = latch_cancellation(&mut abandoned, requested);
    assert_eq!(observed, initial || requested);
    assert_eq!(latch_cancellation(&mut abandoned, false), observed);
    assert!(latch_cancellation(&mut abandoned, true));
    assert!(latch_cancellation(&mut abandoned, false));
    kani::cover!(!initial && requested);
    kani::cover!(initial && !requested);
}

pub(crate) struct ReadBatch {
    io_cancelled: Arc<AtomicBool>,
    receiver: Receiver<Completed>,
    cancelled: Arc<AtomicBool>,
    draining: Arc<AtomicBool>,
    abandoned: bool,
    remaining: usize,
    deadline: Instant,
    #[cfg(test)]
    pub(crate) trace: Vec<(SlotLease, Instant, Instant)>,
    #[cfg(test)]
    wait_started: Option<Sender<()>>,
}
impl ReadBatch {
    /// Stop queued work and remaining read chunks as well as publication.
    /// An active syscall retains its file, destination and ticket until return;
    /// macOS does not guarantee interruption of an already executing file read.
    pub(crate) fn abandon(&mut self) {
        self.io_cancelled.store(true, Ordering::Release);
        self.draining.store(true, Ordering::Release);
        self.abandoned = true;
        self.remaining = 0;
        self.receiver = mpsc::channel().1;
    }
    fn check_cancelled(&mut self) -> Result<()> {
        if latch_cancellation(
            &mut self.abandoned,
            self.cancelled.load(Ordering::Acquire) || self.io_cancelled.load(Ordering::Acquire),
        ) {
            self.abandon();
            anyhow::bail!("native SSD batch cancelled or abandoned after failure");
        }
        Ok(())
    }
    fn wait_next(&mut self) -> Result<Completed> {
        loop {
            self.check_cancelled()?;
            let left = self.deadline.saturating_duration_since(Instant::now());
            #[cfg(test)]
            if let Some(started) = self.wait_started.take() {
                let _ = started.send(());
            }
            match self
                .receiver
                .recv_timeout(left.min(Duration::from_millis(10)))
            {
                Ok(next) => {
                    self.check_cancelled()?;
                    return Ok(next);
                }
                Err(mpsc::RecvTimeoutError::Timeout) if Instant::now() < self.deadline => {}
                Err(error) => {
                    self.abandon();
                    return Err(error).context("native SSD demand deadline or reader failure");
                }
            }
        }
    }
    pub(crate) fn remaining(&self) -> usize {
        self.remaining
    }
    pub(crate) fn poll(&mut self) -> Result<Option<(SlotLease, ReadPayload)>> {
        self.check_cancelled()?;
        ensure!(self.remaining == 1, "prefetch must own one pending read");
        let result = (|| match self.receiver.try_recv() {
            Ok(next) => {
                self.check_cancelled()?;
                self.remaining = 0;
                Ok(Some((next.lease, next.result?)))
            }
            Err(mpsc::TryRecvError::Empty) => {
                ensure!(
                    Instant::now() < self.deadline,
                    "native SSD prefetch deadline"
                );
                Ok(None)
            }
            Err(error) => Err(error.into()),
        })();
        terminal_batch_result(result, || self.abandon())
    }
    pub(crate) fn ready(&mut self, all: bool) -> Result<Vec<(SlotLease, ReadPayload)>> {
        self.ready_min(if all {
            self.remaining
        } else {
            self.remaining.min(1)
        })
    }
    pub(crate) fn ready_min(&mut self, minimum: usize) -> Result<Vec<(SlotLease, ReadPayload)>> {
        self.check_cancelled()?;
        ensure!(
            ready_minimum_valid(minimum, self.remaining),
            "invalid native SSD ready minimum"
        );
        let result = (|| {
            let mut completed = Vec::new();
            while self.remaining > 0 {
                let next = if completed.len() < minimum {
                    self.wait_next()?
                } else {
                    match self.receiver.try_recv() {
                        Ok(next) => next,
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(error) => return Err(error.into()),
                    }
                };
                self.check_cancelled()?;
                self.remaining -= 1;
                #[cfg(test)]
                self.trace.push((next.lease, next.timing.0, next.timing.1));
                completed.push((next.lease, next.result?));
            }
            Ok(completed)
        })();
        terminal_batch_result(result, || self.abandon())
    }
}
impl Drop for ReadBatch {
    fn drop(&mut self) {
        while self.remaining > 0 {
            if self.wait_next().is_err() {
                self.abandon();
                break;
            }
            self.remaining -= 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_support::{DemandPlan, ExpertCache};
    use std::{cell::RefCell, io::Write, rc::Rc};
    pub(super) fn lease() -> Result<SlotLease> {
        Ok(DemandPlan::reserve(&Rc::new(RefCell::new(ExpertCache::new(10)?)), &[17])?.reads[0])
    }
    #[test]
    fn early_enqueue_precedes_next_destination_and_drains_partial_failure() -> Result<()> {
        let pool = ReadService::new_with_schedule(2, false)?;
        let held = Arc::new(TestReadGate::default());
        *pool.held_first.borrow_mut() = Some(held.clone());
        let mut file = tempfile::tempfile()?;
        file.write_all(b"abc")?;
        let lease = lease()?;
        let mut queued_before_second = false;
        let reads = (0..2).map(|i| {
            if i == 1 {
                queued_before_second = pool.in_flight() == 1;
                held.release();
                anyhow::bail!("injected destination failure");
            }
            Ok((
                lease,
                file.try_clone()?,
                0,
                1,
                ReadPayload::Bytes(Vec::new()),
            ))
        });
        let result = pool.start_fallible_payloads(reads, Duration::from_secs(2), false, false);
        held.release();
        assert!(result.is_err() && queued_before_second);
        assert!(failure_tests::drained(&pool));
        *pool.held_first.borrow_mut() = None;
        let mut batch = pool.start(
            vec![(lease, file.try_clone()?, 1, 1)],
            Duration::from_secs(2),
        )?;
        let rows = batch.ready(true)?;
        assert!(matches!(&rows[0].1, ReadPayload::Bytes(v) if v == b"b"));
        for count in [0, 1, 10, 11, usize::MAX] {
            for owners in [0, 1, 10, usize::MAX] {
                for extents in [false, true] {
                    assert_eq!(
                        shared_plan_valid(count, owners, extents),
                        count <= 10 && count == owners && extents
                    );
                }
            }
        }
        Ok(())
    }
    #[test]
    fn reads_distinct_owned_ranges_and_drains_eof_before_reuse() -> Result<()> {
        for dynamic in [false, true] {
            let pool = ReadService::new_with_schedule(4, dynamic)?;
            let mut file = tempfile::tempfile()?;
            file.write_all(b"0123456789abcdef")?;
            let token = lease()?;
            let mut batch = pool.start(
                vec![
                    (token, file.try_clone()?, 2, 5),
                    (token, file.try_clone()?, 9, 7),
                ],
                Duration::from_secs(2),
            )?;
            let mut words = batch
                .ready(true)?
                .into_iter()
                .map(|(_, v)| v.legacy_bytes().to_vec())
                .collect::<Vec<_>>();
            words.sort();
            assert_eq!(words, vec![b"23456".to_vec(), b"9abcdef".to_vec()]);
            assert_eq!(batch.remaining(), 0);
            let mut broken = pool.start(
                vec![
                    (token, file.try_clone()?, 20, 1),
                    (token, file.try_clone()?, 0, 4),
                ],
                Duration::from_secs(2),
            )?;
            assert!(broken.ready(true).is_err());
            drop(broken);
            assert!(failure_tests::drained(&pool));
            let mut retry = pool.start(vec![(token, file, 0, 16)], Duration::from_secs(2))?;
            assert_eq!(retry.ready(true)?[0].1.legacy_bytes(), b"0123456789abcdef");
        }
        Ok(())
    }
    #[test]
    fn a_record_uses_the_first_tensor_offset_and_checks_the_full_pinned_extent() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("records.safetensors");
        let bytes = usize::try_from(EXPERT_BYTES)?;
        let header = serde_json::to_vec(&serde_json::json!({
            "first":{"dtype":"U8","shape":[4],"data_offsets":[0,4]},
            "rest":{"dtype":"U8","shape":[bytes-4],"data_offsets":[4,bytes]}
        }))?;
        let mut file = File::create(&path)?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(&header)?;
        file.write_all(&vec![71; bytes])?;
        let record = crate::checkpoint::read_header(&path)?["first"].clone();
        assert_eq!(record.data_offsets, [0, 4]);
        let pool = ReadService::new(1)?;
        let mut batch = pool.submit([(lease()?, record.clone())])?;
        let result = batch.ready(true)?;
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].1.legacy_bytes().len(), bytes);
        assert!(result[0].1.legacy_bytes().iter().all(|&v| v == 71));
        file.set_len((8 + header.len() + bytes - 1) as u64)?;
        assert!(pool.submit([(lease()?, record)]).is_err());
        Ok(())
    }
    #[test]
    fn cancellation_ack_detaches_before_physical_drain_and_blocks_reuse() -> Result<()> {
        let cancelled = Arc::new(AtomicBool::new(false));
        let mut pool = ReadService::new(1)?;
        pool.set_cancellation(cancelled.clone());
        let mut file = tempfile::tempfile()?;
        file.write_all(b"retained")?;
        let gate = Arc::new(std::sync::Barrier::new(2));
        pool.set_test_gate(Some(gate.clone()));
        let mut batch = pool.start(
            vec![(lease()?, file.try_clone()?, 0, 8)],
            Duration::from_secs(2),
        )?;
        let (started, entered) = mpsc::channel();
        batch.wait_started = Some(started);
        let (done, ack) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            let failed = batch.ready(true).is_err();
            drop(batch); // Must not wait for the held worker after logical cancel.
            let _ = done.send(failed);
        });
        let entered = entered.recv_timeout(Duration::from_secs(1));
        // Only the original worker is gated; a failed admission assertion must
        // not strand a second worker during cleanup.
        pool.set_test_gate(None);
        let at = Instant::now();
        cancelled.store(true, Ordering::Release);
        let acknowledged = ack.recv_timeout(Duration::from_millis(250));
        let ack_elapsed = at.elapsed();
        let retained = pool.in_flight();
        cancelled.store(false, Ordering::Release);
        let refused_demand = pool.start(Vec::new(), Duration::ZERO).is_err();
        let refused_background = pool
            .start_mode(
                vec![(lease()?, file.try_clone()?, 0, 8)],
                Duration::ZERO,
                true,
            )
            .is_err();
        // Always release and join, even when the acknowledgement assertion fails.
        gate.wait();
        waiter
            .join()
            .map_err(|_| anyhow::anyhow!("cancel waiter panicked"))?;
        let until = Instant::now() + Duration::from_secs(2);
        while pool.in_flight() != 0 && Instant::now() < until {
            std::thread::yield_now();
        }
        assert!(entered.is_ok());
        assert_eq!(acknowledged.ok(), Some(true));
        assert!(ack_elapsed < Duration::from_millis(250));
        assert_eq!(retained, 1);
        assert!(refused_demand && refused_background);
        assert_eq!(pool.in_flight(), 0);
        let mut retry = pool.start(vec![(lease()?, file, 0, 8)], Duration::from_secs(2))?;
        assert_eq!(retry.ready(true)?[0].1.legacy_bytes(), b"retained");
        Ok(())
    }

    #[test]
    fn queued_cancelled_completion_cannot_revive_after_flag_clear() -> Result<()> {
        let (sender, receiver) = mpsc::channel();
        let cancelled = Arc::new(AtomicBool::new(true));
        let mut batch = ReadBatch {
            io_cancelled: Arc::new(AtomicBool::new(false)),
            receiver,
            cancelled: cancelled.clone(),
            draining: Arc::new(AtomicBool::new(false)),
            abandoned: false,
            remaining: 1,
            deadline: Instant::now() + Duration::from_secs(2),
            trace: Vec::new(),
            #[cfg(test)]
            wait_started: None,
        };
        sender.send(Completed::new(lease()?, Ok(vec![7])))?;
        assert!(batch.ready(true).is_err());
        cancelled.store(false, Ordering::Release);
        assert!(batch.ready(true).is_err());
        assert!(batch.poll().is_err());
        assert!(sender.send(Completed::new(lease()?, Ok(vec![8]))).is_err());
        Ok(())
    }

    #[test]
    fn timeout_preserves_the_real_worker_and_rejects_reuse_until_drain() -> Result<()> {
        let pool = ReadService::new(1)?;
        let mut file = tempfile::tempfile()?;
        file.write_all(b"owned across cancellation")?;
        let gate = Arc::new(std::sync::Barrier::new(2));
        pool.set_test_gate(Some(gate.clone()));
        let mut pending = pool.start(vec![(lease()?, file.try_clone()?, 0, 5)], Duration::ZERO)?;
        let timed_out = pending.ready(true).is_err();
        drop(pending);
        let still_active = pool.in_flight() == 1;
        let refused_reuse = pool.submit(std::iter::empty()).is_err();
        pool.release_test_gate();
        let deadline = Instant::now() + Duration::from_secs(2);
        while pool.in_flight() != 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(timed_out && still_active && refused_reuse);
        assert_eq!(pool.in_flight(), 0);
        let mut retry = pool.start(vec![(lease()?, file, 0, 5)], Duration::from_secs(2))?;
        assert_eq!(retry.ready(true)?[0].1.legacy_bytes(), b"owned");
        Ok(())
    }
    #[test]
    fn refused_allocation_returns_a_failure_and_releases_the_reader() -> Result<()> {
        let pool = ReadService::new(1)?;
        let file = tempfile::tempfile()?;
        let mut batch = pool.start(
            vec![(lease()?, file, 0, usize::MAX)],
            Duration::from_secs(2),
        )?;
        assert!(batch.ready(true).is_err());
        drop(batch);
        assert_eq!(pool.in_flight(), 0);
        Ok(())
    }
    #[test]
    fn four_background_workers_poll_without_blocking_and_recover_after_timeout() -> Result<()> {
        let pool = ReadService::new(4)?;
        let mut file = tempfile::tempfile()?;
        file.write_all(b"owned background bytes")?;
        let gate = Arc::new(std::sync::Barrier::new(5));
        pool.set_test_gate(Some(gate.clone()));
        let token = lease()?;
        let mut pending = (0..4)
            .map(|_| {
                pool.start_mode(
                    vec![(token, file.try_clone()?, 0, 5)],
                    Duration::from_secs(2),
                    true,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let active = pool.in_flight();
        let refused = pool
            .start_mode(vec![(token, file.try_clone()?, 0, 5)], Duration::ZERO, true)
            .is_err();
        let polls_empty = pending.iter_mut().all(|r| matches!(r.poll(), Ok(None)));
        pool.release_test_gate();
        assert_eq!(active, 4);
        assert!(refused && polls_empty);
        for read in &mut pending {
            assert_eq!(read.ready(true)?[0].1.legacy_bytes(), b"owned");
        }
        assert_eq!(pool.in_flight(), 0);
        assert!(pool.start_mode(Vec::new(), Duration::ZERO, true).is_err());
        assert!(
            pool.start_mode(
                vec![
                    (token, file.try_clone()?, 0, 5),
                    (token, file.try_clone()?, 0, 5)
                ],
                Duration::ZERO,
                true
            )
            .is_err()
        );
        let gate = Arc::new(std::sync::Barrier::new(2));
        pool.set_test_gate(Some(gate));
        let mut timed = pool.start_mode(vec![(token, file, 0, 5)], Duration::ZERO, true)?;
        let failed = timed.poll().is_err();
        drop(timed);
        let retained = pool.in_flight();
        pool.release_test_gate();
        let deadline = Instant::now() + Duration::from_secs(2);
        while pool.in_flight() != 0 && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(failed && retained == 1 && pool.in_flight() == 0);
        Ok(())
    }
    #[test]
    fn unavailable_cohorts_and_reader_counts_fail_without_io() -> Result<()> {
        for count in [0, 5, usize::MAX] {
            assert!(ReadService::new(count).is_err());
        }
        let pool = ReadService::new(1)?;
        pool.active.store(1, Ordering::Release);
        assert!(pool.start(Vec::new(), Duration::from_millis(1)).is_err());
        pool.active.store(0, Ordering::Release);
        let mut empty = pool.submit(std::iter::empty())?;
        assert!(empty.ready(true)?.is_empty());
        Ok(())
    }
    #[test]
    fn reversed_completions_and_deadlines_keep_ownership() -> Result<()> {
        let token = lease()?;
        let (sender, receiver) = mpsc::channel();
        let mut batch = ReadBatch {
            io_cancelled: Arc::new(AtomicBool::new(false)),
            receiver,
            cancelled: Arc::new(AtomicBool::new(false)),
            draining: Arc::new(AtomicBool::new(false)),
            abandoned: false,
            remaining: 2,
            deadline: Instant::now() + Duration::from_secs(1),
            trace: Vec::new(),
            #[cfg(test)]
            wait_started: None,
        };
        sender.send(Completed::new(token, Ok(vec![2])))?;
        sender.send(Completed::new(token, Ok(vec![1])))?;
        assert_eq!(
            batch
                .ready(false)?
                .into_iter()
                .map(|(_, v)| v.legacy_bytes().to_vec())
                .collect::<Vec<_>>(),
            vec![vec![2], vec![1]]
        );
        let (sender, receiver) = mpsc::channel();
        let mut timed = ReadBatch {
            io_cancelled: Arc::new(AtomicBool::new(false)),
            receiver,
            cancelled: Arc::new(AtomicBool::new(false)),
            draining: Arc::new(AtomicBool::new(false)),
            abandoned: false,
            remaining: 1,
            deadline: Instant::now(),
            trace: Vec::new(),
            #[cfg(test)]
            wait_started: None,
        };
        assert!(timed.ready(true).is_err());
        assert_eq!(timed.remaining(), 0);
        assert!(sender.send(Completed::new(token, Ok(vec![9]))).is_err());
        assert!(timed.ready(true).is_err());
        drop(timed);
        Ok(())
    }
}

#[cfg(any(test, kani))]
#[path = "ssd_reads/direct_tests.rs"]
mod direct_tests;

#[cfg(test)]
#[path = "ssd_reads/failure_tests.rs"]
mod failure_tests;

#[cfg(test)]
#[path = "ssd_reads/cancellation_tests.rs"]
mod cancellation_tests;

#[cfg(any(test, kani))]
#[path = "ssd_reads/wave_tests.rs"]
mod wave_tests;

#[cfg(any(test, kani))]
#[path = "ssd_reads/common_queue_tests.rs"]
mod common_queue_tests;

#[cfg(any(test, kani))]
#[path = "ssd_reads/record_reuse_tests.rs"]
mod record_reuse_tests;

#[cfg(any(test, kani))]
#[path = "ssd_reads/split_record_tests.rs"]
mod split_record_tests;

#[path = "ssd_reads/window.rs"]
mod window;
#[cfg(all(target_os = "macos", feature = "direct-metal"))]
pub(crate) use window::BlockReads;

#[cfg(kani)]
#[kani::proof]
fn terminal_batch_error_stops_before_return_and_preserves_the_error() {
    let fails: bool = kani::any();
    let value: u32 = kani::any();
    let original = if fails { Err(value) } else { Ok(value) };
    let mut stopped = false;
    let result = terminal_batch_result(original, || stopped = true);
    assert_eq!(result, original);
    assert_eq!(stopped, fails);
    // After termination, a reset request flag cannot revive the batch.
    assert_eq!(latch_cancellation(&mut stopped, false), fails);
    kani::cover!(fails && value == u32::MAX);
    kani::cover!(!fails && value == 0);
}
