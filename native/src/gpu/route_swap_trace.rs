//! Test-only router opportunity capture, never a throughput measurement.
use anyhow::{Context, Result, ensure};
use std::{
    cell::{Cell, RefCell},
    io::Write,
    path::PathBuf,
    rc::Rc,
};

#[derive(serde::Serialize)]
struct Sample {
    document: usize,
    layer: usize,
    position: usize,
    ids: Vec<u32>,
    weights: Vec<f32>,
    ready: Vec<u32>,
    actual_hits: Option<usize>,
    actual_read_bytes: Option<u64>,
    #[serde(skip)]
    probabilities: Vec<f32>,
}
#[derive(serde::Serialize)]
struct Document {
    name: String,
    input_rows: usize,
}
struct State {
    output: PathBuf,
    documents: Vec<Document>,
    positions: [usize; 48],
    samples: Vec<Sample>,
}
thread_local! {
    static STATE:RefCell<Option<State>>=const {RefCell::new(None)};
    static LAYER:Cell<Option<usize>>=const {Cell::new(None)};
    static QUALITY:RefCell<Option<QualityState>>=const {RefCell::new(None)};
    static DRAFT:Cell<bool>=const {Cell::new(false)};
    static VERIFY_BLOCK:Cell<bool>=const {Cell::new(false)};
    static PREPARING:Cell<bool>=const {Cell::new(false)};
}

#[derive(Clone)]
pub(super) struct DecisionSnapshot {
    positions: [usize; 48],
    starved: [[u8; 512]; 48],
    caches: [Option<Rc<crate::flash_support::ExpertCache>>; 48],
}
#[derive(Clone)]
struct DecisionFrame {
    position: usize,
    starved: [u8; 512],
    cache: Option<Rc<crate::flash_support::ExpertCache>>,
}
impl DecisionSnapshot {
    pub(super) fn retain_layer(&mut self, layer: usize, row: usize) -> Result<()> {
        QUALITY.with_borrow(|s| -> Result<()> {
            let frame = s.as_ref().context("missing Q04 prefix state")?.frames[layer]
                .get(row)
                .context("missing Q04 layer prefix")?;
            self.positions[layer] = frame.position;
            self.starved[layer] = frame.starved;
            self.caches[layer] = frame.cache.clone();
            Ok(())
        })
    }
}
pub(super) fn decision_snapshot() -> Option<DecisionSnapshot> {
    QUALITY.with_borrow(|s| {
        s.as_ref().map(|s| DecisionSnapshot {
            positions: s.positions,
            starved: s.starved,
            caches: s.caches.clone(),
        })
    })
}
pub(super) fn restore_decisions(snapshot: Option<&DecisionSnapshot>) {
    QUALITY.with_borrow_mut(|s| {
        if let (Some(s), Some(snapshot)) = (s, snapshot) {
            s.positions = snapshot.positions;
            s.starved = snapshot.starved;
            s.caches = snapshot.caches.clone();
        }
    });
}
pub(super) fn restore_cache_only(snapshot: Option<&DecisionSnapshot>) {
    QUALITY.with_borrow_mut(|s| {
        if let (Some(s), Some(snapshot)) = (s, snapshot) {
            s.caches = snapshot.caches.clone();
        }
    });
}
pub(super) fn register_cache(
    layer: usize,
    cache: impl FnOnce() -> crate::flash_support::ExpertCache,
) {
    QUALITY.with_borrow_mut(|s| {
        if let Some(s) = s
            && s.caches[layer].is_none()
        {
            s.caches[layer] = Some(Rc::new(cache()));
        }
    });
}
pub(super) struct BlockScope(bool);
impl BlockScope {
    pub(super) fn new() -> Self {
        QUALITY.with_borrow_mut(|s| {
            if let Some(s) = s {
                s.frames = std::array::from_fn(|_| Vec::new());
            }
        });
        Self(VERIFY_BLOCK.replace(true))
    }
}
impl Drop for BlockScope {
    fn drop(&mut self) {
        VERIFY_BLOCK.set(self.0);
    }
}
pub(super) struct Preparing(bool);
impl Preparing {
    pub(super) fn new() -> Self {
        Self(PREPARING.replace(true))
    }
}
impl Drop for Preparing {
    fn drop(&mut self) {
        PREPARING.set(self.0);
    }
}

pub(super) struct SuspendDraft(bool);
impl SuspendDraft {
    pub(super) fn new() -> Self {
        Self(DRAFT.replace(true))
    }
}
impl Drop for SuspendDraft {
    fn drop(&mut self) {
        DRAFT.set(self.0);
    }
}

struct QualityState {
    tau: f32,
    limit: usize,
    prefix: usize,
    inputs: usize,
    positions: [usize; 48],
    starved: [[u8; 512]; 48],
    caches: [Option<Rc<crate::flash_support::ExpertCache>>; 48],
    frames: [Vec<DecisionFrame>; 48],
    rows: usize,
    swaps: usize,
    cold_reads: usize,
    read_bytes: u64,
}

/// Intrusive online AR quality experiment. Never enabled by a production binary.
pub(super) struct Quality;
impl Quality {
    pub(super) fn start(tau: Option<f32>, limit: usize) -> Result<Self> {
        if let Some(tau) = tau {
            ensure!(tau.is_finite() && (0.0..=0.4).contains(&tau) && (1..=3).contains(&limit));
            ensure!(
                !wanted(),
                "quality swaps cannot share an exact route capture"
            );
            QUALITY.with_borrow_mut(|s| -> Result<()> {
                ensure!(s.is_none(), "nested Q04 quality experiment");
                *s = Some(QualityState {
                    tau,
                    limit,
                    prefix: 0,
                    inputs: 0,
                    positions: [0; 48],
                    starved: [[0; 512]; 48],
                    caches: std::array::from_fn(|_| None),
                    frames: std::array::from_fn(|_| Vec::new()),
                    rows: 0,
                    swaps: 0,
                    cold_reads: 0,
                    read_bytes: 0,
                });
                Ok(())
            })?;
        }
        Ok(Self)
    }
    pub(super) fn document(&self, prefix: usize, inputs: usize) -> Result<()> {
        QUALITY.with_borrow_mut(|s| -> Result<()> {
            if let Some(s) = s {
                ensure!(s.positions.iter().all(|&n| n == s.inputs));
                ensure!(prefix > 0 && prefix < inputs);
                s.positions = [0; 48];
                s.starved = [[0; 512]; 48];
                s.prefix = prefix;
                s.inputs = inputs;
            }
            Ok(())
        })
    }
    pub(super) fn finish(&self) -> Result<Option<serde_json::Value>> {
        QUALITY.with_borrow_mut(|s| -> Result<_> {
            s.take().map(|s| -> Result<_> {
                ensure!(s.positions.iter().all(|&n| n == s.inputs));
                ensure!(s.rows > 0);
                Ok(serde_json::json!({"kind":"online_ar_diagnostic","tau":s.tau,"max_swaps":s.limit,
                    "protected":3,"weight_cap":0.1,"rows":s.rows,"swaps":s.swaps,
                    "actual_decode_cold_reads":s.cold_reads,"actual_decode_logical_read_bytes":s.read_bytes,
                    "retained_layer_rows":s.positions.iter().map(|p|p-s.prefix).sum::<usize>(),
                    "timing_is_benchmark":false,"mtp_validated":false}))
            }).transpose()
        })
    }
}
impl Drop for Quality {
    fn drop(&mut self) {
        QUALITY.with_borrow_mut(|s| *s = None);
    }
}

pub(super) fn quality_row() -> Result<bool> {
    if DRAFT.get() {
        return Ok(false);
    }
    QUALITY.with_borrow_mut(|s| -> Result<_> {
        let Some(s) = s else {
            return Ok(false);
        };
        let layer = LAYER
            .get()
            .context("Q04 quality requires actual layer scope")?;
        ensure!(layer < 48 && s.positions[layer] < s.inputs);
        let decode = s.positions[layer] >= s.prefix;
        s.positions[layer] += 1;
        Ok(decode)
    })
}
pub(super) fn quality_active() -> bool {
    !DRAFT.get() && QUALITY.with_borrow(|s| s.is_some())
}

pub(super) fn encode_pick(
    gpu: &super::Metal,
    cmd: &metal::CommandBufferRef,
    routes: &mut super::RouteOutput,
    ready_ids: impl Iterator<Item = u32>,
) -> Result<()> {
    if PREPARING.get() {
        return Ok(());
    }
    QUALITY.with_borrow(|s| -> Result<_> {
        let Some(s) = s else {
            return Ok(());
        };
        let layer = LAYER.get().context("Q04 encoder requires layer scope")?;
        if s.positions[layer] < s.prefix {
            return Ok(());
        }
        let mut ready = [0u32; 512];
        let logical = s.caches[layer]
            .as_ref()
            .context("Q04 logical cache not initialized")?
            .ready_slot_snapshot();
        let _physical_ready = ready_ids;
        for (e, _) in logical.into_iter().flatten() {
            ensure!(e < 512);
            ready[e as usize] = 1;
        }
        routes.q04_pick = Some(encode_picker(
            gpu,
            cmd,
            routes,
            &ready,
            &s.starved[layer],
            s.tau,
            s.limit,
        )?);
        Ok(())
    })
}

fn encode_picker(
    gpu: &super::Metal,
    cmd: &metal::CommandBufferRef,
    routes: &super::RouteOutput,
    ready: &[u32; 512],
    starved: &[u8; 512],
    tau: f32,
    limit: usize,
) -> Result<(super::MatrixView, [metal::Buffer; 2])> {
    use super::view::FloatType;
    let p = routes
        .q04_probabilities
        .as_ref()
        .context("missing canonical Q04 probabilities")?;
    ensure!(
        p.shape() == [1, 512] && routes.scores.shape() == [1, 10] && p.dtype == routes.scores.dtype
    );
    ensure!(tau.is_finite() && (0.0..=0.4).contains(&tau) && (1..=3).contains(&limit));
    let ready = super::upload(&gpu.device, ready)?;
    let starved = super::upload(&gpu.device, starved)?;
    // Raw u32 words; this view never participates in floating-point arithmetic.
    let picked = super::MatrixView::output(
        gpu,
        FloatType::F32,
        1,
        10,
        metal::MTLResourceOptions::StorageModeShared,
    )?;
    let pipeline = &gpu.q04_pickers[usize::from(p.dtype == FloatType::F32)];
    ensure!(
        pipeline.thread_execution_width() == 32
            && pipeline.max_total_threads_per_threadgroup() >= 128
    );
    super::Metal::encode(
        cmd,
        pipeline,
        &[
            &p.buffer,
            &routes.scores.buffer,
            routes.device_indices(),
            &ready,
            &starved,
            &picked.buffer,
        ],
        [(1.0 - tau).to_bits(), limit as u32, u32::from(tau > 0.0)],
        metal::MTLSize::new(1, 1, 1),
        128,
    );
    Ok((picked, [ready, starved]))
}

#[test]
#[ignore = "physical Apple GPU; independent F16/F32 Q04 selection oracle"]
fn gpu_picker_matches_reference_without_changing_original_routes() -> Result<()> {
    use super::{MatrixView, Metal, view::FloatType};
    use crate::tensor_layout::MatrixLayout;
    let gpu = Metal::new()?;
    for half in [false, true] {
        let matrix = |values: &[f32], cols| -> Result<MatrixView> {
            let layout = MatrixLayout {
                shape: [1, cols],
                strides: [i64::from(cols), 1],
                offset: 0,
            };
            if half {
                let words: Vec<_> = values
                    .iter()
                    .map(|&v| half::f16::from_f32(v).to_bits())
                    .collect();
                gpu.f16_storage(&words)?.matrix(layout)
            } else {
                gpu.f32_storage(values)?.matrix(layout)
            }
        };
        for case in 0..6 {
            let mut p = [1.0 / 1024.0; 512];
            p[..10].fill(2.0 / 512.0);
            p[10..13].fill(1.9 / 512.0);
            if case == 4 {
                p.fill(1.0 / 512.0);
            }
            if case == 5 {
                for (i, value) in p[..10].iter_mut().enumerate() {
                    *value = (2.0 - i as f32 * 0.01) / 512.0;
                }
                p[10..13].fill(1.81 / 512.0);
            }
            let p = matrix(&p, 512)?;
            let mut route = gpu.route(&p, 10, true)?;
            let ids = route.indices()?;
            let original: Vec<_> = if case == 4 {
                (502..512).collect()
            } else if case == 5 {
                (0..10).rev().collect()
            } else {
                (0..10).collect()
            };
            ensure!(ids == original, "fixture top-k order changed");
            let mut weights = [0.1; 10];
            if case == 3 {
                weights[4] = 0.11;
            }
            route.scores = matrix(&weights, 10)?;
            route.q04_probabilities = Some(p.clone());
            let mut ready = [0u32; 512];
            for e in [3, 10, 11, 12] {
                ready[e] = 1;
            }
            let mut starved = [0u8; 512];
            if case == 2 {
                starved[4] = 3;
            }
            let tau = if case == 1 {
                0.0
            } else if case == 4 {
                f32::EPSILON / 4.0
            } else {
                0.2
            };
            let cmd = gpu.queue.new_command_buffer();
            let (picked, _held) = encode_picker(&gpu, cmd, &route, &ready, &starved, tau, 3)?;
            cmd.commit();
            cmd.wait_until_completed();
            ensure!(cmd.status() == metal::MTLCommandBufferStatus::Completed);
            let convert = |v: Vec<u32>| {
                v.into_iter()
                    .map(|b| {
                        if p.dtype == FloatType::F32 {
                            f32::from_bits(b)
                        } else {
                            half::f16::from_bits(b as u16).to_f32()
                        }
                    })
                    .collect::<Vec<_>>()
            };
            let probabilities = convert(p.read_output()?);
            let original_words = route.scores.read_output()?;
            let weights = convert(original_words.clone());
            let mut ready_bool = [false; 512];
            for i in 0..512 {
                ready_bool[i] = ready[i] != 0;
            }
            let mut state = starved;
            let picks = crate::route_swap::select(
                &ids,
                &probabilities,
                &weights,
                &ready_bool,
                &mut state,
                tau,
                3,
            )?;
            let mut expected = ids.clone();
            for (rank, e) in picks {
                expected[rank] = e;
            }
            if case == 0 {
                ensure!(expected == vec![0, 1, 2, 3, 10, 11, 12, 7, 8, 9]);
            }
            if case == 4 {
                ensure!(expected == vec![502, 503, 504, 3, 10, 11, 508, 509, 510, 511]);
            }
            if case == 5 {
                ensure!(expected == vec![9, 8, 7, 12, 11, 10, 3, 2, 1, 0]);
            }
            if case == 1 {
                ensure!(expected == ids);
            }
            ensure!(
                picked.read_output()? == expected,
                "GPU picker mismatch, half={half},case={case}"
            );
            ensure!(route.indices()? == ids && route.scores.read_output()? == original_words);
        }
    }
    Ok(())
}
pub(super) fn quality_select(
    ids: &[u32],
    p: &[f32],
    weights: &[f32],
    _ready: &[bool; 512],
) -> Result<Vec<u32>> {
    QUALITY.with_borrow_mut(|s| -> Result<_> {
        let s = s.as_mut().context("inactive Q04 quality")?;
        let layer = LAYER.get().context("missing Q04 layer")?;
        let mut ready = [false; 512];
        for (e, _) in s.caches[layer]
            .as_ref()
            .context("missing Q04 logical cache")?
            .ready_slot_snapshot()
            .into_iter()
            .flatten()
        {
            ready[e as usize] = true;
        }
        let picks = crate::route_swap::select(
            ids,
            p,
            weights,
            &ready,
            &mut s.starved[layer],
            s.tau,
            s.limit,
        )?;
        let mut ids = ids.to_vec();
        for &(rank, e) in &picks {
            ids[rank] = e;
        }
        s.swaps += picks.len();
        Ok(ids)
    })
}
pub(super) fn quality_complete(ids: &[u32], hits: usize, read_bytes: u64) -> Result<()> {
    if DRAFT.get() {
        return Ok(());
    }
    QUALITY.with_borrow_mut(|s| -> Result<_> {
        if let Some(s) = s {
            let layer = LAYER.get().context("missing Q04 completion layer")?;
            let cache = s.caches[layer]
                .as_mut()
                .context("missing Q04 logical completion cache")?;
            Rc::make_mut(cache).reserve(ids)?;
            if VERIFY_BLOCK.get() {
                s.frames[layer].push(DecisionFrame {
                    position: s.positions[layer],
                    starved: s.starved[layer],
                    cache: s.caches[layer].clone(),
                });
            }
            if s.positions[layer] > s.prefix {
                ensure!(hits <= 10 && read_bytes == (10 - hits) as u64 * 1_248_012);
                s.rows += 1;
                s.cold_reads += 10 - hits;
                s.read_bytes += read_bytes;
            }
        }
        Ok(())
    })
}

pub(super) fn block_pick(gpu: &super::Metal, route: &mut super::RouteOutput) -> Result<()> {
    if !quality_active() {
        return Ok(());
    }
    let decode = quality_row()?;
    let original = route.original_indices()?;
    let ids = if decode {
        let probabilities = route
            .q04_probabilities
            .as_ref()
            .context("missing block probabilities")?;
        let convert = |words: Vec<u32>| {
            words
                .into_iter()
                .map(|word| {
                    if probabilities.dtype == super::view::FloatType::F32 {
                        f32::from_bits(word)
                    } else {
                        half::f16::from_bits(word as u16).to_f32()
                    }
                })
                .collect::<Vec<_>>()
        };
        quality_select(
            &original,
            &convert(probabilities.read_output()?),
            &convert(route.completed_scores()?.read_output()?),
            &[false; 512],
        )?
    } else {
        original
    };
    let buffer = super::upload(&gpu.device, &ids)?;
    let picked = super::MatrixView {
        buffer: buffer.clone(),
        dtype: super::view::FloatType::F32,
        layout: crate::tensor_layout::MatrixLayout {
            shape: [1, 10],
            strides: [10, 1],
            offset: 0,
        },
    };
    route.q04_pick = Some((picked, [buffer.clone(), buffer]));
    QUALITY.with_borrow_mut(|s| -> Result<()> {
        let s = s.as_mut().context("missing block Q04 state")?;
        let layer = LAYER.get().context("missing block Q04 layer")?;
        Rc::make_mut(
            s.caches[layer]
                .as_mut()
                .context("missing logical block cache")?,
        )
        .reserve(&ids)?;
        s.frames[layer].push(DecisionFrame {
            position: s.positions[layer],
            starved: s.starved[layer],
            cache: s.caches[layer].clone(),
        });
        Ok(())
    })
}
pub(super) fn block_stats(read_bytes: u64) -> Result<()> {
    QUALITY.with_borrow_mut(|s| -> Result<()> {
        if let Some(s) = s {
            ensure!(read_bytes % 1_248_012 == 0);
            s.rows += 1;
            s.cold_reads += (read_bytes / 1_248_012) as usize;
            s.read_bytes += read_bytes;
        }
        Ok(())
    })
}

pub(super) struct LayerScope {
    old: Option<usize>,
    enabled: bool,
}
impl LayerScope {
    pub(super) fn new(layer: usize) -> Self {
        let enabled = STATE.with_borrow(|s| s.is_some()) || QUALITY.with_borrow(|s| s.is_some());
        Self {
            old: if enabled {
                LAYER.replace(Some(layer))
            } else {
                None
            },
            enabled,
        }
    }
}
impl Drop for LayerScope {
    fn drop(&mut self) {
        if self.enabled {
            LAYER.set(self.old);
        }
    }
}

pub(super) struct Capture {
    active: bool,
}
impl Capture {
    pub(super) fn start(output: Option<PathBuf>) -> Result<Self> {
        let Some(output) = output else {
            return Ok(Self { active: false });
        };
        std::fs::create_dir_all(&output)?;
        STATE.with_borrow_mut(|s| -> Result<()> {
            ensure!(s.is_none(), "nested route capture");
            *s = Some(State {
                output,
                documents: Vec::new(),
                positions: [0; 48],
                samples: Vec::new(),
            });
            Ok(())
        })?;
        Ok(Self { active: true })
    }
    pub(super) fn active(&self) -> bool {
        self.active
    }
    pub(super) fn document(&self, index: usize, name: &str, inputs: usize) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        STATE.with_borrow_mut(|s| -> Result<()> {
            let s = s.as_mut().context("inactive capture")?;
            ensure!(index == s.documents.len() && inputs > 0);
            if let Some(previous) = s.documents.last() {
                ensure!(
                    s.positions.iter().all(|&n| n == previous.input_rows),
                    "missing layer routes"
                );
            }
            let total = s
                .documents
                .iter()
                .try_fold(inputs, |n, d| n.checked_add(d.input_rows))
                .context("input count overflow")?;
            ensure!(
                crate::native_executor::calibration_sample_count(total).is_some(),
                "route capture budget"
            );
            s.documents.push(Document {
                name: name.into(),
                input_rows: inputs,
            });
            s.positions = [0; 48];
            Ok(())
        })
    }
    pub(super) fn finish(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }
        let s = STATE
            .with_borrow_mut(Option::take)
            .context("missing route capture")?;
        self.active = false;
        let last = s.documents.last().context("empty route capture")?;
        ensure!(s.positions.iter().all(|&n| n == last.input_rows));
        let inputs = s
            .documents
            .iter()
            .try_fold(0usize, |n, d| n.checked_add(d.input_rows))
            .context("input count overflow")?;
        ensure!(crate::native_executor::calibration_sample_count(inputs) == Some(s.samples.len()));
        ensure!(
            !s.samples.is_empty()
                && s.samples
                    .iter()
                    .all(|v| v.actual_hits.is_some() && v.actual_read_bytes.is_some())
        );
        let mut binary =
            std::io::BufWriter::new(std::fs::File::create(s.output.join("probabilities.f32"))?);
        for sample in &s.samples {
            ensure!(sample.probabilities.len() == 512);
            for value in &sample.probabilities {
                binary.write_all(&value.to_le_bytes())?;
            }
        }
        binary.flush()?;
        let report = serde_json::json!({"status":"complete_diagnostic_only","probability_format":"little-endian f32,512values/sample in metadata order","sample_count":s.samples.len(),"documents":s.documents,"samples":s.samples,"timing_is_benchmark":false});
        std::fs::write(
            s.output.join("trace.json.tmp"),
            serde_json::to_vec(&report)?,
        )?;
        std::fs::rename(s.output.join("trace.json.tmp"), s.output.join("trace.json"))?;
        Ok(())
    }
}
impl Drop for Capture {
    fn drop(&mut self) {
        if self.active {
            STATE.with_borrow_mut(|s| *s = None);
            LAYER.set(None);
        }
    }
}

pub(super) fn wanted() -> bool {
    LAYER.get().is_some() && STATE.with_borrow(|s| s.is_some())
}
pub(super) fn record(
    ids: &[u32],
    weights: Vec<f32>,
    ready: Vec<u32>,
    probabilities: Vec<f32>,
) -> Result<usize> {
    let layer = LAYER.get().context("missing trace layer")?;
    ensure!(layer < 48 && ids.len() == 10 && weights.len() == 10 && probabilities.len() == 512);
    ensure!(ids.iter().all(|&v| v < 512) && ready.iter().all(|&v| v < 512));
    ensure!(ids.iter().enumerate().all(|(i, id)| !ids[..i].contains(id)));
    ensure!(
        weights
            .iter()
            .chain(&probabilities)
            .all(|v| v.is_finite() && *v >= 0.)
    );
    let weight_sum = weights.iter().sum::<f32>();
    ensure!((0.98..=1.02).contains(&weight_sum));
    let sum = probabilities.iter().sum::<f32>();
    ensure!((0.98..=1.02).contains(&sum));
    STATE.with_borrow_mut(|state| -> Result<usize> {
        let s = state.as_mut().context("inactive capture")?;
        ensure!(s.samples.len() < 32768);
        let document = s
            .documents
            .len()
            .checked_sub(1)
            .context("missing trace document")?;
        let position = s.positions[layer];
        ensure!(position < s.documents[document].input_rows);
        s.positions[layer] += 1;
        let index = s.samples.len();
        s.samples.push(Sample {
            document,
            layer,
            position,
            ids: ids.to_vec(),
            weights,
            ready,
            probabilities,
            actual_hits: None,
            actual_read_bytes: None,
        });
        Ok(index)
    })
}
pub(super) fn complete(index: usize, hits: usize, read_bytes: u64) -> Result<()> {
    STATE.with_borrow_mut(|s| -> Result<()> {
        let sample = s
            .as_mut()
            .context("inactive capture")?
            .samples
            .get_mut(index)
            .context("invalid sample")?;
        ensure!(sample.actual_hits.is_none() && hits <= 10);
        ensure!(
            sample
                .ids
                .iter()
                .filter(|id| sample.ready.contains(id))
                .count()
                == hits,
            "ready/miss trace mismatch"
        );
        sample.actual_hits = Some(hits);
        sample.actual_read_bytes = Some(read_bytes);
        Ok(())
    })
}
