//! Test-only linear depth sweep. Verification stays within existing 1..3-row blocks.
use super::*;
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::time::Instant;

pub(in crate::gpu) struct Pilot {
    depth: usize,
    block: usize,
    pending: VecDeque<u32>,
    expected_anchor: u32,
    pending_prefix: Option<(u32, MatrixView)>,
    widen: Option<HiddenWiden>,
    cancel_before_publish: bool,
    forced_proposals: Option<[u32; 4]>,
    pub(in crate::gpu) stats: Stats,
    pub(in crate::gpu) proposed: usize,
    pub(in crate::gpu) accepted: usize,
    pub(in crate::gpu) widths: [usize; 4],
    pub(in crate::gpu) lengths: [usize; 4],
}
impl Pilot {
    pub(in crate::gpu) fn new(depth: usize, block: usize, gpu: &Metal) -> Result<Self> {
        ensure!((1..=4).contains(&depth) && (1..=3).contains(&block));
        let recursive_cpu = match std::env::var("MLXL3_NATIVE_MTP_RECURSIVE_CPU") {
            Ok(value) => crate::flash_support::hc_chain_flag(Some(&value), false)
                .context("MLXL3_NATIVE_MTP_RECURSIVE_CPU must be 0 or 1")?,
            Err(std::env::VarError::NotPresent) => false,
            Err(error) => return Err(error.into()),
        };
        Ok(Self {
            depth,
            block,
            pending: VecDeque::new(),
            expected_anchor: 0,
            pending_prefix: None,
            widen: if depth > 1 && !recursive_cpu {
                Some(HiddenWiden::new(gpu)?)
            } else {
                None
            },
            cancel_before_publish: false,
            forced_proposals: None,
            stats: Stats::default(),
            proposed: 0,
            accepted: 0,
            widths: [0; 4],
            lengths: [0; 4],
        })
    }
    pub(in crate::gpu) fn advance(
        &mut self,
        target: &mut FlashNext,
        head: &mut FlashMtpHead,
        anchor: u32,
        remaining: usize,
        eos: &[u32],
        cancelled: &std::sync::atomic::AtomicBool,
    ) -> Result<u32> {
        ensure!(!cancelled.load(std::sync::atomic::Ordering::Relaxed) && remaining > 0);
        if let Some(token) = self.pending.front().copied() {
            let (offset, row) = self
                .pending_prefix
                .as_ref()
                .context("missing pending MTP prefix")?;
            ensure!(
                anchor == self.expected_anchor
                    && target.offset() == *offset
                    && target
                        .last_hidden()
                        .is_some_and(|h| std::ptr::eq::<metal::BufferRef>(
                            h.buffer.as_ref(),
                            row.buffer.as_ref()
                        ))
            );
            ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
            self.pending.pop_front();
            self.expected_anchor = token;
            if self.pending.is_empty() {
                self.pending_prefix = None;
            }
            return Ok(token);
        }
        ensure!(head.layer.offset() + 1 == target.offset());
        let available = target
            .context
            .checked_sub(target.offset())
            .context("target offset beyond context")? as usize;
        let depth = crate::speculative::bounded_proposals(available, remaining)
            .context("no target context remaining")?
            .min(self.depth);
        let root = target.snapshot();
        let draft_root = head.layer.snapshot();
        let before_hidden = target
            .last_hidden()
            .context("missing MTP target hidden")?
            .clone();
        let mut round = Stats::default();
        let mut widths = [0; 4];
        let operation = (|| -> Result<(Vec<u32>, usize, usize)> {
            if depth == 0 {
                let start = Instant::now();
                head.extend(target, &before_hidden, anchor)?;
                round.repair_seconds += start.elapsed().as_secs_f64();
                round.kv_repair_rows += 1;
                let start = Instant::now();
                let next = target.forward_greedy(anchor)?;
                round.target_seconds += start.elapsed().as_secs_f64();
                round.target_executed_rows += 1;
                round.target_retained_rows += 1;
                round.tail_steps += 1;
                return Ok((vec![next], 0, 0));
            }
            let mut proposals = Vec::with_capacity(depth);
            let mut draft_hidden = before_hidden.clone();
            let mut token = anchor;
            let mut first_draft = None;
            let draft_start = Instant::now();
            for step in 0..depth {
                ensure!(!cancelled.load(std::sync::atomic::Ordering::Relaxed));
                let (next, hidden) = propose_hidden(
                    head,
                    target,
                    &draft_hidden,
                    token,
                    step + 1 < depth,
                    self.widen.as_ref(),
                    &mut round,
                )?;
                let next = self.forced_proposals.map_or(next, |ids| ids[step]);
                if step == 0 {
                    first_draft = Some(head.layer.snapshot());
                }
                proposals.push(next);
                draft_hidden = hidden;
                token = next;
                if eos.contains(&next) {
                    break;
                }
            }
            round.draft_seconds += draft_start.elapsed().as_secs_f64();
            let n = proposals.len();
            let mut inputs = vec![anchor];
            inputs.extend_from_slice(&proposals);
            let mut outputs = Vec::with_capacity(n + 1);
            let mut hidden = Vec::with_capacity(n + 1);
            let mut accepted = 0;
            let mut at = 0;
            let mut stopped = false;
            while at < inputs.len() && !stopped {
                let end = (at + self.block).min(inputs.len());
                widths[end - at - 1] += 1;
                round.target_executed_rows += end - at;
                let start = Instant::now();
                let mut staged = target.stage_block_headless(&inputs[at..end], cancelled)?;
                round.target_seconds += start.elapsed().as_secs_f64();
                let mut keep = 0;
                for row in 0..end - at {
                    ensure!(!cancelled.load(std::sync::atomic::Ordering::Relaxed));
                    let start = Instant::now();
                    let next = staged.greedy(row + 1, true)?;
                    round.head_seconds += start.elapsed().as_secs_f64();
                    hidden.push(
                        staged
                            .prefix_hidden(row + 1)
                            .context("missing verified draft repair row")?
                            .clone(),
                    );
                    outputs.push(next);
                    keep += 1;
                    let index = at + row;
                    if index < n {
                        round.consulted_by_rank[index] += 1;
                        if next == proposals[index] {
                            accepted += 1;
                            round.accepted_by_rank[index] += 1;
                        } else {
                            stopped = true;
                            round.rejected_by_rank[index] += 1;
                        }
                    }
                    if eos.contains(&next) {
                        round.eos += 1;
                        stopped = true;
                    }
                    if stopped {
                        break;
                    }
                }
                staged.commit(keep)?;
                round.target_retained_rows += keep;
                at += keep;
            }
            let repair_start = Instant::now();
            head.layer
                .restore(&first_draft.context("empty draft chain")?)?;
            for index in 1..outputs.len() {
                head.extend(target, &hidden[index - 1], inputs[index])?;
                round.kv_repair_rows += 1;
            }
            round.repair_seconds += repair_start.elapsed().as_secs_f64();
            round.accepted_prefix[accepted] += 1;
            round.rounds += 1;
            ensure!(head.layer.offset() + 1 == target.offset() && outputs.len() <= remaining);
            Ok((outputs, n, accepted))
        })();
        if self.cancel_before_publish {
            cancelled.store(true, Ordering::Relaxed);
        }
        let operation = crate::speculative::publication_result(
            operation,
            cancelled.load(Ordering::Relaxed),
            anyhow::anyhow!("generation cancelled"),
        );
        match operation {
            Ok((outputs, proposed, accepted)) => {
                ensure!(!outputs.is_empty());
                self.stats.add(&round);
                for (total, count) in self.widths.iter_mut().zip(widths) {
                    *total += count;
                }
                self.proposed += proposed;
                self.accepted += accepted;
                if proposed > 0 {
                    self.lengths[proposed - 1] += 1;
                }
                let first = outputs[0];
                self.pending = outputs[1..].iter().copied().collect();
                self.expected_anchor = first;
                if !self.pending.is_empty() {
                    self.pending_prefix = Some((
                        target.offset(),
                        target
                            .last_hidden()
                            .context("missing pending prefix hidden")?
                            .clone(),
                    ));
                }
                Ok(first)
            }
            Err(error) => {
                let a = target.restore(&root);
                let b = head.layer.restore(&draft_root);
                a?;
                b?;
                Err(error)
            }
        }
    }
}

fn propose_hidden(
    head: &mut FlashMtpHead,
    target: &FlashNext,
    hidden: &MatrixView,
    token: u32,
    need_hidden: bool,
    widen: Option<&HiddenWiden>,
    stats: &mut Stats,
) -> Result<(u32, MatrixView)> {
    let _draft = super::super::super::route_swap_trace::SuspendDraft::new();
    let streams = head.mixed(target, hidden, token)?;
    let output = head.layer.forward(&streams)?;
    let mixed = head.mixer.read(&output, false)?.mixed;
    stats.draft_forwards += 1;
    stats.draft_heads += 1;
    let gpu_widen = need_hidden && output.dtype == FloatType::F16 && widen.is_some();
    let (token, _) = if gpu_widen {
        let widen = widen.context("missing recursive GPU scratch")?;
        widen.reset();
        let selected = target.gpu.command(|cmd| {
            let logits = target.head.encode_row(
                cmd,
                &mixed,
                metal::MTLResourceOptions::StorageModePrivate,
            )?;
            let selected = target.gpu.encode_device_greedy(cmd, &logits.buffer)?;
            widen.encode(target.gpu, cmd, &output)?;
            Ok(selected)
        })?;
        widen.check()?;
        stats.gpu_conversions += 1;
        target
            .gpu
            .device_greedy_heads
            .set(target.gpu.device_greedy_heads.get() + 1);
        (target.gpu.read_device_greedy(&selected)?, None::<Vec<u16>>)
    } else {
        target.gpu.device_greedy_head(&target.head, &mixed, false)?
    };
    if !need_hidden {
        return Ok((token, hidden.clone()));
    }
    if output.dtype == FloatType::F32 {
        return Ok((token, output));
    }
    if gpu_widen {
        return Ok((
            token,
            widen.context("missing widening scratch")?.output.clone(),
        ));
    }
    stats.cpu_conversions += 1;
    let bytes = output.export_bytes(target.gpu)?;
    ensure!(bytes.len() == 10240 * 2);
    let values = bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| half::f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite recursive draft hidden"
    );
    let output = target.gpu.f32_storage(&values)?.matrix(MatrixLayout {
        shape: [1, 10240],
        strides: [10240, 1],
        offset: 0,
    })?;
    Ok((token, output))
}

#[derive(Default, Clone, Debug, PartialEq, serde::Serialize)]
pub(in crate::gpu) struct Stats {
    draft_forwards: usize,
    draft_heads: usize,
    gpu_conversions: usize,
    cpu_conversions: usize,
    kv_repair_rows: usize,
    target_executed_rows: usize,
    target_retained_rows: usize,
    rounds: usize,
    tail_steps: usize,
    eos: usize,
    consulted_by_rank: [usize; 4],
    accepted_by_rank: [usize; 4],
    rejected_by_rank: [usize; 4],
    accepted_prefix: [usize; 5],
    draft_seconds: f64,
    target_seconds: f64,
    head_seconds: f64,
    repair_seconds: f64,
}
impl Stats {
    fn add(&mut self, other: &Self) {
        self.draft_forwards += other.draft_forwards;
        self.draft_heads += other.draft_heads;
        self.gpu_conversions += other.gpu_conversions;
        self.cpu_conversions += other.cpu_conversions;
        self.kv_repair_rows += other.kv_repair_rows;
        self.target_executed_rows += other.target_executed_rows;
        self.target_retained_rows += other.target_retained_rows;
        self.rounds += other.rounds;
        self.tail_steps += other.tail_steps;
        self.eos += other.eos;
        for (a, b) in self
            .consulted_by_rank
            .iter_mut()
            .zip(other.consulted_by_rank)
        {
            *a += b;
        }
        for (a, b) in self.accepted_by_rank.iter_mut().zip(other.accepted_by_rank) {
            *a += b;
        }
        for (a, b) in self.rejected_by_rank.iter_mut().zip(other.rejected_by_rank) {
            *a += b;
        }
        for (a, b) in self.accepted_prefix.iter_mut().zip(other.accepted_prefix) {
            *a += b;
        }
        self.draft_seconds += other.draft_seconds;
        self.target_seconds += other.target_seconds;
        self.head_seconds += other.head_seconds;
        self.repair_seconds += other.repair_seconds;
    }
}

struct HiddenWiden {
    output: MatrixView,
    invalid: metal::Buffer,
}
impl HiddenWiden {
    fn new(gpu: &Metal) -> Result<Self> {
        Ok(Self {
            output: MatrixView::output(
                gpu,
                FloatType::F32,
                1,
                10240,
                metal::MTLResourceOptions::StorageModeShared,
            )?,
            invalid: super::super::super::buffer(
                &gpu.device,
                4,
                metal::MTLResourceOptions::StorageModeShared,
            )?,
        })
    }
    fn reset(&self) {
        // The preceding synchronous draft command has completed; this flag has one owner.
        unsafe {
            *self.invalid.contents().cast::<u32>() = 0;
        }
    }
    fn encode(
        &self,
        gpu: &Metal,
        cmd: &metal::CommandBufferRef,
        source: &MatrixView,
    ) -> Result<()> {
        ensure!(
            gpu.owns_command_queue(cmd)
                && source.dtype == FloatType::F16
                && source.shape() == [1, 10240]
                && source.layout.offset == 0
                && source.layout.strides == [10240, 1]
                && source.buffer.length() == 20480
                && source.buffer.device().registry_id() == gpu.device.registry_id(),
            "invalid recursive hidden widening input"
        );
        Metal::encode(
            cmd,
            &gpu.mtp_hidden_widen,
            &[&source.buffer, &self.output.buffer, &self.invalid],
            [10240u32],
            metal::MTLSize::new(80, 1, 1),
            128,
        );
        Ok(())
    }
    fn check(&self) -> Result<()> {
        // Read only the four-byte reduction after the owning command completes.
        ensure!(
            unsafe { *self.invalid.contents().cast::<u32>() } == 0,
            "nonfinite recursive draft hidden"
        );
        Ok(())
    }
}

#[test]
#[ignore = "physical Apple GPU; exhaustive Half widening and invalid-input oracle"]
fn recursive_gpu_widen_preserves_every_finite_half_word() -> Result<()> {
    let gpu = Metal::new()?;
    let widen = HiddenWiden::new(&gpu)?;
    let words: Vec<_> = (0..=u16::MAX)
        .filter(|&w| half::f16::from_bits(w).is_finite())
        .collect();
    let mut checked = 0;
    for chunk in words.chunks(10240) {
        let mut input = vec![0; 10240];
        input[..chunk.len()].copy_from_slice(chunk);
        let source = gpu.f16_storage(&input)?.matrix(MatrixLayout {
            shape: [1, 10240],
            strides: [10240, 1],
            offset: 0,
        })?;
        widen.reset();
        gpu.command(|cmd| widen.encode(&gpu, cmd, &source))?;
        widen.check()?;
        let actual = widen.output.read_output()?;
        ensure!(actual.len() == input.len());
        for (&bits, &word) in actual.iter().zip(&input) {
            ensure!(
                bits == half::f16::from_bits(word).to_f32().to_bits(),
                "widening mismatch at {word:#06x}"
            );
        }
        checked += chunk.len();
    }
    ensure!(checked == 63488);
    for word in [0x7c00, 0xfc00, 0x7e00, 0x7d00] {
        let mut input = vec![0; 10240];
        input[10239] = word;
        let source = gpu.f16_storage(&input)?.matrix(MatrixLayout {
            shape: [1, 10240],
            strides: [10240, 1],
            offset: 0,
        })?;
        widen.reset();
        gpu.command(|cmd| widen.encode(&gpu, cmd, &source))?;
        ensure!(widen.check().is_err());
        let mut bad = source;
        bad.layout.shape[1] = 10239;
        let cmd = gpu.queue.new_command_buffer();
        ensure!(widen.encode(&gpu, cmd, &bad).is_err());
    }
    eprintln!(
        "recursive widening: {checked} finite Half bit patterns exact; four nonfinite inputs rejected; malformed extents rejected"
    );
    Ok(())
}

#[test]
#[ignore = "physical Flash model; recursive transport, full prefix and final-cancellation oracle"]
fn depth_prefix_and_final_cancellation_match_scalar_target() -> Result<()> {
    use std::collections::BTreeMap;
    use std::sync::atomic::AtomicBool;
    fn state(model: &FlashNext<'_>) -> Result<BTreeMap<String, Vec<u8>>> {
        let mut result = BTreeMap::new();
        result.insert(
            "hidden".into(),
            model
                .last_hidden()
                .context("missing hidden")?
                .export_bytes(model.gpu)?,
        );
        for (index, layer) in model.layers.iter().enumerate() {
            if let Some(recurrent) = layer.recurrent() {
                result.insert(format!("gdn_{index}"), recurrent.export_bytes(model.gpu)?);
            }
            if let super::super::Layer::Qsa(layer) = layer {
                result.insert(
                    format!("offset_{index}"),
                    layer.offset().to_le_bytes().to_vec(),
                );
                for (name, words) in layer.retained_half_words()? {
                    result.insert(
                        format!("qsa_{index}_{name}"),
                        words.into_iter().flat_map(u16::to_le_bytes).collect(),
                    );
                }
            }
        }
        ensure!(result.len() > 48);
        Ok(result)
    }
    let checkpoint =
        crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    let mut target = FlashNext::load(&gpu, &checkpoint, 128)?;
    let mut head = FlashMtpHead::load(&checkpoint, &target)?;
    let mut session = FlashMtpSession::default();
    let cancelled = AtomicBool::new(false);
    for tau in [None, Some(0.2)] {
        let quality = crate::gpu::route_swap_trace::Quality::start(tau, 3)?;
        quality.prepare(&gpu)?;
        quality.document(3, 128)?;
        let logits = session.prefill(&mut target, &mut head, &[248044, 78768, 11], &cancelled)?;
        let first = target.greedy_from_logits(&logits)?;
        let root = target.snapshot();
        let draft_root = head.layer.snapshot();
        let root_state = state(&target)?;
        let root_policy =
            crate::gpu::route_swap_trace::decision_snapshot().map(|p| p.test_signature());
        let draft_state = head.layer.retained_half_words()?;
        let mut expected = Vec::new();
        let mut anchor = first;
        for _ in 0..8 {
            let hidden = target
                .last_hidden()
                .context("missing oracle hidden")?
                .clone();
            head.extend(&target, &hidden, anchor)?;
            anchor = target.forward_greedy(anchor)?;
            expected.push(anchor);
        }
        let expected_state = state(&target)?;
        let expected_draft = head.layer.retained_half_words()?;
        let expected_policy =
            crate::gpu::route_swap_trace::decision_snapshot().map(|p| p.test_signature());
        quality.set_oracle(false);
        for depth in 1..=4 {
            for cpu in [true, false] {
                target.restore(&root)?;
                head.layer.restore(&draft_root)?;
                let mut pilot = Pilot::new(depth, 3, &gpu)?;
                if cpu {
                    pilot.widen = None;
                }
                let mut observed = Vec::new();
                let mut anchor = first;
                while observed.len() < 8 {
                    anchor = pilot.advance(
                        &mut target,
                        &mut head,
                        anchor,
                        8 - observed.len(),
                        &[],
                        &cancelled,
                    )?;
                    observed.push(anchor);
                }
                ensure!(
                    observed == expected
                        && state(&target)? == expected_state
                        && head.layer.retained_half_words()? == expected_draft,
                    "depth {depth},cpu={cpu}: tokens or retained states differ"
                );
                ensure!(
                    crate::gpu::route_swap_trace::decision_snapshot().map(|p| p.test_signature())
                        == expected_policy,
                    "logical policy differs across chunking"
                );
                ensure!(
                    pilot.pending.is_empty()
                        && target.offset() == root.offset + 8
                        && head.layer.offset() + 1 == target.offset()
                );
            }
            for rejected in 0..depth {
                target.restore(&root)?;
                head.layer.restore(&draft_root)?;
                let mut anchor = first;
                for _ in 0..=rejected {
                    let hidden = target
                        .last_hidden()
                        .context("missing forced-reject oracle hidden")?
                        .clone();
                    head.extend(&target, &hidden, anchor)?;
                    anchor = target.forward_greedy(anchor)?;
                }
                let prefix_state = state(&target)?;
                let prefix_draft = head.layer.retained_half_words()?;
                let prefix_policy =
                    crate::gpu::route_swap_trace::decision_snapshot().map(|p| p.test_signature());
                target.restore(&root)?;
                head.layer.restore(&draft_root)?;
                let mut pilot = Pilot::new(depth, 3, &gpu)?;
                let mut ids: [u32; 4] = expected[..4].try_into()?;
                ids[rejected] ^= 1;
                pilot.forced_proposals = Some(ids);
                let mut anchor = first;
                for (index, &expected_token) in expected.iter().enumerate().take(rejected + 1) {
                    anchor = pilot.advance(
                        &mut target,
                        &mut head,
                        anchor,
                        5 - index,
                        &[],
                        &cancelled,
                    )?;
                    ensure!(anchor == expected_token);
                }
                ensure!(
                    pilot.pending.is_empty()
                        && pilot.accepted == rejected
                        && pilot.stats.rejected_by_rank[rejected] == 1
                );
                ensure!(
                    state(&target)? == prefix_state
                        && head.layer.retained_half_words()? == prefix_draft
                );
                ensure!(
                    crate::gpu::route_swap_trace::decision_snapshot().map(|p| p.test_signature())
                        == prefix_policy
                );
            }
            target.restore(&root)?;
            head.layer.restore(&draft_root)?;
            let mut eos_pilot = Pilot::new(depth, 3, &gpu)?;
            eos_pilot.forced_proposals = Some(expected[..4].try_into()?);
            ensure!(
                eos_pilot.advance(&mut target, &mut head, first, 8, &[expected[0]], &cancelled)?
                    == expected[0]
            );
            ensure!(target.offset() == root.offset + 1 && eos_pilot.pending.is_empty());
            for remaining in [1, 4] {
                target.restore(&root)?;
                head.layer.restore(&draft_root)?;
                let mut pilot = Pilot::new(depth, 3, &gpu)?;
                pilot.cancel_before_publish = true;
                ensure!(
                    pilot
                        .advance(&mut target, &mut head, first, remaining, &[], &cancelled)
                        .is_err()
                );
                ensure!(cancelled.load(Ordering::Relaxed));
                ensure!(
                    crate::gpu::route_swap_trace::decision_snapshot().map(|p| p.test_signature())
                        == root_policy
                );
                ensure!(
                    state(&target)? == root_state
                        && head.layer.retained_half_words()? == draft_state
                );
                ensure!(target.offset() == root.offset && head.layer.offset() == root.offset - 1);
                ensure!(
                    pilot.proposed == 0
                        && pilot.accepted == 0
                        && pilot.widths == [0; 4]
                        && pilot.stats == Stats::default()
                        && pilot.pending.is_empty()
                );
                cancelled.store(false, Ordering::Relaxed);
                pilot.cancel_before_publish = false;
                ensure!(
                    pilot.advance(&mut target, &mut head, first, 1, &[], &cancelled)?
                        == expected[0]
                );
                ensure!(gpu.readers()?.in_flight() == 0);
            }
        }
    }
    eprintln!(
        "depth D1..D4: both transports match scalar tokens/full target+draft states; eight final-cancellation rollbacks including D0; resume exact"
    );
    Ok(())
}
