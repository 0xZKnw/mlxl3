//! Exact routed EXL3 experts, read directly into reusable slot-major banks.
use crate::{
    array::{Array, Dtype},
    checkpoint::{Checkpoint, TensorInfo},
    codec::Codebook,
    flash_support::{EXPERT_BYTES, ExpertCache, ExpertFill},
    lfm2::Projection,
    linear, router,
    ssd_prefetch::{PREDICTION_RANKS, ReaderPool, order_predictions},
};
use anyhow::{Context, Result, ensure};
use std::{cell::RefCell, collections::BTreeSet, rc::Rc};

const H: i32 = 2560;
const I: i32 = 640;
const TRELLIS: usize = 409600;

pub struct SsdMoe {
    gate: Projection,
    shared_gate: Projection,
    shared_up: Projection,
    shared_down: Projection,
    multiplier: Projection,
    records: Vec<TensorInfo>,
    banks: Vec<Array>,
    cache: Rc<RefCell<ExpertCache>>,
    last: Option<Array>,
    pub hits: u64,
    pub misses: u64,
    pub read_bytes: u64,
    overlap: bool,
    pair_qmv: bool,
    readers: usize,
    pub(crate) predictor_half: bool,
    pub(crate) prefetcher: Option<(usize, Rc<RefCell<ReaderPool>>)>,
    #[cfg(test)]
    pub(crate) route_ids: Vec<u32>,
    #[cfg(test)]
    pub(crate) profile_seconds: [f64; 2],
    #[cfg(test)]
    pub(crate) prefetched: u64,
    #[cfg(test)]
    pub(crate) prefetch_profile: [u64; 4],
    #[cfg(test)]
    pub(crate) overlap_hot_rows: u64,
    #[cfg(test)]
    pub(crate) import_seconds: f64,
    #[cfg(test)]
    schedule_events: Vec<(&'static str, f64)>,
    #[cfg(test)]
    prediction_ready_at_demand: Option<bool>,
    #[cfg(test)]
    slow_prediction_for_test: bool,
}

impl SsdMoe {
    pub fn load(c: &Checkpoint, prefix: &str, slots: usize, overlap: bool) -> Result<Self> {
        let control = |name, default: &str| -> Result<String> {
            match std::env::var(name) {
                Ok(value) => Ok(value),
                Err(std::env::VarError::NotPresent) => Ok(default.into()),
                Err(error) => Err(error.into()),
            }
        };
        let force_lru =
            crate::flash_support::parse_lru_override(&control("MLXL3_SSD_CACHE_POLICY", "auto")?)
                .context("MLXL3_SSD_CACHE_POLICY must be auto or lru")?;
        let predictor_half = crate::flash_support::parse_predictor_half(&control(
            "MLXL3_SSD_PREDICTOR_DTYPE",
            "f32",
        )?)
        .context("MLXL3_SSD_PREDICTOR_DTYPE must be f32 or f16")?;
        let pair_qmv = match control("MLXL3_SSD_PAIR_QMV", "0")?.as_str() {
            "0" => false,
            "1" => crate::array::is_m5_gpu()?,
            _ => anyhow::bail!("MLXL3_SSD_PAIR_QMV must be 0 or 1"),
        };
        let readers = std::env::var("MLXL3_SSD_READERS")
            .unwrap_or_else(|_| "4".into())
            .parse::<usize>()
            .context("invalid SSD reader count")?;
        ensure!((1..=4).contains(&readers), "SSD readers must be in 1..=4");
        let mut records = Vec::with_capacity(512);
        for expert in 0..512 {
            let p = format!("{prefix}.experts.{expert}");
            let mut start = None;
            let mut cursor = 0;
            let mut file = None;
            for (projection, input, output) in
                [("down_proj", I, H), ("gate_proj", H, I), ("up_proj", H, I)]
            {
                for (part, shape, dtype) in [
                    ("suh", vec![input as usize], "F16"),
                    ("svh", vec![output as usize], "F16"),
                    ("mul1", vec![], "I32"),
                    (
                        "trellis",
                        vec![input as usize / 16, output as usize / 16, 32],
                        "I16",
                    ),
                ] {
                    let name = format!("{p}.{projection}.{part}");
                    let tensor = c
                        .tensors
                        .get(&name)
                        .with_context(|| format!("missing {name}"))?;
                    ensure!(
                        tensor.shape == shape && tensor.dtype == dtype,
                        "unsupported Flash expert tensor {name}"
                    );
                    if start.is_none() {
                        start = Some(tensor.clone());
                        cursor = tensor.data_offsets[0];
                        file = Some(tensor.file.clone());
                    }
                    ensure!(
                        Some(&tensor.file) == file.as_ref() && tensor.data_offsets[0] == cursor,
                        "Flash expert record is not contiguous"
                    );
                    cursor = tensor.data_offsets[1];
                }
            }
            let first = start.context("empty expert record")?;
            ensure!(
                cursor - first.data_offsets[0] == EXPERT_BYTES,
                "invalid Flash expert record size"
            );
            records.push(first);
        }
        let s = i32::try_from(slots)?;
        let cache = Rc::new(RefCell::new(ExpertCache::new(slots)?));
        if force_lru {
            cache.borrow_mut().enable_predictive_lru();
        }
        let banks = [
            (vec![s, 2, 160, 40, 32], Dtype::UInt16), // gate/up trellises
            (vec![s, 1, 40, 160, 32], Dtype::UInt16), // down trellis
            (vec![s, 2, H], Dtype::Float16),
            (vec![s, 2, I], Dtype::Float16),
            (vec![s, 1, I], Dtype::Float16),
            (vec![s, 1, H], Dtype::Float16),
            (vec![s, 3], Dtype::Int32),
        ]
        .into_iter()
        .map(|(shape, dtype)| Array::zeros_dtype(&shape, dtype))
        .collect::<Result<Vec<_>>>()?;
        for bank in &banks {
            bank.eval()?;
        }
        let projection = |suffix: &str, input, output| {
            Projection::load(c, &format!("{prefix}.{suffix}"), input, output, false)
        };
        Ok(Self {
            gate: projection("gate", H, 512)?,
            shared_gate: projection("shared_expert.gate_proj", H, I)?,
            shared_up: projection("shared_expert.up_proj", H, I)?,
            shared_down: projection("shared_expert.down_proj", I, H)?,
            multiplier: projection("shared_expert_gate", H, 1)?,
            records,
            banks,
            cache,
            last: None,
            hits: 0,
            misses: 0,
            read_bytes: 0,
            overlap,
            pair_qmv,
            readers,
            predictor_half,
            prefetcher: None,
            #[cfg(test)]
            route_ids: Vec::new(),
            #[cfg(test)]
            profile_seconds: [0.0; 2],
            #[cfg(test)]
            prefetched: 0,
            #[cfg(test)]
            prefetch_profile: [0; 4],
            #[cfg(test)]
            overlap_hot_rows: 0,
            #[cfg(test)]
            import_seconds: 0.0,
            #[cfg(test)]
            schedule_events: Vec::with_capacity(8),
            #[cfg(test)]
            prediction_ready_at_demand: None,
            #[cfg(test)]
            slow_prediction_for_test: false,
        })
    }

    fn destinations(slot: usize) -> [(usize, usize, usize); 12] {
        [
            (4, slot * I as usize * 2, I as usize * 2),
            (5, slot * H as usize * 2, H as usize * 2),
            (6, slot * 12, 4),
            (1, slot * TRELLIS, TRELLIS),
            (2, slot * 2 * H as usize * 2, H as usize * 2),
            (3, slot * 2 * I as usize * 2, I as usize * 2),
            (6, slot * 12 + 4, 4),
            (0, slot * 2 * TRELLIS, TRELLIS),
            (2, (slot * 2 + 1) * H as usize * 2, H as usize * 2),
            (3, (slot * 2 + 1) * I as usize * 2, I as usize * 2),
            (6, slot * 12 + 8, 4),
            (0, (slot * 2 + 1) * TRELLIS, TRELLIS),
        ]
    }

    fn fill_many(&mut self, jobs: &[(u32, usize)]) -> Result<()> {
        if jobs.is_empty() {
            return Ok(());
        }
        let sources = jobs
            .iter()
            .map(|&(expert, _)| {
                self.records
                    .get(expert as usize)
                    .context("invalid expert")?
                    .source()
            })
            .collect::<Result<Vec<_>>>()?;
        let destinations = jobs
            .iter()
            .map(|&(_, slot)| Self::destinations(slot))
            .collect::<Vec<_>>();
        let views = destinations
            .iter()
            .map(|ranges| ranges.as_slice())
            .collect::<Vec<_>>();
        let bytes = EXPERT_BYTES
            .checked_mul(jobs.len() as u64)
            .and_then(|n| self.read_bytes.checked_add(n))
            .context("SSD byte counter exhausted")?;
        Array::read_records(&sources, &mut self.banks, &views, self.readers)?;
        self.read_bytes = bytes;
        Ok(())
    }

    fn calculate(&self, x: &Array, slots: &[u32], paired: bool) -> Result<Array> {
        let n = i32::try_from(slots.len())?;
        ensure!(n > 0 && x.shape() == [n, H], "invalid expert slot mapping");
        let slots = linear::ExpertSlots::new_paired(slots, self.banks[0].shape()[0], paired)?;
        let indices = slots.indices();
        let expanded = x
            .reshape(&[n, 1, H])?
            .broadcast_to(&[n, 2, H])?
            .reshape(&[n * 2, H])?;
        let suh = self.banks[2].take(indices, 0)?.reshape(&[n * 2, H])?;
        let svh = self.banks[3].take(indices, 0)?.reshape(&[n * 2, I])?;
        let gu = linear::expert_slots(
            &expanded,
            &self.banks[0],
            &suh,
            &svh,
            &slots,
            I,
            2,
            2,
            Codebook::Mul1,
        )?
        .reshape(&[n, 2, I])?;
        let activated = gu
            .slice(1, 0, 1)?
            .reshape(&[n, I])?
            .swiglu(&gu.slice(1, 1, 2)?.reshape(&[n, I])?)?;
        linear::expert_slots(
            &activated,
            &self.banks[1],
            &self.banks[4].take(indices, 0)?.reshape(&[n, I])?,
            &self.banks[5].take(indices, 0)?.reshape(&[n, H])?,
            &slots,
            H,
            1,
            2,
            Codebook::Mul1,
        )
    }

    pub(crate) fn route(&self, x: &Array) -> Result<(Array, Array)> {
        ensure!(
            x.shape() == [1, H] && matches!(x.dtype(), Dtype::Float16 | Dtype::Float32),
            "streamed MoE expects one exact route row"
        );
        let probabilities = self.gate.forward(x)?.softmax_precise()?;
        if x.dtype() == Dtype::Float16 {
            let (indices, scores) = router::topk(&probabilities, 10, false)?;
            Ok((indices, scores.div(&scores.sum(1, true)?)?))
        } else {
            router::topk(&probabilities, 10, true)
        }
    }
    fn predict(&self, x: &Array) -> Result<Array> {
        let x = if self.predictor_half {
            x.astype(Dtype::Float16)?
        } else {
            x.try_clone()?
        };
        let scores = self.gate.forward(&x)?;
        #[cfg(test)]
        let scores = if self.slow_prediction_for_test {
            crate::array::metal_kernel(
                "mlxl3_test_slow_prediction",
                &["scores"], &["out"], "",
                "uint i=thread_position_in_grid.x; float v=float(scores[i]); for(uint j=0;j<1000000u;++j) v=fma(v,0.99999f,0.0001f); out[i]=v;",
                &[&scores], &[scores.shape().to_vec()], &[Dtype::Float32],
                [512, 1, 1], [128, 1, 1],
            )?.remove(0)
        } else {
            scores
        };
        scores
            .argsort()?
            .slice(1, 512 - PREDICTION_RANKS as i32, 512)
    }
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        self.forward_many(x)
    }

    pub(crate) fn warm_all(&mut self) -> Result<()> {
        self.drain()?;
        ensure!(
            self.cache.borrow().capacity() == 512,
            "resident draft requires all expert slots"
        );
        let ids = (0..512u32).collect::<Vec<_>>();
        let routes = self.cache.borrow_mut().reserve(&ids)?;
        let jobs = ids
            .iter()
            .zip(&routes)
            .filter_map(|(&id, &(slot, hit))| (!hit).then_some((id, slot)))
            .collect::<Vec<_>>();
        if let Err(error) = self.fill_many(&jobs) {
            self.cache.borrow_mut().invalidate();
            return Err(error);
        }
        Ok(())
    }

    pub(crate) fn forward_many(&mut self, x: &Array) -> Result<Array> {
        self.forward_lookahead(x, None, 0)
    }

    pub(crate) fn forward_lookahead(
        &mut self,
        x: &Array,
        next: Option<&mut Self>,
        depth: usize,
    ) -> Result<Array> {
        self.forward_rows(x, next, depth, 2)
    }

    #[cfg(test)]
    pub(crate) fn forward_draft_rows(&mut self, x: &Array) -> Result<Array> {
        self.forward_rows(x, None, 0, 8)
    }

    fn forward_rows(
        &mut self,
        x: &Array,
        next: Option<&mut Self>,
        depth: usize,
        row_limit: i32,
    ) -> Result<Array> {
        ensure!(depth <= 10, "lookahead depth must be in 0..=10");
        ensure!(
            x.shape().len() == 2
                && (1..=row_limit).contains(&x.shape()[0])
                && x.shape()[1] == H
                && matches!(x.dtype(), Dtype::Float16 | Dtype::Float32),
            "streamed MoE expects one or two route rows"
        );
        let rows = x.shape()[0];
        if self.cache.borrow().capacity() < rows as usize * 10 {
            let cohort = crate::flash_support::expert_cohort_rows(self.cache.borrow().capacity())
                .context("invalid streamed MoE cache capacity")? as i32;
            let mut output = Vec::new();
            for begin in (0..rows).step_by(cohort as usize) {
                output.push(self.forward_rows(
                    &x.slice(0, begin, (begin + cohort).min(rows))?,
                    None,
                    0,
                    row_limit,
                )?);
            }
            return Array::concatenate(&output.iter().collect::<Vec<_>>(), 0);
        }
        #[cfg(test)]
        let route_start = std::time::Instant::now();
        #[cfg(test)]
        {
            self.schedule_events.clear();
            self.prediction_ready_at_demand = None;
        }
        let pairs = (0..rows)
            .map(|row| self.route(&x.slice(0, row, row + 1)?))
            .collect::<Result<Vec<_>>>()?;
        let selected = Array::concatenate(&pairs.iter().map(|p| &p.0).collect::<Vec<_>>(), 0)?;
        let weights = Array::concatenate(&pairs.iter().map(|p| &p.1).collect::<Vec<_>>(), 0)?;
        let ids = selected.to_u32()?;
        #[cfg(test)]
        {
            self.profile_seconds[0] += route_start.elapsed().as_secs_f64();
            self.schedule_events
                .push(("routes_cpu", route_start.elapsed().as_secs_f64()));
        }
        #[cfg(test)]
        {
            self.route_ids.clone_from(&ids);
        }
        let row_indices = (0..rows as u32)
            .flat_map(|row| std::iter::repeat_n(row, 10))
            .collect::<Vec<_>>();
        let routed_x = x
            .take(&Array::from_u32(&row_indices, &[rows * 10])?, 0)?
            .astype(Dtype::Float16)?;
        if let Some(last) = self.last.take() {
            last.eval()?;
        }
        #[cfg(test)]
        let import_start = std::time::Instant::now();
        if let Some((layer, pool)) = &self.prefetcher {
            self.cache.borrow_mut().enable_predictive_lru();
            let mut pool = pool.borrow_mut();
            pool.poll();
            pool.note_late(*layer, &ids);
        }
        let before = self.cache.borrow().prefetch_hits;
        let fill = ExpertFill::new(&self.cache);
        let routes = self.cache.borrow_mut().reserve(&ids)?;
        let used = self.cache.borrow().prefetch_hits - before;
        if let Some((_, pool)) = &self.prefetcher {
            pool.borrow()
                .counters
                .used
                .fetch_add(used, std::sync::atomic::Ordering::Relaxed);
        }
        #[cfg(test)]
        {
            self.prefetched += used;
            self.import_seconds += import_start.elapsed().as_secs_f64();
        }
        let mut hot = Vec::new();
        let mut cold = Vec::new();
        let mut order = Vec::new();
        for (i, &(slot, hit)) in routes.iter().enumerate() {
            if hit {
                self.hits = self.hits.saturating_add(1);
            } else {
                self.misses = self.misses.saturating_add(1);
            }
            if hit {
                hot.push((i, slot as u32));
            } else {
                cold.push((i, slot as u32));
            }
        }
        let hot_output = if self.overlap && !hot.is_empty() && !cold.is_empty() {
            let value = self.calculate(
                &routed_x.take(
                    &Array::from_u32(
                        &hot.iter().map(|p| p.0 as u32).collect::<Vec<_>>(),
                        &[hot.len() as i32],
                    )?,
                    0,
                )?,
                &hot.iter().map(|p| p.1).collect::<Vec<_>>(),
                rows == 2 && self.pair_qmv,
            )?;
            // Retain submitted reads of the banks even if a later operation fails.
            self.last = Some(value.try_clone()?);
            value.async_eval()?;
            #[cfg(test)]
            {
                self.overlap_hot_rows += hot.len() as u64;
            }
            Some(value)
        } else {
            None
        };
        let prediction = (|| -> Result<Option<Array>> {
            let prediction = if depth > 0 {
                next.as_ref().map(|next| next.predict(x)).transpose()?
            } else {
                None
            };
            if let Some(prediction) = &prediction {
                prediction.async_eval()?;
                #[cfg(test)]
                self.schedule_events
                    .push(("prediction_submitted", route_start.elapsed().as_secs_f64()));
            }
            Ok(prediction)
        })();
        let prediction = match prediction {
            Ok(prediction) => prediction,
            Err(error) => {
                if let Some(hot) = &hot_output {
                    hot.eval()?;
                }
                return Err(error.context("preparing optional expert prediction"));
            }
        };
        let mut filled = BTreeSet::new();
        #[cfg(test)]
        let read_start = std::time::Instant::now();
        let jobs = cold
            .iter()
            .filter_map(|&(i, slot)| filled.insert(slot).then_some((ids[i], slot as usize)))
            .collect::<Vec<_>>();
        #[cfg(test)]
        if !jobs.is_empty() {
            self.prediction_ready_at_demand = prediction
                .as_ref()
                .map(Array::is_available_for_test)
                .transpose()?;
            self.schedule_events
                .push(("demand_begin", route_start.elapsed().as_secs_f64()));
        }
        if let Err(error) = self.fill_many(&jobs) {
            if let Some(hot) = &hot_output {
                hot.eval()?;
            }
            return Err(error.context("loading cold EXL3 experts"));
        }
        fill.commit();
        #[cfg(test)]
        {
            self.profile_seconds[1] += read_start.elapsed().as_secs_f64();
            if !jobs.is_empty() {
                self.schedule_events
                    .push(("demand_end", route_start.elapsed().as_secs_f64()));
            }
        }
        let values = if let Some(hot_output) = hot_output {
            let cold_output = self.calculate(
                &routed_x.take(
                    &Array::from_u32(
                        &cold.iter().map(|p| p.0 as u32).collect::<Vec<_>>(),
                        &[cold.len() as i32],
                    )?,
                    0,
                )?,
                &cold.iter().map(|p| p.1).collect::<Vec<_>>(),
                rows == 2 && self.pair_qmv,
            )?;
            let combined = Array::concatenate(&[&hot_output, &cold_output], 0)?;
            for i in 0..ids.len() {
                order.push(hot.iter().position(|p| p.0 == i).unwrap_or_else(|| {
                    hot.len() + cold.iter().position(|p| p.0 == i).expect("cold route")
                }) as u32);
            }
            combined.take(&Array::from_u32(&order, &[order.len() as i32])?, 0)?
        } else {
            self.calculate(
                &routed_x,
                &routes.iter().map(|p| p.0 as u32).collect::<Vec<_>>(),
                rows == 2 && self.pair_qmv,
            )?
        };
        let routed = values
            .astype(x.dtype())?
            .mul(&weights.reshape(&[rows * 10, 1])?)?
            .reshape(&[rows, 10, H])?
            .sum(1, false)?;
        let mut shared_rows = Vec::new();
        for row in 0..rows {
            let x = x.slice(0, row, row + 1)?;
            let shared = self
                .shared_down
                .forward(
                    &self
                        .shared_gate
                        .forward(&x)?
                        .astype(Dtype::Float16)?
                        .swiglu(&self.shared_up.forward(&x)?.astype(Dtype::Float16)?)?,
                )?
                .astype(x.dtype())?;
            shared_rows.push(shared.mul(&self.multiplier.forward(&x)?.sigmoid()?)?);
        }
        let output = routed.add(&Array::concatenate(
            &shared_rows.iter().collect::<Vec<_>>(),
            0,
        )?)?;
        self.last = Some(output.try_clone()?);
        let predicted_ids = prediction
            .map(|p| -> Result<Vec<u32>> {
                let mut ids = p.to_u32()?;
                ensure!(
                    order_predictions(&mut ids, rows as usize),
                    "invalid lookahead prediction shape or IDs"
                );
                #[cfg(test)]
                self.schedule_events
                    .push(("prediction_cpu", route_start.elapsed().as_secs_f64()));
                Ok(ids)
            })
            .transpose()?;
        if let (Some(next), Some(ids)) = (next, predicted_ids) {
            if ids.iter().any(|&id| !next.cache.borrow().contains(id)) {
                output.async_eval()?;
            }
            next.prefetch(&ids, depth)?;
        }
        Ok(output)
    }

    fn prefetch(&mut self, ids: &[u32], budget: usize) -> Result<()> {
        ensure!(
            !ids.is_empty() && ids.len() <= 2 * PREDICTION_RANKS && ids.iter().all(|&id| id < 512),
            "invalid predicted experts"
        );
        let Some((layer, pool)) = self.prefetcher.as_ref().map(|(l, p)| (*l, Rc::clone(p))) else {
            return Ok(());
        };
        let mut pool = pool.borrow_mut();
        pool.poll();
        self.cache.borrow_mut().protect_predictions(ids)?;
        let mut cold = Vec::new();
        #[cfg(test)]
        {
            self.prefetch_profile[0] += ids.len() as u64;
        }
        for &id in ids {
            #[cfg(test)]
            {
                self.prefetch_profile[1] += 1;
            }
            if self.cache.borrow().contains(id) {
                #[cfg(test)]
                {
                    self.prefetch_profile[2] += 1;
                }
            } else if !cold.contains(&id) {
                cold.push(id);
            }
        }
        if cold.is_empty() || budget == 0 || !pool.has_room() {
            return Ok(());
        }
        self.drain()?;
        let mut admitted = 0;
        for id in cold {
            if admitted == budget.min(4) || !pool.has_room() {
                break;
            }
            let Some(slot) = self.cache.borrow_mut().claim_prefetch(id)? else {
                continue;
            };
            let read = self.records[id as usize]
                .source()
                .and_then(|(file, offset)| {
                    Array::prepare_read(file, offset, &mut self.banks, &Self::destinations(slot))
                });
            let read = match read {
                Ok(read) => read,
                Err(_) => {
                    self.cache.borrow_mut().publish_prefetch(slot, id, false);
                    pool.counters
                        .failed
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    continue;
                }
            };
            if pool.submit(layer, id, slot, Rc::clone(&self.cache), read) {
                admitted += 1;
                #[cfg(test)]
                {
                    self.prefetch_profile[3] += 1;
                }
            }
        }
        Ok(())
    }

    pub fn drain(&mut self) -> Result<()> {
        if let Some(last) = self.last.take() {
            last.eval()?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn cache_for_test_invalidate(&mut self) {
        self.cache.borrow_mut().invalidate();
    }

    #[cfg(test)]
    pub(crate) fn slots_for_test(&self) -> usize {
        let slots = self.cache.borrow().capacity();
        assert!(self.banks.iter().all(|b| b.shape()[0] == slots as i32));
        slots
    }

    #[cfg(test)]
    pub(crate) fn gate_for_test(&self) -> &Projection {
        &self.gate
    }
}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "physical M5; experiment48 exact isolated hot MoE, not tokens/sec"]
    fn shared_pair_hot_moe_microbench_preserves_outputs() -> anyhow::Result<()> {
        use super::*;
        use serde_json::Value;
        use std::{fs::File, path::Path, time::Instant};
        ensure!(crate::array::is_m5_gpu()?);
        let output = std::env::var("MLXL3_FLASH_PAIR_MICRO")?;
        serde_json::to_writer(
            File::create_new(&output)?,
            &serde_json::json!({"status":"incomplete"}),
        )?;
        let root = std::env::var("MLXL3_FLASH_REFERENCE")?;
        let root = Path::new(&root);
        let reference: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
        ensure!(reference["status"] == "complete" && reference["mode"] == "eval");
        let checkpoint =
            crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        crate::array::set_default_cache_limit(512 << 20)?;
        let mut records = Vec::new();
        for layer in [0usize, 23, 47] {
            let mut inputs = Vec::new();
            let mut expected = Vec::new();
            for step in [0, 1] {
                let entry = reference["steps"][step]["components"]
                    .as_array()
                    .context("missing components")?
                    .get(layer)
                    .context("missing micro layer")?;
                let read = |kind: &str| -> Result<Vec<u8>> {
                    let entry = &entry["mlp"][kind];
                    ensure!(entry["count"] == H);
                    let bytes =
                        std::fs::read(root.join(entry["file"].as_str().context("missing file")?))?;
                    ensure!(
                        bytes.len() == H as usize * 4
                            && bytes
                                .as_chunks::<4>()
                                .0
                                .iter()
                                .all(|&b| f32::from_le_bytes(b).is_finite())
                    );
                    Ok(bytes)
                };
                inputs.push(Array::from_bytes(&read("input")?, &[1, H], Dtype::Float32)?);
                expected.push(read("output")?);
            }
            let mut moe = SsdMoe::load(
                &checkpoint,
                &format!("model.language_model.layers.{layer}.mlp"),
                20,
                true,
            )?;
            for rows in [1, 2] {
                let x = if rows == 1 {
                    inputs[0].try_clone()?
                } else {
                    Array::concatenate(&[&inputs[0], &inputs[1]], 0)?
                };
                let oracle = expected[..rows].concat();
                // Populate exactly this cohort, then include all routing/map/graph costs in timings.
                moe.pair_qmv = false;
                ensure!(
                    moe.forward_many(&x)?.to_bytes()? == oracle,
                    "independent micro oracle differs"
                );
                let io = moe.read_bytes;
                let mut ids = Vec::new();
                for _ in 0..2 {
                    for candidate in [false, true] {
                        moe.pair_qmv = candidate;
                        let value = moe.forward_many(&x)?;
                        ensure!(value.shape() == [rows as i32, H]);
                        ensure!(value.to_bytes()? == oracle, "warm shared MoE differs");
                        if ids.is_empty() {
                            ids.clone_from(&moe.route_ids);
                        }
                        ensure!(ids == moe.route_ids && moe.read_bytes == io);
                    }
                }
                let distinct = ids.iter().copied().collect::<BTreeSet<_>>().len();
                ensure!(
                    rows == 1 || distinct < ids.len(),
                    "micro did not exercise shared experts"
                );
                let mut samples = Vec::new();
                for pair in 0..4 {
                    for candidate in if pair % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        moe.pair_qmv = candidate;
                        let start = Instant::now();
                        let mut last = None;
                        for _ in 0..16 {
                            let value = moe.forward_many(&x)?;
                            value.eval()?;
                            last = Some(value);
                        }
                        let seconds = start.elapsed().as_secs_f64() / 16.;
                        ensure!(seconds.is_finite() && seconds > 0.);
                        ensure!(last.context("no micro output")?.to_bytes()? == oracle);
                        ensure!(
                            moe.route_ids == ids && moe.read_bytes == io,
                            "hot micro performed I/O"
                        );
                        samples.push(serde_json::json!({"pair":pair, "candidate":candidate, "seconds":seconds}));
                    }
                }
                records.push(serde_json::json!({"layer":layer, "rows":rows, "route_ids":ids, "io_bytes_after_warm":io, "samples":samples}));
            }
            moe.drain()?;
        }
        std::fs::write(
            output,
            serde_json::to_vec_pretty(&serde_json::json!({
                "status":"complete", "samples_per_cell":8, "forwards_per_sample":16,
                "parity":"byte_exact_independent", "no_timed_io":true, "records":records
            }))?,
        )?;
        println!(
            "FLASH_PAIR_MICRO layers=3 row_shapes=2 samples=48 forwards_per_sample=16 parity=byte_exact no_timed_io=true"
        );
        Ok(())
    }

    fn admission_fixture()
    -> anyhow::Result<(crate::checkpoint::Checkpoint, crate::array::Array, Vec<u8>)> {
        use crate::array::{Array, Dtype};
        use anyhow::{Context, ensure};
        use std::path::Path;
        let root = std::env::var("MLXL3_FLASH_REFERENCE")?;
        let root = Path::new(&root);
        let reference: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(root.join("reference.json"))?)?;
        let fixture = &reference["steps"][0]["components"][0]["mlp"];
        let input =
            std::fs::read(root.join(fixture["input"]["file"].as_str().context("missing input")?))?;
        let expected = std::fs::read(
            root.join(
                fixture["output"]["file"]
                    .as_str()
                    .context("missing output")?,
            ),
        )?;
        ensure!(input.len() == super::H as usize * 4 && expected.len() == input.len());
        let checkpoint =
            crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        Ok((
            checkpoint,
            Array::from_bytes(&input, &[1, super::H], Dtype::Float32)?,
            expected,
        ))
    }

    #[test]
    #[ignore = "physical Apple GPU; demanded reads must not wait for a slow predictor"]
    fn demanded_reads_start_before_the_optional_prediction_finishes() -> anyhow::Result<()> {
        use super::*;
        let (checkpoint, input, expected) = admission_fixture()?;
        let mut current = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 24, true)?;
        let mut next = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 24, true)?;
        next.slow_prediction_for_test = true;
        let pool = Rc::new(RefCell::new(ReaderPool::new()?));
        next.prefetcher = Some((1, Rc::clone(&pool)));
        let actual = current.forward_lookahead(&input, Some(&mut next), 4)?;
        println!(
            "SSD_SCHEDULE_TRACE {}",
            serde_json::json!({
                "clock_domain":"cpu_monotonic_seconds", "events":current.schedule_events,
                "prediction_ready_at_first_demand":current.prediction_ready_at_demand,
                "demand_records":current.read_bytes / EXPERT_BYTES,
                "test_only_slow_prediction":true,
            })
        );
        ensure!(actual.shape() == [1, H] && actual.to_bytes()? == expected);
        ensure!(current.read_bytes == 10 * EXPERT_BYTES);
        ensure!(
            current.prediction_ready_at_demand == Some(false),
            "demand waited for optional prediction"
        );
        let names = current
            .schedule_events
            .iter()
            .map(|e| e.0)
            .collect::<Vec<_>>();
        ensure!(
            names
                == [
                    "routes_cpu",
                    "prediction_submitted",
                    "demand_begin",
                    "demand_end",
                    "prediction_cpu"
                ]
        );
        ensure!(
            current
                .schedule_events
                .windows(2)
                .all(|p| p[0].1.is_finite() && p[0].1 <= p[1].1)
        );
        pool.borrow_mut().wait_for_test()?;
        next.slow_prediction_for_test = false;
        ensure!(next.forward(&input)?.to_bytes()? == expected);
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU; failed prediction must invalidate unfilled routes and recover"]
    fn failed_prediction_invalidates_reserved_routes_and_recovers() -> anyhow::Result<()> {
        use super::*;
        let (checkpoint, input, expected) = admission_fixture()?;
        let mut current = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 24, true)?;
        let mut next = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 24, true)?;
        let ids = current.route(&input)?.0.to_u32()?;
        ensure!(ids.len() == 10 && ids.iter().copied().collect::<BTreeSet<_>>().len() == 10);
        let warm = current.cache.borrow_mut().reserve(&ids[..5])?;
        let jobs = ids[..5]
            .iter()
            .zip(warm)
            .map(|(&id, (slot, _))| (id, slot))
            .collect::<Vec<_>>();
        current.fill_many(&jobs)?;
        std::mem::swap(&mut next.gate, &mut next.shared_down);
        let error = current
            .forward_lookahead(&input, Some(&mut next), 4)
            .err()
            .context("invalid prediction unexpectedly succeeded")?;
        ensure!(error.chain().any(|error| {
            error
                .to_string()
                .contains("projection expects 640 logical inputs")
        }));
        ensure!(
            current.read_bytes == 5 * EXPERT_BYTES,
            "prediction did not fail before cold reads"
        );
        ensure!(current.overlap_hot_rows == 5 && current.misses == 5);
        ensure!(
            ids.iter().all(|&id| !current.cache.borrow().contains(id)),
            "failed prediction left unfilled routes cached"
        );
        std::mem::swap(&mut next.gate, &mut next.shared_down);
        let pool = Rc::new(RefCell::new(ReaderPool::new()?));
        next.prefetcher = Some((1, Rc::clone(&pool)));
        let actual = current.forward_lookahead(&input, Some(&mut next), 4)?;
        ensure!(actual.shape() == [1, H] && actual.to_bytes()? == expected);
        pool.borrow_mut().wait_for_test()?;
        ensure!(next.forward(&input)?.to_bytes()? == expected);
        Ok(())
    }

    #[cfg(kani)]
    #[kani::proof]
    fn actual_slot_dispatch_and_optional_prediction_schedule() {
        demanded_reads_start_before_the_optional_prediction_finishes().unwrap();
        failed_prediction_invalidates_reserved_routes_and_recovers().unwrap();
    }

    #[test]
    #[ignore = "physical Apple GPU; hot failure must invalidate unfilled routes before retry"]
    fn failed_hot_compute_invalidates_reserved_routes_and_recovers() -> anyhow::Result<()> {
        use super::*;
        let (checkpoint, input, expected) = admission_fixture()?;
        let mut moe = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 24, true)?;
        let ids = moe.route(&input)?.0.to_u32()?;
        ensure!(ids.len() == 10 && ids.iter().copied().collect::<BTreeSet<_>>().len() == 10);
        let warm = moe.cache.borrow_mut().reserve(&ids[..5])?;
        let jobs = ids[..5]
            .iter()
            .zip(warm)
            .map(|(&id, (slot, _))| (id, slot))
            .collect::<Vec<_>>();
        moe.fill_many(&jobs)?;
        // Same storage, invalid layout: fail the real hot calculation before any cold read.
        let invalid = moe.banks[0].reshape(&[24, -1])?;
        let bank = std::mem::replace(&mut moe.banks[0], invalid);
        let error = moe
            .forward(&input)
            .err()
            .context("invalid hot bank unexpectedly succeeded")?;
        moe.banks[0] = bank;
        ensure!(
            error
                .to_string()
                .contains("expert slots were validated for a different bank capacity"),
            "{error:#}"
        );
        ensure!(moe.read_bytes == 5 * EXPERT_BYTES && moe.misses == 5);
        ensure!(
            ids.iter().all(|&id| !moe.cache.borrow().contains(id)),
            "unfilled routes cached after hot failure"
        );
        let actual = moe.forward(&input)?;
        ensure!(actual.shape() == [1, H] && actual.to_bytes()? == expected);
        ensure!(
            moe.read_bytes == 15 * EXPERT_BYTES,
            "retry did not reload every route"
        );
        moe.drain()?;
        Ok(())
    }

    #[cfg(kani)]
    #[kani::proof]
    fn actual_hot_failure_recovers_reserved_routes() {
        failed_hot_compute_invalidates_reserved_routes_and_recovers().unwrap();
    }

    #[test]
    #[ignore = "physical Apple GPU; first-layer LES must use LRU without incoming predictions"]
    fn attached_pool_enables_lru_without_an_incoming_prediction() -> anyhow::Result<()> {
        use super::*;
        let (checkpoint, input, expected) = admission_fixture()?;
        let mut moe = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 89, true)?;
        let selected = moe.route(&input)?.0.to_u32()?;
        ensure!(
            selected.len() == 10 && selected.iter().copied().collect::<BTreeSet<_>>().len() == 10
        );
        ensure!(selected.iter().any(|&id| id >= 89));
        let marker = (0..89u32)
            .find(|id| !selected.contains(id))
            .context("fixture needs an unrouted old expert")?;
        let ids = (0..89u32).collect::<Vec<_>>();
        let routes = moe.cache.borrow_mut().reserve(&ids)?;
        let jobs = ids
            .iter()
            .zip(routes)
            .map(|(&id, (slot, _))| (id, slot))
            .collect::<Vec<_>>();
        moe.fill_many(&jobs)?;
        for _ in 0..3 {
            moe.cache.borrow_mut().reserve(&[marker])?;
        }
        let others = ids
            .into_iter()
            .filter(|&id| id != marker)
            .collect::<Vec<_>>();
        for offset in (0..90).step_by(10) {
            let cohort = (offset..offset + 10)
                .map(|i| others[i % others.len()])
                .collect::<Vec<_>>();
            moe.cache.borrow_mut().reserve(&cohort)?;
        }
        ensure!(moe.cache.borrow().contains(marker));
        moe.prefetcher = Some((0, Rc::new(RefCell::new(ReaderPool::new()?))));
        ensure!(
            moe.forward(&input)?.to_bytes()? == expected,
            "independent MoE oracle differs"
        );
        ensure!(
            !moe.cache.borrow().contains(marker),
            "attached first-layer LES pool retained the old LFU policy"
        );
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU; persistent routed experts must overlap remaining SSD reads"]
    fn completed_prefetches_compute_while_other_experts_are_read() -> anyhow::Result<()> {
        use super::*;
        let (checkpoint, input, expected) = admission_fixture()?;
        let mut moe = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 89, true)?;
        let selected = moe.route(&input)?.0.to_u32()?;
        ensure!(
            selected.len() == 10 && selected.iter().copied().collect::<BTreeSet<_>>().len() == 10
        );
        let pool = Rc::new(RefCell::new(ReaderPool::new()?));
        moe.prefetcher = Some((0, Rc::clone(&pool)));
        moe.prefetch(&selected, 4)?;
        pool.borrow_mut().wait_for_test()?;
        ensure!(
            moe.forward(&input)?.to_bytes()? == expected,
            "independent MoE oracle differs"
        );
        ensure!(moe.hits == 4 && moe.misses == 6 && moe.prefetched == 4);
        ensure!(moe.read_bytes == 6 * EXPERT_BYTES);
        ensure!(
            moe.overlap_hot_rows == 4,
            "completed prefetches still wait for cold SSD reads"
        );
        Ok(())
    }

    #[cfg(kani)]
    #[kani::proof]
    fn ready_experts_overlap_real_ssd_path() {
        completed_prefetches_compute_while_other_experts_are_read().unwrap();
    }

    #[test]
    #[ignore = "physical Apple GPU; completed late results must release admission capacity"]
    fn completed_previous_layer_reads_do_not_block_next_prefetch_cohort() -> anyhow::Result<()> {
        use super::*;
        use std::sync::atomic::Ordering;
        let (checkpoint, input, expected) = admission_fixture()?;
        let mut next = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 89, true)?;
        let pool = Rc::new(RefCell::new(ReaderPool::new()?));
        next.prefetcher = Some((1, Rc::clone(&pool)));
        let mut previous =
            SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 24, true)?;
        previous.prefetcher = Some((0, Rc::clone(&pool)));
        previous.prefetch(&[0, 1, 2, 3], 4)?;
        pool.borrow_mut().wait_for_test()?;
        ensure!(pool.borrow().counters.completed.load(Ordering::Relaxed) == 4);
        let before = pool.borrow().counters.submitted.load(Ordering::Relaxed);
        next.prefetch(&[511], 4)?;
        ensure!(
            pool.borrow().counters.submitted.load(Ordering::Relaxed) == before + 1,
            "completed previous-layer results still block the next cohort"
        );
        pool.borrow_mut().wait_for_test()?;
        ensure!(next.forward(&input)?.to_bytes()? == expected);
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU; cached strongest predictions must not consume the read budget"]
    fn cached_top_four_predictions_do_not_starve_useful_cold_prefetches() -> anyhow::Result<()> {
        use super::*;
        use std::sync::atomic::Ordering;
        let (checkpoint, input, expected) = admission_fixture()?;
        let mut current = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 89, true)?;
        let mut next = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 89, true)?;
        let hot = router::topk(&next.gate.forward(&input)?, 4, false)?
            .0
            .to_u32()?;
        ensure!(hot.len() == 4);
        let sorted_hot = next
            .gate
            .forward(&input)?
            .argsort()?
            .slice(1, 508, 512)?
            .to_u32()?;
        ensure!(
            hot.iter().copied().collect::<BTreeSet<_>>()
                == sorted_hot.into_iter().collect::<BTreeSet<_>>(),
            "fixture must warm the strongest four under both selectors"
        );
        let routes = next.cache.borrow_mut().reserve(&hot)?;
        let jobs = hot
            .iter()
            .zip(routes)
            .map(|(&id, (slot, _))| (id, slot))
            .collect::<Vec<_>>();
        next.fill_many(&jobs)?;
        ensure!(hot.iter().all(|&id| next.cache.borrow().contains(id)));
        let pool = Rc::new(RefCell::new(ReaderPool::new()?));
        next.prefetcher = Some((1, Rc::clone(&pool)));
        let before = pool.borrow().counters.submitted.load(Ordering::Relaxed);
        ensure!(
            current
                .forward_lookahead(&input, Some(&mut next), 4)?
                .to_bytes()?
                == expected
        );
        ensure!(
            pool.borrow().counters.submitted.load(Ordering::Relaxed) > before,
            "cached top four starve lower-ranked cold experts"
        );
        pool.borrow_mut().wait_for_test()?;
        ensure!(next.forward(&input)?.to_bytes()? == expected && next.prefetched > 0);
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU; independent CPU sort oracle for one/two prediction rows"]
    fn batched_prediction_selects_top_sixteen_values_without_changing_real_routing()
    -> anyhow::Result<()> {
        use super::*;
        let (checkpoint, input, expected) = admission_fixture()?;
        let mut moe = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 89, true)?;
        for half in [false, true] {
            moe.predictor_half = half;
            for dtype in [Dtype::Float16, Dtype::Float32] {
                for rows in [1, 2] {
                    let x = if rows == 1 {
                        input.try_clone()?
                    } else {
                        Array::concatenate(&[&input, &input.scalar_mul(0.875)?], 0)?
                    }
                    .astype(dtype)?;
                    let gate_input = if half {
                        x.astype(Dtype::Float16)?
                    } else {
                        x.try_clone()?
                    };
                    let values = moe
                        .gate
                        .forward(&gate_input)?
                        .astype(Dtype::Float32)?
                        .to_f32()?;
                    ensure!(
                        values.len() == rows as usize * 512 && values.iter().all(|v| v.is_finite())
                    );
                    let prediction = moe.predict(&x)?;
                    ensure!(prediction.shape() == [rows, PREDICTION_RANKS as i32]);
                    let ids = prediction.to_u32()?;
                    ensure!(ids.len() == rows as usize * PREDICTION_RANKS);
                    for row in 0..rows as usize {
                        let ids = &ids[row * PREDICTION_RANKS..(row + 1) * PREDICTION_RANKS];
                        ensure!(
                            ids.iter().all(|&id| id < 512)
                                && ids.iter().copied().collect::<BTreeSet<_>>().len()
                                    == PREDICTION_RANKS
                        );
                        let mut oracle = values[row * 512..(row + 1) * 512].to_vec();
                        oracle.sort_by(f32::total_cmp);
                        let mut selected = ids
                            .iter()
                            .map(|&id| values[row * 512 + id as usize])
                            .collect::<Vec<_>>();
                        selected.sort_by(f32::total_cmp);
                        ensure!(
                            selected == oracle[512 - PREDICTION_RANKS..],
                            "prediction differs from independent CPU sort"
                        );
                    }
                    let mut ordered = ids;
                    ensure!(order_predictions(&mut ordered, rows as usize));
                }
            }
        }
        ensure!(moe.forward(&input)?.to_bytes()? == expected);
        let pool = Rc::new(RefCell::new(ReaderPool::new()?));
        moe.prefetcher = Some((0, Rc::clone(&pool)));
        moe.cache.borrow_mut().invalidate();
        moe.prefetch(&[0, 0, 1, 2, 511], 0)?;
        ensure!(moe.prefetch(&[0; 33], 4).is_err());
        ensure!(
            pool.borrow()
                .counters
                .submitted
                .load(std::sync::atomic::Ordering::Relaxed)
                == 0
        );
        moe.prefetch(&[0, 0, 1, 2, 511], 2)?;
        ensure!(
            pool.borrow()
                .counters
                .submitted
                .load(std::sync::atomic::Ordering::Relaxed)
                == 2
        );
        pool.borrow_mut().wait_for_test()?;
        moe.prefetch(&[0, 0, 1, 2, 511], 4)?;
        ensure!(
            pool.borrow()
                .counters
                .submitted
                .load(std::sync::atomic::Ordering::Relaxed)
                == 4
        );
        pool.borrow_mut().wait_for_test()?;
        ensure!(moe.forward(&input)?.to_bytes()? == expected);
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU and Flash fixtures; speculative fills and recovery"]
    fn lookahead_preserves_exact_routes_and_invalidates_failed_predictions() -> anyhow::Result<()> {
        use super::*;
        use std::path::Path;
        let root = std::env::var("MLXL3_FLASH_REFERENCE")?;
        let root = Path::new(&root);
        let reference: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(root.join("reference.json"))?)?;
        let fixture = &reference["steps"][0]["components"][0]["mlp"];
        let bytes =
            std::fs::read(root.join(fixture["input"]["file"].as_str().context("missing input")?))?;
        let expected = std::fs::read(
            root.join(
                fixture["output"]["file"]
                    .as_str()
                    .context("missing output")?,
            ),
        )?;
        ensure!(bytes.len() == H as usize * 4 && expected.len() == bytes.len());
        let input = Array::from_bytes(&bytes, &[1, H], Dtype::Float32)?;
        let checkpoint =
            crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        for slots in [10, 20, 24, 89] {
            let mut current = SsdMoe::load(
                &checkpoint,
                "model.language_model.layers.0.mlp",
                slots,
                true,
            )?;
            let mut next = SsdMoe::load(
                &checkpoint,
                "model.language_model.layers.0.mlp",
                slots,
                true,
            )?;
            let pool = Rc::new(RefCell::new(ReaderPool::new()?));
            next.prefetcher = Some((1, Rc::clone(&pool)));
            for rows in [1, 2] {
                for depth in [0, 2, 4, 8, 10] {
                    pool.borrow_mut().wait_for_test()?;
                    pool.borrow_mut().poll();
                    next.cache.borrow_mut().invalidate();
                    let x = if rows == 1 {
                        input.try_clone()?
                    } else {
                        Array::concatenate(&[&input, &input], 0)?
                    };
                    let before = pool
                        .borrow()
                        .counters
                        .submitted
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let output = current.forward_lookahead(&x, Some(&mut next), depth)?;
                    ensure!(
                        output.shape() == [rows, H]
                            && output.to_bytes()? == expected.repeat(rows as usize),
                        "lookahead changed MoE output"
                    );
                    let after = pool
                        .borrow()
                        .counters
                        .submitted
                        .load(std::sync::atomic::Ordering::Relaxed);
                    if depth > 0 && slots > 20 {
                        ensure!(after > before, "test did not submit predictions");
                    } else {
                        ensure!(
                            after == before,
                            "minimum cache must preserve demand capacity"
                        );
                    }
                    pool.borrow_mut().wait_for_test()?;
                    ensure!(
                        next.forward(&input)?.to_bytes()? == expected,
                        "prefetched weights differ"
                    );
                }
            }
            pool.borrow_mut().wait_for_test()?;
            pool.borrow_mut().poll();
            next.cache.borrow_mut().invalidate();
            let selected = next.route(&input)?.0.to_u32()?;
            let before = next.prefetched;
            next.prefetch(&selected, 4)?;
            pool.borrow_mut().wait_for_test()?;
            ensure!(next.forward(&input)?.to_bytes()? == expected);
            ensure!(
                if slots > 20 {
                    next.prefetched > before
                } else {
                    next.prefetched == before
                },
                "persistent admission or minimum-cache fallback was not exercised"
            );
            next.cache.borrow_mut().invalidate();
            let wrong = (0..512)
                .filter(|id| !selected.contains(id))
                .take(8)
                .collect::<Vec<_>>();
            ensure!(wrong.len() == 8);
            next.prefetch(&wrong, 4)?;
            let before = next.prefetched;
            pool.borrow_mut().wait_for_test()?;
            let ready = wrong
                .iter()
                .copied()
                .filter(|&id| next.cache.borrow().contains(id))
                .collect::<Vec<_>>();
            ensure!(ready.len() == if slots > 20 { 4 } else { 0 });
            ensure!(
                next.forward(&input)?.to_bytes()? == expected && next.prefetched == before,
                "wrong predictions changed real routing"
            );
            ensure!(
                ready.iter().all(|&id| next.cache.borrow().contains(id)),
                "unused records did not persist across a real forward"
            );
            let submitted = pool
                .borrow()
                .counters
                .submitted
                .load(std::sync::atomic::Ordering::Relaxed);
            if !ready.is_empty() {
                next.prefetch(&ready, 4)?;
                ensure!(
                    submitted
                        == pool
                            .borrow()
                            .counters
                            .submitted
                            .load(std::sync::atomic::Ordering::Relaxed),
                    "cached records were read twice"
                );
            }
            next.cache.borrow_mut().invalidate();
            next.prefetch(&[0, 511, 0], 4)?;
            let count = pool
                .borrow()
                .counters
                .submitted
                .load(std::sync::atomic::Ordering::Relaxed);
            next.prefetch(&[0, 511, 0], 4)?;
            ensure!(
                pool.borrow()
                    .counters
                    .submitted
                    .load(std::sync::atomic::Ordering::Relaxed)
                    == count,
                "duplicate speculative read"
            );
            pool.borrow_mut().wait_for_test()?;
            pool.borrow_mut().poll();
            ensure!(next.prefetch(&[512], 4).is_err() && next.prefetch(&[], 4).is_err());
            next.cache.borrow_mut().invalidate();
            let saved = next.records[0].data_offsets;
            next.records[0].data_offsets[0] = u64::MAX;
            next.prefetch(&[0, 511], 4)?;
            pool.borrow_mut().wait_for_test()?;
            ensure!(
                !next.cache.borrow_mut().reserve(&[0])?[0].1,
                "failed prediction left a false hit"
            );
            next.cache.borrow_mut().invalidate();
            next.records[0].data_offsets = saved;
            ensure!(
                next.forward(&input)?.to_bytes()? == expected,
                "prediction failure recovery differs"
            );
            ensure!(
                current
                    .forward_lookahead(&input, Some(&mut next), 11)
                    .is_err()
            );
            current.drain()?;
            next.drain()?;
        }
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU and independent Flash MoE inputs"]
    fn two_route_rows_match_oracle_with_cold_hot_small_caches_and_failed_reads()
    -> anyhow::Result<()> {
        use super::*;
        use serde_json::Value;
        use std::{fs::File, path::Path};
        let path = std::env::var("MLXL3_FLASH_REFERENCE")?;
        let root = Path::new(&path);
        let reference: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
        ensure!(
            reference["status"] == "complete" && reference["mode"] == "eval",
            "invalid MoE reference"
        );
        let checkpoint =
            crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        for step in [0, 1, 4] {
            let entry = &reference["steps"][step]["components"][0]["mlp"];
            let read = |kind: &str| -> Result<Vec<u8>> {
                ensure!(entry[kind]["count"] == H, "invalid MoE fixture count");
                let bytes = std::fs::read(
                    root.join(entry[kind]["file"].as_str().context("missing fixture")?),
                )?;
                ensure!(
                    bytes.len() == H as usize * 4
                        && bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .all(|&b| f32::from_le_bytes(b).is_finite()),
                    "invalid MoE fixture bytes"
                );
                Ok(bytes)
            };
            inputs.push(Array::from_bytes(&read("input")?, &[1, H], Dtype::Float32)?);
            expected.push(read("output")?);
        }
        for slots in [10, 19, 20, 24, 71, 89] {
            for (overlap, readers) in [
                (false, 1),
                (true, 1),
                (false, 2),
                (true, 2),
                (false, 4),
                (true, 4),
            ] {
                let mut moe = SsdMoe::load(
                    &checkpoint,
                    "model.language_model.layers.0.mlp",
                    slots,
                    overlap,
                )?;
                moe.readers = readers;
                for pair in [[0, 1], [0, 1], [1, 0], [0, 0], [1, 2]] {
                    let x = Array::concatenate(&[&inputs[pair[0]], &inputs[pair[1]]], 0)?;
                    let output = moe.forward_many(&x)?.to_bytes()?;
                    let mut oracle = expected[pair[0]].clone();
                    oracle.extend(&expected[pair[1]]);
                    ensure!(
                        output == oracle,
                        "two-row MoE differs, slots={slots}, overlap={overlap}, pair={pair:?}"
                    );
                }
                for rows in [1, 2, 4, 8] {
                    let indices = (0..rows).map(|i| i % 3).collect::<Vec<_>>();
                    let x = Array::concatenate(
                        &indices.iter().map(|&i| &inputs[i]).collect::<Vec<_>>(),
                        0,
                    )?;
                    let oracle = indices
                        .iter()
                        .flat_map(|&i| expected[i].iter().copied())
                        .collect::<Vec<_>>();
                    for _ in 0..2 {
                        ensure!(
                            moe.forward_draft_rows(&x)?.to_bytes()? == oracle,
                            "wide MoE differs, slots={slots}, overlap={overlap}, rows={rows}"
                        );
                    }
                }
                let counters = (moe.hits, moe.misses, moe.read_bytes);
                for shape in [[0, H], [3, H], [2, H - 1]] {
                    ensure!(
                        moe.forward_many(&Array::zeros(&shape)?).is_err(),
                        "invalid route shape accepted"
                    );
                }
                ensure!(
                    moe.forward_many(&Array::zeros_dtype(&[1, H], Dtype::UInt32)?)
                        .is_err(),
                    "integer route input accepted"
                );
                for shape in [[0, H], [9, H], [4, H - 1]] {
                    ensure!(
                        moe.forward_draft_rows(&Array::zeros(&shape)?).is_err(),
                        "invalid draft route shape accepted"
                    );
                }
                ensure!(
                    counters == (moe.hits, moe.misses, moe.read_bytes),
                    "invalid shape changed cache counters"
                );
                moe.drain()?;
                moe.cache.borrow_mut().invalidate();
                let expert = moe.route(&inputs[0])?.0.to_u32()?[0] as usize;
                let original = moe.records[expert].data_offsets;
                moe.records[expert].data_offsets[0] = u64::MAX;
                let wide =
                    Array::concatenate(&[&inputs[0], &inputs[1], &inputs[0], &inputs[2]], 0)?;
                ensure!(
                    moe.forward_draft_rows(&wide).is_err(),
                    "failed source accepted"
                );
                let ids = moe.route(&inputs[0])?.0.to_u32()?;
                ensure!(
                    moe.cache.borrow_mut().reserve(&ids)?.iter().all(|p| !p.1),
                    "failed read left valid reservations"
                );
                moe.cache.borrow_mut().invalidate();
                moe.records[expert].data_offsets = original;
                ensure!(
                    moe.forward(&inputs[0])?.to_bytes()? == expected[0],
                    "retry after failed read diverged"
                );
                moe.drain()?;
            }
        }
        Ok(())
    }
    #[test]
    #[ignore = "physical Apple GPU; direct disk read into evaluated shared buffers"]
    fn disk_records_fill_only_owned_ranges_and_reject_bad_sources() -> anyhow::Result<()> {
        use crate::array::{Array, Dtype};
        use std::io::Write;
        let mut file = tempfile::tempfile()?;
        file.write_all(&(0..64u8).collect::<Vec<_>>())?;
        let mut banks = vec![
            Array::zeros_dtype(&[16], Dtype::UInt8)?,
            Array::zeros_dtype(&[16], Dtype::UInt8)?,
        ];
        Array::read_record(&file, 7, &mut banks, &[(0, 3, 5), (1, 9, 4), (0, 11, 2)])?;
        let mut a = vec![0u8; 16];
        a[3..8].copy_from_slice(&[7, 8, 9, 10, 11]);
        a[11..13].copy_from_slice(&[16, 17]);
        let mut b = vec![0u8; 16];
        b[9..13].copy_from_slice(&[12, 13, 14, 15]);
        assert_eq!(banks[0].to_bytes()?, a);
        assert_eq!(banks[1].to_bytes()?, b);
        for (offset, ranges) in [
            (0, vec![(2, 0, 1)]),
            (0, vec![(0, 16, 1)]),
            (0, vec![(0, usize::MAX, 1)]),
            (63, vec![(0, 0, 2)]),
            (u64::MAX, vec![(0, 0, 1)]),
            (0, vec![(0, 0, 0)]),
            (0, vec![]),
        ] {
            assert!(Array::read_record(&file, offset, &mut banks, &ranges).is_err());
            assert_eq!(banks[0].to_bytes()?, a);
            assert_eq!(banks[1].to_bytes()?, b);
        }
        Ok(())
    }

    #[test]
    #[ignore = "physical Apple GPU; concurrent disk reads into shared buffers"]
    fn disk_batches_reject_aliases_and_validate_every_record_before_writing() -> anyhow::Result<()>
    {
        use crate::array::{Array, Dtype};
        use std::io::Write;
        let mut file = tempfile::tempfile()?;
        file.write_all(&(0..64u8).collect::<Vec<_>>())?;
        for workers in 1..=4 {
            let mut banks = vec![Array::zeros_dtype(&[16], Dtype::UInt8)?];
            Array::read_records(
                &[(&file, 8), (&file, 20)],
                &mut banks,
                &[&[(0, 1, 4), (0, 5, 2)], &[(0, 9, 3), (0, 14, 2)]],
                workers,
            )?;
            let expected = vec![0, 8, 9, 10, 11, 12, 13, 0, 0, 20, 21, 22, 0, 0, 23, 24];
            assert_eq!(banks[0].to_bytes()?, expected);
            banks.push(banks[0].clone());
            for destinations in [
                vec![vec![(0, 0, 8)], vec![(0, 7, 2)]],
                vec![vec![(0, 0, 8)], vec![(1, 7, 2)]],
                vec![vec![(0, 0, 8)], vec![(0, 16, 2)]],
                vec![vec![(0, 0, 8)], vec![(2, 0, 2)]],
                vec![vec![(0, 0, 8)], vec![(0, 8, 2), (0, 10, 2)]],
            ] {
                let views = destinations.iter().map(Vec::as_slice).collect::<Vec<_>>();
                assert!(
                    Array::read_records(&[(&file, 0), (&file, 16)], &mut banks, &views, workers)
                        .is_err()
                );
                assert_eq!(banks[0].to_bytes()?, expected);
            }
            assert!(
                Array::read_records(
                    &[(&file, 0), (&file, 64)],
                    &mut banks,
                    &[&[(0, 0, 1)], &[(0, 1, 1)]],
                    workers
                )
                .is_err()
            );
            assert_eq!(banks[0].to_bytes()?, expected);
        }
        Ok(())
    }
}
