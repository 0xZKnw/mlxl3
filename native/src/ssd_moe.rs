//! Exact routed EXL3 experts, read directly into reusable slot-major banks.
use crate::{
    array::{Array, Dtype},
    checkpoint::{Checkpoint, TensorInfo},
    codec::Codebook,
    flash_support::{EXPERT_BYTES, ExpertCache},
    lfm2::Projection,
    linear, router,
};
use anyhow::{Context, Result, ensure};
use std::collections::BTreeSet;

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
    cache: ExpertCache,
    last: Option<Array>,
    pub hits: u64,
    pub misses: u64,
    pub read_bytes: u64,
    overlap: bool,
    #[cfg(test)]
    pub(crate) route_ids: Vec<u32>,
}

impl SsdMoe {
    pub fn load(c: &Checkpoint, prefix: &str, slots: usize, overlap: bool) -> Result<Self> {
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
        let cache = ExpertCache::new(slots)?;
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
            #[cfg(test)]
            route_ids: Vec::new(),
        })
    }

    fn fill(&mut self, expert: u32, slot: usize) -> Result<()> {
        let (file, offset) = self.records[expert as usize].source()?;
        let dest = [
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
        ];
        Array::read_record(file, offset, &mut self.banks, &dest)?;
        self.read_bytes = self
            .read_bytes
            .checked_add(EXPERT_BYTES)
            .context("SSD byte counter exhausted")?;
        Ok(())
    }

    fn calculate(&self, x: &Array, slots: &[u32]) -> Result<Array> {
        let n = i32::try_from(slots.len())?;
        ensure!(
            n > 0
                && x.shape() == [n, H]
                && slots.iter().all(|&s| (s as usize) < self.cache.capacity()),
            "invalid expert slot mapping"
        );
        let indices = Array::from_u32(slots, &[n])?;
        let expanded = x
            .reshape(&[n, 1, H])?
            .broadcast_to(&[n, 2, H])?
            .reshape(&[n * 2, H])?;
        let suh = self.banks[2].take(&indices, 0)?.reshape(&[n * 2, H])?;
        let svh = self.banks[3].take(&indices, 0)?.reshape(&[n * 2, I])?;
        let gu = linear::expert_slots(
            &expanded,
            &self.banks[0],
            &suh,
            &svh,
            &indices,
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
            &self.banks[4].take(&indices, 0)?.reshape(&[n, I])?,
            &self.banks[5].take(&indices, 0)?.reshape(&[n, H])?,
            &indices,
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
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        self.forward_many(x)
    }

    pub(crate) fn warm_all(&mut self) -> Result<()> {
        self.drain()?;
        ensure!(
            self.cache.capacity() == 512,
            "resident draft requires all expert slots"
        );
        let ids = (0..512u32).collect::<Vec<_>>();
        let routes = self.cache.reserve(&ids)?;
        for (&id, &(slot, hit)) in ids.iter().zip(&routes) {
            if !hit && let Err(error) = self.fill(id, slot) {
                self.cache.invalidate();
                return Err(error);
            }
        }
        Ok(())
    }

    pub(crate) fn forward_many(&mut self, x: &Array) -> Result<Array> {
        ensure!(
            x.shape().len() == 2
                && (1..=2).contains(&x.shape()[0])
                && x.shape()[1] == H
                && matches!(x.dtype(), Dtype::Float16 | Dtype::Float32),
            "streamed MoE expects one or two route rows"
        );
        let rows = x.shape()[0];
        if rows == 2 && self.cache.capacity() < 20 {
            let a = self.forward(&x.slice(0, 0, 1)?)?;
            let b = self.forward(&x.slice(0, 1, 2)?)?;
            return Array::concatenate(&[&a, &b], 0);
        }
        let pairs = (0..rows)
            .map(|row| self.route(&x.slice(0, row, row + 1)?))
            .collect::<Result<Vec<_>>>()?;
        let selected = Array::concatenate(&pairs.iter().map(|p| &p.0).collect::<Vec<_>>(), 0)?;
        let weights = Array::concatenate(&pairs.iter().map(|p| &p.1).collect::<Vec<_>>(), 0)?;
        let ids = selected.to_u32()?;
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
        let routes = self.cache.reserve(&ids)?;
        let mut hot = Vec::new();
        let mut cold = Vec::new();
        let mut order = Vec::new();
        for (i, &(slot, hit)) in routes.iter().enumerate() {
            if hit {
                hot.push((i, slot as u32));
                self.hits = self.hits.saturating_add(1);
            } else {
                cold.push((i, slot as u32));
                self.misses = self.misses.saturating_add(1);
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
            )?;
            value.async_eval()?;
            Some(value)
        } else {
            None
        };
        let mut filled = BTreeSet::new();
        for &(i, slot) in &cold {
            if filled.insert(slot)
                && let Err(error) = self.fill(ids[i], slot as usize)
            {
                self.cache.invalidate();
                if let Some(hot) = &hot_output {
                    hot.eval()?;
                }
                return Err(error.context("loading cold EXL3 expert"));
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
        Ok(output)
    }

    pub fn drain(&mut self) -> Result<()> {
        if let Some(last) = self.last.take() {
            last.eval()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
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
        for slots in [10, 20, 143] {
            for overlap in [false, true] {
                let mut moe = SsdMoe::load(
                    &checkpoint,
                    "model.language_model.layers.0.mlp",
                    slots,
                    overlap,
                )?;
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
                ensure!(
                    counters == (moe.hits, moe.misses, moe.read_bytes),
                    "invalid shape changed cache counters"
                );
                moe.drain()?;
                moe.cache.invalidate();
                let expert = moe.route(&inputs[0])?.0.to_u32()?[0] as usize;
                let original = moe.records[expert].data_offsets;
                moe.records[expert].data_offsets[0] = u64::MAX;
                ensure!(
                    moe.forward_many(&inputs[0]).is_err(),
                    "failed source accepted"
                );
                let ids = moe.route(&inputs[0])?.0.to_u32()?;
                ensure!(
                    moe.cache.reserve(&ids)?.iter().all(|p| !p.1),
                    "failed read left valid reservations"
                );
                moe.cache.invalidate();
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
}
