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
            n > 0 && slots.iter().all(|&s| (s as usize) < self.cache.capacity()),
            "invalid expert slot mapping"
        );
        let indices = Array::from_u32(slots, &[n])?;
        let expanded = x
            .reshape(&[1, 1, H])?
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
            x.shape() == [1, H],
            "streamed MoE expects one exact route row"
        );
        let probabilities = self
            .gate
            .forward(x)?
            .astype(Dtype::Float32)?
            .softmax_precise()?;
        router::topk(&probabilities, 10, true)
    }
    pub fn forward(&mut self, x: &Array) -> Result<Array> {
        let (selected, weights) = self.route(x)?;
        let ids = selected.to_u32()?;
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
                self.hits += 1;
            } else {
                cold.push((i, slot as u32));
                self.misses += 1;
            }
        }
        let hot_output = if self.overlap && !hot.is_empty() && !cold.is_empty() {
            let value = self.calculate(
                &x.astype(Dtype::Float16)?,
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
                &x.astype(Dtype::Float16)?,
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
                &x.astype(Dtype::Float16)?,
                &routes.iter().map(|p| p.0 as u32).collect::<Vec<_>>(),
            )?
        };
        let routed = values
            .astype(Dtype::Float32)?
            .mul(&weights.reshape(&[10, 1])?)?
            .sum(0, true)?;
        let shared = self
            .shared_down
            .forward(
                &self
                    .shared_gate
                    .forward(x)?
                    .astype(Dtype::Float16)?
                    .swiglu(&self.shared_up.forward(x)?.astype(Dtype::Float16)?)?,
            )?
            .astype(Dtype::Float32)?;
        let output = routed.add(&shared.mul(&self.multiplier.forward(x)?.sigmoid()?)?)?;
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
