//! Flash PLE compressed row cache shared by the native and reference loaders.
use crate::{
    checkpoint::{Checkpoint, TensorInfo},
    codec::{self, Codebook},
    flash_support,
};
use anyhow::{Context, Result, ensure};
use half::f16;
use std::{
    collections::{BTreeMap, VecDeque},
    os::unix::fs::FileExt,
};
pub(crate) const EOS: u32 = 248044;

struct NgramsWeights {
    shards: Vec<TensorInfo>,
    sizes: [u64; 16],
    offsets: [u64; 16],
    multipliers: [u64; 3],
    bias: Vec<f16>,
}
pub(crate) struct Ngrams {
    weights: NgramsWeights,
    cache: BTreeMap<u64, [u16; 41]>,
    queue: VecDeque<u64>,
    pub(crate) previous: [u32; 2],
    pub(crate) read_bytes: u64,
}
impl Ngrams {
    pub(crate) fn load(c: &Checkpoint, p: &str) -> Result<Self> {
        let integers = |name: &str, n: usize| -> Result<Vec<u64>> {
            let tensor = c
                .tensors
                .get(&format!("{p}.{name}"))
                .context("missing PLE hash metadata")?;
            ensure!(
                tensor.dtype == "I64" && tensor.shape == [n],
                "invalid PLE hash metadata"
            );
            Ok(tensor
                .read_bytes()?
                .as_chunks::<8>()
                .0
                .iter()
                .map(|b| u64::from_le_bytes(*b))
                .collect())
        };
        let sizes: [u64; 16] = integers("head_vocab_sizes", 16)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("wrong PLE heads"))?;
        let offsets: [u64; 16] = integers("head_offsets", 16)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("wrong PLE offsets"))?;
        let multipliers: [u64; 3] = integers("layer_multipliers", 3)?
            .try_into()
            .map_err(|_| anyhow::anyhow!("wrong PLE multipliers"))?;
        let mut end = 0;
        for head in 0..16 {
            ensure!(
                offsets[head] == end && sizes[head] > 0 && sizes[head] <= i64::MAX as u64,
                "invalid PLE head range"
            );
            end = end
                .checked_add(sizes[head])
                .context("PLE row count overflow")?;
        }
        ensure!(
            end <= 320001536 && end > 320001408,
            "unsupported PLE row count"
        );
        let bias = c
            .tensors
            .get(&format!("{p}.head_bias"))
            .context("missing PLE head bias")?;
        ensure!(
            bias.shape == [16, 160] && bias.dtype == "F16",
            "invalid PLE head bias"
        );
        let bias = bias
            .read_bytes()?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&v| f16::from_le_bytes(v))
            .collect();
        let mut shards = Vec::with_capacity(128);
        for index in 0..128 {
            let tensor = c
                .tensors
                .get(&format!("{p}.shard_{index}.trellis"))
                .context("missing PLE trellis shard")?;
            ensure!(
                tensor.dtype == "I16" && tensor.shape == [2500012, 41],
                "unsupported PLE row format"
            );
            shards.push(tensor.clone());
        }
        Ok(Self {
            weights: NgramsWeights {
                shards,
                sizes,
                offsets,
                multipliers,
                bias,
            },
            cache: BTreeMap::new(),
            queue: VecDeque::new(),
            previous: [EOS; 2],
            read_bytes: 0,
        })
    }
    pub(crate) fn cache_bytes(&self) -> usize {
        self.cache.len() * 82
    }
    pub(crate) fn read(&mut self, token: u32) -> Result<Vec<u16>> {
        ensure!(token < 248320, "token outside Flash vocabulary");
        let history = flash_support::segment_tokens(token, self.previous, EOS);
        let mut output = Vec::with_capacity(2560);
        for head in 0..16 {
            let row = flash_support::ngram_row(
                history,
                self.weights.multipliers,
                2 + head / 8,
                self.weights.offsets[head],
                self.weights.sizes[head],
            )
            .context("invalid PLE row")?;
            if !self.cache.contains_key(&row) {
                let shard = &self.weights.shards[(row / 2500012) as usize];
                let (file, base) = shard.source()?;
                let offset = flash_support::row_offset(base, row % 2500012, 2500012, 82)
                    .context("PLE offset overflow")?;
                let mut raw = [0u8; 82];
                file.read_exact_at(&mut raw, offset)?;
                let words =
                    std::array::from_fn(|i| u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]));
                if self.cache.len() == 16384
                    && let Some(old) = self.queue.pop_front()
                {
                    self.cache.remove(&old);
                }
                self.cache.insert(row, words);
                self.queue.push_back(row);
                self.read_bytes += 82;
            }
            let words = &self.cache[&row];
            let scale = f16::from_bits(words[0]).to_f32();
            ensure!(scale.is_finite(), "nonfinite PLE row scale");
            for i in 0..160 {
                let state =
                    flash_support::ple_codeword(words, i).context("PLE component outside row")?;
                let value = f16::from_f32(
                    codec::decode_codeword(state, Codebook::Mul1).to_f32() * scale
                        + self.weights.bias[head * 160 + i].to_f32(),
                );
                ensure!(value.is_finite(), "nonfinite PLE embedding");
                output.push(value.to_bits());
            }
        }
        self.previous = [token, self.previous[0]];
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "physical PLE checkpoint and late packed-row read failure"]
    fn a_failed_tail_read_preserves_token_history_and_reuses_valid_heads() -> Result<()> {
        let path = std::env::var_os("MLXL3_PLE_MODEL").context("missing model")?;
        let c = crate::checkpoint::inspect(std::path::Path::new(&path))?;
        let mut rows = Ngrams::load(
            &c,
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding",
        )?;
        let previous = rows.previous;
        let history = flash_support::segment_tokens(42, previous, EOS);
        let row = flash_support::ngram_row(
            history,
            rows.weights.multipliers,
            3,
            rows.weights.offsets[15],
            rows.weights.sizes[15],
        )
        .context("missing row")?;
        let shard = (row / 2500012) as usize;
        let old = rows.weights.shards[shard].data_offsets;
        rows.weights.shards[shard].data_offsets = [u64::MAX; 2];
        ensure!(rows.read(42).is_err() && rows.previous == previous && rows.read_bytes == 15 * 82);
        rows.weights.shards[shard].data_offsets = old;
        let expected = rows.read(42)?;
        ensure!(expected.len() == 2560 && rows.previous == [42, EOS] && rows.read_bytes == 16 * 82);
        rows.previous = previous;
        ensure!(rows.read(248320).is_err() && rows.previous == previous);
        ensure!(rows.read(42)? == expected && rows.read_bytes == 16 * 82);
        Ok(())
    }
}
