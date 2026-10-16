//! Experimental LFM KV capacity cache, selected only by a qualified V2 opt-in.

use std::convert::TryFrom;

pub(crate) const CAPACITY_STEP: i32 = 256;

/// Only the exact experimental opt-in selects the already qualified V2 path.
pub(crate) fn capacity_enabled(flag: Option<&[u8]>, qualified_v2: bool) -> bool {
    qualified_v2 && matches!(flag, Some([b'1']))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AppendPlan {
    pub end: i32,
    pub capacity: i32,
}

/// Validate the append before constructing any MLX graph or changing its state.
pub(crate) fn append_plan(
    offset: i32,
    capacity: i32,
    rows: i32,
    context_limit: i32,
) -> Option<AppendPlan> {
    if offset < 0
        || capacity < offset
        || capacity > context_limit
        || rows <= 0
        || context_limit <= 0
    {
        return None;
    }
    let end = offset.checked_add(rows)?;
    if end > context_limit {
        return None;
    }
    let capacity = if end <= capacity {
        capacity
    } else {
        let rounded = ((i64::from(end) + i64::from(CAPACITY_STEP) - 1) / i64::from(CAPACITY_STEP))
            * i64::from(CAPACITY_STEP);
        rounded.min(i64::from(context_limit)) as i32
    };
    Some(AppendPlan { end, capacity })
}

/// Bytes for the two F16 tensors [1, heads, capacity, head_dim], excluding aliases.
pub(crate) fn kv_storage_bytes(heads: i32, capacity: i32, head_dim: i32) -> Option<usize> {
    if heads <= 0 || capacity <= 0 || head_dim <= 0 {
        return None;
    }
    usize::try_from(heads)
        .ok()?
        .checked_mul(usize::try_from(capacity).ok()?)?
        .checked_mul(usize::try_from(head_dim).ok()?)?
        .checked_mul(4)
}

#[cfg(feature = "mlx")]
mod gpu {
    use super::{append_plan, kv_storage_bytes};
    use crate::array::{Array, Dtype};
    use anyhow::{Context, Result, ensure};

    /// MLX owns donation and copy-on-write. Only the current logical views are
    /// retained here; callers may keep independent immutable Array clones.
    pub(crate) struct KvCache {
        heads: i32,
        head_dim: i32,
        context_limit: i32,
        offset: i32,
        storage: Option<[Array; 2]>,
        views: Option<[Array; 2]>,
    }

    impl KvCache {
        pub(crate) fn new(heads: i32, head_dim: i32, context_limit: i32) -> Result<Self> {
            kv_storage_bytes(heads, context_limit, head_dim)
                .context("invalid or overflowing LFM KV geometry")?;
            Ok(Self {
                heads,
                head_dim,
                context_limit,
                offset: 0,
                storage: None,
                views: None,
            })
        }

        pub(crate) fn position(&self) -> i32 {
            self.offset
        }

        pub(crate) fn capacity(&self) -> i32 {
            self.storage
                .as_ref()
                .map_or(0, |arrays| arrays[0].shape()[2])
        }

        pub(crate) fn views(&self) -> Option<&[Array; 2]> {
            self.views.as_ref()
        }

        /// Diagnostic only: no descriptor/storage clone. This materializes the
        /// buffers and therefore does not model unsynchronized bridge lifetime.
        #[cfg(test)]
        pub(crate) fn backing_identities(&self) -> Result<Option<[u64; 2]>> {
            self.storage
                .as_ref()
                .map(|storage| {
                    Ok([
                        storage[0].storage_identity()?,
                        storage[1].storage_identity()?,
                    ])
                })
                .transpose()
        }

        /// Logical state bytes stay distinct from reserved/backing bytes.
        pub(crate) fn logical_bytes(&self) -> Result<usize> {
            if self.offset == 0 {
                return Ok(0);
            }
            kv_storage_bytes(self.heads, self.offset, self.head_dim)
                .context("LFM KV logical size overflow")
        }

        pub(crate) fn reserved_bytes(&self) -> Result<usize> {
            if self.capacity() == 0 {
                return Ok(0);
            }
            kv_storage_bytes(self.heads, self.capacity(), self.head_dim)
                .context("LFM KV reserved size overflow")
        }

        pub(crate) fn append(&mut self, keys: &Array, values: &Array) -> Result<&[Array; 2]> {
            // Invalid caller input cannot discard an already valid cache.
            let plan = self.validate(keys, values)?;
            self.append_validated(keys, values, plan)
        }

        fn append_validated(
            &mut self,
            keys: &Array,
            values: &Array,
            plan: super::AppendPlan,
        ) -> Result<&[Array; 2]> {
            if let Err(error) = self.append_inner(keys, values, plan) {
                self.reset();
                return Err(error.context("LFM KV append failed; cache reset"));
            }
            Ok(self.views.as_ref().expect("append created logical views"))
        }

        fn validate(&self, keys: &Array, values: &Array) -> Result<super::AppendPlan> {
            let shape = keys.shape();
            ensure!(
                shape.len() == 4
                    && shape[0] == 1
                    && shape[1] == self.heads
                    && shape[3] == self.head_dim
                    && shape == values.shape()
                    && keys.dtype() == Dtype::Float16
                    && values.dtype() == Dtype::Float16,
                "LFM KV updates must be matching F16 [1, heads, rows, head_dim] arrays"
            );
            let plan = append_plan(self.offset, self.capacity(), shape[2], self.context_limit)
                .context("invalid or overflowing LFM KV append range")?;
            kv_storage_bytes(self.heads, plan.capacity, self.head_dim)
                .context("LFM KV reserved size overflow")?;
            Ok(plan)
        }

        fn append_inner(
            &mut self,
            keys: &Array,
            values: &Array,
            plan: super::AppendPlan,
        ) -> Result<()> {
            // A retained logical view would prevent donation on every append.
            // External aliases remain valid and make MLX copy-on-write instead.
            self.views = None;
            if plan.capacity > self.capacity() {
                let padding_shape = [1, self.heads, plan.capacity - self.offset, self.head_dim];
                let new_keys = Array::zeros_dtype(&padding_shape, Dtype::Float16)?;
                let new_values = Array::zeros_dtype(&padding_shape, Dtype::Float16)?;
                self.storage = Some(if let Some(previous) = self.storage.take() {
                    let old_keys = previous[0].slice(2, 0, self.offset)?;
                    let old_values = previous[1].slice(2, 0, self.offset)?;
                    [
                        Array::concatenate(&[&old_keys, &new_keys], 2)?,
                        Array::concatenate(&[&old_values, &new_values], 2)?,
                    ]
                } else {
                    [new_keys, new_values]
                });
            }
            let storage = self.storage.as_mut().context("missing LFM KV storage")?;
            storage[0].update_slice(2, self.offset, plan.end, keys)?;
            storage[1].update_slice(2, self.offset, plan.end, values)?;
            self.views = Some([
                storage[0].slice(2, 0, plan.end)?,
                storage[1].slice(2, 0, plan.end)?,
            ]);
            self.offset = plan.end;
            Ok(())
        }

        pub(crate) fn reset(&mut self) {
            self.views = None;
            self.storage = None;
            self.offset = 0;
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        type Oracle = [Vec<Vec<u16>>; 2];

        fn oracle(heads: i32) -> Oracle {
            std::array::from_fn(|_| vec![Vec::new(); heads as usize])
        }

        fn words(heads: i32, start: i32, count: i32, width: i32, salt: u16) -> Vec<u16> {
            (0..heads)
                .flat_map(|head| {
                    (start..start + count).flat_map(move |row| {
                        (0..width).map(move |col| {
                            // Small finite half values, including both signs and signed zero.
                            let index = (head * 19 + row * 7 + col * 3) as u16 ^ salt;
                            (index & 0x83ff) | ((index % 14) << 10)
                        })
                    })
                })
                .collect()
        }

        fn rows(heads: i32, start: i32, count: i32, width: i32, salt: u16) -> Result<Array> {
            Array::from_f16_bits(
                &words(heads, start, count, width, salt),
                &[1, heads, count, width],
            )
        }

        // This oracle concatenates the actual input bytes by head on the CPU;
        // it neither repeats the generator nor uses an MLX concatenate/update.
        fn record(expected: &mut Oracle, keys: &Array, values: &Array) -> Result<()> {
            for (heads, input) in expected.iter_mut().zip([keys, values]) {
                let bits = input.to_f16_bits()?;
                let stride = input.shape()[2] as usize * input.shape()[3] as usize;
                for (head, update) in heads.iter_mut().zip(bits.chunks_exact(stride)) {
                    head.extend_from_slice(update);
                }
            }
            Ok(())
        }

        fn append_rows(
            cache: &mut KvCache,
            expected: &mut Oracle,
            count: i32,
            salt: u16,
        ) -> Result<()> {
            let keys = rows(cache.heads, cache.position(), count, cache.head_dim, salt)?;
            let values = rows(
                cache.heads,
                cache.position(),
                count,
                cache.head_dim,
                salt ^ 0x8241,
            )?;
            record(expected, &keys, &values)?;
            cache.append(&keys, &values)?;
            Ok(())
        }

        fn check(cache: &KvCache, expected: &Oracle) -> Result<()> {
            let views = cache.views().context("missing logical KV views")?;
            let storage = cache
                .storage
                .as_ref()
                .context("missing backing KV buffers")?;
            for ((array, backing), expected) in views.iter().zip(storage).zip(expected) {
                assert_eq!(
                    array.shape(),
                    [1, cache.heads, cache.position(), cache.head_dim]
                );
                assert_eq!(array.dtype(), Dtype::Float16);
                let bits = array.to_f16_bits()?;
                assert!(!bits.is_empty());
                assert!(bits.iter().all(|bits| bits & 0x7c00 != 0x7c00));
                assert_eq!(bits, expected.iter().flatten().copied().collect::<Vec<_>>());
                assert_eq!(array.retained_bytes()?, backing.retained_bytes()?);
                assert!(array.retained_bytes()? >= backing.byte_len()?);
            }
            // Compare attention against independently concatenated, dense CPU
            // inputs as well: capacity strides may select a different SDPA path.
            let shape = [1, cache.heads, cache.position(), cache.head_dim];
            let dense = expected
                .iter()
                .map(|heads| {
                    Array::from_f16_bits(
                        &heads.iter().flatten().copied().collect::<Vec<_>>(),
                        &shape,
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let query = rows(cache.heads * 4, 0, 1, cache.head_dim, 0x3156)?;
            let scale = 1.0 / (cache.head_dim as f32).sqrt();
            let actual = Array::sdpa(&query, &views[0], &views[1], scale, false)?;
            let reference = Array::sdpa(&query, &dense[0], &dense[1], scale, false)?;
            let actual_bits = actual.to_f16_bits()?;
            assert!(!actual_bits.is_empty());
            assert!(actual_bits.iter().all(|bits| bits & 0x7c00 != 0x7c00));
            assert_eq!(
                actual_bits,
                reference.to_f16_bits()?,
                "SDPA differed for capacity-strided KV"
            );
            assert_eq!(
                cache.logical_bytes()?,
                cache.position() as usize * cache.heads as usize * cache.head_dim as usize * 4
            );
            assert_eq!(
                cache.reserved_bytes()?,
                cache.capacity() as usize * cache.heads as usize * cache.head_dim as usize * 4
            );
            Ok(())
        }

        // An isolated alias branch exercises MLX COW. There is deliberately no
        // production LFM snapshot/fork API or parallel ownership counter.
        fn fork_for_test(cache: &KvCache) -> Result<KvCache> {
            let storage = cache.storage.as_ref().unwrap();
            let views = cache.views.as_ref().unwrap();
            Ok(KvCache {
                heads: cache.heads,
                head_dim: cache.head_dim,
                context_limit: cache.context_limit,
                offset: cache.offset,
                storage: Some([storage[0].try_clone()?, storage[1].try_clone()?]),
                views: Some([views[0].try_clone()?, views[1].try_clone()?]),
            })
        }

        #[test]
        #[ignore = "physical Apple GPU; run alone after root CPU/build/prover gate"]
        fn capacity_cache_matches_cpu_and_reuses_storage() -> Result<()> {
            for (heads, width) in [(1, 1), (2, 4), (8, 64)] {
                let mut cache = KvCache::new(heads, width, 517)?;
                let mut expected = oracle(heads);
                let mut previous = None;
                for end in 1..=517 {
                    append_rows(&mut cache, &mut expected, 1, 0)?;
                    let storage = cache.storage.as_ref().unwrap();
                    let identities = [
                        storage[0].storage_identity()?,
                        storage[1].storage_identity()?,
                    ];
                    if let Some((capacity, prior)) = previous
                        && capacity == cache.capacity()
                    {
                        assert_eq!(identities, prior, "backing copied without growth at {end}");
                    }
                    if [1, 23, 24, 255, 256, 257, 511, 512, 513, 517].contains(&end) {
                        check(&cache, &expected)?;
                    }
                    previous = Some((cache.capacity(), identities));
                }
                assert_eq!(cache.capacity(), 517);
                assert!(cache.reserved_bytes()? >= cache.logical_bytes()?);
                cache.reset();
                assert!(cache.views().is_none() && cache.storage.is_none());
                assert_eq!((cache.position(), cache.capacity()), (0, 0));
                assert_eq!((cache.logical_bytes()?, cache.reserved_bytes()?), (0, 0));
            }
            Ok(())
        }

        #[test]
        #[ignore = "physical Apple GPU; run alone after root CPU/build/prover gate"]
        fn capacity_cache_preserves_aliases_and_delayed_branches() -> Result<()> {
            let mut cache = KvCache::new(2, 4, 769)?;
            let mut expected = oracle(2);
            append_rows(&mut cache, &mut expected, 255, 0)?;
            check(&cache, &expected)?;
            let saved = cache.views().unwrap()[0].try_clone()?;
            let saved_bytes = saved.to_bytes()?;
            let before = cache.storage.as_ref().unwrap()[0].storage_identity()?;
            append_rows(&mut cache, &mut expected, 1, 0)?;
            let after = cache.storage.as_ref().unwrap()[0].storage_identity()?;
            assert_ne!(before, after, "live alias did not trigger COW");
            assert_eq!(saved.to_bytes()?, saved_bytes);
            check(&cache, &expected)?;
            drop(saved);
            append_rows(&mut cache, &mut expected, 1, 0)?;
            check(&cache, &expected)?;

            let mut branch = fork_for_test(&cache)?;
            let mut branch_expected = expected.clone();
            let saved = cache.views().unwrap()[1].try_clone()?;
            let saved_bytes = saved.to_bytes()?;
            let delayed = saved.slice(2, 0, cache.position())?;
            let query = Array::from_f16_bits(
                &[
                    0x3400, 0xb000, 0x3800, 0xba00, 0xb400, 0x3000, 0xb800, 0x3a00,
                ],
                &[1, 2, 1, 4],
            )?;
            // A real deferred attention consumer, with no identity/eval between appends.
            let views = cache.views().unwrap();
            let attention = Array::sdpa(&query, &views[0], &views[1], 0.5, false)?;
            append_rows(&mut cache, &mut expected, 7, 0)?;
            append_rows(&mut cache, &mut expected, 24, 0)?;
            append_rows(&mut branch, &mut branch_expected, 23, 0x2301)?;
            check(&cache, &expected)?;
            check(&branch, &branch_expected)?;
            check(&cache, &expected)?; // The other branch's evaluation must not overwrite A.
            assert_eq!(delayed.to_bytes()?, saved_bytes);
            let branch_views = branch.views().unwrap();
            let old_keys = branch_views[0].slice(2, 0, 257)?;
            let old_values = branch_views[1].slice(2, 0, 257)?;
            let expected_attention = Array::sdpa(&query, &old_keys, &old_values, 0.5, false)?;
            assert_eq!(attention.to_bytes()?, expected_attention.to_bytes()?);
            assert!(attention.to_f32()?.iter().all(|value| value.is_finite()));
            cache.reset();
            assert_eq!(saved.to_bytes()?, saved_bytes);
            check(&branch, &branch_expected)?;
            expected = oracle(2);
            append_rows(&mut cache, &mut expected, 24, 0)?;
            check(&cache, &expected)?;
            let abandoned = cache.views().unwrap()[0].try_clone()?;
            append_rows(&mut cache, &mut expected, 7, 0)?;
            drop(abandoned); // Drop after graph construction, before evaluation.
            check(&cache, &expected)?;
            eprintln!(
                "capacity cache final memory statistics: {:?}",
                crate::array::memory_stats(false)?
            );
            Ok(())
        }

        #[test]
        #[ignore = "physical Apple GPU; run alone after root CPU/build/prover gate"]
        fn capacity_cache_full_strided_input_and_validation() -> Result<()> {
            let mut cache = KvCache::new(2, 4, 257)?;
            let mut expected = oracle(2);
            // Transpose + narrow slice of a larger QKV-like parent. Expected
            // view indices are computed from CPU parent words, not read via MLX.
            let mut inputs = Vec::new();
            for (side, salt) in [0, 0x8241].into_iter().enumerate() {
                let parent = words(256, 0, 2, 4 * 32, salt);
                let input = Array::from_f16_bits(&parent, &[1, 256, 2, 4 * 32])?
                    .transpose(&[0, 2, 1, 3])?
                    .slice(3, 0, 4)?;
                for (head, expected_head) in expected[side].iter_mut().enumerate() {
                    for row in 0..256 {
                        let begin = (row * 2 + head) * 128;
                        expected_head.extend_from_slice(&parent[begin..begin + 4]);
                    }
                }
                inputs.push(input);
            }
            let values = inputs.pop().unwrap();
            let keys = inputs.pop().unwrap();
            cache.append(&keys, &values)?;
            check(&cache, &expected)?;
            for storage in cache.storage.as_ref().unwrap() {
                // MLX contiguous may retain at most its documented 16KiB slack.
                assert!(storage.retained_bytes()? <= storage.byte_len()? + 16_384);
            }
            append_rows(&mut cache, &mut expected, 1, 0)?;
            check(&cache, &expected)?;
            let position = cache.position();
            let valid = rows(2, position, 1, 4, 0)?;
            assert!(cache.append(&valid, &valid).is_err()); // Context is full.
            assert_eq!(cache.position(), position);
            check(&cache, &expected)?;
            assert!(KvCache::new(0, 4, 257).is_err());
            assert!(KvCache::new(2, -1, 257).is_err());
            assert!(KvCache::new(2, 4, 0).is_err());
            assert!(KvCache::new(i32::MAX, i32::MAX, i32::MAX).is_err());
            for invalid in [
                Array::zeros_dtype(&[1, 2, 1, 4], Dtype::Float32)?,
                Array::zeros_dtype(&[1, 1, 1, 4], Dtype::Float16)?,
                Array::zeros_dtype(&[1, 2, 1, 3], Dtype::Float16)?,
                Array::zeros_dtype(&[1, 2, 1], Dtype::Float16)?,
                Array::zeros_dtype(&[1, 2, 0, 4], Dtype::Float16)?,
                Array::zeros_dtype(&[2, 2, 1, 4], Dtype::Float16)?,
                Array::zeros_dtype(&[1, 2, 258, 4], Dtype::Float16)?,
            ] {
                // Leave room so invalid metadata cannot be masked by a full context.
                cache.reset();
                expected = oracle(2);
                append_rows(&mut cache, &mut expected, 1, 0)?;
                let position = cache.position();
                let capacity = cache.capacity();
                assert!(cache.append(&invalid, &invalid).is_err());
                assert_eq!((cache.position(), cache.capacity()), (position, capacity));
                check(&cache, &expected)?;
            }
            let keys = rows(2, cache.position(), 1, 4, 0)?;
            for invalid_values in [
                Array::zeros_dtype(&[1, 2, 1, 4], Dtype::Float32)?,
                Array::zeros_dtype(&[1, 2, 2, 4], Dtype::Float16)?,
            ] {
                let position = cache.position();
                assert!(cache.append(&keys, &invalid_values).is_err());
                assert_eq!(cache.position(), position);
                check(&cache, &expected)?;
            }
            // Deliberately bypass the cache metadata guard to trigger the real
            // Array error on V after K has already been replaced. Production uses
            // this same cleanup path for graph/allocation construction errors.
            let keys = rows(2, cache.position(), 1, 4, 0)?;
            let plan = cache.validate(&keys, &keys)?;
            let invalid_values = Array::zeros_dtype(keys.shape(), Dtype::Float32)?;
            assert!(
                cache
                    .append_validated(&keys, &invalid_values, plan)
                    .is_err()
            );
            assert!(cache.views().is_none() && cache.storage.is_none());
            assert_eq!((cache.position(), cache.capacity()), (0, 0));
            expected = oracle(2);
            append_rows(&mut cache, &mut expected, 1, 0)?;
            check(&cache, &expected)?;
            Ok(())
        }
    }
}

#[cfg(feature = "mlx")]
pub(crate) use gpu::KvCache;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_flag_requires_exact_opt_in_and_qualified_v2() {
        for (flag, exact_one) in [
            (None, false),
            (Some(&b""[..]), false),
            (Some(&b"0"[..]), false),
            (Some(&b"1"[..]), true),
            (Some(&b"01"[..]), false),
            (Some(&b"true"[..]), false),
            (Some(&b" 1"[..]), false),
            (Some(&b"1 "[..]), false),
            (Some(&[0xff][..]), false),
            (Some(&[b'1'; 64][..]), false),
        ] {
            assert!(!capacity_enabled(flag, false));
            assert_eq!(capacity_enabled(flag, true), exact_one);
        }
    }

    #[test]
    fn capacity_boundaries_and_context_clamping() {
        for (offset, capacity, rows, limit, expected) in [
            (0, 0, 1, 1024, Some((1, 256))),
            (0, 0, 255, 1024, Some((255, 256))),
            (0, 0, 256, 1024, Some((256, 256))),
            (0, 0, 257, 1024, Some((257, 512))),
            (255, 256, 1, 1024, Some((256, 256))),
            (256, 256, 1, 1024, Some((257, 512))),
            (256, 256, 1, 257, Some((257, 257))),
            (0, 0, 24, 25, Some((24, 25))),
            (0, 0, 1, 1, Some((1, 1))),
            (
                i32::MAX - 1,
                i32::MAX - 1,
                1,
                i32::MAX,
                Some((i32::MAX, i32::MAX)),
            ),
            (i32::MAX, i32::MAX, 1, i32::MAX, None),
            (0, 0, 0, 1024, None),
            (-1, 0, 1, 1024, None),
            (1, 0, 1, 1024, None),
            (0, -1, 1, 1024, None),
            (0, 512, 1, 257, None),
            (0, 0, 1, 0, None),
            (0, 0, -1, 1024, None),
            (1, 256, 257, 257, None),
        ] {
            assert_eq!(
                append_plan(offset, capacity, rows, limit).map(|p| (p.end, p.capacity)),
                expected,
                "offset={offset} capacity={capacity} rows={rows} limit={limit}"
            );
        }
    }

    #[test]
    fn capacity_plan_matches_independent_small_domain_oracle() {
        for limit in [1, 23, 24, 25, 255, 256, 257, 511, 512, 513, 1025] {
            for offset in 0..=limit {
                let capacity = ((offset + 255) / 256 * 256).min(limit);
                for rows in [0, 1, 2, 7, 23, 24, 25, 255, 256, 257, 513] {
                    let end = i64::from(offset) + i64::from(rows);
                    let expected = if rows == 0 || end > i64::from(limit) {
                        None
                    } else {
                        // Find the smallest allowed capacity covering the append.
                        let reserved = (capacity..=limit)
                            .find(|&n| {
                                n >= end as i32 && (n == capacity || n == limit || n % 256 == 0)
                            })
                            .unwrap();
                        Some(AppendPlan {
                            end: end as i32,
                            capacity: reserved,
                        })
                    };
                    assert_eq!(append_plan(offset, capacity, rows, limit), expected);
                }
            }
        }
    }

    #[test]
    fn f16_payload_budget_is_checked_and_separate_from_slack() {
        assert_eq!(kv_storage_bytes(8, 256, 64), Some(524_288));
        assert_eq!(kv_storage_bytes(8, 1, 64), Some(2_048));
        assert_eq!(
            10 * (kv_storage_bytes(8, 256, 64).unwrap() - kv_storage_bytes(8, 1, 64).unwrap()),
            5_222_400
        );
        assert_eq!(kv_storage_bytes(0, 256, 64), None);
        assert_eq!(kv_storage_bytes(8, 0, 64), None);
        assert_eq!(kv_storage_bytes(8, 256, -1), None);
        assert_eq!(kv_storage_bytes(i32::MAX, i32::MAX, i32::MAX), None);
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    /// Bounded flag bytes0..4; the integer capacity property below remains full i32.
    #[kani::proof]
    #[kani::unwind(6)]
    fn capacity_gate_requires_qualified_v2_and_exact_one() {
        let bytes: [u8; 4] = kani::any();
        let count = usize::from(kani::any::<u8>() % 5);
        let present: bool = kani::any();
        let qualified: bool = kani::any();
        let flag = if present { Some(&bytes[..count]) } else { None };
        let enabled = capacity_enabled(flag, qualified);
        assert_eq!(
            enabled,
            qualified && present && count == 1 && bytes[0] == b'1'
        );
        kani::cover!(enabled);
        kani::cover!(!qualified && present && count == 1 && bytes[0] == b'1');
        kani::cover!(!present);
        kani::cover!(present && count != 1);
    }

    #[kani::proof]
    fn accepted_appends_have_exact_bounded_ranges() {
        let offset: i32 = kani::any();
        let capacity: i32 = kani::any();
        let rows: i32 = kani::any();
        let limit: i32 = kani::any();
        let plan = append_plan(offset, capacity, rows, limit);
        let mathematical_end = i64::from(offset) + i64::from(rows);
        let valid = offset >= 0
            && capacity >= offset
            && capacity <= limit
            && rows > 0
            && limit > 0
            && mathematical_end <= i64::from(limit);
        assert_eq!(plan.is_some(), valid);
        if let Some(plan) = plan {
            assert_eq!(i64::from(plan.end), mathematical_end);
            assert!(plan.end <= plan.capacity && plan.capacity <= limit);
            assert!(plan.capacity >= capacity);
            if plan.end <= capacity {
                assert_eq!(plan.capacity, capacity);
            } else {
                assert!(i64::from(plan.capacity) - mathematical_end < i64::from(CAPACITY_STEP));
                assert!(plan.capacity == limit || plan.capacity % CAPACITY_STEP == 0);
            }
        }
        kani::cover!(plan.is_some() && mathematical_end <= i64::from(capacity));
        kani::cover!(plan.is_some() && mathematical_end > i64::from(capacity));
        kani::cover!(
            plan.is_some() && limit % CAPACITY_STEP != 0 && mathematical_end == i64::from(limit)
        );
        kani::cover!(plan.is_none());
        kani::cover!(offset >= 0 && rows > 0 && mathematical_end > i64::from(i32::MAX));
    }

    // The wide u128 payload property timed out after 120s. Its unchanged
    // standalone harness and raw evidence live under build/audit-api-lfm/kv-cache;
    // keep it out of automatic package discovery rather than hiding the timeout.
}
