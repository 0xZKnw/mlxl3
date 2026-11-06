//! LFM3 contracts and shared rotary layout, following LiquidAI's reference.
// Reference: LiquidAI/LFM3-1.7B and LFM3-VL-2B (Apache-2.0 model code).

/// The sliding window counts the query itself. Subtract only after ordering.
pub fn window_contains(query: u32, key: u32, window: u32) -> bool {
    key <= query && query - key < window
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn short_conv_sizes(dim: i32) -> Option<[i32; 2]> {
    if dim <= 0 {
        return None;
    }
    dim.checked_mul(7)?;
    Some([dim.checked_mul(3)?, dim.checked_mul(6)?])
}

#[cfg(any(feature = "mlx", test, kani))]
pub(crate) fn attention_gate_sizes(time: i32, heads: i32, dim: i32) -> Option<[i32; 2]> {
    if time <= 0 || heads <= 0 || dim <= 0 {
        return None;
    }
    let gates = time.checked_mul(heads)?;
    Some([gates, gates.checked_mul(dim)?])
}

/// Raster coordinates to the merge-block order used by the vision checkpoint.
pub fn patch_index(row: u32, column: u32, height: u32, width: u32, merge: u32) -> Option<u32> {
    if merge == 0
        || height == 0
        || width == 0
        || !height.is_multiple_of(merge)
        || !width.is_multiple_of(merge)
        || row >= height
        || column >= width
    {
        return None;
    }
    height.checked_mul(width)?;
    Some(
        ((row / merge) * (width / merge) + column / merge) * merge * merge
            + (row % merge) * merge
            + column % merge,
    )
}

/// Decoded, normalized patches; no device arrays or retained file handles.
pub struct PreparedImage {
    pub height: u32,
    pub width: u32,
    pub patches: Vec<f32>,
}
impl PreparedImage {
    pub fn token_count(&self) -> usize {
        (u64::from(self.height) * u64::from(self.width) / 4) as usize
    }
}

#[cfg(feature = "mlx")]
mod runtime {
    use super::PreparedImage;
    use crate::{
        array::{Array, Dtype},
        checkpoint::Checkpoint,
        lfm2::Lfm2,
        lfm3_vision::Vision,
    };
    use anyhow::{Context, Result, ensure};
    use serde_json::Value;

    pub(crate) struct Rope {
        inverse: Vec<f32>,
        dim: i32,
    }
    impl Rope {
        pub(crate) fn from_config(value: &Value, kind: &str, dim: i32) -> Result<Self> {
            let params = &value[kind];
            ensure!(
                params["rope_type"] == "default",
                "unsupported LFM3 RoPE type"
            );
            let theta = params["rope_theta"]
                .as_f64()
                .context("missing LFM3 RoPE theta")?;
            let factor = params
                .get("partial_rotary_factor")
                .map_or(Some(1.), Value::as_f64)
                .context("invalid LFM3 partial rotary factor")?;
            ensure!(
                theta.is_finite()
                    && theta > 0.
                    && factor.is_finite()
                    && factor > 0.
                    && factor <= 1.,
                "invalid LFM3 RoPE parameters"
            );
            ensure!(dim > 0 && dim % 2 == 0, "invalid LFM3 rotary dimensions");
            let count = (factor * f64::from(dim)) as i32 / 2;
            let inverse = (0..dim / 2)
                .map(|i| {
                    if i < count {
                        theta.powf(-f64::from(2 * i) / f64::from(dim)) as f32
                    } else {
                        0.
                    }
                })
                .collect();
            Ok(Self { inverse, dim })
        }
        pub(crate) fn apply(&self, x: &Array, offset: i32) -> Result<Array> {
            let time = x.shape()[2];
            let angles = (0..time)
                .flat_map(|t| self.inverse.iter().map(move |f| (offset + t) as f32 * f))
                .collect::<Vec<_>>();
            rotate(x, &angles, self.dim)
        }
    }

    /// Half-split pairing over the full head: partial RoPE zeroes the tail
    /// frequencies, rather than pairing the prefix as a smaller rotary head.
    pub(crate) fn rotate(x: &Array, angles: &[f32], dim: i32) -> Result<Array> {
        ensure!(
            x.shape().len() == 4 && x.shape()[3] == dim && dim > 0 && dim % 2 == 0,
            "invalid rotary input"
        );
        let time = x.shape()[2];
        ensure!(
            angles.len() == time as usize * (dim / 2) as usize
                && angles.iter().all(|a| a.is_finite()),
            "invalid rotary angles"
        );
        let cos = Array::from_f32(
            &angles.iter().map(|a| a.cos()).collect::<Vec<_>>(),
            &[1, 1, time, dim / 2],
        )?;
        let sin = Array::from_f32(
            &angles.iter().map(|a| a.sin()).collect::<Vec<_>>(),
            &[1, 1, time, dim / 2],
        )?;
        let a = x.slice(3, 0, dim / 2)?.astype(Dtype::Float32)?;
        let b = x.slice(3, dim / 2, dim)?.astype(Dtype::Float32)?;
        let left = a.mul(&cos)?.add(&b.mul(&sin)?.negative()?)?;
        let right = b.mul(&cos)?.add(&a.mul(&sin)?)?;
        Array::concatenate(&[&left, &right], 3)?.astype(x.dtype())
    }

    pub(crate) fn sliding_mask(time: i32, keys: i32, window: i32) -> Result<Array> {
        ensure!(
            time > 0 && keys >= time && window > 0,
            "invalid LFM3 sliding attention shape"
        );
        let count = (time as usize)
            .checked_mul(keys as usize)
            .context("attention mask overflow")?;
        let mut mask = Vec::with_capacity(count);
        for row in 0..time {
            for key in 0..keys {
                mask.push(
                    if super::window_contains((keys - time + row) as u32, key as u32, window as u32)
                    {
                        0.
                    } else {
                        f32::NEG_INFINITY
                    },
                );
            }
        }
        Array::from_f32(&mask, &[time, keys])
    }

    pub struct Lfm3 {
        pub(crate) language: Lfm2,
        vision: Option<Vision>,
        images: Vec<(usize, Array)>,
        image_token: u32,
    }
    impl Lfm3 {
        pub fn from_checkpoint(checkpoint: &Checkpoint) -> Result<Self> {
            let config: Value =
                serde_json::from_slice(&std::fs::read(checkpoint.path.join("config.json"))?)?;
            let vl = checkpoint.model_type == "lfm3_vl";
            ensure!(
                vl || checkpoint.model_type == "lfm3",
                "expected LFM3 checkpoint"
            );
            crate::array::set_default_cache_limit(512 << 20)?;
            let text = if vl { &config["text_config"] } else { &config };
            let language = Lfm2::from_config(
                checkpoint,
                serde_json::from_value(text.clone())?,
                if vl { "model.language_model" } else { "model" },
            )?;
            let vision = vl.then(|| Vision::load(checkpoint, &config)).transpose()?;
            let image_token = if vl {
                u32::try_from(
                    config["image_token_id"]
                        .as_u64()
                        .context("missing image token")?,
                )?
            } else {
                0
            };
            Ok(Self {
                language,
                vision,
                images: Vec::new(),
                image_token,
            })
        }
        pub fn supports_images(&self) -> bool {
            self.vision.is_some()
        }
        pub fn image_features(&self, image: &PreparedImage) -> Result<Array> {
            self.vision
                .as_ref()
                .context("this LFM3 checkpoint does not support images")?
                .forward(image)
        }
        pub fn prepare_images(&mut self, tokens: &[u32], images: &[PreparedImage]) -> Result<()> {
            ensure!(
                self.language.position() == 0,
                "images require an empty model cache"
            );
            self.images.clear();
            let vision = self
                .vision
                .as_ref()
                .context("this LFM3 checkpoint does not support images")?;
            let positions = tokens
                .iter()
                .enumerate()
                .filter_map(|(i, &t)| (t == self.image_token).then_some(i))
                .collect::<Vec<_>>();
            ensure!(
                positions.len() == images.iter().map(PreparedImage::token_count).sum::<usize>(),
                "image token/feature count mismatch"
            );
            let mut used = 0;
            let mut prepared = Vec::new();
            for image in images {
                let count = image.token_count();
                ensure!(
                    count > 0
                        && positions[used..used + count]
                            .windows(2)
                            .all(|p| p[1] == p[0] + 1),
                    "image tokens must be contiguous"
                );
                let features = vision.forward(image)?;
                ensure!(
                    features.shape()[0] as usize == count,
                    "invalid projected image feature count"
                );
                features.eval()?;
                prepared.push((positions[used], features));
                used += count;
            }
            self.images = prepared;
            Ok(())
        }
        pub fn forward(&mut self, tokens: &[u32]) -> Result<Array> {
            let begin = self.language.position() as usize;
            let end = begin
                .checked_add(tokens.len())
                .context("LFM3 position overflow")?;
            let mut hidden = self.language.embed_tokens(tokens)?;
            for (start, features) in &self.images {
                let left = begin.max(*start);
                let right = end.min(*start + features.shape()[0] as usize);
                if left < right {
                    let prefix = hidden.slice(1, 0, (left - begin) as i32)?;
                    let image = features
                        .slice(0, (left - start) as i32, (right - start) as i32)?
                        .reshape(&[1, (right - left) as i32, features.shape()[1]])?;
                    let suffix = hidden.slice(1, (right - begin) as i32, tokens.len() as i32)?;
                    hidden = Array::concatenate(&[&prefix, &image, &suffix], 1)?;
                }
            }
            match self.language.forward_embedded(tokens, Some(hidden)) {
                Ok(logits) => Ok(logits),
                Err(error) => {
                    self.reset();
                    Err(error)
                }
            }
        }
        pub fn reset(&mut self) {
            self.language.reset();
            self.images.clear();
        }
        pub fn context_limit(&self) -> i32 {
            self.language.context_limit()
        }
        pub fn state_arrays(&self) -> Vec<(String, &Array)> {
            self.language.state_arrays()
        }
        pub fn name(&self) -> &'static str {
            if self.supports_images() {
                "LFM3-VL"
            } else {
                "LFM3"
            }
        }
    }
}
#[cfg(feature = "mlx")]
pub use runtime::Lfm3;
#[cfg(feature = "mlx")]
pub(crate) use runtime::{Rope, rotate, sliding_mask};

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "mlx")]
    #[test]
    #[ignore = "physical Apple GPU; run alone, allocator override absent or 0/1 MiB"]
    fn loading_selects_bounded_allocator_cache_and_preserves_live_arrays() -> anyhow::Result<()> {
        use crate::{
            array::{self, Array, Dtype},
            checkpoint::Checkpoint,
        };
        let limit = match std::env::var("MLXL3_ALLOCATOR_CACHE_MIB") {
            Ok(value) => value.parse::<u64>()?,
            Err(std::env::VarError::NotPresent) => 512,
            Err(error) => return Err(error.into()),
        } << 20;
        array::set_default_cache_limit(4096 << 20)?;
        let live = Array::from_f32(&[1.25, -3.5, 0.0], &[3])?;
        live.eval()?;
        let directory = tempfile::tempdir()?;
        for kind in ["lfm3", "lfm3_vl"] {
            {
                let temporary = Array::zeros_dtype(&[320 * 1024 * 1024], Dtype::Float16)?;
                temporary.eval()?;
            }
            // Configuration errors must propagate after the allocator is selected, without weights.
            std::fs::write(
                directory.path().join("config.json"),
                r#"{"text_config":{}}"#,
            )?;
            let checkpoint = Checkpoint {
                path: directory.path().to_owned(),
                model_type: kind.to_owned(),
                bits: None,
                size_bytes: 0,
                modules: Vec::new(),
                tensors: Default::default(),
            };
            let error = match Lfm3::from_checkpoint(&checkpoint) {
                Ok(_) => anyhow::bail!("incomplete checkpoint accepted"),
                Err(error) => error,
            };
            assert!(error.to_string().contains("missing field"), "{error:#}");
            let next = live.add(&live)?;
            next.eval()?;
            array::synchronize()?;
            assert!(array::memory_stats(false)?.mlx_cache_bytes <= limit);
            assert_eq!(next.to_f32()?, [2.5, -7.0, 0.0]);
            assert_eq!(live.to_f32()?, [1.25, -3.5, 0.0]);
        }
        Ok(())
    }

    #[test]
    fn fused_kernel_sizes_reject_invalid_and_overflowing_buffers() {
        assert_eq!(short_conv_sizes(2048), Some([6144, 12288]));
        assert_eq!(
            short_conv_sizes(i32::MAX / 7),
            Some([920350134, 1840700268])
        );
        for dim in [i32::MIN, -1, 0, i32::MAX / 7 + 1, i32::MAX] {
            assert!(short_conv_sizes(dim).is_none());
        }
        assert_eq!(attention_gate_sizes(257, 16, 128), Some([4112, 526336]));
        for shape in [
            (0, 16, 128),
            (1, -1, 128),
            (1, 16, 0),
            (i32::MAX, 2, 1),
            (1, 2, i32::MAX),
        ] {
            assert!(attention_gate_sizes(shape.0, shape.1, shape.2).is_none());
        }
        for time in 1..32 {
            for heads in 1..16 {
                for dim in [1, 7, 32, 128] {
                    let gates = i64::from(time) * i64::from(heads);
                    assert_eq!(
                        attention_gate_sizes(time, heads, dim),
                        Some([gates as i32, (gates * i64::from(dim)) as i32])
                    );
                }
            }
        }
    }
    #[test]
    fn window_counts_query_and_excludes_future() {
        for q in 0..300 {
            for k in 0..300 {
                for w in [0, 1, 2, 7, 256] {
                    assert_eq!(
                        window_contains(q, k, w),
                        k <= q && u64::from(k) + u64::from(w) > u64::from(q)
                    );
                }
            }
        }
        assert!(window_contains(u32::MAX, u32::MAX, 1));
        assert!(!window_contains(0, u32::MAX, 256));
    }
    #[test]
    fn patch_order_is_bijective_for_rectangles() {
        assert_eq!(
            (0..4)
                .flat_map(|r| (0..6).map(move |c| patch_index(r, c, 4, 6, 2).unwrap()))
                .collect::<Vec<_>>(),
            vec![
                0, 1, 4, 5, 8, 9, 2, 3, 6, 7, 10, 11, 12, 13, 16, 17, 20, 21, 14, 15, 18, 19, 22,
                23
            ]
        );
        for h in [2, 4, 6, 16] {
            for w in [2, 4, 8, 18] {
                let mut ids = (0..h)
                    .flat_map(|r| (0..w).map(move |c| patch_index(r, c, h, w, 2).unwrap()))
                    .collect::<Vec<_>>();
                ids.sort_unstable();
                assert_eq!(ids, (0..h * w).collect::<Vec<_>>());
            }
        }
        for args in [
            (0, 0, 0, 2, 2),
            (0, 0, 2, 2, 0),
            (0, 0, 3, 2, 2),
            (2, 0, 2, 2, 2),
            (0, 2, 2, 2, 2),
            (0, 0, u32::MAX, u32::MAX, 1),
        ] {
            assert!(patch_index(args.0, args.1, args.2, args.3, args.4).is_none());
        }
    }
}

#[cfg(kani)]
#[kani::proof]
fn lfm3_window_is_causal_and_bounded() {
    let q: u32 = kani::any();
    let k: u32 = kani::any();
    let w: u32 = kani::any();
    let visible = window_contains(q, k, w);
    assert_eq!(
        visible,
        k <= q && u64::from(k) + u64::from(w) > u64::from(q)
    );
    kani::cover!(visible && q > 256 && k == q - 255 && w == 256);
    kani::cover!(!visible && k > q);
}
#[cfg(kani)]
#[kani::proof]
fn lfm3_patch_index_is_in_bounds() {
    let h: u8 = kani::any();
    let w: u8 = kani::any();
    let r: u8 = kani::any();
    let c: u8 = kani::any();
    let m: u8 = kani::any();
    if let Some(i) = patch_index(r.into(), c.into(), h.into(), w.into(), m.into()) {
        assert!(i < u32::from(h) * u32::from(w));
        assert!(r < h && c < w && m > 0);
        kani::cover!(h == 4 && w == 6 && m == 2 && r == 3 && c == 5 && i == 23);
    }
}

#[cfg(kani)]
#[kani::proof]
fn lfm3_short_conv_buffers_are_bounded() {
    let dim: i32 = kani::any();
    let sizes = short_conv_sizes(dim);
    assert_eq!(
        sizes.is_some(),
        dim > 0 && i64::from(dim) * 7 <= i64::from(i32::MAX)
    );
    if let Some([input, state]) = sizes {
        assert_eq!(i64::from(input), i64::from(dim) * 3);
        assert_eq!(i64::from(state), i64::from(dim) * 6);
        assert!(state == 2 * input && state > dim);
        kani::cover!(dim == 2048);
    }
    kani::cover!(dim == i32::MAX && sizes.is_none());
}

#[cfg(kani)]
#[kani::proof]
fn lfm3_attention_gate_buffers_are_bounded() {
    let time: i32 = kani::any();
    let heads: i32 = kani::any();
    let dim: i32 = kani::any();
    let gates = i64::from(time) * i64::from(heads);
    let valid = time > 0
        && heads > 0
        && dim > 0
        && gates <= i64::from(i32::MAX)
        && gates * i64::from(dim) <= i64::from(i32::MAX);
    let sizes = attention_gate_sizes(time, heads, dim);
    assert_eq!(sizes.is_some(), valid);
    if let Some([gate_count, output]) = sizes {
        assert!(gate_count > 0 && output >= gate_count);
        assert_eq!(i64::from(output), i64::from(gate_count) * i64::from(dim));
        kani::cover!(time == 257 && heads == 16 && dim == 128);
    }
    kani::cover!(time == i32::MAX && heads == 2 && sizes.is_none());
}

#[cfg(all(kani, feature = "mlx"))]
#[kani::proof]
fn allocator_default_runtime_has_no_rust_panics() {
    let bytes: u32 = kani::any();
    let _ = crate::array::set_default_cache_limit(u64::from(bytes));
}
