//! Inkling 2 Mini vision encoder (`model.visual.*`) and its 2 x 2 merger.
//!
//! Follows `Inkling2VisionModel` of Hugging Face Transformers 5.24 (the reference implementation of the
//! source model): 20-pixel patches, a class token in front of each image, learned 32 x 32 positions
//! resampled bilinearly to the patch grid, pre-norm encoder layers with layer scale and a 2D rotary
//! embedding on interleaved channel pairs (columns first, then rows), then four neighbouring patches merged
//! into one language-model token by a two-layer GELU projector. Images arrive at their native resolution,
//! padded to whole 40-pixel tokens (see `vision_input::decode_inkling_image`).
//!
//! The host-side tables (patch order, position taps, rotary angles) are plain functions so they can be
//! checked without a GPU; the encoder itself runs in BF16 like the reference.
use anyhow::{Context, Result, ensure};

/// Pixels on a side of one native patch.
pub const PATCH: u32 = 20;
/// Patches on a side of the square merged into one language-model token.
pub const MERGE: u32 = 2;
/// Pixels on a side of one language-model image token.
pub const TOKEN_PIXELS: u32 = PATCH * MERGE;
/// Values of one native patch: 20 x 20 RGB, rows, then columns, then channels.
pub const PATCH_VALUES: usize = (PATCH * PATCH * 3) as usize;
/// Placeholder token of each image slot (`<|unused_200054|>` in `processor_config.json`).
pub const IMAGE_TOKEN_ID: u32 = 200_054;
/// Text the chat template writes once per image part, before the placeholders are expanded.
pub const IMAGE_PLACEHOLDER: &str = "<|unused_200054|>";

/// Patch-grid `(row, column)` of every patch of a `height` x `width` patch grid, in the encoder's
/// sequence order: merged 2 x 2 blocks in raster order, each block's patches in raster order.
pub fn patch_positions(height: u32, width: u32) -> Result<Vec<(u32, u32)>> {
    ensure!(
        height > 0 && width > 0 && height.is_multiple_of(MERGE) && width.is_multiple_of(MERGE),
        "invalid Inkling patch grid {height} x {width}"
    );
    let count = usize::try_from(u64::from(height) * u64::from(width))?;
    let mut positions = vec![(0, 0); count];
    for row in 0..height {
        for column in 0..width {
            let index = crate::lfm3::patch_index(row, column, height, width, MERGE)
                .context("invalid Inkling patch position")?;
            positions[index as usize] = (row, column);
        }
    }
    Ok(positions)
}

/// The two taps of one axis and their weights, as `F.interpolate(mode="bilinear", align_corners=False)`
/// with border clamping: `src = (index + 0.5) * side / size - 0.5`, taps `floor(src)` and `floor(src) + 1`
/// clamped to the table, weights `max(1 - |src - floor - offset|, 0)` (FP32, the reference's order).
fn axis_taps(index: u32, size: u32, side: u32) -> ([i32; 2], [f32; 2]) {
    let src = (index as f32 + 0.5) * side as f32 / size as f32 - 0.5;
    let floor = src.floor();
    let mut taps = [0; 2];
    let mut weights = [0.; 2];
    for offset in 0..2 {
        taps[offset] = (floor as i64 + offset as i64).clamp(0, i64::from(side) - 1) as i32;
        let distance = (src - floor - offset as f32).abs();
        weights[offset] = (1. - distance).max(0.);
    }
    (taps, weights)
}

/// Rows of the learned `side` x `side` position table and their weights for every patch, four taps per
/// patch (`[(h0, w0), (h0, w1), (h1, w0), (h1, w1)]`), in `patch_positions` order.
pub fn position_taps(height: u32, width: u32, side: u32) -> Result<(Vec<i32>, Vec<f32>)> {
    ensure!(side > 0, "invalid Inkling position grid");
    let positions = patch_positions(height, width)?;
    let mut indices = Vec::with_capacity(positions.len() * 4);
    let mut weights = Vec::with_capacity(positions.len() * 4);
    for (row, column) in positions {
        let (rows, row_weights) = axis_taps(row, height, side);
        let (columns, column_weights) = axis_taps(column, width, side);
        for h in 0..2 {
            for w in 0..2 {
                indices.push(rows[h] * side as i32 + columns[w]);
                weights.push(row_weights[h] * column_weights[w]);
            }
        }
    }
    Ok((indices, weights))
}

/// Rotary angles `[1 + patches, head_dim / 2]`, one per interleaved channel pair: the class token at
/// `(0, 0)` then each patch at `(row + 1, column + 1)`; pairs `[0, head_dim / 4)` turn with the column and
/// the others with the row, at frequencies `1 / theta^(2j / (head_dim / 2))` (FP32, as the reference).
pub fn rotary_angles(height: u32, width: u32, head_dim: u32, theta: f32) -> Result<Vec<f32>> {
    ensure!(
        head_dim > 0 && head_dim.is_multiple_of(4) && theta.is_finite() && theta > 1.,
        "invalid Inkling vision rotary settings"
    );
    let quarter = head_dim / 4;
    let spatial = (head_dim / 2) as f32;
    let inverse: Vec<f32> = (0..quarter)
        .map(|j| 1. / theta.powf((2 * j) as f32 / spatial))
        .collect();
    let mut angles = Vec::new();
    let mut push = |row: u32, column: u32| {
        for position in [column, row] {
            for frequency in &inverse {
                angles.push(position as f32 * frequency);
            }
        }
    };
    push(0, 0);
    for (row, column) in patch_positions(height, width)? {
        push(row + 1, column + 1);
    }
    Ok(angles)
}

/// `(first position, length)` of every run of consecutive image placeholders in `tokens`.
pub fn placeholder_runs(tokens: &[u32]) -> Vec<(usize, usize)> {
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (position, &token) in tokens.iter().enumerate() {
        if token != IMAGE_TOKEN_ID {
            continue;
        }
        match runs.last_mut() {
            Some((first, length)) if *first + *length == position => *length += 1,
            _ => runs.push((position, 1)),
        }
    }
    runs
}

#[cfg(feature = "mlx")]
pub(crate) use runtime::Vision;

#[cfg(feature = "mlx")]
mod runtime {
    use super::{MERGE, PATCH, PATCH_VALUES, position_taps, rotary_angles};
    use crate::{
        array::{Array, Dtype},
        checkpoint::Checkpoint,
        lfm3::PreparedImage,
        linear::checkpoint_array,
    };
    use anyhow::{Context, Result, ensure};
    use serde::Deserialize;

    /// Patches one image may have (the aggregate image budget is far below).
    const MAX_PATCHES: u64 = 1 << 20;

    fn default_channels() -> i32 {
        3
    }
    fn default_act() -> String {
        "gelu".into()
    }
    fn default_ratio() -> f64 {
        4.
    }
    fn default_eps() -> f32 {
        1e-5
    }

    #[derive(Deserialize)]
    struct Config {
        hidden_size: i32,
        num_hidden_layers: usize,
        num_attention_heads: i32,
        #[serde(default = "default_ratio")]
        mlp_ratio: f64,
        #[serde(default)]
        intermediate_size: Option<i32>,
        #[serde(alias = "native_patch_size")]
        patch_size: i32,
        #[serde(alias = "adapter_merge_size")]
        spatial_merge_size: i32,
        #[serde(default = "default_channels")]
        num_channels: i32,
        posemb_grid_size: i32,
        projection_hidden_dim: i32,
        #[serde(default)]
        projector_bias: bool,
        text_hidden_size: i32,
        #[serde(default = "default_eps")]
        layer_norm_eps: f32,
        #[serde(default = "default_act")]
        hidden_act: String,
        #[serde(default)]
        rope_parameters: Option<serde_json::Value>,
    }

    /// `x @ weight.T + bias`, weights `[out, in]` kept as stored and converted to the type of `x` on use.
    struct Dense {
        weight: Array,
        bias: Option<Array>,
    }
    impl Dense {
        fn load(
            cp: &Checkpoint,
            prefix: &str,
            input: i32,
            output: i32,
            bias: bool,
        ) -> Result<Self> {
            Ok(Self {
                weight: stored(cp, &format!("{prefix}.weight"), &[output, input])?,
                bias: bias
                    .then(|| stored(cp, &format!("{prefix}.bias"), &[output]))
                    .transpose()?,
            })
        }
        fn forward(&self, x: &Array) -> Result<Array> {
            let y = x.matmul(&self.weight.astype(x.dtype())?.transpose(&[1, 0])?)?;
            match &self.bias {
                Some(bias) => y.add(&bias.astype(x.dtype())?),
                None => Ok(y),
            }
        }
    }

    /// LayerNorm computed in FP32 with its weight and bias, rounded once to the compute type (the
    /// reference's kernels accumulate and apply the affine part in FP32).
    struct Norm {
        weight: Array,
        bias: Array,
        eps: f32,
        dtype: Dtype,
    }
    impl Norm {
        fn load(cp: &Checkpoint, prefix: &str, dim: i32, eps: f32, dtype: Dtype) -> Result<Self> {
            Ok(Self {
                weight: stored(cp, &format!("{prefix}.weight"), &[dim])?.astype(Dtype::Float32)?,
                bias: stored(cp, &format!("{prefix}.bias"), &[dim])?.astype(Dtype::Float32)?,
                eps,
                dtype,
            })
        }
        fn forward(&self, x: &Array) -> Result<Array> {
            x.astype(Dtype::Float32)?
                .layer_norm_without_weight(self.eps)?
                .mul(&self.weight)?
                .add(&self.bias)?
                .astype(self.dtype)
        }
    }

    struct Layer {
        before: Norm,
        after: Norm,
        q: Dense,
        k: Dense,
        v: Dense,
        out: Dense,
        fc1: Dense,
        fc2: Dense,
        scale_attention: Array,
        scale_mlp: Array,
    }

    pub(crate) struct Vision {
        hidden: i32,
        heads: i32,
        head_dim: i32,
        grid: i32,
        theta: f32,
        text_hidden: i32,
        patch_embedding: Dense,
        position_embedding: Array,
        cls: Array,
        pre: Norm,
        layers: Vec<Layer>,
        proj_in: Dense,
        proj_out: Dense,
        /// Compute type of the activations (FP32 by default, BF16 for the comparisons of the tests).
        dtype: Dtype,
    }

    /// A floating-point checkpoint tensor of `shape`, resident in its stored type (BF16: 0.64 GB).
    fn stored(cp: &Checkpoint, name: &str, shape: &[i32]) -> Result<Array> {
        let weight = checkpoint_array(cp, name)?;
        ensure!(
            matches!(
                weight.dtype(),
                Dtype::BFloat16 | Dtype::Float16 | Dtype::Float32
            ),
            "{name}: expected floating-point weights"
        );
        ensure!(
            weight.shape() == shape,
            "{name}: shape {:?}, expected {shape:?}",
            weight.shape()
        );
        weight.eval()?;
        Ok(weight)
    }

    impl Vision {
        /// The checkpoint ships the vision encoder (`model.visual.*`).
        pub(crate) fn present(cp: &Checkpoint) -> bool {
            cp.tensors
                .contains_key("model.visual.backbone.embeddings.patch_embedding.weight")
        }

        /// The encoder computing in FP32 from its BF16 weights. On Apple M5 its features stay 6 times
        /// closer to the exact FP32 result than a BF16 run (0.40 % against 2.4 % relative L2; the
        /// reference itself in BF16: 1.9 %; 2026-11-23).
        pub(crate) fn load(cp: &Checkpoint, root: &serde_json::Value) -> Result<Self> {
            Self::load_as(cp, root, Dtype::Float32)
        }

        /// The encoder computing its activations in `dtype` (BF16 or FP32; LayerNorm, rotary embedding
        /// and position resampling stay in FP32 either way). The weights stay in their stored type.
        pub(crate) fn load_as(
            cp: &Checkpoint,
            root: &serde_json::Value,
            dtype: Dtype,
        ) -> Result<Self> {
            ensure!(
                matches!(dtype, Dtype::BFloat16 | Dtype::Float32),
                "Inkling vision computes in BF16 or FP32"
            );
            let c: Config = serde_json::from_value(root["vision_config"].clone())
                .context("invalid Inkling vision_config")?;
            let dim = c.hidden_size;
            let ff = match c.intermediate_size {
                Some(ff) => ff,
                None => (f64::from(dim) * c.mlp_ratio) as i32,
            };
            ensure!(
                dim > 0
                    && c.num_attention_heads > 0
                    && dim % c.num_attention_heads == 0
                    && (dim / c.num_attention_heads) % 4 == 0
                    && c.num_hidden_layers > 0
                    && c.num_hidden_layers <= 256
                    && ff > 0
                    && c.posemb_grid_size > 0
                    && c.posemb_grid_size <= 1024
                    && c.projection_hidden_dim > 0
                    && c.text_hidden_size > 0
                    && c.layer_norm_eps.is_finite()
                    && c.layer_norm_eps > 0.,
                "invalid Inkling vision dimensions"
            );
            ensure!(
                c.patch_size == PATCH as i32
                    && c.spatial_merge_size == MERGE as i32
                    && c.num_channels == 3
                    && c.hidden_act == "gelu",
                "unsupported Inkling vision layout"
            );
            let text = root["text_config"]["hidden_size"].as_i64();
            ensure!(
                text == Some(i64::from(c.text_hidden_size)),
                "Inkling vision output does not match the language model width"
            );
            let theta = match &c.rope_parameters {
                None => 10_000.,
                Some(p) => {
                    ensure!(
                        p.get("rope_type").is_none_or(|t| t == "axial"),
                        "unsupported Inkling vision rotary type"
                    );
                    p.get("rope_theta")
                        .map_or(Some(10_000.), |t| t.as_f64())
                        .context("invalid Inkling vision rotary base")? as f32
                }
            };
            ensure!(
                theta.is_finite() && theta > 1.,
                "invalid Inkling vision rotary base"
            );
            let grid = c.posemb_grid_size;
            let p = "model.visual.backbone";
            let mut layers = Vec::with_capacity(c.num_hidden_layers);
            for i in 0..c.num_hidden_layers {
                let l = format!("{p}.layers.{i}");
                let a = format!("{l}.self_attn");
                layers.push(Layer {
                    before: Norm::load(cp, &format!("{l}.ln_1"), dim, c.layer_norm_eps, dtype)?,
                    after: Norm::load(cp, &format!("{l}.ln_2"), dim, c.layer_norm_eps, dtype)?,
                    q: Dense::load(cp, &format!("{a}.q_proj"), dim, dim, true)?,
                    k: Dense::load(cp, &format!("{a}.k_proj"), dim, dim, true)?,
                    v: Dense::load(cp, &format!("{a}.v_proj"), dim, dim, true)?,
                    out: Dense::load(cp, &format!("{a}.out_proj"), dim, dim, true)?,
                    fc1: Dense::load(cp, &format!("{l}.mlp.c_fc"), dim, ff, true)?,
                    fc2: Dense::load(cp, &format!("{l}.mlp.c_proj"), ff, dim, true)?,
                    scale_attention: stored(cp, &format!("{l}.ls_1.gamma"), &[dim])?
                        .astype(dtype)?,
                    scale_mlp: stored(cp, &format!("{l}.ls_2.gamma"), &[dim])?.astype(dtype)?,
                });
            }
            let e = format!("{p}.embeddings");
            // the reference adds the two vectors in its compute type
            let cls = stored(cp, &format!("{e}.class_embedding"), &[dim])?
                .astype(dtype)?
                .add(&stored(cp, &format!("{e}.cls_position_embedding"), &[dim])?.astype(dtype)?)?
                .reshape(&[1, dim])?;
            cls.eval()?;
            Ok(Self {
                hidden: dim,
                heads: c.num_attention_heads,
                head_dim: dim / c.num_attention_heads,
                grid,
                theta,
                text_hidden: c.text_hidden_size,
                patch_embedding: Dense::load(
                    cp,
                    &format!("{e}.patch_embedding"),
                    PATCH_VALUES as i32,
                    dim,
                    false,
                )?,
                position_embedding: stored(
                    cp,
                    &format!("{e}.position_embedding"),
                    &[grid * grid, dim],
                )?,
                cls,
                pre: Norm::load(cp, &format!("{p}.ln_pre"), dim, c.layer_norm_eps, dtype)?,
                layers,
                proj_in: Dense::load(
                    cp,
                    "model.visual.proj_in",
                    dim * (MERGE * MERGE) as i32,
                    c.projection_hidden_dim,
                    c.projector_bias,
                )?,
                proj_out: Dense::load(
                    cp,
                    "model.visual.proj_out",
                    c.projection_hidden_dim,
                    c.text_hidden_size,
                    c.projector_bias,
                )?,
                dtype,
            })
        }

        pub(crate) fn text_hidden(&self) -> i32 {
            self.text_hidden
        }

        /// Image features `[tokens, text_hidden]` in the compute type, one row per placeholder token, in the order of the
        /// image's tokens (raster order of its 40-pixel tiles).
        pub(crate) fn forward(&self, image: &PreparedImage) -> Result<Array> {
            let (h, w) = (image.height, image.width);
            let patches = u64::from(h) * u64::from(w);
            ensure!(
                h > 0
                    && w > 0
                    && h.is_multiple_of(MERGE)
                    && w.is_multiple_of(MERGE)
                    && patches <= MAX_PATCHES
                    && image.patches.len() as u64 == patches * PATCH_VALUES as u64
                    && image.patches.iter().all(|v| v.is_finite()),
                "invalid Inkling image patches"
            );
            let n = i32::try_from(patches)?;
            let dim = self.hidden;
            let pixels =
                Array::from_f32(&image.patches, &[n, PATCH_VALUES as i32])?.astype(self.dtype)?;
            let embedded = self.patch_embedding.forward(&pixels)?;
            // learned positions resampled to this grid: four rows weighted and summed in FP32
            let (taps, weights) = position_taps(h, w, self.grid as u32)?;
            let positions = self
                .position_embedding
                .take(&Array::from_i32(&taps, &[n * 4])?, 0)?
                .astype(Dtype::Float32)?
                .reshape(&[n, 4, dim])?
                .mul(&Array::from_f32(&weights, &[n, 4, 1])?)?
                .sum(1, false)?
                .astype(self.dtype)?;
            let sequence = Array::concatenate(&[&self.cls, &embedded.add(&positions)?], 0)?;
            let mut x = self.pre.forward(&sequence)?;
            let time = n + 1;
            let half = self.head_dim / 2;
            let angles = rotary_angles(h, w, self.head_dim as u32, self.theta)?;
            let cos = Array::from_f32(
                &angles.iter().map(|a| a.cos()).collect::<Vec<_>>(),
                &[time, 1, half, 1],
            )?;
            let sin = Array::from_f32(
                &angles.iter().map(|a| a.sin()).collect::<Vec<_>>(),
                &[time, 1, half, 1],
            )?;
            let rotate = |projected: &Array| -> Result<Array> {
                // interleaved pairs (x1, x2) -> (x1 cos - x2 sin, x2 cos + x1 sin), in FP32
                let pairs = projected
                    .reshape(&[time, self.heads, half, 2])?
                    .astype(Dtype::Float32)?;
                let a = pairs.slice(3, 0, 1)?;
                let b = pairs.slice(3, 1, 2)?;
                let left = a.mul(&cos)?.sub(&b.mul(&sin)?)?;
                let right = b.mul(&cos)?.add(&a.mul(&sin)?)?;
                Array::concatenate(&[&left, &right], 3)?
                    .reshape(&[time, self.heads, self.head_dim])?
                    .astype(self.dtype)?
                    .transpose(&[1, 0, 2])?
                    .reshape(&[1, self.heads, time, self.head_dim])
            };
            let scale = (self.head_dim as f32).powf(-0.5);
            for layer in &self.layers {
                let normed = layer.before.forward(&x)?;
                let q = rotate(&layer.q.forward(&normed)?)?;
                let k = rotate(&layer.k.forward(&normed)?)?;
                let v = layer
                    .v
                    .forward(&normed)?
                    .reshape(&[time, self.heads, self.head_dim])?
                    .transpose(&[1, 0, 2])?
                    .reshape(&[1, self.heads, time, self.head_dim])?;
                let attended = Array::sdpa(&q, &k, &v, scale, false)?
                    .reshape(&[self.heads, time, self.head_dim])?
                    .transpose(&[1, 0, 2])?
                    .reshape(&[time, dim])?;
                x = x.add(&layer.scale_attention.mul(&layer.out.forward(&attended)?)?)?;
                let hidden = layer.fc1.forward(&layer.after.forward(&x)?)?.gelu()?;
                x = x.add(&layer.scale_mlp.mul(&layer.fc2.forward(&hidden)?)?)?;
                x.eval()?;
            }
            let merged = x
                .slice(0, 1, time)?
                .reshape(&[n / 4, dim * (MERGE * MERGE) as i32])?;
            let features = self
                .proj_out
                .forward(&self.proj_in.forward(&merged)?.gelu()?)?;
            features.eval()?;
            ensure!(
                features.shape() == [n / 4, self.text_hidden],
                "unexpected Inkling image feature shape"
            );
            Ok(features)
        }
    }
}

/// Checks against `benchmarks/inkling2_vision_oracle.py` (Hugging Face Transformers on the CPU, the
/// checkpoint's own `model.visual.*` tensors). Run alone on an Apple GPU:
/// `MLXL3_INKLING_MODEL=<dir> MLXL3_INKLING_VISION_ORACLE=<json> cargo test --release --no-default-features
/// --features mlx,chat --lib inkling2_vision::gpu_tests -- --ignored --test-threads=1 --nocapture`.
#[cfg(all(test, feature = "mlx", feature = "chat"))]
mod gpu_tests {
    use crate::{
        array::Dtype,
        checkpoint,
        inkling2::Inkling2,
        tokenizer::ChatTokenizer,
        vision_input::{self, ImageFormat},
    };
    use anyhow::{Context, Result, ensure};
    use serde_json::{Value, json};
    use std::path::{Path, PathBuf};

    fn relative_l2(actual: &[f32], expected: &[f32]) -> f64 {
        let error: f64 = actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (f64::from(*a) - f64::from(*b)).powi(2))
            .sum();
        let norm: f64 = expected.iter().map(|b| f64::from(*b).powi(2)).sum();
        (error / norm).sqrt()
    }

    fn floats(value: &Value) -> Result<Vec<f32>> {
        let values: Vec<f32> = serde_json::from_value(value.clone())?;
        ensure!(
            !values.is_empty() && values.iter().all(|v| v.is_finite()),
            "invalid oracle values"
        );
        Ok(values)
    }

    /// Logits of the last row after a prefill in `chunk`-token calls.
    fn prefill(model: &mut Inkling2, tokens: &[u32], chunk: usize) -> Result<Vec<f32>> {
        let mut last = None;
        for part in tokens.chunks(chunk) {
            last = Some(model.forward_tokens(part)?);
        }
        last.context("empty prompt")?.to_f32()
    }

    fn argmax(row: &[f32]) -> usize {
        row.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |(i, _)| i)
    }

    /// The oracle's picture and prompt prepared and tokenized by the engine, checked against the reference
    /// processor: pixel values bit for bit, prompt tokens exactly.
    struct Case {
        dir: PathBuf,
        oracle: Value,
        tokenizer: ChatTokenizer,
        prompt: String,
        images: Vec<crate::lfm3::PreparedImage>,
        tokens: Vec<u32>,
    }

    fn case() -> Result<Case> {
        let dir = PathBuf::from(std::env::var("MLXL3_INKLING_MODEL")?);
        let oracle: Value = serde_json::from_slice(&std::fs::read(std::env::var(
            "MLXL3_INKLING_VISION_ORACLE",
        )?)?)?;
        let image_path = oracle["image"].as_str().context("oracle image")?;
        let url = vision_input::file_data_url(Path::new(image_path))?;
        let prompt = oracle["prompt"]
            .as_str()
            .context("oracle prompt")?
            .to_owned();
        let message = json!({"role":"user","content":[
            {"type":"image_url","image_url":{"url":url}},{"type":"text","text":prompt}]});
        let (prepared, images) =
            vision_input::prepare_messages_as(&[message], Some(ImageFormat::Inkling2))?;
        ensure!(images.len() == 1, "one image expected");
        let image = &images[0];

        // encoder input: same values as Inkling2ImageProcessor (same FP32 formula, exact)
        let pixels = floats(&oracle["pixel_values"])?;
        let grid: [u32; 2] = serde_json::from_value(oracle["grid"].clone())?;
        ensure!(
            (image.height / 2, image.width / 2) == (grid[0], grid[1])
                && image.patches.len() == pixels.len(),
            "image grid {}x{} patches, oracle {grid:?}",
            image.height,
            image.width
        );
        let worst = image
            .patches
            .iter()
            .zip(&pixels)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        eprintln!("pixel values: {} values, max |diff| {worst}", pixels.len());
        assert_eq!(
            worst, 0.,
            "pixel values differ from the reference processor"
        );

        // prompt tokens: the reference processor's chat template and placeholder expansion
        let tokenizer = ChatTokenizer::load(&dir)?;
        let rendered = tokenizer.render_values(&json!(prepared), None)?;
        let rendered = vision_input::expand_image_placeholders(
            Some(ImageFormat::Inkling2),
            rendered,
            &images,
        )?;
        let tokens = tokenizer.encode(&rendered)?;
        let expected: Vec<u32> = serde_json::from_value(oracle["input_ids"].clone())?;
        ensure!(!expected.is_empty(), "empty oracle tokens");
        assert_eq!(
            tokens, expected,
            "prompt tokens differ from the reference processor"
        );
        let runs = super::placeholder_runs(&tokens);
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].1, image.token_count());
        Ok(Case {
            dir,
            oracle,
            tokenizer,
            prompt,
            images,
            tokens,
        })
    }

    /// Relative L2 of `features` against oracle `key`, with the worst token.
    fn compare(features: &[f32], oracle: &Value, key: &str, rows: usize) -> Result<(f64, f64)> {
        let reference = floats(&oracle[key])?;
        ensure!(reference.len() == features.len() && rows > 0, "{key} size");
        let width = features.len() / rows;
        let worst = (0..rows)
            .map(|r| {
                relative_l2(
                    &features[r * width..(r + 1) * width],
                    &reference[r * width..(r + 1) * width],
                )
            })
            .fold(0f64, f64::max);
        Ok((relative_l2(features, &reference), worst))
    }

    /// Relative L2 error of one GPU FP32 matmul against the same product in FP64 on the host. MLX 0.32.2
    /// runs FP32 matmuls on Apple M5 (`applegpu_g17g`) with reduced-precision kernels (7.7e-4 measured
    /// on 2026-11-23, against 5.7e-7 with `MLX_METAL_GPU_ARCH=applegpu_g16s` or on the CPU).
    fn fp32_matmul_error() -> Result<f64> {
        use crate::array::Array;
        let (m, k, n) = (37usize, 512usize, 96usize);
        let value =
            |i: usize, salt: usize| (((i * 7919 + salt * 104_729) % 2001) as f32 - 1000.) / 997.;
        let a: Vec<f32> = (0..m * k).map(|i| value(i, 1)).collect();
        let b: Vec<f32> = (0..k * n).map(|i| value(i, 2)).collect();
        let out = Array::from_f32(&a, &[m as i32, k as i32])?
            .matmul(&Array::from_f32(&b, &[k as i32, n as i32])?)?
            .to_f32()?;
        let mut reference = vec![0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                reference[i * n + j] = (0..k)
                    .map(|l| f64::from(a[i * k + l]) * f64::from(b[l * n + j]))
                    .sum();
            }
        }
        let error: f64 = out
            .iter()
            .zip(&reference)
            .map(|(o, r)| (f64::from(*o) - r).powi(2))
            .sum();
        Ok((error / reference.iter().map(|r| r * r).sum::<f64>()).sqrt())
    }

    /// Encoder only (0.64 GB): FP32 against the FP32 reference (same arithmetic: limit 1e-4, which needs
    /// exact FP32 matmuls: on M5 run with `MLX_METAL_GPU_ARCH=applegpu_g16s`), and BF16, the compute type
    /// used by the engine, against both references.
    #[test]
    #[ignore = "requires an Apple GPU, MLXL3_INKLING_MODEL (with model-vision.safetensors) and MLXL3_INKLING_VISION_ORACLE"]
    fn vision_encoder_matches_the_transformers_reference() -> Result<()> {
        let matmul = fp32_matmul_error()?;
        let exact_fp32 = matmul < 1e-5;
        eprintln!("GPU FP32 matmul relative L2 {matmul:.2e}");
        let c = case()?;
        let cp = checkpoint::inspect(&c.dir)?;
        let root: Value = serde_json::from_reader(std::fs::File::open(c.dir.join("config.json"))?)?;
        let image = &c.images[0];
        let rows = image.token_count();
        let mut results = Vec::new();
        // a 1920 x 1080 picture: 1,296 tokens, 5,184 patches (timing only)
        let large = crate::lfm3::PreparedImage {
            height: 54,
            width: 96,
            patches: vec![0.25; 54 * 96 * super::PATCH_VALUES],
        };
        for dtype in [Dtype::Float32, Dtype::BFloat16] {
            let vision = super::Vision::load_as(&cp, &root, dtype)?;
            let features = vision.forward(image)?.astype(Dtype::Float32)?.to_f32()?;
            ensure!(
                features.len() == rows * 2304 && features.iter().all(|v| v.is_finite()),
                "invalid {dtype:?} features"
            );
            for key in ["features_f32", "features_bf16"] {
                let (error, worst) = compare(&features, &c.oracle, key, rows)?;
                eprintln!(
                    "engine {dtype:?} vs {key}: relative L2 {error:.6}, worst token {worst:.6}"
                );
                results.push((dtype, key, error));
            }
            vision.forward(&large)?;
            let mut times = Vec::new();
            for _ in 0..3 {
                let started = std::time::Instant::now();
                let out = vision.forward(&large)?;
                ensure!(out.shape() == [1296, 2304], "large picture features");
                times.push(started.elapsed().as_secs_f64());
            }
            eprintln!("engine {dtype:?}: 1920 x 1080 picture encoded in {times:.3?} s (warm)");
        }
        let error = |dtype, key| {
            results
                .iter()
                .find(|r| r.0 == dtype && r.1 == key)
                .map(|r| r.2)
                .unwrap()
        };
        // same arithmetic in FP32: only summation order and transcendental functions differ
        let exact = error(Dtype::Float32, "features_f32");
        if exact_fp32 {
            assert!(
                exact < 1e-4,
                "FP32 encoder differs from the FP32 reference: {exact}"
            );
        } else {
            eprintln!(
                "NOT CHECKED: FP32 exactness (this GPU's FP32 matmuls are reduced precision; rerun with \
                 MLX_METAL_GPU_ARCH=applegpu_g16s on Apple M5)"
            );
        }
        // the engine's default (FP32 activations) must not drift from the exact result more than the
        // reference's own BF16 run (criterion set after the first measurement, see the report)
        let reference_noise = c.oracle["bf16_vs_f32_relative_l2"]
            .as_f64()
            .context("oracle BF16 noise")?;
        let default = super::Vision::load(&cp, &root)?
            .forward(image)?
            .astype(Dtype::Float32)?
            .to_f32()?;
        let (engine_noise, _) = compare(&default, &c.oracle, "features_f32", rows)?;
        eprintln!(
            "default vs FP32 reference: engine {engine_noise:.5}; reference BF16 vs FP32 {reference_noise:.5}; \
             engine BF16 activations vs FP32 {:.5} (not used)",
            error(Dtype::BFloat16, "features_f32")
        );
        assert!(
            engine_noise <= 1.25 * reference_noise,
            "the encoder drifts from FP32 more than the reference's BF16 run: {engine_noise} > 1.25 x {reference_noise}"
        );
        Ok(())
    }

    /// Language model (13 GB, load nothing else): features placed at the placeholders, chunk boundaries
    /// inside the image, text path unchanged, malformed prompts refused.
    #[test]
    #[ignore = "requires an Apple GPU with no other model loaded, MLXL3_INKLING_MODEL and MLXL3_INKLING_VISION_ORACLE"]
    fn image_features_reach_the_inkling_language_model() -> Result<()> {
        let Case {
            dir,
            tokenizer,
            prompt,
            images,
            tokens,
            ..
        } = case()?;
        let mut model = Inkling2::load(&dir)?;
        ensure!(
            model.supports_images(),
            "the checkpoint ships no vision encoder"
        );
        let text = tokenizer
            .encode(&tokenizer.render_values(&json!([{"role":"user","content":prompt}]), None)?)?;
        let text_before = prefill(&mut model, &text, 128)?;
        model.reset();
        model.prepare_images(&tokens, &images)?;
        let with_image = prefill(&mut model, &tokens, 128)?;
        // chunks that cut the image run read the same rows
        model.reset();
        model.prepare_images(&tokens, &images)?;
        let chunked = prefill(&mut model, &tokens, 7)?;
        model.reset();
        model.prepare_images(&tokens, &[])?;
        let placeholders = prefill(&mut model, &tokens, 128)?;
        model.reset();
        let text_after = prefill(&mut model, &text, 128)?;
        ensure!(
            with_image.iter().chain(&chunked).all(|v| v.is_finite()),
            "non-finite logits"
        );
        let moved = relative_l2(&with_image, &placeholders);
        let chunk_error = relative_l2(&chunked, &with_image);
        eprintln!(
            "logits: image vs placeholders relative L2 {moved:.4}, 7-token chunks vs 128 {chunk_error:.5}, \
             argmax {} / {} / {}",
            argmax(&with_image),
            argmax(&chunked),
            argmax(&placeholders)
        );
        assert!(
            moved > 0.05,
            "the image features did not reach the language model"
        );
        assert_eq!(argmax(&chunked), argmax(&with_image));
        assert!(
            chunk_error < 0.05,
            "chunked image prefill differs: {chunk_error}"
        );
        assert!(
            text_before
                .iter()
                .map(|v| v.to_bits())
                .eq(text_after.iter().map(|v| v.to_bits())),
            "text-only logits changed after an image prompt"
        );
        // a prompt whose placeholders do not match its images, or a state already past position 0,
        // is refused before any encoder work
        model.reset();
        assert!(model.prepare_images(&text, &images).is_err());
        model.forward_tokens(&text[..1])?;
        assert!(model.prepare_images(&tokens, &images).is_err());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Independent oracle: the reference's `get_vision_position_ids` reshape (block_rows, merge,
    /// block_cols, merge) -> transpose(1, 2) -> flatten, written out directly.
    fn reference_order(h: u32, w: u32) -> Vec<(u32, u32)> {
        let mut out = Vec::new();
        for block_row in 0..h / MERGE {
            for block_col in 0..w / MERGE {
                for in_row in 0..MERGE {
                    for in_col in 0..MERGE {
                        out.push((block_row * MERGE + in_row, block_col * MERGE + in_col));
                    }
                }
            }
        }
        out
    }

    #[test]
    fn patch_order_matches_the_reference_block_order() {
        for (h, w) in [(2, 2), (2, 4), (4, 2), (6, 10), (64, 2)] {
            assert_eq!(patch_positions(h, w).unwrap(), reference_order(h, w));
        }
        for (h, w) in [(0, 2), (2, 0), (3, 2), (2, 5)] {
            assert!(patch_positions(h, w).is_err());
        }
    }

    /// F.interpolate(bilinear, align_corners=False) of a table holding its own (row, col) coordinates
    /// returns the source coordinates clamped to the table: a closed-form check of taps and weights.
    #[test]
    fn position_taps_reproduce_bilinear_resampling() {
        let side = 32u32;
        for (h, w) in [(2, 2), (8, 6), (32, 32), (64, 40), (100, 2)] {
            let (taps, weights) = position_taps(h, w, side).unwrap();
            let positions = patch_positions(h, w).unwrap();
            assert_eq!(taps.len(), positions.len() * 4);
            for (p, &(row, col)) in positions.iter().enumerate() {
                let sum: f32 = weights[p * 4..p * 4 + 4].iter().sum();
                assert!((sum - 1.).abs() < 1e-5, "weights sum {sum}");
                let (mut r, mut c) = (0f64, 0f64);
                for k in 0..4 {
                    let index = taps[p * 4 + k];
                    assert!(index >= 0 && index < (side * side) as i32);
                    r += f64::from(weights[p * 4 + k]) * f64::from(index / side as i32);
                    c += f64::from(weights[p * 4 + k]) * f64::from(index % side as i32);
                }
                let expect = |i: u32, size: u32| {
                    ((f64::from(i) + 0.5) * f64::from(side) / f64::from(size) - 0.5)
                        .clamp(0., f64::from(side - 1))
                };
                assert!((r - expect(row, h)).abs() < 1e-4, "row {row}/{h}: {r}");
                assert!((c - expect(col, w)).abs() < 1e-4, "col {col}/{w}: {c}");
            }
        }
        // the grid's own size reproduces the table exactly: one tap of weight 1
        let (taps, weights) = position_taps(32, 32, 32).unwrap();
        for p in 0..taps.len() / 4 {
            let one = (0..4).filter(|&k| weights[p * 4 + k] == 1.).count();
            assert_eq!(one, 1);
        }
    }

    #[test]
    fn placeholder_runs_split_on_any_other_token() {
        let p = IMAGE_TOKEN_ID;
        assert_eq!(placeholder_runs(&[]), vec![]);
        assert_eq!(placeholder_runs(&[1, 2, 3]), vec![]);
        assert_eq!(
            placeholder_runs(&[p, p, 7, p, 9, p, p, p]),
            vec![(0, 2), (3, 1), (5, 3)]
        );
        assert_eq!(placeholder_runs(&[5, p]), vec![(1, 1)]);
    }

    #[test]
    fn rotary_angles_put_columns_before_rows() {
        let angles = rotary_angles(2, 4, 64, 10_000.).unwrap();
        assert_eq!(angles.len(), (1 + 8) * 32);
        assert!(
            angles[..32].iter().all(|&a| a == 0.),
            "class token at (0, 0)"
        );
        // patch 1 of the first block is (row 0, col 1): position (1, 2)
        let patch = &angles[2 * 32..3 * 32];
        assert_eq!(patch[0], 2.);
        assert_eq!(patch[16], 1.);
        let inverse_1 = 1. / 10_000f32.powf(2. / 32.);
        assert_eq!(patch[1], 2. * inverse_1);
        assert_eq!(patch[17], inverse_1);
        assert!(rotary_angles(2, 2, 62, 10_000.).is_err());
        assert!(rotary_angles(2, 2, 64, f32::NAN).is_err());
    }
}
