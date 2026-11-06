//! Native LFM3-VL image encoder and projector (LiquidAI reference layout).
use crate::{
    array::{Array, Dtype},
    checkpoint::Checkpoint,
    lfm2::{Projection, Qkv, half_weight},
    lfm3::{PreparedImage, patch_index, rotate},
};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;

#[derive(Deserialize)]
struct Config {
    hidden_size: i32,
    intermediate_size: i32,
    num_hidden_layers: usize,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    norm_eps: f32,
    conv_kernel_size: i32,
    in_channels: i32,
    patch_size: i32,
    temporal_patch_size: i32,
    spatial_merge_size: i32,
    layer_types: Vec<String>,
    rope_parameters: serde_json::Value,
}
struct Mlp {
    gate: Projection,
    up: Projection,
    down: Projection,
}
impl Mlp {
    fn load(cp: &Checkpoint, prefix: &str, dim: i32, ff: i32) -> Result<Self> {
        Ok(Self {
            gate: Projection::load(cp, &format!("{prefix}.w1"), dim, ff, false)?,
            up: Projection::load(cp, &format!("{prefix}.w3"), dim, ff, false)?,
            down: Projection::load(cp, &format!("{prefix}.w2"), ff, dim, false)?,
        })
    }
    fn forward(&self, x: &Array) -> Result<Array> {
        self.down
            .forward(&self.gate.forward(x)?.swiglu(&self.up.forward(x)?)?)
    }
}
// Stored once per layer, as in the text backbone; keep projections inline.
#[allow(clippy::large_enum_variant)]
enum Operator {
    Attention {
        qkv: Qkv,
        output: Projection,
        qnorm: Array,
        knorm: Array,
    },
    Conv {
        input: Projection,
        output: Projection,
        weight: Array,
    },
}
struct Layer {
    operator: Operator,
    norm: Array,
    ffn_norm: Array,
    mlp: Mlp,
}
pub(crate) struct Vision {
    config: Config,
    patchifier: Projection,
    embedding_norm: Array,
    layers: Vec<Layer>,
    projector_norm: Array,
    projector_bias: Array,
    projector_eps: f32,
    projector_in: Projection,
    projector_out: Projection,
}
impl Vision {
    pub(crate) fn load(cp: &Checkpoint, value: &serde_json::Value) -> Result<Self> {
        let c: Config = serde_json::from_value(value["vision_config"].clone())?;
        ensure!(
            c.hidden_size > 0
                && c.hidden_size <= i32::MAX / 3
                && c.intermediate_size > 0
                && c.num_attention_heads > 0
                && c.hidden_size % c.num_attention_heads == 0
                && (c.hidden_size / c.num_attention_heads) % 4 == 0
                && c.num_key_value_heads > 0
                && c.num_attention_heads % c.num_key_value_heads == 0
                && c.num_hidden_layers > 0
                && c.layer_types.len() == c.num_hidden_layers
                && c.layer_types
                    .iter()
                    .all(|t| t == "conv" || t == "full_attention")
                && c.norm_eps.is_finite()
                && c.norm_eps > 0.
                && c.conv_kernel_size > 0
                && c.conv_kernel_size % 2 == 1,
            "invalid LFM3 vision dimensions"
        );
        ensure!(
            c.patch_size == 16
                && c.temporal_patch_size == 2
                && c.spatial_merge_size == 2
                && c.in_channels == 3
                && value["downsample_factor"] == 2,
            "unsupported LFM3 vision patch layout"
        );
        ensure!(
            c.rope_parameters["rope_type"] == "axial"
                && c.rope_parameters["rope_theta"]
                    .as_f64()
                    .is_some_and(|t| t.is_finite() && t > 0.),
            "invalid vision RoPE"
        );
        ensure!(
            value["projector_hidden_act"] == "gelu",
            "unsupported vision projector activation"
        );
        let dim = c.hidden_size;
        let hd = dim / c.num_attention_heads;
        let kv = c.num_key_value_heads * hd;
        let weight = |name: &str, shape: &[i32]| half_weight(cp, name, Some(shape));
        let mut layers = Vec::new();
        for (index, kind) in c.layer_types.iter().enumerate() {
            let p = format!("model.vision_tower.layers.{index}");
            let operator = if kind == "full_attention" {
                let a = format!("{p}.self_attn");
                Operator::Attention {
                    qkv: Qkv::new([
                        Projection::load(cp, &format!("{a}.q_proj"), dim, dim, false)?,
                        Projection::load(cp, &format!("{a}.k_proj"), dim, kv, false)?,
                        Projection::load(cp, &format!("{a}.v_proj"), dim, kv, false)?,
                    ])?,
                    output: Projection::load(cp, &format!("{a}.out_proj"), dim, dim, false)?,
                    qnorm: weight(&format!("{a}.q_layernorm.weight"), &[hd])?,
                    knorm: weight(&format!("{a}.k_layernorm.weight"), &[hd])?,
                }
            } else {
                Operator::Conv {
                    input: Projection::load(cp, &format!("{p}.conv.in_proj"), dim, dim * 3, false)?,
                    output: Projection::load(cp, &format!("{p}.conv.out_proj"), dim, dim, false)?,
                    weight: weight(
                        &format!("{p}.conv.spatial_conv.weight"),
                        &[dim, 1, c.conv_kernel_size, c.conv_kernel_size],
                    )?
                    .transpose(&[0, 2, 3, 1])?,
                }
            };
            layers.push(Layer {
                operator,
                norm: weight(&format!("{p}.operator_norm.weight"), &[dim])?,
                ffn_norm: weight(&format!("{p}.ffn_norm.weight"), &[dim])?,
                mlp: Mlp::load(cp, &format!("{p}.feed_forward"), dim, c.intermediate_size)?,
            });
        }
        let p = "model.multi_modal_projector";
        let ff = i32::try_from(
            value["projector_hidden_size"]
                .as_i64()
                .context("missing projector size")?,
        )?;
        let text = i32::try_from(
            value["text_config"]["hidden_size"]
                .as_i64()
                .context("missing language size")?,
        )?;
        let bias = value["projector_bias"]
            .as_bool()
            .context("missing projector bias setting")?;
        let eps = value["projector_layer_norm_eps"]
            .as_f64()
            .context("missing projector epsilon")? as f32;
        ensure!(
            ff > 0 && text > 0 && dim.checked_mul(4).is_some() && eps.is_finite() && eps > 0.,
            "invalid projector dimensions"
        );
        Ok(Self {
            patchifier: Projection::load(cp, "model.vision_tower.patchifier", 1536, dim, false)?,
            embedding_norm: weight("model.vision_tower.embedding_norm.weight", &[dim])?,
            layers,
            projector_norm: weight(&format!("{p}.layer_norm.weight"), &[dim])?,
            projector_bias: weight(&format!("{p}.layer_norm.bias"), &[dim])?,
            projector_eps: eps,
            projector_in: Projection::load(cp, &format!("{p}.linear_1"), dim * 4, ff, bias)?,
            projector_out: Projection::load(cp, &format!("{p}.linear_2"), ff, text, bias)?,
            config: c,
        })
    }
    pub(crate) fn forward(&self, image: &PreparedImage) -> Result<Array> {
        let c = &self.config;
        let h = i32::try_from(image.height)?;
        let w = i32::try_from(image.width)?;
        let n = h.checked_mul(w).context("vision grid overflow")?;
        ensure!(
            h > 0
                && w > 0
                && h <= 2048
                && w <= 2048
                && n <= 65536
                && h % 2 == 0
                && w % 2 == 0
                && image.patches.len() == n as usize * 1536
                && image.patches.iter().all(|p| p.is_finite()),
            "invalid image patch grid"
        );
        let positions = (0..h)
            .flat_map(|r| (0..w).map(move |col| (r, col)))
            .collect::<Vec<_>>();
        let mut ordered = vec![(0, 0); n as usize];
        for (r, col) in positions {
            ordered[patch_index(r as u32, col as u32, h as u32, w as u32, 2)
                .context("invalid patch position")? as usize] = (r, col);
        }
        let hd = c.hidden_size / c.num_attention_heads;
        let theta = c.rope_parameters["rope_theta"]
            .as_f64()
            .context("missing axial theta")?;
        let angles = ordered
            .iter()
            .flat_map(|&(r, col)| {
                (0..hd / 2).map(move |i| {
                    let pos = if i < hd / 4 { r } else { col };
                    pos as f32
                        * theta.powf(-f64::from(2 * (i % (hd / 4))) / f64::from(hd / 2)) as f32
                })
            })
            .collect::<Vec<_>>();
        let pixels = Array::from_f32(&image.patches, &[n, 1536])?.astype(Dtype::Float16)?;
        let mut x = self
            .patchifier
            .forward(&pixels)?
            .rms_norm(&self.embedding_norm, c.norm_eps)?;
        for layer in &self.layers {
            let norm = x.rms_norm(&layer.norm, c.norm_eps)?;
            let mixed = match &layer.operator {
                Operator::Attention {
                    qkv,
                    output,
                    qnorm,
                    knorm,
                } => {
                    let projections = qkv.forward(&norm)?;
                    let q = rotate(
                        &projections[0]
                            .reshape(&[1, n, c.num_attention_heads, hd])?
                            .rms_norm(qnorm, c.norm_eps)?
                            .transpose(&[0, 2, 1, 3])?,
                        &angles,
                        hd,
                    )?;
                    let k = rotate(
                        &projections[1]
                            .reshape(&[1, n, c.num_key_value_heads, hd])?
                            .rms_norm(knorm, c.norm_eps)?
                            .transpose(&[0, 2, 1, 3])?,
                        &angles,
                        hd,
                    )?;
                    let v = projections[2]
                        .reshape(&[1, n, c.num_key_value_heads, hd])?
                        .transpose(&[0, 2, 1, 3])?;
                    output.forward(
                        &Array::sdpa(&q, &k, &v, (hd as f32).powf(-0.5), false)?
                            .transpose(&[0, 2, 1, 3])?
                            .reshape(&[n, c.hidden_size])?,
                    )?
                }
                Operator::Conv {
                    input,
                    output,
                    weight,
                } => {
                    let bcx = input.forward(&norm)?;
                    let d = c.hidden_size;
                    let gated = bcx.slice(1, 0, d)?.mul(&bcx.slice(1, 2 * d, 3 * d)?)?;
                    let raster = gated
                        .reshape(&[h / 2, w / 2, 2, 2, d])?
                        .transpose(&[0, 2, 1, 3, 4])?
                        .reshape(&[1, h, w, d])?;
                    let mixed = raster
                        .conv2d_depthwise(weight)?
                        .reshape(&[h / 2, 2, w / 2, 2, d])?
                        .transpose(&[0, 2, 1, 3, 4])?
                        .reshape(&[n, d])?;
                    output.forward(&bcx.slice(1, d, 2 * d)?.mul(&mixed)?)?
                }
            };
            x = x.add(&mixed)?;
            x = x.add(
                &layer
                    .mlp
                    .forward(&x.rms_norm(&layer.ffn_norm, c.norm_eps)?)?,
            )?;
            x.eval()?;
        }
        let normalized = x
            .layer_norm_without_weight(self.projector_eps)?
            .mul(&self.projector_norm)?
            .add(&self.projector_bias)?
            .reshape(&[n / 4, c.hidden_size * 4])?;
        self.projector_out
            .forward(&self.projector_in.forward(&normalized)?.gelu()?)
    }
}
