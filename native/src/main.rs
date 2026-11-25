use anyhow::{Context, Result, bail};
use clap::Parser;
mod cli;
#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
use cli::resolve_model_path;
use cli::{Cli, Command, HubAction, McpAction};
use mlxl3_native::{
    checkpoint,
    codec::{self, Codebook},
    registry,
    streaming::ThinkingSplitter,
};
use serde::Deserialize;
#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
use serde::Serialize;
use serde_json::{Value, json};
#[cfg(all(feature = "mlx", feature = "chat"))]
use std::collections::VecDeque;
#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(all(
    not(feature = "mlx"),
    feature = "chat",
    target_os = "macos",
    feature = "direct-metal"
))]
mod direct_bridge;
#[cfg(all(feature = "mlx", feature = "chat"))]
mod mtp_tune;
#[cfg(all(
    not(feature = "mlx"),
    feature = "chat",
    target_os = "macos",
    feature = "direct-metal"
))]
use direct_bridge::{ChatMtpHead, NativeChatModel, PromptCache};
#[cfg(all(feature = "mlx", feature = "chat"))]
type RoundDraft = mlxl3_native::dflash::DFlashWeights;
#[cfg(all(
    not(feature = "mlx"),
    feature = "chat",
    target_os = "macos",
    feature = "direct-metal"
))]
type RoundDraft = mlxl3_native::native_draft::Session;
#[cfg(any(
    test,
    all(
        feature = "chat",
        any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
    )
))]
mod openai;
use std::io::{self, BufRead, Read, Write};
#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
use std::path::PathBuf;

#[derive(Deserialize)]
struct CodecRequest {
    op: String,
    #[serde(default)]
    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), feature = "mlx"))]
    path: String,
    #[serde(default)]
    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), feature = "mlx"))]
    prefix: String,
    #[serde(default)]
    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), feature = "mlx"))]
    prefixes: Vec<String>,
    #[serde(default)]
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    slots: Option<usize>,
    #[serde(default)]
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    token_ids: Vec<u32>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    mode: u32,
    #[serde(default)]
    data: Vec<u16>,
    #[serde(default)]
    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), feature = "mlx"))]
    x: Vec<u16>,
    #[serde(default)]
    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), feature = "mlx"))]
    x_f32: Option<Vec<f32>>,
    #[serde(default)]
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    x_layout: Option<mlxl3_native::tensor_layout::MatrixLayout>,
    #[serde(default)]
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    scale_layout: Option<mlxl3_native::tensor_layout::MatrixLayout>,
    #[serde(default)]
    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), feature = "mlx"))]
    cols: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    suh: Vec<u16>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    svh: Vec<u16>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    widths: Vec<usize>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    q: Vec<u16>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    v: Vec<u16>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    g: Vec<f32>,
    #[serde(default)]
    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), feature = "mlx"))]
    beta: Vec<u16>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    state: Vec<f32>,
    #[serde(default)]
    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), feature = "mlx"))]
    key_heads: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    value_heads: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    value_dims: usize,
    #[serde(default)]
    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), feature = "mlx"))]
    normalize: bool,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    groups: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    top_groups: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    scale: f32,
    #[serde(default)]
    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), feature = "mlx"))]
    selected: Vec<u32>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    projections_per_route: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    projection_stride_tiles: usize,
    #[serde(default)]
    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), feature = "mlx"))]
    return_raw: bool,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    down_data: Vec<u16>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    down_suh: Vec<u16>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    down_svh: Vec<u16>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    layer: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    top_k: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    conv_length: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    conv_state: Vec<u16>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    keys: Vec<u16>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    values: Vec<u16>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    token: u32,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    tokens: Vec<u32>,
}
#[cfg(feature = "mlx")]
impl CodecRequest {
    fn validate_moe_codec(&self) -> Result<()> {
        match self.op.as_str() {
            "mlx-switch" => {
                let experts = i32::try_from(self.key_heads)?;
                let top_k = if self.top_k > 0 {
                    self.top_k
                } else {
                    self.selected.len()
                };
                let rows = mlxl3_native::switch_glu_route_rows(&self.selected, experts, top_k)
                    .context("invalid SwitchGLU routes")? as usize;
                anyhow::ensure!(self.x.len().is_multiple_of(rows), "invalid SwitchGLU rows");
                mlxl3_native::switch_glu_layout(
                    experts,
                    i32::try_from(top_k)?,
                    i32::try_from(self.x.len() / rows)?,
                    i32::try_from(self.cols)?,
                    self.k,
                )
                .context("invalid SwitchGLU layout")?;
            }
            "mlx-checkpoint-switch" => {
                anyhow::ensure!(!self.x.is_empty(), "empty SwitchGLU input");
                mlxl3_native::switch_glu_route_rows(
                    &self.selected,
                    i32::try_from(self.key_heads)?,
                    self.selected.len(),
                )
                .context("invalid checkpoint SwitchGLU routes")?;
            }
            "mlx-swiglu-prepare" => {
                let width = self
                    .cols
                    .checked_mul(2)
                    .filter(|&n| n > 0)
                    .context("invalid gate/up width")?;
                anyhow::ensure!(
                    self.suh.len().is_multiple_of(width),
                    "invalid gate/up scales"
                );
                let experts = i32::try_from(self.suh.len() / width)?;
                mlxl3_native::switch_glu_route_rows(&self.selected, experts, 1)
                    .context("invalid SwiGLU routes")?;
            }
            _ => (),
        }
        Ok(())
    }
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
impl CodecRequest {
    fn input_view(&self, gpu: &mlxl3_native::gpu::Metal) -> Result<mlxl3_native::gpu::MatrixView> {
        let layout = self.x_layout.context("missing input matrix layout")?;
        let storage = if let Some(data) = &self.x_f32 {
            anyhow::ensure!(self.x.is_empty(), "conflicting input buffer dtypes");
            gpu.f32_storage(data)?
        } else {
            gpu.f16_storage(&self.x)?
        };
        storage.matrix(layout)
    }
}

fn default_k() -> usize {
    4
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
fn flash_checkpoint_epsilon(checkpoint: &checkpoint::Checkpoint) -> Result<f32> {
    let config: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(checkpoint.path.join("config.json"))?)?;
    Ok(config["text_config"]["rms_norm_eps"]
        .as_f64()
        .context("missing Flash norm epsilon")? as f32)
}

fn codec_loop() -> Result<()> {
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
    let mut gpu = None;
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for line in io::stdin().lock().lines() {
        let result = (|| -> Result<Value> {
            let request: CodecRequest =
                serde_json::from_str(&line?).context("invalid codec request")?;
            let cb = Codebook::try_from(request.mode)?;
            #[cfg(feature = "mlx")]
            request.validate_moe_codec()?;
            match request.op.as_str() {
                "pack" => Ok(json!(codec::pack(&request.data, request.k)?)),
                "unpack" => Ok(json!(codec::unpack(&request.data, request.k)?)),
                "codebook" => Ok(json!(
                    request
                        .data
                        .iter()
                        .map(|&s| codec::decode_codeword(s, cb).to_bits())
                        .collect::<Vec<_>>()
                )),
                "permutation" => Ok(json!(codec::permutation().to_vec())),
                #[cfg(feature = "mlx")]
                "mlx-linear" => {
                    use mlxl3_native::{array::Array, linear::Exl3Linear};
                    codec::check_k(request.k)?;
                    let rows = i32::try_from(request.suh.len())?;
                    anyhow::ensure!(
                        rows > 0 && request.x.len().is_multiple_of(rows as usize),
                        "invalid linear input rows"
                    );
                    let matrix_rows = i32::try_from(request.x.len() / rows as usize)?;
                    let cols = i32::try_from(request.cols)?;
                    let trellis = Array::from_u16(
                        &request.data,
                        &[rows / 16, cols / 16, (16 * request.k) as i32],
                    )?;
                    let layer = Exl3Linear::new(
                        trellis,
                        Array::from_f16_bits(&request.suh, &[rows])?,
                        Array::from_f16_bits(&request.svh, &[cols])?,
                        None,
                        request.k,
                        cb,
                    )?;
                    let x = Array::from_f16_bits(&request.x, &[matrix_rows, rows])?;
                    Ok(json!(layer.forward(&x)?.to_f16_bits()?))
                }
                #[cfg(feature = "mlx")]
                "mlx-group" => {
                    use mlxl3_native::{
                        array::Array,
                        linear::{Exl3Group, Exl3Linear},
                    };
                    codec::check_k(request.k)?;
                    anyhow::ensure!(!request.widths.is_empty(), "missing group widths");
                    anyhow::ensure!(
                        request.suh.len().is_multiple_of(request.widths.len()),
                        "invalid grouped input scales"
                    );
                    let rows = request.suh.len() / request.widths.len();
                    anyhow::ensure!(
                        rows > 0
                            && rows.is_multiple_of(128)
                            && request.x.len().is_multiple_of(rows),
                        "invalid grouped input size"
                    );
                    let matrix_rows = request.x.len() / rows;
                    anyhow::ensure!(
                        request.widths.len() >= 2 && request.k != 7,
                        "unsupported group"
                    );
                    anyhow::ensure!(
                        request.suh.len()
                            == rows
                                .checked_mul(request.widths.len())
                                .context("scale overflow")?,
                        "invalid grouped input scales"
                    );
                    let mut data_offset = 0usize;
                    let mut scale_offset = 0usize;
                    let mut linears = Vec::new();
                    for (index, width) in request.widths.iter().copied().enumerate() {
                        anyhow::ensure!(
                            width > 0 && width.is_multiple_of(128),
                            "invalid group width"
                        );
                        let words = (rows / 16)
                            .checked_mul(width / 16)
                            .and_then(|n| n.checked_mul(16 * request.k))
                            .context("trellis overflow")?;
                        let end = data_offset
                            .checked_add(words)
                            .context("trellis offset overflow")?;
                        let scale_end = scale_offset
                            .checked_add(width)
                            .context("scale offset overflow")?;
                        let r = i32::try_from(rows)?;
                        let c = i32::try_from(width)?;
                        linears.push(Exl3Linear::new(
                            Array::from_u16(
                                request
                                    .data
                                    .get(data_offset..end)
                                    .context("truncated grouped trellis")?,
                                &[r / 16, c / 16, (16 * request.k) as i32],
                            )?,
                            Array::from_f16_bits(
                                &request.suh[index * rows..(index + 1) * rows],
                                &[r],
                            )?,
                            Array::from_f16_bits(
                                request
                                    .svh
                                    .get(scale_offset..scale_end)
                                    .context("truncated grouped scales")?,
                                &[c],
                            )?,
                            None,
                            request.k,
                            cb,
                        )?);
                        data_offset = end;
                        scale_offset = scale_end;
                    }
                    anyhow::ensure!(
                        data_offset == request.data.len() && scale_offset == request.svh.len(),
                        "extra grouped data"
                    );
                    let group = Exl3Group::new(linears)?;
                    let outputs = group.forward(&Array::from_f16_bits(
                        &request.x,
                        &[i32::try_from(matrix_rows)?, i32::try_from(rows)?],
                    )?)?;
                    Ok(json!(
                        outputs
                            .iter()
                            .map(Array::to_f16_bits)
                            .collect::<Result<Vec<_>>>()?
                    ))
                }
                #[cfg(feature = "mlx")]
                "mlx-gdn" => {
                    use mlxl3_native::{array::Array, gated_delta};
                    let batch = 1;
                    let key_heads = i32::try_from(request.key_heads)?;
                    let value_heads = i32::try_from(request.value_heads)?;
                    let value_dims = i32::try_from(request.value_dims)?;
                    let row_width = usize::try_from(key_heads)? * 128;
                    anyhow::ensure!(
                        row_width > 0 && request.q.len().is_multiple_of(row_width),
                        "invalid GDN time dimension"
                    );
                    let time = i32::try_from(request.q.len() / row_width)?;
                    let q_shape = [batch, time, key_heads, 128];
                    let v_shape = [batch, time, value_heads, value_dims];
                    let state_shape = [batch, value_heads, value_dims, 128];
                    let (output, state) = gated_delta::step(
                        &Array::from_f16_bits(&request.q, &q_shape)?,
                        &Array::from_f16_bits(&request.x, &q_shape)?,
                        &Array::from_f16_bits(&request.v, &v_shape)?,
                        &Array::from_f32(&request.g, &[batch, time, value_heads])?,
                        &Array::from_f16_bits(&request.beta, &[batch, time, value_heads])?,
                        &Array::from_f32(&request.state, &state_shape)?,
                    )?;
                    Ok(json!({
                        "output": output.to_f16_bits()?,
                        "state": state.to_f32()?,
                    }))
                }
                #[cfg(feature = "mlx")]
                "mlx-gdn-vector" => {
                    use mlxl3_native::{array::Array, gated_delta};
                    let heads = i32::try_from(request.key_heads)?;
                    let shape = [1, 1, heads, 128];
                    let state_shape = [1, heads, 128, 128];
                    let (output, state) = gated_delta::step_vector(
                        &Array::from_f16_bits(&request.q, &shape)?,
                        &Array::from_f16_bits(&request.x, &shape)?,
                        &Array::from_f16_bits(&request.v, &shape)?,
                        &Array::from_f32(&request.g, &shape)?,
                        &Array::from_f16_bits(&request.beta, &[1, 1, heads])?,
                        &Array::from_f32(&request.state, &state_shape)?,
                    )?;
                    Ok(json!({
                        "output": output.to_f16_bits()?,
                        "state": state.to_f32()?,
                    }))
                }
                #[cfg(feature = "mlx")]
                "mlx-router" => {
                    use mlxl3_native::{array::Array, router};
                    let elements = request.x_f32.as_ref().map_or(request.data.len(), Vec::len);
                    let experts = if request.cols == 0 {
                        elements
                    } else {
                        request.cols
                    };
                    anyhow::ensure!(
                        experts > 0 && elements > 0 && elements.is_multiple_of(experts),
                        "invalid router rows"
                    );
                    let shape = [i32::try_from(elements / experts)?, i32::try_from(experts)?];
                    let input = if let Some(data) = &request.x_f32 {
                        anyhow::ensure!(request.data.is_empty(), "conflicting router input dtypes");
                        Array::from_f32(data, &shape)?
                    } else {
                        Array::from_f16_bits(&request.data, &shape)?
                    };
                    let (indices, scores) = router::topk(&input, request.k, request.normalize)?;
                    let words = if request.x_f32.is_some() {
                        scores
                            .to_f32()?
                            .into_iter()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>()
                    } else {
                        scores.to_f16_bits()?.into_iter().map(u32::from).collect()
                    };
                    Ok(json!({"indices": indices.to_u32()?, "scores": words}))
                }
                #[cfg(feature = "mlx")]
                "mlx-router-biased" => {
                    use mlxl3_native::{array::Array, router};
                    let experts = i32::try_from(request.data.len())?;
                    let (indices, scores) = router::topk_biased(
                        &Array::from_f16_bits(&request.data, &[1, experts])?,
                        &Array::from_f16_bits(&request.beta, &[experts])?,
                        request.k,
                    )?;
                    Ok(json!({
                        "indices": indices.to_u32()?,
                        "scores": scores.to_f16_bits()?,
                    }))
                }
                #[cfg(feature = "mlx")]
                "mlx-router-grouped" => {
                    use mlxl3_native::{array::Array, router};
                    let experts = i32::try_from(request.g.len())?;
                    let (indices, scores) = router::grouped_topk_biased(
                        &Array::from_f32(&request.g, &[1, experts])?,
                        &Array::from_f32(&request.state, &[experts])?,
                        request.k,
                        request.groups,
                        request.top_groups,
                        request.scale,
                    )?;
                    Ok(json!({
                        "indices": indices.to_u32()?,
                        "scores": scores.to_f32()?,
                    }))
                }
                #[cfg(feature = "mlx")]
                "mlx-expert" => {
                    use mlxl3_native::{array::Array, linear::expert_mapped};
                    codec::check_k(request.k)?;
                    let projections = i32::try_from(request.projections_per_route)?;
                    anyhow::ensure!(projections > 0, "invalid projection count");
                    let rows = i32::try_from(
                        request
                            .selected
                            .len()
                            .checked_mul(request.projections_per_route)
                            .context("expert row count overflow")?,
                    )?;
                    anyhow::ensure!(rows > 0, "missing expert routes");
                    let input_dims = i32::try_from(request.x.len())?
                        .checked_div(rows)
                        .context("invalid expert input dimensions")?;
                    let output_dims = i32::try_from(request.cols)?;
                    let input_tiles = input_dims / 16;
                    let packed_per_tile = i32::try_from(16 * request.k)?;
                    let source_tiles = i32::try_from(request.data.len())?
                        .checked_div(
                            input_tiles
                                .checked_mul(packed_per_tile)
                                .context("expert trellis size overflow")?,
                        )
                        .context("invalid expert trellis dimensions")?;
                    let trellis = Array::from_u16(
                        &request.data,
                        &[input_tiles, source_tiles, packed_per_tile],
                    )?;
                    let x = Array::from_f16_bits(&request.x, &[rows, input_dims])?;
                    let suh = Array::from_f16_bits(&request.suh, &[rows, input_dims])?;
                    let svh = (!request.return_raw)
                        .then(|| Array::from_f16_bits(&request.svh, &[rows, output_dims]))
                        .transpose()?;
                    let output = expert_mapped(
                        &x,
                        &trellis,
                        Some(&suh),
                        svh.as_ref(),
                        &Array::from_u32(
                            &request.selected,
                            &[i32::try_from(request.selected.len())?],
                        )?,
                        output_dims,
                        projections,
                        i32::try_from(request.projection_stride_tiles)?,
                        request.k,
                        cb,
                        false,
                        request.return_raw,
                    )?;
                    Ok(if request.return_raw {
                        json!({ "f32": output.to_f32()? })
                    } else {
                        json!({ "f16": output.to_f16_bits()? })
                    })
                }
                #[cfg(feature = "mlx")]
                "mlx-swiglu-prepare" => {
                    use mlxl3_native::{array::Array, moe};
                    let hidden = i32::try_from(request.cols)?;
                    let slots = i32::try_from(request.selected.len())?;
                    let experts = i32::try_from(request.suh.len())?
                        .checked_div(2 * hidden)
                        .context("invalid gate/up scales")?;
                    let output = moe::swiglu_down_input(
                        &Array::from_f32(&request.state, &[slots * 2, hidden])?,
                        &Array::from_f16_bits(&request.suh, &[experts, 2, hidden])?,
                        &Array::from_f16_bits(&request.svh, &[experts, hidden])?,
                        &Array::from_u32(&request.selected, &[slots])?,
                    )?;
                    Ok(json!(output.to_f16_bits()?))
                }
                #[cfg(feature = "mlx")]
                "mlx-expert-reduce" => {
                    use mlxl3_native::{array::Array, moe};
                    let width = i32::try_from(request.cols)?;
                    let slots = i32::try_from(request.beta.len())?;
                    let output = moe::finish_and_reduce(
                        &Array::from_f32(&request.state, &[slots, width])?,
                        &Array::from_f16_bits(&request.svh, &[slots, width])?,
                        &Array::from_f16_bits(&request.beta, &[1, slots])?,
                    )?;
                    Ok(json!(output.to_f16_bits()?))
                }
                #[cfg(feature = "mlx")]
                "mlx-switch" => {
                    use mlxl3_native::{array::Array, moe::Exl3SwitchGlu};
                    let experts = i32::try_from(request.key_heads)?;
                    let top_k = i32::try_from(if request.top_k > 0 {
                        request.top_k
                    } else {
                        request.selected.len()
                    })?;
                    anyhow::ensure!(
                        top_k > 0
                            && request
                                .selected
                                .len()
                                .is_multiple_of(usize::try_from(top_k)?),
                        "invalid SwitchGLU routes"
                    );
                    let rows = i32::try_from(request.selected.len() / usize::try_from(top_k)?)?;
                    anyhow::ensure!(
                        request.x.len().is_multiple_of(usize::try_from(rows)?),
                        "invalid SwitchGLU rows"
                    );
                    let input = i32::try_from(request.x.len() / usize::try_from(rows)?)?;
                    let hidden = i32::try_from(request.cols)?;
                    let gu = Array::from_u16(
                        &request.data,
                        &[input / 16, experts * hidden / 8, (16 * request.k) as i32],
                    )?;
                    let down = Array::from_u16(
                        &request.down_data,
                        &[hidden / 16, experts * input / 16, (16 * request.k) as i32],
                    )?;
                    let switch = Exl3SwitchGlu::new(
                        gu,
                        Array::from_f16_bits(&request.suh, &[experts, 2, input])?,
                        Array::from_f16_bits(&request.svh, &[experts, 2, hidden])?,
                        down,
                        Array::from_f16_bits(&request.down_suh, &[experts, hidden])?,
                        Array::from_f16_bits(&request.down_svh, &[experts, input])?,
                        top_k,
                        request.k,
                        cb,
                    )?;
                    let output = switch.forward(
                        &Array::from_f16_bits(&request.x, &[rows, input])?,
                        &Array::from_u32(&request.selected, &[rows, top_k])?,
                        &Array::from_f16_bits(&request.beta, &[rows, top_k])?,
                    )?;
                    Ok(json!(output.to_f16_bits()?))
                }
                #[cfg(feature = "mlx")]
                "mlx-checkpoint-group" => {
                    use mlxl3_native::{
                        array::Array,
                        linear::{Exl3Group, Exl3Linear},
                    };
                    let c = checkpoint::inspect(std::path::Path::new(&request.path))?;
                    anyhow::ensure!(!request.prefixes.is_empty(), "empty projection bundle");
                    let layers = request
                        .prefixes
                        .iter()
                        .map(|p| Exl3Linear::from_checkpoint(&c, p))
                        .collect::<Result<Vec<_>>>()?;
                    let grouped = Exl3Group::compatible(&layers.iter().collect::<Vec<_>>());
                    let input =
                        Array::from_f16_bits(&request.x, &[1, i32::try_from(request.x.len())?])?;
                    let outputs = if grouped {
                        Exl3Group::new(layers)?.forward(&input)?
                    } else {
                        layers
                            .iter()
                            .map(|p| p.forward(&input))
                            .collect::<Result<Vec<_>>>()?
                    };
                    Ok(
                        json!({"grouped":grouped,"outputs":outputs.iter().map(|v|v.to_f16_bits()).collect::<Result<Vec<_>>>()?}),
                    )
                }
                #[cfg(feature = "mlx")]
                "mlx-checkpoint-switch" => {
                    use mlxl3_native::{array::Array, moe::Exl3SwitchGlu};
                    let c = checkpoint::inspect(std::path::Path::new(&request.path))?;
                    let experts = i32::try_from(request.key_heads)?;
                    let count = i32::try_from(request.selected.len())?;
                    let input = i32::try_from(request.x.len())?;
                    let layer = if c.model_type == "lfm2_moe" {
                        Exl3SwitchGlu::from_lfm_checkpoint(&c, &request.prefix, experts, count)?
                    } else {
                        Exl3SwitchGlu::from_checkpoint(&c, &request.prefix, experts, count)?
                    };
                    Ok(json!(
                        layer
                            .forward(
                                &Array::from_f16_bits(&request.x, &[1, input])?,
                                &Array::from_u32(&request.selected, &[1, count])?,
                                &Array::from_f16_bits(&request.beta, &[1, count])?
                            )?
                            .to_f16_bits()?
                    ))
                }
                #[cfg(feature = "mlx")]
                "mlx-qwen-moe" => {
                    use mlxl3_native::{array::Array, checkpoint, qwen35::Moe};
                    let checkpoint = checkpoint::inspect(std::path::Path::new(&request.path))?;
                    let hidden = i32::try_from(request.cols)?;
                    let expert_hidden = i32::try_from(request.value_dims)?;
                    let experts = i32::try_from(request.key_heads)?;
                    let block = Moe::load(
                        &checkpoint,
                        &format!("model.language_model.layers.{}.mlp", request.layer),
                        hidden,
                        expert_hidden,
                        experts,
                        request.top_k,
                    )?;
                    let output = block.forward(&Array::from_f16_bits(&request.x, &[1, hidden])?)?;
                    Ok(json!(output.to_f16_bits()?))
                }
                #[cfg(feature = "mlx")]
                "mlx-qwen-gdn" | "mlx-qwen-gdn-sequence" => {
                    use mlxl3_native::{array::Array, checkpoint, qwen35::GatedDelta};
                    let checkpoint = checkpoint::inspect(std::path::Path::new(&request.path))?;
                    let hidden = i32::try_from(request.cols)?;
                    let key_heads = i32::try_from(request.key_heads)?;
                    let value_heads = i32::try_from(request.value_heads)?;
                    let value_dims = i32::try_from(request.value_dims)?;
                    let conv_length = i32::try_from(request.conv_length)?;
                    let key_dims = 128;
                    let conv_dims = 2 * key_heads * key_dims + value_heads * value_dims;
                    let mut block = GatedDelta::load(
                        &checkpoint,
                        &format!("model.language_model.layers.{}.linear_attn", request.layer),
                        hidden,
                        key_heads,
                        value_heads,
                        key_dims,
                        value_dims,
                        conv_length,
                        1e-6,
                    )?;
                    block.set_state(
                        (!request.conv_state.is_empty())
                            .then(|| {
                                Array::from_f16_bits(
                                    &request.conv_state,
                                    &[1, conv_length - 1, conv_dims],
                                )
                            })
                            .transpose()?,
                        (!request.state.is_empty())
                            .then(|| {
                                Array::from_f32(
                                    &request.state,
                                    &[1, value_heads, value_dims, key_dims],
                                )
                            })
                            .transpose()?,
                    )?;
                    let output = if request.op == "mlx-qwen-gdn-sequence" {
                        anyhow::ensure!(
                            hidden > 0
                                && !request.x.is_empty()
                                && request.x.len().is_multiple_of(hidden as usize),
                            "invalid Qwen sequence rows"
                        );
                        let mut rows = Vec::new();
                        for row in request.x.chunks_exact(hidden as usize) {
                            let output =
                                block.forward(&Array::from_f16_bits(row, &[1, 1, hidden])?)?;
                            output.eval()?;
                            let (conv, state) = block.states()?;
                            conv.eval()?;
                            state.eval()?;
                            rows.push(output);
                        }
                        Array::concatenate(&rows.iter().collect::<Vec<_>>(), 1)?
                    } else {
                        block.forward(&Array::from_f16_bits(&request.x, &[1, 1, hidden])?)?
                    };
                    let (conv, recurrent) = block.states()?;
                    Ok(json!({
                        "output": output.to_f16_bits()?,
                        "conv_state": conv.to_f16_bits()?,
                        "state": recurrent.to_f32()?,
                    }))
                }
                #[cfg(feature = "mlx")]
                "mlx-qwen-layer" => {
                    use mlxl3_native::{array::Array, checkpoint, qwen35::LinearLayer};
                    let checkpoint = checkpoint::inspect(std::path::Path::new(&request.path))?;
                    let hidden = i32::try_from(request.cols)?;
                    let key_heads = i32::try_from(request.key_heads)?;
                    let value_heads = i32::try_from(request.value_heads)?;
                    let value_dims = i32::try_from(request.value_dims)?;
                    let conv_length = i32::try_from(request.conv_length)?;
                    let conv_dims = 2 * key_heads * 128 + value_heads * value_dims;
                    let mut layer = LinearLayer::load(
                        &checkpoint,
                        request.layer,
                        hidden,
                        key_heads,
                        value_heads,
                        128,
                        value_dims,
                        conv_length,
                        512,
                        256,
                        request.top_k,
                        1e-6,
                    )?;
                    layer.set_state(
                        (!request.conv_state.is_empty())
                            .then(|| {
                                Array::from_f16_bits(
                                    &request.conv_state,
                                    &[1, conv_length - 1, conv_dims],
                                )
                            })
                            .transpose()?,
                        (!request.state.is_empty())
                            .then(|| {
                                Array::from_f32(&request.state, &[1, value_heads, value_dims, 128])
                            })
                            .transpose()?,
                    )?;
                    let (output, trace) =
                        layer.trace(&Array::from_f16_bits(&request.x, &[1, 1, hidden])?)?;
                    let (conv, recurrent) = layer.states()?;
                    Ok(json!({
                        "output": output.to_f16_bits()?,
                        "trace": trace.iter().map(Array::to_f16_bits).collect::<Result<Vec<_>>>()?,
                        "conv_state": conv.to_f16_bits()?,
                        "state": recurrent.to_f32()?,
                    }))
                }
                #[cfg(feature = "mlx")]
                "mlx-qwen-attn-layer" => {
                    use mlxl3_native::{array::Array, checkpoint, qwen35::AttentionLayer};
                    let checkpoint = checkpoint::inspect(std::path::Path::new(&request.path))?;
                    let hidden = i32::try_from(request.cols)?;
                    let heads = i32::try_from(request.key_heads)?;
                    let kv_heads = i32::try_from(request.value_heads)?;
                    let head_dim = i32::try_from(request.value_dims)?;
                    anyhow::ensure!(
                        request.keys.len() == request.values.len(),
                        "KV cache lengths differ"
                    );
                    let cache_tokens = if request.keys.is_empty() {
                        0
                    } else {
                        i32::try_from(request.keys.len())?
                            .checked_div(kv_heads * head_dim)
                            .context("invalid KV cache size")?
                    };
                    let mut layer = AttentionLayer::load(
                        &checkpoint,
                        request.layer,
                        hidden,
                        heads,
                        kv_heads,
                        head_dim,
                        64,
                        10_000_000.,
                        512,
                        256,
                        request.top_k,
                        1e-6,
                    )?;
                    layer.set_state(
                        (!request.keys.is_empty())
                            .then(|| {
                                Array::from_f16_bits(
                                    &request.keys,
                                    &[1, kv_heads, cache_tokens, head_dim],
                                )
                            })
                            .transpose()?,
                        (!request.values.is_empty())
                            .then(|| {
                                Array::from_f16_bits(
                                    &request.values,
                                    &[1, kv_heads, cache_tokens, head_dim],
                                )
                            })
                            .transpose()?,
                    )?;
                    let output =
                        layer.forward(&Array::from_f16_bits(&request.x, &[1, 1, hidden])?)?;
                    let (keys, values) = layer.states()?;
                    Ok(json!({
                        "output": output.to_f16_bits()?,
                        "keys": keys.to_f16_bits()?,
                        "values": values.to_f16_bits()?,
                    }))
                }
                #[cfg(feature = "mlx")]
                "mlx-qwen-model" => {
                    use mlxl3_native::qwen35::Qwen35Moe;
                    let mut model = Qwen35Moe::load(std::path::Path::new(&request.path))?;
                    let tokens = if request.tokens.is_empty() {
                        vec![request.token]
                    } else {
                        request.tokens
                    };
                    Ok(json!(
                        tokens
                            .into_iter()
                            .map(|token| model.forward(token)?.to_f16_bits())
                            .collect::<Result<Vec<_>>>()?
                    ))
                }
                #[cfg(feature = "mlx")]
                "mlx-qwen-trace" => {
                    use mlxl3_native::qwen35::Qwen35Moe;
                    let mut model = Qwen35Moe::load(std::path::Path::new(&request.path))?;
                    let (logits, layers) = model.trace(request.token)?;
                    Ok(json!({ "layers": layers, "logits": logits.to_f16_bits()? }))
                }
                #[cfg(all(target_os = "macos", feature = "direct-metal"))]
                "metal-pack"
                | "metal-decode"
                | "metal-qmv"
                | "metal-prepare128"
                | "metal-checkpoint-linear"
                | "metal-exl3-linear"
                | "metal-qwen-gdn"
                | "metal-checkpoint-switch"
                | "metal-checkpoint-group"
                | "metal-checkpoint-expert"
                | "metal-swiglu"
                | "metal-prepare-view"
                | "metal-router"
                | "metal-checkpoint-router"
                | "metal-checkpoint-moe"
                | "metal-checkpoint-hc"
                | "metal-checkpoint-mlp"
                | "metal-checkpoint-qsa-layer"
                | "metal-checkpoint-qsa"
                | "metal-checkpoint-gdn"
                | "metal-checkpoint-gdn-layer"
                | "metal-checkpoint-ple"
                | "metal-checkpoint-ple-layer" => {
                    if gpu.is_none() {
                        gpu = Some(mlxl3_native::gpu::Metal::new()?);
                    }
                    let gpu = gpu.as_ref().context("Metal is unavailable")?;
                    match request.op.as_str() {
                        "metal-checkpoint-ple" | "metal-checkpoint-ple-layer" => {
                            let input = request.input_view(gpu)?;
                            let [rows, width] = input.shape();
                            anyhow::ensure!(
                                (1..=16).contains(&rows)
                                    && width == 10240
                                    && request.token_ids.len() == rows as usize
                                    && request.token_ids.iter().all(|&id| id < 248320),
                                "invalid PLE tokens or streams"
                            );
                            let c = checkpoint::inspect(std::path::Path::new(&request.path))?;
                            let eps = flash_checkpoint_epsilon(&c)?;
                            let mut out = Vec::new();
                            if request.op.ends_with("-layer") {
                                let mut block = mlxl3_native::gpu::FlashPleLayer::load(
                                    gpu,
                                    &c,
                                    &request.prefix,
                                    eps,
                                    request.slots.unwrap_or(10),
                                )?;
                                for (i, &t) in request.token_ids.iter().enumerate() {
                                    out.push(json!(
                                        block.forward(&input.row(i as u32)?, t)?.read_output()?
                                    ));
                                }
                            } else {
                                let mut ple = mlxl3_native::gpu::FlashPle::load(
                                    gpu,
                                    &c,
                                    &request.prefix,
                                    eps,
                                )?;
                                for (i, &t) in request.token_ids.iter().enumerate() {
                                    let r =
                                        ple.forward(&input.row(i as u32)?, t, request.return_raw)?;
                                    out.push(json!({"values":r.values.read_output()?,"trace":r.trace,"read_bytes":r.read_bytes}));
                                }
                            }
                            Ok(json!(out))
                        }
                        "metal-checkpoint-qsa" | "metal-checkpoint-qsa-layer" => {
                            let input = request.input_view(gpu)?;
                            let [rows, width] = input.shape();
                            anyhow::ensure!(
                                (1..=512).contains(&rows)
                                    && width
                                        == if request.op.ends_with("-layer") {
                                            10240
                                        } else {
                                            2560
                                        },
                                "invalid Flash QSA rows"
                            );
                            let c = checkpoint::inspect(std::path::Path::new(&request.path))?;
                            let config: serde_json::Value = serde_json::from_reader(
                                std::fs::File::open(c.path.join("config.json"))?,
                            )?;
                            let theta = config["text_config"]["rope_parameters"]["rope_theta"]
                                .as_f64()
                                .context("missing Flash RoPE")?
                                as f32;
                            if request.op.ends_with("-layer") {
                                let mut layer = mlxl3_native::gpu::FlashQsaLayer::load(
                                    gpu,
                                    &c,
                                    &request.prefix,
                                    flash_checkpoint_epsilon(&c)?,
                                    theta,
                                    rows,
                                    request.slots.unwrap_or(10),
                                )?;
                                return (0..rows)
                                    .map(|r| {
                                        Ok(json!(layer.forward(&input.row(r)?)?.read_output()?))
                                    })
                                    .collect::<Result<Vec<_>>>()
                                    .map(|v| json!(v));
                            }
                            let mut a = mlxl3_native::gpu::FlashQsa::load(
                                gpu,
                                &c,
                                &request.prefix,
                                flash_checkpoint_epsilon(&c)?,
                                theta,
                                rows,
                            )?;
                            let mut result = Vec::new();
                            for r in 0..rows {
                                let row = a.forward(&input.row(r)?, request.return_raw)?;
                                result.push(
                                    json!({"values":row.values.read_output()?,"trace":row.trace}),
                                );
                            }
                            Ok(json!(result))
                        }
                        "metal-checkpoint-gdn" | "metal-checkpoint-gdn-layer" => {
                            let input = request.input_view(gpu)?;
                            let [rows, width] = input.shape();
                            let full = request.op.ends_with("-layer");
                            anyhow::ensure!(
                                (1..=8).contains(&rows) && width == if full { 10240 } else { 2560 },
                                "invalid Flash GDN row count or width"
                            );
                            let checkpoint =
                                checkpoint::inspect(std::path::Path::new(&request.path))?;
                            let eps = flash_checkpoint_epsilon(&checkpoint)?;
                            let mut result = Vec::new();
                            if full {
                                let mut layer = mlxl3_native::gpu::FlashGdnLayer::load(
                                    gpu,
                                    &checkpoint,
                                    &request.prefix,
                                    eps,
                                    request.slots.unwrap_or(10),
                                )?;
                                for r in 0..rows {
                                    result
                                        .push(json!(layer.forward(&input.row(r)?)?.read_output()?));
                                }
                            } else {
                                let mut layer = mlxl3_native::gpu::FlashGdn::load(
                                    gpu,
                                    &checkpoint,
                                    &request.prefix,
                                    eps,
                                )?;
                                for r in 0..rows {
                                    let row = layer.forward(&input.row(r)?, request.return_raw)?;
                                    result.push(json!({"values":row.values.read_output()?,"trace":row.trace}));
                                }
                            }
                            Ok(json!(result))
                        }
                        "metal-checkpoint-hc" | "metal-checkpoint-mlp" => {
                            let streams = request.input_view(gpu)?;
                            anyhow::ensure!(
                                streams.shape() == [1, 10240],
                                "expected one Flash row of four streams"
                            );
                            let path = std::path::Path::new(&request.path);
                            let checkpoint = checkpoint::inspect(path)?;
                            let eps = flash_checkpoint_epsilon(&checkpoint)?;
                            let mlp = request.op == "metal-checkpoint-mlp";
                            let prefix = if mlp {
                                format!("{}.mlp_hyper_connection", request.prefix)
                            } else {
                                request.prefix.clone()
                            };
                            let hc = mlxl3_native::gpu::FlashHc::load(
                                gpu,
                                &checkpoint,
                                &prefix,
                                eps,
                                !prefix.ends_with(".hyper_connection_mixer"),
                            )?;
                            let (read, output, route) = if mlp {
                                let mut experts = mlxl3_native::gpu::FlashMoeRow::load(
                                    gpu,
                                    &checkpoint,
                                    &format!("{}.mlp", request.prefix),
                                    request.slots.unwrap_or(10),
                                )?;
                                let (out, route, read) =
                                    hc.forward_moe(&streams, &mut experts, request.return_raw)?;
                                (read, Some(out.read_output()?), Some(route.indices))
                            } else {
                                (hc.read(&streams, request.return_raw)?, None, None)
                            };
                            Ok(
                                json!({"mixed":read.mixed.read_output()?,"gate":read.gate.as_ref().map(|g|g.read_output()).transpose()?,"output":output,"indices":route,"trace":read.trace}),
                            )
                        }
                        "metal-checkpoint-moe" => {
                            let input = request.input_view(gpu)?;
                            let [rows, width] = input.shape();
                            anyhow::ensure!(
                                (1..=8).contains(&rows) && width == 2560,
                                "expected one to eight Flash AR rows"
                            );
                            let checkpoint =
                                checkpoint::inspect(std::path::Path::new(&request.path))?;
                            let mut layer = mlxl3_native::gpu::FlashMoeRow::load(
                                gpu,
                                &checkpoint,
                                &request.prefix,
                                request.slots.unwrap_or(10),
                            )?;
                            let mut results = Vec::new();
                            for r in 0..rows {
                                let x = input.row(r)?;
                                let out = if request.return_raw {
                                    layer.trace(&x)?
                                } else {
                                    layer.submit(&x)?
                                };
                                results.push(json!({"indices":out.indices,"hits":out.hits,"read_bytes":out.read_bytes,"values":out.read()?,"switch":out.multiplier_trace()?}));
                            }
                            Ok(json!(results))
                        }
                        "metal-checkpoint-group" => {
                            let c = checkpoint::inspect(std::path::Path::new(&request.path))?;
                            let layer = mlxl3_native::gpu::ProjectionGroup::load(
                                gpu,
                                &c,
                                &request.prefixes,
                            )?;
                            let input = if request.x_layout.is_some() {
                                request.input_view(gpu)?
                            } else {
                                anyhow::ensure!(
                                    request.x.len() == layer.input_width() as usize,
                                    "invalid grouped input row length"
                                );
                                gpu.f16_storage(&request.x)?.matrix(
                                    mlxl3_native::tensor_layout::MatrixLayout {
                                        shape: [1, layer.input_width()],
                                        strides: [i64::from(layer.input_width()), 1],
                                        offset: 0,
                                    },
                                )?
                            };
                            let outputs = layer
                                .forward(&input)?
                                .iter()
                                .map(|v| v.read_output())
                                .collect::<Result<Vec<_>>>()?;
                            Ok(json!({"grouped":layer.is_grouped(),"outputs":outputs}))
                        }
                        "metal-checkpoint-switch" => {
                            let c = checkpoint::inspect(std::path::Path::new(&request.path))?;
                            let names = if c.model_type == "lfm2_moe" {
                                ["w1", "w3", "w2"]
                            } else {
                                ["gate_proj", "up_proj", "down_proj"]
                            };
                            let layer = mlxl3_native::gpu::MappedSwitchGlu::load(
                                gpu,
                                &c,
                                &request.prefix,
                                u32::try_from(request.key_heads)?,
                                u32::try_from(request.selected.len())?,
                                names,
                            )?;
                            anyhow::ensure!(
                                request.x.len() == layer.hidden_size() as usize,
                                "invalid mapped input row length"
                            );
                            let input = gpu.f16_storage(&request.x)?.matrix(
                                mlxl3_native::tensor_layout::MatrixLayout {
                                    shape: [1, layer.hidden_size()],
                                    strides: [i64::from(layer.hidden_size()), 1],
                                    offset: 0,
                                },
                            )?;
                            let count = u32::try_from(request.beta.len())?;
                            let scores = gpu.f16_storage(&request.beta)?.matrix(
                                mlxl3_native::tensor_layout::MatrixLayout {
                                    shape: [1, count],
                                    strides: [i64::from(count), 1],
                                    offset: 0,
                                },
                            )?;
                            Ok(json!(
                                layer
                                    .forward(&input, &request.selected, &scores)?
                                    .read_output()?
                            ))
                        }
                        "metal-qwen-gdn" => {
                            let c = checkpoint::inspect(std::path::Path::new(&request.path))?;
                            let mut layer = mlxl3_native::gpu::QwenGdn::from_checkpoint(
                                gpu,
                                &c,
                                &request.prefix,
                            )?;
                            let width = layer.hidden_size();
                            anyhow::ensure!(
                                !request.x.is_empty()
                                    && request.x.len().is_multiple_of(width as usize),
                                "invalid Qwen GDN input rows"
                            );
                            let mut output = Vec::new();
                            for row in request.x.chunks_exact(width as usize) {
                                let input = gpu.f16_storage(row)?.matrix(
                                    mlxl3_native::tensor_layout::MatrixLayout {
                                        shape: [1, width],
                                        strides: [i64::from(width), 1],
                                        offset: 0,
                                    },
                                )?;
                                output.extend(layer.forward(&input)?.read_output()?);
                            }
                            let states = layer.states()?.context("missing Qwen GDN state")?;
                            Ok(
                                json!({"output":output,"conv_state":states.conv_state,"state":states.state}),
                            )
                        }
                        "metal-checkpoint-router" => {
                            let input = request.input_view(gpu)?;
                            let checkpoint =
                                checkpoint::inspect(std::path::Path::new(&request.path))?;
                            let gate = mlxl3_native::gpu::FlashGate::load(
                                gpu,
                                &checkpoint,
                                &request.prefix,
                            )?;
                            if request.return_raw {
                                Ok(json!(gate.trace(&input)?))
                            } else {
                                let (indices, scores) = gate.submit(&input)?.read()?;
                                Ok(json!({"indices":indices,"scores":scores}))
                            }
                        }
                        "metal-router" => {
                            let input = request.input_view(gpu)?;
                            let (indices, scores) = gpu
                                .route(&input, u32::try_from(request.k)?, request.normalize)?
                                .read()?;
                            Ok(json!({"indices":indices,"scores":scores}))
                        }
                        "metal-prepare-view" => {
                            let x = request.input_view(gpu)?;
                            let scales = gpu.f16_storage(&request.data)?.matrix(
                                request
                                    .scale_layout
                                    .context("missing scale matrix layout")?,
                            )?;
                            Ok(json!(gpu.prepare_view(&x, &scales)?))
                        }
                        "metal-swiglu" => Ok(json!(gpu.swiglu(&request.x, &request.data)?.read()?)),
                        "metal-checkpoint-linear"
                        | "metal-checkpoint-expert"
                        | "metal-exl3-linear" => {
                            let checkpoint =
                                checkpoint::inspect(std::path::Path::new(&request.path))?;
                            let view = if request.x_layout.is_some() || request.x_f32.is_some() {
                                Some(request.input_view(gpu)?)
                            } else {
                                None
                            };
                            let flight = if request.op == "metal-checkpoint-expert" {
                                let expert = mlxl3_native::gpu::FlashExpert::load(
                                    gpu,
                                    &checkpoint,
                                    &request.prefix,
                                )?;
                                match &view {
                                    Some(x) => expert.submit_fp16_view(x)?,
                                    None => expert.submit(&request.x)?,
                                }
                            } else {
                                let load = if request.op == "metal-exl3-linear" {
                                    mlxl3_native::gpu::FlashProjection::load_exl3
                                } else {
                                    mlxl3_native::gpu::FlashProjection::load
                                };
                                let layer = load(gpu, &checkpoint, &request.prefix)?;
                                match &view {
                                    Some(x) => layer.submit_fp16_view(x)?,
                                    None => layer.submit(&request.x)?,
                                }
                            };
                            Ok(json!(flight.read()?))
                        }
                        "metal-prepare128" => Ok(json!(gpu.prepare128(
                            &request.x,
                            request.cols,
                            &request.data
                        )?)),
                        "metal-pack" => Ok(json!(gpu.pack(&request.data, request.k)?)),
                        "metal-decode" => {
                            let (states, values) = gpu.decode(&request.data, request.k, cb)?;
                            Ok(json!({ "states": states, "values": values }))
                        }
                        _ => Ok(json!(gpu.qmv_inner(
                            &request.x,
                            &request.data,
                            request.cols,
                            request.k,
                            cb
                        )?)),
                    }
                }
                _ => bail!("unknown or unavailable codec operation {}", request.op),
            }
        })();
        let value = match result {
            Ok(data) => json!({"data": data}),
            Err(e) => json!({"error": format!("{e:#}")}),
        };
        serde_json::to_writer(&mut output, &value)?;
        writeln!(output)?;
        output.flush()?;
    }
    Ok(())
}

fn main() -> std::process::ExitCode {
    // Q27D-23: TensorOps verification for the dense Qwen3.8-27B on M5 (not bit-exact, quality
    // measured in Q27D-16/18); `MLXL3_DENSE_TENSOR_ROWS=0` restores the exact kernels.
    #[cfg(feature = "mlx")]
    mlxl3_native::linear::set_dense_tensor_product_default(3);
    if let Err(error) = run(Cli::parse()) {
        eprintln!("Error: {error:#}");
        std::process::ExitCode::from(2)
    } else {
        std::process::ExitCode::SUCCESS
    }
}

fn valid_private_key(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes.len() <= 512 && bytes.iter().all(u8::is_ascii_graphic)
}

fn read_server_key(reader: impl Read) -> Result<String> {
    let mut bytes = Vec::new();
    reader.take(514).read_to_end(&mut bytes)?;
    // Reading one extra byte distinguishes an oversized secret from a valid 512-byte key.
    let key = bytes.strip_suffix(b"\n").unwrap_or(&bytes);
    if !valid_private_key(key) {
        bail!("invalid private server key on standard input");
    }
    Ok(String::from_utf8(key.to_vec())?)
}

#[cfg(kani)]
#[kani::proof]
#[kani::unwind(5)]
fn private_server_key_rejects_empty_and_control_bytes() {
    let bytes: [u8; 3] = kani::any();
    let len: usize = kani::any();
    kani::assume(len <= 3);
    let key = &bytes[..len];
    let valid = valid_private_key(key);
    assert_eq!(
        valid,
        len > 0 && (0..len).all(|i| (33..127).contains(&bytes[i]))
    );
    kani::cover!(len == 0);
    kani::cover!(valid && len == 3);
    kani::cover!(!valid && len == 3);
}

#[cfg(test)]
#[test]
fn private_server_key_validates_size_and_never_echoes_the_secret() {
    assert_eq!(read_server_key(&b"abc-123\n"[..]).unwrap(), "abc-123");
    assert_eq!(read_server_key(&[b'a'; 512][..]).unwrap().len(), 512);
    let mut line = vec![b'a'; 512];
    line.push(b'\n');
    assert_eq!(read_server_key(line.as_slice()).unwrap().len(), 512);
    for bad in [&b""[..], b"\n", b"sensitive\0", b"two\nlines", &[b'x'; 513]] {
        assert_eq!(
            read_server_key(bad).unwrap_err().to_string(),
            "invalid private server key on standard input"
        );
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        #[cfg(all(
            feature = "hybrid-dispatch",
            not(any(feature = "mlx", all(target_os = "macos", feature = "direct-metal")))
        ))]
        Command::Forward { .. } => bail!("forward requires an inference backend"),
        Command::QuantizeWorker {
            root,
            k,
            codebook,
            max_scratch_bytes,
        } => {
            #[cfg(all(target_os = "macos", feature = "direct-metal"))]
            mlxl3_native::quantize::run_worker(&root, k, codebook, max_scratch_bytes)?;
            #[cfg(not(all(target_os = "macos", feature = "direct-metal")))]
            {
                let _ = (root, k, codebook, max_scratch_bytes);
                bail!("tile worker requires native Metal on macOS");
            }
        }
        Command::QuantizeTiles {
            input,
            output,
            k,
            codebook,
            max_scratch_bytes,
        } => {
            #[cfg(all(target_os = "macos", feature = "direct-metal"))]
            mlxl3_native::quantize::run_tiles(&input, &output, k, codebook, max_scratch_bytes)?;
            #[cfg(not(all(target_os = "macos", feature = "direct-metal")))]
            {
                let _ = (input, output, k, codebook, max_scratch_bytes);
                bail!("tile quantization requires native Metal on macOS");
            }
        }
        Command::ConvertMatmul {
            left,
            right,
            output,
            m,
            n,
            k,
            full_fp32,
        } => {
            #[cfg(all(target_os = "macos", feature = "direct-metal"))]
            mlxl3_native::convert_matmul::run(&left, &right, &output, m, n, k, full_fp32)?;
            #[cfg(not(all(target_os = "macos", feature = "direct-metal")))]
            {
                let _ = (left, right, output, m, n, k, full_fp32);
                bail!("converter matrix multiplication requires native Metal on macOS");
            }
        }
        Command::RuntimeInfo => {
            println!(
                "{}",
                json!({
                    "version": env!("CARGO_PKG_VERSION"),
                    "bridge_protocol": 1,
                    "architecture": std::env::consts::ARCH,
                    "mlx_enabled": cfg!(feature = "mlx"),
                    "backend": if cfg!(feature = "mlx") { "mlx" } else if cfg!(all(target_os = "macos", feature = "direct-metal")) { "metal" } else { "cpu" },
                    "chat_enabled": cfg!(feature = "chat"),
                    "commit": env!("MLXL3_BUILD_REVISION"),
                    "profile": env!("MLXL3_BUILD_PROFILE")
                })
            );
        }
        Command::List { json: as_json } => {
            let path = cli
                .registry
                .map(Ok)
                .unwrap_or_else(registry::default_path)?;
            let entries = registry::load(&path)?;
            let mut models = Vec::new();
            for (name, entry) in entries {
                let mut value = serde_json::to_value(&entry)?;
                value["name"] = json!(name);
                value["size"] = json!(human_size(entry.size_bytes));
                models.push(value);
            }
            models.sort_by_key(|v| v["name"].as_str().unwrap_or("").to_lowercase());
            if as_json {
                println!("{}", serde_json::to_string(&models)?);
            } else {
                println!("NAME\tFORMAT\tBPW\tSIZE\tMODEL TYPE\tPATH");
                for value in models {
                    println!(
                        "{}\t{}\t{}\t{}\t{}\t{}",
                        value["name"].as_str().unwrap(),
                        value["format"].as_str().unwrap(),
                        value["bits"],
                        value["size"].as_str().unwrap(),
                        value["model_type"].as_str().unwrap(),
                        value["path"].as_str().unwrap()
                    );
                }
            }
        }
        Command::Inspect { path } => println!(
            "{}",
            serde_json::to_string_pretty(&checkpoint::inspect(&registry::expand_home(&path)?)?)?
        ),
        Command::Register { name, path, force } => {
            let registry_path = cli
                .registry
                .map(Ok)
                .unwrap_or_else(registry::default_path)?;
            let entry = registry::register(&registry_path, &name, &path, force)?;
            println!("Registered {name} ({})", human_size(entry.size_bytes));
        }
        Command::Remove {
            name,
            expected_path,
        } => {
            let path = cli
                .registry
                .map(Ok)
                .unwrap_or_else(registry::default_path)?;
            registry::remove(&path, &name, expected_path.as_deref())?;
            println!("Removed {name} from registry; model files retained.");
        }
        Command::Codec => codec_loop()?,
        Command::Split { prompt } => {
            let chunks: Vec<String> = serde_json::from_reader(io::stdin().lock())?;
            let mut splitter = ThinkingSplitter::new(&prompt);
            let mut result = Vec::new();
            for chunk in chunks {
                result.extend(splitter.feed(&chunk));
            }
            result.extend(splitter.finish());
            println!("{}", serde_json::to_string(&result)?);
        }
        #[cfg(all(
            not(feature = "mlx"),
            feature = "chat",
            target_os = "macos",
            feature = "direct-metal"
        ))]
        Command::Forward {
            model,
            tokens,
            states,
            batch,
        } => {
            anyhow::ensure!(
                !tokens.is_empty()
                    && tokens.len() <= mlxl3_native::flash_support::FLASH_MAX_CONTEXT as usize
                    && tokens.iter().all(|&t| t < 248320),
                "invalid native Flash tokens or context"
            );
            let gpu = mlxl3_native::gpu::Metal::new()?;
            let c = checkpoint::inspect(&model)?;
            let mut target = NativeChatModel::load(&gpu, &c, tokens.len() as u32)?;
            target.admit_request(tokens.len())?;
            let mut out = io::stdout().lock();
            #[derive(serde::Serialize)]
            struct Row {
                token: u32,
                logits: Vec<u16>,
                states: Vec<mlxl3_native::gpu::FlashTraceTensor>,
            }
            let mut emit = |row: Row| -> Result<()> {
                serde_json::to_writer(&mut out, &row)?;
                writeln!(out)?;
                out.flush()?;
                Ok(())
            };
            if batch {
                emit(Row {
                    token: *tokens.last().context("empty Flash input")?,
                    logits: target.prefill(&tokens)?,
                    states: Vec::new(),
                })?;
            } else {
                for token in tokens {
                    let (logits, trace) = if states {
                        target.forward_trace(token)?
                    } else {
                        (target.forward(token)?, Vec::new())
                    };
                    emit(Row {
                        token,
                        logits,
                        states: trace,
                    })?;
                }
            }
        }
        #[cfg(all(feature = "mlx", feature = "chat"))]
        Command::Forward {
            model,
            tokens,
            states,
            batch,
        } => {
            let mut model = NativeChatModel::load(&model)?;
            let mut out = io::stdout().lock();
            if batch {
                let token = *tokens.last().context("empty token batch")?;
                let logits = match &mut model {
                    NativeChatModel::Ling(model) => model.forward_tokens(&tokens)?,
                    _ => model.forward_many(&tokens)?,
                };
                let arrays = match (&model, states) {
                    (NativeChatModel::Lfm3(model), true) => model
                        .state_arrays()
                        .into_iter()
                        .map(|(name, array)| Ok((name, array.try_clone()?)))
                        .collect::<Result<Vec<_>>>()?,
                    (NativeChatModel::Lfm2(model), true) => model
                        .state_arrays()
                        .into_iter()
                        .map(|(name, array)| Ok((name, array.try_clone()?)))
                        .collect::<Result<Vec<_>>>()?,
                    _ => Vec::new(),
                };
                let state = arrays
                    .into_iter()
                    .map(|(name, array)| -> Result<_> {
                        Ok(json!({"name": name, "shape": array.shape(), "data": array.to_bytes()?}))
                    })
                    .collect::<Result<Vec<_>>>()?;
                serde_json::to_writer(
                    &mut out,
                    &json!({"token":token,"logits":logits.to_f16_bits()?,"states":state}),
                )?;
                writeln!(out)?;
                out.flush()?;
                return Ok(());
            }
            for token in tokens {
                let (logits, arrays) = match (&mut model, states) {
                    (NativeChatModel::Flash(model), true) => model.forward_trace(token)?,
                    (NativeChatModel::Qwen(model), true) => {
                        let logits = model.forward(token)?;
                        (logits, model.state_trace()?)
                    }
                    (NativeChatModel::Gemma4(model), true) => model.forward_trace(token)?,
                    (NativeChatModel::Lfm3(model), true) => {
                        let logits = model.forward(&[token])?;
                        let arrays = model
                            .state_arrays()
                            .into_iter()
                            .map(|(name, array)| Ok((name, array.try_clone()?)))
                            .collect::<Result<Vec<_>>>()?;
                        (logits, arrays)
                    }
                    (NativeChatModel::Lfm2(model), true) => {
                        let logits = model.forward(&[token])?;
                        let arrays = model
                            .state_arrays()
                            .into_iter()
                            .map(|(name, array)| Ok((name, array.try_clone()?)))
                            .collect::<Result<Vec<_>>>()?;
                        (logits, arrays)
                    }
                    (NativeChatModel::Ling(model), true) => {
                        let logits = model.forward(token)?;
                        let arrays = model
                            .state_arrays()
                            .into_iter()
                            .map(|(name, array)| Ok((name, array.try_clone()?)))
                            .collect::<Result<Vec<_>>>()?;
                        (logits, arrays)
                    }
                    _ => (model.forward(token)?, Vec::new()),
                };
                let state = arrays
                    .into_iter()
                    .map(|(name, array)| -> Result<_> {
                        Ok(json!({"name": name, "shape": array.shape(), "data": array.to_bytes()?}))
                    })
                    .collect::<Result<Vec<_>>>()?;
                serde_json::to_writer(
                    &mut out,
                    &json!({"token":token,"logits":logits.to_f16_bits()?,"states":state}),
                )?;
                writeln!(out)?;
                out.flush()?;
            }
        }
        Command::Run {
            model,
            prompt,
            max_tokens,
            images,
        } => {
            #[cfg(all(
                feature = "chat",
                any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
            ))]
            {
                let model_path = resolve_model_path(cli.registry.as_deref(), &model)?;
                native_chat(&model_path, prompt, max_tokens, images)?;
            }
            #[cfg(not(all(
                feature = "chat",
                any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
            )))]
            {
                let _ = (prompt, max_tokens, images);
                bail!(
                    "native execution for {model:?} requires a build with --features mlx,chat; the installed app is separate"
                );
            }
        }
        Command::Bridge {
            model,
            context_length,
        } => {
            #[cfg(all(feature = "mlx", feature = "chat"))]
            native_bridge(cli.registry.as_deref(), &model, context_length)?;
            #[cfg(all(
                not(feature = "mlx"),
                feature = "chat",
                target_os = "macos",
                feature = "direct-metal"
            ))]
            direct_bridge::run(cli.registry.as_deref(), &model, context_length)?;
            #[cfg(not(all(
                feature = "chat",
                any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
            )))]
            {
                let _ = (model, context_length);
                bail!("the Desktop bridge requires chat and an inference backend");
            }
        }
        Command::Decide {
            model,
            request,
            trace,
        } => {
            #[cfg(all(feature = "mlx", feature = "chat"))]
            {
                let path = cli::resolve_model_path(cli.registry.as_deref(), &model)?;
                let text = match request {
                    Some(file) => std::fs::read_to_string(&file)
                        .with_context(|| format!("reading {}", file.display()))?,
                    None => {
                        let mut text = String::new();
                        io::stdin().read_to_string(&mut text)?;
                        text
                    }
                };
                let value: mlxl3_native::d1::Json =
                    serde_json::from_str(&text).context("the request is not JSON")?;
                if let Some(folder) = &trace {
                    std::fs::create_dir_all(folder)?;
                }
                let mut engine = mlxl3_native::d1::runtime::Engine::load(&path, &model)?;
                let answer = engine.decide(&value, true, trace.as_deref())?;
                println!("{}", serde_json::to_string(&answer)?);
            }
            #[cfg(not(all(feature = "mlx", feature = "chat")))]
            {
                let _ = (request, trace);
                bail!("decisions with {model:?} require a build with --features mlx,chat");
            }
        }
        Command::DecideBridge { model } => {
            #[cfg(all(feature = "mlx", feature = "chat"))]
            {
                let path = cli::resolve_model_path(cli.registry.as_deref(), &model)?;
                mlxl3_native::d1_server::run_bridge(
                    &path,
                    &model,
                    io::stdin().lock(),
                    io::stdout().lock(),
                )?;
            }
            #[cfg(not(all(feature = "mlx", feature = "chat")))]
            bail!("the decision bridge for {model:?} requires a build with --features mlx,chat");
        }
        Command::DecideServe {
            model,
            host,
            port,
            api_key,
            api_key_stdin,
        } => {
            let api_key = if api_key_stdin {
                read_server_key(io::stdin().lock())?
            } else {
                api_key
            };
            #[cfg(all(feature = "mlx", feature = "chat"))]
            {
                let path = cli::resolve_model_path(cli.registry.as_deref(), &model)?;
                mlxl3_native::d1_server::serve(path, model, &host, port, api_key)?;
            }
            #[cfg(not(all(feature = "mlx", feature = "chat")))]
            {
                let _ = (host, port, api_key);
                bail!("the decision API for {model:?} requires a build with --features mlx,chat");
            }
        }
        Command::Serve {
            model,
            host,
            port,
            api_key,
            api_key_stdin,
            context_length,
        } => {
            let api_key = if api_key_stdin {
                read_server_key(io::stdin().lock())?
            } else {
                api_key
            };
            #[cfg(all(
                feature = "chat",
                any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
            ))]
            {
                let runtime = tokio::runtime::Builder::new_multi_thread()
                    .enable_all()
                    .build()?;
                runtime.block_on(openai::serve(
                    cli.registry.as_deref(),
                    &model,
                    &host,
                    port,
                    &api_key,
                    context_length,
                ))?;
            }
            #[cfg(not(all(
                feature = "chat",
                any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
            )))]
            {
                let _ = (model, host, port, api_key, context_length);
                bail!("the OpenAI API requires chat and an inference backend");
            }
        }
        Command::Hub {
            action,
            query,
            revision,
            folder,
            limit,
        } => {
            let registry_path = cli
                .registry
                .map(Ok)
                .unwrap_or_else(registry::default_path)?;
            let progress = |completed, total, phase: &str| {
                println!(
                    "{}",
                    json!({"type":"progress", "completed":completed, "total":total, "phase":phase})
                );
                let _ = io::stdout().flush();
            };
            match action {
                HubAction::Search => println!(
                    "{}",
                    serde_json::to_string(&mlxl3_native::hub::search(&query, limit)?)?
                ),
                HubAction::Details => println!(
                    "{}",
                    serde_json::to_string(&mlxl3_native::hub::details(
                        &query,
                        revision.as_deref()
                    )?)?
                ),
                HubAction::Download => {
                    let (name, entry) = mlxl3_native::hub::download_with_phase(
                        &query,
                        revision.as_deref().context("--revision is required")?,
                        &folder,
                        &registry_path,
                        progress,
                    )?;
                    println!(
                        "{}",
                        json!({"type":"installed", "model":model_payload(&name, &entry)?})
                    );
                }
                HubAction::Pending => println!(
                    "{}",
                    serde_json::to_string(&mlxl3_native::hub::pending_downloads()?)?
                ),
                HubAction::Resume => {
                    let (name, entry) =
                        mlxl3_native::hub::resume_with_phase(&query, &registry_path, progress)?;
                    println!(
                        "{}",
                        json!({"type":"installed", "model":model_payload(&name, &entry)?})
                    );
                }
                HubAction::Discard => {
                    mlxl3_native::hub::discard(&query)?;
                    println!("{{}}");
                }
                HubAction::Auth => {
                    match query.as_str() {
                        "login" => {
                            let mut token = String::new();
                            io::stdin().take(4096).read_to_string(&mut token)?;
                            mlxl3_native::hub::login(token.trim())?;
                        }
                        "logout" => mlxl3_native::hub::logout()?,
                        _ => bail!("expected login or logout"),
                    }
                    println!("{{}}");
                }
            }
        }
        Command::DflashDraft { inspect, target } => {
            use mlxl3_native::dflash::{Family, ManagedDraft};
            let expected = target
                .as_deref()
                .map(ManagedDraft::from_target)
                .transpose()?;
            let path = if let Some(path) = inspect {
                if let Some(expected) = expected {
                    expected.inspect(&path)?
                } else {
                    mlxl3_native::dflash::inspect(path)?.directory
                }
            } else {
                let progress = |completed, total| {
                    println!(
                        "{}",
                        json!({"type":"progress", "completed":completed, "total":total})
                    );
                    let _ = io::stdout().flush();
                };
                match expected.unwrap_or(ManagedDraft::Packed(Family::Qwen36Moe)) {
                    ManagedDraft::Packed(family) => {
                        mlxl3_native::hub::download_dflash(family, progress)?
                    }
                    ManagedDraft::FlashNext => mlxl3_native::hub::download_flash_dflash(progress)?,
                    ManagedDraft::Inkling2 => mlxl3_native::hub::download_inkling_dflash(progress)?,
                }
            };
            println!("{}", json!({"type":"installed", "path":path}));
        }
        Command::MtpHead { inspect, target } => {
            let target = target
                .as_deref()
                .map(mlxl3_native::mtp::target_layout)
                .transpose()?;
            let path = if let Some(path) = inspect {
                let (checkpoint, layout) = mlxl3_native::mtp::inspect(&path)?;
                if let Some((expected, _)) = &target {
                    anyhow::ensure!(
                        &layout == expected,
                        "MTP head does not match the target model dimensions"
                    );
                }
                checkpoint.path
            } else {
                let kind = if let Some((layout, layers)) = &target {
                    layout
                        .managed_head(*layers)
                        .context("No managed MTP head for this target")?
                } else {
                    mlxl3_native::mtp::ManagedHead::Qwen36Moe
                };
                mlxl3_native::hub::download_mtp(kind, |completed, total| {
                    println!(
                        "{}",
                        json!({"type":"progress", "completed":completed, "total":total})
                    );
                    let _ = io::stdout().flush();
                })?
            };
            println!("{}", json!({"type":"installed", "path":path}));
        }
        Command::Mcp { action } => {
            let registry_path = cli
                .registry
                .map(Ok)
                .unwrap_or_else(registry::default_path)?;
            match action {
                McpAction::List { json: as_json } => {
                    let servers = mlxl3_native::mcp::list_servers(&registry_path)?;
                    if as_json {
                        println!("{}", serde_json::to_string(&servers)?);
                    } else {
                        for (name, server) in servers {
                            println!(
                                "{name}\t{}\t{}",
                                if server.enabled {
                                    "enabled"
                                } else {
                                    "disabled"
                                },
                                server.url.as_deref().unwrap_or(&server.command)
                            );
                        }
                    }
                }
                McpAction::Check { json: as_json } => {
                    let mut manager = mlxl3_native::mcp::Manager::disabled();
                    manager.set_enabled(&registry_path, true, true);
                    let result = json!({
                        "servers":manager.server_count(),
                        "tools":manager.chat_tools(),
                        "errors":manager.errors,
                    });
                    if as_json {
                        println!("{result}");
                    } else {
                        println!(
                            "{} server(s), {} tool(s)",
                            manager.server_count(),
                            manager.tools.len()
                        );
                        for name in manager.tools.keys() {
                            println!("- {name}");
                        }
                        for (name, error) in &manager.errors {
                            eprintln!("{name}: {error}");
                        }
                    }
                }
                McpAction::Config => println!(
                    "{}",
                    mlxl3_native::mcp::config_path(&registry_path).display()
                ),
                McpAction::Add {
                    name,
                    command,
                    args,
                } => {
                    mlxl3_native::mcp::add_server(&registry_path, &name, &command, args)?;
                    println!("Added {name}");
                }
                McpAction::Remove { name } => {
                    mlxl3_native::mcp::remove_server(&registry_path, &name)?;
                    println!("Removed {name}");
                }
            }
        }
    }
    Ok(())
}

fn human_size(size: u64) -> String {
    let mut value = size as f64;
    for unit in ["B", "KB", "MB", "GB", "TB"] {
        if value < 1000.0 || unit == "TB" {
            return if unit == "B" {
                format!("{size} B")
            } else {
                format!("{value:.1} {unit}")
            };
        }
        value /= 1000.0;
    }
    unreachable!()
}

fn model_payload(name: &str, entry: &registry::ModelEntry) -> Result<Value> {
    let mut value = serde_json::to_value(entry)?;
    value["name"] = json!(name);
    value["size"] = json!(human_size(entry.size_bytes));
    Ok(value)
}

#[cfg(all(feature = "mlx", feature = "chat"))]
enum NativeChatModel {
    Gemma4(mlxl3_native::gemma4::Gemma4),
    /// Inkling 2 Mini, its DSpark drafter when loaded, and the checkpoint directory of a drafter that can
    /// be loaded on demand (`mtp.safetensors` present and DSpark not disabled).
    Inkling(
        Box<mlxl3_native::inkling2::Inkling2>,
        Option<Box<mlxl3_native::inkling2_draft::InklingDrafter>>,
        Option<std::path::PathBuf>,
    ),
    Lfm2(mlxl3_native::lfm2::Lfm2),
    Lfm3(Box<mlxl3_native::lfm3::Lfm3>),
    Ling(mlxl3_native::ling::Ling),
    Qwen(mlxl3_native::qwen35::Qwen35Moe),
    Flash(Box<mlxl3_native::qwen4::FlashNext>),
}

/// DSPARK-INK2M-P3: draw DSpark proposals from the draft law. Only for plain softmax sampling (the GPU
/// draw has no top-k, top-p or penalties); opt-in with `MLXL3_INKLING_DSPARK_SAMPLED=1` while measured.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn inkling_sampled_drafts(sampler: &mlxl3_native::sampling::Sampler) -> bool {
    !sampler.is_greedy()
        && sampler.top_k == 0
        && sampler.top_p >= 1.0
        && sampler.repetition_penalty == 1.0
        && sampler.presence_penalty == 0.0
        && std::env::var("MLXL3_INKLING_DSPARK_SAMPLED").as_deref() == Ok("1")
}

/// Seed of the draft noise, derived from the request's random state by splitmix64 so the draft draws
/// do not replay the verification's xorshift sequence.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn draft_noise_seed(random: u64) -> u64 {
    let mut z = random ^ 0x9e37_79b9_7f4a_7c15;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// SPEED-INK2M-P14: GPU token choice for Inkling decoding and whether it runs one step ahead, from the
/// sampler and `MLXL3_INKLING_GPU_DRAW` / `MLXL3_INKLING_PIPELINE` ("0" turns either off; without the
/// GPU draw nothing runs ahead). `None` keeps the CPU sampler.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn inkling_device_draw(
    sampler: &mlxl3_native::sampling::Sampler,
    draw: Option<&str>,
    pipeline: Option<&str>,
) -> Option<(mlxl3_native::inkling2::DrawRule, bool)> {
    if draw == Some("0") {
        return None;
    }
    mlxl3_native::inkling2::DrawRule::for_sampler(sampler).map(|rule| (rule, pipeline != Some("0")))
}

/// Seed of the GPU draws, derived from the request's random state apart from the draft noise and the
/// CPU acceptance draws.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn inkling_draw_seeds(random: u64) -> mlxl3_native::inkling2::DrawSeeds {
    mlxl3_native::inkling2::DrawSeeds::new(draft_noise_seed(
        random.rotate_left(29) ^ 0x5851_f42d_4c95_7f2d,
    ))
}

/// The loaded model is Inkling 2 with a DSpark drafter, loaded or loadable on demand
/// (`ready.dspark_supported`).
#[cfg(all(feature = "mlx", feature = "chat"))]
fn inkling_has_drafter(model: &NativeChatModel) -> bool {
    matches!(
        model,
        NativeChatModel::Inkling(_, Some(_), _) | NativeChatModel::Inkling(_, _, Some(_))
    ) || (matches!(model, NativeChatModel::Inkling(..)) && inkling_dflash2_source().is_some())
}

/// The directory of the loaded Inkling 2 model (set when it loads), where a DFlash2 drafter is looked for.
#[cfg(all(feature = "mlx", feature = "chat"))]
static INKLING_MODEL_DIR: std::sync::Mutex<Option<std::path::PathBuf>> =
    std::sync::Mutex::new(None);

/// DFLASH2-2026-11-24-INK2M-P0: the DFlash2 drafter directory that replaces DSpark for Inkling 2 Mini (its
/// `config.json` present): `MLXL3_INKLING_DFLASH2` when set, else (DECISION-2026-11-25-INK2M-DFLASH2-ON)
/// `dflash2/` inside the model directory, `Inkling-2-Mini-DFlash2/` beside it, or the managed download.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn inkling_dflash2_source() -> Option<std::path::PathBuf> {
    let model = INKLING_MODEL_DIR
        .lock()
        .ok()
        .and_then(|directory| directory.clone());
    inkling_dflash2_source_in(std::env::var_os("MLXL3_INKLING_DFLASH2"), model.as_deref())
        // FEAT-2026-11-25-INK2M-DFLASH2-APP: the drafter the app downloads (`dflash-draft --target`)
        .or_else(|| {
            std::env::var_os("MLXL3_INKLING_DFLASH2")
                .is_none()
                .then(mlxl3_native::hub::installed_inkling_dflash)
                .flatten()
        })
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn inkling_dflash2_source_in(
    flag: Option<std::ffi::OsString>,
    model: Option<&std::path::Path>,
) -> Option<std::path::PathBuf> {
    let valid = |dir: &std::path::PathBuf| dir.join("config.json").is_file();
    if let Some(flag) = flag {
        // an explicit directory is the only one tried (an empty value turns the lookup off)
        return Some(std::path::PathBuf::from(flag)).filter(valid);
    }
    let model = model?;
    [
        Some(model.join("dflash2")),
        model
            .parent()
            .map(|parent| parent.join("Inkling-2-Mini-DFlash2")),
    ]
    .into_iter()
    .flatten()
    .find(valid)
}

/// DECISION-2026-11-25-INK2M-DFLASH2-ON: the speculative choice of a bridge request: its `dspark` field when
/// present, else on when a DFlash2 drafter is available (`dflash2`) unless `MLXL3_INKLING_DFLASH2_DEFAULT`
/// (`flag`) is `0`. The official DSpark drafter stays opt-in (DECISION-2026-11-22-INK2M-DSPARK-OFF).
#[cfg(all(feature = "mlx", feature = "chat"))]
fn inkling_speculation_requested(
    requested: Option<bool>,
    dflash2: bool,
    flag: Option<&str>,
) -> bool {
    requested.unwrap_or(dflash2 && flag != Some("0"))
}

/// A boolean field that may be omitted (`None`) but not null.
#[cfg(feature = "chat")]
#[cfg_attr(not(feature = "mlx"), allow(dead_code))]
fn present_bool<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<bool>, D::Error> {
    bool::deserialize(deserializer).map(Some)
}

/// SPEED-INK2M-P6a: Inkling 2 is warmed up at load unless `MLXL3_INKLING_WARMUP=0`.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn inkling_warmup_enabled() -> bool {
    std::env::var("MLXL3_INKLING_WARMUP").as_deref() != Ok("0")
}

/// DSPARK-INK2M-P10c: the adaptive width policy of a DSpark session. Sampled rows on the 8-bit dense
/// target use the block costs measured on it and spaced narrow probes; greedy rows (whose acceptances come
/// in bursts the measured table handled worse, P10b), the FP16 dense target, or `policy` = `p4c`
/// (`MLXL3_INKLING_DSPARK_POLICY`, kept while measured) keep the P4c linear FP16 costs.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn inkling_width_policy(
    policy: Option<&str>,
    dense_bits: i32,
    greedy: bool,
    width: usize,
) -> Result<mlxl3_native::inkling2_draft::WidthPolicy> {
    use mlxl3_native::inkling2_draft::WidthPolicy;
    match policy {
        Some("p4c") => Ok(WidthPolicy::new(width)),
        None | Some("") | Some("p10") if greedy || dense_bits >= 16 => Ok(WidthPolicy::new(width)),
        None | Some("") | Some("p10") => {
            WidthPolicy::measured(&WidthPolicy::SAMPLED_8BIT_COSTS, width)
        }
        Some(other) => bail!("MLXL3_INKLING_DSPARK_POLICY must be p4c or p10, not {other:?}"),
    }
}

/// DSpark for Inkling 2 Mini is on unless `MLXL3_INKLING_DSPARK=0`.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn inkling_dspark_enabled() -> bool {
    std::env::var("MLXL3_INKLING_DSPARK").as_deref() != Ok("0")
}

#[cfg(all(feature = "mlx", feature = "chat"))]
impl NativeChatModel {
    fn load(path: &std::path::Path) -> Result<Self> {
        let checkpoint = checkpoint::inspect(path)?;
        Self::from_checkpoint(&checkpoint)
    }

    fn from_checkpoint(checkpoint: &checkpoint::Checkpoint) -> Result<Self> {
        match checkpoint.model_type.as_str() {
            "qwen4_exp" => Ok(Self::Flash(Box::new(
                mlxl3_native::qwen4::FlashNext::from_checkpoint(checkpoint)?,
            ))),
            "gemma4" => Ok(Self::Gemma4(mlxl3_native::gemma4::Gemma4::from_checkpoint(
                checkpoint,
            )?)),
            "bailing_hybrid" => Ok(Self::Ling(mlxl3_native::ling::Ling::from_checkpoint(
                checkpoint,
            )?)),
            "inkling2_loop" => {
                let mut target = mlxl3_native::inkling2::Inkling2::from_checkpoint(checkpoint)?;
                // DECISION-2026-11-22-INK2M-DSPARK-OFF: DSpark is off by default, so its drafter is
                // loaded by the first request that asks for it (`prepare_inkling_drafter`)
                let drafter_source = (checkpoint.path.join("mtp.safetensors").is_file()
                    && inkling_dspark_enabled())
                .then(|| checkpoint.path.clone());
                if let Ok(mut directory) = INKLING_MODEL_DIR.lock() {
                    *directory = Some(checkpoint.path.clone());
                }
                // SPEED-INK2M-P6a: pay the Metal kernel compilation while loading, not on the first
                // request (`MLXL3_INKLING_WARMUP=0` skips it)
                if inkling_warmup_enabled() {
                    mlxl3_native::inkling2_draft::warm_up(&mut target, None)
                        .context("warming up Inkling 2")?;
                }
                Ok(Self::Inkling(Box::new(target), None, drafter_source))
            }
            "lfm3" | "lfm3_vl" => Ok(Self::Lfm3(Box::new(
                mlxl3_native::lfm3::Lfm3::from_checkpoint(checkpoint)?,
            ))),
            "lfm2" | "lfm2_moe" => Ok(Self::Lfm2(mlxl3_native::lfm2::Lfm2::from_checkpoint(
                checkpoint,
            )?)),
            "qwen3_5" | "qwen3_5_moe" => Ok(Self::Qwen(
                mlxl3_native::qwen35::Qwen35Moe::from_checkpoint(checkpoint)?,
            )),
            other => bail!("native chat does not support model type {other:?}"),
        }
    }

    fn forward(&mut self, token: u32) -> Result<mlxl3_native::array::Array> {
        match self {
            Self::Gemma4(model) => model.forward(token),
            Self::Inkling(model, ..) => model.forward(token),
            Self::Lfm2(model) => model.forward(&[token]),
            Self::Lfm3(model) => model.forward(&[token]),
            Self::Ling(model) => model.forward(token),
            Self::Qwen(model) => model.forward(token),
            Self::Flash(model) => model.forward(token),
        }
    }

    fn forward_many(&mut self, tokens: &[u32]) -> Result<mlxl3_native::array::Array> {
        anyhow::ensure!(!tokens.is_empty(), "empty token batch");
        if let Self::Lfm2(model) = self
            && model.uses_moe_exact_prefill()
        {
            let (output, stats) = model.prefill_moe_exact(tokens)?;
            if std::env::var_os("MLXL3_LFM_PREFILL_TRACE").is_some_and(|value| value == "1") {
                eprintln!(
                    "{}",
                    json!({
                        "type": "lfm_prefill_dispatch",
                        "tokens": stats.tokens,
                        "blocks": stats.blocks,
                        "grouped_moe_calls": stats.grouped_moe_calls,
                        "router_rows": stats.router_rows,
                        "max_block": stats.max_block,
                    })
                );
            }
            return Ok(output);
        }
        if tokens.len() >= 24 {
            return match self {
                Self::Gemma4(model) if model.can_batch(tokens.len()) => {
                    model.forward_tokens(tokens)
                }
                Self::Lfm2(model) if model.supports_batched_prefill() => model.forward(tokens),
                Self::Lfm3(model) => model.forward(tokens),
                Self::Inkling(model, ..) => model.forward_tokens(tokens),
                Self::Ling(_) => self.forward_many_serial(tokens),
                Self::Qwen(model) => model.forward_tokens(tokens),
                Self::Flash(model) => model.forward_tokens(tokens),
                Self::Gemma4(_) | Self::Lfm2(_) => self.forward_many_serial(tokens),
            };
        }
        self.forward_many_serial(tokens)
    }

    fn forward_many_serial(&mut self, tokens: &[u32]) -> Result<mlxl3_native::array::Array> {
        match self {
            Self::Qwen(model) => return model.prefill_serial(tokens),
            Self::Flash(model) => return model.forward_tokens(tokens),
            Self::Ling(model) => return model.prefill_serial(tokens),
            Self::Inkling(model, ..) => return model.forward_tokens(tokens),
            _ => {}
        }
        let mut output = None;
        for &token in tokens {
            output = Some(self.forward(token)?);
        }
        output.context("empty token batch")
    }

    /// Whole prefill chunks per span; Qwen MoE batches the MLP of a span and
    /// Inkling 2 its experts (SPEED-2026-11-25-INK2M-PREFILL-G, 8 unless
    /// `MLXL3_INKLING_MOE_GROUP` says otherwise).
    fn prefill_group(&self) -> Result<usize> {
        match self {
            Self::Qwen(model) if model.supports_moe_grouping() => q36_moe_group(),
            Self::Inkling(..) => mlxl3_native::inkling2::moe_group(),
            _ => Ok(1),
        }
    }

    /// End of the prefill span from `start`; a Qwen MoE group may take the tail.
    /// An Inkling span is whole chunks only: its tail stays a call of its own,
    /// so the prompt-cache boundary always ends a span.
    fn prefill_span_end(
        &self,
        start: usize,
        len: usize,
        chunk: usize,
        group: usize,
    ) -> Option<usize> {
        let min_tail = match self {
            Self::Inkling(..) => 0,
            _ => mlxl3_native::Q36_MOE_TAIL_MIN,
        };
        mlxl3_native::prefill_span_end_merged(start, len, chunk, group, min_tail)
    }

    /// Runs one span; with `checkpoint`, a span ending in a tail also returns
    /// the state and logits at its last whole chunk.
    #[allow(clippy::type_complexity)]
    fn forward_span(
        &mut self,
        tokens: &[u32],
        checkpoint: bool,
    ) -> Result<(
        mlxl3_native::array::Array,
        Option<(
            mlxl3_native::qwen35::QwenSnapshot,
            mlxl3_native::array::Array,
        )>,
    )> {
        if tokens.len() > self.prefill_chunk_size() {
            match self {
                Self::Qwen(model) => {
                    return model.forward_tokens_grouped_checkpoint(tokens, checkpoint);
                }
                // whole chunks with their experts computed together; no state inside the span
                Self::Inkling(model, ..) => {
                    return Ok((model.forward_tokens_grouped(tokens)?, None));
                }
                _ => {}
            }
        }
        Ok((self.forward_many(tokens)?, None))
    }

    /// Flash has no reusable chat checkpoint at intermediate span boundaries.
    fn advance_flash_prompt(
        &mut self,
        chunk: &[u32],
        total: usize,
        end: usize,
        cancelled: &AtomicBool,
    ) -> Result<bool> {
        let Self::Flash(target) = self else {
            return Ok(false);
        };
        target.advance_chat_prompt(chunk, total, end, cancelled)
    }

    fn prefill_chunk_size(&self) -> usize {
        match self {
            Self::Gemma4(_) | Self::Lfm3(_) => 256,
            Self::Inkling(..) => 128,
            Self::Lfm2(model) if model.supports_batched_prefill() => 256,
            Self::Lfm2(model) if model.uses_moe_exact_prefill() => 8,
            Self::Qwen(_) => 256,
            Self::Flash(_) => mlxl3_native::flash_support::PREFILL_CHUNK,
            Self::Ling(_) => 128,
            Self::Lfm2(_) => 1,
        }
    }

    fn context_limit(&self) -> i32 {
        match self {
            Self::Gemma4(model) => model.context_limit(),
            Self::Inkling(model, ..) => model.context_limit(),
            Self::Lfm2(model) => model.context_limit(),
            Self::Lfm3(model) => model.context_limit(),
            Self::Ling(model) => model.context_limit(),
            Self::Qwen(model) => model.context_limit(),
            Self::Flash(model) => model.context_limit(),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Gemma4(model) => model.reset(),
            Self::Inkling(model, draft, _) => {
                model.reset();
                if let Some(draft) = draft {
                    mlxl3_native::inkling2_draft::BlockDrafter::reset(draft.as_mut());
                }
            }
            Self::Lfm2(model) => model.reset(),
            Self::Lfm3(model) => model.reset(),
            Self::Ling(model) => model.reset(),
            Self::Qwen(model) => model.reset(),
            Self::Flash(model) => model.reset(),
        }
    }

    /// Loads the Inkling DSpark drafter when a request wants DSpark and only its source is known, and frees
    /// it when a request does not (DECISION-2026-11-22-INK2M-DSPARK-OFF). Returns whether it loaded one.
    fn prepare_inkling_drafter(&mut self, wanted: bool) -> Result<bool> {
        let Self::Inkling(target, draft, source) = self else {
            return Ok(false);
        };
        if !wanted {
            *draft = None;
            return Ok(false);
        }
        if draft.is_some() {
            return Ok(false);
        }
        // DFLASH2-2026-11-24-INK2M-P0: DFlash2 replaces DSpark when its directory is given
        if let Some(dir) = inkling_dflash2_source() {
            let mut loaded = mlxl3_native::inkling2_dflash2::Inkling2Dflash2::load(&dir, target)
                .context("loading the Inkling DFlash2 drafter")?;
            // DFLASH2-2026-11-24-INK2M-P3c: every width compiled while loading, opt-in
            // (`MLXL3_INKLING_DFLASH2_WARMUP=1`): +1.5 s on the first request without a measured decode gain
            if inkling_warmup_enabled()
                && std::env::var("MLXL3_INKLING_DFLASH2_WARMUP").as_deref() == Ok("1")
            {
                let widths = mlxl3_native::inkling2_draft::BlockDrafter::max_proposals(&loaded);
                mlxl3_native::inkling2_draft::warm_up_drafter(target, &mut loaded, widths)
                    .context("warming up the Inkling DFlash2 drafter")?;
            }
            *draft = Some(Box::new(
                mlxl3_native::inkling2_draft::InklingDrafter::Dflash2(loaded),
            ));
            return Ok(true);
        }
        let Some(path) = source.as_ref() else {
            return Ok(false);
        };
        let mut loaded = mlxl3_native::inkling2_draft::Inkling2Draft::load(path, target)
            .context("loading the Inkling DSpark drafter")?;
        if inkling_warmup_enabled() {
            mlxl3_native::inkling2_draft::warm_up(target, Some(&mut loaded))
                .context("warming up the Inkling DSpark drafter")?;
        }
        *draft = Some(Box::new(
            mlxl3_native::inkling2_draft::InklingDrafter::Dspark(loaded),
        ));
        Ok(true)
    }

    fn eval_state(&self) -> Result<()> {
        match self {
            Self::Lfm3(model) => {
                for (_, state) in model.state_arrays() {
                    state.eval()?;
                }
            }
            Self::Lfm2(model) => {
                for (_, state) in model.state_arrays() {
                    state.eval()?;
                }
            }
            Self::Inkling(model, ..) => model.eval_state()?,
            Self::Gemma4(_) | Self::Ling(_) | Self::Qwen(_) | Self::Flash(_) => {}
        }
        Ok(())
    }

    fn supports_images(&self) -> bool {
        self.image_format().is_some()
    }

    /// Image input layout of the loaded model (`None`: text only).
    fn image_format(&self) -> Option<mlxl3_native::vision_input::ImageFormat> {
        use mlxl3_native::vision_input::ImageFormat;
        match self {
            Self::Lfm3(model) if model.supports_images() => Some(ImageFormat::Lfm3),
            Self::Inkling(model, ..) if model.supports_images() => Some(ImageFormat::Inkling2),
            _ => None,
        }
    }

    fn prepare_images(
        &mut self,
        tokens: &[u32],
        images: &[mlxl3_native::lfm3::PreparedImage],
    ) -> Result<()> {
        // Inkling also clears the features of a previous prompt when this one has no image
        if let Self::Inkling(model, ..) = self {
            return model.prepare_images(tokens, images);
        }
        if images.is_empty() {
            return Ok(());
        }
        let Self::Lfm3(model) = self else {
            bail!("this model does not support images");
        };
        model.prepare_images(tokens, images)
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Gemma4(_) => "Gemma 4",
            Self::Inkling(..) => "Inkling 2 Mini",
            Self::Lfm2(_) => "LFM2",
            Self::Lfm3(model) => model.name(),
            Self::Ling(_) => "Ling 3",
            Self::Qwen(_) => "Qwen3.5",
            Self::Flash(_) => "Qwen3.8 Flash Next",
        }
    }
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
#[derive(Deserialize)]
struct BridgeRequest {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    request_id: String,
    #[serde(default)]
    conversation_id: String,
    #[serde(default = "reuse_cache_by_default")]
    reuse_prompt_cache: bool,
    #[serde(default)]
    memory_saver: bool,
    #[serde(default)]
    messages: Vec<Value>,
    #[serde(default = "unlimited_tokens")]
    max_tokens: i64,
    #[serde(default)]
    temperature: f32,
    #[serde(default)]
    top_k: usize,
    #[serde(default = "unit_penalty")]
    repetition_penalty: f32,
    #[serde(default = "unit_penalty")]
    top_p: f32,
    #[serde(default)]
    presence_penalty: f32,
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    mcp_enabled: bool,
    #[serde(default)]
    allowed_tools: Option<Vec<String>>,
    #[serde(default)]
    response_schema: Option<String>,
    #[serde(default)]
    dflash2: bool,
    #[serde(default)]
    dflash_draft_path: String,
    #[serde(default = "default_mtp_depth")]
    dflash_mode: usize,
    #[serde(default)]
    mtp: bool,
    #[serde(default)]
    mtp_head_path: String,
    #[serde(default = "default_mtp_depth")]
    mtp_depth: usize,
    /// Inkling 2 speculative decoding. Omitted: on when a DFlash2 drafter is available
    /// (DECISION-2026-11-25-INK2M-DFLASH2-ON), off with the official DSpark drafter alone
    /// (DECISION-2026-11-22-INK2M-DSPARK-OFF: at temperature 1 plain decoding is as fast).
    #[serde(default, deserialize_with = "present_bool")]
    #[cfg_attr(not(feature = "mlx"), allow(dead_code))]
    dspark: Option<bool>,
    /// SPEED-2026-11-23-INK2M-MAYA LK: Inkling 2 prompt lookup without a drafter; omitted = on unless
    /// `MLXL3_INKLING_LOOKUP=0` (DECISION-2026-11-23-INK2M-LOOKUP-ON).
    #[serde(default)]
    #[cfg_attr(not(feature = "mlx"), allow(dead_code))]
    lookup: Option<bool>,
    /// Requested `reasoning_effort` (one of `tokenizer::REASONING_EFFORTS`); omitted = the
    /// chat template's default.
    #[serde(default)]
    #[cfg_attr(not(feature = "mlx"), allow(dead_code))]
    reasoning_effort: Option<String>,
    /// Tool rounds before the final answer (`tool_call::tool_rounds`); omitted = 5, the
    /// chat default. The Agent mode asks for more.
    #[serde(default)]
    tool_rounds: Option<usize>,
    /// Reasoning tokens per step before the engine ends the thinking block and lets the
    /// model act (`agent_loop::reasoning_budget`); omitted = no budget.
    #[serde(default)]
    reasoning_budget: Option<usize>,
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn default_mtp_depth() -> usize {
    1
}

#[cfg(all(feature = "mlx", feature = "chat"))]
enum ChatMtpHead {
    Qwen(Box<mlxl3_native::mtp::Head>),
    Flash(Box<mlxl3_native::qwen4::FlashMtpHead>),
}
#[cfg(all(feature = "mlx", feature = "chat"))]
impl ChatMtpHead {
    fn load(
        path: &std::path::Path,
        model: &NativeChatModel,
        cancelled: &AtomicBool,
    ) -> Result<Self> {
        match model {
            NativeChatModel::Qwen(target) => Ok(Self::Qwen(Box::new(
                mlxl3_native::mtp::Head::load_cancellable(path, target, cancelled)?,
            ))),
            NativeChatModel::Flash(target) => Ok(Self::Flash(Box::new(
                mlxl3_native::qwen4::FlashMtpHead::load(&checkpoint::inspect(path)?, target)?,
            ))),
            _ => bail!("MTP requires a supported Qwen target"),
        }
    }
    fn reset(&mut self) -> Result<()> {
        match self {
            Self::Qwen(head) => head.reset(),
            Self::Flash(head) => {
                head.reset();
                Ok(())
            }
        }
    }
}
#[cfg(all(feature = "mlx", feature = "chat"))]
enum ChatMtpSession {
    Qwen(Box<mlxl3_native::mtp::Session>),
    Flash(mlxl3_native::qwen4::FlashMtpSession),
}
#[cfg(all(feature = "mlx", feature = "chat"))]
impl ChatMtpSession {
    fn new(head: &ChatMtpHead, depth: usize) -> Result<Self> {
        match head {
            ChatMtpHead::Qwen(_) => Ok(Self::Qwen(Box::new(mlxl3_native::mtp::Session::new(
                depth,
            )?))),
            ChatMtpHead::Flash(_) => {
                anyhow::ensure!(depth == 1, "Flash MTP supports depth 1");
                Ok(Self::Flash(Default::default()))
            }
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &mut self,
        model: &mut NativeChatModel,
        head: &mut ChatMtpHead,
        anchor: u32,
        context: usize,
        remaining: usize,
        allow_lookup: impl Fn(&[u32]) -> bool,
        draw: mlxl3_native::sampling::Draw<'_>,
    ) -> Result<u32> {
        match (self, model, head) {
            (Self::Qwen(session), NativeChatModel::Qwen(target), ChatMtpHead::Qwen(head)) => {
                session.advance_sampled(
                    target,
                    head,
                    anchor,
                    context,
                    remaining,
                    allow_lookup,
                    Some(draw),
                )
            }
            (Self::Flash(session), NativeChatModel::Flash(target), ChatMtpHead::Flash(head)) => {
                session.advance_sampled(target, head, anchor, context, remaining, Some(draw))
            }
            _ => bail!("MTP head and target do not match"),
        }
    }
    fn enable_adaptive(&mut self) {
        if let Self::Qwen(session) = self {
            session.enable_adaptive();
        }
    }
    fn enable_prompt_lookup(&mut self, tokens: &[u32]) {
        if let Self::Qwen(session) = self {
            session.enable_prompt_lookup(tokens);
        }
    }
    fn stats(&self) -> (usize, usize, usize, usize, [usize; 4]) {
        match self {
            Self::Qwen(s) => (
                s.proposed,
                s.accepted,
                s.blocks,
                s.lookup_blocks,
                s.depth_blocks,
            ),
            Self::Flash(s) => (s.proposed, s.accepted, s.blocks, 0, [0, s.blocks, 0, 0]),
        }
    }
    /// Head and prompt-lookup cycles by width and accepted prefix (Qwen only).
    fn accept_histograms(&self) -> Option<[mlxl3_native::speculative::AcceptHistogram; 2]> {
        match self {
            Self::Qwen(s) => Some([s.accept_histogram, s.lookup_accept_histogram]),
            Self::Flash(_) => None,
        }
    }
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn prefill_flash_mtp(
    model: &mut NativeChatModel,
    head: &mut mlxl3_native::qwen4::FlashMtpHead,
    tokens: &[u32],
    cache: &mut Option<PromptCache>,
    cancelled: &AtomicBool,
) -> Result<(mlxl3_native::array::Array, Option<DFlashChat>, usize)> {
    let NativeChatModel::Flash(target) = model else {
        bail!("Flash MTP requires Flash Next");
    };
    *cache = None;
    Ok((target.prefill_mtp(head, tokens, cancelled)?, None, 0))
}

#[cfg(all(feature = "mlx", feature = "chat"))]
struct DFlashChat {
    cache: mlxl3_native::dflash::DFlashCache,
    pending: VecDeque<u32>,
    recent: VecDeque<(usize, usize)>,
    proposed: usize,
    accepted: usize,
    blocks: usize,
    block_seconds: f64,
    mode: usize,
    copy: Option<mlxl3_native::speculative::DFlashCopy>,
    lookup_blocks: usize,
    width_cap: Option<usize>,
    /// Q27D-16: TensorOps verification makes wide blocks nearly as cheap as narrow ones.
    wide: bool,
    /// Q27D-37: draft nodes of a greedy verification tree (0 keeps the selector chain).
    tree_nodes: usize,
    /// Q27D-40: sampled blocks are verified as trees too (exact speculative sampling).
    tree_sampled: bool,
    /// Q27D-47: divergence budget (nats per emitted token) of the relaxed sampled tree.
    relax_kl: f64,
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn reuse_cache_by_default() -> bool {
    true
}

#[cfg(all(feature = "mlx", feature = "chat"))]
thread_local! {
    /// DSpark choice of the request being served (the bridge serves one request at a time); off by
    /// default, also for the CLI and the server.
    static DSPARK_REQUESTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// SPEED-2026-11-23-INK2M-MAYA LK: prompt lookup choice of the request being served (`None` = default).
    static LOOKUP_REQUESTED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Sets the prompt lookup choice for the duration of one request; dropping it restores the default.
#[cfg(all(feature = "mlx", feature = "chat"))]
struct LookupRequest;

#[cfg(all(feature = "mlx", feature = "chat"))]
impl LookupRequest {
    fn set(enabled: Option<bool>) -> Self {
        LOOKUP_REQUESTED.with(|cell| cell.set(enabled));
        Self
    }
}

#[cfg(all(feature = "mlx", feature = "chat"))]
impl Drop for LookupRequest {
    fn drop(&mut self) {
        LOOKUP_REQUESTED.with(|cell| cell.set(None));
    }
}

/// SPEED-2026-11-23-INK2M-MAYA LK: the request's choice, else on unless `MLXL3_INKLING_LOOKUP` (`flag`) is
/// `0` (DECISION-2026-11-23-INK2M-LOOKUP-ON).
#[cfg(all(feature = "mlx", feature = "chat"))]
fn inkling_lookup_requested(requested: Option<bool>, flag: Option<&str>) -> bool {
    requested.unwrap_or(flag != Some("0"))
}

/// Sets the DSpark choice for the duration of one request; dropping it restores the default (off).
#[cfg(all(feature = "mlx", feature = "chat"))]
struct DsparkRequest;

#[cfg(all(feature = "mlx", feature = "chat"))]
impl DsparkRequest {
    fn set(enabled: bool) -> Self {
        DSPARK_REQUESTED.with(|cell| cell.set(enabled));
        Self
    }
}

#[cfg(all(feature = "mlx", feature = "chat"))]
impl Drop for DsparkRequest {
    fn drop(&mut self) {
        DSPARK_REQUESTED.with(|cell| cell.set(false));
    }
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
thread_local! {
    /// Reasoning level of the request being served (the bridge serves one request at a time).
    static REASONING_REQUESTED: std::cell::RefCell<Option<String>> =
        const { std::cell::RefCell::new(None) };
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
thread_local! {
    /// AGENT-2026-11-23-BOUCLE B7: request whose prefill reports its progress, and when it last did.
    static PREFILL_PROGRESS: std::cell::RefCell<Option<(String, Instant)>> =
        const { std::cell::RefCell::new(None) };
}

/// Lets a long prefill tell the app it is alive (the app stops an Agent turn after five
/// silent minutes); dropping it ends the reports.
#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
struct PrefillProgress;

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
impl PrefillProgress {
    fn set(request_id: &str) -> Self {
        PREFILL_PROGRESS
            .with(|cell| *cell.borrow_mut() = Some((request_id.to_owned(), Instant::now())));
        Self
    }
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
impl Drop for PrefillProgress {
    fn drop(&mut self) {
        PREFILL_PROGRESS.with(|cell| *cell.borrow_mut() = None);
    }
}

/// At most one status every five seconds while a prompt is read.
#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
#[cfg_attr(not(feature = "mlx"), allow(dead_code))]
fn report_prefill_progress(done: usize, total: usize) -> Result<()> {
    let due = PREFILL_PROGRESS.with(|cell| {
        let mut slot = cell.borrow_mut();
        match slot.as_mut() {
            Some((request, last)) if last.elapsed() >= std::time::Duration::from_secs(5) => {
                *last = Instant::now();
                Some(request.clone())
            }
            _ => None,
        }
    });
    if let Some(request_id) = due {
        emit_event(json!({
            "type":"generation_status", "request_id":request_id, "phase":"prefill",
            "text":format!("Reading context · {done} / {total} tokens")
        }))?;
    }
    Ok(())
}

/// Sets the reasoning level for one request; dropping it restores the template default.
#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
struct ReasoningRequest;

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
impl ReasoningRequest {
    fn set(effort: Option<String>) -> Result<Self> {
        if let Some(level) = &effort {
            anyhow::ensure!(
                mlxl3_native::tokenizer::accepts_reasoning_effort(level),
                "unsupported reasoning effort {level:?}"
            );
        }
        REASONING_REQUESTED.with(|cell| *cell.borrow_mut() = effort);
        Ok(Self)
    }
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
impl Drop for ReasoningRequest {
    fn drop(&mut self) {
        REASONING_REQUESTED.with(|cell| *cell.borrow_mut() = None);
    }
}

#[cfg(all(
    test,
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
#[test]
fn bridge_reasoning_effort_is_optional_validated_and_scoped() -> Result<()> {
    let current = || REASONING_REQUESTED.with(|cell| cell.borrow().clone());
    let legacy: BridgeRequest = serde_json::from_value(json!({"type":"generate"}))?;
    assert!(
        legacy.reasoning_effort.is_none(),
        "older apps keep the template default"
    );
    for level in mlxl3_native::tokenizer::REASONING_EFFORTS {
        let request: BridgeRequest = serde_json::from_value(
            json!({"type":"generate", "reasoning_effort":level, "max_tokens":-1}),
        )?;
        assert!(request.reasoning_budget.is_none());
        assert_eq!(request.max_tokens, -1);
        {
            let _reasoning = ReasoningRequest::set(request.reasoning_effort.clone())?;
            assert_eq!(current().as_deref(), Some(level));
        }
        assert!(
            current().is_none(),
            "the next request starts from the default"
        );
    }
    for invalid in ["", "ultra", "High", "0.9"] {
        assert!(
            ReasoningRequest::set(Some(invalid.into())).is_err(),
            "{invalid:?}"
        );
        assert!(current().is_none());
    }
    for invalid in [json!(1), json!(true), json!(["low"])] {
        assert!(
            serde_json::from_value::<BridgeRequest>(
                json!({"type":"generate", "reasoning_effort":invalid})
            )
            .is_err()
        );
    }
    Ok(())
}

#[cfg(all(test, feature = "mlx", feature = "chat"))]
#[test]
fn bridge_tool_rounds_are_optional_and_bounded() -> Result<()> {
    let legacy: BridgeRequest = serde_json::from_value(json!({"type":"generate"}))?;
    assert_eq!(
        mlxl3_native::tool_call::tool_rounds(legacy.tool_rounds)?,
        5,
        "older apps keep five rounds"
    );
    assert_eq!(
        legacy.reasoning_budget, None,
        "older apps keep free reasoning"
    );
    let agent: BridgeRequest = serde_json::from_value(
        json!({"type":"generate", "tool_rounds":256, "reasoning_budget":8192, "max_tokens":-1}),
    )?;
    assert_eq!(
        mlxl3_native::tool_call::tool_rounds(agent.tool_rounds)?,
        256
    );
    assert_eq!(
        mlxl3_native::agent_loop::reasoning_budget(agent.reasoning_budget)?,
        Some(8192)
    );
    assert_eq!(agent.max_tokens, -1);
    for budget in [0, 255, 131_073] {
        let request: BridgeRequest =
            serde_json::from_value(json!({"type":"generate", "reasoning_budget":budget}))?;
        assert!(mlxl3_native::agent_loop::reasoning_budget(request.reasoning_budget).is_err());
    }
    for rounds in [0, 513] {
        let request: BridgeRequest =
            serde_json::from_value(json!({"type":"generate", "tool_rounds":rounds}))?;
        assert!(mlxl3_native::tool_call::tool_rounds(request.tool_rounds).is_err());
    }
    for invalid in [json!(-1), json!("24"), json!(2.5)] {
        assert!(
            serde_json::from_value::<BridgeRequest>(
                json!({"type":"generate", "tool_rounds":invalid})
            )
            .is_err(),
            "{invalid}"
        );
    }
    Ok(())
}

#[cfg(all(test, feature = "mlx", feature = "chat"))]
#[test]
fn bridge_inkling_lookup_follows_the_request_then_the_flag() -> Result<()> {
    let legacy: BridgeRequest = serde_json::from_value(json!({"type":"generate"}))?;
    assert_eq!(legacy.lookup, None);
    for (requested, flag, want) in [
        (None, None, true),
        (None, Some("1"), true),
        (None, Some("0"), false),
        (None, Some("yes"), true),
        (Some(true), None, true),
        (Some(true), Some("0"), true),
        (Some(false), Some("1"), false),
        (Some(false), None, false),
    ] {
        assert_eq!(inkling_lookup_requested(requested, flag), want);
    }
    let request: BridgeRequest = serde_json::from_value(json!({"type":"generate", "lookup":true}))?;
    assert_eq!(request.lookup, Some(true));
    {
        let _lookup = LookupRequest::set(request.lookup);
        assert_eq!(LOOKUP_REQUESTED.with(std::cell::Cell::get), Some(true));
    }
    assert_eq!(LOOKUP_REQUESTED.with(std::cell::Cell::get), None);
    assert!(
        serde_json::from_value::<BridgeRequest>(json!({"type":"generate", "lookup":"on"})).is_err()
    );
    Ok(())
}

#[cfg(all(test, feature = "mlx", feature = "chat"))]
#[test]
fn bridge_dspark_is_opt_in_and_scoped_to_one_request() -> Result<()> {
    // DECISION-2026-11-22-INK2M-DSPARK-OFF: a request without the field decodes plainly with the official
    // drafter; DECISION-2026-11-25-INK2M-DFLASH2-ON: it speculates when a DFlash2 drafter is available
    let current = || DSPARK_REQUESTED.with(std::cell::Cell::get);
    let legacy: BridgeRequest = serde_json::from_value(json!({"type":"generate"}))?;
    assert_eq!(legacy.dspark, None, "an omitted field is not a choice");
    assert!(!inkling_speculation_requested(None, false, None));
    assert!(inkling_speculation_requested(None, true, None));
    assert!(inkling_speculation_requested(None, true, Some("1")));
    assert!(!inkling_speculation_requested(None, true, Some("0")));
    assert!(!inkling_speculation_requested(Some(false), true, None));
    assert!(inkling_speculation_requested(Some(true), false, Some("0")));
    assert!(!current(), "no request: off");
    // the drafter directory: the flag alone when set, else inside or beside the model directory
    let root = tempfile::tempdir()?;
    let model = root.path().join("Inkling-2-Mini-EXL3");
    std::fs::create_dir_all(&model)?;
    assert_eq!(inkling_dflash2_source_in(None, Some(&model)), None);
    assert_eq!(inkling_dflash2_source_in(None, None), None);
    let beside = root.path().join("Inkling-2-Mini-DFlash2");
    std::fs::create_dir_all(&beside)?;
    assert_eq!(
        inkling_dflash2_source_in(None, Some(&model)),
        None,
        "no config.json yet"
    );
    std::fs::write(beside.join("config.json"), "{}")?;
    assert_eq!(
        inkling_dflash2_source_in(None, Some(&model)),
        Some(beside.clone())
    );
    let inside = model.join("dflash2");
    std::fs::create_dir_all(&inside)?;
    std::fs::write(inside.join("config.json"), "{}")?;
    assert_eq!(
        inkling_dflash2_source_in(None, Some(&model)),
        Some(inside.clone())
    );
    assert_eq!(
        inkling_dflash2_source_in(Some(beside.clone().into_os_string()), Some(&model)),
        Some(beside)
    );
    assert_eq!(
        inkling_dflash2_source_in(Some(std::ffi::OsString::new()), Some(&model)),
        None,
        "an empty flag turns the lookup off"
    );
    assert_eq!(
        inkling_dflash2_source_in(
            Some(root.path().join("missing").into_os_string()),
            Some(&model)
        ),
        None
    );
    for enabled in [false, true] {
        let request: BridgeRequest =
            serde_json::from_value(json!({"type":"generate", "dspark":enabled}))?;
        assert_eq!(request.dspark, Some(enabled));
        {
            let _dspark = DsparkRequest::set(enabled);
            assert_eq!(current(), enabled);
        }
        assert!(!current(), "the next request starts from the default (off)");
    }
    for invalid in [json!(null), json!(1), json!("true")] {
        assert!(
            serde_json::from_value::<BridgeRequest>(json!({"type":"generate", "dspark":invalid}))
                .is_err()
        );
    }
    Ok(())
}

#[cfg(all(test, feature = "mlx", feature = "chat"))]
#[test]
fn bridge_memory_saver_is_opt_in_and_requires_boolean() -> Result<()> {
    let legacy: BridgeRequest = serde_json::from_value(json!({"type":"generate"}))?;
    assert!(!legacy.memory_saver && legacy.reuse_prompt_cache);
    for saver in [false, true] {
        let request: BridgeRequest = serde_json::from_value(json!({
            "type":"generate", "memory_saver":saver, "reuse_prompt_cache":true
        }))?;
        assert_eq!(request.memory_saver, saver);
        assert!(request.reuse_prompt_cache);
    }
    for invalid in [json!(null), json!(1), json!("true"), json!([])] {
        assert!(
            serde_json::from_value::<BridgeRequest>(json!({
                "type":"generate", "memory_saver":invalid
            }))
            .is_err()
        );
    }
    Ok(())
}

/// One prefill checkpoint, never speculative/pending decode state. The resident
/// bridge owns one immutable model/tokenizer/context configuration. Draft reload
/// clears this cache; exact rendered IDs cover changes to system/tools/templates.
#[cfg(all(feature = "mlx", feature = "chat"))]
struct PromptCache {
    conversation: String,
    tokens: Vec<u32>,
    state: CacheState,
    logits: mlxl3_native::array::Array,
    draft: Option<mlxl3_native::dflash::DFlashCache>,
    mtp: Option<mlxl3_native::mtp::Cache>,
}

/// The saved model state of a `PromptCache`.
#[cfg(all(feature = "mlx", feature = "chat"))]
enum CacheState {
    Qwen(mlxl3_native::qwen35::QwenSnapshot),
    /// SPEED-2026-11-23-INK2M-MAYA PC: Inkling 2 resumes its live state at the saved boundary.
    Inkling(mlxl3_native::inkling2::InklingPrefix),
    /// DSPARK-2026-11-24-INK2M-MAYA2-D1: the same with DSpark, the drafter's context included. Only the
    /// DSpark prefill resumes it (its chunks differ from the plain prefill's).
    InklingDspark(mlxl3_native::inkling2_draft::DsparkPrefix),
}

#[cfg(all(feature = "mlx", feature = "chat"))]
impl CacheState {
    fn retained_bytes(&self) -> Result<usize> {
        match self {
            Self::Qwen(state) => state.retained_bytes(),
            Self::Inkling(prefix) => prefix.retained_bytes(),
            Self::InklingDspark(prefix) => prefix.retained_bytes(),
        }
    }
}

/// SPEED-2026-11-23-INK2M-MAYA PC: the Inkling 2 prompt cache, on unless `MLXL3_INKLING_PROMPT_CACHE=0`.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn inkling_prompt_cache() -> bool {
    std::env::var("MLXL3_INKLING_PROMPT_CACHE").as_deref() != Ok("0")
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn prompt_cache_budget() -> Result<usize> {
    Ok(match std::env::var("MLXL3_PROMPT_CACHE_MIB") {
        Ok(value) => value
            .parse::<usize>()
            .ok()
            .and_then(mlxl3_native::memory_policy::mib_budget)
            .context("MLXL3_PROMPT_CACHE_MIB must be an integer in 0..4096")?,
        Err(std::env::VarError::NotPresent) => {
            mlxl3_native::memory_policy::DEFAULT_PROMPT_CACHE_BYTES
        }
        Err(error) => return Err(error.into()),
    })
}

#[cfg(all(feature = "mlx", feature = "chat"))]
impl PromptCache {
    fn capture(
        conversation: &str,
        tokens: &[u32],
        model: &NativeChatModel,
        logits: &mlxl3_native::array::Array,
        dflash: Option<&DFlashChat>,
    ) -> Result<Option<Self>> {
        if let NativeChatModel::Inkling(target, ..) = model {
            if !inkling_prompt_cache() || dflash.is_some() {
                return Ok(None);
            }
            anyhow::ensure!(
                usize::try_from(target.offset()).ok() == Some(tokens.len()),
                "prefix cache position differs from its token history"
            );
            if prompt_cache_budget()? == 0 {
                return Ok(None);
            }
            let saved = Self {
                conversation: conversation.to_owned(),
                tokens: tokens.to_vec(),
                state: CacheState::Inkling(target.prefix_snapshot()?),
                logits: logits.compact_for_cache()?,
                draft: None,
                mtp: None,
            };
            return saved.within_budget();
        }
        let NativeChatModel::Qwen(target) = model else {
            return Ok(None);
        };
        anyhow::ensure!(
            usize::try_from(target.offset()).ok() == Some(tokens.len()),
            "prefix cache position differs from its token history"
        );
        if prompt_cache_budget()? == 0 {
            return Ok(None);
        }
        Self::capture_snapshot(conversation, tokens, target.snapshot()?, logits, dflash)
    }

    /// Saves a Qwen state taken at `tokens.len()`, also from inside a span.
    fn capture_snapshot(
        conversation: &str,
        tokens: &[u32],
        state: mlxl3_native::qwen35::QwenSnapshot,
        logits: &mlxl3_native::array::Array,
        dflash: Option<&DFlashChat>,
    ) -> Result<Option<Self>> {
        anyhow::ensure!(
            usize::try_from(state.offset()).ok() == Some(tokens.len()),
            "prefix cache position differs from its token history"
        );
        if prompt_cache_budget()? == 0 {
            return Ok(None);
        }
        let state = CacheState::Qwen(state.compact_for_cache()?);
        let saved = Self {
            conversation: conversation.to_owned(),
            tokens: tokens.to_vec(),
            state,
            logits: logits.compact_for_cache()?,
            draft: dflash
                .map(|session| session.cache.try_clone()?.compact_for_cache())
                .transpose()?,
            mtp: None,
        };
        saved.within_budget()
    }

    /// DSPARK-2026-11-24-INK2M-MAYA2-D1: saves a DSpark prefill state taken at `tokens.len()`.
    fn capture_dspark(
        conversation: &str,
        tokens: &[u32],
        prefix: mlxl3_native::inkling2_draft::DsparkPrefix,
        logits: &mlxl3_native::array::Array,
    ) -> Result<Option<Self>> {
        anyhow::ensure!(
            usize::try_from(prefix.offset()).ok() == Some(tokens.len()),
            "DSpark prefix position differs from its token history"
        );
        if !inkling_prompt_cache() || prompt_cache_budget()? == 0 {
            return Ok(None);
        }
        Self {
            conversation: conversation.to_owned(),
            tokens: tokens.to_vec(),
            state: CacheState::InklingDspark(prefix),
            logits: logits.compact_for_cache()?,
            draft: None,
            mtp: None,
        }
        .within_budget()
    }

    fn within_budget(self) -> Result<Option<Self>> {
        let budget = prompt_cache_budget()?;
        let fits = mlxl3_native::memory_policy::cache_fits(
            [
                self.state.retained_bytes()?,
                self.logits.retained_bytes()?,
                self.draft
                    .as_ref()
                    .map(|cache| cache.retained_bytes())
                    .transpose()?
                    .unwrap_or(0),
                self.mtp
                    .as_ref()
                    .map(|cache| cache.retained_bytes())
                    .transpose()?
                    .unwrap_or(0),
                self.tokens
                    .len()
                    .checked_mul(4)
                    .context("token cache size overflow")?,
                self.conversation.len(),
            ],
            budget,
        );
        Ok(fits.then_some(self))
    }
}

/// Strict `MLXL3_Q36_MOE_GROUP`; an invalid value fails the request.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn q36_moe_group() -> Result<usize> {
    let value = std::env::var_os("MLXL3_Q36_MOE_GROUP");
    let text = value
        .as_deref()
        .map(|value| value.to_str().context("MLXL3_Q36_MOE_GROUP is not UTF-8"))
        .transpose()?;
    mlxl3_native::q36_moe_group_setting(text).with_context(|| {
        format!(
            "MLXL3_Q36_MOE_GROUP must be an integer from 1 to {}",
            mlxl3_native::Q36_MOE_GROUP_MAX
        )
    })
}

/// DSPARK-2026-11-24-INK2M-MAYA2-D1: the DSpark prefill with the prompt cache. A DSpark prefix of the same
/// conversation that starts the prompt is resumed (target, drafter context and last features), the rest is
/// prefilled by the cold prefill's chunks, and the state at the prompt's last chunk boundary is saved for the
/// next turn (`MLXL3_INKLING_PROMPT_CACHE=0` or a zero budget: every turn prefills from scratch, as before).
#[cfg(all(feature = "mlx", feature = "chat"))]
#[allow(clippy::too_many_arguments)]
fn prefill_dspark_round(
    session: &mut mlxl3_native::inkling2_draft::DsparkSession,
    target: &mut mlxl3_native::inkling2::Inkling2,
    drafter: &mut mlxl3_native::inkling2_draft::InklingDrafter,
    tokens: &[u32],
    images: &[mlxl3_native::lfm3::PreparedImage],
    cache: &mut Option<PromptCache>,
    conversation: &str,
    reuse: bool,
    cancelled: &AtomicBool,
) -> Result<(mlxl3_native::array::Array, usize)> {
    const CHUNK: usize = mlxl3_native::inkling2_draft::DsparkSession::PREFILL_CHUNK;
    // a state computed from image features is neither resumed nor saved for a later turn
    let reuse = reuse && images.is_empty() && inkling_prompt_cache();
    let previous = cache.take().filter(|saved| {
        reuse
            && saved.conversation == conversation
            && saved.draft.is_none()
            && saved.mtp.is_none()
            && matches!(saved.state, CacheState::InklingDspark(_))
            && mlxl3_native::reusable_prefix_len(&saved.tokens, tokens, CHUNK) > 0
    });
    let resume = match previous {
        Some(PromptCache {
            state: CacheState::InklingDspark(prefix),
            logits,
            ..
        }) => Some((prefix, logits)),
        _ => None,
    };
    let checkpoint = (reuse && prompt_cache_budget()? > 0)
        .then_some(tokens.len() / CHUNK * CHUNK)
        .filter(|&boundary| boundary > 0);
    let mut saved = None;
    let (logits, cached) = session.prefill_resumable(
        target,
        drafter,
        tokens,
        images,
        resume,
        checkpoint,
        &mut |prefix, logits| {
            let boundary = usize::try_from(prefix.offset())?;
            saved = PromptCache::capture_dspark(conversation, &tokens[..boundary], prefix, logits)?;
            Ok(())
        },
        &|| cancelled.load(Ordering::Relaxed),
    )?;
    *cache = saved;
    Ok((logits, cached))
}

#[cfg(all(feature = "mlx", feature = "chat"))]
#[allow(clippy::too_many_arguments)]
fn prefill_round(
    model: &mut NativeChatModel,
    tokens: &[u32],
    images: &[mlxl3_native::lfm3::PreparedImage],
    draft: Option<&mlxl3_native::dflash::DFlashWeights>,
    cache: &mut Option<PromptCache>,
    conversation: &str,
    reuse: bool,
    cancelled: &AtomicBool,
) -> Result<(mlxl3_native::array::Array, Option<DFlashChat>, usize)> {
    if cancelled.load(Ordering::Relaxed) {
        model.reset();
        bail!("generation cancelled");
    }
    if let NativeChatModel::Flash(target) = model
        && target.dflash2_active()
    {
        *cache = None;
        return Ok((target.prefill_dflash2(tokens, cancelled)?, None, 0));
    }
    // a state computed from image features is neither resumed nor saved for a later turn
    let reuse = reuse && images.is_empty();
    let chunk_size = if draft.is_some() {
        mlxl3_native::dflash::PREFILL_CHUNK
    } else {
        model.prefill_chunk_size()
    };
    // Preserve cold-prefill chunk boundaries: changing shapes can change floating
    // point rounding even when the mathematical prefix is the same.
    let checkpoint_offset = tokens.len() / chunk_size * chunk_size;
    let previous = cache.take().filter(|saved| {
        reuse
            && images.is_empty()
            && saved.conversation == conversation
            && saved.mtp.is_none()
            && saved.draft.is_some() == draft.is_some()
            && !matches!(saved.state, CacheState::InklingDspark(_))
            && mlxl3_native::reusable_prefix_len(&saved.tokens, tokens, chunk_size) > 0
    });
    let mut cached = 0;
    let mut logits = None;
    // SPEED-2026-11-23-INK2M-MAYA PC: Inkling goes back to the saved boundary of its live state (no reset);
    // when it cannot, the prompt is prefilled from scratch as before
    let previous = match previous {
        Some(PromptCache {
            tokens: saved_tokens,
            state: CacheState::Inkling(prefix),
            logits: saved_logits,
            ..
        }) => {
            if let NativeChatModel::Inkling(target, draft, _) = &mut *model
                && target.resume_prefix(prefix)?
            {
                if let Some(draft) = draft {
                    mlxl3_native::inkling2_draft::BlockDrafter::reset(draft.as_mut());
                }
                cached = saved_tokens.len();
                logits = Some(saved_logits);
                if checkpoint_offset == cached {
                    *cache = PromptCache::capture(
                        conversation,
                        &tokens[..cached],
                        model,
                        logits.as_ref().unwrap(),
                        None,
                    )?;
                }
            }
            None
        }
        other => other,
    };
    if cached == 0 {
        model.reset();
    }
    model.prepare_images(tokens, images)?;
    let mut dflash = draft.map(|_| DFlashChat::new());
    if let Some(saved) = previous {
        let (NativeChatModel::Qwen(target), CacheState::Qwen(state)) = (&mut *model, saved.state)
        else {
            bail!("invalid prefix cache model")
        };
        target.restore(state)?;
        cached = saved.tokens.len();
        logits = Some(saved.logits);
        if let Some(session) = &mut dflash {
            session.cache = saved.draft.context("missing draft prefix state")?;
        }
        if checkpoint_offset == cached {
            *cache = PromptCache::capture(
                conversation,
                &tokens[..cached],
                model,
                logits.as_ref().unwrap(),
                dflash.as_ref(),
            )?;
        }
    }
    // DFlash keeps its per-chunk capture; plain prefill may batch MoE rows of
    // several whole chunks while attention and GDN keep the chunk shapes.
    let group = if draft.is_some() {
        1
    } else {
        model.prefill_group()?
    };
    let mut start = cached;
    while let Some(end) = model.prefill_span_end(start, tokens.len(), chunk_size, group) {
        let chunk = &tokens[start..end];
        let inner_checkpoint = reuse && start < checkpoint_offset && checkpoint_offset < end;
        report_prefill_progress(start, tokens.len())?;
        start = end;
        if cancelled.load(Ordering::Relaxed) {
            model.reset();
            bail!("generation cancelled");
        }
        if draft.is_none()
            && end < tokens.len()
            && model.advance_flash_prompt(chunk, tokens.len(), end, cancelled)?
        {
            #[cfg(test)]
            tests::FLASH_ADVANCED_ENDS.with_borrow_mut(|ends| ends.push(end));
            continue;
        }
        logits = Some(
            if let Some((session, weights)) = dflash.as_mut().zip(draft) {
                let NativeChatModel::Qwen(target) = &mut *model else {
                    bail!("DFlash2 requires a supported Qwen target")
                };
                if usize::try_from(target.offset())?
                    < mlxl3_native::dflash::prefill_capture_start(tokens.len())
                {
                    // The target still reads every token. Only features that
                    // would be discarded before the first draft are skipped.
                    target.forward_tokens(chunk)?
                } else {
                    session.prefill(target, weights, chunk)?
                }
            } else {
                let (logits, saved) = model.forward_span(chunk, inner_checkpoint)?;
                if let Some((state, logits)) = saved {
                    *cache = PromptCache::capture_snapshot(
                        conversation,
                        &tokens[..checkpoint_offset],
                        state,
                        &logits,
                        None,
                    )?;
                }
                logits
            },
        );
        model.eval_state()?;
        let position = end;
        if reuse && position == checkpoint_offset {
            *cache = PromptCache::capture(
                conversation,
                &tokens[..position],
                model,
                logits.as_ref().unwrap(),
                dflash.as_ref(),
            )?;
        }
    }
    let logits = logits.context("no prefill output")?;
    logits.eval()?;
    Ok((logits, dflash, cached))
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn prefill_mtp(
    model: &mut NativeChatModel,
    head: &mut mlxl3_native::mtp::Head,
    tokens: &[u32],
    cache: &mut Option<PromptCache>,
    conversation: &str,
    reuse: bool,
    cancelled: &AtomicBool,
) -> Result<(mlxl3_native::array::Array, Option<DFlashChat>, usize)> {
    const CHUNK: usize = 256;
    let checkpoint = tokens.len() / CHUNK * CHUNK;
    let previous = cache.take().filter(|saved| {
        reuse
            && saved.conversation == conversation
            && saved.mtp.is_some()
            && mlxl3_native::reusable_prefix_len(&saved.tokens, tokens, CHUNK) > 0
    });
    model.reset();
    head.reset()?;
    let mut cached = 0;
    let mut logits = None;
    if let Some(saved) = previous {
        let (NativeChatModel::Qwen(target), CacheState::Qwen(state)) = (&mut *model, saved.state)
        else {
            bail!("MTP requires Qwen");
        };
        target.restore(state)?;
        head.restore(saved.mtp.context("missing MTP prefix state")?)?;
        cached = saved.tokens.len();
        logits = Some(saved.logits);
    }
    if reuse && cached > 0 && checkpoint == cached {
        let mut saved = PromptCache::capture(
            conversation,
            &tokens[..cached],
            model,
            logits.as_ref().unwrap(),
            None,
        )?;
        if let Some(saved) = &mut saved {
            saved.mtp = Some(head.snapshot()?.compact_for_cache()?);
        }
        *cache = saved.map(PromptCache::within_budget).transpose()?.flatten();
    }
    // At every checkpoint the last trunk hidden is deliberately unpaired.
    // The next prompt token is supplied only after prefix eligibility is known.
    // A grouped span runs the trunk once; the head still sees each 256-token
    // chunk with the same pairs and widths as the per-chunk path.
    let group = model.prefill_group()?;
    let mut start = cached;
    while start < tokens.len() {
        if cancelled.load(Ordering::Relaxed) {
            model.reset();
            head.reset()?;
            bail!("generation cancelled");
        }
        let end = model
            .prefill_span_end(start, tokens.len(), CHUNK, group)
            .context("invalid MTP prefill span")?;
        let NativeChatModel::Qwen(target) = &mut *model else {
            bail!("MTP requires Qwen");
        };
        if start > 0 {
            let last = target.mtp_hidden()?.try_clone()?;
            head.extend_cache(target, &last, &tokens[start..start + 1])?;
        }
        let inner = reuse && start < checkpoint && checkpoint < end;
        let (output, raw, mut saved) = if end - start > CHUNK {
            target.forward_mtp_grouped(&tokens[start..end], inner)?
        } else {
            let (output, raw) = target.forward_mtp(&tokens[start..end])?;
            (output, raw, None)
        };
        for chunk in (start..end).step_by(CHUNK) {
            let chunk_end = chunk.saturating_add(CHUNK).min(end);
            let row = i32::try_from(chunk - start)?;
            if chunk > start {
                head.extend_cache(
                    target,
                    &raw.slice(1, row - 1, row)?,
                    &tokens[chunk..chunk + 1],
                )?;
            }
            let paired = i32::try_from(chunk_end - chunk - 1)?;
            if paired > 0 {
                head.extend_cache(
                    target,
                    &raw.slice(1, row, row + paired)?,
                    &tokens[chunk + 1..chunk_end],
                )?;
            }
            if chunk_end == checkpoint
                && let Some((state, checkpoint_logits)) = saved.take()
            {
                let mut prefix = PromptCache::capture_snapshot(
                    conversation,
                    &tokens[..checkpoint],
                    state,
                    &checkpoint_logits,
                    None,
                )?;
                if let Some(prefix) = &mut prefix {
                    prefix.mtp = Some(head.snapshot()?.compact_for_cache()?);
                }
                *cache = prefix
                    .map(PromptCache::within_budget)
                    .transpose()?
                    .flatten();
            }
        }
        anyhow::ensure!(saved.is_none(), "grouped MTP checkpoint was not captured");
        start = end;
        model.eval_state()?;
        logits = Some(output);
        if reuse && end == checkpoint {
            let mut saved = PromptCache::capture(
                conversation,
                &tokens[..end],
                model,
                logits.as_ref().unwrap(),
                None,
            )?;
            if let Some(saved) = &mut saved {
                saved.mtp = Some(head.snapshot()?.compact_for_cache()?);
            }
            *cache = saved.map(PromptCache::within_budget).transpose()?.flatten();
        }
    }
    let logits = logits.context("empty MTP prefill")?;
    logits.eval()?;
    Ok((logits, None, cached))
}

#[cfg(all(feature = "mlx", feature = "chat"))]
impl DFlashChat {
    fn new() -> Self {
        Self {
            cache: Default::default(),
            pending: VecDeque::new(),
            recent: VecDeque::new(),
            proposed: 0,
            accepted: 0,
            blocks: 0,
            block_seconds: 0.,
            mode: 1,
            copy: None,
            lookup_blocks: 0,
            width_cap: None,
            wide: false,
            tree_nodes: 0,
            tree_sampled: false,
            relax_kl: 0.,
        }
    }

    fn configure(&mut self, mode: usize, family: mlxl3_native::dflash::Family, prompt: &[u32]) {
        self.mode = mode;
        // Q27D-16: with TensorOps verification wide blocks are cheap again; keep the policy.
        let tensor = mlxl3_native::linear::dense_tensor_rows_threshold()
            .ok()
            .flatten()
            .is_some_and(|first| first <= 8);
        self.width_cap = mlxl3_native::speculative::dflash_width_cap(
            family == mlxl3_native::dflash::Family::Qwen38Dense,
            mlxl3_native::linear::dense_shared_enabled() && !tensor,
            mode,
        );
        self.wide = tensor && family == mlxl3_native::dflash::Family::Qwen38Dense && mode == 1;
        self.tree_nodes = if self.wide {
            dflash_tree_nodes(
                std::env::var("MLXL3_DFLASH_TREE").ok().as_deref(),
                std::env::var("MLXL3_DFLASH_TREE_NODES").ok().as_deref(),
            )
        } else {
            0
        };
        self.tree_sampled = self.tree_nodes > 0
            && dflash_tree_sampled(std::env::var("MLXL3_DFLASH_TREE_SAMPLED").ok().as_deref());
        self.relax_kl = dflash_relax_kl(std::env::var("MLXL3_DFLASH_RELAX_KL").ok().as_deref());
        self.copy = (mode == 1 && dflash_context_copy_enabled(Some(family)))
            .then(|| mlxl3_native::speculative::DFlashCopy::new(prompt));
    }

    fn proposals_count(&self, context_remaining: usize, output_remaining: usize) -> Option<usize> {
        let (accepted, proposed) = self
            .recent
            .iter()
            .fold((0, 0), |(accepted, proposed), &(a, p)| {
                (accepted + a, proposed + p)
            });
        mlxl3_native::speculative::dflash_proposals(
            if self.wide { 3 } else { self.mode },
            context_remaining,
            output_remaining,
            self.recent.len(),
            accepted,
            proposed,
        )
        .map(|width| self.width_cap.map_or(width, |cap| width.min(cap)))
    }

    fn prefill(
        &mut self,
        target: &mut mlxl3_native::qwen35::Qwen35Moe,
        draft: &mlxl3_native::dflash::DFlashWeights,
        tokens: &[u32],
    ) -> Result<mlxl3_native::array::Array> {
        let position = target.offset();
        let (logits, captured) = if tokens.len() < 24 {
            target.prefill_serial_with_dflash_capture(tokens)?
        } else {
            target.forward_tokens_with_dflash_capture(tokens)?
        };
        self.cache.append_captured(draft, &captured, position)?;
        self.cache.eval()?;
        Ok(logits)
    }

    /// Greedy block (tests; production passes the request policy to `advance_sampled`).
    #[cfg(test)]
    fn advance(
        &mut self,
        target: &mut mlxl3_native::qwen35::Qwen35Moe,
        draft: &mlxl3_native::dflash::DFlashWeights,
        anchor: u32,
        context_limit: usize,
        output_remaining: usize,
    ) -> Result<u32> {
        self.advance_sampled(target, draft, anchor, context_limit, output_remaining, None)
    }

    /// Q27D-09 (D2): `advance` under a non-greedy target policy. The draft
    /// still proposes deterministically; the target then verifies with the
    /// exact speculative rejection sampling MTP uses (`crate::sampling`).
    /// A greedy policy keeps the greedy verification unchanged.
    fn advance_sampled(
        &mut self,
        target: &mut mlxl3_native::qwen35::Qwen35Moe,
        draft: &mlxl3_native::dflash::DFlashWeights,
        anchor: u32,
        context_limit: usize,
        output_remaining: usize,
        draw: Option<mlxl3_native::sampling::Draw<'_>>,
    ) -> Result<u32> {
        let mut draw = draw.filter(|draw| !draw.sampler.is_greedy());
        anyhow::ensure!(output_remaining > 0, "DFlash reached the output limit");
        if let Some(token) = self.pending.pop_front() {
            return Ok(token);
        }
        let remaining = context_limit.saturating_sub(usize::try_from(target.offset())?);
        let proposals_count = self
            .proposals_count(remaining, output_remaining)
            .context("DFlash reached the context limit")?;
        if proposals_count == 0 {
            // This is the last deliverable token. No later draft needs captures.
            let logits = target.forward(anchor)?;
            if let Some(draw) = draw.as_mut() {
                return draw.sample_last_row(&logits);
            }
            return logits
                .chat_greedy_ids()?
                .into_iter()
                .next()
                .context("model returned no logits");
        }
        let block_started = Instant::now();
        let copied = self.copy.as_ref().and_then(|copy| {
            copy.propose_up_to(anchor, proposals_count, if self.wide { 7 } else { 3 })
        });
        let is_copy = copied.is_some();
        if !is_copy
            && (draw.is_none() || self.tree_sampled)
            && self.tree_nodes > 0
            && proposals_count == 7
            && target.supports_tree_verification()?
        {
            return self.advance_tree(target, draft, anchor, block_started, draw);
        }
        let proposals = if let Some(tokens) = copied {
            tokens.to_vec()
        } else {
            let input = target.dflash_input(anchor, draft.family.mask())?;
            let output = draft.forward_hidden(&input, &self.cache, target.offset())?;
            let draft_logits = target.dflash_logits(&output.hidden, proposals_count)?;
            draft.select_greedy(&draft_logits, &output.selector, anchor)?
        };
        let verify = std::iter::once(anchor)
            .chain(proposals.iter().copied())
            .collect::<Vec<_>>();
        let position = target.offset();
        let (target_logits, captured) = target.verify_tokens_exact_with_dflash_capture(&verify)?;
        let acceptance = if let Some(draw) = draw.as_mut() {
            draw.speculative_accept_logits(&proposals, &target_logits)
                .context("invalid DFlash sampled verification")?
        } else {
            let target_tokens = target_logits.chat_greedy_ids()?;
            mlxl3_native::speculative::greedy_accept(&proposals, &target_tokens)
                .context("invalid DFlash target verification")?
        };
        let accepted = acceptance.accepted_draft_tokens;
        self.proposed += proposals.len();
        self.accepted += accepted;
        self.blocks += 1;
        self.recent.push_back((accepted, proposals.len()));
        if self.recent.len() > 8 {
            self.recent.pop_front();
        }
        let retained = accepted + 1;
        target.commit_dflash_verification(retained, verify.len())?;
        self.cache.append_captured(
            draft,
            &captured.slice(0, 0, i32::try_from(retained)?)?,
            position,
        )?;
        if let Some(copy) = &mut self.copy {
            copy.record(anchor, &proposals[..accepted], is_copy);
        }
        self.lookup_blocks += usize::from(is_copy);
        self.pending.extend(proposals[..accepted].iter().copied());
        self.pending.push_back(acceptance.target_token);
        // Wall time through materialized token selection, without extra GPU
        // barriers. Deferred cache/commit work may be paid by the next block.
        self.block_seconds += block_started.elapsed().as_secs_f64();
        self.pending
            .pop_front()
            .context("DFlash produced no target token")
    }

    /// Q27D-37: one greedy block verified as a best-first tree over the selector lattice.
    fn advance_tree(
        &mut self,
        target: &mut mlxl3_native::qwen35::Qwen35Moe,
        draft: &mlxl3_native::dflash::DFlashWeights,
        anchor: u32,
        block_started: Instant,
        draw: Option<mlxl3_native::sampling::Draw<'_>>,
    ) -> Result<u32> {
        // Q27D-38: diagnostic block anatomy, synchronized at each phase only when requested.
        let profile = std::env::var_os("MLXL3_DFLASH_PROFILE");
        let mut marks = Vec::with_capacity(5);
        let mut mark = |sync: bool| -> Result<()> {
            if profile.is_some() {
                if sync {
                    mlxl3_native::array::synchronize()?;
                }
                marks.push(block_started.elapsed().as_secs_f64() * 1e3);
            }
            Ok(())
        };
        let input = target.dflash_input(anchor, draft.family.mask())?;
        let output = draft.forward_hidden(&input, &self.cache, target.offset())?;
        let draft_logits = target.dflash_logits(&output.hidden, 7)?;
        mark(true)?;
        let (candidates, unary, edges) =
            draft.select_lattice(&draft_logits, &output.selector, anchor)?;
        mark(true)?;
        let tree = mlxl3_native::speculative::build_selector_tree(
            anchor,
            &candidates,
            &unary,
            &edges,
            7,
            self.tree_nodes,
            DFLASH_TREE_EDGE_SCALE,
            DFLASH_TREE_TAU,
        )
        .context("invalid DFlash verification tree")?;
        mark(false)?;
        let position = target.offset();
        let (target_logits, captured) = target.verify_tree_with_dflash_capture(&tree)?;
        let (path, accepted, bonus) = if let Some(draw) = draw {
            // Q27D-40: exact speculative sampling down the tree.
            let rows = mlxl3_native::sampling::log_prob_rows(&target_logits)?;
            anyhow::ensure!(
                rows.len() == tree.rows(),
                "invalid DFlash sampled tree verification output"
            );
            let (sampler, history) = (draw.sampler, draw.history);
            let (path, accepted, bonus, terms) = tree
                .accept_sampled_relaxed(
                    |row, extra| sampler.distribution(rows[row].clone(), history, extra),
                    draw.random,
                    self.relax_kl,
                )
                .context("invalid DFlash sampled tree verification")?;
            // Q27D-48: expected acceptance and KL for several budgets on this block's laws.
            if let Some(file) = std::env::var_os("MLXL3_DFLASH_RELAX_CURVE") {
                use std::io::Write as _;
                let laws = (0..tree.rows())
                    .map(|row| {
                        let mut extra = Vec::new();
                        let mut node = row;
                        while node != 0 {
                            extra.push(tree.tokens[node]);
                            node = tree.parents[node];
                        }
                        extra.reverse();
                        sampler.distribution(rows[row].clone(), history, &extra)
                    })
                    .collect::<Result<Vec<_>>>()?;
                let kappas = [0., 0.0025, 0.005, 0.01, 0.02, 0.05];
                let curve = kappas
                    .iter()
                    .map(|&kappa| tree.expected_acceptance(&laws, kappa))
                    .collect::<Result<Vec<_>>>()?;
                let mut out = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(file)?;
                writeln!(
                    out,
                    "{}",
                    json!({"kappas": kappas, "expected_accepted": curve.iter().map(|c| c.0).collect::<Vec<_>>(),
                        "expected_kl": curve.iter().map(|c| c.1).collect::<Vec<_>>(), "rows": tree.rows()})
                )?;
            }
            if let Some(file) = std::env::var_os("MLXL3_DFLASH_RELAX_LOG") {
                use std::io::Write as _;
                let mut out = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(file)?;
                writeln!(
                    out,
                    "{}",
                    json!({"kl": terms.0, "cross_entropy": terms.1, "entropy": terms.2,
                        "tokens": accepted + 1, "accepted": accepted, "relax_kl": self.relax_kl})
                )?;
            }
            (path, accepted, bonus)
        } else {
            let target_tokens = target_logits.chat_greedy_ids()?;
            tree.accept_greedy(&target_tokens)
                .context("invalid DFlash tree verification")?
        };
        mark(true)?;
        self.proposed += tree.rows() - 1;
        self.accepted += accepted;
        self.blocks += 1;
        self.recent.push_back((accepted, 7));
        if self.recent.len() > 8 {
            self.recent.pop_front();
        }
        target.commit_dflash_tree(&tree, &path)?;
        let rows = path
            .iter()
            .map(|&row| i32::try_from(row))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let rows = mlxl3_native::array::Array::from_i32(&rows, &[i32::try_from(rows.len())?])?;
        self.cache
            .append_captured(draft, &captured.take(&rows, 0)?, position)?;
        let tokens = path[1..]
            .iter()
            .map(|&row| tree.tokens[row])
            .collect::<Vec<_>>();
        if let Some(copy) = &mut self.copy {
            copy.record(anchor, &tokens, false);
        }
        self.pending.extend(tokens);
        self.pending.push_back(bonus);
        if let Some(file) = &profile {
            mark(true)?;
            use std::io::Write as _;
            let mut out = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(file)?;
            writeln!(
                out,
                "{}",
                json!({"draft_ms": marks[0], "lattice_ms": marks[1] - marks[0],
                    "tree_ms": marks[2] - marks[1], "verify_ms": marks[3] - marks[2],
                    "commit_ms": marks[4] - marks[3], "total_ms": marks[4],
                    "rows": tree.rows(), "paths": tree.paths().len(), "accepted": accepted,
                    "position": position})
            )?;
        }
        self.block_seconds += block_started.elapsed().as_secs_f64();
        self.pending
            .pop_front()
            .context("DFlash tree produced no target token")
    }
}

/// Q27D-36/37: selector-tree scoring (TensorFold's weights), fitted offline on the 27B.
#[cfg(all(feature = "mlx", feature = "chat"))]
const DFLASH_TREE_EDGE_SCALE: f32 = 0.6;
#[cfg(all(feature = "mlx", feature = "chat"))]
const DFLASH_TREE_TAU: f32 = 1.0;

/// Q27D-37: the greedy verification tree is on by default in the wide TensorOps mode of the
/// 27B (+21 to +25 % on prose, +5 % on code); `MLXL3_DFLASH_TREE=0` keeps the selector chain.
/// `MLXL3_DFLASH_TREE_NODES` = 7..15 draft nodes (default 11).
/// Q27D-47: divergence budget of the sampled tree, in nats per emitted token; on by default at
/// 0.01 (user decision; +4.4 % expected tokens per block, KL 0.0043 per token measured).
/// `MLXL3_DFLASH_RELAX_KL=0` keeps the exact verification; a value outside [0, 1] keeps the
/// default.
#[cfg(all(feature = "mlx", feature = "chat"))]
const DFLASH_RELAX_KL: f64 = 0.01;

#[cfg(all(feature = "mlx", feature = "chat"))]
fn dflash_relax_kl(value: Option<&str>) -> f64 {
    value
        .and_then(|text| text.parse::<f64>().ok())
        .filter(|budget| budget.is_finite() && (0.0..=1.0).contains(budget))
        .unwrap_or(DFLASH_RELAX_KL)
}

/// Q27D-40: sampled blocks use the tree too (exact speculative sampling; +11.7 % on prose,
/// +13.3 % on code with Desktop's Qwen3.8 preset); `MLXL3_DFLASH_TREE_SAMPLED=0` keeps the chain.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn dflash_tree_sampled(flag: Option<&str>) -> bool {
    flag != Some("0")
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn dflash_tree_nodes(enabled: Option<&str>, nodes: Option<&str>) -> usize {
    if enabled == Some("0") {
        return 0;
    }
    nodes
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| (7..=15).contains(value))
        .unwrap_or(11)
}

/// PRODUCT-2026-11-18: the bridge offers DFlash2 for the dense Qwen3.8-27B only; the
/// Qwen3.6-35B-A3B is faster with MTP, so its draft is opt-in with `MLXL3_DFLASH_35B=1`.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn dflash_family_offered(family: mlxl3_native::dflash::Family, opt_in: Option<&str>) -> bool {
    family != mlxl3_native::dflash::Family::Qwen36Moe || opt_in == Some("1")
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn dflash_context_copy_enabled(family: Option<mlxl3_native::dflash::Family>) -> bool {
    // Q27D-23: on by default for the dense 27B (+9 % on copies, outputs unchanged);
    // the 35B stays opt-in. `MLXL3_DFLASH_CONTEXT_COPY=0` turns it off, `1` turns it on.
    let setting = std::env::var("MLXL3_DFLASH_CONTEXT_COPY");
    match family {
        Some(mlxl3_native::dflash::Family::Qwen38Dense) => setting.as_deref() != Ok("0"),
        Some(mlxl3_native::dflash::Family::Qwen36Moe) => setting.as_deref() == Ok("1"),
        _ => false,
    }
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn dflash_runtime_key(base: &str, copy: bool, cap: Option<usize>) -> String {
    // Q27D-08: a capped adaptive mode is a different policy; older Tune choices stay apart.
    match cap {
        None => format!("{base}:dflash-v4:copy={}", u8::from(copy)),
        Some(cap) => format!("{base}:dflash-v4:copy={}:cap={cap}", u8::from(copy)),
    }
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn unlimited_tokens() -> i64 {
    -1
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn unit_penalty() -> f32 {
    1.0
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
#[derive(Clone, Default, Serialize)]
struct NativeStats {
    ttft_seconds: f64,
    prefill_tps: f64,
    decode_tps: f64,
    prompt_tokens: usize,
    generated_tokens: usize,
    model_size_gb: f64,
    cached_prompt_tokens: usize,
    evaluated_prompt_tokens: usize,
    context_used: usize,
    context_limit: usize,
    elapsed_seconds: f64,
    prepare_seconds: f64,
    prefill_seconds: f64,
    decode_seconds: f64,
    decode_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_gpu_greedy_selections: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    native_gpu_greedy_heads: Option<u64>,
    end_to_end_ttft_seconds: Option<f64>,
    tool_seconds: f64,
    draft_load_seconds: f64,
    tool_rounds: usize,
    round_count: usize,
    memory: Option<mlxl3_native::runtime_stats::MemoryStats>,
    dflash_proposed_tokens: Option<usize>,
    dflash_accepted_tokens: Option<usize>,
    dflash_blocks: Option<usize>,
    dflash_lookup_blocks: Option<usize>,
    dflash_block_seconds: f64,
    dflash_profile: Option<mlxl3_native::runtime_stats::FlashDraftProfile>,
    mtp_proposed_tokens: Option<usize>,
    mtp_accepted_tokens: Option<usize>,
    mtp_blocks: Option<usize>,
    /// Inkling 2 DSpark: drafted tokens, accepted drafts and verified blocks of this round.
    #[serde(skip_serializing_if = "Option::is_none")]
    dspark_proposed_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dspark_accepted_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dspark_blocks: Option<usize>,
    /// Inkling 2 DSpark timings and per-position acceptance of this round.
    #[cfg(feature = "mlx")]
    #[serde(skip_serializing_if = "Option::is_none")]
    dspark_profile: Option<mlxl3_native::inkling2_draft::SpecStats>,
    mtp_lookup_blocks: Option<usize>,
    mtp_depth_blocks: Option<[usize; 4]>,
    /// `[width][accepted]` head cycles; width 0 is a plain target step.
    #[serde(skip_serializing_if = "Option::is_none")]
    mtp_accept_histogram: Option<mlxl3_native::speculative::AcceptHistogram>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mtp_lookup_accept_histogram: Option<mlxl3_native::speculative::AcceptHistogram>,
    /// LFM3-PLD verification blocks, proposed and accepted lookup IDs.
    #[serde(skip_serializing_if = "Option::is_none")]
    lfm3_lookup_blocks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lfm3_lookup_proposed_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lfm3_lookup_accepted_tokens: Option<usize>,
    /// SPEED-2026-11-23-INK2M-MAYA LK: Inkling 2 lookup blocks, proposed and accepted IDs.
    #[serde(skip_serializing_if = "Option::is_none")]
    inkling_lookup_blocks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inkling_lookup_proposed_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    inkling_lookup_accepted_tokens: Option<usize>,
    #[cfg(feature = "mlx")]
    #[serde(skip_serializing_if = "Option::is_none")]
    inkling_lookup_profile: Option<mlxl3_native::inkling2_lookup::LookupStats>,
    /// LFM3-CERT-HEAD decode steps certified by the screening head, and
    /// steps that fell back to the full head.
    #[serde(skip_serializing_if = "Option::is_none")]
    lfm3_cert_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lfm3_cert_fallbacks: Option<usize>,
    /// `MLXL3_LFM3_CERT_CHECK=1`: certified steps whose ID or distribution
    /// differed from the full head.
    #[serde(skip_serializing_if = "Option::is_none")]
    lfm3_cert_check_mismatches: Option<usize>,
    /// Q1 top-k readback (on by default for autoregressive sampling, opt-in
    /// `MLXL3_TOPK_READBACK=1` for MTP rows): sampled rows
    /// `[certified, fallback]` during this round.
    #[serde(skip_serializing_if = "Option::is_none")]
    topk_readback_rows: Option<[usize; 2]>,
    /// Q3 (`MLXL3_MTP_SAMPLED_DRAFT=1`): MTP draft levels drawn from the
    /// head's law during this round.
    #[serde(skip_serializing_if = "Option::is_none")]
    mtp_sampled_draft_levels: Option<usize>,
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
impl NativeStats {
    fn accumulate(&mut self, round: &Self) {
        if let Some(value) = round.native_gpu_greedy_selections {
            *self.native_gpu_greedy_selections.get_or_insert(0) += value;
        }
        if let Some(value) = round.native_gpu_greedy_heads {
            *self.native_gpu_greedy_heads.get_or_insert(0) += value;
        }
        self.prompt_tokens += round.prompt_tokens;
        self.generated_tokens += round.generated_tokens;
        self.cached_prompt_tokens += round.cached_prompt_tokens;
        self.evaluated_prompt_tokens += round.evaluated_prompt_tokens;
        self.prepare_seconds += round.prepare_seconds;
        self.prefill_seconds += round.prefill_seconds;
        self.decode_seconds += round.decode_seconds;
        self.decode_tokens += round.decode_tokens;
        self.prefill_tps = self.evaluated_prompt_tokens as f64 / self.prefill_seconds.max(1e-9);
        self.decode_tps = self.decode_tokens as f64 / self.decode_seconds.max(1e-9);
        self.context_used = round.context_used;
        self.context_limit = round.context_limit;
        self.round_count += round.round_count;
        self.dflash_block_seconds += round.dflash_block_seconds;
        if let Some(profile) = round.dflash_profile {
            let sum = self.dflash_profile.get_or_insert_with(Default::default);
            sum.draft_seconds += profile.draft_seconds;
            sum.verification_seconds += profile.verification_seconds;
            sum.commit_cache_seconds += profile.commit_cache_seconds;
            sum.verified_rows += profile.verified_rows;
            sum.drafted_rows += profile.drafted_rows;
            for (sum, value) in sum.drafted_widths.iter_mut().zip(profile.drafted_widths) {
                *sum += value;
            }
            for (sum, value) in sum.verified_widths.iter_mut().zip(profile.verified_widths) {
                *sum += value;
            }
            sum.expert_demand_bytes += profile.expert_demand_bytes;
            for (sum, value) in sum.ssd_prefetch.iter_mut().zip(profile.ssd_prefetch) {
                *sum = sum.saturating_add(value);
            }
        }
        self.memory = round.memory;
        if let Some(counts) = round.mtp_depth_blocks {
            let sum = self.mtp_depth_blocks.get_or_insert([0; 4]);
            for (total, count) in sum.iter_mut().zip(counts) {
                *total += count;
            }
        }
        for (sum, round) in [
            (&mut self.mtp_accept_histogram, &round.mtp_accept_histogram),
            (
                &mut self.mtp_lookup_accept_histogram,
                &round.mtp_lookup_accept_histogram,
            ),
        ] {
            if let Some(round) = round {
                sum.get_or_insert_with(Default::default).merge(round);
            }
        }
        for (sum, value) in [
            (
                &mut self.dflash_proposed_tokens,
                round.dflash_proposed_tokens,
            ),
            (
                &mut self.dflash_accepted_tokens,
                round.dflash_accepted_tokens,
            ),
            (&mut self.dflash_blocks, round.dflash_blocks),
            (&mut self.dflash_lookup_blocks, round.dflash_lookup_blocks),
            (&mut self.mtp_proposed_tokens, round.mtp_proposed_tokens),
            (&mut self.mtp_accepted_tokens, round.mtp_accepted_tokens),
            (&mut self.mtp_blocks, round.mtp_blocks),
            (&mut self.mtp_lookup_blocks, round.mtp_lookup_blocks),
            (&mut self.lfm3_lookup_blocks, round.lfm3_lookup_blocks),
            (
                &mut self.lfm3_lookup_proposed_tokens,
                round.lfm3_lookup_proposed_tokens,
            ),
            (
                &mut self.lfm3_lookup_accepted_tokens,
                round.lfm3_lookup_accepted_tokens,
            ),
            (&mut self.inkling_lookup_blocks, round.inkling_lookup_blocks),
            (
                &mut self.inkling_lookup_proposed_tokens,
                round.inkling_lookup_proposed_tokens,
            ),
            (
                &mut self.inkling_lookup_accepted_tokens,
                round.inkling_lookup_accepted_tokens,
            ),
            (&mut self.lfm3_cert_tokens, round.lfm3_cert_tokens),
            (&mut self.lfm3_cert_fallbacks, round.lfm3_cert_fallbacks),
            (
                &mut self.lfm3_cert_check_mismatches,
                round.lfm3_cert_check_mismatches,
            ),
            (
                &mut self.mtp_sampled_draft_levels,
                round.mtp_sampled_draft_levels,
            ),
        ] {
            if let Some(value) = value {
                *sum = Some(sum.unwrap_or(0) + value);
            }
        }
        if let Some([certified, fallbacks]) = round.topk_readback_rows {
            let [sum_certified, sum_fallbacks] = self.topk_readback_rows.get_or_insert([0, 0]);
            *sum_certified += certified;
            *sum_fallbacks += fallbacks;
        }
    }
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
struct RoundOutput {
    stop_reason: mlxl3_native::GenerationStopReason,
    raw: String,
    answer: String,
    tool_text: String,
    stats: NativeStats,
    first_text_seconds: Option<f64>,
    token_hash: String,
    /// The generated token ids (a forced end of thinking continues from them).
    generated: Vec<u32>,
    /// The round stopped at its reasoning budget; holds the exact text to resume the model.
    reasoning_cut: Option<String>,
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn token_hash(tokens: &[u32]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn emit_event(value: Value) -> Result<()> {
    let mut stdout = io::stdout().lock();
    serde_json::to_writer(&mut stdout, &value)?;
    writeln!(stdout)?;
    stdout.flush()?;
    Ok(())
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn select_token(
    logits: &mlxl3_native::array::Array,
    history: &mlxl3_native::sampling::History,
    sampler: &mlxl3_native::sampling::Sampler,
    random: &mut u64,
) -> Result<u32> {
    if sampler.is_greedy() {
        return logits
            .chat_greedy_ids()?
            .into_iter()
            .next()
            .context("model returned no logits");
    }
    if mlxl3_native::topk_readback::ar_enabled() {
        let mut rows =
            mlxl3_native::topk_readback::ReadbackRows::new(logits, sampler, history, &[])?;
        let last = rows
            .len()
            .checked_sub(1)
            .context("model returned no logits")?;
        return Ok(rows
            .distribution(last, sampler, history, &[])?
            .sample(random));
    }
    sample_values(logits.log_probs()?.to_f32()?, history, sampler, random)
}

/// Whether a bridge round reports `topk_readback_rows`: a readback gate is on
/// and the sampler is outside the greedy rule (greedy rounds never read back).
#[cfg(all(feature = "mlx", feature = "chat"))]
fn readback_reported(ar: bool, mtp: bool, greedy: bool) -> bool {
    (ar || mtp) && !greedy
}

#[cfg(all(feature = "mlx", feature = "chat"))]
#[test]
fn readback_rows_are_reported_for_sampled_rounds_only() {
    for (ar, mtp, greedy, reported) in [
        (true, false, false, true),
        (false, true, false, true),
        (true, true, false, true),
        (false, false, false, false),
        (true, false, true, false),
        (false, true, true, false),
        (true, true, true, false),
        (false, false, true, false),
    ] {
        assert_eq!(readback_reported(ar, mtp, greedy), reported);
    }
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn sample_values(
    values: Vec<f32>,
    history: &mlxl3_native::sampling::History,
    sampler: &mlxl3_native::sampling::Sampler,
    random: &mut u64,
) -> Result<u32> {
    Ok(sampler.distribution(values, history, &[])?.sample(random))
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn bridge_emit_delta(
    request_id: &str,
    phase: &str,
    text: &str,
    buffered: bool,
    answer: &mut String,
) -> Result<()> {
    if phase == "answer" {
        answer.push_str(text);
    }
    if !buffered {
        emit_event(json!({"type":"delta", "request_id":request_id, "phase":phase, "text":text}))?;
    }
    Ok(())
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
impl BridgeRequest {
    fn greedy_unpenalized(&self) -> bool {
        (self.temperature == 0. || self.top_k == 1)
            && self.repetition_penalty == 1.
            && self.presence_penalty == 0.
    }
    fn sampler(&self) -> Result<mlxl3_native::sampling::Sampler> {
        mlxl3_native::sampling::Sampler::new(
            self.temperature,
            self.top_k,
            self.top_p,
            self.repetition_penalty,
            self.presence_penalty,
        )
    }
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn emit_generation_mode(
    request: &BridgeRequest,
    mtp: bool,
    dflash: bool,
    proposals: usize,
    path: Option<String>,
) -> Result<()> {
    emit_event(json!({
        "type":"generation_mode", "request_id":request.request_id,
        "dflash_requested":request.dflash2,"dflash_active":dflash,
        "mtp_requested":request.mtp, "mtp_active":mtp, "mtp_depth":if mtp { request.mtp_depth } else { 0 },
        "mtp_reason":if request.mtp && !mtp { Some(if cfg!(feature = "mlx") { "MTP is unavailable for this request; ordinary sampling remains active" } else { "MTP uses greedy sampling and repetition penalty 1; ordinary sampling remains active" }) } else { None },
        "mtp_sampling":mtp && !request.greedy_unpenalized(),
        "dflash_proposals":proposals, "dflash_mode":if dflash { request.dflash_mode } else { 0 },
        "dflash_draft_path":path
    }))
}

/// LFM3-PLD gate: `MLXL3_LFM3_LOOKUP=1` or `lines` (the LOOKUP-LINES
/// newline rule), an LFM3 target without other drafters, and the greedy rule
/// of `select_token` (`Sampler::is_greedy`). Returns whether to filter.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn lfm3_lookup_enabled(flag: Option<&str>, plain_lfm3: bool, greedy: bool) -> Option<bool> {
    let lines = match flag {
        Some("1") => false,
        Some("lines") => true,
        _ => return None,
    };
    (plain_lfm3 && greedy).then_some(lines)
}

/// L4+L3 gate: `MLXL3_LFM3_LOOKUP=v2` under the conditions of
/// `lfm3_lookup_enabled`; its steps without a proposal use the certified
/// head under the default gate of `lfm3_cert_enabled` (`MLXL3_LFM3_CERT_HEAD`).
#[cfg(all(feature = "mlx", feature = "chat"))]
fn lfm3_lookup_v2_enabled(flag: Option<&str>, plain_lfm3: bool, greedy: bool) -> bool {
    flag == Some("v2") && plain_lfm3 && greedy
}

#[cfg(all(feature = "mlx", feature = "chat"))]
#[test]
fn lfm3_lookup_v2_is_opt_in_and_greedy() {
    assert!(lfm3_lookup_v2_enabled(Some("v2"), true, true));
    for (flag, plain, greedy) in [
        (None, true, true),
        (Some("1"), true, true),
        (Some("lines"), true, true),
        (Some("V2"), true, true),
        (Some("v2"), false, true),
        (Some("v2"), true, false),
    ] {
        assert!(!lfm3_lookup_v2_enabled(flag, plain, greedy));
    }
    assert_eq!(lfm3_lookup_enabled(Some("v2"), true, true), None);
}

/// LFM3-CERT-HEAD gate: on by default (`MLXL3_LFM3_CERT_HEAD` unset or `1`;
/// any other value, such as `0`, disables it) for an LFM3 target without MTP,
/// draft or prompt lookup, under the greedy rule of `select_token`.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn lfm3_cert_enabled(flag: Option<&str>, plain_lfm3: bool, lookup: bool, greedy: bool) -> bool {
    matches!(flag, None | Some("1")) && plain_lfm3 && !lookup && greedy
}

/// L1 gate: on by default (`MLXL3_LFM3_CERT_SAMPLE` unset or `1`; any other
/// value, such as `0`, disables it) for an LFM3 target without MTP, draft or
/// prompt lookup, a sampler outside the greedy rule of `select_token`, and a
/// kept set the head can certify (`Sampler::certified_k`: penalties that
/// only lower values and `1 <= top_k <= SAMPLE_MAX_K`).
#[cfg(all(feature = "mlx", feature = "chat"))]
fn lfm3_cert_sample_enabled(
    flag: Option<&str>,
    plain_lfm3: bool,
    lookup: bool,
    greedy: bool,
    certified_k: bool,
) -> bool {
    matches!(flag, None | Some("1")) && plain_lfm3 && !lookup && !greedy && certified_k
}

/// L2 screening copy of the greedy head: int4 by default,
/// `MLXL3_LFM3_CERT_SCREEN=int8` keeps the previous int8 screen.
#[cfg(all(feature = "mlx", feature = "chat"))]
fn lfm3_cert_screen(flag: Option<&str>) -> mlxl3_native::lfm2::CertScreenKind {
    if flag == Some("int8") {
        mlxl3_native::lfm2::CertScreenKind::Int8
    } else {
        mlxl3_native::lfm2::CertScreenKind::Int4
    }
}

#[cfg(all(feature = "mlx", feature = "chat"))]
#[test]
fn lfm3_cert_sample_is_default_sampled_and_exclusive() {
    for flag in [None, Some("1")] {
        assert!(lfm3_cert_sample_enabled(flag, true, false, false, true));
    }
    for (flag, plain, lookup, greedy, certified_k) in [
        (Some("0"), true, false, false, true),
        (Some("true"), true, false, false, true),
        (Some("1 "), true, false, false, true),
        (Some(""), true, false, false, true),
        (None, false, false, false, true),
        (None, true, true, false, true),
        (None, true, false, true, true),
        (None, true, false, false, false),
        (Some("1"), false, false, false, true),
        (Some("1"), true, true, false, true),
        (Some("1"), true, false, true, true),
        (Some("1"), true, false, false, false),
    ] {
        assert!(!lfm3_cert_sample_enabled(
            flag,
            plain,
            lookup,
            greedy,
            certified_k
        ));
    }
    use mlxl3_native::lfm2::CertScreenKind;
    assert_eq!(lfm3_cert_screen(Some("int8")), CertScreenKind::Int8);
    for flag in [
        None,
        Some(""),
        Some("int4"),
        Some("INT8"),
        Some("int8 "),
        Some("0"),
    ] {
        assert_eq!(lfm3_cert_screen(flag), CertScreenKind::Int4);
    }
}

/// The L1 gate follows the Desktop presets: the default (T 0.2, top-k 80,
/// repetition 1.05) and LFM3 (T 0.1, top-k 50, repetition 1.1) presets are
/// certifiable; top-k 0, above `SAMPLE_MAX_K` or a negative presence penalty
/// are not.
#[cfg(all(feature = "mlx", feature = "chat"))]
#[test]
fn lfm3_cert_sample_gate_covers_the_app_presets() {
    use mlxl3_native::sampling::Sampler;
    for (temperature, top_k, top_p, repetition, presence, k) in [
        (0.2, 80, 1., 1.05, 0., Some(80)),
        (0.1, 50, 1., 1.1, 0., Some(50)),
        (0.7, 256, 0.9, 1., 0., Some(256)),
        (0.7, 257, 0.9, 1., 0., None),
        (0.7, 0, 0.9, 1., 0., None),
        (0.7, 40, 1., 1., -0.5, None),
        (0.7, 40, 1., 0.9, 0., None),
        (0., 40, 1., 1.1, 0., Some(1)),
    ] {
        let sampler = Sampler::new(temperature, top_k, top_p, repetition, presence).unwrap();
        assert_eq!(sampler.certified_k(256), k, "{temperature} {top_k}");
    }
}

#[cfg(all(feature = "mlx", feature = "chat"))]
#[test]
fn lfm3_cert_head_is_default_greedy_and_exclusive() {
    assert!(lfm3_cert_enabled(None, true, false, true));
    assert!(lfm3_cert_enabled(Some("1"), true, false, true));
    for (flag, plain, lookup, greedy) in [
        (Some("0"), true, false, true),
        (Some("true"), true, false, true),
        (Some("1 "), true, false, true),
        (Some(""), true, false, true),
        (None, false, false, true),
        (None, true, true, true),
        (None, true, false, false),
    ] {
        assert!(!lfm3_cert_enabled(flag, plain, lookup, greedy));
    }
}

/// The gates use exactly the greedy rule of `select_token`, presence penalty
/// included.
#[cfg(all(feature = "mlx", feature = "chat"))]
#[test]
fn lfm3_gates_follow_the_select_token_greedy_rule() {
    use mlxl3_native::sampling::Sampler;
    for (temperature, top_k, top_p, repetition, presence, greedy) in [
        (0., 40, 1., 1., 0., true),
        (0.7, 1, 0.8, 1., 0., true),
        (0.2, 40, 1., 1., 0., false),
        (0., 1, 1., 1.1, 0., false),
        (0., 1, 1., 1., 0.5, false),
        (0., 1, 1., 1., -0.5, false),
    ] {
        let sampler = Sampler::new(temperature, top_k, top_p, repetition, presence).unwrap();
        assert_eq!(sampler.is_greedy(), greedy);
        assert_eq!(
            lfm3_cert_enabled(None, true, false, sampler.is_greedy()),
            greedy
        );
        assert_eq!(
            lfm3_lookup_enabled(Some("1"), true, sampler.is_greedy()).is_some(),
            greedy
        );
    }
}

#[cfg(all(feature = "mlx", feature = "chat"))]
#[test]
fn lfm3_lookup_is_opt_in_greedy_and_exclusive() {
    assert_eq!(lfm3_lookup_enabled(Some("1"), true, true), Some(false));
    assert_eq!(lfm3_lookup_enabled(Some("lines"), true, true), Some(true));
    for (flag, plain, greedy) in [
        (None, true, true),
        (Some("0"), true, true),
        (Some("true"), true, true),
        (Some("1"), false, true),
        (Some("lines"), false, true),
        (Some("1"), true, false),
    ] {
        assert_eq!(lfm3_lookup_enabled(flag, plain, greedy), None);
    }
}

/// Image input layout of the loaded chat model (`None`: text only).
#[cfg(all(feature = "mlx", feature = "chat"))]
fn chat_image_format(model: &NativeChatModel) -> Option<mlxl3_native::vision_input::ImageFormat> {
    model.image_format()
}

/// Image input layout of the loaded chat model (`None`: text only); the direct engine reads LFM3 only.
#[cfg(all(
    not(feature = "mlx"),
    feature = "chat",
    target_os = "macos",
    feature = "direct-metal"
))]
fn chat_image_format(model: &NativeChatModel) -> Option<mlxl3_native::vision_input::ImageFormat> {
    model
        .supports_images()
        .then_some(mlxl3_native::vision_input::ImageFormat::Lfm3)
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
#[allow(clippy::too_many_arguments)]
fn bridge_generate_round(
    model: &mut NativeChatModel,
    tokenizer: &mlxl3_native::tokenizer::ChatTokenizer,
    request_id: &str,
    messages: &[Value],
    tools: &[Value],
    max_tokens: i64,
    sampler: mlxl3_native::sampling::Sampler,
    context_limit: usize,
    resident_gb: f64,
    cancelled: &AtomicBool,
    random: &mut u64,
    draft: Option<&RoundDraft>,
    mut mtp: Option<&mut ChatMtpHead>,
    mtp_depth: usize,
    dflash_mode: usize,
    cache: &mut Option<PromptCache>,
    conversation: &str,
    reuse: bool,
    buffer_output: bool,
    continuation: &[u32],
    reasoning_budget: Option<usize>,
) -> Result<RoundOutput> {
    use mlxl3_native::{streaming::Channel, tool_call::StreamFilter};
    let started = Instant::now();
    #[cfg(not(feature = "mlx"))]
    let device_greedy = tools.is_empty()
        && (!buffer_output || model.is_flash())
        && sampler.is_greedy()
        && model.device_greedy_eligible(
            sampler.temperature,
            sampler.top_k,
            sampler.repetition_penalty,
            mtp.is_some() || draft.is_some(),
        );
    #[cfg(not(feature = "mlx"))]
    let device_greedy_before = model.device_greedy_counts();
    anyhow::ensure!(!messages.is_empty(), "messages cannot be empty");
    anyhow::ensure!(
        max_tokens == -1 || max_tokens > 0,
        "max_tokens must be positive, or -1"
    );
    let image_format = chat_image_format(model);
    let (prepared, images) =
        mlxl3_native::vision_input::prepare_messages_as(messages, image_format)?;
    let rendered = REASONING_REQUESTED.with(|effort| {
        tokenizer.render_values_with_effort(
            &Value::Array(prepared),
            Some(tools),
            effort.borrow().as_deref(),
        )
    })?;
    let rendered =
        mlxl3_native::vision_input::expand_image_placeholders(image_format, rendered, &images)?;
    let mut tokens = tokenizer.encode(&rendered)?;
    anyhow::ensure!(!tokens.is_empty(), "chat template produced no tokens");
    // AGENT-2026-11-23-BOUCLE: the same assistant message resumed after a forced end of thinking
    tokens.extend_from_slice(continuation);
    anyhow::ensure!(
        tokens.len() < context_limit,
        "conversation exceeds model context"
    );
    emit_event(json!({
        "type":"context_usage", "request_id":request_id,
        "used_tokens":tokens.len(), "context_limit":context_limit
    }))?;
    let available = context_limit - tokens.len();
    let budget = if max_tokens == -1 {
        available
    } else {
        usize::try_from(max_tokens)?.min(available)
    };
    let reasoning_limit = reasoning_budget
        .and_then(|limit| mlxl3_native::agent_loop::step_reasoning_budget(limit, budget));
    let _prefill_progress = PrefillProgress::set(request_id);
    #[cfg(feature = "mlx")]
    if let NativeChatModel::Flash(target) = &mut *model
        && let Some(plan) = target.admit_request(tokens.len() + budget)?
    {
        emit_event(
            json!({"type":"flash_memory_plan", "request_id":request_id, "ar":plan,
            "memory":target.memory_ledger(None)?}),
        )?;
    }
    #[cfg(not(feature = "mlx"))]
    if let Some(plan) = direct_bridge::admit_request(model, tokens.len() + budget, cache, reuse)? {
        emit_event(
            json!({"type":"flash_memory_plan","request_id":request_id,"ar":plan,"memory":model.memory_ledger(None)?}),
        )?;
    }
    // DSPARK-INK2M-P2: Inkling 2 drafter, exact for greedy and sampled rows. FEAT-2026-11-25-INK2M-SPEC-TOOLS:
    // tool rounds too (as the prompt lookup since AGENT-2026-11-23-VITESSE V1): every token still goes through
    // the same stream filter and stops.
    #[cfg(feature = "mlx")]
    let mut dspark = match &*model {
        NativeChatModel::Inkling(target, Some(drafter), _)
            if mtp.is_none() && draft.is_none() && DSPARK_REQUESTED.with(std::cell::Cell::get) =>
        {
            // DFLASH2-2026-11-24-INK2M-P0: DFlash2 drafts up to its whole block (7), DSpark keeps width 3
            let dflash2 = matches!(
                drafter.as_ref(),
                mlxl3_native::inkling2_draft::InklingDrafter::Dflash2(_)
            );
            let width = if dflash2 {
                mlxl3_native::inkling2_draft::BlockDrafter::max_proposals(drafter.as_ref())
            } else {
                mlxl3_native::inkling2_draft::DsparkSession::DEFAULT_WIDTH
            };
            let session =
                mlxl3_native::inkling2_draft::DsparkSession::new(drafter.as_ref(), width)?;
            let session = if !dflash2 && inkling_sampled_drafts(&sampler) {
                session.with_sampled_drafts(sampler.temperature, draft_noise_seed(*random))?
            } else {
                session
            };
            // SPEED-INK2M-P14: width-0 blocks draw on the GPU, one step ahead
            let session = match inkling_device_draw(
                &sampler,
                std::env::var("MLXL3_INKLING_GPU_DRAW").ok().as_deref(),
                std::env::var("MLXL3_INKLING_PIPELINE").ok().as_deref(),
            ) {
                Some((rule, pipelined)) => {
                    session.with_device_draw(rule, inkling_draw_seeds(*random), pipelined)
                }
                None => session,
            };
            // DSPARK-2026-11-24-INK2M-MAYA2-D5: the prompt lookup in the same controller, under the lookup's own
            // request field and flag (on by default)
            let session = if inkling_lookup_requested(
                LOOKUP_REQUESTED.with(std::cell::Cell::get),
                std::env::var("MLXL3_INKLING_LOOKUP").ok().as_deref(),
            ) {
                session.with_lookup(&tokens)
            } else {
                session
            };
            // DSPARK-INK2M-P4: adaptive width by default; `MLXL3_INKLING_DSPARK_ADAPTIVE=0` keeps the fixed width
            // (3 for DSpark, the whole block of 7 for DFlash2)
            Some(
                if std::env::var("MLXL3_INKLING_DSPARK_ADAPTIVE").as_deref() != Ok("0") {
                    let policy = if dflash2 {
                        // DFLASH2-2026-11-25-INK2M-F2/F3: the measured table (`=f2`, `=p2`: the earlier ones)
                        let policy = mlxl3_native::inkling2_draft::WidthPolicy::measured(
                            mlxl3_native::inkling2_draft::WidthPolicy::dflash2_costs(
                                std::env::var("MLXL3_INKLING_DFLASH2_COSTS").ok().as_deref(),
                            ),
                            width,
                        )?;
                        // DFLASH2-2026-11-24-INK2M-P3 (3): full-width probes rejected as the default (long
                        // prompts lose ≈ 10 %); `MLXL3_INKLING_DFLASH2_PROBE=full` keeps them as an option
                        // DFLASH2-2026-11-25-INK2M-F3: probes in a row after a kept probe
                        let policy =
                            if mlxl3_native::inkling2_draft::WidthPolicy::probe_streak_setting(
                                std::env::var("MLXL3_INKLING_DFLASH2_STREAK")
                                    .ok()
                                    .as_deref(),
                            ) {
                                policy.with_probe_streak()
                            } else {
                                policy
                            };
                        if std::env::var("MLXL3_INKLING_DFLASH2_PROBE").as_deref() == Ok("full") {
                            policy.with_full_probes()
                        } else {
                            policy
                        }
                    } else {
                        inkling_width_policy(
                            std::env::var("MLXL3_INKLING_DSPARK_POLICY").ok().as_deref(),
                            target.precision().dense_bits,
                            sampler.is_greedy(),
                            mlxl3_native::inkling2_draft::DsparkSession::DEFAULT_WIDTH,
                        )?
                    };
                    session.with_adaptive_width(policy)?
                } else {
                    session
                },
            )
        }
        _ => None,
    };
    // SPEED-2026-11-23-INK2M-MAYA LK: prompt lookup without a drafter, plain steps drawn on the GPU
    #[cfg(feature = "mlx")]
    let mut inkling_lookup = match (
        &*model,
        inkling_device_draw(
            &sampler,
            std::env::var("MLXL3_INKLING_GPU_DRAW").ok().as_deref(),
            std::env::var("MLXL3_INKLING_PIPELINE").ok().as_deref(),
        ),
    ) {
        // AGENT-2026-11-23-VITESSE V1: tool rounds too (the Agent mode writes code and edits,
        // where the lookup pays most); every token still goes through the same stream filter.
        (NativeChatModel::Inkling(target, ..), Some((rule, pipelined)))
            if dspark.is_none()
                && mtp.is_none()
                && draft.is_none()
                && inkling_lookup_requested(
                    LOOKUP_REQUESTED.with(std::cell::Cell::get),
                    std::env::var("MLXL3_INKLING_LOOKUP").ok().as_deref(),
                ) =>
        {
            Some(mlxl3_native::inkling2_lookup::LookupSession::new(
                target,
                &tokens,
                rule,
                inkling_draw_seeds(*random),
                pipelined,
            ))
        }
        _ => None,
    };
    let prepare_seconds = started.elapsed().as_secs_f64();
    let prefill_started = Instant::now();
    #[cfg(feature = "mlx")]
    let (mut logits, mut dflash, cached, mut mtp_session) = {
        let (logits, mut dflash, cached) =
            if let (Some(session), NativeChatModel::Inkling(target, Some(drafter), _)) =
                (dspark.as_mut(), &mut *model)
            {
                // DSPARK-2026-11-24-INK2M-MAYA2-D1: the prompt cache with DSpark
                let (logits, cached) = prefill_dspark_round(
                    session,
                    target,
                    drafter.as_mut(),
                    &tokens,
                    &images,
                    cache,
                    conversation,
                    reuse,
                    cancelled,
                )?;
                (logits, None, cached)
            } else if let Some(head) = mtp.as_deref_mut() {
                match head {
                    ChatMtpHead::Qwen(head) => {
                        prefill_mtp(model, head, &tokens, cache, conversation, reuse, cancelled)?
                    }
                    ChatMtpHead::Flash(head) => {
                        prefill_flash_mtp(model, head, &tokens, cache, cancelled)?
                    }
                }
            } else {
                prefill_round(
                    model,
                    &tokens,
                    &images,
                    draft,
                    cache,
                    conversation,
                    reuse,
                    cancelled,
                )?
            };
        if let Some((session, weights)) = dflash.as_mut().zip(draft) {
            session.configure(dflash_mode, weights.family, &tokens);
        }
        let mut mtp_session = mtp
            .as_ref()
            .map(|head| ChatMtpSession::new(head, mtp_depth))
            .transpose()?;
        if std::env::var("MLXL3_MTP_ADAPTIVE").as_deref() == Ok("1")
            && let Some(session) = &mut mtp_session
        {
            session.enable_adaptive();
        }
        if std::env::var("MLXL3_MTP_LOOKUP").as_deref() == Ok("1")
            && let Some(session) = &mut mtp_session
        {
            session.enable_prompt_lookup(&tokens);
        }
        (logits, dflash, cached, mtp_session)
    };
    // LFM3-PLD: opt-in, greedy only; the target verifies every proposal exactly.
    #[cfg(feature = "mlx")]
    let lookup_flag = std::env::var("MLXL3_LFM3_LOOKUP").ok();
    #[cfg(feature = "mlx")]
    let plain_lfm3 = matches!(model, NativeChatModel::Lfm3(_)) && mtp.is_none() && draft.is_none();
    #[cfg(feature = "mlx")]
    let mut lfm3_lookup =
        lfm3_lookup_enabled(lookup_flag.as_deref(), plain_lfm3, sampler.is_greedy())
            .map(|lines| (mlxl3_native::lfm3::PromptLookup::new(&tokens), lines));
    // L4+L3: whole-context lookup composed with the certified head.
    #[cfg(feature = "mlx")]
    let mut lfm3_lookup_certified = false;
    #[cfg(feature = "mlx")]
    if lfm3_lookup_v2_enabled(lookup_flag.as_deref(), plain_lfm3, sampler.is_greedy()) {
        let mut session = mlxl3_native::lfm3::PromptLookup::new_v2(&tokens);
        if lfm3_cert_enabled(
            std::env::var("MLXL3_LFM3_CERT_HEAD").ok().as_deref(),
            plain_lfm3,
            false,
            sampler.is_greedy(),
        ) && let NativeChatModel::Lfm3(target) = &mut *model
            && target.enable_certified_head()?
        {
            session.use_certified_head();
            lfm3_lookup_certified = true;
        }
        lfm3_lookup = Some((session, false));
    }
    // LFM3-CERT-HEAD: default on, greedy only; `[certified, fallback]` steps.
    // `[certified, fallback, check mismatch]` steps.
    #[cfg(feature = "mlx")]
    let mut lfm3_cert = None;
    // L1: the same counters for sampled steps.
    #[cfg(feature = "mlx")]
    let mut lfm3_sample = false;
    #[cfg(feature = "mlx")]
    let lfm3_check = std::env::var("MLXL3_LFM3_CERT_CHECK").as_deref() == Ok("1");
    // Q1: process-wide counters, reported as this round's difference for
    // rounds that may read back rows (greedy rounds never do).
    #[cfg(feature = "mlx")]
    let readback_start = readback_reported(
        mlxl3_native::topk_readback::ar_enabled(),
        mlxl3_native::topk_readback::mtp_enabled(),
        sampler.is_greedy(),
    )
    .then(mlxl3_native::topk_readback::counters);
    #[cfg(feature = "mlx")]
    let sampled_draft_start = mlxl3_native::sampling::sampled_draft_enabled()
        .then(mlxl3_native::sampling::sampled_draft_levels);
    #[cfg(feature = "mlx")]
    if lfm3_cert_enabled(
        std::env::var("MLXL3_LFM3_CERT_HEAD").ok().as_deref(),
        matches!(model, NativeChatModel::Lfm3(_)) && mtp.is_none() && draft.is_none(),
        lfm3_lookup.is_some(),
        sampler.is_greedy(),
    ) && let NativeChatModel::Lfm3(target) = &mut *model
        && target.enable_certified_head_kind(lfm3_cert_screen(
            std::env::var("MLXL3_LFM3_CERT_SCREEN").ok().as_deref(),
        ))?
    {
        lfm3_cert = Some([0usize; 3]);
    } else if lfm3_cert_sample_enabled(
        std::env::var("MLXL3_LFM3_CERT_SAMPLE").ok().as_deref(),
        matches!(model, NativeChatModel::Lfm3(_)) && mtp.is_none() && draft.is_none(),
        lfm3_lookup.is_some(),
        sampler.is_greedy(),
        sampler
            .certified_k(mlxl3_native::lfm2::CERT_SAMPLE_MAX_K)
            .is_some(),
    ) && let NativeChatModel::Lfm3(target) = &mut *model
        && target.enable_certified_head()?
    {
        lfm3_cert = Some([0usize; 3]);
        lfm3_sample = true;
    }
    #[cfg(not(feature = "mlx"))]
    let (mut logits, cached, mut mtp_session) = {
        let _ = dflash_mode;
        let _ = images;
        direct_bridge::prefill(
            model,
            mtp.as_deref_mut(),
            draft,
            &tokens,
            mtp_depth,
            cache,
            conversation,
            reuse,
            cancelled,
        )?
    };
    // SPEED-INK2M-P14: plain Inkling decoding chooses its tokens on the GPU, one step ahead
    #[cfg(feature = "mlx")]
    let mut inkling_decode = match (
        &*model,
        inkling_device_draw(
            &sampler,
            std::env::var("MLXL3_INKLING_GPU_DRAW").ok().as_deref(),
            std::env::var("MLXL3_INKLING_PIPELINE").ok().as_deref(),
        ),
    ) {
        // AGENT-2026-11-23-VITESSE V1: tool rounds draw on the GPU as well
        (NativeChatModel::Inkling(..), Some((rule, pipelined)))
            if dspark.is_none() && inkling_lookup.is_none() && mtp.is_none() && draft.is_none() =>
        {
            Some(mlxl3_native::inkling2::DeviceDecode::start(
                &logits,
                rule,
                inkling_draw_seeds(*random),
                pipelined,
            )?)
        }
        _ => None,
    };
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
    #[cfg(feature = "mlx")]
    let gpu_capture = mlxl3_native::array::capture_for_request(request_id)?;
    let mut decoder = tokenizer.tokenizer().decode_stream(false);
    let mut splitter = ThinkingSplitter::new(&rendered);
    if !continuation.is_empty() {
        splitter.resume_answer();
    }
    let mut thinking_tokens = 0usize;
    let mut reasoning_cut = None;
    let mut closing_plan: Option<(&str, String, usize)> = None;
    let summarizing = buffer_output
        && messages
            .last()
            .is_some_and(|message| message["content"] == mlxl3_native::agent_loop::SUMMARY_REQUEST);
    // The prompt says which call format the model will use: Inkling 2 declares its tools
    // in a `tool_declare` system message, and may write text before a call.
    let inkling_tools = rendered.contains("<|message_system|>tool_declare<|content_xml|>");
    let mut filter = (!tools.is_empty()).then(|| {
        if inkling_tools {
            StreamFilter::inkling()
        } else {
            StreamFilter::new()
        }
    });
    let mut last_tool_progress: Option<Instant> = None;
    let mut decoded = String::new();
    let mut answer = String::new();
    let mut tool_text = String::new();
    let mut generated = Vec::new();
    let mut history = mlxl3_native::sampling::History::default();
    let mut first_token = None;
    let mut first_visible_text = None;
    let mut dflash_next = None;
    #[cfg(not(feature = "mlx"))]
    let mut device_greedy_next = None;
    #[cfg(not(feature = "mlx"))]
    let mut mtp_pending = std::collections::VecDeque::new();
    let mut eos_observed = false;
    for _ in 0..budget {
        if cancelled.load(Ordering::Relaxed) {
            model.reset();
            bail!("generation cancelled");
        }
        let pending = dflash_next.take();
        #[cfg(not(feature = "mlx"))]
        let pending = mtp_pending.pop_front().or(pending);
        let next = if let Some(token) = pending {
            token
        } else {
            #[cfg(feature = "mlx")]
            let sampled = if let (Some(decode), NativeChatModel::Inkling(target, ..)) =
                (inkling_decode.as_mut(), &mut *model)
            {
                // the step consuming this token runs when one more token will be delivered
                decode.next_token(target, generated.len() + 1 < budget)?
            } else {
                select_token(&logits, &history, &sampler, random)?
            };
            #[cfg(not(feature = "mlx"))]
            let sampled = if device_greedy {
                match device_greedy_next.take() {
                    Some(token) => token,
                    None => model.greedy_from_logits(&logits)?,
                }
            } else {
                direct_bridge::select(model, &logits, &history, &sampler, random)?
            };
            sampled
        };
        if tokenizer.eos_ids().contains(&next) {
            eos_observed = true;
            break;
        }
        first_token.get_or_insert_with(|| started.elapsed().as_secs_f64());
        generated.push(next);
        history.push(next);
        if let Some(text) = decoder
            .step(next)
            .map_err(|error| anyhow::anyhow!("token decode failed: {error}"))?
        {
            decoded.push_str(&text);
            for fragment in splitter.feed_token(&text, tokenizer.control_marker(next)) {
                if fragment.channel == Channel::Answer {
                    tool_text.push_str(&fragment.text);
                }
                let visible = if fragment.channel == Channel::Thinking {
                    vec![fragment.text]
                } else if let Some(filter) = &mut filter {
                    filter.feed(&fragment.text)
                } else {
                    vec![fragment.text]
                };
                for text in visible {
                    if !text.is_empty() {
                        first_visible_text.get_or_insert_with(|| started.elapsed().as_secs_f64());
                    }
                    bridge_emit_delta(
                        request_id,
                        if fragment.channel == Channel::Thinking {
                            "thinking"
                        } else {
                            "answer"
                        },
                        &text,
                        buffer_output,
                        &mut answer,
                    )?;
                }
            }
        }
        if let (Some(limit), Some(marker)) = (reasoning_limit, splitter.thinking_close()) {
            thinking_tokens += 1;
            let context_left = available - generated.len();
            if closing_plan.as_ref().is_none_or(|plan| plan.0 != marker) {
                let continuation = |compact| {
                    mlxl3_native::agent_loop::reasoning_continuation(
                        marker,
                        summarizing,
                        !tools.is_empty(),
                        compact,
                    )
                };
                let mut text = continuation(false);
                let mut size = tokenizer.encode(&text)?.len();
                if context_left.saturating_sub(limit.saturating_sub(thinking_tokens)) <= size {
                    text = continuation(true);
                    size = tokenizer.encode(&text)?.len();
                }
                closing_plan = Some((marker, text, size));
            }
            if let Some((_, text, size)) = &closing_plan
                && mlxl3_native::agent_loop::cut_thinking_now(
                    thinking_tokens,
                    limit,
                    context_left,
                    *size,
                    budget - generated.len(),
                )
            {
                reasoning_cut = Some(text.clone());
                break;
            }
        }
        if filter.as_ref().is_some_and(StreamFilter::is_preparing_tool)
            && last_tool_progress
                .is_none_or(|last| last.elapsed() >= std::time::Duration::from_millis(500))
        {
            emit_event(json!({
                "type":"generation_status", "request_id":request_id, "phase":"tool_preparation",
                "text":format!("Preparing tool call · {} tokens", generated.len())
            }))?;
            last_tool_progress = Some(Instant::now());
        }
        if generated.len() < budget {
            #[cfg(feature = "mlx")]
            {
                if let (Some(session), NativeChatModel::Inkling(target, Some(drafter), _)) =
                    (dspark.as_mut(), &mut *model)
                {
                    let mut choose = |proposals: &[u32],
                                      drafts: Option<&[Vec<f32>]>,
                                      verified: &mlxl3_native::array::Array|
                     -> Result<(usize, u32)> {
                        let acceptance = if sampler.is_greedy() {
                            mlxl3_native::speculative::greedy_accept(
                                proposals,
                                &verified.chat_greedy_ids()?,
                            )
                            .context("invalid DSpark verification")?
                        } else {
                            // DSPARK-INK2M-P3: proposals drawn from the draft law q are verified with
                            // min(1, p/q) and the residual max(0, p - q); deterministic proposals (or a
                            // draw the host law gives zero mass) keep the point-mass rule, exact too
                            let laws = drafts
                                .map(|rows| {
                                    rows.iter()
                                        .enumerate()
                                        .map(|(i, row)| {
                                            sampler.distribution(
                                                row.clone(),
                                                &history,
                                                &proposals[..i],
                                            )
                                        })
                                        .collect::<Result<Vec<_>>>()
                                })
                                .transpose()?
                                .filter(|laws| {
                                    laws.len() == proposals.len()
                                        && laws
                                            .iter()
                                            .zip(proposals)
                                            .all(|(law, &token)| law.probability(token) > 0.)
                                });
                            let mut draw = mlxl3_native::sampling::Draw {
                                sampler,
                                history: &history,
                                random: &mut *random,
                            };
                            match laws {
                                Some(laws) => draw.accept_drafts_logits(proposals, &laws, verified),
                                None => draw.speculative_accept_logits(proposals, verified),
                            }
                            .context("invalid DSpark sampled verification")?
                        };
                        Ok((acceptance.accepted_draft_tokens, acceptance.target_token))
                    };
                    dflash_next = Some(session.advance(
                        target,
                        drafter.as_mut(),
                        next,
                        context_limit,
                        budget - generated.len(),
                        &mut choose,
                    )?);
                } else if let (Some(session), NativeChatModel::Inkling(target, ..)) =
                    (inkling_lookup.as_mut(), &mut *model)
                {
                    // the lookup proposals are deterministic: point-mass acceptance keeps every law
                    let mut choose = |proposals: &[u32],
                                      verified: &mlxl3_native::array::Array|
                     -> Result<(usize, u32)> {
                        let acceptance = if sampler.is_greedy() {
                            mlxl3_native::speculative::greedy_accept(
                                proposals,
                                &verified.chat_greedy_ids()?,
                            )
                            .context("invalid lookup verification")?
                        } else {
                            mlxl3_native::sampling::Draw {
                                sampler,
                                history: &history,
                                random: &mut *random,
                            }
                            .speculative_accept_logits(proposals, verified)
                            .context("invalid lookup sampled verification")?
                        };
                        Ok((acceptance.accepted_draft_tokens, acceptance.target_token))
                    };
                    dflash_next = Some(session.advance(
                        target,
                        next,
                        context_limit,
                        budget - generated.len(),
                        &mut choose,
                    )?);
                } else if let Some((session, head)) = mtp_session.as_mut().zip(mtp.as_deref_mut()) {
                    dflash_next = Some(session.advance(
                        model,
                        head,
                        next,
                        context_limit,
                        budget - generated.len(),
                        |ids| {
                            tokenizer
                                .tokenizer()
                                .decode(ids, false)
                                .is_ok_and(|s| s.contains('\n'))
                        },
                        mlxl3_native::sampling::Draw {
                            sampler,
                            history: &history,
                            random: &mut *random,
                        },
                    )?);
                } else if let Some((session, weights)) = dflash.as_mut().zip(draft) {
                    let NativeChatModel::Qwen(target) = &mut *model else {
                        bail!("DFlash2 requires a supported Qwen target")
                    };
                    dflash_next = Some(session.advance_sampled(
                        target,
                        weights,
                        next,
                        context_limit,
                        budget - generated.len(),
                        Some(mlxl3_native::sampling::Draw {
                            sampler,
                            history: &history,
                            random: &mut *random,
                        }),
                    )?);
                } else if let NativeChatModel::Flash(target) = &mut *model
                    && target.dflash2_active()
                {
                    dflash_next = Some(target.advance_dflash2(next, budget - generated.len())?);
                } else if let (Some((session, lines)), NativeChatModel::Lfm3(target)) =
                    (lfm3_lookup.as_mut(), &mut *model)
                {
                    let lines = *lines;
                    dflash_next = Some(session.advance(
                        target,
                        next,
                        budget - generated.len(),
                        tokenizer.eos_ids(),
                        |ids| {
                            !lines
                                || tokenizer
                                    .tokenizer()
                                    .decode(ids, false)
                                    .is_ok_and(|s| s.contains('\n'))
                        },
                    )?);
                } else if let (Some(counts), NativeChatModel::Lfm3(target), true) =
                    (lfm3_cert.as_mut(), &mut *model, lfm3_sample)
                {
                    let step =
                        target.forward_sampled_certified(next, &sampler, &history, lfm3_check)?;
                    counts[usize::from(!step.certified)] += 1;
                    let mut distribution = sampler
                        .distribution(step.values, &history, &[])?
                        .canonical();
                    if let Some(reference) = step.reference {
                        let expected = sampler.distribution(reference, &history, &[])?.canonical();
                        if expected != distribution {
                            counts[2] += 1;
                            eprintln!(
                                "MLXL3_LFM3_CERT_CHECK: sampled distribution mismatch at token {}",
                                generated.len()
                            );
                            distribution = expected;
                        }
                    }
                    dflash_next = Some(distribution.sample(random));
                } else if let (Some(counts), NativeChatModel::Lfm3(target)) =
                    (lfm3_cert.as_mut(), &mut *model)
                {
                    let step = target.forward_greedy_checked(next, lfm3_check)?;
                    counts[usize::from(!step.certified)] += 1;
                    if step.mismatch {
                        counts[2] += 1;
                        eprintln!(
                            "MLXL3_LFM3_CERT_CHECK: greedy ID mismatch at token {}",
                            generated.len()
                        );
                    }
                    dflash_next = Some(step.id);
                } else if inkling_decode.is_none() {
                    logits = model.forward(next)?;
                }
            }
            #[cfg(not(feature = "mlx"))]
            if !mtp_pending.is_empty() {
                continue;
            } else if let Some((session, head)) = mtp_session.as_mut().zip(mtp.as_deref_mut()) {
                mtp_pending.extend(session.advance(
                    model,
                    head,
                    next,
                    u32::try_from(context_limit)?,
                    budget - generated.len(),
                    tokenizer.eos_ids(),
                    cancelled,
                )?);
            } else if let Some(draft) = draft {
                dflash_next = Some(draft.advance(
                    model.flash_mut()?,
                    next,
                    context_limit,
                    budget - generated.len(),
                    tokenizer.eos_ids(),
                    cancelled,
                )?);
            } else if device_greedy {
                device_greedy_next = Some(model.forward_greedy(next)?);
            } else {
                logits = model.forward(next)?;
            }
        }
    }
    let complete = tokenizer.decode(&generated)?;
    let tail = complete
        .strip_prefix(&decoded)
        .context("incremental tokenizer output differs from full decoding")?;
    for fragment in splitter
        .feed_literal(tail)
        .into_iter()
        .chain(splitter.finish())
    {
        if fragment.channel == Channel::Answer {
            tool_text.push_str(&fragment.text);
        }
        let visible = if fragment.channel == Channel::Thinking {
            vec![fragment.text]
        } else if let Some(filter) = &mut filter {
            filter.feed(&fragment.text)
        } else {
            vec![fragment.text]
        };
        for text in visible {
            if !text.is_empty() {
                first_visible_text.get_or_insert_with(|| started.elapsed().as_secs_f64());
            }
            bridge_emit_delta(
                request_id,
                if fragment.channel == Channel::Thinking {
                    "thinking"
                } else {
                    "answer"
                },
                &text,
                buffer_output,
                &mut answer,
            )?;
        }
    }
    if let Some(filter) = &mut filter {
        for text in filter.finish() {
            if !text.is_empty() {
                first_visible_text.get_or_insert_with(|| started.elapsed().as_secs_f64());
            }
            bridge_emit_delta(request_id, "answer", &text, buffer_output, &mut answer)?;
        }
    }
    #[cfg(feature = "mlx")]
    if let Some(capture) = gpu_capture {
        capture.finish()?;
    }
    let elapsed = started.elapsed().as_secs_f64();
    #[cfg(feature = "mlx")]
    if let NativeChatModel::Flash(target) = &*model
        && target.request_memory_limit().is_some()
    {
        emit_event(json!({"type":"flash_memory_end","request_id":request_id,
            "positions":target.offset(),"ledger":target.memory_ledger(None)?}))?;
    }
    let ttft = first_token.unwrap_or(elapsed);
    let stats = NativeStats {
        ttft_seconds: ttft,
        #[cfg(not(feature = "mlx"))]
        native_gpu_greedy_selections: device_greedy
            .then(|| model.device_greedy_counts()[0] - device_greedy_before[0]),
        #[cfg(not(feature = "mlx"))]
        native_gpu_greedy_heads: device_greedy
            .then(|| model.device_greedy_counts()[1] - device_greedy_before[1]),
        prefill_tps: (tokens.len() - cached) as f64 / prefill_seconds.max(1e-9),
        decode_tps: generated.len().saturating_sub(1) as f64 / (elapsed - ttft).max(1e-9),
        prompt_tokens: tokens.len(),
        generated_tokens: generated.len(),
        model_size_gb: resident_gb,
        cached_prompt_tokens: cached,
        evaluated_prompt_tokens: tokens.len() - cached,
        context_used: tokens.len() + generated.len(),
        context_limit,
        elapsed_seconds: elapsed,
        prepare_seconds,
        prefill_seconds,
        decode_seconds: elapsed - ttft,
        decode_tokens: generated.len().saturating_sub(1),
        round_count: 1,
        #[cfg(feature = "mlx")]
        memory: mlxl3_native::array::memory_stats(false).ok(),
        #[cfg(feature = "mlx")]
        dflash_proposed_tokens: dflash.as_ref().map(|session| session.proposed),
        #[cfg(feature = "mlx")]
        dflash_accepted_tokens: dflash.as_ref().map(|session| session.accepted),
        #[cfg(feature = "mlx")]
        dflash_blocks: dflash.as_ref().map(|session| session.blocks),
        #[cfg(feature = "mlx")]
        dflash_lookup_blocks: dflash.as_ref().map(|session| session.lookup_blocks),
        #[cfg(feature = "mlx")]
        dflash_block_seconds: dflash.as_ref().map_or(0., |session| session.block_seconds),
        #[cfg(feature = "mlx")]
        dspark_proposed_tokens: dspark.as_ref().map(|s| s.stats.proposed),
        #[cfg(feature = "mlx")]
        dspark_accepted_tokens: dspark.as_ref().map(|s| s.stats.accepted),
        #[cfg(feature = "mlx")]
        dspark_blocks: dspark.as_ref().map(|s| s.stats.blocks),
        #[cfg(feature = "mlx")]
        dspark_profile: dspark.as_ref().map(|s| s.stats.clone()),
        #[cfg(feature = "mlx")]
        mtp_proposed_tokens: mtp_session.as_ref().map(|s| s.stats().0),
        #[cfg(feature = "mlx")]
        mtp_accepted_tokens: mtp_session.as_ref().map(|s| s.stats().1),
        #[cfg(feature = "mlx")]
        mtp_blocks: mtp_session.as_ref().map(|s| s.stats().2),
        #[cfg(feature = "mlx")]
        mtp_lookup_blocks: mtp_session.as_ref().map(|s| s.stats().3),
        #[cfg(feature = "mlx")]
        mtp_depth_blocks: mtp_session.as_ref().map(|s| s.stats().4),
        #[cfg(feature = "mlx")]
        lfm3_lookup_blocks: lfm3_lookup.as_ref().map(|s| s.0.blocks),
        #[cfg(feature = "mlx")]
        lfm3_lookup_proposed_tokens: lfm3_lookup.as_ref().map(|s| s.0.proposed),
        #[cfg(feature = "mlx")]
        lfm3_lookup_accepted_tokens: lfm3_lookup.as_ref().map(|s| s.0.accepted),
        #[cfg(feature = "mlx")]
        inkling_lookup_blocks: inkling_lookup.as_ref().map(|s| s.stats.blocks),
        #[cfg(feature = "mlx")]
        inkling_lookup_proposed_tokens: inkling_lookup.as_ref().map(|s| s.stats.proposed),
        #[cfg(feature = "mlx")]
        inkling_lookup_accepted_tokens: inkling_lookup.as_ref().map(|s| s.stats.accepted),
        #[cfg(feature = "mlx")]
        inkling_lookup_profile: inkling_lookup.as_ref().map(|s| s.stats.clone()),
        #[cfg(feature = "mlx")]
        lfm3_cert_tokens: lfm3_cert.map(|counts| counts[0]).or_else(|| {
            lfm3_lookup
                .as_ref()
                .filter(|_| lfm3_lookup_certified)
                .map(|s| s.0.certified[0])
        }),
        #[cfg(feature = "mlx")]
        lfm3_cert_fallbacks: lfm3_cert.map(|counts| counts[1]).or_else(|| {
            lfm3_lookup
                .as_ref()
                .filter(|_| lfm3_lookup_certified)
                .map(|s| s.0.certified[1])
        }),
        #[cfg(feature = "mlx")]
        lfm3_cert_check_mismatches: lfm3_cert.filter(|_| lfm3_check).map(|counts| counts[2]),
        #[cfg(feature = "mlx")]
        topk_readback_rows: readback_start.map(|[certified, fallbacks]| {
            let [now_certified, now_fallbacks] = mlxl3_native::topk_readback::counters();
            [
                now_certified.saturating_sub(certified),
                now_fallbacks.saturating_sub(fallbacks),
            ]
        }),
        #[cfg(feature = "mlx")]
        mtp_sampled_draft_levels: sampled_draft_start
            .map(|start| mlxl3_native::sampling::sampled_draft_levels().saturating_sub(start)),
        #[cfg(feature = "mlx")]
        mtp_accept_histogram: mtp_session
            .as_ref()
            .and_then(|s| s.accept_histograms())
            .map(|h| h[0]),
        #[cfg(feature = "mlx")]
        mtp_lookup_accept_histogram: mtp_session
            .as_ref()
            .and_then(|s| s.accept_histograms())
            .map(|h| h[1]),
        #[cfg(not(feature = "mlx"))]
        memory: model.memory_stats().ok(),
        #[cfg(not(feature = "mlx"))]
        mtp_proposed_tokens: mtp_session.as_ref().map(|s| s.stats().0),
        #[cfg(not(feature = "mlx"))]
        mtp_accepted_tokens: mtp_session.as_ref().map(|s| s.stats().1),
        #[cfg(not(feature = "mlx"))]
        mtp_blocks: mtp_session.as_ref().map(|s| s.stats().2),
        #[cfg(not(feature = "mlx"))]
        mtp_depth_blocks: mtp_session.as_ref().and_then(|s| s.stats().3),
        #[cfg(not(feature = "mlx"))]
        dflash_proposed_tokens: draft.map(|s| s.stats().0),
        #[cfg(not(feature = "mlx"))]
        dflash_accepted_tokens: draft.map(|s| s.stats().1),
        #[cfg(not(feature = "mlx"))]
        dflash_blocks: draft.map(|s| s.stats().2),
        #[cfg(not(feature = "mlx"))]
        dflash_profile: draft.map(|s| s.profile()),
        #[cfg(not(feature = "mlx"))]
        dflash_lookup_blocks: draft.map(|_| 0),
        #[cfg(not(feature = "mlx"))]
        dflash_block_seconds: draft.map_or(0., |d| {
            let p = d.profile();
            p.draft_seconds + p.verification_seconds + p.commit_cache_seconds
        }),

        ..Default::default()
    };
    #[cfg(feature = "mlx")]
    let mut stats = stats;
    #[cfg(feature = "mlx")]
    if let NativeChatModel::Flash(target) = &*model
        && let Some((proposed, accepted, blocks, seconds)) = target.dflash2_stats()
    {
        stats.dflash_proposed_tokens = Some(proposed);
        stats.dflash_accepted_tokens = Some(accepted);
        stats.dflash_blocks = Some(blocks);
        stats.dflash_lookup_blocks = Some(0);
        stats.dflash_block_seconds = seconds;
        stats.dflash_profile = target.dflash2_profile();
    }
    Ok(RoundOutput {
        tool_text,
        stop_reason: mlxl3_native::GenerationStopReason::from_round(
            eos_observed,
            tokens.len() + generated.len() >= context_limit,
        ),
        raw: complete,
        answer,
        stats,
        first_text_seconds: first_visible_text,
        token_hash: token_hash(&generated),
        generated,
        reasoning_cut,
    })
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn validated_bridge_messages(messages: Vec<Value>) -> Result<Vec<Value>> {
    let mut output = Vec::new();
    for mut message in messages {
        let object = message
            .as_object_mut()
            .context("each message must be an object")?;
        let role = object
            .get("role")
            .and_then(Value::as_str)
            .context("message requires a role")?
            .to_owned();
        anyhow::ensure!(
            matches!(role.as_str(), "system" | "user" | "assistant" | "tool"),
            "invalid message role"
        );
        let turn = object.remove("turn_context");
        mlxl3_native::vision_input::validate_content(&message)?;
        if let Some(turn) = turn {
            anyhow::ensure!(
                role == "assistant",
                "only assistant messages can contain tool history"
            );
            let encoded = turn.as_str().context("turn_context must be a string")?;
            let expanded: Vec<Value> =
                serde_json::from_str(encoded).context("invalid tool history")?;
            let expanded = validated_bridge_messages(expanded)?;
            anyhow::ensure!(
                expanded
                    .iter()
                    .all(|value| matches!(value["role"].as_str(), Some("assistant" | "tool"))),
                "tool history contains an invalid role"
            );
            output.extend(expanded);
        } else {
            output.push(message);
        }
    }
    anyhow::ensure!(!output.is_empty(), "messages cannot be empty");
    Ok(output)
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
#[allow(clippy::too_many_arguments)]
fn bridge_generate(
    model: &mut NativeChatModel,
    tokenizer: &mlxl3_native::tokenizer::ChatTokenizer,
    request_id: &str,
    messages: Vec<Value>,
    max_tokens: i64,
    sampler: mlxl3_native::sampling::Sampler,
    context_limit: usize,
    resident_gb: f64,
    cancelled: &AtomicBool,
    agent_yield: Option<&AtomicBool>,
    random: &mut u64,
    mcp: &mut mlxl3_native::mcp::Manager,
    draft: Option<&RoundDraft>,
    mtp: Option<&mut ChatMtpHead>,
    mtp_depth: usize,
    dflash_mode: usize,
    started: Instant,
    draft_load_seconds: f64,
    cache: &mut Option<PromptCache>,
    conversation: &str,
    reuse: bool,
    allowed_tools: Option<&[String]>,
    response_schema: Option<&str>,
    tool_rounds: usize,
    reasoning_budget: Option<usize>,
) -> Result<()> {
    let schema = response_schema
        .map(mlxl3_native::structured::Schema::parse)
        .transpose()?;
    let mut dialogue = validated_bridge_messages(messages)?;
    if let Some(schema) = &schema {
        anyhow::ensure!(
            mcp.chat_tools().is_empty(),
            "JSON output and MCP must be used separately"
        );
        let instruction = schema.instruction();
        if dialogue
            .first()
            .is_some_and(|message| message["role"] == "system")
        {
            let existing = dialogue[0]["content"].as_str().unwrap_or("");
            dialogue[0]["content"] = json!(format!("{existing}\n\n{instruction}"));
        } else {
            dialogue.insert(0, json!({"role":"system", "content":instruction}));
        }
    }
    let original_length = dialogue.len();
    // Every model is offered its tools; Inkling 2's `<|content_invoke_tool_json|>` calls are
    // parsed by `tool_call` like the other formats.
    let tools: Vec<_> = mcp
        .chat_tools()
        .into_iter()
        .filter(|tool| {
            tool["function"]["name"]
                .as_str()
                .is_some_and(|name| mlxl3_native::structured::tool_allowed(name, allowed_tools))
        })
        .collect();
    let image_format = chat_image_format(model);
    let mut host = BridgeHost {
        model,
        tokenizer,
        request_id,
        sampler,
        context_limit,
        resident_gb,
        cancelled,
        agent_yield,
        random,
        mcp,
        draft,
        mtp,
        mtp_depth,
        dflash_mode,
        cache,
        conversation,
        reuse,
        image_format,
        started,
        first_text: None,
        summary: None,
        round_stats: Vec::new(),
        tool_seconds: 0.,
    };
    let end = mlxl3_native::agent_turn::run_turn(
        &mut host,
        dialogue,
        &tools,
        &mlxl3_native::agent_turn::Turn {
            request_id,
            context_limit,
            max_tokens,
            tool_rounds,
            reasoning_budget,
            schema: schema.as_ref(),
            allowed_tools,
            original_length,
        },
    )?;
    let mut stats = host.summary.context("missing generation statistics")?;
    stats.elapsed_seconds = started.elapsed().as_secs_f64();
    stats.end_to_end_ttft_seconds = host.first_text;
    stats.tool_seconds = host.tool_seconds;
    stats.draft_load_seconds = draft_load_seconds;
    stats.tool_rounds = if schema.is_some() { 0 } else { end.rounds };
    emit_event(json!({
        "type":"complete", "request_id":request_id,
        "assistant_context":end.assistant, "cache_context":end.raw,
        "token_hash":end.token_hash,
        "stats":stats, "round_stats":host.round_stats, "context_full":end.context_full,
        "stop_reason":end.stop_reason,
        "turn_context":end.turn_context,
        "elapsed_seconds":started.elapsed().as_secs_f64(),
        "first_text_seconds":host.first_text,
        "tool_rounds":end.rounds
    }))
}

/// The engine side of `agent_turn::run_turn`: the loaded model, its tokenizer, the MCP
/// servers and the bridge's event stream, plus the turn's statistics.
/// The backend's model and MTP head; the direct-metal ones borrow the loaded checkpoint.
#[cfg(all(feature = "mlx", feature = "chat"))]
type HostModel<'m> = NativeChatModel;
#[cfg(all(feature = "mlx", feature = "chat"))]
type HostMtp<'m> = ChatMtpHead;
#[cfg(all(
    not(feature = "mlx"),
    feature = "chat",
    target_os = "macos",
    feature = "direct-metal"
))]
type HostModel<'m> = NativeChatModel<'m>;
#[cfg(all(
    not(feature = "mlx"),
    feature = "chat",
    target_os = "macos",
    feature = "direct-metal"
))]
type HostMtp<'m> = ChatMtpHead<'m>;

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
struct BridgeHost<'a, M, H> {
    model: &'a mut M,
    tokenizer: &'a mlxl3_native::tokenizer::ChatTokenizer,
    request_id: &'a str,
    sampler: mlxl3_native::sampling::Sampler,
    context_limit: usize,
    resident_gb: f64,
    cancelled: &'a AtomicBool,
    agent_yield: Option<&'a AtomicBool>,
    random: &'a mut u64,
    mcp: &'a mut mlxl3_native::mcp::Manager,
    draft: Option<&'a RoundDraft>,
    mtp: Option<&'a mut H>,
    mtp_depth: usize,
    dflash_mode: usize,
    cache: &'a mut Option<PromptCache>,
    conversation: &'a str,
    reuse: bool,
    image_format: Option<mlxl3_native::vision_input::ImageFormat>,
    started: Instant,
    first_text: Option<f64>,
    summary: Option<NativeStats>,
    round_stats: Vec<NativeStats>,
    tool_seconds: f64,
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
impl mlxl3_native::agent_turn::Host for BridgeHost<'_, HostModel<'_>, HostMtp<'_>> {
    fn generate(
        &mut self,
        request: mlxl3_native::agent_turn::StepRequest<'_>,
    ) -> Result<mlxl3_native::agent_turn::Step> {
        let round_start = self.started.elapsed().as_secs_f64();
        let output = bridge_generate_round(
            self.model,
            self.tokenizer,
            self.request_id,
            request.dialogue,
            request.tools,
            request.max_tokens,
            self.sampler,
            self.context_limit,
            self.resident_gb,
            self.cancelled,
            self.random,
            self.draft,
            self.mtp.as_deref_mut(),
            self.mtp_depth,
            self.dflash_mode,
            self.cache,
            self.conversation,
            self.reuse,
            request.buffered,
            request.continuation,
            request.reasoning_budget,
        )?;
        if self.first_text.is_none() {
            self.first_text = output
                .first_text_seconds
                .map(|seconds| round_start + seconds);
        }
        if let Some(summary) = &mut self.summary {
            summary.accumulate(&output.stats);
        } else {
            let mut first = output.stats.clone();
            first.ttft_seconds += round_start;
            self.summary = Some(first);
        }
        self.round_stats.push(output.stats.clone());
        Ok(mlxl3_native::agent_turn::Step {
            stop_reason: output.stop_reason,
            raw: output.raw,
            answer: output.answer,
            tool_text: output.tool_text,
            generated: output.generated,
            reasoning_cut: output.reasoning_cut,
            context_used: output.stats.context_used,
            generated_tokens: output.stats.generated_tokens,
            token_hash: output.token_hash,
        })
    }

    fn prompt_tokens(&mut self, dialogue: &[Value], tools: &[Value]) -> Result<usize> {
        bridge_prompt_tokens(self.tokenizer, dialogue, tools, self.image_format)
    }

    fn encode(&mut self, text: &str) -> Result<Vec<u32>> {
        self.tokenizer.encode(text)
    }

    fn decode(&mut self, tokens: &[u32]) -> Result<String> {
        self.tokenizer.decode(tokens)
    }

    fn server(&self, tool: &str) -> Option<String> {
        self.mcp.tools.get(tool).map(|tool| tool.server.clone())
    }

    fn call_tool(&mut self, name: &str, arguments: Value) -> mlxl3_native::mcp::ToolResult {
        let started = Instant::now();
        let result = self.mcp.call_cancellable(name, arguments, self.cancelled);
        self.tool_seconds += started.elapsed().as_secs_f64();
        result
    }

    fn emit(&mut self, event: Value) -> Result<()> {
        emit_event(event)
    }

    fn yield_requested(&self) -> bool {
        self.agent_yield
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    }

    fn cancellation(&self) -> &AtomicBool {
        self.cancelled
    }

    fn forget_prompt_cache(&mut self) {
        *self.cache = None;
    }

    fn reset_model(&mut self) {
        self.model.reset();
    }

    fn answer_shown(&mut self) {
        self.first_text = Some(self.started.elapsed().as_secs_f64());
    }
}

/// Tokens of the prompt the next round would render (same template, tools and reasoning level).
#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn bridge_prompt_tokens(
    tokenizer: &mlxl3_native::tokenizer::ChatTokenizer,
    dialogue: &[Value],
    tools: &[Value],
    image_format: Option<mlxl3_native::vision_input::ImageFormat>,
) -> Result<usize> {
    let (prepared, images) =
        mlxl3_native::vision_input::prepare_messages_as(dialogue, image_format)?;
    let rendered = REASONING_REQUESTED.with(|effort| {
        tokenizer.render_values_with_effort(
            &Value::Array(prepared),
            Some(tools),
            effort.borrow().as_deref(),
        )
    })?;
    let rendered =
        mlxl3_native::vision_input::expand_image_placeholders(image_format, rendered, &images)?;
    Ok(tokenizer.encode(&rendered)?.len())
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn load_dflash_weights(
    slot: &mut Option<(PathBuf, String, mlxl3_native::dflash::DFlashWeights)>,
    cache: &mut Option<PromptCache>,
    path: &str,
    family: mlxl3_native::dflash::Family,
) -> Result<bool> {
    anyhow::ensure!(
        !path.trim().is_empty(),
        "DFlash2 draft folder is not configured"
    );
    let path = registry::expand_home(&PathBuf::from(path))?.canonicalize()?;
    let revision = mtp_tune::artifact_key(&path.join("draft"))?;
    if slot
        .as_ref()
        .is_some_and(|(previous, fingerprint, _)| previous == &path && fingerprint == &revision)
    {
        return Ok(false);
    }
    *slot = None;
    *cache = None;
    mlxl3_native::array::synchronize()?;
    mlxl3_native::array::clear_cache()?;
    let package = mlxl3_native::dflash::inspect(&path)?;
    anyhow::ensure!(
        package.family == family,
        "DFlash draft does not match this target"
    );
    *slot = Some((
        path,
        revision,
        mlxl3_native::dflash::DFlashWeights::load(&package)?,
    ));
    Ok(true)
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn load_mtp_head(
    head: &mut Option<(PathBuf, ChatMtpHead)>,
    revision: &mut Option<String>,
    prompt_cache: &mut Option<PromptCache>,
    requested: &str,
    model: &mut NativeChatModel,
    dflash_weights: &mut Option<(PathBuf, String, mlxl3_native::dflash::DFlashWeights)>,
    cancelled: &AtomicBool,
) -> Result<bool> {
    // A direct generate request can switch modes without a preceding set_mtp.
    // Release both possible DFlash owners before validating or allocating MTP,
    // including the failed-load and already-loaded-head paths.
    let flash_dflash = matches!(&*model, NativeChatModel::Flash(target) if target.dflash2_active());
    if flash_dflash || dflash_weights.is_some() {
        *prompt_cache = None;
    }
    *dflash_weights = None;
    if let NativeChatModel::Flash(target) = &mut *model {
        target.configure_dflash2(None, 1)?;
    }
    anyhow::ensure!(
        !requested.trim().is_empty(),
        "MTP head folder is not configured"
    );
    let path = registry::expand_home(&PathBuf::from(requested))?.canonicalize()?;
    let next_revision = mtp_tune::artifact_key(&path)?;
    if head.as_ref().map(|(p, _)| p) == Some(&path) && revision.as_ref() == Some(&next_revision) {
        return Ok(false);
    }
    // Drop the previous head and its cached state before allocating a replacement.
    *prompt_cache = None;
    *head = None;
    *revision = None;
    mlxl3_native::array::synchronize()?;
    mlxl3_native::array::clear_cache()?;
    *head = Some((path.clone(), ChatMtpHead::load(&path, model, cancelled)?));
    *revision = Some(next_revision);
    Ok(true)
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn release_bridge_idle_state(
    model: &mut NativeChatModel,
    head: Option<&mut ChatMtpHead>,
    memory_saver: bool,
    keep_inkling: bool,
) -> Result<()> {
    // Both prefill paths reset or restore their separate PromptCache on the next
    // request. The final decode state is not a continuation checkpoint, except for
    // Inkling 2 with a saved prefix (SPEED-2026-11-23-INK2M-MAYA PC): its full visits
    // resume from the rows the live state still holds.
    if !(keep_inkling && !memory_saver && matches!(model, NativeChatModel::Inkling(..))) {
        model.reset();
    }
    let reset = head.map_or(Ok(()), ChatMtpHead::reset);
    let released = mlxl3_native::array::release_idle_cache_with_policy(memory_saver);
    reset?;
    released?;
    if memory_saver && let NativeChatModel::Flash(target) = model {
        target.clear_idle_program_cache()?;
    }
    Ok(())
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn native_bridge(
    registry_path: Option<&std::path::Path>,
    name: &str,
    requested: i32,
) -> Result<()> {
    use mlxl3_native::tokenizer::ChatTokenizer;
    let registry_path = registry_path
        .map(PathBuf::from)
        .map(Ok)
        .unwrap_or_else(registry::default_path)?;
    let path = resolve_model_path(Some(&registry_path), name)?;
    let checkpoint = checkpoint::inspect(&path)?;
    emit_event(json!({"type":"loading", "model":name}))?;
    let started = Instant::now();
    let tokenizer = ChatTokenizer::load(&path)?;
    let mut model = NativeChatModel::from_checkpoint(&checkpoint)?;
    let converted_embedding = checkpoint.tensors.iter().any(|(name, tensor)| {
        name.ends_with("embed_tokens.weight")
            && tensor.dtype != "F16"
            && tensor.data_offsets[1] - tensor.data_offsets[0] >= 64 * 1024 * 1024
    });
    if converted_embedding && std::env::var("MLXL3_RETAIN_LOAD_CACHE").as_deref() != Ok("1") {
        mlxl3_native::array::clear_cache()?;
    }
    let model_limit = model.context_limit();
    anyhow::ensure!(
        requested >= 0 && requested <= model_limit,
        "context length must be between 1 and {model_limit}, or 0 for automatic"
    );
    let context_limit = if requested == 0 {
        model_limit
    } else {
        requested
    } as usize;
    let resident_gb = checkpoint.size_bytes as f64 / 1e9;
    let dflash_family = match &model {
        NativeChatModel::Qwen(target) => target.dflash_family(),
        _ => None,
    }
    .filter(|&family| {
        dflash_family_offered(family, std::env::var("MLXL3_DFLASH_35B").ok().as_deref())
    });
    let flash_speculative =
        matches!(&model, NativeChatModel::Flash(target) if target.request_memory_limit().is_none());
    let dflash_supported = (dflash_family.is_some()
        && mlxl3_native::array::supports_packed_dflash_q4()?)
        || flash_speculative;
    let mtp_auto_download_supported = matches!(&model, NativeChatModel::Qwen(_))
        && mlxl3_native::mtp::target_layout(&path)
            .is_ok_and(|(layout, layers)| layout.managed_head(layers).is_some());
    let executable = std::env::current_exe()?.display().to_string();
    let tuning_key = mtp_tune::runtime_key(&path, context_limit)?;
    let dflash_tuning_key = dflash_runtime_key(
        &tuning_key,
        dflash_context_copy_enabled(dflash_family),
        mlxl3_native::speculative::dflash_width_cap(
            dflash_family == Some(mlxl3_native::dflash::Family::Qwen38Dense),
            mlxl3_native::linear::dense_shared_enabled(),
            1,
        ),
    );
    let memory = mlxl3_native::array::memory_stats(false)?;
    let mut ready = json!({
        "flash":match &model { NativeChatModel::Flash(target) => Some(target.runtime_manifest()?), _ => None },
        "type":"ready", "model":name, "modules":checkpoint.modules.len(),
        "images_supported":model.supports_images(),
        "reasoning_effort_supported":true,
        "reasoning_effort_levels":mlxl3_native::tokenizer::REASONING_EFFORTS,
        // Agent mode: `tool_rounds` per request, and `callTimeoutSeconds` per MCP server;
        // AGENT-2026-11-23-BOUCLE: `reasoning_budget`, 512 rounds, pruning, repeated-call stop.
        "tool_rounds_supported":true, "mcp_call_timeout_supported":true,
        "agent_loop_supported":true,
        "load_seconds":started.elapsed().as_secs_f64(), "resident_gb":resident_gb,
        "memory":memory,
        "runtime_commit":env!("MLXL3_BUILD_REVISION"),
        "runtime_profile":env!("MLXL3_BUILD_PROFILE"),
        "mlx_version":env!("MLXL3_MLX_VERSION"),
        "runtime_executable":executable,
        "dflash_supported":dflash_supported, "dflash_family":dflash_family,
        "dflash_sampling_supported":dflash_supported && matches!(&model, NativeChatModel::Qwen(_)),
        "dflash_tune_supported":dflash_family.is_some() && dflash_supported, "dflash_configure_supported":true,
        "dflash_tuning_key":dflash_tuning_key,
        "mtp_supported":matches!(&model, NativeChatModel::Qwen(_)) || flash_speculative,
        "dspark_supported":inkling_has_drafter(&model),
        "mtp_sampling_supported":true,
        "mtp_auto_download_supported":mtp_auto_download_supported,
        "mtp_configure_supported":true,
        "mtp_max_depth":if matches!(&model, NativeChatModel::Flash(_)) { 1 } else { 3 }, "mtp_tune_supported":matches!(&model, NativeChatModel::Qwen(_)),
        "memory_saver_supported":true,
        "context_memory":match &model {
            NativeChatModel::Qwen(target) => Some(target.context_memory_profile()?),
            NativeChatModel::Lfm2(target) => Some(target.context_memory_profile()?),
            _ => None
        },
        "mtp_tuning_key":tuning_key,
        "runtime_version":env!("CARGO_PKG_VERSION"), "bridge_protocol":1,
        "mcp_servers":0, "mcp_tools":0, "mcp_errors":{}, "workspace_features":true,
        "context_limit":context_limit, "model_context_limit":model_limit
    });
    // DECISION-2026-11-25-INK2M-DFLASH2-ON (set apart: the macro above is at its recursion limit)
    ready["dspark_default"] = json!(inkling_speculation_requested(
        None,
        matches!(&model, NativeChatModel::Inkling(..)) && inkling_dflash2_source().is_some(),
        std::env::var("MLXL3_INKLING_DFLASH2_DEFAULT")
            .ok()
            .as_deref(),
    ));
    let agent_yield = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGUSR2, Arc::clone(&agent_yield))?;
    ready["agent_yield_supported"] = json!(true);
    emit_event(ready)?;
    let cancelled = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGUSR1, Arc::clone(&cancelled))?;
    let mut random = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
        | 1;
    let mut mcp = mlxl3_native::mcp::Manager::disabled();
    let mut dflash_weights: Option<(PathBuf, String, mlxl3_native::dflash::DFlashWeights)> = None;
    let mut mtp_head: Option<(PathBuf, ChatMtpHead)> = None;
    let mut mtp_head_revision = None;
    let mut prompt_cache: Option<PromptCache> = None;
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let parsed: Result<BridgeRequest> =
            serde_json::from_str(&line).context("invalid bridge request");
        let request = match parsed {
            Ok(request) => request,
            Err(error) => {
                emit_event(json!({"type":"error", "request_id":"", "message":error.to_string()}))?;
                continue;
            }
        };
        match request.kind.as_str() {
            "shutdown" => break,
            "flash_memory" => {
                let result = match &model {
                    NativeChatModel::Flash(target) => target.memory_ledger(match mtp_head.as_ref() {
                        Some((_, ChatMtpHead::Flash(head))) => Some(head),
                        _ => None,
                    }).map(|ledger|
                        json!({"type":"flash_memory","request_id":request.request_id,"ledger":ledger})),
                    _ => Err(anyhow::anyhow!("flash_memory requires Flash Next")),
                };
                emit_event(result.unwrap_or_else(|e| {
                    json!({"type":"error",
                    "request_id":request.request_id,"message":e.to_string()})
                }))?;
            }
            "ping" => emit_event(json!({"type":"pong", "request_id":request.request_id,
                "memory":mlxl3_native::array::memory_stats(false)?}))?,
            "set_mcp" => {
                mcp.set_enabled(&registry_path, request.enabled, true);
                emit_event(json!({
                    "type":"mcp_status", "request_id":request.request_id, "mcp_servers":mcp.server_count(),
                    "mcp_tools":mcp.tools.len(), "mcp_errors":mcp.errors, "mcp_tool_catalog":mcp.chat_tools()
                }))?;
            }
            "set_mtp" => {
                cancelled.store(false, Ordering::Relaxed);
                prompt_cache = None;
                model.reset();
                let result = (|| -> Result<()> {
                    if request.enabled {
                        anyhow::ensure!(
                            matches!(&model, NativeChatModel::Qwen(_) | NativeChatModel::Flash(_)),
                            "MTP requires a supported Qwen target"
                        );
                        load_mtp_head(
                            &mut mtp_head,
                            &mut mtp_head_revision,
                            &mut prompt_cache,
                            &request.mtp_head_path,
                            &mut model,
                            &mut dflash_weights,
                            &cancelled,
                        )?;
                    } else {
                        mtp_head = None;
                        mtp_head_revision = None;
                    }
                    mlxl3_native::array::synchronize()?;
                    mlxl3_native::array::clear_cache()?;
                    emit_event(json!({"type":"mtp_status", "request_id":request.request_id,
                        "mtp_active":mtp_head.is_some(),
                        "mtp_head_path":mtp_head.as_ref().map(|(path, _)| path),
                        "memory":mlxl3_native::array::memory_stats(false)?}))
                })();
                if let Err(error) = result {
                    mtp_head = None;
                    mtp_head_revision = None;
                    mlxl3_native::array::synchronize()?;
                    mlxl3_native::array::clear_cache()?;
                    emit_event(json!({"type":"error", "request_id":request.request_id,
                        "message":error.to_string(), "mtp_active":false,
                        "memory":mlxl3_native::array::memory_stats(false)?}))?;
                }
            }
            "set_dflash" | "tune_dflash" => {
                cancelled.store(false, Ordering::Relaxed);
                prompt_cache = None;
                model.reset();
                let result = (|| -> Result<()> {
                    if let NativeChatModel::Flash(target) = &mut model {
                        anyhow::ensure!(
                            request.kind != "tune_dflash",
                            "Flash DFlash2 learns its verify budget online; Tune is not supported"
                        );
                        if request.enabled {
                            mtp_head = None;
                            mtp_head_revision = None;
                        }
                        dflash_weights = None;
                        let path =
                            registry::expand_home(&PathBuf::from(&request.dflash_draft_path))?;
                        target.configure_dflash2(
                            request.enabled.then_some(path.as_path()),
                            request.dflash_mode,
                        )?;
                        return emit_event(
                            json!({"type":"dflash_status", "request_id":request.request_id,
                            "dflash_active":target.dflash2_active(), "memory":mlxl3_native::array::memory_stats(false)?}),
                        );
                    }
                    if request.enabled || request.kind == "tune_dflash" {
                        let family = dflash_family.context("Unsupported DFlash target")?;
                        mtp_head = None;
                        mtp_head_revision = None;
                        mlxl3_native::array::synchronize()?;
                        mlxl3_native::array::clear_cache()?;
                        load_dflash_weights(
                            &mut dflash_weights,
                            &mut prompt_cache,
                            &request.dflash_draft_path,
                            family,
                        )?;
                    } else {
                        dflash_weights = None;
                    }
                    if request.kind == "tune_dflash" {
                        mtp_tune::run_dflash(
                            &mut model,
                            &dflash_weights.as_ref().context("missing DFlash draft")?.2,
                            &tokenizer,
                            &request.request_id,
                            &dflash_tuning_key,
                            context_limit,
                            &cancelled,
                            request.sampler()?,
                        )
                    } else {
                        mlxl3_native::array::synchronize()?;
                        mlxl3_native::array::clear_cache()?;
                        emit_event(
                            json!({"type":"dflash_status", "request_id":request.request_id,
                            "dflash_active":dflash_weights.is_some(), "memory":mlxl3_native::array::memory_stats(false)?}),
                        )
                    }
                })();
                if let Err(error) = release_bridge_idle_state(
                    &mut model,
                    mtp_head.as_mut().map(|(_, head)| head),
                    request.memory_saver,
                    false,
                ) {
                    eprintln!("Idle memory cleanup failed: {error:#}");
                }
                if let Err(error) = result {
                    dflash_weights = None;
                    if let NativeChatModel::Flash(target) = &mut model {
                        target.configure_dflash2(None, 1)?;
                    }
                    mlxl3_native::array::synchronize()?;
                    mlxl3_native::array::clear_cache()?;
                    if cancelled.swap(false, Ordering::Relaxed) {
                        emit_event(json!({"type":"cancelled","request_id":request.request_id}))?;
                    } else {
                        emit_event(
                            json!({"type":"error","request_id":request.request_id,"message":error.to_string(),"dflash_active":false}),
                        )?;
                    }
                }
            }
            "tune_mtp" => {
                cancelled.store(false, Ordering::Relaxed);
                prompt_cache = None;
                dflash_weights = None;
                let result = (|| -> Result<()> {
                    let NativeChatModel::Qwen(_) = &model else {
                        bail!("Tune MTP supports Qwen3.5/3.6/3.8 targets");
                    };
                    emit_event(
                        json!({"type":"mtp_tune_progress","request_id":request.request_id,
                        "phase":"loading_head","completed":0,"total":12}),
                    )?;
                    load_mtp_head(
                        &mut mtp_head,
                        &mut mtp_head_revision,
                        &mut prompt_cache,
                        &request.mtp_head_path,
                        &mut model,
                        &mut dflash_weights,
                        &cancelled,
                    )?;
                    let ChatMtpHead::Qwen(head) =
                        &mut mtp_head.as_mut().context("missing MTP head")?.1
                    else {
                        bail!("Tune MTP requires a Qwen target");
                    };
                    mtp_tune::run(
                        &mut model,
                        head,
                        &tokenizer,
                        &request.request_id,
                        &tuning_key,
                        context_limit,
                        &cancelled,
                        request.sampler()?,
                    )
                })();
                if let Err(error) = release_bridge_idle_state(
                    &mut model,
                    mtp_head.as_mut().map(|(_, head)| head),
                    request.memory_saver,
                    false,
                ) {
                    eprintln!("Idle memory cleanup failed: {error:#}");
                }
                if let Err(error) = result {
                    if cancelled.swap(false, Ordering::Relaxed) {
                        emit_event(json!({"type":"cancelled","request_id":request.request_id}))?;
                    } else {
                        emit_event(
                            json!({"type":"error","request_id":request.request_id,"message":error.to_string()}),
                        )?;
                    }
                }
            }
            "generate" => {
                agent_yield.store(false, Ordering::Relaxed);
                if request.memory_saver {
                    prompt_cache = None;
                }
                let started = Instant::now();
                let _ = mlxl3_native::array::memory_stats(true);
                cancelled.store(false, Ordering::Relaxed);
                let schema_valid = request
                    .response_schema
                    .as_deref()
                    .map(mlxl3_native::structured::Schema::parse)
                    .transpose();
                if let Err(error) = schema_valid {
                    emit_event(
                        json!({"type":"error", "request_id":request.request_id, "message":error.to_string()}),
                    )?;
                    continue;
                }
                if request.response_schema.is_some() && request.mcp_enabled {
                    emit_event(
                        json!({"type":"error", "request_id":request.request_id, "message":"JSON output and MCP must be used separately"}),
                    )?;
                    continue;
                }
                mcp.set_enabled(&registry_path, request.mcp_enabled, false);
                let request_id = request.request_id.clone();
                let memory_saver = request.memory_saver;
                let result = (|| -> Result<()> {
                    let mut draft_load_seconds = 0.;
                    anyhow::ensure!(
                        !(request.mtp && request.dflash2),
                        "Choose either MTP or DFlash2"
                    );
                    let sampler = request.sampler()?;
                    // Non-greedy MTP verifies by exact speculative sampling.
                    let mtp_active = request.mtp;
                    if mtp_active {
                        anyhow::ensure!(
                            (1..=3).contains(&request.mtp_depth),
                            "MTP depth must be 1, 2 or 3"
                        );
                        anyhow::ensure!(
                            matches!(&model, NativeChatModel::Qwen(_) | NativeChatModel::Flash(_)),
                            "MTP requires a supported Qwen target"
                        );
                        anyhow::ensure!(
                            !matches!(&model, NativeChatModel::Flash(_)) || request.mtp_depth == 1,
                            "Flash MTP supports depth 1"
                        );
                        let started = Instant::now();
                        if load_mtp_head(
                            &mut mtp_head,
                            &mut mtp_head_revision,
                            &mut prompt_cache,
                            &request.mtp_head_path,
                            &mut model,
                            &mut dflash_weights,
                            &cancelled,
                        )? {
                            prompt_cache = None;
                            draft_load_seconds += started.elapsed().as_secs_f64();
                        }
                    } else {
                        mtp_head = None;
                        mtp_head_revision = None;
                    }
                    if request.dflash2 {
                        anyhow::ensure!(
                            (1..=3).contains(&request.dflash_mode),
                            "DFlash mode must be 1, 2 or 3"
                        );
                        // Q27D-09: packed Qwen drafts verify sampled policies exactly;
                        // the Flash Next DFlash2 path stays greedy-only.
                        anyhow::ensure!(
                            request.greedy_unpenalized()
                                || !matches!(&model, NativeChatModel::Flash(_)),
                            "Flash DFlash2 requires greedy sampling and repetition penalty 1.0"
                        );
                        let draft_started = Instant::now();
                        if let NativeChatModel::Flash(target) = &mut model {
                            dflash_weights = None;
                            prompt_cache = None;
                            let path =
                                registry::expand_home(&PathBuf::from(&request.dflash_draft_path))?;
                            target.configure_dflash2(Some(&path), request.dflash_mode)?;
                            draft_load_seconds += draft_started.elapsed().as_secs_f64();
                        } else if load_dflash_weights(
                            &mut dflash_weights,
                            &mut prompt_cache,
                            &request.dflash_draft_path,
                            dflash_family.context("Unsupported DFlash2 target")?,
                        )? {
                            draft_load_seconds += draft_started.elapsed().as_secs_f64();
                        }
                    } else {
                        if let NativeChatModel::Flash(target) = &mut model {
                            target.configure_dflash2(None, 1)?;
                        }
                        if dflash_weights.is_some() {
                            prompt_cache = None;
                        }
                        dflash_weights = None;
                    }
                    let flash_dflash =
                        matches!(&model, NativeChatModel::Flash(target) if target.dflash2_active());
                    let dflash_active = dflash_weights.is_some() || flash_dflash;
                    let proposals = if let NativeChatModel::Flash(target) = &model {
                        target.dflash2_proposals().unwrap_or(0)
                    } else if dflash_active {
                        match request.dflash_mode {
                            3 => 7,
                            2 => 2,
                            _ => 5,
                        }
                    } else {
                        0
                    };
                    let draft_path = if flash_dflash {
                        Some(request.dflash_draft_path.clone())
                    } else {
                        dflash_weights
                            .as_ref()
                            .map(|(path, _, _)| path.display().to_string())
                    };
                    emit_generation_mode(
                        &request,
                        mtp_active,
                        dflash_active,
                        proposals,
                        draft_path,
                    )?;
                    // DECISION-2026-11-25-INK2M-DFLASH2-ON: an omitted field follows the DFlash2 drafter
                    #[cfg(feature = "mlx")]
                    let dspark = inkling_speculation_requested(
                        request.dspark,
                        matches!(model, NativeChatModel::Inkling(..))
                            && inkling_dflash2_source().is_some(),
                        std::env::var("MLXL3_INKLING_DFLASH2_DEFAULT")
                            .ok()
                            .as_deref(),
                    );
                    #[cfg(feature = "mlx")]
                    if dspark
                        && inkling_has_drafter(&model)
                        && !matches!(model, NativeChatModel::Inkling(_, Some(_), _))
                    {
                        emit_event(json!({
                            "type":"generation_status", "request_id":request.request_id,
                            "phase":"dspark_loading", "text":"Loading DSpark…"
                        }))?;
                    }
                    #[cfg(feature = "mlx")]
                    if let Err(error) = model.prepare_inkling_drafter(dspark) {
                        // the round decodes plainly; the source stays known for a later request
                        eprintln!("DSpark drafter unavailable: {error:#}");
                    }
                    #[cfg(feature = "mlx")]
                    let _dspark = DsparkRequest::set(dspark);
                    let _lookup = LookupRequest::set(request.lookup);
                    let _reasoning = ReasoningRequest::set(request.reasoning_effort.clone())?;
                    bridge_generate(
                        &mut model,
                        &tokenizer,
                        &request.request_id,
                        request.messages,
                        request.max_tokens,
                        sampler,
                        context_limit,
                        resident_gb,
                        &cancelled,
                        request.tool_rounds.map(|_| agent_yield.as_ref()),
                        &mut random,
                        &mut mcp,
                        dflash_weights.as_ref().map(|(_, _, weights)| weights),
                        mtp_head.as_mut().map(|(_, head)| head),
                        request.mtp_depth,
                        request.dflash_mode,
                        started,
                        draft_load_seconds,
                        &mut prompt_cache,
                        if request.conversation_id.is_empty() {
                            &request.request_id
                        } else {
                            &request.conversation_id
                        },
                        request.reuse_prompt_cache && !request.memory_saver,
                        request.allowed_tools.as_deref(),
                        request.response_schema.as_deref(),
                        mlxl3_native::tool_call::tool_rounds(request.tool_rounds)?,
                        mlxl3_native::agent_loop::reasoning_budget(request.reasoning_budget)?,
                    )
                })();
                // Runs after the whole turn (including tools), not between decode
                // tokens. Also release temporary state after cancellation/error.
                let keep_inkling = matches!(
                    &prompt_cache,
                    Some(PromptCache {
                        state: CacheState::Inkling(_) | CacheState::InklingDspark(_),
                        ..
                    })
                );
                if let Err(error) = release_bridge_idle_state(
                    &mut model,
                    mtp_head.as_mut().map(|(_, head)| head),
                    memory_saver,
                    keep_inkling,
                ) {
                    eprintln!("Idle memory cleanup failed: {error:#}");
                }
                if let Err(error) = result {
                    model.reset();
                    prompt_cache = None;
                    mtp_head = None;
                    mtp_head_revision = None;
                    if cancelled.swap(false, Ordering::Relaxed) {
                        emit_event(json!({"type":"cancelled", "request_id":request_id}))?;
                    } else {
                        emit_event(
                            json!({"type":"error", "request_id":request_id, "message":error.to_string()}),
                        )?;
                    }
                }
            }
            other => emit_event(json!({
                "type":"error", "request_id":request.request_id,
                "message":format!("unsupported request type: {other}")
            }))?,
        }
    }
    Ok(())
}

#[cfg(all(
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
fn native_chat(
    path: &std::path::Path,
    prompt: Option<String>,
    max_tokens: usize,
    image_paths: Vec<PathBuf>,
) -> Result<()> {
    use mlxl3_native::{streaming::Channel, tokenizer::ChatTokenizer};
    let tokenizer = ChatTokenizer::load(path)?;
    #[cfg(feature = "mlx")]
    let mut model = NativeChatModel::load(path)?;
    #[cfg(not(feature = "mlx"))]
    let gpu = mlxl3_native::gpu::Metal::new()?;
    #[cfg(not(feature = "mlx"))]
    let mut model = NativeChatModel::load(&gpu, &checkpoint::inspect(path)?, 0)?;
    #[cfg(feature = "mlx")]
    let name = model.name();
    #[cfg(not(feature = "mlx"))]
    let name = model.name();
    eprintln!(
        "Loaded native {} engine (experimental, greedy sampling)",
        name
    );
    let mut messages = Vec::new();
    let interactive = prompt.is_none();
    let mut one_prompt = prompt;
    let mut image_urls = image_paths
        .iter()
        .map(|p| mlxl3_native::vision_input::file_data_url(p))
        .collect::<Result<Vec<_>>>()?;
    let mut stdout = io::stdout().lock();
    loop {
        let user = if interactive {
            write!(stdout, "\n>>> ")?;
            stdout.flush()?;
            let mut line = String::new();
            if io::stdin().read_line(&mut line)? == 0 {
                break;
            }
            let text = line.trim_end().to_owned();
            if matches!(text.as_str(), "/exit" | "/quit") {
                break;
            }
            if text == "/clear" {
                messages.clear();
                model.reset();
                continue;
            }
            if text.is_empty() {
                continue;
            }
            text
        } else {
            one_prompt.take().context("missing prompt")?
        };
        let content = if image_urls.is_empty() {
            json!(user)
        } else {
            let mut parts = vec![json!({"type":"text","text":user})];
            parts.extend(
                image_urls
                    .drain(..)
                    .map(|url| json!({"type":"image_url","image_url":{"url":url}})),
            );
            json!(parts)
        };
        messages.push(json!({"role":"user","content":content}));
        #[cfg(feature = "mlx")]
        let image_format = model.image_format();
        #[cfg(not(feature = "mlx"))]
        let image_format = model
            .supports_images()
            .then_some(mlxl3_native::vision_input::ImageFormat::Lfm3);
        let (prepared, images) =
            mlxl3_native::vision_input::prepare_messages_as(&messages, image_format)?;
        #[cfg(not(feature = "mlx"))]
        let _ = &images;
        let rendered = tokenizer.render_values(&json!(prepared), None)?;
        let rendered =
            mlxl3_native::vision_input::expand_image_placeholders(image_format, rendered, &images)?;
        let tokens = tokenizer.encode(&rendered)?;
        anyhow::ensure!(!tokens.is_empty(), "chat template produced no tokens");
        let available = (model.context_limit() as usize).saturating_sub(tokens.len());
        anyhow::ensure!(
            available > 0,
            "conversation exceeds model context; use /clear"
        );
        let budget = if max_tokens == 0 {
            available
        } else {
            max_tokens.min(available)
        };
        #[cfg(feature = "mlx")]
        if let NativeChatModel::Flash(target) = &mut model {
            target.admit_request(tokens.len() + budget)?;
        }
        // Replay keeps conversation contents exact. Qwen uses bounded QMM chunks;
        // other architectures retain their established token path.
        #[cfg(not(feature = "mlx"))]
        model.admit_request(tokens.len() + budget)?;
        model.reset();
        #[cfg(feature = "mlx")]
        model.prepare_images(&tokens, &images)?;
        let started = std::time::Instant::now();
        #[cfg(feature = "mlx")]
        let mut logits = {
            let mut last = None;
            let (chunk, group) = (model.prefill_chunk_size(), model.prefill_group()?);
            let mut start = 0;
            while let Some(end) = model.prefill_span_end(start, tokens.len(), chunk, group) {
                if end < tokens.len()
                    && model.advance_flash_prompt(
                        &tokens[start..end],
                        tokens.len(),
                        end,
                        &AtomicBool::new(false),
                    )?
                {
                    start = end;
                    continue;
                }
                last = Some(model.forward_span(&tokens[start..end], false)?.0);
                model.eval_state()?;
                start = end;
            }
            let logits = last.context("no prefill output")?;
            logits.eval()?;
            logits
        };
        #[cfg(not(feature = "mlx"))]
        let device_greedy = model.device_greedy_eligible(0., 1, 1., false);
        #[cfg(not(feature = "mlx"))]
        let mut device_greedy_next = None;
        #[cfg(not(feature = "mlx"))]
        let mut logits = model.prefill(&tokens)?;
        let prefill = started.elapsed();
        let mut decoder = tokenizer.tokenizer().decode_stream(false);
        let mut split = ThinkingSplitter::new(&rendered);
        let mut raw_text = String::new();
        let mut generated = Vec::new();
        let mut count = 0usize;
        let mut ttft = None;
        let mut current_channel = None;
        let mut emit = |fragments: Vec<mlxl3_native::streaming::Fragment>| -> Result<()> {
            for fragment in fragments {
                if current_channel != Some(fragment.channel) {
                    write!(
                        stdout,
                        "\n{}\n",
                        if fragment.channel == Channel::Thinking {
                            "— Thinking —"
                        } else {
                            "— Answer —"
                        }
                    )?;
                    current_channel = Some(fragment.channel);
                }
                write!(stdout, "{}", fragment.text)?;
            }
            stdout.flush()?;
            Ok(())
        };
        for _ in 0..budget {
            // Preserve MLX-LM's normalization before greedy selection. A raw
            // logits argmax could choose differently at FP16 rounding ties.
            #[cfg(feature = "mlx")]
            let next =
                mlxl3_native::speculative::greedy_log_prob_id(&logits.log_probs()?.to_f32()?)
                    .context("model returned invalid log probabilities or no finite token")?;
            #[cfg(not(feature = "mlx"))]
            let next = if device_greedy {
                match device_greedy_next.take() {
                    Some(token) => token,
                    None => model.greedy_from_logits(&logits)?,
                }
            } else {
                mlxl3_native::speculative::greedy_log_prob_id(&gpu.flash_log_probs(&logits)?)
                    .context("model returned invalid log probabilities or no finite token")?
            };
            if tokenizer.eos_ids().contains(&next) {
                break;
            }
            ttft.get_or_insert_with(|| started.elapsed());
            count += 1;
            generated.push(next);
            if let Some(text) = decoder
                .step(next)
                .map_err(|e| anyhow::anyhow!("token decode failed: {e}"))?
            {
                raw_text.push_str(&text);
                emit(split.feed(&text))?;
            }
            if count < budget {
                #[cfg(feature = "mlx")]
                {
                    logits = model.forward(next)?;
                }
                #[cfg(not(feature = "mlx"))]
                if device_greedy {
                    device_greedy_next = Some(model.forward_greedy(next)?);
                } else {
                    logits = model.forward(next)?;
                }
            }
        }
        // A token limit can cut a byte-fallback character. Flush the tokenizer's
        // final representation instead of silently dropping buffered bytes.
        let complete = tokenizer.decode(&generated)?;
        let tail = complete
            .strip_prefix(&raw_text)
            .context("incremental tokenizer output differs from full decoding")?;
        emit(split.feed(tail))?;
        raw_text = complete;
        emit(split.finish())?;
        let elapsed = started.elapsed();
        let decode = elapsed
            .saturating_sub(ttft.unwrap_or(elapsed))
            .as_secs_f64();
        writeln!(
            stdout,
            "\n[prefill {:.1} tok/s · decode {:.1} tok/s · TTFT {:.0} ms · {} tokens]",
            tokens.len() as f64 / prefill.as_secs_f64().max(1e-9),
            count.saturating_sub(1) as f64 / decode.max(1e-9),
            ttft.unwrap_or(elapsed).as_secs_f64() * 1000.0,
            count
        )?;
        stdout.flush()?;
        messages.push(json!({"role":"assistant","content":raw_text}));
        if !interactive {
            break;
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "mlx", feature = "chat"))]
mod tests {
    use super::*;

    thread_local! {
        pub(super) static FLASH_ADVANCED_ENDS: std::cell::RefCell<Vec<usize>> = const { std::cell::RefCell::new(Vec::new()) };
    }

    #[test]
    #[ignore = "physical Apple GPU and MLXL3_FLASH_MODEL; run once per prefill configuration"]
    fn flash_prefill_round_matches_scalar_continuation() -> Result<()> {
        use mlxl3_native::array::{Array, Dtype};
        let finite_logits = |a: Array| -> Result<Vec<u8>> {
            let values = a.astype(Dtype::Float32)?.to_f32()?;
            anyhow::ensure!(values.len() == 248320 && values.iter().all(|v| v.is_finite()));
            a.to_bytes()
        };
        let enabled = |key| std::env::var(key).as_deref() == Ok("1");
        let expected_ends: Vec<usize> = if enabled("MLXL3_FLASH_PREFILL_EXPERT4")
            && enabled("MLXL3_FLASH_PREFILL_SERIAL_TAIL")
        {
            (8..=208).step_by(8).collect()
        } else if enabled("MLXL3_FLASH_PREFILL_EXPERT4") || enabled("MLXL3_FLASH_PROMPT_FINAL_HEAD")
        {
            (8..=256).step_by(8).collect()
        } else {
            vec![]
        };
        let tokens: Vec<_> = (0..257).map(|i| 18 + (i * 193) % 8000).collect();
        let mut model =
            NativeChatModel::load(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
        anyhow::ensure!(matches!(model, NativeChatModel::Flash(_)));
        let mut last = None;
        for &token in &tokens {
            last = Some(model.forward(token)?);
        }
        let expected_last = finite_logits(last.context("missing scalar logits")?)?;
        let expected_next = finite_logits(model.forward(28)?)?;
        FLASH_ADVANCED_ENDS.with_borrow_mut(Vec::clear);
        let mut cache = None;
        let (logits, draft, cached) = prefill_round(
            &mut model,
            &tokens,
            &[],
            None,
            &mut cache,
            "flash-review",
            true,
            &AtomicBool::new(false),
        )?;
        anyhow::ensure!(draft.is_none() && cached == 0 && cache.is_none());
        anyhow::ensure!(finite_logits(logits)? == expected_last);
        anyhow::ensure!(finite_logits(model.forward(28)?)? == expected_next);
        FLASH_ADVANCED_ENDS.with_borrow(|actual| assert_eq!(actual, &expected_ends));
        Ok(())
    }

    /// DSPARK-INK2M-P10c: sampled rows on the 8-bit target take the measured costs; greedy rows, the FP16
    /// target and `p4c` the P4c linear costs; any other policy name is refused.
    #[test]
    fn inkling_width_policy_follows_target_precision_and_mode() -> Result<()> {
        use mlxl3_native::inkling2_draft::WidthPolicy;
        for policy in [None, Some(""), Some("p10")] {
            let sampled = inkling_width_policy(policy, 8, false, 3)?;
            for w in 0..=3 {
                assert_eq!(sampled.cost(w), WidthPolicy::SAMPLED_8BIT_COSTS[w]);
            }
        }
        for (policy, bits, greedy) in [
            (None, 8, true),
            (Some("p10"), 8, true),
            (None, 16, false),
            (Some(""), 16, false),
            (Some("p4c"), 8, false),
            (Some("p4c"), 16, true),
        ] {
            let p = inkling_width_policy(policy, bits, greedy, 3)?;
            for w in 0..=3 {
                assert_eq!(p.cost(w), WidthPolicy::linear_cost(w), "{policy:?} {bits}");
            }
        }
        assert!(inkling_width_policy(Some("fast"), 8, false, 3).is_err());
        assert!(inkling_width_policy(None, 8, false, 4).is_err());
        Ok(())
    }

    /// SPEED-INK2M-P14: the GPU draw follows the sampler, "0" turns the draw or the pipeline off, and
    /// nothing runs ahead without the GPU draw.
    #[test]
    fn inkling_device_draw_follows_sampler_and_flags() -> Result<()> {
        use mlxl3_native::{inkling2::DrawRule, sampling::Sampler};
        let app = Sampler::new(1.0, 0, 1.0, 1.0, 0.0)?;
        let greedy = Sampler::greedy();
        assert_eq!(
            inkling_device_draw(&app, None, None),
            Some((DrawRule::Softmax(1.0), true))
        );
        assert_eq!(
            inkling_device_draw(&greedy, Some("1"), Some("1")),
            Some((DrawRule::Greedy, true))
        );
        assert_eq!(
            inkling_device_draw(&app, None, Some("0")),
            Some((DrawRule::Softmax(1.0), false))
        );
        assert_eq!(inkling_device_draw(&app, Some("0"), None), None);
        assert_eq!(inkling_device_draw(&greedy, Some("0"), Some("1")), None);
        for sampler in [
            Sampler::new(1.0, 40, 1.0, 1.0, 0.0)?,
            Sampler::new(1.0, 0, 0.9, 1.0, 0.0)?,
            Sampler::new(0.7, 0, 1.0, 1.05, 0.0)?,
            Sampler::new(0.7, 0, 1.0, 1.0, 0.3)?,
        ] {
            assert_eq!(
                inkling_device_draw(&sampler, None, None),
                None,
                "{sampler:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn moe_codec_rejects_invalid_routes_before_gpu_or_checkpoint_loading() -> Result<()> {
        for request in [
            json!({"op":"mlx-switch","key_heads":2,"cols":128,"top_k":1}),
            json!({"op":"mlx-switch","key_heads":2,"cols":128,"selected":[2]}),
            json!({"op":"mlx-switch","key_heads":2,"cols":128,"selected":[0,1,0],"top_k":2}),
            json!({"op":"mlx-checkpoint-switch","key_heads":256,"selected":[256],"x":[0],"path":"/missing"}),
            json!({"op":"mlx-checkpoint-switch","key_heads":256,"x":[0],"path":"/missing"}),
            json!({"op":"mlx-swiglu-prepare","cols":128,"suh":vec![0;256],"selected":[1]}),
            json!({"op":"mlx-swiglu-prepare","cols":0,"selected":[0]}),
        ] {
            let request: CodecRequest = serde_json::from_value(request)?;
            assert!(request.validate_moe_codec().is_err());
        }
        let request: CodecRequest = serde_json::from_value(
            json!({"op":"mlx-switch","key_heads":2,"cols":128,"selected":[1],"x":vec![0;128]}),
        )?;
        request.validate_moe_codec()?;
        let request: CodecRequest = serde_json::from_value(
            json!({"op":"mlx-swiglu-prepare","cols":128,"selected":[1],"suh":vec![0;512]}),
        )?;
        request.validate_moe_codec()?;
        Ok(())
    }

    #[test]
    fn mlx_checkpoint_codec_keeps_the_requested_projection_prefix() {
        let request: CodecRequest =
            serde_json::from_str(r#"{"op":"mlx-checkpoint-switch","prefix":"model.layers.0.mlp"}"#)
                .unwrap();
        assert_eq!(request.prefix, "model.layers.0.mlp");
        let request: CodecRequest = serde_json::from_str(r#"{"op":"pack"}"#).unwrap();
        assert!(request.prefix.is_empty());
    }

    #[test]
    #[ignore = "requires local Flash target/DFlash checkpoints and Apple GPU; run alone"]
    fn mtp_preparation_retires_flash_draft_before_invalid_head_is_read() -> Result<()> {
        let path = PathBuf::from(std::env::var("MLXL3_FLASH_MODEL")?);
        let draft = PathBuf::from(std::env::var("MLXL3_FLASH_DRAFT_MODEL")?);
        let mut model = NativeChatModel::load(&path)?;
        anyhow::ensure!(
            matches!(&model, NativeChatModel::Flash(_)),
            "requires a Flash Next target"
        );
        let baseline = model.forward(0)?.to_f32()?;
        anyhow::ensure!(
            baseline.len() == 248_320 && baseline.iter().all(|x| x.is_finite()),
            "invalid reference logits"
        );
        model.reset();
        let directory = tempfile::tempdir()?;
        let missing = directory.path().join("absent-mtp-head");
        let missing = missing.to_str().context("non-UTF8 temporary path")?;
        let mut head = None;
        let mut revision = None;
        let mut cache = None;
        let mut packed_draft = None;
        for requested in ["", missing] {
            let NativeChatModel::Flash(target) = &mut model else {
                unreachable!();
            };
            target.configure_dflash2(Some(&draft), 1)?;
            assert!(target.dflash2_active(), "fixture must own a live draft");
            // This exact preparer is used by generate and set_mtp. Rejection of
            // an invalid MTP path must happen after retiring the previous mode.
            assert!(
                load_mtp_head(
                    &mut head,
                    &mut revision,
                    &mut cache,
                    requested,
                    &mut model,
                    &mut packed_draft,
                    &AtomicBool::new(false),
                )
                .is_err()
            );
            assert!(matches!(&model, NativeChatModel::Flash(target) if !target.dflash2_active()));
            assert!(head.is_none() && revision.is_none() && cache.is_none());
            assert!(packed_draft.is_none());
            model.reset();
            let actual = model.forward(0)?.to_f32()?;
            assert_eq!(actual.len(), baseline.len());
            assert!(actual.iter().all(|x| x.is_finite()));
            assert!(
                actual
                    .iter()
                    .zip(&baseline)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "ordinary target logits changed after a failed mode switch"
            );
            model.reset();
        }
        mlxl3_native::array::synchronize()?;
        Ok(())
    }

    #[test]
    fn dflash_width_tracks_recent_acceptance() {
        let mut session = DFlashChat::new();
        session.blocks = 16;
        session.accepted = 48;
        session.proposed = 80;
        session.recent = std::iter::repeat_n((1, 5), 8).collect();
        assert_eq!(session.proposals_count(128, 128), Some(2));
        session.recent = std::iter::repeat_n((4, 5), 7).collect();
        assert_eq!(session.proposals_count(128, 128), Some(5));
        session.recent.push_back((4, 5));
        assert_eq!(session.proposals_count(128, 128), Some(5));
        assert_eq!(session.proposals_count(2, 128), Some(1));
        session.mode = 2;
        assert_eq!(session.proposals_count(128, 128), Some(2));
    }

    #[test]
    fn native_sampler_respects_greedy_argmax() {
        let logits = mlxl3_native::array::Array::from_f32(&[1., 4., 2.], &[1, 3]).unwrap();
        let mut random = 1;
        assert_eq!(
            select_token(
                &logits,
                &mlxl3_native::sampling::History::default(),
                &mlxl3_native::sampling::Sampler::greedy(),
                &mut random,
            )
            .unwrap(),
            1
        );
    }

    #[test]
    fn bridge_token_hash_uses_ordered_token_ids() {
        assert_eq!(token_hash(&[]), "cbf29ce484222325");
        assert_ne!(token_hash(&[1, 2]), token_hash(&[2, 1]));
    }

    #[test]
    fn dflash_tree_setting_is_on_by_default_and_bounded() {
        assert!(dflash_tree_sampled(None) && dflash_tree_sampled(Some("1")));
        assert_eq!(dflash_relax_kl(None), 0.01);
        assert_eq!(dflash_relax_kl(Some("0")), 0.);
        assert_eq!(dflash_relax_kl(Some("0.02")), 0.02);
        for bad in ["-0.1", "2", "NaN", "inf", "x"] {
            assert_eq!(dflash_relax_kl(Some(bad)), 0.01, "{bad}");
        }
        assert!(!dflash_tree_sampled(Some("0")));
        assert_eq!(dflash_tree_nodes(None, None), 11);
        assert_eq!(dflash_tree_nodes(Some("0"), Some("11")), 0);
        assert_eq!(dflash_tree_nodes(Some("1"), None), 11);
        assert_eq!(dflash_tree_nodes(None, Some("7")), 7);
        assert_eq!(dflash_tree_nodes(Some("1"), Some("7")), 7);
        assert_eq!(dflash_tree_nodes(Some("1"), Some("15")), 15);
        for bad in ["6", "16", "", "x", "-1"] {
            assert_eq!(dflash_tree_nodes(Some("1"), Some(bad)), 11);
        }
    }

    #[test]
    fn dflash2_is_offered_for_the_dense_27b_and_opt_in_for_the_35b() {
        use mlxl3_native::dflash::Family;
        assert!(dflash_family_offered(Family::Qwen38Dense, None));
        assert!(dflash_family_offered(Family::Qwen38Dense, Some("0")));
        assert!(!dflash_family_offered(Family::Qwen36Moe, None));
        for value in ["", "0", "true", "1 ", "01"] {
            assert!(!dflash_family_offered(Family::Qwen36Moe, Some(value)));
        }
        assert!(dflash_family_offered(Family::Qwen36Moe, Some("1")));
    }

    #[test]
    fn dflash_runtime_key_distinguishes_copy_policy() {
        assert_eq!(
            dflash_runtime_key("model:runtime", false, None),
            "model:runtime:dflash-v4:copy=0"
        );
        assert_eq!(
            dflash_runtime_key("model:runtime", true, None),
            "model:runtime:dflash-v4:copy=1"
        );
        assert_eq!(
            dflash_runtime_key("model:runtime", false, Some(3)),
            "model:runtime:dflash-v4:copy=0:cap=3"
        );
        assert!(!dflash_context_copy_enabled(None));
        if std::env::var_os("MLXL3_DFLASH_CONTEXT_COPY").is_none() {
            assert!(dflash_context_copy_enabled(Some(
                mlxl3_native::dflash::Family::Qwen38Dense
            )));
            assert!(!dflash_context_copy_enabled(Some(
                mlxl3_native::dflash::Family::Qwen36Moe
            )));
        }
    }

    #[test]
    fn bridge_stats_merge_topk_readback_rows_and_omit_them_when_off() {
        let mut total = NativeStats::default();
        total.accumulate(&NativeStats::default());
        let plain = serde_json::to_value(&total).unwrap();
        assert!(plain.get("topk_readback_rows").is_none());
        total.accumulate(&NativeStats {
            topk_readback_rows: Some([5, 1]),
            ..Default::default()
        });
        total.accumulate(&NativeStats::default());
        total.accumulate(&NativeStats {
            topk_readback_rows: Some([2, 0]),
            ..Default::default()
        });
        let json = serde_json::to_value(&total).unwrap();
        assert_eq!(json["topk_readback_rows"], serde_json::json!([7, 1]));
    }

    #[test]
    fn bridge_stats_merge_mtp_accept_histograms_and_omit_them_without_mtp() {
        use mlxl3_native::speculative::AcceptHistogram;
        let histogram = |cycles: &[(usize, usize)]| {
            let mut h = AcceptHistogram::default();
            for &(width, accepted) in cycles {
                assert!(h.record(width, accepted));
            }
            h
        };
        let mut total = NativeStats::default();
        let plain = serde_json::to_value(&total).unwrap();
        assert!(plain.get("mtp_accept_histogram").is_none());
        assert!(plain.get("mtp_lookup_accept_histogram").is_none());
        total.accumulate(&NativeStats {
            mtp_accept_histogram: Some(histogram(&[(2, 2), (2, 0), (0, 0)])),
            mtp_lookup_accept_histogram: Some(histogram(&[(3, 1)])),
            ..Default::default()
        });
        total.accumulate(&NativeStats::default());
        total.accumulate(&NativeStats {
            mtp_accept_histogram: Some(histogram(&[(2, 2), (1, 1)])),
            ..Default::default()
        });
        let json = serde_json::to_value(&total).unwrap();
        assert_eq!(
            json["mtp_accept_histogram"],
            serde_json::json!([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 2, 0], [0, 0, 0, 0]])
        );
        assert_eq!(
            json["mtp_lookup_accept_histogram"],
            serde_json::json!([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 0]])
        );
    }

    #[test]
    fn bridge_stats_aggregate_work_not_rates_and_keep_latest_context() {
        let mut first = NativeStats {
            ttft_seconds: 0.5,
            prompt_tokens: 100,
            evaluated_prompt_tokens: 100,
            generated_tokens: 11,
            prefill_seconds: 0.25,
            decode_seconds: 1.,
            decode_tokens: 10,
            round_count: 1,
            dflash_profile: Some(mlxl3_native::qwen4::FlashDraftProfile {
                draft_seconds: 0.25,
                verification_seconds: 0.5,
                commit_cache_seconds: 0.125,
                verified_rows: 12,
                drafted_rows: 16,
                drafted_widths: [1, 2, 3, 4, 5, 6, 7, 8],
                verified_widths: [1, 2, 3, 4, 5, 6, 7, 8],
                expert_demand_bytes: 1_248_012,
                ssd_prefetch: [1, 2, 3, 4, 5, 6, 7, 8],
            }),
            dflash_proposed_tokens: Some(10),
            dflash_accepted_tokens: Some(6),
            dflash_blocks: Some(2),
            dflash_lookup_blocks: Some(1),
            mtp_lookup_blocks: Some(3),
            ..Default::default()
        };
        first.accumulate(&NativeStats {
            ttft_seconds: 2.,
            prompt_tokens: 200,
            evaluated_prompt_tokens: 200,
            generated_tokens: 31,
            prefill_seconds: 0.75,
            decode_seconds: 1.5,
            decode_tokens: 30,
            context_used: 231,
            context_limit: 4096,
            round_count: 1,
            dflash_profile: Some(mlxl3_native::qwen4::FlashDraftProfile {
                draft_seconds: 0.125,
                verification_seconds: 1.,
                commit_cache_seconds: 0.125,
                verified_rows: 48,
                drafted_rows: 64,
                drafted_widths: [8, 7, 6, 5, 4, 3, 2, 1],
                verified_widths: [8, 7, 6, 5, 4, 3, 2, 1],
                expert_demand_bytes: 2_496_024,
                ssd_prefetch: [8, 7, 6, 5, 4, 3, 2, 1],
            }),
            dflash_proposed_tokens: Some(40),
            dflash_accepted_tokens: Some(22),
            dflash_blocks: Some(8),
            dflash_lookup_blocks: Some(3),
            mtp_lookup_blocks: Some(4),
            ..Default::default()
        });
        assert_eq!(
            (
                first.prompt_tokens,
                first.generated_tokens,
                first.round_count
            ),
            (300, 42, 2)
        );
        assert_eq!(
            (first.prefill_tps, first.decode_tps, first.ttft_seconds),
            (300., 16., 0.5)
        );
        assert_eq!((first.context_used, first.context_limit), (231, 4096));
        assert_eq!(first.mtp_lookup_blocks, Some(7));
        assert_eq!(first.dflash_lookup_blocks, Some(4));
        let profile = first.dflash_profile.unwrap();
        assert_eq!(
            (
                profile.draft_seconds,
                profile.verification_seconds,
                profile.commit_cache_seconds
            ),
            (0.375, 1.5, 0.25)
        );
        assert_eq!(
            (profile.verified_rows, profile.expert_demand_bytes),
            (60, 3_744_036)
        );
        let encoded = serde_json::to_value(&first).unwrap();
        assert_eq!(encoded["dflash_profile"]["verified_rows"], 60);
        assert_eq!(encoded["dflash_profile"]["verification_seconds"], 1.5);
        assert_eq!(profile.ssd_prefetch, [9; 8]);
        assert_eq!(profile.drafted_rows, 80);
        assert_eq!(profile.drafted_widths, [9; 8]);
        assert_eq!(profile.verified_widths, [9; 8]);
        assert_eq!(
            encoded["dflash_profile"]["ssd_prefetch"],
            serde_json::to_value([9; 8]).unwrap()
        );
        assert_eq!(
            (
                first.dflash_proposed_tokens,
                first.dflash_accepted_tokens,
                first.dflash_blocks
            ),
            (Some(50), Some(28), Some(10))
        );
    }

    #[test]
    #[ignore = "requires local Qwen/MTP checkpoints, MLXL3_MTP_LOOKUP=1 and Apple GPU"]
    fn bridge_lookup_seeds_prompt_and_preserves_budget_and_prefix() -> Result<()> {
        use mlxl3_native::{mtp::Head, tokenizer::ChatTokenizer};
        anyhow::ensure!(
            std::env::var("MLXL3_MTP_LOOKUP").as_deref() == Ok("1"),
            "run this test with MLXL3_MTP_LOOKUP=1"
        );
        let path = std::path::PathBuf::from(
            std::env::var("MLXL3_MTP_TEST_MODEL")
                .unwrap_or_else(|_| "models/Qwen3.6-35B-A3B-EXL3-2.49bpw".into()),
        );
        let mut model = NativeChatModel::load(&path)?;
        let tokenizer = ChatTokenizer::load(&path)?;
        let NativeChatModel::Qwen(target) = &model else {
            bail!("requires Qwen target")
        };
        let mut head = ChatMtpHead::Qwen(Box::new(Head::load(
            std::path::Path::new(
                &std::env::var("MLXL3_MTP_TEST_HEAD")
                    .unwrap_or_else(|_| "models/Qwen3.6-35B-A3B-MTP-4bit".into()),
            ),
            target,
        )?));
        let code = "def normalize_name(value):\n    return value.strip().casefold()\n\ndef average(values):\n    return sum(values) / len(values) if values else 0\n\n";
        let messages = vec![json!({"role":"user", "content":format!(
            "Recopie exactement le code suivant, sans commentaire ni changement :\n```python\n{}```", code.repeat(8))})];
        let cancelled = AtomicBool::new(false);
        let mut cache = None;
        let mut random = 7;
        let mut lookup_blocks = 0;
        let copy_budget = std::env::var("MLXL3_MTP_TEST_COPY_BUDGET")
            .ok()
            .map(|value| value.parse::<i64>())
            .transpose()?
            .unwrap_or(256);
        anyhow::ensure!(
            (256..=1024).contains(&copy_budget),
            "copy test budget must be 256..1024"
        );
        for budget in [1, 2, 3, 17, copy_budget] {
            let baseline = bridge_generate_round(
                &mut model,
                &tokenizer,
                "lookup-reference",
                &messages,
                &[],
                budget,
                mlxl3_native::sampling::Sampler::greedy(),
                4096,
                0.,
                &cancelled,
                &mut random,
                None,
                None,
                1,
                1,
                &mut cache,
                "lookup",
                false,
                false,
                &[],
                None,
            )?;
            assert!(!baseline.raw.is_empty());
            assert!(
                baseline.stats.generated_tokens > 0
                    && baseline.stats.generated_tokens <= budget as usize
            );
            for depth in 1..=3 {
                for pass in 0..2 {
                    let actual = bridge_generate_round(
                        &mut model,
                        &tokenizer,
                        "lookup-candidate",
                        &messages,
                        &[],
                        budget,
                        mlxl3_native::sampling::Sampler::greedy(),
                        4096,
                        0.,
                        &cancelled,
                        &mut random,
                        None,
                        Some(&mut head),
                        depth,
                        1,
                        &mut cache,
                        "lookup",
                        true,
                        false,
                        &[],
                        None,
                    )?;
                    assert_eq!(actual.raw, baseline.raw);
                    assert_eq!(actual.token_hash, baseline.token_hash);
                    assert_eq!(
                        actual.stats.generated_tokens,
                        baseline.stats.generated_tokens
                    );
                    lookup_blocks += actual.stats.mtp_lookup_blocks.unwrap();
                    // The second request at each depth can reuse the exact prefix.
                    if pass == 1 {
                        assert!(actual.stats.cached_prompt_tokens >= 256);
                    }
                }
            }
        }
        assert!(
            lookup_blocks > 0,
            "the production prompt must exercise lookup drafting"
        );
        let messages = vec![json!({"role":"user", "content":
            "Write complete Python code for a strict JSON Lines reader. Reject malformed records with the line number, bound record sizes, and include a small example. Explain the failure cases."})];
        let mut run = |mtp| {
            bridge_generate_round(
                &mut model,
                &tokenizer,
                "lookup-no-break",
                &messages,
                &[],
                128,
                mlxl3_native::sampling::Sampler::greedy(),
                4096,
                0.,
                &cancelled,
                &mut random,
                None,
                mtp,
                2,
                1,
                &mut cache,
                "lookup",
                false,
                false,
                &[],
                None,
            )
        };
        let baseline = run(None)?;
        let actual = run(Some(&mut head))?;
        assert!(!actual.raw.is_empty());
        assert_eq!(actual.raw, baseline.raw);
        assert_eq!(actual.token_hash, baseline.token_hash);
        assert_eq!(actual.stats.generated_tokens, 128);
        assert_eq!(
            actual.stats.mtp_lookup_blocks,
            Some(0),
            "do not draft an in-line reformulation"
        );
        eprintln!(
            "production lookup: 37 nonempty completions, {lookup_blocks} copy lookup blocks; in-line echo skipped"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen/MTP checkpoints (MLXL3_MTP_TEST_MODEL/HEAD) and Apple GPU; run alone"]
    fn bridge_mtp_sampling_matches_autoregressive_policy() -> Result<()> {
        use mlxl3_native::{mtp::Head, sampling::Sampler, tokenizer::ChatTokenizer};
        let path = std::path::PathBuf::from(
            std::env::var("MLXL3_MTP_TEST_MODEL")
                .unwrap_or_else(|_| "models/Qwen3.6-35B-A3B-EXL3-2.49bpw".into()),
        );
        let mut model = NativeChatModel::load(&path)?;
        let tokenizer = ChatTokenizer::load(&path)?;
        let NativeChatModel::Qwen(target) = &model else {
            bail!("requires Qwen target")
        };
        let mut head = ChatMtpHead::Qwen(Box::new(Head::load(
            std::path::Path::new(
                &std::env::var("MLXL3_MTP_TEST_HEAD")
                    .unwrap_or_else(|_| "models/Qwen3.6-35B-A3B-MTP-4bit".into()),
            ),
            target,
        )?));
        let messages = vec![json!({"role":"user", "content":
            "List ten distinct uses of a paperclip, one per line, then explain which is most surprising."})];
        let cancelled = AtomicBool::new(false);
        let mut cache = None;
        let mut run = |sampler, mtp: Option<&mut ChatMtpHead>, depth, seed| {
            let mut random = seed;
            bridge_generate_round(
                &mut model,
                &tokenizer,
                "mtp-sampling",
                &messages,
                &[],
                96,
                sampler,
                4096,
                0.,
                &cancelled,
                &mut random,
                None,
                mtp,
                depth,
                1,
                &mut cache,
                "mtp-sampling",
                false,
                false,
                &[],
                None,
            )
        };
        // Deterministic but penalized: the sampled verifier runs (not the
        // greedy one) and every delivered token is a point mass, so MTP must
        // reproduce autoregressive decoding token for token. A misaligned
        // row, penalty prefix or bonus token diverges here.
        let penalized = Sampler::new(0., 0, 1., 1.1, 0.5)?;
        assert!(!penalized.is_greedy());
        let reference = run(penalized, None, 1, 7)?;
        assert_eq!(reference.stats.generated_tokens, 96);
        for depth in 1..=3 {
            let actual = run(penalized, Some(&mut head), depth, 7)?;
            assert_eq!(actual.raw, reference.raw, "penalized depth {depth}");
            assert_eq!(actual.token_hash, reference.token_hash);
            assert!(actual.stats.mtp_accepted_tokens.unwrap() > 0);
        }
        // Stochastic official preset: MTP must run, accept drafts, finish
        // the budget and be reproducible for a seed.
        let stochastic = Sampler::new(0.7, 20, 0.8, 1., 1.5)?;
        let mut accepted = 0;
        for depth in 1..=3 {
            let first = run(stochastic, Some(&mut head), depth, 11)?;
            let again = run(stochastic, Some(&mut head), depth, 11)?;
            assert!(!first.raw.is_empty());
            assert_eq!(first.raw, again.raw, "seeded stochastic depth {depth}");
            assert_eq!(first.stats.generated_tokens, again.stats.generated_tokens);
            let (proposed, hits) = (
                first.stats.mtp_proposed_tokens.unwrap(),
                first.stats.mtp_accepted_tokens.unwrap(),
            );
            assert!(hits <= proposed);
            accepted += hits;
            eprintln!(
                "stochastic MTP depth {depth}: {} tokens, {hits}/{proposed} drafts accepted",
                first.stats.generated_tokens
            );
        }
        assert!(accepted > 0, "stochastic MTP never accepted a draft");
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen/DFlash checkpoints and Apple GPU"]
    fn dflash_copy_accepts_rejects_and_preserves_draft_cache() -> Result<()> {
        use mlxl3_native::{dflash::DFlashWeights, speculative::DFlashCopy};
        let path = std::path::Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw");
        let mut target = mlxl3_native::qwen35::Qwen35Moe::load(path)?;
        let weights = DFlashWeights::load(&mlxl3_native::dflash::inspect(
            "models/Qwen3.6-35B-A3B-DFlash2",
        )?)?;
        let prompt = (1..=24).collect::<Vec<u32>>();
        let finite_bytes = |array: &mlxl3_native::array::Array| -> Result<Vec<u8>> {
            let values = array.to_f32()?;
            let count = array.shape().iter().try_fold(1usize, |count, &dim| {
                count.checked_mul(usize::try_from(dim).ok()?)
            });
            anyhow::ensure!(
                count == Some(values.len())
                    && !values.is_empty()
                    && values.iter().all(|v| v.is_finite()),
                "invalid copy oracle array"
            );
            array.to_bytes()
        };
        let mut cases = 0;
        for width in 1..=3 {
            for accepted in 0..=width {
                target.reset();
                let mut session = DFlashChat::new();
                let anchor = session
                    .prefill(&mut target, &weights, &prompt)?
                    .chat_greedy_ids()?[0];
                let before = target.snapshot()?;
                let mut reference_cache = session.cache.try_clone()?;
                let mut caches = Vec::new();
                let mut expected = Vec::new();
                let mut token = anchor;
                for _ in 0..width + 2 {
                    let position = target.offset();
                    let (logits, captures) = target.prefill_serial_with_dflash_capture(&[token])?;
                    reference_cache.append_captured(&weights, &captures, position)?;
                    reference_cache.eval()?;
                    token = logits.chat_greedy_ids()?[0];
                    expected.push(token);
                    caches.push(reference_cache.try_clone()?);
                }
                target.restore(before)?;
                let mut proposals = expected[..width].to_vec();
                if accepted < width {
                    proposals[accepted] = (proposals[accepted] + 1) % 248_320;
                }
                let history = (1..=7)
                    .chain([anchor])
                    .chain(proposals.iter().copied())
                    .chain(1..=7)
                    .collect::<Vec<_>>();
                session.copy = Some(DFlashCopy::new(&history));
                assert_eq!(
                    session.copy.as_ref().unwrap().propose(anchor, width),
                    Some(proposals.as_slice())
                );
                let mut delivered = Vec::new();
                for i in 0..=accepted {
                    delivered.push(session.advance(
                        &mut target,
                        &weights,
                        if i == 0 { anchor } else { delivered[i - 1] },
                        4096,
                        width + 1 - i,
                    )?);
                }
                assert_eq!(delivered, expected[..=accepted]);
                assert_eq!(session.lookup_blocks, 1);
                assert!(session.pending.is_empty());
                assert_eq!(target.offset(), 24 + accepted as i32 + 1);
                let input =
                    target.dflash_input(*delivered.last().unwrap(), weights.family.mask())?;
                let actual = weights.forward_hidden(&input, &session.cache, target.offset())?;
                let oracle = weights.forward_hidden(&input, &caches[accepted], target.offset())?;
                assert_eq!(finite_bytes(&actual.hidden)?, finite_bytes(&oracle.hidden)?);
                assert_eq!(
                    finite_bytes(&actual.selector)?,
                    finite_bytes(&oracle.selector)?
                );
                let next =
                    session.advance(&mut target, &weights, *delivered.last().unwrap(), 4096, 4)?;
                assert_eq!(
                    next,
                    expected[accepted + 1],
                    "neural continuation after copy"
                );
                cases += 1;
            }
        }
        assert_eq!(cases, 9);
        eprintln!(
            "9 production copy blocks, all accept0..N/N1..3, finite hidden/selector and draft cache bit exact; neural continuation exact"
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires Qwen/DFlash checkpoints, MLXL3_DFLASH_CONTEXT_COPY=1 and Apple GPU"]
    fn bridge_dflash_copy_seeds_prompt_and_preserves_budget_and_prefix() -> Result<()> {
        anyhow::ensure!(
            dflash_context_copy_enabled(Some(mlxl3_native::dflash::Family::Qwen36Moe)),
            "run with MLXL3_DFLASH_CONTEXT_COPY=1"
        );
        let path = std::path::Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw");
        let mut model = NativeChatModel::load(path)?;
        let tokenizer = mlxl3_native::tokenizer::ChatTokenizer::load(path)?;
        let weights = mlxl3_native::dflash::DFlashWeights::load(&mlxl3_native::dflash::inspect(
            "models/Qwen3.6-35B-A3B-DFlash2",
        )?)?;
        let code = "def normalize_name(value):\n    return value.strip().casefold()\n\ndef average(values):\n    return sum(values) / len(values) if values else 0\n\n";
        let messages = vec![
            json!({"role":"user", "content":format!("Recopie exactement le code suivant, sans commentaire ni changement :\n```python\n{}```", code.repeat(8))}),
        ];
        let cancelled = AtomicBool::new(false);
        let mut cache = None;
        let mut random = 7;
        let mut copies = 0;
        let mut completions = 0;
        for budget in [1, 2, 3, 17, 256] {
            let reference = bridge_generate_round(
                &mut model,
                &tokenizer,
                "copy-reference",
                &messages,
                &[],
                budget,
                mlxl3_native::sampling::Sampler::greedy(),
                4096,
                0.,
                &cancelled,
                &mut random,
                None,
                None,
                1,
                1,
                &mut cache,
                "dflash-copy",
                false,
                false,
                &[],
                None,
            )?;
            assert!(
                !reference.raw.is_empty()
                    && reference.stats.generated_tokens > 0
                    && reference.stats.generated_tokens <= budget as usize
            );
            for mode in 1..=3 {
                for pass in 0..2 {
                    let actual = bridge_generate_round(
                        &mut model,
                        &tokenizer,
                        "copy-candidate",
                        &messages,
                        &[],
                        budget,
                        mlxl3_native::sampling::Sampler::greedy(),
                        4096,
                        0.,
                        &cancelled,
                        &mut random,
                        Some(&weights),
                        None,
                        1,
                        mode,
                        &mut cache,
                        "dflash-copy",
                        true,
                        false,
                        &[],
                        None,
                    )?;
                    assert_eq!(actual.raw, reference.raw);
                    assert_eq!(actual.token_hash, reference.token_hash);
                    assert_eq!(
                        actual.stats.generated_tokens,
                        reference.stats.generated_tokens
                    );
                    if pass == 1 {
                        assert!(actual.stats.cached_prompt_tokens >= 256);
                    }
                    let count = actual.stats.dflash_lookup_blocks.unwrap();
                    if mode != 1 {
                        assert_eq!(count, 0, "fixed modes keep their neural proposals");
                    }
                    copies += count;
                    completions += 1;
                }
            }
            completions += 1;
        }
        assert_eq!(completions, 35);
        assert!(copies > 0, "the production bridge must actually copy");
        eprintln!(
            "35 nonempty completions with budgets/cache/Auto/fixed modes exact; {copies} verified copy blocks"
        );
        Ok(())
    }

    fn q27_dflash_fixture() -> Result<(
        NativeChatModel,
        mlxl3_native::tokenizer::ChatTokenizer,
        mlxl3_native::dflash::DFlashWeights,
    )> {
        let path = std::path::PathBuf::from(std::env::var("MLXL3_DFLASH_TEST_MODEL")?);
        let model = NativeChatModel::load(&path)?;
        let tokenizer = mlxl3_native::tokenizer::ChatTokenizer::load(&path)?;
        let package = mlxl3_native::dflash::inspect(std::env::var("MLXL3_DFLASH_TEST_DRAFT")?)?;
        let weights = mlxl3_native::dflash::DFlashWeights::load(&package)?;
        Ok((model, tokenizer, weights))
    }

    /// Q27D-09 (D2): sampled DFlash2 keeps the target policy exactly.
    #[test]
    #[ignore = "requires MLXL3_DFLASH_TEST_MODEL/DRAFT (Qwen + packed DFlash2) and Apple GPU; run alone"]
    fn bridge_dflash_sampling_matches_autoregressive_policy() -> Result<()> {
        use mlxl3_native::sampling::Sampler;
        let (mut model, tokenizer, weights) = q27_dflash_fixture()?;
        let messages = vec![json!({"role":"user", "content":
            "List ten distinct uses of a paperclip, one per line, then explain which is most surprising."})];
        let cancelled = AtomicBool::new(false);
        let mut cache = None;
        let mut run = |sampler, draft: Option<&RoundDraft>, mode, seed| {
            let mut random = seed;
            bridge_generate_round(
                &mut model,
                &tokenizer,
                "dflash-sampling",
                &messages,
                &[],
                96,
                sampler,
                4096,
                0.,
                &cancelled,
                &mut random,
                draft,
                None,
                1,
                mode,
                &mut cache,
                "dflash-sampling",
                false,
                false,
                &[],
                None,
            )
        };
        // Greedy: every mode (including the capped adaptive one) equals autoregressive IDs.
        let greedy = Sampler::new(0., 1, 1., 1., 0.)?;
        assert!(greedy.is_greedy());
        let reference = run(greedy, None, 1, 3)?;
        assert_eq!(reference.stats.generated_tokens, 96);
        for mode in 1..=3 {
            let actual = run(greedy, Some(&weights), mode, 3)?;
            assert_eq!(actual.raw, reference.raw, "greedy mode {mode}");
            assert_eq!(actual.token_hash, reference.token_hash);
            assert!(actual.stats.dflash_accepted_tokens.unwrap() > 0);
        }
        // Deterministic but penalized: the sampled verifier runs and every delivered
        // token is a point mass, so DFlash2 must reproduce autoregressive decoding.
        // A misaligned row, penalty prefix or corrective token diverges here.
        let penalized = Sampler::new(0., 0, 1., 1.1, 0.5)?;
        assert!(!penalized.is_greedy());
        let reference = run(penalized, None, 1, 7)?;
        assert_eq!(reference.stats.generated_tokens, 96);
        for mode in 1..=3 {
            let actual = run(penalized, Some(&weights), mode, 7)?;
            assert_eq!(actual.raw, reference.raw, "penalized mode {mode}");
            assert_eq!(actual.token_hash, reference.token_hash);
            assert!(actual.stats.dflash_accepted_tokens.unwrap() > 0);
        }
        // Stochastic presets: DFlash2 must run, accept drafts, finish and be seed-reproducible.
        let mut accepted = 0;
        for (sampler, label) in [
            (Sampler::new(0.2, 80, 1., 1.05, 0.)?, "desktop"),
            (Sampler::new(0.6, 20, 0.95, 1., 0.)?, "qwen"),
        ] {
            for mode in 1..=2 {
                let first = run(sampler, Some(&weights), mode, 11)?;
                let again = run(sampler, Some(&weights), mode, 11)?;
                assert!(!first.raw.is_empty());
                assert_eq!(first.raw, again.raw, "seeded {label} mode {mode}");
                let (proposed, hits) = (
                    first.stats.dflash_proposed_tokens.unwrap(),
                    first.stats.dflash_accepted_tokens.unwrap(),
                );
                assert!(hits <= proposed);
                accepted += hits;
                eprintln!(
                    "stochastic DFlash2 {label} mode {mode}: {} tokens, {hits}/{proposed} accepted",
                    first.stats.generated_tokens
                );
            }
        }
        assert!(accepted > 0, "sampled DFlash2 must accept drafts");
        Ok(())
    }

    /// Q27D-10 (D6 + D3): synchronized block timings and per-position draft confidence,
    /// seven proposals per block, greedy target. Diagnostic only.
    #[test]
    #[ignore = "requires MLXL3_DFLASH_TEST_MODEL/DRAFT and Apple GPU; writes MLXL3_DFLASH_PROFILE_REPORT"]
    fn dflash_block_profile_and_confidence() -> Result<()> {
        use mlxl3_native::array::synchronize;
        let output = std::path::PathBuf::from(std::env::var("MLXL3_DFLASH_PROFILE_REPORT")?);
        anyhow::ensure!(!output.exists(), "preserve previous profile reports");
        let (mut model, tokenizer, weights) = q27_dflash_fixture()?;
        let cancelled = AtomicBool::new(false);
        let prompts = [
            "Explique en quelques paragraphes pourquoi le ciel est bleu.",
            "Write a Rust function that parses an ISO 8601 date like 2026-11-16.",
            "Write a short story about a lighthouse keeper who finds a message in a bottle.",
            "Explain step by step how to solve the quadratic equation 3x^2 - 5x - 2 = 0.",
            "Résume les causes de la Révolution française en cinq points détaillés.",
            "Write a Python class implementing an LRU cache with tests.",
            "Describe how a compiler turns source code into machine code.",
            "Donne une recette de crêpes avec les quantités et les étapes.",
        ];
        let budget =
            std::env::var("MLXL3_DFLASH_PROFILE_TOKENS").map_or(Ok(128usize), |v| v.parse())?;
        let vocab = 248_320usize;
        let mut blocks = Vec::new();
        let mut outputs: Vec<Vec<u32>> = Vec::new();
        for (index, prompt) in prompts.iter().enumerate() {
            let tokens = tokenizer.encode(
                &tokenizer.render_values(&json!([{"role":"user", "content":prompt}]), None)?,
            )?;
            model.reset();
            let (logits, session, _) = prefill_round(
                &mut model,
                &tokens,
                &[],
                Some(&weights),
                &mut None,
                "profile",
                false,
                &cancelled,
            )?;
            let mut session = session.context("missing DFlash session")?;
            let NativeChatModel::Qwen(target) = &mut model else {
                bail!("requires a Qwen target")
            };
            let mut anchor = logits.chat_greedy_ids()?[0];
            let mut generated = vec![anchor];
            let mut produced = 1usize;
            while produced + 1 < budget {
                let k = 7usize.min(budget - produced - 1);
                synchronize()?;
                let t0 = Instant::now();
                let input = target.dflash_input(anchor, weights.family.mask())?;
                let draft = weights.forward_hidden(&input, &session.cache, target.offset())?;
                draft.hidden.eval()?;
                draft.selector.eval()?;
                synchronize()?;
                let draft_ms = t0.elapsed().as_secs_f64() * 1e3;
                let t1 = Instant::now();
                let draft_logits = target.dflash_logits(&draft.hidden, k)?;
                let proposals = weights.select_greedy(&draft_logits, &draft.selector, anchor)?;
                let head_ms = t1.elapsed().as_secs_f64() * 1e3;
                // Diagnostic readback after the timed part.
                let rows = draft_logits.log_probs()?.to_f32()?;
                anyhow::ensure!(
                    rows.len() == k * vocab && proposals.len() == k,
                    "short draft rows"
                );
                let mut confidence = Vec::with_capacity(k);
                // Q27D-32: draft top-4 ids per position for offline tree simulation.
                let top_ids = rows
                    .chunks(vocab)
                    .map(|row| {
                        let mut order: Vec<u32> = (0..vocab as u32).collect();
                        order.select_nth_unstable_by(3, |&a, &b| {
                            row[b as usize].total_cmp(&row[a as usize])
                        });
                        let mut top = order[..4].to_vec();
                        top.sort_by(|&a, &b| row[b as usize].total_cmp(&row[a as usize]));
                        top
                    })
                    .collect::<Vec<_>>();
                for (row, &token) in rows.chunks(vocab).zip(&proposals) {
                    let (best, top) =
                        row.iter()
                            .enumerate()
                            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                                if v > acc.1 { (i, v) } else { acc }
                            });
                    let mut second = f32::NEG_INFINITY;
                    for (i, &v) in row.iter().enumerate() {
                        if i != best && v > second {
                            second = v;
                        }
                    }
                    confidence.push(json!({"chosen": row[token as usize], "top": top,
                        "second": second, "argmax": best as u32 == token}));
                }
                let verify = std::iter::once(anchor)
                    .chain(proposals.iter().copied())
                    .collect::<Vec<_>>();
                let position = target.offset();
                synchronize()?;
                let t2 = Instant::now();
                let (target_logits, captured) =
                    target.verify_tokens_exact_with_dflash_capture(&verify)?;
                let target_tokens = target_logits.chat_greedy_ids()?;
                let verify_ms = t2.elapsed().as_secs_f64() * 1e3;
                let acceptance =
                    mlxl3_native::speculative::greedy_accept(&proposals, &target_tokens)
                        .context("invalid DFlash target verification")?;
                let retained = acceptance.accepted_draft_tokens + 1;
                let t3 = Instant::now();
                target.commit_dflash_verification(retained, verify.len())?;
                session.cache.append_captured(
                    &weights,
                    &captured.slice(0, 0, i32::try_from(retained)?)?,
                    position,
                )?;
                session.cache.eval()?;
                synchronize()?;
                let commit_ms = t3.elapsed().as_secs_f64() * 1e3;
                blocks.push(json!({"prompt": index, "offset": position, "proposals": k,
                    "accepted": acceptance.accepted_draft_tokens, "draft_ms": draft_ms,
                    "head_select_ms": head_ms, "verify_ms": verify_ms, "commit_ms": commit_ms,
                    "confidence": confidence, "anchor": anchor, "anchor_index": generated.len() - 1,
                    "chain": proposals, "top_ids": top_ids}));
                generated.extend(
                    proposals[..acceptance.accepted_draft_tokens]
                        .iter()
                        .copied(),
                );
                generated.push(acceptance.target_token);
                anchor = acceptance.target_token;
                produced += retained;
            }
            eprintln!(
                "prompt {index}: {produced} tokens, {} blocks so far",
                blocks.len()
            );
            outputs.push(generated);
        }
        let report = json!({"status": "complete", "budget": budget, "blocks": blocks,
            "outputs": outputs});
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
        Ok(())
    }

    /// Q27D-36: selector lattice of every DFlash2 block (candidates, unary, edges) under the
    /// production greedy policy, for offline tree simulation. Outputs are unchanged.
    #[test]
    #[ignore = "requires MLXL3_DFLASH_TEST_MODEL/DRAFT and Apple GPU; writes MLXL3_DFLASH_PROFILE_REPORT"]
    fn dflash_selector_lattice_dump() -> Result<()> {
        let output = std::path::PathBuf::from(std::env::var("MLXL3_DFLASH_PROFILE_REPORT")?);
        anyhow::ensure!(!output.exists(), "preserve previous profile reports");
        let (mut model, tokenizer, weights) = q27_dflash_fixture()?;
        let cancelled = AtomicBool::new(false);
        let prompts = [
            "Explique en quelques paragraphes pourquoi le ciel est bleu.",
            "Write a Rust function that parses an ISO 8601 date like 2026-11-16.",
            "Write a short story about a lighthouse keeper who finds a message in a bottle.",
            "Explain step by step how to solve the quadratic equation 3x^2 - 5x - 2 = 0.",
            "Résume les causes de la Révolution française en cinq points détaillés.",
            "Write a Python class implementing an LRU cache with tests.",
            "Describe how a compiler turns source code into machine code.",
            "Donne une recette de crêpes avec les quantités et les étapes.",
        ];
        let budget =
            std::env::var("MLXL3_DFLASH_PROFILE_TOKENS").map_or(Ok(256usize), |v| v.parse())?;
        let mask = weights.family.mask();
        let mut blocks = Vec::new();
        let mut outputs: Vec<Vec<u32>> = Vec::new();
        for (index, prompt) in prompts.iter().enumerate() {
            let tokens = tokenizer.encode(
                &tokenizer.render_values(&json!([{"role":"user", "content":prompt}]), None)?,
            )?;
            model.reset();
            let (logits, session, _) = prefill_round(
                &mut model,
                &tokens,
                &[],
                Some(&weights),
                &mut None,
                "lattice",
                false,
                &cancelled,
            )?;
            let mut session = session.context("missing DFlash session")?;
            let NativeChatModel::Qwen(target) = &mut model else {
                bail!("requires a Qwen target")
            };
            let mut anchor = logits.chat_greedy_ids()?[0];
            let mut generated = vec![anchor];
            let mut produced = 1usize;
            while produced + 1 < budget {
                let k = 7usize.min(budget - produced - 1);
                let input = target.dflash_input(anchor, mask)?;
                let draft = weights.forward_hidden(&input, &session.cache, target.offset())?;
                let draft_logits = target.dflash_logits(&draft.hidden, k)?;
                let proposals = weights.select_greedy(&draft_logits, &draft.selector, anchor)?;
                let (candidates, unary, edges) =
                    weights.select_lattice(&draft_logits, &draft.selector, anchor)?;
                anyhow::ensure!(
                    candidates.len() == k * 16
                        && unary.len() == k * 16
                        && edges.len() == k * 256
                        && unary.iter().chain(&edges).all(|v| v.is_finite()),
                    "invalid selector lattice"
                );
                let verify = std::iter::once(anchor)
                    .chain(proposals.iter().copied())
                    .collect::<Vec<_>>();
                let position = target.offset();
                let (target_logits, captured) =
                    target.verify_tokens_exact_with_dflash_capture(&verify)?;
                let target_tokens = target_logits.chat_greedy_ids()?;
                let acceptance =
                    mlxl3_native::speculative::greedy_accept(&proposals, &target_tokens)
                        .context("invalid DFlash target verification")?;
                let retained = acceptance.accepted_draft_tokens + 1;
                target.commit_dflash_verification(retained, verify.len())?;
                session.cache.append_captured(
                    &weights,
                    &captured.slice(0, 0, i32::try_from(retained)?)?,
                    position,
                )?;
                session.cache.eval()?;
                blocks.push(json!({"prompt": index, "anchor": anchor,
                    "anchor_index": generated.len() - 1, "proposals": k,
                    "accepted": acceptance.accepted_draft_tokens, "chain": proposals,
                    "candidates": candidates, "unary": unary, "edges": edges}));
                generated.extend(
                    proposals[..acceptance.accepted_draft_tokens]
                        .iter()
                        .copied(),
                );
                generated.push(acceptance.target_token);
                anchor = acceptance.target_token;
                produced += retained;
            }
            eprintln!(
                "prompt {index}: {produced} tokens, {} blocks so far",
                blocks.len()
            );
            outputs.push(generated);
        }
        let report = json!({"status": "complete", "budget": budget, "edge_scale": 0.25,
            "blocks": blocks, "outputs": outputs});
        std::fs::write(&output, serde_json::to_vec(&report)?)?;
        Ok(())
    }

    /// Q27D-34: what a 16-row mask block (anchor + 15 masks) would add over the 8-row block.
    /// Generation follows the production policy (selector chain of at most 7, greedy target),
    /// so outputs are unchanged; each block also records the per-position argmax of the draft
    /// at 8 and at 16 rows for offline scoring against the recorded greedy output.
    #[test]
    #[ignore = "requires MLXL3_DFLASH_TEST_MODEL/DRAFT and Apple GPU; writes MLXL3_DFLASH_PROFILE_REPORT"]
    fn dflash_block16_depth_accuracy() -> Result<()> {
        let output = std::path::PathBuf::from(std::env::var("MLXL3_DFLASH_PROFILE_REPORT")?);
        anyhow::ensure!(!output.exists(), "preserve previous profile reports");
        let (mut model, tokenizer, weights) = q27_dflash_fixture()?;
        let cancelled = AtomicBool::new(false);
        let prompts = [
            "Explique en quelques paragraphes pourquoi le ciel est bleu.",
            "Write a Rust function that parses an ISO 8601 date like 2026-11-16.",
            "Write a short story about a lighthouse keeper who finds a message in a bottle.",
            "Explain step by step how to solve the quadratic equation 3x^2 - 5x - 2 = 0.",
            "Résume les causes de la Révolution française en cinq points détaillés.",
            "Write a Python class implementing an LRU cache with tests.",
            "Describe how a compiler turns source code into machine code.",
            "Donne une recette de crêpes avec les quantités et les étapes.",
        ];
        let budget =
            std::env::var("MLXL3_DFLASH_PROFILE_TOKENS").map_or(Ok(256usize), |v| v.parse())?;
        let vocab = 248_320usize;
        let mask = weights.family.mask();
        let mut blocks = Vec::new();
        let mut outputs: Vec<Vec<u32>> = Vec::new();
        for (index, prompt) in prompts.iter().enumerate() {
            let tokens = tokenizer.encode(
                &tokenizer.render_values(&json!([{"role":"user", "content":prompt}]), None)?,
            )?;
            model.reset();
            let (logits, session, _) = prefill_round(
                &mut model,
                &tokens,
                &[],
                Some(&weights),
                &mut None,
                "block16",
                false,
                &cancelled,
            )?;
            let mut session = session.context("missing DFlash session")?;
            let NativeChatModel::Qwen(target) = &mut model else {
                bail!("requires a Qwen target")
            };
            let mut anchor = logits.chat_greedy_ids()?[0];
            let mut generated = vec![anchor];
            let mut produced = 1usize;
            while produced + 1 < budget {
                let k = 7usize.min(budget - produced - 1);
                let input = target.dflash_input(anchor, mask)?;
                let draft = weights.forward_hidden(&input, &session.cache, target.offset())?;
                let draft_logits = target.dflash_logits(&draft.hidden, k)?;
                let proposals = weights.select_greedy(&draft_logits, &draft.selector, anchor)?;
                let marginal8 = draft_logits.chat_greedy_ids()?;
                let input16 = target.dflash_input_rows(anchor, mask, 16)?;
                let draft16 = weights.forward_hidden(&input16, &session.cache, target.offset())?;
                let rows16 = target
                    .dflash_logits(&draft16.hidden, 15)?
                    .log_probs()?
                    .to_f32()?;
                anyhow::ensure!(
                    rows16.len() == 15 * vocab && marginal8.len() == k && proposals.len() == k,
                    "short draft rows"
                );
                let (marginal16, confidence16): (Vec<u32>, Vec<f32>) = rows16
                    .chunks(vocab)
                    .map(|row| {
                        let (best, top) = row.iter().enumerate().fold(
                            (0usize, f32::NEG_INFINITY),
                            |acc, (i, &v)| {
                                if v > acc.1 { (i, v) } else { acc }
                            },
                        );
                        (best as u32, top)
                    })
                    .unzip();
                anyhow::ensure!(
                    confidence16.iter().all(|v| v.is_finite()),
                    "non-finite draft row"
                );
                let verify = std::iter::once(anchor)
                    .chain(proposals.iter().copied())
                    .collect::<Vec<_>>();
                let position = target.offset();
                let (target_logits, captured) =
                    target.verify_tokens_exact_with_dflash_capture(&verify)?;
                let target_tokens = target_logits.chat_greedy_ids()?;
                let acceptance =
                    mlxl3_native::speculative::greedy_accept(&proposals, &target_tokens)
                        .context("invalid DFlash target verification")?;
                let retained = acceptance.accepted_draft_tokens + 1;
                target.commit_dflash_verification(retained, verify.len())?;
                session.cache.append_captured(
                    &weights,
                    &captured.slice(0, 0, i32::try_from(retained)?)?,
                    position,
                )?;
                session.cache.eval()?;
                blocks.push(json!({"prompt": index, "anchor": anchor,
                    "anchor_index": generated.len() - 1, "proposals": k,
                    "accepted": acceptance.accepted_draft_tokens, "chain": proposals,
                    "marginal8": marginal8, "marginal16": marginal16,
                    "confidence16": confidence16}));
                generated.extend(
                    proposals[..acceptance.accepted_draft_tokens]
                        .iter()
                        .copied(),
                );
                generated.push(acceptance.target_token);
                anchor = acceptance.target_token;
                produced += retained;
            }
            eprintln!(
                "prompt {index}: {produced} tokens, {} blocks so far",
                blocks.len()
            );
            outputs.push(generated);
        }
        let report = json!({"status": "complete", "budget": budget, "blocks": blocks,
            "outputs": outputs});
        std::fs::write(&output, serde_json::to_vec_pretty(&report)?)?;
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen and DFlash checkpoints and Apple GPU"]
    fn bridge_draft_suffix_keeps_full_prefill_and_replay() -> Result<()> {
        use mlxl3_native::{array::memory_stats, dflash::PREFILL_CHUNK, tokenizer::ChatTokenizer};
        let path = std::path::Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw");
        let mut model = NativeChatModel::load(path)?;
        let tokenizer = ChatTokenizer::load(path)?;
        let package = mlxl3_native::dflash::inspect("models/Qwen3.6-35B-A3B-DFlash2")?;
        let weights = mlxl3_native::dflash::DFlashWeights::load(&package)?;
        let cancelled = AtomicBool::new(false);
        let suffix = tokenizer.encode(&tokenizer.render_values(
            &json!([{"role":"user", "content":"Explain how a KV cache reduces latency."}]),
            None,
        )?)?;
        let mut run = |tokens: &[u32], optimized: bool| -> Result<_> {
            model.reset();
            let _ = memory_stats(true)?;
            let started = Instant::now();
            let (logits, mut session) = if optimized {
                let (logits, session, cached) = prefill_round(
                    &mut model,
                    tokens,
                    &[],
                    Some(&weights),
                    &mut None,
                    "long",
                    false,
                    &cancelled,
                )?;
                assert_eq!(cached, 0);
                (logits, session.unwrap())
            } else {
                // Original prefill: every target chunk is captured/projected,
                // including the draft KV subsequently thrown away by its window.
                let mut session = DFlashChat::new();
                let mut logits = None;
                for chunk in tokens.chunks(PREFILL_CHUNK) {
                    let NativeChatModel::Qwen(target) = &mut model else {
                        unreachable!()
                    };
                    logits = Some(session.prefill(target, &weights, chunk)?);
                    model.eval_state()?;
                }
                (logits.unwrap(), session)
            };
            logits.eval()?;
            let seconds = started.elapsed().as_secs_f64();
            let memory = memory_stats(false)?;
            let first_logits = logits.to_f16_bits()?;
            let mut generated = vec![logits.chat_greedy_ids()?[0]];
            let NativeChatModel::Qwen(target) = &mut model else {
                unreachable!()
            };
            assert_eq!(
                usize::try_from(target.offset())?,
                tokens.len(),
                "target must read the whole prompt"
            );
            let draft = weights.forward_hidden(
                &target.dflash_input(generated[0], 248_077)?,
                &session.cache,
                target.offset(),
            )?;
            let features = (draft.hidden.to_bytes()?, draft.selector.to_bytes()?);
            while generated.len() < 32 {
                let next = session.advance(
                    target,
                    &weights,
                    *generated.last().unwrap(),
                    16_384,
                    32 - generated.len(),
                )?;
                generated.push(next);
            }
            Ok((
                first_logits,
                features,
                generated,
                session.proposed,
                session.accepted,
                seconds,
                memory,
            ))
        };
        for length in [2047, 2048, 2049, 4096, 4097, 8192] {
            let tokens: Vec<u32> = (0..length - suffix.len())
                .map(|i| (i % 97 + 1) as u32)
                .chain(suffix.iter().copied())
                .collect();
            let baseline = run(&tokens, false)?;
            let candidate = run(&tokens, true)?;
            assert_eq!(candidate.0, baseline.0, "prefill logits at {length}");
            assert_eq!(candidate.1, baseline.1, "draft features at {length}");
            assert_eq!(candidate.2, baseline.2, "generation at {length}");
            assert_eq!(
                (candidate.3, candidate.4),
                (baseline.3, baseline.4),
                "acceptance at {length}"
            );
            eprintln!(
                "suffix bridge parity {length}: 32 IDs, logits, draft features and acceptance exact"
            );
            if length == 4096 || length == 8192 {
                let mut times = [Vec::new(), Vec::new()];
                for optimized in [false, true, true, false] {
                    let sample = run(&tokens, optimized)?;
                    assert_eq!(sample.2, baseline.2);
                    eprintln!(
                        "suffix prefill {length}, optimized={optimized}: {:.3} s, memory {}",
                        sample.5,
                        serde_json::to_string(&sample.6)?
                    );
                    times[usize::from(optimized)].push(sample.5);
                }
                eprintln!(
                    "suffix prefill {length} median: {:.3} -> {:.3} s",
                    (times[0][0] + times[0][1]) / 2.,
                    (times[1][0] + times[1][1]) / 2.
                );
            }
        }
        // A partial final chunk must not leave the saved checkpoint with an
        // incomplete draft window. Extend that checkpoint after live decode.
        let tokens: Vec<u32> = (0..4097).map(|i| (i % 97 + 1) as u32).collect();
        let mut saved = None;
        prefill_round(
            &mut model,
            &tokens,
            &[],
            Some(&weights),
            &mut saved,
            "replay",
            true,
            &cancelled,
        )?;
        assert!(
            saved.is_some(),
            "fixture must exercise an actual saved prefix"
        );
        model.forward(37)?.eval()?;
        let extended: Vec<_> = tokens
            .iter()
            .copied()
            .chain(suffix.iter().copied())
            .collect();
        let (warm, mut session, cached) = prefill_round(
            &mut model,
            &extended,
            &[],
            Some(&weights),
            &mut saved,
            "replay",
            true,
            &cancelled,
        )?;
        assert_eq!(cached, 4096);
        let mut actual = vec![warm.chat_greedy_ids()?[0]];
        let NativeChatModel::Qwen(target) = &mut model else {
            unreachable!()
        };
        while actual.len() < 32 {
            actual.push(session.as_mut().unwrap().advance(
                target,
                &weights,
                *actual.last().unwrap(),
                16_384,
                32 - actual.len(),
            )?);
        }
        // Compare to the independent full-capture baseline above, not a second
        // call to the new suffix-only policy.
        let NativeChatModel::Qwen(target) = &mut model else {
            unreachable!()
        };
        target.reset();
        let mut reference = DFlashChat::new();
        let mut logits = None;
        for chunk in extended.chunks(PREFILL_CHUNK) {
            logits = Some(reference.prefill(target, &weights, chunk)?);
        }
        let mut expected = vec![logits.unwrap().chat_greedy_ids()?[0]];
        while expected.len() < 32 {
            expected.push(reference.advance(
                target,
                &weights,
                *expected.last().unwrap(),
                16_384,
                32 - expected.len(),
            )?);
        }
        assert_eq!(actual, expected, "long prefix replay changed tokens");
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen and DFlash checkpoints and Apple GPU"]
    fn long_prompt_dflash_blocks_match_target() -> Result<()> {
        use mlxl3_native::tokenizer::ChatTokenizer;
        let path = std::path::Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw");
        let mut model = NativeChatModel::load(path)?;
        let tokenizer = ChatTokenizer::load(path)?;
        let prompt = std::fs::read_to_string("docs/audit-runtime-2026-09-04.md")?;
        let rendered =
            tokenizer.render_values(&json!([{"role":"user", "content":prompt}]), None)?;
        let tokens = tokenizer.encode(&rendered)?;
        assert_eq!(tokens.len(), 2825);
        let package = mlxl3_native::dflash::inspect("models/Qwen3.6-35B-A3B-DFlash2")?;
        let weights = mlxl3_native::dflash::DFlashWeights::load(&package)?;
        let cancelled = AtomicBool::new(false);
        let (mut logits, _, _) = prefill_round(
            &mut model,
            &tokens,
            &[],
            None,
            &mut None,
            "normal",
            false,
            &cancelled,
        )?;
        let normal_prefill = logits.to_f16_bits()?;
        let NativeChatModel::Qwen(target) = &mut model else {
            unreachable!()
        };
        let mut reference = target.snapshot()?;
        let mut expected = Vec::new();
        while expected.len() < 128 {
            let next = logits.chat_greedy_ids()?[0];
            expected.push(next);
            if expected.len() < 128 {
                logits = target.forward(next)?;
            }
        }
        let (logits, session, _) = prefill_round(
            &mut model,
            &tokens,
            &[],
            Some(&weights),
            &mut None,
            "dflash",
            false,
            &cancelled,
        )?;
        assert_eq!(logits.to_f16_bits()?, normal_prefill);
        let mut session = session.context("missing DFlash session")?;
        let mut generated = vec![logits.chat_greedy_ids()?[0]];
        assert_eq!(generated[0], expected[0]);
        let NativeChatModel::Qwen(target) = &mut model else {
            unreachable!()
        };
        let mut block = 0;
        while generated.len() < expected.len() {
            let anchor = *generated.last().unwrap();
            let before = target.offset();
            let accepted_before = session.accepted;
            let next = session.advance(
                target,
                &weights,
                anchor,
                8192,
                expected.len() - generated.len(),
            )?;
            if target.offset() > before {
                block += 1;
                let accepted = session.accepted - accepted_before;
                let mut committed = vec![anchor];
                if accepted > 0 {
                    committed.push(next);
                    committed.extend(session.pending.iter().take(accepted - 1).copied());
                }
                assert_eq!(target.offset() - before, committed.len() as i32);
                let speculative = target.snapshot()?;
                target.restore(reference)?;
                for token in committed {
                    target.forward(token)?;
                }
                reference = target.snapshot()?;
                target.restore(speculative.clone())?;
                let actual = target.forward(42)?.to_f16_bits()?;
                target.restore(reference.clone())?;
                let expected_logits = target.forward(42)?.to_f16_bits()?;
                target.restore(speculative)?;
                anyhow::ensure!(
                    actual == expected_logits,
                    "DFlash target state differs after block {block}, offset {}",
                    target.offset()
                );
            }
            anyhow::ensure!(
                next == expected[generated.len()],
                "DFlash token differs at index {}, block {block}",
                generated.len()
            );
            generated.push(next);
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen and DFlash checkpoints and Apple GPU"]
    fn dflash_output_budget_matches_target_and_skips_unused_work() -> Result<()> {
        use mlxl3_native::{qwen35::Qwen35Moe, tokenizer::ChatTokenizer};
        let path = std::path::Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw");
        let mut model = Qwen35Moe::load(path)?;
        let tokenizer = ChatTokenizer::load(path)?;
        let package = mlxl3_native::dflash::inspect("models/Qwen3.6-35B-A3B-DFlash2")?;
        let weights = mlxl3_native::dflash::DFlashWeights::load(&package)?;
        for prompt in [
            "Explain lossless speculative decoding in one paragraph.",
            "Write a Rust function to add two integers.",
        ] {
            let rendered =
                tokenizer.render_values(&json!([{"role":"user", "content":prompt}]), None)?;
            let tokens = tokenizer.encode(&rendered)?;
            model.reset();
            let mut logits = model.forward_tokens(&tokens)?;
            // Independent, non-speculative target sequence. Deliberately don't
            // stop at EOS: stopping at every prefix also exercises early stops.
            let mut expected = Vec::new();
            for _ in 0..16 {
                let token = logits.chat_greedy_ids()?[0];
                expected.push(token);
                logits = model.forward(token)?;
            }
            let mut run = |budget: usize, bounded: bool, tight_context: bool| -> Result<f64> {
                model.reset();
                let mut session = DFlashChat::new();
                let first = session
                    .prefill(&mut model, &weights, &tokens)?
                    .chat_greedy_ids()?[0];
                let mut actual = vec![first];
                let context = if tight_context {
                    tokens.len() + budget
                } else {
                    4096
                };
                let started = Instant::now();
                while actual.len() < budget {
                    let remaining = if bounded {
                        budget - actual.len()
                    } else {
                        usize::MAX
                    };
                    let token = session.advance(
                        &mut model,
                        &weights,
                        *actual.last().unwrap(),
                        context,
                        remaining,
                    )?;
                    actual.push(token);
                }
                let elapsed = started.elapsed().as_secs_f64();
                assert_eq!(
                    actual,
                    expected[..budget],
                    "budget {budget}, bounded {bounded}"
                );
                if bounded {
                    assert!(
                        session.pending.is_empty(),
                        "no computed tokens may be left past the output limit"
                    );
                    assert_eq!(usize::try_from(model.offset())?, tokens.len() + budget - 1);
                }
                let position = model.offset();
                let pending = session.pending.clone();
                assert!(
                    session
                        .advance(&mut model, &weights, first, context, 0)
                        .is_err()
                );
                assert_eq!(
                    model.offset(),
                    position,
                    "invalid budget must not mutate state"
                );
                assert_eq!(session.pending, pending);
                Ok(elapsed)
            };
            // All prefixes, including abandoned pending from the old algorithm,
            // start the next request from a fresh prefill in the same model.
            for budget in 1..=16 {
                run(budget, false, false)?;
                run(budget, true, false)?;
                run(budget, true, true)?;
            }
            for budget in [2, 4, 6, 16] {
                let mut times = [Vec::new(), Vec::new()];
                for bounded in [false, true, true, false, false, true, true, false] {
                    times[usize::from(bounded)].push(run(budget, bounded, false)? * 1000.);
                }
                for samples in &mut times {
                    samples.sort_by(f64::total_cmp);
                }
                eprintln!(
                    "output budget {budget} ({prompt}): unbounded {:.3} -> bounded {:.3} ms",
                    (times[0][1] + times[0][2]) / 2.,
                    (times[1][1] + times[1][2]) / 2.
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires local Qwen and DFlash checkpoints and Apple GPU"]
    fn bridge_prefix_cache_replays_cold_logits_exactly() -> Result<()> {
        let mut model =
            NativeChatModel::load(std::path::Path::new("models/Qwen3.6-35B-A3B-EXL3-2.49bpw"))?;
        let package = mlxl3_native::dflash::inspect("models/Qwen3.6-35B-A3B-DFlash2")?;
        let weights = mlxl3_native::dflash::DFlashWeights::load(&package)?;
        let cancelled = AtomicBool::new(false);
        let mut cache = None;
        for draft in [None, Some(&weights)] {
            let block = if draft.is_some() { 128 } else { 256 };
            let original: Vec<u32> = (0..block + 17)
                .map(|index| (index % 97 + 1) as u32)
                .collect();
            let (cold, _, count) = prefill_round(
                &mut model,
                &original,
                &[],
                draft,
                &mut cache,
                "a",
                true,
                &cancelled,
            )?;
            assert_eq!(count, 0, "mode switch must miss");
            let cold = cold.to_f16_bits()?;
            model.forward(51)?.eval()?; // Live decode may be ahead of the saved prefix.
            let (replay, _, count) = prefill_round(
                &mut model,
                &original,
                &[],
                draft,
                &mut cache,
                "a",
                true,
                &cancelled,
            )?;
            assert_eq!(count, block);
            assert_eq!(replay.to_f16_bits()?, cold);
            for suffix in [0, 1, 23, 24, 25] {
                let extended: Vec<_> = original[..block]
                    .iter()
                    .copied()
                    .chain(std::iter::repeat_n(42, suffix))
                    .collect();
                let (expected, _, _) = prefill_round(
                    &mut model,
                    &extended,
                    &[],
                    draft,
                    &mut None,
                    "a",
                    false,
                    &cancelled,
                )?;
                let expected = expected.to_f16_bits()?;
                let (actual, _, count) = prefill_round(
                    &mut model,
                    &extended,
                    &[],
                    draft,
                    &mut cache,
                    "a",
                    true,
                    &cancelled,
                )?;
                assert_eq!(count, block);
                assert_eq!(
                    actual.to_f16_bits()?,
                    expected,
                    "prefill suffix {suffix}, DFlash {}",
                    draft.is_some()
                );
            }
            let mut edited = original.clone();
            edited[0] += 1;
            let (_, _, count) = prefill_round(
                &mut model,
                &edited,
                &[],
                draft,
                &mut cache,
                "a",
                true,
                &cancelled,
            )?;
            assert_eq!(count, 0, "edited history must miss");
            let (_, _, count) = prefill_round(
                &mut model,
                &edited,
                &[],
                draft,
                &mut cache,
                "b",
                true,
                &cancelled,
            )?;
            assert_eq!(count, 0, "conversation switch must miss");
            cancelled.store(true, Ordering::Relaxed);
            assert!(
                prefill_round(
                    &mut model,
                    &edited,
                    &[],
                    draft,
                    &mut cache,
                    "b",
                    true,
                    &cancelled
                )
                .is_err()
            );
            cancelled.store(false, Ordering::Relaxed);
            let (_, _, count) = prefill_round(
                &mut model,
                &edited,
                &[],
                draft,
                &mut cache,
                "b",
                false,
                &cancelled,
            )?;
            assert_eq!(count, 0);
            assert!(cache.is_none(), "cache OFF must release the saved state");
        }
        Ok(())
    }
}

#[cfg(all(
    test,
    feature = "chat",
    any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
))]
mod device_greedy_stats_tests {
    use super::*;
    #[test]
    fn scalar_selector_stats_are_optional_and_accumulate_actual_work() {
        let mut stats = NativeStats::default();
        assert!(
            serde_json::to_value(&stats)
                .unwrap()
                .get("native_gpu_greedy_selections")
                .is_none()
        );
        stats.accumulate(&NativeStats {
            native_gpu_greedy_selections: Some(2),
            native_gpu_greedy_heads: Some(1),
            ..Default::default()
        });
        stats.accumulate(&NativeStats::default());
        stats.accumulate(&NativeStats {
            native_gpu_greedy_selections: Some(3),
            native_gpu_greedy_heads: Some(2),
            ..Default::default()
        });
        assert_eq!(stats.native_gpu_greedy_selections, Some(5));
        assert_eq!(stats.native_gpu_greedy_heads, Some(3));
        let encoded = serde_json::to_value(&stats).unwrap();
        assert_eq!(encoded["native_gpu_greedy_selections"], 5);
        assert_eq!(encoded["native_gpu_greedy_heads"], 3);
    }
}
