use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand, ValueEnum};
use mlxl3_native::{
    checkpoint,
    codec::{self, Codebook},
    registry,
    streaming::ThinkingSplitter,
};
use serde::Deserialize;
#[cfg(all(feature = "mlx", feature = "chat"))]
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, Read, Write},
    path::PathBuf,
};
#[cfg(all(feature = "mlx", feature = "chat"))]
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
#[command(version, about = "Native MLXL3 Rust/Metal runtime")]
struct Cli {
    #[arg(long, global = true)]
    registry: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(alias = "ls")]
    List {
        #[arg(long)]
        json: bool,
    },
    Inspect {
        path: PathBuf,
    },
    #[command(alias = "add")]
    Register {
        name: String,
        path: PathBuf,
        #[arg(long)]
        force: bool,
    },
    #[command(alias = "rm")]
    Remove {
        name: String,
        #[arg(long)]
        expected_path: Option<PathBuf>,
    },
    /// JSON-line codec oracle used for differential tests (not a chat bridge).
    Codec,
    /// Feed a JSON array of text chunks on stdin; return reasoning/answer fragments.
    Split {
        #[arg(long, default_value = "")]
        prompt: String,
    },
    /// Native LFM2, Qwen, Gemma 4 and Ling 3 chat through MLX.
    Run {
        model: String,
        #[arg(short, long)]
        prompt: Option<String>,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
    },
    /// Resident JSON-lines bridge consumed by MLXL3 Desktop.
    Bridge {
        model: String,
        #[arg(long, default_value_t = 0)]
        context_length: i32,
    },
    /// Browse EXL3 repositories on Hugging Face.
    Hub {
        #[arg(value_enum)]
        action: HubAction,
        query: String,
        #[arg(long)]
        revision: Option<String>,
        #[arg(long, default_value = "")]
        folder: String,
        #[arg(long, default_value_t = 60)]
        limit: usize,
    },
    /// Configure and inspect Model Context Protocol servers.
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },
    /// Emit logits for an imposed token sequence (native parity check).
    #[cfg(all(feature = "mlx", feature = "chat"))]
    Forward {
        model: PathBuf,
        #[arg(long, value_delimiter = ',', required = true)]
        tokens: Vec<u32>,
        #[arg(long)]
        states: bool,
    },
}

#[derive(Clone, Copy, ValueEnum)]
enum HubAction {
    Search,
    Details,
    Download,
    Pending,
    Resume,
    Discard,
    Auth,
}

#[derive(Subcommand)]
enum McpAction {
    #[command(alias = "ls")]
    List {
        #[arg(long)]
        json: bool,
    },
    Check {
        #[arg(long)]
        json: bool,
    },
    Config,
    Add {
        name: String,
        command: String,
        args: Vec<String>,
    },
    #[command(alias = "rm")]
    Remove {
        name: String,
    },
}

#[derive(Deserialize)]
struct CodecRequest {
    op: String,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    path: String,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default)]
    mode: u32,
    #[serde(default)]
    data: Vec<u16>,
    #[serde(default)]
    #[cfg(any(target_os = "macos", feature = "mlx"))]
    x: Vec<u16>,
    #[serde(default)]
    #[cfg(any(target_os = "macos", feature = "mlx"))]
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
    #[cfg(feature = "mlx")]
    beta: Vec<u16>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    state: Vec<f32>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    key_heads: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    value_heads: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    value_dims: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
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
    #[cfg(feature = "mlx")]
    selected: Vec<u32>,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    projections_per_route: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
    projection_stride_tiles: usize,
    #[serde(default)]
    #[cfg(feature = "mlx")]
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
fn default_k() -> usize {
    4
}

fn codec_loop() -> Result<()> {
    #[cfg(target_os = "macos")]
    let mut gpu = None;
    let stdout = io::stdout();
    let mut output = stdout.lock();
    for line in io::stdin().lock().lines() {
        let result = (|| -> Result<Value> {
            let request: CodecRequest =
                serde_json::from_str(&line?).context("invalid codec request")?;
            let cb = Codebook::try_from(request.mode)?;
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
                    let experts = i32::try_from(if request.cols > 0 {
                        request.cols
                    } else {
                        request.data.len()
                    })?;
                    anyhow::ensure!(
                        request.data.len().is_multiple_of(usize::try_from(experts)?),
                        "invalid router rows"
                    );
                    let rows = i32::try_from(request.data.len() / usize::try_from(experts)?)?;
                    let (indices, scores) = router::topk(
                        &Array::from_f16_bits(&request.data, &[rows, experts])?,
                        request.k,
                        request.normalize,
                    )?;
                    Ok(json!({
                        "indices": indices.to_u32()?,
                        "scores": scores.to_f16_bits()?,
                    }))
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
                "mlx-qwen-gdn" => {
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
                    let output =
                        block.forward(&Array::from_f16_bits(&request.x, &[1, 1, hidden])?)?;
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
                #[cfg(target_os = "macos")]
                "metal-pack" | "metal-decode" | "metal-qmv" => {
                    if gpu.is_none() {
                        gpu = Some(mlxl3_native::gpu::Metal::new()?);
                    }
                    let gpu = gpu.as_ref().context("Metal is unavailable")?;
                    match request.op.as_str() {
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
    if let Err(error) = run(Cli::parse()) {
        eprintln!("Error: {error:#}");
        std::process::ExitCode::from(2)
    } else {
        std::process::ExitCode::SUCCESS
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
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
        #[cfg(all(feature = "mlx", feature = "chat"))]
        Command::Forward {
            model,
            tokens,
            states,
        } => {
            let mut model = NativeChatModel::load(&model)?;
            let mut out = io::stdout().lock();
            for token in tokens {
                let (logits, arrays) = match (&mut model, states) {
                    (NativeChatModel::Gemma4(model), true) => model.forward_trace(token)?,
                    (NativeChatModel::Lfm2(model), true) => {
                        let logits = model.forward(&[token])?;
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
        } => {
            #[cfg(all(feature = "mlx", feature = "chat"))]
            {
                let candidate = registry::expand_home(&PathBuf::from(&model))?;
                let model_path = if candidate.is_dir() {
                    candidate
                } else {
                    let path = cli
                        .registry
                        .map(Ok)
                        .unwrap_or_else(registry::default_path)?;
                    PathBuf::from(
                        &registry::load(&path)?
                            .get(&model)
                            .context("unknown registered model")?
                            .path,
                    )
                };
                native_chat(&model_path, prompt, max_tokens)?;
            }
            #[cfg(not(all(feature = "mlx", feature = "chat")))]
            {
                let _ = (prompt, max_tokens);
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
            #[cfg(not(all(feature = "mlx", feature = "chat")))]
            {
                let _ = (model, context_length);
                bail!("the Desktop bridge requires a build with --features mlx,chat");
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
            let progress = |completed, total| {
                println!(
                    "{}",
                    json!({"type":"progress", "completed":completed, "total":total})
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
                    let (name, entry) = mlxl3_native::hub::download(
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
                        mlxl3_native::hub::resume(&query, &registry_path, progress)?;
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

#[cfg(all(feature = "mlx", feature = "chat"))]
fn resolve_model_path(registry_path: Option<&std::path::Path>, model: &str) -> Result<PathBuf> {
    let candidate = registry::expand_home(&PathBuf::from(model))?;
    if candidate.is_dir() {
        return Ok(candidate);
    }
    let path = registry_path
        .map(PathBuf::from)
        .map(Ok)
        .unwrap_or_else(registry::default_path)?;
    Ok(PathBuf::from(
        &registry::load(&path)?
            .get(model)
            .context("unknown registered model")?
            .path,
    ))
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
    Lfm2(mlxl3_native::lfm2::Lfm2),
    Ling(mlxl3_native::ling::Ling),
    Qwen(mlxl3_native::qwen35::Qwen35Moe),
}

#[cfg(all(feature = "mlx", feature = "chat"))]
impl NativeChatModel {
    fn load(path: &std::path::Path) -> Result<Self> {
        let config: serde_json::Value =
            serde_json::from_reader(std::fs::File::open(path.join("config.json"))?)?;
        match config["model_type"].as_str() {
            Some("gemma4") => Ok(Self::Gemma4(mlxl3_native::gemma4::Gemma4::load(path)?)),
            Some("bailing_hybrid") => Ok(Self::Ling(mlxl3_native::ling::Ling::load(path)?)),
            Some("lfm2" | "lfm2_moe") => Ok(Self::Lfm2(mlxl3_native::lfm2::Lfm2::load(path)?)),
            Some("qwen3_5" | "qwen3_5_moe") => {
                Ok(Self::Qwen(mlxl3_native::qwen35::Qwen35Moe::load(path)?))
            }
            other => bail!("native chat does not support model type {other:?}"),
        }
    }

    fn forward(&mut self, token: u32) -> Result<mlxl3_native::array::Array> {
        match self {
            Self::Gemma4(model) => model.forward(token),
            Self::Lfm2(model) => model.forward(&[token]),
            Self::Ling(model) => model.forward(token),
            Self::Qwen(model) => model.forward(token),
        }
    }

    fn forward_many(&mut self, tokens: &[u32]) -> Result<mlxl3_native::array::Array> {
        anyhow::ensure!(!tokens.is_empty(), "empty token batch");
        if let (Self::Qwen(model), true) = (&mut *self, tokens.len() >= 24) {
            return model.forward_tokens(tokens);
        }
        let mut output = None;
        for &token in tokens {
            output = Some(self.forward(token)?);
        }
        output.context("empty token batch")
    }

    fn prefill_chunk_size(&self) -> usize {
        if matches!(self, Self::Qwen(_)) { 32 } else { 1 }
    }

    fn context_limit(&self) -> i32 {
        match self {
            Self::Gemma4(model) => model.context_limit(),
            Self::Lfm2(model) => model.context_limit(),
            Self::Ling(model) => model.context_limit(),
            Self::Qwen(model) => model.context_limit(),
        }
    }

    fn reset(&mut self) {
        match self {
            Self::Gemma4(model) => model.reset(),
            Self::Lfm2(model) => model.reset(),
            Self::Ling(model) => model.reset(),
            Self::Qwen(model) => model.reset(),
        }
    }

    fn eval_state(&self) -> Result<()> {
        match self {
            Self::Lfm2(model) => {
                for (_, state) in model.state_arrays() {
                    state.eval()?;
                }
            }
            Self::Gemma4(_) | Self::Ling(_) | Self::Qwen(_) => {}
        }
        Ok(())
    }

    fn name(&self) -> &'static str {
        match self {
            Self::Gemma4(_) => "Gemma 4",
            Self::Lfm2(_) => "LFM2",
            Self::Ling(_) => "Ling 3",
            Self::Qwen(_) => "Qwen3.5",
        }
    }
}

#[cfg(all(feature = "mlx", feature = "chat"))]
#[derive(Deserialize)]
struct BridgeRequest {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    request_id: String,
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
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    mcp_enabled: bool,
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn unlimited_tokens() -> i64 {
    -1
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn unit_penalty() -> f32 {
    1.0
}

#[cfg(all(feature = "mlx", feature = "chat"))]
#[derive(Serialize)]
struct NativeStats {
    ttft_seconds: f64,
    prefill_tps: f64,
    decode_tps: f64,
    prompt_tokens: usize,
    generated_tokens: usize,
    peak_memory_gb: f64,
    cached_prompt_tokens: usize,
    evaluated_prompt_tokens: usize,
    context_used: usize,
    context_limit: usize,
    elapsed_seconds: f64,
}

#[cfg(all(feature = "mlx", feature = "chat"))]
struct RoundOutput {
    raw: String,
    stats: NativeStats,
    first_text_seconds: Option<f64>,
}

#[cfg(all(feature = "mlx", feature = "chat"))]
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
    generated: &[u32],
    temperature: f32,
    top_k: usize,
    repetition_penalty: f32,
    random: &mut u64,
) -> Result<u32> {
    anyhow::ensure!(
        temperature.is_finite() && temperature >= 0.,
        "temperature must be non-negative"
    );
    anyhow::ensure!(
        repetition_penalty.is_finite() && repetition_penalty > 0.,
        "repetition_penalty must be positive"
    );
    let log_probs = logits.log_probs()?;
    if (temperature == 0. || top_k == 1) && repetition_penalty == 1. {
        return log_probs
            .argmax()?
            .to_u32()?
            .into_iter()
            .next()
            .context("model returned no logits");
    }
    let mut values = log_probs.to_f32()?;
    if repetition_penalty != 1. {
        for &token in generated.iter().rev().take(20) {
            if let Some(value) = values.get_mut(token as usize) {
                *value = if *value < 0. {
                    *value * repetition_penalty
                } else {
                    *value / repetition_penalty
                };
            }
        }
    }
    if temperature == 0. || top_k == 1 {
        return values
            .iter()
            .enumerate()
            .filter(|(_, value)| value.is_finite())
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index as u32)
            .context("model returned no finite logits");
    }
    let keep = if top_k == 0 {
        values.len()
    } else {
        top_k.min(values.len())
    };
    let mut candidates: Vec<_> = values.into_iter().enumerate().collect();
    if keep < candidates.len() {
        candidates.select_nth_unstable_by(keep, |left, right| right.1.total_cmp(&left.1));
        candidates.truncate(keep);
    }
    candidates.retain(|(_, value)| value.is_finite());
    let maximum = candidates
        .iter()
        .map(|(_, value)| *value)
        .max_by(f32::total_cmp)
        .context("model returned no finite logits")?;
    let mut total = 0f64;
    let weights: Vec<_> = candidates
        .iter()
        .map(|(_, value)| {
            let weight = (((*value - maximum) / temperature) as f64).exp();
            total += weight;
            weight
        })
        .collect();
    anyhow::ensure!(
        total.is_finite() && total > 0.,
        "invalid sampling distribution"
    );
    *random ^= *random << 13;
    *random ^= *random >> 7;
    *random ^= *random << 17;
    let mut target = (*random as f64 / u64::MAX as f64) * total;
    for ((index, _), weight) in candidates.iter().zip(weights) {
        if target <= weight {
            return Ok(*index as u32);
        }
        target -= weight;
    }
    Ok(candidates.last().expect("nonempty candidates").0 as u32)
}

#[cfg(all(feature = "mlx", feature = "chat"))]
#[allow(clippy::too_many_arguments)]
fn bridge_generate_round(
    model: &mut NativeChatModel,
    tokenizer: &mlxl3_native::tokenizer::ChatTokenizer,
    request_id: &str,
    messages: &[Value],
    tools: &[Value],
    max_tokens: i64,
    temperature: f32,
    top_k: usize,
    repetition_penalty: f32,
    context_limit: usize,
    resident_gb: f64,
    cancelled: &AtomicBool,
    random: &mut u64,
) -> Result<RoundOutput> {
    use mlxl3_native::{array::Array, streaming::Channel, tool_call::StreamFilter};
    anyhow::ensure!(!messages.is_empty(), "messages cannot be empty");
    anyhow::ensure!(
        max_tokens == -1 || max_tokens > 0,
        "max_tokens must be positive, or -1"
    );
    let rendered = tokenizer.render_values(&Value::Array(messages.to_vec()), Some(tools))?;
    let tokens = tokenizer.encode(&rendered)?;
    anyhow::ensure!(!tokens.is_empty(), "chat template produced no tokens");
    anyhow::ensure!(
        tokens.len() < context_limit,
        "conversation exceeds model context"
    );
    emit_event(json!({
        "type":"context_usage", "request_id":request_id,
        "used_tokens":tokens.len(), "context_limit":context_limit
    }))?;
    model.reset();
    let started = Instant::now();
    let mut logits: Option<Array> = None;
    for chunk in tokens.chunks(model.prefill_chunk_size()) {
        if cancelled.load(Ordering::Relaxed) {
            model.reset();
            bail!("generation cancelled");
        }
        logits = Some(model.forward_many(chunk)?);
        model.eval_state()?;
    }
    let mut logits = logits.context("no prefill output")?;
    logits.eval()?;
    let prefill_seconds = started.elapsed().as_secs_f64();
    let available = context_limit - tokens.len();
    let budget = if max_tokens == -1 {
        available
    } else {
        usize::try_from(max_tokens)?.min(available)
    };
    let mut decoder = tokenizer.tokenizer().decode_stream(false);
    let mut splitter = ThinkingSplitter::new(&rendered);
    let mut filter = (!tools.is_empty()).then(StreamFilter::new);
    let mut decoded = String::new();
    let mut generated = Vec::new();
    let mut first_token = None;
    for _ in 0..budget {
        if cancelled.load(Ordering::Relaxed) {
            model.reset();
            bail!("generation cancelled");
        }
        let next = select_token(
            &logits,
            &generated,
            temperature,
            top_k,
            repetition_penalty,
            random,
        )?;
        if tokenizer.eos_ids().contains(&next) {
            break;
        }
        first_token.get_or_insert_with(|| started.elapsed().as_secs_f64());
        generated.push(next);
        if let Some(text) = decoder
            .step(next)
            .map_err(|error| anyhow::anyhow!("token decode failed: {error}"))?
        {
            decoded.push_str(&text);
            for fragment in splitter.feed(&text) {
                let visible = if fragment.channel == Channel::Thinking {
                    vec![fragment.text]
                } else if let Some(filter) = &mut filter {
                    filter.feed(&fragment.text)
                } else {
                    vec![fragment.text]
                };
                for text in visible {
                    emit_event(json!({
                        "type":"delta", "request_id":request_id,
                        "phase":if fragment.channel == Channel::Thinking { "thinking" } else { "answer" },
                        "text":text
                    }))?;
                }
            }
        }
        if generated.len() < budget {
            logits = model.forward(next)?;
        }
    }
    let complete = tokenizer.decode(&generated)?;
    let tail = complete
        .strip_prefix(&decoded)
        .context("incremental tokenizer output differs from full decoding")?;
    for fragment in splitter.feed(tail).into_iter().chain(splitter.finish()) {
        let visible = if fragment.channel == Channel::Thinking {
            vec![fragment.text]
        } else if let Some(filter) = &mut filter {
            filter.feed(&fragment.text)
        } else {
            vec![fragment.text]
        };
        for text in visible {
            emit_event(json!({
                "type":"delta", "request_id":request_id,
                "phase":if fragment.channel == Channel::Thinking { "thinking" } else { "answer" },
                "text":text
            }))?;
        }
    }
    if let Some(filter) = &mut filter {
        for text in filter.finish() {
            emit_event(json!({
                "type":"delta", "request_id":request_id,
                "phase":"answer", "text":text
            }))?;
        }
    }
    let elapsed = started.elapsed().as_secs_f64();
    let ttft = first_token.unwrap_or(elapsed);
    let stats = NativeStats {
        ttft_seconds: ttft,
        prefill_tps: tokens.len() as f64 / prefill_seconds.max(1e-9),
        decode_tps: generated.len().saturating_sub(1) as f64 / (elapsed - ttft).max(1e-9),
        prompt_tokens: tokens.len(),
        generated_tokens: generated.len(),
        peak_memory_gb: resident_gb,
        cached_prompt_tokens: 0,
        evaluated_prompt_tokens: tokens.len(),
        context_used: tokens.len() + generated.len(),
        context_limit,
        elapsed_seconds: elapsed,
    };
    Ok(RoundOutput {
        raw: complete,
        stats,
        first_text_seconds: first_token,
    })
}

#[cfg(all(feature = "mlx", feature = "chat"))]
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
        anyhow::ensure!(
            object.get("content").is_some_and(Value::is_string),
            "message requires string content"
        );
        if let Some(turn) = object.remove("turn_context") {
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

#[cfg(all(feature = "mlx", feature = "chat"))]
#[allow(clippy::too_many_arguments)]
fn bridge_generate(
    model: &mut NativeChatModel,
    tokenizer: &mlxl3_native::tokenizer::ChatTokenizer,
    request_id: &str,
    messages: Vec<Value>,
    max_tokens: i64,
    temperature: f32,
    top_k: usize,
    repetition_penalty: f32,
    context_limit: usize,
    resident_gb: f64,
    cancelled: &AtomicBool,
    random: &mut u64,
    mcp: &mut mlxl3_native::mcp::Manager,
) -> Result<()> {
    let mut dialogue = validated_bridge_messages(messages)?;
    let original_length = dialogue.len();
    let tools = mcp.chat_tools();
    let started = Instant::now();
    let mut first_text = None;
    for round in 0..5 {
        emit_event(json!({
            "type":"generation_status", "request_id":request_id,
            "phase":"prefill", "text":if round == 0 { "Preparing context" } else { "Reading MCP results" }
        }))?;
        let output = bridge_generate_round(
            model,
            tokenizer,
            request_id,
            &dialogue,
            &tools,
            max_tokens,
            temperature,
            top_k,
            repetition_penalty,
            context_limit,
            resident_gb,
            cancelled,
            random,
        )?;
        first_text.get_or_insert_with(|| started.elapsed().as_secs_f64());
        let calls = if tools.is_empty() {
            Vec::new()
        } else {
            mlxl3_native::tool_call::parse(&output.raw)?
        };
        let context_full = output.stats.context_used >= context_limit;
        if calls.is_empty() || context_full {
            let assistant = mlxl3_native::tool_call::without_calls(&output.raw);
            let turn_context = if round == 0 {
                None
            } else {
                let mut transcript = dialogue[original_length..].to_vec();
                transcript.push(json!({"role":"assistant", "content":output.raw}));
                Some(serde_json::to_string(&transcript)?)
            };
            return emit_event(json!({
                "type":"complete", "request_id":request_id,
                "assistant_context":assistant, "cache_context":output.raw,
                "stats":output.stats, "context_full":context_full,
                "turn_context":turn_context,
                "elapsed_seconds":started.elapsed().as_secs_f64(),
                "first_text_seconds":first_text.or(output.first_text_seconds),
                "tool_rounds":round
            }));
        }
        let tool_calls: Vec<_> = calls
            .iter()
            .map(|call| json!({"type":"function", "function":{"name":call.name, "arguments":call.arguments}}))
            .collect();
        dialogue.push(json!({
            "role":"assistant",
            "content":mlxl3_native::tool_call::without_calls(&output.raw),
            "tool_calls":tool_calls
        }));
        for call in calls {
            if cancelled.load(Ordering::Relaxed) {
                model.reset();
                bail!("generation cancelled");
            }
            let tool = mcp.tools.get(&call.name);
            let server = tool.map(|tool| tool.server.clone());
            let call_id = format!("mcp-{round}-{}-{}", dialogue.len(), call.name);
            emit_event(json!({
                "type":"tool_start", "request_id":request_id,
                "tool_call_id":call_id, "tool_name":call.name,
                "server_name":server
            }))?;
            let mut result = mcp.call(&call.name, call.arguments);
            let budget = 8192usize.min((context_limit / 8).max(256));
            let result_tokens = tokenizer.encode(&result.text)?;
            if result_tokens.len() > budget {
                result.text = format!(
                    "{}\n[Tool output truncated to fit context]",
                    tokenizer.decode(&result_tokens[..budget])?
                );
            }
            emit_event(json!({
                "type":"tool_result", "request_id":request_id,
                "tool_call_id":call_id, "tool_name":call.name,
                "server_name":server, "text":result.text,
                "is_error":result.is_error
            }))?;
            dialogue.push(json!({
                "role":"tool", "name":call.name, "content":result.text
            }));
        }
    }
    bail!("the model exceeded the limit of 5 consecutive MCP tool rounds")
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
    let mut model = NativeChatModel::load(&path)?;
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
    emit_event(json!({
        "type":"ready", "model":name, "modules":checkpoint.modules.len(),
        "load_seconds":started.elapsed().as_secs_f64(), "resident_gb":resident_gb,
        "mcp_servers":0, "mcp_tools":0, "mcp_errors":{},
        "context_limit":context_limit, "model_context_limit":model_limit
    }))?;
    let cancelled = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGUSR1, Arc::clone(&cancelled))?;
    let mut random = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
        | 1;
    let mut mcp = mlxl3_native::mcp::Manager::disabled();
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
            "ping" => emit_event(json!({"type":"pong", "request_id":request.request_id}))?,
            "set_mcp" => {
                mcp.set_enabled(&registry_path, request.enabled, true);
                emit_event(json!({
                    "type":"mcp_status", "mcp_servers":mcp.server_count(),
                    "mcp_tools":mcp.tools.len(), "mcp_errors":mcp.errors
                }))?;
            }
            "generate" => {
                cancelled.store(false, Ordering::Relaxed);
                mcp.set_enabled(&registry_path, request.mcp_enabled, false);
                let request_id = request.request_id.clone();
                if let Err(error) = bridge_generate(
                    &mut model,
                    &tokenizer,
                    &request.request_id,
                    request.messages,
                    request.max_tokens,
                    request.temperature,
                    request.top_k,
                    request.repetition_penalty,
                    context_limit,
                    resident_gb,
                    &cancelled,
                    &mut random,
                    &mut mcp,
                ) {
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

#[cfg(all(feature = "mlx", feature = "chat"))]
fn native_chat(path: &std::path::Path, prompt: Option<String>, max_tokens: usize) -> Result<()> {
    use mlxl3_native::{
        array::Array,
        streaming::Channel,
        tokenizer::{ChatTokenizer, Message},
    };
    let tokenizer = ChatTokenizer::load(path)?;
    let mut model = NativeChatModel::load(path)?;
    eprintln!(
        "Loaded native {} engine (experimental, greedy sampling)",
        model.name()
    );
    let mut messages = Vec::new();
    let interactive = prompt.is_none();
    let mut one_prompt = prompt;
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
        messages.push(Message {
            role: "user".into(),
            content: user,
        });
        let rendered = tokenizer.render(&messages)?;
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
        // Replay keeps conversation contents exact. Qwen uses bounded QMM chunks;
        // other architectures retain their established token path.
        model.reset();
        let started = Instant::now();
        let mut last: Option<Array> = None;
        for chunk in tokens.chunks(model.prefill_chunk_size()) {
            last = Some(model.forward_many(chunk)?);
            model.eval_state()?;
        }
        let mut logits = last.context("no prefill output")?;
        logits.eval()?;
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
            let probabilities = logits.log_probs()?.to_f32()?;
            let mut best = None;
            for (i, value) in probabilities.iter().enumerate() {
                anyhow::ensure!(
                    !value.is_nan() && *value != f32::INFINITY,
                    "non-finite model log probabilities"
                );
                if best.is_none_or(|(_, previous)| *value > previous) {
                    best = Some((i, *value));
                }
            }
            let (index, value) = best.context("model returned no logits")?;
            anyhow::ensure!(
                value.is_finite(),
                "model returned no finite log probabilities"
            );
            let next = u32::try_from(index)?;
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
                logits = model.forward(next)?;
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
        messages.push(Message {
            role: "assistant".into(),
            content: raw_text,
        });
        if !interactive {
            break;
        }
    }
    Ok(())
}

#[cfg(all(test, feature = "mlx", feature = "chat"))]
mod tests {
    use super::*;

    #[test]
    fn native_sampler_respects_greedy_argmax() {
        let logits = mlxl3_native::array::Array::from_f32(&[1., 4., 2.], &[1, 3]).unwrap();
        let mut random = 1;
        assert_eq!(
            select_token(&logits, &[], 0., 0, 1., &mut random).unwrap(),
            1
        );
    }
}
