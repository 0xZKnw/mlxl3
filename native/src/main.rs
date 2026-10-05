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
#[cfg(all(feature = "mlx", feature = "chat"))]
use std::{
    collections::VecDeque,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[cfg(all(feature = "mlx", feature = "chat"))]
mod mtp_tune;
#[cfg(all(feature = "mlx", feature = "chat"))]
mod openai;
use std::{
    io::{self, BufRead, Read, Write},
    path::PathBuf,
};

#[derive(Parser)]
#[command(
    version,
    about = "Native MLXL3 Rust/Metal runtime",
    after_help = "Powered by MTPLX https://github.com/youssofal/MTPLX"
)]
struct Cli {
    #[arg(long, global = true)]
    registry: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Machine-readable engine identity; no model or registry mutation.
    RuntimeInfo,
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
    /// Serve an EXL3 model through an OpenAI-compatible HTTP API.
    Serve {
        model: String,
        #[arg(long, default_value = "127.0.0.1")]
        host: String,
        #[arg(long, default_value_t = 8000)]
        port: u16,
        #[arg(long, default_value = "")]
        api_key: String,
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
    /// Install or validate the pinned Qwen DFlash2 draft (no target weights).
    DflashDraft {
        #[arg(long)]
        inspect: Option<PathBuf>,
    },
    /// Install or inspect a native Qwen MTP head without downloading the target.
    MtpHead {
        #[arg(long)]
        inspect: Option<PathBuf>,
        /// Select/validate the head against this target's configuration.
        #[arg(long)]
        target: Option<PathBuf>,
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
        #[arg(long)]
        batch: bool,
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
    #[cfg(any(all(target_os = "macos", feature = "direct-metal"), feature = "mlx"))]
    x: Vec<u16>,
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
    #[cfg(all(target_os = "macos", feature = "direct-metal"))]
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
                #[cfg(all(target_os = "macos", feature = "direct-metal"))]
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
        Command::RuntimeInfo => {
            println!(
                "{}",
                json!({
                    "version": env!("CARGO_PKG_VERSION"),
                    "bridge_protocol": 1,
                    "architecture": std::env::consts::ARCH,
                    "mlx_enabled": cfg!(feature = "mlx"),
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
        Command::Serve {
            model,
            host,
            port,
            api_key,
            context_length,
        } => {
            #[cfg(all(feature = "mlx", feature = "chat"))]
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
            #[cfg(not(all(feature = "mlx", feature = "chat")))]
            {
                let _ = (model, host, port, api_key, context_length);
                bail!("the OpenAI API requires a build with --features mlx,chat");
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
        Command::DflashDraft { inspect } => {
            let path = if let Some(path) = inspect {
                mlxl3_native::dflash::inspect(path)?.directory
            } else {
                mlxl3_native::hub::download_dflash(|completed, total| {
                    println!(
                        "{}",
                        json!({"type":"progress", "completed":completed, "total":total})
                    );
                    let _ = io::stdout().flush();
                })?
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
        let checkpoint = checkpoint::inspect(path)?;
        Self::from_checkpoint(&checkpoint)
    }

    fn from_checkpoint(checkpoint: &checkpoint::Checkpoint) -> Result<Self> {
        match checkpoint.model_type.as_str() {
            "gemma4" => Ok(Self::Gemma4(mlxl3_native::gemma4::Gemma4::from_checkpoint(
                checkpoint,
            )?)),
            "bailing_hybrid" => Ok(Self::Ling(mlxl3_native::ling::Ling::from_checkpoint(
                checkpoint,
            )?)),
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
            Self::Lfm2(model) => model.forward(&[token]),
            Self::Ling(model) => model.forward(token),
            Self::Qwen(model) => model.forward(token),
        }
    }

    fn forward_many(&mut self, tokens: &[u32]) -> Result<mlxl3_native::array::Array> {
        anyhow::ensure!(!tokens.is_empty(), "empty token batch");
        if tokens.len() >= 24 {
            return match self {
                Self::Gemma4(model) if model.can_batch(tokens.len()) => {
                    model.forward_tokens(tokens)
                }
                Self::Lfm2(model) if model.supports_batched_prefill() => model.forward(tokens),
                Self::Ling(_) => self.forward_many_serial(tokens),
                Self::Qwen(model) => model.forward_tokens(tokens),
                Self::Gemma4(_) | Self::Lfm2(_) => self.forward_many_serial(tokens),
            };
        }
        self.forward_many_serial(tokens)
    }

    fn forward_many_serial(&mut self, tokens: &[u32]) -> Result<mlxl3_native::array::Array> {
        match self {
            Self::Qwen(model) => return model.prefill_serial(tokens),
            Self::Ling(model) => return model.prefill_serial(tokens),
            _ => {}
        }
        let mut output = None;
        for &token in tokens {
            output = Some(self.forward(token)?);
        }
        output.context("empty token batch")
    }

    fn prefill_chunk_size(&self) -> usize {
        match self {
            Self::Gemma4(_) => 256,
            Self::Lfm2(model) if model.supports_batched_prefill() => 256,
            Self::Qwen(_) => 256,
            Self::Ling(_) => 128,
            Self::Lfm2(_) => 1,
        }
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
    conversation_id: String,
    #[serde(default = "reuse_cache_by_default")]
    reuse_prompt_cache: bool,
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
    #[serde(default)]
    dflash2: bool,
    #[serde(default)]
    dflash_draft_path: String,
    #[serde(default)]
    mtp: bool,
    #[serde(default)]
    mtp_head_path: String,
    #[serde(default = "default_mtp_depth")]
    mtp_depth: usize,
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn default_mtp_depth() -> usize {
    1
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
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn reuse_cache_by_default() -> bool {
    true
}

/// One prefill checkpoint, never speculative/pending decode state. The resident
/// bridge owns one immutable model/tokenizer/context configuration. Draft reload
/// clears this cache; exact rendered IDs cover changes to system/tools/templates.
#[cfg(all(feature = "mlx", feature = "chat"))]
struct PromptCache {
    conversation: String,
    tokens: Vec<u32>,
    state: mlxl3_native::qwen35::QwenSnapshot,
    logits: mlxl3_native::array::Array,
    draft: Option<mlxl3_native::dflash::DFlashCache>,
    mtp: Option<mlxl3_native::mtp::Cache>,
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
        let NativeChatModel::Qwen(target) = model else {
            return Ok(None);
        };
        anyhow::ensure!(
            usize::try_from(target.offset()).ok() == Some(tokens.len()),
            "prefix cache position differs from its token history"
        );
        let state = target.snapshot()?;
        // ponytail: one checkpoint, capped at 256 MiB logical target state;
        // add a multi-session budget only after its extra retained RAM is measured.
        if state.byte_len()? > 256 * 1024 * 1024 {
            return Ok(None);
        }
        Ok(Some(Self {
            conversation: conversation.to_owned(),
            tokens: tokens.to_vec(),
            state,
            logits: logits.try_clone()?,
            draft: dflash
                .map(|session| session.cache.try_clone())
                .transpose()?,
            mtp: None,
        }))
    }
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn prefill_round(
    model: &mut NativeChatModel,
    tokens: &[u32],
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
            && saved.conversation == conversation
            && saved.mtp.is_none()
            && saved.draft.is_some() == draft.is_some()
            && mlxl3_native::reusable_prefix_len(&saved.tokens, tokens, chunk_size) > 0
    });
    model.reset();
    let mut cached = 0;
    let mut logits = None;
    let mut dflash = draft.map(|_| DFlashChat::new());
    if let Some(saved) = previous {
        let NativeChatModel::Qwen(target) = &mut *model else {
            bail!("invalid prefix cache model")
        };
        target.restore(saved.state)?;
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
    for (index, chunk) in tokens[cached..].chunks(chunk_size).enumerate() {
        if cancelled.load(Ordering::Relaxed) {
            model.reset();
            bail!("generation cancelled");
        }
        logits = Some(
            if let Some((session, weights)) = dflash.as_mut().zip(draft) {
                let NativeChatModel::Qwen(target) = &mut *model else {
                    bail!("DFlash2 requires Qwen3.6-35B-A3B")
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
                model.forward_many(chunk)?
            },
        );
        model.eval_state()?;
        let position = cached + index * chunk_size + chunk.len();
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
        let NativeChatModel::Qwen(target) = &mut *model else {
            bail!("MTP requires Qwen");
        };
        target.restore(saved.state)?;
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
            saved.mtp = Some(head.snapshot()?);
        }
        *cache = saved;
    }
    // At every checkpoint the last trunk hidden is deliberately unpaired.
    // The next prompt token is supplied only after prefix eligibility is known.
    for start in (cached..tokens.len()).step_by(CHUNK) {
        if cancelled.load(Ordering::Relaxed) {
            model.reset();
            head.reset()?;
            bail!("generation cancelled");
        }
        let NativeChatModel::Qwen(target) = &mut *model else {
            bail!("MTP requires Qwen");
        };
        if start > 0 {
            let last = target.mtp_hidden()?.try_clone()?;
            head.extend_cache(target, &last, &tokens[start..start + 1])?;
        }
        let end = start.saturating_add(CHUNK).min(tokens.len());
        let (output, raw) = target.forward_mtp(&tokens[start..end])?;
        let paired = end - start - 1;
        if paired > 0 {
            head.extend_cache(
                target,
                &raw.slice(1, 0, paired as i32)?,
                &tokens[start + 1..end],
            )?;
        }
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
                saved.mtp = Some(head.snapshot()?);
            }
            *cache = saved;
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
        }
    }

    fn proposals_count(&self, context_remaining: usize, output_remaining: usize) -> Option<usize> {
        let (accepted, proposed) = self
            .recent
            .iter()
            .fold((0, 0), |(accepted, proposed), &(a, p)| {
                (accepted + a, proposed + p)
            });
        mlxl3_native::speculative::adaptive_proposals(
            context_remaining,
            output_remaining,
            self.recent.len(),
            accepted,
            proposed,
        )
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

    fn advance(
        &mut self,
        target: &mut mlxl3_native::qwen35::Qwen35Moe,
        draft: &mlxl3_native::dflash::DFlashWeights,
        anchor: u32,
        context_limit: usize,
        output_remaining: usize,
    ) -> Result<u32> {
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
            return target
                .forward(anchor)?
                .chat_greedy_ids()?
                .into_iter()
                .next()
                .context("model returned no logits");
        }
        let block_started = Instant::now();
        let input = target.dflash_input(anchor, 248_077)?;
        let output = draft.forward_hidden(&input, &self.cache, target.offset())?;
        let draft_logits = target.dflash_logits(&output.hidden, proposals_count)?;
        let proposals = draft.select_greedy(&draft_logits, &output.selector, anchor)?;
        let verify = std::iter::once(anchor)
            .chain(proposals.iter().copied())
            .collect::<Vec<_>>();
        let position = target.offset();
        let (target_logits, captured) = target.verify_tokens_exact_with_dflash_capture(&verify)?;
        let target_tokens = target_logits.chat_greedy_ids()?;
        let acceptance = mlxl3_native::speculative::greedy_accept(&proposals, &target_tokens)
            .context("invalid DFlash target verification")?;
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
        self.pending.extend(proposals[..accepted].iter().copied());
        self.pending.push_back(acceptance.target_token);
        // Wall time through materialized token selection, without extra GPU
        // barriers. Deferred cache/commit work may be paid by the next block.
        self.block_seconds += block_started.elapsed().as_secs_f64();
        self.pending
            .pop_front()
            .context("DFlash produced no target token")
    }
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
    end_to_end_ttft_seconds: Option<f64>,
    tool_seconds: f64,
    draft_load_seconds: f64,
    tool_rounds: usize,
    round_count: usize,
    memory: Option<mlxl3_native::array::MemoryStats>,
    dflash_proposed_tokens: Option<usize>,
    dflash_accepted_tokens: Option<usize>,
    dflash_blocks: Option<usize>,
    dflash_block_seconds: f64,
    mtp_proposed_tokens: Option<usize>,
    mtp_accepted_tokens: Option<usize>,
    mtp_blocks: Option<usize>,
}

#[cfg(all(feature = "mlx", feature = "chat"))]
impl NativeStats {
    fn accumulate(&mut self, round: &Self) {
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
        self.memory = round.memory;
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
            (&mut self.mtp_proposed_tokens, round.mtp_proposed_tokens),
            (&mut self.mtp_accepted_tokens, round.mtp_accepted_tokens),
            (&mut self.mtp_blocks, round.mtp_blocks),
        ] {
            if let Some(value) = value {
                *sum = Some(sum.unwrap_or(0) + value);
            }
        }
    }
}

#[cfg(all(feature = "mlx", feature = "chat"))]
struct RoundOutput {
    raw: String,
    stats: NativeStats,
    first_text_seconds: Option<f64>,
    token_hash: String,
}

#[cfg(all(feature = "mlx", feature = "chat"))]
fn token_hash(tokens: &[u32]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for token in tokens {
        for byte in token.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
        }
    }
    format!("{hash:016x}")
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
    if (temperature == 0. || top_k == 1) && repetition_penalty == 1. {
        return logits
            .chat_greedy_ids()?
            .into_iter()
            .next()
            .context("model returned no logits");
    }
    let mut values = logits.log_probs()?.to_f32()?;
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
    draft: Option<&mlxl3_native::dflash::DFlashWeights>,
    mut mtp: Option<&mut mlxl3_native::mtp::Head>,
    mtp_depth: usize,
    cache: &mut Option<PromptCache>,
    conversation: &str,
    reuse: bool,
) -> Result<RoundOutput> {
    use mlxl3_native::{streaming::Channel, tool_call::StreamFilter};
    let started = Instant::now();
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
    let prepare_seconds = started.elapsed().as_secs_f64();
    let prefill_started = Instant::now();
    let (mut logits, mut dflash, cached) = if let Some(head) = mtp.as_deref_mut() {
        prefill_mtp(model, head, &tokens, cache, conversation, reuse, cancelled)?
    } else {
        prefill_round(model, &tokens, draft, cache, conversation, reuse, cancelled)?
    };
    let mut mtp_session = mtp
        .as_ref()
        .map(|_| mlxl3_native::mtp::Session::new(mtp_depth))
        .transpose()?;
    let prefill_seconds = prefill_started.elapsed().as_secs_f64();
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
    let mut first_visible_text = None;
    let mut dflash_next = None;
    for _ in 0..budget {
        if cancelled.load(Ordering::Relaxed) {
            model.reset();
            bail!("generation cancelled");
        }
        let next = if let Some(token) = dflash_next.take() {
            token
        } else {
            select_token(
                &logits,
                &generated,
                temperature,
                top_k,
                repetition_penalty,
                random,
            )?
        };
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
                    if !text.is_empty() {
                        first_visible_text.get_or_insert_with(|| started.elapsed().as_secs_f64());
                    }
                    emit_event(json!({
                        "type":"delta", "request_id":request_id,
                        "phase":if fragment.channel == Channel::Thinking { "thinking" } else { "answer" },
                        "text":text
                    }))?;
                }
            }
        }
        if generated.len() < budget {
            if let Some((session, head)) = mtp_session.as_mut().zip(mtp.as_deref_mut()) {
                let NativeChatModel::Qwen(target) = &mut *model else {
                    bail!("MTP requires Qwen3.5/3.6");
                };
                dflash_next = Some(session.advance(
                    target,
                    head,
                    next,
                    context_limit,
                    budget - generated.len(),
                )?);
            } else if let Some((session, weights)) = dflash.as_mut().zip(draft) {
                let NativeChatModel::Qwen(target) = &mut *model else {
                    bail!("DFlash2 requires Qwen3.6-35B-A3B")
                };
                dflash_next = Some(session.advance(
                    target,
                    weights,
                    next,
                    context_limit,
                    budget - generated.len(),
                )?);
            } else {
                logits = model.forward(next)?;
            }
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
            if !text.is_empty() {
                first_visible_text.get_or_insert_with(|| started.elapsed().as_secs_f64());
            }
            emit_event(json!({
                "type":"delta", "request_id":request_id,
                "phase":if fragment.channel == Channel::Thinking { "thinking" } else { "answer" },
                "text":text
            }))?;
        }
    }
    if let Some(filter) = &mut filter {
        for text in filter.finish() {
            if !text.is_empty() {
                first_visible_text.get_or_insert_with(|| started.elapsed().as_secs_f64());
            }
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
        memory: mlxl3_native::array::memory_stats(false).ok(),
        dflash_proposed_tokens: dflash.as_ref().map(|session| session.proposed),
        dflash_accepted_tokens: dflash.as_ref().map(|session| session.accepted),
        dflash_blocks: dflash.as_ref().map(|session| session.blocks),
        dflash_block_seconds: dflash.as_ref().map_or(0., |session| session.block_seconds),
        mtp_proposed_tokens: mtp_session.as_ref().map(|s| s.proposed),
        mtp_accepted_tokens: mtp_session.as_ref().map(|s| s.accepted),
        mtp_blocks: mtp_session.as_ref().map(|s| s.blocks),
        ..Default::default()
    };
    Ok(RoundOutput {
        raw: complete,
        stats,
        first_text_seconds: first_visible_text,
        token_hash: token_hash(&generated),
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
    draft: Option<&mlxl3_native::dflash::DFlashWeights>,
    mut mtp: Option<&mut mlxl3_native::mtp::Head>,
    mtp_depth: usize,
    started: Instant,
    draft_load_seconds: f64,
    cache: &mut Option<PromptCache>,
    conversation: &str,
    reuse: bool,
) -> Result<()> {
    let mut dialogue = validated_bridge_messages(messages)?;
    let original_length = dialogue.len();
    let tools = mcp.chat_tools();
    let mut first_text = None;
    let mut summary: Option<NativeStats> = None;
    let mut round_stats = Vec::new();
    let mut tool_seconds = 0.;
    for round in 0..5 {
        emit_event(json!({
            "type":"generation_status", "request_id":request_id,
            "phase":"prefill", "text":if round == 0 { "Preparing context" } else { "Reading MCP results" }
        }))?;
        let round_start = started.elapsed().as_secs_f64();
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
            draft,
            mtp.as_deref_mut(),
            mtp_depth,
            cache,
            conversation,
            reuse,
        )?;
        if first_text.is_none() {
            first_text = output
                .first_text_seconds
                .map(|seconds| round_start + seconds);
        }
        if let Some(summary) = &mut summary {
            summary.accumulate(&output.stats);
        } else {
            let mut first = output.stats.clone();
            first.ttft_seconds += round_start;
            summary = Some(first);
        }
        round_stats.push(output.stats.clone());
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
            let mut stats = summary.context("missing generation statistics")?;
            stats.elapsed_seconds = started.elapsed().as_secs_f64();
            stats.end_to_end_ttft_seconds = first_text;
            stats.tool_seconds = tool_seconds;
            stats.draft_load_seconds = draft_load_seconds;
            stats.tool_rounds = round;
            return emit_event(json!({
                "type":"complete", "request_id":request_id,
                "assistant_context":assistant, "cache_context":output.raw,
                "token_hash":output.token_hash,
                "stats":stats, "round_stats":round_stats, "context_full":context_full,
                "turn_context":turn_context,
                "elapsed_seconds":started.elapsed().as_secs_f64(),
                "first_text_seconds":first_text,
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
            let tool_started = Instant::now();
            let mut result = mcp.call(&call.name, call.arguments);
            tool_seconds += tool_started.elapsed().as_secs_f64();
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
fn load_mtp_head(
    head: &mut Option<(PathBuf, mlxl3_native::mtp::Head)>,
    revision: &mut Option<String>,
    prompt_cache: &mut Option<PromptCache>,
    requested: &str,
    target: &mlxl3_native::qwen35::Qwen35Moe,
) -> Result<bool> {
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
    *head = Some((path.clone(), mlxl3_native::mtp::Head::load(&path, target)?));
    *revision = Some(next_revision);
    Ok(true)
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
    let dflash_supported =
        matches!(&model, NativeChatModel::Qwen(target) if target.supports_dflash());
    let mtp_auto_download_supported = matches!(&model, NativeChatModel::Qwen(_))
        && mlxl3_native::mtp::target_layout(&path)
            .is_ok_and(|(layout, layers)| layout.managed_head(layers).is_some());
    let executable = std::env::current_exe()?.display().to_string();
    let tuning_key = mtp_tune::runtime_key(&path, context_limit)?;
    emit_event(json!({
        "type":"ready", "model":name, "modules":checkpoint.modules.len(),
        "load_seconds":started.elapsed().as_secs_f64(), "resident_gb":resident_gb,
        "runtime_commit":env!("MLXL3_BUILD_REVISION"),
        "runtime_profile":env!("MLXL3_BUILD_PROFILE"),
        "mlx_version":env!("MLXL3_MLX_VERSION"),
        "runtime_executable":executable,
        "dflash_supported":dflash_supported,
        "mtp_supported":matches!(&model, NativeChatModel::Qwen(_)),
        "mtp_auto_download_supported":mtp_auto_download_supported,
        "mtp_configure_supported":true,
        "mtp_max_depth":3, "mtp_tune_supported":matches!(&model, NativeChatModel::Qwen(_)),
        "mtp_tuning_key":tuning_key,
        "runtime_version":env!("CARGO_PKG_VERSION"), "bridge_protocol":1,
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
    let mut dflash_weights: Option<(PathBuf, mlxl3_native::dflash::DFlashWeights)> = None;
    let mut mtp_head: Option<(PathBuf, mlxl3_native::mtp::Head)> = None;
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
            "ping" => emit_event(json!({"type":"pong", "request_id":request.request_id}))?,
            "set_mcp" => {
                mcp.set_enabled(&registry_path, request.enabled, true);
                emit_event(json!({
                    "type":"mcp_status", "mcp_servers":mcp.server_count(),
                    "mcp_tools":mcp.tools.len(), "mcp_errors":mcp.errors
                }))?;
            }
            "set_mtp" => {
                prompt_cache = None;
                model.reset();
                let result = (|| -> Result<()> {
                    if request.enabled {
                        let NativeChatModel::Qwen(target) = &model else {
                            bail!("MTP requires a Qwen target");
                        };
                        dflash_weights = None;
                        load_mtp_head(
                            &mut mtp_head,
                            &mut mtp_head_revision,
                            &mut prompt_cache,
                            &request.mtp_head_path,
                            target,
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
            "tune_mtp" => {
                cancelled.store(false, Ordering::Relaxed);
                prompt_cache = None;
                dflash_weights = None;
                let result = (|| -> Result<()> {
                    let NativeChatModel::Qwen(target) = &model else {
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
                        target,
                    )?;
                    mtp_tune::run(
                        &mut model,
                        &mut mtp_head.as_mut().context("missing MTP head")?.1,
                        &tokenizer,
                        &request.request_id,
                        &tuning_key,
                        context_limit,
                        &cancelled,
                    )
                })();
                model.reset();
                if let Some((_, head)) = &mut mtp_head {
                    head.reset()?;
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
                let started = Instant::now();
                let _ = mlxl3_native::array::memory_stats(true);
                cancelled.store(false, Ordering::Relaxed);
                mcp.set_enabled(&registry_path, request.mcp_enabled, false);
                let request_id = request.request_id.clone();
                let result = (|| -> Result<()> {
                    let mut draft_load_seconds = 0.;
                    anyhow::ensure!(
                        !(request.mtp && request.dflash2),
                        "Choose either MTP or DFlash2"
                    );
                    let mtp_active = request.mtp
                        && (request.temperature == 0. || request.top_k == 1)
                        && request.repetition_penalty == 1.;
                    if mtp_active {
                        anyhow::ensure!(
                            (1..=3).contains(&request.mtp_depth),
                            "MTP depth must be 1, 2 or 3"
                        );
                        let NativeChatModel::Qwen(target) = &model else {
                            bail!("MTP supports Qwen3.5/3.6/3.8 targets");
                        };
                        let started = Instant::now();
                        if load_mtp_head(
                            &mut mtp_head,
                            &mut mtp_head_revision,
                            &mut prompt_cache,
                            &request.mtp_head_path,
                            target,
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
                            dflash_supported,
                            "DFlash2 currently supports Qwen3.6-35B-A3B only"
                        );
                        anyhow::ensure!(
                            (request.temperature == 0. || request.top_k == 1)
                                && request.repetition_penalty == 1.,
                            "DFlash2 requires greedy sampling and repetition penalty 1.0"
                        );
                        anyhow::ensure!(
                            !request.dflash_draft_path.trim().is_empty(),
                            "DFlash2 draft folder is not configured"
                        );
                        let draft_path =
                            registry::expand_home(&PathBuf::from(&request.dflash_draft_path))?;
                        if dflash_weights.as_ref().map(|(path, _)| path) != Some(&draft_path) {
                            prompt_cache = None;
                            emit_event(json!({
                                "type":"generation_status", "request_id":request.request_id,
                                "phase":"loading_draft", "text":"Preparing DFlash2"
                            }))?;
                            let draft_started = Instant::now();
                            let package = mlxl3_native::dflash::inspect(&draft_path)?;
                            dflash_weights = Some((
                                draft_path,
                                mlxl3_native::dflash::DFlashWeights::load(&package)?,
                            ));
                            draft_load_seconds = draft_started.elapsed().as_secs_f64();
                        }
                    } else {
                        if dflash_weights.is_some() {
                            prompt_cache = None;
                        }
                        dflash_weights = None;
                    }
                    emit_event(json!({
                        "type":"generation_mode", "request_id":request.request_id,
                        "dflash_requested":request.dflash2,
                        "dflash_active":dflash_weights.is_some(),
                        "mtp_requested":request.mtp, "mtp_active":mtp_active,
                        "mtp_depth":if mtp_active { request.mtp_depth } else { 0 },
                        "mtp_reason":if request.mtp && !mtp_active { Some("MTP uses greedy sampling and repetition penalty 1; ordinary sampling remains active") } else { None },
                        "dflash_proposals":if dflash_weights.is_some() { 5 } else { 0 },
                        "dflash_draft_path":dflash_weights.as_ref().map(|(path, _)| path.display().to_string())
                    }))?;
                    bridge_generate(
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
                        dflash_weights.as_ref().map(|(_, weights)| weights),
                        mtp_head.as_mut().map(|(_, head)| head),
                        request.mtp_depth,
                        started,
                        draft_load_seconds,
                        &mut prompt_cache,
                        if request.conversation_id.is_empty() {
                            &request.request_id
                        } else {
                            &request.conversation_id
                        },
                        request.reuse_prompt_cache,
                    )
                })();
                if let Err(error) = result {
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
    fn dflash_width_tracks_recent_acceptance() {
        let mut session = DFlashChat::new();
        session.blocks = 16;
        session.accepted = 48;
        session.proposed = 80;
        session.recent = std::iter::repeat_n((1, 5), 8).collect();
        assert_eq!(session.proposals_count(128, 128), Some(2));
    }

    #[test]
    fn native_sampler_respects_greedy_argmax() {
        let logits = mlxl3_native::array::Array::from_f32(&[1., 4., 2.], &[1, 3]).unwrap();
        let mut random = 1;
        assert_eq!(
            select_token(&logits, &[], 0., 0, 1., &mut random).unwrap(),
            1
        );
    }

    #[test]
    fn bridge_token_hash_uses_ordered_token_ids() {
        assert_eq!(token_hash(&[]), "cbf29ce484222325");
        assert_ne!(token_hash(&[1, 2]), token_hash(&[2, 1]));
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
            dflash_proposed_tokens: Some(10),
            dflash_accepted_tokens: Some(6),
            dflash_blocks: Some(2),
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
            dflash_proposed_tokens: Some(40),
            dflash_accepted_tokens: Some(22),
            dflash_blocks: Some(8),
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
            &mut model, &tokens, None, &mut None, "normal", false, &cancelled,
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
                &mut model, &original, draft, &mut cache, "a", true, &cancelled,
            )?;
            assert_eq!(count, 0, "mode switch must miss");
            let cold = cold.to_f16_bits()?;
            model.forward(51)?.eval()?; // Live decode may be ahead of the saved prefix.
            let (replay, _, count) = prefill_round(
                &mut model, &original, draft, &mut cache, "a", true, &cancelled,
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
                    &mut model, &extended, draft, &mut None, "a", false, &cancelled,
                )?;
                let expected = expected.to_f16_bits()?;
                let (actual, _, count) = prefill_round(
                    &mut model, &extended, draft, &mut cache, "a", true, &cancelled,
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
                &mut model, &edited, draft, &mut cache, "a", true, &cancelled,
            )?;
            assert_eq!(count, 0, "edited history must miss");
            let (_, _, count) = prefill_round(
                &mut model, &edited, draft, &mut cache, "b", true, &cancelled,
            )?;
            assert_eq!(count, 0, "conversation switch must miss");
            cancelled.store(true, Ordering::Relaxed);
            assert!(
                prefill_round(
                    &mut model, &edited, draft, &mut cache, "b", true, &cancelled
                )
                .is_err()
            );
            cancelled.store(false, Ordering::Relaxed);
            let (_, _, count) = prefill_round(
                &mut model, &edited, draft, &mut cache, "b", false, &cancelled,
            )?;
            assert_eq!(count, 0);
            assert!(cache.is_none(), "cache OFF must release the saved state");
        }
        Ok(())
    }
}
