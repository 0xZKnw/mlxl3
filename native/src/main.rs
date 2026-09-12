use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use mlxl3_native::{
    checkpoint,
    codec::{self, Codebook},
    registry,
    streaming::ThinkingSplitter,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    io::{self, BufRead, Write},
    path::PathBuf,
};

#[derive(Parser)]
#[command(version, about = "Experimental native MLXL3 Rust/Metal port")]
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
    /// Experimental LFM2 chat through native MLX (requires mlx,chat features).
    Run {
        model: String,
        #[arg(short, long)]
        prompt: Option<String>,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
    },
    /// Emit logits for an imposed token sequence (native parity check).
    #[cfg(feature = "mlx")]
    Forward {
        model: PathBuf,
        #[arg(long, value_delimiter = ',', required = true)]
        tokens: Vec<u32>,
        #[arg(long)]
        states: bool,
    },
}

#[derive(Deserialize)]
struct CodecRequest {
    op: String,
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
                    let rows = i32::try_from(request.x.len())?;
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
                    let x = Array::from_f16_bits(&request.x, &[1, rows])?;
                    Ok(json!(layer.forward(&x)?.to_f16_bits()?))
                }
                #[cfg(feature = "mlx")]
                "mlx-group" => {
                    use mlxl3_native::{
                        array::Array,
                        linear::{Exl3Group, Exl3Linear},
                    };
                    codec::check_k(request.k)?;
                    let rows = request.x.len();
                    anyhow::ensure!(
                        rows > 0 && rows.is_multiple_of(128),
                        "invalid grouped input size"
                    );
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
                        &[1, i32::try_from(rows)?],
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
                    let q_shape = [batch, 1, key_heads, 128];
                    let v_shape = [batch, 1, value_heads, value_dims];
                    let state_shape = [batch, value_heads, value_dims, 128];
                    let (output, state) = gated_delta::step(
                        &Array::from_f16_bits(&request.q, &q_shape)?,
                        &Array::from_f16_bits(&request.x, &q_shape)?,
                        &Array::from_f16_bits(&request.v, &v_shape)?,
                        &Array::from_f32(&request.g, &[batch, 1, value_heads])?,
                        &Array::from_f16_bits(&request.beta, &[batch, 1, value_heads])?,
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
                    let experts = i32::try_from(request.data.len())?;
                    let (indices, scores) = router::topk(
                        &Array::from_f16_bits(&request.data, &[1, experts])?,
                        request.k,
                        request.normalize,
                    )?;
                    Ok(json!({
                        "indices": indices.to_u32()?,
                        "scores": scores.to_f16_bits()?,
                    }))
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
        #[cfg(feature = "mlx")]
        Command::Forward {
            model,
            tokens,
            states,
        } => {
            let mut model = mlxl3_native::lfm2::Lfm2::load(&model)?;
            let mut out = io::stdout().lock();
            for token in tokens {
                let logits = model.forward(&[token])?;
                let state = if states {
                    model.state_arrays().into_iter().map(|(name, array)| -> Result<_> {
                        Ok(json!({"name": name, "shape": array.shape(), "data": array.to_bytes()?}))
                    }).collect::<Result<Vec<_>>>()?
                } else {
                    Vec::new()
                };
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

#[cfg(all(feature = "mlx", feature = "chat"))]
fn native_chat(path: &std::path::Path, prompt: Option<String>, max_tokens: usize) -> Result<()> {
    use mlxl3_native::{
        array::Array,
        lfm2::Lfm2,
        streaming::Channel,
        tokenizer::{ChatTokenizer, Message},
    };
    use std::time::Instant;
    let tokenizer = ChatTokenizer::load(path)?;
    eprintln!("Loading native LFM2 engine… (experimental, greedy sampling)");
    let mut model = Lfm2::load(path)?;
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
        // Prototype replay: correct conversation contents, but prefix reuse and
        // batched prefill are not ported yet. Evaluate recurrent state per token
        // so a long prompt cannot retain an unbounded lazy activation graph.
        model.reset();
        let started = Instant::now();
        let mut last: Option<Array> = None;
        for token in &tokens {
            last = Some(model.forward(&[*token])?);
            for (_, state) in model.state_arrays() {
                state.eval()?;
            }
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
                logits = model.forward(&[next])?;
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
