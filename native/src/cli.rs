use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    version,
    about = "Native MLXL3 Rust/Metal runtime",
    after_help = "Powered by MTPLX https://github.com/youssofal/MTPLX"
)]
pub(crate) struct Cli {
    #[arg(long, global = true)]
    pub(crate) registry: Option<PathBuf>,
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
pub(crate) enum Command {
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
    /// Quantize complete FP32 kernel-order tiles with the autonomous Metal trellis.
    QuantizeTiles {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        k: u32,
        #[arg(long, default_value_t = 0)]
        codebook: u32,
        #[arg(long, default_value_t = 256 * 1024 * 1024)]
        max_scratch_bytes: u64,
    },
    /// Owned sequential tile service; retains one Metal device, queue and pipeline.
    QuantizeWorker {
        root: PathBuf,
        #[arg(long)]
        k: u32,
        #[arg(long, default_value_t = 0)]
        codebook: u32,
        #[arg(long, default_value_t = 256 * 1024 * 1024)]
        max_scratch_bytes: u64,
    },
    /// Multiply finite FP32 matrices for the converter's feedback boundary.
    ConvertMatmul {
        left: PathBuf,
        right: PathBuf,
        output: PathBuf,
        #[arg(long)]
        m: u32,
        #[arg(long)]
        n: u32,
        #[arg(long)]
        k: u32,
        #[arg(long)]
        full_fp32: bool,
    },
    /// Feed a JSON array of text chunks on stdin; return reasoning/answer fragments.
    Split {
        #[arg(long, default_value = "")]
        prompt: String,
    },
    /// Chat with a local or registered model using the compiled inference backend.
    Run {
        model: String,
        #[arg(short, long)]
        prompt: Option<String>,
        #[arg(long, default_value_t = 128)]
        max_tokens: usize,
        /// Image files for LFM3-VL (PNG, JPEG or WebP). Requires --prompt.
        #[arg(long = "image", requires = "prompt")]
        images: Vec<PathBuf>,
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
        /// Read a private bearer key from standard input, up to 512 bytes.
        #[arg(long, conflicts_with = "api_key")]
        api_key_stdin: bool,
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
        /// Download/validate the draft matching this target's configuration.
        #[arg(long)]
        target: Option<PathBuf>,
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
    #[cfg(any(
        feature = "hybrid-dispatch",
        all(
            feature = "chat",
            any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
        )
    ))]
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
pub(crate) enum HubAction {
    Search,
    Details,
    Download,
    Pending,
    Resume,
    Discard,
    Auth,
}

#[derive(Subcommand)]
pub(crate) enum McpAction {
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

#[cfg(any(
    feature = "hybrid-dispatch",
    all(
        feature = "chat",
        any(feature = "mlx", all(target_os = "macos", feature = "direct-metal"))
    )
))]
#[allow(dead_code)] // The CPU-only main parses Forward but cannot execute an inference backend.
pub(crate) fn resolve_model_path(
    registry_path: Option<&std::path::Path>,
    model: &str,
) -> anyhow::Result<PathBuf> {
    use anyhow::Context;
    use mlxl3_native::registry;
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
