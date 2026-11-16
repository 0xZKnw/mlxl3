//! Platform logic of MLXL3 Desktop for Windows.
//!
//! The Tauri shell stays thin; everything that can be tested off Windows
//! lives here: the JSON-lines bridge framing and events, the ExLlamaV3
//! runtime specification and install plan, the engine process supervisor,
//! GPU/WSL detection parsing, Windows→WSL path translation and the model
//! library. The engine itself is upstream ExLlamaV3 driven by the Python
//! worker in `engines/exl3`.

pub mod agent;
pub mod documents;
pub mod extract;
pub mod framer;
pub mod gpu;
pub mod hub;
pub mod library;
pub mod mcp;
pub mod protocol;
pub mod runtime;
pub mod server;
pub mod structured;
pub mod supervisor;
pub mod tool_call;
pub mod turn;
pub mod update;
pub mod wsl;

/// Which ExLlamaV3 build runs the model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    /// Native Windows process, upstream CUDA 12.8 wheel.
    Cuda,
    /// Process inside a WSL2 distribution, upstream ROCm 10 Linux wheel.
    RocmWsl,
}

impl Backend {
    pub fn label(self) -> &'static str {
        match self {
            Self::Cuda => "NVIDIA CUDA",
            Self::RocmWsl => "AMD ROCm (WSL2)",
        }
    }
}
