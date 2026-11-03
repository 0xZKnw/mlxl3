//! Header-only checkpoint inspection. Weight payloads are read only on request.
use crate::contracts;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufReader, Read},
    os::unix::fs::FileExt,
    path::{Path, PathBuf},
    sync::Arc,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [u64; 2],
    #[serde(skip)]
    pub file: PathBuf,
    #[serde(skip)]
    pub payload_offset: u64,
    #[serde(skip)]
    handle: Option<Arc<File>>,
}

fn dtype_size(dtype: &str) -> Result<u64> {
    Ok(match dtype {
        "BOOL" | "U8" | "I8" | "F8_E4M3" | "F8_E5M2" => 1,
        "U16" | "I16" | "F16" | "BF16" => 2,
        "U32" | "I32" | "F32" => 4,
        "U64" | "I64" | "F64" => 8,
        _ => bail!("unsupported safetensors dtype {dtype}"),
    })
}

pub fn read_header(path: &Path) -> Result<BTreeMap<String, TensorInfo>> {
    let mut file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_len = file.metadata()?.len();
    let mut length = [0u8; 8];
    file.read_exact(&mut length)
        .context("missing safetensors header length")?;
    let header_len = u64::from_le_bytes(length);
    ensure!(
        header_len <= 100_000_000 && header_len <= file_len.saturating_sub(8),
        "invalid safetensors header length"
    );
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let mut entries: BTreeMap<String, Value> =
        serde_json::from_slice(&header).context("invalid safetensors header JSON")?;
    if let Some(metadata) = entries.remove("__metadata__") {
        let _: BTreeMap<String, String> =
            serde_json::from_value(metadata).context("invalid safetensors metadata")?;
    }
    let payload_offset = header_len + 8;
    // Keep the inspected inode alive: an atomic download replacement must not
    // pair old offsets with a newly opened shard. read_at avoids shared cursors.
    let handle = Arc::new(file);
    let mut tensors = BTreeMap::new();
    let mut ranges = Vec::new();
    for (name, value) in entries {
        let mut info: TensorInfo =
            serde_json::from_value(value).with_context(|| format!("invalid tensor {name}"))?;
        let bytes = contracts::tensor_bytes(&info.shape, dtype_size(&info.dtype)?)
            .context("tensor size overflow")?;
        let [start, end] = info.data_offsets;
        ensure!(
            contracts::valid_data_range(start, end, file_len - payload_offset, bytes),
            "invalid data range for tensor {name}"
        );
        info.file = path.to_owned();
        info.payload_offset = payload_offset;
        info.handle = Some(Arc::clone(&handle));
        ranges.push((start, end));
        tensors.insert(name, info);
    }
    ranges.sort_unstable();
    let mut end = 0;
    for (start, next) in ranges {
        ensure!(
            start == end,
            "safetensors payload has gaps or overlapping tensors"
        );
        end = next;
    }
    ensure!(
        end == file_len - payload_offset,
        "unclaimed safetensors payload"
    );
    Ok(tensors)
}

impl TensorInfo {
    pub fn source(&self) -> Result<(&File, u64)> {
        let [absolute_start, absolute_end] =
            contracts::absolute_data_range(self.payload_offset, self.data_offsets)
                .context("invalid tensor interval or file offset overflow")?;
        let file = self
            .handle
            .as_deref()
            .context("tensor has no inspected shard handle")?;
        ensure!(
            file.metadata()?.len() >= absolute_end,
            "checkpoint was truncated after inspection"
        );
        Ok((file, absolute_start))
    }

    pub fn read_bytes(&self) -> Result<Vec<u8>> {
        let [start, end] = self.data_offsets;
        let (file, offset) = self.source()?;
        let size = usize::try_from(end - start).context("tensor does not fit address space")?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .context("not enough memory for tensor")?;
        bytes.resize(size, 0);
        file.read_exact_at(&mut bytes, offset)?;
        Ok(bytes)
    }

    pub fn read_u16(&self) -> Result<Vec<u16>> {
        ensure!(self.dtype == "U16", "expected U16, got {}", self.dtype);
        Ok(self
            .read_bytes()?
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect())
    }
}

#[derive(Debug, Serialize)]
pub struct Checkpoint {
    pub path: PathBuf,
    pub model_type: String,
    pub bits: Option<f64>,
    pub size_bytes: u64,
    pub modules: Vec<String>,
    pub tensors: BTreeMap<String, TensorInfo>,
}

#[cfg(any(feature = "mlx", all(target_os = "macos", feature = "direct-metal")))]
pub(crate) fn flash_expert_records(c: &Checkpoint, prefix: &str) -> Result<Vec<TensorInfo>> {
    const H: usize = 2560;
    const I: usize = 640;
    let mut records = Vec::with_capacity(512);
    for expert in 0..512 {
        let p = format!("{prefix}.experts.{expert}");
        let mut start = None;
        let mut cursor = 0;
        let mut file = None;
        for (projection, input, output) in
            [("down_proj", I, H), ("gate_proj", H, I), ("up_proj", H, I)]
        {
            for (part, shape, dtype) in [
                ("suh", vec![input], "F16"),
                ("svh", vec![output], "F16"),
                ("mul1", vec![], "I32"),
                ("trellis", vec![input / 16, output / 16, 32], "I16"),
            ] {
                let name = format!("{p}.{projection}.{part}");
                let tensor = c
                    .tensors
                    .get(&name)
                    .with_context(|| format!("missing {name}"))?;
                ensure!(
                    tensor.shape == shape && tensor.dtype == dtype,
                    "unsupported Flash expert tensor {name}"
                );
                if start.is_none() {
                    start = Some(tensor.clone());
                    cursor = tensor.data_offsets[0];
                    file = Some(tensor.file.clone());
                }
                ensure!(
                    Some(&tensor.file) == file.as_ref() && tensor.data_offsets[0] == cursor,
                    "Flash expert record is not contiguous"
                );
                cursor = tensor.data_offsets[1];
            }
        }
        let first = start.context("empty expert record")?;
        ensure!(
            cursor.checked_sub(first.data_offsets[0]) == Some(crate::flash_support::EXPERT_BYTES),
            "invalid Flash expert record size"
        );
        records.push(first);
    }
    Ok(records)
}

pub fn inspect(path: &Path) -> Result<Checkpoint> {
    let path = path
        .canonicalize()
        .with_context(|| format!("model directory {}", path.display()))?;
    let config: Value =
        serde_json::from_reader(BufReader::new(File::open(path.join("config.json"))?))?;
    ensure!(config.is_object(), "model config must be an object");
    let quant: Value = serde_json::from_reader(BufReader::new(File::open(
        path.join("quantization_config.json"),
    )?))?;
    ensure!(
        quant["quant_method"] == "exl3",
        "expected quant_method='exl3'"
    );
    let storage = quant["tensor_storage"]
        .as_object()
        .context("invalid EXL3 tensor_storage")?;
    let mut tensors = BTreeMap::new();
    let mut size_bytes = 0u64;
    let mut tokenizer = false;
    for entry in fs::read_dir(&path)? {
        let entry = entry?;
        if !entry.path().is_file() {
            continue;
        }
        size_bytes = size_bytes
            .checked_add(entry.metadata()?.len())
            .context("checkpoint size overflow")?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        tokenizer |=
            name.starts_with("tokenizer") || name.ends_with(".model") || name == "vocab.json";
        let flash_sidecar = config["model_type"] == "qwen4_exp"
            && matches!(
                name.as_ref(),
                "ngram_embedding.safetensors" | "mtp_hyper_connection_mixer_patch.safetensors"
            );
        if (name.starts_with("model") || flash_sidecar) && name.ends_with(".safetensors") {
            for (name, info) in read_header(&entry.path())? {
                ensure!(
                    tensors.insert(name.clone(), info).is_none(),
                    "duplicate tensor {name}"
                );
            }
        }
    }
    ensure!(!tensors.is_empty(), "checkpoint has no safetensors weights");
    ensure!(
        tokenizer,
        "missing tokenizer files; download the complete checkpoint"
    );
    let model_type = config["model_type"]
        .as_str()
        .unwrap_or("unknown")
        .to_owned();
    let mut modules = Vec::new();
    for (prefix, metadata) in storage {
        ensure!(
            metadata.is_object(),
            "invalid tensor_storage entry {prefix}"
        );
        if let Some(stored) = metadata.get("stored_tensors") {
            for (name, spec) in stored.as_object().context("invalid stored_tensors")? {
                let shape: Vec<usize> = serde_json::from_value(spec["shape"].clone())
                    .context("invalid stored tensor shape")?;
                let info = tensors
                    .get(name)
                    .or_else(|| {
                        if model_type == "lfm2_moe"
                            && name.starts_with("model.layers.")
                            && name.ends_with(".feed_forward.gate.expert_bias")
                        {
                            tensors.get(&name.replace(".gate.expert_bias", ".expert_bias"))
                        } else {
                            None
                        }
                    })
                    .with_context(|| format!("missing tensor {name}"))?;
                ensure!(info.shape == shape, "invalid shape for tensor {name}");
            }
        }
        if metadata["quant_format"] != "exl3" {
            continue;
        }
        let t = tensors
            .get(&format!("{prefix}.trellis"))
            .with_context(|| format!("missing trellis {prefix}"))?;
        ensure!(
            matches!(t.dtype.as_str(), "U16" | "I16")
                && t.shape.len() == 3
                && t.shape.iter().all(|&n| n > 0)
                && t.shape[2].is_multiple_of(16)
                && t.shape[2] <= 128,
            "invalid EXL3 trellis {prefix}"
        );
        for (axis, primary, legacy) in [(0, "suh", "su"), (1, "svh", "sv")] {
            let scale = tensors
                .get(&format!("{prefix}.{primary}"))
                .or_else(|| tensors.get(&format!("{prefix}.{legacy}")))
                .with_context(|| format!("missing scale {prefix}.{primary}"))?;
            let length = t.shape[axis]
                .checked_mul(16)
                .context("scale length overflow")?;
            ensure!(
                scale.shape == [length],
                "invalid scale shape {prefix}.{primary}"
            );
            ensure!(
                ["F16", "BF16", "F32"].contains(&scale.dtype.as_str()),
                "invalid scale dtype {prefix}.{primary}"
            );
        }
        modules.push(prefix.clone());
    }
    ensure!(!modules.is_empty(), "checkpoint contains no EXL3 modules");
    modules.sort();
    let index = path.join("model.safetensors.index.json");
    if index.exists() {
        let value: Value = serde_json::from_reader(BufReader::new(File::open(index)?))?;
        for (name, shard) in value["weight_map"]
            .as_object()
            .context("invalid shard index")?
        {
            let shard = shard.as_str().context("invalid shard name")?;
            ensure!(
                Path::new(shard).file_name().and_then(|s| s.to_str()) == Some(shard),
                "invalid shard path"
            );
            let info = tensors
                .get(name)
                .with_context(|| format!("missing indexed tensor {name}"))?;
            ensure!(
                info.file.file_name().and_then(|s| s.to_str()) == Some(shard),
                "tensor {name} is in the wrong shard"
            );
        }
    }
    Ok(Checkpoint {
        path,
        model_type,
        bits: quant["bits"].as_f64(),
        size_bytes,
        modules,
        tensors,
    })
}

pub fn artifact_key(path: &std::path::Path) -> Result<String> {
    use std::hash::{Hash, Hasher};
    let mut fingerprint = std::collections::hash_map::DefaultHasher::new();
    path.canonicalize()?.hash(&mut fingerprint);
    let mut files = std::fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;
    files.sort_by_key(|f| f.file_name());
    for file in files {
        if file
            .path()
            .extension()
            .is_some_and(|e| e == "json" || e == "safetensors" || e == "bin")
        {
            let meta = file.metadata()?;
            file.file_name().hash(&mut fingerprint);
            meta.len().hash(&mut fingerprint);
            meta.modified()?.hash(&mut fingerprint);
        }
    }
    Ok(format!("{:016x}", fingerprint.finish()))
}
