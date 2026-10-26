//! Hugging Face catalogue and resumable model/draft transfers.
use anyhow::{Context, Result, bail, ensure};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::Write,
    os::unix::fs::{OpenOptionsExt, PermissionsExt},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::Duration,
};

const API: &str = "https://huggingface.co/api";
#[cfg(test)]
const DFLASH_REPO: &str = "incoai/Qwen3.6-35B-A3B-Splash";
#[cfg(test)]
const DFLASH_COMMIT: &str = "0f4714b2db37b5f3c42a10de07281e74f88e4adc";
const DFLASH_FILES: [&str; 7] = [
    "draft/layer-0.bin",
    "draft/layer-1.bin",
    "draft/layer-2.bin",
    "draft/layer-3.bin",
    "draft/layer-4.bin",
    "draft/layer-5.bin",
    "draft/model.bin",
];

const DFLASH_MOE_PINNED: [(&str, u64, &str); 7] = [
    (
        "draft/layer-0.bin",
        34324480,
        "6e49b64d9448bf9965826d6c92badcd40c3321a7b86066c25b7c30852cc4c5b9",
    ),
    (
        "draft/layer-1.bin",
        34324480,
        "755ef7fcf8107c3a2cd5bd24e96ad726aa06174f72d628413e32fa7b7d8ffdda",
    ),
    (
        "draft/layer-2.bin",
        34324480,
        "126cedd11a4b13febc299df0101a096dd6f6761e86bc0ac5db1dc6f2f78c31c1",
    ),
    (
        "draft/layer-3.bin",
        34324480,
        "d96490a76307408fc2d2e42d86481f3061b04a2289193d2e8face84531b68f9e",
    ),
    (
        "draft/layer-4.bin",
        34324480,
        "92103acb13eb656fc46220f07bdc4f1da0233fed8b27ab5f2877544f1147bd93",
    ),
    (
        "draft/layer-5.bin",
        34324480,
        "2de274812de6d8cdf8bdfb2b6edd43da1b7b4b82e5ffe7316e36d6d50404600d",
    ),
    (
        "draft/model.bin",
        273498112,
        "532f9ca34fd57b0d042f79232bea70770c28e23d69eeb2293a46e9640959926f",
    ),
];

const DFLASH_DENSE_PINNED: [(&str, u64, &str); 6] = [
    (
        "draft/layer-0.bin",
        187449344,
        "48250072cd0c121ec449b67a53bb98c73d16135d5d9a7d0366b49b7e4573a29b",
    ),
    (
        "draft/layer-1.bin",
        187449344,
        "b8f1b37e71e25f4cb3ac32d34d4e81be2e0634210cdb64bbee945413e511a664",
    ),
    (
        "draft/layer-2.bin",
        187449344,
        "f232403c4ef212c6e53abe03f0e48aaac355d3612384dd103e29274b5bbe25db",
    ),
    (
        "draft/layer-3.bin",
        187449344,
        "12acda893bc646c02b162ddf09a3af45babc73cf45975974ecc007cf14b05127",
    ),
    (
        "draft/layer-4.bin",
        187449344,
        "db6683f4f30da4639163428fc39e87440e96dc9a3e4dcc617a8ed44ed3701e1f",
    ),
    (
        "draft/model.bin",
        328794112,
        "5cb77c477c32faab66918fe505c96ba5b9d311e91eb519281f54cb9c5bb3a4f2",
    ),
];

fn pinned_dflash_files(
    family: crate::dflash::Family,
) -> &'static [(&'static str, u64, &'static str)] {
    match family {
        crate::dflash::Family::Qwen36Moe => &DFLASH_MOE_PINNED,
        crate::dflash::Family::Qwen38Dense => &DFLASH_DENSE_PINNED,
    }
}

const MTP_MOE_FILES: [(&str, u64, &str); 2] = [
    (
        "config.json",
        3180,
        "7b38a336fa246a285ee23cc990351989bf2ee6c040506a2b793ee24e463f35a7",
    ),
    (
        "model.safetensors",
        475130833,
        "77fbc6594cdd830cae89e0b693f18278dd0a7a7c5749fd33d4bc5817dbb91bad",
    ),
];
const MTP_DENSE_FILES: [(&str, u64, &str); 2] = [
    (
        "config.json",
        3804,
        "16094efa6177985ab3725a9d6d61d6ab248b71e4b42a114efddcdb2aaddc0a55",
    ),
    (
        "model.safetensors",
        238934137,
        "76663c101e7e8ea9c0ae17bcb95183cd7f733ce424c912b8b264a7b1c48e4cc6",
    ),
];

/// The target and tokenizer are never downloaded by this operation.
pub fn download_mtp(
    kind: crate::mtp::ManagedHead,
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf> {
    let (repo, commit, files) = match kind {
        crate::mtp::ManagedHead::Qwen36Moe => (
            "mlx-community/Qwen3.6-35B-A3B-MTP-4bit",
            "0295b81421bf4d0fccca9a7c0fcfb1418dda3516",
            MTP_MOE_FILES,
        ),
        crate::mtp::ManagedHead::Qwen38Dense => (
            "mlx-community/Qwen3.8-27B-MTP-4bit",
            "b643c01b6d3b094e325edb6ebd832e16c486c575",
            MTP_DENSE_FILES,
        ),
    };
    let root = managed_drafts_path()?;
    fs::create_dir_all(&root)?;
    let root = root.canonicalize()?;
    let destination = root.join(
        repo.rsplit('/')
            .next()
            .context("missing MTP package name")?,
    );
    let regular = |path: &Path| {
        !path
            .symlink_metadata()
            .is_ok_and(|m| m.file_type().is_symlink())
    };
    ensure!(regular(&destination), "invalid MTP destination");
    let verified = |directory: &Path| -> Result<bool> {
        for (name, size, hash) in files {
            let file = directory.join(name);
            if !regular(&file)
                || !file
                    .metadata()
                    .is_ok_and(|m| m.is_file() && m.len() == size)
                || !sha256_matches(&file, hash)?
            {
                return Ok(false);
            }
        }
        crate::mtp::inspect(directory)?;
        Ok(true)
    };
    if destination.exists() && verified(&destination)? {
        return Ok(destination);
    }
    let downloads = root.join(".downloads");
    ensure!(regular(&downloads), "invalid MTP downloads directory");
    fs::create_dir_all(&downloads)?;
    let stage = downloads.join(format!("mtp-{commit}"));
    ensure!(regular(&stage), "invalid MTP staging directory");
    fs::create_dir_all(&stage)?;
    let lock_path = downloads.join(format!("mtp-{commit}.lock"));
    ensure!(regular(&lock_path), "invalid MTP download lock");
    let lock = OpenOptions::new()
        .create(true)
        .append(true)
        .open(lock_path)?;
    lock.try_lock_exclusive()
        .context("MTP head is already downloading")?;
    let cancelled = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&cancelled))?;
    let total = files.iter().map(|(_, size, _)| size).sum();
    for (name, size, hash) in files {
        let file = stage.join(name);
        ensure!(regular(&file), "invalid MTP staging file");
        if file.metadata().is_ok_and(|m| m.len() == size) && sha256_matches(&file, hash)? {
            continue;
        }
        if file.metadata().is_ok_and(|m| m.len() >= size) {
            fs::remove_file(&file)?;
        }
        download_file(
            &format!("https://huggingface.co/{repo}/resolve/{commit}/{name}?download=true"),
            &file,
            &cancelled,
            || progress(directory_bytes(&stage).min(total), total),
        )?;
        ensure!(
            file.metadata()?.len() == size && sha256_matches(&file, hash)?,
            "MTP head integrity check failed for {name}"
        );
    }
    ensure!(verified(&stage)?, "MTP staging verification failed");
    if destination.exists() {
        fs::rename(
            &destination,
            root.join(format!(".invalid-mtp-{commit}-{}", std::process::id())),
        )?;
    }
    fs::rename(&stage, &destination)?;
    progress(total, total);
    Ok(destination)
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelSummary {
    pub id: String,
    #[serde(default)]
    pub downloads: u64,
    #[serde(default)]
    pub likes: u64,
    #[serde(default)]
    pub gated: Value,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Variant {
    pub id: String,
    pub label: String,
    pub size_bytes: u64,
    pub preview: Option<ModelPreview>,
    #[serde(skip_serializing)]
    pub files: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelPreview {
    pub model_type: Option<String>,
    pub architecture_supported: bool,
    pub maximum_context: Option<u64>,
    pub kv_bytes_per_token: Option<u64>,
    pub recurrent_bytes: Option<u64>,
}

pub fn kv_bytes_per_token(layers: u64, heads: u64, dimension: u64) -> Option<u64> {
    layers
        .checked_mul(heads)?
        .checked_mul(dimension)?
        .checked_mul(4)
}

fn model_preview(config: &Value) -> ModelPreview {
    let text = config.get("text_config").unwrap_or(config);
    let model_type = config["model_type"].as_str().map(str::to_owned);
    let architecture_supported = matches!(
        model_type.as_deref(),
        Some("qwen3_5" | "qwen3_5_moe" | "lfm2" | "lfm2_moe" | "gemma4" | "bailing_hybrid")
    );
    let qwen_cache = || -> Option<(u64, u64)> {
        let kinds = text["layer_types"].as_array()?;
        if kinds.len() as u64 != text["num_hidden_layers"].as_u64()?
            || !kinds
                .iter()
                .all(|kind| matches!(kind.as_str(), Some("full_attention" | "linear_attention")))
        {
            return None;
        }
        let full = kinds
            .iter()
            .filter(|kind| **kind == "full_attention")
            .count() as u64;
        let linear = kinds.len() as u64 - full;
        let kv = kv_bytes_per_token(
            full,
            text["num_key_value_heads"].as_u64()?,
            text["head_dim"].as_u64()?,
        )?;
        if linear == 0 {
            return Some((kv, 0));
        }
        let key_heads = text["linear_num_key_heads"].as_u64()?;
        let value_heads = text["linear_num_value_heads"].as_u64()?;
        let key_dim = text["linear_key_head_dim"].as_u64()?;
        let value_dim = text["linear_value_head_dim"].as_u64()?;
        let recurrence = value_heads
            .checked_mul(key_dim)?
            .checked_mul(value_dim)?
            .checked_mul(4)?;
        let conv_dim = key_heads
            .checked_mul(key_dim)?
            .checked_mul(2)?
            .checked_add(value_heads.checked_mul(value_dim)?)?;
        let conv = text["linear_conv_kernel_dim"]
            .as_u64()?
            .checked_sub(1)?
            .checked_mul(conv_dim)?
            .checked_mul(2)?;
        Some((kv, recurrence.checked_add(conv)?.checked_mul(linear)?))
    };
    let cache = if matches!(model_type.as_deref(), Some("qwen3_5" | "qwen3_5_moe")) {
        qwen_cache()
    } else {
        None
    };
    ModelPreview {
        model_type,
        architecture_supported,
        maximum_context: text["max_position_embeddings"]
            .as_u64()
            .filter(|value| *value > 0),
        kv_bytes_per_token: cache.map(|value| value.0),
        recurrent_bytes: cache.map(|value| value.1),
    }
}

#[derive(Debug, Serialize)]
pub struct Details {
    pub id: String,
    pub revision: String,
    pub commit: String,
    pub branches: Vec<String>,
    pub variants: Vec<Variant>,
    pub readme: String,
    pub gated: bool,
    pub downloads: u64,
    pub likes: u64,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct PendingDownload {
    pub id: String,
    pub repo: String,
    pub commit: String,
    pub folder: String,
    pub name: String,
    pub destination: String,
    pub size_bytes: u64,
    #[serde(default)]
    pub retained_bytes: u64,
}

#[derive(Deserialize)]
struct RepoFile {
    rfilename: String,
    #[serde(default)]
    size: u64,
    lfs: Option<LfsFile>,
}

#[derive(Deserialize)]
struct LfsFile {
    sha256: String,
}

#[derive(Deserialize)]
struct RepoInfo {
    id: String,
    sha: String,
    #[serde(default)]
    gated: Value,
    #[serde(default)]
    downloads: u64,
    #[serde(default)]
    likes: u64,
    #[serde(default)]
    siblings: Vec<RepoFile>,
}

#[derive(Deserialize)]
struct Reference {
    name: String,
}

#[derive(Deserialize)]
struct References {
    #[serde(default)]
    branches: Vec<Reference>,
    #[serde(default)]
    tags: Vec<Reference>,
}

fn encode(value: &str) -> String {
    value.bytes().fold(String::new(), |mut out, byte| {
        if byte.is_ascii_alphanumeric() || b"-._~/".contains(&byte) {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
        out
    })
}

fn token() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .filter(|value| !value.is_empty())
        .or_else(|| {
            crate::registry::expand_home(std::path::Path::new("~/.cache/huggingface/token"))
                .ok()
                .and_then(|path| std::fs::read_to_string(path).ok())
                .map(|value| value.trim().to_owned())
                .filter(|value| !value.is_empty())
        })
}

fn get(url: &str, limit: usize) -> Result<Vec<u8>> {
    let mut command = Command::new("/usr/bin/curl");
    command
        .args(["-fsSL", "--max-time", "30", "--max-filesize"])
        .arg(limit.to_string())
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if token().is_some() {
        command.args(["--config", "-"]).stdin(Stdio::piped());
    }
    let mut child = command
        .spawn()
        .context("starting the system HTTPS client")?;
    if let (Some(token), Some(mut input)) = (token(), child.stdin.take()) {
        ensure!(
            token.starts_with("hf_") && token.bytes().all(|byte| byte.is_ascii_graphic()),
            "invalid Hugging Face token"
        );
        writeln!(input, "header = \"Authorization: Bearer {token}\"")?;
    }
    let output = child.wait_with_output()?;
    if !output.status.success() {
        bail!(
            "Hugging Face request failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    ensure!(
        output.stdout.len() <= limit,
        "Hugging Face response is too large"
    );
    Ok(output.stdout)
}

fn get_json<T: for<'de> Deserialize<'de>>(url: &str, limit: usize) -> Result<T> {
    serde_json::from_slice(&get(url, limit)?).context("invalid Hugging Face response")
}

fn managed_models_path() -> Result<PathBuf> {
    match std::env::var_os("MLXL3_MODELS_DIR").filter(|path| !path.is_empty()) {
        Some(path) => crate::registry::expand_home(Path::new(&path)),
        None => crate::registry::expand_home(Path::new(
            "~/Library/Application Support/io.mlxl3.desktop/Models",
        )),
    }
}

fn managed_drafts_path() -> Result<PathBuf> {
    Ok(managed_models_path()?
        .parent()
        .context("managed models directory has no parent")?
        .join("Drafts"))
}

fn dflash_files(info: &RepoInfo, family: crate::dflash::Family) -> Result<Vec<(&str, u64, &str)>> {
    ensure!(
        info.sha == family.revision(),
        "DFlash draft revision changed"
    );
    let mut result = Vec::with_capacity(7);
    for name in DFLASH_FILES
        .into_iter()
        .filter(|name| family.layers() == 6 || *name != "draft/layer-5.bin")
    {
        let file = info
            .siblings
            .iter()
            .find(|file| file.rfilename == name)
            .with_context(|| format!("DFlash draft file {name} is missing"))?;
        let hash = file
            .lfs
            .as_ref()
            .context("DFlash draft has no LFS checksum")?
            .sha256
            .as_str();
        ensure!(
            file.size > 0 && hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "invalid DFlash draft metadata"
        );
        result.push((file.rfilename.as_str(), file.size, hash));
    }
    Ok(result)
}

fn sha256_matches(path: &Path, expected: &str) -> Result<bool> {
    let output = Command::new("/usr/bin/shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .context("checking DFlash draft checksum")?;
    ensure!(output.status.success(), "DFlash checksum check failed");
    Ok(output
        .stdout
        .get(..64)
        .is_some_and(|hash| hash.eq_ignore_ascii_case(expected.as_bytes())))
}

/// Download only the six or seven pinned draft files, excluding Splash's target.
/// Partial files stay in a managed staging directory for a later retry.
pub fn download_dflash(
    family: crate::dflash::Family,
    mut progress: impl FnMut(u64, u64),
) -> Result<PathBuf> {
    let root = managed_drafts_path()?;
    fs::create_dir_all(&root)?;
    let root = root.canonicalize()?;
    let destination = root.join(family.name());
    ensure!(
        !destination
            .symlink_metadata()
            .is_ok_and(|meta| meta.file_type().is_symlink()),
        "invalid DFlash destination"
    );
    let files = pinned_dflash_files(family);
    if destination.exists() {
        for (name, size, hash) in files {
            let file = destination.join(name);
            ensure!(
                !file.symlink_metadata()?.file_type().is_symlink()
                    && file.metadata()?.is_file()
                    && file.metadata()?.len() == *size
                    && sha256_matches(&file, hash)?,
                "DFlash installed draft checksum or size mismatch for {name}"
            );
        }
        ensure!(
            crate::dflash::inspect(&destination)?.family == family,
            "DFlash destination belongs to another target"
        );
        return Ok(destination);
    }
    let stage = root.join(".downloads").join(family.revision());
    ensure!(
        !stage
            .symlink_metadata()
            .is_ok_and(|meta| meta.file_type().is_symlink()),
        "invalid DFlash staging directory"
    );
    fs::create_dir_all(&stage)?;
    ensure!(
        !stage
            .join("draft")
            .symlink_metadata()
            .is_ok_and(|meta| meta.file_type().is_symlink()),
        "invalid DFlash draft staging directory"
    );
    let lock = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join(".downloads").join("dflash.lock"))?;
    lock.try_lock_exclusive()
        .context("DFlash draft is already downloading")?;
    let metadata = info(family.repository(), family.revision())?;
    let published = dflash_files(&metadata, family)?;
    ensure!(published == files, "DFlash pinned checksums changed");
    let total: u64 = files.iter().map(|(_, size, _)| size).sum();
    let cancelled = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&cancelled))?;
    progress(directory_bytes(&stage).min(total), total);
    for (name, expected_size, expected_hash) in files {
        let target = stage.join(name);
        ensure!(
            !target
                .symlink_metadata()
                .is_ok_and(|meta| meta.file_type().is_symlink()),
            "invalid DFlash staging file"
        );
        if target
            .metadata()
            .is_ok_and(|meta| meta.len() == *expected_size)
            && sha256_matches(&target, expected_hash)?
        {
            continue;
        }
        // A complete but corrupt file cannot be resumed safely.
        if target
            .metadata()
            .is_ok_and(|meta| meta.len() >= *expected_size)
        {
            fs::remove_file(&target)?;
        }
        download_file(
            &format!(
                "https://huggingface.co/{}/resolve/{}/{}?download=true",
                encode(family.repository()),
                family.revision(),
                name
            ),
            &target,
            &cancelled,
            || progress(directory_bytes(&stage).min(total), total),
        )?;
        ensure!(
            target.metadata()?.len() == *expected_size && sha256_matches(&target, expected_hash)?,
            "DFlash draft checksum or size mismatch for {name}"
        );
    }
    ensure!(
        crate::dflash::inspect(&stage)?.family == family,
        "DFlash staged package belongs to another target"
    );
    fs::rename(&stage, &destination)?;
    progress(total, total);
    Ok(destination)
}

fn directory_bytes(path: &Path) -> u64 {
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                directory_bytes(&path)
            } else {
                entry.metadata().map_or(0, |metadata| metadata.len())
            }
        })
        .sum()
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let parent = path.parent().context("JSON path has no parent")?;
    fs::create_dir_all(parent)?;
    let mut temporary = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temporary, value)?;
    temporary.write_all(b"\n")?;
    temporary.as_file().sync_all()?;
    temporary.persist(path).context("saving download state")?;
    Ok(())
}

fn download_id(repo: &str, commit: &str, folder: &str) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in repo
        .bytes()
        .chain(b"\n".iter().copied())
        .chain(commit.bytes())
        .chain(b"\n".iter().copied())
        .chain(folder.bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")[..12].to_owned()
}

fn local_name(repo: &str, folder: &str, id: &str) -> String {
    let source = format!(
        "{}{}-{}",
        repo.rsplit('/').next().unwrap_or("model"),
        if folder.is_empty() { "" } else { "-" },
        if folder.is_empty() { &id[..8] } else { folder }
    );
    let mut result = String::with_capacity(source.len());
    for character in source.chars().take(100) {
        result.push(
            if character.is_ascii_alphanumeric() || "._-".contains(character) {
                character
            } else {
                '-'
            },
        );
    }
    result
}

fn download_file(
    url: &str,
    destination: &Path,
    cancelled: &AtomicBool,
    mut report: impl FnMut(),
) -> Result<()> {
    let parent = destination
        .parent()
        .context("download path has no parent")?;
    fs::create_dir_all(parent)?;
    let mut command = Command::new("/usr/bin/curl");
    command
        .args([
            "-fsSL",
            "--retry",
            "3",
            "--retry-all-errors",
            "--continue-at",
            "-",
            "--output",
        ])
        .arg(destination)
        .arg(url)
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    if token().is_some() {
        command.args(["--config", "-"]).stdin(Stdio::piped());
    }
    let mut child = command.spawn().context("starting model download")?;
    if let (Some(token), Some(mut input)) = (token(), child.stdin.take()) {
        ensure!(
            token.starts_with("hf_") && token.bytes().all(|byte| byte.is_ascii_graphic()),
            "invalid Hugging Face token"
        );
        writeln!(input, "header = \"Authorization: Bearer {token}\"")?;
    }
    loop {
        if cancelled.load(Ordering::Relaxed) {
            child.kill()?;
            let _ = child.wait();
            bail!("download cancelled");
        }
        if let Some(status) = child.try_wait()? {
            if status.success() {
                report();
                return Ok(());
            }
            let mut error = String::new();
            if let Some(stderr) = child.stderr.take() {
                use std::io::Read;
                let _ = stderr.take(8_000).read_to_string(&mut error);
            }
            bail!("model download failed: {}", error.trim());
        }
        report();
        thread::sleep(Duration::from_millis(150));
    }
}

fn valid_repo(repo: &str) -> bool {
    valid_repo_bytes(repo.as_bytes())
}

fn valid_repo_bytes(repo: &[u8]) -> bool {
    let mut slash = None;
    for (index, &byte) in repo.iter().enumerate() {
        if !(byte.is_ascii_alphanumeric() || b"-._/".contains(&byte)) {
            return false;
        }
        if byte == b'/' && slash.replace(index).is_some() {
            return false;
        }
    }
    slash.is_some_and(|index| index > 0 && index + 1 < repo.len())
}

pub fn search(query: &str, limit: usize) -> Result<Vec<ModelSummary>> {
    let query = query.trim();
    ensure!(query.len() <= 160, "search query is too long");
    let terms: Vec<_> = query
        .to_ascii_lowercase()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let needle = terms
        .iter()
        .filter(|term| term.as_str() != "exl3")
        .max_by_key(|term| term.len())
        .map(String::as_str)
        .unwrap_or("exl3");
    let requested = limit.clamp(1, 600);
    let url = format!(
        "{API}/models?search={}&filter=exl3&sort=downloads&direction=-1&limit={}",
        encode(needle),
        (requested * 3).min(600)
    );
    let mut found: Vec<ModelSummary> = get_json(&url, 8 * 1024 * 1024)?;
    found.retain(|model| {
        let id = model.id.to_ascii_lowercase();
        (id.contains("exl3")
            || model
                .tags
                .iter()
                .any(|tag| tag.eq_ignore_ascii_case("exl3")))
            && terms.iter().all(|term| id.contains(term))
    });
    found.sort_by(|left, right| {
        right
            .downloads
            .cmp(&left.downloads)
            .then(left.id.cmp(&right.id))
    });
    found.truncate(requested);
    Ok(found)
}

fn descriptor_label(name: &str) -> Option<Option<String>> {
    let base = name.rsplit('/').next()?;
    if base == "quantization_config.json" {
        return Some(None);
    }
    let label = base
        .strip_prefix("quantization_config")?
        .strip_suffix(".json")?
        .trim_start_matches(['.', '_', '-']);
    (!label.is_empty()).then(|| Some(label.to_owned()))
}

fn safe_file(name: &str) -> bool {
    safe_file_bytes(name.as_bytes())
}

fn safe_file_bytes(name: &[u8]) -> bool {
    if name.is_empty() || name[0] == b'/' || name.contains(&b'\\') {
        return false;
    }
    let mut start = 0;
    for end in 0..=name.len() {
        if end == name.len() || name[end] == b'/' {
            let part = &name[start..end];
            if part.is_empty() || part == b"." || part == b".." {
                return false;
            }
            start = end + 1;
        }
    }
    true
}

fn parent(name: &str) -> &str {
    name.rsplit_once('/').map_or(".", |(folder, _)| folder)
}

fn base(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}

fn belongs(name: &str, label: &str) -> bool {
    let stem = name.rsplit_once('.').map_or(name, |(stem, _)| stem);
    stem.match_indices(label).any(|(start, value)| {
        let end = start + value.len();
        let boundary = |byte: Option<&u8>| {
            byte.is_none_or(|byte| !byte.is_ascii_alphanumeric() && *byte != b'.')
        };
        boundary(stem.as_bytes().get(start.wrapping_sub(1))) && boundary(stem.as_bytes().get(end))
    })
}

pub fn variants(files: &BTreeMap<String, u64>) -> Vec<Variant> {
    let files: BTreeMap<_, _> = files
        .iter()
        .filter(|(name, _)| safe_file(name))
        .map(|(name, size)| (name.clone(), *size))
        .collect();
    let descriptors: Vec<_> = files
        .keys()
        .filter_map(|name| descriptor_label(name).map(|label| (name, label)))
        .collect();
    let mut result = Vec::new();
    for (descriptor, quant_label) in &descriptors {
        let folder = parent(descriptor);
        let mut weights: Vec<_> = files
            .keys()
            .filter(|name| {
                parent(name) == folder
                    && base(name).starts_with("model")
                    && name.ends_with(".safetensors")
            })
            .filter(|name| {
                quant_label
                    .as_ref()
                    .is_none_or(|label| belongs(name, label))
            })
            .cloned()
            .collect();
        if quant_label.is_none() {
            let other: Vec<_> = descriptors
                .iter()
                .filter(|(name, label)| parent(name) == folder && label.is_some())
                .filter_map(|(_, label)| label.as_deref())
                .collect();
            weights.retain(|name| !other.iter().any(|label| belongs(name, label)));
        }
        if weights.is_empty() {
            continue;
        }
        let mut selected = BTreeMap::new();
        let mut ancestors = vec!["."];
        if folder != "." {
            for (index, byte) in folder.bytes().enumerate() {
                if byte == b'/' {
                    ancestors.push(&folder[..index]);
                }
            }
            ancestors.push(folder);
        }
        for ancestor in ancestors {
            for name in files.keys().filter(|name| parent(name) == ancestor) {
                let base = base(name);
                if matches!(
                    base,
                    "config.json"
                        | "generation_config.json"
                        | "merges.txt"
                        | "vocab.json"
                        | "vocab.txt"
                        | "special_tokens_map.json"
                ) || base.starts_with("tokenizer")
                    || base.starts_with("chat_template")
                    || base.starts_with("preprocessor")
                    || base.starts_with("processor")
                    || base.starts_with("LICENSE")
                    || base.starts_with("README")
                    || base.ends_with(".tiktoken")
                    || base.ends_with(".model")
                {
                    selected.insert(base.to_owned(), name.clone());
                }
            }
        }
        for name in files.keys().filter(|name| parent(name) == folder) {
            let file = base(name);
            if weights.contains(name)
                || matches!(
                    file,
                    "ngram_embedding.safetensors" | "mtp_hyper_connection_mixer_patch.safetensors"
                )
                || (quant_label.is_none() && file == "model.safetensors.index.json")
            {
                selected.insert(file.to_owned(), name.clone());
            }
            if let Some(label) = quant_label.as_deref()
                && file.starts_with("config")
                && file.ends_with(".json")
                && belongs(file, label)
            {
                selected.insert("config.json".into(), name.clone());
            }
        }
        selected.insert("quantization_config.json".into(), (*descriptor).clone());
        if !selected.contains_key("config.json") {
            continue;
        }
        let identity = if folder == "." {
            String::new()
        } else {
            folder.to_owned()
        };
        let id = match quant_label {
            Some(_) => format!("{identity}#{}", base(descriptor)),
            None => identity,
        };
        result.push(Variant {
            id,
            label: quant_label.clone().unwrap_or_else(|| {
                if folder == "." {
                    "Root".into()
                } else {
                    folder.into()
                }
            }),
            size_bytes: selected.values().filter_map(|name| files.get(name)).sum(),
            files: selected,
            preview: None,
        });
    }
    result
}

fn info(repo: &str, revision: &str) -> Result<RepoInfo> {
    ensure!(valid_repo(repo), "invalid Hugging Face repository");
    get_json(
        &format!(
            "{API}/models/{}/revision/{}?blobs=true",
            encode(repo),
            encode(revision)
        ),
        32 * 1024 * 1024,
    )
}

fn is_gated(value: &Value) -> bool {
    value
        .as_bool()
        .unwrap_or_else(|| value.as_str().is_some_and(|value| value != "false"))
}

pub fn details(repo: &str, revision: Option<&str>) -> Result<Details> {
    ensure!(valid_repo(repo), "invalid Hugging Face repository");
    let refs: References = get_json(&format!("{API}/models/{}/refs", encode(repo)), 1024 * 1024)?;
    let branches: Vec<_> = refs
        .branches
        .into_iter()
        .chain(refs.tags)
        .map(|item| item.name)
        .collect();
    let mut selected = revision.unwrap_or("main").to_owned();
    let mut model = info(repo, &selected)?;
    let mut files: BTreeMap<_, _> = model
        .siblings
        .iter()
        .map(|file| (file.rfilename.clone(), file.size))
        .collect();
    let mut choices = variants(&files);
    if choices.is_empty()
        && revision.is_none()
        && let Some(candidate) = branches
            .iter()
            .find(|name| name.to_ascii_lowercase().contains("bpw"))
    {
        selected = candidate.clone();
        model = info(repo, &selected)?;
        files = model
            .siblings
            .iter()
            .map(|file| (file.rfilename.clone(), file.size))
            .collect();
        choices = variants(&files);
    }
    let mut previews = BTreeMap::new();
    for variant in choices.iter_mut().take(8) {
        if let Some(path) = variant.files.get("config.json") {
            if !previews.contains_key(path) {
                let preview = get_json::<Value>(
                    &format!(
                        "https://huggingface.co/{}/resolve/{}/{}",
                        encode(repo),
                        encode(&model.sha),
                        encode(path)
                    ),
                    1024 * 1024,
                )
                .ok()
                .map(|config| model_preview(&config));
                previews.insert(path.clone(), preview);
            }
            variant.preview = previews.get(path).cloned().flatten();
        }
    }
    let readme = if files
        .get("README.md")
        .is_some_and(|size| *size > 0 && *size <= 256_000)
    {
        String::from_utf8_lossy(&get(
            &format!(
                "https://huggingface.co/{}/resolve/{}/README.md",
                encode(repo),
                encode(&model.sha)
            ),
            256_000,
        )?)
        .into_owned()
    } else {
        String::new()
    };
    Ok(Details {
        id: model.id,
        revision: selected,
        commit: model.sha,
        branches,
        variants: choices,
        readme,
        gated: is_gated(&model.gated),
        downloads: model.downloads,
        likes: model.likes,
    })
}

fn run_download(
    request: &PendingDownload,
    registry_path: &Path,
    mut progress: impl FnMut(u64, u64),
) -> Result<(String, crate::registry::ModelEntry)> {
    let root = managed_models_path()?;
    fs::create_dir_all(&root)?;
    let root = root.canonicalize()?;
    let stage = root.join(".downloads").join(&request.id);
    fs::create_dir_all(&stage)?;
    let lock_path = stage
        .parent()
        .context("download stage has no parent")?
        .join(format!("{}.lock", request.id));
    let lock = OpenOptions::new()
        .create(true)
        .append(true)
        .open(lock_path)?;
    lock.try_lock_exclusive()
        .context("this variant is already downloading")?;
    write_json(&stage.join("request.json"), request)?;

    let detail = details(&request.repo, Some(&request.commit))?;
    ensure!(detail.commit == request.commit, "download revision changed");
    let selected = detail
        .variants
        .into_iter()
        .find(|variant| variant.id == request.folder)
        .context("selected EXL3 variant no longer exists")?;
    ensure!(
        selected.size_bytes == request.size_bytes,
        "selected EXL3 variant changed size"
    );
    let sizes: BTreeMap<_, _> = info(&request.repo, &request.commit)?
        .siblings
        .into_iter()
        .map(|file| (file.rfilename, file.size))
        .collect();
    let snapshot = stage.join("snapshot");
    let cancelled = Arc::new(AtomicBool::new(false));
    signal_hook::flag::register(signal_hook::consts::SIGTERM, Arc::clone(&cancelled))?;
    progress(
        directory_bytes(&snapshot).min(request.size_bytes),
        request.size_bytes,
    );
    for (local, remote) in &selected.files {
        ensure!(
            safe_file(remote) && !local.contains('/'),
            "unsafe checkpoint file path"
        );
        let target = snapshot.join(remote);
        let expected = *sizes
            .get(remote)
            .context("selected checkpoint file disappeared")?;
        if target
            .metadata()
            .is_ok_and(|metadata| metadata.len() == expected)
        {
            continue;
        }
        download_file(
            &format!(
                "https://huggingface.co/{}/resolve/{}/{}?download=true",
                encode(&request.repo),
                encode(&request.commit),
                encode(remote)
            ),
            &target,
            &cancelled,
            || {
                progress(
                    directory_bytes(&snapshot).min(request.size_bytes),
                    request.size_bytes,
                )
            },
        )?;
        ensure!(
            target.metadata()?.len() == expected,
            "downloaded file has the wrong size"
        );
    }

    let assembled = stage.join("model");
    if assembled.exists() {
        fs::remove_dir_all(&assembled)?;
    }
    fs::create_dir(&assembled)?;
    for (local, remote) in &selected.files {
        let source = snapshot.join(remote);
        let target = assembled.join(local);
        fs::hard_link(&source, &target).or_else(|_| fs::copy(&source, &target).map(|_| ()))?;
    }
    crate::checkpoint::inspect(&assembled)?;
    let destination = PathBuf::from(&request.destination);
    ensure!(!destination.exists(), "model destination already exists");
    ensure!(
        destination.parent() == Some(root.as_path()),
        "model destination escaped managed storage"
    );
    fs::rename(&assembled, &destination)?;
    let entry = match crate::registry::register(registry_path, &request.name, &destination, false) {
        Ok(entry) => entry,
        Err(error) => {
            fs::rename(&destination, &assembled)?;
            return Err(error);
        }
    };
    fs::remove_dir_all(&stage)?;
    progress(request.size_bytes, request.size_bytes);
    Ok((request.name.clone(), entry))
}

pub fn download(
    repo: &str,
    commit: &str,
    folder: &str,
    registry_path: &Path,
    progress: impl FnMut(u64, u64),
) -> Result<(String, crate::registry::ModelEntry)> {
    ensure!(
        commit.len() == 40 && commit.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "select a resolved revision before downloading"
    );
    let detail = details(repo, Some(commit))?;
    let selected = detail
        .variants
        .iter()
        .find(|variant| variant.id == folder)
        .context("no complete EXL3 checkpoint in the selected folder")?;
    let id = download_id(repo, commit, folder);
    let root = managed_models_path()?;
    fs::create_dir_all(&root)?;
    let root = root.canonicalize()?;
    let name = local_name(repo, folder, &id);
    let request = PendingDownload {
        id,
        repo: repo.to_owned(),
        commit: commit.to_owned(),
        folder: folder.to_owned(),
        destination: root.join(&name).to_string_lossy().into_owned(),
        name,
        size_bytes: selected.size_bytes,
        retained_bytes: 0,
    };
    run_download(&request, registry_path, progress)
}

pub fn pending_downloads() -> Result<Vec<PendingDownload>> {
    let downloads = managed_models_path()?.join(".downloads");
    let Ok(entries) = fs::read_dir(downloads) else {
        return Ok(Vec::new());
    };
    let mut result = Vec::new();
    for entry in entries.flatten().filter(|entry| entry.path().is_dir()) {
        let path = entry.path();
        let Ok(file) = File::open(path.join("request.json")) else {
            continue;
        };
        let Ok(mut request) = serde_json::from_reader::<_, PendingDownload>(file) else {
            continue;
        };
        request.retained_bytes = directory_bytes(&path.join("snapshot"));
        result.push(request);
    }
    result.sort_by(|left, right| left.repo.cmp(&right.repo));
    Ok(result)
}

pub fn resume(
    id: &str,
    registry_path: &Path,
    progress: impl FnMut(u64, u64),
) -> Result<(String, crate::registry::ModelEntry)> {
    let request = pending_downloads()?
        .into_iter()
        .find(|request| request.id == id)
        .context("interrupted download not found")?;
    run_download(&request, registry_path, progress)
}

pub fn discard(id: &str) -> Result<()> {
    ensure!(
        id.len() == 12 && id.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "invalid download identifier"
    );
    let root = managed_models_path()?.join(".downloads");
    fs::create_dir_all(&root)?;
    let stage = root.join(id);
    ensure!(
        !stage
            .symlink_metadata()
            .is_ok_and(|metadata| metadata.file_type().is_symlink()),
        "invalid download directory"
    );
    let lock = OpenOptions::new()
        .create(true)
        .append(true)
        .open(root.join(format!("{id}.lock")))?;
    lock.try_lock_exclusive()
        .context("pause the download before removing its partial files")?;
    if stage.is_dir() {
        fs::remove_dir_all(stage)?;
    }
    Ok(())
}

fn token_path() -> Result<PathBuf> {
    crate::registry::expand_home(Path::new("~/.cache/huggingface/token"))
}

pub fn login(value: &str) -> Result<()> {
    ensure!(
        value.starts_with("hf_") && value.bytes().all(|byte| byte.is_ascii_graphic()),
        "invalid Hugging Face token"
    );
    let path = token_path()?;
    fs::create_dir_all(path.parent().context("token path has no parent")?)?;
    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .mode(0o600)
        .open(&path)?;
    writeln!(file, "{value}")?;
    file.sync_all()?;
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))?;
    Ok(())
}

pub fn logout() -> Result<()> {
    match fs::remove_file(token_path()?) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error.into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_reports_architecture_and_keeps_hybrid_cache_unknown() {
        let hybrid = model_preview(
            &serde_json::json!({"model_type":"qwen3_5_moe","text_config":{"max_position_embeddings":32768,"num_hidden_layers":40,"num_key_value_heads":4,"head_dim":128}}),
        );
        assert!(hybrid.architecture_supported);
        assert_eq!(hybrid.maximum_context, Some(32768));
        assert_eq!(hybrid.kv_bytes_per_token, None);
        let unknown = model_preview(&serde_json::json!({"model_type":"unknown"}));
        assert!(!unknown.architecture_supported);
        assert_eq!(unknown.kv_bytes_per_token, None);
        let dimensions = serde_json::json!({"model_type":"qwen3_5","text_config":{"num_hidden_layers":2,"layer_types":["full_attention","linear_attention"],"num_key_value_heads":2,"head_dim":128,"linear_num_key_heads":2,"linear_num_value_heads":4,"linear_key_head_dim":8,"linear_value_head_dim":16,"linear_conv_kernel_dim":4}});
        let preview = model_preview(&dimensions);
        assert_eq!(preview.kv_bytes_per_token, Some(1024));
        assert_eq!(preview.recurrent_bytes, Some(2624));
        assert_eq!(kv_bytes_per_token(4, 2, 128), Some(4096));
        assert_eq!(kv_bytes_per_token(u64::MAX, 2, 128), None);
    }

    #[test]
    fn dflash_download_selects_only_pinned_draft_files() {
        let mut siblings: Vec<RepoFile> = DFLASH_FILES
            .iter()
            .map(|name| RepoFile {
                rfilename: (*name).into(),
                size: 12,
                lfs: Some(LfsFile {
                    sha256: "a".repeat(64),
                }),
            })
            .collect();
        siblings.push(RepoFile {
            rfilename: "target/model.bin".into(),
            size: 999_999,
            lfs: Some(LfsFile {
                sha256: "b".repeat(64),
            }),
        });
        let mut info = RepoInfo {
            id: DFLASH_REPO.into(),
            sha: DFLASH_COMMIT.into(),
            gated: Value::Null,
            downloads: 0,
            likes: 0,
            siblings,
        };
        let selected = dflash_files(&info, crate::dflash::Family::Qwen36Moe).unwrap();
        assert_eq!(selected.len(), 7);
        assert_eq!(selected.iter().map(|(_, size, _)| size).sum::<u64>(), 84);
        assert!(
            selected
                .iter()
                .all(|(name, _, _)| name.starts_with("draft/"))
        );
        info.siblings.pop();
        info.siblings.pop();
        assert!(dflash_files(&info, crate::dflash::Family::Qwen36Moe).is_err());
        info.sha = "wrong-revision".into();
        assert!(dflash_files(&info, crate::dflash::Family::Qwen36Moe).is_err());
    }

    #[test]
    fn both_dflash_downloads_match_pinned_geometry_and_reject_bad_metadata() {
        for (family, bytes) in [
            (crate::dflash::Family::Qwen36Moe, 479_444_992),
            (crate::dflash::Family::Qwen38Dense, 1_266_040_832),
        ] {
            let mut info = RepoInfo {
                id: family.repository().into(),
                sha: family.revision().into(),
                gated: Value::Null,
                downloads: 0,
                likes: 0,
                siblings: pinned_dflash_files(family)
                    .iter()
                    .map(|(name, size, hash)| RepoFile {
                        rfilename: (*name).into(),
                        size: *size,
                        lfs: Some(LfsFile {
                            sha256: (*hash).into(),
                        }),
                    })
                    .collect(),
            };
            assert_eq!(
                dflash_files(&info, family).unwrap(),
                pinned_dflash_files(family)
            );
            assert_eq!(info.siblings.len(), family.layers() + 1);
            assert_eq!(info.siblings.iter().map(|f| f.size).sum::<u64>(), bytes);
            info.siblings[0].lfs = None;
            assert!(dflash_files(&info, family).is_err());
            info.siblings[0].lfs = Some(LfsFile {
                sha256: "g".repeat(64),
            });
            assert!(dflash_files(&info, family).is_err());
            info.siblings.pop();
            assert!(dflash_files(&info, family).is_err());
        }
    }

    #[test]
    fn flash_variants_include_present_sidecars_and_exact_total() {
        for present in 0u8..4 {
            let mut files = BTreeMap::from([
                ("config.json".into(), 10),
                ("tokenizer.json".into(), 20),
                ("quantization_config.json".into(), 5),
                ("model.safetensors".into(), 100),
                ("model.safetensors.index.json".into(), 7),
            ]);
            if present & 1 != 0 {
                files.insert("ngram_embedding.safetensors".into(), 26_240_152_672);
            }
            if present & 2 != 0 {
                files.insert(
                    "mtp_hyper_connection_mixer_patch.safetensors".into(),
                    13_128_048,
                );
            }
            let choices = variants(&files);
            assert_eq!(choices.len(), 1);
            let choice = &choices[0];
            let mut expected = BTreeMap::from([
                ("config.json".into(), "config.json".into()),
                ("tokenizer.json".into(), "tokenizer.json".into()),
                (
                    "quantization_config.json".into(),
                    "quantization_config.json".into(),
                ),
                ("model.safetensors".into(), "model.safetensors".into()),
                (
                    "model.safetensors.index.json".into(),
                    "model.safetensors.index.json".into(),
                ),
            ]);
            let mut bytes = 142;
            if present & 1 != 0 {
                expected.insert(
                    "ngram_embedding.safetensors".into(),
                    "ngram_embedding.safetensors".into(),
                );
                bytes += 26_240_152_672;
            }
            if present & 2 != 0 {
                expected.insert(
                    "mtp_hyper_connection_mixer_patch.safetensors".into(),
                    "mtp_hyper_connection_mixer_patch.safetensors".into(),
                );
                bytes += 13_128_048;
            }
            assert_eq!(choice.files, expected);
            assert_eq!(choice.size_bytes, bytes);
            files.remove("config.json");
            assert!(variants(&files).is_empty());
            files.insert("config.json".into(), 10);
            files.remove("model.safetensors");
            assert!(
                variants(&files).is_empty(),
                "sidecars cannot replace the main weights"
            );
        }
    }

    #[test]
    fn flash_sidecars_stay_in_the_variant_folder_and_do_not_match_quantized_names() {
        let files = BTreeMap::from([
            ("config.json".into(), 10),
            ("tokenizer.json".into(), 20),
            ("quantization_config.json".into(), 5),
            ("model.safetensors".into(), 100),
            ("ngram_embedding.safetensors".into(), 1_000),
            ("mtp_hyper_connection_mixer_patch.safetensors".into(), 2_000),
            ("sub/config.json".into(), 11),
            ("sub/quantization_config.json".into(), 6),
            ("sub/model.safetensors".into(), 200),
            ("sub/model.safetensors.index.json".into(), 13),
            ("sub/quantization_config-4bpw.json".into(), 7),
            ("sub/config-4bpw.json".into(), 8),
            ("sub/model-4bpw.safetensors".into(), 400),
            ("sub/ngram_embedding.safetensors".into(), 3_000),
            (
                "sub/mtp_hyper_connection_mixer_patch.safetensors".into(),
                4_000,
            ),
            ("sub/ngram_embedding-4bpw.safetensors".into(), 9_000),
            (
                "sub/mtp_hyper_connection_mixer_patch-4bpw.safetensors".into(),
                9_000,
            ),
            ("sub/other.safetensors".into(), 9_000),
            ("sub/draft/ngram_embedding.safetensors".into(), 9_000),
            (
                "other/mtp_hyper_connection_mixer_patch.safetensors".into(),
                9_000,
            ),
            ("../quantization_config.json".into(), 9_000),
            ("../config.json".into(), 9_000),
            ("../model.safetensors".into(), 9_000),
            ("sub/../ngram_embedding.safetensors".into(), 9_000),
            (
                "sub/./mtp_hyper_connection_mixer_patch.safetensors".into(),
                9_000,
            ),
            ("/ngram_embedding.safetensors".into(), 9_000),
            (
                "sub\\mtp_hyper_connection_mixer_patch.safetensors".into(),
                9_000,
            ),
        ]);
        let choices = variants(&files);
        assert_eq!(choices.len(), 3);
        for (id, folder, size, model) in [
            ("", "", 3_135, "model.safetensors"),
            ("sub", "sub/", 7_250, "model.safetensors"),
            (
                "sub#quantization_config-4bpw.json",
                "sub/",
                7_435,
                "model-4bpw.safetensors",
            ),
        ] {
            let choice = choices.iter().find(|choice| choice.id == id).unwrap();
            assert_eq!(choice.size_bytes, size);
            for name in [
                "ngram_embedding.safetensors",
                "mtp_hyper_connection_mixer_patch.safetensors",
            ] {
                assert_eq!(choice.files[name], format!("{folder}{name}"));
            }
            let actual_weights: Vec<_> = choice
                .files
                .keys()
                .filter(|name| name.ends_with(".safetensors"))
                .map(String::as_str)
                .collect();
            let mut expected = vec![
                model,
                "ngram_embedding.safetensors",
                "mtp_hyper_connection_mixer_patch.safetensors",
            ];
            expected.sort_unstable();
            assert_eq!(actual_weights, expected);
            assert!(choice.files.values().all(|name| safe_file(name)));
        }
    }

    #[test]
    fn selected_flash_variant_assembles_into_an_inspectable_checkpoint() {
        fn shard(path: &Path, header: Value, payload: &[u8]) {
            let header = serde_json::to_vec(&header).unwrap();
            let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
            bytes.extend(header);
            bytes.extend(payload);
            fs::write(path, bytes).unwrap();
        }
        let root = tempfile::tempdir().unwrap();
        let source = root.path().join("source");
        let assembled = root.path().join("assembled");
        fs::create_dir(&source).unwrap();
        fs::create_dir(&assembled).unwrap();
        fs::write(source.join("config.json"), r#"{"model_type":"qwen4_exp"}"#).unwrap();
        fs::write(source.join("tokenizer.json"), "{}").unwrap();
        fs::write(source.join("quantization_config.json"), serde_json::to_vec(&serde_json::json!({
            "quant_method":"exl3", "bits":2.05,
            "tensor_storage":{
                "model.linear":{"quant_format":"exl3"},
                "sidecars":{"stored_tensors":{"ple.sidecar":{"shape":[1]},"mtp.sidecar":{"shape":[1]}}}
            }
        })).unwrap()).unwrap();
        shard(
            &source.join("model.safetensors"),
            serde_json::json!({
                "model.linear.trellis":{"dtype":"U16","shape":[1,1,16],"data_offsets":[0,32]},
                "model.linear.suh":{"dtype":"F16","shape":[16],"data_offsets":[32,64]},
                "model.linear.svh":{"dtype":"F16","shape":[16],"data_offsets":[64,96]}
            }),
            &[0; 96],
        );
        for (name, tensor) in [
            ("ngram_embedding.safetensors", "ple.sidecar"),
            (
                "mtp_hyper_connection_mixer_patch.safetensors",
                "mtp.sidecar",
            ),
        ] {
            shard(
                &source.join(name),
                serde_json::json!({tensor:{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}),
                &[0; 2],
            );
        }
        let files = fs::read_dir(&source)
            .unwrap()
            .map(|entry| {
                let entry = entry.unwrap();
                (
                    entry.file_name().into_string().unwrap(),
                    entry.metadata().unwrap().len(),
                )
            })
            .collect();
        let choices = variants(&files);
        assert_eq!(choices.len(), 1);
        for (local, remote) in &choices[0].files {
            fs::copy(source.join(remote), assembled.join(local)).unwrap();
        }
        let checkpoint = crate::checkpoint::inspect(&assembled).unwrap();
        assert_eq!(checkpoint.model_type, "qwen4_exp");
        assert_eq!(checkpoint.tensors.len(), 5);
        for tensor in ["ple.sidecar", "mtp.sidecar"] {
            assert_eq!(checkpoint.tensors[tensor].read_bytes().unwrap(), [0, 0]);
        }
    }

    #[test]
    fn discovers_root_and_named_quantizations() {
        let files = BTreeMap::from([
            ("config.json".into(), 10),
            ("tokenizer.json".into(), 20),
            ("quantization_config.json".into(), 5),
            ("model.safetensors".into(), 100),
            ("quantization_config-4bpw.json".into(), 6),
            ("model-4bpw.safetensors".into(), 200),
        ]);
        let choices = variants(&files);
        assert_eq!(choices.len(), 2);
        let root = choices.iter().find(|choice| choice.id.is_empty()).unwrap();
        let named = choices.iter().find(|choice| !choice.id.is_empty()).unwrap();
        assert_eq!(root.size_bytes, 135);
        assert_eq!(named.id, "#quantization_config-4bpw.json");
        assert_eq!(named.size_bytes, 236);
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    #[kani::unwind(130)]
    fn flash_sidecar_collection_preserves_present_files_and_size() {
        let ngram: bool = kani::any();
        let mtp: bool = kani::any();
        let mut files = BTreeMap::from([
            ("config.json".into(), 10),
            ("quantization_config.json".into(), 5),
            ("model.safetensors".into(), 100),
        ]);
        if ngram {
            files.insert("ngram_embedding.safetensors".into(), 26_240_152_672);
        }
        if mtp {
            files.insert(
                "mtp_hyper_connection_mixer_patch.safetensors".into(),
                13_128_048,
            );
        }
        let choices = variants(&files);
        assert_eq!(choices.len(), 1);
        let choice = &choices[0];
        assert_eq!(
            choice.files.contains_key("ngram_embedding.safetensors"),
            ngram
        );
        assert_eq!(
            choice
                .files
                .contains_key("mtp_hyper_connection_mixer_patch.safetensors"),
            mtp
        );
        assert_eq!(
            choice.files.len(),
            3 + usize::from(ngram) + usize::from(mtp)
        );
        assert_eq!(
            choice.size_bytes,
            115 + u64::from(ngram) * 26_240_152_672 + u64::from(mtp) * 13_128_048
        );
        kani::cover!(ngram && mtp);
        kani::cover!(!ngram && !mtp);
    }

    #[kani::proof]
    #[kani::unwind(10)]
    fn repository_names_have_one_nonempty_safe_separator() {
        let bytes: [u8; 8] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= bytes.len());
        let name = &bytes[..len];
        let expected = !name.is_empty()
            && name.iter().filter(|&&byte| byte == b'/').count() == 1
            && name[0] != b'/'
            && name[name.len() - 1] != b'/'
            && name
                .iter()
                .all(|byte| byte.is_ascii_alphanumeric() || b"-._/".contains(byte));
        assert_eq!(valid_repo_bytes(name), expected);
        kani::cover!(valid_repo_bytes(name));
        kani::cover!(!valid_repo_bytes(name));
    }

    #[kani::proof]
    #[kani::unwind(12)]
    fn accepted_checkpoint_paths_have_no_traversal_component() {
        let bytes: [u8; 8] = kani::any();
        let len: usize = kani::any();
        kani::assume(len <= bytes.len());
        let name = &bytes[..len];
        if safe_file_bytes(name) {
            assert!(!name.is_empty());
            assert_ne!(name[0], b'/');
            assert!(!name.contains(&b'\\'));
            assert!(
                name.split(|&byte| byte == b'/')
                    .all(|part| !part.is_empty() && part != b"." && part != b"..")
            );
        }
        kani::cover!(safe_file_bytes(name));
        kani::cover!(!safe_file_bytes(name));
    }
}

#[cfg(kani)]
mod preview_proofs {
    #[kani::proof]
    fn preview_cache_arithmetic() {
        let layers = u64::from(kani::any::<u8>());
        let heads = u64::from(kani::any::<u8>());
        let dimension = u64::from(kani::any::<u8>());
        let result = super::kv_bytes_per_token(layers, heads, dimension).unwrap();
        assert_eq!(
            u128::from(result),
            u128::from(layers) * u128::from(heads) * u128::from(dimension) * 4
        );
        kani::cover!(layers == 0);
        kani::cover!(layers == 255 && heads == 255 && dimension == 255);
        assert_eq!(super::kv_bytes_per_token(u64::MAX, 2, 1), None);
    }
}
