//! Hugging Face EXL3 catalog, ported from the Mac app (`native/src/hub.rs`).
//!
//! - `search`: EXL3 repositories sorted by downloads (an empty query lists the most popular ones).
//! - `details`: branches/tags, complete EXL3 variants per branch, README.
//! - `download`: staged, resumable download of one variant into managed storage.
//!   Every file is pinned to a commit, checked for size and, for LFS files, sha256.
//!
//! Network access goes through [`Http`] so the logic is tested without the network.

use crate::library;
use crate::runtime::Command;
use crate::supervisor;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

const API: &str = "https://huggingface.co/api";
const SITE: &str = "https://huggingface.co";
pub const README_LIMIT: usize = 256_000;

/// Minimal HTTPS transport.
pub trait Http: Sync {
    /// GET a small response, at most `limit` bytes.
    fn get(&self, url: &str, limit: usize) -> Result<Vec<u8>>;
    /// Download `url` into `destination`, resuming a partial file. Polls `cancelled`;
    /// calls `report` periodically while it runs.
    fn fetch(
        &self,
        url: &str,
        destination: &Path,
        cancelled: &AtomicBool,
        report: &mut dyn FnMut(),
    ) -> Result<()>;
}

/// The system `curl` (built into Windows 10 1803+ and Windows 11), like the Mac app.
pub struct Curl {
    pub program: String,
    /// Hugging Face token, sent only to huggingface.co.
    pub token: Option<String>,
}

/// A child process spawned elsewhere in the app can hold a duplicate of the lock descriptor
/// between `fork` and `exec`; retry for a bounded quarter second before reporting contention.
/// Open (or create) a lock file. Windows `LockFileEx` needs read or write access on the
/// handle: an append-only handle fails with ERROR_ACCESS_DENIED.
pub(crate) fn open_lock_file(path: &Path) -> std::io::Result<File> {
    OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(false)
        .open(path)
}

/// Take the lock within 250 ms. `busy` is the error when another holder keeps it; any other
/// failure is reported as itself.
fn lock_briefly(file: &File, busy: &str) -> Result<()> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_millis(250);
    loop {
        match file.try_lock() {
            Ok(()) => return Ok(()),
            Err(std::fs::TryLockError::WouldBlock) if std::time::Instant::now() < deadline => {
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
            Err(std::fs::TryLockError::WouldBlock) => anyhow::bail!("{busy}"),
            Err(std::fs::TryLockError::Error(error)) => {
                return Err(anyhow::Error::new(error).context("locking the download state"));
            }
        }
    }
}

impl Curl {
    pub fn system(token: Option<String>) -> Self {
        let program = if cfg!(windows) {
            std::env::var("SystemRoot")
                .map(|root| format!(r"{root}\System32\curl.exe"))
                .unwrap_or_else(|_| r"C:\Windows\System32\curl.exe".into())
        } else {
            "curl".into()
        };
        Self { program, token }
    }

    fn command(&self, url: &str, args: &[&str]) -> Result<(std::process::Command, Option<String>)> {
        ensure!(
            url.starts_with("https://"),
            "only HTTPS downloads are allowed"
        );
        let mut argv: Vec<String> = args.iter().map(|a| (*a).to_owned()).collect();
        let header = match &self.token {
            Some(token) if sends_token(url) => {
                ensure!(valid_token(token), "invalid Hugging Face token");
                argv.extend(["--config".into(), "-".into()]);
                Some(format!("header = \"Authorization: Bearer {token}\"\n"))
            }
            _ => None,
        };
        argv.push(url.to_owned());
        let mut process = supervisor::process(&Command {
            label: "curl".into(),
            program: self.program.clone(),
            args: argv,
            env: vec![],
        });
        process
            .stdin(if header.is_some() {
                std::process::Stdio::piped()
            } else {
                std::process::Stdio::null()
            })
            .stderr(std::process::Stdio::piped());
        Ok((process, header))
    }
}

/// The token goes to Hugging Face only (never to GitHub or a redirect target given on the command line).
pub fn sends_token(url: &str) -> bool {
    url.starts_with("https://huggingface.co/")
}

pub fn valid_token(token: &str) -> bool {
    token.starts_with("hf_") && token.len() < 512 && token.bytes().all(|b| b.is_ascii_graphic())
}

fn stderr_text(child: &mut std::process::Child) -> String {
    let mut text = String::new();
    if let Some(stderr) = child.stderr.take() {
        let _ = stderr.take(8_000).read_to_string(&mut text);
    }
    text.trim().to_owned()
}

impl Http for Curl {
    fn get(&self, url: &str, limit: usize) -> Result<Vec<u8>> {
        let limit_text = limit.to_string();
        let (mut process, header) = self.command(
            url,
            &[
                "-fsSL",
                "--proto",
                "=https",
                "--max-time",
                "30",
                "--max-filesize",
                &limit_text,
            ],
        )?;
        process.stdout(std::process::Stdio::piped());
        let mut child = process.spawn().context("starting curl")?;
        if let (Some(header), Some(mut input)) = (header, child.stdin.take()) {
            input.write_all(header.as_bytes())?;
        }
        let output = child.wait_with_output()?;
        if !output.status.success() {
            bail!(
                "request failed: {}",
                String::from_utf8_lossy(&output.stderr).trim()
            );
        }
        ensure!(output.stdout.len() <= limit, "response is too large");
        Ok(output.stdout)
    }

    fn fetch(
        &self,
        url: &str,
        destination: &Path,
        cancelled: &AtomicBool,
        report: &mut dyn FnMut(),
    ) -> Result<()> {
        fs::create_dir_all(
            destination
                .parent()
                .context("download path has no parent")?,
        )?;
        let output = destination.to_string_lossy().into_owned();
        let (mut process, header) = self.command(
            url,
            &[
                "-fsSL",
                "--proto",
                "=https",
                "--proto-redir",
                "=https",
                "--retry",
                "3",
                "--retry-all-errors",
                "--continue-at",
                "-",
                "--output",
                &output,
            ],
        )?;
        process.stdout(std::process::Stdio::null());
        let mut child = process.spawn().context("starting curl")?;
        if let (Some(header), Some(mut input)) = (header, child.stdin.take()) {
            input.write_all(header.as_bytes())?;
        }
        loop {
            if cancelled.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                bail!("download paused");
            }
            if let Some(status) = child.try_wait()? {
                report();
                if status.success() {
                    return Ok(());
                }
                bail!("download failed: {}", stderr_text(&mut child));
            }
            report();
            std::thread::sleep(Duration::from_millis(200));
        }
    }
}

/// `HF_TOKEN`, else the token saved by `hf auth login` (`%USERPROFILE%\.cache\huggingface\token`).
pub fn token_from_environment() -> Option<String> {
    std::env::var("HF_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty())
        .or_else(|| fs::read_to_string(token_file(&home_dir()?)).ok())
        .map(|t| t.trim().to_owned())
        .filter(|t| valid_token(t))
}

pub fn home_dir() -> Option<PathBuf> {
    std::env::var_os("USERPROFILE")
        .or_else(|| std::env::var_os("HOME"))
        .map(PathBuf::from)
}

/// Where `hf auth login` keeps the token, so MLXL3 and the Hugging Face tools share one sign-in.
pub fn token_file(home: &Path) -> PathBuf {
    home.join(".cache").join("huggingface").join("token")
}

/// Which token the app would send, without revealing it.
#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct AccountStatus {
    /// `"environment"` (HF_TOKEN), `"file"` (saved sign-in) or `None`.
    pub source: Option<&'static str>,
}

pub fn account_status(environment: Option<&str>, home: Option<&Path>) -> AccountStatus {
    let usable = |t: &str| valid_token(t.trim());
    let source = if environment.is_some_and(usable) {
        Some("environment")
    } else if home
        .and_then(|h| fs::read_to_string(token_file(h)).ok())
        .is_some_and(|t| usable(&t))
    {
        Some("file")
    } else {
        None
    };
    AccountStatus { source }
}

/// The account name for the token `http` sends (`GET /api/whoami-v2`).
pub fn whoami(http: &dyn Http) -> Result<String> {
    #[derive(Deserialize)]
    struct Who {
        name: String,
    }
    let who: Who = get_json(http, &format!("{API}/whoami-v2"), 1024 * 1024)
        .context("Hugging Face rejected this token")?;
    ensure!(
        !who.name.trim().is_empty(),
        "Hugging Face returned no account name"
    );
    Ok(who.name)
}

/// Saves a token like `hf auth login` (a single line, replaced atomically).
pub fn save_token(home: &Path, token: &str) -> Result<()> {
    let token = token.trim();
    ensure!(valid_token(token), "a Hugging Face token starts with hf_");
    let path = token_file(home);
    let dir = path.parent().context("token path has no parent")?;
    fs::create_dir_all(dir)?;
    let staged = dir.join("token.mlxl3-tmp");
    fs::write(&staged, token)?;
    fs::rename(&staged, &path).inspect_err(|_| {
        let _ = fs::remove_file(&staged);
    })?;
    Ok(())
}

/// Signs out: removes the saved token. Returns whether one existed.
pub fn clear_token(home: &Path) -> Result<bool> {
    match fs::remove_file(token_file(home)) {
        Ok(()) => Ok(true),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
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

#[derive(Clone, Debug, Serialize, PartialEq)]
pub struct Variant {
    pub id: String,
    pub label: String,
    pub size_bytes: u64,
    /// local file name → path in the repository
    #[serde(skip_serializing)]
    pub files: BTreeMap<String, String>,
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

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct PendingDownload {
    pub id: String,
    pub repo: String,
    pub commit: String,
    /// Variant identifier (see [`variants`]).
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

pub fn encode(value: &str) -> String {
    value.bytes().fold(String::new(), |mut out, byte| {
        if byte.is_ascii_alphanumeric() || b"-._~/".contains(&byte) {
            out.push(byte as char);
        } else {
            out.push_str(&format!("%{byte:02X}"));
        }
        out
    })
}

fn get_json<T: for<'de> Deserialize<'de>>(http: &dyn Http, url: &str, limit: usize) -> Result<T> {
    serde_json::from_slice(&http.get(url, limit)?).context("invalid Hugging Face response")
}

/// `owner/name` with the characters Hugging Face allows.
pub fn valid_repo(repo: &str) -> bool {
    let mut slash = None;
    for (index, byte) in repo.bytes().enumerate() {
        if !(byte.is_ascii_alphanumeric() || b"-._/".contains(&byte)) {
            return false;
        }
        if byte == b'/' && slash.replace(index).is_some() {
            return false;
        }
    }
    repo.len() <= 200
        && !repo.contains("..")
        && slash.is_some_and(|index| index > 0 && index + 1 < repo.len())
}

pub fn repo_url(repo: &str) -> Result<String> {
    ensure!(valid_repo(repo), "invalid Hugging Face repository");
    Ok(format!("{SITE}/{repo}"))
}

pub fn search(http: &dyn Http, query: &str, limit: usize) -> Result<Vec<ModelSummary>> {
    let query = query.trim();
    ensure!(query.len() <= 160, "search query is too long");
    let terms: Vec<String> = query
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
    let requested = limit.clamp(1, 200);
    let url = format!(
        "{API}/models?search={}&filter=exl3&sort=downloads&direction=-1&limit={}",
        encode(needle),
        (requested * 3).min(600)
    );
    let mut found: Vec<ModelSummary> = get_json(http, &url, 8 * 1024 * 1024)?;
    found.retain(|model| {
        let id = model.id.to_ascii_lowercase();
        valid_repo(&model.id)
            && (id.contains("exl3") || model.tags.iter().any(|t| t.eq_ignore_ascii_case("exl3")))
            && terms.iter().all(|term| id.contains(term.as_str()))
    });
    found.sort_by(|a, b| b.downloads.cmp(&a.downloads).then(a.id.cmp(&b.id)));
    found.dedup_by(|a, b| a.id == b.id);
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

/// A relative repository path with no empty, `.` or `..` component and no Windows-special character.
pub fn safe_file(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 400
        && !name.starts_with('/')
        && !name.contains(['\\', ':', '*', '?', '"', '<', '>', '|', '\0'])
        && name
            .split('/')
            .all(|p| !p.is_empty() && p != "." && p != "..")
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
        boundary(start.checked_sub(1).and_then(|i| stem.as_bytes().get(i)))
            && boundary(stem.as_bytes().get(end))
    })
}

fn supporting(base: &str) -> bool {
    matches!(
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
}

/// Complete EXL3 checkpoints in one revision's file list (path → size), as on the Mac.
///
/// A variant is anchored on a `quantization_config*.json` descriptor: its folder's
/// `model*.safetensors` (filtered by the descriptor's label when there is one), the
/// supporting tokenizer/config files of that folder and its ancestors, and a config.
pub fn variants(files: &BTreeMap<String, u64>) -> Vec<Variant> {
    let files: BTreeMap<&str, u64> = files
        .iter()
        .filter(|(name, _)| safe_file(name))
        .map(|(name, size)| (name.as_str(), *size))
        .collect();
    let descriptors: Vec<(&str, Option<String>)> = files
        .keys()
        .filter_map(|name| descriptor_label(name).map(|label| (*name, label)))
        .collect();
    let mut result = Vec::new();
    for (descriptor, quant_label) in &descriptors {
        let folder = parent(descriptor);
        let mut weights: Vec<&str> = files
            .keys()
            .copied()
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
            .collect();
        if quant_label.is_none() {
            let other: Vec<&str> = descriptors
                .iter()
                .filter(|(name, label)| parent(name) == folder && label.is_some())
                .filter_map(|(_, label)| label.as_deref())
                .collect();
            weights.retain(|name| !other.iter().any(|label| belongs(name, label)));
        }
        if weights.is_empty() {
            continue;
        }
        let mut selected: BTreeMap<String, String> = BTreeMap::new();
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
                if supporting(base(name)) {
                    selected.insert(base(name).to_owned(), (*name).to_owned());
                }
            }
        }
        for name in files.keys().filter(|name| parent(name) == folder) {
            let file = base(name);
            if weights.contains(name)
                || (quant_label.is_none() && file == "model.safetensors.index.json")
            {
                selected.insert(file.to_owned(), (*name).to_owned());
            }
            if let Some(label) = quant_label.as_deref()
                && file.starts_with("config")
                && file.ends_with(".json")
                && belongs(file, label)
            {
                selected.insert("config.json".into(), (*name).to_owned());
            }
        }
        selected.insert("quantization_config.json".into(), (*descriptor).to_owned());
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
            size_bytes: selected
                .values()
                .filter_map(|name| files.get(name.as_str()))
                .sum(),
            files: selected,
        });
    }
    result
}

fn info(http: &dyn Http, repo: &str, revision: &str) -> Result<RepoInfo> {
    ensure!(valid_repo(repo), "invalid Hugging Face repository");
    ensure!(
        !revision.is_empty() && revision.len() <= 200 && !revision.contains(".."),
        "invalid revision"
    );
    let info: RepoInfo = get_json(
        http,
        &format!(
            "{API}/models/{}/revision/{}?blobs=true",
            encode(repo),
            encode(revision)
        ),
        32 * 1024 * 1024,
    )?;
    ensure!(
        info.sha.len() == 40 && info.sha.bytes().all(|b| b.is_ascii_hexdigit()),
        "Hugging Face returned an invalid commit"
    );
    Ok(info)
}

fn is_gated(value: &Value) -> bool {
    value
        .as_bool()
        .unwrap_or_else(|| value.as_str().is_some_and(|v| v != "false"))
}

fn sizes(info: &RepoInfo) -> BTreeMap<String, u64> {
    info.siblings
        .iter()
        .map(|f| (f.rfilename.clone(), f.size))
        .collect()
}

pub fn details(http: &dyn Http, repo: &str, revision: Option<&str>) -> Result<Details> {
    ensure!(valid_repo(repo), "invalid Hugging Face repository");
    let refs: References = get_json(
        http,
        &format!("{API}/models/{}/refs", encode(repo)),
        1024 * 1024,
    )?;
    let branches: Vec<String> = refs
        .branches
        .into_iter()
        .chain(refs.tags)
        .map(|r| r.name)
        .collect();
    let mut selected = revision.unwrap_or("main").to_owned();
    let mut model = info(http, repo, &selected)?;
    let mut files = sizes(&model);
    let mut choices = variants(&files);
    // Many EXL3 repositories keep each bitrate on its own branch with an empty main.
    if choices.is_empty()
        && revision.is_none()
        && let Some(candidate) = branches
            .iter()
            .find(|n| n.to_ascii_lowercase().contains("bpw"))
    {
        selected = candidate.clone();
        model = info(http, repo, &selected)?;
        files = sizes(&model);
        choices = variants(&files);
    }
    let readme = if files
        .get("README.md")
        .is_some_and(|s| *s > 0 && *s <= README_LIMIT as u64)
    {
        String::from_utf8_lossy(&http.get(
            &format!(
                "{SITE}/{}/resolve/{}/README.md",
                encode(repo),
                encode(&model.sha)
            ),
            README_LIMIT,
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

/// FNV-1a of repo/commit/variant, 12 hex digits: the staging directory name.
pub fn download_id(repo: &str, commit: &str, folder: &str) -> String {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in repo
        .bytes()
        .chain(*b"\n")
        .chain(commit.bytes())
        .chain(*b"\n")
        .chain(folder.bytes())
    {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }
    format!("{hash:016x}")[..12].to_owned()
}

/// Folder name for the finished model: `<repo name>-<variant>`, restricted to `[A-Za-z0-9._-]`.
pub fn local_name(repo: &str, folder: &str, id: &str) -> String {
    let label = folder.rsplit('#').next().unwrap_or(folder);
    let label = label.strip_prefix("quantization_config").unwrap_or(label);
    let label = label
        .strip_suffix(".json")
        .unwrap_or(label)
        .trim_matches(['.', '_', '-', '/']);
    let source = format!(
        "{}-{}",
        repo.rsplit('/').next().unwrap_or("model"),
        if label.is_empty() { &id[..8] } else { label }
    );
    let name: String = source
        .chars()
        .take(100)
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '-'
            }
        })
        .collect();
    let name = name.trim_matches('.').to_owned();
    if name.is_empty() { id.to_owned() } else { name }
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
                entry.metadata().map_or(0, |m| m.len())
            }
        })
        .sum()
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

fn staging_root(models: &Path) -> PathBuf {
    models.join(".downloads")
}

fn valid_download_id(id: &str) -> bool {
    id.len() == 12 && id.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Resolve a variant and describe its download; nothing is written.
pub fn plan_download(
    http: &dyn Http,
    models: &Path,
    repo: &str,
    commit: &str,
    folder: &str,
) -> Result<PendingDownload> {
    ensure!(
        commit.len() == 40 && commit.bytes().all(|b| b.is_ascii_hexdigit()),
        "select a resolved revision before downloading"
    );
    let detail = details(http, repo, Some(commit))?;
    ensure!(
        detail.commit == commit,
        "the revision changed on Hugging Face"
    );
    let selected = detail
        .variants
        .iter()
        .find(|v| v.id == folder)
        .context("no complete EXL3 checkpoint in the selected variant")?;
    let id = download_id(repo, commit, folder);
    let name = local_name(repo, folder, &id);
    Ok(PendingDownload {
        destination: models.join(&name).to_string_lossy().into_owned(),
        id,
        repo: repo.to_owned(),
        commit: commit.to_owned(),
        folder: folder.to_owned(),
        name,
        size_bytes: selected.size_bytes,
        retained_bytes: 0,
    })
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    library::write_atomic(path, &bytes)
}

/// Download (or resume) one variant. Returns the finished model folder.
///
/// Files land in `<models>/.downloads/<id>/snapshot/<repo path>`, are checked for size and
/// sha256, then assembled flat into `<id>/model` (hard links, else copies), validated as an
/// EXL3 checkpoint and renamed to `request.destination` inside `<models>`.
pub fn run_download(
    http: &dyn Http,
    models: &Path,
    request: &PendingDownload,
    cancelled: &AtomicBool,
    progress: &mut dyn FnMut(u64, u64),
) -> Result<PathBuf> {
    ensure!(
        valid_download_id(&request.id),
        "invalid download identifier"
    );
    ensure!(
        request.id == download_id(&request.repo, &request.commit, &request.folder),
        "download state does not match its identifier"
    );
    fs::create_dir_all(models)?;
    let root = models.canonicalize()?;
    let destination = PathBuf::from(&request.destination);
    ensure!(
        destination
            .parent()
            .and_then(|p| p.canonicalize().ok())
            .as_deref()
            == Some(root.as_path())
            && destination
                .file_name()
                .is_some_and(|n| n.to_string_lossy() == request.name),
        "model destination escaped managed storage"
    );
    ensure!(
        !destination.exists(),
        "a model named {} already exists",
        request.name
    );
    let stage = staging_root(&root).join(&request.id);
    fs::create_dir_all(&stage)?;
    let lock = open_lock_file(&staging_root(&root).join(format!("{}.lock", request.id)))?;
    lock_briefly(&lock, "this variant is already downloading")?;
    write_json(&stage.join("request.json"), request)?;

    let model = info(http, &request.repo, &request.commit)?;
    ensure!(model.sha == request.commit, "download revision changed");
    let all = sizes(&model);
    let hashes: BTreeMap<&str, &str> = model
        .siblings
        .iter()
        .filter_map(|f| {
            f.lfs
                .as_ref()
                .map(|l| (f.rfilename.as_str(), l.sha256.as_str()))
        })
        .collect();
    let selected = variants(&all)
        .into_iter()
        .find(|v| v.id == request.folder)
        .context("the selected EXL3 variant no longer exists")?;
    ensure!(
        selected.size_bytes == request.size_bytes,
        "the selected EXL3 variant changed size"
    );

    let snapshot = stage.join("snapshot");
    let total = request.size_bytes;
    progress(directory_bytes(&snapshot).min(total), total);
    for (local, remote) in &selected.files {
        ensure!(
            safe_file(remote) && safe_file(local) && !local.contains('/'),
            "unsafe checkpoint file path"
        );
        let target = snapshot.join(remote);
        let expected = *all.get(remote).context("a checkpoint file disappeared")?;
        let current = target.metadata().map_or(0, |m| m.len());
        if current == expected && expected > 0 {
            continue;
        }
        if current > expected {
            fs::remove_file(&target)?;
        }
        if expected == 0 {
            fs::create_dir_all(target.parent().context("no parent")?)?;
            File::create(&target)?;
            continue;
        }
        let url = format!(
            "{SITE}/{}/resolve/{}/{}?download=true",
            encode(&request.repo),
            encode(&request.commit),
            encode(remote)
        );
        http.fetch(&url, &target, cancelled, &mut || {
            progress(directory_bytes(&snapshot).min(total), total)
        })?;
        let got = target.metadata().map_or(0, |m| m.len());
        if got != expected {
            fs::remove_file(&target).ok();
            bail!("{remote}: downloaded {got} bytes, expected {expected}");
        }
        if let Some(expected_hash) = hashes.get(remote.as_str()) {
            let actual = sha256_file(&target)?;
            if !actual.eq_ignore_ascii_case(expected_hash) {
                fs::remove_file(&target).ok();
                bail!("{remote}: sha256 mismatch (got {actual}, expected {expected_hash})");
            }
        }
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
    library::inspect(&assembled)
        .context("the downloaded files are not a loadable EXL3 checkpoint")?;
    fs::rename(&assembled, &destination)?;
    drop(lock);
    fs::remove_dir_all(&stage)?;
    fs::remove_file(staging_root(&root).join(format!("{}.lock", request.id))).ok();
    progress(total, total);
    Ok(destination)
}

/// Interrupted downloads that can be resumed or discarded.
pub fn pending_downloads(models: &Path) -> Vec<PendingDownload> {
    let Ok(entries) = fs::read_dir(staging_root(models)) else {
        return vec![];
    };
    let mut result: Vec<PendingDownload> = entries
        .flatten()
        .filter(|e| e.path().is_dir())
        .filter_map(|entry| {
            let path = entry.path();
            let text = fs::read(path.join("request.json")).ok()?;
            let mut request: PendingDownload = serde_json::from_slice(&text).ok()?;
            (valid_download_id(&request.id)
                && path
                    .file_name()
                    .is_some_and(|n| n.to_string_lossy() == request.id))
            .then(|| {
                request.retained_bytes = directory_bytes(&path.join("snapshot"));
                request
            })
        })
        .collect();
    result.sort_by(|a, b| a.repo.cmp(&b.repo).then(a.id.cmp(&b.id)));
    result
}

pub fn find_pending(models: &Path, id: &str) -> Result<PendingDownload> {
    pending_downloads(models)
        .into_iter()
        .find(|r| r.id == id)
        .context("interrupted download not found")
}

/// Delete an interrupted download's partial files.
pub fn discard(models: &Path, id: &str) -> Result<()> {
    ensure!(valid_download_id(id), "invalid download identifier");
    let root = staging_root(models);
    fs::create_dir_all(&root)?;
    let stage = root.join(id);
    ensure!(
        !stage
            .symlink_metadata()
            .is_ok_and(|m| m.file_type().is_symlink()),
        "invalid download directory"
    );
    let lock_path = root.join(format!("{id}.lock"));
    let lock = open_lock_file(&lock_path)?;
    lock_briefly(
        &lock,
        "pause the download before removing its partial files",
    )?;
    if stage.is_dir() {
        fs::remove_dir_all(&stage)?;
    }
    drop(lock);
    fs::remove_file(lock_path).ok();
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn account_token_round_trip_and_status() {
        let home = tempfile::tempdir().unwrap();
        assert_eq!(account_status(None, Some(home.path())).source, None);
        assert!(save_token(home.path(), "not-a-token").is_err());
        assert!(!token_file(home.path()).exists());
        save_token(home.path(), "  hf_abcDEF123\n").unwrap();
        assert_eq!(
            fs::read_to_string(token_file(home.path())).unwrap(),
            "hf_abcDEF123"
        );
        assert_eq!(account_status(None, Some(home.path())).source, Some("file"));
        // HF_TOKEN wins, exactly like the download client.
        assert_eq!(
            account_status(Some("hf_env"), Some(home.path())).source,
            Some("environment")
        );
        assert_eq!(
            account_status(Some("bogus"), Some(home.path())).source,
            Some("file")
        );
        save_token(home.path(), "hf_second").unwrap();
        assert_eq!(
            fs::read_to_string(token_file(home.path())).unwrap(),
            "hf_second"
        );
        assert!(
            !home
                .path()
                .join(".cache/huggingface/token.mlxl3-tmp")
                .exists()
        );
        assert!(clear_token(home.path()).unwrap());
        assert!(!clear_token(home.path()).unwrap());
        assert_eq!(account_status(None, Some(home.path())).source, None);
    }

    #[test]
    fn whoami_reads_the_account_name() {
        let http = FakeHttp::default();
        assert!(whoami(&http).is_err());
        http.route(
            &format!("{API}/whoami-v2"),
            r#"{"name":"someone","type":"user"}"#,
        );
        assert_eq!(whoami(&http).unwrap(), "someone");
        http.route(&format!("{API}/whoami-v2"), r#"{"name":" "}"#);
        assert!(whoami(&http).is_err());
        http.route(
            &format!("{API}/whoami-v2"),
            r#"{"error":"Invalid credentials"}"#,
        );
        assert!(whoami(&http).is_err());
    }

    /// In-memory Hugging Face: URL → body. `fetch` resumes like `curl --continue-at -`.
    #[derive(Default)]
    pub struct FakeHttp {
        pub routes: Mutex<BTreeMap<String, Vec<u8>>>,
        pub requests: Mutex<Vec<String>>,
        /// Cut a fetch after this many bytes and fail (simulates a dropped connection).
        pub cut_after: Mutex<Option<usize>>,
    }

    impl FakeHttp {
        pub fn route(&self, url: &str, body: impl Into<Vec<u8>>) {
            self.routes.lock().unwrap().insert(url.into(), body.into());
        }
        fn body(&self, url: &str) -> Result<Vec<u8>> {
            self.requests.lock().unwrap().push(url.into());
            self.routes
                .lock()
                .unwrap()
                .get(url)
                .cloned()
                .with_context(|| format!("404 {url}"))
        }
    }

    impl Http for FakeHttp {
        fn get(&self, url: &str, limit: usize) -> Result<Vec<u8>> {
            let body = self.body(url)?;
            ensure!(body.len() <= limit, "response is too large");
            Ok(body)
        }
        fn fetch(
            &self,
            url: &str,
            destination: &Path,
            cancelled: &AtomicBool,
            report: &mut dyn FnMut(),
        ) -> Result<()> {
            let body = self.body(url)?;
            fs::create_dir_all(destination.parent().unwrap())?;
            let start = destination.metadata().map_or(0, |m| m.len() as usize);
            ensure!(start <= body.len(), "range not satisfiable");
            let mut end = body.len();
            let mut guard = self.cut_after.lock().unwrap();
            let cut = guard.filter(|cut| body.len() - start > *cut);
            if let Some(cut) = cut {
                guard.take();
                end = start + cut;
            }
            drop(guard);
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(destination)?;
            file.write_all(&body[start..end])?;
            report();
            ensure!(!cancelled.load(Ordering::Relaxed), "download paused");
            ensure!(cut.is_none(), "connection reset");
            Ok(())
        }
    }

    fn files(names: &[(&str, u64)]) -> BTreeMap<String, u64> {
        names.iter().map(|(n, s)| ((*n).to_owned(), *s)).collect()
    }

    #[test]
    fn variants_match_the_mac_algorithm() {
        // Root checkpoint.
        let root = variants(&files(&[
            ("config.json", 10),
            ("quantization_config.json", 5),
            ("model.safetensors", 100),
            ("tokenizer.json", 20),
            ("README.md", 7),
            (".gitattributes", 1),
        ]));
        assert_eq!(root.len(), 1);
        assert_eq!(root[0].id, "");
        assert_eq!(root[0].label, "Root");
        assert_eq!(root[0].size_bytes, 142);
        assert!(!root[0].files.contains_key(".gitattributes"));

        // Bitrate subfolders share the root tokenizer.
        let folders = variants(&files(&[
            ("tokenizer.json", 20),
            ("4.0bpw/config.json", 10),
            ("4.0bpw/quantization_config.json", 5),
            ("4.0bpw/model-00001-of-00002.safetensors", 100),
            ("4.0bpw/model-00002-of-00002.safetensors", 50),
            ("6.0bpw/config.json", 10),
            ("6.0bpw/quantization_config.json", 5),
            ("6.0bpw/model.safetensors", 300),
            ("8.0bpw/config.json", 10),
        ]));
        let ids: Vec<_> = folders.iter().map(|v| v.id.as_str()).collect();
        assert_eq!(
            ids,
            ["4.0bpw", "6.0bpw"],
            "8.0bpw has no descriptor or weights"
        );
        assert_eq!(folders[0].size_bytes, 185);
        assert_eq!(folders[0].files["tokenizer.json"], "tokenizer.json");
        assert_eq!(folders[0].files["config.json"], "4.0bpw/config.json");

        // Labelled descriptors in one folder pick their own weights and config.
        let labelled = variants(&files(&[
            ("config.json", 10),
            ("config_4bpw.json", 11),
            ("quantization_config.json", 5),
            ("quantization_config_4bpw.json", 5),
            ("model.safetensors", 100),
            ("model_4bpw.safetensors", 60),
        ]));
        assert_eq!(labelled.len(), 2);
        let plain = labelled.iter().find(|v| v.id.is_empty()).unwrap();
        assert!(!plain.files.values().any(|f| f == "model_4bpw.safetensors"));
        let four = labelled.iter().find(|v| v.label == "4bpw").unwrap();
        assert_eq!(four.id, "#quantization_config_4bpw.json");
        assert_eq!(four.files["config.json"], "config_4bpw.json");
        assert_eq!(
            four.files["model_4bpw.safetensors"],
            "model_4bpw.safetensors"
        );

        // Unsafe paths are never selected.
        let unsafe_paths = variants(&files(&[
            ("../config.json", 1),
            ("a/../quantization_config.json", 1),
            ("C:/model.safetensors", 1),
        ]));
        assert!(unsafe_paths.is_empty());
    }

    #[test]
    fn belongs_respects_label_boundaries() {
        assert!(belongs("model_4bpw.safetensors", "4bpw"));
        assert!(!belongs("model_14bpw.safetensors", "4bpw"));
        assert!(!belongs("model_4bpwx.safetensors", "4bpw"));
        assert!(!belongs("model_4.0bpw.safetensors", "0bpw"));
        assert!(belongs("4bpw", "4bpw"));
    }

    #[test]
    fn repository_and_file_validation() {
        for good in ["turboderp/Qwen3-8B-exl3", "a/b", "A.b_c-d/e.f"] {
            assert!(valid_repo(good), "{good}");
        }
        for bad in [
            "", "a", "/b", "a/", "a/b/c", "a/../b", "a b/c", "a/b?x", "a\\b",
        ] {
            assert!(!valid_repo(bad), "{bad}");
        }
        assert!(safe_file("4bpw/model.safetensors"));
        for bad in ["", "/etc/x", "a//b", "a/./b", "..", "a\\b", "c:x", "a/b?"] {
            assert!(!safe_file(bad), "{bad}");
        }
        assert_eq!(encode("a b/é#"), "a%20b/%C3%A9%23");
        assert_eq!(repo_url("a/b").unwrap(), "https://huggingface.co/a/b");
        assert!(repo_url("a/b/../c").is_err());
    }

    #[test]
    fn token_is_only_sent_to_hugging_face() {
        assert!(sends_token("https://huggingface.co/api/models"));
        assert!(!sends_token("https://huggingface.co.evil.test/"));
        assert!(!sends_token("https://api.github.com/repos"));
        assert!(!sends_token("http://huggingface.co/"));
        assert!(valid_token("hf_abc"));
        assert!(!valid_token("hf_a\"b c"));
        assert!(!valid_token("ghp_x"));
        let curl = Curl {
            program: "curl".into(),
            token: Some("hf_secret".into()),
        };
        let (gh, header) = curl.command("https://api.github.com/x", &[]).unwrap();
        assert!(header.is_none());
        assert!(!format!("{gh:?}").contains("hf_secret"));
        let (hf, header) = curl.command("https://huggingface.co/api/x", &[]).unwrap();
        assert!(header.unwrap().contains("Bearer hf_secret"));
        assert!(
            !format!("{hf:?}").contains("hf_secret"),
            "the token never appears in argv"
        );
        assert!(curl.command("http://huggingface.co/x", &[]).is_err());
    }

    #[test]
    fn local_names_are_safe_folder_names() {
        let id = download_id("a/Qwen3-8B-exl3", &"0".repeat(40), "4.0bpw");
        assert_eq!(id.len(), 12);
        assert_ne!(
            id,
            download_id("a/Qwen3-8B-exl3", &"0".repeat(40), "6.0bpw")
        );
        assert_eq!(
            local_name("a/Qwen3-8B-exl3", "4.0bpw", &id),
            "Qwen3-8B-exl3-4.0bpw"
        );
        assert_eq!(
            local_name("a/M", "#quantization_config_4bpw.json", &id),
            "M-4bpw"
        );
        assert_eq!(local_name("a/M", "", &id), format!("M-{}", &id[..8]));
        assert_eq!(local_name("a/M", "x/../y z", &id), "M-x-..-y-z");
        assert!(
            local_name("a/..", "..", &id)
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || "._-".contains(c))
        );
    }

    const COMMIT: &str = "0123456789abcdef0123456789abcdef01234567";

    fn hex(bytes: &[u8]) -> String {
        Sha256::digest(bytes)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect()
    }

    /// One repository whose main is empty and whose 4.0bpw branch holds a checkpoint.
    pub fn fake_repo(http: &FakeHttp, weights: &[u8]) {
        let config = br#"{"architectures":["Qwen3ForCausalLM"],"quantization_config":{"quant_method":"exl3","bits":4.0}}"#;
        let quant = br#"{"quant_method":"exl3","bits":4.0}"#;
        http.route(
            &format!("{API}/models/o/M-exl3/refs"),
            r#"{"branches":[{"name":"main"},{"name":"4.0bpw"}],"tags":[]}"#,
        );
        http.route(
            &format!("{API}/models/o/M-exl3/revision/main?blobs=true"),
            format!(r#"{{"id":"o/M-exl3","sha":"{}","siblings":[{{"rfilename":"README.md","size":4}}]}}"#, "f".repeat(40)),
        );
        let siblings = format!(
            r#"[{{"rfilename":"config.json","size":{}}},{{"rfilename":"quantization_config.json","size":{}}},{{"rfilename":"model.safetensors","size":{},"lfs":{{"sha256":"{}"}}}},{{"rfilename":"README.md","size":6}}]"#,
            config.len(),
            quant.len(),
            weights.len(),
            hex(weights)
        );
        let info = format!(
            r#"{{"id":"o/M-exl3","sha":"{COMMIT}","downloads":42,"gated":false,"siblings":{siblings}}}"#
        );
        http.route(
            &format!("{API}/models/o/M-exl3/revision/4.0bpw?blobs=true"),
            info.clone(),
        );
        http.route(
            &format!("{API}/models/o/M-exl3/revision/{COMMIT}?blobs=true"),
            info,
        );
        http.route(
            &format!("{SITE}/o/M-exl3/resolve/{COMMIT}/README.md"),
            "# M\nhi",
        );
        for (name, body) in [
            ("config.json", &config[..]),
            ("quantization_config.json", quant),
            ("model.safetensors", weights),
            ("README.md", b"# M\nhi"),
        ] {
            http.route(
                &format!("{SITE}/o/M-exl3/resolve/{COMMIT}/{name}?download=true"),
                body,
            );
        }
    }

    #[test]
    fn search_filters_and_sorts_like_the_mac() {
        let http = FakeHttp::default();
        http.route(
            &format!("{API}/models?search=exl3&filter=exl3&sort=downloads&direction=-1&limit=6"),
            r#"[{"id":"a/Small-exl3","downloads":5},{"id":"b/Big-EXL3","downloads":50},
                {"id":"c/plain","downloads":99,"tags":["exl3"]},{"id":"d/gguf","downloads":1000},
                {"id":"bad/../x-exl3","downloads":7}]"#,
        );
        let found = search(&http, "", 2).unwrap();
        let ids: Vec<_> = found.iter().map(|m| m.id.as_str()).collect();
        assert_eq!(ids, ["c/plain", "b/Big-EXL3"]);
        http.route(
            &format!("{API}/models?search=qwen3&filter=exl3&sort=downloads&direction=-1&limit=30"),
            r#"[{"id":"a/Qwen3-8B-exl3","downloads":5},{"id":"a/Qwen3-4B-exl3","downloads":9},{"id":"a/Qwen2-exl3"}]"#,
        );
        let found = search(&http, "qwen3 8b exl3", 10).unwrap();
        assert_eq!(
            found.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            ["a/Qwen3-8B-exl3"]
        );
        assert!(search(&http, &"x".repeat(161), 1).is_err());
    }

    #[test]
    fn details_fall_back_to_a_bpw_branch() {
        let http = FakeHttp::default();
        fake_repo(&http, b"weights!");
        let d = details(&http, "o/M-exl3", None).unwrap();
        assert_eq!(d.revision, "4.0bpw");
        assert_eq!(d.commit, COMMIT);
        assert_eq!(d.branches, ["main", "4.0bpw"]);
        assert_eq!(d.variants.len(), 1);
        assert_eq!(d.readme, "# M\nhi");
        assert_eq!(d.downloads, 42);
        assert!(!d.gated);
        assert!(details(&http, "o/../x", None).is_err());
    }

    #[test]
    fn download_resumes_verifies_and_installs() {
        let http = FakeHttp::default();
        let weights = vec![7u8; 5000];
        fake_repo(&http, &weights);
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("models");
        let request = plan_download(&http, &models, "o/M-exl3", COMMIT, "").unwrap();
        assert_eq!(request.name, format!("M-exl3-{}", &request.id[..8]));
        let cancelled = AtomicBool::new(false);
        let mut seen = vec![];

        // A dropped connection leaves a resumable partial download.
        *http.cut_after.lock().unwrap() = Some(1000);
        let error = run_download(&http, &models, &request, &cancelled, &mut |d, t| {
            seen.push((d, t))
        })
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("connection reset"),
            "{error:#}"
        );
        let pending = pending_downloads(&models);
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].id, request.id);
        assert!(pending[0].retained_bytes >= 1000);

        // Resume completes and only fetches the remaining bytes.
        let path = run_download(&http, &models, &pending[0], &cancelled, &mut |d, t| {
            seen.push((d, t))
        })
        .unwrap();
        assert_eq!(path, models.join(&request.name));
        assert_eq!(fs::read(path.join("model.safetensors")).unwrap(), weights);
        let record = library::inspect(&path).unwrap();
        assert_eq!(record.bits, Some(4.0));
        assert!(pending_downloads(&models).is_empty());
        assert!(!models.join(".downloads").join(&request.id).exists());
        assert_eq!(seen.last(), Some(&(request.size_bytes, request.size_bytes)));
        assert!(seen.iter().all(|(d, t)| d <= t));

        // The same name cannot be downloaded twice.
        assert!(run_download(&http, &models, &request, &cancelled, &mut |_, _| {}).is_err());
    }

    #[test]
    fn corrupt_files_are_rejected_and_removed() {
        let http = FakeHttp::default();
        fake_repo(&http, b"good-weights");
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("models");
        let request = plan_download(&http, &models, "o/M-exl3", COMMIT, "").unwrap();
        // Same size, different content: only the sha256 can tell.
        http.route(
            &format!("{SITE}/o/M-exl3/resolve/{COMMIT}/model.safetensors?download=true"),
            "evil-weights",
        );
        let error = run_download(
            &http,
            &models,
            &request,
            &AtomicBool::new(false),
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(
            format!("{error:#}").contains("sha256 mismatch"),
            "{error:#}"
        );
        assert!(!models.join(&request.name).exists());
        assert!(
            !models
                .join(".downloads")
                .join(&request.id)
                .join("snapshot/model.safetensors")
                .exists()
        );

        // Pause, then discard removes the partial files.
        let error = run_download(
            &http,
            &models,
            &request,
            &AtomicBool::new(true),
            &mut |_, _| {},
        )
        .unwrap_err();
        assert!(format!("{error:#}").contains("paused"));
        discard(&models, &request.id).unwrap();
        assert!(pending_downloads(&models).is_empty());
        assert!(discard(&models, "../../etc").is_err());
    }

    #[test]
    fn tampered_requests_cannot_escape_storage() {
        let http = FakeHttp::default();
        fake_repo(&http, b"w");
        let dir = tempfile::tempdir().unwrap();
        let models = dir.path().join("models");
        let request = plan_download(&http, &models, "o/M-exl3", COMMIT, "").unwrap();
        let outside = PendingDownload {
            destination: dir.path().join("M").to_string_lossy().into(),
            ..request.clone()
        };
        assert!(
            run_download(
                &http,
                &models,
                &outside,
                &AtomicBool::new(false),
                &mut |_, _| {}
            )
            .is_err()
        );
        let renamed = PendingDownload {
            name: "other".into(),
            ..request.clone()
        };
        assert!(
            run_download(
                &http,
                &models,
                &renamed,
                &AtomicBool::new(false),
                &mut |_, _| {}
            )
            .is_err()
        );
        let wrong_id = PendingDownload {
            id: "aaaaaaaaaaaa".into(),
            ..request.clone()
        };
        assert!(
            run_download(
                &http,
                &models,
                &wrong_id,
                &AtomicBool::new(false),
                &mut |_, _| {}
            )
            .is_err()
        );
        let other_variant = PendingDownload {
            folder: "x".into(),
            id: download_id("o/M-exl3", COMMIT, "x"),
            ..request
        };
        assert!(
            run_download(
                &http,
                &models,
                &other_variant,
                &AtomicBool::new(false),
                &mut |_, _| {}
            )
            .is_err()
        );
    }
}
