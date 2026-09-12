//! Read-only Hugging Face EXL3 catalogue. Transfers are added separately.
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
    let mut parts = repo.split('/');
    matches!((parts.next(), parts.next(), parts.next()), (Some(owner), Some(name), None)
        if !owner.is_empty() && !name.is_empty()
        && repo.bytes().all(|byte| byte.is_ascii_alphanumeric() || b"-._/".contains(&byte)))
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
    !name.is_empty()
        && !name.starts_with('/')
        && !name.contains('\\')
        && name.split('/').all(|part| !matches!(part, "" | "." | ".."))
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
