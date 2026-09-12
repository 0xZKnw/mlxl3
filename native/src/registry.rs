//! Shared version-1 registry. Lock file and atomic replacement match Python.
use crate::checkpoint;
use anyhow::{Context, Result, ensure};
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    env,
    fs::{self, File, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelEntry {
    pub path: String,
    pub model_type: String,
    pub format: String,
    pub bits: Option<f64>,
    pub size_bytes: u64,
    pub modules: usize,
    pub added_at: String,
}

#[derive(Deserialize, Serialize)]
struct RegistryFile {
    version: u32,
    models: BTreeMap<String, ModelEntry>,
}

pub fn expand_home(path: &Path) -> Result<PathBuf> {
    if path == Path::new("~") || path.starts_with("~/") {
        let home = env::var_os("HOME").context("HOME is unavailable")?;
        Ok(PathBuf::from(home).join(path.strip_prefix("~")?))
    } else {
        Ok(path.to_owned())
    }
}

pub fn default_path() -> Result<PathBuf> {
    let home = match env::var_os("MLXL3_HOME").filter(|path| !path.is_empty()) {
        Some(path) => expand_home(Path::new(&path))?,
        None => expand_home(Path::new("~/.config/mlxl3"))?,
    };
    Ok(home.join("models.json"))
}

pub fn validate_name(name: &str) -> Result<()> {
    ensure!(
        name.as_bytes()
            .first()
            .is_some_and(u8::is_ascii_alphanumeric)
            && name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || b"._:/-".contains(&c)),
        "invalid model name"
    );
    Ok(())
}

pub fn load(path: &Path) -> Result<BTreeMap<String, ModelEntry>> {
    let file = match File::open(path) {
        Ok(file) => file,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(BTreeMap::new()),
        Err(error) => return Err(error.into()),
    };
    let registry: RegistryFile = serde_json::from_reader(file).context("invalid model registry")?;
    ensure!(
        registry.version == 1,
        "unsupported registry version {}",
        registry.version
    );
    Ok(registry.models)
}

fn update<T>(
    path: &Path,
    change: impl FnOnce(&mut BTreeMap<String, ModelEntry>) -> Result<T>,
) -> Result<T> {
    let parent = path.parent().context("registry has no parent")?;
    let parent = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    fs::create_dir_all(parent)?;
    let lock = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path.with_extension("lock"))?;
    lock.lock_exclusive()?;
    let mut models = load(path)?;
    let result = change(&mut models)?;
    let mut temp = tempfile::NamedTempFile::new_in(parent)?;
    serde_json::to_writer_pretty(&mut temp, &RegistryFile { version: 1, models })?;
    temp.write_all(b"\n")?;
    temp.as_file().sync_all()?;
    temp.persist(path).context("saving registry")?;
    File::open(parent)?.sync_all()?;
    Ok(result)
}

pub fn register(path: &Path, name: &str, model: &Path, force: bool) -> Result<ModelEntry> {
    validate_name(name)?;
    let model = checkpoint::inspect(&expand_home(model)?)?;
    let entry = ModelEntry {
        path: model
            .path
            .to_str()
            .context("model path must be UTF-8")?
            .to_owned(),
        model_type: model.model_type,
        format: "EXL3".into(),
        bits: model.bits,
        size_bytes: model.size_bytes,
        modules: model.modules.len(),
        added_at: time::OffsetDateTime::now_utc()
            .format(&time::format_description::well_known::Rfc3339)?,
    };
    update(path, |models| {
        if let Some(existing) = models.get(name) {
            ensure!(
                force || existing.path == entry.path,
                "model already exists; use --force to replace it"
            );
        }
        models.insert(name.to_owned(), entry.clone());
        Ok(entry)
    })
}

/// Remove only the registration, never model files.
pub fn remove(path: &Path, name: &str, expected: Option<&Path>) -> Result<ModelEntry> {
    update(path, |models| {
        let entry = models.get(name).context("unknown model")?;
        if let Some(expected) = expected {
            // Accept a stale path after a folder was removed, but never another registration.
            let stored = Path::new(&entry.path);
            ensure!(
                stored == expected
                    || (stored.exists()
                        && expected.exists()
                        && stored.canonicalize()? == expected.canonicalize()?),
                "model path changed; refresh before removing"
            );
        }
        Ok(models.remove(name).expect("entry checked above"))
    })
}
