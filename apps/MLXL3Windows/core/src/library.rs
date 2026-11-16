//! Local EXL3 model library persisted as JSON.

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

/// Validate a Hugging Face `owner/name` and an optional subfolder before they
/// become local path components. Returns (directory name, normalized folder).
pub fn validate_download(repo: &str, folder: &str) -> Result<(String, String)> {
    let name_ok = |part: &str| {
        (1..=96).contains(&part.len())
            && part.starts_with(|c: char| c.is_ascii_alphanumeric())
            && part
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
            && !part.contains("..")
    };
    let mut parts = repo.split('/');
    let (owner, name) = (parts.next().unwrap_or(""), parts.next().unwrap_or(""));
    ensure!(
        parts.next().is_none() && name_ok(owner) && name_ok(name),
        "repository must look like owner/name"
    );
    let folder = folder.trim().trim_matches('/');
    if !folder.is_empty() {
        ensure!(
            folder.split('/').all(|p| !p.is_empty()
                && p != "."
                && p != ".."
                && !p.contains(['\\', ':', '*', '?', '"', '<', '>', '|'])),
            "folder must be a relative path inside the repository"
        );
    }
    Ok((format!("{owner}__{name}"), folder.to_owned()))
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ModelRecord {
    pub id: String,
    pub name: String,
    pub path: String,
    pub architecture: Option<String>,
    pub bits: Option<f64>,
    pub context_limit: Option<u64>,
    pub size_bytes: u64,
}

/// Inspect a folder: it must hold a config.json declaring EXL3 quantization and safetensors shards.
pub fn inspect(folder: &Path) -> Result<ModelRecord> {
    let config_path = folder.join("config.json");
    let text = std::fs::read_to_string(&config_path)
        .with_context(|| format!("no readable config.json in {}", folder.display()))?;
    let config: Value = serde_json::from_str(&text).context("config.json is not valid JSON")?;
    let quant = config.get("quantization_config").and_then(Value::as_object);
    let method = quant
        .and_then(|q| q.get("quant_method"))
        .and_then(Value::as_str)
        .unwrap_or("");
    ensure!(
        method.eq_ignore_ascii_case("exl3"),
        "this folder is not an EXL3 checkpoint (quant_method {method:?})"
    );
    let mut size_bytes = 0u64;
    let mut shards = 0usize;
    for entry in std::fs::read_dir(folder)? {
        let entry = entry?;
        if entry.path().extension().is_some_and(|e| e == "safetensors") {
            shards += 1;
            size_bytes = size_bytes.saturating_add(entry.metadata()?.len());
        }
    }
    ensure!(shards > 0, "the folder has no .safetensors weights");
    let canonical = std::fs::canonicalize(folder)?;
    let path = display_path(&canonical);
    let name = canonical
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .context("model folder has no name")?;
    let parent = canonical
        .parent()
        .and_then(Path::file_name)
        .map(|n| n.to_string_lossy().into_owned());
    // Bitrate subfolders (".../Model-EXL3/4bpw") are named after their parent.
    let name = match parent {
        Some(parent) if name.ends_with("bpw") => format!("{parent} {name}"),
        _ => name,
    };
    Ok(ModelRecord {
        id: model_id(&path),
        name,
        path,
        architecture: config
            .get("architectures")
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(Value::as_str)
            .map(str::to_owned),
        bits: quant.and_then(|q| q.get("bits")).and_then(Value::as_f64),
        context_limit: config
            .get("max_position_embeddings")
            .or_else(|| {
                config
                    .get("text_config")
                    .and_then(|t| t.get("max_position_embeddings"))
            })
            .and_then(Value::as_u64),
        size_bytes,
    })
}

/// Strip the `\\?\` prefix `canonicalize` adds on Windows; keep `\\?\UNC\` network forms as UNC.
fn display_path(path: &Path) -> String {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") {
        format!(r"\\{rest}")
    } else if let Some(rest) = text.strip_prefix(r"\\?\") {
        rest.to_owned()
    } else {
        text.into_owned()
    }
}

pub fn model_id(path: &str) -> String {
    let digest = Sha256::digest(path.to_lowercase().as_bytes());
    digest.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// Write via a sibling temporary file and rename, so a crash never leaves a truncated file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().context("path has no parent")?;
    std::fs::create_dir_all(parent)?;
    let tmp = parent.join(format!(
        ".{}.tmp",
        path.file_name().context("no file name")?.to_string_lossy()
    ));
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[derive(Debug)]
pub struct Library {
    file: PathBuf,
    pub models: Vec<ModelRecord>,
}

impl Library {
    /// A missing file is an empty library; a corrupt file is an error (never silently overwritten).
    pub fn load(file: PathBuf) -> Result<Self> {
        let models = match std::fs::read(&file) {
            Ok(bytes) => serde_json::from_slice(&bytes).with_context(|| {
                format!("{} is corrupt; move it aside to start over", file.display())
            })?,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(error) => return Err(error.into()),
        };
        Ok(Self { file, models })
    }

    pub fn save(&self) -> Result<()> {
        write_atomic(&self.file, &serde_json::to_vec_pretty(&self.models)?)
    }

    /// Insert or refresh (same canonical path → same id).
    pub fn upsert(&mut self, record: ModelRecord) -> Result<()> {
        match self.models.iter_mut().find(|m| m.id == record.id) {
            Some(existing) => *existing = record,
            None => self.models.push(record),
        }
        self.save()
    }

    pub fn remove(&mut self, id: &str) -> Result<()> {
        let before = self.models.len();
        self.models.retain(|m| m.id != id);
        if before == self.models.len() {
            bail!("no model with id {id}");
        }
        self.save()
    }

    pub fn get(&self, id: &str) -> Option<&ModelRecord> {
        self.models.iter().find(|m| m.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint(root: &Path, name: &str, config: &str, shard: bool) -> PathBuf {
        let folder = root.join(name);
        std::fs::create_dir_all(&folder).unwrap();
        std::fs::write(folder.join("config.json"), config).unwrap();
        if shard {
            std::fs::write(folder.join("model.safetensors"), [0u8; 10]).unwrap();
        }
        folder
    }

    const EXL3: &str = r#"{"architectures":["Qwen3ForCausalLM"],"max_position_embeddings":40960,
        "quantization_config":{"quant_method":"exl3","bits":4.0}}"#;

    #[test]
    fn inspect_exl3_folder() {
        let dir = tempfile::tempdir().unwrap();
        let folder = checkpoint(dir.path(), "Qwen3-8B-EXL3/4bpw", EXL3, true);
        let record = inspect(&folder).unwrap();
        assert_eq!(record.name, "Qwen3-8B-EXL3 4bpw");
        assert_eq!(record.architecture.as_deref(), Some("Qwen3ForCausalLM"));
        assert_eq!(
            (record.bits, record.context_limit, record.size_bytes),
            (Some(4.0), Some(40960), 10)
        );
        assert_eq!(record.id, model_id(&record.path));
    }

    #[test]
    fn inspect_rejections() {
        let dir = tempfile::tempdir().unwrap();
        let gptq = r#"{"quantization_config":{"quant_method":"gptq"}}"#;
        for (name, config, shard) in [
            ("a", gptq, true),
            ("b", EXL3, false),
            ("c", "{", true),
            ("d", "{}", true),
        ] {
            assert!(
                inspect(&checkpoint(dir.path(), name, config, shard)).is_err(),
                "{name}"
            );
        }
        assert!(inspect(&dir.path().join("missing")).is_err());
    }

    #[test]
    fn download_names() {
        assert_eq!(
            validate_download("0xzknw/LFM3-1.7B-EXL3", "/4bpw/").unwrap(),
            ("0xzknw__LFM3-1.7B-EXL3".into(), "4bpw".into())
        );
        assert_eq!(validate_download("a/b", "x/y").unwrap().1, "x/y");
        for (repo, folder) in [
            ("a", ""),
            ("a/b/c", ""),
            ("../b", ""),
            ("a\\..\\x/b", ""),
            ("a/b..c", ""),
            ("a/.b", ""),
            ("a/b", "../x"),
            ("a/b", "x/../y"),
            ("a/b", "C:\\x"),
            ("a/b", "x//y"),
            ("a/b", "x\\y"),
        ] {
            assert!(validate_download(repo, folder).is_err(), "{repo} {folder}");
        }
    }

    #[test]
    fn display_paths() {
        assert_eq!(display_path(Path::new(r"\\?\C:\M")), r"C:\M");
        assert_eq!(
            display_path(Path::new(r"\\?\UNC\wsl.localhost\U\m")),
            r"\\wsl.localhost\U\m"
        );
        assert_eq!(display_path(Path::new("/m")), "/m");
    }

    #[test]
    fn library_persistence() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("library.json");
        let mut library = Library::load(file.clone()).unwrap();
        assert!(library.models.is_empty());
        let record = inspect(&checkpoint(dir.path(), "M", EXL3, true)).unwrap();
        library.upsert(record.clone()).unwrap();
        library
            .upsert(ModelRecord {
                name: "renamed".into(),
                ..record.clone()
            })
            .unwrap();
        let reloaded = Library::load(file.clone()).unwrap();
        assert_eq!(reloaded.models.len(), 1);
        assert_eq!(reloaded.get(&record.id).unwrap().name, "renamed");
        library.remove(&record.id).unwrap();
        assert!(library.remove(&record.id).is_err());
        assert!(Library::load(file.clone()).unwrap().models.is_empty());
        std::fs::write(&file, b"{corrupt").unwrap();
        assert!(Library::load(file.clone()).is_err());
        assert_eq!(std::fs::read(&file).unwrap(), b"{corrupt");
    }
}
