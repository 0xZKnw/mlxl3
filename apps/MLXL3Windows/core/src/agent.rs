//! Agent mode workspace: an isolated copy of a project folder plus the four tools the Mac
//! Agent offers (`list_files`, `read_file`, `write_file`, `run`).
//!
//! Mac reference: `AgentSandbox.swift` / `AgentProjectStore.swift` (origin/main v2.0.4). What
//! is kept: the import limits and exclusions, the path rule, the tool limits and texts, the
//! review-then-apply flow with conflict detection against the hashes recorded at import.
//!
//! What differs on Windows, and is shown to the user:
//! - there is no `sandbox-exec`: `run` executes `cmd.exe` **with the user's own rights**
//!   (network and personal files reachable). It is therefore off by default, refused while
//!   file writes are off, bounded by a deadline, an output cap and a process-tree kill;
//! - checkpoints are one baseline copy (taken at import or after apply), not a git history.
//!
//! Layout: `<root>/files` (working copy, tool root), `<root>/baseline` (reviewed state),
//! `<root>/project.json` (source, hashes, permissions).

use crate::mcp::ToolResult;
use crate::turn::ToolBox;
use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub const MAX_FILES: usize = 4000;
pub const MAX_TOTAL_BYTES: u64 = 64 * 1024 * 1024;
pub const MAX_VISITED: usize = 20_000;
pub const MAX_FILE_BYTES: u64 = 2 * 1024 * 1024;
pub const OUTPUT_LIMIT: usize = 32 * 1024;
pub const WRITE_LIMIT: usize = 8192;
pub const COMMAND_LIMIT: usize = 16 * 1024;
pub const SERVER: &str = "workspace";
const EXCLUDED: &[&str] = &[
    ".git",
    ".venv",
    "venv",
    "node_modules",
    ".build",
    "target",
    "dist",
    "build",
    ".DS_Store",
    ".ssh",
    ".aws",
    ".env",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Permissions {
    pub read: bool,
    pub write: bool,
    /// Off by default on Windows: commands are not sandboxed.
    pub run: bool,
    pub command_seconds: u32,
}

impl Default for Permissions {
    fn default() -> Self {
        Self {
            read: true,
            write: true,
            run: false,
            command_seconds: 5,
        }
    }
}

impl Permissions {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (1..=30).contains(&self.command_seconds),
            "command timeout must be between 1 and 30 seconds"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct State {
    pub source: Option<PathBuf>,
    /// sha256 of each source file at import (or last apply), by relative path.
    pub original_hashes: BTreeMap<String, String>,
    pub permissions: Permissions,
    pub skipped_files: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ChangeKind {
    Added,
    Modified,
    Deleted,
}

#[derive(Clone, Debug, Serialize)]
pub struct Change {
    pub path: String,
    pub kind: ChangeKind,
    /// UTF-8 text before/after; `None` when absent or binary.
    pub before: Option<String>,
    pub after: Option<String>,
    pub binary: bool,
}

/// Mac path rule, plus the Windows-specific refusals (`\`, `:`, device names, trailing dots).
pub fn valid_path(path: &str) -> bool {
    if path.is_empty() || path.len() > 1024 || path.starts_with('/') {
        return false;
    }
    if path.chars().any(|c| {
        (c as u32) < 32
            || c as u32 == 127
            || matches!(c, '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
    }) {
        return false;
    }
    path.split('/').all(|segment| {
        let stem = segment
            .split('.')
            .next()
            .unwrap_or_default()
            .to_ascii_lowercase();
        let device = matches!(stem.as_str(), "con" | "prn" | "aux" | "nul")
            || ((stem.starts_with("com") || stem.starts_with("lpt"))
                && stem.len() == 4
                && stem.as_bytes()[3].is_ascii_digit());
        !segment.is_empty()
            && segment != "."
            && segment != ".."
            && !segment.eq_ignore_ascii_case(".git")
            && !segment.ends_with(['.', ' '])
            && !device
    })
}

fn excluded(name: &str) -> bool {
    (EXCLUDED.contains(&name) || name.starts_with(".env.")) && name != ".env.example"
}

fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Relative `/`-separated paths of the regular files under `root`, sorted, within the limits.
/// Symlinks are skipped and counted.
fn walk(root: &Path) -> Result<(Vec<(String, u64)>, usize)> {
    let mut files = vec![];
    let mut skipped = 0usize;
    let mut visited = 0usize;
    let mut total = 0u64;
    let mut stack = vec![(root.to_path_buf(), String::new())];
    while let Some((dir, prefix)) = stack.pop() {
        let mut entries: Vec<_> = fs::read_dir(&dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .collect::<std::io::Result<_>>()?;
        entries.sort_by_key(|e| e.file_name());
        for entry in entries {
            visited += 1;
            ensure!(
                visited <= MAX_VISITED,
                "project has more than {MAX_VISITED} entries"
            );
            let name = entry.file_name().to_string_lossy().into_owned();
            let relative = if prefix.is_empty() {
                name.clone()
            } else {
                format!("{prefix}/{name}")
            };
            let meta = entry.path().symlink_metadata()?;
            if excluded(&name) || !valid_path(&relative) {
                skipped += 1;
                continue;
            }
            if meta.file_type().is_symlink() {
                skipped += 1;
            } else if meta.is_dir() {
                stack.push((entry.path(), relative));
            } else if meta.is_file() {
                if meta.len() > MAX_FILE_BYTES {
                    skipped += 1;
                    continue;
                }
                total += meta.len();
                ensure!(
                    files.len() < MAX_FILES,
                    "project has more than {MAX_FILES} files"
                );
                ensure!(total <= MAX_TOTAL_BYTES, "project exceeds 64 MiB");
                files.push((relative, meta.len()));
            } else {
                skipped += 1;
            }
        }
    }
    files.sort();
    Ok((files, skipped))
}

fn copy_tree(from: &Path, to: &Path, files: &[(String, u64)]) -> Result<()> {
    for (relative, _) in files {
        let target = to.join(relative);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(from.join(relative), &target)?;
    }
    Ok(())
}

pub struct Workspace {
    root: PathBuf,
    state: State,
}

impl Workspace {
    pub fn files(&self) -> PathBuf {
        self.root.join("files")
    }
    fn baseline(&self) -> PathBuf {
        self.root.join("baseline")
    }
    pub fn state(&self) -> &State {
        &self.state
    }

    /// Open (or create empty) the workspace of one task.
    pub fn open(root: &Path) -> Result<Self> {
        fs::create_dir_all(root.join("files"))?;
        fs::create_dir_all(root.join("baseline"))?;
        let path = root.join("project.json");
        let state = if path.exists() {
            serde_json::from_slice(&fs::read(&path)?).context("damaged project.json")?
        } else {
            State::default()
        };
        state.permissions.validate()?;
        Ok(Self {
            root: root.to_path_buf(),
            state,
        })
    }

    fn save(&self) -> Result<()> {
        let path = self.root.join("project.json");
        let temporary = path.with_extension("json.tmp");
        fs::write(&temporary, serde_json::to_vec_pretty(&self.state)?)?;
        fs::rename(temporary, path)?;
        Ok(())
    }

    /// Copy `source` into the empty working copy and record its hashes.
    pub fn import(&mut self, source: &Path) -> Result<usize> {
        let source = source.canonicalize().context("project folder not found")?;
        let root = self.root.canonicalize()?;
        ensure!(
            source.parent().is_some(),
            "choose a project folder, not a drive root"
        );
        if let Some(home) = std::env::var_os(if cfg!(windows) { "USERPROFILE" } else { "HOME" }) {
            ensure!(
                Path::new(&home).canonicalize().ok().as_deref() != Some(source.as_path()),
                "choose a project folder, not your home folder"
            );
        }
        ensure!(
            !root.starts_with(&source) && !source.starts_with(&root),
            "the project and the task workspace must not contain each other"
        );
        ensure!(
            fs::read_dir(self.files())?.next().is_none() && self.state.source.is_none(),
            "this task already has a workspace"
        );
        let (files, skipped) = walk(&source)?;
        copy_tree(&source, &self.files(), &files)?;
        copy_tree(&source, &self.baseline(), &files)?;
        self.state.original_hashes = files
            .iter()
            .map(|(path, _)| Ok((path.clone(), hash(&fs::read(source.join(path))?))))
            .collect::<Result<_>>()?;
        self.state.source = Some(source);
        self.state.skipped_files = skipped;
        self.save()?;
        Ok(files.len())
    }

    pub fn set_permissions(&mut self, permissions: Permissions) -> Result<()> {
        permissions.validate()?;
        self.state.permissions = permissions;
        self.save()
    }

    pub fn list(&self) -> Result<Vec<(String, u64)>> {
        Ok(walk(&self.files())?.0)
    }

    /// Resolve a tool path inside the working copy, refusing symlinked ancestors.
    fn resolve(&self, path: &str) -> Result<PathBuf> {
        ensure!(
            valid_path(path),
            "path refused: use a relative path inside the workspace"
        );
        let files = self.files();
        let mut current = files.clone();
        for component in Path::new(path).components() {
            let Component::Normal(part) = component else {
                bail!("path refused");
            };
            current.push(part);
            if let Ok(meta) = current.symlink_metadata() {
                ensure!(
                    !meta.file_type().is_symlink(),
                    "path refused: symbolic link"
                );
            }
        }
        Ok(current)
    }

    pub fn read_file(&self, path: &str) -> Result<String> {
        let file = self.resolve(path)?;
        let mut bytes = vec![];
        fs::File::open(&file)
            .with_context(|| format!("{path}: no such file"))?
            .take(OUTPUT_LIMIT as u64 + 1)
            .read_to_end(&mut bytes)?;
        let truncated = bytes.len() > OUTPUT_LIMIT;
        bytes.truncate(OUTPUT_LIMIT);
        let text = String::from_utf8_lossy(&bytes).into_owned();
        Ok(if truncated {
            format!("{text}\n[Output truncated at 32 KiB]")
        } else {
            text
        })
    }

    pub fn write_file(&self, path: &str, content: &str) -> Result<()> {
        ensure!(content.len() <= WRITE_LIMIT, "content exceeds 8192 bytes");
        let file = self.resolve(path)?;
        let (count, total) = self
            .list()?
            .iter()
            .fold((0usize, 0u64), |(n, t), (_, size)| (n + 1, t + size));
        ensure!(
            file.exists() || count < MAX_FILES,
            "workspace limit of {MAX_FILES} files reached"
        );
        ensure!(
            total + content.len() as u64 <= MAX_TOTAL_BYTES,
            "workspace limit of 64 MiB reached"
        );
        if let Some(parent) = file.parent() {
            fs::create_dir_all(parent)?;
        }
        ensure!(!file.is_dir(), "{path} is a folder");
        fs::write(file, content)?;
        Ok(())
    }

    /// Run `command` in the working copy. Not sandboxed on Windows (see the module docs).
    pub fn run(&self, command: &str, cancelled: &AtomicBool) -> Result<(String, Option<i32>)> {
        ensure!(
            !command.is_empty() && command.len() <= COMMAND_LIMIT && !command.contains('\0'),
            "command must be 1 to 16384 bytes"
        );
        let files = self.files();
        let mut process = if cfg!(windows) {
            let root = std::env::var_os("SystemRoot").unwrap_or_else(|| "C:\\Windows".into());
            let system = PathBuf::from(&root).join("System32");
            let mut process = Command::new(system.join("cmd.exe"));
            process
                .args(["/D", "/S", "/C", command])
                .env_clear()
                .env("SystemRoot", &root)
                .env(
                    "PATH",
                    format!("{};{}", system.display(), PathBuf::from(&root).display()),
                )
                .env("PATHEXT", ".COM;.EXE;.BAT;.CMD")
                .env("ComSpec", system.join("cmd.exe"));
            process
        } else {
            let mut process = Command::new("/bin/sh");
            process
                .args(["-c", command])
                .env_clear()
                .env("PATH", "/usr/bin:/bin")
                .env("LC_ALL", "C");
            process
        };
        process
            .current_dir(&files)
            .env("HOME", &files)
            .env("USERPROFILE", &files)
            .env("TEMP", &files)
            .env("TMP", &files)
            .env("TMPDIR", &files)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        crate::mcp::detach(&mut process);
        let mut child = process.spawn().context("starting the command")?;
        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let overflow = Arc::new(AtomicBool::new(false));
        let readers: Vec<_> = [
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
        ]
        .into_iter()
        .flatten()
        .map(|mut stream| {
            let output = output.clone();
            let overflow = overflow.clone();
            std::thread::spawn(move || {
                let mut chunk = [0u8; 4096];
                while let Ok(n) = stream.read(&mut chunk) {
                    if n == 0 {
                        break;
                    }
                    let mut out = output.lock().unwrap();
                    let room = OUTPUT_LIMIT.saturating_sub(out.len());
                    out.extend_from_slice(&chunk[..n.min(room)]);
                    if n > room {
                        overflow.store(true, Ordering::Relaxed);
                    }
                }
            })
        })
        .collect();
        let deadline =
            Instant::now() + Duration::from_secs(self.state.permissions.command_seconds.into());
        let status = loop {
            if let Some(status) = child.try_wait()? {
                break Some(status);
            }
            if Instant::now() >= deadline
                || cancelled.load(Ordering::Relaxed)
                || overflow.load(Ordering::Relaxed)
            {
                crate::mcp::kill_tree(&mut child);
                let _ = child.wait();
                break None;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        for reader in readers {
            // A grandchild may keep a pipe open; never wait past the deadline for it.
            let started = Instant::now();
            while !reader.is_finished() && started.elapsed() < Duration::from_millis(500) {
                std::thread::sleep(Duration::from_millis(10));
            }
        }
        let text = String::from_utf8_lossy(&output.lock().unwrap()).into_owned();
        Ok((text, status.and_then(|s| s.code())))
    }

    /// Differences between the reviewed baseline and the working copy.
    pub fn changes(&self) -> Result<Vec<Change>> {
        let current: BTreeMap<String, u64> = self.list()?.into_iter().collect();
        let baseline: BTreeMap<String, u64> = walk(&self.baseline())?.0.into_iter().collect();
        let read = |root: PathBuf, path: &str| -> Result<(Option<String>, bool)> {
            let bytes = fs::read(root.join(path))?;
            Ok(match String::from_utf8(bytes) {
                Ok(text) if !text.contains('\0') && text.len() <= 256 * 1024 => (Some(text), false),
                _ => (None, true),
            })
        };
        let mut changes = vec![];
        for path in current
            .keys()
            .chain(baseline.keys().filter(|p| !current.contains_key(*p)))
        {
            let (kind, before, after, binary) =
                match (baseline.contains_key(path), current.contains_key(path)) {
                    (false, true) => {
                        let (after, binary) = read(self.files(), path)?;
                        (ChangeKind::Added, None, after, binary)
                    }
                    (true, false) => {
                        let (before, binary) = read(self.baseline(), path)?;
                        (ChangeKind::Deleted, before, None, binary)
                    }
                    _ => {
                        let old = fs::read(self.baseline().join(path))?;
                        let new = fs::read(self.files().join(path))?;
                        if old == new {
                            continue;
                        }
                        let (before, b1) = read(self.baseline(), path)?;
                        let (after, b2) = read(self.files(), path)?;
                        (ChangeKind::Modified, before, after, b1 || b2)
                    }
                };
            changes.push(Change {
                path: path.clone(),
                kind,
                before,
                after,
                binary,
            });
        }
        Ok(changes)
    }

    /// Write the reviewed changes to the source folder, all or nothing. A source file that
    /// changed since import (or the last apply) is a conflict and nothing is written.
    pub fn apply(&mut self) -> Result<usize> {
        let source = self
            .state
            .source
            .clone()
            .context("this task has no source project")?;
        let changes = self.changes()?;
        ensure!(!changes.is_empty(), "no changes to apply");
        for change in &changes {
            let target = source.join(&change.path);
            let now = fs::read(&target).ok().map(|b| hash(&b));
            let expected = self.state.original_hashes.get(&change.path).cloned();
            ensure!(
                now == expected,
                "Conflict: {}. The original changed after import.",
                change.path
            );
        }
        let mut undo: Vec<(PathBuf, Option<Vec<u8>>)> = vec![];
        let result = (|| -> Result<()> {
            for change in &changes {
                let target = source.join(&change.path);
                undo.push((target.clone(), fs::read(&target).ok()));
                match change.kind {
                    ChangeKind::Deleted => fs::remove_file(&target)?,
                    _ => {
                        if let Some(parent) = target.parent() {
                            fs::create_dir_all(parent)?;
                        }
                        fs::copy(self.files().join(&change.path), &target)?;
                    }
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            for (path, bytes) in undo.into_iter().rev() {
                let _ = match bytes {
                    Some(bytes) => fs::write(&path, bytes),
                    None => fs::remove_file(&path),
                };
            }
            return Err(error.context("nothing was applied"));
        }
        for change in &changes {
            let baseline = self.baseline().join(&change.path);
            match change.kind {
                ChangeKind::Deleted => {
                    let _ = fs::remove_file(&baseline);
                    self.state.original_hashes.remove(&change.path);
                }
                _ => {
                    if let Some(parent) = baseline.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    let bytes = fs::read(self.files().join(&change.path))?;
                    fs::write(&baseline, &bytes)?;
                    self.state
                        .original_hashes
                        .insert(change.path.clone(), hash(&bytes));
                }
            }
        }
        self.save()?;
        Ok(changes.len())
    }

    /// Function tools the model may call under the current permissions.
    pub fn tools(&self) -> Vec<Value> {
        let p = self.state.permissions;
        let function = |name: &str, description: String, parameters: Value| json!({"type":"function","function":{"name":format!("{SERVER}.{name}"),"description":description,"parameters":parameters}});
        let mut tools = vec![];
        if p.read {
            tools.push(function(
                "list_files",
                "List files in the private workspace.".into(),
                json!({"type":"object","properties":{}}),
            ));
            tools.push(function("read_file", "Read a UTF-8 file using a relative workspace path. Output limit: 32 KiB.".into(),
                json!({"type":"object","properties":{"path":{"type":"string","maxLength":1024}},"required":["path"],"additionalProperties":false})));
        }
        if p.write {
            tools.push(function("write_file", "Create or replace a UTF-8 file in the private workspace. Parent folders are created. Content limit: 8192 bytes.".into(),
                json!({"type":"object","properties":{"path":{"type":"string","maxLength":1024},"content":{"type":"string","maxLength":8192}},"required":["path","content"],"additionalProperties":false})));
        }
        if p.run && p.write {
            let shell = if cfg!(windows) { "cmd.exe" } else { "/bin/sh" };
            tools.push(function("run", format!(
                "Run a {shell} command in the task's working copy. It is not sandboxed: it runs with the user's account. Use relative paths. Deadline: {} seconds; output: 32 KiB.",
                p.command_seconds),
                json!({"type":"object","properties":{"command":{"type":"string","maxLength":16384}},"required":["command"],"additionalProperties":false})));
        }
        tools
    }

    fn call(&self, tool: &str, arguments: &Value, cancelled: &AtomicBool) -> Result<String> {
        let p = self.state.permissions;
        let text = |key: &str| {
            arguments[key]
                .as_str()
                .with_context(|| format!("{key} must be a string"))
        };
        match tool {
            "list_files" if p.read => {
                let files = self.list()?;
                Ok(if files.is_empty() {
                    "The workspace is empty.".into()
                } else {
                    files
                        .iter()
                        .map(|(path, _)| path.as_str())
                        .collect::<Vec<_>>()
                        .join("\n")
                })
            }
            "read_file" if p.read => self.read_file(text("path")?),
            "write_file" if p.write => {
                let path = text("path")?;
                self.write_file(path, text("content")?)?;
                Ok(format!("Wrote {path}"))
            }
            "run" if p.run && p.write => {
                let (output, status) = self.run(text("command")?, cancelled)?;
                Ok(match status {
                    Some(code) => format!("{output}\nExit status: {code}"),
                    None => {
                        format!("{output}\nCommand stopped: time/output limit or parent exited.")
                    }
                })
            }
            _ => bail!("tool disabled in this task: {tool}"),
        }
    }
}

/// The Agent tools, shareable with the turn driver.
pub struct AgentTools(pub Mutex<Workspace>);

impl ToolBox for AgentTools {
    fn server_of(&self, _: &str) -> String {
        SERVER.into()
    }

    fn call(&self, name: &str, arguments: Value, cancelled: &AtomicBool) -> ToolResult {
        let Some(tool) = name.strip_prefix("workspace.") else {
            return ToolResult {
                text: format!("Unknown MCP tool: {name}"),
                is_error: true,
            };
        };
        if !arguments.is_object() {
            return ToolResult {
                text: "Tool arguments must be an object".into(),
                is_error: true,
            };
        }
        let workspace = self.0.lock().unwrap();
        match workspace.call(tool, &arguments, cancelled) {
            Ok(text) => {
                let failed = tool == "run" && !text.ends_with("Exit status: 0");
                ToolResult {
                    text,
                    is_error: failed,
                }
            }
            Err(error) => ToolResult {
                text: format!("{error:#}"),
                is_error: true,
            },
        }
    }
}

/// Mac `AgentPrompt`, with the Windows command reality instead of the macOS sandbox.
pub fn system_prompt(permissions: &Permissions) -> String {
    let commands = if permissions.run && permissions.write {
        format!(
            "Commands run with cmd.exe in the working copy, are NOT sandboxed, have a {} second deadline and a 32 KiB output limit.",
            permissions.command_seconds
        )
    } else {
        "Commands are disabled for this task.".into()
    };
    format!(
        "You are a local coding agent. Use list_files and read_file to explore the project, write_file to edit it, and run to execute commands and verify your changes. Only enabled tools are available. You work on an isolated copy; originals are changed only after user review. {commands} Use relative paths. Never claim to have executed a command unless its tool result confirms it. Explain failures and limits honestly."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn project() -> (tempfile::TempDir, PathBuf, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("project");
        fs::create_dir_all(source.join("src")).unwrap();
        fs::create_dir_all(source.join("node_modules/x")).unwrap();
        fs::write(source.join("src/main.txt"), "one\n").unwrap();
        fs::write(source.join("README.md"), "readme").unwrap();
        fs::write(source.join(".env"), "SECRET=1").unwrap();
        fs::write(source.join(".env.example"), "SECRET=").unwrap();
        fs::write(source.join("node_modules/x/a.js"), "x").unwrap();
        let task = dir.path().join("task");
        (dir, source, task)
    }

    #[test]
    fn path_rule() {
        for ok in ["a", "src/main.rs", ".env.example", "a b/c.txt", "console/x"] {
            assert!(valid_path(ok), "{ok}");
        }
        for bad in [
            "",
            "/etc",
            "../x",
            "a/../b",
            "a//b",
            "./a",
            ".git/config",
            "A/.GIT/x",
            "a\\b",
            "C:x",
            "nul",
            "con.d",
            "COM1.txt",
            "a./b",
            "b ",
            "x\u{7f}",
            "x\ny",
            &"a".repeat(1025),
        ] {
            assert!(!valid_path(bad), "{bad:?}");
        }
    }

    #[test]
    fn import_review_apply_and_conflicts() {
        let (_dir, source, task) = project();
        let mut workspace = Workspace::open(&task).unwrap();
        assert_eq!(workspace.import(&source).unwrap(), 3);
        let files: Vec<String> = workspace
            .list()
            .unwrap()
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        assert_eq!(files, [".env.example", "README.md", "src/main.txt"]);
        assert_eq!(workspace.state().skipped_files, 2, ".env and node_modules");
        assert!(workspace.import(&source).is_err(), "a task imports once");
        assert!(workspace.changes().unwrap().is_empty());

        workspace.write_file("src/main.txt", "two\n").unwrap();
        workspace.write_file("docs/new.md", "new").unwrap();
        fs::remove_file(workspace.files().join("README.md")).unwrap();
        let changes = workspace.changes().unwrap();
        let summary: Vec<(&str, &ChangeKind)> =
            changes.iter().map(|c| (c.path.as_str(), &c.kind)).collect();
        assert_eq!(
            summary,
            [
                ("docs/new.md", &ChangeKind::Added),
                ("src/main.txt", &ChangeKind::Modified),
                ("README.md", &ChangeKind::Deleted)
            ]
        );
        assert_eq!(changes[1].before.as_deref(), Some("one\n"));

        // The original changed after import: refused, nothing written.
        fs::write(source.join("src/main.txt"), "edited elsewhere").unwrap();
        let error = workspace.apply().unwrap_err().to_string();
        assert!(error.contains("Conflict: src/main.txt"), "{error}");
        assert!(source.join("README.md").exists() && !source.join("docs/new.md").exists());
        fs::write(source.join("src/main.txt"), "one\n").unwrap();

        assert_eq!(workspace.apply().unwrap(), 3);
        assert_eq!(
            fs::read_to_string(source.join("src/main.txt")).unwrap(),
            "two\n"
        );
        assert_eq!(
            fs::read_to_string(source.join("docs/new.md")).unwrap(),
            "new"
        );
        assert!(!source.join("README.md").exists());
        assert!(
            workspace.changes().unwrap().is_empty(),
            "the baseline moved to the applied state"
        );
        let reopened = Workspace::open(&task).unwrap();
        assert!(reopened.state().original_hashes.contains_key("docs/new.md"));
    }

    #[test]
    fn import_refuses_overlaps() {
        let (dir, source, _) = project();
        let inside = source.join("task");
        assert!(Workspace::open(&inside).unwrap().import(&source).is_err());
        let mut outer = Workspace::open(&dir.path().join("t2")).unwrap();
        assert!(
            outer.import(dir.path()).is_err(),
            "source contains the workspace"
        );
    }

    #[test]
    fn tools_follow_permissions_and_refuse_escapes() {
        let (_dir, source, task) = project();
        let mut workspace = Workspace::open(&task).unwrap();
        workspace.import(&source).unwrap();
        let names = |w: &Workspace| -> Vec<String> {
            w.tools()
                .iter()
                .map(|t| t["function"]["name"].as_str().unwrap().to_owned())
                .collect()
        };
        assert_eq!(
            names(&workspace),
            [
                "workspace.list_files",
                "workspace.read_file",
                "workspace.write_file"
            ]
        );
        workspace
            .set_permissions(Permissions {
                run: true,
                ..Permissions::default()
            })
            .unwrap();
        assert!(names(&workspace).contains(&"workspace.run".to_owned()));
        workspace
            .set_permissions(Permissions {
                run: true,
                write: false,
                ..Permissions::default()
            })
            .unwrap();
        assert_eq!(
            names(&workspace).len(),
            2,
            "no writes means no commands either"
        );
        assert!(
            workspace
                .set_permissions(Permissions {
                    command_seconds: 31,
                    ..Permissions::default()
                })
                .is_err()
        );
        workspace.set_permissions(Permissions::default()).unwrap();

        let tools = AgentTools(Mutex::new(workspace));
        let flag = AtomicBool::new(false);
        let read = tools.call("workspace.read_file", json!({"path":"src/main.txt"}), &flag);
        assert_eq!((read.text.as_str(), read.is_error), ("one\n", false));
        for bad in ["../project/README.md", "/etc/passwd", ".git/x", "a\\..\\b"] {
            assert!(
                tools
                    .call("workspace.read_file", json!({"path":bad}), &flag)
                    .is_error,
                "{bad}"
            );
        }
        assert!(
            tools
                .call(
                    "workspace.write_file",
                    json!({"path":"big","content":"x".repeat(8193)}),
                    &flag
                )
                .is_error
        );
        assert!(
            tools
                .call("workspace.run", json!({"command":"echo hi"}), &flag)
                .is_error,
            "run is off by default"
        );
        assert!(tools.call("fs.read", json!({}), &flag).is_error);
        assert!(
            tools
                .call("workspace.list_files", json!([1]), &flag)
                .is_error
        );
        let listing = tools.call("workspace.list_files", json!({}), &flag);
        assert_eq!(listing.text, ".env.example\nREADME.md\nsrc/main.txt");
    }

    #[cfg(unix)]
    #[test]
    fn commands_are_bounded() {
        let (_dir, source, task) = project();
        let mut workspace = Workspace::open(&task).unwrap();
        workspace.import(&source).unwrap();
        workspace
            .set_permissions(Permissions {
                run: true,
                command_seconds: 1,
                ..Permissions::default()
            })
            .unwrap();
        let flag = AtomicBool::new(false);
        let (out, status) = workspace
            .run("cat src/main.txt; echo $HOME | grep -c files", &flag)
            .unwrap();
        assert_eq!((out.as_str(), status), ("one\n1\n", Some(0)));
        let started = Instant::now();
        let (_, status) = workspace.run("sleep 5", &flag).unwrap();
        assert!(status.is_none() && started.elapsed() < Duration::from_secs(3));
        let (out, status) = workspace.run("yes", &flag).unwrap();
        assert!(status.is_none() && out.len() == OUTPUT_LIMIT);
        let tools = AgentTools(Mutex::new(workspace));
        let failed = tools.call("workspace.run", json!({"command":"exit 3"}), &flag);
        assert!(failed.is_error && failed.text.ends_with("Exit status: 3"));
    }
}
