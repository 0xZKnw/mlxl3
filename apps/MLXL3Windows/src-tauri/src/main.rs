//! MLXL3 Desktop for Windows: thin Tauri shell over `mlxl3-windows-core`.
//!
//! The engine is upstream ExLlamaV3, run by the bundled Python worker in a
//! private runtime: natively with CUDA, or inside WSL2 with ROCm for AMD GPUs.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{Context, Result, ensure};
use mlxl3_windows_core::hub::{self, Curl, PendingDownload};
use mlxl3_windows_core::library::{self, Library};
use mlxl3_windows_core::protocol::{self, Message, Sampling};
use mlxl3_windows_core::runtime::{self, Layout, Marker, WslTarget};
use mlxl3_windows_core::supervisor::{self, Engine, EngineMessage};
use mlxl3_windows_core::update::{self, Channel, ReleaseId, Update};
use mlxl3_windows_core::{Backend, agent, documents, extract, gpu, mcp, server, turn, wsl};
use serde::Serialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

const PROBE_TIMEOUT: Duration = Duration::from_secs(15);
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);
const UPDATE_FIRST_CHECK: Duration = Duration::from_secs(4);
const UPDATE_INTERVAL: Duration = Duration::from_secs(6 * 60 * 60);

struct Loaded {
    epoch: u64,
    model_id: String,
    context: u32,
    engine: Engine,
}

struct AppState {
    layout: Layout,
    library: Mutex<Library>,
    engine: Mutex<Option<Loaded>>,
    epoch: AtomicU64,
    busy: AtomicBool,
    /// One model download at a time; `pause` stops it, keeping its partial files.
    download: Mutex<Option<Arc<AtomicBool>>>,
    update: Mutex<UpdateStatus>,
    updating: AtomicBool,
    channel: Mutex<Channel>,
    /// Project document library, opened on first use.
    documents: Mutex<Option<documents::Library>>,
    /// Bridge events of server-mode requests, by request id.
    routes: Mutex<HashMap<String, mpsc::Sender<Value>>>,
    server: Mutex<Option<server::Handle>>,
    /// Chat turns run by the core turn driver, by request id: their cancel flags.
    turns: Mutex<HashMap<String, Arc<AtomicBool>>>,
    /// MCP connections (off until the user enables MCP).
    mcp: Arc<Mutex<mcp::Manager>>,
    /// Open Agent workspaces, by task id.
    agents: Mutex<HashMap<String, Arc<agent::AgentTools>>>,
}

#[derive(Clone, Default, Serialize)]
struct UpdateStatus {
    /// idle, checking, current, downloading, ready, failed
    phase: &'static str,
    current: String,
    channel: Channel,
    update: Option<Update>,
    done: u64,
    total: u64,
    error: Option<String>,
    #[serde(skip)]
    installer: Option<PathBuf>,
}

type Reply<T> = std::result::Result<T, String>;

fn reply<T>(result: Result<T>) -> Reply<T> {
    result.map_err(|e| format!("{e:#}"))
}

fn tool(command: &str, args: &[&str]) -> runtime::Command {
    runtime::Command {
        label: command.into(),
        program: command.into(),
        args: args.iter().map(|a| (*a).to_owned()).collect(),
        env: vec![],
    }
}

/// stdout of a short diagnostic command, or None when it is missing or fails.
fn capture(command: runtime::Command) -> Option<String> {
    let done = supervisor::run(&command, Some(PROBE_TIMEOUT), |_| {}).ok()?;
    (done.code == Some(0)).then_some(done.stdout)
}

fn wsl_distros() -> Vec<String> {
    let Ok(done) = supervisor::run(
        &tool("wsl.exe", &["--list", "--quiet"]),
        Some(PROBE_TIMEOUT),
        |_| {},
    ) else {
        return vec![];
    };
    if done.code != Some(0) {
        return vec![];
    }
    // wsl.exe writes UTF-16LE: parse the raw bytes.
    wsl::parse_distro_list(&done.stdout_bytes)
        .into_iter()
        .filter(|d| !d.starts_with("docker-desktop"))
        .collect()
}

#[derive(Serialize)]
struct SystemInfo {
    app_version: String,
    nvidia: Vec<gpu::NvidiaGpu>,
    amd: Vec<String>,
    wsl_distros: Vec<String>,
    recommendation: gpu::Recommendation,
    runtimes: Value,
    runtime_ids: Value,
}

#[tauri::command]
async fn system_info(app: AppHandle, state: State<'_, AppState>) -> Reply<SystemInfo> {
    let layout = state.layout.clone();
    let version = current_release(&app);
    tauri::async_runtime::spawn_blocking(move || {
        let nvidia = capture(tool(
            "nvidia-smi",
            &[
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
        ))
        .map(|s| gpu::parse_nvidia_smi(&s))
        .unwrap_or_default();
        let amd = capture(tool(
            "powershell.exe",
            &[
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                "Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }",
            ],
        ))
        .map(|s| gpu::amd_adapters(&s))
        .unwrap_or_default();
        let distros = wsl_distros();
        let recommendation = gpu::recommend(&nvidia, &amd, &distros);
        SystemInfo {
            app_version: version,
            nvidia,
            amd,
            wsl_distros: distros,
            recommendation,
            runtimes: json!({
                "cuda": runtime::installed(&layout, Backend::Cuda),
                "rocm_wsl": runtime::installed(&layout, Backend::RocmWsl),
            }),
            runtime_ids: json!({
                "cuda": runtime::runtime_id(Backend::Cuda),
                "rocm_wsl": runtime::runtime_id(Backend::RocmWsl),
            }),
        }
    })
    .await
    .map_err(|e| e.to_string())
}

fn wsl_target(distro: &str) -> Result<WslTarget> {
    wsl::validate_distro(distro)?;
    let done = supervisor::run(
        &tool("wsl.exe", &["-d", distro, "--exec", "printenv", "HOME"]),
        Some(Duration::from_secs(120)),
        |_| {},
    )?;
    ensure!(
        done.code == Some(0),
        "WSL distribution {distro} did not start: {}",
        done.stderr_tail.trim()
    );
    Ok(WslTarget {
        distro: distro.to_owned(),
        home: wsl::parse_home(done.stdout.as_bytes())?,
    })
}

/// Parse the probe's single JSON line (the last JSON object on stdout).
fn probe_report(stdout: &str) -> Result<Value> {
    stdout
        .lines()
        .rev()
        .find_map(|l| {
            serde_json::from_str::<Value>(l.trim())
                .ok()
                .filter(Value::is_object)
        })
        .context("the GPU check printed no report")
}

/// "GPU check failed: …" followed by the probe's hints, one per line.
fn probe_failure(report: &Value) -> String {
    let mut message = format!(
        "GPU check failed: {}",
        report
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or("unknown error")
    );
    for hint in report
        .get("hints")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        message.push_str("\n• ");
        message.push_str(hint);
    }
    message
}

fn install(
    app: &AppHandle,
    layout: &Layout,
    backend: Backend,
    distro: Option<String>,
) -> Result<Marker> {
    let target = match backend {
        Backend::RocmWsl => Some(wsl_target(
            distro.as_deref().context("choose a WSL2 distribution")?,
        )?),
        Backend::Cuda => None,
    };
    let plan = runtime::install_plan(backend, layout, target.as_ref())?;
    let total = plan.len();
    let mut report = Value::Null;
    for (index, step) in plan.iter().enumerate() {
        let _ = app.emit(
            "runtime-progress",
            json!({"backend": backend, "step": index + 1, "total": total, "label": step.label}),
        );
        let handle = app.clone();
        let done = supervisor::run(step, None, move |line| {
            let _ = handle.emit("runtime-log", json!({"backend": backend, "line": line}));
        })?;
        if index + 1 == total {
            report = probe_report(&done.stdout)?;
            ensure!(
                report.get("ok").and_then(Value::as_bool) == Some(true),
                "{}",
                probe_failure(&report)
            );
        } else {
            ensure!(
                done.code == Some(0),
                "{} failed (exit {:?}):\n{}",
                step.label,
                done.code,
                done.stderr_tail.trim()
            );
        }
    }
    let marker = Marker {
        id: runtime::runtime_id(backend),
        backend,
        wsl: target,
        probe: report,
    };
    runtime::write_marker(layout, &marker)?;
    Ok(marker)
}

struct BusyGuard<'a>(&'a AtomicBool);

impl Drop for BusyGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::SeqCst);
    }
}

fn claim(state: &AppState) -> Result<BusyGuard<'_>> {
    ensure!(
        state
            .busy
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_ok(),
        "an installation or download is already running"
    );
    Ok(BusyGuard(&state.busy))
}

#[tauri::command]
async fn install_runtime(
    app: AppHandle,
    backend: Backend,
    distro: Option<String>,
) -> Reply<Marker> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let _busy = claim(&state)?;
        install(&app, &state.layout, backend, distro)
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(reply)
}

#[tauri::command]
fn list_models(state: State<'_, AppState>) -> Vec<library::ModelRecord> {
    state.library.lock().unwrap().models.clone()
}

#[tauri::command]
fn add_model(state: State<'_, AppState>, path: String) -> Reply<library::ModelRecord> {
    reply((|| {
        let record = library::inspect(&PathBuf::from(&path))?;
        state.library.lock().unwrap().upsert(record.clone())?;
        Ok(record)
    })())
}

#[tauri::command]
fn remove_model(state: State<'_, AppState>, id: String) -> Reply<()> {
    let loaded = state
        .engine
        .lock()
        .unwrap()
        .as_ref()
        .map(|l| l.model_id.clone());
    if loaded.as_deref() == Some(id.as_str()) {
        return Err("unload the model before removing it".into());
    }
    reply(state.library.lock().unwrap().remove(&id))
}

fn http() -> Curl {
    Curl::system(hub::token_from_environment())
}

/// The release tag this build came from. CI sets `MLXL3_RELEASE_TAG` (a beta build's
/// package version is only its numeric part); local builds use the package version.
fn current_release(app: &AppHandle) -> String {
    option_env!("MLXL3_RELEASE_TAG")
        .and_then(ReleaseId::parse)
        .map(|id| id.to_string())
        .unwrap_or_else(|| app.package_info().version.to_string())
}

fn channel_file(layout: &Layout) -> PathBuf {
    layout.data_dir.join("update-channel")
}

fn load_channel(layout: &Layout) -> Channel {
    match std::fs::read_to_string(channel_file(layout))
        .as_deref()
        .map(str::trim)
    {
        Ok("beta") => Channel::Beta,
        _ => Channel::Stable,
    }
}

#[derive(Serialize)]
struct Account {
    /// "environment" (HF_TOKEN), "file" (saved sign-in) or null.
    source: Option<&'static str>,
    name: Option<String>,
}

fn account(name: Option<String>) -> Account {
    let environment = std::env::var("HF_TOKEN").ok();
    let home = hub::home_dir();
    Account {
        source: hub::account_status(environment.as_deref(), home.as_deref()).source,
        name,
    }
}

const MAX_BACKUP_BYTES: u64 = 64 * 1024 * 1024;

/// Save `contents` where the user picks, through the native dialog (Mac NSSavePanel).
/// Returns false when the user cancels.
#[tauri::command]
async fn save_text_file(app: AppHandle, name: String, contents: String) -> Reply<bool> {
    use tauri_plugin_dialog::DialogExt;
    let extension = match Path::new(&name).extension().and_then(|e| e.to_str()) {
        Some("json") => "json",
        Some("md") => "md",
        _ => return Err("only .json and .md exports are supported".into()),
    };
    if contents.len() as u64 > MAX_BACKUP_BYTES {
        return Err("the export is too large".into());
    }
    tauri::async_runtime::spawn_blocking(move || {
        let Some(picked) = app
            .dialog()
            .file()
            .set_file_name(&name)
            .add_filter(extension, &[extension])
            .blocking_save_file()
        else {
            return Ok(false);
        };
        let path = picked
            .into_path()
            .context("the chosen location is not a file path")?;
        let staged = path.with_extension(format!("{extension}.mlxl3-tmp"));
        std::fs::write(&staged, contents.as_bytes())?;
        std::fs::rename(&staged, &path).inspect_err(|_| {
            let _ = std::fs::remove_file(&staged);
        })?;
        Ok(true)
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(reply)
}

/// Read a backup the user picks (JSON, at most 64 MB). `None` when the user cancels.
#[tauri::command]
async fn open_backup_file(app: AppHandle) -> Reply<Option<String>> {
    use std::io::Read;
    use tauri_plugin_dialog::DialogExt;
    tauri::async_runtime::spawn_blocking(move || {
        let Some(picked) = app
            .dialog()
            .file()
            .add_filter("json", &["json"])
            .blocking_pick_file()
        else {
            return Ok(None);
        };
        let path = picked
            .into_path()
            .context("the chosen file is not a file path")?;
        let mut text = String::new();
        std::fs::File::open(&path)?
            .take(MAX_BACKUP_BYTES + 1)
            .read_to_string(&mut text)
            .context("the backup is not UTF-8 text")?;
        ensure!(
            text.len() as u64 <= MAX_BACKUP_BYTES,
            "the backup is larger than 64 MB"
        );
        Ok(Some(text))
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(reply)
}

/// "Response complete" toast (Mac CompletionNotifications). The UI sends only its fixed,
/// localized wording: no prompt, answer, model path or conversation title leaves the app.
#[tauri::command]
fn notify_done(app: AppHandle, title: String, body: String) -> Reply<()> {
    use tauri_plugin_notification::NotificationExt;
    reply((|| {
        ensure!(
            title.chars().count() <= 80 && body.chars().count() <= 200,
            "notification text is too long"
        );
        app.notification()
            .builder()
            .title(title)
            .body(body)
            .show()
            .context("showing the notification")
    })())
}

#[tauri::command]
fn hf_account() -> Account {
    account(None)
}

/// Checks the token with Hugging Face (`whoami`), then saves it like `hf auth login`.
#[tauri::command]
async fn hf_sign_in(token: String) -> Reply<Account> {
    tauri::async_runtime::spawn_blocking(move || {
        let token = token.trim().to_owned();
        ensure!(
            hub::valid_token(&token),
            "a Hugging Face token starts with hf_"
        );
        let name = hub::whoami(&Curl::system(Some(token.clone())))?;
        hub::save_token(&hub::home_dir().context("no user profile folder")?, &token)?;
        Ok(account(Some(name)))
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(reply)
}

#[tauri::command]
fn hf_sign_out() -> Reply<Account> {
    reply((|| {
        hub::clear_token(&hub::home_dir().context("no user profile folder")?)?;
        Ok(account(None))
    })())
}

fn models_dir(layout: &Layout) -> PathBuf {
    layout.data_dir.join("models")
}

#[tauri::command]
async fn hub_search(query: String, limit: usize) -> Reply<Vec<hub::ModelSummary>> {
    tauri::async_runtime::spawn_blocking(move || hub::search(&http(), &query, limit))
        .await
        .map_err(|e| e.to_string())
        .and_then(reply)
}

#[tauri::command]
async fn hub_details(repo: String, revision: Option<String>) -> Reply<hub::Details> {
    tauri::async_runtime::spawn_blocking(move || {
        hub::details(
            &http(),
            &repo,
            revision.as_deref().filter(|r| !r.is_empty()),
        )
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(reply)
}

/// Run one download on a background thread; progress and the outcome arrive as events.
fn start_download(app: &AppHandle, request: PendingDownload) -> Result<()> {
    let state = app.state::<AppState>();
    let cancelled = Arc::new(AtomicBool::new(false));
    {
        let mut active = state.download.lock().unwrap();
        ensure!(active.is_none(), "another model is already downloading");
        *active = Some(cancelled.clone());
    }
    let app = app.clone();
    std::thread::spawn(move || {
        let state = app.state::<AppState>();
        let models = models_dir(&state.layout);
        let summary = json!({"id": request.id, "name": request.name, "repo": request.repo});
        let mut last = std::time::Instant::now() - Duration::from_secs(1);
        let result = hub::run_download(
            &http(),
            &models,
            &request,
            &cancelled,
            &mut |done, total| {
                if last.elapsed() >= Duration::from_millis(250) || done == total {
                    last = std::time::Instant::now();
                    let _ = app.emit(
                        "download-progress",
                        json!({"download": summary, "done": done, "total": total}),
                    );
                }
            },
        )
        .and_then(|path| {
            let record = library::inspect(&path)?;
            state.library.lock().unwrap().upsert(record.clone())?;
            Ok(record)
        });
        *state.download.lock().unwrap() = None;
        match result {
            Ok(record) => {
                let _ = app.emit(
                    "download-finished",
                    json!({"download": summary, "model": record}),
                );
            }
            Err(error) => {
                let _ = app.emit(
                    "download-failed",
                    json!({
                        "download": summary,
                        "paused": cancelled.load(Ordering::SeqCst),
                        "error": format!("{error:#}"),
                    }),
                );
            }
        }
    });
    Ok(())
}

#[tauri::command]
async fn hub_download(
    app: AppHandle,
    repo: String,
    commit: String,
    folder: String,
) -> Reply<PendingDownload> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let request =
            hub::plan_download(&http(), &models_dir(&state.layout), &repo, &commit, &folder)?;
        start_download(&app, request.clone())?;
        Ok(request)
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(reply)
}

#[tauri::command]
fn hub_pending(state: State<'_, AppState>) -> Vec<PendingDownload> {
    hub::pending_downloads(&models_dir(&state.layout))
}

#[tauri::command]
fn hub_resume(app: AppHandle, id: String) -> Reply<PendingDownload> {
    reply((|| {
        let state = app.state::<AppState>();
        let request = hub::find_pending(&models_dir(&state.layout), &id)?;
        start_download(&app, request.clone())?;
        Ok(request)
    })())
}

#[tauri::command]
fn hub_pause(state: State<'_, AppState>) {
    if let Some(flag) = state.download.lock().unwrap().as_ref() {
        flag.store(true, Ordering::SeqCst);
    }
}

#[tauri::command]
fn hub_discard(state: State<'_, AppState>, id: String) -> Reply<()> {
    reply(hub::discard(&models_dir(&state.layout), &id))
}

/// Open a fixed https URL in the default browser (never a value taken verbatim from the UI).
fn open_external(url: &str) -> Result<()> {
    ensure!(
        url.starts_with("https://"),
        "only https links can be opened"
    );
    let mut command = supervisor::process(&tool("explorer.exe", &[url]));
    command.spawn().context("opening the browser")?;
    Ok(())
}

#[tauri::command]
fn open_model_page(repo: String) -> Reply<()> {
    reply(hub::repo_url(&repo).and_then(|url| open_external(&url)))
}

#[tauri::command]
fn open_release_page(state: State<'_, AppState>) -> Reply<()> {
    let tag = state
        .update
        .lock()
        .unwrap()
        .update
        .as_ref()
        .map(|u| u.tag.clone());
    let url = match tag {
        Some(tag) => format!(
            "https://github.com/{}/releases/tag/{tag}",
            update::REPOSITORY
        ),
        None => format!("https://github.com/{}/releases", update::REPOSITORY),
    };
    reply(open_external(&url))
}

#[tauri::command]
fn reveal_model(state: State<'_, AppState>, id: String) -> Reply<()> {
    reply((|| {
        let path = state
            .library
            .lock()
            .unwrap()
            .get(&id)
            .map(|r| r.path.clone())
            .context("unknown model")?;
        ensure!(
            Path::new(&path).is_dir(),
            "the model folder no longer exists"
        );
        let mut command = supervisor::process(&tool("explorer.exe", &[&path]));
        command.spawn().context("opening File Explorer")?;
        Ok(())
    })())
}

/// Models downloaded into the app's own storage are deleted from disk; imported folders
/// are only removed from the library (the user's files are left alone).
#[tauri::command]
fn delete_model(state: State<'_, AppState>, id: String) -> Reply<()> {
    reply((|| {
        let loaded = state
            .engine
            .lock()
            .unwrap()
            .as_ref()
            .map(|l| l.model_id.clone());
        ensure!(
            loaded.as_deref() != Some(id.as_str()),
            "unload the model before deleting it"
        );
        let path = state
            .library
            .lock()
            .unwrap()
            .get(&id)
            .map(|r| r.path.clone())
            .context("unknown model")?;
        let models = std::fs::canonicalize(models_dir(&state.layout)).ok();
        let folder = std::fs::canonicalize(&path).ok();
        if let (Some(models), Some(folder)) = (models, folder)
            && folder.parent() == Some(models.as_path())
        {
            std::fs::remove_dir_all(&folder)?;
        }
        state.library.lock().unwrap().remove(&id)
    })())
}

fn set_update(app: &AppHandle, change: impl FnOnce(&mut UpdateStatus)) {
    let state = app.state::<AppState>();
    let snapshot = {
        let mut status = state.update.lock().unwrap();
        change(&mut status);
        status.clone()
    };
    let _ = app.emit("update-state", snapshot);
}

/// Check GitHub, then download and verify a newer installer in the background.
fn refresh_update(app: &AppHandle) {
    let state = app.state::<AppState>();
    if state.updating.swap(true, Ordering::SeqCst) {
        return;
    }
    let current = current_release(app);
    let channel = *state.channel.lock().unwrap();
    let dir = state.layout.data_dir.join("updates");
    set_update(app, |s| {
        s.phase = "checking";
        s.current = current.clone();
        s.channel = channel;
        s.error = None;
    });
    let http = Curl::system(None);
    let outcome = update::check(&http, &current, channel).and_then(|found| {
        let Some(found) = found else {
            set_update(app, |s| {
                s.phase = "current";
                s.update = None;
            });
            return Ok(());
        };
        set_update(app, |s| {
            s.phase = "downloading";
            s.update = Some(found.clone());
            s.done = 0;
            s.total = found.size;
        });
        let mut last = std::time::Instant::now();
        let path = update::download(
            &http,
            &dir,
            &found,
            &AtomicBool::new(false),
            &mut |done, total| {
                if last.elapsed() >= Duration::from_millis(500) {
                    last = std::time::Instant::now();
                    set_update(app, |s| {
                        s.done = done;
                        s.total = total;
                    });
                }
            },
        )?;
        set_update(app, |s| {
            s.phase = "ready";
            s.done = found.size;
            s.installer = Some(path);
        });
        Ok(())
    });
    if let Err(error) = outcome {
        set_update(app, |s| {
            s.phase = "failed";
            s.error = Some(format!("{error:#}"));
        });
    }
    state.updating.store(false, Ordering::SeqCst);
}

#[tauri::command]
fn update_status(state: State<'_, AppState>) -> UpdateStatus {
    state.update.lock().unwrap().clone()
}

#[tauri::command]
fn update_check(app: AppHandle) {
    std::thread::spawn(move || refresh_update(&app));
}

/// Stable or beta (Mac "Update channel"); saved, then checked right away.
#[tauri::command]
fn set_update_channel(app: AppHandle, channel: Channel) -> Reply<()> {
    reply((|| {
        let state = app.state::<AppState>();
        let name = if channel == Channel::Beta {
            "beta"
        } else {
            "stable"
        };
        std::fs::write(channel_file(&state.layout), name)?;
        *state.channel.lock().unwrap() = channel;
        set_update(&app, |s| s.channel = channel);
        std::thread::spawn(move || refresh_update(&app));
        Ok(())
    })())
}

/// Run the verified installer (passive, relaunches MLXL3) and quit so it can replace the app.
#[tauri::command]
fn update_install(app: AppHandle) -> Reply<()> {
    reply((|| {
        let state = app.state::<AppState>();
        let installer = {
            let status = state.update.lock().unwrap();
            ensure!(status.phase == "ready", "the update is not ready yet");
            status
                .installer
                .clone()
                .context("the update is not ready yet")?
        };
        let found = state
            .update
            .lock()
            .unwrap()
            .update
            .clone()
            .context("no update")?;
        ensure!(
            hub::sha256_file(&installer)? == found.sha256,
            "the downloaded installer changed on disk; check again"
        );
        stop_engine(&state);
        std::process::Command::new(&installer)
            .args(update::install_command(&installer).args)
            .spawn()
            .context("starting the installer")?;
        app.exit(0);
        Ok(())
    })())
}

fn with_documents<T>(
    state: &AppState,
    action: impl FnOnce(&mut documents::Library) -> Result<T>,
) -> Result<T> {
    let mut guard = state.documents.lock().unwrap();
    if guard.is_none() {
        *guard = Some(documents::Library::open(
            &state.layout.data_dir.join("documents"),
        )?);
    }
    action(guard.as_mut().unwrap())
}

#[tauri::command]
fn documents_list(state: State<'_, AppState>, project_id: String) -> Reply<Vec<documents::Entry>> {
    reply(with_documents(&state, |library| {
        Ok(library.list(&project_id))
    }))
}

/// Add files (or a whole folder) the user picks to a project. `None` when cancelled.
#[tauri::command]
async fn documents_add(
    app: AppHandle,
    project_id: String,
    folder: bool,
) -> Reply<Option<documents::AddReport>> {
    use tauri_plugin_dialog::DialogExt;
    tauri::async_runtime::spawn_blocking(move || {
        let dialog = app.dialog().file();
        let picked: Vec<PathBuf> = if folder {
            dialog
                .blocking_pick_folder()
                .and_then(|p| p.into_path().ok())
                .into_iter()
                .collect()
        } else {
            let mut extensions: Vec<&str> = extract::TEXT_EXTENSIONS.to_vec();
            extensions.push("pdf");
            dialog
                .add_filter("documents", &extensions)
                .blocking_pick_files()
                .unwrap_or_default()
                .into_iter()
                .filter_map(|p| p.into_path().ok())
                .collect()
        };
        if picked.is_empty() {
            return Ok(None);
        }
        let state = app.state::<AppState>();
        with_documents(&state, |library| {
            Ok(Some(library.add_paths(&project_id, &picked)))
        })
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(reply)
}

#[tauri::command]
fn documents_remove(state: State<'_, AppState>, id: String) -> Reply<()> {
    reply(with_documents(&state, |library| library.remove(&id)))
}

#[tauri::command]
fn documents_remove_project(state: State<'_, AppState>, project_id: String) -> Reply<usize> {
    reply(with_documents(&state, |library| {
        library.remove_project(&project_id)
    }))
}

#[tauri::command]
async fn documents_search(
    app: AppHandle,
    project_id: String,
    query: String,
    limit: usize,
) -> Reply<Vec<documents::Excerpt>> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        with_documents(&state, |library| library.search(&project_id, &query, limit))
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(reply)
}

/// Text per attached file, as on the Mac (the composer also bounds the total).
const MAX_ATTACHMENT_TEXT: usize = 256 * 1024;
const MAX_ATTACHMENTS: usize = 8;

#[derive(Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Attachment {
    Text(extract::Extracted),
    Image(extract::Image),
    Error { file_name: String, error: String },
}

fn read_attachment(path: &Path) -> Attachment {
    let outcome = if extract::is_image(path) {
        extract::read_image(path).map(Attachment::Image)
    } else {
        extract::extract(path, MAX_ATTACHMENT_TEXT).map(Attachment::Text)
    };
    outcome.unwrap_or_else(|error| Attachment::Error {
        file_name: path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default(),
        error: error.to_string(),
    })
}

/// Files dropped on the window (paths come from the webview drag-drop event).
#[tauri::command]
async fn read_attachments(paths: Vec<String>) -> Reply<Vec<Attachment>> {
    tauri::async_runtime::spawn_blocking(move || {
        paths
            .iter()
            .take(MAX_ATTACHMENTS)
            .map(|p| read_attachment(Path::new(p)))
            .collect()
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
async fn pick_attachments(app: AppHandle) -> Reply<Vec<Attachment>> {
    use tauri_plugin_dialog::DialogExt;
    tauri::async_runtime::spawn_blocking(move || {
        let mut extensions: Vec<&str> = extract::TEXT_EXTENSIONS.to_vec();
        extensions.extend(["pdf", "png", "jpg", "jpeg", "webp"]);
        app.dialog()
            .file()
            .add_filter("attachments", &extensions)
            .blocking_pick_files()
            .unwrap_or_default()
            .into_iter()
            .filter_map(|p| p.into_path().ok())
            .take(MAX_ATTACHMENTS)
            .map(|p| read_attachment(&p))
            .collect()
    })
    .await
    .map_err(|e| e.to_string())
}

/// A link clicked in an answer: http(s) only, opened in the default browser.
#[tauri::command]
fn open_url(url: String) -> Reply<()> {
    reply((|| {
        let lower = url.to_ascii_lowercase();
        ensure!(
            (lower.starts_with("https://") || lower.starts_with("http://"))
                && url.len() <= 2048
                && !url
                    .chars()
                    .any(|c| c.is_whitespace() || c.is_control() || c == '"'),
            "only http and https links can be opened"
        );
        let mut command = supervisor::process(&tool("explorer.exe", &[&url]));
        command.spawn().context("opening the browser")?;
        Ok(())
    })())
}

/// Server mode's view of the app: the loaded model and its engine.
struct AppEngine(AppHandle);

impl server::Engine for AppEngine {
    fn model(&self) -> Option<String> {
        let state = self.0.state::<AppState>();
        let id = state.engine.lock().unwrap().as_ref()?.model_id.clone();
        state
            .library
            .lock()
            .unwrap()
            .get(&id)
            .map(|r| r.name.clone())
    }

    fn submit(
        &self,
        request_id: &str,
        messages: &[Message],
        sampling: &Sampling,
        tools: &[Value],
    ) -> Result<mpsc::Receiver<Value>> {
        let state = self.0.state::<AppState>();
        let (sender, receiver) = mpsc::channel();
        state
            .routes
            .lock()
            .unwrap()
            .insert(request_id.to_owned(), sender);
        let sent = send(
            &state,
            protocol::generate_request_with_tools(request_id, messages, sampling, tools),
        );
        if let Err(error) = sent {
            state.routes.lock().unwrap().remove(request_id);
            return Err(error);
        }
        Ok(receiver)
    }

    fn cancel(&self, request_id: &str) {
        let state = self.0.state::<AppState>();
        let _ = send(&state, protocol::control_request("cancel", request_id));
    }
}

fn server_file(layout: &Layout) -> PathBuf {
    layout.data_dir.join("server.json")
}

#[derive(Serialize)]
struct ServerStatus {
    running: bool,
    address: Option<String>,
    config: server::Config,
}

fn server_status_of(state: &AppState) -> ServerStatus {
    let config = std::fs::read(server_file(&state.layout))
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default();
    let address = state
        .server
        .lock()
        .unwrap()
        .as_ref()
        .map(|h| h.address.to_string());
    ServerStatus {
        running: address.is_some(),
        address,
        config,
    }
}

#[tauri::command]
fn server_status(state: State<'_, AppState>) -> ServerStatus {
    server_status_of(&state)
}

#[tauri::command]
fn server_start(app: AppHandle, config: server::Config) -> Reply<ServerStatus> {
    reply((|| {
        let state = app.state::<AppState>();
        config.address()?;
        if let Some(old) = state.server.lock().unwrap().take() {
            old.stop();
        }
        std::fs::write(
            server_file(&state.layout),
            serde_json::to_vec_pretty(&config)?,
        )?;
        let emitter = app.clone();
        let handle = server::start(
            &config,
            Arc::new(AppEngine(app.clone())),
            Arc::new(move |line| {
                let _ = emitter.emit("server-log", line);
            }),
        )?;
        *state.server.lock().unwrap() = Some(handle);
        Ok(server_status_of(&state))
    })())
}

#[tauri::command]
fn server_stop(state: State<'_, AppState>) -> ServerStatus {
    if let Some(handle) = state.server.lock().unwrap().take() {
        handle.stop();
    }
    server_status_of(&state)
}

fn stop_engine(state: &AppState) {
    let loaded = state.engine.lock().unwrap().take();
    if let Some(loaded) = loaded {
        loaded.engine.shutdown(SHUTDOWN_GRACE);
    }
}

#[tauri::command]
async fn load_model(
    app: AppHandle,
    id: String,
    backend: Backend,
    context_length: u32,
) -> Reply<()> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        stop_engine(&state);
        let record = state
            .library
            .lock()
            .unwrap()
            .get(&id)
            .cloned()
            .context("unknown model")?;
        let marker = runtime::installed(&state.layout, backend)
            .context("install the engine runtime first")?;
        let command = runtime::bridge_command(
            backend,
            &state.layout,
            marker.wsl.as_ref(),
            &record.path,
            context_length,
        )?;
        let epoch = state.epoch.fetch_add(1, Ordering::SeqCst) + 1;
        let handle = app.clone();
        let engine = Engine::spawn(&command, move |message| match message {
            EngineMessage::Event(event) => {
                // Rounds of server requests and chat turns go to their turn driver, which
                // emits the turn-level events itself.
                if let Some(id) = event.request_id.as_deref() {
                    let state = handle.state::<AppState>();
                    let mut routes = state.routes.lock().unwrap();
                    if let Some(route) = routes.get(id) {
                        let _ = route.send(event.payload.clone());
                        if event.is_terminal_for_request() {
                            routes.remove(id);
                        }
                        return;
                    }
                }
                let _ = handle.emit(
                    "engine-event",
                    json!({"epoch": epoch, "event": event.payload}),
                );
            }
            EngineMessage::Malformed(line) => {
                let _ = handle.emit("engine-log", json!({"epoch": epoch, "line": line}));
            }
            EngineMessage::Exited { code, stderr_tail } => {
                let state = handle.state::<AppState>();
                // Dropping the senders ends every waiting server request with "engine exited".
                state.routes.lock().unwrap().clear();
                let mut engine = state.engine.lock().unwrap();
                if engine.as_ref().is_some_and(|l| l.epoch == epoch) {
                    engine.take();
                }
                let _ = handle.emit(
                    "engine-exited",
                    json!({"epoch": epoch, "code": code, "stderr_tail": stderr_tail}),
                );
            }
        })?;
        *state.engine.lock().unwrap() = Some(Loaded {
            epoch,
            model_id: id,
            context: context_length,
            engine,
        });
        let _ = app.emit("engine-started", json!({"epoch": epoch}));
        Ok(())
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(reply)
}

fn send(state: &AppState, line: Result<Vec<u8>>) -> Result<()> {
    let line = line?;
    let engine = state.engine.lock().unwrap();
    engine
        .as_ref()
        .context("no model is loaded")?
        .engine
        .send(&line)
}

/// Options of one chat turn beyond the conversation itself.
#[derive(serde::Deserialize, Default)]
#[serde(rename_all = "camelCase", default)]
struct TurnOptions {
    /// Validated JSON reply: the raw schema text.
    schema: Option<String>,
    /// Offer the connected MCP tools.
    mcp: bool,
    /// Project allow-list of MCP tool names (`None` = all).
    allowed_tools: Option<Vec<String>>,
    /// Agent mode: the task whose workspace tools are offered.
    agent_task: Option<String>,
}

#[tauri::command]
fn generate(
    app: AppHandle,
    state: State<'_, AppState>,
    request_id: String,
    messages: Vec<Message>,
    sampling: Sampling,
    options: Option<TurnOptions>,
) -> Reply<()> {
    reply((|| {
        let options = options.unwrap_or_default();
        let (epoch, context) = {
            let engine = state.engine.lock().unwrap();
            let loaded = engine.as_ref().context("no model is loaded")?;
            (loaded.epoch, loaded.context as usize)
        };
        let (tools, toolbox): (Vec<Value>, Option<Arc<dyn turn::ToolBox>>) =
            if let Some(task) = &options.agent_task {
                let tools = state
                    .agents
                    .lock()
                    .unwrap()
                    .get(task)
                    .cloned()
                    .context("open the Agent task first")?;
                let offered = tools.0.lock().unwrap().tools();
                (offered, Some(tools))
            } else if options.mcp {
                // Mac: with nothing connected the turn is a plain generation.
                let offered = state.mcp.lock().unwrap().chat_tools();
                (
                    offered,
                    Some(Arc::new(mcp::SharedManager(state.mcp.clone()))),
                )
            } else {
                (vec![], None)
            };
        let allowed = if options.agent_task.is_some() {
            None
        } else {
            options.allowed_tools
        };
        let turn = turn::start(
            Arc::new(AppEngine(app.clone())),
            toolbox,
            &request_id,
            turn::TurnRequest {
                messages,
                sampling,
                schema: options.schema.filter(|s| !s.trim().is_empty()),
                tools,
                allowed,
                context_limit: context,
            },
        )?;
        state
            .turns
            .lock()
            .unwrap()
            .insert(request_id.clone(), turn.cancel.clone());
        let handle = app.clone();
        std::thread::spawn(move || {
            for event in turn.events {
                let terminal =
                    matches!(event["type"].as_str(), Some("done" | "cancelled" | "error"));
                let _ = handle.emit("engine-event", json!({"epoch": epoch, "event": event}));
                if terminal {
                    break;
                }
            }
            handle
                .state::<AppState>()
                .turns
                .lock()
                .unwrap()
                .remove(&request_id);
        });
        Ok(())
    })())
}

#[tauri::command]
fn cancel(state: State<'_, AppState>, request_id: String) -> Reply<()> {
    if let Some(flag) = state.turns.lock().unwrap().get(&request_id) {
        flag.store(true, Ordering::SeqCst);
        return Ok(());
    }
    reply(send(
        &state,
        protocol::control_request("cancel", &request_id),
    ))
}

fn mcp_registry(state: &AppState) -> PathBuf {
    state.layout.data_dir.join("library.json")
}

#[derive(Serialize)]
struct McpServer {
    name: String,
    config: mcp::ServerConfig,
    connected: bool,
    error: Option<String>,
    tools: Vec<String>,
}

#[derive(Serialize)]
struct McpStatus {
    enabled: bool,
    servers: Vec<McpServer>,
    error: Option<String>,
}

fn mcp_status_of(state: &AppState) -> McpStatus {
    let registry = mcp_registry(state);
    let manager = state.mcp.lock().unwrap();
    let (configs, error) = match mcp::list_servers(&registry) {
        Ok(configs) => (configs, manager.errors.get("configuration").cloned()),
        Err(error) => (Default::default(), Some(format!("{error:#}"))),
    };
    let servers = configs
        .into_iter()
        .map(|(name, config)| {
            let tools: Vec<String> = manager
                .tools
                .values()
                .filter(|t| t.server == name)
                .map(|t| t.public_name.clone())
                .collect();
            McpServer {
                connected: manager.enabled && config.enabled && !manager.errors.contains_key(&name),
                error: manager.errors.get(&name).cloned(),
                tools,
                name,
                config,
            }
        })
        .collect();
    McpStatus {
        enabled: manager.enabled,
        servers,
        error,
    }
}

#[tauri::command]
fn mcp_status(state: State<'_, AppState>) -> McpStatus {
    mcp_status_of(&state)
}

/// Enable/disable MCP, or reconnect every server (`refresh`). Connecting can take seconds.
#[tauri::command]
async fn mcp_set_enabled(app: AppHandle, enabled: bool, refresh: bool) -> Reply<McpStatus> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let registry = mcp_registry(&state);
        state
            .mcp
            .lock()
            .unwrap()
            .set_enabled(&registry, enabled, refresh);
        mcp_status_of(&state)
    })
    .await
    .map_err(|e| e.to_string())
}

#[tauri::command]
fn mcp_save(
    state: State<'_, AppState>,
    name: String,
    config: mcp::ServerConfig,
) -> Reply<McpStatus> {
    reply(mcp::save_server(&mcp_registry(&state), &name, config).map(|()| mcp_status_of(&state)))
}

#[tauri::command]
fn mcp_remove(state: State<'_, AppState>, name: String) -> Reply<McpStatus> {
    reply(mcp::remove_server(&mcp_registry(&state), &name).map(|()| mcp_status_of(&state)))
}

fn agent_dir(state: &AppState, task: &str) -> Result<PathBuf> {
    ensure!(
        !task.is_empty()
            && task.len() <= 64
            && task
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'),
        "invalid Agent task id"
    );
    Ok(state.layout.data_dir.join("agent").join(task))
}

fn agent_tools(state: &AppState, task: &str) -> Result<Arc<agent::AgentTools>> {
    let mut agents = state.agents.lock().unwrap();
    if let Some(tools) = agents.get(task) {
        return Ok(tools.clone());
    }
    let workspace = agent::Workspace::open(&agent_dir(state, task)?)?;
    let tools = Arc::new(agent::AgentTools(Mutex::new(workspace)));
    agents.insert(task.to_owned(), tools.clone());
    Ok(tools)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AgentInfo {
    task: String,
    source: Option<PathBuf>,
    permissions: agent::Permissions,
    files: usize,
    skipped_files: usize,
    system_prompt: String,
    sandboxed: bool,
}

fn agent_info(tools: &agent::AgentTools, task: &str) -> Result<AgentInfo> {
    let workspace = tools.0.lock().unwrap();
    let state = workspace.state();
    Ok(AgentInfo {
        task: task.to_owned(),
        source: state.source.clone(),
        permissions: state.permissions,
        files: workspace.list()?.len(),
        skipped_files: state.skipped_files,
        system_prompt: agent::system_prompt(&state.permissions),
        sandboxed: false,
    })
}

#[tauri::command]
fn agent_open(state: State<'_, AppState>, task: String) -> Reply<AgentInfo> {
    reply(agent_tools(&state, &task).and_then(|tools| agent_info(&tools, &task)))
}

/// Choose the project folder and copy it into the task workspace.
#[tauri::command]
async fn agent_import(app: AppHandle, task: String) -> Reply<Option<AgentInfo>> {
    use tauri_plugin_dialog::DialogExt;
    tauri::async_runtime::spawn_blocking(move || {
        let Some(folder) = app
            .dialog()
            .file()
            .blocking_pick_folder()
            .and_then(|p| p.into_path().ok())
        else {
            return Ok(None);
        };
        let state = app.state::<AppState>();
        let tools = agent_tools(&state, &task)?;
        tools.0.lock().unwrap().import(&folder)?;
        agent_info(&tools, &task).map(Some)
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(reply)
}

#[tauri::command]
fn agent_set_permissions(
    state: State<'_, AppState>,
    task: String,
    permissions: agent::Permissions,
) -> Reply<AgentInfo> {
    reply((|| {
        let tools = agent_tools(&state, &task)?;
        tools.0.lock().unwrap().set_permissions(permissions)?;
        agent_info(&tools, &task)
    })())
}

#[tauri::command]
fn agent_changes(state: State<'_, AppState>, task: String) -> Reply<Vec<agent::Change>> {
    reply(agent_tools(&state, &task).and_then(|tools| tools.0.lock().unwrap().changes()))
}

#[tauri::command]
fn agent_apply(state: State<'_, AppState>, task: String) -> Reply<usize> {
    reply(agent_tools(&state, &task).and_then(|tools| tools.0.lock().unwrap().apply()))
}

/// Delete a task's private workspace (the source project is never touched).
#[tauri::command]
fn agent_discard(state: State<'_, AppState>, task: String) -> Reply<()> {
    reply((|| {
        let dir = agent_dir(&state, &task)?;
        state.agents.lock().unwrap().remove(&task);
        if dir.exists() {
            std::fs::remove_dir_all(dir)?;
        }
        Ok(())
    })())
}

#[tauri::command]
async fn unload_model(app: AppHandle) -> Reply<()> {
    tauri::async_runtime::spawn_blocking(move || stop_engine(&app.state::<AppState>()))
        .await
        .map_err(|e| e.to_string())
}

fn layout(app: &AppHandle) -> Result<Layout> {
    Ok(Layout {
        data_dir: app.path().app_local_data_dir()?,
        resources: app.path().resource_dir()?,
    })
}

fn main() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .setup(|app| {
            let layout = layout(app.handle())?;
            std::fs::create_dir_all(&layout.data_dir)?;
            let library = Library::load(layout.data_dir.join("library.json"))?;
            let channel = load_channel(&layout);
            let current = current_release(app.handle());
            app.manage(AppState {
                layout,
                library: Mutex::new(library),
                engine: Mutex::new(None),
                epoch: AtomicU64::new(0),
                busy: AtomicBool::new(false),
                download: Mutex::new(None),
                update: Mutex::new(UpdateStatus {
                    phase: "idle",
                    current,
                    channel,
                    ..UpdateStatus::default()
                }),
                updating: AtomicBool::new(false),
                channel: Mutex::new(channel),
                documents: Mutex::new(None),
                routes: Mutex::new(HashMap::new()),
                server: Mutex::new(None),
                turns: Mutex::new(HashMap::new()),
                mcp: Arc::new(Mutex::new(mcp::Manager::disabled())),
                agents: Mutex::new(HashMap::new()),
            });
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                std::thread::sleep(UPDATE_FIRST_CHECK);
                loop {
                    refresh_update(&handle);
                    std::thread::sleep(UPDATE_INTERVAL);
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            system_info,
            install_runtime,
            list_models,
            add_model,
            remove_model,
            delete_model,
            reveal_model,
            hub_search,
            hub_details,
            hub_download,
            hub_pending,
            hub_resume,
            hub_pause,
            hub_discard,
            open_model_page,
            open_release_page,
            update_status,
            update_check,
            set_update_channel,
            update_install,
            notify_done,
            save_text_file,
            open_backup_file,
            hf_account,
            hf_sign_in,
            hf_sign_out,
            documents_list,
            documents_add,
            documents_remove,
            documents_remove_project,
            documents_search,
            read_attachments,
            pick_attachments,
            open_url,
            server_status,
            server_start,
            server_stop,
            load_model,
            generate,
            cancel,
            unload_model,
            mcp_status,
            mcp_set_enabled,
            mcp_save,
            mcp_remove,
            agent_open,
            agent_import,
            agent_set_permissions,
            agent_changes,
            agent_apply,
            agent_discard
        ])
        .build(tauri::generate_context!())
        .expect("failed to build the MLXL3 window");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            let state = handle.state::<AppState>();
            if let Some(flag) = state.download.lock().unwrap().as_ref() {
                flag.store(true, Ordering::SeqCst);
            }
            if let Some(handle) = state.server.lock().unwrap().take() {
                handle.stop();
            }
            for flag in state.turns.lock().unwrap().values() {
                flag.store(true, Ordering::SeqCst);
            }
            stop_engine(&state);
        }
    });
}
