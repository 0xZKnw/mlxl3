//! MLXL3 Desktop for Windows: thin Tauri shell over `mlxl3-windows-core`.
//!
//! The engine is upstream ExLlamaV3, run by the bundled Python worker in a
//! private runtime: natively with CUDA, or inside WSL2 with ROCm for AMD GPUs.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::{Context, Result, ensure};
use mlxl3_windows_core::library::{self, Library};
use mlxl3_windows_core::protocol::{self, Message, Sampling};
use mlxl3_windows_core::runtime::{self, Layout, Marker, WslTarget};
use mlxl3_windows_core::supervisor::{self, Engine, EngineMessage};
use mlxl3_windows_core::{Backend, gpu, wsl};
use serde::Serialize;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

const PROBE_TIMEOUT: Duration = Duration::from_secs(15);
const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

struct Loaded {
    epoch: u64,
    model_id: String,
    engine: Engine,
}

struct AppState {
    layout: Layout,
    library: Mutex<Library>,
    engine: Mutex<Option<Loaded>>,
    epoch: AtomicU64,
    busy: AtomicBool,
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
    let version = app.package_info().version.to_string();
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
                "GPU check failed: {}",
                report
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown error")
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

#[tauri::command]
async fn download_model(
    app: AppHandle,
    backend: Backend,
    repo: String,
    folder: String,
) -> Reply<library::ModelRecord> {
    tauri::async_runtime::spawn_blocking(move || {
        let state = app.state::<AppState>();
        let _busy = claim(&state)?;
        let marker = runtime::installed(&state.layout, backend)
            .context("install the engine runtime first")?;
        let (directory, folder) = library::validate_download(&repo, &folder)?;
        let destination = state.layout.data_dir.join("models").join(&directory);
        let destination = destination.to_string_lossy().into_owned();
        let command = runtime::download_command(
            backend,
            &state.layout,
            marker.wsl.as_ref(),
            &repo,
            &folder,
            &destination,
        )?;
        let handle = app.clone();
        let done = supervisor::run(&command, None, move |line| {
            let _ = handle.emit("download-log", json!({"line": line}));
        })?;
        ensure!(
            done.code == Some(0),
            "download failed:\n{}",
            done.stdout.trim()
        );
        let path = PathBuf::from(&destination);
        let path = if folder.is_empty() {
            path
        } else {
            path.join(&folder)
        };
        let record = library::inspect(&path)?;
        state.library.lock().unwrap().upsert(record.clone())?;
        Ok(record)
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(reply)
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

#[tauri::command]
fn generate(
    state: State<'_, AppState>,
    request_id: String,
    messages: Vec<Message>,
    sampling: Sampling,
) -> Reply<()> {
    reply(send(
        &state,
        protocol::generate_request(&request_id, &messages, &sampling),
    ))
}

#[tauri::command]
fn cancel(state: State<'_, AppState>, request_id: String) -> Reply<()> {
    reply(send(
        &state,
        protocol::control_request("cancel", &request_id),
    ))
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
        .setup(|app| {
            let layout = layout(app.handle())?;
            std::fs::create_dir_all(&layout.data_dir)?;
            let library = Library::load(layout.data_dir.join("library.json"))?;
            app.manage(AppState {
                layout,
                library: Mutex::new(library),
                engine: Mutex::new(None),
                epoch: AtomicU64::new(0),
                busy: AtomicBool::new(false),
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            system_info,
            install_runtime,
            list_models,
            add_model,
            remove_model,
            download_model,
            load_model,
            generate,
            cancel,
            unload_model
        ])
        .build(tauri::generate_context!())
        .expect("failed to build the MLXL3 window");
    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            stop_engine(&handle.state::<AppState>());
        }
    });
}
