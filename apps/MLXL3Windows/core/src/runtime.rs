//! Private ExLlamaV3 runtimes: pinned specification, install plan, markers and launch commands.
//!
//! CUDA: a uv-managed CPython 3.12 venv under `%LOCALAPPDATA%\MLXL3\runtimes`.
//! AMD: the same layout inside a WSL2 distribution's ext4 home (a venv on
//! drvfs would be slow and lose symlinks); only the marker stays on Windows.
//! The worker package is not installed: it is loaded from the app resources
//! through `PYTHONPATH`, so an app update ships a new worker without a reinstall.

use crate::{Backend, wsl};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

pub const PYTHON_VERSION: &str = "3.12";
pub const CUDA_REQUIREMENTS: &str =
    include_str!("../../../../engines/exl3/runtime/cuda-windows.txt");
pub const ROCM_REQUIREMENTS: &str = include_str!("../../../../engines/exl3/runtime/rocm-wsl.txt");

/// Where the app keeps its data and where the installer put its resources.
#[derive(Clone, Debug)]
pub struct Layout {
    /// `%LOCALAPPDATA%\MLXL3`
    pub data_dir: PathBuf,
    /// Tauri resource directory: `engine\mlxl3_exl3`, `runtime\*.txt`, `bin\uv.exe`, `bin\uv`.
    pub resources: PathBuf,
}

/// Target of the AMD backend.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WslTarget {
    pub distro: String,
    /// `$HOME` inside the distribution.
    pub home: String,
}

pub fn requirements(backend: Backend) -> &'static str {
    match backend {
        Backend::Cuda => CUDA_REQUIREMENTS,
        Backend::RocmWsl => ROCM_REQUIREMENTS,
    }
}

/// Content-addressed runtime id: a change of pins yields a fresh runtime.
pub fn runtime_id(backend: Backend) -> String {
    let mut hash = Sha256::new();
    hash.update(format!("{backend:?}\n{PYTHON_VERSION}\n").as_bytes());
    hash.update(requirements(backend).as_bytes());
    let digest = hash.finalize();
    let hex: String = digest.iter().take(6).map(|b| format!("{b:02x}")).collect();
    let prefix = match backend {
        Backend::Cuda => "cuda",
        Backend::RocmWsl => "rocm",
    };
    format!("{prefix}-{hex}")
}

/// One external command. Arguments are passed as argv, never through a shell.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Command {
    pub label: String,
    pub program: String,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
}

fn s(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

fn cuda_python(layout: &Layout) -> PathBuf {
    layout
        .data_dir
        .join("runtimes")
        .join(runtime_id(Backend::Cuda))
        .join("Scripts")
        .join("python.exe")
}

fn wsl_root(target: &WslTarget) -> String {
    format!("{}/.local/share/mlxl3", target.home)
}

fn wsl_python(target: &WslTarget) -> String {
    format!(
        "{}/runtimes/{}/bin/python",
        wsl_root(target),
        runtime_id(Backend::RocmWsl)
    )
}

/// AMD's WSL2 GPU bridge (ROCm/librocdxg v1.2.2), bundled under `bin\` by CI
/// after a SHA-256 check. Without it ROCm inside WSL2 enumerates no GPU.
pub const ROCDXG_DEB: &str = "rocdxg-roct_1.2.2_amd64.deb";

/// Environment every ROCm worker gets inside WSL2: DXG device discovery for the
/// HSA runtime, and the directories holding librocdxg and DXCore.
pub const ROCM_WSL_ENV: [&str; 2] = [
    "HSA_ENABLE_DXG_DETECTION=1",
    "LD_LIBRARY_PATH=/opt/rocm/lib:/usr/lib/wsl/lib",
];

fn wsl_exec(
    target: &WslTarget,
    env: &[String],
    program: String,
    args: &[String],
) -> Result<(String, Vec<String>)> {
    wsl::validate_distro(&target.distro)?;
    let mut argv = vec![
        "-d".into(),
        target.distro.clone(),
        "--exec".into(),
        "/usr/bin/env".into(),
    ];
    argv.extend(env.iter().cloned());
    argv.push(program);
    argv.extend(args.iter().cloned());
    Ok(("wsl.exe".into(), argv))
}

/// Commands that create the runtime, in order; the last one is the probe.
pub fn install_plan(
    backend: Backend,
    layout: &Layout,
    wsl_target: Option<&WslTarget>,
) -> Result<Vec<Command>> {
    let id = runtime_id(backend);
    match backend {
        Backend::Cuda => {
            let uv = s(&layout.resources.join("bin").join("uv.exe"));
            let venv = layout.data_dir.join("runtimes").join(&id);
            let env = vec![
                ("UV_CACHE_DIR".into(), s(&layout.data_dir.join("uv-cache"))),
                (
                    "UV_PYTHON_INSTALL_DIR".into(),
                    s(&layout.data_dir.join("python")),
                ),
                ("UV_NO_CONFIG".into(), "1".into()),
            ];
            let requirements = s(&layout.resources.join("runtime").join("cuda-windows.txt"));
            Ok(vec![
                Command {
                    label: "Python 3.12".into(),
                    program: uv.clone(),
                    args: strings(&[
                        "venv",
                        "--python",
                        PYTHON_VERSION,
                        "--python-preference",
                        "only-managed",
                        "--clear",
                        &s(&venv),
                    ]),
                    env: env.clone(),
                },
                Command {
                    label: "PyTorch CUDA 12.8 + ExLlamaV3".into(),
                    program: uv,
                    args: strings(&[
                        "pip",
                        "install",
                        "--python",
                        &s(&cuda_python(layout)),
                        "-r",
                        &requirements,
                    ]),
                    env,
                },
                probe_command(backend, layout, None)?,
            ])
        }
        Backend::RocmWsl => {
            let target = wsl_target.context("the AMD backend needs a WSL2 distribution")?;
            let uv =
                wsl::to_wsl_path(&s(&layout.resources.join("bin").join("uv")), &target.distro)?;
            let requirements = wsl::to_wsl_path(
                &s(&layout.resources.join("runtime").join("rocm-wsl.txt")),
                &target.distro,
            )?;
            let root = wsl_root(target);
            let env = vec![
                format!("UV_CACHE_DIR={}/.cache/mlxl3-uv", target.home),
                format!("UV_PYTHON_INSTALL_DIR={root}/python"),
                "UV_NO_CONFIG=1".into(),
            ];
            let venv = format!("{root}/runtimes/{id}");
            let deb = wsl::to_wsl_path(
                &s(&layout.resources.join("bin").join(ROCDXG_DEB)),
                &target.distro,
            )?;
            wsl::validate_distro(&target.distro)?;
            let bridge_args = strings(&[
                "-d",
                &target.distro,
                "-u",
                "root",
                "--exec",
                "/usr/bin/dpkg",
                "-i",
                &deb,
            ]);
            let (program, venv_args) = wsl_exec(
                target,
                &env,
                uv.clone(),
                &strings(&[
                    "venv",
                    "--python",
                    PYTHON_VERSION,
                    "--python-preference",
                    "only-managed",
                    "--clear",
                    &venv,
                ]),
            )?;
            let (_, pip_args) = wsl_exec(
                target,
                &env,
                uv,
                &strings(&[
                    "pip",
                    "install",
                    "--python",
                    &wsl_python(target),
                    "--index-strategy",
                    "unsafe-best-match",
                    "-r",
                    &requirements,
                ]),
            )?;
            Ok(vec![
                Command {
                    label: "AMD WSL GPU bridge (librocdxg 1.2.2, installed as root)".into(),
                    program: "wsl.exe".into(),
                    args: bridge_args,
                    env: vec![],
                },
                Command {
                    label: "Python 3.12 (WSL2)".into(),
                    program: program.clone(),
                    args: venv_args,
                    env: vec![],
                },
                Command {
                    label: "PyTorch ROCm 10 + ExLlamaV3".into(),
                    program,
                    args: pip_args,
                    env: vec![],
                },
                probe_command(backend, layout, Some(target))?,
            ])
        }
    }
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn worker_command(
    backend: Backend,
    layout: &Layout,
    wsl_target: Option<&WslTarget>,
    label: &str,
    worker_args: &[String],
) -> Result<Command> {
    let engine = layout.resources.join("engine");
    let mut args = strings(&["-s", "-X", "utf8", "-m", "mlxl3_exl3"]);
    args.extend(worker_args.iter().cloned());
    match backend {
        Backend::Cuda => Ok(Command {
            label: label.into(),
            program: s(&cuda_python(layout)),
            args,
            env: vec![
                ("PYTHONPATH".into(), s(&engine)),
                ("PYTHONUNBUFFERED".into(), "1".into()),
                ("PYTHONNOUSERSITE".into(), "1".into()),
            ],
        }),
        Backend::RocmWsl => {
            let target = wsl_target.context("the AMD backend needs a WSL2 distribution")?;
            let env = vec![
                format!(
                    "PYTHONPATH={}",
                    wsl::to_wsl_path(&s(&engine), &target.distro)?
                ),
                "PYTHONUNBUFFERED=1".into(),
                "PYTHONNOUSERSITE=1".into(),
            ]
            .into_iter()
            .chain(ROCM_WSL_ENV.iter().map(|v| (*v).to_owned()))
            .collect::<Vec<_>>();
            let (program, argv) = wsl_exec(target, &env, wsl_python(target), &args)?;
            Ok(Command {
                label: label.into(),
                program,
                args: argv,
                env: vec![],
            })
        }
    }
}

pub fn probe_command(
    backend: Backend,
    layout: &Layout,
    wsl_target: Option<&WslTarget>,
) -> Result<Command> {
    worker_command(
        backend,
        layout,
        wsl_target,
        "GPU check",
        &strings(&["probe"]),
    )
}

/// The resident engine for one model. `model_path` is the Windows path the user chose.
pub fn bridge_command(
    backend: Backend,
    layout: &Layout,
    wsl_target: Option<&WslTarget>,
    model_path: &str,
    context_length: u32,
) -> Result<Command> {
    let model = match (backend, wsl_target) {
        (Backend::RocmWsl, Some(target)) => wsl::to_wsl_path(model_path, &target.distro)?,
        _ => model_path.to_owned(),
    };
    let args = vec![
        "bridge".into(),
        model,
        "--context-length".into(),
        context_length.to_string(),
    ];
    worker_command(backend, layout, wsl_target, "engine", &args)
}

pub fn download_command(
    backend: Backend,
    layout: &Layout,
    wsl_target: Option<&WslTarget>,
    repo: &str,
    folder: &str,
    destination: &str,
) -> Result<Command> {
    let destination = match (backend, wsl_target) {
        (Backend::RocmWsl, Some(target)) => wsl::to_wsl_path(destination, &target.distro)?,
        _ => destination.to_owned(),
    };
    let args = strings(&["download", repo, &destination, "--folder", folder]);
    worker_command(backend, layout, wsl_target, "download", &args)
}

/// Written after a successful probe; its absence means "not installed".
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Marker {
    pub id: String,
    pub backend: Backend,
    pub wsl: Option<WslTarget>,
    pub probe: serde_json::Value,
}

pub fn marker_path(layout: &Layout, backend: Backend) -> PathBuf {
    layout
        .data_dir
        .join("runtimes")
        .join(format!("{}.json", runtime_id(backend)))
}

pub fn write_marker(layout: &Layout, marker: &Marker) -> Result<()> {
    ensure!(
        marker.id == runtime_id(marker.backend),
        "marker id does not match the pinned runtime"
    );
    ensure!(
        marker.probe.get("ok").and_then(serde_json::Value::as_bool) == Some(true),
        "the GPU check did not succeed"
    );
    crate::library::write_atomic(
        &marker_path(layout, marker.backend),
        &serde_json::to_vec_pretty(marker)?,
    )
}

/// The installed runtime for `backend`, if its marker matches the current pins.
pub fn installed(layout: &Layout, backend: Backend) -> Option<Marker> {
    let bytes = std::fs::read(marker_path(layout, backend)).ok()?;
    let marker: Marker = serde_json::from_slice(&bytes).ok()?;
    let valid = marker.id == runtime_id(backend)
        && marker.backend == backend
        && (backend != Backend::Cuda || cuda_python(layout).is_file())
        && (backend != Backend::RocmWsl || marker.wsl.is_some());
    valid.then_some(marker)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layout() -> Layout {
        Layout {
            data_dir: PathBuf::from(r"C:\Users\Me\AppData\Local\MLXL3"),
            resources: PathBuf::from(r"C:\Program Files\MLXL3"),
        }
    }

    fn target() -> WslTarget {
        WslTarget {
            distro: "Ubuntu-24.04".into(),
            home: "/home/me".into(),
        }
    }

    #[test]
    fn ids_follow_pins() {
        assert!(runtime_id(Backend::Cuda).starts_with("cuda-"));
        assert!(runtime_id(Backend::RocmWsl).starts_with("rocm-"));
        assert_eq!(runtime_id(Backend::Cuda).len(), "cuda-".len() + 12);
        assert_ne!(
            runtime_id(Backend::Cuda)[5..],
            runtime_id(Backend::RocmWsl)[5..]
        );
    }

    #[test]
    fn requirements_pin_binary_wheels() {
        for line in CUDA_REQUIREMENTS.lines().filter(|l| l.contains(" @ ")) {
            assert!(line.contains("#sha256="), "{line}");
            assert!(
                line.contains("cp312") && line.contains("win_amd64"),
                "{line}"
            );
        }
        assert_eq!(
            CUDA_REQUIREMENTS
                .lines()
                .filter(|l| l.contains(" @ "))
                .count(),
            3
        );
        assert!(
            CUDA_REQUIREMENTS.contains("torch2.10.0")
                && CUDA_REQUIREMENTS.contains("torch-2.10.0%2Bcu128")
        );
        let rocm_wheel = ROCM_REQUIREMENTS
            .lines()
            .find(|l| l.starts_with("exllamav3 @"))
            .unwrap();
        assert!(
            rocm_wheel.contains("rocm10.0.torch2.13.0-cp312-cp312-linux_x86_64")
                && rocm_wheel.contains("#sha256=")
        );
        assert!(ROCM_REQUIREMENTS.contains("torch==2.13.0+rocm10.0.0"));
    }

    #[test]
    fn cuda_plan_is_argv_only() {
        let plan = install_plan(Backend::Cuda, &layout(), None).unwrap();
        assert_eq!(plan.len(), 3);
        assert!(plan[0].program.ends_with("uv.exe"));
        assert!(plan[1].args.iter().any(|a| a.ends_with("cuda-windows.txt")));
        assert!(
            plan[2]
                .args
                .ends_with(&["mlxl3_exl3".into(), "probe".into()])
        );
        assert!(
            plan[2]
                .env
                .iter()
                .any(|(k, v)| k == "PYTHONPATH" && v.ends_with("engine"))
        );
    }

    #[test]
    fn rocm_plan_runs_inside_wsl() {
        assert!(install_plan(Backend::RocmWsl, &layout(), None).is_err());
        let plan = install_plan(Backend::RocmWsl, &layout(), Some(&target())).unwrap();
        assert!(plan.iter().all(|command| command.program == "wsl.exe"));
        // Step 1: AMD's DXG bridge, installed by dpkg as root from the bundled package.
        assert_eq!(
            plan[0].args,
            [
                "-d",
                "Ubuntu-24.04",
                "-u",
                "root",
                "--exec",
                "/usr/bin/dpkg",
                "-i",
                "/mnt/c/Program Files/MLXL3/bin/rocdxg-roct_1.2.2_amd64.deb"
            ]
        );
        for command in &plan[1..] {
            assert_eq!(
                &command.args[..4],
                ["-d", "Ubuntu-24.04", "--exec", "/usr/bin/env"]
            );
        }
        assert!(
            plan[1]
                .args
                .contains(&"/mnt/c/Program Files/MLXL3/bin/uv".to_owned())
        );
        assert!(
            plan[2]
                .args
                .contains(&"/mnt/c/Program Files/MLXL3/runtime/rocm-wsl.txt".to_owned())
        );
        assert!(
            plan[2]
                .args
                .iter()
                .any(|a| a.starts_with("/home/me/.local/share/mlxl3/runtimes/rocm-"))
        );
        // The probe (and every worker) gets DXG discovery and the bridge's library path.
        let probe = plan.last().unwrap();
        for value in ROCM_WSL_ENV {
            assert!(probe.args.contains(&value.to_owned()), "{value}");
        }
        let program = probe
            .args
            .iter()
            .position(|a| a.ends_with("/bin/python"))
            .unwrap();
        let env_end = probe
            .args
            .iter()
            .position(|a| a.starts_with("HSA_ENABLE"))
            .unwrap();
        assert!(
            env_end < program,
            "env assignments precede the program for /usr/bin/env"
        );
        let bridge = bridge_command(
            Backend::RocmWsl,
            &layout(),
            Some(&target()),
            r"C:\Models\m",
            0,
        )
        .unwrap();
        assert!(
            bridge
                .args
                .contains(&"HSA_ENABLE_DXG_DETECTION=1".to_owned())
        );
        let bad = WslTarget {
            distro: "x;y".into(),
            home: "/h".into(),
        };
        assert!(install_plan(Backend::RocmWsl, &layout(), Some(&bad)).is_err());
    }

    #[test]
    fn bridge_paths_per_backend() {
        let cuda = bridge_command(Backend::Cuda, &layout(), None, r"D:\Models\Q", 8192).unwrap();
        assert!(cuda.args.ends_with(&[
            "bridge".into(),
            r"D:\Models\Q".into(),
            "--context-length".into(),
            "8192".into()
        ]));
        let rocm = bridge_command(
            Backend::RocmWsl,
            &layout(),
            Some(&target()),
            r"D:\Models\Q",
            0,
        )
        .unwrap();
        assert!(rocm.args.contains(&"/mnt/d/Models/Q".to_owned()));
        assert!(
            rocm.args
                .contains(&"PYTHONPATH=/mnt/c/Program Files/MLXL3/engine".to_owned())
        );
        assert!(
            bridge_command(Backend::RocmWsl, &layout(), Some(&target()), r"\\nas\m", 0).is_err()
        );
        let download = download_command(
            Backend::RocmWsl,
            &layout(),
            Some(&target()),
            "a/b",
            "4bpw",
            r"C:\M",
        )
        .unwrap();
        assert!(download.args.ends_with(&[
            "download".into(),
            "a/b".into(),
            "/mnt/c/M".into(),
            "--folder".into(),
            "4bpw".into()
        ]));
    }

    #[test]
    fn markers_round_trip_and_reject_stale() {
        let dir = tempfile::tempdir().unwrap();
        let layout = Layout {
            data_dir: dir.path().into(),
            resources: dir.path().into(),
        };
        let marker = Marker {
            id: runtime_id(Backend::RocmWsl),
            backend: Backend::RocmWsl,
            wsl: Some(target()),
            probe: serde_json::json!({"ok": true}),
        };
        assert!(installed(&layout, Backend::RocmWsl).is_none());
        write_marker(&layout, &marker).unwrap();
        assert_eq!(installed(&layout, Backend::RocmWsl), Some(marker.clone()));
        assert!(
            write_marker(
                &layout,
                &Marker {
                    probe: serde_json::json!({"ok": false}),
                    ..marker.clone()
                }
            )
            .is_err()
        );
        assert!(
            write_marker(
                &layout,
                &Marker {
                    id: "rocm-000000000000".into(),
                    ..marker.clone()
                }
            )
            .is_err()
        );
        std::fs::write(
            marker_path(&layout, Backend::RocmWsl),
            br#"{"id":"rocm-000000000000","backend":"rocm_wsl","wsl":null,"probe":{}}"#,
        )
        .unwrap();
        assert!(installed(&layout, Backend::RocmWsl).is_none());
        // CUDA marker without the venv's python.exe is not an installation.
        let cuda = Marker {
            id: runtime_id(Backend::Cuda),
            backend: Backend::Cuda,
            wsl: None,
            probe: serde_json::json!({"ok": true}),
        };
        write_marker(&layout, &cuda).unwrap();
        assert!(installed(&layout, Backend::Cuda).is_none());
        let python = cuda_python(&layout);
        std::fs::create_dir_all(python.parent().unwrap()).unwrap();
        std::fs::write(&python, b"").unwrap();
        assert_eq!(installed(&layout, Backend::Cuda), Some(cuda));
    }
}
