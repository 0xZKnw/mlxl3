use std::{env, path::PathBuf, process::Command};

fn run(command: &mut Command) {
    let status = command.status().expect("could not run native compiler");
    assert!(status.success(), "native compiler failed: {command:?}");
}

fn main() {
    // Resolve Git paths too: .git can be a file in a worktree, and HEAD itself
    // does not change when a commit advances the current branch.
    let reference = Command::new("git")
        .args(["symbolic-ref", "-q", "HEAD"])
        .output()
        .ok();
    let reference = reference
        .as_ref()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned());
    for path in [
        Some("HEAD"),
        Some("index"),
        Some("packed-refs"),
        reference.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if let Ok(output) = Command::new("git")
            .args(["rev-parse", "--git-path", path])
            .output()
            && output.status.success()
        {
            println!(
                "cargo:rerun-if-changed={}",
                String::from_utf8_lossy(&output.stdout).trim()
            );
        }
    }
    println!("cargo:rerun-if-changed=native");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=Cargo.lock");
    let revision = Command::new("git")
        .args([
            "describe",
            "--always",
            "--dirty",
            "--abbrev=12",
            "--match",
            "",
        ])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|revision| !revision.is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=MLXL3_BUILD_REVISION={revision}");
    println!(
        "cargo:rustc-env=MLXL3_BUILD_PROFILE={}",
        env::var("PROFILE").unwrap_or_else(|_| "unknown".to_owned())
    );
    println!("cargo:rerun-if-env-changed=MLXL3_MLX_ROOT");
    println!("cargo:rerun-if-changed=native/ffi/mlx_bridge.cpp");
    if env::var_os("CARGO_FEATURE_MLX").is_none() {
        return;
    }
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "macos",
        "feature mlx requires macOS"
    );
    assert_eq!(
        env::var("CARGO_CFG_TARGET_ARCH").unwrap(),
        "aarch64",
        "feature mlx requires Apple Silicon"
    );
    let root = PathBuf::from(
        env::var_os("MLXL3_MLX_ROOT")
            .expect("set MLXL3_MLX_ROOT to the installed MLX 0.32.2 package directory"),
    )
    .canonicalize()
    .expect("MLXL3_MLX_ROOT does not exist");
    for file in [
        "include/mlx/version.h",
        "include/mlx/fast.h",
        "lib/libmlx.dylib",
        "lib/libjaccl.dylib",
        "lib/mlx.metallib",
    ] {
        assert!(
            root.join(file).is_file(),
            "missing MLX component: {}",
            root.join(file).display()
        );
        println!("cargo:rerun-if-changed={}", root.join(file).display());
    }
    let version_header = std::fs::read_to_string(root.join("include/mlx/version.h"))
        .expect("cannot read MLX version header");
    let part = |name: &str| {
        version_header
            .lines()
            .find_map(|line| line.strip_prefix(&format!("#define {name} ")))
            .map(str::trim)
            .expect("missing MLX version macro")
            .to_owned()
    };
    println!(
        "cargo:rustc-env=MLXL3_MLX_VERSION={}.{}.{}",
        part("MLX_VERSION_MAJOR"),
        part("MLX_VERSION_MINOR"),
        part("MLX_VERSION_PATCH")
    );
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let object = out.join("mlx_bridge.o");
    let archive = out.join("libmlxl3_mlx_bridge.a");
    run(Command::new("/usr/bin/clang++")
        .args([
            "-std=c++20",
            "-O2",
            "-fPIC",
            "-fvisibility=hidden",
            "-c",
            "native/ffi/mlx_bridge.cpp",
            "-o",
        ])
        .arg(&object)
        .arg("-I")
        .arg(root.join("include")));
    run(Command::new("/usr/bin/ar")
        .arg("crs")
        .arg(&archive)
        .arg(&object));
    println!("cargo:rustc-link-search=native={}", out.display());
    println!(
        "cargo:rustc-link-search=native={}",
        root.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=mlxl3_mlx_bridge");
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-lib=dylib=c++");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");
    println!(
        "cargo:rustc-link-arg=-Wl,-rpath,{}",
        root.join("lib").display()
    );
    println!("cargo:rustc-env=MLXL3_MLX_BUILD_ROOT={}", root.display());
}
