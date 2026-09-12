use std::{env, path::PathBuf, process::Command};

fn run(command: &mut Command) {
    let status = command.status().expect("could not run native compiler");
    assert!(status.success(), "native compiler failed: {command:?}");
}

fn main() {
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
