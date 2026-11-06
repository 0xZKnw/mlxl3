//! One archive entrypoint, with separately compiled Metal and MLX inference processes.
use anyhow::{Context, Result, ensure};
use clap::Parser;
use mlxl3_native::{backend, registry};
use std::{
    io::Read,
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::Command,
};
// Delegated options are parsed here, then consumed by the selected real helper.
#[allow(dead_code)]
mod cli;

fn payloads() -> Result<[(&'static [u8], &'static str); 2]> {
    #[cfg(mlxl3_hybrid_payload)]
    return Ok([
        (
            include_bytes!(env!("MLXL3_MLX_HELPER")),
            env!("MLXL3_MLX_HELPER_SHA256"),
        ),
        (
            include_bytes!(env!("MLXL3_METAL_HELPER")),
            env!("MLXL3_METAL_HELPER_SHA256"),
        ),
    ]);
    #[cfg(not(mlxl3_hybrid_payload))]
    anyhow::bail!(
        "build the hybrid engine with MLXL3_ENGINE_BACKEND=hybrid scripts/build-engine.sh"
    )
}

fn model_path(args: &cli::Cli) -> Result<Option<PathBuf>> {
    Ok(match &args.command {
        cli::Command::Run { model, .. }
        | cli::Command::Bridge { model, .. }
        | cli::Command::Serve { model, .. } => {
            Some(cli::resolve_model_path(args.registry.as_deref(), model)?)
        }
        cli::Command::Forward { model, .. } => Some(model.clone()),
        _ => None,
    })
}

fn flash_checkpoint(path: &Path) -> Result<bool> {
    let mut bytes = Vec::new();
    std::fs::File::open(path.join("config.json"))?
        .take(1024 * 1024 + 1)
        .read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= 1024 * 1024,
        "model configuration exceeds 1 MiB"
    );
    let config: serde_json::Value = serde_json::from_slice(&bytes)?;
    let kind = config["model_type"]
        .as_str()
        .context("missing model_type")?;
    Ok(backend::native_flash(kind.as_bytes()))
}

fn run(args: cli::Cli) -> Result<()> {
    let helpers = payloads()?;
    if matches!(args.command, cli::Command::RuntimeInfo) {
        println!(
            "{}",
            serde_json::json!({"version":env!("CARGO_PKG_VERSION"),"commit":env!("MLXL3_BUILD_REVISION"),
            "profile":env!("MLXL3_BUILD_PROFILE"),"bridge_protocol":1,"backend":"hybrid",
            "mlx_enabled":true,"chat_enabled":true,"architecture":std::env::consts::ARCH,
            "native_flash_enabled":true,"mlx_helper_sha256":helpers[0].1,"metal_helper_sha256":helpers[1].1})
        );
        return Ok(());
    }
    let native = match model_path(&args)? {
        Some(path) => flash_checkpoint(&path)?,
        None => matches!(
            args.command,
            cli::Command::QuantizeTiles { .. } | cli::Command::ConvertMatmul { .. }
        ),
    };
    let base = registry::default_path()?
        .parent()
        .context("registry has no parent")?
        .join("backends");
    backend::private_directory(&base)?;
    let index = usize::from(native);
    let helper = backend::install_helper(
        &base.join(if native { "metal" } else { "mlx" }),
        helpers[index].1,
        helpers[index].0,
    )?;
    let mut command = Command::new(helper);
    command.args(std::env::args_os().skip(1));
    if native {
        command.env("MLXL3_NATIVE_FLASH_ONLY", "1");
    } else {
        command.env_remove("MLXL3_NATIVE_FLASH_ONLY");
        let executable = std::env::current_exe()?.canonicalize()?;
        command.env(
            "DYLD_LIBRARY_PATH",
            executable.parent().context("runtime directory missing")?,
        );
    }
    Err(command.exec()).context("cannot execute verified inference backend")
}

fn main() -> std::process::ExitCode {
    match run(cli::Cli::parse()) {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("Error: {error:#}");
            std::process::ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn routing_reads_actual_config_and_rejects_missing_invalid_and_oversized_data() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let config = dir.path().join("config.json");
        assert!(flash_checkpoint(dir.path()).is_err());
        for text in ["", "null", "{}", "{\"model_type\":7}"] {
            std::fs::write(&config, text)?;
            assert!(flash_checkpoint(dir.path()).is_err());
        }
        for (name, native) in [
            ("qwen4_exp", true),
            ("lfm3", false),
            ("lfm3_vl", false),
            ("qwen3_5_moe", false),
        ] {
            std::fs::write(
                &config,
                serde_json::to_vec(&serde_json::json!({"model_type":name}))?,
            )?;
            assert_eq!(flash_checkpoint(dir.path())?, native);
        }
        let registry = dir.path().join("registry.json");
        std::fs::write(
            &registry,
            serde_json::to_vec(&serde_json::json!({"version":1,"models":{"alias":{
            "path":dir.path(),"model_type":"qwen4_exp","format":"exl3","bits":4.0,
            "size_bytes":1,"modules":1,"added_at":"test"}}}))?,
        )?;
        let args = cli::Cli::try_parse_from([
            "mlxl3",
            "--registry",
            registry.to_str().unwrap(),
            "bridge",
            "alias",
        ])?;
        assert_eq!(model_path(&args)?, Some(dir.path().to_path_buf()));
        assert!(!flash_checkpoint(&model_path(&args)?.unwrap())?);
        std::fs::write(&config, vec![b' '; 1024 * 1024 + 1])?;
        assert!(flash_checkpoint(dir.path()).is_err());
        Ok(())
    }
    #[test]
    fn existing_cli_preserves_inference_options_and_registry_positions() {
        let args = cli::Cli::try_parse_from([
            "mlxl3",
            "bridge",
            "alias",
            "--context-length",
            "4096",
            "--registry",
            "/tmp/registry",
        ])
        .unwrap();
        assert_eq!(args.registry.as_deref(), Some(Path::new("/tmp/registry")));
        assert!(matches!(
            args.command,
            cli::Command::Bridge {
                context_length: 4096,
                ..
            }
        ));
        assert!(
            cli::Cli::try_parse_from(["mlxl3", "run", "model", "--image", "image.png"]).is_err()
        );
        let args = cli::Cli::try_parse_from([
            "mlxl3",
            "forward",
            "/tmp/model",
            "--tokens",
            "1,2",
            "--states",
        ])
        .unwrap();
        assert!(
            matches!(args.command,cli::Command::Forward{states:true,tokens,..} if tokens==[1,2])
        );
    }
}
