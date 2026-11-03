//! Conversion trust boundaries through the production executable. No GPU allocation is needed.
#[cfg(all(target_os = "macos", feature = "direct-metal"))]
#[test]
fn invalid_quantizer_requests_never_publish_or_replace_output() -> anyhow::Result<()> {
    use std::{io::Write, process::Command};
    let root = tempfile::tempdir()?;
    let input = root.path().join("input.f32");
    let output = root.path().join("result");
    let check = |args: &[&str], bytes: &[u8]| -> anyhow::Result<()> {
        std::fs::write(&input, bytes)?;
        let result = Command::new(env!("CARGO_BIN_EXE_mlxl3-rs"))
            .arg("quantize-tiles")
            .arg(&input)
            .arg(&output)
            .args(args)
            .output()?;
        assert!(!result.status.success(), "bad conversion accepted");
        assert!(!result.stderr.is_empty());
        assert!(!output.try_exists()?);
        assert!(
            root.path()
                .read_dir()?
                .all(|e| e.unwrap().file_name() == "input.f32")
        );
        Ok(())
    };
    let zeros = vec![0; 1024];
    for bytes in [&[][..], &zeros[..1020], &[0u8; 1025]] {
        check(&["--k", "4"], bytes)?;
    }
    for args in [
        vec!["--k", "1"],
        vec!["--k", "9"],
        vec!["--k", "4", "--codebook", "3"],
        vec!["--k", "4", "--max-scratch-bytes", "0"],
    ] {
        check(&args, &zeros)?;
    }
    for invalid in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut bytes = zeros.clone();
        bytes[..4].copy_from_slice(&invalid.to_le_bytes());
        check(&["--k", "4"], &bytes)?;
    }
    let mut large = std::fs::File::create(&input)?;
    large.set_len(256 * 1024 * 1024 + 1024)?;
    large.flush()?;
    let result = Command::new(env!("CARGO_BIN_EXE_mlxl3-rs"))
        .arg("quantize-tiles")
        .arg(&input)
        .arg(&output)
        .args(["--k", "4"])
        .output()?;
    assert!(!result.status.success() && !output.try_exists()?);
    std::fs::write(&input, zeros)?;
    std::fs::create_dir(&output)?;
    std::fs::write(output.join("keep"), "keep")?;
    let result = Command::new(env!("CARGO_BIN_EXE_mlxl3-rs"))
        .arg("quantize-tiles")
        .arg(&input)
        .arg(&output)
        .args(["--k", "4"])
        .output()?;
    assert!(!result.status.success());
    assert_eq!(std::fs::read_to_string(output.join("keep"))?, "keep");
    Ok(())
}
