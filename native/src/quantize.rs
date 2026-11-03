//! Contracts for the standalone conversion service.

/// Scratch includes all predecessor and half-cost buffers used by a single launch.
pub fn tile_search_plan(values: usize, k: u32, scratch_bytes: u64) -> Option<(usize, u64)> {
    if !(2..=8).contains(&k)
        || values == 0
        || values > u32::MAX as usize
        || !values.is_multiple_of(256)
    {
        return None;
    }
    let per_tile = 516 * (1u64 << (16 - k));
    let tiles = u64::try_from(values / 256)
        .ok()?
        .min(scratch_bytes / per_tile);
    (tiles > 0).then_some((usize::try_from(tiles).ok()?, per_tile))
}

#[test]
fn tile_search_budgets_cover_complete_tiles_and_reject_incomplete_input() {
    for k in 2..=8 {
        let per_tile = 516 * (1u64 << (16 - k));
        assert_eq!(tile_search_plan(768, k, per_tile * 2), Some((2, per_tile)));
        assert_eq!(tile_search_plan(256, k, per_tile - 1), None);
        assert_eq!(tile_search_plan(255, k, u64::MAX), None);
        assert_eq!(tile_search_plan(0, k, u64::MAX), None);
    }
    for k in [0, 1, 9, u32::MAX] {
        assert_eq!(tile_search_plan(256, k, u64::MAX), None);
    }
    assert_eq!(tile_search_plan(usize::MAX, 4, u64::MAX), None);
}

#[cfg(kani)]
#[kani::proof]
fn native_tile_search_launch_stays_inside_input_and_scratch() {
    let values: usize = kani::any();
    let k: u32 = kani::any();
    let budget: u64 = kani::any();
    let result = tile_search_plan(values, k, budget);
    if let Some((tiles, bytes)) = result {
        assert!((2..=8).contains(&k) && values > 0 && values <= u32::MAX as usize);
        assert!(values.is_multiple_of(256) && tiles > 0 && tiles <= values / 256);
        assert_eq!(bytes, ((1u64 << 16) >> k) * (256 * 2 + 2 * 2));
        assert!(tiles as u64 <= budget / bytes);
        kani::cover!(k == 2 && values == 768 && tiles == 2);
        kani::cover!(k == 8 && tiles == 1);
    }
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
pub(crate) fn publish_directory(
    stage: &std::path::Path,
    destination: &std::path::Path,
) -> std::io::Result<()> {
    use std::{ffi::CString, os::unix::ffi::OsStrExt};
    let source = CString::new(stage.as_os_str().as_bytes())?;
    let target = CString::new(destination.as_os_str().as_bytes())?;
    // Both names stay owned C strings until the exclusive atomic rename returns.
    if unsafe {
        libc::renameatx_np(
            libc::AT_FDCWD,
            source.as_ptr(),
            libc::AT_FDCWD,
            target.as_ptr(),
            libc::RENAME_EXCL,
        )
    } != 0
    {
        return Err(std::io::Error::last_os_error());
    }
    Ok(())
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
pub fn run_tiles(
    input: &std::path::Path,
    destination: &std::path::Path,
    k: u32,
    codebook: u32,
    scratch: u64,
) -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use std::{
        io::{Read, Write},
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };
    let mut file = std::fs::File::open(input)?;
    let bytes = file.metadata()?.len();
    ensure!(
        bytes > 0 && bytes <= 256 * 1024 * 1024 && bytes.is_multiple_of(1024),
        "input must contain complete finite FP32 tiles, at most 256 MiB"
    );
    let values = usize::try_from(bytes / 4)?;
    let (launch_tiles, _) =
        tile_search_plan(values, k, scratch).context("invalid tile count, K or scratch budget")?;
    let cb = crate::codec::Codebook::try_from(codebook)?;
    ensure!(
        !destination.try_exists()?,
        "output directory already exists"
    );
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let stage = tempfile::Builder::new()
        .prefix(".mlxl3-tiles-")
        .tempdir_in(parent)?;
    let mut data = Vec::new();
    data.try_reserve_exact(usize::try_from(bytes)? + 1)?;
    (&mut file).take(bytes + 1).read_to_end(&mut data)?;
    ensure!(
        data.len() == usize::try_from(bytes)?,
        "input changed during read"
    );
    let tiles: Vec<f32> = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    ensure!(
        tiles.iter().all(|v| v.is_finite()),
        "tile values must be finite"
    );
    let cancelled = Arc::new(AtomicBool::new(false));
    let mut signals = Vec::new();
    for signal in [libc::SIGINT, libc::SIGTERM, libc::SIGUSR1] {
        match signal_hook::flag::register(signal, cancelled.clone()) {
            Ok(id) => signals.push(id),
            Err(error) => {
                for id in signals {
                    signal_hook::low_level::unregister(id);
                }
                return Err(error.into());
            }
        }
    }
    let result = (|| -> anyhow::Result<()> {
        let (decoded, states) =
            crate::gpu::quantize::quantize_tiles(&tiles, k, cb, scratch, &cancelled)?;
        let packed = crate::codec::pack(&states, k as usize)?;
        for (name, data) in [
            (
                "decoded.f32",
                decoded
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<_>>(),
            ),
            (
                "states.u16",
                states.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
            (
                "packed.u16",
                packed.iter().flat_map(|v| v.to_le_bytes()).collect(),
            ),
        ] {
            let mut out = std::fs::File::create(stage.path().join(name))?;
            out.write_all(&data)?;
            out.sync_all()?;
        }
        ensure!(!cancelled.load(Ordering::Relaxed), "conversion cancelled");
        let report = serde_json::json!({"schema":1,"status":"complete", "tiles":values / 256,"values":values,"k":k,"codebook":codebook,"launch_tiles":launch_tiles,"max_scratch_bytes":scratch});
        std::fs::write(
            stage.path().join("report.json"),
            serde_json::to_vec(&report)?,
        )?;
        publish_directory(stage.path(), destination)?;
        let _ = stage.keep();
        println!("{report}");
        Ok(())
    })();
    for signal in signals {
        signal_hook::low_level::unregister(signal);
    }
    result
}

#[cfg(all(test, target_os = "macos", feature = "direct-metal"))]
#[test]
fn published_tile_directory_is_atomic_and_never_replaces_an_existing_path() -> anyhow::Result<()> {
    let root = tempfile::tempdir()?;
    let stage = tempfile::tempdir_in(root.path())?;
    std::fs::write(stage.path().join("result"), "complete")?;
    let target = root.path().join("result");
    std::fs::create_dir(&target)?;
    let error = publish_directory(stage.path(), &target).unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
    assert!(stage.path().join("result").is_file() && target.read_dir()?.next().is_none());
    std::fs::remove_dir(&target)?;
    publish_directory(stage.path(), &target)?;
    assert_eq!(std::fs::read_to_string(target.join("result"))?, "complete");
    Ok(())
}
