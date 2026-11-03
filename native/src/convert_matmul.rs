//! Bounded FP32 products used by conversion feedback, without model weights.

/// Exact row-major element counts for a Metal matrix product.
/// Each input and output fits the standalone conversion protocol's 256 MiB cap.
pub fn matrix_plan(m: u32, n: u32, k: u32) -> Option<(usize, usize, usize)> {
    if m == 0 || n == 0 || k == 0 {
        return None;
    }
    let left = u64::from(m) * u64::from(k);
    let right = u64::from(k) * u64::from(n);
    let output = u64::from(m) * u64::from(n);
    const LIMIT: u64 = 64 * 1024 * 1024;
    if left > LIMIT || right > LIMIT || output > LIMIT {
        return None;
    }
    Some((
        usize::try_from(left).ok()?,
        usize::try_from(right).ok()?,
        usize::try_from(output).ok()?,
    ))
}

#[test]
fn matrix_geometry_rejects_empty_overflowing_or_excessive_arrays() {
    assert_eq!(matrix_plan(96, 128, 32), Some((3072, 4096, 12288)));
    assert_eq!(matrix_plan(3, 32, 2048), Some((6144, 65536, 96)));
    assert_eq!(
        matrix_plan(8, 8, 8_388_608),
        Some((67_108_864, 67_108_864, 64))
    );
    for (m, n, k) in [
        (0, 8, 8),
        (8, 0, 8),
        (8, 8, 0),
        (u32::MAX, u32::MAX, u32::MAX),
        (8, 8, 8_388_616),
        (16_384, 16_384, 8),
    ] {
        assert!(matrix_plan(m, n, k).is_none());
    }
}

#[cfg(kani)]
#[kani::proof]
fn conversion_matmul_geometry_covers_all_shader_addresses() {
    let m: u32 = kani::any();
    let n: u32 = kani::any();
    let k: u32 = kani::any();
    let row: u32 = kani::any();
    let col: u32 = kani::any();
    let depth: u32 = kani::any();
    if let Some((left, right, output)) = matrix_plan(m, n, k) {
        assert!(m > 0 && n > 0 && k > 0);
        assert!(
            left <= 64 * 1024 * 1024 && right <= 64 * 1024 * 1024 && output <= 64 * 1024 * 1024
        );
        assert_eq!(left as u64, u64::from(m) * u64::from(k));
        assert_eq!(right as u64, u64::from(k) * u64::from(n));
        assert_eq!(output as u64, u64::from(m) * u64::from(n));
        if row < m && col < n && depth < k {
            assert!(u64::from(row) * u64::from(k) + u64::from(depth) < left as u64);
            assert!(u64::from(depth) * u64::from(n) + u64::from(col) < right as u64);
            assert!(u64::from(row) * u64::from(n) + u64::from(col) < output as u64);
        }
        kani::cover!(m == 96 && n == 128 && k == 32 && row == 95 && col == 127 && depth == 31);
        kani::cover!(m == 8 && n == 8 && k == 8_388_608);
    }
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
fn read_values(path: &std::path::Path, values: usize) -> anyhow::Result<Vec<f32>> {
    use anyhow::ensure;
    use std::io::Read;
    let mut file = std::fs::File::open(path)?;
    let bytes = values
        .checked_mul(4)
        .ok_or_else(|| anyhow::anyhow!("matrix byte extent overflow"))?;
    ensure!(
        file.metadata()?.len() == bytes as u64,
        "matrix input has the wrong length"
    );
    let mut data = Vec::new();
    data.try_reserve_exact(bytes + 1)?;
    (&mut file).take(bytes as u64 + 1).read_to_end(&mut data)?;
    ensure!(data.len() == bytes, "matrix input changed during read");
    let result: Vec<f32> = data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| f32::from_le_bytes(*v))
        .collect();
    ensure!(
        result.len() == values && result.iter().all(|v| v.is_finite()),
        "matrix input must be finite"
    );
    Ok(result)
}

#[cfg(all(target_os = "macos", feature = "direct-metal"))]
pub fn run(
    left: &std::path::Path,
    right: &std::path::Path,
    destination: &std::path::Path,
    m: u32,
    n: u32,
    k: u32,
    full_fp32: bool,
) -> anyhow::Result<()> {
    use anyhow::{Context, ensure};
    use std::{
        io::Write,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
    };
    let (left_values, right_values, output_values) =
        matrix_plan(m, n, k).context("matrix dimensions must be positive within 256 MiB")?;
    ensure!(
        !destination.try_exists()?,
        "output directory already exists"
    );
    let left = read_values(left, left_values)?;
    let right = read_values(right, right_values)?;
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| std::path::Path::new("."));
    let stage = tempfile::Builder::new()
        .prefix(".mlxl3-matmul-")
        .tempdir_in(parent)?;
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
        let (values, arithmetic) =
            crate::gpu::convert_matmul::matmul(&left, &right, m, n, k, full_fp32, &cancelled)?;
        ensure!(
            values.len() == output_values && values.iter().all(|v| v.is_finite()),
            "incomplete matrix result"
        );
        let mut file =
            std::io::BufWriter::new(std::fs::File::create(stage.path().join("output.f32"))?);
        for value in values {
            file.write_all(&value.to_le_bytes())?;
        }
        file.flush()?;
        file.get_ref().sync_all()?;
        drop(file);
        let report = serde_json::json!({"schema":1,"status":"complete","m":m,"n":n,"k":k,"values":output_values,"arithmetic":arithmetic});
        std::fs::write(
            stage.path().join("report.json"),
            serde_json::to_vec(&report)?,
        )?;
        ensure!(!cancelled.load(Ordering::Relaxed), "conversion cancelled");
        crate::quantize::publish_directory(stage.path(), destination)?;
        let _ = stage.keep();
        println!("{report}");
        Ok(())
    })();
    for signal in signals {
        signal_hook::low_level::unregister(signal);
    }
    result
}
