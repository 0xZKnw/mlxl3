use super::{apply, kernel_requests_for_test, last_rows_for_test, reset_kernel_requests_for_test};
use crate::array::{Array, Dtype, MetalCapture, synchronize};
use anyhow::{Context, Result, ensure};
use half::f16;
use std::path::{Path, PathBuf};

fn capture_bytes(path: &Path) -> Result<u64> {
    let root = path.canonicalize().context("missing capture package")?;
    ensure!(root.is_dir(), "capture package is not a directory");
    let mut total = 0u64;
    let mut pending = vec![root.clone()];
    while let Some(directory) = pending.pop() {
        for entry in std::fs::read_dir(directory)? {
            let entry = entry?;
            let metadata = std::fs::symlink_metadata(entry.path())?;
            let kind = metadata.file_type();
            if kind.is_dir() {
                pending.push(entry.path());
            } else {
                if kind.is_symlink() {
                    let target = entry
                        .path()
                        .canonicalize()
                        .context("invalid capture link")?;
                    ensure!(
                        target.starts_with(&root) && std::fs::metadata(&target)?.is_file(),
                        "capture link target is not an internal regular file"
                    );
                } else {
                    ensure!(kind.is_file(), "unexpected capture entry type");
                }
                total = total
                    .checked_add(metadata.len())
                    .context("capture byte count overflow")?;
                ensure!(total < 1 << 30, "capture exceeds 1 GiB diagnostic budget");
            }
        }
    }
    ensure!(total > 0, "empty Metal capture");
    Ok(total)
}

fn capture_size_counts_nested_files_and_internal_symlink_once() -> Result<()> {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir()?;
    let root = temporary.path().join("capture.gputrace");
    std::fs::create_dir_all(root.join("nested"))?;
    std::fs::write(root.join("root.bin"), [0x11; 7])?;
    std::fs::write(root.join("nested/data.bin"), [0x27; 13])?;
    ensure!(capture_bytes(&root)? == 20);

    let relative = "nested/data.bin";
    symlink(relative, root.join("relative"))?;
    ensure!(capture_bytes(&root)? == 20 + relative.len() as u64);
    let absolute = root.join(relative);
    symlink(&absolute, root.join("absolute"))?;
    ensure!(
        capture_bytes(&root)? == 20 + relative.len() as u64 + absolute.as_os_str().len() as u64,
        "internal link payload was omitted or target bytes were counted twice"
    );
    ensure!(std::fs::read(root.join("root.bin"))? == [0x11; 7]);
    ensure!(std::fs::read(root.join(relative))? == [0x27; 13]);
    println!("GDN_CAPTURE_SIZE_INTERNAL cases=3 payload_bytes=20 targets_recounted=false");
    Ok(())
}

fn capture_size_rejects_external_dangling_directory_loop_and_special_entries() -> Result<()> {
    use std::os::unix::fs::symlink;

    let temporary = tempfile::tempdir()?;
    let root = temporary.path().join("capture.gputrace");
    std::fs::create_dir_all(root.join("nested"))?;
    std::fs::write(root.join("data.bin"), [0x11; 7])?;
    let outside = temporary.path().join("outside.bin");
    std::fs::write(&outside, [0x22; 5])?;
    let prefix_sibling = temporary.path().join("capture.gputrace-other");
    std::fs::create_dir(&prefix_sibling)?;
    let prefix_target = prefix_sibling.join("data.bin");
    std::fs::write(&prefix_target, [0x33; 3])?;
    let link = root.join("bad");
    let mut cases = 0;
    for target in [
        PathBuf::from("../outside.bin"),
        outside,
        prefix_target,
        PathBuf::from("missing"),
        PathBuf::from("nested"),
        PathBuf::from("bad"),
    ] {
        symlink(target, &link)?;
        ensure!(
            capture_bytes(&root).is_err(),
            "unsafe capture link accepted"
        );
        std::fs::remove_file(&link)?;
        cases += 1;
    }
    symlink("other", &link)?;
    symlink("bad", root.join("other"))?;
    ensure!(capture_bytes(&root).is_err(), "capture link cycle accepted");
    std::fs::remove_file(&link)?;
    std::fs::remove_file(root.join("other"))?;
    cases += 1;

    let fifo = root.join("fifo");
    let name = std::ffi::CString::new(fifo.as_os_str().as_encoded_bytes())?;
    ensure!(
        unsafe { libc::mkfifo(name.as_ptr(), 0o600) } == 0,
        "creating disposable FIFO: {}",
        std::io::Error::last_os_error()
    );
    ensure!(
        capture_bytes(&root).is_err(),
        "special capture entry accepted"
    );
    std::fs::remove_file(fifo)?;
    cases += 1;
    ensure!(cases == 8 && capture_bytes(&root)? == 7);
    println!("GDN_CAPTURE_SIZE_INVALID_LINKS cases={cases} escaped=false");
    Ok(())
}

fn capture_size_rejects_empty_missing_and_non_directory_packages() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let root = temporary.path().join("capture.gputrace");
    ensure!(capture_bytes(&root).is_err());
    std::fs::create_dir(&root)?;
    ensure!(capture_bytes(&root).is_err());
    std::fs::create_dir(root.join("empty"))?;
    ensure!(capture_bytes(&root).is_err());
    let file = temporary.path().join("not-a-package");
    std::fs::write(&file, [0x44])?;
    ensure!(capture_bytes(&file).is_err());
    println!("GDN_CAPTURE_SIZE_MISSING cases=4");
    Ok(())
}

fn capture_size_enforces_sparse_file_limit_boundary() -> Result<()> {
    let temporary = tempfile::tempdir()?;
    let root = temporary.path().join("capture.gputrace");
    std::fs::create_dir(&root)?;
    let file = std::fs::File::create(root.join("sparse"))?;
    let limit = 1u64 << 30;
    file.set_len(limit - 1)?;
    ensure!(capture_bytes(&root)? == limit - 1);
    file.set_len(limit)?;
    ensure!(capture_bytes(&root).is_err());
    file.set_len(limit + 1)?;
    ensure!(capture_bytes(&root).is_err());
    println!("GDN_CAPTURE_SIZE_LIMIT cases=3 limit_bytes={limit}");
    Ok(())
}

#[test]
fn capture_size_validates_internal_links_rejections_and_sparse_limits() -> Result<()> {
    capture_size_counts_nested_files_and_internal_symlink_once()?;
    capture_size_rejects_external_dangling_directory_loop_and_special_entries()?;
    capture_size_rejects_empty_missing_and_non_directory_packages()?;
    capture_size_enforces_sparse_file_limit_boundary()?;
    println!("GDN_CAPTURE_SIZE_CPU groups=4 cases=18");
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn actual_gdn_capture_size_filesystem_entry() {
    capture_size_counts_nested_files_and_internal_symlink_once().unwrap();
}

#[test]
#[ignore = "physical Apple GPU; MTL_CAPTURE_ENABLED=1 and fresh MLXL3_GDN_CAPTURE_OUTPUT; run alone"]
fn captures_real_stock_and_fused_b1_b2_epilogue_commands() -> Result<()> {
    ensure!(
        std::env::var("MTL_CAPTURE_ENABLED").is_ok_and(|value| value == "1"),
        "capture requires MTL_CAPTURE_ENABLED=1 before process startup"
    );
    let directory = PathBuf::from(
        std::env::var_os("MLXL3_GDN_CAPTURE_OUTPUT")
            .context("MLXL3_GDN_CAPTURE_OUTPUT is required")?,
    );
    ensure!(directory.is_dir(), "capture output directory must exist");
    for rows in [1, 2] {
        for arm in ["stock", "fused"] {
            ensure!(
                !directory.join(format!("{arm}-b{rows}.gputrace")).exists(),
                "refusing to overwrite an existing capture"
            );
        }
    }

    const WIDTH: usize = 6144;
    let mut compared_values = 0;
    let mut total_capture_bytes = 0u64;
    let mut captures = 0;
    for rows in [1usize, 2] {
        let normed = (0..rows)
            .map(|row| {
                let values = (0..WIDTH)
                    .map(|i| ((i * 37 + row * 101) % 2039) as f32 / 257.0 - 3.7)
                    .collect::<Vec<_>>();
                Array::from_f32(&values, &[1, 1, 48, 128])
            })
            .collect::<Result<Vec<_>>>()?;
        let bits = (0..rows * WIDTH)
            .map(|i| f16::from_f32(((i * 73 % 4093) as f32 - 2046.0) / 511.0).to_bits())
            .collect::<Vec<_>>();
        let z = Array::from_f16_bits(&bits, &[1, rows as i32, 48, 128])?;
        for input in normed.iter().chain(std::iter::once(&z)) {
            input.eval()?;
        }
        let before = normed
            .iter()
            .chain(std::iter::once(&z))
            .map(Array::to_bytes)
            .collect::<Result<Vec<_>>>()?;
        super::tests::stock(&normed, &z)?.eval()?;
        apply(&normed, &z)?.eval()?;
        synchronize()?;

        let stock_path = directory.join(format!("stock-b{rows}.gputrace"));
        let capture = MetalCapture::start(&stock_path)?;
        // Rotate MLX's preexisting buffer so the graph uses a buffer created in capture.
        synchronize()?;
        let reference = super::tests::stock(&normed, &z)?;
        reference.eval()?;
        capture.finish()?;

        let fused_path = directory.join(format!("fused-b{rows}.gputrace"));
        reset_kernel_requests_for_test();
        let capture = MetalCapture::start(&fused_path)?;
        synchronize()?;
        let actual = apply(&normed, &z)?;
        actual.eval()?;
        capture.finish()?;
        ensure!(
            kernel_requests_for_test() == 1 && last_rows_for_test() == rows,
            "capture did not exercise the fused kernel request path"
        );

        for output in [&reference, &actual] {
            ensure!(
                output.shape() == [rows as i32, WIDTH as i32] && output.dtype() == Dtype::Float16,
                "capture output metadata differs"
            );
            let values = output.to_f32()?;
            ensure!(
                values.len() == rows * WIDTH && values.iter().all(|v| v.is_finite()),
                "empty or nonfinite capture output"
            );
        }
        let bytes = reference.to_bytes()?;
        ensure!(
            bytes.len() == rows * WIDTH * 2 && actual.to_bytes()? == bytes,
            "captured GDN epilogue differs from complete stock oracle"
        );
        ensure!(
            normed
                .iter()
                .chain(std::iter::once(&z))
                .map(Array::to_bytes)
                .collect::<Result<Vec<_>>>()?
                == before,
            "capture mutated a source alias"
        );
        for path in [&stock_path, &fused_path] {
            ensure!(path.is_dir(), "missing Metal capture");
            total_capture_bytes = total_capture_bytes
                .checked_add(capture_bytes(path)?)
                .context("capture byte count overflow")?;
            ensure!(total_capture_bytes < 1 << 30, "four captures exceed 1 GiB");
            captures += 1;
        }
        compared_values += rows * WIDTH;
    }
    ensure!(captures == 4 && compared_values == 18432 && total_capture_bytes > 0);
    println!(
        "GDN_EPILOGUE_CAPTURE captures={captures} compared_values={compared_values} compared_bytes={} capture_bytes={total_capture_bytes} status=captured_exact_outputs_dispatch_inspection_pending",
        compared_values * 2,
    );
    Ok(())
}
