//! Native, draft-only Q4 projection. Target logits always use the original head.
use crate::checkpoint::{Checkpoint, read_header};
#[cfg(feature = "mlx")]
use anyhow::Context;
use anyhow::{Result, ensure};
use fs2::FileExt;
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    os::unix::fs::FileExt as _,
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::{Duration, Instant},
};

const HIDDEN: usize = 2048;
const VOCAB: usize = 248_320;
const CHUNK: usize = 4096;
const VOCABULARY: &str = include_str!("../data/qwen36_mtp_vocab.txt");

fn check_cancelled(cancelled: &AtomicBool) -> Result<()> {
    ensure!(!cancelled.load(Ordering::Relaxed), "generation cancelled");
    Ok(())
}

fn chunk_end(start: usize, total: usize, chunk: usize) -> Option<usize> {
    if start >= total
        || !start.is_multiple_of(128)
        || !total.is_multiple_of(128)
        || chunk == 0
        || !chunk.is_multiple_of(128)
    {
        return None;
    }
    Some(start + (total - start).min(chunk))
}

fn projection_offsets(rows: usize, hidden: usize) -> Option<[u64; 4]> {
    if rows == 0 || hidden == 0 || !hidden.is_multiple_of(64) {
        return None;
    }
    let values = u64::try_from(rows)
        .ok()?
        .checked_mul(u64::try_from(hidden).ok()?)?;
    let weight = values / 2;
    let scales = weight.checked_add(values / 32)?;
    Some([0, weight, scales, scales.checked_add(values / 32)?])
}

fn vocabulary() -> Result<Vec<u32>> {
    let mut ids = VOCABULARY
        .lines()
        .map(str::parse::<u32>)
        .collect::<std::result::Result<Vec<_>, _>>()?;
    ensure!(
        ids.len() == 79_591
            && ids.iter().all(|&id| id < VOCAB as u32)
            && ids.windows(2).all(|pair| pair[0] < pair[1]),
        "invalid bundled MTP vocabulary"
    );
    let mut next = 0;
    while !ids.len().is_multiple_of(64) {
        match ids.binary_search(&next) {
            Ok(_) => (),
            Err(index) => ids.insert(index, next),
        }
        next += 1;
    }
    Ok(ids)
}

fn rotate128(values: &mut [f32]) {
    for block in values.as_chunks_mut::<128>().0 {
        for width in [1, 2, 4, 8, 16, 32, 64] {
            for pair in block.chunks_exact_mut(width * 2) {
                for i in 0..width {
                    let (a, b) = (pair[i], pair[i + width]);
                    pair[i] = a + b;
                    pair[i + width] = a - b;
                }
            }
        }
        for value in block {
            *value /= 128_f32.sqrt();
        }
    }
}

fn digest_reader(mut reader: impl Read, cancelled: &AtomicBool) -> Result<String> {
    let mut hash = Sha256::new();
    let mut bytes = [0; 64 * 1024];
    loop {
        check_cancelled(cancelled)?;
        let n = reader.read(&mut bytes)?;
        if n == 0 {
            break;
        }
        hash.update(&bytes[..n]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn source_key(source: &Checkpoint, cancelled: &AtomicBool) -> Result<String> {
    let mut hash = Sha256::new();
    hash.update(b"mlxl3-qwen36-mtp-q4-f16-v1");
    hash.update(VOCABULARY.as_bytes());
    let mut bytes = [0; 64 * 1024];
    for (name, tensor) in &source.tensors {
        hash.update(name.as_bytes());
        hash.update(serde_json::to_vec(tensor)?);
        let (file, start) = tensor.source()?;
        let length = tensor.data_offsets[1] - tensor.data_offsets[0];
        let mut offset = 0;
        while offset < length {
            check_cancelled(cancelled)?;
            let n = bytes.len().min((length - offset) as usize);
            file.read_exact_at(&mut bytes[..n], start + offset)?;
            hash.update(&bytes[..n]);
            offset += n as u64;
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn valid_cache(path: &Path, metadata: &[u8], cancelled: &AtomicBool) -> Result<()> {
    ensure!(
        fs::read(path.join("draft_head_q4.json"))? == metadata,
        "stale MTP projection metadata"
    );
    let file = path.join("draft_head_q4.safetensors");
    let expected = fs::read_to_string(path.join("projection.sha256"))?;
    ensure!(
        expected.len() == 64 && digest_reader(File::open(&file)?, cancelled)? == expected,
        "damaged MTP projection cache"
    );
    ensure!(
        read_header(&file)?.len() == 3,
        "invalid MTP projection tensors"
    );
    Ok(())
}

fn cached_projection(
    root: &Path,
    key: &str,
    metadata: &[u8],
    cancelled: &AtomicBool,
    build: impl FnOnce(&Path) -> Result<()>,
) -> Result<PathBuf> {
    fs::create_dir_all(root)?;
    let lock = OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.join(format!("{key}.lock")))?;
    let started = Instant::now();
    loop {
        check_cancelled(cancelled)?;
        match lock.try_lock_exclusive() {
            Ok(()) => break,
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                ensure!(
                    started.elapsed() < Duration::from_secs(60),
                    "MTP projection cache is busy"
                );
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(e.into()),
        }
    }
    let destination = root.join(key);
    if valid_cache(&destination, metadata, cancelled).is_ok() {
        return Ok(destination);
    }
    check_cancelled(cancelled)?;
    let stage = tempfile::tempdir_in(root)?;
    let projection = stage.path().join("draft_head_q4.safetensors");
    build(&projection)?;
    check_cancelled(cancelled)?;
    fs::write(
        stage.path().join("projection.sha256"),
        digest_reader(File::open(&projection)?, cancelled)?,
    )?;
    fs::write(stage.path().join("draft_head_q4.json"), metadata)?;
    valid_cache(stage.path(), metadata, cancelled)?;
    check_cancelled(cancelled)?;
    // Keep an invalid old entry until its complete replacement is ready.
    let retired = tempfile::tempdir_in(root)?;
    if destination.symlink_metadata().is_ok() {
        fs::rename(&destination, retired.path().join("invalid"))?;
    }
    fs::rename(stage.path(), &destination)?;
    Ok(destination)
}

#[cfg(feature = "mlx")]
pub(crate) fn source(checkpoint: &Checkpoint) -> Option<Checkpoint> {
    let tensors = &checkpoint.tensors;
    if !tensors.contains_key("lm_head.mcg")
        || tensors.contains_key("lm_head.mul1")
        || tensors.contains_key("lm_head.bias")
        || ![
            ("trellis", "I16", vec![HIDDEN / 16, VOCAB / 16, 96]),
            ("suh", "F16", vec![HIDDEN]),
            ("svh", "F16", vec![VOCAB]),
        ]
        .iter()
        .all(|(part, dtype, shape)| {
            tensors
                .get(&format!("lm_head.{part}"))
                .is_some_and(|t| t.dtype == *dtype && t.shape == *shape)
        })
    {
        return None;
    }
    Some(Checkpoint {
        path: checkpoint.path.clone(),
        model_type: checkpoint.model_type.clone(),
        bits: checkpoint.bits,
        size_bytes: 0,
        modules: Vec::new(),
        tensors: tensors
            .iter()
            .filter(|(name, _)| name.starts_with("lm_head."))
            .map(|(name, tensor)| (name.clone(), tensor.clone()))
            .collect(),
    })
}

#[cfg(feature = "mlx")]
fn write_projection(
    source: &Checkpoint,
    ids: &[u32],
    path: &Path,
    cancelled: &AtomicBool,
) -> Result<()> {
    use crate::{
        array::Array,
        codec::{self, Codebook},
    };
    use half::f16;
    let offsets = projection_offsets(ids.len(), HIDDEN).context("invalid projection size")?;
    let mut header = serde_json::Map::new();
    for (i, (part, dtype, width)) in [
        ("weight", "U32", HIDDEN / 8),
        ("scales", "F16", HIDDEN / 64),
        ("biases", "F16", HIDDEN / 64),
    ]
    .into_iter()
    .enumerate()
    {
        header.insert(
            format!("lm_head.{part}"),
            serde_json::json!({
            "dtype":dtype, "shape":[ids.len(),width], "data_offsets":[offsets[i],offsets[i+1]]}),
        );
    }
    let mut header = serde_json::to_vec(&header)?;
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut file = File::create(path)?;
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    let payload = 8 + header.len() as u64;
    file.set_len(payload + offsets[3])?;
    let scales = |name: &str| -> Result<Vec<f32>> {
        let bytes = source.tensors[name].read_bytes()?;
        let values = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16::from_bits(u16::from_le_bytes(*b)).to_f32())
            .collect::<Vec<_>>();
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "non-finite EXL3 head scales"
        );
        Ok(values)
    };
    let (su, sv) = (scales("lm_head.suh")?, scales("lm_head.svh")?);
    let (trellis, base) = source.tensors["lm_head.trellis"].source()?;
    let mut written = 0;
    for start in (0..VOCAB).step_by(CHUNK) {
        check_cancelled(cancelled)?;
        let end = chunk_end(start, VOCAB, CHUNK).context("invalid reconstruction chunk")?;
        let selected_end = ids.partition_point(|&id| id < end as u32);
        let selected = &ids[written..selected_end];
        if selected.is_empty() {
            continue;
        }
        let cols = end - start;
        let tile_bytes = cols / 16 * 96 * 2;
        let mut bytes = vec![0; HIDDEN / 16 * tile_bytes];
        for (row, target) in bytes.chunks_exact_mut(tile_bytes).enumerate() {
            let offset = (row * (VOCAB / 16) * 96 + start / 16 * 96) * 2;
            trellis.read_exact_at(target, base + offset as u64)?;
        }
        let packed = bytes
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect::<Vec<_>>();
        let mut inner = codec::reconstruct_inner(&packed, 6, Codebook::Mcg, HIDDEN, cols)?
            .into_iter()
            .map(f16::to_f32)
            .collect::<Vec<_>>();
        rotate128(&mut inner);
        let mut weights = Vec::with_capacity(selected.len() * HIDDEN);
        for &id in selected {
            let col = id as usize - start;
            let mut row = (0..HIDDEN)
                .map(|input| inner[input * cols + col])
                .collect::<Vec<_>>();
            rotate128(&mut row);
            for (value, &scale) in row.iter().zip(&su) {
                let value = f16::from_f32(value * scale * sv[id as usize]);
                ensure!(value.is_finite(), "non-finite reconstructed MTP weight");
                weights.push(value.to_bits());
            }
        }
        let x = Array::from_f16_bits(&weights, &[selected.len() as i32, HIDDEN as i32])?;
        let (w, s, b) = x.quantize4()?;
        for (i, (array, row_bytes)) in [(&w, HIDDEN / 2), (&s, HIDDEN / 32), (&b, HIDDEN / 32)]
            .into_iter()
            .enumerate()
        {
            let bytes = array.to_bytes()?;
            ensure!(
                bytes.len() == selected.len() * row_bytes,
                "incomplete quantized projection"
            );
            file.write_all_at(&bytes, payload + offsets[i] + (written * row_bytes) as u64)?;
        }
        written = selected_end;
    }
    ensure!(written == ids.len(), "incomplete compact projection");
    file.sync_all()?;
    Ok(())
}

#[cfg(feature = "mlx")]
pub(crate) fn prepare(source: &Checkpoint, cancelled: &AtomicBool) -> Result<PathBuf> {
    let ids = vocabulary()?;
    let metadata = serde_json::to_vec(&serde_json::json!({"input":HIDDEN,"output":ids.len(),
        "bits":4,"group_size":64,"mode":"affine","token_ids":ids}))?;
    let key = source_key(source, cancelled)?;
    cached_projection(
        &crate::hub::managed_drafts_path()?.join("qwen36-mtp-q4-v1"),
        &key,
        &metadata,
        cancelled,
        |file| {
            eprintln!("Preparing compact MTP projection (first use)...");
            write_projection(source, &ids, file, cancelled)
        },
    )
}

#[cfg(kani)]
#[kani::proof]
fn compact_projection_chunks_and_offsets_are_bounded() {
    let (start, total, chunk): (usize, usize, usize) = (kani::any(), kani::any(), kani::any());
    if let Some(end) = chunk_end(start, total, chunk) {
        assert!(start < end && end <= total && end - start <= chunk);
        assert!(end % 128 == 0 && (end - start) % 128 == 0);
    }
    let (rows, hidden): (usize, usize) = (kani::any(), kani::any());
    if let Some([zero, weight, scales, end]) = projection_offsets(rows, hidden) {
        assert!(zero == 0 && rows > 0 && hidden > 0 && hidden % 64 == 0);
        assert!(weight > 0 && weight < scales && scales < end);
        assert!(end - scales == scales - weight);
    }
    kani::cover!(chunk_end(0, VOCAB, CHUNK) == Some(CHUNK));
    kani::cover!(chunk_end(VOCAB - 128, VOCAB, CHUNK) == Some(VOCAB));
    kani::cover!(chunk_end(start, total, chunk).is_none());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_vocabulary_matches_measured_projection() -> Result<()> {
        assert_eq!(
            format!("{:x}", Sha256::digest(VOCABULARY.as_bytes())),
            "88d5b483a849ae9245b78b69f41f11cdfc8b5c024f0786c1c8196263857cc93e"
        );
        let ids = vocabulary()?;
        assert_eq!(ids.len(), 79_616);
        assert!(ids.windows(2).all(|p| p[0] < p[1]) && ids[ids.len() - 1] < VOCAB as u32);
        assert_eq!(
            projection_offsets(ids.len(), HIDDEN),
            Some([0, 81_526_784, 86_622_208, 91_717_632])
        );
        assert_eq!(chunk_end(VOCAB - 128, VOCAB, CHUNK), Some(VOCAB));
        for invalid in [
            (0, 0, 128),
            (1, 256, 128),
            (0, 255, 128),
            (0, 256, 0),
            (0, 256, 127),
        ] {
            assert!(chunk_end(invalid.0, invalid.1, invalid.2).is_none());
        }
        assert!(projection_offsets(usize::MAX, HIDDEN).is_none());
        Ok(())
    }

    #[test]
    fn rotation_matches_independent_sylvester_matrix() {
        let input = (0..256)
            .map(|i| ((i * 37 % 103) as f32 - 51.) / 13.)
            .collect::<Vec<_>>();
        let mut actual = input.clone();
        rotate128(&mut actual);
        for (block, values) in input
            .as_chunks::<128>()
            .0
            .iter()
            .zip(actual.as_chunks::<128>().0)
        {
            for (i, &value) in values.iter().enumerate() {
                let expected = block
                    .iter()
                    .enumerate()
                    .map(|(j, &x)| {
                        if (i & j).count_ones() % 2 == 0 {
                            f64::from(x)
                        } else {
                            -f64::from(x)
                        }
                    })
                    .sum::<f64>()
                    / 128_f64.sqrt();
                assert!((f64::from(value) - expected).abs() < 1e-5);
            }
        }
    }

    fn fixture(path: &Path) -> Result<()> {
        let header = br#"{"a":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},"b":{"dtype":"U8","shape":[1],"data_offsets":[1,2]},"c":{"dtype":"U8","shape":[1],"data_offsets":[2,3]}}"#;
        let mut file = File::create(path)?;
        file.write_all(&(header.len() as u64).to_le_bytes())?;
        file.write_all(header)?;
        file.write_all(&[1, 2, 3])?;
        Ok(())
    }

    #[test]
    fn cache_reuses_only_complete_verified_entries_and_recovers() -> Result<()> {
        let root = tempfile::tempdir()?;
        let cancelled = AtomicBool::new(false);
        let path = cached_projection(root.path(), "test", b"metadata", &cancelled, fixture)?;
        let reused = cached_projection(root.path(), "test", b"metadata", &cancelled, |_| {
            panic!("cache rebuilt")
        })?;
        assert_eq!(path, reused);
        fs::write(path.join("projection.sha256"), "invalid")?;
        assert!(
            cached_projection(
                root.path(),
                "test",
                b"metadata",
                &cancelled,
                |_| anyhow::bail!("interrupted")
            )
            .is_err()
        );
        cached_projection(root.path(), "test", b"metadata", &cancelled, fixture)?;
        let file = OpenOptions::new()
            .write(true)
            .open(path.join("draft_head_q4.safetensors"))?;
        file.write_all_at(&[9], file.metadata()?.len() - 1)?;
        assert!(valid_cache(&path, b"metadata", &cancelled).is_err());
        cached_projection(root.path(), "test", b"metadata", &cancelled, fixture)?;
        assert!(
            cached_projection(root.path(), "test", b"changed", &cancelled, |_| {
                cancelled.store(true, Ordering::Relaxed);
                Ok(())
            })
            .is_err()
        );
        cancelled.store(false, Ordering::Relaxed);
        valid_cache(&path, b"metadata", &cancelled)?;
        assert_eq!(
            fs::read_dir(root.path())?.count(),
            2,
            "temporary files leaked"
        );
        Ok(())
    }

    #[test]
    fn source_identity_changes_with_head_bytes_and_cancellation_is_observed() -> Result<()> {
        let root = tempfile::tempdir()?;
        let file = root.path().join("source.safetensors");
        fixture(&file)?;
        let source = Checkpoint {
            path: root.path().to_owned(),
            model_type: "test".into(),
            bits: None,
            size_bytes: 0,
            modules: Vec::new(),
            tensors: read_header(&file)?,
        };
        let cancelled = AtomicBool::new(false);
        let old = source_key(&source, &cancelled)?;
        let f = OpenOptions::new().write(true).open(&file)?;
        f.write_all_at(&[8], f.metadata()?.len() - 1)?;
        assert_ne!(source_key(&source, &cancelled)?, old);
        cancelled.store(true, Ordering::Relaxed);
        assert!(source_key(&source, &cancelled).is_err());
        assert!(
            cached_projection(
                root.path(),
                "cancelled",
                b"metadata",
                &cancelled,
                |_| panic!("built after cancellation")
            )
            .is_err()
        );
        assert!(!root.path().join("cancelled").exists());
        Ok(())
    }

    #[cfg(feature = "mlx")]
    #[test]
    #[ignore = "requires local Qwen/Q4 reference and physical Apple GPU; run alone"]
    fn native_projection_reproduces_measured_q4_rows() -> Result<()> {
        let checkpoint =
            crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_MTP_TEST_MODEL")?))?;
        let source = source(&checkpoint).context("unsupported fixture head")?;
        let root = tempfile::tempdir()?;
        let output = root.path().join("generated.safetensors");
        let started = Instant::now();
        write_projection(&source, &vocabulary()?, &output, &AtomicBool::new(false))?;
        eprintln!(
            "Native compact preparation: {:.3}s",
            started.elapsed().as_secs_f64()
        );
        let actual = read_header(&output)?;
        let reference = read_header(
            &Path::new(&std::env::var("MLXL3_MTP_TEST_COMPACT_HEAD")?)
                .join("draft_head_q4.safetensors"),
        )?;
        assert_eq!(actual.len(), 3);
        for (name, expected) in reference {
            assert_eq!(actual[&name].shape, expected.shape);
            assert_eq!(actual[&name].dtype, expected.dtype);
            assert_eq!(
                actual[&name].read_bytes()?,
                expected.read_bytes()?,
                "{name}"
            );
            eprintln!(
                "{name}: all {} bytes match measured projection",
                expected.data_offsets[1] - expected.data_offsets[0]
            );
        }
        Ok(())
    }
}
