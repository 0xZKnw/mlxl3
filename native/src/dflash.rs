//! Strict reader for IncoAI's packed DFlash 2 draft files.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};

const MAGIC: &[u8; 8] = b"MDFD0004";
const ALIGNMENT: u64 = 16 * 1024;
const HEADER_BYTES: u64 = 16;

pub const LAYERS: usize = 6;
pub const HIDDEN: u64 = 2048;
pub const DYNAMIC: u64 = 512;
pub const QKV: u64 = 6144;
pub const ATTENTION: u64 = 4096;
pub const INTERMEDIATE: u64 = 6144;
pub const TARGET_HIDDEN: u64 = 8 * HIDDEN;
pub const SELECTOR: u64 = 256;
pub const VOCABULARY: u64 = 248_320;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedSection {
    pub name: &'static str,
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedFile {
    pub path: PathBuf,
    pub layer: u32,
    pub kind: u32,
    pub bytes: u64,
    pub sections: Vec<PackedSection>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DFlashPackage {
    pub directory: PathBuf,
    pub layers: Vec<PackedFile>,
    pub model: PackedFile,
}

#[derive(Default)]
struct Plan {
    cursor: u64,
    sections: Vec<PackedSection>,
}

impl Plan {
    fn new() -> Self {
        Self {
            cursor: HEADER_BYTES,
            sections: Vec::new(),
        }
    }

    fn section(&mut self, name: &'static str, bytes: u64) -> Result<()> {
        self.cursor = align(self.cursor).context("DFlash section offset overflow")?;
        self.sections.push(PackedSection {
            name,
            offset: self.cursor,
            bytes,
        });
        self.cursor = self
            .cursor
            .checked_add(bytes)
            .context("DFlash section size overflow")?;
        Ok(())
    }

    fn bf16(&mut self, name: &'static str, elements: u64) -> Result<()> {
        self.section(
            name,
            elements
                .checked_mul(2)
                .context("DFlash BF16 size overflow")?,
        )
    }

    fn q4(&mut self, name: &'static str, output: u64, input: u64) -> Result<()> {
        self.section(name, q4_packed_bytes(output, input)?)
    }
}

fn align(value: u64) -> Option<u64> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|sum| sum & !(ALIGNMENT - 1))
}

fn q4_packed_bytes(output: u64, input: u64) -> Result<u64> {
    ensure!(
        valid_q4_geometry(output, input),
        "invalid DFlash Q4 geometry"
    );
    q4_packed_bytes_checked(output, input).context("DFlash Q4 size overflow")
}

fn valid_q4_geometry(output: u64, input: u64) -> bool {
    output > 0 && output % 256 == 0 && input > 0 && input % 64 == 0
}

fn q4_packed_bytes_checked(output: u64, input: u64) -> Option<u64> {
    output
        .checked_mul(input)
        .and_then(|elements| elements.checked_mul(9))
        .map(|bytes| bytes / 16)
}

fn layer_plan() -> Result<Plan> {
    let mut plan = Plan::new();
    plan.bf16("input_norm", HIDDEN)?;
    plan.bf16("attention_convolution", 4 * HIDDEN)?;
    plan.q4("attention_dynamic", DYNAMIC, HIDDEN)?;
    plan.q4("qkv", QKV, HIDDEN)?;
    plan.bf16("query_norm", 128)?;
    plan.bf16("key_norm", 128)?;
    plan.q4("attention_output", HIDDEN, ATTENTION)?;
    plan.bf16("post_attention_norm", HIDDEN)?;
    plan.bf16("mlp_convolution", 4 * HIDDEN)?;
    plan.q4("mlp_dynamic", DYNAMIC, HIDDEN)?;
    plan.q4("gate", INTERMEDIATE, HIDDEN)?;
    plan.q4("up", INTERMEDIATE, HIDDEN)?;
    plan.q4("down", HIDDEN, INTERMEDIATE)?;
    Ok(plan)
}

fn model_plan() -> Result<Plan> {
    let mut plan = Plan::new();
    plan.q4("context_projection", HIDDEN, TARGET_HIDDEN)?;
    plan.bf16("hidden_norm", HIDDEN)?;
    plan.bf16("final_norm", HIDDEN)?;
    plan.q4("selector_projection", SELECTOR, HIDDEN)?;
    plan.bf16("predecessor_codebook", VOCABULARY * SELECTOR)?;
    plan.bf16("successor_codebook", VOCABULARY * SELECTOR)?;
    Ok(plan)
}

fn inspect_file(path: PathBuf, layer: u32, kind: u32, plan: Plan) -> Result<PackedFile> {
    let metadata = path
        .metadata()
        .with_context(|| format!("reading {} metadata", path.display()))?;
    ensure!(metadata.is_file(), "{} is not a file", path.display());
    ensure!(
        metadata.len() == plan.cursor,
        "{} has {} bytes; expected {}",
        path.display(),
        metadata.len(),
        plan.cursor
    );

    let mut header = [0_u8; HEADER_BYTES as usize];
    File::open(&path)
        .with_context(|| format!("opening {}", path.display()))?
        .read_exact(&mut header)
        .with_context(|| format!("reading {} header", path.display()))?;
    ensure!(
        &header[..8] == MAGIC,
        "invalid DFlash magic in {}",
        path.display()
    );
    ensure!(
        u32::from_le_bytes(header[8..12].try_into().unwrap()) == layer,
        "invalid DFlash layer id in {}",
        path.display()
    );
    ensure!(
        u32::from_le_bytes(header[12..16].try_into().unwrap()) == kind,
        "invalid DFlash file type in {}",
        path.display()
    );
    ensure!(
        plan.sections.iter().all(|section| {
            section.offset % ALIGNMENT == 0
                && section
                    .offset
                    .checked_add(section.bytes)
                    .is_some_and(|end| end <= metadata.len())
        }),
        "invalid DFlash section layout in {}",
        path.display()
    );

    Ok(PackedFile {
        path,
        layer,
        kind,
        bytes: metadata.len(),
        sections: plan.sections,
    })
}

/// Validate the complete draft subtree without reading or requiring Splash's
/// target weights. The returned offsets can be mapped directly by the runtime.
pub fn inspect(directory: impl AsRef<Path>) -> Result<DFlashPackage> {
    let directory = directory
        .as_ref()
        .canonicalize()
        .with_context(|| format!("opening DFlash package {}", directory.as_ref().display()))?;
    let draft = directory.join("draft");
    let layers = (0..LAYERS)
        .map(|layer| {
            inspect_file(
                draft.join(format!("layer-{layer}.bin")),
                layer as u32,
                0,
                layer_plan()?,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let model = inspect_file(draft.join("model.bin"), LAYERS as u32, 1, model_plan()?)?;
    Ok(DFlashPackage {
        directory,
        layers,
        model,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom, Write};

    #[test]
    fn plans_match_the_published_splash_layout() -> Result<()> {
        let layer = layer_plan()?;
        assert_eq!(layer.cursor, 34_324_480);
        assert_eq!(layer.sections[0].offset, 16_384);
        assert_eq!(layer.sections[3].offset, 638_976);
        assert_eq!(layer.sections[12].offset, 27_246_592);

        let model = model_plan()?;
        assert_eq!(model.cursor, 273_498_112);
        assert_eq!(model.sections[0].offset, 16_384);
        assert_eq!(model.sections[4].offset, 19_218_432);
        assert_eq!(model.sections[5].offset, 146_358_272);
        Ok(())
    }

    #[test]
    fn rejects_a_bad_header_before_exposing_offsets() -> Result<()> {
        let temporary = tempfile::tempdir()?;
        let path = temporary.path().join("layer-0.bin");
        let plan = layer_plan()?;
        let mut file = File::create(&path)?;
        file.set_len(plan.cursor)?;
        file.seek(SeekFrom::Start(0))?;
        file.write_all(b"NOTDFLASH\0\0\0\0\0\0\0")?;
        drop(file);
        assert!(inspect_file(path, 0, 0, plan).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires the downloaded 457 MiB DFlash package"]
    fn inspects_the_real_qwen_draft_package() -> Result<()> {
        let package = inspect("models/Qwen3.6-35B-A3B-DFlash2")?;
        assert_eq!(package.layers.len(), LAYERS);
        assert_eq!(package.model.bytes, 273_498_112);
        Ok(())
    }
}

#[cfg(kani)]
mod verification {
    use super::*;

    #[kani::proof]
    fn alignment_is_monotonic_and_aligned_when_representable() {
        let value: u64 = kani::any();
        if let Some(aligned) = align(value) {
            assert!(aligned >= value);
            assert_eq!(aligned % ALIGNMENT, 0);
            assert!(aligned - value < ALIGNMENT);
        }
    }

    #[kani::proof]
    fn q4_geometry_accepts_exactly_the_storage_multiples() {
        let output: u16 = kani::any();
        let input: u16 = kani::any();
        let valid = output != 0 && output % 256 == 0 && input != 0 && input % 64 == 0;
        kani::cover!(valid);
        kani::cover!(!valid);
        assert_eq!(valid_q4_geometry(output as u64, input as u64), valid);
        if valid {
            assert_eq!(
                q4_packed_bytes_checked(output as u64, input as u64),
                Some(output as u64 * input as u64 * 9 / 16)
            );
        }
    }
}
