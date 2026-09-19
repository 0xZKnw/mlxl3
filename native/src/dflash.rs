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

// Kani 0.68 ICEs on the equivalent integer `is_multiple_of` intrinsic.
#[allow(clippy::manual_is_multiple_of)]
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

#[cfg(feature = "mlx")]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Q4Kernel {
    N128,
    N128Pipelined,
    N256,
}

#[cfg(feature = "mlx")]
pub struct Q4Projection {
    weights: crate::array::Array,
    scales: crate::array::Array,
    biases: crate::array::Array,
    input: i32,
    output: i32,
}

#[cfg(feature = "mlx")]
impl Q4Projection {
    pub fn from_section(
        file: &File,
        section: &PackedSection,
        input: i32,
        output: i32,
    ) -> Result<Self> {
        use crate::array::{Array, Dtype};

        ensure!(input > 0 && output > 0, "invalid Q4 projection shape");
        let elements = i64::from(input)
            .checked_mul(i64::from(output))
            .context("Q4 projection shape overflow")?;
        let weights = elements / 2;
        let parameters = elements / 64;
        ensure!(
            section.bytes == u64::try_from(weights + parameters * 4)?,
            "Q4 section size does not match projection shape"
        );
        let scales_offset = section
            .offset
            .checked_add(u64::try_from(weights)?)
            .context("Q4 scale offset overflow")?;
        let biases_offset = scales_offset
            .checked_add(u64::try_from(parameters * 2)?)
            .context("Q4 bias offset overflow")?;
        let projection = Self {
            weights: Array::from_file(
                file,
                section.offset,
                &[i32::try_from(weights)?],
                Dtype::UInt8,
            )?,
            scales: Array::from_file(
                file,
                scales_offset,
                &[i32::try_from(parameters)?],
                Dtype::BFloat16,
            )?,
            biases: Array::from_file(
                file,
                biases_offset,
                &[i32::try_from(parameters)?],
                Dtype::BFloat16,
            )?,
            input,
            output,
        };
        projection.weights.eval()?;
        projection.scales.eval()?;
        projection.biases.eval()?;
        Ok(projection)
    }

    pub fn forward(
        &self,
        input: &crate::array::Array,
        kernel: Q4Kernel,
        groups: i32,
    ) -> Result<crate::array::Array> {
        use crate::array::{self, Dtype};

        ensure!(
            input.shape() == [8, self.input] && input.dtype() == Dtype::BFloat16,
            "DFlash Q4 decode expects BF16 [8,input]"
        );
        let (tile, pipelined) = match kernel {
            Q4Kernel::N128 => (128, false),
            Q4Kernel::N128Pipelined => (128, true),
            Q4Kernel::N256 => (256, false),
        };
        let tiles = self.output / tile;
        ensure!(
            self.input % 256 == 0 && self.output % 256 == 0 && groups > 0 && groups <= tiles,
            "invalid DFlash Q4 dispatch geometry"
        );
        let header = format!(
            "#define DFLASH_INPUT {}\n#define DFLASH_OUTPUT {}\n#define DFLASH_TILE {}\n#define DFLASH_GROUPS {}\n#define DFLASH_PIPELINED {}\n{}\n",
            self.input,
            self.output,
            tile,
            groups,
            if pipelined { "true" } else { "false" },
            include_str!("../shaders/dflash_q4_mpp.h")
        );
        Ok(array::metal_kernel(
            &format!(
                "mlxl3_dflash_q4_{}_{}_t{tile}_g{groups}_p{}",
                self.input,
                self.output,
                u8::from(pipelined)
            ),
            &["input", "weights", "scales", "biases"],
            &["output"],
            &header,
            include_str!("../shaders/dflash_q4.metal"),
            &[input, &self.weights, &self.scales, &self.biases],
            &[vec![8, self.output]],
            &[Dtype::BFloat16],
            [groups.checked_mul(256).context("Q4 grid overflow")?, 1, 1],
            [256, 1, 1],
        )?
        .remove(0))
    }
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

    #[cfg(feature = "mlx")]
    #[test]
    #[ignore = "requires Apple M5 and the downloaded DFlash package"]
    fn benchmarks_real_q4_qkv_kernels() -> Result<()> {
        use crate::array::{Array, Dtype};
        use std::time::Instant;

        ensure!(crate::array::is_m5_gpu()?, "benchmark requires Apple M5");
        let package = inspect("models/Qwen3.6-35B-A3B-DFlash2")?;
        let section = package.layers[0]
            .sections
            .iter()
            .find(|section| section.name == "qkv")
            .context("missing qkv section")?;
        let file = File::open(&package.layers[0].path)?;
        let projection = Q4Projection::from_section(&file, section, 2048, 6144)?;
        let values = (0..8 * 2048)
            .map(|index| ((index * 37 % 257) as f32 - 128.0) / 128.0)
            .collect::<Vec<_>>();
        let input = Array::from_f32(&values, &[8, 2048])?.astype(Dtype::BFloat16)?;
        input.eval()?;

        let candidates = [
            (Q4Kernel::N128, 24),
            (Q4Kernel::N128, 36),
            (Q4Kernel::N128, 40),
            (Q4Kernel::N128, 44),
            (Q4Kernel::N128, 48),
            (Q4Kernel::N128Pipelined, 24),
            (Q4Kernel::N128Pipelined, 36),
            (Q4Kernel::N128Pipelined, 40),
            (Q4Kernel::N128Pipelined, 44),
            (Q4Kernel::N128Pipelined, 48),
            (Q4Kernel::N256, 12),
            (Q4Kernel::N256, 18),
            (Q4Kernel::N256, 20),
            (Q4Kernel::N256, 22),
            (Q4Kernel::N256, 24),
        ];
        let reference = projection.forward(&input, Q4Kernel::N128, 48)?;
        reference.eval()?;
        let reference = reference.to_bytes()?;
        let packed = std::fs::read(&package.layers[0].path)?;
        let weight_bytes = 6144_usize * 2048 / 2;
        let parameter_bytes = 6144_usize * 2048 / 32;
        let weight_base = usize::try_from(section.offset)?;
        let scale_base = weight_base + weight_bytes;
        let bias_base = scale_base + parameter_bytes;
        let bf16_at = |offset: usize| {
            half::bf16::from_bits(u16::from_le_bytes([packed[offset], packed[offset + 1]])).to_f32()
        };
        let gpu_at = |row: usize, column: usize| {
            let offset = (row * 6144 + column) * 2;
            half::bf16::from_bits(u16::from_ne_bytes([
                reference[offset],
                reference[offset + 1],
            ]))
            .to_f32()
        };
        let cpu_error = |low_nibble_first: bool| {
            let mut maximum = 0.0_f32;
            let mut total = 0.0_f32;
            for row in 0..8 {
                for column in 0..64 {
                    let tile = column / 256;
                    let within_tile = column % 256;
                    let mut accumulated = 0.0_f32;
                    for group in 0..32 {
                        let mut dot = 0.0_f32;
                        let mut sum = 0.0_f32;
                        for inner in 0..64 {
                            let input =
                                half::bf16::from_f32(values[row * 2048 + group * 64 + inner])
                                    .to_f32();
                            let offset = weight_base
                                + tile * 32 * 256 * 32
                                + (group * 256 + within_tile) * 32
                                + inner / 2;
                            let byte = packed[offset];
                            let low = inner % 2 == 0;
                            let weight = if low == low_nibble_first {
                                byte & 0x0f
                            } else {
                                byte >> 4
                            };
                            dot = input.mul_add(f32::from(weight), dot);
                            sum += input;
                        }
                        let parameter = (tile * 32 + group) * 256 + within_tile;
                        accumulated += dot * bf16_at(scale_base + parameter * 2)
                            + sum * bf16_at(bias_base + parameter * 2);
                    }
                    let error =
                        (half::bf16::from_f32(accumulated).to_f32() - gpu_at(row, column)).abs();
                    maximum = maximum.max(error);
                    total += error;
                }
            }
            (maximum, total / (8.0 * 64.0))
        };
        let low_error = cpu_error(true);
        let high_error = cpu_error(false);
        eprintln!("DFlash Q4 CPU check low={low_error:?}, high={high_error:?}");
        assert!(
            low_error.0 <= 0.125 && low_error.1 <= 0.01,
            "Q4 CPU reference diverged: {low_error:?}"
        );
        assert!(
            low_error.1 * 10.0 < high_error.1,
            "Q4 nibble order was not independently distinguished"
        );
        for &(kernel, groups) in &candidates {
            let output = projection.forward(&input, kernel, groups)?;
            output.eval()?;
            assert_eq!(
                output.to_bytes()?,
                reference,
                "{kernel:?} g{groups} changed BF16 output"
            );
        }
        for _ in 0..100 {
            projection.forward(&input, Q4Kernel::N256, 24)?.eval()?;
        }
        let mut timings = vec![Vec::with_capacity(30); candidates.len()];
        for round in 0..6 {
            for position in 0..candidates.len() {
                let index = if round % 2 == 0 {
                    position
                } else {
                    candidates.len() - 1 - position
                };
                let (kernel, groups) = candidates[index];
                for _ in 0..5 {
                    let start = Instant::now();
                    projection.forward(&input, kernel, groups)?.eval()?;
                    timings[index].push(start.elapsed());
                }
            }
        }
        for (index, (kernel, groups)) in candidates.into_iter().enumerate() {
            let samples = &mut timings[index];
            samples.sort_unstable();
            eprintln!(
                "DFlash Q4 {kernel:?} g{groups}: median {:?}, p10 {:?}, p90 {:?}",
                samples[samples.len() / 2],
                samples[samples.len() / 10],
                samples[samples.len() * 9 / 10]
            );
        }
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
