use super::*;
use half::f16;

fn pipeline(gpu: &Metal, name: &str) -> Result<ComputePipelineState> {
    let function = gpu
        .ar20_library
        .get_function(name, None)
        .map_err(anyhow::Error::msg)?;
    gpu.device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(anyhow::Error::msg)
}

fn compiled_bits(gpu: &Metal) -> Result<u32> {
    let p = pipeline(gpu, "ar20_compiled_bits")?;
    let out = buffer(&gpu.device, 4, Storage::StorageModeShared)?;
    gpu.command(|cmd| {
        Metal::encode(cmd, &p, &[&out], [0u32], MTLSize::new(1, 1, 1), 1);
        Ok(())
    })?;
    let words = Metal::read(&out, 2);
    let bits = u32::from(words[0]) | (u32::from(words[1]) << 16);
    ensure!(bits == gpu.ar20_kernel_bits);
    Ok(bits)
}

#[test]
#[ignore = "physical AR20 B05 exhaustive codebook and actual compiled switch; run alone"]
fn physical_ar20_codebook_preserves_all_half_states() -> Result<()> {
    let gpu = Metal::new()?;
    let bits = compiled_bits(&gpu)?;
    ensure!(bits & 1 != 0, "enable MUL1_SWAR");
    let canonical = pipeline(&gpu, "ar20_codebook_original")?;
    let mutant = pipeline(&gpu, "ar20_codebook_omit_high_byte")?;
    let original = buffer(&gpu.device, 131072, Storage::StorageModeShared)?;
    let wrong = buffer(&gpu.device, 131072, Storage::StorageModeShared)?;
    gpu.command(|cmd| {
        for (p, out) in [(&canonical, &original), (&mutant, &wrong)] {
            Metal::encode(cmd, p, &[out], [0u32], MTLSize::new(256, 1, 1), 256);
        }
        Ok(())
    })?;
    let actual = gpu.codebook(Codebook::Mul1)?;
    let expected = Metal::read(&original, 65536);
    ensure!(actual == expected && actual.len() == 65536);
    for (cw, &word) in actual.iter().enumerate() {
        ensure!(
            f16::from_bits(word).is_finite()
                && word == codec::decode_codeword(cw as u16, Codebook::Mul1).to_bits()
        );
    }
    ensure!(
        Metal::read(&wrong, 65536) != expected,
        "bad byte-sum mutation was not detected"
    );
    eprintln!(
        "AR20_MUL1_SWAR {{\"parity\":true,\"finite_half_words\":65536,\"compiled_bits\":{bits},\"mutant_detected\":true}}"
    );
    Ok(())
}
