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

#[test]
#[ignore = "physical AR20 B06 codec windows, wraps and tail bits; run alone"]
fn physical_ar20_reader8_preserves_every_packed_state() -> Result<()> {
    let gpu = Metal::new()?;
    let bits = compiled_bits(&gpu)?;
    ensure!(bits & 2 != 0, "enable TRELLIS_READER8");
    let reader = pipeline(&gpu, "ar20_reader8_states")?;
    let mutant = pipeline(&gpu, "ar20_reader8_omit_high_word")?;
    let mut checked = 0;
    let mut detected = false;
    for k in 2usize..=8 {
        for pattern in 0..4 {
            let packed = (0..4 * 16 * k)
                .map(|i| match pattern {
                    0 => 0,
                    1 => u16::MAX,
                    2 => {
                        if i % 2 == 0 {
                            0xaaaa
                        } else {
                            0x5555
                        }
                    }
                    _ => (i as u16).wrapping_mul(701).wrapping_add(33),
                })
                .collect::<Vec<_>>();
            let source = upload(&gpu.device, &packed)?;
            let output = buffer(&gpu.device, 2048, Storage::StorageModeShared)?;
            let wrong = buffer(&gpu.device, 2048, Storage::StorageModeShared)?;
            gpu.command(|cmd| {
                Metal::encode(
                    cmd,
                    &reader,
                    &[&source, &output],
                    [k as u32],
                    MTLSize::new(4, 1, 1),
                    32,
                );
                if k <= 6 {
                    Metal::encode(
                        cmd,
                        &mutant,
                        &[&source, &wrong],
                        [k as u32],
                        MTLSize::new(4, 1, 1),
                        32,
                    );
                }
                Ok(())
            })?;
            let expected = codec::unpack(&packed, k)?;
            ensure!(expected.len() == 1024 && Metal::read(&output, 1024) == expected);
            if k <= 6 {
                detected |= Metal::read(&wrong, 1024) != expected;
            }
            checked += expected.len();
        }
    }
    ensure!(detected && checked == 28672);
    eprintln!(
        "AR20_READER8 {{\"parity\":true,\"state_words\":{checked},\"compiled_bits\":{bits},\"high_word_mutant_detected\":true,\"k7_k8_fallback\":true}}"
    );
    Ok(())
}
