use super::*;

fn pipeline(gpu: &Metal, name: &str) -> Result<ComputePipelineState> {
    let function = gpu
        .ar20_library
        .get_function(name, None)
        .map_err(anyhow::Error::msg)?;
    gpu.device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(anyhow::Error::msg)
}

pub(super) fn compiled_cooperative(gpu: &Metal) -> Result<bool> {
    let pipeline = pipeline(gpu, "ar30_compiled_cooperative")?;
    let out = buffer(&gpu.device, 4, Storage::StorageModeShared)?;
    gpu.command(|cmd| {
        Metal::encode(cmd, &pipeline, &[&out], [0u32], MTLSize::new(1, 1, 1), 1);
        Ok(())
    })?;
    let words = Metal::read(&out, 2);
    ensure!(words.len() == 2 && words[1] == 0 && words[0] <= 1);
    ensure!(words[0] == u16::from(gpu.ar30_cooperative));
    Ok(words[0] == 1)
}

// Every single packed bit is a basis vector for these data-independent linear
// bit windows. The corpus also checks zero, all-one, alternating and mixed words.
fn packed_basis(k: usize) -> Vec<u16> {
    let tile_words = 16 * k;
    let mut packed = vec![0; (tile_words * 16 + 4) * tile_words];
    for bit in 0..tile_words * 16 {
        packed[bit * tile_words + bit / 16] = 1 << (bit % 16);
    }
    let end = tile_words * tile_words * 16;
    for pattern in 0..4 {
        for word in 0..tile_words {
            packed[end + pattern * tile_words + word] = match pattern {
                0 => 0,
                1 => u16::MAX,
                2 => {
                    if word % 2 == 0 {
                        0xaaaa
                    } else {
                        0x5555
                    }
                }
                _ => (word as u16).wrapping_mul(701).wrapping_add(33),
            };
        }
    }
    packed
}

#[test]
#[ignore = "physical AR30 C01 complete packed-bit basis, wraps and K5+ fallback; run alone"]
fn physical_ar30_cooperative_words_preserve_every_window_bit() -> Result<()> {
    let gpu = Metal::new()?;
    ensure!(compiled_cooperative(&gpu)?, "enable QMV_COOPERATIVE");
    let reader = pipeline(&gpu, "ar30_cooperative_states")?;
    let mutant = pipeline(&gpu, "ar30_cooperative_wrong_owner")?;
    let mut checked = 0;
    let mut mutant_rates = 0;
    for k in 2usize..=8 {
        let packed = packed_basis(k);
        let tiles = packed.len() / (16 * k);
        let count = tiles * 256;
        let source = upload(&gpu.device, &packed)?;
        let actual = buffer(&gpu.device, count as u64 * 2, Storage::StorageModeShared)?;
        let canonical = buffer(&gpu.device, count as u64 * 2, Storage::StorageModeShared)?;
        let wrong = buffer(&gpu.device, count as u64 * 2, Storage::StorageModeShared)?;
        gpu.command(|cmd| {
            Metal::encode(
                cmd,
                &reader,
                &[&source, &actual, &canonical],
                [k as u32],
                MTLSize::new(tiles as u64, 1, 1),
                32,
            );
            if k <= 4 {
                Metal::encode(
                    cmd,
                    &mutant,
                    &[&source, &wrong],
                    [k as u32],
                    MTLSize::new(tiles as u64, 1, 1),
                    32,
                );
            }
            Ok(())
        })?;
        let expected = codec::unpack(&packed, k)?;
        ensure!(expected.len() == count && count > 0);
        ensure!(
            Metal::read(&actual, count) == expected,
            "cooperative mismatch K{k}"
        );
        ensure!(
            Metal::read(&canonical, count) == expected,
            "canonical mismatch K{k}"
        );
        if k <= 4 {
            ensure!(
                Metal::read(&wrong, count) != expected,
                "wrong-owner mutant undetected K{k}"
            );
            mutant_rates += 1;
        }
        checked += count;
    }
    ensure!(checked == 2_300_928 && mutant_rates == 3);
    eprintln!(
        "AR30_COOPERATIVE_WINDOWS {}",
        serde_json::json!({
            "parity":true,"compiled_cooperative":true,"state_words":checked,
            "supported_rates":[2,3,4],"fallback_rates":[5,6,7,8],
            "basis_vectors":8960,"mutant_rates_detected":mutant_rates
        })
    );
    Ok(())
}
