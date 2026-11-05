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

pub(super) fn compiled_bits(gpu: &Metal) -> Result<u32> {
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

fn float_bits(buffer: &Buffer, count: usize) -> Vec<u32> {
    Metal::read(buffer, count * 2)
        .chunks_exact(2)
        .map(|v| u32::from(v[0]) | (u32::from(v[1]) << 16))
        .collect()
}

#[test]
#[ignore = "physical AR20 B07 actual QMV vs CPU FMA tree, odd and empty split tails; run alone"]
fn physical_ar20_qmv_preload_preserves_tail_accumulators() -> Result<()> {
    let gpu = Metal::new()?;
    let bits = compiled_bits(&gpu)?;
    ensure!(bits & 4 != 0, "enable QMV_K_PRELOAD");
    let mut words = 0;
    let mut cases = 0;
    for k in [2usize, 3, 4] {
        for rows in [128usize, 256, 384] {
            let cols = 32;
            let packed = (0..rows / 16 * (cols / 16) * 16 * k)
                .map(|i| (i as u16).wrapping_mul(701).wrapping_add(33))
                .collect::<Vec<_>>();
            let weight = codec::reconstruct_inner(&packed, k, Codebook::Mul1, rows, cols)?;
            let x = (0..rows)
                .map(|i| f16::from_f32((i as f32 % 17. - 8.) / 23.).to_bits())
                .collect::<Vec<_>>();
            let input = upload(&gpu.device, &x)?;
            let trellis = upload(&gpu.device, &packed)?;
            let svh = upload(&gpu.device, &vec![f16::ONE.to_bits(); cols])?;
            for splits in [1usize, 2, 3] {
                let output = buffer(
                    &gpu.device,
                    (splits * cols * 4) as u64,
                    Storage::StorageModeShared,
                )?;
                gpu.command(|cmd| {
                    Metal::encode(
                        cmd,
                        &gpu.flash_qmv_pipeline,
                        &[&input, &trellis, &svh, &output],
                        [1, k as u32, 2, rows as u32, cols as u32],
                        MTLSize::new((cols / 16) as u64, 1, splits as u64),
                        128,
                    );
                    Ok(())
                })?;
                let got = float_bits(&output, splits * cols);
                let tiles = rows / 16;
                let per = tiles.div_ceil(splits);
                for split in 0..splits {
                    let begin = split * per;
                    let end = (begin + per).min(tiles);
                    for col in 0..cols {
                        let mut sum = 0f32;
                        for row in 0..16 {
                            for sg in 0..4 {
                                let mut part = 0f32;
                                for tile in ((begin + sg)..end).step_by(4) {
                                    let ix = tile * 16 + row;
                                    part = f16::from_bits(x[ix])
                                        .to_f32()
                                        .mul_add(weight[ix * cols + col].to_f32(), part);
                                }
                                sum += part;
                            }
                        }
                        ensure!(
                            sum.is_finite() && got[split * cols + col] == sum.to_bits(),
                            "QMV CPU oracle differs K{k} rows{rows} split{split}/{splits} col{col}"
                        );
                        words += 1;
                    }
                }
                cases += 1;
            }
        }
    }
    ensure!(cases == 27 && words == 1728);
    eprintln!(
        "AR20_QMV_PRELOAD {{\"parity\":true,\"cases\":{cases},\"finite_float_words\":{words},\"compiled_bits\":{bits},\"odd_and_empty_tails\":true}}"
    );
    Ok(())
}

#[test]
#[ignore = "physical AR20 B08 exact H128 Half boundaries and multi-route CTA; run alone"]
fn physical_ar20_h128_preserves_radix_rounding_and_route_groups() -> Result<()> {
    let gpu = Metal::new()?;
    let bits = compiled_bits(&gpu)?;
    ensure!(bits & 8 != 0, "enable H128_SIMD");
    let compare = pipeline(&gpu, "ar20_h128_compare")?;
    let routes = pipeline(&gpu, "ar20_h128_routes")?;
    ensure!(
        routes.thread_execution_width() == 32 && routes.max_total_threads_per_threadgroup() >= 640
    );
    let corpus = (0..8192)
        .map(|i| {
            let block = i / 128;
            match block % 4 {
                0 => f16::from_f32(((i * 13 + block * 7) % 257) as f32 / 91. - 128. / 91.).to_f32(),
                1 => f16::from_bits(
                    (((i * 37 + 1) % 1024) as u16) | if i % 2 == 0 { 0 } else { 0x8000 },
                )
                .to_f32(),
                2 => f32::from_bits(if i % 2 == 0 { 0 } else { 0x80000000 }),
                _ => f16::from_bits(
                    (i as u16).wrapping_mul(701).wrapping_add(block as u16 * 31) & 0xbfff,
                )
                .to_f32(),
            }
        })
        .collect::<Vec<_>>();
    ensure!(corpus.len() == 8192 && corpus.iter().all(|x| x.is_finite()));
    let input = upload(&gpu.device, &corpus)?;
    let output = || buffer(&gpu.device, 32768, Storage::StorageModeShared);
    let old = output()?;
    let actual = output()?;
    let bad = output()?;
    gpu.command(|cmd| {
        Metal::encode(
            cmd,
            &compare,
            &[&input, &old, &actual, &bad],
            [0u32],
            MTLSize::new(64, 1, 1),
            64,
        );
        Ok(())
    })?;
    let expected = float_bits(&old, 8192);
    ensure!(expected.iter().all(|&v| f32::from_bits(v).is_finite()));
    ensure!(
        float_bits(&actual, 8192) == expected && float_bits(&bad, 8192) != expected,
        "H128 or stage8 mutation oracle failed"
    );
    gpu.command(|cmd| {
        Metal::encode(
            cmd,
            &routes,
            &[&input, &old, &actual],
            [0u32],
            MTLSize::new(4, 1, 1),
            640,
        );
        Ok(())
    })?;
    let expected = float_bits(&old, 5120);
    ensure!(
        expected.iter().all(|&v| f32::from_bits(v).is_finite())
            && float_bits(&actual, 5120) == expected
    );
    eprintln!(
        "AR20_H128_SIMD {{\"parity\":true,\"finite_float_words\":8192,\"multi_route_float_words\":5120,\"compiled_bits\":{bits},\"middle_rounding_mutant_detected\":true,\"signed_zero_subnormal\":true}}"
    );
    Ok(())
}
