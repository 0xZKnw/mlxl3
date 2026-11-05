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

pub(super) fn compiled_codeword_table(gpu: &Metal) -> Result<bool> {
    let p = pipeline(gpu, "ar33_compiled_codeword_table")?;
    let out = buffer(&gpu.device, 4, Storage::StorageModeShared)?;
    gpu.command(|cmd| {
        Metal::encode(cmd, &p, &[&out], [0u32], MTLSize::new(1, 1, 1), 1);
        Ok(())
    })?;
    let words = Metal::read(&out, 2);
    ensure!(words.len() == 2 && words[1] == 0 && words[0] <= 1);
    ensure!(words[0] == u16::from(gpu.ar33_codeword_table));
    Ok(words[0] == 1)
}

#[test]
#[ignore = "physical AR33 T01 exhaustive canonical codewords, high bits, table endpoints and tails; run alone"]
fn physical_ar33_codeword_table_matches_canonical_every_state() -> Result<()> {
    let gpu = Metal::new()?;
    ensure!(
        compiled_codeword_table(&gpu)?,
        "enable CODEWORD_TABLE on exact M5"
    );
    let states = pipeline(&gpu, "ar33_codeword_states")?;
    let entries = pipeline(&gpu, "ar33_codeword_entries")?;
    let sentinel = f16::MAX.to_bits();
    let mut checked = 0;
    let mut mutations = 0;
    for cb in [Codebook::Default, Codebook::Mcg, Codebook::Mul1] {
        for high in [0u32, 0xffff0000, 0xa5a50000] {
            for count in [65536usize, 257] {
                let seed = vec![sentinel; count + 17];
                let actual = upload(&gpu.device, &seed)?;
                let canonical = upload(&gpu.device, &seed)?;
                let mutant = upload(&gpu.device, &seed)?;
                gpu.command(|cmd| {
                    Metal::encode(
                        cmd,
                        &states,
                        &[&actual, &canonical, &mutant],
                        [cb as u32, high, count as u32, 0],
                        MTLSize::new(count.div_ceil(256) as u64, 1, 1),
                        256,
                    );
                    Ok(())
                })?;
                let actual = Metal::read(&actual, seed.len());
                let canonical = Metal::read(&canonical, seed.len());
                let mutant = Metal::read(&mutant, seed.len());
                ensure!(actual.len() == seed.len() && actual == canonical);
                ensure!(actual[count..] == seed[count..] && mutant[count..] == seed[count..]);
                for (cw, &word) in actual[..count].iter().enumerate() {
                    ensure!(
                        f16::from_bits(word).is_finite()
                            && word == codec::decode_codeword(cw as u16, cb).to_bits()
                    );
                }
                if cb == Codebook::Mul1 {
                    ensure!(
                        mutant[..count] != canonical[..count],
                        "index-shift mutant escaped"
                    );
                    mutations += 1;
                }
                checked += count;
            }
        }
    }
    let seed = vec![sentinel; 1021 + 17];
    let actual = upload(&gpu.device, &seed)?;
    let canonical = upload(&gpu.device, &seed)?;
    gpu.command(|cmd| {
        Metal::encode(
            cmd,
            &entries,
            &[&actual, &canonical],
            [1021u32],
            MTLSize::new(5, 1, 1),
            256,
        );
        Ok(())
    })?;
    let actual = Metal::read(&actual, seed.len());
    let canonical = Metal::read(&canonical, seed.len());
    ensure!(actual.len() == seed.len() && actual == canonical && actual[1021..] == seed[1021..]);
    ensure!(
        actual[..1021]
            .iter()
            .all(|&v| f16::from_bits(v).is_finite())
    );
    ensure!(checked == 592137 && mutations == 6);
    eprintln!(
        "AR33_CODEWORD_TABLE {}",
        serde_json::json!({
            "parity":true,"compiled_table":true,"finite_state_words":checked,
            "table_entries":1021,"table_bytes":2042,"codebooks":3,"high_masks":3,
            "tail_guards":true,"index_mutants_detected":mutations
        })
    );
    Ok(())
}
