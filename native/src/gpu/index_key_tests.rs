use super::*;
use crate::tensor_layout::MatrixLayout;
use half::f16;

fn words(gpu: &Metal, buffer: &Buffer, count: u32) -> Result<Vec<u32>> {
    let full = u32::try_from(buffer.length() / 2)?;
    ensure!(count > 0 && count <= full && buffer.length().is_multiple_of(2));
    let bytes = MatrixView {
        buffer: buffer.clone(),
        layout: MatrixLayout {
            shape: [1, full],
            strides: [i64::from(full), 1],
            offset: 0,
        },
        dtype: FloatType::F16,
    }
    .export_bytes(gpu)?;
    ensure!(bytes.len() == full as usize * 2);
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .take(count as usize)
        .map(|b| u32::from(u16::from_le_bytes(*b)))
        .collect())
}

#[test]
#[ignore = "physical compact index K, full canonical projection, tails and sparse transition"]
fn physical_index_key_matches_full_projection_and_sparse_transition() -> Result<()> {
    let c = crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    ensure!(gpu.flash_index_key_qmv.is_some() && gpu.qsa.pool_key.is_some());
    let qsa = FlashQsa::load(
        &gpu,
        &c,
        "model.language_model.layers.3.self_attn",
        1e-6,
        10000.,
        4096,
    )?;
    let index = qsa.index.as_ref().context("real QSA index required")?;
    let input = (0..2560)
        .map(|i| ((i * 17 % 257) as f32 - 128.) / 237.)
        .collect::<Vec<_>>();
    let x = gpu.f32_storage(&input)?.matrix(MatrixLayout {
        shape: [1, 2560],
        strides: [2560, 1],
        offset: 0,
    })?;
    let full = gpu.command(|cmd| index.qk.encode_row(cmd, &x, Storage::StorageModeShared))?;
    let key = gpu
        .command(|cmd| {
            index
                .qk
                .encode_index_key(cmd, &x, Storage::StorageModeShared)
        })?
        .context("compact kernel fell back")?;
    let expected = full.read_output()?;
    let actual = key.read_output()?;
    ensure!(expected.len() == 640 && actual.len() == 128 && actual == expected[512..]);
    ensure!(actual.iter().all(|&v| f16::from_bits(v as u16).is_finite()));
    let mut states = [qsa.state.clone(), qsa.state.clone()];
    for state in &mut states {
        state.offset = 2046;
        state.slab = Slab::new(&gpu, 4096, true)?;
        state.tail = Some(super::super::upload(
            &gpu.device,
            &vec![f16::from_f32(0.25).to_bits(); 256],
        )?);
        let values = (0..1024 * 128)
            .map(|i| f16::from_f32(((i * 11 % 257) as f32 - 128.) / 997.).to_bits())
            .collect::<Vec<_>>();
        let source = super::super::upload(&gpu.device, &values)?;
        gpu.command(|cmd| {
            let blit = cmd.new_blit_command_encoder();
            blit.copy_from_buffer(
                &source,
                0,
                state.slab.blocks.as_ref().unwrap(),
                0,
                source.length(),
            );
            blit.end_encoding();
            Ok(())
        })?;
    }
    let mut cases = Vec::new();
    for offset in 2046..2054 {
        gpu.qsa_index_key_only.set(false);
        let expected = index.append(&gpu, &states[0].slab, &states[0], &x, 1e-6, false)?;
        gpu.qsa_index_key_only.set(true);
        let uses = gpu.index_key_uses.get();
        let actual = index.append(&gpu, &states[1].slab, &states[1], &x, 1e-6, false)?;
        let taken = gpu.index_key_uses.get() - uses;
        ensure!(taken == usize::from(offset < 2051) && expected.positions == actual.positions);
        ensure!(expected.trace.is_empty() && actual.trace.is_empty());
        match (&expected.tail, &actual.tail) {
            (Some(a), Some(b)) => {
                ensure!(
                    words(&gpu, a, (offset + 1) % 4 * 128)?
                        == words(&gpu, b, (offset + 1) % 4 * 128)?
                )
            }
            (None, None) => (),
            _ => anyhow::bail!("tail shape changed"),
        }
        let blocks = (offset + 1) / 4 * 128;
        let a = words(&gpu, states[0].slab.blocks.as_ref().unwrap(), blocks)?;
        let b = words(&gpu, states[1].slab.blocks.as_ref().unwrap(), blocks)?;
        ensure!(a == b && a.iter().all(|&v| f16::from_bits(v as u16).is_finite()));
        states[0].tail = expected.tail;
        states[1].tail = actual.tail;
        for state in &mut states {
            state.offset += 1;
        }
        cases.push(serde_json::json!({"offset":offset,"compact_uses":taken,"blocks_words":blocks}));
    }
    states[1].offset = 17;
    states[1].tail = Some(super::super::upload(
        &gpu.device,
        &vec![f16::from_f32(0.25).to_bits(); 128],
    )?);
    let uses = gpu.index_key_uses.get();
    let traced = index.append(&gpu, &states[1].slab, &states[1], &x, 1e-6, true)?;
    ensure!(gpu.index_key_uses.get() == uses && traced.trace["index_qk"].len() == 640);
    eprintln!(
        "AR42_INDEX_KEY {}",
        serde_json::json!({"parity":true,"key_half_words":128,"cases":cases,"trace_fallback":true,"sparse_transition":true})
    );
    Ok(())
}
