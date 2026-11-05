use super::*;
use crate::tensor_layout::MatrixLayout;
use half::f16;

fn finite(bytes: &[u8], half: bool) -> bool {
    if half {
        bytes.len().is_multiple_of(2)
            && bytes
                .as_chunks::<2>()
                .0
                .iter()
                .all(|v| f16::from_bits(u16::from_le_bytes(*v)).is_finite())
    } else {
        bytes.len().is_multiple_of(4)
            && bytes
                .as_chunks::<4>()
                .0
                .iter()
                .all(|v| f32::from_le_bytes(*v).is_finite())
    }
}
#[test]
#[ignore = "physical native GDN head-gate broadcast and actual two-step retained frames; exclusive GPU"]
fn physical_ar20_head_gates_preserve_boundary_corpus_and_retained_frames() -> Result<()> {
    let gpu = Metal::new()?;
    ensure!(gpu.gdn_whole.get() && gpu.gdn_head_gates.get());
    let canonical = gpu.gdn.whole.as_ref().context("enable GDN_WHOLE=1")?;
    let candidate = gpu
        .gdn
        .whole_head_gates
        .as_ref()
        .context("enable GDN_HEAD_GATES=1")?;
    let half = |v: Vec<u16>| gpu.f16_storage(&v);
    let values = |n: usize, salt: usize| {
        (0..n)
            .map(|i| f16::from_f32((((i + salt) % 67) as f32 - 33.) / 97.).to_bits())
            .collect::<Vec<_>>()
    };
    let qkv = half(values(10240, 0))?;
    let z = half(
        (0..6144)
            .map(|i| {
                if i % 17 == 0 {
                    0xc8dc
                } else {
                    f16::from_f32(((i % 67) as f32 - 33.) / 97.).to_bits()
                }
            })
            .collect(),
    )?;
    let history = half(values(30720, 3))?;
    let weight = half(values(40960, 7))?;
    let state = gpu.f32_storage(
        &(0..786432)
            .map(|i| ((i % 71) as f32 - 35.) / 111.)
            .collect::<Vec<_>>(),
    )?;
    let mut gain_values = vec![1.; 128];
    gain_values[0] = f32::from_bits(0x3f801001);
    let gain = gpu.f32_storage(&gain_values)?;
    let a_log = gpu.f32_storage(
        &(0..48)
            .map(|i| [-20., -1., 0., 1.][i % 4])
            .collect::<Vec<_>>(),
    )?;
    let boundary = [
        0x0000, 0x8000, 0x0001, 0x8001, 0x3c00, 0xbc00, 0x7bff, 0xfbff, 0x49ff, 0xc9ff, 0x5000,
        0xd000,
    ];
    for phase in 0..3 {
        let a = half(
            (0..48)
                .map(|h| boundary[(h + phase) % boundary.len()])
                .collect(),
        )?;
        let b = half(
            (0..48)
                .map(|h| boundary[(h * 7 + phase) % boundary.len()])
                .collect(),
        )?;
        let bias = half((0..48).map(|h| boundary[(h * 3 + phase) % 4]).collect())?;
        let run = |pipeline: &ComputePipelineState| -> Result<[Vec<u8>; 3]> {
            let out = [
                MatrixView::output(&gpu, FloatType::F16, 1, 30720, Storage::StorageModeShared)?,
                MatrixView::output(&gpu, FloatType::F32, 1, 786432, Storage::StorageModeShared)?,
                MatrixView::output(&gpu, FloatType::F16, 1, 6144, Storage::StorageModeShared)?,
            ];
            gpu.command(|cmd| {
                Metal::encode(
                    cmd,
                    pipeline,
                    &[
                        &qkv.buffer,
                        &z.buffer,
                        &a.buffer,
                        &b.buffer,
                        &history.buffer,
                        &weight.buffer,
                        &a_log.buffer,
                        &bias.buffer,
                        &state.buffer,
                        &out[0].buffer,
                        &out[1].buffer,
                        &gain.buffer,
                        &out[2].buffer,
                    ],
                    [(f32::EPSILON * 0.5).to_bits()],
                    MTLSize::new(48, 1, 1),
                    512,
                );
                Ok(())
            })?;
            Ok([
                out[0].export_bytes(&gpu)?,
                out[1].export_bytes(&gpu)?,
                out[2].export_bytes(&gpu)?,
            ])
        };
        let expected = run(canonical)?;
        ensure!(expected.iter().map(Vec::len).collect::<Vec<_>>() == [61440, 3145728, 12288]);
        ensure!(
            finite(&expected[0], true) && finite(&expected[1], false) && finite(&expected[2], true)
        );
        ensure!(
            run(candidate)? == expected,
            "head-gate boundary corpus phase{phase}"
        );
    }
    let c = crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let mut gdn = FlashGdn::load(&gpu, &c, "model.language_model.layers.0.linear_attn", 1e-6)?;
    let read = |p: &PendingGdn| -> Result<[Vec<u8>; 4]> {
        Ok([
            p.next.history.export_bytes(&gpu)?,
            p.next.recurrent.export_bytes(&gpu)?,
            p.test_gated.export_bytes(&gpu)?,
            p.values.export_bytes(&gpu)?,
        ])
    };
    let start = gpu.gdn_head_gate_uses.get();
    for phase in 0..2 {
        let x = gpu
            .f32_storage(
                &(0..2560)
                    .map(|i| (((i + phase * 13) % 113) as f32 - 56.) / 79.)
                    .collect::<Vec<_>>(),
            )?
            .matrix(MatrixLayout {
                shape: [1, 2560],
                strides: [2560, 1],
                offset: 0,
            })?;
        let before = gdn.snapshot();
        gpu.gdn_head_gates.set(false);
        let expected = gpu.command(|cmd| gdn.encode_for_consumer(cmd, &x, false, true))?;
        let bytes = read(&expected)?;
        gdn.finish(expected)?;
        let held = gdn.snapshot();
        gdn.restore(&before)?;
        gpu.gdn_head_gates.set(true);
        let actual = gpu.command(|cmd| gdn.encode_for_consumer(cmd, &x, false, true))?;
        ensure!(read(&actual)? == bytes);
        gdn.finish(actual)?;
        let frame = held.frame.as_ref().context("retained frame missing")?;
        ensure!(
            frame.history.export_bytes(&gpu)? == bytes[0]
                && frame.recurrent.export_bytes(&gpu)? == bytes[1]
        );
    }
    ensure!(gpu.gdn_head_gate_uses.get() - start == 2);
    eprintln!(
        "AR20_GDN_HEAD_GATES {{\"parity\":true,\"boundary_cases\":3,\"boundary_state_words\":2359296,\"boundary_history_half_words\":92160,\"boundary_gated_half_words\":18432,\"actual_runtime_uses\":2,\"retained_frames\":true,\"canonical_psos_separate\":true}}"
    );
    Ok(())
}
