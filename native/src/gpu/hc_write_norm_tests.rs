use super::*;
use crate::{checkpoint, tensor_layout::MatrixLayout};
use half::f16;

fn row(gpu: &Metal, values: &[f32], fp32: bool) -> Result<MatrixView> {
    let mut reversed = vec![9.0; values.len() + 3];
    for (i, &x) in values.iter().enumerate() {
        reversed[values.len() + 1 - i] = x;
    }
    let storage = if fp32 {
        gpu.f32_storage(&reversed)?
    } else {
        gpu.f16_storage(
            &reversed
                .iter()
                .map(|&x| f16::from_f32(x).to_bits())
                .collect::<Vec<_>>(),
        )?
    };
    storage.matrix(MatrixLayout {
        shape: [1, values.len() as u32],
        strides: [0, -1],
        offset: values.len() as i64 + 1,
    })
}
#[test]
#[ignore = "physical HC write/norm full outputs and exact float/half-stride fallbacks"]
fn physical_hc_write_norm_preserves_written_normalized_and_mixed() -> Result<()> {
    let c = checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let gpu = Metal::new()?;
    ensure!(gpu.hc_write_norm.get() && !gpu.hc_workspace_enabled.get());
    let hc = FlashHc::load(
        &gpu,
        &c,
        "model.language_model.layers.0.mlp_hyper_connection",
        1e-6,
        true,
    )?;
    let values = (0..10240)
        .map(|i| ((i % 211) as f32 - 105.) / 111.)
        .collect::<Vec<_>>();
    let delta = (0..2560)
        .map(|i| ((i % 97) as f32 - 48.) / 73.)
        .collect::<Vec<_>>();
    let mut compared = 0;
    for stream_fp32 in [false, true] {
        for branch_fp32 in [false, true] {
            let streams = row(&gpu, &values, stream_fp32)?;
            let branch = row(&gpu, &delta, branch_fp32)?;
            let gate = row(&gpu, &[0.25, -0.5, 0.75, -1.], stream_fp32)?;
            gpu.hc_write_norm.set(false);
            let (written, read) =
                gpu.command(|cmd| hc.encode_after_write(cmd, &streams, &branch, &gate))?;
            let expected = [
                written.export_bytes(&gpu)?,
                read.normalized.export_bytes(&gpu)?,
                read.mixed.export_bytes(&gpu)?,
            ];
            gpu.hc_write_norm.set(true);
            let before = gpu.hc_write_norm_uses.get();
            let (written, read) =
                gpu.command(|cmd| hc.encode_after_write(cmd, &streams, &branch, &gate))?;
            ensure!(gpu.hc_write_norm_uses.get() - before == usize::from(stream_fp32));
            let actual = [
                written.export_bytes(&gpu)?,
                read.normalized.export_bytes(&gpu)?,
                read.mixed.export_bytes(&gpu)?,
            ];
            for (a, b) in actual.iter().zip(&expected) {
                ensure!(a == b && !a.is_empty());
                if written.dtype == FloatType::F32 {
                    ensure!(
                        a.as_chunks::<4>()
                            .0
                            .iter()
                            .all(|w| f32::from_le_bytes(*w).is_finite())
                    );
                } else {
                    ensure!(
                        a.as_chunks::<2>()
                            .0
                            .iter()
                            .all(|w| f16::from_le_bytes(*w).is_finite())
                    );
                }
                compared += a.len();
            }
        }
    }
    eprintln!(
        "AR10_HC_WRITE_NORM {}",
        serde_json::json!({"parity":true,"cases":4,"compared_bytes":compared,"half_and_float_branch":true,"negative_strides":true,"half_stream_fallback":true,"actual_fused_cases":2})
    );
    Ok(())
}
