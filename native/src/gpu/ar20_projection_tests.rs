use super::routed_scratch_tests::{Case, check_words, input, record};
use super::*;
use std::{fs, path::PathBuf};

#[test]
#[ignore = "physical AR20 compiled kernels, independent frozen routed records; run alone"]
fn physical_ar20_qmv_kernels_preserve_frozen_routed_stock() -> Result<()> {
    frozen_routed_stock(false)
}

#[test]
#[ignore = "physical AR30 cooperative QMV, independent frozen routed records; run alone"]
fn physical_ar30_cooperative_preserves_frozen_routed_stock() -> Result<()> {
    frozen_routed_stock(true)
}

fn frozen_routed_stock(cooperative: bool) -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("MLXL3_ROUTED_SCRATCH_FIXTURES").context("frozen fixtures required")?,
    );
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(root.join("cases.json"))?)?;
    let refs: Vec<Vec<u16>> = serde_json::from_slice(&fs::read(root.join("stock.json"))?)?;
    ensure!(cases.len() >= 5 && cases.len() == refs.len());
    let gpu = Metal::new()?;
    if cooperative {
        ensure!(super::super::ar30_kernel_tests::compiled_cooperative(&gpu)?);
    } else {
        ensure!(gpu.ar20_kernel_bits != 0);
    }
    let mut words = 0;
    for (case, expected) in cases[..cases.len() - 1].iter().zip(&refs) {
        let backing = record(&gpu, case)?;
        let expert = FlashExpert::from_backing(&gpu, &backing)?;
        for fp32 in [false, true] {
            let x = input(&gpu, case, fp32)?;
            let output =
                gpu.command(|cmd| expert.encode_view(cmd, &x, Storage::StorageModeShared))?;
            check_words(&Metal::read(&output, 2560), expected)?;
            words += 2560;
        }
    }
    ensure!(words > 0);
    if cooperative {
        let dispatches = gpu.ar30_cooperative_uses[1].get();
        ensure!(dispatches > 0, "cooperative routed path was not executed");
        eprintln!(
            "AR30_COOPERATIVE_ROUTED {}",
            serde_json::json!({
                "parity":true,"compiled_cooperative":true,
                "finite_half_words":words,"routed_dispatches":dispatches
            })
        );
        return Ok(());
    }
    eprintln!(
        "AR20_QMV_FROZEN {{\"parity\":true,\"finite_half_words\":{words},\"compiled_bits\":{}}}",
        gpu.ar20_kernel_bits
    );
    Ok(())
}
