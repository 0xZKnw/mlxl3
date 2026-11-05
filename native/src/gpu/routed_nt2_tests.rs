use super::routed_scratch_tests::{Case, check_words, input, record};
use super::*;
use std::{fs, path::PathBuf, time::Instant};

fn select(gpu: &Metal, mode: usize) {
    gpu.routed_nt2_gu.set(mode & 1 != 0);
    gpu.routed_nt2_down.set(mode & 2 != 0);
}

#[test]
#[ignore = "physical frozen routed EXL3 NT2 parity, followed by bounded component wall-time screen"]
fn physical_routed_nt2_preserves_stock_and_screens_stage_shapes() -> Result<()> {
    let root = PathBuf::from(
        std::env::var_os("MLXL3_ROUTED_SCRATCH_FIXTURES").context("fixtures required")?,
    );
    let cases: Vec<Case> = serde_json::from_slice(&fs::read(root.join("cases.json"))?)?;
    let refs: Vec<Vec<u16>> = serde_json::from_slice(&fs::read(root.join("stock.json"))?)?;
    ensure!(cases.len() >= 5 && cases.len() == refs.len());
    let gpu = Metal::new()?;
    ensure!(
        gpu.routed_nt2_pipelines.is_some(),
        "enable both routed NT2 flags to load the two PSOs"
    );
    let mut checked = 0;
    for mode in 0..4 {
        select(&gpu, mode);
        for (case, expected) in cases[..cases.len() - 1].iter().zip(&refs) {
            for fp32 in [false, true] {
                let backing = record(&gpu, case)?;
                let expert = FlashExpert::from_backing(&gpu, &backing)?;
                let x = input(&gpu, case, fp32)?;
                let before = gpu.routed_nt2_uses.each_ref().map(std::cell::Cell::get);
                let values =
                    gpu.command(|cmd| expert.encode_view(cmd, &x, Storage::StorageModeShared))?;
                let delta =
                    std::array::from_fn::<_, 2, _>(|i| gpu.routed_nt2_uses[i].get() - before[i]);
                ensure!(
                    delta
                        == [
                            if mode & 1 != 0 { 2 } else { 0 },
                            if mode & 2 != 0 { 1 } else { 0 }
                        ],
                    "routed NT2 fell back"
                );
                check_words(&Metal::read(&values, 2560), expected)?;
                checked += 2560;
            }
        }
    }
    // The timer begins only after every selected stage matched the independent stock.
    let backing = record(&gpu, &cases[0])?;
    let expert = FlashExpert::from_backing(&gpu, &backing)?;
    let x = input(&gpu, &cases[0], true)?;
    for mode in 0..4 {
        select(&gpu, mode);
        gpu.command(|cmd| {
            for _ in 0..3 {
                std::hint::black_box(expert.encode_view(cmd, &x, Storage::StorageModePrivate)?);
            }
            Ok(())
        })?;
    }
    let mut timings: [Vec<f64>; 4] = std::array::from_fn(|_| Vec::new());
    for repeat in 0..6 {
        let order = if repeat % 2 == 0 {
            [0, 1, 2, 3]
        } else {
            [3, 2, 1, 0]
        };
        for mode in order {
            select(&gpu, mode);
            let start = Instant::now();
            gpu.command(|cmd| {
                for _ in 0..16 {
                    std::hint::black_box(expert.encode_view(
                        cmd,
                        &x,
                        Storage::StorageModePrivate,
                    )?);
                }
                Ok(())
            })?;
            timings[mode].push(start.elapsed().as_secs_f64() / 16.);
        }
    }
    for times in &mut timings {
        times.sort_by(f64::total_cmp);
    }
    let medians = timings.each_ref().map(|v| (v[2] + v[3]) * 0.5);
    eprintln!(
        "AR10_ROUTED_NT2 {}",
        serde_json::json!({"parity":true,"finite_stock_half_words":checked,"modes":["off","GU","down","both"],"samples_per_mode":6,"experts_per_sample":16,"scope":"host encoding+allocation and GPU command completion, not perkernel GPU time or model gain","seconds_per_expert_medians":medians,"samples_seconds_per_expert":timings})
    );
    Ok(())
}
