use super::*;
use std::time::Instant;

#[test]
#[ignore = "physical Apple GPU; prepared eval component oracle then micro, run alone"]
fn prepared_cold_chain_quality_and_micro() -> anyhow::Result<()> {
    let prepared = std::env::var("MLXL3_PREPARED_METAL")?;
    ensure!(prepared == "0" || prepared == "1");
    let mut checked = 0;
    for (input, output, projections) in [(2560usize, 640usize, 2usize), (640, 2560, 1)] {
        let words = input / 16 * (output / 16) * 32;
        let packed = (0..10 * projections * words)
            .map(|i| (i as u16).wrapping_mul(1777))
            .collect::<Vec<_>>();
        let bank = Array::from_u16(
            &packed,
            &[
                10,
                projections as i32,
                (input / 16) as i32,
                (output / 16) as i32,
                32,
            ],
        )?;
        for routes_count in [1usize, 2, 4, 8, 16, 32] {
            let routes = (0..routes_count)
                .map(|r| (r % 10) as u32)
                .collect::<Vec<_>>();
            let slots = ExpertSlots::new(&routes, 10)?;
            let rows = routes_count * projections;
            let value = |r: usize, c: usize| ((r * 37 + c * 13) % 997) as f32 / 1024. - 0.5;
            let values = (0..rows)
                .flat_map(|r| (0..input).map(move |c| value(r, c)))
                .collect::<Vec<_>>();
            let scales = |width: usize| {
                (0..rows * width)
                    .map(|i| half::f16::from_f32(0.5 + (i % 11) as f32 / 32.).to_bits())
                    .collect::<Vec<_>>()
            };
            let su = scales(input);
            let sv = scales(output);
            let suh = Array::from_f16_bits(&su, &[rows as i32, input as i32])?;
            let svh = Array::from_f16_bits(&sv, &[rows as i32, output as i32])?;
            for dtype in [Dtype::Float16, Dtype::Float32] {
                let mut expected = Vec::new();
                for row in 0..rows {
                    let index =
                        routes[row / projections] as usize * projections + row % projections;
                    let mut reference = Exl3Linear::new(
                        Array::from_u16(
                            &packed[index * words..(index + 1) * words],
                            &[(input / 16) as i32, (output / 16) as i32, 32],
                        )?,
                        Array::from_f16_bits(&su[row * input..(row + 1) * input], &[input as i32])?,
                        Array::from_f16_bits(
                            &sv[row * output..(row + 1) * output],
                            &[output as i32],
                        )?,
                        None,
                        2,
                        Codebook::Mul1,
                    )?;
                    reference.simdgroups = 4;
                    reference.reference_accumulation = true;
                    let x = Array::from_f32(
                        &values[row * input..(row + 1) * input],
                        &[1, input as i32],
                    )?
                    .astype(dtype)?;
                    expected.extend(reference.forward(&x)?.to_bytes()?);
                }
                for strided in [false, true] {
                    let x = if strided {
                        let transposed = (0..input)
                            .flat_map(|c| (0..rows).map(move |r| value(r, c)))
                            .collect::<Vec<_>>();
                        Array::from_f32(&transposed, &[input as i32, rows as i32])?
                            .astype(dtype)?
                            .transpose(&[1, 0])?
                    } else {
                        Array::from_f32(&values, &[rows as i32, input as i32])?.astype(dtype)?
                    };
                    let make = || {
                        expert_slots(
                            &x,
                            &bank,
                            &suh,
                            &svh,
                            &slots,
                            output as i32,
                            projections as i32,
                            2,
                            Codebook::Mul1,
                            false,
                            false,
                        )
                    };
                    let y = make()?;
                    ensure!(y.shape() == [rows as i32, output as i32]);
                    let floats = y.to_f32()?;
                    ensure!(floats.len() == rows * output && floats.iter().all(|v| v.is_finite()));
                    ensure!(
                        y.to_bytes()? == expected,
                        "prepared Cold differs from independent single-row projection"
                    );
                    checked += 1;
                    if strided || dtype != Dtype::Float32 {
                        continue;
                    }
                    for _ in 0..8 {
                        make()?.eval()?;
                    }
                    for batch in 0..8 {
                        let mut times = [0.0f64; 3];
                        for _ in 0..32 {
                            let t = Instant::now();
                            let y = make()?;
                            times[0] += t.elapsed().as_secs_f64();
                            let t = Instant::now();
                            y.async_eval()?;
                            times[1] += t.elapsed().as_secs_f64();
                            let t = Instant::now();
                            y.eval()?;
                            times[2] += t.elapsed().as_secs_f64();
                            ensure!(y.to_bytes()? == expected, "micro output diverged");
                        }
                        eprintln!(
                            "PREPARED_MICRO flag={prepared} input={input} output={output} routes={routes_count} rows={rows} batch={batch} calls=32 build_us={} submit_us={} wait_us={}",
                            times[0] * 1e6 / 32.,
                            times[1] * 1e6 / 32.,
                            times[2] * 1e6 / 32.
                        );
                    }
                }
            }
        }
    }
    ensure!(checked == 48);
    eprintln!("PREPARED_CHAIN_QUALITY cases={checked} exact=true");
    Ok(())
}
