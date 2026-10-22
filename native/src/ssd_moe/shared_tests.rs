use super::*;

fn fixtures() -> Result<Vec<(SsdMoe, Vec<f32>)>> {
    let root = std::path::PathBuf::from(std::env::var("MLXL3_FLASH_REFERENCE")?);
    let reference: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(root.join("reference.json"))?)?;
    ensure!(reference["status"] == "complete");
    let checkpoint = crate::checkpoint::inspect(&std::path::PathBuf::from(std::env::var(
        "MLXL3_FLASH_MODEL",
    )?))?;
    crate::array::set_default_cache_limit(512 << 20)?;
    [0, 23, 47]
        .into_iter()
        .map(|layer| {
            let entry = &reference["steps"][0]["components"][layer]["mlp"]["input"];
            let bytes = std::fs::read(root.join(entry["file"].as_str().context("missing input")?))?;
            ensure!(entry["count"] == H && bytes.len() == H as usize * 4);
            let values = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&b| f32::from_le_bytes(b))
                .collect::<Vec<_>>();
            ensure!(values.iter().all(|v| v.is_finite()));
            let moe = SsdMoe::load(
                &checkpoint,
                &format!("model.language_model.layers.{layer}.mlp"),
                10,
                true,
            )?;
            ensure!(
                moe.shared_gate.is_exl3() && moe.shared_up.is_exl3() && moe.shared_down.is_exl3()
            );
            Ok((moe, values))
        })
        .collect()
}

fn input(base: &[f32], rows: i32, dtype: Dtype, strided: bool) -> Result<Array> {
    let value = |r: i32, c: i32| {
        if r == 1 {
            0.0
        } else {
            let sign = if r % 2 == 0 { 1.0 } else { -1.0 };
            sign * base[c as usize] * (1.0 + r as f32 / 32.0)
                + ((r * 17 + c * 3) % 19 - 9) as f32 / 128.0
        }
    };
    let values = if strided {
        (0..H)
            .flat_map(|c| (0..rows).map(move |r| value(r, c)))
            .collect::<Vec<_>>()
    } else {
        (0..rows)
            .flat_map(|r| (0..H).map(move |c| value(r, c)))
            .collect::<Vec<_>>()
    };
    if strided {
        Array::from_f32(&values, &[H, rows])?
            .astype(dtype)?
            .transpose(&[1, 0])
    } else {
        Array::from_f32(&values, &[rows, H])?.astype(dtype)
    }
}

fn exact(actual: &Array, expected: &Array, shape: &[i32]) -> Result<()> {
    ensure!(actual.shape() == shape && expected.shape() == shape);
    let values = actual.to_f32()?;
    ensure!(
        values.len() == shape.iter().product::<i32>() as usize
            && !values.is_empty()
            && values.iter().all(|v| v.is_finite())
    );
    ensure!(
        actual.to_bytes()? == expected.to_bytes()?,
        "shared expert differs"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; exact shared-expert batches on real Flash weights"]
fn shared_batches_match_single_row_projections_and_scalar_gates() -> Result<()> {
    let mut cases = 0;
    for (mut moe, base) in fixtures()? {
        for rows in 1..=8 {
            for dtype in [Dtype::Float16, Dtype::Float32] {
                for strided in [false, true] {
                    let x = input(&base, rows, dtype, strided)?;
                    let mut gates = Vec::new();
                    let mut ups = Vec::new();
                    let mut downs = Vec::new();
                    let mut shared = Vec::new();
                    for row in 0..rows {
                        let row = x.slice(0, row, row + 1)?;
                        let gate = moe.shared_gate.forward(&row)?.astype(Dtype::Float16)?;
                        let up = moe.shared_up.forward(&row)?.astype(Dtype::Float16)?;
                        let down = moe.shared_down.forward(&gate.swiglu(&up)?)?.astype(dtype)?;
                        shared.push(down.mul(&moe.multiplier.forward(&row)?.sigmoid()?)?);
                        gates.push(gate);
                        ups.push(up);
                        downs.push(down);
                    }
                    let join = |values: &[Array]| {
                        Array::concatenate(&values.iter().collect::<Vec<_>>(), 0)
                    };
                    exact(
                        &moe.shared_gate.forward(&x)?.astype(Dtype::Float16)?,
                        &join(&gates)?,
                        &[rows, I],
                    )?;
                    exact(
                        &moe.shared_up.forward(&x)?.astype(Dtype::Float16)?,
                        &join(&ups)?,
                        &[rows, I],
                    )?;
                    exact(&moe.shared_projection(&x)?, &join(&downs)?, &[rows, H])?;
                    let expected = join(&shared)?;
                    moe.shared_batch = false;
                    exact(&moe.shared(&x)?, &expected, &[rows, H])?;
                    moe.shared_batch = true;
                    let before = Array::host_readbacks_for_test();
                    let actual = moe.shared(&x)?;
                    ensure!(Array::host_readbacks_for_test() == before);
                    exact(&actual, &expected, &[rows, H])?;
                    if cases == 8 {
                        std::mem::swap(&mut moe.shared_gate, &mut moe.shared_up);
                        let wrong = moe.shared(&x)?.to_bytes()?;
                        std::mem::swap(&mut moe.shared_gate, &mut moe.shared_up);
                        ensure!(
                            wrong != expected.to_bytes()?,
                            "bad projection escaped oracle"
                        );
                        exact(&moe.shared(&x)?, &expected, &[rows, H])?;
                    }
                    cases += 1;
                }
            }
        }
        for shape in [vec![0, H], vec![9, H], vec![1, H - 1], vec![H]] {
            ensure!(
                moe.shared(&Array::zeros_dtype(&shape, Dtype::Float32)?)
                    .is_err()
            );
        }
        ensure!(
            moe.shared(&Array::zeros_dtype(&[1, H], Dtype::UInt32)?)
                .is_err()
        );
        moe.drain()?;
    }
    ensure!(cases == 96);
    println!(
        "FLASH_SHARED_BATCH cases=96 intermediate_checks=288 scalar_gates=byte_exact invalid=15 mutation=detected_restored parity=byte_exact"
    );
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn actual_shared_batches_preserve_serial_outputs() {
    shared_batches_match_single_row_projections_and_scalar_gates().unwrap();
}

#[test]
#[ignore = "physical Apple GPU; shared component wall time, not exclusive GPU time or tok/s"]
fn shared_batch_component_microbench() -> Result<()> {
    let mut fixtures = fixtures()?;
    let mut samples = Vec::new();
    let mut profiles = Vec::new();
    for rows in [2, 4] {
        let inputs = fixtures
            .iter()
            .map(|(_, base)| input(base, rows, Dtype::Float32, false))
            .collect::<Result<Vec<_>>>()?;
        for candidate in [false, true] {
            for (layer, ((moe, _), x)) in [0, 23, 47]
                .into_iter()
                .zip(fixtures.iter_mut().zip(&inputs))
            {
                moe.shared_batch = false;
                let expected = moe.shared(x)?;
                moe.shared_batch = candidate;
                exact(&moe.shared(x)?, &expected, &[rows, H])?;
                for _ in 0..8 {
                    moe.shared(x)?.eval()?;
                }
                crate::flash_profile::start(true)?;
                {
                    let _total = crate::flash_profile::span("shared_expert_total");
                    moe.shared(x)?.eval()?;
                }
                profiles.push(
                    serde_json::json!({"layer":layer,"rows":rows,"candidate":candidate,
                    "events":crate::flash_profile::finish()?}),
                );
            }
        }
        for pair in 0..5 {
            for candidate in if pair % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                for (moe, _) in &mut fixtures {
                    moe.shared_batch = candidate;
                }
                let started = std::time::Instant::now();
                for _ in 0..32 {
                    for ((moe, _), x) in fixtures.iter().zip(&inputs) {
                        moe.shared(x)?.eval()?;
                    }
                }
                samples.push(
                    serde_json::json!({"rows":rows,"pair":pair,"candidate":candidate,
                    "seconds":started.elapsed().as_secs_f64(),"calls":96}),
                );
            }
        }
    }
    let report = serde_json::json!({"status":"complete","parity":"byte_exact","layers":[0,23,47],
        "meaning":"component wall time includes host construction, dispatch and evaluation; not exclusive GPU time or generation throughput",
        "warmups_per_layer_variant":8,"samples":samples,
        "profiles":profiles,"profile_meaning":"separate fenced completion wall intervals; inclusive parents overlap children, never add to native timings"});
    std::fs::write(
        std::env::var("MLXL3_SHARED_MICRO_OUTPUT")?,
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("FLASH_SHARED_MICRO rows=B2,B4 samples=20 calls_per_sample=96 parity=byte_exact");
    Ok(())
}
