use super::*;
use std::{fs::File, path::Path, sync::atomic::Ordering, time::Instant};

#[test]
#[ignore = "physical M5 and Flash EXL3 checkpoint; isolated shared-tile qualification"]
fn mia_prefill_tile_quality_and_speed() -> Result<()> {
    let output = std::env::var("MLXL3_MIA_TILE_REPORT")?;
    serde_json::to_writer(
        File::create_new(&output)?,
        &serde_json::json!({"status":"incomplete","parity":false}),
    )?;
    let speed = std::env::var("MLXL3_MIA_TILE_PHASE").as_deref() == Ok("speed");
    let checkpoint = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    crate::array::set_default_cache_limit(64 << 20)?;
    ensure!(crate::array::is_m5_gpu()?);
    let mut cases = 0;
    let mut records = Vec::new();
    for layer in [0, 23, 47] {
        let prefix = format!("model.language_model.layers.{layer}.mlp");
        let mut moe = SsdMoe::load(&checkpoint, &prefix, 53, true)?;
        moe.fill_many(&(0..5).map(|i| (i, i as usize)).collect::<Vec<_>>())?;
        let reference = (0..5)
            .map(|id| {
                ["gate_proj", "up_proj", "down_proj"]
                    .map(|name| {
                        crate::linear::Exl3Linear::from_checkpoint(
                            &checkpoint,
                            &format!("{prefix}.experts.{id}.{name}"),
                        )
                    })
                    .into_iter()
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        for rows in [1, 2, 3, 4, 5, 7, 8, 16, 32] {
            for mixed in [false, true] {
                let ids = (0..rows)
                    .map(|i| if mixed { (i * 7 % 5) as u32 } else { 0 })
                    .collect::<Vec<_>>();
                for strided in [false, true] {
                    let value = |r, c| ((r * 37 + c * 13) % 997) as f32 / 1024. - 0.5;
                    let x = if strided {
                        Array::from_f32(
                            &(0..H)
                                .flat_map(|c| (0..rows).map(move |r| value(r, c)))
                                .collect::<Vec<_>>(),
                            &[H, rows],
                        )?
                        .transpose(&[1, 0])?
                    } else {
                        Array::from_f32(
                            &(0..rows)
                                .flat_map(|r| (0..H).map(move |c| value(r, c)))
                                .collect::<Vec<_>>(),
                            &[rows, H],
                        )?
                        .astype(Dtype::Float16)?
                    };
                    let mut activated = Vec::new();
                    let expected = (0..rows)
                        .map(|r| {
                            let p = &reference[ids[r as usize] as usize];
                            let input = x.slice(0, r, r + 1)?;
                            let a = p[0].forward(&input)?.swiglu(&p[1].forward(&input)?)?;
                            let y = p[2].forward(&a)?;
                            activated.push(a);
                            Ok(y)
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let expected = Array::concatenate(&expected.iter().collect::<Vec<_>>(), 0)?;
                    let bytes = expected.to_bytes()?;
                    ensure!(
                        expected.shape() == [rows, H]
                            && bytes.len()
                                == rows as usize * H as usize * if strided { 4 } else { 2 }
                    );
                    let make_slots = |variant| -> Result<linear::ExpertSlots> {
                        Ok(match variant {
                            0 => linear::ExpertSlots::new(&ids, 53)?,
                            1 => linear::ExpertSlots::grouped(&ids, 53)?,
                            _ => linear::ExpertSlots::prefill_tile(&ids, 53)?,
                        })
                    };
                    let run = |variant| -> Result<Array> {
                        moe.calculate_slots(&x, &make_slots(variant)?)
                    };
                    let mut launches = 0;
                    for variant in 0..3 {
                        let before = linear::PREFILL_TILE_LAUNCHES.load(Ordering::Relaxed);
                        let actual = run(variant)?;
                        let data = actual.to_f32()?;
                        ensure!(
                            actual.shape() == expected.shape()
                                && data.len() == rows as usize * H as usize
                                && data.iter().all(|v| v.is_finite())
                        );
                        ensure!(
                            actual.to_bytes()? == bytes,
                            "Mia tile differs layer={layer}/rows={rows}/mixed={mixed}/strided={strided}/variant={variant}"
                        );
                        let delta = linear::PREFILL_TILE_LAUNCHES.load(Ordering::Relaxed) - before;
                        if variant == 2 {
                            launches = delta;
                        } else {
                            ensure!(delta == 0);
                        }
                    }
                    if !mixed && rows >= 2 {
                        ensure!(launches >= 2, "tile took only a fallback");
                    }
                    if rows == 1 {
                        ensure!(launches == 0);
                    }
                    let slots = (0..3).map(make_slots).collect::<Result<Vec<_>>>()?;
                    let transform = |input: &Array, scales: Array, input_dims| -> Result<Array> {
                        let n = input.shape()[0];
                        input
                            .astype(Dtype::Float16)?
                            .mul(&scales.reshape(&[n, input_dims])?)?
                            .reshape(&[n, input_dims / 128, 128])?
                            .hadamard_transform(Some(1.0 / 128.0f32.sqrt()))?
                            .reshape(&[n, input_dims])
                    };
                    let gu = transform(
                        &x.reshape(&[rows, 1, H])?
                            .broadcast_to(&[rows, 2, H])?
                            .reshape(&[rows * 2, H])?,
                        slots[0].gather_scales(&moe.banks[2])?,
                        H,
                    )?;
                    let down = transform(
                        &Array::concatenate(&activated.iter().collect::<Vec<_>>(), 0)?,
                        slots[0].gather_scales(&moe.banks[4])?,
                        I,
                    )?;
                    for (component, input, bank, out, repeat) in [
                        ("raw_gu", &gu, &moe.banks[0], I, 2),
                        ("raw_down", &down, &moe.banks[1], H, 1),
                    ] {
                        input.eval()?;
                        let raw = |variant: usize| {
                            linear::expert_tile_raw(input, bank, &slots[variant], out, repeat)
                        };
                        let expected = raw(0)?;
                        let expected_bytes = expected.to_bytes()?;
                        ensure!(
                            expected.shape() == [rows * repeat, out] && !expected_bytes.is_empty()
                        );
                        for variant in 1..3 {
                            let actual = raw(variant)?;
                            let floats = actual.to_f32()?;
                            ensure!(
                                floats.len() == (rows * repeat * out) as usize
                                    && floats.iter().all(|v| v.is_finite())
                            );
                            ensure!(
                                actual.to_bytes()? == expected_bytes,
                                "raw F32 partial sums differ {component}/M{rows}/variant{variant}"
                            );
                        }
                        if speed && !strided && [1, 2, 4, 8, 16, 32].contains(&rows) {
                            let samples = measure(raw, &expected_bytes)?;
                            records.push(serde_json::json!({"component":component,"layer":layer,"rows":rows,"mixed":mixed,"launches_per_contribution":launches,"samples":samples}));
                        }
                    }
                    cases += 1;
                    println!(
                        "MIA_TILE_CASE layer={layer} rows={rows} mixed={mixed} strided={strided} launches={launches} full_values={} parity=true",
                        data_len(&expected)
                    );
                    if speed && !strided && [1, 2, 4, 8, 16, 32].contains(&rows) {
                        let io = moe.read_bytes;
                        let samples = measure(run, &bytes)?;
                        ensure!(moe.read_bytes == io);
                        records.push(serde_json::json!({"component":"contribution","layer":layer,"rows":rows,"mixed":mixed,"launches_per_call":launches,"input_dtype":"Float16","samples":samples}));
                    }
                }
            }
        }
        for (ids, capacity) in [
            (vec![], 53),
            (vec![0; 81], 53),
            (vec![53; 2], 53),
            (vec![u32::MAX; 2], 53),
            (vec![0; 2], 0),
            (vec![0; 2], 513),
        ] {
            ensure!(linear::ExpertSlots::prefill_tile(&ids, capacity).is_err());
        }
        moe.drain()?;
    }
    ensure!(cases == 108 && (!speed || records.len() == 108));
    serde_json::to_writer_pretty(
        File::create(&output)?,
        &serde_json::json!({"status":"complete","parity":"byte_exact_resident_independent","cases":cases,"phase":if speed {"speed"}else{"quality"},"no_timed_io":true,"forwards_per_sample":32,"records":records,"scope":"real EXL3 experts 0..4 in layers0/23/47, repeated or mixed routed rows; full MoE contribution, not full-model throughput"}),
    )?;
    println!(
        "MIA_TILE_QUALIFIED cases={cases} errors=true rows=1,2,4,8,16,32 phase={}",
        if speed { "speed" } else { "quality" }
    );
    Ok(())
}

fn data_len(value: &Array) -> usize {
    value.shape().iter().map(|&n| n as usize).product()
}

fn measure(
    mut run: impl FnMut(usize) -> Result<Array>,
    expected: &[u8],
) -> Result<Vec<serde_json::Value>> {
    ensure!(!expected.is_empty());
    for _ in 0..3 {
        for variant in 0..3 {
            run(variant)?.eval()?;
        }
    }
    let mut samples = Vec::new();
    for pair in 0..12 {
        for variant in if pair % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
            let started = Instant::now();
            for _ in 0..32 {
                run(variant)?.eval()?;
            }
            let seconds = started.elapsed().as_secs_f64();
            ensure!(seconds.is_finite() && seconds > 0. && run(variant)?.to_bytes()? == expected);
            samples.push(serde_json::json!({"pair":pair,"variant":variant,"seconds":seconds}));
        }
    }
    Ok(samples)
}

#[cfg(kani)]
#[kani::proof]
fn actual_mia_shared_tile_entry() {
    mia_prefill_tile_quality_and_speed().unwrap();
}
