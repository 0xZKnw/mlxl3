use super::*;

#[test]
#[ignore = "physical M5; CPU-known EXL3 groups, isolated component, no tokens/sec claim"]
fn grouped_experts_match_captured_routes_before_timing() -> Result<()> {
    use std::{fs::File, path::Path, time::Instant};
    let output = std::env::var("MLXL3_FLASH_GROUP_MICRO")?;
    serde_json::to_writer(
        File::create_new(&output)?,
        &serde_json::json!({"status":"incomplete","parity":false}),
    )?;
    let root = std::env::var("MLXL3_FLASH_REFERENCE")?;
    let root = Path::new(&root);
    let reference: serde_json::Value =
        serde_json::from_reader(File::open(root.join("reference.json"))?)?;
    ensure!(reference["status"] == "complete");
    let checkpoint = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    crate::array::set_default_cache_limit(512 << 20)?;
    ensure!(crate::array::is_m5_gpu()?);
    let mut records = Vec::new();
    for layer in [0usize, 23, 47] {
        let mut inputs = Vec::new();
        let mut oracle = Vec::new();
        for step in [0, 1] {
            let entry = &reference["steps"][step]["components"][layer]["mlp"];
            let read = |key: &str| -> Result<Vec<u8>> {
                ensure!(entry[key]["count"] == H);
                let bytes = std::fs::read(
                    root.join(
                        entry[key]["file"]
                            .as_str()
                            .context("missing group oracle")?,
                    ),
                )?;
                ensure!(
                    bytes.len() == H as usize * 4
                        && bytes
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .all(|&b| f32::from_le_bytes(b).is_finite())
                );
                Ok(bytes)
            };
            inputs.push(Array::from_bytes(&read("input")?, &[1, H], Dtype::Float32)?);
            oracle.push(read("output")?);
        }
        let mut moe = SsdMoe::load(
            &checkpoint,
            &format!("model.language_model.layers.{layer}.mlp"),
            53,
            true,
        )?;
        moe.shared_exl3 = false;
        moe.gpu_hot = true;
        moe.gpu_hot_wide = true;
        for rows in [1usize, 2, 4, 8] {
            let x = Array::concatenate(&(0..rows).map(|i| &inputs[i % 2]).collect::<Vec<_>>(), 0)?;
            ensure!(
                moe.forward_draft_rows(&x)?.to_bytes()?
                    == (0..rows)
                        .flat_map(|i| oracle[i % 2].iter().copied())
                        .collect::<Vec<_>>()
            );
            let mut ids = Vec::new();
            for i in 0..rows {
                ids.extend(moe.route(&inputs[i % 2])?.0.to_u32()?);
            }
            ensure!(ids.len() == rows * 10);
            let ready = moe.cache.borrow().resident_slot_map()?;
            let slots = ids.iter().map(|&id| ready[id as usize]).collect::<Vec<_>>();
            ensure!(slots.iter().all(|&slot| slot < 53));
            let routed = x
                .take(
                    &Array::from_u32(
                        &(0..rows as u32)
                            .flat_map(|i| std::iter::repeat_n(i, 10))
                            .collect::<Vec<_>>(),
                        &[ids.len() as i32],
                    )?,
                    0,
                )?
                .astype(Dtype::Float16)?;
            let ordinary = moe.calculate(&routed, &slots)?.to_bytes()?;
            ensure!(ordinary.len() == ids.len() * H as usize * 2);
            let io = moe.read_bytes;
            for mixed in [false, true] {
                let unique = ids.iter().copied().collect::<BTreeSet<_>>();
                let cold = unique
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &id)| (!mixed || i % 2 == 0).then_some(id))
                    .collect::<BTreeSet<_>>();
                let positions = ids
                    .iter()
                    .enumerate()
                    .filter_map(|(i, id)| cold.contains(id).then_some(i as u32))
                    .collect::<Vec<_>>();
                let known_slots = positions
                    .iter()
                    .map(|&i| slots[i as usize])
                    .collect::<Vec<_>>();
                let shared_enabled =
                    linear::ExpertSlots::grouped(&known_slots, 53)?.shares_weights();
                let input =
                    routed.take(&Array::from_u32(&positions, &[positions.len() as i32])?, 0)?;
                let expected = positions
                    .iter()
                    .flat_map(|&row| {
                        ordinary[row as usize * H as usize * 2..(row as usize + 1) * H as usize * 2]
                            .iter()
                            .copied()
                    })
                    .collect::<Vec<_>>();
                let run = |candidate| -> Result<Array> {
                    let selected = if candidate {
                        linear::ExpertSlots::grouped(&known_slots, 53)?
                    } else {
                        linear::ExpertSlots::new(&known_slots, 53)?
                    };
                    moe.calculate_slots(&input, &selected)
                };
                let warm = Instant::now();
                while warm.elapsed().as_secs_f64() < 1. {
                    for candidate in [false, true] {
                        let value = run(candidate)?;
                        ensure!(
                            value.shape() == [positions.len() as i32, H]
                                && value.to_bytes()? == expected
                        );
                        ensure!(value.to_f32()?.iter().all(|v| v.is_finite()));
                    }
                }
                let mut samples = Vec::new();
                for pair in 0..4 {
                    for candidate in if pair % 2 == 0 {
                        [false, true]
                    } else {
                        [true, false]
                    } {
                        let started = Instant::now();
                        for _ in 0..1024 {
                            run(candidate)?.eval()?;
                        }
                        let seconds = started.elapsed().as_secs_f64();
                        ensure!(seconds.is_finite() && seconds > 0.);
                        ensure!(run(candidate)?.to_bytes()? == expected && moe.read_bytes == io);
                        samples.push(serde_json::json!({"pair":pair,"candidate":candidate,"seconds":seconds}));
                    }
                }
                records.push(serde_json::json!({"layer":layer,"rows":rows,"mixed":mixed,"route_ids":ids,"known_slots":known_slots,"shared_enabled":shared_enabled,"io_bytes_after_warm":io,"samples":samples}));
            }
        }
        moe.drain()?;
    }
    ensure!(records.len() == 24);
    serde_json::to_writer_pretty(
        File::create(&output)?,
        &serde_json::json!({"status":"complete","parity":"byte_exact_independent","no_timed_io":true,"forwards_per_sample":1024,"samples_per_cell":8,"records":records,"scope":"CPU-known contributions, B4/B8 repeat two captured real positions; not full-model throughput or Hot GPU grouping"}),
    )?;
    Ok(())
}
