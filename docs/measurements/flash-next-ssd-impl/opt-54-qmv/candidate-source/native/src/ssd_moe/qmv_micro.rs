#[test]
#[ignore = "physical M5; experiment54 exact isolated hot MoE, not tokens/sec"]
fn hot_moe_preserves_outputs() -> anyhow::Result<()> {
    use super::*;
    use serde_json::Value;
    use std::{fs::File, path::Path, time::Instant};
    ensure!(crate::array::is_m5_gpu()?);
    let experiment = ExpertQmv::parse(&std::env::var("MLXL3_FLASH_QMV_EXPERIMENT")?)
        .context("invalid experiment")?;
    let output = std::env::var("MLXL3_FLASH_QMV_MICRO")?;
    serde_json::to_writer(
        File::create_new(&output)?,
        &serde_json::json!({"status":"incomplete"}),
    )?;
    let root = std::env::var("MLXL3_FLASH_REFERENCE")?;
    let root = Path::new(&root);
    let reference: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
    ensure!(reference["status"] == "complete" && reference["mode"] == "eval");
    let checkpoint = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    crate::array::set_default_cache_limit(512 << 20)?;
    let mut records = Vec::new();
    for layer in [0usize, 23, 47] {
        let mut inputs = Vec::new();
        let mut expected = Vec::new();
        for step in [0, 1] {
            let entry = reference["steps"][step]["components"]
                .as_array()
                .context("missing components")?
                .get(layer)
                .context("missing micro layer")?;
            let read = |kind: &str| -> Result<Vec<u8>> {
                let entry = &entry["mlp"][kind];
                ensure!(entry["count"] == H);
                let bytes =
                    std::fs::read(root.join(entry["file"].as_str().context("missing file")?))?;
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
            expected.push(read("output")?);
        }
        let mut moe = SsdMoe::load(
            &checkpoint,
            &format!("model.language_model.layers.{layer}.mlp"),
            20,
            true,
        )?;
        for rows in [1, 2] {
            let x = if rows == 1 {
                inputs[0].try_clone()?
            } else {
                Array::concatenate(&[&inputs[0], &inputs[1]], 0)?
            };
            let oracle = expected[..rows].concat();
            // Populate exactly this cohort, then include all routing/map/graph costs in timings.
            moe.qmv_variant = ExpertQmv::Reference;
            ensure!(
                moe.forward_many(&x)?.to_bytes()? == oracle,
                "independent micro oracle differs"
            );
            let io = moe.read_bytes;
            let mut ids = Vec::new();
            let warmup = Instant::now();
            let mut sweeps = 0;
            while sweeps < 2 || warmup.elapsed().as_secs_f64() < 2. {
                for candidate in [false, true] {
                    moe.qmv_variant = if candidate {
                        experiment
                    } else {
                        ExpertQmv::Reference
                    };
                    let value = moe.forward_many(&x)?;
                    ensure!(value.shape() == [rows as i32, H]);
                    ensure!(value.to_bytes()? == oracle, "warm shared MoE differs");
                    if ids.is_empty() {
                        ids.clone_from(&moe.route_ids);
                    }
                    ensure!(ids == moe.route_ids && moe.read_bytes == io);
                }
                sweeps += 1;
            }
            let distinct = ids.iter().copied().collect::<BTreeSet<_>>().len();
            ensure!(
                rows == 1 || distinct < ids.len(),
                "micro did not exercise shared experts"
            );
            let mut samples = Vec::new();
            for pair in 0..4 {
                for candidate in if pair % 2 == 0 {
                    [false, true]
                } else {
                    [true, false]
                } {
                    moe.qmv_variant = if candidate {
                        experiment
                    } else {
                        ExpertQmv::Reference
                    };
                    let start = Instant::now();
                    let mut last = None;
                    for _ in 0..256 {
                        let value = moe.forward_many(&x)?;
                        value.eval()?;
                        last = Some(value);
                    }
                    let seconds = start.elapsed().as_secs_f64() / 256.;
                    ensure!(seconds.is_finite() && seconds > 0.);
                    ensure!(last.context("no micro output")?.to_bytes()? == oracle);
                    ensure!(
                        moe.route_ids == ids && moe.read_bytes == io,
                        "hot micro performed I/O"
                    );
                    samples.push(
                        serde_json::json!({"pair":pair, "candidate":candidate, "seconds":seconds}),
                    );
                }
            }
            records.push(serde_json::json!({"layer":layer, "rows":rows, "route_ids":ids, "io_bytes_after_warm":io, "samples":samples}));
        }
        moe.drain()?;
    }
    std::fs::write(
        output,
        serde_json::to_vec_pretty(&serde_json::json!({
            "status":"complete", "experiment":format!("{experiment:?}"), "samples_per_cell":8, "forwards_per_sample":256,
            "parity":"byte_exact_independent", "no_timed_io":true, "records":records
        }))?,
    )?;
    println!(
        "FLASH_QMV_MICRO layers=3 row_shapes=2 samples=48 forwards_per_sample=256 parity=byte_exact no_timed_io=true"
    );
    Ok(())
}
