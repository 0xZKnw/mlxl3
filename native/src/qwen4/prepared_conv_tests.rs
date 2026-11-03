//! Physical checks of the production prepared-convolution path, with stock forced OFF.
use super::tests::{input, tensor_state_bytes};
use super::*;
use std::{io::Write, path::Path, time::Instant};

const PREFIX: &str = "model.language_model.layers.2.linear_attn";
type StateBytes = (Vec<u8>, Vec<u8>);
type Snapshot = (Option<Array>, Option<Array>);

fn fixture() -> Result<(Checkpoint, Vec<Array>, Vec<Array>)> {
    ensure!(std::env::var("MLXL3_FLASH_PREPARED_CONV").as_deref() == Ok("1"));
    ensure!(std::env::var("MLXL3_FLASH_DENSE_EPILOGUE").as_deref() == Ok("0"));
    ensure!(std::env::var("MLXL3_FLASH_GDN_EPILOGUE").as_deref() == Ok("0"));
    let root = PathBuf::from(std::env::var("MLXL3_FLASH_REFERENCE")?);
    let data: Value = serde_json::from_reader(File::open(root.join("reference.json"))?)?;
    ensure!(data["status"] == "complete" && data["mode"] == "eval");
    let steps = data["steps"]
        .as_array()
        .context("missing GDN fixture steps")?;
    ensure!(steps.len() >= 5, "insufficient independent GDN coverage");
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for step in &steps[..5] {
        let case = &step["components"][2]["linear_attn"];
        inputs.push(input(&root, &case["input"], H)?);
        outputs.push(input(&root, &case["output"], H)?);
    }
    for value in inputs.iter().chain(&outputs) {
        ensure!(value.shape() == [1, H] && value.dtype() == Dtype::Float32);
        tensor_state_bytes(value)?;
        ensure!(
            value.to_f32()?.iter().any(|&v| v != 0.0),
            "vacuous GDN fixture"
        );
    }
    let checkpoint = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    ensure!(checkpoint.model_type == "qwen4_exp");
    Ok((checkpoint, inputs, outputs))
}

fn pair(checkpoint: &Checkpoint) -> Result<(Gdn, Gdn)> {
    let mut off = Gdn::load(checkpoint, PREFIX, 1e-6)?;
    let mut on = Gdn::load(checkpoint, PREFIX, 1e-6)?;
    ensure!(
        off.prepared_conv.is_some() && on.prepared_conv.is_some(),
        "loader lost ON flag"
    );
    off.prepared_conv = None;
    off.gdn_epilogue = false;
    on.gdn_epilogue = false;
    Ok((off, on))
}

fn state(gdn: &Gdn) -> Result<StateBytes> {
    let history = gdn.history.as_ref().context("missing GDN history")?;
    let value = gdn.state.as_ref().context("missing GDN state")?;
    ensure!(history.shape() == [1, 3, 10240] && history.dtype() == Dtype::Float16);
    ensure!(value.shape() == [1, 48, 128, 128] && value.dtype() == Dtype::Float32);
    Ok((tensor_state_bytes(history)?, tensor_state_bytes(value)?))
}

fn outputs(values: &[Array]) -> Result<Vec<Vec<u8>>> {
    ensure!(!values.is_empty());
    values
        .iter()
        .map(|value| {
            ensure!(value.shape() == [1, H] && value.dtype() == Dtype::Float32);
            tensor_state_bytes(value)
        })
        .collect()
}

fn forward(gdn: &mut Gdn, inputs: &[Array]) -> Result<Vec<Array>> {
    if inputs.len() == 1 {
        Ok(vec![gdn.forward(&inputs[0])?])
    } else {
        Ok(gdn.forward_rows(inputs)?.0)
    }
}

fn restore(gdn: &mut Gdn, saved: &Snapshot) -> Result<()> {
    gdn.history = clone_array(&saved.0)?;
    gdn.state = clone_array(&saved.1)?;
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU, Flash fixtures and PREPARED_CONV=1; run alone"]
fn prepared_conv_matches_stock_coefficients_trajectory_and_prefixes() -> Result<()> {
    let (checkpoint, inputs, references) = fixture()?;
    let mut coefficients = 0;
    for name in checkpoint
        .tensors
        .keys()
        .filter(|name| name.ends_with(".linear_attn.conv1d.weight"))
    {
        let conv = half_weight(&checkpoint, name, Some(&[10240, 1, 4]))?.transpose(&[0, 2, 1])?;
        let bits = conv.to_f16_bits()?;
        ensure!(bits.len() == 40960);
        let prepared = Gdn::prepare_conv(&conv)?;
        ensure!(prepared.shape() == [1, 4, 10240] && prepared.dtype() == Dtype::Float32);
        let values = prepared.to_f32()?;
        ensure!(values.len() == 40960 && values.iter().all(|v| v.is_finite()));
        for time in 0..4 {
            for feature in 0..10240 {
                ensure!(
                    values[time * 10240 + feature].to_bits()
                        == f16::from_bits(bits[feature * 4 + time]).to_f32().to_bits(),
                    "CPU coefficient transpose/conversion differs: {name}"
                );
            }
        }
        coefficients += 1;
    }
    ensure!(coefficients == 36, "wrong GDN weight count");
    {
        let _retain = crate::flash_profile::StageRetention::new(true);
        tests::flash_attention_stages_match_upstream()?;
    }
    let _retain = crate::flash_profile::StageRetention::new(false);
    let (mut off, mut on) = pair(&checkpoint)?;
    for (x, expected) in inputs.iter().zip(&references) {
        let a = off.forward(x)?;
        let b = on.forward(x)?;
        ensure!(b.to_bytes()? == expected.to_bytes()?);
        ensure!(outputs(&[a])? == outputs(&[b])?);
        ensure!(state(&off)? == state(&on)?);
    }
    for rows in 1..=8 {
        off.reset();
        on.reset();
        for x in &inputs[..2] {
            let a = off.forward(x)?;
            let b = on.forward(x)?;
            ensure!(outputs(&[a])? == outputs(&[b])? && state(&off)? == state(&on)?);
        }
        let cohort = (0..rows)
            .map(|row| inputs[(row + 2) % inputs.len()].try_clone())
            .collect::<Result<Vec<_>>>()?;
        let mut serial = Vec::new();
        let mut states = Vec::new();
        for x in &cohort {
            serial.push(off.forward(x)?);
            states.push(state(&off)?);
        }
        let (actual, prefixes) = if rows == 1 {
            (vec![on.forward(&cohort[0])?], Vec::new())
        } else {
            on.forward_rows(&cohort)?
        };
        ensure!(actual.len() == rows && prefixes.len() == rows - 1);
        ensure!(
            outputs(&actual)? == outputs(&serial)?
                && state(&on)? == *states.last().context("empty serial states")?
        );
        for (row, prefix) in prefixes.into_iter().enumerate() {
            let AttentionState::GdnTape {
                history,
                tape,
                retained,
            } = prefix
            else {
                anyhow::bail!("missing GDN prefix tape")
            };
            ensure!(retained == row as i32 + 1);
            ensure!(
                tensor_state_bytes(history.as_ref().context("missing prefix history")?)?
                    == states[row].0
                    && tensor_state_bytes(&tape.replay(retained)?)? == states[row].1
            );
        }
        let before = state(&on)?;
        ensure!(on.forward_rows(&[]).is_err() && on.forward_rows(&cohort[..1]).is_err());
        ensure!(state(&on)? == before, "invalid cohort mutated state");
        let a = off.forward(&inputs[0])?;
        let b = on.forward(&inputs[0])?;
        ensure!(outputs(&[a])? == outputs(&[b])? && state(&off)? == state(&on)?);
        on.reset();
        ensure!(on.prepared_conv.is_some() && on.history.is_none() && on.state.is_none());
        ensure!(on.forward(&inputs[0])?.to_bytes()? == references[0].to_bytes()?);
    }
    on.reset();
    let original = on
        .prepared_conv
        .take()
        .context("missing prepared coefficient")?;
    let mut shifted = original.to_f32()?;
    shifted.rotate_left(10240);
    ensure!(
        shifted.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            != original
                .to_f32()?
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
    );
    on.prepared_conv = Some(Array::from_f32(&shifted, &[1, 4, 10240])?);
    let negative = (|| -> Result<bool> {
        Ok(on.forward(&inputs[0])?.to_bytes()? != references[0].to_bytes()?)
    })();
    on.prepared_conv = Some(original);
    on.reset();
    ensure!(
        negative?,
        "wrong temporal coefficient order was not detected"
    );
    ensure!(on.forward(&inputs[0])?.to_bytes()? == references[0].to_bytes()?);
    println!(
        "GDN_CONST_QUALITY weights=36 external_positions=5 cohorts=1..8 outputs_states_histories_prefixes=byte_exact reset=true invalid_atomic=true mutation=detected_restored"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU; only after prepared-conv quality; run alone"]
fn microbench_complete_gdn_prepared_conv_alternating() -> Result<()> {
    let destination = PathBuf::from(std::env::var("MLXL3_GDN_CONST_OUTPUT")?);
    ensure!(!destination.exists() && destination.parent().is_some_and(Path::is_dir));
    let (checkpoint, inputs, _) = fixture()?;
    let _retain = crate::flash_profile::StageRetention::new(false);
    let (mut off, mut on) = pair(&checkpoint)?;
    for x in &inputs[..2] {
        on.forward(x)?.eval()?;
    }
    let saved = (clone_array(&on.history)?, clone_array(&on.state)?);
    saved.0.as_ref().context("missing seeded history")?.eval()?;
    saved.1.as_ref().context("missing seeded state")?.eval()?;
    for x in &inputs {
        x.eval()?;
    }
    let mut samples = Vec::new();
    for rows in [1usize, 2, 4, 8] {
        let cohort = (0..rows)
            .map(|row| inputs[row % inputs.len()].try_clone())
            .collect::<Result<Vec<_>>>()?;
        for _ in 0..3 {
            for gdn in [&mut off, &mut on] {
                restore(gdn, &saved)?;
                for output in forward(gdn, &cohort)? {
                    output.eval()?;
                }
            }
        }
        let mut oracle = None;
        for round in 0..12 {
            let arms = if round % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for prepared in arms {
                let gdn = if prepared { &mut on } else { &mut off };
                restore(gdn, &saved)?;
                let started = Instant::now();
                let mut last = Vec::new();
                for iteration in 0..16 {
                    let values = forward(gdn, &cohort)?;
                    for value in &values {
                        value.eval()?;
                    }
                    if iteration == 15 {
                        last = values;
                    }
                }
                let seconds = started.elapsed().as_secs_f64();
                ensure!(seconds.is_finite() && seconds > 0.0 && gdn.stages.is_empty());
                let result = (outputs(&last)?, state(gdn)?);
                match &oracle {
                    Some(expected) => {
                        ensure!(&result == expected, "micro output/state differs B{rows}")
                    }
                    None => oracle = Some(result),
                }
                samples.push(serde_json::json!({"rows":rows,"round":round,"prepared":prepared,"forwards":16,"seconds":seconds}));
            }
        }
    }
    ensure!(samples.len() == 96);
    let report = serde_json::json!({"status":"complete","scope":"complete_GDN_component_not_model_gain","samples":samples,"warmups":24,"measured_forwards":1536,"test_stage_retention":false,"limits":"same retained seeded snapshots per arm; construction+submission+completion host, no exclusive GPU latency; parity outside chrono"});
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(destination)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    file.write_all(b"\n")?;
    println!(
        "GDN_CONST_MICRO status=complete samples=96 forwards=1536 warmups=24 rows=1,2,4,8 parity=byte_exact"
    );
    Ok(())
}

#[test]
#[ignore = "isolated invalid-setting process and Flash checkpoint metadata"]
fn prepared_conv_rejects_invalid_loader_setting() -> Result<()> {
    let setting = std::env::var("MLXL3_FLASH_PREPARED_CONV")?;
    ensure!(!matches!(setting.as_str(), "0" | "1"));
    let checkpoint = crate::checkpoint::inspect(Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    match Gdn::load(&checkpoint, PREFIX, 1e-6) {
        Ok(_) => anyhow::bail!("invalid prepared-conv setting accepted"),
        Err(error) => ensure!(error.to_string().contains("MLXL3_FLASH_PREPARED_CONV")),
    }
    println!("GDN_CONST_INVALID_SETTING rejected_before_weight_load=true");
    Ok(())
}
