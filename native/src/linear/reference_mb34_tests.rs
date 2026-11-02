//! Candidates C/D: real GDN weights, stock MB2 and independent per-row dispatches.
use super::*;

fn selected_roles() -> Result<crate::contracts::FlashReferenceMb34Roles> {
    let value = match std::env::var("MLXL3_FLASH_REFERENCE_MB34_ROLES") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    crate::contracts::flash_reference_mb34_roles(value.as_deref())
        .context("invalid MB34 test roles")
}

fn empty_checkpoint() -> Checkpoint {
    Checkpoint {
        path: Default::default(),
        model_type: "qwen4_exp".into(),
        bits: None,
        size_bytes: 0,
        modules: Vec::new(),
        tensors: Default::default(),
    }
}

fn compare(actual: &Array, expected: &Array, rows: i32, output: i32, dtype: Dtype) -> Result<()> {
    let count = (rows as usize)
        .checked_mul(output as usize)
        .context("reference MB34 output size overflow")?;
    ensure!(count > 0);
    for value in [actual, expected] {
        ensure!(value.shape() == [rows, output] && value.dtype() == dtype);
        let finite = value.to_f32()?;
        ensure!(finite.len() == count && finite.iter().all(|v| v.is_finite()));
    }
    let bytes = actual.to_bytes()?;
    let expected = expected.to_bytes()?;
    let itemsize = if dtype == Dtype::Float16 { 2 } else { 4 };
    ensure!(bytes.len() == count * itemsize && expected.len() == bytes.len());
    ensure!(
        bytes == expected,
        "reference MB34 differs bytewise from oracle"
    );
    Ok(())
}

#[test]
#[ignore = "select alone with MLXL3_FLASH_REFERENCE_MB34=invalid; no checkpoint or GPU needed"]
fn flash_reference_mb34_rejects_invalid_flag_before_loading_weights() -> Result<()> {
    ensure!(std::env::var("MLXL3_FLASH_REFERENCE_MB34").as_deref() == Ok("invalid"));
    let error = Exl3Linear::from_checkpoint(
        &empty_checkpoint(),
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
    )
    .err()
    .context("invalid reference MB34 flag accepted")?;
    ensure!(error.to_string() == "MLXL3_FLASH_REFERENCE_MB34 must be 0 or 1");
    Ok(())
}

#[test]
#[ignore = "select alone with MLXL3_FLASH_REFERENCE_MB34_ROLES=invalid; no checkpoint or GPU needed"]
fn flash_reference_mb34_rejects_invalid_roles_before_loading_weights() -> Result<()> {
    ensure!(std::env::var("MLXL3_FLASH_REFERENCE_MB34_ROLES").as_deref() == Ok("invalid"));
    let error = Exl3Linear::from_checkpoint(
        &empty_checkpoint(),
        "model.language_model.layers.0.linear_attn.in_proj_z",
    )
    .err()
    .context("invalid reference MB34 roles accepted")?;
    ensure!(error.to_string() == "MLXL3_FLASH_REFERENCE_MB34_ROLES must be qkv, z, out or all");
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU, Flash checkpoint, REFERENCE_MB34=1; select alone"]
fn flash_reference_mb34_matches_single_rows_and_mb2() -> Result<()> {
    ensure!(std::env::var("MLXL3_FLASH_REFERENCE_MB34").as_deref() == Ok("1"));
    let checkpoint =
        crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let selected = selected_roles()?;
    let mut cases = 0;
    let mut requests = [0; 2];
    let mut projections = 0;
    for (suffix, input, output) in [
        ("in_proj_qkv", 2560, 10240),
        ("in_proj_z", 2560, 6144),
        ("out_proj", 6144, 2560),
    ] {
        let name = format!("model.language_model.layers.0.linear_attn.{suffix}");
        if !crate::contracts::flash_reference_mb34_role_scope(&name, input, output, selected) {
            continue;
        }
        projections += 1;
        let mut linear = Exl3Linear::from_checkpoint(&checkpoint, &name)?;
        ensure!(linear.reference_mb34 && linear.reference_accumulation && linear.simdgroups == 4);
        ensure!(linear.rows == input && linear.cols == output && linear.k != 7);
        for rows in [1, 2, 3, 4, 5, 8] {
            for dtype in [Dtype::Float16, Dtype::Float32] {
                for strided in [false, true] {
                    let value = |r: i32, c: i32| ((r * 37 + c * 13) % 997) as f32 / 1024. - 0.5;
                    let input = if strided {
                        let values = (0..linear.rows)
                            .flat_map(|c| (0..rows).map(move |r| value(r, c)))
                            .collect::<Vec<_>>();
                        Array::from_f32(&values, &[linear.rows, rows])?
                            .astype(dtype)?
                            .transpose(&[1, 0])?
                    } else {
                        let values = (0..rows)
                            .flat_map(|r| (0..linear.rows).map(move |c| value(r, c)))
                            .collect::<Vec<_>>();
                        Array::from_f32(&values, &[rows, linear.rows])?.astype(dtype)?
                    };
                    let original = input.to_bytes()?;
                    linear.flash_batch = false;
                    linear.reference_mb34 = false;
                    let serial = (0..rows)
                        .map(|r| linear.forward(&input.slice(0, r, r + 1)?))
                        .collect::<Result<Vec<_>>>()?;
                    let serial = Array::concatenate(&serial.iter().collect::<Vec<_>>(), 0)?;
                    linear.flash_batch = true;
                    let paired = linear.forward(&input)?;
                    compare(&paired, &serial, rows, linear.cols, dtype)?;
                    linear.reference_mb34 = true;
                    REFERENCE_MB34_REQUESTS_TEST.with(|counts| counts.set([0; 2]));
                    let readbacks = Array::host_readbacks_for_test();
                    let candidate = linear.forward(&input)?;
                    ensure!(
                        Array::host_readbacks_for_test() == readbacks,
                        "MB34 added host readback"
                    );
                    let count = REFERENCE_MB34_REQUESTS_TEST.with(std::cell::Cell::get);
                    let expected_requests = match rows {
                        3 => [1, 0],
                        4 => [0, 1],
                        _ => [0, 0],
                    };
                    ensure!(
                        count == expected_requests,
                        "MB34 dispatch or fallback was not selected"
                    );
                    requests[0] += count[0];
                    requests[1] += count[1];
                    compare(&candidate, &serial, rows, linear.cols, dtype)?;
                    compare(&candidate, &paired, rows, linear.cols, dtype)?;
                    ensure!(
                        input.to_bytes()? == original,
                        "MB34 mutated its input alias"
                    );
                    if rows > 1 {
                        ensure!(
                            serial.slice(0, 0, 1)?.to_bytes()?
                                != serial.slice(0, 1, 2)?.to_bytes()?,
                            "vacuous row oracle"
                        );
                    }
                    cases += 1;
                }
            }
        }
        ensure!(
            linear
                .forward(&Array::zeros_dtype(&[0, linear.rows], Dtype::Float16)?)
                .is_err()
        );
        ensure!(
            linear
                .forward(&Array::zeros_dtype(&[3, linear.rows - 1], Dtype::Float16)?)
                .is_err()
        );
        // Metadata refusal is independent of the opt-in candidate and exercises forward.
        linear.mtp_hidden_batch = true;
        ensure!(
            linear
                .forward(&Array::zeros_dtype(&[3, linear.rows], Dtype::Float16)?)
                .is_err()
        );
    }
    let expected_projections = if selected == crate::contracts::FlashReferenceMb34Roles::All {
        3
    } else {
        1
    };
    ensure!(projections == expected_projections && cases == 24 * expected_projections);
    ensure!(requests == [4 * expected_projections; 2]);
    println!(
        "FLASH_REFERENCE_MB34 roles={selected:?} projections={projections} cases={cases} requests={requests:?} parity=byte_exact"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU, Flash checkpoint, REFERENCE_MB34=1; select alone"]
fn flash_reference_mb34_loader_excludes_head_and_other_projections() -> Result<()> {
    ensure!(std::env::var("MLXL3_FLASH_REFERENCE_MB34").as_deref() == Ok("1"));
    let checkpoint =
        crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let selected = selected_roles()?;
    let mut cases = 0;
    for name in [
        "model.language_model.layers.0.linear_attn.in_proj_qkv",
        "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.out_proj",
        "model.language_model.layers.3.self_attn.o_proj",
        "mtp.layers.0.self_attn.o_proj",
        "lm_head",
    ] {
        let mut linear = Exl3Linear::from_checkpoint(&checkpoint, name)?;
        let eligible = crate::contracts::flash_reference_mb34_role_scope(
            name,
            linear.rows,
            linear.cols,
            selected,
        );
        ensure!(linear.reference_mb34 == eligible && linear.reference_accumulation);
        if eligible {
            continue;
        }
        linear.flash_batch = true;
        let input = Array::from_f32(
            &(0..3 * linear.rows)
                .map(|i| ((i * 17) % 193) as f32 / 256. - 0.25)
                .collect::<Vec<_>>(),
            &[3, linear.rows],
        )?;
        REFERENCE_MB34_REQUESTS_TEST.with(|counts| counts.set([0; 2]));
        let actual = linear.forward(&input)?;
        ensure!(REFERENCE_MB34_REQUESTS_TEST.with(std::cell::Cell::get) == [0; 2]);
        linear.flash_batch = false;
        let expected = linear.forward(&input)?;
        compare(&actual, &expected, 3, linear.cols, Dtype::Float32)?;
        cases += 1;
    }
    let expected_cases = if selected == crate::contracts::FlashReferenceMb34Roles::All {
        3
    } else {
        5
    };
    ensure!(cases == expected_cases);
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn actual_flash_reference_mb34_forward_entry() {
    assert!(flash_reference_mb34_matches_single_rows_and_mb2().is_ok());
}
