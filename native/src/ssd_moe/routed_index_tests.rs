use super::*;

fn load(checkpoint: &Checkpoint) -> Result<SsdMoe> {
    let moe = SsdMoe::load(checkpoint, "model.language_model.layers.0.mlp", 53, true)?;
    ensure!(
        moe.routed_index_cache
            == (std::env::var("MLXL3_SSD_ROUTED_INDEX_CACHE").as_deref() == Ok("1")),
        "routed index cache option was not loaded"
    );
    ensure!(moe.routed_indices.iter().all(Option::is_none));
    Ok(moe)
}

fn input(rows: i32, seed: i32, dtype: Dtype, strided: bool) -> Result<(Array, Vec<u8>)> {
    let value =
        |row: i32, column: i32| ((row * 37 + column * 13 + seed) % 997) as f32 / 1024. - 0.5;
    let values = if strided {
        (0..H)
            .flat_map(|column| (0..rows).map(move |row| value(row, column)))
            .collect()
    } else {
        (0..rows)
            .flat_map(|row| (0..H).map(move |column| value(row, column)))
            .collect::<Vec<_>>()
    };
    let x = if strided {
        Array::from_f32(&values, &[H, rows])?
            .astype(dtype)?
            .transpose(&[1, 0])?
    } else {
        Array::from_f32(&values, &[rows, H])?.astype(dtype)?
    };
    let expected = (0..rows)
        .flat_map(|row| {
            (0..10).flat_map(move |_| {
                (0..H).flat_map(move |column| {
                    half::f16::from_f32(value(row, column))
                        .to_bits()
                        .to_le_bytes()
                })
            })
        })
        .collect();
    Ok((x, expected))
}

#[test]
#[ignore = "physical Apple GPU and Flash checkpoint; select alone"]
fn routed_indices_reuse_only_shape_and_keep_live_graph_inputs() -> Result<()> {
    let checkpoint =
        crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let mut moe = load(&checkpoint)?;
    let mut pending = Vec::new();
    let mut cases = 0;
    for rows in 1..=8 {
        for dtype in [Dtype::Float16, Dtype::Float32] {
            for strided in [false, true] {
                for seed in [0, 19] {
                    let (x, expected) = input(rows, seed, dtype, strided)?;
                    let readbacks = Array::host_readbacks_for_test();
                    moe.routed_index_cache = false;
                    let builds = moe.routed_index_builds;
                    let stock = moe.routed_input(&x)?;
                    ensure!(moe.routed_index_builds == builds + 1);
                    moe.routed_index_cache = true;
                    let populated = moe.routed_indices[rows as usize - 1].is_some();
                    let builds = moe.routed_index_builds;
                    let hits = moe.routed_index_hits;
                    let cached = moe.routed_input(&x)?;
                    ensure!(moe.routed_index_builds == builds + usize::from(!populated));
                    ensure!(moe.routed_index_hits == hits + usize::from(populated));
                    ensure!(
                        Array::host_readbacks_for_test() == readbacks,
                        "routing added a host readback"
                    );
                    ensure!(cached.shape() == [rows * 10, H] && cached.dtype() == Dtype::Float16);
                    pending.push((stock, cached, expected));
                    // Both graphs keep their own x alive while later calls reuse only the indices.
                    drop(x);
                    cases += 1;
                }
            }
        }
    }
    ensure!(cases == 64 && moe.routed_indices.iter().all(Option::is_some));
    let mut payload = 0;
    for (slot, indices) in moe.routed_indices.iter().enumerate() {
        let indices = indices.as_ref().context("missing cached shape")?;
        ensure!(indices.shape() == [(slot as i32 + 1) * 10] && indices.dtype() == Dtype::UInt32);
        let values = indices.to_u32()?;
        ensure!(values.len() == (slot + 1) * 10);
        ensure!(
            values
                .as_chunks::<10>()
                .0
                .iter()
                .enumerate()
                .all(|(row, chunk)| chunk.iter().all(|&value| value == row as u32))
        );
        payload += values.len() * std::mem::size_of::<u32>();
    }
    ensure!(payload == 1440);
    // A shape outside the cache domain uses stock allocation without growing the eight slots.
    let (x, expected) = input(9, 71, Dtype::Float32, false)?;
    let before = (moe.routed_index_builds, moe.routed_index_hits);
    let first = moe.routed_input(&x)?;
    let second = moe.routed_input(&x)?;
    ensure!((moe.routed_index_builds, moe.routed_index_hits) == (before.0 + 2, before.1));
    pending.push((first, second, expected));
    let mut other = load(&checkpoint)?;
    other.routed_index_cache = true;
    let (x, expected) = input(2, 93, Dtype::Float16, true)?;
    let output = other.routed_input(&x)?;
    ensure!(other.routed_index_builds == 1 && other.routed_index_hits == 0);
    ensure!(
        other
            .routed_indices
            .iter()
            .filter(|entry| entry.is_some())
            .count()
            == 1
    );
    pending.push((output.try_clone()?, output, expected));
    drop(other);
    drop(moe);
    for (stock, cached, expected) in pending {
        ensure!(
            !expected.is_empty() && stock.to_bytes()? == expected && cached.to_bytes()? == expected
        );
    }
    println!(
        "ROUTED_INDEX_COMPONENT cases=64 out_of_domain=1 isolated_instance=1 payload_bytes=1440 parity=byte_exact"
    );
    Ok(())
}

#[test]
#[ignore = "physical Apple GPU and external Flash fixtures; select alone"]
fn routed_indices_cache_preserves_real_hot_cold_and_invalid_inputs() -> Result<()> {
    let (checkpoint, input, expected) = super::tests::admission_fixture()?;
    let mut moe = load(&checkpoint)?;
    moe.gpu_hot_wide = true;
    moe.unique_cohorts = true;
    let mut checked = 0;
    for enabled in [false, true] {
        moe.routed_index_cache = enabled;
        for hot in [false, true] {
            moe.gpu_hot = hot;
            for rows in [1, 2, 3, 4, 8] {
                moe.drain()?;
                moe.cache.borrow_mut().invalidate();
                let x =
                    Array::concatenate(&std::iter::repeat_n(&input, rows).collect::<Vec<_>>(), 0)?;
                let baseline = expected.repeat(rows);
                let first = moe.forward_draft_rows(&x)?;
                ensure!(first.shape() == [rows as i32, H] && first.to_bytes()? == baseline);
                let second = moe.forward_draft_rows(&x)?;
                ensure!(second.to_bytes()? == baseline);
                checked += 2;
            }
        }
    }
    ensure!(checked == 40 && moe.routed_index_hits > 0);
    let before = (moe.routed_index_builds, moe.routed_index_hits);
    for (shape, dtype) in [
        (vec![0, H], Dtype::Float16),
        (vec![1, H - 1], Dtype::Float32),
        (vec![9, H], Dtype::Float16),
        (vec![1, H, 1], Dtype::Float16),
        (vec![1, H], Dtype::UInt32),
    ] {
        ensure!(
            moe.forward_draft_rows(&Array::zeros_dtype(&shape, dtype)?)
                .is_err()
        );
        ensure!(before == (moe.routed_index_builds, moe.routed_index_hits));
    }
    moe.drain()?;
    println!("ROUTED_INDEX_MOE outputs=40 invalid=5 parity=external_byte_exact");
    Ok(())
}

#[test]
#[ignore = "set ROUTED_INDEX_CACHE=invalid and select alone; no checkpoint or GPU"]
fn routed_index_cache_invalid_option_fails_before_weights() -> Result<()> {
    ensure!(std::env::var("MLXL3_SSD_ROUTED_INDEX_CACHE").as_deref() == Ok("invalid"));
    let checkpoint = Checkpoint {
        path: Default::default(),
        model_type: String::new(),
        bits: None,
        size_bytes: 0,
        modules: Vec::new(),
        tensors: Default::default(),
    };
    let error = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 53, true)
        .err()
        .context("invalid index cache option accepted")?;
    ensure!(error.to_string() == "MLXL3_SSD_ROUTED_INDEX_CACHE must be 0 or 1");
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn actual_routed_index_cache_refinement_entry() {
    let component: bool = kani::any();
    assert!(
        if component {
            routed_indices_reuse_only_shape_and_keep_live_graph_inputs()
        } else {
            routed_indices_cache_preserves_real_hot_cold_and_invalid_inputs()
        }
        .is_ok()
    );
    kani::cover!(component);
    kani::cover!(!component);
}
