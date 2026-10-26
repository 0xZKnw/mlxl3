use super::*;

fn prepared_cache(offset: i32) -> Result<QsaState> {
    let make = |shape: &[i32], salt: i32| -> Result<Array> {
        let count = shape.iter().product::<i32>();
        let values = (0..count)
            .map(|i| ((i * 17 + salt * 13) % 997) as f32 / 8192. - 0.05)
            .collect::<Vec<_>>();
        Array::from_f32(&values, shape)?.astype(Dtype::Float16)
    };
    Ok(QsaState {
        keys: if offset > 0 {
            Some(make(&[1, 2, offset, 256], 1)?)
        } else {
            None
        },
        values: if offset > 0 {
            Some(make(&[1, 2, offset, 256], 2)?)
        } else {
            None
        },
        blocks: if offset >= 4 {
            Some(make(&[1, 1, offset / 4, 128], 3)?)
        } else {
            None
        },
        index_tail: if offset % 4 > 0 {
            Some(make(&[1, offset % 4, 128], 4)?)
        } else {
            None
        },
        offset,
    })
}

#[test]
#[ignore = "physical Apple GPU; QSA B2 cache, pooling, sparse boundaries and prefix continuation"]
fn qsa_two_rows_preserve_outputs_states_and_prefixes() -> Result<()> {
    let checkpoint =
        crate::checkpoint::inspect(std::path::Path::new(&std::env::var("MLXL3_FLASH_MODEL")?))?;
    let mut cases = 0;
    let mut layers = 0;
    for layer in 0..48 {
        let prefix = format!("model.language_model.layers.{layer}.self_attn");
        if !checkpoint
            .tensors
            .contains_key(&format!("{prefix}.q_proj.trellis"))
        {
            continue;
        }
        let mut q = Qsa::load(&checkpoint, &prefix, 1e-6, 10_000_000.)?;
        let inputs = (0..2)
            .map(|r| {
                Array::from_f32(
                    &(0..H)
                        .map(|c| ((r * 37 + c * 13) % 997) as f32 / 1024. - 0.5)
                        .collect::<Vec<_>>(),
                    &[1, H],
                )
            })
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            q.forward_rows(&[]).is_err() && q.forward_rows(&inputs[..1]).is_err() && q.offset == 0
        );
        let invalid = [Array::zeros(&[2, H])?, inputs[1].try_clone()?];
        ensure!(q.forward_rows(&invalid).is_err() && q.offset == 0);
        for offset in [0, 3, 7, 2050, 2051, 2052, 4094] {
            q.restore(prepared_cache(offset)?);
            let initial = q.snapshot()?;
            let first = q.forward(&inputs[0])?.to_bytes()?;
            let prefix_state = tests::qsa_bytes(&q)?;
            let second = q.forward(&inputs[1])?.to_bytes()?;
            let final_state = tests::qsa_bytes(&q)?;
            q.restore(initial);
            let calls = q.batch_calls;
            let (actual, mut snapshots) = q.forward_rows(&inputs)?;
            ensure!(actual.len() == 2 && snapshots.len() == 1 && q.batch_calls == calls + 1);
            for (out, expected) in actual.iter().zip([&first, &second]) {
                let values = out.to_f32()?;
                ensure!(
                    out.shape() == [1, H]
                        && values.len() == H as usize
                        && values.iter().all(|v| v.is_finite())
                );
                ensure!(
                    &out.to_bytes()? == expected,
                    "QSA batch output differs layer={layer} offset={offset}"
                );
            }
            ensure!(
                tests::qsa_bytes(&q)? == final_state,
                "QSA final state differs layer={layer} offset={offset}"
            );
            q.restore(snapshots.pop().context("missing prefix")?);
            ensure!(tests::qsa_bytes(&q)? == prefix_state);
            ensure!(
                q.forward(&inputs[1])?.to_bytes()? == second
                    && tests::qsa_bytes(&q)? == final_state
            );
            cases += 1;
        }
        layers += 1;
    }
    ensure!(layers == 12 && cases == 84);
    println!(
        "QSA_BATCH_STATES layers={layers} cases={cases} parity=byte_exact prefix_continuation=true"
    );
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn actual_qsa_batch_entry() {
    qsa_two_rows_preserve_outputs_states_and_prefixes().unwrap();
}
