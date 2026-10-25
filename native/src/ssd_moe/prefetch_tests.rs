use super::*;

#[test]
#[ignore = "physical Apple GPU; DFlash B1..8, Hot with pending banks and next-layer reads"]
fn draft_prefetch_preserves_hot_splits_and_next_layer_outputs() -> Result<()> {
    let (checkpoint, input, expected_first) = super::tests::admission_fixture()?;
    let base = input.to_f32()?;
    ensure!(base.len() == H as usize && base.iter().all(|v| v.is_finite()));
    let mut reference = SsdMoe::load(&checkpoint, "model.language_model.layers.0.mlp", 80, false)?;
    let mut next_reference =
        SsdMoe::load(&checkpoint, "model.language_model.layers.1.mlp", 80, false)?;
    let mut cases = 0;
    let mut split_cases = 0;
    for capacity in [24, 53] {
        let pool = Rc::new(RefCell::new(ReaderPool::new()?));
        let mut current = SsdMoe::load(
            &checkpoint,
            "model.language_model.layers.0.mlp",
            capacity,
            true,
        )?;
        let mut next = SsdMoe::load(
            &checkpoint,
            "model.language_model.layers.1.mlp",
            capacity,
            true,
        )?;
        for (index, moe) in [&mut current, &mut next].into_iter().enumerate() {
            for bank in &mut moe.banks {
                // An accidental read of an unfilled scale becomes NaN, not a plausible zero.
                *bank =
                    Array::from_bytes(&vec![0xff; bank.byte_len()?], bank.shape(), bank.dtype())?;
            }
            moe.prefetch_pipeline = true;
            moe.gpu_hot = true;
            moe.gpu_hot_wide = true;
            moe.unique_cohorts = true;
            moe.prefetcher = Some((index, Rc::clone(&pool)));
        }
        for rows in 1..=8 {
            pool.borrow_mut().wait_for_test()?;
            current.drain()?;
            next.drain()?;
            current.cache.borrow_mut().invalidate();
            next.cache.borrow_mut().invalidate();
            current.cohort_sizes.clear();
            let values = (0..rows)
                .flat_map(|r| {
                    base.iter().enumerate().map(move |(c, &v)| {
                        if r == 0 {
                            v
                        } else {
                            ((r * 73 + c * 29) % 1021) as f32 / 256. - 2.
                        }
                    })
                })
                .collect::<Vec<_>>();
            let x = Array::from_f32(&values, &[rows as i32, H])?;
            let expected = reference.forward_draft_rows(&x)?;
            ensure!(expected.slice(0, 0, 1)?.to_bytes()? == expected_first);
            let expected_next = next_reference.forward_draft_rows(&x)?;
            let ids = current.route(&input)?.0.to_u32()?;
            let routes = current.cache.borrow_mut().reserve(&ids[..5])?;
            current.fill_many(
                &ids[..5]
                    .iter()
                    .zip(routes)
                    .map(|(&id, (slot, _))| (id, slot))
                    .collect::<Vec<_>>(),
            )?;
            let submitted = pool
                .borrow()
                .counters
                .submitted
                .load(std::sync::atomic::Ordering::Relaxed);
            let predicted = next.prefetch_profile[0];
            let hot = current.hybrid_submissions;
            let actual = current.forward_draft_lookahead(&x, Some(&mut next), 2)?;
            ensure!(actual.shape() == [rows as i32, H]);
            let actual_values = actual.to_f32()?;
            ensure!(
                actual_values.len() == rows * H as usize
                    && actual_values.iter().all(|v| v.is_finite())
            );
            ensure!(
                actual.to_bytes()? == expected.to_bytes()?,
                "draft prefetch changed routed values"
            );
            ensure!(
                current.hybrid_submissions > hot,
                "prefetch disabled early Hot"
            );
            ensure!(next.prefetch_profile[0] - predicted == (rows.min(2) * 16) as u64);
            ensure!(
                pool.borrow()
                    .counters
                    .submitted
                    .load(std::sync::atomic::Ordering::Relaxed)
                    - submitted
                    == 2
            );
            split_cases += usize::from(current.cohort_sizes.len() > 1);
            pool.borrow_mut().wait_for_test()?;
            let used = pool
                .borrow()
                .counters
                .used
                .load(std::sync::atomic::Ordering::Relaxed);
            let actual_next = next.forward_draft_rows(&x)?;
            ensure!(
                actual_next.shape() == [rows as i32, H]
                    && actual_next.to_bytes()? == expected_next.to_bytes()?
            );
            ensure!(
                pool.borrow()
                    .counters
                    .used
                    .load(std::sync::atomic::Ordering::Relaxed)
                    > used
            );
            cases += 1;
        }
        current.drain()?;
        next.drain()?;
        pool.borrow_mut().wait_for_test()?;
    }
    ensure!(cases == 16 && split_cases > 0);
    println!(
        "SSD_DRAFT_PIPELINE cases={cases} split_cases={split_cases} hot_preserved=true predictions=verified next_outputs=byte_exact"
    );
    Ok(())
}

#[cfg(kani)]
#[kani::proof]
fn actual_uninitialized_slot_reads_preserve_outputs() {
    draft_prefetch_preserves_hot_splits_and_next_layer_outputs().unwrap();
}

#[cfg(kani)]
#[kani::proof]
fn actual_draft_prefetch_pipeline_preserves_outputs() {
    draft_prefetch_preserves_hot_splits_and_next_layer_outputs().unwrap();
}
